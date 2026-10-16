# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Temporal image-source reuse, motion coherence, and exact refresh guards."""

from types import SimpleNamespace

from sensing_test_helpers import *  # noqa: F401,F403
import mmWaveRadar.simulation.env_human_coupling as coupling
from mmWaveRadar.simulation.coupling_incremental import CouplingIncrementalState


@pytest.fixture
def image_case():
    vertices = np.concatenate([
        _front_facing_triangle(2., scale=.015, y_shift=y, z_shift=-.04)
        for y in (-.08, .08)
    ]).astype(float)
    faces = np.arange(6, dtype=np.uint32).reshape(-1, 3)
    floors = np.array([
        [[-5., -5., z], [5., -5., z], [0., 5., z]] for z in (-1., 1.)
    ])
    pec = PhysicalOpticsMaterial(1., 0., model="pec")
    reflectors = EnvironmentReflectorBank(
        vertices=floors, centroids=floors.mean(1), normals=np.array([[0., 0., 1.]] * 2),
        areas=np.array([50., 50.]), materials=(pec, pec), object_names=("floor", "ceiling"),
        face_indices=np.array([0, 0]),
    )
    radar = _phase_one_radar(num_chirps=4)
    hardware = RadarHardware.from_positions(
        [[0., 0., 0.], [0., .0025, 0.]], [[0., 0., 0.], [0., 0., .0025]])
    radar = replace(radar, hardware=replace(hardware, scalar_antenna_gain_dbi=6.,
        virtual_channel_phase_signs=np.array([1., -1., -1., 1.])),
        fmcw=replace(radar.fmcw, slope=1e12, sampling_frequency=1e6,
                     chirp_duration=40e-6, num_adc_samples=32, num_tx=2))
    visibility = HumanVisibilityScene(faces, vertices, name="incremental-image-human")
    config = HumanPOMobilityConfig(
        incremental_update=True, incremental_full_refresh_chirps=0,
        incremental_centroid_displacement_threshold_m=.002,
        incremental_area_relative_threshold=.05,
        po_integration_mode="parent_face_adaptive_analytic",
        po_quadrature_max_refinement_depth=3, po_quadrature_max_subfaces_per_parent=64,
    )

    def solve(points, *, config=config, previous=None, state=None, **kwargs):
        visibility.update_vertices(points)
        current_radar = kwargs.pop("radar", radar)
        return coupling.compute_env_human_coupling_channels(
            vertices=points, faces=faces, tx_positions=current_radar.world_tx_positions(),
            rx_positions=current_radar.world_rx_positions(), frequency_hz=current_radar.fmcw.carrier_frequency,
            human_material=pec, reflectors=reflectors, human_visibility_scene=visibility,
            radar=current_radar, po_config=config, incremental_state=state,
            human_env_active_sets=None if previous is None else previous.human_env_active_sets,
            env_human_active_sets=None if previous is None else previous.env_human_active_sets,
            **kwargs,
        )

    def adc(result, order):
        return MmWaveRadarSimulator(compute_backend="numpy").synthesize_adc(
            getattr(result, order + "_coefficients"), getattr(result, order + "_delays_s"),
            getattr(result, order + "_valid"), radar.fmcw,
            delay_response=getattr(result, order + "_delay_response"))

    return SimpleNamespace(vertices=vertices, radar=radar, config=config, solve=solve, adc=adc)


@pytest.mark.parametrize("mode", ["face_centroid", "parent_face_quadrature",
                                 "parent_face_adaptive_analytic", "parent_face_far_field_analytic"])
def test_static_image_terms_reuse_integrals_and_preserve_adc(image_case, monkeypatch, mode):
    case = image_case
    config = replace(case.config, po_integration_mode=mode)
    calls = []
    original = coupling._integrate_coupling_faces

    def record(**kwargs):
        calls.append(kwargs["face_indices"].size)
        return original(**kwargs)

    monkeypatch.setattr(coupling, "_integrate_coupling_faces", record)
    state = CouplingIncrementalState()
    first = case.solve(case.vertices, config=config, state=state)
    assert sum(calls) == 8  # Two facets, two reflectors, two path directions.
    calls.clear()
    # Rediscovering visibility alone must not reintegrate unchanged facets.
    for previous in (first, None):
        cached = case.solve(case.vertices, config=config, previous=previous, state=state)
        assert not calls
        assert state.recomputed_faces == 0 and state.phase_updated_faces == 8
        for order in ("human_env", "env_human"):
            np.testing.assert_allclose(case.adc(cached, order), case.adc(first, order),
                                       rtol=1e-12, atol=1e-20)


def test_image_motion_updates_phase_and_wideband_delay_without_integration(image_case):
    case = image_case
    state = CouplingIncrementalState()
    initial = previous = case.solve(case.vertices, state=state)
    for step in range(1, 5):
        points = case.vertices + [step * 1e-5, 0., 0.]
        cached = case.solve(points, previous=previous, state=state)
        exact = case.solve(points, previous=previous)
        assert state.recomputed_faces == 0 and state.phase_updated_faces == 8
        for order in ("human_env", "env_human"):
            actual, reference = case.adc(cached, order), case.adc(exact, order)
            assert np.linalg.norm(actual - reference) / np.linalg.norm(reference) < 1e-3
            np.testing.assert_array_equal(getattr(cached, order + "_delays_s"),
                                          getattr(exact, order + "_delays_s"))
            assert np.linalg.norm(actual - case.adc(initial, order)) / np.linalg.norm(actual) > .01
        previous = cached


@pytest.mark.parametrize("change", ["centroid", "normal", "area"])
def test_image_refresh_uses_accumulated_drift_since_last_exact_solve(image_case, change):
    case = image_case
    state = CouplingIncrementalState()
    previous = case.solve(case.vertices, state=state)
    for step, expected in ((.6, 0), (1.2, 8)):
        points = case.vertices.copy()
        if change == "centroid":
            points += [step * .002, 0., 0.]
        else:
            triangles = points.reshape(-1, 3, 3)
            centers = triangles.mean(1, keepdims=True)
            if change == "normal":
                angle = np.deg2rad(step * 5.)
                rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                                     [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
                triangles[:] = (triangles - centers) @ rotation.T + centers
            else:
                triangles[:] = (triangles - centers) * np.sqrt(1. + step * .05) + centers
        cached = case.solve(points, previous=previous, state=state)
        assert state.recomputed_faces == expected
        previous = cached
    exact = case.solve(points)
    for order in ("human_env", "env_human"):
        np.testing.assert_allclose(case.adc(cached, order), case.adc(exact, order),
                                   rtol=1e-12, atol=1e-20)


def test_image_full_refresh_schedule_and_new_facets(image_case):
    case = image_case
    state = CouplingIncrementalState()
    config = replace(case.config, incremental_full_refresh_chirps=2)
    previous = None
    for expected in (8, 0, 8, 0):
        previous = case.solve(case.vertices, config=config, previous=previous, state=state)
        assert state.recomputed_faces == expected
    # Dropped facets must be integrated again when visibility discovers them.
    previous = replace(previous, human_env_active_sets=(), env_human_active_sets=())
    case.solve(case.vertices, config=config, previous=previous, state=state)
    assert not state.facets
    case.solve(case.vertices, config=config, state=state)
    assert state.recomputed_faces == 8


def test_image_if_edge_preserves_exact_clipping_during_motion(image_case):
    case = image_case
    center = case.vertices.reshape(-1, 3, 3).mean(1)[0]
    cutoff = (np.linalg.norm(center) + np.linalg.norm(center - [0., 0., -2.])) / c
    radar = replace(case.radar, fmcw=replace(case.radar.fmcw,
                    if_bandwidth_hz=abs(case.radar.fmcw.slope) * cutoff))
    state = CouplingIncrementalState()
    previous = case.solve(case.vertices, state=state)
    # A changed receiver invalidates the cache, then the IF-edge guard keeps
    # exact leaf geometry even when motion is below the normal/area thresholds.
    for shift in (1e-5, 2e-5):
        points = case.vertices + [shift, 0., 0.]
        previous = case.solve(points, previous=previous, state=state, radar=radar)
        exact = case.solve(points, radar=radar)
        assert state.recomputed_faces == 8
        for order in ("human_env", "env_human"):
            def synthesize(result):
                return getattr(result, order + "_delay_response").synthesize_adc(
                    getattr(result, order + "_delays_s"),
                    getattr(result, order + "_valid"), radar.fmcw)
            np.testing.assert_allclose(synthesize(previous), synthesize(exact),
                                       rtol=1e-12, atol=1e-20)
