# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Temporal image-source reuse, motion coherence, and exact refresh guards."""

from types import SimpleNamespace

from sensing_test_helpers import *  # noqa: F401,F403
import mmWaveRadar.simulation.env_human_coupling as coupling
from mmWaveRadar.simulation.coupling_incremental import CouplingIncrementalState


@pytest.fixture
def image_case():
    vertices = np.concatenate(
        [_front_facing_triangle(2.0, scale=0.015, y_shift=y, z_shift=-0.04) for y in (-0.08, 0.08)]
    ).astype(float)
    faces = np.arange(6, dtype=np.uint32).reshape(-1, 3)
    floors = np.array([[[-5.0, -5.0, z], [5.0, -5.0, z], [0.0, 5.0, z]] for z in (-1.0, 1.0)])
    pec = PhysicalOpticsMaterial(1.0, 0.0, model="pec")
    reflectors = EnvironmentReflectorBank(
        vertices=floors,
        centroids=floors.mean(1),
        normals=np.array([[0.0, 0.0, 1.0]] * 2),
        areas=np.array([50.0, 50.0]),
        materials=(pec, pec),
        object_names=("floor", "ceiling"),
        face_indices=np.array([0, 0]),
    )
    radar = _phase_one_radar(num_chirps=4)
    hardware = RadarHardware.from_positions(
        [[0.0, 0.0, 0.0], [0.0, 0.0025, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0025]]
    )
    radar = replace(
        radar,
        hardware=replace(
            hardware,
            scalar_antenna_gain_dbi=6.0,
            virtual_channel_phase_signs=np.array([1.0, -1.0, -1.0, 1.0]),
        ),
        fmcw=replace(
            radar.fmcw,
            slope=1e12,
            sampling_frequency=1e6,
            chirp_duration=40e-6,
            num_adc_samples=32,
            num_tx=2,
        ),
    )
    visibility = HumanVisibilityScene(faces, vertices, name="incremental-image-human")
    # Ten frames of four chirps: no periodic refresh within these short runs.
    config = POConfig(
        po_refresh_frames=10,
        po_integration_mode="parent_face_adaptive_analytic",
        po_quadrature_max_refinement_depth=3,
        po_quadrature_max_subfaces_per_parent=64,
    )

    def solve(points, *, config=config, previous=None, state=None, **kwargs):
        visibility.update_vertices(points)
        current_radar = kwargs.pop("radar", radar)
        return coupling.compute_env_human_coupling_channels(
            vertices=points,
            faces=faces,
            tx_positions=current_radar.world_tx_positions(),
            rx_positions=current_radar.world_rx_positions(),
            frequency_hz=current_radar.fmcw.carrier_frequency,
            human_material=pec,
            reflectors=reflectors,
            human_visibility_scene=visibility,
            radar=current_radar,
            po_config=config,
            incremental_state=state,
            human_env_active_sets=None if previous is None else previous.human_env_active_sets,
            env_human_active_sets=None if previous is None else previous.env_human_active_sets,
            **kwargs,
        )

    def adc(result, order):
        return MmWaveRadarSimulator(compute_backend="numpy").synthesize_adc(
            getattr(result, order + "_coefficients"),
            getattr(result, order + "_delays_s"),
            getattr(result, order + "_valid"),
            radar.fmcw,
            delay_response=getattr(result, order + "_delay_response"),
        )

    return SimpleNamespace(vertices=vertices, radar=radar, config=config, solve=solve, adc=adc)


def test_coupling_wideband_grid_uses_integration_precision(image_case):
    case = image_case
    results = [
        case.solve(case.vertices, precision=precision, max_frequency_offset_hz=4e9)
        for precision in ("float32", "float64")
    ]
    for order in ("human_env", "env_human"):
        assert len(getattr(results[0], order + "_delay_response").frequencies_hz) < len(
            getattr(results[1], order + "_delay_response").frequencies_hz
        )


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_exact_torch_coupling_matches_numpy_and_reuses_integrated_fields(
    image_case, monkeypatch, precision, device
):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if device == "mps" and (not torch.backends.mps.is_available() or precision == "float64"):
        pytest.skip("MPS float64 or device unavailable")
    monkeypatch.setenv("MMWAVE_TORCH_DEVICE", device)
    case = image_case
    reference = case.solve(case.vertices, precision=precision)
    state = CouplingIncrementalState()
    actual = case.solve(case.vertices, backend="torch", precision=precision, state=state)
    tolerance = 2e-3 if precision == "float32" else 1e-10
    for order in ("human_env", "env_human"):
        expected_adc, actual_adc = case.adc(reference, order), case.adc(actual, order)
        assert np.linalg.norm(actual_adc - expected_adc) / np.linalg.norm(expected_adc) < tolerance
    cached = case.solve(
        case.vertices, backend="torch", precision=precision, state=state, previous=actual
    )
    assert state.recomputed_faces == 0 and state.phase_updated_faces == 8
    for order in ("human_env", "env_human"):
        np.testing.assert_allclose(
            case.adc(cached, order), case.adc(actual, order), rtol=2e-12, atol=1e-20
        )


@pytest.mark.parametrize(
    "mode",
    [
        "face_centroid",
        "parent_face_quadrature",
        "parent_face_adaptive_analytic",
        "parent_face_far_field_analytic",
    ],
)
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
            np.testing.assert_allclose(
                case.adc(cached, order), case.adc(first, order), rtol=1e-12, atol=1e-20
            )


def test_image_motion_updates_phase_and_wideband_delay_without_integration(image_case):
    case = image_case
    state = CouplingIncrementalState()
    initial = previous = case.solve(case.vertices, state=state)
    for step in range(1, 5):
        points = case.vertices + [step * 1e-5, 0.0, 0.0]
        cached = case.solve(points, previous=previous, state=state)
        exact = case.solve(points, previous=previous)
        assert state.recomputed_faces == 0 and state.phase_updated_faces == 8
        for order in ("human_env", "env_human"):
            actual, reference = case.adc(cached, order), case.adc(exact, order)
            assert np.linalg.norm(actual - reference) / np.linalg.norm(reference) < 1e-3
            np.testing.assert_array_equal(
                getattr(cached, order + "_delays_s"), getattr(exact, order + "_delays_s")
            )
            assert (
                np.linalg.norm(actual - case.adc(initial, order)) / np.linalg.norm(actual) > 0.01
            )
        previous = cached


@pytest.mark.parametrize("change", ["centroid", "normal", "area"])
def test_image_phase_error_refresh_matches_exact_for_motion(image_case, change):
    case = image_case
    state = CouplingIncrementalState()
    config = replace(case.config, po_refresh_on_phase_error_rad=1e-9)
    previous = case.solve(case.vertices, config=config, state=state)
    for step, expected in ((0.6, 8), (1.2, 8)):
        points = case.vertices.copy()
        if change == "centroid":
            points += [step * 0.002, 0.0, 0.0]
        else:
            triangles = points.reshape(-1, 3, 3)
            centers = triangles.mean(1, keepdims=True)
            if change == "normal":
                angle = np.deg2rad(step * 5.0)
                rotation = np.array(
                    [
                        [np.cos(angle), -np.sin(angle), 0.0],
                        [np.sin(angle), np.cos(angle), 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
                triangles[:] = (triangles - centers) @ rotation.T + centers
            else:
                triangles[:] = (triangles - centers) * np.sqrt(1.0 + step * 0.05) + centers
        cached = case.solve(points, config=config, previous=previous, state=state)
        assert state.recomputed_faces == expected
        previous = cached
    exact = case.solve(points)
    for order in ("human_env", "env_human"):
        np.testing.assert_allclose(
            case.adc(cached, order), case.adc(exact, order), rtol=1e-12, atol=1e-20
        )


def test_image_full_refresh_schedule_and_new_facets(image_case):
    case = image_case
    state = CouplingIncrementalState()
    # One frame is four chirps, so the exact refresh recurs every fourth solve.
    config = replace(case.config, po_refresh_frames=1)
    previous = None
    for expected in (8, 0, 0, 0, 8, 0):
        previous = case.solve(case.vertices, config=config, previous=previous, state=state)
        assert state.recomputed_faces == expected
    # Dropped facets must be integrated again when visibility discovers them.
    previous = replace(previous, human_env_active_sets=(), env_human_active_sets=())
    case.solve(case.vertices, config=config, previous=previous, state=state)
    assert not state.facets
    case.solve(case.vertices, config=config, state=state)
    assert state.recomputed_faces == 8


@pytest.mark.parametrize("change", ["area", "roll"])
def test_image_amplitude_guards_refresh_even_with_a_large_phase_tolerance(image_case, change):
    case = image_case
    # A 0.2 rad roll of this slightly tilted face rotates its normal by only
    # about 2.3 degrees, so request a strict limit explicitly for this test.
    config = replace(case.config, po_refresh_on_phase_error_rad=1e6, po_refresh_on_normal_deg=1.0)
    initial = case.vertices.reshape(-1, 3, 3).copy()
    centres = initial.mean(1, keepdims=True)
    # Tilt the facet so a rotation about the line of sight changes its normal.
    tilt = 0.2
    rotation = np.array(
        [[np.cos(tilt), -np.sin(tilt), 0.0], [np.sin(tilt), np.cos(tilt), 0.0], [0.0, 0.0, 1.0]]
    )
    initial = (initial - centres) @ rotation.T + centres
    state = CouplingIncrementalState()
    previous = case.solve(initial.reshape(-1, 3), config=config, state=state)
    if change == "area":
        moved = (initial - centres) * 1.3 + centres
    else:
        angle = 0.2
        roll = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(angle), -np.sin(angle)],
                [0.0, np.sin(angle), np.cos(angle)],
            ]
        )
        moved = (initial - centres) @ roll.T + centres
    cached = case.solve(moved.reshape(-1, 3), config=config, previous=previous, state=state)
    exact = case.solve(moved.reshape(-1, 3), config=config)
    assert state.recomputed_faces == 8
    for order in ("human_env", "env_human"):
        np.testing.assert_allclose(
            case.adc(cached, order), case.adc(exact, order), rtol=1e-12, atol=1e-20
        )


def test_image_if_edge_preserves_exact_clipping_during_motion(image_case):
    case = image_case
    center = case.vertices.reshape(-1, 3, 3).mean(1)[0]
    cutoff = (np.linalg.norm(center) + np.linalg.norm(center - [0.0, 0.0, -2.0])) / c
    radar = replace(
        case.radar,
        fmcw=replace(case.radar.fmcw, if_bandwidth_hz=abs(case.radar.fmcw.slope) * cutoff),
    )
    state = CouplingIncrementalState()
    previous = case.solve(case.vertices, state=state)
    # A changed receiver invalidates the cache, then the IF-edge guard keeps
    # exact leaf geometry even when motion is below the phase-error threshold.
    for shift in (1e-5, 2e-5):
        points = case.vertices + [shift, 0.0, 0.0]
        previous = case.solve(points, previous=previous, state=state, radar=radar)
        exact = case.solve(points, radar=radar)
        assert state.recomputed_faces == 8
        for order in ("human_env", "env_human"):

            def synthesize(result):
                return getattr(result, order + "_delay_response").synthesize_adc(
                    getattr(result, order + "_delays_s"),
                    getattr(result, order + "_valid"),
                    radar.fmcw,
                )

            np.testing.assert_allclose(
                synthesize(previous), synthesize(exact), rtol=1e-12, atol=1e-20
            )
