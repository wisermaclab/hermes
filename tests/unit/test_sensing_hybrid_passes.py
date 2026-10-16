# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Hybrid entry-point parity, visibility scheduling, gains, and cancellation."""

from dataclasses import fields
from types import SimpleNamespace

from sensing_test_helpers import *  # noqa: F401,F403
import mmWaveRadar.simulation.hybrid_passes as passes
import mmWaveRadar.simulation.hybrid_pipeline as pipeline
from mmWaveRadar.simulation.reflector_selection import select_image_source_reflectors


@pytest.fixture
def hybrid_case(monkeypatch):
    radar = _phase_one_radar(num_chirps=4, chirp_repetition_time=.1)
    radar = replace(radar, fmcw=replace(radar.fmcw, frame_period=.4))
    first = _front_facing_triangle(2., scale=.08, y_shift=-.12, z_shift=-.04)
    last = first + [0., .24, 0.]
    target = MeshTarget(
        "shared-pass-human",
        MeshSequence(np.stack([first, last]), np.array([[0, 1, 2]], np.uint32),
                     np.array([0., .7])),
        material=human_skin_material("shared-pass-skin"),
    )
    floor = np.array([[-5., -5., -1.], [5., -5., -1.], [0., 5., -1.]])
    reflectors = EnvironmentReflectorBank(
        vertices=floor[None], centroids=floor.mean(0, keepdims=True),
        normals=np.array([[0., 0., 1.]]), areas=np.array([50.]),
        materials=(PhysicalOpticsMaterial(1., 0., model="pec"),),
        object_names=("floor",), face_indices=np.array([0]),
    )
    bank = StaticPathBank(
        coefficients=np.array([[.01 + .02j]]), delays_s=np.array([8./c]),
        valid=np.array([True]), segment_starts=np.array([[0., 0., 0.]]),
        segment_ends=np.array([[4., 0., 0.]]), segment_path_indices=np.array([0]),
    )
    scene = load_scene()
    trace_calls = []

    def trace(simulator, current_scene):
        simulator.check_cancelled()
        assert target.name not in current_scene.objects
        trace_calls.append(current_scene)
        return object()

    monkeypatch.setattr(MmWaveRadarSimulator, "trace_scene", trace)
    monkeypatch.setattr(passes, "extract_static_path_bank", lambda _paths, radar, **kw:
                        replace(bank, coefficients=np.repeat(
                            bank.coefficients, radar.hardware.num_virtual_channels, axis=0)))
    monkeypatch.setattr(passes, "select_image_source_reflectors", lambda *a, **kw: reflectors)

    def config(mode="parent_face_adaptive_analytic", incremental=True, refresh=3):
        return HybridStaticEnvPOConfig(
            blocking_fade_chirps=1, coupling_enabled=True,
            human_po_config=HumanPOMobilityConfig(
                visibility_samples_per_face=1, visibility_fade_chirps=1,
                po_integration_mode=mode, po_quadrature_max_refinement_depth=2,
                po_quadrature_max_subfaces_per_parent=16,
                incremental_update=incremental, incremental_full_refresh_chirps=0,
                incremental_visibility_refresh_chirps=refresh,
                incremental_centroid_displacement_threshold_m=0.,
            ),
        )

    def simulator(config, **kwargs):
        return MmWaveRadarSimulator(
            mobility_mode="hybrid_static_env_po", hybrid_static_env_po_config=config,
            compute_backend="numpy", compute_precision="float64", **kwargs,
        )

    def run(config, **kwargs):
        return pipeline.run_calibrated_hybrid_po(
            scene=scene, radar=radar, target=target, num_frames=2,
            hybrid_config=config, compute_backend="numpy", compute_precision="float64",
            progress=False, print_reflector_summary=False, **kwargs,
        )

    return SimpleNamespace(radar=radar, target=target, scene=scene, config=config,
                           simulator=simulator, run=run, trace_calls=trace_calls)


def _assert_same_components(left, right):
    np.testing.assert_array_equal(left.times, right.times)
    for name in ("human_po", "static_environment_unblocked", "static_environment_blocked",
                 "static_path_visibility_weights", "human_env", "env_human"):
        np.testing.assert_allclose(left.components[name], right.components[name],
                                   rtol=1e-10, atol=1e-18, err_msg=name)
    np.testing.assert_allclose(left.adc, right.adc, rtol=1e-10, atol=1e-18)
    for name in ("static_path_counts", "blocked_static_path_counts",
                 "human_env_path_counts", "env_human_path_counts",
                 "human_env_min_path_lengths_m", "env_human_mean_path_lengths_m"):
        np.testing.assert_allclose(getattr(left.metadata, name), getattr(right.metadata, name),
                                   rtol=1e-10, atol=1e-15)


@pytest.mark.parametrize("mode", ["parent_face_quadrature", "parent_face_adaptive_analytic"])
@pytest.mark.parametrize("incremental,refresh", [(True, 3), (True, 64), (False, 3)])
def test_hybrid_entry_points_share_outputs_and_visibility_schedule(
    hybrid_case, monkeypatch, mode, incremental, refresh,
):
    case = hybrid_case
    config = case.config(mode, incremental, refresh)
    calls = []
    original = passes.compute_env_human_coupling_channels

    def record(**kwargs):
        fresh = kwargs.get("human_env_active_sets") is None
        assert fresh == (kwargs.get("env_human_active_sets") is None)
        calls.append(fresh)
        return original(**kwargs)

    monkeypatch.setattr(passes, "compute_env_human_coupling_channels", record)
    obj = case.target.add_to_scene(case.scene)
    simulated = case.simulator(config).run(case.scene, case.radar, [case.target], num_frames=2)
    progress_events = []
    fresh = case.run(config, progress_callback=progress_events.append)
    if incremental:
        counts = fresh.cube.metadata.runtime_profile_counts
        assert counts["hybrid_coupling_recomputed_faces"] > 0
        # This parity fixture deliberately forces exact solves on any motion.
        assert counts["hybrid_coupling_phase_updated_faces"] == 0
    for solver_mode in (
        "human_only_po", "hybrid_static_environment", "hybrid_coupling",
    ):
        chirps = [event for event in progress_events
                  if event.get("solver_mode") == solver_mode and "chirp_index" in event]
        assert [(event["frame_index"], event["chirp_index"]) for event in chirps] == [
            (frame, chirp) for frame in range(2) for chirp in range(4)
        ]
        assert all(event["num_frames"] == 2 and event["num_chirps"] == 4
                   for event in chirps)
    trace_count = len(case.trace_calls)
    rebuilt = pipeline.hybrid_rt_path_coupling(
        scene=case.scene, radar=case.radar, target=case.target,
        base_cube=fresh.base_cube, simulator=case.simulator(config), progress=False,
    ).cube
    assert len(case.trace_calls) == trace_count == 2  # Cached coupling performs no RT trace.
    assert obj.scene is case.scene and case.target.name in case.scene.objects
    expected = [not incremental or i % refresh == 0 for i in range(8)]
    assert calls == expected * 3
    _assert_same_components(simulated, fresh.cube)
    _assert_same_components(fresh.cube, rebuilt)
    assert rebuilt.metadata.adc_if_filter == fresh.cube.metadata.adc_if_filter
    assert np.any(np.abs(fresh.cube.components["human_env"]) > 0.)
    assert np.any(np.abs(fresh.cube.components["env_human"]) > 0.)
    assert np.any(fresh.cube.metadata.blocked_static_path_counts > 0)
    assert np.any(fresh.cube.metadata.blocked_static_path_counts == 0)


@pytest.mark.parametrize("override", [False, True])
def test_coupling_incremental_override_controls_surface_reuse(hybrid_case, monkeypatch, override):
    case = hybrid_case
    base = case.config()
    config = replace(base, coupling_incremental_update=override,
                     human_po_config=replace(base.human_po_config,
                         incremental_centroid_displacement_threshold_m=None))
    observed = []
    original = passes.compute_env_human_coupling_channels

    def record(**kwargs):
        observed.append(kwargs["incremental_state"] is not None)
        return original(**kwargs)

    monkeypatch.setattr(passes, "compute_env_human_coupling_channels", record)
    result = case.run(config)
    assert observed == [override] * 8
    counts = result.cube.metadata.runtime_profile_counts
    assert (counts.get("hybrid_coupling_phase_updated_faces", 0) > 0) == override


def test_image_source_po_if_clipping_matches_cached_and_simulator_runs(hybrid_case):
    case = hybrid_case
    config = case.config()
    center = case.target.mesh_sequence.vertices[0].mean(0)
    image_radar = np.array([0., 0., -2.])
    cutoff = abs(case.radar.fmcw.slope) * (
        np.linalg.norm(center) + np.linalg.norm(center-image_radar)) / c
    radar = replace(case.radar, fmcw=replace(case.radar.fmcw, if_bandwidth_hz=cutoff))
    simulated = case.simulator(config).run(case.scene, radar, [case.target], num_frames=2)
    fresh = pipeline.run_calibrated_hybrid_po(
        scene=case.scene, radar=radar, target=case.target, num_frames=2,
        hybrid_config=config, compute_backend="numpy", compute_precision="float64",
        progress=False, print_reflector_summary=False)
    rebuilt = pipeline.hybrid_rt_path_coupling(
        scene=case.scene, radar=radar, target=case.target, base_cube=fresh.base_cube,
        simulator=case.simulator(config), progress=False).cube
    _assert_same_components(simulated, fresh.cube)
    _assert_same_components(fresh.cube, rebuilt)
    assert rebuilt.metadata.adc_if_filter == fresh.cube.metadata.adc_if_filter
    counts = fresh.cube.metadata.adc_if_filter
    assert counts["rejected_path_channels"] > 0
    assert counts["components"]["coupling"]["partially_filtered_path_channels"] > 0


def test_private_hybrid_uses_human_po_times_for_tdm_blocking_and_coupling(hybrid_case, monkeypatch):
    case = hybrid_case
    radar = replace(case.radar,
        hardware=RadarHardware.from_positions(
            [[0., 0., 0.], [0., .0025, 0.]], [[0., 0., 0.]]),
        fmcw=replace(case.radar.fmcw, num_tx=2, tdm_enabled=True, frame_period=.8))
    blocking_vertices, coupling_vertices = [], []
    block = passes.blocked_path_visibility
    couple = passes.compute_env_human_coupling_channels

    def record_blocking(*args, **kwargs):
        blocking_vertices.append(kwargs["blocker_vertices"].copy())
        return block(*args, **kwargs)

    def record_coupling(**kwargs):
        coupling_vertices.append(kwargs["vertices"].copy())
        return couple(**kwargs)

    monkeypatch.setattr(passes, "blocked_path_visibility", record_blocking)
    monkeypatch.setattr(passes, "compute_env_human_coupling_channels", record_coupling)
    # Deliberately bypass run()'s TDM expansion: the two passes must still
    # consume the same authoritative timeline returned by human PO.
    result = case.simulator(case.config())._run_hybrid_static_env_po(
        case.scene, radar, [case.target], num_frames=2)
    times = np.array([[radar.fmcw.chirp_time(frame, chirp) for chirp in range(4)]
                      for frame in range(2)])
    expected_vertices = np.stack([case.target.mesh_sequence.vertices_at(time)
                                  for time in times.ravel()])
    np.testing.assert_array_equal(result.times, times)
    np.testing.assert_array_equal(blocking_vertices, expected_vertices)
    np.testing.assert_array_equal(coupling_vertices, expected_vertices)


def test_real_reflector_selection_matches_across_hybrid_entry_points(hybrid_case, monkeypatch):
    case = hybrid_case
    for index, height in enumerate((-1., -3.)):
        vertices = np.array([[-5., -5., height], [5., -5., height], [0., 5., height]])
        floor = MeshTarget(
            f"selection-floor-{index}",
            MeshSequence(vertices[None], np.array([[0, 1, 2]], np.uint32), np.array([0.])),
            material=human_skin_material(f"selection-floor-material-{index}"),
        )
        floor.add_to_scene(case.scene)
    # Use the real geometry, material, antenna, finite-support and obstruction
    # calculations instead of the parity fixture's predetermined reflector bank.
    monkeypatch.setattr(passes, "select_image_source_reflectors", select_image_source_reflectors)
    config = replace(case.config(), coupling_power_threshold_db=-13., coupling_max_reflectors=2,
                     coupling_min_reflector_area_m2=.5, coupling_reflection_scale=.6)
    case.target.add_to_scene(case.scene)
    simulated = case.simulator(config).run(case.scene, case.radar, [case.target], num_frames=2)
    fresh = case.run(config)
    rebuilt = pipeline.hybrid_rt_path_coupling(
        scene=case.scene, radar=case.radar, target=case.target, base_cube=fresh.base_cube,
        simulator=case.simulator(config), progress=False,
    ).cube
    with case.target.detached_from_scene(case.scene):
        expected = select_image_source_reflectors(
            case.scene, case.radar, case.target, enabled=True,
            power_threshold_db=-13., max_count=2, min_area_m2=.5, reflection_scale=.6,
            time_range_s=(0., case.radar.fmcw.chirp_time(1, 3)),
        ).selection_metadata
    assert expected["candidate_count"] == 2
    assert expected["retained_count"] > 0
    for cube in (simulated, fresh.cube, rebuilt):
        selections = [event["reflector_selection"] for event in cube.metadata.events
                      if isinstance(event.get("reflector_selection"), dict)]
        assert selections[-1] == expected
    assert len(case.trace_calls) == 2
    assert np.any(abs(simulated.components["human_env"]) > 0.)
    _assert_same_components(simulated, fresh.cube)
    _assert_same_components(fresh.cube, rebuilt)


def test_cached_coupling_applies_each_chirps_gain_once(hybrid_case):
    case = hybrid_case
    config = case.config()
    raw = case.run(config)
    gains = np.arange(2., 10.).reshape(2, 4)
    scale = gains[..., None, None]
    # A saved calibrated base already contains the gain on the direct human
    # term, while its static environment terms keep their physical amplitudes.
    components = {name: value.copy() for name, value in raw.base_cube.components.items()}
    components["human_po"] *= scale
    base = replace(raw.base_cube,
        adc=components["human_po"] + components["static_environment_blocked"],
        components=components, metadata=replace(raw.base_cube.metadata,
            po_calibration={"mode": "rt_sequence", "human_po_amplitude_gain": 2.,
                            "human_po_amplitude_gain_map": gains}))
    rebuilt = pipeline.hybrid_rt_path_coupling(
        scene=case.scene, radar=case.radar, target=case.target, base_cube=base,
        simulator=case.simulator(config), progress=False,
    ).cube
    for name in ("human_po", "human_env", "env_human"):
        np.testing.assert_allclose(rebuilt.components[name], raw.cube.components[name]*scale,
                                   rtol=1e-10, atol=1e-18)
    np.testing.assert_array_equal(rebuilt.components["static_environment_blocked"],
                                  raw.cube.components["static_environment_blocked"])


def test_explicit_calibration_gain_matches_across_entry_points(hybrid_case, monkeypatch):
    case = hybrid_case
    config = case.config()
    raw = case.run(config)
    gain = 3.
    reference_cube = replace(raw.cube, adc=raw.cube.components["human_po"]*gain)
    # Keep the actual ADC-power fitting; replace only the stochastic RT input.
    values = {field.name: np.zeros((2, 4)) for field in fields(pipeline.FullRTMobilityBaselineResult)}
    values.update(cube=reference_cube, human_reference_adc=reference_cube.adc[0],
        human_reference_path_counts=np.ones(4, int), retrace_count=1, reference_adc_power={})
    reference = pipeline.FullRTMobilityBaselineResult(**values)
    monkeypatch.setattr(pipeline, "run_full_rt_mobility_baseline", lambda **kw: reference)
    monkeypatch.setattr(MmWaveRadarSimulator, "_run_rt_baseline_for_calibration",
        lambda *a, **kw: pipeline._rt_result_to_calibration_baseline(
            reference, periodic_retrace=False, periodic_retrace_period_chirps=4))
    calibration = POCalibrationConfig(mode="rt_first_pose", keep_rt_baseline=False)
    config = replace(config, human_po_config=replace(config.human_po_config, calibration=calibration))
    simulated = case.simulator(config).run(case.scene, case.radar, [case.target], num_frames=2)
    calibrated = case.run(config, calibration_config=calibration)
    _assert_same_components(simulated, calibrated.cube)
    assert calibrated.calibration["human_po_amplitude_gain"] == pytest.approx(gain)
    for name in ("human_po", "human_env", "env_human"):
        np.testing.assert_allclose(calibrated.cube.components[name], raw.cube.components[name]*gain,
                                   rtol=1e-10, atol=1e-18)


@pytest.mark.parametrize("entry", ["simulator", "pipeline", "recompute"])
@pytest.mark.parametrize("failure", ["cancel", "exception"])
def test_hybrid_failure_stops_coupling_and_restores_mesh(hybrid_case, monkeypatch, entry, failure):
    case = hybrid_case
    config = case.config()
    base = case.run(config).base_cube
    obj = case.target.add_to_scene(case.scene)
    state = {"cancelled": False, "calls": 0}
    original = passes.compute_env_human_coupling_channels

    def stop(**kwargs):
        state["calls"] += 1
        assert case.target.name not in case.scene.objects
        if failure == "exception":
            raise RuntimeError("coupling failure")
        result = original(**kwargs)
        state["cancelled"] = True
        return result

    monkeypatch.setattr(passes, "compute_env_human_coupling_channels", stop)
    cancel_check = lambda: state["cancelled"]
    expected = InterruptedError if failure == "cancel" else RuntimeError
    with pytest.raises(expected):
        if entry == "simulator":
            case.simulator(config, cancel_check=cancel_check).run(
                case.scene, case.radar, [case.target], num_frames=2)
        elif entry == "pipeline":
            case.run(config, cancel_check=cancel_check)
        else:
            pipeline.hybrid_rt_path_coupling(
                scene=case.scene, radar=case.radar, target=case.target, base_cube=base,
                simulator=case.simulator(config, cancel_check=cancel_check), progress=False)
    assert state["calls"] == 1
    assert obj.scene is case.scene and case.target.name in case.scene.objects
