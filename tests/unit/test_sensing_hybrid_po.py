# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Hybrid static-environment PO tests."""

from __future__ import annotations

from types import SimpleNamespace

from sensing_test_helpers import *  # noqa: F401,F403
import mmWaveRadar.simulation.hybrid_pipeline as hybrid_pipeline
from mmWaveRadar.simulation.hybrid_pipeline import (
    CalibratedHybridPOResult,
    HybridCouplingRecomputeResult,
    hybrid_rt_path_coupling,
    run_calibrated_hybrid_po,
    run_hybrid_rt_path_pipeline,
)


def test_full_rt_baseline_propagates_explicit_virtual_channel_mapping():
    captured = {}

    class TestSimulator(MmWaveRadarSimulator):
        def trace_scene(self, scene):
            del scene
            return object()

        def target_interaction_counts(self, paths, targets):
            del paths, targets
            return np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64)

        def path_arrays_from_paths(self, paths, targets, **kwargs):
            del paths, targets
            captured.update(kwargs)
            return (
                np.ones((4, 1), dtype=np.complex128),
                np.array([1e-9], dtype=float),
                np.array([True]),
            )

        def synthesize_adc(self, a, tau, valid, fmcw, **kwargs):
            del a, tau, valid, kwargs
            return np.zeros(
                (fmcw.num_adc_samples, 4),
                dtype=np.complex128,
            )

    fmcw = FMCWConfig(
        carrier_frequency=60e9,
        slope=1e12,
        chirp_duration=1e-3,
        chirp_repetition_time=0.5,
        sampling_frequency=2e3,
        num_adc_samples=4,
        num_chirps_per_frame=1,
        frame_period=1.0,
        num_tx=2,
    )
    expected_tx = np.array([1, 0, 1, 0], dtype=np.int64)
    expected_rx = np.array([1, 1, 0, 0], dtype=np.int64)
    radar = RadarSensor(
        name="radar",
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        hardware=RadarHardware(
            name="permuted",
            tx_positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]]),
            rx_positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]]),
            virtual_channel_tx_indices=expected_tx,
            virtual_channel_rx_indices=expected_rx,
        ),
        fmcw=fmcw,
    )
    target = MeshTarget(
        "human",
        _single_face_mesh_sequence(x0=1.0),
        material=human_skin_material("permuted-mapping-human"),
    )
    simulator = TestSimulator(
        mobility_mode="rt_retrace",
        periodic_retrace=False,
        compute_backend="numpy",
        compute_precision="float64",
    )

    result = hybrid_pipeline.run_full_rt_mobility_baseline(
        scene=load_scene(),
        radar=radar,
        targets=[target],
        simulator=simulator,
        num_frames=1,
    )

    assert result.cube.adc.shape == (1, 1, 4, 4)
    assert captured["virtual_channel_order"] == "tx_major"
    assert np.array_equal(captured["virtual_channel_tx_indices"], expected_tx)
    assert np.array_equal(captured["virtual_channel_rx_indices"], expected_rx)


def test_hybrid_static_env_po_blocks_static_component_and_preserves_sum():
    class FakeStaticPaths:
        def cir(self, normalize_delays=False, out_type="numpy"):
            del normalize_delays, out_type
            a = np.ones((1, 1, 1, 1, 1, 1), dtype=np.complex128)
            tau = np.array([[[1e-9]]], dtype=float)
            return a, tau

        @property
        def vertices(self):
            return np.zeros((1, 1, 1, 1, 3), dtype=float)

        @property
        def interactions(self):
            return np.zeros((1, 1, 1, 1), dtype=np.uint32)

    class TestSimulator(MmWaveRadarSimulator):
        def _trace(self, scene):
            del scene
            return FakeStaticPaths()

    fmcw = FMCWConfig(carrier_frequency=60e9, slope=1e12,
                      chirp_duration=1e-3, chirp_repetition_time=0.5,
                      sampling_frequency=2e3, num_adc_samples=4,
                      num_chirps_per_frame=2, frame_period=1.0)
    radar = RadarSensor(
        name="radar",
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        hardware=RadarHardware.from_positions(
            [[-0.5, 0.0, 0.0]],
            [[0.5, 0.0, 0.0]],
        ),
        fmcw=fmcw,
    )
    target = MeshTarget(
        "human",
        MeshSequence(
            vertices=np.stack([
                _triangle_from_yz(0.0, (-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)),
                _triangle_from_yz(0.0, (-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)),
            ]),
            faces=np.array([[0, 1, 2]], dtype=np.uint32),
            times=np.array([0.0, 1.0]),
        ),
        material=human_skin_material("hybrid-human"),
    )
    base_human_po_config = HumanPOMobilityConfig(
        visibility_samples_per_face=2,
        visibility_fade_chirps=7,
    )
    sim = TestSimulator(
        mobility_mode="hybrid_static_env_po",
        compute_backend="numpy",
        compute_precision="float64",
        human_po_config=base_human_po_config,
        hybrid_static_env_po_config=HybridStaticEnvPOConfig(
            blocking_enabled=True,
            blocking_fade_chirps=1,
            human_po_config=HumanPOMobilityConfig(
                visibility_samples_per_face=1,
                visibility_fade_chirps=1,
            ),
        ),
    )

    cube = sim.run(load_scene(), radar, [target], num_frames=1)

    assert sim.human_po_config == base_human_po_config
    assert cube.metadata.mode == "hybrid_static_env_po"
    assert cube.adc.shape == (1, 2, 4, 1)
    assert np.all(np.isfinite(cube.adc))
    assert cube.components is not None
    assert np.allclose(
        cube.adc,
        cube.components["human_po"]
        + cube.components["static_environment_blocked"]
        + cube.components["human_env"]
        + cube.components["env_human"],
    )
    assert np.allclose(cube.components["static_environment_blocked"], 0.0)
    assert np.allclose(cube.components["human_env"], 0.0)
    assert np.allclose(cube.components["env_human"], 0.0)
    assert np.all(np.abs(cube.components["static_environment_unblocked"]) > 0.0)
    assert np.array_equal(cube.metadata.blocked_static_path_counts, [[1, 1]])
    assert np.array_equal(cube.metadata.static_path_counts, [[1, 1]])
    assert np.allclose(cube.metadata.mean_static_path_visibility, 0.0)
    for key in (
        "hybrid_coupling_setup",
        "hybrid_human_env_channel",
        "hybrid_env_human_channel",
        "hybrid_human_env_adc_synthesis",
        "hybrid_env_human_adc_synthesis",
    ):
        assert key in cube.metadata.runtime_profile_s
        assert key in cube.metadata.runtime_profile_counts

    no_block = TestSimulator(
        mobility_mode="hybrid_static_env_po",
        compute_backend="numpy",
        compute_precision="float64",
        hybrid_static_env_po_config=HybridStaticEnvPOConfig(
            blocking_enabled=False,
            human_po_config=HumanPOMobilityConfig(
                visibility_samples_per_face=1,
                visibility_fade_chirps=1,
            ),
        ),
    ).run(load_scene(), radar, [target], num_frames=1)

    assert np.allclose(
        no_block.components["static_environment_blocked"],
        no_block.components["static_environment_unblocked"],
    )
    assert np.allclose(
        no_block.adc,
        no_block.components["human_po"]
        + no_block.components["static_environment_unblocked"]
        + no_block.components["human_env"]
        + no_block.components["env_human"],
    )


def test_hybrid_rt_path_coupling_is_quiet_and_does_not_propagate_rt_baseline(
    capsys,
):
    radar = _phase_one_radar(num_chirps=1)
    target = MeshTarget(
        "human",
        _single_face_mesh_sequence(x0=1.0),
        material=human_skin_material("hybrid-alias-human"),
    )
    adc = np.ones((1, 1, radar.fmcw.num_adc_samples, 1), dtype=np.complex128)
    zero_adc = np.zeros_like(adc)
    rt_baseline_adc = 2.0 * adc
    first_frame_reference = 3.0 * adc
    base_cube = RadarCube(
        adc=adc,
        times=np.array([[0.0]], dtype=float),
        metadata=SensingMetadata(
            events=[],
            path_counts=np.zeros((1, 1), dtype=np.int64),
            max_displacements=np.zeros((1, 1), dtype=float),
            virtual_channel_order=radar.hardware.virtual_channel_order,
            mode="hybrid_static_env_po",
            runtime_profile_s={},
            runtime_profile_counts={},
        ),
        components={
            "human_po": adc.copy(),
            "static_environment_unblocked": zero_adc.copy(),
            "static_environment_blocked": zero_adc.copy(),
            "static_path_visibility_weights": np.zeros((1, 1, 1, 1), dtype=float),
            "human_env": zero_adc.copy(),
            "env_human": zero_adc.copy(),
            "rt_baseline": rt_baseline_adc,
            "rt_baseline_full_scene": rt_baseline_adc,
            "rt_baseline_one_human_first_frame": first_frame_reference,
        },
    )
    sim = MmWaveRadarSimulator(
        mobility_mode="hybrid_static_env_po",
        compute_backend="numpy",
        compute_precision="float64",
        hybrid_static_env_po_config=HybridStaticEnvPOConfig(
            coupling_enabled=False,
            human_po_config=HumanPOMobilityConfig(),
        ),
    )

    result = hybrid_rt_path_coupling(
        scene=load_scene(),
        radar=radar,
        target=target,
        base_cube=base_cube,
        simulator=sim,
        recompute=False,
        progress=False,
    )

    assert result.cube.components is not None
    assert "rt_baseline" not in result.cube.components
    assert "rt_baseline_full_scene" not in result.cube.components
    assert np.array_equal(
        result.cube.components["rt_baseline_one_human_first_frame"],
        first_frame_reference,
    )
    assert capsys.readouterr().out == ""


def test_cached_hybrid_pipeline_preserves_multi_tx_motion_phase_timing(
    monkeypatch,
):
    hardware = RadarHardware.from_positions(
        [[0.0, 0.0, 0.0], [0.0, 0.0025, 0.0]],
        [[0.0, 0.0, 0.0]],
    )
    fmcw = FMCWConfig(
        carrier_frequency=60e9,
        slope=1e12,
        chirp_duration=0.5e-3,
        chirp_repetition_time=1e-3,
        sampling_frequency=4e3,
        num_adc_samples=4,
        num_chirps_per_frame=4,
        frame_period=0.1,
        num_tx=2,
        tdm_enabled=True,
    )
    radar = RadarSensor(
        name="cached-tdm",
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        hardware=hardware,
        fmcw=fmcw,
    )
    doppler_hz = 25.0
    adc = np.empty((1, 4, 4, 2), dtype=np.complex128)
    for chirp in range(4):
        for channel, tx_index in enumerate(hardware.virtual_tx_indices()):
            phase = np.exp(
                2j * np.pi * doppler_hz
                * fmcw.tx_chirp_time(0, chirp, int(tx_index))
            )
            adc[0, chirp, :, channel] = phase
    zeros = np.zeros_like(adc)
    packed_times = np.asarray([[0.25, 0.50, 0.90, 1.40]])
    base_cube = RadarCube(
        adc=adc,
        times=packed_times,
        metadata=SensingMetadata(
            events=[],
            path_counts=np.ones((1, 4), dtype=np.int64),
            max_displacements=np.zeros((1, 4), dtype=float),
            virtual_channel_order=hardware.virtual_channel_order,
            mode="hybrid_static_env_po",
            runtime_profile_s={},
            runtime_profile_counts={},
        ),
        components={
            "human_po": adc.copy(),
            "static_environment_unblocked": zeros.copy(),
            "static_environment_blocked": zeros.copy(),
            "static_path_visibility_weights": np.zeros((1, 4, 1)),
            "human_env": zeros.copy(),
            "env_human": zeros.copy(),
        },
    )
    observed_vertices = []

    class TestSimulator(MmWaveRadarSimulator):
        def trace_scene(self, scene):
            del scene
            return object()

        def synthesize_adc(self, a, tau, valid, fmcw, **kwargs):
            del a, tau, valid, kwargs
            return np.zeros(
                (fmcw.num_adc_samples, hardware.num_virtual_channels),
                dtype=np.complex128,
            )

    class TestVisibilityScene:
        def __init__(self, faces, vertices, *, name):
            del faces, vertices, name

        def update_vertices(self, vertices):
            del vertices

    def fake_coupling(**kwargs):
        observed_vertices.append(np.asarray(kwargs["vertices"]).copy())
        coefficients = np.zeros(
            (hardware.num_virtual_channels, 1),
            dtype=np.complex128,
        )
        delays = np.asarray([1.0e-9])
        valid = np.asarray([True])
        return SimpleNamespace(
            human_env_coefficients=coefficients,
            human_env_delays_s=delays,
            human_env_valid=valid,
            env_human_coefficients=coefficients,
            env_human_delays_s=delays,
            env_human_valid=valid,
            runtime_profile_s={},
            runtime_profile_counts={},
        )

    monkeypatch.setattr(
        RadarSensor,
        "configure_scene",
        lambda self, scene: (None, None),
    )
    monkeypatch.setattr(
        hybrid_pipeline,
        "extract_static_path_bank",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        hybrid_pipeline,
        "extract_rt_reflector_bank",
        lambda *args, **kwargs: SimpleNamespace(
            vertices=np.zeros((1, 3, 3)),
        ),
    )
    monkeypatch.setattr(
        hybrid_pipeline,
        "HumanVisibilityScene",
        TestVisibilityScene,
    )
    monkeypatch.setattr(
        hybrid_pipeline,
        "compute_env_human_coupling_channels",
        fake_coupling,
    )

    simulator = TestSimulator(
        mobility_mode="hybrid_static_env_po",
        compute_backend="numpy",
        compute_precision="float64",
        hybrid_static_env_po_config=HybridStaticEnvPOConfig(
            coupling_enabled=True,
        ),
    )
    target = MeshTarget(
        "moving-human",
        _single_face_mesh_sequence(x0=1.0, x1=1.1),
        material=human_skin_material("cached-tdm-human"),
    )

    result = run_hybrid_rt_path_pipeline(
        scene=SimpleNamespace(mi_scene=None),
        radar=radar,
        target=target,
        base_cube=base_cube,
        simulator=simulator,
        recompute=True,
        progress=False,
    )

    np.testing.assert_allclose(result.cube.adc, adc)
    np.testing.assert_allclose(result.cube.times, packed_times)
    np.testing.assert_allclose(
        result.cube.metadata.virtual_channel_times_s[0, :, 1]
        - result.cube.metadata.virtual_channel_times_s[0, :, 0],
        fmcw.chirp_repetition_time,
    )
    expected_phase_offset = -2.0 * np.pi * doppler_hz * fmcw.chirp_repetition_time
    np.testing.assert_allclose(
        np.angle(
            result.cube.adc[0, :, 0, 0]
            * result.cube.adc[0, :, 0, 1].conj()
        ),
        expected_phase_offset,
    )
    physical_times = (
        packed_times[:, :, None]
        + np.arange(2) * fmcw.chirp_repetition_time
    ).reshape(-1)
    observed_x = np.asarray(
        [np.mean(vertices[:, 0]) for vertices in observed_vertices]
    )
    np.testing.assert_allclose(observed_x, 1.0 + 0.05 * physical_times)
    assert observed_x[1] - observed_x[0] == pytest.approx(
        0.05 * fmcw.chirp_repetition_time,
        rel=2.0e-3,
    )


def test_run_calibrated_hybrid_po_wraps_base_and_coupling(monkeypatch, capsys):
    class TestSimulator(MmWaveRadarSimulator):
        def _run_human_only_po(
            self,
            radar,
            targets,
            *,
            num_frames,
            static_visibility_scene=None,
            rt_reference_scene=None,
            human_po_config=None,
        ):
            del targets, static_visibility_scene, rt_reference_scene
            del human_po_config
            shape = (
                int(num_frames),
                radar.fmcw.num_chirps_per_frame,
                radar.fmcw.num_adc_samples,
                radar.hardware.num_virtual_channels,
            )
            adc = 2.0 * np.ones(shape, dtype=np.complex128)
            times = np.zeros(shape[:2], dtype=float)
            for frame in range(shape[0]):
                for chirp in range(shape[1]):
                    times[frame, chirp] = radar.fmcw.chirp_time(frame, chirp)
            return RadarCube(
                adc=adc,
                times=times,
                metadata=SensingMetadata(
                    events=[],
                    path_counts=np.ones(shape[:2], dtype=np.int64),
                    max_displacements=np.zeros(shape[:2], dtype=float),
                    virtual_channel_order=radar.hardware.virtual_channel_order,
                    mode="human_only_po",
                    total_path_power=np.full(shape[:2], 4.0, dtype=np.float64),
                ),
            )

    calibration_calls = []

    def fake_full_rt_baseline(*, scene, radar, targets, simulator, num_frames):
        del scene, simulator
        calibration_calls.append({
            "num_frames": int(num_frames),
            "start_time_s": float(targets[0].mesh_sequence.start_time_s),
        })
        shape = (
            int(num_frames),
            radar.fmcw.num_chirps_per_frame,
            radar.fmcw.num_adc_samples,
            radar.hardware.num_virtual_channels,
        )
        adc = np.ones(shape, dtype=np.complex128)
        times = np.zeros(shape[:2], dtype=float)
        for frame in range(shape[0]):
            for chirp in range(shape[1]):
                times[frame, chirp] = radar.fmcw.chirp_time(frame, chirp)
        metadata = SensingMetadata(
            events=[{
                "frame": 0,
                "chirp": 0,
                "time": 0.0,
                "coherent_path_update": True,
            }],
            path_counts=np.ones(shape[:2], dtype=np.int64),
            max_displacements=np.zeros(shape[:2], dtype=float),
            virtual_channel_order=radar.hardware.virtual_channel_order,
            mode="rt_baseline",
        )
        power = np.full(shape[:2], 4.0, dtype=np.float64)
        return hybrid_pipeline.FullRTMobilityBaselineResult(
            cube=RadarCube(adc=adc, times=times, metadata=metadata),
            human_reference_adc=adc[0],
            human_reference_path_counts=np.ones(shape[1], dtype=np.int64),
            human_touch_path_counts=np.ones(shape[:2], dtype=np.int64),
            one_human_touch_path_counts=np.ones(shape[:2], dtype=np.int64),
            single_human_only_path_counts=np.ones(shape[:2], dtype=np.int64),
            human_env_coupled_path_counts=np.zeros(shape[:2], dtype=np.int64),
            human_multi_touch_path_counts=np.zeros(shape[:2], dtype=np.int64),
            path_depth_histogram=np.zeros(shape[:2] + (3,), dtype=np.int64),
            total_path_power=power,
            human_touch_path_power=power,
            one_human_touch_path_power=power,
            single_human_only_path_power=power,
            human_env_coupled_path_power=np.zeros(shape[:2], dtype=np.float64),
            human_multi_touch_path_power=np.zeros(shape[:2], dtype=np.float64),
            retrace_count=int(num_frames),
        )

    def fake_static_pass(*, scene, radar, target, times, simulator, config):
        del scene, radar, target, simulator, config
        adc_shape = times.shape + (4, 1)
        return hybrid_pipeline._StaticEnvironmentPass(
            static_unblocked_adc=5.0 * np.ones(adc_shape, dtype=np.complex128),
            static_blocked_adc=3.0 * np.ones(adc_shape, dtype=np.complex128),
            visibility_weights=np.ones(times.shape + (1,), dtype=float),
            static_path_counts=np.ones(times.shape, dtype=np.int64),
            blocked_counts=np.zeros(times.shape, dtype=np.int64),
            mean_visibility=np.ones(times.shape, dtype=float),
            static_bank=SimpleNamespace(
                valid=np.ones(1, dtype=bool),
                segment_path_indices=np.arange(1, dtype=np.int64),
            ),
            reflector_bank=SimpleNamespace(vertices=np.zeros((0, 3, 3))),
            runtime_profile_s={},
            runtime_profile_counts={},
            wall_time_s=0.0,
        )

    def fake_coupling_pass(**kwargs):
        times = kwargs["times"]
        adc_shape = times.shape + (4, 1)
        return hybrid_pipeline._CouplingPass(
            human_env_adc=7.0 * np.ones(adc_shape, dtype=np.complex128),
            env_human_adc=11.0 * np.ones(adc_shape, dtype=np.complex128),
            human_env_counts=np.ones(times.shape, dtype=np.int64),
            env_human_counts=2 * np.ones(times.shape, dtype=np.int64),
            human_env_min_lengths=np.zeros(times.shape, dtype=float),
            human_env_mean_lengths=np.zeros(times.shape, dtype=float),
            human_env_max_lengths=np.zeros(times.shape, dtype=float),
            env_human_min_lengths=np.zeros(times.shape, dtype=float),
            env_human_mean_lengths=np.zeros(times.shape, dtype=float),
            env_human_max_lengths=np.zeros(times.shape, dtype=float),
            runtime_profile_s={},
            runtime_profile_counts={},
            wall_time_s=0.0,
        )

    monkeypatch.setattr(hybrid_pipeline, "MmWaveRadarSimulator", TestSimulator)
    monkeypatch.setattr(
        hybrid_pipeline,
        "run_full_rt_mobility_baseline",
        fake_full_rt_baseline,
    )
    monkeypatch.setattr(
        hybrid_pipeline,
        "_run_static_environment_pass",
        fake_static_pass,
    )
    monkeypatch.setattr(hybrid_pipeline, "_run_coupling_pass", fake_coupling_pass)

    radar = _phase_one_radar(num_chirps=1)
    target = MeshTarget(
        "human",
        _single_face_mesh_sequence(x0=1.0, x1=1.0),
        material=human_skin_material("hybrid-flow-human"),
    )

    result = run_calibrated_hybrid_po(
        scene=load_scene(),
        coupling_scene=load_scene(),
        radar=radar,
        target=target,
        num_frames=2,
        coupling_enabled=False,
        rt_samples_per_src=1,
        rt_max_num_paths_per_src=1,
        compute_backend="numpy",
        compute_precision="float64",
        progress=True,
        print_reflector_summary=False,
    )

    assert isinstance(result, CalibratedHybridPOResult)
    assert result.base_config.coupling_enabled is False
    assert result.hybrid_config.coupling_enabled is False
    assert result.calibration_config.mode == "rt_sequence"
    assert calibration_calls == [{"num_frames": 1, "start_time_s": 0.0}]
    assert result.num_chirps == int(np.prod(result.cube.adc.shape[:2]))
    assert np.allclose(result.cube.adc, 23.0)
    assert np.allclose(
        result.cube.adc,
        result.cube.components["human_po"]
        + result.cube.components["static_environment_blocked"]
        + result.cube.components["human_env"]
        + result.cube.components["env_human"],
    )
    assert "rt_baseline_full_scene" not in result.base_cube.components
    assert "rt_baseline" not in result.base_cube.components
    assert "rt_baseline_one_human_first_frame" not in result.base_cube.components
    output = capsys.readouterr().out
    assert "[hybrid-po] stage 1/6: full-scene RT calibration frames=[0, 1)" in output
    assert "[hybrid-po] stage 2/6: human-only PO" in output
    assert "[hybrid-po] stage 3/6: static-environment RT" in output
    assert "[hybrid-po] stage 4/6: PO for H_env" in output
    assert "[hybrid-po] stage 5/6: report E_human" in output
    assert "[hybrid-po] stage 6/6: assemble coherent Hybrid PO cube" in output

    calibration_calls.clear()
    shifted = run_calibrated_hybrid_po(
        scene=load_scene(),
        coupling_scene=load_scene(),
        radar=radar,
        target=target,
        num_frames=2,
        calibration_frame_range=(1, 2),
        coupling_enabled=False,
        rt_samples_per_src=1,
        rt_max_num_paths_per_src=1,
        compute_backend="numpy",
        compute_precision="float64",
        progress=False,
        print_reflector_summary=False,
    )
    assert calibration_calls[0]["num_frames"] == 1
    assert calibration_calls[0]["start_time_s"] == pytest.approx(
        radar.fmcw.chirp_time(1, 0)
    )
    assert shifted.calibration["calibration_frame_range"] == (1, 2)


def test_run_calibrated_hybrid_po_rejects_invalid_calibration_frame_range():
    radar = _phase_one_radar(num_chirps=1)
    target = MeshTarget(
        "human",
        _single_face_mesh_sequence(x0=1.0, x1=1.0),
        material=human_skin_material("hybrid-range-human"),
    )

    with pytest.raises(ValueError, match="start must be non-negative"):
        run_calibrated_hybrid_po(
            scene=None,
            radar=radar,
            target=target,
            num_frames=2,
            calibration_frame_range=(-1, 1),
        )
    with pytest.raises(ValueError, match="stop must be greater than start"):
        run_calibrated_hybrid_po(
            scene=None,
            radar=radar,
            target=target,
            num_frames=2,
            calibration_frame_range=(1, 1),
        )
    with pytest.raises(ValueError, match="less than or equal to num_frames"):
        run_calibrated_hybrid_po(
            scene=None,
            radar=radar,
            target=target,
            num_frames=2,
            calibration_frame_range=(1, 3),
        )
