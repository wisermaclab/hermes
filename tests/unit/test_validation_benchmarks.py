# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Tests for dataset-neutral real-radar validation benchmarks."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mmWaveRadar.radar import FMCWConfig, RadarHardware, RadarSensor
from bundle_prepare.contract import (
    Bundle as BenchmarkClip,
    BundleSensorConfig as BenchmarkSensorConfig,
)
from bundle_prepare.integrity import (
    check_bundle_integrity,
    main as bundle_integrity_main,
)
from validation.cli import _parse_args, main as validation_main
from validation.metrics import (
    MeshRangeSummary,
    background_subtract_adc,
    mesh_range_summary,
    normalized_correlation,
    range_gate_mask,
    suppress_zero_doppler,
    zero_doppler_power_fraction,
)
from validation.runner import (
    _BundleFrameMeshSequence,
    _PlacedBundleMotionSequence,
    DEFAULT_VALIDATION_MOBILITY_MODES,
    VALIDATION_MOBILITY_MODES,
    BenchmarkRunConfig,
    _channel_tx_rx_indices_1based,
    _hybrid_static_env_po_config,
    _clip_uses_tdm_virtual_adc,
    _labeled_mobility_modes,
    _load_amass_motion_sequence,
    _resolve_clutter_removal,
    _resolve_scene_path,
    _simulate_adc,
    _simultaneous_tdm_sensor,
    _subarray_selection,
    _tdm_slot_rx_outputs,
    _simulated_adc_cache_matches,
    _simulation_mesh_sequence,
    _sensor_config_cache_payload,
    _ti_digitized_pattern_fingerprint,
    _write_simulated_adc_cache,
    run_validation_benchmark,
    simulate_bundle_adc,
)


@pytest.fixture(autouse=True)
def _stub_amass_runtime(monkeypatch):
    """Avoid licensed SMPL model files in dataset-neutral validation tests."""

    import validation.runner as runner
    from mmWaveRadar.targets import MeshSequence

    def load_motion(path, _config):
        with np.load(path, allow_pickle=False) as data:
            timing_key = "bundle_times" if "bundle_times" in data.files else "times"
            times = np.asarray(data[timing_key], dtype=float)
            translations = np.asarray(data["trans"], dtype=np.float32)
            faces = np.asarray(data["faces"], dtype=np.uint32)
        base = np.asarray(
            [[2.0, -0.2, 0.0], [2.0, 0.2, 0.0], [2.0, 0.0, 0.5]],
            dtype=np.float32,
        )
        vertices = np.stack(
            [base + translation for translation in translations], axis=0
        )
        return MeshSequence(vertices=vertices, faces=faces, times=times)

    monkeypatch.setattr(runner, "_load_amass_motion_sequence", load_motion)


def test_simulate_bundle_adc_builds_bundle_aware_runtime_config(monkeypatch):
    import validation.runner as runner

    captured = {}
    expected = np.ones((1, 2, 4, 3), dtype=np.complex64)

    def simulate(clip, config, *, mobility_mode, cancel_check=None):
        captured.update(
            clip=clip,
            config=config,
            mobility_mode=mobility_mode,
            cancel_check=cancel_check,
        )
        return expected

    monkeypatch.setattr(runner, "_simulate_adc", simulate)
    clip = SimpleNamespace(root=Path("bundle"))

    result = simulate_bundle_adc(
        clip,
        mobility_mode="rt_coherent_bank",
        smpl_model_dir="/models/smpl",
    )

    assert result is expected
    assert captured["clip"] is clip
    assert captured["mobility_mode"] == "rt_coherent_bank"
    assert captured["cancel_check"] is None
    assert captured["config"].mobility_modes == ("rt_coherent_bank",)
    assert captured["config"].smpl_model_dir == "/models/smpl"
    sensor = SimpleNamespace(
        hardware=SimpleNamespace(
            num_rx=3,
            virtual_tx_indices=lambda: np.asarray([0, 0, 0]),
            virtual_rx_indices=lambda: np.asarray([0, 1, 2]),
        )
    )
    channel_clip = SimpleNamespace(
        root=Path("bundle"),
        real_adc=np.zeros((1, 2, 4, 3), dtype=np.complex64),
        sensor_config=SimpleNamespace(metadata={}),
        sensor=lambda: sensor,
    )
    assert simulate_bundle_adc(
        channel_clip,
        mobility_mode="human_only_po",
        channel_zero_only=True,
    ) is expected
    assert captured["config"].subarray_tx_indices == (1,)
    assert captured["config"].subarray_rx_indices == (1,)
    cancel_check = lambda: False
    assert simulate_bundle_adc(
        clip,
        mobility_mode="human_only_po",
        cancel_check=cancel_check,
    ) is expected
    assert captured["cancel_check"] is cancel_check
    with pytest.raises(ValueError, match="cancel_check must be callable"):
        simulate_bundle_adc(
            clip,
            mobility_mode="human_only_po",
            cancel_check=False,
        )
    with pytest.raises(ValueError, match="mobility_mode must be one of"):
        simulate_bundle_adc(clip, mobility_mode="unknown")


def _write_clip(root: Path, *, adc_ndim: int = 4) -> np.ndarray:
    root.mkdir(parents=True, exist_ok=True)
    (root / "bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "hermes",
                "primary_adc": "primary",
                "adcs": {
                    "primary": {
                        "path": "radar_adc.npz",
                        "origin": "measurement",
                    },
                    "background": {
                        "path": "background_adc.npz",
                        "origin": "background",
                    },
                },
                "motion": {"parameters": "amass_sequence.npz"},
            }
        ),
        encoding="utf-8",
    )
    sensor = {
        "board_model": "IWR6843ISK",
        "pattern_mode": "none",
        "metadata": {
            "dataset": "RT-Pose",
            "tdm_virtual_adc": True,
            "source_adc_axes": ["adc_sample", "chirp_loop", "rx", "tx"],
            "tx_to_enable": [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        },
        "radar": {
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0],
        },
        "fmcw": {
            "carrier_frequency_hz": 60.0e9,
            "slope_hz_per_s": 20.0e12,
            "chirp_duration_s": 60.0e-6,
            "chirp_repetition_time_s": 80.0e-6,
            "sampling_frequency_hz": 2.0e6,
            "num_adc_samples": 16,
            "num_chirps_per_frame": 4,
            "frame_period_s": 0.1,
            "num_tx": 3,
        },
    }
    (root / "sensor.json").write_text(json.dumps(sensor), encoding="utf-8")
    faces = np.asarray([[0, 1, 2]], dtype=np.uint32)
    # Keep one pose beyond the final radar frame so tests exercise the same
    # full-acquisition coverage required from prepared bundles.
    times = np.asarray([0.0, 0.1, 0.2], dtype=float)
    np.savez(
        root / "amass_sequence.npz",
        poses=np.zeros((3, 72), dtype=np.float32),
        trans=np.asarray(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
            dtype=np.float32,
        ),
        betas=np.zeros(10, dtype=np.float32),
        faces=faces,
        times=times,
        gender=np.asarray("neutral"),
        model_type=np.asarray("smpl"),
    )
    adc = _synthetic_adc(frames=2, chirps=4, samples=16, channels=12)
    if adc_ndim != 4:
        adc_to_write = adc.reshape((adc.shape[0], adc.shape[1], -1))
    else:
        adc_to_write = adc
    np.savez(root / "radar_adc.npz", adc=adc_to_write)
    np.savez(root / "background_adc.npz", adc=np.zeros_like(adc[:1]))
    frames_json = {
        "frames": [
            {
                "benchmark_index": 0,
                "motion_time_s": 0.0,
                "radar_frame_id": "000001",
                "camera_frame_id": "000001",
                "pose_frame_id": "000001",
                "metadata": {"subject": "S01"},
            },
            {
                "benchmark_index": 1,
                "motion_time_s": 0.1,
                "radar_frame_id": "000002",
                "camera_frame_id": "000004",
                "pose_frame_id": "000004",
            },
        ]
    }
    (root / "frames.json").write_text(json.dumps(frames_json), encoding="utf-8")
    return adc


def test_bundle_sensor_maps_legacy_tdm_adc_metadata_to_fmcw(tmp_path):
    _write_clip(tmp_path)

    sensor = BenchmarkSensorConfig.from_json(tmp_path / "sensor.json").to_sensor()

    assert sensor.fmcw.tdm_enabled is True
    assert sensor.fmcw.slow_time_interval == pytest.approx(3 * 80.0e-6)


def test_bundle_sensor_explicit_tdm_flag_takes_precedence(tmp_path):
    _write_clip(tmp_path)
    path = tmp_path / "sensor.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tdm_enabled"] = False
    payload["metadata"]["tdm_virtual_adc"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    sensor = BenchmarkSensorConfig.from_json(path).to_sensor()

    assert sensor.fmcw.tdm_enabled is False
    assert sensor.fmcw.slow_time_interval == pytest.approx(80.0e-6)


def test_bundle_sensor_accepts_explicit_fmcw_tdm_flag(tmp_path):
    _write_clip(tmp_path)
    path = tmp_path / "sensor.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metadata"].pop("tdm_virtual_adc")
    payload["fmcw"]["tdm_enabled"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    sensor = BenchmarkSensorConfig.from_json(path).to_sensor()

    assert sensor.fmcw.tdm_enabled is True
    assert sensor.fmcw.slow_time_interval == pytest.approx(3 * 80.0e-6)

    clip = BenchmarkClip.load(tmp_path)
    assert _clip_uses_tdm_virtual_adc(clip) is True


def test_bundle_sensor_without_tdm_marker_defaults_to_simultaneous(tmp_path):
    _write_clip(tmp_path)
    path = tmp_path / "sensor.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metadata"].pop("tdm_virtual_adc")
    payload["metadata"].pop("source_adc_axes")
    path.write_text(json.dumps(payload), encoding="utf-8")

    sensor = BenchmarkSensorConfig.from_json(path).to_sensor()

    assert sensor.fmcw.tdm_enabled is False
    assert sensor.fmcw.slow_time_interval == pytest.approx(80.0e-6)


def test_bundle_sensor_recognizes_legacy_tdm_adc_axes(tmp_path):
    _write_clip(tmp_path)
    path = tmp_path / "sensor.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metadata"].pop("tdm_virtual_adc")
    path.write_text(json.dumps(payload), encoding="utf-8")

    sensor = BenchmarkSensorConfig.from_json(path).to_sensor()

    assert sensor.fmcw.tdm_enabled is True


def test_validation_cli_defaults_to_running_simulation():
    args = _parse_args(["--clip", "clip", "--out", "out"])

    assert args.simulated_adc is None
    assert args.mobility_mode == DEFAULT_VALIDATION_MOBILITY_MODES
    assert args.background_subtract is False
    assert args.hybrid_po_calibration == "rt_first_pose"
    assert args.ignore_tdm_timing is False


def test_validation_cli_help_lists_mobility_modes_and_preprocessing(capsys):
    with pytest.raises(SystemExit) as exc:
        _parse_args(["-h"])

    assert exc.value.code == 0
    output = capsys.readouterr().out
    flat_output = " ".join(output.split())
    for mode in VALIDATION_MOBILITY_MODES:
        assert mode in output
    assert "--background-subtract" in flat_output
    assert "--no-background-subtract" not in flat_output
    assert "Default behavior is no real-ADC background subtraction" in flat_output
    assert "DSP-domain slow-time clutter removal" in flat_output
    assert "--clutter-removal {none,mean}" in flat_output
    assert "--ignore-tdm-timing" in flat_output


def test_validation_cli_parses_optional_simulated_adc():
    args = _parse_args([
        "--clip",
        "clip",
        "--out",
        "out",
        "--simulated-adc",
        "sim_adc.npz",
    ])

    assert str(args.simulated_adc) == "sim_adc.npz"


def test_default_clutter_removal_is_none_and_ignores_bundle_metadata(tmp_path):
    _write_clip(tmp_path)
    clip = BenchmarkClip.load(tmp_path)

    assert _resolve_clutter_removal(clip, BenchmarkRunConfig()) == "none"

    sensor_path = tmp_path / "sensor.json"
    sensor = json.loads(sensor_path.read_text(encoding="utf-8"))
    sensor["metadata"]["default_clutter_removal"] = "mean"
    sensor_path.write_text(json.dumps(sensor), encoding="utf-8")

    clip = BenchmarkClip.load(tmp_path)
    assert _resolve_clutter_removal(clip, BenchmarkRunConfig()) == "none"
    assert _resolve_clutter_removal(
        clip,
        BenchmarkRunConfig(clutter_removal="mean"),
    ) == "mean"


def test_benchmark_clip_loads_3d_background_with_chirp_times(tmp_path):
    adc = _write_clip(tmp_path)
    background = np.zeros_like(adc[0])
    times = np.arange(adc.shape[1], dtype=float)
    np.savez(tmp_path / "background_adc.npz", adc=background, times=times)

    clip = BenchmarkClip.load(tmp_path)

    assert clip.background_adc is not None
    assert clip.background_adc.shape == adc.shape[1:]
    assert np.array_equal(clip.background_adc, background)


def test_benchmark_clip_loads_4d_background_with_frame_chirp_times(tmp_path):
    adc = _write_clip(tmp_path)
    background = np.zeros_like(adc[:1])
    times = np.zeros((1, adc.shape[1]), dtype=float)
    np.savez(tmp_path / "background_adc.npz", adc=background, times=times)

    clip = BenchmarkClip.load(tmp_path)

    assert clip.background_adc is not None
    assert clip.background_adc.shape == (1, *adc.shape[1:])


def test_benchmark_clip_rejects_background_times_with_wrong_axes(tmp_path):
    adc = _write_clip(tmp_path)
    np.savez(
        tmp_path / "background_adc.npz",
        adc=np.zeros_like(adc[0]),
        times=np.zeros((1, adc.shape[1]), dtype=float),
    )

    with pytest.raises(ValueError, match=r"times must have shape \(4,\)"):
        BenchmarkClip.load(tmp_path)


def test_background_subtraction_is_opt_in_and_requires_background_file(tmp_path):
    clip_dir = tmp_path / "clip"
    adc = _write_clip(clip_dir)
    (clip_dir / "background_adc.npz").unlink()
    descriptor = json.loads((clip_dir / "bundle.json").read_text(encoding="utf-8"))
    descriptor["adcs"].pop("background")
    (clip_dir / "bundle.json").write_text(
        json.dumps(descriptor),
        encoding="utf-8",
    )
    sim_path = tmp_path / "sim_adc.npz"
    np.savez(sim_path, adc=adc)
    clip = BenchmarkClip.load(clip_dir)

    metrics = run_validation_benchmark(
        clip,
        BenchmarkRunConfig(
            suite="smoke",
            out_dir=tmp_path / "out_default",
            simulated_adc_path=sim_path,
            simulate=False,
            write_plots=False,
        ),
    )
    assert metrics["background_subtract"] is False
    assert metrics["fmcw"]["tdm_enabled"] is True

    with pytest.raises(ValueError, match="background_adc\\.npz"):
        run_validation_benchmark(
            clip,
            BenchmarkRunConfig(
                suite="smoke",
                out_dir=tmp_path / "out_enabled",
                simulated_adc_path=sim_path,
                simulate=False,
                background_subtract=True,
                write_plots=False,
            ),
        )


def test_validation_cli_parses_simulation_controls():
    args = _parse_args([
        "--clip",
        "clip",
        "--out",
        "out",
        "--mobility-mode",
        "hybrid_static_env_po",
        "rt_retrace",
        "--no-diffuse-reflection",
        "--refraction",
        "--human-specular-reflection",
        "--hybrid-po-calibration",
        "rt_sequence",
        "--ignore-tdm-timing",
    ])

    assert args.mobility_mode == ["hybrid_static_env_po", "rt_retrace"]
    assert args.no_diffuse_reflection is True
    assert args.refraction is True
    assert args.human_specular_reflection is True
    assert args.hybrid_po_calibration == "rt_sequence"
    assert args.ignore_tdm_timing is True


def test_validation_hybrid_po_calibration_defaults_to_first_pose():
    config = BenchmarkRunConfig()
    hybrid = _hybrid_static_env_po_config(
        "hybrid_static_env_po",
        config.hybrid_po_calibration_mode,
    )

    assert config.hybrid_po_calibration_mode == "rt_first_pose"
    assert hybrid.human_po_config.calibration.mode == "rt_first_pose"
    assert _hybrid_static_env_po_config("rt_retrace") is None


def test_validation_hybrid_po_calibration_can_be_disabled():
    hybrid = _hybrid_static_env_po_config("hybrid_static_env_po", "none")

    assert hybrid.human_po_config.calibration.mode == "none"


def test_validation_rejects_unknown_hybrid_po_calibration_mode():
    with pytest.raises(ValueError, match="hybrid_po_calibration_mode"):
        BenchmarkRunConfig(hybrid_po_calibration_mode="unknown")


def test_validation_cli_rejects_duplicate_mobility_modes():
    with pytest.raises(SystemExit):
        _parse_args([
            "--clip",
            "clip",
            "--out",
            "out",
            "--mobility-mode",
            "rt_retrace",
            "rt_retrace",
        ])


def test_validation_range_gate_uses_exact_mesh_extent_without_margin():
    summary = MeshRangeSummary(
        min_m=np.asarray([1.75]),
        median_m=np.asarray([2.0]),
        max_m=np.asarray([2.25]),
    )
    ranges = np.asarray([1.74, 1.75, 2.0, 2.25, 2.26])

    assert range_gate_mask(ranges, summary).tolist() == [
        False,
        True,
        True,
        True,
        False,
    ]


def test_validation_runner_labels_common_po_rt_mobility_pair():
    assert _labeled_mobility_modes(("human_only_po", "rt_coherent_bank")) == (
        ("po", "human_only_po"),
        ("rt", "rt_coherent_bank"),
    )
    assert _labeled_mobility_modes(("hybrid_static_env_po",)) == (
        ("sim", "hybrid_static_env_po"),
    )
    assert _labeled_mobility_modes(("rt_retrace", "rt_coherent_bank")) == (
        ("rt_retrace", "rt_retrace"),
        ("rt_coherent_bank", "rt_coherent_bank"),
    )


def _synthetic_adc(*, frames: int, chirps: int, samples: int, channels: int):
    frame = np.arange(frames, dtype=float)[:, None, None, None]
    chirp = np.arange(chirps, dtype=float)[None, :, None, None]
    sample = np.arange(samples, dtype=float)[None, None, :, None]
    channel = np.arange(channels, dtype=float)[None, None, None, :]
    phase = 2.0 * np.pi * (
        0.18 * sample + 0.07 * chirp + 0.01 * channel + 0.03 * frame
    )
    amplitude = 1.0 + 0.05 * frame
    return (amplitude * np.exp(1j * phase)).astype(np.complex64)


def test_benchmark_clip_loads_standard_bundle(tmp_path):
    adc = _write_clip(tmp_path)

    clip = BenchmarkClip.load(tmp_path)

    assert clip.num_frames == 2
    assert clip.real_adc.shape == adc.shape
    assert clip.sensor().hardware.num_virtual_channels == 12
    assert clip.frames[1].radar_frame_id == "000002"
    assert np.allclose(clip.motion_times_s, [0.0, 0.1])


def test_benchmark_clip_loads_environment_json_scene_path(tmp_path):
    _write_clip(tmp_path)
    (tmp_path / "environment.json").write_text(
        json.dumps({"scene_path": "environment_scene.xml"}),
        encoding="utf-8",
    )
    descriptor = json.loads((tmp_path / "bundle.json").read_text(encoding="utf-8"))
    descriptor["environment"] = "environment.json"
    (tmp_path / "bundle.json").write_text(
        json.dumps(descriptor),
        encoding="utf-8",
    )

    clip = BenchmarkClip.load(tmp_path)
    scene_path = _resolve_scene_path(clip, BenchmarkRunConfig(out_dir=tmp_path / "out"))

    assert clip.environment["scene_path"] == "environment_scene.xml"
    assert scene_path == str((tmp_path / "environment_scene.xml").resolve())


def test_benchmark_clip_rejects_adc_without_frame_chirp_sample_channel_axes(tmp_path):
    _write_clip(tmp_path, adc_ndim=3)

    with pytest.raises(ValueError, match=r"adc must have shape"):
        BenchmarkClip.load(tmp_path)


def test_simulated_adc_cache_invalidates_when_sensor_json_changes(tmp_path):
    adc = _write_clip(tmp_path)
    clip = BenchmarkClip.load(tmp_path)
    config = BenchmarkRunConfig(out_dir=tmp_path / "out")
    cache_path = config.out_dir / "simulated_adc_sim.npz"
    _write_simulated_adc_cache(
        cache_path,
        adc,
        clip,
        config,
        mobility_mode="human_only_po",
    )
    with np.load(cache_path, allow_pickle=False) as data:
        assert _simulated_adc_cache_matches(
            data,
            clip,
            config,
            mobility_mode="human_only_po",
        )

    sensor = json.loads((tmp_path / "sensor.json").read_text(encoding="utf-8"))
    sensor["fmcw"]["carrier_frequency_hz"] = 61.0e9
    (tmp_path / "sensor.json").write_text(json.dumps(sensor), encoding="utf-8")
    changed_clip = BenchmarkClip.load(tmp_path)

    with np.load(cache_path, allow_pickle=False) as data:
        assert not _simulated_adc_cache_matches(
            data,
            changed_clip,
            config,
            mobility_mode="human_only_po",
        )


def test_sensor_cache_payload_distinguishes_tdm_from_simultaneous(tmp_path):
    _write_clip(tmp_path)
    clip = BenchmarkClip.load(tmp_path)
    tdm_payload = _sensor_config_cache_payload(clip)
    simultaneous_sensor = replace(
        clip.sensor_config,
        fmcw=replace(clip.sensor_config.fmcw, tdm_enabled=False),
    )
    simultaneous_payload = _sensor_config_cache_payload(
        SimpleNamespace(sensor_config=simultaneous_sensor)
    )

    assert tdm_payload["fmcw"]["tdm_enabled"] is True
    assert simultaneous_payload["fmcw"]["tdm_enabled"] is False
    assert tdm_payload != simultaneous_payload


def test_simulated_adc_cache_invalidates_when_amass_sequence_changes(tmp_path):
    adc = _write_clip(tmp_path)
    clip = BenchmarkClip.load(tmp_path)
    config = BenchmarkRunConfig(out_dir=tmp_path / "out")
    cache_path = config.out_dir / "simulated_adc_sim.npz"
    _write_simulated_adc_cache(
        cache_path,
        adc,
        clip,
        config,
        mobility_mode="human_only_po",
    )

    motion_path = tmp_path / "amass_sequence.npz"
    with np.load(motion_path, allow_pickle=False) as motion:
        arrays = {name: np.array(motion[name], copy=True) for name in motion.files}
    arrays["trans"] = np.asarray(arrays["trans"], dtype=np.float32) + 0.1
    np.savez(motion_path, **arrays)
    changed_clip = BenchmarkClip.load(tmp_path)

    with np.load(cache_path, allow_pickle=False) as data:
        assert not _simulated_adc_cache_matches(
            data,
            changed_clip,
            config,
            mobility_mode="human_only_po",
        )


def test_simulated_adc_cache_invalidates_when_tdm_timing_policy_changes(tmp_path):
    adc = _write_clip(tmp_path)
    clip = BenchmarkClip.load(tmp_path)
    cached_config = BenchmarkRunConfig(out_dir=tmp_path / "out")
    cache_path = cached_config.out_dir / "simulated_adc_sim.npz"
    _write_simulated_adc_cache(
        cache_path,
        adc,
        clip,
        cached_config,
        mobility_mode="human_only_po",
    )

    simultaneous_config = BenchmarkRunConfig(
        out_dir=tmp_path / "out",
        ignore_tdm_timing=True,
    )
    with np.load(cache_path, allow_pickle=False) as data:
        assert not _simulated_adc_cache_matches(
            data,
            clip,
            simultaneous_config,
            mobility_mode="human_only_po",
        )


def test_simulated_adc_cache_invalidates_when_scene_file_changes(tmp_path):
    adc = _write_clip(tmp_path)
    scene_path = tmp_path / "scene.xml"
    scene_path.write_text("<scene version='1'/>", encoding="utf-8")
    clip = BenchmarkClip.load(tmp_path)
    config = BenchmarkRunConfig(
        out_dir=tmp_path / "out",
        scene_path=str(scene_path),
    )
    cache_path = config.out_dir / "simulated_adc_sim.npz"
    _write_simulated_adc_cache(
        cache_path,
        adc,
        clip,
        config,
        mobility_mode="hybrid_static_env_po",
    )
    with np.load(cache_path, allow_pickle=False) as data:
        assert _simulated_adc_cache_matches(
            data,
            clip,
            config,
            mobility_mode="hybrid_static_env_po",
        )

    scene_path.write_text("<scene version='2'/>", encoding="utf-8")

    with np.load(cache_path, allow_pickle=False) as data:
        assert not _simulated_adc_cache_matches(
            data,
            clip,
            config,
            mobility_mode="hybrid_static_env_po",
        )


def test_simulated_adc_cache_invalidates_when_referenced_scene_asset_changes(
    tmp_path,
):
    adc = _write_clip(tmp_path / "clip")
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    mesh_path = scene_dir / "room.obj"
    mesh_path.write_text("v 0 0 0\n", encoding="utf-8")
    include_path = scene_dir / "geometry.xml"
    include_path.write_text(
        "<scene><shape type='obj'><string name='filename' "
        "value='room.obj'/></shape></scene>",
        encoding="utf-8",
    )
    scene_path = scene_dir / "scene.xml"
    scene_path.write_text(
        "<scene><include filename='geometry.xml'/></scene>",
        encoding="utf-8",
    )
    clip = BenchmarkClip.load(tmp_path / "clip")
    config = BenchmarkRunConfig(
        out_dir=tmp_path / "out",
        scene_path=str(scene_path),
    )
    cache_path = config.out_dir / "simulated_adc_sim.npz"
    _write_simulated_adc_cache(
        cache_path,
        adc,
        clip,
        config,
        mobility_mode="hybrid_static_env_po",
    )

    mesh_path.write_text("v 1 0 0\n", encoding="utf-8")

    with np.load(cache_path, allow_pickle=False) as data:
        assert not _simulated_adc_cache_matches(
            data,
            clip,
            config,
            mobility_mode="hybrid_static_env_po",
        )


def test_simulated_adc_cache_invalidates_when_environment_scene_changes(tmp_path):
    clip_dir = tmp_path / "clip"
    adc = _write_clip(clip_dir)
    (clip_dir / "scene_a.xml").write_text("<scene/>", encoding="utf-8")
    (clip_dir / "scene_b.xml").write_text("<scene version='2'/>", encoding="utf-8")
    environment_path = clip_dir / "environment.json"
    environment_path.write_text(
        json.dumps({"scene_path": "scene_a.xml"}),
        encoding="utf-8",
    )
    clip = BenchmarkClip.load(clip_dir)
    config = BenchmarkRunConfig(out_dir=tmp_path / "out")
    cache_path = config.out_dir / "simulated_adc_sim.npz"
    _write_simulated_adc_cache(
        cache_path,
        adc,
        clip,
        config,
        mobility_mode="hybrid_static_env_po",
    )

    environment_path.write_text(
        json.dumps({"scene_path": "scene_b.xml"}),
        encoding="utf-8",
    )
    changed_clip = BenchmarkClip.load(clip_dir)

    with np.load(cache_path, allow_pickle=False) as data:
        assert not _simulated_adc_cache_matches(
            data,
            changed_clip,
            config,
            mobility_mode="hybrid_static_env_po",
        )


def test_simulated_adc_cache_invalidates_when_smpl_model_changes(tmp_path):
    adc = _write_clip(tmp_path / "clip")
    model_file = tmp_path / "models/smpl/SMPL_NEUTRAL.pkl"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"model version one")
    clip = BenchmarkClip.load(tmp_path / "clip")
    config = BenchmarkRunConfig(
        out_dir=tmp_path / "out",
        smpl_model_dir=tmp_path / "models",
    )
    cache_path = config.out_dir / "simulated_adc_sim.npz"
    _write_simulated_adc_cache(
        cache_path,
        adc,
        clip,
        config,
        mobility_mode="human_only_po",
    )

    model_file.write_bytes(b"model version two")

    with np.load(cache_path, allow_pickle=False) as data:
        assert not _simulated_adc_cache_matches(
            data,
            clip,
            config,
            mobility_mode="human_only_po",
        )


def test_ti_digitized_pattern_fingerprint_tracks_external_asset(
    tmp_path,
    monkeypatch,
):
    asset = tmp_path / "ti_digitized_patterns.npz"
    asset.write_bytes(b"pattern version one")
    monkeypatch.setenv("MMWAVE_TI_PATTERN_NPZ", str(asset))
    clip = SimpleNamespace(
        sensor_config=SimpleNamespace(
            board_model="IWR6843ISK",
            pattern_mode="digitized",
        )
    )

    first = _ti_digitized_pattern_fingerprint(clip)
    asset.write_bytes(b"pattern version two")

    assert _ti_digitized_pattern_fingerprint(clip) != first


def test_unreadable_simulated_adc_cache_is_rebuilt(tmp_path, monkeypatch):
    import validation.runner as runner

    adc = _write_clip(tmp_path / "clip")
    clip = BenchmarkClip.load(tmp_path / "clip")
    config = BenchmarkRunConfig(out_dir=tmp_path / "out")
    cache_path = config.out_dir / "simulated_adc_sim.npz"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(b"interrupted cache write")
    calls = []

    def fake_simulate(*_args, **_kwargs):
        calls.append(True)
        return adc

    monkeypatch.setattr(runner, "_simulate_adc", fake_simulate)

    rebuilt_adc, cache = runner._load_or_simulate_cached_adc(
        clip,
        config,
        label="sim",
        mobility_mode="human_only_po",
    )

    assert len(calls) == 1
    assert cache["status"] == "miss_rebuilt"
    assert np.array_equal(rebuilt_adc, adc)
    with np.load(cache_path, allow_pickle=False) as rebuilt:
        assert np.array_equal(rebuilt["adc"], adc)


def test_simulated_adc_cache_rejects_nonfinite_adc(tmp_path):
    adc = _write_clip(tmp_path / "clip")
    clip = BenchmarkClip.load(tmp_path / "clip")
    config = BenchmarkRunConfig(out_dir=tmp_path / "out")
    cache_path = config.out_dir / "simulated_adc_sim.npz"
    corrupted = np.array(adc, copy=True)
    corrupted.reshape(-1)[0] = np.nan + 1j * np.nan
    _write_simulated_adc_cache(
        cache_path,
        corrupted,
        clip,
        config,
        mobility_mode="human_only_po",
    )

    with np.load(cache_path, allow_pickle=False) as data:
        assert not _simulated_adc_cache_matches(
            data,
            clip,
            config,
            mobility_mode="human_only_po",
        )


def test_simulated_adc_cache_invalidates_when_hybrid_calibration_changes(tmp_path):
    adc = _write_clip(tmp_path)
    clip = BenchmarkClip.load(tmp_path)
    calibrated = BenchmarkRunConfig(out_dir=tmp_path / "out")
    cache_path = calibrated.out_dir / "simulated_adc_sim.npz"
    _write_simulated_adc_cache(
        cache_path,
        adc,
        clip,
        calibrated,
        mobility_mode="hybrid_static_env_po",
    )

    uncalibrated = BenchmarkRunConfig(
        out_dir=tmp_path / "out",
        hybrid_po_calibration_mode="none",
    )
    with np.load(cache_path, allow_pickle=False) as data:
        assert not _simulated_adc_cache_matches(
            data,
            clip,
            uncalibrated,
            mobility_mode="hybrid_static_env_po",
        )


def test_bundle_integrity_checker_reports_standard_bundle(tmp_path, capsys):
    adc = _write_clip(tmp_path)

    code = bundle_integrity_main([str(tmp_path)])
    captured = capsys.readouterr()

    assert code == 0
    assert "Bundle integrity: OK" in captured.out
    assert f"ADC shape: {list(adc.shape)}" in captured.out


def test_bundle_integrity_checker_json_report_contains_shapes(tmp_path, capsys):
    adc = _write_clip(tmp_path)

    code = bundle_integrity_main([str(tmp_path), "--json"])
    captured = capsys.readouterr()
    report = json.loads(captured.out)

    assert code == 0
    assert report["valid"] is True
    assert report["primary_adc_artifact"]["shape"] == list(adc.shape)
    assert report["frames"]["count"] == 2
    assert report["required_files"]["sensor.json"] is True


def test_bundle_integrity_checker_returns_error_for_bad_bundle(tmp_path, capsys):
    _write_clip(tmp_path, adc_ndim=3)

    code = bundle_integrity_main([str(tmp_path)])
    captured = capsys.readouterr()

    assert code == 1
    assert "adc must have shape" in captured.err


def test_bundle_integrity_checker_validates_amass_sequence_shape(tmp_path):
    _write_clip(tmp_path)

    report = check_bundle_integrity(tmp_path)

    assert report["amass_sequence"]["present"] is True
    assert report["amass_sequence"]["poses_shape"] == [3, 72]


def test_bundle_frame_motion_uses_explicit_frame_timestamps():
    class FakeMotion:
        faces = np.asarray([[0, 1, 2]], dtype=np.uint32)
        vertex_count = 3

        @staticmethod
        def vertices_at(time):
            return np.full((3, 3), float(time), dtype=np.float32)

    motion = _BundleFrameMeshSequence(
        FakeMotion(),
        frame_times_s=np.asarray([1.5, 3.0]),
        frame_period_s=0.1,
    )

    assert motion.vertices_at(0.025)[0, 0] == pytest.approx(1.525)
    assert motion.vertices_at(0.1)[0, 0] == pytest.approx(3.0)
    assert motion.vertices_at(0.125)[0, 0] == pytest.approx(3.025)


def test_bundle_human_placement_is_applied_to_parameterized_motion():
    class FakeMotion:
        times = np.asarray([0.0, 0.1])
        faces = np.asarray([[0, 1, 2]], dtype=np.uint32)
        vertex_count = 3

        @staticmethod
        def vertices_at(time):
            return np.asarray(
                [[0.0 + time, 0.0, 0.0], [2.0 + time, 0.0, 0.0], [0.0 + time, 2.0, 0.0]],
                dtype=np.float32,
            )

    motion = _PlacedBundleMotionSequence(
        FakeMotion(),
        position=(1.5, -0.25, 0.4),
        yaw_deg=90.0,
    )

    initial = motion.vertices_at(0.0)
    bounds_center = 0.5 * (np.min(initial, axis=0) + np.max(initial, axis=0))
    assert bounds_center == pytest.approx([1.5, -0.25, 0.4])
    assert motion.vertices_at(0.1)[0] - initial[0] == pytest.approx(
        [0.0, 0.1, 0.0]
    )


def test_validation_rejects_motion_that_ends_before_final_physical_chirp(tmp_path):
    _write_clip(tmp_path)
    motion_path = tmp_path / "amass_sequence.npz"
    np.savez(
        motion_path,
        poses=np.zeros((2, 72), dtype=np.float32),
        trans=np.zeros((2, 3), dtype=np.float32),
        betas=np.zeros(10, dtype=np.float32),
        faces=np.asarray([[0, 1, 2]], dtype=np.uint32),
        times=np.asarray([0.0, 0.1]),
    )
    clip = BenchmarkClip.load(tmp_path)

    with pytest.raises(ValueError, match="does not cover the full radar acquisition"):
        _simulation_mesh_sequence(
            clip,
            BenchmarkRunConfig(out_dir=tmp_path / "out", smpl_model_dir=tmp_path),
        )


def test_smpl_loaders_reject_pickled_object_arrays(tmp_path):
    _write_clip(tmp_path)
    smpl_path = tmp_path / "amass_sequence.npz"
    np.savez(
        smpl_path,
        poses=np.zeros((2, 72), dtype=object),
        trans=np.zeros((2, 3), dtype=np.float32),
        betas=np.zeros((10,), dtype=np.float32),
        times=np.asarray([0.0, 0.1], dtype=np.float64),
        faces=np.asarray([[0, 1, 2]], dtype=np.uint32),
    )

    with pytest.raises(ValueError, match="allow_pickle=False"):
        check_bundle_integrity(tmp_path)
    with pytest.raises(ValueError, match="allow_pickle=False"):
        _load_amass_motion_sequence(
            smpl_path,
            BenchmarkRunConfig(smpl_model_dir=tmp_path / "smpl_models"),
        )


def test_aw2243_cascade_sensor_config_from_profile(tmp_path):
    sensor_path = tmp_path / "sensor.json"
    sensor_path.write_text(
        json.dumps(
            {
                "board_model": "MMWCAS-RF-EVM",
                "fmcw_profile": "rtpose_high_resolution_mimo",
                "pattern_mode": "none",
            }
        ),
        encoding="utf-8",
    )

    sensor_config = BenchmarkSensorConfig.from_json(sensor_path)
    sensor = sensor_config.to_sensor()

    assert sensor.hardware.num_tx == 12
    assert sensor.hardware.num_rx == 16
    assert sensor.hardware.num_virtual_channels == 192
    assert sensor.fmcw.num_adc_samples == 256
    assert sensor.fmcw.num_chirps_per_frame == 64
    assert sensor.fmcw.tdm_enabled is True


def test_tdm_profile_uses_per_transmitter_slow_time_count(tmp_path):
    sensor_path = tmp_path / "sensor.json"
    sensor_path.write_text(
        json.dumps(
            {
                "board_model": "MMWCAS-RF-EVM",
                "fmcw_profile": "rtpose_high_resolution_mimo",
                "pattern_mode": "none",
                "metadata": {"tdm_virtual_adc": True},
            }
        ),
        encoding="utf-8",
    )

    sensor = BenchmarkSensorConfig.from_json(sensor_path).to_sensor()

    assert sensor.fmcw.tdm_enabled is True
    assert sensor.fmcw.num_chirps_per_frame == 64
    assert sensor.fmcw.slow_time_interval == pytest.approx(12 * 65.0e-6)


@pytest.mark.parametrize(
    ("profile", "chirp_repetition_time_s", "published_frame_period_s"),
    (
        ("ti_mimo_srr", 45.0e-6, 69.0e-3),
        ("ti_mimo_mrr", 27.0e-6, 4.4e-3),
    ),
)
def test_ti_cascade_mimo_profiles_use_per_tx_slow_time_count(
    tmp_path,
    profile,
    chirp_repetition_time_s,
    published_frame_period_s,
):
    sensor_path = tmp_path / "sensor.json"
    sensor_path.write_text(
        json.dumps(
            {
                "board_model": "MMWCAS-RF-EVM",
                "fmcw_profile": profile,
                "pattern_mode": "none",
            }
        ),
        encoding="utf-8",
    )

    sensor = BenchmarkSensorConfig.from_json(sensor_path).to_sensor()

    assert sensor.fmcw.tdm_enabled is True
    assert sensor.fmcw.num_chirps_per_frame == 128
    assert sensor.fmcw.slow_time_interval == pytest.approx(
        12 * chirp_repetition_time_s
    )
    assert sensor.fmcw.frame_period == pytest.approx(
        128 * 12 * chirp_repetition_time_s
    )
    assert sensor.fmcw.frame_period > published_frame_period_s
    assert sensor.fmcw.chirp_time(1, 0) >= sensor.fmcw.frame_duration


def test_ti_cascade_txbf_profile_remains_simultaneous(tmp_path):
    sensor_path = tmp_path / "sensor.json"
    sensor_path.write_text(
        json.dumps(
            {
                "board_model": "MMWCAS-RF-EVM",
                "fmcw_profile": "ti_txbf",
                "pattern_mode": "none",
            }
        ),
        encoding="utf-8",
    )

    sensor = BenchmarkSensorConfig.from_json(sensor_path).to_sensor()

    assert sensor.fmcw.tdm_enabled is False
    assert sensor.fmcw.num_chirps_per_frame == 128
    assert sensor.fmcw.slow_time_interval == pytest.approx(27.0e-6)


def test_bundle_sensor_restores_custom_cosine_beamwidth(tmp_path):
    sensor_path = tmp_path / "sensor.json"
    sensor_path.write_text(
        json.dumps(
            {
                "board_model": "MMWCAS-RF-EVM",
                "fmcw_profile": "rtpose_high_resolution_mimo",
                "pattern_mode": "cosine",
                "cosine_3db_beamwidth_deg": 44.0,
            }
        ),
        encoding="utf-8",
    )

    sensor_config = BenchmarkSensorConfig.from_json(sensor_path)
    sensor = sensor_config.to_sensor()

    assert sensor_config.cosine_3db_beamwidth_deg == 44.0
    assert sensor.hardware.antenna_pattern.half_power_angle_deg == 22.0


def test_bundle_sensor_restores_selected_tx_and_rx_elements(tmp_path):
    sensor_path = tmp_path / "sensor.json"
    sensor_path.write_text(
        json.dumps(
            {
                "board_model": "IWR6843ISK",
                "pattern_mode": "none",
                "tx_indices_zero_based": [2],
                "rx_indices_zero_based": [1, 3],
                "fmcw": {
                    "carrier_frequency_hz": 60.0e9,
                    "slope_hz_per_s": 20.0e12,
                    "chirp_duration_s": 60.0e-6,
                    "chirp_repetition_time_s": 80.0e-6,
                    "sampling_frequency_hz": 2.0e6,
                    "num_adc_samples": 16,
                    "num_chirps_per_frame": 4,
                    "frame_period_s": 0.1,
                    "num_tx": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    sensor_config = BenchmarkSensorConfig.from_json(sensor_path)
    sensor = sensor_config.to_sensor()

    assert sensor_config.tx_indices_zero_based == (2,)
    assert sensor_config.rx_indices_zero_based == (1, 3)
    assert sensor.hardware.num_tx == 1
    assert sensor.hardware.num_rx == 2
    assert sensor.hardware.num_virtual_channels == 2


def test_bundle_sensor_applies_declared_rx_major_channel_order(tmp_path):
    sensor_path = tmp_path / "sensor.json"
    sensor_path.write_text(
        json.dumps(
            {
                "board_model": "MMWCAS-RF-EVM",
                "fmcw_profile": "rtpose_high_resolution_mimo",
                "pattern_mode": "none",
                "virtual_channel_order": "rx_major",
            }
        ),
        encoding="utf-8",
    )

    sensor_config = BenchmarkSensorConfig.from_json(sensor_path)
    sensor = sensor_config.to_sensor()

    assert sensor_config.virtual_channel_order == "rx_major"
    assert sensor.hardware.virtual_channel_order == "rx_major"
    assert np.array_equal(
        sensor.hardware.virtual_tx_indices(),
        np.tile(np.arange(sensor.hardware.num_tx), sensor.hardware.num_rx),
    )
    assert np.array_equal(
        sensor.hardware.virtual_rx_indices(),
        np.repeat(np.arange(sensor.hardware.num_rx), sensor.hardware.num_tx),
    )


def test_bundle_sensor_rejects_unknown_channel_order(tmp_path):
    sensor_path = tmp_path / "sensor.json"
    sensor_path.write_text(
        json.dumps(
            {
                "board_model": "MMWCAS-RF-EVM",
                "fmcw_profile": "rtpose_high_resolution_mimo",
                "virtual_channel_order": "vendor_specific",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="virtual_channel_order"):
        BenchmarkSensorConfig.from_json(sensor_path)


def test_background_subtraction_and_dsp_metrics_helpers(tmp_path):
    _write_clip(tmp_path)
    clip = BenchmarkClip.load(tmp_path)
    centered = background_subtract_adc(clip.real_adc)
    rd_power = np.ones((2, 5, 8), dtype=float)
    rd_power[:, 2, :] = 10.0

    sensor = clip.sensor()
    motion = _simulation_mesh_sequence(
        clip,
        BenchmarkRunConfig(out_dir=tmp_path / "out", smpl_model_dir=tmp_path),
    )
    summary = mesh_range_summary(motion, sensor, clip.motion_times_s)
    suppressed = suppress_zero_doppler(rd_power, width_bins=0)

    assert centered.shape == clip.real_adc.shape
    assert normalized_correlation(rd_power, rd_power) == pytest.approx(1.0)
    assert zero_doppler_power_fraction(rd_power, width_bins=0) > 0.0
    assert np.array_equal(suppressed, rd_power)
    assert np.all(summary.min_m > 1.9)
    assert np.all(summary.max_m < 2.2)


def test_validation_cli_smoke_writes_metrics_maps_and_plots(tmp_path):
    clip_dir = tmp_path / "clip"
    adc = _write_clip(clip_dir)
    sim_path = tmp_path / "sim_adc.npz"
    np.savez(sim_path, adc=adc)
    out_dir = tmp_path / "out"

    code = validation_main([
        "--clip",
        str(clip_dir),
        "--suite",
        "smoke",
        "--out",
        str(out_dir),
        "--simulated-adc",
        str(sim_path),
        "--clutter-removal",
        "mean",
    ])

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert code == 0
    assert metrics["adc_shape"] == [2, 4, 16, 12]
    assert metrics["sensor_metadata"]["dataset"] == "RT-Pose"
    assert metrics["sensor_metadata"]["source_adc_axes"] == [
        "adc_sample",
        "chirp_loop",
        "rx",
        "tx",
    ]
    assert metrics["frames"][0]["pose_frame_id"] == "000001"
    assert metrics["frames"][0]["metadata"]["subject"] == "S01"
    assert metrics["clutter_removal"] == "mean"
    assert "mean_clutter_removal" not in metrics
    assert metrics["range_time"]["normalized_correlation"] == pytest.approx(1.0)
    assert metrics["angle_fft"]["selection_mode"] == (
        "single_range_time_peak_bin_shared_sim_closest_to_real"
    )
    assert metrics["angle_fft"]["real_selected_range_bin"] == (
        metrics["angle_fft"]["simulation_selected_range_bin"]
    )
    assert (out_dir / "maps.npz").exists()
    assert (out_dir / "range_time_real_vs_sim.png").exists()
    assert (out_dir / "range_profile_real_vs_sim.png").exists()
    assert (out_dir / "range_doppler_real_vs_sim.png").exists()
    assert (out_dir / "angle_fft_real_vs_sim.png").exists()
    maps = np.load(out_dir / "maps.npz")
    assert maps["real_angle_fft_power"].shape == (2, 64, 128)
    assert np.array_equal(
        maps["real_angle_fft_range_bin"],
        maps["sim_angle_fft_range_bin"],
    )


def test_validation_cli_subarray_selects_tdm_tx_rx_channels(tmp_path):
    clip_dir = tmp_path / "clip"
    adc = _write_clip(clip_dir)
    sensor_path = clip_dir / "sensor.json"
    sensor = json.loads(sensor_path.read_text(encoding="utf-8"))
    sensor["metadata"]["tx_to_enable"] = [3, 2, 1]
    sensor_path.write_text(json.dumps(sensor), encoding="utf-8")
    sim_path = tmp_path / "sim_adc.npz"
    np.savez(sim_path, adc=adc)
    out_dir = tmp_path / "out"

    code = validation_main([
        "--clip",
        str(clip_dir),
        "--suite",
        "smoke",
        "--out",
        str(out_dir),
        "--simulated-adc",
        str(sim_path),
        "--subarray-tx",
        "1,3",
        "--subarray-rx",
        "2-3",
        "--clutter-removal",
        "none",
        "--no-plots",
    ])

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    subarray = metrics["subarray"]
    cube = np.load(out_dir / "subarray_adc.npz")

    assert code == 0
    assert metrics["source_adc_shape"] == [2, 4, 16, 12]
    assert metrics["adc_shape"] == [2, 4, 16, 4]
    assert subarray["enabled"] is True
    assert subarray["channel_indices"] == [1, 2, 9, 10]
    assert subarray["channel_tx_indices_1based"] == [3, 3, 1, 1]
    assert subarray["channel_rx_indices_1based"] == [2, 3, 2, 3]
    assert cube["real_adc"].shape == (2, 4, 16, 4)
    assert cube["sim_adc"].shape == (2, 4, 16, 4)
    assert np.array_equal(cube["real_adc"], adc[..., [1, 2, 9, 10]])


@pytest.mark.parametrize(
    ("virtual_order", "channel_tx", "channel_rx", "expected_outputs"),
    (
        (
            "rx_major",
            [0, 1, 0, 1, 0, 1],
            [0, 0, 1, 1, 2, 2],
            {0: [(1, 0), (3, 1), (5, 2)], 1: [(0, 0), (2, 1), (4, 2)]},
        ),
        (
            "tx_major",
            [1, 0, 1, 0, 1, 0],
            [2, 0, 0, 2, 1, 1],
            {0: [(0, 0), (2, 1), (4, 2)], 1: [(1, 0), (3, 1), (5, 2)]},
        ),
    ),
)
def test_tdm_slot_mapping_uses_actual_virtual_channel_pairs(
    virtual_order,
    channel_tx,
    channel_rx,
    expected_outputs,
):
    hardware = RadarHardware(
        name="ordered-validation-hardware",
        tx_positions=np.zeros((2, 3)),
        rx_positions=np.zeros((3, 3)),
        virtual_channel_order=virtual_order,
        virtual_channel_tx_indices=np.asarray(channel_tx),
        virtual_channel_rx_indices=np.asarray(channel_rx),
    )
    fmcw = FMCWConfig(
        carrier_frequency=60.0e9,
        slope=20.0e12,
        chirp_duration=60.0e-6,
        chirp_repetition_time=80.0e-6,
        sampling_frequency=2.0e6,
        num_adc_samples=16,
        num_chirps_per_frame=4,
        frame_period=0.1,
        num_tx=2,
        tdm_enabled=True,
    )
    sensor = RadarSensor(
        name="ordered-validation-radar",
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        hardware=hardware,
        fmcw=fmcw,
    )
    clip = SimpleNamespace(
        sensor_config=SimpleNamespace(
            fmcw=fmcw,
            metadata={},
            virtual_channel_order=virtual_order,
        ),
        sensor=lambda: sensor,
    )
    actual_tx, actual_rx = _channel_tx_rx_indices_1based(
        clip,
        sensor,
        full_channels=6,
    )
    selection = _subarray_selection(
        clip,
        BenchmarkRunConfig(),
        full_channels=6,
    )

    np.testing.assert_array_equal(actual_tx, np.asarray(channel_tx) + 1)
    np.testing.assert_array_equal(actual_rx, np.asarray(channel_rx) + 1)
    assert _tdm_slot_rx_outputs(
        selection,
        tx_order=np.asarray([1, 0]),
        hardware=hardware,
    ) == expected_outputs


def test_ignore_tdm_timing_uses_one_simultaneous_selected_hardware_run(
    tmp_path,
    monkeypatch,
):
    import validation.runner as runner

    clip_dir = tmp_path / "clip"
    _write_clip(clip_dir)
    sensor_path = clip_dir / "sensor.json"
    sensor_data = json.loads(sensor_path.read_text(encoding="utf-8"))
    sensor_data["metadata"]["tx_to_enable"] = [3, 2, 1]
    sensor_path.write_text(json.dumps(sensor_data), encoding="utf-8")
    clip = BenchmarkClip.load(clip_dir)
    config = BenchmarkRunConfig(
        ignore_tdm_timing=True,
        subarray_tx_indices=(1, 3),
        subarray_rx_indices=(2, 3),
    )
    selection = _subarray_selection(clip, config, full_channels=12)
    sensor, output_order = _simultaneous_tdm_sensor(clip, selection)

    assert sensor.hardware.num_tx == 2
    assert sensor.hardware.num_rx == 2
    assert sensor.hardware.num_virtual_channels == 4
    assert sensor.fmcw.num_tx == 2
    assert sensor.fmcw.chirp_repetition_time == pytest.approx(3 * 80.0e-6)
    assert sensor.fmcw.tdm_enabled is False
    assert sensor.fmcw.slow_time_interval == pytest.approx(3 * 80.0e-6)
    assert output_order.tolist() == [2, 3, 0, 1]

    calls = []

    def fake_simultaneous(*args, **kwargs):
        calls.append((args, kwargs))
        return np.zeros((2, 4, 16, 4), dtype=np.complex64)

    def fail_slot_simulation(*_args, **_kwargs):
        raise AssertionError("per-slot TDM simulation must not run")

    monkeypatch.setattr(runner, "_simulate_simultaneous_tdm_adc", fake_simultaneous)
    monkeypatch.setattr(runner, "_simulate_tdm_virtual_adc", fail_slot_simulation)

    adc = _simulate_adc(clip, config, mobility_mode="human_only_po")

    assert adc.shape == (2, 4, 16, 4)
    assert len(calls) == 1


def test_validation_cli_passes_smpl_model_dir(tmp_path, monkeypatch):
    import validation.cli as validation_cli

    clip_dir = tmp_path / "clip"
    _write_clip(clip_dir)
    out_dir = tmp_path / "out"
    smpl_model_dir = tmp_path / "smpl_models"
    captured = {}

    def fake_run_validation_benchmark(clip, config):
        captured["clip_root"] = clip.root
        captured["smpl_model_dir"] = config.smpl_model_dir
        captured["simulate"] = config.simulate
        captured["mobility_modes"] = config.mobility_modes
        captured["diffuse_reflection"] = config.diffuse_reflection
        captured["refraction"] = config.refraction
        captured["human_specular_reflection"] = config.human_specular_reflection
        captured["hybrid_po_calibration_mode"] = config.hybrid_po_calibration_mode
        captured["ignore_tdm_timing"] = config.ignore_tdm_timing
        return {}

    monkeypatch.setattr(
        validation_cli,
        "run_validation_benchmark",
        fake_run_validation_benchmark,
    )

    code = validation_cli.main([
        "--clip",
        str(clip_dir),
        "--suite",
        "smoke",
        "--out",
        str(out_dir),
        "--mobility-mode",
        "hybrid_static_env_po",
        "rt_retrace",
        "--no-diffuse-reflection",
        "--refraction",
        "--human-specular-reflection",
        "--ignore-tdm-timing",
        "--smpl-model-dir",
        str(smpl_model_dir),
    ])

    assert code == 0
    assert captured["clip_root"] == clip_dir
    assert captured["smpl_model_dir"] == str(smpl_model_dir)
    assert captured["simulate"] is True
    assert captured["mobility_modes"] == ("hybrid_static_env_po", "rt_retrace")
    assert captured["diffuse_reflection"] is False
    assert captured["refraction"] is True
    assert captured["human_specular_reflection"] is True
    assert captured["hybrid_po_calibration_mode"] == "rt_first_pose"
    assert captured["ignore_tdm_timing"] is True


def test_validation_cli_uses_smpl_model_dir_environment(tmp_path, monkeypatch):
    import validation.cli as validation_cli

    clip_dir = tmp_path / "clip"
    _write_clip(clip_dir)
    environment_model_dir = tmp_path / "environment_models"
    captured = {}
    monkeypatch.setenv("MMWAVE_SMPL_MODEL_DIR", str(environment_model_dir))

    def fake_run_validation_benchmark(clip, config):
        captured["smpl_model_dir"] = config.smpl_model_dir
        return {}

    monkeypatch.setattr(
        validation_cli,
        "run_validation_benchmark",
        fake_run_validation_benchmark,
    )

    code = validation_cli.main([
        "--clip",
        str(clip_dir),
        "--suite",
        "smoke",
        "--out",
        str(tmp_path / "out"),
    ])

    assert code == 0
    assert captured["smpl_model_dir"] == str(environment_model_dir)


def test_smpl_model_dir_config_overrides_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "MMWAVE_SMPL_MODEL_DIR",
        str(tmp_path / "environment_models"),
    )
    explicit_model_dir = tmp_path / "explicit_models"

    config = BenchmarkRunConfig(smpl_model_dir=explicit_model_dir)

    assert config.smpl_model_dir == str(explicit_model_dir)


def test_smpl_model_dir_config_overrides_npz_value(tmp_path, monkeypatch):
    import mmWaveRadar.targets as targets

    smpl_npz = tmp_path / "amass_sequence.npz"
    np.savez(
        smpl_npz,
        poses=np.zeros((2, 72), dtype=np.float32),
        trans=np.zeros((2, 3), dtype=np.float32),
        betas=np.zeros(10, dtype=np.float32),
        times=np.asarray([7.2, 7.3]),
        bundle_times=np.asarray([-0.1, 0.0]),
        faces=np.zeros((1, 3), dtype=np.uint32),
        smpl_model_dir=np.asarray("/stale/smpl_models"),
    )
    captured = {}

    class FakeSMPLMotionSequence:
        def __init__(self, **kwargs):
            captured["smpl_model_dir"] = kwargs["smpl_model_dir"]
            captured["times"] = np.asarray(kwargs["times"])
            self.faces = kwargs["faces"]
            self.times = kwargs["times"]

    monkeypatch.setattr(targets, "AMASSSMPLMotionSequence", FakeSMPLMotionSequence)

    config = BenchmarkRunConfig(
        out_dir=tmp_path / "out",
        smpl_model_dir="/local/smpl_models",
    )

    _load_amass_motion_sequence(smpl_npz, config)

    assert captured["smpl_model_dir"] == "/local/smpl_models"
    assert captured["times"].tolist() == pytest.approx([-0.1, 0.0])


def test_validation_multi_mode_plot_outputs_name_and_label_every_mode(
    tmp_path,
    monkeypatch,
):
    from validation.runner import _write_plots

    import matplotlib.axes

    legend_labels = []
    original_legend = matplotlib.axes.Axes.legend

    def record_legend(axis, *args, **kwargs):
        legend_labels.append(axis.get_legend_handles_labels()[1])
        return original_legend(axis, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "legend", record_legend)

    out_dir = tmp_path / "plots"
    maps = {
        "real_range_time_power": np.ones((4, 8)),
        "po_range_time_power": np.ones((4, 8)) * 0.5,
        "rt_range_time_power": np.ones((4, 8)) * 0.25,
        "hybrid_static_env_po_range_time_power": np.ones((4, 8)) * 0.125,
        "sim_range_time_power": np.ones((4, 8)) * 0.5,
        "range_time_ranges_m": np.linspace(0.0, 1.0, 8),
        "real_range_doppler_power": np.ones((1, 4, 8)),
        "po_range_doppler_power": np.ones((1, 4, 8)) * 0.5,
        "rt_range_doppler_power": np.ones((1, 4, 8)) * 0.25,
        "hybrid_static_env_po_range_doppler_power": (
            np.ones((1, 4, 8)) * 0.125
        ),
        "sim_range_doppler_power": np.ones((1, 4, 8)) * 0.5,
        "range_doppler_ranges_m": np.linspace(0.0, 1.0, 8),
        "range_doppler_velocities_mps": np.linspace(-0.5, 0.5, 4),
        "real_angle_fft_power": np.ones((1, 8, 16)),
        "po_angle_fft_power": np.ones((1, 8, 16)) * 0.5,
        "rt_angle_fft_power": np.ones((1, 8, 16)) * 0.25,
        "hybrid_static_env_po_angle_fft_power": np.ones((1, 8, 16)) * 0.125,
        "sim_angle_fft_power": np.ones((1, 8, 16)) * 0.5,
        "angle_fft_u": np.linspace(-1.0, 1.0, 16),
        "angle_fft_v": np.linspace(-1.0, 1.0, 8),
    }

    _write_plots(
        out_dir,
        maps,
        simulation_modes=(
            ("po", "human_only_po"),
            ("rt", "rt_coherent_bank"),
            ("hybrid_static_env_po", "hybrid_static_env_po"),
        ),
    )

    suffix = (
        "real_vs_human_only_po_vs_rt_coherent_bank_"
        "vs_hybrid_static_env_po.png"
    )
    assert (out_dir / f"range_profile_{suffix}").exists()
    assert (out_dir / f"range_time_{suffix}").exists()
    assert (out_dir / f"range_doppler_{suffix}").exists()
    assert (out_dir / f"angle_fft_{suffix}").exists()
    assert not (out_dir / "range_profile_real_vs_sim.png").exists()
    assert not (out_dir / "range_time_real_vs_sim.png").exists()
    assert not (out_dir / "range_doppler_real_vs_sim.png").exists()
    assert not (out_dir / "angle_fft_real_vs_sim.png").exists()
    assert legend_labels == [[
        "Real",
        "human_only_po",
        "rt_coherent_bank",
        "hybrid_static_env_po",
    ]]
