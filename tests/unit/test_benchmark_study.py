# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Configuration studies, native validation boundaries, and resumable artifacts."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

from validation import simulation as _validation_simulation

import numpy as np
import pytest

from bundle_prepare.contract import Bundle
from validation import study
from validation.runner import (
    BenchmarkRunConfig,
    _benchmark_cube,
    _simulation_mesh_sequence,
    _simulated_adc_cache_metadata,
    _slice_clip,
    _validation_simulator,
)
from validation.settings import resolve_settings
from validation.study_metrics import aggregate_seed_powers, compare, scheduled_displacement
from validation.study_runtime import code_provenance, file_hash, stable_hash, write_json


def make_bundle(root, *, moving=False, measured=True):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    vertices = np.array(
        [[2.0, -0.025, -0.025], [2.0, 0.0, 0.025], [2.0, 0.025, -0.025]], np.float32
    )
    faces = np.array([[0, 1, 2]], np.uint32)
    descriptor = {
        "schema_version": 1,
        "profile": "hermes",
        "primary_adc": "primary",
        "adcs": {
            "primary": {
                "path": "radar_adc.npz",
                "origin": "measurement" if measured else "simulation",
            }
        },
    }
    if moving:
        np.savez(
            root / "mesh.npz",
            vertices=np.stack([vertices, vertices + [0.01, 0, 0]]),
            faces=faces,
            times=np.array([0.0, 0.05]),
        )
        descriptor["motion"] = {"evaluated_mesh": "mesh.npz"}
    else:
        np.savez(root / "mesh.npz", vertices=vertices, faces=faces)
        descriptor["target"] = {"mesh": "mesh.npz"}
    write_json(root / "bundle.json", descriptor)
    write_json(
        root / "sensor.json",
        {
            "board_model": "IWR6843AOPEVM",
            "pattern_mode": "none",
            "metadata": {"test_fixture": True},
            "fmcw": {
                "carrier_frequency_hz": 60e9,
                "carrier_reference": "first_adc_sample",
                "slope_hz_per_s": 1e12,
                "chirp_duration_s": 40e-6,
                "chirp_repetition_time_s": 50e-6,
                "sampling_frequency_hz": 1e6,
                "num_adc_samples": 16,
                "num_chirps_per_frame": 4,
                "frame_period_s": 0.02,
                "num_tx": 3,
                "tx_schedule": "tdm",
            },
        },
    )
    write_json(
        root / "frames.json",
        {
            "frames": [
                {"benchmark_index": i, "motion_time_s": i * 0.02, "radar_frame_id": str(i)}
                for i in range(2)
            ]
        },
    )
    adc = np.exp(2j * np.pi * np.arange(16) / 16)[None, None, :, None] * np.ones((2, 4, 1, 12))
    np.savez(root / "radar_adc.npz", adc=adc)
    return root


def make_study(root):
    bundle = make_bundle(root / "capture", moving=True)
    settings = {
        "compute_backend": "numpy",
        "compute_precision": "float32",
        "rt_samples_per_source": 100,
        "rt_max_paths_per_source": 100,
        "rt_max_depth": 1,
        "rt_diffuse_reflection": False,
        "rt_human_specular_reflection": True,
        "hybrid_calibration": "none",
        "po_integration_mode": "face_centroid",
        "po_visibility_samples_per_face": 1,
    }
    config = {
        "schema_version": 1,
        "name": "test-study",
        "output_root": "results",
        "seeds": [0],
        "settings": settings,
        "timing": {
            "warmups": 1,
            "repetitions": 1,
            "minimum_seconds": 0.0,
            "maximum_repetitions": 2,
        },
        "reference": {"settings": {"po_refresh_frames": 0}},
        "cases": [{"id": "walking", "scenario": "walking", "bundle": str(bundle)}],
        "experiments": [
            {
                "id": "cadence",
                "track": "numerical",
                "solvers": ["po"],
                "sweep": {"po_refresh_frames": [1, 2]},
            }
        ],
    }
    path = root / "study.json"
    write_json(path, config)
    return path


def test_visibility_study_compares_adaptive_and_uniform_at_each_sample_count():
    loaded = study.load_study(
        Path(__file__).resolve().parents[2] / "configs/benchmark_6843aop.json"
    )
    assert resolve_settings({})["po_adaptive_visibility_sampling"] is True
    assert loaded["reference"]["settings"]["po_adaptive_visibility_sampling"] is False
    jobs = study.expand_jobs(loaded, experiments={"visibility"})
    for case in loaded["cases"]:
        for solver in ("po", "hybrid_po"):
            pairs = {
                (
                    job["settings"]["po_visibility_samples_per_face"],
                    job["settings"]["po_adaptive_visibility_sampling"],
                )
                for job in jobs
                if job["case"]["id"] == case["id"] and job["solver"] == solver
            }
            assert pairs == {
                (count, adaptive) for count in (1, 4, 16) for adaptive in (False, True)
            }


def test_provenance_hashes_actual_solver_and_benchmark_sources():
    provenance = code_provenance()
    files = provenance["source_files"]
    root = Path(__file__).resolve().parents[2]
    for name in (
        "validation/study.py",
        "src/mmWaveRadar/simulation/simulator.py",
        "tools/bundle_prepare/contract.py",
    ):
        assert files[name] == file_hash(root / name)
    assert provenance["source_sha256"] == stable_hash(files)


def test_plan_expands_one_at_a_time_deduplicates_and_separates_platforms(tmp_path):
    path = make_study(tmp_path)
    loaded = study.load_study(path)
    jobs = study.expand_jobs(loaded)
    assert len(jobs) == 2
    assert jobs[0]["axis"] == "baseline"
    assert {j["settings"]["po_refresh_frames"] for j in jobs} == {1, 2}
    assert study.main(["--config", str(path), "--platform", "apple-m4-pro", "--plan"]) == 0
    assert study.main(["--config", str(path), "--platform", "rtx-5090", "--plan"]) == 0
    for platform in ("apple-m4-pro", "rtx-5090"):
        plan = json.loads((tmp_path / "results/test-study" / platform / "plan.json").read_text())
        assert plan["job_count"] == 2 and plan["platform_label"] == platform
    with pytest.raises(ValueError, match="identifier"):
        study.safe_name("../escape")


def test_every_solver_knob_affects_cache_fingerprint_and_reaches_native_api(tmp_path, monkeypatch):
    clip = Bundle.load(make_bundle(tmp_path / "capture"))
    base = BenchmarkRunConfig(hybrid_calibration="none")
    original = _simulated_adc_cache_metadata(clip, base, solver="hybrid_po")["input_fingerprint"]
    values = {
        "rt_samples_per_source": 123,
        "rt_seed": 4,
        "rt_max_depth": 4,
        "po_curvature_error_rad": 0.003,
        "po_refresh_frames": 4,
        "po_visibility_refresh_frames": 0,
        "rt_visibility_refresh_frames": 0,
        "po_visibility_samples_per_face": 2,
        "hybrid_coupling": False,
        "hybrid_max_reflectors": 2,
        "array_model": "subarray",
    }
    for key, value in values.items():
        config = replace(base, solver_settings={key: value})
        changed = _simulated_adc_cache_metadata(clip, config, solver="hybrid_po")[
            "input_fingerprint"
        ]
        assert changed != original, key
    import mmWaveRadar.simulation as simulation

    calls = []
    monkeypatch.setattr(simulation, "MmWaveRadarSimulator", lambda **kwargs: calls.append(kwargs))
    config = replace(base, solver_settings=values)
    _validation_simulator(config, "hybrid_po")
    assert calls[0]["rt_samples_per_source"] == 123
    assert calls[0]["po_config"].po_refresh_frames == 4
    assert calls[0]["hybrid_config"].hybrid_coupling is False
    with pytest.raises(ValueError, match="Unknown solver"):
        resolve_settings({"po_refresh_framez": 2})


def test_geometry_inputs_work_without_licensed_body_models(tmp_path):
    for moving in (False, True):
        clip = Bundle.load(make_bundle(tmp_path / str(moving), moving=moving))
        sequence = _simulation_mesh_sequence(clip, BenchmarkRunConfig())
        assert sequence.vertices_at(0.021).shape == (3, 3)
        displacement = scheduled_displacement(clip, BenchmarkRunConfig())
        assert displacement["po"]["scheduled_max_vertex_drift_mm"] >= 0


def test_frame_slicing_preserves_geometry_and_adc_provenance(tmp_path):
    for moving in (False, True):
        clip = Bundle.load(make_bundle(tmp_path / str(moving), moving=moving, measured=False))
        sliced = _slice_clip(clip, 1)
        assert sliced.num_frames == 1
        assert sliced.data_origin == "simulation"
        assert sliced.descriptor == clip.descriptor
        assert sliced.evaluated_motion_path == clip.evaluated_motion_path
        assert sliced.target_mesh_path == clip.target_mesh_path
        assert sliced.scene_path == clip.scene_path
        assert _simulation_mesh_sequence(sliced, BenchmarkRunConfig()).vertices_at(0.0).shape == (
            3,
            3,
        )


def test_displacement_tracks_independent_tdm_slots_and_cycle_interval(tmp_path):
    clip = Bundle.load(make_bundle(tmp_path / "capture", moving=True))
    for ignore_tdm in (False, True):
        config = BenchmarkRunConfig(ignore_tdm_timing=ignore_tdm)
        displacement = scheduled_displacement(clip, config)
        assert displacement["po"]["scheduled_max_vertex_drift_mm"] == pytest.approx(
            0.09, abs=0.0003
        )


def test_frozen_gain_scales_human_and_coupling_but_preserves_room():
    from mmWaveRadar.simulation import RadarCube

    components = {
        "human_po": np.ones((1, 1, 1, 1)),
        "human_env": np.full((1, 1, 1, 1), 2.0),
        "env_human": np.full((1, 1, 1, 1), 3.0),
    }
    cube = RadarCube(
        adc=np.full((1, 1, 1, 1), 16.0 + 0j),
        times=np.zeros((1, 1)),
        metadata=SimpleNamespace(),
        components=components,
    )
    config = BenchmarkRunConfig(hybrid_calibration="none", frozen_human_po_amplitude_gain=2.0)
    adjusted = _benchmark_cube(cube, config, "hybrid_po")
    np.testing.assert_array_equal(adjusted.adc, 22.0)  # Room=10, calibrated body/coupling=12.
    np.testing.assert_array_equal(cube.adc, 16.0)
    with pytest.raises(ValueError, match="requires hybrid_calibration"):
        BenchmarkRunConfig(frozen_human_po_amplitude_gain=2.0)


def test_diffuse_aggregation_averages_powers_not_complex_fields(tmp_path):
    files = []
    for seed, power in enumerate((1.0, 9.0)):
        path = tmp_path / f"{seed}.npz"
        np.savez(
            path,
            reference_range_time=np.full((2, 3), 5.0),
            candidate_range_time=np.full((2, 3), power),
        )
        files.append(path)
    result = aggregate_seed_powers(files)
    assert result["range_time"]["relative_l2"] == 0.0


def test_dsp_metrics_match_identical_adc_and_detect_range_shift(tmp_path):
    clip = Bundle.load(make_bundle(tmp_path / "capture"))
    config = BenchmarkRunConfig()
    same, _ = compare(clip, config, clip.real_adc, clip.real_adc, coherent=True)
    assert same["adc_relative_l2"] == 0.0
    assert same["range_time"]["relative_l2"] == pytest.approx(0.0, abs=1e-14)
    shifted = clip.real_adc * np.exp(2j * np.pi * np.arange(16) / 16)[None, None, :, None]
    different, _ = compare(clip, config, shifted, clip.real_adc)
    assert different["peak_range_error_m"] > 0


def test_reference_identity_is_common_across_platforms_and_all_parameters_are_fingerprinted(
    tmp_path,
):
    loaded = study.load_study(make_study(tmp_path))
    jobs = study.expand_jobs(loaded)
    inputs = study.bundle_fingerprint(jobs[0]["case"]["bundle"])
    source = {"source_sha256": "fixed"}
    first = study.reference_for(loaded, jobs[0], source, inputs)
    second = study.reference_for(loaded, jobs[1], source, inputs)
    assert first[1] == second[1]  # Candidate cadence overridden by the exact reference.
    changed = study.configuration_id(
        jobs[0], source, inputs, {**jobs[0]["settings"], "rt_seed": 123}, None
    )
    assert changed != study.configuration_id(jobs[0], source, inputs, jobs[0]["settings"], None)


def test_static_cases_do_not_sweep_motion_thresholds_and_calibration_is_rejected(tmp_path):
    loaded = study.load_study(make_study(tmp_path))
    loaded["cases"][0]["scenario"] = "plate"
    assert len(study.expand_jobs(loaded)) == 1
    loaded["settings"]["hybrid_calibration"] = "rt_first_pose"
    with pytest.raises(ValueError, match="Numerical sweeps"):
        study.expand_jobs(loaded)


def test_timing_executes_solver_until_native_duration_and_excludes_artifact_cache(
    tmp_path, monkeypatch
):
    from validation import runner, study_runtime

    path = make_bundle(tmp_path / "capture")
    calls = []

    def simulate(clip, config, *, solver):
        calls.append(solver)
        adc = np.ones_like(clip.real_adc)
        config.cube_observer(
            SimpleNamespace(
                adc=adc,
                metadata=SimpleNamespace(
                    path_counts=np.zeros(1),
                    po_calibration={},
                    runtime_profile_s={"validation_simulator_run": 0.6},
                    runtime_profile_counts={},
                    events=[],
                    incremental_recomputed_face_counts=np.array([2, 3]),
                    incremental_phase_updated_face_counts=np.array([7, 11]),
                ),
            )
        )
        return adc

    monkeypatch.setattr(_validation_simulation, "simulate_adc", simulate)
    monkeypatch.setattr(study_runtime, "synchronize", lambda: None)
    result = study_runtime.worker(
        {
            "bundle": str(path),
            "directory": str(tmp_path / "timing"),
            "phase": "timing",
            "solver": "po",
            "settings": resolve_settings({}),
            "timing": {
                "warmups": 1,
                "repetitions": 2,
                "minimum_seconds": 1.5,
                "maximum_repetitions": 10,
            },
        }
    )
    assert len(calls) == 4  # First-process warmup, then three fresh timed solver calls.
    assert result["timing_samples_seconds"] == [0.6, 0.6, 0.6]
    assert result["timing_duration_target_met"]
    assert result["po_recomputed_faces"] == 5
    assert result["po_phase_updated_faces"] == 18
    assert len(result["pipeline_timing_samples_seconds"]) == 3
    assert not (tmp_path / "timing/adc.npz").exists()


def test_resume_keeps_complete_timings_but_protocol_changes_run_again(tmp_path, monkeypatch):
    loaded = study.load_study(make_study(tmp_path))
    jobs = study.expand_jobs(loaded)
    inputs = {"walking": study.bundle_fingerprint(jobs[0]["case"]["bundle"])}
    provenance = {"source_sha256": "fixed"}
    phases = []

    def launch(request, **kwargs):
        phases.append(request["phase"])
        directory = Path(request["directory"])
        directory.mkdir(parents=True, exist_ok=True)
        if request["phase"] == "accuracy":
            np.savez(directory / "adc.npz", adc=Bundle.load(request["bundle"]).real_adc)
        return {
            "phase_center_checks": [],
            "timing_median_seconds": 0.1,
            "timing_iqr_seconds": 0.01,
        }

    monkeypatch.setattr(study, "launch_worker", launch)
    study.prepare_references(loaded, jobs, provenance, inputs)
    phases.clear()
    directory = tmp_path / "platform"
    study.run_jobs(loaded, jobs, provenance, inputs, directory, {"cpu": "test"})
    assert phases.count("timing") == 2
    study.run_jobs(loaded, jobs, provenance, inputs, directory, {"cpu": "test"}, resume=True)
    assert phases.count("timing") == 2
    loaded["timing"]["repetitions"] = 2
    study.run_jobs(loaded, jobs, provenance, inputs, directory, {"cpu": "test"}, resume=True)
    assert phases.count("timing") == 4
    assert len(list(directory.rglob("result.json"))) == 4
    from validation import study_report

    monkeypatch.setattr(study_report, "plot_summary", lambda *args: None)
    summaries = study.report(directory)
    assert len(summaries) == 4  # Separate protocol cohorts, each with one deterministic PO seed.


def test_frozen_calibration_rejects_evaluation_reuse_and_changed_held_out_data(tmp_path):
    loaded = study.load_study(make_study(tmp_path))
    case = loaded["cases"][0]
    case["calibration_bundle"] = case["bundle"]
    with pytest.raises(ValueError, match="separate held-out"):
        study.calibrate(loaded, [case], {"source_sha256": "fixed"})
    case["calibration_bundle"] = str(make_bundle(tmp_path / "held-out"))
    inputs = study.case_fingerprint(loaded, case)
    gain = {
        "source_sha256": "fixed",
        "evaluation_bundle_sha256": inputs["sha256"],
        "amplitude_gain": 2.0,
        "calibration_inputs": study.case_fingerprint(
            loaded, {**case, "bundle": case["calibration_bundle"], "scenario": "stationary_human"}
        ),
    }
    write_json(case["frozen_gain_file"], gain)
    job = {"case": case, "solver": "hybrid_po", "track": "physical"}
    assert study.frozen_gain(job, {"source_sha256": "fixed"}, inputs)[0] == 2.0
    write_json(Path(case["calibration_bundle"]) / "capture-provenance.json", {"new": "content"})
    with pytest.raises(ValueError, match="held-out"):
        study.frozen_gain(job, {"source_sha256": "fixed"}, inputs)


def test_walking_calibration_pools_first_frame_powers_over_slots_and_seeds(tmp_path, monkeypatch):
    loaded = study.load_study(make_study(tmp_path))
    loaded["seeds"] = [0, 1]
    case = loaded["cases"][0]
    case["calibration_bundle"] = str(make_bundle(tmp_path / "held-out"))
    requests = []

    def launch(request, **kwargs):
        requests.append(request)
        return {
            "calibration_powers": [
                {"rt_reference_rms_power": 4.0, "raw_po_reference_rms_power": 1.0, "weight": 1},
                {"rt_reference_rms_power": 16.0, "raw_po_reference_rms_power": 4.0, "weight": 3},
            ]
        }

    monkeypatch.setattr(study, "launch_worker", launch)
    study.calibrate(loaded, [case], {"source_sha256": "fixed"})
    gain = json.loads(Path(case["frozen_gain_file"]).read_text())
    assert gain["amplitude_gain"] == 2.0
    assert len(gain["powers"]) == 2
    assert gain["powers"][0]["rt_power"] == 13.0
    assert {r["settings"]["rt_seed"] for r in requests} == {0, 1}
    assert all(
        r["max_frames"] == 1 and r["settings"]["hybrid_calibration"] == "rt_first_pose"
        for r in requests
    )


@pytest.mark.parametrize("reference", [None, {"adc_sha256": "same-cpu-reference"}])
def test_cross_platform_difference_is_once_per_configuration_with_reference_identity(
    tmp_path, reference
):
    for platform, amplitude in (("local", 1.0), ("peer", 1.1)):
        for experiment in ("one", "two"):
            directory = tmp_path / platform / "runs" / experiment
            (directory / "accuracy").mkdir(parents=True)
            np.savez(directory / "accuracy/adc.npz", adc=np.ones((1, 1, 2, 1)) * amplitude)
            write_json(
                directory / "result.json", {"configuration_id": "shared", "reference": reference}
            )
    study.compare_platforms(tmp_path / "local", tmp_path / "peer")
    differences = json.loads((tmp_path / "local/cross_platform.json").read_text())
    assert len(differences) == 1
    assert differences[0]["adc_relative_l2"] == pytest.approx(0.1 / 1.1)
    assert differences[0]["identical_reference_artifact"] is (True if reference else None)


def test_geometry_controls_reach_direct_and_hybrid_solver_configs():
    from validation.settings import simulator_arguments

    settings = resolve_settings(
        {"po_refresh_on_area_fraction": 0.3, "po_refresh_on_normal_deg": 20.0}
    )
    arguments = simulator_arguments(settings)
    for config in (arguments["po_config"], arguments["hybrid_config"].po_config):
        assert config.po_refresh_on_area_fraction == 0.3
        assert config.po_refresh_on_normal_deg == 20.0
