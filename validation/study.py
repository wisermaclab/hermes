# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Configuration studies over the existing validation runner; no solver fork."""

import argparse
import csv
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import re

import numpy as np

from .configuration import BenchmarkRunConfig
from .channels import slice_clip as _slice_clip
from .uncertainty import seed_mean_interval
from .settings import resolve_settings
from .study_runtime import (
    code_provenance,
    file_hash,
    launch_worker,
    machine_info,
    platform_slug,
    stable_hash,
    worker,
    write_json,
)

TEMPORAL_KEYS = frozenset(
    {
        "rt_refresh_frames",
        "po_refresh_frames",
        "po_refresh_on_phase_error_rad",
        "rt_refresh_on_displacement",
        "rt_visibility_refresh_frames",
        "po_visibility_refresh_frames",
    }
)


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError(f"Expected an identifier of letters, numbers, '-' or '_': {value!r}")
    return value


def load_study(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1:
        raise ValueError("Benchmark configuration needs schema_version=1")
    data["name"] = safe_name(data["name"])
    data["config_path"] = str(path)

    def absolute(value):
        p = Path(value).expanduser()
        return str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())

    data["output_root"] = absolute(
        data.get("output_root", "../benchmarks/configuration_tradeoffs/results")
    )
    data["reference_root"] = (
        absolute(data["reference_root"])
        if data.get("reference_root")
        else str(Path(data["output_root"]) / data["name"] / "references")
    )
    data["settings"] = {
        "compute_backend": "auto",
        "compute_precision": "float32",
        "adc_compute_precision": "float32",
        "hybrid_calibration": "none",
        **data.get("settings", {}),
    }
    resolve_settings(data["settings"])
    timing = {
        "warmups": 1,
        "repetitions": 5,
        "minimum_seconds": 10.0,
        "maximum_repetitions": 1000,
        **data.get("timing", {}),
    }
    for key in ("warmups", "repetitions", "maximum_repetitions"):
        if isinstance(timing[key], bool) or not isinstance(timing[key], int) or timing[key] < 1:
            raise ValueError(f"timing.{key} must be a positive integer")
    if (
        not np.isfinite(timing["minimum_seconds"])
        or timing["minimum_seconds"] < 0
        or timing["maximum_repetitions"] < timing["repetitions"]
    ):
        raise ValueError("Invalid timing duration or repetition limit")
    data["timing"] = timing
    seen = set()
    for case in data["cases"]:
        case["id"] = safe_name(case["id"])
        if case["id"] in seen:
            raise ValueError("Case IDs must be unique")
        seen.add(case["id"])
        if case["scenario"] not in {"plate", "sphere", "stationary_human", "walking"}:
            raise ValueError("Unknown scenario")
        for key in ("bundle", "calibration_bundle", "frozen_gain_file", "smpl_model_dir"):
            if case.get(key):
                case[key] = absolute(case[key])
        if "bundle" not in case:
            raise ValueError("Every case needs a bundle path")
        measurement_gain = case.get("measurement_amplitude_gain", 1.0)
        if not np.isfinite(measurement_gain) or measurement_gain <= 0:
            raise ValueError("measurement_amplitude_gain must be finite and positive")
        if case.get("absolute_power_calibrated") and not case.get(
            "measurement_calibration_source"
        ):
            raise ValueError("Absolute power metrics require measurement_calibration_source")
        if case.get("frozen_gain_file") is None:
            case["frozen_gain_file"] = str(
                Path(data["reference_root"]) / "calibration" / f"{case['id']}.json"
            )
    seen = set()
    for experiment in data["experiments"]:
        safe_name(experiment["id"])
        if experiment["id"] in seen:
            raise ValueError("Experiment IDs must be unique")
        seen.add(experiment["id"])
        if experiment["track"] not in {"numerical", "physical"}:
            raise ValueError("Experiments need track=numerical or physical")
        if experiment.get("design", "one_at_a_time") not in {"one_at_a_time", "grid"}:
            raise ValueError("design must be one_at_a_time or grid")
        if not set(experiment.get("cases", [])) <= {c["id"] for c in data["cases"]}:
            raise ValueError("Experiment names an unknown case")
    seeds = data.get("seeds", [0, 1, 2, 3, 4])
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds)
    ):
        raise ValueError("seeds must be unique non-negative integers")
    data["seeds"] = seeds
    return data


def variants(experiment):
    yield "baseline", {}
    sweep = experiment.get("sweep", {})
    if any(not isinstance(v, list) or not v for v in sweep.values()):
        raise ValueError("Every sweep parameter needs a nonempty list")
    if experiment.get("design", "one_at_a_time") == "grid":
        for values in product(*sweep.values()):
            yield "grid", dict(zip(sweep, values))
    else:
        for key, values in sweep.items():
            for value in values:
                yield key, {key: value}


def expand_jobs(study, *, cases=None, experiments=None):
    jobs = []
    for case in study["cases"]:
        if not case.get("enabled", True) or cases and case["id"] not in cases:
            continue
        for experiment in study["experiments"]:
            if (
                not experiment.get("enabled", True)
                or experiments
                and experiment["id"] not in experiments
            ):
                continue
            if experiment.get("cases") and case["id"] not in experiment["cases"]:
                continue
            if experiment.get("scenarios") and case["scenario"] not in experiment["scenarios"]:
                continue
            base = {
                **study["settings"],
                **case.get("settings", {}),
                **experiment.get("settings", {}),
            }
            material = experiment.get("target_material", case.get("target_material", {}))
            for solver in experiment.get("solvers", ["po", "rt", "hybrid_po"]):
                if solver not in {"po", "rt", "hybrid_po"}:
                    raise ValueError(f"Unknown solver {solver}")
                unique = set()
                for axis, changes in variants(experiment):
                    if (
                        set(changes) & TEMPORAL_KEYS
                        and case["scenario"] != "walking"
                        and not case.get("has_motion", False)
                    ):
                        continue
                    settings = resolve_settings({**base, **changes})
                    # A frozen/default-free numerical comparison needs a fixed scale.
                    if experiment["track"] == "numerical" and (
                        settings["hybrid_calibration"] != "none"
                        or settings["po_calibration"] != "none"
                    ):
                        raise ValueError("Numerical sweeps require both calibrations='none'")
                    if stable_hash(settings) in unique:
                        continue
                    unique.add(stable_hash(settings))
                    seeds = [0] if solver == "po" else study["seeds"]
                    for seed in seeds:
                        seeded = {**settings, "rt_seed": seed}
                        jobs.append(
                            {
                                "case": case,
                                "experiment": experiment["id"],
                                "track": experiment["track"],
                                "solver": solver,
                                "seed": seed,
                                "axis": axis,
                                "changes": changes,
                                "settings": seeded,
                                "target_material": material,
                                "reference_settings": experiment.get("reference_settings", {}),
                            }
                        )
    if len(jobs) > study.get("maximum_jobs", 5000):
        raise ValueError("Plan exceeds maximum_jobs; restrict cases/experiments or trim the grid")
    return jobs


def bundle_fingerprint(path):
    root = Path(path)
    files = {
        str(p.relative_to(root)): file_hash(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and not any(
            part in {"diagnostics", "results", "__pycache__"} for part in p.relative_to(root).parts
        )
    }
    if not files or "bundle.json" not in files:
        raise ValueError(f"No measurement bundle at {root}")
    return {"sha256": stable_hash(files), "files": files}


def load_case(study, case):
    from bundle_prepare.contract import Bundle

    clip = _slice_clip(Bundle.load(case["bundle"]), case.get("max_frames"))
    expected = case.get("board_model", study.get("board_model", "IWR6843AOPEVM"))
    if expected is not None and clip.sensor_config.board_model != expected:
        raise ValueError(
            f"{case['id']}: expected {expected}, found {clip.sensor_config.board_model}"
        )
    if case["scenario"] == "walking" and clip.num_frames < 2:
        raise ValueError("Walking benchmarks need multi-frame measured ADC")
    return clip


def case_fingerprint(study, case):
    """Include external body models, antenna assets and scene dependencies."""
    import os
    import xml.etree.ElementTree as ET
    from .simulation import resolve_scene_path as _resolve_scene_path
    from .cache import (
        scene_xml_file_references as _scene_xml_file_references,
        smpl_model_fingerprint as _smpl_model_fingerprint,
        ti_digitized_pattern_fingerprint as _ti_digitized_pattern_fingerprint,
    )

    clip = load_case(study, case)
    config = BenchmarkRunConfig(smpl_model_dir=case.get("smpl_model_dir"))
    external = {
        "body_model": _smpl_model_fingerprint(clip, config),
        "antenna_pattern": _ti_digitized_pattern_fingerprint(clip),
    }
    if config.smpl_model_dir and Path(config.smpl_model_dir).is_file():
        external["body_model"] = file_hash(config.smpl_model_dir)
    scene = _resolve_scene_path(clip, config)
    if scene:
        root = Path(scene).resolve().parent
        pending, visited, assets = [Path(scene)], set(), {}
        while pending:
            path = pending.pop().resolve()
            if path in visited:
                continue
            visited.add(path)
            assets[os.path.relpath(path, root)] = file_hash(path)
            if path.suffix.lower() == ".xml":
                for value in _scene_xml_file_references(ET.parse(path).getroot()):
                    referenced = Path(value).expanduser()
                    pending.append(
                        referenced if referenced.is_absolute() else path.parent / referenced
                    )
        external["scene_assets"] = assets
    inputs = bundle_fingerprint(case["bundle"])
    return {
        **inputs,
        "external_assets": external,
        "sha256": stable_hash({"bundle": inputs["sha256"], "external_assets": external}),
    }


def request_for(job, directory, *, settings=None, phase="accuracy", frozen_gain=None):
    case = job["case"]
    return {
        "bundle": case["bundle"],
        "directory": str(directory),
        "solver": job["solver"],
        "settings": job["settings"] if settings is None else settings,
        "phase": phase,
        "target_material": job["target_material"],
        "max_frames": case.get("max_frames"),
        "frozen_gain": frozen_gain,
        "ignore_tdm_timing": case.get("ignore_tdm_timing", False),
        "smpl_model_dir": case.get("smpl_model_dir"),
        "selected_tx_indices": case.get("selected_tx_indices"),
        "selected_rx_indices": case.get("selected_rx_indices"),
    }


def configuration_id(job, provenance, inputs, settings, gain):
    case = job["case"]
    return stable_hash(
        {
            "source_sha256": provenance["source_sha256"],
            "inputs": inputs["sha256"],
            "solver": job["solver"],
            "settings": settings,
            "material": job["target_material"],
            "frozen_gain": gain,
            "max_frames": case.get("max_frames"),
            "ignore_tdm_timing": case.get("ignore_tdm_timing", False),
            "selected_tx_indices": case.get("selected_tx_indices"),
            "selected_rx_indices": case.get("selected_rx_indices"),
            "analysis": analysis_settings(case),
        }
    )


def analysis_settings(case):
    return {
        "background_subtract": case.get("background_subtract", False),
        "clutter_removal": case.get("clutter_removal", "none"),
        "measurement_amplitude_gain": case.get("measurement_amplitude_gain", 1.0),
        "absolute_power_calibrated": case.get("absolute_power_calibrated", False),
        "measurement_calibration_source": case.get("measurement_calibration_source"),
    }


def reference_for(study, job, provenance, inputs):
    settings = resolve_settings(
        {
            **job["settings"],
            **study.get("reference", {}).get("settings", {}),
            **job.get("reference_settings", {}),
            "compute_backend": "numpy",
            "compute_precision": "float64",
            "adc_compute_backend": "numpy",
            "adc_compute_precision": "float64",
            "po_calibration": "none",
            "hybrid_calibration": "none",
        }
    )
    key = configuration_id(job, provenance, inputs, settings, None)
    directory = Path(study["reference_root"]) / "cpu-float64" / job["case"]["id"] / key[:20]
    return directory, key, settings


def read_reference(directory, key):
    metadata = json.loads((directory / "reference.json").read_text())
    if metadata["configuration_id"] != key or metadata["adc_sha256"] != file_hash(
        directory / "adc.npz"
    ):
        raise ValueError(f"Reference integrity mismatch at {directory}")
    with np.load(directory / "adc.npz", allow_pickle=False) as data:
        return np.array(data["adc"], copy=True), metadata


def prepare_references(study, jobs, provenance, inputs):
    seen = set()
    for job in jobs:
        if job["track"] != "numerical":
            continue
        directory, key, settings = reference_for(study, job, provenance, inputs[job["case"]["id"]])
        if key in seen:
            continue
        seen.add(key)
        if (directory / "reference.json").exists():
            read_reference(directory, key)
            print(
                f"Reference cached: {job['case']['id']} {job['solver']} seed={job['seed']}",
                flush=True,
            )
            continue
        print(f"Reference: {job['case']['id']} {job['solver']} seed={job['seed']}", flush=True)
        result = launch_worker(request_for(job, directory, settings=settings), cpu_reference=True)
        write_json(
            directory / "reference.json",
            {
                "configuration_id": key,
                "settings": settings,
                "provenance": provenance,
                "inputs": inputs[job["case"]["id"]],
                "worker": result,
                "adc_sha256": file_hash(directory / "adc.npz"),
                "reference_kind": "CPU high-fidelity numerical reference, convergence must be established",
            },
        )


def calibrate(study, cases, provenance):
    """Pool native first-frame RT/PO reference powers on held-out bundles."""

    for case in cases:
        if not case.get("calibration_bundle"):
            continue
        if Path(case["calibration_bundle"]).resolve() == Path(case["bundle"]).resolve():
            raise ValueError("Calibration must use a separate held-out bundle")
        calibration_case = {
            **case,
            "bundle": case["calibration_bundle"],
            "max_frames": 1,
            "scenario": "stationary_human",
        }
        inputs = case_fingerprint(study, calibration_case)
        if inputs["sha256"] == case_fingerprint(study, case)["sha256"]:
            raise ValueError("Calibration and evaluation bundles have identical contents")
        settings = resolve_settings(
            {
                **study["settings"],
                **case.get("settings", {}),
                **study.get("calibration", {}).get("settings", {}),
                "hybrid_calibration": "rt_first_pose",
                "po_calibration": "none",
                "hybrid_coupling": False,
                "compute_backend": "numpy",
                "compute_precision": "float64",
                "adc_compute_backend": "numpy",
                "adc_compute_precision": "float64",
            }
        )
        powers = []
        for seed in study["seeds"]:
            seeded = {**settings, "rt_seed": seed}
            key = stable_hash(
                {
                    "inputs": inputs["sha256"],
                    "settings": seeded,
                    "source": provenance["source_sha256"],
                }
            )
            directory = Path(study["reference_root"]) / "calibration" / case["id"] / key[:20]
            job = {
                "case": calibration_case,
                "solver": "hybrid_po",
                "settings": seeded,
                "target_material": case.get("target_material", {}),
            }
            result = launch_worker(request_for(job, directory), cpu_reference=True)
            if not result["calibration_powers"]:
                raise ValueError("Native calibration did not report reference powers")
            rt_sum = sum(
                p["rt_reference_rms_power"] * p["weight"] for p in result["calibration_powers"]
            )
            po_sum = sum(
                p["raw_po_reference_rms_power"] * p["weight"] for p in result["calibration_powers"]
            )
            weight = sum(p["weight"] for p in result["calibration_powers"])
            powers.append({"seed": seed, "rt_power": rt_sum / weight, "po_power": po_sum / weight})
        rt_power, po_power = np.mean([p["rt_power"] for p in powers]), np.mean(
            [p["po_power"] for p in powers]
        )
        if rt_power <= 0 or po_power <= 0:
            raise ValueError("Cannot fit a frozen gain from zero reference power")
        write_json(
            case["frozen_gain_file"],
            {
                "amplitude_gain": float(np.sqrt(rt_power / po_power)),
                "powers": powers,
                "definition": "sqrt(mean held-out RT human power / mean held-out PO human power)",
                "source_sha256": provenance["source_sha256"],
                "calibration_inputs": inputs,
                "evaluation_bundle_sha256": case_fingerprint(study, case)["sha256"],
                "calibration_settings": settings,
                "case_id": case["id"],
                "target_material": case.get("target_material", {}),
                "rt_power_relative_std": float(np.std([p["rt_power"] for p in powers]) / rt_power),
            },
        )
        print(f"Frozen gain saved: {case['frozen_gain_file']}", flush=True)


def frozen_gain(job, provenance, inputs, study=None):
    if job["track"] != "physical" or job["solver"] == "rt":
        return None, None
    path = Path(job["case"]["frozen_gain_file"])
    if not path.exists():
        if job["solver"] == "hybrid_po" and job["case"].get("require_frozen_gain", False):
            raise ValueError(f"Missing {path}; run study --calibrate on held-out data first")
        return None, None
    data = json.loads(path.read_text())
    if (
        data.get("source_sha256") != provenance["source_sha256"]
        or data.get("evaluation_bundle_sha256") != inputs["sha256"]
    ):
        raise ValueError(
            "Frozen gain provenance does not match the current code/evaluation bundle"
        )
    case = job["case"]
    if case.get("calibration_bundle") and data.get("calibration_inputs", {}).get("sha256") != (
        case_fingerprint(
            study or {},
            {
                **case,
                "bundle": case["calibration_bundle"],
                "scenario": "stationary_human",
                "max_frames": 1,
            },
        )["sha256"]
    ):
        raise ValueError("Frozen gain does not match the current held-out calibration bundle")
    if data.get("target_material", {}) != case.get("target_material", {}):
        raise ValueError("Frozen gain does not match the calibration target material")
    gain = data["amplitude_gain"]
    if not np.isfinite(gain) or gain < 0:
        raise ValueError("Invalid frozen amplitude gain")
    return gain, {"path": str(path), "sha256": file_hash(path), "source": data}


def run_jobs(study, jobs, provenance, inputs, platform_directory, info, *, resume=False):
    from .study_metrics import compare, measured_adc, scheduled_displacement

    runtime_key = stable_hash({"machine": info, "timing_protocol": study["timing"]})[:10]
    motion_cache = {}
    for index, job in enumerate(jobs, 1):
        case = job["case"]
        clip = load_case(study, case)
        if job["track"] == "physical" and clip.data_origin != "measurement":
            raise ValueError(
                "Physical comparisons require an ADC artifact declared as measurement"
            )
        gain, gain_metadata = frozen_gain(job, provenance, inputs[case["id"]], study)
        key = configuration_id(job, provenance, inputs[case["id"]], job["settings"], gain)
        # Include experiment labels in paths while keeping a portable configuration ID.
        directory = (
            platform_directory
            / "runs"
            / case["id"]
            / job["experiment"]
            / (key[:20] + "-" + runtime_key)
        )
        if resume and (directory / "result.json").exists():
            prior = json.loads((directory / "result.json").read_text())
            if (
                prior.get("configuration_id") == key
                and prior.get("status") == "complete"
                and prior.get("accuracy_adc_sha256") == file_hash(directory / "accuracy/adc.npz")
            ):
                print(f"Resume completed record: {directory}", flush=True)
                continue
        print(
            f"[{index}/{len(jobs)}] {case['id']} {job['experiment']} {job['solver']} seed={job['seed']}",
            flush=True,
        )
        numerical, reference_metadata, reference = None, None, None
        if job["track"] == "numerical":
            ref_directory, ref_key, _ = reference_for(study, job, provenance, inputs[case["id"]])
            if not (ref_directory / "reference.json").exists():
                raise ValueError(
                    f"Missing CPU reference {ref_directory}; run study --references first"
                )
            reference, reference_metadata = read_reference(ref_directory, ref_key)
        request = request_for(job, directory / "accuracy", frozen_gain=gain)
        accuracy = launch_worker(request)
        if (
            job["track"] == "physical"
            and case.get("require_valid_array", True)
            and any(check.get("warning") for check in accuracy["phase_center_checks"])
        ):
            raise ValueError(
                f"Array validity check failed for {case['id']}; choose compact subarrays"
            )
        with np.load(directory / "accuracy/adc.npz", allow_pickle=False) as data:
            adc = np.array(data["adc"], copy=True)
        config = BenchmarkRunConfig(
            solvers=(job["solver"],),
            hybrid_calibration="none",
            solver_settings=job["settings"],
            target_material=job["target_material"],
            selected_tx_indices=case.get("selected_tx_indices"),
            selected_rx_indices=case.get("selected_rx_indices"),
            smpl_model_dir=case.get("smpl_model_dir"),
            ignore_tdm_timing=case.get("ignore_tdm_timing", False),
            background_subtract=case.get("background_subtract", False),
            clutter_removal=case.get("clutter_removal", "none"),
        )
        measurement_gain = case.get("measurement_amplitude_gain", 1.0)
        measurement, measurement_arrays = compare(
            clip,
            config,
            adc * measurement_gain,
            measured_adc(clip, config),
            coherent=False,
            absolute_power=case.get("absolute_power_calibrated", False),
        )
        np.savez_compressed(directory / "measurement_maps.npz", **measurement_arrays)
        if job["track"] == "numerical":
            numerical, numerical_arrays = compare(
                clip,
                config,
                adc,
                reference,
                coherent=not job["settings"]["rt_diffuse_reflection"] or job["solver"] == "po",
            )
            np.savez_compressed(directory / "numerical_maps.npz", **numerical_arrays)
        timing_request = {
            **request_for(job, directory / "timing", phase="timing", frozen_gain=gain),
            "timing": study["timing"],
        }
        timing = launch_worker(timing_request)
        motion_key = stable_hash(
            {
                "inputs": inputs[case["id"]]["sha256"],
                "max_frames": case.get("max_frames"),
                "ignore_tdm_timing": config.ignore_tdm_timing,
                "tx": config.selected_tx_indices,
                "rx": config.selected_rx_indices,
                "rt_refresh_frames": job["settings"]["rt_refresh_frames"],
                "po_refresh_frames": job["settings"]["po_refresh_frames"],
            }
        )
        if motion_key not in motion_cache:
            motion_cache[motion_key] = scheduled_displacement(clip, config)
        result = {
            "status": "complete",
            "configuration_id": key,
            "execution_environment_id": runtime_key,
            "timing_protocol": study["timing"],
            "case": case["id"],
            "scenario": case["scenario"],
            "experiment": job["experiment"],
            "track": job["track"],
            "solver": job["solver"],
            "seed": job["seed"],
            "axis": job["axis"],
            "changes": job["changes"],
            "settings": job["settings"],
            "target_material": job["target_material"],
            "provenance": provenance,
            "case_metadata": case.get("metadata", {}),
            "analysis_settings": analysis_settings(case),
            "selected_tx_indices": case.get("selected_tx_indices"),
            "selected_rx_indices": case.get("selected_rx_indices"),
            "input_fingerprint": inputs[case["id"]],
            "frozen_gain": gain_metadata,
            "measurement_amplitude_gain": measurement_gain,
            "measurement_calibration_source": case.get("measurement_calibration_source"),
            "ignore_tdm_timing": case.get("ignore_tdm_timing", False),
            "measurement": measurement,
            "numerical": numerical,
            "reference": (
                {
                    "configuration_id": reference_metadata["configuration_id"],
                    "adc_sha256": reference_metadata["adc_sha256"],
                    "settings": reference_metadata["settings"],
                }
                if reference_metadata
                else None
            ),
            "accuracy": accuracy,
            "timing": timing,
            "motion_axis": motion_cache[motion_key],
            "accuracy_adc_sha256": file_hash(directory / "accuracy/adc.npz"),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(directory / "result.json", result)


def report(platform_directory):
    from .study_metrics import aggregate_seed_powers

    groups = {}
    for path in sorted((platform_directory / "runs").rglob("result.json")):
        row = json.loads(path.read_text())
        settings_without_seed = {k: v for k, v in row["settings"].items() if k != "rt_seed"}
        context = {
            "source": row["provenance"]["source_sha256"],
            "inputs": row["input_fingerprint"]["sha256"],
            "material": row["target_material"],
            "gain": row["frozen_gain"],
            "environment": row.get("execution_environment_id"),
            "analysis": row.get("analysis_settings"),
            "selected_tx_indices": row.get("selected_tx_indices"),
            "selected_rx_indices": row.get("selected_rx_indices"),
            "ignore_tdm_timing": row["ignore_tdm_timing"],
            "reference_settings": {
                k: v
                for k, v in (row["reference"] or {}).get("settings", {}).items()
                if k != "rt_seed"
            },
        }
        cohort = stable_hash({"context": context, "settings": settings_without_seed})
        grouping = (row["case"], row["experiment"], row["solver"], cohort)
        row["comparison_context"] = stable_hash(context)
        groups.setdefault(grouping, []).append((path, row))
    summaries = []
    for (case, experiment, solver, cohort), records in groups.items():
        first = records[0][1]
        medians = [r["timing"]["timing_median_seconds"] for _, r in records]
        row = {
            "case": case,
            "experiment": experiment,
            "solver": solver,
            "track": first["track"],
            "changes": first["changes"],
            "cohort": cohort,
            "comparison_context": first["comparison_context"],
            "execution_environment_id": first.get("execution_environment_id"),
            "seed_count": len(records),
            "seeds": [r["seed"] for _, r in records],
            "median_seconds": float(np.median(medians)),
            "timing_iqr_seconds": float(
                np.median([r["timing"]["timing_iqr_seconds"] for _, r in records])
            ),
            "motion_axis": first["motion_axis"],
            "measurement_seed_mean_power": aggregate_seed_powers(
                [p.parent / "measurement_maps.npz" for p, _ in records]
            ),
            "numerical_seed_mean_power": (
                aggregate_seed_powers([p.parent / "numerical_maps.npz" for p, _ in records])
                if first["track"] == "numerical"
                else None
            ),
            "records": [str(p.relative_to(platform_directory)) for p, _ in records],
        }
        individual = [
            r["measurement"]["range_time"]["relative_l2"]
            for _, r in records
            if r["measurement"]["range_time"]["relative_l2"] is not None
        ]
        row["measurement_error_seed_iqr"] = (
            float(np.percentile(individual, 75) - np.percentile(individual, 25))
            if individual
            else None
        )
        row["measurement_error_seed_mean"] = float(np.mean(individual)) if individual else None
        row["measurement_error_seed_interval"] = (
            seed_mean_interval(individual) if individual else None
        )
        row["numerical_error"] = (
            row["numerical_seed_mean_power"]["range_time"]["relative_l2"]
            if row["numerical_seed_mean_power"]
            else None
        )
        numerical_errors = [
            r["numerical"]["range_time"]["relative_l2"]
            for _, r in records
            if r["numerical"] and r["numerical"]["range_time"]["relative_l2"] is not None
        ]
        row["numerical_error_seed_iqr"] = (
            float(np.percentile(numerical_errors, 75) - np.percentile(numerical_errors, 25))
            if numerical_errors
            else None
        )
        summaries.append(row)
    write_json(platform_directory / "summary.json", summaries)
    with (platform_directory / "summary.csv").open("w", newline="") as stream:
        fields = [
            "case",
            "experiment",
            "solver",
            "track",
            "changes",
            "seed_count",
            "median_seconds",
            "timing_iqr_seconds",
            "numerical_error",
            "numerical_error_seed_iqr",
            "measurement_error_seed_mean",
            "measurement_error_seed_iqr",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {**r, "changes": json.dumps(r["changes"], sort_keys=True)} for r in summaries
        )
    if summaries:
        from .study_report import plot_summary

        plot_summary(platform_directory, summaries)
    return summaries


def compare_platforms(platform_directory, peer_directory):
    from .study_metrics import relative_error

    def records(directory):
        selected = {}
        for path in sorted((directory / "runs").rglob("result.json")):
            row = json.loads(path.read_text())
            key = row["configuration_id"]
            priority = (bool(row["reference"]), row.get("completed_utc", ""))
            if key not in selected or priority > selected[key][0]:
                selected[key] = (priority, path, row)
        return selected

    peers = records(peer_directory)
    results = []
    for key, (_, p, row) in records(platform_directory).items():
        if key not in peers:
            continue
        _, peer, peer_row = peers[key]
        with (
            np.load(p.parent / "accuracy/adc.npz", allow_pickle=False) as a,
            np.load(peer.parent / "accuracy/adc.npz", allow_pickle=False) as b,
        ):
            if a["adc"].shape != b["adc"].shape:
                raise ValueError("Cross-platform ADC shapes differ")
            local_ref = (row["reference"] or {}).get("adc_sha256")
            peer_ref = (peer_row["reference"] or {}).get("adc_sha256")
            results.append(
                {
                    "configuration_id": row["configuration_id"],
                    "adc_relative_l2": relative_error(b["adc"], a["adc"]),
                    "adc_relative_l2_definition": "norm(local - peer) / norm(peer)",
                    "reference_adc_sha256": local_ref,
                    "peer_reference_adc_sha256": peer_ref,
                    "identical_reference_artifact": (
                        local_ref == peer_ref if local_ref and peer_ref else None
                    ),
                    "local_numerical_error": row.get("numerical"),
                    "peer_numerical_error": peer_row.get("numerical"),
                    "local": str(p),
                    "peer": str(peer),
                }
            )
    write_json(platform_directory / "cross_platform.json", results)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--plan", action="store_true", help="Validate and list jobs without simulation (default)"
    )
    actions.add_argument(
        "--references", action="store_true", help="Generate shared CPU numerical references"
    )
    actions.add_argument(
        "--calibrate", action="store_true", help="Fit a frozen gain on held-out data"
    )
    actions.add_argument(
        "--run", action="store_true", help="Execute accuracy and fresh timing workers"
    )
    actions.add_argument(
        "--report", action="store_true", help="Rebuild tables/figures without simulation"
    )
    parser.add_argument(
        "--platform", help="Output-directory label; detected hardware is always recorded"
    )
    parser.add_argument("--case", action="append", help="Restrict to a case ID; repeatable")
    parser.add_argument(
        "--experiment", action="append", help="Restrict to an experiment ID; repeatable"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep completed records, including their original timings",
    )
    parser.add_argument(
        "--peer-platform",
        type=Path,
        help="Compare matching ADC outputs from another platform directory",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        help="Use the identical shared CPU reference artifacts on both machines",
    )
    args = parser.parse_args(argv)
    if args.worker:
        worker(json.loads(args.worker.read_text()))
        return 0
    if args.config is None:
        parser.error("--config is required")
    try:
        study = load_study(args.config)
        if args.reference_root:
            original_root = Path(study["reference_root"])
            study["reference_root"] = str(args.reference_root.resolve())
            for case in study["cases"]:
                if Path(case["frozen_gain_file"]).is_relative_to(original_root):
                    case["frozen_gain_file"] = str(
                        Path(study["reference_root"])
                        / Path(case["frozen_gain_file"]).relative_to(original_root)
                    )
        jobs = expand_jobs(
            study, cases=set(args.case or []), experiments=set(args.experiment or [])
        )
        info = machine_info()
        label = safe_name(args.platform or platform_slug(info))
        directory = Path(study["output_root"]) / study["name"] / label
        directory.mkdir(parents=True, exist_ok=True)
        provenance = code_provenance()
        plan = {
            "name": study["name"],
            "platform_label": label,
            "machine": info,
            "provenance": provenance,
            "job_count": len(jobs),
            "timing": study["timing"],
            "reference_root": study["reference_root"],
            "cold_start_policy": "first-in-process, existing disk caches preserved; warmth not assumed",
            "baseline": "resolved shipped defaults plus explicit suite/case/experiment overrides",
            "measurement_inputs": [
                {
                    "id": c["id"],
                    "path": c["bundle"],
                    "available": Path(c["bundle"], "bundle.json").is_file(),
                }
                for c in study["cases"]
            ],
            "jobs": jobs,
            "minimum_warmed_timing_seconds": len(jobs) * study["timing"]["minimum_seconds"],
            "minimum_native_executions": len(jobs)
            * (1 + study["timing"]["warmups"] + study["timing"]["repetitions"]),
            "cost_estimate": "Use pilot timing medians × warmups/repetitions/seeds; no unmeasured ETA is invented",
        }
        write_json(directory / "plan.json", plan)
        print(f"{len(jobs)} jobs; platform results: {directory}", flush=True)
        if args.report:
            report(directory)
        elif args.references or args.calibrate or args.run:
            selected = {j["case"]["id"]: j["case"] for j in jobs}
            inputs = {key: case_fingerprint(study, case) for key, case in selected.items()}
            if args.references:
                prepare_references(study, jobs, provenance, inputs)
            elif args.calibrate:
                calibrate(study, list(selected.values()), provenance)
            else:
                run_jobs(study, jobs, provenance, inputs, directory, info, resume=args.resume)
                report(directory)
        if args.peer_platform:
            compare_platforms(directory, args.peer_platform.resolve())
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
