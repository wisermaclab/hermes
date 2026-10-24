# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Machine provenance and isolated execution for the existing validation runner."""

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import warnings

import numpy as np


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=jsonable, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, default=jsonable, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def code_provenance():
    from mmWaveRadar.provenance import software_provenance

    root = Path(__file__).resolve().parents[1]
    sources = {}
    for folder in (root / "src", root / "validation", root / "tools/bundle_prepare"):
        for source in sorted(folder.rglob("*.py")):
            sources[str(source.relative_to(root))] = file_hash(source)

    def git(*args):
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        **software_provenance(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "source_sha256": stable_hash(sources),
        "source_files": sources,
    }


def machine_info():
    from mmWaveRadar.simulation.runtime import torch_runtime_summary

    info = {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
        **torch_runtime_summary(),
    }
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        )
        if result.returncode == 0:
            info["cpu"] = result.stdout.strip()
    info["versions"] = {}
    for package in ("numpy", "scipy", "torch", "mitsuba", "drjit", "sionna-rt"):
        try:
            info["versions"][package] = version(package)
        except PackageNotFoundError:
            info["versions"][package] = None
    return info


def platform_slug(info):
    import re

    cpu = info.get("cpu")
    if not cpu or cpu.lower() in {"arm", "arm64", "aarch64", "x86_64", "amd64"}:
        cpu = info["system"] + "-" + info["machine"]
    label = info.get("torch_cuda_device_name") or cpu
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def synchronize():
    import drjit as dr
    from mmWaveRadar.simulation.compute import torch_cuda_is_available, torch_mps_is_available

    dr.sync_thread()
    try:
        import torch

        if torch_cuda_is_available(torch):
            torch.cuda.synchronize()
        if torch_mps_is_available(torch):
            torch.mps.synchronize()
    except ImportError:
        pass


def resolved_backends(settings):
    import mitsuba as mi
    from mmWaveRadar.simulation.compute import resolve_torch_runtime

    def resolve(backend, precision):
        runtime = resolve_torch_runtime(backend, precision)
        if runtime is not None:
            return {"backend": "torch", "device": str(runtime.device), "precision": precision}
        return {"backend": "numpy", "device": "cpu", "precision": precision}

    return {
        "mitsuba_variant": mi.variant(),
        "po": resolve(settings["compute_backend"], settings["compute_precision"]),
        "adc": resolve(
            settings["adc_compute_backend"] or settings["compute_backend"],
            settings["adc_compute_precision"] or settings["compute_precision"],
        ),
        "rt_trace_precision": "native Mitsuba variant; not promoted by ADC precision",
    }


def worker(request):
    """Execute fresh native simulations; artifact writes are outside timers."""

    from bundle_prepare.contract import Bundle
    from .configuration import BenchmarkRunConfig
    from .simulation import simulate_adc as _simulate_adc
    from .channels import slice_clip as _slice_clip
    import drjit as dr
    import mitsuba as mi
    from mmWaveRadar.simulation.runtime import configure_mitsuba_variant

    folder = Path(request["directory"])
    folder.mkdir(parents=True, exist_ok=True)
    configure_mitsuba_variant()
    accuracy = request["phase"] != "timing"
    if accuracy and mi.variant().startswith("llvm"):
        dr.set_thread_count(1)
    clip = _slice_clip(Bundle.load(request["bundle"]), request.get("max_frames"))
    diagnostics = []

    def observe(cube):
        metadata = cube.metadata
        counts = np.asarray(metadata.path_counts)
        calibration = dict(metadata.po_calibration or {})
        diagnostics.append(
            {
                "provenance": getattr(metadata, "provenance", None),
                "adc_shape": list(cube.adc.shape),
                "runtime_profile_s": metadata.runtime_profile_s or {},
                "runtime_profile_counts": metadata.runtime_profile_counts or {},
                "max_valid_paths": int(np.max(counts, initial=0)),
                "calibration_powers": {
                    k: float(calibration[k])
                    for k in ("rt_reference_rms_power", "raw_po_reference_rms_power")
                    if k in calibration
                },
                "phase_center_checks": [
                    e for e in metadata.events if e.get("reason") == "phase_center_approximation"
                ],
                "calibration_weight": int(np.prod(cube.adc.shape[1:])),
                "po_recomputed_faces": (
                    int(np.sum(metadata.incremental_recomputed_face_counts))
                    if metadata.incremental_recomputed_face_counts is not None
                    else None
                ),
                "po_phase_updated_faces": (
                    int(np.sum(metadata.incremental_phase_updated_face_counts))
                    if metadata.incremental_phase_updated_face_counts is not None
                    else None
                ),
            }
        )

    config = BenchmarkRunConfig(
        solvers=(request["solver"],),
        solver_settings=request["settings"],
        hybrid_calibration="none",
        cube_observer=observe,
        target_material=request.get("target_material", {}),
        frozen_human_po_amplitude_gain=request.get("frozen_gain"),
        ignore_tdm_timing=request.get("ignore_tdm_timing", False),
        smpl_model_dir=request.get("smpl_model_dir"),
        selected_tx_indices=request.get("selected_tx_indices"),
        selected_rx_indices=request.get("selected_rx_indices"),
    )

    def execute():
        diagnostics.clear()
        synchronize()
        start = time.perf_counter()
        adc = _simulate_adc(clip, config, solver=request["solver"])
        synchronize()
        pipeline_seconds = time.perf_counter() - start
        native_seconds = sum(
            float(d["runtime_profile_s"]["validation_simulator_run"]) for d in diagnostics
        )
        return adc, pipeline_seconds, native_seconds

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adc, first_seconds, first_native = execute()
        if not np.all(np.isfinite(adc)):
            raise ValueError("Simulator returned non-finite ADC")
        timing = request.get("timing", {})
        samples = []
        pipeline_samples = []
        if not accuracy:
            for _ in range(max(0, timing.get("warmups", 1) - 1)):
                execute()
            while len(samples) < timing.get("repetitions", 5) or sum(samples) < timing.get(
                "minimum_seconds", 10.0
            ):
                if len(samples) >= timing.get("maximum_repetitions", 1000):
                    break
                adc, pipeline_elapsed, native_elapsed = execute()
                samples.append(native_elapsed)
                pipeline_samples.append(pipeline_elapsed)
        else:
            np.savez_compressed(
                folder / "adc.npz",
                adc=adc,
                provenance_json=np.asarray(
                    json.dumps([d["provenance"] for d in diagnostics], default=jsonable)
                ),
            )
    profile = {}
    profile_counts = {}
    calibration_powers = []
    checks = []
    for item in diagnostics:
        for name, seconds in item["runtime_profile_s"].items():
            profile[name] = profile.get(name, 0.0) + float(seconds)
        for name, count in item["runtime_profile_counts"].items():
            profile_counts[name] = profile_counts.get(name, 0) + int(count)
        if item["calibration_powers"]:
            calibration_powers.append(
                {**item["calibration_powers"], "weight": item["calibration_weight"]}
            )
        checks.extend(item["phase_center_checks"])
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "provenance": code_provenance(),
        "seed": request["settings"]["rt_seed"],
        "settings": request["settings"],
        "phase": request["phase"],
        "first_in_process_seconds": first_seconds,
        "first_native_simulator_seconds": first_native,
        "cold_start_definition": "first simulation in process, existing on-disk JIT caches",
        "timing_samples_seconds": samples,
        "pipeline_timing_samples_seconds": pipeline_samples,
        "timing_scope": "native simulator.run calls with device synchronization; validation preparation excluded",
        "pipeline_timing_median_seconds": (
            float(np.median(pipeline_samples)) if pipeline_samples else None
        ),
        "timing_median_seconds": float(np.median(samples)) if samples else None,
        "timing_iqr_seconds": (
            float(np.percentile(samples, 75) - np.percentile(samples, 25)) if samples else None
        ),
        "timing_duration_target_met": (
            sum(samples) >= timing.get("minimum_seconds", 10.0) if samples else None
        ),
        "runtime_profile_s": profile,
        "runtime_profile_counts": profile_counts,
        "po_recomputed_faces": sum(d["po_recomputed_faces"] or 0 for d in diagnostics),
        "po_phase_updated_faces": sum(d["po_phase_updated_faces"] or 0 for d in diagnostics),
        "profile_definition": "native stages from last execution, summed across physical TDM slots",
        "machine": machine_info(),
        "resolved_backends": resolved_backends(request["settings"]),
        "accuracy_cpu_threads": 1 if accuracy and mi.variant().startswith("llvm") else None,
        "peak_process_rss_bytes": int(peak_rss if sys.platform == "darwin" else peak_rss * 1024),
        "adc_shape": list(adc.shape),
        "adc_dtype": str(adc.dtype),
        "warnings": sorted({str(w.message) for w in caught}),
        "phase_center_checks": checks,
        "calibration_powers": calibration_powers,
        "max_valid_paths": max((d["max_valid_paths"] for d in diagnostics), default=0),
        "path_cap": request["settings"]["rt_max_paths_per_source"],
    }
    result["path_cap_reached_indicator"] = result["max_valid_paths"] >= result["path_cap"]
    result["path_cap_note"] = (
        "Post-filter/group path counts are an indicator, not a proof of no truncation"
    )
    write_json(folder / "worker_result.json", result)
    return result


def launch_worker(request, *, cpu_reference=False):
    folder = Path(request["directory"])
    folder.mkdir(parents=True, exist_ok=True)
    request_path = folder / "request.json"
    write_json(request_path, request)
    environment = os.environ.copy()
    root = Path(__file__).resolve().parents[1]
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), str(root), str(root / "tools"), environment.get("PYTHONPATH", "")]
    )
    if cpu_reference:
        environment["MMWAVE_MITSUBA_VARIANT"] = "llvm_ad_mono_polarized"
    with (folder / "worker.log").open("w") as log:
        result = subprocess.run(
            [sys.executable, "-m", "validation.study", "--worker", str(request_path)],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"Benchmark worker failed; see {folder / 'worker.log'}")
    return json.loads((folder / "worker_result.json").read_text())
