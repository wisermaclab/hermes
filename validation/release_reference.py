# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Immutable release references and gates independent of the current source hash."""

from itertools import combinations
import json
from pathlib import Path
import shutil

import numpy as np

from .study_metrics import relative_error
from .study_runtime import file_hash, stable_hash, write_json
from .uncertainty import seed_mean_interval


def power_features(adc):
    """Unnormalized range-time power, mean over loops and antenna pairs."""
    adc = np.asarray(adc)
    if adc.ndim != 4 or not np.all(np.isfinite(adc)):
        raise ValueError("Expected finite [frame, loop, sample, channel] ADC")
    transformed = np.fft.fft(adc * np.hanning(adc.shape[2])[None, None, :, None], axis=2)
    return np.mean(abs(transformed) ** 2, axis=(1, 3))


def noise_band(features):
    """Largest observed seed-to-seed relative L2 distance, frozen at release time.

    This conservative empirical envelope is a regression tolerance, not a CI.
    It has no guaranteed coverage for unseen random seeds.
    """
    features = np.asarray(features)
    if len(features) < 5:
        raise ValueError("Stochastic references need at least five independent seeds")
    scale = float(np.linalg.norm(np.mean(features, axis=0)))
    if scale == 0:
        raise ValueError("A zero-power stochastic reference cannot establish a noise band")
    distances = [float(np.linalg.norm(a - b) / scale) for a, b in combinations(features, 2)]
    return {
        "relative_l2_limit": max(distances),
        "pairwise_relative_l2": distances,
        "definition": "maximum observed seed-to-seed linear-power relative L2 / norm(seed mean)",
    }


def checked_adc(root, record):
    relative = Path(record["adc"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("ADC reference must be a portable path inside its run directory")
    path = Path(root) / relative
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("ADC reference escapes its run directory")
    if file_hash(path) != record["adc_sha256"]:
        raise ValueError(f"ADC checksum mismatch: {record['adc']}")
    with np.load(path, allow_pickle=False) as data:
        return np.array(data["adc"], copy=True)


def grouped_records(run):
    result = {}
    for record in run["records"]:
        key = (record["fixture"], record["solver"])
        records = result.setdefault(key, [])
        if any(r["seed"] == record["seed"] for r in records):
            raise ValueError("Duplicate seed within a fixture/solver")
        records.append(record)
    if not result:
        raise ValueError("No release results")
    return result


def freeze_reference(run_root, destination, release):
    """Create once; deliberate baseline changes need a different destination."""
    run_root, destination = Path(run_root), Path(destination)
    run = json.loads((run_root / "run.json").read_text())
    sources = {
        r.get("runtime", {}).get("provenance", {}).get("source_sha256") for r in run["records"]
    }
    sources.discard(None)
    if len(sources) > 1:
        raise ValueError(
            "Cannot freeze workers from different source snapshots; rerun from one checkout"
        )
    groups = grouped_records(run)
    bands = {}
    for (fixture, solver), records in groups.items():
        arrays = [checked_adc(run_root, r) for r in records]
        if solver != "po":
            bands[f"{fixture}/{solver}"] = noise_band([power_features(a) for a in arrays])
    if destination.exists():
        raise ValueError("Reference destination already exists; frozen references are immutable")
    destination.mkdir(parents=True)
    for record in run["records"]:
        target = destination / record["adc"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run_root / record["adc"], target)
    reference = {**run, "release": release, "noise_bands": bands}
    write_json(destination / "run.json", reference)
    return reference


def compare_reference(
    reference_root, candidate_root, *, po_rtol=1e-10, po_atol=1e-16, stochastic_floor=1e-6
):
    """Fail closed on missing fixtures, protocol changes or damaged references."""
    reference_root, candidate_root = Path(reference_root), Path(candidate_root)
    reference = json.loads((reference_root / "run.json").read_text())
    candidate = json.loads((candidate_root / "run.json").read_text())
    for run in (reference, candidate):
        if stable_hash(run["protocol"]) != run["protocol_sha256"]:
            raise ValueError("Protocol checksum mismatch")
        sources = {
            r.get("runtime", {}).get("provenance", {}).get("source_sha256") for r in run["records"]
        }
        sources.discard(None)
        if len(sources) > 1:
            raise ValueError("Regression requires one source snapshot per run")
    if reference["protocol_sha256"] != candidate["protocol_sha256"]:
        raise ValueError("Inputs, settings, seeds or dependency/backend protocol changed")
    old, new = grouped_records(reference), grouped_records(candidate)
    if old.keys() != new.keys():
        raise ValueError("Reference and candidate fixture/solver sets differ")
    rows = []
    for (fixture, solver), records in old.items():
        a = [checked_adc(reference_root, r) for r in records]
        b = [checked_adc(candidate_root, r) for r in new[(fixture, solver)]]
        if len(a) != len(b) or any(x.shape != a[0].shape for x in a + b):
            raise ValueError("Reference/candidate seed counts or ADC shapes differ")
        if solver == "po":
            passed = bool(np.allclose(a, b, rtol=po_rtol, atol=po_atol))
            details = {
                "adc_relative_l2": relative_error(np.array(a), np.array(b)),
                "rtol": po_rtol,
                "atol": po_atol,
            }
        else:
            ap, bp = np.array([power_features(x) for x in a]), np.array(
                [power_features(x) for x in b]
            )
            error = relative_error(ap.mean(0), bp.mean(0))
            limit = max(
                reference["noise_bands"][f"{fixture}/{solver}"]["relative_l2_limit"],
                stochastic_floor,
            )
            passed = error is not None and error <= limit
            details = {
                "seed_mean_power_relative_l2": error,
                "frozen_noise_limit": limit,
                "reference_power_interval": seed_mean_interval(ap),
                "candidate_power_interval": seed_mean_interval(bp),
            }
        rows.append({"fixture": fixture, "solver": solver, "passed": passed, **details})
    return {
        "reference_release": reference["release"],
        "reference_source": reference["provenance"],
        "candidate_source": candidate["provenance"],
        "passed": all(r["passed"] for r in rows),
        "comparisons": rows,
        "reference_manifest_sha256": file_hash(reference_root / "run.json"),
        "protocol_sha256": stable_hash(reference["protocol"]),
    }
