# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Independent verification, seed statistics and release regression failures."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from mmWaveRadar.provenance import array_hash, capture_provenance, directory_hashes
from validation.analytic import (
    native_po_rcs,
    plate_mesh,
    plate_rcs,
    sphere_optical_rcs,
    square_trihedral_rcs,
)
from validation.release import load_config, prepare_pec_plate, run_suite, write_report
from validation.release_reference import (
    compare_reference,
    freeze_reference,
    noise_band,
    power_features,
)
from validation.study_runtime import file_hash, stable_hash, write_json
from validation.uncertainty import seed_mean_interval


def fake_run(root, *, change=1.0, source="old"):
    root.mkdir()
    records = []
    for solver in ("po", "rt", "hybrid_po"):
        for seed in ([0] if solver == "po" else range(5)):
            adc = np.exp(2j * np.pi * np.arange(16) / 16)[None, None, :, None]
            adc = adc * (1 + 0.01 * seed) * change
            path = root / f"{solver}-{seed}.npz"
            np.savez(path, adc=adc)
            records.append(
                {
                    "fixture": "plate",
                    "solver": solver,
                    "seed": seed,
                    "kind": "synthetic",
                    "adc": path.name,
                    "adc_sha256": file_hash(path),
                }
            )
    protocol = {"settings": "fixed", "inputs": "fixed", "seeds": list(range(5))}
    write_json(
        root / "run.json",
        {
            "protocol": protocol,
            "protocol_sha256": stable_hash(protocol),
            "provenance": {"version": source, "git_commit": source, "git_dirty": False},
            "records": records,
        },
    )
    return root


def test_releases_compare_across_source_changes_but_do_not_overwrite(tmp_path):
    old = fake_run(tmp_path / "old")
    new = fake_run(tmp_path / "new", source="new")
    reference = tmp_path / "reference"
    freeze_reference(old, reference, "0.2.0")
    assert compare_reference(reference, new)["passed"]
    with pytest.raises(ValueError, match="already exists"):
        freeze_reference(new, reference, "0.3.0")


def test_real_physics_change_fails_po_and_stochastic_gates(tmp_path):
    old = fake_run(tmp_path / "old")
    new = fake_run(tmp_path / "new", change=2)
    reference = tmp_path / "reference"
    freeze_reference(old, reference, "0.2.0")
    result = compare_reference(reference, new)
    assert not result["passed"]
    assert all(not row["passed"] for row in result["comparisons"])


def test_po_detects_phase_change_even_when_power_is_identical(tmp_path):
    old = fake_run(tmp_path / "old")
    reference = tmp_path / "reference"
    freeze_reference(old, reference, "0.2.0")
    new = fake_run(tmp_path / "new", change=1j)
    comparisons = compare_reference(reference, new)["comparisons"]
    assert not next(row for row in comparisons if row["solver"] == "po")["passed"]
    assert all(row["passed"] for row in comparisons if row["solver"] != "po")


@pytest.mark.parametrize("damage", ["checksum", "protocol", "missing", "escape"])
def test_release_reference_rejects_damaged_or_incompatible_artifacts(tmp_path, damage):
    old = fake_run(tmp_path / "old")
    new = fake_run(tmp_path / "new")
    reference = tmp_path / "reference"
    freeze_reference(old, reference, "0.2.0")
    manifest = json.loads((new / "run.json").read_text())
    if damage == "checksum":
        (new / manifest["records"][0]["adc"]).write_bytes(b"broken")
    elif damage == "protocol":
        manifest["protocol"]["settings"] = "changed"
        manifest["protocol_sha256"] = stable_hash(manifest["protocol"])
    elif damage == "missing":
        manifest["records"] = manifest["records"][1:]
    else:
        manifest["records"][0]["adc"] = "../outside.npz"
    write_json(new / "run.json", manifest)
    with pytest.raises((ValueError, FileNotFoundError)):
        compare_reference(reference, new)


def test_noise_envelope_requires_five_nonzero_seed_observations():
    values = np.arange(1, 6, dtype=float)[:, None]
    band = noise_band(values)
    assert band["relative_l2_limit"] == pytest.approx(4 / 3)
    with pytest.raises(ValueError, match="five"):
        noise_band(values[:4])
    with pytest.raises(ValueError, match="zero"):
        noise_band(np.zeros_like(values))


def test_seed_interval_uses_student_t_and_never_invents_single_seed_uncertainty():
    interval = seed_mean_interval([1, 2, 3, 4, 5])
    assert interval["mean"] == 3
    assert interval["lower"] == pytest.approx(1.03675684)
    assert interval["upper"] == pytest.approx(4.96324316)
    assert seed_mean_interval([7])["lower"] is None
    assert seed_mean_interval(np.ones((5, 3)))["std"].shape == (3,)


def test_stochastic_power_is_invariant_to_global_phase():
    adc = np.ones((2, 3, 16, 4), complex)
    np.testing.assert_allclose(power_features(adc), power_features(adc * 1j))


def test_native_pec_plate_checks_radiometry_and_off_axis_integral_without_fit():
    width, frequency, distance = 0.152, 60e9, 1000
    for angle in (0, 0.5, 1):
        vertices, faces = plate_mesh(width, width, distance, angle, 4)
        actual = native_po_rcs(
            vertices, faces, frequency, distance, "parent_face_far_field_analytic"
        )
        assert actual == pytest.approx(plate_rcs(width, width, frequency, angle), rel=0.001)
    assert sphere_optical_rcs(2) == pytest.approx(4 * np.pi)
    assert square_trihedral_rcs(0.1, frequency) > plate_rcs(0.1, 0.1, frequency)


def test_bundle_writer_records_actual_packaged_file_hashes(tmp_path):
    root = prepare_pec_plate(tmp_path / "plate")
    record = json.loads((root / "provenance.json").read_text())
    assert record["input_hashes"] == directory_hashes(root)
    assert record["software"]["version"] == "0.2.0"
    assert record["software"]["dependencies"]["numpy"]
    assert "torch" in record["software"]["dependencies"]
    original = array_hash(np.arange(8))
    assert array_hash(np.arange(8) + 1) != original


def test_release_suite_runs_every_seed_through_native_worker_boundary(tmp_path, monkeypatch):
    import validation.release as release

    config_path = Path(__file__).resolve().parents[2] / "configs/release_validation.json"
    config = load_config(config_path)
    calls = []

    def worker(request, **_kwargs):
        calls.append(request)
        folder = Path(request["directory"])
        folder.mkdir(parents=True, exist_ok=True)
        adc = np.ones((1, 1, 64, 12), complex) * (1 + request["settings"]["rt_seed"] / 100)
        np.savez(folder / "adc.npz", adc=adc)
        return {
            "resolved_backends": {"adc": "numpy"},
            "accuracy_cpu_threads": 1,
            "adc_dtype": str(adc.dtype),
        }

    monkeypatch.setattr(release, "launch_worker", worker)
    result = run_suite(config, tmp_path / "run")
    assert len(calls) == 2 * (1 + 8 + 8)
    assert all(c["phase"] == "accuracy" for c in calls)
    assert result["provenance"]["dependencies"]["hermes-radar-sim"] == "0.2.0"
    assert "hermes-radar-sim" not in result["protocol"]["dependencies"]
    assert all("provenance.json" not in f["input_hashes"] for f in result["protocol"]["fixtures"])
    assert (tmp_path / "run/report.md").exists()


def test_report_never_labels_synthetic_results_as_measurement(tmp_path):
    root = fake_run(tmp_path / "run")
    report = write_report(root)
    assert not report["angular_studies"]
    assert all(r["measurement_range_time_error"] is None for r in report["fixtures"])
    assert "not measured" in (root / "report.md").read_text()


def test_provenance_has_no_personal_paths():
    record = capture_provenance(
        seed=7, runtime={"backend": "numpy", "precision": "float64"}, input_hashes={"scene": "abc"}
    )
    assert record["execution"]["seed"] == 7
    assert record["input_hashes"]["scene"] == "abc"
    assert str(Path.home()) not in json.dumps(record)


def test_angular_report_discloses_missing_return_instead_of_fitting_a_floor(tmp_path):
    root = fake_run(tmp_path / "run")
    run = json.loads((root / "run.json").read_text())
    originals = [r for r in run["records"] if r["solver"] == "po"]
    for angle, power in ((0, 1.0), (10, 0.0)):
        row = {
            **originals[0],
            "fixture": f"plate-{angle}",
            "angular_study": "copper",
            "angle_deg": angle,
            "peak_power": power,
            "measured_peak_power": 1.0,
        }
        run["records"].append(row)
    write_json(root / "run.json", run)
    result = write_report(root)["angular_studies"][0]
    assert result["angular_response_rmse_db"] is None
    assert result["missing_return_angles_deg"] == [10]


@pytest.mark.parametrize("operation", ["freeze", "compare"])
def test_release_gates_reject_mixed_worker_source_snapshots(tmp_path, operation):
    root = fake_run(tmp_path / "run")
    reference = tmp_path / "reference"
    freeze_reference(root, reference, "0.2.0")
    run = json.loads((root / "run.json").read_text())
    for i, record in enumerate(run["records"]):
        record["runtime"] = {"provenance": {"source_sha256": str(i)}}
    write_json(root / "run.json", run)
    with pytest.raises(ValueError, match="source snapshot"):
        if operation == "freeze":
            freeze_reference(root, tmp_path / "new-reference", "next")
        else:
            compare_reference(reference, root)


def test_citation_version_matches_package():
    root = Path(__file__).resolve().parents[2]
    version = re.search(
        r'^version = "([^"]+)"', (root / "pyproject.toml").read_text(), re.MULTILINE
    )[1]
    citation = (root / "CITATION.cff").read_text()
    assert f"version: {version}\n" in citation
