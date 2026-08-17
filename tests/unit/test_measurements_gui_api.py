# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Tests for the public measurement facade used by the GUI."""

from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np
from numpy.lib import format as npformat
import pytest
from jsonschema import Draft202012Validator

from mmWaveRadar.measurements import (
    compute_radar_products,
    load_bundle,
)
import mmWaveRadar.measurements as measurements_module


REPO_ROOT = Path(__file__).resolve().parents[2]
SMALL_BUNDLE = (
    REPO_ROOT
    / "data"
    / "validation_bundles"
    / "mmradarpose_p1_an0_ac4_r0_frame0240"
)


def test_measurement_bundle_loads_products_and_self_comparison():
    measurement = load_bundle(SMALL_BUNDLE)

    products = measurement.primary_products()
    comparison = measurement.compare_adc(measurement.primary_adc)

    assert measurement.schema_version == 1
    assert measurement.profile == "hermes"
    assert measurement.data_origin == "measurement"
    assert measurement.can_resimulate is True
    assert measurement.integrity_report["valid"] is True
    assert len(measurement.fingerprint) == 64
    assert products.adc.shape == measurement.primary_adc.shape
    assert products.range_time_power.ndim == 2
    assert products.range_doppler_power.ndim == 3
    assert np.isclose(comparison.metrics.range_time_correlation, 1.0)
    assert np.isclose(comparison.metrics.range_doppler_correlation, 1.0)
    assert np.isclose(comparison.metrics.complex_correlation, 1.0)
    assert np.isclose(comparison.metrics.scale_adjusted_complex_nmse, 0.0)
    assert np.isclose(comparison.metrics.peak_range_error_m, 0.0)


def test_measurement_comparison_can_select_channel_zero():
    measurement = load_bundle(SMALL_BUNDLE)
    full_adc = np.asarray(measurement.primary_adc)
    selected_adc = full_adc[..., :1]

    selected_candidate = measurement.compare_adc(
        selected_adc,
        channel_indices=(0,),
    )
    full_candidate = measurement.compare_adc(
        full_adc,
        channel_indices=(0,),
    )

    assert selected_candidate.primary.adc.shape[-1] == 1
    assert selected_candidate.candidate.adc.shape[-1] == 1
    assert np.isclose(selected_candidate.metrics.complex_correlation, 1.0)
    assert full_candidate.primary.adc.shape == selected_candidate.primary.adc.shape
    with pytest.raises(ValueError, match="outside the primary ADC channel axis"):
        measurement.compare_adc(full_adc, channel_indices=(full_adc.shape[-1],))


def test_compute_products_rejects_ambiguous_adc_shape():
    sensor = load_bundle(SMALL_BUNDLE).bundle.sensor_config

    try:
        compute_radar_products(np.zeros((4, 8, 2)), sensor.fmcw)
    except ValueError as exc:
        assert "[frames, chirps, samples, channels]" in str(exc)
    else:
        raise AssertionError("three-dimensional ADC should be rejected")


def test_bundle_json_schemas_are_well_formed_json():
    schema_root = REPO_ROOT / "validation" / "schema"
    names = {
        "bundle.schema.json",
        "sensor.schema.json",
        "frames.schema.json",
        "environment.schema.json",
    }
    assert {path.name for path in schema_root.glob("*.json")} == names
    for name in names:
        schema = json.loads((schema_root / name).read_text(encoding="utf-8"))
        assert schema["$schema"].endswith("2020-12/schema")
        Draft202012Validator.check_schema(schema)


def test_committed_validation_bundles_conform_to_normative_schemas():
    schema_root = REPO_ROOT / "validation" / "schema"
    mappings = {
        "bundle.json": "bundle.schema.json",
        "sensor.json": "sensor.schema.json",
        "frames.json": "frames.schema.json",
        "environment.json": "environment.schema.json",
    }
    validators = {
        document: Draft202012Validator(
            json.loads((schema_root / schema).read_text(encoding="utf-8"))
        )
        for document, schema in mappings.items()
    }
    bundle_roots = sorted(
        path
        for path in (REPO_ROOT / "data" / "validation_bundles").glob("*")
        if path.is_dir()
    )

    assert bundle_roots
    for bundle_root in bundle_roots:
        for document_name, validator in validators.items():
            document_path = bundle_root / document_name
            if document_name == "environment.json" and not document_path.exists():
                continue
            document = json.loads(document_path.read_text(encoding="utf-8"))
            errors = list(validator.iter_errors(document))
            assert not errors, (
                f"{document_path} violates its normative schema: "
                f"{errors[0].message}"
            )


def test_bundle_zip_extraction_enforces_member_and_size_limits(
    tmp_path,
    monkeypatch,
):
    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr("bundle.json", "{}")
        archive.writestr("sensor.json", "{}")

    monkeypatch.setattr(measurements_module, "_MAX_BUNDLE_ZIP_MEMBERS", 1)
    with pytest.raises(ValueError, match="too many members"):
        measurements_module._extract_bundle_zip(
            archive_path,
            tmp_path / "member-limit",
        )

    monkeypatch.setattr(measurements_module, "_MAX_BUNDLE_ZIP_MEMBERS", 10)
    monkeypatch.setattr(measurements_module, "_MAX_BUNDLE_ZIP_TOTAL_BYTES", 1)
    with pytest.raises(ValueError, match="uncompressed size"):
        measurements_module._extract_bundle_zip(
            archive_path,
            tmp_path / "size-limit",
        )


def test_bundle_zip_extraction_rejects_extreme_compression_ratio(
    tmp_path,
    monkeypatch,
):
    archive_path = tmp_path / "compressed.zip"
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr("large.txt", "0" * 100_000)

    monkeypatch.setattr(
        measurements_module,
        "_MAX_BUNDLE_ZIP_COMPRESSION_RATIO",
        2.0,
    )
    with pytest.raises(ValueError, match="unsafe compression ratio"):
        measurements_module._extract_bundle_zip(
            archive_path,
            tmp_path / "ratio-limit",
        )


def test_gui_bundle_preflight_rejects_declared_npz_bomb_before_integrity(
    tmp_path,
    monkeypatch,
):
    member = BytesIO()
    npformat.write_array_header_1_0(
        member,
        {
            "descr": np.dtype("<f8").str,
            "fortran_order": False,
            "shape": (64_000_001,),
        },
    )
    with zipfile.ZipFile(tmp_path / "bomb.npz", "w") as archive:
        archive.writestr("adc.npy", member.getvalue())
    monkeypatch.setattr(
        measurements_module,
        "check_bundle_integrity",
        lambda _root: pytest.fail("integrity must not run before NPZ preflight"),
    )

    with pytest.raises(ValueError, match="too many elements"):
        load_bundle(
            tmp_path,
            resource_limits=measurements_module.GUI_BUNDLE_RESOURCE_LIMITS,
        )


def test_gui_bundle_preflight_caps_arrays_across_multiple_npz_files(tmp_path):
    np.savez(tmp_path / "one.npz", value=np.zeros(4, dtype=np.float32))
    np.savez(tmp_path / "two.npz", value=np.zeros(4, dtype=np.float32))
    limits = replace(
        measurements_module.GUI_BUNDLE_RESOURCE_LIMITS,
        max_npz_array_total_bytes=16,
    )

    with pytest.raises(ValueError, match="aggregate byte limit"):
        load_bundle(tmp_path, resource_limits=limits)


def test_gui_bundle_preflight_caps_ratio_across_multiple_npz_files(tmp_path):
    for name in ("one", "two"):
        np.savez_compressed(
            tmp_path / f"{name}.npz",
            value=np.zeros(9 * 1024**2, dtype=np.uint8),
        )

    with pytest.raises(ValueError, match="unsafe aggregate compression ratio"):
        load_bundle(
            tmp_path,
            resource_limits=measurements_module.GUI_BUNDLE_RESOURCE_LIMITS,
        )


def test_gui_bundle_preflight_caps_json_before_parsing(tmp_path, monkeypatch):
    (tmp_path / "bundle.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        measurements_module,
        "check_bundle_integrity",
        lambda _root: pytest.fail("JSON must be bounded before parsing"),
    )
    limits = replace(
        measurements_module.GUI_BUNDLE_RESOURCE_LIMITS,
        max_json_file_bytes=1,
    )

    with pytest.raises(ValueError, match="JSON file exceeds the size limit"):
        load_bundle(tmp_path, resource_limits=limits)


def test_gui_outer_bundle_zip_uses_strict_resource_policy(tmp_path):
    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("bundle.json", "{}")
    limits = replace(
        measurements_module.GUI_BUNDLE_RESOURCE_LIMITS,
        max_zip_total_bytes=1,
    )

    with pytest.raises(ValueError, match="uncompressed size exceeds"):
        measurements_module._extract_bundle_zip(
            archive_path,
            tmp_path / "strict-extract",
            limits=limits,
        )


def test_trusted_saved_adc_none_limits_use_default_preflight(tmp_path):
    path = tmp_path / "saved.npz"
    expected = np.zeros((1, 2, 3, 4), dtype=np.complex64)
    np.savez(path, adc=expected)

    loaded = measurements_module.load_simulated_adc(path, npz_limits=None)

    np.testing.assert_array_equal(loaded, expected)


def test_gui_bundle_rejects_adc_path_alias_before_integrity(
    tmp_path,
    monkeypatch,
):
    np.savez(tmp_path / "radar_adc.npz", adc=np.zeros((1, 1, 1, 1)))
    descriptor = {
        "profile": "hermes",
        "schema_version": 1,
        "primary_adc": "primary",
        "adcs": {
            "primary": {"path": "radar_adc.npz", "origin": "measurement"},
            "alias": {"path": "radar_adc.npz", "origin": "simulation"},
        },
    }
    (tmp_path / "bundle.json").write_text(
        json.dumps(descriptor),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        measurements_module,
        "check_bundle_integrity",
        lambda _root: pytest.fail("integrity must not run for aliased ADCs"),
    )

    with pytest.raises(ValueError, match="unique paths"):
        load_bundle(
            tmp_path,
            resource_limits=measurements_module.GUI_BUNDLE_RESOURCE_LIMITS,
        )
