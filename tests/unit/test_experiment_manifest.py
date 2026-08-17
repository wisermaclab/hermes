# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Tests for portable GUI/batch experiment manifests."""

from __future__ import annotations

import json

import pytest

from mmWaveRadar.experiments import (
    ExperimentManifest,
    MeasurementReference,
    RadarExperimentConfig,
    SceneExperimentConfig,
    SolverExperimentConfig,
)


def _manifest() -> ExperimentManifest:
    return ExperimentManifest(
        name="measured hybrid comparison",
        scene=SceneExperimentConfig(
            scenario="bundle_environment",
            parameters={"frame": 0},
        ),
        radar=RadarExperimentConfig(
            board_model="IWR6843AOPEVM",
            parameters={"pattern_mode": "cosine30"},
        ),
        solver=SolverExperimentConfig(
            mode="hybrid_rt_po",
            fidelity="preview",
            parameters={"max_depth": 2},
        ),
        measurement=MeasurementReference(
            bundle_id="example",
            bundle_fingerprint="abc123",
            frame_indices=(0,),
        ),
    )


def test_manifest_round_trip_and_fingerprint_are_deterministic(tmp_path):
    manifest = _manifest()
    path = tmp_path / "manifest.json"
    manifest.write_json(path)

    loaded = ExperimentManifest.read_json(path)

    assert loaded == manifest
    assert loaded.fingerprint == manifest.fingerprint
    assert json.loads(manifest.canonical_json()) == manifest.to_dict()
    assert str(tmp_path) not in manifest.canonical_json()


def test_manifest_reader_uses_documented_default_outputs():
    payload = _manifest().to_dict()
    del payload["outputs"]

    loaded = ExperimentManifest.from_dict(payload)

    assert loaded.outputs == (
        "raw_adc",
        "range_profile",
        "range_doppler",
        "diagnostics",
    )


def test_manifest_rejects_unknown_solver_and_duplicate_frames():
    with pytest.raises(ValueError, match="solver.mode"):
        SolverExperimentConfig(mode="unknown")
    with pytest.raises(ValueError, match="duplicates"):
        MeasurementReference(bundle_id="clip", frame_indices=(0, 0))
