# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Tests for the geometry-only radial velocity diagnostic."""

import json

import numpy as np
import pytest

from mmWaveRadar.diagnostics.radial_velocity_geometry import (
    RadialVelocityDiagnosticConfig,
    main as radial_velocity_main,
    parse_positions_matrix,
    parse_vector3,
    run_radial_velocity_diagnostic,
)


def _write_test_mesh_sequence(path):
    vertices = np.array([
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
        [[1.1, 0.0, 0.0], [1.1, 1.0, 0.0], [1.1, 0.0, 1.0]],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    times = np.array([0.0, 0.1], dtype=float)
    np.savez_compressed(path, vertices=vertices, faces=faces, times=times)


def test_radial_velocity_diagnostic_mesh_sequence(tmp_path):
    mesh_npz = tmp_path / "motion_mesh.npz"
    _write_test_mesh_sequence(mesh_npz)

    result = run_radial_velocity_diagnostic(
        RadialVelocityDiagnosticConfig(
            mesh_sequence_npz=mesh_npz,
            radar_position=(0.0, 0.0, 0.0),
            tx_positions=np.array([[0.0, 0.0, 0.0]]),
            rx_positions=np.array([[0.0, 0.0, 0.0]]),
            point_kinds=("face_centroid",),
        )
    )

    assert result.source_path == mesh_npz
    assert result.samples["face_centroid"].radial_velocity_mps.shape == (1, 1, 1)
    assert result.summaries["face_centroid"]["signed_max_mps"] > 0.0


def test_radial_velocity_diagnostic_cli_writes_json(tmp_path):
    mesh_npz = tmp_path / "motion_mesh.npz"
    report_json = tmp_path / "report.json"
    _write_test_mesh_sequence(mesh_npz)

    assert radial_velocity_main([
        "--mesh-sequence-npz",
        str(mesh_npz),
        "--radar-position",
        "0,0,0",
        "--tx-positions",
        "0,0,0",
        "--rx-positions",
        "0,0,0",
        "--point-kind",
        "face_centroid",
        "--json",
        str(report_json),
    ]) == 0

    report = json.loads(report_json.read_text())
    assert report["source_path"] == str(mesh_npz)
    assert "face_centroid" in report["summaries"]


def test_radial_velocity_diagnostic_parse_helpers():
    assert parse_vector3("1, 2, 3") == (1.0, 2.0, 3.0)
    assert np.allclose(
        parse_positions_matrix("0,0,0; 1,2,3"),
        np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]),
    )


@pytest.mark.parametrize(
    ("option", "value"),
    (
        ("--radar-position", "1,2"),
        ("--radar-orientation", "1,two,3"),
        ("--look-at", "1,2,3,4"),
        ("--tx-positions", "0,0;1,2,3"),
        ("--rx-positions", "0,0,not-a-number"),
    ),
)
def test_radial_velocity_cli_reports_malformed_vectors_as_usage_errors(
    option,
    value,
    capsys,
):
    with pytest.raises(SystemExit) as exc:
        radial_velocity_main([
            "--mesh-sequence-npz",
            "missing.npz",
            option,
            value,
        ])

    error = capsys.readouterr().err
    assert exc.value.code == 2
    assert f"argument {option}" in error
    assert "expected" in error
    assert "Traceback" not in error
