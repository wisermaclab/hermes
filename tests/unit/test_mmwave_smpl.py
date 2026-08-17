# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Unit tests for optional mmWave SMPL/AMASS helpers."""

from __future__ import annotations

import numpy as np
import pytest

import mmWaveRadar.smpl as smpl_mod


def test_load_amass_npz_decodes_marker_labels(tmp_path):
    path = tmp_path / "sample_amass.npz"
    np.savez(
        path,
        marker_labels=np.array([b"hip", b"shoulder"]),
        marker_data=np.zeros((2, 2, 3), dtype=np.float32),
    )

    data = smpl_mod.load_amass_npz(str(path))

    assert data["marker_labels"].tolist() == ["hip", "shoulder"]
    assert data["marker_data"].shape == (2, 2, 3)


@pytest.mark.parametrize("gender", [b"male", np.asarray(b"female")])
def test_canonical_gender_decodes_byte_scalars(gender):
    assert smpl_mod._canonical_gender(gender) in {"male", "female"}


def test_load_amass_npz_rejects_pickled_object_arrays(tmp_path):
    path = tmp_path / "unsafe_amass.npz"
    np.savez(path, marker_labels=np.array([{"label": "hip"}], dtype=object))

    with pytest.raises(ValueError, match="pickled object array"):
        smpl_mod.load_amass_npz(str(path))


def test_visualize_amass_markers_validates_shape():
    with pytest.raises(ValueError, match="marker_data"):
        smpl_mod.visualize_amass_markers({"marker_data": np.zeros((2, 2))})


def test_amass_to_mesh_npz_reports_missing_optional_dependencies(monkeypatch):
    def fake_import_module(name):
        if name in {"torch", "smplx"}:
            raise ImportError(name)
        raise AssertionError(f"Unexpected import request: {name}")

    monkeypatch.setattr(smpl_mod.importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match="torch' and 'smplx"):
        smpl_mod.amass_to_mesh_npz("dummy_motion.npz", "dummy_models")


def test_pose_parameter_slices_use_standard_amass_smplh_layout():
    layout = smpl_mod._pose_parameter_slices("smplh", 156)

    assert layout["body_pose"] == slice(3, 66)
    assert layout["left_hand_pose"] == slice(66, 111)
    assert layout["right_hand_pose"] == slice(111, 156)


def test_pose_parameter_slices_convert_amass_body_to_smpl_safely():
    layout = smpl_mod._pose_parameter_slices("smpl", 156)

    assert layout["body_pose"] == slice(3, 66)
    assert layout["append_zero_hand_joints"] is True


def test_pose_parameter_slices_support_native_smplx_layout():
    layout = smpl_mod._pose_parameter_slices("smplx", 165)

    assert layout["body_pose"] == slice(3, 66)
    assert layout["jaw_pose"] == slice(66, 69)
    assert layout["left_hand_pose"] == slice(75, 120)
    assert layout["right_hand_pose"] == slice(120, 165)


@pytest.mark.parametrize(
    ("model_type", "pose_dim"),
    [("smpl", 71), ("smplh", 72), ("smplx", 160), ("unknown", 72)],
)
def test_pose_parameter_slices_reject_incompatible_layouts(model_type, pose_dim):
    with pytest.raises(ValueError):
        smpl_mod._pose_parameter_slices(model_type, pose_dim)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("poses", np.array([[np.nan, 0.0, 0.0]]), "poses"),
        ("translations", np.array([[0.0, np.inf, 0.0]]), "translations"),
        ("times", np.array([np.nan]), "times"),
    ],
)
def test_amass_pose_interpolation_rejects_nonfinite_inputs(field, value, message):
    values = {
        "poses": np.zeros((1, 3)),
        "translations": np.zeros((1, 3)),
        "times": np.zeros(1),
        "time": 0.0,
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        smpl_mod.interpolate_amass_pose_params(**values)


def test_amass_pose_interpolation_rejects_nonfinite_query_time():
    with pytest.raises(ValueError, match="time must be finite"):
        smpl_mod.interpolate_amass_pose_params(
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            np.zeros(1),
            np.nan,
        )
