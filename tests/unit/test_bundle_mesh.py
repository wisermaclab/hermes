# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


PUBLIC_ROOT = Path(__file__).resolve().parents[2]


def _load_tool():
    path = PUBLIC_ROOT / "tools/bundle_prepare/_mesh.py"
    spec = importlib.util.spec_from_file_location("bundle_mesh_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bake_smpl_motion_mesh_interpolates_and_removes_fitted_shape(tmp_path):
    module = _load_tool()
    observed = {}

    def fake_evaluator(path, _model_dir, *, out_npz_path):
        assert out_npz_path is None
        with np.load(path, allow_pickle=False) as archive:
            observed.update(
                poses=np.array(archive["poses"], copy=True),
                trans=np.array(archive["trans"], copy=True),
                betas=np.array(archive["betas"], copy=True),
                gender=archive["gender"].item(),
            )
        base = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=np.float32,
        )
        vertices = np.stack(
            [base + translation for translation in observed["trans"]],
            axis=0,
        )
        faces = np.asarray(
            [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
            dtype=np.uint32,
        )
        return faces, vertices, np.arange(len(vertices), dtype=np.float64)

    vertices, faces, times = module.bake_smpl_motion_mesh(
        poses=np.zeros((3, 72), dtype=np.float32),
        translations=np.asarray(
            [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        betas=np.arange(10, dtype=np.float32),
        gender="female",
        relative_times=np.asarray([-0.1, 0.0, 0.1]),
        acquisition_duration_s=0.05,
        samples=3,
        smpl_model_dir=tmp_path,
        output_axes=np.asarray([0, 1, 2]),
        output_signs=np.ones(3),
        neutral_shape=True,
        mesh_evaluator=fake_evaluator,
    )

    assert observed["gender"] == "neutral"
    assert np.count_nonzero(observed["betas"]) == 0
    assert observed["trans"][:, 0].tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert times.tolist() == pytest.approx([0.0, 0.025, 0.05])
    assert vertices.shape == (3, 4, 3)
    assert faces.shape == (4, 3)


def test_bake_smpl_motion_mesh_requires_post_center_coverage(tmp_path):
    module = _load_tool()

    with pytest.raises(ValueError, match="does not cover"):
        module.bake_smpl_motion_mesh(
            poses=np.zeros((2, 72), dtype=np.float32),
            translations=np.zeros((2, 3), dtype=np.float32),
            betas=np.zeros(10, dtype=np.float32),
            gender="neutral",
            relative_times=np.asarray([-0.1, 0.01]),
            acquisition_duration_s=0.05,
            samples=3,
            smpl_model_dir=tmp_path,
            output_axes=np.asarray([0, 1, 2]),
            output_signs=np.ones(3),
            mesh_evaluator=lambda *_args, **_kwargs: None,
        )
