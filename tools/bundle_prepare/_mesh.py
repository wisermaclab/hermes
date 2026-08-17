#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Shared, dataset-neutral mesh baking for source-checkout bundle adapters."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Callable

import numpy as np


MeshEvaluator = Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]]


def bake_smpl_motion_mesh(
    *,
    poses: np.ndarray,
    translations: np.ndarray,
    betas: np.ndarray,
    gender: str,
    relative_times: np.ndarray,
    acquisition_duration_s: float,
    samples: int,
    smpl_model_dir: Path,
    output_axes: np.ndarray,
    output_signs: np.ndarray,
    neutral_shape: bool = False,
    mesh_evaluator: MeshEvaluator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate joint-angle motion over one physical radar acquisition.

    Pose rotations are interpolated before evaluating SMPL. The returned mesh
    archive therefore needs no pose, translation, shape, or gender fields.
    """

    source_poses = np.asarray(poses, dtype=np.float32)
    source_translations = np.asarray(translations, dtype=np.float32)
    source_times = np.asarray(relative_times, dtype=np.float64)
    shape = np.asarray(betas, dtype=np.float32).reshape(-1)
    axes = np.asarray(output_axes, dtype=np.int64)
    signs = np.asarray(output_signs, dtype=np.float32)
    count = int(samples)
    duration = float(acquisition_duration_s)

    if source_poses.ndim != 2 or source_poses.shape[0] < 2:
        raise ValueError("At least two source poses are required to bake motion")
    if source_translations.shape != (source_poses.shape[0], 3):
        raise ValueError("translations must have shape (T, 3) matching poses")
    if source_times.shape != (source_poses.shape[0],):
        raise ValueError("relative_times must have shape (T,) matching poses")
    if not (
        np.all(np.isfinite(source_poses))
        and np.all(np.isfinite(source_translations))
        and np.all(np.isfinite(source_times))
        and np.all(np.isfinite(shape))
    ):
        raise ValueError("Motion parameters must contain only finite values")
    if np.any(np.diff(source_times) <= 0.0):
        raise ValueError("relative_times must be strictly increasing")
    if count < 2 or count > 256:
        raise ValueError("mesh samples must be between 2 and 256")
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("acquisition duration must be positive and finite")
    tolerance = 1e-9
    if source_times[0] > tolerance or source_times[-1] < duration - tolerance:
        raise ValueError(
            "Motion window does not cover the radar acquisition: "
            f"source=[{source_times[0]:.9g}, {source_times[-1]:.9g}], "
            f"required=[0, {duration:.9g}]"
        )
    if axes.shape != (3,) or sorted(axes.tolist()) != [0, 1, 2]:
        raise ValueError("output_axes must be a permutation of [0, 1, 2]")
    if signs.shape != (3,) or not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError("output_signs must contain three values chosen from -1 and 1")
    if shape.size < 10:
        raise ValueError("betas must contain at least 10 values")

    from mmWaveRadar.targets.smpl import (
        amass_to_mesh_npz,
        interpolate_amass_pose_params,
    )

    query_times = np.linspace(0.0, duration, count, dtype=np.float64)
    interpolated = [
        interpolate_amass_pose_params(
            source_poses,
            source_translations,
            source_times,
            float(time),
        )
        for time in query_times
    ]
    baked_poses = np.stack([value[0] for value in interpolated], axis=0)
    baked_translations = np.stack([value[1] for value in interpolated], axis=0)
    baked_betas = (
        np.zeros(10, dtype=np.float32)
        if neutral_shape
        else shape[:10].astype(np.float32, copy=True)
    )
    baked_gender = "neutral" if neutral_shape else str(gender).strip().lower()
    if baked_gender not in {"neutral", "male", "female"}:
        raise ValueError("gender must be neutral, male, or female")

    evaluator = mesh_evaluator or amass_to_mesh_npz
    with tempfile.TemporaryDirectory(prefix="hermes_baked_motion_") as tmp:
        archive = Path(tmp) / "motion.npz"
        np.savez_compressed(
            archive,
            poses=baked_poses.astype(np.float32, copy=False),
            trans=baked_translations.astype(np.float32, copy=False),
            betas=baked_betas,
            gender=np.asarray(baked_gender),
            mocap_framerate=np.asarray(1.0, dtype=np.float64),
        )
        faces, vertices, _ = evaluator(
            str(archive),
            str(smpl_model_dir),
            out_npz_path=None,
        )

    transformed_vertices = (
        np.asarray(vertices, dtype=np.float32)[..., axes] * signs
    )
    from mmWaveRadar.targets.mesh import ensure_outward_face_winding

    outward_faces = ensure_outward_face_winding(
        transformed_vertices,
        np.asarray(faces, dtype=np.uint32),
    )
    return transformed_vertices, outward_faces, query_times
