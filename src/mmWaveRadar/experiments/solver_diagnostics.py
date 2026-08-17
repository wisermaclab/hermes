# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Compact, GUI-friendly diagnostics for RT paths and PO mesh facets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..visualization import (
    path_lengths_m,
    top_path_indices,
    top_path_segments,
)


_INVALID_INDEX = np.uint32(0xFFFFFFFF)


@dataclass(frozen=True)
class RTPathDiagnostics:
    """Strongest-path geometry and aggregate path-depth statistics."""

    selected_path_indices: np.ndarray
    selected_path_magnitudes: np.ndarray
    selected_path_lengths_m: np.ndarray
    segment_starts_m: np.ndarray
    segment_ends_m: np.ndarray
    segment_colors_rgb: np.ndarray
    depth_histogram: np.ndarray
    valid_link_path_count: int


@dataclass(frozen=True)
class POSurfaceDiagnostics:
    """Strongest PO facets and their geometry/visibility statistics."""

    face_indices: np.ndarray
    face_centroids_m: np.ndarray
    face_normals: np.ndarray
    face_areas_m2: np.ndarray
    face_power: np.ndarray
    face_visibility: np.ndarray
    visible_face_count: int


@dataclass(frozen=True)
class SolverDiagnostics:
    """Paired RT and PO diagnostics for one simulated target state."""

    rt: RTPathDiagnostics
    po: POSurfaceDiagnostics


def _numpy(value) -> np.ndarray:
    """Convert NumPy, Dr.Jit, or tensor-like diagnostic fields to NumPy."""

    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def summarize_rt_paths(
    paths,
    *,
    top_k: int = 12,
    required_object_ids: Iterable[int] = (),
    excluded_object_ids: Iterable[int] = (),
) -> RTPathDiagnostics:
    """Select strongest RT paths and retain line geometry for GUI overlays."""

    selection = top_path_indices(
        paths,
        k=top_k,
        required_object_ids=required_object_ids,
        excluded_object_ids=excluded_object_ids,
    )
    starts, ends, colors = top_path_segments(paths, selection)
    valid = _numpy(paths.valid).astype(bool, copy=False)
    valid_link_path_count = int(np.count_nonzero(valid))

    objects = _numpy(paths.objects)
    if objects.ndim < 2 or valid.ndim < 1:
        depth_histogram = np.zeros((1,), dtype=np.int64)
    else:
        path_axis_size = valid.shape[-1]
        flat_valid = valid.reshape((-1, path_axis_size))
        flat_objects = objects.reshape((objects.shape[0], -1, path_axis_size))
        depths = np.sum(flat_objects != _INVALID_INDEX, axis=0)
        valid_depths = depths[flat_valid]
        depth_histogram = np.bincount(
            np.asarray(valid_depths, dtype=np.int64),
            minlength=objects.shape[0] + 1,
        ).astype(np.int64, copy=False)

    return RTPathDiagnostics(
        selected_path_indices=np.asarray(selection.indices, dtype=np.int64),
        selected_path_magnitudes=np.asarray(
            selection.magnitudes,
            dtype=float,
        ),
        selected_path_lengths_m=path_lengths_m(paths, selection),
        segment_starts_m=np.asarray(starts, dtype=float).reshape((-1, 3)),
        segment_ends_m=np.asarray(ends, dtype=float).reshape((-1, 3)),
        segment_colors_rgb=np.asarray(colors, dtype=float).reshape((-1, 3)),
        depth_histogram=depth_histogram,
        valid_link_path_count=valid_link_path_count,
    )


def summarize_po_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_power: np.ndarray,
    face_visibility: np.ndarray,
    *,
    top_k: int = 12,
) -> POSurfaceDiagnostics:
    """Return the strongest PO facets with stable, original-mesh indices."""

    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    power = np.asarray(face_power, dtype=float)
    visibility = np.asarray(face_visibility, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape [V, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape [F, 3]")
    if power.shape != (faces.shape[0],):
        raise ValueError("face_power must have shape [F]")
    if visibility.shape != (faces.shape[0],):
        raise ValueError("face_visibility must have shape [F]")

    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    double_area = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(double_area[:, None], 1e-18)
    centroids = np.mean(triangles, axis=1)
    finite_power = np.where(np.isfinite(power), power, -np.inf)
    order = np.argsort(finite_power, kind="stable")[::-1]
    order = order[np.isfinite(finite_power[order])][:top_k]

    return POSurfaceDiagnostics(
        face_indices=order.astype(np.int64, copy=False),
        face_centroids_m=centroids[order],
        face_normals=normals[order],
        face_areas_m2=0.5 * double_area[order],
        face_power=power[order],
        face_visibility=visibility[order],
        visible_face_count=int(np.count_nonzero(visibility > 0.0)),
    )


__all__ = [
    "POSurfaceDiagnostics",
    "RTPathDiagnostics",
    "SolverDiagnostics",
    "summarize_po_surface",
    "summarize_rt_paths",
]
