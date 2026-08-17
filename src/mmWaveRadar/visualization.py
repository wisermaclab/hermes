# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Visualization helpers for mmWave radar ray-tracing notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


_INVALID_INDEX = np.uint32(0xFFFFFFFF)


@dataclass(frozen=True)
class TopPathSelection:
    """Top path indices and their aggregate magnitudes."""

    indices: np.ndarray
    magnitudes: np.ndarray


def _valid_by_path(paths) -> np.ndarray:
    """Reduces Sionna path validity over all non-path axes."""

    valid = np.asarray(paths.valid.numpy(), dtype=bool)
    return np.any(valid.reshape(-1, valid.shape[-1]), axis=0)


def _path_hits_object(paths, object_ids: Iterable[int]) -> np.ndarray:
    """Returns a per-path mask for paths that touch any requested object ID."""

    ids = {int(object_id) for object_id in object_ids}
    objects = np.asarray(paths.objects.numpy())
    valid = np.asarray(paths.valid.numpy(), dtype=bool)
    num_paths = valid.shape[-1]
    hits = np.zeros((num_paths,), dtype=bool)
    for object_id in ids:
        object_hits = objects == object_id
        object_hits = object_hits.reshape(-1, num_paths)
        hits |= np.any(object_hits, axis=0)
    return hits


def top_path_indices(paths, k: int = 8, *,
                     required_object_ids: Iterable[int] = (),
                     excluded_object_ids: Iterable[int] = ()) -> TopPathSelection:
    """
    Selects the strongest path indices based on aggregate CIR magnitude.

    Magnitudes are reduced over receivers, transmitters, antenna elements, and
    time samples so each geometrical path gets one score.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return TopPathSelection(
            indices=np.empty((0,), dtype=int),
            magnitudes=np.empty((0,), dtype=float))

    a, _ = paths.cir(normalize_delays=False, out_type="numpy")
    a = np.asarray(a)
    if a.ndim < 2:
        return TopPathSelection(
            indices=np.empty((0,), dtype=int),
            magnitudes=np.empty((0,), dtype=float))

    path_axis = a.ndim - 2
    reduce_axes = tuple(i for i in range(a.ndim) if i != path_axis)
    magnitudes = np.max(np.abs(a), axis=reduce_axes)

    valid_by_path = _valid_by_path(paths)
    if valid_by_path.shape[0] != magnitudes.shape[0]:
        raise RuntimeError("Path validity and CIR path dimensions differ")
    keep = valid_by_path.copy()
    required_object_ids = tuple(required_object_ids)
    if required_object_ids:
        keep &= _path_hits_object(paths, required_object_ids)
    excluded_object_ids = tuple(excluded_object_ids)
    if excluded_object_ids:
        keep &= ~_path_hits_object(paths, excluded_object_ids)
    magnitudes = np.where(keep, magnitudes, -np.inf)

    finite = np.isfinite(magnitudes)
    if not np.any(finite):
        return TopPathSelection(
            indices=np.empty((0,), dtype=int),
            magnitudes=np.empty((0,), dtype=float))

    order = np.argsort(magnitudes)[::-1]
    order = order[finite[order]][:k]
    return TopPathSelection(indices=order.astype(int),
                            magnitudes=magnitudes[order].astype(float))


def path_lengths_m(paths, selection: TopPathSelection) -> np.ndarray:
    """Returns propagation lengths in meters for selected path indices."""
    if selection.indices.size == 0:
        return np.empty((0,), dtype=float)

    tau = np.asarray(paths.tau.numpy(), dtype=float)
    valid = np.asarray(paths.valid.numpy(), dtype=bool)
    num_paths = tau.shape[-1]
    tau = tau.reshape(-1, num_paths)
    valid = valid.reshape(-1, num_paths)

    lengths = []
    for path_index in selection.indices:
        if path_index >= num_paths:
            lengths.append(np.nan)
            continue
        mask = valid[:, path_index] & np.isfinite(tau[:, path_index]) \
            & (tau[:, path_index] >= 0.0)
        if not np.any(mask):
            lengths.append(np.nan)
            continue
        lengths.append(float(np.mean(tau[mask, path_index]) * 299792458.0))
    return np.asarray(lengths, dtype=float)


def _synthetic_path_components(paths):
    """Extracts path geometry arrays from synthetic-array Sionna paths."""

    if not paths.synthetic_array:
        raise NotImplementedError(
            "Top-path mesh highlighting currently expects synthetic_array=True")
    return (
        np.asarray(paths.vertices.numpy(), dtype=float),
        np.asarray(paths.valid.numpy(), dtype=bool),
        np.asarray(paths.interactions.numpy()),
        np.asarray(paths.objects.numpy()),
        np.asarray(paths.primitives.numpy()),
    )


def top_path_segments(paths, selection: TopPathSelection, *,
                      line_color=None):
    """
    Returns line segments for the selected top paths.

    The color convention follows Sionna's preview path colors.
    """
    from sionna.rt.constants import INTERACTION_TYPE_TO_COLOR, InteractionType

    vertices, valid, interactions, _, _ = _synthetic_path_components(paths)
    max_depth = vertices.shape[0]
    src_positions = np.asarray(paths.sources.numpy()).T
    tgt_positions = np.asarray(paths.targets.numpy()).T

    line_color = (None if line_color is None
                  else np.asarray(line_color, dtype=float))
    starts, ends, colors = [], [], []
    selected = set(int(i) for i in selection.indices)
    for rx in range(valid.shape[0]):
        for tx in range(valid.shape[1]):
            for path_index in selected:
                if path_index >= valid.shape[2] or not valid[rx, tx, path_index]:
                    continue
                start = src_positions[tx]
                color = INTERACTION_TYPE_TO_COLOR[None]
                for depth in range(max_depth):
                    interaction = interactions[depth, rx, tx, path_index]
                    if interaction == InteractionType.NONE:
                        break
                    end = vertices[depth, rx, tx, path_index]
                    starts.append(start)
                    ends.append(end)
                    colors.append(color if line_color is None else line_color)
                    start = end
                    color = INTERACTION_TYPE_TO_COLOR.get(int(interaction),
                                                          color)
                starts.append(start)
                ends.append(tgt_positions[rx])
                colors.append(color if line_color is None else line_color)

    if not starts:
        empty = np.empty((0, 3), dtype=float)
        return empty, empty, empty

    return (np.vstack(starts).astype(float),
            np.vstack(ends).astype(float),
            np.vstack(colors).astype(float))


def mesh_face_hits(paths, targets: Iterable, selection: TopPathSelection):
    """Returns intersected face indices for each selected mesh target."""
    _, valid, _, objects, primitives = _synthetic_path_components(paths)
    selected = set(int(i) for i in selection.indices)
    result = {}

    for target in targets:
        scene_object = target.ensure_scene_object()
        object_id = int(np.asarray(scene_object.object_id).reshape(-1)[0])
        hits = set()
        for depth in range(objects.shape[0]):
            for rx in range(valid.shape[0]):
                for tx in range(valid.shape[1]):
                    for path_index in selected:
                        if (path_index >= valid.shape[2]
                                or not valid[rx, tx, path_index]):
                            continue
                        if int(objects[depth, rx, tx, path_index]) != object_id:
                            continue
                        primitive = primitives[depth, rx, tx, path_index]
                        if primitive != _INVALID_INDEX:
                            hits.add(int(primitive))
        result[target.name] = np.array(sorted(hits), dtype=int)

    return result


def target_object_ids(targets: Iterable) -> list[int]:
    """Returns Sionna object IDs for mesh targets."""
    return [
        int(np.asarray(target.ensure_scene_object().object_id).reshape(-1)[0])
        for target in targets
    ]


def preview_top_paths(scene, paths, targets: Iterable = (), *,
                      top_k: int = 8, clip_at: float | None = None,
                      resolution=(900, 650), line_width: float = 4.0,
                      face_color=(1.0, 0.35, 0.05),
                      target_path_color=None,
                      environment_top_k: int = 0,
                      environment_path_color=(0.05, 0.45, 1.0),
                      mesh_time: float | None = None,
                      require_target_hit: bool = False):
    """
    Shows the scene with Sionna's preview widget and overlays top-K paths.

    This relies on Sionna's preview widget for scene/device rendering and its
    internal line/mesh plotting helpers for the selected path overlay.
    """
    targets = list(targets)
    target_ids = target_object_ids(targets)
    required_ids = target_ids if require_target_hit else ()
    selection = top_path_indices(paths, top_k,
                                 required_object_ids=required_ids)
    environment_selection = top_path_indices(
        paths,
        environment_top_k,
        excluded_object_ids=target_ids,
    ) if environment_top_k > 0 else TopPathSelection(
        indices=np.empty((0,), dtype=int),
        magnitudes=np.empty((0,), dtype=float))
    scene.preview(clip_at=clip_at, paths=None, resolution=resolution,
                  show_devices=True, point_picker=True)

    widget = scene._preview_widget  # pylint: disable=protected-access
    starts, ends, colors = top_path_segments(
        paths,
        selection,
        line_color=target_path_color)
    if starts.size:
        widget._plot_lines(starts, ends, colors, line_width)  # pylint: disable=protected-access
    starts, ends, colors = top_path_segments(
        paths,
        environment_selection,
        line_color=environment_path_color)
    if starts.size:
        widget._plot_lines(starts, ends, colors, line_width)  # pylint: disable=protected-access

    hits = mesh_face_hits(paths, targets, selection)
    for target in targets:
        face_indices = hits.get(target.name, np.empty((0,), dtype=int))
        if face_indices.size == 0:
            continue
        time = (float(target.mesh_sequence.times[0]) if mesh_time is None
                else float(mesh_time))
        vertices = np.asarray(target.update_to_time(time))
        faces = np.asarray(target.mesh_sequence.faces[face_indices], dtype=np.uint32)
        widget._plot_mesh(vertices, faces, persist=False,  # pylint: disable=protected-access
                          colors=np.asarray(face_color, dtype=np.float32))

    return selection, hits


def plot_top_paths_with_mesh_hits(paths, targets: Iterable, *,
                                  top_k: int = 8, ax=None,
                                  mesh_time: float | None = None,
                                  target_path_color=(1.0, 0.35, 0.05),
                                  environment_top_k: int = 0,
                                  environment_path_color=(0.05, 0.45, 1.0),
                                  require_target_hit: bool = False):
    """Creates a static Matplotlib 3D plot of top-K paths and mesh face hits."""
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # pylint: disable=import-outside-toplevel

    targets = list(targets)
    target_ids = target_object_ids(targets)
    required_ids = target_ids if require_target_hit else ()
    selection = top_path_indices(paths, top_k,
                                 required_object_ids=required_ids)
    environment_selection = top_path_indices(
        paths,
        environment_top_k,
        excluded_object_ids=target_ids,
    ) if environment_top_k > 0 else TopPathSelection(
        indices=np.empty((0,), dtype=int),
        magnitudes=np.empty((0,), dtype=float))
    starts, ends, colors = top_path_segments(
        paths,
        selection,
        line_color=target_path_color)
    env_starts, env_ends, env_colors = top_path_segments(
        paths,
        environment_selection,
        line_color=environment_path_color)
    hits = mesh_face_hits(paths, targets, selection)

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    all_points = []
    for target in targets:
        time = (float(target.mesh_sequence.times[0]) if mesh_time is None
                else float(mesh_time))
        vertices = np.asarray(target.mesh_sequence.vertices_at(
            time))
        faces = np.asarray(target.mesh_sequence.faces)
        hit_faces = hits.get(target.name, np.empty((0,), dtype=int))

        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                   s=3, color="#9a9a9a", alpha=0.18)
        all_points.append(vertices)
        if hit_faces.size:
            polys = vertices[faces[hit_faces]]
            coll = Poly3DCollection(polys, facecolor="#ff6f3c",
                                    edgecolor="#6b1d12", alpha=0.75)
            ax.add_collection3d(coll)

    for start, end, color in zip(starts, ends, colors):
        segment = np.vstack([start, end])
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                color=color, linewidth=2.4, alpha=0.95)
        all_points.append(segment)
    for start, end, color in zip(env_starts, env_ends, env_colors):
        segment = np.vstack([start, end])
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                color=color, linewidth=2.0, alpha=0.75, linestyle="--")
        all_points.append(segment)

    if all_points:
        points = np.vstack(all_points)
        center = 0.5 * (points.min(axis=0) + points.max(axis=0))
        radius = 0.5 * np.max(points.max(axis=0) - points.min(axis=0))
        radius = max(radius, 0.25)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Top target paths, environment paths, and mesh face hits")
    return fig, ax, selection, hits
