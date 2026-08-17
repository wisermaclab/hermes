# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Static RT path blocking by a moving human mesh."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mitsuba as mi
import numpy as np

from sionna.rt.constants import InteractionType, INVALID_PRIMITIVE, INVALID_SHAPE

from ..radar.hardware import resolve_virtual_channel_pairs
from .channel_gain import apply_path_hardware_gain
from .physical_optics import HumanVisibilityScene


@dataclass(frozen=True)
class StaticPathBank:
    """Static RT paths and segment geometry used for human blocking."""

    coefficients: np.ndarray
    delays_s: np.ndarray
    valid: np.ndarray
    segment_starts: np.ndarray
    segment_ends: np.ndarray
    segment_path_indices: np.ndarray
    path_vertices: np.ndarray | None = None
    path_interactions: np.ndarray | None = None
    path_objects: np.ndarray | None = None
    path_primitives: np.ndarray | None = None


@dataclass(frozen=True)
class PathVisibilityState:
    """Per-static-path visibility state used for temporal fading."""

    was_visible: np.ndarray
    fade_weights: np.ndarray
    latched_visibility: np.ndarray


@dataclass(frozen=True)
class PathVisibilityResult:
    """Human-blocking visibility result for one chirp."""

    raw_visibility: np.ndarray
    visibility_weights: np.ndarray
    blocked_paths: np.ndarray
    state: PathVisibilityState


def extract_static_path_bank(
    paths,
    radar,
    virtual_channel_order: str = "tx_major",
    *,
    min_segment_length_m: float = 1e-9,
) -> StaticPathBank:
    """Extracts static Sionna RT paths as coefficients, delays, and segments."""

    coefficients, delays_s, valid = _path_arrays(
        paths,
        virtual_channel_order=virtual_channel_order,
        virtual_channel_tx_indices=radar.hardware.virtual_tx_indices(),
        virtual_channel_rx_indices=radar.hardware.virtual_rx_indices(),
    )
    coefficients = apply_path_hardware_gain(
        radar.hardware,
        coefficients,
        paths=paths,
        frequency_hz=radar.fmcw.carrier_frequency,
    )
    num_paths = int(delays_s.size)
    tx_center = np.mean(radar.world_tx_positions(), axis=0)
    rx_center = np.mean(radar.world_rx_positions(), axis=0)
    vertices = _path_vertices(paths)
    interactions = _path_interactions(paths, num_paths)
    objects = _path_objects(paths, interactions.shape, num_paths)
    primitives = _path_primitives(paths, interactions.shape, num_paths)

    starts = []
    ends = []
    path_indices = []
    for path_idx in range(num_paths):
        if not bool(valid[path_idx]):
            continue
        hit_mask = interactions[:, path_idx] != InteractionType.NONE
        hit_vertices = vertices[hit_mask, path_idx, :]
        finite = np.all(np.isfinite(hit_vertices), axis=1)
        hit_vertices = hit_vertices[finite]
        points = [tx_center]
        points.extend(np.asarray(v, dtype=np.float64) for v in hit_vertices)
        points.append(rx_center)
        for start, end in zip(points[:-1], points[1:]):
            start = np.asarray(start, dtype=np.float64).reshape(3)
            end = np.asarray(end, dtype=np.float64).reshape(3)
            if np.linalg.norm(end - start) <= float(min_segment_length_m):
                continue
            starts.append(start)
            ends.append(end)
            path_indices.append(path_idx)

    return StaticPathBank(
        coefficients=coefficients,
        delays_s=delays_s,
        valid=valid,
        segment_starts=np.asarray(starts, dtype=np.float64).reshape((-1, 3)),
        segment_ends=np.asarray(ends, dtype=np.float64).reshape((-1, 3)),
        segment_path_indices=np.asarray(path_indices, dtype=np.int64),
        path_vertices=vertices,
        path_interactions=interactions,
        path_objects=objects,
        path_primitives=primitives,
    )


def blocked_path_visibility(
    bank: StaticPathBank,
    visibility_scene: HumanVisibilityScene,
    *,
    blocker_vertices: np.ndarray | None = None,
    aabb_culling: bool = True,
    aabb_margin_m: float = 0.02,
    state: PathVisibilityState | None = None,
    fade_chirps: int = 8,
    ray_epsilon_m: float = 1e-5,
    ray_chunk_size: int = 65536,
) -> PathVisibilityResult:
    """Computes per-path human blocking with temporal fading."""

    raw_visibility = np.ones(bank.delays_s.shape, dtype=np.float64)
    raw_visibility[~bank.valid] = 0.0
    candidate_segment_indices = None
    if aabb_culling and blocker_vertices is not None:
        candidate_segment_indices = segment_indices_intersecting_aabb(
            bank,
            blocker_vertices,
            margin_m=aabb_margin_m,
        )
    blocked_paths = segment_blocked_paths(
        bank,
        visibility_scene,
        candidate_segment_indices=candidate_segment_indices,
        ray_epsilon_m=ray_epsilon_m,
        ray_chunk_size=ray_chunk_size,
    )
    raw_visibility[blocked_paths] = 0.0
    fade_weights, latched_visibility = update_path_visibility_state(
        raw_visibility,
        state=state,
        fade_chirps=fade_chirps,
    )
    weights = latched_visibility * fade_weights
    return PathVisibilityResult(
        raw_visibility=raw_visibility,
        visibility_weights=weights,
        blocked_paths=blocked_paths,
        state=PathVisibilityState(
            was_visible=raw_visibility > 0.0,
            fade_weights=fade_weights.copy(),
            latched_visibility=latched_visibility.copy(),
        ),
    )


def segment_blocked_paths(
    bank: StaticPathBank,
    visibility_scene: HumanVisibilityScene,
    *,
    candidate_segment_indices: np.ndarray | None = None,
    ray_epsilon_m: float = 1e-5,
    ray_chunk_size: int = 65536,
) -> np.ndarray:
    """Returns a boolean per-path mask for paths blocked by any segment."""

    blocked = np.zeros(bank.delays_s.shape, dtype=bool)
    starts = np.asarray(bank.segment_starts, dtype=np.float64)
    ends = np.asarray(bank.segment_ends, dtype=np.float64)
    if starts.size == 0:
        return blocked
    directions = ends - starts
    distances = np.linalg.norm(directions, axis=1)
    castable = distances > 2.0 * float(ray_epsilon_m)
    if not np.any(castable):
        return blocked

    if candidate_segment_indices is None:
        candidate_indices = np.flatnonzero(castable)
    else:
        candidate_indices = np.asarray(
            candidate_segment_indices,
            dtype=np.int64,
        ).reshape(-1)
        if candidate_indices.size == 0:
            return blocked
        if np.any(candidate_indices < 0) or np.any(candidate_indices >= starts.shape[0]):
            raise ValueError("candidate_segment_indices contains invalid indices")
        candidate_indices = candidate_indices[castable[candidate_indices]]
        if candidate_indices.size == 0:
            return blocked
    chunk_size = max(int(ray_chunk_size), 1)
    for chunk_start in range(0, candidate_indices.size, chunk_size):
        chunk_indices = candidate_indices[chunk_start:chunk_start + chunk_size]
        chunk_directions = directions[chunk_indices]
        chunk_distances = distances[chunk_indices]
        unit_directions = chunk_directions / chunk_distances[:, None]
        origins = starts[chunk_indices] + float(ray_epsilon_m) * unit_directions
        maxt = np.maximum(chunk_distances - 2.0 * float(ray_epsilon_m), 0.0)
        ray = mi.Ray3f(
            o=mi.Point3f(origins[:, 0], origins[:, 1], origins[:, 2]),
            d=mi.Vector3f(
                unit_directions[:, 0],
                unit_directions[:, 1],
                unit_directions[:, 2],
            ),
            maxt=mi.Float(maxt),
            time=0.0,
            wavelengths=mi.Color0f(),
        )
        pi_hit = visibility_scene.scene.ray_intersect_preliminary(ray)
        hit = np.asarray(pi_hit.is_valid(), dtype=bool)
        if np.any(hit):
            blocked[bank.segment_path_indices[chunk_indices[hit]]] = True
    return blocked


def segment_indices_intersecting_aabb(
    bank: StaticPathBank,
    vertices: np.ndarray,
    *,
    margin_m: float = 0.02,
) -> np.ndarray:
    """Returns static path segments whose segment AABB overlaps a mesh AABB."""

    if margin_m < 0.0:
        raise ValueError("margin_m must be non-negative")
    starts = np.asarray(bank.segment_starts, dtype=np.float64)
    ends = np.asarray(bank.segment_ends, dtype=np.float64)
    if starts.size == 0:
        return np.zeros((0,), dtype=np.int64)

    vertices = np.asarray(vertices, dtype=np.float64).reshape((-1, 3))
    finite = np.all(np.isfinite(vertices), axis=1)
    if not np.any(finite):
        return np.zeros((0,), dtype=np.int64)

    margin = float(margin_m)
    aabb_min = np.min(vertices[finite], axis=0) - margin
    aabb_max = np.max(vertices[finite], axis=0) + margin
    segment_min = np.minimum(starts, ends)
    segment_max = np.maximum(starts, ends)
    overlaps = np.all(
        (segment_max >= aabb_min[None, :])
        & (segment_min <= aabb_max[None, :]),
        axis=1,
    )
    return np.flatnonzero(overlaps)


def update_path_visibility_state(
    raw_visibility: np.ndarray,
    *,
    state: PathVisibilityState | None,
    fade_chirps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Updates per-path fade weights using the PO visibility fade semantics."""

    raw_visibility = np.asarray(raw_visibility, dtype=np.float64).reshape(-1)
    raw_visible = raw_visibility > 0.0
    if fade_chirps <= 0:
        raise ValueError("fade_chirps must be positive")
    if state is None:
        fade_weights = raw_visible.astype(np.float64)
        return fade_weights, raw_visibility.copy()

    prev_visible = np.asarray(state.was_visible, dtype=bool).reshape(-1)
    prev_fade = np.asarray(state.fade_weights, dtype=np.float64).reshape(-1)
    prev_latched = np.asarray(
        state.latched_visibility,
        dtype=np.float64,
    ).reshape(-1)
    if (
        prev_visible.shape != raw_visible.shape
        or prev_fade.shape != raw_visible.shape
        or prev_latched.shape != raw_visible.shape
    ):
        raise ValueError("visibility state shape mismatch")

    step = 1.0 / float(fade_chirps)
    fade_weights = prev_fade.copy()
    latched = prev_latched.copy()

    fading_out = ~raw_visible
    fade_weights[fading_out] = np.maximum(prev_fade[fading_out] - step, 0.0)
    latched[fading_out & (prev_fade <= 0.0)] = 0.0

    continuing_visible = raw_visible & prev_visible & (prev_fade >= 1.0 - 1e-12)
    fade_weights[continuing_visible] = 1.0
    ramp_visible = raw_visible & ~continuing_visible
    fade_weights[ramp_visible] = np.minimum(prev_fade[ramp_visible] + step, 1.0)
    latched[raw_visible] = raw_visibility[raw_visible]

    return fade_weights, latched


def synthesize_blocked_static_cube(
    bank: StaticPathBank,
    visibility_weights: np.ndarray,
    times: np.ndarray,
    fmcw,
    synthesize_adc: Callable,
    *,
    backend: str = "numpy",
    precision: str = "float64",
) -> np.ndarray:
    """Synthesizes a static ADC cube from per-chirp path visibility weights."""

    times = np.asarray(times)
    weights = np.asarray(visibility_weights, dtype=np.float64)
    if weights.shape != times.shape + bank.delays_s.shape:
        raise ValueError(
            "visibility_weights must have shape times.shape + [num_paths]"
        )
    num_vc = bank.coefficients.shape[0]
    out = np.zeros(
        times.shape + (int(fmcw.num_adc_samples), num_vc),
        dtype=np.complex64 if precision == "float32" else np.complex128,
    )
    for index in np.ndindex(times.shape):
        weighted_a = bank.coefficients * weights[index][None, :]
        valid = bank.valid & (weights[index] > 0.0)
        out[index] = synthesize_adc(
            weighted_a,
            bank.delays_s,
            valid,
            fmcw,
            backend=backend,
            precision=precision,
        )
    return out


def _path_arrays(
    paths,
    *,
    virtual_channel_order: str,
    virtual_channel_tx_indices: np.ndarray | None = None,
    virtual_channel_rx_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts channel/path coefficients, delays, and validity from Sionna paths."""

    a, tau = paths.cir(normalize_delays=False, out_type="numpy")
    a = np.asarray(a)
    tau = np.asarray(tau)
    if a.ndim != 6:
        raise RuntimeError(
            "Expected CIR coefficients with shape "
            "[rx, rx_ant, tx, tx_ant, paths, time]"
        )
    a = a[0, :, 0, :, :, 0]
    tau = tau[0, 0, :] if tau.ndim == 3 else tau.reshape(-1)
    valid = tau >= 0.0

    vc_tx, vc_rx = resolve_virtual_channel_pairs(
        int(a.shape[1]),
        int(a.shape[0]),
        order=virtual_channel_order,
        tx_indices=virtual_channel_tx_indices,
        rx_indices=virtual_channel_rx_indices,
    )
    a = a[vc_rx, vc_tx, :]

    if a.shape[-1] == 0:
        return (
            np.zeros((vc_tx.size, 0), dtype=np.complex128),
            tau,
            valid,
        )
    return a, tau, valid


def _path_vertices(paths) -> np.ndarray:
    """Normalizes Sionna path-vertex tensors to ``[depth, paths, 3]``."""

    vertices = _to_numpy(paths.vertices)
    if vertices.ndim == 7:
        return np.asarray(vertices[:, 0, 0, 0, 0, :, :], dtype=np.float64)
    if vertices.ndim == 5:
        return np.asarray(vertices[:, 0, 0, :, :], dtype=np.float64)
    if vertices.ndim == 3:
        return np.asarray(vertices, dtype=np.float64)
    raise RuntimeError(f"Unsupported path vertices shape: {vertices.shape}")


def _path_objects(paths, shape: tuple[int, ...], num_paths: int) -> np.ndarray:
    """Returns per-interaction object IDs or an invalid-object placeholder."""

    if not hasattr(paths, "objects"):
        return np.full(shape, INVALID_SHAPE, dtype=np.uint32)
    return _path_component_array(
        getattr(paths, "objects"),
        num_paths=num_paths,
        invalid_value=INVALID_SHAPE,
        name="objects",
    )


def _path_primitives(paths, shape: tuple[int, ...], num_paths: int) -> np.ndarray:
    """Returns per-interaction primitive IDs or an invalid-primitive placeholder."""

    if not hasattr(paths, "primitives"):
        return np.full(shape, INVALID_PRIMITIVE, dtype=np.uint32)
    return _path_component_array(
        getattr(paths, "primitives"),
        num_paths=num_paths,
        invalid_value=INVALID_PRIMITIVE,
        name="primitives",
    )


def _path_component_array(
    value,
    *,
    num_paths: int,
    invalid_value: int,
    name: str,
) -> np.ndarray:
    """Normalizes Sionna path metadata arrays to ``[depth, paths]``."""

    array = _to_numpy(value)
    if array.ndim == 6:
        return np.asarray(array[:, 0, 0, 0, 0, :], dtype=np.uint32)
    if array.ndim == 4:
        return np.asarray(array[:, 0, 0, :], dtype=np.uint32)
    if array.ndim == 2:
        return np.asarray(array, dtype=np.uint32)
    if array.size == 0:
        return np.full((0, num_paths), invalid_value, dtype=np.uint32)
    raise RuntimeError(f"Unsupported path {name} shape: {array.shape}")


def _path_interactions(paths, num_paths: int) -> np.ndarray:
    """Normalizes Sionna interaction-type tensors to ``[depth, paths]``."""

    interactions = _to_numpy(paths.interactions)
    if interactions.ndim == 6:
        return np.asarray(interactions[:, 0, 0, 0, 0, :], dtype=np.uint32)
    if interactions.ndim == 4:
        return np.asarray(interactions[:, 0, 0, :], dtype=np.uint32)
    if interactions.ndim == 2:
        return np.asarray(interactions, dtype=np.uint32)
    if interactions.size == 0:
        return np.zeros((0, num_paths), dtype=np.uint32)
    raise RuntimeError(f"Unsupported path interactions shape: {interactions.shape}")


def _to_numpy(value) -> np.ndarray:
    """Converts tensor-like objects to NumPy arrays."""

    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)
