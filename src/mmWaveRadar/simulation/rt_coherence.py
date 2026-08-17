# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Coherent RT path-bank helpers for dynamic mesh updates."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence
import weakref

import numpy as np
from scipy.constants import c, pi

from sionna.rt.constants import InteractionType, INVALID_PRIMITIVE, INVALID_SHAPE

from ..radar.hardware import resolve_virtual_channel_pairs
from .channel_gain import apply_path_hardware_gain
from .runtime import _torch_cuda_is_available
from .static_blocking import (
    _path_arrays,
    _path_interactions,
    _path_objects,
    _path_primitives,
    _path_vertices,
)
from ..targets import MeshTarget


_TORCH_COHERENT_BANK_CACHE: dict[
    tuple[int, str],
    tuple[weakref.ReferenceType, dict],
] = {}


@dataclass(frozen=True)
class CoherentRTPathBank:
    """Anchor RT paths with enough metadata for small-motion phase updates."""

    coefficients: np.ndarray
    delays_s: np.ndarray
    valid: np.ndarray
    anchor_path_lengths_m: np.ndarray
    anchor_delay_lengths_m: np.ndarray
    anchor_spread_products_m: np.ndarray
    path_vertices: np.ndarray
    path_interactions: np.ndarray
    path_objects: np.ndarray
    path_primitives: np.ndarray
    dynamic_target_indices: np.ndarray
    dynamic_barycentrics: np.ndarray
    target_faces: tuple[np.ndarray, ...]
    tx_positions: np.ndarray
    rx_positions: np.ndarray
    tx_center: np.ndarray
    rx_center: np.ndarray
    virtual_channel_order: str
    frequency_hz: float
    path_signatures: tuple[tuple[tuple[int, int, int], ...], ...]
    virtual_channel_tx_indices: np.ndarray | None = None
    virtual_channel_rx_indices: np.ndarray | None = None


@dataclass(frozen=True)
class CoherentRTPathUpdate:
    """Updated path arrays synthesized from a coherent RT path bank."""

    coefficients: np.ndarray
    delays_s: np.ndarray
    valid: np.ndarray
    path_lengths_m: np.ndarray
    delay_lengths_m: np.ndarray
    spread_products_m: np.ndarray


@dataclass(frozen=True)
class CoherentRTPathTransition:
    """Path correspondence between two consecutive coherent RT banks."""

    matched_old_indices: np.ndarray
    matched_new_indices: np.ndarray
    old_only_indices: np.ndarray
    new_only_indices: np.ndarray


@dataclass(frozen=True)
class CoherentRTTransitionSynthesis:
    """Weighted path arrays and source indices for one transition chirp."""

    coefficients: np.ndarray
    delays_s: np.ndarray
    valid: np.ndarray
    roles: np.ndarray
    source_is_new: np.ndarray
    source_indices: np.ndarray
    alpha: float
    old_weight: float
    new_weight: float


TRANSITION_ROLE_PERSISTENT = 0
TRANSITION_ROLE_DEATH = 1
TRANSITION_ROLE_BIRTH = 2


def extract_coherent_rt_path_bank(
    paths,
    radar,
    targets: Sequence[MeshTarget],
    *,
    virtual_channel_order: str = "tx_major",
    anchor_target_vertices: Sequence[np.ndarray] | None = None,
) -> CoherentRTPathBank:
    """Extracts an anchor RT path bank for coherent dynamic mesh updates."""

    virtual_channel_order = radar.hardware.virtual_channel_order
    vc_tx = radar.hardware.virtual_tx_indices()
    vc_rx = radar.hardware.virtual_rx_indices()
    coefficients, delays_s, valid = _path_arrays(
        paths,
        virtual_channel_order=virtual_channel_order,
        virtual_channel_tx_indices=vc_tx,
        virtual_channel_rx_indices=vc_rx,
    )
    coefficients = apply_path_hardware_gain(
        radar.hardware,
        coefficients,
        paths=paths,
        frequency_hz=radar.fmcw.carrier_frequency,
    )
    num_paths = int(delays_s.size)
    path_vertices = _path_vertices(paths)
    interactions = _path_interactions(paths, num_paths)
    objects = _path_objects(paths, interactions.shape, num_paths)
    primitives = _path_primitives(paths, interactions.shape, num_paths)
    target_ids = _target_object_ids(targets)
    target_faces = tuple(
        np.asarray(target.mesh_sequence.faces, dtype=np.int64)
        for target in targets
    )
    if anchor_target_vertices is None:
        target_anchor_vertices = tuple(
            np.asarray(target.mesh_sequence.vertices[0], dtype=np.float64)
            for target in targets
        )
    else:
        target_anchor_vertices = tuple(
            np.asarray(vertices, dtype=np.float64)
            for vertices in anchor_target_vertices
        )
        if len(target_anchor_vertices) != len(targets):
            raise ValueError("anchor_target_vertices must match targets")

    dynamic_target_indices, dynamic_barycentrics = _dynamic_hit_coordinates(
        path_vertices=path_vertices,
        objects=objects,
        primitives=primitives,
        target_ids=target_ids,
        target_faces=target_faces,
        target_anchor_vertices=target_anchor_vertices,
    )
    tx_positions = np.asarray(radar.world_tx_positions(), dtype=np.float64)
    rx_positions = np.asarray(radar.world_rx_positions(), dtype=np.float64)
    tx_center = np.mean(tx_positions, axis=0)
    rx_center = np.mean(rx_positions, axis=0)
    anchor_lengths, anchor_spreads = _path_lengths_and_spreads(
        path_vertices=path_vertices,
        interactions=interactions,
        primitives=primitives,
        dynamic_target_indices=dynamic_target_indices,
        dynamic_barycentrics=dynamic_barycentrics,
        target_faces=target_faces,
        target_vertices=None,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        virtual_channel_order=virtual_channel_order,
        virtual_channel_tx_indices=vc_tx,
        virtual_channel_rx_indices=vc_rx,
    )
    if anchor_lengths.shape[0] == 1:
        anchor_delay_lengths = anchor_lengths[0]
    else:
        anchor_delay_lengths, _ = _path_lengths_and_spreads(
            path_vertices=path_vertices,
            interactions=interactions,
            primitives=primitives,
            dynamic_target_indices=dynamic_target_indices,
            dynamic_barycentrics=dynamic_barycentrics,
            target_faces=target_faces,
            target_vertices=None,
            tx_positions=tx_center[None, :],
            rx_positions=rx_center[None, :],
            virtual_channel_order="tx_major",
        )
        anchor_delay_lengths = anchor_delay_lengths[0]
    path_signatures = _path_signatures(interactions, objects, primitives, num_paths)
    return CoherentRTPathBank(
        coefficients=coefficients,
        delays_s=delays_s,
        valid=valid,
        anchor_path_lengths_m=anchor_lengths,
        anchor_delay_lengths_m=anchor_delay_lengths,
        anchor_spread_products_m=anchor_spreads,
        path_vertices=path_vertices,
        path_interactions=interactions,
        path_objects=objects,
        path_primitives=primitives,
        dynamic_target_indices=dynamic_target_indices,
        dynamic_barycentrics=dynamic_barycentrics,
        target_faces=target_faces,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        tx_center=tx_center,
        rx_center=rx_center,
        virtual_channel_order=virtual_channel_order,
        frequency_hz=float(radar.fmcw.carrier_frequency),
        path_signatures=path_signatures,
        virtual_channel_tx_indices=vc_tx.copy(),
        virtual_channel_rx_indices=vc_rx.copy(),
    )


def update_coherent_rt_path_bank(
    bank: CoherentRTPathBank,
    target_vertices: Sequence[np.ndarray] | Mapping[int, np.ndarray],
    *,
    frequency_hz: float | None = None,
    backend: str = "numpy",
) -> CoherentRTPathUpdate:
    """Updates path geometry, using CUDA when requested and available."""

    current_vertices = _coerce_target_vertices(target_vertices)
    if _use_cuda_coherent_update(backend):
        import torch  # pylint: disable=import-outside-toplevel

        return _update_coherent_rt_path_bank_torch(
            bank,
            current_vertices,
            frequency_hz=frequency_hz,
            device=torch.device("cuda"),
        )
    if backend not in ("auto", "numpy", "torch"):
        raise ValueError("backend must be 'auto', 'numpy', or 'torch'")
    path_lengths = np.asarray(
        bank.anchor_path_lengths_m,
        dtype=np.float64,
    ).copy()
    delay_lengths = np.asarray(
        bank.anchor_delay_lengths_m,
        dtype=np.float64,
    ).copy()
    spreads = np.asarray(
        bank.anchor_spread_products_m,
        dtype=np.float64,
    ).copy()

    dynamic_paths = np.any(
        np.asarray(bank.dynamic_target_indices, dtype=np.int64) >= 0,
        axis=0,
    )
    path_indices = np.flatnonzero(dynamic_paths)
    if path_indices.size:
        points, active = _current_dynamic_path_points(
            bank,
            current_vertices,
            path_indices,
        )
        (
            first_points,
            last_points,
            internal_lengths,
            internal_spreads,
        ) = _shared_dynamic_path_geometry(points, active)

        tx_positions = np.asarray(bank.tx_positions, dtype=np.float64)
        rx_positions = np.asarray(bank.rx_positions, dtype=np.float64)
        vc_tx, vc_rx = resolve_virtual_channel_pairs(
            int(tx_positions.shape[0]),
            int(rx_positions.shape[0]),
            order=bank.virtual_channel_order,
            tx_indices=bank.virtual_channel_tx_indices,
            rx_indices=bank.virtual_channel_rx_indices,
        )
        updated_lengths, updated_spreads = _dynamic_endpoint_geometry(
            first_points=first_points,
            last_points=last_points,
            internal_lengths=internal_lengths,
            internal_spreads=internal_spreads,
            tx_positions=tx_positions[vc_tx],
            rx_positions=rx_positions[vc_rx],
        )
        path_lengths[:, path_indices] = updated_lengths
        spreads[:, path_indices] = updated_spreads

        center_lengths, _ = _dynamic_endpoint_geometry(
            first_points=first_points,
            last_points=last_points,
            internal_lengths=internal_lengths,
            internal_spreads=internal_spreads,
            tx_positions=np.asarray(bank.tx_center, dtype=np.float64).reshape(1, 3),
            rx_positions=np.asarray(bank.rx_center, dtype=np.float64).reshape(1, 3),
        )
        delay_lengths[path_indices] = center_lengths[0]
    freq = bank.frequency_hz if frequency_hz is None else float(frequency_hz)
    k0 = 2.0 * pi * freq / c
    length_delta = path_lengths - bank.anchor_path_lengths_m
    phase = np.exp(1j * k0 * length_delta)
    spread_ratio = bank.anchor_spread_products_m / np.maximum(spreads, 1e-18)
    coefficients = bank.coefficients * phase * spread_ratio
    return CoherentRTPathUpdate(
        coefficients=coefficients.astype(bank.coefficients.dtype, copy=False),
        delays_s=(delay_lengths / c).astype(bank.delays_s.dtype, copy=False),
        valid=bank.valid.copy(),
        path_lengths_m=path_lengths,
        delay_lengths_m=delay_lengths,
        spread_products_m=spreads,
    )


def match_coherent_rt_path_banks(
    old_bank: CoherentRTPathBank,
    new_bank: CoherentRTPathBank,
    boundary_target_vertices: Sequence[np.ndarray] | Mapping[int, np.ndarray],
    *,
    wavelength_m: float,
    match_delay_tolerance_fraction: float = 0.5,
    match_separation_fraction: float = 0.25,
    backend: str = "numpy",
) -> CoherentRTPathTransition:
    """Matches conservative path identities across consecutive retraces.

    Paths are first bucketed by their interaction/object/primitive signature.
    Within each bucket, delay length at the new retrace boundary is used only
    when the assignment is unambiguous. Ambiguous paths remain unmatched and
    are handled by birth/death fading.
    """

    if wavelength_m <= 0.0:
        raise ValueError("wavelength_m must be positive")
    tolerance_m = float(wavelength_m) * float(match_delay_tolerance_fraction)
    separation_m = float(wavelength_m) * float(match_separation_fraction)
    if tolerance_m < 0.0:
        raise ValueError("match_delay_tolerance_fraction must be non-negative")
    if separation_m < 0.0:
        raise ValueError("match_separation_fraction must be non-negative")

    old_update = update_coherent_rt_path_bank(
        old_bank,
        boundary_target_vertices,
        backend=backend,
    )
    new_update = update_coherent_rt_path_bank(
        new_bank,
        boundary_target_vertices,
        backend=backend,
    )
    old_valid = np.asarray(old_update.valid, dtype=bool)
    new_valid = np.asarray(new_update.valid, dtype=bool)

    old_by_signature = _signature_index_map(old_bank.path_signatures, old_valid)
    new_by_signature = _signature_index_map(new_bank.path_signatures, new_valid)

    matched_old: list[int] = []
    matched_new: list[int] = []
    for signature in sorted(set(old_by_signature) & set(new_by_signature)):
        old_indices = old_by_signature[signature]
        new_indices = new_by_signature[signature]
        if len(old_indices) == 1 and len(new_indices) == 1:
            old_index = old_indices[0]
            new_index = new_indices[0]
            delta = abs(
                float(old_update.delay_lengths_m[old_index])
                - float(new_update.delay_lengths_m[new_index])
            )
            if delta <= tolerance_m:
                matched_old.append(old_index)
                matched_new.append(new_index)
            continue

        pairs = _unambiguous_delay_matches(
            old_indices,
            new_indices,
            np.asarray(old_update.delay_lengths_m, dtype=np.float64),
            np.asarray(new_update.delay_lengths_m, dtype=np.float64),
            tolerance_m=tolerance_m,
            separation_m=separation_m,
        )
        for old_index, new_index in pairs:
            matched_old.append(old_index)
            matched_new.append(new_index)

    matched_old_arr = np.asarray(matched_old, dtype=np.int64)
    matched_new_arr = np.asarray(matched_new, dtype=np.int64)
    old_valid_indices = np.flatnonzero(old_valid)
    new_valid_indices = np.flatnonzero(new_valid)
    old_only = np.setdiff1d(old_valid_indices, matched_old_arr, assume_unique=False)
    new_only = np.setdiff1d(new_valid_indices, matched_new_arr, assume_unique=False)
    return CoherentRTPathTransition(
        matched_old_indices=matched_old_arr,
        matched_new_indices=matched_new_arr,
        old_only_indices=old_only.astype(np.int64, copy=False),
        new_only_indices=new_only.astype(np.int64, copy=False),
    )


def coherent_transition_weights(
    interval_chirps: int,
    interval_offset: int,
) -> tuple[float, float, float]:
    """Returns alpha, old-only fade weight, and new-only fade weight."""

    interval_chirps = max(int(interval_chirps), 1)
    interval_offset = min(max(int(interval_offset), 0), interval_chirps - 1)
    if interval_chirps <= 1:
        alpha = 0.0
    else:
        alpha = interval_offset / float(interval_chirps - 1)
    alpha = min(max(alpha, 0.0), 1.0)
    return alpha, float(np.sqrt(1.0 - alpha)), float(np.sqrt(alpha))


def synthesize_coherent_rt_transition(
    current_update: CoherentRTPathUpdate,
    next_update: CoherentRTPathUpdate | None = None,
    transition: CoherentRTPathTransition | None = None,
    *,
    interval_chirps: int = 1,
    interval_offset: int = 0,
) -> CoherentRTTransitionSynthesis:
    """Builds weighted path arrays for one chirp in a retrace interval."""

    if next_update is None or transition is None:
        indices = np.flatnonzero(np.asarray(current_update.valid, dtype=bool))
        coeffs = current_update.coefficients[:, indices]
        delays = current_update.delays_s[indices]
        valid = np.ones(indices.shape, dtype=bool)
        return CoherentRTTransitionSynthesis(
            coefficients=coeffs,
            delays_s=delays,
            valid=valid,
            roles=np.full(indices.shape, TRANSITION_ROLE_PERSISTENT, dtype=np.int8),
            source_is_new=np.zeros(indices.shape, dtype=bool),
            source_indices=indices.astype(np.int64, copy=False),
            alpha=0.0,
            old_weight=1.0,
            new_weight=0.0,
        )

    alpha, old_weight, new_weight = coherent_transition_weights(
        interval_chirps,
        interval_offset,
    )
    coeff_parts = []
    delay_parts = []
    role_parts = []
    source_is_new_parts = []
    source_index_parts = []

    def append_part(update, indices, *, weight, role, source_is_new):
        """Appends a weighted subset of old/new path-bank entries."""

        indices = np.asarray(indices, dtype=np.int64)
        if indices.size == 0 or weight <= 0.0:
            return
        valid = np.asarray(update.valid, dtype=bool)[indices]
        if not np.any(valid):
            return
        indices = indices[valid]
        coeff_parts.append(update.coefficients[:, indices] * weight)
        delay_parts.append(update.delays_s[indices])
        role_parts.append(np.full(indices.shape, role, dtype=np.int8))
        source_is_new_parts.append(np.full(indices.shape, source_is_new, dtype=bool))
        source_index_parts.append(indices)

    append_part(
        current_update,
        transition.matched_old_indices,
        weight=1.0,
        role=TRANSITION_ROLE_PERSISTENT,
        source_is_new=False,
    )
    append_part(
        current_update,
        transition.old_only_indices,
        weight=old_weight,
        role=TRANSITION_ROLE_DEATH,
        source_is_new=False,
    )
    append_part(
        next_update,
        transition.new_only_indices,
        weight=new_weight,
        role=TRANSITION_ROLE_BIRTH,
        source_is_new=True,
    )

    if not coeff_parts:
        num_vc = current_update.coefficients.shape[0]
        return CoherentRTTransitionSynthesis(
            coefficients=np.zeros((num_vc, 0), dtype=current_update.coefficients.dtype),
            delays_s=np.zeros((0,), dtype=current_update.delays_s.dtype),
            valid=np.zeros((0,), dtype=bool),
            roles=np.zeros((0,), dtype=np.int8),
            source_is_new=np.zeros((0,), dtype=bool),
            source_indices=np.zeros((0,), dtype=np.int64),
            alpha=alpha,
            old_weight=old_weight,
            new_weight=new_weight,
        )

    source_indices = np.concatenate(source_index_parts).astype(np.int64, copy=False)
    return CoherentRTTransitionSynthesis(
        coefficients=np.concatenate(coeff_parts, axis=1),
        delays_s=np.concatenate(delay_parts).astype(current_update.delays_s.dtype,
                                                    copy=False),
        valid=np.ones(source_indices.shape, dtype=bool),
        roles=np.concatenate(role_parts),
        source_is_new=np.concatenate(source_is_new_parts),
        source_indices=source_indices,
        alpha=alpha,
        old_weight=old_weight,
        new_weight=new_weight,
    )


def _signature_index_map(
    signatures: tuple[tuple[tuple[int, int, int], ...], ...],
    valid: np.ndarray,
) -> dict[tuple[tuple[int, int, int], ...], list[int]]:
    """Maps each valid path signature to the path indices carrying it."""

    out: dict[tuple[tuple[int, int, int], ...], list[int]] = defaultdict(list)
    for index, signature in enumerate(signatures):
        if bool(valid[index]):
            out[signature].append(index)
    return out


def _unambiguous_delay_matches(
    old_indices: list[int],
    new_indices: list[int],
    old_lengths_m: np.ndarray,
    new_lengths_m: np.ndarray,
    *,
    tolerance_m: float,
    separation_m: float,
) -> list[tuple[int, int]]:
    """Matches old/new paths whose delay distance is isolated and below tolerance."""

    if not old_indices or not new_indices:
        return []
    old_arr = np.asarray(old_indices, dtype=np.int64)
    new_arr = np.asarray(new_indices, dtype=np.int64)
    costs = np.abs(old_lengths_m[old_arr, None] - new_lengths_m[new_arr][None, :])
    candidates: list[tuple[float, int, int]] = []
    for old_pos, old_index in enumerate(old_arr):
        order = np.argsort(costs[old_pos])
        best_new_pos = int(order[0])
        best_cost = float(costs[old_pos, best_new_pos])
        if best_cost > tolerance_m:
            continue
        row_alt = float(costs[old_pos, int(order[1])]) if order.size > 1 else np.inf
        column = np.sort(costs[:, best_new_pos])
        column_alt = float(column[1]) if column.size > 1 else np.inf
        if min(row_alt, column_alt) - best_cost < separation_m:
            continue
        candidates.append((best_cost, int(old_index), int(new_arr[best_new_pos])))

    pairs = []
    used_old: set[int] = set()
    used_new: set[int] = set()
    for _, old_index, new_index in sorted(candidates):
        if old_index in used_old or new_index in used_new:
            continue
        used_old.add(old_index)
        used_new.add(new_index)
        pairs.append((old_index, new_index))
    return pairs


def barycentric_coordinates(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """Returns barycentric coordinates of ``point`` on ``triangle``."""

    p = np.asarray(point, dtype=np.float64).reshape(3)
    tri = np.asarray(triangle, dtype=np.float64).reshape(3, 3)
    v0 = tri[1] - tri[0]
    v1 = tri[2] - tri[0]
    v2 = p - tri[0]
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = float(np.dot(v2, v0))
    d21 = float(np.dot(v2, v1))
    denom = d00 * d11 - d01 * d01
    if abs(denom) <= 1e-18:
        raise ValueError("cannot compute barycentric coordinates for degenerate triangle")
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w], dtype=np.float64)


def _target_object_ids(targets: Sequence[MeshTarget]) -> dict[int, int]:
    """Maps target scene object IDs to target-list indices."""

    target_ids = {}
    for target_index, target in enumerate(targets):
        object_id = int(
            np.asarray(target.ensure_scene_object().object_id).reshape(-1)[0]
        )
        target_ids[object_id] = target_index
    return target_ids


def _dynamic_hit_coordinates(
    *,
    path_vertices: np.ndarray,
    objects: np.ndarray,
    primitives: np.ndarray,
    target_ids: Mapping[int, int],
    target_faces: Sequence[np.ndarray],
    target_anchor_vertices: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Stores dynamic-target hit identity and barycentrics at anchor geometry."""

    dynamic_target_indices = np.full(objects.shape, -1, dtype=np.int64)
    dynamic_barycentrics = np.full(objects.shape + (3,), np.nan, dtype=np.float64)
    for depth_index in range(objects.shape[0]):
        for path_index in range(objects.shape[1]):
            object_id = int(objects[depth_index, path_index])
            target_index = target_ids.get(object_id)
            if target_index is None:
                continue
            primitive = int(primitives[depth_index, path_index])
            if primitive < 0 or primitive == int(INVALID_PRIMITIVE):
                continue
            faces = target_faces[target_index]
            if primitive >= faces.shape[0]:
                continue
            point = path_vertices[depth_index, path_index]
            if not np.all(np.isfinite(point)):
                continue
            triangle = target_anchor_vertices[target_index][faces[primitive]]
            dynamic_target_indices[depth_index, path_index] = target_index
            dynamic_barycentrics[depth_index, path_index] = barycentric_coordinates(
                point,
                triangle,
            )
    return dynamic_target_indices, dynamic_barycentrics


def _coerce_target_vertices(
    target_vertices: Sequence[np.ndarray] | Mapping[int, np.ndarray],
) -> tuple[np.ndarray, ...]:
    """Normalizes target vertices to a tuple indexed by target id."""

    if isinstance(target_vertices, Mapping):
        if not target_vertices:
            return ()
        max_index = max(int(k) for k in target_vertices)
        return tuple(
            np.asarray(target_vertices[index], dtype=np.float64)
            for index in range(max_index + 1)
        )
    return tuple(np.asarray(vertices, dtype=np.float64) for vertices in target_vertices)


def _use_cuda_coherent_update(backend: str) -> bool:
    """Returns whether coherent-bank geometry should use Torch CUDA."""

    if backend == "numpy":
        return False
    if backend not in ("auto", "torch"):
        raise ValueError("backend must be 'auto', 'numpy', or 'torch'")
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError:
        if backend == "torch":
            raise
        return False
    return _torch_cuda_is_available(torch)


def _update_coherent_rt_path_bank_torch(
    bank: CoherentRTPathBank,
    target_vertices: tuple[np.ndarray, ...],
    *,
    frequency_hz: float | None,
    device,
) -> CoherentRTPathUpdate:
    """Torch implementation of the vectorized coherent-bank update."""

    import torch  # pylint: disable=import-outside-toplevel

    dtype = torch.float64
    bank_data = _torch_coherent_bank_data(bank, device=device)
    path_lengths = bank_data["anchor_path_lengths"].clone()
    delay_lengths = bank_data["anchor_delay_lengths"].clone()
    spreads = bank_data["anchor_spreads"].clone()
    path_indices = bank_data["path_indices_numpy"]
    if path_indices.size:
        path_indices_t = bank_data["path_indices"]
        points = bank_data["path_vertices"].clone()
        active = bank_data["active"]
        barycentrics = bank_data["barycentrics"]

        for target_index, vertices in enumerate(target_vertices):
            if target_index >= len(bank_data["target_hits"]):
                continue
            (
                depth_indices,
                local_path_indices,
                triangle_vertex_indices,
            ) = bank_data["target_hits"][target_index]
            if depth_indices.numel() == 0:
                continue
            vertices_t = torch.as_tensor(
                np.asarray(vertices, dtype=np.float64),
                dtype=dtype,
                device=device,
            )
            triangles = vertices_t[triangle_vertex_indices]
            weights = barycentrics[depth_indices, local_path_indices]
            points[depth_indices, local_path_indices] = torch.einsum(
                "ij,ijk->ik",
                weights,
                triangles,
            )

        num_dynamic_paths = int(path_indices.size)
        seen = torch.zeros(num_dynamic_paths, dtype=torch.bool, device=device)
        first_points = torch.zeros(
            (num_dynamic_paths, 3),
            dtype=dtype,
            device=device,
        )
        previous_points = torch.zeros_like(first_points)
        internal_lengths = torch.zeros(
            num_dynamic_paths,
            dtype=dtype,
            device=device,
        )
        internal_spreads = torch.ones_like(internal_lengths)
        for depth_index in range(points.shape[0]):
            depth_active = active[depth_index]
            first = depth_active & ~seen
            continuing = depth_active & seen
            first_points[first] = points[depth_index, first]
            segment_lengths = torch.linalg.vector_norm(
                points[depth_index, continuing]
                - previous_points[continuing],
                dim=1,
            )
            internal_lengths[continuing] += segment_lengths
            internal_spreads[continuing] *= torch.clamp(
                segment_lengths,
                min=1e-18,
            )
            previous_points[depth_active] = points[depth_index, depth_active]
            seen |= depth_active

        updated_lengths, updated_spreads = _torch_dynamic_endpoint_geometry(
            first_points=first_points,
            last_points=previous_points,
            internal_lengths=internal_lengths,
            internal_spreads=internal_spreads,
            tx_positions=bank_data["tx_positions"],
            rx_positions=bank_data["rx_positions"],
        )
        path_lengths[:, path_indices_t] = updated_lengths
        spreads[:, path_indices_t] = updated_spreads
        center_lengths, _ = _torch_dynamic_endpoint_geometry(
            first_points=first_points,
            last_points=previous_points,
            internal_lengths=internal_lengths,
            internal_spreads=internal_spreads,
            tx_positions=bank_data["tx_center"],
            rx_positions=bank_data["rx_center"],
        )
        delay_lengths[path_indices_t] = center_lengths[0]

    frequency = bank.frequency_hz if frequency_hz is None else float(frequency_hz)
    wave_number = 2.0 * pi * frequency / c
    anchor_lengths = bank_data["anchor_path_lengths"]
    anchor_spreads = bank_data["anchor_spreads"]
    coefficients = bank_data["coefficients"]
    phase = torch.exp(1j * wave_number * (path_lengths - anchor_lengths))
    spread_ratio = anchor_spreads / torch.clamp(spreads, min=1e-18)
    coefficients = coefficients * phase * spread_ratio
    path_lengths_np = path_lengths.cpu().numpy()
    delay_lengths_np = delay_lengths.cpu().numpy()
    spreads_np = spreads.cpu().numpy()
    return CoherentRTPathUpdate(
        coefficients=coefficients.cpu().numpy().astype(
            bank.coefficients.dtype,
            copy=False,
        ),
        delays_s=(delay_lengths_np / c).astype(bank.delays_s.dtype, copy=False),
        valid=bank.valid.copy(),
        path_lengths_m=path_lengths_np,
        delay_lengths_m=delay_lengths_np,
        spread_products_m=spreads_np,
    )


def _torch_coherent_bank_data(bank: CoherentRTPathBank, *, device) -> dict:
    """Returns invariant coherent-bank tensors cached on one Torch device."""

    import torch  # pylint: disable=import-outside-toplevel

    cache_key = (id(bank), str(device))
    cached = _TORCH_COHERENT_BANK_CACHE.get(cache_key)
    if cached is not None and cached[0]() is bank:
        return cached[1]

    dtype = torch.float64
    dynamic_paths = np.any(
        np.asarray(bank.dynamic_target_indices, dtype=np.int64) >= 0,
        axis=0,
    )
    path_indices = np.flatnonzero(dynamic_paths)
    active = np.asarray(
        bank.path_interactions[:, path_indices] != InteractionType.NONE,
        dtype=bool,
    )
    if path_indices.size and not np.all(np.any(active, axis=0)):
        raise RuntimeError(
            "dynamic coherent paths must contain at least one interaction"
        )
    primitives = np.asarray(
        bank.path_primitives[:, path_indices],
        dtype=np.int64,
    )
    target_indices = np.asarray(
        bank.dynamic_target_indices[:, path_indices],
        dtype=np.int64,
    )
    barycentrics = np.asarray(
        bank.dynamic_barycentrics[:, path_indices, :],
        dtype=np.float64,
    )
    target_hits = []
    for target_index, target_faces in enumerate(bank.target_faces):
        faces = np.asarray(target_faces, dtype=np.int64)
        mask = (
            active
            & (target_indices == target_index)
            & np.all(np.isfinite(barycentrics), axis=-1)
            & (primitives >= 0)
            & (primitives < faces.shape[0])
        )
        depth_indices, local_path_indices = np.nonzero(mask)
        triangle_vertex_indices = faces[
            primitives[depth_indices, local_path_indices]
        ]
        target_hits.append((
            torch.as_tensor(
                depth_indices,
                dtype=torch.int64,
                device=device,
            ),
            torch.as_tensor(
                local_path_indices,
                dtype=torch.int64,
                device=device,
            ),
            torch.as_tensor(
                triangle_vertex_indices,
                dtype=torch.int64,
                device=device,
            ),
        ))
    tx_positions = np.asarray(bank.tx_positions, dtype=np.float64)
    rx_positions = np.asarray(bank.rx_positions, dtype=np.float64)
    vc_tx, vc_rx = resolve_virtual_channel_pairs(
        int(tx_positions.shape[0]),
        int(rx_positions.shape[0]),
        order=bank.virtual_channel_order,
        tx_indices=bank.virtual_channel_tx_indices,
        rx_indices=bank.virtual_channel_rx_indices,
    )
    data = {
        "anchor_path_lengths": torch.as_tensor(
            bank.anchor_path_lengths_m,
            dtype=dtype,
            device=device,
        ),
        "anchor_delay_lengths": torch.as_tensor(
            bank.anchor_delay_lengths_m,
            dtype=dtype,
            device=device,
        ),
        "anchor_spreads": torch.as_tensor(
            bank.anchor_spread_products_m,
            dtype=dtype,
            device=device,
        ),
        "coefficients": torch.as_tensor(
            bank.coefficients,
            dtype=torch.complex128,
            device=device,
        ),
        "path_indices_numpy": path_indices,
        "path_indices": torch.as_tensor(
            path_indices,
            dtype=torch.int64,
            device=device,
        ),
        "path_vertices": torch.as_tensor(
            np.asarray(bank.path_vertices[:, path_indices, :], dtype=np.float64),
            dtype=dtype,
            device=device,
        ),
        "active": torch.as_tensor(
            active,
            dtype=torch.bool,
            device=device,
        ),
        "barycentrics": torch.as_tensor(
            barycentrics,
            dtype=dtype,
            device=device,
        ),
        "target_hits": tuple(target_hits),
        "tx_positions": torch.as_tensor(
            tx_positions[vc_tx],
            dtype=dtype,
            device=device,
        ),
        "rx_positions": torch.as_tensor(
            rx_positions[vc_rx],
            dtype=dtype,
            device=device,
        ),
        "tx_center": torch.as_tensor(
            bank.tx_center,
            dtype=dtype,
            device=device,
        ).reshape(1, 3),
        "rx_center": torch.as_tensor(
            bank.rx_center,
            dtype=dtype,
            device=device,
        ).reshape(1, 3),
    }

    def remove_cache_entry(_reference, *, key=cache_key):
        _TORCH_COHERENT_BANK_CACHE.pop(key, None)

    _TORCH_COHERENT_BANK_CACHE[cache_key] = (
        weakref.ref(bank, remove_cache_entry),
        data,
    )
    return data


def _torch_dynamic_endpoint_geometry(
    *,
    first_points,
    last_points,
    internal_lengths,
    internal_spreads,
    tx_positions,
    rx_positions,
):
    """Torch counterpart of :func:`_dynamic_endpoint_geometry`."""

    import torch  # pylint: disable=import-outside-toplevel

    tx_lengths = torch.linalg.vector_norm(
        first_points[None, :, :] - tx_positions[:, None, :],
        dim=-1,
    )
    rx_lengths = torch.linalg.vector_norm(
        rx_positions[:, None, :] - last_points[None, :, :],
        dim=-1,
    )
    lengths = tx_lengths + internal_lengths[None, :] + rx_lengths
    spreads = (
        torch.clamp(tx_lengths, min=1e-18)
        * internal_spreads[None, :]
        * torch.clamp(rx_lengths, min=1e-18)
    )
    return lengths, spreads


def _current_dynamic_path_points(
    bank: CoherentRTPathBank,
    target_vertices: tuple[np.ndarray, ...],
    path_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns current interaction points for paths touching dynamic targets."""

    points = np.asarray(
        bank.path_vertices[:, path_indices, :],
        dtype=np.float64,
    ).copy()
    interactions = np.asarray(bank.path_interactions[:, path_indices])
    active = interactions != InteractionType.NONE
    primitives = np.asarray(
        bank.path_primitives[:, path_indices],
        dtype=np.int64,
    )
    target_indices = np.asarray(
        bank.dynamic_target_indices[:, path_indices],
        dtype=np.int64,
    )
    barycentrics = np.asarray(
        bank.dynamic_barycentrics[:, path_indices, :],
        dtype=np.float64,
    )

    for target_index, vertices in enumerate(target_vertices):
        if target_index >= len(bank.target_faces):
            continue
        faces = np.asarray(bank.target_faces[target_index], dtype=np.int64)
        mask = (
            active
            & (target_indices == target_index)
            & np.all(np.isfinite(barycentrics), axis=-1)
            & (primitives >= 0)
            & (primitives < faces.shape[0])
        )
        if not np.any(mask):
            continue
        depth_indices, local_path_indices = np.nonzero(mask)
        face_indices = primitives[depth_indices, local_path_indices]
        triangles = np.asarray(vertices, dtype=np.float64)[faces[face_indices]]
        weights = barycentrics[depth_indices, local_path_indices]
        points[depth_indices, local_path_indices] = np.einsum(
            "ij,ijk->ik",
            weights,
            triangles,
            optimize=True,
        )
    return points, active


def _shared_dynamic_path_geometry(
    points: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes channel-invariant geometry once for each dynamic path."""

    num_paths = int(points.shape[1])
    seen = np.zeros(num_paths, dtype=bool)
    first_points = np.zeros((num_paths, 3), dtype=np.float64)
    previous_points = np.zeros_like(first_points)
    internal_lengths = np.zeros(num_paths, dtype=np.float64)
    internal_spreads = np.ones(num_paths, dtype=np.float64)

    for depth_index in range(points.shape[0]):
        depth_active = active[depth_index]
        if not np.any(depth_active):
            continue
        first = depth_active & ~seen
        continuing = depth_active & seen
        if np.any(first):
            first_points[first] = points[depth_index, first]
        if np.any(continuing):
            segment_lengths = np.linalg.norm(
                points[depth_index, continuing] - previous_points[continuing],
                axis=1,
            )
            internal_lengths[continuing] += segment_lengths
            internal_spreads[continuing] *= np.maximum(segment_lengths, 1e-18)
        previous_points[depth_active] = points[depth_index, depth_active]
        seen |= depth_active

    if not np.all(seen):
        raise RuntimeError(
            "dynamic coherent paths must contain at least one interaction"
        )
    return first_points, previous_points, internal_lengths, internal_spreads


def _dynamic_endpoint_geometry(
    *,
    first_points: np.ndarray,
    last_points: np.ndarray,
    internal_lengths: np.ndarray,
    internal_spreads: np.ndarray,
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Adds virtual-channel endpoint segments to shared dynamic geometry."""

    tx = np.asarray(tx_positions, dtype=np.float64)
    rx = np.asarray(rx_positions, dtype=np.float64)
    tx_lengths = np.linalg.norm(
        first_points[None, :, :] - tx[:, None, :],
        axis=-1,
    )
    rx_lengths = np.linalg.norm(
        rx[:, None, :] - last_points[None, :, :],
        axis=-1,
    )
    lengths = tx_lengths + internal_lengths[None, :] + rx_lengths
    spreads = (
        np.maximum(tx_lengths, 1e-18)
        * internal_spreads[None, :]
        * np.maximum(rx_lengths, 1e-18)
    )
    return lengths, spreads


def _path_lengths_and_spreads(
    *,
    path_vertices: np.ndarray,
    interactions: np.ndarray,
    primitives: np.ndarray,
    dynamic_target_indices: np.ndarray,
    dynamic_barycentrics: np.ndarray,
    target_faces: Sequence[np.ndarray],
    target_vertices: Sequence[np.ndarray] | None,
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    virtual_channel_order: str = "tx_major",
    virtual_channel_tx_indices: np.ndarray | None = None,
    virtual_channel_rx_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes per-virtual-channel path lengths and spreading products."""

    tx_positions = np.asarray(tx_positions, dtype=np.float64)
    rx_positions = np.asarray(rx_positions, dtype=np.float64)
    vc_tx, vc_rx = resolve_virtual_channel_pairs(
        int(tx_positions.shape[0]),
        int(rx_positions.shape[0]),
        order=virtual_channel_order,
        tx_indices=virtual_channel_tx_indices,
        rx_indices=virtual_channel_rx_indices,
    )
    vc_pairs = list(zip(vc_tx.tolist(), vc_rx.tolist()))
    if len(vc_pairs) == 1:
        tx_index, rx_index = vc_pairs[0]
        return _single_vc_path_lengths_and_spreads(
            path_vertices=path_vertices,
            interactions=interactions,
            primitives=primitives,
            dynamic_target_indices=dynamic_target_indices,
            dynamic_barycentrics=dynamic_barycentrics,
            target_faces=target_faces,
            target_vertices=target_vertices,
            tx_position=tx_positions[tx_index],
            rx_position=rx_positions[rx_index],
        )

    num_paths = path_vertices.shape[1]
    lengths = np.zeros((len(vc_pairs), num_paths), dtype=np.float64)
    spreads = np.ones_like(lengths)
    for path_index in range(num_paths):
        hit_depths = np.flatnonzero(
            interactions[:, path_index] != InteractionType.NONE
        )
        for vc_index, (tx_index, rx_index) in enumerate(vc_pairs):
            points = [tx_positions[tx_index]]
            for depth_index in hit_depths:
                points.append(_interaction_point(
                    depth_index=depth_index,
                    path_index=path_index,
                    path_vertices=path_vertices,
                    primitives=primitives,
                    dynamic_target_indices=dynamic_target_indices,
                    dynamic_barycentrics=dynamic_barycentrics,
                    target_faces=target_faces,
                    target_vertices=target_vertices,
                ))
            points.append(rx_positions[rx_index])
            segment_lengths = [
                float(np.linalg.norm(np.asarray(end) - np.asarray(start)))
                for start, end in zip(points[:-1], points[1:])
            ]
            lengths[vc_index, path_index] = float(np.sum(segment_lengths))
            spreads[vc_index, path_index] = float(
                np.prod(np.maximum(segment_lengths, 1e-18))
            )
    return lengths, spreads


def _single_vc_path_lengths_and_spreads(
    *,
    path_vertices: np.ndarray,
    interactions: np.ndarray,
    primitives: np.ndarray,
    dynamic_target_indices: np.ndarray,
    dynamic_barycentrics: np.ndarray,
    target_faces: Sequence[np.ndarray],
    target_vertices: Sequence[np.ndarray] | None,
    tx_position: np.ndarray,
    rx_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized length/spread computation for a single virtual channel."""

    num_paths = path_vertices.shape[1]
    lengths = np.zeros(num_paths, dtype=np.float64)
    spreads = np.ones(num_paths, dtype=np.float64)
    previous_points = np.broadcast_to(
        np.asarray(tx_position, dtype=np.float64).reshape(1, 3),
        (num_paths, 3),
    ).copy()

    for depth_index in range(interactions.shape[0]):
        active = interactions[depth_index] != InteractionType.NONE
        if not np.any(active):
            continue
        points = _interaction_points_for_depth(
            depth_index=depth_index,
            path_vertices=path_vertices,
            primitives=primitives,
            dynamic_target_indices=dynamic_target_indices,
            dynamic_barycentrics=dynamic_barycentrics,
            target_faces=target_faces,
            target_vertices=target_vertices,
        )
        segment_lengths = np.linalg.norm(points[active] - previous_points[active], axis=1)
        lengths[active] += segment_lengths
        spreads[active] *= np.maximum(segment_lengths, 1e-18)
        previous_points[active] = points[active]

    rx = np.asarray(rx_position, dtype=np.float64).reshape(1, 3)
    final_lengths = np.linalg.norm(rx - previous_points, axis=1)
    lengths += final_lengths
    spreads *= np.maximum(final_lengths, 1e-18)
    return lengths[None, :], spreads[None, :]


def _interaction_points_for_depth(
    *,
    depth_index: int,
    path_vertices: np.ndarray,
    primitives: np.ndarray,
    dynamic_target_indices: np.ndarray,
    dynamic_barycentrics: np.ndarray,
    target_faces: Sequence[np.ndarray],
    target_vertices: Sequence[np.ndarray] | None,
) -> np.ndarray:
    """Returns interaction points at one depth, updating dynamic target hits."""

    points = np.asarray(path_vertices[depth_index], dtype=np.float64).copy()
    if target_vertices is None:
        return points

    target_indices = dynamic_target_indices[depth_index]
    barycentrics = dynamic_barycentrics[depth_index]
    primitive_indices = np.asarray(primitives[depth_index], dtype=np.int64)
    finite_bary = np.all(np.isfinite(barycentrics), axis=1)

    for target_index, vertices in enumerate(target_vertices):
        if target_index >= len(target_faces):
            continue
        faces = target_faces[target_index]
        mask = (
            (target_indices == target_index)
            & finite_bary
            & (primitive_indices >= 0)
            & (primitive_indices < faces.shape[0])
        )
        if not np.any(mask):
            continue
        triangles = np.asarray(vertices, dtype=np.float64)[faces[primitive_indices[mask]]]
        points[mask] = np.einsum(
            "ij,ijk->ik",
            barycentrics[mask],
            triangles,
            optimize=True,
        )
    return points


def _interaction_point(
    *,
    depth_index: int,
    path_index: int,
    path_vertices: np.ndarray,
    primitives: np.ndarray,
    dynamic_target_indices: np.ndarray,
    dynamic_barycentrics: np.ndarray,
    target_faces: Sequence[np.ndarray],
    target_vertices: Sequence[np.ndarray] | None,
) -> np.ndarray:
    """Returns one interaction point, reprojected on current target vertices."""

    target_index = int(dynamic_target_indices[depth_index, path_index])
    if target_index < 0 or target_vertices is None:
        return np.asarray(path_vertices[depth_index, path_index], dtype=np.float64)
    bary = dynamic_barycentrics[depth_index, path_index]
    if not np.all(np.isfinite(bary)):
        return np.asarray(path_vertices[depth_index, path_index], dtype=np.float64)
    primitive = int(primitives[depth_index, path_index])
    faces = target_faces[target_index]
    if primitive < 0 or primitive >= faces.shape[0]:
        return np.asarray(path_vertices[depth_index, path_index], dtype=np.float64)
    triangle = np.asarray(target_vertices[target_index], dtype=np.float64)[faces[primitive]]
    return bary @ triangle


def _path_signatures(
    interactions: np.ndarray,
    objects: np.ndarray,
    primitives: np.ndarray,
    num_paths: int,
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    """Builds interaction/object/primitive signatures for path matching."""

    signatures = []
    for path_index in range(num_paths):
        parts = []
        for depth_index in range(interactions.shape[0]):
            interaction = int(interactions[depth_index, path_index])
            if interaction == int(InteractionType.NONE):
                continue
            object_id = int(objects[depth_index, path_index])
            primitive = int(primitives[depth_index, path_index])
            if object_id == int(INVALID_SHAPE):
                object_id = -1
            if primitive == int(INVALID_PRIMITIVE):
                primitive = -1
            parts.append((interaction, object_id, primitive))
        signatures.append(tuple(parts))
    return tuple(signatures)
