# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Geometry-only motion diagnostics for radar-facing mesh sequences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..radar.hardware import resolve_virtual_channel_pairs


@dataclass(frozen=True)
class RadialVelocitySamples:
    """Geometry-derived equivalent bistatic radial velocity samples."""

    times_s: np.ndarray
    radial_velocity_mps: np.ndarray
    weights: np.ndarray
    point_kind: str
    virtual_channel_order: str


def equivalent_bistatic_radial_velocity(
    points: np.ndarray,
    velocities: np.ndarray,
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    *,
    virtual_channel_order: str = "tx_major",
    virtual_channel_tx_indices: np.ndarray | None = None,
    virtual_channel_rx_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Returns equivalent monostatic radial velocity for Tx-point-Rx paths.

    For a moving point ``p(t)``, the direct bistatic path length is
    ``L = |p - tx| + |p - rx|``. The Doppler velocity convention used by the
    DSP helpers is the monostatic-equivalent range rate ``0.5 * dL/dt``.

    :param points: Point coordinates with shape ``[num_points, 3]``.
    :param velocities: Point velocities with shape ``[num_points, 3]``.
    :param tx_positions: Tx element positions with shape ``[num_tx, 3]``.
    :param rx_positions: Rx element positions with shape ``[num_rx, 3]``.
    :param virtual_channel_order: ``"tx_major"`` or ``"rx_major"`` ordering.
    :returns: Array with shape ``[num_points, num_tx * num_rx]``. Positive
        values mean increasing bistatic path length.
    """

    points = np.asarray(points, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    tx_positions = np.asarray(tx_positions, dtype=np.float64)
    rx_positions = np.asarray(rx_positions, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [num_points, 3]")
    if velocities.shape != points.shape:
        raise ValueError("velocities must match points shape")
    if tx_positions.ndim != 2 or tx_positions.shape[1] != 3:
        raise ValueError("tx_positions must have shape [num_tx, 3]")
    if rx_positions.ndim != 2 or rx_positions.shape[1] != 3:
        raise ValueError("rx_positions must have shape [num_rx, 3]")
    vc_tx, vc_rx = resolve_virtual_channel_pairs(
        int(tx_positions.shape[0]),
        int(rx_positions.shape[0]),
        order=virtual_channel_order,
        tx_indices=virtual_channel_tx_indices,
        rx_indices=virtual_channel_rx_indices,
    )

    tx_vectors = points[None, :, :] - tx_positions[:, None, :]
    rx_vectors = points[None, :, :] - rx_positions[:, None, :]
    tx_unit = tx_vectors / np.maximum(
        np.linalg.norm(tx_vectors, axis=-1, keepdims=True),
        1e-18,
    )
    rx_unit = rx_vectors / np.maximum(
        np.linalg.norm(rx_vectors, axis=-1, keepdims=True),
        1e-18,
    )
    tx_rate = np.einsum("npc,pc->np", tx_unit, velocities)
    rx_rate = np.einsum("mpc,pc->mp", rx_unit, velocities)
    rates = [
        0.5 * (tx_rate[int(tx_index)] + rx_rate[int(rx_index)])
        for tx_index, rx_index in zip(vc_tx, vc_rx)
    ]
    return np.stack(rates, axis=-1) if rates else np.zeros((points.shape[0], 0))


def mesh_sequence_radial_velocity_samples(
    mesh_sequence,
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    *,
    point_kind: str = "face_centroid",
    time_start_s: float | None = None,
    time_stop_s: float | None = None,
    virtual_channel_order: str = "tx_major",
    virtual_channel_tx_indices: np.ndarray | None = None,
    virtual_channel_rx_indices: np.ndarray | None = None,
) -> RadialVelocitySamples:
    """Profiles geometry-only radial velocities over native mesh intervals.

    Velocities are computed from consecutive native mesh samples, then projected
    onto the equivalent bistatic radar range direction. For ``point_kind``
    ``"face_centroid"``, vertex velocities are averaged to face centroids and
    face areas are returned as weights. For ``"vertex"``, every vertex receives
    unit weight.
    """

    vertices = np.asarray(mesh_sequence.vertices, dtype=np.float64)
    times = np.asarray(mesh_sequence.times, dtype=np.float64)
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError("mesh_sequence.vertices must have shape [T, V, 3]")
    if times.ndim != 1 or times.size != vertices.shape[0]:
        raise ValueError("mesh_sequence.times must have shape [T]")
    if times.size < 2:
        raise ValueError("At least two mesh samples are required")

    dt = np.diff(times)
    if np.any(dt <= 0.0):
        raise ValueError("mesh_sequence.times must be strictly increasing")
    interval_times = 0.5 * (times[:-1] + times[1:])
    mask = np.ones(interval_times.shape, dtype=bool)
    if time_start_s is not None:
        mask &= interval_times >= float(time_start_s)
    if time_stop_s is not None:
        mask &= interval_times <= float(time_stop_s)
    interval_indices = np.flatnonzero(mask)
    if interval_indices.size == 0:
        raise ValueError("Selected time window contains no mesh intervals")

    faces = np.asarray(mesh_sequence.faces, dtype=np.int64)
    radial_blocks = []
    weight_blocks = []
    for idx in interval_indices:
        v0 = vertices[idx]
        v1 = vertices[idx + 1]
        point_mid = 0.5 * (v0 + v1)
        velocity = (v1 - v0) / float(dt[idx])
        if point_kind == "face_centroid":
            if faces.ndim != 2 or faces.shape[1] != 3:
                raise ValueError("mesh_sequence.faces must have shape [F, 3]")
            tri_points = point_mid[faces]
            tri_velocities = velocity[faces]
            points = np.mean(tri_points, axis=1)
            point_velocities = np.mean(tri_velocities, axis=1)
            weights = _triangle_areas(tri_points)
        elif point_kind == "vertex":
            points = point_mid
            point_velocities = velocity
            weights = np.ones(points.shape[0], dtype=np.float64)
        else:
            raise ValueError("point_kind must be 'face_centroid' or 'vertex'")
        radial_blocks.append(
            equivalent_bistatic_radial_velocity(
                points,
                point_velocities,
                tx_positions,
                rx_positions,
                virtual_channel_order=virtual_channel_order,
                virtual_channel_tx_indices=virtual_channel_tx_indices,
                virtual_channel_rx_indices=virtual_channel_rx_indices,
            )
        )
        weight_blocks.append(weights)

    return RadialVelocitySamples(
        times_s=interval_times[interval_indices],
        radial_velocity_mps=np.stack(radial_blocks, axis=0),
        weights=np.stack(weight_blocks, axis=0),
        point_kind=point_kind,
        virtual_channel_order=virtual_channel_order,
    )


def radial_velocity_summary(
    samples: RadialVelocitySamples,
    *,
    percentiles: tuple[float, ...] = (50.0, 90.0, 95.0, 99.0, 99.9, 100.0),
) -> dict[str, object]:
    """Summarizes signed and absolute radial velocity distributions."""

    values = np.asarray(samples.radial_velocity_mps, dtype=np.float64).reshape(-1)
    channel_count = int(samples.radial_velocity_mps.shape[-1])
    weights = np.repeat(np.asarray(samples.weights, dtype=np.float64).reshape(-1), channel_count)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[finite]
    weights = weights[finite]
    if values.size == 0:
        raise ValueError("No finite radial velocity samples")
    abs_values = np.abs(values)
    return {
        "sample_count": int(values.size),
        "time_start_s": float(np.min(samples.times_s)),
        "time_stop_s": float(np.max(samples.times_s)),
        "point_kind": samples.point_kind,
        "virtual_channels": channel_count,
        "signed_min_mps": float(np.min(values)),
        "signed_mean_mps": float(np.average(values, weights=weights)),
        "signed_max_mps": float(np.max(values)),
        "abs_mean_mps": float(np.average(abs_values, weights=weights)),
        "abs_percentiles_mps": {
            float(p): float(_weighted_percentile(abs_values, weights, p))
            for p in percentiles
        },
        "signed_percentiles_mps": {
            float(p): float(_weighted_percentile(values, weights, p))
            for p in percentiles
        },
        "fraction_abs_ge_5_mps": float(np.average(abs_values >= 5.0, weights=weights)),
        "fraction_abs_ge_10_mps": float(np.average(abs_values >= 10.0, weights=weights)),
        "fraction_abs_ge_20_mps": float(np.average(abs_values >= 20.0, weights=weights)),
    }


def _triangle_areas(triangles: np.ndarray) -> np.ndarray:
    """Computes areas for triangles with shape ``[num_faces, 3, 3]``."""

    edge0 = triangles[:, 1] - triangles[:, 0]
    edge1 = triangles[:, 2] - triangles[:, 0]
    return 0.5 * np.linalg.norm(np.cross(edge0, edge1), axis=1)


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    """Computes a weighted percentile after dropping invalid samples."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.shape != weights.shape:
        raise ValueError("values and weights must have the same shape")
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[finite]
    weights = weights[finite]
    if values.size == 0:
        return float("nan")
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    cutoff = float(percentile) / 100.0 * cdf[-1]
    return float(values[min(int(np.searchsorted(cdf, cutoff, side="left")), values.size - 1)])
