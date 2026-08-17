# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Lightweight periodic RT retracing profiler."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Mapping, Sequence

import numpy as np

from .rt_coherence import (
    CoherentRTPathBank,
    match_coherent_rt_path_banks,
    update_coherent_rt_path_bank,
)


@dataclass(frozen=True)
class RTPilotSelection:
    """Sparse RT retraces and transition pairs selected under a budget."""

    indices: np.ndarray
    pairs: tuple[tuple[int, int, int], ...]
    budget_trace_count: int
    profile_trace_fraction: float


@dataclass(frozen=True)
class RTTransitionMetrics:
    """Transition quality measured between two profiled RT banks."""

    old_index: int
    new_index: int
    interval_chirps: int
    old_valid: int
    new_valid: int
    persistent: int
    old_only: int
    new_only: int
    persistent_fraction_old: float
    persistent_fraction_new: float
    path_death_fraction: float
    path_birth_fraction: float
    churn_fraction: float
    old_only_power_fraction: float
    new_only_power_fraction: float
    channel_error: float


@dataclass(frozen=True)
class RTScheduleCostPoint:
    """One profiled point on the periodic retracing cost-performance curve."""

    policy: str
    parameter_name: str
    parameter_value: float
    estimated_retrace_count: int
    estimated_retrace_fraction: float
    profile_sample_count: int
    mean_channel_error: float
    p95_channel_error: float
    max_channel_error: float
    mean_path_death_fraction: float
    mean_path_birth_fraction: float
    schedule_indices: np.ndarray


@dataclass(frozen=True)
class RTFixedScheduleCurveProfile:
    """Cost-performance curve for periodic retracing intervals."""

    periodic_curve: tuple[RTScheduleCostPoint, ...]


def rt_profile_trace_budget(
    total_chirps: int,
    *,
    max_fraction: float = 0.05,
) -> int:
    """Returns the maximum number of RT retraces for profiling."""

    total_chirps = int(total_chirps)
    if total_chirps < 0:
        raise ValueError("total_chirps must be non-negative")
    if not 0.0 <= max_fraction <= 1.0:
        raise ValueError("max_fraction must be in [0, 1]")
    return int(floor(float(total_chirps) * float(max_fraction)))


def select_budgeted_pilot_indices(
    vertices_by_chirp: Sequence[np.ndarray],
    *,
    candidate_periods_chirps: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
    num_chirps_per_frame: int | None = None,
    max_profile_fraction: float = 0.05,
    max_trace_count: int | None = None,
    pairs_per_interval: int = 2,
    vertex_stride: int = 8,
) -> RTPilotSelection:
    """Selects high-motion RT pilot pairs while staying under budget."""

    vertices = np.asarray(vertices_by_chirp)
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError("vertices_by_chirp must have shape [chirp, vertex, 3]")
    total_chirps = int(vertices.shape[0])
    budget = (
        rt_profile_trace_budget(total_chirps, max_fraction=max_profile_fraction)
        if max_trace_count is None else int(max_trace_count)
    )
    budget = max(min(budget, total_chirps), 0)
    if budget < 2 or total_chirps < 2:
        return RTPilotSelection(
            indices=np.empty(0, dtype=np.int64),
            pairs=(),
            budget_trace_count=budget,
            profile_trace_fraction=0.0,
        )

    sampled_vertices = vertices[:, ::max(int(vertex_stride), 1), :]
    periods = _sanitize_periods(candidate_periods_chirps, total_chirps, num_chirps_per_frame)
    pair_candidates_by_interval: dict[int, list[tuple[float, int, int]]] = {}
    for interval in periods:
        displacement = np.linalg.norm(
            sampled_vertices[interval:] - sampled_vertices[:-interval],
            axis=2,
        )
        score = np.max(displacement, axis=1)
        order = np.argsort(score)[::-1]
        keep = max(int(pairs_per_interval), 1)
        anchors = [int(index) for index in order[:keep]]
        if score.size > keep:
            anchors.extend(_even_anchor_indices(score.size, keep))
        seen = set()
        candidates = []
        for anchor in anchors:
            if anchor in seen:
                continue
            seen.add(anchor)
            candidates.append((float(score[anchor]), anchor, anchor + interval))
        pair_candidates_by_interval[interval] = candidates

    selected: set[int] = set()
    selected_pairs: list[tuple[int, int, int]] = []
    selected_pair_keys: set[tuple[int, int]] = set()

    def try_add_pair(old_index: int, new_index: int, interval: int) -> None:
        """Adds a pilot pair if it fits the remaining unique-index budget."""

        key = (int(old_index), int(new_index))
        if key in selected_pair_keys:
            return
        required = {int(old_index), int(new_index)}
        if len(selected | required) > budget:
            return
        selected.update(required)
        selected_pair_keys.add(key)
        selected_pairs.append((int(old_index), int(new_index), int(interval)))

    star_intervals = _interval_probe_order(periods, max(budget - 1, 0))
    if star_intervals:
        star_anchor = _best_common_anchor(sampled_vertices, star_intervals)
        for interval in star_intervals:
            if star_anchor + interval < total_chirps:
                try_add_pair(star_anchor, star_anchor + interval, interval)

    for rank in range(max(len(v) for v in pair_candidates_by_interval.values())):
        for interval in periods:
            candidates = pair_candidates_by_interval.get(interval, [])
            if rank >= len(candidates):
                continue
            _, old_index, new_index = candidates[rank]
            try_add_pair(old_index, new_index, interval)

    selected_indices = np.asarray(sorted(selected), dtype=np.int64)
    return RTPilotSelection(
        indices=selected_indices,
        pairs=tuple(selected_pairs),
        budget_trace_count=budget,
        profile_trace_fraction=(
            float(selected_indices.size) / float(total_chirps)
            if total_chirps else 0.0
        ),
    )


def transition_metrics_for_pair(
    *,
    old_index: int,
    new_index: int,
    old_bank: CoherentRTPathBank,
    new_bank: CoherentRTPathBank,
    old_vertices: np.ndarray,
    new_vertices: np.ndarray,
    wavelength_m: float,
    match_delay_tolerance_fraction: float = 0.5,
    match_separation_fraction: float = 0.25,
) -> RTTransitionMetrics:
    """Computes path churn and channel error for one profiled transition."""

    if wavelength_m <= 0.0:
        raise ValueError("wavelength_m must be positive")
    transition = match_coherent_rt_path_banks(
        old_bank,
        new_bank,
        [new_vertices],
        wavelength_m=wavelength_m,
        match_delay_tolerance_fraction=match_delay_tolerance_fraction,
        match_separation_fraction=match_separation_fraction,
    )
    old_update = update_coherent_rt_path_bank(old_bank, [new_vertices])
    new_update = update_coherent_rt_path_bank(new_bank, [new_vertices])

    old_valid = int(np.count_nonzero(old_update.valid))
    new_valid = int(np.count_nonzero(new_update.valid))
    persistent = int(transition.matched_old_indices.size)
    old_only = int(transition.old_only_indices.size)
    new_only = int(transition.new_only_indices.size)
    all_paths = max(persistent + old_only + new_only, 1)

    old_channel = _coherent_sum(old_update.coefficients, old_update.valid)
    new_channel = _coherent_sum(new_update.coefficients, new_update.valid)
    channel_error = _relative_norm(old_channel - new_channel, new_channel)

    old_valid_indices = np.flatnonzero(np.asarray(old_update.valid, dtype=bool))
    new_valid_indices = np.flatnonzero(np.asarray(new_update.valid, dtype=bool))
    old_total_power = _coefficient_power(old_update.coefficients, old_valid_indices)
    new_total_power = _coefficient_power(new_update.coefficients, new_valid_indices)
    old_only_power = _coefficient_power(old_update.coefficients, transition.old_only_indices)
    new_only_power = _coefficient_power(new_update.coefficients, transition.new_only_indices)

    return RTTransitionMetrics(
        old_index=int(old_index),
        new_index=int(new_index),
        interval_chirps=int(new_index) - int(old_index),
        old_valid=old_valid,
        new_valid=new_valid,
        persistent=persistent,
        old_only=old_only,
        new_only=new_only,
        persistent_fraction_old=persistent / max(old_valid, 1),
        persistent_fraction_new=persistent / max(new_valid, 1),
        path_death_fraction=old_only / max(old_valid, 1),
        path_birth_fraction=new_only / max(new_valid, 1),
        churn_fraction=(old_only + new_only) / all_paths,
        old_only_power_fraction=old_only_power / max(old_total_power, 1e-30),
        new_only_power_fraction=new_only_power / max(new_total_power, 1e-30),
        channel_error=channel_error,
    )


def profile_transition_metrics(
    *,
    banks_by_index: Mapping[int, CoherentRTPathBank],
    vertices_by_chirp: Sequence[np.ndarray],
    wavelength_m: float,
    pairs: Sequence[tuple[int, int, int]],
    match_delay_tolerance_fraction: float = 0.5,
    match_separation_fraction: float = 0.25,
) -> tuple[RTTransitionMetrics, ...]:
    """Computes metrics for all selected pilot pairs."""

    metrics = []
    for old_index, new_index, _ in pairs:
        metrics.append(
            transition_metrics_for_pair(
                old_index=old_index,
                new_index=new_index,
                old_bank=banks_by_index[int(old_index)],
                new_bank=banks_by_index[int(new_index)],
                old_vertices=vertices_by_chirp[int(old_index)],
                new_vertices=vertices_by_chirp[int(new_index)],
                wavelength_m=wavelength_m,
                match_delay_tolerance_fraction=match_delay_tolerance_fraction,
                match_separation_fraction=match_separation_fraction,
            )
        )
    return tuple(metrics)


def periodic_retrace_schedule_indices(
    total_chirps: int,
    *,
    period_chirps: int,
) -> np.ndarray:
    """Returns causal periodic retrace indices starting at chirp 0."""

    total_chirps = int(total_chirps)
    if total_chirps <= 0:
        return np.empty(0, dtype=np.int64)
    interval = max(int(period_chirps), 1)
    return np.arange(0, total_chirps, interval, dtype=np.int64)


def normalize_retrace_schedule_indices(
    schedule_indices: Sequence[int],
    *,
    total_chirps: int,
) -> np.ndarray:
    """Sorts, clips, uniquifies, and makes a schedule start at chirp 0."""

    total_chirps = int(total_chirps)
    if total_chirps <= 0:
        return np.empty(0, dtype=np.int64)
    indices = np.asarray(schedule_indices, dtype=np.int64).reshape(-1)
    indices = indices[(indices >= 0) & (indices < total_chirps)]
    if indices.size == 0 or int(np.min(indices)) > 0:
        indices = np.concatenate([np.array([0], dtype=np.int64), indices])
    return np.unique(indices).astype(np.int64, copy=False)


def profile_fixed_retrace_scheme_curves(
    metrics: Sequence[RTTransitionMetrics],
    *,
    vertices_by_chirp: Sequence[np.ndarray],
    periodic_periods_chirps: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
    num_chirps_per_frame: int | None = None,
) -> RTFixedScheduleCurveProfile:
    """Profiles periodic retracing periods K as a cost-performance curve."""

    vertices = np.asarray(vertices_by_chirp)
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError("vertices_by_chirp must have shape [chirp, vertex, 3]")
    total_chirps = int(vertices.shape[0])
    if total_chirps <= 0:
        raise ValueError("vertices_by_chirp cannot be empty")
    metrics = tuple(metrics)
    if not metrics:
        raise ValueError("metrics cannot be empty")

    periodic_curve = tuple(
        _periodic_cost_point(
            metrics,
            total_chirps=total_chirps,
            period_chirps=period,
        )
        for period in _sanitize_periods(
            periodic_periods_chirps,
            total_chirps,
            num_chirps_per_frame,
        )
    )
    return RTFixedScheduleCurveProfile(periodic_curve=periodic_curve)


def recommend_periodic_retrace_period(
    profile: RTFixedScheduleCurveProfile,
    *,
    max_mean_channel_error: float | None = None,
    max_retrace_fraction: float | None = None,
) -> RTScheduleCostPoint | None:
    """Chooses the periodic K to pass as ``periodic_retrace_period_chirps``.

    With an error target, this returns the cheapest profiled K satisfying
    the target. With only a cost target, it returns the lowest-error period
    within the cost budget. If both targets are provided, both must be met.
    """

    candidates = list(profile.periodic_curve)
    if max_mean_channel_error is not None:
        candidates = [
            point for point in candidates
            if point.mean_channel_error <= float(max_mean_channel_error)
        ]
    if max_retrace_fraction is not None:
        candidates = [
            point for point in candidates
            if point.estimated_retrace_fraction <= float(max_retrace_fraction)
        ]
    if not candidates:
        return None
    if max_mean_channel_error is not None:
        return min(
            candidates,
            key=lambda point: (
                point.estimated_retrace_fraction,
                point.mean_channel_error,
            ),
        )
    return min(
        candidates,
        key=lambda point: (
            point.mean_channel_error,
            point.estimated_retrace_fraction,
        ),
    )


def _periodic_cost_point(
    metrics: Sequence[RTTransitionMetrics],
    *,
    total_chirps: int,
    period_chirps: int,
) -> RTScheduleCostPoint:
    """Builds a cost point for a fixed periodic retrace period."""

    period = max(int(period_chirps), 1)
    schedule = periodic_retrace_schedule_indices(total_chirps, period_chirps=period)
    profiled = () if period == 1 else _metrics_for_periodic_period(metrics, period)
    return _cost_point_from_metrics(
        policy="periodic",
        parameter_name="period_chirps",
        parameter_value=float(period),
        schedule_indices=schedule,
        total_chirps=total_chirps,
        metrics=profiled,
    )


def _cost_point_from_metrics(
    *,
    policy: str,
    parameter_name: str,
    parameter_value: float,
    schedule_indices: np.ndarray,
    total_chirps: int,
    metrics: Sequence[RTTransitionMetrics],
) -> RTScheduleCostPoint:
    """Aggregates profiled transition metrics into one schedule cost point."""

    if not metrics:
        errors = deaths = births = np.asarray([0.0], dtype=np.float64)
    else:
        errors = np.asarray([metric.channel_error for metric in metrics], dtype=np.float64)
        deaths = np.asarray([metric.path_death_fraction for metric in metrics], dtype=np.float64)
        births = np.asarray([metric.path_birth_fraction for metric in metrics], dtype=np.float64)
    return RTScheduleCostPoint(
        policy=policy,
        parameter_name=parameter_name,
        parameter_value=float(parameter_value),
        estimated_retrace_count=int(schedule_indices.size),
        estimated_retrace_fraction=float(schedule_indices.size) / max(int(total_chirps), 1),
        profile_sample_count=len(metrics),
        mean_channel_error=float(np.mean(errors)),
        p95_channel_error=float(np.percentile(errors, 95)),
        max_channel_error=float(np.max(errors)),
        mean_path_death_fraction=float(np.mean(deaths)),
        mean_path_birth_fraction=float(np.mean(births)),
        schedule_indices=np.asarray(schedule_indices, dtype=np.int64),
    )


def _metrics_for_periodic_period(
    metrics: Sequence[RTTransitionMetrics],
    period_chirps: int,
) -> tuple[RTTransitionMetrics, ...]:
    """Selects profiled intervals representative of a periodic retrace period."""

    period = max(int(period_chirps), 1)
    within = tuple(metric for metric in metrics if 0 < int(metric.interval_chirps) < period)
    if within:
        return within
    closest = min(metrics, key=lambda metric: abs(int(metric.interval_chirps) - max(period - 1, 1)))
    return (closest,)


def _interval_probe_order(intervals: Sequence[int], count: int) -> tuple[int, ...]:
    """Chooses interval probes spread toward longer retrace gaps."""

    intervals = tuple(int(interval) for interval in intervals)
    count = min(max(int(count), 0), len(intervals))
    if count == 0:
        return ()
    if count == len(intervals):
        return intervals
    indices = {len(intervals) - 1}
    positions = np.linspace(0, len(intervals) - 1, count + 1)[1:]
    indices.update(int(round(position)) for position in positions)
    while len(indices) > count:
        indices.remove(min(indices))
    candidate = len(intervals) - 1
    while len(indices) < count and candidate >= 0:
        indices.add(candidate)
        candidate -= 1
    return tuple(intervals[index] for index in sorted(indices))


def _best_common_anchor(sampled_vertices: np.ndarray, intervals: Sequence[int]) -> int:
    """Selects the anchor chirp with largest motion over requested intervals."""

    if not intervals:
        return 0
    max_interval = max(int(interval) for interval in intervals)
    limit = sampled_vertices.shape[0] - max_interval
    if limit <= 1:
        return 0
    scores = np.zeros(limit, dtype=np.float64)
    for interval in intervals:
        displacement = np.linalg.norm(
            sampled_vertices[:limit] - sampled_vertices[interval:interval + limit],
            axis=2,
        )
        scores = np.maximum(scores, np.max(displacement, axis=1))
    return int(np.argmax(scores))


def _sanitize_periods(
    candidate_periods_chirps: Sequence[int],
    total_chirps: int,
    num_chirps_per_frame: int | None = None,
) -> tuple[int, ...]:
    """Filters candidate periods to valid positive chirp intervals."""

    total_chirps = int(total_chirps)
    frame_chirps = None
    if num_chirps_per_frame is not None:
        frame_chirps = int(num_chirps_per_frame)
        if frame_chirps <= 0:
            raise ValueError("num_chirps_per_frame must be positive")
    periods = sorted({
        int(period)
        for period in candidate_periods_chirps
        if (
            0 < int(period) <= total_chirps
            and (frame_chirps is None or frame_chirps % int(period) == 0)
        )
    })
    if not periods and total_chirps > 0:
        periods = [1]
    return tuple(periods)


def _even_anchor_indices(count: int, keep: int) -> list[int]:
    """Returns approximately evenly spaced anchor indices."""

    if count <= 0 or keep <= 0:
        return []
    return [int(round(v)) for v in np.linspace(0, count - 1, keep)]


def _coherent_sum(coefficients: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Sums valid path coefficients coherently per virtual channel."""

    coefficients = np.asarray(coefficients, dtype=np.complex128)
    valid = np.asarray(valid, dtype=bool)
    if coefficients.ndim != 2:
        raise ValueError("coefficients must have shape [channel, path]")
    if valid.shape != (coefficients.shape[1],):
        raise ValueError("valid must have shape [path]")
    if not np.any(valid):
        return np.zeros(coefficients.shape[0], dtype=np.complex128)
    return np.sum(coefficients[:, valid], axis=1)


def _coefficient_power(coefficients: np.ndarray, indices: np.ndarray) -> float:
    """Returns total power for selected coefficient columns."""

    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return 0.0
    return float(np.sum(np.abs(np.asarray(coefficients)[:, indices]) ** 2))


def _relative_norm(error: np.ndarray, reference: np.ndarray) -> float:
    """Returns error norm relative to a reference norm with zero-reference fallback."""

    reference_norm = float(np.linalg.norm(reference))
    if reference_norm <= 1e-30:
        return float(np.linalg.norm(error))
    return float(np.linalg.norm(error) / reference_norm)
