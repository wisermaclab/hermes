# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Unit tests for the lightweight periodic RT retracing profiler."""

from __future__ import annotations

import numpy as np

from mmWaveRadar.simulation import (
    RTTransitionMetrics,
    periodic_retrace_schedule_indices,
    profile_fixed_retrace_scheme_curves,
    recommend_periodic_retrace_period,
    rt_profile_trace_budget,
    select_budgeted_pilot_indices,
)


def _moving_vertices(num_chirps=16, step_m=0.1):
    base = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return np.asarray([
        base + np.array([index * step_m, 0.0, 0.0])
        for index in range(num_chirps)
    ])


def _metric(interval, *, persistent=95, old_only=2, new_only=3, error=0.05):
    return RTTransitionMetrics(
        old_index=0,
        new_index=interval,
        interval_chirps=interval,
        old_valid=100,
        new_valid=100,
        persistent=persistent,
        old_only=old_only,
        new_only=new_only,
        persistent_fraction_old=persistent / 100.0,
        persistent_fraction_new=persistent / 100.0,
        path_death_fraction=old_only / 100.0,
        path_birth_fraction=new_only / 100.0,
        churn_fraction=(old_only + new_only) / max(persistent + old_only + new_only, 1),
        old_only_power_fraction=0.01 * old_only,
        new_only_power_fraction=0.01 * new_only,
        channel_error=error,
    )


def test_profile_trace_budget_is_fractional_floor():
    assert rt_profile_trace_budget(64, max_fraction=0.05) == 3
    assert rt_profile_trace_budget(100, max_fraction=0.05) == 5


def test_pilot_selection_stays_under_budget_and_uses_pair_endpoints():
    vertices = _moving_vertices(num_chirps=32, step_m=0.01)

    selection = select_budgeted_pilot_indices(
        vertices,
        candidate_periods_chirps=(1, 4, 8),
        max_trace_count=6,
        pairs_per_interval=1,
    )

    assert selection.indices.size <= 6
    assert selection.profile_trace_fraction <= 6 / 32
    assert selection.pairs
    selected = set(int(index) for index in selection.indices)
    for old_index, new_index, interval in selection.pairs:
        assert old_index in selected
        assert new_index in selected
        assert new_index - old_index == interval


def test_periodic_schedule_respects_requested_interval():
    indices = periodic_retrace_schedule_indices(10, period_chirps=3)

    assert np.array_equal(indices, np.array([0, 3, 6, 9]))


def test_fixed_retrace_scheme_curves_profile_periodic_family():
    vertices = _moving_vertices(num_chirps=10, step_m=0.1)
    metrics = (
        _metric(1, error=0.02),
        _metric(2, error=0.05),
        _metric(4, error=0.20),
    )

    profile = profile_fixed_retrace_scheme_curves(
        metrics,
        vertices_by_chirp=vertices,
        periodic_periods_chirps=(1, 2, 3, 4),
        num_chirps_per_frame=4,
    )

    assert [point.parameter_value for point in profile.periodic_curve] == [1.0, 2.0, 4.0]
    assert profile.periodic_curve[0].mean_channel_error == 0.0
    assert profile.periodic_curve[0].estimated_retrace_fraction == 1.0
    assert profile.periodic_curve[1].policy == "periodic"
    assert profile.periodic_curve[1].profile_sample_count == 1
    assert profile.periodic_curve[2].estimated_retrace_count < vertices.shape[0]


def test_recommend_periodic_retrace_period_returns_simulator_chirp_period():
    vertices = _moving_vertices(num_chirps=10, step_m=0.1)
    metrics = (
        _metric(1, error=0.02),
        _metric(2, error=0.05),
        _metric(4, error=0.20),
    )
    profile = profile_fixed_retrace_scheme_curves(
        metrics,
        vertices_by_chirp=vertices,
        periodic_periods_chirps=(1, 2, 3, 4),
        num_chirps_per_frame=4,
    )

    point = recommend_periodic_retrace_period(
        profile,
        max_mean_channel_error=0.06,
    )

    assert point is not None
    assert point.parameter_name == "period_chirps"
    assert int(point.parameter_value) == 4
