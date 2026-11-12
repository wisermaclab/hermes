# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Refinement diagnostics report actual blockers without retesting converged leaves."""

from hashlib import sha256

import numpy as np
import pytest

import mmWaveRadar.simulation.physical_optics as po

MODES = ["quadrature", "adaptive_analytic", "far_field"]


def _geometry(mode, vertices, *, depth=8, cap=0, events=None, **overrides):
    faces = np.arange(len(vertices), dtype=np.uint32).reshape(-1, 3)
    kwargs = dict(
        vertices=vertices,
        faces=faces,
        selected_parent_indices=np.arange(len(faces)),
        tx_positions=np.zeros((1, 3)),
        rx_positions=np.zeros((1, 3)),
        frequency_hz=60e9,
        max_refinement_depth=depth,
        max_subfaces_per_parent=cap,
        progress_callback=None if events is None else events.append,
    )
    if mode == "far_field":
        kwargs["far_field_max_edge_m"] = 1e-4
        function = po.parent_face_far_field_analytic_geometry
    else:
        kwargs["phase_span_scale_rad"] = 1e-5
        function = po.parent_face_quadrature_geometry
        if mode == "adaptive_analytic":
            kwargs["curvature_error_rad"] = 1e-6
            function = po.parent_face_adaptive_analytic_geometry
    kwargs.update(overrides)
    return function(**kwargs)


TRIANGLE = np.array([[2.0, -0.5, -0.5], [2.0, -0.5, 0.5], [2.0, 0.5, -0.5]])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "depth,cap,count,limit",
    [
        (0, 1, 1, "max_refinement_depth"),
        (1, 1, 1, "max_subfaces_per_parent"),
        (1, 32, 4, "max_refinement_depth"),
        (2, 32, 16, "max_refinement_depth"),
        (2, 4, 4, "max_subfaces_per_parent"),
    ],
)
def test_done_event_has_consistent_fields_and_names_only_the_limit_that_blocked(
    mode, depth, cap, count, limit
):
    events = []
    result = _geometry(mode, TRIANGLE, depth=depth, cap=cap, events=events)
    done = events[-1]
    assert done["stage"] == f"{mode}_geometry_done"
    assert len(result[0]) == count
    assert done["unresolved_subfaces"] == count
    assert done["unresolved_parent_faces"] == 1
    assert done["refinement_limits_reached"] == [limit]
    assert done["refinement_depth_reached"] == {1: 0, 4: 1, 16: 2}[count]
    assert len(bytes.fromhex(done["subdivision_sha256"])) == 32
    if mode != "quadrature":
        triangles = result[4]
        expected = sha256(np.ascontiguousarray(triangles, dtype="<f8"))
        expected.update(np.ascontiguousarray(result[3], dtype="<i8"))
        assert done["subdivision_sha256"] == expected.hexdigest()


def test_converged_parent_at_final_depth_does_not_ascribe_depth_limit_to_cap_blocked_parent():
    small_triangle = np.array(
        [
            [1.999235935, 0.111429690, 0.000496239],
            [1.997835297, 0.093904152, -0.018693283],
            [2.000719077, 0.080420856, 0.015828245],
        ]
    )
    events = []
    result = _geometry(
        "adaptive_analytic",
        np.vstack([TRIANGLE, small_triangle]),
        depth=2,
        cap=13,
        events=events,
        phase_span_scale_rad=3.0,
        curvature_error_rad=10.0,
    )
    done = events[-1]
    np.testing.assert_array_equal(np.bincount(result[3]), [4, 7])
    assert done["refinement_depth_reached"] == 2
    assert done["unresolved_subfaces"] == 4
    assert done["unresolved_parent_faces"] == 1
    assert done["refinement_limits_reached"] == ["max_subfaces_per_parent"]


@pytest.mark.parametrize("mode", MODES)
def test_converged_geometry_reports_no_limits_and_matches_callback_free_result(mode):
    events = []
    options = (
        {"far_field_max_edge_m": 0.4} if mode == "far_field" else {"phase_span_scale_rad": 1e4}
    )
    if mode == "adaptive_analytic":
        options["curvature_error_rad"] = 1e4
    reference = _geometry(mode, TRIANGLE, **options)
    result = _geometry(mode, TRIANGLE, events=events, **options)
    for actual, expected in zip(result, reference):
        np.testing.assert_array_equal(actual, expected)
    done = events[-1]
    assert done["unresolved_subfaces"] == 0
    assert done["unresolved_parent_faces"] == 0
    assert done["refinement_limits_reached"] == []


def test_quadrature_progress_does_not_add_a_split_test_on_convergence(monkeypatch):
    tested_sizes = []
    original = po._triangle_phase_span_rad

    def counted(triangles, *args):
        tested_sizes.append(len(triangles))
        return original(triangles, *args)

    monkeypatch.setattr(po, "_triangle_phase_span_rad", counted)
    result = _geometry("quadrature", TRIANGLE, phase_span_scale_rad=20)
    expected_sizes = tested_sizes.copy()
    tested_sizes.clear()
    events = []
    actual = _geometry("quadrature", TRIANGLE, phase_span_scale_rad=20, events=events)
    assert len(result[0]) > 1
    assert tested_sizes == expected_sizes
    for emitted, expected in zip(actual, result):
        np.testing.assert_array_equal(emitted, expected)
    assert events[-1]["unresolved_subfaces"] == 0
