# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Geometry and clock invariants shared by lazy mesh-sequence views."""

import numpy as np
import pytest

from mmWaveRadar.targets import (
    MeshSequence,
    MeshSequenceTimeWindow,
    PlacedMeshSequence,
    RigidTransformMeshSequence,
    face_winding_signed_volume,
)
from mmWaveRadar.experiments.human_room import _PlacedMotionSequence, _MotionTimeWindow
from mmWaveRadar.simulation.hybrid_pipeline import _FrameRangeMeshSequence
from validation.runner import _BundleFrameMeshSequence, _TimedMeshSequence


def sequence(epoch=0):
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.uint32)
    return MeshSequence(
        np.stack([vertices + [t, 0, 0] for t in (0, 1, 2)]),
        faces,
        np.asarray([epoch, epoch + 1, epoch + 2]),
    )


def test_shared_placement_distinguishes_source_and_gui_clocks():
    base = sequence(1000)
    native = PlacedMeshSequence(base, position=None)
    gui = _PlacedMotionSequence(base, position=None, yaw_deg=0)
    np.testing.assert_array_equal(native.times, [1000, 1001, 1002])
    np.testing.assert_array_equal(gui.times, [0, 1, 2])
    assert gui.time_offset_s == 1000
    for t in (0, 0.5, 1.5, 2):
        np.testing.assert_array_equal(native.vertices_at(1000 + t), gui.vertices_at(t))
        np.testing.assert_array_equal(gui.vertices_at(t), base.vertices_at(1000 + t))


def test_placement_yaw_and_winding_are_shared():
    base = sequence()
    inverted = MeshSequence(base.vertices, base.faces[:, [0, 2, 1]], base.times)
    placed = PlacedMeshSequence(inverted, position=(3, 2, 1), yaw_deg=90)
    initial = placed.vertices_at(0)
    np.testing.assert_allclose(0.5 * (initial.min(0) + initial.max(0)), [3, 2, 1])
    assert face_winding_signed_volume(initial, placed.faces) > 0
    np.testing.assert_allclose(
        placed.vertices_at(1) - initial, np.tile([0, 1, 0], (4, 1)), atol=1e-6
    )


def test_time_windows_share_evaluation_but_preserve_distinct_sample_times():
    base = sequence()
    source_knots = MeshSequenceTimeWindow(base, 0.25, 1.5)
    frame_range = _FrameRangeMeshSequence(base, 0.25, 1.5)
    np.testing.assert_allclose(source_knots.times, [0, 0.75, 1.5])
    np.testing.assert_allclose(frame_range.times, [0, 1.5])
    for t in (-1, 0, 0.2, 1, 1.5, 9):
        np.testing.assert_array_equal(source_knots.vertices_at(t), frame_range.vertices_at(t))
    radar = _MotionTimeWindow(
        base, motion_start_s=0.25, motion_end_s=1.75, num_frames=3, frame_period_s=0.5
    )
    np.testing.assert_allclose(radar.times, [0, 0.5, 1])
    np.testing.assert_array_equal(radar.vertices_at(1.7), base.vertices_at(1.95))


def test_view_geometry_operations_preserve_physical_frame_and_slot_times():
    base = sequence(1000)
    views = [
        PlacedMeshSequence(base, position=None),
        RigidTransformMeshSequence(base, np.eye(3)),
        _TimedMeshSequence(base, np.asarray([1000, 1001.5])),
        _BundleFrameMeshSequence(base, frame_times_s=np.asarray([1000, 1001]), frame_period_s=0.1),
    ]
    for view in views:
        assert view.vertex_count == 4
        assert view.face_count == 4
        np.testing.assert_array_equal(
            view.vertices, np.stack([view.vertices_at(t) for t in view.times])
        )
        assert view.max_vertex_displacement(float(view.times[0]), view.vertices[0]) == 0
        with pytest.raises(ValueError, match="shape mismatch"):
            view.max_vertex_displacement(float(view.times[0]), np.zeros((3, 3)))
    # Irregular bundle frame starts retain within-frame physical slot offsets.
    np.testing.assert_array_equal(views[-1].vertices_at(0.125), base.vertices_at(1001.025))
