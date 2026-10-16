# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Tests for compact RT-path and PO-facet diagnostic contracts."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest

from mmWaveRadar.experiments import (
    load_amass_motion,
    load_scene_xml,
    motion_source_to_radar_transform,
    prepare_human_room_preview,
    run_human_room_diagnostics,
    run_human_room_experiment,
    summarize_po_surface,
    summarize_rt_paths,
)
from mmWaveRadar.experiments.human_room import (
    _FrameWindowMotionSequence,
    _load_environment_scene,
    _resolve_motion_frame_window,
    prepared_room_boxes,
)
from mmWaveRadar.targets import MeshSequence


def test_motion_frame_window_resamples_selected_source_frames():
    sequence = MeshSequence(
        vertices=np.asarray(
            [
                [[0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0]],
                [[3.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        ),
        faces=np.empty((0, 3), dtype=np.uint32),
        times=np.asarray([0.0, 0.1, 0.2, 0.3]),
    )
    window = _FrameWindowMotionSequence(
        sequence,
        begin_frame_index=1,
        end_frame_index=3,
        frame_period_s=0.05,
    )

    assert window.times.tolist() == pytest.approx([0.0, 0.05, 0.1])
    assert window.source_frame_indices.tolist() == [1, 2, 3]
    assert window.vertices_at(0.0)[0, 0] == pytest.approx(1.0)
    assert window.vertices_at(0.025)[0, 0] == pytest.approx(1.5)
    assert window.vertices_at(0.1)[0, 0] == pytest.approx(3.0)
    single_frame = _FrameWindowMotionSequence(
        sequence,
        begin_frame_index=1,
        end_frame_index=1,
        frame_period_s=0.05,
    )
    assert single_frame.vertices_at(0.025)[0, 0] == pytest.approx(1.5)


def test_motion_frame_window_validation_and_legacy_count():
    assert _resolve_motion_frame_window(
        total_frames=5,
        begin_frame_index=1,
        end_frame_index=3,
    ) == (1, 3)
    assert _resolve_motion_frame_window(
        total_frames=5,
        begin_frame_index=2,
        num_frames=2,
    ) == (2, 3)
    with pytest.raises(ValueError, match="not both"):
        _resolve_motion_frame_window(
            total_frames=5,
            end_frame_index=2,
            num_frames=2,
        )
    with pytest.raises(ValueError, match="end_frame_index"):
        _resolve_motion_frame_window(
            total_frames=5,
            begin_frame_index=3,
            end_frame_index=2,
        )
    with pytest.raises(ValueError, match="simulation_mode must be one of"):
        run_human_room_experiment(
            human_motion=b"",
            simulation_mode="unsupported",
        )
    with pytest.raises(InterruptedError, match="cancelled by user"):
        run_human_room_experiment(
            human_motion=b"",
            cancel_check=lambda: True,
        )
    with pytest.raises(ValueError, match="progress_callback"):
        run_human_room_experiment(
            human_motion=b"",
            progress_callback="not callable",
        )


class _Tensor:
    def __init__(self, value):
        self._value = np.asarray(value)

    def numpy(self):
        return self._value


class _Paths:
    synthetic_array = True

    def __init__(self):
        from sionna.rt.constants import InteractionType

        invalid = np.uint32(0xFFFFFFFF)
        self.valid = _Tensor(np.asarray([[[True, True]]]))
        objects = np.full((2, 1, 1, 2), invalid, dtype=np.uint32)
        objects[0, 0, 0, 0] = 7
        self.objects = _Tensor(objects)
        primitives = np.full_like(objects, invalid)
        primitives[0, 0, 0, 0] = 4
        self.primitives = _Tensor(primitives)
        interactions = np.full(
            (2, 1, 1, 2),
            int(InteractionType.NONE),
            dtype=np.int32,
        )
        interactions[0, 0, 0, 0] = int(InteractionType.SPECULAR)
        self.interactions = _Tensor(interactions)
        vertices = np.zeros((2, 1, 1, 2, 3), dtype=float)
        vertices[0, 0, 0, 0] = (1.0, 0.1, 0.0)
        self.vertices = _Tensor(vertices)
        self.sources = _Tensor(np.asarray([[0.0], [0.0], [0.0]]))
        self.targets = _Tensor(np.asarray([[0.0], [0.0], [0.0]]))
        self.tau = _Tensor(np.asarray([[[10e-9, 20e-9]]]))

    def cir(self, *, normalize_delays, out_type):
        assert normalize_delays is False
        assert out_type == "numpy"
        coefficients = np.zeros((1, 1, 1, 1, 2, 1), dtype=np.complex64)
        coefficients[..., 0, :] = 2.0
        coefficients[..., 1, :] = 1.0
        return coefficients, self.tau.numpy()


def test_rt_diagnostics_select_target_paths_and_report_depths():
    diagnostics = summarize_rt_paths(
        _Paths(),
        top_k=4,
        required_object_ids=(7,),
    )

    assert diagnostics.selected_path_indices.tolist() == [0]
    assert diagnostics.selected_path_magnitudes.tolist() == [2.0]
    assert diagnostics.selected_path_lengths_m[0] == pytest.approx(
        2.99792458
    )
    assert diagnostics.segment_starts_m.shape == (2, 3)
    assert diagnostics.segment_ends_m.shape == (2, 3)
    assert diagnostics.depth_histogram[:2].tolist() == [1, 1]
    assert diagnostics.valid_link_path_count == 2


def test_po_diagnostics_preserve_original_face_indices():
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = np.asarray([(0, 1, 2), (0, 3, 1), (0, 2, 3)])
    diagnostics = summarize_po_surface(
        vertices,
        faces,
        face_power=np.asarray([0.1, 4.0, 2.0]),
        face_visibility=np.asarray([0.0, 1.0, 0.5]),
        top_k=2,
    )

    assert diagnostics.face_indices.tolist() == [1, 2]
    assert diagnostics.face_power.tolist() == [4.0, 2.0]
    assert diagnostics.visible_face_count == 2
    assert np.allclose(np.linalg.norm(diagnostics.face_normals, axis=1), 1.0)


def test_prepared_room_motion_is_radar_relative():
    vertices = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.4]],
            [[0.1, 0.0, 0.0], [0.3, 0.0, 0.0], [0.1, 0.2, 0.4]],
        ],
        dtype=np.float32,
    )
    sequence = MeshSequence(
        vertices=vertices,
        faces=np.asarray([(0, 1, 2)], dtype=np.uint32),
        times=np.asarray([0.0, 1.0 / 30.0]),
    )
    preview = prepare_human_room_preview(
        sequence,
        human_motion_filename="motion.npz",
        human_position_m=(1.5, -0.25, 0.2),
        human_yaw_deg=90.0,
    )

    assert sequence.times.tolist() == pytest.approx([0.0, 1.0 / 30.0])
    initial = preview.mesh_sequence.vertices_at(0.0)
    initial_center = 0.5 * (
        np.min(initial, axis=0) + np.max(initial, axis=0)
    )
    assert initial_center == pytest.approx((1.5, -0.25, 0.2))
    floor = next(box for box in preview.room_boxes if box["id"] == "floor")
    assert floor["translate"][2] < -1.0
    assert preview.source_name == "motion.npz"


@pytest.mark.parametrize(
    ("up", "forward", "expected"),
    [
        ("+z", "+x", np.eye(3)),
        ("+y", "+z", np.asarray([[0, 0, 1], [1, 0, 0], [0, 1, 0]])),
        ("+z", "-y", np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])),
        ("-z", "+x", np.diag([1, -1, -1])),
    ],
)
def test_motion_frame_transform_matches_declared_source_axes(up, forward, expected):
    transform = motion_source_to_radar_transform(up, forward)

    np.testing.assert_array_equal(transform, expected)
    assert np.linalg.det(transform) == pytest.approx(1.0)


def test_manual_motion_frame_overrides_metadata_before_placement():
    vertices = np.asarray(
        [
            [[0, 0, 0], [10, 0, 0], [0, 200, 0]],
            [[0, 0, 10], [10, 0, 10], [0, 200, 10]],
        ], dtype=np.float32,
    )
    payload = BytesIO()
    np.savez(
        payload, vertices=vertices, faces=np.asarray([[0, 1, 2]]),
        times=np.asarray([0.0, 0.1]),
        output_axes=np.asarray([1, 2, 0]),
        output_signs=np.asarray([-1, 1, -1]),
    )
    loaded = load_amass_motion(payload.getvalue(), filename="cm-mesh.npz")
    preview = prepare_human_room_preview(
        loaded,
        motion_source_up_axis="+y", motion_source_forward_axis="+z",
        motion_source_length_unit="cm", human_position_m=(1.6, 0, 0),
    )
    placed = preview.mesh_sequence.vertices_at(0)
    np.testing.assert_allclose(np.ptp(placed, axis=0), [0, 0.1, 2.0])
    np.testing.assert_allclose((placed.min(0) + placed.max(0)) / 2, [1.6, 0, 0])
    delta = preview.mesh_sequence.vertices_at(0.1) - placed
    np.testing.assert_allclose(delta, np.tile([0.1, 0, 0], (3, 1)), atol=1e-6)
    assert preview.motion_coordinate_settings["length_unit"] == "cm"
    assert preview.motion_coordinate_settings["up_axis"] == "+y"


def test_left_handed_motion_frame_preserves_open_surface_normals():
    sequence = MeshSequence(
        vertices=np.asarray([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32),
        faces=np.asarray([[0, 1, 2]]), times=np.asarray([0.0]),
    )
    preview = prepare_human_room_preview(
        sequence, motion_source_up_axis="+z", motion_source_forward_axis="+x",
        motion_source_handedness="left", human_position_m=(0, 0, 0),
    )
    triangle = preview.mesh_sequence.vertices_at(0)[preview.mesh_sequence.faces[0]]
    np.testing.assert_allclose(np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0]),
                               [0, 0, 1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"up_axis": "+z", "forward_axis": "-z"},
        {"up_axis": "unknown", "forward_axis": "+x"},
        {"up_axis": "+z", "forward_axis": "+x", "handedness": "unknown"},
        {"up_axis": "+z", "forward_axis": "+x", "length_unit": "feet"},
    ],
)
def test_motion_frame_transform_rejects_invalid_declarations(kwargs):
    with pytest.raises(ValueError):
        motion_source_to_radar_transform(**kwargs)


def test_prepared_room_rt_scene_uses_rough_wall_scattering():
    pytest.importorskip("sionna.rt")
    scene = _load_environment_scene(prepared_room_boxes())

    assert float(
        np.asarray(scene.radio_materials["wall"].scattering_coefficient)[0]
    ) == pytest.approx(0.2)
    assert scene.objects["floor"].radio_material is scene.radio_materials[
        "concrete"
    ]
    assert scene.objects["right_wall"].radio_material is scene.radio_materials[
        "wall"
    ]


def test_custom_scene_xml_replaces_prepared_preview_geometry():
    sequence = MeshSequence(
        vertices=np.asarray(
            [[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.4]]],
            dtype=np.float32,
        ),
        faces=np.asarray([(0, 1, 2)], dtype=np.uint32),
        times=np.asarray([0.0]),
    )
    scene_xml = b'<scene version="2.1.0"></scene>'

    validated, name = load_scene_xml(scene_xml, filename="lab.xml")
    preview = prepare_human_room_preview(
        sequence,
        scene_xml=scene_xml,
        scene_xml_filename="lab.xml",
    )

    assert validated == scene_xml
    assert name == "lab.xml"
    assert preview.room_boxes == ()
    assert preview.scene_xml == scene_xml
    assert preview.scene_source_name == "lab.xml"


def test_amass_loader_accepts_pose_parameters(monkeypatch, tmp_path):
    faces = np.asarray([(0, 1, 2)], dtype=np.uint32)

    class FakeSMPLSequence:
        def __init__(self, **kwargs):
            self.times = np.asarray(kwargs["times"], dtype=float)
            self.faces = np.asarray(kwargs["faces"], dtype=np.uint32)
            self.vertex_count = 3
            self.face_count = len(self.faces)

        def vertices_at(self, time_s):
            return np.asarray(
                [
                    [float(time_s), 0.0, 0.0],
                    [float(time_s), 0.2, 0.0],
                    [float(time_s), 0.0, 0.4],
                ],
                dtype=np.float32,
            )

    monkeypatch.setattr(
        "mmWaveRadar.experiments.human_room.AMASSSMPLMotionSequence",
        FakeSMPLSequence,
    )
    stream = BytesIO()
    np.savez_compressed(
        stream,
        poses=np.zeros((2, 72), dtype=np.float32),
        trans=np.zeros((2, 3), dtype=np.float32),
        betas=np.zeros(10, dtype=np.float32),
        times=np.asarray([0.0, 0.1]),
        faces=faces,
        gender=np.asarray("neutral"),
        model_type=np.asarray("smpl"),
    )

    loaded = load_amass_motion(
        stream.getvalue(),
        filename="motion.npz",
        smpl_model_dir=tmp_path,
    )

    assert loaded.source_name == "motion.npz"
    assert loaded.sequence.times.tolist() == [0.0, 0.1]
    assert loaded.payload["poses"].shape == (2, 72)
    assert loaded.payload["faces"].shape == (1, 3)


def test_motion_loader_prefers_supplied_meshes_over_pose_parameters(monkeypatch):
    def fail_if_smpl_is_used(**_kwargs):
        pytest.fail("precomputed mesh motion must not evaluate SMPL")

    monkeypatch.setattr(
        "mmWaveRadar.experiments.human_room.AMASSSMPLMotionSequence",
        fail_if_smpl_is_used,
    )
    vertices = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.4]],
            [[0.1, 0.0, 0.0], [0.3, 0.0, 0.0], [0.1, 0.2, 0.0], [0.1, 0.0, 0.4]],
        ],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 3]], dtype=np.int32)
    stream = BytesIO()
    np.savez(
        stream,
        vertices=vertices,
        faces=faces,
        times=np.asarray([10.0, 10.1]),
        poses=np.zeros((2, 72), dtype=np.float32),
        trans=np.zeros((2, 3), dtype=np.float32),
        betas=np.zeros(300, dtype=np.float32),
    )

    loaded = load_amass_motion(stream.getvalue(), filename="custom-mesh.npz")

    assert loaded.source_name == "custom-mesh.npz"
    assert loaded.payload is None
    assert loaded.sequence.vertex_count == 4
    np.testing.assert_array_equal(loaded.sequence.faces, faces)
    np.testing.assert_allclose(loaded.sequence.vertices_at(10.0), vertices[0])
    np.testing.assert_allclose(
        loaded.sequence.vertices_at(10.05), np.mean(vertices, axis=0), atol=1e-7
    )
    preview = prepare_human_room_preview(loaded)
    assert preview.mesh_sequence.times.tolist() == pytest.approx([0.0, 0.1])
    assert preview.motion_payload is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("faces", np.asarray([[0, 1, 3]]), "outside vertices"),
        ("times", np.asarray([0.0]), "match its frame count"),
        ("times", np.asarray([0.0, 0.0]), "strictly increasing"),
        ("vertices", np.full((2, 3, 3), np.nan), "finite values"),
        ("output_axes", np.asarray([0, 0, 1]), "permutation"),
    ],
)
def test_mesh_motion_loader_validates_supplied_geometry(field, value, message):
    arrays = {
        "vertices": np.zeros((2, 3, 3), dtype=np.float32),
        "faces": np.asarray([[0, 1, 2]], dtype=np.int32),
        "times": np.asarray([0.0, 0.1]),
    }
    arrays[field] = value
    stream = BytesIO()
    np.savez(stream, **arrays)

    with pytest.raises(ValueError, match=message):
        load_amass_motion(stream.getvalue(), filename="mesh-motion.npz")


def test_small_human_room_run_uses_fixed_origin_and_hybrid_components():
    pytest.importorskip("sionna.rt")
    sequence = MeshSequence(
        vertices=np.asarray(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.0],
                    [0.0, 0.2, 0.0],
                    [0.0, 0.0, 0.4],
                ],
                [
                    [0.01, 0.0, 0.0],
                    [0.21, 0.0, 0.0],
                    [0.01, 0.2, 0.0],
                    [0.01, 0.0, 0.4],
                ],
                [
                    [0.02, 0.0, 0.0],
                    [0.22, 0.0, 0.0],
                    [0.02, 0.2, 0.0],
                    [0.02, 0.0, 0.4],
                ],
            ],
            dtype=np.float32,
        ),
        faces=np.asarray(
            [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)],
            dtype=np.uint32,
        ),
        times=np.asarray([0.0, 0.1, 0.2]),
    )
    progress_events = []
    result = run_human_room_experiment(
        human_motion=sequence,
        human_motion_filename="tiny.npz",
        scene_xml=b'<scene version="2.1.0"></scene>',
        scene_xml_filename="empty-lab.xml",
        num_adc_samples=16,
        num_chirps_per_frame=1,
        tdm_enabled=False,
        selected_tx_indices=(2,),
        selected_rx_indices=(0, 3),
        coupling_enabled=False,
        rt_samples_per_source=100,
        rt_max_paths_per_source=100,
        rt_max_depth=1,
        po_visibility_samples_per_face=1,
        po_integration_mode="face_centroid",
        po_quadrature_max_refinement_depth=0,
        po_quadrature_max_subfaces_per_parent=1,
        begin_frame_index=1,
        end_frame_index=2,
        progress_callback=progress_events.append,
    )

    assert result.radar_position_m.tolist() == [0.0, 0.0, 0.0]
    assert result.simulation_mode == "hybrid_po"
    assert result.metadata.po_calibration["mode"] == "none"
    assert result.manifest.solver.parameters[
        "po_calibration_mode"
    ] == "none"
    assert result.manifest.scene.scenario == "custom_xml"
    assert result.manifest.scene.parameters["scene_source"] == "empty-lab.xml"
    assert result.adc.shape == (2, 1, 16, 2)
    assert result.range_time_power.shape[0] == 2
    assert result.range_doppler_power.shape[:2] == (2, 1)
    assert result.manifest.scene.parameters[
        "motion_begin_frame_index"
    ] == 1
    assert result.manifest.scene.parameters[
        "motion_end_frame_index"
    ] == 2
    assert result.manifest.scene.parameters[
        "source_motion_frame_count"
    ] == 3
    assert result.manifest.solver.parameters["num_frames"] == 2
    assert result.manifest.radar.parameters[
        "tx_indices_zero_based"
    ] == [2]
    assert result.manifest.radar.parameters[
        "rx_indices_zero_based"
    ] == [0, 3]
    assert {
        "human_po",
        "static_environment_blocked",
        "human_env",
        "env_human",
    }.issubset(result.components)
    assert all(
        component.shape == result.adc.shape
        for component in result.components.values()
    )
    assert progress_events
    assert {
        event["simulation_mode"] for event in progress_events
    } == {"hybrid_po"}
    assert max(event["frame_index"] for event in progress_events) == 1
    assert {
        event["num_frames"] for event in progress_events
    } == {2}
    diagnostics = run_human_room_diagnostics(
        result,
        frame_index=0,
        top_k=2,
        include_rt=True,
        include_po=True,
    )
    assert diagnostics.frame_index == 0
    assert diagnostics.source_frame_index == 1
    assert diagnostics.rt is not None
    assert diagnostics.po is not None
    assert diagnostics.po_face_power.shape == (sequence.faces.shape[0],)
    assert diagnostics.po_face_visibility.shape == (sequence.faces.shape[0],)
    assert diagnostics.rt.segment_starts_m.shape[1:] == (3,)
    assert diagnostics.rt.segment_ends_m.shape == (
        diagnostics.rt.segment_starts_m.shape
    )


@pytest.mark.parametrize(
    ("integration_mode", "stage_prefix"),
    [
        ("parent_face_quadrature", "quadrature"),
        ("parent_face_far_field_analytic", "far_field"),
    ],
)
def test_human_room_po_face_progress_preserves_the_current_frame(
    integration_mode, stage_prefix,
):
    pytest.importorskip("sionna.rt")
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.4],
        ],
        dtype=np.float32,
    )
    # Rotation triggers exact integration rather than a cached phase update.
    angle = np.deg2rad(20.0)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0.0],
         [np.sin(angle), np.cos(angle), 0.0],
         [0.0, 0.0, 1.0]],
    )
    sequence = MeshSequence(
        vertices=np.stack([
            vertices, vertices @ rotation.T + [0.1, 0.0, 0.0],
        ]),
        faces=np.asarray(
            [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)],
            dtype=np.uint32,
        ),
        times=np.asarray([0.0, 0.05]),
    )
    events = []

    run_human_room_experiment(
        human_motion=sequence,
        scene_xml=b'<scene version="2.1.0"></scene>',
        simulation_mode="human_only_po",
        num_adc_samples=8,
        num_chirps_per_frame=1,
        selected_tx_indices=(2,),
        selected_rx_indices=(0,),
        begin_frame_index=0,
        end_frame_index=1,
        po_integration_mode=integration_mode,
        po_quadrature_phase_span_scale_rad=np.pi / 8,
        po_quadrature_max_refinement_depth=3,
        po_quadrature_max_subfaces_per_parent=64,
        progress_callback=events.append,
    )

    completed = [
        event for event in events
        if event.get("stage") == f"{stage_prefix}_aggregate_done"
    ]
    assert {event["frame_index"] for event in completed} == {0, 1}
    assert all(event["simulation_mode"] == "human_only_po" for event in events)
    assert all(event["num_frames"] == 2 for event in events)
    for event in completed:
        assert 0 < event["parent_faces_total"] <= sequence.face_count
        assert event["parent_faces_processed"] == event["parent_faces_total"]
        assert event["subfaces_processed"] == event["subfaces_total"]
        assert event["subfaces_total"] >= event["parent_faces_total"]


@pytest.mark.parametrize(
    ("simulation_mode", "metadata_mode"),
    [
        ("full_rt", "rt_retrace"),
        ("coherent_rt", "rt_coherent_bank"),
        ("human_only_po", "human_only_po"),
    ],
)
def test_small_human_room_exposes_each_standalone_solver_mode(
    simulation_mode,
    metadata_mode,
):
    pytest.importorskip("sionna.rt")
    sequence = MeshSequence(
        vertices=np.asarray(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.0],
                    [0.0, 0.2, 0.0],
                    [0.0, 0.0, 0.4],
                ]
            ],
            dtype=np.float32,
        ),
        faces=np.asarray(
            [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)],
            dtype=np.uint32,
        ),
        times=np.asarray([0.0]),
    )
    result = run_human_room_experiment(
        human_motion=sequence,
        human_motion_filename="tiny.npz",
        scene_xml=b'<scene version="2.1.0"></scene>',
        scene_xml_filename="empty-lab.xml",
        simulation_mode=simulation_mode,
        num_adc_samples=8,
        num_chirps_per_frame=1,
        tdm_enabled=False,
        selected_tx_indices=(2,),
        selected_rx_indices=(0,),
        rt_samples_per_source=100,
        rt_max_paths_per_source=100,
        rt_max_depth=1,
        po_visibility_samples_per_face=1,
        po_integration_mode="face_centroid",
        po_quadrature_max_refinement_depth=0,
        po_quadrature_max_subfaces_per_parent=1,
    )

    assert result.simulation_mode == simulation_mode
    assert result.metadata.mode == metadata_mode
    assert result.adc.shape == (1, 1, 8, 1)
    assert result.range_time_power.shape[0] == 1
    assert result.range_doppler_power.shape[:2] == (1, 1)
