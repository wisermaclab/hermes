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


def test_prepared_room_rt_scene_uses_rough_wall_scattering():
    pytest.importorskip("sionna.rt")
    scene = _load_environment_scene(prepared_room_boxes())

    assert float(
        np.asarray(scene.radio_materials["wall"].scattering_coefficient)[0]
    ) == pytest.approx(0.2)
    assert float(
        np.asarray(
            scene.radio_materials["reflective-wall"].scattering_coefficient
        )[0]
    ) == pytest.approx(0.2)


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
    assert result.metadata.po_calibration["mode"] == "rt_sequence"
    assert result.manifest.solver.parameters[
        "po_calibration_mode"
    ] == "rt_sequence"
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
