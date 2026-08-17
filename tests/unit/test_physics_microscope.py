# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Tests for the data-free public Physics Microscope scenario."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest

import mmWaveRadar.experiments.physics_microscope as physics_module
from mmWaveRadar.experiments import (
    load_human_mesh,
    make_rectangular_plate_mesh,
    make_trihedral_corner_mesh,
    material_preset_parameters,
    run_physics_microscope,
    run_static_target_experiment,
)
from mmWaveRadar.targets import face_winding_signed_volume


def test_static_material_diffuse_scattering_coefficients():
    pec = material_preset_parameters("pec")
    aluminum = material_preset_parameters("aluminum")
    concrete = material_preset_parameters("concrete")

    assert pec["diffuse_coefficient"] == 0.20
    assert aluminum["diffuse_coefficient"] == 0.0
    assert concrete["diffuse_coefficient"] == 0.35
    assert pec["backscattering_lambda"] == 0.20
    assert aluminum["backscattering_lambda"] == 0.0
    assert concrete["backscattering_lambda"] == 0.35


def test_plate_mesh_faces_radar_and_respects_tessellation():
    vertices, faces = make_rectangular_plate_mesh(
        range_m=2.0,
        lateral_m=0.4,
        vertical_m=0.7,
        width_m=0.2,
        height_m=0.1,
        aspect_deg=0.0,
        max_edge_m=0.05,
    )

    normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    assert vertices.shape == (15, 3)
    assert faces.shape == (16, 3)
    assert np.all(normals[:, 0] < 0.0)
    assert np.allclose(vertices[:, 0], 2.0)
    assert np.mean(vertices[:, 1]) == pytest.approx(0.4)
    assert np.mean(vertices[:, 2]) == pytest.approx(0.7)


def test_physics_microscope_produces_exportable_live_result():
    result = run_physics_microscope(
        aspect_deg=20.0,
        range_m=2.0,
        fidelity="preview",
    )

    assert result.manifest.scene.scenario == "static_plate"
    assert result.manifest.solver.mode == "hybrid_rt_po"
    solver_parameters = result.manifest.solver.parameters
    assert (
        solver_parameters["po_integration_mode"]
        == "parent_face_quadrature"
    )
    assert solver_parameters["po_quadrature_phase_span_scale_rad"] == 1.0
    assert solver_parameters["po_quadrature_max_refinement_depth"] == 2
    assert solver_parameters["po_quadrature_max_subfaces_per_parent"] == 16
    assert solver_parameters["rt_samples_per_source"] == 100_000
    assert solver_parameters["rt_max_paths_per_source"] == 5_000
    assert solver_parameters["rt_max_depth"] == 3
    assert result.po_adc.shape == (1, 16, 225, 12)
    assert result.rt_adc.shape == result.po_adc.shape
    assert result.face_power.shape == (result.faces.shape[0],)
    assert np.max(result.face_power) > 0.0
    assert result.visible_face_count > 0
    assert result.po_range_profile_power.shape == result.ranges_m.shape
    assert result.rt_range_profile_power.shape == result.ranges_m.shape
    assert result.rt_path_count >= 0
    assert result.po_runtime_s >= 0.0
    assert result.rt_runtime_s >= 0.0
    assert result.runtime_s >= 0.0


def test_static_experiment_uses_custom_fmcw_and_tdm_selection():
    result = run_static_target_experiment(
        target_type="plate",
        num_adc_samples=64,
        num_chirps_per_frame=4,
        carrier_frequency_hz=61.0e9,
        slope_hz_per_s=42.0e12,
        chirp_duration_s=40.0e-6,
        chirp_repetition_time_s=50.0e-6,
        sampling_frequency_hz=3.0e6,
        frame_period_s=20.0e-3,
        tdm_enabled=False,
        selected_tx_indices=(2,),
        selected_rx_indices=(1, 3),
        target_y_m=0.3,
        target_z_m=0.4,
        radar_yaw_deg=12.0,
        radar_pitch_deg=-7.0,
        radar_roll_deg=4.0,
        antenna_pattern_mode="none",
        cosine_3db_beamwidth_deg=42.0,
        po_integration_mode="face_centroid",
        rt_samples_per_source=1_000,
        rt_max_paths_per_source=750,
        rt_max_depth=2,
    )

    assert result.po_adc.shape == (1, 4, 64, 2)
    assert result.sensor_parameters["tdm_enabled"] is False
    assert result.sensor_parameters["tx_indices_zero_based"] == [2]
    assert result.sensor_parameters["rx_indices_zero_based"] == [1, 3]
    assert result.sensor_parameters["orientation"] == pytest.approx(
        np.deg2rad((12.0, -7.0, 4.0))
    )
    assert result.sensor_parameters["orientation_deg"] == pytest.approx(
        (12.0, -7.0, 4.0)
    )
    assert result.sensor_parameters["pattern_mode"] == "none"
    assert result.sensor_parameters["cosine_3db_beamwidth_deg"] == 42.0
    assert result.target_parameters["position_m"] == [2.0, 0.3, 0.4]
    assert result.target_parameters["target_y_m"] == 0.3
    assert result.target_parameters["target_z_m"] == 0.4
    assert result.target_parameters["distance_to_radar_m"] == pytest.approx(
        np.sqrt(4.25)
    )
    assert result.target_parameters["rt_diffuse_reflection_coefficient"] == 0.20
    assert result.target_parameters["rt_backscattering_lambda"] == 0.20
    fmcw = result.sensor_parameters["fmcw"]
    assert fmcw["carrier_frequency_hz"] == pytest.approx(61.0e9)
    assert fmcw["slope_hz_per_s"] == pytest.approx(42.0e12)
    assert fmcw["num_adc_samples"] == 64
    assert fmcw["num_chirps_per_frame"] == 4
    assert fmcw["num_tx"] == 1
    assert fmcw["tdm_enabled"] is False
    radar_parameters = result.manifest.radar.parameters
    assert radar_parameters["tdm_enabled"] is False
    assert radar_parameters["tx_indices_zero_based"] == [2]
    assert radar_parameters["rx_indices_zero_based"] == [1, 3]
    assert radar_parameters["radar_yaw_deg"] == 12.0
    assert radar_parameters["radar_pitch_deg"] == -7.0
    assert radar_parameters["radar_roll_deg"] == 4.0
    assert radar_parameters["antenna_pattern_mode"] == "none"
    assert radar_parameters["cosine_3db_beamwidth_deg"] == 42.0
    assert radar_parameters["sampling_frequency_hz"] == pytest.approx(3.0e6)
    solver_parameters = result.manifest.solver.parameters
    assert solver_parameters["po_integration_mode"] == "face_centroid"
    assert solver_parameters["rt_samples_per_source"] == 1_000
    assert solver_parameters["rt_max_paths_per_source"] == 750
    assert solver_parameters["rt_max_depth"] == 2


def test_plate_rejects_edge_on_or_invalid_dimensions():
    with pytest.raises(ValueError, match="aspect_deg"):
        make_rectangular_plate_mesh(
            range_m=2.0,
            width_m=0.2,
            height_m=0.1,
            aspect_deg=90.0,
            max_edge_m=0.05,
        )


def test_trihedral_mesh_and_uploaded_obj_human_run():
    vertices, faces = make_trihedral_corner_mesh(
        range_m=2.0,
        edge_m=0.2,
        max_edge_m=0.05,
    )
    normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    directions = -vertices[faces].mean(axis=1)
    assert faces.shape == (96, 3)
    assert np.all(np.sum(normals * directions, axis=1) > 0.0)
    # The three duplicated apex vertices sit behind the aperture: the opening
    # extends from x=2 toward the radar at the origin.
    apex = np.asarray((2.0, 0.0, 0.0))
    assert np.count_nonzero(np.all(np.isclose(vertices, apex), axis=1)) == 3
    aperture_direction = np.mean(vertices, axis=0) - apex
    aperture_direction /= np.linalg.norm(aperture_direction)
    assert aperture_direction == pytest.approx((-1.0, 0.0, 0.0), abs=1e-6)
    assert np.max(vertices[:, 0]) == pytest.approx(apex[0])

    obj = b"v 0 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\n"
    human_vertices, human_faces = load_human_mesh(obj, filename="person.obj")
    assert human_vertices.shape == (3, 3)
    assert human_faces.shape == (1, 3)
    result = run_static_target_experiment(
        target_type="human_mesh",
        human_mesh=obj,
        human_mesh_filename="person.obj",
        diffuse_reflection_coefficient=0.6,
    )
    assert result.target_type == "human_mesh"
    assert result.material_name == "human_tissue"
    assert result.target_parameters["diffuse_reflection_coefficient"] == 0.6
    assert result.rt_adc.shape == result.po_adc.shape


def test_human_mesh_loader_normalizes_inward_closed_mesh_winding():
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    outward_faces = np.asarray(
        [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)],
        dtype=np.uint32,
    )
    inward_faces = outward_faces[:, [0, 2, 1]]
    payload = BytesIO()
    np.savez(payload, vertices=vertices, faces=inward_faces)

    loaded_vertices, loaded_faces = load_human_mesh(
        payload.getvalue(),
        filename="inward-human.npz",
    )

    assert face_winding_signed_volume(vertices, inward_faces) < 0.0
    assert face_winding_signed_volume(loaded_vertices, loaded_faces) > 0.0
    np.testing.assert_array_equal(loaded_faces, outward_faces)


def test_human_obj_loader_caps_vertices_and_fan_triangulation(monkeypatch):
    obj = b"v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n"
    monkeypatch.setattr(physics_module, "_MAX_HUMAN_OBJ_VERTICES", 3)
    with pytest.raises(ValueError, match="vertex limit"):
        load_human_mesh(obj, filename="person.obj")

    monkeypatch.setattr(physics_module, "_MAX_HUMAN_OBJ_VERTICES", 4)
    monkeypatch.setattr(physics_module, "_MAX_HUMAN_OBJ_TRIANGLES", 1)
    with pytest.raises(ValueError, match="triangle limit"):
        load_human_mesh(obj, filename="person.obj")


def test_human_obj_loader_caps_input_bytes(monkeypatch):
    monkeypatch.setattr(physics_module, "_MAX_HUMAN_OBJ_BYTES", 3)

    with pytest.raises(ValueError, match="64 MiB input limit"):
        load_human_mesh(b"v 0 0 0\n", filename="person.obj")
