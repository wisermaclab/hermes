# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""PO solver tests."""

from __future__ import annotations

from sensing_test_helpers import *  # noqa: F401,F403

import mmWaveRadar.simulation.physical_optics as physical_optics_module
import mmWaveRadar.simulation.simulator as simulator_module


def test_sorted_face_position_matching_matches_dictionary_lookup():
    rng = np.random.default_rng(42)
    for _ in range(100):
        previous = np.sort(rng.choice(20_000, size=rng.integers(0, 5_000), replace=False))
        selected = np.sort(rng.choice(20_000, size=rng.integers(0, 5_000), replace=False))
        previous_position = {int(face): position for position, face in enumerate(previous)}
        expected = np.asarray(
            [previous_position.get(int(face), -1) for face in selected],
            dtype=np.int64,
        )
        actual = simulator_module._match_sorted_face_positions(
            previous,
            selected,
        )
        np.testing.assert_array_equal(actual, expected)


def test_po_static_mesh_is_constant_across_chirps():
    scene = load_scene()
    radar = _phase_one_radar(num_chirps=3, chirp_repetition_time=0.25)
    target = MeshTarget(
        "human",
        _single_face_mesh_sequence(x0=1.0),
        material=human_skin_material("human-po-static"),
    )
    sim = MmWaveRadarSimulator(
        solver="po",
        po_config=POConfig(po_visibility_fade_frames=1),
    )

    cube = sim.run(scene, radar, [target], num_frames=1)

    assert cube.metadata.solver == "po"
    assert np.allclose(cube.adc[0, 0], cube.adc[0, 1])
    assert np.allclose(cube.adc[0, 1], cube.adc[0, 2])
    assert np.array_equal(cube.metadata.path_counts[0], [1, 1, 1])


def test_po_without_targets_preserves_chirp_times():
    scene = load_scene()
    radar = _phase_one_radar(num_chirps=3, chirp_repetition_time=0.25)
    sim = MmWaveRadarSimulator(solver="po")

    cube = sim.run(scene, radar, [], num_frames=2)

    expected = np.asarray(
        [[radar.fmcw.chirp_time(frame, chirp) for chirp in range(3)] for frame in range(2)]
    )
    assert np.array_equal(cube.times, expected)
    assert not np.any(cube.adc)


@pytest.mark.parametrize("calibration_mode", ["rt_first_pose", "rt_sequence"])
def test_po_calibration_scales_adc_and_keeps_baseline(calibration_mode):
    class CalibrationSimulator(MmWaveRadarSimulator):
        def _run_rt_baseline_for_calibration(
            self,
            scene,
            radar,
            targets,
            *,
            num_frames,
            rt_refresh_frames=1,
            calibration_reference_component=None,
        ):
            del scene, targets
            assert rt_refresh_frames == 1
            fmcw = radar.fmcw
            adc = np.full(
                (
                    num_frames,
                    fmcw.num_chirps_per_frame,
                    fmcw.num_adc_samples,
                    radar.hardware.num_virtual_channels,
                ),
                2.0 + 0.0j,
                dtype=np.complex128,
            )
            metadata = SensingMetadata(
                events=[],
                path_counts=np.ones(adc.shape[:2], dtype=np.int64),
                max_displacements=np.zeros(adc.shape[:2], dtype=float),
                virtual_channel_order=radar.hardware.virtual_channel_order,
                solver="rt",
            )
            return {
                "cube": RadarCube(adc=adc, times=np.zeros(adc.shape[:2]), metadata=metadata),
                "human_reference_adc": adc[0],
                "reference_adc_power": {
                    "single_human_only": np.mean(np.abs(adc) ** 2, axis=2),
                },
                "human_reference_path_counts": np.ones(
                    fmcw.num_chirps_per_frame,
                    dtype=np.int64,
                ),
                "retrace_count": int(num_frames),
                "rt_refresh_frames": int(rt_refresh_frames),
                "retrace_period_chirps": fmcw.num_chirps_per_frame,
                "coherent_path_update": True,
            }

    scene = load_scene()
    radar = _phase_one_radar(num_chirps=2, chirp_repetition_time=0.25)
    target = MeshTarget(
        "human",
        _single_face_mesh_sequence(x0=1.0),
        material=human_skin_material("human-po-calibration"),
    )
    sim = CalibrationSimulator(
        solver="po",
        po_config=POConfig(
            po_visibility_samples_per_face=1,
            po_calibration=POCalibrationConfig(
                mode=calibration_mode,
                rt_refresh_frames=1,
            ),
        ),
    )

    cube = sim.run(scene, radar, [target], num_frames=1)

    assert cube.metadata.po_calibration["applied"] is True
    assert cube.metadata.po_calibration["reference_power_kind"] == "adc_rms_power"
    assert np.isclose(np.mean(np.abs(cube.adc[0]) ** 2), 4.0)
    assert cube.metadata.po_calibration["rt_baseline_refresh_frames"] == 1
    assert cube.metadata.po_calibration["rt_baseline_retrace_period_chirps"] == 2
    assert cube.metadata.po_calibration["rt_baseline_coherent_path_update"] is True
    assert cube.components is not None
    assert cube.components["rt_baseline"].shape == cube.adc.shape
    assert cube.components["rt_baseline_full_scene"].shape == cube.adc.shape
    assert cube.components["rt_baseline_one_human_first_frame"].shape == cube.adc.shape


def test_po_parent_face_quadrature_reuses_parent_visibility(monkeypatch):
    scene = load_scene()
    radar = _phase_one_radar(num_chirps=3, chirp_repetition_time=0.25)
    vertices = np.stack(
        [
            _front_facing_triangle(1.0, scale=2.0),
            _front_facing_triangle(1.0, scale=2.0),
        ]
    )
    target = MeshTarget(
        "human",
        MeshSequence(
            vertices=vertices,
            faces=np.array([[0, 1, 2]], dtype=np.uint32),
            times=np.array([0.0, 1.0], dtype=float),
        ),
        material=human_skin_material("human-po-parent-quadrature-cache"),
    )
    visibility_calls = {"count": 0}
    original_visibility = physical_optics_module._fractional_face_visibility

    def wrapped_visibility(*args, **kwargs):
        visibility_calls["count"] += 1
        return original_visibility(*args, **kwargs)

    monkeypatch.setattr(
        physical_optics_module,
        "_fractional_face_visibility",
        wrapped_visibility,
    )
    sim = MmWaveRadarSimulator(
        solver="po",
        po_config=POConfig(
            po_visibility_samples_per_face=1,
            po_refresh_frames=1,
            po_visibility_refresh_frames=1,
            po_integration_mode="parent_face_quadrature",
            po_quadrature_phase_span_scale_rad=1.0,
            po_quadrature_max_refinement_depth=1,
        ),
    )

    cube = sim.run(scene, radar, [target], num_frames=1)

    assert visibility_calls["count"] == 1
    assert np.array_equal(cube.metadata.path_counts[0], [1, 1, 1])


def test_po_parent_face_quadrature_uses_parent_visibility_topology():
    scene = load_scene()
    radar = _phase_one_radar(num_chirps=2, chirp_repetition_time=0.25)
    vertices = np.stack(
        [
            _front_facing_triangle(1.0, scale=2.0),
            _front_facing_triangle(1.0, scale=2.0),
        ]
    )
    target = MeshTarget(
        "human",
        MeshSequence(
            vertices=vertices,
            faces=np.array([[0, 1, 2]], dtype=np.uint32),
            times=np.array([0.0, 1.0], dtype=float),
        ),
        material=human_skin_material("human-po-parent-quadrature"),
    )
    sim = MmWaveRadarSimulator(
        solver="po",
        po_config=POConfig(
            po_visibility_samples_per_face=1,
            po_refresh_frames=1,
            po_integration_mode="parent_face_quadrature",
            po_quadrature_phase_span_scale_rad=1.0,
            po_quadrature_max_refinement_depth=1,
        ),
    )

    cube = sim.run(scene, radar, [target], num_frames=1)

    assert np.array_equal(cube.metadata.visible_face_counts[0], [1, 1])
    assert np.array_equal(cube.metadata.path_counts[0], [1, 1])
    assert np.array_equal(cube.metadata.incremental_full_refresh[0], [True, False])
    assert np.array_equal(cube.metadata.incremental_recomputed_face_counts[0], [1, 0])
    assert np.array_equal(cube.metadata.incremental_phase_updated_face_counts[0], [0, 1])
    assert cube.metadata.events[0]["po_integration_mode"] == "parent_face_quadrature"


def test_po_parent_face_quadrature_incremental_recompute_uses_mode(monkeypatch):
    scene = load_scene()
    radar = _phase_one_radar(num_chirps=2, chirp_repetition_time=0.25)
    vertices = np.stack(
        [
            _front_facing_triangle(1.0, scale=2.0),
            _front_facing_triangle(1.001, scale=2.0),
        ]
    )
    target = MeshTarget(
        "human",
        MeshSequence(
            vertices=vertices,
            faces=np.array([[0, 1, 2]], dtype=np.uint32),
            times=np.array([0.0, 1.0], dtype=float),
        ),
        material=human_skin_material("human-po-parent-quadrature-incremental"),
    )
    calls = []
    original_geometry = simulator_module.parent_face_quadrature_geometry

    def wrapped_geometry(*args, **kwargs):
        calls.append(np.asarray(kwargs["selected_parent_indices"]).copy())
        return original_geometry(*args, **kwargs)

    monkeypatch.setattr(
        simulator_module,
        "parent_face_quadrature_geometry",
        wrapped_geometry,
    )
    sim = MmWaveRadarSimulator(
        solver="po",
        po_config=POConfig(
            po_visibility_samples_per_face=1,
            po_refresh_frames=1,
            po_visibility_refresh_frames=1,
            po_refresh_on_phase_error_rad=0.0,
            po_integration_mode="parent_face_quadrature",
            po_quadrature_phase_span_scale_rad=1.0,
            po_quadrature_max_refinement_depth=1,
        ),
    )

    cube = sim.run(scene, radar, [target], num_frames=1)

    assert np.array_equal(cube.metadata.incremental_full_refresh[0], [True, False])
    assert np.array_equal(cube.metadata.incremental_recomputed_face_counts[0], [1, 1])
    assert len(calls) == 1
    assert np.array_equal(calls[0], np.array([0], dtype=np.int64))


def test_po_parent_far_field_refreshes_edge_rule_with_visibility(monkeypatch):
    scene = load_scene()
    radar = _phase_one_radar(num_chirps=3, chirp_repetition_time=0.25)
    vertices = np.stack(
        [
            _front_facing_triangle(1.0, scale=0.2),
            _front_facing_triangle(1.0, scale=0.2),
        ]
    )
    target = MeshTarget(
        "human",
        MeshSequence(
            vertices=vertices,
            faces=np.array([[0, 1, 2]], dtype=np.uint32),
            times=np.array([0.0, 1.0], dtype=float),
        ),
        material=human_skin_material("human-po-parent-far-field-incremental"),
    )
    calls = []
    original_edge_rule = simulator_module.parent_face_far_field_edge_rule

    def wrapped_edge_rule(*args, **kwargs):
        calls.append(True)
        return original_edge_rule(*args, **kwargs)

    monkeypatch.setattr(
        simulator_module,
        "parent_face_far_field_edge_rule",
        wrapped_edge_rule,
    )
    sim = MmWaveRadarSimulator(
        solver="po",
        po_config=POConfig(
            po_visibility_samples_per_face=1,
            po_refresh_frames=2,
            po_visibility_refresh_frames=1,
            po_refresh_on_phase_error_rad=0.0,
            po_integration_mode="parent_face_far_field_analytic",
            po_quadrature_max_refinement_depth=4,
        ),
    )

    cube = sim.run(scene, radar, [target], num_frames=2)

    # Exact solve at frame 0; visibility (and the edge rule) refresh at frame 1.
    assert np.array_equal(
        cube.metadata.incremental_full_refresh,
        [[True, False, False], [False, False, False]],
    )
    assert np.array_equal(
        cube.metadata.incremental_visibility_refresh,
        [[False, False, False], [True, False, False]],
    )
    assert np.array_equal(
        cube.metadata.incremental_recomputed_face_counts,
        [[1, 0, 0], [0, 0, 0]],
    )
    assert len(calls) == 1


def test_po_incremental_visibility_refresh_is_not_full_rebuild():
    scene = load_scene()
    radar = _phase_one_radar(num_chirps=2, chirp_repetition_time=0.25)
    target = MeshTarget(
        "human",
        _single_face_mesh_sequence(x0=1.0),
        material=human_skin_material("human-po-incremental-refresh"),
    )
    sim = MmWaveRadarSimulator(
        solver="po",
        po_config=POConfig(
            po_visibility_samples_per_face=1,
            po_visibility_fade_frames=1,
            po_refresh_frames=2,
            po_visibility_refresh_frames=1,
            po_refresh_on_phase_error_rad=0.1,
        ),
    )

    cube = sim.run(scene, radar, [target], num_frames=3)

    # Full refreshes every two frames; visibility alone in between.
    assert np.array_equal(
        cube.metadata.incremental_full_refresh,
        [[True, False], [False, False], [True, False]],
    )
    assert np.array_equal(
        cube.metadata.incremental_visibility_refresh,
        [[False, False], [True, False], [False, False]],
    )
    assert np.array_equal(
        cube.metadata.incremental_recomputed_face_counts,
        [[1, 0], [0, 0], [1, 0]],
    )
    assert np.array_equal(cube.metadata.path_counts, np.ones((3, 2)))


def test_po_incremental_path_phase_drift_triggers_recompute():
    scene = load_scene()
    radar = _phase_one_radar(num_chirps=3, chirp_repetition_time=1.0)
    target = MeshTarget(
        "human",
        _single_face_mesh_sequence(x0=1.0, x1=1.006),
        material=human_skin_material("human-po-incremental-drift"),
    )
    sim = MmWaveRadarSimulator(
        solver="po",
        po_config=POConfig(
            po_visibility_samples_per_face=1,
            po_refresh_frames=1,
            po_visibility_refresh_frames=1,
            po_refresh_on_phase_error_rad=0.001,
        ),
    )

    cube = sim.run(scene, radar, [target], num_frames=1)

    assert np.array_equal(
        cube.metadata.incremental_full_refresh[0],
        [True, False, False],
    )
    assert np.array_equal(
        cube.metadata.incremental_visibility_refresh[0],
        [False, False, False],
    )
    assert np.array_equal(
        cube.metadata.incremental_recomputed_face_counts[0],
        [1, 1, 1],
    )


@pytest.mark.parametrize("incremental", [False, True])
def test_po_rigid_translation_matches_expected_phase_slope(incremental):
    scene = load_scene()
    radar = _phase_one_radar(num_chirps=3, chirp_repetition_time=1.0)
    target = MeshTarget(
        "human",
        _single_face_mesh_sequence(x0=1.0, x1=1.002),
        material=human_skin_material("human-po-motion"),
    )
    sim = MmWaveRadarSimulator(
        solver="po",
        po_config=POConfig(
            po_refresh_frames=int(incremental),
        ),
    )

    cube = sim.run(scene, radar, [target], num_frames=1)

    phase = np.unwrap(np.angle(cube.adc[0, :, 0, 0]))
    measured = np.diff(phase)
    centroids = np.array(
        [target.mesh_sequence.vertices_at(float(t)).mean(axis=0) for t in cube.times[0]]
    )
    path_lengths = 2.0 * np.linalg.norm(centroids, axis=1)
    expected = 2.0 * np.pi * np.diff(path_lengths) / radar.fmcw.wavelength
    assert np.allclose(measured, expected, atol=1e-2)
