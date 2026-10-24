# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Compatibility and numerical parity of settings-based simulation APIs."""

from dataclasses import replace
import inspect

import numpy as np
import pytest
from sionna.rt import load_scene

from mmWaveRadar.experiments import (
    HumanMotionSettings,
    HumanRoomSettings,
    HumanRoomSolverSettings,
    HumanPOSettings,
    RoomSceneSettings,
    RadarSettings,
    StaticTargetSettings,
    StaticTargetSolverSettings,
    StaticPOSettings,
    StaticRayTracingSettings,
    run_human_room_experiment,
    run_static_target_experiment,
    simulate_human_room,
    simulate_static_target,
)
from mmWaveRadar.experiments.radar import build_radar
from mmWaveRadar.materials import human_skin_material, PhysicalOpticsMaterial
from mmWaveRadar.targets import MeshSequence, MeshTarget
from mmWaveRadar.simulation import (
    HybridPipelineSettings,
    HybridFrameSettings,
    HybridPhysicsSettings,
    HybridRayTracingSettings,
    HybridExecutionSettings,
    POConfig,
    POChannelSettings,
    POArraySettings,
    POIntegrationSettings,
    POVisibilitySettings,
    evaluate_po_channel,
    po_channel,
    run_calibrated_hybrid_po,
    run_hybrid_po_pipeline,
)


def mesh():
    vertices = np.array([[2.0, -0.1, -0.1], [2.0, -0.1, 0.1], [2.0, 0.1, -0.1]])
    return MeshSequence(
        vertices=np.stack([vertices, vertices + [0.02, 0.01, 0.0]]),
        faces=np.array([[0, 1, 2]], dtype=np.uint32),
        times=np.array([0.0, 0.2]),
    )


def human_settings():
    return HumanRoomSettings(
        motion=HumanMotionSettings(
            human_motion=mesh(), human_position_m=None, motion_start_s=0.03, num_frames=2
        ),
        scene=RoomSceneSettings(room_preset="empty_room"),
        radar=RadarSettings(
            num_adc_samples=16,
            num_chirps_per_frame=2,
            frame_period_s=0.025,
            antenna_pattern_mode="none",
            selected_tx_indices=(2,),
            selected_rx_indices=(1, 3),
        ),
        solver=HumanRoomSolverSettings(
            solver="po", po=HumanPOSettings(po_integration_mode="face_centroid")
        ),
    )


@pytest.mark.parametrize(
    "factory,legacy,excluded",
    [
        (
            lambda: HumanRoomSettings(motion=HumanMotionSettings(human_motion="placeholder")),
            run_human_room_experiment,
            {"human_motion"},
        ),
        (StaticTargetSettings, run_static_target_experiment, set()),
        (
            lambda: HybridPipelineSettings(frames=HybridFrameSettings(num_frames=2)),
            run_calibrated_hybrid_po,
            {"scene", "radar", "target", "coupling_scene", "_reflector_bank", "num_frames"},
        ),
        (
            lambda: POChannelSettings(
                array=POArraySettings(
                    tx_positions=np.zeros((1, 3)),
                    rx_positions=np.zeros((1, 3)),
                    frequency_hz=60e9,
                    material=PhysicalOpticsMaterial(
                        relative_permittivity=1.0, conductivity_s_per_m=0.0, model="pec"
                    ),
                )
            ),
            po_channel,
            {"vertices", "faces", "tx_positions", "rx_positions", "frequency_hz", "material"},
        ),
    ],
)
def test_settings_defaults_match_legacy_contract(factory, legacy, excluded):
    values = factory().as_keywords()
    for name, parameter in inspect.signature(legacy).parameters.items():
        if name in excluded or parameter.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        assert name in values
        if parameter.default is not inspect.Parameter.empty:
            assert values[name] == parameter.default, name


def test_human_settings_preserve_motion_clock_channel_subset_and_frame_products():
    settings = human_settings()
    actual = simulate_human_room(settings)
    legacy = run_human_room_experiment(**settings.as_keywords())
    assert actual.adc.shape == (2, 2, 16, 2)
    assert np.max(np.abs(actual.adc)) > 0
    np.testing.assert_allclose(actual.adc, legacy.adc, rtol=1e-12, atol=1e-18)
    np.testing.assert_array_equal(actual.adc_times_s, legacy.adc_times_s)
    np.testing.assert_array_equal(
        actual.preview.mesh_sequence.times, legacy.preview.mesh_sequence.times
    )
    assert actual.manifest.radar.parameters == legacy.manifest.radar.parameters


def test_static_settings_preserve_geometry_waveform_and_po_amplitude():
    settings = StaticTargetSettings(
        radar=RadarSettings(
            carrier_frequency_hz=61e9,
            num_adc_samples=32,
            num_chirps_per_frame=2,
            antenna_pattern_mode="none",
            orientation_deg=(12.0, -7.0, 4.0),
            selected_tx_indices=(2,),
            selected_rx_indices=(1, 3),
        ),
        solver=StaticTargetSolverSettings(
            po=StaticPOSettings(po_integration_mode="face_centroid"),
            rt=StaticRayTracingSettings(rt_samples_per_source=256, rt_max_paths_per_source=256),
        ),
    )
    actual = simulate_static_target(settings)
    legacy = run_static_target_experiment(**settings.as_keywords())
    np.testing.assert_array_equal(actual.vertices, legacy.vertices)
    np.testing.assert_allclose(actual.po_adc, legacy.po_adc, rtol=1e-12, atol=1e-18)
    assert actual.sensor_parameters["fmcw"] == legacy.sensor_parameters["fmcw"]


@pytest.mark.parametrize(
    "integration", ["face_centroid", "parent_face_quadrature", "parent_face_adaptive_analytic"]
)
def test_po_settings_preserve_complex_channel_and_wideband_response(integration):
    sequence = mesh()
    settings = POChannelSettings(
        array=POArraySettings(
            tx_positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.003, 0.0]]),
            rx_positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.002]]),
            frequency_hz=60.4e9,
            material=PhysicalOpticsMaterial(
                relative_permittivity=1.0, conductivity_s_per_m=0.0, model="pec"
            ),
            max_frequency_offset_hz=1e9,
        ),
        visibility=POVisibilitySettings(visibility="none"),
        integration=POIntegrationSettings(
            po_integration_mode=integration,
            po_quadrature_max_refinement_depth=2,
            po_quadrature_max_subfaces_per_parent=16,
        ),
    )
    actual = evaluate_po_channel(sequence.vertices[0], sequence.faces, settings=settings)
    legacy = po_channel(sequence.vertices[0], sequence.faces, **settings.as_keywords())
    assert np.max(np.abs(actual.coefficients)) > 0
    np.testing.assert_allclose(actual.coefficients, legacy.coefficients, rtol=1e-12, atol=1e-18)
    np.testing.assert_array_equal(actual.delays_s, legacy.delays_s)
    if actual.delay_response is not None:
        np.testing.assert_allclose(actual.delay_response.values, legacy.delay_response.values)


@pytest.mark.parametrize("tx_schedule", ["simultaneous", "tdm"])
def test_hybrid_settings_preserve_components_and_target_scene_lifecycle(tx_schedule):
    radar = build_radar(
        RadarSettings(
            num_adc_samples=16,
            num_chirps_per_frame=2,
            antenna_pattern_mode="none",
            selected_tx_indices=(0, 2),
            tx_schedule=tx_schedule,
            selected_rx_indices=(0,),
        )
    )
    target = MeshTarget("grouped-target", mesh(), material=human_skin_material("grouped-skin"))
    scene = load_scene()
    settings = HybridPipelineSettings(
        frames=HybridFrameSettings(num_frames=2),
        physics=HybridPhysicsSettings(
            hybrid_coupling=False,
            hybrid_calibration="none",
            po_config=POConfig(po_integration_mode="face_centroid"),
        ),
        rt=HybridRayTracingSettings(rt_samples_per_source=128, rt_max_paths_per_source=128),
        execution=HybridExecutionSettings(
            compute_backend="numpy",
            compute_precision="float64",
            progress=False,
            print_reflector_summary=False,
        ),
    )
    actual = run_hybrid_po_pipeline(scene=scene, radar=radar, target=target, settings=settings)
    legacy = run_calibrated_hybrid_po(
        scene=scene, radar=radar, target=target, **settings.as_keywords()
    )
    assert np.max(np.abs(actual.cube.adc)) > 0
    np.testing.assert_allclose(actual.cube.adc, legacy.cube.adc, rtol=1e-12, atol=1e-18)
    for key in actual.cube.components:
        np.testing.assert_allclose(
            actual.cube.components[key], legacy.cube.components[key], rtol=1e-12, atol=1e-18
        )
    assert target.name not in scene.objects


def test_replacing_nested_settings_keeps_original_input_and_arrays():
    original = human_settings()
    changed = replace(original, radar=replace(original.radar, selected_tx_indices=(1,)))
    assert original.radar.selected_tx_indices == (2,)
    assert changed.radar.selected_tx_indices == (1,)
    assert changed.motion.human_motion is original.motion.human_motion


@pytest.mark.parametrize(
    "module_name,legacy_name,new_name",
    [
        ("mmWaveRadar.experiments.human_room", "run_human_room_experiment", "simulate_human_room"),
        (
            "mmWaveRadar.experiments.physics_microscope",
            "run_static_target_experiment",
            "simulate_static_target",
        ),
        (
            "mmWaveRadar.simulation.hybrid_pipeline",
            "run_calibrated_hybrid_po",
            "run_hybrid_po_pipeline",
        ),
        ("mmWaveRadar.simulation.physical_optics", "po_channel", "evaluate_po_channel"),
    ],
)
def test_legacy_adapter_forwards_every_option_to_grouped_api(
    monkeypatch, module_name, legacy_name, new_name
):
    from importlib import import_module

    module = import_module(module_name)
    legacy = getattr(module, legacy_name)
    signature = inspect.signature(getattr(module, new_name))
    values = {
        name: object()
        for name, parameter in inspect.signature(legacy).parameters.items()
        if parameter.kind is not inspect.Parameter.VAR_KEYWORD
    }

    def capture(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        settings = bound.arguments.pop("settings")
        return {**settings.as_keywords(), **bound.arguments}

    monkeypatch.setattr(module, new_name, capture)
    forwarded = legacy(**values)
    assert forwarded.keys() == values.keys()
    for name in values:
        assert forwarded[name] is values[name], name


def test_grouped_human_run_preserves_cancellation_before_motion_loading():
    from mmWaveRadar.experiments import ExperimentExecutionSettings

    settings = HumanRoomSettings(
        motion=HumanMotionSettings(human_motion="missing-motion.npz"),
        solver=HumanRoomSolverSettings(
            execution=ExperimentExecutionSettings(cancel_check=lambda: True)
        ),
    )
    with pytest.raises(InterruptedError, match="cancelled"):
        simulate_human_room(settings)
