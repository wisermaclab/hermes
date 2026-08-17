# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Reproducible experiment contracts and curated public experiment runners."""

from .human_room import (
    HUMAN_ROOM_SIMULATION_MODES,
    HumanRoomDiagnostics,
    HumanRoomPreview,
    HumanRoomResult,
    LoadedAMASSMotion,
    PREPARED_ROOM_PRESETS,
    load_amass_motion,
    load_human_motion,
    load_scene_xml,
    prepared_room_boxes,
    prepare_human_room_preview,
    run_human_room_diagnostics,
    run_human_room_experiment,
)
from .manifest import (
    EXPERIMENT_MANIFEST_SCHEMA_VERSION,
    ExperimentManifest,
    MeasurementReference,
    RadarExperimentConfig,
    SceneExperimentConfig,
    SolverExperimentConfig,
)
from .physics_microscope import (
    MATERIAL_PRESETS,
    PhysicsMicroscopeResult,
    TARGET_TYPES,
    load_human_mesh,
    make_rectangular_plate_mesh,
    make_trihedral_corner_mesh,
    material_preset_parameters,
    run_physics_microscope,
    run_static_target_diagnostics,
    run_static_target_experiment,
)
from .solver_diagnostics import (
    POSurfaceDiagnostics,
    RTPathDiagnostics,
    SolverDiagnostics,
    summarize_po_surface,
    summarize_rt_paths,
)

__all__ = [
    "EXPERIMENT_MANIFEST_SCHEMA_VERSION",
    "ExperimentManifest",
    "HUMAN_ROOM_SIMULATION_MODES",
    "HumanRoomDiagnostics",
    "HumanRoomPreview",
    "HumanRoomResult",
    "LoadedAMASSMotion",
    "MeasurementReference",
    "MATERIAL_PRESETS",
    "POSurfaceDiagnostics",
    "PREPARED_ROOM_PRESETS",
    "PhysicsMicroscopeResult",
    "RadarExperimentConfig",
    "SceneExperimentConfig",
    "SolverExperimentConfig",
    "SolverDiagnostics",
    "TARGET_TYPES",
    "RTPathDiagnostics",
    "load_human_mesh",
    "load_amass_motion",
    "load_human_motion",
    "load_scene_xml",
    "make_rectangular_plate_mesh",
    "make_trihedral_corner_mesh",
    "material_preset_parameters",
    "prepared_room_boxes",
    "prepare_human_room_preview",
    "run_human_room_diagnostics",
    "run_human_room_experiment",
    "run_physics_microscope",
    "run_static_target_diagnostics",
    "run_static_target_experiment",
    "summarize_po_surface",
    "summarize_rt_paths",
]
