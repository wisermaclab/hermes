# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Typed settings for grouped experiment and solver entry points.

Arrays and scene/model objects are retained by reference; create a new settings
object with dataclasses.replace to change one option between runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Callable, TYPE_CHECKING

from ..targets import MeshSequence
from ..simulation.types import POConfig, DEFAULT_PO_REFRESH_ON_PHASE_ERROR_RAD
from ..simulation.options import ArraySettings
from .radar import RadarSettings

if TYPE_CHECKING:
    from .human_room import LoadedAMASSMotion


@dataclass(frozen=True, kw_only=True)
class HumanMotionSettings:
    """Motion settings for simulate_human_room."""

    human_motion: str | Path | bytes | bytearray | BytesIO | MeshSequence | LoadedAMASSMotion
    human_motion_filename: str | None = None
    smpl_model_dir: str | Path | None = None
    human_position_m: tuple[float, float, float] | None = (1.6, 0.0, 0.0)
    human_yaw_deg: float = 0.0
    motion_source_up_axis: str | None = None
    motion_source_forward_axis: str | None = None
    motion_source_handedness: str = "right"
    motion_source_length_unit: str = "m"
    motion_start_s: float = 0.0
    motion_end_s: float | None = None
    num_frames: int | None = None


@dataclass(frozen=True, kw_only=True)
class RoomSceneSettings:
    """Scene settings for simulate_human_room."""

    room_preset: str = "tutorial_bedroom"
    scene_xml: str | Path | bytes | bytearray | BytesIO | None = None
    scene_xml_filename: str | None = None


@dataclass(frozen=True, kw_only=True)
class HumanRayTracingSettings:
    """Solver rt settings for simulate_human_room."""

    rt_samples_per_source: int = 100000
    rt_max_paths_per_source: int = 5000
    rt_max_depth: int = 3
    rt_human_bounces: str = "single"
    rt_human_diffuse_coefficient: float = 0.35
    rt_refresh_frames: int = 1
    rt_visibility_refresh_frames: int = 1
    rt_visibility_fade_frames: int = 0


@dataclass(frozen=True, kw_only=True)
class HumanPOSettings:
    """Solver po settings for simulate_human_room."""

    po_refresh_frames: int = 1
    po_refresh_on_phase_error_rad: float = DEFAULT_PO_REFRESH_ON_PHASE_ERROR_RAD
    po_refresh_on_area_fraction: float = POConfig().po_refresh_on_area_fraction
    po_refresh_on_normal_deg: float = POConfig().po_refresh_on_normal_deg
    po_visibility_refresh_frames: int = 1
    po_visibility_fade_frames: int = 0
    po_visibility_samples_per_face: int = 1
    po_adaptive_visibility_sampling: bool = POConfig().po_adaptive_visibility_sampling
    po_adaptive_visibility_edge_margin: float = POConfig().po_adaptive_visibility_edge_margin
    po_integration_mode: str = "parent_face_adaptive_analytic"
    po_quadrature_phase_span_scale_rad: float | None = None
    po_curvature_error_rad: float = 0.005
    po_quadrature_max_refinement_depth: int = 3
    po_quadrature_max_subfaces_per_parent: int = 64


@dataclass(frozen=True, kw_only=True)
class HumanHybridSettings:
    """Solver hybrid settings for simulate_human_room."""

    hybrid_coupling: bool = True
    hybrid_reflector_cutoff_db: float = -20.0
    hybrid_max_reflectors: int = 4
    hybrid_calibration: str = "rt_first_pose"


@dataclass(frozen=True, kw_only=True)
class ExperimentExecutionSettings:
    """Solver execution settings for simulate_human_room."""

    compute_backend: str = "auto"
    cancel_check: Callable[[], bool] | None = None
    progress_callback: Callable[[dict], None] | None = None


@dataclass(frozen=True, kw_only=True)
class HumanRoomSolverSettings:
    """Solver settings for simulate_human_room."""

    solver: str = "hybrid_po"
    array: ArraySettings = field(default_factory=ArraySettings)
    rt: HumanRayTracingSettings = field(default_factory=HumanRayTracingSettings)
    po: HumanPOSettings = field(default_factory=HumanPOSettings)
    hybrid: HumanHybridSettings = field(default_factory=HumanHybridSettings)
    execution: ExperimentExecutionSettings = field(default_factory=ExperimentExecutionSettings)


@dataclass(frozen=True, kw_only=True)
class HumanRoomSettings:
    """Grouped settings accepted by :func:`simulate_human_room`."""

    motion: HumanMotionSettings
    scene: RoomSceneSettings = field(default_factory=RoomSceneSettings)
    radar: RadarSettings = field(default_factory=RadarSettings)
    solver: HumanRoomSolverSettings = field(default_factory=HumanRoomSolverSettings)

    def as_keywords(self) -> dict:
        """Return the equivalent legacy keyword arguments without copying arrays."""
        return {
            "human_motion": self.motion.human_motion,
            "human_motion_filename": self.motion.human_motion_filename,
            "smpl_model_dir": self.motion.smpl_model_dir,
            "room_preset": self.scene.room_preset,
            "scene_xml": self.scene.scene_xml,
            "scene_xml_filename": self.scene.scene_xml_filename,
            "human_position_m": self.motion.human_position_m,
            "human_yaw_deg": self.motion.human_yaw_deg,
            "motion_source_up_axis": self.motion.motion_source_up_axis,
            "motion_source_forward_axis": self.motion.motion_source_forward_axis,
            "motion_source_handedness": self.motion.motion_source_handedness,
            "motion_source_length_unit": self.motion.motion_source_length_unit,
            "radar_orientation_deg": self.radar.orientation_deg,
            "board_model": self.radar.board_model,
            "carrier_frequency_hz": self.radar.carrier_frequency_hz,
            "slope_hz_per_s": self.radar.slope_hz_per_s,
            "chirp_duration_s": self.radar.chirp_duration_s,
            "chirp_repetition_time_s": self.radar.chirp_repetition_time_s,
            "sampling_frequency_hz": self.radar.sampling_frequency_hz,
            "adc_mode": self.radar.adc_mode,
            "if_bandwidth_hz": self.radar.if_bandwidth_hz,
            "num_adc_samples": self.radar.num_adc_samples,
            "num_chirps_per_frame": self.radar.num_chirps_per_frame,
            "frame_period_s": self.radar.frame_period_s,
            "tx_schedule": self.radar.tx_schedule,
            "selected_tx_indices": self.radar.selected_tx_indices,
            "selected_rx_indices": self.radar.selected_rx_indices,
            "antenna_pattern_mode": self.radar.antenna_pattern_mode,
            "cosine_3db_beamwidth_deg": self.radar.cosine_3db_beamwidth_deg,
            "solver": self.solver.solver,
            "array_model": self.solver.array.array_model,
            "array_tx_groups": self.solver.array.array_tx_groups,
            "array_rx_groups": self.solver.array.array_rx_groups,
            "motion_start_s": self.motion.motion_start_s,
            "motion_end_s": self.motion.motion_end_s,
            "num_frames": self.motion.num_frames,
            "rt_samples_per_source": self.solver.rt.rt_samples_per_source,
            "rt_max_paths_per_source": self.solver.rt.rt_max_paths_per_source,
            "rt_max_depth": self.solver.rt.rt_max_depth,
            "rt_human_bounces": self.solver.rt.rt_human_bounces,
            "rt_human_diffuse_coefficient": self.solver.rt.rt_human_diffuse_coefficient,
            "rt_refresh_frames": self.solver.rt.rt_refresh_frames,
            "rt_visibility_refresh_frames": self.solver.rt.rt_visibility_refresh_frames,
            "rt_visibility_fade_frames": self.solver.rt.rt_visibility_fade_frames,
            "po_refresh_frames": self.solver.po.po_refresh_frames,
            "po_refresh_on_phase_error_rad": self.solver.po.po_refresh_on_phase_error_rad,
            "po_refresh_on_area_fraction": self.solver.po.po_refresh_on_area_fraction,
            "po_refresh_on_normal_deg": self.solver.po.po_refresh_on_normal_deg,
            "po_visibility_refresh_frames": self.solver.po.po_visibility_refresh_frames,
            "po_visibility_fade_frames": self.solver.po.po_visibility_fade_frames,
            "po_visibility_samples_per_face": self.solver.po.po_visibility_samples_per_face,
            "po_adaptive_visibility_sampling": self.solver.po.po_adaptive_visibility_sampling,
            "po_adaptive_visibility_edge_margin": self.solver.po.po_adaptive_visibility_edge_margin,
            "po_integration_mode": self.solver.po.po_integration_mode,
            "po_quadrature_phase_span_scale_rad": self.solver.po.po_quadrature_phase_span_scale_rad,
            "po_curvature_error_rad": self.solver.po.po_curvature_error_rad,
            "po_quadrature_max_refinement_depth": self.solver.po.po_quadrature_max_refinement_depth,
            "po_quadrature_max_subfaces_per_parent": self.solver.po.po_quadrature_max_subfaces_per_parent,
            "hybrid_coupling": self.solver.hybrid.hybrid_coupling,
            "hybrid_reflector_cutoff_db": self.solver.hybrid.hybrid_reflector_cutoff_db,
            "hybrid_max_reflectors": self.solver.hybrid.hybrid_max_reflectors,
            "hybrid_calibration": self.solver.hybrid.hybrid_calibration,
            "compute_backend": self.solver.execution.compute_backend,
            "cancel_check": self.solver.execution.cancel_check,
            "progress_callback": self.solver.execution.progress_callback,
        }


@dataclass(frozen=True, kw_only=True)
class StaticTargetGeometrySettings:
    """Target settings for simulate_static_target."""

    target_type: str = "plate"
    range_m: float = 2.0
    target_y_m: float = 0.0
    target_z_m: float = 0.0
    width_m: float = 0.3
    height_m: float = 0.3
    corner_edge_m: float = 0.2
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    material_name: str = "pec"
    human_mesh: str | Path | bytes | None = None
    human_mesh_filename: str | None = None
    fidelity: str = "preview"


@dataclass(frozen=True, kw_only=True)
class StaticRayTracingSettings:
    """Solver rt settings for simulate_static_target."""

    rt_human_diffuse_coefficient: float = 0.35
    rt_samples_per_source: int = 100000
    rt_max_paths_per_source: int = 5000
    rt_max_depth: int = 3


@dataclass(frozen=True, kw_only=True)
class StaticPOSettings:
    """Solver po settings for simulate_static_target."""

    po_integration_mode: str = "parent_face_quadrature"
    po_quadrature_phase_span_scale_rad: float = 1.0
    po_curvature_error_rad: float = 0.005
    po_quadrature_max_refinement_depth: int = 2
    po_quadrature_max_subfaces_per_parent: int = 16


@dataclass(frozen=True, kw_only=True)
class StaticExecutionSettings:
    """Solver execution settings for simulate_static_target."""

    compute_backend: str = "numpy"
    cancel_check: Callable[[], bool] | None = None


@dataclass(frozen=True, kw_only=True)
class StaticTargetSolverSettings:
    """Solver controls for a static-target experiment."""

    array: ArraySettings = field(default_factory=ArraySettings)
    rt: StaticRayTracingSettings = field(default_factory=StaticRayTracingSettings)
    po: StaticPOSettings = field(default_factory=StaticPOSettings)
    execution: StaticExecutionSettings = field(default_factory=StaticExecutionSettings)


@dataclass(frozen=True, kw_only=True)
class StaticTargetSettings:
    """Grouped settings accepted by :func:`simulate_static_target`."""

    target: StaticTargetGeometrySettings = field(default_factory=StaticTargetGeometrySettings)
    radar: RadarSettings = field(
        default_factory=lambda: RadarSettings(carrier_frequency_hz=None, num_chirps_per_frame=None)
    )
    solver: StaticTargetSolverSettings = field(default_factory=StaticTargetSolverSettings)

    def as_keywords(self) -> dict:
        """Return the equivalent legacy keyword arguments without copying arrays."""
        return {
            "target_type": self.target.target_type,
            "range_m": self.target.range_m,
            "target_y_m": self.target.target_y_m,
            "target_z_m": self.target.target_z_m,
            "width_m": self.target.width_m,
            "height_m": self.target.height_m,
            "corner_edge_m": self.target.corner_edge_m,
            "yaw_deg": self.target.yaw_deg,
            "pitch_deg": self.target.pitch_deg,
            "roll_deg": self.target.roll_deg,
            "material_name": self.target.material_name,
            "human_mesh": self.target.human_mesh,
            "human_mesh_filename": self.target.human_mesh_filename,
            "rt_human_diffuse_coefficient": self.solver.rt.rt_human_diffuse_coefficient,
            "fidelity": self.target.fidelity,
            "board_model": self.radar.board_model,
            "carrier_frequency_hz": self.radar.carrier_frequency_hz,
            "slope_hz_per_s": self.radar.slope_hz_per_s,
            "chirp_duration_s": self.radar.chirp_duration_s,
            "chirp_repetition_time_s": self.radar.chirp_repetition_time_s,
            "sampling_frequency_hz": self.radar.sampling_frequency_hz,
            "adc_mode": self.radar.adc_mode,
            "if_bandwidth_hz": self.radar.if_bandwidth_hz,
            "num_adc_samples": self.radar.num_adc_samples,
            "num_chirps_per_frame": self.radar.num_chirps_per_frame,
            "frame_period_s": self.radar.frame_period_s,
            "tx_schedule": self.radar.tx_schedule,
            "selected_tx_indices": self.radar.selected_tx_indices,
            "selected_rx_indices": self.radar.selected_rx_indices,
            "radar_yaw_deg": self.radar.orientation_deg[0],
            "radar_pitch_deg": self.radar.orientation_deg[1],
            "radar_roll_deg": self.radar.orientation_deg[2],
            "antenna_pattern_mode": self.radar.antenna_pattern_mode,
            "cosine_3db_beamwidth_deg": self.radar.cosine_3db_beamwidth_deg,
            "po_integration_mode": self.solver.po.po_integration_mode,
            "po_quadrature_phase_span_scale_rad": self.solver.po.po_quadrature_phase_span_scale_rad,
            "po_curvature_error_rad": self.solver.po.po_curvature_error_rad,
            "po_quadrature_max_refinement_depth": self.solver.po.po_quadrature_max_refinement_depth,
            "po_quadrature_max_subfaces_per_parent": self.solver.po.po_quadrature_max_subfaces_per_parent,
            "rt_samples_per_source": self.solver.rt.rt_samples_per_source,
            "rt_max_paths_per_source": self.solver.rt.rt_max_paths_per_source,
            "rt_max_depth": self.solver.rt.rt_max_depth,
            "array_model": self.solver.array.array_model,
            "array_tx_groups": self.solver.array.array_tx_groups,
            "array_rx_groups": self.solver.array.array_rx_groups,
            "compute_backend": self.solver.execution.compute_backend,
            "cancel_check": self.solver.execution.cancel_check,
        }
