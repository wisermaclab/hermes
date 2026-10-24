# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Typed settings for grouped experiment and solver entry points.

Arrays and scene/model objects are retained by reference; create a new settings
object with dataclasses.replace to change one option between runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .calibration import POCalibrationConfig
    from .compute import ComputeBackend, ComputePrecision
    from .types import POConfig, HybridPOConfig
    from .physical_optics import (
        FaceVisibilityState,
        HumanVisibilityScene,
        POIntegrationMode,
        VisibilityMode,
    )


@dataclass(frozen=True, kw_only=True)
class ArraySettings:
    """Solver array settings for run_human_room_experiment."""

    array_model: str = "phase_center"
    array_tx_groups: list[list[int]] | None = None
    array_rx_groups: list[list[int]] | None = None


@dataclass(frozen=True, kw_only=True)
class HybridFrameSettings:
    """Frames settings for run_calibrated_hybrid_po."""

    num_frames: int
    calibration_frame_range: tuple[int, int] = (0, 1)


@dataclass(frozen=True, kw_only=True)
class HybridRayTracingSettings:
    """Rt settings for run_calibrated_hybrid_po."""

    rt_samples_per_source: int = 80000
    rt_max_paths_per_source: int = 80000
    rt_max_depth: int = 2
    rt_human_bounces: str = "single"
    rt_visibility_refresh_frames: int = 1
    rt_visibility_fade_frames: int = 0
    rt_diffuse_reflection: bool = True
    rt_seed: int = 42


@dataclass(frozen=True, kw_only=True)
class HybridPhysicsSettings:
    """Physics settings for run_calibrated_hybrid_po."""

    po_visibility_samples_per_face: int | None = None
    hybrid_coupling: bool = True
    hybrid_reflector_cutoff_db: float = -20.0
    hybrid_max_reflectors: int = 4
    hybrid_reflection_scale: float = 1.0
    hybrid_calibration: POCalibrationConfig | str | None = None
    po_config: POConfig | None = None
    hybrid_config: HybridPOConfig | None = None


@dataclass(frozen=True, kw_only=True)
class HybridExecutionSettings:
    """Execution settings for run_calibrated_hybrid_po."""

    compute_backend: ComputeBackend = "auto"
    compute_precision: ComputePrecision = "float32"
    adc_compute_precision: ComputePrecision | None = None
    progress: bool = True
    progress_every_frames: int = 10
    print_reflector_summary: bool = True
    cancel_check: Callable[[], bool] | None = None
    progress_callback: Callable[[dict], None] | None = None


@dataclass(frozen=True, kw_only=True)
class HybridPipelineSettings:
    """Grouped settings accepted by :func:`run_hybrid_po_pipeline`."""

    frames: HybridFrameSettings
    array: ArraySettings = field(default_factory=ArraySettings)
    rt: HybridRayTracingSettings = field(default_factory=HybridRayTracingSettings)
    physics: HybridPhysicsSettings = field(default_factory=HybridPhysicsSettings)
    execution: HybridExecutionSettings = field(default_factory=HybridExecutionSettings)

    def as_keywords(self) -> dict:
        """Return the equivalent legacy keyword arguments without copying arrays."""
        return {
            "num_frames": self.frames.num_frames,
            "calibration_frame_range": self.frames.calibration_frame_range,
            "po_visibility_samples_per_face": self.physics.po_visibility_samples_per_face,
            "hybrid_coupling": self.physics.hybrid_coupling,
            "hybrid_reflector_cutoff_db": self.physics.hybrid_reflector_cutoff_db,
            "hybrid_max_reflectors": self.physics.hybrid_max_reflectors,
            "hybrid_reflection_scale": self.physics.hybrid_reflection_scale,
            "hybrid_calibration": self.physics.hybrid_calibration,
            "rt_samples_per_source": self.rt.rt_samples_per_source,
            "rt_max_paths_per_source": self.rt.rt_max_paths_per_source,
            "rt_max_depth": self.rt.rt_max_depth,
            "rt_human_bounces": self.rt.rt_human_bounces,
            "rt_visibility_refresh_frames": self.rt.rt_visibility_refresh_frames,
            "rt_visibility_fade_frames": self.rt.rt_visibility_fade_frames,
            "rt_diffuse_reflection": self.rt.rt_diffuse_reflection,
            "rt_seed": self.rt.rt_seed,
            "array_model": self.array.array_model,
            "array_tx_groups": self.array.array_tx_groups,
            "array_rx_groups": self.array.array_rx_groups,
            "compute_backend": self.execution.compute_backend,
            "compute_precision": self.execution.compute_precision,
            "adc_compute_precision": self.execution.adc_compute_precision,
            "progress": self.execution.progress,
            "progress_every_frames": self.execution.progress_every_frames,
            "print_reflector_summary": self.execution.print_reflector_summary,
            "po_config": self.physics.po_config,
            "hybrid_config": self.physics.hybrid_config,
            "cancel_check": self.execution.cancel_check,
            "progress_callback": self.execution.progress_callback,
        }


@dataclass(frozen=True, kw_only=True)
class POArraySettings:
    """Array settings for po_channel."""

    tx_positions: np.ndarray
    rx_positions: np.ndarray
    frequency_hz: float
    material: object
    virtual_channel_tx_indices: np.ndarray | None = None
    virtual_channel_rx_indices: np.ndarray | None = None
    virtual_channel_order: str = "tx_major"
    polarization_axis: np.ndarray | None = None
    max_frequency_offset_hz: float | None = None
    receiver_fmcw: object = None


@dataclass(frozen=True, kw_only=True)
class POVisibilitySettings:
    """Visibility settings for po_channel."""

    visibility: VisibilityMode = "fractional_shadow_fade"
    visibility_state: FaceVisibilityState | None = None
    visibility_scene: HumanVisibilityScene | None = None
    static_visibility_scene: object = None
    po_visibility_samples_per_face: int = 4
    visibility_fade_steps: int = 1
    visibility_fade_advance: bool = True
    po_adaptive_visibility_sampling: bool = True
    po_adaptive_visibility_edge_margin: float = 0.15
    precomputed_raw_fractional_visibility: np.ndarray | None = None
    precomputed_visible_sample_counts: np.ndarray | None = None


@dataclass(frozen=True, kw_only=True)
class POIntegrationSettings:
    """Integration settings for po_channel."""

    po_integration_mode: POIntegrationMode = "face_centroid"
    po_quadrature_phase_span_scale_rad: float | None = None
    po_curvature_error_rad: float = 0.005
    po_quadrature_max_refinement_depth: int = 16
    po_quadrature_max_subfaces_per_parent: int = 0


@dataclass(frozen=True, kw_only=True)
class POComputeSettings:
    """Compute settings for po_channel."""

    progress_callback: Callable[[dict], None] | None = None
    compute_backend: ComputeBackend = "numpy"
    compute_precision: ComputePrecision = "float64"


@dataclass(frozen=True, kw_only=True)
class POChannelSettings:
    """Grouped settings accepted by :func:`evaluate_po_channel`."""

    array: POArraySettings
    visibility: POVisibilitySettings = field(default_factory=POVisibilitySettings)
    integration: POIntegrationSettings = field(default_factory=POIntegrationSettings)
    compute: POComputeSettings = field(default_factory=POComputeSettings)

    def as_keywords(self) -> dict:
        """Return the equivalent legacy keyword arguments without copying arrays."""
        return {
            "tx_positions": self.array.tx_positions,
            "rx_positions": self.array.rx_positions,
            "frequency_hz": self.array.frequency_hz,
            "material": self.array.material,
            "virtual_channel_tx_indices": self.array.virtual_channel_tx_indices,
            "virtual_channel_rx_indices": self.array.virtual_channel_rx_indices,
            "virtual_channel_order": self.array.virtual_channel_order,
            "visibility": self.visibility.visibility,
            "visibility_state": self.visibility.visibility_state,
            "visibility_scene": self.visibility.visibility_scene,
            "static_visibility_scene": self.visibility.static_visibility_scene,
            "po_visibility_samples_per_face": self.visibility.po_visibility_samples_per_face,
            "visibility_fade_steps": self.visibility.visibility_fade_steps,
            "visibility_fade_advance": self.visibility.visibility_fade_advance,
            "po_adaptive_visibility_sampling": self.visibility.po_adaptive_visibility_sampling,
            "po_adaptive_visibility_edge_margin": self.visibility.po_adaptive_visibility_edge_margin,
            "precomputed_raw_fractional_visibility": self.visibility.precomputed_raw_fractional_visibility,
            "precomputed_visible_sample_counts": self.visibility.precomputed_visible_sample_counts,
            "po_integration_mode": self.integration.po_integration_mode,
            "po_quadrature_phase_span_scale_rad": self.integration.po_quadrature_phase_span_scale_rad,
            "po_curvature_error_rad": self.integration.po_curvature_error_rad,
            "po_quadrature_max_refinement_depth": self.integration.po_quadrature_max_refinement_depth,
            "po_quadrature_max_subfaces_per_parent": self.integration.po_quadrature_max_subfaces_per_parent,
            "progress_callback": self.compute.progress_callback,
            "compute_backend": self.compute.compute_backend,
            "compute_precision": self.compute.compute_precision,
            "polarization_axis": self.array.polarization_axis,
            "max_frequency_offset_hz": self.array.max_frequency_offset_hz,
            "receiver_fmcw": self.array.receiver_fmcw,
        }
