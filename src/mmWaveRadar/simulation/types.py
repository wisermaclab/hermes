# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Public simulation result and configuration types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from .calibration import POCalibrationConfig
from .physical_optics import ComputeBackend, ComputePrecision, POIntegrationMode


CouplingMode = Literal["one_target_bounce", "one_human_bounce", "unrestricted"]

MobilityMode = Literal[
    "rt_retrace",
    "rt_coherent_bank",
    "human_only_po",
    "hybrid_static_env_po",
]

SENSING_METADATA_SCHEMA_VERSION = 1


def _validate_integer(name: str, value, *, minimum: int) -> int:
    """Validates an integer configuration field without silently truncating."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be an integer")
    if int(value) < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return int(value)


def _validate_finite(name: str, value, *, minimum: float, inclusive: bool) -> float:
    """Validates a finite scalar configuration field and its lower bound."""

    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar value") from exc
    invalid_bound = scalar < minimum if inclusive else scalar <= minimum
    if not np.isfinite(scalar) or invalid_bound:
        qualifier = "non-negative" if inclusive and minimum == 0.0 else "positive"
        raise ValueError(f"{name} must be a finite {qualifier} value")
    return scalar


def _validate_boolean(
    name: str,
    value,
    *,
    optional: bool = False,
) -> bool | None:
    """Validates a boolean option without applying truth-value coercion."""

    if value is None and optional:
        return None
    if not isinstance(value, (bool, np.bool_)):
        suffix = " or None" if optional else ""
        raise ValueError(f"{name} must be a boolean{suffix}")
    return bool(value)


@dataclass(frozen=True)
class RTCoherentTransitionConfig:
    """Configuration for coherent RT path transitions across retraces."""

    enabled: bool = True
    match_delay_tolerance_fraction: float = 0.5
    match_separation_fraction: float = 0.25
    crossfade: bool | None = None

    def __post_init__(self):
        enabled = _validate_boolean("enabled", self.enabled)
        delay_tolerance = _validate_finite(
            "match_delay_tolerance_fraction",
            self.match_delay_tolerance_fraction,
            minimum=0.0,
            inclusive=True,
        )
        separation = _validate_finite(
            "match_separation_fraction",
            self.match_separation_fraction,
            minimum=0.0,
            inclusive=True,
        )
        crossfade = _validate_boolean(
            "crossfade",
            self.crossfade,
            optional=True,
        )
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(
            self,
            "match_delay_tolerance_fraction",
            delay_tolerance,
        )
        object.__setattr__(self, "match_separation_fraction", separation)
        object.__setattr__(self, "crossfade", crossfade)


@dataclass(frozen=True)
class HumanPOMobilityConfig:
    """Configuration for the Phase 1 human-only PO mobility model."""

    visibility_samples_per_face: int = 4
    visibility_fade_chirps: int = 8
    visibility_use_phase_center: bool = True
    adaptive_visibility_sampling: bool = False
    adaptive_visibility_edge_margin: float = 0.15
    incremental_update: bool = False
    incremental_visibility_refresh_chirps: int = 8
    incremental_full_refresh_chirps: int | None = 64
    incremental_normal_threshold_deg: float = 5.0
    incremental_centroid_displacement_threshold_m: float | None = None
    incremental_area_relative_threshold: float | None = None
    compute_backend: ComputeBackend | None = None
    compute_precision: ComputePrecision | None = None
    po_integration_mode: POIntegrationMode = "face_centroid"
    po_quadrature_phase_span_scale_rad: float = 1.0
    po_quadrature_max_refinement_depth: int = 16
    po_quadrature_max_subfaces_per_parent: int = 0
    progress_callback: Callable[[dict], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    precomputed_raw_fractional_visibility: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    precomputed_visible_sample_counts: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    calibration: POCalibrationConfig = field(default_factory=POCalibrationConfig)

    def __post_init__(self):
        visibility_samples = _validate_integer(
            "visibility_samples_per_face",
            self.visibility_samples_per_face,
            minimum=1,
        )
        visibility_fade = _validate_integer(
            "visibility_fade_chirps",
            self.visibility_fade_chirps,
            minimum=1,
        )
        visibility_refresh = _validate_integer(
            "incremental_visibility_refresh_chirps",
            self.incremental_visibility_refresh_chirps,
            minimum=1,
        )
        full_refresh = self.incremental_full_refresh_chirps
        if self.incremental_full_refresh_chirps is not None:
            full_refresh = _validate_integer(
                "incremental_full_refresh_chirps",
                self.incremental_full_refresh_chirps,
                minimum=0,
            )
        edge_margin = _validate_finite(
            "adaptive_visibility_edge_margin",
            self.adaptive_visibility_edge_margin,
            minimum=0.0,
            inclusive=True,
        )
        if edge_margin > 1.0:
            raise ValueError("adaptive_visibility_edge_margin must not exceed 1")
        normal_threshold = _validate_finite(
            "incremental_normal_threshold_deg",
            self.incremental_normal_threshold_deg,
            minimum=0.0,
            inclusive=True,
        )
        optional_thresholds: dict[str, float | None] = {}
        for name, value in (
            (
                "incremental_centroid_displacement_threshold_m",
                self.incremental_centroid_displacement_threshold_m,
            ),
            (
                "incremental_area_relative_threshold",
                self.incremental_area_relative_threshold,
            ),
        ):
            if value is not None:
                value = _validate_finite(
                    name,
                    value,
                    minimum=0.0,
                    inclusive=True,
                )
            optional_thresholds[name] = value
        if self.compute_backend not in (None, "auto", "numpy", "torch"):
            raise ValueError(
                "compute_backend must be 'auto', 'numpy', 'torch', or None"
            )
        if self.compute_precision not in (None, "float64", "float32"):
            raise ValueError(
                "compute_precision must be 'float64', 'float32', or None"
            )
        if self.po_integration_mode not in (
            "face_centroid",
            "parent_face_quadrature",
            "parent_face_far_field_analytic",
        ):
            raise ValueError(
                "po_integration_mode must be 'face_centroid', "
                "'parent_face_quadrature', or "
                "'parent_face_far_field_analytic'"
            )
        quadrature_scale = _validate_finite(
            "po_quadrature_phase_span_scale_rad",
            self.po_quadrature_phase_span_scale_rad,
            minimum=0.0,
            inclusive=False,
        )
        refinement_depth = _validate_integer(
            "po_quadrature_max_refinement_depth",
            self.po_quadrature_max_refinement_depth,
            minimum=0,
        )
        max_subfaces = _validate_integer(
            "po_quadrature_max_subfaces_per_parent",
            self.po_quadrature_max_subfaces_per_parent,
            minimum=0,
        )
        if self.progress_callback is not None and not callable(self.progress_callback):
            raise ValueError("progress_callback must be callable or None")
        if not isinstance(self.calibration, POCalibrationConfig):
            raise ValueError("calibration must be a POCalibrationConfig instance")

        raw_visibility = self.precomputed_raw_fractional_visibility
        visible_counts = self.precomputed_visible_sample_counts
        if visible_counts is not None and raw_visibility is None:
            raise ValueError(
                "precomputed_visible_sample_counts requires "
                "precomputed_raw_fractional_visibility"
            )
        if raw_visibility is not None:
            raw = np.asarray(raw_visibility)
            if (
                raw.ndim != 1
                or not np.issubdtype(raw.dtype, np.number)
                or np.issubdtype(raw.dtype, np.complexfloating)
            ):
                raise ValueError(
                    "precomputed_raw_fractional_visibility must be a numeric 1-D array"
                )
            if not np.all(np.isfinite(raw)) or np.any((raw < 0.0) | (raw > 1.0)):
                raise ValueError(
                    "precomputed_raw_fractional_visibility must contain values in [0, 1]"
                )
            if visible_counts is not None:
                counts = np.asarray(visible_counts)
                if (
                    counts.shape != raw.shape
                    or not np.issubdtype(counts.dtype, np.integer)
                    or np.any(counts < 0)
                    or np.any(counts > int(self.visibility_samples_per_face))
                ):
                    raise ValueError(
                        "precomputed_visible_sample_counts must be integer counts "
                        "matching visibility and visibility_samples_per_face"
                    )

        object.__setattr__(self, "visibility_samples_per_face", visibility_samples)
        object.__setattr__(self, "visibility_fade_chirps", visibility_fade)
        object.__setattr__(
            self,
            "visibility_use_phase_center",
            _validate_boolean(
                "visibility_use_phase_center",
                self.visibility_use_phase_center,
            ),
        )
        object.__setattr__(
            self,
            "adaptive_visibility_sampling",
            _validate_boolean(
                "adaptive_visibility_sampling",
                self.adaptive_visibility_sampling,
            ),
        )
        object.__setattr__(
            self,
            "incremental_update",
            _validate_boolean("incremental_update", self.incremental_update),
        )
        object.__setattr__(self, "adaptive_visibility_edge_margin", edge_margin)
        object.__setattr__(
            self,
            "incremental_visibility_refresh_chirps",
            visibility_refresh,
        )
        object.__setattr__(
            self,
            "incremental_full_refresh_chirps",
            full_refresh,
        )
        object.__setattr__(
            self,
            "incremental_normal_threshold_deg",
            normal_threshold,
        )
        for name, value in optional_thresholds.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "po_quadrature_phase_span_scale_rad",
            quadrature_scale,
        )
        object.__setattr__(
            self,
            "po_quadrature_max_refinement_depth",
            refinement_depth,
        )
        object.__setattr__(
            self,
            "po_quadrature_max_subfaces_per_parent",
            max_subfaces,
        )


@dataclass(frozen=True)
class HybridStaticEnvPOConfig:
    """Configuration for static-environment RT plus dynamic human PO."""

    blocking_enabled: bool = True
    blocking_fade_chirps: int = 8
    blocking_aabb_culling: bool = True
    blocking_aabb_margin_m: float = 0.02
    blocking_ray_epsilon_m: float = 1e-5
    blocking_ray_chunk_size: int = 65536
    coupling_enabled: bool = False
    coupling_max_reflectors: int = 16
    coupling_min_reflector_area_m2: float = 1e-4
    coupling_reflection_scale: float = 1.0
    coupling_incremental_update: bool | None = None
    coupling_visibility_refresh_chirps: int | None = None
    human_po_config: HumanPOMobilityConfig | None = None

    def __post_init__(self):
        blocking_fade = _validate_integer(
            "blocking_fade_chirps",
            self.blocking_fade_chirps,
            minimum=1,
        )
        aabb_margin = _validate_finite(
            "blocking_aabb_margin_m",
            self.blocking_aabb_margin_m,
            minimum=0.0,
            inclusive=True,
        )
        ray_epsilon = _validate_finite(
            "blocking_ray_epsilon_m",
            self.blocking_ray_epsilon_m,
            minimum=0.0,
            inclusive=False,
        )
        ray_chunk_size = _validate_integer(
            "blocking_ray_chunk_size", self.blocking_ray_chunk_size, minimum=1
        )
        max_reflectors = _validate_integer(
            "coupling_max_reflectors", self.coupling_max_reflectors, minimum=0
        )
        min_reflector_area = _validate_finite(
            "coupling_min_reflector_area_m2",
            self.coupling_min_reflector_area_m2,
            minimum=0.0,
            inclusive=True,
        )
        reflection_scale = _validate_finite(
            "coupling_reflection_scale",
            self.coupling_reflection_scale,
            minimum=0.0,
            inclusive=True,
        )
        if (
            self.coupling_visibility_refresh_chirps is not None
            and not (
                isinstance(
                    self.coupling_visibility_refresh_chirps,
                    (int, np.integer),
                )
                and not isinstance(
                    self.coupling_visibility_refresh_chirps,
                    (bool, np.bool_),
                )
                and self.coupling_visibility_refresh_chirps > 0
            )
        ):
            raise ValueError(
                "coupling_visibility_refresh_chirps must be a positive integer"
            )
        if self.human_po_config is not None and not isinstance(
            self.human_po_config,
            HumanPOMobilityConfig,
        ):
            raise ValueError(
                "human_po_config must be a HumanPOMobilityConfig or None"
            )
        object.__setattr__(
            self,
            "blocking_enabled",
            _validate_boolean("blocking_enabled", self.blocking_enabled),
        )
        object.__setattr__(
            self,
            "blocking_aabb_culling",
            _validate_boolean(
                "blocking_aabb_culling",
                self.blocking_aabb_culling,
            ),
        )
        object.__setattr__(
            self,
            "coupling_enabled",
            _validate_boolean("coupling_enabled", self.coupling_enabled),
        )
        object.__setattr__(
            self,
            "coupling_incremental_update",
            _validate_boolean(
                "coupling_incremental_update",
                self.coupling_incremental_update,
                optional=True,
            ),
        )
        object.__setattr__(self, "blocking_fade_chirps", blocking_fade)
        object.__setattr__(self, "blocking_aabb_margin_m", aabb_margin)
        object.__setattr__(self, "blocking_ray_epsilon_m", ray_epsilon)
        object.__setattr__(self, "blocking_ray_chunk_size", ray_chunk_size)
        object.__setattr__(self, "coupling_max_reflectors", max_reflectors)
        object.__setattr__(
            self,
            "coupling_min_reflector_area_m2",
            min_reflector_area,
        )
        object.__setattr__(self, "coupling_reflection_scale", reflection_scale)
        if self.coupling_visibility_refresh_chirps is not None:
            object.__setattr__(
                self,
                "coupling_visibility_refresh_chirps",
                int(self.coupling_visibility_refresh_chirps),
            )
        if self.human_po_config is None:
            object.__setattr__(self, "human_po_config", HumanPOMobilityConfig())


@dataclass(frozen=True)
class SensingMetadata:
    """Metadata collected during a sensing simulation."""

    events: list[dict]
    path_counts: np.ndarray
    max_displacements: np.ndarray
    virtual_channel_order: str
    metadata_schema_version: int = SENSING_METADATA_SCHEMA_VERSION
    mode: str = "rt_retrace"
    visible_face_counts: np.ndarray | None = None
    effective_visible_area_fraction: np.ndarray | None = None
    runtime_per_chirp_s: np.ndarray | None = None
    max_unwrapped_phase_jump: np.ndarray | None = None
    runtime_profile_s: dict[str, float] | None = None
    runtime_profile_counts: dict[str, int] | None = None
    incremental_recomputed_face_counts: np.ndarray | None = None
    incremental_phase_updated_face_counts: np.ndarray | None = None
    incremental_full_refresh: np.ndarray | None = None
    incremental_visibility_refresh: np.ndarray | None = None
    static_path_counts: np.ndarray | None = None
    blocked_static_path_counts: np.ndarray | None = None
    mean_static_path_visibility: np.ndarray | None = None
    human_env_path_counts: np.ndarray | None = None
    env_human_path_counts: np.ndarray | None = None
    human_touch_path_counts: np.ndarray | None = None
    one_human_touch_path_counts: np.ndarray | None = None
    single_human_only_path_counts: np.ndarray | None = None
    human_env_coupled_path_counts: np.ndarray | None = None
    human_multi_touch_path_counts: np.ndarray | None = None
    path_depth_histogram: np.ndarray | None = None
    total_path_power: np.ndarray | None = None
    human_touch_path_power: np.ndarray | None = None
    one_human_touch_path_power: np.ndarray | None = None
    single_human_only_path_power: np.ndarray | None = None
    human_env_coupled_path_power: np.ndarray | None = None
    human_multi_touch_path_power: np.ndarray | None = None
    human_env_min_path_lengths_m: np.ndarray | None = None
    human_env_mean_path_lengths_m: np.ndarray | None = None
    human_env_max_path_lengths_m: np.ndarray | None = None
    env_human_min_path_lengths_m: np.ndarray | None = None
    env_human_mean_path_lengths_m: np.ndarray | None = None
    env_human_max_path_lengths_m: np.ndarray | None = None
    po_calibration: dict | None = None
    rt_transition_persistent_path_counts: np.ndarray | None = None
    rt_transition_birth_path_counts: np.ndarray | None = None
    rt_transition_death_path_counts: np.ndarray | None = None
    rt_transition_alpha: np.ndarray | None = None
    rt_transition_old_weights: np.ndarray | None = None
    rt_transition_new_weights: np.ndarray | None = None
    virtual_channel_times_s: np.ndarray | None = None


@dataclass(frozen=True)
class RadarCube:
    r"""FMCW ADC cube and metadata."""

    adc: np.ndarray
    times: np.ndarray
    metadata: SensingMetadata
    components: dict[str, np.ndarray] | None = None


__all__ = [
    "CouplingMode",
    "HybridStaticEnvPOConfig",
    "HumanPOMobilityConfig",
    "MobilityMode",
    "POIntegrationMode",
    "RTCoherentTransitionConfig",
    "RadarCube",
    "SENSING_METADATA_SCHEMA_VERSION",
    "SensingMetadata",
]
