# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Public simulation result and configuration types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..values import boolean
from ..naming import (
    HumanBounces,
    SOLVERS,
    Solver,
    validate_frames,
    validate_hybrid_calibration,
)
from .calibration import POCalibrationConfig
from .compute import (
    ComputeBackend,
    ComputePrecision,
    numpy_dtypes_for_precision,
    validate_compute_backend,
)
from .physical_optics import (
    POIntegrationMode,
    default_phase_span_scale_rad,
)

SENSING_METADATA_SCHEMA_VERSION = 2
DEFAULT_PO_REFRESH_ON_PHASE_ERROR_RAD = 0.1


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


_validate_boolean = boolean


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


def _calibration_config(name: str, value) -> POCalibrationConfig:
    """Accepts a calibration name or a full calibration configuration."""

    if isinstance(value, POCalibrationConfig):
        return value
    return POCalibrationConfig(mode=validate_hybrid_calibration(value, name=name))


@dataclass(frozen=True)
class POConfig:
    """Physical-optics settings for the PO solver and the Hybrid PO body.

    Exact integration runs at a frame start every ``po_refresh_frames``
    frames; 0 integrates every chirp. Between exact solves each facet keeps
    its integrated amplitude while its per-channel phase, delay, and
    spreading follow the motion. Facets that drift past the
    ``po_refresh_on_phase_error_rad`` are integrated again. This bounds the
    change since the last exact solve in vertex two-way path lengths relative
    to the facet centre, multiplied by the wavenumber. Area changes above 30%
    and normal changes above 20 degrees (configurable with
    ``po_refresh_on_area_fraction`` and ``po_refresh_on_normal_deg``)
    also force an exact solve, measured
    from the last exact geometry, to catch amplitude and polarization drift.
    Visibility
    (self-shadowing) is refreshed every ``po_visibility_refresh_frames``
    frames (0 = every chirp), and visibility changes fade over
    ``po_visibility_fade_frames`` frames (0 = a step at the frame start).
    Adaptive visibility sampling is enabled by default: additional face samples
    are checked near centroid visibility boundaries and grazing incidence.
    Disable it to check every configured sample on every face; adaptive checks
    can miss partial occlusion within faces not flagged by those criteria.
    """

    po_refresh_frames: int = 1
    po_visibility_refresh_frames: int = 1
    po_visibility_fade_frames: int = 0
    po_visibility_samples_per_face: int = 4
    po_adaptive_visibility_sampling: bool = True
    po_adaptive_visibility_edge_margin: float = 0.15
    po_refresh_on_phase_error_rad: float = DEFAULT_PO_REFRESH_ON_PHASE_ERROR_RAD
    po_refresh_on_area_fraction: float = 0.30
    po_refresh_on_normal_deg: float = 20.0
    po_integration_mode: POIntegrationMode = "face_centroid"
    # None selects the integration mode's default; see
    # effective_po_phase_span_scale_rad.
    po_quadrature_phase_span_scale_rad: float | None = None
    po_curvature_error_rad: float = 0.005
    po_quadrature_max_refinement_depth: int = 16
    po_quadrature_max_subfaces_per_parent: int = 0
    # Optional RT-reference amplitude fit for the PO solver. Hybrid PO uses
    # HybridPOConfig.hybrid_calibration instead.
    po_calibration: POCalibrationConfig | str = "none"
    compute_backend: ComputeBackend | None = None
    compute_precision: ComputePrecision | None = None
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

    def __post_init__(self):
        for name in (
            "po_refresh_frames",
            "po_visibility_refresh_frames",
            "po_visibility_fade_frames",
        ):
            object.__setattr__(self, name, validate_frames(getattr(self, name), name=name))
        for name in ("po_refresh_on_area_fraction", "po_refresh_on_normal_deg"):
            value = _validate_finite(name, getattr(self, name), minimum=0.0, inclusive=True)
            if name == "po_refresh_on_normal_deg" and value > 180.0:
                raise ValueError("po_refresh_on_normal_deg must not exceed 180")
            object.__setattr__(self, name, value)
        visibility_samples = _validate_integer(
            "po_visibility_samples_per_face",
            self.po_visibility_samples_per_face,
            minimum=1,
        )
        edge_margin = _validate_finite(
            "po_adaptive_visibility_edge_margin",
            self.po_adaptive_visibility_edge_margin,
            minimum=0.0,
            inclusive=True,
        )
        if edge_margin > 1.0:
            raise ValueError("po_adaptive_visibility_edge_margin must not exceed 1")
        phase_error_threshold = _validate_finite(
            "po_refresh_on_phase_error_rad",
            self.po_refresh_on_phase_error_rad,
            minimum=0.0,
            inclusive=True,
        )
        if self.compute_backend is not None:
            validate_compute_backend(self.compute_backend, name="compute_backend")
        if self.compute_precision is not None:
            numpy_dtypes_for_precision(self.compute_precision)
        if self.po_integration_mode not in (
            "face_centroid",
            "parent_face_quadrature",
            "parent_face_adaptive_analytic",
            "parent_face_far_field_analytic",
        ):
            raise ValueError(
                "po_integration_mode must be 'face_centroid', "
                "'parent_face_quadrature', 'parent_face_adaptive_analytic', or "
                "'parent_face_far_field_analytic'"
            )
        quadrature_scale = (
            None
            if self.po_quadrature_phase_span_scale_rad is None
            else _validate_finite(
                "po_quadrature_phase_span_scale_rad",
                self.po_quadrature_phase_span_scale_rad,
                minimum=0.0,
                inclusive=False,
            )
        )
        refinement_depth = _validate_integer(
            "po_quadrature_max_refinement_depth",
            self.po_quadrature_max_refinement_depth,
            minimum=0,
        )
        curvature_error = _validate_finite(
            "po_curvature_error_rad",
            self.po_curvature_error_rad,
            minimum=0.0,
            inclusive=False,
        )
        max_subfaces = _validate_integer(
            "po_quadrature_max_subfaces_per_parent",
            self.po_quadrature_max_subfaces_per_parent,
            minimum=0,
        )
        if self.progress_callback is not None and not callable(self.progress_callback):
            raise ValueError("progress_callback must be callable or None")
        object.__setattr__(
            self,
            "po_calibration",
            _calibration_config("po_calibration", self.po_calibration),
        )

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
                    or np.any(counts > int(self.po_visibility_samples_per_face))
                ):
                    raise ValueError(
                        "precomputed_visible_sample_counts must be integer counts "
                        "matching visibility and po_visibility_samples_per_face"
                    )

        object.__setattr__(self, "po_visibility_samples_per_face", visibility_samples)
        object.__setattr__(
            self,
            "po_adaptive_visibility_sampling",
            _validate_boolean(
                "po_adaptive_visibility_sampling",
                self.po_adaptive_visibility_sampling,
            ),
        )
        object.__setattr__(self, "po_adaptive_visibility_edge_margin", edge_margin)
        object.__setattr__(self, "po_refresh_on_phase_error_rad", phase_error_threshold)
        object.__setattr__(
            self,
            "po_quadrature_phase_span_scale_rad",
            quadrature_scale,
        )
        object.__setattr__(self, "po_curvature_error_rad", curvature_error)
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

    @property
    def effective_po_phase_span_scale_rad(self) -> float:
        """Returns the explicit phase span or the integration mode's default."""

        if self.po_quadrature_phase_span_scale_rad is not None:
            return float(self.po_quadrature_phase_span_scale_rad)
        return default_phase_span_scale_rad(self.po_integration_mode)

    @property
    def incremental(self) -> bool:
        """Whether facets are phase-updated between exact integrations."""

        return self.po_refresh_frames > 0


@dataclass(frozen=True)
class HybridPOConfig:
    """Hybrid PO settings: RT static room, PO body, blocking, and coupling.

    Blocking of room paths by the body is always on and follows the
    simulator's ``rt_visibility_*`` cadence. The body and both coupling
    directions use ``po_config``. Reflector selection is fixed for the run.
    """

    hybrid_coupling: bool = True
    # Keep image powers within this level of the strongest, then take at most
    # hybrid_max_reflectors, with a hard supported maximum of four.
    hybrid_reflector_cutoff_db: float = -20.0
    hybrid_max_reflectors: int = 4
    hybrid_reflection_scale: float = 1.0
    hybrid_min_reflector_area_m2: float = 1e-4
    hybrid_calibration: POCalibrationConfig | str = "rt_first_pose"
    blocking_aabb_culling: bool = True
    blocking_aabb_margin_m: float = 0.02
    blocking_ray_epsilon_m: float = 1e-5
    blocking_ray_chunk_size: int = 65536
    po_config: POConfig | None = None

    def __post_init__(self):
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
        cutoff = self.hybrid_reflector_cutoff_db
        if isinstance(cutoff, (bool, np.bool_)) or cutoff is None:
            raise ValueError("hybrid_reflector_cutoff_db must be finite and <= 0")
        try:
            cutoff = float(cutoff)
        except (TypeError, ValueError) as exc:
            raise ValueError("hybrid_reflector_cutoff_db must be finite and <= 0") from exc
        if not np.isfinite(cutoff) or cutoff > 0.0:
            raise ValueError("hybrid_reflector_cutoff_db must be finite and <= 0")
        max_reflectors = _validate_integer(
            "hybrid_max_reflectors", self.hybrid_max_reflectors, minimum=1
        )
        if max_reflectors > 4:
            raise ValueError("hybrid_max_reflectors must be between 1 and 4")
        min_reflector_area = _validate_finite(
            "hybrid_min_reflector_area_m2",
            self.hybrid_min_reflector_area_m2,
            minimum=0.0,
            inclusive=True,
        )
        reflection_scale = _validate_finite(
            "hybrid_reflection_scale",
            self.hybrid_reflection_scale,
            minimum=0.0,
            inclusive=True,
        )
        if self.po_config is not None and not isinstance(self.po_config, POConfig):
            raise ValueError("po_config must be a POConfig or None")
        object.__setattr__(
            self,
            "hybrid_coupling",
            _validate_boolean("hybrid_coupling", self.hybrid_coupling),
        )
        object.__setattr__(
            self,
            "blocking_aabb_culling",
            _validate_boolean("blocking_aabb_culling", self.blocking_aabb_culling),
        )
        object.__setattr__(
            self,
            "hybrid_calibration",
            _calibration_config("hybrid_calibration", self.hybrid_calibration),
        )
        object.__setattr__(self, "blocking_aabb_margin_m", aabb_margin)
        object.__setattr__(self, "blocking_ray_epsilon_m", ray_epsilon)
        object.__setattr__(self, "blocking_ray_chunk_size", ray_chunk_size)
        object.__setattr__(self, "hybrid_reflector_cutoff_db", cutoff)
        object.__setattr__(self, "hybrid_max_reflectors", max_reflectors)
        object.__setattr__(self, "hybrid_min_reflector_area_m2", min_reflector_area)
        object.__setattr__(self, "hybrid_reflection_scale", reflection_scale)
        if self.po_config is None:
            object.__setattr__(self, "po_config", POConfig())


@dataclass(frozen=True)
class SensingMetadata:
    """Metadata collected during a sensing simulation."""

    events: list[dict]
    path_counts: np.ndarray
    max_displacements: np.ndarray
    virtual_channel_order: str
    metadata_schema_version: int = SENSING_METADATA_SCHEMA_VERSION
    solver: Solver = "rt"
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
    adc_if_filter: dict | None = None
    po_calibration: dict | None = None
    rt_transition_persistent_path_counts: np.ndarray | None = None
    rt_transition_birth_path_counts: np.ndarray | None = None
    rt_transition_death_path_counts: np.ndarray | None = None
    rt_transition_alpha: np.ndarray | None = None
    rt_transition_old_weights: np.ndarray | None = None
    rt_transition_new_weights: np.ndarray | None = None
    rt_visibility_refresh: np.ndarray | None = None
    rt_blocked_path_counts: np.ndarray | None = None
    virtual_channel_times_s: np.ndarray | None = None
    provenance: dict | None = None


@dataclass(frozen=True)
class RadarCube:
    r"""FMCW ADC cube and metadata."""

    adc: np.ndarray
    times: np.ndarray
    metadata: SensingMetadata
    components: dict[str, np.ndarray] | None = None


__all__ = [
    "HumanBounces",
    "HybridPOConfig",
    "POConfig",
    "POIntegrationMode",
    "RTCoherentTransitionConfig",
    "RadarCube",
    "SENSING_METADATA_SCHEMA_VERSION",
    "SOLVERS",
    "SensingMetadata",
    "Solver",
]
