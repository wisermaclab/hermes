# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""mmWave radar sensing simulation driver."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, replace
from typing import Callable, Iterable, TYPE_CHECKING
import time
import warnings

import numpy as np

from ..arrays import as_numpy
from ..provenance import capture_provenance, simulation_input_hashes
from scipy.constants import c

if TYPE_CHECKING:
    from sionna.rt import PathSolver

from .path_products import coherent_channel
from .adc_synthesis import (
    _numpy_dtypes_for_precision,
    _synthesize_adc,
    _synthesize_adc_torch,
    numpy_dtypes_for_precision,
    synthesize_adc_from_paths,
)
from .compute import ComputeBackend, ComputePrecision, validate_compute_backend
from .physical_optics import (
    FaceVisibilityState,
    HumanVisibilityScene,
    POChannelResult,
    _aggregate_parent_face_far_field_analytic_contributions,
    _aggregate_parent_face_quadrature_contributions,
    _coerce_material,
    _fractional_face_visibility,
    _update_visibility_state,
    face_geometry,
    parent_face_adaptive_analytic_geometry,
    parent_face_far_field_analytic_geometry,
    parent_face_far_field_edge_rule,
    parent_face_quadrature_geometry,
    po_channel,
    project_phase_center_contributions_to_virtual_channels,
    vector_facet_face_contributions,
)
from .hybrid_passes import (
    _run_coupling_pass,
    _run_static_environment_pass,
    _select_reflectors,
    _with_shared_reflector_profile,
)
from .cadence import refresh_period_chirps
from .static_blocking import (
    PathVisibilityState,
    blocked_segments,
    update_path_visibility_state,
)
from ..naming import (
    ArrayModel,
    HumanBounces,
    Solver,
    reject_removed_arguments,
    validate_array_groups,
    validate_array_model,
    validate_frames,
    validate_rt_human_bounces,
    validate_solver,
)
from ..targets import MeshTarget
from ..radar import RadarSensor
from ..geometry import rotation_matrix
from ..radar.hardware import resolve_virtual_channel_pairs
from .runtime import configure_mitsuba_variant
from .po_wideband import PODelayResponse
from .po_incremental import (
    POFacetState,
    vertex_path_offsets,
    match_sorted_face_positions as _match_sorted_face_positions,
    phase_update_coefficients,
    update_po_facets,
)
from .rt_coherence import (
    CoherentRTPathBank,
    CoherentRTPathTransition,
    coherent_path_segments,
    extract_coherent_rt_path_bank,
    match_coherent_rt_path_banks,
    synthesize_coherent_rt_transition,
    update_coherent_rt_path_bank,
)
from .calibration import (
    POCalibrationConfig,
    POCalibrationReference,
    calibration_gain_array,
    rt_baseline_schedule_metadata,
    fit_first_pose_adc_calibration,
    fit_sequence_adc_calibration,
    po_path_power_from_coefficients,
    rt_calibration_enabled,
)
from .channel_gain import apply_path_hardware_gain
from .path_filters import (
    _drop_target_specular_bank,
    _one_human_bounce_mask,
    _one_target_bounce_mask,
    _one_target_bounce_mask_from_bank,
    _single_target_only_bounce_mask,
    _target_interaction_counts,
    _target_object_ids,
    _target_specular_path_mask,
    _target_specular_path_mask_from_arrays,
    drop_target_specular_bank,
    one_target_bounce_mask,
    one_target_bounce_mask_from_bank,
    single_target_only_bounce_mask,
    target_interaction_counts,
    target_object_ids,
    target_specular_path_mask,
    target_specular_path_mask_from_arrays,
)

# Retain the schema constant at its established import location.
# pylint: disable-next=unused-import
from .types import (
    HybridPOConfig,
    POConfig,
    RTCoherentTransitionConfig,
    RadarCube,
    SENSING_METADATA_SCHEMA_VERSION as SENSING_METADATA_SCHEMA_VERSION,
    SensingMetadata,
)
from .subarray import (
    group_pairs,
    max_group_offsets,
    merge_subarray_cubes,
    partition_event,
    run_group_pairs,
)
from .tdm import (
    collapse_tdm_cube,
    expanded_tdm_radar,
    tdm_expansion_required,
)


@dataclass(frozen=True)
class _POChannelInputs:
    """Fixed geometry and solver settings for one PO run."""

    radar: RadarSensor
    target: MeshTarget
    config: POConfig
    tx_positions: np.ndarray
    rx_positions: np.ndarray
    visibility_scene: HumanVisibilityScene | None
    static_visibility_scene: object
    compute_backend: ComputeBackend
    compute_precision: ComputePrecision

    @property
    def faces(self):
        return self.target.mesh_sequence.faces

    @property
    def max_frequency_offset_hz(self):
        fmcw = self.radar.fmcw
        return fmcw.slope * (fmcw.num_adc_samples - 1) / fmcw.sampling_frequency

    def geometry_arguments(self):
        return {
            "faces": self.faces,
            "tx_positions": self.tx_positions,
            "rx_positions": self.rx_positions,
            "virtual_channel_tx_indices": self.radar.hardware.virtual_tx_indices(),
            "virtual_channel_rx_indices": self.radar.hardware.virtual_rx_indices(),
            "virtual_channel_order": self.radar.hardware.virtual_channel_order,
        }

    def channel_arguments(self):
        config = self.config
        return {
            **self.geometry_arguments(),
            "frequency_hz": self.radar.fmcw.carrier_frequency,
            "material": self.target.material,
            "visibility_scene": self.visibility_scene,
            "static_visibility_scene": self.static_visibility_scene,
            "po_visibility_samples_per_face": config.po_visibility_samples_per_face,
            "po_adaptive_visibility_sampling": config.po_adaptive_visibility_sampling,
            "po_adaptive_visibility_edge_margin": config.po_adaptive_visibility_edge_margin,
            "po_integration_mode": config.po_integration_mode,
            "po_quadrature_phase_span_scale_rad": config.effective_po_phase_span_scale_rad,
            "po_curvature_error_rad": config.po_curvature_error_rad,
            "po_quadrature_max_refinement_depth": config.po_quadrature_max_refinement_depth,
            "po_quadrature_max_subfaces_per_parent": config.po_quadrature_max_subfaces_per_parent,
            "progress_callback": config.progress_callback,
            "compute_backend": self.compute_backend,
            "compute_precision": self.compute_precision,
            "polarization_axis": rotation_matrix(self.radar.orientation)[:, 2],
            "receiver_fmcw": self.radar.fmcw,
            "max_frequency_offset_hz": self.max_frequency_offset_hz,
        }


@dataclass
class _RTRetracePlan:
    """Anchors selected before tracing, shared by matching and chirp synthesis."""

    events: list[dict]
    flat_indices: list[int]
    anchor_vertices: list[list[np.ndarray]]


@dataclass
class _POIncrementalState:
    """Cached geometry, channel, and visibility state for incremental PO updates."""

    face_indices: np.ndarray
    coefficients: np.ndarray
    delays_s: np.ndarray
    phase_center_contributions: np.ndarray
    face_centroids: np.ndarray
    face_normals: np.ndarray
    face_areas: np.ndarray
    exact_vertex_path_offsets_m: np.ndarray
    exact_face_areas: np.ndarray
    exact_face_normals: np.ndarray
    path_lengths_m: np.ndarray
    spread_products_m2: np.ndarray
    phase_center_lengths_m: np.ndarray
    phase_center_spreads_m2: np.ndarray
    raw_fractional_visibility: np.ndarray
    latched_fractional_visibility: np.ndarray
    temporal_fade_weights: np.ndarray
    effective_weights: np.ndarray
    visible_sample_counts: np.ndarray
    total_area: float
    effective_visible_area_fraction: float
    visibility_state: FaceVisibilityState
    sample_count_per_face: int
    far_field_max_edge_m: float | None = None
    far_field_nearest_distance_m: float | None = None
    delay_response: PODelayResponse | None = None


def _target_face_hit_counts(
    paths,
    targets: Iterable[MeshTarget],
) -> dict[str, np.ndarray]:
    """Count valid RT interactions with each target's original mesh faces."""

    targets = list(targets)
    counts = {
        target.name: np.zeros(
            target.mesh_sequence.face_count,
            dtype=np.int64,
        )
        for target in targets
    }
    if not targets or not all(
        hasattr(paths, field) for field in ("valid", "objects", "primitives")
    ):
        return counts

    valid = as_numpy(paths.valid).astype(bool, copy=False)
    objects = as_numpy(paths.objects)
    primitives = as_numpy(paths.primitives)
    if objects.ndim < 2 or primitives.shape != objects.shape or objects.shape[1:] != valid.shape:
        return counts

    valid_interactions = valid[None, ...]
    invalid_index = np.uint32(0xFFFFFFFF)
    for target in targets:
        scene_object = target.ensure_scene_object()
        object_id = int(as_numpy(scene_object.object_id).reshape(-1)[0])
        face_count = target.mesh_sequence.face_count
        hit_mask = valid_interactions & (objects == object_id)
        hit_faces = np.asarray(primitives[hit_mask], dtype=np.int64)
        hit_faces = hit_faces[
            (hit_faces >= 0) & (hit_faces < face_count) & (hit_faces != int(invalid_index))
        ]
        if hit_faces.size:
            counts[target.name] += np.bincount(
                hit_faces,
                minlength=face_count,
            )[:face_count]
    return counts


# Thresholds for the phase-center approximation warning: RT's wavefront-
# curvature phase error across the aperture, and PO's per-facet phase error
# from evaluating the surface integral at the array phase center.
RT_PHASE_CENTER_ERROR_LIMIT_RAD = np.pi / 8.0
PO_PHASE_CENTER_ERROR_LIMIT_RAD = 1.0


class PhaseCenterApproximationWarning(UserWarning):
    """The array aperture is too large for the phase-center approximation."""


class MmWaveRadarSimulator:
    # pylint: disable=too-many-instance-attributes
    r"""Simulates FMCW radar sensing over a Sionna scene with mesh targets.

    ``solver`` selects ``rt`` (ray tracing of the whole scene), ``po``
    (physical optics on the targets, ignoring the environment), or
    ``hybrid_po`` (RT for the static room plus PO for the body, with
    blocking and human-room coupling). ``rt_*`` settings apply wherever RT
    runs, and the PO and Hybrid PO settings live in ``po_config`` and
    ``hybrid_config``.

    ``array_model="phase_center"`` traces or integrates once at the Tx and Rx
    array centres and gives each virtual channel its own phase (and, in PO,
    its own delay and spreading). ``array_model="subarray"`` does this once per
    pair of a group in ``array_tx_groups`` and a group in ``array_rx_groups``,
    which must each split the radar's Tx or Rx elements (zero-based; ``None``
    is one group). Calibrated PO parts use the gain fitted on the first pair.

    RT retraces at a frame start every ``rt_refresh_frames`` frames, or at
    every chirp when it is 0; between retraces each path's geometry and phase
    follow the motion. ``rt_refresh_on_displacement`` optionally adds early
    retraces when any vertex moves more than this distance in metres since
    the last trace. It does not replace or reset the frame cadence; ``None``
    disables the trigger. Known paths are re-tested against the moving body every
    ``rt_visibility_refresh_frames`` frames (0 = every chirp), and visibility
    changes fade over ``rt_visibility_fade_frames`` frames. The same visibility
    cadence drives Hybrid PO's blocking of room paths.
    """

    def __init__(
        self,
        *,
        solver: Solver = "rt",
        array_model: ArrayModel = "phase_center",
        array_tx_groups=None,
        array_rx_groups=None,
        path_solver: PathSolver | None = None,
        rt_max_depth: int = 2,
        rt_samples_per_source: int = 100000,
        rt_max_paths_per_source: int = 100000,
        rt_human_bounces: HumanBounces = "single",
        rt_refresh_frames: int = 1,
        rt_refresh_on_displacement: float | None = None,
        rt_visibility_refresh_frames: int = 1,
        rt_visibility_fade_frames: int = 0,
        rt_diffuse_reflection: bool = False,
        rt_refraction: bool = False,
        rt_human_specular_reflection: bool = False,
        rt_seed: int = 42,
        rt_coherent_transition_config: RTCoherentTransitionConfig | None = None,
        po_config: POConfig | None = None,
        hybrid_config: HybridPOConfig | None = None,
        progress: bool = False,
        progress_interval: int = 10,
        compute_backend: ComputeBackend = "numpy",
        compute_precision: ComputePrecision = "float64",
        adc_compute_backend: ComputeBackend | None = None,
        adc_compute_precision: ComputePrecision | None = None,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[dict], None] | None = None,
        **removed,
    ):
        reject_removed_arguments(removed, where="MmWaveRadarSimulator")
        configure_mitsuba_variant()
        self.solver = validate_solver(solver)
        self.array_model = validate_array_model(array_model)
        self.array_tx_groups = validate_array_groups(array_tx_groups, name="array_tx_groups")
        self.array_rx_groups = validate_array_groups(array_rx_groups, name="array_rx_groups")
        if self.array_model == "phase_center" and (
            self.array_tx_groups is not None or self.array_rx_groups is not None
        ):
            raise ValueError("array_tx_groups and array_rx_groups need array_model='subarray'")
        self.rt_human_bounces = validate_rt_human_bounces(rt_human_bounces)
        self.rt_refresh_frames = validate_frames(rt_refresh_frames, name="rt_refresh_frames")
        self.rt_visibility_refresh_frames = validate_frames(
            rt_visibility_refresh_frames, name="rt_visibility_refresh_frames"
        )
        self.rt_visibility_fade_frames = validate_frames(
            rt_visibility_fade_frames, name="rt_visibility_fade_frames"
        )
        if rt_refresh_on_displacement is not None:
            rt_refresh_on_displacement = float(rt_refresh_on_displacement)
            if not np.isfinite(rt_refresh_on_displacement) or rt_refresh_on_displacement <= 0.0:
                raise ValueError(
                    "rt_refresh_on_displacement must be a positive distance in metres"
                )
        self.rt_refresh_on_displacement = rt_refresh_on_displacement
        self.path_solver = path_solver
        self.rt_max_depth = int(rt_max_depth)
        self.rt_samples_per_source = int(rt_samples_per_source)
        self.rt_max_paths_per_source = int(rt_max_paths_per_source)
        self.rt_diffuse_reflection = bool(rt_diffuse_reflection)
        self.rt_refraction = bool(rt_refraction)
        self.rt_human_specular_reflection = bool(rt_human_specular_reflection)
        self.rt_seed = int(rt_seed)
        self.progress = bool(progress)
        self.progress_interval = max(int(progress_interval), 1)
        self.compute_backend = validate_compute_backend(compute_backend, name="compute_backend")
        numpy_dtypes_for_precision(compute_precision)
        self.compute_precision = compute_precision
        self.adc_compute_backend = (
            compute_backend if adc_compute_backend is None else adc_compute_backend
        )
        self.adc_compute_precision = (
            compute_precision if adc_compute_precision is None else adc_compute_precision
        )
        validate_compute_backend(self.adc_compute_backend, name="adc_compute_backend")
        numpy_dtypes_for_precision(self.adc_compute_precision)
        if cancel_check is not None and not callable(cancel_check):
            raise ValueError("cancel_check must be callable or None")
        if progress_callback is not None and not callable(progress_callback):
            raise ValueError("progress_callback must be callable or None")
        self.cancel_check = cancel_check
        self.progress_callback = progress_callback
        if po_config is not None and not isinstance(po_config, POConfig):
            raise ValueError("po_config must be a POConfig or None")
        if hybrid_config is not None and not isinstance(hybrid_config, HybridPOConfig):
            raise ValueError("hybrid_config must be a HybridPOConfig or None")
        self.po_config = POConfig() if po_config is None else po_config
        self.rt_coherent_transition_config = (
            RTCoherentTransitionConfig()
            if rt_coherent_transition_config is None
            else rt_coherent_transition_config
        )
        self._hybrid_config_explicit = hybrid_config is not None
        self.hybrid_config = HybridPOConfig() if hybrid_config is None else hybrid_config
        self.last_target_face_hit_counts: dict[str, np.ndarray] = {}
        self.adc_if_filter: dict = {}

    @property
    def rt_coherent(self) -> bool:
        """Whether RT phase-updates a path bank between retraces."""

        return self.rt_refresh_frames > 0

    def run(
        self,
        scene,
        radar: RadarSensor,
        targets: Iterable[MeshTarget],
        *,
        num_frames: int = 1,
    ) -> RadarCube:
        """Runs the sensing simulation."""
        self.check_cancelled()
        targets = list(targets)
        self.last_target_face_hit_counts = {}
        self.adc_if_filter = {}
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        input_hashes = simulation_input_hashes(scene, radar, targets)
        approximation_event = self._phase_center_approximation(radar, targets)
        if tdm_expansion_required(radar):
            cube = collapse_tdm_cube(
                self._run_array(
                    scene,
                    expanded_tdm_radar(radar),
                    targets,
                    num_frames=int(num_frames),
                ),
                radar,
            )
        else:
            cube = self._run_array(scene, radar, targets, num_frames=int(num_frames))
        if approximation_event is not None:
            cube = replace(
                cube,
                metadata=replace(
                    cube.metadata,
                    events=[*(cube.metadata.events or []), approximation_event],
                ),
            )
        from .compute import resolve_torch_runtime

        def resolved(backend, precision):
            runtime = resolve_torch_runtime(backend, precision)
            return {
                "backend": "torch" if runtime is not None else "numpy",
                "device": str(runtime.device) if runtime is not None else "cpu",
                "precision": precision,
            }

        provenance = capture_provenance(
            seed=self.rt_seed,
            input_hashes=input_hashes,
            settings={
                "solver": self.solver,
                "num_frames": num_frames,
                "po": self.po_config,
                "hybrid": self.hybrid_config,
                "rt_coherent_transition": self.rt_coherent_transition_config,
                "array_model": self.array_model,
                "array_tx_groups": self.array_tx_groups,
                "array_rx_groups": self.array_rx_groups,
                "compute_backend_requested": self.compute_backend,
                "adc_compute_backend_requested": self.adc_compute_backend,
                **{
                    key: value
                    for key, value in vars(self).items()
                    if key.startswith("rt_")
                    and isinstance(value, (int, float, bool, str, type(None)))
                },
            },
            runtime={
                "po": resolved(self.compute_backend, self.compute_precision),
                "adc": resolved(self.adc_compute_backend, self.adc_compute_precision),
                "adc_dtype": str(cube.adc.dtype),
            },
        )
        return replace(cube, metadata=replace(cube.metadata, provenance=provenance))

    def _run_array(
        self,
        scene,
        radar: RadarSensor,
        targets: list[MeshTarget],
        *,
        num_frames: int,
    ) -> RadarCube:
        """Runs the solver once, or once per Tx/Rx group pair for subarrays."""

        if self.array_model == "phase_center":
            return self._run_solver(scene, radar, targets, num_frames=num_frames)
        pairs = group_pairs(radar, self.array_tx_groups, self.array_rx_groups)
        callback = self.progress_callback
        reflector_bank = None
        reflector_selection_s = 0.0
        if self.solver == "hybrid_po" and len(targets) == 1:
            times = np.array(
                [
                    radar.fmcw.chirp_time(0, 0),
                    radar.fmcw.chirp_time(num_frames - 1, radar.fmcw.num_chirps_per_frame - 1),
                ]
            )
            with targets[0].detached_from_scene(scene):
                selection_t0 = time.perf_counter()
                reflector_bank = _select_reflectors(
                    scene,
                    radar,
                    targets[0],
                    times,
                    self.hybrid_config,
                )
                reflector_selection_s = time.perf_counter() - selection_t0

        def execute(subradar, report):
            self.progress_callback = report
            options = {} if reflector_bank is None else {"reflector_bank": reflector_bank}
            return self._run_solver(
                scene,
                subradar,
                targets,
                num_frames=num_frames,
                **options,
            )

        try:
            cubes = run_group_pairs(
                radar,
                pairs,
                execute,
                scenes=(scene,),
                callback=callback,
                check_cancelled=self.check_cancelled,
                progress=self.progress,
            )
        finally:
            self.progress_callback = callback
        cube = merge_subarray_cubes(
            cubes,
            pairs,
            radar.hardware.num_virtual_channels,
            solver=self.solver,
        )
        if reflector_bank is not None:
            cube = _with_shared_reflector_profile(cube, reflector_selection_s)
        event = partition_event(pairs)
        return replace(
            cube,
            metadata=replace(cube.metadata, events=[event, *cube.metadata.events]),
        )

    def _run_solver(
        self,
        scene,
        radar: RadarSensor,
        targets: list[MeshTarget],
        *,
        num_frames: int,
        reflector_bank=None,
    ) -> RadarCube:
        """Dispatches one run to the selected solver."""

        if self.solver == "po":
            return self._run_po(
                radar,
                targets,
                num_frames=num_frames,
                rt_reference_scene=scene,
            )
        if self.solver == "hybrid_po":
            # RT leaves targets in the scene; the static trace must not
            # include a frozen copy of the moving mesh.
            with ExitStack() as stack:
                for target in targets:
                    stack.enter_context(target.detached_from_scene(scene))
                return self._run_hybrid_po(
                    scene,
                    radar,
                    targets,
                    num_frames=num_frames,
                    reflector_bank=reflector_bank,
                )
        if self.rt_coherent:
            return self._run_rt_path_bank(scene, radar, targets, num_frames=num_frames)
        return self._run_rt_every_chirp(scene, radar, targets, num_frames=num_frames)

    def _phase_center_approximation(
        self,
        radar: RadarSensor,
        targets: list[MeshTarget],
    ) -> dict | None:
        """Checks the phase-center approximation and warns when it is coarse.

        RT applies one plane-wave phase per element and ignores wavefront
        curvature, so its error grows as pi*(d_tx**2 + d_rx**2)/(lambda*R).
        PO evaluates each facet's surface integral at the phase center, so a
        facet of size a sees an error of about k*a*(d_tx + d_rx)/R. Here d is
        the largest element offset from its phase center (its group's centre
        for subarrays) and R the nearest target distance in the first chirp.
        This also checks that subarray groups split the radar's elements.
        """

        tx_groups = self.array_tx_groups if self.array_model == "subarray" else None
        rx_groups = self.array_rx_groups if self.array_model == "subarray" else None
        tx_offset, rx_offset = max_group_offsets(radar, tx_groups, rx_groups)
        if not targets or (tx_offset == 0.0 and rx_offset == 0.0):
            return None
        tx = np.asarray(radar.world_tx_positions(), dtype=np.float64).reshape(-1, 3)
        rx = np.asarray(radar.world_rx_positions(), dtype=np.float64).reshape(-1, 3)
        center = 0.5 * (tx.mean(axis=0) + rx.mean(axis=0))
        wavelength = float(radar.fmcw.wavelength)
        nearest_m = np.inf
        largest_edge_m = 0.0
        for target in targets:
            sequence = target.mesh_sequence
            vertices = np.asarray(
                sequence.vertices_at(float(np.asarray(sequence.times)[0])),
                dtype=np.float64,
            )
            nearest_m = min(
                nearest_m,
                float(np.min(np.linalg.norm(vertices - center, axis=1))),
            )
            faces = np.asarray(sequence.faces, dtype=np.int64).reshape(-1, 3)
            if faces.size:
                corners = vertices[faces]
                edges = np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2)
                largest_edge_m = max(largest_edge_m, float(np.max(edges)))
        if not np.isfinite(nearest_m) or nearest_m <= 0.0:
            return None
        tx_group_count = 1 if tx_groups is None else len(tx_groups)
        rx_group_count = 1 if rx_groups is None else len(rx_groups)
        event = {
            "reason": "phase_center_approximation",
            "array_model": self.array_model,
            "array_tx_group_count": tx_group_count,
            "array_rx_group_count": rx_group_count,
            "nearest_target_distance_m": nearest_m,
            "max_tx_offset_m": tx_offset,
            "max_rx_offset_m": rx_offset,
        }
        problems = []
        # Splitting an array into n groups shrinks d by about n.
        shrink = 1.0
        if self.solver == "rt":
            error = np.pi * (tx_offset**2 + rx_offset**2) / (wavelength * nearest_m)
            event["rt_curvature_phase_error_rad"] = float(error)
            event["rt_phase_error_limit_rad"] = float(RT_PHASE_CENTER_ERROR_LIMIT_RAD)
            if error > RT_PHASE_CENTER_ERROR_LIMIT_RAD:
                shrink = max(shrink, float(np.sqrt(error / RT_PHASE_CENTER_ERROR_LIMIT_RAD)))
                problems.append(
                    f"RT wavefront-curvature phase error {error:.2f} rad exceeds "
                    f"{RT_PHASE_CENTER_ERROR_LIMIT_RAD:.2f} rad"
                )
        if self.solver in ("po", "hybrid_po") and largest_edge_m > 0.0:
            error = 2.0 * np.pi / wavelength * largest_edge_m * (tx_offset + rx_offset) / nearest_m
            event["po_facet_phase_error_rad"] = float(error)
            event["po_phase_error_limit_rad"] = float(PO_PHASE_CENTER_ERROR_LIMIT_RAD)
            event["largest_facet_edge_m"] = largest_edge_m
            if error > PO_PHASE_CENTER_ERROR_LIMIT_RAD:
                shrink = max(shrink, error / PO_PHASE_CENTER_ERROR_LIMIT_RAD)
                problems.append(
                    f"PO phase error across a {1e3 * largest_edge_m:.1f} mm facet "
                    f"{error:.2f} rad exceeds {PO_PHASE_CENTER_ERROR_LIMIT_RAD:.2f} rad"
                )
        event["warning"] = bool(problems)
        if problems:
            suggested_tx = (
                min(radar.hardware.num_tx, int(np.ceil(tx_group_count * shrink)))
                if tx_offset > 0.0
                else tx_group_count
            )
            suggested_rx = (
                min(radar.hardware.num_rx, int(np.ceil(rx_group_count * shrink)))
                if rx_offset > 0.0
                else rx_group_count
            )
            event["suggested_array_tx_group_count"] = suggested_tx
            event["suggested_array_rx_group_count"] = suggested_rx
            warnings.warn(
                "The phase-center approximation is coarse for this aperture at "
                f"{nearest_m:.2f} m: "
                + "; ".join(problems)
                + f". Split the array into about {suggested_tx} Tx and "
                f"{suggested_rx} Rx groups (array_model='subarray'), move the "
                "target farther away, or reduce the active elements.",
                PhaseCenterApproximationWarning,
                stacklevel=3,
            )
        return event

    @staticmethod
    def _human_visibility_scene(targets, vertices):
        """Builds one occlusion mesh from the current target meshes."""

        faces = []
        offset = 0
        for target, target_vertices in zip(targets, vertices):
            faces.append(np.asarray(target.mesh_sequence.faces, dtype=np.int64) + offset)
            offset += int(np.asarray(target_vertices).shape[0])
        return HumanVisibilityScene(
            faces=np.concatenate(faces, axis=0).astype(np.uint32),
            vertices=np.concatenate(vertices, axis=0),
            name="rt-path-visibility",
        )

    def check_cancelled(self) -> None:
        """Raises at a safe solver boundary when cancellation was requested."""

        if self.cancel_check is not None and bool(self.cancel_check()):
            raise InterruptedError("Simulation cancelled by user")

    def report_frame_progress(
        self,
        *,
        mode: str,
        frame: int,
        num_frames: int,
    ) -> None:
        """Reports only the active solver mode and one-based frame progress."""

        if self.progress_callback is not None:
            self.progress_callback(
                {
                    "solver_mode": str(mode),
                    "frame_index": int(frame),
                    "num_frames": int(num_frames),
                }
            )

    def report_chirp_progress(
        self,
        *,
        mode: str,
        frame: int,
        num_frames: int,
        chirp: int,
        num_chirps: int,
    ) -> None:
        """Reports the active chirp within the current solver pass."""

        if self.progress_callback is not None:
            self.progress_callback(
                {
                    "solver_mode": str(mode),
                    "frame_index": int(frame),
                    "num_frames": int(num_frames),
                    "chirp_index": int(chirp),
                    "num_chirps": int(num_chirps),
                }
            )

    def _log_frame_progress(
        self,
        *,
        mode: str,
        frame: int,
        num_frames: int,
        frame_start_s: float,
        extra: str = "",
    ) -> None:
        """Prints throttled frame progress when simulator progress is enabled."""

        if not self.progress:
            return
        is_final_frame = frame == num_frames - 1
        is_interval_frame = (frame + 1) % self.progress_interval == 0
        if not is_final_frame and not is_interval_frame:
            return
        elapsed = time.perf_counter() - frame_start_s
        rate = elapsed / max(frame + 1, 1)
        remaining = rate * max(num_frames - frame - 1, 0)
        suffix = f" {extra}" if extra else ""
        print(
            f"[{mode}] frame {frame + 1}/{num_frames} "
            f"elapsed={elapsed:.1f}s eta={remaining:.1f}s{suffix}",
            flush=True,
        )

    def log_frame_progress(
        self,
        *,
        mode: str,
        frame: int,
        num_frames: int,
        frame_start_s: float,
        extra: str = "",
    ) -> None:
        """Logs progress for long-running frame/chirp simulation loops."""

        self._log_frame_progress(
            mode=mode,
            frame=frame,
            num_frames=num_frames,
            frame_start_s=frame_start_s,
            extra=extra,
        )

    def _log_chirp_progress(
        self,
        *,
        mode: str,
        frame: int,
        num_frames: int,
        chirp: int,
        num_chirps: int,
        run_start_s: float,
        chirp_runtime_s: float,
        extra: str = "",
    ) -> None:
        """Prints throttled chirp progress when simulator progress is enabled."""

        if not self.progress:
            return
        chirps_total = int(num_frames) * int(num_chirps)
        chirp_index = int(frame) * int(num_chirps) + int(chirp)
        chirps_done = chirp_index + 1
        is_final_chirp = chirps_done == chirps_total
        log_every = 1 if chirps_total <= 64 else max(1, chirps_total // 20)
        if not is_final_chirp and chirps_done % log_every != 0:
            return
        elapsed = time.perf_counter() - run_start_s
        rate = elapsed / max(chirps_done, 1)
        remaining = rate * max(chirps_total - chirps_done, 0)
        suffix = f" {extra}" if extra else ""
        print(
            f"[{mode}] chirp {chirps_done}/{chirps_total} "
            f"(frame {frame + 1}/{num_frames}, chirp {chirp + 1}/{num_chirps}) "
            f"chirp_elapsed={chirp_runtime_s:.1f}s "
            f"elapsed={elapsed:.1f}s eta={remaining:.1f}s{suffix}",
            flush=True,
        )

    def rt_transition_crossfade_enabled(self) -> bool:
        """Returns whether coherent RT transitions should crossfade."""

        return self._rt_transition_crossfade_enabled()

    def _rt_transition_crossfade_enabled(self) -> bool:
        """Resolves coherent RT transition crossfade behavior from configuration."""

        config = self.rt_coherent_transition_config
        if not bool(config.enabled):
            return False
        if config.crossfade is None:
            return self.rt_refresh_on_displacement is not None
        return bool(config.crossfade)

    def rt_retrace_period_chirps(self, fmcw) -> int:
        """Returns the retrace interval in chirps implied by rt_refresh_frames."""

        return refresh_period_chirps(self.rt_refresh_frames, fmcw.num_chirps_per_frame)

    def periodic_chirp_retrace_due(
        self,
        *,
        frame: int,
        chirp: int,
        num_chirps_per_frame: int,
        period_chirps: int,
        initialized: bool,
    ) -> bool:
        """Returns whether a periodic retrace is due for this chirp."""

        return self._periodic_chirp_retrace_due(
            frame=frame,
            chirp=chirp,
            num_chirps_per_frame=num_chirps_per_frame,
            period_chirps=period_chirps,
            initialized=initialized,
        )

    def _periodic_chirp_retrace_due(
        self,
        *,
        frame: int,
        chirp: int,
        num_chirps_per_frame: int,
        period_chirps: int,
        initialized: bool,
    ) -> bool:
        """Returns true for initial traces and chirps aligned to the retrace period."""

        if not initialized:
            return True
        flat_chirp = int(frame) * int(num_chirps_per_frame) + int(chirp)
        return flat_chirp % int(period_chirps) == 0

    @staticmethod
    def _periodic_chirp_retrace_reason(
        *,
        period_chirps: int,
        num_chirps_per_frame: int,
    ) -> str:
        """Returns the metadata reason for frame-periodic versus chirp-periodic RT."""

        return "frame" if int(period_chirps) % int(num_chirps_per_frame) == 0 else "periodic_chirp"

    def retrace_event_reason(
        self,
        *,
        first_event: bool,
        period_chirps: int,
        num_chirps_per_frame: int,
        periodic_due: bool = True,
    ) -> str:
        """Returns the event reason string used in retrace metadata."""

        return self._retrace_event_reason(
            first_event=first_event,
            period_chirps=period_chirps,
            num_chirps_per_frame=num_chirps_per_frame,
            periodic_due=periodic_due,
        )

    def _retrace_event_reason(
        self,
        *,
        first_event: bool,
        period_chirps: int,
        num_chirps_per_frame: int,
        periodic_due: bool = True,
    ) -> str:
        """Returns the metadata reason for the current RT retrace event."""

        if first_event:
            return "initial"
        if periodic_due:
            return self._periodic_chirp_retrace_reason(
                period_chirps=period_chirps,
                num_chirps_per_frame=num_chirps_per_frame,
            )
        return "displacement"

    def _run_rt_every_chirp(
        self,
        scene,
        radar: RadarSensor,
        targets: list[MeshTarget],
        *,
        num_frames: int,
        reference_collector=None,
    ) -> RadarCube:
        """Runs exact RT retracing on every chirp (rt_refresh_frames=0)."""

        radar.configure_scene(scene)
        for target in targets:
            target.add_to_scene(scene)

        fmcw = radar.fmcw
        num_chirps = fmcw.num_chirps_per_frame
        retrace_period_chirps = self.rt_retrace_period_chirps(fmcw)
        num_adc = fmcw.num_adc_samples
        num_vc = radar.hardware.num_virtual_channels
        vc_tx = radar.hardware.virtual_tx_indices()
        vc_rx = radar.hardware.virtual_rx_indices()
        fallback_tx, fallback_rx = resolve_virtual_channel_pairs(
            radar.hardware.num_tx,
            radar.hardware.num_rx,
            order=radar.hardware.virtual_channel_order,
        )
        path_array_channel_mapping = {}
        if not np.array_equal(vc_tx, fallback_tx) or not np.array_equal(vc_rx, fallback_rx):
            path_array_channel_mapping = {
                "virtual_channel_tx_indices": vc_tx,
                "virtual_channel_rx_indices": vc_rx,
            }
        _, adc_complex_dtype = self._numpy_dtypes_for_precision(self.adc_compute_precision)
        adc = np.zeros((num_frames, num_chirps, num_adc, num_vc), dtype=adc_complex_dtype)
        times = (
            np.arange(num_frames, dtype=float)[:, None] * fmcw.frame_period
            + np.arange(num_chirps, dtype=float)[None, :] * fmcw.chirp_repetition_time
        )
        path_counts = np.zeros((num_frames, num_chirps), dtype=np.int64)
        max_displacements = np.zeros((num_frames, num_chirps), dtype=float)
        coherent_channel = np.zeros((num_frames, num_chirps, num_vc), dtype=np.complex128)
        events = []

        frame_start_s = time.perf_counter()
        frame_retraces = 0
        for frame in range(num_frames):
            self.report_frame_progress(
                mode="rt",
                frame=frame,
                num_frames=num_frames,
            )
            if frame > 0:
                self._log_frame_progress(
                    mode="rt",
                    frame=frame - 1,
                    num_frames=num_frames,
                    frame_start_s=frame_start_s,
                    extra=f"retraces={frame_retraces}",
                )
                frame_retraces = 0
            for chirp in range(num_chirps):
                self.check_cancelled()
                chirp_time = fmcw.chirp_time(frame, chirp)
                times[frame, chirp] = chirp_time

                max_disp = 0.0
                for target in targets:
                    vertices = target.update_to_time(chirp_time)
                    if target.reference_vertices is None:
                        target.reference_vertices = vertices.copy()
                    disp = np.max(np.linalg.norm(vertices - target.reference_vertices, axis=1))
                    max_disp = max(max_disp, float(disp))
                max_displacements[frame, chirp] = max_disp

                paths = self.trace_scene(scene)
                frame_retraces += 1
                retrace_face_hits = _target_face_hit_counts(paths, targets)
                for name, hit_counts in retrace_face_hits.items():
                    if name not in self.last_target_face_hit_counts:
                        self.last_target_face_hit_counts[name] = np.zeros_like(hit_counts)
                    self.last_target_face_hit_counts[name] += hit_counts
                for target in targets:
                    target.reference_vertices = target.mesh_sequence.vertices_at(chirp_time)
                events.append(
                    {
                        "frame": frame,
                        "chirp": chirp,
                        "time": chirp_time,
                        "reason": self._retrace_event_reason(
                            first_event=len(events) == 0,
                            period_chirps=retrace_period_chirps,
                            num_chirps_per_frame=num_chirps,
                        ),
                        "max_displacement": max_disp,
                        "retrace_period_chirps": retrace_period_chirps,
                        "human_specular_policy": (
                            "allow" if self.rt_human_specular_reflection else "drop"
                        ),
                    }
                )

                a, tau, valid = self.path_arrays_from_paths(
                    paths,
                    targets,
                    virtual_channel_order=radar.hardware.virtual_channel_order,
                    **path_array_channel_mapping,
                )
                a = apply_path_hardware_gain(
                    radar.hardware,
                    a,
                    paths=paths,
                    frequency_hz=fmcw.carrier_frequency,
                    radar_orientation=radar.orientation,
                )
                if reference_collector is not None:
                    target_counts, total_counts = self.target_interaction_counts(paths, targets)
                    reference_collector.observe(
                        frame, chirp, a, tau, valid, target_counts, total_counts
                    )
                path_counts[frame, chirp] = int(np.count_nonzero(valid))
                coherent_channel[frame, chirp] = self._coherent_channel(a, valid)
                adc[frame, chirp] = self.synthesize_adc(
                    a,
                    tau,
                    valid,
                    fmcw,
                    backend=self.adc_compute_backend,
                    precision=self.adc_compute_precision,
                )

        self._log_frame_progress(
            mode="rt",
            frame=num_frames - 1,
            num_frames=num_frames,
            frame_start_s=frame_start_s,
            extra=f"retraces={frame_retraces}",
        )

        phase_jumps = self._phase_jump_diagnostics(coherent_channel)
        metadata = SensingMetadata(
            adc_if_filter=dict(self.adc_if_filter),
            events=events,
            path_counts=path_counts,
            max_displacements=max_displacements,
            virtual_channel_order=radar.hardware.virtual_channel_order,
            solver="rt",
            max_unwrapped_phase_jump=phase_jumps,
        )
        return RadarCube(adc=adc, times=times, metadata=metadata)

    def _plan_rt_retraces(self, radar, targets, *, num_frames, times, max_displacements):
        """Choose anchors from cadence and displacement before tracing any paths."""
        fmcw = radar.fmcw
        num_chirps = fmcw.num_chirps_per_frame
        retrace_period_chirps = self.rt_retrace_period_chirps(fmcw)
        threshold = self.rt_refresh_on_displacement
        reference_vertices = [None for _ in targets]
        events: list[dict] = []
        event_flat_indices: list[int] = []
        event_anchor_vertices: list[list[np.ndarray]] = []

        for frame in range(num_frames):
            for chirp in range(num_chirps):
                self.check_cancelled()
                flat_index = frame * num_chirps + chirp
                chirp_time = fmcw.chirp_time(frame, chirp)
                times[frame, chirp] = chirp_time
                periodic_due = self._periodic_chirp_retrace_due(
                    frame=frame,
                    chirp=chirp,
                    num_chirps_per_frame=num_chirps,
                    period_chirps=retrace_period_chirps,
                    initialized=bool(events),
                )
                if threshold is None and not periodic_due:
                    continue

                current_vertices = [
                    target.mesh_sequence.vertices_at(chirp_time) for target in targets
                ]

                max_disp = 0.0
                for target_index, vertices in enumerate(current_vertices):
                    if reference_vertices[target_index] is None:
                        reference_vertices[target_index] = vertices.copy()
                    disp = np.max(
                        np.linalg.norm(vertices - reference_vertices[target_index], axis=1)
                    )
                    max_disp = max(max_disp, float(disp))
                max_displacements[frame, chirp] = max_disp

                if not periodic_due and not (threshold is not None and max_disp > threshold):
                    continue

                for target_index, vertices in enumerate(current_vertices):
                    reference_vertices[target_index] = vertices.copy()
                event_flat_indices.append(flat_index)
                event_anchor_vertices.append([v.copy() for v in current_vertices])
                events.append(
                    {
                        "frame": frame,
                        "chirp": chirp,
                        "time": float(chirp_time),
                        "reason": self._retrace_event_reason(
                            first_event=len(events) == 0,
                            period_chirps=retrace_period_chirps,
                            num_chirps_per_frame=num_chirps,
                            periodic_due=periodic_due,
                        ),
                        "max_displacement": max_disp,
                        **({"threshold": threshold} if threshold is not None else {}),
                        "retrace_period_chirps": retrace_period_chirps,
                        "coherent_path_update": True,
                        "rt_human_specular_reflection": bool(self.rt_human_specular_reflection),
                        "human_specular_policy": (
                            "allow" if self.rt_human_specular_reflection else "drop"
                        ),
                    }
                )

        return _RTRetracePlan(events, event_flat_indices, event_anchor_vertices)

    def _trace_rt_banks(
        self, scene, radar, targets, plan, *, num_frames, reference_collector, add_runtime_profile
    ):
        """Trace each anchor and retain its coherent path bank and diagnostics."""
        events, event_anchor_vertices = plan.events, plan.anchor_vertices
        target_ids = self._target_object_ids(targets)
        event_banks: list[CoherentRTPathBank] = []
        event_counts = []
        for event, anchor_vertices in zip(events, event_anchor_vertices):
            self.check_cancelled()
            self.report_frame_progress(
                mode="rt",
                frame=int(event["frame"]),
                num_frames=num_frames,
            )
            for target, vertices in zip(targets, anchor_vertices):
                target.update_to_time(
                    float(event["time"]),
                    vertices=vertices,
                )
                target.reference_vertices = vertices.copy()
            trace_t0 = time.perf_counter()
            paths = self.trace_scene(scene)
            add_runtime_profile(
                "rt_path_bank_trace",
                time.perf_counter() - trace_t0,
            )
            for name, hit_counts in _target_face_hit_counts(paths, targets).items():
                if name not in self.last_target_face_hit_counts:
                    self.last_target_face_hit_counts[name] = np.zeros_like(hit_counts)
                self.last_target_face_hit_counts[name] += hit_counts
            if reference_collector is not None:
                event_counts.append(self.target_interaction_counts(paths, targets))
            extract_t0 = time.perf_counter()
            bank = extract_coherent_rt_path_bank(
                paths,
                radar,
                targets,
                virtual_channel_order=radar.hardware.virtual_channel_order,
                anchor_target_vertices=anchor_vertices,
            )
            if not self.rt_human_specular_reflection:
                bank = self._drop_target_specular_bank(bank, target_ids)
            add_runtime_profile(
                "rt_path_bank_extract",
                time.perf_counter() - extract_t0,
            )
            event_banks.append(bank)

        return event_banks, event_counts

    def _match_rt_bank_transitions(self, radar, plan, event_banks, add_runtime_profile):
        """Match adjacent same-frame banks for transition diagnostics and fading."""
        fmcw = radar.fmcw
        events = plan.events
        event_flat_indices = plan.flat_indices
        event_anchor_vertices = plan.anchor_vertices
        transition_config = self.rt_coherent_transition_config
        transition_matching_enabled = bool(transition_config.enabled)
        transition_crossfade_enabled = self._rt_transition_crossfade_enabled()
        transitions: list[CoherentRTPathTransition | None] = []
        for event_index in range(max(len(event_banks) - 1, 0)):
            transition_interval_chirps = int(
                event_flat_indices[event_index + 1] - event_flat_indices[event_index]
            )
            same_frame_retrace = int(events[event_index]["frame"]) == int(
                events[event_index + 1]["frame"]
            )
            events[event_index]["transition_interval_chirps"] = transition_interval_chirps
            events[event_index]["transition_same_frame"] = bool(same_frame_retrace)
            events[event_index]["transition_diagnostic_valid"] = False
            events[event_index]["transition_crossfade"] = False
            if not transition_matching_enabled or not same_frame_retrace:
                transitions.append(None)
                continue

            next_vertices = event_anchor_vertices[event_index + 1]
            match_t0 = time.perf_counter()
            transition = match_coherent_rt_path_banks(
                event_banks[event_index],
                event_banks[event_index + 1],
                next_vertices,
                wavelength_m=fmcw.wavelength,
                match_delay_tolerance_fraction=(transition_config.match_delay_tolerance_fraction),
                match_separation_fraction=(transition_config.match_separation_fraction),
                backend=self.compute_backend,
                precision=self.compute_precision,
            )
            add_runtime_profile(
                "rt_path_bank_transition_match",
                time.perf_counter() - match_t0,
            )
            transitions.append(transition)
            events[event_index]["transition_diagnostic_valid"] = True
            events[event_index]["persistent_path_count"] = int(transition.matched_old_indices.size)
            events[event_index]["death_path_count"] = int(transition.old_only_indices.size)
            events[event_index]["birth_path_count"] = int(transition.new_only_indices.size)
            events[event_index]["transition_crossfade"] = bool(transition_crossfade_enabled)
        if events:
            events[-1].setdefault("transition_interval_chirps", 0)
            events[-1].setdefault("transition_same_frame", False)
            events[-1].setdefault("transition_diagnostic_valid", False)
            events[-1].setdefault(
                "persistent_path_count",
                int(np.count_nonzero(event_banks[-1].valid)) if event_banks else 0,
            )
            events[-1].setdefault("death_path_count", 0)
            events[-1].setdefault("birth_path_count", 0)
            events[-1].setdefault("transition_crossfade", False)

        return transitions

    def _run_rt_path_bank(
        self,
        scene,
        radar: RadarSensor,
        targets: list[MeshTarget],
        *,
        num_frames: int,
        reference_collector=None,
    ) -> RadarCube:
        """Runs RT with coherent chirp updates between retraces."""
        return _RTPathBankRun(
            self,
            scene=scene,
            radar=radar,
            targets=targets,
            num_frames=num_frames,
            reference_collector=reference_collector,
        ).run()

    def _run_hybrid_po(
        self,
        scene,
        radar: RadarSensor,
        targets: list[MeshTarget],
        *,
        num_frames: int,
        reflector_bank=None,
    ) -> RadarCube:
        """Runs Hybrid PO: static-environment RT plus human PO."""

        config = self.hybrid_config
        if not targets:
            raise ValueError("hybrid_po requires one mesh target")
        if len(targets) > 1:
            raise ValueError("hybrid_po supports one mesh target")

        total_t0 = time.perf_counter()
        runtime_profile_s = {
            "hybrid_static_trace": 0.0,
            "hybrid_segment_extraction": 0.0,
            "hybrid_reflector_extraction": 0.0,
            "hybrid_rt_baseline": 0.0,
            "hybrid_po_calibration": 0.0,
            "hybrid_human_po": 0.0,
            "hybrid_blocking_rays": 0.0,
            "hybrid_static_adc_synthesis": 0.0,
            "hybrid_coupling_channel": 0.0,
            "hybrid_coupling_setup": 0.0,
            "hybrid_human_env_channel": 0.0,
            "hybrid_env_human_channel": 0.0,
            "hybrid_coupling_adc_synthesis": 0.0,
            "hybrid_human_env_adc_synthesis": 0.0,
            "hybrid_env_human_adc_synthesis": 0.0,
            "hybrid_assembly": 0.0,
        }
        runtime_profile_counts = dict.fromkeys(runtime_profile_s, 0)

        fmcw = radar.fmcw
        radar.configure_scene(scene)
        human_t0 = time.perf_counter()
        human_cube = self._run_po(
            radar,
            targets,
            num_frames=num_frames,
            static_visibility_scene=scene.mi_scene,
            po_config=config.po_config,
        )
        runtime_profile_s["hybrid_human_po"] = time.perf_counter() - human_t0
        runtime_profile_counts["hybrid_human_po"] = 1
        if human_cube.metadata.runtime_profile_s:
            for key, value in human_cube.metadata.runtime_profile_s.items():
                runtime_profile_s[f"human_{key}"] = float(value)
        if human_cube.metadata.runtime_profile_counts:
            for key, value in human_cube.metadata.runtime_profile_counts.items():
                runtime_profile_counts[f"human_{key}"] = int(value)

        times = human_cube.times.copy()
        static_pass = _run_static_environment_pass(
            scene=scene,
            radar=radar,
            target=targets[0],
            times=times,
            simulator=self,
            config=config,
            reflector_bank=reflector_bank,
        )
        runtime_profile_s.update(static_pass.runtime_profile_s)
        runtime_profile_counts.update(static_pass.runtime_profile_counts)
        static_bank = static_pass.static_bank
        reflector_bank = static_pass.reflector_bank
        coupling_reflector_count = int(reflector_bank.vertices.shape[0])
        static_path_count = int(np.count_nonzero(static_bank.valid))
        static_unblocked_adc = static_pass.static_unblocked_adc
        static_blocked_adc = static_pass.static_blocked_adc
        path_visibility_weights = static_pass.visibility_weights
        static_path_counts = static_pass.static_path_counts
        blocked_counts = static_pass.blocked_counts
        mean_visibility = static_pass.mean_visibility

        _, adc_complex_dtype = self._numpy_dtypes_for_precision(self.adc_compute_precision)
        calibration = config.hybrid_calibration
        po_calibration = {"mode": calibration.mode}
        human_po_gain = 1.0
        rt_baseline_components = {}
        if rt_calibration_enabled(calibration):
            calibration_num_frames = num_frames if calibration.mode == "rt_sequence" else 1
            if self.progress:
                print(
                    "[hybrid_po] starting full-scene RT baseline "
                    f"{calibration_num_frames} frames x "
                    f"{fmcw.num_chirps_per_frame} chirps "
                    f"for {calibration.mode} calibration",
                    flush=True,
                )
            baseline_t0 = time.perf_counter()
            rt_baseline = self._run_rt_baseline_for_calibration(
                scene,
                radar,
                targets,
                num_frames=calibration_num_frames,
                calibration_reference_component=(
                    calibration.reference_component if calibration.mode == "rt_sequence" else None
                ),
                rt_refresh_frames=calibration.rt_refresh_frames,
            )
            baseline_elapsed_s = time.perf_counter() - baseline_t0
            runtime_profile_s["hybrid_rt_baseline"] = baseline_elapsed_s
            runtime_profile_counts["hybrid_rt_baseline"] = 1
            if self.progress:
                print(
                    "[hybrid_po] finished full-scene RT baseline "
                    f"elapsed={baseline_elapsed_s:.1f}s "
                    f"retraces={int(rt_baseline.get('retrace_count', 0))}",
                    flush=True,
                )
            calibration_t0 = time.perf_counter()
            if calibration.mode == "rt_sequence":
                human_po_gain, po_calibration = fit_sequence_adc_calibration(
                    rt_baseline=rt_baseline,
                    po_times=human_cube.times,
                    po_adc=human_cube.adc,
                    config=calibration,
                )
            else:
                human_po_gain, po_calibration = self._po_amplitude_calibration(
                    rt_reference_adc=rt_baseline["human_reference_adc"],
                    po_reference_adc=human_cube.adc[0],
                    config=calibration,
                )
                po_calibration["reference_chirps"] = int(fmcw.num_chirps_per_frame)
            po_calibration["rt_reference_human_path_count_min"] = int(
                np.min(rt_baseline["human_reference_path_counts"])
            )
            po_calibration["rt_reference_human_path_count_mean"] = float(
                np.mean(rt_baseline["human_reference_path_counts"])
            )
            po_calibration["rt_reference_human_path_count_max"] = int(
                np.max(rt_baseline["human_reference_path_counts"])
            )
            po_calibration["rt_baseline_retrace_count"] = int(rt_baseline.get("retrace_count", 0))
            po_calibration["rt_baseline_num_frames"] = int(calibration_num_frames)
            po_calibration.update(rt_baseline_schedule_metadata(rt_baseline))
            runtime_profile_s["hybrid_po_calibration"] = time.perf_counter() - calibration_t0
            runtime_profile_counts["hybrid_po_calibration"] = 1
            if calibration.keep_rt_baseline:
                rt_baseline_components = {
                    "rt_baseline_one_human_first_frame": (
                        rt_baseline["human_reference_adc"][None, ...]
                    ),
                }

        human_po_gain_map = calibration_gain_array(
            po_calibration,
            human_cube.times.shape,
            fallback=human_po_gain,
        )
        chirp_gain = human_po_gain_map[..., None, None]
        human_po_adc = (human_cube.adc * chirp_gain).astype(
            adc_complex_dtype,
            copy=False,
        )
        coupling_pass = _run_coupling_pass(
            scene=scene,
            radar=radar,
            target=targets[0],
            times=times,
            simulator=self,
            config=config,
            reflector_bank=reflector_bank,
            gain_map=human_po_gain_map,
            progress=self.progress,
            progress_every_frames=self.progress_interval,
        )
        human_env_adc = coupling_pass.human_env_adc
        env_human_adc = coupling_pass.env_human_adc
        human_env_counts = coupling_pass.human_env_counts
        env_human_counts = coupling_pass.env_human_counts
        human_env_min_lengths = coupling_pass.human_env_min_lengths
        human_env_mean_lengths = coupling_pass.human_env_mean_lengths
        human_env_max_lengths = coupling_pass.human_env_max_lengths
        env_human_min_lengths = coupling_pass.env_human_min_lengths
        env_human_mean_lengths = coupling_pass.env_human_mean_lengths
        env_human_max_lengths = coupling_pass.env_human_max_lengths
        runtime_profile_s.update(coupling_pass.runtime_profile_s)
        runtime_profile_counts.update(coupling_pass.runtime_profile_counts)

        assembly_t0 = time.perf_counter()
        adc = (static_blocked_adc + human_po_adc + human_env_adc + env_human_adc).astype(
            adc_complex_dtype
        )
        runtime_profile_s["hybrid_assembly"] = time.perf_counter() - assembly_t0
        runtime_profile_counts["hybrid_assembly"] = 1
        runtime_profile_s["hybrid_total"] = time.perf_counter() - total_t0
        runtime_profile_counts["hybrid_total"] = 1

        metadata = SensingMetadata(
            adc_if_filter=dict(self.adc_if_filter),
            events=[
                {
                    "reason": "static_environment_trace",
                    "static_path_count": static_path_count,
                    "segment_count": int(static_bank.segment_path_indices.size),
                    "blocking_aabb_culling": bool(config.blocking_aabb_culling),
                    "blocking_aabb_margin_m": float(config.blocking_aabb_margin_m),
                    "hybrid_coupling": bool(config.hybrid_coupling),
                    "coupling_reflector_count": coupling_reflector_count,
                    "coupling_candidate_source": "scene_geometry_image_source_power",
                    "hybrid_reflector_cutoff_db": config.hybrid_reflector_cutoff_db,
                    "hybrid_max_reflectors": int(config.hybrid_max_reflectors),
                    "reflector_selection": getattr(reflector_bank, "selection_metadata", None),
                    "po_calibration_mode": po_calibration.get("mode"),
                    "po_calibration_gain": po_calibration.get("human_po_amplitude_gain", 1.0),
                },
                {"reason": "po_calibration", **po_calibration},
            ],
            path_counts=(
                static_path_counts
                + human_cube.metadata.path_counts
                + human_env_counts
                + env_human_counts
            ),
            max_displacements=human_cube.metadata.max_displacements,
            virtual_channel_order=radar.hardware.virtual_channel_order,
            solver="hybrid_po",
            visible_face_counts=human_cube.metadata.visible_face_counts,
            effective_visible_area_fraction=(human_cube.metadata.effective_visible_area_fraction),
            runtime_per_chirp_s=human_cube.metadata.runtime_per_chirp_s,
            runtime_profile_s=runtime_profile_s,
            runtime_profile_counts=runtime_profile_counts,
            incremental_recomputed_face_counts=(
                human_cube.metadata.incremental_recomputed_face_counts
            ),
            incremental_phase_updated_face_counts=(
                human_cube.metadata.incremental_phase_updated_face_counts
            ),
            incremental_full_refresh=human_cube.metadata.incremental_full_refresh,
            incremental_visibility_refresh=(human_cube.metadata.incremental_visibility_refresh),
            static_path_counts=static_path_counts,
            blocked_static_path_counts=blocked_counts,
            mean_static_path_visibility=mean_visibility,
            human_env_path_counts=human_env_counts,
            env_human_path_counts=env_human_counts,
            human_env_min_path_lengths_m=human_env_min_lengths,
            human_env_mean_path_lengths_m=human_env_mean_lengths,
            human_env_max_path_lengths_m=human_env_max_lengths,
            env_human_min_path_lengths_m=env_human_min_lengths,
            env_human_mean_path_lengths_m=env_human_mean_lengths,
            env_human_max_path_lengths_m=env_human_max_lengths,
            po_calibration=po_calibration,
        )
        return RadarCube(
            adc=adc,
            times=times,
            metadata=metadata,
            components={
                "human_po": human_po_adc.copy(),
                "static_environment_unblocked": static_unblocked_adc,
                "static_environment_blocked": static_blocked_adc,
                "human_env": human_env_adc,
                "env_human": env_human_adc,
                "static_path_visibility_weights": path_visibility_weights,
                **rt_baseline_components,
            },
        )

    def _run_rt_baseline_for_calibration(
        self,
        scene,
        radar: RadarSensor,
        targets: list[MeshTarget],
        *,
        num_frames: int,
        rt_refresh_frames: int = 1,
        calibration_reference_component: POCalibrationReference | None = None,
    ) -> dict:
        """Runs a full-scene RT reference and extracts first-frame human paths."""

        if scene is None or not targets:
            return {}

        from .hybrid_pipeline import run_rt_baseline

        saved = (self.solver, self.rt_refresh_frames, self.rt_refresh_on_displacement)
        self.solver = "rt"
        self.rt_refresh_frames = validate_frames(rt_refresh_frames, name="rt_refresh_frames")
        self.rt_refresh_on_displacement = None
        retrace_period_chirps = self.rt_retrace_period_chirps(radar.fmcw)
        try:
            result = run_rt_baseline(
                scene=scene,
                radar=radar,
                targets=targets,
                simulator=self,
                num_frames=num_frames,
                calibration_reference_component=calibration_reference_component,
            )
        finally:
            self.solver, self.rt_refresh_frames, self.rt_refresh_on_displacement = saved
        return {
            "cube": result.cube,
            "human_reference_adc": result.human_reference_adc,
            "reference_adc_power": result.reference_adc_power,
            "human_reference_path_counts": result.human_reference_path_counts,
            "human_touch_path_counts": result.human_touch_path_counts,
            "one_human_touch_path_counts": result.one_human_touch_path_counts,
            "single_human_only_path_counts": result.single_human_only_path_counts,
            "human_env_coupled_path_counts": result.human_env_coupled_path_counts,
            "human_multi_touch_path_counts": result.human_multi_touch_path_counts,
            "path_depth_histogram": result.path_depth_histogram,
            "total_path_power": result.total_path_power,
            "human_touch_path_power": result.human_touch_path_power,
            "one_human_touch_path_power": result.one_human_touch_path_power,
            "single_human_only_path_power": result.single_human_only_path_power,
            "human_env_coupled_path_power": result.human_env_coupled_path_power,
            "human_multi_touch_path_power": result.human_multi_touch_path_power,
            "retrace_count": result.retrace_count,
            "rt_refresh_frames": int(rt_refresh_frames),
            "retrace_period_chirps": int(retrace_period_chirps),
            "coherent_path_update": bool(
                result.cube.metadata.events
                and result.cube.metadata.events[0].get("coherent_path_update", False)
            ),
        }

    @staticmethod
    def _po_amplitude_calibration(
        *,
        rt_reference_adc: np.ndarray,
        po_reference_adc: np.ndarray,
        config: POCalibrationConfig,
    ) -> tuple[float, dict]:
        """Fits the scalar PO amplitude gain from RT and PO reference ADCs."""

        return fit_first_pose_adc_calibration(
            rt_reference_adc=rt_reference_adc,
            po_reference_adc=po_reference_adc,
            config=config,
        )

    def _calibrate_po_waveforms(
        self, radar, targets, waveforms, *, config, rt_reference_scene, events, add_runtime_profile
    ):
        """Fit and apply the configured RT-to-PO amplitude calibration."""
        adc, times, phase_center_channel = waveforms
        fmcw = radar.fmcw
        num_frames, num_chirps = times.shape
        adc_complex_dtype = adc.dtype
        po_calibration = {"mode": config.po_calibration.mode}
        human_po_gain = 1.0
        components = None
        if rt_calibration_enabled(config.po_calibration):
            if rt_reference_scene is None:
                po_calibration = {
                    "mode": config.po_calibration.mode,
                    "gain_type": config.po_calibration.gain_type,
                    "applied": False,
                    "human_po_amplitude_gain": 1.0,
                    "human_po_amplitude_gain_db": 0.0,
                    "reference_frame": 0,
                    "warnings": ["rt_reference_scene_missing"],
                }
            else:
                calibration_num_frames = (
                    num_frames if config.po_calibration.mode == "rt_sequence" else 1
                )
                if self.progress:
                    print(
                        "[po] starting full-scene RT baseline "
                        f"{calibration_num_frames} frames x "
                        f"{fmcw.num_chirps_per_frame} chirps "
                        f"for {config.po_calibration.mode} calibration",
                        flush=True,
                    )
                baseline_t0 = time.perf_counter()
                rt_baseline = self._run_rt_baseline_for_calibration(
                    rt_reference_scene,
                    radar,
                    targets,
                    num_frames=calibration_num_frames,
                    calibration_reference_component=(
                        config.po_calibration.reference_component
                        if config.po_calibration.mode == "rt_sequence"
                        else None
                    ),
                    rt_refresh_frames=config.po_calibration.rt_refresh_frames,
                )
                baseline_elapsed_s = time.perf_counter() - baseline_t0
                add_runtime_profile(
                    "po_calibration_rt_baseline",
                    baseline_elapsed_s,
                )
                if self.progress:
                    print(
                        "[po] finished full-scene RT baseline "
                        f"elapsed={baseline_elapsed_s:.1f}s "
                        f"retraces={int(rt_baseline.get('retrace_count', 0))}",
                        flush=True,
                    )
                fit_t0 = time.perf_counter()
                if config.po_calibration.mode == "rt_sequence":
                    gain, po_calibration = fit_sequence_adc_calibration(
                        rt_baseline=rt_baseline,
                        po_times=times,
                        po_adc=adc,
                        config=config.po_calibration,
                    )
                else:
                    gain, po_calibration = self._po_amplitude_calibration(
                        rt_reference_adc=rt_baseline["human_reference_adc"],
                        po_reference_adc=adc[0],
                        config=config.po_calibration,
                    )
                    po_calibration["reference_chirps"] = int(num_chirps)
                po_calibration["rt_reference_human_path_count_min"] = int(
                    np.min(rt_baseline["human_reference_path_counts"])
                )
                po_calibration["rt_reference_human_path_count_mean"] = float(
                    np.mean(rt_baseline["human_reference_path_counts"])
                )
                po_calibration["rt_reference_human_path_count_max"] = int(
                    np.max(rt_baseline["human_reference_path_counts"])
                )
                po_calibration["rt_baseline_retrace_count"] = int(
                    rt_baseline.get("retrace_count", 0)
                )
                po_calibration["rt_baseline_num_frames"] = int(calibration_num_frames)
                po_calibration.update(rt_baseline_schedule_metadata(rt_baseline))
                add_runtime_profile(
                    "po_calibration_gain_fit",
                    time.perf_counter() - fit_t0,
                )
                human_po_gain = gain
                human_po_gain_map = calibration_gain_array(
                    po_calibration,
                    times.shape,
                    fallback=gain,
                )
                adc = (adc * human_po_gain_map[..., None, None]).astype(
                    adc_complex_dtype,
                    copy=False,
                )
                phase_center_channel = phase_center_channel * human_po_gain_map[..., None]
                if config.po_calibration.keep_rt_baseline:
                    components = {
                        "rt_baseline": rt_baseline["cube"].adc,
                        "rt_baseline_full_scene": rt_baseline["cube"].adc,
                        "rt_baseline_one_human_first_frame": (
                            rt_baseline["human_reference_adc"][None, ...]
                        ),
                    }
            events.append({"reason": "po_calibration", **po_calibration})

        return adc, phase_center_channel, po_calibration, human_po_gain, components

    def _run_po(
        self,
        radar: RadarSensor,
        targets: list[MeshTarget],
        *,
        num_frames: int,
        static_visibility_scene=None,
        rt_reference_scene=None,
        po_config: POConfig | None = None,
    ) -> RadarCube:
        """Runs the PO solver on the human targets without tracing the environment."""
        return _POFrameRun(
            self,
            radar=radar,
            targets=targets,
            num_frames=num_frames,
            static_visibility_scene=static_visibility_scene,
            rt_reference_scene=rt_reference_scene,
            po_config=po_config,
        ).run()

    @staticmethod
    def _apply_po_hardware_gain(
        result: POChannelResult,
        radar: RadarSensor,
    ) -> POChannelResult:
        """Applies radar hardware gains to a PO result."""

        azimuth, elevation = radar.local_azimuth_elevation(result.face_centroids)
        gains = radar.hardware.channel_gain(
            azimuth_deg=azimuth,
            elevation_deg=elevation,
            frequency_hz=radar.fmcw.carrier_frequency,
        )
        coefficients = result.coefficients * gains
        phase_center_gains = radar.hardware.channel_gain(
            azimuth_deg=azimuth,
            elevation_deg=elevation,
            frequency_hz=radar.fmcw.carrier_frequency,
            include_phase=False,
        )
        phase_center_contributions = result.phase_center_contributions * np.mean(
            phase_center_gains, axis=0
        )
        return replace(
            result,
            coefficients=coefficients.astype(result.coefficients.dtype, copy=False),
            delay_response=(
                None
                if result.delay_response is None
                else result.delay_response.with_channel_gains(
                    result.delay_response.channel_gains * gains
                )
            ),
            phase_center_contributions=phase_center_contributions.astype(
                result.phase_center_contributions.dtype,
                copy=False,
            ),
            phase_center_channel=complex(np.sum(phase_center_contributions)),
        )

    def _po_incremental_state_from_result(
        self,
        result: POChannelResult,
        *,
        vertices: np.ndarray,
        faces: np.ndarray,
        tx_positions: np.ndarray,
        rx_positions: np.ndarray,
        virtual_channel_tx_indices: np.ndarray | None = None,
        virtual_channel_rx_indices: np.ndarray | None = None,
        virtual_channel_order: str = "tx_major",
    ) -> _POIncrementalState:
        """Initializes incremental PO state from an exact PO channel result."""

        path_lengths, spreads = self._face_path_geometry(
            result.face_centroids,
            tx_positions,
            rx_positions,
            virtual_channel_tx_indices=virtual_channel_tx_indices,
            virtual_channel_rx_indices=virtual_channel_rx_indices,
            virtual_channel_order=virtual_channel_order,
        )
        tx_phase_center = np.mean(np.asarray(tx_positions, dtype=float), axis=0)
        rx_phase_center = np.mean(np.asarray(rx_positions, dtype=float), axis=0)
        phase_lengths, phase_spreads = self._face_path_geometry(
            result.face_centroids,
            tx_phase_center[None, :],
            rx_phase_center[None, :],
        )
        return _POIncrementalState(
            face_indices=result.face_indices.copy(),
            delay_response=result.delay_response,
            coefficients=result.coefficients.copy(),
            delays_s=result.delays_s.copy(),
            phase_center_contributions=(result.phase_center_contributions.copy()),
            face_centroids=result.face_centroids.copy(),
            face_normals=result.face_normals.copy(),
            face_areas=result.face_areas.copy(),
            exact_vertex_path_offsets_m=vertex_path_offsets(
                vertices[faces[result.face_indices]], tx_phase_center, rx_phase_center
            ),
            exact_face_areas=result.face_areas.copy(),
            exact_face_normals=result.face_normals.copy(),
            path_lengths_m=path_lengths,
            spread_products_m2=spreads,
            phase_center_lengths_m=phase_lengths.reshape(-1),
            phase_center_spreads_m2=phase_spreads.reshape(-1),
            raw_fractional_visibility=result.raw_fractional_visibility.copy(),
            latched_fractional_visibility=(result.latched_fractional_visibility.copy()),
            temporal_fade_weights=result.temporal_fade_weights.copy(),
            effective_weights=result.effective_weights.copy(),
            visible_sample_counts=result.visible_sample_counts.copy(),
            total_area=result.total_area,
            effective_visible_area_fraction=(result.effective_visible_area_fraction),
            visibility_state=result.visibility_state,
            sample_count_per_face=result.sample_count_per_face,
            far_field_max_edge_m=result.far_field_max_edge_m,
            far_field_nearest_distance_m=result.far_field_nearest_distance_m,
        )

    def _update_po_incremental_state(
        self,
        *,
        state: _POIncrementalState,
        vertices: np.ndarray,
        channel: _POChannelInputs,
        visibility_fade_steps: int,
        visibility_fade_advance: bool,
        refresh_visibility: bool,
        runtime_profile_s: dict[str, float] | None = None,
        runtime_profile_counts: dict[str, int] | None = None,
    ) -> tuple[POChannelResult, _POIncrementalState, int]:
        """Update cached PO state and exactly recompute changed faces."""
        return _POIncrementalUpdate(
            self,
            state=state,
            vertices=vertices,
            channel=channel,
            visibility_fade_steps=visibility_fade_steps,
            visibility_fade_advance=visibility_fade_advance,
            refresh_visibility=refresh_visibility,
            runtime_profile_s=runtime_profile_s,
            runtime_profile_counts=runtime_profile_counts,
        ).run()

    @staticmethod
    def _exact_po_parent_face_contributions(
        *,
        po_integration_mode: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        centroids: np.ndarray,
        normals: np.ndarray,
        areas: np.ndarray,
        recompute_faces: np.ndarray,
        effective_weights: np.ndarray,
        tx_positions: np.ndarray,
        rx_positions: np.ndarray,
        frequency_hz: float,
        material,
        po_quadrature_phase_span_scale_rad: float,
        po_curvature_error_rad: float,
        po_quadrature_max_refinement_depth: int,
        po_quadrature_max_subfaces_per_parent: int,
        far_field_max_edge_m: float | None,
        far_field_nearest_distance_m: float | None,
        progress_callback,
        compute_backend: ComputeBackend,
        compute_precision: ComputePrecision,
        polarization_axis: np.ndarray | None = None,
        delay_response: PODelayResponse | None = None,
        delay_path_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float | None, float | None]:
        """Exactly recomputes parent-face PO contributions for one mode."""

        recompute_faces = np.asarray(recompute_faces, dtype=np.int64).reshape(-1)
        if recompute_faces.size == 0:
            return (
                np.zeros((0,), dtype=np.complex128),
                far_field_max_edge_m,
                far_field_nearest_distance_m,
            )
        order = np.argsort(recompute_faces)
        sorted_faces = recompute_faces[order]
        po_material = _coerce_material(material)
        tx_phase_center = np.mean(np.asarray(tx_positions, dtype=float), axis=0)
        rx_phase_center = np.mean(np.asarray(rx_positions, dtype=float), axis=0)

        if po_integration_mode == "face_centroid":
            sorted_contrib, _ = vector_facet_face_contributions(
                face_indices=sorted_faces,
                centroids=centroids,
                normals=normals,
                areas=areas,
                tx_position=tx_phase_center,
                rx_position=rx_phase_center,
                frequency_hz=frequency_hz,
                material=po_material,
                weights=effective_weights[sorted_faces],
                backend=compute_backend,
                precision=compute_precision,
                polarization_axis=polarization_axis,
            )
        elif po_integration_mode == "parent_face_quadrature":
            (
                quadrature_centroids,
                quadrature_normals,
                quadrature_areas,
                quadrature_parent_indices,
            ) = parent_face_quadrature_geometry(
                vertices=vertices,
                faces=faces,
                selected_parent_indices=sorted_faces,
                tx_positions=tx_positions,
                rx_positions=rx_positions,
                frequency_hz=frequency_hz,
                phase_span_scale_rad=po_quadrature_phase_span_scale_rad,
                max_refinement_depth=po_quadrature_max_refinement_depth,
                max_subfaces_per_parent=po_quadrature_max_subfaces_per_parent,
                progress_callback=progress_callback,
            )
            sorted_contrib = _aggregate_parent_face_quadrature_contributions(
                selected_parent_indices=sorted_faces,
                integration_centroids=quadrature_centroids,
                integration_normals=quadrature_normals,
                integration_areas=quadrature_areas,
                path_face_indices=quadrature_parent_indices,
                effective_weights=effective_weights,
                tx_phase_center=tx_phase_center,
                rx_phase_center=rx_phase_center,
                frequency_hz=frequency_hz,
                material=po_material,
                backend=compute_backend,
                precision=compute_precision,
                progress_callback=progress_callback,
                polarization_axis=polarization_axis,
                delay_response=delay_response,
                delay_path_indices=(
                    None if delay_path_indices is None else delay_path_indices[order]
                ),
            )
        elif po_integration_mode in (
            "parent_face_adaptive_analytic",
            "parent_face_far_field_analytic",
        ):
            geometry_kwargs = dict(
                vertices=vertices,
                faces=faces,
                selected_parent_indices=sorted_faces,
                tx_positions=tx_positions,
                rx_positions=rx_positions,
                frequency_hz=frequency_hz,
                max_refinement_depth=po_quadrature_max_refinement_depth,
                max_subfaces_per_parent=po_quadrature_max_subfaces_per_parent,
                progress_callback=progress_callback,
            )
            if po_integration_mode == "parent_face_adaptive_analytic":
                (
                    far_field_centroids,
                    far_field_normals,
                    far_field_areas,
                    far_field_parent_indices,
                    far_field_triangles,
                ) = parent_face_adaptive_analytic_geometry(
                    **geometry_kwargs,
                    phase_span_scale_rad=po_quadrature_phase_span_scale_rad,
                    curvature_error_rad=po_curvature_error_rad,
                )
            else:
                (
                    far_field_centroids,
                    far_field_normals,
                    far_field_areas,
                    far_field_parent_indices,
                    far_field_triangles,
                    far_field_max_edge_m,
                    far_field_nearest_distance_m,
                ) = parent_face_far_field_analytic_geometry(
                    **geometry_kwargs,
                    far_field_max_edge_m=far_field_max_edge_m,
                    far_field_nearest_distance_m=far_field_nearest_distance_m,
                )
            sorted_contrib = _aggregate_parent_face_far_field_analytic_contributions(
                selected_parent_indices=sorted_faces,
                integration_triangles=far_field_triangles,
                integration_centroids=far_field_centroids,
                integration_normals=far_field_normals,
                integration_areas=far_field_areas,
                path_face_indices=far_field_parent_indices,
                effective_weights=effective_weights,
                tx_phase_center=tx_phase_center,
                rx_phase_center=rx_phase_center,
                frequency_hz=frequency_hz,
                material=po_material,
                backend=compute_backend,
                precision=compute_precision,
                progress_callback=progress_callback,
                polarization_axis=polarization_axis,
                delay_response=delay_response,
                delay_path_indices=(
                    None if delay_path_indices is None else delay_path_indices[order]
                ),
            )
        else:
            raise ValueError(f"unsupported po_integration_mode: {po_integration_mode}")

        phase_contrib = np.empty_like(sorted_contrib)
        phase_contrib[order] = sorted_contrib
        return phase_contrib, far_field_max_edge_m, far_field_nearest_distance_m

    @staticmethod
    def _face_path_geometry(
        centroids: np.ndarray,
        tx_positions: np.ndarray,
        rx_positions: np.ndarray,
        *,
        virtual_channel_tx_indices: np.ndarray | None = None,
        virtual_channel_rx_indices: np.ndarray | None = None,
        virtual_channel_order: str = "tx_major",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Computes Tx-face-Rx path lengths and spreading products per channel."""

        centroids = np.asarray(centroids, dtype=float)
        tx_positions = np.asarray(tx_positions, dtype=float)
        rx_positions = np.asarray(rx_positions, dtype=float)
        num_tx = tx_positions.shape[0]
        num_rx = rx_positions.shape[0]
        vc_tx, vc_rx = resolve_virtual_channel_pairs(
            int(num_tx),
            int(num_rx),
            order=virtual_channel_order,
            tx_indices=virtual_channel_tx_indices,
            rx_indices=virtual_channel_rx_indices,
        )
        tx_dist = np.linalg.norm(
            tx_positions[vc_tx, None, :] - centroids[None, :, :],
            axis=2,
        )
        rx_dist = np.linalg.norm(
            rx_positions[vc_rx, None, :] - centroids[None, :, :],
            axis=2,
        )
        path_lengths = tx_dist + rx_dist
        spreads = np.maximum(tx_dist * rx_dist, 1e-18)
        return path_lengths, spreads

    @staticmethod
    def _phase_update_coefficients(
        coefficients: np.ndarray,
        *,
        old_lengths_m: np.ndarray,
        new_lengths_m: np.ndarray,
        old_spreads_m2: np.ndarray,
        new_spreads_m2: np.ndarray,
        frequency_hz: float,
    ) -> np.ndarray:
        """Updates coefficients for geometric phase and spreading changes."""

        return phase_update_coefficients(
            coefficients,
            old_lengths_m=old_lengths_m,
            new_lengths_m=new_lengths_m,
            old_spreads_m2=old_spreads_m2,
            new_spreads_m2=new_spreads_m2,
            frequency_hz=frequency_hz,
        )

    def trace_paths(
        self,
        scene,
        radar: RadarSensor,
        targets: Iterable[MeshTarget],
        *,
        time: float = 0.0,
    ):
        """Traces paths for a single scene/target state."""
        targets = list(targets)
        radar.configure_scene(scene)
        for target in targets:
            target.add_to_scene(scene)
            target.update_to_time(float(time))
        paths = self._trace(scene)
        self.last_target_face_hit_counts = _target_face_hit_counts(
            paths,
            targets,
        )
        return paths

    def trace_scene(self, scene):
        """Traces the currently configured scene without mutating targets."""

        return self._trace(scene)

    def synthesize_adc(
        self,
        a: np.ndarray,
        tau: np.ndarray,
        valid: np.ndarray,
        fmcw,
        *,
        backend: ComputeBackend | None = None,
        precision: ComputePrecision | None = None,
        delay_response: PODelayResponse | None = None,
    ) -> np.ndarray:
        """Synthesizes one chirp from path arrays and an explicit valid mask."""

        return self._synthesize_adc(
            a,
            tau,
            valid,
            fmcw,
            backend=self.adc_compute_backend if backend is None else backend,
            precision=self.adc_compute_precision if precision is None else precision,
            delay_response=delay_response,
            diagnostics=self.adc_if_filter,
        )

    def _trace(self, scene):
        """Invokes the configured Sionna path solver with simulator RT settings."""

        self.check_cancelled()
        if self.path_solver is None:
            from sionna.rt import PathSolver

            self.path_solver = PathSolver()
        paths = self.path_solver(
            scene,
            max_depth=self.rt_max_depth,
            max_num_paths_per_src=self.rt_max_paths_per_source,
            samples_per_src=self.rt_samples_per_source,
            synthetic_array=True,
            # Radar returns bounce at least once. The direct Tx-to-Rx path
            # is board leakage, which HERMES does not model; it appears once
            # the Tx and Rx array centres differ.
            los=False,
            specular_reflection=True,
            diffuse_reflection=self.rt_diffuse_reflection,
            refraction=self.rt_refraction,
            diffraction=False,
            seed=self.rt_seed,
        )
        self.check_cancelled()
        return paths

    def path_arrays_from_paths(
        self,
        paths,
        targets: list[MeshTarget],
        *,
        virtual_channel_order: str = "tx_major",
        virtual_channel_tx_indices: np.ndarray | None = None,
        virtual_channel_rx_indices: np.ndarray | None = None,
    ):
        """Extracts ADC synthesis arrays from traced Sionna paths."""

        channel_mapping = {}
        if virtual_channel_tx_indices is not None:
            channel_mapping["virtual_channel_tx_indices"] = virtual_channel_tx_indices
        if virtual_channel_rx_indices is not None:
            channel_mapping["virtual_channel_rx_indices"] = virtual_channel_rx_indices
        return self._path_arrays(
            paths,
            targets,
            virtual_channel_order=virtual_channel_order,
            **channel_mapping,
        )

    def _path_arrays(
        self,
        paths,
        targets: list[MeshTarget],
        *,
        virtual_channel_order: str = "tx_major",
        virtual_channel_tx_indices: np.ndarray | None = None,
        virtual_channel_rx_indices: np.ndarray | None = None,
    ):
        """Converts Sionna paths to channel/path coefficient arrays for ADC synthesis."""

        a, tau = paths.cir(normalize_delays=False, out_type="numpy")
        a = np.asarray(a)
        tau = np.asarray(tau)
        if a.ndim != 6:
            raise RuntimeError(
                "Expected CIR coefficients with shape [rx, rx_ant, tx, tx_ant, paths, time]"
            )
        a = a[0, :, 0, :, :, 0]  # [rx_ant, tx_ant, paths]
        tau = tau[0, 0, :]  # [paths]
        valid = tau >= 0.0

        if self.rt_human_bounces == "single" and tau.size > 0:
            valid &= self._one_target_bounce_mask(paths, targets)
        if (not self.rt_human_specular_reflection) and tau.size > 0:
            valid &= ~self._target_specular_path_mask(
                paths,
                targets,
                num_paths=tau.size,
            )

        vc_tx, vc_rx = resolve_virtual_channel_pairs(
            int(a.shape[1]),
            int(a.shape[0]),
            order=virtual_channel_order,
            tx_indices=virtual_channel_tx_indices,
            rx_indices=virtual_channel_rx_indices,
        )
        a = a[vc_rx, vc_tx, :]

        # Flatten [major, minor, paths] with the requested virtual-channel
        # ordering preserved in the leading axis.
        if a.shape[-1] == 0:
            return (
                np.zeros((vc_tx.size, 0), dtype=np.complex128),
                tau,
                valid,
            )
        return a, tau, valid

    synthesize_adc_from_paths = staticmethod(synthesize_adc_from_paths)
    _synthesize_adc = staticmethod(_synthesize_adc)
    _numpy_dtypes_for_precision = staticmethod(_numpy_dtypes_for_precision)
    numpy_dtypes_for_precision = staticmethod(numpy_dtypes_for_precision)
    _synthesize_adc_torch = staticmethod(_synthesize_adc_torch)

    single_target_only_bounce_mask = staticmethod(single_target_only_bounce_mask)
    _single_target_only_bounce_mask = staticmethod(_single_target_only_bounce_mask)
    _target_interaction_counts = staticmethod(_target_interaction_counts)
    target_object_ids = staticmethod(target_object_ids)
    _target_object_ids = staticmethod(_target_object_ids)
    target_specular_path_mask_from_arrays = staticmethod(target_specular_path_mask_from_arrays)
    _target_specular_path_mask_from_arrays = staticmethod(_target_specular_path_mask_from_arrays)
    target_specular_path_mask = staticmethod(target_specular_path_mask)
    _target_specular_path_mask = staticmethod(_target_specular_path_mask)
    drop_target_specular_bank = staticmethod(drop_target_specular_bank)
    _drop_target_specular_bank = staticmethod(_drop_target_specular_bank)
    one_target_bounce_mask_from_bank = staticmethod(one_target_bounce_mask_from_bank)
    _one_target_bounce_mask_from_bank = staticmethod(_one_target_bounce_mask_from_bank)
    one_target_bounce_mask = staticmethod(one_target_bounce_mask)
    _one_target_bounce_mask = staticmethod(_one_target_bounce_mask)
    _one_human_bounce_mask = staticmethod(_one_human_bounce_mask)
    target_interaction_counts = staticmethod(target_interaction_counts)

    @staticmethod
    def _phase_jump_diagnostics(channel: np.ndarray) -> np.ndarray:
        """Returns the per-chirp maximum absolute unwrapped phase step.

        ``channel`` may be a scalar slow-time channel with shape
        ``[num_frames, num_chirps]`` or a per-virtual-channel response with
        shape ``[num_frames, num_chirps, num_virtual_channels]``.
        """

        channel = np.asarray(channel, dtype=np.complex128)
        if channel.ndim < 2:
            raise ValueError("channel must include frame and chirp axes")
        out = np.zeros(channel.shape[:2], dtype=float)
        slow_time = channel.reshape((-1,) + channel.shape[2:])
        if slow_time.shape[0] <= 1:
            return out
        phase = np.angle(slow_time)
        unwrapped = np.unwrap(phase, axis=0)
        jumps = np.abs(np.diff(unwrapped, axis=0)).reshape((slow_time.shape[0] - 1, -1))
        out.reshape(-1)[1:] = np.max(jumps, axis=1)
        return out

    _coherent_channel = staticmethod(coherent_channel)


class _RTPathBankRun:
    """State for one solver pass, with explicit geometry, synthesis and recording stages."""

    def __init__(self, engine, scene, radar, targets, num_frames, reference_collector):
        self.engine = engine
        self.scene = scene
        self.radar = radar
        self.targets = targets
        self.num_frames = num_frames
        self.reference_collector = reference_collector

    def add_runtime_profile(self, key: str, elapsed_s: float):
        """Accumulates elapsed time and count for one RT-coherence stage."""

        self.runtime_profile_s[key] = self.runtime_profile_s.get(key, 0.0) + float(elapsed_s)
        self.runtime_profile_counts[key] = self.runtime_profile_counts.get(key, 0) + 1

    def _stage_allocate(self):
        """Allocate for this solver pass."""
        self.radar.configure_scene(self.scene)
        for self.target in self.targets:
            self.target.add_to_scene(self.scene)

        self.fmcw = self.radar.fmcw
        self.num_chirps = self.fmcw.num_chirps_per_frame
        self.num_adc = self.fmcw.num_adc_samples
        self.num_vc = self.radar.hardware.num_virtual_channels
        self._, self.adc_complex_dtype = self.engine._numpy_dtypes_for_precision(
            self.engine.adc_compute_precision
        )
        self.adc = np.zeros(
            (self.num_frames, self.num_chirps, self.num_adc, self.num_vc),
            dtype=self.adc_complex_dtype,
        )
        self.times = (
            np.arange(self.num_frames, dtype=float)[:, None] * self.fmcw.frame_period
            + np.arange(self.num_chirps, dtype=float)[None, :] * self.fmcw.chirp_repetition_time
        )
        self.path_counts = np.zeros((self.num_frames, self.num_chirps), dtype=np.int64)
        self.max_displacements = np.zeros((self.num_frames, self.num_chirps), dtype=float)
        self.coherent_channel = np.zeros(
            (self.num_frames, self.num_chirps, self.num_vc), dtype=np.complex128
        )
        self.transition_persistent_counts = np.zeros_like(self.path_counts)
        self.transition_birth_counts = np.zeros_like(self.path_counts)
        self.transition_death_counts = np.zeros_like(self.path_counts)
        self.transition_alpha = np.zeros((self.num_frames, self.num_chirps), dtype=float)
        self.transition_old_weights = np.ones((self.num_frames, self.num_chirps), dtype=float)
        self.transition_new_weights = np.zeros((self.num_frames, self.num_chirps), dtype=float)
        self.transition_crossfade_enabled = self.engine._rt_transition_crossfade_enabled()

        self.runtime_profile_s = {
            "rt_path_bank_trace": 0.0,
            "rt_path_bank_extract": 0.0,
            "rt_path_bank_transition_match": 0.0,
            "rt_path_bank_update": 0.0,
            "rt_path_bank_visibility": 0.0,
            "rt_path_bank_adc_synthesis": 0.0,
        }
        self.runtime_profile_counts = dict.fromkeys(self.runtime_profile_s, 0)

    def _stage_trace_and_visibility(self):
        """Trace and visibility for this solver pass."""
        self.plan = self.engine._plan_rt_retraces(
            self.radar,
            self.targets,
            num_frames=self.num_frames,
            times=self.times,
            max_displacements=self.max_displacements,
        )
        self.events = self.plan.events
        self.event_flat_indices = self.plan.flat_indices
        self.event_anchor_vertices = self.plan.anchor_vertices
        self.event_banks, self.event_counts = self.engine._trace_rt_banks(
            self.scene,
            self.radar,
            self.targets,
            self.plan,
            num_frames=self.num_frames,
            reference_collector=self.reference_collector,
            add_runtime_profile=self.add_runtime_profile,
        )
        self.transitions = self.engine._match_rt_bank_transitions(
            self.radar,
            self.plan,
            self.event_banks,
            self.add_runtime_profile,
        )
        self.target_ids = self.engine._target_object_ids(self.targets)

        self.bank_coupling_masks = [
            self.engine._one_target_bounce_mask_from_bank(bank, self.target_ids)
            for bank in self.event_banks
        ]

        self.frame_retrace_counts = np.zeros(self.num_frames, dtype=np.int64)
        for self.event in self.events:
            self.frame_retrace_counts[int(self.event["frame"])] += 1

        # Between retraces, known paths are re-tested against the moving body
        # on the RT visibility cadence; a retrace re-discovers visibility.
        self.visibility_per_chirp = self.engine.rt_visibility_refresh_frames == 0
        self.visibility_refresh_chirps = refresh_period_chirps(
            self.engine.rt_visibility_refresh_frames,
            self.num_chirps,
        )
        self.visibility_fade_steps = (
            1 if self.visibility_per_chirp else self.engine.rt_visibility_fade_frames + 1
        )
        self.rt_visibility_refresh = np.zeros((self.num_frames, self.num_chirps), dtype=bool)
        self.rt_blocked_path_counts = np.zeros((self.num_frames, self.num_chirps), dtype=np.int64)
        self.path_visibility_scene = None
        self.visibility_bank_index = -1
        self.path_raw_visibility = None
        self.path_visibility_state = None
        self.path_visibility_weights = None

    def _stage_synthesize(self):
        """Synthesize for this solver pass."""
        self.frame_start_s = time.perf_counter()
        for self.frame in range(self.num_frames):
            self.engine.report_frame_progress(
                mode="rt",
                frame=self.frame,
                num_frames=self.num_frames,
            )
            if self.frame > 0:
                self.engine._log_frame_progress(
                    mode="rt",
                    frame=self.frame - 1,
                    num_frames=self.num_frames,
                    frame_start_s=self.frame_start_s,
                    extra=f"retraces={int(self.frame_retrace_counts[self.frame - 1])}",
                )
            for self.chirp in range(self.num_chirps):
                self._synthesize_chirp()

        self.engine._log_frame_progress(
            mode="rt",
            frame=self.num_frames - 1,
            num_frames=self.num_frames,
            frame_start_s=self.frame_start_s,
            extra=f"retraces={int(self.frame_retrace_counts[self.num_frames - 1])}",
        )

    def _stage_result(self):
        """Result for this solver pass."""
        self.phase_jumps = self.engine._phase_jump_diagnostics(self.coherent_channel)
        self.metadata = SensingMetadata(
            adc_if_filter=dict(self.engine.adc_if_filter),
            events=self.events,
            path_counts=self.path_counts,
            max_displacements=self.max_displacements,
            virtual_channel_order=self.radar.hardware.virtual_channel_order,
            solver="rt",
            max_unwrapped_phase_jump=self.phase_jumps,
            runtime_profile_s=self.runtime_profile_s,
            runtime_profile_counts=self.runtime_profile_counts,
            rt_transition_persistent_path_counts=self.transition_persistent_counts,
            rt_transition_birth_path_counts=self.transition_birth_counts,
            rt_transition_death_path_counts=self.transition_death_counts,
            rt_transition_alpha=self.transition_alpha,
            rt_transition_old_weights=self.transition_old_weights,
            rt_transition_new_weights=self.transition_new_weights,
            rt_visibility_refresh=self.rt_visibility_refresh,
            rt_blocked_path_counts=self.rt_blocked_path_counts,
        )
        return RadarCube(adc=self.adc, times=self.times, metadata=self.metadata)

    def _stage_sample_pose(self):
        """Sample pose for this solver pass."""
        self.engine.check_cancelled()
        self.flat_index = self.frame * self.num_chirps + self.chirp
        self.event_index = int(
            np.searchsorted(self.event_flat_indices, self.flat_index, side="right") - 1
        )
        if self.event_index < 0:
            raise RuntimeError("coherent RT path bank was not initialized")
        self.current_bank = self.event_banks[self.event_index]
        self.event_chirp = self.flat_index == self.event_flat_indices[self.event_index]
        if self.event_chirp:
            self.current_vertices = self.event_anchor_vertices[self.event_index]
        else:
            self.current_vertices = [
                target.mesh_sequence.vertices_at(self.times[self.frame, self.chirp])
                for target in self.targets
            ]
        if not self.event_chirp:
            self.anchor_vertices = self.event_anchor_vertices[self.event_index]
            self.max_displacements[self.frame, self.chirp] = max(
                (
                    float(
                        np.max(
                            np.linalg.norm(
                                vertices - reference,
                                axis=1,
                            )
                        )
                    )
                    for vertices, reference in zip(
                        self.current_vertices,
                        self.anchor_vertices,
                    )
                ),
                default=0.0,
            )

    def _stage_visibility(self):
        """Visibility for this solver pass."""
        self.update_t0 = time.perf_counter()
        self.current_update = update_coherent_rt_path_bank(
            self.current_bank,
            self.current_vertices,
            backend=self.engine.compute_backend,
            precision=self.engine.compute_precision,
        )
        if self.event_index != self.visibility_bank_index:
            self.visibility_bank_index = self.event_index
            self.path_raw_visibility = self.current_bank.valid.astype(np.float64)
            self.path_visibility_state = PathVisibilityState(
                was_visible=self.current_bank.valid.copy(),
                fade_weights=self.path_raw_visibility.copy(),
                latched_visibility=self.path_raw_visibility.copy(),
            )
            self.path_visibility_weights = self.path_raw_visibility.copy()
        elif self.targets:
            self.refresh_due = self.flat_index % self.visibility_refresh_chirps == 0
            if self.refresh_due:
                self.visibility_t0 = time.perf_counter()
                if self.path_visibility_scene is None:
                    self.path_visibility_scene = self.engine._human_visibility_scene(
                        self.targets, self.current_vertices
                    )
                else:
                    self.path_visibility_scene.update_vertices(
                        np.concatenate(self.current_vertices, axis=0)
                    )
                self.starts, self.ends, self.owners, self.start_margins, self.end_margins = (
                    coherent_path_segments(self.current_bank, self.current_vertices)
                )
                self.blocked = blocked_segments(
                    self.starts,
                    self.ends,
                    self.path_visibility_scene,
                    start_margins_m=self.start_margins,
                    end_margins_m=self.end_margins,
                )
                self.path_raw_visibility = self.current_bank.valid.astype(np.float64)
                self.path_raw_visibility[self.owners[self.blocked]] = 0.0
                self.rt_visibility_refresh[self.frame, self.chirp] = True
                self.add_runtime_profile(
                    "rt_path_bank_visibility",
                    time.perf_counter() - self.visibility_t0,
                )
            if self.refresh_due or self.visibility_per_chirp or self.chirp == 0:
                self.fade, self.latched = update_path_visibility_state(
                    self.path_raw_visibility,
                    state=self.path_visibility_state,
                    fade_steps=self.visibility_fade_steps,
                )
                self.path_visibility_state = PathVisibilityState(
                    was_visible=self.path_raw_visibility > 0.0,
                    fade_weights=self.fade,
                    latched_visibility=self.latched,
                )
                self.path_visibility_weights = self.latched * self.fade
        self.rt_blocked_path_counts[self.frame, self.chirp] = int(
            np.count_nonzero(self.current_bank.valid & ~(self.path_visibility_weights > 0.0))
        )
        if np.any(self.path_visibility_weights[self.current_bank.valid] < 1.0):
            self.current_update = replace(
                self.current_update,
                coefficients=(
                    self.current_update.coefficients * self.path_visibility_weights[None, :]
                ),
                valid=self.current_update.valid & (self.path_visibility_weights > 0.0),
            )

    def _stage_transition(self):
        """Transition for this solver pass."""
        self.next_update = None
        self.transition = None
        self.interval_chirps = 1
        self.interval_offset = 0
        if (
            self.event_index < len(self.event_banks) - 1
            and self.transition_crossfade_enabled
            and self.transitions[self.event_index] is not None
        ):
            self.transition = self.transitions[self.event_index]
            self.interval_chirps = int(
                self.event_flat_indices[self.event_index + 1]
                - self.event_flat_indices[self.event_index]
            )
            self.interval_offset = int(self.flat_index - self.event_flat_indices[self.event_index])
            self.next_update = update_coherent_rt_path_bank(
                self.event_banks[self.event_index + 1],
                self.current_vertices,
                backend=self.engine.compute_backend,
                precision=self.engine.compute_precision,
            )
        self.synth = synthesize_coherent_rt_transition(
            self.current_update,
            self.next_update,
            self.transition,
            interval_chirps=self.interval_chirps,
            interval_offset=self.interval_offset,
        )
        self.add_runtime_profile(
            "rt_path_bank_update",
            time.perf_counter() - self.update_t0,
        )

    def _stage_filter_paths(self):
        """Filter paths for this solver pass."""
        self.valid = self.synth.valid.copy()
        if self.engine.rt_human_bounces == "single" and self.valid.size > 0:
            self.source_masks = np.zeros(self.valid.shape, dtype=bool)
            self.next_source = self.synth.source_is_new
            self.current_source = ~self.next_source
            if np.any(self.current_source):
                self.source_masks[self.current_source] = self.bank_coupling_masks[
                    self.event_index
                ][self.synth.source_indices[self.current_source]]
            if np.any(self.next_source) and self.event_index + 1 < len(self.bank_coupling_masks):
                self.source_masks[self.next_source] = self.bank_coupling_masks[
                    self.event_index + 1
                ][self.synth.source_indices[self.next_source]]
            self.valid &= self.source_masks

        if self.reference_collector is not None:
            self.target_counts = np.zeros(self.valid.shape, dtype=np.int64)
            self.total_counts = np.zeros(self.valid.shape, dtype=np.int64)
            for self.is_new, self.bank_index in (
                (False, self.event_index),
                (True, self.event_index + 1),
            ):
                self.selected = self.synth.source_is_new == self.is_new
                if np.any(self.selected):
                    self.counts, self.totals = self.event_counts[self.bank_index]
                    self.indices = self.synth.source_indices[self.selected]
                    self.target_counts[self.selected], self.total_counts[self.selected] = (
                        self.counts[self.indices],
                        self.totals[self.indices],
                    )
            self.reference_collector.observe(
                self.frame,
                self.chirp,
                self.synth.coefficients,
                self.synth.delays_s,
                self.valid,
                self.target_counts,
                self.total_counts,
            )
        self.path_counts[self.frame, self.chirp] = int(np.count_nonzero(self.valid))

    def _stage_diagnostics(self):
        """Diagnostics for this solver pass."""
        self.diagnostic_transition = (
            self.transitions[self.event_index]
            if self.event_index < len(self.transitions)
            else None
        )
        if self.diagnostic_transition is None:
            self.transition_persistent_counts[self.frame, self.chirp] = int(
                np.count_nonzero(self.current_update.valid)
            )
            self.transition_death_counts[self.frame, self.chirp] = 0
            self.transition_birth_counts[self.frame, self.chirp] = 0
        else:
            self.transition_persistent_counts[self.frame, self.chirp] = int(
                self.diagnostic_transition.matched_old_indices.size
            )
            self.transition_death_counts[self.frame, self.chirp] = int(
                self.diagnostic_transition.old_only_indices.size
            )
            self.transition_birth_counts[self.frame, self.chirp] = int(
                self.diagnostic_transition.new_only_indices.size
            )
        self.transition_alpha[self.frame, self.chirp] = self.synth.alpha
        self.transition_old_weights[self.frame, self.chirp] = self.synth.old_weight
        self.transition_new_weights[self.frame, self.chirp] = self.synth.new_weight
        self.coherent_channel[self.frame, self.chirp] = self.engine._coherent_channel(
            self.synth.coefficients, self.valid
        )

    def _stage_adc(self):
        """Adc for this solver pass."""
        self.adc_t0 = time.perf_counter()
        self.adc[self.frame, self.chirp] = self.engine.synthesize_adc(
            self.synth.coefficients,
            self.synth.delays_s,
            self.valid,
            self.fmcw,
            backend=self.engine.adc_compute_backend,
            precision=self.engine.adc_compute_precision,
        )
        self.add_runtime_profile(
            "rt_path_bank_adc_synthesis",
            time.perf_counter() - self.adc_t0,
        )

    def _synthesize_chirp(self):
        self._stage_sample_pose()
        self._stage_visibility()
        self._stage_transition()
        self._stage_filter_paths()
        self._stage_diagnostics()
        self._stage_adc()

    def run(self):
        self._stage_allocate()
        self._stage_trace_and_visibility()
        self._stage_synthesize()
        return self._stage_result()


class _POFrameRun:
    """State for one solver pass, with explicit geometry, synthesis and recording stages."""

    def __init__(
        self,
        engine,
        radar,
        targets,
        num_frames,
        static_visibility_scene,
        rt_reference_scene,
        po_config,
    ):
        self.engine = engine
        self.radar = radar
        self.targets = targets
        self.num_frames = num_frames
        self.static_visibility_scene = static_visibility_scene
        self.rt_reference_scene = rt_reference_scene
        self.po_config = po_config

    def add_runtime_profile(self, key: str, elapsed_s: float):
        """Accumulates elapsed time and count for one human-PO stage."""

        self.runtime_profile_s[key] = self.runtime_profile_s.get(key, 0.0) + float(elapsed_s)
        self.runtime_profile_counts[key] = self.runtime_profile_counts.get(key, 0) + 1

    def _stage_allocate(self):
        """Allocate for this solver pass."""
        self.config = self.engine.po_config if self.po_config is None else self.po_config
        self.po_compute_backend = (
            self.engine.compute_backend
            if self.config.compute_backend is None
            else self.config.compute_backend
        )
        self.po_compute_precision = (
            self.engine.compute_precision
            if self.config.compute_precision is None
            else self.config.compute_precision
        )
        # Exact integration and visibility refreshes happen at frame starts
        # every N frames; N = 0 means every chirp.
        self.frame_chirps = int(self.radar.fmcw.num_chirps_per_frame)
        self.full_refresh_chirps = refresh_period_chirps(
            self.config.po_refresh_frames, self.frame_chirps
        )
        self.visibility_refresh_chirps = refresh_period_chirps(
            self.config.po_visibility_refresh_frames,
            self.frame_chirps,
        )
        self.visibility_per_chirp = self.config.po_visibility_refresh_frames == 0
        # A fade over F frames takes F + 1 frame-start steps; per-chirp
        # visibility is exact and needs no fade.
        self.visibility_fade_steps = (
            1 if self.visibility_per_chirp else self.config.po_visibility_fade_frames + 1
        )
        if len(self.targets) > 1:
            raise ValueError("po currently supports at most one mesh target")

        self.fmcw = self.radar.fmcw
        self.num_chirps = self.fmcw.num_chirps_per_frame
        self.num_adc = self.fmcw.num_adc_samples
        self.num_vc = self.radar.hardware.num_virtual_channels
        self._, self.adc_complex_dtype = self.engine._numpy_dtypes_for_precision(
            self.engine.adc_compute_precision
        )
        self.adc = np.zeros(
            (self.num_frames, self.num_chirps, self.num_adc, self.num_vc),
            dtype=self.adc_complex_dtype,
        )
        self.times = (
            np.arange(self.num_frames, dtype=float)[:, None] * self.fmcw.frame_period
            + np.arange(self.num_chirps, dtype=float)[None, :] * self.fmcw.chirp_repetition_time
        )
        self.path_counts = np.zeros((self.num_frames, self.num_chirps), dtype=np.int64)
        self.po_path_power = np.zeros((self.num_frames, self.num_chirps), dtype=np.float64)
        self.max_displacements = np.zeros((self.num_frames, self.num_chirps), dtype=float)
        self.visible_face_counts = np.zeros((self.num_frames, self.num_chirps), dtype=np.int64)
        self.effective_visible_area_fraction = np.zeros(
            (self.num_frames, self.num_chirps), dtype=float
        )
        self.runtime_per_chirp_s = np.zeros((self.num_frames, self.num_chirps), dtype=float)
        self.phase_center_channel = np.zeros(
            (self.num_frames, self.num_chirps), dtype=np.complex128
        )
        self.incremental_recomputed_face_counts = np.zeros(
            (self.num_frames, self.num_chirps),
            dtype=np.int64,
        )
        self.incremental_phase_updated_face_counts = np.zeros(
            (self.num_frames, self.num_chirps),
            dtype=np.int64,
        )
        self.incremental_full_refresh = np.zeros(
            (self.num_frames, self.num_chirps),
            dtype=bool,
        )
        self.incremental_visibility_refresh = np.zeros(
            (self.num_frames, self.num_chirps),
            dtype=bool,
        )
        self.runtime_profile_s = {
            "mesh_update": 0.0,
            "po_channel": 0.0,
            "po_full_rebuild": 0.0,
            "po_incremental_update": 0.0,
            "po_incremental_face_geometry": 0.0,
            "po_incremental_visibility_refresh": 0.0,
            "po_incremental_visibility_state": 0.0,
            "po_incremental_path_geometry": 0.0,
            "po_incremental_bank_merge": 0.0,
            "po_incremental_phase_update": 0.0,
            "po_incremental_drift_check": 0.0,
            "po_incremental_exact_recompute": 0.0,
            "po_incremental_state_update": 0.0,
            "adc_synthesis": 0.0,
            "po_calibration_rt_baseline": 0.0,
            "po_calibration_gain_fit": 0.0,
            "metadata": 0.0,
        }
        self.runtime_profile_counts = dict.fromkeys(self.runtime_profile_s, 0)

        self.tx_positions = self.radar.world_tx_positions()
        self.rx_positions = self.radar.world_rx_positions()
        self.events = [
            {
                "frame": 0,
                "chirp": 0,
                "time": float(self.fmcw.chirp_time(0, 0)),
                "reason": "po",
                "po_visibility_samples_per_face": self.config.po_visibility_samples_per_face,
                "po_refresh_frames": self.config.po_refresh_frames,
                "po_refresh_on_phase_error_rad": self.config.po_refresh_on_phase_error_rad,
                "po_refresh_on_normal_deg": self.config.po_refresh_on_normal_deg,
                "po_refresh_on_area_fraction": self.config.po_refresh_on_area_fraction,
                "po_visibility_refresh_frames": self.config.po_visibility_refresh_frames,
                "po_visibility_fade_frames": self.config.po_visibility_fade_frames,
                "po_integration_mode": self.config.po_integration_mode,
                "po_quadrature_phase_span_scale_rad": (
                    self.config.effective_po_phase_span_scale_rad
                ),
                "po_curvature_error_rad": self.config.po_curvature_error_rad,
                "po_quadrature_max_refinement_depth": (
                    self.config.po_quadrature_max_refinement_depth
                ),
                "po_quadrature_max_subfaces_per_parent": (
                    self.config.po_quadrature_max_subfaces_per_parent
                ),
            }
        ]

        if not self.targets:
            self.metadata = SensingMetadata(
                adc_if_filter=dict(self.engine.adc_if_filter),
                events=self.events,
                path_counts=self.path_counts,
                max_displacements=self.max_displacements,
                virtual_channel_order=self.radar.hardware.virtual_channel_order,
                solver="po",
                visible_face_counts=self.visible_face_counts,
                effective_visible_area_fraction=self.effective_visible_area_fraction,
                runtime_per_chirp_s=self.runtime_per_chirp_s,
                max_unwrapped_phase_jump=np.zeros((self.num_frames, self.num_chirps), dtype=float),
                runtime_profile_s=self.runtime_profile_s,
                runtime_profile_counts=self.runtime_profile_counts,
                incremental_recomputed_face_counts=(self.incremental_recomputed_face_counts),
                incremental_phase_updated_face_counts=(self.incremental_phase_updated_face_counts),
                incremental_full_refresh=self.incremental_full_refresh,
                incremental_visibility_refresh=self.incremental_visibility_refresh,
                total_path_power=self.po_path_power,
            )
            return RadarCube(adc=self.adc, times=self.times, metadata=self.metadata)

    def _stage_prepare_target(self):
        """Prepare target for this solver pass."""
        self.target = self.targets[0]
        self.mesh_sequence = self.target.mesh_sequence
        self.precomputed_visibility = self.config.precomputed_raw_fractional_visibility
        self.visibility_scene = None
        if self.precomputed_visibility is None or self.config.incremental:
            self.visibility_scene = HumanVisibilityScene(
                faces=self.mesh_sequence.faces,
                vertices=self.mesh_sequence.vertices[0],
                name=f"{self.target.name}-human-po",
            )
        self.channel_inputs = _POChannelInputs(
            radar=self.radar,
            target=self.target,
            config=self.config,
            tx_positions=self.tx_positions,
            rx_positions=self.rx_positions,
            visibility_scene=self.visibility_scene,
            static_visibility_scene=self.static_visibility_scene,
            compute_backend=self.po_compute_backend,
            compute_precision=self.po_compute_precision,
        )
        self.visibility_state = None
        self.incremental_state = None
        self.previous_vertices = None
        self.frame_start_s = time.perf_counter()

    def _stage_synthesize(self):
        """Synthesize for this solver pass."""
        for self.frame in range(self.num_frames):
            self.engine.report_frame_progress(
                mode="po",
                frame=self.frame,
                num_frames=self.num_frames,
            )
            if self.frame > 0:
                self.engine._log_frame_progress(
                    mode="po",
                    frame=self.frame - 1,
                    num_frames=self.num_frames,
                    frame_start_s=self.frame_start_s,
                    extra=(
                        f"mean_visible_faces="
                        f"{float(np.mean(self.visible_face_counts[self.frame - 1])):.1f}"
                    ),
                )
            for self.chirp in range(self.num_chirps):
                self._synthesize_chirp()

        self.engine._log_frame_progress(
            mode="po",
            frame=self.num_frames - 1,
            num_frames=self.num_frames,
            frame_start_s=self.frame_start_s,
            extra=(
                f"mean_visible_faces="
                f"{float(np.mean(self.visible_face_counts[self.num_frames - 1])):.1f}"
            ),
        )

    def _stage_result(self):
        """Result for this solver pass."""
        (
            self.adc,
            self.phase_center_channel,
            self.po_calibration,
            self.human_po_gain,
            self.components,
        ) = self.engine._calibrate_po_waveforms(
            self.radar,
            self.targets,
            (self.adc, self.times, self.phase_center_channel),
            config=self.config,
            rt_reference_scene=self.rt_reference_scene,
            events=self.events,
            add_runtime_profile=self.add_runtime_profile,
        )

        self.phase_jumps = self.engine._phase_jump_diagnostics(self.phase_center_channel)
        self.metadata = SensingMetadata(
            adc_if_filter=dict(self.engine.adc_if_filter),
            events=self.events,
            path_counts=self.path_counts,
            max_displacements=self.max_displacements,
            virtual_channel_order=self.radar.hardware.virtual_channel_order,
            solver="po",
            visible_face_counts=self.visible_face_counts,
            effective_visible_area_fraction=self.effective_visible_area_fraction,
            runtime_per_chirp_s=self.runtime_per_chirp_s,
            max_unwrapped_phase_jump=self.phase_jumps,
            runtime_profile_s=self.runtime_profile_s,
            runtime_profile_counts=self.runtime_profile_counts,
            incremental_recomputed_face_counts=(self.incremental_recomputed_face_counts),
            incremental_phase_updated_face_counts=(self.incremental_phase_updated_face_counts),
            incremental_full_refresh=self.incremental_full_refresh,
            incremental_visibility_refresh=self.incremental_visibility_refresh,
            total_path_power=(
                self.po_path_power
                * calibration_gain_array(
                    self.po_calibration,
                    self.times.shape,
                    fallback=self.human_po_gain,
                )
                ** 2
            ),
            po_calibration=self.po_calibration,
        )
        return RadarCube(
            adc=self.adc, times=self.times, metadata=self.metadata, components=self.components
        )

    def _stage_sample_pose(self):
        """Sample pose for this solver pass."""
        self.engine.check_cancelled()
        self.t0 = time.perf_counter()
        self.engine.report_chirp_progress(
            mode="po",
            frame=self.frame,
            num_frames=self.num_frames,
            chirp=self.chirp,
            num_chirps=self.num_chirps,
        )
        self.chirp_time = self.fmcw.chirp_time(self.frame, self.chirp)
        self.times[self.frame, self.chirp] = self.chirp_time
        self.mesh_t0 = time.perf_counter()
        self.vertices = self.mesh_sequence.vertices_at(self.chirp_time)
        if self.previous_vertices is not None:
            self.max_displacements[self.frame, self.chirp] = float(
                np.max(np.linalg.norm(self.vertices - self.previous_vertices, axis=1))
            )
        self.previous_vertices = self.vertices.copy()
        self.add_runtime_profile(
            "mesh_update",
            time.perf_counter() - self.mesh_t0,
        )

    def _stage_channel(self):
        """Channel for this solver pass."""
        self.po_t0 = time.perf_counter()
        self.chirp_index_abs = self.frame * self.num_chirps + self.chirp
        self.force_full_refresh = (
            self.incremental_state is None or self.chirp_index_abs % self.full_refresh_chirps == 0
        )
        self.visibility_refresh_due = self.chirp_index_abs % self.visibility_refresh_chirps == 0
        self.visibility_fade_advance = self.visibility_per_chirp or self.chirp == 0
        self.force_visibility_refresh = not self.force_full_refresh and self.visibility_refresh_due
        if self.force_full_refresh:
            self.full_rebuild_t0 = time.perf_counter()
            self.raw_visibility_override = self.precomputed_visibility
            self.visible_sample_counts_override = self.config.precomputed_visible_sample_counts
            if (
                self.raw_visibility_override is None
                and self.incremental_state is not None
                and not self.visibility_refresh_due
            ):
                self.raw_visibility_override = self.incremental_state.raw_fractional_visibility
                self.visible_sample_counts_override = self.incremental_state.visible_sample_counts
            self.result = po_channel(
                vertices=self.vertices,
                visibility="fractional_shadow_fade",
                visibility_state=self.visibility_state,
                visibility_fade_steps=self.visibility_fade_steps,
                visibility_fade_advance=self.visibility_fade_advance,
                precomputed_raw_fractional_visibility=self.raw_visibility_override,
                precomputed_visible_sample_counts=self.visible_sample_counts_override,
                **self.channel_inputs.channel_arguments(),
            )
            self.incremental_state = self.engine._po_incremental_state_from_result(
                self.result,
                vertices=self.vertices,
                **self.channel_inputs.geometry_arguments(),
            )
            self.incremental_recomputed_face_counts[self.frame, self.chirp] = (
                self.result.face_indices.size
            )
            self.incremental_full_refresh[self.frame, self.chirp] = True
            self.add_runtime_profile(
                "po_full_rebuild",
                time.perf_counter() - self.full_rebuild_t0,
            )
        else:
            self.incremental_update_t0 = time.perf_counter()
            self.result, self.incremental_state, self.recomputed_count = (
                self.engine._update_po_incremental_state(
                    state=self.incremental_state,
                    vertices=self.vertices,
                    channel=self.channel_inputs,
                    visibility_fade_steps=self.visibility_fade_steps,
                    visibility_fade_advance=self.visibility_fade_advance,
                    refresh_visibility=self.force_visibility_refresh,
                    runtime_profile_s=self.runtime_profile_s,
                    runtime_profile_counts=self.runtime_profile_counts,
                )
            )
            self.add_runtime_profile(
                "po_incremental_update",
                time.perf_counter() - self.incremental_update_t0,
            )
            self.incremental_recomputed_face_counts[self.frame, self.chirp] = self.recomputed_count
            self.incremental_phase_updated_face_counts[self.frame, self.chirp] = (
                self.result.face_indices.size - self.recomputed_count
            )
            self.incremental_visibility_refresh[self.frame, self.chirp] = (
                self.force_visibility_refresh
            )
        self.visibility_state = self.result.visibility_state
        self.add_runtime_profile(
            "po_channel",
            time.perf_counter() - self.po_t0,
        )
        self.result = self.engine._apply_po_hardware_gain(self.result, self.radar)

    def _stage_record(self):
        """Record for this solver pass."""
        self.valid = np.any(np.abs(self.result.coefficients) > 0.0, axis=0)
        if self.result.delay_response is not None:
            self.valid |= self.result.delay_response.valid_paths
        self.path_counts[self.frame, self.chirp] = int(np.count_nonzero(self.valid))
        self.po_path_power[self.frame, self.chirp] = po_path_power_from_coefficients(
            self.result.coefficients,
            self.valid,
        )
        self.adc_t0 = time.perf_counter()
        self.adc[self.frame, self.chirp] = self.engine.synthesize_adc(
            self.result.coefficients,
            self.result.delays_s,
            self.valid,
            self.fmcw,
            backend=self.engine.adc_compute_backend,
            precision=self.engine.adc_compute_precision,
            delay_response=self.result.delay_response,
        )
        self.add_runtime_profile(
            "adc_synthesis",
            time.perf_counter() - self.adc_t0,
        )
        self.metadata_t0 = time.perf_counter()
        self.phase_center_channel[self.frame, self.chirp] = self.result.phase_center_channel
        self.visible_face_counts[self.frame, self.chirp] = int(
            np.count_nonzero(self.result.raw_fractional_visibility > 0.0)
        )

        self.effective_visible_area_fraction[self.frame, self.chirp] = (
            self.result.effective_visible_area_fraction
        )
        self.add_runtime_profile(
            "metadata",
            time.perf_counter() - self.metadata_t0,
        )
        self.runtime_per_chirp_s[self.frame, self.chirp] = time.perf_counter() - self.t0
        self.engine._log_chirp_progress(
            mode="po",
            frame=self.frame,
            num_frames=self.num_frames,
            chirp=self.chirp,
            num_chirps=self.num_chirps,
            run_start_s=self.frame_start_s,
            chirp_runtime_s=float(self.runtime_per_chirp_s[self.frame, self.chirp]),
            extra=(
                f"integration={self.config.po_integration_mode} "
                f"po_refresh_frames={self.config.po_refresh_frames} "
                f"full_refresh={bool(self.incremental_full_refresh[self.frame, self.chirp])} "
                f"visibility_refresh={bool(self.incremental_visibility_refresh[self.frame, self.chirp])} "
                f"recomputed_faces={int(self.incremental_recomputed_face_counts[self.frame, self.chirp])} "
                f"phase_updated_faces={int(self.incremental_phase_updated_face_counts[self.frame, self.chirp])} "
                f"visible_faces={int(self.visible_face_counts[self.frame, self.chirp])} "
                f"paths={int(self.path_counts[self.frame, self.chirp])}"
            ),
        )

    def _synthesize_chirp(self):
        self._stage_sample_pose()
        self._stage_channel()
        self._stage_record()

    def run(self):
        empty_cube = self._stage_allocate()
        if empty_cube is not None:
            return empty_cube
        self._stage_prepare_target()
        self._stage_synthesize()
        return self._stage_result()


class _POIncrementalUpdate:
    """State for one solver pass, with explicit geometry, synthesis and recording stages."""

    def __init__(
        self,
        engine,
        state,
        vertices,
        channel,
        visibility_fade_steps,
        visibility_fade_advance,
        refresh_visibility,
        runtime_profile_s,
        runtime_profile_counts,
    ):
        self.engine = engine
        self.state = state
        self.vertices = vertices
        self.channel = channel
        self.visibility_fade_steps = visibility_fade_steps
        self.visibility_fade_advance = visibility_fade_advance
        self.refresh_visibility = refresh_visibility
        self.runtime_profile_s = runtime_profile_s
        self.runtime_profile_counts = runtime_profile_counts

    def add_profile(self, key: str, elapsed_s: float):
        """Accumulates optional incremental-PO profiling counters."""

        if self.runtime_profile_s is not None:
            self.runtime_profile_s[key] = self.runtime_profile_s.get(key, 0.0) + float(elapsed_s)
        if self.runtime_profile_counts is not None:
            self.runtime_profile_counts[key] = self.runtime_profile_counts.get(key, 0) + 1

    def integrate_changed(self, recompute_positions):
        # Far-field bounds are retained on this incremental update.
        phase_contrib, self.far_field_max_edge_m, self.far_field_nearest_distance_m = (
            self.engine._exact_po_parent_face_contributions(
                po_integration_mode=self.channel.config.po_integration_mode,
                vertices=self.vertices,
                faces=self.faces,
                centroids=self.centroids,
                normals=self.normals,
                areas=self.areas,
                recompute_faces=self.selected[recompute_positions],
                effective_weights=self.effective_weights,
                tx_positions=self.channel.tx_positions,
                rx_positions=self.channel.rx_positions,
                frequency_hz=self.channel.radar.fmcw.carrier_frequency,
                material=self.channel.target.material,
                po_quadrature_phase_span_scale_rad=self.channel.config.effective_po_phase_span_scale_rad,
                po_curvature_error_rad=self.channel.config.po_curvature_error_rad,
                po_quadrature_max_refinement_depth=self.channel.config.po_quadrature_max_refinement_depth,
                po_quadrature_max_subfaces_per_parent=self.channel.config.po_quadrature_max_subfaces_per_parent,
                far_field_max_edge_m=self.far_field_max_edge_m,
                far_field_nearest_distance_m=self.far_field_nearest_distance_m,
                progress_callback=self.channel.config.progress_callback,
                compute_backend=self.channel.compute_backend,
                compute_precision=self.channel.compute_precision,
                polarization_axis=rotation_matrix(self.channel.radar.orientation)[:, 2],
                delay_response=self.delay_response,
                delay_path_indices=recompute_positions,
            )
        )
        projected, _, _, _ = project_phase_center_contributions_to_virtual_channels(
            phase_contrib,
            self.selected_centroids[recompute_positions],
            self.channel.tx_positions,
            self.channel.rx_positions,
            self.channel.radar.fmcw.carrier_frequency,
            virtual_channel_tx_indices=self.channel.radar.hardware.virtual_tx_indices(),
            virtual_channel_rx_indices=self.channel.radar.hardware.virtual_rx_indices(),
            virtual_channel_order=self.channel.radar.hardware.virtual_channel_order,
            precision=self.channel.compute_precision,
        )
        return projected, phase_contrib

    def _stage_geometry_and_visibility(self):
        """Geometry and visibility for this solver pass."""
        self.visibility_scene = self.channel.visibility_scene

        self.faces = np.asarray(self.channel.faces, dtype=np.uint32)
        self.face_geometry_t0 = time.perf_counter()
        self.centroids, self.normals, self.areas = face_geometry(self.vertices, self.faces)
        self.add_profile(
            "po_incremental_face_geometry",
            time.perf_counter() - self.face_geometry_t0,
        )
        self.raw_visibility = self.state.raw_fractional_visibility
        self.visible_sample_counts = self.state.visible_sample_counts
        if self.refresh_visibility:
            self.visibility_t0 = time.perf_counter()
            if self.visibility_scene is None:
                self.visibility_scene = HumanVisibilityScene(
                    faces=self.faces,
                    vertices=self.vertices,
                    name="human-po-incremental-visibility",
                )
            else:
                self.visibility_scene.update_vertices(self.vertices)
            self.raw_visibility, self.visible_sample_counts = _fractional_face_visibility(
                vertices=self.vertices,
                faces=self.faces,
                face_normals=self.normals,
                tx_position=np.mean(np.asarray(self.channel.tx_positions, dtype=float), axis=0),
                rx_position=np.mean(np.asarray(self.channel.rx_positions, dtype=float), axis=0),
                visibility_scene=self.visibility_scene,
                static_visibility_scene=self.channel.static_visibility_scene,
                samples_per_face=self.channel.config.po_visibility_samples_per_face,
                adaptive_edge_sampling=self.channel.config.po_adaptive_visibility_sampling,
                edge_front_margin=self.channel.config.po_adaptive_visibility_edge_margin,
            )
            self.add_profile(
                "po_incremental_visibility_refresh",
                time.perf_counter() - self.visibility_t0,
            )

        self.visibility_state_t0 = time.perf_counter()
        self.fade_weights, self.latched_visibility = _update_visibility_state(
            raw_fractional_visibility=self.raw_visibility,
            state=self.state.visibility_state,
            fade_chirps=self.visibility_fade_steps,
            advance=self.visibility_fade_advance,
        )
        self.effective_weights = self.latched_visibility * self.fade_weights
        if self.refresh_visibility:
            self.selected = np.flatnonzero(self.effective_weights > 0.0)
        else:
            self.selected = self.state.face_indices[
                self.effective_weights[self.state.face_indices] > 0.0
            ]
        self.add_profile(
            "po_incremental_visibility_state",
            time.perf_counter() - self.visibility_state_t0,
        )
        self.selected_centroids = self.centroids[self.selected]
        self.selected_normals = self.normals[self.selected]
        self.selected_areas = self.areas[self.selected]

    def _stage_path_geometry(self):
        """Path geometry for this solver pass."""
        self.path_geometry_t0 = time.perf_counter()
        self.path_lengths, self.spreads = self.engine._face_path_geometry(
            self.selected_centroids,
            self.channel.tx_positions,
            self.channel.rx_positions,
            virtual_channel_tx_indices=self.channel.radar.hardware.virtual_tx_indices(),
            virtual_channel_rx_indices=self.channel.radar.hardware.virtual_rx_indices(),
            virtual_channel_order=self.channel.radar.hardware.virtual_channel_order,
        )
        self.phase_lengths, self.phase_spreads = self.engine._face_path_geometry(
            self.selected_centroids,
            np.mean(np.asarray(self.channel.tx_positions, dtype=float), axis=0)[None, :],
            np.mean(np.asarray(self.channel.rx_positions, dtype=float), axis=0)[None, :],
        )
        self.phase_lengths = self.phase_lengths.reshape(-1)
        self.phase_spreads = self.phase_spreads.reshape(-1)
        self.add_profile(
            "po_incremental_path_geometry",
            time.perf_counter() - self.path_geometry_t0,
        )

        self.delay_response = None
        if (
            self.channel.max_frequency_offset_hz is not None
            and self.channel.config.po_integration_mode != "face_centroid"
        ):
            self.delay_response = PODelayResponse.for_mesh(
                self.vertices,
                self.faces,
                self.selected.size,
                self.channel.max_frequency_offset_hz,
                compute_precision=self.channel.compute_precision,
                previous=self.state.delay_response,
                receiver_fmcw=self.channel.radar.fmcw,
                channel_delay_margin_s=(
                    np.max(
                        np.linalg.norm(
                            self.channel.tx_positions - np.mean(self.channel.tx_positions, axis=0),
                            axis=1,
                        )
                    )
                    + np.max(
                        np.linalg.norm(
                            self.channel.rx_positions - np.mean(self.channel.rx_positions, axis=0),
                            axis=1,
                        )
                    )
                )
                / c,
            )
            self.delay_response.set_reference(
                np.arange(self.selected.size),
                self.selected_centroids,
                np.mean(self.channel.tx_positions, axis=0),
                np.mean(self.channel.rx_positions, axis=0),
            )

        self.far_field_max_edge_m = self.state.far_field_max_edge_m
        self.far_field_nearest_distance_m = self.state.far_field_nearest_distance_m
        if self.channel.config.po_integration_mode == "parent_face_far_field_analytic" and (
            self.refresh_visibility or self.far_field_max_edge_m is None
        ):
            self.far_field_max_edge_m, self.far_field_nearest_distance_m = (
                parent_face_far_field_edge_rule(
                    vertices=self.vertices,
                    faces=self.faces,
                    tx_positions=self.channel.tx_positions,
                    rx_positions=self.channel.rx_positions,
                    frequency_hz=self.channel.radar.fmcw.carrier_frequency,
                )
            )

    def _stage_update_facets(self):
        """Update facets for this solver pass."""
        self.previous_surface = POFacetState(
            self.state.face_indices,
            self.state.coefficients,
            self.state.phase_center_contributions,
            self.state.exact_vertex_path_offsets_m,
            self.state.path_lengths_m,
            self.state.spread_products_m2,
            self.state.phase_center_lengths_m,
            self.state.phase_center_spreads_m2,
            self.state.effective_weights[self.state.face_indices],
            self.state.delay_response,
            np.arange(self.state.face_indices.size),
            self.state.exact_face_areas,
            self.state.exact_face_normals,
        )
        self.surface, self.recompute_positions = update_po_facets(
            previous=self.previous_surface,
            selected=self.selected,
            vertex_offsets=vertex_path_offsets(
                self.vertices[self.faces[self.selected]],
                np.mean(self.channel.tx_positions, axis=0),
                np.mean(self.channel.rx_positions, axis=0),
            ),
            path_lengths=self.path_lengths,
            spreads=self.spreads,
            phase_lengths=self.phase_lengths,
            phase_spreads=self.phase_spreads,
            weights=self.effective_weights[self.selected],
            frequency_hz=self.channel.radar.fmcw.carrier_frequency,
            phase_error_threshold_rad=self.channel.config.po_refresh_on_phase_error_rad,
            area_fraction_threshold=self.channel.config.po_refresh_on_area_fraction,
            normal_threshold_deg=self.channel.config.po_refresh_on_normal_deg,
            delay_response=self.delay_response,
            response_indices=np.arange(self.selected.size),
            integrate=self.integrate_changed,
            dtype=self.state.coefficients.dtype,
            add_profile=self.add_profile,
            areas=self.selected_areas,
            normals=self.selected_normals,
        )
        self.coefficients = self.surface.coefficients
        self.phase_center_contrib = self.surface.phase_center_contributions

        if self.delay_response is not None:
            self.gains, self._, self._, self._ = (
                project_phase_center_contributions_to_virtual_channels(
                    np.ones(self.selected.size, complex),
                    self.selected_centroids,
                    self.channel.tx_positions,
                    self.channel.rx_positions,
                    self.channel.radar.fmcw.carrier_frequency,
                    virtual_channel_tx_indices=self.channel.radar.hardware.virtual_tx_indices(),
                    virtual_channel_rx_indices=self.channel.radar.hardware.virtual_rx_indices(),
                    virtual_channel_order=self.channel.radar.hardware.virtual_channel_order,
                    precision=self.channel.compute_precision,
                )
            )
            self.delay_response = self.delay_response.with_channel_gains(self.gains)

    def _stage_result(self):
        """Result for this solver pass."""
        self.state.delay_response = self.delay_response
        self.state_update_t0 = time.perf_counter()
        self.state.face_indices = self.selected.copy()
        self.state.coefficients = self.coefficients
        if self.path_lengths.shape[0] == 1:
            self.state.delays_s = (self.phase_lengths / c).astype(self.state.delays_s.dtype)
        else:
            self.state.delays_s = (self.path_lengths / c).astype(self.state.delays_s.dtype)
        self.state.phase_center_contributions = self.phase_center_contrib
        self.state.face_centroids = self.selected_centroids.copy()
        self.state.face_normals = self.selected_normals.copy()
        self.state.face_areas = self.selected_areas.copy()
        self.state.exact_vertex_path_offsets_m = self.surface.exact_vertex_path_offsets_m
        self.state.exact_face_areas = self.surface.exact_areas
        self.state.exact_face_normals = self.surface.exact_normals
        self.state.path_lengths_m = self.path_lengths
        self.state.spread_products_m2 = self.spreads
        self.state.phase_center_lengths_m = self.phase_lengths
        self.state.phase_center_spreads_m2 = self.phase_spreads
        self.state.raw_fractional_visibility = self.raw_visibility.copy()
        self.state.visible_sample_counts = self.visible_sample_counts.copy()
        self.state.temporal_fade_weights = self.fade_weights
        self.state.latched_fractional_visibility = self.latched_visibility
        self.state.effective_weights = self.effective_weights
        self.state.visibility_state = FaceVisibilityState(
            was_visible=self.raw_visibility > 0.0,
            fade_weights=self.fade_weights.copy(),
            latched_fractional_visibility=self.latched_visibility.copy(),
        )
        self.state.far_field_max_edge_m = self.far_field_max_edge_m
        self.state.far_field_nearest_distance_m = self.far_field_nearest_distance_m
        self.state.total_area = float(np.sum(self.areas))
        if self.state.total_area > 0.0:
            self.state.effective_visible_area_fraction = float(
                np.sum(self.areas * self.state.effective_weights) / self.state.total_area
            )
        else:
            self.state.effective_visible_area_fraction = 0.0

        self.result = POChannelResult(
            coefficients=self.state.coefficients,
            delay_response=self.state.delay_response,
            delays_s=self.state.delays_s,
            face_indices=self.state.face_indices,
            face_centroids=self.state.face_centroids,
            face_normals=self.state.face_normals,
            face_areas=self.state.face_areas,
            total_area=self.state.total_area,
            effective_visible_area_fraction=(self.state.effective_visible_area_fraction),
            raw_fractional_visibility=self.state.raw_fractional_visibility,
            latched_fractional_visibility=self.state.latched_fractional_visibility,
            temporal_fade_weights=self.state.temporal_fade_weights,
            effective_weights=self.state.effective_weights,
            visible_sample_counts=self.state.visible_sample_counts,
            sample_count_per_face=self.state.sample_count_per_face,
            phase_center_contributions=self.state.phase_center_contributions,
            phase_center_channel=complex(np.sum(self.state.phase_center_contributions)),
            visibility_state=self.state.visibility_state,
            far_field_max_edge_m=self.state.far_field_max_edge_m,
            far_field_nearest_distance_m=self.state.far_field_nearest_distance_m,
        )
        self.add_profile(
            "po_incremental_state_update",
            time.perf_counter() - self.state_update_t0,
        )
        return self.result, self.state, int(self.recompute_positions.size)

    def run(self):
        self._stage_geometry_and_visibility()
        self._stage_path_geometry()
        self._stage_update_facets()
        return self._stage_result()
