# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""mmWave radar sensing simulation driver."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable
import time

import numpy as np
from scipy.constants import c, pi

from .adc_synthesis import (
    _numpy_dtypes_for_precision,
    _synthesize_adc,
    _synthesize_adc_torch,
    _torch_cuda_is_available,
    _torch_device_and_dtypes,
    _torch_mps_available_for_precision,
    _use_torch_backend,
    numpy_dtypes_for_precision,
    synthesize_adc_from_paths,
)
from .physical_optics import (
    ComputeBackend,
    ComputePrecision,
    FaceVisibilityState,
    HumanVisibilityScene,
    POChannelResult,
    _aggregate_parent_face_far_field_analytic_contributions,
    _aggregate_parent_face_quadrature_contributions,
    _coerce_material,
    _fractional_face_visibility,
    _update_visibility_state,
    face_geometry,
    parent_face_far_field_analytic_geometry,
    parent_face_far_field_edge_rule,
    parent_face_quadrature_geometry,
    po_channel,
    project_phase_center_contributions_to_virtual_channels,
    vector_facet_face_contributions,
)
from .env_human_coupling import (
    compute_env_human_coupling_channels,
    extract_rt_reflector_bank,
)
from ..targets import MeshTarget
from ..radar import RadarSensor
from ..radar.hardware import resolve_virtual_channel_pairs
from .runtime import configure_mitsuba_variant
from .static_blocking import (
    PathVisibilityState,
    blocked_path_visibility,
    extract_static_path_bank,
)
from .rt_coherence import (
    CoherentRTPathBank,
    CoherentRTPathTransition,
    extract_coherent_rt_path_bank,
    match_coherent_rt_path_banks,
    synthesize_coherent_rt_transition,
    update_coherent_rt_path_bank,
)
from .calibration import (
    POCalibrationConfig,
    calibration_gain_array,
    fit_first_pose_adc_calibration,
    fit_sequence_path_power_calibration,
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
from .types import (
    CouplingMode,
    HybridStaticEnvPOConfig,
    HumanPOMobilityConfig,
    MobilityMode,
    RTCoherentTransitionConfig,
    RadarCube,
    SENSING_METADATA_SCHEMA_VERSION,
    SensingMetadata,
)
from .tdm import (
    collapse_tdm_cube,
    expanded_tdm_radar,
    tdm_expansion_required,
)


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
    exact_face_centroids: np.ndarray
    exact_face_normals: np.ndarray
    exact_face_areas: np.ndarray
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


def _match_sorted_face_positions(
    previous_faces: np.ndarray,
    selected_faces: np.ndarray,
) -> np.ndarray:
    """Maps sorted selected face ids to sorted previous-bank positions."""

    previous_faces = np.asarray(previous_faces, dtype=np.int64).reshape(-1)
    selected_faces = np.asarray(selected_faces, dtype=np.int64).reshape(-1)
    positions = np.full(selected_faces.shape, -1, dtype=np.int64)
    if previous_faces.size == 0 or selected_faces.size == 0:
        return positions
    candidate_positions = np.searchsorted(previous_faces, selected_faces)
    in_bounds = candidate_positions < previous_faces.size
    matched = np.zeros(selected_faces.shape, dtype=bool)
    matched[in_bounds] = (
        previous_faces[candidate_positions[in_bounds]]
        == selected_faces[in_bounds]
    )
    positions[matched] = candidate_positions[matched]
    return positions


def _path_value_numpy(value) -> np.ndarray:
    """Convert a Sionna path tensor or an array-like value to NumPy."""

    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


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
        hasattr(paths, field)
        for field in ("valid", "objects", "primitives")
    ):
        return counts

    valid = _path_value_numpy(paths.valid).astype(bool, copy=False)
    objects = _path_value_numpy(paths.objects)
    primitives = _path_value_numpy(paths.primitives)
    if (
        objects.ndim < 2
        or primitives.shape != objects.shape
        or objects.shape[1:] != valid.shape
    ):
        return counts

    valid_interactions = valid[None, ...]
    invalid_index = np.uint32(0xFFFFFFFF)
    for target in targets:
        scene_object = target.ensure_scene_object()
        object_id = int(
            _path_value_numpy(scene_object.object_id).reshape(-1)[0]
        )
        face_count = target.mesh_sequence.face_count
        hit_mask = valid_interactions & (objects == object_id)
        hit_faces = np.asarray(primitives[hit_mask], dtype=np.int64)
        hit_faces = hit_faces[
            (hit_faces >= 0)
            & (hit_faces < face_count)
            & (hit_faces != int(invalid_index))
        ]
        if hit_faces.size:
            counts[target.name] += np.bincount(
                hit_faces,
                minlength=face_count,
            )[:face_count]
    return counts


class MmWaveRadarSimulator:
    # pylint: disable=too-many-instance-attributes
    r"""Simulates FMCW radar sensing over a Sionna scene with mesh targets."""

    def __init__(
        self,
        *,
        path_solver: PathSolver | None = None,
        max_depth: int = 2,
        samples_per_src: int = 100000,
        max_num_paths_per_src: int = 100000,
        coupling_mode: CouplingMode = "one_target_bounce",
        retrace_displacement_fraction: float = 0.25,
        periodic_retrace: bool | None = None,
        retrace_once_per_frame: bool | None = None,
        periodic_retrace_period_chirps: int | None = None,
        diffuse_reflection: bool = False,
        refraction: bool = False,
        human_specular_reflection: bool = False,
        seed: int = 42,
        mobility_mode: MobilityMode = "rt_retrace",
        human_po_config: HumanPOMobilityConfig | None = None,
        hybrid_static_env_po_config: HybridStaticEnvPOConfig | None = None,
        rt_coherent_transition_config: RTCoherentTransitionConfig | None = None,
        progress: bool = False,
        progress_interval: int = 10,
        compute_backend: ComputeBackend = "numpy",
        compute_precision: ComputePrecision = "float64",
        adc_compute_backend: ComputeBackend | None = None,
        adc_compute_precision: ComputePrecision | None = None,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[dict], None] | None = None,
    ):
        configure_mitsuba_variant()
        if coupling_mode not in ("one_target_bounce", "one_human_bounce",
                                 "unrestricted"):
            raise ValueError("coupling_mode must be 'one_target_bounce' or"
                             " 'unrestricted'")
        if mobility_mode not in ("rt_retrace", "rt_coherent_bank",
                                 "human_only_po", "hybrid_static_env_po"):
            raise ValueError("mobility_mode must be 'rt_retrace',"
                             " 'rt_coherent_bank', 'human_only_po',"
                             " or 'hybrid_static_env_po'")
        if retrace_displacement_fraction <= 0.0:
            raise ValueError("retrace_displacement_fraction must be positive")
        if (periodic_retrace_period_chirps is not None
                and int(periodic_retrace_period_chirps) <= 0):
            raise ValueError("periodic_retrace_period_chirps must be positive")
        if periodic_retrace is None:
            periodic_retrace = (
                True
                if retrace_once_per_frame is None
                else bool(retrace_once_per_frame)
            )
        elif (retrace_once_per_frame is not None
              and bool(retrace_once_per_frame) != bool(periodic_retrace)):
            raise ValueError(
                "periodic_retrace and legacy retrace_once_per_frame disagree"
            )
        self.path_solver = path_solver
        self.max_depth = int(max_depth)
        self.samples_per_src = int(samples_per_src)
        self.max_num_paths_per_src = int(max_num_paths_per_src)
        if coupling_mode == "one_human_bounce":
            coupling_mode = "one_target_bounce"
        self.coupling_mode = coupling_mode
        self.retrace_displacement_fraction = float(
            retrace_displacement_fraction)
        self.periodic_retrace = bool(periodic_retrace)
        self.retrace_once_per_frame = self.periodic_retrace
        self.periodic_retrace_period_chirps = (
            None if periodic_retrace_period_chirps is None
            else int(periodic_retrace_period_chirps)
        )
        self.diffuse_reflection = bool(diffuse_reflection)
        self.refraction = bool(refraction)
        self.human_specular_reflection = bool(human_specular_reflection)
        self.seed = int(seed)
        self.mobility_mode = mobility_mode
        self.progress = bool(progress)
        self.progress_interval = max(int(progress_interval), 1)
        self.compute_backend = compute_backend
        self.compute_precision = compute_precision
        self.adc_compute_backend = (
            compute_backend if adc_compute_backend is None
            else adc_compute_backend
        )
        self.adc_compute_precision = (
            compute_precision if adc_compute_precision is None
            else adc_compute_precision
        )
        if cancel_check is not None and not callable(cancel_check):
            raise ValueError("cancel_check must be callable or None")
        if progress_callback is not None and not callable(progress_callback):
            raise ValueError("progress_callback must be callable or None")
        self.cancel_check = cancel_check
        self.progress_callback = progress_callback
        self.human_po_config = (
            HumanPOMobilityConfig()
            if human_po_config is None
            else human_po_config
        )
        self.rt_coherent_transition_config = (
            RTCoherentTransitionConfig()
            if rt_coherent_transition_config is None
            else rt_coherent_transition_config
        )
        self.hybrid_static_env_po_config = (
            HybridStaticEnvPOConfig()
            if hybrid_static_env_po_config is None
            else hybrid_static_env_po_config
        )
        self.last_target_face_hit_counts: dict[str, np.ndarray] = {}

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
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if tdm_expansion_required(radar):
            expanded_cube = self.run(
                scene,
                expanded_tdm_radar(radar),
                targets,
                num_frames=int(num_frames),
            )
            return collapse_tdm_cube(expanded_cube, radar)
        if self.mobility_mode == "human_only_po":
            return self._run_human_only_po(
                radar,
                targets,
                num_frames=int(num_frames),
                rt_reference_scene=scene,
            )
        if self.mobility_mode == "hybrid_static_env_po":
            return self._run_hybrid_static_env_po(
                scene,
                radar,
                targets,
                num_frames=int(num_frames),
            )
        if self.mobility_mode == "rt_coherent_bank":
            return self._run_rt_coherent_bank(
                scene, radar, targets, num_frames=int(num_frames))

        return self._run_rt_retrace(scene, radar, targets, num_frames=int(num_frames))

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
            return not self.retrace_once_per_frame
        return bool(config.crossfade)

    def effective_periodic_retrace_period_chirps(self, fmcw) -> int:
        """Returns the validated periodic retrace interval in chirps."""

        return self._effective_periodic_retrace_period_chirps(fmcw)

    def _effective_periodic_retrace_period_chirps(self, fmcw) -> int:
        """Validates and returns the active periodic retrace period."""

        num_chirps = int(fmcw.num_chirps_per_frame)
        interval = (
            num_chirps
            if self.periodic_retrace_period_chirps is None
            else int(self.periodic_retrace_period_chirps)
        )
        if interval <= 0:
            raise ValueError("periodic_retrace_period_chirps must be positive")
        if num_chirps % interval != 0:
            raise ValueError(
                "num_chirps_per_frame must be divisible by "
                "periodic_retrace_period_chirps"
            )
        return interval

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

        return (
            "frame" if int(period_chirps) == int(num_chirps_per_frame)
            else "periodic_chirp"
        )

    def retrace_event_reason(
        self,
        *,
        first_event: bool,
        period_chirps: int,
        num_chirps_per_frame: int,
    ) -> str:
        """Returns the event reason string used in retrace metadata."""

        return self._retrace_event_reason(
            first_event=first_event,
            period_chirps=period_chirps,
            num_chirps_per_frame=num_chirps_per_frame,
        )

    def _retrace_event_reason(
        self,
        *,
        first_event: bool,
        period_chirps: int,
        num_chirps_per_frame: int,
    ) -> str:
        """Returns the metadata reason for the current RT retrace event."""

        if first_event:
            return "initial"
        if self.retrace_once_per_frame:
            return self._periodic_chirp_retrace_reason(
                period_chirps=period_chirps,
                num_chirps_per_frame=num_chirps_per_frame,
            )
        return "displacement"

    def _run_rt_retrace(
        self,
        scene,
        radar: RadarSensor,
        targets: list[MeshTarget],
        *,
        num_frames: int,
    ) -> RadarCube:
        """Runs classic RT retracing with optional displacement-triggered updates."""

        radar.configure_scene(scene)
        for target in targets:
            target.add_to_scene(scene)

        fmcw = radar.fmcw
        num_chirps = fmcw.num_chirps_per_frame
        periodic_retrace_period_chirps = (
            self._effective_periodic_retrace_period_chirps(fmcw)
        )
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
        if (not np.array_equal(vc_tx, fallback_tx)
                or not np.array_equal(vc_rx, fallback_rx)):
            path_array_channel_mapping = {
                "virtual_channel_tx_indices": vc_tx,
                "virtual_channel_rx_indices": vc_rx,
            }
        _, adc_complex_dtype = self._numpy_dtypes_for_precision(
            self.adc_compute_precision
        )
        adc = np.zeros((num_frames, num_chirps, num_adc, num_vc),
                       dtype=adc_complex_dtype)
        times = (
            np.arange(num_frames, dtype=float)[:, None] * fmcw.frame_period
            + np.arange(num_chirps, dtype=float)[None, :]
            * fmcw.chirp_repetition_time
        )
        path_counts = np.zeros((num_frames, num_chirps), dtype=np.int64)
        max_displacements = np.zeros((num_frames, num_chirps), dtype=float)
        coherent_channel = np.zeros((num_frames, num_chirps, num_vc),
                                    dtype=np.complex128)
        events = []

        paths = None
        threshold = fmcw.wavelength * self.retrace_displacement_fraction
        frame_start_s = time.perf_counter()
        frame_retraces = 0
        for frame in range(num_frames):
            self.report_frame_progress(
                mode="rt_retrace",
                frame=frame,
                num_frames=num_frames,
            )
            if frame > 0:
                self._log_frame_progress(
                    mode="rt_retrace",
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
                    disp = np.max(np.linalg.norm(
                        vertices - target.reference_vertices, axis=1))
                    max_disp = max(max_disp, float(disp))
                max_displacements[frame, chirp] = max_disp

                if self.retrace_once_per_frame:
                    need_retrace = self._periodic_chirp_retrace_due(
                        frame=frame,
                        chirp=chirp,
                        num_chirps_per_frame=num_chirps,
                        period_chirps=periodic_retrace_period_chirps,
                        initialized=paths is not None,
                    )
                else:
                    need_retrace = paths is None or max_disp > threshold
                if need_retrace:
                    paths = self._trace(scene)
                    frame_retraces += 1
                    retrace_face_hits = _target_face_hit_counts(paths, targets)
                    for name, hit_counts in retrace_face_hits.items():
                        if name not in self.last_target_face_hit_counts:
                            self.last_target_face_hit_counts[name] = (
                                np.zeros_like(hit_counts)
                            )
                        self.last_target_face_hit_counts[name] += hit_counts
                    for target in targets:
                        target.reference_vertices = \
                            target.mesh_sequence.vertices_at(chirp_time)
                    events.append({
                        "frame": frame,
                        "chirp": chirp,
                        "time": chirp_time,
                        "reason": self._retrace_event_reason(
                            first_event=len(events) == 0,
                            period_chirps=periodic_retrace_period_chirps,
                            num_chirps_per_frame=num_chirps,
                        ),
                        "max_displacement": max_disp,
                        "threshold": threshold,
                        "periodic_retrace_period_chirps": (
                            periodic_retrace_period_chirps
                            if self.retrace_once_per_frame else None
                        ),
                        "human_specular_policy": (
                            "allow" if self.human_specular_reflection else "drop"
                        ),
                    })

                a, tau, valid = self._path_arrays(
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
                )
                path_counts[frame, chirp] = int(np.count_nonzero(valid))
                coherent_channel[frame, chirp] = self._coherent_channel(a, valid)
                adc[frame, chirp] = self._synthesize_adc(
                    a,
                    tau,
                    valid,
                    fmcw,
                    backend=self.adc_compute_backend,
                    precision=self.adc_compute_precision,
                )

        self._log_frame_progress(
            mode="rt_retrace",
            frame=num_frames - 1,
            num_frames=num_frames,
            frame_start_s=frame_start_s,
            extra=f"retraces={frame_retraces}",
        )

        phase_jumps = self._phase_jump_diagnostics(coherent_channel)
        metadata = SensingMetadata(
            events=events,
            path_counts=path_counts,
            max_displacements=max_displacements,
            virtual_channel_order=radar.hardware.virtual_channel_order,
            mode="rt_retrace",
            max_unwrapped_phase_jump=phase_jumps,
        )
        return RadarCube(adc=adc, times=times, metadata=metadata)

    def _run_rt_coherent_bank(
        self,
        scene,
        radar: RadarSensor,
        targets: list[MeshTarget],
        *,
        num_frames: int,
    ) -> RadarCube:
        """Runs RT with coherent chirp updates between retraces."""

        radar.configure_scene(scene)
        for target in targets:
            target.add_to_scene(scene)

        fmcw = radar.fmcw
        num_chirps = fmcw.num_chirps_per_frame
        periodic_retrace_period_chirps = (
            self._effective_periodic_retrace_period_chirps(fmcw)
        )
        num_adc = fmcw.num_adc_samples
        num_vc = radar.hardware.num_virtual_channels
        _, adc_complex_dtype = self._numpy_dtypes_for_precision(
            self.adc_compute_precision
        )
        adc = np.zeros((num_frames, num_chirps, num_adc, num_vc),
                       dtype=adc_complex_dtype)
        times = (
            np.arange(num_frames, dtype=float)[:, None] * fmcw.frame_period
            + np.arange(num_chirps, dtype=float)[None, :]
            * fmcw.chirp_repetition_time
        )
        path_counts = np.zeros((num_frames, num_chirps), dtype=np.int64)
        max_displacements = np.zeros((num_frames, num_chirps), dtype=float)
        coherent_channel = np.zeros((num_frames, num_chirps, num_vc),
                                    dtype=np.complex128)
        transition_persistent_counts = np.zeros_like(path_counts)
        transition_birth_counts = np.zeros_like(path_counts)
        transition_death_counts = np.zeros_like(path_counts)
        transition_alpha = np.zeros((num_frames, num_chirps), dtype=float)
        transition_old_weights = np.ones((num_frames, num_chirps), dtype=float)
        transition_new_weights = np.zeros((num_frames, num_chirps), dtype=float)
        threshold = fmcw.wavelength * self.retrace_displacement_fraction
        transition_config = self.rt_coherent_transition_config
        transition_matching_enabled = bool(transition_config.enabled)
        transition_crossfade_enabled = self._rt_transition_crossfade_enabled()

        runtime_profile_s = {
            "rt_coherent_trace": 0.0,
            "rt_coherent_bank_extract": 0.0,
            "rt_coherent_transition_match": 0.0,
            "rt_coherent_bank_update": 0.0,
            "rt_coherent_adc_synthesis": 0.0,
        }
        runtime_profile_counts = dict.fromkeys(runtime_profile_s, 0)

        def add_runtime_profile(key: str, elapsed_s: float):
            """Accumulates elapsed time and count for one RT-coherence stage."""

            runtime_profile_s[key] = runtime_profile_s.get(key, 0.0) + float(elapsed_s)
            runtime_profile_counts[key] = runtime_profile_counts.get(key, 0) + 1

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
                if self.retrace_once_per_frame:
                    need_retrace = self._periodic_chirp_retrace_due(
                        frame=frame,
                        chirp=chirp,
                        num_chirps_per_frame=num_chirps,
                        period_chirps=periodic_retrace_period_chirps,
                        initialized=bool(events),
                    )
                    if not need_retrace:
                        continue

                current_vertices = [
                    target.mesh_sequence.vertices_at(chirp_time)
                    for target in targets
                ]

                max_disp = 0.0
                for target_index, vertices in enumerate(current_vertices):
                    if reference_vertices[target_index] is None:
                        reference_vertices[target_index] = vertices.copy()
                    disp = np.max(np.linalg.norm(
                        vertices - reference_vertices[target_index], axis=1))
                    max_disp = max(max_disp, float(disp))
                max_displacements[frame, chirp] = max_disp

                if not self.retrace_once_per_frame:
                    need_retrace = not events or max_disp > threshold
                if not need_retrace:
                    continue

                for target_index, vertices in enumerate(current_vertices):
                    reference_vertices[target_index] = vertices.copy()
                event_flat_indices.append(flat_index)
                event_anchor_vertices.append([v.copy() for v in current_vertices])
                events.append({
                    "frame": frame,
                    "chirp": chirp,
                    "time": float(chirp_time),
                    "reason": self._retrace_event_reason(
                        first_event=len(events) == 0,
                        period_chirps=periodic_retrace_period_chirps,
                        num_chirps_per_frame=num_chirps,
                    ),
                    "max_displacement": max_disp,
                    "threshold": threshold,
                    "periodic_retrace_period_chirps": (
                        periodic_retrace_period_chirps
                        if self.retrace_once_per_frame else None
                    ),
                    "coherent_path_update": True,
                    "human_specular_reflection": bool(
                        self.human_specular_reflection
                    ),
                    "human_specular_policy": (
                        "allow" if self.human_specular_reflection else "drop"
                    ),
                })

        target_ids = self._target_object_ids(targets)
        event_banks: list[CoherentRTPathBank] = []
        for event, anchor_vertices in zip(events, event_anchor_vertices):
            self.check_cancelled()
            self.report_frame_progress(
                mode="rt_coherent_bank",
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
            paths = self._trace(scene)
            add_runtime_profile(
                "rt_coherent_trace",
                time.perf_counter() - trace_t0,
            )
            extract_t0 = time.perf_counter()
            bank = extract_coherent_rt_path_bank(
                paths,
                radar,
                targets,
                virtual_channel_order=radar.hardware.virtual_channel_order,
                anchor_target_vertices=anchor_vertices,
            )
            if not self.human_specular_reflection:
                bank = self._drop_target_specular_bank(bank, target_ids)
            add_runtime_profile(
                "rt_coherent_bank_extract",
                time.perf_counter() - extract_t0,
            )
            event_banks.append(bank)

        transitions: list[CoherentRTPathTransition | None] = []
        for event_index in range(max(len(event_banks) - 1, 0)):
            transition_interval_chirps = int(
                event_flat_indices[event_index + 1] - event_flat_indices[event_index]
            )
            same_frame_retrace = (
                int(events[event_index]["frame"])
                == int(events[event_index + 1]["frame"])
            )
            events[event_index]["transition_interval_chirps"] = (
                transition_interval_chirps
            )
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
                match_delay_tolerance_fraction=(
                    transition_config.match_delay_tolerance_fraction
                ),
                match_separation_fraction=(
                    transition_config.match_separation_fraction
                ),
                backend=self.compute_backend,
            )
            add_runtime_profile(
                "rt_coherent_transition_match",
                time.perf_counter() - match_t0,
            )
            transitions.append(transition)
            events[event_index]["transition_diagnostic_valid"] = True
            events[event_index]["persistent_path_count"] = int(
                transition.matched_old_indices.size
            )
            events[event_index]["death_path_count"] = int(
                transition.old_only_indices.size
            )
            events[event_index]["birth_path_count"] = int(
                transition.new_only_indices.size
            )
            events[event_index]["transition_crossfade"] = bool(
                transition_crossfade_enabled
            )
        if events:
            events[-1].setdefault("transition_interval_chirps", 0)
            events[-1].setdefault("transition_same_frame", False)
            events[-1].setdefault("transition_diagnostic_valid", False)
            events[-1].setdefault("persistent_path_count", int(np.count_nonzero(
                event_banks[-1].valid)) if event_banks else 0)
            events[-1].setdefault("death_path_count", 0)
            events[-1].setdefault("birth_path_count", 0)
            events[-1].setdefault("transition_crossfade", False)

        bank_coupling_masks = [
            self._one_target_bounce_mask_from_bank(bank, target_ids)
            for bank in event_banks
        ]

        frame_retrace_counts = np.zeros(num_frames, dtype=np.int64)
        for event in events:
            frame_retrace_counts[int(event["frame"])] += 1

        frame_start_s = time.perf_counter()
        for frame in range(num_frames):
            self.report_frame_progress(
                mode="rt_coherent_bank",
                frame=frame,
                num_frames=num_frames,
            )
            if frame > 0:
                self._log_frame_progress(
                    mode="rt_coherent_bank",
                    frame=frame - 1,
                    num_frames=num_frames,
                    frame_start_s=frame_start_s,
                    extra=f"retraces={int(frame_retrace_counts[frame - 1])}",
                )
            for chirp in range(num_chirps):
                self.check_cancelled()
                flat_index = frame * num_chirps + chirp
                event_index = int(
                    np.searchsorted(event_flat_indices, flat_index, side="right") - 1
                )
                if event_index < 0:
                    raise RuntimeError("coherent RT path bank was not initialized")
                current_bank = event_banks[event_index]
                event_chirp = flat_index == event_flat_indices[event_index]
                if self.retrace_once_per_frame and event_chirp:
                    current_vertices = event_anchor_vertices[event_index]
                else:
                    current_vertices = [
                        target.mesh_sequence.vertices_at(times[frame, chirp])
                        for target in targets
                    ]
                if self.retrace_once_per_frame and not event_chirp:
                    anchor_vertices = event_anchor_vertices[event_index]
                    max_displacements[frame, chirp] = max(
                        (
                            float(np.max(np.linalg.norm(
                                vertices - reference,
                                axis=1,
                            )))
                            for vertices, reference in zip(
                                current_vertices,
                                anchor_vertices,
                            )
                        ),
                        default=0.0,
                    )

                update_t0 = time.perf_counter()
                current_update = update_coherent_rt_path_bank(
                    current_bank,
                    current_vertices,
                    backend=self.compute_backend,
                )
                next_update = None
                transition = None
                interval_chirps = 1
                interval_offset = 0
                if event_index < len(event_banks) - 1 and transition_crossfade_enabled:
                    transition = transitions[event_index]
                    interval_chirps = int(
                        event_flat_indices[event_index + 1]
                        - event_flat_indices[event_index]
                    )
                    interval_offset = int(flat_index - event_flat_indices[event_index])
                    next_update = update_coherent_rt_path_bank(
                        event_banks[event_index + 1],
                        current_vertices,
                        backend=self.compute_backend,
                    )
                synth = synthesize_coherent_rt_transition(
                    current_update,
                    next_update,
                    transition,
                    interval_chirps=interval_chirps,
                    interval_offset=interval_offset,
                )
                add_runtime_profile(
                    "rt_coherent_bank_update",
                    time.perf_counter() - update_t0,
                )

                valid = synth.valid.copy()
                if self.coupling_mode == "one_target_bounce" and valid.size > 0:
                    source_masks = np.zeros(valid.shape, dtype=bool)
                    next_source = synth.source_is_new
                    current_source = ~next_source
                    if np.any(current_source):
                        source_masks[current_source] = bank_coupling_masks[
                            event_index
                        ][synth.source_indices[current_source]]
                    if np.any(next_source) and event_index + 1 < len(bank_coupling_masks):
                        source_masks[next_source] = bank_coupling_masks[
                            event_index + 1
                        ][synth.source_indices[next_source]]
                    valid &= source_masks

                path_counts[frame, chirp] = int(np.count_nonzero(valid))
                diagnostic_transition = (
                    transitions[event_index]
                    if event_index < len(transitions)
                    else None
                )
                if diagnostic_transition is None:
                    transition_persistent_counts[frame, chirp] = int(
                        np.count_nonzero(current_update.valid)
                    )
                    transition_death_counts[frame, chirp] = 0
                    transition_birth_counts[frame, chirp] = 0
                else:
                    transition_persistent_counts[frame, chirp] = int(
                        diagnostic_transition.matched_old_indices.size
                    )
                    transition_death_counts[frame, chirp] = int(
                        diagnostic_transition.old_only_indices.size
                    )
                    transition_birth_counts[frame, chirp] = int(
                        diagnostic_transition.new_only_indices.size
                    )
                transition_alpha[frame, chirp] = synth.alpha
                transition_old_weights[frame, chirp] = synth.old_weight
                transition_new_weights[frame, chirp] = synth.new_weight
                coherent_channel[frame, chirp] = self._coherent_channel(
                    synth.coefficients, valid)
                adc_t0 = time.perf_counter()
                adc[frame, chirp] = self._synthesize_adc(
                    synth.coefficients,
                    synth.delays_s,
                    valid,
                    fmcw,
                    backend=self.adc_compute_backend,
                    precision=self.adc_compute_precision,
                )
                add_runtime_profile(
                    "rt_coherent_adc_synthesis",
                    time.perf_counter() - adc_t0,
                )

        self._log_frame_progress(
            mode="rt_coherent_bank",
            frame=num_frames - 1,
            num_frames=num_frames,
            frame_start_s=frame_start_s,
            extra=f"retraces={int(frame_retrace_counts[num_frames - 1])}",
        )

        phase_jumps = self._phase_jump_diagnostics(coherent_channel)
        metadata = SensingMetadata(
            events=events,
            path_counts=path_counts,
            max_displacements=max_displacements,
            virtual_channel_order=radar.hardware.virtual_channel_order,
            mode="rt_coherent_bank",
            max_unwrapped_phase_jump=phase_jumps,
            runtime_profile_s=runtime_profile_s,
            runtime_profile_counts=runtime_profile_counts,
            rt_transition_persistent_path_counts=transition_persistent_counts,
            rt_transition_birth_path_counts=transition_birth_counts,
            rt_transition_death_path_counts=transition_death_counts,
            rt_transition_alpha=transition_alpha,
            rt_transition_old_weights=transition_old_weights,
            rt_transition_new_weights=transition_new_weights,
        )
        return RadarCube(adc=adc, times=times, metadata=metadata)

    def _run_hybrid_static_env_po(
        self,
        scene,
        radar: RadarSensor,
        targets: list[MeshTarget],
        *,
        num_frames: int,
    ) -> RadarCube:
        """Runs the hybrid static-environment plus human-PO mobility mode."""

        config = self.hybrid_static_env_po_config
        if not targets:
            raise ValueError("hybrid_static_env_po requires one mesh target")
        if len(targets) > 1:
            raise ValueError("hybrid_static_env_po supports one mesh target")

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
        runtime_profile_counts = {
            "hybrid_static_trace": 0,
            "hybrid_segment_extraction": 0,
            "hybrid_reflector_extraction": 0,
            "hybrid_rt_baseline": 0,
            "hybrid_po_calibration": 0,
            "hybrid_human_po": 0,
            "hybrid_blocking_rays": 0,
            "hybrid_static_adc_synthesis": 0,
            "hybrid_coupling_channel": 0,
            "hybrid_coupling_setup": 0,
            "hybrid_human_env_channel": 0,
            "hybrid_env_human_channel": 0,
            "hybrid_coupling_adc_synthesis": 0,
            "hybrid_human_env_adc_synthesis": 0,
            "hybrid_env_human_adc_synthesis": 0,
            "hybrid_assembly": 0,
        }

        radar.configure_scene(scene)
        trace_t0 = time.perf_counter()
        paths = self._trace(scene)
        runtime_profile_s["hybrid_static_trace"] = time.perf_counter() - trace_t0
        runtime_profile_counts["hybrid_static_trace"] = 1

        extraction_t0 = time.perf_counter()
        static_bank = extract_static_path_bank(
            paths,
            radar,
            virtual_channel_order=radar.hardware.virtual_channel_order,
        )
        runtime_profile_s["hybrid_segment_extraction"] = (
            time.perf_counter() - extraction_t0
        )
        runtime_profile_counts["hybrid_segment_extraction"] = 1

        reflector_t0 = time.perf_counter()
        reflector_bank = extract_rt_reflector_bank(
            scene,
            static_bank,
            max_reflectors=(
                config.coupling_max_reflectors if config.coupling_enabled else -1
            ),
            min_area_m2=config.coupling_min_reflector_area_m2,
            frequency_hz=radar.fmcw.carrier_frequency,
        )
        runtime_profile_s["hybrid_reflector_extraction"] = (
            time.perf_counter() - reflector_t0
        )
        runtime_profile_counts["hybrid_reflector_extraction"] = 1
        coupling_reflector_count = int(reflector_bank.vertices.shape[0])

        human_t0 = time.perf_counter()
        human_cube = self._run_human_only_po(
            radar,
            targets,
            num_frames=num_frames,
            static_visibility_scene=scene.mi_scene,
            human_po_config=config.human_po_config,
        )
        runtime_profile_s["hybrid_human_po"] = time.perf_counter() - human_t0
        runtime_profile_counts["hybrid_human_po"] = 1
        if human_cube.metadata.runtime_profile_s:
            for key, value in human_cube.metadata.runtime_profile_s.items():
                runtime_profile_s[f"human_{key}"] = float(value)
        if human_cube.metadata.runtime_profile_counts:
            for key, value in human_cube.metadata.runtime_profile_counts.items():
                runtime_profile_counts[f"human_{key}"] = int(value)

        fmcw = radar.fmcw
        times = human_cube.times.copy()
        _, adc_complex_dtype = self._numpy_dtypes_for_precision(
            self.adc_compute_precision
        )
        coupling_po_config = config.human_po_config
        coupling_compute_backend = (
            self.compute_backend if coupling_po_config.compute_backend is None
            else coupling_po_config.compute_backend
        )
        coupling_compute_precision = (
            self.compute_precision if coupling_po_config.compute_precision is None
            else coupling_po_config.compute_precision
        )
        coupling_incremental_update = (
            bool(coupling_po_config.incremental_update)
            if config.coupling_incremental_update is None
            else bool(config.coupling_incremental_update)
        )
        coupling_visibility_refresh_chirps = (
            int(coupling_po_config.incremental_visibility_refresh_chirps)
            if config.coupling_visibility_refresh_chirps is None
            else int(config.coupling_visibility_refresh_chirps)
        )
        static_unblocked_adc = np.zeros_like(human_cube.adc, dtype=adc_complex_dtype)
        static_blocked_adc = np.zeros_like(human_cube.adc, dtype=adc_complex_dtype)
        human_env_adc = np.zeros_like(human_cube.adc, dtype=adc_complex_dtype)
        env_human_adc = np.zeros_like(human_cube.adc, dtype=adc_complex_dtype)
        path_visibility_weights = np.zeros(
            times.shape + static_bank.delays_s.shape,
            dtype=np.float64,
        )
        static_path_count = int(np.count_nonzero(static_bank.valid))
        static_path_counts = np.full(times.shape, static_path_count, dtype=np.int64)
        blocked_counts = np.zeros(times.shape, dtype=np.int64)
        mean_visibility = np.zeros(times.shape, dtype=np.float64)
        human_env_counts = np.zeros(times.shape, dtype=np.int64)
        env_human_counts = np.zeros(times.shape, dtype=np.int64)
        human_env_min_lengths = np.full(times.shape, np.nan, dtype=np.float64)
        human_env_mean_lengths = np.full(times.shape, np.nan, dtype=np.float64)
        human_env_max_lengths = np.full(times.shape, np.nan, dtype=np.float64)
        env_human_min_lengths = np.full(times.shape, np.nan, dtype=np.float64)
        env_human_mean_lengths = np.full(times.shape, np.nan, dtype=np.float64)
        env_human_max_lengths = np.full(times.shape, np.nan, dtype=np.float64)

        unblocked_weights = static_bank.valid.astype(np.float64)
        unblocked_adc = self._synthesize_adc(
            static_bank.coefficients,
            static_bank.delays_s,
            static_bank.valid,
            fmcw,
            backend=self.adc_compute_backend,
            precision=self.adc_compute_precision,
        )
        static_unblocked_adc[:] = unblocked_adc[None, None, :, :]

        visibility_scene = HumanVisibilityScene(
            targets[0].mesh_sequence.faces,
            targets[0].mesh_sequence.vertices_at(float(times.reshape(-1)[0])),
            name="hybrid-static-blocking-human",
        )
        visibility_state: PathVisibilityState | None = None
        cached_human_env_active_sets = None
        cached_env_human_active_sets = None
        frame_start_s = time.perf_counter()

        for frame in range(num_frames):
            self.report_frame_progress(
                mode="hybrid_static_env_po",
                frame=frame,
                num_frames=num_frames,
            )
            if frame > 0:
                self._log_frame_progress(
                    mode="hybrid_static_env_po",
                    frame=frame - 1,
                    num_frames=num_frames,
                    frame_start_s=frame_start_s,
                    extra=f"static_paths={static_path_count}",
                )
            for chirp in range(fmcw.num_chirps_per_frame):
                self.check_cancelled()
                chirp_time = float(times[frame, chirp])
                vertices = targets[0].mesh_sequence.vertices_at(chirp_time)
                visibility_scene.update_vertices(vertices)

                if config.blocking_enabled:
                    blocking_t0 = time.perf_counter()
                    visibility = blocked_path_visibility(
                        static_bank,
                        visibility_scene,
                        blocker_vertices=vertices,
                        aabb_culling=config.blocking_aabb_culling,
                        aabb_margin_m=config.blocking_aabb_margin_m,
                        state=visibility_state,
                        fade_chirps=config.blocking_fade_chirps,
                        ray_epsilon_m=config.blocking_ray_epsilon_m,
                        ray_chunk_size=config.blocking_ray_chunk_size,
                    )
                    runtime_profile_s["hybrid_blocking_rays"] += (
                        time.perf_counter() - blocking_t0
                    )
                    runtime_profile_counts["hybrid_blocking_rays"] += 1
                    visibility_state = visibility.state
                    weights = visibility.visibility_weights
                    blocked_counts[frame, chirp] = int(
                        np.count_nonzero(visibility.blocked_paths & static_bank.valid)
                    )
                else:
                    weights = unblocked_weights
                    blocked_counts[frame, chirp] = 0

                path_visibility_weights[frame, chirp] = weights
                if static_path_count > 0:
                    mean_visibility[frame, chirp] = float(
                        np.mean(weights[static_bank.valid])
                    )
                else:
                    mean_visibility[frame, chirp] = 0.0

                static_adc_t0 = time.perf_counter()
                weighted_a = static_bank.coefficients * weights[None, :]
                valid = static_bank.valid & (weights > 0.0)
                static_blocked_adc[frame, chirp] = self._synthesize_adc(
                    weighted_a,
                    static_bank.delays_s,
                    valid,
                    fmcw,
                    backend=self.adc_compute_backend,
                    precision=self.adc_compute_precision,
                )
                runtime_profile_s["hybrid_static_adc_synthesis"] += (
                    time.perf_counter() - static_adc_t0
                )
                runtime_profile_counts["hybrid_static_adc_synthesis"] += 1

                if config.coupling_enabled and coupling_reflector_count > 0:
                    chirp_index = frame * fmcw.num_chirps_per_frame + chirp
                    refresh_coupling_visibility = (
                        not coupling_incremental_update
                        or cached_human_env_active_sets is None
                        or cached_env_human_active_sets is None
                        or chirp_index % coupling_visibility_refresh_chirps == 0
                    )
                    coupling_t0 = time.perf_counter()
                    coupling = compute_env_human_coupling_channels(
                        vertices=vertices,
                        faces=targets[0].mesh_sequence.faces,
                        tx_positions=radar.world_tx_positions(),
                        rx_positions=radar.world_rx_positions(),
                        virtual_channel_tx_indices=(
                            radar.hardware.virtual_tx_indices()
                        ),
                        virtual_channel_rx_indices=(
                            radar.hardware.virtual_rx_indices()
                        ),
                        virtual_channel_order=(
                            radar.hardware.virtual_channel_order
                        ),
                        frequency_hz=fmcw.carrier_frequency,
                        human_material=targets[0].material,
                        reflectors=reflector_bank,
                        human_visibility_scene=visibility_scene,
                        static_visibility_scene=scene.mi_scene,
                        reflection_scale=config.coupling_reflection_scale,
                        backend=coupling_compute_backend,
                        precision=coupling_compute_precision,
                        human_env_active_sets=(
                            None if refresh_coupling_visibility
                            else cached_human_env_active_sets
                        ),
                        env_human_active_sets=(
                            None if refresh_coupling_visibility
                            else cached_env_human_active_sets
                        ),
                    )
                    cached_human_env_active_sets = coupling.human_env_active_sets
                    cached_env_human_active_sets = coupling.env_human_active_sets
                    coupling_elapsed_s = time.perf_counter() - coupling_t0
                    runtime_profile_s["hybrid_coupling_channel"] += coupling_elapsed_s
                    runtime_profile_counts["hybrid_coupling_channel"] += 1
                    coupling_profile_s = coupling.runtime_profile_s or {}
                    coupling_profile_counts = coupling.runtime_profile_counts or {}
                    runtime_profile_s["hybrid_coupling_setup"] += (
                        coupling_profile_s.get("coupling_setup", 0.0)
                        + coupling_profile_s.get("coupling_assembly", 0.0)
                    )
                    runtime_profile_counts["hybrid_coupling_setup"] += (
                        coupling_profile_counts.get("coupling_setup", 0)
                        + coupling_profile_counts.get("coupling_assembly", 0)
                    )
                    runtime_profile_s["hybrid_human_env_channel"] += (
                        coupling_profile_s.get("human_env_channel", 0.0)
                    )
                    runtime_profile_counts["hybrid_human_env_channel"] += (
                        coupling_profile_counts.get("human_env_channel", 0)
                    )
                    runtime_profile_s["hybrid_env_human_channel"] += (
                        coupling_profile_s.get("env_human_channel", 0.0)
                    )
                    runtime_profile_counts["hybrid_env_human_channel"] += (
                        coupling_profile_counts.get("env_human_channel", 0)
                    )
                    human_env_counts[frame, chirp] = int(
                        np.count_nonzero(coupling.human_env_valid)
                    )
                    env_human_counts[frame, chirp] = int(
                        np.count_nonzero(coupling.env_human_valid)
                    )
                    human_env_delays = np.asarray(coupling.human_env_delays_s)
                    if human_env_delays.ndim == 2:
                        human_env_delays = np.mean(human_env_delays, axis=0)
                    human_env_lengths = c * human_env_delays[
                        coupling.human_env_valid
                    ]
                    if human_env_lengths.size:
                        human_env_min_lengths[frame, chirp] = float(
                            np.min(human_env_lengths)
                        )
                        human_env_mean_lengths[frame, chirp] = float(
                            np.mean(human_env_lengths)
                        )
                        human_env_max_lengths[frame, chirp] = float(
                            np.max(human_env_lengths)
                        )
                    env_human_delays = np.asarray(coupling.env_human_delays_s)
                    if env_human_delays.ndim == 2:
                        env_human_delays = np.mean(env_human_delays, axis=0)
                    env_human_lengths = c * env_human_delays[
                        coupling.env_human_valid
                    ]
                    if env_human_lengths.size:
                        env_human_min_lengths[frame, chirp] = float(
                            np.min(env_human_lengths)
                        )
                        env_human_mean_lengths[frame, chirp] = float(
                            np.mean(env_human_lengths)
                        )
                        env_human_max_lengths[frame, chirp] = float(
                            np.max(env_human_lengths)
                        )

                    human_env_adc_t0 = time.perf_counter()
                    human_env_adc[frame, chirp] = self._synthesize_adc(
                        coupling.human_env_coefficients,
                        coupling.human_env_delays_s,
                        coupling.human_env_valid,
                        fmcw,
                        backend=self.adc_compute_backend,
                        precision=self.adc_compute_precision,
                    )
                    human_env_adc_elapsed_s = time.perf_counter() - human_env_adc_t0
                    env_human_adc_t0 = time.perf_counter()
                    env_human_adc[frame, chirp] = self._synthesize_adc(
                        coupling.env_human_coefficients,
                        coupling.env_human_delays_s,
                        coupling.env_human_valid,
                        fmcw,
                        backend=self.adc_compute_backend,
                        precision=self.adc_compute_precision,
                    )
                    env_human_adc_elapsed_s = time.perf_counter() - env_human_adc_t0
                    runtime_profile_s["hybrid_human_env_adc_synthesis"] += (
                        human_env_adc_elapsed_s
                    )
                    runtime_profile_counts["hybrid_human_env_adc_synthesis"] += 1
                    runtime_profile_s["hybrid_env_human_adc_synthesis"] += (
                        env_human_adc_elapsed_s
                    )
                    runtime_profile_counts["hybrid_env_human_adc_synthesis"] += 1
                    runtime_profile_s["hybrid_coupling_adc_synthesis"] += (
                        human_env_adc_elapsed_s + env_human_adc_elapsed_s
                    )
                    runtime_profile_counts["hybrid_coupling_adc_synthesis"] += 1

        self._log_frame_progress(
            mode="hybrid_static_env_po",
            frame=num_frames - 1,
            num_frames=num_frames,
            frame_start_s=frame_start_s,
            extra=f"static_paths={static_path_count}",
        )

        po_calibration = {"mode": config.human_po_config.calibration.mode}
        human_po_gain = 1.0
        rt_baseline_components = {}
        if rt_calibration_enabled(config.human_po_config.calibration):
            calibration_num_frames = (
                num_frames
                if config.human_po_config.calibration.mode == "rt_sequence"
                else 1
            )
            if self.progress:
                print(
                    "[hybrid_static_env_po] starting full-scene RT baseline "
                    f"{calibration_num_frames} frames x "
                    f"{fmcw.num_chirps_per_frame} chirps "
                    f"for {config.human_po_config.calibration.mode} calibration",
                    flush=True,
                )
            baseline_t0 = time.perf_counter()
            rt_baseline = self._run_rt_baseline_for_calibration(
                scene,
                radar,
                targets,
                num_frames=calibration_num_frames,
                periodic_retrace=(
                    config.human_po_config.calibration.rt_periodic_retrace
                ),
                periodic_retrace_period_chirps=(
                    config.human_po_config.calibration.rt_periodic_retrace_period_chirps
                ),
            )
            baseline_elapsed_s = time.perf_counter() - baseline_t0
            runtime_profile_s["hybrid_rt_baseline"] = baseline_elapsed_s
            runtime_profile_counts["hybrid_rt_baseline"] = 1
            if self.progress:
                print(
                    "[hybrid_static_env_po] finished full-scene RT baseline "
                    f"elapsed={baseline_elapsed_s:.1f}s "
                    f"retraces={int(rt_baseline.get('retrace_count', 0))}",
                    flush=True,
                )
            calibration_t0 = time.perf_counter()
            if config.human_po_config.calibration.mode == "rt_sequence":
                human_po_gain, po_calibration = fit_sequence_path_power_calibration(
                    rt_baseline=rt_baseline,
                    po_times=human_cube.times,
                    po_path_power=human_cube.metadata.total_path_power,
                    config=config.human_po_config.calibration,
                )
            else:
                human_po_gain, po_calibration = self._po_amplitude_calibration(
                    rt_reference_adc=rt_baseline["human_reference_adc"],
                    po_reference_adc=human_cube.adc[0],
                    config=config.human_po_config.calibration,
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
            po_calibration["rt_baseline_retrace_count"] = int(
                rt_baseline.get("retrace_count", 0)
            )
            po_calibration["rt_baseline_num_frames"] = int(calibration_num_frames)
            periodic = bool(rt_baseline.get(
                "periodic_retrace",
                rt_baseline.get("retrace_once_per_frame", False),
            ))
            period_chirps = int(rt_baseline.get(
                "periodic_retrace_period_chirps",
                int(fmcw.num_chirps_per_frame),
            ))
            po_calibration["rt_baseline_periodic_retrace"] = periodic
            po_calibration["rt_baseline_retrace_once_per_frame"] = periodic
            po_calibration["rt_baseline_periodic_retrace_period_chirps"] = period_chirps
            po_calibration["rt_baseline_retrace_policy"] = (
                "periodic_frame"
                if periodic and period_chirps == int(fmcw.num_chirps_per_frame)
                else ("periodic_chirp" if periodic else "adaptive_displacement")
            )
            coherent_path_update = bool(rt_baseline.get(
                "coherent_path_update",
                False,
            ))
            po_calibration["rt_baseline_coherent_path_update"] = (
                coherent_path_update
            )
            po_calibration["rt_baseline_motion_between_retraces"] = (
                (
                    "coherent_bank_periodic_active"
                    if periodic else "coherent_bank_transition_crossfade"
                )
                if coherent_path_update
                else ("held_path_periodic" if periodic else "held_path_adaptive")
            )
            po_calibration["rt_baseline_phase_continuity_note"] = (
                (
                    "Periodic rt_coherent_bank updates the active retrace bank "
                    "chirp by chirp and switches banks at scheduled retrace "
                    "boundaries; birth/death matching is diagnostic only."
                    if periodic
                    else
                    "Adaptive rt_coherent_bank updates persistent paths chirp by "
                    "chirp, but adaptive retrace timing can still place "
                    "transition birth/death fades inside Doppler frames."
                )
                if coherent_path_update
                else (
                    "The RT baseline reuses the last traced path coefficients "
                    "and delays between retraces; it is a held-path control "
                    "and does not validate phase continuity between retrace "
                    "boundaries."
                )
            )
            runtime_profile_s["hybrid_po_calibration"] = (
                time.perf_counter() - calibration_t0
            )
            runtime_profile_counts["hybrid_po_calibration"] = 1
            if config.human_po_config.calibration.keep_rt_baseline:
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
        human_env_adc = (human_env_adc * chirp_gain).astype(
            adc_complex_dtype,
            copy=False,
        )
        env_human_adc = (env_human_adc * chirp_gain).astype(
            adc_complex_dtype,
            copy=False,
        )

        assembly_t0 = time.perf_counter()
        adc = (
            static_blocked_adc
            + human_po_adc
            + human_env_adc
            + env_human_adc
        ).astype(adc_complex_dtype)
        runtime_profile_s["hybrid_assembly"] = time.perf_counter() - assembly_t0
        runtime_profile_counts["hybrid_assembly"] = 1
        runtime_profile_s["hybrid_total"] = time.perf_counter() - total_t0
        runtime_profile_counts["hybrid_total"] = 1

        metadata = SensingMetadata(
            events=[
                {
                    "reason": "static_environment_trace",
                    "static_path_count": static_path_count,
                    "segment_count": int(static_bank.segment_path_indices.size),
                    "blocking_enabled": bool(config.blocking_enabled),
                    "blocking_aabb_culling": bool(config.blocking_aabb_culling),
                    "blocking_aabb_margin_m": float(config.blocking_aabb_margin_m),
                    "coupling_enabled": bool(config.coupling_enabled),
                    "coupling_reflector_count": coupling_reflector_count,
                    "coupling_candidate_source": "static_rt_paths",
                    "coupling_max_reflectors": int(config.coupling_max_reflectors),
                    "po_calibration_mode": po_calibration.get("mode"),
                    "po_calibration_gain": po_calibration.get(
                        "human_po_amplitude_gain", 1.0),
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
            mode="hybrid_static_env_po",
            visible_face_counts=human_cube.metadata.visible_face_counts,
            effective_visible_area_fraction=(
                human_cube.metadata.effective_visible_area_fraction
            ),
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
            incremental_visibility_refresh=(
                human_cube.metadata.incremental_visibility_refresh
            ),
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
        periodic_retrace: bool | None = None,
        periodic_retrace_period_chirps: int | None = None,
        retrace_once_per_frame: bool | None = None,
    ) -> dict:
        """Runs a full-scene RT baseline and extracts first-frame human paths."""

        if scene is None or not targets:
            return {}

        from .hybrid_pipeline import run_rt_mobility_baseline

        old_mobility_mode = self.mobility_mode
        old_periodic_retrace = self.periodic_retrace
        old_retrace_once_per_frame = self.retrace_once_per_frame
        old_periodic_retrace_period_chirps = self.periodic_retrace_period_chirps
        if periodic_retrace is None:
            effective_periodic_retrace = old_periodic_retrace
        else:
            effective_periodic_retrace = bool(periodic_retrace)
        if retrace_once_per_frame is not None:
            if (periodic_retrace is not None
                    and bool(retrace_once_per_frame) != effective_periodic_retrace):
                raise ValueError(
                    "periodic_retrace and legacy retrace_once_per_frame disagree"
                )
            effective_periodic_retrace = bool(retrace_once_per_frame)
        self.mobility_mode = "rt_coherent_bank"
        self.periodic_retrace = effective_periodic_retrace
        self.retrace_once_per_frame = effective_periodic_retrace
        if periodic_retrace_period_chirps is not None:
            if int(periodic_retrace_period_chirps) <= 0:
                raise ValueError("periodic_retrace_period_chirps must be positive")
            self.periodic_retrace_period_chirps = int(periodic_retrace_period_chirps)
        effective_periodic_retrace_period_chirps = (
            self._effective_periodic_retrace_period_chirps(radar.fmcw)
            if effective_periodic_retrace else None
        )
        try:
            result = run_rt_mobility_baseline(
                scene=scene,
                radar=radar,
                targets=targets,
                simulator=self,
                num_frames=num_frames,
            )
        finally:
            self.mobility_mode = old_mobility_mode
            self.periodic_retrace = old_periodic_retrace
            self.retrace_once_per_frame = old_retrace_once_per_frame
            self.periodic_retrace_period_chirps = old_periodic_retrace_period_chirps
        return {
            "cube": result.cube,
            "human_reference_adc": result.human_reference_adc,
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
            "periodic_retrace": effective_periodic_retrace,
            "retrace_once_per_frame": effective_periodic_retrace,
            "periodic_retrace_period_chirps": effective_periodic_retrace_period_chirps,
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




    def _run_human_only_po(
        self,
        radar: RadarSensor,
        targets: list[MeshTarget],
        *,
        num_frames: int,
        static_visibility_scene=None,
        rt_reference_scene=None,
        human_po_config: HumanPOMobilityConfig | None = None,
    ) -> RadarCube:
        """Runs human-only physical-optics mobility without tracing the environment."""

        config = self.human_po_config if human_po_config is None else human_po_config
        po_compute_backend = (
            self.compute_backend if config.compute_backend is None
            else config.compute_backend
        )
        po_compute_precision = (
            self.compute_precision if config.compute_precision is None
            else config.compute_precision
        )
        visibility_refresh_chirps = max(
            int(config.incremental_visibility_refresh_chirps),
            1,
        )
        full_refresh_chirps = (
            None if config.incremental_full_refresh_chirps is None
            else int(config.incremental_full_refresh_chirps)
        )
        if full_refresh_chirps is not None and full_refresh_chirps <= 0:
            full_refresh_chirps = None
        normal_threshold_rad = np.deg2rad(
            max(float(config.incremental_normal_threshold_deg), 0.0)
        )
        centroid_displacement_threshold_m = (
            None if config.incremental_centroid_displacement_threshold_m is None
            else float(config.incremental_centroid_displacement_threshold_m)
        )
        area_relative_threshold = (
            None if config.incremental_area_relative_threshold is None
            else float(config.incremental_area_relative_threshold)
        )
        po_parent_face_integration = config.po_integration_mode in (
            "parent_face_quadrature",
            "parent_face_far_field_analytic",
        )
        if len(targets) > 1:
            raise ValueError(
                "Phase 1 human_only_po currently supports at most one mesh target"
            )

        fmcw = radar.fmcw
        num_chirps = fmcw.num_chirps_per_frame
        num_adc = fmcw.num_adc_samples
        num_vc = radar.hardware.num_virtual_channels
        _, adc_complex_dtype = self._numpy_dtypes_for_precision(
            self.adc_compute_precision
        )
        adc = np.zeros((num_frames, num_chirps, num_adc, num_vc),
                       dtype=adc_complex_dtype)
        times = (
            np.arange(num_frames, dtype=float)[:, None] * fmcw.frame_period
            + np.arange(num_chirps, dtype=float)[None, :]
            * fmcw.chirp_repetition_time
        )
        path_counts = np.zeros((num_frames, num_chirps), dtype=np.int64)
        po_path_power = np.zeros((num_frames, num_chirps), dtype=np.float64)
        max_displacements = np.zeros((num_frames, num_chirps), dtype=float)
        visible_face_counts = np.zeros((num_frames, num_chirps), dtype=np.int64)
        effective_visible_area_fraction = np.zeros((num_frames, num_chirps),
                                                   dtype=float)
        runtime_per_chirp_s = np.zeros((num_frames, num_chirps), dtype=float)
        phase_center_channel = np.zeros((num_frames, num_chirps),
                                        dtype=np.complex128)
        incremental_recomputed_face_counts = np.zeros(
            (num_frames, num_chirps),
            dtype=np.int64,
        )
        incremental_phase_updated_face_counts = np.zeros(
            (num_frames, num_chirps),
            dtype=np.int64,
        )
        incremental_full_refresh = np.zeros(
            (num_frames, num_chirps),
            dtype=bool,
        )
        incremental_visibility_refresh = np.zeros(
            (num_frames, num_chirps),
            dtype=bool,
        )
        runtime_profile_s = {
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
        runtime_profile_counts = dict.fromkeys(runtime_profile_s, 0)

        def add_runtime_profile(key: str, elapsed_s: float):
            """Accumulates elapsed time and count for one human-PO stage."""

            runtime_profile_s[key] = (
                runtime_profile_s.get(key, 0.0) + float(elapsed_s)
            )
            runtime_profile_counts[key] = (
                runtime_profile_counts.get(key, 0) + 1
            )

        tx_positions = radar.world_tx_positions()
        rx_positions = radar.world_rx_positions()
        events = [{
            "frame": 0,
            "chirp": 0,
            "time": float(fmcw.chirp_time(0, 0)),
            "reason": "human_only_po",
            "visibility_samples_per_face": config.visibility_samples_per_face,
            "visibility_fade_chirps": config.visibility_fade_chirps,
            "po_integration_mode": config.po_integration_mode,
            "po_quadrature_phase_span_scale_rad": (
                config.po_quadrature_phase_span_scale_rad
            ),
            "po_quadrature_max_refinement_depth": (
                config.po_quadrature_max_refinement_depth
            ),
            "po_quadrature_max_subfaces_per_parent": (
                config.po_quadrature_max_subfaces_per_parent
            ),
        }]

        if not targets:
            metadata = SensingMetadata(
                events=events,
                path_counts=path_counts,
                max_displacements=max_displacements,
                virtual_channel_order=radar.hardware.virtual_channel_order,
                mode="human_only_po",
                visible_face_counts=visible_face_counts,
                effective_visible_area_fraction=effective_visible_area_fraction,
                runtime_per_chirp_s=runtime_per_chirp_s,
                max_unwrapped_phase_jump=np.zeros((num_frames, num_chirps),
                                                  dtype=float),
                runtime_profile_s=runtime_profile_s,
                runtime_profile_counts=runtime_profile_counts,
                incremental_recomputed_face_counts=(
                    incremental_recomputed_face_counts),
                incremental_phase_updated_face_counts=(
                    incremental_phase_updated_face_counts),
                incremental_full_refresh=incremental_full_refresh,
                incremental_visibility_refresh=incremental_visibility_refresh,
                total_path_power=po_path_power,
            )
            return RadarCube(adc=adc, times=times, metadata=metadata)

        target = targets[0]
        mesh_sequence = target.mesh_sequence
        precomputed_visibility = config.precomputed_raw_fractional_visibility
        visibility_scene = None
        if precomputed_visibility is None or config.incremental_update:
            visibility_scene = HumanVisibilityScene(
                faces=mesh_sequence.faces,
                vertices=mesh_sequence.vertices[0],
                name=f"{target.name}-human-po",
            )
        visibility_state = None
        incremental_state = None
        previous_vertices = None
        frame_start_s = time.perf_counter()

        for frame in range(num_frames):
            self.report_frame_progress(
                mode="human_only_po",
                frame=frame,
                num_frames=num_frames,
            )
            if frame > 0:
                self._log_frame_progress(
                    mode="human_only_po",
                    frame=frame - 1,
                    num_frames=num_frames,
                    frame_start_s=frame_start_s,
                    extra=(f"mean_visible_faces="
                           f"{float(np.mean(visible_face_counts[frame - 1])):.1f}"),
                )
            for chirp in range(num_chirps):
                self.check_cancelled()
                t0 = time.perf_counter()
                chirp_time = fmcw.chirp_time(frame, chirp)
                times[frame, chirp] = chirp_time
                mesh_t0 = time.perf_counter()
                vertices = mesh_sequence.vertices_at(chirp_time)
                if previous_vertices is not None:
                    max_displacements[frame, chirp] = float(np.max(
                        np.linalg.norm(vertices - previous_vertices, axis=1)
                    ))
                previous_vertices = vertices.copy()
                add_runtime_profile(
                    "mesh_update",
                    time.perf_counter() - mesh_t0,
                )

                po_t0 = time.perf_counter()
                chirp_index_abs = frame * num_chirps + chirp
                force_full_refresh = (
                    not config.incremental_update
                    or incremental_state is None
                    or (
                        full_refresh_chirps is not None
                        and chirp_index_abs % full_refresh_chirps == 0
                    )
                )
                visibility_refresh_due = (
                    config.incremental_update
                    and chirp_index_abs % visibility_refresh_chirps == 0
                )
                force_visibility_refresh = (
                    not force_full_refresh
                    and visibility_refresh_due
                )
                if force_full_refresh:
                    full_rebuild_t0 = time.perf_counter()
                    raw_visibility_override = precomputed_visibility
                    visible_sample_counts_override = (
                        config.precomputed_visible_sample_counts
                    )
                    if (
                        raw_visibility_override is None
                        and po_parent_face_integration
                        and config.incremental_update
                        and incremental_state is not None
                        and not visibility_refresh_due
                    ):
                        raw_visibility_override = (
                            incremental_state.raw_fractional_visibility
                        )
                        visible_sample_counts_override = (
                            incremental_state.visible_sample_counts
                        )
                    result = po_channel(
                        vertices=vertices,
                        faces=mesh_sequence.faces,
                        tx_positions=tx_positions,
                        rx_positions=rx_positions,
                        virtual_channel_tx_indices=(
                            radar.hardware.virtual_tx_indices()
                        ),
                        virtual_channel_rx_indices=(
                            radar.hardware.virtual_rx_indices()
                        ),
                        virtual_channel_order=(
                            radar.hardware.virtual_channel_order
                        ),
                        frequency_hz=fmcw.carrier_frequency,
                        material=target.material,
                        visibility="fractional_shadow_fade",
                        visibility_state=visibility_state,
                        visibility_scene=visibility_scene,
                        static_visibility_scene=static_visibility_scene,
                        visibility_samples_per_face=(
                            config.visibility_samples_per_face),
                        visibility_fade_chirps=config.visibility_fade_chirps,
                        visibility_use_phase_center=(
                            config.visibility_use_phase_center),
                        adaptive_visibility_sampling=config.adaptive_visibility_sampling,
                        adaptive_visibility_edge_margin=config.adaptive_visibility_edge_margin,
                        precomputed_raw_fractional_visibility=(
                            raw_visibility_override),
                        precomputed_visible_sample_counts=(
                            visible_sample_counts_override),
                        po_integration_mode=config.po_integration_mode,
                        po_quadrature_phase_span_scale_rad=(
                            config.po_quadrature_phase_span_scale_rad),
                        po_quadrature_max_refinement_depth=(
                            config.po_quadrature_max_refinement_depth),
                        po_quadrature_max_subfaces_per_parent=(
                            config.po_quadrature_max_subfaces_per_parent),
                        progress_callback=config.progress_callback,
                        compute_backend=po_compute_backend,
                        compute_precision=po_compute_precision,
                    )
                    incremental_state = self._po_incremental_state_from_result(
                        result,
                        tx_positions=tx_positions,
                        rx_positions=rx_positions,
                        virtual_channel_tx_indices=(
                            radar.hardware.virtual_tx_indices()
                        ),
                        virtual_channel_rx_indices=(
                            radar.hardware.virtual_rx_indices()
                        ),
                        virtual_channel_order=(
                            radar.hardware.virtual_channel_order
                        ),
                    )
                    incremental_recomputed_face_counts[frame, chirp] = (
                        result.face_indices.size
                    )
                    incremental_full_refresh[frame, chirp] = True
                    add_runtime_profile(
                        "po_full_rebuild",
                        time.perf_counter() - full_rebuild_t0,
                    )
                else:
                    incremental_update_t0 = time.perf_counter()
                    result, incremental_state, recomputed_count = (
                        self._update_po_incremental_state(
                            state=incremental_state,
                            vertices=vertices,
                            faces=mesh_sequence.faces,
                            tx_positions=tx_positions,
                            rx_positions=rx_positions,
                            virtual_channel_tx_indices=(
                                radar.hardware.virtual_tx_indices()
                            ),
                            virtual_channel_rx_indices=(
                                radar.hardware.virtual_rx_indices()
                            ),
                            virtual_channel_order=(
                                radar.hardware.virtual_channel_order
                            ),
                            frequency_hz=fmcw.carrier_frequency,
                            material=target.material,
                            normal_threshold_rad=normal_threshold_rad,
                            centroid_displacement_threshold_m=(
                                centroid_displacement_threshold_m),
                            area_relative_threshold=area_relative_threshold,
                            visibility_fade_chirps=config.visibility_fade_chirps,
                            refresh_visibility=force_visibility_refresh,
                            visibility_scene=visibility_scene,
                            static_visibility_scene=static_visibility_scene,
                            visibility_samples_per_face=(
                                config.visibility_samples_per_face),
                            adaptive_visibility_sampling=(
                                config.adaptive_visibility_sampling),
                            adaptive_visibility_edge_margin=(
                                config.adaptive_visibility_edge_margin),
                            po_integration_mode=config.po_integration_mode,
                            po_quadrature_phase_span_scale_rad=(
                                config.po_quadrature_phase_span_scale_rad),
                            po_quadrature_max_refinement_depth=(
                                config.po_quadrature_max_refinement_depth),
                            po_quadrature_max_subfaces_per_parent=(
                                config.po_quadrature_max_subfaces_per_parent),
                            progress_callback=config.progress_callback,
                            compute_backend=po_compute_backend,
                            compute_precision=po_compute_precision,
                            runtime_profile_s=runtime_profile_s,
                            runtime_profile_counts=runtime_profile_counts,
                        )
                    )
                    add_runtime_profile(
                        "po_incremental_update",
                        time.perf_counter() - incremental_update_t0,
                    )
                    incremental_recomputed_face_counts[frame, chirp] = (
                        recomputed_count
                    )
                    incremental_phase_updated_face_counts[frame, chirp] = (
                        result.face_indices.size - recomputed_count
                    )
                    incremental_visibility_refresh[frame, chirp] = (
                        force_visibility_refresh
                    )
                visibility_state = result.visibility_state
                add_runtime_profile(
                    "po_channel",
                    time.perf_counter() - po_t0,
                )
                result = self._apply_po_hardware_gain(result, radar)

                valid = np.any(np.abs(result.coefficients) > 0.0, axis=0)
                path_counts[frame, chirp] = int(np.count_nonzero(valid))
                po_path_power[frame, chirp] = po_path_power_from_coefficients(
                    result.coefficients,
                    valid,
                )
                adc_t0 = time.perf_counter()
                adc[frame, chirp] = self._synthesize_adc(
                    result.coefficients,
                    result.delays_s,
                    valid,
                    fmcw,
                    backend=self.adc_compute_backend,
                    precision=self.adc_compute_precision,
                )
                add_runtime_profile(
                    "adc_synthesis",
                    time.perf_counter() - adc_t0,
                )
                metadata_t0 = time.perf_counter()
                phase_center_channel[frame, chirp] = result.phase_center_channel
                visible_face_counts[frame, chirp] = int(np.count_nonzero(
                    result.raw_fractional_visibility > 0.0
                ))

                effective_visible_area_fraction[frame, chirp] = (
                    result.effective_visible_area_fraction
                )
                add_runtime_profile(
                    "metadata",
                    time.perf_counter() - metadata_t0,
                )
                runtime_per_chirp_s[frame, chirp] = time.perf_counter() - t0
                self._log_chirp_progress(
                    mode="human_only_po",
                    frame=frame,
                    num_frames=num_frames,
                    chirp=chirp,
                    num_chirps=num_chirps,
                    run_start_s=frame_start_s,
                    chirp_runtime_s=float(runtime_per_chirp_s[frame, chirp]),
                    extra=(
                        f"integration={config.po_integration_mode} "
                        f"incremental={bool(config.incremental_update)} "
                        f"full_refresh={bool(incremental_full_refresh[frame, chirp])} "
                        f"visibility_refresh={bool(incremental_visibility_refresh[frame, chirp])} "
                        f"recomputed_faces={int(incremental_recomputed_face_counts[frame, chirp])} "
                        f"phase_updated_faces={int(incremental_phase_updated_face_counts[frame, chirp])} "
                        f"visible_faces={int(visible_face_counts[frame, chirp])} "
                        f"paths={int(path_counts[frame, chirp])}"
                    ),
                )

        self._log_frame_progress(
            mode="human_only_po",
            frame=num_frames - 1,
            num_frames=num_frames,
            frame_start_s=frame_start_s,
            extra=(f"mean_visible_faces="
                   f"{float(np.mean(visible_face_counts[num_frames - 1])):.1f}"),
        )

        po_calibration = {"mode": config.calibration.mode}
        human_po_gain = 1.0
        components = None
        if rt_calibration_enabled(config.calibration):
            if rt_reference_scene is None:
                po_calibration = {
                    "mode": config.calibration.mode,
                    "gain_type": config.calibration.gain_type,
                    "applied": False,
                    "human_po_amplitude_gain": 1.0,
                    "human_po_amplitude_gain_db": 0.0,
                    "reference_frame": 0,
                    "warnings": ["rt_reference_scene_missing"],
                }
            else:
                calibration_num_frames = (
                    num_frames
                    if config.calibration.mode == "rt_sequence"
                    else 1
                )
                if self.progress:
                    print(
                        "[human_only_po] starting full-scene RT baseline "
                        f"{calibration_num_frames} frames x "
                        f"{fmcw.num_chirps_per_frame} chirps "
                        f"for {config.calibration.mode} calibration",
                        flush=True,
                    )
                baseline_t0 = time.perf_counter()
                rt_baseline = self._run_rt_baseline_for_calibration(
                    rt_reference_scene,
                    radar,
                    targets,
                    num_frames=calibration_num_frames,
                    periodic_retrace=config.calibration.rt_periodic_retrace,
                    periodic_retrace_period_chirps=(
                        config.calibration.rt_periodic_retrace_period_chirps
                    ),
                )
                baseline_elapsed_s = time.perf_counter() - baseline_t0
                add_runtime_profile(
                    "po_calibration_rt_baseline",
                    baseline_elapsed_s,
                )
                if self.progress:
                    print(
                        "[human_only_po] finished full-scene RT baseline "
                        f"elapsed={baseline_elapsed_s:.1f}s "
                        f"retraces={int(rt_baseline.get('retrace_count', 0))}",
                        flush=True,
                    )
                fit_t0 = time.perf_counter()
                if config.calibration.mode == "rt_sequence":
                    gain, po_calibration = fit_sequence_path_power_calibration(
                        rt_baseline=rt_baseline,
                        po_times=times,
                        po_path_power=po_path_power,
                        config=config.calibration,
                    )
                else:
                    gain, po_calibration = self._po_amplitude_calibration(
                        rt_reference_adc=rt_baseline["human_reference_adc"],
                        po_reference_adc=adc[0],
                        config=config.calibration,
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
                periodic = bool(rt_baseline.get(
                    "periodic_retrace",
                    rt_baseline.get("retrace_once_per_frame", False),
                ))
                period_chirps = int(rt_baseline.get(
                    "periodic_retrace_period_chirps",
                    int(fmcw.num_chirps_per_frame),
                ))
                po_calibration["rt_baseline_periodic_retrace"] = periodic
                po_calibration["rt_baseline_retrace_once_per_frame"] = periodic
                po_calibration["rt_baseline_periodic_retrace_period_chirps"] = period_chirps
                po_calibration["rt_baseline_retrace_policy"] = (
                    "periodic_frame"
                    if periodic and period_chirps == int(fmcw.num_chirps_per_frame)
                    else ("periodic_chirp" if periodic else "adaptive_displacement")
                )
                coherent_path_update = bool(rt_baseline.get(
                    "coherent_path_update",
                    False,
                ))
                po_calibration["rt_baseline_coherent_path_update"] = (
                    coherent_path_update
                )
                po_calibration["rt_baseline_motion_between_retraces"] = (
                    (
                        "coherent_bank_periodic_active"
                        if periodic else "coherent_bank_transition_crossfade"
                    )
                    if coherent_path_update
                    else ("held_path_periodic" if periodic else "held_path_adaptive")
                )
                po_calibration["rt_baseline_phase_continuity_note"] = (
                    (
                        "Periodic rt_coherent_bank updates the active retrace bank "
                        "chirp by chirp and switches banks at scheduled retrace "
                        "boundaries; birth/death matching is diagnostic only."
                        if periodic
                        else
                        "Adaptive rt_coherent_bank updates persistent paths chirp by "
                        "chirp, but adaptive retrace timing can still place "
                        "transition birth/death fades inside Doppler frames."
                    )
                    if coherent_path_update
                    else (
                        "The RT baseline reuses the last traced path coefficients "
                        "and delays between retraces; it is a held-path control "
                        "and does not validate phase continuity between retrace "
                        "boundaries."
                    )
                )
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
                phase_center_channel = (
                    phase_center_channel * human_po_gain_map[..., None]
                )
                if config.calibration.keep_rt_baseline:
                    components = {
                        "rt_baseline": rt_baseline["cube"].adc,
                        "rt_baseline_full_scene": rt_baseline["cube"].adc,
                        "rt_baseline_one_human_first_frame": (
                            rt_baseline["human_reference_adc"][None, ...]
                        ),
                    }
            events.append({"reason": "po_calibration", **po_calibration})

        phase_jumps = self._phase_jump_diagnostics(phase_center_channel)
        metadata = SensingMetadata(
            events=events,
            path_counts=path_counts,
            max_displacements=max_displacements,
            virtual_channel_order=radar.hardware.virtual_channel_order,
            mode="human_only_po",
            visible_face_counts=visible_face_counts,
            effective_visible_area_fraction=effective_visible_area_fraction,
            runtime_per_chirp_s=runtime_per_chirp_s,
            max_unwrapped_phase_jump=phase_jumps,
            runtime_profile_s=runtime_profile_s,
            runtime_profile_counts=runtime_profile_counts,
            incremental_recomputed_face_counts=(
                incremental_recomputed_face_counts),
            incremental_phase_updated_face_counts=(
                incremental_phase_updated_face_counts),
            incremental_full_refresh=incremental_full_refresh,
            incremental_visibility_refresh=incremental_visibility_refresh,
            total_path_power=(
                po_path_power
                * calibration_gain_array(
                    po_calibration,
                    times.shape,
                    fallback=human_po_gain,
                ) ** 2
            ),
            po_calibration=po_calibration,
        )
        return RadarCube(adc=adc, times=times, metadata=metadata, components=components)

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
        phase_center_contributions = (
            result.phase_center_contributions * np.mean(phase_center_gains, axis=0)
        )
        return replace(
            result,
            coefficients=coefficients.astype(result.coefficients.dtype, copy=False),
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
            coefficients=result.coefficients.copy(),
            delays_s=result.delays_s.copy(),
            phase_center_contributions=(
                result.phase_center_contributions.copy()),
            face_centroids=result.face_centroids.copy(),
            face_normals=result.face_normals.copy(),
            face_areas=result.face_areas.copy(),
            exact_face_centroids=result.face_centroids.copy(),
            exact_face_normals=result.face_normals.copy(),
            exact_face_areas=result.face_areas.copy(),
            path_lengths_m=path_lengths,
            spread_products_m2=spreads,
            phase_center_lengths_m=phase_lengths.reshape(-1),
            phase_center_spreads_m2=phase_spreads.reshape(-1),
            raw_fractional_visibility=result.raw_fractional_visibility.copy(),
            latched_fractional_visibility=(
                result.latched_fractional_visibility.copy()),
            temporal_fade_weights=result.temporal_fade_weights.copy(),
            effective_weights=result.effective_weights.copy(),
            visible_sample_counts=result.visible_sample_counts.copy(),
            total_area=result.total_area,
            effective_visible_area_fraction=(
                result.effective_visible_area_fraction),
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
        faces: np.ndarray,
        tx_positions: np.ndarray,
        rx_positions: np.ndarray,
        virtual_channel_tx_indices: np.ndarray | None,
        virtual_channel_rx_indices: np.ndarray | None,
        virtual_channel_order: str,
        frequency_hz: float,
        material,
        normal_threshold_rad: float,
        centroid_displacement_threshold_m: float | None,
        area_relative_threshold: float | None,
        visibility_fade_chirps: int,
        refresh_visibility: bool,
        visibility_scene: HumanVisibilityScene | None,
        static_visibility_scene,
        visibility_samples_per_face: int,
        adaptive_visibility_sampling: bool,
        adaptive_visibility_edge_margin: float,
        po_integration_mode: str,
        po_quadrature_phase_span_scale_rad: float,
        po_quadrature_max_refinement_depth: int,
        po_quadrature_max_subfaces_per_parent: int,
        progress_callback,
        compute_backend: ComputeBackend,
        compute_precision: ComputePrecision,
        runtime_profile_s: dict[str, float] | None = None,
        runtime_profile_counts: dict[str, int] | None = None,
    ) -> tuple[POChannelResult, _POIncrementalState, int]:
        """Updates cached PO state and exactly recomputes changed faces."""

        def add_profile(key: str, elapsed_s: float):
            """Accumulates optional incremental-PO profiling counters."""

            if runtime_profile_s is not None:
                runtime_profile_s[key] = (
                    runtime_profile_s.get(key, 0.0) + float(elapsed_s)
                )
            if runtime_profile_counts is not None:
                runtime_profile_counts[key] = (
                    runtime_profile_counts.get(key, 0) + 1
                )

        faces = np.asarray(faces, dtype=np.uint32)
        face_geometry_t0 = time.perf_counter()
        centroids, normals, areas = face_geometry(vertices, faces)
        add_profile(
            "po_incremental_face_geometry",
            time.perf_counter() - face_geometry_t0,
        )
        raw_visibility = state.raw_fractional_visibility
        visible_sample_counts = state.visible_sample_counts
        if refresh_visibility:
            visibility_t0 = time.perf_counter()
            if visibility_scene is None:
                visibility_scene = HumanVisibilityScene(
                    faces=faces,
                    vertices=vertices,
                    name="human-po-incremental-visibility",
                )
            else:
                visibility_scene.update_vertices(vertices)
            raw_visibility, visible_sample_counts = _fractional_face_visibility(
                vertices=vertices,
                faces=faces,
                face_normals=normals,
                tx_position=np.mean(np.asarray(tx_positions, dtype=float), axis=0),
                rx_position=np.mean(np.asarray(rx_positions, dtype=float), axis=0),
                visibility_scene=visibility_scene,
                static_visibility_scene=static_visibility_scene,
                samples_per_face=visibility_samples_per_face,
                adaptive_edge_sampling=adaptive_visibility_sampling,
                edge_front_margin=adaptive_visibility_edge_margin,
            )
            add_profile(
                "po_incremental_visibility_refresh",
                time.perf_counter() - visibility_t0,
            )

        visibility_state_t0 = time.perf_counter()
        fade_weights, latched_visibility = _update_visibility_state(
            raw_fractional_visibility=raw_visibility,
            state=state.visibility_state,
            fade_chirps=visibility_fade_chirps,
        )
        effective_weights = latched_visibility * fade_weights
        if refresh_visibility:
            selected = np.flatnonzero(effective_weights > 0.0)
        else:
            selected = state.face_indices[
                effective_weights[state.face_indices] > 0.0
            ]
        add_profile(
            "po_incremental_visibility_state",
            time.perf_counter() - visibility_state_t0,
        )
        selected_centroids = centroids[selected]
        selected_normals = normals[selected]
        selected_areas = areas[selected]

        path_geometry_t0 = time.perf_counter()
        path_lengths, spreads = self._face_path_geometry(
            selected_centroids,
            tx_positions,
            rx_positions,
            virtual_channel_tx_indices=virtual_channel_tx_indices,
            virtual_channel_rx_indices=virtual_channel_rx_indices,
            virtual_channel_order=virtual_channel_order,
        )
        phase_lengths, phase_spreads = self._face_path_geometry(
            selected_centroids,
            np.mean(np.asarray(tx_positions, dtype=float), axis=0)[None, :],
            np.mean(np.asarray(rx_positions, dtype=float), axis=0)[None, :],
        )
        phase_lengths = phase_lengths.reshape(-1)
        phase_spreads = phase_spreads.reshape(-1)
        add_profile(
            "po_incremental_path_geometry",
            time.perf_counter() - path_geometry_t0,
        )

        bank_merge_t0 = time.perf_counter()
        old_positions = _match_sorted_face_positions(
            state.face_indices,
            selected,
        )
        existing_mask = old_positions >= 0
        num_vc = int(np.asarray(tx_positions).shape[0]
                     * np.asarray(rx_positions).shape[0])
        coefficients = np.zeros(
            (num_vc, selected.shape[0]),
            dtype=state.coefficients.dtype,
        )
        phase_center_contrib = np.zeros(
            selected.shape[0],
            dtype=state.phase_center_contributions.dtype,
        )
        add_profile(
            "po_incremental_bank_merge",
            time.perf_counter() - bank_merge_t0,
        )

        if np.any(existing_mask):
            phase_update_t0 = time.perf_counter()
            existing_positions = np.flatnonzero(existing_mask)
            cached_positions = old_positions[existing_positions]
            existing_faces = selected[existing_positions]
            old_selected_weights = state.effective_weights[
                existing_faces
            ].copy()
            new_selected_weights = effective_weights[existing_faces]
            weight_scale = np.divide(
                new_selected_weights,
                np.maximum(old_selected_weights, 1e-18),
            )
            coefficients[:, existing_positions] = (
                self._phase_update_coefficients(
                    state.coefficients[:, cached_positions]
                    * weight_scale[None, :],
                    old_lengths_m=state.path_lengths_m[:, cached_positions],
                    new_lengths_m=path_lengths[:, existing_positions],
                    old_spreads_m2=(
                        state.spread_products_m2[:, cached_positions]),
                    new_spreads_m2=spreads[:, existing_positions],
                    frequency_hz=frequency_hz,
                )
            )
            phase_center_contrib[existing_positions] = (
                self._phase_update_coefficients(
                    (state.phase_center_contributions[cached_positions]
                     * weight_scale)[None, :],
                    old_lengths_m=(
                        state.phase_center_lengths_m[cached_positions]
                    )[None, :],
                    new_lengths_m=phase_lengths[existing_positions][None, :],
                    old_spreads_m2=(
                        state.phase_center_spreads_m2[cached_positions]
                    )[None, :],
                    new_spreads_m2=phase_spreads[existing_positions][None, :],
                    frequency_hz=frequency_hz,
                ).reshape(-1)
            )
            add_profile(
                "po_incremental_phase_update",
                time.perf_counter() - phase_update_t0,
            )

        drift_check_t0 = time.perf_counter()
        recompute_mask = ~existing_mask
        exact_centroids = selected_centroids.copy()
        exact_normals = selected_normals.copy()
        exact_areas = selected_areas.copy()
        if np.any(existing_mask):
            existing_positions = np.flatnonzero(existing_mask)
            cached_positions = old_positions[existing_positions]
            exact_centroids[existing_positions] = (
                state.exact_face_centroids[cached_positions]
            )
            exact_normals[existing_positions] = (
                state.exact_face_normals[cached_positions]
            )
            exact_areas[existing_positions] = (
                state.exact_face_areas[cached_positions]
            )
            normal_cos = np.einsum(
                "ij,ij->i",
                np.asarray(state.exact_face_normals[cached_positions],
                           dtype=float),
                selected_normals[existing_positions],
            )
            normal_cos = np.clip(normal_cos, -1.0, 1.0)
            if normal_threshold_rad <= 0.0:
                recompute_mask[existing_positions] = True
            else:
                recompute_mask[existing_positions] = (
                    normal_cos < np.cos(normal_threshold_rad)
                )
            if (centroid_displacement_threshold_m is not None
                    and centroid_displacement_threshold_m >= 0.0):
                centroid_shift = np.linalg.norm(
                    selected_centroids[existing_positions]
                    - state.exact_face_centroids[cached_positions],
                    axis=1,
                )
                recompute_mask[existing_positions] |= (
                    centroid_shift > centroid_displacement_threshold_m
                )
            if (area_relative_threshold is not None
                    and area_relative_threshold >= 0.0):
                reference_area = np.maximum(
                    np.abs(state.exact_face_areas[cached_positions]),
                    1e-18,
                )
                area_relative_change = np.abs(
                    selected_areas[existing_positions]
                    - state.exact_face_areas[cached_positions]
                ) / reference_area
                recompute_mask[existing_positions] |= (
                    area_relative_change > area_relative_threshold
                )
        recompute_positions = np.flatnonzero(recompute_mask)
        far_field_max_edge_m = state.far_field_max_edge_m
        far_field_nearest_distance_m = state.far_field_nearest_distance_m
        if (
            po_integration_mode == "parent_face_far_field_analytic"
            and (refresh_visibility or far_field_max_edge_m is None)
        ):
            far_field_max_edge_m, far_field_nearest_distance_m = (
                parent_face_far_field_edge_rule(
                    vertices=vertices,
                    faces=faces,
                    tx_positions=tx_positions,
                    rx_positions=rx_positions,
                    frequency_hz=frequency_hz,
                )
            )
        add_profile(
            "po_incremental_drift_check",
            time.perf_counter() - drift_check_t0,
        )

        if recompute_positions.size > 0:
            exact_recompute_t0 = time.perf_counter()
            recompute_faces = selected[recompute_positions]
            phase_contrib, far_field_max_edge_m, far_field_nearest_distance_m = (
                self._exact_po_parent_face_contributions(
                    po_integration_mode=po_integration_mode,
                    vertices=vertices,
                    faces=faces,
                    centroids=centroids,
                    normals=normals,
                    areas=areas,
                    recompute_faces=recompute_faces,
                    effective_weights=effective_weights,
                    tx_positions=tx_positions,
                    rx_positions=rx_positions,
                    frequency_hz=frequency_hz,
                    material=material,
                    po_quadrature_phase_span_scale_rad=(
                        po_quadrature_phase_span_scale_rad),
                    po_quadrature_max_refinement_depth=(
                        po_quadrature_max_refinement_depth),
                    po_quadrature_max_subfaces_per_parent=(
                        po_quadrature_max_subfaces_per_parent),
                    far_field_max_edge_m=far_field_max_edge_m,
                    far_field_nearest_distance_m=far_field_nearest_distance_m,
                    progress_callback=progress_callback,
                    compute_backend=compute_backend,
                    compute_precision=compute_precision,
                )
            )
            # Approximation: recompute exact PO only at the phase center for
            # drifted/new faces, then project to virtual channels using
            # path-length phase, spreading, and per-channel delays.
            projected_coefficients, _, _, _ = (
                project_phase_center_contributions_to_virtual_channels(
                    phase_contrib,
                    selected_centroids[recompute_positions],
                    tx_positions,
                    rx_positions,
                    frequency_hz,
                    virtual_channel_tx_indices=virtual_channel_tx_indices,
                    virtual_channel_rx_indices=virtual_channel_rx_indices,
                    virtual_channel_order=virtual_channel_order,
                    precision=compute_precision,
                )
            )
            coefficients[:, recompute_positions] = projected_coefficients
            phase_center_contrib[recompute_positions] = phase_contrib
            exact_centroids[recompute_positions] = (
                selected_centroids[recompute_positions]
            )
            exact_normals[recompute_positions] = (
                selected_normals[recompute_positions]
            )
            exact_areas[recompute_positions] = selected_areas[
                recompute_positions
            ]
            add_profile(
                "po_incremental_exact_recompute",
                time.perf_counter() - exact_recompute_t0,
            )

        state_update_t0 = time.perf_counter()
        state.face_indices = selected.copy()
        state.coefficients = coefficients
        if path_lengths.shape[0] == 1:
            state.delays_s = (phase_lengths / c).astype(state.delays_s.dtype)
        else:
            state.delays_s = (path_lengths / c).astype(state.delays_s.dtype)
        state.phase_center_contributions = phase_center_contrib
        state.face_centroids = selected_centroids.copy()
        state.face_normals = selected_normals.copy()
        state.face_areas = selected_areas.copy()
        state.exact_face_centroids = exact_centroids.copy()
        state.exact_face_normals = exact_normals.copy()
        state.exact_face_areas = exact_areas.copy()
        state.path_lengths_m = path_lengths
        state.spread_products_m2 = spreads
        state.phase_center_lengths_m = phase_lengths
        state.phase_center_spreads_m2 = phase_spreads
        state.raw_fractional_visibility = raw_visibility.copy()
        state.visible_sample_counts = visible_sample_counts.copy()
        state.temporal_fade_weights = fade_weights
        state.latched_fractional_visibility = latched_visibility
        state.effective_weights = effective_weights
        state.visibility_state = FaceVisibilityState(
            was_visible=raw_visibility > 0.0,
            fade_weights=fade_weights.copy(),
            latched_fractional_visibility=latched_visibility.copy(),
        )
        state.far_field_max_edge_m = far_field_max_edge_m
        state.far_field_nearest_distance_m = far_field_nearest_distance_m
        state.total_area = float(np.sum(areas))
        if state.total_area > 0.0:
            state.effective_visible_area_fraction = float(
                np.sum(areas * state.effective_weights) / state.total_area
            )
        else:
            state.effective_visible_area_fraction = 0.0

        result = POChannelResult(
            coefficients=state.coefficients,
            delays_s=state.delays_s,
            face_indices=state.face_indices,
            face_centroids=state.face_centroids,
            face_normals=state.face_normals,
            face_areas=state.face_areas,
            total_area=state.total_area,
            effective_visible_area_fraction=(
                state.effective_visible_area_fraction),
            raw_fractional_visibility=state.raw_fractional_visibility,
            latched_fractional_visibility=state.latched_fractional_visibility,
            temporal_fade_weights=state.temporal_fade_weights,
            effective_weights=state.effective_weights,
            visible_sample_counts=state.visible_sample_counts,
            sample_count_per_face=state.sample_count_per_face,
            phase_center_contributions=state.phase_center_contributions,
            phase_center_channel=complex(
                np.sum(state.phase_center_contributions)),
            visibility_state=state.visibility_state,
            far_field_max_edge_m=state.far_field_max_edge_m,
            far_field_nearest_distance_m=state.far_field_nearest_distance_m,
        )
        add_profile(
            "po_incremental_state_update",
            time.perf_counter() - state_update_t0,
        )
        return result, state, int(recompute_positions.size)

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
        po_quadrature_max_refinement_depth: int,
        po_quadrature_max_subfaces_per_parent: int,
        far_field_max_edge_m: float | None,
        far_field_nearest_distance_m: float | None,
        progress_callback,
        compute_backend: ComputeBackend,
        compute_precision: ComputePrecision,
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
            )
        elif po_integration_mode == "parent_face_far_field_analytic":
            (
                far_field_centroids,
                far_field_normals,
                far_field_areas,
                far_field_parent_indices,
                far_field_triangles,
                far_field_max_edge_m,
                far_field_nearest_distance_m,
            ) = parent_face_far_field_analytic_geometry(
                vertices=vertices,
                faces=faces,
                selected_parent_indices=sorted_faces,
                tx_positions=tx_positions,
                rx_positions=rx_positions,
                frequency_hz=frequency_hz,
                max_refinement_depth=po_quadrature_max_refinement_depth,
                max_subfaces_per_parent=po_quadrature_max_subfaces_per_parent,
                far_field_max_edge_m=far_field_max_edge_m,
                far_field_nearest_distance_m=far_field_nearest_distance_m,
                progress_callback=progress_callback,
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

        coefficients = np.asarray(coefficients)
        k0 = 2.0 * pi * frequency_hz / c
        # Keep PO phase updates in the dechirped ADC convention: increasing
        # path length gives a positive slow-time phase slope.
        phase = np.exp(1j * k0 * (new_lengths_m - old_lengths_m))
        spreading = old_spreads_m2 / np.maximum(new_spreads_m2, 1e-18)
        return (coefficients * phase * spreading).astype(coefficients.dtype)

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
    ) -> np.ndarray:
        """Synthesizes one chirp from path arrays and an explicit valid mask."""

        return self._synthesize_adc(
            a,
            tau,
            valid,
            fmcw,
            backend=self.adc_compute_backend if backend is None else backend,
            precision=self.adc_compute_precision if precision is None else precision,
        )

    def _trace(self, scene):
        """Invokes the configured Sionna path solver with simulator RT settings."""

        self.check_cancelled()
        if self.path_solver is None:
            from sionna.rt import PathSolver

            self.path_solver = PathSolver()
        paths = self.path_solver(
            scene,
            max_depth=self.max_depth,
            max_num_paths_per_src=self.max_num_paths_per_src,
            samples_per_src=self.samples_per_src,
            synthetic_array=True,
            los=True,
            specular_reflection=True,
            diffuse_reflection=self.diffuse_reflection,
            refraction=self.refraction,
            diffraction=False,
            seed=self.seed)
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

        return self._path_arrays(
            paths,
            targets,
            virtual_channel_order=virtual_channel_order,
            virtual_channel_tx_indices=virtual_channel_tx_indices,
            virtual_channel_rx_indices=virtual_channel_rx_indices,
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
            raise RuntimeError("Expected CIR coefficients with shape"
                               " [rx, rx_ant, tx, tx_ant, paths, time]")
        a = a[0, :, 0, :, :, 0]  # [rx_ant, tx_ant, paths]
        tau = tau[0, 0, :]       # [paths]
        valid = tau >= 0.0

        if self.coupling_mode == "one_target_bounce" and tau.size > 0:
            valid &= self._one_target_bounce_mask(paths, targets)
        if (not self.human_specular_reflection) and tau.size > 0:
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
    _torch_cuda_is_available = staticmethod(_torch_cuda_is_available)
    _use_torch_backend = staticmethod(_use_torch_backend)
    _torch_mps_available_for_precision = staticmethod(
        _torch_mps_available_for_precision
    )
    _torch_device_and_dtypes = staticmethod(_torch_device_and_dtypes)
    _synthesize_adc_torch = staticmethod(_synthesize_adc_torch)

    single_target_only_bounce_mask = staticmethod(single_target_only_bounce_mask)
    _single_target_only_bounce_mask = staticmethod(_single_target_only_bounce_mask)
    _target_interaction_counts = staticmethod(_target_interaction_counts)
    target_object_ids = staticmethod(target_object_ids)
    _target_object_ids = staticmethod(_target_object_ids)
    target_specular_path_mask_from_arrays = staticmethod(
        target_specular_path_mask_from_arrays
    )
    _target_specular_path_mask_from_arrays = staticmethod(
        _target_specular_path_mask_from_arrays
    )
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
        jumps = np.abs(np.diff(unwrapped, axis=0)).reshape(
            (slow_time.shape[0] - 1, -1)
        )
        out.reshape(-1)[1:] = np.max(jumps, axis=1)
        return out

    @staticmethod
    def _coherent_channel(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Returns the coherent sum of valid path coefficients per channel."""

        a = np.asarray(a, dtype=np.complex128)
        valid = np.asarray(valid, dtype=bool)
        if a.ndim != 2:
            raise ValueError("a must have shape [num_virtual_channels,"
                             " num_paths]")
        if valid.shape != (a.shape[1],):
            raise ValueError("valid must have shape [num_paths]")
        if not np.any(valid):
            return np.zeros(a.shape[0], dtype=np.complex128)
        return np.sum(a[:, valid], axis=1)
