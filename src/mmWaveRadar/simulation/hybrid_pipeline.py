# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Reusable hybrid RT-path/PO pipeline helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import time
from typing import Callable

import numpy as np
from scipy.constants import c

from .calibration import (
    POCalibrationConfig,
    calibration_gain_array,
    fit_first_pose_adc_calibration,
    fit_sequence_path_power_calibration,
)
from .channel_gain import apply_path_hardware_gain
from .env_human_coupling import (
    compute_env_human_coupling_channels,
    extract_rt_reflector_bank,
)
from .physical_optics import HumanVisibilityScene
from ..radar import RadarSensor
from .types import (
    HybridStaticEnvPOConfig,
    HumanPOMobilityConfig,
    RadarCube,
    SensingMetadata,
)
from .tdm import (
    collapse_tdm_cube,
    collapse_tdm_diagnostic,
    collapse_tdm_first_frame_adc,
    expand_tdm_cube,
    expanded_tdm_radar,
    tdm_expansion_required,
)
from .simulator import MmWaveRadarSimulator
from .static_blocking import (
    PathVisibilityState,
    blocked_path_visibility,
    extract_static_path_bank,
)
from .rt_coherence import (
    extract_coherent_rt_path_bank,
    match_coherent_rt_path_banks,
    synthesize_coherent_rt_transition,
    update_coherent_rt_path_bank,
)
from ..targets import MeshTarget


@dataclass(frozen=True)
class HybridCouplingRecomputeResult:
    """Result of recomputing hybrid human/environment coupling from a base cube."""

    cube: RadarCube
    wall_time_s: float
    num_chirps: int
    coupling_wall_time_s: float
    reflector_count: int


@dataclass(frozen=True)
class CalibratedHybridPOResult:
    """Result of the calibrated hybrid PO algorithm flow."""

    cube: RadarCube
    base_cube: RadarCube
    coupling_result: HybridCouplingRecomputeResult
    base_wall_time_s: float
    base_num_chirps: int
    calibration_config: POCalibrationConfig
    human_po_config: HumanPOMobilityConfig
    base_config: HybridStaticEnvPOConfig
    hybrid_config: HybridStaticEnvPOConfig

    @property
    def wall_time_s(self) -> float:
        """End-to-end wall time for the calibrated coupling pass in seconds."""

        return self.coupling_result.wall_time_s

    @property
    def num_chirps(self) -> int:
        """Number of chirps synthesized by the calibrated hybrid run."""

        return self.coupling_result.num_chirps

    @property
    def coupling_wall_time_s(self) -> float:
        """Wall time spent in environment-human coupling recomputation."""

        return self.coupling_result.coupling_wall_time_s

    @property
    def reflector_count(self) -> int:
        """Number of reflector candidates retained by the coupling pass."""

        return self.coupling_result.reflector_count

    @property
    def calibration(self) -> dict:
        """PO calibration metadata copied from the baseline cube."""

        return dict(self.base_cube.metadata.po_calibration or {})


@dataclass(frozen=True)
class FullRTMobilityBaselineResult:
    """Result of tracing the full scene over the requested motion window."""

    cube: RadarCube
    human_reference_adc: np.ndarray
    human_reference_path_counts: np.ndarray
    human_touch_path_counts: np.ndarray
    one_human_touch_path_counts: np.ndarray
    single_human_only_path_counts: np.ndarray
    human_env_coupled_path_counts: np.ndarray
    human_multi_touch_path_counts: np.ndarray
    path_depth_histogram: np.ndarray
    total_path_power: np.ndarray
    human_touch_path_power: np.ndarray
    one_human_touch_path_power: np.ndarray
    single_human_only_path_power: np.ndarray
    human_env_coupled_path_power: np.ndarray
    human_multi_touch_path_power: np.ndarray
    retrace_count: int


RTMobilityBaselineResult = FullRTMobilityBaselineResult


def _collapse_tdm_full_rt_result(
    result: FullRTMobilityBaselineResult,
    radar: RadarSensor,
) -> FullRTMobilityBaselineResult:
    """Packs a physical-slot full-RT result into per-channel slow time."""

    cube = collapse_tdm_cube(result.cube, radar)

    def first_frame(value: np.ndarray, name: str) -> np.ndarray:
        return collapse_tdm_diagnostic(
            np.asarray(value)[None, ...],
            radar,
            name=name,
        )[0]

    metadata = cube.metadata
    return replace(
        result,
        cube=cube,
        human_reference_adc=collapse_tdm_first_frame_adc(
            result.human_reference_adc,
            radar,
        ),
        human_reference_path_counts=first_frame(
            result.human_reference_path_counts,
            "human_reference_path_counts",
        ),
        human_touch_path_counts=metadata.human_touch_path_counts,
        one_human_touch_path_counts=metadata.one_human_touch_path_counts,
        single_human_only_path_counts=metadata.single_human_only_path_counts,
        human_env_coupled_path_counts=metadata.human_env_coupled_path_counts,
        human_multi_touch_path_counts=metadata.human_multi_touch_path_counts,
        path_depth_histogram=metadata.path_depth_histogram,
        total_path_power=metadata.total_path_power,
        human_touch_path_power=metadata.human_touch_path_power,
        one_human_touch_path_power=metadata.one_human_touch_path_power,
        single_human_only_path_power=metadata.single_human_only_path_power,
        human_env_coupled_path_power=metadata.human_env_coupled_path_power,
        human_multi_touch_path_power=metadata.human_multi_touch_path_power,
    )


@dataclass(frozen=True)
class _FrameRangeMeshSequence:
    """Time-rebased mesh view for a selected radar-frame range."""

    base_sequence: object
    start_time_s: float
    duration_s: float

    @property
    def times(self) -> np.ndarray:
        """Local two-sample time axis spanning the selected frame range."""

        if self.duration_s <= 0.0:
            return np.asarray([0.0], dtype=float)
        return np.asarray([0.0, float(self.duration_s)], dtype=float)

    @property
    def faces(self):
        """Faces from the wrapped mesh sequence."""

        return self.base_sequence.faces

    @property
    def vertex_count(self) -> int:
        """Number of vertices in each mesh frame."""

        count = getattr(self.base_sequence, "vertex_count", None)
        if count is not None:
            return int(count)
        return int(self.vertices_at(0.0).shape[0])

    @property
    def face_count(self) -> int:
        """Number of mesh faces in the wrapped sequence."""

        count = getattr(self.base_sequence, "face_count", None)
        if count is not None:
            return int(count)
        return int(np.asarray(self.faces).shape[0])

    @property
    def vertices(self) -> np.ndarray:
        """Vertices sampled at the local start and end of the frame range."""

        return np.stack([self.vertices_at(float(t)) for t in self.times], axis=0)

    def vertices_at(self, time: float) -> np.ndarray:
        """Returns vertices at a local time clipped to the selected range."""

        local_time = min(max(float(time), 0.0), max(float(self.duration_s), 0.0))
        return self.base_sequence.vertices_at(float(self.start_time_s) + local_time)

    def max_vertex_displacement(
        self,
        time: float,
        reference_vertices: np.ndarray,
    ) -> float:
        """Returns the largest vertex displacement from ``reference_vertices``."""

        vertices = self.vertices_at(time)
        ref = np.asarray(reference_vertices, dtype=np.float32)
        if ref.shape != vertices.shape:
            raise ValueError("reference_vertices shape mismatch")
        return float(np.max(np.linalg.norm(vertices - ref, axis=1)))


@dataclass(frozen=True)
class _StaticEnvironmentPass:
    """Static RT, dynamic blocking, and RT-derived reflector candidates."""

    static_unblocked_adc: np.ndarray
    static_blocked_adc: np.ndarray
    visibility_weights: np.ndarray
    static_path_counts: np.ndarray
    blocked_counts: np.ndarray
    mean_visibility: np.ndarray
    static_bank: object
    reflector_bank: object
    runtime_profile_s: dict[str, float]
    runtime_profile_counts: dict[str, int]
    wall_time_s: float


@dataclass(frozen=True)
class _CouplingPass:
    """Human/environment coupling terms and diagnostics."""

    human_env_adc: np.ndarray
    env_human_adc: np.ndarray
    human_env_counts: np.ndarray
    env_human_counts: np.ndarray
    human_env_min_lengths: np.ndarray
    human_env_mean_lengths: np.ndarray
    human_env_max_lengths: np.ndarray
    env_human_min_lengths: np.ndarray
    env_human_mean_lengths: np.ndarray
    env_human_max_lengths: np.ndarray
    runtime_profile_s: dict[str, float]
    runtime_profile_counts: dict[str, int]
    wall_time_s: float


def _runtime_alias_key(key: str) -> str | None:
    """Maps legacy ``full_rt_*`` profile keys to shorter ``rt_*`` aliases."""

    if not key.startswith("full_rt_"):
        return None
    return "rt_" + key[len("full_rt_"):]


def _path_length_stats(valid: np.ndarray, delays_s: np.ndarray):
    """Returns min/mean/max valid path lengths in meters."""

    delays = np.asarray(delays_s)
    if delays.ndim == 2:
        delays = np.mean(delays, axis=0)
    lengths = c * delays[np.asarray(valid, dtype=bool)]
    if lengths.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.min(lengths)), float(np.mean(lengths)), float(np.max(lengths))


def _zeroed_metadata_arrays(times_shape):
    """Allocates zero/NaN metadata arrays matching a frame/chirp time shape."""

    return (
        np.zeros(times_shape, dtype=np.int64),
        np.zeros(times_shape, dtype=np.int64),
        np.full(times_shape, np.nan, dtype=np.float64),
        np.full(times_shape, np.nan, dtype=np.float64),
        np.full(times_shape, np.nan, dtype=np.float64),
        np.full(times_shape, np.nan, dtype=np.float64),
        np.full(times_shape, np.nan, dtype=np.float64),
        np.full(times_shape, np.nan, dtype=np.float64),
    )


def _reset_runtime_keys(profile_s: dict[str, float], profile_counts: dict[str, int]):
    """Initializes hybrid runtime profile keys on existing metadata dictionaries."""

    for key in (
        "hybrid_static_trace",
        "hybrid_segment_extraction",
        "hybrid_reflector_extraction",
        "hybrid_coupling_channel",
        "hybrid_coupling_setup",
        "hybrid_human_env_channel",
        "hybrid_env_human_channel",
        "hybrid_coupling_adc_synthesis",
        "hybrid_human_env_adc_synthesis",
        "hybrid_env_human_adc_synthesis",
    ):
        profile_s[key] = 0.0
        profile_counts[key] = 0


def _require_components(base_cube: RadarCube, keys: set[str]):
    """Raises when a loaded base cube lacks required component arrays."""

    if base_cube.components is None:
        raise RuntimeError("base cube was loaded without components")
    missing = keys.difference(base_cube.components)
    if missing:
        raise RuntimeError(f"base cube is missing components: {sorted(missing)}")


def _print_reflector_summary(reflector_bank, *, max_rows: int = 8):
    """Prints a compact summary of RT-derived environment reflectors."""

    reflector_count = int(reflector_bank.vertices.shape[0])
    print(f"RT-derived coupling candidates: {reflector_count}", flush=True)
    if reflector_count <= 0:
        return
    reflector_bounds = [
        (
            reflector_bank.object_names[idx],
            np.array2string(reflector_bank.vertices[idx].min(axis=0), precision=3),
            np.array2string(reflector_bank.vertices[idx].max(axis=0), precision=3),
        )
        for idx in range(min(reflector_count, max_rows))
    ]
    print("RT reflector world bounds [name, min_xyz, max_xyz]:", reflector_bounds, flush=True)
    if reflector_bank.reflection_gains is not None:
        gains = np.asarray(reflector_bank.reflection_gains)[:min(reflector_count, max_rows)]
        print(
            "RT-calibrated reflector gains [name, |gain|, phase_deg]:",
            [
                (
                    reflector_bank.object_names[idx],
                    float(np.abs(gains[idx])),
                    float(np.angle(gains[idx], deg=True)),
                )
                for idx in range(gains.size)
            ],
            flush=True,
        )


def _run_full_rt_coherent_transition_baseline(
    *,
    scene,
    radar: RadarSensor,
    targets: list[MeshTarget],
    simulator: MmWaveRadarSimulator,
    num_frames: int,
) -> FullRTMobilityBaselineResult:
    """Runs full-scene RT while preserving coherent paths between retraces."""

    fmcw = radar.fmcw
    num_chirps = fmcw.num_chirps_per_frame
    periodic_retrace_period_chirps = (
        simulator.effective_periodic_retrace_period_chirps(fmcw)
    )
    num_adc = fmcw.num_adc_samples
    num_vc = radar.hardware.num_virtual_channels
    _, adc_complex_dtype = simulator.numpy_dtypes_for_precision(
        simulator.adc_compute_precision
    )
    adc = np.zeros(
        (num_frames, num_chirps, num_adc, num_vc),
        dtype=adc_complex_dtype,
    )
    times = np.zeros((num_frames, num_chirps), dtype=float)
    path_counts = np.zeros((num_frames, num_chirps), dtype=np.int64)
    human_touch_path_counts = np.zeros_like(path_counts)
    one_human_touch_path_counts = np.zeros_like(path_counts)
    single_human_only_path_counts = np.zeros_like(path_counts)
    human_env_coupled_path_counts = np.zeros_like(path_counts)
    human_multi_touch_path_counts = np.zeros_like(path_counts)
    transition_persistent_counts = np.zeros_like(path_counts)
    transition_birth_counts = np.zeros_like(path_counts)
    transition_death_counts = np.zeros_like(path_counts)
    transition_alpha = np.zeros((num_frames, num_chirps), dtype=float)
    transition_old_weights = np.ones((num_frames, num_chirps), dtype=float)
    transition_new_weights = np.zeros((num_frames, num_chirps), dtype=float)
    path_depth_histogram = np.zeros(
        (num_frames, num_chirps, int(simulator.max_depth) + 1),
        dtype=np.int64,
    )
    total_path_power = np.zeros((num_frames, num_chirps), dtype=np.float64)
    human_touch_path_power = np.zeros_like(total_path_power)
    one_human_touch_path_power = np.zeros_like(total_path_power)
    single_human_only_path_power = np.zeros_like(total_path_power)
    human_env_coupled_path_power = np.zeros_like(total_path_power)
    human_multi_touch_path_power = np.zeros_like(total_path_power)
    max_displacements = np.zeros((num_frames, num_chirps), dtype=float)
    human_reference_adc = np.zeros(
        (num_chirps, num_adc, num_vc),
        dtype=adc_complex_dtype,
    )
    human_reference_path_counts = np.zeros(num_chirps, dtype=np.int64)
    threshold = fmcw.wavelength * simulator.retrace_displacement_fraction
    transition_config = simulator.rt_coherent_transition_config
    transition_matching_enabled = bool(transition_config.enabled)
    transition_crossfade_enabled = simulator.rt_transition_crossfade_enabled()
    runtime_profile_s = {
        "full_rt_trace": 0.0,
        "full_rt_coherent_bank_extract": 0.0,
        "full_rt_coherent_transition_match": 0.0,
        "full_rt_coherent_bank_update": 0.0,
        "full_rt_path_arrays": 0.0,
        "full_rt_interaction_counts": 0.0,
        "full_rt_adc_synthesis": 0.0,
    }
    runtime_profile_counts = dict.fromkeys(runtime_profile_s, 0)

    def add_runtime_profile(key: str, elapsed_s: float):
        """Accumulates elapsed time and count for one full-RT stage."""

        elapsed_s = float(elapsed_s)
        runtime_profile_s[key] = runtime_profile_s.get(key, 0.0) + elapsed_s
        runtime_profile_counts[key] = runtime_profile_counts.get(key, 0) + 1
        alias_key = _runtime_alias_key(key)
        if alias_key is not None:
            runtime_profile_s[alias_key] = runtime_profile_s.get(alias_key, 0.0) + elapsed_s
            runtime_profile_counts[alias_key] = runtime_profile_counts.get(alias_key, 0) + 1

    reference_vertices = [None for _ in targets]
    events: list[dict] = []
    event_flat_indices: list[int] = []
    event_anchor_vertices: list[list[np.ndarray]] = []
    for frame in range(num_frames):
        for chirp in range(num_chirps):
            flat_index = frame * num_chirps + chirp
            chirp_time = fmcw.chirp_time(frame, chirp)
            times[frame, chirp] = chirp_time
            if simulator.retrace_once_per_frame:
                need_retrace = simulator.periodic_chirp_retrace_due(
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
            if not simulator.retrace_once_per_frame:
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
                "reason": simulator.retrace_event_reason(
                    first_event=len(events) == 0,
                    period_chirps=periodic_retrace_period_chirps,
                    num_chirps_per_frame=num_chirps,
                ),
                "max_displacement": max_disp,
                "threshold": threshold,
                "periodic_retrace_period_chirps": (
                    periodic_retrace_period_chirps
                    if simulator.retrace_once_per_frame else None
                ),
                "coherent_path_update": True,
                "human_specular_reflection": bool(
                    simulator.human_specular_reflection
                ),
                "human_specular_policy": (
                    "allow" if simulator.human_specular_reflection else "drop"
                ),
            })

    event_banks = []
    event_counts = []
    for event, anchor_vertices in zip(events, event_anchor_vertices):
        for target, vertices in zip(targets, anchor_vertices):
            target.update_to_time(
                float(event["time"]),
                vertices=vertices,
            )
            target.reference_vertices = vertices.copy()
        trace_t0 = time.perf_counter()
        paths = simulator.trace_scene(scene)
        add_runtime_profile("full_rt_trace", time.perf_counter() - trace_t0)
        counts_t0 = time.perf_counter()
        event_counts.append(
            simulator.target_interaction_counts(
                paths,
                targets,
            )
        )
        add_runtime_profile(
            "full_rt_interaction_counts",
            time.perf_counter() - counts_t0,
        )
        extract_t0 = time.perf_counter()
        bank = extract_coherent_rt_path_bank(
            paths,
            radar,
            targets,
            virtual_channel_order=radar.hardware.virtual_channel_order,
            anchor_target_vertices=anchor_vertices,
        )
        if not simulator.human_specular_reflection:
            bank = simulator.drop_target_specular_bank(
                bank,
                targets,
            )
        event_banks.append(bank)
        add_runtime_profile(
            "full_rt_coherent_bank_extract",
            time.perf_counter() - extract_t0,
        )

    transitions = []
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
        match_t0 = time.perf_counter()
        transition = match_coherent_rt_path_banks(
            event_banks[event_index],
            event_banks[event_index + 1],
            event_anchor_vertices[event_index + 1],
            wavelength_m=fmcw.wavelength,
            match_delay_tolerance_fraction=(
                transition_config.match_delay_tolerance_fraction
            ),
            match_separation_fraction=transition_config.match_separation_fraction,
            backend=simulator.compute_backend,
        )
        add_runtime_profile(
            "full_rt_coherent_transition_match",
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

    frame_retrace_counts = np.zeros(num_frames, dtype=np.int64)
    for event in events:
        frame_retrace_counts[int(event["frame"])] += 1

    frame_start_s = time.perf_counter()
    for frame in range(num_frames):
        if frame > 0:
            simulator.log_frame_progress(
                mode="rt_baseline",
                frame=frame - 1,
                num_frames=num_frames,
                frame_start_s=frame_start_s,
                extra=f"retraces={int(frame_retrace_counts[frame - 1])}",
            )
        for chirp in range(num_chirps):
            flat_index = frame * num_chirps + chirp
            event_index = int(
                np.searchsorted(event_flat_indices, flat_index, side="right") - 1
            )
            if event_index < 0:
                raise RuntimeError("coherent RT path bank was not initialized")
            event_chirp = flat_index == event_flat_indices[event_index]
            if simulator.retrace_once_per_frame and event_chirp:
                current_vertices = event_anchor_vertices[event_index]
            else:
                current_vertices = [
                    target.mesh_sequence.vertices_at(times[frame, chirp])
                    for target in targets
                ]
            if simulator.retrace_once_per_frame and not event_chirp:
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
                event_banks[event_index],
                current_vertices,
                backend=simulator.compute_backend,
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
                    backend=simulator.compute_backend,
                )
            synth = synthesize_coherent_rt_transition(
                current_update,
                next_update,
                transition,
                interval_chirps=interval_chirps,
                interval_offset=interval_offset,
            )
            add_runtime_profile(
                "full_rt_coherent_bank_update",
                time.perf_counter() - update_t0,
            )
            a = synth.coefficients
            tau = synth.delays_s
            valid = synth.valid
            current_target_counts, current_total_counts = event_counts[event_index]
            target_counts = np.zeros(synth.source_indices.shape, dtype=np.int64)
            total_counts = np.zeros(synth.source_indices.shape, dtype=np.int64)
            next_source = synth.source_is_new
            current_source = ~next_source
            if np.any(current_source):
                target_counts[current_source] = current_target_counts[
                    synth.source_indices[current_source]
                ]
                total_counts[current_source] = current_total_counts[
                    synth.source_indices[current_source]
                ]
            if np.any(next_source) and event_index + 1 < len(event_counts):
                next_target_counts, next_total_counts = event_counts[event_index + 1]
                target_counts[next_source] = next_target_counts[
                    synth.source_indices[next_source]
                ]
                total_counts[next_source] = next_total_counts[
                    synth.source_indices[next_source]
                ]

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

            human_touch = valid & (target_counts > 0)
            one_human_touch = valid & (target_counts == 1)
            single_human_only = one_human_touch & (total_counts == 1)
            human_env_coupled = human_touch & (total_counts > target_counts)
            human_multi_touch = valid & (target_counts > 1)
            path_power = np.sum(
                np.abs(np.asarray(a, dtype=np.complex128)) ** 2,
                axis=0,
            ) if a.size else np.zeros(valid.shape, dtype=np.float64)
            path_power = np.nan_to_num(
                path_power,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            total_path_power[frame, chirp] = float(np.sum(path_power[valid]))
            human_touch_path_power[frame, chirp] = float(
                np.sum(path_power[human_touch])
            )
            one_human_touch_path_power[frame, chirp] = float(
                np.sum(path_power[one_human_touch])
            )
            single_human_only_path_power[frame, chirp] = float(
                np.sum(path_power[single_human_only])
            )
            human_env_coupled_path_power[frame, chirp] = float(
                np.sum(path_power[human_env_coupled])
            )
            human_multi_touch_path_power[frame, chirp] = float(
                np.sum(path_power[human_multi_touch])
            )
            human_touch_path_counts[frame, chirp] = int(
                np.count_nonzero(human_touch)
            )
            one_human_touch_path_counts[frame, chirp] = int(
                np.count_nonzero(one_human_touch)
            )
            single_human_only_path_counts[frame, chirp] = int(
                np.count_nonzero(single_human_only)
            )
            human_env_coupled_path_counts[frame, chirp] = int(
                np.count_nonzero(human_env_coupled)
            )
            human_multi_touch_path_counts[frame, chirp] = int(
                np.count_nonzero(human_multi_touch)
            )
            if valid.size > 0:
                depth_bins = path_depth_histogram.shape[-1]
                valid_depths = np.clip(total_counts[valid], 0, depth_bins - 1)
                path_depth_histogram[frame, chirp, :] = np.bincount(
                    valid_depths,
                    minlength=depth_bins,
                )[:depth_bins]
            adc_t0 = time.perf_counter()
            adc[frame, chirp] = simulator.synthesize_adc(
                a,
                tau,
                valid,
                fmcw,
                backend=simulator.adc_compute_backend,
                precision=simulator.adc_compute_precision,
            )
            add_runtime_profile(
                "full_rt_adc_synthesis",
                time.perf_counter() - adc_t0,
            )
            if frame == 0:
                human_valid = single_human_only
                human_reference_path_counts[chirp] = int(
                    single_human_only_path_counts[frame, chirp]
                )
                human_reference_adc[chirp] = simulator.synthesize_adc(
                    a,
                    tau,
                    human_valid,
                    fmcw,
                    backend=simulator.adc_compute_backend,
                    precision=simulator.adc_compute_precision,
                )

    simulator.log_frame_progress(
        mode="rt_baseline",
        frame=num_frames - 1,
        num_frames=num_frames,
        frame_start_s=frame_start_s,
        extra=f"retraces={int(frame_retrace_counts[num_frames - 1])}",
    )
    metadata = SensingMetadata(
        events=events,
        path_counts=path_counts,
        max_displacements=max_displacements,
        virtual_channel_order=radar.hardware.virtual_channel_order,
        mode="rt_baseline",
        human_touch_path_counts=human_touch_path_counts,
        one_human_touch_path_counts=one_human_touch_path_counts,
        single_human_only_path_counts=single_human_only_path_counts,
        human_env_coupled_path_counts=human_env_coupled_path_counts,
        human_multi_touch_path_counts=human_multi_touch_path_counts,
        path_depth_histogram=path_depth_histogram,
        total_path_power=total_path_power,
        human_touch_path_power=human_touch_path_power,
        one_human_touch_path_power=one_human_touch_path_power,
        single_human_only_path_power=single_human_only_path_power,
        human_env_coupled_path_power=human_env_coupled_path_power,
        human_multi_touch_path_power=human_multi_touch_path_power,
        runtime_profile_s=runtime_profile_s,
        runtime_profile_counts=runtime_profile_counts,
        rt_transition_persistent_path_counts=transition_persistent_counts,
        rt_transition_birth_path_counts=transition_birth_counts,
        rt_transition_death_path_counts=transition_death_counts,
        rt_transition_alpha=transition_alpha,
        rt_transition_old_weights=transition_old_weights,
        rt_transition_new_weights=transition_new_weights,
    )
    cube = RadarCube(adc=adc, times=times, metadata=metadata)
    return FullRTMobilityBaselineResult(
        cube=cube,
        human_reference_adc=human_reference_adc,
        human_reference_path_counts=human_reference_path_counts,
        human_touch_path_counts=human_touch_path_counts,
        one_human_touch_path_counts=one_human_touch_path_counts,
        single_human_only_path_counts=single_human_only_path_counts,
        human_env_coupled_path_counts=human_env_coupled_path_counts,
        human_multi_touch_path_counts=human_multi_touch_path_counts,
        path_depth_histogram=path_depth_histogram,
        total_path_power=total_path_power,
        human_touch_path_power=human_touch_path_power,
        one_human_touch_path_power=one_human_touch_path_power,
        single_human_only_path_power=single_human_only_path_power,
        human_env_coupled_path_power=human_env_coupled_path_power,
        human_multi_touch_path_power=human_multi_touch_path_power,
        retrace_count=int(len(event_banks)),
    )


def run_full_rt_mobility_baseline(
    *,
    scene,
    radar: RadarSensor,
    targets: list[MeshTarget],
    simulator: MmWaveRadarSimulator,
    num_frames: int,
) -> FullRTMobilityBaselineResult:
    """Runs full-scene RT over the motion window and extracts first-frame human RT.

    This intentionally stays as a separate top-level helper from the hybrid
    pipeline so mobility-specific tracing/cache optimizations can evolve here
    without entangling the PO/RT hybrid component assembly.
    """

    if scene is None:
        raise ValueError("scene is required")
    if not targets:
        raise ValueError("at least one target is required")
    num_frames = int(num_frames)
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if tdm_expansion_required(radar):
        expanded_result = run_full_rt_mobility_baseline(
            scene=scene,
            radar=expanded_tdm_radar(radar),
            targets=targets,
            simulator=simulator,
            num_frames=num_frames,
        )
        return _collapse_tdm_full_rt_result(expanded_result, radar)

    added_objects = []
    old_references = [
        None if target.reference_vertices is None else target.reference_vertices.copy()
        for target in targets
    ]
    try:
        radar.configure_scene(scene)
        for target in targets:
            obj = target.ensure_scene_object()
            was_in_scene = obj.scene is scene and obj.name in scene.objects
            if not was_in_scene:
                scene.edit(add=[obj])
                added_objects.append(obj)
            target.update_to_time(0.0)

        if simulator.mobility_mode == "rt_coherent_bank":
            return _run_full_rt_coherent_transition_baseline(
                scene=scene,
                radar=radar,
                targets=targets,
                simulator=simulator,
                num_frames=num_frames,
            )

        fmcw = radar.fmcw
        num_chirps = fmcw.num_chirps_per_frame
        periodic_retrace_period_chirps = (
            simulator.effective_periodic_retrace_period_chirps(fmcw)
        )
        num_adc = fmcw.num_adc_samples
        num_vc = radar.hardware.num_virtual_channels
        _, adc_complex_dtype = simulator.numpy_dtypes_for_precision(
            simulator.adc_compute_precision
        )
        adc = np.zeros(
            (num_frames, num_chirps, num_adc, num_vc),
            dtype=adc_complex_dtype,
        )
        times = np.zeros((num_frames, num_chirps), dtype=float)
        path_counts = np.zeros((num_frames, num_chirps), dtype=np.int64)
        human_touch_path_counts = np.zeros_like(path_counts)
        one_human_touch_path_counts = np.zeros_like(path_counts)
        single_human_only_path_counts = np.zeros_like(path_counts)
        human_env_coupled_path_counts = np.zeros_like(path_counts)
        human_multi_touch_path_counts = np.zeros_like(path_counts)
        path_depth_histogram = np.zeros(
            (num_frames, num_chirps, int(simulator.max_depth) + 1),
            dtype=np.int64,
        )
        total_path_power = np.zeros((num_frames, num_chirps), dtype=np.float64)
        human_touch_path_power = np.zeros_like(total_path_power)
        one_human_touch_path_power = np.zeros_like(total_path_power)
        single_human_only_path_power = np.zeros_like(total_path_power)
        human_env_coupled_path_power = np.zeros_like(total_path_power)
        human_multi_touch_path_power = np.zeros_like(total_path_power)
        max_displacements = np.zeros((num_frames, num_chirps), dtype=float)
        human_reference_adc = np.zeros(
            (num_chirps, num_adc, num_vc),
            dtype=adc_complex_dtype,
        )
        human_reference_path_counts = np.zeros(num_chirps, dtype=np.int64)
        reference_vertices = [None for _ in targets]
        current_target_vertices = [None for _ in targets]
        paths = None
        cached_path_arrays = None
        cached_target_counts = None
        coherent_bank = None
        coherent_updates = simulator.mobility_mode == "rt_coherent_bank"
        threshold = fmcw.wavelength * simulator.retrace_displacement_fraction
        events = []
        frame_start_s = time.perf_counter()
        total_retraces = 0
        runtime_profile_s = {
            "full_rt_trace": 0.0,
            "full_rt_coherent_bank_extract": 0.0,
            "full_rt_coherent_bank_update": 0.0,
            "full_rt_path_arrays": 0.0,
            "full_rt_interaction_counts": 0.0,
            "full_rt_adc_synthesis": 0.0,
        }
        runtime_profile_counts = dict.fromkeys(runtime_profile_s, 0)

        def add_runtime_profile(key: str, elapsed_s: float):
            """Accumulates elapsed time and count for one full-RT stage."""

            elapsed_s = float(elapsed_s)
            runtime_profile_s[key] = runtime_profile_s.get(key, 0.0) + elapsed_s
            runtime_profile_counts[key] = runtime_profile_counts.get(key, 0) + 1
            alias_key = _runtime_alias_key(key)
            if alias_key is not None:
                runtime_profile_s[alias_key] = runtime_profile_s.get(alias_key, 0.0) + elapsed_s
                runtime_profile_counts[alias_key] = runtime_profile_counts.get(alias_key, 0) + 1

        for frame in range(num_frames):
            if frame > 0:
                simulator.log_frame_progress(
                    mode="rt_baseline",
                    frame=frame - 1,
                    num_frames=num_frames,
                    frame_start_s=frame_start_s,
                    extra=f"retraces={total_retraces}",
                )
            for chirp in range(num_chirps):
                chirp_time = fmcw.chirp_time(frame, chirp)
                times[frame, chirp] = chirp_time

                max_disp = 0.0
                for target_index, target in enumerate(targets):
                    vertices = target.update_to_time(chirp_time)
                    current_target_vertices[target_index] = vertices
                    if reference_vertices[target_index] is None:
                        reference_vertices[target_index] = vertices.copy()
                    disp = np.max(np.linalg.norm(
                        vertices - reference_vertices[target_index], axis=1))
                    max_disp = max(max_disp, float(disp))
                max_displacements[frame, chirp] = max_disp

                if simulator.retrace_once_per_frame:
                    need_retrace = simulator.periodic_chirp_retrace_due(
                        frame=frame,
                        chirp=chirp,
                        num_chirps_per_frame=num_chirps,
                        period_chirps=periodic_retrace_period_chirps,
                        initialized=paths is not None,
                    )
                else:
                    need_retrace = paths is None or max_disp > threshold
                if need_retrace:
                    trace_t0 = time.perf_counter()
                    paths = simulator.trace_scene(scene)
                    add_runtime_profile("full_rt_trace", time.perf_counter() - trace_t0)
                    total_retraces += 1
                    for target_index, vertices in enumerate(current_target_vertices):
                        reference_vertices[target_index] = vertices.copy()
                    counts_t0 = time.perf_counter()
                    cached_target_counts = simulator.target_interaction_counts(
                        paths,
                        targets,
                    )
                    add_runtime_profile(
                        "full_rt_interaction_counts",
                        time.perf_counter() - counts_t0,
                    )
                    if coherent_updates:
                        extract_t0 = time.perf_counter()
                        coherent_bank = extract_coherent_rt_path_bank(
                            paths,
                            radar,
                            targets,
                            virtual_channel_order=radar.hardware.virtual_channel_order,
                            anchor_target_vertices=[
                                vertices.copy() for vertices in current_target_vertices
                            ],
                        )
                        if not simulator.human_specular_reflection:
                            coherent_bank = simulator.drop_target_specular_bank(
                                coherent_bank,
                                targets,
                            )
                        add_runtime_profile(
                            "full_rt_coherent_bank_extract",
                            time.perf_counter() - extract_t0,
                        )
                        cached_path_arrays = None
                    else:
                        arrays_t0 = time.perf_counter()
                        cached_path_arrays = simulator.path_arrays_from_paths(
                            paths,
                            targets,
                            virtual_channel_order=radar.hardware.virtual_channel_order,
                            virtual_channel_tx_indices=(
                                radar.hardware.virtual_tx_indices()
                            ),
                            virtual_channel_rx_indices=(
                                radar.hardware.virtual_rx_indices()
                            ),
                        )
                        cached_path_arrays = (
                            apply_path_hardware_gain(
                                radar.hardware,
                                cached_path_arrays[0],
                                paths=paths,
                                frequency_hz=fmcw.carrier_frequency,
                            ),
                            cached_path_arrays[1],
                            cached_path_arrays[2],
                        )
                        add_runtime_profile(
                            "full_rt_path_arrays",
                            time.perf_counter() - arrays_t0,
                        )
                    events.append({
                        "frame": frame,
                        "chirp": chirp,
                        "time": float(chirp_time),
                        "reason": simulator.retrace_event_reason(
                            first_event=len(events) == 0,
                            period_chirps=periodic_retrace_period_chirps,
                            num_chirps_per_frame=num_chirps,
                        ),
                        "max_displacement": max_disp,
                        "threshold": threshold,
                        "periodic_retrace_period_chirps": (
                            periodic_retrace_period_chirps
                            if simulator.retrace_once_per_frame else None
                        ),
                        "coherent_path_update": bool(coherent_updates),
                        "diffuse_reflection": bool(simulator.diffuse_reflection),
                        "human_specular_reflection": bool(
                            simulator.human_specular_reflection
                        ),
                        "human_specular_policy": (
                            "allow"
                            if simulator.human_specular_reflection
                            else "drop"
                        ),
                    })

                if coherent_updates:
                    if coherent_bank is None:
                        raise RuntimeError("coherent RT path bank was not initialized")
                    update_t0 = time.perf_counter()
                    coherent_update = update_coherent_rt_path_bank(
                        coherent_bank,
                        current_target_vertices,
                        backend=simulator.compute_backend,
                    )
                    add_runtime_profile(
                        "full_rt_coherent_bank_update",
                        time.perf_counter() - update_t0,
                    )
                    a = coherent_update.coefficients
                    tau = coherent_update.delays_s
                    valid = coherent_update.valid
                else:
                    if cached_path_arrays is None:
                        raise RuntimeError("cached RT path arrays were not initialized")
                    a, tau, valid = cached_path_arrays
                if cached_target_counts is None:
                    raise RuntimeError("cached RT interaction counts were not initialized")
                path_counts[frame, chirp] = int(np.count_nonzero(valid))
                target_counts, total_counts = cached_target_counts
                human_touch = valid & (target_counts > 0)
                one_human_touch = valid & (target_counts == 1)
                single_human_only = one_human_touch & (total_counts == 1)
                human_env_coupled = human_touch & (total_counts > target_counts)
                human_multi_touch = valid & (target_counts > 1)
                path_power = np.sum(
                    np.abs(np.asarray(a, dtype=np.complex128)) ** 2,
                    axis=0,
                ) if a.size else np.zeros(valid.shape, dtype=np.float64)
                path_power = np.nan_to_num(
                    path_power,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                total_path_power[frame, chirp] = float(np.sum(path_power[valid]))
                human_touch_path_power[frame, chirp] = float(
                    np.sum(path_power[human_touch])
                )
                one_human_touch_path_power[frame, chirp] = float(
                    np.sum(path_power[one_human_touch])
                )
                single_human_only_path_power[frame, chirp] = float(
                    np.sum(path_power[single_human_only])
                )
                human_env_coupled_path_power[frame, chirp] = float(
                    np.sum(path_power[human_env_coupled])
                )
                human_multi_touch_path_power[frame, chirp] = float(
                    np.sum(path_power[human_multi_touch])
                )
                human_touch_path_counts[frame, chirp] = int(
                    np.count_nonzero(human_touch)
                )
                one_human_touch_path_counts[frame, chirp] = int(
                    np.count_nonzero(one_human_touch)
                )
                single_human_only_path_counts[frame, chirp] = int(
                    np.count_nonzero(single_human_only)
                )
                human_env_coupled_path_counts[frame, chirp] = int(
                    np.count_nonzero(human_env_coupled)
                )
                human_multi_touch_path_counts[frame, chirp] = int(
                    np.count_nonzero(human_multi_touch)
                )
                if valid.size > 0:
                    depth_bins = path_depth_histogram.shape[-1]
                    valid_depths = np.clip(total_counts[valid], 0, depth_bins - 1)
                    path_depth_histogram[frame, chirp, :] = np.bincount(
                        valid_depths,
                        minlength=depth_bins,
                    )[:depth_bins]
                adc_t0 = time.perf_counter()
                adc[frame, chirp] = simulator.synthesize_adc(
                    a,
                    tau,
                    valid,
                    fmcw,
                    backend=simulator.adc_compute_backend,
                    precision=simulator.adc_compute_precision,
                )
                add_runtime_profile(
                    "full_rt_adc_synthesis",
                    time.perf_counter() - adc_t0,
                )
                if frame == 0:
                    human_valid = single_human_only
                    human_reference_path_counts[chirp] = int(
                        single_human_only_path_counts[frame, chirp]
                    )
                    human_reference_adc[chirp] = simulator.synthesize_adc(
                        a,
                        tau,
                        human_valid,
                        fmcw,
                        backend=simulator.adc_compute_backend,
                        precision=simulator.adc_compute_precision,
                    )

        simulator.log_frame_progress(
            mode="rt_baseline",
            frame=num_frames - 1,
            num_frames=num_frames,
            frame_start_s=frame_start_s,
            extra=f"retraces={total_retraces}",
        )
        metadata = SensingMetadata(
            events=events,
            path_counts=path_counts,
            max_displacements=max_displacements,
            virtual_channel_order=radar.hardware.virtual_channel_order,
            mode="rt_baseline",
            human_touch_path_counts=human_touch_path_counts,
            one_human_touch_path_counts=one_human_touch_path_counts,
            single_human_only_path_counts=single_human_only_path_counts,
            human_env_coupled_path_counts=human_env_coupled_path_counts,
            human_multi_touch_path_counts=human_multi_touch_path_counts,
            path_depth_histogram=path_depth_histogram,
            total_path_power=total_path_power,
            human_touch_path_power=human_touch_path_power,
            one_human_touch_path_power=one_human_touch_path_power,
            single_human_only_path_power=single_human_only_path_power,
            human_env_coupled_path_power=human_env_coupled_path_power,
            human_multi_touch_path_power=human_multi_touch_path_power,
            runtime_profile_s=runtime_profile_s,
            runtime_profile_counts=runtime_profile_counts,
        )
        cube = RadarCube(adc=adc, times=times, metadata=metadata)
        return FullRTMobilityBaselineResult(
            cube=cube,
            human_reference_adc=human_reference_adc,
            human_reference_path_counts=human_reference_path_counts,
            human_touch_path_counts=human_touch_path_counts,
            one_human_touch_path_counts=one_human_touch_path_counts,
            single_human_only_path_counts=single_human_only_path_counts,
            human_env_coupled_path_counts=human_env_coupled_path_counts,
            human_multi_touch_path_counts=human_multi_touch_path_counts,
            path_depth_histogram=path_depth_histogram,
            total_path_power=total_path_power,
            human_touch_path_power=human_touch_path_power,
            one_human_touch_path_power=one_human_touch_path_power,
            single_human_only_path_power=single_human_only_path_power,
            human_env_coupled_path_power=human_env_coupled_path_power,
            human_multi_touch_path_power=human_multi_touch_path_power,
            retrace_count=int(total_retraces),
        )
    finally:
        if added_objects:
            try:
                scene.edit(remove=added_objects)
            finally:
                for obj in added_objects:
                    obj._scene = lambda: None  # pylint: disable=protected-access
        for target, reference in zip(targets, old_references):
            target.reference_vertices = reference


def run_rt_mobility_baseline(
    *,
    scene,
    radar: RadarSensor,
    targets: list[MeshTarget],
    simulator: MmWaveRadarSimulator,
    num_frames: int,
) -> RTMobilityBaselineResult:
    """Backward-compatible name for the full-scene RT mobility baseline."""

    return run_full_rt_mobility_baseline(
        scene=scene,
        radar=radar,
        targets=targets,
        simulator=simulator,
        num_frames=num_frames,
    )


def _validate_calibration_frame_range(
    calibration_frame_range: tuple[int, int],
    *,
    num_frames: int,
) -> tuple[int, int]:
    """Validates an inclusive/exclusive calibration frame range."""

    try:
        start, stop = calibration_frame_range
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "calibration_frame_range must be a (start, stop) tuple"
        ) from exc
    start = int(start)
    stop = int(stop)
    if start < 0:
        raise ValueError("calibration_frame_range start must be non-negative")
    if stop <= start:
        raise ValueError("calibration_frame_range stop must be greater than start")
    if stop > int(num_frames):
        raise ValueError(
            "calibration_frame_range stop must be less than or equal to num_frames"
        )
    return start, stop


def _frame_range_target(
    target: MeshTarget,
    radar: RadarSensor,
    *,
    frame_start: int,
    frame_stop: int,
) -> MeshTarget:
    """Builds a time-rebased target view for calibration frames only."""

    fmcw = radar.fmcw
    start_time_s = float(fmcw.chirp_time(frame_start, 0))
    local_last_time_s = float(
        fmcw.chirp_time(
            frame_stop - frame_start - 1,
            fmcw.num_chirps_per_frame - 1,
        )
    )
    return MeshTarget(
        name=f"{target.name}_calibration",
        mesh_sequence=_FrameRangeMeshSequence(
            base_sequence=target.mesh_sequence,
            start_time_s=start_time_s,
            duration_s=local_last_time_s,
        ),
        material=target.material,
    )


def _shift_rt_result_times(
    result: FullRTMobilityBaselineResult,
    *,
    offset_s: float,
) -> FullRTMobilityBaselineResult:
    """Shifts RT cube and event timestamps back into the full-run time base."""

    offset_s = float(offset_s)
    if offset_s == 0.0:
        return result
    events = []
    for event in result.cube.metadata.events or []:
        shifted = dict(event)
        if "time" in shifted:
            shifted["time"] = float(shifted["time"]) + offset_s
        events.append(shifted)
    metadata = replace(result.cube.metadata, events=events)
    cube = replace(
        result.cube,
        times=np.asarray(result.cube.times, dtype=float) + offset_s,
        metadata=metadata,
    )
    return replace(result, cube=cube)


def _rt_result_to_calibration_baseline(
    result: FullRTMobilityBaselineResult,
    *,
    periodic_retrace: bool,
    periodic_retrace_period_chirps: int,
) -> dict:
    """Converts a full-RT result into the calibration baseline dictionary."""

    events = result.cube.metadata.events or []
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
        "periodic_retrace": bool(periodic_retrace),
        "retrace_once_per_frame": bool(periodic_retrace),
        "periodic_retrace_period_chirps": int(periodic_retrace_period_chirps),
        "coherent_path_update": bool(
            events and events[0].get("coherent_path_update", False)
        ),
    }


def _expand_selected_gain_map(
    po_calibration: dict,
    *,
    full_shape: tuple[int, int],
    frame_start: int,
    frame_stop: int,
) -> dict:
    """Expands a calibration-window gain map to the full frame/chirp grid."""

    value = po_calibration.get("human_po_amplitude_gain_map")
    if value is None:
        return po_calibration
    gains = np.asarray(value, dtype=np.float64)
    if gains.shape == full_shape:
        return po_calibration
    selected_shape = (int(frame_stop - frame_start), int(full_shape[1]))
    if gains.shape != selected_shape:
        return po_calibration

    full = np.empty(full_shape, dtype=np.float64)
    if frame_start > 0:
        full[:frame_start] = gains[0]
    full[frame_start:frame_stop] = gains
    if frame_stop < full_shape[0]:
        full[frame_stop:] = gains[-1]
    map_db = 20.0 * np.log10(np.maximum(full, 1e-300))
    out = dict(po_calibration)
    out["human_po_amplitude_gain_map"] = full
    out["human_po_amplitude_gain_map_expanded_from_frame_range"] = True
    out["human_po_amplitude_gain_map_min"] = float(np.min(full))
    out["human_po_amplitude_gain_map_mean"] = float(np.mean(full))
    out["human_po_amplitude_gain_map_max"] = float(np.max(full))
    out["human_po_amplitude_gain_map_min_db"] = float(np.min(map_db))
    out["human_po_amplitude_gain_map_mean_db"] = float(
        20.0 * np.log10(max(float(np.mean(full)), 1e-300))
    )
    out["human_po_amplitude_gain_map_max_db"] = float(np.max(map_db))
    return out


def _calibrate_human_po(
    *,
    calibration_result: FullRTMobilityBaselineResult | None,
    human_cube: RadarCube,
    calibration_config: POCalibrationConfig,
    frame_start: int,
    frame_stop: int,
    periodic_retrace: bool,
    periodic_retrace_period_chirps: int,
) -> tuple[float, dict]:
    """Fits and returns the human-only PO amplitude calibration metadata."""

    if calibration_config.mode == "none":
        return 1.0, {
            "mode": "none",
            "applied": False,
            "human_po_amplitude_gain": 1.0,
            "human_po_amplitude_gain_db": 0.0,
            "calibration_frame_range": (int(frame_start), int(frame_stop)),
        }
    if calibration_result is None:
        raise RuntimeError("RT calibration result is required for PO calibration")

    rt_baseline = _rt_result_to_calibration_baseline(
        calibration_result,
        periodic_retrace=periodic_retrace,
        periodic_retrace_period_chirps=periodic_retrace_period_chirps,
    )
    fit_t0 = time.perf_counter()
    if calibration_config.mode == "rt_first_pose":
        gain, po_calibration = fit_first_pose_adc_calibration(
            rt_reference_adc=rt_baseline["human_reference_adc"],
            po_reference_adc=human_cube.adc[frame_start],
            config=calibration_config,
        )
        po_calibration["reference_frame"] = int(frame_start)
        po_calibration["reference_chirps"] = int(human_cube.adc.shape[1])
    else:
        po_path_power = human_cube.metadata.total_path_power
        if po_path_power is None:
            raise RuntimeError("human PO cube is missing total path power metadata")
        gain, po_calibration = fit_sequence_path_power_calibration(
            rt_baseline=rt_baseline,
            po_times=human_cube.times[frame_start:frame_stop],
            po_path_power=np.asarray(po_path_power)[frame_start:frame_stop],
            config=calibration_config,
        )
        po_calibration = _expand_selected_gain_map(
            po_calibration,
            full_shape=human_cube.times.shape,
            frame_start=frame_start,
            frame_stop=frame_stop,
        )

    path_counts = np.asarray(rt_baseline["human_reference_path_counts"])
    po_calibration["calibration_frame_range"] = (int(frame_start), int(frame_stop))
    po_calibration["calibration_runtime_s"] = float(time.perf_counter() - fit_t0)
    po_calibration["rt_reference_human_path_count_min"] = int(np.min(path_counts))
    po_calibration["rt_reference_human_path_count_mean"] = float(
        np.mean(path_counts)
    )
    po_calibration["rt_reference_human_path_count_max"] = int(np.max(path_counts))
    po_calibration["rt_baseline_retrace_count"] = int(
        rt_baseline.get("retrace_count", 0)
    )
    po_calibration["rt_baseline_num_frames"] = int(frame_stop - frame_start)
    po_calibration["rt_baseline_periodic_retrace"] = bool(periodic_retrace)
    po_calibration["rt_baseline_retrace_once_per_frame"] = bool(periodic_retrace)
    po_calibration["rt_baseline_periodic_retrace_period_chirps"] = int(
        periodic_retrace_period_chirps
    )
    po_calibration["rt_baseline_retrace_policy"] = (
        "periodic_frame"
        if periodic_retrace
        and int(periodic_retrace_period_chirps) == int(human_cube.adc.shape[1])
        else ("periodic_chirp" if periodic_retrace else "adaptive_displacement")
    )
    coherent_path_update = bool(rt_baseline.get("coherent_path_update", False))
    po_calibration["rt_baseline_coherent_path_update"] = coherent_path_update
    po_calibration["rt_baseline_motion_between_retraces"] = (
        (
            "coherent_bank_periodic_active"
            if periodic_retrace else "coherent_bank_transition_crossfade"
        )
        if coherent_path_update
        else ("held_path_periodic" if periodic_retrace else "held_path_adaptive")
    )
    po_calibration["rt_baseline_phase_continuity_note"] = (
        (
            "Periodic rt_coherent_bank updates the active retrace bank chirp by "
            "chirp and switches banks at scheduled retrace boundaries; "
            "birth/death matching is diagnostic only."
            if periodic_retrace
            else
            "Adaptive rt_coherent_bank updates persistent paths chirp by chirp, "
            "but adaptive retrace timing can still place transition birth/death "
            "fades inside Doppler frames."
        )
        if coherent_path_update
        else (
            "The RT baseline reuses the last traced path coefficients and delays "
            "between retraces; it is a held-path control and does not validate "
            "phase continuity between retrace boundaries."
        )
    )
    return gain, po_calibration


def _run_static_environment_pass(
    *,
    scene,
    radar: RadarSensor,
    target: MeshTarget,
    times: np.ndarray,
    simulator: MmWaveRadarSimulator,
    config: HybridStaticEnvPOConfig,
) -> _StaticEnvironmentPass:
    """Traces static paths and synthesizes blocked/unblocked environment ADC."""

    runtime_profile_s = {
        "hybrid_static_trace": 0.0,
        "hybrid_segment_extraction": 0.0,
        "hybrid_reflector_extraction": 0.0,
        "hybrid_blocking_rays": 0.0,
        "hybrid_static_adc_synthesis": 0.0,
    }
    runtime_profile_counts = dict.fromkeys(runtime_profile_s, 0)

    total_t0 = time.perf_counter()
    radar.configure_scene(scene)
    trace_t0 = time.perf_counter()
    paths = simulator.trace_scene(scene)
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

    _, adc_complex_dtype = simulator.numpy_dtypes_for_precision(
        simulator.adc_compute_precision
    )
    static_unblocked_adc = np.zeros(
        times.shape + (radar.fmcw.num_adc_samples,
                       radar.hardware.num_virtual_channels),
        dtype=adc_complex_dtype,
    )
    static_blocked_adc = np.zeros_like(static_unblocked_adc)
    visibility_weights = np.zeros(
        times.shape + np.asarray(static_bank.delays_s).shape,
        dtype=np.float64,
    )
    static_path_count = int(np.count_nonzero(static_bank.valid))
    static_path_counts = np.full(times.shape, static_path_count, dtype=np.int64)
    blocked_counts = np.zeros(times.shape, dtype=np.int64)
    mean_visibility = np.zeros(times.shape, dtype=np.float64)
    unblocked_weights = static_bank.valid.astype(np.float64)
    unblocked_adc = simulator.synthesize_adc(
        static_bank.coefficients,
        static_bank.delays_s,
        static_bank.valid,
        radar.fmcw,
        backend=simulator.adc_compute_backend,
        precision=simulator.adc_compute_precision,
    )
    static_unblocked_adc[:] = unblocked_adc[None, None, :, :]

    visibility_scene = HumanVisibilityScene(
        target.mesh_sequence.faces,
        target.mesh_sequence.vertices_at(float(times.reshape(-1)[0])),
        name="hybrid-static-blocking-human",
    )
    visibility_state: PathVisibilityState | None = None
    frame_start_s = time.perf_counter()
    for frame in range(times.shape[0]):
        simulator.report_frame_progress(
            mode="hybrid_static_environment",
            frame=frame,
            num_frames=times.shape[0],
        )
        if frame > 0:
            simulator.log_frame_progress(
                mode="hybrid_static_env",
                frame=frame - 1,
                num_frames=times.shape[0],
                frame_start_s=frame_start_s,
                extra=f"static_paths={static_path_count}",
            )
        for chirp in range(times.shape[1]):
            simulator.check_cancelled()
            chirp_time = float(times[frame, chirp])
            vertices = target.mesh_sequence.vertices_at(chirp_time)
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

            visibility_weights[frame, chirp] = weights
            mean_visibility[frame, chirp] = (
                float(np.mean(weights[static_bank.valid]))
                if static_path_count > 0 else 0.0
            )
            static_adc_t0 = time.perf_counter()
            weighted_a = static_bank.coefficients * weights[None, :]
            valid = static_bank.valid & (weights > 0.0)
            static_blocked_adc[frame, chirp] = simulator.synthesize_adc(
                weighted_a,
                static_bank.delays_s,
                valid,
                radar.fmcw,
                backend=simulator.adc_compute_backend,
                precision=simulator.adc_compute_precision,
            )
            runtime_profile_s["hybrid_static_adc_synthesis"] += (
                time.perf_counter() - static_adc_t0
            )
            runtime_profile_counts["hybrid_static_adc_synthesis"] += 1

    simulator.log_frame_progress(
        mode="hybrid_static_env",
        frame=times.shape[0] - 1,
        num_frames=times.shape[0],
        frame_start_s=frame_start_s,
        extra=f"static_paths={static_path_count}",
    )
    return _StaticEnvironmentPass(
        static_unblocked_adc=static_unblocked_adc,
        static_blocked_adc=static_blocked_adc,
        visibility_weights=visibility_weights,
        static_path_counts=static_path_counts,
        blocked_counts=blocked_counts,
        mean_visibility=mean_visibility,
        static_bank=static_bank,
        reflector_bank=reflector_bank,
        runtime_profile_s=runtime_profile_s,
        runtime_profile_counts=runtime_profile_counts,
        wall_time_s=time.perf_counter() - total_t0,
    )


def _run_coupling_pass(
    *,
    scene,
    radar: RadarSensor,
    target: MeshTarget,
    times: np.ndarray,
    simulator: MmWaveRadarSimulator,
    config: HybridStaticEnvPOConfig,
    reflector_bank,
    gain_map: np.ndarray,
    progress: bool,
    progress_every_frames: int,
) -> _CouplingPass:
    """Computes dynamic human-environment coupling ADC over the time grid."""

    runtime_profile_s = {
        "hybrid_coupling_channel": 0.0,
        "hybrid_coupling_setup": 0.0,
        "hybrid_human_env_channel": 0.0,
        "hybrid_env_human_channel": 0.0,
        "hybrid_coupling_adc_synthesis": 0.0,
        "hybrid_human_env_adc_synthesis": 0.0,
        "hybrid_env_human_adc_synthesis": 0.0,
    }
    runtime_profile_counts = dict.fromkeys(runtime_profile_s, 0)
    _, adc_complex_dtype = simulator.numpy_dtypes_for_precision(
        simulator.adc_compute_precision
    )
    adc_shape = times.shape + (
        radar.fmcw.num_adc_samples,
        radar.hardware.num_virtual_channels,
    )
    human_env_adc = np.zeros(adc_shape, dtype=adc_complex_dtype)
    env_human_adc = np.zeros_like(human_env_adc)
    (
        human_env_counts,
        env_human_counts,
        human_env_min_lengths,
        human_env_mean_lengths,
        human_env_max_lengths,
        env_human_min_lengths,
        env_human_mean_lengths,
        env_human_max_lengths,
    ) = _zeroed_metadata_arrays(times.shape)

    total_t0 = time.perf_counter()
    reflector_count = int(reflector_bank.vertices.shape[0])
    if not config.coupling_enabled or reflector_count <= 0:
        return _CouplingPass(
            human_env_adc=human_env_adc,
            env_human_adc=env_human_adc,
            human_env_counts=human_env_counts,
            env_human_counts=env_human_counts,
            human_env_min_lengths=human_env_min_lengths,
            human_env_mean_lengths=human_env_mean_lengths,
            human_env_max_lengths=human_env_max_lengths,
            env_human_min_lengths=env_human_min_lengths,
            env_human_mean_lengths=env_human_mean_lengths,
            env_human_max_lengths=env_human_max_lengths,
            runtime_profile_s=runtime_profile_s,
            runtime_profile_counts=runtime_profile_counts,
            wall_time_s=time.perf_counter() - total_t0,
        )

    coupling_po_config = config.human_po_config
    coupling_compute_backend = (
        simulator.compute_backend if coupling_po_config.compute_backend is None
        else coupling_po_config.compute_backend
    )
    coupling_compute_precision = (
        simulator.compute_precision if coupling_po_config.compute_precision is None
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
    visibility_scene = HumanVisibilityScene(
        target.mesh_sequence.faces,
        target.mesh_sequence.vertices_at(float(times.reshape(-1)[0])),
        name="hybrid-coupling-human",
    )

    frame_start_s = time.perf_counter()
    progress_every_frames = max(1, int(progress_every_frames))
    cached_human_env_active_sets = None
    cached_env_human_active_sets = None
    for frame in range(times.shape[0]):
        simulator.report_frame_progress(
            mode="hybrid_coupling",
            frame=frame,
            num_frames=times.shape[0],
        )
        for chirp in range(times.shape[1]):
            simulator.check_cancelled()
            chirp_index = frame * times.shape[1] + chirp
            chirp_time = float(times[frame, chirp])
            vertices = target.mesh_sequence.vertices_at(chirp_time)
            visibility_scene.update_vertices(vertices)
            refresh_coupling_visibility = (
                not coupling_incremental_update
                or cached_human_env_active_sets is None
                or cached_env_human_active_sets is None
                or chirp_index % coupling_visibility_refresh_chirps == 0
            )

            coupling_t0 = time.perf_counter()
            coupling = compute_env_human_coupling_channels(
                vertices=vertices,
                faces=target.mesh_sequence.faces,
                tx_positions=radar.world_tx_positions(),
                rx_positions=radar.world_rx_positions(),
                virtual_channel_tx_indices=(
                    radar.hardware.virtual_tx_indices()
                ),
                virtual_channel_rx_indices=(
                    radar.hardware.virtual_rx_indices()
                ),
                virtual_channel_order=radar.hardware.virtual_channel_order,
                frequency_hz=radar.fmcw.carrier_frequency,
                human_material=target.material,
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
            (
                human_env_min_lengths[frame, chirp],
                human_env_mean_lengths[frame, chirp],
                human_env_max_lengths[frame, chirp],
            ) = _path_length_stats(
                coupling.human_env_valid,
                coupling.human_env_delays_s,
            )
            (
                env_human_min_lengths[frame, chirp],
                env_human_mean_lengths[frame, chirp],
                env_human_max_lengths[frame, chirp],
            ) = _path_length_stats(
                coupling.env_human_valid,
                coupling.env_human_delays_s,
            )

            chirp_gain = float(gain_map[frame, chirp])
            human_env_adc_t0 = time.perf_counter()
            human_env_adc[frame, chirp] = chirp_gain * simulator.synthesize_adc(
                coupling.human_env_coefficients,
                coupling.human_env_delays_s,
                coupling.human_env_valid,
                radar.fmcw,
                backend=simulator.adc_compute_backend,
                precision=simulator.adc_compute_precision,
            )
            human_env_adc_elapsed_s = time.perf_counter() - human_env_adc_t0
            env_human_adc_t0 = time.perf_counter()
            env_human_adc[frame, chirp] = chirp_gain * simulator.synthesize_adc(
                coupling.env_human_coefficients,
                coupling.env_human_delays_s,
                coupling.env_human_valid,
                radar.fmcw,
                backend=simulator.adc_compute_backend,
                precision=simulator.adc_compute_precision,
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

        completed_frames = frame + 1
        if progress and (
            completed_frames == 1
            or completed_frames == times.shape[0]
            or completed_frames % progress_every_frames == 0
        ):
            completed_chirps = completed_frames * times.shape[1]
            elapsed_s = time.perf_counter() - frame_start_s
            print(
                f"Coupling progress: frame {completed_frames}/{times.shape[0]} "
                f"({completed_chirps}/{times.size} chirps), "
                f"elapsed={elapsed_s:.1f} s, "
                f"H_env mean paths={np.mean(human_env_counts[:completed_frames]):.1f}, "
                f"E_human mean paths={np.mean(env_human_counts[:completed_frames]):.1f}",
                flush=True,
            )

    return _CouplingPass(
        human_env_adc=human_env_adc,
        env_human_adc=env_human_adc,
        human_env_counts=human_env_counts,
        env_human_counts=env_human_counts,
        human_env_min_lengths=human_env_min_lengths,
        human_env_mean_lengths=human_env_mean_lengths,
        human_env_max_lengths=human_env_max_lengths,
        env_human_min_lengths=env_human_min_lengths,
        env_human_mean_lengths=env_human_mean_lengths,
        env_human_max_lengths=env_human_max_lengths,
        runtime_profile_s=runtime_profile_s,
        runtime_profile_counts=runtime_profile_counts,
        wall_time_s=time.perf_counter() - total_t0,
    )


def hybrid_rt_path_coupling(
    *,
    scene,
    radar: RadarSensor,
    target: MeshTarget,
    base_cube: RadarCube,
    simulator: MmWaveRadarSimulator,
    config: HybridStaticEnvPOConfig | None = None,
    base_wall_time_s: float = 0.0,
    base_cube_path: str | Path | None = None,
    recompute: bool = True,
    progress: bool | None = None,
    progress_every_frames: int = 10,
    print_reflector_summary: bool = True,
) -> HybridCouplingRecomputeResult:
    """Recomputes H_env/E_human from a cached H_direct/E_static base cube.

    The base cube is expected to contain the direct human PO and blocked static
    RT components. This helper retraces the static scene only to rebuild the
    RT-derived reflector bank, synthesizes the two single-bounce coupling
    components, and returns an assembled hybrid cube.
    """

    if tdm_expansion_required(radar):
        expanded_result = hybrid_rt_path_coupling(
            scene=scene,
            radar=expanded_tdm_radar(radar),
            target=target,
            base_cube=expand_tdm_cube(base_cube, radar),
            simulator=simulator,
            config=config,
            base_wall_time_s=base_wall_time_s,
            base_cube_path=base_cube_path,
            recompute=recompute,
            progress=progress,
            progress_every_frames=progress_every_frames,
            print_reflector_summary=print_reflector_summary,
        )
        cube = collapse_tdm_cube(expanded_result.cube, radar)
        return replace(
            expanded_result,
            cube=cube,
            num_chirps=int(np.prod(cube.adc.shape[:2])),
        )

    config = config or simulator.hybrid_static_env_po_config
    if config is None:
        raise ValueError("a HybridStaticEnvPOConfig is required")
    progress = simulator.progress if progress is None else bool(progress)
    progress_every_frames = max(1, int(progress_every_frames))
    _require_components(
        base_cube,
        {
            "human_po",
            "static_environment_unblocked",
            "static_environment_blocked",
            "static_path_visibility_weights",
        },
    )

    times = base_cube.times.copy()
    _, adc_complex_dtype = simulator.numpy_dtypes_for_precision(
        simulator.adc_compute_precision
    )
    profile_s = dict(base_cube.metadata.runtime_profile_s or {})
    profile_counts = dict(base_cube.metadata.runtime_profile_counts or {})
    _reset_runtime_keys(profile_s, profile_counts)

    if recompute:
        human_env_adc = np.zeros_like(base_cube.adc, dtype=adc_complex_dtype)
        env_human_adc = np.zeros_like(base_cube.adc, dtype=adc_complex_dtype)
        (
            human_env_counts,
            env_human_counts,
            human_env_min_lengths,
            human_env_mean_lengths,
            human_env_max_lengths,
            env_human_min_lengths,
            env_human_mean_lengths,
            env_human_max_lengths,
        ) = _zeroed_metadata_arrays(times.shape)

        total_t0 = time.perf_counter()
        radar.configure_scene(scene)
        trace_t0 = time.perf_counter()
        paths = simulator.trace_scene(scene)
        profile_s["hybrid_static_trace"] = time.perf_counter() - trace_t0
        profile_counts["hybrid_static_trace"] = 1

        extraction_t0 = time.perf_counter()
        static_bank = extract_static_path_bank(
            paths,
            radar,
            virtual_channel_order=radar.hardware.virtual_channel_order,
        )
        profile_s["hybrid_segment_extraction"] = time.perf_counter() - extraction_t0
        profile_counts["hybrid_segment_extraction"] = 1

        reflector_t0 = time.perf_counter()
        reflector_bank = extract_rt_reflector_bank(
            scene,
            static_bank,
            max_reflectors=config.coupling_max_reflectors,
            min_area_m2=config.coupling_min_reflector_area_m2,
            frequency_hz=radar.fmcw.carrier_frequency,
        )
        profile_s["hybrid_reflector_extraction"] = time.perf_counter() - reflector_t0
        profile_counts["hybrid_reflector_extraction"] = 1
        reflector_count = int(reflector_bank.vertices.shape[0])
        if progress and print_reflector_summary:
            _print_reflector_summary(reflector_bank)

        po_calibration = base_cube.metadata.po_calibration or {}
        human_po_gain = float(po_calibration.get("human_po_amplitude_gain", 1.0))
        human_po_gain_map = calibration_gain_array(
            po_calibration,
            times.shape,
            fallback=human_po_gain,
        )
        if progress:
            print(
                "Human PO calibration [mode, gain, dB, reference chirps]:",
                (
                    po_calibration.get("mode", "none"),
                    human_po_gain,
                    po_calibration.get("human_po_amplitude_gain_db", 0.0),
                    po_calibration.get("reference_chirps", 0),
                ),
                flush=True,
            )

        coupling_po_config = config.human_po_config
        coupling_compute_backend = (
            simulator.compute_backend if coupling_po_config.compute_backend is None
            else coupling_po_config.compute_backend
        )
        coupling_compute_precision = (
            simulator.compute_precision if coupling_po_config.compute_precision is None
            else coupling_po_config.compute_precision
        )
        visibility_scene = HumanVisibilityScene(
            target.mesh_sequence.faces,
            target.mesh_sequence.vertices_at(float(times.reshape(-1)[0])),
            name="hybrid-coupling-human",
        )

        if config.coupling_enabled and reflector_count > 0:
            progress_t0 = time.perf_counter()
            for frame in range(times.shape[0]):
                for chirp in range(times.shape[1]):
                    chirp_time = float(times[frame, chirp])
                    vertices = target.mesh_sequence.vertices_at(chirp_time)
                    visibility_scene.update_vertices(vertices)

                    coupling_t0 = time.perf_counter()
                    coupling = compute_env_human_coupling_channels(
                        vertices=vertices,
                        faces=target.mesh_sequence.faces,
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
                        frequency_hz=radar.fmcw.carrier_frequency,
                        human_material=target.material,
                        reflectors=reflector_bank,
                        human_visibility_scene=visibility_scene,
                        static_visibility_scene=scene.mi_scene,
                        reflection_scale=config.coupling_reflection_scale,
                        backend=coupling_compute_backend,
                        precision=coupling_compute_precision,
                    )
                    coupling_elapsed_s = time.perf_counter() - coupling_t0
                    profile_s["hybrid_coupling_channel"] += coupling_elapsed_s
                    profile_counts["hybrid_coupling_channel"] += 1
                    coupling_profile_s = coupling.runtime_profile_s or {}
                    coupling_profile_counts = coupling.runtime_profile_counts or {}
                    profile_s["hybrid_coupling_setup"] += (
                        coupling_profile_s.get("coupling_setup", 0.0)
                        + coupling_profile_s.get("coupling_assembly", 0.0)
                    )
                    profile_counts["hybrid_coupling_setup"] += (
                        coupling_profile_counts.get("coupling_setup", 0)
                        + coupling_profile_counts.get("coupling_assembly", 0)
                    )
                    profile_s["hybrid_human_env_channel"] += coupling_profile_s.get(
                        "human_env_channel", 0.0
                    )
                    profile_counts["hybrid_human_env_channel"] += (
                        coupling_profile_counts.get("human_env_channel", 0)
                    )
                    profile_s["hybrid_env_human_channel"] += coupling_profile_s.get(
                        "env_human_channel", 0.0
                    )
                    profile_counts["hybrid_env_human_channel"] += (
                        coupling_profile_counts.get("env_human_channel", 0)
                    )

                    human_env_counts[frame, chirp] = int(np.count_nonzero(coupling.human_env_valid))
                    env_human_counts[frame, chirp] = int(np.count_nonzero(coupling.env_human_valid))
                    (
                        human_env_min_lengths[frame, chirp],
                        human_env_mean_lengths[frame, chirp],
                        human_env_max_lengths[frame, chirp],
                    ) = _path_length_stats(coupling.human_env_valid, coupling.human_env_delays_s)
                    (
                        env_human_min_lengths[frame, chirp],
                        env_human_mean_lengths[frame, chirp],
                        env_human_max_lengths[frame, chirp],
                    ) = _path_length_stats(coupling.env_human_valid, coupling.env_human_delays_s)

                    human_env_adc_t0 = time.perf_counter()
                    chirp_gain = human_po_gain_map[frame, chirp]
                    human_env_adc[frame, chirp] = chirp_gain * simulator.synthesize_adc(
                        coupling.human_env_coefficients,
                        coupling.human_env_delays_s,
                        coupling.human_env_valid,
                        radar.fmcw,
                        backend=simulator.adc_compute_backend,
                        precision=simulator.adc_compute_precision,
                    )
                    human_env_adc_elapsed_s = time.perf_counter() - human_env_adc_t0
                    env_human_adc_t0 = time.perf_counter()
                    env_human_adc[frame, chirp] = chirp_gain * simulator.synthesize_adc(
                        coupling.env_human_coefficients,
                        coupling.env_human_delays_s,
                        coupling.env_human_valid,
                        radar.fmcw,
                        backend=simulator.adc_compute_backend,
                        precision=simulator.adc_compute_precision,
                    )
                    env_human_adc_elapsed_s = time.perf_counter() - env_human_adc_t0
                    profile_s["hybrid_human_env_adc_synthesis"] += human_env_adc_elapsed_s
                    profile_counts["hybrid_human_env_adc_synthesis"] += 1
                    profile_s["hybrid_env_human_adc_synthesis"] += env_human_adc_elapsed_s
                    profile_counts["hybrid_env_human_adc_synthesis"] += 1
                    profile_s["hybrid_coupling_adc_synthesis"] += (
                        human_env_adc_elapsed_s + env_human_adc_elapsed_s
                    )
                    profile_counts["hybrid_coupling_adc_synthesis"] += 1

                completed_frames = frame + 1
                if progress and (
                    completed_frames == 1
                    or completed_frames == times.shape[0]
                    or completed_frames % progress_every_frames == 0
                ):
                    completed_chirps = completed_frames * times.shape[1]
                    elapsed_s = time.perf_counter() - progress_t0
                    print(
                        f"Coupling progress: frame {completed_frames}/{times.shape[0]} "
                        f"({completed_chirps}/{times.size} chirps), "
                        f"elapsed={elapsed_s:.1f} s, "
                        f"H_env mean paths={np.mean(human_env_counts[:completed_frames]):.1f}, "
                        f"E_human mean paths={np.mean(env_human_counts[:completed_frames]):.1f}",
                        flush=True,
                    )

        coupling_wall_time_s = time.perf_counter() - total_t0
    else:
        _require_components(base_cube, {"human_env", "env_human"})
        if progress:
            print(
                "Using coupling components from loaded cube because "
                "recompute=False",
                flush=True,
            )
        human_env_adc = base_cube.components["human_env"].copy()
        env_human_adc = base_cube.components["env_human"].copy()
        human_env_counts = base_cube.metadata.human_env_path_counts
        env_human_counts = base_cube.metadata.env_human_path_counts
        human_env_min_lengths = base_cube.metadata.human_env_min_path_lengths_m
        human_env_mean_lengths = base_cube.metadata.human_env_mean_path_lengths_m
        human_env_max_lengths = base_cube.metadata.human_env_max_path_lengths_m
        env_human_min_lengths = base_cube.metadata.env_human_min_path_lengths_m
        env_human_mean_lengths = base_cube.metadata.env_human_mean_path_lengths_m
        env_human_max_lengths = base_cube.metadata.env_human_max_path_lengths_m
        if human_env_counts is None or env_human_counts is None:
            (
                human_env_counts,
                env_human_counts,
                human_env_min_lengths,
                human_env_mean_lengths,
                human_env_max_lengths,
                env_human_min_lengths,
                env_human_mean_lengths,
                env_human_max_lengths,
            ) = _zeroed_metadata_arrays(times.shape)
        coupling_wall_time_s = 0.0
        reflector_count = 0

    profile_s["hybrid_coupling_recompute_total"] = coupling_wall_time_s
    profile_counts["hybrid_coupling_recompute_total"] = 1
    if progress:
        print(
            f"Recomputed H_env/E_human: {coupling_wall_time_s:.2f} s",
            flush=True,
        )
        print(
            "Coupling runtime profile [s]:",
            {
                key: profile_s.get(key, 0.0)
                for key in (
                    "hybrid_static_trace",
                    "hybrid_segment_extraction",
                    "hybrid_reflector_extraction",
                    "hybrid_coupling_setup",
                    "hybrid_human_env_channel",
                    "hybrid_env_human_channel",
                    "hybrid_coupling_channel",
                    "hybrid_human_env_adc_synthesis",
                    "hybrid_env_human_adc_synthesis",
                    "hybrid_coupling_adc_synthesis",
                    "hybrid_coupling_recompute_total",
                )
            },
            flush=True,
        )

    human_po_adc = base_cube.components["human_po"].copy()
    static_unblocked_adc = base_cube.components["static_environment_unblocked"].copy()
    static_blocked_adc = base_cube.components["static_environment_blocked"].copy()
    static_path_visibility_weights = base_cube.components["static_path_visibility_weights"].copy()
    hybrid_adc = (human_po_adc + static_blocked_adc + human_env_adc + env_human_adc).astype(
        base_cube.adc.dtype,
        copy=False,
    )

    static_counts = base_cube.metadata.static_path_counts
    if static_counts is None:
        static_counts = np.zeros(times.shape, dtype=np.int64)
    human_counts = base_cube.metadata.visible_face_counts
    if human_counts is None:
        human_counts = np.zeros(times.shape, dtype=np.int64)
    path_counts = static_counts + human_counts + human_env_counts + env_human_counts
    hybrid_wall_time_s = float(base_wall_time_s) + coupling_wall_time_s
    profile_s["hybrid_total"] = hybrid_wall_time_s
    profile_counts["hybrid_total"] = 1

    base_events = list(base_cube.metadata.events or [])
    event = {
        "reason": "loaded_base_recomputed_rt_path_coupling",
        "coupling_candidate_source": "static_rt_paths",
        "coupling_recomputed": bool(recompute),
        "coupling_max_reflectors": int(config.coupling_max_reflectors),
        "po_calibration_gain": (base_cube.metadata.po_calibration or {}).get(
            "human_po_amplitude_gain", 1.0
        ),
    }
    if base_cube_path is not None:
        event["base_cube_path"] = str(base_cube_path)
    hybrid_metadata = replace(
        base_cube.metadata,
        events=base_events + [event],
        path_counts=path_counts,
        runtime_profile_s=profile_s,
        runtime_profile_counts=profile_counts,
        human_env_path_counts=human_env_counts,
        env_human_path_counts=env_human_counts,
        human_env_min_path_lengths_m=human_env_min_lengths,
        human_env_mean_path_lengths_m=human_env_mean_lengths,
        human_env_max_path_lengths_m=human_env_max_lengths,
        env_human_min_path_lengths_m=env_human_min_lengths,
        env_human_mean_path_lengths_m=env_human_mean_lengths,
        env_human_max_path_lengths_m=env_human_max_lengths,
    )
    components = {
        "human_po": human_po_adc,
        "static_environment_unblocked": static_unblocked_adc,
        "static_environment_blocked": static_blocked_adc,
        "human_env": human_env_adc,
        "env_human": env_human_adc,
        "static_path_visibility_weights": static_path_visibility_weights,
    }
    if "rt_baseline_one_human_first_frame" in base_cube.components:
        components["rt_baseline_one_human_first_frame"] = (
            base_cube.components["rt_baseline_one_human_first_frame"]
        )
    hybrid_cube = RadarCube(
        adc=hybrid_adc,
        times=times,
        metadata=hybrid_metadata,
        components=components,
    )
    return HybridCouplingRecomputeResult(
        cube=hybrid_cube,
        wall_time_s=hybrid_wall_time_s,
        num_chirps=int(np.prod(hybrid_cube.adc.shape[:2])),
        coupling_wall_time_s=coupling_wall_time_s,
        reflector_count=reflector_count,
    )


def recompute_hybrid_rt_path_coupling(
    **kwargs,
) -> HybridCouplingRecomputeResult:
    """Backward-compatible alias for ``hybrid_rt_path_coupling``."""

    return hybrid_rt_path_coupling(**kwargs)


def run_calibrated_hybrid_po(
    *,
    scene,
    radar: RadarSensor,
    target: MeshTarget,
    num_frames: int,
    calibration_frame_range: tuple[int, int] = (0, 1),
    coupling_scene=None,
    po_visibility_samples_per_face: int = 4,
    coupling_enabled: bool = True,
    coupling_max_reflectors: int = 0,
    coupling_reflection_scale: float = 1.0,
    rt_samples_per_src: int = 80_000,
    rt_max_num_paths_per_src: int = 80_000,
    max_depth: int = 2,
    diffuse_reflection: bool = True,
    seed: int = 42,
    compute_backend="auto",
    compute_precision="float32",
    adc_compute_precision=None,
    progress: bool = True,
    progress_every_frames: int = 10,
    print_reflector_summary: bool = True,
    calibration_config: POCalibrationConfig | None = None,
    human_po_config: HumanPOMobilityConfig | None = None,
    base_config: HybridStaticEnvPOConfig | None = None,
    hybrid_config: HybridStaticEnvPOConfig | None = None,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> CalibratedHybridPOResult:
    """Runs calibrated Hybrid PO as an explicit six-stage algorithm flow."""

    num_frames = int(num_frames)
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if cancel_check is not None and not callable(cancel_check):
        raise ValueError("cancel_check must be callable or None")
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable or None")
    if tdm_expansion_required(radar):
        expanded_result = run_calibrated_hybrid_po(
            scene=scene,
            radar=expanded_tdm_radar(radar),
            target=target,
            num_frames=num_frames,
            calibration_frame_range=calibration_frame_range,
            coupling_scene=coupling_scene,
            po_visibility_samples_per_face=po_visibility_samples_per_face,
            coupling_enabled=coupling_enabled,
            coupling_max_reflectors=coupling_max_reflectors,
            coupling_reflection_scale=coupling_reflection_scale,
            rt_samples_per_src=rt_samples_per_src,
            rt_max_num_paths_per_src=rt_max_num_paths_per_src,
            max_depth=max_depth,
            diffuse_reflection=diffuse_reflection,
            seed=seed,
            compute_backend=compute_backend,
            compute_precision=compute_precision,
            adc_compute_precision=adc_compute_precision,
            progress=progress,
            progress_every_frames=progress_every_frames,
            print_reflector_summary=print_reflector_summary,
            calibration_config=calibration_config,
            human_po_config=human_po_config,
            base_config=base_config,
            hybrid_config=hybrid_config,
            cancel_check=cancel_check,
            progress_callback=progress_callback,
        )
        cube = collapse_tdm_cube(expanded_result.cube, radar)
        base_cube = collapse_tdm_cube(expanded_result.base_cube, radar)
        coupling_result = replace(
            expanded_result.coupling_result,
            cube=cube,
            num_chirps=int(np.prod(cube.adc.shape[:2])),
        )
        return replace(
            expanded_result,
            cube=cube,
            base_cube=base_cube,
            coupling_result=coupling_result,
            base_num_chirps=int(np.prod(base_cube.adc.shape[:2])),
        )

    def check_cancelled():
        if cancel_check is not None and bool(cancel_check()):
            raise InterruptedError("Simulation cancelled by user")

    check_cancelled()
    calibration_frame_start, calibration_frame_stop = (
        _validate_calibration_frame_range(
            calibration_frame_range,
            num_frames=num_frames,
        )
    )

    pipeline_t0 = time.perf_counter()
    total_stages = 6

    def print_stage(stage: int, message: str, *, elapsed_s: float | None = None):
        """Prints one progress line when progress reporting is enabled."""

        if not progress:
            return
        suffix = "" if elapsed_s is None else f" elapsed={elapsed_s:.2f}s"
        print(
            f"[hybrid-po] stage {stage}/{total_stages}: {message}{suffix}",
            flush=True,
        )

    if human_po_config is None and hybrid_config is not None:
        human_po_config = hybrid_config.human_po_config
    if human_po_config is None and base_config is not None:
        human_po_config = base_config.human_po_config

    if calibration_config is None and human_po_config is not None:
        calibration_config = human_po_config.calibration
    if calibration_config is None:
        calibration_config = POCalibrationConfig(
            mode="rt_sequence",
            gain_type="amplitude",
            sample_mode="all_chirps",
            reference_component="human_touch",
            alignment="same_index",
            keep_rt_baseline=False,
            rt_retrace_once_per_frame=True,
        )

    raw_calibration_config = replace(calibration_config, mode="none")
    if human_po_config is None:
        human_po_config = HumanPOMobilityConfig(
            visibility_samples_per_face=po_visibility_samples_per_face,
            visibility_fade_chirps=8,
            visibility_use_phase_center=True,
            adaptive_visibility_sampling=True,
            adaptive_visibility_edge_margin=0.15,
            incremental_update=True,
            incremental_visibility_refresh_chirps=64,
            incremental_full_refresh_chirps=0,
            incremental_normal_threshold_deg=20.0,
            incremental_centroid_displacement_threshold_m=0.020,
            incremental_area_relative_threshold=0.30,
            calibration=raw_calibration_config,
        )
    else:
        human_po_config = replace(
            human_po_config,
            calibration=raw_calibration_config,
        )

    if base_config is None:
        base_config = HybridStaticEnvPOConfig(
            blocking_enabled=True,
            blocking_fade_chirps=8,
            blocking_ray_epsilon_m=1e-5,
            blocking_ray_chunk_size=65536,
            coupling_enabled=False,
            coupling_max_reflectors=0,
            coupling_min_reflector_area_m2=1e-4,
            coupling_reflection_scale=coupling_reflection_scale,
            human_po_config=human_po_config,
        )
    else:
        base_config = replace(
            base_config,
            coupling_enabled=False,
            human_po_config=human_po_config,
        )

    if hybrid_config is None:
        hybrid_config = HybridStaticEnvPOConfig(
            blocking_enabled=True,
            blocking_fade_chirps=8,
            blocking_ray_epsilon_m=1e-5,
            blocking_ray_chunk_size=65536,
            coupling_enabled=coupling_enabled,
            coupling_max_reflectors=coupling_max_reflectors,
            coupling_min_reflector_area_m2=1e-4,
            coupling_reflection_scale=coupling_reflection_scale,
            human_po_config=human_po_config,
        )
    else:
        hybrid_config = replace(hybrid_config, human_po_config=human_po_config)

    environment_scene = scene if coupling_scene is None else coupling_scene
    periodic_retrace = bool(calibration_config.rt_retrace_once_per_frame)
    periodic_retrace_period_chirps = (
        int(calibration_config.rt_periodic_retrace_period_chirps)
        if calibration_config.rt_periodic_retrace_period_chirps is not None
        else int(radar.fmcw.num_chirps_per_frame)
    )
    simulator_kwargs = dict(
        compute_backend=compute_backend,
        compute_precision=compute_precision,
        progress=progress,
        progress_interval=progress_every_frames,
        coupling_mode="unrestricted",
        max_depth=max_depth,
        samples_per_src=rt_samples_per_src,
        max_num_paths_per_src=rt_max_num_paths_per_src,
        diffuse_reflection=diffuse_reflection,
        retrace_once_per_frame=periodic_retrace,
        periodic_retrace_period_chirps=periodic_retrace_period_chirps,
        seed=seed,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )
    if adc_compute_precision is not None:
        simulator_kwargs["adc_compute_precision"] = adc_compute_precision

    calibration_result = None
    calibration_wall_time_s = 0.0
    if calibration_config.mode == "none":
        print_stage(1, "calibration RT skipped because calibration mode is none")
    else:
        check_cancelled()
        print_stage(
            1,
            (
                "full-scene RT calibration "
                f"frames=[{calibration_frame_start}, {calibration_frame_stop})"
            ),
        )
        calibration_simulator = MmWaveRadarSimulator(
            mobility_mode="rt_coherent_bank",
            **simulator_kwargs,
        )
        calibration_target = _frame_range_target(
            target,
            radar,
            frame_start=calibration_frame_start,
            frame_stop=calibration_frame_stop,
        )
        calibration_t0 = time.perf_counter()
        calibration_result = run_full_rt_mobility_baseline(
            scene=scene,
            radar=radar,
            targets=[calibration_target],
            simulator=calibration_simulator,
            num_frames=calibration_frame_stop - calibration_frame_start,
        )
        calibration_result = _shift_rt_result_times(
            calibration_result,
            offset_s=float(radar.fmcw.chirp_time(calibration_frame_start, 0)),
        )
        calibration_wall_time_s = time.perf_counter() - calibration_t0
        if progress:
            print(
                "[hybrid-po] stage 1/6 complete: "
                f"{calibration_wall_time_s:.2f}s "
                f"retraces={calibration_result.retrace_count}",
                flush=True,
            )

    print_stage(
        2,
        (
            "human-only PO over full window and scalar calibration "
            f"mode={calibration_config.mode}/{calibration_config.reference_component}"
        ),
        elapsed_s=time.perf_counter() - pipeline_t0,
    )
    check_cancelled()
    human_simulator = MmWaveRadarSimulator(
        mobility_mode="human_only_po",
        **simulator_kwargs,
        human_po_config=human_po_config,
    )
    radar.configure_scene(environment_scene)
    human_t0 = time.perf_counter()
    raw_human_cube = human_simulator._run_human_only_po(  # pylint: disable=protected-access
        radar,
        [target],
        num_frames=num_frames,
        static_visibility_scene=environment_scene.mi_scene,
        human_po_config=human_po_config,
    )
    human_wall_time_s = time.perf_counter() - human_t0
    calibration_fit_t0 = time.perf_counter()
    human_po_gain, po_calibration = _calibrate_human_po(
        calibration_result=calibration_result,
        human_cube=raw_human_cube,
        calibration_config=calibration_config,
        frame_start=calibration_frame_start,
        frame_stop=calibration_frame_stop,
        periodic_retrace=periodic_retrace,
        periodic_retrace_period_chirps=periodic_retrace_period_chirps,
    )
    calibration_fit_wall_time_s = time.perf_counter() - calibration_fit_t0
    gain_map = calibration_gain_array(
        po_calibration,
        raw_human_cube.times.shape,
        fallback=human_po_gain,
    )
    _, adc_complex_dtype = human_simulator.numpy_dtypes_for_precision(
        human_simulator.adc_compute_precision
    )
    human_po_adc = (raw_human_cube.adc * gain_map[..., None, None]).astype(
        adc_complex_dtype,
        copy=False,
    )
    if progress:
        gain_values = np.asarray(gain_map, dtype=float)
        gain_values = gain_values[np.isfinite(gain_values) & (gain_values > 0.0)]
        if gain_values.size:
            print(
                "[hybrid-po] calibration gain field "
                f"min/mean/max={np.min(gain_values):.3g}/"
                f"{np.mean(gain_values):.3g}/{np.max(gain_values):.3g}",
                flush=True,
            )
        print(
            "[hybrid-po] RT calibration baseline "
            f"retraces={po_calibration.get('rt_baseline_retrace_count', 'n/a')} "
            f"frames={po_calibration.get('rt_baseline_num_frames', 'n/a')} "
            f"runtime={calibration_wall_time_s:.2f}s",
            flush=True,
        )
        print(
            "[hybrid-po] stage 2/6 complete: "
            f"human_po={human_wall_time_s:.2f}s "
            f"calibration_fit={calibration_fit_wall_time_s:.2f}s",
            flush=True,
        )

    print_stage(
        3,
        "static-environment RT, dynamic human blocking, reflector selection",
        elapsed_s=time.perf_counter() - pipeline_t0,
    )
    check_cancelled()
    static_simulator = MmWaveRadarSimulator(
        mobility_mode="rt_retrace",
        **simulator_kwargs,
        hybrid_static_env_po_config=hybrid_config,
    )
    static_pass = _run_static_environment_pass(
        scene=environment_scene,
        radar=radar,
        target=target,
        times=raw_human_cube.times,
        simulator=static_simulator,
        config=hybrid_config,
    )
    if progress:
        print(
            "[hybrid-po] stage 3/6 complete: "
            f"{static_pass.wall_time_s:.2f}s "
            f"static_paths={int(np.count_nonzero(static_pass.static_bank.valid))} "
            f"reflectors={int(static_pass.reflector_bank.vertices.shape[0])}",
            flush=True,
        )
        if print_reflector_summary:
            _print_reflector_summary(static_pass.reflector_bank)

    print_stage(
        4,
        "PO for H_env/E_human using RT-selected image sources",
        elapsed_s=time.perf_counter() - pipeline_t0,
    )
    check_cancelled()
    coupling_simulator = MmWaveRadarSimulator(
        mobility_mode="rt_retrace",
        **simulator_kwargs,
        hybrid_static_env_po_config=hybrid_config,
    )
    coupling_pass = _run_coupling_pass(
        scene=environment_scene,
        radar=radar,
        target=target,
        times=raw_human_cube.times,
        simulator=coupling_simulator,
        config=hybrid_config,
        reflector_bank=static_pass.reflector_bank,
        gain_map=gain_map,
        progress=progress,
        progress_every_frames=progress_every_frames,
    )
    if progress:
        print(
            "[hybrid-po] stage 4/6 complete: "
            f"H_env mean paths={np.mean(coupling_pass.human_env_counts):.1f}",
            flush=True,
        )

    print_stage(
        5,
        "report E_human from shared coupling pass",
        elapsed_s=time.perf_counter() - pipeline_t0,
    )
    check_cancelled()
    if progress:
        print(
            "[hybrid-po] stage 5/6 complete: "
            f"E_human mean paths={np.mean(coupling_pass.env_human_counts):.1f}",
            flush=True,
        )

    print_stage(
        6,
        "assemble coherent Hybrid PO cube",
        elapsed_s=time.perf_counter() - pipeline_t0,
    )
    check_cancelled()
    runtime_profile_s = {
        "hybrid_rt_baseline": calibration_wall_time_s,
        "hybrid_po_calibration": calibration_fit_wall_time_s,
        "hybrid_human_po": human_wall_time_s,
        **static_pass.runtime_profile_s,
        **coupling_pass.runtime_profile_s,
    }
    runtime_profile_counts = {
        "hybrid_rt_baseline": 1 if calibration_result is not None else 0,
        "hybrid_po_calibration": 1,
        "hybrid_human_po": 1,
        **static_pass.runtime_profile_counts,
        **coupling_pass.runtime_profile_counts,
    }
    if raw_human_cube.metadata.runtime_profile_s:
        for key, value in raw_human_cube.metadata.runtime_profile_s.items():
            runtime_profile_s[f"human_{key}"] = float(value)
    if raw_human_cube.metadata.runtime_profile_counts:
        for key, value in raw_human_cube.metadata.runtime_profile_counts.items():
            runtime_profile_counts[f"human_{key}"] = int(value)

    assembly_t0 = time.perf_counter()
    hybrid_adc = (
        human_po_adc
        + static_pass.static_blocked_adc
        + coupling_pass.human_env_adc
        + coupling_pass.env_human_adc
    ).astype(adc_complex_dtype)
    assembly_wall_time_s = time.perf_counter() - assembly_t0
    runtime_profile_s["hybrid_assembly"] = assembly_wall_time_s
    runtime_profile_counts["hybrid_assembly"] = 1
    base_wall_time_s = (
        calibration_wall_time_s
        + calibration_fit_wall_time_s
        + human_wall_time_s
        + static_pass.wall_time_s
    )
    hybrid_wall_time_s = base_wall_time_s + coupling_pass.wall_time_s
    runtime_profile_s["hybrid_total"] = hybrid_wall_time_s
    runtime_profile_counts["hybrid_total"] = 1

    human_counts = raw_human_cube.metadata.path_counts
    path_counts = (
        static_pass.static_path_counts
        + human_counts
        + coupling_pass.human_env_counts
        + coupling_pass.env_human_counts
    )
    events = [
        {
            "reason": "hybrid_po_calibration_rt",
            "calibration_frame_range": (
                int(calibration_frame_start),
                int(calibration_frame_stop),
            ),
            "calibration_mode": calibration_config.mode,
            "rt_calibration_frames": int(
                calibration_frame_stop - calibration_frame_start
            ),
        },
        {
            "reason": "static_environment_trace",
            "static_path_count": int(np.count_nonzero(static_pass.static_bank.valid)),
            "segment_count": int(static_pass.static_bank.segment_path_indices.size),
            "blocking_enabled": bool(hybrid_config.blocking_enabled),
            "coupling_enabled": bool(hybrid_config.coupling_enabled),
            "coupling_reflector_count": int(
                static_pass.reflector_bank.vertices.shape[0]
            ),
            "coupling_candidate_source": "static_rt_paths",
            "coupling_max_reflectors": int(hybrid_config.coupling_max_reflectors),
            "po_calibration_gain": po_calibration.get(
                "human_po_amplitude_gain",
                1.0,
            ),
        },
        {"reason": "po_calibration", **po_calibration},
    ]
    metadata = SensingMetadata(
        events=events,
        path_counts=path_counts,
        max_displacements=raw_human_cube.metadata.max_displacements,
        virtual_channel_order=radar.hardware.virtual_channel_order,
        mode="hybrid_static_env_po",
        visible_face_counts=raw_human_cube.metadata.visible_face_counts,
        effective_visible_area_fraction=(
            raw_human_cube.metadata.effective_visible_area_fraction
        ),
        runtime_per_chirp_s=raw_human_cube.metadata.runtime_per_chirp_s,
        max_unwrapped_phase_jump=raw_human_cube.metadata.max_unwrapped_phase_jump,
        runtime_profile_s=runtime_profile_s,
        runtime_profile_counts=runtime_profile_counts,
        incremental_recomputed_face_counts=(
            raw_human_cube.metadata.incremental_recomputed_face_counts
        ),
        incremental_phase_updated_face_counts=(
            raw_human_cube.metadata.incremental_phase_updated_face_counts
        ),
        incremental_full_refresh=raw_human_cube.metadata.incremental_full_refresh,
        incremental_visibility_refresh=(
            raw_human_cube.metadata.incremental_visibility_refresh
        ),
        static_path_counts=static_pass.static_path_counts,
        blocked_static_path_counts=static_pass.blocked_counts,
        mean_static_path_visibility=static_pass.mean_visibility,
        human_env_path_counts=coupling_pass.human_env_counts,
        env_human_path_counts=coupling_pass.env_human_counts,
        human_env_min_path_lengths_m=coupling_pass.human_env_min_lengths,
        human_env_mean_path_lengths_m=coupling_pass.human_env_mean_lengths,
        human_env_max_path_lengths_m=coupling_pass.human_env_max_lengths,
        env_human_min_path_lengths_m=coupling_pass.env_human_min_lengths,
        env_human_mean_path_lengths_m=coupling_pass.env_human_mean_lengths,
        env_human_max_path_lengths_m=coupling_pass.env_human_max_lengths,
        po_calibration=po_calibration,
    )
    base_cube = RadarCube(
        adc=(human_po_adc + static_pass.static_blocked_adc).astype(adc_complex_dtype),
        times=raw_human_cube.times,
        metadata=metadata,
        components={
            "human_po": human_po_adc.copy(),
            "static_environment_unblocked": static_pass.static_unblocked_adc.copy(),
            "static_environment_blocked": static_pass.static_blocked_adc.copy(),
            "static_path_visibility_weights": static_pass.visibility_weights.copy(),
        },
    )
    hybrid_cube = RadarCube(
        adc=hybrid_adc,
        times=raw_human_cube.times,
        metadata=metadata,
        components={
            "human_po": human_po_adc.copy(),
            "static_environment_unblocked": static_pass.static_unblocked_adc.copy(),
            "static_environment_blocked": static_pass.static_blocked_adc.copy(),
            "human_env": coupling_pass.human_env_adc.copy(),
            "env_human": coupling_pass.env_human_adc.copy(),
            "static_path_visibility_weights": static_pass.visibility_weights.copy(),
        },
    )
    coupling_result = HybridCouplingRecomputeResult(
        cube=hybrid_cube,
        wall_time_s=hybrid_wall_time_s,
        num_chirps=int(np.prod(hybrid_cube.adc.shape[:2])),
        coupling_wall_time_s=coupling_pass.wall_time_s,
        reflector_count=int(static_pass.reflector_bank.vertices.shape[0]),
    )
    if progress:
        print(
            "[hybrid-po] stage 6/6 complete: "
            f"shape={hybrid_cube.adc.shape} total={hybrid_wall_time_s:.2f}s",
            flush=True,
        )

    return CalibratedHybridPOResult(
        cube=hybrid_cube,
        base_cube=base_cube,
        coupling_result=coupling_result,
        base_wall_time_s=base_wall_time_s,
        base_num_chirps=int(np.prod(base_cube.adc.shape[:2])),
        calibration_config=calibration_config,
        human_po_config=human_po_config,
        base_config=base_config,
        hybrid_config=hybrid_config,
    )


def run_full_scene_rt_comparison(
    *,
    scene,
    radar: RadarSensor,
    target: MeshTarget,
    num_frames: int,
    rt_samples_per_src: int = 80_000,
    rt_max_num_paths_per_src: int = 80_000,
    max_depth: int = 2,
    diffuse_reflection: bool = True,
    retrace_once_per_frame: bool = True,
    seed: int = 42,
    compute_backend="auto",
    compute_precision="float32",
    adc_compute_precision=None,
    progress: bool = True,
) -> FullRTMobilityBaselineResult:
    """Runs a full-scene RT comparison outside the hybrid PO algorithm."""

    simulator_kwargs = dict(
        mobility_mode="rt_coherent_bank",
        compute_backend=compute_backend,
        compute_precision=compute_precision,
        progress=progress,
        coupling_mode="unrestricted",
        max_depth=max_depth,
        samples_per_src=rt_samples_per_src,
        max_num_paths_per_src=rt_max_num_paths_per_src,
        diffuse_reflection=diffuse_reflection,
        retrace_once_per_frame=retrace_once_per_frame,
        seed=seed,
    )
    if adc_compute_precision is not None:
        simulator_kwargs["adc_compute_precision"] = adc_compute_precision

    if progress:
        print(
            "[full-rt-comparison] running full-scene RT "
            f"{num_frames} frames x {radar.fmcw.num_chirps_per_frame} chirps",
            flush=True,
        )
    return run_full_rt_mobility_baseline(
        scene=scene,
        radar=radar,
        targets=[target],
        simulator=MmWaveRadarSimulator(**simulator_kwargs),
        num_frames=num_frames,
    )


def run_hybrid_rt_path_pipeline(
    *,
    scene,
    radar: RadarSensor,
    target: MeshTarget,
    base_cube: RadarCube,
    simulator: MmWaveRadarSimulator,
    config: HybridStaticEnvPOConfig | None = None,
    base_wall_time_s: float = 0.0,
    base_cube_path: str | Path | None = None,
    recompute: bool = True,
    progress: bool | None = None,
    progress_every_frames: int = 10,
    print_reflector_summary: bool = True,
) -> HybridCouplingRecomputeResult:
    """Runs the top-level hybrid RT-path/PO assembly from a cached base cube."""

    return hybrid_rt_path_coupling(
        scene=scene,
        radar=radar,
        target=target,
        base_cube=base_cube,
        simulator=simulator,
        config=config,
        base_wall_time_s=base_wall_time_s,
        base_cube_path=base_cube_path,
        recompute=recompute,
        progress=progress,
        progress_every_frames=progress_every_frames,
        print_reflector_summary=print_reflector_summary,
    )
