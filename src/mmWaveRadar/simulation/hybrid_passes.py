# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Shared static blocking and human/environment coupling passes.

Entry points supply a configured simulator and explicit time/gain arrays. This
module does not import the simulator at runtime, so both the simulator and the
higher-level pipeline can use the same chirp loops.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import c

from .env_human_coupling import compute_env_human_coupling_channels
from .coupling_incremental import CouplingIncrementalState
from .cadence import refresh_period_chirps
from .physical_optics import HumanVisibilityScene
from .reflector_selection import select_image_source_reflectors
from .static_blocking import (
    PathVisibilityState,
    blocked_path_visibility,
    extract_static_path_bank,
)

if TYPE_CHECKING:
    from ..radar import RadarSensor
    from ..targets import MeshTarget
    from .simulator import MmWaveRadarSimulator
    from .types import HybridPOConfig


@dataclass(frozen=True)
class _StaticEnvironmentPass:
    """Static RT, dynamic blocking, and power-ranked image-source candidates."""

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


def _select_reflectors(scene, radar, target, times, config):
    """Select image sources over the same time window with one config mapping."""

    return select_image_source_reflectors(
        scene,
        radar,
        target,
        enabled=config.hybrid_coupling,
        power_threshold_db=config.hybrid_reflector_cutoff_db,
        max_count=config.hybrid_max_reflectors,
        min_area_m2=config.hybrid_min_reflector_area_m2,
        reflection_scale=config.hybrid_reflection_scale,
        time_range_s=(float(times.min()), float(times.max())),
    )


def _with_shared_reflector_profile(cube, elapsed_s):
    """Account once for selection performed before the subarray pair loop."""

    seconds = dict(cube.metadata.runtime_profile_s or {})
    counts = dict(cube.metadata.runtime_profile_counts or {})
    key = "hybrid_reflector_extraction"
    seconds[key] = seconds.get(key, 0.0) + elapsed_s
    counts[key] = counts.get(key, 0) + 1
    if "hybrid_total" in seconds:
        seconds["hybrid_total"] += elapsed_s
    return replace(
        cube,
        metadata=replace(
            cube.metadata,
            runtime_profile_s=seconds,
            runtime_profile_counts=counts,
        ),
    )


def _run_static_environment_pass(
    *,
    scene,
    radar: RadarSensor,
    target: MeshTarget,
    times: np.ndarray,
    simulator: MmWaveRadarSimulator,
    config: HybridPOConfig,
    reflector_bank=None,
) -> _StaticEnvironmentPass:
    """Traces static paths and synthesizes blocked/unblocked environment ADC."""

    simulator.check_cancelled()
    simulator.report_frame_progress(
        mode="hybrid_static_environment",
        frame=0,
        num_frames=times.shape[0],
    )
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
    runtime_profile_s["hybrid_segment_extraction"] = time.perf_counter() - extraction_t0
    runtime_profile_counts["hybrid_segment_extraction"] = 1

    if reflector_bank is None:
        reflector_t0 = time.perf_counter()
        reflector_bank = _select_reflectors(scene, radar, target, times, config)
        runtime_profile_s["hybrid_reflector_extraction"] = time.perf_counter() - reflector_t0
        runtime_profile_counts["hybrid_reflector_extraction"] = 1

    _, adc_complex_dtype = simulator.numpy_dtypes_for_precision(simulator.adc_compute_precision)
    static_unblocked_adc = np.zeros(
        times.shape + (radar.fmcw.num_adc_samples, radar.hardware.num_virtual_channels),
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
    # Blocking is the visibility of RT room paths, so it follows the RT
    # visibility cadence: refreshed at frame starts every N frames (0 = every
    # chirp), with fades that step only at frame starts.
    blocking_per_chirp = simulator.rt_visibility_refresh_frames == 0
    blocking_refresh_chirps = refresh_period_chirps(
        simulator.rt_visibility_refresh_frames,
        times.shape[1],
    )
    blocking_fade_steps = 1 if blocking_per_chirp else simulator.rt_visibility_fade_frames + 1
    last_raw_visibility = None
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
            simulator.report_chirp_progress(
                mode="hybrid_static_environment",
                frame=frame,
                num_frames=times.shape[0],
                chirp=chirp,
                num_chirps=times.shape[1],
            )
            refresh = (
                last_raw_visibility is None
                or (frame * times.shape[1] + chirp) % blocking_refresh_chirps == 0
            )
            vertices = None
            if refresh:
                vertices = target.mesh_sequence.vertices_at(float(times[frame, chirp]))
                visibility_scene.update_vertices(vertices)
            blocking_t0 = time.perf_counter()
            visibility = blocked_path_visibility(
                static_bank,
                visibility_scene,
                blocker_vertices=vertices,
                aabb_culling=config.blocking_aabb_culling,
                aabb_margin_m=config.blocking_aabb_margin_m,
                state=visibility_state,
                fade_steps=blocking_fade_steps,
                fade_advance=blocking_per_chirp or chirp == 0,
                refresh=refresh,
                previous_raw_visibility=last_raw_visibility,
                ray_epsilon_m=config.blocking_ray_epsilon_m,
                ray_chunk_size=config.blocking_ray_chunk_size,
            )
            if refresh:
                runtime_profile_s["hybrid_blocking_rays"] += time.perf_counter() - blocking_t0
                runtime_profile_counts["hybrid_blocking_rays"] += 1
            last_raw_visibility = visibility.raw_visibility
            visibility_state = visibility.state
            weights = visibility.visibility_weights
            blocked_counts[frame, chirp] = int(
                np.count_nonzero(visibility.blocked_paths & static_bank.valid)
            )

            visibility_weights[frame, chirp] = weights
            mean_visibility[frame, chirp] = (
                float(np.mean(weights[static_bank.valid])) if static_path_count > 0 else 0.0
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
            runtime_profile_s["hybrid_static_adc_synthesis"] += time.perf_counter() - static_adc_t0
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
    config: HybridPOConfig,
    reflector_bank,
    gain_map: np.ndarray,
    progress: bool,
    progress_every_frames: int,
) -> _CouplingPass:
    """Computes dynamic human-environment coupling ADC over the time grid."""

    simulator.check_cancelled()
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
    _, adc_complex_dtype = simulator.numpy_dtypes_for_precision(simulator.adc_compute_precision)
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
    if not config.hybrid_coupling or reflector_count <= 0:
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

    simulator.report_frame_progress(
        mode="hybrid_coupling",
        frame=0,
        num_frames=times.shape[0],
    )
    coupling_po_config = config.po_config
    coupling_compute_backend = (
        simulator.compute_backend
        if coupling_po_config.compute_backend is None
        else coupling_po_config.compute_backend
    )
    coupling_compute_precision = (
        simulator.compute_precision
        if coupling_po_config.compute_precision is None
        else coupling_po_config.compute_precision
    )
    # Coupling is PO on the body, so it follows the po_* cadence.
    coupling_incremental_update = coupling_po_config.incremental
    coupling_visibility_refresh_chirps = refresh_period_chirps(
        coupling_po_config.po_visibility_refresh_frames,
        times.shape[1],
    )
    incremental_state = CouplingIncrementalState() if coupling_incremental_update else None
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
            simulator.report_chirp_progress(
                mode="hybrid_coupling",
                frame=frame,
                num_frames=times.shape[0],
                chirp=chirp,
                num_chirps=times.shape[1],
            )
            chirp_index = frame * times.shape[1] + chirp
            chirp_time = float(times[frame, chirp])
            vertices = target.mesh_sequence.vertices_at(chirp_time)
            refresh_coupling_visibility = (
                cached_human_env_active_sets is None
                or cached_env_human_active_sets is None
                or chirp_index % coupling_visibility_refresh_chirps == 0
            )
            if refresh_coupling_visibility:
                visibility_scene.update_vertices(vertices)

            coupling_t0 = time.perf_counter()
            coupling = compute_env_human_coupling_channels(
                radar=radar,
                vertices=vertices,
                faces=target.mesh_sequence.faces,
                tx_positions=radar.world_tx_positions(),
                rx_positions=radar.world_rx_positions(),
                virtual_channel_tx_indices=(radar.hardware.virtual_tx_indices()),
                virtual_channel_rx_indices=(radar.hardware.virtual_rx_indices()),
                virtual_channel_order=radar.hardware.virtual_channel_order,
                frequency_hz=radar.fmcw.carrier_frequency,
                human_material=target.material,
                reflectors=reflector_bank,
                human_visibility_scene=visibility_scene,
                static_visibility_scene=scene.mi_scene,
                reflection_scale=config.hybrid_reflection_scale,
                backend=coupling_compute_backend,
                precision=coupling_compute_precision,
                po_config=coupling_po_config,
                incremental_state=incremental_state,
                human_env_active_sets=(
                    None if refresh_coupling_visibility else cached_human_env_active_sets
                ),
                env_human_active_sets=(
                    None if refresh_coupling_visibility else cached_env_human_active_sets
                ),
            )
            simulator.check_cancelled()
            cached_human_env_active_sets = coupling.human_env_active_sets
            cached_env_human_active_sets = coupling.env_human_active_sets
            coupling_elapsed_s = time.perf_counter() - coupling_t0
            runtime_profile_s["hybrid_coupling_channel"] += coupling_elapsed_s
            runtime_profile_counts["hybrid_coupling_channel"] += 1
            coupling_profile_s = coupling.runtime_profile_s or {}
            coupling_profile_counts = coupling.runtime_profile_counts or {}
            for name in ("coupling_recomputed_faces", "coupling_phase_updated_faces"):
                if name in coupling_profile_counts:
                    key = f"hybrid_{name}"
                    runtime_profile_counts[key] = (
                        runtime_profile_counts.get(key, 0) + coupling_profile_counts[name]
                    )
            runtime_profile_s["hybrid_coupling_setup"] += coupling_profile_s.get(
                "coupling_setup", 0.0
            ) + coupling_profile_s.get("coupling_assembly", 0.0)
            runtime_profile_counts["hybrid_coupling_setup"] += coupling_profile_counts.get(
                "coupling_setup", 0
            ) + coupling_profile_counts.get("coupling_assembly", 0)
            runtime_profile_s["hybrid_human_env_channel"] += coupling_profile_s.get(
                "human_env_channel", 0.0
            )
            runtime_profile_counts["hybrid_human_env_channel"] += coupling_profile_counts.get(
                "human_env_channel", 0
            )
            runtime_profile_s["hybrid_env_human_channel"] += coupling_profile_s.get(
                "env_human_channel", 0.0
            )
            runtime_profile_counts["hybrid_env_human_channel"] += coupling_profile_counts.get(
                "env_human_channel", 0
            )

            human_env_counts[frame, chirp] = int(np.count_nonzero(coupling.human_env_valid))
            env_human_counts[frame, chirp] = int(np.count_nonzero(coupling.env_human_valid))
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
                delay_response=coupling.human_env_delay_response,
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
                delay_response=coupling.env_human_delay_response,
            )
            env_human_adc_elapsed_s = time.perf_counter() - env_human_adc_t0
            runtime_profile_s["hybrid_human_env_adc_synthesis"] += human_env_adc_elapsed_s
            runtime_profile_counts["hybrid_human_env_adc_synthesis"] += 1
            runtime_profile_s["hybrid_env_human_adc_synthesis"] += env_human_adc_elapsed_s
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
