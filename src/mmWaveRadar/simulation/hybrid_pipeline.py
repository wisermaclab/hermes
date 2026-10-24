# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Reusable hybrid RT-path/PO pipeline helpers."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, replace
import functools
from pathlib import Path
import time
from typing import Callable

from .options import (
    ArraySettings,
    HybridExecutionSettings,
    HybridFrameSettings,
    HybridPhysicsSettings,
    HybridPipelineSettings,
    HybridRayTracingSettings,
)
from ..radar.adc import merge_if_filter_diagnostics
import numpy as np

from .calibration import (
    POCalibrationConfig,
    POCalibrationReference,
    calibration_gain_array,
    fit_first_pose_adc_calibration,
    fit_sequence_adc_calibration,
    rt_baseline_schedule_metadata,
)
from .cadence import refresh_period_chirps
from ..naming import reject_removed_arguments, validate_array_groups, validate_array_model
from .env_human_coupling import _reflector_support
from .hybrid_passes import (
    _run_coupling_pass,
    _run_static_environment_pass,
    _select_reflectors,
    _zeroed_metadata_arrays,
    _with_shared_reflector_profile,
)
from ..targets.mesh import EndpointTimeWindowMeshSequence as _FrameRangeMeshSequence
from ..radar import RadarSensor
from .types import (
    HybridPOConfig,
    POConfig,
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
from .subarray import group_pairs, merge_subarray_cubes, partition_event, run_group_pairs
from .rt_reference import RTBaselineResult, RTReferenceCollector
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
    hybrid_calibration: POCalibrationConfig
    po_config: POConfig
    base_config: HybridPOConfig
    hybrid_config: HybridPOConfig

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


def _collapse_tdm_rt_baseline_result(
    result: RTBaselineResult,
    radar: RadarSensor,
) -> RTBaselineResult:
    """Packs a physical-slot RT baseline into per-channel slow time."""

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
        reference_adc_power={
            component: np.stack(
                [
                    collapse_tdm_first_frame_adc(frame[:, None, :], radar)[:, 0, :]
                    for frame in power
                ]
            )
            for component, power in result.reference_adc_power.items()
        },
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


def _with_target_detached(*scene_arguments: str):
    """Runs a hybrid entry point with its mesh target out of the named scenes.

    An earlier RT run can leave the moving mesh in a scene. The static trace,
    reflector discovery, and static occlusion must not see that frozen copy.
    """

    def decorate(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            with ExitStack() as stack:
                detached = set()
                for name in scene_arguments:
                    scene = kwargs.get(name)
                    if scene is not None and id(scene) not in detached:
                        detached.add(id(scene))
                        stack.enter_context(kwargs["target"].detached_from_scene(scene))
                return function(*args, **kwargs)

        return wrapper

    return decorate


def _print_reflector_summary(reflector_bank, *, max_rows: int = 8):
    """Prints selected image planes and the relative-power selection summary."""

    reflector_count = int(reflector_bank.vertices.shape[0])
    selection = getattr(reflector_bank, "selection_metadata", None)
    if selection is not None:
        print(
            "Image-source reflector selection [candidates, retained, cutoff_dB]:",
            (selection["candidate_count"], reflector_count, selection["power_threshold_db"]),
            flush=True,
        )
        print(
            "Image-source estimated relative powers [name, dB]:",
            [
                (entry["object"], entry["relative_power_db"])
                for entry in selection["candidates"]
                if entry["retained"]
            ][:max_rows],
            flush=True,
        )
    else:
        print(f"Image-source reflectors: {reflector_count}", flush=True)
    if reflector_count <= 0:
        return
    support_vertices = [
        _reflector_support(reflector_bank, idx).reshape(-1, 3)
        for idx in range(min(reflector_count, max_rows))
    ]
    reflector_bounds = [
        (
            reflector_bank.object_names[idx],
            np.array2string(support_vertices[idx].min(axis=0), precision=3),
            np.array2string(support_vertices[idx].max(axis=0), precision=3),
        )
        for idx in range(min(reflector_count, max_rows))
    ]
    print("Reflector world bounds [name, min_xyz, max_xyz]:", reflector_bounds, flush=True)
    if reflector_bank.reflection_gains is not None:
        gains = np.asarray(reflector_bank.reflection_gains)[: min(reflector_count, max_rows)]
        print(
            "Static RT reference gains (diagnostic only) [name, |gain|, phase_deg]:",
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


def run_rt_baseline(
    *,
    scene,
    radar: RadarSensor,
    targets: list[MeshTarget],
    simulator: MmWaveRadarSimulator,
    num_frames: int,
    calibration_reference_component: POCalibrationReference | None = None,
) -> RTBaselineResult:
    """Run the simulator's RT engine and observe coherent calibration components."""
    if scene is None:
        raise ValueError("scene is required")
    if not targets:
        raise ValueError("at least one target is required")
    num_frames = int(num_frames)
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if calibration_reference_component is not None:
        POCalibrationConfig(reference_component=calibration_reference_component)
    if tdm_expansion_required(radar):
        result = run_rt_baseline(
            scene=scene,
            radar=expanded_tdm_radar(radar),
            targets=targets,
            simulator=simulator,
            num_frames=num_frames,
            calibration_reference_component=calibration_reference_component,
        )
        return _collapse_tdm_rt_baseline_result(result, radar)
    added_objects = []
    old_references = [
        None if target.reference_vertices is None else target.reference_vertices.copy()
        for target in targets
    ]
    try:
        for target in targets:
            obj = target.ensure_scene_object()
            if not (obj.scene is scene and obj.name in scene.objects):
                added_objects.append(obj)
        collector = RTReferenceCollector(
            simulator, radar, num_frames, calibration_reference_component
        )
        engine = (
            simulator._run_rt_path_bank if simulator.rt_coherent else simulator._run_rt_every_chirp
        )
        cube = engine(scene, radar, targets, num_frames=num_frames, reference_collector=collector)
        return collector.finish(cube)
    finally:
        if added_objects:
            try:
                scene.edit(remove=[obj for obj in added_objects if obj.name in scene.objects])
            finally:
                for obj in added_objects:
                    obj._scene = lambda: None
        for target, reference in zip(targets, old_references):
            target.reference_vertices = reference


def _validate_calibration_frame_range(
    calibration_frame_range: tuple[int, int],
    *,
    num_frames: int,
) -> tuple[int, int]:
    """Validates an inclusive/exclusive calibration frame range."""

    try:
        start, stop = calibration_frame_range
    except (TypeError, ValueError) as exc:
        raise ValueError("calibration_frame_range must be a (start, stop) tuple") from exc
    start = int(start)
    stop = int(stop)
    if start < 0:
        raise ValueError("calibration_frame_range start must be non-negative")
    if stop <= start:
        raise ValueError("calibration_frame_range stop must be greater than start")
    if stop > int(num_frames):
        raise ValueError("calibration_frame_range stop must be less than or equal to num_frames")
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
    result: RTBaselineResult,
    *,
    offset_s: float,
) -> RTBaselineResult:
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
    result: RTBaselineResult,
    *,
    rt_refresh_frames: int,
    retrace_period_chirps: int,
) -> dict:
    """Converts an RT baseline result into the calibration baseline dictionary."""

    events = result.cube.metadata.events or []
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
        "coherent_path_update": bool(events and events[0].get("coherent_path_update", False)),
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
    calibration_result: RTBaselineResult | None,
    human_cube: RadarCube,
    calibration_config: POCalibrationConfig,
    frame_start: int,
    frame_stop: int,
    rt_refresh_frames: int,
    retrace_period_chirps: int,
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
        rt_refresh_frames=rt_refresh_frames,
        retrace_period_chirps=retrace_period_chirps,
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
        gain, po_calibration = fit_sequence_adc_calibration(
            rt_baseline=rt_baseline,
            po_times=human_cube.times[frame_start:frame_stop],
            po_adc=human_cube.adc[frame_start:frame_stop],
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
    po_calibration["rt_reference_human_path_count_mean"] = float(np.mean(path_counts))
    po_calibration["rt_reference_human_path_count_max"] = int(np.max(path_counts))
    po_calibration["rt_baseline_retrace_count"] = int(rt_baseline.get("retrace_count", 0))
    po_calibration["rt_baseline_num_frames"] = int(frame_stop - frame_start)
    po_calibration.update(rt_baseline_schedule_metadata(rt_baseline))
    return gain, po_calibration


@_with_target_detached("scene")
def hybrid_rt_path_coupling(
    *,
    scene,
    radar: RadarSensor,
    target: MeshTarget,
    base_cube: RadarCube,
    simulator: MmWaveRadarSimulator,
    config: HybridPOConfig | None = None,
    base_wall_time_s: float = 0.0,
    base_cube_path: str | Path | None = None,
    recompute: bool = True,
    progress: bool | None = None,
    progress_every_frames: int = 10,
    print_reflector_summary: bool = True,
) -> HybridCouplingRecomputeResult:
    """Recomputes H_env/E_human from a cached H_direct/E_static base cube.

    The base cube is expected to contain the direct human PO and blocked static
    RT components. This helper selects image planes directly from the scene
    geometry, synthesizes the two single-bounce coupling components, and
    returns an assembled hybrid cube.
    """

    simulator.check_cancelled()
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

    if config is None:
        if not simulator._hybrid_config_explicit:
            raise ValueError(
                "an explicit HybridPOConfig is required for coupling recomputation; "
                "pass config or construct the simulator with hybrid_config"
            )
        config = simulator.hybrid_config
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
    profile_s = dict(base_cube.metadata.runtime_profile_s or {})
    profile_counts = dict(base_cube.metadata.runtime_profile_counts or {})
    _reset_runtime_keys(profile_s, profile_counts)

    reflector_bank = None
    if recompute:
        simulator.adc_if_filter = {}
        total_t0 = time.perf_counter()
        radar.configure_scene(scene)
        reflector_t0 = time.perf_counter()
        reflector_bank = _select_reflectors(scene, radar, target, times, config)
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
        coupling_pass = _run_coupling_pass(
            scene=scene,
            radar=radar,
            target=target,
            times=times,
            simulator=simulator,
            config=config,
            reflector_bank=reflector_bank,
            gain_map=human_po_gain_map,
            progress=progress,
            progress_every_frames=progress_every_frames,
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
        profile_s.update(coupling_pass.runtime_profile_s)
        profile_counts.update(coupling_pass.runtime_profile_counts)
        coupling_wall_time_s = time.perf_counter() - total_t0
    else:
        _require_components(base_cube, {"human_env", "env_human"})
        if progress:
            print(
                "Using coupling components from loaded cube because recompute=False",
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
        "coupling_candidate_source": "scene_geometry_image_source_power",
        "coupling_recomputed": bool(recompute),
        "hybrid_reflector_cutoff_db": config.hybrid_reflector_cutoff_db,
        "hybrid_max_reflectors": int(config.hybrid_max_reflectors),
        "reflector_selection": getattr(reflector_bank, "selection_metadata", None),
        "po_calibration_gain": (base_cube.metadata.po_calibration or {}).get(
            "human_po_amplitude_gain", 1.0
        ),
    }
    if base_cube_path is not None:
        event["base_cube_path"] = str(base_cube_path)
    receiver_components = dict((base_cube.metadata.adc_if_filter or {}).get("components", {}))
    if recompute:
        if not receiver_components and base_cube.metadata.adc_if_filter:
            receiver_components["base"] = base_cube.metadata.adc_if_filter
        receiver_components["coupling"] = getattr(simulator, "adc_if_filter", {})
    hybrid_metadata = replace(
        base_cube.metadata,
        adc_if_filter=(
            merge_if_filter_diagnostics(**receiver_components)
            if recompute
            else base_cube.metadata.adc_if_filter
        ),
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
        components["rt_baseline_one_human_first_frame"] = base_cube.components[
            "rt_baseline_one_human_first_frame"
        ]
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


def _run_calibrated_hybrid_po_subarrays(call: dict, check_cancelled):
    """Runs the calibrated flow per Tx/Rx group pair and merges the results."""

    radar = call["radar"]
    pairs = group_pairs(
        radar,
        validate_array_groups(call["array_tx_groups"], name="array_tx_groups"),
        validate_array_groups(call["array_rx_groups"], name="array_rx_groups"),
    )
    callback = call["progress_callback"]
    check_cancelled()
    environment_scene = (
        call["coupling_scene"] if call["coupling_scene"] is not None else call["scene"]
    )
    times = np.array(
        [
            radar.fmcw.chirp_time(0, 0),
            radar.fmcw.chirp_time(call["num_frames"] - 1, radar.fmcw.num_chirps_per_frame - 1),
        ]
    )
    reflector_bank = call["_reflector_bank"]
    selection_s = 0.0
    if reflector_bank is None:
        selection_t0 = time.perf_counter()
        reflector_bank = _select_reflectors(
            environment_scene,
            radar,
            call["target"],
            times,
            call["hybrid_config"],
        )
        selection_s = time.perf_counter() - selection_t0

    def execute(subradar, report):
        return run_calibrated_hybrid_po(
            **{
                **call,
                "radar": subradar,
                "array_model": "phase_center",
                "array_tx_groups": None,
                "array_rx_groups": None,
                "progress_callback": report,
                "_reflector_bank": reflector_bank,
            }
        )

    results = run_group_pairs(
        radar,
        pairs,
        execute,
        scenes=(call["scene"], environment_scene),
        callback=callback,
        check_cancelled=check_cancelled,
        progress=call["progress"],
    )
    channels = radar.hardware.num_virtual_channels
    cube = merge_subarray_cubes(
        [result.cube for result in results],
        pairs,
        channels,
        solver="hybrid_po",
    )
    base_cube = merge_subarray_cubes(
        [result.base_cube for result in results],
        pairs,
        channels,
        solver="hybrid_po",
    )
    if call["_reflector_bank"] is None:
        cube = _with_shared_reflector_profile(cube, selection_s)
        base_cube = _with_shared_reflector_profile(base_cube, selection_s)
    event = partition_event(pairs)
    cube = replace(
        cube,
        metadata=replace(
            cube.metadata,
            events=[event, *cube.metadata.events],
        ),
    )
    coupling_result = replace(
        results[0].coupling_result,
        cube=cube,
        wall_time_s=float(sum(r.coupling_result.wall_time_s for r in results) + selection_s),
        num_chirps=int(np.prod(cube.adc.shape[:2])),
        coupling_wall_time_s=float(sum(r.coupling_result.coupling_wall_time_s for r in results)),
    )
    return replace(
        results[0],
        cube=cube,
        base_cube=base_cube,
        coupling_result=coupling_result,
        base_wall_time_s=float(sum(result.base_wall_time_s for result in results) + selection_s),
        base_num_chirps=int(np.prod(base_cube.adc.shape[:2])),
    )


def run_calibrated_hybrid_po(
    *,
    scene,
    radar: RadarSensor,
    target: MeshTarget,
    num_frames: int,
    calibration_frame_range: tuple[int, int] = (0, 1),
    coupling_scene=None,
    po_visibility_samples_per_face: int | None = None,
    hybrid_coupling: bool = True,
    hybrid_reflector_cutoff_db: float = -20.0,
    hybrid_max_reflectors: int = 4,
    hybrid_reflection_scale: float = 1.0,
    hybrid_calibration: POCalibrationConfig | str | None = None,
    rt_samples_per_source: int = 80_000,
    rt_max_paths_per_source: int = 80_000,
    rt_max_depth: int = 2,
    rt_human_bounces: str = "single",
    rt_visibility_refresh_frames: int = 1,
    rt_visibility_fade_frames: int = 0,
    rt_diffuse_reflection: bool = True,
    rt_seed: int = 42,
    array_model: str = "phase_center",
    array_tx_groups=None,
    array_rx_groups=None,
    compute_backend="auto",
    compute_precision="float32",
    adc_compute_precision=None,
    progress: bool = True,
    progress_every_frames: int = 10,
    print_reflector_summary: bool = True,
    po_config: POConfig | None = None,
    hybrid_config: HybridPOConfig | None = None,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    _reflector_bank=None,
    **removed,
) -> CalibratedHybridPOResult:
    """Runs Hybrid PO as an explicit six-stage algorithm flow.

    Stage 1 fits the PO amplitude to an RT reference according to
    ``hybrid_calibration`` (default ``rt_first_pose``; ``none`` keeps the
    absolute PO amplitude). ``rt_first_pose`` compares single-human-only RT
    with human-only PO, averaging ADC power over all chirps, virtual channels,
    and samples in the first selected calibration frame (frame 0 by default).
    One amplitude gain is reused for every frame and chirp of direct-human PO
    and both coupling orders; static-environment RT is not rescaled.
    When ``hybrid_config`` is given it supplies the
    coupling, reflector, and calibration settings; otherwise they come from
    the ``hybrid_*`` arguments. PO defaults always come from ``POConfig``;
    an explicit ``po_config`` overrides the hybrid PO configuration, and an
    explicit ``po_visibility_samples_per_face`` overrides its sample count.
    Blocking of room paths by the body is always
    on and follows ``rt_visibility_refresh_frames``. With
    ``array_model="subarray"`` the flow runs once per pair of a Tx group and an
    Rx group, and the pairs share one calibration gain.
    """
    reject_removed_arguments(removed, where="run_calibrated_hybrid_po")
    return run_hybrid_po_pipeline(
        settings=HybridPipelineSettings(
            frames=HybridFrameSettings(
                num_frames=num_frames, calibration_frame_range=calibration_frame_range
            ),
            array=ArraySettings(
                array_model=array_model,
                array_tx_groups=array_tx_groups,
                array_rx_groups=array_rx_groups,
            ),
            rt=HybridRayTracingSettings(
                rt_samples_per_source=rt_samples_per_source,
                rt_max_paths_per_source=rt_max_paths_per_source,
                rt_max_depth=rt_max_depth,
                rt_human_bounces=rt_human_bounces,
                rt_visibility_refresh_frames=rt_visibility_refresh_frames,
                rt_visibility_fade_frames=rt_visibility_fade_frames,
                rt_diffuse_reflection=rt_diffuse_reflection,
                rt_seed=rt_seed,
            ),
            physics=HybridPhysicsSettings(
                po_visibility_samples_per_face=po_visibility_samples_per_face,
                hybrid_coupling=hybrid_coupling,
                hybrid_reflector_cutoff_db=hybrid_reflector_cutoff_db,
                hybrid_max_reflectors=hybrid_max_reflectors,
                hybrid_reflection_scale=hybrid_reflection_scale,
                hybrid_calibration=hybrid_calibration,
                po_config=po_config,
                hybrid_config=hybrid_config,
            ),
            execution=HybridExecutionSettings(
                compute_backend=compute_backend,
                compute_precision=compute_precision,
                adc_compute_precision=adc_compute_precision,
                progress=progress,
                progress_every_frames=progress_every_frames,
                print_reflector_summary=print_reflector_summary,
                cancel_check=cancel_check,
                progress_callback=progress_callback,
            ),
        ),
        scene=scene,
        radar=radar,
        target=target,
        coupling_scene=coupling_scene,
        _reflector_bank=_reflector_bank,
    )


@_with_target_detached("scene", "coupling_scene")
def run_hybrid_po_pipeline(
    *,
    scene,
    radar: RadarSensor,
    target: MeshTarget,
    settings: HybridPipelineSettings,
    coupling_scene=None,
    _reflector_bank=None,
) -> CalibratedHybridPOResult:
    """Run with grouped settings; the legacy keyword entry point remains supported."""
    if not isinstance(settings, HybridPipelineSettings):
        raise TypeError("settings must be a HybridPipelineSettings")
    return _HybridPipelineRun(
        settings, scene, radar, target, coupling_scene, _reflector_bank
    ).run()


class _HybridPipelineRun:
    """Per-run preparation, solver passes and result construction."""

    def __init__(
        self,
        settings: HybridPipelineSettings,
        scene,
        radar,
        target,
        coupling_scene,
        _reflector_bank,
    ):
        self.settings = settings
        self.array_model = self.settings.array.array_model
        self.array_rx_groups = self.settings.array.array_rx_groups
        self.array_tx_groups = self.settings.array.array_tx_groups
        self.hybrid_config = self.settings.physics.hybrid_config
        self.num_frames = self.settings.frames.num_frames
        self.po_config = self.settings.physics.po_config
        self.scene = scene
        self.radar = radar
        self.target = target
        self.coupling_scene = coupling_scene
        self._reflector_bank = _reflector_bank
        self.progress = self.settings.execution.progress
        self.progress_every_frames = self.settings.execution.progress_every_frames
        self.cancel_check = self.settings.execution.cancel_check
        self.progress_callback = self.settings.execution.progress_callback

    def check_cancelled(self):
        if self.cancel_check is not None and bool(self.cancel_check()):
            raise InterruptedError("Simulation cancelled by user")

    def print_stage(self, stage: int, message: str, *, elapsed_s: float | None = None):
        """Prints one progress line when progress reporting is enabled."""

        if not self.progress:
            return
        suffix = "" if elapsed_s is None else f" elapsed={elapsed_s:.2f}s"
        print(
            f"[hybrid-po] stage {stage}/{self.total_stages}: {message}{suffix}",
            flush=True,
        )

    def _stage_prepare(self):
        """Prepare this run using its settings and prior stage outputs."""
        self.array_model = validate_array_model(self.array_model)
        self.array_tx_groups = validate_array_groups(self.array_tx_groups, name="array_tx_groups")
        self.array_rx_groups = validate_array_groups(self.array_rx_groups, name="array_rx_groups")
        if self.array_model != "subarray" and (
            self.array_tx_groups is not None or self.array_rx_groups is not None
        ):
            raise ValueError("array_tx_groups and array_rx_groups need array_model='subarray'")
        if self.po_config is None:
            self.po_config = (
                POConfig() if self.hybrid_config is None else self.hybrid_config.po_config
            )
        if self.settings.physics.po_visibility_samples_per_face is not None:
            self.po_config = replace(
                self.po_config,
                po_visibility_samples_per_face=self.settings.physics.po_visibility_samples_per_face,
            )
        if self.hybrid_config is None:
            self.hybrid_config = HybridPOConfig(
                hybrid_coupling=self.settings.physics.hybrid_coupling,
                hybrid_reflector_cutoff_db=self.settings.physics.hybrid_reflector_cutoff_db,
                hybrid_max_reflectors=self.settings.physics.hybrid_max_reflectors,
                hybrid_reflection_scale=self.settings.physics.hybrid_reflection_scale,
                po_config=self.po_config,
                **(
                    {}
                    if self.settings.physics.hybrid_calibration is None
                    else {"hybrid_calibration": self.settings.physics.hybrid_calibration}
                ),
            )
        else:
            self.hybrid_config = replace(
                self.hybrid_config,
                po_config=self.po_config,
                **(
                    {}
                    if self.settings.physics.hybrid_calibration is None
                    else {"hybrid_calibration": self.settings.physics.hybrid_calibration}
                ),
            )
        self.num_frames = int(self.num_frames)
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if self.cancel_check is not None and not callable(self.cancel_check):
            raise ValueError("cancel_check must be callable or None")
        if self.progress_callback is not None and not callable(self.progress_callback):
            raise ValueError("progress_callback must be callable or None")
        call = {
            "scene": self.scene,
            "radar": self.radar,
            "target": self.target,
            "num_frames": self.num_frames,
            "calibration_frame_range": self.settings.frames.calibration_frame_range,
            "coupling_scene": self.coupling_scene,
            "po_visibility_samples_per_face": self.settings.physics.po_visibility_samples_per_face,
            "hybrid_coupling": self.settings.physics.hybrid_coupling,
            "hybrid_reflector_cutoff_db": self.settings.physics.hybrid_reflector_cutoff_db,
            "hybrid_max_reflectors": self.settings.physics.hybrid_max_reflectors,
            "hybrid_reflection_scale": self.settings.physics.hybrid_reflection_scale,
            "hybrid_calibration": self.settings.physics.hybrid_calibration,
            "rt_samples_per_source": self.settings.rt.rt_samples_per_source,
            "rt_max_paths_per_source": self.settings.rt.rt_max_paths_per_source,
            "rt_max_depth": self.settings.rt.rt_max_depth,
            "rt_human_bounces": self.settings.rt.rt_human_bounces,
            "rt_visibility_refresh_frames": self.settings.rt.rt_visibility_refresh_frames,
            "rt_visibility_fade_frames": self.settings.rt.rt_visibility_fade_frames,
            "rt_diffuse_reflection": self.settings.rt.rt_diffuse_reflection,
            "rt_seed": self.settings.rt.rt_seed,
            "array_model": self.array_model,
            "array_tx_groups": self.array_tx_groups,
            "array_rx_groups": self.array_rx_groups,
            "compute_backend": self.settings.execution.compute_backend,
            "compute_precision": self.settings.execution.compute_precision,
            "adc_compute_precision": self.settings.execution.adc_compute_precision,
            "progress": self.progress,
            "progress_every_frames": self.progress_every_frames,
            "print_reflector_summary": self.settings.execution.print_reflector_summary,
            "po_config": self.po_config,
            "hybrid_config": self.hybrid_config,
            "cancel_check": self.cancel_check,
            "progress_callback": self.progress_callback,
            "_reflector_bank": self._reflector_bank,
        }

        if tdm_expansion_required(self.radar):
            expanded_result = run_calibrated_hybrid_po(
                **{**call, "radar": expanded_tdm_radar(self.radar)}
            )
            cube = collapse_tdm_cube(expanded_result.cube, self.radar)
            self.base_cube = collapse_tdm_cube(expanded_result.base_cube, self.radar)
            self.coupling_result = replace(
                expanded_result.coupling_result,
                cube=cube,
                num_chirps=int(np.prod(cube.adc.shape[:2])),
            )
            return replace(
                expanded_result,
                cube=cube,
                base_cube=self.base_cube,
                coupling_result=self.coupling_result,
                base_num_chirps=int(np.prod(self.base_cube.adc.shape[:2])),
            )

        if self.array_model == "subarray":
            return _run_calibrated_hybrid_po_subarrays(call, self.check_cancelled)

        self.check_cancelled()

    def _stage_setup(self):
        """Setup this run using its settings and prior stage outputs."""
        self.calibration_frame_start, self.calibration_frame_stop = (
            _validate_calibration_frame_range(
                self.settings.frames.calibration_frame_range,
                num_frames=self.num_frames,
            )
        )

        self.pipeline_t0 = time.perf_counter()
        self.total_stages = 6

        # The pipeline fits the calibration itself, so the PO and Hybrid passes
        # run with absolute amplitudes.
        self.calibration_config = self.hybrid_config.hybrid_calibration
        if (
            self.calibration_config.mode != "none"
            and self.calibration_config.reference_component != "single_human_only"
        ):
            raise ValueError(
                "Hybrid PO calibration requires reference_component="
                "'single_human_only'; environment coupling is synthesized separately"
            )
        self.po_config = replace(self.po_config, po_calibration="none")
        self.hybrid_config = replace(
            self.hybrid_config, hybrid_calibration="none", po_config=self.po_config
        )
        self.base_config = replace(self.hybrid_config, hybrid_coupling=False)

        self.environment_scene = self.scene if self.coupling_scene is None else self.coupling_scene
        self.calibration_refresh_frames = int(self.calibration_config.rt_refresh_frames)
        self.retrace_period_chirps = refresh_period_chirps(
            self.calibration_refresh_frames,
            self.radar.fmcw.num_chirps_per_frame,
        )
        self.simulator_kwargs = dict(
            array_model=self.array_model,
            compute_backend=self.settings.execution.compute_backend,
            compute_precision=self.settings.execution.compute_precision,
            progress=self.progress,
            progress_interval=self.progress_every_frames,
            rt_human_bounces=self.settings.rt.rt_human_bounces,
            rt_max_depth=self.settings.rt.rt_max_depth,
            rt_samples_per_source=self.settings.rt.rt_samples_per_source,
            rt_max_paths_per_source=self.settings.rt.rt_max_paths_per_source,
            rt_diffuse_reflection=self.settings.rt.rt_diffuse_reflection,
            rt_refresh_frames=self.calibration_refresh_frames,
            rt_visibility_refresh_frames=self.settings.rt.rt_visibility_refresh_frames,
            rt_visibility_fade_frames=self.settings.rt.rt_visibility_fade_frames,
            rt_seed=self.settings.rt.rt_seed,
            cancel_check=self.cancel_check,
            progress_callback=self.progress_callback,
        )
        if self.settings.execution.adc_compute_precision is not None:
            self.simulator_kwargs["adc_compute_precision"] = (
                self.settings.execution.adc_compute_precision
            )

    def _stage_calibration(self):
        """Calibration this run using its settings and prior stage outputs."""
        self.calibration_result = None
        self.calibration_wall_time_s = 0.0
        if self.calibration_config.mode == "none":
            self.print_stage(1, "calibration RT skipped because calibration mode is none")
        else:
            self.check_cancelled()
            self.print_stage(
                1,
                (
                    "full-scene RT calibration "
                    f"frames=[{self.calibration_frame_start}, {self.calibration_frame_stop})"
                ),
            )
            calibration_simulator = MmWaveRadarSimulator(
                solver="rt",
                **self.simulator_kwargs,
            )
            calibration_target = _frame_range_target(
                self.target,
                self.radar,
                frame_start=self.calibration_frame_start,
                frame_stop=self.calibration_frame_stop,
            )
            calibration_t0 = time.perf_counter()
            self.calibration_result = run_rt_baseline(
                scene=self.scene,
                radar=self.radar,
                targets=[calibration_target],
                simulator=calibration_simulator,
                num_frames=self.calibration_frame_stop - self.calibration_frame_start,
                calibration_reference_component=(
                    self.calibration_config.reference_component
                    if self.calibration_config.mode == "rt_sequence"
                    else None
                ),
            )
            self.calibration_result = _shift_rt_result_times(
                self.calibration_result,
                offset_s=float(self.radar.fmcw.chirp_time(self.calibration_frame_start, 0)),
            )
            self.calibration_wall_time_s = time.perf_counter() - calibration_t0
            if self.progress:
                print(
                    "[hybrid-po] stage 1/6 complete: "
                    f"{self.calibration_wall_time_s:.2f}s "
                    f"retraces={self.calibration_result.retrace_count}",
                    flush=True,
                )

    def _stage_human(self):
        """Human this run using its settings and prior stage outputs."""
        self.print_stage(
            2,
            (
                "human-only PO over full window and scalar calibration "
                f"mode={self.calibration_config.mode}/{self.calibration_config.reference_component}"
            ),
            elapsed_s=time.perf_counter() - self.pipeline_t0,
        )
        self.check_cancelled()
        human_simulator = MmWaveRadarSimulator(
            solver="po",
            **self.simulator_kwargs,
            po_config=self.po_config,
        )
        self.radar.configure_scene(self.environment_scene)
        human_t0 = time.perf_counter()
        self.raw_human_cube = human_simulator._run_po(  # pylint: disable=protected-access
            self.radar,
            [self.target],
            num_frames=self.num_frames,
            static_visibility_scene=self.environment_scene.mi_scene,
            po_config=self.po_config,
        )
        self.human_wall_time_s = time.perf_counter() - human_t0
        calibration_fit_t0 = time.perf_counter()
        human_po_gain, self.po_calibration = _calibrate_human_po(
            calibration_result=self.calibration_result,
            human_cube=self.raw_human_cube,
            calibration_config=self.calibration_config,
            frame_start=self.calibration_frame_start,
            frame_stop=self.calibration_frame_stop,
            rt_refresh_frames=self.calibration_refresh_frames,
            retrace_period_chirps=self.retrace_period_chirps,
        )
        self.calibration_fit_wall_time_s = time.perf_counter() - calibration_fit_t0
        self.gain_map = calibration_gain_array(
            self.po_calibration,
            self.raw_human_cube.times.shape,
            fallback=human_po_gain,
        )
        _, self.adc_complex_dtype = human_simulator.numpy_dtypes_for_precision(
            human_simulator.adc_compute_precision
        )
        self.human_po_adc = (self.raw_human_cube.adc * self.gain_map[..., None, None]).astype(
            self.adc_complex_dtype,
            copy=False,
        )
        if self.progress:
            gain_values = np.asarray(self.gain_map, dtype=float)
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
                f"retraces={self.po_calibration.get('rt_baseline_retrace_count', 'n/a')} "
                f"frames={self.po_calibration.get('rt_baseline_num_frames', 'n/a')} "
                f"runtime={self.calibration_wall_time_s:.2f}s",
                flush=True,
            )
            print(
                "[hybrid-po] stage 2/6 complete: "
                f"human_po={self.human_wall_time_s:.2f}s "
                f"calibration_fit={self.calibration_fit_wall_time_s:.2f}s",
                flush=True,
            )

    def _stage_environment(self):
        """Environment this run using its settings and prior stage outputs."""
        self.print_stage(
            3,
            "static-environment RT, dynamic human blocking, reflector selection",
            elapsed_s=time.perf_counter() - self.pipeline_t0,
        )
        self.check_cancelled()
        self.static_simulator = MmWaveRadarSimulator(
            solver="rt",
            **self.simulator_kwargs,
            hybrid_config=self.hybrid_config,
        )
        self.static_pass = _run_static_environment_pass(
            scene=self.environment_scene,
            radar=self.radar,
            target=self.target,
            times=self.raw_human_cube.times,
            simulator=self.static_simulator,
            config=self.hybrid_config,
            reflector_bank=self._reflector_bank,
        )
        if self.progress:
            print(
                "[hybrid-po] stage 3/6 complete: "
                f"{self.static_pass.wall_time_s:.2f}s "
                f"static_paths={int(np.count_nonzero(self.static_pass.static_bank.valid))} "
                f"reflectors={int(self.static_pass.reflector_bank.vertices.shape[0])}",
                flush=True,
            )
            if self.settings.execution.print_reflector_summary:
                _print_reflector_summary(self.static_pass.reflector_bank)

    def _stage_coupling(self):
        """Coupling this run using its settings and prior stage outputs."""
        self.print_stage(
            4,
            "PO for H_env/E_human using RT-selected image sources",
            elapsed_s=time.perf_counter() - self.pipeline_t0,
        )
        self.check_cancelled()
        self.coupling_simulator = MmWaveRadarSimulator(
            solver="rt",
            **self.simulator_kwargs,
            hybrid_config=self.hybrid_config,
        )
        self.coupling_pass = _run_coupling_pass(
            scene=self.environment_scene,
            radar=self.radar,
            target=self.target,
            times=self.raw_human_cube.times,
            simulator=self.coupling_simulator,
            config=self.hybrid_config,
            reflector_bank=self.static_pass.reflector_bank,
            gain_map=self.gain_map,
            progress=self.progress,
            progress_every_frames=self.progress_every_frames,
        )
        if self.progress:
            print(
                "[hybrid-po] stage 4/6 complete: "
                f"H_env mean paths={np.mean(self.coupling_pass.human_env_counts):.1f}",
                flush=True,
            )

        self.print_stage(
            5,
            "report E_human from shared coupling pass",
            elapsed_s=time.perf_counter() - self.pipeline_t0,
        )
        self.check_cancelled()
        if self.progress:
            print(
                "[hybrid-po] stage 5/6 complete: "
                f"E_human mean paths={np.mean(self.coupling_pass.env_human_counts):.1f}",
                flush=True,
            )

    def _stage_products(self):
        """Products this run using its settings and prior stage outputs."""
        self.print_stage(
            6,
            "assemble coherent Hybrid PO cube",
            elapsed_s=time.perf_counter() - self.pipeline_t0,
        )
        self.check_cancelled()
        self.runtime_profile_s = {
            "hybrid_rt_baseline": self.calibration_wall_time_s,
            "hybrid_po_calibration": self.calibration_fit_wall_time_s,
            "hybrid_human_po": self.human_wall_time_s,
            **self.static_pass.runtime_profile_s,
            **self.coupling_pass.runtime_profile_s,
        }
        self.runtime_profile_counts = {
            "hybrid_rt_baseline": 1 if self.calibration_result is not None else 0,
            "hybrid_po_calibration": 1,
            "hybrid_human_po": 1,
            **self.static_pass.runtime_profile_counts,
            **self.coupling_pass.runtime_profile_counts,
        }
        if self.raw_human_cube.metadata.runtime_profile_s:
            for key, value in self.raw_human_cube.metadata.runtime_profile_s.items():
                self.runtime_profile_s[f"human_{key}"] = float(value)
        if self.raw_human_cube.metadata.runtime_profile_counts:
            for key, value in self.raw_human_cube.metadata.runtime_profile_counts.items():
                self.runtime_profile_counts[f"human_{key}"] = int(value)

        assembly_t0 = time.perf_counter()
        self.hybrid_adc = (
            self.human_po_adc
            + self.static_pass.static_blocked_adc
            + self.coupling_pass.human_env_adc
            + self.coupling_pass.env_human_adc
        ).astype(self.adc_complex_dtype)
        assembly_wall_time_s = time.perf_counter() - assembly_t0
        self.runtime_profile_s["hybrid_assembly"] = assembly_wall_time_s
        self.runtime_profile_counts["hybrid_assembly"] = 1
        self.base_wall_time_s = (
            self.calibration_wall_time_s
            + self.calibration_fit_wall_time_s
            + self.human_wall_time_s
            + self.static_pass.wall_time_s
        )
        self.hybrid_wall_time_s = self.base_wall_time_s + self.coupling_pass.wall_time_s
        self.runtime_profile_s["hybrid_total"] = self.hybrid_wall_time_s
        self.runtime_profile_counts["hybrid_total"] = 1

        human_counts = self.raw_human_cube.metadata.path_counts
        self.path_counts = (
            self.static_pass.static_path_counts
            + human_counts
            + self.coupling_pass.human_env_counts
            + self.coupling_pass.env_human_counts
        )

    def _stage_metadata(self):
        """Metadata this run using its settings and prior stage outputs."""
        events = [
            {
                "reason": "hybrid_po_calibration_rt",
                "calibration_frame_range": (
                    int(self.calibration_frame_start),
                    int(self.calibration_frame_stop),
                ),
                "calibration_mode": self.calibration_config.mode,
                "rt_calibration_frames": int(
                    self.calibration_frame_stop - self.calibration_frame_start
                ),
            },
            {
                "reason": "static_environment_trace",
                "static_path_count": int(np.count_nonzero(self.static_pass.static_bank.valid)),
                "segment_count": int(self.static_pass.static_bank.segment_path_indices.size),
                "hybrid_coupling": bool(self.hybrid_config.hybrid_coupling),
                "coupling_reflector_count": int(self.static_pass.reflector_bank.vertices.shape[0]),
                "coupling_candidate_source": "scene_geometry_image_source_power",
                "hybrid_reflector_cutoff_db": self.hybrid_config.hybrid_reflector_cutoff_db,
                "hybrid_max_reflectors": int(self.hybrid_config.hybrid_max_reflectors),
                "reflector_selection": getattr(
                    self.static_pass.reflector_bank, "selection_metadata", None
                ),
                "po_calibration_gain": self.po_calibration.get(
                    "human_po_amplitude_gain",
                    1.0,
                ),
            },
            {"reason": "po_calibration", **self.po_calibration},
        ]
        self.metadata = SensingMetadata(
            adc_if_filter=merge_if_filter_diagnostics(
                human_po=self.raw_human_cube.metadata.adc_if_filter,
                static=getattr(self.static_simulator, "adc_if_filter", {}),
                coupling=getattr(self.coupling_simulator, "adc_if_filter", {}),
            ),
            events=events,
            path_counts=self.path_counts,
            max_displacements=self.raw_human_cube.metadata.max_displacements,
            virtual_channel_order=self.radar.hardware.virtual_channel_order,
            solver="hybrid_po",
            visible_face_counts=self.raw_human_cube.metadata.visible_face_counts,
            effective_visible_area_fraction=(
                self.raw_human_cube.metadata.effective_visible_area_fraction
            ),
            runtime_per_chirp_s=self.raw_human_cube.metadata.runtime_per_chirp_s,
            max_unwrapped_phase_jump=self.raw_human_cube.metadata.max_unwrapped_phase_jump,
            runtime_profile_s=self.runtime_profile_s,
            runtime_profile_counts=self.runtime_profile_counts,
            incremental_recomputed_face_counts=(
                self.raw_human_cube.metadata.incremental_recomputed_face_counts
            ),
            incremental_phase_updated_face_counts=(
                self.raw_human_cube.metadata.incremental_phase_updated_face_counts
            ),
            incremental_full_refresh=self.raw_human_cube.metadata.incremental_full_refresh,
            incremental_visibility_refresh=(
                self.raw_human_cube.metadata.incremental_visibility_refresh
            ),
            static_path_counts=self.static_pass.static_path_counts,
            blocked_static_path_counts=self.static_pass.blocked_counts,
            mean_static_path_visibility=self.static_pass.mean_visibility,
            human_env_path_counts=self.coupling_pass.human_env_counts,
            env_human_path_counts=self.coupling_pass.env_human_counts,
            human_env_min_path_lengths_m=self.coupling_pass.human_env_min_lengths,
            human_env_mean_path_lengths_m=self.coupling_pass.human_env_mean_lengths,
            human_env_max_path_lengths_m=self.coupling_pass.human_env_max_lengths,
            env_human_min_path_lengths_m=self.coupling_pass.env_human_min_lengths,
            env_human_mean_path_lengths_m=self.coupling_pass.env_human_mean_lengths,
            env_human_max_path_lengths_m=self.coupling_pass.env_human_max_lengths,
            po_calibration=self.po_calibration,
        )

    def _stage_result(self):
        """Result this run using its settings and prior stage outputs."""
        self.base_cube = RadarCube(
            adc=(self.human_po_adc + self.static_pass.static_blocked_adc).astype(
                self.adc_complex_dtype
            ),
            times=self.raw_human_cube.times,
            metadata=self.metadata,
            components={
                "human_po": self.human_po_adc.copy(),
                "static_environment_unblocked": self.static_pass.static_unblocked_adc.copy(),
                "static_environment_blocked": self.static_pass.static_blocked_adc.copy(),
                "static_path_visibility_weights": self.static_pass.visibility_weights.copy(),
            },
        )
        hybrid_cube = RadarCube(
            adc=self.hybrid_adc,
            times=self.raw_human_cube.times,
            metadata=self.metadata,
            components={
                "human_po": self.human_po_adc.copy(),
                "static_environment_unblocked": self.static_pass.static_unblocked_adc.copy(),
                "static_environment_blocked": self.static_pass.static_blocked_adc.copy(),
                "human_env": self.coupling_pass.human_env_adc.copy(),
                "env_human": self.coupling_pass.env_human_adc.copy(),
                "static_path_visibility_weights": self.static_pass.visibility_weights.copy(),
            },
        )
        self.coupling_result = HybridCouplingRecomputeResult(
            cube=hybrid_cube,
            wall_time_s=self.hybrid_wall_time_s,
            num_chirps=int(np.prod(hybrid_cube.adc.shape[:2])),
            coupling_wall_time_s=self.coupling_pass.wall_time_s,
            reflector_count=int(self.static_pass.reflector_bank.vertices.shape[0]),
        )
        if self.progress:
            print(
                "[hybrid-po] stage 6/6 complete: "
                f"shape={hybrid_cube.adc.shape} total={self.hybrid_wall_time_s:.2f}s",
                flush=True,
            )

        return CalibratedHybridPOResult(
            cube=hybrid_cube,
            base_cube=self.base_cube,
            coupling_result=self.coupling_result,
            base_wall_time_s=self.base_wall_time_s,
            base_num_chirps=int(np.prod(self.base_cube.adc.shape[:2])),
            hybrid_calibration=self.calibration_config,
            po_config=self.po_config,
            base_config=self.base_config,
            hybrid_config=self.hybrid_config,
        )

    def run(self):
        early_result = self._stage_prepare()
        if early_result is not None:
            return early_result
        self._stage_setup()
        self._stage_calibration()
        self._stage_human()
        self._stage_environment()
        self._stage_coupling()
        self._stage_products()
        self._stage_metadata()
        return self._stage_result()


def run_full_scene_rt_comparison(
    *,
    scene,
    radar: RadarSensor,
    target: MeshTarget,
    num_frames: int,
    rt_samples_per_source: int = 80_000,
    rt_max_paths_per_source: int = 80_000,
    rt_max_depth: int = 2,
    rt_human_bounces: str = "single",
    rt_refresh_frames: int = 1,
    rt_diffuse_reflection: bool = True,
    rt_seed: int = 42,
    compute_backend="auto",
    compute_precision="float32",
    adc_compute_precision=None,
    progress: bool = True,
    **removed,
) -> RTBaselineResult:
    """Runs a full-scene RT comparison outside the hybrid PO algorithm."""

    reject_removed_arguments(removed, where="run_full_scene_rt_comparison")
    simulator_kwargs = dict(
        solver="rt",
        compute_backend=compute_backend,
        compute_precision=compute_precision,
        progress=progress,
        rt_human_bounces=rt_human_bounces,
        rt_max_depth=rt_max_depth,
        rt_samples_per_source=rt_samples_per_source,
        rt_max_paths_per_source=rt_max_paths_per_source,
        rt_diffuse_reflection=rt_diffuse_reflection,
        rt_refresh_frames=rt_refresh_frames,
        rt_seed=rt_seed,
    )
    if adc_compute_precision is not None:
        simulator_kwargs["adc_compute_precision"] = adc_compute_precision

    if progress:
        print(
            "[rt-comparison] running full-scene RT "
            f"{num_frames} frames x {radar.fmcw.num_chirps_per_frame} chirps",
            flush=True,
        )
    return run_rt_baseline(
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
    config: HybridPOConfig | None = None,
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
