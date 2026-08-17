# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Profiling helpers for human-only physical optics sweeps."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from mmWaveRadar.dsp import range_fft

from .types import (
    SENSING_METADATA_SCHEMA_VERSION,
    HumanPOMobilityConfig,
    RadarCube,
    SensingMetadata,
)


@dataclass(frozen=True)
class POIncrementalSweepSetting:
    """One incremental-PO profiling point."""

    label: str
    visibility_refresh_chirps: int
    normal_threshold_deg: float
    centroid_displacement_threshold_m: float
    area_relative_threshold: float


@dataclass(frozen=True)
class POIncrementalProfileResult:
    """Accuracy and runtime summary for one incremental-PO run."""

    label: str
    setting: POIncrementalSweepSetting
    cube: RadarCube
    wall_time_s: float
    num_chirps: int
    speedup: float
    adc_relative_error: float
    range_profile_relative_error: float
    path_count_delta_min: int
    path_count_delta_mean: float
    path_count_delta_max: int
    recomputed_mean: float
    recomputed_max: int
    phase_updated_mean: float
    full_refresh_count: int
    visibility_refresh_count: int
    path: Path | None = None


def label_slug(label: str) -> str:
    """Converts a free-form label into a stable filename component."""

    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in label)
    return "_".join(part for part in slug.split("_") if part)


def flatten_adc_for_fft(adc: np.ndarray) -> np.ndarray:
    """Flattens an ADC cube to ``[chirps, samples, channels]``."""

    x = np.asarray(adc)
    if x.ndim == 4:
        return x.reshape((-1, x.shape[-2], x.shape[-1]))
    if x.ndim == 3:
        return x
    raise ValueError("adc must have shape [F, M, N, C] or [M, N, C]")


def relative_norm_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Computes normalized L2 error for real or complex arrays."""

    reference = np.asarray(reference, dtype=np.complex128)
    estimate = np.asarray(estimate, dtype=np.complex128)
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have the same shape")
    return float(
        np.linalg.norm(reference - estimate) / max(np.linalg.norm(reference), 1e-30)
    )


def make_incremental_po_config(
    setting: POIncrementalSweepSetting,
    *,
    visibility_samples_per_face: int,
    adaptive_visibility_edge_margin: float = 0.15,
    visibility_fade_chirps: int = 8,
    visibility_use_phase_center: bool = True,
    full_refresh_chirps: int = 0,
) -> HumanPOMobilityConfig:
    """Builds a simulator config from a sweep setting."""

    return HumanPOMobilityConfig(
        visibility_samples_per_face=int(visibility_samples_per_face),
        visibility_fade_chirps=int(visibility_fade_chirps),
        visibility_use_phase_center=bool(visibility_use_phase_center),
        adaptive_visibility_sampling=True,
        adaptive_visibility_edge_margin=float(adaptive_visibility_edge_margin),
        incremental_update=True,
        incremental_visibility_refresh_chirps=int(
            setting.visibility_refresh_chirps
        ),
        incremental_full_refresh_chirps=int(full_refresh_chirps),
        incremental_normal_threshold_deg=float(setting.normal_threshold_deg),
        incremental_centroid_displacement_threshold_m=float(
            setting.centroid_displacement_threshold_m
        ),
        incremental_area_relative_threshold=float(setting.area_relative_threshold),
    )


def _dict_arrays(mapping: dict[str, float | int] | None):
    """Converts a scalar dictionary into parallel key/value arrays for NPZ."""

    mapping = mapping or {}
    return (
        np.asarray(list(mapping.keys()), dtype=str),
        np.asarray(list(mapping.values())),
    )


def _profile_dict_from_npz(data, prefix: str) -> dict[str, float]:
    """Loads a float profile dictionary from NPZ key/value arrays."""

    key_name = f"{prefix}_keys"
    value_name = f"{prefix}_values"
    if key_name not in data or value_name not in data:
        return {}
    return {
        str(key): float(value)
        for key, value in zip(data[key_name].tolist(), data[value_name].tolist())
    }


def _count_dict_from_npz(data, prefix: str) -> dict[str, int]:
    """Loads an integer count dictionary from NPZ key/value arrays."""

    key_name = f"{prefix}_keys"
    value_name = f"{prefix}_values"
    if key_name not in data or value_name not in data:
        return {}
    return {
        str(key): int(value)
        for key, value in zip(data[key_name].tolist(), data[value_name].tolist())
    }



def _json_default(value):
    """JSON encoder fallback for NumPy scalars and arrays."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _optional_json(data, key: str):
    """Loads an optional JSON scalar field from an NPZ archive."""

    if key not in data:
        return None
    return json.loads(str(data[key].item()))

def _optional_array(data, key: str):
    """Returns an optional array from an NPZ archive."""

    return data[key] if key in data else None


def save_radar_cube_npz(
    path: str | Path,
    cube: RadarCube,
    *,
    wall_time_s: float,
    num_chirps: int,
    label: str = "",
) -> None:
    """Saves a radar cube with enough metadata for profiling notebooks."""

    path = Path(path)
    profile_keys, profile_values = _dict_arrays(cube.metadata.runtime_profile_s)
    count_keys, count_values = _dict_arrays(cube.metadata.runtime_profile_counts)
    payload = {
        "adc": cube.adc,
        "times": cube.times,
        "mode": np.asarray(cube.metadata.mode),
        "metadata_schema_version": np.asarray(
            int(cube.metadata.metadata_schema_version)
        ),
        "virtual_channel_order": np.asarray(cube.metadata.virtual_channel_order),
        "path_counts": cube.metadata.path_counts,
        "wall_time_s": np.asarray(float(wall_time_s)),
        "num_chirps": np.asarray(int(num_chirps)),
        "label": np.asarray(label),
        "runtime_profile_s_keys": profile_keys,
        "runtime_profile_s_values": profile_values.astype(float),
        "runtime_profile_counts_keys": count_keys,
        "runtime_profile_counts_values": count_values.astype(np.int64),
    }
    if cube.metadata.events:
        payload["events_json"] = np.asarray(
            json.dumps(cube.metadata.events, default=_json_default)
        )
    if cube.metadata.po_calibration is not None:
        payload["po_calibration_json"] = np.asarray(
            json.dumps(cube.metadata.po_calibration, default=_json_default)
        )
    for key in [
        "max_displacements",
        "visible_face_counts",
        "effective_visible_area_fraction",
        "runtime_per_chirp_s",
        "max_unwrapped_phase_jump",
        "incremental_recomputed_face_counts",
        "incremental_phase_updated_face_counts",
        "incremental_full_refresh",
        "incremental_visibility_refresh",
        "static_path_counts",
        "blocked_static_path_counts",
        "mean_static_path_visibility",
        "human_env_path_counts",
        "env_human_path_counts",
        "human_touch_path_counts",
        "one_human_touch_path_counts",
        "single_human_only_path_counts",
        "human_env_coupled_path_counts",
        "human_multi_touch_path_counts",
        "path_depth_histogram",
        "total_path_power",
        "human_touch_path_power",
        "one_human_touch_path_power",
        "single_human_only_path_power",
        "human_env_coupled_path_power",
        "human_multi_touch_path_power",
        "human_env_min_path_lengths_m",
        "human_env_mean_path_lengths_m",
        "human_env_max_path_lengths_m",
        "env_human_min_path_lengths_m",
        "env_human_mean_path_lengths_m",
        "env_human_max_path_lengths_m",
        "rt_transition_persistent_path_counts",
        "rt_transition_birth_path_counts",
        "rt_transition_death_path_counts",
        "rt_transition_alpha",
        "rt_transition_old_weights",
        "rt_transition_new_weights",
        "virtual_channel_times_s",
    ]:
        value = getattr(cube.metadata, key, None)
        if value is not None:
            payload[key] = value
    if cube.components:
        component_keys = []
        used_slugs: set[str] = set()
        for key, value in cube.components.items():
            if not isinstance(value, np.ndarray):
                continue
            slug = label_slug(str(key))
            if not slug:
                raise ValueError(
                    f"component name {key!r} has no usable filename characters"
                )
            if slug in used_slugs:
                raise ValueError(
                    "component names must remain unique after filename "
                    f"normalization; duplicate slug {slug!r}"
                )
            used_slugs.add(slug)
            component_keys.append((str(key), slug))
            payload[f"component_{slug}"] = value
        payload["component_keys"] = np.asarray(
            [key for key, _ in component_keys],
            dtype=str,
        )
        payload["component_slugs"] = np.asarray(
            [slug for _, slug in component_keys],
            dtype=str,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_radar_cube_npz(path: str | Path) -> tuple[RadarCube, float, int]:
    """Loads a cube saved by :func:`save_radar_cube_npz`."""

    with np.load(path, allow_pickle=False) as archive:
        data = {
            key: np.array(archive[key], copy=True) for key in archive.files
        }
    metadata = SensingMetadata(
        events=_optional_json(data, "events_json") or [],
        path_counts=data["path_counts"],
        max_displacements=_optional_array(data, "max_displacements"),
        virtual_channel_order=str(data["virtual_channel_order"].item())
        if "virtual_channel_order" in data
        else "tx_major",
        metadata_schema_version=int(data["metadata_schema_version"].item())
        if "metadata_schema_version" in data
        else SENSING_METADATA_SCHEMA_VERSION,
        mode=str(data["mode"].item()) if "mode" in data else "human_only_po",
        visible_face_counts=_optional_array(data, "visible_face_counts"),
        effective_visible_area_fraction=_optional_array(
            data, "effective_visible_area_fraction"
        ),
        runtime_per_chirp_s=_optional_array(data, "runtime_per_chirp_s"),
        max_unwrapped_phase_jump=_optional_array(
            data, "max_unwrapped_phase_jump"
        ),
        runtime_profile_s=_profile_dict_from_npz(data, "runtime_profile_s"),
        runtime_profile_counts=_count_dict_from_npz(
            data, "runtime_profile_counts"
        ),
        incremental_recomputed_face_counts=_optional_array(
            data, "incremental_recomputed_face_counts"
        ),
        incremental_phase_updated_face_counts=_optional_array(
            data, "incremental_phase_updated_face_counts"
        ),
        incremental_full_refresh=_optional_array(
            data, "incremental_full_refresh"
        ),
        incremental_visibility_refresh=_optional_array(
            data, "incremental_visibility_refresh"
        ),
        static_path_counts=_optional_array(data, "static_path_counts"),
        blocked_static_path_counts=_optional_array(
            data, "blocked_static_path_counts"
        ),
        mean_static_path_visibility=_optional_array(
            data, "mean_static_path_visibility"
        ),
        human_env_path_counts=_optional_array(data, "human_env_path_counts"),
        env_human_path_counts=_optional_array(data, "env_human_path_counts"),
        human_touch_path_counts=_optional_array(data, "human_touch_path_counts"),
        one_human_touch_path_counts=_optional_array(
            data, "one_human_touch_path_counts"
        ),
        single_human_only_path_counts=_optional_array(
            data, "single_human_only_path_counts"
        ),
        human_env_coupled_path_counts=_optional_array(
            data, "human_env_coupled_path_counts"
        ),
        human_multi_touch_path_counts=_optional_array(
            data, "human_multi_touch_path_counts"
        ),
        path_depth_histogram=_optional_array(data, "path_depth_histogram"),
        total_path_power=_optional_array(data, "total_path_power"),
        human_touch_path_power=_optional_array(data, "human_touch_path_power"),
        one_human_touch_path_power=_optional_array(
            data, "one_human_touch_path_power"
        ),
        single_human_only_path_power=_optional_array(
            data, "single_human_only_path_power"
        ),
        human_env_coupled_path_power=_optional_array(
            data, "human_env_coupled_path_power"
        ),
        human_multi_touch_path_power=_optional_array(
            data, "human_multi_touch_path_power"
        ),
        human_env_min_path_lengths_m=_optional_array(
            data, "human_env_min_path_lengths_m"
        ),
        human_env_mean_path_lengths_m=_optional_array(
            data, "human_env_mean_path_lengths_m"
        ),
        human_env_max_path_lengths_m=_optional_array(
            data, "human_env_max_path_lengths_m"
        ),
        env_human_min_path_lengths_m=_optional_array(
            data, "env_human_min_path_lengths_m"
        ),
        env_human_mean_path_lengths_m=_optional_array(
            data, "env_human_mean_path_lengths_m"
        ),
        env_human_max_path_lengths_m=_optional_array(
            data, "env_human_max_path_lengths_m"
        ),
        rt_transition_persistent_path_counts=_optional_array(
            data, "rt_transition_persistent_path_counts"
        ),
        rt_transition_birth_path_counts=_optional_array(
            data, "rt_transition_birth_path_counts"
        ),
        rt_transition_death_path_counts=_optional_array(
            data, "rt_transition_death_path_counts"
        ),
        rt_transition_alpha=_optional_array(data, "rt_transition_alpha"),
        rt_transition_old_weights=_optional_array(
            data, "rt_transition_old_weights"
        ),
        rt_transition_new_weights=_optional_array(
            data, "rt_transition_new_weights"
        ),
        virtual_channel_times_s=_optional_array(
            data, "virtual_channel_times_s"
        ),
        po_calibration=_optional_json(data, "po_calibration_json"),
    )
    components = None
    if "component_keys" in data and "component_slugs" in data:
        components = {}
        for key, slug in zip(data["component_keys"], data["component_slugs"]):
            component_key = f"component_{str(slug)}"
            if component_key in data:
                components[str(key)] = data[component_key]
    cube = RadarCube(
        adc=data["adc"],
        times=data["times"],
        metadata=metadata,
        components=components,
    )
    wall_time_s = (
        float(data["wall_time_s"].item())
        if "wall_time_s" in data
        else float("nan")
    )
    if "num_chirps" in data:
        num_chirps = int(data["num_chirps"].item())
    elif cube.adc.ndim == 4:
        num_chirps = int(cube.adc.shape[0] * cube.adc.shape[1])
    elif cube.adc.ndim == 3:
        num_chirps = int(cube.adc.shape[0])
    else:
        raise ValueError(
            "stored adc must have shape [F, M, N, C] or [M, N, C]"
        )
    return cube, wall_time_s, num_chirps


def cube_matches_run(
    cube: RadarCube,
    *,
    num_frames: int,
    fmcw,
    last_chirp_time_s: float | None = None,
    atol_s: float = 1e-9,
) -> bool:
    """Checks whether a cached cube matches a planned frame/chirp setup."""

    frames = int(num_frames)
    chirps = int(fmcw.num_chirps_per_frame)
    samples = int(fmcw.num_adc_samples)
    if cube.adc.ndim == 4:
        if cube.adc.shape[:3] != (frames, chirps, samples):
            return False
        expected_time_shapes = {(frames, chirps)}
    elif cube.adc.ndim == 3:
        if frames != 1 or cube.adc.shape[:2] != (chirps, samples):
            return False
        expected_time_shapes = {(chirps,), (1, chirps)}
    else:
        return False
    if cube.times.shape not in expected_time_shapes:
        return False
    if last_chirp_time_s is None:
        return True
    try:
        tolerance = float(atol_s)
        expected_last_time = float(last_chirp_time_s)
    except (TypeError, ValueError, OverflowError):
        return False
    if (
        not np.isfinite(tolerance)
        or tolerance < 0.0
        or not np.isfinite(expected_last_time)
        or cube.times.size == 0
    ):
        return False
    return bool(
        np.isclose(
            float(np.ravel(cube.times)[-1]),
            expected_last_time,
            atol=tolerance,
        )
    )


def range_profiles_for_cube(cube: RadarCube, fmcw, *, nfft_mult: int = 4):
    """Computes complex range profiles for a cube."""

    return range_fft(
        flatten_adc_for_fft(cube.adc),
        fmcw=fmcw,
        window="hann",
        nfft_mult=nfft_mult,
    )


def summarize_incremental_po_result(
    *,
    label: str,
    setting: POIncrementalSweepSetting,
    cube: RadarCube,
    wall_time_s: float,
    num_chirps: int,
    baseline_cube: RadarCube,
    baseline_wall_time_s: float,
    baseline_range_profiles: np.ndarray,
    fmcw,
    path: str | Path | None = None,
) -> POIncrementalProfileResult:
    """Summarizes speed and fidelity against a full-PO baseline."""

    inc_profiles, _ = range_profiles_for_cube(cube, fmcw)
    path_count_delta = cube.metadata.path_counts - baseline_cube.metadata.path_counts
    recomputed = cube.metadata.incremental_recomputed_face_counts
    phase_updated = cube.metadata.incremental_phase_updated_face_counts
    full_refresh = cube.metadata.incremental_full_refresh
    visibility_refresh = cube.metadata.incremental_visibility_refresh
    return POIncrementalProfileResult(
        label=label,
        setting=setting,
        cube=cube,
        wall_time_s=float(wall_time_s),
        num_chirps=int(num_chirps),
        speedup=float(baseline_wall_time_s) / max(float(wall_time_s), 1e-12),
        adc_relative_error=relative_norm_error(baseline_cube.adc, cube.adc),
        range_profile_relative_error=relative_norm_error(
            baseline_range_profiles, inc_profiles
        ),
        path_count_delta_min=int(np.min(path_count_delta)),
        path_count_delta_mean=float(np.mean(path_count_delta)),
        path_count_delta_max=int(np.max(path_count_delta)),
        recomputed_mean=(
            float(np.mean(recomputed)) if recomputed is not None else float("nan")
        ),
        recomputed_max=int(np.max(recomputed)) if recomputed is not None else 0,
        phase_updated_mean=(
            float(np.mean(phase_updated))
            if phase_updated is not None
            else float("nan")
        ),
        full_refresh_count=(
            int(np.count_nonzero(full_refresh)) if full_refresh is not None else 0
        ),
        visibility_refresh_count=(
            int(np.count_nonzero(visibility_refresh))
            if visibility_refresh is not None
            else 0
        ),
        path=None if path is None else Path(path),
    )


def run_incremental_po_sweep(
    *,
    settings: Iterable[POIncrementalSweepSetting],
    baseline_cube: RadarCube,
    baseline_wall_time_s: float,
    fmcw,
    visibility_samples_per_face: int,
    run_cube: Callable[[str, HumanPOMobilityConfig], tuple[RadarCube, float, int]],
    output_dir: str | Path | None = None,
    reuse_saved: bool = True,
    cache_match: Callable[[RadarCube], bool] | None = None,
    filename_prefix: str = "human_only_po_incremental",
) -> list[POIncrementalProfileResult]:
    """Runs or loads a set of incremental PO configurations."""

    baseline_range_profiles, _ = range_profiles_for_cube(baseline_cube, fmcw)
    output_path = None if output_dir is None else Path(output_dir)
    results: list[POIncrementalProfileResult] = []
    for setting in settings:
        cache_path = None
        if output_path is not None:
            cache_path = output_path / f"{filename_prefix}_{label_slug(setting.label)}_cube.npz"

        if reuse_saved and cache_path is not None and cache_path.exists():
            cube, wall_time_s, num_chirps = load_radar_cube_npz(cache_path)
            if cache_match is not None and not cache_match(cube):
                config = make_incremental_po_config(
                    setting,
                    visibility_samples_per_face=visibility_samples_per_face,
                )
                cube, wall_time_s, num_chirps = run_cube(setting.label, config)
                save_radar_cube_npz(
                    cache_path,
                    cube,
                    wall_time_s=wall_time_s,
                    num_chirps=num_chirps,
                    label=setting.label,
                )
        else:
            config = make_incremental_po_config(
                setting,
                visibility_samples_per_face=visibility_samples_per_face,
            )
            cube, wall_time_s, num_chirps = run_cube(setting.label, config)
            if cache_path is not None:
                save_radar_cube_npz(
                    cache_path,
                    cube,
                    wall_time_s=wall_time_s,
                    num_chirps=num_chirps,
                    label=setting.label,
                )

        results.append(
            summarize_incremental_po_result(
                label=setting.label,
                setting=setting,
                cube=cube,
                wall_time_s=wall_time_s,
                num_chirps=num_chirps,
                baseline_cube=baseline_cube,
                baseline_wall_time_s=baseline_wall_time_s,
                baseline_range_profiles=baseline_range_profiles,
                fmcw=fmcw,
                path=cache_path,
            )
        )
    return results
