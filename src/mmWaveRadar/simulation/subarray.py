# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Subarray partition: one phase centre per Tx-group and Rx-group pair.

``array_model="subarray"`` splits the Tx and Rx arrays into groups. Each
pair of a Tx group and an Rx group is solved as its own radar, traced or
integrated once at the centres of its two groups, and its virtual channels
are written back into the full array's channel order. Calibrated PO parts
share one gain across the array: first-pose calibration is refitted from every
pair's reference powers, and sequence calibration uses the first pair's fit.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import warnings

import numpy as np

from ..naming import array_group_partition
from ..radar import RadarSensor
from .calibration import calibration_gain_array
from .types import RadarCube, SensingMetadata


@dataclass(frozen=True)
class GroupPair:
    """One Tx group and one Rx group, and the full-array channels they own."""

    tx_group: tuple[int, ...]
    rx_group: tuple[int, ...]
    channels: np.ndarray


def group_pairs(
    radar: RadarSensor,
    tx_groups: tuple[tuple[int, ...], ...] | None,
    rx_groups: tuple[tuple[int, ...], ...] | None,
) -> tuple[GroupPair, ...]:
    """Pairs every Tx group with every Rx group that shares a virtual channel."""

    hardware = radar.hardware
    tx_groups = array_group_partition(tx_groups, hardware.num_tx, name="array_tx_groups")
    rx_groups = array_group_partition(rx_groups, hardware.num_rx, name="array_rx_groups")
    channel_tx = np.asarray(hardware.virtual_tx_indices())
    channel_rx = np.asarray(hardware.virtual_rx_indices())
    pairs = []
    for tx_group in tx_groups:
        for rx_group in rx_groups:
            channels = np.flatnonzero(
                np.isin(channel_tx, tx_group) & np.isin(channel_rx, rx_group)
            )
            if channels.size:
                pairs.append(GroupPair(tx_group, rx_group, channels))
    return tuple(pairs)


def local_array_groups(groups, active: list[int], *, name: str):
    """Maps groups of board antenna indices to the radar's local indices."""

    if groups is None:
        return None
    listed = {index for group in groups for index in group}
    missing = [index for index in active if index not in listed]
    inactive = sorted(listed - set(active))
    if missing or inactive:
        problems = []
        if missing:
            problems.append(f"leave out {missing}")
        if inactive:
            problems.append(f"list {inactive}, which are not active")
        raise ValueError(
            f"{name} must split the active antennas {active} into groups; "
            "they " + " and ".join(problems)
        )
    local = {board: position for position, board in enumerate(active)}
    return tuple(tuple(local[index] for index in group) for group in groups)


def subarray_radar(radar: RadarSensor, pair: GroupPair) -> RadarSensor:
    """Returns the radar restricted to one group pair.

    Hardware subsets keep the full array's virtual-channel order, so the
    sub-radar's channel ``j`` is the full array's channel ``pair.channels[j]``.
    """

    hardware = radar.hardware.subset_tx(pair.tx_group).subset_rx(pair.rx_group)
    return replace(
        radar,
        hardware=hardware,
        fmcw=replace(radar.fmcw, num_tx=hardware.num_tx),
    )


def pair_progress(callback, index: int, count: int):
    """Reports a group pair's progress as part of the whole partitioned run."""

    def report(event: dict) -> None:
        event = dict(event)
        if "frame_index" in event and "num_frames" in event:
            num_frames = int(event["num_frames"])
            event["frame_index"] = index * num_frames + int(event["frame_index"])
            event["num_frames"] = count * num_frames
        event["subarray_index"] = int(index)
        event["num_subarrays"] = int(count)
        callback(event)

    return report


def run_group_pairs(
    radar,
    pairs,
    execute,
    *,
    scenes=(),
    callback=None,
    check_cancelled=None,
    progress=False,
):
    """Executes a partition and restores full-array devices on every exit."""

    try:
        results = []
        for index, pair in enumerate(pairs):
            if check_cancelled is not None:
                check_cancelled()
            if progress:
                print(
                    f"[subarray {index + 1}/{len(pairs)}] "
                    f"Tx {list(pair.tx_group)} x Rx {list(pair.rx_group)}",
                    flush=True,
                )
            report = None if callback is None else pair_progress(callback, index, len(pairs))
            results.append(execute(subarray_radar(radar, pair), report))
        return results
    finally:
        seen = set()
        for scene in scenes:
            if scene is not None and id(scene) not in seen:
                seen.add(id(scene))
                if scene.get(f"{radar.name}-tx") is not None:
                    radar.configure_scene(scene)


def partition_event(pairs):
    """Describes the partition consistently for all solver entry points."""

    return {
        "reason": "subarray_partition",
        "array_tx_groups": [list(group) for group in dict.fromkeys(p.tx_group for p in pairs)],
        "array_rx_groups": [list(group) for group in dict.fromkeys(p.rx_group for p in pairs)],
        "group_pairs": len(pairs),
    }


def max_group_offsets(radar: RadarSensor, tx_groups, rx_groups) -> tuple[float, float]:
    """Largest element distance from its group centre, for Tx and for Rx [m]."""

    def offset(positions, groups, count, name):
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        largest = 0.0
        for group in array_group_partition(groups, count, name=name):
            members = positions[list(group)]
            largest = max(
                largest, float(np.max(np.linalg.norm(members - members.mean(axis=0), axis=1)))
            )
        return largest

    return (
        offset(radar.world_tx_positions(), tx_groups, radar.hardware.num_tx, "array_tx_groups"),
        offset(radar.world_rx_positions(), rx_groups, radar.hardware.num_rx, "array_rx_groups"),
    )


# Every array field of SensingMetadata, by how pair values combine. A unit
# test checks that new fields are added here.
_SUM_FIELDS = frozenset(
    {
        "path_counts",
        "static_path_counts",
        "blocked_static_path_counts",
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
        "incremental_recomputed_face_counts",
        "incremental_phase_updated_face_counts",
        "runtime_per_chirp_s",
        "rt_transition_persistent_path_counts",
        "rt_transition_birth_path_counts",
        "rt_transition_death_path_counts",
        "rt_blocked_path_counts",
    }
)
_MAX_FIELDS = frozenset(
    {
        "max_displacements",
        "max_unwrapped_phase_jump",
        "human_env_max_path_lengths_m",
        "env_human_max_path_lengths_m",
    }
)
_MIN_FIELDS = frozenset({"human_env_min_path_lengths_m", "env_human_min_path_lengths_m"})
_MEAN_FIELDS = frozenset(
    {
        "visible_face_counts",
        "effective_visible_area_fraction",
        "mean_static_path_visibility",
        "human_env_mean_path_lengths_m",
        "env_human_mean_path_lengths_m",
        "rt_transition_alpha",
        "rt_transition_old_weights",
        "rt_transition_new_weights",
    }
)
_ANY_FIELDS = frozenset(
    {
        "incremental_full_refresh",
        "incremental_visibility_refresh",
        "rt_visibility_refresh",
    }
)
_CHANNEL_FIELDS = frozenset({"virtual_channel_times_s"})
MERGED_ARRAY_FIELDS = (
    _SUM_FIELDS | _MAX_FIELDS | _MIN_FIELDS | _MEAN_FIELDS | _ANY_FIELDS | _CHANNEL_FIELDS
)

# Hybrid PO components scaled by the PO calibration gain; static RT is not.
_CALIBRATED_HYBRID_COMPONENTS = ("human_po", "human_env", "env_human")


def _joint_first_pose_calibration(calibrations: list[dict], weights) -> dict | None:
    """Refits a first-pose gain from every pair's reference powers.

    Each pair records mean RT and PO reference powers over its channels, so
    the whole array's means are the channel-weighted averages.
    """

    keys = ("rt_reference_rms_power", "raw_po_reference_rms_power")
    if any(
        item.get("mode") != "rt_first_pose" or any(key not in item for key in keys)
        for item in calibrations
    ):
        return None
    weights = np.asarray(weights, dtype=np.float64)
    rt_power = float(np.dot(weights, [item[keys[0]] for item in calibrations]) / weights.sum())
    po_power = float(np.dot(weights, [item[keys[1]] for item in calibrations]) / weights.sum())
    floor = float(calibrations[0].get("reference_power_floor", 0.0))
    if not (rt_power > floor and po_power > floor):
        return None
    gain = float(np.sqrt(rt_power / po_power))
    return {
        **calibrations[0],
        "human_po_amplitude_gain": gain,
        "human_po_amplitude_gain_db": float(20.0 * np.log10(gain)),
        "rt_reference_rms_power": rt_power,
        "raw_po_reference_rms_power": po_power,
        "calibrated_po_reference_rms_power": float(gain * gain * po_power),
        "fitted_on_subarrays": "all",
    }


def share_calibration(
    cubes: list[RadarCube],
    solver: str,
    weights=None,
) -> tuple[list[RadarCube], dict | None]:
    """Rescales each pair's calibrated PO parts to one gain for the array.

    Each pair fits its own PO gain when calibration is on; keeping those would
    put amplitude steps across the array. First-pose gains are refitted from
    all pairs' reference powers (weighted by channel count); sequence gains
    use the first pair's fit. The PO solver's ADC is the gain times the raw PO
    response, and Hybrid PO adds the gain-scaled human and coupling
    components to the static room, so rescaling is exact. Returns the cubes
    and the shared calibration metadata (``None`` when calibration is off).
    """

    if solver not in ("po", "hybrid_po"):
        return cubes, None
    calibrations = [cube.metadata.po_calibration or {} for cube in cubes]
    if calibrations[0].get("mode", "none") == "none":
        return cubes, None
    weights = np.ones(len(cubes)) if weights is None else weights
    shared = _joint_first_pose_calibration(calibrations, weights)
    if shared is None:
        shared = {**calibrations[0], "fitted_on_subarrays": [0]}
    shared["subarray_fitted_gains"] = [
        float(item.get("human_po_amplitude_gain", 1.0)) for item in calibrations
    ]
    shape = np.asarray(cubes[0].times).shape
    reference = calibration_gain_array(
        shared,
        shape,
        fallback=float(shared.get("human_po_amplitude_gain", 1.0)),
    )
    rescaled = []
    for cube, own_calibration in zip(cubes, calibrations):
        own = calibration_gain_array(
            own_calibration,
            shape,
            fallback=float(own_calibration.get("human_po_amplitude_gain", 1.0)),
        )
        ratio = np.divide(reference, own, out=np.zeros_like(reference), where=own != 0.0)
        if np.allclose(ratio, 1.0):
            rescaled.append(cube)
            continue
        scale = ratio[..., None, None]
        adc = np.asarray(cube.adc)
        components = dict(cube.components or {})
        if solver == "po":
            adc = adc * scale
        else:
            calibrated = [name for name in _CALIBRATED_HYBRID_COMPONENTS if name in components]
            delta = sum(np.asarray(components[name]) for name in calibrated)
            adc = adc + (scale - 1.0) * delta
            for name in calibrated:
                components[name] = (np.asarray(components[name]) * scale).astype(
                    np.asarray(components[name]).dtype,
                    copy=False,
                )
        rescaled.append(
            replace(
                cube,
                adc=adc.astype(np.asarray(cube.adc).dtype, copy=False),
                components=components or cube.components,
            )
        )
    return rescaled, shared


def merge_subarray_cubes(
    cubes: list[RadarCube],
    pairs: tuple[GroupPair, ...],
    num_channels: int,
    *,
    solver: str,
) -> RadarCube:
    """Writes each pair's channels into one full-array cube."""

    if len(cubes) != len(pairs) or not cubes:
        raise ValueError("need one cube per group pair")
    cubes, calibration = share_calibration(
        list(cubes),
        solver,
        weights=[pair.channels.size for pair in pairs],
    )
    times = np.asarray(cubes[0].times)
    for cube in cubes[1:]:
        if np.asarray(cube.times).shape != times.shape:
            raise ValueError("group-pair runs returned different slow-time axes")

    adc = _scatter([np.asarray(cube.adc) for cube in cubes], pairs, num_channels)
    return RadarCube(
        adc=adc,
        times=times,
        metadata=_merge_metadata(
            [cube.metadata for cube in cubes],
            pairs,
            num_channels,
            calibration,
        ),
        components=_merge_components(cubes, pairs, num_channels),
    )


def _scatter(arrays: list[np.ndarray], pairs, num_channels: int) -> np.ndarray:
    """Places each pair's last-axis channels at their full-array indices."""

    first = arrays[0]
    out = np.zeros(first.shape[:-1] + (int(num_channels),), dtype=first.dtype)
    for array, pair in zip(arrays, pairs):
        out[..., pair.channels] = array
    return out


def _is_channel_array(array: np.ndarray, adc: np.ndarray, pair: GroupPair) -> bool:
    """Whether an array is ADC-like: ADC shape up to its frame count."""

    return (
        array.ndim == adc.ndim
        and array.shape[1:-1] == adc.shape[1:-1]
        and array.shape[-1] == pair.channels.size
    )


def _merge_components(cubes, pairs, num_channels) -> dict[str, np.ndarray] | None:
    """Scatters ADC-like components; keeps other components per pair."""

    if all(cube.components is None for cube in cubes):
        return None
    names = [name for name in (cubes[0].components or {})]
    for cube in cubes[1:]:
        names += [name for name in (cube.components or {}) if name not in names]
    merged: dict[str, np.ndarray] = {}
    for name in names:
        arrays = [None if cube.components is None else cube.components.get(name) for cube in cubes]
        channel_like = all(
            array is not None and _is_channel_array(np.asarray(array), np.asarray(cube.adc), pair)
            for array, cube, pair in zip(arrays, cubes, pairs)
        )
        if channel_like and len({np.asarray(a).shape[:-1] for a in arrays}) == 1:
            merged[name] = _scatter([np.asarray(a) for a in arrays], pairs, num_channels)
            continue
        # Per-path or per-pair data has no full-array layout; keep each pair's.
        for index, array in enumerate(arrays):
            if array is not None:
                merged[f"{name}_subarray{index}"] = np.asarray(array)
    return merged


def _merge_metadata(
    metas: list[SensingMetadata],
    pairs,
    num_channels,
    calibration: dict | None = None,
) -> SensingMetadata:
    """Combines per-pair metadata into one record for the full array."""

    updates: dict[str, object] = {}
    for item in fields(SensingMetadata):
        name = item.name
        values = [getattr(meta, name) for meta in metas]
        first = values[0]
        if name == "events":
            updates[name] = [
                {**event, "subarray": index}
                for index, meta in enumerate(metas)
                for event in (meta.events or [])
            ]
        elif name in ("runtime_profile_s", "runtime_profile_counts"):
            if all(value is None for value in values):
                continue
            total: dict = {}
            for value in values:
                for key, amount in (value or {}).items():
                    total[key] = total.get(key, 0) + amount
            updates[name] = total
        elif name == "po_calibration" and calibration is not None:
            updates[name] = calibration
        elif isinstance(first, np.ndarray) and all(
            isinstance(value, np.ndarray) for value in values
        ):
            updates[name] = _merge_array(name, values, pairs, num_channels)
    return replace(metas[0], **updates)


def _merge_array(name: str, values: list[np.ndarray], pairs, num_channels) -> np.ndarray:
    """Combines one metadata array across pairs by its field rule."""

    if name in _CHANNEL_FIELDS:
        return _scatter(values, pairs, num_channels)
    if len({value.shape for value in values}) != 1:
        return values[0]
    stack = np.stack(values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if name in _SUM_FIELDS:
            return np.sum(stack, axis=0).astype(values[0].dtype, copy=False)
        if name in _ANY_FIELDS:
            return np.any(stack, axis=0)
        if name in _MAX_FIELDS:
            return np.nanmax(stack, axis=0)
        if name in _MIN_FIELDS:
            return np.nanmin(stack, axis=0)
        if name in _MEAN_FIELDS:
            mean = np.nanmean(stack.astype(np.float64), axis=0)
            if np.issubdtype(values[0].dtype, np.integer):
                return np.rint(mean).astype(values[0].dtype)
            return mean.astype(values[0].dtype, copy=False)
    return values[0]


__all__ = [
    "GroupPair",
    "MERGED_ARRAY_FIELDS",
    "group_pairs",
    "local_array_groups",
    "max_group_offsets",
    "merge_subarray_cubes",
    "pair_progress",
    "partition_event",
    "run_group_pairs",
    "share_calibration",
    "subarray_radar",
]
