# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Profile extraction and range-bin selection helpers for radar DSP cubes."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np


def range_profile_from_cube(
    range_cube: np.ndarray,
    *,
    combine_antennas: Literal[
        "sum_power", "mean_power", "sum_mag", "mean_mag"
    ] = "sum_power",
    combine_chirps: Optional[Literal["mean", "sum"]] = None,
) -> np.ndarray:
    """
    Builds range profiles from a range-FFT cube.

    :param range_cube: Complex range FFT cube with shape
        ``[num_chirps, num_range_bins, num_channels]``.
    :param combine_antennas: Channel-combination method.
    :param combine_chirps: Optional slow-time reduction.
    """
    y = np.asarray(range_cube)
    if y.ndim != 3:
        raise ValueError("range_cube must have shape [M, R, P]")

    if combine_antennas in ("sum_power", "mean_power"):
        rp = np.sum(np.abs(y) ** 2, axis=-1)
        if combine_antennas == "mean_power":
            rp = rp / y.shape[-1]
    elif combine_antennas in ("sum_mag", "mean_mag"):
        rp = np.sum(np.abs(y), axis=-1)
        if combine_antennas == "mean_mag":
            rp = rp / y.shape[-1]
    else:
        raise ValueError(f"Unknown combine_antennas: {combine_antennas}")

    if combine_chirps is None:
        return rp
    if combine_chirps == "mean":
        return np.mean(rp, axis=0)
    if combine_chirps == "sum":
        return np.sum(rp, axis=0)
    raise ValueError(f"Unknown combine_chirps: {combine_chirps}")


def select_range_bin(
    profile: np.ndarray,
    ranges_m: np.ndarray,
    target_range_m: float,
    *,
    mode: Literal["target", "dominant_near_target"] = "target",
    search_half_width_m: float = 0.0,
    power_floor: float = 0.0,
) -> tuple[int, dict[str, object]]:
    """
    Selects a range bin from a 1D range profile.

    ``mode="target"`` selects the bin nearest ``target_range_m``.
    ``mode="dominant_near_target"`` selects the strongest bin within
    ``search_half_width_m`` of the target, falling back to the target bin when
    no local bin exceeds ``power_floor``.
    """

    ranges = np.asarray(ranges_m, dtype=float)
    values = np.asarray(profile, dtype=float)
    if ranges.ndim != 1:
        raise ValueError("ranges_m must be a 1D array")
    if values.ndim != 1:
        raise ValueError("profile must be a 1D array")
    if ranges.size == 0:
        raise ValueError("ranges_m and profile must be nonempty")
    if ranges.shape != values.shape:
        raise ValueError("ranges_m and profile must have the same shape")
    if not np.all(np.isfinite(ranges)):
        raise ValueError("ranges_m must contain only finite values")
    if not np.all(np.isfinite(values)):
        raise ValueError("profile must contain only finite values")

    target_range = float(target_range_m)
    half_width = float(search_half_width_m)
    floor = float(power_floor)
    if not np.isfinite(target_range):
        raise ValueError("target_range_m must be finite")
    if not np.isfinite(half_width) or half_width < 0.0:
        raise ValueError("search_half_width_m must be a nonnegative finite value")
    if not np.isfinite(floor):
        raise ValueError("power_floor must be finite")
    if mode not in ("target", "dominant_near_target"):
        raise ValueError("mode must be 'target' or 'dominant_near_target'")

    target_index = int(np.argmin(np.abs(ranges - target_range)))
    dominant_index = target_index
    search_mask = np.abs(ranges - target_range) <= half_width
    if np.any(search_mask):
        search_indices = np.flatnonzero(search_mask)
        local_values = values[search_mask]
        if float(np.max(local_values)) > floor:
            dominant_index = int(search_indices[int(np.argmax(local_values))])

    selected_index = target_index
    if mode == "dominant_near_target":
        selected_index = dominant_index

    return selected_index, {
        "target_range_index": target_index,
        "target_range_m_actual": float(ranges[target_index]),
        "target_range_power": float(values[target_index]),
        "dominant_range_index": dominant_index,
        "dominant_range_m": float(ranges[dominant_index]),
        "dominant_range_power": float(values[dominant_index]),
        "selected_range_mode": mode,
        "range_profile_is_zero": bool(float(np.max(values)) <= floor),
    }
