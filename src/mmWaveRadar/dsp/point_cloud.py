# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Simple range/angle FFT point-cloud helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np

from .angle import angle_map_fft, direction_cosine_valid_mask
from .cfar import ca_cfar_1d
from .fft import range_fft


def point_cloud_3d_from_adc(
    adc: np.ndarray,
    *,
    virtual_positions_lambda: np.ndarray,
    sampling_frequency: float | None = None,
    slope: float | None = None,
    carrier_frequency: float | None = None,
    fmcw=None,
    range_window: Optional[str] = "hann",
    range_nfft_mult: int = 4,
    range_cfar: Optional[dict[str, Any]] = None,
    range_limits_m: tuple[float, float] | None = None,
    channel_calibration: np.ndarray | None = None,
    chirp_index: int | None = None,
    angle_window: Optional[str] = None,
    angle_fft_size: tuple[int, int] | int | None = (64, 128),
    angle_cfar: Optional[dict[str, Any]] = None,
    half_lambda_scale: int = 2,
    spacing_y_lambda: float | None = None,
    spacing_z_lambda: float | None = None,
) -> Dict[str, Any]:
    """
    Builds a simple local-frame 3D point cloud from ADC samples.

    The pipeline is intentionally small and transparent:

    1. range FFT over fast time,
    2. 1D CA-CFAR over the channel/chirp-combined range profile,
    3. for each detected range, 2D FFT over the virtual array,
    4. 2D CA-CFAR over the range-selected angle map.

    ``virtual_positions_lambda`` must have shape ``[num_channels, 2]`` with
    ``[horizontal, vertical]`` virtual element coordinates in wavelengths.
    The coordinates are gridded onto a regular wavelength-normalized lattice
    before the angle FFT. Per-axis spacings are inferred from regular geometry
    unless ``spacing_y_lambda`` or ``spacing_z_lambda`` is supplied; the
    legacy ``half_lambda_scale`` controls the fallback for singleton or
    irregular axes. ``channel_calibration`` can be used to multiply each
    virtual channel before gridding, for example to remove fixed phase signs.
    ``angle_cfar`` accepts the 2D CFAR arguments plus post-filter keys such as
    ``min_relative_power_db``, ``max_peaks``, and
    ``fallback_to_global_peak``.

    :returns: Dict containing ``points`` in local ``[x, y, z]`` meters,
        per-point ``range_m``, ``azimuth_deg``, ``elevation_deg``, ``power``,
        plus intermediate range/CFAR/angle products for diagnostics.
    """

    fc_hz = _fmcw_value(fmcw, "carrier_frequency", carrier_frequency)
    if fc_hz is None:
        raise ValueError("carrier_frequency or fmcw.carrier_frequency is required")

    x = _as_adc_cube(adc)
    rt, ranges_m = range_fft(
        x,
        sampling_frequency=sampling_frequency,
        slope=slope,
        fmcw=fmcw,
        window=range_window,
        nfft_mult=range_nfft_mult,
    )
    if rt.shape[-1] != np.asarray(virtual_positions_lambda).shape[0]:
        raise ValueError(
            "virtual_positions_lambda must have one row per ADC virtual channel"
        )

    range_power = np.sum(np.abs(rt) ** 2, axis=(0, 2))
    range_detect = _range_cfar(range_power, ranges_m, range_cfar, range_limits_m)

    points = []
    range_indices = []
    angle_indices = []
    angle_products = {}

    for range_index in range_detect["peaks"]:
        snapshot = _range_snapshot(rt, int(range_index), chirp_index)
        if channel_calibration is not None:
            calibration = np.asarray(channel_calibration, dtype=np.complex128)
            if calibration.shape != snapshot.shape:
                raise ValueError(
                    "channel_calibration must have shape [num_virtual_channels]"
                )
            snapshot = snapshot * calibration

        grid, grid_meta = _virtual_snapshot_grid(
            snapshot,
            virtual_positions_lambda,
            half_lambda_scale=half_lambda_scale,
            spacing_y_lambda=spacing_y_lambda,
            spacing_z_lambda=spacing_z_lambda,
        )
        angle = angle_map_fft(
            grid,
            fc_hz=float(fc_hz),
            window=angle_window,
            fft_size=angle_fft_size,
            power=True,
            spacing_y_lambda=grid_meta["spacing_y_lambda"],
            spacing_z_lambda=grid_meta["spacing_z_lambda"],
        )
        if "u" not in angle or "v" not in angle:
            raise ValueError("point_cloud_3d_from_adc requires a 2D virtual array")

        angle_power = np.asarray(angle["map"], dtype=float)
        u_axis = np.asarray(angle["u"], dtype=float)
        v_axis = np.asarray(angle["v"], dtype=float)
        angle_cfar_kwargs, angle_post = _angle_cfar_config(angle_cfar)
        angle_detect = _ca_cfar_2d(angle_power, **angle_cfar_kwargs)
        angle_detect = _postprocess_angle_detections(
            angle_detect,
            angle_power,
            u_axis,
            v_axis,
            **angle_post,
        )

        angle_products[int(range_index)] = {
            "grid": grid,
            "grid_meta": grid_meta,
            "angle": angle,
            "angle_cfar": angle_detect,
        }

        for v_index, u_index in angle_detect["peaks"]:
            u = float(u_axis[int(u_index)])
            v = float(v_axis[int(v_index)])
            xyz = _direction_cosine_point(float(ranges_m[int(range_index)]), u, v)
            if xyz is None:
                continue
            points.append({
                "range_m": float(ranges_m[int(range_index)]),
                "azimuth_deg": _azimuth_deg(u, v),
                "elevation_deg": float(np.degrees(np.arcsin(np.clip(v, -1.0, 1.0)))),
                "u": u,
                "v": v,
                "power": float(angle_power[int(v_index), int(u_index)]),
                "range_bin": int(range_index),
                "angle_bin": (int(v_index), int(u_index)),
                "xyz": xyz,
            })
            range_indices.append(int(range_index))
            angle_indices.append((int(v_index), int(u_index)))

    point_xyz = (
        np.asarray([p["xyz"] for p in points], dtype=float)
        if points else np.zeros((0, 3), dtype=float)
    )

    return {
        "points": point_xyz,
        "detections": points,
        "range_m": np.asarray([p["range_m"] for p in points], dtype=float),
        "azimuth_deg": np.asarray([p["azimuth_deg"] for p in points], dtype=float),
        "elevation_deg": np.asarray(
            [p["elevation_deg"] for p in points],
            dtype=float,
        ),
        "power": np.asarray([p["power"] for p in points], dtype=float),
        "range_bins": np.asarray(range_indices, dtype=np.int64),
        "angle_bins": np.asarray(angle_indices, dtype=np.int64).reshape((-1, 2)),
        "range_fft": rt,
        "ranges_m": ranges_m,
        "range_power": range_power,
        "range_cfar": range_detect,
        "angle_products": angle_products,
    }


def _fmcw_value(fmcw, name: str, fallback):
    """Returns an FMCW attribute or a provided fallback value."""

    return fallback if fmcw is None else getattr(fmcw, name, fallback)


def _as_adc_cube(adc: np.ndarray) -> np.ndarray:
    """Normalizes ADC input to ``[chirps, samples, channels]``."""

    x = np.asarray(adc)
    if x.ndim == 2:
        return x[:, :, None]
    if x.ndim == 3:
        return x
    if x.ndim == 4:
        return x.reshape((-1, x.shape[-2], x.shape[-1]))
    raise ValueError(
        "adc must have shape [chirps, samples], [chirps, samples, channels], "
        "or [frames, chirps, samples, channels]"
    )


def _range_cfar(
    range_power: np.ndarray,
    ranges_m: np.ndarray,
    config: Optional[dict[str, Any]],
    range_limits_m: tuple[float, float] | None,
) -> dict[str, Any]:
    """Runs range CFAR and applies optional metric range limits."""

    kwargs = {
        "num_train": 16,
        "num_guard": 4,
        "pfa": 1e-3,
        "peak_only": True,
        "min_sep": 2,
    }
    if config:
        kwargs.update(config)
    result = ca_cfar_1d(range_power, **kwargs)
    peaks = np.asarray(result["peaks"], dtype=np.int64)
    if range_limits_m is not None:
        lo, hi = (float(v) for v in range_limits_m)
        keep = (ranges_m[peaks] >= lo) & (ranges_m[peaks] <= hi)
        peaks = peaks[keep]
    result = dict(result)
    result["peaks"] = peaks
    return result


def _range_snapshot(
    rt: np.ndarray,
    range_index: int,
    chirp_index: int | None,
) -> np.ndarray:
    """Extracts one channel snapshot at a range bin from one or all chirps."""

    if chirp_index is None:
        return np.mean(rt[:, range_index, :], axis=0)
    return np.asarray(rt[int(chirp_index), range_index, :])


def _virtual_snapshot_grid(
    snapshot: np.ndarray,
    virtual_positions_lambda: np.ndarray,
    *,
    half_lambda_scale: int,
    spacing_y_lambda: float | None,
    spacing_z_lambda: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Places sparse virtual-channel samples onto a rectangular aperture grid."""

    if half_lambda_scale <= 0:
        raise ValueError("half_lambda_scale must be positive")
    values = np.asarray(snapshot, dtype=np.complex128).reshape(-1)
    positions = np.asarray(virtual_positions_lambda, dtype=float)
    if positions.shape != (values.size, 2):
        raise ValueError(
            "virtual_positions_lambda must have shape [num_virtual_channels, 2]"
        )

    origin = np.min(positions, axis=0)
    relative = positions - origin
    fallback_spacing = 1.0 / float(half_lambda_scale)
    spacing_y = _axis_grid_spacing(
        "spacing_y_lambda",
        positions[:, 0],
        spacing_y_lambda,
        fallback_spacing,
    )
    spacing_z = _axis_grid_spacing(
        "spacing_z_lambda",
        positions[:, 1],
        spacing_z_lambda,
        fallback_spacing,
    )
    grid_spacing = np.array([spacing_y, spacing_z], dtype=float)
    grid_index = np.rint(relative / grid_spacing).astype(np.int64)
    reconstructed = origin + grid_index * grid_spacing
    if not np.allclose(reconstructed, positions, atol=1e-8):
        raise ValueError(
            "virtual_positions_lambda coordinates must lie on a common "
            "regular y/z wavelength-normalized grid"
        )

    shifted = grid_index
    nx = int(shifted[:, 0].max()) + 1
    ny = int(shifted[:, 1].max()) + 1
    grid = np.zeros((ny, nx), dtype=np.complex128)
    counts = np.zeros((ny, nx), dtype=float)
    for channel, (ix, iy) in enumerate(shifted):
        grid[int(iy), int(ix)] += values[channel]
        counts[int(iy), int(ix)] += 1.0
    occupied = counts > 0.0
    grid[occupied] /= counts[occupied]
    y_lambda = origin[0] + np.arange(nx) * spacing_y
    z_lambda = origin[1] + np.arange(ny) * spacing_z
    return grid, {
        "x_lambda": y_lambda,
        "y_lambda": y_lambda,
        "z_lambda": z_lambda,
        "spacing_y_lambda": spacing_y,
        "spacing_z_lambda": spacing_z,
        "occupied": occupied,
    }


def _axis_grid_spacing(
    name: str,
    coordinates: np.ndarray,
    configured: float | None,
    fallback: float,
) -> float:
    """Returns a configured or geometry-derived regular-axis spacing."""

    if configured is not None:
        try:
            spacing = float(configured)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be finite and positive") from exc
        if not np.isfinite(spacing) or spacing <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return spacing

    unique = np.unique(np.asarray(coordinates, dtype=float))
    differences = np.diff(unique)
    differences = differences[differences > 1e-10]
    if differences.size == 0:
        return fallback
    candidate = float(np.min(differences))
    if np.allclose(
        differences / candidate,
        np.rint(differences / candidate),
        atol=1e-8,
    ):
        return candidate
    return fallback


def _angle_cfar_config(
    config: Optional[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Splits angle-CFAR options into detector and postprocessing settings."""

    kwargs = {
        "num_train": (4, 4),
        "num_guard": (8, 8),
        "pfa": 1e-3,
        "peak_only": True,
        "min_sep": (2, 2),
    }
    post = {
        "min_relative_power_db": None,
        "max_peaks": None,
        "fallback_to_global_peak": False,
        "include_global_peak": False,
        "valid_direction_cosines_only": True,
    }
    if config:
        remaining = dict(config)
        for key in tuple(post):
            if key in remaining:
                post[key] = remaining.pop(key)
        kwargs.update(remaining)
    return kwargs, post


def _postprocess_angle_detections(
    result: dict[str, Any],
    angle_power: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    *,
    min_relative_power_db: float | None,
    max_peaks: int | None,
    fallback_to_global_peak: bool,
    include_global_peak: bool,
    valid_direction_cosines_only: bool,
) -> dict[str, Any]:
    """Filters, limits, and optionally augments angle-CFAR detections."""

    angle_power = np.asarray(angle_power, dtype=float)
    peaks = np.asarray(result["peaks"], dtype=np.int64).reshape((-1, 2))
    out = dict(result)
    out["raw_peaks"] = peaks.copy()

    if peaks.size and valid_direction_cosines_only:
        peaks = peaks[_valid_angle_peak_mask(peaks, u_axis, v_axis)]

    if peaks.size and min_relative_power_db is not None:
        max_power = float(np.max(angle_power))
        cutoff = max_power * 10.0 ** (float(min_relative_power_db) / 10.0)
        keep = angle_power[peaks[:, 0], peaks[:, 1]] >= cutoff
        peaks = peaks[keep]

    fallback_added = False
    global_peak = None
    if include_global_peak or (fallback_to_global_peak and peaks.shape[0] == 0):
        global_peak = _global_angle_peak(
            angle_power,
            u_axis,
            v_axis,
            valid_direction_cosines_only=valid_direction_cosines_only,
        )
    if global_peak is not None:
        global_peak_array = np.asarray(global_peak, dtype=np.int64).reshape((1, 2))
        already_present = (
            peaks.shape[0] > 0
            and np.any(np.all(peaks == global_peak_array, axis=1))
        )
        if not already_present:
            peaks = np.vstack([peaks, global_peak_array])
            fallback_added = fallback_to_global_peak

    if max_peaks is not None:
        limit = int(max_peaks)
        if limit < 1:
            raise ValueError("max_peaks must be >= 1")
        if peaks.shape[0] > limit:
            order = np.argsort(angle_power[peaks[:, 0], peaks[:, 1]])[::-1]
            peaks = peaks[order[:limit]]

    out["peaks"] = peaks.astype(np.int64, copy=False)
    out["fallback_peak_added"] = bool(fallback_added)
    return out


def _valid_angle_peak_mask(
    peaks: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
) -> np.ndarray:
    """Marks angle peak indices whose direction cosines are physically valid."""

    if peaks.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    u = np.asarray(u_axis, dtype=float)[peaks[:, 1]]
    v = np.asarray(v_axis, dtype=float)[peaks[:, 0]]
    return (u * u + v * v) <= 1.0 + 1e-12


def _global_angle_peak(
    angle_power: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    *,
    valid_direction_cosines_only: bool,
) -> tuple[int, int] | None:
    """Returns the strongest angle-map bin, optionally within the valid disk."""

    power = np.asarray(angle_power, dtype=float)
    if power.size == 0:
        return None
    if valid_direction_cosines_only:
        valid = direction_cosine_valid_mask(u_axis, v_axis)
        if not np.any(valid):
            return None
        masked = np.where(valid, power, -np.inf)
        flat_index = int(np.argmax(masked))
    else:
        flat_index = int(np.argmax(power))
    if not np.isfinite(power.reshape(-1)[flat_index]):
        return None
    return tuple(int(v) for v in np.unravel_index(flat_index, power.shape))


def _as_pair(value: int | Iterable[int], name: str) -> tuple[int, int]:
    """Normalizes a scalar or two-item iterable into a nonnegative integer pair."""

    if np.isscalar(value):
        out = (int(value), int(value))
    else:
        seq = tuple(int(v) for v in value)
        if len(seq) != 2:
            raise ValueError(f"{name} must be an int or a pair")
        out = (seq[0], seq[1])
    if out[0] < 0 or out[1] < 0:
        raise ValueError(f"{name} must be nonnegative")
    return out


def _ca_cfar_2d(
    x: np.ndarray,
    *,
    num_train: int | tuple[int, int] = (4, 4),
    num_guard: int | tuple[int, int] = (1, 1),
    pfa: float = 1e-3,
    scale: float | None = None,
    peak_only: bool = True,
    min_sep: int | tuple[int, int] = (2, 2),
) -> Dict[str, Any]:
    """Runs rectangular 2D cell-averaging CFAR on a nonnegative map."""

    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be 2D")
    if np.any(x < 0):
        raise ValueError("x must be nonnegative")
    if not 0.0 < pfa < 1.0:
        raise ValueError("pfa must be between 0 and 1")

    train_y, train_x = _as_pair(num_train, "num_train")
    guard_y, guard_x = _as_pair(num_guard, "num_guard")
    sep_y, sep_x = _as_pair(min_sep, "min_sep")
    if train_y < 1 or train_x < 1:
        raise ValueError("num_train must be >= 1")

    ny, nx = x.shape
    pad_y = train_y + guard_y
    pad_x = train_x + guard_x
    train_count = (
        (2 * pad_y + 1) * (2 * pad_x + 1)
        - (2 * guard_y + 1) * (2 * guard_x + 1)
    )
    alpha = (
        train_count * (pfa ** (-1.0 / train_count) - 1.0)
        if scale is None else float(scale)
    )

    det = np.zeros_like(x, dtype=bool)
    thr = np.full_like(x, np.nan, dtype=float)
    noise = np.full_like(x, np.nan, dtype=float)
    for iy in range(pad_y, ny - pad_y):
        for ix in range(pad_x, nx - pad_x):
            window = x[iy - pad_y: iy + pad_y + 1,
                       ix - pad_x: ix + pad_x + 1]
            guard = x[iy - guard_y: iy + guard_y + 1,
                      ix - guard_x: ix + guard_x + 1]
            z = (float(np.sum(window)) - float(np.sum(guard))) / train_count
            noise[iy, ix] = z
            thr[iy, ix] = alpha * z
            det[iy, ix] = x[iy, ix] > thr[iy, ix]

    peaks = np.argwhere(det)
    if peak_only and peaks.size:
        peaks = _local_2d_peaks(x, peaks)
    if peaks.shape[0] > 1 and (sep_y > 0 or sep_x > 0):
        peaks = _suppress_2d_peaks(x, peaks, sep_y=sep_y, sep_x=sep_x)

    return {
        "det_mask": det,
        "threshold": thr,
        "noise_est": noise,
        "peaks": peaks.astype(np.int64, copy=False),
    }


def _local_2d_peaks(x: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Keeps candidate 2D peaks that are local maxima in a 3x3 neighborhood."""

    chosen = []
    ny, nx = x.shape
    for iy, ix in peaks:
        lo_y = max(0, int(iy) - 1)
        hi_y = min(ny, int(iy) + 2)
        lo_x = max(0, int(ix) - 1)
        hi_x = min(nx, int(ix) + 2)
        if x[int(iy), int(ix)] >= np.max(x[lo_y:hi_y, lo_x:hi_x]):
            chosen.append((int(iy), int(ix)))
    return np.asarray(chosen, dtype=np.int64).reshape((-1, 2))


def _suppress_2d_peaks(
    x: np.ndarray,
    peaks: np.ndarray,
    *,
    sep_y: int,
    sep_x: int,
) -> np.ndarray:
    """Performs greedy non-maximum suppression on 2D peak candidates."""

    order = np.argsort(x[peaks[:, 0], peaks[:, 1]])[::-1]
    taken = np.zeros_like(x, dtype=bool)
    chosen = []
    for k in order:
        iy, ix = (int(v) for v in peaks[int(k)])
        if taken[iy, ix]:
            continue
        chosen.append((iy, ix))
        lo_y = max(0, iy - sep_y)
        hi_y = min(x.shape[0], iy + sep_y + 1)
        lo_x = max(0, ix - sep_x)
        hi_x = min(x.shape[1], ix + sep_x + 1)
        taken[lo_y:hi_y, lo_x:hi_x] = True
    return np.asarray(sorted(chosen), dtype=np.int64).reshape((-1, 2))


def _direction_cosine_point(range_m: float, u: float, v: float) -> np.ndarray | None:
    """Converts range and direction cosines into an ``[x, y, z]`` point."""

    radial_x_sq = 1.0 - u * u - v * v
    if radial_x_sq < 0.0:
        return None
    radial_x = float(np.sqrt(max(radial_x_sq, 0.0)))
    return float(range_m) * np.array([radial_x, u, v], dtype=float)


def _azimuth_deg(u: float, v: float) -> float:
    """Computes azimuth angle from horizontal and vertical direction cosines."""

    radial_x = np.sqrt(max(1.0 - u * u - v * v, 0.0))
    return float(np.degrees(np.arctan2(u, radial_x)))


__all__ = ["point_cloud_3d_from_adc"]
