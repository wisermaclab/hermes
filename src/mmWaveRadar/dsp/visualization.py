# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Plotting helpers for mmWave radar DSP outputs."""

from __future__ import annotations

from typing import Optional

import numpy as np


def plot_axis_image(
    image: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    *,
    ax=None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    cmap=None,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = False,
    colorbar_label: str | None = None,
    origin: str = "lower",
    aspect: str = "auto",
):
    """Plots a 2D image whose columns/rows are described by explicit axes."""

    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    z = np.asarray(image)
    x = np.asarray(x_axis, dtype=float).reshape(-1)
    y = np.asarray(y_axis, dtype=float).reshape(-1)
    if z.ndim != 2:
        raise ValueError("image must have shape [num_y, num_x]")
    if x.shape not in ((z.shape[1],), (z.shape[1] + 1,)):
        raise ValueError(
            "x_axis length must match image columns or image column edges"
        )
    if y.shape not in ((z.shape[0],), (z.shape[0] + 1,)):
        raise ValueError(
            "y_axis length must match image rows or image row edges"
        )
    if x.size == 0 or y.size == 0:
        raise ValueError("x_axis and y_axis must be non-empty")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    im = ax.imshow(
        z,
        origin=origin,
        aspect=aspect,
        extent=[
            *_axis_extent_bounds(x),
            *_axis_extent_bounds(y),
        ],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(im, ax=ax, label=colorbar_label)
    return im


def plot_range_profile(
    ranges_m: np.ndarray,
    values: np.ndarray,
    *,
    ax=None,
    label: str | None = None,
    range_limits_m: tuple[float, float] | None = None,
    target_range_m: float | None = None,
    xlabel: str = "range [m]",
    ylabel: str | None = None,
    title: str | None = None,
    grid: bool = True,
    **plot_kwargs,
):
    """Plots a range profile with optional range limits and target marker."""

    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    ranges = np.asarray(ranges_m, dtype=float).reshape(-1)
    y = np.asarray(values).reshape(-1)
    if ranges.shape != y.shape:
        raise ValueError("ranges_m and values must have the same shape")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))

    mask = np.ones(ranges.shape, dtype=bool)
    if range_limits_m is not None:
        lo, hi = (float(v) for v in range_limits_m)
        mask = (ranges >= lo) & (ranges <= hi)
        ax.set_xlim(lo, hi)

    line, = ax.plot(ranges[mask], y[mask], label=label, **plot_kwargs)
    if target_range_m is not None:
        ax.axvline(
            float(target_range_m),
            color="cyan",
            linestyle="--",
            linewidth=1.2,
            label="expected",
        )
    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if grid:
        ax.grid(True)
    return line


def plot_point_cloud_projection(
    points: np.ndarray,
    *,
    x_axis: str = "y",
    y_axis: str = "z",
    power: np.ndarray | None = None,
    ax=None,
    cmap: str = "viridis",
    empty_label: str = "no detections",
    colorbar: bool = False,
    colorbar_label: str = "power",
):
    """Plots a local-frame point-cloud projection."""

    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        pts = np.zeros((0, 3), dtype=float)
    else:
        pts = pts.reshape((-1, 3))
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    x_index = _point_axis_index(x_axis)
    y_index = _point_axis_index(y_axis)
    if pts.shape[0] == 0:
        ax.text(0.5, 0.5, empty_label, transform=ax.transAxes,
                ha="center", va="center")
        scatter = None
    elif power is None:
        scatter = ax.scatter(pts[:, x_index], pts[:, y_index])
    else:
        colors = np.asarray(power, dtype=float).reshape(-1)
        if colors.shape != (pts.shape[0],):
            raise ValueError("power length must match number of points")
        scatter = ax.scatter(
            pts[:, x_index],
            pts[:, y_index],
            c=colors,
            cmap=cmap,
        )
        if colorbar:
            ax.figure.colorbar(scatter, ax=ax, label=colorbar_label)

    ax.set_xlabel(f"{x_axis} [m]")
    ax.set_ylabel(f"{y_axis} [m]")
    ax.set_aspect("equal", adjustable="box")
    return scatter


def plot_range_time_map(
    range_time: np.ndarray,
    ranges_m: np.ndarray,
    times_s: np.ndarray | None = None,
    *,
    use_db: bool = True,
    title: str = "Range-Time Map",
    ax=None,
):
    """
    Plots a range-time map or a single range profile.

    The function imports Matplotlib lazily so `mmWaveRadar.dsp` remains usable in
    non-plotting environments.
    """
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    rt = np.asarray(range_time)
    ranges_m = np.asarray(ranges_m)
    if rt.ndim != 2:
        raise ValueError("range_time must have shape [num_times, num_ranges]")
    if ranges_m.shape != (rt.shape[1],):
        raise ValueError("ranges_m length must match range_time")

    z = np.asarray(rt, dtype=float)
    if use_db:
        z = 10.0 * np.log10(np.maximum(z, 1e-24))

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))

    if rt.shape[0] == 1:
        ax.plot(ranges_m, z[0])
        ax.set_ylabel("Power (dB)" if use_db else "Power")
        ax.grid(True)
    else:
        if times_s is None:
            times_s = np.arange(rt.shape[0], dtype=float)
        times_s = np.asarray(times_s, dtype=float).reshape(-1)
        if times_s.shape != (rt.shape[0],):
            raise ValueError("times_s length must match range_time")
        im = ax.imshow(z, aspect="auto", origin="lower",
                       extent=[float(ranges_m[0]), float(ranges_m[-1]),
                               float(times_s[0]), float(times_s[-1])])
        ax.set_ylabel("Time (s)")
        plt.colorbar(im, ax=ax, label="Power (dB)" if use_db else "Power")

    ax.set_xlabel("Range (m)")
    ax.set_title(title)
    return ax


def plot_angle_map(
    angle_map: np.ndarray,
    *,
    u: Optional[np.ndarray] = None,
    v: Optional[np.ndarray] = None,
    db: bool = True,
    db_floor: float = -40.0,
    angle_unit: str = "deg",
    ax=None,
    title: Optional[str] = None,
):
    """
    Plots FFT-based angle spectra for 1D ULA or 2D URA outputs.
    """
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    angle_map = np.asarray(angle_map)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    if angle_map.ndim == 1:
        if u is None and v is None:
            raise ValueError("For 1D ULA, provide u or v")
        dc = np.asarray(u if u is not None else v)
        theta = np.arcsin(np.clip(dc, -1.0, 1.0))
        if angle_unit == "deg":
            theta = np.degrees(theta)
        elif angle_unit != "rad":
            raise ValueError("angle_unit must be 'deg' or 'rad'")
        y = angle_map.astype(float)
        if db:
            y = np.maximum(10.0 * np.log10(np.maximum(y, 1e-24)), db_floor)
        ax.plot(theta, y)
        ax.set_xlabel(f"Angle ({angle_unit})")
        ax.set_ylabel("Power (dB)" if db else "Power")
        ax.grid(True)
    elif angle_map.ndim == 2:
        if u is None or v is None:
            raise ValueError("For 2D URA, provide both u and v")
        theta_x = np.arcsin(np.clip(u, -1.0, 1.0))
        theta_y = np.arcsin(np.clip(v, -1.0, 1.0))
        if angle_unit == "deg":
            theta_x = np.degrees(theta_x)
            theta_y = np.degrees(theta_y)
        elif angle_unit != "rad":
            raise ValueError("angle_unit must be 'deg' or 'rad'")
        z = angle_map.astype(float)
        if db:
            z = np.maximum(10.0 * np.log10(np.maximum(z, 1e-24)), db_floor)
        im = ax.imshow(z, origin="lower", aspect="auto",
                       extent=[theta_x[0], theta_x[-1],
                               theta_y[0], theta_y[-1]])
        ax.set_xlabel(f"Azimuth angle ({angle_unit})")
        ax.set_ylabel(f"Elevation angle ({angle_unit})")
        plt.colorbar(im, ax=ax, label="Power (dB)" if db else "Power")
    else:
        raise ValueError("angle_map must be 1D or 2D")

    if title is not None:
        ax.set_title(title)
    return ax



def plot_pseudospectrum(
    pseudospectrum,
    *,
    u: Optional[np.ndarray] = None,
    v: Optional[np.ndarray] = None,
    db: bool = True,
    db_floor: float = -50.0,
    angle_unit: str = "deg",
    ax=None,
    title: Optional[str] = None,
    colorbar: bool = True,
):
    """
    Plots MUSIC/MVDR pseudospectra returned by the array-processing helpers.

    ``pseudospectrum`` may be either a result dict with ``map``/``u``/``v``
    entries or a raw 1D/2D array with axes supplied separately.
    """
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    if isinstance(pseudospectrum, dict):
        spec = np.asarray(pseudospectrum["map"])
        u = pseudospectrum.get("u", u)
        v = pseudospectrum.get("v", v)
    else:
        spec = np.asarray(pseudospectrum)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    z = np.asarray(spec, dtype=float)
    if db:
        peak = np.maximum(np.max(z), 1e-24)
        z = 10.0 * np.log10(np.maximum(z, 1e-24) / peak)
        z = np.maximum(z, db_floor)

    if z.ndim == 1:
        if u is None and v is None:
            raise ValueError("For 1D pseudospectra, provide u or v")
        axis = np.asarray(u if u is not None else v, dtype=float)
        theta = _angle_axis(axis, angle_unit)
        ax.plot(theta, z)
        ax.set_xlabel(f"Angle ({angle_unit})")
        ax.set_ylabel("Relative pseudo power (dB)" if db else "Pseudo power")
        ax.grid(True)
    elif z.ndim == 2:
        if u is None or v is None:
            raise ValueError("For 2D pseudospectra, provide both u and v")
        theta_u = _angle_axis(np.asarray(u, dtype=float), angle_unit)
        theta_v = _angle_axis(np.asarray(v, dtype=float), angle_unit)
        im = ax.imshow(
            z,
            origin="lower",
            aspect="auto",
            extent=[theta_u[0], theta_u[-1], theta_v[0], theta_v[-1]],
        )
        ax.set_xlabel(f"Azimuth angle ({angle_unit})")
        ax.set_ylabel(f"Elevation angle ({angle_unit})")
        if colorbar:
            plt.colorbar(
                im,
                ax=ax,
                label="Relative pseudo power (dB)" if db else "Pseudo power",
            )
    else:
        raise ValueError("pseudospectrum must be 1D or 2D")

    if title is not None:
        ax.set_title(title)
    return ax


def _angle_axis(direction_cosine: np.ndarray, angle_unit: str) -> np.ndarray:
    """Converts direction cosines to the requested plotting angle unit."""

    theta = np.arcsin(np.clip(direction_cosine, -1.0, 1.0))
    if angle_unit == "deg":
        return np.degrees(theta)
    if angle_unit == "rad":
        return theta
    if angle_unit in ("direction_cosine", "u"):
        return direction_cosine
    raise ValueError("angle_unit must be 'deg', 'rad', or 'direction_cosine'")


def _axis_extent_bounds(axis: np.ndarray) -> tuple[float, float]:
    """Returns image extent bounds for a one- or multi-sample axis."""

    if axis.size == 1:
        value = float(axis[0])
        return value - 0.5, value + 0.5
    return float(axis[0]), float(axis[-1])


def _point_axis_index(axis: str) -> int:
    """Maps a Cartesian axis label to its point-coordinate index."""

    axes = {"x": 0, "y": 1, "z": 2}
    try:
        return axes[str(axis)]
    except KeyError as exc:
        raise ValueError("axis must be one of 'x', 'y', or 'z'") from exc
