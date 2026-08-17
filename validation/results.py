# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Result serialization for validation benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def json_default(value):
    """Converts NumPy and simulator objects into JSON-serializable values."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "to_jsonable"):
        return value.to_jsonable()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_metrics_json(path: str | Path, metrics: Mapping[str, Any]) -> None:
    """Writes benchmark metrics as stable, pretty-printed JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )


def save_maps_npz(path: str | Path, **arrays) -> None:
    """Writes non-``None`` diagnostic map arrays to a compressed NPZ file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        key: value
        for key, value in arrays.items()
        if value is not None
    }
    np.savez_compressed(path, **payload)


def plot_range_time(path: str | Path, power: np.ndarray, ranges_m: np.ndarray) -> None:
    """Writes a single range-time power image."""

    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    image = _relative_db(power).T
    extent = [0, power.shape[0] - 1, ranges_m[0], ranges_m[-1]]
    im = ax.imshow(image, aspect="auto", origin="lower", extent=extent,
                   cmap="viridis", vmin=-60.0, vmax=0.0)
    ax.set_xlabel("Slow-time index")
    ax.set_ylabel("Range [m]")
    ax.set_title("Range-time power")
    fig.colorbar(im, ax=ax, label="Relative power [dB]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_range_time_comparison(
    path: str | Path,
    real_power: np.ndarray,
    simulated_power: np.ndarray,
    ranges_m: np.ndarray,
    *,
    simulated_label: str = "Simulated",
    rt_power: np.ndarray | None = None,
    additional_simulated_powers: Mapping[str, np.ndarray] | None = None,
) -> None:
    """Plots real and one or more simulated range-time maps side by side.

    ``additional_simulated_powers`` preserves mapping order, allowing callers
    to display every requested simulator mode using its command-line name.
    ``rt_power`` remains supported for compatibility with older callers.
    """

    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    real = np.asarray(real_power, dtype=float)
    simulated = np.asarray(simulated_power, dtype=float)
    panels = [real, simulated]
    titles = ["Real range-time", f"{simulated_label} range-time"]
    if rt_power is not None:
        panels.append(np.asarray(rt_power, dtype=float))
        titles.append("RT range-time")
    for label, power in (additional_simulated_powers or {}).items():
        panels.append(np.asarray(power, dtype=float))
        titles.append(f"{label} range-time")
    extent = [0, real.shape[0] - 1, ranges_m[0], ranges_m[-1]]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4),
                             sharey=True,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    im = None
    for ax, panel, title in zip(axes, panels, titles):
        image = _relative_db(panel).T
        im = ax.imshow(image, aspect="auto", origin="lower", extent=extent,
                       cmap="viridis", vmin=-60.0, vmax=0.0)
        ax.set_xlabel("Slow-time index")
        ax.set_title(title)
    axes[0].set_ylabel("Range [m]")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Relative power [dB]")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_range_profile_comparison(
    path: str | Path,
    real_power: np.ndarray,
    simulated_power: np.ndarray,
    ranges_m: np.ndarray,
    *,
    simulated_label: str = "Simulated",
    rt_power: np.ndarray | None = None,
    additional_simulated_powers: Mapping[str, np.ndarray] | None = None,
) -> None:
    """Plots real and simulated mean range profiles for every supplied mode."""

    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("Real", np.asarray(real_power, dtype=float)),
        (simulated_label, np.asarray(simulated_power, dtype=float)),
    ]
    if rt_power is not None:
        panels.append(("RT", np.asarray(rt_power, dtype=float)))
    for label, power in (additional_simulated_powers or {}).items():
        panels.append((label, np.asarray(power, dtype=float)))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, power in panels:
        profile = _mean_range_profile(power)
        ax.plot(ranges_m, _relative_db(profile), label=label)
    ax.set_xlabel("Range [m]")
    ax.set_ylabel("Mean range magnitude [dB, per curve]")
    ax.set_title("Mean range profile")
    ax.set_ylim(-60.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_range_doppler(
    path: str | Path,
    power: np.ndarray,
    ranges_m: np.ndarray,
    velocities_mps: np.ndarray,
    *,
    frame_index: int = 0,
) -> None:
    """Writes a single-frame range-Doppler power image."""

    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.asarray(power)[int(frame_index)]
    fig, ax = plt.subplots(figsize=(7, 4))
    image, plot_velocities = _image_with_increasing_velocity(
        _relative_db(frame),
        velocities_mps,
    )
    extent = [ranges_m[0], ranges_m[-1], plot_velocities[0], plot_velocities[-1]]
    im = ax.imshow(image, aspect="auto", origin="lower", extent=extent,
                   cmap="magma", vmin=-60.0, vmax=0.0)
    ax.set_xlabel("Range [m]")
    ax.set_ylabel("Velocity [m/s, +away]")
    ax.set_title("Range-Doppler power")
    fig.colorbar(im, ax=ax, label="Relative power [dB]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_range_doppler_comparison(
    path: str | Path,
    real_power: np.ndarray,
    simulated_power: np.ndarray,
    ranges_m: np.ndarray,
    velocities_mps: np.ndarray,
    *,
    frame_index: int = 0,
    simulated_label: str = "Simulated",
    rt_power: np.ndarray | None = None,
    additional_simulated_powers: Mapping[str, np.ndarray] | None = None,
) -> None:
    """Plots real and simulated range-Doppler maps for every supplied mode."""

    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    real = np.asarray(real_power)[int(frame_index)]
    simulated = np.asarray(simulated_power)[int(frame_index)]
    panels = [real, simulated]
    titles = [
        "Real range-Doppler",
        f"{simulated_label} range-Doppler",
    ]
    if rt_power is not None:
        panels.append(np.asarray(rt_power)[int(frame_index)])
        titles.append("RT range-Doppler")
    for label, power in (additional_simulated_powers or {}).items():
        panels.append(np.asarray(power)[int(frame_index)])
        titles.append(f"{label} range-Doppler")
    plot_velocities = _increasing_velocity_axis(velocities_mps)
    extent = [ranges_m[0], ranges_m[-1], plot_velocities[0], plot_velocities[-1]]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4),
                             sharey=True,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    im = None
    for ax, panel, title in zip(axes, panels, titles):
        image, _ = _image_with_increasing_velocity(_relative_db(panel), velocities_mps)
        im = ax.imshow(image, aspect="auto", origin="lower", extent=extent,
                       cmap="magma", vmin=-60.0, vmax=0.0)
        ax.set_xlabel("Range [m]")
        ax.set_title(title)
    axes[0].set_ylabel("Velocity [m/s, +away]")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Relative power [dB]")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_angle_fft_comparison(
    path: str | Path,
    real_power: np.ndarray,
    simulated_power: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    *,
    frame_index: int = 0,
    real_label: str = "Real",
    simulated_label: str = "Simulated",
    rt_power: np.ndarray | None = None,
    rt_label: str = "RT",
    additional_simulated_powers: Mapping[str, np.ndarray] | None = None,
) -> None:
    """Plots real and simulated 2D FFT angle maps for every supplied mode."""

    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    real = _select_frame(real_power, frame_index)
    simulated = _select_frame(simulated_power, frame_index)
    panels = [real, simulated]
    titles = [
        f"{real_label} angle FFT",
        f"{simulated_label} angle FFT",
    ]
    if rt_power is not None:
        panels.append(_select_frame(rt_power, frame_index))
        titles.append(f"{rt_label} angle FFT")
    for label, power in (additional_simulated_powers or {}).items():
        panels.append(_select_frame(power, frame_index))
        titles.append(f"{label} angle FFT")

    u = np.asarray(u_axis, dtype=float)
    v = np.asarray(v_axis, dtype=float)
    extent = [u[0], u[-1], v[0], v[-1]]
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(6 * len(panels), 4),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    im = None
    for ax, panel, title in zip(axes, panels, titles):
        image = _relative_db(panel)
        im = ax.imshow(
            image,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="cividis",
            vmin=-60.0,
            vmax=0.0,
        )
        ax.set_xlabel("Horizontal direction cosine u")
        ax.set_title(title)
    axes[0].set_ylabel("Vertical direction cosine v")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Relative power [dB]")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _relative_db(power: np.ndarray, *, reference_max: float | None = None) -> np.ndarray:
    value = np.maximum(np.asarray(power, dtype=float), 1e-30)
    denom = float(np.max(value)) if reference_max is None else float(reference_max)
    return 10.0 * np.log10(value / max(denom, 1e-30))


def _increasing_velocity_axis(velocities_mps: np.ndarray) -> np.ndarray:
    velocities = np.asarray(velocities_mps, dtype=float)
    if velocities.ndim != 1:
        raise ValueError("velocities_mps must be one-dimensional")
    if velocities.size >= 2 and velocities[0] > velocities[-1]:
        return velocities[::-1]
    return velocities


def _image_with_increasing_velocity(
    image: np.ndarray,
    velocities_mps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    velocities = np.asarray(velocities_mps, dtype=float)
    image_arr = np.asarray(image)
    if velocities.ndim != 1:
        raise ValueError("velocities_mps must be one-dimensional")
    if velocities.size >= 2 and velocities[0] > velocities[-1]:
        return np.flip(image_arr, axis=0), velocities[::-1]
    return image_arr, velocities


def _select_frame(power: np.ndarray, frame_index: int) -> np.ndarray:
    value = np.asarray(power, dtype=float)
    if value.ndim == 2:
        return value
    return value[int(frame_index)]


def _mean_range_profile(power: np.ndarray) -> np.ndarray:
    value = np.asarray(power, dtype=float)
    if value.ndim == 1:
        return value
    return np.mean(value, axis=tuple(range(value.ndim - 1)))
