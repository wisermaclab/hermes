# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Hardware channel-gain helpers for simulated path coefficients."""

from __future__ import annotations

import numpy as np


def path_azimuth_elevation_deg(paths) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Extracts common path AoD angles from a Sionna ``Paths`` object.

    Sionna reports zenith ``theta`` and azimuth ``phi`` in the device local
    spherical frame. The simulator's TI board convention uses azimuth around
    local boresight ``x`` and elevation from the local ``x-y`` plane.
    """

    if not hasattr(paths, "theta_t") or not hasattr(paths, "phi_t"):
        return None, None
    theta = _last_axis_vector(getattr(paths, "theta_t"))
    phi = _last_axis_vector(getattr(paths, "phi_t"))
    if theta is None or phi is None:
        return None, None
    if theta.shape != phi.shape:
        return None, None
    azimuth = np.degrees(phi)
    elevation = 90.0 - np.degrees(theta)
    return azimuth, elevation


def apply_path_hardware_gain(
    hardware,
    coefficients: np.ndarray,
    *,
    paths=None,
    frequency_hz: float | None = None,
) -> np.ndarray:
    """Applies hardware phase/scalar/pattern gains to path coefficients."""

    azimuth = elevation = None
    if paths is not None:
        azimuth, elevation = path_azimuth_elevation_deg(paths)
    return hardware.apply_channel_gain(
        coefficients,
        azimuth_deg=azimuth,
        elevation_deg=elevation,
        frequency_hz=frequency_hz,
    )


def _last_axis_vector(value) -> np.ndarray | None:
    """Extracts a first-link vector from a Sionna tensor-like angle field."""

    if hasattr(value, "numpy"):
        value = value.numpy()
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return np.zeros((0,), dtype=float)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return arr.reshape((1,))
    index = (0,) * (arr.ndim - 1) + (slice(None),)
    return np.asarray(arr[index], dtype=float).reshape(-1)
