# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Hardware channel-gain helpers for simulated path coefficients."""

from __future__ import annotations

import numpy as np

from ..radar.sensor import _rotation_matrix_numpy


def path_azimuth_elevation_deg(
    paths,
    *,
    radar_orientation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    endpoint: str = "tx",
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Returns radar-local endpoint look angles from scene-frame Sionna angles.

    Sionna reports departure and arrival look directions in the world frame. Rotate
    their unit directions into the radar frame before evaluating hardware
    patterns. ``radar_orientation`` is the device's scene-frame yaw, pitch,
    roll in radians. Local ``x`` is boresight and local ``z`` is up.
    """

    suffix = {"tx": "t", "rx": "r"}[endpoint]
    if not hasattr(paths, f"theta_{suffix}") or not hasattr(paths, f"phi_{suffix}"):
        return None, None
    theta = _last_axis_vector(getattr(paths, f"theta_{suffix}"))
    phi = _last_axis_vector(getattr(paths, f"phi_{suffix}"))
    if theta is None or phi is None:
        return None, None
    if theta.shape != phi.shape:
        return None, None
    scene_directions = np.stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ], axis=-1)
    # R maps local column vectors to the scene frame; row-vector inverse is @ R.
    local_directions = scene_directions @ _rotation_matrix_numpy(radar_orientation)
    azimuth = np.degrees(np.arctan2(local_directions[:, 1], local_directions[:, 0]))
    elevation = np.degrees(np.arctan2(
        local_directions[:, 2],
        np.hypot(local_directions[:, 0], local_directions[:, 1]),
    ))
    return azimuth, elevation


def apply_path_hardware_gain(
    hardware,
    coefficients: np.ndarray,
    *,
    paths=None,
    frequency_hz: float | None = None,
    radar_orientation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Applies hardware gains after converting scene directions to radar-local."""

    azimuth = elevation = None
    if paths is not None:
        azimuth, elevation = path_azimuth_elevation_deg(
            paths, radar_orientation=radar_orientation,
        )
        rx_az, rx_el = path_azimuth_elevation_deg(
            paths, radar_orientation=radar_orientation, endpoint="rx",
        )
        if azimuth is not None and rx_az is not None:
            # The hardware model supplies a combined channel pattern. Use the
            # same equal Tx/Rx split as PO, with scalar gain and phase once.
            tx_gain = hardware.channel_gain(
                azimuth_deg=azimuth, elevation_deg=elevation,
                frequency_hz=frequency_hz, include_phase=False,
            )
            rx_gain = hardware.channel_gain(
                azimuth_deg=rx_az, elevation_deg=rx_el,
                frequency_hz=frequency_hz, include_phase=False,
            )
            phase = hardware.channel_gain(
                include_antenna_gain=False, include_pattern=False,
            ).reshape((-1, 1))
            return np.asarray(coefficients) * np.sqrt(tx_gain * rx_gain) * phase
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
