# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Analytic FMCW point-target synthesis helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import operator
from typing import Any

import numpy as np

from ..dsp import angle_map_fft, range_fft
from ..dsp.constants import SPEED_OF_LIGHT
from ..radar.fmcw import FMCWConfig
from ..radar.hardware import RadarHardware
from .adc_synthesis import synthesize_adc_from_paths


@dataclass(frozen=True)
class PointTarget:
    """Far-field point target in range/azimuth/elevation coordinates."""

    range_m: float
    azimuth_deg: float
    elevation_deg: float
    reflectivity: complex = 1.0 + 0.0j

    def __post_init__(self):
        range_m = _finite_float("range_m", self.range_m)
        if range_m < 0.0:
            raise ValueError("range_m must be non-negative")
        azimuth_deg = _finite_float("azimuth_deg", self.azimuth_deg)
        elevation_deg = _finite_float("elevation_deg", self.elevation_deg)
        try:
            reflectivity = complex(self.reflectivity)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("reflectivity must be a finite complex scalar") from exc
        if not np.isfinite(reflectivity):
            raise ValueError("reflectivity must be a finite complex scalar")
        object.__setattr__(self, "range_m", range_m)
        object.__setattr__(self, "azimuth_deg", azimuth_deg)
        object.__setattr__(self, "elevation_deg", elevation_deg)
        object.__setattr__(self, "reflectivity", reflectivity)


@dataclass(frozen=True)
class PointTargetSynthesisResult:
    """ADC synthesis output for a set of analytic point targets."""

    hardware: RadarHardware
    fmcw: FMCWConfig
    targets: tuple[PointTarget, ...]
    coefficients: np.ndarray
    delays_s: np.ndarray
    adc: np.ndarray


def make_point_target_fmcw_config(
    *,
    carrier_frequency: float,
    slope: float = 68.0e12,
    chirp_duration: float = 58.0e-6,
    chirp_repetition_time: float = 65.0e-6,
    sampling_frequency: float = 4.5e6,
    num_adc_samples: int = 225,
    num_chirps_per_frame: int = 96,
    frame_period: float = 50.0e-3,
    num_tx: int = 1,
    tdm_enabled: bool = False,
) -> FMCWConfig:
    """Builds a configurable FMCW profile for analytic point-target examples."""

    return FMCWConfig(
        carrier_frequency=carrier_frequency,
        slope=slope,
        chirp_duration=chirp_duration,
        chirp_repetition_time=chirp_repetition_time,
        sampling_frequency=sampling_frequency,
        num_adc_samples=num_adc_samples,
        num_chirps_per_frame=num_chirps_per_frame,
        frame_period=frame_period,
        num_tx=num_tx,
        tdm_enabled=tdm_enabled,
    )


def target_direction_cosines(
    azimuth_deg: float,
    elevation_deg: float,
) -> np.ndarray:
    """Returns ``[horizontal, vertical]`` direction cosines."""

    az = np.deg2rad(_finite_float("azimuth_deg", azimuth_deg))
    el = np.deg2rad(_finite_float("elevation_deg", elevation_deg))
    return np.array([np.sin(az) * np.cos(el), np.sin(el)], dtype=float)


def synthesize_virtual_center_point_targets(
    hardware: RadarHardware,
    fmcw: FMCWConfig,
    targets: Sequence[PointTarget | Mapping[str, Any]],
    *,
    include_channel_phase: bool = True,
    include_antenna_gain: bool = False,
    include_pattern: bool = False,
    apply_range_spreading: bool = True,
    backend: str = "numpy",
    precision: str = "float64",
) -> PointTargetSynthesisResult:
    """
    Synthesizes one analytic FMCW ADC snapshot for virtual-array point targets.

    This is a far-field virtual-center model: every virtual channel shares the
    target-center round-trip delay, while virtual-channel geometry enters
    through the array phase and optional channel coefficients. Use a full
    per-channel path-length model for near-field propagation.
    """

    positions_lambda = _virtual_positions_lambda(
        hardware,
        wavelength=fmcw.wavelength,
    )
    target_tuple = tuple(_coerce_point_target(target) for target in targets)
    coeffs = np.zeros(
        (hardware.num_virtual_channels, len(target_tuple)),
        dtype=np.complex128,
    )
    delays_s = np.zeros((len(target_tuple),), dtype=float)

    for path_index, target in enumerate(target_tuple):
        direction_yz = target_direction_cosines(
            target.azimuth_deg,
            target.elevation_deg,
        )
        delays_s[path_index] = 2.0 * target.range_m / SPEED_OF_LIGHT
        array_phase = np.exp(
            1j * 2.0 * np.pi * (positions_lambda @ direction_yz)
        )
        gain = hardware.channel_gain(
            azimuth_deg=target.azimuth_deg,
            elevation_deg=target.elevation_deg,
            frequency_hz=fmcw.carrier_frequency,
            include_phase=include_channel_phase,
            include_antenna_gain=include_antenna_gain,
            include_pattern=include_pattern,
        )
        if apply_range_spreading:
            spreading = 1.0 / max(target.range_m * target.range_m, 1e-12)
        else:
            spreading = 1.0
        coeffs[:, path_index] = (
            target.reflectivity * spreading * gain * array_phase
        )

    adc = synthesize_adc_from_paths(
        coeffs,
        delays_s,
        fmcw,
        backend=backend,
        precision=precision,
    )
    return PointTargetSynthesisResult(
        hardware=hardware,
        fmcw=fmcw,
        targets=target_tuple,
        coefficients=coeffs,
        delays_s=delays_s,
        adc=adc,
    )


def virtual_snapshot_grid(
    range_cube: np.ndarray,
    hardware: RadarHardware,
    *,
    half_lambda_scale: int = 2,
    wavelength: float | None = None,
    align_origin: bool = False,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """
    Places virtual-channel snapshots on a regular wavelength-normalized grid.

    Set ``align_origin`` for regular arrays whose coordinate origin is offset
    from the grid, such as an even-sized centered URA. When ``return_metadata``
    is true, callers receive ``(grid, metadata)``, where metadata contains the
    occupied-cell mask, local ``y``/``z`` coordinate axes, and their spacings
    in wavelengths. Synthetic virtual URAs use their configured per-axis
    spacings. Other hardware retains the legacy
    lambda/``half_lambda_scale`` lattice.
    """

    scale = _positive_integer("half_lambda_scale", half_lambda_scale)
    if not isinstance(align_origin, (bool, np.bool_)):
        raise ValueError("align_origin must be a boolean")
    if not isinstance(return_metadata, (bool, np.bool_)):
        raise ValueError("return_metadata must be a boolean")
    positions = _virtual_positions_lambda(hardware, wavelength=wavelength)
    spacing_y, spacing_z = _virtual_grid_spacings_lambda(hardware, scale)
    grid_spacing = np.array([spacing_y, spacing_z], dtype=float)
    origin = np.min(positions, axis=0) if align_origin else np.zeros(2)
    grid_positions = positions - origin[None, :]
    grid_index = np.rint(grid_positions / grid_spacing[None, :]).astype(int)
    if not np.allclose(
        grid_index * grid_spacing[None, :],
        grid_positions,
        atol=1e-8,
    ):
        lattice = (
            f"lambda/{scale}"
            if np.allclose(grid_spacing, 1.0 / float(scale))
            else (
                f"the configured y/z wavelength grid "
                f"({spacing_y:g}, {spacing_z:g})"
            )
        )
        raise ValueError(
            f"{hardware.name} virtual coordinates are not on a "
            f"{lattice} grid"
        )

    x = np.asarray(range_cube)
    if x.ndim < 1 or x.shape[-1] != hardware.num_virtual_channels:
        raise ValueError(
            "range_cube must have the virtual-channel axis last and match "
            "hardware.num_virtual_channels"
        )

    min_index = grid_index.min(axis=0)
    shifted = grid_index - min_index
    nx = int(shifted[:, 0].max()) + 1
    ny = int(shifted[:, 1].max()) + 1
    grid = np.zeros((*x.shape[:-1], ny, nx), dtype=np.complex128)
    counts = np.zeros((ny, nx), dtype=float)

    for channel, (ix, iy) in enumerate(shifted):
        grid[..., iy, ix] += x[..., channel]
        counts[iy, ix] += 1.0

    occupied = counts > 0.0
    grid[..., occupied] /= counts[occupied]
    if return_metadata:
        y_lambda = (
            origin[0]
            + (min_index[0] + np.arange(nx)) * spacing_y
        )
        z_lambda = (
            origin[1]
            + (min_index[1] + np.arange(ny)) * spacing_z
        )
        metadata = {
            "occupied": occupied,
            "y_lambda": y_lambda,
            "z_lambda": z_lambda,
            "spacing_y_lambda": spacing_y,
            "spacing_z_lambda": spacing_z,
            # Compatibility alias for callers that treated the horizontal
            # aperture as a generic array-x coordinate.
            "x_lambda": y_lambda,
        }
        return grid, metadata
    return grid


def calibrate_range_cube_channel_phase(
    range_cube: np.ndarray,
    hardware: RadarHardware,
    *,
    enabled: bool = True,
) -> np.ndarray:
    """Removes fixed virtual-channel phase signs before array FFT processing."""

    if not enabled:
        return range_cube
    signs = hardware.virtual_channel_phase_signs
    if signs is None:
        return range_cube
    x = np.asarray(range_cube)
    if x.ndim < 1 or x.shape[-1] != hardware.num_virtual_channels:
        raise ValueError(
            "range_cube must have the virtual-channel axis last and match "
            "hardware.num_virtual_channels"
        )
    safe = np.where(np.abs(signs) > 0.0, signs, 1.0 + 0.0j).astype(
        np.complex128
    )
    safe = safe.reshape((1,) * (x.ndim - 1) + (safe.size,))
    return x / safe


def range_resolved_angle_fft(
    adc: np.ndarray,
    hardware: RadarHardware,
    fmcw: FMCWConfig,
    *,
    range_nfft_mult: int = 8,
    angle_fft_size: tuple[int, int] | int = (64, 128),
    range_window: str | None = "hann",
    angle_window: str | None = None,
    calibrate_channel_phase: bool = True,
    half_lambda_scale: int = 2,
    power: bool = True,
) -> dict[str, Any]:
    """Computes a range-resolved FFT angle map for a virtual-array ADC snapshot."""

    x = np.asarray(adc)
    single_snapshot = x.ndim == 2
    if single_snapshot:
        x = x[None, :, :]
    elif x.ndim != 3:
        raise ValueError(
            "adc must have shape [num_adc_samples, num_virtual_channels] or "
            "[num_chirps, num_adc_samples, num_virtual_channels]"
        )
    rt, ranges_m = range_fft(
        x,
        fmcw=fmcw,
        window=range_window,
        nfft_mult=range_nfft_mult,
    )
    range_cube = calibrate_range_cube_channel_phase(
        rt[0] if single_snapshot else rt,
        hardware,
        enabled=calibrate_channel_phase,
    )
    grid, grid_metadata = virtual_snapshot_grid(
        range_cube,
        hardware,
        half_lambda_scale=half_lambda_scale,
        wavelength=fmcw.wavelength,
        align_origin=True,
        return_metadata=True,
    )
    angle = angle_map_fft(
        grid,
        fc_hz=fmcw.carrier_frequency,
        window=angle_window,
        fft_size=angle_fft_size,
        power=power,
        spacing_y_lambda=grid_metadata["spacing_y_lambda"],
        spacing_z_lambda=grid_metadata["spacing_z_lambda"],
    )
    return {
        "ranges_m": ranges_m,
        "range_cube": range_cube,
        "grid": grid,
        "grid_metadata": grid_metadata,
        "angle": angle,
    }


def azimuth_elevation_axes(angle: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Returns independent broadside-plane projections of ``u`` and ``v``.

    The first result equals physical azimuth only where ``v == 0``. Use
    :func:`azimuth_elevation_grids` when both direction-cosine dimensions vary.
    """

    azimuth_deg = np.rad2deg(
        np.arcsin(np.clip(np.asarray(angle["u"]), -1.0, 1.0))
    )
    elevation_deg = np.rad2deg(
        np.arcsin(np.clip(np.asarray(angle["v"]), -1.0, 1.0))
    )
    return azimuth_deg, elevation_deg


def azimuth_elevation_grids(
    angle: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Converts a rectangular ``u``/``v`` map to coupled physical-angle grids.

    The returned arrays both have shape ``[len(v), len(u)]``. Samples outside
    the visible direction-cosine disk are represented by ``nan``.
    """

    u_axis = np.asarray(angle["u"], dtype=float).reshape(-1)
    v_axis = np.asarray(angle["v"], dtype=float).reshape(-1)
    if u_axis.size == 0 or v_axis.size == 0:
        raise ValueError("u and v must be nonempty")
    if not np.all(np.isfinite(u_axis)) or not np.all(np.isfinite(v_axis)):
        raise ValueError("u and v must contain only finite values")
    u_grid, v_grid = np.meshgrid(u_axis, v_axis, indexing="xy")
    forward_squared = 1.0 - u_grid**2 - v_grid**2
    valid = forward_squared >= -1e-12
    forward = np.sqrt(np.maximum(forward_squared, 0.0))
    azimuth_deg = np.rad2deg(np.arctan2(u_grid, forward))
    elevation_deg = np.rad2deg(np.arcsin(np.clip(v_grid, -1.0, 1.0)))
    return (
        np.where(valid, azimuth_deg, np.nan),
        np.where(valid, elevation_deg, np.nan),
    )


def db_relative(power: np.ndarray) -> np.ndarray:
    """Converts a power map to dB relative to its maximum."""

    power = np.asarray(power, dtype=float)
    if power.size == 0:
        raise ValueError("power must be non-empty")
    if not np.all(np.isfinite(power)) or np.any(power < 0.0):
        raise ValueError("power must contain finite, non-negative values")
    return 10.0 * np.log10(
        np.maximum(power, 1e-30) / np.maximum(np.max(power), 1e-30)
    )


def _virtual_positions_lambda(
    hardware: RadarHardware,
    *,
    wavelength: float | None = None,
) -> np.ndarray:
    """Returns virtual-array ``[horizontal, vertical]`` coordinates in wavelengths."""

    positions = hardware.virtual_channel_positions_lambda
    if positions is None and wavelength is None:
        raise ValueError(
            "hardware.virtual_channel_positions_lambda is required when "
            "wavelength is not supplied"
        )
    if positions is None:
        wavelength_m = _finite_float("wavelength", wavelength)
        if wavelength_m <= 0.0:
            raise ValueError("wavelength must be positive")
        tx_indices = hardware.virtual_tx_indices()
        rx_indices = hardware.virtual_rx_indices()
        tx_yz = hardware.tx_positions[tx_indices, 1:3]
        rx_yz = hardware.rx_positions[rx_indices, 1:3]
        # The far-field MIMO phase term is k * (tx_yz + rx_yz) dot direction;
        # this is the virtual-array coordinate used by the steering phase.
        positions = (tx_yz + rx_yz) / wavelength_m
    positions = np.asarray(positions, dtype=float)
    if positions.shape != (hardware.num_virtual_channels, 2):
        raise ValueError(
            "hardware.virtual_channel_positions_lambda must have shape "
            "(num_virtual_channels, 2)"
        )
    if not np.all(np.isfinite(positions)):
        raise ValueError(
            "hardware virtual-channel positions must contain finite values"
        )
    return positions


def _virtual_grid_spacings_lambda(
    hardware: RadarHardware,
    half_lambda_scale: int,
) -> tuple[float, float]:
    """Returns the local-y/local-z grid spacings in wavelengths."""

    fallback = 1.0 / float(half_lambda_scale)
    metadata = hardware.board_metadata or {}
    legacy = metadata.get("spacing_lambda")
    spacing_y_value = metadata.get("spacing_y_lambda", legacy)
    spacing_z_value = metadata.get("spacing_z_lambda", legacy)
    spacing_y = (
        fallback
        if spacing_y_value is None
        else _finite_float("spacing_y_lambda", spacing_y_value)
    )
    spacing_z = (
        fallback
        if spacing_z_value is None
        else _finite_float("spacing_z_lambda", spacing_z_value)
    )
    if spacing_y <= 0.0:
        raise ValueError("spacing_y_lambda must be positive")
    if spacing_z <= 0.0:
        raise ValueError("spacing_z_lambda must be positive")
    return spacing_y, spacing_z


def _finite_float(name: str, value: object) -> float:
    """Returns ``value`` as a finite real scalar."""

    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite real scalar")
    return scalar


def _positive_integer(name: str, value: object) -> int:
    """Returns a positive integer without truncating floats or booleans."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(integer)


def _coerce_point_target(
    target: PointTarget | Mapping[str, Any],
) -> PointTarget:
    """Converts a mapping or existing instance into a :class:`PointTarget`."""

    if isinstance(target, PointTarget):
        return target
    if not isinstance(target, Mapping):
        raise TypeError("targets must contain PointTarget or mapping entries")
    return PointTarget(
        range_m=target["range_m"],
        azimuth_deg=target["azimuth_deg"],
        elevation_deg=target["elevation_deg"],
        reflectivity=target.get("reflectivity", 1.0 + 0.0j),
    )


__all__ = [
    "PointTarget",
    "PointTargetSynthesisResult",
    "azimuth_elevation_axes",
    "azimuth_elevation_grids",
    "calibrate_range_cube_channel_phase",
    "db_relative",
    "make_point_target_fmcw_config",
    "range_resolved_angle_fft",
    "synthesize_virtual_center_point_targets",
    "target_direction_cosines",
    "virtual_snapshot_grid",
]
