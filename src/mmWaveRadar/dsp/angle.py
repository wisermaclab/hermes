# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""FFT angle spectrum helpers."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

from .constants import SPEED_OF_LIGHT
from .window import window_1d


def direction_cosine_valid_mask(
    u: np.ndarray,
    v: np.ndarray | None = None,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Returns the physically valid direction-cosine mask for angle axes.

    For a 2D ``u``/``v`` grid, valid points satisfy ``u**2 + v**2 <= 1``.
    If ``v`` is omitted, the returned 1D mask checks ``abs(u) <= 1``.
    """

    tolerance = float(eps)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("eps must be a nonnegative finite value")

    u_axis = np.asarray(u, dtype=float).reshape(-1)
    if u_axis.size == 0:
        raise ValueError("u must be nonempty")
    if not np.all(np.isfinite(u_axis)):
        raise ValueError("u must contain only finite values")
    if v is None:
        return np.abs(u_axis) <= 1.0 + tolerance

    v_axis = np.asarray(v, dtype=float).reshape(-1)
    if v_axis.size == 0:
        raise ValueError("v must be nonempty")
    if not np.all(np.isfinite(v_axis)):
        raise ValueError("v must contain only finite values")
    return (v_axis[:, None] ** 2 + u_axis[None, :] ** 2) <= 1.0 + tolerance


def angle_map_2d_fft(
    x_ura: np.ndarray,
    *,
    fc_hz: float,
    window: Optional[str] = "hann",
    fft_size: Optional[tuple[int, int]] = None,
    power: bool = True,
    c: float = SPEED_OF_LIGHT,
    spacing: str = "lambda_over_2",
    spacing_y_lambda: float | None = None,
    spacing_z_lambda: float | None = None,
) -> Dict[str, Any]:
    """
    Computes a 2D FFT angle map for rectangular URA snapshots.

    ``x_ura`` must have shape ``[..., Nz, Ny]``. The last axis follows the
    radar-local ``y`` coordinate and the penultimate axis follows radar-local
    ``z``. ``spacing_y_lambda`` and ``spacing_z_lambda`` give the respective
    grid-cell spacings in wavelengths; both default to ``0.5``.
    """
    x_ura = np.asarray(x_ura)
    if x_ura.ndim < 2:
        raise ValueError("x_ura must have at least 2 dims [..., Nz, Ny]")

    nz, ny = x_ura.shape[-2], x_ura.shape[-1]
    nfft_z, nfft_y = (nz, ny) if fft_size is None else (
        int(fft_size[0]), int(fft_size[1]))
    if spacing != "lambda_over_2":
        raise ValueError("Only spacing='lambda_over_2' is supported")
    wave_speed = _positive_finite("c", c)
    carrier_frequency = _positive_finite("fc_hz", fc_hz)
    wavelength = wave_speed / carrier_frequency
    spacing_y = _normalized_spacing("spacing_y_lambda", spacing_y_lambda)
    spacing_z = _normalized_spacing("spacing_z_lambda", spacing_z_lambda)

    wz = window_1d(nz, window)
    wy = window_1d(ny, window)
    spec = np.fft.fftshift(
        np.fft.fft2(x_ura * (wz[:, None] * wy[None, :]),
                    s=(nfft_z, nfft_y), axes=(-2, -1)),
        axes=(-2, -1))
    fy = np.fft.fftshift(
        np.fft.fftfreq(nfft_y, d=wavelength * spacing_y)
    )
    fz = np.fft.fftshift(
        np.fft.fftfreq(nfft_z, d=wavelength * spacing_z)
    )
    mag = np.abs(spec)
    return {
        "map": mag ** 2 if power else mag,
        "spectrum": spec,
        "u": wavelength * fy,
        "v": wavelength * fz,
        "spacing_y_lambda": spacing_y,
        "spacing_z_lambda": spacing_z,
    }


def angle_map_fft(
    x_array: np.ndarray,
    *,
    fc_hz: float,
    window: Optional[str] = "hann",
    fft_size: Optional[Tuple[int, int] | int] = None,
    power: bool = True,
    c: float = SPEED_OF_LIGHT,
    spacing: Literal["lambda_over_2"] = "lambda_over_2",
    spacing_y_lambda: float | None = None,
    spacing_z_lambda: float | None = None,
) -> Dict[str, Any]:
    """
    Computes FFT angle spectra for 1D ULA or 2D URA snapshots.

    ``x_array`` must have shape ``[..., Nz, Ny]``. The last axis follows the
    radar-local ``y`` coordinate and the penultimate axis follows radar-local
    ``z``. For a ULA, one of those dimensions should be one. The corresponding
    wavelength-normalized grid spacings default to ``0.5``.
    """
    x_array = np.asarray(x_array)
    if x_array.ndim < 2:
        raise ValueError("x_array must have at least 2 dims [..., Nz, Ny]")

    nz, ny = x_array.shape[-2], x_array.shape[-1]
    if nz < 1 or ny < 1:
        raise ValueError("Invalid array shape")

    if nz > 1 and ny > 1:
        if isinstance(fft_size, int):
            fft_size = (fft_size, fft_size)
        return angle_map_2d_fft(x_array, fc_hz=fc_hz, window=window,
                                fft_size=fft_size, power=power, c=c,
                                spacing=spacing,
                                spacing_y_lambda=spacing_y_lambda,
                                spacing_z_lambda=spacing_z_lambda)

    if spacing != "lambda_over_2":
        raise ValueError("Only spacing='lambda_over_2' is supported")
    wave_speed = _positive_finite("c", c)
    carrier_frequency = _positive_finite("fc_hz", fc_hz)
    wavelength = wave_speed / carrier_frequency
    spacing_y = _normalized_spacing("spacing_y_lambda", spacing_y_lambda)
    spacing_z = _normalized_spacing("spacing_z_lambda", spacing_z_lambda)

    if nz == 1 and ny > 1:
        nfft = ny if fft_size is None else (
            int(fft_size) if isinstance(fft_size, int) else int(fft_size[1]))
        wy = window_1d(ny, window)
        spec = np.fft.fftshift(np.fft.fft(x_array * wy, n=nfft, axis=-1),
                               axes=(-1,))
        fy = np.fft.fftshift(
            np.fft.fftfreq(nfft, d=wavelength * spacing_y)
        )
        mag = np.abs(spec)
        return {"map": np.squeeze(mag ** 2 if power else mag, axis=-2),
                "spectrum": np.squeeze(spec, axis=-2),
                "u": wavelength * fy,
                "spacing_y_lambda": spacing_y}

    if ny == 1 and nz > 1:
        nfft = nz if fft_size is None else (
            int(fft_size) if isinstance(fft_size, int) else int(fft_size[0]))
        wz = window_1d(nz, window)
        spec = np.fft.fftshift(
            np.fft.fft(x_array * wz[:, None], n=nfft, axis=-2),
            axes=(-2,))
        fz = np.fft.fftshift(
            np.fft.fftfreq(nfft, d=wavelength * spacing_z)
        )
        mag = np.abs(spec)
        return {"map": np.squeeze(mag ** 2 if power else mag, axis=-1),
                "spectrum": np.squeeze(spec, axis=-1),
                "v": wavelength * fz,
                "spacing_z_lambda": spacing_z}

    spec = x_array.astype(np.complex128)
    mag = np.abs(spec)
    return {"map": mag ** 2 if power else mag, "spectrum": spec}


def _normalized_spacing(name: str, value: float | None) -> float:
    """Returns a positive wavelength-normalized grid spacing."""

    return _positive_finite(name, 0.5 if value is None else value)


def _positive_finite(name: str, value: object) -> float:
    """Returns ``value`` as a positive finite float."""

    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return scalar
