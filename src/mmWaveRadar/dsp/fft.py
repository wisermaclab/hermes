# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Range FFT helpers for FMCW radar data."""

from __future__ import annotations

import operator
from typing import Optional

import numpy as np

from .constants import SPEED_OF_LIGHT
from .clutter import remove_slow_time_mean
from .profiles import range_profile_from_cube
from .window import window_1d


def _fmcw_value(fmcw, name: str):
    """Reads an FMCW attribute from an optional configuration object."""

    return None if fmcw is None else getattr(fmcw, name)


def phase2distance(theta, wavelength):
    """Converts monostatic phase change [rad] to displacement [m]."""
    return np.asarray(theta) * float(wavelength) / (4.0 * np.pi)


def range_fft(
    adc: np.ndarray,
    sampling_frequency: float | None = None,
    slope: float | None = None,
    *,
    fmcw=None,
    window: Optional[str] = "hann",
    nfft_mult: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the one-sided range FFT in the chirp-slope direction.

    :param adc: ADC samples with shape ``[num_chirps, num_adc_samples]`` or
        ``[num_chirps, num_adc_samples, num_channels]``.
    :param sampling_frequency: ADC sampling frequency [Hz].
    :param slope: FMCW slope [Hz/s].
    :param fmcw: Optional object with ``sampling_frequency`` and ``slope``.
    :param window: Fast-time window kind.
    :param nfft_mult: Range FFT zero-padding multiplier.
    """
    if sampling_frequency is None:
        sampling_frequency = _fmcw_value(fmcw, "sampling_frequency")
    if slope is None:
        slope = _fmcw_value(fmcw, "slope")
    if sampling_frequency is None or slope is None:
        raise ValueError("sampling_frequency and slope are required")
    try:
        sampling_frequency = float(sampling_frequency)
        slope = float(slope)
    except (TypeError, ValueError) as exc:
        raise ValueError("sampling_frequency and slope must be real scalars") \
            from exc
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0.0:
        raise ValueError("sampling_frequency must be finite and positive")
    if not np.isfinite(slope) or slope == 0.0:
        raise ValueError("slope must be finite and non-zero")
    if isinstance(nfft_mult, (bool, np.bool_)):
        raise ValueError("nfft_mult must be an integer >= 1")
    try:
        nfft_mult = operator.index(nfft_mult)
    except TypeError as exc:
        raise ValueError("nfft_mult must be an integer >= 1") from exc
    if nfft_mult < 1:
        raise ValueError("nfft_mult must be an integer >= 1")
    nfft_mult = int(nfft_mult)

    x = np.asarray(adc)
    if x.ndim not in (2, 3):
        raise ValueError("adc must have shape [M, N] or [M, N, P]")

    n = x.shape[1]
    if n == 0:
        raise ValueError("adc must contain at least one fast-time sample")
    nfft = nfft_mult * n
    w = window_1d(n, window)
    xw = x * w[None, :, None] if x.ndim == 3 else x * w[None, :]

    rt = np.fft.fft(xw, n=nfft, axis=1)
    num_range_bins = nfft // 2
    offsets = np.arange(num_range_bins, dtype=np.int64)
    fft_bins = offsets if slope > 0.0 else (-offsets) % nfft
    rt = np.take(rt, fft_bins, axis=1)
    beat_frequencies = offsets.astype(float) * sampling_frequency / nfft
    ranges_m = SPEED_OF_LIGHT * beat_frequencies / (2.0 * abs(slope))
    return rt, ranges_m


def range_time_map(
    adc: np.ndarray,
    sampling_frequency: float | None = None,
    slope: float | None = None,
    *,
    fmcw=None,
    window: Optional[str] = "hann",
    nfft_mult: int = 1,
    combine_channels: str = "sum_power",
    remove_mean_clutter: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds a power range-time map from ADC data.

    ``adc`` may be a single frame with shape
    ``[num_chirps, num_adc_samples, num_channels]`` or a multi-frame cube with
    shape ``[num_frames, num_chirps, num_adc_samples, num_channels]``. The
    returned map flattens frame/chirp slow time into one axis.
    """
    x = np.asarray(adc)
    if x.ndim == 4:
        x = x.reshape((-1, x.shape[-2], x.shape[-1]))
    if x.ndim == 2:
        x = x[:, :, None]
    if x.ndim != 3:
        raise ValueError("adc must have shape [M, N], [M, N, P], or"
                         " [F, M, N, P]")

    rt, ranges_m = range_fft(x, sampling_frequency, slope, fmcw=fmcw,
                             window=window, nfft_mult=nfft_mult)
    if remove_mean_clutter:
        rt = remove_slow_time_mean(rt)
    return range_profile_from_cube(rt, combine_antennas=combine_channels), \
        ranges_m
