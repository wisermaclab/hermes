# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Range-Doppler processing helpers."""

from __future__ import annotations

import operator
from typing import Literal, Optional

import numpy as np

from .constants import SPEED_OF_LIGHT
from .clutter import remove_slow_time_mean
from .fft import range_fft
from .window import window_1d


def _attr(obj, *names):
    """Returns the first available named attribute from an optional object."""

    if obj is None:
        return None
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _positive_float(name: str, value: object) -> float:
    """Returns a finite positive scalar with a user-facing error."""

    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite positive scalar") from exc
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
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


def _boolean(name: str, value: object) -> bool:
    """Returns a strict boolean with a user-facing error."""

    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _resolve_doppler_parameters(
    *,
    sampling_frequency,
    slope,
    carrier_frequency,
    pri,
    fmcw,
    num_tx,
    tdm_enabled,
) -> tuple[object, object, float, float]:
    """Resolves shared waveform parameters for Doppler helpers."""

    if sampling_frequency is None:
        sampling_frequency = _attr(fmcw, "sampling_frequency")
    if slope is None:
        slope = _attr(fmcw, "slope")
    if carrier_frequency is None:
        carrier_frequency = _attr(fmcw, "carrier_frequency")
    if pri is None:
        chirp_repetition_time = _attr(fmcw, "chirp_repetition_time")
        if tdm_enabled is None:
            tdm_enabled = _attr(fmcw, "tdm_enabled")
        if tdm_enabled is None:
            tdm_enabled = False
        tdm_enabled = _boolean("tdm_enabled", tdm_enabled)
        if num_tx is None:
            num_tx = _attr(fmcw, "num_tx")
        if num_tx is None:
            num_tx = 1
        num_tx = _positive_integer("num_tx", num_tx)
        if chirp_repetition_time is not None:
            pri = (num_tx if tdm_enabled else 1) * _positive_float(
                "chirp_repetition_time",
                chirp_repetition_time,
            )
    if (
        sampling_frequency is None
        or slope is None
        or carrier_frequency is None
        or pri is None
    ):
        raise ValueError(
            "sampling_frequency, slope, carrier_frequency, and pri are required"
        )
    return (
        sampling_frequency,
        slope,
        _positive_float("carrier_frequency", carrier_frequency),
        _positive_float("pri", pri),
    )


def range_doppler_map(
    adc: np.ndarray,
    sampling_frequency: float | None = None,
    slope: float | None = None,
    carrier_frequency: float | None = None,
    pri: float | None = None,
    *,
    fmcw=None,
    num_tx: int | None = None,
    tdm_enabled: bool | None = None,
    win_range: Optional[str] = "hann",
    win_doppler: Optional[str] = "hann",
    remove_mean_clutter: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes range-Doppler maps per virtual channel.

    :param adc: ADC cube with shape
        ``[num_chirps, num_adc_samples, num_channels]``.
    :param sampling_frequency: ADC sampling frequency [Hz].
    :param slope: FMCW slope [Hz/s].
    :param carrier_frequency: Carrier frequency [Hz].
    :param pri: Slow-time interval between samples for one Tx-Rx pair [s].
        If omitted and ``fmcw`` is provided, this defaults to one chirp
        repetition in simultaneous mode and ``num_tx`` chirp repetitions in
        TDM mode.
    :param fmcw: Optional object with FMCW attributes.
    :param num_tx: Number of TDM transmitters. When omitted, use
        ``fmcw.num_tx`` if available, otherwise one.
    :param tdm_enabled: Whether the slow-time samples use a TDM schedule. When
        omitted, use ``fmcw.tdm_enabled`` if available, otherwise false.
    :param remove_mean_clutter: If true, subtract the per-range-bin,
        per-channel slow-time mean from the range-time cube before Doppler
        windowing and FFT.
    :returns: Tuple ``(rd_cube, ranges_m, velocities_mps)`` where ``rd_cube``
        has shape ``[num_doppler_bins, num_range_bins, num_channels]``.
    """
    sampling_frequency, slope, carrier_frequency, pri = (
        _resolve_doppler_parameters(
            sampling_frequency=sampling_frequency,
            slope=slope,
            carrier_frequency=carrier_frequency,
            pri=pri,
            fmcw=fmcw,
            num_tx=num_tx,
            tdm_enabled=tdm_enabled,
        )
    )

    x = np.asarray(adc)
    if x.ndim != 3:
        raise ValueError("adc must have shape [M, N, P]")
    if any(size == 0 for size in x.shape):
        raise ValueError("adc dimensions must be non-empty")

    rt, ranges_m = range_fft(x, sampling_frequency, slope, window=win_range)
    if remove_mean_clutter:
        rt = remove_slow_time_mean(rt)
    wd = window_1d(rt.shape[0], win_doppler)[:, None, None]
    rd = np.fft.fftshift(np.fft.fft(rt * wd, axis=0), axes=0)

    fd_hz = np.fft.fftshift(np.fft.fftfreq(rt.shape[0], d=pri))
    wavelength = SPEED_OF_LIGHT / carrier_frequency
    # The simulator and RT-Pose/TI MATLAB pipeline use the effective
    # tx(t) * conj(rx(t)) ADC convention: positive Doppler phase slope
    # corresponds to increasing bistatic path length.
    velocities_mps = 0.5 * wavelength * fd_hz
    return rd, ranges_m, velocities_mps


def framewise_range_doppler_map(
    adc: np.ndarray,
    sampling_frequency: float | None = None,
    slope: float | None = None,
    carrier_frequency: float | None = None,
    pri: float | None = None,
    *,
    fmcw=None,
    num_tx: int | None = None,
    tdm_enabled: bool | None = None,
    win_range: Optional[str] = "hann",
    win_doppler: Optional[str] = "hann",
    remove_mean_clutter: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes one range-Doppler map per frame.

    :param adc: ADC cube with shape ``[num_frames, num_chirps,
        num_adc_samples, num_channels]``. A single-frame cube with shape
        ``[num_chirps, num_adc_samples, num_channels]`` is accepted and a
        leading frame axis of length one is returned.
    :returns: Tuple ``(rd_frames, ranges_m, velocities_mps)`` where
        ``rd_frames`` has shape ``[num_frames, num_doppler_bins,
        num_range_bins, num_channels]``. The Doppler FFT is never computed
        across frame boundaries.
    """

    x = np.asarray(adc)
    if x.ndim == 3:
        rd, ranges_m, velocities_mps = range_doppler_map(
            x,
            sampling_frequency=sampling_frequency,
            slope=slope,
            carrier_frequency=carrier_frequency,
            pri=pri,
            fmcw=fmcw,
            num_tx=num_tx,
            tdm_enabled=tdm_enabled,
            win_range=win_range,
            win_doppler=win_doppler,
            remove_mean_clutter=remove_mean_clutter,
        )
        return rd[None, ...], ranges_m, velocities_mps
    if x.ndim != 4:
        raise ValueError(
            "adc must have shape [F, M, N, P] or [M, N, P]"
        )
    if x.shape[0] == 0:
        raise ValueError("adc must contain at least one frame")
    rd_frames = []
    ranges_m = None
    velocities_mps = None
    for frame in range(x.shape[0]):
        rd, frame_ranges_m, frame_velocities_mps = range_doppler_map(
            x[frame],
            sampling_frequency=sampling_frequency,
            slope=slope,
            carrier_frequency=carrier_frequency,
            pri=pri,
            fmcw=fmcw,
            num_tx=num_tx,
            tdm_enabled=tdm_enabled,
            win_range=win_range,
            win_doppler=win_doppler,
            remove_mean_clutter=remove_mean_clutter,
        )
        if ranges_m is None:
            ranges_m = frame_ranges_m
            velocities_mps = frame_velocities_mps
        rd_frames.append(rd)
    return np.stack(rd_frames, axis=0), ranges_m, velocities_mps


def doppler_time_spectrum(
    adc: np.ndarray,
    sampling_frequency: float | None = None,
    slope: float | None = None,
    carrier_frequency: float | None = None,
    pri: float | None = None,
    *,
    fmcw=None,
    num_tx: int | None = None,
    tdm_enabled: bool | None = None,
    win_range: Optional[str] = "hann",
    win_doppler: Optional[str] = "hann",
    nfft_mult: int = 1,
    window_chirps: int | None = None,
    hop_chirps: int | None = None,
    doppler_fft_size: int | None = None,
    slow_time_s: np.ndarray | None = None,
    range_limits_m: tuple[float, float] | None = None,
    range_bins: np.ndarray | list[int] | None = None,
    combine: Literal["sum_power", "mean_power"] = "sum_power",
    remove_mean_clutter: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes a Doppler-time power spectrum from ADC data.

    The function first performs a range FFT, selects the requested range bins,
    then applies a sliding Doppler FFT over slow time. Range bins and virtual
    channels are combined after the Doppler FFT.

    :param adc: ADC samples with shape ``[num_chirps, num_adc_samples,
        num_channels]`` or ``[num_frames, num_chirps, num_adc_samples,
        num_channels]``.
    :param sampling_frequency: ADC sampling frequency [Hz].
    :param slope: FMCW slope [Hz/s].
    :param carrier_frequency: Carrier frequency [Hz].
    :param pri: Slow-time interval between chirps for one Tx-Rx pair [s].
    :param fmcw: Optional object with FMCW attributes.
    :param num_tx: Number of TDM transmitters. When omitted, use
        ``fmcw.num_tx`` if available, otherwise one.
    :param tdm_enabled: Whether the slow-time samples use a TDM schedule. When
        omitted, use ``fmcw.tdm_enabled`` if available, otherwise false.
    :param win_range: Fast-time window kind.
    :param win_doppler: Slow-time Doppler window kind.
    :param nfft_mult: Range FFT zero-padding multiplier.
    :param window_chirps: Number of chirps per Doppler snapshot.
    :param hop_chirps: Hop between Doppler snapshots.
    :param doppler_fft_size: Doppler FFT length. Defaults to
        ``window_chirps``.
    :param slow_time_s: Optional slow-time coordinates for the flattened chirps.
        If provided, returned time bins are window-center means from this array.
    :param range_limits_m: Optional inclusive ``(min_m, max_m)`` range gate.
    :param range_bins: Optional explicit range-bin indices or boolean mask.
    :param combine: Range/channel power combination method.
    :param remove_mean_clutter: If true, subtract the per-range-bin,
        per-channel slow-time mean inside each Doppler window before the
        Doppler FFT.
    :returns: Tuple ``(spectrum, times_s, velocities_mps, selected_ranges_m)``
        where ``spectrum`` has shape ``[num_time_bins, num_doppler_bins]``.
    """
    sampling_frequency, slope, carrier_frequency, pri = (
        _resolve_doppler_parameters(
            sampling_frequency=sampling_frequency,
            slope=slope,
            carrier_frequency=carrier_frequency,
            pri=pri,
            fmcw=fmcw,
            num_tx=num_tx,
            tdm_enabled=tdm_enabled,
        )
    )

    x = np.asarray(adc)
    if x.ndim == 4:
        x = x.reshape((-1, x.shape[-2], x.shape[-1]))
    if x.ndim == 2:
        x = x[:, :, None]
    if x.ndim != 3:
        raise ValueError("adc must have shape [M, N], [M, N, P], or"
                         " [F, M, N, P]")

    rt, ranges_m = range_fft(
        x,
        sampling_frequency,
        slope,
        window=win_range,
        nfft_mult=nfft_mult,
    )
    range_selector = _doppler_time_range_selector(
        ranges_m,
        range_limits_m=range_limits_m,
        range_bins=range_bins,
    )
    rt = rt[:, range_selector, :]
    selected_ranges_m = ranges_m[range_selector]

    num_slow = rt.shape[0]
    if window_chirps is None:
        window_chirps = min(num_slow, int(_attr(fmcw, "num_chirps_per_frame")
                                          or 64))
    window_chirps = _positive_integer("window_chirps", window_chirps)
    if window_chirps > num_slow:
        raise ValueError("window_chirps must be in [1, num_chirps]")
    if hop_chirps is None:
        hop_chirps = window_chirps
    hop_chirps = _positive_integer("hop_chirps", hop_chirps)
    if doppler_fft_size is None:
        doppler_fft_size = window_chirps
    doppler_fft_size = _positive_integer("doppler_fft_size", doppler_fft_size)
    if doppler_fft_size < window_chirps:
        raise ValueError("doppler_fft_size must be >= window_chirps")

    if slow_time_s is None:
        slow_time = np.arange(num_slow, dtype=np.float64) * float(pri)
    else:
        slow_time = np.asarray(slow_time_s, dtype=np.float64).reshape(-1)
        if slow_time.shape[0] != num_slow:
            raise ValueError("slow_time_s must match the flattened chirp count")
        if not np.all(np.isfinite(slow_time)):
            raise ValueError("slow_time_s must contain only finite values")

    starts = np.arange(0, num_slow - window_chirps + 1, hop_chirps, dtype=int)
    wd = window_1d(window_chirps, win_doppler)[:, None, None]
    spectrum = np.empty((starts.size, doppler_fft_size), dtype=np.float64)
    times_s = np.empty(starts.size, dtype=np.float64)

    for out_index, start in enumerate(starts):
        stop = int(start + window_chirps)
        rt_window = rt[start:stop]
        if remove_mean_clutter:
            rt_window = remove_slow_time_mean(rt_window)
        doppler_cube = np.fft.fftshift(
            np.fft.fft(rt_window * wd, n=doppler_fft_size, axis=0),
            axes=0,
        )
        power = np.sum(np.abs(doppler_cube) ** 2, axis=(1, 2))
        if combine == "mean_power":
            power = power / float(rt.shape[1] * rt.shape[2])
        elif combine != "sum_power":
            raise ValueError(f"Unknown combine: {combine}")
        spectrum[out_index] = power
        times_s[out_index] = float(np.mean(slow_time[start:stop]))

    fd_hz = np.fft.fftshift(np.fft.fftfreq(doppler_fft_size, d=pri))
    wavelength = SPEED_OF_LIGHT / carrier_frequency
    # The simulator and RT-Pose/TI MATLAB pipeline use the effective
    # tx(t) * conj(rx(t)) ADC convention: positive Doppler phase slope
    # corresponds to increasing bistatic path length.
    velocities_mps = 0.5 * wavelength * fd_hz
    return spectrum, times_s, velocities_mps, selected_ranges_m


def _doppler_time_range_selector(
    ranges_m: np.ndarray,
    *,
    range_limits_m: tuple[float, float] | None,
    range_bins: np.ndarray | list[int] | None,
) -> np.ndarray:
    """Builds the range-bin selector for Doppler-time analysis."""

    if range_limits_m is not None and range_bins is not None:
        raise ValueError("range_limits_m and range_bins are mutually exclusive")
    if range_bins is not None:
        selector = np.asarray(range_bins)
        if selector.dtype == bool and selector.shape != ranges_m.shape:
            raise ValueError("boolean range_bins must match the range axis")
        return selector
    if range_limits_m is None:
        return np.ones(ranges_m.shape, dtype=bool)
    lo, hi = (float(range_limits_m[0]), float(range_limits_m[1]))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("range_limits_m must contain finite values")
    if hi < lo:
        raise ValueError("range_limits_m must be ordered as (min_m, max_m)")
    selector = (ranges_m >= lo) & (ranges_m <= hi)
    if not np.any(selector):
        raise ValueError("range_limits_m selects no range bins")
    return selector
