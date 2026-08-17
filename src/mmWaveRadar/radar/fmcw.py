# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""FMCW radar timing configuration."""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator

import numpy as np


def _finite_float(name: str, value: object, *, allow_negative: bool) -> float:
    """Returns a normalized finite float for one waveform field."""

    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if allow_negative:
        if scalar == 0.0:
            raise ValueError(f"{name} must be non-zero")
    elif scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _positive_integer(name: str, value: object) -> int:
    """Returns an integer count without silently truncating floats or booleans."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(integer)


@dataclass(frozen=True)
class FMCWConfig:
    # pylint: disable=too-many-instance-attributes
    r"""
    FMCW radar waveform and frame timing.

    :param carrier_frequency: Carrier frequency [Hz]
    :param slope: Chirp slope [Hz/s]
    :param chirp_duration: Active ramp duration [s]
    :param chirp_repetition_time: Time between chirp starts [s]
    :param sampling_frequency: ADC sampling frequency [Hz]
    :param num_adc_samples: Number of ADC samples per chirp
    :param num_chirps_per_frame: Number of chirps per frame
    :param frame_period: Time between frame starts [s]
    :param num_tx: Number of transmitters enabled by the chirp schedule
    :param tdm_enabled: Whether transmitters occupy consecutive physical chirp
        slots. When false, all transmitters are sampled simultaneously.
    """

    carrier_frequency: float
    slope: float
    chirp_duration: float
    chirp_repetition_time: float
    sampling_frequency: float
    num_adc_samples: int
    num_chirps_per_frame: int
    frame_period: float
    num_tx: int = 1
    tdm_enabled: bool = False

    def __post_init__(self):
        float_fields = {
            "carrier_frequency": _finite_float(
                "carrier_frequency",
                self.carrier_frequency,
                allow_negative=False,
            ),
            "slope": _finite_float(
                "slope",
                self.slope,
                allow_negative=True,
            ),
            "chirp_duration": _finite_float(
                "chirp_duration",
                self.chirp_duration,
                allow_negative=False,
            ),
            "chirp_repetition_time": _finite_float(
                "chirp_repetition_time",
                self.chirp_repetition_time,
                allow_negative=False,
            ),
            "sampling_frequency": _finite_float(
                "sampling_frequency",
                self.sampling_frequency,
                allow_negative=False,
            ),
            "frame_period": _finite_float(
                "frame_period",
                self.frame_period,
                allow_negative=False,
            ),
        }
        integer_fields = {
            "num_adc_samples": _positive_integer(
                "num_adc_samples",
                self.num_adc_samples,
            ),
            "num_chirps_per_frame": _positive_integer(
                "num_chirps_per_frame",
                self.num_chirps_per_frame,
            ),
            "num_tx": _positive_integer("num_tx", self.num_tx),
        }
        if not isinstance(self.tdm_enabled, (bool, np.bool_)):
            raise ValueError("tdm_enabled must be a boolean")
        object.__setattr__(self, "tdm_enabled", bool(self.tdm_enabled))
        for fields in (float_fields, integer_fields):
            for name, value in fields.items():
                object.__setattr__(self, name, value)

    @property
    def wavelength(self) -> float:
        """Carrier wavelength [m]."""
        return 299792458.0 / self.carrier_frequency

    @property
    def frame_duration(self) -> float:
        """Duration covered by the per-transmitter samples in one frame [s]."""
        return self.num_chirps_per_frame * self.slow_time_interval

    @property
    def slow_time_interval(self) -> float:
        """Time between consecutive samples for one Tx-Rx pair [s]."""

        multiplier = self.num_tx if self.tdm_enabled else 1
        return multiplier * self.chirp_repetition_time

    def chirp_time(self, frame_index: int, chirp_index: int) -> float:
        """Returns a simultaneous chirp or first-Tx slot start time [s]."""
        frame = _nonnegative_index("frame_index", frame_index)
        chirp = _nonnegative_index("chirp_index", chirp_index)
        if chirp >= self.num_chirps_per_frame:
            raise ValueError(
                "chirp_index must be smaller than num_chirps_per_frame"
            )
        return (
            frame * self.frame_period
            + chirp * self.slow_time_interval
        )

    def tx_chirp_time(
        self,
        frame_index: int,
        chirp_index: int,
        tx_index: int,
    ) -> float:
        """Returns the acquisition time for one transmitter sample [s].

        In simultaneous mode every transmitter has the same chirp time. In
        TDM mode transmitter ``tx_index`` occupies its corresponding physical
        chirp slot after the first transmitter.
        """

        tx = _nonnegative_index("tx_index", tx_index)
        if tx >= self.num_tx:
            raise ValueError("tx_index must be smaller than num_tx")
        slot_offset = tx * self.chirp_repetition_time if self.tdm_enabled else 0.0
        return self.chirp_time(frame_index, chirp_index) + slot_offset


def _nonnegative_index(name: str, value: object) -> int:
    """Validates an integer frame/chirp index."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        index = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if index < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(index)
