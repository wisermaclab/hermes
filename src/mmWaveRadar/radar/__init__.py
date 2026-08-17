# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Radar waveform, hardware, and sensor definitions."""

from .fmcw import FMCWConfig
from .hardware import RadarHardware
from .sensor import RadarSensor
from .ti import (
    TIBoardSpec,
    TIDigitizedPattern,
    TIDigitizedPatternUnavailableError,
    VirtualChannelSpec,
    available_ti_boards,
    get_ti_board_spec,
    is_ti_board_model,
    load_ti_digitized_pattern,
    ti_digitized_pattern_asset_available,
)

__all__ = [
    "FMCWConfig",
    "RadarHardware",
    "RadarSensor",
    "TIBoardSpec",
    "TIDigitizedPattern",
    "TIDigitizedPatternUnavailableError",
    "VirtualChannelSpec",
    "available_ti_boards",
    "get_ti_board_spec",
    "is_ti_board_model",
    "load_ti_digitized_pattern",
    "ti_digitized_pattern_asset_available",
]
