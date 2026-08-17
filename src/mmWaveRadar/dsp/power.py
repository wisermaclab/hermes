# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Power-to-dB conversion helpers."""

from __future__ import annotations

import numpy as np


def power_to_db(power, *, floor_power: float = 1e-30):
    """Converts linear power to dB without peak normalization."""

    return 10.0 * np.log10(
        np.maximum(np.asarray(power, dtype=float), float(floor_power))
    )


def power_to_relative_db(
    power,
    *,
    reference_power=None,
    floor_db: float = -80.0,
    reference_floor_power: float = 1e-30,
):
    """Converts linear power to dB relative to a reference peak."""

    values = np.asarray(power, dtype=float)
    reference = values if reference_power is None else np.asarray(
        reference_power,
        dtype=float,
    )
    reference_peak = max(float(np.max(reference)), float(reference_floor_power))
    floor_power = reference_peak * 10.0 ** (float(floor_db) / 10.0)
    return 10.0 * np.log10(np.maximum(values, floor_power) / reference_peak)
