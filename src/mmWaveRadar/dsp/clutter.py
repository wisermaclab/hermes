# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Clutter-removal helpers for DSP cubes."""

from __future__ import annotations

import numpy as np


def remove_slow_time_mean(range_time_cube: np.ndarray) -> np.ndarray:
    """Subtract the slow-time mean for each range bin and channel.

    This mirrors the optional MATLAB ``DopplerProcClutterRemove`` step, where
    the mean across chirps is removed from each range-bin/channel series before
    Doppler windowing and FFT.
    """
    rt = np.asarray(range_time_cube)
    if rt.ndim != 3:
        raise ValueError("range_time_cube must have shape [M, R, P]")
    return rt - np.mean(rt, axis=0, keepdims=True)
