# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Window helpers for radar DSP."""

from __future__ import annotations

from typing import Optional

import numpy as np


def window_1d(n: int, kind: Optional[str]) -> np.ndarray:
    """
    Returns a 1D window of length ``n``.

    Supported window kinds are ``None``, ``"none"``, ``"hann"``,
    ``"hamming"``, and ``"blackman"``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if kind is None or str(kind).lower() in ("none", ""):
        return np.ones(n, dtype=float)
    k = str(kind).lower()
    if k in ("hann", "hanning"):
        return np.hanning(n)
    if k == "hamming":
        return np.hamming(n)
    if k == "blackman":
        return np.blackman(n)
    raise ValueError(f"Unsupported window: {kind}")
