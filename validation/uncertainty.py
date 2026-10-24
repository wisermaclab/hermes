# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Seed uncertainty in linear power; separate from experimental uncertainty."""

import numpy as np
from scipy.stats import t


def seed_mean_interval(values, confidence=0.95):
    """Two-sided Student-t interval for a mean of independent simulation seeds.

    Intervals are pointwise, assume approximately normal seed means, and do not
    include model bias, measurement uncertainty or a multiple-comparison correction.
    A single deterministic run has no estimated confidence interval.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 0 or values.shape[0] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Seed samples must be nonempty and finite")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie between zero and one")
    count = values.shape[0]
    mean = np.mean(values, axis=0)
    deviation = np.std(values, axis=0, ddof=1) if count > 1 else None
    half = (
        t.ppf((1 + confidence) / 2, count - 1) * deviation / np.sqrt(count) if count > 1 else None
    )
    return {
        "count": count,
        "mean": mean,
        "std": deviation,
        "confidence": confidence,
        "lower": mean - half if half is not None else None,
        "upper": mean + half if half is not None else None,
        "method": "pointwise Student-t across simulation seeds; no measurement/model uncertainty",
    }
