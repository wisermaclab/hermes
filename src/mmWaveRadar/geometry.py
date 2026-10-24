# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Coordinate transforms shared by radar poses, targets and diagnostics."""

import numpy as np


def rotation_matrix(
    orientation: tuple[float, float, float], *, degrees: bool = False
) -> np.ndarray:
    """Returns the world rotation matrix for a radar pose."""

    angles = np.asarray(orientation, dtype=float)
    if angles.shape != (3,) or not np.all(np.isfinite(angles)):
        raise ValueError("orientation angles must be three finite values")
    a, b, c = np.deg2rad(angles) if degrees else angles
    sin_a, cos_a = np.sin(a), np.cos(a)
    sin_b, cos_b = np.sin(b), np.cos(b)
    sin_c, cos_c = np.sin(c), np.cos(c)
    return np.array(
        [
            [
                cos_a * cos_b,
                cos_a * sin_b * sin_c - sin_a * cos_c,
                cos_a * sin_b * cos_c + sin_a * sin_c,
            ],
            [
                sin_a * cos_b,
                sin_a * sin_b * sin_c + cos_a * cos_c,
                sin_a * sin_b * cos_c - cos_a * sin_c,
            ],
            [-sin_b, cos_b * sin_c, cos_b * cos_c],
        ],
        dtype=float,
    )
