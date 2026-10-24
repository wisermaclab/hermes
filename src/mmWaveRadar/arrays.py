# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Array conversion at tensor and NumPy boundaries."""

import numpy as np


def as_numpy(value) -> np.ndarray:
    """Convert NumPy, Dr.Jit and tensor-like values without changing dtype."""
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)
