# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Reductions of resolved path coefficients."""

import numpy as np


def coherent_channel(coefficients: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Sum valid path coefficients coherently for each virtual channel."""
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    valid = np.asarray(valid, dtype=bool)
    if coefficients.ndim != 2:
        raise ValueError("coefficients must have shape [num_virtual_channels, num_paths]")
    if valid.shape != (coefficients.shape[1],):
        raise ValueError("valid must have shape [num_paths]")
    return np.sum(coefficients[:, valid], axis=1)
