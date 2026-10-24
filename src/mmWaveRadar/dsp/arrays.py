# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Array shapes and waveform attribute access used by DSP operations."""

from __future__ import annotations

import numpy as np

_MISSING = object()


def fmcw_value(fmcw, name: str, fallback=_MISSING):
    """Read an optional waveform; require its attribute unless a fallback is given."""
    if fmcw is None:
        return None if fallback is _MISSING else fallback
    return getattr(fmcw, name) if fallback is _MISSING else getattr(fmcw, name, fallback)


def snapshot_matrix(snapshots: np.ndarray, num_channels: int | None = None) -> np.ndarray:
    """Flatten leading axes, retaining the last axis as virtual channels.

    Covariance estimation requires at least two channels and one snapshot.
    Imaging supplies an explicit channel count and can represent an empty image.
    """
    x = np.asarray(snapshots, dtype=np.complex128)
    if x.ndim == 0:
        raise ValueError("snapshots must include a virtual-channel axis")
    channels = x.shape[-1]
    if num_channels is not None:
        if channels != num_channels:
            raise ValueError("snapshots last axis must match virtual positions")
    elif channels < 2:
        raise ValueError("snapshots must include at least two virtual channels")
    x = x.reshape((-1, channels))
    if num_channels is None and x.shape[0] < 1:
        raise ValueError("snapshots must include at least one snapshot")
    return x


def flatten_steering(steering: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Flatten steering-grid axes into columns and retain the grid shape."""
    return steering.reshape((steering.shape[0], -1)), steering.shape[1:]
