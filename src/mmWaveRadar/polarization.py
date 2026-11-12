# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Explicit orthonormal reference bases for linear polarization.

These bases describe field coordinates, not measured antenna patterns. Real
antenna cross-polarization belongs in the complex antenna response. In
particular, this projected-V convention is not an assertion of Ludwig-3 or
any acquisition system's calibration convention.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

PolarizationCallback = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def _project_transverse(directions, axis):
    """Project an axis using NumPy or Torch arrays with unit directions."""
    return axis - (directions * axis).sum(axis=-1, keepdims=True) * directions


def _unit_vectors(vectors):
    """Normalize finite, nonzero vectors without squaring their original scale."""
    scale = np.max(np.abs(vectors), axis=-1, keepdims=True)
    if np.any(scale == 0.0):
        raise ValueError("directions and vertical_axis must be nonzero")
    scaled = vectors / scale
    return scaled / np.linalg.norm(scaled, axis=-1, keepdims=True)


def _transverse_polarizations(propagation_hats, preferred_axis, *, tolerance=1e-18):
    """Project V, falling back to +Y then +X only for degenerate rows."""
    propagation_hats = np.asarray(propagation_hats, dtype=np.float64)
    preferred_axis = np.asarray(preferred_axis, dtype=np.float64)
    if np.any(preferred_axis):
        preferred_axis = _unit_vectors(preferred_axis)
    polarization = _project_transverse(propagation_hats, preferred_axis)
    norms = np.linalg.norm(polarization, axis=-1, keepdims=True)
    for alt_axis in (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])):
        missing = norms <= tolerance
        if not np.any(missing):
            break
        polarization = np.where(
            missing, _project_transverse(propagation_hats, alt_axis), polarization
        )
        norms = np.linalg.norm(polarization, axis=-1, keepdims=True)
    return polarization / np.maximum(norms, 1e-18)


def transverse_linear_basis(
    propagation_directions: np.ndarray,
    vertical_axis=(0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Return unit ``(V, H)`` fields, with ``H = k cross V``.

    Directions have shape ``[..., 3]`` and point in the physical propagation
    direction. V is the normalized transverse projection of ``vertical_axis``.
    The same convention applies to an arriving wave at the receiver; an
    antenna look direction would have the opposite sign. Reversing k reverses
    H, so endpoint/basis transformations must retain this phase sign.

    For directions parallel to the reference axis (transverse projection norm
    at most 1e-10), V uses the projected +Y axis, or +X if +Y is also parallel.
    This deterministic convention affects only those rows of a batch. Like
    any polar reference basis, it has a phase discontinuity at the poles.
    """
    directions = np.asarray(propagation_directions, dtype=np.float64)
    axis = np.asarray(vertical_axis, dtype=np.float64)
    if directions.ndim == 0 or directions.shape[-1] != 3:
        raise ValueError("propagation_directions must have shape [..., 3]")
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError("vertical_axis must be a finite three-vector")
    if not np.all(np.isfinite(directions)):
        raise ValueError("propagation_directions must be finite")
    k_hat = _unit_vectors(directions)
    axis = _unit_vectors(axis)
    vertical = _transverse_polarizations(k_hat, axis, tolerance=1e-10)
    horizontal = np.cross(k_hat, vertical)
    return vertical, horizontal
