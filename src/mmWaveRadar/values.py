# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Scalar validation shared by waveform, hardware and solver settings."""

from __future__ import annotations

import math
import operator

import numpy as np


def finite_float(name: str, value: object) -> float:
    """Return a finite scalar, reporting the offending setting by name."""
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be a finite real scalar")
    return scalar


def positive_float(name: str, value: object) -> float:
    """Return a finite, strictly positive scalar."""
    scalar = finite_float(name, value)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def nonzero_float(name: str, value: object) -> float:
    """Return a finite nonzero scalar, allowing either sign."""
    scalar = finite_float(name, value)
    if scalar == 0.0:
        raise ValueError(f"{name} must be non-zero")
    return scalar


def positive_integer(name: str, value: object) -> int:
    """Return an integer count without truncating floats or booleans."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(integer)


def boolean(name: str, value: object, *, optional: bool = False) -> bool | None:
    """Validate a boolean without applying truth-value coercion."""
    if value is None and optional:
        return None
    if not isinstance(value, (bool, np.bool_)):
        suffix = " or None" if optional else ""
        raise ValueError(f"{name} must be a boolean{suffix}")
    return bool(value)
