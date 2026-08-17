# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""CFAR detection helpers."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import numpy as np


def ca_cfar_1d(
    x: np.ndarray,
    *,
    num_train: int = 16,
    num_guard: int = 4,
    pfa: float = 1e-3,
    scale: Optional[float] = None,
    mode: Literal["power", "magnitude"] = "power",
    peak_only: bool = True,
    min_sep: int = 1,
) -> Dict[str, Any]:
    """
    Runs 1D cell-averaging CFAR on a nonnegative sequence.

    ``mode="power"`` treats ``x`` as power. ``mode="magnitude"`` squares the
    input before estimating exponential noise power, then reports the noise
    estimate and threshold back in magnitude units.

    :returns: Dict with ``det_mask``, ``threshold``, ``noise_est``, and
        ``peaks``.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values")
    if np.any(x < 0):
        raise ValueError("x must be nonnegative")
    if mode not in ("power", "magnitude"):
        raise ValueError("mode must be 'power' or 'magnitude'")

    r = x.shape[0]
    integer_parameters = (
        ("num_train", num_train, 1),
        ("num_guard", num_guard, 0),
        ("min_sep", min_sep, 1),
    )
    for name, value, minimum in integer_parameters:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer >= {minimum}")
        if int(value) < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
    num_train = int(num_train)
    num_guard = int(num_guard)
    min_sep = int(min_sep)

    try:
        pfa = float(pfa)
    except (TypeError, ValueError) as exc:
        raise ValueError("pfa must be a real scalar between 0 and 1") from exc
    if not np.isfinite(pfa) or not 0.0 < pfa < 1.0:
        raise ValueError("pfa must be between 0 and 1")
    if scale is not None:
        try:
            scale = float(scale)
        except (TypeError, ValueError) as exc:
            raise ValueError("scale must be a finite positive scalar") from exc
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be a finite positive scalar")

    if mode == "magnitude":
        with np.errstate(over="ignore", invalid="ignore"):
            cfar_input = np.square(x)
        if not np.all(np.isfinite(cfar_input)):
            raise ValueError("x is too large to convert to power")
    else:
        cfar_input = x

    num_cells = 2 * num_train
    alpha = (num_cells * (pfa ** (-1.0 / num_cells) - 1.0)
             if scale is None else scale)
    pad = num_guard + num_train

    det = np.zeros(r, dtype=bool)
    thr = np.full(r, np.nan, dtype=float)
    noise = np.full(r, np.nan, dtype=float)

    for i in range(pad, r - pad):
        left = cfar_input[i - pad: i - num_guard]
        right = cfar_input[i + num_guard + 1: i + pad + 1]
        z = np.mean(np.concatenate([left, right]))
        noise[i] = z
        thr[i] = alpha * z
        det[i] = cfar_input[i] > thr[i]

    if mode == "magnitude":
        noise = np.sqrt(noise)
        thr = np.sqrt(thr)

    idx = np.where(det)[0]
    if not peak_only:
        return {"det_mask": det, "threshold": thr, "noise_est": noise,
                "peaks": idx}

    peaks = []
    for i in idx:
        if i <= 0 or i >= r - 1:
            continue
        if x[i] >= x[i - 1] and x[i] >= x[i + 1]:
            peaks.append(i)
    peaks = np.asarray(peaks, dtype=int)

    if peaks.size > 1 and min_sep > 1:
        order = np.argsort(x[peaks])[::-1]
        chosen = []
        taken = np.zeros(r, dtype=bool)
        for k in order:
            p = peaks[k]
            if taken[p]:
                continue
            chosen.append(p)
            lo = max(0, p - min_sep)
            hi = min(r, p + min_sep + 1)
            taken[lo:hi] = True
        peaks = np.asarray(sorted(chosen), dtype=int)

    return {"det_mask": det, "threshold": thr, "noise_est": noise,
            "peaks": peaks}
