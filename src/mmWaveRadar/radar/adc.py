# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Receiver sampling modes and ideal anti-alias IF limits."""

import numpy as np

ADC_MODES = ("complex1x", "complex2x", "real")


def adc_mode(config=None, *, default="complex1x"):
    mode = getattr(config, "adc_mode", default)
    if mode not in ADC_MODES:
        raise ValueError(f"adc_mode must be one of {ADC_MODES}")
    return mode


def if_cutoff_hz(config):
    """Usable IF, including the sampling-mode guard band and optional HW cap.

    TI specifies 0.9 fs for Complex 1x and 0.45 fs for Complex 2x/real.
    The simulator models this as an ideal brick-wall receiver filter.
    """
    fs = float(config.sampling_frequency)
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError("sampling_frequency must be finite and positive")
    cutoff = (0.9 if adc_mode(config) == "complex1x" else 0.45) * fs
    bandwidth = getattr(config, "if_bandwidth_hz", None)
    if bandwidth is not None:
        bandwidth = float(bandwidth)
        if not np.isfinite(bandwidth) or bandwidth <= 0.0:
            raise ValueError("if_bandwidth_hz must be finite and positive")
        cutoff = min(cutoff, bandwidth)
    return cutoff


def max_if_delay_s(config):
    return if_cutoff_hz(config) / abs(float(config.slope))


def record_if_filter(diagnostics, config, *, tested=0, rejected=0, partial=0):
    """Accumulate path-channel counts (PO counts refer to original parents)."""
    if diagnostics is None:
        return
    diagnostics.update(
        adc_mode=adc_mode(config), cutoff_hz=if_cutoff_hz(config), model="ideal_brick_wall"
    )
    for key, count in (
        ("tested_path_channels", tested),
        ("rejected_path_channels", rejected),
        ("partially_filtered_path_channels", partial),
    ):
        diagnostics[key] = diagnostics.get(key, 0) + int(count)


def merge_if_filter_diagnostics(**components):
    """Keep component counts so replacing cached coupling never double counts."""
    components = {key: value for key, value in components.items() if value}
    if not components:
        return None
    first = next(iter(components.values()))
    result = {key: first[key] for key in ("adc_mode", "cutoff_hz", "model") if key in first}
    result["components"] = components
    for key in (
        "tested_path_channels",
        "rejected_path_channels",
        "partially_filtered_path_channels",
    ):
        result[key] = sum(value.get(key, 0) for value in components.values())
    return result
