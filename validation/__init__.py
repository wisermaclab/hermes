# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Dataset-agnostic real-radar validation benchmark helpers."""

from .metrics import (
    DSPMapMetrics,
    MeshRangeSummary,
    background_subtract_adc,
    dsp_map_metrics,
    energy_ratio_in_range_gate,
    mesh_range_summary,
    normalized_correlation,
    peak_range_error_m,
    suppress_zero_doppler,
    zero_doppler_power_fraction,
)

__all__ = [
    "DSPMapMetrics",
    "MeshRangeSummary",
    "background_subtract_adc",
    "dsp_map_metrics",
    "energy_ratio_in_range_gate",
    "mesh_range_summary",
    "normalized_correlation",
    "peak_range_error_m",
    "suppress_zero_doppler",
    "zero_doppler_power_fraction",
]
