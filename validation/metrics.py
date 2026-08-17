# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Metrics and preprocessing for real-radar validation benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class MeshRangeSummary:
    """Mesh range gate and summary statistics for one or more frames."""

    min_m: np.ndarray
    median_m: np.ndarray
    max_m: np.ndarray

    def to_jsonable(self) -> dict:
        """Converts NumPy range arrays into JSON-compatible lists."""

        return {
            key: np.asarray(value, dtype=float).tolist()
            for key, value in asdict(self).items()
        }


@dataclass(frozen=True)
class DSPMapMetrics:
    """Comparison metrics for one DSP map family."""

    normalized_correlation: float
    peak_range_error_m: float
    reference_gate_energy_ratio: float
    estimate_gate_energy_ratio: float

    def to_jsonable(self) -> dict:
        """Converts scalar map metrics into JSON-compatible numbers."""

        return {
            "normalized_correlation": float(self.normalized_correlation),
            "peak_range_error_m": float(self.peak_range_error_m),
            "reference_gate_energy_ratio": float(
                self.reference_gate_energy_ratio
            ),
            "estimate_gate_energy_ratio": float(self.estimate_gate_energy_ratio),
        }


def background_subtract_adc(
    adc: np.ndarray,
    background_adc: np.ndarray | None = None,
) -> np.ndarray:
    """
    Removes a static ADC background.

    If no explicit background is supplied, the per-chirp/sample/channel mean
    over frames is used. A single-frame clip is returned unchanged because its
    frame mean would remove all signal.
    """

    x = np.asarray(adc)
    if x.ndim != 4:
        raise ValueError("adc must have shape [frames, chirps, samples, channels]")
    if background_adc is None:
        if x.shape[0] <= 1:
            return x.copy()
        background = np.mean(x, axis=0, keepdims=True)
    else:
        background = _broadcast_background(background_adc, x.shape)
    return x - background


def _broadcast_background(background_adc: np.ndarray, adc_shape: tuple[int, ...]):
    background = np.asarray(background_adc)
    if background.shape == adc_shape:
        return background
    if background.shape == adc_shape[1:]:
        return background[None, ...]
    if background.shape == (1, *adc_shape[1:]):
        return background
    raise ValueError(
        "background_adc must have shape [F,M,N,C], [M,N,C], or [1,M,N,C]"
    )


def suppress_zero_doppler(power_map: np.ndarray, *, width_bins: int = 1) -> np.ndarray:
    """Zeros a band around the center Doppler bin in a power map."""

    out = np.asarray(power_map).copy()
    if out.ndim < 2:
        raise ValueError("power_map must include a Doppler axis")
    width = max(0, int(width_bins))
    if width == 0:
        return out
    center = out.shape[-2] // 2
    lo = max(0, center - width)
    hi = min(out.shape[-2], center + width + 1)
    out[..., lo:hi, :] = 0.0
    return out


def zero_doppler_power_fraction(power_map: np.ndarray, *, width_bins: int = 1) -> float:
    """Returns the fraction of power in the center Doppler band."""

    power = np.asarray(power_map, dtype=float)
    if power.ndim < 2:
        raise ValueError("power_map must include a Doppler axis")
    width = max(0, int(width_bins))
    center = power.shape[-2] // 2
    lo = max(0, center - width)
    hi = min(power.shape[-2], center + width + 1)
    total = float(np.sum(power))
    if total <= 0.0:
        return 0.0
    return float(np.sum(power[..., lo:hi, :]) / total)


def normalized_correlation(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Pearson-style normalized correlation for real power maps."""

    ref = np.asarray(reference, dtype=float).ravel()
    est = np.asarray(estimate, dtype=float).ravel()
    if ref.shape != est.shape:
        raise ValueError("reference and estimate must have the same shape")
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    denom = float(np.linalg.norm(ref) * np.linalg.norm(est))
    if denom <= 1e-30:
        return 0.0
    return float(np.vdot(ref, est).real / denom)


def mesh_range_summary(
    mesh_sequence,
    sensor,
    times_s: np.ndarray,
) -> MeshRangeSummary:
    """Computes min, median, and max radar range for mesh vertices per frame."""

    mins = []
    medians = []
    maxs = []
    for time_s in np.asarray(times_s, dtype=float):
        vertices = mesh_sequence.vertices_at(float(time_s))
        local = sensor.world_to_local_points(vertices).reshape((-1, 3))
        ranges = np.linalg.norm(local, axis=1)
        mins.append(float(np.min(ranges)))
        medians.append(float(np.median(ranges)))
        maxs.append(float(np.max(ranges)))
    return MeshRangeSummary(
        min_m=np.asarray(mins, dtype=float),
        median_m=np.asarray(medians, dtype=float),
        max_m=np.asarray(maxs, dtype=float),
    )


def range_gate_mask(
    ranges_m: np.ndarray,
    summary: MeshRangeSummary,
) -> np.ndarray:
    """Builds a single range gate enclosing all mesh-frame range summaries."""

    ranges = np.asarray(ranges_m, dtype=float)
    lo = float(np.min(summary.min_m))
    hi = float(np.max(summary.max_m))
    return (ranges >= lo) & (ranges <= hi)


def energy_ratio_in_range_gate(
    power_map: np.ndarray,
    ranges_m: np.ndarray,
    summary: MeshRangeSummary,
) -> float:
    """Fraction of total map power inside the mesh-derived range gate."""

    power = np.asarray(power_map, dtype=float)
    gate = range_gate_mask(ranges_m, summary)
    if gate.shape[0] != power.shape[-1]:
        raise ValueError("ranges_m length must match the final map axis")
    total = float(np.sum(power))
    if total <= 0.0:
        return 0.0
    return float(np.sum(power[..., gate]) / total)


def peak_range_error_m(
    reference_power: np.ndarray,
    estimate_power: np.ndarray,
    ranges_m: np.ndarray,
    summary: MeshRangeSummary,
) -> float:
    """Distance between dominant range peaks inside the mesh range gate."""

    ranges = np.asarray(ranges_m, dtype=float)
    gate = range_gate_mask(ranges, summary)
    if not np.any(gate):
        return float("nan")
    ref_profile = _range_profile(reference_power)
    est_profile = _range_profile(estimate_power)
    if ref_profile.shape[0] != ranges.shape[0] or est_profile.shape[0] != ranges.shape[0]:
        raise ValueError("map final axis must match ranges_m")
    gate_indices = np.flatnonzero(gate)
    ref_range = ranges[gate_indices[int(np.argmax(ref_profile[gate]))]]
    est_range = ranges[gate_indices[int(np.argmax(est_profile[gate]))]]
    return float(abs(est_range - ref_range))


def _range_profile(power_map: np.ndarray) -> np.ndarray:
    power = np.asarray(power_map, dtype=float)
    if power.ndim == 1:
        return power
    return np.mean(power, axis=tuple(range(power.ndim - 1)))


def dsp_map_metrics(
    reference_power: np.ndarray,
    estimate_power: np.ndarray,
    ranges_m: np.ndarray,
    summary: MeshRangeSummary,
) -> DSPMapMetrics:
    """Computes common DSP-map comparison metrics."""

    return DSPMapMetrics(
        normalized_correlation=normalized_correlation(reference_power, estimate_power),
        peak_range_error_m=peak_range_error_m(
            reference_power,
            estimate_power,
            ranges_m,
            summary,
        ),
        reference_gate_energy_ratio=energy_ratio_in_range_gate(
            reference_power,
            ranges_m,
            summary,
        ),
        estimate_gate_energy_ratio=energy_ratio_in_range_gate(
            estimate_power,
            ranges_m,
            summary,
        ),
    )
