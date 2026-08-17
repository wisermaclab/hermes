# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Calibration helpers for aligning custom PO amplitudes to RT references."""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Literal

import numpy as np


POCalibrationMode = Literal["none", "rt_first_pose", "rt_sequence"]
POCalibrationGainType = Literal["amplitude"]
POCalibrationSampleMode = Literal[
    "first_frame", "all_chirps", "retrace_chirps", "retraced_chirps"
]
POCalibrationReference = Literal[
    "single_human_only",
    "one_human_touch",
    "human_touch",
    "human_env_coupled",
    "human_multi_touch",
]
POCalibrationAlignment = Literal["last_retrace", "nearest", "same_index"]


def _boolean(name: str, value: object, *, optional: bool = False) -> bool | None:
    """Validates a boolean option without applying truth-value coercion."""

    if value is None and optional:
        return None
    if not isinstance(value, (bool, np.bool_)):
        suffix = " or None" if optional else ""
        raise ValueError(f"{name} must be a boolean{suffix}")
    return bool(value)


def _nonnegative_float(name: str, value: object) -> float:
    """Returns a finite, non-negative scalar."""

    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return scalar


def _optional_positive_integer(name: str, value: object) -> int | None:
    """Returns an optional positive integer without silent truncation."""

    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer or None")
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer or None") from exc
    if integer <= 0:
        raise ValueError(f"{name} must be a positive integer or None")
    return int(integer)


@dataclass(frozen=True)
class POCalibrationConfig:
    """Calibration policy for custom human PO amplitudes.

    The simulator's PO path coefficients are deterministic but may not share
    the same absolute amplitude scale as an RT baseline.  This policy controls
    whether a scalar or per-chirp amplitude gain is fitted from RT reference
    power and then applied to PO human-path coefficients.

    Attributes:
        mode: ``"none"`` disables fitting, ``"rt_first_pose"`` uses the first
            RT/PO ADC frame, and ``"rt_sequence"`` fits from path-power samples
            selected across the available sequence.
        gain_type: Gain domain. Only amplitude gains are currently supported.
        reference_power_floor: Minimum RT/PO power allowed when forming a gain;
            samples at or below the floor are ignored or reported as warnings.
        keep_rt_baseline: Whether higher-level pipelines should retain the RT
            baseline cube/metadata after calibration.
        rt_periodic_retrace: Enables RT reference retracing on a periodic
            schedule. ``rt_retrace_once_per_frame`` is the legacy alias.
        rt_periodic_retrace_period_chirps: Optional chirp period for retraces.
        sample_mode: Which PO chirps contribute to sequence fitting.
        reference_component: RT path-power component used as the reference.
        alignment: How selected PO chirps are matched to RT reference chirps.
    """

    mode: POCalibrationMode = "none"
    gain_type: POCalibrationGainType = "amplitude"
    reference_power_floor: float = 1e-30
    keep_rt_baseline: bool = True
    rt_periodic_retrace: bool | None = None
    rt_periodic_retrace_period_chirps: int | None = None
    rt_retrace_once_per_frame: bool | None = None
    sample_mode: POCalibrationSampleMode = "retrace_chirps"
    reference_component: POCalibrationReference = "single_human_only"
    alignment: POCalibrationAlignment = "last_retrace"

    def __post_init__(self):
        if self.mode not in ("none", "rt_first_pose", "rt_sequence"):
            raise ValueError(
                "PO calibration mode must be 'none', 'rt_first_pose', "
                "or 'rt_sequence'"
            )
        if self.gain_type != "amplitude":
            raise ValueError("Only amplitude PO calibration is currently supported")
        reference_power_floor = _nonnegative_float(
            "reference_power_floor",
            self.reference_power_floor,
        )
        keep_rt_baseline = _boolean("keep_rt_baseline", self.keep_rt_baseline)
        rt_periodic_retrace = _boolean(
            "rt_periodic_retrace",
            self.rt_periodic_retrace,
            optional=True,
        )
        legacy_retrace = _boolean(
            "rt_retrace_once_per_frame",
            self.rt_retrace_once_per_frame,
            optional=True,
        )
        retrace_period = _optional_positive_integer(
            "rt_periodic_retrace_period_chirps",
            self.rt_periodic_retrace_period_chirps,
        )
        if rt_periodic_retrace is None:
            rt_periodic_retrace = (
                True
                if legacy_retrace is None
                else legacy_retrace
            )
        elif legacy_retrace is not None and legacy_retrace != rt_periodic_retrace:
            raise ValueError(
                "rt_periodic_retrace and legacy rt_retrace_once_per_frame disagree"
            )
        object.__setattr__(self, "reference_power_floor", reference_power_floor)
        object.__setattr__(self, "keep_rt_baseline", keep_rt_baseline)
        object.__setattr__(self, "rt_periodic_retrace", rt_periodic_retrace)
        object.__setattr__(self, "rt_retrace_once_per_frame", rt_periodic_retrace)
        object.__setattr__(
            self,
            "rt_periodic_retrace_period_chirps",
            retrace_period,
        )
        if self.sample_mode == "retraced_chirps":
            object.__setattr__(self, "sample_mode", "retrace_chirps")
        if self.sample_mode not in ("first_frame", "all_chirps", "retrace_chirps"):
            raise ValueError(
                "sample_mode must be 'first_frame', 'all_chirps', "
                "'retrace_chirps', or 'retraced_chirps'"
            )
        if self.reference_component not in (
            "single_human_only",
            "one_human_touch",
            "human_touch",
            "human_env_coupled",
            "human_multi_touch",
        ):
            raise ValueError(
                "reference_component must be one of: single_human_only, "
                "one_human_touch, human_touch, human_env_coupled, "
                "human_multi_touch"
            )
        if self.alignment not in ("last_retrace", "nearest", "same_index"):
            raise ValueError(
                "alignment must be 'last_retrace', 'nearest', or 'same_index'"
            )


def rt_calibration_enabled(config: POCalibrationConfig) -> bool:
    """Returns whether ``config`` requires an RT baseline to be generated."""

    return config.mode in ("rt_first_pose", "rt_sequence")


def adc_rms_power(adc: np.ndarray) -> float:
    """Returns mean complex-sample power for an ADC tensor.

    Empty tensors report ``0.0`` so callers can safely feed optional reference
    cubes into calibration without special-casing missing samples first.
    """

    adc = np.asarray(adc, dtype=np.complex128)
    if adc.size == 0:
        return 0.0
    return float(np.mean(np.abs(adc) ** 2))


def fit_amplitude_gain_from_powers(
    *,
    rt_power: float,
    po_power: float,
    config: POCalibrationConfig,
    metadata: dict | None = None,
) -> tuple[float, dict]:
    """Fits one amplitude gain from RT and raw-PO reference powers.

    The fitted gain is ``sqrt(rt_power / po_power)`` when both powers are above
    ``config.reference_power_floor``. The metadata dictionary records the raw
    and calibrated powers, gain in linear and dB units, whether the gain was
    applied, and warning strings for skipped or non-finite references.
    """

    rt_power = float(rt_power)
    po_power = float(po_power)
    floor = float(config.reference_power_floor)
    warnings = []
    gain = 1.0
    applied = False
    if rt_power <= floor:
        warnings.append("rt_reference_power_below_floor")
    elif po_power <= floor:
        warnings.append("po_reference_power_below_floor")
    else:
        gain = float(np.sqrt(rt_power / po_power))
        applied = bool(np.isfinite(gain) and gain >= 0.0)
        if not applied:
            gain = 1.0
            warnings.append("nonfinite_calibration_gain")
    calibrated_power = po_power * gain * gain
    gain_db = 20.0 * np.log10(max(gain, 1e-300))
    out = {
        "mode": config.mode,
        "gain_type": config.gain_type,
        "applied": applied,
        "human_po_amplitude_gain": float(gain),
        "human_po_amplitude_gain_db": float(gain_db),
        "rt_reference_rms_power": rt_power,
        "raw_po_reference_rms_power": po_power,
        "calibrated_po_reference_rms_power": float(calibrated_power),
        "reference_power_floor": floor,
        "warnings": warnings,
    }
    if metadata:
        out.update(metadata)
    return gain, out


def fit_first_pose_adc_calibration(
    *,
    rt_reference_adc: np.ndarray,
    po_reference_adc: np.ndarray,
    config: POCalibrationConfig,
) -> tuple[float, dict]:
    """Fits the legacy first-frame ADC RMS calibration gain.

    This path is retained for callers that calibrate directly from synthesized
    ADC tensors instead of path-power samples.  It uses the full ADC RMS power
    of each reference tensor and reports metadata compatible with sequence
    calibration outputs.
    """

    return fit_amplitude_gain_from_powers(
        rt_power=adc_rms_power(rt_reference_adc),
        po_power=adc_rms_power(po_reference_adc),
        config=config,
        metadata={
            "reference_frame": 0,
            "sample_mode": "first_frame",
            "alignment": "same_index",
            "reference_component": "single_human_only_adc",
            "reference_power_kind": "adc_rms_power",
        },
    )


def po_path_power_from_coefficients(
    coefficients: np.ndarray,
    valid: np.ndarray | None = None,
) -> float:
    """Sums PO coefficient power over valid face paths.

    ``coefficients`` is expected to have virtual channels on axis 0 and paths on
    axis 1.  The function first sums channel power per path, then applies the
    optional path mask.  NaN and infinite powers are treated as zero.
    """

    coefficients = np.asarray(coefficients, dtype=np.complex128)
    if coefficients.size == 0:
        return 0.0
    path_power = np.sum(np.abs(coefficients) ** 2, axis=0)
    path_power = np.nan_to_num(path_power, nan=0.0, posinf=0.0, neginf=0.0)
    if valid is None:
        valid = path_power > 0.0
    return float(np.sum(path_power[np.asarray(valid, dtype=bool)]))


def rt_reference_power_array(rt_baseline: dict, component: str) -> np.ndarray | None:
    """Returns the RT path-power array used by a sequence calibration.

    The lookup first checks the baseline dictionary, then falls back to matching
    metadata attributes on ``rt_baseline["cube"]``. ``None`` means the requested
    reference component was not recorded by the RT run.
    """

    key = {
        "single_human_only": "single_human_only_path_power",
        "one_human_touch": "one_human_touch_path_power",
        "human_touch": "human_touch_path_power",
        "human_env_coupled": "human_env_coupled_path_power",
        "human_multi_touch": "human_multi_touch_path_power",
    }[component]
    value = rt_baseline.get(key)
    if value is None:
        cube = rt_baseline.get("cube")
        metadata = None if cube is None else getattr(cube, "metadata", None)
        value = None if metadata is None else getattr(metadata, key, None)
    return None if value is None else np.asarray(value, dtype=np.float64)


def _flatten_times(times: np.ndarray) -> np.ndarray:
    """Returns frame/chirp times as a one-dimensional float array."""

    return np.asarray(times, dtype=float).reshape(-1)


def _rt_events(rt_baseline: dict) -> list[dict]:
    """Extracts RT retrace events that include frame and chirp indices."""

    cube = rt_baseline.get("cube")
    metadata = None if cube is None else getattr(cube, "metadata", None)
    events = [] if metadata is None else list(getattr(metadata, "events", []) or [])
    return [event for event in events if "frame" in event and "chirp" in event]


def _event_indices(rt_baseline: dict, rt_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Returns valid RT event frame/chirp indices, defaulting to the first chirp."""

    events = _rt_events(rt_baseline)
    pairs = []
    for event in events:
        frame = int(event.get("frame", -1))
        chirp = int(event.get("chirp", -1))
        if 0 <= frame < rt_shape[0] and 0 <= chirp < rt_shape[1]:
            pairs.append((frame, chirp))
    if not pairs:
        pairs = [(0, 0)]
    frame_idx, chirp_idx = np.asarray(pairs, dtype=np.int64).T
    return frame_idx, chirp_idx


def _selected_po_indices(
    *,
    rt_baseline: dict,
    rt_shape: tuple[int, int],
    rt_times: np.ndarray,
    po_times: np.ndarray,
    sample_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Selects PO frame/chirp samples according to the calibration sample mode."""

    if sample_mode == "first_frame":
        po_frame_idx = np.zeros(po_times.shape[1], dtype=np.int64)
        po_chirp_idx = np.arange(po_times.shape[1], dtype=np.int64)
    elif sample_mode == "retrace_chirps":
        event_frames, event_chirps = _event_indices(rt_baseline, rt_shape)
        event_times = rt_times[event_frames, event_chirps]
        po_flat_times = _flatten_times(po_times)
        nearest_po = np.searchsorted(po_flat_times, event_times)
        nearest_po = np.clip(nearest_po, 0, po_flat_times.size - 1)
        prev_po = np.maximum(nearest_po - 1, 0)
        use_prev = (
            np.abs(po_flat_times[prev_po] - event_times)
            <= np.abs(po_flat_times[nearest_po] - event_times)
        )
        nearest_po = np.where(use_prev, prev_po, nearest_po)
        po_frame_idx, po_chirp_idx = np.unravel_index(nearest_po, po_times.shape)
    else:
        po_frame_idx, po_chirp_idx = np.indices(po_times.shape)
        po_frame_idx = po_frame_idx.reshape(-1)
        po_chirp_idx = po_chirp_idx.reshape(-1)
    return po_frame_idx.astype(np.int64), po_chirp_idx.astype(np.int64)


def _aligned_rt_indices(
    *,
    rt_baseline: dict,
    rt_times: np.ndarray,
    po_times: np.ndarray,
    selected_po_times: np.ndarray,
    po_frame_idx: np.ndarray,
    po_chirp_idx: np.ndarray,
    alignment: str,
    warnings: list[str],
) -> tuple[np.ndarray, np.ndarray, str]:
    """Maps selected PO samples to RT reference samples using an alignment mode."""

    if alignment == "same_index":
        if po_times.shape != rt_times.shape:
            warnings.append("same_index_shape_mismatch_falling_back_to_nearest")
            alignment = "nearest"
        else:
            return po_frame_idx, po_chirp_idx, "same_index"

    if alignment == "nearest":
        rt_flat_times = _flatten_times(rt_times)
        nearest_rt = np.searchsorted(rt_flat_times, selected_po_times)
        nearest_rt = np.clip(nearest_rt, 0, rt_flat_times.size - 1)
        prev_rt = np.maximum(nearest_rt - 1, 0)
        use_prev = (
            np.abs(rt_flat_times[prev_rt] - selected_po_times)
            <= np.abs(rt_flat_times[nearest_rt] - selected_po_times)
        )
        nearest_rt = np.where(use_prev, prev_rt, nearest_rt)
        return (*np.unravel_index(nearest_rt, rt_times.shape), "nearest")

    if alignment == "last_retrace":
        event_frames, event_chirps = _event_indices(rt_baseline, rt_times.shape)
        event_times = rt_times[event_frames, event_chirps]
        order = np.argsort(event_times)
        event_times = event_times[order]
        event_frames = event_frames[order]
        event_chirps = event_chirps[order]
        event_pos = np.searchsorted(event_times, selected_po_times, side="right") - 1
        if np.any(event_pos < 0):
            warnings.append("po_time_before_first_rt_retrace_clamped")
        event_pos = np.clip(event_pos, 0, event_times.size - 1)
        return event_frames[event_pos], event_chirps[event_pos], "last_retrace"

    raise ValueError(f"unsupported calibration alignment: {alignment!r}")


def _sequence_power_samples(
    *,
    rt_baseline: dict,
    po_times: np.ndarray,
    po_path_power: np.ndarray,
    config: POCalibrationConfig,
) -> tuple[dict[str, np.ndarray], dict]:
    """Builds aligned finite RT/PO power samples for gain fitting."""

    cube = rt_baseline.get("cube")
    if cube is None:
        return {}, {"warnings": ["rt_baseline_cube_missing"]}
    rt_times = np.asarray(cube.times, dtype=float)
    rt_power = rt_reference_power_array(rt_baseline, config.reference_component)
    if rt_power is None:
        return {}, {"warnings": ["rt_reference_power_missing"]}
    po_times = np.asarray(po_times, dtype=float)
    po_path_power = np.asarray(po_path_power, dtype=np.float64)
    if po_times.shape != po_path_power.shape:
        raise ValueError("po_times and po_path_power must have the same shape")
    if rt_times.shape != rt_power.shape:
        raise ValueError("RT times and RT reference power must have the same shape")

    warnings: list[str] = []
    po_frame_idx, po_chirp_idx = _selected_po_indices(
        rt_baseline=rt_baseline,
        rt_shape=rt_times.shape,
        rt_times=rt_times,
        po_times=po_times,
        sample_mode=config.sample_mode,
    )
    selected_po_times = po_times[po_frame_idx, po_chirp_idx]
    selected_po_power = po_path_power[po_frame_idx, po_chirp_idx]
    rt_frame_idx, rt_chirp_idx, effective_alignment = _aligned_rt_indices(
        rt_baseline=rt_baseline,
        rt_times=rt_times,
        po_times=po_times,
        selected_po_times=selected_po_times,
        po_frame_idx=po_frame_idx,
        po_chirp_idx=po_chirp_idx,
        alignment=config.alignment,
        warnings=warnings,
    )
    selected_rt_power = rt_power[rt_frame_idx, rt_chirp_idx]

    valid = np.isfinite(selected_rt_power) & np.isfinite(selected_po_power)
    valid &= selected_rt_power >= 0.0
    valid &= selected_po_power >= 0.0
    if not np.all(valid):
        warnings.append("nonfinite_or_negative_power_samples_dropped")

    samples = {
        "rt_power": selected_rt_power[valid],
        "po_power": selected_po_power[valid],
        "po_time": selected_po_times[valid],
        "rt_time": rt_times[rt_frame_idx, rt_chirp_idx][valid],
        "po_frame_idx": po_frame_idx[valid],
        "po_chirp_idx": po_chirp_idx[valid],
        "rt_frame_idx": rt_frame_idx[valid],
        "rt_chirp_idx": rt_chirp_idx[valid],
        "po_times_full": po_times,
    }
    meta = {
        "warnings": warnings,
        "reference_chirps": int(samples["rt_power"].size),
        "reference_time_start_s": (
            None if samples["po_time"].size == 0 else float(np.min(samples["po_time"]))
        ),
        "reference_time_stop_s": (
            None if samples["po_time"].size == 0 else float(np.max(samples["po_time"]))
        ),
        "rt_reference_time_start_s": (
            None if samples["rt_time"].size == 0 else float(np.min(samples["rt_time"]))
        ),
        "rt_reference_time_stop_s": (
            None if samples["rt_time"].size == 0 else float(np.max(samples["rt_time"]))
        ),
        "alignment": config.alignment,
        "effective_alignment": effective_alignment,
        "sample_mode": config.sample_mode,
        "reference_component": config.reference_component,
        "reference_power_kind": "path_coefficient_power",
    }
    return samples, meta


def aligned_sequence_powers(
    *,
    rt_baseline: dict,
    po_times: np.ndarray,
    po_path_power: np.ndarray,
    config: POCalibrationConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Aligns raw PO path powers to an RT path-power reference.

    The returned arrays contain only finite, non-negative matched samples.  The
    metadata describes the effective alignment, selected time span, sample
    count, and any warnings emitted while mapping PO chirps to RT retraces.
    """

    samples, meta = _sequence_power_samples(
        rt_baseline=rt_baseline,
        po_times=po_times,
        po_path_power=po_path_power,
        config=config,
    )
    return (
        np.asarray(samples.get("rt_power", []), dtype=float),
        np.asarray(samples.get("po_power", []), dtype=float),
        meta,
    )


def _time_varying_gain_map(
    *,
    samples: dict[str, np.ndarray],
    scalar_gain: float,
    config: POCalibrationConfig,
) -> tuple[np.ndarray, dict, list[str]]:
    """Builds either a scalar or last-retrace time-varying PO gain map."""

    po_times = np.asarray(samples["po_times_full"], dtype=float)
    gain_map = np.full(po_times.shape, float(scalar_gain), dtype=np.float64)
    warnings: list[str] = []
    if config.sample_mode != "retrace_chirps" or config.alignment != "last_retrace":
        return gain_map, {"calibration_gain_strategy": "global_mean"}, warnings

    rt_power = np.asarray(samples["rt_power"], dtype=np.float64)
    po_power = np.asarray(samples["po_power"], dtype=np.float64)
    floor = float(config.reference_power_floor)
    valid_gain = np.isfinite(rt_power) & np.isfinite(po_power)
    valid_gain &= rt_power > floor
    valid_gain &= po_power > floor
    if not np.any(valid_gain):
        warnings.append("no_valid_time_varying_gain_samples")
        return gain_map, {"calibration_gain_strategy": "global_mean_fallback"}, warnings

    sample_times = np.asarray(samples["rt_time"], dtype=float)[valid_gain]
    sample_gains = np.sqrt(rt_power[valid_gain] / po_power[valid_gain])
    order = np.argsort(sample_times)
    sample_times = sample_times[order]
    sample_gains = sample_gains[order]
    unique_times, last_indices = np.unique(sample_times, return_index=True)
    if unique_times.size != sample_times.size:
        # Keep the last gain for duplicate retrace timestamps.
        _, reverse_last = np.unique(sample_times[::-1], return_index=True)
        last_indices = sample_times.size - 1 - reverse_last
        order = np.argsort(sample_times[last_indices])
        last_indices = last_indices[order]
        unique_times = sample_times[last_indices]
    sample_gains = sample_gains[last_indices]

    po_flat_times = _flatten_times(po_times)
    sample_pos = np.searchsorted(unique_times, po_flat_times, side="right") - 1
    if np.any(sample_pos < 0):
        warnings.append("po_time_before_first_gain_sample_clamped")
    sample_pos = np.clip(sample_pos, 0, unique_times.size - 1)
    gain_map = sample_gains[sample_pos].reshape(po_times.shape)
    gain_db = 20.0 * np.log10(np.maximum(sample_gains, 1e-300))
    map_db = 20.0 * np.log10(np.maximum(gain_map, 1e-300))
    meta = {
        "calibration_gain_strategy": "time_varying_last_retrace",
        "gain_schedule_chirps": int(sample_gains.size),
        "gain_schedule_time_start_s": float(unique_times[0]),
        "gain_schedule_time_stop_s": float(unique_times[-1]),
        "human_po_amplitude_gain_schedule_times_s": unique_times,
        "human_po_amplitude_gain_schedule": sample_gains,
        "human_po_amplitude_gain_map": gain_map,
        "human_po_amplitude_gain_map_min": float(np.min(gain_map)),
        "human_po_amplitude_gain_map_mean": float(np.mean(gain_map)),
        "human_po_amplitude_gain_map_max": float(np.max(gain_map)),
        "human_po_amplitude_gain_map_min_db": float(np.min(map_db)),
        "human_po_amplitude_gain_map_mean_db": float(20.0 * np.log10(max(float(np.mean(gain_map)), 1e-300))),
        "human_po_amplitude_gain_map_max_db": float(np.max(map_db)),
        "human_po_amplitude_gain_schedule_min_db": float(np.min(gain_db)),
        "human_po_amplitude_gain_schedule_max_db": float(np.max(gain_db)),
        "human_po_amplitude_gain_schedule_range_db": float(np.max(gain_db) - np.min(gain_db)),
    }
    return gain_map, meta, warnings


def fit_sequence_path_power_calibration(
    *,
    rt_baseline: dict,
    po_times: np.ndarray,
    po_path_power: np.ndarray,
    config: POCalibrationConfig,
) -> tuple[float, dict]:
    """Fits PO amplitude gains from an aligned RT path-power sequence.

    A scalar gain is always returned for compatibility with older call sites.
    When ``sample_mode="retrace_chirps"`` and ``alignment="last_retrace"``, the
    metadata can also include ``human_po_amplitude_gain_map`` for per-chirp
    calibration between RT retrace events.
    """

    samples, meta = _sequence_power_samples(
        rt_baseline=rt_baseline,
        po_times=po_times,
        po_path_power=po_path_power,
        config=config,
    )
    warnings = list(meta.pop("warnings", []))
    rt_values = np.asarray(samples.get("rt_power", []), dtype=float)
    po_values = np.asarray(samples.get("po_power", []), dtype=float)
    if rt_values.size == 0 or po_values.size == 0:
        warnings.append("no_calibration_power_samples")
        gain, out = fit_amplitude_gain_from_powers(
            rt_power=0.0,
            po_power=0.0,
            config=config,
            metadata=meta,
        )
        out["warnings"] = warnings + out.get("warnings", [])
        return gain, out

    rt_integrated = float(np.sum(rt_values))
    po_integrated = float(np.sum(po_values))
    rt_mean = float(np.mean(rt_values))
    po_mean = float(np.mean(po_values))
    gain, out = fit_amplitude_gain_from_powers(
        rt_power=rt_mean,
        po_power=po_mean,
        config=config,
        metadata=meta,
    )
    gain_map, gain_meta, gain_warnings = _time_varying_gain_map(
        samples=samples,
        scalar_gain=gain,
        config=config,
    )
    warnings.extend(gain_warnings)
    out["warnings"] = warnings + out.get("warnings", [])
    out["rt_reference_integrated_power"] = rt_integrated
    out["raw_po_reference_integrated_power"] = po_integrated
    out["calibrated_po_reference_integrated_power"] = float(
        po_integrated * gain * gain
    )
    out["rt_reference_power_peak"] = float(np.max(rt_values))
    out["raw_po_reference_power_peak"] = float(np.max(po_values))
    out["rt_reference_power_peak_over_mean_db"] = float(
        10.0 * np.log10(max(np.max(rt_values), 1e-300) / max(rt_mean, 1e-300))
    )
    out.update(gain_meta)
    if gain_meta.get("calibration_gain_strategy") == "time_varying_last_retrace":
        out["calibrated_po_gain_map_mean_power"] = float(
            np.mean(np.asarray(po_path_power, dtype=float) * gain_map * gain_map)
        )
    return gain, out


def calibration_gain_array(
    calibration: dict | None,
    shape: tuple[int, int],
    *,
    fallback: float = 1.0,
) -> np.ndarray:
    """Returns a per-chirp amplitude-gain array from calibration metadata.

    ``shape`` is normally ``[num_frames, num_chirps_per_frame]``. Metadata with
    an exact gain-map shape is returned as-is, scalar gains are broadcast, and
    incompatible or missing metadata falls back to ``fallback``.
    """

    if not calibration:
        return np.full(shape, float(fallback), dtype=np.float64)
    value = calibration.get("human_po_amplitude_gain_map")
    if value is None:
        value = calibration.get("human_po_amplitude_gain", fallback)
    gains = np.asarray(value, dtype=np.float64)
    if gains.shape == shape:
        return gains
    if gains.shape == ():
        return np.full(shape, float(gains), dtype=np.float64)
    if gains.size == int(np.prod(shape)):
        return gains.reshape(shape)
    return np.full(shape, float(fallback), dtype=np.float64)
