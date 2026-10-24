# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Common measurement/numerical metrics and seed-power aggregation."""

import numpy as np

from .metrics import background_subtract_adc, normalized_correlation
from .dsp_maps import (
    compute_maps as _compute_maps,
    compute_angle_fft_maps as _compute_angle_fft_maps,
)
from .channels import (
    subarray_selection as _subarray_selection,
    select_adc_channels as _select_adc_channels,
)
from .simulation import (
    simulation_mesh_sequence as _simulation_mesh_sequence,
    clip_uses_tdm_virtual_adc as _clip_uses_tdm_virtual_adc,
    tdm_tx_order as _tdm_tx_order,
)


def relative_error(reference, estimate):
    denominator = float(np.linalg.norm(reference))
    return float(np.linalg.norm(estimate - reference) / denominator) if denominator else None


def power_comparison(reference, estimate):
    rsum, esum = float(np.sum(reference)), float(np.sum(estimate))
    shape = relative_error(reference / rsum, estimate / esum) if rsum > 0 and esum > 0 else None
    return {
        "relative_l2": relative_error(reference, estimate),
        "shape_relative_l2": shape,
        "normalized_correlation": float(normalized_correlation(reference, estimate)),
        "native_power_ratio_db": (
            float(10.0 * np.log10(esum / rsum)) if rsum > 0 and esum > 0 else None
        ),
    }


def compare(clip, config, candidate, reference, *, coherent=False, absolute_power=False):
    """Use the existing validation DSP for both reference types."""

    if candidate.shape != reference.shape:
        raise ValueError(f"Comparison ADC shape mismatch: {candidate.shape} != {reference.shape}")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise ValueError("Comparison ADC contains non-finite values")
    maps = _compute_maps(
        reference,
        {"candidate": candidate},
        clip.sensor_config.fmcw,
        primary_label="candidate",
        remove_mean_clutter=config.clutter_removal == "mean",
    )
    selection = _subarray_selection(clip, config, full_channels=clip.real_adc.shape[-1])
    maps.update(
        _compute_angle_fft_maps(
            reference,
            {"candidate": candidate},
            clip.sensor_config.fmcw,
            selection.virtual_positions_lambda,
            maps,
            primary_label="candidate",
        )
    )
    metrics = {
        "adc_relative_l2": relative_error(reference, candidate) if coherent else None,
        "coherent_adc_metric_enabled": coherent,
        "absolute_power_calibrated": absolute_power,
        "angle_definition": "existing validation geometry-aware matched-filter map",
    }
    arrays = {}
    for domain, ref_key, est_key in (
        ("range_time", "real_range_time_power", "candidate_range_time_power"),
        ("range_doppler", "real_range_doppler_power", "candidate_range_doppler_power"),
        ("angle", "real_angle_fft_power", "candidate_angle_fft_power"),
    ):
        if ref_key not in maps:
            continue
        r, e = np.asarray(maps[ref_key]), np.asarray(maps[est_key])
        metrics[domain] = power_comparison(r, e)
        if absolute_power:
            metrics[domain]["absolute_power_error_db"] = metrics[domain]["native_power_ratio_db"]
        arrays[f"reference_{domain}"] = r
        arrays[f"candidate_{domain}"] = e
    ranges = maps["range_time_ranges_m"]
    rprofile = np.mean(maps["real_range_time_power"], axis=0)
    eprofile = np.mean(maps["candidate_range_time_power"], axis=0)
    metrics["peak_range_error_m"] = float(
        abs(ranges[np.argmax(rprofile)] - ranges[np.argmax(eprofile)])
    )
    velocities = maps["range_doppler_velocities_mps"]
    # Existing DSP maps have axes [frame, Doppler, range].
    rrd, erd = maps["real_range_doppler_power"], maps["candidate_range_doppler_power"]
    rd, ed = rrd.sum(axis=-1), erd.sum(axis=-1)
    metrics["doppler_time"] = power_comparison(rd, ed)
    rcent = np.sum(rd * velocities[None], axis=-1) / np.maximum(rd.sum(axis=-1), 1e-30)
    ecent = np.sum(ed * velocities[None], axis=-1) / np.maximum(ed.sum(axis=-1), 1e-30)
    metrics["doppler_centroid_mae_mps"] = float(np.mean(abs(rcent - ecent)))
    arrays.update(
        reference_doppler_time=rd,
        candidate_doppler_time=ed,
        ranges_m=ranges,
        velocities_mps=velocities,
    )
    return metrics, arrays


def measured_adc(clip, config):
    selection = _subarray_selection(clip, config, full_channels=clip.real_adc.shape[-1])
    adc = _select_adc_channels(np.asarray(clip.real_adc), selection)
    if config.background_subtract:
        if clip.background_adc is None:
            raise ValueError("background_subtract requires a measured background artifact")
        background = _select_adc_channels(np.asarray(clip.background_adc), selection)
        adc = background_subtract_adc(adc, background)
    return adc


def scheduled_displacement(clip, config):
    """Describe the physical motion axis, not an inferred error guarantee."""

    motion = _simulation_mesh_sequence(clip, config)
    fmcw = clip.sensor_config.fmcw
    loops = int(fmcw.num_chirps_per_frame)
    is_tdm = _clip_uses_tdm_virtual_adc(clip)
    stride = int(fmcw.num_tx) if is_tdm else 1
    offsets = [0]
    if is_tdm and not config.ignore_tdm_timing:
        selection = _subarray_selection(clip, config, full_channels=clip.real_adc.shape[-1])
        offsets = [
            slot
            for slot, tx in enumerate(_tdm_tx_order(clip))
            if int(tx) + 1 in selection.tx_indices_1based
        ]
    slot_times = [
        (
            clip.motion_times_s[:, None]
            + (np.arange(loops)[None] * stride + offset) * float(fmcw.chirp_repetition_time)
        ).reshape(-1)
        for offset in offsets
    ]
    settings = config.effective_solver_settings
    outputs = {}
    for solver in ("rt", "po"):
        period = max(1, int(settings[f"{solver}_refresh_frames"]) * loops)
        values = []
        for times in slot_times:
            anchor = None
            for i, t in enumerate(times):
                vertices = motion.vertices_at(float(t))
                if i % period == 0:
                    anchor = vertices.copy()
                values.append(float(np.max(np.linalg.norm(vertices - anchor, axis=1))))
        outputs[solver] = {
            "scheduled_max_vertex_drift_mm": 1000.0 * max(values),
            "scheduled_p95_vertex_drift_mm": 1000.0 * float(np.percentile(values, 95)),
            "scheduled_max_vertex_drift_wavelengths": max(values) / fmcw.wavelength,
            "refresh_frames": settings[f"{solver}_refresh_frames"],
            "definition": "drift since scheduled refresh; early per-facet/drift triggers may refresh sooner",
        }
    return outputs


def aggregate_seed_powers(paths):
    """Average powers, never complex fields, across stochastic seeds."""

    means = {}
    for count, path in enumerate(paths, 1):
        with np.load(path, allow_pickle=False) as arrays:
            for key in arrays.files:
                if key.startswith(("reference_", "candidate_")):
                    value = np.asarray(arrays[key], dtype=float)
                    if key not in means:
                        means[key] = value.copy()
                    else:
                        means[key] += (value - means[key]) / count
    result = {}
    for key in means:
        if key.startswith("reference_"):
            domain = key.removeprefix("reference_")
            result[domain] = power_comparison(means[key], means[f"candidate_{domain}"])
    return result
