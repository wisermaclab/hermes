# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

# Imports below preserve the established runner facade for existing callers.
# pylint: disable=unused-import

"""Orchestration and compatibility facade for validation benchmarks."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import numpy as np
from bundle_prepare.contract import Bundle as BenchmarkClip
from .metrics import (
    background_subtract_adc,
    dsp_map_metrics,
    mesh_range_summary,
    normalized_correlation,
    suppress_zero_doppler,
    zero_doppler_power_fraction,
)
from .results import save_maps_npz, write_metrics_json
from mmWaveRadar.provenance import capture_provenance
from . import cache as _cache
from . import channels as _channels
from . import configuration as _configuration
from . import dsp_maps as _dsp_maps
from . import simulation as _simulation

# Existing import paths remain available; shared implementations have public homes.
from .configuration import (
    SuiteName,
    ClutterRemovalMode,
    DEFAULT_VALIDATION_OUTPUT_DIR,
    DEFAULT_ZERO_DOPPLER_WIDTH_BINS,
    VALIDATION_SOLVERS,
    DEFAULT_VALIDATION_SOLVERS,
    BenchmarkRunConfig,
    zero_based_indices as _zero_based_indices,
    resolve_clutter_removal as _resolve_clutter_removal,
)
from .channels import (
    SubarraySelection as _SubarraySelection,
    slice_clip as _slice_clip,
    subarray_selection as _subarray_selection,
    channel_tx_rx_indices_1based as _channel_tx_rx_indices_1based,
    channel_virtual_positions_lambda as _channel_virtual_positions_lambda,
    requested_indices_or_all as _requested_indices_or_all,
    ordered_unique as _ordered_unique,
    select_adc_channels as _select_adc_channels,
    select_or_validate_adc_channels as _select_or_validate_adc_channels,
    select_background_channels as _select_background_channels,
    subarray_metadata as _subarray_metadata,
    write_subarray_adc as _write_subarray_adc,
    index_tuple_token as _index_tuple_token,
)
from .simulation import (
    simulate_bundle_adc,
    simulate_adc as _simulate_adc,
    resolve_scene_path as _resolve_scene_path,
    clip_uses_tdm_virtual_adc as _clip_uses_tdm_virtual_adc,
    load_bundle_scene as _load_bundle_scene,
    tdm_tx_order as _tdm_tx_order,
    simulate_tdm_virtual_adc as _simulate_tdm_virtual_adc,
    simulate_simultaneous_tdm_adc as _simulate_simultaneous_tdm_adc,
    simultaneous_tdm_sensor as _simultaneous_tdm_sensor,
    tdm_slot_rx_outputs as _tdm_slot_rx_outputs,
    TimedMeshSequence as _TimedMeshSequence,
    BundleFrameMeshSequence as _BundleFrameMeshSequence,
    simulation_mesh_sequence as _simulation_mesh_sequence,
    PlacedBundleMotionSequence as _PlacedBundleMotionSequence,
    validate_motion_acquisition_coverage as _validate_motion_acquisition_coverage,
    solver_cache_settings as _solver_cache_settings,
    validation_simulator as _validation_simulator,
    target_material as _target_material,
    benchmark_cube as _benchmark_cube,
    run_observed_simulator as _run_observed_simulator,
    load_amass_motion_sequence as _load_amass_motion_sequence,
    TransformedSMPLMotionSequence as _TransformedSMPLMotionSequence,
    optional_npz_string as _optional_npz_string,
)
from .cache import (
    load_or_simulate_adcs as _load_or_simulate_adcs,
    labeled_solvers as _labeled_solvers,
    load_or_simulate_cached_adc as _load_or_simulate_cached_adc,
    simulated_adc_cache_path as _simulated_adc_cache_path,
    simulated_adc_cache_metadata as _simulated_adc_cache_metadata,
    simulated_adc_input_fingerprint as _simulated_adc_input_fingerprint,
    sensor_config_cache_payload as _sensor_config_cache_payload,
    frames_cache_payload as _frames_cache_payload,
    array_fingerprint as _array_fingerprint,
    file_fingerprint as _file_fingerprint,
    cache_resource_name as _cache_resource_name,
    scene_input_fingerprint as _scene_input_fingerprint,
    scene_xml_file_references as _scene_xml_file_references,
    smpl_model_fingerprint as _smpl_model_fingerprint,
    ti_digitized_pattern_fingerprint as _ti_digitized_pattern_fingerprint,
    stable_json_hash as _stable_json_hash,
    jsonable as _jsonable,
    simulated_adc_cache_matches as _simulated_adc_cache_matches,
    write_simulated_adc_cache as _write_simulated_adc_cache,
)
from .dsp_maps import (
    compute_maps as _compute_maps,
    selected_phase_signs as _selected_phase_signs,
    compute_angle_fft_maps as _compute_angle_fft_maps,
    range_time_peak_bins as _range_time_peak_bins,
    shared_simulation_angle_fft_bins as _shared_simulation_angle_fft_bins,
    angle_fft_power_from_adc as _angle_fft_power_from_adc,
    write_plots as _write_plots,
    plot_comparison_suffix as _plot_comparison_suffix,
    plot_mode_filename_token as _plot_mode_filename_token,
    plot_mode_display_name as _plot_mode_display_name,
    angle_fft_label as _angle_fft_label,
    rms as _rms,
)


def run_validation_benchmark(
    clip: BenchmarkClip,
    config: _configuration.BenchmarkRunConfig,
) -> dict:
    """Runs one validation benchmark and writes metrics/maps under ``out_dir``."""

    out_dir = Path(config.out_dir)
    if not config.target_material:
        config = replace(
            config,
            target_material=dict(
                ((clip.descriptor or {}).get("target") or {}).get("material_parameters", {})
            ),
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    clip = _channels.slice_clip(clip, config.max_frames)
    real_adc_full = np.asarray(clip.real_adc)
    subarray = _channels.subarray_selection(clip, config, full_channels=real_adc_full.shape[3])
    sim_adcs, sim_adc_cache = _cache.load_or_simulate_adcs(clip, config)
    real_adc = _channels.select_adc_channels(real_adc_full, subarray)
    for label, sim_adc in sim_adcs.items():
        sim_adcs[label] = _channels.select_or_validate_adc_channels(
            sim_adc,
            subarray,
            full_shape=real_adc_full.shape,
            selected_shape=real_adc.shape,
            label=label,
        )
    for label, sim_adc in sim_adcs.items():
        if sim_adc.shape != real_adc.shape:
            raise ValueError(
                f"{label} ADC shape must match real ADC shape "
                f"({sim_adc.shape} != {real_adc.shape})"
            )
    primary_label = "po" if "po" in sim_adcs else next(iter(sim_adcs))
    primary_adc = sim_adcs[primary_label]

    sensor = clip.sensor()
    range_mesh_sequence = _simulation.simulation_mesh_sequence(clip, config)
    mesh_summary = mesh_range_summary(
        range_mesh_sequence,
        sensor,
        clip.motion_times_s,
    )

    real_for_metrics = real_adc
    sim_for_metrics = dict(sim_adcs)
    if config.background_subtract:
        if clip.background_adc is None:
            raise ValueError(
                "background_subtract=True requires background_adc.npz in the "
                "validation clip bundle"
            )
        background_adc = _channels.select_background_channels(
            clip.background_adc,
            subarray,
            full_shape=real_adc_full.shape,
        )
        real_for_metrics = background_subtract_adc(real_adc, background_adc)
        # The v1 simulator paths are human-only by default, so they have no
        # static scene component to remove. Keep simulated ADC unchanged; real
        # ADC is clutter-suppressed before comparison.
        sim_for_metrics = dict(sim_adcs)

    fmcw = clip.sensor_config.fmcw
    clutter_removal = _configuration.resolve_clutter_removal(clip, config)
    remove_mean_clutter = clutter_removal == "mean"
    scene_path = _simulation.resolve_scene_path(clip, config)
    metrics = {
        "provenance": capture_provenance(
            settings=config.effective_solver_settings,
            seed=config.effective_solver_settings["rt_seed"],
            input_hashes={
                label: _simulated_adc_cache_metadata(clip, config, solver=solver)[
                    "input_fingerprint"
                ]
                for label, solver in _labeled_solvers(config.solvers)
            },
        ),
        "suite": config.suite,
        "clip": str(clip.root),
        "num_frames": int(real_adc.shape[0]),
        "adc_shape": list(real_adc.shape),
        "source_adc_shape": list(real_adc_full.shape),
        "board_model": clip.sensor_config.board_model,
        "sensor_metadata": dict(clip.sensor_config.metadata or {}),
        "virtual_channel_order": clip.sensor_config.virtual_channel_order,
        "frames": [
            {
                "benchmark_index": int(frame.benchmark_index),
                "motion_time_s": float(frame.motion_time_s),
                "radar_frame_id": frame.radar_frame_id,
                "camera_frame_id": frame.camera_frame_id,
                "pose_frame_id": frame.pose_frame_id,
                "dataset_frame_id": frame.dataset_frame_id,
                "metadata": dict(frame.metadata or {}),
            }
            for frame in clip.frames
        ],
        "fmcw": {
            "carrier_frequency_hz": float(fmcw.carrier_frequency),
            "slope_hz_per_s": float(fmcw.slope),
            "chirp_duration_s": float(fmcw.chirp_duration),
            "chirp_repetition_time_s": float(fmcw.chirp_repetition_time),
            "sampling_frequency_hz": float(fmcw.sampling_frequency),
            "adc_mode": fmcw.adc_mode,
            "if_bandwidth_hz": fmcw.if_bandwidth_hz,
            "num_adc_samples": int(fmcw.num_adc_samples),
            "num_chirps_per_frame": int(fmcw.num_chirps_per_frame),
            "frame_period_s": float(fmcw.frame_period),
            "num_tx": int(fmcw.num_tx),
            "tx_schedule": str(fmcw.tx_schedule),
        },
        "mesh_range_summary": mesh_summary.to_jsonable(),
        "background_subtract": bool(config.background_subtract),
        "clutter_removal": clutter_removal,
        "scene_path": None if scene_path is None else str(scene_path),
        "doppler_velocity_convention": "range_rate_mps_positive_away",
        "rt_refresh_frames": int(config.rt_refresh_frames),
        "po_refresh_on_phase_error_rad": float(config.po_refresh_on_phase_error_rad),
        "rt_diffuse_reflection": bool(config.rt_diffuse_reflection),
        "rt_refraction": bool(config.rt_refraction),
        "rt_human_specular_reflection": bool(config.rt_human_specular_reflection),
        "hybrid_calibration": str(config.hybrid_calibration),
        "ignore_tdm_timing": bool(config.ignore_tdm_timing),
        "solvers": list(config.solvers),
        "subarray": _channels.subarray_metadata(subarray),
        "primary_simulation": primary_label,
        "simulated_adc_cache": sim_adc_cache,
    }

    maps = _dsp_maps.compute_maps(
        real_for_metrics,
        sim_for_metrics,
        fmcw,
        primary_label=primary_label,
        remove_mean_clutter=remove_mean_clutter,
    )
    angle_maps = _dsp_maps.compute_angle_fft_maps(
        real_for_metrics,
        sim_for_metrics,
        fmcw,
        subarray.virtual_positions_lambda,
        maps,
        phase_signs=_dsp_maps.selected_phase_signs(clip, subarray),
        primary_label=primary_label,
    )
    maps.update(angle_maps)
    metrics["raw_diagnostics"] = {
        "real_adc_rms": _dsp_maps.rms(real_adc),
        "simulated_adc_rms": _dsp_maps.rms(primary_adc),
        **{f"{label}_adc_rms": _dsp_maps.rms(adc) for label, adc in sim_adcs.items()},
    }
    metrics["comparisons"] = {}
    if subarray.active:
        _channels.write_subarray_adc(out_dir / "subarray_adc.npz", real_adc, sim_adcs, subarray)

    if config.suite in ("smoke", "range_time", "dsp_maps"):
        for label in sim_adcs:
            rt_metrics = dsp_map_metrics(
                maps["real_range_time_power"],
                maps[f"{label}_range_time_power"],
                maps["range_time_ranges_m"],
                mesh_summary,
            )
            metrics["comparisons"].setdefault(label, {})["range_time"] = rt_metrics.to_jsonable()
        metrics["range_time"] = metrics["comparisons"][primary_label]["range_time"]

    if config.suite in ("smoke", "range_doppler", "dsp_maps"):
        real_rd_power = maps["real_range_doppler_power"]
        raw_diag = {
            "real_zero_doppler_power_fraction": zero_doppler_power_fraction(
                real_rd_power,
                width_bins=_configuration.DEFAULT_ZERO_DOPPLER_WIDTH_BINS,
            )
        }
        real_rd_eval = suppress_zero_doppler(
            real_rd_power,
            width_bins=_configuration.DEFAULT_ZERO_DOPPLER_WIDTH_BINS,
        )
        maps["real_range_doppler_power_suppressed"] = real_rd_eval
        for label in sim_adcs:
            sim_rd_power = maps[f"{label}_range_doppler_power"]
            raw_diag[f"{label}_zero_doppler_power_fraction"] = zero_doppler_power_fraction(
                sim_rd_power,
                width_bins=_configuration.DEFAULT_ZERO_DOPPLER_WIDTH_BINS,
            )
            sim_rd_eval = suppress_zero_doppler(
                sim_rd_power,
                width_bins=_configuration.DEFAULT_ZERO_DOPPLER_WIDTH_BINS,
            )
            maps[f"{label}_range_doppler_power_suppressed"] = sim_rd_eval
            rd_metrics = dsp_map_metrics(
                real_rd_eval,
                sim_rd_eval,
                maps["range_doppler_ranges_m"],
                mesh_summary,
            )
            metrics["comparisons"].setdefault(label, {})[
                "range_doppler"
            ] = rd_metrics.to_jsonable()
        raw_diag["simulated_zero_doppler_power_fraction"] = raw_diag[
            f"{primary_label}_zero_doppler_power_fraction"
        ]
        metrics["range_doppler_raw_diagnostics"] = raw_diag
        metrics["range_doppler"] = metrics["comparisons"][primary_label]["range_doppler"]
        maps["sim_range_doppler_power_suppressed"] = maps[
            f"{primary_label}_range_doppler_power_suppressed"
        ]

    if "real_angle_fft_power" in maps:
        metrics["angle_fft"] = {
            "description": (
                "Single-range-bin geometry-aware matched-filter angle map. "
                "Real uses its own "
                "range-time profile peak bin. PO/RT share one simulation "
                "range-time peak bin, selected from whichever simulation peak "
                "is closer to the real peak. The map uses the board virtual "
                "channel phase-center coordinates directly instead of gridding "
                "the snapshots into a dense virtual-array image."
            ),
            "method": str(np.asarray(maps["angle_fft_method"]).item()),
            "virtual_positions_lambda": (maps["angle_fft_virtual_positions_lambda"].tolist()),
            "grid_size": maps["angle_fft_fft_size"].tolist(),
            "selection_mode": str(np.asarray(maps["angle_fft_selection_mode"]).item()),
            "range_gate_m": maps["angle_fft_range_gate_m"].tolist(),
            "real_selected_range_bin": (maps["real_angle_fft_range_bin"].tolist()),
            "real_selected_range_m": (maps["real_angle_fft_range_m"].tolist()),
            "simulation_selected_range_bin": (maps["sim_angle_fft_range_bin"].tolist()),
            "simulation_selected_range_m": (maps["sim_angle_fft_range_m"].tolist()),
            "simulation_selection_source": (maps["angle_fft_sim_selection_source"].tolist()),
        }
        for label in sim_adcs:
            metrics["comparisons"].setdefault(label, {})["angle_fft"] = {
                "normalized_correlation": normalized_correlation(
                    maps["real_angle_fft_power"],
                    maps[f"{label}_angle_fft_power"],
                )
            }

    if config.suite in ("smoke", "pose_range", "dsp_maps"):
        metrics["pose_range"] = {
            "mesh_min_m": mesh_summary.min_m,
            "mesh_median_m": mesh_summary.median_m,
            "mesh_max_m": mesh_summary.max_m,
        }

    save_maps_npz(out_dir / "maps.npz", **maps)
    write_metrics_json(out_dir / "metrics.json", metrics)
    if config.write_plots:
        plot_modes = (
            (("sim", "sim"),)
            if config.simulated_adc_path is not None
            else _cache.labeled_solvers(config.solvers)
        )
        _dsp_maps.write_plots(out_dir, maps, simulated_solvers=plot_modes)
    return metrics
