# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Benchmark runner for dataset-neutral real-radar validation clips."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from importlib import resources
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Literal, Mapping
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from mmWaveRadar.dsp import (
    framewise_range_doppler_map,
    psf_cross_correlation_image,
    range_fft,
    range_time_map,
)

from bundle_prepare.contract import Bundle as BenchmarkClip
from .metrics import (
    background_subtract_adc,
    dsp_map_metrics,
    mesh_range_summary,
    normalized_correlation,
    suppress_zero_doppler,
    zero_doppler_power_fraction,
)
from .results import (
    plot_angle_fft_comparison,
    plot_range_doppler_comparison,
    plot_range_profile_comparison,
    plot_range_time_comparison,
    save_maps_npz,
    write_metrics_json,
)


SuiteName = Literal["smoke", "range_time", "range_doppler", "pose_range", "dsp_maps"]
ClutterRemovalMode = Literal["none", "mean"]
HybridPOCalibrationMode = Literal["none", "rt_first_pose", "rt_sequence"]
DEFAULT_VALIDATION_OUTPUT_DIR = (
    Path(tempfile.gettempdir()) / "mmwave_validation"
)
DEFAULT_ZERO_DOPPLER_WIDTH_BINS = 1
VALIDATION_MOBILITY_MODES = (
    "human_only_po",
    "rt_retrace",
    "rt_coherent_bank",
    "hybrid_static_env_po",
)
DEFAULT_VALIDATION_MOBILITY_MODES = ("human_only_po", "rt_coherent_bank")


@dataclass(frozen=True)
class BenchmarkRunConfig:
    """Runtime controls for a validation benchmark run."""

    suite: SuiteName = "dsp_maps"
    out_dir: Path = DEFAULT_VALIDATION_OUTPUT_DIR
    simulated_adc_path: Path | None = None
    max_frames: int | None = None
    background_subtract: bool = False
    write_plots: bool = True
    simulate: bool = True
    clutter_removal: ClutterRemovalMode = "none"
    scene_path: str | None = None
    mobility_modes: tuple[str, ...] = DEFAULT_VALIDATION_MOBILITY_MODES
    diffuse_reflection: bool = True
    refraction: bool = False
    human_specular_reflection: bool = False
    hybrid_po_calibration_mode: HybridPOCalibrationMode = "rt_first_pose"
    smpl_model_dir: str | None = None
    subarray_tx_indices: tuple[int, ...] | None = None
    subarray_rx_indices: tuple[int, ...] | None = None
    progress: bool = False
    ignore_tdm_timing: bool = False

    def __post_init__(self):
        modes = tuple(str(mode) for mode in self.mobility_modes)
        if not modes:
            raise ValueError("mobility_modes must select at least one mode")
        unknown = [mode for mode in modes if mode not in VALIDATION_MOBILITY_MODES]
        if unknown:
            raise ValueError(
                "mobility_modes contains unsupported mode(s): "
                + ", ".join(unknown)
            )
        if len(set(modes)) != len(modes):
            raise ValueError("mobility_modes must not contain duplicates")
        object.__setattr__(self, "mobility_modes", modes)
        if self.hybrid_po_calibration_mode not in (
            "none",
            "rt_first_pose",
            "rt_sequence",
        ):
            raise ValueError(
                "hybrid_po_calibration_mode must be 'none', "
                "'rt_first_pose', or 'rt_sequence'"
            )
        smpl_model_dir = self.smpl_model_dir
        if smpl_model_dir is None:
            smpl_model_dir = os.environ.get("MMWAVE_SMPL_MODEL_DIR") or None
        if smpl_model_dir is not None:
            object.__setattr__(
                self,
                "smpl_model_dir",
                str(Path(smpl_model_dir).expanduser()),
            )


@dataclass(frozen=True)
class _SubarraySelection:
    """Resolved virtual-channel subset and the matching TX/RX metadata."""

    channel_indices: np.ndarray
    tx_indices_1based: tuple[int, ...]
    rx_indices_1based: tuple[int, ...]
    channel_tx_indices_1based: np.ndarray
    channel_rx_indices_1based: np.ndarray
    virtual_positions_lambda: np.ndarray
    full_channel_count: int

    @property
    def active(self) -> bool:
        """Whether the run uses fewer channels than the full sensor."""

        return self.channel_indices.size != int(self.full_channel_count)

    @property
    def channel_count(self) -> int:
        """Number of selected virtual channels."""

        return int(self.channel_indices.size)


def simulate_bundle_adc(
    clip: BenchmarkClip,
    *,
    mobility_mode: str,
    smpl_model_dir: str | None = None,
    channel_zero_only: bool = False,
    diffuse_reflection: bool = True,
    refraction: bool = False,
    human_specular_reflection: bool = False,
    hybrid_po_calibration_mode: HybridPOCalibrationMode = "rt_first_pose",
    cancel_check: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Simulate one ADC cube directly from a validated measurement bundle."""

    if mobility_mode not in VALIDATION_MOBILITY_MODES:
        raise ValueError(
            "mobility_mode must be one of: "
            + ", ".join(VALIDATION_MOBILITY_MODES)
        )
    subarray_tx_indices = None
    subarray_rx_indices = None
    if channel_zero_only:
        sensor = clip.sensor()
        channel_tx, channel_rx = _channel_tx_rx_indices_1based(
            clip,
            sensor,
            full_channels=int(np.asarray(clip.real_adc).shape[3]),
        )
        subarray_tx_indices = (int(channel_tx[0]),)
        subarray_rx_indices = (int(channel_rx[0]),)
    config = BenchmarkRunConfig(
        write_plots=False,
        simulate=True,
        mobility_modes=(mobility_mode,),
        diffuse_reflection=diffuse_reflection,
        refraction=refraction,
        human_specular_reflection=human_specular_reflection,
        hybrid_po_calibration_mode=hybrid_po_calibration_mode,
        smpl_model_dir=smpl_model_dir,
        subarray_tx_indices=subarray_tx_indices,
        subarray_rx_indices=subarray_rx_indices,
        progress=False,
    )
    if cancel_check is not None and not callable(cancel_check):
        raise ValueError("cancel_check must be callable or None")
    if cancel_check is None:
        return _simulate_adc(clip, config, mobility_mode=mobility_mode)
    return _simulate_adc(
        clip, config, mobility_mode=mobility_mode, cancel_check=cancel_check
    )


def run_validation_benchmark(
    clip: BenchmarkClip,
    config: BenchmarkRunConfig,
) -> dict:
    """Runs one validation benchmark and writes metrics/maps under ``out_dir``."""

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clip = _slice_clip(clip, config.max_frames)
    real_adc_full = np.asarray(clip.real_adc)
    subarray = _subarray_selection(clip, config, full_channels=real_adc_full.shape[3])
    sim_adcs, sim_adc_cache = _load_or_simulate_adcs(clip, config)
    real_adc = _select_adc_channels(real_adc_full, subarray)
    for label, sim_adc in sim_adcs.items():
        sim_adcs[label] = _select_or_validate_adc_channels(
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
    range_mesh_sequence = _simulation_mesh_sequence(clip, config)
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
        background_adc = _select_background_channels(
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
    clutter_removal = _resolve_clutter_removal(clip, config)
    remove_mean_clutter = clutter_removal == "mean"
    scene_path = _resolve_scene_path(clip, config)
    metrics = {
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
            "num_adc_samples": int(fmcw.num_adc_samples),
            "num_chirps_per_frame": int(fmcw.num_chirps_per_frame),
            "frame_period_s": float(fmcw.frame_period),
            "num_tx": int(fmcw.num_tx),
            "tdm_enabled": bool(fmcw.tdm_enabled),
        },
        "mesh_range_summary": mesh_summary.to_jsonable(),
        "background_subtract": bool(config.background_subtract),
        "clutter_removal": clutter_removal,
        "scene_path": None if scene_path is None else str(scene_path),
        "doppler_velocity_convention": "range_rate_mps_positive_away",
        "diffuse_reflection": bool(config.diffuse_reflection),
        "refraction": bool(config.refraction),
        "human_specular_reflection": bool(config.human_specular_reflection),
        "hybrid_po_calibration_mode": str(config.hybrid_po_calibration_mode),
        "ignore_tdm_timing": bool(config.ignore_tdm_timing),
        "mobility_modes": list(config.mobility_modes),
        "subarray": _subarray_metadata(subarray),
        "primary_simulation": primary_label,
        "simulated_adc_cache": sim_adc_cache,
    }

    maps = _compute_maps(
        real_for_metrics,
        sim_for_metrics,
        fmcw,
        primary_label=primary_label,
        remove_mean_clutter=remove_mean_clutter,
    )
    angle_maps = _compute_angle_fft_maps(
        real_for_metrics,
        sim_for_metrics,
        fmcw,
        subarray.virtual_positions_lambda,
        maps,
        primary_label=primary_label,
    )
    maps.update(angle_maps)
    metrics["raw_diagnostics"] = {
        "real_adc_rms": _rms(real_adc),
        "simulated_adc_rms": _rms(primary_adc),
        **{f"{label}_adc_rms": _rms(adc) for label, adc in sim_adcs.items()},
    }
    metrics["comparisons"] = {}
    if subarray.active:
        _write_subarray_adc(out_dir / "subarray_adc.npz", real_adc, sim_adcs, subarray)

    if config.suite in ("smoke", "range_time", "dsp_maps"):
        for label in sim_adcs:
            rt_metrics = dsp_map_metrics(
                maps["real_range_time_power"],
                maps[f"{label}_range_time_power"],
                maps["range_time_ranges_m"],
                mesh_summary,
            )
            metrics["comparisons"].setdefault(label, {})["range_time"] = (
                rt_metrics.to_jsonable()
            )
        metrics["range_time"] = (
            metrics["comparisons"][primary_label]["range_time"]
        )

    if config.suite in ("smoke", "range_doppler", "dsp_maps"):
        real_rd_power = maps["real_range_doppler_power"]
        raw_diag = {
            "real_zero_doppler_power_fraction": zero_doppler_power_fraction(
                real_rd_power,
                width_bins=DEFAULT_ZERO_DOPPLER_WIDTH_BINS,
            )
        }
        real_rd_eval = suppress_zero_doppler(
            real_rd_power,
            width_bins=DEFAULT_ZERO_DOPPLER_WIDTH_BINS,
        )
        maps["real_range_doppler_power_suppressed"] = real_rd_eval
        for label in sim_adcs:
            sim_rd_power = maps[f"{label}_range_doppler_power"]
            raw_diag[f"{label}_zero_doppler_power_fraction"] = (
                zero_doppler_power_fraction(
                    sim_rd_power,
                    width_bins=DEFAULT_ZERO_DOPPLER_WIDTH_BINS,
                )
            )
            sim_rd_eval = suppress_zero_doppler(
                sim_rd_power,
                width_bins=DEFAULT_ZERO_DOPPLER_WIDTH_BINS,
            )
            maps[f"{label}_range_doppler_power_suppressed"] = sim_rd_eval
            rd_metrics = dsp_map_metrics(
                real_rd_eval,
                sim_rd_eval,
                maps["range_doppler_ranges_m"],
                mesh_summary,
            )
            metrics["comparisons"].setdefault(label, {})["range_doppler"] = (
                rd_metrics.to_jsonable()
            )
        raw_diag["simulated_zero_doppler_power_fraction"] = raw_diag[
            f"{primary_label}_zero_doppler_power_fraction"
        ]
        metrics["range_doppler_raw_diagnostics"] = raw_diag
        metrics["range_doppler"] = (
            metrics["comparisons"][primary_label]["range_doppler"]
        )
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
            "virtual_positions_lambda": (
                maps["angle_fft_virtual_positions_lambda"].tolist()
            ),
            "grid_size": maps["angle_fft_fft_size"].tolist(),
            "selection_mode": str(np.asarray(
                maps["angle_fft_selection_mode"]).item()),
            "range_gate_m": maps["angle_fft_range_gate_m"].tolist(),
            "real_selected_range_bin": (
                maps["real_angle_fft_range_bin"].tolist()
            ),
            "real_selected_range_m": (
                maps["real_angle_fft_range_m"].tolist()
            ),
            "simulation_selected_range_bin": (
                maps["sim_angle_fft_range_bin"].tolist()
            ),
            "simulation_selected_range_m": (
                maps["sim_angle_fft_range_m"].tolist()
            ),
            "simulation_selection_source": (
                maps["angle_fft_sim_selection_source"].tolist()
            ),
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
            else _labeled_mobility_modes(config.mobility_modes)
        )
        _write_plots(out_dir, maps, simulation_modes=plot_modes)
    return metrics


def _slice_clip(clip: BenchmarkClip, max_frames: int | None) -> BenchmarkClip:
    if max_frames is None or max_frames >= clip.num_frames:
        return clip
    count = max(1, int(max_frames))
    background = clip.background_adc
    if background is not None and background.shape == clip.real_adc.shape:
        background = background[:count]
    return BenchmarkClip(
        root=clip.root,
        sensor_config=clip.sensor_config,
        real_adc=clip.real_adc[:count],
        frames=clip.frames[:count],
        motion_path=clip.motion_path,
        times=None if clip.times is None else clip.times[:count],
        background_adc=background,
        environment=clip.environment,
    )


def _subarray_selection(
    clip: BenchmarkClip,
    config: BenchmarkRunConfig,
    *,
    full_channels: int,
) -> _SubarraySelection:
    sensor = clip.sensor()
    channel_tx, channel_rx = _channel_tx_rx_indices_1based(
        clip,
        sensor,
        full_channels=full_channels,
    )
    positions = _channel_virtual_positions_lambda(sensor, channel_tx, channel_rx)
    requested_tx = _requested_indices_or_all(
        config.subarray_tx_indices,
        channel_tx,
        name="subarray TX",
    )
    requested_rx = _requested_indices_or_all(
        config.subarray_rx_indices,
        channel_rx,
        name="subarray RX",
    )
    mask = (
        np.isin(channel_tx, np.asarray(requested_tx, dtype=np.int64))
        & np.isin(channel_rx, np.asarray(requested_rx, dtype=np.int64))
    )
    channel_indices = np.flatnonzero(mask)
    if channel_indices.size == 0:
        raise ValueError(
            "Subarray TX/RX selection produced no virtual channels "
            f"(tx={requested_tx}, rx={requested_rx})"
        )
    return _SubarraySelection(
        channel_indices=channel_indices.astype(np.int64),
        tx_indices_1based=tuple(int(v) for v in requested_tx),
        rx_indices_1based=tuple(int(v) for v in requested_rx),
        channel_tx_indices_1based=channel_tx[channel_indices].astype(np.int64),
        channel_rx_indices_1based=channel_rx[channel_indices].astype(np.int64),
        virtual_positions_lambda=positions[channel_indices].astype(float),
        full_channel_count=int(full_channels),
    )


def _channel_tx_rx_indices_1based(
    clip: BenchmarkClip,
    sensor,
    *,
    full_channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_rx = int(sensor.hardware.num_rx)
    metadata = clip.sensor_config.metadata or {}
    hardware_order = getattr(sensor.hardware, "virtual_channel_order", "tx_major")
    declared_order = getattr(
        clip.sensor_config,
        "virtual_channel_order",
        hardware_order,
    )
    exported_axes = metadata.get("exported_adc_axes") or []
    legacy_tx_major = declared_order == "tx_major" and (
        metadata.get("source_adc_axes")
        == ["adc_sample", "chirp_loop", "rx", "tx"]
        or any("tx_major" in str(axis) for axis in exported_axes)
    )
    if _clip_uses_tdm_virtual_adc(clip) and legacy_tx_major:
        try:
            tx_order = _tdm_tx_order(clip)
        except ValueError:
            tx_order = None
        if tx_order is not None and tx_order.size * num_rx == int(full_channels):
            return (
                np.repeat(tx_order + 1, num_rx).astype(np.int64),
                np.tile(np.arange(1, num_rx + 1, dtype=np.int64), tx_order.size),
            )

    tx = sensor.hardware.virtual_tx_indices() + 1
    rx = sensor.hardware.virtual_rx_indices() + 1
    if tx.shape != (int(full_channels),) or rx.shape != (int(full_channels),):
        raise ValueError(
            "Could not derive TX/RX labels for ADC channel axis "
            f"({tx.size}, {rx.size}) != {full_channels}"
        )
    return tx.astype(np.int64), rx.astype(np.int64)


def _channel_virtual_positions_lambda(
    sensor,
    channel_tx_1based: np.ndarray,
    channel_rx_1based: np.ndarray,
) -> np.ndarray:
    hardware = sensor.hardware
    positions = hardware.virtual_channel_positions_lambda
    if positions is not None:
        lookup = {
            (int(tx), int(rx)): np.asarray(pos, dtype=float)
            for tx, rx, pos in zip(
                hardware.virtual_tx_indices() + 1,
                hardware.virtual_rx_indices() + 1,
                np.asarray(positions, dtype=float),
            )
        }
        out = []
        for tx, rx in zip(channel_tx_1based, channel_rx_1based):
            key = (int(tx), int(rx))
            if key not in lookup:
                break
            out.append(lookup[key])
        else:
            return np.asarray(out, dtype=float)

    lambda_m = float(
        (hardware.board_metadata or {}).get("lambda_m", sensor.fmcw.wavelength)
    )
    tx_positions = np.asarray(hardware.tx_positions, dtype=float)
    rx_positions = np.asarray(hardware.rx_positions, dtype=float)
    out = []
    for tx, rx in zip(channel_tx_1based, channel_rx_1based):
        tx_index = int(tx) - 1
        rx_index = int(rx) - 1
        out.append((tx_positions[tx_index, 1:] + rx_positions[rx_index, 1:]) / lambda_m)
    return np.asarray(out, dtype=float)


def _requested_indices_or_all(
    requested: tuple[int, ...] | None,
    available: np.ndarray,
    *,
    name: str,
) -> tuple[int, ...]:
    available_unique = _ordered_unique(int(v) for v in np.asarray(available).reshape(-1))
    if requested is None:
        return available_unique
    values = tuple(int(v) for v in requested)
    if not values:
        raise ValueError(f"{name} selection must not be empty")
    missing = sorted(set(values) - set(available_unique))
    if missing:
        raise ValueError(
            f"{name} selection contains unavailable indices {missing}; "
            f"available indices are {list(available_unique)}"
        )
    return values


def _ordered_unique(values) -> tuple[int, ...]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        ivalue = int(value)
        if ivalue not in seen:
            seen.add(ivalue)
            out.append(ivalue)
    return tuple(out)


def _select_adc_channels(adc: np.ndarray, selection: _SubarraySelection) -> np.ndarray:
    x = np.asarray(adc)
    if not selection.active:
        return x
    return x[..., selection.channel_indices]


def _select_or_validate_adc_channels(
    adc: np.ndarray,
    selection: _SubarraySelection,
    *,
    full_shape: tuple[int, ...],
    selected_shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    x = np.asarray(adc)
    if x.shape == selected_shape:
        return x
    if selection.active and x.shape == full_shape:
        return _select_adc_channels(x, selection)
    raise ValueError(
        f"{label} ADC shape must match either selected real ADC shape "
        f"{selected_shape} or full source ADC shape {full_shape}; got {x.shape}"
    )


def _select_background_channels(
    background_adc: np.ndarray | None,
    selection: _SubarraySelection,
    *,
    full_shape: tuple[int, ...],
) -> np.ndarray | None:
    if background_adc is None or not selection.active:
        return background_adc
    background = np.asarray(background_adc)
    selected_channels = selection.channel_count
    full_channels = int(full_shape[-1])
    if background.shape[-1] == selected_channels:
        return background
    if background.shape[-1] != full_channels:
        return background
    return background[..., selection.channel_indices]


def _subarray_metadata(selection: _SubarraySelection) -> dict:
    return {
        "enabled": bool(selection.active),
        "full_channel_count": int(selection.full_channel_count),
        "channel_count": int(selection.channel_count),
        "requested_tx_indices_1based": list(selection.tx_indices_1based),
        "requested_rx_indices_1based": list(selection.rx_indices_1based),
        "channel_indices": selection.channel_indices.astype(int).tolist(),
        "channel_tx_indices_1based": (
            selection.channel_tx_indices_1based.astype(int).tolist()
        ),
        "channel_rx_indices_1based": (
            selection.channel_rx_indices_1based.astype(int).tolist()
        ),
        "virtual_positions_lambda": selection.virtual_positions_lambda.tolist(),
        "channel_order": "preserved_from_input_adc_axis",
    }


def _write_subarray_adc(
    path: Path,
    real_adc: np.ndarray,
    sim_adcs: dict[str, np.ndarray],
    selection: _SubarraySelection,
) -> None:
    payload = {
        "real_adc": np.asarray(real_adc),
        "channel_indices": selection.channel_indices.astype(np.int64),
        "channel_tx_indices_1based": (
            selection.channel_tx_indices_1based.astype(np.int64)
        ),
        "channel_rx_indices_1based": (
            selection.channel_rx_indices_1based.astype(np.int64)
        ),
        "virtual_positions_lambda": selection.virtual_positions_lambda.astype(float),
    }
    for label, adc in sim_adcs.items():
        payload[f"{label}_adc"] = np.asarray(adc)
    np.savez_compressed(path, **payload)


def _resolve_clutter_removal(
    clip: BenchmarkClip,
    config: BenchmarkRunConfig,
) -> ClutterRemovalMode:
    if config.clutter_removal not in ("none", "mean"):
        raise ValueError("clutter_removal must be 'none' or 'mean'")
    return config.clutter_removal


def _load_or_simulate_adcs(
    clip: BenchmarkClip,
    config: BenchmarkRunConfig,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    if config.simulated_adc_path is not None:
        with np.load(config.simulated_adc_path, allow_pickle=False) as data:
            if "adc" not in data.files:
                raise ValueError(f"{config.simulated_adc_path} must contain adc")
            adc = np.array(data["adc"], copy=True)
        return {
            "sim": adc
        }, {
            "sim": {
                "status": "provided",
                "path": str(config.simulated_adc_path),
            }
        }
    if not config.simulate:
        raise ValueError("simulate=False requires simulated_adc_path")
    sim_adcs: dict[str, np.ndarray] = {}
    sim_caches: dict[str, dict] = {}
    for label, mobility_mode in _labeled_mobility_modes(config.mobility_modes):
        adc, cache = _load_or_simulate_cached_adc(
            clip,
            config,
            label=label,
            mobility_mode=mobility_mode,
        )
        sim_adcs[label] = adc
        sim_caches[label] = cache
    return sim_adcs, sim_caches


def _labeled_mobility_modes(mobility_modes: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    modes = tuple(str(mode) for mode in mobility_modes)
    if len(modes) == 1:
        return (("sim", modes[0]),)
    if (
        len(modes) == 2
        and "human_only_po" in modes
        and any(mode in ("rt_retrace", "rt_coherent_bank") for mode in modes)
    ):
        return tuple(
            ("po", mode) if mode == "human_only_po" else ("rt", mode)
            for mode in modes
        )
    return tuple((mode, mode) for mode in modes)


def _load_or_simulate_cached_adc(
    clip: BenchmarkClip,
    config: BenchmarkRunConfig,
    *,
    label: str,
    mobility_mode: str,
) -> tuple[np.ndarray, dict]:
    cache_path = _simulated_adc_cache_path(config, label)
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                if "adc" in data.files and _simulated_adc_cache_matches(
                    data,
                    clip,
                    config,
                    mobility_mode=mobility_mode,
                ):
                    return np.array(data["adc"], copy=True), {
                        "status": "hit",
                        "path": str(cache_path),
                        "mobility_mode": str(mobility_mode),
                    }
        except (
            EOFError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
            zipfile.BadZipFile,
        ) as exc:
            if config.progress:
                print(
                    "[validation] rebuilding unreadable simulated-ADC cache "
                    f"{cache_path}: {type(exc).__name__}",
                    flush=True,
                )
    adc = _simulate_adc(clip, config, mobility_mode=mobility_mode)
    _write_simulated_adc_cache(
        cache_path,
        adc,
        clip,
        config,
        mobility_mode=mobility_mode,
    )
    return adc, {
        "status": "miss_rebuilt",
        "path": str(cache_path),
        "mobility_mode": str(mobility_mode),
    }


def _simulated_adc_cache_path(config: BenchmarkRunConfig, label: str) -> Path:
    return Path(config.out_dir) / f"simulated_adc_{label}.npz"


def _simulated_adc_cache_metadata(
    clip: BenchmarkClip,
    config: BenchmarkRunConfig,
    *,
    mobility_mode: str,
) -> dict[str, object]:
    scene_path = _resolve_scene_path(clip, config)
    selection = _subarray_selection(
        clip,
        config,
        full_channels=int(np.asarray(clip.real_adc).shape[3]),
    )
    source_shape = tuple(int(v) for v in np.asarray(clip.real_adc).shape)
    adc_shape = (
        source_shape[:-1] + (int(selection.channel_count),)
        if selection.active else source_shape
    )
    metadata = {
        "cache_schema_version": 5,
        "clip_root": str(Path(clip.root)),
        "adc_shape": adc_shape,
        "source_adc_shape": source_shape,
        "board_model": str(clip.sensor_config.board_model),
        "num_frames": int(clip.num_frames),
        "mobility_mode": str(mobility_mode),
        "scene_path": "" if scene_path is None else str(scene_path),
        "diffuse_reflection": bool(config.diffuse_reflection),
        "refraction": bool(config.refraction),
        "human_specular_reflection": bool(config.human_specular_reflection),
        "hybrid_po_calibration_mode": (
            str(config.hybrid_po_calibration_mode)
            if mobility_mode == "hybrid_static_env_po"
            else "not_applicable"
        ),
        "ignore_tdm_timing": bool(config.ignore_tdm_timing),
        "amass_sequence_path": str(Path(clip.motion_path)),
        "subarray_tx_indices": _index_tuple_token(config.subarray_tx_indices),
        "subarray_rx_indices": _index_tuple_token(config.subarray_rx_indices),
        "subarray_channel_indices": _index_tuple_token(
            tuple(int(v) for v in selection.channel_indices)
        ),
    }
    metadata["input_fingerprint"] = _simulated_adc_input_fingerprint(
        clip,
        config,
        mobility_mode=mobility_mode,
        scene_path=scene_path,
        selection=selection,
        source_shape=source_shape,
        adc_shape=adc_shape,
    )
    return metadata


def _simulated_adc_input_fingerprint(
    clip: BenchmarkClip,
    config: BenchmarkRunConfig,
    *,
    mobility_mode: str,
    scene_path: str | None,
    selection: _SubarraySelection,
    source_shape: tuple[int, ...],
    adc_shape: tuple[int, ...],
) -> str:
    """Hashes clip inputs and simulation controls that affect cached ADC output."""

    root = Path(clip.root)
    payload = {
        "cache_schema": 5,
        "clip_root": str(root),
        "source_adc_shape": list(source_shape),
        "adc_shape": list(adc_shape),
        "mobility_mode": str(mobility_mode),
        "scene_path": "" if scene_path is None else str(scene_path),
        "scene_inputs": _scene_input_fingerprint(scene_path),
        "smpl_model": _smpl_model_fingerprint(clip, config),
        "ti_digitized_pattern": _ti_digitized_pattern_fingerprint(clip),
        "simulation_config": {
            "diffuse_reflection": bool(config.diffuse_reflection),
            "refraction": bool(config.refraction),
            "human_specular_reflection": bool(config.human_specular_reflection),
            "hybrid_po_calibration_mode": (
                str(config.hybrid_po_calibration_mode)
                if mobility_mode == "hybrid_static_env_po"
                else "not_applicable"
            ),
            "ignore_tdm_timing": bool(config.ignore_tdm_timing),
            "smpl_model_dir": (
                "" if config.smpl_model_dir is None else str(config.smpl_model_dir)
            ),
            "subarray_tx_indices": _index_tuple_token(config.subarray_tx_indices),
            "subarray_rx_indices": _index_tuple_token(config.subarray_rx_indices),
            "subarray_channel_indices": _index_tuple_token(
                tuple(int(v) for v in selection.channel_indices)
            ),
        },
        "sensor_config": _sensor_config_cache_payload(clip),
        "frames": _frames_cache_payload(clip),
        "clip_files": {
            "sensor_json": _file_fingerprint(root / "sensor.json"),
            "frames_json": _file_fingerprint(root / "frames.json"),
            "amass_sequence_npz": _file_fingerprint(clip.motion_path),
            "environment_json": _file_fingerprint(root / "environment.json"),
        },
    }
    return _stable_json_hash(payload)


def _sensor_config_cache_payload(clip: BenchmarkClip) -> dict[str, object]:
    """Returns deterministic sensor metadata for ADC-cache fingerprinting."""

    sensor = clip.sensor_config
    fmcw = sensor.fmcw
    return {
        "board_model": str(sensor.board_model),
        "name": str(sensor.name),
        "position": [float(v) for v in sensor.position],
        "orientation": [float(v) for v in sensor.orientation],
        "pattern_mode": str(sensor.pattern_mode),
        "tx_power_dbm": (
            None if sensor.tx_power_dbm is None else float(sensor.tx_power_dbm)
        ),
        "virtual_channel_order": str(sensor.virtual_channel_order),
        "metadata": _jsonable(sensor.metadata or {}),
        "fmcw": {
            "carrier_frequency_hz": float(fmcw.carrier_frequency),
            "slope_hz_per_s": float(fmcw.slope),
            "chirp_duration_s": float(fmcw.chirp_duration),
            "chirp_repetition_time_s": float(fmcw.chirp_repetition_time),
            "sampling_frequency_hz": float(fmcw.sampling_frequency),
            "num_adc_samples": int(fmcw.num_adc_samples),
            "num_chirps_per_frame": int(fmcw.num_chirps_per_frame),
            "frame_period_s": float(fmcw.frame_period),
            "num_tx": int(fmcw.num_tx),
            "tdm_enabled": bool(fmcw.tdm_enabled),
        },
    }


def _frames_cache_payload(clip: BenchmarkClip) -> list[dict[str, object]]:
    """Returns deterministic frame metadata for ADC-cache fingerprinting."""

    return [
        {
            "benchmark_index": int(frame.benchmark_index),
            "motion_time_s": float(frame.motion_time_s),
            "radar_frame_id": frame.radar_frame_id,
            "camera_frame_id": frame.camera_frame_id,
            "pose_frame_id": frame.pose_frame_id,
            "dataset_frame_id": frame.dataset_frame_id,
            "metadata": _jsonable(frame.metadata or {}),
        }
        for frame in clip.frames
    ]


def _array_fingerprint(value: np.ndarray) -> dict[str, object]:
    """Returns a shape, dtype, and content hash for a numeric array."""

    arr = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(json.dumps(list(arr.shape)).encode("utf-8"))
    digest.update(arr.view(np.uint8))
    return {
        "dtype": str(arr.dtype),
        "shape": [int(v) for v in arr.shape],
        "sha256": digest.hexdigest(),
    }


def _file_fingerprint(path: str | Path | None) -> str:
    """Returns a stable fingerprint for a file path or missing-file state."""

    if path is None:
        return "none"
    candidate = Path(path)
    try:
        stat = candidate.stat()
    except FileNotFoundError:
        return f"missing:{candidate}"
    if not candidate.is_file():
        return f"not_file:{candidate}:{int(stat.st_mtime_ns)}:{int(stat.st_size)}"
    digest = hashlib.sha256()
    with candidate.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _scene_input_fingerprint(path: str | Path | None) -> str:
    """Hashes a scene XML and the local files it references recursively."""

    if path is None:
        return "none"
    root_path = Path(path).expanduser().resolve()
    pending = [root_path]
    visited: set[Path] = set()
    inputs: dict[str, str] = {}
    while pending:
        candidate = pending.pop()
        candidate = candidate.expanduser().resolve()
        if candidate in visited:
            continue
        visited.add(candidate)
        inputs[str(candidate)] = _file_fingerprint(candidate)
        if candidate.suffix.lower() != ".xml" or not candidate.is_file():
            continue
        try:
            document = ET.parse(candidate)
        except (ET.ParseError, OSError):
            continue
        for reference in _scene_xml_file_references(document.getroot()):
            referenced_path = Path(reference).expanduser()
            if not referenced_path.is_absolute():
                referenced_path = candidate.parent / referenced_path
            pending.append(referenced_path)
    return _stable_json_hash(inputs)


def _scene_xml_file_references(root: ET.Element) -> tuple[str, ...]:
    """Returns Mitsuba/Sionna XML filename references in document order."""

    references: list[str] = []
    for element in root.iter():
        filename = element.attrib.get("filename")
        if filename:
            references.append(str(filename))
        if (
            str(element.tag).rsplit("}", 1)[-1] == "string"
            and str(element.attrib.get("name", "")).casefold() == "filename"
            and element.attrib.get("value")
        ):
            references.append(str(element.attrib["value"]))
    return tuple(references)


def _smpl_model_fingerprint(
    clip: BenchmarkClip,
    config: BenchmarkRunConfig,
) -> str:
    """Hashes the effective SMPL-family model files used by this bundle."""

    model_dir = config.smpl_model_dir
    model_type = "smpl"
    gender = "neutral"
    try:
        with np.load(clip.motion_path, allow_pickle=False) as data:
            model_dir = model_dir or _optional_npz_string(data, "smpl_model_dir")
            model_type = _optional_npz_string(data, "model_type") or model_type
            gender = _optional_npz_string(data, "gender") or gender
    except (OSError, ValueError):
        # The motion archive itself is fingerprinted separately and the normal
        # loader will report malformed motion with its more specific error.
        pass
    if not model_dir:
        return "unconfigured"

    root = Path(model_dir).expanduser().resolve()
    if root.is_file():
        return _stable_json_hash({str(root): _file_fingerprint(root)})

    model_root = root / str(model_type).casefold()
    if not model_root.is_dir():
        return _stable_json_hash({str(model_root): _file_fingerprint(model_root)})

    model_files = sorted(
        path
        for path in model_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".pkl", ".npz"}
    )
    gender_token = str(gender).casefold()
    gender_files = [
        path for path in model_files if gender_token in path.stem.casefold()
    ]
    selected = gender_files or model_files
    if not selected:
        return _stable_json_hash({str(model_root): "no_model_files"})
    return _stable_json_hash(
        {
            str(path.relative_to(root)): _file_fingerprint(path)
            for path in selected
        }
    )


def _ti_digitized_pattern_fingerprint(clip: BenchmarkClip) -> str:
    """Hashes the effective optional TI pattern asset when it affects the run."""

    mode = str(clip.sensor_config.pattern_mode).casefold().replace("-", "_")
    if mode != "digitized":
        return "not_applicable"

    from mmWaveRadar.radar import get_ti_board_spec

    spec = get_ti_board_spec(str(clip.sensor_config.board_model))
    if spec.pattern_asset_key is None:
        return "board_has_no_digitized_pattern"

    configured = os.environ.get("MMWAVE_TI_PATTERN_NPZ")
    if configured:
        return _file_fingerprint(Path(configured).expanduser().resolve())

    asset = resources.files("mmWaveRadar.radar").joinpath(
        "data",
        "ti_digitized_patterns.npz",
    )
    if not asset.is_file():
        return "packaged_asset_missing"
    digest = hashlib.sha256()
    with asset.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _stable_json_hash(value: object) -> str:
    """Hashes JSON-compatible data after deterministic key ordering."""

    text = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _jsonable(value: object) -> object:
    """Converts common Python/Numpy values into JSON-serializable values."""

    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _simulated_adc_cache_matches(
    data: np.lib.npyio.NpzFile,
    clip: BenchmarkClip,
    config: BenchmarkRunConfig,
    *,
    mobility_mode: str,
) -> bool:
    expected = _simulated_adc_cache_metadata(
        clip,
        config,
        mobility_mode=mobility_mode,
    )
    adc = np.asarray(data["adc"])
    if tuple(int(v) for v in adc.shape) != expected["adc_shape"]:
        return False
    if not np.issubdtype(adc.dtype, np.number) or not np.all(np.isfinite(adc)):
        return False
    for key, value in expected.items():
        if key not in data.files:
            return False
        stored = np.asarray(data[key])
        if key == "adc_shape":
            if tuple(int(v) for v in stored.reshape(-1)) != value:
                return False
        elif key == "source_adc_shape":
            if tuple(int(v) for v in stored.reshape(-1)) != value:
                return False
        elif isinstance(value, bool):
            if bool(stored.item()) != value:
                return False
        elif isinstance(value, int):
            if int(stored.item()) != value:
                return False
        else:
            if str(stored.item()) != str(value):
                return False
    return True


def _write_simulated_adc_cache(
    path: Path,
    adc: np.ndarray,
    clip: BenchmarkClip,
    config: BenchmarkRunConfig,
    *,
    mobility_mode: str,
) -> None:
    metadata = _simulated_adc_cache_metadata(
        clip,
        config,
        mobility_mode=mobility_mode,
    )
    payload = {"adc": np.asarray(adc)}
    for key, value in metadata.items():
        dtype = np.int64 if key in {"adc_shape", "source_adc_shape"} else None
        payload[key] = np.asarray(value, dtype=dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".npz",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        np.savez_compressed(temporary_path, **payload)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _simulate_adc(
    clip: BenchmarkClip,
    config: BenchmarkRunConfig,
    *,
    mobility_mode: str,
    cancel_check: Callable[[], bool] | None = None,
) -> np.ndarray:
    if cancel_check is not None and bool(cancel_check()):
        raise InterruptedError("Simulation cancelled by user")
    if _clip_uses_tdm_virtual_adc(clip):
        if config.ignore_tdm_timing:
            return _simulate_simultaneous_tdm_adc(
                clip,
                config,
                mobility_mode=mobility_mode,
                cancel_check=cancel_check,
            )
        return _simulate_tdm_virtual_adc(
            clip,
            config,
            mobility_mode=mobility_mode,
            cancel_check=cancel_check,
        )

    from sionna.rt import load_scene

    from mmWaveRadar.materials import human_skin_material
    from mmWaveRadar.simulation import MmWaveRadarSimulator
    from mmWaveRadar.targets import MeshTarget

    scene_path = _resolve_scene_path(clip, config)
    scene = load_scene() if scene_path is None else load_scene(scene_path)
    motion = _simulation_mesh_sequence(clip, config)
    target = MeshTarget(
        name="benchmark_human",
        mesh_sequence=_BundleFrameMeshSequence(
            motion,
            frame_times_s=clip.motion_times_s,
            frame_period_s=float(clip.sensor_config.fmcw.frame_period),
        ),
        material=human_skin_material("benchmark-human-skin"),
    )
    simulator = MmWaveRadarSimulator(
        mobility_mode=mobility_mode,
        diffuse_reflection=config.diffuse_reflection,
        refraction=config.refraction,
        human_specular_reflection=config.human_specular_reflection,
        hybrid_static_env_po_config=_hybrid_static_env_po_config(
            mobility_mode,
            config.hybrid_po_calibration_mode,
        ),
        progress=config.progress,
        cancel_check=cancel_check,
    )
    cube = simulator.run(scene, clip.sensor(), [target], num_frames=clip.num_frames)
    selection = _subarray_selection(
        clip,
        config,
        full_channels=int(np.asarray(clip.real_adc).shape[3]),
    )
    return _select_adc_channels(np.asarray(cube.adc), selection)


def _index_tuple_token(values: tuple[int, ...] | None) -> str:
    if values is None:
        return ""
    return ",".join(str(int(v)) for v in values)


def _resolve_scene_path(clip: BenchmarkClip, config: BenchmarkRunConfig) -> str | None:
    """Returns the scene path from CLI override, environment sidecar, or metadata."""

    value = config.scene_path
    if value is None and clip.environment is not None:
        value = clip.environment.get("scene_path")
    if value is None:
        value = (clip.sensor_config.metadata or {}).get("scene_path")
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)
    return str((Path(clip.root) / path).resolve())


def _clip_uses_tdm_virtual_adc(clip: BenchmarkClip) -> bool:
    fmcw = getattr(clip.sensor_config, "fmcw", None)
    if fmcw is not None and hasattr(fmcw, "tdm_enabled"):
        return bool(fmcw.tdm_enabled)
    metadata = clip.sensor_config.metadata or {}
    if bool(metadata.get("tdm_virtual_adc", False)):
        return True
    source_axes = metadata.get("source_adc_axes")
    if source_axes == ["adc_sample", "chirp_loop", "rx", "tx"]:
        return True
    return False


def _tdm_tx_order(clip: BenchmarkClip) -> np.ndarray:
    metadata = clip.sensor_config.metadata or {}
    value = metadata.get("tx_to_enable") or metadata.get("tdm_tx_order")
    if value is None:
        value = list(range(1, int(clip.sensor_config.fmcw.num_tx) + 1))
    order = np.asarray(value, dtype=np.int64).reshape(-1) - 1
    num_tx = int(clip.sensor_config.fmcw.num_tx)
    if order.shape != (num_tx,):
        raise ValueError(
            f"TDM tx order must contain {num_tx} entries, got {order.size}"
        )
    if sorted(int(v) for v in order) != list(range(num_tx)):
        raise ValueError("TDM tx order must be a permutation of 1..num_tx")
    return order


def _simulate_tdm_virtual_adc(
    clip: BenchmarkClip,
    config: BenchmarkRunConfig,
    *,
    mobility_mode: str,
    cancel_check: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Simulates real TDM-MIMO timing and reassembles virtual-channel ADC.

    The standardized RT-Pose ADC bundle stores one frame as
    ``[loop, sample, tx_major_virtual_channel]`` after separating the physical
    chirp stream into TX slots. For loop ``l`` and scheduled TX slot ``s``, the
    physical chirp time is
    ``clip.motion_times_s[frame] + (l*num_tx + s)*T_chirp``. This helper
    evaluates the moving mesh at that physical time, keeps only the active TX
    group's requested RX outputs, and writes them back into the virtual-channel
    cube. The output shape remains compatible with real ADC maps.

    This is the default TDM policy. It invokes the selected mobility mode once
    per selected physical TX slot so each run receives that slot's timestamps.
    """

    from sionna.rt import load_scene

    from mmWaveRadar.materials import human_skin_material
    from mmWaveRadar.radar import FMCWConfig, RadarSensor
    from mmWaveRadar.simulation import MmWaveRadarSimulator
    from mmWaveRadar.targets import MeshTarget

    fmcw = clip.sensor_config.fmcw
    sensor = clip.sensor()
    tx_order = _tdm_tx_order(clip)
    selection = _subarray_selection(
        clip,
        config,
        full_channels=int(np.asarray(clip.real_adc).shape[3]),
    )
    num_frames = clip.num_frames
    num_loops = int(fmcw.num_chirps_per_frame)
    num_tx = int(fmcw.num_tx)
    num_rx = int(sensor.hardware.num_rx)
    num_adc = int(fmcw.num_adc_samples)
    slot_rx_outputs = _tdm_slot_rx_outputs(
        selection,
        tx_order=tx_order,
        hardware=sensor.hardware,
    )
    out = np.zeros(
        (num_frames, num_loops, num_adc, selection.channel_count),
        dtype=np.complex128,
    )
    mesh_sequence = _simulation_mesh_sequence(clip, config)

    for slot, tx_index in enumerate(tx_order):
        if cancel_check is not None and bool(cancel_check()):
            raise InterruptedError("Simulation cancelled by user")
        selected_outputs = slot_rx_outputs.get(int(slot))
        if not selected_outputs:
            continue
        tx_index_int = int(tx_index)
        slot_times = np.empty((num_frames, num_loops), dtype=float)
        for frame in range(num_frames):
            for loop in range(num_loops):
                slot_times[frame, loop] = (
                    float(clip.motion_times_s[frame])
                    + (loop * num_tx + slot) * float(fmcw.chirp_repetition_time)
                )
        slot_sequence = _TimedMeshSequence(mesh_sequence, slot_times.reshape(-1))
        slot_fmcw = FMCWConfig(
            carrier_frequency=fmcw.carrier_frequency,
            slope=fmcw.slope,
            chirp_duration=fmcw.chirp_duration,
            # Mesh lookup for this internal per-slot run uses synthetic flat
            # indices; the wrapper maps them back to physical TDM times.
            # ADC synthesis itself does not depend on chirp repetition time.
            chirp_repetition_time=1.0,
            sampling_frequency=fmcw.sampling_frequency,
            num_adc_samples=fmcw.num_adc_samples,
            num_chirps_per_frame=fmcw.num_chirps_per_frame,
            frame_period=float(num_loops),
            num_tx=1,
        )
        slot_sensor = RadarSensor(
            hardware=sensor.hardware.subset_tx([tx_index_int]),
            fmcw=slot_fmcw,
            name=f"{sensor.name}-tdm-tx{tx_index_int + 1}",
            position=sensor.position,
            orientation=sensor.orientation,
            tx_power_dbm=sensor.tx_power_dbm,
        )
        scene_path = _resolve_scene_path(clip, config)
        scene = load_scene() if scene_path is None else load_scene(scene_path)
        target = MeshTarget(
            name=f"benchmark_human_tdm_tx{tx_index_int + 1}",
            mesh_sequence=slot_sequence,
            material=human_skin_material("benchmark-human-skin"),
        )
        simulator = MmWaveRadarSimulator(
            mobility_mode=mobility_mode,
            diffuse_reflection=config.diffuse_reflection,
            refraction=config.refraction,
            human_specular_reflection=config.human_specular_reflection,
            hybrid_static_env_po_config=_hybrid_static_env_po_config(
                mobility_mode,
                config.hybrid_po_calibration_mode,
            ),
            progress=config.progress,
            cancel_check=cancel_check,
        )
        cube = simulator.run(scene, slot_sensor, [target], num_frames=num_frames)
        slot_adc = np.asarray(cube.adc)
        if slot_adc.shape != (num_frames, num_loops, num_adc, num_rx):
            raise ValueError(
                "TDM slot simulation returned unexpected shape "
                f"{slot_adc.shape}"
            )
        for out_index, rx_index in selected_outputs:
            out[..., int(out_index)] = slot_adc[..., int(rx_index)]
    return out


def _simulate_simultaneous_tdm_adc(
    clip: BenchmarkClip,
    config: BenchmarkRunConfig,
    *,
    mobility_mode: str,
    cancel_check: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Simulates selected TDM channels together at each TDM-cycle start.

    This intentionally drops the per-slot transmit-time offsets while retaining
    the interval between successive samples of a physical TX. The resulting ADC
    channels are reordered to match the source bundle's TDM channel axis. All
    selected TX/RX pairs share one simulator invocation per mobility mode.
    """

    from sionna.rt import load_scene

    from mmWaveRadar.materials import human_skin_material
    from mmWaveRadar.simulation import MmWaveRadarSimulator
    from mmWaveRadar.targets import MeshTarget

    selection = _subarray_selection(
        clip,
        config,
        full_channels=int(np.asarray(clip.real_adc).shape[3]),
    )
    sensor, output_order = _simultaneous_tdm_sensor(clip, selection)
    scene_path = _resolve_scene_path(clip, config)
    scene = load_scene() if scene_path is None else load_scene(scene_path)
    motion = _simulation_mesh_sequence(clip, config)
    target = MeshTarget(
        name="benchmark_human_tdm_simultaneous",
        mesh_sequence=_BundleFrameMeshSequence(
            motion,
            frame_times_s=clip.motion_times_s,
            frame_period_s=float(clip.sensor_config.fmcw.frame_period),
        ),
        material=human_skin_material("benchmark-human-skin"),
    )
    simulator = MmWaveRadarSimulator(
        mobility_mode=mobility_mode,
        diffuse_reflection=config.diffuse_reflection,
        refraction=config.refraction,
        human_specular_reflection=config.human_specular_reflection,
        hybrid_static_env_po_config=_hybrid_static_env_po_config(
            mobility_mode,
            config.hybrid_po_calibration_mode,
        ),
        progress=config.progress,
        cancel_check=cancel_check,
    )
    cube = simulator.run(scene, sensor, [target], num_frames=clip.num_frames)
    adc = np.asarray(cube.adc)
    expected_shape = (
        clip.num_frames,
        int(clip.sensor_config.fmcw.num_chirps_per_frame),
        int(clip.sensor_config.fmcw.num_adc_samples),
        selection.channel_count,
    )
    if adc.shape != expected_shape:
        raise ValueError(
            "Simultaneous TDM simulation returned unexpected shape "
            f"{adc.shape}; expected {expected_shape}"
        )
    return adc[..., output_order]


def _simultaneous_tdm_sensor(
    clip: BenchmarkClip,
    selection: _SubarraySelection,
):
    """Builds selected simultaneous-TX hardware and its source-axis ordering."""

    from mmWaveRadar.radar import RadarSensor

    source = clip.sensor()
    tx_indices = np.asarray(
        sorted(set(int(v) for v in selection.channel_tx_indices_1based)),
        dtype=np.int64,
    ) - 1
    rx_indices = np.asarray(
        sorted(set(int(v) for v in selection.channel_rx_indices_1based)),
        dtype=np.int64,
    ) - 1
    hardware = source.hardware.subset_tx(tx_indices).subset_rx(rx_indices)

    # Each output chirp is one TDM loop. Preserve that loop-to-loop PRI while
    # removing only the offsets among TX slots inside the loop.
    fmcw = replace(
        source.fmcw,
        chirp_repetition_time=(
            float(source.fmcw.chirp_repetition_time)
            * int(source.fmcw.num_tx)
        ),
        num_tx=hardware.num_tx,
        tdm_enabled=False,
    )
    sensor = RadarSensor(
        hardware=hardware,
        fmcw=fmcw,
        name=f"{source.name}-tdm-simultaneous",
        position=source.position,
        orientation=source.orientation,
        tx_power_dbm=source.tx_power_dbm,
    )

    simulated_pairs = list(zip(
        (tx_indices[hardware.virtual_tx_indices()] + 1).tolist(),
        (rx_indices[hardware.virtual_rx_indices()] + 1).tolist(),
    ))
    pair_to_index = {
        (int(tx), int(rx)): index
        for index, (tx, rx) in enumerate(simulated_pairs)
    }
    desired_pairs = zip(
        selection.channel_tx_indices_1based,
        selection.channel_rx_indices_1based,
    )
    output_order = np.asarray(
        [pair_to_index[(int(tx), int(rx))] for tx, rx in desired_pairs],
        dtype=np.int64,
    )
    return sensor, output_order


def _tdm_slot_rx_outputs(
    selection: _SubarraySelection,
    *,
    tx_order: np.ndarray,
    hardware,
) -> dict[int, list[tuple[int, int]]]:
    """Maps selected source channels to physical TDM slots and slot outputs."""

    slot_by_tx = {
        int(tx_index) + 1: slot
        for slot, tx_index in enumerate(np.asarray(tx_order).reshape(-1))
    }
    rx_output_by_tx: dict[int, dict[int, int]] = {}
    for tx_1based in slot_by_tx:
        slot_hardware = hardware.subset_tx([tx_1based - 1])
        rx_indices = slot_hardware.virtual_rx_indices() + 1
        rx_outputs = {
            int(rx_1based): output_index
            for output_index, rx_1based in enumerate(rx_indices)
        }
        if len(rx_outputs) != int(hardware.num_rx):
            raise ValueError(
                f"TDM Tx {tx_1based} does not map each receiver exactly once"
            )
        rx_output_by_tx[tx_1based] = rx_outputs

    outputs: dict[int, list[tuple[int, int]]] = {}
    channel_pairs = zip(
        selection.channel_tx_indices_1based,
        selection.channel_rx_indices_1based,
    )
    for out_index, (tx_1based, rx_1based) in enumerate(channel_pairs):
        tx = int(tx_1based)
        rx = int(rx_1based)
        if tx not in slot_by_tx:
            raise ValueError(f"Selected TDM channel uses unscheduled Tx {tx}")
        if rx not in rx_output_by_tx[tx]:
            raise ValueError(
                f"Selected TDM channel uses unavailable Tx {tx}/Rx {rx} pair"
            )
        slot = slot_by_tx[tx]
        rx_output = rx_output_by_tx[tx][rx]
        outputs.setdefault(slot, []).append((int(out_index), rx_output))
    return outputs


class _TimedMeshSequence:
    """Mesh sequence view sampled by benchmark-frame timestamps."""

    def __init__(self, base, times_s: np.ndarray):
        self._base = base
        self._times_s = np.asarray(times_s, dtype=float).reshape(-1)
        self.faces = np.asarray(base.faces, dtype=np.uint32)
        self.times = np.arange(self._times_s.size, dtype=float)

    @property
    def vertices(self) -> np.ndarray:
        """Vertices sampled at every benchmark-frame timestamp."""

        return np.stack([self.vertices_at(float(i)) for i in self.times], axis=0)

    @property
    def vertex_count(self) -> int:
        """Number of vertices per mesh frame."""

        return int(self._base.vertex_count)

    @property
    def face_count(self) -> int:
        """Number of triangular faces in the mesh."""

        return int(self.faces.shape[0])

    def vertices_at(self, time: float) -> np.ndarray:
        """Returns vertices for the nearest benchmark-frame index."""

        index = int(round(float(time)))
        index = min(max(index, 0), self._times_s.size - 1)
        return self._base.vertices_at(float(self._times_s[index]))

    def max_vertex_displacement(self, time: float, reference_vertices: np.ndarray) -> float:
        """Returns the largest vertex displacement from ``reference_vertices``."""

        vertices = self.vertices_at(time)
        ref = np.asarray(reference_vertices, dtype=np.float32)
        if ref.shape != vertices.shape:
            raise ValueError("reference_vertices shape mismatch")
        return float(np.max(np.linalg.norm(vertices - ref, axis=1)))


class _BundleFrameMeshSequence:
    """Map simulator-local frame/chirp time to explicit bundle motion time."""

    def __init__(
        self,
        base,
        *,
        frame_times_s: np.ndarray,
        frame_period_s: float,
    ):
        self._base = base
        self._frame_times_s = np.asarray(frame_times_s, dtype=float).reshape(-1)
        self._frame_period_s = float(frame_period_s)
        if self._frame_times_s.size == 0 or not np.all(
            np.isfinite(self._frame_times_s)
        ):
            raise ValueError("bundle frame motion times must be non-empty and finite")
        if not np.isfinite(self._frame_period_s) or self._frame_period_s <= 0.0:
            raise ValueError("bundle frame period must be positive and finite")
        self.faces = np.asarray(base.faces, dtype=np.uint32)
        self.times = (
            np.arange(self._frame_times_s.size, dtype=float) * self._frame_period_s
        )

    @property
    def vertices(self) -> np.ndarray:
        """Vertices sampled at every bundle frame start."""

        return np.stack([self.vertices_at(float(t)) for t in self.times], axis=0)

    @property
    def vertex_count(self) -> int:
        return int(self._base.vertex_count)

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])

    def _source_time(self, simulation_time_s: float) -> float:
        local_time = max(0.0, float(simulation_time_s))
        frame = min(
            int(np.floor(local_time / self._frame_period_s + 1.0e-12)),
            self._frame_times_s.size - 1,
        )
        frame_local_start = frame * self._frame_period_s
        return float(self._frame_times_s[frame] + local_time - frame_local_start)

    def vertices_at(self, time: float) -> np.ndarray:
        return np.asarray(
            self._base.vertices_at(self._source_time(time)),
            dtype=np.float32,
        )

    def max_vertex_displacement(
        self,
        time: float,
        reference_vertices: np.ndarray,
    ) -> float:
        vertices = self.vertices_at(time)
        ref = np.asarray(reference_vertices, dtype=np.float32)
        if ref.shape != vertices.shape:
            raise ValueError("reference_vertices shape mismatch")
        return float(np.max(np.linalg.norm(vertices - ref, axis=1)))


def _simulation_mesh_sequence(clip: BenchmarkClip, config: BenchmarkRunConfig):
    motion = _load_amass_motion_sequence(clip.motion_path, config)
    environment = clip.environment or {}
    if "human_position_m" in environment:
        motion = _PlacedBundleMotionSequence(
            motion,
            position=environment["human_position_m"],
            yaw_deg=environment.get("human_yaw_deg", 0.0),
        )
    _validate_motion_acquisition_coverage(clip, motion)
    return motion


class _PlacedBundleMotionSequence:
    """Apply a bundle-declared radar-relative human placement lazily."""

    def __init__(self, base, *, position, yaw_deg: float):
        self._base = base
        self.times = np.asarray(base.times, dtype=float)
        position_array = np.asarray(position, dtype=float)
        if position_array.shape != (3,) or not np.all(np.isfinite(position_array)):
            raise ValueError("environment human_position_m must contain 3 finite values")
        yaw = np.deg2rad(float(yaw_deg))
        if not np.isfinite(yaw):
            raise ValueError("environment human_yaw_deg must be finite")
        self._position = position_array
        self._rotation = np.asarray(
            (
                (np.cos(yaw), -np.sin(yaw), 0.0),
                (np.sin(yaw), np.cos(yaw), 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=float,
        )
        initial = np.asarray(base.vertices_at(float(self.times[0])), dtype=float)
        self._center = 0.5 * (
            np.min(initial, axis=0) + np.max(initial, axis=0)
        )
        self.faces = np.asarray(base.faces, dtype=np.uint32)

    @property
    def vertices(self) -> np.ndarray:
        return np.stack(
            [self.vertices_at(float(value)) for value in self.times],
            axis=0,
        )

    @property
    def vertex_count(self) -> int:
        return int(self._base.vertex_count)

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])

    def vertices_at(self, time: float) -> np.ndarray:
        vertices = np.asarray(self._base.vertices_at(float(time)), dtype=float)
        vertices = (vertices - self._center) @ self._rotation.T
        vertices += self._position
        return np.asarray(vertices, dtype=np.float32)

    def max_vertex_displacement(
        self,
        time: float,
        reference_vertices: np.ndarray,
    ) -> float:
        vertices = self.vertices_at(time)
        reference = np.asarray(reference_vertices, dtype=np.float32)
        if reference.shape != vertices.shape:
            raise ValueError("reference_vertices shape mismatch")
        return float(np.max(np.linalg.norm(vertices - reference, axis=1)))


def _validate_motion_acquisition_coverage(clip: BenchmarkClip, motion) -> None:
    """Require motion coverage through the final physical chirp in each frame."""

    times = np.asarray(motion.times, dtype=float).reshape(-1)
    if times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("AMASS motion timeline must be non-empty and finite")
    fmcw = clip.sensor_config.fmcw
    slots_per_loop = int(fmcw.num_tx) if _clip_uses_tdm_virtual_adc(clip) else 1
    physical_chirps = int(fmcw.num_chirps_per_frame) * slots_per_loop
    final_chirp_offset_s = max(0, physical_chirps - 1) * float(
        fmcw.chirp_repetition_time
    )
    required_stop_s = float(np.max(clip.motion_times_s)) + final_chirp_offset_s
    tolerance = max(1.0e-12, abs(required_stop_s) * 1.0e-12)
    if required_stop_s > float(times[-1]) + tolerance:
        raise ValueError(
            "AMASS motion does not cover the full radar acquisition: "
            f"motion ends at {float(times[-1]):.9g}s but the final physical "
            f"chirp requires {required_stop_s:.9g}s"
        )


def _hybrid_static_env_po_config(
    mobility_mode: str,
    calibration_mode: HybridPOCalibrationMode = "rt_first_pose",
):
    if mobility_mode != "hybrid_static_env_po":
        return None
    from mmWaveRadar.simulation import (
        HybridStaticEnvPOConfig,
        HumanPOMobilityConfig,
        POCalibrationConfig,
    )

    return HybridStaticEnvPOConfig(
        coupling_enabled=True,
        human_po_config=HumanPOMobilityConfig(
            calibration=POCalibrationConfig(mode=calibration_mode),
        ),
    )


def _load_amass_motion_sequence(path: Path, config: BenchmarkRunConfig):
    from mmWaveRadar.targets import AMASSSMPLMotionSequence

    with np.load(path, allow_pickle=False) as data:
        required = ("poses", "trans", "betas", "times", "faces")
        missing = [name for name in required if name not in data.files]
        if missing:
            raise ValueError(f"{path} missing required SMPL arrays: {missing}")

        poses = np.array(data["poses"], dtype=np.float32, copy=True)
        betas = np.array(data["betas"], dtype=np.float32, copy=True)
        translations = np.array(data["trans"], dtype=np.float32, copy=True)
        faces = np.array(data["faces"], dtype=np.uint32, copy=True)
        timing_key = "bundle_times" if "bundle_times" in data.files else "times"
        times = np.array(data[timing_key], dtype=float, copy=True)
        npz_model_dir = _optional_npz_string(data, "smpl_model_dir")
        gender = _optional_npz_string(data, "gender") or "neutral"
        model_type = _optional_npz_string(data, "model_type") or "smpl"
        axes = (
            np.array(data["output_axes"], dtype=np.int64, copy=True)
            if "output_axes" in data.files
            else None
        )
        signs = (
            np.array(data["output_signs"], dtype=np.float32, copy=True)
            if "output_signs" in data.files
            else None
        )

    model_dir = config.smpl_model_dir or npz_model_dir
    if not model_dir:
        raise ValueError(
            f"{path} requires smpl_model_dir; pass it in config or set "
            "MMWAVE_SMPL_MODEL_DIR"
        )
    base = AMASSSMPLMotionSequence(
        poses=poses,
        betas=betas,
        translations=translations,
        faces=faces,
        times=times,
        smpl_model_dir=str(model_dir),
        model_type=str(model_type),
        gender=str(gender),
    )
    return _TransformedSMPLMotionSequence(base, axes=axes, signs=signs)


class _TransformedSMPLMotionSequence:
    """SMPL mesh sequence with optional axis permutation and sign flips."""

    def __init__(self, base, *, axes: np.ndarray | None, signs: np.ndarray | None):
        from mmWaveRadar.targets.mesh import ensure_outward_face_winding

        self._base = base
        self._axes = axes
        self._signs = signs
        self.times = np.asarray(base.times, dtype=float)
        faces = np.asarray(base.faces, dtype=np.uint32)
        if self.times.size and hasattr(base, "vertices_at"):
            faces = ensure_outward_face_winding(
                self.vertices_at(float(self.times[0])),
                faces,
            )
        self.faces = faces

    @property
    def vertices(self) -> np.ndarray:
        """Transformed vertices sampled at all source sequence times."""

        return np.stack([self.vertices_at(float(t)) for t in self.times], axis=0)

    @property
    def vertex_count(self) -> int:
        """Number of vertices per transformed frame."""

        return int(self._base.vertex_count)

    @property
    def face_count(self) -> int:
        """Number of triangular faces in the transformed mesh."""

        return int(self.faces.shape[0])

    def vertices_at(self, time: float) -> np.ndarray:
        """Returns transformed vertices at ``time`` in seconds."""

        vertices = self._base.vertices_at(float(time))
        if self._axes is not None:
            vertices = vertices[..., self._axes]
        if self._signs is not None:
            vertices = vertices * self._signs
        return np.asarray(vertices, dtype=np.float32)

    def max_vertex_displacement(self, time: float, reference_vertices: np.ndarray) -> float:
        """Returns the largest transformed displacement from ``reference_vertices``."""

        vertices = self.vertices_at(time)
        ref = np.asarray(reference_vertices, dtype=np.float32)
        if ref.shape != vertices.shape:
            raise ValueError("reference_vertices shape mismatch")
        return float(np.max(np.linalg.norm(vertices - ref, axis=1)))


def _optional_npz_string(data, key: str) -> str | None:
    if key not in data.files:
        return None
    value = data[key]
    if np.asarray(value).shape == ():
        return str(value.item())
    return str(value)


def _compute_maps(
    real_adc: np.ndarray,
    sim_adcs: dict[str, np.ndarray],
    fmcw,
    *,
    primary_label: str,
    remove_mean_clutter: bool,
):
    real_rt_power, rt_ranges_m = range_time_map(
        real_adc,
        fmcw=fmcw,
        remove_mean_clutter=remove_mean_clutter,
    )
    real_rd, rd_ranges_m, velocities_mps = framewise_range_doppler_map(
        real_adc,
        fmcw=fmcw,
        num_tx=fmcw.num_tx,
        remove_mean_clutter=remove_mean_clutter,
    )
    maps = {
        "real_range_time_power": real_rt_power,
        "range_time_ranges_m": rt_ranges_m,
        "real_range_doppler_power": np.sum(np.abs(real_rd) ** 2, axis=-1),
        "range_doppler_ranges_m": rd_ranges_m,
        "range_doppler_velocities_mps": velocities_mps,
    }
    for label, adc in sim_adcs.items():
        label_rt_power, _ = range_time_map(
            adc,
            fmcw=fmcw,
            remove_mean_clutter=remove_mean_clutter,
        )
        label_rd, _, _ = framewise_range_doppler_map(
            adc,
            fmcw=fmcw,
            num_tx=fmcw.num_tx,
            remove_mean_clutter=remove_mean_clutter,
        )
        maps[f"{label}_range_time_power"] = label_rt_power
        rd_power = np.sum(
            np.abs(label_rd) ** 2,
            axis=-1,
        )
        maps[f"{label}_range_doppler_power"] = rd_power
    maps["sim_range_time_power"] = maps[f"{primary_label}_range_time_power"]
    maps["sim_range_doppler_power"] = maps[
        f"{primary_label}_range_doppler_power"
    ]
    return maps


def _compute_angle_fft_maps(
    real_adc: np.ndarray,
    sim_adcs: dict[str, np.ndarray],
    fmcw,
    virtual_positions_lambda: np.ndarray | None,
    range_time_maps: dict[str, np.ndarray],
    *,
    primary_label: str,
) -> dict[str, np.ndarray]:
    """Computes one-bin geometry-aware angle maps using range-time peak selection."""

    if virtual_positions_lambda is None:
        return {}
    positions = np.asarray(virtual_positions_lambda, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(
            "virtual_channel_positions_lambda must have shape [channels, 2]"
        )
    if positions.shape[0] < 2 or np.allclose(positions[:, 1], positions[0, 1]):
        return {
            "angle_fft_skipped": np.asarray(
                "subarray_has_no_vertical_aperture"
            ),
            "angle_fft_virtual_positions_lambda": positions,
        }
    frames = int(np.asarray(real_adc).shape[0])
    ranges_m = np.asarray(range_time_maps["range_time_ranges_m"], dtype=float)
    real_bins = _range_time_peak_bins(
        range_time_maps["real_range_time_power"],
        frames=frames,
    )
    sim_peak_bins = {
        label: _range_time_peak_bins(
            range_time_maps[f"{label}_range_time_power"],
            frames=frames,
        )
        for label in sim_adcs
    }
    sim_bins, sim_sources = _shared_simulation_angle_fft_bins(
        real_bins,
        sim_peak_bins,
        primary_label=primary_label,
    )

    real_power, u_axis, v_axis, real_ranges_m, real_bins, fft_size = (
        _angle_fft_power_from_adc(
            real_adc,
            fmcw,
            positions,
            real_bins,
        )
    )
    maps = {
        "real_angle_fft_power": real_power,
        "angle_fft_u": u_axis,
        "angle_fft_v": v_axis,
        "angle_fft_selection_mode": np.asarray(
            "single_range_time_peak_bin_shared_sim_closest_to_real"
        ),
        "angle_fft_method": np.asarray(
            "geometry_aware_matched_filter_iwr6843aop_virtual_positions"
        ),
        "angle_fft_range_gate_m": np.stack(
            [real_ranges_m, ranges_m[np.asarray(sim_bins, dtype=np.int64)]],
            axis=1,
        ),
        "real_angle_fft_range_bin": np.asarray(real_bins, dtype=np.int64),
        "real_angle_fft_range_m": np.asarray(real_ranges_m, dtype=float),
        "sim_angle_fft_range_bin": np.asarray(sim_bins, dtype=np.int64),
        "sim_angle_fft_range_m": ranges_m[np.asarray(sim_bins, dtype=np.int64)],
        "angle_fft_sim_selection_source": np.asarray(sim_sources),
        "angle_fft_virtual_positions_lambda": positions,
        "angle_fft_fft_size": np.asarray(fft_size, dtype=np.int64),
    }
    for label, peak_bins in sim_peak_bins.items():
        peak_bins_arr = np.asarray(peak_bins, dtype=np.int64)
        maps[f"{label}_angle_fft_range_time_peak_bin"] = peak_bins_arr
        maps[f"{label}_angle_fft_range_time_peak_m"] = ranges_m[peak_bins_arr]
    for label, adc in sim_adcs.items():
        label_power, _, _, label_ranges_m, label_bins, _ = (
            _angle_fft_power_from_adc(
                adc,
                fmcw,
                positions,
                sim_bins,
                fft_size=fft_size,
            )
        )
        maps[f"{label}_angle_fft_power"] = label_power
        maps[f"{label}_angle_fft_range_bin"] = np.asarray(
            label_bins,
            dtype=np.int64,
        )
        maps[f"{label}_angle_fft_range_m"] = np.asarray(
            label_ranges_m,
            dtype=float,
        )
    maps["sim_angle_fft_power"] = maps[f"{primary_label}_angle_fft_power"]
    return maps


def _range_time_peak_bins(
    range_time_power: np.ndarray,
    *,
    frames: int,
) -> np.ndarray:
    power = np.asarray(range_time_power, dtype=float)
    if power.ndim != 2:
        raise ValueError("range_time_power must have shape [slow_time, range]")
    if frames <= 0 or power.shape[0] % int(frames) != 0:
        raise ValueError("range_time_power slow-time axis must divide into frames")
    per_frame = power.reshape((int(frames), -1, power.shape[-1]))
    return np.argmax(np.mean(per_frame, axis=1), axis=1).astype(np.int64)


def _shared_simulation_angle_fft_bins(
    real_bins: np.ndarray,
    sim_peak_bins: dict[str, np.ndarray],
    *,
    primary_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    real = np.asarray(real_bins, dtype=np.int64).reshape(-1)
    if not sim_peak_bins:
        raise ValueError("At least one simulated ADC is required for angle FFT")
    preferred_labels = [label for label in ("po", "rt") if label in sim_peak_bins]
    labels = preferred_labels or list(sim_peak_bins)
    selected_bins = np.empty_like(real)
    selected_sources: list[str] = []
    for frame_index, real_bin in enumerate(real):
        best_label = min(
            labels,
            key=lambda label: (
                abs(int(sim_peak_bins[label][frame_index]) - int(real_bin)),
                0 if label == primary_label else 1,
                label,
            ),
        )
        selected_bins[frame_index] = int(sim_peak_bins[best_label][frame_index])
        selected_sources.append(best_label)
    return selected_bins, np.asarray(selected_sources)


def _angle_fft_power_from_adc(
    adc: np.ndarray,
    fmcw,
    virtual_positions_lambda: np.ndarray,
    selected_range_bins: np.ndarray,
    *,
    fft_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    x = np.asarray(adc)
    if x.ndim != 4:
        raise ValueError("adc must have shape [frames, chirps, samples, channels]")
    frames, chirps, samples, channels = x.shape
    positions = np.asarray(virtual_positions_lambda, dtype=float)
    if positions.shape != (channels, 2):
        raise ValueError("ADC channel axis does not match virtual array geometry")
    rt, ranges_m = range_fft(
        x.reshape((frames * chirps, samples, channels)),
        fmcw=fmcw,
    )
    rt = rt.reshape((frames, chirps, rt.shape[1], channels))
    if fft_size is None:
        fft_size = (64, 128)
    selected_bins = np.asarray(selected_range_bins, dtype=np.int64).reshape(-1)
    if selected_bins.size != frames:
        raise ValueError("selected_range_bins length must match frame count")
    powers = []
    selected_ranges = []
    used_bins = []
    for frame_index in range(frames):
        range_bin = int(np.clip(selected_bins[frame_index], 0, rt.shape[2] - 1))
        snapshots = rt[frame_index, :, range_bin:range_bin + 1, :].reshape(
            (-1, channels)
        )
        angle = psf_cross_correlation_image(
            snapshots,
            positions,
            grid_size=fft_size,
            snapshot_mode="mean_power",
            mask_invalid_directions=True,
        )
        power = np.asarray(angle["dirty_image"], dtype=float)
        valid_mask = angle.get("valid_mask")
        if valid_mask is not None:
            power = np.where(np.asarray(valid_mask, dtype=bool), power, 0.0)
        powers.append(power)
        selected_ranges.append(float(ranges_m[range_bin]))
        used_bins.append(range_bin)
    return (
        np.asarray(powers, dtype=float),
        np.asarray(angle["u"], dtype=float),
        np.asarray(angle["v"], dtype=float),
        np.asarray(selected_ranges, dtype=float),
        np.asarray(used_bins, dtype=np.int64),
        (int(fft_size[0]), int(fft_size[1])),
    )


def _write_plots(
    out_dir: Path,
    maps: dict[str, np.ndarray],
    *,
    simulation_modes: tuple[tuple[str, str], ...] | None = None,
) -> None:
    """Write comparison plots whose names and labels identify every mode."""

    if simulation_modes is None:
        if "po_range_time_power" in maps and "rt_range_time_power" in maps:
            simulation_modes = (("po", "po"), ("rt", "rt"))
        else:
            simulation_modes = (("sim", "sim"),)
    if not simulation_modes:
        raise ValueError("simulation_modes must contain at least one entry")

    first_key, first_mode = simulation_modes[0]
    suffix = _plot_comparison_suffix(simulation_modes)
    first_display = _plot_mode_display_name(first_mode)
    additional_rt = {
        _plot_mode_display_name(mode): maps[f"{key}_range_time_power"]
        for key, mode in simulation_modes[1:]
    }
    plot_range_time_comparison(
        out_dir / f"range_time_{suffix}.png",
        maps["real_range_time_power"],
        maps[f"{first_key}_range_time_power"],
        maps["range_time_ranges_m"],
        simulated_label=first_display,
        additional_simulated_powers=additional_rt,
    )
    plot_range_profile_comparison(
        out_dir / f"range_profile_{suffix}.png",
        maps["real_range_time_power"],
        maps[f"{first_key}_range_time_power"],
        maps["range_time_ranges_m"],
        simulated_label=first_display,
        additional_simulated_powers=additional_rt,
    )

    additional_rd = {
        _plot_mode_display_name(mode): maps[f"{key}_range_doppler_power"]
        for key, mode in simulation_modes[1:]
    }
    plot_range_doppler_comparison(
        out_dir / f"range_doppler_{suffix}.png",
        maps["real_range_doppler_power"],
        maps[f"{first_key}_range_doppler_power"],
        maps["range_doppler_ranges_m"],
        maps["range_doppler_velocities_mps"],
        simulated_label=first_display,
        additional_simulated_powers=additional_rd,
    )

    if "real_angle_fft_power" in maps:
        additional_angle = {
            _angle_fft_label(mode, maps, key): maps[f"{key}_angle_fft_power"]
            for key, mode in simulation_modes[1:]
        }
        plot_angle_fft_comparison(
            out_dir / f"angle_fft_{suffix}.png",
            maps["real_angle_fft_power"],
            maps[f"{first_key}_angle_fft_power"],
            maps["angle_fft_u"],
            maps["angle_fft_v"],
            real_label=_angle_fft_label("Real", maps, "real"),
            simulated_label=_angle_fft_label(first_display, maps, first_key),
            additional_simulated_powers=additional_angle,
        )


def _plot_comparison_suffix(
    simulation_modes: tuple[tuple[str, str], ...],
) -> str:
    """Return a stable filename suffix containing every simulated mode."""

    tokens = [_plot_mode_filename_token(mode) for _, mode in simulation_modes]
    return "real_vs_" + "_vs_".join(tokens)


def _plot_mode_filename_token(mode: str) -> str:
    """Normalize one mobility-mode name into a portable filename token."""

    token = "".join(
        character.lower() if character.isalnum() else "_"
        for character in str(mode)
    ).strip("_")
    return token or "sim"


def _plot_mode_display_name(mode: str) -> str:
    """Return the human-readable plot label for a simulation mode."""

    return "Simulated" if mode == "sim" else str(mode)


def _angle_fft_label(prefix: str, maps: dict[str, np.ndarray], label: str) -> str:
    range_key = f"{label}_angle_fft_range_m"
    bin_key = f"{label}_angle_fft_range_bin"
    if range_key not in maps:
        return prefix
    ranges = np.asarray(maps[range_key], dtype=float).reshape(-1)
    if ranges.size == 0:
        return prefix
    if bin_key not in maps:
        return f"{prefix} @ {ranges[0]:.3f} m"
    bins = np.asarray(maps[bin_key], dtype=np.int64).reshape(-1)
    if bins.size == 0:
        return f"{prefix} @ {ranges[0]:.3f} m"
    return f"{prefix} bin {int(bins[0])} @ {ranges[0]:.3f} m"


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(value) ** 2)))
