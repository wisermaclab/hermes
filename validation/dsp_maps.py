# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""DSP maps and comparison plot generation."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from mmWaveRadar.dsp import (
    framewise_range_doppler_map,
    psf_cross_correlation_image,
    range_fft,
    range_time_map,
)
from mmWaveRadar.naming import SOLVER_LABELS
from .results import (
    plot_angle_fft_comparison,
    plot_range_doppler_comparison,
    plot_range_profile_comparison,
    plot_range_time_comparison,
)


def compute_maps(
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
    maps["sim_range_doppler_power"] = maps[f"{primary_label}_range_doppler_power"]
    return maps


def selected_phase_signs(clip, selection):
    hardware = clip.sensor_config.full_hardware()
    signs = hardware.virtual_channel_phase_signs
    if signs is None:
        return None
    lookup = {
        (int(t) + 1, int(r) + 1): sign
        for t, r, sign in zip(hardware.virtual_tx_indices(), hardware.virtual_rx_indices(), signs)
    }
    return np.asarray(
        [
            lookup[(int(t), int(r))]
            for t, r in zip(
                selection.channel_tx_indices_1based, selection.channel_rx_indices_1based
            )
        ]
    )


def compute_angle_fft_maps(
    real_adc: np.ndarray,
    sim_adcs: dict[str, np.ndarray],
    fmcw,
    virtual_positions_lambda: np.ndarray | None,
    range_time_maps: dict[str, np.ndarray],
    *,
    primary_label: str,
    phase_signs=None,
) -> dict[str, np.ndarray]:
    """Computes one-bin geometry-aware angle maps using range-time peak selection."""

    if virtual_positions_lambda is None:
        return {}
    positions = np.asarray(virtual_positions_lambda, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("virtual_channel_positions_lambda must have shape [channels, 2]")
    if positions.shape[0] < 2 or np.allclose(positions[:, 1], positions[0, 1]):
        return {
            "angle_fft_skipped": np.asarray("subarray_has_no_vertical_aperture"),
            "angle_fft_virtual_positions_lambda": positions,
        }
    frames = int(np.asarray(real_adc).shape[0])
    ranges_m = np.asarray(range_time_maps["range_time_ranges_m"], dtype=float)
    real_bins = range_time_peak_bins(
        range_time_maps["real_range_time_power"],
        frames=frames,
    )
    sim_peak_bins = {
        label: range_time_peak_bins(
            range_time_maps[f"{label}_range_time_power"],
            frames=frames,
        )
        for label in sim_adcs
    }
    sim_bins, sim_sources = shared_simulation_angle_fft_bins(
        real_bins,
        sim_peak_bins,
        primary_label=primary_label,
    )

    real_power, u_axis, v_axis, real_ranges_m, real_bins, fft_size = angle_fft_power_from_adc(
        real_adc,
        fmcw,
        positions,
        real_bins,
        phase_signs=phase_signs,
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
        label_power, _, _, label_ranges_m, label_bins, _ = angle_fft_power_from_adc(
            adc,
            fmcw,
            positions,
            sim_bins,
            phase_signs=phase_signs,
            fft_size=fft_size,
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


def range_time_peak_bins(
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


def shared_simulation_angle_fft_bins(
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


def angle_fft_power_from_adc(
    adc: np.ndarray,
    fmcw,
    virtual_positions_lambda: np.ndarray,
    selected_range_bins: np.ndarray,
    *,
    fft_size: tuple[int, int] | None = None,
    phase_signs=None,
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
    if phase_signs is not None:
        from mmWaveRadar.simulation.point_targets import calibrate_range_cube_channel_phase

        rt = calibrate_range_cube_channel_phase(rt, phase_signs=phase_signs)
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
        snapshots = rt[frame_index, :, range_bin : range_bin + 1, :].reshape((-1, channels))
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


def write_plots(
    out_dir: Path,
    maps: dict[str, np.ndarray],
    *,
    simulated_solvers: tuple[tuple[str, str], ...] | None = None,
) -> None:
    """Write comparison plots whose names and labels identify every mode."""

    if simulated_solvers is None:
        if "po_range_time_power" in maps and "rt_range_time_power" in maps:
            simulated_solvers = (("po", "po"), ("rt", "rt"))
        else:
            simulated_solvers = (("sim", "sim"),)
    if not simulated_solvers:
        raise ValueError("simulated_solvers must contain at least one entry")

    first_key, first_mode = simulated_solvers[0]
    suffix = plot_comparison_suffix(simulated_solvers)
    first_display = plot_mode_display_name(first_mode)
    additional_rt = {
        plot_mode_display_name(mode): maps[f"{key}_range_time_power"]
        for key, mode in simulated_solvers[1:]
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
        plot_mode_display_name(mode): maps[f"{key}_range_doppler_power"]
        for key, mode in simulated_solvers[1:]
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
            angle_fft_label(mode, maps, key): maps[f"{key}_angle_fft_power"]
            for key, mode in simulated_solvers[1:]
        }
        plot_angle_fft_comparison(
            out_dir / f"angle_fft_{suffix}.png",
            maps["real_angle_fft_power"],
            maps[f"{first_key}_angle_fft_power"],
            maps["angle_fft_u"],
            maps["angle_fft_v"],
            real_label=angle_fft_label("Real", maps, "real"),
            simulated_label=angle_fft_label(first_display, maps, first_key),
            additional_simulated_powers=additional_angle,
        )


def plot_comparison_suffix(
    simulated_solvers: tuple[tuple[str, str], ...],
) -> str:
    """Return a stable filename suffix containing every simulated mode."""

    tokens = [plot_mode_filename_token(mode) for _, mode in simulated_solvers]
    return "real_vs_" + "_vs_".join(tokens)


def plot_mode_filename_token(mode: str) -> str:
    """Normalize one solver name into a portable filename token."""

    token = "".join(
        character.lower() if character.isalnum() else "_" for character in str(mode)
    ).strip("_")
    return token or "sim"


def plot_mode_display_name(mode: str) -> str:
    """Return the human-readable plot label for a solver."""

    if mode == "sim":
        return "Simulated"
    return SOLVER_LABELS.get(str(mode), str(mode))


def angle_fft_label(prefix: str, maps: dict[str, np.ndarray], label: str) -> str:
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


def rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(value) ** 2)))
