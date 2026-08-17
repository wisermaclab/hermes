# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Benchmark human-only PO on a mesh-sequence NPZ."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    """Finds the simulator repository root from this benchmark script."""

    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "src" / "mmWaveRadar").exists():
            return parent
    raise RuntimeError("Could not locate mmWave simulator repository root")


REPO_ROOT = _repo_root()
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "mmwave-radar-mpl"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / "mmwave-radar-cache"),
)


from sionna.rt import load_scene  # noqa: E402

from mmWaveRadar.dsp import (  # noqa: E402
    doppler_time_spectrum,
    range_doppler_map,
    range_time_map,
)
from mmWaveRadar.materials import human_skin_material  # noqa: E402
from mmWaveRadar.radar import FMCWConfig, RadarHardware, RadarSensor  # noqa: E402
from mmWaveRadar.simulation import (  # noqa: E402
    HumanPOMobilityConfig,
    MmWaveRadarSimulator,
    POIncrementalSweepSetting,
    cube_matches_run,
    flatten_adc_for_fft,
    label_slug,
    load_radar_cube_npz,
    runtime_summary,
    run_incremental_po_sweep,
    save_radar_cube_npz,
)
from mmWaveRadar.targets import MeshSequence, MeshTarget  # noqa: E402


DEFAULT_SETTINGS = [
    POIncrementalSweepSetting("k8 nominal n5 d2mm a5pct", 8, 5.0, 0.002, 0.05),
    POIncrementalSweepSetting("k16 nominal n5 d2mm a5pct", 16, 5.0, 0.002, 0.05),
    POIncrementalSweepSetting("k32 nominal n5 d2mm a5pct", 32, 5.0, 0.002, 0.05),
    POIncrementalSweepSetting("k64 nominal n5 d2mm a5pct", 64, 5.0, 0.002, 0.05),
    POIncrementalSweepSetting("k32 loose n15 d10mm a20pct", 32, 15.0, 0.010, 0.20),
    POIncrementalSweepSetting("k64 loose n20 d20mm a30pct", 64, 20.0, 0.020, 0.30),
    POIncrementalSweepSetting("k128 aggressive n30 d50mm a50pct", 128, 30.0, 0.050, 0.50),
]


def _orientation_toward(target_point, radar_position):
    """Matches Sionna RadioDevice.look_at: local +x points toward target_point."""

    direction = np.asarray(target_point, dtype=float) - np.asarray(
        radar_position, dtype=float)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        raise ValueError("radar position and look-at target are identical")
    direction /= norm
    theta = np.arccos(np.clip(direction[2], -1.0, 1.0))
    phi = np.arctan2(direction[1], direction[0])
    return (float(phi), float(theta - 0.5 * np.pi), 0.0)


def _count_radar_frames_in_mesh_sequence(mesh_sequence, fmcw):
    times = np.asarray(mesh_sequence.times, dtype=float)
    if times.size == 0:
        raise ValueError("mesh sequence must contain at least one time sample")
    sequence_span_s = float(times[-1] - times[0])
    last_chirp_offset_s = (fmcw.num_chirps_per_frame - 1) * (
        fmcw.chirp_repetition_time)
    if sequence_span_s <= last_chirp_offset_s:
        return 1
    return max(
        1,
        int(np.floor((sequence_span_s - last_chirp_offset_s)
                     / fmcw.frame_period)) + 1,
    )


def _parse_vec3(value: str, *, name: str) -> tuple[float, float, float]:
    try:
        parts = [float(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{name} must be comma-separated floats, e.g. 0,0,1") from exc
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"{name} must have exactly three comma-separated values")
    return (parts[0], parts[1], parts[2])


def _parse_setting(value: str) -> POIncrementalSweepSetting:
    parts = value.split(":")
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "--setting must be label:K:normal_deg:centroid_m:area_rel")
    label = parts[0].strip()
    if not label:
        raise argparse.ArgumentTypeError("setting label must not be empty")
    try:
        return POIncrementalSweepSetting(
            label=label,
            visibility_refresh_chirps=int(parts[1]),
            normal_threshold_deg=float(parts[2]),
            centroid_displacement_threshold_m=float(parts[3]),
            area_relative_threshold=float(parts[4]),
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--setting must be label:K:normal_deg:centroid_m:area_rel") from exc


def _load_mesh_sequence(path: Path) -> MeshSequence:
    return MeshSequence.from_npz(path)


def _default_radar_pose(mesh_sequence: MeshSequence, clearance_m: float):
    vertices = mesh_sequence.vertices_at(float(mesh_sequence.times[0]))
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = 0.5 * (mins + maxs)
    look_at = np.array(
        [center[0], center[1], mins[2] + 0.65 * (maxs[2] - mins[2])],
        dtype=float,
    )
    radar_position = look_at + np.array([-float(clearance_m), 0.0, 0.0])
    radar_orientation = _orientation_toward(look_at, radar_position)
    return radar_position, radar_orientation, look_at


def _build_radar(args, mesh_sequence: MeshSequence):
    hardware = RadarHardware.from_positions(
        tx_positions=[[0.0, 0.0, 0.0]],
        rx_positions=[[0.0, 0.0, 0.0]],
        name="single-virtual-channel",
    )
    fmcw = FMCWConfig(
        carrier_frequency=args.carrier_frequency,
        slope=args.slope,
        chirp_duration=args.chirp_duration,
        chirp_repetition_time=args.chirp_repetition_time,
        sampling_frequency=args.sampling_frequency,
        num_adc_samples=args.num_adc_samples,
        num_chirps_per_frame=args.num_chirps_per_frame,
        frame_period=args.frame_period,
        num_tx=hardware.num_tx,
    )
    if args.radar_position is None:
        radar_position, radar_orientation, look_at = _default_radar_pose(
            mesh_sequence, args.radar_clearance_m)
    else:
        radar_position = np.asarray(args.radar_position, dtype=float)
        if args.radar_look_at is not None:
            look_at = np.asarray(args.radar_look_at, dtype=float)
            radar_orientation = _orientation_toward(look_at, radar_position)
        elif args.radar_orientation is not None:
            radar_orientation = tuple(args.radar_orientation)
            look_at = np.asarray(mesh_sequence.vertices_at(
                float(mesh_sequence.times[0])).mean(axis=0), dtype=float)
        else:
            raise ValueError(
                "--radar-position requires --radar-look-at or "
                "--radar-orientation")
    radar = RadarSensor(
        name="radar",
        position=tuple(float(v) for v in radar_position),
        orientation=tuple(float(v) for v in radar_orientation),
        hardware=hardware,
        fmcw=fmcw,
    )
    return radar, np.asarray(look_at, dtype=float)


def _run_human_only_po(
    *,
    label: str,
    config: HumanPOMobilityConfig,
    radar: RadarSensor,
    target: MeshTarget,
    num_frames: int,
    args,
):
    scene = load_scene()
    sim_kwargs = {
        "mobility_mode": "human_only_po",
        "progress": args.progress,
        "human_po_config": config,
    }
    if args.compute_backend is not None:
        sim_kwargs["compute_backend"] = args.compute_backend
    if args.compute_precision is not None:
        sim_kwargs["compute_precision"] = args.compute_precision
    if args.adc_compute_backend is not None:
        sim_kwargs["adc_compute_backend"] = args.adc_compute_backend
    if args.adc_compute_precision is not None:
        sim_kwargs["adc_compute_precision"] = args.adc_compute_precision
    simulator = MmWaveRadarSimulator(**sim_kwargs)
    run_target = MeshTarget(
        name=f"{target.name}_{label_slug(label)}",
        mesh_sequence=target.mesh_sequence,
        material=target.material,
    )
    start = time.perf_counter()
    cube = simulator.run(scene, radar, [run_target], num_frames=num_frames)
    elapsed = time.perf_counter() - start
    num_chirps = int(np.prod(cube.adc.shape[:2]))
    print(
        f"{label}: {elapsed:.2f} s "
        f"({elapsed / num_chirps:.4f} s/chirp)",
        flush=True,
    )
    return cube, elapsed, num_chirps


def _relative_db(power_like):
    power_like = np.maximum(power_like, 1e-30)
    return 10.0 * np.log10(power_like / np.max(power_like))


def _frames_for_duration(times, duration_s):
    return max(
        1,
        min(
            int(times.shape[0]),
            int(np.searchsorted(times[:, 0], duration_s, side="left")),
        ),
    )


def _unique_results(results):
    seen = set()
    unique = []
    for result in results:
        label = result["label"]
        if label in seen:
            continue
        seen.add(label)
        unique.append(result)
    return unique


def _write_summary(output_dir: Path, metadata: dict, rows: list[dict]):
    summary = dict(metadata)
    summary["results"] = rows
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote summary JSON: {json_path}", flush=True)
    print(f"Wrote summary CSV: {csv_path}", flush=True)


def _plot_results(args, output_dir, fmcw, baseline, baseline_wall_time_s,
                  baseline_num_chirps, results):
    import matplotlib

    matplotlib.use(args.matplotlib_backend)
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    rows = [result for result in results]
    labels = [row["label"] for row in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
    axes[0].bar(x, [row["speedup"] for row in rows], color="#4c78a8")
    axes[0].set_ylabel("speedup vs full PO")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].scatter(
        [row["range_profile_relative_error"] for row in rows],
        [row["speedup"] for row in rows],
        s=45,
        color="#54a24b",
    )
    for row in rows:
        axes[1].annotate(
            row["label"],
            (row["range_profile_relative_error"], row["speedup"]),
            fontsize=7,
            xytext=(4, 3),
            textcoords="offset points",
        )
    axes[1].set_xlabel("range-profile relative L2 error")
    axes[1].set_ylabel("speedup vs full PO")
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(x, [row["recomputed_mean"] for row in rows], color="#f58518")
    axes[2].set_ylabel("mean recomputed faces/chirp")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=35, ha="right")
    axes[2].grid(True, axis="y", alpha=0.3)
    fig.savefig(output_dir / "po_sweep_tradeoff.png", dpi=args.plot_dpi)
    plt.close(fig)

    selected = _unique_results([
        {
            "label": "Full PO",
            "cube": baseline,
            "wall_time_s": baseline_wall_time_s,
            "num_chirps": baseline_num_chirps,
        },
        min(rows, key=lambda row: abs(row["setting_K"] - 64)),
        max(rows, key=lambda row: row["speedup"]),
        min(rows, key=lambda row: row["range_profile_relative_error"]),
    ])

    range_time_plot_duration_s = min(
        args.plot_duration_s, float(baseline.times[-1, -1]))
    comparison = []
    for row in selected:
        frame_count = _frames_for_duration(row["cube"].times,
                                           range_time_plot_duration_s)
        comparison.append((row["label"], row["cube"], row["wall_time_s"],
                           frame_count))

    range_time_results = []
    for label, cube, wall_time_s, frame_count in comparison:
        rt_map, ranges_m = range_time_map(
            cube.adc[:frame_count],
            fmcw=fmcw,
            window="hann",
            nfft_mult=args.range_nfft_mult,
        )
        range_mask = ranges_m <= args.range_plot_limit_m
        slow_time_ms = cube.times[:frame_count].reshape(-1) * 1e3
        range_time_results.append((
            label,
            wall_time_s,
            _relative_db(rt_map[:, range_mask]),
            slow_time_ms,
            ranges_m[range_mask],
        ))

    fig, axes = plt.subplots(
        1,
        len(range_time_results),
        figsize=(5 * len(range_time_results), 4.5),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for ax, (label, wall_time_s, db_map, slow_time_ms, ranges_m) in zip(
            axes, range_time_results):
        im = ax.imshow(
            db_map.T,
            origin="lower",
            aspect="auto",
            extent=[slow_time_ms[0], slow_time_ms[-1],
                    ranges_m[0], ranges_m[-1]],
            vmin=args.plot_vmin_db,
            vmax=0,
            cmap="viridis",
        )
        ax.set_title(f"{label} range-time\n{wall_time_s:.1f} s total")
        ax.set_xlabel("slow time [ms]")
    axes[0].set_ylabel("range [m]")
    fig.colorbar(im, ax=axes, label="relative power [dB]")
    fig.savefig(output_dir / "po_range_time.png", dpi=args.plot_dpi)
    plt.close(fig)

    rd_frame_count = min(frame_count for _, _, _, frame_count in comparison)
    range_doppler_results = []
    for label, cube, _, _ in comparison:
        rd_cube, rd_ranges_m, velocities_mps = range_doppler_map(
            flatten_adc_for_fft(cube.adc[:rd_frame_count]),
            fmcw=fmcw,
            win_range="hann",
            win_doppler="hann",
        )
        rd_power = np.sum(np.abs(rd_cube) ** 2, axis=-1)
        rd_mask = rd_ranges_m <= args.range_plot_limit_m
        range_doppler_results.append((
            label,
            _relative_db(rd_power[:, rd_mask]),
            rd_ranges_m[rd_mask],
            velocities_mps,
        ))

    fig, axes = plt.subplots(
        1,
        len(range_doppler_results),
        figsize=(5 * len(range_doppler_results), 4.5),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for ax, (label, rd_db, rd_ranges_m, velocities_mps) in zip(
            axes, range_doppler_results):
        im = ax.imshow(
            rd_db,
            origin="lower",
            aspect="auto",
            extent=[rd_ranges_m[0], rd_ranges_m[-1],
                    velocities_mps[0], velocities_mps[-1]],
            vmin=args.plot_vmin_db,
            vmax=0,
            cmap="magma",
        )
        ax.set_title(f"{label} range-Doppler")
        ax.set_xlabel("range [m]")
    axes[0].set_ylabel("velocity [m/s]")
    fig.colorbar(im, ax=axes, label="relative power [dB]")
    fig.savefig(output_dir / "po_range_doppler.png", dpi=args.plot_dpi)
    plt.close(fig)

    doppler_time_results = []
    for label, cube, _, _ in comparison:
        dt_spectrum, dt_times_s, dt_velocities_mps, _ = doppler_time_spectrum(
            cube.adc[:rd_frame_count],
            fmcw=fmcw,
            win_range="hann",
            win_doppler="hann",
            nfft_mult=args.range_nfft_mult,
            window_chirps=fmcw.num_chirps_per_frame,
            hop_chirps=fmcw.num_chirps_per_frame,
            doppler_fft_size=fmcw.num_chirps_per_frame,
            slow_time_s=cube.times[:rd_frame_count].reshape(-1),
            range_limits_m=(0.0, args.range_plot_limit_m),
        )
        doppler_time_results.append((
            label,
            _relative_db(dt_spectrum),
            dt_times_s * 1e3,
            dt_velocities_mps,
        ))

    fig, axes = plt.subplots(
        1,
        len(doppler_time_results),
        figsize=(5 * len(doppler_time_results), 4.5),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for ax, (label, dt_db, dt_time_ms, dt_velocities_mps) in zip(
            axes, doppler_time_results):
        im = ax.imshow(
            dt_db.T,
            origin="lower",
            aspect="auto",
            extent=[dt_time_ms[0], dt_time_ms[-1],
                    dt_velocities_mps[0], dt_velocities_mps[-1]],
            vmin=args.plot_vmin_db,
            vmax=0,
            cmap="plasma",
        )
        ax.set_title(f"{label} Doppler-time")
        ax.set_xlabel("slow time [ms]")
    axes[0].set_ylabel("velocity [m/s]")
    fig.colorbar(im, ax=axes, label="relative power [dB]")
    fig.savefig(output_dir / "po_doppler_time.png", dpi=args.plot_dpi)
    plt.close(fig)

    print(f"Wrote plots to: {output_dir}", flush=True)


def _result_row(result):
    setting = result.setting
    return {
        "label": result.label,
        "setting_K": int(setting.visibility_refresh_chirps),
        "normal_threshold_deg": float(setting.normal_threshold_deg),
        "centroid_displacement_threshold_m": float(
            setting.centroid_displacement_threshold_m),
        "area_relative_threshold": float(setting.area_relative_threshold),
        "wall_time_s": float(result.wall_time_s),
        "num_chirps": int(result.num_chirps),
        "seconds_per_chirp": float(result.wall_time_s / result.num_chirps),
        "speedup": float(result.speedup),
        "adc_relative_error": float(result.adc_relative_error),
        "range_profile_relative_error": float(
            result.range_profile_relative_error),
        "path_count_delta_min": int(result.path_count_delta_min),
        "path_count_delta_mean": float(result.path_count_delta_mean),
        "path_count_delta_max": int(result.path_count_delta_max),
        "recomputed_mean": float(result.recomputed_mean),
        "recomputed_max": int(result.recomputed_max),
        "phase_updated_mean": float(result.phase_updated_mean),
        "full_refresh_count": int(result.full_refresh_count),
        "visibility_refresh_count": int(result.visibility_refresh_count),
        "cube_path": "" if result.path is None else str(result.path),
    }


def _single_result_row(label, cube, wall_time_s: float, num_chirps: int):
    row = {
        "label": str(label),
        "wall_time_s": float(wall_time_s),
        "num_chirps": int(num_chirps),
        "seconds_per_frame": float(wall_time_s / cube.adc.shape[0]),
        "seconds_per_chirp": float(wall_time_s / num_chirps),
        "adc_shape": list(cube.adc.shape),
        "mean_paths": float(np.mean(cube.metadata.path_counts)),
        "events": int(len(cube.metadata.events)),
    }
    if cube.metadata.runtime_profile_s is not None:
        row["runtime_profile_s"] = {
            key: float(value)
            for key, value in cube.metadata.runtime_profile_s.items()
        }
    if cube.metadata.runtime_profile_counts is not None:
        row["runtime_profile_counts"] = {
            key: int(value)
            for key, value in cube.metadata.runtime_profile_counts.items()
        }
    if cube.metadata.incremental_recomputed_face_counts is not None:
        recomputed = cube.metadata.incremental_recomputed_face_counts
        phase_updated = cube.metadata.incremental_phase_updated_face_counts
        full_refresh = cube.metadata.incremental_full_refresh
        visibility_refresh = cube.metadata.incremental_visibility_refresh
        row["incremental_recomputed_faces_min"] = int(np.min(recomputed))
        row["incremental_recomputed_faces_mean"] = float(np.mean(recomputed))
        row["incremental_recomputed_faces_max"] = int(np.max(recomputed))
        row["incremental_phase_updated_faces_min"] = int(np.min(phase_updated))
        row["incremental_phase_updated_faces_mean"] = float(np.mean(phase_updated))
        row["incremental_phase_updated_faces_max"] = int(np.max(phase_updated))
        row["incremental_full_refresh_fraction"] = (
            float(np.mean(full_refresh)) if full_refresh is not None else None
        )
        row["incremental_visibility_refresh_fraction"] = (
            float(np.mean(visibility_refresh))
            if visibility_refresh is not None else None
        )
    return row


def _single_po_config(args) -> HumanPOMobilityConfig:
    return HumanPOMobilityConfig(
        visibility_samples_per_face=args.visibility_samples_per_face,
        visibility_fade_chirps=args.visibility_fade_chirps,
        visibility_use_phase_center=True,
        adaptive_visibility_sampling=args.adaptive_visibility_sampling,
        adaptive_visibility_edge_margin=args.adaptive_visibility_edge_margin,
        incremental_update=args.incremental_update,
        incremental_visibility_refresh_chirps=(
            args.incremental_visibility_refresh_chirps),
        incremental_full_refresh_chirps=args.incremental_full_refresh_chirps,
        incremental_normal_threshold_deg=args.incremental_normal_threshold_deg,
        incremental_centroid_displacement_threshold_m=(
            args.incremental_centroid_displacement_threshold_m),
        incremental_area_relative_threshold=(
            args.incremental_area_relative_threshold),
    )


def _parse_args() -> argparse.Namespace:
    """Parses CLI controls for single-run and sweep PO benchmarks."""

    parser = argparse.ArgumentParser(
        description=(
            "Run PO-only human benchmarks on a precomputed mesh-sequence NPZ "
            "containing vertices, faces, and times."
        )
    )
    parser.add_argument("--mode", choices=("single", "sweep"), default="sweep")
    parser.add_argument("--mesh-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(tempfile.gettempdir()) / "mmwave_human_po")
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--reuse", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--setting", type=_parse_setting, action="append",
                        default=None,
                        help=("Override default sweep. Repeat as needed. "
                              "Format: label:K:normal_deg:centroid_m:area_rel"))

    parser.add_argument("--radar-clearance-m", type=float, default=0.50)
    parser.add_argument("--radar-position", type=lambda s: _parse_vec3(
        s, name="radar-position"), default=None)
    parser.add_argument("--radar-look-at", type=lambda s: _parse_vec3(
        s, name="radar-look-at"), default=None)
    parser.add_argument("--radar-orientation", type=lambda s: _parse_vec3(
        s, name="radar-orientation"), default=None)

    parser.add_argument("--visibility-samples-per-face", type=int, default=4)
    parser.add_argument("--visibility-fade-chirps", type=int, default=8)
    parser.add_argument("--adaptive-visibility-sampling",
                        action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--adaptive-visibility-edge-margin", type=float,
                        default=0.15)
    parser.add_argument("--incremental-update",
                        action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--incremental-visibility-refresh-chirps", type=int,
                        default=8)
    parser.add_argument("--incremental-full-refresh-chirps", type=int,
                        default=64)
    parser.add_argument("--incremental-normal-threshold-deg", type=float,
                        default=5.0)
    parser.add_argument("--incremental-centroid-displacement-threshold-m",
                        type=float, default=None)
    parser.add_argument("--incremental-area-relative-threshold", type=float,
                        default=None)
    parser.add_argument("--compute-backend", choices=("numpy", "torch", "auto"),
                        default="auto")
    parser.add_argument("--compute-precision", choices=("float64", "float32"),
                        default="float32")
    parser.add_argument("--adc-compute-backend",
                        choices=("numpy", "torch", "auto"), default=None)
    parser.add_argument("--adc-compute-precision",
                        choices=("float64", "float32"), default=None)

    parser.add_argument("--carrier-frequency", type=float, default=60e9)
    parser.add_argument("--slope", type=float, default=68e12)
    parser.add_argument("--chirp-duration", type=float, default=58e-6)
    parser.add_argument("--chirp-repetition-time", type=float, default=65e-6)
    parser.add_argument("--sampling-frequency", type=float, default=4.5e6)
    parser.add_argument("--num-adc-samples", type=int, default=225)
    parser.add_argument("--num-chirps-per-frame", type=int, default=64)
    parser.add_argument("--frame-period", type=float, default=50e-3)
    parser.add_argument("--plot-duration-s", type=float, default=5.0)
    parser.add_argument("--range-plot-limit-m", type=float, default=4.0)
    parser.add_argument("--range-nfft-mult", type=int, default=4)
    parser.add_argument("--plot-vmin-db", type=float, default=-50.0)
    parser.add_argument("--plot-dpi", type=int, default=150)
    parser.add_argument("--matplotlib-backend", default="Agg")
    return parser.parse_args()


def main() -> None:
    """Runs the human-only PO benchmark and writes summaries/plots."""

    args = _parse_args()
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "mmwave-radar-mpl"),
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Runtime:", runtime_summary(), flush=True)
    mesh_sequence = _load_mesh_sequence(args.mesh_npz)
    radar, look_at = _build_radar(args, mesh_sequence)
    target = MeshTarget(
        name="mesh_sequence_po",
        mesh_sequence=mesh_sequence,
        material=human_skin_material("human-skin-po-benchmark"),
    )

    max_sequence_frames = _count_radar_frames_in_mesh_sequence(
        mesh_sequence, radar.fmcw)
    duration_frames = max(1, int(np.floor(args.duration_s / radar.fmcw.frame_period)))
    num_frames = (
        min(max_sequence_frames, duration_frames)
        if args.num_frames is None
        else min(max_sequence_frames, int(args.num_frames))
    )
    last_chirp_time = radar.fmcw.chirp_time(
        num_frames - 1, radar.fmcw.num_chirps_per_frame - 1)

    print(f"Mesh NPZ: {args.mesh_npz}", flush=True)
    print(
        f"Mesh sequence: vertices={mesh_sequence.vertices.shape}, "
        f"faces={mesh_sequence.faces.shape}",
        flush=True,
    )
    print(
        f"Mesh time span: {float(mesh_sequence.times[0]):.3f} to "
        f"{float(mesh_sequence.times[-1]):.3f} s",
        flush=True,
    )
    print(
        f"Running {num_frames} frames x {radar.fmcw.num_chirps_per_frame} "
        f"chirps for ~{args.duration_s:.1f} s",
        flush=True,
    )
    print(f"Radar position: {np.asarray(radar.position)}", flush=True)
    print(f"Radar orientation: {np.asarray(radar.orientation)}", flush=True)
    print(f"Radar look-at: {look_at}", flush=True)

    metadata = {
        "runtime": runtime_summary(),
        "mode": args.mode,
        "mesh_npz": str(args.mesh_npz),
        "num_frames": int(num_frames),
        "num_chirps_per_frame": int(radar.fmcw.num_chirps_per_frame),
        "radar_position": [float(v) for v in radar.position],
        "radar_orientation": [float(v) for v in radar.orientation],
        "radar_look_at": [float(v) for v in look_at],
        "fmcw": {
            "carrier_frequency": float(radar.fmcw.carrier_frequency),
            "slope": float(radar.fmcw.slope),
            "chirp_duration": float(radar.fmcw.chirp_duration),
            "chirp_repetition_time": float(radar.fmcw.chirp_repetition_time),
            "sampling_frequency": float(radar.fmcw.sampling_frequency),
            "num_adc_samples": int(radar.fmcw.num_adc_samples),
            "num_chirps_per_frame": int(radar.fmcw.num_chirps_per_frame),
            "frame_period": float(radar.fmcw.frame_period),
        },
    }

    if args.mode == "single":
        cube, wall_time_s, num_chirps = _run_human_only_po(
            label="Single PO",
            config=_single_po_config(args),
            radar=radar,
            target=target,
            num_frames=num_frames,
            args=args,
        )
        rows = [_single_result_row("Single PO", cube, wall_time_s, num_chirps)]
        _write_summary(output_dir, metadata, rows)
        return

    baseline_path = output_dir / (
        f"full_po_{label_slug(args.mesh_npz.stem)}_"
        f"{num_frames}f_{radar.fmcw.num_chirps_per_frame}c.npz"
    )
    baseline_config = HumanPOMobilityConfig(
        visibility_samples_per_face=args.visibility_samples_per_face,
        visibility_fade_chirps=args.visibility_fade_chirps,
        visibility_use_phase_center=True,
        adaptive_visibility_sampling=False,
        incremental_update=False,
    )

    def cache_match(cube):
        """Checks whether a saved cube matches this benchmark configuration."""

        return cube_matches_run(
            cube,
            num_frames=num_frames,
            fmcw=radar.fmcw,
            last_chirp_time_s=last_chirp_time,
        )

    if args.reuse and baseline_path.exists():
        baseline, baseline_wall_time_s, baseline_num_chirps = load_radar_cube_npz(
            baseline_path)
        if cache_match(baseline):
            print(f"Loaded saved full PO baseline: {baseline_path}", flush=True)
        else:
            print("Saved baseline does not match current setup; rerunning",
                  flush=True)
            baseline, baseline_wall_time_s, baseline_num_chirps = (
                _run_human_only_po(
                    label="Full PO baseline",
                    config=baseline_config,
                    radar=radar,
                    target=target,
                    num_frames=num_frames,
                    args=args,
                )
            )
            save_radar_cube_npz(
                baseline_path,
                baseline,
                wall_time_s=baseline_wall_time_s,
                num_chirps=baseline_num_chirps,
                label="Full PO baseline",
            )
    else:
        baseline, baseline_wall_time_s, baseline_num_chirps = _run_human_only_po(
            label="Full PO baseline",
            config=baseline_config,
            radar=radar,
            target=target,
            num_frames=num_frames,
            args=args,
        )
        save_radar_cube_npz(
            baseline_path,
            baseline,
            wall_time_s=baseline_wall_time_s,
            num_chirps=baseline_num_chirps,
            label="Full PO baseline",
        )

    settings = args.setting if args.setting is not None else DEFAULT_SETTINGS

    def run_cube(label, config):
        """Runs or loads one incremental PO benchmark cube."""

        config = HumanPOMobilityConfig(
            visibility_samples_per_face=args.visibility_samples_per_face,
            visibility_fade_chirps=args.visibility_fade_chirps,
            visibility_use_phase_center=True,
            adaptive_visibility_sampling=True,
            adaptive_visibility_edge_margin=args.adaptive_visibility_edge_margin,
            incremental_update=True,
            incremental_visibility_refresh_chirps=(
                config.incremental_visibility_refresh_chirps),
            incremental_full_refresh_chirps=0,
            incremental_normal_threshold_deg=(
                config.incremental_normal_threshold_deg),
            incremental_centroid_displacement_threshold_m=(
                config.incremental_centroid_displacement_threshold_m),
            incremental_area_relative_threshold=(
                config.incremental_area_relative_threshold),
        )
        return _run_human_only_po(
            label=label,
            config=config,
            radar=radar,
            target=target,
            num_frames=num_frames,
            args=args,
        )

    profile_results = run_incremental_po_sweep(
        settings=settings,
        baseline_cube=baseline,
        baseline_wall_time_s=baseline_wall_time_s,
        fmcw=radar.fmcw,
        visibility_samples_per_face=args.visibility_samples_per_face,
        run_cube=run_cube,
        output_dir=output_dir,
        reuse_saved=args.reuse,
        cache_match=cache_match,
        filename_prefix=f"incremental_po_{label_slug(args.mesh_npz.stem)}",
    )
    rows = [_result_row(result) for result in profile_results]
    rows.sort(key=lambda row: (row["setting_K"], row["range_profile_relative_error"]))

    print("\nPO incremental sweep summary", flush=True)
    print(
        "label                              K  normal  disp_m  area    speedup  "
        "adc_err   range_err  recompute_mean",
        flush=True,
    )
    result_by_label = {result.label: result for result in profile_results}
    for row in rows:
        print(
            f"{row['label']:<34} "
            f"{row['setting_K']:>3d} "
            f"{row['normal_threshold_deg']:>6.1f} "
            f"{row['centroid_displacement_threshold_m']:>7.4f} "
            f"{row['area_relative_threshold']:>6.2f} "
            f"{row['speedup']:>8.2f} "
            f"{row['adc_relative_error']:>8.3e} "
            f"{row['range_profile_relative_error']:>9.3e} "
            f"{row['recomputed_mean']:>14.1f}",
            flush=True,
        )

    plot_rows = []
    for row in rows:
        result = result_by_label[row["label"]]
        plot_row = dict(row)
        plot_row["cube"] = result.cube
        plot_rows.append(plot_row)

    metadata.update({
        "baseline_cube_path": str(baseline_path),
        "baseline_wall_time_s": float(baseline_wall_time_s),
        "baseline_num_chirps": int(baseline_num_chirps),
        "settings": [asdict(setting) for setting in settings],
    })
    _write_summary(output_dir, metadata, rows)

    if args.plot:
        _plot_results(
            args,
            output_dir,
            radar.fmcw,
            baseline,
            baseline_wall_time_s,
            baseline_num_chirps,
            plot_rows,
        )


if __name__ == "__main__":
    main()
