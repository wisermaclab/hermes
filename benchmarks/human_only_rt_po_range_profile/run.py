# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Compare human-only PO and RT range profiles on one mesh frame.

The benchmark is intended to illustrate that full RT range profiles are
sensitive to the ray/path budget and to the human diffuse scattering
coefficient, while human-only PO provides a deterministic reference curve.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from dataclasses import replace
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from sionna.rt import load_scene  # noqa: E402

from mmWaveRadar import load_amass_npz  # noqa: E402
from mmWaveRadar.dsp import range_fft, range_profile_from_cube  # noqa: E402
from mmWaveRadar.materials import human_skin_material  # noqa: E402
from mmWaveRadar.radar import FMCWConfig, RadarSensor  # noqa: E402
from mmWaveRadar.simulation import (  # noqa: E402
    HumanPOMobilityConfig,
    MmWaveRadarSimulator,
    flatten_adc_for_fft,
    runtime_summary,
)
from mmWaveRadar.targets import MeshSequence, MeshTarget  # noqa: E402
from tools.amass_to_mesh_sequence import (  # noqa: E402
    convert_amass_to_mesh_sequence,
)


DEFAULT_TI_BOARD = "XWRL6844"


def _default_amass_npz() -> Path:
    """Resolves the motion override, dataset root, or bundled CMU walk."""

    value = os.environ.get("MMWAVE_AMASS_NPZ")
    if value:
        return Path(value).expanduser()
    dataset_root = os.environ.get("MMWAVE_DATASET_ROOT")
    if dataset_root:
        return (
            Path(dataset_root).expanduser()
            / "AMASS"
            / "walking_poses_cmu_105_02.npz"
        )
    return REPO_ROOT / "data" / "AMASS" / "walking_poses_cmu_105_02.npz"


def _default_smpl_model_dir() -> Path:
    """Resolves the model override or the repository-local model directory."""

    value = os.environ.get("MMWAVE_SMPL_MODEL_DIR")
    if value:
        return Path(value).expanduser()
    return REPO_ROOT / "models" / "smpl_models"


def _ti_board_key(value: str) -> str:
    """Normalizes the user's IWR6844 shorthand to the cataloged 6844 EVM."""

    key = str(value).strip()
    if key.upper().replace("-", "").replace("_", "") == "IWR6844":
        return DEFAULT_TI_BOARD
    return key


def _parse_csv_ints(value: str) -> list[int]:
    try:
        items = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integers, e.g. 5000,20000,80000"
        ) from exc
    if not items or any(item <= 0 for item in items):
        raise argparse.ArgumentTypeError("ray budgets must be positive")
    return items


def _parse_csv_floats(value: str) -> list[float]:
    try:
        items = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated floats, e.g. 0.05,0.35,0.75"
        ) from exc
    if not items or any(item < 0.0 or item > 1.0 for item in items):
        raise argparse.ArgumentTypeError(
            "scattering coefficients must be in [0, 1]"
        )
    return items


def _label_slug(label: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in label)
    return "_".join(part for part in slug.split("_") if part)


def _orientation_toward(target_point, radar_position):
    """Matches Sionna RadioDevice.look_at: local +x points at target_point."""

    direction = np.asarray(target_point, dtype=float) - np.asarray(
        radar_position, dtype=float)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        raise ValueError("radar position and look-at target are identical")
    direction /= norm
    theta = np.arccos(np.clip(direction[2], -1.0, 1.0))
    phi = np.arctan2(direction[1], direction[0])
    return (float(phi), float(theta - 0.5 * np.pi), 0.0)


def _axis_direction(axis: str) -> np.ndarray:
    directions = {
        "+x": np.array([1.0, 0.0, 0.0], dtype=float),
        "-x": np.array([-1.0, 0.0, 0.0], dtype=float),
        "+y": np.array([0.0, 1.0, 0.0], dtype=float),
        "-y": np.array([0.0, -1.0, 0.0], dtype=float),
    }
    return directions[axis]


def _mean_marker(markers, names):
    points = [markers[name] for name in names if name in markers]
    if not points:
        return None
    return np.mean(np.asarray(points, dtype=float), axis=0)


def _amass_marker_dict(amass_data, frame_index: int):
    if not amass_data or "marker_data" not in amass_data:
        return {}
    if "marker_labels" not in amass_data:
        return {}
    labels = [
        label.decode("utf-8") if isinstance(label, bytes) else str(label)
        for label in amass_data["marker_labels"]
    ]
    marker_data = np.asarray(amass_data["marker_data"])
    frame = min(max(int(frame_index), 0), marker_data.shape[0] - 1)
    points = marker_data[frame]
    return dict(zip(labels, points))


def _front_direction_from_amass(amass_data, frame_index: int):
    markers = _amass_marker_dict(amass_data, frame_index)
    if not markers:
        return None
    front_ref = _mean_marker(markers, ["STRN", "LFSH", "RFSH"])
    back_ref = _mean_marker(markers, ["TOPBACK", "MIDBACK", "LBSH", "RBSH"])
    if front_ref is None or back_ref is None:
        return None
    front_xy = np.asarray(front_ref[:2], dtype=float) - np.asarray(
        back_ref[:2], dtype=float)
    norm = np.linalg.norm(front_xy)
    if norm < 1e-6:
        return None
    return np.array([front_xy[0] / norm, front_xy[1] / norm, 0.0], dtype=float)


def _radar_pose_in_front_of_subject(mesh_sequence: MeshSequence, args,
                                    amass_data=None):
    vertices = mesh_sequence.vertices_at(float(mesh_sequence.times[0]))
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = 0.5 * (mins + maxs)
    front_dir = (
        _front_direction_from_amass(amass_data, args.amass_frame_index)
        if args.subject_front_axis == "auto"
        else None
    )
    if front_dir is None:
        front_dir = _axis_direction(args.subject_front_axis_fallback)
    height = float(mins[2] + args.radar_height_fraction * (maxs[2] - mins[2]))
    look_at = np.array([center[0], center[1], height], dtype=float)
    front_surface_projection = float(np.max(vertices @ front_dir))
    look_at_projection = float(look_at @ front_dir)
    front_surface_point = (
        look_at + (front_surface_projection - look_at_projection) * front_dir
    )
    radar_position = (
        front_surface_point + float(args.radar_clearance_m) * front_dir
    )
    radar_orientation = _orientation_toward(look_at, radar_position)
    return radar_position, radar_orientation, look_at


def _load_mesh_sequence(args):
    if args.mesh_npz is not None:
        return MeshSequence.from_npz(str(args.mesh_npz)), None, str(args.mesh_npz)
    if args.amass_npz is None or args.smpl_model_dir is None:
        raise ValueError(
            "Without --mesh-npz, an AMASS-format motion and SMPL model are "
            "required. The bundled CMU motion is used by default. Obtain the "
            "licensed SMPL model files separately and place them under "
            f"{REPO_ROOT / 'models' / 'smpl_models'}, set "
            "MMWAVE_SMPL_MODEL_DIR, or pass --smpl-model-dir."
        )

    mesh_cache_path = (
        args.output_dir
        / f"{Path(args.amass_npz).stem}_frame{int(args.amass_frame_index):04d}_mesh.npz"
    )
    if args.force_mesh_cache or not mesh_cache_path.exists():
        convert_amass_to_mesh_sequence(
            amass_npz=args.amass_npz,
            output_npz=mesh_cache_path,
            smpl_model_dir=args.smpl_model_dir,
            model_type=args.smpl_model_type,
            device=args.smpl_device,
            start_frame=args.amass_frame_index,
            num_frames=1,
            force=args.force_mesh_cache,
        )
    mesh_sequence = MeshSequence.from_npz(str(mesh_cache_path))
    return mesh_sequence, load_amass_npz(str(args.amass_npz)), str(args.amass_npz)


def _load_inputs(args):
    mesh_sequence, amass_data, source_path = _load_mesh_sequence(args)
    radar_position, radar_orientation, look_at = _radar_pose_in_front_of_subject(
        mesh_sequence,
        args,
        amass_data=amass_data,
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
    )
    requested_ti_board = _ti_board_key(args.ti_board)
    radar = RadarSensor.from_ti_board(
        requested_ti_board,
        fmcw=fmcw,
        name="radar",
        position=tuple(float(v) for v in radar_position),
        orientation=tuple(float(v) for v in radar_orientation),
        pattern_mode=args.pattern_mode,
    )
    fmcw = radar.fmcw
    if args.num_chirps_per_frame is not None:
        fmcw = replace(fmcw, num_chirps_per_frame=int(args.num_chirps_per_frame))
    if args.num_adc_samples is not None:
        fmcw = replace(fmcw, num_adc_samples=int(args.num_adc_samples))
    if fmcw is not radar.fmcw:
        radar = replace(radar, fmcw=fmcw)
    return radar, mesh_sequence, look_at, source_path


def _make_target(label: str, mesh_sequence: MeshSequence, scattering: float):
    material = human_skin_material(f"human-skin-{_label_slug(label)}")
    material.scattering_coefficient = float(scattering)
    return MeshTarget(
        name=f"human_{_label_slug(label)}",
        mesh_sequence=mesh_sequence,
        material=material,
    )


def _run_cube(label: str, mode: str, args, radar, mesh_sequence, scattering: float,
              ray_budget: int | None = None):
    scene = load_scene()
    target = _make_target(label, mesh_sequence, scattering)
    sim_kwargs = {
        "mobility_mode": mode,
        "progress": args.progress,
        "compute_backend": args.compute_backend,
        "compute_precision": args.compute_precision,
        "adc_compute_backend": args.adc_compute_backend,
        "adc_compute_precision": args.adc_compute_precision,
    }
    if mode == "human_only_po":
        sim_kwargs["human_po_config"] = HumanPOMobilityConfig(
            visibility_samples_per_face=args.visibility_samples_per_face,
            visibility_fade_chirps=args.visibility_fade_chirps,
            visibility_use_phase_center=True,
            adaptive_visibility_sampling=args.adaptive_visibility_sampling,
            adaptive_visibility_edge_margin=args.adaptive_visibility_edge_margin,
            compute_backend=args.po_compute_backend,
            compute_precision=args.po_compute_precision,
        )
    else:
        sim_kwargs.update(
            coupling_mode=args.coupling_mode,
            max_depth=args.max_depth,
            samples_per_src=int(ray_budget),
            max_num_paths_per_src=int(
                ray_budget
                if args.max_num_paths_per_src is None
                else args.max_num_paths_per_src
            ),
            diffuse_reflection=True,
            human_specular_reflection=args.human_specular_reflection,
            retrace_once_per_frame=True,
            seed=args.seed,
        )

    simulator = MmWaveRadarSimulator(**sim_kwargs)
    start = time.perf_counter()
    cube = simulator.run(scene, radar, [target], num_frames=1)
    wall_time_s = time.perf_counter() - start
    return cube, wall_time_s


def _range_profile(cube, fmcw, *, nfft_mult: int):
    range_cube, ranges_m = range_fft(
        flatten_adc_for_fft(cube.adc),
        fmcw=fmcw,
        window="hann",
        nfft_mult=nfft_mult,
    )
    profile = range_profile_from_cube(
        range_cube,
        combine_antennas="sum_power",
        combine_chirps="mean",
    )
    return np.asarray(profile, dtype=float), np.asarray(ranges_m, dtype=float)


def _profile_row(label: str, kind: str, profile, ranges_m, cube, wall_time_s,
                 *, scattering: float, ray_budget: int | None,
                 reference_peak: float):
    peak_index = int(np.argmax(profile))
    total_power = float(np.sum(profile))
    return {
        "label": label,
        "kind": kind,
        "ray_budget": "" if ray_budget is None else int(ray_budget),
        "scattering_coefficient": float(scattering),
        "wall_time_s": float(wall_time_s),
        "mean_paths": float(np.mean(cube.metadata.path_counts)),
        "max_paths": int(np.max(cube.metadata.path_counts)),
        "peak_range_m": float(ranges_m[peak_index]),
        "peak_power_db_ref_po": float(
            10.0 * np.log10(max(float(profile[peak_index]), 1e-30)
                            / reference_peak)
        ),
        "total_power_db_ref_po": float(
            10.0 * np.log10(max(total_power, 1e-30) / reference_peak)
        ),
    }


def _write_profiles_npz(output_dir: Path, ranges_m, profiles: dict[str, np.ndarray]):
    payload = {"ranges_m": ranges_m}
    for label, profile in profiles.items():
        payload[f"profile_{_label_slug(label)}"] = profile
    np.savez_compressed(output_dir / "range_profiles.npz", **payload)


def _write_summary(output_dir: Path, metadata: dict, rows: list[dict]):
    summary = dict(metadata)
    summary["results"] = rows
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote summary JSON: {json_path}", flush=True)
    print(f"Wrote summary CSV: {csv_path}", flush=True)


def _db_relative(profile, reference_peak):
    return 10.0 * np.log10(np.maximum(profile, 1e-30) / reference_peak)


def _range_plot_limits(mesh_sequence: MeshSequence, radar, args):
    vertices = mesh_sequence.vertices_at(float(mesh_sequence.times[0]))
    local = radar.world_to_local_points(vertices)
    padding = float(args.range_plot_padding_m)
    xmin = max(0.0, float(np.min(local[:, 0])) - padding)
    xmax = float(np.max(local[:, 0])) + padding
    xmax = min(float(args.range_limit_m), max(xmax, xmin + 0.25))
    return xmin, xmax


def _plot_top_down_panel(ax, mesh_sequence: MeshSequence, radar):
    vertices = mesh_sequence.vertices_at(float(mesh_sequence.times[0]))
    faces = np.asarray(mesh_sequence.faces, dtype=np.int64)
    local = radar.world_to_local_points(vertices)
    tri = ax.tripcolor(
        local[:, 0],
        local[:, 1],
        faces,
        facecolors=np.mean(local[faces, 2], axis=1),
        shading="flat",
        cmap="cividis",
        edgecolors="none",
        alpha=0.95,
    )
    ax.scatter([0.0], [0.0], color="#d62728", marker="^", s=45,
               label="radar")
    mesh_center = np.mean(local[:, :2], axis=0)
    ax.annotate(
        "",
        xy=(mesh_center[0], mesh_center[1]),
        xytext=(0.0, 0.0),
        arrowprops={"arrowstyle": "->", "color": "#d62728", "lw": 1.2},
    )
    limits = np.vstack([local[:, :2], np.zeros((1, 2), dtype=float)])
    mins = limits.min(axis=0)
    maxs = limits.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = max(float(np.max(maxs - mins)), 0.5)
    radius = 0.58 * span
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Top-down radar view")
    ax.set_xlabel("boresight range x [m]")
    ax.set_ylabel("cross-range y [m]")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    return tri


def _plot_profile_panel(ax, mesh_sequence: MeshSequence, radar):
    vertices = mesh_sequence.vertices_at(float(mesh_sequence.times[0]))
    faces = np.asarray(mesh_sequence.faces, dtype=np.int64)
    local = radar.world_to_local_points(vertices)
    tri = ax.tripcolor(
        local[:, 0],
        local[:, 2],
        faces,
        facecolors=np.mean(local[faces, 1], axis=1),
        shading="flat",
        cmap="plasma",
        edgecolors="none",
        alpha=0.95,
    )
    ax.scatter([0.0], [0.0], color="#d62728", marker="^", s=45,
               label="radar")
    mesh_center = np.mean(local[:, [0, 2]], axis=0)
    ax.annotate(
        "",
        xy=(mesh_center[0], mesh_center[1]),
        xytext=(0.0, 0.0),
        arrowprops={"arrowstyle": "->", "color": "#d62728", "lw": 1.2},
    )
    limits = np.vstack([local[:, [0, 2]], np.zeros((1, 2), dtype=float)])
    mins = limits.min(axis=0)
    maxs = limits.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = max(float(np.max(maxs - mins)), 0.5)
    radius = 0.58 * span
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Side radar view")
    ax.set_xlabel("boresight range x [m]")
    ax.set_ylabel("height z [m]")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    return tri


def _plot(output_dir: Path, args, ranges_m, profiles: dict[str, np.ndarray],
          *, mesh_sequence: MeshSequence, radar, reference_peak: float,
          nominal_scattering: float, nominal_ray_budget: int):
    import matplotlib

    matplotlib.use(args.matplotlib_backend)
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    range_min_m, range_max_m = _range_plot_limits(mesh_sequence, radar, args)
    mask = (ranges_m >= range_min_m) & (ranges_m <= range_max_m)
    fig = plt.figure(figsize=(16.5, 6.2), constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=(1.05, 1.0, 1.0),
        height_ratios=(1.0, 1.0),
    )
    top_ax = fig.add_subplot(grid[0, 0])
    profile_ax = fig.add_subplot(grid[1, 0])
    axes = [
        fig.add_subplot(grid[:, 1]),
        fig.add_subplot(grid[:, 2]),
    ]
    top_im = _plot_top_down_panel(top_ax, mesh_sequence, radar)
    fig.colorbar(top_im, ax=top_ax, label="height z [m]",
                 fraction=0.046, pad=0.04)
    profile_im = _plot_profile_panel(profile_ax, mesh_sequence, radar)
    fig.colorbar(profile_im, ax=profile_ax, label="cross-range y [m]",
                 fraction=0.046, pad=0.04)

    po_profile = profiles["PO"]
    axes[0].plot(
        ranges_m[mask],
        _db_relative(po_profile, reference_peak)[mask],
        color="black",
        linewidth=2.0,
        label="PO",
    )
    for ray_budget in args.ray_budgets:
        label = f"RT rays={ray_budget:g}, S={nominal_scattering:g}"
        profile = profiles[label]
        axes[0].plot(
            ranges_m[mask],
            _db_relative(profile, reference_peak)[mask],
            linewidth=1.5,
            label=f"RT {ray_budget:g} rays",
        )
    axes[0].set_title(f"RT ray-budget sensitivity (S={nominal_scattering:g})")
    axes[0].set_xlabel("range [m]")
    axes[0].set_ylabel("power [dB re PO peak]")
    axes[0].set_xlim(range_min_m, range_max_m)
    axes[0].set_ylim(args.plot_vmin_db, args.plot_vmax_db)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(
        ranges_m[mask],
        _db_relative(po_profile, reference_peak)[mask],
        color="black",
        linewidth=2.0,
        label="PO",
    )
    for scattering in args.scattering_coefficients:
        label = f"RT rays={nominal_ray_budget:g}, S={scattering:g}"
        profile = profiles[label]
        axes[1].plot(
            ranges_m[mask],
            _db_relative(profile, reference_peak)[mask],
            linewidth=1.5,
            label=f"RT S={scattering:g}",
        )
    axes[1].set_title(f"RT scattering sensitivity ({nominal_ray_budget:g} rays)")
    axes[1].set_xlabel("range [m]")
    axes[1].set_ylabel("power [dB re PO peak]")
    axes[1].set_xlim(range_min_m, range_max_m)
    axes[1].set_ylim(args.plot_vmin_db, args.plot_vmax_db)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    figure_path = output_dir / "human_only_rt_po_range_profile.png"
    fig.savefig(figure_path, dpi=args.plot_dpi)
    plt.close(fig)
    print(f"Wrote figure: {figure_path}", flush=True)


def _parse_args() -> argparse.Namespace:
    """Parses CLI controls for the PO-vs-RT range-profile comparison."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare human-only PO and RT range profiles on one RT-Pose/AMASS "
            "mesh frame, sweeping RT ray budget and human scattering."
        )
    )
    parser.add_argument(
        "--amass-npz",
        type=Path,
        default=_default_amass_npz(),
        help=(
            "AMASS-format motion file. Defaults to MMWAVE_AMASS_NPZ, then "
            "MMWAVE_DATASET_ROOT/AMASS/walking_poses_cmu_105_02.npz, then "
            "the bundled CMU walking fixture."
        ),
    )
    parser.add_argument("--amass-frame-index", type=int, default=0)
    parser.add_argument("--mesh-npz", type=Path, default=None,
                        help="Optional precomputed mesh sequence. Overrides --amass-npz.")
    parser.add_argument(
        "--smpl-model-dir",
        type=Path,
        default=_default_smpl_model_dir(),
        help=(
            "Licensed SMPL model directory. Defaults to "
            "MMWAVE_SMPL_MODEL_DIR, then models/smpl_models under the "
            "repository root."
        ),
    )
    parser.add_argument("--smpl-model-type", default="smpl")
    parser.add_argument("--smpl-device", default="cpu")
    parser.add_argument("--force-mesh-cache", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(tempfile.gettempdir())
            / "mmwave_radar_outputs"
            / "human_only_rt_po_range_profile"
        ),
    )
    parser.add_argument("--ray-budgets", type=_parse_csv_ints,
                        default=_parse_csv_ints("5000,20000,80000"))
    parser.add_argument("--scattering-coefficients", type=_parse_csv_floats,
                        default=_parse_csv_floats("0.05,0.35,0.75"))
    parser.add_argument("--nominal-scattering", type=float, default=0.35)
    parser.add_argument("--nominal-ray-budget", type=int, default=None)
    parser.add_argument("--max-num-paths-per-src", type=int, default=None)
    parser.add_argument("--ti-board", default=DEFAULT_TI_BOARD,
                        help="TI radar board/catalog alias to use.")
    parser.add_argument("--pattern-mode", default="cosine30",
                        choices=("digitized", "none", "scalar", "cosine30"))
    parser.add_argument("--radar-clearance-m", type=float, default=1.5,
                        help="Distance from subject front surface to radar.")
    parser.add_argument("--subject-front-axis", default="auto",
                        choices=("auto", "+x", "-x", "+y", "-y"),
                        help="Subject front direction. 'auto' uses AMASS markers when available.")
    parser.add_argument("--subject-front-axis-fallback", default="+x",
                        choices=("+x", "-x", "+y", "-y"),
                        help="Fallback front direction if marker-based front inference is unavailable.")
    parser.add_argument("--radar-height-fraction", type=float, default=0.62,
                        help="Look-at/radar height as fraction of mesh z span.")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--coupling-mode", default="one_target_bounce",
                        choices=("one_target_bounce", "unrestricted"))
    parser.add_argument("--human-specular-reflection",
                        action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--carrier-frequency", type=float, default=60.0e9)
    parser.add_argument("--slope", type=float, default=68.0e12)
    parser.add_argument("--chirp-duration", type=float, default=58.0e-6)
    parser.add_argument("--chirp-repetition-time", type=float, default=65.0e-6)
    parser.add_argument("--sampling-frequency", type=float, default=4.5e6)
    parser.add_argument("--num-adc-samples", type=int, default=225)
    parser.add_argument("--num-chirps-per-frame", type=int, default=96)
    parser.add_argument("--frame-period", type=float, default=50.0e-3)
    parser.add_argument("--range-nfft-mult", type=int, default=4)
    parser.add_argument("--range-limit-m", type=float, default=4.0)
    parser.add_argument("--range-plot-padding-m", type=float, default=0.35,
                        help="Padding around the human mesh range extent in range-profile panels.")
    parser.add_argument("--visibility-samples-per-face", type=int, default=1)
    parser.add_argument("--visibility-fade-chirps", type=int, default=8)
    parser.add_argument("--adaptive-visibility-sampling",
                        action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--adaptive-visibility-edge-margin", type=float,
                        default=0.15)
    parser.add_argument("--po-compute-backend",
                        choices=("numpy", "torch", "auto"), default="auto")
    parser.add_argument("--po-compute-precision",
                        choices=("float64", "float32"), default="float32")
    parser.add_argument("--compute-backend", choices=("numpy", "torch", "auto"),
                        default="auto")
    parser.add_argument("--compute-precision",
                        choices=("float64", "float32"), default="float32")
    parser.add_argument("--adc-compute-backend",
                        choices=("numpy", "torch", "auto"), default=None)
    parser.add_argument("--adc-compute-precision",
                        choices=("float64", "float32"), default=None)
    parser.add_argument("--matplotlib-backend", default="Agg")
    parser.add_argument("--plot-dpi", type=int, default=180)
    parser.add_argument("--plot-vmin-db", type=float, default=-80.0)
    parser.add_argument("--plot-vmax-db", type=float, default=10.0)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Runs the PO baseline, RT sweeps, and range-profile report generation."""

    args = _parse_args()
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "mmwave-radar-mpl"),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.nominal_scattering not in args.scattering_coefficients:
        args.scattering_coefficients = [
            *args.scattering_coefficients,
            float(args.nominal_scattering),
        ]
    nominal_ray_budget = (
        max(args.ray_budgets)
        if args.nominal_ray_budget is None
        else int(args.nominal_ray_budget)
    )
    if nominal_ray_budget not in args.ray_budgets:
        args.ray_budgets = [*args.ray_budgets, nominal_ray_budget]

    radar, mesh_sequence, radar_look_at, mesh_source_path = _load_inputs(args)
    print("Runtime:", runtime_summary(), flush=True)
    print(f"Mesh source: {mesh_source_path}", flush=True)
    print(
        f"Mesh: vertices={mesh_sequence.vertices.shape}, "
        f"faces={mesh_sequence.faces.shape}",
        flush=True,
    )
    print(
        f"Radar: board={radar.hardware.name}, "
        f"position={np.asarray(radar.position)}, "
        f"look_at={np.asarray(radar_look_at)}, "
        f"chirps={radar.fmcw.num_chirps_per_frame}, "
        f"samples={radar.fmcw.num_adc_samples}, "
        f"virtual_channels={radar.hardware.num_virtual_channels}",
        flush=True,
    )

    print("Running PO baseline", flush=True)
    po_cube, po_wall_time_s = _run_cube(
        "PO",
        "human_only_po",
        args,
        radar,
        mesh_sequence,
        scattering=args.nominal_scattering,
    )
    po_profile, ranges_m = _range_profile(
        po_cube,
        radar.fmcw,
        nfft_mult=args.range_nfft_mult,
    )
    reference_peak = max(float(np.max(po_profile)), 1e-30)

    profiles = {"PO": po_profile}
    rows = [
        _profile_row(
            "PO",
            "po",
            po_profile,
            ranges_m,
            po_cube,
            po_wall_time_s,
            scattering=args.nominal_scattering,
            ray_budget=None,
            reference_peak=reference_peak,
        )
    ]

    rt_points: list[tuple[int, float]] = []
    for ray_budget in args.ray_budgets:
        rt_points.append((int(ray_budget), float(args.nominal_scattering)))
    for scattering in args.scattering_coefficients:
        rt_points.append((int(nominal_ray_budget), float(scattering)))
    rt_points = list(dict.fromkeys(rt_points))

    for ray_budget, scattering in rt_points:
        label = f"RT rays={ray_budget:g}, S={scattering:g}"
        print(f"Running {label}", flush=True)
        cube, wall_time_s = _run_cube(
            label,
            "rt_retrace",
            args,
            radar,
            mesh_sequence,
            scattering=scattering,
            ray_budget=ray_budget,
        )
        profile, run_ranges_m = _range_profile(
            cube,
            radar.fmcw,
            nfft_mult=args.range_nfft_mult,
        )
        if not np.allclose(run_ranges_m, ranges_m):
            raise RuntimeError("range bins changed across runs")
        profiles[label] = profile
        rows.append(
            _profile_row(
                label,
                "rt",
                profile,
                ranges_m,
                cube,
                wall_time_s,
                scattering=scattering,
                ray_budget=ray_budget,
                reference_peak=reference_peak,
            )
        )

    range_plot_min_m, range_plot_max_m = _range_plot_limits(
        mesh_sequence,
        radar,
        args,
    )
    metadata = {
        "runtime": runtime_summary(),
        "mesh_source": str(mesh_source_path),
        "amass_npz": "" if args.mesh_npz is not None else str(args.amass_npz),
        "amass_frame_index": int(args.amass_frame_index),
        "mesh_npz": "" if args.mesh_npz is None else str(args.mesh_npz),
        "ti_board_requested": str(args.ti_board),
        "sensor_board_model": radar.hardware.name,
        "pattern_mode": str(args.pattern_mode),
        "carrier_frequency_hz": float(radar.fmcw.carrier_frequency),
        "slope_hz_per_s": float(radar.fmcw.slope),
        "sampling_frequency_hz": float(radar.fmcw.sampling_frequency),
        "radar_position": list(radar.position),
        "radar_orientation": list(radar.orientation),
        "radar_look_at": np.asarray(radar_look_at, dtype=float).tolist(),
        "radar_clearance_m": float(args.radar_clearance_m),
        "subject_front_axis": str(args.subject_front_axis),
        "subject_front_axis_fallback": str(args.subject_front_axis_fallback),
        "radar_height_fraction": float(args.radar_height_fraction),
        "num_chirps_per_frame": int(radar.fmcw.num_chirps_per_frame),
        "num_adc_samples": int(radar.fmcw.num_adc_samples),
        "num_virtual_channels": int(radar.hardware.num_virtual_channels),
        "range_plot_min_m": float(range_plot_min_m),
        "range_plot_max_m": float(range_plot_max_m),
        "range_plot_padding_m": float(args.range_plot_padding_m),
        "ray_budgets": [int(v) for v in args.ray_budgets],
        "scattering_coefficients": [
            float(v) for v in args.scattering_coefficients
        ],
        "nominal_scattering": float(args.nominal_scattering),
        "nominal_ray_budget": int(nominal_ray_budget),
        "rt_diffuse_reflection": True,
        "rt_human_specular_reflection": bool(args.human_specular_reflection),
    }
    _write_profiles_npz(args.output_dir, ranges_m, profiles)
    _write_summary(args.output_dir, metadata, rows)
    _plot(
        args.output_dir,
        args,
        ranges_m,
        profiles,
        mesh_sequence=mesh_sequence,
        radar=radar,
        reference_peak=reference_peak,
        nominal_scattering=float(args.nominal_scattering),
        nominal_ray_budget=int(nominal_ray_budget),
    )


if __name__ == "__main__":
    main()
