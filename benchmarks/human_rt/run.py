# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""One-frame/full-sequence benchmark for human-motion RT sensing."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
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

from mmWaveRadar import amass_to_mesh_sequence, load_amass_npz  # noqa: E402
from mmWaveRadar.materials import human_skin_material  # noqa: E402
from mmWaveRadar.radar import FMCWConfig, RadarHardware, RadarSensor  # noqa: E402
from mmWaveRadar.simulation import (  # noqa: E402
    MmWaveRadarSimulator,
    RTCoherentTransitionConfig,
    label_slug,
    runtime_summary,
)
from mmWaveRadar.targets import MeshTarget  # noqa: E402


RT_MOBILITY_MODES = ("rt_retrace", "rt_coherent_bank")
RT_MODE_CHOICES = (*RT_MOBILITY_MODES, "both")


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


def _orientation_toward(target_point, radar_position):
    """Matches Sionna RadioDevice.look_at: local +x points toward target_point."""

    direction = np.asarray(target_point, dtype=float) - np.asarray(
        radar_position, dtype=float)
    direction /= np.linalg.norm(direction)
    theta = np.arccos(np.clip(direction[2], -1.0, 1.0))
    phi = np.arctan2(direction[1], direction[0])
    return (float(phi), float(theta - 0.5 * np.pi), 0.0)


def _mean_marker(markers, names):
    points = [markers[name] for name in names if name in markers]
    if not points:
        return None
    return np.mean(np.asarray(points, dtype=float), axis=0)


def _amass_marker_dict(amass_data):
    if "marker_data" not in amass_data or "marker_labels" not in amass_data:
        return {}
    labels = [
        label.decode("utf-8") if isinstance(label, bytes) else str(label)
        for label in amass_data["marker_labels"]
    ]
    points = np.asarray(amass_data["marker_data"])[0]
    return dict(zip(labels, points))


def _smpl_chest_front(vertices, amass_data):
    markers = _amass_marker_dict(amass_data)
    if not markers:
        return None
    sternum = _mean_marker(markers, ["STRN", "CLAV"])
    front_ref = _mean_marker(markers, ["STRN", "LFSH", "RFSH"])
    back_ref = _mean_marker(markers, ["TOPBACK", "MIDBACK", "LBSH", "RBSH"])
    if sternum is None or front_ref is None or back_ref is None:
        return None
    front_xy = front_ref[:2] - back_ref[:2]
    front_norm = np.linalg.norm(front_xy)
    if front_norm < 1e-6:
        return None
    front_dir_xy = front_xy / front_norm
    lateral_dir_xy = np.array([-front_dir_xy[1], front_dir_xy[0]])
    z_span = max(float(vertices[:, 2].max() - vertices[:, 2].min()), 1e-6)
    lateral = vertices[:, :2] @ lateral_dir_xy
    lateral_center = float(sternum[:2] @ lateral_dir_xy)
    torso = (
        (np.abs(vertices[:, 2] - sternum[2]) <= max(0.12, 0.12 * z_span))
        & (np.abs(lateral - lateral_center) <= 0.30)
    )
    if np.count_nonzero(torso) < 16:
        torso = np.ones(vertices.shape[0], dtype=bool)
    proj = vertices[:, :2] @ front_dir_xy
    front_surface_proj = float(np.max(proj[torso]))
    sternum_proj = float(sternum[:2] @ front_dir_xy)
    chest_front_xy = (
        sternum[:2] + (front_surface_proj - sternum_proj) * front_dir_xy
    )
    chest_front_point = np.array(
        [chest_front_xy[0], chest_front_xy[1], sternum[2]], dtype=float)
    front_dir = np.array([front_dir_xy[0], front_dir_xy[1], 0.0], dtype=float)
    return chest_front_point, front_dir


def _radar_pose_in_front_of_chest(mesh_sequence, *, clearance_m, amass_data):
    vertices = mesh_sequence.vertices_at(float(mesh_sequence.times[0]))
    chest = _smpl_chest_front(vertices, amass_data)
    if chest is None:
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        chest_front_point = np.array(
            [mins[0], 0.5 * (mins[1] + maxs[1]), 0.65 * maxs[2]],
            dtype=float,
        )
        front_dir = np.array([-1.0, 0.0, 0.0], dtype=float)
    else:
        chest_front_point, front_dir = chest
    radar_position = chest_front_point + float(clearance_m) * front_dir
    radar_orientation = _orientation_toward(chest_front_point, radar_position)
    return radar_position, radar_orientation, chest_front_point


def _build_radar_and_target(args):
    required_paths = {
        "motion_npz": args.motion_npz,
        "smpl_model_dir": args.smpl_model_dir,
    }
    missing_paths = [
        f"{name} ({path})" if path is not None else name
        for name, path in required_paths.items()
        if path is None or not path.exists()
    ]
    if missing_paths:
        missing = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Benchmark requires AMASS/SMPL inputs, but these paths are missing:\n"
            f"{missing}\n"
            "Obtain the licensed SMPL model files separately and place them "
            f"under {REPO_ROOT / 'models' / 'smpl_models'}, set "
            "MMWAVE_SMPL_MODEL_DIR, or pass --smpl-model-dir. Override the "
            "bundled motion with --motion-npz when needed."
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
    hardware = RadarHardware.from_positions(
        tx_positions=[[0.0, 0.0, 0.0]],
        rx_positions=[[0.0, 0.0, 0.0]],
        name="single-virtual-channel",
    )
    amass_data = load_amass_npz(str(args.motion_npz))
    mesh_sequence = amass_to_mesh_sequence(
        str(args.motion_npz),
        str(args.smpl_model_dir),
        model_type=args.smpl_model_type,
        device=args.smpl_device,
    )
    radar_position, radar_orientation, chest_front_point = (
        _radar_pose_in_front_of_chest(
            mesh_sequence,
            clearance_m=args.radar_clearance_m,
            amass_data=amass_data,
        )
    )
    radar = RadarSensor(
        name="radar",
        position=tuple(float(v) for v in radar_position),
        orientation=tuple(float(v) for v in radar_orientation),
        hardware=hardware,
        fmcw=fmcw,
    )
    target = MeshTarget(
        name="amass_human",
        mesh_sequence=mesh_sequence,
        material=human_skin_material("human-skin-benchmark"),
    )
    return radar, target, mesh_sequence, chest_front_point


def _parse_period_chirps(value: str) -> int | None:
    text = value.strip().lower()
    if text in {"", "none", "default", "frame", "per_frame"}:
        return None
    try:
        period = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "period_chirps must be a positive integer or one of "
            "none/default/frame"
        ) from exc
    if period <= 0:
        raise argparse.ArgumentTypeError("period_chirps must be positive")
    return period


def _modes_from_args(args) -> tuple[str, ...]:
    if args.mode == "both":
        return RT_MOBILITY_MODES
    return (args.mode,)


def _count_stats(prefix: str, values: np.ndarray) -> dict:
    array = np.asarray(values)
    return {
        f"{prefix}_min": int(np.min(array)),
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_max": int(np.max(array)),
    }


def _float_stats(prefix: str, values: np.ndarray) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        f"{prefix}_min": float(np.min(array)),
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_max": float(np.max(array)),
    }


def _run_rt(mobility_mode: str, args, radar, target):
    scene = load_scene()
    transition_config = RTCoherentTransitionConfig(
        enabled=args.coherent_transition,
        match_delay_tolerance_fraction=(
            args.coherent_match_delay_tolerance_fraction),
        match_separation_fraction=args.coherent_match_separation_fraction,
        crossfade=args.coherent_crossfade,
    )
    periodic_retrace_period_chirps = (
        args.coherent_bank_retrace_period_chirps
        if mobility_mode == "rt_coherent_bank" else None
    )
    sim_kwargs = {
        "mobility_mode": mobility_mode,
        "progress": args.progress,
        "coupling_mode": args.coupling_mode,
        "max_depth": args.max_depth,
        "samples_per_src": args.samples_per_src,
        "max_num_paths_per_src": args.max_num_paths_per_src,
        "diffuse_reflection": args.diffuse_reflection,
        "periodic_retrace": True,
        "periodic_retrace_period_chirps": periodic_retrace_period_chirps,
        "rt_coherent_transition_config": transition_config,
    }
    if args.compute_backend is not None:
        sim_kwargs["compute_backend"] = args.compute_backend
    if args.compute_precision is not None:
        sim_kwargs["compute_precision"] = args.compute_precision
    if args.adc_compute_backend is not None:
        sim_kwargs["adc_compute_backend"] = args.adc_compute_backend
    if args.adc_compute_precision is not None:
        sim_kwargs["adc_compute_precision"] = args.adc_compute_precision

    target_for_run = MeshTarget(
        name=f"{target.name}_{label_slug(mobility_mode)}",
        mesh_sequence=target.mesh_sequence,
        material=target.material,
    )
    simulator = MmWaveRadarSimulator(**sim_kwargs)
    effective_period_chirps = simulator.effective_periodic_retrace_period_chirps(
        radar.fmcw)
    start = time.perf_counter()
    cube = simulator.run(
        scene,
        radar,
        [target_for_run],
        num_frames=args.num_frames,
    )
    elapsed = time.perf_counter() - start
    num_chirps = args.num_frames * radar.fmcw.num_chirps_per_frame
    result = {
        "label": mobility_mode,
        "mobility_mode": mobility_mode,
        "max_depth": args.max_depth,
        "samples_per_src": args.samples_per_src,
        "max_num_paths_per_src": args.max_num_paths_per_src,
        "periodic_retrace": True,
        "periodic_retrace_period_chirps": (
            periodic_retrace_period_chirps),
        "effective_periodic_retrace_period_chirps": effective_period_chirps,
        "coherent_transition_enabled": bool(args.coherent_transition),
        "coherent_crossfade": args.coherent_crossfade,
        "coherent_match_delay_tolerance_fraction": (
            args.coherent_match_delay_tolerance_fraction),
        "coherent_match_separation_fraction": (
            args.coherent_match_separation_fraction),
        "seconds": elapsed,
        "seconds_per_frame": elapsed / args.num_frames,
        "seconds_per_chirp": elapsed / num_chirps,
        "adc_shape": list(cube.adc.shape),
        "events": int(len(cube.metadata.events)),
    }
    result.update(_count_stats("path_count", cube.metadata.path_counts))
    if cube.metadata.runtime_profile_s is not None:
        result["runtime_profile_s"] = {
            key: float(value)
            for key, value in cube.metadata.runtime_profile_s.items()
        }
    if cube.metadata.runtime_profile_counts is not None:
        result["runtime_profile_counts"] = {
            key: int(value)
            for key, value in cube.metadata.runtime_profile_counts.items()
        }
        result["runtime_profile_per_event_s"] = {
            key: (
                float(result["runtime_profile_s"].get(key, 0.0)) / int(count)
                if int(count) > 0 else None
            )
            for key, count in result["runtime_profile_counts"].items()
        }
    transition_count_fields = {
        "rt_transition_persistent_path_counts": "transition_persistent_count",
        "rt_transition_birth_path_counts": "transition_birth_count",
        "rt_transition_death_path_counts": "transition_death_count",
    }
    for attr_name, prefix in transition_count_fields.items():
        values = getattr(cube.metadata, attr_name, None)
        if values is not None:
            result.update(_count_stats(prefix, values))
    transition_float_fields = {
        "rt_transition_alpha": "transition_alpha",
        "rt_transition_old_weights": "transition_old_weight",
        "rt_transition_new_weights": "transition_new_weight",
    }
    for attr_name, prefix in transition_float_fields.items():
        values = getattr(cube.metadata, attr_name, None)
        if values is not None:
            result.update(_float_stats(prefix, values))
    return result


def _parse_args() -> argparse.Namespace:
    """Parses CLI controls for RT benchmark mode, data paths, and output."""

    parser = argparse.ArgumentParser(
        description="Benchmark human-motion RT on an AMASS sequence.")
    parser.add_argument(
        "--motion-npz",
        type=Path,
        default=_default_amass_npz(),
        help=(
            "AMASS-format motion file. Defaults to MMWAVE_AMASS_NPZ, then "
            "MMWAVE_DATASET_ROOT/AMASS/walking_poses_cmu_105_02.npz, then "
            "the bundled CMU walking fixture."
        ),
    )
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
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--radar-clearance-m", type=float, default=0.50)
    parser.add_argument("--compute-backend",
                        choices=("numpy", "torch", "auto"),
                        default=None)
    parser.add_argument("--compute-precision",
                        choices=("float64", "float32"),
                        default=None)
    parser.add_argument("--adc-compute-backend",
                        choices=("numpy", "torch", "auto"),
                        default=None)
    parser.add_argument("--adc-compute-precision",
                        choices=("float64", "float32"),
                        default=None)
    parser.add_argument("--mode",
                        choices=RT_MODE_CHOICES,
                        default="rt_retrace",
                        help=("RT mobility mode to benchmark. Use 'both' to "
                              "run rt_retrace and rt_coherent_bank with the "
                              "same ray-tracing parameters."))
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--samples-per-src", type=int, default=20_000)
    parser.add_argument("--max-num-paths-per-src", type=int, default=20_000)
    parser.add_argument("--coherent-bank-retrace-period-chirps",
                        type=_parse_period_chirps,
                        default=None,
                        help=("Retrace interval in chirps for "
                              "rt_coherent_bank. Defaults to one retrace per "
                              "frame."))
    parser.add_argument("--diffuse-reflection",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help=("Enable Sionna diffuse RT so "
                              "scattering_coefficient is used."))
    parser.add_argument("--coupling-mode", default="one_target_bounce",
                        choices=("one_target_bounce", "unrestricted"))
    parser.add_argument("--coherent-transition",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help=("Enable path matching for rt_coherent_bank "
                              "transition diagnostics/synthesis."))
    parser.add_argument("--coherent-crossfade",
                        action=argparse.BooleanOptionalAction,
                        default=None,
                        help=("Force coherent bank crossfade on/off. Omit to "
                              "use the simulator policy."))
    parser.add_argument("--coherent-match-delay-tolerance-fraction",
                        type=float,
                        default=0.5)
    parser.add_argument("--coherent-match-separation-fraction",
                        type=float,
                        default=0.25)
    parser.add_argument("--carrier-frequency", type=float, default=77e9)
    parser.add_argument("--slope", type=float, default=40e12)
    parser.add_argument("--chirp-duration", type=float, default=80e-6)
    parser.add_argument("--chirp-repetition-time", type=float, default=1e-3)
    parser.add_argument("--sampling-frequency", type=float, default=2e6)
    parser.add_argument("--num-adc-samples", type=int, default=128)
    parser.add_argument("--num-chirps-per-frame", type=int, default=4)
    parser.add_argument("--frame-period", type=float, default=4e-3)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(tempfile.gettempdir()) / "mmwave_human_rt")
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help=(
            "Optional explicit summary JSON path. Defaults to "
            "--output-dir/summary.json."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Runs the requested RT benchmark modes and writes a summary JSON file."""

    args = _parse_args()
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "mmwave-radar-mpl"),
    )
    print("Runtime:", runtime_summary(), flush=True)
    radar, target, mesh_sequence, chest_front_point = _build_radar_and_target(args)
    print(f"Motion NPZ: {args.motion_npz}", flush=True)
    print(f"SMPL device: {args.smpl_device}", flush=True)
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
    print(f"Radar position: {np.asarray(radar.position)}", flush=True)
    print(f"Radar orientation: {np.asarray(radar.orientation)}", flush=True)
    print(f"Estimated chest front: {np.asarray(chest_front_point)}", flush=True)
    modes = _modes_from_args(args)
    print(
        "RT modes: " + ", ".join(modes),
        flush=True,
    )

    results = {
        "runtime": runtime_summary(),
        "motion_npz": str(args.motion_npz),
        "smpl_device": args.smpl_device,
        "num_frames": args.num_frames,
        "mesh_vertices": list(mesh_sequence.vertices.shape),
        "mesh_faces": list(mesh_sequence.faces.shape),
        "radar_position": list(radar.position),
        "radar_orientation": list(radar.orientation),
        "rt_diffuse_reflection": args.diffuse_reflection,
        "rt_coupling_mode": args.coupling_mode,
        "rt_mode": args.mode,
        "rt_modes": list(modes),
        "rt_periodic_retrace": True,
        "rt_coherent_bank_retrace_period_chirps": (
            args.coherent_bank_retrace_period_chirps),
        "rt_coherent_transition_config": {
            "enabled": args.coherent_transition,
            "crossfade": args.coherent_crossfade,
            "match_delay_tolerance_fraction": (
                args.coherent_match_delay_tolerance_fraction),
            "match_separation_fraction": (
                args.coherent_match_separation_fraction),
        },
        "rt_max_depth": args.max_depth,
        "rt_samples_per_src": args.samples_per_src,
        "rt_max_num_paths_per_src": args.max_num_paths_per_src,
        "runs": [],
    }
    for mode in modes:
        run_result = _run_rt(mode, args, radar, target)
        results["runs"].append(run_result)
        print(
            f"{mode}: {run_result['seconds']:.3f} s, "
            f"{run_result['seconds_per_chirp']:.6f} s/chirp, "
            f"mean paths={run_result['path_count_mean']:.1f}, "
            f"events={run_result['events']}",
            flush=True,
        )
    if len(results["runs"]) == 1:
        results["rt"] = results["runs"][0]

    json_path = (
        args.json if args.json is not None else args.output_dir / "summary.json"
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote JSON: {json_path}", flush=True)


if __name__ == "__main__":
    main()
