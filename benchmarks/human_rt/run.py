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

REPO_ROOT = Path(__file__).resolve().parents[2]
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

from mmWaveRadar.defaults import default_amass_npz as _default_amass_npz
from mmWaveRadar.defaults import default_smpl_model_dir as _default_smpl_model_dir
from mmWaveRadar.tutorial_support import orientation_toward as _orientation_toward
from mmWaveRadar.tutorial_support import mean_marker as _mean_marker
from mmWaveRadar.tutorial_support import amass_marker_dict as _amass_marker_dict
from mmWaveRadar.tutorial_support import smpl_chest_front as _smpl_chest_front
from mmWaveRadar import amass_to_mesh_sequence, load_amass_npz  # noqa: E402
from mmWaveRadar.materials import human_skin_material  # noqa: E402
from mmWaveRadar.naming import RT_HUMAN_BOUNCES, add_removed_options  # noqa: E402
from mmWaveRadar.radar import FMCWConfig, RadarHardware, RadarSensor  # noqa: E402
from mmWaveRadar.simulation import (  # noqa: E402
    MmWaveRadarSimulator,
    RTCoherentTransitionConfig,
    label_slug,
    runtime_summary,
)
from mmWaveRadar.targets import MeshTarget  # noqa: E402


def _radar_pose_in_front_of_chest(mesh_sequence, *, clearance_m, amass_data):
    from mmWaveRadar.tutorial_support import radar_pose_in_front_of_chest

    position, orientation, chest, _ = radar_pose_in_front_of_chest(
        mesh_sequence, clearance_m=clearance_m, amass_data=amass_data
    )
    return position, orientation, chest


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
    radar_position, radar_orientation, chest_front_point = _radar_pose_in_front_of_chest(
        mesh_sequence,
        clearance_m=args.radar_clearance_m,
        amass_data=amass_data,
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


def _frame_count(value: str) -> int:
    count = int(value)
    if count < 0:
        raise argparse.ArgumentTypeError("must be a non-negative frame count")
    return count


def _run_label(rt_refresh_frames: int) -> str:
    return f"rt_refresh_frames_{int(rt_refresh_frames)}"


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


def _run_rt(rt_refresh_frames: int, args, radar, target):
    scene = load_scene()
    transition_config = RTCoherentTransitionConfig(
        enabled=args.coherent_transition,
        match_delay_tolerance_fraction=(args.coherent_match_delay_tolerance_fraction),
        match_separation_fraction=args.coherent_match_separation_fraction,
        crossfade=args.coherent_crossfade,
    )
    label = _run_label(rt_refresh_frames)
    sim_kwargs = {
        "solver": "rt",
        "progress": args.progress,
        "rt_human_bounces": args.rt_human_bounces,
        "rt_max_depth": args.rt_max_depth,
        "rt_samples_per_source": args.rt_samples_per_source,
        "rt_max_paths_per_source": args.rt_max_paths_per_source,
        "rt_diffuse_reflection": args.rt_diffuse_reflection,
        "rt_refresh_frames": rt_refresh_frames,
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
        name=f"{target.name}_{label_slug(label)}",
        mesh_sequence=target.mesh_sequence,
        material=target.material,
    )
    simulator = MmWaveRadarSimulator(**sim_kwargs)
    retrace_period_chirps = simulator.rt_retrace_period_chirps(radar.fmcw)
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
        "label": label,
        "solver": "rt",
        "rt_refresh_frames": int(rt_refresh_frames),
        "rt_retrace_period_chirps": retrace_period_chirps,
        "rt_max_depth": args.rt_max_depth,
        "rt_samples_per_source": args.rt_samples_per_source,
        "rt_max_paths_per_source": args.rt_max_paths_per_source,
        "coherent_transition_enabled": bool(args.coherent_transition),
        "coherent_crossfade": args.coherent_crossfade,
        "coherent_match_delay_tolerance_fraction": (args.coherent_match_delay_tolerance_fraction),
        "coherent_match_separation_fraction": (args.coherent_match_separation_fraction),
        "seconds": elapsed,
        "seconds_per_frame": elapsed / args.num_frames,
        "seconds_per_chirp": elapsed / num_chirps,
        "adc_shape": list(cube.adc.shape),
        "events": int(len(cube.metadata.events)),
    }
    result.update(_count_stats("path_count", cube.metadata.path_counts))
    if cube.metadata.runtime_profile_s is not None:
        result["runtime_profile_s"] = {
            key: float(value) for key, value in cube.metadata.runtime_profile_s.items()
        }
    if cube.metadata.runtime_profile_counts is not None:
        result["runtime_profile_counts"] = {
            key: int(value) for key, value in cube.metadata.runtime_profile_counts.items()
        }
        result["runtime_profile_per_event_s"] = {
            key: (
                float(result["runtime_profile_s"].get(key, 0.0)) / int(count)
                if int(count) > 0
                else None
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
    """Parses CLI controls for RT benchmark cadence, data paths, and output."""

    parser = argparse.ArgumentParser(description="Benchmark human-motion RT on an AMASS sequence.")
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
    parser.add_argument("--compute-backend", choices=("numpy", "torch", "auto"), default=None)
    parser.add_argument("--compute-precision", choices=("float64", "float32"), default=None)
    parser.add_argument("--adc-compute-backend", choices=("numpy", "torch", "auto"), default=None)
    parser.add_argument("--adc-compute-precision", choices=("float64", "float32"), default=None)
    parser.add_argument(
        "--rt-refresh-frames",
        type=_frame_count,
        nargs="+",
        default=[0],
        help=(
            "RT retrace cadences to benchmark, one run "
            "each: 0 retraces every chirp; N retraces every "
            "N frames and phase-updates paths in between. "
            "Pass '0 1' to compare both."
        ),
    )
    parser.add_argument("--rt-max-depth", type=int, default=2)
    parser.add_argument("--rt-samples-per-source", type=int, default=20_000)
    parser.add_argument("--rt-max-paths-per-source", type=int, default=20_000)
    parser.add_argument(
        "--rt-diffuse-reflection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("Enable Sionna diffuse RT so scattering_coefficient is used."),
    )
    parser.add_argument("--rt-human-bounces", default="single", choices=RT_HUMAN_BOUNCES)
    parser.add_argument(
        "--coherent-transition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("Enable path matching between retraces for transition diagnostics/synthesis."),
    )
    parser.add_argument(
        "--coherent-crossfade",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=("Force coherent bank crossfade on/off. Omit to use the simulator policy."),
    )
    parser.add_argument("--coherent-match-delay-tolerance-fraction", type=float, default=0.5)
    parser.add_argument("--coherent-match-separation-fraction", type=float, default=0.25)
    parser.add_argument("--carrier-frequency", type=float, default=77e9)
    parser.add_argument("--slope", type=float, default=40e12)
    parser.add_argument("--chirp-duration", type=float, default=80e-6)
    parser.add_argument("--chirp-repetition-time", type=float, default=1e-3)
    parser.add_argument("--sampling-frequency", type=float, default=2e6)
    parser.add_argument("--num-adc-samples", type=int, default=128)
    parser.add_argument("--num-chirps-per-frame", type=int, default=4)
    parser.add_argument("--frame-period", type=float, default=4e-3)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(tempfile.gettempdir()) / "mmwave_human_rt"
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help=("Optional explicit summary JSON path. Defaults to --output-dir/summary.json."),
    )
    add_removed_options(
        parser,
        {
            "--mode": "--rt-refresh-frames (0 = every chirp, N = every N frames)",
            "--coherent-bank-retrace-period-chirps": "--rt-refresh-frames",
            "--max-depth": "--rt-max-depth",
            "--samples-per-src": "--rt-samples-per-source",
            "--max-num-paths-per-src": "--rt-max-paths-per-source",
            "--diffuse-reflection": "--rt-diffuse-reflection",
            "--no-diffuse-reflection": "--no-rt-diffuse-reflection",
            "--coupling-mode": "--rt-human-bounces (single, any)",
        },
    )
    return parser.parse_args()


def main() -> None:
    """Runs the requested RT benchmark cadences and writes a summary JSON file."""

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
    cadences = tuple(dict.fromkeys(args.rt_refresh_frames))
    print(
        "RT refresh frames: " + ", ".join(str(value) for value in cadences),
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
        "rt_diffuse_reflection": args.rt_diffuse_reflection,
        "rt_human_bounces": args.rt_human_bounces,
        "rt_refresh_frames": list(cadences),
        "rt_coherent_transition_config": {
            "enabled": args.coherent_transition,
            "crossfade": args.coherent_crossfade,
            "match_delay_tolerance_fraction": (args.coherent_match_delay_tolerance_fraction),
            "match_separation_fraction": (args.coherent_match_separation_fraction),
        },
        "rt_max_depth": args.rt_max_depth,
        "rt_samples_per_source": args.rt_samples_per_source,
        "rt_max_paths_per_source": args.rt_max_paths_per_source,
        "runs": [],
    }
    for rt_refresh_frames in cadences:
        run_result = _run_rt(rt_refresh_frames, args, radar, target)
        results["runs"].append(run_result)
        print(
            f"{run_result['label']}: {run_result['seconds']:.3f} s, "
            f"{run_result['seconds_per_chirp']:.6f} s/chirp, "
            f"mean paths={run_result['path_count_mean']:.1f}, "
            f"events={run_result['events']}",
            flush=True,
        )
    if len(results["runs"]) == 1:
        results["rt"] = results["runs"][0]

    json_path = args.json if args.json is not None else args.output_dir / "summary.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote JSON: {json_path}", flush=True)


if __name__ == "__main__":
    main()
