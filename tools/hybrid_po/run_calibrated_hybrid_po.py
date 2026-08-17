#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Run the calibrated Hybrid PO demo flow from the command line."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np


def _repo_root() -> Path:
    """Finds the simulator repository root from this script location."""

    root = Path(__file__).resolve()
    for parent in (root, *root.parents):
        if (parent / "src" / "mmWaveRadar").exists():
            return parent
    raise RuntimeError("could not locate repository root containing src/mmWaveRadar")


REPO_ROOT = _repo_root()
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from mmWaveRadar import amass_to_smpl_motion_sequence, load_amass_npz  # noqa: E402
from mmWaveRadar.materials import human_skin_material  # noqa: E402
from mmWaveRadar.radar import FMCWConfig, RadarSensor  # noqa: E402
from mmWaveRadar.simulation import (  # noqa: E402
    run_calibrated_hybrid_po,
    runtime_summary,
    save_radar_cube_npz,
    select_torch_device,
)
from mmWaveRadar.targets import MeshTarget  # noqa: E402
from mmWaveRadar.tutorial_support import (  # noqa: E402
    count_radar_frames_in_mesh_sequence,
    load_bedroom_scene,
    radar_pose_in_front_of_chest,
    time_window_mesh_sequence,
)


def _default_amass_npz(dataset_root: Path) -> Path:
    """Resolves the motion override or the selected dataset-root fallback."""

    value = os.environ.get("MMWAVE_AMASS_NPZ")
    if value:
        return Path(value).expanduser()
    return (
        dataset_root.expanduser()
        / "AMASS"
        / "walking_poses_cmu_105_02.npz"
    )


def _default_smpl_model_dir() -> Path:
    """Resolves the model override or the repository-local model directory."""

    value = os.environ.get("MMWAVE_SMPL_MODEL_DIR")
    if value:
        return Path(value).expanduser()
    return REPO_ROOT / "models" / "smpl_models"


def _parse_args() -> argparse.Namespace:
    """Parses command-line controls for the calibrated Hybrid PO run."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the same calibrated Hybrid PO flow used by demo/Hybrid-PO.ipynb."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("MMWAVE_DATASET_ROOT", REPO_ROOT / "data")),
    )
    parser.add_argument(
        "--amass-npz",
        type=Path,
        default=None,
        help=(
            "AMASS motion NPZ. Defaults to $MMWAVE_AMASS_NPZ, or to "
            "$MMWAVE_DATASET_ROOT/AMASS/walking_poses_cmu_105_02.npz, "
            "with the bundled CMU walking fixture as the final fallback."
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
    parser.add_argument("--smpl-device", default=None)
    parser.add_argument("--motion-start-time-s", type=float, default=0.0)
    parser.add_argument("--simulation-duration-s", type=float, default=5.0)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--num-chirps-per-frame", type=int, default=64)
    parser.add_argument("--num-adc-samples", type=int, default=225)
    parser.add_argument("--ti-board-model", default="IWR6843AOPEVM")
    parser.add_argument("--ti-pattern-mode", default="cosine30")
    parser.add_argument("--po-visibility-samples-per-face", type=int, default=4)
    parser.add_argument(
        "--coupling-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--coupling-max-reflectors",
        type=int,
        default=0,
        help="0 means all RT single-specular reflector candidates.",
    )
    parser.add_argument("--coupling-reflection-scale", type=float, default=1.0)
    parser.add_argument("--rt-samples-per-src", type=int, default=80_000)
    parser.add_argument("--rt-max-num-paths-per-src", type=int, default=80_000)
    parser.add_argument("--calibration-frame-start", type=int, default=0)
    parser.add_argument("--calibration-frame-stop", type=int, default=1)
    parser.add_argument(
        "--compute-backend",
        choices=("auto", "numpy", "torch"),
        default="auto",
    )
    parser.add_argument(
        "--compute-precision",
        choices=("float32", "float64"),
        default="float32",
    )
    parser.add_argument(
        "--adc-compute-precision",
        choices=("float32", "float64"),
        default=None,
    )
    parser.add_argument("--progress-every-frames", type=int, default=10)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-reflector-summary", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _build_radar(args: argparse.Namespace) -> RadarSensor:
    """Builds the TI radar model and FMCW profile requested by the CLI."""

    fmcw = FMCWConfig(
        carrier_frequency=60e9,
        slope=68e12,
        chirp_duration=58e-6,
        chirp_repetition_time=65e-6,
        sampling_frequency=4.5e6,
        num_adc_samples=int(args.num_adc_samples),
        num_chirps_per_frame=int(args.num_chirps_per_frame),
        frame_period=50e-3,
    )
    return RadarSensor.from_ti_board(
        args.ti_board_model,
        fmcw=fmcw,
        pattern_mode=args.ti_pattern_mode,
    )


def _build_target_and_radar_pose(
    args: argparse.Namespace,
    radar: RadarSensor,
) -> tuple[MeshTarget, RadarSensor]:
    """Loads AMASS/SMPL inputs and places the radar in front of the subject."""

    dataset_root = args.dataset_root.expanduser()
    amass_npz = args.amass_npz or _default_amass_npz(dataset_root)
    if args.smpl_model_dir is None:
        raise FileNotFoundError(
            "AMASS motion and an SMPL model are required. The bundled CMU "
            "motion is used by default. Obtain the licensed SMPL model files "
            "separately and place them under "
            f"{REPO_ROOT / 'models' / 'smpl_models'}, set "
            "MMWAVE_SMPL_MODEL_DIR, or pass --smpl-model-dir."
        )
    amass_npz = amass_npz.expanduser()
    smpl_model_dir = args.smpl_model_dir.expanduser()
    missing = [path for path in (amass_npz, smpl_model_dir) if not path.exists()]
    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "required AMASS/SMPL inputs are missing:\n"
            f"{lines}\n"
            "Obtain the licensed SMPL model files separately and place them "
            f"under {REPO_ROOT / 'models' / 'smpl_models'}, set "
            "MMWAVE_SMPL_MODEL_DIR, or pass --smpl-model-dir. Override the "
            "bundled motion with --amass-npz, MMWAVE_AMASS_NPZ, or "
            "MMWAVE_DATASET_ROOT when needed."
        )

    smpl_device = args.smpl_device or select_torch_device()
    amass_data = load_amass_npz(str(amass_npz))
    raw_mesh_sequence = amass_to_smpl_motion_sequence(
        str(amass_npz),
        str(smpl_model_dir),
        model_type=args.smpl_model_type,
        device=smpl_device,
    )
    mesh_sequence = time_window_mesh_sequence(
        raw_mesh_sequence,
        start_time_s=float(args.motion_start_time_s),
        duration_s=float(args.simulation_duration_s),
    )
    (
        radar_position,
        radar_orientation,
        chest_front_point,
        _,
    ) = radar_pose_in_front_of_chest(
        mesh_sequence,
        clearance_m=0.50,
        amass_data=amass_data,
    )
    posed_radar = RadarSensor.from_ti_board(
        args.ti_board_model,
        fmcw=radar.fmcw,
        position=tuple(float(v) for v in radar_position),
        orientation=tuple(float(v) for v in radar_orientation),
        pattern_mode=args.ti_pattern_mode,
    )
    target = MeshTarget(
        name="amass_human_po",
        mesh_sequence=mesh_sequence,
        material=human_skin_material("human-po-skin"),
    )

    print("Runtime:", runtime_summary(), flush=True)
    print(f"AMASS path: {amass_npz}", flush=True)
    print(f"SMPL model dir: {smpl_model_dir}", flush=True)
    print(f"SMPL device: {smpl_device}", flush=True)
    print(
        "Mesh sequence: "
        f"frames={len(mesh_sequence.times)} vertices={mesh_sequence.vertex_count} "
        f"faces={mesh_sequence.faces.shape[0]} "
        f"time={float(mesh_sequence.times[0]):.3f}-{float(mesh_sequence.times[-1]):.3f}s",
        flush=True,
    )
    print(
        "Radar hardware: "
        f"{posed_radar.hardware.name}, Tx={posed_radar.hardware.num_tx}, "
        f"Rx={posed_radar.hardware.num_rx}, "
        f"virtual_channels={posed_radar.hardware.num_virtual_channels}",
        flush=True,
    )
    print(
        "Radar pose: "
        f"position={np.array2string(np.asarray(posed_radar.position), precision=3)} "
        f"orientation={np.array2string(np.asarray(posed_radar.orientation), precision=3)} "
        f"chest_front={np.array2string(np.asarray(chest_front_point), precision=3)}",
        flush=True,
    )
    return target, posed_radar


def _num_frames(args: argparse.Namespace, mesh_sequence, fmcw: FMCWConfig) -> int:
    """Clamps the requested frame count to the available mesh time span."""

    max_sequence_frames = count_radar_frames_in_mesh_sequence(mesh_sequence, fmcw)
    requested = (
        int(args.num_frames)
        if args.num_frames is not None
        else max(
            1,
            int(np.floor(float(args.simulation_duration_s) / fmcw.frame_period)),
        )
    )
    return min(max_sequence_frames, requested)


def _print_runtime_profile(result) -> None:
    """Prints the stage-level runtime table stored in cube metadata."""

    profile_s = dict(result.cube.metadata.runtime_profile_s or {})
    profile_counts = dict(result.cube.metadata.runtime_profile_counts or {})
    num_chirps = max(int(np.prod(result.cube.adc.shape[:2])), 1)
    rows = [
        ("H_direct", "direct human PO", ["hybrid_human_po"]),
        ("E_static", "static RT trace", ["hybrid_static_trace"]),
        ("E_static", "path-bank extraction", ["hybrid_segment_extraction"]),
        ("E_static", "human blocking rays", ["hybrid_blocking_rays"]),
        ("E_static", "blocked static ADC", ["hybrid_static_adc_synthesis"]),
        ("shared", "RT reflector extraction", ["hybrid_reflector_extraction"]),
        ("shared", "coupling setup/assembly", ["hybrid_coupling_setup"]),
        ("H_env", "coupling channel", ["hybrid_human_env_channel"]),
        ("H_env", "coupling ADC", ["hybrid_human_env_adc_synthesis"]),
        ("E_human", "coupling channel", ["hybrid_env_human_channel"]),
        ("E_human", "coupling ADC", ["hybrid_env_human_adc_synthesis"]),
        ("total", "assembly", ["hybrid_assembly"]),
        ("total", "end-to-end", ["hybrid_total"]),
    ]
    print(f"Profiled chirps: {num_chirps}", flush=True)
    print(
        "{:<12} {:<28} {:>12} {:>8} {:>12}".format(
            "component",
            "stage",
            "seconds",
            "calls",
            "ms/chirp",
        ),
        flush=True,
    )
    print("-" * 78, flush=True)
    for component, stage, keys in rows:
        seconds = float(sum(profile_s.get(key, 0.0) for key in keys))
        calls = int(sum(profile_counts.get(key, 0) for key in keys))
        print(
            f"{component:<12} {stage:<28} {seconds:12.3f} "
            f"{calls:8d} {1e3 * seconds / num_chirps:12.3f}",
            flush=True,
        )


def main() -> int:
    """Runs the calibrated Hybrid PO benchmark and optionally saves a cube."""

    args = _parse_args()
    radar = _build_radar(args)
    target, radar = _build_target_and_radar_pose(args, radar)
    num_frames = _num_frames(args, target.mesh_sequence, radar.fmcw)
    print(
        f"Running {num_frames} frames x {radar.fmcw.num_chirps_per_frame} chirps "
        f"({num_frames * radar.fmcw.num_chirps_per_frame} chirps total)",
        flush=True,
    )
    last_chirp_time = radar.fmcw.chirp_time(
        num_frames - 1,
        radar.fmcw.num_chirps_per_frame - 1,
    )
    print(
        f"Last chirp time: {last_chirp_time:.3f}s; "
        f"mesh ends at {float(target.mesh_sequence.times[-1]):.3f}s",
        flush=True,
    )

    wall_t0 = time.perf_counter()
    result = run_calibrated_hybrid_po(
        scene=load_bedroom_scene(merge_shapes=False),
        coupling_scene=load_bedroom_scene(merge_shapes=False),
        radar=radar,
        target=target,
        num_frames=num_frames,
        calibration_frame_range=(
            int(args.calibration_frame_start),
            int(args.calibration_frame_stop),
        ),
        po_visibility_samples_per_face=int(args.po_visibility_samples_per_face),
        coupling_enabled=bool(args.coupling_enabled),
        coupling_max_reflectors=int(args.coupling_max_reflectors),
        coupling_reflection_scale=float(args.coupling_reflection_scale),
        rt_samples_per_src=int(args.rt_samples_per_src),
        rt_max_num_paths_per_src=int(args.rt_max_num_paths_per_src),
        compute_backend=args.compute_backend,
        compute_precision=args.compute_precision,
        adc_compute_precision=args.adc_compute_precision,
        progress=not args.quiet,
        progress_every_frames=int(args.progress_every_frames),
        print_reflector_summary=not args.no_reflector_summary,
    )
    wall_elapsed = time.perf_counter() - wall_t0
    print("Hybrid PO summary", flush=True)
    print(f"  cube shape: {result.cube.adc.shape}", flush=True)
    print(f"  wall time measured by script [s]: {wall_elapsed:.2f}", flush=True)
    print(f"  pipeline wall time [s]: {result.wall_time_s:.2f}", flush=True)
    print(f"  base wall time [s]: {result.base_wall_time_s:.2f}", flush=True)
    print(f"  coupling wall time [s]: {result.coupling_wall_time_s:.2f}", flush=True)
    print(f"  reflectors: {result.reflector_count}", flush=True)
    print(f"  calibration: {result.calibration}", flush=True)
    _print_runtime_profile(result)

    if args.output is not None:
        save_radar_cube_npz(
            args.output,
            result.cube,
            wall_time_s=result.wall_time_s,
            num_chirps=result.num_chirps,
            label="hybrid_po",
        )
        print(f"Saved cube: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
