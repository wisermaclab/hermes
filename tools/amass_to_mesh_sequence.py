#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Convert AMASS pose NPZ files to MeshSequence NPZ files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from mmWaveRadar import amass_to_mesh_sequence, load_amass_npz


def _default_smpl_model_dir() -> Path:
    """Resolves the model override or the repository-local model directory."""

    value = os.environ.get("MMWAVE_SMPL_MODEL_DIR")
    if value:
        return Path(value).expanduser()
    return REPO_ROOT / "models" / "smpl_models"


def _slice_amass_npz(
    *,
    source_path: Path,
    output_path: Path,
    start_frame: int,
    num_frames: int | None,
) -> tuple[Path, int, int]:
    data = load_amass_npz(str(source_path))
    if "poses" not in data:
        raise ValueError(f"{source_path} is missing required AMASS array 'poses'")
    total_frames = int(np.asarray(data["poses"]).shape[0])
    if total_frames == 0:
        raise ValueError(f"{source_path} contains no pose frames")
    start = int(start_frame)
    if start < 0 or start >= total_frames:
        raise ValueError(
            f"start_frame must be between 0 and {total_frames - 1}"
        )
    if num_frames is None:
        stop = total_frames
    else:
        count = int(num_frames)
        if count <= 0:
            raise ValueError("num_frames must be positive")
        stop = min(total_frames, start + count)
    payload = {}
    for key, value in data.items():
        if key in ("poses", "trans", "dmpls", "marker_data"):
            payload[key] = value[start:stop]
        else:
            payload[key] = value
    np.savez_compressed(output_path, **payload)
    return output_path, start, stop - start


def convert_amass_to_mesh_sequence(
    *,
    amass_npz: str | Path,
    output_npz: str | Path,
    smpl_model_dir: str | Path | None = None,
    model_type: str = "smpl",
    device: str = "cpu",
    start_frame: int = 0,
    num_frames: int | None = None,
    force: bool = False,
    keep_sliced_amass: bool = False,
) -> dict:
    """Converts AMASS parameters to a mesh-sequence NPZ and returns a summary."""

    amass_npz = Path(amass_npz).expanduser()
    output_npz = Path(output_npz).expanduser()
    if smpl_model_dir is None:
        smpl_model_dir = _default_smpl_model_dir()
    smpl_model_dir = Path(smpl_model_dir).expanduser()
    if not smpl_model_dir.is_dir():
        raise FileNotFoundError(
            f"SMPL model directory was not found: {smpl_model_dir}\n"
            "Obtain the licensed SMPL model files separately and place them "
            f"under {REPO_ROOT / 'models' / 'smpl_models'}, set "
            "MMWAVE_SMPL_MODEL_DIR, or pass --smpl-model-dir."
        )
    if output_npz.exists() and not force:
        raise FileExistsError(
            f"{output_npz} already exists; pass --force to overwrite"
        )
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = None
    source_for_conversion = amass_npz
    sliced_amass_path = None
    if int(start_frame) != 0 or num_frames is not None:
        if keep_sliced_amass:
            sliced_amass_path = output_npz.with_suffix(".amass_slice.npz")
            source_for_conversion, start, count = _slice_amass_npz(
                source_path=amass_npz,
                output_path=sliced_amass_path,
                start_frame=start_frame,
                num_frames=num_frames,
            )
        else:
            temp_dir = tempfile.TemporaryDirectory(
                prefix="mmwave-amass-slice-"
            )
            sliced_amass_path = Path(temp_dir.name) / "amass_slice.npz"
            source_for_conversion, start, count = _slice_amass_npz(
                source_path=amass_npz,
                output_path=sliced_amass_path,
                start_frame=start_frame,
                num_frames=num_frames,
            )
    else:
        data = load_amass_npz(str(amass_npz))
        if "poses" not in data:
            raise ValueError(
                f"{amass_npz} is missing required AMASS array 'poses'"
            )
        start = 0
        count = int(np.asarray(data["poses"]).shape[0])
        if count == 0:
            raise ValueError(f"{amass_npz} contains no pose frames")

    try:
        mesh_sequence = amass_to_mesh_sequence(
            str(source_for_conversion),
            str(smpl_model_dir),
            model_type=model_type,
            device=device,
            out_npz_path=str(output_npz),
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    return {
        "amass_npz": str(amass_npz),
        "output_npz": str(output_npz),
        "sliced_amass_npz": (
            "" if not keep_sliced_amass or sliced_amass_path is None
            else str(sliced_amass_path)
        ),
        "smpl_model_dir": str(smpl_model_dir),
        "model_type": str(model_type),
        "device": str(device),
        "start_frame": int(start),
        "num_frames": int(count),
        "vertices_shape": list(mesh_sequence.vertices.shape),
        "faces_shape": list(mesh_sequence.faces.shape),
        "time_start_s": float(mesh_sequence.times[0]),
        "time_end_s": float(mesh_sequence.times[-1]),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the CLI parser for AMASS-to-mesh conversion."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amass-npz", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
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
    parser.add_argument("--model-type", default="smpl")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=None,
                        help="Shortcut for --start-frame N --num-frames 1.")
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--keep-sliced-amass", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    """Runs the CLI conversion and prints a JSON summary."""

    args = build_arg_parser().parse_args()
    start_frame = (
        int(args.frame_index)
        if args.frame_index is not None
        else int(args.start_frame)
    )
    num_frames = 1 if args.frame_index is not None else args.num_frames
    summary = convert_amass_to_mesh_sequence(
        amass_npz=args.amass_npz,
        output_npz=args.output_npz,
        smpl_model_dir=args.smpl_model_dir,
        model_type=args.model_type,
        device=args.device,
        start_frame=start_frame,
        num_frames=num_frames,
        force=args.force,
        keep_sliced_amass=args.keep_sliced_amass,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
