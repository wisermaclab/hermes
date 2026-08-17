#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Prepare a standard HERMES validation bundle from one mmRadarPose frame.

The tool consumes a pickle-free AMASS-like base-SMPL sequence and discovers the
trial's measured radar cube and radar-to-SMPL alignment from the dataset. The
adapter owns dataset frame selection, radar channel export, pose/radar timing,
optional laser-only environment fitting, and bundle metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, BinaryIO, Mapping
import zipfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = REPO_ROOT / "src"
TOOL_ROOT = Path(__file__).resolve().parent
for source_root in (Path(__file__).resolve().parents[1], TOOL_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from mmradarpose_environment import (
    environment_payload,
    generated_environment_files,
    prepare_environment_geometry,
)

MMRADARPOSE_ROOT_ENV_VAR = "MMRADARPOSE_ROOT"
SMPL_MODEL_DIR_ENV_VAR = "MMWAVE_SMPL_MODEL_DIR"
DEFAULT_OUTPUT_ROOT = (
    Path(tempfile.gettempdir()) / "hermes-validation-bundles" / "mmradarpose"
)
TRIAL_PATTERN = re.compile(
    r"p(?P<participant>\d+)_an(?P<angle>\d+)_ac(?P<activity>\d+)_r(?P<recording>\d+)"
)
_MMRADARPOSE_INPUT_TO_TX_RX = np.asarray(
    [
        [2, 1],
        [2, 0],
        [3, 1],
        [3, 0],
        [0, 3],
        [0, 2],
        [1, 3],
        [1, 2],
        [2, 3],
        [2, 2],
        [3, 3],
        [3, 2],
    ],
    dtype=np.int64,
).reshape(3, 4, 2)
_MMRADARPOSE_INPUT_PADDING = ((0, 0), (0, 1), (1, 0), (1, 1))
MMRADARPOSE_FRAME_RATE_HZ = 15.0
MMRADARPOSE_FAST_TIME_SAMPLES = 64
MMRADARPOSE_SLOW_TIME_LOOPS = 128
MMRADARPOSE_NUM_TX = 3
MMRADARPOSE_NUM_RX = 4
MMRADARPOSE_CARRIER_FREQUENCY_HZ = 60.0e9
MMRADARPOSE_BANDWIDTH_HZ = 1.02e9
MMRADARPOSE_CHIRP_DURATION_S = 17.0e-6
MMRADARPOSE_ADC_SAMPLING_FREQUENCY_HZ = 3.8e6
MMRADARPOSE_MEASUREMENT_DURATION_S = 32.0e-3
AMASS_TIMING_FIELDS = {
    "times": "Original selected-window source times in seconds.",
    "source_capture_times": "Absolute source capture times in seconds.",
    "bundle_times": "Times relative to the selected radar frame, which is t=0.",
}
MMRADARPOSE_CHIRP_REPETITION_TIME_S = (
    MMRADARPOSE_MEASUREMENT_DURATION_S
    / (MMRADARPOSE_SLOW_TIME_LOOPS * MMRADARPOSE_NUM_TX)
)
# Base-SMPL is evaluated internally in its native +Y-up/+Z-forward frame.
# AMASS archives rotate that world to +Z-up/-Y-forward before serialization.
Y_UP_TO_AMASS_WORLD = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
MMRADARPOSE_SLOPE_HZ_PER_S = (
    MMRADARPOSE_BANDWIDTH_HZ / MMRADARPOSE_CHIRP_DURATION_S
)
IWR6843AOP_TX_MAJOR_VIRTUAL_CHANNELS = [
    {
        "channel": int(index),
        "label": f"TX{tx} to RX{rx}",
        "tx": int(tx),
        "rx": int(rx),
        "position_lambda_yz": [float(y), float(z)],
        "phase_sign": int(sign),
    }
    for index, (tx, rx, y, z, sign) in enumerate(
        (
            (1, 1, 0.5, 1.0, -1),
            (1, 2, 0.5, 1.5, 1),
            (1, 3, 0.0, 1.0, -1),
            (1, 4, 0.0, 1.5, 1),
            (2, 1, 1.5, 0.0, -1),
            (2, 2, 1.5, 0.5, 1),
            (2, 3, 1.0, 0.0, -1),
            (2, 4, 1.0, 0.5, 1),
            (3, 1, 0.5, 0.0, -1),
            (3, 2, 0.5, 0.5, 1),
            (3, 3, 0.0, 0.0, -1),
            (3, 4, 0.0, 0.5, 1),
        )
    )
]


class ArrayArchive(dict[str, np.ndarray]):
    """In-memory, pickle-free replacement for a NumPy ``NpzFile``."""

    @property
    def files(self) -> tuple[str, ...]:
        return tuple(self)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        dest="dataset_dir",
        type=Path,
        default=default_dataset_dir(),
        help=f"mmRadarPose root. Defaults to ${MMRADARPOSE_ROOT_ENV_VAR}.",
    )
    parser.add_argument("--trial", required=True, help="Trial ID such as p1_an0_ac4_r0.")
    parser.add_argument("--radar-frame-id", dest="frame", required=True)
    parser.add_argument(
        "--amass-npz",
        type=Path,
        required=True,
        help=(
            "Pickle-free AMASS-like base-SMPL sequence with frame IDs and "
            "explicit times or mocap_framerate."
        ),
    )
    parser.add_argument(
        "--smpl-model-dir",
        type=Path,
        default=default_smpl_model_dir(),
        help=(
            "Licensed SMPL model directory. Defaults to $MMWAVE_SMPL_MODEL_DIR "
            "or models/smpl_models."
        ),
    )
    parser.add_argument(
        "--environment-geometry",
        type=Path,
        help=(
            "Reusable mmRadarPose/radar geometry, or a laser point cloud already "
            "registered to that frame. When omitted, standard dataset geometry "
            "or laser locations are discovered automatically."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Directory that receives bundle folders. Defaults below the OS "
            "temporary directory at hermes-validation-bundles/mmradarpose/"
            "<trial>/bundles."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def default_dataset_dir() -> Path | None:
    value = os.environ.get(MMRADARPOSE_ROOT_ENV_VAR)
    return Path(value).expanduser() if value else None


def default_smpl_model_dir() -> Path:
    value = os.environ.get(SMPL_MODEL_DIR_ENV_VAR)
    return Path(value).expanduser() if value else REPO_ROOT / "models" / "smpl_models"


def resolve_dataset_dir(path: Path | None) -> Path:
    if path is None:
        raise ValueError(
            "mmRadarPose dataset directory is required; pass --dataset-dir or set "
            f"${MMRADARPOSE_ROOT_ENV_VAR}"
        )
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"mmRadarPose dataset directory was not found: {resolved}")
    return resolved


def trial_parts(trial: str) -> dict[str, int]:
    match = TRIAL_PATTERN.fullmatch(str(trial).strip())
    if match is None:
        raise ValueError(
            f"Invalid mmRadarPose trial {trial!r}; expected p<id>_an<angle>_ac<activity>_r<recording>"
        )
    return {name: int(value) for name, value in match.groupdict().items()}


def frame_id(frame: int | str) -> str:
    text = str(frame).strip()
    if text.lower().startswith("frame"):
        text = text[5:]
    if not text.isdecimal():
        raise ValueError(f"Invalid frame {frame!r}; expected a non-negative integer")
    return f"{int(text):06d}"


def frame_name(frame: int | str) -> str:
    return f"frame{int(frame_id(frame)):04d}"


def resolve_user_path(path: Path) -> Path:
    out = path.expanduser()
    return out.resolve() if out.is_absolute() else (Path.cwd() / out).resolve()


def resolve_output_root(trial: str, output_root: Path | None) -> Path:
    if output_root is not None:
        return resolve_user_path(output_root)
    return (DEFAULT_OUTPUT_ROOT / trial / "bundles").resolve()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _replaceable_output(
    path: Path,
    *,
    trial: str,
    frame: str,
) -> bool:
    if not path.exists():
        return not path.is_symlink()
    if path.is_symlink() or not path.is_dir():
        return False
    if not any(path.iterdir()):
        return True
    marker = path / "bundle_summary.json"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return bool(
        isinstance(payload, dict)
        and payload.get("dataset") == "mmRadarPose"
        and payload.get("trial") == trial
        and payload.get("frame") == frame
    )


def source_path_label(source_path: Path, dataset_dir: Path) -> str:
    if _is_relative_to(source_path, dataset_dir):
        return source_path.relative_to(dataset_dir).as_posix()
    return source_path.name


def benchmark_bundle_contents(copied_files: list[str]) -> dict[str, Any]:
    """Map filenames to the semantic keys shared by benchmark sidecars."""

    present = set(copied_files)
    contents: dict[str, Any] = {
        "descriptor": "bundle.json",
        "adc": "radar_adc.npz",
        "frames": "frames.json",
        "sensor": "sensor.json",
        "amass_motion_sequence": "amass_sequence.npz",
    }
    optional = {
        "background_adc.npz": "background_adc",
        "environment.json": "environment",
    }
    for filename, key in optional.items():
        if filename in present:
            contents[key] = filename
    environment_geometry = {
        Path(filename).name: filename
        for filename in sorted(present)
        if filename.startswith("environment_geometry/")
    }
    if environment_geometry:
        contents["environment_geometry"] = environment_geometry
    return contents


def build_benchmark_metadata(
    *,
    source_label: str,
    trial: str,
    radar_frame: int | str,
    pose_frame: int | str,
    bundle_name: str,
    has_environment_geometry: bool,
    copied_files: list[str],
) -> dict[str, Any]:
    parts = trial_parts(trial)
    radar_frame_text = frame_id(radar_frame)
    pose_frame_text = frame_id(pose_frame)
    action_value = activity_label(parts["activity"])
    environment_value = "fitted" if has_environment_geometry else "empty"
    subject_count_value = "single"
    note = (
        f"Orientation angle {parts['angle']}, activity {parts['activity']}; "
        f"{environment_value} environment."
    )
    return {
        "schema_version": 1,
        "id": (
            bundle_name
            if bundle_name.startswith(("mmradarpose_", "mmradpose_"))
            else f"mmradarpose_{trial}_{bundle_name}"
        ),
        "dataset": "mmRadarPose",
        "trial": trial,
        "mmradarpose_trial_id": trial,
        "participant": parts["participant"],
        "participant_id": parts["participant"],
        "angle": parts["angle"],
        "orientation_id": parts["angle"],
        "activity": parts["activity"],
        "activity_id": parts["activity"],
        "recording": parts["recording"],
        "recording_id": parts["recording"],
        "action": action_value,
        "environment": environment_value,
        "environment_is_clean": False,
        "environment_is_cluttered": False,
        "subject_count": subject_count_value,
        "single_subject": subject_count_value == "single",
        "pose_frame_id": pose_frame_text,
        "radar_frame_id": radar_frame_text,
        "camera_frame_id": None,
        "rtpose_sequence_id": None,
        "source": source_label,
        "note": note,
        "evaluation_notes": {
            "po_human_only": "Use amass_sequence.npz and an empty scene.",
            "rt_with_environment": (
                "Use the packaged environment.json when present; otherwise the bundle "
                "contains no separately selected environment sidecar."
            ),
        },
        "motion": {
            "file": "amass_sequence.npz",
            "format": "canonical AMASS-like base-SMPL window",
            "mesh_generation": (
                "generated from the AMASS parameters and a licensed SMPL model "
                "when simulation is run"
            ),
            "parameter_source": (
                "poses, translations, betas, gender, and producer fields copied "
                "from the selected source sequence without refitting"
            ),
            "timing_fields": AMASS_TIMING_FIELDS,
        },
        "bundle_contents": benchmark_bundle_contents(copied_files),
    }


def benchmark_frame_metadata(benchmark: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "id",
        "dataset",
        "action",
        "environment",
        "subject_count",
        "pose_frame_id",
        "radar_frame_id",
        "camera_frame_id",
        "rtpose_sequence_id",
        "mmradarpose_trial_id",
    )
    return {key: benchmark[key] for key in keys}


def read_json(path: Path) -> dict[str, Any] | list[Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, (dict, list)):
        raise ValueError(f"{path} must contain a JSON object or list")
    return payload


def write_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def publish_staged_bundle(
    staging: Path,
    bundle_dir: Path,
    *,
    force: bool,
    trial: str,
    frame: str,
) -> None:
    """Publish ``staging`` without losing a prior owned bundle on failure.

    An empty destination placeholder closes the final check-to-rename window.
    When replacing an owned bundle, the old directory is retained as a backup
    until the staged bundle has been published successfully.
    """

    backup_container: Path | None = None
    backup: Path | None = None
    try:
        try:
            bundle_dir.mkdir()
        except FileExistsError:
            if not force:
                raise FileExistsError(
                    f"{bundle_dir} appeared while preparing the bundle; pass --force to replace it"
                )
            if not _replaceable_output(bundle_dir, trial=trial, frame=frame):
                raise ValueError(
                    f"Refusing to replace directory not owned by this tool: {bundle_dir}"
                )
            backup_container = Path(
                tempfile.mkdtemp(prefix=f".{bundle_dir.name}.backup.", dir=bundle_dir.parent)
            )
            backup = backup_container / "previous"
            os.replace(bundle_dir, backup)
            if not _replaceable_output(backup, trial=trial, frame=frame):
                os.replace(backup, bundle_dir)
                raise ValueError(
                    f"Refusing to replace directory not owned by this tool: {bundle_dir}"
                )
            try:
                bundle_dir.mkdir()
            except BaseException:
                if not _path_exists(bundle_dir):
                    os.replace(backup, bundle_dir)
                raise

        try:
            os.replace(staging, bundle_dir)
        except BaseException:
            try:
                bundle_dir.rmdir()
            except OSError:
                pass
            if backup is not None and _path_exists(backup) and not _path_exists(bundle_dir):
                os.replace(backup, bundle_dir)
            raise

        if backup is not None and _path_exists(backup):
            shutil.rmtree(backup)
    finally:
        if backup_container is not None and backup_container.exists():
            try:
                backup_container.rmdir()
            except OSError:
                # Preserve an old bundle that could not be restored rather than
                # deleting the only remaining copy during cleanup.
                pass


def load_array_archive(path: Path) -> ArrayArchive:
    """Load every NPZ member without permitting executable pickle payloads."""

    loaded = np.load(path, allow_pickle=False)
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise ValueError(f"Expected a NumPy .npz archive: {path}")
    with loaded:
        return ArrayArchive(
            (name, np.array(loaded[name], copy=True)) for name in loaded.files
        )


def _scalar_archive_text(data: Mapping[str, np.ndarray], name: str) -> str:
    value = np.asarray(data[name])
    if value.ndim != 0 or value.dtype.kind not in {"S", "U"}:
        raise ValueError(f"SMPL {name} must be a scalar string")
    item = value.item()
    if isinstance(item, bytes):
        try:
            return item.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"SMPL {name} must be valid UTF-8") from exc
    return str(item)


def validate_smpl_sequence(
    data: ArrayArchive,
    *,
    expected_trial: str,
) -> None:
    """Validate the pose-only AMASS-like handoff used by this adapter."""

    missing = [
        name
        for name in (
            "poses",
            "trans",
            "betas",
            "gender",
            "world_up_axis",
            "world_forward_axis",
        )
        if name not in data
    ]
    if missing:
        raise ValueError(f"SMPL archive is missing required fields: {missing}")
    poses = np.asarray(data["poses"])
    translations = np.asarray(data["trans"])
    betas = np.asarray(data["betas"])
    if poses.ndim != 2 or poses.shape[0] == 0 or poses.shape[1] != 72:
        raise ValueError("SMPL poses must have shape (T, 72) with T > 0")
    if translations.shape != (poses.shape[0], 3):
        raise ValueError("SMPL trans must have shape (T, 3) matching poses")
    if betas.shape != (10,):
        raise ValueError("SMPL betas must have shape (10,)")
    for name, values in (("poses", poses), ("trans", translations), ("betas", betas)):
        if values.dtype.kind not in {"i", "u", "f"} or not np.all(np.isfinite(values)):
            raise ValueError(f"SMPL {name} must contain only finite real values")
        with np.errstate(over="ignore", invalid="ignore"):
            narrowed = np.asarray(values, dtype=np.float32)
        if not np.all(np.isfinite(narrowed)):
            raise ValueError(f"SMPL {name} values must be representable as float32")
    if _scalar_archive_text(data, "gender").strip().lower() not in {
        "neutral",
        "male",
        "female",
    }:
        raise ValueError("SMPL gender must be neutral, male, or female")
    world_up = _scalar_archive_text(data, "world_up_axis").strip().lower()
    world_forward = _scalar_archive_text(data, "world_forward_axis").strip().lower()
    if (world_up, world_forward) != ("z", "-y"):
        raise ValueError(
            "SMPL archive must use canonical AMASS world axes: "
            "world_up_axis='z' and world_forward_axis='-y'"
        )
    if "profile" in data:
        profile = _scalar_archive_text(data, "profile").strip().lower()
        if profile and profile != "mmradarpose":
            raise ValueError(
                f"SMPL archive profile must be mmradarpose, got {profile!r}"
            )
    if "source_sequence_id" in data:
        sequence = _scalar_archive_text(data, "source_sequence_id").strip()
        if sequence and sequence != expected_trial:
            raise ValueError(
                f"SMPL source sequence {sequence!r} does not match trial {expected_trial!r}"
            )
    if "model_type" in data and _scalar_archive_text(data, "model_type").lower() != "smpl":
        raise ValueError("SMPL model_type must be 'smpl'")
    pose_frame_ids(data)
    pose_times(data)


def pose_frame_ids(data: ArrayArchive) -> np.ndarray:
    if "frame_ids" not in data:
        raise ValueError("SMPL archive must contain canonical pose frame_ids")
    total = int(np.asarray(data.get("poses", np.empty((0, 72)))).shape[0])
    raw = np.asarray(data["frame_ids"])
    if raw.ndim != 1 or len(raw) != total or raw.dtype.kind not in {"S", "U", "i", "u"}:
        raise ValueError(f"SMPL frame_ids must have shape ({total},)")
    values = np.asarray([frame_id(value) for value in raw], dtype=str)
    numeric = np.asarray([int(value) for value in values], dtype=np.int64)
    if len(numeric) > 1 and np.any(np.diff(numeric) <= 0):
        raise ValueError("SMPL frame_ids must be strictly increasing and unique")
    return values


def pose_times(data: ArrayArchive) -> np.ndarray:
    """Return the self-contained pose timeline from times or AMASS frame rate."""

    total = int(np.asarray(data.get("poses", np.empty((0, 72)))).shape[0])
    if "times" in data:
        raw = np.asarray(data["times"])
        if raw.shape != (total,) or raw.dtype.kind not in {"i", "u", "f"}:
            raise ValueError(f"SMPL times must contain {total} real numeric values")
        times = np.asarray(raw, dtype=np.float64)
    elif "mocap_framerate" in data:
        raw_rate = np.asarray(data["mocap_framerate"])
        if raw_rate.ndim != 0 or raw_rate.dtype.kind not in {"i", "u", "f"}:
            raise ValueError("SMPL mocap_framerate must be a real numeric scalar")
        rate = float(np.asarray(raw_rate, dtype=np.float64).item())
        if not np.isfinite(rate) or rate <= 0.0:
            raise ValueError("SMPL mocap_framerate must be positive and finite")
        times = np.arange(total, dtype=np.float64) / rate
    else:
        raise ValueError("SMPL archive must contain times or mocap_framerate")
    if not np.all(np.isfinite(times)):
        raise ValueError("SMPL pose times must be finite")
    if len(times) > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("SMPL pose times must be strictly increasing")
    return times


def select_pose_index(
    data: ArrayArchive,
    *,
    radar_frame: str,
) -> int:
    pose_ids = pose_frame_ids(data)
    wanted = frame_id(radar_frame)
    hits = np.flatnonzero(pose_ids == wanted)
    if len(hits) == 0:
        raise ValueError(f"Pose frame {wanted} was not found in the SMPL archive")
    return int(hits[0])


def select_motion_window(
    *,
    times: np.ndarray,
    center_index: int,
    radius: int = 1,
    required_after_s: float,
) -> np.ndarray:
    total = len(times)
    radius = int(radius)
    if radius < 0:
        raise ValueError("Motion-window radius must be non-negative")
    start = max(0, center_index - radius)
    stop = min(total, center_index + radius + 1)
    target = float(times[center_index]) + float(required_after_s)
    coverage_index = int(np.searchsorted(times, target, side="left"))
    if coverage_index >= total:
        available = float(times[-1] - times[center_index])
        raise ValueError(
            "Pose sequence does not extend far enough beyond the selected pose "
            f"({available:.9g}s available, {required_after_s:.9g}s required)"
        )
    stop = max(stop, coverage_index + 1)
    return np.arange(start, stop, dtype=np.int64)


def direct_input_paths(
    dataset_dir: Path,
    trial: str,
    args: argparse.Namespace,
) -> dict[str, Path]:
    return {
        "radar_npz": (
            dataset_dir / "radar" / f"data_cube_parsed_{trial}.npz"
        ).resolve(),
        "alignment_json": (
            dataset_dir / "radar_to_smpl_alignment_inferred.json"
        ).resolve(),
        "amass_npz": resolve_user_path(args.amass_npz),
        "smpl_model_dir": resolve_user_path(args.smpl_model_dir),
    }


def load_mmradarpose_frame(
    path: Path,
    frame_index: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Stream one raw radar frame while inspecting every NPZ member header."""

    candidate: tuple[zipfile.ZipInfo, tuple[int, ...], np.dtype] | None = None
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Expected a NumPy .npz archive: {path}") from exc
    with archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.endswith(".npy"):
                raise ValueError(f"Radar NPZ contains an unexpected member {info.filename!r}")
            with archive.open(info) as stream:
                shape, fortran_order, dtype, header_size = _read_npy_header(stream)
            if dtype.hasobject or dtype.fields is not None:
                raise ValueError(
                    f"Radar archive member {info.filename!r} must use a pickle-free plain dtype"
                )
            expected_size = header_size + math.prod(shape) * dtype.itemsize
            if info.file_size != expected_size:
                raise ValueError(f"Radar archive member {info.filename!r} has an invalid NPY size")
            if len(shape) == 5 and tuple(shape[1:3]) == (4, 4):
                if candidate is not None:
                    raise ValueError(
                        f"{path} must contain exactly one radar cube shaped (T,4,4,S,L)"
                    )
                if fortran_order:
                    raise ValueError("mmRadarPose radar cube must use C-contiguous NPY storage")
                candidate = (info, shape, dtype)
        if candidate is None:
            raise ValueError(
                f"{path} must contain exactly one radar cube shaped (T,4,4,S,L)"
            )
        info, source_shape, dtype = candidate
        expected_tail = (
            4,
            4,
            MMRADARPOSE_FAST_TIME_SAMPLES,
            MMRADARPOSE_SLOW_TIME_LOOPS,
        )
        if source_shape[1:] != expected_tail:
            raise ValueError(
                "mmRadarPose radar cube must have shape "
                f"(T,{','.join(str(value) for value in expected_tail)}), got {source_shape}"
            )
        if frame_index < 0 or frame_index >= source_shape[0]:
            raise IndexError(f"Radar frame {frame_index} is outside [0, {source_shape[0]})")
        if dtype.kind not in {"i", "u", "f", "c"}:
            raise ValueError(f"Radar cube {info.filename!r} must contain numeric values")
        frame_values = math.prod(expected_tail)
        frame_bytes = frame_values * dtype.itemsize
        with archive.open(info) as stream:
            _, _, reread_dtype, _ = _read_npy_header(stream)
            stream.seek(frame_index * frame_bytes, 1)
            payload = stream.read(frame_bytes)
        if len(payload) != frame_bytes or reread_dtype != dtype:
            raise ValueError("Radar cube ended before the selected frame was complete")
        raw_frame = np.frombuffer(payload, dtype=dtype).reshape(expected_tail)
        with np.errstate(over="ignore", invalid="ignore"):
            cube_frame = np.asarray(raw_frame, dtype=np.complex64)
    expected_tail = (
        4,
        4,
        MMRADARPOSE_FAST_TIME_SAMPLES,
        MMRADARPOSE_SLOW_TIME_LOOPS,
    )
    if cube_frame.shape != expected_tail or not np.all(np.isfinite(cube_frame)):
        raise ValueError("Selected radar frame must remain finite after conversion to complex64")
    return cube_frame, source_shape


def _read_npy_header(
    stream: BinaryIO,
) -> tuple[tuple[int, ...], bool, np.dtype, int]:
    """Read an NPY header from the current ZIP member without array payloads."""

    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version in {(2, 0), (3, 0)}:
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"Unsupported NPY version {version}")
    return tuple(int(value) for value in shape), bool(fortran_order), np.dtype(dtype), stream.tell()


def export_mmradarpose_adc(cube_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert mmRadarPose serialization into explicit 3-TX by 4-RX ADC.

    Decode the upstream archive to TX/RX axes, then flatten those axes for the
    generic bundle ADC contract. Input layout details remain private to this
    dataset adapter and are not exported as radar configuration.
    """

    source = np.asarray(cube_frame, dtype=np.complex64)
    if source.shape != (
        4,
        4,
        MMRADARPOSE_FAST_TIME_SAMPLES,
        MMRADARPOSE_SLOW_TIME_LOOPS,
    ):
        raise ValueError("One radar cube frame must have shape (4,4,64,128)")
    for row, column in _MMRADARPOSE_INPUT_PADDING:
        if np.any(source[row, column] != 0):
            raise ValueError(
                "mmRadarPose serialized-grid padding must be identically zero; "
                f"cell ({row},{column}) contains measured values"
            )
    tx_rx = np.empty(
        (
            MMRADARPOSE_NUM_TX,
            MMRADARPOSE_NUM_RX,
            MMRADARPOSE_FAST_TIME_SAMPLES,
            MMRADARPOSE_SLOW_TIME_LOOPS,
        ),
        dtype=np.complex64,
    )
    for tx_index in range(MMRADARPOSE_NUM_TX):
        for rx_index in range(MMRADARPOSE_NUM_RX):
            row, column = _MMRADARPOSE_INPUT_TO_TX_RX[tx_index, rx_index]
            tx_rx[tx_index, rx_index] = source[int(row), int(column)]
    explicit_adc = tx_rx.transpose(3, 2, 0, 1)[None, ...]
    exported = explicit_adc.reshape(
        1,
        MMRADARPOSE_SLOW_TIME_LOOPS,
        MMRADARPOSE_FAST_TIME_SAMPLES,
        MMRADARPOSE_NUM_TX * MMRADARPOSE_NUM_RX,
    )
    return exported, explicit_adc


def mmradarpose_acquisition_duration_s(adc: np.ndarray) -> float:
    if adc.ndim != 4 or adc.shape[1] == 0:
        raise ValueError("Exported ADC must have shape (1, loops, samples, channels)")
    if adc.shape[1:] != (
        MMRADARPOSE_SLOW_TIME_LOOPS,
        MMRADARPOSE_FAST_TIME_SAMPLES,
        12,
    ):
        raise ValueError("Exported ADC axes do not match the mmRadarPose acquisition")
    physical_slots = int(adc.shape[1]) * MMRADARPOSE_NUM_TX
    return float((physical_slots - 1) * MMRADARPOSE_CHIRP_REPETITION_TIME_S)


def load_alignment(
    path: Path,
    *,
    trial: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    root_payload = read_json(path)
    if not isinstance(root_payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    payload = root_payload
    if "all_angles" in payload and isinstance(payload["all_angles"], dict):
        payload = payload["all_angles"]
    if "radar_to_smpl" not in payload and trial in payload and isinstance(payload[trial], dict):
        payload = payload[trial]
    if "alignment" in payload and "radar_to_smpl" not in payload:
        nested = payload["alignment"]
        if isinstance(nested, dict):
            payload = nested
    transform = payload.get("radar_to_smpl")
    if not isinstance(transform, dict):
        raise ValueError(f"{path} must define radar_to_smpl")
    rotation = np.asarray(transform.get("rotation_matrix"), dtype=np.float64)
    translation = np.asarray(transform.get("translation"), dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("radar_to_smpl.rotation_matrix must be a finite 3x3 matrix")
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("radar_to_smpl.translation must contain three finite values")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        raise ValueError("radar_to_smpl.rotation_matrix must be a proper rotation")
    # The fitted poses/translations are already rotated from native SMPL world
    # into AMASS world. Compose the dataset calibration with that serializer
    # rotation before subtracting its translation or transforming the mesh.
    radar_to_amass_rotation = Y_UP_TO_AMASS_WORLD @ rotation
    radar_to_amass_translation = Y_UP_TO_AMASS_WORLD @ translation
    radar_from_amass = radar_to_amass_rotation.T
    axes = np.argmax(np.abs(radar_from_amass), axis=1).astype(np.int64)
    signs = radar_from_amass[np.arange(3), axes]
    signed_permutation = np.zeros((3, 3), dtype=np.float64)
    signed_permutation[np.arange(3), axes] = signs
    if sorted(axes.tolist()) != [0, 1, 2] or not np.allclose(
        radar_from_amass, signed_permutation, atol=1e-6
    ) or not np.allclose(np.abs(signs), 1.0, atol=1e-6):
        raise ValueError(
            "radar-to-AMASS rotation must invert to an axis permutation with signs"
        )
    metadata: dict[str, Any] = {
        key: value
        for key, value in payload.items()
        if key in {"radar_to_smpl", "radar_to_hss_like"}
    }
    for key in ("description", "status", "units", "convention", "important_notes"):
        if key in root_payload:
            metadata[key] = root_payload[key]
    metadata["selected_alignment"] = "all_angles" if "all_angles" in root_payload else "root"
    native_radar_from_smpl = rotation.T
    native_axes = np.argmax(np.abs(native_radar_from_smpl), axis=1).astype(np.int64)
    native_signs = native_radar_from_smpl[np.arange(3), native_axes]
    metadata["smpl_model_to_radar_output_axes"] = native_axes.tolist()
    metadata["smpl_model_to_radar_output_signs"] = native_signs.tolist()
    metadata["source"] = path.name
    return radar_to_amass_translation, axes, signs.astype(np.float32), metadata


def build_direct_center_mesh(
    *,
    smpl_data: ArrayArchive,
    amass_npz: Path,
    smpl_model_dir: Path,
    center_index: int,
    translation: np.ndarray,
    output_axes: np.ndarray,
    output_signs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from mmWaveRadar.targets.smpl import amass_to_mesh_npz

    adjusted_translation = (
        np.asarray(smpl_data["trans"][center_index : center_index + 1], dtype=np.float32)
        - np.asarray(translation, dtype=np.float32)
    )
    with tempfile.TemporaryDirectory(prefix="mmradarpose_bundle_") as tmp:
        subset = Path(tmp) / "center_smpl.npz"
        np.savez(
            subset,
            poses=np.asarray(smpl_data["poses"][center_index : center_index + 1], dtype=np.float32),
            trans=adjusted_translation,
            betas=np.asarray(smpl_data["betas"], dtype=np.float32),
            gender=smpl_data["gender"],
            mocap_framerate=np.asarray(smpl_data.get("mocap_framerate", MMRADARPOSE_FRAME_RATE_HZ)),
            source_amass_npz=np.asarray(amass_npz.name),
        )
        faces, vertices, _ = amass_to_mesh_npz(
            str(subset),
            str(smpl_model_dir),
            out_npz_path=None,
        )
    vertices = (
        np.asarray(vertices, dtype=np.float32)[..., output_axes] * output_signs
    )
    from mmWaveRadar.targets.mesh import ensure_outward_face_winding

    faces = ensure_outward_face_winding(vertices, np.asarray(faces, dtype=np.uint32))
    return vertices, faces


def write_direct_amass_sequence(
    path: Path,
    *,
    smpl_data: ArrayArchive,
    source_label: str,
    indices: np.ndarray,
    center_index: int,
    source_times: np.ndarray,
    pose_ids: np.ndarray,
    faces: np.ndarray,
    translation: np.ndarray,
    output_axes: np.ndarray,
    output_signs: np.ndarray,
    alignment_metadata: Mapping[str, Any],
) -> None:
    bundle_times = source_times[indices] - float(source_times[center_index])
    payload: dict[str, np.ndarray] = {
        "poses": np.asarray(smpl_data["poses"][indices], dtype=np.float32),
        "trans": (
            np.asarray(smpl_data["trans"][indices], dtype=np.float32)
            - np.asarray(translation, dtype=np.float32)
        ),
        "betas": np.asarray(smpl_data["betas"], dtype=np.float32),
        "times": np.asarray(source_times[indices], dtype=np.float64),
        "source_capture_times": np.asarray(source_times[indices], dtype=np.float64),
        "bundle_times": np.asarray(bundle_times, dtype=np.float64),
        "faces": np.asarray(faces, dtype=np.uint32),
        "gender": np.asarray(smpl_data["gender"]),
        "model_type": np.asarray("smpl"),
        "world_up_axis": np.asarray(smpl_data["world_up_axis"]),
        "world_forward_axis": np.asarray(smpl_data["world_forward_axis"]),
        "frame_ids": np.asarray(pose_ids[indices], dtype=str),
        "radar_frame_ids": np.asarray(pose_ids[indices], dtype=str),
        "source_amass_npz": np.asarray(source_label),
        "source_indices": np.asarray(indices, dtype=np.int64),
        "selected_source_index": np.asarray(center_index, dtype=np.int64),
        "selected_window_index": np.asarray(
            int(np.flatnonzero(indices == center_index)[0]), dtype=np.int64
        ),
        "output_axes": np.asarray(output_axes, dtype=np.int64),
        "output_signs": np.asarray(output_signs, dtype=np.float32),
        "coordinate_frame": np.asarray("mmRadarPose/radar"),
        "radar_to_smpl_translation": np.asarray(
            alignment_metadata["radar_to_smpl"]["translation"],
            dtype=np.float32,
        ),
        "radar_to_smpl_rotation": np.asarray(
            alignment_metadata["radar_to_smpl"]["rotation_matrix"],
            dtype=np.float32,
        ),
        "timing_reference": np.asarray(
            "times=source sequence; source_capture_times=source absolute; "
            "bundle_times=selected radar frame at t=0"
        ),
    }
    for name in ("mocap_framerate", "profile", "source_sequence_id"):
        if name in smpl_data:
            payload[name] = np.asarray(smpl_data[name])
    np.savez_compressed(path, **payload)


def _mesh_range_stats(vertices: np.ndarray) -> dict[str, Any]:
    values = np.asarray(vertices, dtype=np.float64)
    ranges = np.linalg.norm(values, axis=1)
    return {
        "mesh_center_xyz_m": [float(value) for value in np.mean(values, axis=0)],
        "mesh_range_min_m": float(np.min(ranges)),
        "mesh_range_median_m": float(np.median(ranges)),
        "mesh_range_max_m": float(np.max(ranges)),
    }


def write_direct_sensor_json(
    path: Path,
    *,
    trial: str,
    parts: Mapping[str, int],
    adc_shape: tuple[int, ...],
    alignment: Mapping[str, Any],
    source_radar: str,
    source_amass: str,
    frame_mapping: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> None:
    metadata = {
        "dataset": "mmRadarPose",
        "trial": trial,
        **{key: int(value) for key, value in parts.items()},
        "alignment": alignment,
        "coordinate_frames": {
            "bundle_world_frame": "mmRadarPose/radar",
            "bundle_world_axes": {
                "x": "lateral axis",
                "y": "range/boresight axis, positive away from radar",
                "z": "vertical axis",
            },
        },
        "source_radar_npz": source_radar,
        "source_amass_npz": source_amass,
        "motion_source": "amass_sequence.npz",
        "amass_sequence_included": True,
        "mesh_sequence_included": False,
        "amass_sequence_timing_fields": AMASS_TIMING_FIELDS,
        "neutral_shape": False,
        "frame_mapping": dict(frame_mapping),
        "exported_adc_shape": list(adc_shape),
        "exported_adc_axes": [
            "frame",
            "chirp_slow_time",
            "adc_fast_time",
            "virtual_channel_iwr6843aop_tx_major",
        ],
        "tdm_virtual_adc": True,
        "tdm_tx_order": [1, 2, 3],
        "tdm_note": (
            "mmRadarPose stores 128 slow-time chirp loops per transmitter; "
            "radar_adc.npz stores the 12 virtual channels in TX-major order."
        ),
        "radar_geometry": {
            "board_model": "IWR6843AOPEVM",
            "virtual_channel_order": "tx_major",
            "virtual_channels": IWR6843AOP_TX_MAJOR_VIRTUAL_CHANNELS,
        },
        "environment": dict(environment),
        "benchmark": benchmark,
    }
    write_json(
        path,
        {
            "board_model": "IWR6843AOPEVM",
            "pattern_mode": "none",
            "radar": {
                "name": f"mmradarpose_{trial}_radar",
                "position": [0.0, 0.0, 0.0],
                "orientation": [float(np.pi / 2.0), 0.0, 0.0],
            },
            "fmcw": {
                "carrier_frequency_hz": MMRADARPOSE_CARRIER_FREQUENCY_HZ,
                "bandwidth_hz": MMRADARPOSE_BANDWIDTH_HZ,
                "slope_hz_per_s": MMRADARPOSE_SLOPE_HZ_PER_S,
                "chirp_duration_s": MMRADARPOSE_CHIRP_DURATION_S,
                "chirp_repetition_time_s": MMRADARPOSE_CHIRP_REPETITION_TIME_S,
                "sampling_frequency_hz": MMRADARPOSE_ADC_SAMPLING_FREQUENCY_HZ,
                "num_adc_samples": MMRADARPOSE_FAST_TIME_SAMPLES,
                "num_chirps_per_frame": MMRADARPOSE_SLOW_TIME_LOOPS,
                "frame_period_s": 1.0 / MMRADARPOSE_FRAME_RATE_HZ,
                "num_tx": MMRADARPOSE_NUM_TX,
                "tdm_enabled": True,
                "physical_num_tx": MMRADARPOSE_NUM_TX,
                "num_rx": MMRADARPOSE_NUM_RX,
                "exported_virtual_channels": int(adc_shape[3]),
                "range_resolution_m": 0.148,
                "unambiguous_range_m": 9.49,
                "doppler_resolution_mps": 0.078,
                "unambiguous_doppler_velocity_mps": 5.02,
                "source": (
                    "Table 3 of the mmRadarPose paper; chirp repetition time is "
                    "32 ms / (128 loops * 3 transmitters)."
                ),
            },
            "virtual_channel_order": "tx_major",
            "metadata": metadata,
        },
    )


def write_direct_frames_json(
    path: Path,
    *,
    trial: str,
    radar_frame: str,
    pose_frame: str,
    source_smpl_index: int,
    source_smpl_time: float,
    frame_mapping: Mapping[str, Any],
    mesh_stats: Mapping[str, Any],
    benchmark: Mapping[str, Any],
) -> None:
    write_json(
        path,
        {
            "frames": [
                {
                    "benchmark_index": 0,
                    "motion_time_s": 0.0,
                    "radar_frame_id": radar_frame,
                    "pose_frame_id": pose_frame,
                    "camera_frame_id": None,
                    "dataset_frame_id": f"{trial}:{radar_frame}",
                    "metadata": {
                        "source_smpl_index": int(source_smpl_index),
                        "source_smpl_time_s": float(source_smpl_time),
                        "frame_mapping": dict(frame_mapping),
                        **dict(mesh_stats),
                        "benchmark": benchmark_frame_metadata(benchmark),
                    },
                }
            ]
        },
    )


def check_bundle_output_integrity(bundle_dir: Path) -> dict[str, Any]:
    for path in (SRC_PATH, REPO_ROOT / "tools"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from bundle_prepare.integrity import check_bundle_integrity

    report = check_bundle_integrity(bundle_dir)
    return {
        "valid": bool(report["valid"]),
        "radar_adc_shape": report["primary_adc_artifact"]["shape"],
        "amass_poses_shape": report["amass_sequence"]["poses_shape"],
        "frame_count": report["frames"]["count"],
    }


def _prepare_direct_bundle(args: argparse.Namespace) -> dict[str, Any]:
    trial = str(args.trial).strip()
    parts = trial_parts(trial)
    radar_frame = frame_id(args.frame)
    radar_frame_index = int(radar_frame)
    dataset_dir = resolve_dataset_dir(args.dataset_dir)
    paths = direct_input_paths(dataset_dir, trial, args)
    output_root = resolve_output_root(trial, args.output_root)
    bundle_name = frame_name(radar_frame)
    bundle_dir = (output_root / bundle_name).resolve()

    source_labels = {
        name: source_path_label(path, dataset_dir)
        for name, path in paths.items()
        if name != "smpl_model_dir"
    }
    planned = {
        "schema_version": 2,
        "tool": "prepare_mmradarpose_bundle",
        "dataset": "mmRadarPose",
        "trial": trial,
        "frame": radar_frame,
        "bundle_dir": bundle_name,
        "source_amass_npz": source_labels["amass_npz"],
        "source_radar_npz": source_labels["radar_npz"],
        "source_alignment_json": source_labels["alignment_json"],
        "amass_sequence_included": True,
        "mesh_sequence_included": False,
        "neutral_shape": False,
    }

    required = [paths["amass_npz"], paths["radar_npz"], paths["alignment_json"]]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing mmRadarPose bundle inputs: {missing}")
    if not paths["smpl_model_dir"].is_dir():
        raise FileNotFoundError(
            f"SMPL model directory was not found: {paths['smpl_model_dir']}"
        )
    if bundle_dir.exists() and not args.force:
        raise FileExistsError(f"{bundle_dir} exists; pass --force to replace it")
    if (bundle_dir.exists() or bundle_dir.is_symlink()) and not _replaceable_output(
        bundle_dir,
        trial=trial,
        frame=radar_frame,
    ):
        raise ValueError(f"Refusing to replace directory not owned by this tool: {bundle_dir}")

    smpl_data = load_array_archive(paths["amass_npz"])
    validate_smpl_sequence(smpl_data, expected_trial=trial)
    pose_ids = pose_frame_ids(smpl_data)
    motion_times = pose_times(smpl_data)
    center_index = select_pose_index(
        smpl_data,
        radar_frame=radar_frame,
    )
    pose_frame = str(pose_ids[center_index])
    radar_time = radar_frame_index / MMRADARPOSE_FRAME_RATE_HZ
    pose_time = float(motion_times[center_index])
    frame_mapping = {
        "source": "mmRadarPose common zero-based trial frame index",
        "radar_frame_id": radar_frame,
        "radar_time_s": radar_time,
        "pose_frame_id": pose_frame,
        "source_smpl_index": center_index,
        "source_smpl_time_s": pose_time,
        "pose_to_radar_time_offset_s": radar_time - pose_time,
        "mapping_override": pose_frame != radar_frame,
    }
    radar_cube_frame, _ = load_mmradarpose_frame(
        paths["radar_npz"],
        radar_frame_index,
    )
    adc, _ = export_mmradarpose_adc(radar_cube_frame)
    acquisition_duration = mmradarpose_acquisition_duration_s(adc)
    window_indices = select_motion_window(
        times=motion_times,
        center_index=center_index,
        radius=1,
        required_after_s=acquisition_duration,
    )
    translation, output_axes, output_signs, alignment = load_alignment(
        paths["alignment_json"],
        trial=trial,
    )
    relative_times = (
        motion_times[window_indices] - float(motion_times[center_index])
    )
    mesh_vertices, mesh_faces = build_direct_center_mesh(
        smpl_data=smpl_data,
        amass_npz=paths["amass_npz"],
        smpl_model_dir=paths["smpl_model_dir"],
        center_index=center_index,
        translation=translation,
        output_axes=output_axes,
        output_signs=output_signs,
    )
    mesh_stats = _mesh_range_stats(mesh_vertices[0])
    copied_files = [
        "bundle.json",
        "radar_adc.npz",
        "sensor.json",
        "frames.json",
        "environment.json",
        "amass_sequence.npz",
    ]

    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{bundle_name}.", dir=output_root))
    try:
        np.savez_compressed(staging / "radar_adc.npz", adc=adc)
        write_direct_amass_sequence(
            staging / "amass_sequence.npz",
            smpl_data=smpl_data,
            source_label=source_labels["amass_npz"],
            indices=window_indices,
            center_index=center_index,
            source_times=motion_times,
            pose_ids=pose_ids,
            faces=mesh_faces,
            translation=translation,
            output_axes=output_axes,
            output_signs=output_signs,
            alignment_metadata=alignment,
        )
        geometry_outputs = prepare_environment_geometry(
            dataset_dir=dataset_dir,
            bundle_dir=staging,
            body_vertices=mesh_vertices[0],
            provided_geometry=args.environment_geometry,
        )
        copied_files.extend(generated_environment_files(geometry_outputs))
        environment = environment_payload(geometry_outputs)
        benchmark = build_benchmark_metadata(
            source_label=f"mmRadarPose trial {trial} with AMASS-like base-SMPL sequence",
            trial=trial,
            radar_frame=radar_frame,
            pose_frame=pose_frame,
            bundle_name=bundle_name,
            has_environment_geometry=bool(geometry_outputs),
            copied_files=copied_files,
        )
        write_json(staging / "benchmark_metadata.json", benchmark)
        write_json(staging / "environment.json", environment)
        write_direct_sensor_json(
            staging / "sensor.json",
            trial=trial,
            parts=parts,
            adc_shape=tuple(int(value) for value in adc.shape),
            alignment=alignment,
            source_radar=source_labels["radar_npz"],
            source_amass=source_labels["amass_npz"],
            frame_mapping=frame_mapping,
            benchmark=benchmark,
            environment=environment,
        )
        write_direct_frames_json(
            staging / "frames.json",
            trial=trial,
            radar_frame=radar_frame,
            pose_frame=pose_frame,
            source_smpl_index=center_index,
            source_smpl_time=pose_time,
            frame_mapping=frame_mapping,
            mesh_stats=mesh_stats,
            benchmark=benchmark,
        )
        write_json(
            staging / "bundle.json",
            {
                "schema_version": 1,
                "profile": "hermes",
                "primary_adc": "primary",
                "adcs": {
                    "primary": {
                        "path": "radar_adc.npz",
                        "origin": "measurement",
                    }
                },
                "motion": {"parameters": "amass_sequence.npz"},
                "environment": "environment.json",
            },
        )
        integrity = check_bundle_output_integrity(staging)
        summary = {
            **planned,
            "action": benchmark["action"],
            "environment": benchmark["environment"],
            "pose_frame_id": pose_frame,
            "radar_frame_id": radar_frame,
            "source_smpl_index": center_index,
            "source_smpl_time_s": pose_time,
            "radar_time_s": radar_time,
            "frame_mapping": frame_mapping,
            "radar_acquisition_duration_s": acquisition_duration,
            "radar_adc_shape": list(adc.shape),
            "faces_shape": list(mesh_faces.shape),
            "amass_sequence_source_indices": [int(value) for value in window_indices],
            "amass_sequence_relative_times_s": [float(value) for value in relative_times],
            "amass_sequence_npz": "amass_sequence.npz",
            "amass_sequence_included": True,
            "mesh_sequence_included": False,
            "neutral_shape": False,
            "amass_sequence_timing": (
                "amass_times" if "times" in smpl_data else "amass_mocap_framerate"
            ),
            "amass_sequence_timing_fields": AMASS_TIMING_FIELDS,
            "coordinate_frame": "mmRadarPose/radar",
            "geometry_outputs": geometry_outputs,
            **mesh_stats,
            "generated_files": sorted([*copied_files, "benchmark_metadata.json"]),
            "benchmark": benchmark,
            "benchmark_metadata": "benchmark_metadata.json",
            "integrity_check": integrity,
            "summary_path": "bundle_summary.json",
        }
        write_json(staging / "bundle_summary.json", summary)
        publish_staged_bundle(
            staging,
            bundle_dir,
            force=args.force,
            trial=trial,
            frame=radar_frame,
        )
        return summary
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def prepare_bundle(args: argparse.Namespace) -> dict[str, Any]:
    return _prepare_direct_bundle(args)


def activity_label(activity: int) -> str:
    labels = {
        0: "T-pose",
        1: "Left upper limb extension",
        2: "Right upper limb extension",
        3: "Bilateral upper limb extension",
        4: "Bicep curls",
        5: "Front arm rotation",
        6: "Torso forward bending",
        7: "Left front lunge",
        8: "Right front lunge",
        9: "Squats",
        10: "Side lower limb extension",
        11: "Front lower limb extension",
    }
    return labels.get(int(activity), f"Activity {int(activity)}")


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = prepare_bundle(args)
    except (OSError, IndexError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
