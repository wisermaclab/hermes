#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Prepare a standard validation bundle for an RT-Pose radar frame.

The pose input is a pickle-free, AMASS-like base-SMPL sequence produced by a
separate fitting tool.  This adapter owns the RT-Pose-specific work: it maps
pose frame IDs to ``Radar_frameID`` values from ``Train.json``, aligns the pose
timeline to the selected radar frame, decodes the measured radar capture, and
writes the standard simulator bundle.  A standard dataset-local ADC cache is
reused automatically when present.

Generated frame bundles default below the operating system's temporary
directory so participant-derived outputs are not written into the source tree.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
RTPOSE_TOOL_ROOT = Path(__file__).resolve().parent
for source_root in (REPO_ROOT / "src", RTPOSE_TOOL_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from rtpose_adc import RTPOSE_MIMO_RX_ORDER, build_measured_adc_cube
from reconstruct_rtpose_background import reconstruct_background

RTPOSE_ROOT_ENV_VAR = "RTPOSE_ROOT"
SMPL_MODEL_DIR_ENV_VAR = "MMWAVE_SMPL_MODEL_DIR"
DEFAULT_OUTPUT_ROOT = (
    Path(tempfile.gettempdir()) / "hermes-validation-bundles" / "rtpose"
)
WINDOWS_RESERVED_NAMES = {
    "AUX",
    "CON",
    "NUL",
    "PRN",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
# SparseSMPLFit and AMASS serialize a right-handed world with +Z up, -Y
# forward, and +X pointing toward the subject's left. RT-Pose radar coordinates
# are [range_forward, lateral_left, up], so [x, y, z] -> [-y, x, z].
AMASS_TO_RTPOSE_AXES = np.asarray([1, 0, 2], dtype=np.int64)
AMASS_TO_RTPOSE_SIGNS = np.asarray([-1.0, 1.0, 1.0], dtype=np.float32)
AMASS_TIMING_FIELDS = {
    "times": "Original selected-segment local sample times in seconds.",
    "source_capture_times": "Absolute source capture times in seconds.",
    "bundle_times": "Times relative to the selected radar frame, which is t=0.",
    "radar_capture_times": "Absolute RT-Pose radar frame times in seconds.",
}


class ArrayArchive(dict[str, np.ndarray]):
    """In-memory, pickle-free replacement for a NumPy ``NpzFile``."""

    @property
    def files(self) -> tuple[str, ...]:
        return tuple(self)


def load_array_archive(path: Path) -> ArrayArchive:
    """Loads every NPZ array without permitting executable pickle payloads."""

    data = np.load(path, allow_pickle=False)
    if not isinstance(data, np.lib.npyio.NpzFile):
        raise ValueError(f"Expected a NumPy .npz archive: {path}")
    with data:
        return ArrayArchive(
            (key, np.array(data[key], copy=True)) for key in data.files
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = prepare_bundle(args)
    except (OSError, IndexError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        dest="rtpose_root",
        type=Path,
        default=default_rtpose_root(),
        help=(
            "RT-Pose checkout root or its Data directory. Defaults to "
            f"${RTPOSE_ROOT_ENV_VAR}."
        ),
    )
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--radar-frame-id", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Directory that receives bundle folders. Defaults to "
            "hermes-validation-bundles/rtpose/sequence_<seq>/bundles below "
            "the OS temporary directory."
        ),
    )
    parser.add_argument("--force", action="store_true")

    parser.add_argument(
        "--amass-npz",
        type=Path,
        help=(
            "Pickle-free AMASS-like base-SMPL sequence. When omitted, the "
            "adapter discovers it below Data/sequences/<sequence>/smpl. It "
            "must contain pose frame IDs and either explicit times or "
            "mocap_framerate."
        ),
    )
    parser.add_argument(
        "--smpl-model-dir",
        type=Path,
        default=default_smpl_model_dir(),
        help=(
            "Licensed SMPL model directory. Defaults to "
            f"${SMPL_MODEL_DIR_ENV_VAR}."
        ),
    )
    parser.add_argument(
        "--environment-geometry",
        type=Path,
        help=(
            "Reusable RT-Pose/radar geometry directory, scene XML, or mesh NPZ. "
            "When omitted, geometry is reused from the dataset or reconstructed "
            "from the selected LiDAR/stereo frame."
        ),
    )
    return parser


def default_rtpose_root() -> Path | None:
    """Returns the RT-Pose root from the environment, if configured."""

    value = os.environ.get(RTPOSE_ROOT_ENV_VAR)
    return Path(value).expanduser() if value else None


def default_smpl_model_dir() -> Path:
    """Returns the private SMPL model directory from the environment."""

    value = os.environ.get(SMPL_MODEL_DIR_ENV_VAR)
    return Path(value).expanduser() if value else REPO_ROOT / "models" / "smpl_models"


def resolve_rtpose_root(path: Path | None) -> Path:
    """Resolve either the RT-Pose checkout root or its ``Data`` directory."""

    if path is None:
        raise ValueError(
            f"RT-Pose dataset directory is required. Pass --dataset-dir or set "
            f"{RTPOSE_ROOT_ENV_VAR}."
        )
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"RT-Pose dataset directory was not found: {resolved}")
    # Users naturally point --dataset-dir at the directory containing
    # ``sequences``. Normalize that supported form to the checkout root because
    # Train.json and the radar calibration live beside ``Data``.
    if resolved.name.casefold() == "data" and (resolved / "sequences").is_dir():
        return resolved.parent
    return resolved


def sequence_id(value: str | int) -> str:
    text = str(value).strip()
    if not text.isdecimal():
        raise ValueError(f"Invalid RT-Pose sequence {value!r}; expected a non-negative integer")
    return str(int(text))


def replaceable_bundle(path: Path, *, sequence: str, radar_frame_id: str) -> bool:
    """Return whether ``path`` is recognizably owned by this producer."""

    if not path.exists():
        return True
    if path.is_symlink() or not path.is_dir():
        return False
    if not any(path.iterdir()):
        return True
    marker = path / "bundle_summary.json"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    benchmark = payload.get("benchmark", {}) if isinstance(payload, dict) else {}
    marker_sequence = payload.get("sequence") if isinstance(payload, dict) else None
    if marker_sequence is None and isinstance(benchmark, dict):
        marker_sequence = benchmark.get("rtpose_sequence_id")
    return bool(
        isinstance(payload, dict)
        and (payload.get("dataset") == "RT-Pose" or benchmark.get("dataset") == "RT-Pose")
        and str(payload.get("radar_frame_id")) == radar_frame_id
        and str(marker_sequence) == sequence
    )


def prepare_bundle(args: argparse.Namespace) -> dict[str, Any]:
    """Stage, validate, and atomically publish one RT-Pose bundle."""

    sequence = sequence_id(args.sequence)
    radar_frame_id = frame_id(args.radar_frame_id)
    output_root = resolve_output_root(sequence, args.output_root)
    bundle_name = rtpose_frame_name(radar_frame_id)
    bundle_dir = (output_root / bundle_name).resolve()
    if bundle_dir.parent != output_root.resolve():
        raise ValueError("Bundle destination must be directly below --output-root")
    if bundle_dir.exists() and not args.force:
        raise FileExistsError(f"{bundle_dir} exists; pass --force to replace it")
    if bundle_dir.exists() and not replaceable_bundle(
        bundle_dir,
        sequence=sequence,
        radar_frame_id=radar_frame_id,
    ):
        raise ValueError(f"Refusing to replace directory not owned by this tool: {bundle_dir}")

    output_root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{bundle_name}.", dir=output_root)
    )
    staged_args = argparse.Namespace(**vars(args))
    staged_args.output_root = staging_root
    staged_args.force = False
    try:
        result = _prepare_bundle_unstaged(staged_args)
        staged_bundle = staging_root / bundle_name
        publish_staged_bundle(
            staged_bundle,
            bundle_dir,
            force=args.force,
            sequence=sequence,
            radar_frame_id=radar_frame_id,
        )
        return result
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)


def _prepare_bundle_unstaged(args: argparse.Namespace) -> dict[str, Any]:
    rtpose_root = resolve_rtpose_root(args.rtpose_root)
    sequence = sequence_id(args.sequence)
    radar_frame_id = frame_id(args.radar_frame_id)
    labels_path = resolve_labels_path(rtpose_root)
    amass_npz = resolve_amass_npz(
        args.amass_npz,
        rtpose_root=rtpose_root,
        sequence=sequence,
        labels_path=labels_path,
        radar_frame_id=radar_frame_id,
    )
    source_smpl_archive = portable_provenance_path(amass_npz, rtpose_root, REPO_ROOT)
    source_smpl_segment = (
        amass_npz.parent.name
        if re.fullmatch(r"segment_\d+", amass_npz.parent.name)
        else None
    )
    selection_kind = "segment" if source_smpl_segment is not None else "archive"
    print(
        f"Selected SMPL {selection_kind} for radar frame {radar_frame_id}: "
        f"{source_smpl_archive}",
        file=sys.stderr,
    )
    filemeta_path = resolve_filemeta_path(rtpose_root)
    sequence_metadata = load_rtpose_sequence_metadata(filemeta_path, sequence)
    output_root = resolve_output_root(sequence, args.output_root)
    bundle_name = rtpose_frame_name(radar_frame_id)
    bundle_dir = (output_root / bundle_name).resolve()
    if bundle_dir.parent != output_root.resolve():
        raise ValueError("Bundle destination must be directly below --output-root")

    smpl_model_dir = args.smpl_model_dir.expanduser().resolve()
    if not amass_npz.is_file():
        raise FileNotFoundError(f"AMASS-like motion archive was not found: {amass_npz}")
    if not smpl_model_dir.is_dir():
        raise FileNotFoundError(f"SMPL model directory was not found: {smpl_model_dir}")
    if bundle_dir.exists():
        raise FileExistsError(f"Staged bundle path already exists: {bundle_dir}")
    bundle_dir.mkdir(parents=True)

    smpl_data = load_array_archive(amass_npz)
    validate_smpl_sequence(smpl_data, expected_sequence=sequence)
    pose_ids = pose_frame_ids(smpl_data)
    pose_times = source_pose_times(smpl_data)
    segment_times = segment_pose_times(smpl_data)
    if "source_capture_times" in smpl_data:
        pose_timing_source = "source_capture_times"
    elif "times" in smpl_data:
        pose_timing_source = "times"
    else:
        pose_timing_source = "mocap_framerate"
    aligned_records, object_index = select_rtpose_frame_mapping(
        labels_path,
        sequence,
        pose_ids=pose_ids,
        radar_frame_id=radar_frame_id,
    )
    center_index = select_center_smpl_index(
        frame_mapping=aligned_records,
        radar_frame_id=radar_frame_id,
    )
    adc_cube, adc_metadata, adc_source = load_or_build_measured_adc(
        rtpose_root,
        sequence,
        radar_frame_id,
    )
    acquisition_duration_s = radar_acquisition_duration_s(adc_cube, adc_metadata)
    window_indices = select_window_indices(
        total=int(smpl_data["poses"].shape[0]),
        center_index=center_index,
        radius=1,
        source_times=pose_times,
        required_after_s=acquisition_duration_s,
    )
    exported_adc = export_adc(adc_cube)
    np.savez_compressed(bundle_dir / "radar_adc.npz", adc=exported_adc)

    motion_times = motion_relative_times(
        smpl_data,
        window_indices=window_indices,
        center_index=center_index,
    )
    mesh_vertices, mesh_faces = build_center_mesh(
        smpl_data=smpl_data,
        amass_npz=amass_npz,
        smpl_model_dir=smpl_model_dir,
        center_index=center_index,
    )

    write_amass_sequence(
        bundle_dir / "amass_sequence.npz",
        smpl_data=smpl_data,
        source_amass_npz_label=portable_provenance_path(
            amass_npz,
            rtpose_root,
            REPO_ROOT,
        ),
        indices=window_indices,
        center_index=center_index,
        bundle_times=motion_times,
        segment_times=segment_times,
        source_capture_times=pose_times,
        pose_frame_ids=pose_ids,
        radar_frame_ids=np.asarray(
            [record["radar_frame_id"] for record in aligned_records],
            dtype=str,
        ),
        faces=mesh_faces,
    )

    frame_mapping = frame_mapping_metadata(
        aligned_records,
        source_times=pose_times,
        center_index=center_index,
        real_radar_frame_id=radar_frame_id,
    )
    mesh_stats = mesh_range_stats(mesh_vertices[0])
    camera_images = copy_rtpose_camera_frame(
        rtpose_root=rtpose_root,
        sequence=sequence,
        camera_frame_id=str(frame_mapping["source_frame_id"]),
        bundle_dir=bundle_dir,
    )

    geometry_outputs = prepare_environment_geometry(
        rtpose_root=rtpose_root,
        sequence=sequence,
        bundle_dir=bundle_dir,
        source_frame_id=str(frame_mapping["source_frame_id"]),
        body_vertices=mesh_vertices[0],
        provided_geometry=args.environment_geometry,
    )
    sanitize_staged_environment_geometry(bundle_dir)
    scene_path = stage_scene_path(
        geometry_outputs=geometry_outputs,
        bundle_dir=bundle_dir,
    )
    environment_geometry = environment_geometry_bundle_contents(bundle_dir)

    benchmark_metadata = build_benchmark_metadata(
        sequence=sequence,
        radar_frame_id=radar_frame_id,
        frame_mapping=frame_mapping,
        bundle_name=bundle_name,
        environment_geometry=environment_geometry,
        camera_images=camera_images,
        sequence_metadata=sequence_metadata,
    )
    (bundle_dir / "benchmark_metadata.json").write_text(
        json.dumps(benchmark_metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    write_frames_json(
        bundle_dir / "frames.json",
        radar_frame_id=radar_frame_id,
        frame_mapping=frame_mapping,
        mesh_stats=mesh_stats,
        benchmark_frame_metadata=benchmark_frame_metadata(benchmark_metadata),
    )
    environment_json = write_environment_json(
        bundle_dir / "environment.json",
        scene_path=scene_path,
        geometry_outputs=geometry_outputs,
        roots=(bundle_dir, rtpose_root, REPO_ROOT),
    )

    sensor_metadata = write_sensor_json(
        bundle_dir / "sensor.json",
        sequence=sequence,
        rtpose_root=rtpose_root,
        amass_npz=amass_npz,
        adc_source=adc_source,
        adc_cube=adc_cube,
        exported_adc=exported_adc,
        adc_metadata=adc_metadata,
        scene_path=scene_path,
        frame_mapping=frame_mapping,
        motion_times=motion_times,
        motion_timing=f"amass_{pose_timing_source}",
        geometry_outputs=geometry_outputs,
        benchmark_metadata=benchmark_metadata,
    )
    (bundle_dir / "bundle.json").write_text(
        json.dumps(
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
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    integrity_check = check_bundle_output_integrity(bundle_dir)

    summary = {
        "tool": "prepare_rtpose_bundle",
        "dataset": "RT-Pose",
        "sequence": sequence,
        "bundle_dir": bundle_name,
        "source_amass_npz": source_smpl_archive,
        "source_smpl_segment": source_smpl_segment,
        "source_labels": portable_provenance_path(labels_path, rtpose_root),
        "source_filemeta": portable_provenance_path(filemeta_path, rtpose_root),
        "source_sequence_metadata": sequence_metadata,
        "camera_images": camera_images,
        "rtpose_object_index": object_index,
        "selected_gender": npz_scalar_to_py(smpl_data["gender"]),
        "radar_adc_shape": list(exported_adc.shape),
        "faces_shape": list(mesh_faces.shape),
        "source_smpl_time_s": float(pose_times[center_index]),
        "source_smpl_index": int(center_index),
        "radar_acquisition_duration_s": acquisition_duration_s,
        "radar_frame_id": radar_frame_id,
        "pose_frame_id": str(frame_mapping["source_frame_id"]),
        "source_radar_frame_id": str(frame_mapping["source_radar_frame_id"]),
        **mesh_stats,
        "face_winding": "corrected_with_smpl_model_faces_to_rtpose_frame",
        "amass_sequence_npz": "amass_sequence.npz",
        "amass_sequence_included": True,
        "mesh_sequence_included": False,
        "neutral_shape": False,
        "amass_sequence_source_indices": [int(i) for i in window_indices],
        "amass_sequence_relative_times_s": [float(v) for v in motion_times],
        "amass_sequence_timing": f"amass_{pose_timing_source}",
        "amass_sequence_timing_fields": AMASS_TIMING_FIELDS,
        "simulation_motion": (
            "AMASSSMPLMotionSequence joint-angle SLERP from amass_sequence.npz"
        ),
        "sensor_metadata": sensor_metadata,
        "geometry_outputs": portable_geometry_outputs(
            geometry_outputs,
            roots=(bundle_dir, rtpose_root, REPO_ROOT),
        ),
        "environment_json": None if environment_json is None else "environment.json",
        "integrity_check": integrity_check,
    }
    summary = {
        "action": benchmark_metadata.get("action"),
        "benchmark": benchmark_metadata,
        **summary,
        "camera_frame_id": benchmark_metadata.get("camera_frame_id"),
        "environment": benchmark_metadata.get("environment"),
        "environment_geometry": environment_geometry,
        "subject_count": benchmark_metadata.get("subject_count"),
    }
    (bundle_dir / "bundle_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def check_bundle_output_integrity(bundle_dir: Path) -> dict[str, Any]:
    """Check the completed standard files before reporting success."""

    src_path = REPO_ROOT / "src"
    for path in (src_path, REPO_ROOT / "tools"):
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


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def publish_staged_bundle(
    staged: Path,
    bundle_dir: Path,
    *,
    force: bool,
    sequence: str,
    radar_frame_id: str,
) -> None:
    """Reserve the destination and retain an owned previous bundle for rollback."""

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
            if not replaceable_bundle(
                bundle_dir,
                sequence=sequence,
                radar_frame_id=radar_frame_id,
            ):
                raise ValueError(
                    f"Refusing to replace directory not owned by this tool: {bundle_dir}"
                )
            backup_container = Path(
                tempfile.mkdtemp(prefix=f".{bundle_dir.name}.backup.", dir=bundle_dir.parent)
            )
            backup = backup_container / "previous"
            os.replace(bundle_dir, backup)
            if not replaceable_bundle(
                backup,
                sequence=sequence,
                radar_frame_id=radar_frame_id,
            ):
                os.replace(backup, bundle_dir)
                raise ValueError(
                    f"Refusing to replace directory whose ownership changed: {bundle_dir}"
                )
            try:
                bundle_dir.mkdir()
            except BaseException:
                if not _path_exists(bundle_dir):
                    os.replace(backup, bundle_dir)
                raise

        try:
            os.replace(staged, bundle_dir)
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
                # Do not delete a retained previous bundle if restoration failed.
                pass


def resolve_amass_npz(
    path: Path | None,
    *,
    rtpose_root: Path,
    sequence: str,
    labels_path: Path,
    radar_frame_id: str,
) -> Path:
    """Resolve an explicit archive or select the segment for a radar frame."""

    if path is not None:
        resolved = resolve_user_path(path)
        if resolved.suffix.lower() != ".npz":
            raise ValueError("--amass-npz must name a .npz archive")
        return resolved

    smpl_dir = rtpose_root / f"Data/sequences/{sequence}/smpl"
    if smpl_dir.is_symlink() or not smpl_dir.is_dir():
        raise FileNotFoundError(
            "AMASS-like motion archive was not supplied and the RT-Pose SMPL "
            f"directory was not found: {smpl_dir}"
        )

    target_pose_ids = pose_ids_for_radar_frame(
        labels_path,
        sequence,
        radar_frame_id,
    )
    manifest = smpl_dir / "segments.json"
    if manifest.is_file() and not manifest.is_symlink():
        candidates = segment_manifest_archives(manifest, smpl_dir, sequence)
    else:
        candidates = sorted(
            candidate.resolve()
            for candidate in smpl_dir.rglob("*.npz")
            if candidate.is_file() and not candidate.is_symlink()
        )
    if not candidates:
        raise FileNotFoundError(
            "No AMASS-like .npz archive was found below the RT-Pose SMPL "
            f"directory: {smpl_dir}. Pass --amass-npz explicitly."
        )

    matching = [
        candidate
        for candidate in candidates
        if archive_pose_frame_ids(candidate) & target_pose_ids
    ]
    if not matching:
        pose_choices = ", ".join(sorted(target_pose_ids))
        raise ValueError(
            f"No fitted SMPL segment contains pose/camera frame(s) {pose_choices} "
            f"synchronized to radar frame {radar_frame_id}. The requested frame "
            "may have been rejected as bad; pass --amass-npz only if a compatible "
            "archive exists."
        )
    if len(matching) == 1:
        return matching[0]

    # A documented top-level release artifact may coexist with old checkpoint
    # archives. A segmented manifest, however, must remain non-overlapping.
    if not manifest.is_file():
        preferred = [
            candidate
            for candidate in matching
            if candidate.parent == smpl_dir
            and candidate.name in {
                "rtpose_best_gender_smpl_amass_like.npz",
                "fit.npz",
            }
        ]
        if len(preferred) == 1:
            return preferred[0]

    choices = ", ".join(str(candidate.relative_to(smpl_dir)) for candidate in matching)
    raise ValueError(
        "Multiple fitted SMPL archives contain the synchronized pose frame "
        f"for radar frame {radar_frame_id} ({choices}); pass --amass-npz "
        "explicitly."
    )


def pose_ids_for_radar_frame(
    labels_path: Path,
    sequence: str,
    radar_frame_id: str,
) -> set[str]:
    """Return every pose/camera frame synchronized to one radar frame."""

    try:
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"RT-Pose label file was not found: {labels_path}") from exc
    raw_sequence = labels.get(sequence) if isinstance(labels, dict) else None
    if not isinstance(raw_sequence, dict):
        raise KeyError(f"Sequence {sequence!r} was not found in {labels_path}")
    target = frame_id(radar_frame_id)
    pose_ids: set[str] = set()
    for raw_pose_id, annotations in raw_sequence.items():
        if not isinstance(annotations, list):
            continue
        for annotation in annotations:
            if not isinstance(annotation, dict) or "Radar_frameID" not in annotation:
                continue
            if frame_id(annotation["Radar_frameID"]) == target:
                pose_ids.add(frame_id(raw_pose_id))
                break
    if not pose_ids:
        raise KeyError(
            f"Radar frame {target} has no synchronized pose frame in sequence "
            f"{sequence} of {labels_path}"
        )
    return pose_ids


def segment_manifest_archives(
    manifest: Path,
    smpl_dir: Path,
    sequence: str,
) -> list[Path]:
    """Resolve the NPZ outputs declared by a SparseSMPLFit segment manifest."""

    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid SparseSMPLFit segment manifest: {manifest}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
        raise ValueError(f"SparseSMPLFit segment manifest has no segments list: {manifest}")
    parent_sequence = payload.get("parent_sequence_id")
    if parent_sequence is not None and sequence_id(parent_sequence) != sequence:
        raise ValueError(
            f"Segment manifest sequence {parent_sequence!r} does not match {sequence}"
        )

    root = smpl_dir.resolve()
    archives: list[Path] = []
    for index, segment in enumerate(payload["segments"]):
        output_path = segment.get("output_path") if isinstance(segment, dict) else None
        if not isinstance(output_path, str) or not output_path.strip():
            raise ValueError(f"Segment {index} in {manifest} has no output_path")
        relative = Path(output_path)
        if relative.is_absolute() or "\\" in output_path or any(
            part in {"", ".", ".."} or ":" in part for part in relative.parts
        ):
            raise ValueError(f"Segment {index} has unsafe output_path {output_path!r}")
        candidate = (smpl_dir / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Segment {index} output escapes the SMPL directory") from exc
        if candidate.is_symlink() or not candidate.is_file():
            raise FileNotFoundError(f"Segment archive was not found: {candidate}")
        if candidate.suffix.lower() != ".npz":
            raise ValueError(f"Segment output must be an NPZ archive: {candidate}")
        archives.append(candidate)
    if len(set(archives)) != len(archives):
        raise ValueError(
            f"SparseSMPLFit segment manifest contains duplicate outputs: {manifest}"
        )
    return archives


def archive_pose_frame_ids(path: Path) -> set[str]:
    """Read canonical frame IDs from one candidate without loading its arrays."""

    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not inspect AMASS-like candidate {path}: {exc}") from exc
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError(f"Expected an NPZ archive while inspecting {path}")
    with archive:
        if "frame_ids" not in archive.files:
            return set()
        values = np.asarray(archive["frame_ids"])
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"AMASS-like candidate has invalid frame_ids: {path}")
    try:
        return {frame_id(value) for value in values}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"AMASS-like candidate has invalid frame_ids: {path}") from exc


def resolve_labels_path(rtpose_root: Path) -> Path:
    """Resolve Train.json from the supported RT-Pose checkout layouts."""

    candidates = (
        rtpose_root / "RT-POSE" / "Train.json",
        rtpose_root / "Train.json",
        rtpose_root / "Data" / "Train.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "RT-Pose Train.json was not found below the dataset directory"
    )


def resolve_filemeta_path(rtpose_root: Path) -> Path:
    """Resolve RT-Pose's sequence-level activity metadata file."""

    candidates = (
        rtpose_root / "Data" / "filemeta.txt",
        rtpose_root / "filemeta.txt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "RT-Pose Data/filemeta.txt was not found below the dataset directory"
    )


def load_rtpose_sequence_metadata(path: Path, sequence: str) -> dict[str, str]:
    """Load and normalize one sequence entry from RT-Pose ``filemeta.txt``."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"RT-Pose metadata file was not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"RT-Pose metadata file is invalid JSON: {path}") from exc
    entry = payload.get(sequence) if isinstance(payload, dict) else None
    if not isinstance(entry, dict):
        raise KeyError(f"Sequence {sequence!r} was not found in {path}")

    def required_text(*names: str) -> str:
        for name in names:
            value = entry.get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
        choices = ", ".join(repr(name) for name in names)
        raise ValueError(
            f"Sequence {sequence!r} in {path} is missing string field {choices}"
        )

    return {
        "activity": required_text("Activity"),
        "location": required_text("Location"),
        # The released RT-Pose file spells this field ``Senarios``. Accept the
        # corrected spelling as well so the parser survives an upstream fix.
        "scenario": required_text("Senarios", "Scenarios", "Scenario"),
        "occlusion": required_text("Occlusion"),
        "subject_type": required_text("Subject type", "Subject Type"),
    }


def copy_rtpose_camera_frame(
    *,
    rtpose_root: Path,
    sequence: str,
    camera_frame_id: str,
    bundle_dir: Path,
) -> dict[str, str]:
    """Copy the synchronized original left/right PNGs into the bundle."""

    normalized_frame_id = frame_id(camera_frame_id)
    copied: dict[str, str] = {}
    for side in ("left", "right"):
        source = (
            rtpose_root
            / f"Data/sequences/{sequence}/camera/{side}/{normalized_frame_id}.png"
        )
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(
                f"RT-Pose {side} camera frame was not found: {source}"
            )
        with source.open("rb") as stream:
            if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                raise ValueError(f"RT-Pose camera frame is not a PNG file: {source}")
        relative = Path("camera") / side / source.name
        destination = bundle_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied[side] = relative.as_posix()
    return copied


def resolve_output_root(sequence: str, path: Path | None) -> Path:
    if path is not None:
        return resolve_user_path(path)
    return (DEFAULT_OUTPUT_ROOT / f"sequence_{sequence}" / "bundles").resolve()


def resolve_user_path(path: Path) -> Path:
    """Resolves user-provided output paths without tying them to the dataset."""

    out = path.expanduser()
    return out.resolve() if out.is_absolute() else (Path.cwd() / out).resolve()


def frame_id(value: str | int) -> str:
    text = str(value).strip()
    if text.lower().startswith("frame"):
        text = text[5:]
    if not text.isdecimal():
        raise ValueError(f"Invalid frame ID {value!r}; expected a non-negative integer")
    return f"{int(text):06d}"


def rtpose_frame_name(value: str | int) -> str:
    """Returns RT-Pose's four-digit frame directory/file stem."""

    return f"frame{int(frame_id(value)):04d}"


def validate_smpl_sequence(
    data: ArrayArchive,
    *,
    expected_sequence: str | None = None,
) -> None:
    """Validate the pose-only AMASS-like handoff used by this adapter."""

    missing = [
        key
        for key in (
            "poses",
            "trans",
            "betas",
            "gender",
            "world_up_axis",
            "world_forward_axis",
        )
        if key not in data
    ]
    if missing:
        raise ValueError(f"SMPL archive is missing required fields: {missing}")
    poses = np.asarray(data["poses"])
    trans = np.asarray(data["trans"])
    betas = np.asarray(data["betas"]).reshape(-1)
    if poses.ndim != 2 or poses.shape[1] != 72 or poses.shape[0] == 0:
        raise ValueError("SMPL poses must have shape (T, 72) with T > 0")
    if trans.shape != (poses.shape[0], 3):
        raise ValueError("SMPL trans must have shape (T, 3) matching poses")
    if betas.shape != (10,):
        raise ValueError("SMPL betas must have shape (10,)")
    for name, values in (("poses", poses), ("trans", trans), ("betas", betas)):
        if values.dtype.kind not in {"i", "u", "f"} or not np.all(np.isfinite(values)):
            raise ValueError(f"SMPL {name} must contain only finite real values")
        with np.errstate(over="ignore", invalid="ignore"):
            narrowed = np.asarray(values, dtype=np.float32)
        if not np.all(np.isfinite(narrowed)):
            raise ValueError(f"SMPL {name} values must be representable as float32")
    gender = np.asarray(data["gender"])
    if gender.ndim != 0 or str(npz_scalar_to_py(gender)).lower() not in {
        "neutral",
        "male",
        "female",
    }:
        raise ValueError("SMPL gender must be a scalar neutral, male, or female string")
    world_up = str(npz_scalar_to_py(data["world_up_axis"])).strip().lower()
    world_forward = str(npz_scalar_to_py(data["world_forward_axis"])).strip().lower()
    if (world_up, world_forward) != ("z", "-y"):
        raise ValueError(
            "SMPL archive must use canonical AMASS world axes: "
            "world_up_axis='z' and world_forward_axis='-y'"
        )
    if "profile" in data:
        profile = np.asarray(data["profile"])
        if profile.ndim != 0:
            raise ValueError("SMPL profile provenance must be a scalar string")
        profile_name = str(npz_scalar_to_py(profile)).strip().lower()
        if profile_name and profile_name != "rtpose":
            raise ValueError(f"SMPL archive profile must be rtpose, got {profile_name!r}")
    if "model_type" in data:
        model_type = np.asarray(data["model_type"])
        if model_type.ndim != 0:
            raise ValueError("SMPL model_type must be a scalar string")
        model_name = str(npz_scalar_to_py(model_type)).strip().lower()
        if model_name != "smpl":
            raise ValueError(f"SMPL model_type must be 'smpl', got {model_name!r}")
    if expected_sequence is not None and "source_sequence_id" in data:
        source_sequence = np.asarray(data["source_sequence_id"])
        if source_sequence.ndim != 0:
            raise ValueError("SMPL source_sequence_id must be a scalar")
        observed_sequence = sequence_id(npz_scalar_to_py(source_sequence))
        if observed_sequence != expected_sequence:
            raise ValueError(
                f"SMPL source sequence {observed_sequence} does not match "
                f"requested sequence {expected_sequence}"
            )
    pose_frame_ids(data)
    source_pose_times(data)


def pose_frame_ids(data: ArrayArchive) -> np.ndarray:
    """Return canonical pose/camera frame IDs without consuming radar metadata."""

    if "frame_ids" not in data:
        raise ValueError("SMPL archive must contain canonical pose frame_ids")
    values = np.asarray(data["frame_ids"])
    total = int(np.asarray(data.get("poses", np.empty((0, 72)))).shape[0])
    if values.ndim != 1 or len(values) != total:
        raise ValueError(f"SMPL frame_ids must have shape ({total},)")
    normalized = np.asarray([frame_id(value) for value in values], dtype=str)
    if len(set(normalized.tolist())) != len(normalized):
        raise ValueError("SMPL frame_ids contains duplicate pose frame IDs")
    numeric_ids = np.asarray([int(value) for value in normalized], dtype=np.int64)
    if len(numeric_ids) > 1 and np.any(np.diff(numeric_ids) <= 0):
        raise ValueError("SMPL frame_ids must be strictly increasing in pose-time order")
    return normalized


def source_pose_times(data: ArrayArchive) -> np.ndarray:
    """Return absolute source capture times when available."""

    total = int(np.asarray(data.get("poses", np.empty((0, 72)))).shape[0])
    if "source_capture_times" not in data:
        return segment_pose_times(data)
    raw_times = np.asarray(data["source_capture_times"])
    if raw_times.shape != (total,) or raw_times.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"SMPL source_capture_times must have numeric shape ({total},)")
    times = np.asarray(raw_times, dtype=np.float64)
    validate_pose_timeline(times, label="SMPL source capture times")
    return times


def segment_pose_times(data: ArrayArchive) -> np.ndarray:
    """Return the AMASS segment-local timeline from times or frame rate."""

    total = int(np.asarray(data.get("poses", np.empty((0, 72)))).shape[0])
    if "times" in data:
        raw_times = np.asarray(data["times"])
        if raw_times.shape != (total,):
            raise ValueError(f"SMPL times must have shape ({total},)")
        if raw_times.dtype.kind not in {"i", "u", "f"}:
            raise ValueError("SMPL times must contain real numeric values")
        times = np.asarray(raw_times, dtype=np.float64)
    elif "mocap_framerate" in data:
        raw_rate = np.asarray(data["mocap_framerate"])
        if raw_rate.dtype.kind not in {"i", "u", "f"}:
            raise ValueError("SMPL mocap_framerate must be real numeric")
        rate_array = np.asarray(raw_rate, dtype=np.float64)
        if rate_array.ndim != 0:
            raise ValueError("SMPL mocap_framerate must be a scalar")
        rate = float(rate_array.item())
        if not np.isfinite(rate) or rate <= 0.0:
            raise ValueError("SMPL mocap_framerate must be positive and finite")
        times = np.arange(total, dtype=np.float64) / rate
    else:
        raise ValueError("SMPL archive must contain times or mocap_framerate")
    validate_pose_timeline(times, label="SMPL pose times")
    return times


def validate_pose_timeline(times: np.ndarray, *, label: str) -> None:
    """Validate one finite, strictly increasing pose timeline."""

    if not np.all(np.isfinite(times)):
        raise ValueError(f"{label} must be finite")
    if len(times) > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError(f"{label} must be strictly increasing")


def load_rtpose_frame_mapping(
    labels_path: Path,
    sequence: str,
    object_index: int,
) -> dict[str, dict[str, Any]]:
    """Load pose-to-radar alignment from RT-Pose labels, independently of SMPL."""

    if object_index < 0:
        raise ValueError("RT-Pose annotation object index must be non-negative")
    try:
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"RT-Pose label file was not found: {labels_path}") from exc
    if not isinstance(labels, dict) or sequence not in labels:
        raise KeyError(f"Sequence {sequence!r} was not found in {labels_path}")
    raw_sequence = labels[sequence]
    if not isinstance(raw_sequence, dict) or not raw_sequence:
        raise ValueError(f"Sequence {sequence!r} has no labeled frames")

    mapping: dict[str, dict[str, Any]] = {}
    for raw_pose_id, annotations in raw_sequence.items():
        pose_id = frame_id(raw_pose_id)
        if pose_id in mapping:
            raise ValueError(f"RT-Pose labels contain duplicate pose frame {pose_id}")
        if not isinstance(annotations, list):
            raise ValueError(f"RT-Pose frame {pose_id} annotations must be a list")
        if object_index >= len(annotations):
            continue
        annotation = annotations[object_index]
        if not isinstance(annotation, dict) or "Radar_frameID" not in annotation:
            raise ValueError(
                f"Sequence {sequence} pose frame {pose_id} object {object_index} "
                "has no Radar_frameID"
            )
        try:
            lidar_valid = int(annotation.get("LiDAR val", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Sequence {sequence} pose frame {pose_id} has invalid LiDAR val"
            ) from exc
        mapping[pose_id] = {
            "pose_frame_id": pose_id,
            "camera_frame_id": pose_id,
            "radar_frame_id": frame_id(annotation["Radar_frameID"]),
            "lidar_valid": lidar_valid,
        }
    if not mapping:
        raise IndexError(
            f"Sequence {sequence} has no annotations at object index {object_index}"
        )
    return mapping


def rtpose_object_count(labels_path: Path, sequence: str) -> int:
    """Return the largest annotation-list size in one RT-Pose sequence."""

    try:
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"RT-Pose label file was not found: {labels_path}") from exc
    raw_sequence = labels.get(sequence) if isinstance(labels, dict) else None
    if not isinstance(raw_sequence, dict) or not raw_sequence:
        raise KeyError(f"Sequence {sequence!r} was not found in {labels_path}")
    sizes = [len(value) for value in raw_sequence.values() if isinstance(value, list)]
    if not sizes or max(sizes) == 0:
        raise ValueError(f"Sequence {sequence!r} has no person annotations")
    return max(sizes)


def select_rtpose_frame_mapping(
    labels_path: Path,
    sequence: str,
    *,
    pose_ids: np.ndarray,
    radar_frame_id: str,
) -> tuple[list[dict[str, Any]], int]:
    """Select the sole participant mapping compatible with the motion archive."""

    candidates: list[tuple[list[dict[str, Any]], int]] = []
    for object_index in range(rtpose_object_count(labels_path, sequence)):
        try:
            mapping = load_rtpose_frame_mapping(labels_path, sequence, object_index)
            aligned = align_pose_frames_to_rtpose(pose_ids, mapping)
        except (IndexError, ValueError):
            continue
        radar_ids = [record["radar_frame_id"] for record in aligned]
        if radar_ids.count(radar_frame_id) == 1:
            candidates.append((aligned, object_index))
    if not candidates:
        raise ValueError(
            f"Radar frame {radar_frame_id} has no participant motion in sequence {sequence}"
        )
    if len(candidates) > 1:
        indices = ", ".join(str(index) for _, index in candidates)
        raise ValueError(
            f"Radar frame {radar_frame_id} is ambiguous across RT-Pose objects "
            f"{indices}; provide a single-participant AMASS-like sequence"
        )
    return candidates[0]


def align_pose_frames_to_rtpose(
    pose_ids: np.ndarray,
    mapping: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Align every fitted pose frame to an RT-Pose annotation record."""

    missing = [pose_id for pose_id in pose_ids.tolist() if pose_id not in mapping]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"SMPL pose frames are missing from RT-Pose labels: {preview}")
    return [dict(mapping[pose_id]) for pose_id in pose_ids.tolist()]


def select_center_smpl_index(
    *,
    frame_mapping: list[dict[str, Any]],
    radar_frame_id: str,
) -> int:
    radar_ids = np.asarray([record["radar_frame_id"] for record in frame_mapping], dtype=str)
    hits = np.where(radar_ids == radar_frame_id)[0]
    if len(hits) == 0:
        raise ValueError(f"Radar frame {radar_frame_id} has no fitted pose in RT-Pose labels")
    if len(hits) > 1:
        raise ValueError(
            f"Radar frame {radar_frame_id} maps to multiple fitted poses"
        )
    return int(hits[0])


def select_window_indices(
    *,
    total: int,
    center_index: int,
    radius: int,
    source_times: np.ndarray | None = None,
    required_after_s: float = 0.0,
) -> np.ndarray:
    """Select a local pose window that covers the complete radar acquisition."""

    if total <= 0:
        raise ValueError("SMPL archive contains no poses")
    if center_index < 0 or center_index >= total:
        raise IndexError(f"SMPL index {center_index} is outside [0, {total})")
    radius = int(radius)
    if radius < 0:
        raise ValueError("--window-radius must be non-negative")
    start = max(0, center_index - radius)
    stop = min(total, center_index + radius + 1)
    required_after_s = float(required_after_s)
    if not np.isfinite(required_after_s) or required_after_s < 0.0:
        raise ValueError("Required post-center motion duration must be finite and non-negative")
    if required_after_s > 0.0:
        if source_times is None:
            raise ValueError("Pose times are required to cover the radar acquisition")
        times = np.asarray(source_times, dtype=np.float64)
        if times.shape != (total,) or not np.all(np.isfinite(times)):
            raise ValueError(f"Pose times must have shape ({total},) and be finite")
        if len(times) > 1 and np.any(np.diff(times) <= 0.0):
            raise ValueError("Pose times must be strictly increasing")
        target_time = float(times[center_index]) + required_after_s
        coverage_index = int(np.searchsorted(times, target_time, side="left"))
        if coverage_index >= total:
            available = float(times[-1] - times[center_index])
            raise ValueError(
                "Pose sequence does not extend far enough beyond the selected pose "
                f"to cover the radar acquisition ({available:.9g}s available, "
                f"{required_after_s:.9g}s required)"
            )
        stop = max(stop, coverage_index + 1)
    return np.arange(start, stop, dtype=np.int64)


def motion_relative_times(
    data: ArrayArchive,
    *,
    window_indices: np.ndarray,
    center_index: int,
) -> np.ndarray:
    source_times = source_pose_times(data)
    return source_times[window_indices] - float(source_times[center_index])


def load_or_build_measured_adc(
    rtpose_root: Path,
    sequence: str,
    radar_frame_id: str,
) -> tuple[np.ndarray, dict[str, Any], Path]:
    """Load a prepared cube or decode it directly from the measured capture."""

    cache = (
        rtpose_root
        / f"Data/sequences/{sequence}/radar/npy_raw/"
        f"frame{int(radar_frame_id):04d}_adc.npz"
    )
    if cache.exists() or cache.is_symlink():
        if cache.is_symlink() or not cache.is_file():
            raise ValueError(f"ADC cache path is not a regular file: {cache}")
        try:
            adc, metadata = load_adc_cube(cache)
            adc, metadata = normalize_adc_rx_order(adc, metadata)
            validate_adc_cache_identity(
                metadata,
                expected_sequence=sequence,
                expected_radar_frame_id=radar_frame_id,
                path=cache,
            )
        except (KeyError, OSError, ValueError) as exc:
            raise ValueError(f"Existing ADC cache is invalid: {cache}: {exc}") from exc
        return adc, metadata, cache

    adc, metadata = build_measured_adc_cube(
        rtpose_root,
        sequence,
        radar_frame_id,
    )
    adc = validate_adc_cube(adc, label="decoded measured ADC")
    adc, metadata = normalize_adc_rx_order(adc, metadata)
    source = rtpose_root / f"Data/sequences/{sequence}/radar/bin"
    validate_adc_cache_identity(
        metadata,
        expected_sequence=sequence,
        expected_radar_frame_id=radar_frame_id,
        path=source,
    )
    return adc, metadata, source


def normalize_adc_rx_order(
    adc_cube: np.ndarray,
    metadata: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return ADC with its receiver axis ordered by physical RX label.

    RT-Pose's calibrated MIMO-processing cube uses physical receiver labels
    ``13-16, 1-4, 9-12, 5-8``. The portable bundle contract instead stores
    RX1 through RXN within every TX-major block. Reordering here keeps that
    dataset-specific convention out of bundle loading and validation.
    """

    cube = validate_adc_cube(adc_cube, label="measured ADC")
    rx_count = int(cube.shape[2])
    if not isinstance(metadata, dict):
        raise ValueError("ADC metadata must be an object")

    order_value = metadata.get("rx_order")
    if order_value is None:
        order_value = metadata.get("exported_rx_order")
    layout = metadata.get("layout")
    if (
        order_value is None
        and rx_count == len(RTPOSE_MIMO_RX_ORDER)
        and isinstance(layout, (list, tuple))
        and "rx_mimo_order" in layout
    ):
        # Older caches named the axis unambiguously but did not serialize the
        # physical labels separately.
        order_value = RTPOSE_MIMO_RX_ORDER
    if order_value is None:
        raise ValueError(
            "ADC metadata must declare rx_order using physical 1-based labels; "
            "receiver ordering cannot be inferred safely"
        )
    if not isinstance(order_value, (list, tuple)) or len(order_value) != rx_count:
        raise ValueError(f"ADC metadata rx_order must contain {rx_count} entries")
    current_order = tuple(
        strict_metadata_integer(value, label="ADC metadata rx_order entry")
        for value in order_value
    )
    canonical_order = tuple(range(1, rx_count + 1))
    if sorted(current_order) != list(canonical_order):
        raise ValueError(
            "ADC metadata rx_order must be a permutation of physical receiver "
            f"labels 1..{rx_count}"
        )

    source_value = metadata.get("source_rx_order", current_order)
    if not isinstance(source_value, (list, tuple)) or len(source_value) != rx_count:
        raise ValueError(
            f"ADC metadata source_rx_order must contain {rx_count} entries"
        )
    source_order = tuple(
        strict_metadata_integer(value, label="ADC metadata source_rx_order entry")
        for value in source_value
    )
    if sorted(source_order) != list(canonical_order):
        raise ValueError(
            "ADC metadata source_rx_order must be a permutation of physical "
            f"receiver labels 1..{rx_count}"
        )

    indices = np.asarray(
        [current_order.index(label) for label in canonical_order],
        dtype=np.int64,
    )
    normalized = np.asarray(cube[:, :, indices, :], dtype=np.complex64)
    normalized_metadata = dict(metadata)
    normalized_metadata["source_rx_order"] = list(source_order)
    normalized_metadata["exported_rx_order"] = list(canonical_order)
    normalized_metadata["rx_order"] = list(canonical_order)
    normalized_metadata["layout"] = [
        "adc_sample",
        "chirp_loop",
        "rx_physical_label_order",
        "tx_slot",
    ]
    normalized_metadata["rx_order_normalization"] = (
        "source receiver slots reordered to ascending physical RX labels"
    )
    return normalized, normalized_metadata


def load_adc_cube(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    data = load_array_archive(path)
    if "adc_cube" not in data.files:
        raise ValueError(f"{path} missing adc_cube")
    adc_cube = validate_adc_cube(data["adc_cube"], label=str(path))

    metadata: dict[str, Any] = {}
    if "metadata_json" in data.files:
        raw_metadata = np.asarray(data["metadata_json"])
        if raw_metadata.ndim != 0:
            raise ValueError(f"{path} metadata_json must be a scalar JSON string")
        metadata = json.loads(str(npz_scalar_to_py(raw_metadata)))
        if not isinstance(metadata, dict):
            raise ValueError(f"{path} metadata_json root must be an object")
    return adc_cube, metadata


def validate_adc_cube(value: Any, *, label: str) -> np.ndarray:
    adc_cube = np.asarray(value)
    if adc_cube.ndim != 4 or any(size <= 0 for size in adc_cube.shape):
        raise ValueError(
            f"{label} must have non-empty [samples, loops, rx, tx] axes"
        )
    if not np.issubdtype(adc_cube.dtype, np.number) or not np.all(np.isfinite(adc_cube)):
        raise ValueError(f"{label} must contain finite numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        converted_adc = np.asarray(adc_cube, dtype=np.complex64)
    if not np.all(np.isfinite(converted_adc)):
        raise ValueError(f"{label} values must be representable as complex64")
    return converted_adc


def validate_adc_cache_identity(
    metadata: dict[str, Any],
    *,
    expected_sequence: str,
    expected_radar_frame_id: str,
    path: Path,
) -> None:
    """Reject a cache whose available provenance identifies another frame."""

    layers = [metadata]
    for key in ("metadata", "benchmark"):
        nested = metadata.get(key)
        if isinstance(nested, dict):
            layers.append(nested)

    sequence_keys = (
        "sequence",
        "sequence_id",
        "source_sequence_id",
        "rtpose_sequence_id",
    )
    radar_keys = (
        "radar_frame_id",
        "source_radar_frame_id",
        "Radar_frameID",
        "frame_id",
        "frame",
    )
    for layer in layers:
        for key in sequence_keys:
            if key not in layer or layer[key] is None:
                continue
            try:
                observed = sequence_id(layer[key])
            except ValueError as exc:
                raise ValueError(f"{path} has invalid {key} provenance") from exc
            if observed != expected_sequence:
                raise ValueError(
                    f"{path} identifies RT-Pose sequence {observed}, "
                    f"not requested sequence {expected_sequence}"
                )
        for key in radar_keys:
            if key not in layer or layer[key] is None:
                continue
            try:
                observed = frame_id(layer[key])
            except ValueError as exc:
                raise ValueError(f"{path} has invalid {key} provenance") from exc
            if observed != expected_radar_frame_id:
                raise ValueError(
                    f"{path} identifies radar frame {observed}, "
                    f"not requested frame {expected_radar_frame_id}"
                )


def radar_acquisition_duration_s(
    adc_cube: np.ndarray,
    metadata: dict[str, Any],
) -> float:
    """Return the elapsed time from the first through final physical TX slot."""

    cube = np.asarray(adc_cube)
    if cube.ndim != 4 or any(size <= 0 for size in cube.shape):
        raise ValueError("adc_cube must have non-empty [samples, loops, rx, tx] axes")
    params = metadata.get("params", {})
    if not isinstance(params, dict):
        raise ValueError("ADC metadata params must be an object")
    ramp_end = float(params.get("chirp_ramp_end_time", 60.0e-6))
    idle = float(params.get("chirp_idle_time", 5.0e-6))
    if not np.isfinite(ramp_end) or ramp_end <= 0.0:
        raise ValueError("ADC chirp_ramp_end_time must be positive and finite")
    if not np.isfinite(idle) or idle < 0.0:
        raise ValueError("ADC chirp_idle_time must be finite and non-negative")
    expected_sizes = {
        "num_adc_samples": int(cube.shape[0]),
        "num_loops": int(cube.shape[1]),
        "num_rx": int(cube.shape[2]),
        "num_tx": int(cube.shape[3]),
    }
    for key, expected in expected_sizes.items():
        if key not in params:
            continue
        observed = strict_metadata_integer(params[key], label=f"ADC metadata {key}")
        if observed != expected:
            raise ValueError(
                f"ADC metadata {key}={observed} does not match adc_cube axis size {expected}"
            )
    if "tx_to_enable" in params:
        schedule = params["tx_to_enable"]
        if not isinstance(schedule, (list, tuple)) or len(schedule) != cube.shape[3]:
            raise ValueError("ADC metadata tx_to_enable must list one entry per TX axis")
        normalized_schedule = [
            strict_metadata_integer(value, label="ADC metadata tx_to_enable entry")
            for value in schedule
        ]
        if len(set(normalized_schedule)) != len(normalized_schedule):
            raise ValueError("ADC metadata tx_to_enable entries must be unique")
    loop_count = int(cube.shape[1])
    tx_count = int(cube.shape[3])
    return float(max(loop_count * tx_count - 1, 0) * (ramp_end + idle))


def strict_metadata_integer(value: Any, *, label: str) -> int:
    """Parse one JSON scalar without silently truncating fractional values."""

    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{label} must be an integer")
    return int(numeric)


def export_adc(adc_cube: np.ndarray) -> np.ndarray:
    if adc_cube.ndim != 4:
        raise ValueError("adc_cube must have shape [samples, loops, rx, tx]")
    samples, loops, rx_count, tx_count = adc_cube.shape
    return adc_cube.transpose(1, 0, 3, 2).reshape(
        1,
        loops,
        samples,
        tx_count * rx_count,
    )


def build_center_mesh(
    *,
    smpl_data: ArrayArchive,
    amass_npz: Path,
    smpl_model_dir: Path,
    center_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    from mmWaveRadar.targets.smpl import amass_to_mesh_npz

    with tempfile.TemporaryDirectory(prefix="rtpose_bundle_") as tmp:
        tmpdir = Path(tmp)
        subset = tmpdir / "center_smpl.npz"
        np.savez(
            subset,
            poses=np.asarray(smpl_data["poses"][center_index : center_index + 1], dtype=np.float32),
            trans=np.asarray(smpl_data["trans"][center_index : center_index + 1], dtype=np.float32),
            betas=np.asarray(smpl_data["betas"], dtype=np.float32),
            gender=smpl_data["gender"],
            mocap_framerate=np.asarray(smpl_data["mocap_framerate"])
            if "mocap_framerate" in smpl_data.files
            else np.asarray(15.0, dtype=np.float32),
            source_amass_npz=str(amass_npz),
        )
        faces, vertices, _ = amass_to_mesh_npz(
            str(subset),
            str(smpl_model_dir),
            out_npz_path=None,
        )

    from mmWaveRadar.targets.mesh import ensure_outward_face_winding

    vertices = (
        np.asarray(vertices, dtype=np.float32)[..., AMASS_TO_RTPOSE_AXES]
        * AMASS_TO_RTPOSE_SIGNS
    )
    faces = ensure_outward_face_winding(
        vertices,
        np.asarray(faces, dtype=np.uint32),
    )
    return vertices, faces


def write_amass_sequence(
    path: Path,
    *,
    smpl_data: ArrayArchive,
    source_amass_npz_label: str,
    indices: np.ndarray,
    center_index: int,
    bundle_times: np.ndarray,
    segment_times: np.ndarray,
    source_capture_times: np.ndarray,
    pose_frame_ids: np.ndarray,
    radar_frame_ids: np.ndarray,
    faces: np.ndarray,
) -> None:
    """Write a canonical AMASS-like window copied from the selected segment."""

    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("AMASS output indices must be a non-empty vector")
    center_hits = np.flatnonzero(indices == int(center_index))
    if len(center_hits) != 1:
        raise ValueError("AMASS output window must contain the selected center frame once")
    total = int(np.asarray(smpl_data["poses"]).shape[0])
    for name, values in (
        ("segment_times", segment_times),
        ("source_capture_times", source_capture_times),
        ("pose_frame_ids", pose_frame_ids),
        ("radar_frame_ids", radar_frame_ids),
    ):
        if np.asarray(values).shape != (total,):
            raise ValueError(f"{name} must have shape ({total},)")

    if "mocap_framerate" in smpl_data:
        mocap_framerate = np.asarray(smpl_data["mocap_framerate"])
    elif len(segment_times) > 1:
        mocap_framerate = np.asarray(
            1.0 / float(np.median(np.diff(segment_times))),
            dtype=np.float64,
        )
    else:
        raise ValueError("A single-frame AMASS window requires mocap_framerate")
    rate = float(np.asarray(mocap_framerate, dtype=np.float64).item())
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("AMASS mocap_framerate must be positive and finite")

    arrays: dict[str, np.ndarray] = {
        "poses": np.asarray(smpl_data["poses"][indices], dtype=np.float32),
        "trans": np.asarray(smpl_data["trans"][indices], dtype=np.float32),
        "betas": np.asarray(smpl_data["betas"], dtype=np.float32),
        "gender": np.asarray(smpl_data["gender"]),
        "mocap_framerate": np.asarray(mocap_framerate),
        "times": np.asarray(segment_times[indices], dtype=np.float64),
        "mocap_time_length": np.asarray(indices.size / rate, dtype=np.float64),
        "world_up_axis": np.asarray(smpl_data["world_up_axis"]),
        "world_forward_axis": np.asarray(smpl_data["world_forward_axis"]),
        "frame_ids": np.asarray(pose_frame_ids[indices]).astype(str),
        "source_capture_times": np.asarray(
            source_capture_times[indices], dtype=np.float64
        ),
        "bundle_times": np.asarray(bundle_times, dtype=np.float64),
        "faces": np.asarray(faces, dtype=np.uint32),
        "model_type": np.asarray("smpl"),
        "radar_frame_ids": np.asarray(radar_frame_ids[indices]).astype(str),
        "radar_capture_times": np.asarray(
            [int(frame_id(value)) * 0.1 for value in radar_frame_ids[indices]],
            dtype=np.float64,
        ),
        "source_smpl_indices": indices,
        "selected_window_index": np.asarray(int(center_hits[0]), dtype=np.int64),
        "selected_source_smpl_index": np.asarray(int(center_index), dtype=np.int64),
        "selected_frame_id": np.asarray(pose_frame_ids[center_index]).astype(str),
        "selected_radar_frame_id": np.asarray(
            radar_frame_ids[center_index]
        ).astype(str),
        "source_amass_npz": np.asarray(source_amass_npz_label),
        "output_axes": AMASS_TO_RTPOSE_AXES,
        "output_signs": AMASS_TO_RTPOSE_SIGNS,
        "coordinate_frame": np.asarray("canonical AMASS; RT-Pose transform metadata"),
        "face_winding_note": np.asarray(
            "corrected_with_smpl_model_faces_to_rtpose_frame"
        ),
        "timing_reference": np.asarray(
            "times=source segment-local; source_capture_times=source absolute; "
            "bundle_times=selected radar frame at t=0"
        ),
    }

    # Preserve standard/provenance fields supplied by SparseSMPLFit without
    # copying fitting diagnostics or changing the canonical AMASS pose values.
    passthrough_fields = (
        "dmpls",
        "output_format",
        "surface_model_type",
        "sparse_smpl_fit_version",
        "profile",
        "source_sequence_id",
        "subject_id",
        "smpl_root_joint_offset",
    )
    for name in passthrough_fields:
        if name not in smpl_data or name in arrays:
            continue
        value = np.asarray(smpl_data[name])
        if name == "dmpls" and value.ndim > 0 and value.shape[0] == total:
            value = value[indices]
        arrays[name] = np.asarray(value)
    np.savez_compressed(path, **arrays)


def frame_mapping_metadata(
    frame_mapping: list[dict[str, Any]],
    *,
    source_times: np.ndarray,
    center_index: int,
    real_radar_frame_id: str,
) -> dict[str, Any]:
    record = frame_mapping[center_index]
    source_radar_frame_id = frame_id(record["radar_frame_id"])
    source_frame_id = frame_id(record["pose_frame_id"])
    pose_time_s = float(source_times[center_index])
    real_radar_time_s = int(real_radar_frame_id) * 0.1
    source_radar_time_s = int(source_radar_frame_id) * 0.1
    return {
        "source": "RT-Pose Train.json pose-frame to Radar_frameID mapping",
        "real_radar_frame_id": real_radar_frame_id,
        "real_radar_time_s": real_radar_time_s,
        "source_radar_frame_id": source_radar_frame_id,
        "source_radar_time_s": source_radar_time_s,
        "source_frame_id": source_frame_id,
        "source_smpl_index": int(center_index),
        "source_smpl_time_s": pose_time_s,
        "pose_to_real_radar_time_offset_s": real_radar_time_s - pose_time_s,
        "source_lidar_valid": int(record.get("lidar_valid", 0)),
        "mapping_override": source_radar_frame_id != real_radar_frame_id,
    }


def mesh_range_stats(vertices: np.ndarray) -> dict[str, Any]:
    ranges = np.linalg.norm(np.asarray(vertices, dtype=np.float64), axis=1)
    center = np.mean(vertices, axis=0)
    return {
        "mesh_center_xyz_m": [float(v) for v in center],
        "mesh_range_min_m": float(np.min(ranges)),
        "mesh_range_median_m": float(np.median(ranges)),
        "mesh_range_max_m": float(np.max(ranges)),
    }


def copy_sequence_environment_geometry(
    *,
    rtpose_root: Path,
    sequence: str,
    bundle_dir: Path,
) -> dict[str, Any]:
    """Copies sequence-level fitted environment geometry into a frame bundle."""

    source_dir = find_sequence_environment_geometry_dir(
        rtpose_root=rtpose_root,
        sequence=sequence,
    )
    if source_dir is None:
        return {}
    return copy_environment_geometry_directory(source_dir, bundle_dir)


def copy_provided_environment_geometry(
    source: Path,
    bundle_dir: Path,
) -> dict[str, Any]:
    """Stage a reusable environment folder, scene XML, or mesh NPZ."""

    source = source.expanduser()
    if source.is_symlink() or not source.exists():
        raise FileNotFoundError(f"Provided environment geometry was not found: {source}")
    source = source.resolve()
    if source.is_dir():
        copied = copy_environment_geometry_directory(source, bundle_dir)
    elif source.is_file() and source.suffix.lower() == ".xml":
        copied = copy_environment_geometry_directory(source.parent, bundle_dir)
        destination = bundle_dir / "environment_geometry/environment_scene.xml"
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy_portable_scene_assets(source, destination)
        copied["scene_path"] = "environment_geometry/environment_scene.xml"
    elif source.is_file() and source.suffix.lower() == ".npz":
        destination = (
            bundle_dir / "environment_geometry/environment_mesh_sequence.npz"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        repack_environment_mesh_sequence(source, destination)
        copied = {
            "environment_mesh_sequence": (
                "environment_geometry/environment_mesh_sequence.npz"
            )
        }
    else:
        raise ValueError(
            "--environment-geometry must name a directory, scene XML, or mesh NPZ"
        )
    if not copied:
        raise ValueError(
            f"Provided geometry contains no supported environment assets: {source}"
        )
    copied["geometry_source"] = "provided"
    return copied


def copy_environment_geometry_directory(
    source_dir: Path,
    bundle_dir: Path,
) -> dict[str, Any]:
    """Copy the portable, supported subset of an environment directory."""

    if source_dir.is_symlink() or not source_dir.is_dir():
        raise ValueError(
            f"Environment geometry source must be a regular directory: {source_dir}"
        )
    dest_dir = bundle_dir / "environment_geometry"
    copied: dict[str, Any] = {}
    for filename in (
        "environment_fit_preview.png",
        "environment_scene.xml",
        "environment_surfaces.obj",
    ):
        source = source_dir / filename
        if not source.exists():
            continue
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Environment asset is not a safe regular file: {source}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        destination = dest_dir / filename
        if source.suffix.lower() in {".obj", ".xml"}:
            copy_portable_scene_assets(source, destination)
        else:
            shutil.copy2(source, destination)

    for json_name in ("environment_surfaces.json", "reconstruction_stats.json"):
        json_source = source_dir / json_name
        if not json_source.exists():
            continue
        if json_source.is_symlink() or not json_source.is_file():
            raise ValueError(
                f"Environment asset is not a safe regular file: {json_source}"
            )
        try:
            payload = json.loads(json_source.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid environment JSON: {json_source}") from exc
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / json_source.name).write_text(
            json.dumps(sanitize_environment_json(payload), indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

    mesh_sequence = source_dir / "environment_mesh_sequence.npz"
    if mesh_sequence.exists():
        if mesh_sequence.is_symlink() or not mesh_sequence.is_file():
            raise ValueError(
                f"Environment asset is not a safe regular file: {mesh_sequence}"
            )
        dest_dir.mkdir(parents=True, exist_ok=True)
        repack_environment_mesh_sequence(
            mesh_sequence,
            dest_dir / mesh_sequence.name,
        )

    if (dest_dir / "environment_surfaces.json").exists():
        copied["environment_surfaces_json"] = "environment_geometry/environment_surfaces.json"
    if (dest_dir / "environment_surfaces.obj").exists():
        copied["environment_obj"] = "environment_geometry/environment_surfaces.obj"
    if (dest_dir / "environment_scene.xml").exists():
        copied["scene_path"] = "environment_geometry/environment_scene.xml"
    if (dest_dir / "environment_mesh_sequence.npz").exists():
        copied["environment_mesh_sequence"] = (
            "environment_geometry/environment_mesh_sequence.npz"
        )
    if (dest_dir / "reconstruction_stats.json").exists():
        copied["reconstruction_stats"] = (
            "environment_geometry/reconstruction_stats.json"
        )
    return copied


def sanitize_environment_json(value: Any) -> Any:
    """Remove machine-local absolute prefixes from informational JSON fields."""

    if isinstance(value, dict):
        return {str(key): sanitize_environment_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_environment_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_environment_json(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        file_path = text[7:] if text.lower().startswith("file://") else text
        if Path(file_path).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", file_path):
            basename = re.split(r"[\\/]", file_path.rstrip("\\/"))[-1]
            if not basename:
                raise ValueError("Environment JSON contains an unusable absolute path")
            return basename
    return value


def repack_environment_mesh_sequence(source: Path, destination: Path) -> None:
    """Copy only the validated numeric mesh-sequence contract from an NPZ."""

    try:
        archive = np.load(source, allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError("expected an NPZ archive")
        with archive:
            missing = [
                key for key in ("vertices", "faces", "times")
                if key not in archive.files
            ]
            if missing:
                raise ValueError(
                    f"Environment mesh sequence is missing fields: {missing}"
                )
            vertices = np.array(archive["vertices"], copy=True)
            faces = np.array(archive["faces"], copy=True)
            times = np.array(archive["times"], copy=True)
    except (OSError, ValueError) as exc:
        raise ValueError(
            "Environment vertices, faces, and times must be pickle-free numeric "
            f"NPZ members: {source}"
        ) from exc
    if vertices.ndim != 3 or vertices.shape[0] == 0 or vertices.shape[2] != 3:
        raise ValueError("Environment vertices must have shape (T, V, 3) with T > 0")
    if vertices.dtype.kind not in {"i", "u", "f"} or not np.all(np.isfinite(vertices)):
        raise ValueError("Environment vertices must contain finite real values")
    with np.errstate(over="ignore", invalid="ignore"):
        vertices_float32 = np.asarray(vertices, dtype=np.float32)
    if not np.all(np.isfinite(vertices_float32)):
        raise ValueError("Environment vertices must be representable as float32")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in {"i", "u"}:
        raise ValueError("Environment faces must have an integer shape (F, 3)")
    if faces.size and (np.min(faces) < 0 or np.max(faces) >= vertices.shape[1]):
        raise ValueError("Environment faces contain out-of-range vertex indices")
    if times.shape != (vertices.shape[0],) or times.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("Environment times must have shape (T,) and be real numeric")
    times = np.asarray(times, dtype=np.float64)
    if not np.all(np.isfinite(times)) or (
        len(times) > 1 and np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("Environment times must be finite and strictly increasing")
    np.savez_compressed(
        destination,
        vertices=vertices_float32,
        faces=np.asarray(faces, dtype=np.uint32),
        times=times,
    )


def sanitize_staged_environment_geometry(bundle_dir: Path) -> None:
    """Validate optional generated or copied environment artifacts in-place."""

    environment_dir = bundle_dir / "environment_geometry"
    if not environment_dir.exists():
        return
    if environment_dir.is_symlink() or not environment_dir.is_dir():
        raise ValueError("Environment geometry output must be a regular directory")

    for json_name in ("environment_surfaces.json", "reconstruction_stats.json"):
        json_path = environment_dir / json_name
        if not json_path.exists():
            continue
        if json_path.is_symlink() or not json_path.is_file():
            raise ValueError(f"Environment JSON is not a safe regular file: {json_path}")
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid environment JSON: {json_path}") from exc
        json_path.write_text(
            json.dumps(sanitize_environment_json(payload), indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

    mesh_sequence = environment_dir / "environment_mesh_sequence.npz"
    if mesh_sequence.exists():
        if mesh_sequence.is_symlink() or not mesh_sequence.is_file():
            raise ValueError(
                f"Environment mesh sequence is not a safe regular file: {mesh_sequence}"
            )
        repack_environment_mesh_sequence(mesh_sequence, mesh_sequence)

    surfaces_obj = environment_dir / "environment_surfaces.obj"
    if surfaces_obj.exists():
        if surfaces_obj.is_symlink() or not surfaces_obj.is_file():
            raise ValueError(f"Environment OBJ is not a safe regular file: {surfaces_obj}")
        copy_portable_scene_assets(surfaces_obj, surfaces_obj)


def find_sequence_environment_geometry_dir(
    *,
    rtpose_root: Path,
    sequence: str,
) -> Path | None:
    """Find legacy fitted sequence-level environment geometry, if present."""

    candidates = (
        rtpose_root / f"Data/sequences/{sequence}/map/environment_geometry",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def environment_geometry_bundle_contents(bundle_dir: Path) -> dict[str, str]:
    """Returns benchmark bundle-content paths for copied environment files."""

    env_dir = bundle_dir / "environment_geometry"
    contents: dict[str, str] = {}
    for filename in (
        "environment_fit_preview.png",
        "environment_mesh_sequence.npz",
        "environment_scene.xml",
        "environment_surfaces.json",
        "environment_surfaces.obj",
        "reconstruction_stats.json",
    ):
        path = env_dir / filename
        if path.exists():
            contents[filename] = f"environment_geometry/{filename}"
    if (env_dir / "environment_scene.xml").exists():
        contents["scene_path"] = "environment_geometry/environment_scene.xml"
    return contents


def stage_scene_path(
    *,
    geometry_outputs: dict[str, Any],
    bundle_dir: Path,
) -> Path | None:
    """Validate the automatically prepared scene and its local references."""

    raw_scene = geometry_outputs.get("scene_path")
    if raw_scene is None:
        return None
    scene = Path(str(raw_scene)).expanduser()
    if not scene.is_absolute():
        scene = bundle_dir / scene
    if not scene.is_file() or scene.is_symlink():
        raise FileNotFoundError(f"Prepared scene file was not found or is unsafe: {scene}")
    scene = scene.resolve()
    if scene.suffix.lower() != ".xml":
        raise ValueError("Prepared scene must be an XML file")
    try:
        scene.relative_to(bundle_dir.resolve())
    except ValueError as exc:
        raise ValueError("Prepared scene must be inside the staged bundle") from exc
    copy_portable_scene_assets(scene, scene)
    geometry_outputs["scene_path"] = str(scene)
    return scene


def copy_portable_scene_assets(source: Path, destination: Path) -> None:
    """Copy an XML scene and its local file references into one portable tree."""

    source = source.resolve()
    source_root = source.parent
    destination = destination.resolve()
    destination_root = destination.parent
    visited: set[Path] = set()

    def copy_asset(asset: Path, relative: Path) -> None:
        resolved = asset.resolve()
        try:
            resolved.relative_to(source_root)
        except ValueError as exc:
            raise ValueError(f"Scene asset escapes its source directory: {asset}") from exc
        if resolved in visited:
            return
        if asset.is_symlink() or not resolved.is_file():
            raise FileNotFoundError(f"Scene asset was not found or is unsafe: {asset}")
        visited.add(resolved)
        target = destination_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if resolved != target.resolve():
            shutil.copy2(resolved, target)

        suffix = resolved.suffix.lower()
        if suffix == ".xml":
            for reference in scene_xml_references(resolved):
                ref_path = safe_scene_reference(reference)
                copy_asset(resolved.parent / ref_path, relative.parent / ref_path)
        elif suffix == ".obj":
            for reference in obj_material_references(resolved):
                ref_path = safe_scene_reference(reference)
                copy_asset(resolved.parent / ref_path, relative.parent / ref_path)
        elif suffix == ".mtl":
            for reference in material_texture_references(resolved):
                ref_path = safe_scene_reference(reference)
                copy_asset(resolved.parent / ref_path, relative.parent / ref_path)

    copy_asset(source, Path(destination.name))


def safe_scene_reference(value: str) -> Path:
    """Return a normalized, bundle-relative scene asset path."""

    text = value.strip()
    if not text or "\\" in text or Path(text).is_absolute() or re.match(
        r"^[A-Za-z][A-Za-z0-9+.-]*:", text
    ):
        raise ValueError(f"Scene contains a non-portable asset reference: {value!r}")
    path = Path(text)
    if any(
        part in {"", ".", ".."}
        or part.endswith(".")
        or part.split(".", maxsplit=1)[0].upper() in WINDOWS_RESERVED_NAMES
        for part in path.parts
    ):
        raise ValueError(f"Scene contains an unsafe asset reference: {value!r}")
    return path


def scene_xml_references(path: Path) -> list[str]:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, UnicodeDecodeError) as exc:
        raise ValueError(f"Scene must be valid XML: {path}") from exc
    references: list[str] = []
    for element in root.iter():
        for key in ("filename", "path", "src", "href"):
            if key in element.attrib:
                references.append(element.attrib[key])
        name = element.attrib.get("name", "").strip().lower()
        if name in {"filename", "path"} and "value" in element.attrib:
            references.append(element.attrib["value"])
    return references


def obj_material_references(path: Path) -> list[str]:
    references: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"OBJ scene asset must be UTF-8 text: {path}") from exc
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("mtllib "):
            references.append(stripped.split(maxsplit=1)[1])
    return references


def material_texture_references(path: Path) -> list[str]:
    references: list[str] = []
    texture_commands = {
        "bump",
        "decal",
        "disp",
        "map_bump",
        "map_d",
        "map_ka",
        "map_kd",
        "map_ke",
        "map_ks",
        "map_ns",
        "refl",
    }
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"MTL scene asset must be UTF-8 text: {path}") from exc
    for line in lines:
        try:
            fields = shlex.split(line, comments=True)
        except ValueError as exc:
            raise ValueError(f"Invalid MTL line in {path}: {line!r}") from exc
        if fields and fields[0].lower() in texture_commands and len(fields) >= 2:
            references.append(fields[-1])
    return references


def build_benchmark_metadata(
    *,
    sequence: str,
    radar_frame_id: str,
    frame_mapping: dict[str, Any],
    bundle_name: str,
    environment_geometry: dict[str, str],
    camera_images: dict[str, str],
    sequence_metadata: dict[str, str],
) -> dict[str, Any]:
    """Builds the benchmark metadata sidecar used by prepared RT-Pose clips."""

    benchmark_id = f"rtpose_seq{sequence}_{bundle_name}"
    scenario = sequence_metadata["scenario"].strip()
    scenario_key = scenario.casefold()
    subject_type = sequence_metadata["subject_type"].strip()
    subject_key = re.sub(r"[^a-z]+", "", subject_type.casefold())
    single_subject = subject_key == "singleperson"
    subject_count = "single" if single_subject else "multiple"
    note = (
        f"{sequence_metadata['activity']}; sequence {sequence} clip at radar "
        f"frame {radar_frame_id}."
    )

    bundle_contents: dict[str, Any] = {
        "adc": "radar_adc.npz",
        "camera_images": camera_images,
    }
    if environment_geometry:
        bundle_contents["environment_geometry"] = environment_geometry
    bundle_contents.update(
        {
            "descriptor": "bundle.json",
            "frames": "frames.json",
            "sensor": "sensor.json",
            "amass_motion_sequence": "amass_sequence.npz",
        }
    )

    return {
        "action": sequence_metadata["activity"],
        "bundle_contents": bundle_contents,
        "camera_frame_id": frame_mapping["source_frame_id"],
        "camera_privacy": {
            "faces_obfuscated": False,
            "method": "none",
            "original_images_included": True,
        },
        "dataset": "RT-Pose",
        "environment": scenario_key,
        "environment_is_clean": scenario_key == "clean",
        "environment_is_cluttered": scenario_key == "cluttered",
        "evaluation_notes": {
            "po_human_only": "Use amass_sequence.npz and an empty scene.",
            "rt_with_environment": (
                "Use benchmark_metadata.bundle_contents.environment_geometry.scene_path "
                "when present; otherwise the bundle is self-contained for human-only "
                "simulation and measured-ADC DSP validation."
            ),
        },
        "motion": {
            "file": "amass_sequence.npz",
            "format": "canonical AMASS-like base-SMPL window",
            "mesh_sequence_included": False,
            "mesh_generation": (
                "generated from the AMASS parameters and a licensed SMPL model "
                "when simulation is run"
            ),
            "parameter_source": (
                "poses, translations, betas, gender, and producer fields copied "
                "from the selected fitted segment without refitting"
            ),
            "timing_fields": AMASS_TIMING_FIELDS,
        },
        "id": benchmark_id,
        "note": note,
        "location": sequence_metadata["location"],
        "occlusion": sequence_metadata["occlusion"],
        "pose_frame_id": frame_mapping["source_frame_id"],
        "radar_frame_id": radar_frame_id,
        "rtpose_sequence_id": sequence,
        "scenario": scenario,
        "single_subject": single_subject,
        "source_subject_type": subject_type,
        "subject_count": subject_count,
    }


def benchmark_frame_metadata(
    benchmark_metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if benchmark_metadata is None:
        return None
    return {
        "action": benchmark_metadata["action"],
        "camera_frame_id": benchmark_metadata["camera_frame_id"],
        "dataset": benchmark_metadata["dataset"],
        "environment": benchmark_metadata["environment"],
        "id": benchmark_metadata["id"],
        "location": benchmark_metadata["location"],
        "occlusion": benchmark_metadata["occlusion"],
        "pose_frame_id": benchmark_metadata["pose_frame_id"],
        "radar_frame_id": benchmark_metadata["radar_frame_id"],
        "rtpose_sequence_id": benchmark_metadata["rtpose_sequence_id"],
        "scenario": benchmark_metadata["scenario"],
        "single_subject": benchmark_metadata["single_subject"],
        "source_subject_type": benchmark_metadata["source_subject_type"],
        "subject_count": benchmark_metadata["subject_count"],
    }


def write_frames_json(
    path: Path,
    *,
    radar_frame_id: str,
    frame_mapping: dict[str, Any],
    mesh_stats: dict[str, Any],
    benchmark_frame_metadata: dict[str, Any],
) -> None:
    metadata = {
        "benchmark": benchmark_frame_metadata,
        "mapping_override": frame_mapping["mapping_override"],
        "mesh_center_xyz_m": mesh_stats["mesh_center_xyz_m"],
        "mesh_range_max_m": mesh_stats["mesh_range_max_m"],
        "mesh_range_median_m": mesh_stats["mesh_range_median_m"],
        "mesh_range_min_m": mesh_stats["mesh_range_min_m"],
        "purpose": "rtpose_validation_bundle",
        "real_radar_frame_id": frame_mapping["real_radar_frame_id"],
        "real_radar_time_s": frame_mapping["real_radar_time_s"],
        "selection_note": "",
        "source": frame_mapping["source"],
        "source_frame_id": frame_mapping["source_frame_id"],
        "source_lidar_valid": frame_mapping["source_lidar_valid"],
        "source_radar_frame_id": frame_mapping["source_radar_frame_id"],
        "source_radar_time_s": frame_mapping["source_radar_time_s"],
        "source_smpl_index": frame_mapping["source_smpl_index"],
        "source_smpl_time_s": frame_mapping["source_smpl_time_s"],
        "pose_to_real_radar_time_offset_s": frame_mapping[
            "pose_to_real_radar_time_offset_s"
        ],
    }
    frames = {
        "frames": [
            {
                "benchmark_index": 0,
                "motion_time_s": 0.0,
                "radar_frame_id": radar_frame_id,
                "camera_frame_id": frame_mapping["source_frame_id"],
                "pose_frame_id": frame_mapping["source_frame_id"],
                "dataset_frame_id": radar_frame_id,
                "metadata": metadata,
            }
        ]
    }
    path.write_text(
        json.dumps(frames, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_sensor_json(
    path: Path,
    *,
    sequence: str,
    rtpose_root: Path,
    amass_npz: Path,
    adc_source: Path,
    adc_cube: np.ndarray,
    exported_adc: np.ndarray,
    adc_metadata: dict[str, Any],
    scene_path: Path | None,
    frame_mapping: dict[str, Any],
    motion_times: np.ndarray,
    motion_timing: str,
    geometry_outputs: dict[str, Any],
    benchmark_metadata: dict[str, Any],
) -> dict[str, Any]:
    params = adc_metadata.get("params", {})
    start_freq = float(params.get("start_freq", 77.0e9))
    slope = float(params.get("chirp_slope", 64.985e12))
    adc_start = float(params.get("adc_start_time", 5.0e-6))
    sample_rate = float(params.get("adc_sample_rate", 5.0e6))
    samples = int(params.get("num_adc_samples", adc_cube.shape[0]))
    loops = int(params.get("num_loops", adc_cube.shape[1]))
    tx_count = int(params.get("num_tx", adc_cube.shape[3]))
    ramp_end = float(params.get("chirp_ramp_end_time", 60.0e-6))
    idle = float(params.get("chirp_idle_time", 5.0e-6))
    chirp_ramp_time = samples / sample_rate
    carrier = start_freq + (adc_start + chirp_ramp_time / 2.0) * slope
    velocity_resolution = (3.0e8 / carrier) / (
        2.0 * loops * (ramp_end + idle) * tx_count
    )

    metadata = {
        "benchmark": benchmark_metadata,
        "dataset": "RT-Pose",
        "sequence_id": sequence,
        "source_adc": portable_provenance_path(
            adc_source,
            rtpose_root,
            REPO_ROOT,
        ),
        "source_amass_npz": portable_provenance_path(
            amass_npz,
            rtpose_root,
            REPO_ROOT,
        ),
        "source_adc_shape": list(adc_cube.shape),
        "exported_adc_shape": list(exported_adc.shape),
        "source_adc_axes": ["adc_sample", "chirp_loop", "rx", "tx"],
        "exported_adc_axes": [
            "frame",
            "chirp_loop",
            "adc_sample",
            "virtual_channel_tx_major",
        ],
        "source_rx_order": list(adc_metadata["source_rx_order"]),
        "exported_rx_order": list(adc_metadata["exported_rx_order"]),
        "rx_order": list(adc_metadata["rx_order"]),
        "rx_order_note": (
            "Receiver values are physical 1-based AWR2243 labels. Each "
            "TX-major block is exported in ascending RX1..RX16 order."
        ),
        "range_resolution_m": float(
            adc_metadata.get(
                "range_resolution_m",
                3.0e8 / (2.0 * slope * chirp_ramp_time),
            )
        ),
        "velocity_resolution_mps": float(
            adc_metadata.get("velocity_resolution_mps", velocity_resolution)
        ),
        "face_winding": "corrected_with_smpl_model_faces_to_rtpose_frame",
        "tdm_virtual_adc": True,
        "tx_to_enable": list(params.get("tx_to_enable", range(tx_count, 0, -1))),
        "tdm_schedule_note": (
            "RT-Pose AWR2243 cascade physical chirp order; ADC bundle stores "
            "chirp loops with TX slots separated into tx-major virtual channels "
            "and physical receivers ordered RX1 through RX16 within each block."
        ),
        "source_adc_conversion": (
            "prepared RT-Pose ADC cache"
            if adc_source.suffix.lower() == ".npz"
            else "bundled RT-Pose AWR2243 raw decoder"
        ),
        "radar_acquisition_duration_s": radar_acquisition_duration_s(
            adc_cube,
            adc_metadata,
        ),
        "frame_mapping": frame_mapping,
        "motion_source": "amass_sequence.npz",
        "amass_sequence_included": True,
        "mesh_sequence_included": False,
        "neutral_shape": False,
        "geometry_outputs": portable_geometry_outputs(
            geometry_outputs,
            roots=(path.parent, rtpose_root, REPO_ROOT),
        ),
    }
    metadata["amass_motion_timing"] = motion_timing
    metadata["amass_motion_relative_times_s"] = [
        float(value) for value in motion_times
    ]
    if scene_path is not None:
        metadata["scene_path"] = portable_provenance_path(
            scene_path,
            path.parent,
            rtpose_root,
            REPO_ROOT,
        )
    payload = {
        "board_model": "MMWCAS-RF-EVM",
        "pattern_mode": "cosine30",
        "virtual_channel_order": "tx_major",
        "radar": {
            "name": f"rtpose_seq{sequence}_radar",
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0],
        },
        "fmcw": {
            "carrier_frequency_hz": carrier,
            "start_frequency_hz": start_freq,
            "slope_hz_per_s": slope,
            "chirp_duration_s": ramp_end,
            "chirp_repetition_time_s": ramp_end + idle,
            "adc_start_time_s": adc_start,
            "sampling_frequency_hz": sample_rate,
            "num_adc_samples": samples,
            "num_chirps_per_frame": loops,
            "frame_period_s": 0.1,
            "num_tx": tx_count,
            "tdm_enabled": True,
        },
        "metadata": metadata,
    }
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return metadata


def prepare_environment_geometry(
    *,
    rtpose_root: Path,
    sequence: str,
    bundle_dir: Path,
    source_frame_id: str,
    body_vertices: np.ndarray,
    provided_geometry: Path | None,
) -> dict[str, Any]:
    """Reuse geometry or reconstruct and fit it as one preparation stage."""

    if provided_geometry is not None:
        return copy_provided_environment_geometry(
            provided_geometry,
            bundle_dir,
        )

    outputs = copy_sequence_environment_geometry(
        rtpose_root=rtpose_root,
        sequence=sequence,
        bundle_dir=bundle_dir,
    )
    if outputs:
        outputs["geometry_source"] = "dataset_cache"
        return outputs

    background_ply = find_background_lidar_cloud(
        rtpose_root=rtpose_root,
        sequence=sequence,
        source_frame_id=source_frame_id,
    )
    if background_ply is not None:
        outputs = fit_background_environment(background_ply, bundle_dir)
        outputs["geometry_source"] = "prepared_background"
        outputs["source_background_ply"] = background_ply.name
        return outputs

    with tempfile.TemporaryDirectory(
        prefix=".rtpose_environment_",
        dir=bundle_dir.parent,
    ) as temporary_dir:
        reconstruction = reconstruct_background(
            dataset_dir=rtpose_root,
            sequence=sequence,
            source_frame_id=source_frame_id,
            body_vertices=body_vertices,
            output_dir=Path(temporary_dir),
        )
        outputs = fit_background_environment(
            reconstruction.background_ply,
            bundle_dir,
        )
        stats_destination = (
            bundle_dir / "environment_geometry/reconstruction_stats.json"
        )
        shutil.copy2(reconstruction.stats_path, stats_destination)
        outputs.update(
            {
                "geometry_source": "reconstructed_lidar_stereo",
                "reconstruction_stats": (
                    "environment_geometry/reconstruction_stats.json"
                ),
                "source_background_ply": reconstruction.background_ply.name,
            }
        )
        return outputs


def fit_background_environment(
    background_ply: Path,
    bundle_dir: Path,
) -> dict[str, Any]:
    """Run the bundled deterministic fitter on one prepared background cloud."""

    script = Path(__file__).with_name("fit_rtpose_environment_geometry.py")
    if not script.is_file():
        raise FileNotFoundError(f"Bundled environment fitter was not found: {script}")
    output_dir = bundle_dir / "environment_geometry"
    command = [
        sys.executable,
        str(script),
        str(background_ply),
        "--output-dir",
        str(output_dir),
        "--wall-mode",
        "solid",
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise ValueError(f"RT-Pose environment fitting failed{suffix}") from exc
    expected = {
        "environment_surfaces_json": output_dir / "environment_surfaces.json",
        "environment_obj": output_dir / "environment_surfaces.obj",
        "scene_path": output_dir / "environment_scene.xml",
        "environment_mesh_sequence": output_dir / "environment_mesh_sequence.npz",
    }
    missing = [name for name, output in expected.items() if not output.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Environment fitting did not create expected outputs: {missing}"
        )
    return {name: str(output) for name, output in expected.items()}


def find_background_lidar_cloud(
    *,
    rtpose_root: Path,
    sequence: str,
    source_frame_id: str,
) -> Path | None:
    filename = f"{source_frame_id}_background_lidar_colored.ply"
    candidates = (
        rtpose_root
        / f"Data/sequences/{sequence}/map/background_reconstruction/{filename}",
        rtpose_root
        / f"Data/sequences/{sequence}/background_reconstruction/"
        f"frame{source_frame_id}/{filename}",
    )
    for candidate in candidates:
        if candidate.is_symlink():
            raise ValueError(f"Background LiDAR cloud must not be a symlink: {candidate}")
        if candidate.is_file():
            return candidate.resolve()
    return None


def portable_provenance_path(path: Path, *roots: Path) -> str:
    """Return a relative provenance label without leaking an external root."""

    expanded = path.expanduser()
    if not expanded.is_absolute():
        return expanded.as_posix()
    resolved = expanded.resolve()
    for root in roots:
        try:
            return resolved.relative_to(root.expanduser().resolve()).as_posix()
        except ValueError:
            continue
    return path.name


def portable_geometry_outputs(
    outputs: dict[str, Any],
    *,
    roots: tuple[Path, ...],
) -> dict[str, Any]:
    """Return geometry metadata without repository- or dataset-local prefixes."""

    portable: dict[str, Any] = {}
    for key, value in outputs.items():
        if isinstance(value, str) and (
            key.endswith(("_path", "_json", "_obj", "_ply", "_stats"))
            or key in {"scene_path", "environment_mesh_sequence"}
        ):
            portable[key] = portable_provenance_path(Path(value), *roots)
        else:
            portable[key] = value
    return portable


def write_environment_json(
    path: Path,
    *,
    scene_path: Path | None,
    geometry_outputs: dict[str, Any],
    roots: tuple[Path, ...],
) -> Path | None:
    """Writes an optional validation-bundle environment sidecar."""

    if scene_path is None and not geometry_outputs:
        return None

    payload: dict[str, Any] = {
        "metadata": {
            "source": "tools/bundle_prepare/rtpose/prepare_rtpose_bundle.py",
            "format": "mmwave_validation_environment",
        }
    }
    if scene_path is not None:
        payload["scene_path"] = portable_provenance_path(Path(scene_path), *roots)
    normalized_outputs: dict[str, Any] = {}
    for key, value in sorted(geometry_outputs.items()):
        if isinstance(value, (str, Path)):
            value_path = Path(value)
            normalized_value = (
                portable_provenance_path(value_path, *roots)
                if value_path.suffix or value_path.exists()
                else str(value)
            )
        else:
            normalized_value = value
        normalized_outputs[key] = normalized_value
        if key in {
            "environment_surfaces_json",
            "environment_obj",
            "environment_mesh_sequence",
            "reconstruction_stats",
        }:
            payload[key] = normalized_value
    if normalized_outputs:
        payload["geometry_outputs"] = normalized_outputs

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def npz_scalar_to_py(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        if isinstance(item, np.generic):
            return item.item()
        return item
    return arr.tolist()


if __name__ == "__main__":
    raise SystemExit(main())
