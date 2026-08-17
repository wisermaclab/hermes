# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Structural and cross-file integrity checks for prepared HERMES bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import sysconfig
from typing import Any, Mapping
import warnings
import xml.etree.ElementTree as ET

from jsonschema import Draft202012Validator
import numpy as np
from PIL import Image, UnidentifiedImageError

from mmWaveRadar.targets import MeshSequence

from .contract import (
    Bundle,
    load_bundle_descriptor,
    load_static_target_mesh,
    read_json,
)


CORE_REQUIRED_FILES = (
    "bundle.json",
    "sensor.json",
    "frames.json",
)
OPTIONAL_FILES = (
    "background_adc.npz",
    "environment.json",
    "benchmark_metadata.json",
    "bundle_summary.json",
    "experiment_manifest.json",
)
REQUIRED_AMASS_ARRAYS = ("poses", "trans", "betas", "times", "faces")
OPTIONAL_AMASS_TIMELINES = (
    "bundle_times",
    "source_capture_times",
    "radar_capture_times",
)
SMPL_VERTEX_COUNT = 6890
_MAX_CAMERA_FILE_BYTES = 32 * 1024**2
_MAX_CAMERA_PIXELS = 25_000_000
_MAX_CAMERA_DECODED_BYTES = 100 * 1024**2
_MAX_SCENE_XML_BYTES = 2 * 1024**2
_MAX_ENVIRONMENT_OBJ_BYTES = 64 * 1024**2
_MAX_OBJ_LINE_BYTES = 4 * 1024**2
_MAX_OBJ_VERTICES = 2_000_000
_MAX_OBJ_TRIANGLES = 4_000_000
ENVIRONMENT_PATH_KEYS = (
    "scene_path",
    "scene_file",
    "environment_surfaces_json",
    "environment_obj",
    "environment_mesh_sequence",
    "reconstruction_stats",
)
SCHEMA_FILES = {
    "bundle.json": "bundle.schema.json",
    "sensor.json": "sensor.schema.json",
    "frames.json": "frames.schema.json",
    "environment.json": "environment.schema.json",
}


def _schema_root() -> Path:
    """Locate the normative Bundle v1 schemas in a checkout or installation."""

    source_root = Path(__file__).resolve().parents[2]
    candidates = (
        source_root / "validation" / "schema",
        Path(__file__).resolve().parents[1]
        / "share"
        / "doc"
        / "hermes-radar-sim"
        / "validation"
        / "schema",
        Path(sysconfig.get_path("data"))
        / "share"
        / "doc"
        / "hermes-radar-sim"
        / "validation"
        / "schema",
    )
    for candidate in candidates:
        if all((candidate / name).is_file() for name in SCHEMA_FILES.values()):
            return candidate
    raise FileNotFoundError(
        "Normative HERMES Bundle v1 schemas are not installed"
    )


def _validate_bundle_schemas(root: Path) -> dict[str, str]:
    """Validate all present normative bundle sidecars with Draft 2020-12."""

    schema_root = _schema_root()
    validated: dict[str, str] = {}
    for document_name, schema_name in SCHEMA_FILES.items():
        document_path = root / document_name
        if document_name == "environment.json" and not document_path.exists():
            continue
        schema = read_json(schema_root / schema_name)
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        errors = sorted(
            validator.iter_errors(read_json(document_path)),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
        if errors:
            error = errors[0]
            location = ".".join(str(part) for part in error.absolute_path)
            suffix = f" at {location}" if location else ""
            raise ValueError(
                f"{document_name} violates {schema_name}{suffix}: "
                f"{error.message}"
            )
        validated[document_name] = schema_name
    return validated


def check_bundle_integrity(root: str | Path) -> dict[str, Any]:
    """Check bundle structure without running simulation or comparison metrics."""

    root = Path(root)
    descriptor = load_bundle_descriptor(root)
    profile = str(descriptor["profile"])
    _validate_file_boundary(root, CORE_REQUIRED_FILES)
    bundle = Bundle.load(root)
    amass_report = (
        _validate_amass_sequence(bundle.motion_path)
        if bundle.motion_path is not None
        else {"present": False}
    )
    target_report = _validate_static_target(bundle.target_mesh_path)
    target_obj_report = (
        _preflight_obj_file(bundle.target_obj_path)
        if bundle.target_obj_path is not None
        else {"present": False}
    )
    scene_report = (
        _validate_scene_xml(root, bundle.scene_path)
        if bundle.scene_path is not None
        else {"present": False}
    )
    environment_report = _validate_environment(root, bundle.environment)
    camera_report = _validate_benchmark_camera_images(root)
    sidecar_reports = {
        name: _validate_optional_json_object(root / name)
        for name in ("benchmark_metadata.json", "bundle_summary.json")
    }
    # Run schema conformance after the more specific semantic checks above so
    # existing callers retain actionable cross-file error messages. The schema
    # pass still rejects constraints that the runtime model does not consume.
    schema_report = _validate_bundle_schemas(root)
    return {
        "valid": True,
        "check": "bundle_integrity",
        "path": str(root),
        "profile": profile,
        "data_origin": bundle.data_origin,
        "primary_adc": bundle.primary_adc_key,
        "capabilities": list(bundle.capabilities),
        "required_files": _file_presence(root, CORE_REQUIRED_FILES),
        "optional_files": _file_presence(root, OPTIONAL_FILES),
        "schema_validation": schema_report,
        "sensor": {
            "board_model": bundle.sensor_config.board_model,
            "num_adc_samples": int(bundle.sensor_config.fmcw.num_adc_samples),
            "num_chirps_per_frame": int(
                bundle.sensor_config.fmcw.num_chirps_per_frame
            ),
            "virtual_channels": int(bundle.sensor().hardware.num_virtual_channels),
            "virtual_channel_order": bundle.sensor_config.virtual_channel_order,
        },
        "primary_adc_artifact": {
            "artifact": bundle.primary_adc_key,
            "path": str(bundle.adc_paths[bundle.primary_adc_key].name),
            "origin": bundle.data_origin,
            "shape": _shape(bundle.real_adc),
            "dtype": str(bundle.real_adc.dtype),
            "times_shape": None if bundle.times is None else _shape(bundle.times),
            "finite": True,
        },
        "frames": {
            "count": len(bundle.frames),
            "motion_times_s": [float(value) for value in bundle.motion_times_s],
        },
        "background_adc": None
        if bundle.background_adc is None
        else {
            "shape": _shape(bundle.background_adc),
            "dtype": str(bundle.background_adc.dtype),
            "finite": True,
        },
        "amass_sequence": amass_report,
        "static_target": target_report,
        "target_obj": target_obj_report,
        "scene": scene_report,
        "bundled_simulations": {
            solver: {
                "path": path.name,
                "shape": _shape(bundle.load_bundled_simulation(solver)),
            }
            for solver, path in (bundle.simulated_adc_paths or {}).items()
        },
        "adc_artifacts": {
            name: {
                "path": path.name,
                "origin": bundle.adc_origins[name],
            }
            for name, path in (bundle.adc_paths or {}).items()
        },
        "environment": environment_report,
        "camera_images": camera_report,
        "json_sidecars": sidecar_reports,
    }


def _validate_file_boundary(
    root: Path,
    required_files: tuple[str, ...],
) -> None:
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"Bundle must be a regular directory: {root}")
    for name in required_files:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"Required bundle file was not found: {path}")
    for name in OPTIONAL_FILES:
        path = root / name
        if path.is_symlink():
            raise ValueError(f"Bundle files must not be symlinks: {path}")
        if path.exists() and not path.is_file():
            raise ValueError(f"Bundle path must be a regular file: {path}")


def _file_presence(root: Path, names: tuple[str, ...]) -> dict[str, bool]:
    return {name: (root / name).is_file() for name in names}


def _shape(value: np.ndarray) -> list[int]:
    return [int(item) for item in np.asarray(value).shape]


def _validate_static_target(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"present": False, "not_applicable": True}
    vertices, faces = load_static_target_mesh(path)
    return {
        "present": True,
        "path": path.name,
        "vertices_shape": _shape(vertices),
        "faces_shape": _shape(faces),
        "bounds_m": [
            [float(value) for value in np.min(vertices, axis=0)],
            [float(value) for value in np.max(vertices, axis=0)],
        ],
    }


def _numeric_array(
    value: np.ndarray,
    *,
    label: str,
    kinds: set[str] | None = None,
) -> np.ndarray:
    array = np.asarray(value)
    allowed = {"i", "u", "f"} if kinds is None else kinds
    if array.dtype.kind not in allowed or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain finite plain numeric values")
    return array


def _validate_amass_sequence(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"present": False}
    archive = np.load(path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError(f"Expected a NumPy NPZ archive: {path}")
    with archive:
        missing = [name for name in REQUIRED_AMASS_ARRAYS if name not in archive.files]
        if missing:
            raise ValueError(f"{path} missing required SMPL arrays: {missing}")
        array_names = tuple(archive.files)
        loaded_names = set(REQUIRED_AMASS_ARRAYS).union(
            OPTIONAL_AMASS_TIMELINES
        )
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in array_names
            if name in loaded_names
        }

    poses = _numeric_array(arrays["poses"], label=f"{path} poses")
    trans = _numeric_array(arrays["trans"], label=f"{path} trans")
    betas = _numeric_array(arrays["betas"], label=f"{path} betas")
    faces = np.asarray(arrays["faces"])
    if poses.ndim != 2 or poses.shape[0] == 0 or poses.shape[1] % 3 != 0:
        raise ValueError(f"{path} poses must have non-empty shape [T, 3*K]")
    if trans.shape != (poses.shape[0], 3):
        raise ValueError(f"{path} trans must have shape [{poses.shape[0]}, 3]")
    if betas.ndim != 1 or betas.size < 10:
        raise ValueError(f"{path} betas must be a vector with at least 10 values")

    timelines: dict[str, np.ndarray] = {}
    for key in ("times", *OPTIONAL_AMASS_TIMELINES):
        if key not in arrays:
            continue
        timeline = _numeric_array(arrays[key], label=f"{path} {key}")
        if timeline.shape != (poses.shape[0],):
            raise ValueError(f"{path} {key} must have shape [{poses.shape[0]}]")
        if len(timeline) > 1 and np.any(np.diff(timeline) <= 0.0):
            raise ValueError(f"{path} {key} must be strictly increasing")
        timelines[key] = timeline
    timing_key = "bundle_times" if "bundle_times" in timelines else "times"
    times = timelines[timing_key]

    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{path} faces must have integer shape [F, 3]")
    if faces.size and np.min(faces) < 0:
        raise ValueError(f"{path} faces must contain non-negative vertex indices")
    if faces.size and np.max(faces) >= SMPL_VERTEX_COUNT:
        raise ValueError(
            f"{path} faces reference vertices outside base-SMPL topology"
        )
    return {
        "present": True,
        "poses_shape": _shape(poses),
        "trans_shape": _shape(trans),
        "betas_shape": _shape(betas),
        "times_shape": _shape(times),
        "timing_key": timing_key,
        "timeline_shapes": {
            key: _shape(value) for key, value in timelines.items()
        },
        "faces_shape": _shape(faces),
        "optional_arrays": sorted(
            name for name in array_names if name not in REQUIRED_AMASS_ARRAYS
        ),
    }


def _validate_optional_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"present": False}
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path.name} must contain a JSON object")
    return {"present": True, "keys": sorted(str(key) for key in value)}


def _portable_bundle_path(
    root: Path,
    value: Any,
    *,
    label: str,
    missing_kind: str = "environment asset",
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or "\\" in value or any(
        part in {"", ".", ".."} or ":" in part for part in relative.parts
    ):
        raise ValueError(f"{label} must be a safe bundle-relative path")
    resolved_root = root.resolve(strict=True)
    candidate = resolved_root / relative
    cursor = resolved_root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except FileNotFoundError:
        raise FileNotFoundError(f"Referenced {missing_kind} was not found: {relative}")
    except ValueError as exc:
        raise ValueError(f"{label} escapes the bundle directory") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"Referenced {missing_kind} was not found: {relative}")
    return resolved


def _validate_benchmark_camera_images(root: Path) -> dict[str, Any]:
    """Validate stereo images referenced by optional benchmark metadata."""

    metadata_path = root / "benchmark_metadata.json"
    if not metadata_path.exists():
        return {"present": False, "referenced_images": {}}
    metadata = read_json(metadata_path)
    if not isinstance(metadata, Mapping):
        raise ValueError("benchmark_metadata.json must contain a JSON object")
    contents = metadata.get("bundle_contents", {})
    if not isinstance(contents, Mapping):
        raise ValueError("benchmark_metadata.json bundle_contents must be an object")
    images = contents.get("camera_images")
    if images is None:
        return {"present": False, "referenced_images": {}}
    if not isinstance(images, Mapping) or set(images) != {"left", "right"}:
        raise ValueError("camera_images must contain exactly left and right paths")

    checked: dict[str, str] = {}
    for side in ("left", "right"):
        value = images[side]
        image = _portable_bundle_path(
            root,
            value,
            label=f"benchmark_metadata.json camera_images.{side}",
            missing_kind=f"{side} camera image",
        )
        if image.suffix.lower() != ".png":
            raise ValueError(f"Referenced {side} camera image must be a PNG: {value}")
        if image.stat().st_size > _MAX_CAMERA_FILE_BYTES:
            raise ValueError(f"Referenced {side} camera image is too large: {value}")
        try:
            # ``verify`` checks the complete PNG container without decoding
            # pixels. Reopening and loading then exercises the image decoder,
            # catching truncated or otherwise unreadable pixel data as well.
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(image) as opened:
                    image_format = opened.format
                    width, height = opened.size
                    pixel_count = int(width) * int(height)
                    decoded_bytes = (
                        pixel_count * max(len(opened.getbands()), 1) * 4
                    )
                    if pixel_count > _MAX_CAMERA_PIXELS:
                        raise ValueError(
                            f"Referenced {side} camera image exceeds the "
                            f"{_MAX_CAMERA_PIXELS:,}-pixel limit"
                        )
                    if decoded_bytes > _MAX_CAMERA_DECODED_BYTES:
                        raise ValueError(
                            f"Referenced {side} camera image exceeds the "
                            "decoded-byte limit"
                        )
                    opened.verify()
                with Image.open(image) as opened:
                    opened.load()
                    decoded_format = opened.format
        except (
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
            UnidentifiedImageError,
            OSError,
            SyntaxError,
        ) as exc:
            raise ValueError(
                f"Referenced {side} camera image could not be decoded: {value}"
            ) from exc
        if image_format != "PNG" or decoded_format != "PNG":
            raise ValueError(f"Referenced {side} camera image is not PNG data: {value}")
        checked[side] = str(value)
    return {"present": True, "referenced_images": checked}


def _environment_references(environment: Mapping[str, Any]) -> dict[str, str]:
    references: dict[str, str] = {}
    for key in ENVIRONMENT_PATH_KEYS:
        value = environment.get(key)
        if value is not None:
            references[key] = value
    outputs = environment.get("geometry_outputs", {})
    if outputs is not None:
        if not isinstance(outputs, Mapping):
            raise ValueError("environment.json geometry_outputs must be an object")
        for key in ENVIRONMENT_PATH_KEYS:
            value = outputs.get(key)
            if value is not None:
                references.setdefault(key, value)
    return references


def _validate_scene_xml(root: Path, scene_path: Path) -> dict[str, Any]:
    """Validate a contained, self-contained Mitsuba scene document."""

    if scene_path.stat().st_size > _MAX_SCENE_XML_BYTES:
        raise ValueError(f"Environment scene XML is too large: {scene_path}")
    try:
        scene_tree = ET.parse(scene_path)
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"Invalid environment scene XML: {scene_path}") from exc
    scene_root = scene_tree.getroot()
    for element in scene_root.iter():
        local_name = str(element.tag).rsplit("}", maxsplit=1)[-1]
        if local_name == "include":
            raise ValueError(
                "Mitsuba <include> elements are not permitted in portable "
                "HERMES bundle scenes"
            )
        if not (
            local_name == "string"
            and element.get("name") == "filename"
        ):
            continue
        scene_asset = element.get("value")
        if not isinstance(scene_asset, str) or not scene_asset.strip():
            raise ValueError("Mitsuba filename references must be non-empty")
        relative = Path(scene_asset)
        if relative.is_absolute() or "\\" in scene_asset or any(
            part in {"", ".", ".."} or ":" in part
            for part in relative.parts
        ):
            raise ValueError("Mitsuba filename must be bundle-relative")
        local_candidate = scene_path.parent / relative
        candidate_value = (
            local_candidate.resolve(strict=False)
            if local_candidate.exists()
            else (root / relative).resolve(strict=False)
        )
        try:
            portable_value = candidate_value.relative_to(root.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError("Mitsuba filename escapes the bundle root") from exc
        _portable_bundle_path(
            root,
            portable_value,
            label=f"{scene_path.name} shape filename",
            missing_kind="scene mesh",
        )
    return {"present": True, "path": scene_path.name}


def _preflight_obj_file(path: Path) -> dict[str, Any]:
    """Bound text and triangulation expansion for a bundle OBJ."""

    size = path.stat().st_size
    if size > _MAX_ENVIRONMENT_OBJ_BYTES:
        raise ValueError(f"Bundle OBJ is too large: {path}")
    vertices = 0
    triangles = 0
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if len(raw_line) > _MAX_OBJ_LINE_BYTES:
                raise ValueError(
                    f"Bundle OBJ line {line_number} exceeds the line-size limit"
                )
            parts = raw_line.strip().split()
            if not parts or parts[0].startswith(b"#"):
                continue
            if parts[0] == b"v":
                vertices += 1
                if vertices > _MAX_OBJ_VERTICES:
                    raise ValueError("Bundle OBJ exceeds the vertex limit")
            elif parts[0] == b"f":
                if len(parts) < 4:
                    raise ValueError(
                        f"Invalid bundle OBJ face at line {line_number}"
                    )
                triangles += len(parts) - 3
                if triangles > _MAX_OBJ_TRIANGLES:
                    raise ValueError("Bundle OBJ exceeds the triangle limit")
    if size and vertices < 3:
        raise ValueError(f"Bundle OBJ contains too few vertices: {path}")
    if size and triangles < 1:
        raise ValueError(f"Bundle OBJ contains no triangular faces: {path}")
    if not size:
        raise ValueError(f"Bundle OBJ must not be empty: {path}")
    return {
        "present": True,
        "path": path.name,
        "vertices": vertices,
        "triangles": triangles,
    }


def _validate_environment(
    root: Path,
    environment: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if environment is None:
        return {"present": False, "referenced_assets": {}}
    references = _environment_references(environment)
    checked: dict[str, str] = {}
    for key, value in references.items():
        asset = _portable_bundle_path(root, value, label=f"environment.json {key}")
        suffix = asset.suffix.lower()
        if key == "environment_mesh_sequence" or suffix == ".npz":
            # This validates optional static environment geometry only. Body
            # motion is accepted exclusively through amass_sequence.npz.
            MeshSequence.from_npz(str(asset))
        elif suffix == ".json":
            read_json(asset)
        elif suffix == ".xml":
            _validate_scene_xml(root, asset)
        elif suffix == ".obj":
            _preflight_obj_file(asset)
        checked[key] = str(value)
    return {
        "present": True,
        "keys": sorted(str(key) for key in environment),
        "referenced_assets": checked,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check the structural and cross-file integrity of a HERMES bundle "
            "without running simulation validation."
        )
    )
    parser.add_argument("bundle", type=Path, help="Path to a prepared bundle")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = check_bundle_integrity(args.bundle)
    except Exception as exc:  # pylint: disable=broad-except
        if args.json:
            print(
                json.dumps(
                    {
                        "valid": False,
                        "check": "bundle_integrity",
                        "path": str(args.bundle),
                        "error": str(exc),
                    },
                    indent=2,
                )
            )
        else:
            print(f"Bundle integrity: ERROR: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_text_report(report)
    return 0


def _print_text_report(report: Mapping[str, Any]) -> None:
    sensor = report["sensor"]
    adc = report["primary_adc_artifact"]
    print("Bundle integrity: OK")
    print(f"Path: {report['path']}")
    print(f"Board: {sensor['board_model']}")
    print(f"ADC shape: {adc['shape']} ({adc['dtype']})")
    print(
        "Sensor samples/chirps/channels: "
        f"{sensor['num_adc_samples']}/"
        f"{sensor['num_chirps_per_frame']}/"
        f"{sensor['virtual_channels']}"
    )
    print(f"Profile/origin: {report['profile']}/{report['data_origin']}")
    motion = report["amass_sequence"]
    if motion.get("present"):
        print(
            "SMPL poses/trans/faces: "
            f"{motion['poses_shape']}/{motion['trans_shape']}/{motion['faces_shape']}"
        )
    elif report["static_target"].get("present"):
        target = report["static_target"]
        print(
            "Static target vertices/faces: "
            f"{target['vertices_shape']}/{target['faces_shape']}"
        )
    print(f"Frames: {report['frames']['count']}")
    optional = [
        name for name, present in report["optional_files"].items() if present
    ]
    print(f"Optional files: {', '.join(optional) if optional else 'none'}")


if __name__ == "__main__":
    raise SystemExit(main())
