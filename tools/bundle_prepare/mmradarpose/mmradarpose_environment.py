#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Prepare portable mmRadarPose environment geometry from laser data.

mmRadarPose geometry is expressed in the radar frame: x is lateral, y is
range, and z is vertical.  The shared room fitter uses x as range and y as
lateral, so this adapter swaps x/y for fitting and converts every fitted asset
back before it enters a validation bundle.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Any

import numpy as np


TOOL_ROOT = Path(__file__).resolve().parent
BUNDLE_PREPARE_ROOT = TOOL_ROOT.parent
RTPOSE_ROOT = BUNDLE_PREPARE_ROOT / "rtpose"
COORDINATE_FRAME = "mmRadarPose/radar: x=lateral, y=range, z=vertical"
LASER_SUFFIXES = {".csv", ".npy", ".npz", ".pcd", ".ply", ".txt", ".xyz"}


def _load_module(name: str, path: Path) -> ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load bundled helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _rtpose_bundle_tool() -> ModuleType:
    # The RT-Pose adapter owns the shared safe scene-copying implementation.
    if str(RTPOSE_ROOT) not in sys.path:
        sys.path.insert(0, str(RTPOSE_ROOT))
    return _load_module(
        "_hermes_rtpose_bundle_environment_helpers",
        RTPOSE_ROOT / "prepare_rtpose_bundle.py",
    )


def _room_fitter() -> ModuleType:
    return _load_module(
        "_hermes_room_environment_fitter",
        RTPOSE_ROOT / "fit_rtpose_environment_geometry.py",
    )


def _is_environment_mesh(path: Path) -> bool:
    if path.suffix.lower() != ".npz":
        return False
    try:
        archive = np.load(path, allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            return False
        with archive:
            return {"vertices", "faces", "times"}.issubset(archive.files)
    except (OSError, ValueError):
        return False


def _auto_environment_source(dataset_dir: Path) -> Path | None:
    geometry_dirs = (
        dataset_dir / "environment_geometry",
        dataset_dir / "environment" / "environment_geometry",
        dataset_dir / "laser" / "environment_geometry",
        dataset_dir / "lidar" / "environment_geometry",
    )
    for candidate in geometry_dirs:
        if candidate.is_symlink():
            raise ValueError(f"Environment source must not be a symlink: {candidate}")
        if candidate.is_dir():
            return candidate.resolve()

    stems = ("environment", "environment_laser", "background", "map", "laser_map")
    roots = (dataset_dir, dataset_dir / "environment", dataset_dir / "laser", dataset_dir / "lidar")
    for root in roots:
        for stem in stems:
            for suffix in sorted(LASER_SUFFIXES):
                candidate = root / f"{stem}{suffix}"
                if candidate.is_symlink():
                    raise ValueError(f"Laser source must not be a symlink: {candidate}")
                if candidate.is_file():
                    return candidate.resolve()
    return None


def _load_pcd(path: Path) -> np.ndarray:
    """Read finite XYZ values from an ASCII or binary PCD cloud."""

    with path.open("rb") as handle:
        header: dict[str, list[str]] = {}
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"PCD header ended before DATA: {path}")
            parts = line.decode("latin1").strip().split()
            if parts and not parts[0].startswith("#"):
                header[parts[0].upper()] = parts[1:]
            if parts and parts[0].upper() == "DATA":
                payload = handle.read()
                break

    fields = header.get("FIELDS", [])
    sizes = [int(value) for value in header.get("SIZE", [])]
    kinds = header.get("TYPE", [])
    counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
    count = int(header.get("POINTS", header.get("WIDTH", ["0"]))[0])
    if not fields or not (len(fields) == len(sizes) == len(kinds) == len(counts)):
        raise ValueError(f"Incomplete PCD declaration: {path}")
    if any(value != 1 for value in counts):
        raise ValueError(f"Vector-valued PCD fields are not supported: {path}")
    scalar_types = {
        ("F", 4): "<f4", ("F", 8): "<f8",
        ("U", 1): "u1", ("U", 2): "<u2", ("U", 4): "<u4",
        ("I", 1): "i1", ("I", 2): "<i2", ("I", 4): "<i4",
    }
    try:
        dtype = np.dtype(
            [(field, scalar_types[(kind.upper(), size)])
             for field, kind, size in zip(fields, kinds, sizes)]
        )
    except KeyError as exc:
        raise ValueError(f"Unsupported PCD scalar type: {path}") from exc

    data_kind = header.get("DATA", [""])[0].lower()
    if data_kind == "binary":
        cloud = np.frombuffer(payload, dtype=dtype, count=count)
    elif data_kind == "ascii":
        raw = np.loadtxt(payload.decode("utf-8").splitlines(), dtype=np.float64)
        raw = np.atleast_2d(raw)
        if raw.shape != (count, len(fields)):
            raise ValueError(f"PCD payload shape {raw.shape} does not match its header")
        cloud = np.empty(count, dtype=dtype)
        for column, field in enumerate(fields):
            cloud[field] = raw[:, column]
    else:
        raise ValueError(f"Unsupported PCD DATA mode {data_kind!r}: {path}")
    if len(cloud) != count or any(axis not in fields for axis in ("x", "y", "z")):
        raise ValueError(f"PCD must contain the declared number of XYZ points: {path}")
    return np.stack([cloud[axis] for axis in ("x", "y", "z")], axis=1)


def load_laser_points(path: Path) -> np.ndarray:
    """Load a pickle-free laser cloud without treating radar targets as geometry."""

    suffix = path.suffix.lower()
    if suffix == ".pcd":
        values = _load_pcd(path)
    elif suffix == ".ply":
        values, _ = _room_fitter().read_ascii_ply(path)
    elif suffix == ".npy":
        values = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError(f"Expected an NPZ laser archive: {path}")
        with archive:
            keys = [key for key in ("points", "xyz", "pointcloud", "vertices") if key in archive.files]
            if len(keys) != 1:
                raise ValueError(
                    "Laser NPZ must contain exactly one of points, xyz, pointcloud, or vertices"
                )
            values = np.array(archive[keys[0]], copy=True)
    elif suffix in {".csv", ".txt", ".xyz"}:
        values = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    else:
        raise ValueError(f"Unsupported laser point-cloud format: {path.suffix}")

    points = np.asarray(values)
    if points.ndim != 2 or points.shape[1] < 3 or points.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("Laser points must be a real numeric array shaped (N, 3+)")
    points = np.asarray(points[:, :3], dtype=np.float64)
    if len(points) == 0 or not np.all(np.isfinite(points)):
        raise ValueError("Laser points must be non-empty and finite")
    return points


def _write_ascii_ply(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for x, y, z in points:
            handle.write(f"{x:.6f} {y:.6f} {z:.6f} 160 160 160\n")


def _remove_body_points(points: np.ndarray, body_vertices: np.ndarray) -> tuple[np.ndarray, int]:
    body = np.asarray(body_vertices, dtype=np.float64)
    if body.ndim != 2 or body.shape[1] != 3 or not np.all(np.isfinite(body)):
        raise ValueError("Body vertices must be a finite array shaped (V, 3)")
    lower = np.min(body, axis=0) - np.asarray([0.20, 0.20, 0.12])
    upper = np.max(body, axis=0) + np.asarray([0.20, 0.20, 0.12])
    inside = np.all((points >= lower) & (points <= upper), axis=1)
    remaining = points[~inside]
    if len(remaining) == 0:
        raise ValueError("All laser points overlap the selected body mesh")
    return remaining, int(np.count_nonzero(inside))


def _convert_fitted_assets(
    temporary_output: Path,
    destination: Path,
    *,
    source_name: str,
    input_points: int,
    removed_body_points: int,
) -> dict[str, str]:
    fitter = _room_fitter()
    payload = json.loads(
        (temporary_output / "environment_surfaces.json").read_text(encoding="utf-8")
    )
    surfaces = []
    for item in payload.get("surfaces", []):
        center = np.asarray(item["center"], dtype=np.float64)[[1, 0, 2]]
        size = np.asarray(item["size"], dtype=np.float64)[[1, 0, 2]]
        surfaces.append(
            fitter.BoxSurface(
                surface_id=str(item["id"]),
                semantic=str(item["semantic"]),
                material=str(item["material"]),
                center=center,
                size=size,
                source_points=int(item["source_points"]),
                confidence=float(item["confidence"]),
            )
        )

    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "environment_surfaces.json"
    obj_path = destination / "environment_surfaces.obj"
    xml_path = destination / "environment_scene.xml"
    mesh_path = destination / "environment_mesh_sequence.npz"
    fit = dict(payload.get("fit", {}))
    converted_fit = {
        "input_points": int(input_points),
        "body_points_removed": int(removed_body_points),
        "floor_z_m": fit.pop("floor_z_m", None),
        "back_wall_y_m": fit.pop("back_wall_x_m", None),
        "left_wall_x_m": fit.pop("left_wall_y_m", None),
        "right_wall_x_m": fit.pop("right_wall_y_m", None),
        **fit,
    }
    converted_payload = {
        "source_laser": source_name,
        "coordinate_frame": COORDINATE_FRAME,
        "fit": converted_fit,
        "material_profiles": payload.get("material_profiles", {}),
        "surfaces": [fitter.surface_to_json(surface) for surface in surfaces],
        "outputs": {
            "obj": obj_path.name,
            "xml": xml_path.name,
            "mesh_sequence_npz": mesh_path.name,
            "preview": None,
        },
    }
    json_path.write_text(
        json.dumps(converted_payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    fitter.write_obj(obj_path, surfaces)
    obj_path.write_text(
        obj_path.read_text(encoding="utf-8").replace(
            "# RT-Pose fitted environment scatterer geometry",
            "# HERMES fitted mmRadarPose environment geometry",
            1,
        ),
        encoding="utf-8",
        newline="\n",
    )
    fitter.write_xml(xml_path, surfaces)
    vertices, faces = fitter.build_mesh(surfaces)
    np.savez_compressed(
        mesh_path,
        vertices=np.asarray(vertices, dtype=np.float32)[None, ...],
        faces=np.asarray(faces, dtype=np.uint32),
        times=np.asarray([0.0], dtype=np.float64),
    )
    return {
        "environment_surfaces_json": "environment_geometry/environment_surfaces.json",
        "environment_obj": "environment_geometry/environment_surfaces.obj",
        "scene_path": "environment_geometry/environment_scene.xml",
        "environment_mesh_sequence": "environment_geometry/environment_mesh_sequence.npz",
        "geometry_source": "laser_fit",
        "source_laser": source_name,
    }


def fit_laser_environment(
    source: Path,
    bundle_dir: Path,
    *,
    body_vertices: np.ndarray,
) -> dict[str, str]:
    points = load_laser_points(source)
    input_point_count = len(points)
    points, removed = _remove_body_points(points, body_vertices)
    # Shared fitter coordinates: x=range, y=lateral, z=vertical.
    fit_points = points[:, [1, 0, 2]]
    bounds = np.stack([np.min(fit_points, axis=0), np.max(fit_points, axis=0)])
    padding = np.asarray([0.10, 0.10, 0.10])
    low, high = bounds[0] - padding, bounds[1] + padding
    roi = f"{low[0]},{high[0]},{low[1]},{high[1]},{low[2]},{high[2]}"

    with tempfile.TemporaryDirectory(prefix=".mmradarpose_laser_", dir=bundle_dir.parent) as tmp:
        temporary = Path(tmp)
        ply = temporary / "registered_laser.ply"
        output = temporary / "fit"
        _write_ascii_ply(ply, fit_points)
        command = [
            sys.executable,
            str(RTPOSE_ROOT / "fit_rtpose_environment_geometry.py"),
            str(ply),
            "--output-dir", str(output),
            "--roi", roi,
            "--wall-mode", "solid",
            "--no-preview",
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            raise ValueError(f"Laser environment fitting failed: {detail}") from exc
        return _convert_fitted_assets(
            output,
            bundle_dir / "environment_geometry",
            source_name=source.name,
            input_points=input_point_count,
            removed_body_points=removed,
        )


def prepare_environment_geometry(
    *,
    dataset_dir: Path,
    bundle_dir: Path,
    body_vertices: np.ndarray,
    provided_geometry: Path | None,
) -> dict[str, Any]:
    """Reuse fitted geometry or fit a registered laser cloud; never read cameras."""

    source = provided_geometry
    if source is not None:
        source = source.expanduser().resolve()
        if source.is_symlink() or not source.exists():
            raise FileNotFoundError(f"Environment source was not found: {source}")
    else:
        source = _auto_environment_source(dataset_dir)
    if source is None:
        return {}

    if source.is_dir() or source.suffix.lower() == ".xml" or _is_environment_mesh(source):
        outputs = _rtpose_bundle_tool().copy_provided_environment_geometry(
            source,
            bundle_dir,
        )
        outputs["geometry_source"] = "provided" if provided_geometry is not None else "dataset_cache"
        return outputs
    if source.suffix.lower() not in LASER_SUFFIXES:
        raise ValueError(
            "--environment-geometry must name fitted geometry or a supported laser cloud"
        )
    outputs = fit_laser_environment(source, bundle_dir, body_vertices=body_vertices)
    if provided_geometry is None:
        outputs["geometry_source"] = "dataset_laser_fit"
    return outputs


def environment_payload(outputs: dict[str, Any]) -> dict[str, Any]:
    if not outputs:
        return {
            "type": "empty",
            "coordinate_frame": "mmRadarPose/radar",
            "note": "No fitted geometry or registered laser cloud was available.",
        }
    payload: dict[str, Any] = {
        "type": "fitted_geometry",
        "coordinate_frame": "mmRadarPose/radar",
        "metadata": {
            "source": "tools/bundle_prepare/mmradarpose/mmradarpose_environment.py",
            "format": "mmwave_validation_environment",
            "sensor_inputs": ["laser"],
        },
        "geometry_outputs": dict(outputs),
    }
    for key in (
        "scene_path",
        "environment_surfaces_json",
        "environment_obj",
        "environment_mesh_sequence",
    ):
        if key in outputs:
            payload[key] = outputs[key]
    return payload


def generated_environment_files(outputs: dict[str, Any]) -> list[str]:
    files = {
        str(value)
        for key, value in outputs.items()
        if key in {
            "scene_path",
            "environment_surfaces_json",
            "environment_obj",
            "environment_mesh_sequence",
            "reconstruction_stats",
        }
        and isinstance(value, str)
    }
    return sorted(files)
