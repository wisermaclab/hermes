#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Reconstruct a fitted-frame RT-Pose background from LiDAR and stereo images.

This module is an internal stage of ``prepare_rtpose_bundle.py``. LiDAR supplies
metric geometry, the synchronized cameras supply color, and the already baked
SMPL mesh removes foreground returns. It never loads a pickle-backed body
archive and does not expose a second user-facing preparation command.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROI_XYZ = (0.0, 11.6, -6.0, 8.0, -1.6, 3.2)
BODY_BBOX_PAD_XYZ = (0.25, 0.35, 0.30)
BODY_IMAGE_PAD_XY = (85.0, 120.0)
BODY_DEPTH_PAD = (0.35, 0.45)


@dataclass(frozen=True)
class CameraProjection:
    side: str
    intrinsic: np.ndarray
    extrinsic: np.ndarray
    image_path: Path
    image: Image.Image


@dataclass(frozen=True)
class ProjectionResult:
    uv: np.ndarray
    cam_xyz: np.ndarray
    visible: np.ndarray


@dataclass(frozen=True)
class ReconstructionResult:
    background_ply: Path
    stats_path: Path
    stats: dict[str, Any]


def pcd_dtype(
    fields: list[str],
    sizes: list[int],
    types: list[str],
    counts: list[int],
) -> np.dtype:
    """Translate the scalar PCD field declaration used by RT-Pose."""

    dtype_fields: list[tuple[str, Any]] = []
    scalar_types = {
        ("F", 4): "<f4",
        ("F", 8): "<f8",
        ("U", 1): "u1",
        ("U", 2): "<u2",
        ("U", 4): "<u4",
        ("I", 1): "i1",
        ("I", 2): "<i2",
        ("I", 4): "<i4",
    }
    for field, size, pcd_type, count in zip(fields, sizes, types, counts):
        try:
            base = scalar_types[(pcd_type, size)]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported PCD field type: {field} {pcd_type}{size}"
            ) from exc
        dtype_fields.append((field, base if count == 1 else (base, count)))
    return np.dtype(dtype_fields)


def read_pcd(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read an ASCII or binary PCD file and return finite XYZ points."""

    with path.open("rb") as handle:
        header_lines: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"PCD header ended before DATA: {path}")
            decoded = line.decode("latin1").strip()
            header_lines.append(decoded)
            if decoded.startswith("DATA"):
                payload = handle.read()
                break

    header: dict[str, list[str]] = {}
    for line in header_lines:
        parts = line.split()
        if parts and not parts[0].startswith("#"):
            header[parts[0]] = parts[1:]

    fields = list(header.get("FIELDS", []))
    sizes = [int(value) for value in header.get("SIZE", [])]
    types = list(header.get("TYPE", []))
    counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
    points = int(header.get("POINTS", header.get("WIDTH", ["0"]))[0])
    data_kind = header.get("DATA", [""])[0]
    if not fields or not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ValueError(f"Incomplete PCD field declaration: {path}")
    dtype = pcd_dtype(fields, sizes, types, counts)

    if data_kind == "binary":
        cloud = np.frombuffer(payload, dtype=dtype, count=points)
        if len(cloud) != points:
            raise ValueError(f"PCD payload is shorter than declared: {path}")
    elif data_kind == "ascii":
        raw = np.loadtxt(payload.decode("utf-8").splitlines(), dtype=np.float64)
        if raw.ndim == 1:
            raw = raw[None, :]
        if raw.shape != (points, len(fields)):
            raise ValueError(
                f"PCD ASCII payload has shape {raw.shape}, expected "
                f"({points}, {len(fields)})"
            )
        cloud = np.empty(points, dtype=dtype)
        for column, field in enumerate(fields):
            cloud[field] = raw[:, column]
    else:
        raise ValueError(f"Unsupported PCD DATA mode {data_kind!r}: {path}")

    missing = [field for field in ("x", "y", "z") if field not in fields]
    if missing:
        raise ValueError(f"PCD is missing XYZ fields {missing}: {path}")
    xyz = np.stack([cloud["x"], cloud["y"], cloud["z"]], axis=1).astype(
        np.float64
    )
    if xyz.shape != (points, 3) or not np.all(np.isfinite(xyz)):
        raise ValueError(f"PCD XYZ values must be finite: {path}")
    return xyz, {"fields": fields, "data": data_kind, "points": points}


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        intrinsic = np.asarray(payload["intrinsic"], dtype=np.float64).reshape(3, 4)
        extrinsic = np.asarray(payload["extrinsic"], dtype=np.float64).reshape(4, 4)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid RT-Pose camera calibration: {path}") from exc
    if not np.all(np.isfinite(intrinsic)) or not np.all(np.isfinite(extrinsic)):
        raise ValueError(f"Camera calibration must contain finite values: {path}")
    return intrinsic, extrinsic


def load_camera(
    dataset_dir: Path,
    sequence: str,
    side: str,
    source_frame_id: str,
) -> CameraProjection:
    calibration_path = dataset_dir / f"Data/calib/camera/{side}.json"
    image_path = (
        dataset_dir
        / f"Data/sequences/{sequence}/camera/{side}/{source_frame_id}.png"
    )
    for label, path in (("calibration", calibration_path), ("image", image_path)):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"RT-Pose {side} camera {label} was not found: {path}")
    intrinsic, extrinsic = load_calibration(calibration_path)
    try:
        image = Image.open(image_path).convert("RGB")
        image.load()
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid RT-Pose camera image: {image_path}") from exc
    return CameraProjection(
        side=side,
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        image_path=image_path,
        image=image,
    )


def project_points(points: np.ndarray, camera: CameraProjection) -> ProjectionResult:
    points_h = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    cam_xyz = (camera.extrinsic @ points_h.T).T[:, :3]
    uvw = (
        camera.intrinsic
        @ np.column_stack((cam_xyz, np.ones(len(cam_xyz), dtype=np.float64))).T
    ).T
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid_depth = np.abs(uvw[:, 2]) > 1.0e-8
    uv[valid_depth] = uvw[valid_depth, :2] / uvw[valid_depth, 2, None]
    width, height = camera.image.size
    visible = (
        (cam_xyz[:, 2] > 0.0)
        & np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < height)
    )
    return ProjectionResult(uv=uv, cam_xyz=cam_xyz, visible=visible)


def sample_camera_colors(
    points: np.ndarray,
    cameras: tuple[CameraProjection, CameraProjection],
) -> tuple[np.ndarray, np.ndarray, dict[str, ProjectionResult]]:
    colors = np.full((len(points), 3), 160, dtype=np.uint8)
    visible_any = np.zeros(len(points), dtype=bool)
    best_score = np.full(len(points), np.inf, dtype=np.float64)
    projections: dict[str, ProjectionResult] = {}
    for camera in cameras:
        projection = project_points(points, camera)
        projections[camera.side] = projection
        visible_indices = np.flatnonzero(projection.visible)
        if len(visible_indices) == 0:
            continue
        uv = np.rint(projection.uv[visible_indices]).astype(np.int64)
        width, height = camera.image.size
        uv[:, 0] = np.clip(uv[:, 0], 0, width - 1)
        uv[:, 1] = np.clip(uv[:, 1], 0, height - 1)
        center = np.asarray([width * 0.5, height * 0.5], dtype=np.float64)
        score = np.linalg.norm(projection.uv[visible_indices] - center, axis=1)
        take = score < best_score[visible_indices]
        selected = visible_indices[take]
        image = np.asarray(camera.image)
        colors[selected] = image[uv[take, 1], uv[take, 0]]
        best_score[selected] = score[take]
        visible_any[visible_indices] = True
    return colors, visible_any, projections


def roi_mask(points: np.ndarray) -> np.ndarray:
    x_min, x_max, y_min, y_max, z_min, z_max = ROI_XYZ
    return (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
        & (points[:, 2] >= z_min)
        & (points[:, 2] <= z_max)
    )


def body_cull_mask(
    points: np.ndarray,
    body_vertices: np.ndarray,
    camera: CameraProjection,
    point_projection: ProjectionResult,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cull points inside the world/projected bounds of the baked body mesh."""

    vertices = np.asarray(body_vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("Body vertices must have shape (V, 3) with V > 0")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("Body vertices must contain finite values")

    mins = vertices.min(axis=0) - np.asarray(BODY_BBOX_PAD_XYZ)
    maxs = vertices.max(axis=0) + np.asarray(BODY_BBOX_PAD_XYZ)
    in_world_box = np.all((points >= mins) & (points <= maxs), axis=1)

    vertex_projection = project_points(vertices, camera)
    valid_vertices = (
        (vertex_projection.cam_xyz[:, 2] > 0.0)
        & np.isfinite(vertex_projection.uv).all(axis=1)
    )
    in_image_box = np.zeros(len(points), dtype=bool)
    projected_bbox: list[float] | None = None
    if np.any(valid_vertices):
        low = vertex_projection.uv[valid_vertices].min(axis=0) - np.asarray(
            BODY_IMAGE_PAD_XY
        )
        high = vertex_projection.uv[valid_vertices].max(axis=0) + np.asarray(
            BODY_IMAGE_PAD_XY
        )
        vertex_depths = vertex_projection.cam_xyz[valid_vertices, 2]
        min_depth = float(vertex_depths.min() - BODY_DEPTH_PAD[0])
        max_depth = float(vertex_depths.max() + BODY_DEPTH_PAD[1])
        in_image_box = (
            point_projection.visible
            & (point_projection.uv[:, 0] >= low[0])
            & (point_projection.uv[:, 0] <= high[0])
            & (point_projection.uv[:, 1] >= low[1])
            & (point_projection.uv[:, 1] <= high[1])
            & (point_projection.cam_xyz[:, 2] >= min_depth)
            & (point_projection.cam_xyz[:, 2] <= max_depth)
        )
        projected_bbox = [float(low[0]), float(low[1]), float(high[0]), float(high[1])]

    cull = in_world_box | in_image_box
    return cull, {
        "body_bbox_world_min": [float(value) for value in mins],
        "body_bbox_world_max": [float(value) for value in maxs],
        "projected_mesh_bbox_xyxy": projected_bbox,
        "world_box_cull_points": int(in_world_box.sum()),
        "image_box_cull_points": int(in_image_box.sum()),
    }


def write_ascii_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors):
            handle.write(
                f"{point[0]:.5f} {point[1]:.5f} {point[2]:.5f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def reconstruct_background(
    *,
    dataset_dir: Path,
    sequence: str,
    source_frame_id: str,
    body_vertices: np.ndarray,
    output_dir: Path,
) -> ReconstructionResult:
    """Create the colored background cloud needed by the environment fitter."""

    lidar_path = dataset_dir / f"Data/sequences/{sequence}/lidar/{source_frame_id}.pcd"
    if lidar_path.is_symlink() or not lidar_path.is_file():
        raise FileNotFoundError(f"RT-Pose LiDAR frame was not found: {lidar_path}")
    points, pcd_metadata = read_pcd(lidar_path)
    cameras = (
        load_camera(dataset_dir, sequence, "left", source_frame_id),
        load_camera(dataset_dir, sequence, "right", source_frame_id),
    )
    colors, visible, projections = sample_camera_colors(points, cameras)
    visible_roi = visible & roi_mask(points)
    body_cull, body_metadata = body_cull_mask(
        points,
        body_vertices,
        cameras[0],
        projections["left"],
    )
    background = visible_roi & ~body_cull
    if not np.any(background):
        raise ValueError(
            f"No RT-Pose background points remain for sequence {sequence}, "
            f"frame {source_frame_id}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    background_ply = output_dir / f"{source_frame_id}_background_lidar_colored.ply"
    write_ascii_ply(background_ply, points[background], colors[background])
    stats: dict[str, Any] = {
        "sequence": sequence,
        "source_frame_id": source_frame_id,
        "coordinate_frame": "RT-Pose/radar: x=range, y=lateral, z=vertical",
        "source_lidar": lidar_path.relative_to(dataset_dir).as_posix(),
        "source_camera_images": [
            camera.image_path.relative_to(dataset_dir).as_posix()
            for camera in cameras
        ],
        "pcd": pcd_metadata,
        "visible_camera_points": int(visible.sum()),
        "visible_roi_points": int(visible_roi.sum()),
        "body_culled_visible_roi_points": int((visible_roi & body_cull).sum()),
        "background_points": int(background.sum()),
        "roi_xyz_m": list(ROI_XYZ),
        "body_cull": body_metadata,
        "output_background_ply": background_ply.name,
    }
    stats_path = output_dir / "reconstruction_stats.json"
    stats_path.write_text(
        json.dumps(stats, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return ReconstructionResult(
        background_ply=background_ply,
        stats_path=stats_path,
        stats=stats,
    )
