#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Fit simple RT-Pose environment scatterer geometry from a background cloud.

Input is a colored PLY from ``reconstruct_rtpose_background.py``. The script
fits axis-aligned room surfaces and clusters remaining above-floor points into
simple furniture cuboids. Outputs include JSON, OBJ, Mitsuba/Sionna XML, and a
static mesh-sequence NPZ.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class AxisPlane:
    name: str
    axis: int
    value: float
    inliers: np.ndarray
    count: int
    extents: dict[str, tuple[float, float]]


@dataclass
class BoxSurface:
    surface_id: str
    semantic: str
    material: str
    center: np.ndarray
    size: np.ndarray
    source_points: int
    confidence: float


AXIS_NAMES = ("x", "y", "z")
MATERIAL_PROFILES = {
    # ITU material choices are restricted to the valid Sionna catalogue at
    # 77-81 GHz. For example, marble and plywood would be visually plausible
    # for panels/wood products, but their local ITU definitions stop below this
    # radar band.
    "wall": {
        "itu_type": "concrete",
        "thickness_m": 0.03,
        "scattering_coefficient": 0.12,
        "xpd_coefficient": 0.03,
        "rationale": (
            "Closest valid 77-81 GHz proxy for ceramic or laminate/composite "
            "cladding panels; marble/plywood are outside the local ITU "
            "frequency range."
        ),
    },
    "floor": {
        "itu_type": "wood",
        "thickness_m": 0.04,
        "scattering_coefficient": 0.20,
        "xpd_coefficient": 0.05,
        "rationale": (
            "Indoor floor proxy. ITU floorboard is only defined above 50 GHz "
            "and fails at Sionna's default scene-load frequency, so wood is "
            "used as a robust valid substitute."
        ),
    },
    "furniture": {
        "itu_type": "chipboard",
        "thickness_m": 0.03,
        "scattering_coefficient": 0.35,
        "xpd_coefficient": 0.10,
        "rationale": (
            "Effective material for laminate/melamine-covered MDF or "
            "particleboard tabletops plus mixed office-chair fabric/plastic/"
            "metal clusters."
        ),
    },
}


def parse_float_list(text: str, expected: int, name: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if len(values) != expected:
        raise argparse.ArgumentTypeError(
            f"{name} must contain {expected} comma-separated values"
        )
    return values


def read_ascii_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        vertex_count = None
        properties: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path} ended before PLY end_header")
            stripped = line.strip()
            if stripped.startswith("element vertex"):
                vertex_count = int(stripped.split()[-1])
            elif stripped.startswith("property"):
                parts = stripped.split()
                if len(parts) == 3:
                    properties.append(parts[2])
            elif stripped == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"{path} does not declare element vertex")
        data = np.loadtxt(handle, max_rows=vertex_count)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 3:
        raise ValueError(f"{path} has fewer than 3 vertex columns")
    points = data[:, :3].astype(np.float64)
    colors = np.full((len(points), 3), 160, dtype=np.uint8)
    if data.shape[1] >= 6:
        colors = np.clip(data[:, 3:6], 0, 255).astype(np.uint8)
    return points, colors


def axis_plane_candidates(
    points: np.ndarray,
    axis: int,
    *,
    bin_width: float,
    min_bin_count: int,
    threshold: float,
    min_inliers: int,
) -> list[AxisPlane]:
    values = points[:, axis]
    bins = np.arange(values.min(), values.max() + bin_width * 1.5, bin_width)
    hist, edges = np.histogram(values, bins=bins)
    peak_bins = [int(i) for i in np.argsort(hist)[::-1] if hist[i] >= min_bin_count]
    candidates: list[AxisPlane] = []
    other_axes = [i for i in range(3) if i != axis]
    accepted_values: list[float] = []
    for peak_bin in peak_bins:
        seed = float((edges[peak_bin] + edges[peak_bin + 1]) * 0.5)
        if any(abs(seed - value) <= threshold * 2.0 for value in accepted_values):
            continue
        inliers = np.abs(values - seed) <= threshold
        count = int(inliers.sum())
        if count < min_inliers:
            continue
        refined = float(np.median(values[inliers]))
        if any(abs(refined - value) <= threshold * 2.0 for value in accepted_values):
            continue
        inliers = np.abs(values - refined) <= threshold
        count = int(inliers.sum())
        if count < min_inliers:
            continue
        extents = {}
        for other_axis in other_axes:
            vals = points[inliers, other_axis]
            extents[AXIS_NAMES[other_axis]] = (
                float(np.percentile(vals, 2)),
                float(np.percentile(vals, 98)),
            )
        candidates.append(
            AxisPlane(
                name=AXIS_NAMES[axis],
                axis=axis,
                value=refined,
                inliers=inliers,
                count=count,
                extents=extents,
            )
        )
        accepted_values.append(refined)
    candidates.sort(key=lambda candidate: candidate.count, reverse=True)
    return candidates


def choose_floor(points: np.ndarray, args) -> AxisPlane:
    candidates = axis_plane_candidates(
        points,
        2,
        bin_width=args.plane_bin_width,
        min_bin_count=args.min_plane_bin_count,
        threshold=args.plane_threshold,
        min_inliers=args.min_plane_inliers,
    )
    if not candidates:
        raise SystemExit("Could not fit a floor plane from z-axis peaks")
    # Prefer a strong low-z horizontal plane.
    z_limit = float(np.percentile(points[:, 2], 35))
    low_candidates = [candidate for candidate in candidates if candidate.value <= z_limit]
    return low_candidates[0] if low_candidates else candidates[0]


def valid_wall_candidate(candidate: AxisPlane, *, min_height: float, min_span: float) -> bool:
    extents = candidate.extents
    z_span = extents.get("z", (0.0, 0.0))[1] - extents.get("z", (0.0, 0.0))[0]
    if candidate.axis == 0:
        lateral_span = extents.get("y", (0.0, 0.0))[1] - extents.get("y", (0.0, 0.0))[0]
    else:
        lateral_span = extents.get("x", (0.0, 0.0))[1] - extents.get("x", (0.0, 0.0))[0]
    return z_span >= min_height and lateral_span >= min_span


def choose_walls(points: np.ndarray, args) -> tuple[Optional[AxisPlane], Optional[AxisPlane], Optional[AxisPlane]]:
    x_candidates = axis_plane_candidates(
        points,
        0,
        bin_width=args.plane_bin_width,
        min_bin_count=args.min_wall_bin_count,
        threshold=args.wall_threshold,
        min_inliers=args.min_wall_inliers,
    )
    x_candidates = [
        candidate
        for candidate in x_candidates
        if candidate.value >= np.percentile(points[:, 0], 45)
        and valid_wall_candidate(candidate, min_height=args.min_wall_height, min_span=args.min_wall_span)
    ]
    back_wall = x_candidates[0] if x_candidates else None

    y_candidates = axis_plane_candidates(
        points,
        1,
        bin_width=args.plane_bin_width,
        min_bin_count=args.min_wall_bin_count,
        threshold=args.wall_threshold,
        min_inliers=args.min_wall_inliers,
    )
    y_candidates = [
        candidate
        for candidate in y_candidates
        if valid_wall_candidate(candidate, min_height=args.min_wall_height, min_span=args.min_wall_span)
    ]
    if not y_candidates:
        return back_wall, None, None
    y_values = np.array([candidate.value for candidate in y_candidates])
    left_wall = y_candidates[int(np.argmin(y_values))]
    right_wall = y_candidates[int(np.argmax(y_values))]
    if left_wall is right_wall:
        right_wall = None
    return back_wall, left_wall, right_wall


def box_from_bounds(
    surface_id: str,
    semantic: str,
    material: str,
    bounds: tuple[float, float, float, float, float, float],
    *,
    source_points: int,
    confidence: float,
) -> BoxSurface:
    x0, x1, y0, y1, z0, z1 = bounds
    return BoxSurface(
        surface_id=surface_id,
        semantic=semantic,
        material=material,
        center=np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5], dtype=np.float64),
        size=np.array([x1 - x0, y1 - y0, z1 - z0], dtype=np.float64),
        source_points=int(source_points),
        confidence=float(confidence),
    )


def room_surfaces(
    points: np.ndarray,
    floor: AxisPlane,
    back_wall: Optional[AxisPlane],
    left_wall: Optional[AxisPlane],
    right_wall: Optional[AxisPlane],
    args,
) -> list[BoxSurface]:
    if args.wall_mode == "patch":
        return room_patch_surfaces(points, floor, back_wall, left_wall, right_wall, args)
    return room_solid_surfaces(points, floor, back_wall, left_wall, right_wall, args)


def room_solid_surfaces(
    points: np.ndarray,
    floor: AxisPlane,
    back_wall: Optional[AxisPlane],
    left_wall: Optional[AxisPlane],
    right_wall: Optional[AxisPlane],
    args,
) -> list[BoxSurface]:
    z_floor = floor.value
    z_top = float(np.percentile(points[:, 2], args.room_top_percentile))
    x_min = max(0.0, float(np.percentile(points[:, 0], 1)) - 0.3)
    x_max = back_wall.value if back_wall is not None else float(np.percentile(points[:, 0], 98))
    y_min = left_wall.value if left_wall is not None else float(np.percentile(points[:, 1], 2))
    y_max = right_wall.value if right_wall is not None else float(np.percentile(points[:, 1], 98))
    thickness = args.room_surface_thickness
    surfaces: list[BoxSurface] = []
    surfaces.append(
        box_from_bounds(
            "floor",
            "floor",
            "floor",
            (x_min, x_max, y_min, y_max, z_floor - thickness, z_floor),
            source_points=floor.count,
            confidence=min(1.0, floor.count / max(1, len(points)) * 3.0),
        )
    )
    if back_wall is not None:
        surfaces.append(
            box_from_bounds(
                "back_wall",
                "wall",
                "wall",
                (x_max, x_max + thickness, y_min, y_max, z_floor, z_top),
                source_points=back_wall.count,
                confidence=min(1.0, back_wall.count / max(1, len(points)) * 3.0),
            )
        )
    if left_wall is not None:
        surfaces.append(
            box_from_bounds(
                "left_wall",
                "wall",
                "wall",
                (x_min, x_max, y_min - thickness, y_min, z_floor, z_top),
                source_points=left_wall.count,
                confidence=min(1.0, left_wall.count / max(1, len(points)) * 3.0),
            )
        )
    if right_wall is not None:
        surfaces.append(
            box_from_bounds(
                "right_wall",
                "wall",
                "wall",
                (x_min, x_max, y_max, y_max + thickness, z_floor, z_top),
                source_points=right_wall.count,
                confidence=min(1.0, right_wall.count / max(1, len(points)) * 3.0),
            )
        )
    return surfaces


def room_bounds_from_planes(
    points: np.ndarray,
    floor: AxisPlane,
    back_wall: Optional[AxisPlane],
    left_wall: Optional[AxisPlane],
    right_wall: Optional[AxisPlane],
    args,
) -> tuple[float, float, float, float, float, float]:
    z_floor = floor.value
    z_top = float(np.percentile(points[:, 2], args.room_top_percentile))
    x_min = max(0.0, float(np.percentile(points[:, 0], 1)) - 0.3)
    x_max = back_wall.value if back_wall is not None else float(np.percentile(points[:, 0], 98))
    y_min = left_wall.value if left_wall is not None else float(np.percentile(points[:, 1], 2))
    y_max = right_wall.value if right_wall is not None else float(np.percentile(points[:, 1], 98))
    return x_min, x_max, y_min, y_max, z_floor, z_top


def dilate_bool_grid(grid: np.ndarray, iterations: int) -> np.ndarray:
    out = np.asarray(grid, dtype=bool)
    for _ in range(max(0, int(iterations))):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        grown = np.zeros_like(out)
        for di in range(3):
            for dj in range(3):
                grown |= padded[di : di + out.shape[0], dj : dj + out.shape[1]]
        out = grown
    return out


def occupied_grid_components(grid: np.ndarray) -> list[np.ndarray]:
    visited = np.zeros_like(grid, dtype=bool)
    components: list[np.ndarray] = []
    rows, cols = grid.shape
    for row in range(rows):
        for col in range(cols):
            if not grid[row, col] or visited[row, col]:
                continue
            queue = deque([(row, col)])
            visited[row, col] = True
            cells = []
            while queue:
                current = queue.popleft()
                cells.append(current)
                cr, cc = current
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < rows
                            and 0 <= nc < cols
                            and grid[nr, nc]
                            and not visited[nr, nc]
                        ):
                            visited[nr, nc] = True
                            queue.append((nr, nc))
            components.append(np.asarray(cells, dtype=np.int64))
    components.sort(key=len, reverse=True)
    return components


def wall_patch_surfaces(
    points: np.ndarray,
    wall: AxisPlane,
    surface_prefix: str,
    room_bounds: tuple[float, float, float, float, float, float],
    args,
) -> list[BoxSurface]:
    x_min, x_max, y_min, y_max, z_floor, z_top = room_bounds
    thickness = args.room_surface_thickness
    if wall.axis == 0:
        u_axis = 1
        u_min, u_max = y_min, y_max
        fixed_bounds = (wall.value, wall.value + thickness)
    elif wall.axis == 1:
        u_axis = 0
        u_min, u_max = x_min, x_max
        if wall.value <= (y_min + y_max) * 0.5:
            fixed_bounds = (wall.value - thickness, wall.value)
        else:
            fixed_bounds = (wall.value, wall.value + thickness)
    else:
        raise ValueError("wall_patch_surfaces only supports vertical x/y walls")

    inliers = wall.inliers.copy()
    inliers &= points[:, 2] >= z_floor + args.wall_patch_min_above_floor
    inliers &= points[:, 2] <= z_top
    inliers &= points[:, u_axis] >= u_min
    inliers &= points[:, u_axis] <= u_max
    wall_points = points[inliers]
    if len(wall_points) < args.min_wall_patch_points:
        return []

    u_bins = np.arange(u_min, u_max + args.wall_patch_grid, args.wall_patch_grid)
    z_bins = np.arange(z_floor, z_top + args.wall_patch_grid, args.wall_patch_grid)
    if len(u_bins) < 3 or len(z_bins) < 3:
        return []
    hist, _, _ = np.histogram2d(wall_points[:, u_axis], wall_points[:, 2], bins=(u_bins, z_bins))
    occupied = hist >= args.wall_patch_min_cell_points
    occupied = dilate_bool_grid(occupied, args.wall_patch_dilation)
    components = occupied_grid_components(occupied)

    surfaces: list[BoxSurface] = []
    for component in components:
        u0 = float(u_bins[int(component[:, 0].min())])
        u1 = float(u_bins[int(component[:, 0].max()) + 1])
        z0 = float(z_bins[int(component[:, 1].min())])
        z1 = float(z_bins[int(component[:, 1].max()) + 1])
        if (u1 - u0) < args.min_wall_patch_span or (z1 - z0) < args.min_wall_patch_height:
            continue
        point_mask = (
            inliers
            & (points[:, u_axis] >= u0)
            & (points[:, u_axis] <= u1)
            & (points[:, 2] >= z0)
            & (points[:, 2] <= z1)
        )
        source_points = int(point_mask.sum())
        if source_points < args.min_wall_patch_points:
            continue
        surface_id = f"{surface_prefix}_patch_{len(surfaces) + 1:02d}"
        if wall.axis == 0:
            bounds = (fixed_bounds[0], fixed_bounds[1], u0, u1, z0, z1)
        else:
            bounds = (u0, u1, fixed_bounds[0], fixed_bounds[1], z0, z1)
        surfaces.append(
            box_from_bounds(
                surface_id,
                "wall",
                "wall",
                bounds,
                source_points=source_points,
                confidence=min(1.0, source_points / max(1, args.min_wall_patch_points * 5)),
            )
        )
        if len(surfaces) >= args.max_wall_patches_per_wall:
            break
    return surfaces


def room_patch_surfaces(
    points: np.ndarray,
    floor: AxisPlane,
    back_wall: Optional[AxisPlane],
    left_wall: Optional[AxisPlane],
    right_wall: Optional[AxisPlane],
    args,
) -> list[BoxSurface]:
    x_min, x_max, y_min, y_max, z_floor, _ = room_bounds_from_planes(
        points, floor, back_wall, left_wall, right_wall, args
    )
    thickness = args.room_surface_thickness
    surfaces = [
        box_from_bounds(
            "floor",
            "floor",
            "floor",
            (x_min, x_max, y_min, y_max, z_floor - thickness, z_floor),
            source_points=floor.count,
            confidence=min(1.0, floor.count / max(1, len(points)) * 3.0),
        )
    ]
    room_bounds = room_bounds_from_planes(points, floor, back_wall, left_wall, right_wall, args)
    if back_wall is not None:
        patches = wall_patch_surfaces(points, back_wall, "back_wall", room_bounds, args)
        surfaces.extend(
            patches
            if patches
            else room_solid_surfaces(points, floor, back_wall, None, None, args)[1:]
        )
    if left_wall is not None:
        patches = wall_patch_surfaces(points, left_wall, "left_wall", room_bounds, args)
        surfaces.extend(
            patches
            if patches
            else room_solid_surfaces(points, floor, None, left_wall, None, args)[1:]
        )
    if right_wall is not None:
        patches = wall_patch_surfaces(points, right_wall, "right_wall", room_bounds, args)
        surfaces.extend(
            patches
            if patches
            else room_solid_surfaces(points, floor, None, None, right_wall, args)[1:]
        )
    return surfaces


def connected_components_grid(
    points_xy: np.ndarray,
    *,
    grid_size: float,
    min_points: int,
) -> list[np.ndarray]:
    if len(points_xy) == 0:
        return []
    origin = points_xy.min(axis=0)
    cells = np.floor((points_xy - origin[None, :]) / grid_size).astype(np.int64)
    cell_to_indices: dict[tuple[int, int], list[int]] = {}
    for index, cell in enumerate(cells):
        cell_to_indices.setdefault((int(cell[0]), int(cell[1])), []).append(index)

    visited: set[tuple[int, int]] = set()
    components: list[np.ndarray] = []
    for cell in cell_to_indices:
        if cell in visited:
            continue
        queue = deque([cell])
        visited.add(cell)
        component_cells = []
        while queue:
            current = queue.popleft()
            component_cells.append(current)
            cx, cy = current
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (cx + dx, cy + dy)
                    if neighbor in cell_to_indices and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
        indices = np.array(
            [index for component_cell in component_cells for index in cell_to_indices[component_cell]],
            dtype=np.int64,
        )
        if len(indices) >= min_points:
            components.append(indices)
    components.sort(key=len, reverse=True)
    return components


def furniture_surfaces(
    points: np.ndarray,
    room: list[BoxSurface],
    floor: AxisPlane,
    back_wall: Optional[AxisPlane],
    left_wall: Optional[AxisPlane],
    right_wall: Optional[AxisPlane],
    args,
) -> list[BoxSurface]:
    z_floor = floor.value
    candidate = (
        (points[:, 2] > z_floor + args.furniture_min_above_floor)
        & (points[:, 2] < z_floor + args.furniture_max_above_floor)
    )
    candidate &= np.abs(points[:, 2] - z_floor) > args.plane_threshold
    if back_wall is not None:
        candidate &= points[:, 0] <= back_wall.value - args.wall_exclusion
        candidate &= np.abs(points[:, 0] - back_wall.value) > args.wall_exclusion
    if left_wall is not None:
        candidate &= points[:, 1] >= left_wall.value + args.wall_exclusion
        candidate &= np.abs(points[:, 1] - left_wall.value) > args.wall_exclusion
    if right_wall is not None:
        candidate &= points[:, 1] <= right_wall.value - args.wall_exclusion
        candidate &= np.abs(points[:, 1] - right_wall.value) > args.wall_exclusion

    candidate_indices = np.where(candidate)[0]
    components = connected_components_grid(
        points[candidate_indices, :2],
        grid_size=args.furniture_grid_size,
        min_points=args.min_furniture_points,
    )
    surfaces: list[BoxSurface] = []
    for local_indices in components:
        if len(surfaces) >= args.max_furniture_objects:
            break
        indices = candidate_indices[local_indices]
        cluster = points[indices]
        low = np.percentile(cluster, 2, axis=0)
        high = np.percentile(cluster, 98, axis=0)
        size = high - low
        if size[0] < args.min_furniture_size_xy and size[1] < args.min_furniture_size_xy:
            continue
        if size[2] < args.min_furniture_height:
            high[2] = low[2] + args.min_furniture_height
        low[:2] -= args.furniture_padding_xy
        high[:2] += args.furniture_padding_xy
        low[2] = max(z_floor + args.furniture_bottom_clearance, low[2] - args.furniture_padding_z)
        high[2] = high[2] + args.furniture_padding_z
        semantic = "table_chair_cluster"
        cluster_number = len(surfaces) + 1
        surfaces.append(
            box_from_bounds(
                f"furniture_{cluster_number:02d}",
                semantic,
                "furniture",
                (float(low[0]), float(high[0]), float(low[1]), float(high[1]), float(low[2]), float(high[2])),
                source_points=len(indices),
                confidence=min(1.0, len(indices) / max(1, args.min_furniture_points * 4)),
            )
        )
    return surfaces


def box_vertices_faces(surface: BoxSurface) -> tuple[np.ndarray, np.ndarray]:
    cx, cy, cz = surface.center
    sx, sy, sz = surface.size * 0.5
    vertices = np.array(
        [
            [cx - sx, cy - sy, cz - sz],
            [cx + sx, cy - sy, cz - sz],
            [cx + sx, cy + sy, cz - sz],
            [cx - sx, cy + sy, cz - sz],
            [cx - sx, cy - sy, cz + sz],
            [cx + sx, cy - sy, cz + sz],
            [cx + sx, cy + sy, cz + sz],
            [cx - sx, cy + sy, cz + sz],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.uint32,
    )
    return vertices, faces


def build_mesh(surfaces: list[BoxSurface]) -> tuple[np.ndarray, np.ndarray]:
    vertices_all = []
    faces_all = []
    offset = 0
    for surface in surfaces:
        vertices, faces = box_vertices_faces(surface)
        vertices_all.append(vertices)
        faces_all.append(faces + offset)
        offset += len(vertices)
    if vertices_all:
        vertices = np.vstack(vertices_all).astype(np.float32)
        faces = np.vstack(faces_all).astype(np.uint32)
    else:
        vertices = np.zeros((0, 3), dtype=np.float32)
        faces = np.zeros((0, 3), dtype=np.uint32)
    return vertices, faces


def write_obj(path: Path, surfaces: list[BoxSurface]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertex_offset = 1
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# RT-Pose fitted environment scatterer geometry\n")
        for surface in surfaces:
            vertices, faces = box_vertices_faces(surface)
            handle.write(f"o {surface.surface_id}\n")
            for vertex in vertices:
                handle.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            for face in faces:
                a, b, c = face + vertex_offset
                handle.write(f"f {a} {b} {c}\n")
            vertex_offset += len(vertices)


def write_xml(path: Path, surfaces: list[BoxSurface]) -> None:
    materials = sorted({surface.material for surface in surfaces})
    lines = ['<scene version="2.1.0">']
    for material in materials:
        profile = MATERIAL_PROFILES.get(material, MATERIAL_PROFILES["furniture"])
        material_id = f"mat-env-{material}"
        lines.extend(
            [
                f'    <bsdf type="itu-radio-material" id="{material_id}">',
                f'        <string name="type" value="{profile["itu_type"]}"/>',
                f'        <float name="thickness" value="{profile["thickness_m"]:.6f}"/>',
                (
                    '        <float name="scattering_coefficient" '
                    f'value="{profile["scattering_coefficient"]:.6f}"/>'
                ),
                (
                    '        <float name="xpd_coefficient" '
                    f'value="{profile["xpd_coefficient"]:.6f}"/>'
                ),
                "    </bsdf>",
            ]
        )
    for surface in surfaces:
        cx, cy, cz = surface.center
        sx, sy, sz = np.maximum(surface.size * 0.5, 1e-3)
        lines.extend(
            [
                f'    <shape type="cube" id="{surface.surface_id}">',
                (
                    '        <transform name="to_world">'
                    f'<scale x="{sx:.6f}" y="{sy:.6f}" z="{sz:.6f}"/>'
                    f'<translate x="{cx:.6f}" y="{cy:.6f}" z="{cz:.6f}"/>'
                    "</transform>"
                ),
                f'        <ref name="bsdf" id="mat-env-{surface.material}"/>',
                "    </shape>",
            ]
        )
    lines.append("</scene>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_preview(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    surfaces: list[BoxSurface],
    roi: tuple[float, float, float, float, float, float],
) -> None:
    width, height = 1200, 560
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((15, 10), "RT-Pose fitted environment geometry", fill=(0, 0, 0))
    top_box = (30, 55, 570, 530)
    side_box = (640, 55, 1170, 530)
    x_min, x_max, y_min, y_max, z_min, z_max = roi

    def map_xy(x, y, box, xr, yr):
        x0, y0, x1, y1 = box
        return (
            x0 + (x - xr[0]) / (xr[1] - xr[0]) * (x1 - x0),
            y1 - (y - yr[0]) / (yr[1] - yr[0]) * (y1 - y0),
        )

    def draw_points(box, a, b, ar, br, title):
        x0, y0, x1, y1 = box
        draw.rectangle(box, outline=(0, 0, 0))
        draw.text((x0, y0 - 20), title, fill=(0, 0, 0))
        mask = (
            (points[:, a] >= ar[0])
            & (points[:, a] <= ar[1])
            & (points[:, b] >= br[0])
            & (points[:, b] <= br[1])
        )
        idx = np.where(mask)[0]
        stride = max(1, len(idx) // 12000)
        for i in idx[::stride]:
            px, py = map_xy(points[i, a], points[i, b], box, ar, br)
            draw.point((px, py), fill=tuple(int(v) for v in colors[i]))

    def draw_box_top(surface: BoxSurface):
        c = surface.center
        s = surface.size * 0.5
        x0, x1 = c[0] - s[0], c[0] + s[0]
        y0, y1 = c[1] - s[1], c[1] + s[1]
        p0 = map_xy(x0, y0, top_box, (x_min, x_max), (y_min, y_max))
        p1 = map_xy(x1, y1, top_box, (x_min, x_max), (y_min, y_max))
        color = (210, 40, 40) if surface.semantic == "wall" else (20, 120, 220)
        if surface.semantic == "floor":
            color = (60, 160, 60)
        draw.rectangle((p0[0], p1[1], p1[0], p0[1]), outline=color, width=2)
        draw.text((p0[0] + 3, p1[1] + 3), surface.surface_id, fill=color)

    def draw_box_side(surface: BoxSurface):
        c = surface.center
        s = surface.size * 0.5
        x0, x1 = c[0] - s[0], c[0] + s[0]
        z0, z1 = c[2] - s[2], c[2] + s[2]
        p0 = map_xy(x0, z0, side_box, (x_min, x_max), (z_min, z_max))
        p1 = map_xy(x1, z1, side_box, (x_min, x_max), (z_min, z_max))
        color = (210, 40, 40) if surface.semantic == "wall" else (20, 120, 220)
        if surface.semantic == "floor":
            color = (60, 160, 60)
        draw.rectangle((p0[0], p1[1], p1[0], p0[1]), outline=color, width=2)

    draw_points(top_box, 0, 1, (x_min, x_max), (y_min, y_max), "Top view: X/Y")
    draw_points(side_box, 0, 2, (x_min, x_max), (z_min, z_max), "Side view: X/Z")
    for surface in surfaces:
        draw_box_top(surface)
        draw_box_side(surface)
    image.save(path)


def surface_to_json(surface: BoxSurface) -> dict:
    profile = MATERIAL_PROFILES.get(surface.material, MATERIAL_PROFILES["furniture"])
    return {
        "id": surface.surface_id,
        "semantic": surface.semantic,
        "material": surface.material,
        "itu_material": profile["itu_type"],
        "material_profile": {
            "thickness_m": float(profile["thickness_m"]),
            "scattering_coefficient": float(profile["scattering_coefficient"]),
            "xpd_coefficient": float(profile["xpd_coefficient"]),
            "rationale": profile["rationale"],
        },
        "center": [float(v) for v in surface.center],
        "size": [float(v) for v in surface.size],
        "source_points": int(surface.source_points),
        "confidence": float(surface.confidence),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("background_ply", type=Path, help="Colored background PLY")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument(
        "--roi",
        type=lambda s: parse_float_list(s, 6, "--roi"),
        default=(0.0, 11.6, -6.0, 8.0, -1.6, 3.2),
        help="x_min,x_max,y_min,y_max,z_min,z_max in meters",
    )
    parser.add_argument("--plane-bin-width", type=float, default=0.05)
    parser.add_argument("--plane-threshold", type=float, default=0.08)
    parser.add_argument("--wall-threshold", type=float, default=0.10)
    parser.add_argument("--min-plane-bin-count", type=int, default=120)
    parser.add_argument("--min-wall-bin-count", type=int, default=140)
    parser.add_argument("--min-plane-inliers", type=int, default=400)
    parser.add_argument("--min-wall-inliers", type=int, default=300)
    parser.add_argument("--min-wall-height", type=float, default=1.0)
    parser.add_argument("--min-wall-span", type=float, default=1.0)
    parser.add_argument("--room-top-percentile", type=float, default=99.0)
    parser.add_argument("--room-surface-thickness", type=float, default=0.04)
    parser.add_argument(
        "--wall-mode",
        choices=("patch", "solid"),
        default="solid",
        help="Fit walls as occupied patches with openings, or one solid box per wall.",
    )
    parser.add_argument("--wall-patch-grid", type=float, default=0.18)
    parser.add_argument("--wall-patch-min-cell-points", type=int, default=3)
    parser.add_argument("--wall-patch-dilation", type=int, default=1)
    parser.add_argument("--wall-patch-min-above-floor", type=float, default=0.18)
    parser.add_argument("--min-wall-patch-points", type=int, default=70)
    parser.add_argument("--min-wall-patch-span", type=float, default=0.35)
    parser.add_argument("--min-wall-patch-height", type=float, default=0.35)
    parser.add_argument("--max-wall-patches-per-wall", type=int, default=8)
    parser.add_argument("--wall-exclusion", type=float, default=0.22)
    parser.add_argument("--furniture-grid-size", type=float, default=0.28)
    parser.add_argument("--min-furniture-points", type=int, default=45)
    parser.add_argument("--max-furniture-objects", type=int, default=12)
    parser.add_argument("--furniture-min-above-floor", type=float, default=0.18)
    parser.add_argument("--furniture-max-above-floor", type=float, default=1.75)
    parser.add_argument("--furniture-padding-xy", type=float, default=0.06)
    parser.add_argument("--furniture-padding-z", type=float, default=0.04)
    parser.add_argument("--furniture-bottom-clearance", type=float, default=0.03)
    parser.add_argument("--min-furniture-size-xy", type=float, default=0.22)
    parser.add_argument("--min-furniture-height", type=float, default=0.08)
    parser.add_argument("--no-furniture", action="store_true", help="Only fit floor/walls")
    parser.add_argument("--no-preview", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    background_ply = args.background_ply.resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = background_ply.parent / "environment_geometry"
    else:
        output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    points, colors = read_ascii_ply(background_ply)
    roi = (
        (points[:, 0] >= args.roi[0])
        & (points[:, 0] <= args.roi[1])
        & (points[:, 1] >= args.roi[2])
        & (points[:, 1] <= args.roi[3])
        & (points[:, 2] >= args.roi[4])
        & (points[:, 2] <= args.roi[5])
    )
    points = points[roi]
    colors = colors[roi]
    if len(points) == 0:
        raise SystemExit("No points remain inside --roi")

    floor = choose_floor(points, args)
    back_wall, left_wall, right_wall = choose_walls(points, args)
    surfaces = room_surfaces(points, floor, back_wall, left_wall, right_wall, args)
    if not args.no_furniture:
        surfaces.extend(furniture_surfaces(points, surfaces, floor, back_wall, left_wall, right_wall, args))

    vertices, faces = build_mesh(surfaces)

    json_path = output_dir / "environment_surfaces.json"
    obj_path = output_dir / "environment_surfaces.obj"
    xml_path = output_dir / "environment_scene.xml"
    npz_path = output_dir / "environment_mesh_sequence.npz"
    preview_path = output_dir / "environment_fit_preview.png"

    payload = {
        "source_background_ply": background_ply.name,
        "coordinate_frame": "RT-Pose/radar: x=range/depth, y=lateral, z=vertical",
        "material_profiles": MATERIAL_PROFILES,
        "fit": {
            "input_points": int(len(points)),
            "floor_z_m": float(floor.value),
            "back_wall_x_m": None if back_wall is None else float(back_wall.value),
            "left_wall_y_m": None if left_wall is None else float(left_wall.value),
            "right_wall_y_m": None if right_wall is None else float(right_wall.value),
            "wall_mode": args.wall_mode,
            "wall_patch_grid_m": float(args.wall_patch_grid),
            "wall_patch_dilation_cells": int(args.wall_patch_dilation),
        },
        "surfaces": [surface_to_json(surface) for surface in surfaces],
        "outputs": {
            "obj": obj_path.name,
            "xml": xml_path.name,
            "mesh_sequence_npz": npz_path.name,
            "preview": None if args.no_preview else preview_path.name,
        },
    }
    json_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_obj(obj_path, surfaces)
    write_xml(xml_path, surfaces)
    np.savez_compressed(
        npz_path,
        vertices=vertices[None, :, :],
        faces=faces,
        times=np.array([0.0], dtype=np.float64),
    )
    if not args.no_preview:
        write_preview(preview_path, points, colors, surfaces, args.roi)

    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
