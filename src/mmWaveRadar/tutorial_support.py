# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Shared helpers for the mmWave radar tutorial notebooks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


BEDROOM_SCENE_XML = """
<scene version="2.1.0">
    <bsdf type="itu-radio-material" id="mat-wall">
        <string name="type" value="plasterboard"/>
        <float name="scattering_coefficient" value="0.2"/>
    </bsdf>
    <bsdf type="itu-radio-material" id="mat-reflective-wall">
        <string name="type" value="metal"/>
        <float name="scattering_coefficient" value="0.2"/>
    </bsdf>
    <bsdf type="itu-radio-material" id="mat-wood"><string name="type" value="wood"/></bsdf>
    <bsdf type="itu-radio-material" id="mat-fabric"><string name="type" value="chipboard"/></bsdf>

    <shape type="cube" id="floor">
        <transform name="to_world"><scale x="2.95" y="4.35" z="0.04"/><translate x="2.5" y="0.75" z="-0.04"/></transform>
        <ref name="bsdf" id="mat-wood"/>
    </shape>
    <shape type="cube" id="back_wall">
        <transform name="to_world"><scale x="0.05" y="4.35" z="1.5"/><translate x="5.45" y="0.75" z="1.5"/></transform>
        <ref name="bsdf" id="mat-wall"/>
    </shape>
    <shape type="cube" id="left_wall">
        <transform name="to_world"><scale x="2.95" y="0.05" z="1.5"/><translate x="2.5" y="-3.6" z="1.5"/></transform>
        <ref name="bsdf" id="mat-wall"/>
    </shape>
    <shape type="cube" id="right_wall">
        <transform name="to_world"><scale x="2.95" y="0.05" z="1.5"/><translate x="2.5" y="5.1" z="1.5"/></transform>
        <ref name="bsdf" id="mat-reflective-wall"/>
    </shape>
    <shape type="cube" id="bed">
        <transform name="to_world"><scale x="0.55" y="0.9" z="0.18"/><translate x="4.85" y="-2.65" z="0.28"/></transform>
        <ref name="bsdf" id="mat-fabric"/>
    </shape>
    <shape type="cube" id="nightstand">
        <transform name="to_world"><scale x="0.25" y="0.25" z="0.25"/><translate x="4.95" y="-1.3" z="0.25"/></transform>
        <ref name="bsdf" id="mat-wood"/>
    </shape>
    <shape type="cube" id="wardrobe">
        <transform name="to_world"><scale x="0.35" y="0.9" z="1.0"/><translate x="4.85" y="3.8" z="1.0"/></transform>
        <ref name="bsdf" id="mat-wood"/>
    </shape>
</scene>
""".strip()


BEDROOM_SCENE_BOXES = (
    {"id": "floor", "scale": (2.95, 4.35, 0.04), "translate": (2.5, 0.75, -0.04)},
    {"id": "back_wall", "scale": (0.05, 4.35, 1.5), "translate": (5.45, 0.75, 1.5)},
    {"id": "left_wall", "scale": (2.95, 0.05, 1.5), "translate": (2.5, -3.6, 1.5)},
    {"id": "right_wall", "scale": (2.95, 0.05, 1.5), "translate": (2.5, 5.1, 1.5)},
    {"id": "bed", "scale": (0.55, 0.9, 0.18), "translate": (4.85, -2.65, 0.28)},
    {"id": "nightstand", "scale": (0.25, 0.25, 0.25), "translate": (4.95, -1.3, 0.25)},
    {"id": "wardrobe", "scale": (0.35, 0.9, 1.0), "translate": (4.85, 3.8, 1.0)},
)


def orientation_toward(target_point, radar_position):
    """Returns a Sionna orientation whose local +x points at ``target_point``."""

    direction = np.asarray(target_point, dtype=float) - np.asarray(
        radar_position, dtype=float)
    direction /= np.linalg.norm(direction)
    theta = np.arccos(np.clip(direction[2], -1.0, 1.0))
    phi = np.arctan2(direction[1], direction[0])
    return (float(phi), float(theta - 0.5 * np.pi), 0.0)


def _mean_marker(markers, names):
    """Averages available AMASS markers by name, or returns ``None``."""

    points = [markers[name] for name in names if name in markers]
    if not points:
        return None
    return np.mean(np.asarray(points, dtype=float), axis=0)


def amass_marker_dict(amass_data):
    """Returns first-frame AMASS marker positions keyed by marker label."""

    if (
        amass_data is None
        or "marker_data" not in amass_data
        or "marker_labels" not in amass_data
    ):
        return {}
    labels = [
        label.decode("utf-8") if isinstance(label, bytes) else str(label)
        for label in amass_data["marker_labels"]
    ]
    points = np.asarray(amass_data["marker_data"])[0]
    return dict(zip(labels, points))


def smpl_chest_front(vertices, amass_data):
    """Estimates a chest-front point and outward front direction."""

    markers = amass_marker_dict(amass_data)
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
    chest_front_point = np.array([
        chest_front_xy[0],
        chest_front_xy[1],
        sternum[2],
    ], dtype=float)
    front_dir = np.array([front_dir_xy[0], front_dir_xy[1], 0.0], dtype=float)
    return chest_front_point, front_dir


def radar_pose_in_front_of_chest(
    mesh_sequence,
    *,
    clearance_m: float = 0.50,
    amass_data=None,
):
    """Places a radar in front of the first-frame chest estimate."""

    vertices = mesh_sequence.vertices_at(float(mesh_sequence.times[0]))
    chest = smpl_chest_front(vertices, amass_data)
    if chest is None:
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        chest_front_point = np.array([
            mins[0],
            0.5 * (mins[1] + maxs[1]),
            0.65 * maxs[2],
        ], dtype=float)
        front_dir = np.array([-1.0, 0.0, 0.0], dtype=float)
    else:
        chest_front_point, front_dir = chest
    radar_position = chest_front_point + float(clearance_m) * front_dir
    radar_orientation = orientation_toward(chest_front_point, radar_position)
    return radar_position, radar_orientation, chest_front_point, front_dir


def count_radar_frames_in_mesh_sequence(mesh_sequence, fmcw) -> int:
    """Returns how many radar frames fit in a mesh sequence time span."""

    times = np.asarray(mesh_sequence.times, dtype=float)
    if times.size == 0:
        raise ValueError("mesh sequence must contain at least one time sample")
    sequence_span_s = float(times[-1] - times[0])
    last_chirp_offset_s = fmcw.tx_chirp_time(
        0,
        fmcw.num_chirps_per_frame - 1,
        fmcw.num_tx - 1,
    )
    if sequence_span_s <= last_chirp_offset_s:
        return 1
    return max(
        1,
        int(np.floor((sequence_span_s - last_chirp_offset_s) / fmcw.frame_period)) + 1,
    )


@dataclass(frozen=True)
class RigidTransformMeshSequence:
    """Rigidly transformed view of a mesh sequence."""

    base_sequence: object
    rotation: np.ndarray
    translation: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )

    def __post_init__(self):
        rotation = np.asarray(self.rotation, dtype=np.float32)
        translation = np.asarray(self.translation, dtype=np.float32)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError("rotation must be a finite 3x3 matrix")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
            raise ValueError("rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise ValueError("rotation must be right-handed")
        if translation.shape != (3,) or not np.all(np.isfinite(translation)):
            raise ValueError("translation must be a finite 3-vector")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    @property
    def times(self):
        """Sample times from the base sequence."""

        return self.base_sequence.times

    @property
    def faces(self):
        """Triangle indices from the base sequence."""

        return self.base_sequence.faces

    @property
    def vertex_count(self) -> int:
        """Number of vertices in one mesh."""

        return int(self.base_sequence.vertex_count)

    @property
    def face_count(self) -> int:
        """Number of triangular faces."""

        return int(self.base_sequence.face_count)

    @property
    def vertices(self) -> np.ndarray:
        """Transformed vertices at the base sample times."""

        return np.stack(
            [self.vertices_at(float(time_s)) for time_s in self.times],
            axis=0,
        )

    def vertices_at(self, time: float) -> np.ndarray:
        """Evaluates and transforms vertices at ``time`` seconds."""

        vertices = np.asarray(self.base_sequence.vertices_at(time))
        return (
            vertices @ self.rotation.T + self.translation[None, :]
        ).astype(np.float32, copy=False)

    def max_vertex_displacement(
        self,
        time: float,
        reference_vertices: np.ndarray,
    ) -> float:
        """Maximum displacement in the transformed coordinate frame."""

        vertices = self.vertices_at(time)
        reference = np.asarray(reference_vertices, dtype=np.float32)
        if reference.shape != vertices.shape:
            raise ValueError("reference_vertices shape mismatch")
        return float(np.max(np.linalg.norm(vertices - reference, axis=1)))


def rigid_transform_mesh_sequence(
    mesh_sequence,
    *,
    rotation=None,
    translation=(0.0, 0.0, 0.0),
) -> RigidTransformMeshSequence:
    """Returns a rigidly rotated and translated mesh-sequence view."""

    return RigidTransformMeshSequence(
        base_sequence=mesh_sequence,
        rotation=np.eye(3, dtype=np.float32) if rotation is None else rotation,
        translation=translation,
    )


@dataclass(frozen=True)
class MeshSequenceTimeWindow:
    """Time-rebased view of a mesh sequence.

    The wrapper is useful for AMASS tutorial motions whose raw sequence starts
    with calibration poses or contains a much longer global trajectory than a
    short radar demo needs.
    """

    base_sequence: object
    start_time_s: float
    duration_s: float | None = None
    times: np.ndarray = field(init=False)
    end_time_s: float = field(init=False)

    def __post_init__(self):
        base_times = np.asarray(self.base_sequence.times, dtype=float)
        if base_times.ndim != 1 or base_times.size == 0:
            raise ValueError("base sequence times must be a non-empty 1D array")
        if base_times.size > 1 and np.any(np.diff(base_times) <= 0.0):
            raise ValueError("base sequence times must be strictly increasing")

        start = float(self.start_time_s)
        base_start = float(base_times[0])
        base_end = float(base_times[-1])
        if start < base_start or start >= base_end:
            raise ValueError(
                f"start_time_s must be in [{base_start}, {base_end})")

        if self.duration_s is None:
            end = base_end
        else:
            duration = float(self.duration_s)
            if duration <= 0.0:
                raise ValueError("duration_s must be positive")
            end = min(start + duration, base_end)
        if end <= start:
            raise ValueError("time window must contain a positive span")

        window_base_times = base_times[
            (base_times >= start) & (base_times <= end)
        ]
        if (
            window_base_times.size == 0
            or not np.isclose(window_base_times[0], start)
        ):
            window_base_times = np.concatenate(([start], window_base_times))
        if not np.isclose(window_base_times[-1], end):
            window_base_times = np.concatenate((window_base_times, [end]))
        rebased_times = np.unique(np.round(window_base_times - start, 12))

        object.__setattr__(self, "start_time_s", start)
        object.__setattr__(self, "end_time_s", end)
        object.__setattr__(self, "times", rebased_times.astype(float))

    @property
    def faces(self):
        """Triangle indices from the base sequence."""

        return self.base_sequence.faces

    @property
    def vertex_count(self) -> int:
        """Number of vertices in one evaluated mesh."""

        count = getattr(self.base_sequence, "vertex_count", None)
        if count is not None:
            return int(count)
        return int(self.vertices_at(0.0).shape[0])

    @property
    def face_count(self) -> int:
        """Number of triangular faces."""

        count = getattr(self.base_sequence, "face_count", None)
        if count is not None:
            return int(count)
        return int(np.asarray(self.faces).shape[0])

    @property
    def vertices(self) -> np.ndarray:
        """Vertices at the rebased sample times."""

        return np.stack([self.vertices_at(float(t)) for t in self.times], axis=0)

    def vertices_at(self, time: float) -> np.ndarray:
        """Evaluates vertices at rebased ``time`` seconds."""

        base_time = self.start_time_s + float(time)
        base_time = min(max(base_time, self.start_time_s), self.end_time_s)
        return self.base_sequence.vertices_at(base_time)

    def max_vertex_displacement(self, time: float,
                                reference_vertices: np.ndarray) -> float:
        """Maximum vertex displacement relative to ``reference_vertices``."""

        vertices = self.vertices_at(time)
        ref = np.asarray(reference_vertices, dtype=np.float32)
        if ref.shape != vertices.shape:
            raise ValueError("reference_vertices shape mismatch")
        return float(np.max(np.linalg.norm(vertices - ref, axis=1)))


def time_window_mesh_sequence(
    mesh_sequence,
    *,
    start_time_s: float,
    duration_s: float | None = None,
) -> MeshSequenceTimeWindow:
    """Returns a time-rebased window of ``mesh_sequence``."""

    return MeshSequenceTimeWindow(
        base_sequence=mesh_sequence,
        start_time_s=start_time_s,
        duration_s=duration_s,
    )


def plot_mesh_projection(
    ax,
    sequence,
    time_s,
    radar_position,
    title,
    *,
    axes=(0, 1),
    chest_front_point=None,
):
    """Plots an opaque, depth-shaded mesh projection and radar pose."""

    from matplotlib.collections import PolyCollection  # pylint: disable=import-outside-toplevel

    vertices = sequence.vertices_at(time_s)
    a0, a1 = axes
    labels = ("x", "y", "z")
    hidden_axis = next(axis for axis in range(3) if axis not in (a0, a1))
    triangles = vertices[np.asarray(sequence.faces, dtype=np.int64)]
    depths = np.mean(triangles[:, :, hidden_axis], axis=1)
    order = np.argsort(depths)

    depth_span = float(np.ptp(depths))
    normalized_depth = (
        np.zeros_like(depths)
        if depth_span <= 1e-12
        else (depths - np.min(depths)) / depth_span
    )
    base_color = np.array([0.80, 0.42, 0.35])
    shade = 0.65 + 0.35 * normalized_depth
    face_colors = np.column_stack((
        np.clip(base_color[None, :] * shade[:, None], 0.0, 1.0),
        np.ones(depths.shape[0]),
    ))
    surface = PolyCollection(
        triangles[order][:, :, [a0, a1]],
        facecolors=face_colors[order],
        edgecolors="none",
        linewidths=0.0,
        zorder=1,
    )
    ax.add_collection(surface)
    ax.scatter(
        [radar_position[a0]],
        [radar_position[a1]],
        color="#1f77b4",
        s=70,
        marker="^",
        label="radar",
        zorder=3,
    )
    plot_points = [
        vertices[:, [a0, a1]],
        np.asarray(radar_position, dtype=float)[[a0, a1]][None, :],
    ]
    if chest_front_point is not None:
        chest_front_point = np.asarray(chest_front_point, dtype=float)
        ax.scatter(
            [chest_front_point[a0]],
            [chest_front_point[a1]],
            color="#2ca02c",
            s=50,
            marker="x",
            label="chest front",
            zorder=3,
        )
        plot_points.append(chest_front_point[[a0, a1]][None, :])
    plot_points = np.vstack(plot_points)
    mins = plot_points.min(axis=0)
    maxs = plot_points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins) + 0.2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{labels[a0]} [m]")
    ax.set_ylabel(f"{labels[a1]} [m]")
    ax.set_title(title)
    # A fixed location avoids Matplotlib's expensive overlap search across every
    # projected triangle. With a full SMPL mesh, ``loc="best"`` can dominate
    # the rendering time for an otherwise lightweight tutorial preview.
    ax.legend(loc="upper left")
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.25)


def bedroom_scene_boxes() -> tuple[dict, ...]:
    """Returns axis-aligned boxes for the tutorial bedroom scene."""

    return BEDROOM_SCENE_BOXES


def plot_bedroom_scene_projection(
    ax,
    *,
    axes=(0, 1),
    adjust_limits: bool = True,
):
    """Overlays the tutorial bedroom scene as projected box outlines."""

    from matplotlib.patches import Rectangle  # pylint: disable=import-outside-toplevel

    a0, a1 = axes
    scene_points = []
    for index, box in enumerate(BEDROOM_SCENE_BOXES):
        scale = np.asarray(box["scale"], dtype=float)
        translate = np.asarray(box["translate"], dtype=float)
        mins = translate - scale
        maxs = translate + scale
        xy = mins[[a0, a1]]
        width, height = maxs[[a0, a1]] - xy
        scene_points.append(np.asarray([xy, xy + [width, height]], dtype=float))
        is_structure = box["id"] in {"floor", "back_wall", "left_wall", "right_wall"}
        color = "#4b5563" if is_structure else "#8b5e34"
        alpha = 0.22 if is_structure else 0.32
        ax.add_patch(
            Rectangle(
                xy,
                width,
                height,
                facecolor=color,
                edgecolor=color,
                alpha=alpha,
                linewidth=1.0,
                label="bedroom scene" if index == 0 else None,
                zorder=0,
            )
        )

    if adjust_limits and scene_points:
        scene_points_array = np.vstack(scene_points)
        current_points = np.asarray([
            [ax.get_xlim()[0], ax.get_ylim()[0]],
            [ax.get_xlim()[1], ax.get_ylim()[1]],
        ])
        plot_points = np.vstack([scene_points_array, current_points])
        mins = plot_points.min(axis=0)
        maxs = plot_points.max(axis=0)
        center = 0.5 * (mins + maxs)
        radius = 0.5 * np.max(maxs - mins) + 0.2
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_aspect("equal", adjustable="box")

    # Keep this fixed for the same reason as ``plot_mesh_projection``: the axes
    # may already contain tens of thousands of projected mesh triangles.
    ax.legend(loc="upper left")
    return tuple(scene_points)


def bedroom_scene_xml() -> str:
    """Returns the Mitsuba XML string used by the tutorial bedroom scene."""
    return BEDROOM_SCENE_XML


def load_bedroom_scene(*, merge_shapes: bool = False):
    """Creates the tutorial bedroom scene."""
    import sionna.rt as rt  # pylint: disable=import-outside-toplevel

    return rt.load_scene_from_string(BEDROOM_SCENE_XML, merge_shapes=merge_shapes)


def make_expanding_cylinder_sequence(
    *,
    num_times: int = 96,
    num_theta: int = 16,
    num_z: int = 10,
    duration_s: float = 0.6,
    center: tuple[float, float, float] = (1.55, 0.0, 1.15),
    base_radius: float = 0.18,
    expansion_amplitude: float = 0.035,
    height: float = 1.1,
    oscillation_hz: float = 1.0,
) -> MeshSequence:
    r"""
    Builds a fixed-topology expanding cylinder mesh sequence.

    The mesh is centered in the bedroom scene and expands/contracts in radius
    over time, mirroring the simple deforming-cylinder tutorial used in mmSim.
    """

    if num_times < 2:
        raise ValueError("num_times must be >= 2")
    if num_theta < 3:
        raise ValueError("num_theta must be >= 3")
    if num_z < 2:
        raise ValueError("num_z must be >= 2")
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if base_radius <= 0.0:
        raise ValueError("base_radius must be positive")
    if height <= 0.0:
        raise ValueError("height must be positive")

    from .targets import MeshSequence  # pylint: disable=import-outside-toplevel

    theta = np.linspace(0.0, 2.0 * np.pi, num_theta, endpoint=False)
    z_axis = np.linspace(-0.5 * height, 0.5 * height, num_z)
    center = np.asarray(center, dtype=np.float32)

    faces = []
    for z_idx in range(num_z - 1):
        for theta_idx in range(num_theta):
            a = z_idx * num_theta + theta_idx
            b = z_idx * num_theta + (theta_idx + 1) % num_theta
            c = (z_idx + 1) * num_theta + theta_idx
            d = (z_idx + 1) * num_theta + (theta_idx + 1) % num_theta
            faces.append([a, c, b])
            faces.append([b, c, d])
    faces = np.asarray(faces, dtype=np.uint32)

    times = np.linspace(0.0, duration_s, num_times, dtype=np.float32)
    vertices = []
    cos_theta = np.cos(theta).astype(np.float32)
    sin_theta = np.sin(theta).astype(np.float32)

    for time_s in times:
        radius = base_radius + expansion_amplitude * np.sin(
            2.0 * np.pi * oscillation_hz * float(time_s))
        frame_vertices = []
        for z_value in z_axis:
            ring = np.stack([
                center[0] + radius * cos_theta,
                center[1] + radius * sin_theta,
                np.full_like(cos_theta, center[2] + z_value),
            ], axis=1)
            frame_vertices.append(ring)
        vertices.append(np.concatenate(frame_vertices, axis=0))

    return MeshSequence(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=faces,
        times=times)


def make_expanding_chest_sequence(
    *,
    num_times: int = 96,
    num_theta: int = 32,
    num_z: int = 12,
    duration_s: float = 0.6,
    center: tuple[float, float, float] = (1.55, 0.0, 1.10),
    half_width: float = 0.36,
    height: float = 0.48,
    depth: float = 0.18,
    expansion_amplitude: float = 0.035,
    oscillation_hz: float = 1.0,
) -> MeshSequence:
    r"""
    Builds a fixed-topology oval chest-like mesh sequence.

    The target is a watertight torso segment with an oval top-down ``x-y``
    cross-section, extruded over chest height in ``z``. The wide axis is along
    ``y`` and the shallow axis is along ``x``, so the smaller-``x`` front side
    faces the tutorial radar near the origin looking along ``+x``. Breathing
    motion moves the front side toward the radar and slightly expands the
    horizontal oval.
    """

    if num_times < 2:
        raise ValueError("num_times must be >= 2")
    if num_theta < 8:
        raise ValueError("num_theta must be >= 8")
    if num_z < 2:
        raise ValueError("num_z must be >= 2")
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if half_width <= 0.0:
        raise ValueError("half_width must be positive")
    if height <= 0.0:
        raise ValueError("height must be positive")
    if depth <= 0.0:
        raise ValueError("depth must be positive")

    from .targets import MeshSequence  # pylint: disable=import-outside-toplevel

    theta = np.linspace(0.0, 2.0 * np.pi, num_theta, endpoint=False)
    cos_theta = np.cos(theta).astype(np.float32)
    sin_theta = np.sin(theta).astype(np.float32)
    z_axis = np.linspace(-0.5 * height, 0.5 * height, num_z)
    center = np.asarray(center, dtype=np.float32)

    bottom_center = num_z * num_theta
    top_center = bottom_center + 1

    def ring_idx(z_idx, theta_idx):
        """Returns the wrapped vertex index for one angular ring sample."""

        return z_idx * num_theta + theta_idx % num_theta

    faces = []
    for z_idx in range(num_z - 1):
        for theta_idx in range(num_theta):
            a = ring_idx(z_idx, theta_idx)
            b = ring_idx(z_idx, theta_idx + 1)
            c = ring_idx(z_idx + 1, theta_idx)
            d = ring_idx(z_idx + 1, theta_idx + 1)
            faces.append([a, c, b])
            faces.append([b, c, d])

    # Bottom cap. Winding gives outward normals toward -z.
    for theta_idx in range(num_theta):
        faces.append([bottom_center,
                      ring_idx(0, theta_idx),
                      ring_idx(0, theta_idx + 1)])

    # Top cap. Opposite winding gives outward normals toward +z.
    for theta_idx in range(num_theta):
        faces.append([top_center,
                      ring_idx(num_z - 1, theta_idx + 1),
                      ring_idx(num_z - 1, theta_idx)])
    faces = np.asarray(faces, dtype=np.uint32)

    times = np.linspace(0.0, duration_s, num_times, dtype=np.float32)
    vertices = []
    for time_s in times:
        breath = expansion_amplitude * np.sin(
            2.0 * np.pi * oscillation_hz * float(time_s))
        current_depth = depth + breath
        width = half_width + 0.15 * breath
        x_center = center[0] - 0.5 * breath

        frame_vertices = []
        for z_value in z_axis:
            frame_vertices.append(np.stack([
                x_center + 0.5 * current_depth * cos_theta,
                center[1] + width * sin_theta,
                np.full_like(cos_theta, center[2] + z_value),
            ], axis=1))
        frame_vertices = np.concatenate(frame_vertices, axis=0)
        cap_centers = np.array([
            [x_center, center[1], center[2] + z_axis[0]],
            [x_center, center[1], center[2] + z_axis[-1]],
        ], dtype=np.float32)
        vertices.append(np.concatenate([frame_vertices, cap_centers], axis=0))

    return MeshSequence(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=faces,
        times=times)


__all__ = [
    "BEDROOM_SCENE_BOXES",
    "BEDROOM_SCENE_XML",
    "MeshSequenceTimeWindow",
    "RigidTransformMeshSequence",
    "amass_marker_dict",
    "bedroom_scene_boxes",
    "bedroom_scene_xml",
    "load_bedroom_scene",
    "make_expanding_chest_sequence",
    "make_expanding_cylinder_sequence",
    "count_radar_frames_in_mesh_sequence",
    "orientation_toward",
    "plot_bedroom_scene_projection",
    "plot_mesh_projection",
    "radar_pose_in_front_of_chest",
    "rigid_transform_mesh_sequence",
    "smpl_chest_front",
    "time_window_mesh_sequence",
]
