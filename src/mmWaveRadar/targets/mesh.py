# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Dynamic mesh target support for mmWave radar sensing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from bundle_prepare.safe_npz import (
    DEFAULT_NPZ_LIMITS,
    preflight_npz,
    require_array,
)

if TYPE_CHECKING:
    from sionna.rt import SceneObject


from ..materials import human_skin_material


def face_winding_signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Return the signed volume implied by a closed triangular mesh winding."""

    verts = np.asarray(vertices, dtype=np.float64)
    raw_faces = np.asarray(faces)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("vertices must have shape [V, 3]")
    if not np.all(np.isfinite(verts)):
        raise ValueError("vertices must contain only finite values")
    if raw_faces.ndim != 2 or raw_faces.shape[1] != 3:
        raise ValueError("faces must have shape [F, 3]")
    if not np.issubdtype(raw_faces.dtype, np.integer):
        raise ValueError("faces must contain integer vertex indices")
    tri_faces = np.asarray(raw_faces, dtype=np.int64)
    if tri_faces.size == 0:
        return 0.0
    if np.any(tri_faces < 0) or np.any(tri_faces >= verts.shape[0]):
        raise ValueError("faces reference vertex indices outside vertices")

    tri = verts[tri_faces]
    origin = np.mean(verts, axis=0)
    shifted = tri - origin[None, None, :]
    return float(
        np.sum(
            np.einsum(
                "ij,ij->i",
                shifted[:, 0],
                np.cross(shifted[:, 1], shifted[:, 2]),
            )
        )
        / 6.0
    )


def ensure_outward_face_winding(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    volume_epsilon: float = 1e-12,
) -> np.ndarray:
    """Return faces wound so normals point outward for a closed mesh.

    Degenerate or open meshes with near-zero signed volume are left unchanged.
    """

    epsilon = float(volume_epsilon)
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("volume_epsilon must be finite and non-negative")

    verts = np.asarray(vertices)
    if verts.ndim == 3:
        if verts.shape[0] == 0:
            raise ValueError("vertices sequence must contain at least one frame")
        verts = verts[0]
    raw_faces = np.asarray(faces)
    signed_volume = face_winding_signed_volume(verts, raw_faces)
    tri_faces = np.asarray(raw_faces, dtype=np.uint32)
    if signed_volume < -epsilon:
        return tri_faces[:, [0, 2, 1]].copy()
    return tri_faces.copy()


@dataclass(frozen=True)
class MeshSequence:
    r"""
    Fixed-topology mesh sequence.

    :param vertices: Vertex positions with shape ``[num_times, num_vertices, 3]``
    :param faces: Triangle indices with shape ``[num_faces, 3]``
    :param times: Mesh sample times [s], strictly increasing
    """

    vertices: np.ndarray
    faces: np.ndarray
    times: np.ndarray

    def __post_init__(self):
        vertices = np.asarray(self.vertices, dtype=np.float32)
        raw_faces = np.asarray(self.faces)
        times = np.asarray(self.times, dtype=float)
        if vertices.ndim != 3 or vertices.shape[2] != 3:
            raise ValueError("vertices must have shape [T, V, 3]")
        if vertices.shape[1] == 0:
            raise ValueError("mesh sequence must contain at least one vertex")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("vertices must contain only finite values")
        if raw_faces.ndim != 2 or raw_faces.shape[1] != 3:
            raise ValueError("faces must have shape [F, 3]")
        if not np.issubdtype(raw_faces.dtype, np.integer):
            raise ValueError("faces must contain integer vertex indices")
        faces = np.asarray(raw_faces, dtype=np.int64)
        if times.ndim != 1 or times.shape[0] != vertices.shape[0]:
            raise ValueError("times must have shape [T]")
        if times.shape[0] == 0:
            raise ValueError("mesh sequence must contain at least one sample")
        if not np.all(np.isfinite(times)):
            raise ValueError("times must contain only finite values")
        if times.shape[0] > 1 and np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be strictly increasing")
        if faces.size > 0 and (
            np.any(faces < 0) or np.any(faces >= vertices.shape[1])
        ):
            raise ValueError("faces reference vertex indices outside vertices")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces.astype(np.uint32, copy=False))
        object.__setattr__(self, "times", times)

    @classmethod
    def from_npz(cls, filename: str) -> "MeshSequence":
        """Loads a sequence from an NPZ file with `vertices`, `faces`, `times`."""
        arrays = preflight_npz(filename, limits=DEFAULT_NPZ_LIMITS)
        vertices = require_array(
            arrays,
            "vertices",
            allowed_ndim=(3,),
            dtype_kinds=frozenset({"i", "u", "f"}),
            trailing_shape=(3,),
            max_elements=120_000_000,
        )
        require_array(
            arrays,
            "faces",
            allowed_ndim=(2,),
            dtype_kinds=frozenset({"i", "u"}),
            trailing_shape=(3,),
            max_elements=60_000_000,
        )
        times = require_array(
            arrays,
            "times",
            allowed_ndim=(1,),
            dtype_kinds=frozenset({"i", "u", "f"}),
            max_elements=10_000_000,
        )
        if times.shape != (vertices.shape[0],):
            raise ValueError("mesh sequence times must match its frame count")
        with np.load(filename, allow_pickle=False) as data:
            return cls(
                vertices=data["vertices"].copy(),
                faces=data["faces"].copy(),
                times=data["times"].copy(),
            )

    @property
    def vertex_count(self) -> int:
        """Number of vertices."""
        return int(self.vertices.shape[1])

    @property
    def face_count(self) -> int:
        """Number of triangular faces."""
        return int(self.faces.shape[0])

    def vertices_at(self, time: float) -> np.ndarray:
        """Linearly interpolates vertices at ``time`` [s]."""
        t = float(time)
        if not np.isfinite(t):
            raise ValueError("time must be finite")
        if self.times.shape[0] == 1 or t <= self.times[0]:
            return self.vertices[0].copy()
        if t >= self.times[-1]:
            return self.vertices[-1].copy()
        i1 = int(np.searchsorted(self.times, t, side="right"))
        i0 = i1 - 1
        w = (t - self.times[i0]) / (self.times[i1] - self.times[i0])
        return ((1.0 - w) * self.vertices[i0] + w * self.vertices[i1]).astype(
            np.float32, copy=False)

    def max_vertex_displacement(self, time: float,
                                reference_vertices: np.ndarray) -> float:
        """Maximum vertex displacement relative to ``reference_vertices``."""
        v = self.vertices_at(time)
        ref = np.asarray(reference_vertices, dtype=np.float32)
        if ref.shape != v.shape:
            raise ValueError("reference_vertices shape mismatch")
        return float(np.max(np.linalg.norm(v - ref, axis=1)))


@dataclass
class MeshTarget:
    r"""Radar sensing target represented by a fixed-topology mesh sequence."""

    name: str
    mesh_sequence: MeshSequence
    material: object | None = None
    scene_object: SceneObject | None = None
    reference_vertices: np.ndarray | None = None

    def ensure_scene_object(self) -> SceneObject:
        """Creates the backing Sionna scene object if needed."""
        if self.scene_object is not None:
            return self.scene_object

        material = self.material
        if material is None:
            raise ValueError(
                "MeshTarget.material must be set explicitly. For the tutorial "
                "human proxy, pass human_skin_material(...).")

        import mitsuba as mi  # pylint: disable=import-outside-toplevel
        from sionna.rt import SceneObject

        start_time = float(np.asarray(self.mesh_sequence.times, dtype=float)[0])
        vertices = self.mesh_sequence.vertices_at(start_time)
        faces = self.mesh_sequence.faces
        mesh = mi.Mesh(name=self.name,
                       vertex_count=vertices.shape[0],
                       face_count=faces.shape[0],
                       has_vertex_normals=False,
                       has_vertex_texcoords=False)
        params = mi.traverse(mesh)
        params["vertex_positions"] = mi.Float(vertices.reshape(-1))
        params["faces"] = mi.UInt(faces.reshape(-1))
        params.update()
        mesh.set_bsdf(material)
        self.scene_object = SceneObject(mi_mesh=mesh, name=self.name,
                                        radio_material=material)
        self.reference_vertices = vertices.copy()
        return self.scene_object

    def add_to_scene(self, scene) -> SceneObject:
        """Adds this target's mesh to ``scene`` if it is not already present."""
        obj = self.ensure_scene_object()
        if obj.scene is None:
            scene.edit(add=[obj])
        elif obj.scene is not scene:
            raise ValueError("Mesh target already belongs to another scene")
        return obj

    def update_to_time(
        self,
        time: float,
        *,
        vertices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Updates the scene object, optionally reusing precomputed vertices."""

        obj = self.ensure_scene_object()
        current_vertices = (
            self.mesh_sequence.vertices_at(time)
            if vertices is None
            else np.asarray(vertices, dtype=np.float32)
        )
        _update_scene_object_vertices(obj, current_vertices)
        return current_vertices


def _update_scene_object_vertices(obj: "SceneObject", vertices: np.ndarray) -> None:
    """Updates a Sionna scene object's fixed-topology mesh vertices."""

    vertices = np.asarray(vertices, dtype=np.float32)
    updater = getattr(obj, "update_vertex_positions", None)
    if updater is not None:
        updater(vertices)
        return

    import mitsuba as mi  # pylint: disable=import-outside-toplevel

    mesh = getattr(obj, "mi_mesh", None)
    if mesh is None:
        raise AttributeError(
            "SceneObject has neither update_vertex_positions nor mi_mesh"
        )
    if vertices.shape != (int(mesh.vertex_count()), 3):
        raise ValueError("vertices shape mismatch for scene object mesh")

    scene = getattr(obj, "scene", None)
    if scene is not None and hasattr(scene, "mi_scene_params"):
        key = f"{mesh.id()}.vertex_positions"
        params = scene.mi_scene_params
        if key in params:
            params[key] = mi.Float(vertices.reshape(-1))
            params.update()
            geometry_updated = getattr(scene, "scene_geometry_updated", None)
            if geometry_updated is not None:
                geometry_updated()
            return

    mi_scene = getattr(scene, "mi_scene", None) if scene is not None else None
    if mi_scene is not None:
        for key in (f"{mesh.id()}.vertex_positions", f"{obj.name}.vertex_positions"):
            params = mi.traverse(mi_scene)
            if key in params:
                params[key] = mi.Float(vertices.reshape(-1))
                params.update()
                geometry_updated = getattr(scene, "scene_geometry_updated", None)
                if geometry_updated is not None:
                    geometry_updated()
                return

    params = mi.traverse(mesh)
    params["vertex_positions"] = mi.Float(vertices.reshape(-1))
    params.update()
