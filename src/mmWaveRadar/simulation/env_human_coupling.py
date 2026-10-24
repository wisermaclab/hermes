# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Single-bounce static-environment/human PO coupling helpers."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import TYPE_CHECKING

import mitsuba as mi
import numpy as np
from scipy.constants import c, pi
from scipy.spatial import cKDTree

from sionna.rt.constants import InteractionType, INVALID_PRIMITIVE, INVALID_SHAPE

from ..radar.hardware import resolve_virtual_channel_pairs
from ..radar.ti import SimpleCosinePattern
from .compute import ComputeBackend, ComputePrecision, validate_compute_backend
from .cadence import refresh_period_chirps
from .physical_optics import (
    HumanVisibilityScene,
    PhysicalOpticsMaterial,
    PolarizationCallback,
    _aggregate_parent_face_far_field_analytic_contributions,
    _aggregate_parent_face_quadrature_contributions,
    _coerce_material,
    _local_incidence_bases,
    _normalize_rows,
    _numpy_dtypes_for_precision,
    _phase_center_delays_for_centroids,
    _transverse_polarizations,
    face_geometry,
    parent_face_adaptive_analytic_geometry,
    fresnel_coefficients,
    parent_face_far_field_analytic_geometry,
    parent_face_quadrature_geometry,
    vector_facet_face_contributions,
)
from ..geometry import rotation_matrix
from .types import POConfig
from .po_wideband import PODelayResponse
from .coupling_incremental import CouplingIncrementalState

if TYPE_CHECKING:
    from ..radar import RadarSensor


@dataclass(frozen=True)
class EnvironmentReflectorBank:
    """Reflector planes, seed triangles, and finite coplanar surface support."""

    vertices: np.ndarray
    centroids: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    materials: tuple[PhysicalOpticsMaterial, ...]
    object_names: tuple[str, ...]
    face_indices: np.ndarray
    path_indices: np.ndarray | None = None
    # Static-RT diagnostic only: coupled fields use their own TE/TM geometry.
    reflection_gains: np.ndarray | None = None
    specular_amplitude_scales: np.ndarray | None = None
    # One union of coplanar triangles per image plane. Legacy/manual banks
    # without this field's data retain their seed triangle as finite support.
    support_triangles: tuple[np.ndarray, ...] | None = None
    estimated_powers: np.ndarray | None = None
    relative_power_db: np.ndarray | None = None
    selection_metadata: dict | None = None


@dataclass(frozen=True)
class EnvHumanCouplingResult:
    """Coupled human/environment single-reflection channel arrays."""

    human_env_coefficients: np.ndarray
    human_env_delays_s: np.ndarray
    human_env_valid: np.ndarray
    env_human_coefficients: np.ndarray
    env_human_delays_s: np.ndarray
    env_human_valid: np.ndarray
    runtime_profile_s: dict[str, float] | None = None
    runtime_profile_counts: dict[str, int] | None = None
    human_env_active_sets: tuple["_CouplingActiveSet", ...] = ()
    env_human_active_sets: tuple["_CouplingActiveSet", ...] = ()
    human_env_delay_response: PODelayResponse | None = None
    env_human_delay_response: PODelayResponse | None = None


@dataclass(frozen=True)
class _CouplingActiveSet:
    """Phase-center feasible human face paths for one reflector."""

    reflector_index: int
    active_indices: np.ndarray


def extract_environment_reflector_bank(
    scene,
    *,
    max_reflectors: int = 0,
    min_area_m2: float = 1e-4,
) -> EnvironmentReflectorBank:
    """Extract finite planar surfaces, one image per surface; zero means all.

    The optional count cap remains for low-level compatibility. Production
    hybrid selection discovers all planes and applies an image-power cutoff.
    """

    if max_reflectors < 0:
        return _empty_reflector_bank()
    if min_area_m2 < 0.0:
        raise ValueError("min_area_m2 must be non-negative")

    candidates = []
    mesh_cache = {}
    surface_cache = {}
    seen_surfaces = set()
    for object_name, obj in scene.objects.items():
        mesh = getattr(obj, "mi_mesh", None)
        if mesh is None:
            continue
        try:
            vertices, faces, material = _mesh_arrays_for_object(
                obj,
                mesh_cache,
            )
        except (AttributeError, TypeError, ValueError):
            continue

        if vertices.size == 0 or faces.size == 0:
            continue
        centroids, normals, areas = face_geometry(vertices, faces)
        specular_scale = _specular_amplitude_scale(obj.radio_material)
        for face_index, (face, centroid, normal, area) in enumerate(
            zip(faces, centroids, normals, areas)
        ):
            if not np.isfinite(area) or area <= 1e-18:
                continue
            surface_id, support, surface_area = _coplanar_surface_for_face(
                vertices,
                faces,
                face_index,
                surface_cache.setdefault(str(object_name), {}),
            )
            surface_key = (str(object_name), surface_id)
            if surface_key in seen_surfaces or surface_area < float(min_area_m2):
                continue
            seen_surfaces.add(surface_key)
            tri = vertices[np.asarray(face, dtype=np.int64)]
            candidates.append(
                (
                    surface_area,
                    tri,
                    centroid,
                    normal,
                    material,
                    str(object_name),
                    int(face_index),
                    specular_scale,
                    support,
                )
            )

    if not candidates:
        return _empty_reflector_bank()

    candidates.sort(key=lambda item: item[0], reverse=True)
    if max_reflectors > 0:
        candidates = candidates[: int(max_reflectors)]
    return EnvironmentReflectorBank(
        vertices=np.asarray([item[1] for item in candidates], dtype=np.float64),
        centroids=np.asarray([item[2] for item in candidates], dtype=np.float64),
        normals=np.asarray([item[3] for item in candidates], dtype=np.float64),
        areas=np.asarray([item[0] for item in candidates], dtype=np.float64),
        materials=tuple(item[4] for item in candidates),
        object_names=tuple(item[5] for item in candidates),
        face_indices=np.asarray([item[6] for item in candidates], dtype=np.int64),
        path_indices=np.full(len(candidates), -1, dtype=np.int64),
        reflection_gains=None,
        specular_amplitude_scales=np.asarray([item[7] for item in candidates]),
        support_triangles=tuple(item[8] for item in candidates),
    )


def extract_rt_reflector_bank(
    scene,
    static_bank,
    *,
    max_reflectors: int = 0,
    min_area_m2: float = 1e-4,
    frequency_hz: float | None = None,
) -> EnvironmentReflectorBank:
    """Extract unique finite reflector planes seeded by static specular paths.

    ``max_reflectors=0`` means no explicit cap. Negative values return an empty
    bank and are useful for disabled coupling call sites. When ``frequency_hz``
    is provided, each reflector records a diagnostic complex reflection gain
    derived from the corresponding static single-specular propagation
    coefficient, before radar hardware gains. Coupling evaluates the material's
    Fresnel response at its own incidence angles instead of reusing this gain.
    The RT face identifies an image plane; any coplanar triangle of that scene
    object can support a coupled path. Repeated hits on that surface are merged.
    """

    if max_reflectors < 0:
        return _empty_reflector_bank()
    if min_area_m2 < 0.0:
        raise ValueError("min_area_m2 must be non-negative")
    if (
        static_bank.path_vertices is None
        or static_bank.path_interactions is None
        or static_bank.path_objects is None
        or static_bank.path_primitives is None
    ):
        return _empty_reflector_bank()

    object_by_id = {}
    mesh_cache = {}
    surface_cache = {}
    seen_surfaces = set()
    for object_name, obj in scene.objects.items():
        try:
            object_by_id[_as_int(obj.object_id)] = (str(object_name), obj)
        except (AttributeError, TypeError, ValueError):
            continue

    candidates = []
    interactions = np.asarray(static_bank.path_interactions)
    objects = np.asarray(static_bank.path_objects)
    primitives = np.asarray(static_bank.path_primitives)
    vertices = np.asarray(static_bank.path_vertices, dtype=np.float64)
    valid_paths = np.flatnonzero(static_bank.valid)
    for path_idx in valid_paths:
        hit_indices = np.flatnonzero(interactions[:, path_idx] != InteractionType.NONE)
        if hit_indices.size != 1:
            continue
        depth_idx = int(hit_indices[0])
        if int(interactions[depth_idx, path_idx]) != InteractionType.SPECULAR:
            continue
        object_id = int(objects[depth_idx, path_idx])
        primitive_index = int(primitives[depth_idx, path_idx])
        if object_id == int(INVALID_SHAPE) or primitive_index == int(INVALID_PRIMITIVE):
            continue
        entry = object_by_id.get(object_id)
        if entry is None:
            continue
        object_name, obj = entry
        try:
            mesh_vertices, mesh_faces, material = _mesh_arrays_for_object(
                obj,
                mesh_cache,
            )
        except (AttributeError, TypeError, ValueError):
            continue
        if primitive_index < 0 or primitive_index >= mesh_faces.shape[0]:
            continue
        face = mesh_faces[primitive_index]
        tri = mesh_vertices[np.asarray(face, dtype=np.int64)]
        edge_0 = tri[1] - tri[0]
        edge_1 = tri[2] - tri[0]
        normal = np.cross(edge_0, edge_1)
        area = 0.5 * float(np.linalg.norm(normal))
        if not np.isfinite(area) or area <= 1e-18:
            continue
        surface_id, support, surface_area = _coplanar_surface_for_face(
            mesh_vertices,
            mesh_faces,
            primitive_index,
            surface_cache.setdefault(object_id, {}),
        )
        surface_key = (object_id, surface_id)
        if surface_key in seen_surfaces or surface_area < float(min_area_m2):
            continue
        normal = normal / max(2.0 * area, 1e-18)
        hit_point = vertices[depth_idx, path_idx]
        if not np.all(np.isfinite(hit_point)):
            continue
        # Keep the RT hit's tangential coordinates, but use the mesh plane
        # rather than its small intersection-roundoff offset along the normal.
        hit_point = hit_point - np.dot(hit_point - tri[0], normal) * normal
        seen_surfaces.add(surface_key)
        reflection_gain = _rt_calibrated_reflection_gain(
            static_bank,
            int(path_idx),
            frequency_hz=frequency_hz,
        )
        candidates.append(
            (
                int(path_idx),
                tri,
                hit_point,
                normal,
                surface_area,
                material,
                object_name,
                primitive_index,
                reflection_gain,
                _specular_amplitude_scale(obj.radio_material),
                support,
            )
        )

    if not candidates:
        return _empty_reflector_bank()
    if max_reflectors > 0:
        candidates = candidates[: int(max_reflectors)]
    return EnvironmentReflectorBank(
        vertices=np.asarray([item[1] for item in candidates], dtype=np.float64),
        centroids=np.asarray([item[2] for item in candidates], dtype=np.float64),
        normals=np.asarray([item[3] for item in candidates], dtype=np.float64),
        areas=np.asarray([item[4] for item in candidates], dtype=np.float64),
        materials=tuple(item[5] for item in candidates),
        object_names=tuple(item[6] for item in candidates),
        face_indices=np.asarray([item[7] for item in candidates], dtype=np.int64),
        path_indices=np.asarray([item[0] for item in candidates], dtype=np.int64),
        reflection_gains=np.asarray([item[8] for item in candidates], dtype=np.complex128),
        specular_amplitude_scales=np.asarray([item[9] for item in candidates]),
        support_triangles=tuple(item[10] for item in candidates),
    )


def _rt_calibrated_reflection_gain(
    static_bank,
    path_idx: int,
    *,
    frequency_hz: float | None,
) -> complex:
    """Divide the propagation coefficient by free-space loss, excluding hardware."""

    if frequency_hz is None:
        return 1.0 + 0.0j
    propagation_coefficients = getattr(static_bank, "propagation_coefficients", None)
    if propagation_coefficients is None:
        propagation_coefficients = static_bank.coefficients
    coefficients = np.asarray(propagation_coefficients, dtype=np.complex128)
    delays_s = np.asarray(static_bank.delays_s, dtype=np.float64)
    if path_idx < 0 or path_idx >= coefficients.shape[1] or path_idx >= delays_s.size:
        return 1.0 + 0.0j
    path_length_m = float(c * delays_s[path_idx])
    if not np.isfinite(path_length_m) or path_length_m <= 1e-12:
        return 1.0 + 0.0j
    static_coeff = _representative_static_coefficient(coefficients[:, path_idx])
    free_space = _free_space_channel(path_length_m, float(frequency_hz))
    if abs(free_space) <= 1e-30:
        return 1.0 + 0.0j
    gain = static_coeff / free_space
    if not np.isfinite(gain.real) or not np.isfinite(gain.imag):
        return 1.0 + 0.0j
    return complex(gain)


def _representative_static_coefficient(coefficients: np.ndarray) -> complex:
    """Selects the strongest finite channel coefficient for one static path."""

    coefficients = np.asarray(coefficients, dtype=np.complex128).reshape(-1)
    finite = np.isfinite(coefficients.real) & np.isfinite(coefficients.imag)
    coefficients = coefficients[finite]
    if coefficients.size == 0:
        return 0.0 + 0.0j
    return complex(coefficients[int(np.argmax(np.abs(coefficients)))])


def _free_space_channel(path_length_m: float, frequency_hz: float) -> complex:
    """Returns the complex scalar free-space channel for one path length."""

    wavelength_m = c / float(frequency_hz)
    amplitude = wavelength_m / (4.0 * pi * max(float(path_length_m), 1e-18))
    phase = np.exp(-1j * 2.0 * pi * float(path_length_m) / wavelength_m)
    return complex(amplitude * phase)


def _coplanar_surface_for_face(vertices, faces, face_index, cache):
    """Find the union on a seed plane, retaining holes and disconnected pieces.

    Scene-object faces share a material. Unoriented normals allow reversed
    triangle winding; the vertex-to-plane tolerance handles float32 scene
    geometry without merging distinct nearby parallel surfaces.
    """

    if "geometry" not in cache:
        triangles = np.asarray(vertices, dtype=float)[np.asarray(faces, dtype=np.int64)]
        _, normals, areas = face_geometry(vertices, faces)
        cache["geometry"] = triangles, normals, areas
        cache["surfaces"] = {}
    surfaces = cache["surfaces"]
    if face_index in surfaces:
        return surfaces[face_index]
    triangles, normals, areas = cache["geometry"]
    normal = normals[face_index]
    plane_point = triangles[face_index, 0]
    parallel = np.abs(normals @ normal) >= 1.0 - 1e-8
    offsets = (triangles - plane_point) @ normal
    coplanar = parallel & (areas > 1e-18) & np.all(np.abs(offsets) <= 1e-5, axis=1)
    members = np.flatnonzero(coplanar)
    surface = (int(members[0]), triangles[members], float(np.sum(areas[members])))
    for member in members:
        surfaces[int(member)] = surface
    return surface


def _reflector_support(reflectors, reflector_index):
    """Return finite surface support, falling back to a manually supplied seed."""

    support = getattr(reflectors, "support_triangles", None)
    if support is not None:
        return support[reflector_index]
    return reflectors.vertices[reflector_index][None, ...]


def _specular_amplitude_scale(material) -> float:
    """Preserve Sionna's sqrt(1-S^2) loss of the coherent reflected field."""

    scattering = float(np.asarray(getattr(material, "scattering_coefficient", 0.0)).reshape(-1)[0])
    return float(np.sqrt(max(0.0, 1.0 - scattering**2)))


def _mesh_arrays_for_object(obj, mesh_cache):
    """Returns cached vertices, faces, and material for a Sionna scene object."""

    mesh = obj.mi_mesh
    try:
        cache_key = _as_int(obj.object_id)
    except (AttributeError, TypeError, ValueError):
        cache_key = id(mesh)
    cached = mesh_cache.get(cache_key)
    if cached is not None:
        return cached

    vertex_indices = mi.UInt32(np.arange(mesh.vertex_count(), dtype=np.uint32))
    face_indices = mi.UInt32(np.arange(mesh.face_count(), dtype=np.uint32))
    vertices = np.asarray(mesh.vertex_position(vertex_indices), dtype=np.float64).T
    faces = np.asarray(mesh.face_indices(face_indices), dtype=np.int64).T
    value = (
        vertices.reshape((-1, 3)),
        faces.reshape((-1, 3)),
        _coerce_material(obj.radio_material),
    )
    mesh_cache[cache_key] = value
    return value


def _as_int(value) -> int:
    """Converts scalar tensor-like values to Python ``int``."""

    if hasattr(value, "numpy"):
        value = value.numpy()
    return int(np.asarray(value).reshape(()))


def compute_env_human_coupling_channels(
    vertices: np.ndarray,
    faces: np.ndarray,
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    frequency_hz: float,
    human_material,
    reflectors: EnvironmentReflectorBank,
    *,
    virtual_channel_tx_indices: np.ndarray | None = None,
    virtual_channel_rx_indices: np.ndarray | None = None,
    virtual_channel_order: str = "tx_major",
    human_visibility_scene: HumanVisibilityScene,
    static_visibility_scene=None,
    reflection_scale: float = 1.0,
    backend: ComputeBackend = "numpy",
    precision: ComputePrecision = "float64",
    human_env_active_sets: tuple[_CouplingActiveSet, ...] | None = None,
    env_human_active_sets: tuple[_CouplingActiveSet, ...] | None = None,
    radar: RadarSensor | None = None,
    po_config: POConfig | None = None,
    polarization_axis: np.ndarray | None = None,
    max_frequency_offset_hz: float | None = None,
    incremental_state: CouplingIncrementalState | None = None,
) -> EnvHumanCouplingResult:
    """Compute coupled channels, optionally applying hardware at the real endpoints.

    Environment reflection transports complex TE/TM fields at each integration
    point, using the image-ray incidence and material at the phase center.
    When ``radar`` is supplied, its combined antenna pattern is split equally
    in dB between the actual Tx departure and Rx arrival directions. Scalar
    gain and channel phase signs are applied once.

    ``po_config`` selects the same surface integration and refinement limits
    as direct human PO. Temporary subface fields are summed coherently onto
    the original faces in each reflector's image geometry. A compact frequency
    response retains subface delays across the ADC bandwidth when ``radar`` or
    ``max_frequency_offset_hz`` is supplied. Omitting the
    configuration retains the low-level face-centroid behavior.

    Supply a sequence-local ``incremental_state`` to reuse surface integrals
    across chirps. Geometry thresholds and full refreshes follow ``po_config``;
    visibility follows the active-set refresh schedule.
    """
    return _CouplingChannelRun(
        _CouplingChannelInputs(
            vertices=vertices,
            faces=faces,
            tx_positions=tx_positions,
            rx_positions=rx_positions,
            frequency_hz=frequency_hz,
            human_material=human_material,
            reflectors=reflectors,
            virtual_channel_tx_indices=virtual_channel_tx_indices,
            virtual_channel_rx_indices=virtual_channel_rx_indices,
            virtual_channel_order=virtual_channel_order,
            human_visibility_scene=human_visibility_scene,
            static_visibility_scene=static_visibility_scene,
            reflection_scale=reflection_scale,
            backend=backend,
            precision=precision,
            human_env_active_sets=human_env_active_sets,
            env_human_active_sets=env_human_active_sets,
            radar=radar,
            po_config=po_config,
            polarization_axis=polarization_axis,
            max_frequency_offset_hz=max_frequency_offset_hz,
            incremental_state=incremental_state,
        )
    ).run()


@dataclass(frozen=True, kw_only=True)
class _CouplingChannelInputs:
    """Geometry and options retained by reference for one coupling solve."""

    vertices: np.ndarray
    faces: np.ndarray
    tx_positions: np.ndarray
    rx_positions: np.ndarray
    frequency_hz: float
    human_material: object
    reflectors: EnvironmentReflectorBank
    virtual_channel_tx_indices: np.ndarray | None = None
    virtual_channel_rx_indices: np.ndarray | None = None
    virtual_channel_order: str = "tx_major"
    human_visibility_scene: HumanVisibilityScene
    static_visibility_scene: object = None
    reflection_scale: float = 1.0
    backend: ComputeBackend = "numpy"
    precision: ComputePrecision = "float64"
    human_env_active_sets: tuple[_CouplingActiveSet, ...] | None = None
    env_human_active_sets: tuple[_CouplingActiveSet, ...] | None = None
    radar: RadarSensor | None = None
    po_config: POConfig | None = None
    polarization_axis: np.ndarray | None = None
    max_frequency_offset_hz: float | None = None
    incremental_state: CouplingIncrementalState | None = None


class _CouplingChannelRun:
    """Per-call state with explicit preparation, channel/product and output stages."""

    def __init__(self, inputs: _CouplingChannelInputs):
        self.vertices = inputs.vertices
        self.faces = inputs.faces
        self.tx_positions = inputs.tx_positions
        self.rx_positions = inputs.rx_positions
        self.frequency_hz = inputs.frequency_hz
        self.human_material = inputs.human_material
        self.reflectors = inputs.reflectors
        self.virtual_channel_tx_indices = inputs.virtual_channel_tx_indices
        self.virtual_channel_rx_indices = inputs.virtual_channel_rx_indices
        self.virtual_channel_order = inputs.virtual_channel_order
        self.human_visibility_scene = inputs.human_visibility_scene
        self.static_visibility_scene = inputs.static_visibility_scene
        self.reflection_scale = inputs.reflection_scale
        self.backend = inputs.backend
        self.precision = inputs.precision
        self.human_env_active_sets = inputs.human_env_active_sets
        self.env_human_active_sets = inputs.env_human_active_sets
        self.radar = inputs.radar
        self.po_config = inputs.po_config
        self.polarization_axis = inputs.polarization_axis
        self.max_frequency_offset_hz = inputs.max_frequency_offset_hz
        self.incremental_state = inputs.incremental_state

    def timed_coupling_for_pair(self, order, tx_position, rx_position):
        """Runs one uncached coupling solve and records its runtime bucket."""

        pair_t0 = time.perf_counter()
        out = _coupling_for_pair(
            self.centroids,
            self.normals,
            self.areas,
            self.face_indices,
            tx_position,
            rx_position,
            self.frequency_hz,
            self.human_material,
            self.reflectors,
            human_visibility_scene=self.human_visibility_scene,
            static_visibility_scene=self.static_visibility_scene,
            order=order,
            reflection_scale=self.reflection_scale,
            backend=self.backend,
            precision=self.precision,
            vertices=self.vertices,
            faces=self.faces,
            po_config=self.po_config,
            tx_positions=self.tx_positions,
            rx_positions=self.rx_positions,
            polarization_axis=self.polarization_axis,
            delay_response=self.responses[order],
            incremental_state=self.incremental_state,
        )
        key = f"{order}_channel"
        self.runtime_profile_s[key] += time.perf_counter() - pair_t0
        self.runtime_profile_counts[key] += 1
        return out

    def timed_cached_coupling_for_pair(self, order, tx_position, rx_position, active_sets):
        """Runs one active-set coupling solve and records its runtime bucket."""

        pair_t0 = time.perf_counter()
        out = _coupling_for_pair_active_sets(
            self.centroids,
            self.normals,
            self.areas,
            self.face_indices,
            tx_position,
            rx_position,
            self.frequency_hz,
            self.human_material,
            self.reflectors,
            order=order,
            reflection_scale=self.reflection_scale,
            active_sets=active_sets,
            static_visibility_scene=self.static_visibility_scene,
            backend=self.backend,
            precision=self.precision,
            vertices=self.vertices,
            faces=self.faces,
            po_config=self.po_config,
            tx_positions=self.tx_positions,
            rx_positions=self.rx_positions,
            polarization_axis=self.polarization_axis,
            delay_response=self.responses[order],
            incremental_state=self.incremental_state,
        )
        key = f"{order}_channel"
        self.runtime_profile_s[key] += time.perf_counter() - pair_t0
        self.runtime_profile_counts[key] += 1
        return out

    def timed_cached_coupling_for_pairs(self, order, active_sets, center_coefficients):
        """Runs all-channel active-set coupling and records its runtime bucket."""

        pair_t0 = time.perf_counter()
        out = _coupling_for_pairs_active_sets(
            self.centroids,
            self.normals,
            self.areas,
            self.face_indices,
            self.tx_positions,
            self.rx_positions,
            self.frequency_hz,
            self.human_material,
            self.reflectors,
            order=order,
            reflection_scale=self.reflection_scale,
            active_sets=active_sets,
            center_coefficients=center_coefficients,
            virtual_channel_tx_indices=self.vc_tx,
            virtual_channel_rx_indices=self.vc_rx,
            precision=self.precision,
        )
        key = f"{order}_channel"
        self.runtime_profile_s[key] += time.perf_counter() - pair_t0
        self.runtime_profile_counts[key] += 1
        return out

    def _stage_allocate(self):
        """Allocate for this call."""
        self.total_t0 = time.perf_counter()
        self.runtime_profile_s = {
            "coupling_setup": 0.0,
            "human_env_channel": 0.0,
            "env_human_channel": 0.0,
            "coupling_assembly": 0.0,
            "coupling_total": 0.0,
        }
        self.runtime_profile_counts = dict.fromkeys(self.runtime_profile_s, 0)

        self._, self.complex_dtype = _numpy_dtypes_for_precision(self.precision)
        validate_compute_backend(self.backend)
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.uint32)
        self.tx_positions = np.asarray(self.tx_positions, dtype=np.float64)
        self.rx_positions = np.asarray(self.rx_positions, dtype=np.float64)
        self.vc_tx, self.vc_rx = resolve_virtual_channel_pairs(
            int(self.tx_positions.shape[0]),
            int(self.rx_positions.shape[0]),
            order=self.virtual_channel_order,
            tx_indices=self.virtual_channel_tx_indices,
            rx_indices=self.virtual_channel_rx_indices,
        )
        self.num_vc = int(self.vc_tx.size)
        self.num_reflectors = int(self.reflectors.vertices.shape[0])
        self.num_faces = int(self.faces.shape[0])
        self.num_paths = self.num_reflectors * self.num_faces
        if self.num_paths == 0:
            self.result = _empty_coupling_result(self.num_vc, self.precision)
            self.result.runtime_profile_s["coupling_total"] = time.perf_counter() - self.total_t0
            self.result.runtime_profile_counts["coupling_total"] = 1
            return self.result

    def _stage_geometry_and_responses(self):
        """Geometry and responses for this call."""
        self.setup_t0 = time.perf_counter()
        self.po_config = self.po_config or POConfig()
        if self.polarization_axis is None:
            self.polarization_axis = (
                np.array([0.0, 0.0, 1.0])
                if self.radar is None
                else rotation_matrix(self.radar.orientation)[:, 2]
            )
        self.centroids, self.normals, self.areas = face_geometry(self.vertices, self.faces)
        self.face_indices = np.arange(self.num_faces, dtype=np.int64)
        self.human_material = _coerce_material(self.human_material)
        self.tx_center = np.mean(self.tx_positions, axis=0)
        self.rx_center = np.mean(self.rx_positions, axis=0)
        self.runtime_profile_s["coupling_setup"] = time.perf_counter() - self.setup_t0
        self.runtime_profile_counts["coupling_setup"] = 1

        if self.max_frequency_offset_hz is None and self.radar is not None:
            self.max_frequency_offset_hz = (
                self.radar.fmcw.slope
                * (self.radar.fmcw.num_adc_samples - 1)
                / self.radar.fmcw.sampling_frequency
            )
        if self.incremental_state is not None:
            self.incremental_state.begin_chirp(
                (
                    id(self.reflectors),
                    id(self.static_visibility_scene),
                    self.vertices.shape,
                    self.faces.tobytes(),
                    self.tx_positions.tobytes(),
                    self.rx_positions.tobytes(),
                    self.frequency_hz,
                    self.human_material,
                    np.asarray(self.polarization_axis).tobytes(),
                    self.reflection_scale,
                    self.backend,
                    self.precision,
                    self.po_config.po_integration_mode,
                    self.po_config.effective_po_phase_span_scale_rad,
                    self.po_config.po_curvature_error_rad,
                    self.po_config.po_quadrature_max_refinement_depth,
                    self.po_config.po_quadrature_max_subfaces_per_parent,
                    self.max_frequency_offset_hz,
                    None if self.radar is None else repr(self.radar.fmcw),
                ),
                (
                    None
                    if self.radar is None or self.po_config.po_refresh_frames == 0
                    else refresh_period_chirps(
                        self.po_config.po_refresh_frames, self.radar.fmcw.num_chirps_per_frame
                    )
                ),
            )
        self.responses = {
            order: (
                PODelayResponse.for_mesh(
                    self.vertices,
                    self.faces,
                    self.num_paths,
                    self.max_frequency_offset_hz,
                    compute_precision=self.precision,
                    previous=(
                        None
                        if self.incremental_state is None
                        else next(
                            (
                                state.delay_response
                                for key, state in self.incremental_state.facets.items()
                                if key[0] == order and state.delay_response is not None
                            ),
                            None,
                        )
                    ),
                    receiver_fmcw=None if self.radar is None else self.radar.fmcw,
                    channel_delay_margin_s=(
                        0.0
                        if self.radar is None
                        else (
                            np.max(
                                np.linalg.norm(
                                    self.radar.world_tx_positions()
                                    - np.mean(self.radar.world_tx_positions(), axis=0),
                                    axis=1,
                                )
                            )
                            + np.max(
                                np.linalg.norm(
                                    self.radar.world_rx_positions()
                                    - np.mean(self.radar.world_rx_positions(), axis=0),
                                    axis=1,
                                )
                            )
                        )
                        / c
                    ),
                )
                if self.max_frequency_offset_hz is not None
                and self.po_config.po_integration_mode != "face_centroid"
                else None
            )
            for order in ("human_env", "env_human")
        }

    def _stage_center_channels(self):
        """Center channels for this call."""
        if self.human_env_active_sets is None:
            (
                self.human_env_center_a,
                self.human_env_tau,
                self.human_env_valid,
                self.human_env_active_sets,
            ) = self.timed_coupling_for_pair("human_env", self.tx_center, self.rx_center)
        else:
            (
                self.human_env_center_a,
                self.human_env_tau,
                self.human_env_valid,
            ) = self.timed_cached_coupling_for_pair(
                "human_env",
                self.tx_center,
                self.rx_center,
                self.human_env_active_sets,
            )
        if self.env_human_active_sets is None:
            (
                self.env_human_center_a,
                self.env_human_tau,
                self.env_human_valid,
                self.env_human_active_sets,
            ) = self.timed_coupling_for_pair("env_human", self.tx_center, self.rx_center)
        else:
            (
                self.env_human_center_a,
                self.env_human_tau,
                self.env_human_valid,
            ) = self.timed_cached_coupling_for_pair(
                "env_human",
                self.tx_center,
                self.rx_center,
                self.env_human_active_sets,
            )

    def _stage_project_channels(self):
        """Project channels for this call."""
        self.assembly_t0 = time.perf_counter()
        self.human_env_a = np.zeros((self.num_vc, self.num_paths), dtype=self.complex_dtype)
        self.env_human_a = np.zeros((self.num_vc, self.num_paths), dtype=self.complex_dtype)
        if (
            self.num_vc == 1
            and np.allclose(self.tx_positions[0], self.tx_center)
            and np.allclose(self.rx_positions[0], self.rx_center)
        ):
            self.human_env_a[0] = self.human_env_center_a
            self.env_human_a[0] = self.env_human_center_a
            self.human_env_tau_vc = self.human_env_tau
            self.env_human_tau_vc = self.env_human_tau
            self.assembly_elapsed_s = time.perf_counter() - self.assembly_t0
        else:
            self.assembly_elapsed_s = time.perf_counter() - self.assembly_t0
            self.human_env_a[:], self.human_env_tau_vc = self.timed_cached_coupling_for_pairs(
                "human_env",
                self.human_env_active_sets,
                self.human_env_center_a,
            )
            self.env_human_a[:], self.env_human_tau_vc = self.timed_cached_coupling_for_pairs(
                "env_human",
                self.env_human_active_sets,
                self.env_human_center_a,
            )

    def _stage_wideband_and_hardware(self):
        """Wideband and hardware for this call."""
        self.assembly_t0 = time.perf_counter()
        for self.order, self.active_sets in (
            ("human_env", self.human_env_active_sets),
            ("env_human", self.env_human_active_sets),
        ):
            self.response = self.responses[self.order]
            if self.response is None:
                continue
            if self.num_vc == 1:
                self.gains = np.ones((1, self.num_paths), complex)
            else:
                self.gains, self._ = _coupling_for_pairs_active_sets(
                    self.centroids,
                    self.normals,
                    self.areas,
                    self.face_indices,
                    self.tx_positions,
                    self.rx_positions,
                    self.frequency_hz,
                    self.human_material,
                    self.reflectors,
                    order=self.order,
                    reflection_scale=1.0,
                    active_sets=self.active_sets,
                    center_coefficients=np.ones(self.num_paths, complex),
                    virtual_channel_tx_indices=self.vc_tx,
                    virtual_channel_rx_indices=self.vc_rx,
                    precision=self.precision,
                )
            if self.radar is not None:
                _apply_coupling_hardware_gain(
                    self.gains,
                    radar=self.radar,
                    centroids=self.centroids,
                    reflectors=self.reflectors,
                    active_sets=self.active_sets,
                    order=self.order,
                    tx_positions=self.tx_positions,
                    rx_positions=self.rx_positions,
                    vc_tx=self.vc_tx,
                    vc_rx=self.vc_rx,
                )
            self.responses[self.order] = self.response.with_channel_gains(self.gains)
        if self.responses["human_env"] is not None:
            self.human_env_valid |= self.responses["human_env"].valid_paths
            self.env_human_valid |= self.responses["env_human"].valid_paths

        if self.radar is not None:
            for self.order, self.coefficients, self.active_sets in (
                ("human_env", self.human_env_a, self.human_env_active_sets),
                ("env_human", self.env_human_a, self.env_human_active_sets),
            ):
                _apply_coupling_hardware_gain(
                    self.coefficients,
                    radar=self.radar,
                    centroids=self.centroids,
                    reflectors=self.reflectors,
                    active_sets=self.active_sets,
                    order=self.order,
                    tx_positions=self.tx_positions,
                    rx_positions=self.rx_positions,
                    vc_tx=self.vc_tx,
                    vc_rx=self.vc_rx,
                )
        self.runtime_profile_s["coupling_assembly"] += (
            self.assembly_elapsed_s + time.perf_counter() - self.assembly_t0
        )
        self.runtime_profile_counts["coupling_assembly"] += 1

    def _stage_result(self):
        """Result for this call."""
        self.runtime_profile_s["coupling_total"] = time.perf_counter() - self.total_t0
        self.runtime_profile_counts["coupling_total"] = 1
        if self.incremental_state is not None:
            self.incremental_state.end_chirp()
            self.runtime_profile_counts["coupling_recomputed_faces"] = (
                self.incremental_state.recomputed_faces
            )
            self.runtime_profile_counts["coupling_phase_updated_faces"] = (
                self.incremental_state.phase_updated_faces
            )
        return EnvHumanCouplingResult(
            human_env_coefficients=self.human_env_a,
            human_env_delays_s=self.human_env_tau_vc,
            human_env_valid=self.human_env_valid,
            env_human_coefficients=self.env_human_a,
            env_human_delays_s=self.env_human_tau_vc,
            env_human_valid=self.env_human_valid,
            runtime_profile_s=self.runtime_profile_s,
            runtime_profile_counts=self.runtime_profile_counts,
            human_env_active_sets=self.human_env_active_sets,
            env_human_active_sets=self.env_human_active_sets,
            human_env_delay_response=self.responses["human_env"],
            env_human_delay_response=self.responses["env_human"],
        )

    def run(self):
        empty_result = self._stage_allocate()
        if empty_result is not None:
            return empty_result
        self._stage_geometry_and_responses()
        self._stage_center_channels()
        self._stage_project_channels()
        self._stage_wideband_and_hardware()
        return self._stage_result()


def _apply_coupling_hardware_gain(
    coefficients: np.ndarray,
    *,
    radar: RadarSensor,
    centroids: np.ndarray,
    reflectors: EnvironmentReflectorBank,
    active_sets: tuple[_CouplingActiveSet, ...],
    order: str,
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    vc_tx: np.ndarray,
    vc_rx: np.ndarray,
) -> None:
    """Weight real first/last legs after scene-to-radar antenna-frame conversion.

    Image endpoints locate the reflection point in the scene frame. Pattern
    angles are evaluated at the real antennas and rotated to radar-local once.
    """

    tx_ids, tx_inverse = np.unique(vc_tx, return_inverse=True)
    rx_ids, rx_inverse = np.unique(vc_rx, return_inverse=True)
    reflected_ids = rx_ids if order == "human_env" else tx_ids
    reflected_inverse = rx_inverse if order == "human_env" else tx_inverse
    num_endpoints = reflected_ids.size
    pattern = radar.hardware.antenna_pattern
    independent_pattern = pattern is None or type(pattern) is SimpleCosinePattern
    endpoint_patterns = (
        {
            count: None if pattern is None else pattern.subset(np.arange(count))
            for count in {tx_ids.size, rx_ids.size}
        }
        if independent_pattern
        else {}
    )

    def endpoint_gain(points, element_positions, inverse):
        local = radar.world_to_local_points(points) - element_positions[:, None, :]
        azimuth = np.degrees(np.arctan2(local[..., 1], local[..., 0]))
        elevation = np.degrees(
            np.arctan2(
                local[..., 2],
                np.hypot(local[..., 0], local[..., 1]),
            )
        )
        if independent_pattern:
            # Identical channel patterns need one evaluation per real element.
            gain_db = np.zeros(azimuth.shape, dtype=float)
            if radar.hardware.scalar_antenna_gain_dbi is not None:
                gain_db += float(radar.hardware.scalar_antenna_gain_dbi)
            endpoint_pattern = endpoint_patterns[element_positions.shape[0]]
            if endpoint_pattern is not None:
                gain_db += endpoint_pattern.loss_db(
                    azimuth_deg=azimuth,
                    elevation_deg=elevation,
                    frequency_hz=radar.fmcw.carrier_frequency,
                )
            return np.power(10.0, gain_db / 20.0).astype(np.complex128)[inverse]
        # Digitized/custom patterns can differ per virtual channel. Reuse
        # element geometry, then evaluate each channel's own pattern curve.
        return radar.hardware.channel_gain(
            azimuth_deg=azimuth[inverse],
            elevation_deg=elevation[inverse],
            frequency_hz=radar.fmcw.carrier_frequency,
            include_phase=False,
        )

    phase = radar.hardware.channel_gain(
        include_antenna_gain=False,
        include_pattern=False,
        frequency_hz=radar.fmcw.carrier_frequency,
    )[:, None]
    for active_set in active_sets:
        reflector_index = int(active_set.reflector_index)
        active = np.asarray(active_set.active_indices, dtype=np.int64)
        if active.size == 0:
            continue
        face_points = centroids[active]
        point = reflectors.centroids[reflector_index]
        normal = reflectors.normals[reflector_index]
        triangles = _reflector_support(reflectors, reflector_index)
        if order == "human_env":
            virtual = _mirror_points(rx_positions[rx_ids], point, normal)
            starts = np.tile(face_points, (num_endpoints, 1))
            ends = np.repeat(virtual, active.size, axis=0)
        else:
            virtual = _mirror_points(tx_positions[tx_ids], point, normal)
            starts = np.repeat(virtual, active.size, axis=0)
            ends = np.tile(face_points, (num_endpoints, 1))
        reflection_points, finite = _reflection_points_on_surface(
            starts,
            ends,
            point,
            normal,
            triangles,
        )
        finite = finite.reshape(num_endpoints, active.size)
        reflection_points = reflection_points.reshape(num_endpoints, active.size, 3)
        reflection_points = np.where(
            finite[..., None],
            reflection_points,
            face_points[None, :, :],
        )
        tx_points, rx_points = (
            (face_points[None, :, :], reflection_points)
            if order == "human_env"
            else (reflection_points, face_points[None, :, :])
        )

        tx_gain = endpoint_gain(tx_points, radar.hardware.tx_positions[tx_ids], tx_inverse)
        rx_gain = endpoint_gain(rx_points, radar.hardware.rx_positions[rx_ids], rx_inverse)
        gain = phase * np.sqrt(tx_gain * rx_gain)
        indices = reflector_index * centroids.shape[0] + active
        coefficients[:, indices] = np.where(
            finite[reflected_inverse],
            coefficients[:, indices] * gain,
            0.0,
        ).astype(coefficients.dtype, copy=False)


def _image_polarization_callback(
    *,
    tx_position: np.ndarray,
    rx_position: np.ndarray,
    reflector_normal: np.ndarray,
    material: PhysicalOpticsMaterial,
    frequency_hz: float,
    order: str,
    scale: float,
    polarization_axis: np.ndarray,
    reflector_point: np.ndarray | None = None,
    support_triangles: np.ndarray | None = None,
    static_visibility_scene=None,
) -> PolarizationCallback:
    """Transport antenna fields through one planar reflection in image geometry.

    For an Rx image, propagate the receive field backwards by reciprocity.
    Conjugating that transported field supplies the Hermitian receive vector
    expected by PO, so the forward channel retains Gamma, not conj(Gamma).
    """

    if order not in ("human_env", "env_human"):
        raise ValueError("order must be 'human_env' or 'env_human'")
    normal = _normalize(reflector_normal)
    image = rx_position if order == "human_env" else tx_position
    direct = tx_position if order == "human_env" else rx_position

    def polarizations(points):
        reflected_hat = _normalize_rows(points - np.asarray(image))
        incident_hat = reflected_hat - 2.0 * (reflected_hat @ normal)[:, None] * normal
        normals = np.broadcast_to(normal, incident_hat.shape)
        te, tm_inc = _local_incidence_bases(incident_hat, normals, polarization_axis)
        tm_ref = np.cross(te, reflected_hat)
        antenna_pol = _transverse_polarizations(incident_hat, polarization_axis)
        cos_theta = np.abs(incident_hat @ normal)
        r_te, r_tm = fresnel_coefficients(cos_theta, material, frequency_hz)
        field = float(scale) * (
            (r_te * np.einsum("ij,ij->i", antenna_pol, te))[:, None] * te
            + (r_tm * np.einsum("ij,ij->i", antenna_pol, tm_inc))[:, None] * tm_ref
        )
        if support_triangles is not None:
            reflection_points, supported = _reflection_points_on_surface(
                image,
                points,
                reflector_point,
                normal,
                support_triangles,
            )
            if static_visibility_scene is not None:
                indices = np.flatnonzero(supported)
                real_endpoint = _mirror_point(image, reflector_point, normal)
                supported[indices] &= (
                    _segments_clear_in_scene(
                        starts=direct, ends=points[indices], scene=static_visibility_scene
                    )
                    & _segments_clear_in_scene(
                        starts=points[indices],
                        ends=reflection_points[indices],
                        scene=static_visibility_scene,
                    )
                    & _segments_clear_in_scene(
                        starts=real_endpoint,
                        ends=reflection_points[indices],
                        scene=static_visibility_scene,
                    )
                )
            field[~supported] = 0.0
        direct_pol = _transverse_polarizations(
            _normalize_rows(points - np.asarray(direct)),
            polarization_axis,
        )
        return (direct_pol, field.conjugate()) if order == "human_env" else (field, direct_pol)

    return polarizations


def _coupling_face_contributions(
    *,
    incremental_state=None,
    cache_key=None,
    **kwargs,
):
    """Evaluate surface fields, optionally reusing their last exact integral."""

    if incremental_state is None:
        return _integrate_coupling_faces(**kwargs)
    return incremental_state.evaluate(
        integrate=_integrate_coupling_faces,
        cache_key=cache_key,
        **kwargs,
    )


def _integrate_coupling_faces(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_indices: np.ndarray,
    centroids: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    tx_position: np.ndarray,
    rx_position: np.ndarray,
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    frequency_hz: float,
    material: PhysicalOpticsMaterial,
    po_config: POConfig,
    backend: ComputeBackend,
    precision: ComputePrecision,
    polarization_axis: np.ndarray | None = None,
    reflector_normal: np.ndarray | None = None,
    reflector_material: PhysicalOpticsMaterial | None = None,
    order: str | None = None,
    reflection_scale: float = 1.0,
    reflector_point: np.ndarray | None = None,
    support_triangles: np.ndarray | None = None,
    delay_response: PODelayResponse | None = None,
    path_offset: int = 0,
    static_visibility_scene=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate visible parent faces using the current real/image endpoints.

    Visibility and stored delays stay on parent faces, as in direct human PO.
    Refinement uses the reflected path's phase variation. TE/TM environment
    reflection is applied to each subface field before the coherent sum.
    """

    polarization_callback = None
    if reflector_normal is not None:
        polarization_callback = _image_polarization_callback(
            tx_position=tx_position,
            rx_position=rx_position,
            reflector_normal=reflector_normal,
            material=reflector_material,
            frequency_hz=frequency_hz,
            order=order,
            scale=reflection_scale,
            polarization_axis=(
                np.array([0.0, 0.0, 1.0]) if polarization_axis is None else polarization_axis
            ),
            reflector_point=reflector_point,
            support_triangles=support_triangles,
            static_visibility_scene=static_visibility_scene,
        )
    if po_config.po_integration_mode == "face_centroid":
        return vector_facet_face_contributions(
            face_indices=face_indices,
            centroids=centroids,
            normals=normals,
            areas=areas,
            tx_position=tx_position,
            rx_position=rx_position,
            frequency_hz=frequency_hz,
            material=material,
            backend=backend,
            precision=precision,
            polarization_axis=polarization_axis,
            polarization_callback=polarization_callback,
        )

    geometry_kwargs = dict(
        vertices=vertices,
        faces=faces,
        selected_parent_indices=face_indices,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        frequency_hz=frequency_hz,
        max_refinement_depth=po_config.po_quadrature_max_refinement_depth,
        max_subfaces_per_parent=po_config.po_quadrature_max_subfaces_per_parent,
        progress_callback=po_config.progress_callback,
    )
    if po_config.po_integration_mode == "parent_face_quadrature":
        sample_centroids, sample_normals, sample_areas, parent_indices = (
            parent_face_quadrature_geometry(
                **geometry_kwargs,
                phase_span_scale_rad=po_config.effective_po_phase_span_scale_rad,
            )
        )
        aggregate = _aggregate_parent_face_quadrature_contributions
        integration_kwargs = {}
    elif po_config.po_integration_mode == "parent_face_adaptive_analytic":
        (
            sample_centroids,
            sample_normals,
            sample_areas,
            parent_indices,
            sample_triangles,
        ) = parent_face_adaptive_analytic_geometry(
            **geometry_kwargs,
            phase_span_scale_rad=po_config.effective_po_phase_span_scale_rad,
            curvature_error_rad=po_config.po_curvature_error_rad,
        )
        aggregate = _aggregate_parent_face_far_field_analytic_contributions
        integration_kwargs = {"integration_triangles": sample_triangles}
    else:
        (
            sample_centroids,
            sample_normals,
            sample_areas,
            parent_indices,
            sample_triangles,
            _,
            _,
        ) = parent_face_far_field_analytic_geometry(**geometry_kwargs)
        aggregate = _aggregate_parent_face_far_field_analytic_contributions
        integration_kwargs = {"integration_triangles": sample_triangles}

    if delay_response is not None:
        delay_response.set_reference(
            path_offset + face_indices, centroids[face_indices], tx_position, rx_position
        )
    contributions = aggregate(
        selected_parent_indices=face_indices,
        integration_centroids=sample_centroids,
        integration_normals=sample_normals,
        integration_areas=sample_areas,
        path_face_indices=parent_indices,
        effective_weights=np.ones(faces.shape[0], dtype=float),
        tx_phase_center=tx_position,
        rx_phase_center=rx_position,
        frequency_hz=frequency_hz,
        material=material,
        backend=backend,
        precision=precision,
        progress_callback=po_config.progress_callback,
        polarization_axis=polarization_axis,
        polarization_callback=polarization_callback,
        delay_response=delay_response,
        delay_path_indices=path_offset + face_indices,
        **integration_kwargs,
    )
    delays_s = _phase_center_delays_for_centroids(
        centroids[face_indices],
        tx_position,
        rx_position,
        precision,
    )
    return contributions, delays_s


def _coupling_for_pair(
    centroids: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    face_indices: np.ndarray,
    tx_position: np.ndarray,
    rx_position: np.ndarray,
    frequency_hz: float,
    human_material: PhysicalOpticsMaterial,
    reflectors: EnvironmentReflectorBank,
    *,
    human_visibility_scene: HumanVisibilityScene,
    static_visibility_scene,
    order: str,
    reflection_scale: float,
    backend: ComputeBackend,
    precision: ComputePrecision,
    vertices: np.ndarray,
    faces: np.ndarray,
    po_config: POConfig,
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    polarization_axis: np.ndarray | None = None,
    delay_response: PODelayResponse | None = None,
    incremental_state: CouplingIncrementalState | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[_CouplingActiveSet, ...]]:
    """Computes coupling paths for one Tx/Rx pair and records active sets."""

    _, complex_dtype = _numpy_dtypes_for_precision(precision)
    num_faces = face_indices.size
    num_reflectors = int(reflectors.vertices.shape[0])
    num_paths = num_faces * num_reflectors
    coefficients = np.zeros((num_paths,), dtype=complex_dtype)
    delays = np.zeros((num_paths,), dtype=float)
    valid = np.zeros((num_paths,), dtype=bool)
    active_sets: list[_CouplingActiveSet] = []

    if order not in ("human_env", "env_human"):
        raise ValueError("order must be 'human_env' or 'env_human'")
    real_endpoint = tx_position if order == "human_env" else rx_position
    # Visibility is geometric and reciprocal; the shared PO dyadic handles
    # illumination. A one-sided normal test here would reject just one of a
    # reciprocal pair of bistatic paths.
    real_leg_clear = _segments_clear_in_scene(
        starts=real_endpoint,
        ends=centroids,
        scene=human_visibility_scene.scene,
    )
    if static_visibility_scene is not None:
        real_leg_clear &= _segments_clear_in_scene(
            starts=real_endpoint,
            ends=centroids,
            scene=static_visibility_scene,
        )

    for reflector_index in range(num_reflectors):
        path_offset = reflector_index * num_faces
        reflector_point = reflectors.centroids[reflector_index]
        reflector_normal = _normalize(reflectors.normals[reflector_index])
        reflector_vertices = _reflector_support(reflectors, reflector_index)
        material = reflectors.materials[reflector_index]
        specular_scale = (
            1.0
            if reflectors.specular_amplitude_scales is None
            else float(reflectors.specular_amplitude_scales[reflector_index])
        )

        if order == "human_env":
            virtual_endpoint = _mirror_point(
                rx_position,
                reflector_point,
                reflector_normal,
            )
            reflection_points, finite = _reflection_points_on_surface(
                centroids,
                virtual_endpoint,
                reflector_point,
                reflector_normal,
                reflector_vertices,
            )
            tx_for_po = tx_position
            rx_for_po = virtual_endpoint
            tx_array_for_po = tx_positions
            rx_array_for_po = _mirror_points(
                rx_positions,
                reflector_point,
                reflector_normal,
            )
        else:
            virtual_endpoint = _mirror_point(
                tx_position,
                reflector_point,
                reflector_normal,
            )
            reflection_points, finite = _reflection_points_on_surface(
                virtual_endpoint,
                centroids,
                reflector_point,
                reflector_normal,
                reflector_vertices,
            )
            tx_for_po = virtual_endpoint
            rx_for_po = rx_position
            tx_array_for_po = _mirror_points(
                tx_positions,
                reflector_point,
                reflector_normal,
            )
            rx_array_for_po = rx_positions

        # Feasibility is tested on the real broken path. The mirrored endpoint
        # only parameterizes the image-method field/delay calculation below.
        candidate_indices = np.flatnonzero(finite & real_leg_clear)
        if candidate_indices.size == 0:
            continue

        reflection_candidate_points = reflection_points[candidate_indices]
        centroid_candidate_points = centroids[candidate_indices]
        face_to_reflector_clear = _segments_clear_in_scene(
            starts=reflection_candidate_points,
            ends=centroid_candidate_points,
            scene=human_visibility_scene.scene,
        )
        if order == "human_env":
            radar_to_reflector_clear = _segments_clear_in_scene(
                starts=rx_position,
                ends=reflection_candidate_points,
                scene=human_visibility_scene.scene,
            )
        else:
            radar_to_reflector_clear = _segments_clear_in_scene(
                starts=tx_position,
                ends=reflection_candidate_points,
                scene=human_visibility_scene.scene,
            )
        if static_visibility_scene is not None:
            face_to_reflector_clear &= _segments_clear_in_scene(
                starts=reflection_candidate_points,
                ends=centroid_candidate_points,
                scene=static_visibility_scene,
            )
            radar_to_reflector_clear &= _segments_clear_in_scene(
                starts=rx_position if order == "human_env" else tx_position,
                ends=reflection_candidate_points,
                scene=static_visibility_scene,
            )
        active_indices = candidate_indices[face_to_reflector_clear & radar_to_reflector_clear]
        if active_indices.size == 0:
            continue
        active_sets.append(
            _CouplingActiveSet(
                reflector_index=int(reflector_index),
                active_indices=active_indices.astype(np.int64, copy=True),
            )
        )

        contrib, tau = _coupling_face_contributions(
            incremental_state=incremental_state,
            cache_key=(order, int(reflector_index)),
            vertices=vertices,
            faces=faces,
            po_config=po_config,
            face_indices=face_indices[active_indices],
            centroids=centroids,
            normals=normals,
            areas=areas,
            tx_position=tx_for_po,
            rx_position=rx_for_po,
            tx_positions=tx_array_for_po,
            rx_positions=rx_array_for_po,
            frequency_hz=frequency_hz,
            material=human_material,
            backend=backend,
            precision=precision,
            polarization_axis=polarization_axis,
            reflector_normal=reflector_normal,
            reflector_material=material,
            order=order,
            reflection_scale=reflection_scale * specular_scale,
            reflector_point=reflector_point,
            support_triangles=reflector_vertices,
            delay_response=delay_response,
            path_offset=path_offset,
            static_visibility_scene=static_visibility_scene,
        )
        out_indices = path_offset + active_indices
        coefficients[out_indices] = contrib
        delays[out_indices] = tau
        valid[out_indices] = np.isfinite(tau) & (np.abs(contrib) > 0.0)

    return coefficients, delays, valid, tuple(active_sets)


def _coupling_for_pair_active_sets(
    centroids: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    face_indices: np.ndarray,
    tx_position: np.ndarray,
    rx_position: np.ndarray,
    frequency_hz: float,
    human_material: PhysicalOpticsMaterial,
    reflectors: EnvironmentReflectorBank,
    *,
    order: str,
    reflection_scale: float,
    active_sets: tuple[_CouplingActiveSet, ...],
    static_visibility_scene=None,
    backend: ComputeBackend,
    precision: ComputePrecision,
    vertices: np.ndarray,
    faces: np.ndarray,
    po_config: POConfig,
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    polarization_axis: np.ndarray | None = None,
    delay_response: PODelayResponse | None = None,
    incremental_state: CouplingIncrementalState | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recomputes one Tx/Rx coupling pair using previously visible active sets."""

    _, complex_dtype = _numpy_dtypes_for_precision(precision)
    num_faces = face_indices.size
    num_reflectors = int(reflectors.vertices.shape[0])
    coefficients = np.zeros((num_faces * num_reflectors,), dtype=complex_dtype)
    delays = np.zeros((num_faces * num_reflectors,), dtype=float)
    valid = np.zeros((num_faces * num_reflectors,), dtype=bool)

    for active_set in active_sets:
        reflector_index = int(active_set.reflector_index)
        if reflector_index < 0 or reflector_index >= num_reflectors:
            continue
        active_indices = np.asarray(active_set.active_indices, dtype=np.int64)
        if active_indices.size == 0:
            continue

        path_offset = reflector_index * num_faces
        reflector_point = reflectors.centroids[reflector_index]
        reflector_normal = _normalize(reflectors.normals[reflector_index])
        reflector_vertices = _reflector_support(reflectors, reflector_index)
        material = reflectors.materials[reflector_index]
        specular_scale = (
            1.0
            if reflectors.specular_amplitude_scales is None
            else float(reflectors.specular_amplitude_scales[reflector_index])
        )

        if order == "human_env":
            virtual_endpoint = _mirror_point(
                rx_position,
                reflector_point,
                reflector_normal,
            )
            reflection_points, finite = _reflection_points_on_surface(
                centroids[active_indices],
                virtual_endpoint,
                reflector_point,
                reflector_normal,
                reflector_vertices,
            )
            tx_for_po = tx_position
            rx_for_po = virtual_endpoint
            tx_array_for_po = tx_positions
            rx_array_for_po = _mirror_points(
                rx_positions,
                reflector_point,
                reflector_normal,
            )
        elif order == "env_human":
            virtual_endpoint = _mirror_point(
                tx_position,
                reflector_point,
                reflector_normal,
            )
            reflection_points, finite = _reflection_points_on_surface(
                virtual_endpoint,
                centroids[active_indices],
                reflector_point,
                reflector_normal,
                reflector_vertices,
            )
            tx_for_po = virtual_endpoint
            rx_for_po = rx_position
            tx_array_for_po = _mirror_points(
                tx_positions,
                reflector_point,
                reflector_normal,
            )
            rx_array_for_po = rx_positions
        else:
            raise ValueError("order must be 'human_env' or 'env_human'")

        if static_visibility_scene is not None:
            direct = tx_position if order == "human_env" else rx_position
            reflected = rx_position if order == "human_env" else tx_position
            finite &= _segments_clear_in_scene(
                starts=direct, ends=centroids[active_indices], scene=static_visibility_scene
            )
            finite &= _segments_clear_in_scene(
                starts=centroids[active_indices],
                ends=reflection_points,
                scene=static_visibility_scene,
            )
            finite &= _segments_clear_in_scene(
                starts=reflected, ends=reflection_points, scene=static_visibility_scene
            )
        current_indices = active_indices[finite]
        if current_indices.size == 0:
            continue
        contrib, tau = _coupling_face_contributions(
            incremental_state=incremental_state,
            cache_key=(order, int(reflector_index)),
            vertices=vertices,
            faces=faces,
            po_config=po_config,
            face_indices=face_indices[current_indices],
            centroids=centroids,
            normals=normals,
            areas=areas,
            tx_position=tx_for_po,
            rx_position=rx_for_po,
            tx_positions=tx_array_for_po,
            rx_positions=rx_array_for_po,
            frequency_hz=frequency_hz,
            material=human_material,
            backend=backend,
            precision=precision,
            polarization_axis=polarization_axis,
            reflector_normal=reflector_normal,
            reflector_material=material,
            order=order,
            reflection_scale=reflection_scale * specular_scale,
            reflector_point=reflector_point,
            support_triangles=reflector_vertices,
            delay_response=delay_response,
            path_offset=path_offset,
            static_visibility_scene=static_visibility_scene,
        )
        out_indices = path_offset + current_indices
        coefficients[out_indices] = contrib
        delays[out_indices] = tau
        valid[out_indices] = np.isfinite(tau) & (np.abs(coefficients[out_indices]) > 0.0)

    return coefficients, delays, valid


def _coupling_for_pairs_active_sets(
    centroids: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    face_indices: np.ndarray,
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    frequency_hz: float,
    human_material: PhysicalOpticsMaterial,
    reflectors: EnvironmentReflectorBank,
    *,
    order: str,
    reflection_scale: float,
    active_sets: tuple[_CouplingActiveSet, ...],
    center_coefficients: np.ndarray,
    virtual_channel_tx_indices: np.ndarray,
    virtual_channel_rx_indices: np.ndarray,
    precision: ComputePrecision,
) -> tuple[np.ndarray, np.ndarray]:
    """Projects phase-center coupling coefficients to virtual channels.

    Approximation: reflector feasibility and human PO scattering are taken from
    the phase-center active set/coefficients. Per-VC coupling keeps the image
    geometry, carrier phase, spreading, and delay, but does not recompute PO
    currents or visibility for each Tx/Rx pair.
    """

    del reflection_scale
    _, complex_dtype = _numpy_dtypes_for_precision(precision)
    num_faces = face_indices.size
    num_reflectors = int(reflectors.vertices.shape[0])
    tx_positions = np.asarray(tx_positions, dtype=np.float64).reshape(-1, 3)
    rx_positions = np.asarray(rx_positions, dtype=np.float64).reshape(-1, 3)
    vc_tx, vc_rx = resolve_virtual_channel_pairs(
        int(tx_positions.shape[0]),
        int(rx_positions.shape[0]),
        tx_indices=virtual_channel_tx_indices,
        rx_indices=virtual_channel_rx_indices,
    )
    pair_tx = tx_positions[vc_tx]
    pair_rx = rx_positions[vc_rx]
    tx_ids, tx_inverse = np.unique(vc_tx, return_inverse=True)
    rx_ids, rx_inverse = np.unique(vc_rx, return_inverse=True)
    num_pairs = int(pair_tx.shape[0])
    coefficients = np.zeros(
        (num_pairs, num_faces * num_reflectors),
        dtype=complex_dtype,
    )
    delays = np.zeros((num_pairs, num_faces * num_reflectors), dtype=float)
    center_coefficients = np.asarray(
        center_coefficients,
        dtype=complex_dtype,
    ).reshape(-1)
    tx_center = np.mean(tx_positions, axis=0)
    rx_center = np.mean(rx_positions, axis=0)
    k0 = 2.0 * pi * frequency_hz / c

    for active_set in active_sets:
        reflector_index = int(active_set.reflector_index)
        if reflector_index < 0 or reflector_index >= num_reflectors:
            continue
        active_indices = np.asarray(active_set.active_indices, dtype=np.int64)
        if active_indices.size == 0:
            continue

        path_offset = reflector_index * num_faces
        reflector_point = reflectors.centroids[reflector_index]
        reflector_normal = _normalize(reflectors.normals[reflector_index])
        reflector_vertices = _reflector_support(reflectors, reflector_index)

        face_points = centroids[active_indices]
        if order == "human_env":
            virtual_endpoints = _mirror_points(
                rx_positions[rx_ids],
                reflector_point,
                reflector_normal,
            )
            endpoint_inverse = rx_inverse
            starts = np.tile(face_points, (rx_ids.size, 1))
            ends = np.repeat(virtual_endpoints, active_indices.size, axis=0)
            tx_for_po = pair_tx
            rx_for_po = virtual_endpoints[rx_inverse]
            center_tx_for_po = tx_center
            center_rx_for_po = _mirror_point(
                rx_center,
                reflector_point,
                reflector_normal,
            )
        elif order == "env_human":
            virtual_endpoints = _mirror_points(
                tx_positions[tx_ids],
                reflector_point,
                reflector_normal,
            )
            endpoint_inverse = tx_inverse
            starts = np.repeat(virtual_endpoints, active_indices.size, axis=0)
            ends = np.tile(face_points, (tx_ids.size, 1))
            tx_for_po = virtual_endpoints[tx_inverse]
            rx_for_po = pair_rx
            center_tx_for_po = _mirror_point(
                tx_center,
                reflector_point,
                reflector_normal,
            )
            center_rx_for_po = rx_center
        else:
            raise ValueError("order must be 'human_env' or 'env_human'")

        _, finite = _reflection_points_on_surface(
            starts,
            ends,
            reflector_point,
            reflector_normal,
            reflector_vertices,
        )
        # Finite-surface feasibility depends only on the reflected physical
        # endpoint. Restore virtual-channel order before applying the mask.
        finite = finite.reshape(virtual_endpoints.shape[0], active_indices.size)[endpoint_inverse]
        if not np.any(finite):
            continue

        tx_dist = np.linalg.norm(
            tx_for_po[:, None, :] - face_points[None, :, :],
            axis=2,
        )
        rx_dist = np.linalg.norm(
            rx_for_po[:, None, :] - face_points[None, :, :],
            axis=2,
        )
        path_lengths = tx_dist + rx_dist
        spreads = np.maximum(tx_dist * rx_dist, 1e-18)
        center_tx_dist = np.linalg.norm(
            center_tx_for_po[None, :] - face_points,
            axis=1,
        )
        center_rx_dist = np.linalg.norm(
            center_rx_for_po[None, :] - face_points,
            axis=1,
        )
        center_lengths = center_tx_dist + center_rx_dist
        center_spreads = np.maximum(center_tx_dist * center_rx_dist, 1e-18)
        out_indices = path_offset + active_indices
        # Reuse the center coupling coefficient as the scattering/reflection
        # amplitude; only the geometric array phase, spreading, and tau_vc vary.
        # Physical channel coefficients use negative propagation phase.
        projected = (
            center_coefficients[out_indices][None, :]
            * np.exp(-1j * k0 * (path_lengths - center_lengths[None, :]))
            * (center_spreads[None, :] / np.maximum(spreads, 1e-18))
        ).astype(complex_dtype)
        coefficients[:, out_indices] = np.where(
            finite,
            projected,
            np.zeros((), dtype=complex_dtype),
        )
        delays[:, out_indices] = path_lengths / c

    return coefficients, delays


def _segments_clear_in_scene(
    *,
    starts: np.ndarray,
    ends: np.ndarray,
    scene,
    epsilon_m: float = 1e-5,
) -> np.ndarray:
    """Tests whether line segments are unoccluded in a Mitsuba scene."""

    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    if starts.ndim == 1:
        starts = np.broadcast_to(starts.reshape(1, 3), ends.shape)
    if ends.ndim == 1:
        ends = np.broadcast_to(ends.reshape(1, 3), starts.shape)
    vectors = ends - starts
    distances = np.linalg.norm(vectors, axis=1)
    clear = distances <= 2.0 * float(epsilon_m)
    candidate_indices = np.flatnonzero(~clear)
    if candidate_indices.size == 0:
        return clear

    directions = vectors[candidate_indices] / distances[candidate_indices, None]
    origins = starts[candidate_indices] + float(epsilon_m) * directions
    maxt = np.maximum(distances[candidate_indices] - 2.0 * float(epsilon_m), 0.0)
    ray = mi.Ray3f(
        o=mi.Point3f(origins[:, 0], origins[:, 1], origins[:, 2]),
        d=mi.Vector3f(directions[:, 0], directions[:, 1], directions[:, 2]),
        maxt=mi.Float(maxt),
        time=0.0,
        wavelengths=mi.Color0f(),
    )
    hit = np.asarray(scene.ray_intersect_preliminary(ray).is_valid(), dtype=bool)
    clear[candidate_indices] = ~hit
    return clear


def _reflection_points_on_surface(
    starts: np.ndarray,
    ends: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect an image segment with a plane and its finite triangle union."""

    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    if starts.ndim == 1:
        starts = np.broadcast_to(starts.reshape(1, 3), ends.shape)
    if ends.ndim == 1:
        ends = np.broadcast_to(ends.reshape(1, 3), starts.shape)
    direction = ends - starts
    denom = np.einsum("ij,j->i", direction, plane_normal)
    valid = np.abs(denom) > 1e-12
    t = np.zeros(starts.shape[0], dtype=np.float64)
    t[valid] = (
        np.einsum("j,ij->i", plane_normal, plane_point[None, :] - starts)[valid] / denom[valid]
    )
    valid &= (t > 1e-9) & (t < 1.0 - 1e-9)
    points = starts + t[:, None] * direction
    inside = np.zeros(valid.shape, dtype=bool)
    triangles = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
    candidates = np.flatnonzero(valid)
    if candidates.size == 0:
        return points, valid
    # A densely tessellated floor should not require every integration point
    # to be compared with every triangle. Query a point tree by triangle bounds.
    point_tree = (
        cKDTree(points[candidates]) if len(triangles) > 32 and len(candidates) > 128 else None
    )
    for triangle in triangles:
        lower, upper = triangle.min(axis=0) - 1e-5, triangle.max(axis=0) + 1e-5
        current = candidates
        if point_tree is not None:
            center = 0.5 * (lower + upper)
            radius = np.linalg.norm(0.5 * (upper - lower))
            current = candidates[point_tree.query_ball_point(center, radius)]
        current = current[~inside[current]]
        if current.size:
            current = current[
                np.all((points[current] >= lower) & (points[current] <= upper), axis=1)
            ]
        if current.size:
            inside[current] = _points_in_triangle(points[current], triangle)
    valid &= inside
    return points, valid


def _points_in_triangle(points: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """Returns a barycentric inside-triangle mask for candidate points."""

    points = np.asarray(points, dtype=np.float64)
    tri = np.asarray(triangle, dtype=np.float64)
    v0 = tri[1] - tri[0]
    v1 = tri[2] - tri[0]
    v2 = points - tri[0][None, :]
    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot11 = float(np.dot(v1, v1))
    dot02 = np.einsum("j,ij->i", v0, v2)
    dot12 = np.einsum("j,ij->i", v1, v2)
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) <= 1e-18:
        return np.zeros(points.shape[0], dtype=bool)
    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    tol = 1e-7
    return (u >= -tol) & (v >= -tol) & (u + v <= 1.0 + tol)


def _mirror_point(point: np.ndarray, plane_point: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Mirrors a point across a plane."""

    point = np.asarray(point, dtype=np.float64).reshape(3)
    normal = _normalize(normal)
    return point - 2.0 * np.dot(point - plane_point, normal) * normal


def _mirror_points(
    points: np.ndarray,
    plane_point: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    """Mirrors multiple points across a plane."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    normal = _normalize(normal)
    offsets = np.einsum("ij,j->i", points - plane_point, normal)
    return points - 2.0 * offsets[:, None] * normal


def _normalize(vector: np.ndarray) -> np.ndarray:
    """Returns a unit 3-vector, using boresight for near-zero vectors."""

    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(vector)
    if norm <= 1e-18:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return vector / norm


def _empty_reflector_bank() -> EnvironmentReflectorBank:
    """Returns an empty reflector bank with correctly typed arrays."""

    return EnvironmentReflectorBank(
        vertices=np.zeros((0, 3, 3), dtype=np.float64),
        centroids=np.zeros((0, 3), dtype=np.float64),
        normals=np.zeros((0, 3), dtype=np.float64),
        areas=np.zeros((0,), dtype=np.float64),
        materials=(),
        object_names=(),
        face_indices=np.zeros((0,), dtype=np.int64),
        path_indices=np.zeros((0,), dtype=np.int64),
        reflection_gains=np.zeros((0,), dtype=np.complex128),
        specular_amplitude_scales=np.zeros((0,), dtype=np.float64),
        support_triangles=(),
    )


def _empty_coupling_result(
    num_virtual_channels: int,
    precision: ComputePrecision,
) -> EnvHumanCouplingResult:
    """Returns an empty coupling result for the requested channel count."""

    _, complex_dtype = _numpy_dtypes_for_precision(precision)
    runtime_profile_s = {
        "coupling_setup": 0.0,
        "human_env_channel": 0.0,
        "env_human_channel": 0.0,
        "coupling_assembly": 0.0,
        "coupling_total": 0.0,
    }
    return EnvHumanCouplingResult(
        human_env_coefficients=np.zeros(
            (num_virtual_channels, 0),
            dtype=complex_dtype,
        ),
        human_env_delays_s=np.zeros((0,), dtype=float),
        human_env_valid=np.zeros((0,), dtype=bool),
        env_human_coefficients=np.zeros(
            (num_virtual_channels, 0),
            dtype=complex_dtype,
        ),
        env_human_delays_s=np.zeros((0,), dtype=float),
        env_human_valid=np.zeros((0,), dtype=bool),
        runtime_profile_s=runtime_profile_s,
        runtime_profile_counts=dict.fromkeys(runtime_profile_s, 0),
    )
