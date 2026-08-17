# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Single-bounce static-environment/human PO coupling helpers."""

from __future__ import annotations

from dataclasses import dataclass
import time

import mitsuba as mi
import numpy as np
from scipy.constants import c, pi

from sionna.rt.constants import InteractionType, INVALID_PRIMITIVE, INVALID_SHAPE

from ..radar.hardware import resolve_virtual_channel_pairs
from .physical_optics import (
    ComputeBackend,
    ComputePrecision,
    HumanVisibilityScene,
    PhysicalOpticsMaterial,
    _coerce_material,
    _numpy_dtypes_for_precision,
    _samples_visible_from_origin,
    face_geometry,
    fresnel_coefficients,
    vector_facet_face_contributions,
)


@dataclass(frozen=True)
class EnvironmentReflectorBank:
    """Static reflector triangles used for Phase 2C image-method coupling."""

    vertices: np.ndarray
    centroids: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    materials: tuple[PhysicalOpticsMaterial, ...]
    object_names: tuple[str, ...]
    face_indices: np.ndarray
    path_indices: np.ndarray | None = None
    reflection_gains: np.ndarray | None = None


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


@dataclass(frozen=True)
class _CouplingActiveSet:
    """Phase-center feasible human face paths for one reflector."""

    reflector_index: int
    active_indices: np.ndarray


def extract_environment_reflector_bank(
    scene,
    *,
    max_reflectors: int = 16,
    min_area_m2: float = 1e-4,
) -> EnvironmentReflectorBank:
    """Extracts the largest static scene triangles as reflector candidates."""

    if max_reflectors <= 0:
        return _empty_reflector_bank()
    if min_area_m2 < 0.0:
        raise ValueError("min_area_m2 must be non-negative")

    candidates = []
    mesh_cache = {}
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
        for face_index, (face, centroid, normal, area) in enumerate(
            zip(faces, centroids, normals, areas)
        ):
            if not np.isfinite(area) or area < float(min_area_m2):
                continue
            tri = vertices[np.asarray(face, dtype=np.int64)]
            candidates.append(
                (
                    float(area),
                    tri,
                    centroid,
                    normal,
                    material,
                    str(object_name),
                    int(face_index),
                )
            )

    if not candidates:
        return _empty_reflector_bank()

    candidates.sort(key=lambda item: item[0], reverse=True)
    candidates = candidates[:int(max_reflectors)]
    return EnvironmentReflectorBank(
        vertices=np.asarray([item[1] for item in candidates], dtype=np.float64),
        centroids=np.asarray([item[2] for item in candidates], dtype=np.float64),
        normals=np.asarray([item[3] for item in candidates], dtype=np.float64),
        areas=np.asarray([item[0] for item in candidates], dtype=np.float64),
        materials=tuple(item[4] for item in candidates),
        object_names=tuple(item[5] for item in candidates),
        face_indices=np.asarray([item[6] for item in candidates], dtype=np.int64),
        path_indices=np.full(len(candidates), -1, dtype=np.int64),
        reflection_gains=np.ones((len(candidates),), dtype=np.complex128),
    )



def extract_rt_reflector_bank(
    scene,
    static_bank,
    *,
    max_reflectors: int = 0,
    min_area_m2: float = 1e-4,
    frequency_hz: float | None = None,
) -> EnvironmentReflectorBank:
    """Extracts single-specular reflector candidates from static RT paths.

    ``max_reflectors=0`` means no explicit cap. Negative values return an empty
    bank and are useful for disabled coupling call sites. When ``frequency_hz``
    is provided, each reflector gets an RT-calibrated complex reflection gain
    derived from the corresponding static single-specular path coefficient.
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
        if not np.isfinite(area) or area < float(min_area_m2):
            continue
        normal = normal / max(2.0 * area, 1e-18)
        hit_point = vertices[depth_idx, path_idx]
        if not np.all(np.isfinite(hit_point)):
            continue
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
                area,
                material,
                object_name,
                primitive_index,
                reflection_gain,
            )
        )

    if not candidates:
        return _empty_reflector_bank()
    if max_reflectors > 0:
        candidates = candidates[:int(max_reflectors)]
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
    )



def _rt_calibrated_reflection_gain(
    static_bank,
    path_idx: int,
    *,
    frequency_hz: float | None,
) -> complex:
    """Estimates a reflector gain by dividing RT coefficient by free-space loss."""

    if frequency_hz is None:
        return 1.0 + 0.0j
    coefficients = np.asarray(static_bank.coefficients, dtype=np.complex128)
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
    phase = np.exp(1j * 2.0 * pi * float(path_length_m) / wavelength_m)
    return complex(amplitude * phase)

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
) -> EnvHumanCouplingResult:
    """Computes Tx-human-env-Rx and Tx-env-human-Rx channel candidates."""

    total_t0 = time.perf_counter()
    runtime_profile_s = {
        "coupling_setup": 0.0,
        "human_env_channel": 0.0,
        "env_human_channel": 0.0,
        "coupling_assembly": 0.0,
        "coupling_total": 0.0,
    }
    runtime_profile_counts = dict.fromkeys(runtime_profile_s, 0)

    _, complex_dtype = _numpy_dtypes_for_precision(precision)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.uint32)
    tx_positions = np.asarray(tx_positions, dtype=np.float64)
    rx_positions = np.asarray(rx_positions, dtype=np.float64)
    vc_tx, vc_rx = resolve_virtual_channel_pairs(
        int(tx_positions.shape[0]),
        int(rx_positions.shape[0]),
        order=virtual_channel_order,
        tx_indices=virtual_channel_tx_indices,
        rx_indices=virtual_channel_rx_indices,
    )
    num_vc = int(vc_tx.size)
    num_reflectors = int(reflectors.vertices.shape[0])
    num_faces = int(faces.shape[0])
    num_paths = num_reflectors * num_faces
    if num_paths == 0:
        result = _empty_coupling_result(num_vc, precision)
        result.runtime_profile_s["coupling_total"] = time.perf_counter() - total_t0
        result.runtime_profile_counts["coupling_total"] = 1
        return result

    setup_t0 = time.perf_counter()
    centroids, normals, areas = face_geometry(vertices, faces)
    face_indices = np.arange(num_faces, dtype=np.int64)
    human_material = _coerce_material(human_material)
    tx_center = np.mean(tx_positions, axis=0)
    rx_center = np.mean(rx_positions, axis=0)
    runtime_profile_s["coupling_setup"] = time.perf_counter() - setup_t0
    runtime_profile_counts["coupling_setup"] = 1

    def timed_coupling_for_pair(order, tx_position, rx_position):
        """Runs one uncached coupling solve and records its runtime bucket."""

        pair_t0 = time.perf_counter()
        out = _coupling_for_pair(
            centroids,
            normals,
            areas,
            face_indices,
            tx_position,
            rx_position,
            frequency_hz,
            human_material,
            reflectors,
            human_visibility_scene=human_visibility_scene,
            static_visibility_scene=static_visibility_scene,
            order=order,
            reflection_scale=reflection_scale,
            backend=backend,
            precision=precision,
        )
        key = f"{order}_channel"
        runtime_profile_s[key] += time.perf_counter() - pair_t0
        runtime_profile_counts[key] += 1
        return out

    def timed_cached_coupling_for_pair(
        order,
        tx_position,
        rx_position,
        active_sets,
    ):
        """Runs one active-set coupling solve and records its runtime bucket."""

        pair_t0 = time.perf_counter()
        out = _coupling_for_pair_active_sets(
            centroids,
            normals,
            areas,
            face_indices,
            tx_position,
            rx_position,
            frequency_hz,
            human_material,
            reflectors,
            order=order,
            reflection_scale=reflection_scale,
            active_sets=active_sets,
            backend=backend,
            precision=precision,
        )
        key = f"{order}_channel"
        runtime_profile_s[key] += time.perf_counter() - pair_t0
        runtime_profile_counts[key] += 1
        return out

    def timed_cached_coupling_for_pairs(order, active_sets, center_coefficients):
        """Runs all-channel active-set coupling and records its runtime bucket."""

        pair_t0 = time.perf_counter()
        out = _coupling_for_pairs_active_sets(
            centroids,
            normals,
            areas,
            face_indices,
            tx_positions,
            rx_positions,
            frequency_hz,
            human_material,
            reflectors,
            order=order,
            reflection_scale=reflection_scale,
            active_sets=active_sets,
            center_coefficients=center_coefficients,
            virtual_channel_tx_indices=vc_tx,
            virtual_channel_rx_indices=vc_rx,
            precision=precision,
        )
        key = f"{order}_channel"
        runtime_profile_s[key] += time.perf_counter() - pair_t0
        runtime_profile_counts[key] += 1
        return out

    if human_env_active_sets is None:
        (
            human_env_center_a,
            human_env_tau,
            human_env_valid,
            human_env_active_sets,
        ) = timed_coupling_for_pair("human_env", tx_center, rx_center)
    else:
        (
            human_env_center_a,
            human_env_tau,
            human_env_valid,
        ) = timed_cached_coupling_for_pair(
            "human_env",
            tx_center,
            rx_center,
            human_env_active_sets,
        )
    if env_human_active_sets is None:
        (
            env_human_center_a,
            env_human_tau,
            env_human_valid,
            env_human_active_sets,
        ) = timed_coupling_for_pair("env_human", tx_center, rx_center)
    else:
        (
            env_human_center_a,
            env_human_tau,
            env_human_valid,
        ) = timed_cached_coupling_for_pair(
            "env_human",
            tx_center,
            rx_center,
            env_human_active_sets,
        )

    assembly_t0 = time.perf_counter()
    human_env_a = np.zeros((num_vc, num_paths), dtype=complex_dtype)
    env_human_a = np.zeros((num_vc, num_paths), dtype=complex_dtype)
    if (
        num_vc == 1
        and np.allclose(tx_positions[0], tx_center)
        and np.allclose(rx_positions[0], rx_center)
    ):
        human_env_a[0] = human_env_center_a
        env_human_a[0] = env_human_center_a
        runtime_profile_s["coupling_assembly"] += time.perf_counter() - assembly_t0
        runtime_profile_counts["coupling_assembly"] += 1
        runtime_profile_s["coupling_total"] = time.perf_counter() - total_t0
        runtime_profile_counts["coupling_total"] = 1
        return EnvHumanCouplingResult(
            human_env_coefficients=human_env_a,
            human_env_delays_s=human_env_tau,
            human_env_valid=human_env_valid,
            env_human_coefficients=env_human_a,
            env_human_delays_s=env_human_tau,
            env_human_valid=env_human_valid,
            runtime_profile_s=runtime_profile_s,
            runtime_profile_counts=runtime_profile_counts,
            human_env_active_sets=human_env_active_sets,
            env_human_active_sets=env_human_active_sets,
        )
    runtime_profile_s["coupling_assembly"] += time.perf_counter() - assembly_t0
    runtime_profile_counts["coupling_assembly"] += 1

    human_env_a[:], human_env_tau_vc = timed_cached_coupling_for_pairs(
        "human_env",
        human_env_active_sets,
        human_env_center_a,
    )
    env_human_a[:], env_human_tau_vc = timed_cached_coupling_for_pairs(
        "env_human",
        env_human_active_sets,
        env_human_center_a,
    )

    runtime_profile_s["coupling_total"] = time.perf_counter() - total_t0
    runtime_profile_counts["coupling_total"] = 1
    return EnvHumanCouplingResult(
        human_env_coefficients=human_env_a,
        human_env_delays_s=human_env_tau_vc,
        human_env_valid=human_env_valid,
        env_human_coefficients=env_human_a,
        env_human_delays_s=env_human_tau_vc,
        env_human_valid=env_human_valid,
        runtime_profile_s=runtime_profile_s,
        runtime_profile_counts=runtime_profile_counts,
        human_env_active_sets=human_env_active_sets,
        env_human_active_sets=env_human_active_sets,
    )


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

    if order == "human_env":
        real_leg_clear = _samples_visible_from_origin(
            points=centroids,
            normals=normals,
            origin=tx_position,
            scene=human_visibility_scene.scene,
            static_scene=static_visibility_scene,
        )
    elif order == "env_human":
        real_leg_clear = _segments_clear_in_scene(
            starts=rx_position,
            ends=centroids,
            scene=human_visibility_scene.scene,
        )
        if static_visibility_scene is not None:
            real_leg_clear &= _segments_clear_in_scene(
                starts=rx_position,
                ends=centroids,
                scene=static_visibility_scene,
            )
    else:
        raise ValueError("order must be 'human_env' or 'env_human'")

    for reflector_index in range(num_reflectors):
        path_offset = reflector_index * num_faces
        reflector_point = reflectors.centroids[reflector_index]
        reflector_normal = _normalize(reflectors.normals[reflector_index])
        reflector_vertices = reflectors.vertices[reflector_index]
        material = reflectors.materials[reflector_index]

        if order == "human_env":
            virtual_endpoint = _mirror_point(
                rx_position,
                reflector_point,
                reflector_normal,
            )
            reflection_points, finite = _reflection_points_on_triangle(
                centroids,
                virtual_endpoint,
                reflector_point,
                reflector_normal,
                reflector_vertices,
            )
            tx_for_po = tx_position
            rx_for_po = virtual_endpoint
            virtual_origin = virtual_endpoint
        else:
            virtual_endpoint = _mirror_point(
                tx_position,
                reflector_point,
                reflector_normal,
            )
            reflection_points, finite = _reflection_points_on_triangle(
                virtual_endpoint,
                centroids,
                reflector_point,
                reflector_normal,
                reflector_vertices,
            )
            tx_for_po = virtual_endpoint
            rx_for_po = rx_position
            virtual_origin = virtual_endpoint

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
        active_indices = candidate_indices[
            face_to_reflector_clear
            & radar_to_reflector_clear
        ]
        if active_indices.size == 0:
            continue
        active_sets.append(
            _CouplingActiveSet(
                reflector_index=int(reflector_index),
                active_indices=active_indices.astype(np.int64, copy=True),
            )
        )

        contrib, tau = vector_facet_face_contributions(
            face_indices=face_indices[active_indices],
            centroids=centroids,
            normals=normals,
            areas=areas,
            tx_position=tx_for_po,
            rx_position=rx_for_po,
            frequency_hz=frequency_hz,
            material=human_material,
            weights=np.ones(active_indices.size, dtype=float),
            backend=backend,
            precision=precision,
        )

        reflection_gain = _reflector_gain(reflectors, reflector_index)
        refl = _reflection_amplitude(
            reflection_points=reflection_points[active_indices],
            face_points=centroids[active_indices],
            reflector_normal=reflector_normal,
            reflector_area=float(reflectors.areas[reflector_index]),
            material=material,
            frequency_hz=frequency_hz,
            scale=reflection_scale,
            calibrated_gain=reflection_gain,
        )
        contrib = contrib * refl.astype(contrib.dtype, copy=False)
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
    backend: ComputeBackend,
    precision: ComputePrecision,
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
        reflector_vertices = reflectors.vertices[reflector_index]
        material = reflectors.materials[reflector_index]

        if order == "human_env":
            virtual_endpoint = _mirror_point(
                rx_position,
                reflector_point,
                reflector_normal,
            )
            reflection_points, finite = _reflection_points_on_triangle(
                centroids[active_indices],
                virtual_endpoint,
                reflector_point,
                reflector_normal,
                reflector_vertices,
            )
            tx_for_po = tx_position
            rx_for_po = virtual_endpoint
        elif order == "env_human":
            virtual_endpoint = _mirror_point(
                tx_position,
                reflector_point,
                reflector_normal,
            )
            reflection_points, finite = _reflection_points_on_triangle(
                virtual_endpoint,
                centroids[active_indices],
                reflector_point,
                reflector_normal,
                reflector_vertices,
            )
            tx_for_po = virtual_endpoint
            rx_for_po = rx_position
        else:
            raise ValueError("order must be 'human_env' or 'env_human'")

        current_indices = active_indices[finite]
        if current_indices.size == 0:
            continue
        current_reflection_points = reflection_points[finite]
        contrib, tau = vector_facet_face_contributions(
            face_indices=face_indices[current_indices],
            centroids=centroids,
            normals=normals,
            areas=areas,
            tx_position=tx_for_po,
            rx_position=rx_for_po,
            frequency_hz=frequency_hz,
            material=human_material,
            weights=np.ones(current_indices.size, dtype=float),
            backend=backend,
            precision=precision,
        )

        reflection_gain = _reflector_gain(reflectors, reflector_index)
        refl = _reflection_amplitude(
            reflection_points=current_reflection_points,
            face_points=centroids[current_indices],
            reflector_normal=reflector_normal,
            reflector_area=float(reflectors.areas[reflector_index]),
            material=material,
            frequency_hz=frequency_hz,
            scale=reflection_scale,
            calibrated_gain=reflection_gain,
        )
        out_indices = path_offset + current_indices
        coefficients[out_indices] = contrib * refl.astype(contrib.dtype, copy=False)
        delays[out_indices] = tau
        valid[out_indices] = np.isfinite(tau) & (
            np.abs(coefficients[out_indices]) > 0.0
        )

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
        reflector_vertices = reflectors.vertices[reflector_index]

        face_points = centroids[active_indices]
        if order == "human_env":
            virtual_endpoints = _mirror_points(
                pair_rx,
                reflector_point,
                reflector_normal,
            )
            starts = np.tile(face_points, (num_pairs, 1))
            ends = np.repeat(virtual_endpoints, active_indices.size, axis=0)
            tx_for_po = pair_tx
            rx_for_po = virtual_endpoints
            center_tx_for_po = tx_center
            center_rx_for_po = _mirror_point(
                rx_center,
                reflector_point,
                reflector_normal,
            )
        elif order == "env_human":
            virtual_endpoints = _mirror_points(
                pair_tx,
                reflector_point,
                reflector_normal,
            )
            starts = np.repeat(virtual_endpoints, active_indices.size, axis=0)
            ends = np.tile(face_points, (num_pairs, 1))
            tx_for_po = virtual_endpoints
            rx_for_po = pair_rx
            center_tx_for_po = _mirror_point(
                tx_center,
                reflector_point,
                reflector_normal,
            )
            center_rx_for_po = rx_center
        else:
            raise ValueError("order must be 'human_env' or 'env_human'")

        reflection_points, finite = _reflection_points_on_triangle(
            starts,
            ends,
            reflector_point,
            reflector_normal,
            reflector_vertices,
        )
        finite = finite.reshape(num_pairs, active_indices.size)
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
        # The phase sign matches the dechirped tx(t)*conj(rx(t)) ADC convention.
        projected = (
            center_coefficients[out_indices][None, :]
            * np.exp(1j * k0 * (path_lengths - center_lengths[None, :]))
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


def _reflection_points_on_triangle(
    starts: np.ndarray,
    ends: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    triangle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersects start/end segments with a reflector triangle plane."""

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
        np.einsum("j,ij->i", plane_normal, plane_point[None, :] - starts)[valid]
        / denom[valid]
    )
    valid &= (t > 1e-9) & (t < 1.0 - 1e-9)
    points = starts + t[:, None] * direction
    valid &= _points_in_triangle(points, triangle)
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



def _reflector_gain(reflectors: EnvironmentReflectorBank, reflector_index: int) -> complex | None:
    """Returns a finite calibrated gain for one reflector, when available."""

    gains = reflectors.reflection_gains
    if gains is None:
        return None
    gains = np.asarray(gains, dtype=np.complex128).reshape(-1)
    if reflector_index < 0 or reflector_index >= gains.size:
        return None
    gain = complex(gains[reflector_index])
    if not np.isfinite(gain.real) or not np.isfinite(gain.imag):
        return None
    return gain

def _reflection_amplitude(
    *,
    reflection_points: np.ndarray,
    face_points: np.ndarray,
    reflector_normal: np.ndarray,
    reflector_area: float,
    material: PhysicalOpticsMaterial,
    frequency_hz: float,
    scale: float,
    calibrated_gain: complex | None = None,
) -> np.ndarray:
    """Computes reflector amplitude from calibrated gain or Fresnel/aperture terms."""

    if calibrated_gain is not None:
        gain = complex(calibrated_gain)
        if np.isfinite(gain.real) and np.isfinite(gain.imag):
            # Hybrid coupling terms are synthesized as separate single
            # environment-reflected ADC components, so the calibrated static
            # reflector gain must retain the environment-bounce phase.
            return np.full(
                np.asarray(reflection_points).shape[0],
                float(scale) * gain,
                dtype=np.complex128,
            )

    direction = face_points - reflection_points
    direction_norm = np.linalg.norm(direction, axis=1)
    direction_hat = direction / np.maximum(direction_norm[:, None], 1e-18)
    cos_theta = np.abs(np.einsum("ij,j->i", direction_hat, reflector_normal))
    r_te, r_tm = fresnel_coefficients(cos_theta, material, frequency_hz)
    fresnel_amplitude = np.sqrt((np.abs(r_te) ** 2 + np.abs(r_tm) ** 2) * 0.5)
    projected_area = max(float(reflector_area), 0.0) * cos_theta
    solid_angle_fraction = np.clip(
        projected_area / (4.0 * np.pi * np.maximum(direction_norm ** 2, 1e-18)),
        0.0,
        1.0,
    )
    aperture_amplitude = np.sqrt(solid_angle_fraction)
    amplitude = fresnel_amplitude * aperture_amplitude
    return float(scale) * amplitude.astype(np.complex128)


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
