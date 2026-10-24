# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Physical-optics helpers for mmWave human sensing."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Literal

import mitsuba as mi
import numpy as np
from scipy.constants import c, epsilon_0, pi

from .options import (
    POArraySettings,
    POChannelSettings,
    POComputeSettings,
    POIntegrationSettings,
    POVisibilitySettings,
)
from ..radar.hardware import resolve_virtual_channel_pairs
from ..targets.mesh import ensure_outward_face_winding
from .compute import (
    ComputeBackend,
    ComputePrecision,
    TorchRuntime,
    numpy_dtypes_for_precision as _numpy_dtypes_for_precision,
    resolve_torch_runtime,
    torch_vector_cross,
    validate_compute_backend,
)
from .po_wideband import PODelayResponse

VisibilityMode = Literal["fractional_shadow_fade", "none"]
MaterialModel = Literal["slab", "interface", "pec"]
POIntegrationMode = Literal[
    "face_centroid",
    "parent_face_quadrature",
    "parent_face_adaptive_analytic",
    "parent_face_far_field_analytic",
]
PolarizationCallback = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def default_phase_span_scale_rad(po_integration_mode: str) -> float:
    """Returns the subdivision phase span suited to one PO integration mode.

    Midpoint quadrature samples the propagation phase once per subface, so a
    subface must stay well below a pi span. The analytic modes integrate the
    linear phase exactly; their span only bounds amplitude variation.
    """

    return 1.0 if po_integration_mode == "parent_face_quadrature" else 6.0


PHASE_SPAN_SAMPLE_BARYCENTRICS = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    ],
    dtype=np.float64,
)
PARENT_QUADRATURE_CONTRIBUTION_CHUNK_SIZE = 1_000_000


@dataclass(frozen=True)
class PhysicalOpticsMaterial:
    """Material parameters used by the facet PO model."""

    relative_permittivity: float
    conductivity_s_per_m: float
    thickness_m: float = 0.0
    model: MaterialModel = "slab"
    surface_attenuation_thickness_m: float = 0.0


@dataclass(frozen=True)
class FaceVisibilityState:
    """Per-face state used to smooth visibility transitions across chirps."""

    was_visible: np.ndarray
    fade_weights: np.ndarray
    latched_fractional_visibility: np.ndarray


@dataclass(frozen=True)
class POChannelResult:
    """Result of a human-only physical-optics channel evaluation."""

    coefficients: np.ndarray
    delays_s: np.ndarray
    face_indices: np.ndarray
    face_centroids: np.ndarray
    face_normals: np.ndarray
    face_areas: np.ndarray
    total_area: float
    effective_visible_area_fraction: float
    raw_fractional_visibility: np.ndarray
    latched_fractional_visibility: np.ndarray
    temporal_fade_weights: np.ndarray
    effective_weights: np.ndarray
    visible_sample_counts: np.ndarray
    sample_count_per_face: int
    phase_center_contributions: np.ndarray
    phase_center_channel: complex
    visibility_state: FaceVisibilityState
    far_field_max_edge_m: float | None = None
    far_field_nearest_distance_m: float | None = None
    delay_response: PODelayResponse | None = None


def complex_relative_permittivity(
    epsilon_r: float,
    conductivity_s_per_m: float,
    frequency_hz: float,
) -> complex:
    """Returns the complex relative permittivity for a lossy dielectric."""

    omega = 2.0 * pi * frequency_hz
    return complex(epsilon_r, -conductivity_s_per_m / (omega * epsilon_0))


def fresnel_interface_coefficients(
    cos_theta: np.ndarray,
    epsilon_r_complex: complex,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns TE/TM Fresnel coefficients for a dielectric interface."""

    cos_theta = np.clip(np.asarray(cos_theta, dtype=float), 0.0, 1.0)
    sin_theta_sqr = 1.0 - cos_theta * cos_theta
    a = np.sqrt(epsilon_r_complex - sin_theta_sqr + 0j)
    r_te = (cos_theta - a) / (cos_theta + a)
    r_tm = (epsilon_r_complex * cos_theta - a) / (epsilon_r_complex * cos_theta + a)
    return r_te, r_tm


def fresnel_slab_coefficients(
    cos_theta: np.ndarray,
    epsilon_r_complex: complex,
    thickness_m: float,
    wavelength_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns TE/TM Fresnel coefficients for a single-layer slab."""

    cos_theta = np.clip(np.asarray(cos_theta, dtype=float), 0.0, 1.0)
    if thickness_m <= 0.0:
        return fresnel_interface_coefficients(cos_theta, epsilon_r_complex)

    sin_theta_sqr = 1.0 - cos_theta * cos_theta
    a = np.sqrt(epsilon_r_complex - sin_theta_sqr + 0j)

    r_te_p = (cos_theta - a) / (cos_theta + a)
    r_tm_p = (epsilon_r_complex * cos_theta - a) / (epsilon_r_complex * cos_theta + a)

    q = 2.0 * pi * thickness_m / wavelength_m * np.sqrt(epsilon_r_complex - sin_theta_sqr + 0j)
    exp_j_2q = np.exp(-2.0j * q)

    r_te = r_te_p * (1.0 - exp_j_2q) / (1.0 - (r_te_p * r_te_p) * exp_j_2q)
    r_tm = r_tm_p * (1.0 - exp_j_2q) / (1.0 - (r_tm_p * r_tm_p) * exp_j_2q)
    return r_te, r_tm


def fresnel_coefficients(
    cos_theta: np.ndarray,
    material: PhysicalOpticsMaterial,
    frequency_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns TE/TM Fresnel coefficients for a PO material."""

    if material.model == "pec":
        cos_theta = np.asarray(cos_theta, dtype=float)
        return (
            -np.ones_like(cos_theta, dtype=np.complex128),
            np.ones_like(cos_theta, dtype=np.complex128),
        )

    epsilon_r_complex = complex_relative_permittivity(
        epsilon_r=material.relative_permittivity,
        conductivity_s_per_m=material.conductivity_s_per_m,
        frequency_hz=frequency_hz,
    )
    wavelength_m = c / frequency_hz
    if material.model == "slab":
        return fresnel_slab_coefficients(
            cos_theta=cos_theta,
            epsilon_r_complex=epsilon_r_complex,
            thickness_m=material.thickness_m,
            wavelength_m=wavelength_m,
        )
    if material.model == "interface":
        return fresnel_interface_coefficients(
            cos_theta=cos_theta,
            epsilon_r_complex=epsilon_r_complex,
        )
    raise ValueError(f"Unknown Fresnel material model: {material.model}")


def coherent_surface_attenuation(
    cos_theta: np.ndarray,
    material: PhysicalOpticsMaterial,
    frequency_hz: float,
) -> np.ndarray:
    """Returns a two-way coherent lossy-depth attenuation factor."""

    cos_theta = np.clip(np.asarray(cos_theta, dtype=float), 0.0, 1.0)
    if material.surface_attenuation_thickness_m <= 0.0:
        return np.ones_like(cos_theta, dtype=np.float64)

    epsilon_r_complex = complex_relative_permittivity(
        epsilon_r=material.relative_permittivity,
        conductivity_s_per_m=material.conductivity_s_per_m,
        frequency_hz=frequency_hz,
    )
    wavelength_m = c / frequency_hz
    k0 = 2.0 * pi / wavelength_m
    sin_theta_sqr = 1.0 - cos_theta * cos_theta
    kz_medium = k0 * np.sqrt(epsilon_r_complex - sin_theta_sqr + 0j)
    alpha_z = np.abs(np.imag(kz_medium))
    return np.exp(-2.0 * alpha_z * material.surface_attenuation_thickness_m)


def face_geometry(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    orient_outward: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns face centroids, outward normals, and areas."""

    if orient_outward:
        outward_faces = ensure_outward_face_winding(vertices, faces)
        if not np.array_equal(outward_faces, np.asarray(faces, dtype=np.uint32)):
            return face_geometry(vertices, outward_faces)

    tri = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    edge_1 = tri[:, 1] - tri[:, 0]
    edge_2 = tri[:, 2] - tri[:, 0]
    cross = np.cross(edge_1, edge_2)
    twice_area = np.linalg.norm(cross, axis=1)
    areas = 0.5 * twice_area
    normals = cross / np.maximum(twice_area[:, None], 1e-18)
    centroids = tri.mean(axis=1)

    return centroids, normals, areas


def parent_face_quadrature_geometry(
    vertices: np.ndarray,
    faces: np.ndarray,
    selected_parent_indices: np.ndarray,
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    frequency_hz: float,
    phase_span_scale_rad: float,
    *,
    max_refinement_depth: int = 16,
    max_subfaces_per_parent: int = 0,
    progress_callback: Callable[[dict], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns internal subface geometry for parent-face PO quadrature.

    The returned parent index array maps every emitted subface path back to the
    original mesh face whose visibility weight should be used. Geometry is kept
    piecewise-linear on the original parent triangle; this refines PO
    integration without changing the visibility mesh topology.
    """

    triangles, parent_indices = _adaptive_parent_triangles(
        vertices,
        faces,
        selected_parent_indices,
        tx_positions,
        rx_positions,
        frequency_hz,
        phase_span_scale_rad,
        max_refinement_depth=max_refinement_depth,
        max_subfaces_per_parent=max_subfaces_per_parent,
        progress_callback=progress_callback,
    )
    centroids, normals, areas = _triangle_geometry_from_triangles(triangles)
    return centroids, normals, areas, parent_indices


def parent_face_adaptive_analytic_geometry(
    vertices: np.ndarray,
    faces: np.ndarray,
    selected_parent_indices: np.ndarray,
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    frequency_hz: float,
    phase_span_scale_rad: float = 6.0,
    *,
    curvature_error_rad: float = 0.005,
    max_refinement_depth: int = 16,
    max_subfaces_per_parent: int = 0,
    progress_callback: Callable[[dict], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Refine by phase span OR edge-midpoint curvature, retaining parent ids.

    Each leaf is evaluated with the exact triangle integral of its locally
    linear propagation phase. Curvature uses k_c/3 times the largest midpoint
    path-length interpolation error over all edges and Tx/Rx combinations.
    """
    triangles, parent_indices = _adaptive_parent_triangles(
        vertices,
        faces,
        selected_parent_indices,
        tx_positions,
        rx_positions,
        frequency_hz,
        phase_span_scale_rad,
        curvature_error_rad=curvature_error_rad,
        max_refinement_depth=max_refinement_depth,
        max_subfaces_per_parent=max_subfaces_per_parent,
        progress_callback=progress_callback,
    )
    centroids, normals, areas = _triangle_geometry_from_triangles(triangles)
    return centroids, normals, areas, parent_indices, triangles


def _adaptive_parent_triangles(
    vertices,
    faces,
    selected_parent_indices,
    tx_positions,
    rx_positions,
    frequency_hz,
    phase_span_scale_rad,
    *,
    curvature_error_rad=None,
    max_refinement_depth=16,
    max_subfaces_per_parent=0,
    progress_callback=None,
):
    """Common four-child refinement; only the analytic mode tests curvature."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.uint32)
    selected_parent_indices = np.asarray(
        selected_parent_indices,
        dtype=np.int64,
    ).reshape(-1)
    if selected_parent_indices.size == 0:
        return np.zeros((0, 3, 3), dtype=np.float64), selected_parent_indices.copy()
    if not np.isfinite(phase_span_scale_rad) or phase_span_scale_rad <= 0.0:
        raise ValueError("phase_span_scale_rad must be finite and positive")
    if curvature_error_rad is not None and (
        not np.isfinite(curvature_error_rad) or curvature_error_rad <= 0.0
    ):
        raise ValueError("curvature_error_rad must be finite and positive")
    if max_refinement_depth < 0:
        raise ValueError("max_refinement_depth must be non-negative")
    if max_subfaces_per_parent < 0:
        raise ValueError("max_subfaces_per_parent must be non-negative")

    triangles = vertices[faces[selected_parent_indices]].astype(
        np.float64,
        copy=True,
    )
    parent_indices = selected_parent_indices.copy()
    num_faces = int(faces.shape[0])
    max_depth = int(max_refinement_depth)
    max_per_parent = int(max_subfaces_per_parent)
    stage_prefix = "quadrature" if curvature_error_rad is None else "adaptive_analytic"
    if progress_callback is not None:
        progress_callback(
            {
                "stage": f"{stage_prefix}_geometry_start",
                "parent_faces_processed": 0,
                "parent_faces_total": int(selected_parent_indices.size),
                "subfaces_total": int(triangles.shape[0]),
            }
        )

    def split_tests(current):
        span_function = (
            _triangle_phase_span_rad
            if curvature_error_rad is None
            else _triangle_linear_phase_span_rad
        )
        span = span_function(current, tx_positions, rx_positions, frequency_hz)
        split = span > float(phase_span_scale_rad)
        curvature = None
        if curvature_error_rad is not None:
            curvature = _triangle_phase_curvature_rad(
                current, tx_positions, rx_positions, frequency_hz
            )
            split |= curvature > float(curvature_error_rad)
        return split, span, curvature

    for depth in range(max_depth):
        split_mask, _, _ = split_tests(triangles)
        if max_per_parent > 0 and np.any(split_mask):
            current_counts = np.bincount(parent_indices, minlength=num_faces)
            split_counts = np.bincount(
                parent_indices[split_mask],
                minlength=num_faces,
            )
            allowed_parent = current_counts + 3 * split_counts <= max_per_parent
            split_mask &= allowed_parent[parent_indices]
        split_count = int(np.count_nonzero(split_mask))
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": f"{stage_prefix}_geometry_refinement",
                    "refinement_depth": int(depth),
                    "parent_faces_processed": (
                        int(
                            selected_parent_indices.size
                            - np.unique(parent_indices[split_mask]).size
                        )
                        if split_count
                        else int(selected_parent_indices.size)
                    ),
                    "parent_faces_total": int(selected_parent_indices.size),
                    "parent_faces_to_split": (
                        int(np.unique(parent_indices[split_mask]).size) if split_count else 0
                    ),
                    "subfaces_total": int(triangles.shape[0]),
                    "subfaces_to_split": split_count,
                }
            )
        if not np.any(split_mask):
            break
        keep_mask = ~split_mask
        split_triangles, split_parent_indices = _split_triangles_four(
            triangles[split_mask],
            parent_indices[split_mask],
        )
        triangles = np.concatenate(
            [triangles[keep_mask], split_triangles],
            axis=0,
        )
        parent_indices = np.concatenate(
            [parent_indices[keep_mask], split_parent_indices],
            axis=0,
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": f"{stage_prefix}_geometry_split",
                    "refinement_depth": int(depth + 1),
                    "parent_faces_processed": int(
                        selected_parent_indices.size - np.unique(split_parent_indices).size
                    ),
                    "parent_faces_total": int(selected_parent_indices.size),
                    "parent_faces_to_split": int(np.unique(split_parent_indices).size),
                    "subfaces_total": int(triangles.shape[0]),
                }
            )

    if progress_callback is not None:
        diagnostics = {}
        if curvature_error_rad is not None:
            unresolved, span, curvature = split_tests(triangles)
            diagnostics = {
                "unresolved_subfaces": int(np.count_nonzero(unresolved)),
                "unresolved_parent_faces": int(np.unique(parent_indices[unresolved]).size),
                "max_phase_span_rad": float(np.max(span, initial=0.0)),
                "max_curvature_error_rad": float(np.max(curvature, initial=0.0)),
            }
        progress_callback(
            {
                "stage": f"{stage_prefix}_geometry_done",
                "parent_faces_processed": int(selected_parent_indices.size),
                "parent_faces_total": int(selected_parent_indices.size),
                "subfaces_total": int(triangles.shape[0]),
                **diagnostics,
            }
        )
    return triangles, parent_indices


def parent_face_far_field_analytic_geometry(
    vertices: np.ndarray,
    faces: np.ndarray,
    selected_parent_indices: np.ndarray,
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    frequency_hz: float,
    *,
    max_refinement_depth: int = 16,
    max_subfaces_per_parent: int = 0,
    far_field_max_edge_m: float | None = None,
    far_field_nearest_distance_m: float | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    """Returns subface geometry for global far-field analytic PO.

    A single conservative maximum edge is computed from the nearest Tx/Rx
    distance to the mesh using the Fraunhofer criterion ``R >= 2 D^2/lambda``.
    Parent faces are midpoint-refined until every temporary subtriangle has
    max edge ``D <= sqrt(lambda * R_min / 2)``. The persistent path bank still
    remains one path per original parent face.
    """

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.uint32)
    selected_parent_indices = np.asarray(
        selected_parent_indices,
        dtype=np.int64,
    ).reshape(-1)
    if selected_parent_indices.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            selected_parent_indices.copy(),
            np.zeros((0, 3, 3), dtype=np.float64),
            0.0,
            0.0,
        )
    if max_refinement_depth < 0:
        raise ValueError("max_refinement_depth must be non-negative")
    if max_subfaces_per_parent < 0:
        raise ValueError("max_subfaces_per_parent must be non-negative")

    if far_field_max_edge_m is None:
        max_edge_m, nearest_distance_m = parent_face_far_field_edge_rule(
            vertices=vertices,
            faces=faces,
            tx_positions=tx_positions,
            rx_positions=rx_positions,
            frequency_hz=frequency_hz,
        )
    else:
        max_edge_m = float(far_field_max_edge_m)
        if max_edge_m <= 0.0:
            raise ValueError("far_field_max_edge_m must be positive")
        nearest_distance_m = (
            float("nan")
            if far_field_nearest_distance_m is None
            else float(far_field_nearest_distance_m)
        )
    triangles = vertices[faces[selected_parent_indices]].astype(
        np.float64,
        copy=True,
    )
    parent_indices = selected_parent_indices.copy()
    num_faces = int(faces.shape[0])
    max_depth = int(max_refinement_depth)
    max_per_parent = int(max_subfaces_per_parent)
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "far_field_geometry_start",
                "parent_faces_processed": 0,
                "parent_faces_total": int(selected_parent_indices.size),
                "subfaces_total": int(triangles.shape[0]),
                "nearest_mesh_distance_m": float(nearest_distance_m),
                "far_field_max_edge_m": float(max_edge_m),
            }
        )
    for depth in range(max_depth):
        max_edges = _triangle_max_edge_lengths(triangles)
        split_mask = max_edges > max_edge_m
        if max_per_parent > 0 and np.any(split_mask):
            current_counts = np.bincount(parent_indices, minlength=num_faces)
            split_counts = np.bincount(
                parent_indices[split_mask],
                minlength=num_faces,
            )
            allowed_parent = current_counts + 3 * split_counts <= max_per_parent
            split_mask &= allowed_parent[parent_indices]
        split_count = int(np.count_nonzero(split_mask))
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "far_field_geometry_refinement",
                    "refinement_depth": int(depth),
                    "parent_faces_processed": (
                        int(
                            selected_parent_indices.size
                            - np.unique(parent_indices[split_mask]).size
                        )
                        if split_count
                        else int(selected_parent_indices.size)
                    ),
                    "parent_faces_total": int(selected_parent_indices.size),
                    "parent_faces_to_split": (
                        int(np.unique(parent_indices[split_mask]).size) if split_count else 0
                    ),
                    "subfaces_total": int(triangles.shape[0]),
                    "subfaces_to_split": split_count,
                    "nearest_mesh_distance_m": float(nearest_distance_m),
                    "far_field_max_edge_m": float(max_edge_m),
                }
            )
        if not np.any(split_mask):
            break
        keep_mask = ~split_mask
        split_triangles, split_parent_indices = _split_triangles_four(
            triangles[split_mask],
            parent_indices[split_mask],
        )
        triangles = np.concatenate(
            [triangles[keep_mask], split_triangles],
            axis=0,
        )
        parent_indices = np.concatenate(
            [parent_indices[keep_mask], split_parent_indices],
            axis=0,
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "far_field_geometry_split",
                    "refinement_depth": int(depth + 1),
                    "parent_faces_processed": int(
                        selected_parent_indices.size - np.unique(split_parent_indices).size
                    ),
                    "parent_faces_total": int(selected_parent_indices.size),
                    "parent_faces_to_split": int(np.unique(split_parent_indices).size),
                    "subfaces_total": int(triangles.shape[0]),
                    "nearest_mesh_distance_m": float(nearest_distance_m),
                    "far_field_max_edge_m": float(max_edge_m),
                }
            )
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "far_field_geometry_done",
                "parent_faces_processed": int(selected_parent_indices.size),
                "parent_faces_total": int(selected_parent_indices.size),
                "subfaces_total": int(triangles.shape[0]),
                "nearest_mesh_distance_m": float(nearest_distance_m),
                "far_field_max_edge_m": float(max_edge_m),
                "max_subface_edge_m": float(
                    np.max(_triangle_max_edge_lengths(triangles), initial=0.0)
                ),
            }
        )
    centroids, normals, areas = _triangle_geometry_from_triangles(triangles)
    return (
        centroids,
        normals,
        areas,
        parent_indices,
        triangles,
        float(max_edge_m),
        float(nearest_distance_m),
    )


def parent_face_far_field_edge_rule(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    frequency_hz: float,
) -> tuple[float, float]:
    """Returns ``(D_max, R_min)`` for global far-field analytic subdivision."""

    source_positions = np.vstack(
        [
            np.asarray(tx_positions, dtype=np.float64).reshape(-1, 3),
            np.asarray(rx_positions, dtype=np.float64).reshape(-1, 3),
        ]
    )
    nearest_distance_m = _nearest_distance_to_triangle_mesh(
        vertices,
        faces,
        source_positions,
    )
    wavelength_m = c / float(frequency_hz)
    max_edge_m = math.sqrt(max(wavelength_m * max(nearest_distance_m, 1e-18) / 2.0, 0.0))
    return float(max_edge_m), float(nearest_distance_m)


def _triangle_geometry_from_triangles(
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns centroids, normals, and areas for explicit triangle vertices."""

    triangles = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    if triangles.shape[0] == 0:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )
    edge_1 = triangles[:, 1] - triangles[:, 0]
    edge_2 = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(edge_1, edge_2)
    twice_area = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(twice_area[:, None], 1e-18)
    areas = 0.5 * twice_area
    centroids = triangles.mean(axis=1)
    return centroids, normals, areas


def _split_triangles_four(
    triangles: np.ndarray,
    parent_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Splits each triangle into four midpoint child triangles."""

    triangles = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    parent_indices = np.asarray(parent_indices, dtype=np.int64).reshape(-1)
    if triangles.shape[0] != parent_indices.shape[0]:
        raise ValueError("parent_indices must match triangles")
    if triangles.shape[0] == 0:
        return triangles.copy(), parent_indices.copy()
    a = triangles[:, 0]
    b = triangles[:, 1]
    c_ = triangles[:, 2]
    ab = 0.5 * (a + b)
    bc = 0.5 * (b + c_)
    ca = 0.5 * (c_ + a)
    children = np.empty((triangles.shape[0] * 4, 3, 3), dtype=np.float64)
    children[0::4] = np.stack([a, ab, ca], axis=1)
    children[1::4] = np.stack([ab, b, bc], axis=1)
    children[2::4] = np.stack([ca, bc, c_], axis=1)
    children[3::4] = np.stack([ab, bc, ca], axis=1)
    return children, np.repeat(parent_indices, 4)


def _triangle_max_edge_lengths(triangles: np.ndarray) -> np.ndarray:
    triangles = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    if triangles.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.max(
        np.stack(
            [
                np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
                np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
                np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
            ],
            axis=1,
        ),
        axis=1,
    )


def _point_segment_distance_squared(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    segment = end - start
    denom = np.einsum("ij,ij->i", segment, segment)
    t = np.divide(
        np.einsum("ij,ij->i", point[None, :] - start, segment),
        denom,
        out=np.zeros_like(denom),
        where=denom > 1e-30,
    )
    t = np.clip(t, 0.0, 1.0)
    closest = start + t[:, None] * segment
    delta = point[None, :] - closest
    return np.einsum("ij,ij->i", delta, delta)


def _point_triangle_distance_squared(
    point: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    triangles = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    point = np.asarray(point, dtype=np.float64).reshape(3)
    if triangles.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    a = triangles[:, 0]
    b = triangles[:, 1]
    c_ = triangles[:, 2]
    ab = b - a
    ac = c_ - a
    normal = np.cross(ab, ac)
    normal_norm_sqr = np.einsum("ij,ij->i", normal, normal)
    signed_plane = np.divide(
        np.einsum("ij,ij->i", point[None, :] - a, normal),
        normal_norm_sqr,
        out=np.zeros_like(normal_norm_sqr),
        where=normal_norm_sqr > 1e-30,
    )
    projection = point[None, :] - signed_plane[:, None] * normal
    ap = projection - a
    d00 = np.einsum("ij,ij->i", ab, ab)
    d01 = np.einsum("ij,ij->i", ab, ac)
    d11 = np.einsum("ij,ij->i", ac, ac)
    d20 = np.einsum("ij,ij->i", ap, ab)
    d21 = np.einsum("ij,ij->i", ap, ac)
    denom = d00 * d11 - d01 * d01
    bary_v = np.divide(
        d11 * d20 - d01 * d21,
        denom,
        out=np.zeros_like(denom),
        where=np.abs(denom) > 1e-30,
    )
    bary_w = np.divide(
        d00 * d21 - d01 * d20,
        denom,
        out=np.zeros_like(denom),
        where=np.abs(denom) > 1e-30,
    )
    inside = (
        (normal_norm_sqr > 1e-30) & (bary_v >= 0.0) & (bary_w >= 0.0) & (bary_v + bary_w <= 1.0)
    )
    plane_distance_sqr = signed_plane * signed_plane * normal_norm_sqr
    edge_distance_sqr = np.minimum.reduce(
        [
            _point_segment_distance_squared(point, a, b),
            _point_segment_distance_squared(point, b, c_),
            _point_segment_distance_squared(point, c_, a),
        ]
    )
    return np.where(inside, plane_distance_sqr, edge_distance_sqr)


def _nearest_distance_to_triangle_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    positions: np.ndarray,
    *,
    triangle_chunk: int = 65536,
) -> float:
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if faces.shape[0] == 0 or positions.shape[0] == 0:
        return 0.0
    min_distance_sqr = float("inf")
    chunk_size = max(1, int(triangle_chunk))
    for start in range(0, faces.shape[0], chunk_size):
        stop = min(start + chunk_size, faces.shape[0])
        triangles = vertices[faces[start:stop]]
        for position in positions:
            distance_sqr = _point_triangle_distance_squared(position, triangles)
            if distance_sqr.size:
                min_distance_sqr = min(
                    min_distance_sqr,
                    float(np.min(distance_sqr)),
                )
    return math.sqrt(max(min_distance_sqr, 0.0))


def _triangle_phase_span_rad(
    triangles: np.ndarray,
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    frequency_hz: float,
) -> np.ndarray:
    """Returns max k*Delta path over sample points for each triangle."""

    triangles = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    if triangles.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    tx_positions = np.asarray(tx_positions, dtype=np.float64).reshape(-1, 3)
    rx_positions = np.asarray(rx_positions, dtype=np.float64).reshape(-1, 3)
    samples = np.einsum(
        "bs,tsc->tbc",
        PHASE_SPAN_SAMPLE_BARYCENTRICS,
        triangles,
    )
    # Rx distances do not depend on the Tx loop. Cache them once while
    # preserving the legacy Tx-major/Rx-minor maximum-reduction order.
    rx_distances = [
        np.linalg.norm(samples - rx_position[None, None, :], axis=2)
        for rx_position in rx_positions
    ]
    max_delta_m = np.zeros((triangles.shape[0],), dtype=np.float64)
    path = np.empty((triangles.shape[0], samples.shape[1]), dtype=np.float64)
    for tx_position in tx_positions:
        tx_dist = np.linalg.norm(samples - tx_position[None, None, :], axis=2)
        for rx_dist in rx_distances:
            np.add(tx_dist, rx_dist, out=path)
            delta_m = np.max(path, axis=1) - np.min(path, axis=1)
            np.maximum(max_delta_m, delta_m, out=max_delta_m)
    return (2.0 * pi * float(frequency_hz) / c) * max_delta_m


def _triangle_phase_curvature_rad(triangles, tx_positions, rx_positions, frequency_hz):
    """Return k_c/3 max_pq,edge |L(mid) - (L(v_i)+L(v_j))/2|.

    Distance is convex along each edge, so both leg errors have the same sign.
    The exact maximum over Tx/Rx pairs is therefore the sum of the separate
    endpoint maxima for each edge; no pairwise PO integrations are involved.
    """
    triangles = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    midpoints = 0.5 * (triangles + triangles[:, [1, 2, 0]])
    max_error = np.zeros((2, len(triangles), 3), dtype=np.float64)
    for leg, endpoints in enumerate((tx_positions, rx_positions)):
        for endpoint in np.asarray(endpoints, dtype=np.float64).reshape(-1, 3):
            distances = np.linalg.norm(triangles - endpoint, axis=-1)
            error = 0.5 * (distances + distances[:, [1, 2, 0]]) - np.linalg.norm(
                midpoints - endpoint, axis=-1
            )
            np.maximum(max_error[leg], error, out=max_error[leg])
    return (2.0 * pi * float(frequency_hz) / (3.0 * c)) * np.max(max_error.sum(axis=0), axis=1)


def _triangle_linear_phase_span_rad(triangles, tx_positions, rx_positions, frequency_hz):
    """Worst Tx/Rx span of the centroid-linearized phase over the vertices.

    A linear function's span on a triangle is its largest absolute edge
    increment. Endpoint projection extrema give the exact maximum over all
    Tx/Rx combinations without constructing a pair-by-face tensor.
    """
    triangles = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    centers = triangles.mean(axis=1)
    edges = triangles[:, [1, 2, 0]] - triangles
    minima = np.full((2, len(triangles), 3), np.inf)
    maxima = np.full_like(minima, -np.inf)
    for leg, endpoints in enumerate((tx_positions, rx_positions)):
        for endpoint in np.asarray(endpoints, dtype=np.float64).reshape(-1, 3):
            vectors = endpoint - centers
            directions = vectors / np.maximum(np.linalg.norm(vectors, axis=1)[:, None], 1e-18)
            projections = np.einsum("ned,nd->ne", edges, directions)
            np.minimum(minima[leg], projections, out=minima[leg])
            np.maximum(maxima[leg], projections, out=maxima[leg])
    span = np.max(np.maximum(maxima.sum(axis=0), -minima.sum(axis=0)), axis=1)
    return (2.0 * pi * float(frequency_hz) / c) * span


def _phase_center_delays_for_centroids(
    centroids: np.ndarray,
    tx_position: np.ndarray,
    rx_position: np.ndarray,
    precision: ComputePrecision,
) -> np.ndarray:
    """Returns phase-center round-trip delays for centroid path geometry."""

    float_dtype, _ = _numpy_dtypes_for_precision(precision)
    centroids = np.asarray(centroids, dtype=float_dtype).reshape(-1, 3)
    if centroids.shape[0] == 0:
        return np.zeros((0,), dtype=float_dtype)
    tx_position = np.asarray(tx_position, dtype=float_dtype).reshape(3)
    rx_position = np.asarray(rx_position, dtype=float_dtype).reshape(3)
    tx_dist = np.linalg.norm(tx_position[None, :] - centroids, axis=1)
    rx_dist = np.linalg.norm(rx_position[None, :] - centroids, axis=1)
    return ((tx_dist + rx_dist) / c).astype(float_dtype)


def _aggregate_parent_face_quadrature_contributions(
    *,
    selected_parent_indices: np.ndarray,
    integration_centroids: np.ndarray,
    integration_normals: np.ndarray,
    integration_areas: np.ndarray,
    path_face_indices: np.ndarray,
    effective_weights: np.ndarray,
    tx_phase_center: np.ndarray,
    rx_phase_center: np.ndarray,
    frequency_hz: float,
    material: PhysicalOpticsMaterial,
    backend: ComputeBackend,
    precision: ComputePrecision,
    chunk_size: int = PARENT_QUADRATURE_CONTRIBUTION_CHUNK_SIZE,
    progress_callback: Callable[[dict], None] | None = None,
    polarization_axis: np.ndarray | None = None,
    polarization_callback: PolarizationCallback | None = None,
    delay_response: PODelayResponse | None = None,
    delay_path_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Sums temporary quadrature subface PO terms onto parent face ids.

    Parent-face quadrature refines the PO integral inside each original SMPL
    face, but the persistent path bank remains one path per original face.
    Chunking keeps the temporary subface bank from being materialized on the
    accelerator all at once.
    """

    float_dtype, complex_dtype = _numpy_dtypes_for_precision(precision)
    selected_parent_indices = np.asarray(
        selected_parent_indices,
        dtype=np.int64,
    ).reshape(-1)
    path_face_indices = np.asarray(path_face_indices, dtype=np.int64).reshape(-1)
    contributions = np.zeros(selected_parent_indices.shape, dtype=complex_dtype)
    if selected_parent_indices.size == 0 or path_face_indices.size == 0:
        return contributions

    parent_positions = np.searchsorted(selected_parent_indices, path_face_indices)
    valid = parent_positions < selected_parent_indices.size
    matched = np.zeros(path_face_indices.shape, dtype=bool)
    matched[valid] = selected_parent_indices[parent_positions[valid]] == path_face_indices[valid]
    if not np.all(matched):
        raise ValueError("quadrature path_face_indices must reference selected parent faces")

    effective_weights = np.asarray(effective_weights, dtype=float_dtype).reshape(-1)
    runtime = resolve_torch_runtime(backend, precision)
    max_chunk = max(1, int(chunk_size))
    if delay_response is not None:
        max_chunk = min(max_chunk, 32768)
    response_indices = (
        np.arange(selected_parent_indices.size)
        if delay_path_indices is None
        else np.asarray(delay_path_indices)
    )
    chunk_total = int((path_face_indices.size + max_chunk - 1) // max_chunk)
    if progress_callback is not None:
        parent_subface_totals = np.bincount(
            parent_positions,
            minlength=selected_parent_indices.size,
        )
        parent_subfaces_done = np.zeros_like(parent_subface_totals)
        progress_callback(
            {
                "stage": "quadrature_aggregate_start",
                "parent_faces_processed": 0,
                "parent_faces_total": int(selected_parent_indices.size),
                "subfaces_processed": 0,
                "subfaces_total": int(path_face_indices.size),
                "chunk_index": 0,
                "chunk_total": chunk_total,
            }
        )
    for start in range(0, path_face_indices.size, max_chunk):
        stop = min(start + max_chunk, path_face_indices.size)
        local_indices = np.arange(stop - start, dtype=np.int64)
        chunk_parent_positions = parent_positions[start:stop]
        chunk_parent_indices = path_face_indices[start:stop]
        chunk_contrib, _ = vector_facet_face_contributions(
            face_indices=local_indices,
            centroids=integration_centroids[start:stop],
            normals=integration_normals[start:stop],
            areas=integration_areas[start:stop],
            tx_position=tx_phase_center,
            rx_position=rx_phase_center,
            frequency_hz=frequency_hz,
            material=material,
            weights=effective_weights[chunk_parent_indices],
            backend=backend,
            precision=precision,
            polarization_axis=polarization_axis,
            polarization_callback=polarization_callback,
        )
        if delay_response is not None:
            points = integration_centroids[start:stop]
            sample_delays = (
                np.linalg.norm(points - tx_phase_center, axis=1)
                + np.linalg.norm(points - rx_phase_center, axis=1)
            ) / c
            delay_response.accumulate(
                chunk_contrib,
                sample_delays,
                response_indices[chunk_parent_positions],
                backend=backend,
                precision=precision,
                runtime=runtime,
            )
        np.add.at(contributions, chunk_parent_positions, chunk_contrib)
        if progress_callback is not None:
            parent_subfaces_done += np.bincount(
                chunk_parent_positions,
                minlength=selected_parent_indices.size,
            )
            progress_callback(
                {
                    "stage": "quadrature_aggregate_progress",
                    "parent_faces_processed": int(
                        np.count_nonzero(parent_subfaces_done >= parent_subface_totals)
                    ),
                    "parent_faces_total": int(selected_parent_indices.size),
                    "subfaces_processed": int(stop),
                    "subfaces_total": int(path_face_indices.size),
                    "chunk_index": int(start // max_chunk + 1),
                    "chunk_total": chunk_total,
                }
            )
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "quadrature_aggregate_done",
                "parent_faces_processed": int(selected_parent_indices.size),
                "parent_faces_total": int(selected_parent_indices.size),
                "subfaces_processed": int(path_face_indices.size),
                "subfaces_total": int(path_face_indices.size),
                "chunk_index": chunk_total,
                "chunk_total": chunk_total,
            }
        )
    return contributions


def _unit_interval_exp_integral(argument: np.ndarray) -> np.ndarray:
    argument = np.asarray(argument, dtype=np.float64)
    return np.exp(-0.5j * argument) * np.sinc(argument / (2.0 * pi))


def _standard_simplex_exp_integral(
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    alpha, beta = np.broadcast_arrays(alpha, beta)
    result = np.empty(alpha.shape, dtype=np.complex128)
    small = np.maximum(np.abs(alpha), np.abs(beta)) < 1e-4
    if np.any(small):
        a = alpha[small]
        b = beta[small]
        series = np.zeros(a.shape, dtype=np.complex128)
        for order in range(8):
            powers = np.zeros(a.shape, dtype=np.float64)
            for m in range(order + 1):
                powers += (a**m) * (b ** (order - m))
            series += ((-1j) ** order) * powers / float(math.factorial(order + 2))
        result[small] = series
    regular = ~small
    if np.any(regular):
        a = alpha[regular]
        b = beta[regular]
        values = np.empty(a.shape, dtype=np.complex128)
        use_beta = np.abs(b) >= np.abs(a)
        if np.any(use_beta):
            aa = a[use_beta]
            bb = b[use_beta]
            values[use_beta] = (
                _unit_interval_exp_integral(aa)
                - np.exp(-1j * bb) * _unit_interval_exp_integral(aa - bb)
            ) / (1j * bb)
        use_alpha = ~use_beta
        if np.any(use_alpha):
            aa = a[use_alpha]
            bb = b[use_alpha]
            values[use_alpha] = (
                _unit_interval_exp_integral(bb)
                - np.exp(-1j * aa) * _unit_interval_exp_integral(bb - aa)
            ) / (1j * aa)
        result[regular] = values
    return result


def _standard_simplex_exp_integral_torch(alpha, beta):
    """Analytic simplex integral on the input tensors' device and precision."""
    import torch

    alpha, beta = torch.broadcast_tensors(alpha, beta)
    # The regular expression subtracts nearly equal terms near zero. A wider
    # series region controls cancellation when computing in float32.
    threshold = 0.1 if alpha.dtype == torch.float32 else 1e-4
    small = torch.maximum(alpha.abs(), beta.abs()) < threshold
    use_beta = beta.abs() >= alpha.abs()
    a = torch.where(use_beta, alpha, beta)
    b = torch.where(small, torch.ones_like(beta), torch.where(use_beta, beta, alpha))

    def interval(argument):
        return torch.exp(-0.5j * argument) * torch.sinc(argument / (2.0 * pi))

    result = (interval(a) - torch.exp(-1j * b) * interval(a - b)) / (1j * b)
    a, b = alpha[small], beta[small]
    if a.numel():
        # h_n = sum(a**m * b**(n-m)); avoid repeated powers in the series.
        h_previous, h_before_previous = torch.ones_like(a), torch.zeros_like(a)
        series = 0.5 * h_previous.to(result.dtype)
        for order in range(1, 8):
            h = (a + b) * h_previous - a * b * h_before_previous
            series += ((-1j) ** order / math.factorial(order + 2)) * h
            h_before_previous, h_previous = h_previous, h
        result[small] = series
    return result


def _triangle_linear_phase_integrals(
    triangles: np.ndarray,
    centroids: np.ndarray,
    q_vectors: np.ndarray,
) -> np.ndarray:
    return _triangle_phase_integrals_from_geometry(
        _triangle_linear_phase_geometry(triangles, centroids, q_vectors)
    )


def _triangle_linear_phase_geometry(triangles, centroids, q_vectors):
    """Prepare triangle Jacobians and phase coordinates once per geometry."""
    triangles = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    centroids = np.asarray(centroids, dtype=np.float64).reshape(-1, 3)
    q_vectors = np.asarray(q_vectors, dtype=np.float64)
    if q_vectors.ndim != 3 or q_vectors.shape[1] != triangles.shape[0]:
        raise ValueError("q_vectors must have shape [num_pairs, num_faces, 3]")
    v0 = triangles[:, 0]
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    jacobian = np.linalg.norm(np.cross(edge_a, edge_b), axis=1)
    alpha = np.einsum("pfd,fd->pf", q_vectors, edge_a)
    beta = np.einsum("pfd,fd->pf", q_vectors, edge_b)
    offset_phase = np.einsum("pfd,fd->pf", q_vectors, v0 - centroids)
    return jacobian, alpha, beta, offset_phase


def _triangle_phase_integrals_from_geometry(geometry, scale=1.0, *, runtime=None):
    """Evaluate cached phase coordinates at broadcast wavenumber scales."""
    if runtime is not None:
        import torch

        jacobian, alpha, beta, offset_phase = (
            torch.as_tensor(value, dtype=runtime.float_dtype, device=runtime.device)
            for value in geometry
        )
        scale = torch.as_tensor(scale, dtype=runtime.float_dtype, device=runtime.device)
        return (
            jacobian[None]
            * torch.exp(-1j * (scale * offset_phase))
            * _standard_simplex_exp_integral_torch(scale * alpha, scale * beta)
        )
    jacobian, alpha, beta, offset_phase = geometry
    return (
        jacobian[None, :]
        * np.exp(-1j * (scale * offset_phase))
        * _standard_simplex_exp_integral(scale * alpha, scale * beta)
    )


def _aggregate_parent_face_far_field_analytic_contributions(
    *,
    selected_parent_indices: np.ndarray,
    integration_triangles: np.ndarray,
    integration_centroids: np.ndarray,
    integration_normals: np.ndarray,
    integration_areas: np.ndarray,
    path_face_indices: np.ndarray,
    effective_weights: np.ndarray,
    tx_phase_center: np.ndarray,
    rx_phase_center: np.ndarray,
    frequency_hz: float,
    material: PhysicalOpticsMaterial,
    backend: ComputeBackend,
    precision: ComputePrecision,
    chunk_size: int = PARENT_QUADRATURE_CONTRIBUTION_CHUNK_SIZE,
    progress_callback: Callable[[dict], None] | None = None,
    polarization_axis: np.ndarray | None = None,
    polarization_callback: PolarizationCallback | None = None,
    delay_response: PODelayResponse | None = None,
    delay_path_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Sums analytic linear-phase subface PO terms onto parent face ids."""

    float_dtype, complex_dtype = _numpy_dtypes_for_precision(precision)
    selected_parent_indices = np.asarray(
        selected_parent_indices,
        dtype=np.int64,
    ).reshape(-1)
    path_face_indices = np.asarray(path_face_indices, dtype=np.int64).reshape(-1)
    contributions = np.zeros(selected_parent_indices.shape, dtype=complex_dtype)
    if selected_parent_indices.size == 0 or path_face_indices.size == 0:
        return contributions

    parent_positions = np.searchsorted(selected_parent_indices, path_face_indices)
    valid = parent_positions < selected_parent_indices.size
    matched = np.zeros(path_face_indices.shape, dtype=bool)
    matched[valid] = selected_parent_indices[parent_positions[valid]] == path_face_indices[valid]
    if not np.all(matched):
        raise ValueError("far-field path_face_indices must reference selected parent faces")

    effective_weights = np.asarray(effective_weights, dtype=float_dtype).reshape(-1)
    runtime = resolve_torch_runtime(backend, precision)
    max_chunk = max(1, int(chunk_size))
    if delay_response is not None:
        max_chunk = min(max_chunk, 32768, max(1, 1_000_000 // len(delay_response.frequencies_hz)))
    response_indices = (
        np.arange(selected_parent_indices.size)
        if delay_path_indices is None
        else np.asarray(delay_path_indices)
    )
    chunk_total = int((path_face_indices.size + max_chunk - 1) // max_chunk)
    if progress_callback is not None:
        parent_subface_totals = np.bincount(
            parent_positions,
            minlength=selected_parent_indices.size,
        )
        parent_subfaces_done = np.zeros_like(parent_subface_totals)
        progress_callback(
            {
                "stage": "far_field_aggregate_start",
                "parent_faces_processed": 0,
                "parent_faces_total": int(selected_parent_indices.size),
                "subfaces_processed": 0,
                "subfaces_total": int(path_face_indices.size),
                "chunk_index": 0,
                "chunk_total": chunk_total,
            }
        )
    wavelength_m = c / float(frequency_hz)
    k0 = 2.0 * pi / wavelength_m
    tx_phase_center = np.asarray(tx_phase_center, dtype=np.float64).reshape(3)
    rx_phase_center = np.asarray(rx_phase_center, dtype=np.float64).reshape(3)
    for start in range(0, path_face_indices.size, max_chunk):
        stop = min(start + max_chunk, path_face_indices.size)
        local_indices = np.arange(stop - start, dtype=np.int64)
        chunk_parent_positions = parent_positions[start:stop]
        chunk_parent_indices = path_face_indices[start:stop]
        chunk_centroids = integration_centroids[start:stop]
        chunk_areas = integration_areas[start:stop]
        chunk_contrib, _ = vector_facet_face_contributions(
            face_indices=local_indices,
            centroids=chunk_centroids,
            normals=integration_normals[start:stop],
            areas=chunk_areas,
            tx_position=tx_phase_center,
            rx_position=rx_phase_center,
            frequency_hz=frequency_hz,
            material=material,
            weights=effective_weights[chunk_parent_indices],
            backend=backend,
            precision=precision,
            polarization_axis=polarization_axis,
            polarization_callback=polarization_callback,
        )
        tx_vec = tx_phase_center[None, :] - chunk_centroids
        rx_vec = rx_phase_center[None, :] - chunk_centroids
        tx_dist = np.linalg.norm(tx_vec, axis=1)
        rx_dist = np.linalg.norm(rx_vec, axis=1)
        tx_hat = tx_vec / np.maximum(tx_dist[:, None], 1e-18)
        rx_hat = rx_vec / np.maximum(rx_dist[:, None], 1e-18)
        # The integral evaluates exp(-j q·delta), while the local path length
        # is L(delta) = L(0) - (tx_hat + rx_hat)·delta.
        q_vectors = -k0 * (tx_hat + rx_hat)[None, :, :]
        phase_geometry = _triangle_linear_phase_geometry(
            integration_triangles[start:stop],
            chunk_centroids,
            q_vectors,
        )
        # Keep analytic evaluation and leaf reduction on the configured device.
        # The first scale supplies the carrier term; remaining scales supply
        # the wideband nodes, without copying per-leaf integrals back to CPU.
        scales = np.ones(1)
        if delay_response is not None:
            scales = np.concatenate(
                (scales, (k0 + 2.0 * pi * delay_response.frequencies_hz / c) / k0)
            )
        integrals = _triangle_phase_integrals_from_geometry(
            phase_geometry, scales[:, None, None], runtime=runtime
        )[:, 0, :]
        if runtime is None:
            ratios = np.divide(
                integrals, chunk_areas, out=np.zeros_like(integrals), where=chunk_areas > 0.0
            )
            area_ratio = ratios[0].astype(complex_dtype)
        else:
            import torch

            areas_t = torch.as_tensor(
                chunk_areas, dtype=runtime.float_dtype, device=runtime.device
            )
            ratios = torch.where(
                areas_t > 0.0, integrals / areas_t.clamp_min(1e-30), torch.zeros_like(integrals)
            )
            area_ratio = ratios[0].cpu().numpy().astype(complex_dtype, copy=False)
        if delay_response is not None:
            delay_response.accumulate(
                chunk_contrib,
                (tx_dist + rx_dist) / c,
                response_indices[chunk_parent_positions],
                phase_integrals=ratios[1:],
                backend=backend,
                precision=precision,
                runtime=runtime,
                vertex_delay_offsets_s=-np.einsum(
                    "nvi,ni->nv",
                    integration_triangles[start:stop] - chunk_centroids[:, None],
                    tx_hat + rx_hat,
                )
                / c,
                carrier_frequency_hz=frequency_hz,
            )
        chunk_contrib = chunk_contrib.astype(complex_dtype, copy=False) * area_ratio
        np.add.at(contributions, chunk_parent_positions, chunk_contrib)
        if progress_callback is not None:
            parent_subfaces_done += np.bincount(
                chunk_parent_positions,
                minlength=selected_parent_indices.size,
            )
            progress_callback(
                {
                    "stage": "far_field_aggregate_progress",
                    "parent_faces_processed": int(
                        np.count_nonzero(parent_subfaces_done >= parent_subface_totals)
                    ),
                    "parent_faces_total": int(selected_parent_indices.size),
                    "subfaces_processed": int(stop),
                    "subfaces_total": int(path_face_indices.size),
                    "chunk_index": int(start // max_chunk + 1),
                    "chunk_total": chunk_total,
                }
            )
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "far_field_aggregate_done",
                "parent_faces_processed": int(selected_parent_indices.size),
                "parent_faces_total": int(selected_parent_indices.size),
                "subfaces_processed": int(path_face_indices.size),
                "subfaces_total": int(path_face_indices.size),
                "chunk_index": chunk_total,
                "chunk_total": chunk_total,
            }
        )
    return contributions


def vector_facet_channel(
    face_indices: np.ndarray,
    centroids: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    tx_position: np.ndarray,
    rx_position: np.ndarray,
    frequency_hz: float,
    material: PhysicalOpticsMaterial,
) -> tuple[complex, int]:
    """Returns the coherent PO channel for one Tx/Rx pair."""

    contributions, _ = vector_facet_face_contributions(
        face_indices=face_indices,
        centroids=centroids,
        normals=normals,
        areas=areas,
        tx_position=tx_position,
        rx_position=rx_position,
        frequency_hz=frequency_hz,
        material=material,
    )
    return complex(np.sum(contributions)), int(np.count_nonzero(np.abs(contributions) > 0.0))


def _reciprocal_facet_kernel(
    tx_hat, rx_hat, normals, tx_pol, rx_pol, material, frequency_hz, preferred_axis
):
    """Reciprocal tangent-plane scattering dyadic projected onto antenna fields.

    Symmetrize the local dielectric PO dyadic with its reciprocal transpose,
    before integration. This preserves the monostatic and specular limits.
    Receive fields use a Hermitian projection, hence conjugate both fields
    when exchanging the source and receiver in the reciprocal term.
    """
    eta0 = 1.0 / (c * epsilon_0)
    result = np.zeros(tx_hat.shape[0], dtype=np.complex128)
    for source, observer, incident, receive in (
        (tx_hat, rx_hat, tx_pol, rx_pol),
        (rx_hat, tx_hat, rx_pol.conjugate(), tx_pol.conjugate()),
    ):
        cosine = np.einsum("ij,ij->i", normals, source)
        r_te, r_tm = fresnel_coefficients(cosine, material, frequency_hz)
        attenuation = coherent_surface_attenuation(cosine, material, frequency_hz)
        k_inc = -source
        k_ref = k_inc - 2.0 * np.einsum("ij,ij->i", k_inc, normals)[:, None] * normals
        te, tm_inc = _local_incidence_bases(k_inc, normals, preferred_axis)
        tm_ref = _normalize_rows(np.cross(te, k_ref), fallback=tm_inc)
        e_ref = attenuation[:, None] * (
            (r_te * np.einsum("ij,ij->i", incident, te))[:, None] * te
            + (r_tm * np.einsum("ij,ij->i", incident, tm_inc))[:, None] * tm_ref
        )
        j = np.cross(normals, np.cross(k_ref, e_ref) / eta0)
        m = -np.cross(normals, e_ref)
        radiation = eta0 * np.cross(observer, np.cross(observer, j)) + np.cross(observer, m)
        result += 0.5 * (cosine > 0.0) * np.einsum("ij,ij->i", receive.conjugate(), radiation)
    return result


def vector_facet_face_contributions(
    face_indices: np.ndarray,
    centroids: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    tx_position: np.ndarray,
    rx_position: np.ndarray,
    frequency_hz: float,
    material: PhysicalOpticsMaterial,
    *,
    weights: np.ndarray | None = None,
    backend: ComputeBackend = "numpy",
    precision: ComputePrecision = "float64",
    polarization_axis: np.ndarray | None = None,
    polarization_callback: PolarizationCallback | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return facet fields projected onto the Tx/Rx polarization responses.

    The optional callback receives selected integration points and returns
    complex incident and receive polarization vectors there. Their magnitudes
    carry reflection loss and must not be normalized. The receive projection
    uses the Hermitian inner product, as for ordinary antenna polarizations.
    """

    float_dtype, complex_dtype = _numpy_dtypes_for_precision(precision)
    validate_compute_backend(backend)
    face_indices = np.asarray(face_indices, dtype=np.int64).reshape(-1)
    if face_indices.size == 0:
        return (
            np.zeros((0,), dtype=complex_dtype),
            np.zeros((0,), dtype=float_dtype),
        )
    polarizations = None
    if polarization_callback is not None:
        vectors = polarization_callback(np.asarray(centroids)[face_indices])
        polarizations = tuple(np.asarray(p, dtype=complex_dtype) for p in vectors)
        if len(polarizations) != 2 or any(
            p.shape != (face_indices.size, 3) for p in polarizations
        ):
            raise ValueError("polarization callback must return two [selected faces, 3] arrays")
    runtime = resolve_torch_runtime(backend, precision)
    if runtime is not None:
        return _torch_vector_facet_face_contributions(
            face_indices=face_indices,
            centroids=centroids,
            normals=normals,
            areas=areas,
            tx_position=tx_position,
            rx_position=rx_position,
            frequency_hz=frequency_hz,
            material=material,
            weights=weights,
            runtime=runtime,
            polarization_axis=polarization_axis,
            polarizations=polarizations,
        )

    wavelength_m = c / frequency_hz
    k0 = 2.0 * pi / wavelength_m
    preferred_axis = np.asarray(
        [0.0, 0.0, 1.0] if polarization_axis is None else polarization_axis,
        dtype=float_dtype,
    )

    c_sel = np.asarray(centroids, dtype=float_dtype)[face_indices]
    n_sel = np.asarray(normals, dtype=float_dtype)[face_indices]
    a_sel = np.asarray(areas, dtype=float_dtype)[face_indices]
    if weights is None:
        weights = np.ones_like(a_sel, dtype=float_dtype)
    else:
        weights = np.asarray(weights, dtype=float_dtype).reshape(-1)
        if weights.shape != a_sel.shape:
            raise ValueError("weights shape mismatch for selected faces")

    tx_vec = np.asarray(tx_position, dtype=float_dtype)[None, :] - c_sel
    rx_vec = np.asarray(rx_position, dtype=float_dtype)[None, :] - c_sel
    tx_dist = np.linalg.norm(tx_vec, axis=1)
    rx_dist = np.linalg.norm(rx_vec, axis=1)
    tx_hat = tx_vec / np.maximum(tx_dist[:, None], 1e-18)
    rx_hat = rx_vec / np.maximum(rx_dist[:, None], 1e-18)
    delays_s = (tx_dist + rx_dist) / c
    contributions = np.zeros(face_indices.shape, dtype=complex_dtype)

    cos_inc = np.einsum("ij,ij->i", n_sel, tx_hat)
    illum_mask = (weights > 0.0) & ((cos_inc > 0.0) | (np.einsum("ij,ij->i", n_sel, rx_hat) > 0.0))
    if not np.any(illum_mask):
        return contributions, delays_s.astype(float_dtype)

    n_vis = n_sel[illum_mask]
    a_vis = a_sel[illum_mask]
    w_vis = weights[illum_mask]
    tx_hat_vis = tx_hat[illum_mask]
    rx_hat_vis = rx_hat[illum_mask]
    tx_dist_vis = tx_dist[illum_mask]
    rx_dist_vis = rx_dist[illum_mask]

    if polarizations is None:
        tx_pol = _transverse_polarizations(-tx_hat_vis, preferred_axis)
        rx_pol = _transverse_polarizations(rx_hat_vis, preferred_axis)
    else:
        tx_pol, rx_pol = (p[illum_mask] for p in polarizations)
    kernel = _reciprocal_facet_kernel(
        tx_hat_vis,
        rx_hat_vis,
        n_vis,
        tx_pol,
        rx_pol,
        material,
        frequency_hz,
        preferred_axis,
    )
    # e^(j omega t): the outgoing Green function gives +jk times this kernel.
    phase = np.exp(-1j * k0 * (tx_dist_vis + rx_dist_vis))
    contributions[illum_mask] = (
        wavelength_m
        * phase
        * (1j * k0)
        * a_vis
        * w_vis
        * kernel
        / (16.0 * pi**2 * np.maximum(tx_dist_vis * rx_dist_vis, 1e-18))
    ).astype(complex_dtype)

    return contributions, delays_s.astype(float_dtype)


def vector_facet_face_contributions_for_pairs(
    face_indices: np.ndarray,
    centroids: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    frequency_hz: float,
    material: PhysicalOpticsMaterial,
    *,
    weights: np.ndarray | None = None,
    precision: ComputePrecision = "float64",
) -> tuple[np.ndarray, np.ndarray]:
    """Returns per-face PO contributions for many Tx/Rx pairs.

    The output shape is ``(num_pairs, num_faces)`` and is numerically aligned
    with repeated calls to :func:`vector_facet_face_contributions`.
    """

    float_dtype, complex_dtype = _numpy_dtypes_for_precision(precision)
    face_indices = np.asarray(face_indices, dtype=np.int64).reshape(-1)
    tx_positions = np.asarray(tx_positions, dtype=float_dtype).reshape(-1, 3)
    rx_positions = np.asarray(rx_positions, dtype=float_dtype).reshape(-1, 3)
    if tx_positions.shape != rx_positions.shape:
        raise ValueError("tx_positions and rx_positions must have matching shapes")
    num_pairs = int(tx_positions.shape[0])
    if face_indices.size == 0 or num_pairs == 0:
        return (
            np.zeros((num_pairs, face_indices.size), dtype=complex_dtype),
            np.zeros((num_pairs, face_indices.size), dtype=float_dtype),
        )

    wavelength_m = c / frequency_hz
    k0 = 2.0 * pi / wavelength_m
    preferred_axis = np.array([0.0, 0.0, 1.0], dtype=float_dtype)

    c_sel = np.asarray(centroids, dtype=float_dtype)[face_indices]
    n_sel = np.asarray(normals, dtype=float_dtype)[face_indices]
    a_sel = np.asarray(areas, dtype=float_dtype)[face_indices]
    if weights is None:
        weights = np.ones_like(a_sel, dtype=float_dtype)
    else:
        weights = np.asarray(weights, dtype=float_dtype).reshape(-1)
        if weights.shape != a_sel.shape:
            raise ValueError("weights shape mismatch for selected faces")

    tx_vec = tx_positions[:, None, :] - c_sel[None, :, :]
    rx_vec = rx_positions[:, None, :] - c_sel[None, :, :]
    tx_dist = np.linalg.norm(tx_vec, axis=2)
    rx_dist = np.linalg.norm(rx_vec, axis=2)
    tx_hat = tx_vec / np.maximum(tx_dist[:, :, None], 1e-18)
    rx_hat = rx_vec / np.maximum(rx_dist[:, :, None], 1e-18)
    delays_s = (tx_dist + rx_dist) / c
    contributions = np.zeros((num_pairs, face_indices.size), dtype=complex_dtype)

    cos_inc = np.einsum("fn,pfn->pf", n_sel, tx_hat)
    illum_mask = (weights[None, :] > 0.0) & (
        (cos_inc > 0.0) | (np.einsum("fn,pfn->pf", n_sel, rx_hat) > 0.0)
    )
    if not np.any(illum_mask):
        return contributions, delays_s.astype(float_dtype)

    pair_indices, local_face_indices = np.nonzero(illum_mask)
    n_vis = n_sel[local_face_indices]
    a_vis = a_sel[local_face_indices]
    w_vis = weights[local_face_indices]
    tx_hat_vis = tx_hat[pair_indices, local_face_indices]
    rx_hat_vis = rx_hat[pair_indices, local_face_indices]
    tx_dist_vis = tx_dist[pair_indices, local_face_indices]
    rx_dist_vis = rx_dist[pair_indices, local_face_indices]

    tx_pol = _transverse_polarizations(-tx_hat_vis, preferred_axis)
    rx_pol = _transverse_polarizations(rx_hat_vis, preferred_axis)
    kernel = _reciprocal_facet_kernel(
        tx_hat_vis,
        rx_hat_vis,
        n_vis,
        tx_pol,
        rx_pol,
        material,
        frequency_hz,
        preferred_axis,
    )
    phase = np.exp(-1j * k0 * (tx_dist_vis + rx_dist_vis))
    contributions[pair_indices, local_face_indices] = (
        wavelength_m
        * phase
        * (1j * k0)
        * a_vis
        * w_vis
        * kernel
        / (16.0 * pi**2 * np.maximum(tx_dist_vis * rx_dist_vis, 1e-18))
    ).astype(complex_dtype)

    return contributions, delays_s.astype(float_dtype)


def project_phase_center_contributions_to_virtual_channels(
    phase_center_contributions: np.ndarray,
    centroids: np.ndarray,
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    frequency_hz: float,
    *,
    virtual_channel_tx_indices: np.ndarray | None = None,
    virtual_channel_rx_indices: np.ndarray | None = None,
    virtual_channel_order: str = "tx_major",
    precision: ComputePrecision = "float64",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Projects phase-center PO paths to virtual channels by geometry.

    Returns ``coefficients, delays_s, path_lengths_m, spread_products_m2``.
    The coefficient projection preserves the phase-center scattering term and
    applies per-channel carrier phase and free-space spreading deltas.

    Approximation: the PO surface-current/scattering term is evaluated once at
    the phase center. Virtual channels reuse that scattering amplitude and only
    get geometry-dependent carrier phase, spreading, and beat-delay updates.
    """

    float_dtype, complex_dtype = _numpy_dtypes_for_precision(precision)
    phase_center_contributions = np.asarray(
        phase_center_contributions,
        dtype=complex_dtype,
    ).reshape(-1)
    centroids = np.asarray(centroids, dtype=float_dtype).reshape(-1, 3)
    tx_positions = np.asarray(tx_positions, dtype=float_dtype).reshape(-1, 3)
    rx_positions = np.asarray(rx_positions, dtype=float_dtype).reshape(-1, 3)
    num_paths = int(phase_center_contributions.size)
    num_tx = int(tx_positions.shape[0])
    num_rx = int(rx_positions.shape[0])
    vc_tx, vc_rx = resolve_virtual_channel_pairs(
        num_tx,
        num_rx,
        order=virtual_channel_order,
        tx_indices=virtual_channel_tx_indices,
        rx_indices=virtual_channel_rx_indices,
    )
    num_vc = int(vc_tx.size)
    if centroids.shape[0] != num_paths:
        raise ValueError("centroids and phase_center_contributions shape mismatch")
    if num_paths == 0:
        return (
            np.zeros((num_vc, 0), dtype=complex_dtype),
            np.zeros((num_vc, 0), dtype=float_dtype),
            np.zeros((num_vc, 0), dtype=float_dtype),
            np.ones((num_vc, 0), dtype=float_dtype),
        )

    tx_phase_center = np.mean(tx_positions, axis=0)
    rx_phase_center = np.mean(rx_positions, axis=0)
    center_tx_dist = np.linalg.norm(tx_phase_center[None, :] - centroids, axis=1)
    center_rx_dist = np.linalg.norm(rx_phase_center[None, :] - centroids, axis=1)
    center_lengths = center_tx_dist + center_rx_dist
    center_spreads = np.maximum(center_tx_dist * center_rx_dist, 1e-18)

    pair_tx = tx_positions[vc_tx]
    pair_rx = rx_positions[vc_rx]
    tx_dist = np.linalg.norm(
        pair_tx[:, None, :] - centroids[None, :, :],
        axis=2,
    )
    rx_dist = np.linalg.norm(
        pair_rx[:, None, :] - centroids[None, :, :],
        axis=2,
    )
    path_lengths = (tx_dist + rx_dist).astype(float_dtype, copy=False)
    spreads = np.maximum(tx_dist * rx_dist, 1e-18).astype(
        float_dtype,
        copy=False,
    )

    # This is a virtual-array approximation: do not recompute PO currents for
    # each Tx/Rx pair, but keep per-VC carrier phase, spreading, and tau_vc.
    k0 = 2.0 * pi * frequency_hz / c
    # Retain the physical channel's negative phase when changing path length.
    phase = np.exp(-1j * k0 * (path_lengths - center_lengths[None, :]))
    spreading = center_spreads[None, :] / np.maximum(spreads, 1e-18)
    coefficients = (phase_center_contributions[None, :] * phase * spreading).astype(complex_dtype)
    delays_s = (path_lengths / c).astype(float_dtype)
    return coefficients, delays_s, path_lengths, spreads


def _torch_vector_facet_face_contributions(
    face_indices: np.ndarray,
    centroids: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    tx_position: np.ndarray,
    rx_position: np.ndarray,
    frequency_hz: float,
    material: PhysicalOpticsMaterial,
    *,
    weights: np.ndarray | None = None,
    runtime: TorchRuntime,
    polarization_axis: np.ndarray | None = None,
    polarizations: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Torch implementation of per-face PO contributions.

    The operation is intentionally kept API-compatible with the NumPy path and
    returns NumPy arrays. It uses CUDA when available through ``backend='auto'``
    and falls back to Torch CPU only when explicitly requested.
    """

    import torch  # pylint: disable=import-outside-toplevel

    device, dtype, cdtype = runtime.device, runtime.float_dtype, runtime.complex_dtype
    float_dtype, complex_dtype = runtime.numpy_float_dtype, runtime.numpy_complex_dtype

    face_indices = np.asarray(face_indices, dtype=np.int64).reshape(-1)
    c_sel = torch.as_tensor(
        np.asarray(centroids, dtype=float_dtype)[face_indices],
        dtype=dtype,
        device=device,
    )
    n_sel = torch.as_tensor(
        np.asarray(normals, dtype=float_dtype)[face_indices],
        dtype=dtype,
        device=device,
    )
    a_sel = torch.as_tensor(
        np.asarray(areas, dtype=float_dtype)[face_indices],
        dtype=dtype,
        device=device,
    )
    if weights is None:
        w_sel = torch.ones_like(a_sel)
    else:
        weights = np.asarray(weights, dtype=float_dtype).reshape(-1)
        if weights.shape != tuple(a_sel.shape):
            raise ValueError("weights shape mismatch for selected faces")
        w_sel = torch.as_tensor(weights, dtype=dtype, device=device)

    tx_pos = torch.as_tensor(
        np.asarray(tx_position, dtype=float_dtype).reshape(3),
        dtype=dtype,
        device=device,
    )
    rx_pos = torch.as_tensor(
        np.asarray(rx_position, dtype=float_dtype).reshape(3),
        dtype=dtype,
        device=device,
    )
    tx_vec = tx_pos[None, :] - c_sel
    rx_vec = rx_pos[None, :] - c_sel
    tx_dist = torch.linalg.norm(tx_vec, dim=1)
    rx_dist = torch.linalg.norm(rx_vec, dim=1)
    tx_hat = tx_vec / torch.clamp(tx_dist[:, None], min=1e-18)
    rx_hat = rx_vec / torch.clamp(rx_dist[:, None], min=1e-18)
    delays_s = (tx_dist + rx_dist) / c
    contributions = torch.zeros((face_indices.size,), dtype=cdtype, device=device)

    cos_inc = torch.sum(n_sel * tx_hat, dim=1)
    illum_mask = (w_sel > 0.0) & ((cos_inc > 0.0) | (torch.sum(n_sel * rx_hat, dim=1) > 0.0))
    if not bool(torch.any(illum_mask).item()):
        return (
            contributions.cpu().numpy().astype(complex_dtype),
            delays_s.cpu().numpy().astype(float_dtype),
        )

    n_vis = n_sel[illum_mask]
    a_vis = a_sel[illum_mask]
    w_vis = w_sel[illum_mask]
    tx_hat_vis = tx_hat[illum_mask]
    rx_hat_vis = rx_hat[illum_mask]
    tx_dist_vis = tx_dist[illum_mask]
    rx_dist_vis = rx_dist[illum_mask]

    wavelength_m = c / frequency_hz
    k0 = 2.0 * pi / wavelength_m
    eta0 = 1.0 / (c * epsilon_0)
    preferred_axis = torch.as_tensor(
        [0.0, 0.0, 1.0] if polarization_axis is None else polarization_axis,
        dtype=dtype,
        device=device,
    )

    if polarizations is None:
        tx_pol = _torch_transverse_polarizations(-tx_hat_vis, preferred_axis).to(cdtype)
        rx_pol = _torch_transverse_polarizations(rx_hat_vis, preferred_axis).to(cdtype)
    else:
        tx_pol, rx_pol = (
            torch.as_tensor(p, dtype=cdtype, device=device)[illum_mask] for p in polarizations
        )
    kernel = torch.zeros_like(tx_dist_vis, dtype=cdtype)
    for source, observer, incident, receive in (
        (tx_hat_vis, rx_hat_vis, tx_pol, rx_pol),
        (rx_hat_vis, tx_hat_vis, rx_pol.conj(), tx_pol.conj()),
    ):
        cosine = torch.sum(n_vis * source, dim=1)
        r_te, r_tm = _torch_fresnel_coefficients(
            cosine,
            material,
            frequency_hz,
            wavelength_m,
            cdtype,
        )
        attenuation = _torch_coherent_surface_attenuation(
            cosine,
            material,
            frequency_hz,
            dtype,
            cdtype,
        )
        k_inc = -source
        k_ref = k_inc - 2.0 * torch.sum(k_inc * n_vis, dim=1)[:, None] * n_vis
        te, tm_inc = _torch_local_incidence_bases(k_inc, n_vis, preferred_axis)
        tm_ref = _torch_normalize_rows(torch_vector_cross(te, k_ref, dim=1), fallback=tm_inc)
        e_ref = attenuation[:, None] * (
            (r_te * torch.sum(incident * te, dim=1))[:, None] * te
            + (r_tm * torch.sum(incident * tm_inc, dim=1))[:, None] * tm_ref
        )
        n_c, k_c, o_c = n_vis.to(cdtype), k_ref.to(cdtype), observer.to(cdtype)
        j = torch_vector_cross(n_c, torch_vector_cross(k_c, e_ref, dim=1) / eta0, dim=1)
        m = -torch_vector_cross(n_c, e_ref, dim=1)
        radiation = eta0 * torch_vector_cross(
            o_c, torch_vector_cross(o_c, j, dim=1), dim=1
        ) + torch_vector_cross(o_c, m, dim=1)
        kernel += 0.5 * (cosine > 0.0) * torch.sum(receive.conj() * radiation, dim=1)
    phase = torch.exp(
        -1j * torch.as_tensor(k0, dtype=dtype, device=device) * (tx_dist_vis + rx_dist_vis)
    ).to(cdtype)
    contributions[illum_mask] = (
        wavelength_m
        * phase
        * (1j * k0)
        * a_vis
        * w_vis
        * kernel
        / (16.0 * pi**2 * torch.clamp(tx_dist_vis * rx_dist_vis, min=1e-18))
    )

    return (
        contributions.cpu().numpy().astype(complex_dtype),
        delays_s.cpu().numpy().astype(float_dtype),
    )


def _torch_fresnel_coefficients(
    cos_theta,
    material: PhysicalOpticsMaterial,
    frequency_hz: float,
    wavelength_m: float,
    cdtype,
):
    """Torch Fresnel reflection coefficients for PEC, interface, or slab models."""

    import torch  # pylint: disable=import-outside-toplevel

    cos_theta = torch.clamp(cos_theta, 0.0, 1.0)
    if material.model == "pec":
        return (
            -torch.ones_like(cos_theta, dtype=cdtype),
            torch.ones_like(cos_theta, dtype=cdtype),
        )
    omega = 2.0 * pi * frequency_hz
    epsilon_r_complex = complex(
        material.relative_permittivity,
        -material.conductivity_s_per_m / (omega * epsilon_0),
    )
    eps = torch.as_tensor(
        epsilon_r_complex,
        dtype=cdtype,
        device=cos_theta.device,
    )
    cos_c = cos_theta.to(cdtype)
    sin_theta_sqr = 1.0 - cos_theta * cos_theta
    a = torch.sqrt(eps - sin_theta_sqr.to(cdtype) + 0j)
    r_te_p = (cos_c - a) / (cos_c + a)
    r_tm_p = (eps * cos_c - a) / (eps * cos_c + a)
    if material.model == "interface" or material.thickness_m <= 0.0:
        return r_te_p, r_tm_p
    if material.model != "slab":
        raise ValueError(f"Unknown Fresnel material model: {material.model}")
    q = (
        2.0
        * pi
        * material.thickness_m
        / wavelength_m
        * torch.sqrt(eps - sin_theta_sqr.to(cdtype) + 0j)
    )
    exp_j_2q = torch.exp(-2.0j * q)
    r_te = r_te_p * (1.0 - exp_j_2q) / (1.0 - (r_te_p * r_te_p) * exp_j_2q)
    r_tm = r_tm_p * (1.0 - exp_j_2q) / (1.0 - (r_tm_p * r_tm_p) * exp_j_2q)
    return r_te, r_tm


def _torch_coherent_surface_attenuation(
    cos_theta,
    material: PhysicalOpticsMaterial,
    frequency_hz: float,
    dtype,
    cdtype,
):
    """Torch coherent attenuation through a lossy surface layer."""

    import torch  # pylint: disable=import-outside-toplevel

    cos_theta = torch.clamp(cos_theta, 0.0, 1.0)
    if material.surface_attenuation_thickness_m <= 0.0:
        return torch.ones_like(cos_theta, dtype=dtype)
    omega = 2.0 * pi * frequency_hz
    epsilon_r_complex = complex(
        material.relative_permittivity,
        -material.conductivity_s_per_m / (omega * epsilon_0),
    )
    eps = torch.as_tensor(
        epsilon_r_complex,
        dtype=cdtype,
        device=cos_theta.device,
    )
    wavelength_m = c / frequency_hz
    k0 = 2.0 * pi / wavelength_m
    sin_theta_sqr = 1.0 - cos_theta * cos_theta
    kz_medium = k0 * torch.sqrt(eps - sin_theta_sqr.to(cdtype) + 0j)
    alpha_z = torch.abs(torch.imag(kz_medium))
    return torch.exp(-2.0 * alpha_z * material.surface_attenuation_thickness_m)


def _torch_normalize_rows(vectors, fallback=None):
    """Normalizes Torch vector rows, substituting a fallback for zero rows."""

    import torch  # pylint: disable=import-outside-toplevel

    norms = torch.linalg.norm(vectors, dim=1)
    result = vectors / torch.clamp(norms[:, None], min=1e-18)
    valid = norms > 1e-18
    if not bool(torch.all(valid).item()):
        if fallback is None:
            raise ValueError("Zero-length vector row without fallback")
        result = torch.where(valid[:, None], result, fallback)
    return result


def _torch_transverse_polarizations(propagation_hats, preferred_axis):
    """Builds Torch polarization vectors transverse to propagation directions."""

    import torch  # pylint: disable=import-outside-toplevel

    preferred_axis = preferred_axis / torch.clamp(
        torch.linalg.norm(preferred_axis),
        min=1e-18,
    )
    polarization = preferred_axis[None, :] - (
        torch.sum(propagation_hats * preferred_axis[None, :], dim=1)[:, None] * propagation_hats
    )
    norms = torch.linalg.norm(polarization, dim=1)
    for alt in (
        torch.tensor(
            [0.0, 1.0, 0.0], dtype=propagation_hats.dtype, device=propagation_hats.device
        ),
        torch.tensor(
            [1.0, 0.0, 0.0], dtype=propagation_hats.dtype, device=propagation_hats.device
        ),
    ):
        missing = norms <= 1e-18
        if not bool(torch.any(missing).item()):
            break
        alt_pol = alt[None, :] - (
            torch.sum(propagation_hats * alt[None, :], dim=1)[:, None] * propagation_hats
        )
        polarization = torch.where(missing[:, None], alt_pol, polarization)
        norms = torch.linalg.norm(polarization, dim=1)
    return polarization / torch.clamp(norms[:, None], min=1e-18)


def _torch_local_incidence_bases(k_hat_inc, normals, preferred_axis):
    """Builds Torch TE/TM bases for incident wave directions and face normals."""

    import torch  # pylint: disable=import-outside-toplevel

    te_hat = torch_vector_cross(k_hat_inc, normals, dim=1)
    te_norms = torch.linalg.norm(te_hat, dim=1)
    # TE/TM are degenerate at normal incidence. Avoid amplifying roundoff in
    # the almost-zero cross product when the entire scene is rotated.
    singular = te_norms <= math.sqrt(torch.finfo(k_hat_inc.dtype).eps)
    if bool(torch.any(singular).item()):
        fallback_te = _torch_transverse_polarizations(
            k_hat_inc,
            preferred_axis=preferred_axis,
        )
        te_hat = torch.where(singular[:, None], fallback_te, te_hat)
        te_norms = torch.linalg.norm(te_hat, dim=1)
    te_hat = te_hat / torch.clamp(te_norms[:, None], min=1e-18)
    tm_hat = torch_vector_cross(te_hat, k_hat_inc, dim=1)
    tm_hat = _torch_normalize_rows(tm_hat, fallback=normals)
    return te_hat, tm_hat


class HumanVisibilityScene:
    """Mitsuba-backed human-only visibility accelerator."""

    def __init__(self, faces: np.ndarray, vertices: np.ndarray, *, name: str):
        self._name = str(name)
        self._faces = np.asarray(faces, dtype=np.uint32)
        self._adjacent_face_pairs = _visibility_adjacent_face_pairs(self._faces)
        self._mesh = mi.Mesh(
            name=self._name,
            vertex_count=int(np.asarray(vertices).shape[0]),
            face_count=int(self._faces.shape[0]),
            has_vertex_normals=False,
            has_vertex_texcoords=False,
        )
        mesh_params = mi.traverse(self._mesh)
        mesh_params["vertex_positions"] = mi.Float(
            np.asarray(vertices, dtype=np.float32).reshape(-1)
        )
        mesh_params["faces"] = mi.UInt(self._faces.reshape(-1))
        mesh_params.update()

        self._scene = mi.load_dict(
            {
                "type": "scene",
                self._name: self._mesh,
            }
        )
        self._scene_params = mi.traverse(self._scene)

    @property
    def scene(self):
        """Mitsuba scene containing the mutable human visibility mesh."""

        return self._scene

    @property
    def adjacent_face_pairs(self) -> np.ndarray:
        """Returns cached pairs of mesh faces that share an edge."""

        return self._adjacent_face_pairs

    def update_vertices(self, vertices: np.ndarray) -> None:
        """Updates the mesh vertices while preserving the acceleration structure."""

        self._scene_params[f"{self._name}.vertex_positions"] = mi.Float(
            np.asarray(vertices, dtype=np.float32).reshape(-1)
        )
        self._scene_params.update()


def po_channel(
    vertices: np.ndarray,
    faces: np.ndarray,
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    frequency_hz: float,
    material,
    *,
    virtual_channel_tx_indices: np.ndarray | None = None,
    virtual_channel_rx_indices: np.ndarray | None = None,
    virtual_channel_order: str = "tx_major",
    visibility: VisibilityMode = "fractional_shadow_fade",
    visibility_state: FaceVisibilityState | None = None,
    visibility_scene: HumanVisibilityScene | None = None,
    static_visibility_scene=None,
    po_visibility_samples_per_face: int = 4,
    visibility_fade_steps: int = 1,
    visibility_fade_advance: bool = True,
    po_adaptive_visibility_sampling: bool = True,
    po_adaptive_visibility_edge_margin: float = 0.15,
    precomputed_raw_fractional_visibility: np.ndarray | None = None,
    precomputed_visible_sample_counts: np.ndarray | None = None,
    po_integration_mode: POIntegrationMode = "face_centroid",
    po_quadrature_phase_span_scale_rad: float | None = None,
    po_curvature_error_rad: float = 0.005,
    po_quadrature_max_refinement_depth: int = 16,
    po_quadrature_max_subfaces_per_parent: int = 0,
    progress_callback: Callable[[dict], None] | None = None,
    compute_backend: ComputeBackend = "numpy",
    compute_precision: ComputePrecision = "float64",
    polarization_axis: np.ndarray | None = None,
    max_frequency_offset_hz: float | None = None,
    receiver_fmcw=None,
) -> POChannelResult:
    """Evaluates a human-only PO virtual channel over one chirp."""
    return evaluate_po_channel(
        vertices,
        faces,
        settings=POChannelSettings(
            array=POArraySettings(
                tx_positions=tx_positions,
                rx_positions=rx_positions,
                frequency_hz=frequency_hz,
                material=material,
                virtual_channel_tx_indices=virtual_channel_tx_indices,
                virtual_channel_rx_indices=virtual_channel_rx_indices,
                virtual_channel_order=virtual_channel_order,
                polarization_axis=polarization_axis,
                max_frequency_offset_hz=max_frequency_offset_hz,
                receiver_fmcw=receiver_fmcw,
            ),
            visibility=POVisibilitySettings(
                visibility=visibility,
                visibility_state=visibility_state,
                visibility_scene=visibility_scene,
                static_visibility_scene=static_visibility_scene,
                po_visibility_samples_per_face=po_visibility_samples_per_face,
                visibility_fade_steps=visibility_fade_steps,
                visibility_fade_advance=visibility_fade_advance,
                po_adaptive_visibility_sampling=po_adaptive_visibility_sampling,
                po_adaptive_visibility_edge_margin=po_adaptive_visibility_edge_margin,
                precomputed_raw_fractional_visibility=precomputed_raw_fractional_visibility,
                precomputed_visible_sample_counts=precomputed_visible_sample_counts,
            ),
            integration=POIntegrationSettings(
                po_integration_mode=po_integration_mode,
                po_quadrature_phase_span_scale_rad=po_quadrature_phase_span_scale_rad,
                po_curvature_error_rad=po_curvature_error_rad,
                po_quadrature_max_refinement_depth=po_quadrature_max_refinement_depth,
                po_quadrature_max_subfaces_per_parent=po_quadrature_max_subfaces_per_parent,
            ),
            compute=POComputeSettings(
                progress_callback=progress_callback,
                compute_backend=compute_backend,
                compute_precision=compute_precision,
            ),
        ),
    )


def evaluate_po_channel(
    vertices: np.ndarray, faces: np.ndarray, *, settings: POChannelSettings
) -> POChannelResult:
    """Run with grouped settings; the legacy keyword entry point remains supported."""
    if not isinstance(settings, POChannelSettings):
        raise TypeError("settings must be a POChannelSettings")
    return _POChannelSolve(settings, vertices, faces).run()


class _POChannelSolve:
    """Per-run preparation, solver passes and result construction."""

    def __init__(self, settings: POChannelSettings, vertices, faces):
        self.settings = settings
        self.po_quadrature_phase_span_scale_rad = (
            self.settings.integration.po_quadrature_phase_span_scale_rad
        )
        self.rx_positions = self.settings.array.rx_positions
        self.tx_positions = self.settings.array.tx_positions
        self.visibility_scene = self.settings.visibility.visibility_scene
        self.vertices = vertices
        self.faces = faces
        self.frequency_hz = self.settings.array.frequency_hz
        self.po_visibility_samples_per_face = (
            self.settings.visibility.po_visibility_samples_per_face
        )
        self.visibility_fade_steps = self.settings.visibility.visibility_fade_steps
        self.po_integration_mode = self.settings.integration.po_integration_mode
        self.po_curvature_error_rad = self.settings.integration.po_curvature_error_rad
        self.po_quadrature_max_refinement_depth = (
            self.settings.integration.po_quadrature_max_refinement_depth
        )
        self.po_quadrature_max_subfaces_per_parent = (
            self.settings.integration.po_quadrature_max_subfaces_per_parent
        )
        self.compute_backend = self.settings.compute.compute_backend
        self.compute_precision = self.settings.compute.compute_precision

    def _stage_prepare(self):
        """Prepare this run using its settings and prior stage outputs."""
        validate_compute_backend(self.compute_backend)
        self.float_dtype, complex_dtype = _numpy_dtypes_for_precision(self.compute_precision)
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.uint32)
        self.tx_positions = np.asarray(self.tx_positions, dtype=np.float64)
        self.rx_positions = np.asarray(self.rx_positions, dtype=np.float64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must have shape [num_vertices, 3]")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must have shape [num_faces, 3]")
        if self.tx_positions.ndim != 2 or self.tx_positions.shape[1] != 3:
            raise ValueError("tx_positions must have shape [num_tx, 3]")
        if self.rx_positions.ndim != 2 or self.rx_positions.shape[1] != 3:
            raise ValueError("rx_positions must have shape [num_rx, 3]")
        if self.po_visibility_samples_per_face <= 0:
            raise ValueError("po_visibility_samples_per_face must be positive")
        if self.visibility_fade_steps <= 0:
            raise ValueError("visibility_fade_steps must be positive")
        if self.po_integration_mode not in (
            "face_centroid",
            "parent_face_quadrature",
            "parent_face_adaptive_analytic",
            "parent_face_far_field_analytic",
        ):
            raise ValueError(
                "po_integration_mode must be 'face_centroid', "
                "'parent_face_quadrature', 'parent_face_adaptive_analytic', or "
                "'parent_face_far_field_analytic'"
            )
        if self.po_quadrature_phase_span_scale_rad is None:
            self.po_quadrature_phase_span_scale_rad = default_phase_span_scale_rad(
                self.po_integration_mode
            )
        if self.po_quadrature_phase_span_scale_rad <= 0.0:
            raise ValueError("po_quadrature_phase_span_scale_rad must be positive")
        if not np.isfinite(self.po_curvature_error_rad) or self.po_curvature_error_rad <= 0.0:
            raise ValueError("po_curvature_error_rad must be finite and positive")
        if self.po_quadrature_max_refinement_depth < 0:
            raise ValueError("po_quadrature_max_refinement_depth must be non-negative")
        if self.po_quadrature_max_subfaces_per_parent < 0:
            raise ValueError("po_quadrature_max_subfaces_per_parent must be non-negative")

        self.po_material = _coerce_material(self.settings.array.material)
        self.centroids, self.normals, self.areas = face_geometry(self.vertices, self.faces)

    def _stage_visibility(self):
        """Visibility this run using its settings and prior stage outputs."""
        num_faces = self.faces.shape[0]
        self.raw_visibility = np.ones(num_faces, dtype=np.float64)
        self.visible_sample_counts = np.full(
            num_faces,
            int(self.po_visibility_samples_per_face),
            dtype=np.int64,
        )
        if self.settings.visibility.precomputed_raw_fractional_visibility is not None:
            self.raw_visibility = np.asarray(
                self.settings.visibility.precomputed_raw_fractional_visibility,
                dtype=np.float64,
            ).reshape(-1)
            if self.raw_visibility.shape != (num_faces,):
                raise ValueError(
                    "precomputed_raw_fractional_visibility must have one entry per face"
                )
            if not np.all(np.isfinite(self.raw_visibility)):
                raise ValueError("precomputed_raw_fractional_visibility must be finite")
            self.raw_visibility = np.clip(self.raw_visibility, 0.0, 1.0)
            if self.settings.visibility.precomputed_visible_sample_counts is None:
                self.visible_sample_counts = np.rint(
                    self.raw_visibility * int(self.po_visibility_samples_per_face)
                ).astype(np.int64)
            else:
                self.visible_sample_counts = np.asarray(
                    self.settings.visibility.precomputed_visible_sample_counts,
                    dtype=np.int64,
                ).reshape(-1)
                if self.visible_sample_counts.shape != (num_faces,):
                    raise ValueError(
                        "precomputed_visible_sample_counts must have one entry per face"
                    )
        elif self.settings.visibility.visibility == "fractional_shadow_fade":
            if self.visibility_scene is None:
                self.visibility_scene = HumanVisibilityScene(
                    faces=self.faces,
                    vertices=self.vertices,
                    name="human-po-visibility",
                )
            else:
                self.visibility_scene.update_vertices(self.vertices)
            self.tx_phase_center = np.mean(self.tx_positions, axis=0)
            self.rx_phase_center = np.mean(self.rx_positions, axis=0)
            self.raw_visibility, self.visible_sample_counts = _fractional_face_visibility(
                vertices=self.vertices,
                faces=self.faces,
                face_normals=self.normals,
                tx_position=self.tx_phase_center,
                rx_position=self.rx_phase_center,
                visibility_scene=self.visibility_scene,
                static_visibility_scene=self.settings.visibility.static_visibility_scene,
                samples_per_face=self.po_visibility_samples_per_face,
                adaptive_edge_sampling=self.settings.visibility.po_adaptive_visibility_sampling,
                edge_front_margin=self.settings.visibility.po_adaptive_visibility_edge_margin,
            )
        elif self.settings.visibility.visibility != "none":
            raise ValueError("visibility must be 'fractional_shadow_fade' or 'none'")

        self.fade_weights, self.latched_visibility = _update_visibility_state(
            raw_fractional_visibility=self.raw_visibility,
            state=self.settings.visibility.visibility_state,
            fade_chirps=self.visibility_fade_steps,
            advance=self.settings.visibility.visibility_fade_advance,
        )
        self.effective_weights = self.latched_visibility * self.fade_weights

    def _stage_integrate(self):
        """Integrate this run using its settings and prior stage outputs."""
        selected_face_indices = np.flatnonzero(self.effective_weights > 0.0)
        self.tx_phase_center = np.mean(self.tx_positions, axis=0)
        self.rx_phase_center = np.mean(self.rx_positions, axis=0)
        self.delay_response = None
        if (
            self.settings.array.max_frequency_offset_hz is not None
            and self.po_integration_mode != "face_centroid"
        ):
            self.delay_response = PODelayResponse.for_mesh(
                self.vertices,
                self.faces,
                selected_face_indices.size,
                self.settings.array.max_frequency_offset_hz,
                compute_precision=self.compute_precision,
                receiver_fmcw=self.settings.array.receiver_fmcw,
                channel_delay_margin_s=(
                    np.max(np.linalg.norm(self.tx_positions - self.tx_phase_center, axis=1))
                    + np.max(np.linalg.norm(self.rx_positions - self.rx_phase_center, axis=1))
                )
                / c,
            )
            self.delay_response.set_reference(
                np.arange(selected_face_indices.size),
                self.centroids[selected_face_indices],
                self.tx_phase_center,
                self.rx_phase_center,
            )
        self.far_field_max_edge_m = None
        self.far_field_nearest_distance_m = None
        if self.po_integration_mode == "parent_face_quadrature":
            (
                quadrature_centroids,
                quadrature_normals,
                quadrature_areas,
                quadrature_parent_indices,
            ) = parent_face_quadrature_geometry(
                vertices=self.vertices,
                faces=self.faces,
                selected_parent_indices=selected_face_indices,
                tx_positions=self.tx_positions,
                rx_positions=self.rx_positions,
                frequency_hz=self.frequency_hz,
                phase_span_scale_rad=self.po_quadrature_phase_span_scale_rad,
                max_refinement_depth=self.po_quadrature_max_refinement_depth,
                max_subfaces_per_parent=self.po_quadrature_max_subfaces_per_parent,
                progress_callback=self.settings.compute.progress_callback,
            )
            self.path_face_indices = selected_face_indices
            self.integration_centroids = self.centroids[selected_face_indices]
            self.integration_normals = self.normals[selected_face_indices]
            self.integration_areas = self.areas[selected_face_indices]
            self.phase_center_contrib = _aggregate_parent_face_quadrature_contributions(
                selected_parent_indices=selected_face_indices,
                integration_centroids=quadrature_centroids,
                integration_normals=quadrature_normals,
                integration_areas=quadrature_areas,
                path_face_indices=quadrature_parent_indices,
                effective_weights=self.effective_weights,
                tx_phase_center=self.tx_phase_center,
                rx_phase_center=self.rx_phase_center,
                frequency_hz=self.frequency_hz,
                material=self.po_material,
                backend=self.compute_backend,
                precision=self.compute_precision,
                progress_callback=self.settings.compute.progress_callback,
                polarization_axis=self.settings.array.polarization_axis,
                delay_response=self.delay_response,
            )
            self.phase_center_delays_s = _phase_center_delays_for_centroids(
                self.integration_centroids,
                self.tx_phase_center,
                self.rx_phase_center,
                self.compute_precision,
            )
        elif self.po_integration_mode in (
            "parent_face_adaptive_analytic",
            "parent_face_far_field_analytic",
        ):
            geometry_kwargs = dict(
                vertices=self.vertices,
                faces=self.faces,
                selected_parent_indices=selected_face_indices,
                tx_positions=self.tx_positions,
                rx_positions=self.rx_positions,
                frequency_hz=self.frequency_hz,
                max_refinement_depth=self.po_quadrature_max_refinement_depth,
                max_subfaces_per_parent=self.po_quadrature_max_subfaces_per_parent,
                progress_callback=self.settings.compute.progress_callback,
            )
            if self.po_integration_mode == "parent_face_adaptive_analytic":
                (
                    far_field_centroids,
                    far_field_normals,
                    far_field_areas,
                    far_field_parent_indices,
                    far_field_triangles,
                ) = parent_face_adaptive_analytic_geometry(
                    **geometry_kwargs,
                    phase_span_scale_rad=self.po_quadrature_phase_span_scale_rad,
                    curvature_error_rad=self.po_curvature_error_rad,
                )
            else:
                (
                    far_field_centroids,
                    far_field_normals,
                    far_field_areas,
                    far_field_parent_indices,
                    far_field_triangles,
                    self.far_field_max_edge_m,
                    self.far_field_nearest_distance_m,
                ) = parent_face_far_field_analytic_geometry(**geometry_kwargs)
            self.path_face_indices = selected_face_indices
            self.integration_centroids = self.centroids[selected_face_indices]
            self.integration_normals = self.normals[selected_face_indices]
            self.integration_areas = self.areas[selected_face_indices]
            self.phase_center_contrib = _aggregate_parent_face_far_field_analytic_contributions(
                selected_parent_indices=selected_face_indices,
                integration_triangles=far_field_triangles,
                integration_centroids=far_field_centroids,
                integration_normals=far_field_normals,
                integration_areas=far_field_areas,
                path_face_indices=far_field_parent_indices,
                effective_weights=self.effective_weights,
                tx_phase_center=self.tx_phase_center,
                rx_phase_center=self.rx_phase_center,
                frequency_hz=self.frequency_hz,
                material=self.po_material,
                backend=self.compute_backend,
                precision=self.compute_precision,
                progress_callback=self.settings.compute.progress_callback,
                polarization_axis=self.settings.array.polarization_axis,
                delay_response=self.delay_response,
            )
            self.phase_center_delays_s = _phase_center_delays_for_centroids(
                self.integration_centroids,
                self.tx_phase_center,
                self.rx_phase_center,
                self.compute_precision,
            )
        else:
            self.path_face_indices = selected_face_indices
            self.integration_centroids = self.centroids[selected_face_indices]
            self.integration_normals = self.normals[selected_face_indices]
            self.integration_areas = self.areas[selected_face_indices]

            local_indices = np.arange(self.path_face_indices.size, dtype=np.int64)
            self.phase_center_contrib, self.phase_center_delays_s = (
                vector_facet_face_contributions(
                    face_indices=local_indices,
                    centroids=self.integration_centroids,
                    normals=self.integration_normals,
                    areas=self.integration_areas,
                    tx_position=self.tx_phase_center,
                    rx_position=self.rx_phase_center,
                    frequency_hz=self.frequency_hz,
                    material=self.po_material,
                    weights=self.effective_weights[self.path_face_indices],
                    backend=self.compute_backend,
                    precision=self.compute_precision,
                    polarization_axis=self.settings.array.polarization_axis,
                )
            )

    def _stage_result(self):
        """Result this run using its settings and prior stage outputs."""
        coefficients, delays_s, _, _ = project_phase_center_contributions_to_virtual_channels(
            self.phase_center_contrib,
            self.integration_centroids,
            self.tx_positions,
            self.rx_positions,
            self.frequency_hz,
            virtual_channel_tx_indices=self.settings.array.virtual_channel_tx_indices,
            virtual_channel_rx_indices=self.settings.array.virtual_channel_rx_indices,
            virtual_channel_order=self.settings.array.virtual_channel_order,
            precision=self.compute_precision,
        )
        if coefficients.shape[0] == 1:
            delays_s = np.asarray(self.phase_center_delays_s, dtype=self.float_dtype).reshape(-1)

        if self.delay_response is not None:
            gains, _, _, _ = project_phase_center_contributions_to_virtual_channels(
                np.ones(self.path_face_indices.size, complex),
                self.integration_centroids,
                self.tx_positions,
                self.rx_positions,
                self.frequency_hz,
                virtual_channel_tx_indices=self.settings.array.virtual_channel_tx_indices,
                virtual_channel_rx_indices=self.settings.array.virtual_channel_rx_indices,
                virtual_channel_order=self.settings.array.virtual_channel_order,
                precision=self.compute_precision,
            )
            self.delay_response = self.delay_response.with_channel_gains(gains)
        state_out = FaceVisibilityState(
            was_visible=self.raw_visibility > 0.0,
            fade_weights=self.fade_weights.copy(),
            latched_fractional_visibility=self.latched_visibility.copy(),
        )
        total_area = float(np.sum(self.areas))
        if total_area > 0.0:
            effective_visible_area_fraction = float(
                np.sum(self.areas * self.effective_weights) / total_area
            )
        else:
            effective_visible_area_fraction = 0.0
        return POChannelResult(
            coefficients=coefficients,
            delays_s=delays_s,
            face_indices=self.path_face_indices.astype(np.int64),
            face_centroids=self.integration_centroids.copy(),
            face_normals=self.integration_normals.copy(),
            face_areas=self.integration_areas.copy(),
            total_area=total_area,
            effective_visible_area_fraction=effective_visible_area_fraction,
            raw_fractional_visibility=self.raw_visibility,
            latched_fractional_visibility=self.latched_visibility,
            temporal_fade_weights=self.fade_weights,
            effective_weights=self.effective_weights,
            visible_sample_counts=self.visible_sample_counts,
            sample_count_per_face=int(self.po_visibility_samples_per_face),
            phase_center_contributions=self.phase_center_contrib,
            phase_center_channel=complex(np.sum(self.phase_center_contrib)),
            visibility_state=state_out,
            far_field_max_edge_m=self.far_field_max_edge_m,
            far_field_nearest_distance_m=self.far_field_nearest_distance_m,
            delay_response=self.delay_response,
        )

    def run(self):
        self._stage_prepare()
        self._stage_visibility()
        self._stage_integrate()
        return self._stage_result()


def _coerce_material(material) -> PhysicalOpticsMaterial:
    """Converts simulator or Sionna-like radio materials to PO material params."""

    if isinstance(material, PhysicalOpticsMaterial):
        return material

    if all(
        hasattr(material, attr)
        for attr in (
            "relative_permittivity",
            "conductivity",
            "thickness",
        )
    ):
        return PhysicalOpticsMaterial(
            relative_permittivity=float(np.asarray(material.relative_permittivity).reshape(-1)[0]),
            conductivity_s_per_m=float(np.asarray(material.conductivity).reshape(-1)[0]),
            thickness_m=float(np.asarray(material.thickness).reshape(-1)[0]),
            model="slab",
            surface_attenuation_thickness_m=0.0,
        )

    raise TypeError("material must be a PhysicalOpticsMaterial or a RadioMaterial-like object")


def _normalize_rows(
    vectors: np.ndarray,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    """Normalizes vector rows, substituting a fallback for zero rows."""

    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    result = vectors.copy()
    valid = norms > 1e-18
    result[valid] /= norms[valid, None]

    if not np.all(valid):
        if fallback is None:
            raise ValueError("Zero-length vector row without fallback")
        fallback = np.asarray(fallback, dtype=np.float64)
        if fallback.ndim == 1:
            fallback = np.broadcast_to(fallback, result.shape)
        result[~valid] = fallback[~valid]

    return result


def _transverse_polarizations(
    propagation_hats: np.ndarray,
    preferred_axis: np.ndarray,
) -> np.ndarray:
    """Builds polarization vectors transverse to propagation directions."""

    propagation_hats = np.asarray(propagation_hats, dtype=np.float64)
    preferred_axis = np.asarray(preferred_axis, dtype=np.float64)
    preferred_axis = preferred_axis / max(np.linalg.norm(preferred_axis), 1e-18)

    polarization = preferred_axis[None, :] - (
        np.einsum("ij,j->i", propagation_hats, preferred_axis)[:, None] * propagation_hats
    )
    norms = np.linalg.norm(polarization, axis=1)

    for alt_axis in (
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
    ):
        missing = norms <= 1e-18
        if not np.any(missing):
            break
        polarization[missing] = alt_axis[None, :] - (
            np.einsum("ij,j->i", propagation_hats[missing], alt_axis)[:, None]
            * propagation_hats[missing]
        )
        norms = np.linalg.norm(polarization, axis=1)

    polarization /= np.maximum(norms[:, None], 1e-18)
    return polarization


def _local_incidence_bases(
    k_hat_inc: np.ndarray,
    normals: np.ndarray,
    preferred_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds TE/TM unit bases for incident wave directions and face normals."""

    te_hat = np.cross(k_hat_inc, normals)
    te_norms = np.linalg.norm(te_hat, axis=1)
    # Near normal incidence both polarizations have the same physical action;
    # use a stable transverse basis instead of normalizing cancellation noise.
    singular = te_norms <= np.sqrt(np.finfo(np.asarray(k_hat_inc).dtype).eps)
    if np.any(singular):
        te_hat[singular] = _transverse_polarizations(
            k_hat_inc[singular],
            preferred_axis=preferred_axis,
        )
        te_norms = np.linalg.norm(te_hat, axis=1)
    te_hat /= np.maximum(te_norms[:, None], 1e-18)

    tm_hat = np.cross(te_hat, k_hat_inc)
    tm_hat = _normalize_rows(tm_hat, fallback=normals)
    return te_hat, tm_hat


def _face_sample_barycentrics(samples_per_face: int) -> np.ndarray:
    """Returns deterministic barycentric samples used for visibility tests."""

    if samples_per_face <= 0:
        raise ValueError("samples_per_face must be positive")

    base = [
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        (0.6, 0.2, 0.2),
        (0.2, 0.6, 0.2),
        (0.2, 0.2, 0.6),
    ]
    samples = list(base[:samples_per_face])
    k = 1
    while len(samples) < samples_per_face:
        u = (k * 0.6180339887498949) % 1.0
        v = (k * 0.4142135623730950) % 1.0
        if u + v >= 1.0:
            u = 1.0 - u
            v = 1.0 - v
        eps = 0.05
        a = eps + (1.0 - 3.0 * eps) * u
        b = eps + (1.0 - 3.0 * eps) * v
        c_bar = 1.0 - a - b
        if c_bar > eps:
            samples.append((a, b, c_bar))
        k += 1
    return np.asarray(samples, dtype=np.float64)


def _fractional_face_visibility(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_normals: np.ndarray,
    tx_position: np.ndarray,
    rx_position: np.ndarray,
    visibility_scene: HumanVisibilityScene,
    samples_per_face: int,
    *,
    static_visibility_scene=None,
    adaptive_edge_sampling: bool = False,
    edge_front_margin: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes fractional face visibility using deterministic barycentric samples."""

    tri = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    normals = np.asarray(face_normals, dtype=np.float64)
    sample_bary = _face_sample_barycentrics(samples_per_face)
    sample_points = np.einsum("sv,fvk->fsk", sample_bary, tri)

    counts = np.zeros(faces.shape[0], dtype=np.int64)
    sample_totals = np.zeros(faces.shape[0], dtype=np.int64)
    if adaptive_edge_sampling and sample_bary.shape[0] > 1:
        points = sample_points[:, 0, :]
        visible_tx = _samples_visible_from_origin(
            points=points,
            normals=normals,
            origin=np.asarray(tx_position, dtype=np.float64),
            scene=visibility_scene.scene,
            static_scene=static_visibility_scene,
        )
        visible_rx = _samples_visible_from_origin(
            points=points,
            normals=normals,
            origin=np.asarray(rx_position, dtype=np.float64),
            scene=visibility_scene.scene,
            static_scene=static_visibility_scene,
        )
        centroid_visible = visible_tx & visible_rx
        counts += centroid_visible.astype(np.int64)
        sample_totals += 1
        edge_faces = _visibility_edge_faces(
            faces=faces,
            centroid_visible=centroid_visible,
            adjacent_face_pairs=visibility_scene.adjacent_face_pairs,
        )
        if edge_front_margin > 0.0:
            edge_faces |= _grazing_visibility_faces(
                points=points,
                normals=normals,
                tx_position=np.asarray(tx_position, dtype=np.float64),
                rx_position=np.asarray(rx_position, dtype=np.float64),
                edge_front_margin=float(edge_front_margin),
            )
        candidate_indices = np.flatnonzero(edge_faces)
        for sample_index in range(1, sample_bary.shape[0]):
            if candidate_indices.size == 0:
                break
            points = sample_points[candidate_indices, sample_index, :]
            candidate_normals = normals[candidate_indices]
            visible_tx = _samples_visible_from_origin(
                points=points,
                normals=candidate_normals,
                origin=np.asarray(tx_position, dtype=np.float64),
                scene=visibility_scene.scene,
                static_scene=static_visibility_scene,
            )
            visible_rx = _samples_visible_from_origin(
                points=points,
                normals=candidate_normals,
                origin=np.asarray(rx_position, dtype=np.float64),
                scene=visibility_scene.scene,
                static_scene=static_visibility_scene,
            )
            counts[candidate_indices] += (visible_tx & visible_rx).astype(np.int64)
            sample_totals[candidate_indices] += 1
        return counts / np.maximum(sample_totals, 1), counts

    for sample_index in range(sample_bary.shape[0]):
        points = sample_points[:, sample_index, :]
        visible_tx = _samples_visible_from_origin(
            points=points,
            normals=normals,
            origin=np.asarray(tx_position, dtype=np.float64),
            scene=visibility_scene.scene,
            static_scene=static_visibility_scene,
        )
        visible_rx = _samples_visible_from_origin(
            points=points,
            normals=normals,
            origin=np.asarray(rx_position, dtype=np.float64),
            scene=visibility_scene.scene,
            static_scene=static_visibility_scene,
        )
        counts += (visible_tx & visible_rx).astype(np.int64)

    return counts / float(sample_bary.shape[0]), counts


def _visibility_adjacent_face_pairs(faces: np.ndarray) -> np.ndarray:
    """Returns mesh-face pairs that share an edge in legacy traversal order."""

    faces = np.asarray(faces, dtype=np.int64)
    edge_to_face: dict[tuple[int, int], int] = {}
    adjacent_pairs: list[tuple[int, int]] = []
    for face_index, face in enumerate(faces):
        for edge in (
            tuple(sorted((int(face[0]), int(face[1])))),
            tuple(sorted((int(face[1]), int(face[2])))),
            tuple(sorted((int(face[2]), int(face[0])))),
        ):
            other = edge_to_face.get(edge)
            if other is None:
                edge_to_face[edge] = face_index
            else:
                adjacent_pairs.append((other, face_index))
    return np.asarray(adjacent_pairs, dtype=np.int64).reshape(-1, 2)


def _visibility_edge_faces(
    faces: np.ndarray,
    centroid_visible: np.ndarray,
    *,
    adjacent_face_pairs: np.ndarray | None = None,
) -> np.ndarray:
    """Finds faces adjacent to a face with different centroid visibility."""

    faces = np.asarray(faces, dtype=np.int64)
    centroid_visible = np.asarray(centroid_visible, dtype=bool).reshape(-1)
    edge_faces = np.zeros(faces.shape[0], dtype=bool)
    pairs = (
        _visibility_adjacent_face_pairs(faces)
        if adjacent_face_pairs is None
        else np.asarray(adjacent_face_pairs, dtype=np.int64).reshape(-1, 2)
    )
    if pairs.size == 0:
        return edge_faces
    changed = centroid_visible[pairs[:, 0]] != centroid_visible[pairs[:, 1]]
    edge_faces[pairs[changed].reshape(-1)] = True
    return edge_faces


def _grazing_visibility_faces(
    points: np.ndarray,
    normals: np.ndarray,
    tx_position: np.ndarray,
    rx_position: np.ndarray,
    edge_front_margin: float,
) -> np.ndarray:
    """Marks front-facing faces near the grazing-visibility boundary."""

    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    tx_margin = _front_facing_margin(points, normals, tx_position)
    rx_margin = _front_facing_margin(points, normals, rx_position)
    min_margin = np.minimum(tx_margin, rx_margin)
    return (min_margin > 0.0) & (min_margin <= float(edge_front_margin))


def _front_facing_margin(
    points: np.ndarray,
    normals: np.ndarray,
    origin: np.ndarray,
) -> np.ndarray:
    """Returns cosine front-facing margins from face points toward an origin."""

    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    to_origin = origin[None, :] - points
    distances = np.linalg.norm(to_origin, axis=1)
    toward_origin = to_origin / np.maximum(distances[:, None], 1e-18)
    return np.einsum("ij,ij->i", normals, toward_origin)


def _samples_visible_from_origin(
    points: np.ndarray,
    normals: np.ndarray,
    origin: np.ndarray,
    scene,
    static_scene=None,
) -> np.ndarray:
    """Returns whether sampled points are front-facing and unoccluded."""

    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)

    to_origin = origin[None, :] - points
    distances = np.linalg.norm(to_origin, axis=1)
    toward_origin = to_origin / np.maximum(distances[:, None], 1e-18)
    ray_directions = -toward_origin
    front_facing = np.einsum("ij,ij->i", normals, toward_origin) > 1e-9
    visible = np.zeros(points.shape[0], dtype=bool)
    candidate_indices = np.flatnonzero(front_facing & (distances > 1e-9))
    if candidate_indices.size == 0:
        return visible

    directions = ray_directions[candidate_indices]
    origins = origin[None, :] + 1e-5 * directions
    maxt = np.maximum(distances[candidate_indices] - 2e-5, 0.0)
    ray = mi.Ray3f(
        o=mi.Point3f(origins[:, 0], origins[:, 1], origins[:, 2]),
        d=mi.Vector3f(directions[:, 0], directions[:, 1], directions[:, 2]),
        maxt=mi.Float(maxt),
        time=0.0,
        wavelengths=mi.Color0f(),
    )
    pi_hit = scene.ray_intersect_preliminary(ray)
    human_clear = ~np.asarray(pi_hit.is_valid(), dtype=bool)
    visible_indices = candidate_indices[human_clear]
    if static_scene is None or visible_indices.size == 0:
        visible[visible_indices] = True
        return visible

    static_directions = ray_directions[visible_indices]
    static_origins = origin[None, :] + 1e-5 * static_directions
    static_maxt = np.maximum(distances[visible_indices] - 2e-5, 0.0)
    static_ray = mi.Ray3f(
        o=mi.Point3f(
            static_origins[:, 0],
            static_origins[:, 1],
            static_origins[:, 2],
        ),
        d=mi.Vector3f(
            static_directions[:, 0],
            static_directions[:, 1],
            static_directions[:, 2],
        ),
        maxt=mi.Float(static_maxt),
        time=0.0,
        wavelengths=mi.Color0f(),
    )
    static_hit = np.asarray(
        static_scene.ray_intersect_preliminary(static_ray).is_valid(),
        dtype=bool,
    )
    visible[visible_indices] = ~static_hit
    return visible


def _update_visibility_state(
    raw_fractional_visibility: np.ndarray,
    state: FaceVisibilityState | None,
    fade_chirps: int,
    advance: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Updates visibility fade weights and latched fractional visibility.

    Each call with ``advance`` moves fading facets one step of
    ``1/fade_chirps``. With ``advance`` false the previous weights are held,
    so visibility stays constant within a frame.
    """

    raw_fractional_visibility = np.asarray(raw_fractional_visibility, dtype=np.float64).reshape(-1)
    raw_visible = raw_fractional_visibility > 0.0
    if state is None:
        fade_weights = raw_visible.astype(np.float64)
        return fade_weights, raw_fractional_visibility.copy()

    prev_visible = np.asarray(state.was_visible, dtype=bool).reshape(-1)
    prev_fade = np.asarray(state.fade_weights, dtype=np.float64).reshape(-1)
    prev_latched = np.asarray(
        state.latched_fractional_visibility,
        dtype=np.float64,
    ).reshape(-1)
    if (
        prev_visible.shape != raw_visible.shape
        or prev_fade.shape != raw_visible.shape
        or prev_latched.shape != raw_visible.shape
    ):
        raise ValueError("visibility state shape mismatch")

    if not advance:
        return prev_fade.copy(), prev_latched.copy()

    step = 1.0 / float(fade_chirps)
    fade_weights = prev_fade.copy()
    latched = prev_latched.copy()

    fading_out = ~raw_visible
    fade_weights[fading_out] = np.maximum(prev_fade[fading_out] - step, 0.0)
    latched[fading_out & (prev_fade <= 0.0)] = 0.0

    currently_visible = raw_visible
    continuing_visible = currently_visible & prev_visible & (prev_fade >= 1.0 - 1e-12)
    fade_weights[continuing_visible] = 1.0
    ramp_visible = currently_visible & ~continuing_visible
    fade_weights[ramp_visible] = np.minimum(prev_fade[ramp_visible] + step, 1.0)
    latched[currently_visible] = raw_fractional_visibility[currently_visible]

    return fade_weights, latched
