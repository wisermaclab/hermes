# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Shared incremental facet updates for direct and image-source human PO."""

from dataclasses import dataclass
import time

import numpy as np
from scipy.constants import c

from .po_wideband import PODelayResponse


def match_sorted_face_positions(previous_faces, selected_faces):
    """Maps sorted selected face ids to sorted previous-bank positions."""

    previous_faces = np.asarray(previous_faces, dtype=np.int64).reshape(-1)
    selected_faces = np.asarray(selected_faces, dtype=np.int64).reshape(-1)
    positions = np.full(selected_faces.shape, -1, dtype=np.int64)
    if previous_faces.size == 0 or selected_faces.size == 0:
        return positions
    candidate_positions = np.searchsorted(previous_faces, selected_faces)
    in_bounds = candidate_positions < previous_faces.size
    matched = np.zeros(selected_faces.shape, dtype=bool)
    matched[in_bounds] = previous_faces[candidate_positions[in_bounds]] == selected_faces[in_bounds]
    positions[matched] = candidate_positions[matched]
    return positions


def phase_update_coefficients(coefficients, *, old_lengths_m, new_lengths_m,
                              old_spreads_m2, new_spreads_m2, frequency_hz):
    """Updates physical channel coefficients for geometric phase/spreading."""

    coefficients = np.asarray(coefficients)
    k0 = 2. * np.pi * frequency_hz / c
    phase = np.exp(-1j * k0 * (new_lengths_m - old_lengths_m))
    spreading = old_spreads_m2 / np.maximum(new_spreads_m2, 1e-18)
    return (coefficients * phase * spreading).astype(coefficients.dtype)


@dataclass
class POFacetState:
    """Surface fields, current path geometry, and last-exact facet geometry."""

    face_indices: np.ndarray
    coefficients: np.ndarray
    phase_center_contributions: np.ndarray
    exact_face_centroids: np.ndarray
    exact_face_normals: np.ndarray
    exact_face_areas: np.ndarray
    path_lengths_m: np.ndarray
    spread_products_m2: np.ndarray
    phase_center_lengths_m: np.ndarray
    phase_center_spreads_m2: np.ndarray
    weights: np.ndarray
    delay_response: PODelayResponse | None
    response_indices: np.ndarray


def update_po_facets(
    *, previous, selected, centroids, normals, areas, path_lengths, spreads,
    phase_lengths, phase_spreads, weights, frequency_hz, normal_threshold_rad,
    centroid_displacement_threshold_m, area_relative_threshold, delay_response,
    response_indices, integrate, dtype, force_recompute=False, add_profile=None,
):
    """Apply the human-only PO update rules with a caller-supplied exact solve.

    ``integrate(positions)`` returns projected and phase-center contributions
    for only the new/drifted faces. Direct and reflected PO supply different
    integrals; matching, drift tests, phase updates, and IF-edge refreshes are
    shared. Response indices allow reflected paths to occupy a common bank.
    """

    def profile(name, start):
        if add_profile is not None:
            add_profile(name, time.perf_counter() - start)

    start = time.perf_counter()
    old_positions = match_sorted_face_positions(
        np.empty(0, dtype=int) if previous is None else previous.face_indices, selected)
    existing_mask = old_positions >= 0
    current = np.flatnonzero(existing_mask)
    old = old_positions[current]
    coefficients = np.zeros(path_lengths.shape, dtype=dtype)
    phase_center = np.zeros(selected.size, dtype=dtype)
    same_response_grid = (delay_response is None or (
        previous is not None and previous.delay_response is not None
        and np.array_equal(delay_response.frequencies_hz,
                           previous.delay_response.frequencies_hz)))
    profile("po_incremental_bank_merge", start)

    if current.size:
        start = time.perf_counter()
        scale = weights[current] / np.maximum(previous.weights[old], 1e-18)
        coefficients[:, current] = phase_update_coefficients(
            previous.coefficients[:, old] * scale[None],
            old_lengths_m=previous.path_lengths_m[:, old],
            new_lengths_m=path_lengths[:, current],
            old_spreads_m2=previous.spread_products_m2[:, old],
            new_spreads_m2=spreads[:, current], frequency_hz=frequency_hz)
        phase_center[current] = phase_update_coefficients(
            (previous.phase_center_contributions[old] * scale)[None],
            old_lengths_m=previous.phase_center_lengths_m[old][None],
            new_lengths_m=phase_lengths[current][None],
            old_spreads_m2=previous.phase_center_spreads_m2[old][None],
            new_spreads_m2=phase_spreads[current][None], frequency_hz=frequency_hz).reshape(-1)
        if delay_response is not None and same_response_grid:
            old_response = previous.response_indices[old]
            new_response = response_indices[current]
            delay_response.values[:, new_response] = phase_update_coefficients(
                previous.delay_response.values[:, old_response] * scale[None],
                old_lengths_m=previous.phase_center_lengths_m[old][None],
                new_lengths_m=phase_lengths[current][None],
                old_spreads_m2=previous.phase_center_spreads_m2[old][None],
                new_spreads_m2=phase_spreads[current][None], frequency_hz=frequency_hz)
            delay_response.min_delta_s[new_response] = previous.delay_response.min_delta_s[old_response]
            delay_response.max_delta_s[new_response] = previous.delay_response.max_delta_s[old_response]
        profile("po_incremental_phase_update", start)

    start = time.perf_counter()
    recompute = ~existing_mask
    exact_centroids, exact_normals, exact_areas = centroids.copy(), normals.copy(), areas.copy()
    if current.size:
        exact_centroids[current] = previous.exact_face_centroids[old]
        exact_normals[current] = previous.exact_face_normals[old]
        exact_areas[current] = previous.exact_face_areas[old]
        cosine = np.clip(np.einsum("ij,ij->i", previous.exact_face_normals[old].astype(float),
                                   normals[current]), -1., 1.)
        recompute[current] = (True if normal_threshold_rad <= 0.
                              else cosine < np.cos(normal_threshold_rad))
        if centroid_displacement_threshold_m is not None and centroid_displacement_threshold_m >= 0.:
            recompute[current] |= (np.linalg.norm(
                centroids[current] - previous.exact_face_centroids[old], axis=1)
                > centroid_displacement_threshold_m)
        if area_relative_threshold is not None and area_relative_threshold >= 0.:
            recompute[current] |= (abs(areas[current] - previous.exact_face_areas[old])
                                   / np.maximum(abs(previous.exact_face_areas[old]), 1e-18)
                                   > area_relative_threshold)
    if delay_response is not None:
        # Retained leaves must follow the current geometry for exact IF clipping.
        recompute |= delay_response.retained_paths[response_indices]
    if not same_response_grid or force_recompute:
        recompute[:] = True
    positions = np.flatnonzero(recompute)
    profile("po_incremental_drift_check", start)

    if positions.size:
        start = time.perf_counter()
        if delay_response is not None:
            delay_response.reset_paths(response_indices[positions])
        projected, surface = integrate(positions)
        coefficients[:, positions] = projected
        phase_center[positions] = surface
        exact_centroids[positions] = centroids[positions]
        exact_normals[positions] = normals[positions]
        exact_areas[positions] = areas[positions]
        profile("po_incremental_exact_recompute", start)
    return POFacetState(
        selected.copy(), coefficients, phase_center, exact_centroids, exact_normals,
        exact_areas, path_lengths, spreads, phase_lengths, phase_spreads, weights.copy(),
        delay_response, response_indices.copy(),
    ), positions
