# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Phase-error guards shared by direct and image-source PO."""

import numpy as np
import pytest
from scipy.constants import c

from mmWaveRadar.simulation.po_incremental import update_po_facets, vertex_path_offsets
from mmWaveRadar.simulation.types import POConfig


def test_vertex_offsets_subtract_the_facet_centre_path():
    triangle = np.array([[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]])
    offsets = vertex_path_offsets(triangle, np.zeros(3), np.zeros(3))
    np.testing.assert_allclose(offsets, 6.0 - 2.0 * np.sqrt(3.0))


def test_phase_trigger_accumulates_since_exact_solve_and_resets_after_refresh():
    state = None
    counts = []

    def integrate(positions):
        counts.append(positions.size)
        return np.ones((1, positions.size), complex), np.ones(positions.size, complex)

    # k=1 rad/m: the centre moves each time, while only the residual vertex
    # phase counts toward the 1 rad trigger. Small steps accumulate.
    for centre_length, residual, expected in (
        (2.0, 0.0, [0]),
        (3.0, 0.6, []),
        (4.0, 1.2, [0]),
        (5.0, 1.7, []),
        (6.0, 2.4, [0]),
    ):
        state, recomputed = update_po_facets(
            previous=state,
            selected=np.array([0]),
            vertex_offsets=np.array([[0.0, residual, -residual]]),
            path_lengths=np.array([[centre_length]]),
            spreads=np.ones((1, 1)),
            phase_lengths=np.array([centre_length]),
            phase_spreads=np.ones(1),
            weights=np.ones(1),
            frequency_hz=c / (2.0 * np.pi),
            phase_error_threshold_rad=1.0,
            delay_response=None,
            response_indices=np.array([0]),
            integrate=integrate,
            dtype=np.complex128,
        )
        np.testing.assert_array_equal(recomputed, expected)
        if not expected:
            np.testing.assert_allclose(state.coefficients, np.exp(-1j))
    assert counts == [1, 1, 1]


@pytest.mark.parametrize("change", ["rotation", "view", "stretch"])
@pytest.mark.parametrize("rx", [np.zeros(3), np.array([0.0, 0.0, -2.0])])
def test_phase_residual_detects_motion_for_direct_and_image_source_paths(change, rx):
    triangle = np.array([[[2.0, -0.05, -0.03], [2.0, 0.0, 0.06], [2.0, 0.05, -0.03]]])
    moved = triangle.copy()
    centre = triangle.mean(axis=1, keepdims=True)
    if change == "rotation":
        angle = 0.1
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        moved = (triangle - centre) @ rotation.T + centre
    elif change == "view":
        moved += [0.0, 0.1, 0.0]
    else:
        moved = (triangle - centre) * 1.1 + centre
    phase_error = (
        2.0
        * np.pi
        * 60e9
        / c
        * np.max(
            np.abs(
                vertex_path_offsets(moved, np.zeros(3), rx)
                - vertex_path_offsets(triangle, np.zeros(3), rx)
            )
        )
    )
    assert phase_error > POConfig().po_refresh_on_phase_error_rad


@pytest.mark.parametrize("threshold", [-1.0, np.nan, np.inf, None])
def test_phase_threshold_requires_a_finite_nonnegative_value(threshold):
    with pytest.raises(ValueError, match="po_refresh_on_phase_error_rad"):
        POConfig(po_refresh_on_phase_error_rad=threshold)


@pytest.mark.parametrize("scale", [0.1, 1.0])
@pytest.mark.parametrize("change", ["area", "normal"])
def test_amplitude_guards_accumulate_since_last_exact_solve(change, scale):
    state = None
    # Each step is below the guard; two steps together exceed it.
    for step, expected in ((0, [0]), (1, []), (2, [0]), (3, []), (4, [0])):
        angle = np.deg2rad(0.6 * step * scale) if change == "normal" else 0.0
        state, recomputed = update_po_facets(
            previous=state,
            selected=np.array([0]),
            vertex_offsets=np.zeros((1, 3)),
            path_lengths=np.ones((1, 1)),
            spreads=np.ones((1, 1)),
            phase_lengths=np.ones(1),
            phase_spreads=np.ones(1),
            weights=np.ones(1),
            frequency_hz=77e9,
            phase_error_threshold_rad=0.1,
            delay_response=None,
            response_indices=np.array([0]),
            integrate=lambda ids: (np.ones((1, ids.size), complex), np.ones(ids.size, complex)),
            dtype=np.complex128,
            area_fraction_threshold=0.02 * scale,
            normal_threshold_deg=1.0 * scale,
            areas=np.array([1.0 + 0.011 * step * scale if change == "area" else 1.0]),
            normals=np.array([[np.cos(angle), np.sin(angle), 0.0]]),
        )
        np.testing.assert_array_equal(recomputed, expected)


def test_small_facet_stretch_is_below_phase_threshold_but_refreshes_amplitude():
    triangle = np.array([[[2.0, -0.01, -0.01], [2.0, 0.0, 0.01], [2.0, 0.01, -0.01]]])
    centre = triangle.mean(1, keepdims=True)
    stretched = (triangle - centre) * 1.3 + centre
    old_offsets = vertex_path_offsets(triangle, np.zeros(3), np.zeros(3))
    new_offsets = vertex_path_offsets(stretched, np.zeros(3), np.zeros(3))
    assert 2.0 * np.pi * 77e9 / c * abs(new_offsets - old_offsets).max() < 0.1
    state = None
    for offsets, area in ((old_offsets, 1.0), (new_offsets, 1.69)):
        state, recomputed = update_po_facets(
            previous=state,
            selected=np.array([0]),
            vertex_offsets=offsets,
            path_lengths=np.full((1, 1), 4.0),
            spreads=np.ones((1, 1)),
            phase_lengths=np.full(1, 4.0),
            phase_spreads=np.ones(1),
            weights=np.ones(1),
            frequency_hz=77e9,
            phase_error_threshold_rad=0.1,
            delay_response=None,
            response_indices=np.array([0]),
            integrate=lambda ids: (
                np.full((1, ids.size), area, complex),
                np.full(ids.size, area, complex),
            ),
            dtype=np.complex128,
            areas=np.array([area]),
            normals=np.array([[-1.0, 0.0, 0.0]]),
        )
        np.testing.assert_array_equal(recomputed, [0])
    np.testing.assert_allclose(state.coefficients, 1.69)


def test_line_of_sight_roll_changes_normal_without_changing_vertex_phase():
    triangle = np.array([[[0.0, -0.005, -0.005], [0.0, 0.0, 0.01], [0.0, 0.005, -0.005]]])
    tilt = 0.3
    rotation = np.array(
        [[np.cos(tilt), -np.sin(tilt), 0.0], [np.sin(tilt), np.cos(tilt), 0.0], [0.0, 0.0, 1.0]]
    )
    triangle = triangle @ rotation.T + [2.0, 0.0, 0.0]
    angle = 0.5
    roll = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )
    moved = (triangle - [2.0, 0.0, 0.0]) @ roll.T + [2.0, 0.0, 0.0]
    old_offsets = vertex_path_offsets(triangle, np.zeros(3), np.zeros(3))
    new_offsets = vertex_path_offsets(moved, np.zeros(3), np.zeros(3))
    np.testing.assert_allclose(new_offsets, old_offsets, atol=1e-14)
    old_normal = np.array([[-np.cos(tilt), -np.sin(tilt), 0.0]])
    state = None
    for offsets, normal in ((old_offsets, old_normal), (new_offsets, old_normal @ roll.T)):
        state, recomputed = update_po_facets(
            previous=state,
            selected=np.array([0]),
            vertex_offsets=offsets,
            path_lengths=np.full((1, 1), 4.0),
            spreads=np.ones((1, 1)),
            phase_lengths=np.full(1, 4.0),
            phase_spreads=np.ones(1),
            weights=np.ones(1),
            frequency_hz=77e9,
            phase_error_threshold_rad=0.1,
            delay_response=None,
            response_indices=np.array([0]),
            integrate=lambda ids: (np.ones((1, ids.size), complex), np.ones(ids.size, complex)),
            dtype=np.complex128,
            areas=np.ones(1),
            normals=normal,
            normal_threshold_deg=1.0,
        )
        np.testing.assert_array_equal(recomputed, [0])


@pytest.mark.parametrize("name", ["po_refresh_on_area_fraction", "po_refresh_on_normal_deg"])
@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, None])
def test_geometry_threshold_requires_finite_nonnegative_value(name, value):
    with pytest.raises(ValueError, match=name):
        POConfig(**{name: value})


def test_normal_threshold_cannot_exceed_180_degrees():
    with pytest.raises(ValueError, match="po_refresh_on_normal_deg"):
        POConfig(po_refresh_on_normal_deg=181.0)


def test_geometry_threshold_defaults():
    config = POConfig()
    assert config.po_refresh_on_area_fraction == 0.3
    assert config.po_refresh_on_normal_deg == 20.0
