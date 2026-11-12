# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Reference-basis checks independent of the scattering calculation."""

import numpy as np
import pytest

from mmWaveRadar.polarization import transverse_linear_basis

ROUNDING_ATOL = 32 * np.finfo(np.float64).eps


def test_basis_is_orthonormal_through_the_nist_search_window():
    az, el = np.meshgrid(
        np.deg2rad(np.linspace(-22, 22, 17)), np.deg2rad(np.linspace(-30, 15, 19))
    )
    k = np.stack((np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)), axis=-1)
    vertical, horizontal = transverse_linear_basis(3 * k)
    np.testing.assert_allclose(np.sum(vertical * horizontal, axis=-1), 0, atol=ROUNDING_ATOL)
    np.testing.assert_allclose(np.sum(vertical * k, axis=-1), 0, atol=ROUNDING_ATOL)
    np.testing.assert_allclose(np.sum(horizontal * k, axis=-1), 0, atol=ROUNDING_ATOL)
    np.testing.assert_allclose(np.linalg.norm(vertical, axis=-1), 1, rtol=0, atol=ROUNDING_ATOL)
    np.testing.assert_allclose(np.linalg.norm(horizontal, axis=-1), 1, rtol=0, atol=ROUNDING_ATOL)


def test_off_boresight_basis_does_not_inherit_independent_axis_projection_leakage():
    az, el = np.deg2rad([20, -30])
    k = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    vertical, horizontal = transverse_linear_basis(k)
    independently_projected_h = np.array([0.0, 1.0, 0.0]) - k[1] * k
    independently_projected_h /= np.linalg.norm(independently_projected_h)
    assert 20 * np.log10(abs(vertical @ independently_projected_h)) == pytest.approx(
        -14.94, abs=0.01
    )
    assert abs(vertical @ horizontal) < ROUNDING_ATOL


def test_basis_rotation_and_propagation_reversal_preserve_documented_phase_sign():
    k = np.array([1.0, 0.2, -0.3])
    vertical, horizontal = transverse_linear_basis(k)
    reversed_v, reversed_h = transverse_linear_basis(-k)
    np.testing.assert_allclose(reversed_v, vertical)
    np.testing.assert_allclose(reversed_h, -horizontal)
    angle = 0.7
    rotation = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    rotated_v, rotated_h = transverse_linear_basis(rotation @ k, rotation @ [0.0, 0.0, 1.0])
    np.testing.assert_allclose(rotated_v, rotation @ vertical, atol=ROUNDING_ATOL)
    np.testing.assert_allclose(rotated_h, rotation @ horizontal, atol=ROUNDING_ATOL)


@pytest.mark.parametrize("scale", [1e-300, 1.0, 1e200, 1e308])
def test_basis_is_invariant_to_extreme_direction_and_axis_scales(scale):
    direction = np.array([1.0, 1.0, 0.0])
    expected = transverse_linear_basis(direction)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = transverse_linear_basis(scale * direction, scale * np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=ROUNDING_ATOL)


def test_parallel_directions_use_fallback_only_on_the_degenerate_batch_rows():
    directions = np.array([[1.0, 0.2, -0.3], [0, 0, 1], [0, 0, -1]])
    vertical, horizontal = transverse_linear_basis(directions)
    regular_v, regular_h = transverse_linear_basis(directions[0])
    np.testing.assert_allclose(vertical[0], regular_v)
    np.testing.assert_allclose(horizontal[0], regular_h)
    np.testing.assert_allclose(vertical[1:], [[0, 1, 0], [0, 1, 0]])
    np.testing.assert_allclose(horizontal[1:], [[-1, 0, 0], [1, 0, 0]])
    np.testing.assert_allclose(
        transverse_linear_basis([0, 1, 0], vertical_axis=[0, 1, 0]),
        [[1, 0, 0], [0, 0, -1]],
    )


@pytest.mark.parametrize(
    "direction,axis",
    [
        ([0, 0, 0], [0, 0, 1]),
        ([1, 0, 0], [0, 0, 0]),
        ([np.nan, 0, 0], [0, 0, 1]),
    ],
)
def test_undefined_bases_are_rejected(direction, axis):
    with pytest.raises(ValueError):
        transverse_linear_basis(direction, axis)
