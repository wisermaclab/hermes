# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Unit tests for DSP profile and angle-grid utilities."""

import numpy as np
import pytest

from mmWaveRadar.dsp import direction_cosine_valid_mask, select_range_bin


def test_select_range_bin_target_mode_uses_nearest_target_bin():
    ranges_m = np.array([0.0, 1.0, 2.0, 3.0])
    profile = np.array([0.0, 1.0, 8.0, 2.0])

    selected_index, metadata = select_range_bin(
        profile,
        ranges_m,
        1.1,
        mode="target",
        search_half_width_m=1.0,
        power_floor=0.0,
    )

    assert selected_index == 1
    assert metadata["target_range_index"] == 1
    assert metadata["dominant_range_index"] == 2
    assert metadata["selected_range_mode"] == "target"
    assert metadata["range_profile_is_zero"] is False


def test_select_range_bin_dominant_near_target_mode_uses_local_peak():
    ranges_m = np.array([0.0, 1.0, 2.0, 3.0])
    profile = np.array([0.0, 1.0, 8.0, 2.0])

    selected_index, metadata = select_range_bin(
        profile,
        ranges_m,
        1.1,
        mode="dominant_near_target",
        search_half_width_m=1.0,
        power_floor=0.0,
    )

    assert selected_index == 2
    assert metadata["target_range_index"] == 1
    assert metadata["dominant_range_index"] == 2
    assert metadata["dominant_range_m"] == 2.0
    assert metadata["dominant_range_power"] == 8.0


def test_select_range_bin_power_floor_falls_back_to_target_bin():
    ranges_m = np.array([0.0, 1.0, 2.0, 3.0])
    profile = np.array([0.0, 0.1, 0.2, 0.0])

    selected_index, metadata = select_range_bin(
        profile,
        ranges_m,
        1.1,
        mode="dominant_near_target",
        search_half_width_m=1.0,
        power_floor=1.0,
    )

    assert selected_index == 1
    assert metadata["dominant_range_index"] == 1
    assert metadata["range_profile_is_zero"] is True


def test_select_range_bin_rejects_invalid_inputs():
    ranges_m = np.array([0.0, 1.0])
    profile = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="same shape"):
        select_range_bin(np.array([1.0]), ranges_m, 0.0)
    with pytest.raises(ValueError, match="nonnegative"):
        select_range_bin(profile, ranges_m, 0.0, search_half_width_m=-1.0)
    with pytest.raises(ValueError, match="mode"):
        select_range_bin(profile, ranges_m, 0.0, mode="peak")
    with pytest.raises(ValueError, match="finite"):
        select_range_bin(np.array([1.0, np.nan]), ranges_m, 0.0)


def test_direction_cosine_valid_mask_handles_rectangular_uv_axes():
    u = np.array([-1.0, 0.0, 0.9, 1.2])
    v = np.array([-0.6, 0.0, 0.6])

    mask = direction_cosine_valid_mask(u, v)

    assert mask.shape == (3, 4)
    assert mask[1, 1]
    assert mask[1, 0]
    assert not mask[1, 3]
    assert not mask[0, 2]


def test_direction_cosine_valid_mask_handles_1d_axes():
    mask = direction_cosine_valid_mask(np.array([-1.1, -1.0, 0.0, 1.0, 1.1]))

    np.testing.assert_array_equal(mask, [False, True, True, True, False])


def test_direction_cosine_valid_mask_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="nonempty"):
        direction_cosine_valid_mask([])
    with pytest.raises(ValueError, match="finite"):
        direction_cosine_valid_mask([0.0, np.inf])
    with pytest.raises(ValueError, match="nonnegative"):
        direction_cosine_valid_mask([0.0], eps=-1.0)
