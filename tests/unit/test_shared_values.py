# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Edge cases shared by waveform, solver and DSP input validation."""

import numpy as np
import pytest

from mmWaveRadar.values import boolean, nonzero_float, positive_integer
from mmWaveRadar.dsp.arrays import fmcw_value, snapshot_matrix


@pytest.mark.parametrize("value", [True, np.bool_(True), 1.5, 0, -1, "2"])
def test_counts_never_coerce_boolean_or_fractional_values(value):
    with pytest.raises(ValueError, match="positive integer"):
        positive_integer("count", value)


@pytest.mark.parametrize("value", [1, "false", [], np.array([True])])
def test_boolean_options_do_not_use_truth_value_coercion(value):
    with pytest.raises(ValueError, match="boolean"):
        boolean("enabled", value)


def test_optional_boolean_and_signed_chirp_slope():
    assert boolean("enabled", None, optional=True) is None
    assert boolean("enabled", np.bool_(True)) is True
    assert nonzero_float("slope", -1e12) == -1e12
    with pytest.raises(ValueError, match="non-zero"):
        nonzero_float("slope", 0)


def test_snapshot_shape_policies_preserve_imaging_and_covariance_requirements():
    with pytest.raises(ValueError, match="two virtual channels"):
        snapshot_matrix(np.ones((3, 1)))
    assert snapshot_matrix(np.ones((3, 1)), 1).shape == (3, 1)
    with pytest.raises(ValueError, match="at least one snapshot"):
        snapshot_matrix(np.ones((0, 2)))
    assert snapshot_matrix(np.ones((0, 2)), 2).shape == (0, 2)
    with pytest.raises(ValueError, match="match virtual positions"):
        snapshot_matrix(np.ones((3, 2)), 3)


def test_waveform_lookup_preserves_explicit_fallback_policy():
    assert fmcw_value(None, "slope") is None
    assert fmcw_value(None, "slope", 3) == 3
    assert fmcw_value(object(), "slope", 3) == 3
    with pytest.raises(AttributeError):
        fmcw_value(object(), "slope")
