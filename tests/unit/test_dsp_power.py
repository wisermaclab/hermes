# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Tests for public DSP power conversion helpers."""

import numpy as np

from mmWaveRadar.dsp import power_to_db, power_to_relative_db


def test_power_to_db_uses_absolute_simulator_power():
    converted = power_to_db(np.asarray([1.0, 1.0e-3, 0.0]))

    assert np.allclose(converted[:2], np.asarray([0.0, -30.0]))
    assert converted[2] == -300.0


def test_power_to_relative_db_uses_shared_reference_and_floor():
    converted = power_to_relative_db(
        np.asarray([1.0, 1.0e-12, 0.0]),
        reference_power=np.asarray([10.0]),
        floor_db=-60.0,
    )

    assert np.allclose(converted, np.asarray([-10.0, -60.0, -60.0]))
