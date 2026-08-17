# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Validation tests for public radar configuration types."""

from __future__ import annotations

import numpy as np
import pytest

from mmWaveRadar.radar import FMCWConfig, RadarHardware, RadarSensor


def _fmcw(**overrides) -> FMCWConfig:
    values = {
        "carrier_frequency": 60e9,
        "slope": 2e12,
        "chirp_duration": 58e-6,
        "chirp_repetition_time": 65e-6,
        "sampling_frequency": 4.5e6,
        "num_adc_samples": 225,
        "num_chirps_per_frame": 32,
        "frame_period": 50e-3,
        "num_tx": 1,
    }
    values.update(overrides)
    return FMCWConfig(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("carrier_frequency", np.nan),
        ("slope", np.inf),
        ("chirp_duration", np.nan),
        ("chirp_repetition_time", np.inf),
        ("sampling_frequency", np.nan),
        ("frame_period", np.inf),
        ("num_adc_samples", 1.5),
        ("num_chirps_per_frame", True),
        ("num_tx", 0),
    ],
)
def test_fmcw_rejects_nonfinite_or_noninteger_values(field, value):
    with pytest.raises(ValueError, match=field):
        _fmcw(**{field: value})


@pytest.mark.parametrize(
    ("frame_index", "chirp_index"),
    [(-1, 0), (0, -1), (0.5, 0), (0, 0.5), (0, 32)],
)
def test_fmcw_chirp_time_rejects_invalid_indices(frame_index, chirp_index):
    with pytest.raises(ValueError):
        _fmcw().chirp_time(frame_index, chirp_index)


def test_fmcw_defaults_to_simultaneous_transmit_timing():
    fmcw = _fmcw(num_tx=3)

    assert fmcw.tdm_enabled is False
    assert fmcw.slow_time_interval == pytest.approx(
        fmcw.chirp_repetition_time
    )
    assert fmcw.chirp_time(0, 1) == pytest.approx(
        fmcw.chirp_repetition_time
    )
    assert fmcw.tx_chirp_time(0, 1, 0) == pytest.approx(
        fmcw.tx_chirp_time(0, 1, 2)
    )


def test_fmcw_tdm_uses_per_tx_slots_and_per_channel_pri():
    fmcw = _fmcw(num_tx=3, tdm_enabled=np.bool_(True))
    repetition = fmcw.chirp_repetition_time

    assert fmcw.tdm_enabled is True
    assert fmcw.slow_time_interval == pytest.approx(3 * repetition)
    assert fmcw.tx_chirp_time(0, 0, 0) == pytest.approx(0.0)
    assert fmcw.tx_chirp_time(0, 0, 1) == pytest.approx(repetition)
    assert fmcw.tx_chirp_time(0, 0, 2) == pytest.approx(2 * repetition)
    assert fmcw.tx_chirp_time(0, 1, 1) == pytest.approx(4 * repetition)
    assert fmcw.frame_duration == pytest.approx(
        fmcw.num_chirps_per_frame * 3 * repetition
    )


@pytest.mark.parametrize("value", [1, "true", None])
def test_fmcw_rejects_non_boolean_tdm_flag(value):
    with pytest.raises(ValueError, match="tdm_enabled"):
        _fmcw(tdm_enabled=value)


def test_fmcw_rejects_out_of_range_tx_time_index():
    with pytest.raises(ValueError, match="tx_index"):
        _fmcw(num_tx=2).tx_chirp_time(0, 0, 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tx_positions": np.empty((0, 3))},
        {"tx_positions": [[np.nan, 0.0, 0.0]]},
        {"virtual_channel_tx_indices": [0.0],
         "virtual_channel_rx_indices": [0]},
        {"virtual_channel_phase_signs": [np.nan + 0.0j]},
        {"scalar_antenna_gain_dbi": np.inf},
        {"virtual_channel_labels": [""]},
    ],
)
def test_hardware_rejects_invalid_geometry_or_gain(kwargs):
    values = {
        "name": "test-hardware",
        "tx_positions": [[0.0, 0.0, 0.0]],
        "rx_positions": [[0.0, 0.0, 0.0]],
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        RadarHardware(**values)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"position": (0.0, np.nan, 0.0)},
        {"orientation": (0.0, 0.0)},
        {"tx_power_dbm": np.inf},
    ],
)
def test_sensor_rejects_invalid_pose_or_power(kwargs):
    values = {
        "name": "radar",
        "position": (0.0, 0.0, 0.0),
        "orientation": (0.0, 0.0, 0.0),
        "hardware": RadarHardware.from_positions(
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
        ),
        "fmcw": _fmcw(),
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        RadarSensor(**values)


def test_sensor_rejects_waveform_hardware_transmitter_mismatch():
    hardware = RadarHardware.from_positions(
        [[0.0, 0.0, 0.0], [0.0, 0.001, 0.0]],
        [[0.0, 0.0, 0.0]],
    )

    with pytest.raises(
        ValueError,
        match=r"fmcw\.num_tx \(1\).*transmitter count \(2\)",
    ):
        RadarSensor(
            name="radar",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0),
            hardware=hardware,
            fmcw=_fmcw(num_tx=1),
        )


def test_hardware_preserves_board_specific_virtual_channel_permutation():
    hardware = RadarHardware(
        name="permuted-hardware",
        tx_positions=np.zeros((2, 3), dtype=float),
        rx_positions=np.zeros((2, 3), dtype=float),
        virtual_channel_tx_indices=np.array([1, 0, 1, 0]),
        virtual_channel_rx_indices=np.array([1, 0, 0, 1]),
    )

    assert np.array_equal(hardware.virtual_tx_indices(), [1, 0, 1, 0])
    assert np.array_equal(hardware.virtual_rx_indices(), [1, 0, 0, 1])


def test_hardware_rejects_incomplete_virtual_channel_pair_map():
    with pytest.raises(ValueError, match="complete Cartesian"):
        RadarHardware(
            name="duplicate-pair-hardware",
            tx_positions=np.zeros((2, 3), dtype=float),
            rx_positions=np.zeros((2, 3), dtype=float),
            virtual_channel_tx_indices=np.array([0, 0, 1, 1]),
            virtual_channel_rx_indices=np.array([0, 0, 0, 1]),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rows": 1.5},
        {"rows": True},
        {"rows": 1, "cols": 0},
        {"rows": 1, "wavelength": np.nan},
        {"rows": 1, "spacing_lambda": np.inf},
        {"rows": 1, "spacing_y_lambda": 0.0},
        {"rows": 1, "spacing_z_lambda": np.nan},
        {"rows": 1, "centered": "yes"},
    ],
)
def test_virtual_ura_rejects_invalid_dimensions(kwargs):
    values = {"rows": 1, "wavelength": 0.005}
    values.update(kwargs)

    with pytest.raises(ValueError):
        RadarHardware.from_virtual_ura(**values)


@pytest.mark.parametrize("indices", [[0.0], [0, 0], [2]])
def test_hardware_subset_rejects_invalid_indices(indices):
    hardware = RadarHardware.from_positions(
        [[0.0, 0.0, 0.0], [0.0, 0.001, 0.0]],
        [[0.0, 0.0, 0.0]],
    )

    with pytest.raises(ValueError):
        hardware.subset_tx(indices)


@pytest.mark.parametrize("spacing", [0.0, -0.001, np.nan, np.inf])
def test_xwr_hardware_rejects_invalid_spacing(spacing):
    with pytest.raises(ValueError, match="spacing"):
        RadarHardware.from_xwr68xx(spacing=spacing)


def test_hardware_rejects_nonintegral_pattern_channel_count():
    class Pattern:
        num_virtual_channels = 1.5

    with pytest.raises(ValueError, match="num_virtual_channels"):
        RadarHardware.from_virtual_ura(
            rows=1,
            wavelength=0.005,
            antenna_pattern=Pattern(),
        )
