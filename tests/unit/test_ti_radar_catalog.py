# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Tests for TI mmWave board digital-twin catalog entries."""

from __future__ import annotations

import os
from os.path import join
from pathlib import Path
import tempfile

import numpy as np
import pytest

from mmWaveRadar.radar import (
    FMCWConfig,
    RadarHardware,
    RadarSensor,
    TIDigitizedPatternUnavailableError,
    available_ti_boards,
    get_ti_board_spec,
    is_ti_board_model,
    load_ti_digitized_pattern,
    ti_digitized_pattern_asset_available,
)
from mmWaveRadar.radar.ti import SimpleCosinePattern, TIDigitizedPattern


def _fmcw(num_tx: int = 1) -> FMCWConfig:
    return FMCWConfig(
        carrier_frequency=60.0e9,
        slope=68.0e12,
        chirp_duration=58.0e-6,
        chirp_repetition_time=65.0e-6,
        sampling_frequency=4.5e6,
        num_adc_samples=225,
        num_chirps_per_frame=96,
        frame_period=50.0e-3,
        num_tx=num_tx,
    )


def _digitized_hardware_or_skip(model: str) -> RadarHardware:
    if not ti_digitized_pattern_asset_available():
        pytest.skip("optional digitized TI pattern asset is unavailable")
    return RadarHardware.from_ti_board(model, pattern_mode="digitized")


def test_ti_catalog_lookup_and_awrl6844_channel_metadata():
    assert "AWRL6844EVM" in available_ti_boards()
    assert is_ti_board_model("xwrl6844")

    spec = get_ti_board_spec("AWRL6844")
    hardware = RadarHardware.from_ti_board("AWRL6844EVM", pattern_mode="none")

    assert spec.num_tx == 4
    assert spec.num_rx == 4
    assert hardware.num_virtual_channels == 16
    assert hardware.virtual_channel_labels[:4] == (
        "TX1 to RX1",
        "TX1 to RX2",
        "TX1 to RX3",
        "TX1 to RX4",
    )
    assert np.allclose(
        hardware.virtual_channel_positions_lambda[4],
        [0.0, 0.0],
    )
    assert np.array_equal(
        hardware.virtual_channel_phase_signs.real,
        [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1],
    )


def test_aw2243_cascade_catalog_geometry_and_profiles():
    assert "MMWCAS_RF_EVM" in available_ti_boards()
    assert is_ti_board_model("AWR2243 cascade")

    spec = get_ti_board_spec("MMWCAS-RF-EVM")
    hardware = RadarHardware.from_ti_board("TIDEP-01012", pattern_mode="none")

    assert spec.num_tx == 12
    assert spec.num_rx == 16
    assert hardware.num_virtual_channels == 192
    assert hardware.antenna_pattern is None
    assert hardware.virtual_channel_labels[:4] == (
        "TX1 to RX1",
        "TX1 to RX2",
        "TX1 to RX3",
        "TX1 to RX4",
    )

    vc_pos = hardware.virtual_channel_positions_lambda
    azimuth_only = vc_pos[np.isclose(vc_pos[:, 1], 0.0), 0]
    assert len(np.unique(azimuth_only)) == 86
    assert np.isclose(azimuth_only.min(), 0.0)
    assert np.isclose(azimuth_only.max(), 42.5)

    metadata = spec.metadata()
    profile = metadata["fmcw_profiles"]["rtpose_high_resolution_mimo"]
    assert profile["tdm_enabled"] is True
    assert profile["num_tx"] == 12
    assert profile["num_adc_samples"] == 256
    assert profile["num_chirps_per_frame"] == 768
    assert np.isclose(profile["frame_period_s"], 0.1)
    assert metadata["fmcw_profiles"]["ti_mimo_srr"]["tdm_enabled"] is True
    assert metadata["fmcw_profiles"]["ti_mimo_mrr"]["tdm_enabled"] is True
    assert "tdm_enabled" not in metadata["fmcw_profiles"]["ti_txbf"]
    assert metadata["fmcw_profiles"]["ti_mimo_srr"][
        "published_frame_period_s"
    ] == pytest.approx(69.0e-3)
    assert metadata["fmcw_profiles"]["ti_mimo_srr"][
        "frame_period_s"
    ] == pytest.approx(69.12e-3)
    assert metadata["fmcw_profiles"]["ti_mimo_mrr"][
        "published_frame_period_s"
    ] == pytest.approx(4.4e-3)
    assert metadata["fmcw_profiles"]["ti_mimo_mrr"][
        "frame_period_s"
    ] == pytest.approx(41.472e-3)


def test_ti_default_simple_cosine_pattern_needs_no_digitized_asset(monkeypatch):
    missing = Path(tempfile.gettempdir()) / "missing-ti-digitized-patterns.npz"
    monkeypatch.setenv("MMWAVE_TI_PATTERN_NPZ", str(missing))

    assert not ti_digitized_pattern_asset_available()
    hardware = RadarHardware.from_ti_board("MMWCAS-RF-EVM")
    pattern = hardware.antenna_pattern

    assert pattern is not None
    assert pattern.num_virtual_channels == 192
    np.testing.assert_allclose(
        pattern.loss_db(azimuth_deg=0.0, elevation_deg=0.0),
        np.zeros(192),
    )
    np.testing.assert_allclose(
        pattern.loss_db(azimuth_deg=30.0, elevation_deg=0.0),
        np.full(192, -10.0 * np.log10(2.0)),
    )
    np.testing.assert_allclose(
        pattern.loss_db(azimuth_deg=0.0, elevation_deg=30.0),
        np.full(192, -10.0 * np.log10(2.0)),
    )

    assert np.all(pattern.loss_db(azimuth_deg=100.0) <= -79.0)
    tx_subset = hardware.subset_tx([0])
    assert tx_subset.antenna_pattern.num_virtual_channels == 16
    pair_subset = tx_subset.subset_rx([4])
    assert pair_subset.num_tx == 1
    assert pair_subset.num_rx == 1
    assert pair_subset.num_virtual_channels == 1
    assert pair_subset.antenna_pattern.num_virtual_channels == 1


def test_ti_cosine_pattern_accepts_custom_half_power_angle():
    hardware = RadarHardware.from_ti_board(
        "IWR6843AOPEVM",
        pattern_mode="cosine",
        cosine_half_power_angle_deg=21.0,
    )
    pattern = hardware.antenna_pattern

    assert isinstance(pattern, SimpleCosinePattern)
    assert pattern.half_power_angle_deg == pytest.approx(21.0)
    np.testing.assert_allclose(
        pattern.loss_db(azimuth_deg=21.0, elevation_deg=0.0),
        np.full(hardware.num_virtual_channels, -10.0 * np.log10(2.0)),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_virtual_channels": True},
        {"num_virtual_channels": 1.5},
        {"num_virtual_channels": 0},
        {"num_virtual_channels": 1, "half_power_angle_deg": 0.0},
        {"num_virtual_channels": 1, "half_power_angle_deg": 90.0},
        {"num_virtual_channels": 1, "floor_loss_db": 1.0},
    ],
)
def test_simple_cosine_pattern_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        SimpleCosinePattern(**kwargs)


def _test_digitized_pattern(**overrides):
    values = {
        "board_key": "TEST",
        "channel_labels": ("channel 1", "channel 2"),
        "band_centers_hz": np.array([60e9, 61e9]),
        "azimuth_angles_deg": np.array([-10.0, 0.0, 10.0]),
        "elevation_angles_deg": np.array([-5.0, 5.0]),
        "azimuth_loss_db": np.zeros((2, 2, 3)),
        "elevation_loss_db": np.zeros((2, 2, 2)),
        "metadata": {},
    }
    values.update(overrides)
    return TIDigitizedPattern(**values)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"band_centers_hz": np.array([61e9, 60e9])},
        {"azimuth_angles_deg": np.array([0.0, 0.0, 10.0])},
        {"azimuth_loss_db": np.zeros((2, 2, 2))},
        {"elevation_loss_db": np.full((2, 2, 2), np.nan)},
        {"channel_labels": ("same", "same")},
    ],
)
def test_digitized_pattern_validates_axes_and_table_shapes(kwargs):
    with pytest.raises(ValueError):
        _test_digitized_pattern(**kwargs)


@pytest.mark.parametrize("indices", [[0.0], [], [-1], [0, 0], [2]])
def test_digitized_pattern_subset_validates_indices(indices):
    with pytest.raises(ValueError, match="indices"):
        _test_digitized_pattern().subset(np.asarray(indices))


def test_aop_phase_signs_and_default_cosine_pattern():
    hardware = RadarHardware.from_ti_board("IWR6843AOP")

    assert hardware.num_tx == 3
    assert hardware.num_rx == 4
    assert np.array_equal(
        hardware.virtual_channel_phase_signs.real,
        [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
    )

    pattern = hardware.antenna_pattern
    assert pattern is not None
    assert pattern.half_power_angle_deg == 30.0
    np.testing.assert_allclose(
        pattern.loss_db(azimuth_deg=30.0),
        np.full(12, -10.0 * np.log10(2.0)),
    )


def test_aop_digitized_pattern_bands():
    pattern = _digitized_hardware_or_skip("IWR6843AOP").antenna_pattern

    boresight = pattern.loss_db(
        azimuth_deg=0.0,
        elevation_deg=0.0,
        frequency_hz=61.0e9,
    )
    assert np.allclose(boresight, 0.0)

    low_band = pattern.loss_db(azimuth_deg=70.0, frequency_hz=61.0e9)
    high_band = pattern.loss_db(azimuth_deg=70.0, frequency_hz=63.0e9)
    assert low_band.shape == (12,)
    assert high_band.shape == (12,)
    assert not np.allclose(low_band, high_band)


def test_digitized_pattern_env_override_loads_authorized_asset(monkeypatch):
    configured_asset = os.environ.get("MMWAVE_TI_PATTERN_NPZ")
    if not configured_asset:
        pytest.skip("no authorized digitized TI pattern asset is configured")
    asset = Path(configured_asset).expanduser()
    if not asset.is_file():
        pytest.skip("configured digitized TI pattern asset is unavailable")

    monkeypatch.setenv("MMWAVE_TI_PATTERN_NPZ", str(asset))

    assert ti_digitized_pattern_asset_available()
    assert load_ti_digitized_pattern("IWR6843ISK").board_key == "IWR6843ISK"


def test_explicit_digitized_mode_reports_missing_asset(monkeypatch, tmp_path):
    missing = tmp_path / "missing-ti-digitized-patterns.npz"
    monkeypatch.setenv("MMWAVE_TI_PATTERN_NPZ", str(missing))

    assert not ti_digitized_pattern_asset_available()
    with pytest.raises(
        TIDigitizedPatternUnavailableError,
        match="MMWAVE_TI_PATTERN_NPZ.*cosine30",
    ):
        RadarHardware.from_ti_board("IWR6843ISK", pattern_mode="digitized")


def test_radar_sensor_from_ti_board_uses_explicit_fmcw_and_board_power():
    radar = RadarSensor.from_ti_board(
        "IWR6843ISK",
        fmcw=_fmcw(num_tx=1),
        pattern_mode="none",
    )

    assert radar.hardware.name == "IWR6843ISK"
    assert radar.hardware.num_virtual_channels == 12
    assert radar.fmcw.num_tx == 3
    assert radar.tx_power_dbm == 12.0
    assert radar.fmcw.num_adc_samples == 225


def test_virtual_ura_hardware_builds_centered_half_lambda_grid():
    wavelength = 0.005

    hardware = RadarHardware.from_virtual_ura(
        rows=16,
        wavelength=wavelength,
    )

    assert hardware.name == "VIRTUAL_URA16X16"
    assert hardware.num_tx == 1
    assert hardware.num_rx == 256
    assert hardware.num_virtual_channels == 256
    coords = (np.arange(16, dtype=float) - 7.5) * 0.5
    np.testing.assert_allclose(
        np.unique(hardware.virtual_channel_positions_lambda[:, 0]),
        coords,
    )
    np.testing.assert_allclose(
        np.unique(hardware.virtual_channel_positions_lambda[:, 1]),
        coords,
    )
    np.testing.assert_allclose(
        hardware.rx_positions[:, 1:],
        hardware.virtual_channel_positions_lambda * wavelength,
    )
    metadata = hardware.board_metadata
    assert metadata["kind"] == "virtual_ura"
    assert metadata["rows"] == 16
    assert metadata["cols"] == 16
    assert metadata["spacing_lambda"] == 0.5
    assert metadata["transmission"] == "concurrent"


def test_virtual_ura_hardware_supports_rectangular_metadata():
    wavelength = 0.004

    hardware = RadarHardware.from_virtual_ura(
        rows=8,
        cols=16,
        wavelength=wavelength,
        name="analysis-ura",
    )

    metadata = hardware.board_metadata
    assert hardware.name == "analysis-ura"
    assert hardware.num_rx == 128
    assert metadata["rows"] == 8
    assert metadata["cols"] == 16
    assert np.isclose(metadata["aperture_m_y"], 15 * 0.5 * wavelength)
    assert np.isclose(metadata["aperture_m_z"], 7 * 0.5 * wavelength)


def test_virtual_ura_hardware_supports_asymmetric_y_z_spacing():
    wavelength = 0.004

    hardware = RadarHardware.from_virtual_ura(
        rows=3,
        cols=4,
        wavelength=wavelength,
        spacing_y_lambda=0.25,
        spacing_z_lambda=0.75,
    )

    positions = hardware.virtual_channel_positions_lambda
    np.testing.assert_allclose(np.unique(positions[:, 0]), [-0.375, -0.125, 0.125, 0.375])
    np.testing.assert_allclose(np.unique(positions[:, 1]), [-0.75, 0.0, 0.75])
    metadata = hardware.board_metadata
    assert metadata["spacing_lambda"] is None
    assert metadata["spacing_y_lambda"] == pytest.approx(0.25)
    assert metadata["spacing_z_lambda"] == pytest.approx(0.75)
    assert metadata["aperture_m_y"] == pytest.approx(0.75 * wavelength)
    assert metadata["aperture_m_z"] == pytest.approx(1.5 * wavelength)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"rows": 0, "wavelength": 0.005}, "rows"),
        ({"rows": 4, "cols": 0, "wavelength": 0.005}, "cols"),
        ({"rows": 4, "wavelength": 0.0}, "wavelength"),
        ({"rows": 4, "wavelength": 0.005, "spacing_lambda": 0.0}, "spacing"),
        ({"rows": 4, "wavelength": 0.005, "spacing_y_lambda": 0.0}, "spacing_y"),
        ({"rows": 4, "wavelength": 0.005, "spacing_z_lambda": np.inf}, "spacing_z"),
    ],
)
def test_virtual_ura_hardware_validates_geometry(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RadarHardware.from_virtual_ura(**kwargs)


def test_radar_sensor_from_virtual_ura_preserves_pose_and_uses_one_tx():
    fmcw = _fmcw(num_tx=4)

    radar = RadarSensor.from_virtual_ura(
        rows=8,
        cols=16,
        fmcw=fmcw,
        name="virtual-analysis",
        position=(1.0, 2.0, 3.0),
        orientation=(0.1, 0.2, 0.3),
        tx_power_dbm=9.5,
    )

    assert radar.name == "virtual-analysis"
    assert radar.position == (1.0, 2.0, 3.0)
    assert radar.orientation == (0.1, 0.2, 0.3)
    assert radar.tx_power_dbm == 9.5
    assert radar.fmcw.num_tx == 1
    assert radar.hardware.name == "VIRTUAL_URA8X16"
    assert radar.hardware.num_tx == 1
    assert radar.hardware.num_rx == 128
    np.testing.assert_allclose(
        radar.hardware.rx_positions[:, 1:],
        radar.hardware.virtual_channel_positions_lambda * fmcw.wavelength,
    )


def test_ti_cli_config_uses_catalog_hardware_and_enabled_tx_mask():
    cfg = """
profileCfg 0 60 7 7 58 0 0 68 1 225 4500 0 0 158
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 4
frameCfg 0 1 32 0 50 1 0
"""
    path = join(tempfile.gettempdir(), "mmwave_test_ti_catalog.cfg")
    with open(path, "w", encoding="utf-8") as f:
        f.write(cfg)

    radar = RadarSensor.from_ti_cli_config(
        path,
        hardware_model="IWR6843AOPEVM",
        pattern_mode="none",
    )

    assert radar.hardware.name == "IWR6843AOPEVM"
    assert radar.hardware.num_tx == 2
    assert radar.hardware.num_rx == 4
    assert radar.hardware.num_virtual_channels == 8
    assert radar.fmcw.num_tx == 2
    assert radar.fmcw.tdm_enabled is True
    assert radar.hardware.virtual_channel_labels[:4] == (
        "TX1 to RX1",
        "TX1 to RX2",
        "TX1 to RX3",
        "TX1 to RX4",
    )
    assert radar.hardware.virtual_channel_labels[4:] == (
        "TX3 to RX1",
        "TX3 to RX2",
        "TX3 to RX3",
        "TX3 to RX4",
    )


def test_ti_cli_multi_tx_mask_is_simultaneous_not_tdm(tmp_path):
    cfg = """
profileCfg 0 60 7 7 58 0 0 68 1 225 4500 0 0 158
chirpCfg 0 0 0 0 0 0 0 3
frameCfg 0 0 32 0 50 1 0
"""
    path = tmp_path / "simultaneous-mimo.cfg"
    path.write_text(cfg, encoding="utf-8")

    radar = RadarSensor.from_ti_cli_config(
        str(path),
        hardware_model="IWR6843AOPEVM",
        pattern_mode="none",
    )

    assert radar.hardware.num_tx == 2
    assert radar.fmcw.num_tx == 2
    assert radar.fmcw.tdm_enabled is False
    assert radar.fmcw.num_chirps_per_frame == 32


def test_ti_cli_preserves_nonascending_tdm_slot_order(tmp_path):
    cfg = """
profileCfg 0 60 7 7 58 0 0 68 1 225 4500 0 0 158
chirpCfg 0 0 0 0 0 0 0 4
chirpCfg 1 1 0 0 0 0 0 1
frameCfg 0 1 32 0 50 1 0
"""
    path = tmp_path / "nonascending-tdm.cfg"
    path.write_text(cfg, encoding="utf-8")

    radar = RadarSensor.from_ti_cli_config(
        str(path),
        hardware_model="IWR6843AOPEVM",
        pattern_mode="none",
    )

    assert radar.fmcw.tdm_enabled is True
    assert radar.fmcw.num_chirps_per_frame == 32
    assert radar.hardware.virtual_channel_labels[:4] == (
        "TX1 to RX1",
        "TX1 to RX2",
        "TX1 to RX3",
        "TX1 to RX4",
    )
    assert radar.hardware.virtual_channel_labels[4:] == (
        "TX3 to RX1",
        "TX3 to RX2",
        "TX3 to RX3",
        "TX3 to RX4",
    )
    np.testing.assert_array_equal(
        radar.hardware.virtual_tx_indices(),
        [1, 1, 1, 1, 0, 0, 0, 0],
    )


@pytest.mark.parametrize(
    "chirp_lines",
    (
        """
chirpCfg 0 1 0 0 0 0 0 1
chirpCfg 2 2 0 0 0 0 0 4
frameCfg 0 2 32 0 50 1 0
""",
        """
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 3
frameCfg 0 1 32 0 50 1 0
""",
    ),
)
def test_ti_cli_rejects_unrepresentable_changing_tx_masks(
    tmp_path,
    chirp_lines,
):
    cfg = (
        "profileCfg 0 60 7 7 58 0 0 68 1 225 4500 0 0 158\n"
        + chirp_lines
    )
    path = tmp_path / "changing-tx-mask.cfg"
    path.write_text(cfg, encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="constant simultaneous mask or a complete repeated single-Tx",
    ):
        RadarSensor.from_ti_cli_config(
            str(path),
            hardware_model="IWR6843AOPEVM",
            pattern_mode="none",
        )


def test_channel_gain_applies_scalar_gain_and_phase_to_coefficients():
    hardware = RadarHardware.from_ti_board("IWR6843AOPEVM", pattern_mode="none")
    coeffs = np.ones((hardware.num_virtual_channels, 2), dtype=np.complex128)

    out = hardware.apply_channel_gain(coeffs)

    expected_amp = 10.0 ** (10.0 / 20.0)
    np.testing.assert_allclose(
        out[:, 0],
        hardware.virtual_channel_phase_signs * expected_amp,
    )
    np.testing.assert_allclose(out[:, 1], out[:, 0])


def test_channel_gain_treats_one_dimensional_angles_as_paths():
    hardware = RadarHardware.from_ti_board("IWR6843ISK")
    angles = np.linspace(-30.0, 30.0, hardware.num_virtual_channels)
    coeffs = np.ones(
        (hardware.num_virtual_channels, angles.size),
        dtype=np.complex128,
    )

    actual = hardware.apply_channel_gain(coeffs, azimuth_deg=angles)
    expected = np.column_stack([
        hardware.channel_gain(azimuth_deg=float(angle))
        for angle in angles
    ])

    assert actual.shape == coeffs.shape
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("vector_axis", ["azimuth", "elevation"])
def test_digitized_pattern_broadcasts_scalar_and_vector_angles(vector_axis):
    pattern = _digitized_hardware_or_skip("IWR6843ISK").antenna_pattern
    vector = np.array([-20.0, 0.0, 25.0])
    kwargs = {"azimuth_deg": 5.0, "elevation_deg": 10.0}
    kwargs[f"{vector_axis}_deg"] = vector

    actual = pattern.loss_db(**kwargs)
    expected = np.column_stack([
        pattern.loss_db(**{
            "azimuth_deg": float(value) if vector_axis == "azimuth" else 5.0,
            "elevation_deg": (
                float(value) if vector_axis == "elevation" else 10.0
            ),
        })
        for value in vector
    ])

    assert actual.shape == (pattern.num_virtual_channels, vector.size)
    np.testing.assert_allclose(actual, expected)


def test_digitized_pattern_accepts_explicit_per_channel_angles():
    pattern = _digitized_hardware_or_skip("IWR6843ISK").antenna_pattern
    angles = np.linspace(-20.0, 20.0, pattern.num_virtual_channels)

    actual = pattern.loss_db(azimuth_deg=angles[:, None])
    expected = np.asarray([
        pattern.loss_db(azimuth_deg=float(angle))[channel]
        for channel, angle in enumerate(angles)
    ])[:, None]

    assert actual.shape == (pattern.num_virtual_channels, 1)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("frequency_hz", [0.0, np.nan, np.inf])
def test_digitized_pattern_rejects_invalid_frequency(frequency_hz):
    pattern = _digitized_hardware_or_skip("IWR6843ISK").antenna_pattern

    with pytest.raises(ValueError, match="frequency_hz"):
        pattern.loss_db(azimuth_deg=0.0, frequency_hz=frequency_hz)


def test_ti_cli_ignores_chirps_outside_selected_frame(tmp_path):
    cfg = """
profileCfg 0 60 7 7 58 0 0 68 1 225 4500 0 0 158
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 10 10 99 1 2 3 4 4
frameCfg 0 0 32 0 50 1 0
"""
    path = tmp_path / "selected-frame.cfg"
    path.write_text(cfg, encoding="utf-8")

    radar = RadarSensor.from_ti_cli_config(
        str(path),
        hardware_model="IWR6843ISK",
        pattern_mode="none",
    )

    assert radar.hardware.num_tx == 1
    assert radar.fmcw.num_tx == 1
    assert radar.fmcw.num_chirps_per_frame == 32
    assert radar.fmcw.tdm_enabled is False


def test_ti_cli_honors_chirp_ranges_and_compatible_profile_ids(tmp_path):
    cfg = """
profileCfg 0 60 7 7 58 0 0 68 1 225 4500 0 0 158
profileCfg 1 60 7 7 58 0 0 68 1 225 4500 0 0 158
chirpCfg 0 1 0 0 0 0 0 1
chirpCfg 2 2 1 0 0 0 0 4
frameCfg 1 2 16 0 50 1 0
"""
    path = tmp_path / "profile-map.cfg"
    path.write_text(cfg, encoding="utf-8")

    radar = RadarSensor.from_ti_cli_config(
        str(path),
        hardware_model="IWR6843ISK",
        pattern_mode="none",
    )

    assert radar.hardware.num_tx == 2
    assert radar.fmcw.num_tx == 2
    assert radar.fmcw.num_chirps_per_frame == 16
    assert radar.fmcw.tdm_enabled is True


def test_ti_cli_rejects_incompatible_in_frame_profiles(tmp_path):
    cfg = """
profileCfg 0 60 7 7 58 0 0 68 1 225 4500 0 0 158
profileCfg 1 61 7 7 58 0 0 68 1 225 4500 0 0 158
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 1 0 0 0 0 2
frameCfg 0 1 16 0 50 1 0
"""
    path = tmp_path / "incompatible-profiles.cfg"
    path.write_text(cfg, encoding="utf-8")

    with pytest.raises(ValueError, match="incompatible profiles"):
        RadarSensor.from_ti_cli_config(str(path), pattern_mode="none")


def test_configure_scene_refreshes_existing_same_name_devices():
    class Scene:
        def __init__(self):
            self.devices = {}

        def get(self, name):
            return self.devices.get(name)

        def add(self, device):
            self.devices[device.name] = device

    initial = RadarSensor.from_virtual_ura(
        rows=1,
        cols=1,
        fmcw=_fmcw(),
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        tx_power_dbm=12.0,
    )
    updated = RadarSensor(
        name=initial.name,
        position=(1.0, 2.0, 3.0),
        orientation=(0.1, 0.2, 0.3),
        hardware=initial.hardware,
        fmcw=initial.fmcw,
        tx_power_dbm=7.5,
    )
    scene = Scene()

    old_tx, old_rx = initial.configure_scene(scene)
    tx, rx = updated.configure_scene(scene)

    assert tx is old_tx
    assert rx is old_rx
    np.testing.assert_allclose(np.asarray(tx.position).reshape(-1), updated.position)
    np.testing.assert_allclose(np.asarray(rx.position).reshape(-1), updated.position)
    np.testing.assert_allclose(
        np.asarray(tx.orientation).reshape(-1),
        updated.orientation,
    )
    np.testing.assert_allclose(
        np.asarray(rx.orientation).reshape(-1),
        updated.orientation,
    )
    assert np.isclose(float(np.asarray(tx.power_dbm).reshape(-1)[0]), 7.5)
