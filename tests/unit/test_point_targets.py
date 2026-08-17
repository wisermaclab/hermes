# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Tests for generic FMCW point-target helpers."""

import numpy as np
import pytest

from mmWaveRadar.dsp import range_fft
from mmWaveRadar.radar import RadarHardware
from mmWaveRadar.simulation.point_targets import (
    PointTarget,
    azimuth_elevation_axes,
    azimuth_elevation_grids,
    calibrate_range_cube_channel_phase,
    db_relative,
    make_point_target_fmcw_config,
    range_resolved_angle_fft,
    synthesize_virtual_center_point_targets,
    target_direction_cosines,
    virtual_snapshot_grid,
)


def test_target_direction_cosines_broadside_and_axes():
    np.testing.assert_allclose(target_direction_cosines(0.0, 0.0), [0.0, 0.0])
    np.testing.assert_allclose(
        target_direction_cosines(30.0, 0.0),
        [0.5, 0.0],
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("range_m", np.nan),
        ("azimuth_deg", np.inf),
        ("elevation_deg", -np.inf),
        ("reflectivity", np.nan + 0.0j),
    ],
)
def test_point_target_rejects_nonfinite_parameters(field, value):
    values = {
        "range_m": 1.0,
        "azimuth_deg": 0.0,
        "elevation_deg": 0.0,
        "reflectivity": 1.0 + 0.0j,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        PointTarget(**values)


@pytest.mark.parametrize("field", ["num_adc_samples", "num_tx"])
def test_point_target_fmcw_helper_does_not_truncate_counts(field):
    with pytest.raises(ValueError, match=field):
        make_point_target_fmcw_config(
            carrier_frequency=60.0e9,
            **{field: 1.5},
        )
    np.testing.assert_allclose(
        target_direction_cosines(0.0, 30.0),
        [0.0, 0.5],
        atol=1e-12,
    )


def test_angle_axes_and_relative_db_helpers():
    angle = {
        "u": np.array([-0.5, 0.0, 0.5]),
        "v": np.array([0.0, 0.5]),
    }

    azimuth_deg, elevation_deg = azimuth_elevation_axes(angle)
    image_db = db_relative(np.array([1.0, 10.0]))

    np.testing.assert_allclose(azimuth_deg, [-30.0, 0.0, 30.0])
    np.testing.assert_allclose(elevation_deg, [0.0, 30.0])
    np.testing.assert_allclose(image_db, [-10.0, 0.0])


def test_coupled_angle_grids_recover_off_plane_azimuth():
    azimuth_deg = 30.0
    elevation_deg = 60.0
    elevation_rad = np.deg2rad(elevation_deg)
    angle = {
        "u": np.array([
            np.sin(np.deg2rad(azimuth_deg)) * np.cos(elevation_rad)
        ]),
        "v": np.array([np.sin(elevation_rad)]),
    }

    azimuth_grid, elevation_grid = azimuth_elevation_grids(angle)

    assert azimuth_grid[0, 0] == pytest.approx(azimuth_deg)
    assert elevation_grid[0, 0] == pytest.approx(elevation_deg)


def test_point_target_synthesis_range_peak_matches_target_range():
    hardware = RadarHardware.from_ti_board("IWR6843ISK", pattern_mode="none")
    fmcw = make_point_target_fmcw_config(
        carrier_frequency=60.0e9,
        num_tx=hardware.num_tx,
    )
    target_range_m = 1.6

    result = synthesize_virtual_center_point_targets(
        hardware,
        fmcw,
        [PointTarget(target_range_m, 0.0, 0.0)],
        include_channel_phase=False,
    )

    assert result.adc.shape == (
        fmcw.num_adc_samples,
        hardware.num_virtual_channels,
    )
    rt, ranges_m = range_fft(
        result.adc[None, :, :],
        fmcw=fmcw,
        window=None,
        nfft_mult=8,
    )
    peak = int(np.argmax(np.abs(rt[0, :, 0])))
    assert abs(ranges_m[peak] - target_range_m) < 0.06


def test_point_target_synthesis_derives_virtual_positions_from_geometry():
    hardware = RadarHardware.from_xwr68xx("IWR6843AOP")
    fmcw = make_point_target_fmcw_config(
        carrier_frequency=60.0e9,
        num_tx=hardware.num_tx,
    )

    result = synthesize_virtual_center_point_targets(
        hardware,
        fmcw,
        [PointTarget(1.0, 10.0, 5.0)],
        include_channel_phase=False,
    )

    assert result.coefficients.shape == (hardware.num_virtual_channels, 1)
    assert result.adc.shape == (
        fmcw.num_adc_samples,
        hardware.num_virtual_channels,
    )


def test_range_resolved_angle_fft_peaks_at_broadside_for_uniform_target():
    hardware = RadarHardware.from_ti_board("AWRL6844EVM", pattern_mode="none")
    fmcw = make_point_target_fmcw_config(
        carrier_frequency=60.0e9,
        num_tx=hardware.num_tx,
    )
    target_range_m = 1.4
    result = synthesize_virtual_center_point_targets(
        hardware,
        fmcw,
        [{"range_m": target_range_m, "azimuth_deg": 0.0,
          "elevation_deg": 0.0}],
        include_channel_phase=False,
    )

    angle_result = range_resolved_angle_fft(
        result.adc,
        hardware,
        fmcw,
        range_nfft_mult=8,
        angle_fft_size=(64, 64),
        calibrate_channel_phase=False,
    )

    range_index = int(
        np.argmin(np.abs(angle_result["ranges_m"] - target_range_m))
    )
    power = np.asarray(angle_result["angle"]["map"])[range_index]
    v_index, u_index = np.unravel_index(int(np.argmax(power)), power.shape)
    assert abs(angle_result["angle"]["u"][u_index]) < 0.04
    assert abs(angle_result["angle"]["v"][v_index]) < 0.04


def test_range_resolved_angle_fft_preserves_multiple_chirp_snapshots():
    hardware = RadarHardware.from_virtual_ura(rows=2, wavelength=0.005)
    fmcw = make_point_target_fmcw_config(
        carrier_frequency=60.0e9,
        num_adc_samples=16,
    )
    adc = np.ones(
        (2, fmcw.num_adc_samples, hardware.num_virtual_channels),
        dtype=np.complex128,
    )
    adc[1, :, 1::2] *= -1.0

    result = range_resolved_angle_fft(
        adc,
        hardware,
        fmcw,
        range_nfft_mult=1,
        angle_fft_size=(4, 4),
    )

    assert result["range_cube"].shape[:2] == (2, fmcw.num_adc_samples // 2)
    assert result["angle"]["map"].shape[:2] == (
        2,
        fmcw.num_adc_samples // 2,
    )
    assert not np.array_equal(
        result["angle"]["map"][0],
        result["angle"]["map"][1],
    )


def test_range_resolved_angle_fft_uses_virtual_ura_y_z_spacing():
    hardware = RadarHardware.from_virtual_ura(
        rows=8,
        cols=16,
        wavelength=0.005,
        spacing_y_lambda=0.25,
        spacing_z_lambda=0.75,
    )
    fmcw = make_point_target_fmcw_config(
        carrier_frequency=60.0e9,
        num_tx=hardware.num_tx,
    )
    target_u = 0.5
    target_v = 0.25
    elevation_deg = np.rad2deg(np.arcsin(target_v))
    azimuth_deg = np.rad2deg(
        np.arcsin(target_u / np.cos(np.deg2rad(elevation_deg)))
    )
    synthesis = synthesize_virtual_center_point_targets(
        hardware,
        fmcw,
        [PointTarget(1.4, azimuth_deg, elevation_deg)],
        include_channel_phase=False,
    )

    result = range_resolved_angle_fft(
        synthesis.adc,
        hardware,
        fmcw,
        angle_window=None,
        angle_fft_size=(256, 256),
        calibrate_channel_phase=False,
    )

    range_index = int(np.argmin(np.abs(result["ranges_m"] - 1.4)))
    power = np.asarray(result["angle"]["map"])[range_index]
    v_index, u_index = np.unravel_index(int(np.argmax(power)), power.shape)
    assert result["angle"]["u"][u_index] == pytest.approx(target_u)
    assert result["angle"]["v"][v_index] == pytest.approx(target_v)
    assert result["grid_metadata"]["spacing_y_lambda"] == pytest.approx(0.25)
    assert result["grid_metadata"]["spacing_z_lambda"] == pytest.approx(0.75)


def test_virtual_snapshot_grid_maps_half_lambda_virtual_positions():
    hardware = RadarHardware.from_ti_board("AWRL6844EVM", pattern_mode="none")
    range_cube = np.arange(
        hardware.num_virtual_channels,
        dtype=np.complex128,
    )[None, :]

    grid = virtual_snapshot_grid(range_cube, hardware)

    assert grid.shape == (1, 4, 4)
    grid_index = np.rint(
        hardware.virtual_channel_positions_lambda * 2
    ).astype(int)
    for channel, (ix, iy) in enumerate(grid_index):
        assert grid[0, iy, ix] == range_cube[0, channel]


def test_virtual_snapshot_grid_can_return_coordinate_metadata():
    hardware = RadarHardware.from_ti_board("AWRL6844EVM", pattern_mode="none")
    range_cube = np.arange(
        hardware.num_virtual_channels,
        dtype=np.complex128,
    )[None, :]

    grid, metadata = virtual_snapshot_grid(
        range_cube,
        hardware,
        return_metadata=True,
    )

    assert grid.shape == (1, 4, 4)
    assert metadata["occupied"].shape == (4, 4)
    assert np.count_nonzero(metadata["occupied"]) == hardware.num_virtual_channels
    np.testing.assert_allclose(metadata["y_lambda"], [0.0, 0.5, 1.0, 1.5])
    np.testing.assert_allclose(metadata["z_lambda"], [0.0, 0.5, 1.0, 1.5])
    assert metadata["spacing_y_lambda"] == pytest.approx(0.5)
    assert metadata["spacing_z_lambda"] == pytest.approx(0.5)


def test_virtual_snapshot_grid_can_align_centered_even_array_origin():
    coord = (np.arange(4, dtype=float) - 1.5) * 0.5
    yy, zz = np.meshgrid(coord, coord, indexing="xy")
    positions_lambda = np.column_stack([yy.reshape(-1), zz.reshape(-1)])
    hardware = RadarHardware(
        name="centered-ura",
        tx_positions=np.zeros((1, 3)),
        rx_positions=np.zeros((positions_lambda.shape[0], 3)),
        virtual_channel_positions_lambda=positions_lambda,
    )
    range_cube = np.arange(hardware.num_virtual_channels)[None, :]

    with pytest.raises(ValueError, match="lambda/2"):
        virtual_snapshot_grid(range_cube, hardware)

    grid, metadata = virtual_snapshot_grid(
        range_cube,
        hardware,
        align_origin=True,
        return_metadata=True,
    )

    assert grid.shape == (1, 4, 4)
    assert np.count_nonzero(metadata["occupied"]) == hardware.num_virtual_channels
    np.testing.assert_allclose(metadata["y_lambda"], coord)
    np.testing.assert_allclose(metadata["z_lambda"], coord)


def test_calibrate_range_cube_channel_phase_divides_fixed_signs():
    hardware = RadarHardware.from_ti_board("AWRL6844EVM", pattern_mode="none")
    signs = hardware.virtual_channel_phase_signs
    range_cube = np.tile(signs, (3, 1))

    calibrated = calibrate_range_cube_channel_phase(range_cube, hardware)

    np.testing.assert_allclose(calibrated, np.ones_like(range_cube))
    assert calibrate_range_cube_channel_phase(
        range_cube,
        hardware,
        enabled=False,
    ) is range_cube


def test_virtual_snapshot_grid_rejects_non_grid_positions():
    hardware = RadarHardware(
        name="off-grid",
        tx_positions=np.zeros((1, 3)),
        rx_positions=np.zeros((2, 3)),
        virtual_channel_positions_lambda=np.array(
            [[0.0, 0.0], [0.3, 0.0]],
            dtype=float,
        ),
    )
    range_cube = np.ones((4, hardware.num_virtual_channels))

    with pytest.raises(ValueError, match="lambda/2"):
        virtual_snapshot_grid(range_cube, hardware)


@pytest.mark.parametrize("scale", [True, 1.5, 0])
def test_virtual_snapshot_grid_rejects_invalid_scale(scale):
    hardware = RadarHardware.from_virtual_ura(rows=2, wavelength=0.005)
    range_cube = np.ones((1, hardware.num_virtual_channels))

    with pytest.raises(ValueError, match="half_lambda_scale"):
        virtual_snapshot_grid(
            range_cube,
            hardware,
            half_lambda_scale=scale,
        )


@pytest.mark.parametrize("power", [np.array([]), np.array([np.nan]), np.array([-1.0])])
def test_db_relative_rejects_invalid_power(power):
    with pytest.raises(ValueError, match="power"):
        db_relative(power)
