# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Unit tests for mmWave DSP helpers."""

import numpy as np
import pytest

from mmWaveRadar.dsp import (
    angle_map_fft,
    analytical_array_psf,
    backprojection,
    ca_cfar_1d,
    doppler_time_spectrum,
    framewise_range_doppler_map,
    music_pseudospectrum,
    mvdr_pseudospectrum,
    phase2distance,
    point_cloud_3d_from_adc,
    psf_cross_correlation_image,
    range_doppler_map,
    range_fft,
    range_time_map,
    steering_matrix,
    wiener_psf_image,
)
from mmWaveRadar.radar import FMCWConfig


def test_range_fft_peak_matches_analytic_range():
    num_adc = 64
    sampling_frequency = 1.0e6
    slope = 2.0e12
    target_range = 3.0
    tau = 2.0 * target_range / 299792458.0
    t = np.arange(num_adc) / sampling_frequency
    adc = np.exp(1j * 2.0 * np.pi * slope * tau * t)[None, :, None]

    rt, ranges_m = range_fft(adc, sampling_frequency, slope,
                             window=None, nfft_mult=8)

    peak = int(np.argmax(np.abs(rt[0, :, 0])))
    assert abs(ranges_m[peak] - target_range) < 0.15


def test_range_fft_negative_slope_uses_negative_frequency_bins():
    num_adc = 64
    sampling_frequency = 1.0e6
    slope = -2.0e12
    target_range = 3.0
    tau = 2.0 * target_range / 299792458.0
    t = np.arange(num_adc) / sampling_frequency
    adc = np.exp(1j * 2.0 * np.pi * slope * tau * t)[None, :]

    rt, ranges_m = range_fft(
        adc,
        sampling_frequency,
        slope,
        window=None,
        nfft_mult=8,
    )

    peak = int(np.argmax(np.abs(rt[0])))
    assert np.all(ranges_m >= 0.0)
    assert np.all(np.diff(ranges_m) > 0.0)
    assert abs(ranges_m[peak] - target_range) < 0.15
    assert np.abs(rt[0, peak]) > 0.95 * num_adc


@pytest.mark.parametrize(
    "sampling_frequency,slope,message",
    [
        (0.0, 1.0e12, "sampling_frequency"),
        (-1.0, 1.0e12, "sampling_frequency"),
        (np.nan, 1.0e12, "sampling_frequency"),
        (np.inf, 1.0e12, "sampling_frequency"),
        (1.0e6, 0.0, "slope"),
        (1.0e6, np.nan, "slope"),
        (1.0e6, np.inf, "slope"),
    ],
)
def test_range_fft_validates_physical_parameters(
    sampling_frequency,
    slope,
    message,
):
    with pytest.raises(ValueError, match=message):
        range_fft(
            np.ones((1, 8)),
            sampling_frequency=sampling_frequency,
            slope=slope,
        )


@pytest.mark.parametrize("nfft_mult", [True, 1.5, 0])
def test_range_fft_rejects_invalid_zero_padding_multiplier(nfft_mult):
    with pytest.raises(ValueError, match="nfft_mult"):
        range_fft(
            np.ones((1, 8)),
            sampling_frequency=1.0e6,
            slope=1.0e12,
            nfft_mult=nfft_mult,
        )


def test_phase2distance_uses_tx_conj_rx_phase_sign():
    wavelength = 0.004
    theta = 4.0 * np.pi * 0.01 / wavelength

    assert np.isclose(phase2distance(theta, wavelength), 0.01)


def test_angle_fft_2d_uses_asymmetric_wavelength_normalized_spacing():
    count_y = 16
    count_z = 8
    spacing_y = 0.25
    spacing_z = 0.75
    target_u = 0.5
    target_v = 0.25
    y_lambda = np.arange(count_y) * spacing_y
    z_lambda = np.arange(count_z) * spacing_z
    snapshot = np.exp(
        1j
        * 2.0
        * np.pi
        * (
            z_lambda[:, None] * target_v
            + y_lambda[None, :] * target_u
        )
    )

    angle = angle_map_fft(
        snapshot,
        fc_hz=60.0e9,
        window=None,
        fft_size=(256, 256),
        spacing_y_lambda=spacing_y,
        spacing_z_lambda=spacing_z,
    )

    v_index, u_index = np.unravel_index(
        int(np.argmax(angle["map"])),
        angle["map"].shape,
    )
    assert angle["u"][u_index] == pytest.approx(target_u)
    assert angle["v"][v_index] == pytest.approx(target_v)


@pytest.mark.parametrize(
    ("shape", "spacing_name", "direction_key"),
    [
        ((1, 16), "spacing_y_lambda", "u"),
        ((16, 1), "spacing_z_lambda", "v"),
    ],
)
def test_angle_fft_1d_uses_configured_axis_spacing(
    shape,
    spacing_name,
    direction_key,
):
    spacing_lambda = 0.25
    direction_cosine = 0.5
    phase = np.exp(
        1j
        * 2.0
        * np.pi
        * np.arange(16)
        * spacing_lambda
        * direction_cosine
    )
    snapshot = phase.reshape(shape)

    angle = angle_map_fft(
        snapshot,
        fc_hz=60.0e9,
        window=None,
        fft_size=256,
        **{spacing_name: spacing_lambda},
    )

    peak = int(np.argmax(angle["map"]))
    assert angle[direction_key][peak] == pytest.approx(direction_cosine)


def test_range_time_map_flattens_frames_and_combines_channels():
    adc = np.ones((2, 3, 8, 2), dtype=np.complex128)

    rt_map, ranges_m = range_time_map(adc, sampling_frequency=1e6,
                                      slope=1e12, window=None)

    assert rt_map.shape == (6, 4)
    assert ranges_m.shape == (4,)
    assert np.all(rt_map[:, 0] > 0.0)


def test_range_time_map_can_remove_slow_time_mean_clutter():
    adc, _, moving_range_bin, _ = _static_plus_moving_adc()

    raw_map, _ = range_time_map(
        adc,
        sampling_frequency=1.0e6,
        slope=1.0e12,
        window=None,
    )
    clean_map, _ = range_time_map(
        adc,
        sampling_frequency=1.0e6,
        slope=1.0e12,
        window=None,
        remove_mean_clutter=True,
    )

    assert np.sum(clean_map) < np.sum(raw_map)
    assert np.all(clean_map[:, moving_range_bin] > 0.0)


def test_doppler_time_spectrum_tracks_slow_time_tone():
    num_chirps = 16
    num_adc = 64
    sampling_frequency = 1.0e6
    slope = 2.0e12
    carrier_frequency = 77.0e9
    pri = 1.0e-3
    target_range = 3.0
    doppler_hz = 125.0
    tau = 2.0 * target_range / 299792458.0
    fast_time = np.arange(num_adc) / sampling_frequency
    slow_phase = np.exp(1j * 2.0 * np.pi * doppler_hz
                        * np.arange(num_chirps) * pri)
    range_phase = np.exp(1j * 2.0 * np.pi * slope * tau * fast_time)
    adc = (slow_phase[:, None] * range_phase[None, :])[:, :, None]

    spectrum, times_s, velocities_mps, selected_ranges_m = (
        doppler_time_spectrum(
            adc,
            sampling_frequency=sampling_frequency,
            slope=slope,
            carrier_frequency=carrier_frequency,
            pri=pri,
            window_chirps=8,
            hop_chirps=4,
            doppler_fft_size=16,
            win_range=None,
            win_doppler=None,
            nfft_mult=8,
            range_limits_m=(2.5, 3.5),
        )
    )

    peak_bin = int(np.argmax(np.sum(spectrum, axis=0)))
    expected_velocity = 0.5 * (299792458.0 / carrier_frequency) * doppler_hz
    assert spectrum.shape == (3, 16)
    assert times_s.shape == (3,)
    assert selected_ranges_m.size > 0
    assert abs(velocities_mps[peak_bin] - expected_velocity) < 1e-9


def test_framewise_range_doppler_map_does_not_cross_frame_boundaries():
    frames = 2
    num_chirps = 8
    num_adc = 32
    adc = np.ones((frames, num_chirps, num_adc, 1), dtype=np.complex128)
    adc[1] *= -1.0

    rd_frames, ranges_m, velocities_mps = framewise_range_doppler_map(
        adc,
        sampling_frequency=1.0e6,
        slope=1.0e12,
        carrier_frequency=60.0e9,
        pri=1.0e-3,
        win_range=None,
        win_doppler=None,
    )

    assert rd_frames.shape == (frames, num_chirps, num_adc // 2, 1)
    assert ranges_m.shape == (num_adc // 2,)
    assert velocities_mps.shape == (num_chirps,)
    per_frame_power = np.sum(np.abs(rd_frames) ** 2, axis=(2, 3))
    peak_bins = np.argmax(per_frame_power, axis=1)
    assert np.all(peak_bins == num_chirps // 2)


def test_range_doppler_uses_tdm_transmitter_count_for_default_pri():
    fmcw = FMCWConfig(
        carrier_frequency=60.0e9,
        slope=1.0e12,
        chirp_duration=1.0e-3,
        chirp_repetition_time=2.0e-3,
        sampling_frequency=1.0e6,
        num_adc_samples=8,
        num_chirps_per_frame=4,
        frame_period=0.1,
        num_tx=3,
        tdm_enabled=True,
    )
    adc = np.ones((4, 8, 1), dtype=np.complex128)

    _, _, inferred_velocity = range_doppler_map(adc, fmcw=fmcw)
    _, _, explicit_velocity = range_doppler_map(
        adc,
        fmcw=fmcw,
        pri=fmcw.num_tx * fmcw.chirp_repetition_time,
    )

    np.testing.assert_allclose(inferred_velocity, explicit_velocity)


def test_range_doppler_uses_one_chirp_repetition_when_tdm_is_off():
    fmcw = FMCWConfig(
        carrier_frequency=60.0e9,
        slope=1.0e12,
        chirp_duration=1.0e-3,
        chirp_repetition_time=2.0e-3,
        sampling_frequency=1.0e6,
        num_adc_samples=8,
        num_chirps_per_frame=4,
        frame_period=0.1,
        num_tx=3,
    )
    adc = np.ones((4, 8, 1), dtype=np.complex128)

    _, _, inferred_velocity = range_doppler_map(adc, fmcw=fmcw)
    _, _, explicit_velocity = range_doppler_map(
        adc,
        fmcw=fmcw,
        pri=fmcw.chirp_repetition_time,
    )

    np.testing.assert_allclose(inferred_velocity, explicit_velocity)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sampling_frequency", 0.0),
        ("slope", 0.0),
        ("carrier_frequency", 0.0),
        ("pri", 0.0),
    ],
)
def test_range_doppler_does_not_replace_invalid_explicit_values(field, value):
    fmcw = FMCWConfig(
        carrier_frequency=60.0e9,
        slope=1.0e12,
        chirp_duration=1.0e-3,
        chirp_repetition_time=2.0e-3,
        sampling_frequency=1.0e6,
        num_adc_samples=8,
        num_chirps_per_frame=4,
        frame_period=0.1,
    )

    with pytest.raises(ValueError, match=field):
        range_doppler_map(
            np.ones((4, 8, 1)),
            fmcw=fmcw,
            **{field: value},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("window_chirps", 2.5), ("hop_chirps", True), ("doppler_fft_size", 3.5)],
)
def test_doppler_time_rejects_nonintegral_window_sizes(field, value):
    window_options = {"window_chirps": 2, "doppler_fft_size": 2}
    window_options[field] = value
    with pytest.raises(ValueError, match=field):
        doppler_time_spectrum(
            np.ones((4, 8, 1)),
            sampling_frequency=1.0e6,
            slope=1.0e12,
            carrier_frequency=60.0e9,
            pri=1.0e-3,
            **window_options,
        )


def _static_plus_moving_adc():
    num_chirps = 16
    num_adc = 64
    static_range_bin = 5
    moving_range_bin = 9
    moving_doppler_offset = 2
    fast_time = np.arange(num_adc, dtype=float)
    slow_time = np.arange(num_chirps, dtype=float)

    static_tone = np.exp(
        1j * 2.0 * np.pi * static_range_bin * fast_time / num_adc
    )
    moving_tone = np.exp(
        1j * 2.0 * np.pi * moving_range_bin * fast_time / num_adc
    )
    moving_slow_phase = np.exp(
        1j * 2.0 * np.pi * moving_doppler_offset * slow_time / num_chirps
    )
    adc = (
        10.0 * static_tone[None, :]
        + moving_slow_phase[:, None] * moving_tone[None, :]
    )[:, :, None]
    return adc, static_range_bin, moving_range_bin, moving_doppler_offset


def test_range_doppler_map_can_remove_slow_time_mean_clutter():
    adc, static_range_bin, moving_range_bin, moving_doppler_offset = (
        _static_plus_moving_adc()
    )

    kwargs = dict(
        sampling_frequency=1.0e6,
        slope=1.0e12,
        carrier_frequency=60.0e9,
        pri=1.0e-3,
        win_range=None,
        win_doppler=None,
    )
    rd_raw, _, _ = range_doppler_map(adc, **kwargs)
    rd_clean, _, _ = range_doppler_map(
        adc,
        remove_mean_clutter=True,
        **kwargs,
    )

    zero_bin = adc.shape[0] // 2
    moving_bin = zero_bin + moving_doppler_offset
    raw_static_power = abs(rd_raw[zero_bin, static_range_bin, 0]) ** 2
    clean_static_power = abs(rd_clean[zero_bin, static_range_bin, 0]) ** 2
    raw_moving_power = abs(rd_raw[moving_bin, moving_range_bin, 0]) ** 2
    clean_moving_power = abs(rd_clean[moving_bin, moving_range_bin, 0]) ** 2

    assert clean_static_power < raw_static_power * 1e-20
    assert np.isclose(clean_moving_power, raw_moving_power)


def test_doppler_time_spectrum_can_remove_window_mean_clutter():
    adc, static_range_bin, moving_range_bin, moving_doppler_offset = (
        _static_plus_moving_adc()
    )
    num_chirps = adc.shape[0]

    kwargs = dict(
        sampling_frequency=1.0e6,
        slope=1.0e12,
        carrier_frequency=60.0e9,
        pri=1.0e-3,
        win_range=None,
        win_doppler=None,
        window_chirps=num_chirps,
        doppler_fft_size=num_chirps,
        range_bins=[static_range_bin, moving_range_bin],
    )
    spectrum_raw, _, _, _ = doppler_time_spectrum(adc, **kwargs)
    spectrum_clean, _, _, _ = doppler_time_spectrum(
        adc,
        remove_mean_clutter=True,
        **kwargs,
    )

    zero_bin = num_chirps // 2
    moving_bin = zero_bin + moving_doppler_offset
    assert spectrum_clean[0, zero_bin] < spectrum_raw[0, zero_bin] * 1e-20
    assert spectrum_clean[0, moving_bin] > 0.99 * spectrum_raw[0, moving_bin]


def test_ca_cfar_detects_isolated_peak():
    x = np.ones(64)
    x[32] = 100.0

    result = ca_cfar_1d(x, num_train=4, num_guard=1, pfa=1e-3)

    assert 32 in result["peaks"]


def test_ca_cfar_magnitude_mode_reports_magnitude_units():
    magnitude = np.full(32, 2.0)
    magnitude[16] = 20.0

    result = ca_cfar_1d(
        magnitude,
        num_train=4,
        num_guard=1,
        scale=4.0,
        mode="magnitude",
        peak_only=False,
    )

    assert result["det_mask"][16]
    assert np.isclose(result["noise_est"][16], 2.0)
    assert np.isclose(result["threshold"][16], 4.0)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"mode": "amplitude"}, "mode"),
        ({"num_train": 1.5}, "num_train"),
        ({"num_guard": -1}, "num_guard"),
        ({"min_sep": 0}, "min_sep"),
        ({"pfa": np.nan}, "pfa"),
        ({"scale": 0.0}, "scale"),
        ({"scale": np.inf}, "scale"),
    ],
)
def test_ca_cfar_validates_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ca_cfar_1d(np.ones(16), **kwargs)


def test_ca_cfar_rejects_nonfinite_input():
    x = np.ones(16)
    x[8] = np.nan

    with pytest.raises(ValueError, match="finite"):
        ca_cfar_1d(x)


def test_point_cloud_3d_from_adc_detects_asymmetric_spacing_ura_target():
    num_adc = 64
    sampling_frequency = 1.0e6
    slope = 2.0e12
    carrier_frequency = 60.0e9
    target_range = 3.0
    azimuth_deg = 18.0
    elevation_deg = 10.0

    horizontal = 0.25 * np.arange(4, dtype=float)
    vertical = 0.75 * np.arange(4, dtype=float)
    positions = np.asarray(
        [[x, y] for y in vertical for x in horizontal],
        dtype=float,
    )

    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    u = np.sin(az) * np.cos(el)
    v = np.sin(el)
    channel_phase = np.exp(1j * 2.0 * np.pi * (positions @ np.array([u, v])))

    tau = 2.0 * target_range / 299792458.0
    fast_time = np.arange(num_adc) / sampling_frequency
    range_phase = np.exp(1j * 2.0 * np.pi * slope * tau * fast_time)
    adc = (range_phase[:, None] * channel_phase[None, :])[None, :, :]

    result = point_cloud_3d_from_adc(
        adc,
        virtual_positions_lambda=positions,
        sampling_frequency=sampling_frequency,
        slope=slope,
        carrier_frequency=carrier_frequency,
        range_window=None,
        range_nfft_mult=8,
        angle_fft_size=(64, 64),
    )

    assert result["points"].shape[0] >= 1
    nearest = int(np.argmin(np.abs(result["range_m"] - target_range)))
    assert abs(result["range_m"][nearest] - target_range) < 0.15
    assert abs(result["azimuth_deg"][nearest] - azimuth_deg) < 4.0
    assert abs(result["elevation_deg"][nearest] - elevation_deg) < 4.0
    assert result["points"].shape[1] == 3
    grid_metadata = next(iter(result["angle_products"].values()))["grid_meta"]
    assert grid_metadata["spacing_y_lambda"] == pytest.approx(0.25)
    assert grid_metadata["spacing_z_lambda"] == pytest.approx(0.75)


def test_point_cloud_accepts_centered_even_half_lambda_ura():
    num_adc = 64
    sampling_frequency = 1.0e6
    slope = 2.0e12
    carrier_frequency = 60.0e9
    target_range = 2.0

    coord = (np.arange(4, dtype=float) - 1.5) * 0.5
    yy, zz = np.meshgrid(coord, coord, indexing="xy")
    positions = np.column_stack([yy.reshape(-1), zz.reshape(-1)])

    tau = 2.0 * target_range / 299792458.0
    fast_time = np.arange(num_adc) / sampling_frequency
    range_phase = np.exp(1j * 2.0 * np.pi * slope * tau * fast_time)
    adc = np.repeat(range_phase[:, None], positions.shape[0], axis=1)[None, :, :]

    result = point_cloud_3d_from_adc(
        adc,
        virtual_positions_lambda=positions,
        sampling_frequency=sampling_frequency,
        slope=slope,
        carrier_frequency=carrier_frequency,
        range_window="hann",
        range_nfft_mult=8,
        range_cfar={"num_train": 2, "num_guard": 1, "pfa": 0.9, "min_sep": 1},
        angle_fft_size=(32, 32),
        angle_cfar={"fallback_to_global_peak": True, "max_peaks": 1},
        range_limits_m=(1.5, 2.5),
    )

    assert result["points"].shape[0] >= 1
    strongest = int(np.argmax(result["power"]))
    assert abs(result["range_m"][strongest] - target_range) < 0.15


def test_point_cloud_angle_cfar_fallback_uses_broad_main_lobe_peak():
    num_adc = 225
    sampling_frequency = 4.5e6
    slope = 68.0e12
    carrier_frequency = 59.0e9
    target_range = 1.0

    horizontal = 0.5 * np.arange(4, dtype=float)
    vertical = 0.5 * np.arange(4, dtype=float)
    positions = np.asarray(
        [[x, y] for y in vertical for x in horizontal],
        dtype=float,
    )

    tau = 2.0 * target_range / 299792458.0
    fast_time = np.arange(num_adc) / sampling_frequency
    range_phase = np.exp(1j * 2.0 * np.pi * slope * tau * fast_time)
    channel_phase = np.ones(positions.shape[0], dtype=np.complex128)
    adc = (range_phase[:, None] * channel_phase[None, :])[None, :, :]

    result = point_cloud_3d_from_adc(
        adc,
        virtual_positions_lambda=positions,
        sampling_frequency=sampling_frequency,
        slope=slope,
        carrier_frequency=carrier_frequency,
        range_window="hann",
        range_nfft_mult=8,
        range_limits_m=(0.45, 1.55),
        angle_window=None,
        angle_fft_size=(64, 128),
        range_cfar={"num_train": 12, "num_guard": 4, "pfa": 1e-2, "min_sep": 2},
        angle_cfar={
            "num_train": (4, 4),
            "num_guard": (8, 8),
            "pfa": 1e-2,
            "min_sep": (3, 3),
            "min_relative_power_db": -6.0,
            "fallback_to_global_peak": True,
        },
    )

    assert result["points"].shape[0] >= 1
    strongest = int(np.argmax(result["power"]))
    assert abs(result["range_m"][strongest] - target_range) < 0.04
    assert abs(result["azimuth_deg"][strongest]) < 2.0
    assert abs(result["elevation_deg"][strongest]) < 2.0
    angle_product = result["angle_products"][int(result["range_bins"][strongest])]
    assert angle_product["angle_cfar"]["fallback_peak_added"]


def test_music_and_mvdr_pseudospectra_peak_on_ula_source():
    positions = 0.5 * np.arange(8, dtype=float)
    true_azimuth_deg = 20.0
    true_u = np.sin(np.deg2rad(true_azimuth_deg))
    steering = np.exp(1j * 2.0 * np.pi * positions * true_u)

    rng = np.random.default_rng(4)
    amplitudes = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=64))
    noise = 0.02 * (
        rng.standard_normal((64, positions.size))
        + 1j * rng.standard_normal((64, positions.size))
    )
    snapshots = amplitudes[:, None] * steering[None, :] + noise
    u_axis = np.linspace(-0.8, 0.8, 401)

    music = music_pseudospectrum(
        snapshots,
        positions,
        num_sources=1,
        u=u_axis,
    )
    mvdr = mvdr_pseudospectrum(
        snapshots,
        positions,
        u=u_axis,
        diagonal_loading=1e-2,
    )

    music_peak_u = music["u"][int(np.argmax(music["map"]))]
    mvdr_peak_u = mvdr["u"][int(np.argmax(mvdr["map"]))]
    assert abs(music_peak_u - true_u) < 0.02
    assert abs(mvdr_peak_u - true_u) < 0.02


def test_mvdr_low_rank_path_matches_dense_inverse():
    positions = 0.5 * np.arange(8, dtype=float)
    rng = np.random.default_rng(12)
    snapshots = (
        rng.standard_normal((3, positions.size))
        + 1j * rng.standard_normal((3, positions.size))
    )
    u_axis = np.linspace(-0.7, 0.7, 31)

    mvdr = mvdr_pseudospectrum(
        snapshots,
        positions,
        u=u_axis,
        diagonal_loading=1e-2,
    )
    assert mvdr["inverse_method"] == "woodbury"

    cov = mvdr["covariance"]
    steering = mvdr["steering"]
    weighted = np.linalg.pinv(cov, hermitian=True) @ steering
    denom = np.real(np.sum(steering.conj() * weighted, axis=0))
    dense = 1.0 / np.maximum(denom, 1e-18)
    np.testing.assert_allclose(mvdr["map"], dense, rtol=1e-9, atol=1e-9)


def test_music_pseudospectrum_2d_ura_peak():
    horizontal = 0.5 * np.arange(4, dtype=float)
    vertical = 0.5 * np.arange(4, dtype=float)
    positions = np.asarray(
        [[x, y] for y in vertical for x in horizontal],
        dtype=float,
    )

    azimuth_deg = 15.0
    elevation_deg = 8.0
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)
    true_u = np.sin(azimuth) * np.cos(elevation)
    true_v = np.sin(elevation)
    direction = np.array([true_u, true_v])
    steering = np.exp(1j * 2.0 * np.pi * (positions @ direction))

    rng = np.random.default_rng(5)
    amplitudes = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=80))
    noise = 0.02 * (
        rng.standard_normal((80, positions.shape[0]))
        + 1j * rng.standard_normal((80, positions.shape[0]))
    )
    snapshots = amplitudes[:, None] * steering[None, :] + noise
    u_axis = np.linspace(-0.6, 0.6, 121)
    v_axis = np.linspace(-0.4, 0.4, 81)

    music = music_pseudospectrum(
        snapshots,
        positions,
        num_sources=1,
        u=u_axis,
        v=v_axis,
    )

    v_idx, u_idx = np.unravel_index(
        int(np.argmax(music["map"])),
        music["map"].shape,
    )
    assert abs(music["u"][u_idx] - true_u) < 0.03
    assert abs(music["v"][v_idx] - true_v) < 0.03


def test_analytical_array_psf_is_centered_and_sum_normalized():
    positions = 0.5 * np.arange(4, dtype=float)
    u_axis = np.linspace(-0.5, 0.5, 33)

    result = analytical_array_psf(positions, u=u_axis)

    psf = result["psf"]
    assert psf.shape == u_axis.shape
    assert int(np.argmax(psf)) == u_axis.size // 2
    assert abs(float(np.sum(psf)) - 1.0) < 1e-12


def test_backprojection_localizes_synthetic_ura_source():
    horizontal = 0.5 * np.arange(4, dtype=float)
    vertical = 0.5 * np.arange(4, dtype=float)
    positions = np.asarray(
        [[x, y] for y in vertical for x in horizontal],
        dtype=float,
    )
    u_axis = np.linspace(-0.75, 0.75, 97)
    v_axis = np.linspace(-0.55, 0.55, 81)
    true_u = 0.25
    true_v = -0.12
    snapshot = steering_matrix(
        positions,
        np.asarray([true_u]),
        np.asarray([true_v]),
        normalize=False,
    )[:, 0, 0]

    result = backprojection(
        snapshot[None, :],
        positions,
        u=u_axis,
        v=v_axis,
    )

    assert abs(result["peak_u"] - true_u) < 0.03
    assert abs(result["peak_v"] - true_v) < 0.03
    assert result["image"].shape == (v_axis.size, u_axis.size)
    assert result["valid_mask"].shape == (v_axis.size, u_axis.size)


def test_wiener_psf_image_sharpens_synthetic_ura_source():
    horizontal = 0.5 * np.arange(4, dtype=float)
    vertical = 0.5 * np.arange(4, dtype=float)
    positions = np.asarray(
        [[x, y] for y in vertical for x in horizontal],
        dtype=float,
    )
    u_axis = np.linspace(-0.75, 0.75, 97)
    v_axis = np.linspace(-0.55, 0.55, 81)
    true_u = 0.25
    true_v = -0.12
    snapshot = steering_matrix(
        positions,
        np.asarray([true_u]),
        np.asarray([true_v]),
        normalize=False,
    )[:, 0, 0]

    result = wiener_psf_image(
        snapshot[None, :],
        positions,
        u=u_axis,
        v=v_axis,
        wiener_balance=1e-3,
    )

    assert abs(result["peak_u"] - true_u) < 0.02
    assert abs(result["peak_v"] - true_v) < 0.02
    image = result["image"]
    dirty = result["dirty_image"]
    restored_area = int(np.count_nonzero(image > 0.5 * np.max(image)))
    dirty_area = int(np.count_nonzero(dirty > 0.5 * np.max(dirty)))
    assert restored_area < dirty_area


def test_psf_cross_correlation_image_localizes_synthetic_ura_source():
    horizontal = 0.5 * np.arange(4, dtype=float)
    vertical = 0.5 * np.arange(4, dtype=float)
    positions = np.asarray(
        [[x, y] for y in vertical for x in horizontal],
        dtype=float,
    )
    u_axis = np.linspace(-0.75, 0.75, 97)
    v_axis = np.linspace(-0.55, 0.55, 81)
    true_u = 0.25
    true_v = -0.12
    snapshot = steering_matrix(
        positions,
        np.asarray([true_u]),
        np.asarray([true_v]),
        normalize=False,
    )[:, 0, 0]

    result = psf_cross_correlation_image(
        snapshot[None, :],
        positions,
        u=u_axis,
        v=v_axis,
    )

    assert abs(result["peak_u"] - true_u) < 0.03
    assert abs(result["peak_v"] - true_v) < 0.03
    assert result["image"].shape == (v_axis.size, u_axis.size)
