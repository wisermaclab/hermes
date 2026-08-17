# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Regression tests for simultaneous and TDM physical acquisition timing."""

from __future__ import annotations

import numpy as np
import pytest

from mmWaveRadar.dsp import range_doppler_map
from mmWaveRadar.radar import FMCWConfig, RadarHardware, RadarSensor
from mmWaveRadar.simulation import MmWaveRadarSimulator, RadarCube, SensingMetadata


def _radar(*, tdm_enabled: bool) -> RadarSensor:
    hardware = RadarHardware.from_positions(
        [[0.0, 0.0, 0.0], [0.0, 0.0025, 0.0]],
        [[0.0, 0.0, 0.0]],
    )
    fmcw = FMCWConfig(
        carrier_frequency=60.0e9,
        slope=1.0e12,
        chirp_duration=0.5e-3,
        chirp_repetition_time=1.0e-3,
        sampling_frequency=8.0e3,
        num_adc_samples=8,
        num_chirps_per_frame=16,
        frame_period=0.1,
        num_tx=2,
        tdm_enabled=tdm_enabled,
    )
    return RadarSensor(
        name="tdm-test",
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        hardware=hardware,
        fmcw=fmcw,
    )


def _tone_cube(radar: RadarSensor, *, num_frames: int, doppler_hz: float):
    fmcw = radar.fmcw
    adc = np.empty(
        (
            num_frames,
            fmcw.num_chirps_per_frame,
            fmcw.num_adc_samples,
            radar.hardware.num_virtual_channels,
        ),
        dtype=np.complex128,
    )
    times = np.empty(adc.shape[:2], dtype=float)
    for frame in range(num_frames):
        for chirp in range(fmcw.num_chirps_per_frame):
            time_s = fmcw.chirp_time(frame, chirp)
            times[frame, chirp] = time_s
            adc[frame, chirp] = np.exp(2j * np.pi * doppler_hz * time_s)
    metadata = SensingMetadata(
        events=[],
        path_counts=np.ones(adc.shape[:2], dtype=np.int64),
        max_displacements=np.zeros(adc.shape[:2], dtype=float),
        virtual_channel_order=radar.hardware.virtual_channel_order,
    )
    return RadarCube(adc=adc, times=times, metadata=metadata)


def test_tdm_simulator_samples_each_tx_in_its_physical_slot(monkeypatch):
    radar = _radar(tdm_enabled=True)
    simulator = MmWaveRadarSimulator(mobility_mode="rt_retrace")
    doppler_bin = 2
    doppler_hz = doppler_bin / (
        radar.fmcw.num_chirps_per_frame * radar.fmcw.slow_time_interval
    )
    captured = {}

    def fake_rt_retrace(_scene, expanded_radar, _targets, *, num_frames):
        captured["fmcw"] = expanded_radar.fmcw
        return _tone_cube(
            expanded_radar,
            num_frames=num_frames,
            doppler_hz=doppler_hz,
        )

    monkeypatch.setattr(simulator, "_run_rt_retrace", fake_rt_retrace)

    cube = simulator.run(None, radar, [], num_frames=1)

    assert captured["fmcw"].tdm_enabled is False
    assert captured["fmcw"].num_chirps_per_frame == 32
    assert cube.adc.shape == (1, 16, 8, 2)
    np.testing.assert_allclose(
        cube.times[0],
        np.arange(16) * radar.fmcw.slow_time_interval,
    )
    expected_channel_times = np.stack(
        [
            np.arange(16) * radar.fmcw.slow_time_interval,
            np.arange(16) * radar.fmcw.slow_time_interval
            + radar.fmcw.chirp_repetition_time,
        ],
        axis=-1,
    )
    np.testing.assert_allclose(
        cube.metadata.virtual_channel_times_s[0],
        expected_channel_times,
    )
    np.testing.assert_allclose(
        np.angle(cube.adc[0, :, 0, 0] * cube.adc[0, :, 0, 1].conj()),
        -2.0 * np.pi * doppler_hz * radar.fmcw.chirp_repetition_time,
    )

    rd_cube, _, velocities_mps = range_doppler_map(
        cube.adc[0],
        fmcw=radar.fmcw,
        win_range=None,
        win_doppler=None,
    )
    peak = int(np.argmax(np.sum(np.abs(rd_cube) ** 2, axis=(1, 2))))
    expected_velocity = 0.5 * radar.fmcw.wavelength * doppler_hz
    assert velocities_mps[peak] == pytest.approx(expected_velocity)


def test_non_tdm_simulator_keeps_all_channels_on_one_chirp_timeline(monkeypatch):
    radar = _radar(tdm_enabled=False)
    simulator = MmWaveRadarSimulator(mobility_mode="rt_retrace")
    captured = {}

    def fake_rt_retrace(_scene, simultaneous_radar, _targets, *, num_frames):
        captured["fmcw"] = simultaneous_radar.fmcw
        return _tone_cube(simultaneous_radar, num_frames=num_frames, doppler_hz=0.0)

    monkeypatch.setattr(simulator, "_run_rt_retrace", fake_rt_retrace)

    cube = simulator.run(None, radar, [], num_frames=1)

    assert captured["fmcw"] is radar.fmcw
    assert cube.adc.shape == (1, 16, 8, 2)
    assert cube.metadata.virtual_channel_times_s is None
    np.testing.assert_allclose(
        cube.times[0],
        np.arange(16) * radar.fmcw.chirp_repetition_time,
    )
