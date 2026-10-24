# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Regression tests for simultaneous and TDM physical acquisition timing."""

from __future__ import annotations

from dataclasses import replace
import json
import warnings
import numpy as np
import pytest

from mmWaveRadar.dsp import range_doppler_map
from mmWaveRadar.radar import FMCWConfig, RadarHardware, RadarSensor
from mmWaveRadar.simulation import MmWaveRadarSimulator, RadarCube, SensingMetadata


@pytest.mark.parametrize(
    "name, expected", [("distance", 3.0), ("distance_min_m", 2.0), ("distance_max_m", 4.0)]
)
def test_unobserved_slot_diagnostics_stay_nan_without_warnings(name, expected):
    from mmWaveRadar.simulation.tdm import _collapse_slot_array

    values = np.asarray([[np.nan, np.nan, 2.0, 4.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        packed = _collapse_slot_array(values, num_chirps=2, num_tx=2, name=name)
    np.testing.assert_allclose(packed, [[np.nan, expected]], equal_nan=True)


@pytest.mark.parametrize(
    "order,stored", [((2, 0, 1), None), ((0, 1, 2), (0, 2)), ((0, 2), (2, 0))]
)
def test_bundle_tdm_sensors_reject_unsupported_core_slot_mappings(order, stored):
    from bundle_prepare import BundleSensorConfig
    from importlib import resources
    from mmWaveRadar.simulation.tdm import expanded_tdm_radar, virtual_channel_times

    source = resources.files("bundle_prepare").joinpath("data", "static_target", "sensor.json")
    config = BundleSensorConfig.from_mapping(json.loads(source.read_text()))
    config = replace(
        config,
        fmcw=replace(config.fmcw, tx_schedule="tdm", num_tx=len(order)),
        tdm_tx_order=order,
        tx_indices_zero_based=stored,
    )
    sensor = config.to_sensor()
    assert sensor.fmcw.slow_time_interval == pytest.approx(len(order) * 5e-5)
    simulator = MmWaveRadarSimulator(solver="po")
    for function in (
        expanded_tdm_radar,
        lambda s: virtual_channel_times(s, 1),
        lambda s: simulator.run(None, s, [], num_frames=1),
    ):
        with pytest.raises(ValueError, match="slot-aware bundle validation runner"):
            function(sensor)


def test_local_tx_order_can_represent_a_physical_subset_schedule():
    from bundle_prepare import BundleSensorConfig
    from importlib import resources
    from mmWaveRadar.simulation.tdm import expanded_tdm_radar, virtual_channel_times

    source = resources.files("bundle_prepare").joinpath("data", "static_target", "sensor.json")
    config = BundleSensorConfig.from_mapping(json.loads(source.read_text()))
    config = replace(
        config,
        fmcw=replace(config.fmcw, tx_schedule="tdm", num_tx=2),
        tdm_tx_order=(2, 0),
        tx_indices_zero_based=(2, 0),
    )
    sensor = config.to_sensor()
    assert expanded_tdm_radar(sensor).fmcw.num_chirps_per_frame == 8
    times = virtual_channel_times(sensor, 1)[0, 0]
    tx = sensor.hardware.virtual_tx_indices()
    np.testing.assert_allclose(times[tx == 0], 0)
    np.testing.assert_allclose(times[tx == 1], 5e-5)


def _radar(*, tx_schedule: str) -> RadarSensor:
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
        tx_schedule=tx_schedule,
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
    radar = _radar(tx_schedule="tdm")
    simulator = MmWaveRadarSimulator(solver="rt", rt_refresh_frames=0)
    doppler_bin = 2
    doppler_hz = doppler_bin / (radar.fmcw.num_chirps_per_frame * radar.fmcw.slow_time_interval)
    captured = {}

    def fake_rt_every_chirp(_scene, expanded_radar, _targets, *, num_frames):
        captured["fmcw"] = expanded_radar.fmcw
        return _tone_cube(
            expanded_radar,
            num_frames=num_frames,
            doppler_hz=doppler_hz,
        )

    monkeypatch.setattr(simulator, "_run_rt_every_chirp", fake_rt_every_chirp)

    cube = simulator.run(None, radar, [], num_frames=1)

    assert captured["fmcw"].tx_schedule == "simultaneous"
    assert captured["fmcw"].num_chirps_per_frame == 32
    assert cube.adc.shape == (1, 16, 8, 2)
    np.testing.assert_allclose(
        cube.times[0],
        np.arange(16) * radar.fmcw.slow_time_interval,
    )
    expected_channel_times = np.stack(
        [
            np.arange(16) * radar.fmcw.slow_time_interval,
            np.arange(16) * radar.fmcw.slow_time_interval + radar.fmcw.chirp_repetition_time,
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
    radar = _radar(tx_schedule="simultaneous")
    simulator = MmWaveRadarSimulator(solver="rt", rt_refresh_frames=0)
    captured = {}

    def fake_rt_every_chirp(_scene, simultaneous_radar, _targets, *, num_frames):
        captured["fmcw"] = simultaneous_radar.fmcw
        return _tone_cube(simultaneous_radar, num_frames=num_frames, doppler_hz=0.0)

    monkeypatch.setattr(simulator, "_run_rt_every_chirp", fake_rt_every_chirp)

    cube = simulator.run(None, radar, [], num_frames=1)

    assert captured["fmcw"] is radar.fmcw
    assert cube.adc.shape == (1, 16, 8, 2)
    assert cube.metadata.virtual_channel_times_s is None
    np.testing.assert_allclose(
        cube.times[0],
        np.arange(16) * radar.fmcw.chirp_repetition_time,
    )
