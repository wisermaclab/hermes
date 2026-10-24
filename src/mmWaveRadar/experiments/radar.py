# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Shared radar settings and construction for static and moving experiments."""

from dataclasses import dataclass
import numpy as np

from ..radar import FMCWConfig, RadarHardware, RadarSensor, get_ti_board_spec


@dataclass(frozen=True, kw_only=True)
class RadarSettings:
    board_model: str = "IWR6843AOPEVM"
    orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    antenna_pattern_mode: str = "cosine"
    cosine_3db_beamwidth_deg: float = 60.0
    carrier_frequency_hz: float | None = 60000000000.0
    slope_hz_per_s: float = 68000000000000.0
    chirp_duration_s: float = 5.8e-05
    chirp_repetition_time_s: float = 6.5e-05
    sampling_frequency_hz: float = 4500000.0
    num_adc_samples: int = 225
    num_chirps_per_frame: int | None = 16
    frame_period_s: float = 0.05
    tx_schedule: str = "simultaneous"
    adc_mode: str = "complex1x"
    if_bandwidth_hz: float | None = None
    selected_tx_indices: tuple[int, ...] | None = None
    selected_rx_indices: tuple[int, ...] | None = None


def build_radar(settings: RadarSettings, *, name: str = "radar") -> RadarSensor:
    board = get_ti_board_spec(settings.board_model)
    orientation = np.asarray(settings.orientation_deg, dtype=float)
    if orientation.shape != (3,) or not np.all(np.isfinite(orientation)):
        raise ValueError("radar orientation angles must be three finite values")
    pattern = str(settings.antenna_pattern_mode).casefold().replace("-", "_")
    if pattern not in {"digitized", "none", "cosine"}:
        raise ValueError("antenna_pattern_mode must be 'digitized', 'none', or 'cosine'")
    beamwidth = float(settings.cosine_3db_beamwidth_deg)
    if not np.isfinite(beamwidth) or not 0 < beamwidth < 180:
        raise ValueError("cosine_3db_beamwidth_deg must be between 0 and 180 degrees")
    hardware = RadarHardware.from_ti_board(
        settings.board_model, pattern_mode=pattern, cosine_half_power_angle_deg=0.5 * beamwidth
    )
    tx = (
        tuple(range(hardware.num_tx))
        if settings.selected_tx_indices is None
        else settings.selected_tx_indices
    )
    rx = (
        tuple(range(hardware.num_rx))
        if settings.selected_rx_indices is None
        else settings.selected_rx_indices
    )
    hardware = hardware.subset_tx(tx).subset_rx(rx)
    fmcw = FMCWConfig(
        carrier_frequency=(
            board.design_frequency_hz
            if settings.carrier_frequency_hz is None
            else settings.carrier_frequency_hz
        ),
        slope=settings.slope_hz_per_s,
        chirp_duration=settings.chirp_duration_s,
        chirp_repetition_time=settings.chirp_repetition_time_s,
        sampling_frequency=settings.sampling_frequency_hz,
        adc_mode=settings.adc_mode,
        if_bandwidth_hz=settings.if_bandwidth_hz,
        num_adc_samples=settings.num_adc_samples,
        num_chirps_per_frame=settings.num_chirps_per_frame,
        frame_period=settings.frame_period_s,
        num_tx=hardware.num_tx,
        tx_schedule=settings.tx_schedule,
    )
    return RadarSensor(
        name=name,
        hardware=hardware,
        fmcw=fmcw,
        position=(0, 0, 0),
        orientation=tuple(np.deg2rad(orientation)),
        tx_power_dbm=board.tx_power_dbm,
    )
