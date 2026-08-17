# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Radar sensor helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fmcw import FMCWConfig
from .hardware import RadarHardware


def _finite_pose_vector(name: str, value) -> tuple[float, float, float]:
    """Validates and normalizes one three-component radar pose vector."""

    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain three finite values") from exc
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite values")
    return tuple(float(component) for component in vector)


def _positions_to_point3f(positions: np.ndarray, wavelength: float):
    """Converts local element positions in meters to Mitsuba wavelength units."""

    import mitsuba as mi  # pylint: disable=import-outside-toplevel

    pos = np.asarray(positions, dtype=float)
    return mi.Point3f(pos[:, 0] / wavelength,
                      pos[:, 1] / wavelength,
                      pos[:, 2] / wavelength)


def _rotation_matrix_numpy(orientation: tuple[float, float, float]) -> np.ndarray:
    """Returns the world rotation matrix for a radar pose."""

    a, b, c = (float(v) for v in orientation)
    sin_a, cos_a = np.sin(a), np.cos(a)
    sin_b, cos_b = np.sin(b), np.cos(b)
    sin_c, cos_c = np.sin(c), np.cos(c)
    return np.array([
        [cos_a * cos_b, cos_a * sin_b * sin_c - sin_a * cos_c,
         cos_a * sin_b * cos_c + sin_a * sin_c],
        [sin_a * cos_b, sin_a * sin_b * sin_c + cos_a * cos_c,
         sin_a * sin_b * cos_c - cos_a * sin_c],
        [-sin_b, cos_b * sin_c, cos_b * cos_c],
    ], dtype=float)


@dataclass(frozen=True)
class RadarSensor:
    r"""
    Radar pose, hardware, and FMCW configuration.

    The radar is represented in Sionna as one transmitter and one receiver
    sharing the same pose but using hardware-specific Tx/Rx arrays.
    """

    name: str
    position: tuple[float, float, float]
    orientation: tuple[float, float, float]
    hardware: RadarHardware
    fmcw: FMCWConfig
    tx_power_dbm: float = 12.0

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.hardware, RadarHardware):
            raise ValueError("hardware must be a RadarHardware instance")
        if not isinstance(self.fmcw, FMCWConfig):
            raise ValueError("fmcw must be an FMCWConfig instance")

        position = _finite_pose_vector("position", self.position)
        orientation = _finite_pose_vector("orientation", self.orientation)
        try:
            tx_power_dbm = float(self.tx_power_dbm)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("tx_power_dbm must be a finite scalar") from exc
        if not np.isfinite(tx_power_dbm):
            raise ValueError("tx_power_dbm must be a finite scalar")
        if self.fmcw.num_tx != self.hardware.num_tx:
            raise ValueError(
                f"fmcw.num_tx ({self.fmcw.num_tx}) must match the hardware "
                f"transmitter count ({self.hardware.num_tx})"
            )

        object.__setattr__(self, "position", position)
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "tx_power_dbm", tx_power_dbm)

    @classmethod
    def from_ti_board(cls, model: str, *, fmcw: FMCWConfig,
                      name: str = "radar",
                      position=(0.0, 0.0, 0.0),
                      orientation=(0.0, 0.0, 0.0),
                      pattern_mode: str = "cosine30",
                      cosine_half_power_angle_deg: float = 30.0,
                      tx_power_dbm: float | None = None) -> "RadarSensor":
        """Builds a radar sensor from a supported TI board catalog entry."""

        from .ti import get_ti_board_spec  # pylint: disable=import-outside-toplevel

        spec = get_ti_board_spec(model)
        hardware = RadarHardware.from_ti_board(
            spec.key,
            pattern_mode=pattern_mode,
            cosine_half_power_angle_deg=cosine_half_power_angle_deg,
        )
        fmcw = FMCWConfig(
            carrier_frequency=fmcw.carrier_frequency,
            slope=fmcw.slope,
            chirp_duration=fmcw.chirp_duration,
            chirp_repetition_time=fmcw.chirp_repetition_time,
            sampling_frequency=fmcw.sampling_frequency,
            num_adc_samples=fmcw.num_adc_samples,
            num_chirps_per_frame=fmcw.num_chirps_per_frame,
            frame_period=fmcw.frame_period,
            num_tx=hardware.num_tx,
            tdm_enabled=fmcw.tdm_enabled,
        )
        power = spec.tx_power_dbm if tx_power_dbm is None else float(tx_power_dbm)
        return cls(name=name,
                   position=tuple(position),
                   orientation=tuple(orientation),
                   hardware=hardware,
                   fmcw=fmcw,
                   tx_power_dbm=power)

    @classmethod
    def from_virtual_ura(cls, *, rows: int,
                         cols: int | None = None,
                         fmcw: FMCWConfig,
                         name: str = "radar",
                         position=(0.0, 0.0, 0.0),
                         orientation=(0.0, 0.0, 0.0),
                         spacing_lambda: float = 0.5,
                         spacing_y_lambda: float | None = None,
                         spacing_z_lambda: float | None = None,
                         centered: bool = True,
                         tx_power_dbm: float = 12.0,
                         scalar_antenna_gain_dbi: float | None = 0.0,
                         antenna_pattern: object | None = None
                         ) -> "RadarSensor":
        """
        Builds a radar sensor from a synthetic virtual URA.

        The virtual URA is represented as one phase-center Tx and one
        synthetic Rx per virtual channel, so the returned FMCW configuration
        always uses ``num_tx=1``.
        """

        hardware = RadarHardware.from_virtual_ura(
            rows=rows,
            cols=cols,
            wavelength=fmcw.wavelength,
            spacing_lambda=spacing_lambda,
            spacing_y_lambda=spacing_y_lambda,
            spacing_z_lambda=spacing_z_lambda,
            centered=centered,
            scalar_antenna_gain_dbi=scalar_antenna_gain_dbi,
            antenna_pattern=antenna_pattern,
        )
        fmcw = FMCWConfig(
            carrier_frequency=fmcw.carrier_frequency,
            slope=fmcw.slope,
            chirp_duration=fmcw.chirp_duration,
            chirp_repetition_time=fmcw.chirp_repetition_time,
            sampling_frequency=fmcw.sampling_frequency,
            num_adc_samples=fmcw.num_adc_samples,
            num_chirps_per_frame=fmcw.num_chirps_per_frame,
            frame_period=fmcw.frame_period,
            num_tx=1,
            tdm_enabled=False,
        )
        return cls(name=name,
                   position=tuple(position),
                   orientation=tuple(orientation),
                   hardware=hardware,
                   fmcw=fmcw,
                   tx_power_dbm=float(tx_power_dbm))

    @classmethod
    def from_ti_cli_config(cls, filename: str, *,
                           name: str = "radar",
                           position=(0.0, 0.0, 0.0),
                           orientation=(0.0, 0.0, 0.0),
                           hardware_model: str = "IWR6843AOP",
                           default_carrier_frequency: float = 77e9,
                           pattern_mode: str = "cosine30",
                           tx_power_dbm: float = 12.0) -> "RadarSensor":
        """Builds a radar sensor from a TI mmWave CLI config file."""
        fmcw, enabled_tx_indices = _parse_ti_cli_config(
            filename, default_carrier_frequency)
        from .ti import is_ti_board_model  # pylint: disable=import-outside-toplevel

        if is_ti_board_model(hardware_model):
            hardware = RadarHardware.from_ti_board(
                hardware_model,
                pattern_mode=pattern_mode,
            )
        else:
            hardware = RadarHardware.from_xwr68xx(hardware_model)
        if enabled_tx_indices is not None:
            hardware = hardware.subset_tx(enabled_tx_indices)
        fmcw = FMCWConfig(
            carrier_frequency=fmcw.carrier_frequency,
            slope=fmcw.slope,
            chirp_duration=fmcw.chirp_duration,
            chirp_repetition_time=fmcw.chirp_repetition_time,
            sampling_frequency=fmcw.sampling_frequency,
            num_adc_samples=fmcw.num_adc_samples,
            num_chirps_per_frame=fmcw.num_chirps_per_frame,
            frame_period=fmcw.frame_period,
            num_tx=hardware.num_tx,
            tdm_enabled=fmcw.tdm_enabled)
        return cls(name=name,
                   position=tuple(position),
                   orientation=tuple(orientation),
                   hardware=hardware,
                   fmcw=fmcw,
                   tx_power_dbm=tx_power_dbm)

    def configure_scene(self, scene) -> tuple[Transmitter, Receiver]:
        """
        Adds radar Tx/Rx devices to ``scene`` and configures their arrays.

        Existing scene Tx/Rx arrays are replaced because Sionna scenes use one
        shared Tx array and one shared Rx array.
        """
        from sionna.rt import AntennaArray, Receiver, Transmitter
        from sionna.rt.antenna_pattern import antenna_pattern_registry

        wavelength = self.fmcw.wavelength
        pattern = antenna_pattern_registry.get("iso")(polarization="V")
        scene.tx_array = AntennaArray(
            pattern,
            _positions_to_point3f(self.hardware.tx_positions, wavelength))
        scene.rx_array = AntennaArray(
            pattern,
            _positions_to_point3f(self.hardware.rx_positions, wavelength))
        scene.frequency = self.fmcw.carrier_frequency

        tx_name = f"{self.name}-tx"
        rx_name = f"{self.name}-rx"
        tx = scene.get(tx_name)
        rx = scene.get(rx_name)
        if tx is None:
            tx = Transmitter(name=tx_name,
                             position=self.position,
                             orientation=self.orientation,
                             power_dbm=self.tx_power_dbm)
            scene.add(tx)
        else:
            tx.position = self.position
            tx.orientation = self.orientation
            tx.power_dbm = self.tx_power_dbm
        if rx is None:
            rx = Receiver(name=rx_name,
                          position=self.position,
                          orientation=self.orientation)
            scene.add(rx)
        else:
            rx.position = self.position
            rx.orientation = self.orientation
        return tx, rx

    def world_tx_positions(self) -> np.ndarray:
        """Returns Tx element positions in world coordinates [m]."""

        return self._world_positions(self.hardware.tx_positions)

    def world_rx_positions(self) -> np.ndarray:
        """Returns Rx element positions in world coordinates [m]."""

        return self._world_positions(self.hardware.rx_positions)

    def world_to_local_points(self, points: np.ndarray) -> np.ndarray:
        """Transforms world points to the radar local coordinate frame."""

        pts = np.asarray(points, dtype=float)
        flat = pts.reshape((-1, 3))
        rot = _rotation_matrix_numpy(self.orientation)
        local = (rot.T @ (flat - np.asarray(self.position, dtype=float)).T).T
        return local.reshape(pts.shape)

    def local_azimuth_elevation(self, points: np.ndarray
                                ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns local azimuth/elevation angles for world points in degrees.

        The local ``x`` axis is boresight, ``y`` is horizontal/azimuth, and
        ``z`` is vertical/elevation.
        """

        local = self.world_to_local_points(points).reshape((-1, 3))
        azimuth = np.degrees(np.arctan2(local[:, 1], local[:, 0]))
        elevation = np.degrees(
            np.arctan2(local[:, 2], np.hypot(local[:, 0], local[:, 1]))
        )
        return azimuth, elevation

    def _world_positions(self, local_positions: np.ndarray) -> np.ndarray:
        """Transforms local element positions to world coordinates in meters."""

        rot = _rotation_matrix_numpy(self.orientation)
        pos = np.asarray(local_positions, dtype=float)
        world = (rot @ pos.T).T
        return world + np.asarray(self.position, dtype=float)[None, :]


def _parse_ti_cli_config(filename: str,
                         default_carrier_frequency: float
                         ) -> tuple[FMCWConfig, tuple[int, ...] | None]:
    """Parses TI ``profileCfg``, ``chirpCfg``, and ``frameCfg`` commands."""

    profiles: dict[int, dict[str, float | int]] = {}
    chirp_configs: list[dict[str, object]] = []
    frame_config: tuple[int, int, int, float] | None = None

    with open(filename, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("%") or line.startswith("#"):
                continue
            toks = line.split()
            if toks[0] == "profileCfg":
                try:
                    profile_id = int(float(toks[1]))
                    profiles[profile_id] = {
                        "start_freq_ghz": float(toks[2]),
                        "idle_us": float(toks[3]),
                        "ramp_us": float(toks[5]),
                        "slope_mhz_per_us": float(toks[8]),
                        "num_adc": int(float(toks[10])),
                        "fs_ksps": float(toks[11]),
                    }
                except (IndexError, ValueError) as exc:
                    raise ValueError(f"Could not parse profileCfg: {line}") \
                        from exc
            elif toks[0] == "chirpCfg":
                try:
                    chirp_configs.append({
                        "start": int(float(toks[1])),
                        "end": int(float(toks[2])),
                        "profile_id": int(float(toks[3])),
                        "variations": tuple(float(toks[index])
                                            for index in range(4, 8)),
                        "tx_mask": int(float(toks[8])),
                    })
                except (IndexError, ValueError) as exc:
                    raise ValueError(f"Could not parse chirpCfg: {line}") \
                        from exc
            elif toks[0] == "frameCfg":
                try:
                    frame_config = (
                        int(float(toks[1])),
                        int(float(toks[2])),
                        int(float(toks[3])),
                        float(toks[5]),
                    )
                except (IndexError, ValueError) as exc:
                    raise ValueError(f"Could not parse frameCfg: {line}") \
                        from exc

    if not profiles or frame_config is None:
        raise ValueError("TI config must contain profileCfg and frameCfg")

    chirp_start, chirp_end, num_loops, frame_period_ms = frame_config
    if chirp_start < 0 or chirp_end < chirp_start:
        raise ValueError("frameCfg chirp range must be nonnegative and ordered")
    if num_loops <= 0:
        raise ValueError("frameCfg numLoops must be positive")

    num_chirps_per_loop = chirp_end - chirp_start + 1
    schedule: list[dict[str, object]] = []
    if chirp_configs:
        for chirp_index in range(chirp_start, chirp_end + 1):
            matches = [
                config for config in chirp_configs
                if int(config["start"]) <= chirp_index <= int(config["end"])
            ]
            if not matches:
                raise ValueError(
                    f"No chirpCfg covers in-frame chirp index {chirp_index}"
                )
            if len(matches) > 1:
                raise ValueError(
                    f"Multiple chirpCfg entries cover in-frame chirp index"
                    f" {chirp_index}"
                )
            schedule.append(matches[0])
    elif len(profiles) != 1:
        raise ValueError(
            "Multiple profileCfg entries require an in-frame chirpCfg mapping"
        )

    if schedule:
        for config in schedule:
            profile_id = int(config["profile_id"])
            if profile_id not in profiles:
                raise ValueError(
                    f"In-frame chirpCfg references missing profile {profile_id}"
                )
            if any(value != 0.0 for value in config["variations"]):
                raise ValueError(
                    "In-frame chirpCfg parameter variations are not supported"
                )
        profile_ids = tuple(dict.fromkeys(
            int(config["profile_id"]) for config in schedule
        ))
    else:
        profile_ids = (next(iter(profiles)),)

    profile = profiles[profile_ids[0]]
    waveform_fields = (
        "start_freq_ghz",
        "idle_us",
        "ramp_us",
        "slope_mhz_per_us",
        "num_adc",
        "fs_ksps",
    )
    for profile_id in profile_ids[1:]:
        candidate = profiles[profile_id]
        if any(candidate[field] != profile[field]
               for field in waveform_fields):
            raise ValueError(
                "In-frame chirpCfg entries reference incompatible profiles"
            )

    start_freq_ghz = float(profile["start_freq_ghz"])
    carrier_frequency = (default_carrier_frequency
                         if not np.isfinite(start_freq_ghz)
                         else start_freq_ghz * 1e9)
    enabled_tx_indices = None
    tdm_enabled = False
    if schedule:
        tx_masks = tuple(int(config["tx_mask"]) for config in schedule)
        mask = 0
        for tx_mask in tx_masks:
            mask |= tx_mask
        if mask <= 0:
            raise ValueError("In-frame chirpCfg entries must enable a transmitter")
        enabled_tx_indices = tuple(
            index for index in range(mask.bit_length())
            if mask & (1 << index)
        )
        single_tx_slots = all(
            tx_mask > 0 and not tx_mask & (tx_mask - 1)
            for tx_mask in tx_masks
        )
        constant_mask = len(set(tx_masks)) == 1
        complete_cycle = False
        if len(enabled_tx_indices) > 1 and single_tx_slots:
            scheduled_tx_indices = tuple(
                tx_mask.bit_length() - 1 for tx_mask in tx_masks
            )
            cycle_length = len(enabled_tx_indices)
            first_cycle = scheduled_tx_indices[:cycle_length]
            complete_cycle = (
                len(set(first_cycle)) == cycle_length
                and len(scheduled_tx_indices) % cycle_length == 0
                and all(
                    scheduled_tx_indices[start:start + cycle_length]
                    == first_cycle
                    for start in range(0, len(scheduled_tx_indices), cycle_length)
                )
            )
            if complete_cycle:
                enabled_tx_indices = first_cycle
                tdm_enabled = True
        if not constant_mask and not complete_cycle:
            raise ValueError(
                "Changing in-frame Tx masks are unsupported; use one constant "
                "simultaneous mask or a complete repeated single-Tx TDM cycle"
            )

    num_slow_time_samples = (
        num_loops * (num_chirps_per_loop // len(enabled_tx_indices))
        if tdm_enabled
        else num_loops * num_chirps_per_loop
    )

    fmcw = FMCWConfig(
        carrier_frequency=carrier_frequency,
        slope=float(profile["slope_mhz_per_us"]) * 1e12,
        chirp_duration=float(profile["ramp_us"]) * 1e-6,
        chirp_repetition_time=(float(profile["idle_us"])
                               + float(profile["ramp_us"])) * 1e-6,
        sampling_frequency=float(profile["fs_ksps"]) * 1e3,
        num_adc_samples=int(profile["num_adc"]),
        num_chirps_per_frame=num_slow_time_samples,
        frame_period=frame_period_ms * 1e-3,
        num_tx=(len(enabled_tx_indices)
                if enabled_tx_indices is not None else num_chirps_per_loop),
        tdm_enabled=tdm_enabled,
    )
    return fmcw, enabled_tx_indices
