# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Radar sensor helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from .ti_cli import parse_ti_cli_config

from ..geometry import rotation_matrix

from .fmcw import FMCWConfig
from .hardware import RadarHardware

if TYPE_CHECKING:
    from sionna.rt import Transmitter, Receiver


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
    return mi.Point3f(pos[:, 0] / wavelength, pos[:, 1] / wavelength, pos[:, 2] / wavelength)


_rotation_matrix_numpy = rotation_matrix  # Compatibility alias.


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
        # A TDM cycle may contain slots whose TX channels were not retained.
        # Its slot count sets Doppler timing independently of stored hardware.
        if self.fmcw.tx_schedule != "tdm" and self.fmcw.num_tx != self.hardware.num_tx:
            raise ValueError(
                f"fmcw.num_tx ({self.fmcw.num_tx}) must match the hardware "
                f"transmitter count ({self.hardware.num_tx})"
            )

        # Include the board's analog IF limit in receiver filtering.
        board_if = (self.hardware.board_metadata or {}).get("adc", {}).get("if_bandwidth_hz")
        if board_if is not None:
            limit = min(float(board_if), self.fmcw.if_bandwidth_hz or float(board_if))
            object.__setattr__(self, "fmcw", replace(self.fmcw, if_bandwidth_hz=limit))
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "tx_power_dbm", tx_power_dbm)

    @classmethod
    def from_ti_board(
        cls,
        model: str,
        *,
        fmcw: FMCWConfig,
        name: str = "radar",
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        pattern_mode: str = "cosine30",
        cosine_half_power_angle_deg: float = 30.0,
        tx_power_dbm: float | None = None,
    ) -> "RadarSensor":
        """Builds a radar sensor from a supported TI board catalog entry."""

        from .ti import get_ti_board_spec  # pylint: disable=import-outside-toplevel

        spec = get_ti_board_spec(model)
        hardware = RadarHardware.from_ti_board(
            spec.key,
            pattern_mode=pattern_mode,
            cosine_half_power_angle_deg=cosine_half_power_angle_deg,
        )
        fmcw = replace(fmcw, num_tx=hardware.num_tx)
        power = spec.tx_power_dbm if tx_power_dbm is None else float(tx_power_dbm)
        return cls(
            name=name,
            position=tuple(position),
            orientation=tuple(orientation),
            hardware=hardware,
            fmcw=fmcw,
            tx_power_dbm=power,
        )

    @classmethod
    def from_virtual_ura(
        cls,
        *,
        rows: int,
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
        antenna_pattern: object | None = None,
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
        fmcw = replace(fmcw, num_tx=1, tx_schedule="simultaneous")
        return cls(
            name=name,
            position=tuple(position),
            orientation=tuple(orientation),
            hardware=hardware,
            fmcw=fmcw,
            tx_power_dbm=float(tx_power_dbm),
        )

    @classmethod
    def from_ti_cli_config(
        cls,
        filename: str,
        *,
        name: str = "radar",
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        hardware_model: str = "IWR6843AOP",
        default_carrier_frequency: float = 77e9,
        pattern_mode: str = "cosine30",
        tx_power_dbm: float = 12.0,
    ) -> "RadarSensor":
        """Builds a radar sensor from a TI mmWave CLI config file."""
        fmcw, enabled_tx_indices = parse_ti_cli_config(filename, default_carrier_frequency)
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
        fmcw = replace(fmcw, num_tx=hardware.num_tx)
        return cls(
            name=name,
            position=tuple(position),
            orientation=tuple(orientation),
            hardware=hardware,
            fmcw=fmcw,
            tx_power_dbm=tx_power_dbm,
        )

    def configure_scene(self, scene) -> tuple[Transmitter, Receiver]:
        """
        Adds radar Tx/Rx devices to ``scene`` and configures their arrays.

        The Tx device sits at the Tx-array centre and the Rx device at the
        Rx-array centre, with element positions relative to them. Sionna
        traces synthetic arrays from the device positions, so these centres
        are RT's phase centres, matching PO and the RT path bank. Existing
        scene Tx/Rx arrays are replaced because Sionna scenes use one shared
        Tx array and one shared Rx array.
        """
        from sionna.rt import AntennaArray, Receiver, Transmitter
        from sionna.rt.antenna_pattern import antenna_pattern_registry

        wavelength = self.fmcw.wavelength
        pattern = antenna_pattern_registry.get("iso")(polarization="V")
        tx_local = np.asarray(self.hardware.tx_positions, dtype=float)
        rx_local = np.asarray(self.hardware.rx_positions, dtype=float)
        scene.tx_array = AntennaArray(
            pattern, _positions_to_point3f(tx_local - tx_local.mean(axis=0), wavelength)
        )
        scene.rx_array = AntennaArray(
            pattern, _positions_to_point3f(rx_local - rx_local.mean(axis=0), wavelength)
        )
        scene.frequency = self.fmcw.carrier_frequency
        tx_position = tuple(float(v) for v in self.world_tx_positions().mean(axis=0))
        rx_position = tuple(float(v) for v in self.world_rx_positions().mean(axis=0))

        tx_name = f"{self.name}-tx"
        rx_name = f"{self.name}-rx"
        tx = scene.get(tx_name)
        rx = scene.get(rx_name)
        if tx is None:
            tx = Transmitter(
                name=tx_name,
                position=tx_position,
                orientation=self.orientation,
                power_dbm=self.tx_power_dbm,
            )
            scene.add(tx)
        else:
            tx.position = tx_position
            tx.orientation = self.orientation
            tx.power_dbm = self.tx_power_dbm
        if rx is None:
            rx = Receiver(name=rx_name, position=rx_position, orientation=self.orientation)
            scene.add(rx)
        else:
            rx.position = rx_position
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
        rot = rotation_matrix(self.orientation)
        local = (rot.T @ (flat - np.asarray(self.position, dtype=float)).T).T
        return local.reshape(pts.shape)

    def local_azimuth_elevation(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns local azimuth/elevation angles for world points in degrees.

        The local ``x`` axis is boresight, ``y`` is horizontal/azimuth, and
        ``z`` is vertical/elevation.
        """

        local = self.world_to_local_points(points).reshape((-1, 3))
        azimuth = np.degrees(np.arctan2(local[:, 1], local[:, 0]))
        elevation = np.degrees(np.arctan2(local[:, 2], np.hypot(local[:, 0], local[:, 1])))
        return azimuth, elevation

    def _world_positions(self, local_positions: np.ndarray) -> np.ndarray:
        """Transforms local element positions to world coordinates in meters."""

        rot = rotation_matrix(self.orientation)
        pos = np.asarray(local_positions, dtype=float)
        world = (rot @ pos.T).T
        return world + np.asarray(self.position, dtype=float)[None, :]


_parse_ti_cli_config = parse_ti_cli_config  # Compatibility alias.
