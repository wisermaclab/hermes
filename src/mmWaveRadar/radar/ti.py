# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Texas Instruments mmWave radar board catalog."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
import json
import operator
import os
from pathlib import Path
from typing import Mapping

import numpy as np

from .hardware import _broadcast_channel_angles


_PATTERN_ASSET = "ti_digitized_patterns.npz"
_PATTERN_ASSET_ENV = "MMWAVE_TI_PATTERN_NPZ"


class TIDigitizedPatternUnavailableError(FileNotFoundError):
    """Raised when an explicitly requested digitized TI pattern is unavailable."""


def _digitized_pattern_asset():
    """Returns the configured or packaged digitized-pattern asset."""

    configured = os.environ.get(_PATTERN_ASSET_ENV)
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_file():
            raise TIDigitizedPatternUnavailableError(
                f"{_PATTERN_ASSET_ENV} points to a missing or non-file path: "
                f"{path}. Use pattern_mode='cosine30' for the built-in "
                "synthetic pattern."
            )
        return path, f"external:{path}"

    asset = files(__package__).joinpath("data", _PATTERN_ASSET)
    if not asset.is_file():
        raise TIDigitizedPatternUnavailableError(
            "Digitized TI antenna patterns are optional, and no pattern NPZ "
            "asset is installed. Set MMWAVE_TI_PATTERN_NPZ to an authorized "
            "ti_digitized_patterns.npz file, or use "
            "pattern_mode='cosine30' for the built-in synthetic pattern."
        )
    return asset, f"package:{__package__}/data/{_PATTERN_ASSET}"


def ti_digitized_pattern_asset_available() -> bool:
    """Returns whether a configured or packaged digitized-pattern NPZ exists."""

    try:
        _digitized_pattern_asset()
    except TIDigitizedPatternUnavailableError:
        return False
    return True


def _canonical_key(model: str) -> str:
    """Normalizes a TI board model or alias to a catalog key."""

    token = "".join(ch for ch in str(model).upper() if ch.isalnum())
    for key, spec in _SPECS.items():
        aliases = {key, *(spec.aliases)}
        if token in {"".join(ch for ch in alias.upper() if ch.isalnum())
                    for alias in aliases}:
            return key
    raise KeyError(f"Unknown TI radar board model: {model!r}")


def is_ti_board_model(model: str) -> bool:
    """Returns ``True`` if ``model`` is a known TI board/catalog alias."""

    try:
        _canonical_key(model)
    except KeyError:
        return False
    return True


def available_ti_boards() -> tuple[str, ...]:
    """Returns canonical TI board names available in the local catalog."""

    return tuple(_SPECS)


def get_ti_board_spec(model: str) -> "TIBoardSpec":
    """Returns the board spec for ``model`` or one of its aliases."""

    return _SPECS[_canonical_key(model)]


@dataclass(frozen=True)
class VirtualChannelSpec:
    """One documented virtual channel in simulator channel order."""

    ti_id: int
    label: str
    tx_index: int
    rx_index: int
    position_lambda: tuple[float, float]
    phase_sign: complex = 1.0 + 0.0j


@dataclass(frozen=True)
class TIBoardSpec:
    """Digital-twin metadata for a supported TI mmWave evaluation board."""

    key: str
    aliases: tuple[str, ...]
    device: str
    frequency_range_hz: tuple[float, float]
    design_frequency_hz: float
    lambda_m: float
    tx_positions_lambda_yz: tuple[tuple[float, float], ...]
    rx_positions_lambda_yz: tuple[tuple[float, float], ...]
    virtual_channels: tuple[VirtualChannelSpec, ...]
    tx_power_dbm: float
    antenna_gain_dbi_per_element: float
    noise: Mapping[str, float | str]
    adc: Mapping[str, float | int | str]
    fov_deg: Mapping[str, tuple[float, float]]
    angular_resolution_deg: Mapping[str, float]
    pattern_asset_key: str | None = None
    fmcw_profiles: Mapping[str, Mapping[str, object]] | None = None

    @property
    def num_tx(self) -> int:
        """Number of transmitters on the board."""

        return len(self.tx_positions_lambda_yz)

    @property
    def num_rx(self) -> int:
        """Number of receivers on the board."""

        return len(self.rx_positions_lambda_yz)

    @property
    def num_virtual_channels(self) -> int:
        """Number of simulator virtual channels."""

        return self.num_tx * self.num_rx

    @property
    def combined_antenna_gain_dbi(self) -> float:
        """Nominal Tx/Rx scalar antenna gain in dB for channel coefficients."""

        return 2.0 * float(self.antenna_gain_dbi_per_element)

    def tx_positions_m(self) -> np.ndarray:
        """Returns Tx element positions in local ``[x, y, z]`` meters."""

        yz = np.asarray(self.tx_positions_lambda_yz, dtype=float) * self.lambda_m
        out = np.zeros((yz.shape[0], 3), dtype=float)
        out[:, 1:] = yz
        return out

    def rx_positions_m(self) -> np.ndarray:
        """Returns Rx element positions in local ``[x, y, z]`` meters."""

        yz = np.asarray(self.rx_positions_lambda_yz, dtype=float) * self.lambda_m
        out = np.zeros((yz.shape[0], 3), dtype=float)
        out[:, 1:] = yz
        return out

    def virtual_channel_labels(self) -> tuple[str, ...]:
        """Returns virtual-channel labels in simulator channel order."""

        return tuple(vc.label for vc in self.virtual_channels)

    def virtual_channel_tx_indices(self) -> np.ndarray:
        """Returns zero-based Tx indices for virtual channels."""

        return np.asarray([vc.tx_index for vc in self.virtual_channels],
                          dtype=np.int64)

    def virtual_channel_rx_indices(self) -> np.ndarray:
        """Returns zero-based Rx indices for virtual channels."""

        return np.asarray([vc.rx_index for vc in self.virtual_channels],
                          dtype=np.int64)

    def virtual_channel_positions_lambda(self) -> np.ndarray:
        """Returns documented ``[horizontal, vertical]`` positions in wavelengths."""

        return np.asarray([vc.position_lambda for vc in self.virtual_channels],
                          dtype=float)

    def channel_phase_signs(self) -> np.ndarray:
        """Returns fixed complex phase signs for virtual channels."""

        return np.asarray([vc.phase_sign for vc in self.virtual_channels],
                          dtype=np.complex128)

    def pattern(
        self,
        pattern_mode: str = "cosine30",
        *,
        cosine_half_power_angle_deg: float = 30.0,
    ) -> "TIDigitizedPattern | SimpleCosinePattern | None":
        """Returns the requested antenna-pattern model for this board."""

        mode = str(pattern_mode).lower().replace("-", "_")
        if mode in ("none", "off", "scalar", "isotropic"):
            return None
        if mode in ("cosine", "cosine30", "cosine_30", "cosine_3db_30"):
            return SimpleCosinePattern(
                num_virtual_channels=self.num_virtual_channels,
                half_power_angle_deg=cosine_half_power_angle_deg,
            )
        if mode != "digitized":
            raise ValueError(
                "pattern_mode must be 'digitized', 'none'/'scalar'/"
                "'isotropic', or 'cosine'/'cosine30'"
            )
        if self.pattern_asset_key is None:
            return None
        return load_ti_digitized_pattern(self.pattern_asset_key)

    def metadata(self) -> dict:
        """Returns serializable board metadata."""

        return {
            "key": self.key,
            "device": self.device,
            "frequency_range_hz": tuple(float(v) for v in self.frequency_range_hz),
            "design_frequency_hz": float(self.design_frequency_hz),
            "lambda_m": float(self.lambda_m),
            "tx_power_dbm": float(self.tx_power_dbm),
            "antenna_gain_dbi_per_element": float(
                self.antenna_gain_dbi_per_element
            ),
            "combined_antenna_gain_dbi": float(self.combined_antenna_gain_dbi),
            "noise": dict(self.noise),
            "adc": dict(self.adc),
            "fov_deg": dict(self.fov_deg),
            "angular_resolution_deg": dict(self.angular_resolution_deg),
            "fmcw_profiles": {
                str(key): dict(value)
                for key, value in (self.fmcw_profiles or {}).items()
            },
        }


@dataclass(frozen=True)
class TIDigitizedPattern:
    """Digitized 2D combined Tx/Rx pattern lookup tables."""

    board_key: str
    channel_labels: tuple[str, ...]
    band_centers_hz: np.ndarray
    azimuth_angles_deg: np.ndarray
    elevation_angles_deg: np.ndarray
    azimuth_loss_db: np.ndarray
    elevation_loss_db: np.ndarray
    metadata: Mapping[str, object]

    def __post_init__(self):
        board_key = str(self.board_key).strip()
        labels = tuple(str(label).strip() for label in self.channel_labels)
        bands = np.array(self.band_centers_hz, dtype=float, copy=True)
        azimuth_angles = np.array(
            self.azimuth_angles_deg,
            dtype=float,
            copy=True,
        )
        elevation_angles = np.array(
            self.elevation_angles_deg,
            dtype=float,
            copy=True,
        )
        azimuth_loss = np.array(self.azimuth_loss_db, dtype=float, copy=True)
        elevation_loss = np.array(
            self.elevation_loss_db,
            dtype=float,
            copy=True,
        )
        if not board_key:
            raise ValueError("board_key must be a non-empty string")
        if not labels or any(not label for label in labels):
            raise ValueError("channel_labels must contain non-empty labels")
        if len(set(labels)) != len(labels):
            raise ValueError("channel_labels must be unique")
        for name, axis, positive in (
            ("band_centers_hz", bands, True),
            ("azimuth_angles_deg", azimuth_angles, False),
            ("elevation_angles_deg", elevation_angles, False),
        ):
            if axis.ndim != 1 or axis.size == 0:
                raise ValueError(f"{name} must be a non-empty 1-D array")
            if not np.all(np.isfinite(axis)):
                raise ValueError(f"{name} must contain only finite values")
            if positive and np.any(axis <= 0.0):
                raise ValueError(f"{name} must contain positive frequencies")
            if axis.size > 1 and np.any(np.diff(axis) <= 0.0):
                raise ValueError(f"{name} must be strictly increasing")
        expected_azimuth_shape = (
            bands.size,
            len(labels),
            azimuth_angles.size,
        )
        expected_elevation_shape = (
            bands.size,
            len(labels),
            elevation_angles.size,
        )
        if azimuth_loss.shape != expected_azimuth_shape:
            raise ValueError(
                f"azimuth_loss_db must have shape {expected_azimuth_shape}"
            )
        if elevation_loss.shape != expected_elevation_shape:
            raise ValueError(
                f"elevation_loss_db must have shape {expected_elevation_shape}"
            )
        if not (
            np.all(np.isfinite(azimuth_loss))
            and np.all(np.isfinite(elevation_loss))
        ):
            raise ValueError("pattern loss tables must contain only finite values")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "board_key", board_key)
        object.__setattr__(self, "channel_labels", labels)
        object.__setattr__(self, "band_centers_hz", bands)
        object.__setattr__(self, "azimuth_angles_deg", azimuth_angles)
        object.__setattr__(self, "elevation_angles_deg", elevation_angles)
        object.__setattr__(self, "azimuth_loss_db", azimuth_loss)
        object.__setattr__(self, "elevation_loss_db", elevation_loss)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def num_virtual_channels(self) -> int:
        """Number of virtual-channel curves in this pattern."""

        return len(self.channel_labels)

    def subset(self, indices: np.ndarray) -> "TIDigitizedPattern":
        """Returns a pattern with only ``indices`` virtual channels."""

        raw_indices = np.asarray(indices)
        if not np.issubdtype(raw_indices.dtype, np.integer):
            raise ValueError("indices must contain integers")
        ind = np.asarray(raw_indices, dtype=np.int64).reshape(-1)
        if ind.size == 0:
            raise ValueError("indices must select at least one channel")
        if np.any((ind < 0) | (ind >= self.num_virtual_channels)):
            raise ValueError("indices contains an out-of-range channel")
        if np.unique(ind).size != ind.size:
            raise ValueError("indices must not contain duplicate channels")
        return TIDigitizedPattern(
            board_key=self.board_key,
            channel_labels=tuple(self.channel_labels[int(i)] for i in ind),
            band_centers_hz=self.band_centers_hz.copy(),
            azimuth_angles_deg=self.azimuth_angles_deg.copy(),
            elevation_angles_deg=self.elevation_angles_deg.copy(),
            azimuth_loss_db=self.azimuth_loss_db[:, ind, :].copy(),
            elevation_loss_db=self.elevation_loss_db[:, ind, :].copy(),
            metadata=dict(self.metadata),
        )

    def loss_db(
        self,
        *,
        azimuth_deg=None,
        elevation_deg=None,
        frequency_hz: float | None = None,
    ) -> np.ndarray:
        """Returns separable normalized combined-pattern loss in dB.

        One-dimensional angle arrays are common path angles. Channel-specific
        inputs must have at least two dimensions with the channel axis first.
        """

        az_curves, el_curves = self._curves_for_frequency(frequency_hz)
        azimuth, elevation, shape = _broadcast_channel_angles(
            self.num_virtual_channels,
            azimuth_deg,
            elevation_deg,
        )
        out = np.zeros(shape, dtype=float)
        if azimuth is not None:
            out += self._interp_per_channel(
                azimuth,
                self.azimuth_angles_deg,
                az_curves,
            )
        if elevation is not None:
            out += self._interp_per_channel(
                elevation,
                self.elevation_angles_deg,
                el_curves,
            )
        return out

    def _curves_for_frequency(self, frequency_hz: float | None
                              ) -> tuple[np.ndarray, np.ndarray]:
        """Interpolates azimuth/elevation pattern curves at ``frequency_hz``."""

        if frequency_hz is None:
            return self.azimuth_loss_db[0], self.elevation_loss_db[0]
        try:
            freq = float(frequency_hz)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("frequency_hz must be finite and positive") from exc
        if not np.isfinite(freq) or freq <= 0.0:
            raise ValueError("frequency_hz must be finite and positive")
        if self.band_centers_hz.size <= 1:
            return self.azimuth_loss_db[0], self.elevation_loss_db[0]

        centers = np.asarray(self.band_centers_hz, dtype=float)
        weights = np.zeros(centers.size, dtype=float)
        if freq <= centers[0]:
            weights[0] = 1.0
        elif freq >= centers[-1]:
            weights[-1] = 1.0
        else:
            high = int(np.searchsorted(centers, freq, side="right"))
            low = high - 1
            frac = (freq - centers[low]) / (centers[high] - centers[low])
            weights[low] = 1.0 - frac
            weights[high] = frac
        az = np.tensordot(weights, self.azimuth_loss_db, axes=(0, 0))
        el = np.tensordot(weights, self.elevation_loss_db, axes=(0, 0))
        return az, el

    def _interp_per_channel(
        self,
        angle_deg,
        table_angles: np.ndarray,
        curves_db: np.ndarray,
    ) -> np.ndarray:
        """Interpolates one pattern table independently for each virtual channel."""

        angles = np.asarray(angle_deg, dtype=float)
        if angles.shape[0:1] != (self.num_virtual_channels,):
            raise ValueError("angle_deg must have a virtual-channel first axis")
        out = np.empty(angles.shape, dtype=float)
        for channel in range(self.num_virtual_channels):
            out[channel] = np.interp(
                angles[channel],
                table_angles,
                curves_db[channel],
            )
        return out


@dataclass(frozen=True)
class SimpleCosinePattern:
    """Channel-independent cosine power pattern around local boresight."""

    num_virtual_channels: int
    half_power_angle_deg: float = 30.0
    floor_loss_db: float = -80.0

    def __post_init__(self):
        if isinstance(self.num_virtual_channels, (bool, np.bool_)):
            raise ValueError("num_virtual_channels must be a positive integer")
        try:
            channel_count = operator.index(self.num_virtual_channels)
        except TypeError as exc:
            raise ValueError(
                "num_virtual_channels must be a positive integer"
            ) from exc
        try:
            half_power_angle = float(self.half_power_angle_deg)
            floor_loss = float(self.floor_loss_db)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "half_power_angle_deg and floor_loss_db must be finite scalars"
            ) from exc
        if channel_count <= 0:
            raise ValueError("num_virtual_channels must be a positive integer")
        if not np.isfinite(half_power_angle) or not 0.0 < half_power_angle < 90.0:
            raise ValueError("half_power_angle_deg must be between 0 and 90 degrees")
        if not np.isfinite(floor_loss) or floor_loss > 0.0:
            raise ValueError("floor_loss_db must be finite and non-positive")
        object.__setattr__(self, "num_virtual_channels", int(channel_count))
        object.__setattr__(self, "half_power_angle_deg", half_power_angle)
        object.__setattr__(self, "floor_loss_db", floor_loss)

    def subset(self, indices: np.ndarray) -> "SimpleCosinePattern":
        """Returns an equivalent cosine pattern for a selected channel subset."""

        return SimpleCosinePattern(
            num_virtual_channels=int(np.asarray(indices, dtype=np.int64).size),
            half_power_angle_deg=float(self.half_power_angle_deg),
            floor_loss_db=float(self.floor_loss_db),
        )

    def loss_db(
        self,
        *,
        azimuth_deg=None,
        elevation_deg=None,
        frequency_hz: float | None = None,
    ) -> np.ndarray:
        """Returns power-pattern loss in dB, with -3 dB at half-power angle."""

        del frequency_hz
        if azimuth_deg is None and elevation_deg is None:
            return np.zeros((int(self.num_virtual_channels),), dtype=float)

        azimuth, elevation, shape = _broadcast_channel_angles(
            int(self.num_virtual_channels),
            azimuth_deg,
            elevation_deg,
        )
        if azimuth is None:
            azimuth = np.zeros(shape, dtype=float)
        if elevation is None:
            elevation = np.zeros(shape, dtype=float)
        cos_off_axis = np.cos(np.deg2rad(elevation)) * np.cos(np.deg2rad(azimuth))
        half_power_cos = np.cos(np.deg2rad(float(self.half_power_angle_deg)))
        exponent = np.log(0.5) / np.log(max(float(half_power_cos), 1e-12))
        power_gain = np.maximum(np.maximum(cos_off_axis, 0.0) ** exponent, 1e-30)
        loss = 10.0 * np.log10(power_gain)
        loss = np.maximum(loss, float(self.floor_loss_db))

        return loss.astype(float, copy=False)


_PATTERN_CACHE: dict[tuple[str, str], TIDigitizedPattern] = {}


def load_ti_digitized_pattern(board_key: str) -> TIDigitizedPattern:
    """Loads digitized pattern data from an external or packaged NPZ asset."""

    key = _canonical_key(board_key)
    data_path, source_key = _digitized_pattern_asset()
    cache_key = (key, source_key)
    if cache_key in _PATTERN_CACHE:
        return _PATTERN_CACHE[cache_key]

    with data_path.open("rb") as f:
        with np.load(f, allow_pickle=False) as data:
            prefix = f"{key}__"
            metadata = json.loads(str(data[f"{prefix}metadata_json"]))
            pattern = TIDigitizedPattern(
                board_key=key,
                channel_labels=tuple(
                    str(v) for v in data[f"{prefix}channel_labels"]
                ),
                band_centers_hz=np.asarray(
                    data[f"{prefix}band_centers_hz"],
                    dtype=float,
                ),
                azimuth_angles_deg=np.asarray(
                    data[f"{prefix}azimuth_angles_deg"],
                    dtype=float,
                ),
                elevation_angles_deg=np.asarray(
                    data[f"{prefix}elevation_angles_deg"],
                    dtype=float,
                ),
                azimuth_loss_db=np.asarray(
                    data[f"{prefix}azimuth_loss_db"],
                    dtype=float,
                ),
                elevation_loss_db=np.asarray(
                    data[f"{prefix}elevation_loss_db"],
                    dtype=float,
                ),
                metadata=metadata,
            )
    _PATTERN_CACHE[cache_key] = pattern
    return pattern


def _tx_major_channels(
    labels: tuple[tuple[int, int, int, float, float, complex], ...]
) -> tuple[VirtualChannelSpec, ...]:
    """Builds documented TI virtual-channel specs from TX-major label tuples."""

    return tuple(
        VirtualChannelSpec(
            ti_id=ti_id,
            label=f"TX{tx} to RX{rx}",
            tx_index=tx - 1,
            rx_index=rx - 1,
            position_lambda=(float(y), float(z)),
            phase_sign=complex(sign),
        )
        for ti_id, tx, rx, y, z, sign in labels
    )


def _virtual_channels_from_positions(
    tx_positions_lambda_yz: tuple[tuple[float, float], ...],
    rx_positions_lambda_yz: tuple[tuple[float, float], ...],
) -> tuple[VirtualChannelSpec, ...]:
    """Derives virtual-channel specs from Tx/Rx aperture coordinates."""

    channels = []
    ti_id = 1
    for tx_index, (tx_y, tx_z) in enumerate(tx_positions_lambda_yz):
        for rx_index, (rx_y, rx_z) in enumerate(rx_positions_lambda_yz):
            channels.append(
                VirtualChannelSpec(
                    ti_id=ti_id,
                    label=f"TX{tx_index + 1} to RX{rx_index + 1}",
                    tx_index=tx_index,
                    rx_index=rx_index,
                    position_lambda=(float(tx_y + rx_y), float(tx_z + rx_z)),
                    phase_sign=1.0 + 0.0j,
                )
            )
            ti_id += 1
    return tuple(channels)


_AWRL6844_CHANNELS = _tx_major_channels((
    (3, 1, 1, 1.0, 0.0, +1),
    (7, 1, 2, 1.0, 0.5, +1),
    (8, 1, 3, 1.5, 0.5, +1),
    (4, 1, 4, 1.5, 0.0, +1),
    (1, 2, 1, 0.0, 0.0, -1),
    (5, 2, 2, 0.0, 0.5, -1),
    (6, 2, 3, 0.5, 0.5, -1),
    (2, 2, 4, 0.5, 0.0, -1),
    (9, 3, 1, 0.0, 1.0, +1),
    (13, 3, 2, 0.0, 1.5, +1),
    (14, 3, 3, 0.5, 1.5, +1),
    (10, 3, 4, 0.5, 1.0, +1),
    (11, 4, 1, 1.0, 1.0, -1),
    (15, 4, 2, 1.0, 1.5, -1),
    (16, 4, 3, 1.5, 1.5, -1),
    (12, 4, 4, 1.5, 1.0, -1),
))

_IWR6843ISK_CHANNELS = _tx_major_channels((
    (1, 1, 1, 0.0, 0.0, +1),
    (2, 1, 2, 0.5, 0.0, +1),
    (3, 1, 3, 1.0, 0.0, +1),
    (4, 1, 4, 1.5, 0.0, +1),
    (5, 2, 1, 1.0, 0.5, +1),
    (6, 2, 2, 1.5, 0.5, +1),
    (7, 2, 3, 2.0, 0.5, +1),
    (8, 2, 4, 2.5, 0.5, +1),
    (9, 3, 1, 2.0, 0.0, +1),
    (10, 3, 2, 2.5, 0.0, +1),
    (11, 3, 3, 3.0, 0.0, +1),
    (12, 3, 4, 3.5, 0.0, +1),
))

_IWR6843AOP_CHANNELS = _tx_major_channels((
    (1, 1, 1, 0.5, 1.0, -1),
    (2, 1, 2, 0.5, 1.5, +1),
    (3, 1, 3, 0.0, 1.0, -1),
    (4, 1, 4, 0.0, 1.5, +1),
    (5, 2, 1, 1.5, 0.0, -1),
    (6, 2, 2, 1.5, 0.5, +1),
    (7, 2, 3, 1.0, 0.0, -1),
    (8, 2, 4, 1.0, 0.5, +1),
    (9, 3, 1, 0.5, 0.0, -1),
    (10, 3, 2, 0.5, 0.5, +1),
    (11, 3, 3, 0.0, 0.0, -1),
    (12, 3, 4, 0.0, 0.5, +1),
))

_MMWCAS_RF_EVM_TX_POSITIONS_HALF_LAMBDA_YZ = (
    (11, 6),
    (10, 4),
    (9, 1),
    (32, 0),
    (28, 0),
    (24, 0),
    (20, 0),
    (16, 0),
    (12, 0),
    (8, 0),
    (4, 0),
    (0, 0),
)
_MMWCAS_RF_EVM_RX_POSITIONS_HALF_LAMBDA_YZ = (
    (11, 0),
    (12, 0),
    (13, 0),
    (14, 0),
    (50, 0),
    (51, 0),
    (52, 0),
    (53, 0),
    (46, 0),
    (47, 0),
    (48, 0),
    (49, 0),
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0),
)
_MMWCAS_RF_EVM_TX_POSITIONS_LAMBDA_YZ = tuple(
    (0.5 * float(y), 0.5 * float(z))
    for y, z in _MMWCAS_RF_EVM_TX_POSITIONS_HALF_LAMBDA_YZ
)
_MMWCAS_RF_EVM_RX_POSITIONS_LAMBDA_YZ = tuple(
    (0.5 * float(y), 0.5 * float(z))
    for y, z in _MMWCAS_RF_EVM_RX_POSITIONS_HALF_LAMBDA_YZ
)
_MMWCAS_RF_EVM_CHANNELS = _virtual_channels_from_positions(
    _MMWCAS_RF_EVM_TX_POSITIONS_LAMBDA_YZ,
    _MMWCAS_RF_EVM_RX_POSITIONS_LAMBDA_YZ,
)

_MMWCAS_RF_EVM_FMCW_PROFILES = {
    "rtpose_high_resolution_mimo": {
        "description": "RT-Pose 10 FPS 12-TX TDM-MIMO profile.",
        "carrier_frequency_hz": 78.024e9,
        "start_frequency_hz": 77.0e9,
        "slope_hz_per_s": 64.985e12,
        "chirp_duration_s": 60.0e-6,
        "chirp_repetition_time_s": 65.0e-6,
        "adc_start_time_s": 5.0e-6,
        "sampling_frequency_hz": 5.0e6,
        "num_adc_samples": 256,
        "num_tx": 12,
        "tdm_enabled": True,
        "num_chirps_per_tx": 64,
        "num_chirps_per_frame": 768,
        "frame_period_s": 0.1,
        "range_resolution_m": 0.045,
        "velocity_resolution_mps": 0.039,
        "source": "RT-Pose hardware_param.m / dataset description.",
    },
    "ti_mimo_srr": {
        "description": "TIDEP-01012 Table 2 MIMO SRR example.",
        "idle_time_s": 5.0e-6,
        "adc_start_time_s": 6.0e-6,
        "ramp_end_time_s": 40.0e-6,
        "num_adc_samples": 256,
        "slope_hz_per_s": 79.0e12,
        "sampling_frequency_hz": 8.0e6,
        "tdm_enabled": True,
        "num_chirps_per_frame_per_tx": 128,
        "effective_chirp_time_s": 34.0e-6,
        "bandwidth_hz": 2.528e9,
        "published_frame_period_s": 69.0e-3,
        # Table 2 rounds the 12-Tx train to 69 ms. Use its exact physical
        # duration at runtime so consecutive simulated frames cannot overlap.
        "frame_period_s": 12 * 128 * (5.0e-6 + 40.0e-6),
        "source": "TIDUEN5A Table 2.",
    },
    "ti_mimo_mrr": {
        "description": "TIDEP-01012 Table 2 MIMO MRR example.",
        "idle_time_s": 4.0e-6,
        "adc_start_time_s": 5.0e-6,
        "ramp_end_time_s": 23.0e-6,
        "num_adc_samples": 256,
        "slope_hz_per_s": 15.0e12,
        "sampling_frequency_hz": 15.0e6,
        "tdm_enabled": True,
        "num_chirps_per_frame_per_tx": 128,
        "effective_chirp_time_s": 17.0e-6,
        "bandwidth_hz": 256.0e6,
        "published_frame_period_s": 4.4e-3,
        # Table 2's published 4.4 ms is shorter than the 12-Tx train. Preserve
        # it above as provenance and use the exact nonoverlapping duration.
        "frame_period_s": 12 * 128 * (4.0e-6 + 23.0e-6),
        "source": "TIDUEN5A Table 2.",
    },
    "ti_txbf": {
        "description": "TIDEP-01012 Table 3 TX beamforming example.",
        "idle_time_s": 4.0e-6,
        "adc_start_time_s": 5.0e-6,
        "ramp_end_time_s": 23.0e-6,
        "num_adc_samples": 256,
        "slope_hz_per_s": 2.5e12,
        "sampling_frequency_hz": 15.0e6,
        "num_chirps_per_frame": 128,
        "effective_chirp_time_s": 17.0e-6,
        "bandwidth_hz": 43.0e6,
        "frame_period_s": 3.5e-3,
        "source": "TIDUEN5A Table 3.",
    },
}


_SPECS: dict[str, TIBoardSpec] = {
    "MMWCAS_RF_EVM": TIBoardSpec(
        key="MMWCAS_RF_EVM",
        aliases=(
            "MMWCAS-RF-EVM",
            "MMWCASRFEVM",
            "AWR2243CASCADE",
            "AWR2243_CASCADE",
            "TIDEP01012",
            "TIDEP-01012",
        ),
        device="4 x AWR2243 cascade imaging radar",
        frequency_range_hz=(76.0e9, 81.0e9),
        design_frequency_hz=76.8e9,
        lambda_m=299792458.0 / 76.8e9,
        tx_positions_lambda_yz=_MMWCAS_RF_EVM_TX_POSITIONS_LAMBDA_YZ,
        rx_positions_lambda_yz=_MMWCAS_RF_EVM_RX_POSITIONS_LAMBDA_YZ,
        virtual_channels=_MMWCAS_RF_EVM_CHANNELS,
        tx_power_dbm=12.0,
        antenna_gain_dbi_per_element=0.0,
        noise={"metric": "unspecified", "source": "not listed in TIDUEN5A"},
        adc={
            "bits": 16,
            "if_bandwidth_hz": 15.0e6,
            "rtpose_sampling_frequency_hz": 5.0e6,
        },
        fov_deg={"azimuth": (-70.0, 70.0), "elevation": (-20.0, 20.0)},
        angular_resolution_deg={"azimuth": 1.4, "elevation": 18.0},
        fmcw_profiles=_MMWCAS_RF_EVM_FMCW_PROFILES,
    ),
    "AWRL6844EVM": TIBoardSpec(
        key="AWRL6844EVM",
        aliases=("AWRL6844", "IWRL6844", "XWRL6844EVM", "XWRL6844"),
        device="AWRL6844 / IWRL6844",
        frequency_range_hz=(57.0e9, 64.0e9),
        design_frequency_hz=59.0e9,
        lambda_m=5.081e-3,
        tx_positions_lambda_yz=((1.0, 0.0), (0.0, 0.0),
                                (0.0, 1.0), (1.0, 1.0)),
        rx_positions_lambda_yz=((0.0, 0.0), (0.0, 0.5),
                                (0.5, 0.5), (0.5, 0.0)),
        virtual_channels=_AWRL6844_CHANNELS,
        tx_power_dbm=12.5,
        antenna_gain_dbi_per_element=5.5,
        noise={"metric": "NF", "nf_db": 12.5, "n_10mhz_dbm": -91.5},
        adc={"bits": 12, "real_sample_rate_sps": 25.0e6,
             "if_bandwidth_hz": 10.0e6},
        fov_deg={"azimuth": (-60.0, 60.0), "elevation": (-60.0, 60.0)},
        angular_resolution_deg={"azimuth": 29.0, "elevation": 29.0},
        pattern_asset_key="AWRL6844EVM",
    ),
    "IWR6843ISK": TIBoardSpec(
        key="IWR6843ISK",
        aliases=("XWR68XX", "XWR68XXISK", "XWR6843ISK"),
        device="IWR6843",
        frequency_range_hz=(60.0e9, 64.0e9),
        design_frequency_hz=60.0e9,
        lambda_m=5.0e-3,
        tx_positions_lambda_yz=((0.0, 0.0), (1.0, 0.5), (2.0, 0.0)),
        rx_positions_lambda_yz=((0.0, 0.0), (0.5, 0.0),
                                (1.0, 0.0), (1.5, 0.0)),
        virtual_channels=_IWR6843ISK_CHANNELS,
        tx_power_dbm=12.0,
        antenna_gain_dbi_per_element=7.0,
        noise={"metric": "NF", "nf_db": 12.0, "n_10mhz_dbm": -92.0},
        adc={"bits": 12, "real_or_complex2x_sample_rate_sps": 25.0e6,
             "complex1x_sample_rate_sps": 12.5e6, "if_bandwidth_hz": 10.0e6},
        fov_deg={"azimuth": (-60.0, 60.0), "elevation": (-15.0, 15.0)},
        angular_resolution_deg={"azimuth": 15.0, "elevation": 58.0},
        pattern_asset_key="IWR6843ISK",
    ),
    "IWR6843AOPEVM": TIBoardSpec(
        key="IWR6843AOPEVM",
        aliases=("IWR6843AOP", "AWR6843AOP", "XWR68XXAOP",
                 "XWR6843AOP"),
        device="IWR6843AOP",
        frequency_range_hz=(60.0e9, 64.0e9),
        design_frequency_hz=60.0e9,
        lambda_m=5.0e-3,
        tx_positions_lambda_yz=((0.0, 1.0), (1.0, 0.0), (0.0, 0.0)),
        rx_positions_lambda_yz=((0.5, 0.0), (0.5, 0.5),
                                (0.0, 0.0), (0.0, 0.5)),
        virtual_channels=_IWR6843AOP_CHANNELS,
        tx_power_dbm=10.0,
        antenna_gain_dbi_per_element=5.0,
        noise={"metric": "EINF", "einf_db": 9.0, "n_10mhz_dbm": -95.0},
        adc={"bits": 12, "real_or_complex2x_sample_rate_sps": 25.0e6,
             "complex1x_sample_rate_sps": 12.5e6, "if_bandwidth_hz": 10.0e6},
        fov_deg={"azimuth": (-60.0, 60.0), "elevation": (-60.0, 60.0)},
        angular_resolution_deg={"azimuth": 29.0, "elevation": 29.0},
        pattern_asset_key="IWR6843AOPEVM",
    ),
}
