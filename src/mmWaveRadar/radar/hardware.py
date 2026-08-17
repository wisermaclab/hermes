# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Radar hardware descriptions."""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Mapping

import numpy as np


def _as_positions(value, count: int, name: str) -> np.ndarray:
    """Validates and converts an element-position table to ``[count, 3]``."""

    arr = np.asarray(value, dtype=float)
    if arr.shape != (count, 3):
        raise ValueError(f"{name} must have shape ({count}, 3)")
    return arr


def _ula(count: int, spacing: float, axis: int) -> np.ndarray:
    """Builds centered ULA element positions along one local coordinate axis."""

    ind = np.arange(count, dtype=float) - (count - 1) / 2.0
    pos = np.zeros((count, 3), dtype=float)
    pos[:, axis] = ind * spacing
    return pos


def _broadcast_channel_angles(
    num_channels: int,
    azimuth_deg,
    elevation_deg,
) -> tuple[np.ndarray | None, np.ndarray | None, tuple[int, ...]]:
    """Broadcasts angle inputs to an unambiguous channel-first shape.

    Scalars and one-dimensional arrays describe common angles (one value per
    path for a one-dimensional input). Channel-specific angles use an explicit
    array with at least two dimensions and a leading ``num_channels`` axis;
    ``[num_channels, 1]`` therefore represents one angle per channel.
    """

    values = (azimuth_deg, elevation_deg)
    arrays: list[np.ndarray | None] = []
    channel_first: list[bool] = []
    for value in values:
        if value is None:
            arrays.append(None)
            channel_first.append(False)
            continue
        arr = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError("azimuth_deg and elevation_deg must be finite")
        arrays.append(arr)
        channel_first.append(
            arr.ndim >= 2 and arr.shape[0] == int(num_channels)
        )

    active = [index for index, arr in enumerate(arrays) if arr is not None]
    if not active:
        return None, None, (int(num_channels),)

    try:
        if any(channel_first[index] for index in active):
            prepared = []
            for index in active:
                arr = arrays[index]
                if not channel_first[index]:
                    arr = arr.reshape((1, *arr.shape))
                prepared.append(arr)
            broadcast = np.broadcast_arrays(*prepared)
            output_shape = broadcast[0].shape
        else:
            common = np.broadcast_arrays(*(arrays[index] for index in active))
            path_shape = common[0].shape
            output_shape = ((int(num_channels),) if path_shape == () else
                            (int(num_channels), *path_shape))
            broadcast = [
                np.broadcast_to(arr, output_shape)
                for arr in common
            ]
    except ValueError as exc:
        raise ValueError(
            "azimuth_deg and elevation_deg must be broadcast-compatible"
        ) from exc

    normalized: list[np.ndarray | None] = [None, None]
    for index, arr in zip(active, broadcast):
        normalized[index] = arr
    return normalized[0], normalized[1], output_shape


def _virtual_channel_pairs(num_tx: int, num_rx: int,
                           order: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns Tx/Rx index arrays for the requested virtual-channel ordering."""

    if order == "tx_major":
        tx_indices = np.repeat(np.arange(num_tx, dtype=np.int64), num_rx)
        rx_indices = np.tile(np.arange(num_rx, dtype=np.int64), num_tx)
    elif order == "rx_major":
        tx_indices = np.tile(np.arange(num_tx, dtype=np.int64), num_rx)
        rx_indices = np.repeat(np.arange(num_rx, dtype=np.int64), num_tx)
    else:
        raise ValueError("virtual_channel_order must be 'tx_major' or 'rx_major'")
    return tx_indices, rx_indices


def resolve_virtual_channel_pairs(
    num_tx: int,
    num_rx: int,
    *,
    order: str = "tx_major",
    tx_indices: np.ndarray | None = None,
    rx_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the authoritative Tx/Rx pair for every virtual channel.

    Explicit index arrays describe a board-specific permutation of the full
    Cartesian Tx/Rx product. When they are omitted, ``order`` supplies the
    regular Tx-major or Rx-major fallback.
    """

    tx_count = _positive_count("num_tx", num_tx)
    rx_count = _positive_count("num_rx", num_rx)
    if order not in ("tx_major", "rx_major"):
        raise ValueError(
            "virtual_channel_order must be 'tx_major' or 'rx_major'"
        )
    if (tx_indices is None) != (rx_indices is None):
        raise ValueError(
            "virtual-channel Tx and Rx indices must be supplied together"
        )
    if tx_indices is None:
        return _virtual_channel_pairs(tx_count, rx_count, order)

    raw_tx = np.asarray(tx_indices)
    raw_rx = np.asarray(rx_indices)
    if not np.issubdtype(raw_tx.dtype, np.integer):
        raise ValueError("virtual-channel Tx indices must contain integers")
    if not np.issubdtype(raw_rx.dtype, np.integer):
        raise ValueError("virtual-channel Rx indices must contain integers")
    vc_tx = np.asarray(raw_tx, dtype=np.int64).reshape(-1)
    vc_rx = np.asarray(raw_rx, dtype=np.int64).reshape(-1)
    num_vc = tx_count * rx_count
    if vc_tx.shape != (num_vc,) or vc_rx.shape != (num_vc,):
        raise ValueError(
            "virtual-channel Tx/Rx indices must have one entry per "
            "Tx/Rx pair"
        )
    if np.any((vc_tx < 0) | (vc_tx >= tx_count)):
        raise ValueError(
            "virtual-channel Tx indices contain an out-of-range index"
        )
    if np.any((vc_rx < 0) | (vc_rx >= rx_count)):
        raise ValueError(
            "virtual-channel Rx indices contain an out-of-range index"
        )

    linear_pairs = vc_tx * rx_count + vc_rx
    if not np.array_equal(np.sort(linear_pairs), np.arange(num_vc)):
        raise ValueError(
            "virtual-channel Tx/Rx indices must be a permutation of the "
            "complete Cartesian Tx/Rx product"
        )
    return vc_tx, vc_rx


def _selection_indices(name: str, value, *, upper_bound: int) -> np.ndarray:
    """Validates a non-empty, unique integer channel selection."""

    raw = np.asarray(value)
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name} must contain integer indices")
    indices = np.asarray(raw, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        raise ValueError(f"{name} must select at least one element")
    if np.any((indices < 0) | (indices >= upper_bound)):
        raise ValueError(f"{name} contains an out-of-range index")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{name} must not contain duplicate indices")
    return indices


def _positive_count(name: str, value) -> int:
    """Validates a positive integer array dimension."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        count = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if count <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(count)


@dataclass(frozen=True)
class RadarHardware:
    r"""
    Physical Tx/Rx element layout for a radar.

    Positions are expressed in meters in Sionna's radio-device local frame,
    where the device x-axis is boresight.
    """

    name: str
    tx_positions: np.ndarray
    rx_positions: np.ndarray
    virtual_channel_order: str = "tx_major"
    virtual_channel_labels: tuple[str, ...] | None = None
    virtual_channel_positions_lambda: np.ndarray | None = None
    virtual_channel_tx_indices: np.ndarray | None = None
    virtual_channel_rx_indices: np.ndarray | None = None
    virtual_channel_phase_signs: np.ndarray | None = None
    scalar_antenna_gain_dbi: float | None = None
    antenna_pattern: object | None = None
    board_metadata: Mapping[str, object] | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        tx = np.array(self.tx_positions, dtype=float, copy=True)
        rx = np.array(self.rx_positions, dtype=float, copy=True)
        if tx.ndim != 2 or tx.shape[1] != 3:
            raise ValueError("tx_positions must have shape (num_tx, 3)")
        if rx.ndim != 2 or rx.shape[1] != 3:
            raise ValueError("rx_positions must have shape (num_rx, 3)")
        if tx.shape[0] == 0 or rx.shape[0] == 0:
            raise ValueError("tx_positions and rx_positions must not be empty")
        if not np.all(np.isfinite(tx)) or not np.all(np.isfinite(rx)):
            raise ValueError("tx_positions and rx_positions must be finite")
        if self.virtual_channel_order not in ("tx_major", "rx_major"):
            raise ValueError("virtual_channel_order must be 'tx_major' or"
                             " 'rx_major'")
        num_vc = tx.shape[0] * rx.shape[0]

        labels = None
        if self.virtual_channel_labels is not None:
            labels = tuple(str(label) for label in self.virtual_channel_labels)
            if len(labels) != num_vc:
                raise ValueError("virtual_channel_labels must have one entry"
                                 " per virtual channel")
            if any(not label.strip() for label in labels):
                raise ValueError("virtual_channel_labels must be non-empty")
            if len(set(labels)) != len(labels):
                raise ValueError("virtual_channel_labels must be unique")

        positions_lambda = None
        if self.virtual_channel_positions_lambda is not None:
            positions_lambda = np.array(
                self.virtual_channel_positions_lambda,
                dtype=float,
                copy=True,
            )
            if positions_lambda.shape != (num_vc, 2):
                raise ValueError("virtual_channel_positions_lambda must have"
                                 " shape (num_virtual_channels, 2)")
            if not np.all(np.isfinite(positions_lambda)):
                raise ValueError(
                    "virtual_channel_positions_lambda must be finite"
                )

        tx_indices = None
        rx_indices = None
        if self.virtual_channel_tx_indices is not None:
            raw_tx_indices = np.asarray(self.virtual_channel_tx_indices)
            if not np.issubdtype(raw_tx_indices.dtype, np.integer):
                raise ValueError(
                    "virtual_channel_tx_indices must contain integers"
                )
            tx_indices = np.asarray(raw_tx_indices, dtype=np.int64)
            if tx_indices.shape != (num_vc,):
                raise ValueError("virtual_channel_tx_indices must have shape"
                                 " (num_virtual_channels,)")
            if np.any((tx_indices < 0) | (tx_indices >= tx.shape[0])):
                raise ValueError("virtual_channel_tx_indices contains an"
                                 " out-of-range Tx index")
        if self.virtual_channel_rx_indices is not None:
            raw_rx_indices = np.asarray(self.virtual_channel_rx_indices)
            if not np.issubdtype(raw_rx_indices.dtype, np.integer):
                raise ValueError(
                    "virtual_channel_rx_indices must contain integers"
                )
            rx_indices = np.asarray(raw_rx_indices, dtype=np.int64)
            if rx_indices.shape != (num_vc,):
                raise ValueError("virtual_channel_rx_indices must have shape"
                                 " (num_virtual_channels,)")
            if np.any((rx_indices < 0) | (rx_indices >= rx.shape[0])):
                raise ValueError("virtual_channel_rx_indices contains an"
                                 " out-of-range Rx index")
        if (tx_indices is None) != (rx_indices is None):
            raise ValueError("virtual_channel_tx_indices and"
                             " virtual_channel_rx_indices must be supplied"
                             " together")
        if tx_indices is not None:
            resolve_virtual_channel_pairs(
                tx.shape[0],
                rx.shape[0],
                order=self.virtual_channel_order,
                tx_indices=tx_indices,
                rx_indices=rx_indices,
            )

        phase_signs = None
        if self.virtual_channel_phase_signs is not None:
            phase_signs = np.asarray(self.virtual_channel_phase_signs,
                                     dtype=np.complex128)
            if phase_signs.shape != (num_vc,):
                raise ValueError("virtual_channel_phase_signs must have shape"
                                 " (num_virtual_channels,)")
            if not np.all(
                np.isfinite(phase_signs.real) & np.isfinite(phase_signs.imag)
            ):
                raise ValueError("virtual_channel_phase_signs must be finite")

        scalar_gain = self.scalar_antenna_gain_dbi
        if scalar_gain is not None:
            try:
                scalar_gain = float(scalar_gain)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "scalar_antenna_gain_dbi must be a finite scalar or None"
                ) from exc
            if not np.isfinite(scalar_gain):
                raise ValueError(
                    "scalar_antenna_gain_dbi must be a finite scalar or None"
                )

        pattern_channels = getattr(
            self.antenna_pattern,
            "num_virtual_channels",
            num_vc,
        )
        pattern_channel_count = _positive_count(
            "antenna_pattern num_virtual_channels",
            pattern_channels,
        )
        if pattern_channel_count != num_vc:
            raise ValueError(
                "antenna_pattern num_virtual_channels does not match hardware"
            )
        if self.board_metadata is not None and not isinstance(
            self.board_metadata,
            Mapping,
        ):
            raise ValueError("board_metadata must be a mapping or None")

        object.__setattr__(self, "tx_positions", tx)
        object.__setattr__(self, "rx_positions", rx)
        object.__setattr__(self, "virtual_channel_labels", labels)
        object.__setattr__(self, "virtual_channel_positions_lambda",
                           positions_lambda)
        object.__setattr__(self, "virtual_channel_tx_indices", tx_indices)
        object.__setattr__(self, "virtual_channel_rx_indices", rx_indices)
        object.__setattr__(self, "virtual_channel_phase_signs", phase_signs)
        object.__setattr__(self, "scalar_antenna_gain_dbi", scalar_gain)
        if self.board_metadata is not None:
            object.__setattr__(self, "board_metadata", dict(self.board_metadata))

    @property
    def num_tx(self) -> int:
        """Number of transmit elements."""
        return int(self.tx_positions.shape[0])

    @property
    def num_rx(self) -> int:
        """Number of receive elements."""
        return int(self.rx_positions.shape[0])

    @property
    def num_virtual_channels(self) -> int:
        """Number of Tx/Rx virtual channels."""
        return self.num_tx * self.num_rx

    def virtual_tx_indices(self) -> np.ndarray:
        """Returns zero-based Tx indices for each virtual channel."""

        tx_indices, _ = resolve_virtual_channel_pairs(
            self.num_tx,
            self.num_rx,
            order=self.virtual_channel_order,
            tx_indices=self.virtual_channel_tx_indices,
            rx_indices=self.virtual_channel_rx_indices,
        )
        return tx_indices.copy()

    def virtual_rx_indices(self) -> np.ndarray:
        """Returns zero-based Rx indices for each virtual channel."""

        _, rx_indices = resolve_virtual_channel_pairs(
            self.num_tx,
            self.num_rx,
            order=self.virtual_channel_order,
            tx_indices=self.virtual_channel_tx_indices,
            rx_indices=self.virtual_channel_rx_indices,
        )
        return rx_indices.copy()

    def channel_gain(self, *, azimuth_deg=None, elevation_deg=None,
                     frequency_hz: float | None = None,
                     include_phase: bool = True,
                     include_antenna_gain: bool = True,
                     include_pattern: bool = True) -> np.ndarray:
        """
        Returns complex voltage multipliers for virtual channels.

        ``azimuth_deg`` and ``elevation_deg`` can be scalars or common path
        arrays. A one-dimensional input always means one value per path. For
        channel-specific values, pass a rank-two-or-higher channel-first array
        (for example, ``[num_virtual_channels, 1]`` for one value per channel).
        """

        shape = self._channel_gain_shape(azimuth_deg, elevation_deg)
        gain_db = np.zeros(shape, dtype=float)
        if include_antenna_gain and self.scalar_antenna_gain_dbi is not None:
            gain_db += float(self.scalar_antenna_gain_dbi)
        if include_pattern and self.antenna_pattern is not None:
            gain_db += self.antenna_pattern.loss_db(
                azimuth_deg=azimuth_deg,
                elevation_deg=elevation_deg,
                frequency_hz=frequency_hz,
            )

        gain = np.power(10.0, gain_db / 20.0).astype(np.complex128)
        if include_phase and self.virtual_channel_phase_signs is not None:
            signs = self.virtual_channel_phase_signs.astype(np.complex128)
            signs = signs.reshape((self.num_virtual_channels,)
                                  + (1,) * (gain.ndim - 1))
            gain *= signs
        return gain

    def apply_channel_gain(self, coefficients: np.ndarray, *,
                           azimuth_deg=None,
                           elevation_deg=None,
                           frequency_hz: float | None = None) -> np.ndarray:
        """Applies :meth:`channel_gain` to a coefficient matrix."""

        coeffs = np.asarray(coefficients)
        if coeffs.ndim == 0:
            raise ValueError("coefficients must include a virtual-channel axis")
        if coeffs.shape[0] != self.num_virtual_channels:
            raise ValueError("coefficients first axis must match the number of"
                             " virtual channels")
        gains = self.channel_gain(
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            frequency_hz=frequency_hz,
        )
        if gains.shape == (self.num_virtual_channels,) and coeffs.ndim > 1:
            gains = gains.reshape((self.num_virtual_channels,)
                                  + (1,) * (coeffs.ndim - 1))
        return coeffs * gains

    def subset_tx(self, tx_indices) -> "RadarHardware":
        """Returns a copy containing only the selected Tx elements."""

        indices = _selection_indices(
            "tx_indices",
            tx_indices,
            upper_bound=self.num_tx,
        )

        old_vc_tx = self.virtual_tx_indices()
        old_vc_rx = self.virtual_rx_indices()
        selected = np.flatnonzero(np.isin(old_vc_tx, indices))
        tx_remap = {int(old): int(new) for new, old in enumerate(indices)}
        pattern = self.antenna_pattern
        if pattern is not None and hasattr(pattern, "subset"):
            pattern = pattern.subset(selected)

        def _subset_array(value):
            """Slices an optional virtual-channel array by selected channels."""

            return None if value is None else value[selected].copy()

        labels = None
        if self.virtual_channel_labels is not None:
            labels = tuple(self.virtual_channel_labels[int(i)]
                           for i in selected)

        return RadarHardware(
            name=self.name,
            tx_positions=self.tx_positions[indices],
            rx_positions=self.rx_positions,
            virtual_channel_order=self.virtual_channel_order,
            virtual_channel_labels=labels,
            virtual_channel_positions_lambda=_subset_array(
                self.virtual_channel_positions_lambda),
            virtual_channel_tx_indices=np.asarray(
                [tx_remap[int(old_vc_tx[i])] for i in selected],
                dtype=np.int64,
            ),
            virtual_channel_rx_indices=old_vc_rx[selected],
            virtual_channel_phase_signs=_subset_array(
                self.virtual_channel_phase_signs),
            scalar_antenna_gain_dbi=self.scalar_antenna_gain_dbi,
            antenna_pattern=pattern,
            board_metadata=self.board_metadata,
        )

    def subset_rx(self, rx_indices) -> "RadarHardware":
        """Returns a copy containing only the selected Rx elements."""

        indices = _selection_indices(
            "rx_indices",
            rx_indices,
            upper_bound=self.num_rx,
        )

        old_vc_tx = self.virtual_tx_indices()
        old_vc_rx = self.virtual_rx_indices()
        selected = np.flatnonzero(np.isin(old_vc_rx, indices))
        rx_remap = {int(old): int(new) for new, old in enumerate(indices)}
        pattern = self.antenna_pattern
        if pattern is not None and hasattr(pattern, "subset"):
            pattern = pattern.subset(selected)

        def _subset_array(value):
            """Slices an optional virtual-channel array by selected channels."""

            return None if value is None else value[selected].copy()

        labels = None
        if self.virtual_channel_labels is not None:
            labels = tuple(self.virtual_channel_labels[int(i)]
                           for i in selected)

        return RadarHardware(
            name=self.name,
            tx_positions=self.tx_positions,
            rx_positions=self.rx_positions[indices],
            virtual_channel_order=self.virtual_channel_order,
            virtual_channel_labels=labels,
            virtual_channel_positions_lambda=_subset_array(
                self.virtual_channel_positions_lambda),
            virtual_channel_tx_indices=old_vc_tx[selected],
            virtual_channel_rx_indices=np.asarray(
                [rx_remap[int(old_vc_rx[i])] for i in selected],
                dtype=np.int64,
            ),
            virtual_channel_phase_signs=_subset_array(
                self.virtual_channel_phase_signs),
            scalar_antenna_gain_dbi=self.scalar_antenna_gain_dbi,
            antenna_pattern=pattern,
            board_metadata=self.board_metadata,
        )

    def _channel_gain_shape(self, azimuth_deg, elevation_deg) -> tuple[int, ...]:
        """Infers the broadcast output shape for per-channel antenna gains."""

        _, _, shape = _broadcast_channel_angles(
            self.num_virtual_channels,
            azimuth_deg,
            elevation_deg,
        )
        return shape

    @classmethod
    def from_xwr68xx(cls, model: str = "IWR6843AOP",
                     spacing: float | None = None) -> "RadarHardware":
        r"""
        Approximate xWR68xx/IWR6843-family element layout.

        The default spacing is half a wavelength at about 77 GHz. Element
        positions are intended for simulation convenience; exact board
        calibration should be supplied explicitly when needed.
        """
        key = str(model).upper()
        d = 0.001948 if spacing is None else float(spacing)
        if not np.isfinite(d) or d <= 0.0:
            raise ValueError("spacing must be finite and positive")
        if key in ("IWR6843AOP", "AWR6843AOP", "XWR68XX_AOP"):
            lam = 2.0 * d
            # Sionna's planar array lives in the local y-z plane.
            rx = np.array([
                [0.0, 0.0, 0.0],
                [0.0, -lam / 2.0, 0.0],
                [0.0, 0.0, -lam / 2.0],
                [0.0, -lam / 2.0, -lam / 2.0],
            ], dtype=float)
            tx = np.array([
                [0.0, 0.0, -lam],
                [0.0, lam, 0.0],
                [0.0, lam, -lam],
            ], dtype=float)
        elif key in ("IWR6843ISK", "XWR68XX", "XWR68XX_ISK"):
            rx = np.array([
                [0.0, 0.0 * d, 0.0],
                [0.0, 1.0 * d, 0.0],
                [0.0, 2.0 * d, 0.0],
                [0.0, 3.0 * d, 0.0],
            ], dtype=float)
            tx = np.array([
                [0.0, 0.0 * d, 0.0],
                [0.0, 1.0 * d, 1.0 * d],
                [0.0, 2.0 * d, 0.0],
            ], dtype=float)
        else:
            tx = _ula(3, d, axis=1)
            rx = _ula(4, d, axis=2)
        return cls(name=key, tx_positions=tx, rx_positions=rx)

    @classmethod
    def from_ti_board(cls, model: str,
                      pattern_mode: str = "cosine30",
                      cosine_half_power_angle_deg: float = 30.0
                      ) -> "RadarHardware":
        """Builds hardware from the local TI board digital-twin catalog."""

        from .ti import get_ti_board_spec  # pylint: disable=import-outside-toplevel

        spec = get_ti_board_spec(model)
        return cls(
            name=spec.key,
            tx_positions=spec.tx_positions_m(),
            rx_positions=spec.rx_positions_m(),
            virtual_channel_order="tx_major",
            virtual_channel_labels=spec.virtual_channel_labels(),
            virtual_channel_positions_lambda=(
                spec.virtual_channel_positions_lambda()),
            virtual_channel_tx_indices=spec.virtual_channel_tx_indices(),
            virtual_channel_rx_indices=spec.virtual_channel_rx_indices(),
            virtual_channel_phase_signs=spec.channel_phase_signs(),
            scalar_antenna_gain_dbi=spec.combined_antenna_gain_dbi,
            antenna_pattern=spec.pattern(
                pattern_mode,
                cosine_half_power_angle_deg=cosine_half_power_angle_deg,
            ),
            board_metadata=spec.metadata(),
        )

    @classmethod
    def from_virtual_ura(
        cls,
        *,
        rows: int,
        cols: int | None = None,
        wavelength: float,
        spacing_lambda: float = 0.5,
        spacing_y_lambda: float | None = None,
        spacing_z_lambda: float | None = None,
        centered: bool = True,
        name: str | None = None,
        scalar_antenna_gain_dbi: float | None = 0.0,
        antenna_pattern: object | None = None,
    ) -> "RadarHardware":
        """
        Builds a synthetic virtual uniform rectangular array.

        This is a direct virtual-aperture model with one phase-center Tx and
        one synthetic Rx per virtual channel. It assumes concurrent
        transmission/no TDM effects and does not describe a physical MIMO
        Tx/Rx decomposition. The local ``y`` axis is horizontal and ``z`` is
        vertical. ``spacing_y_lambda`` and ``spacing_z_lambda`` specify the
        element spacing on those axes in wavelengths. The legacy
        ``spacing_lambda`` value supplies both axes when an axis-specific
        value is omitted.
        """

        row_count = _positive_count("rows", rows)
        col_count = (
            row_count if cols is None else _positive_count("cols", cols)
        )
        wavelength_m = float(wavelength)
        spacing = float(spacing_lambda)
        if not np.isfinite(wavelength_m) or wavelength_m <= 0.0:
            raise ValueError("wavelength must be finite and positive")
        if not np.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("spacing_lambda must be finite and positive")
        spacing_y = (
            spacing
            if spacing_y_lambda is None
            else float(spacing_y_lambda)
        )
        spacing_z = (
            spacing
            if spacing_z_lambda is None
            else float(spacing_z_lambda)
        )
        if not np.isfinite(spacing_y) or spacing_y <= 0.0:
            raise ValueError(
                "spacing_y_lambda must be finite and positive"
            )
        if not np.isfinite(spacing_z) or spacing_z <= 0.0:
            raise ValueError(
                "spacing_z_lambda must be finite and positive"
            )
        if not isinstance(centered, (bool, np.bool_)):
            raise ValueError("centered must be a boolean")

        y_index = np.arange(col_count, dtype=float)
        z_index = np.arange(row_count, dtype=float)
        if centered:
            y_index -= 0.5 * (col_count - 1)
            z_index -= 0.5 * (row_count - 1)
        y_lambda = y_index * spacing_y
        z_lambda = z_index * spacing_z
        yy_lambda, zz_lambda = np.meshgrid(y_lambda, z_lambda, indexing="xy")
        virtual_positions_lambda = np.column_stack([
            yy_lambda.reshape(-1),
            zz_lambda.reshape(-1),
        ])

        tx_positions = np.zeros((1, 3), dtype=float)
        rx_positions = np.zeros((virtual_positions_lambda.shape[0], 3),
                                dtype=float)
        rx_positions[:, 1:] = virtual_positions_lambda * wavelength_m

        labels = tuple(
            f"VURA r{row + 1} c{col + 1}"
            for row in range(row_count)
            for col in range(col_count)
        )
        metadata = {
            "kind": "virtual_ura",
            "rows": row_count,
            "cols": col_count,
            "spacing_lambda": (
                spacing_y if np.isclose(spacing_y, spacing_z) else None
            ),
            "spacing_y_lambda": spacing_y,
            "spacing_z_lambda": spacing_z,
            "centered": bool(centered),
            "aperture_m_y": (
                max(col_count - 1, 0) * spacing_y * wavelength_m
            ),
            "aperture_m_z": (
                max(row_count - 1, 0) * spacing_z * wavelength_m
            ),
            "wavelength_m": wavelength_m,
            "transmission": "concurrent",
        }
        hardware_name = (
            f"VIRTUAL_URA{row_count}X{col_count}"
            if name is None
            else str(name)
        )
        return cls(
            name=hardware_name,
            tx_positions=tx_positions,
            rx_positions=rx_positions,
            virtual_channel_order="tx_major",
            virtual_channel_labels=labels,
            virtual_channel_positions_lambda=virtual_positions_lambda,
            scalar_antenna_gain_dbi=scalar_antenna_gain_dbi,
            antenna_pattern=antenna_pattern,
            board_metadata=metadata,
        )

    @classmethod
    def from_positions(cls, tx_positions, rx_positions,
                       name: str = "custom",
                       virtual_channel_order: str = "tx_major"):
        """Builds hardware from explicit Tx/Rx element positions."""
        tx = np.asarray(tx_positions, dtype=float)
        rx = np.asarray(rx_positions, dtype=float)
        return cls(name=name, tx_positions=tx, rx_positions=rx,
                   virtual_channel_order=virtual_channel_order)
