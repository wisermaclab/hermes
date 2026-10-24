# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Bundle slicing and physical TX/RX channel selection."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
from bundle_prepare.contract import Bundle as BenchmarkClip
from . import configuration as _configuration


@dataclass(frozen=True)
class SubarraySelection:
    """Resolved virtual-channel subset and the matching TX/RX metadata."""

    channel_indices: np.ndarray
    tx_indices_1based: tuple[int, ...]
    rx_indices_1based: tuple[int, ...]
    channel_tx_indices_1based: np.ndarray
    channel_rx_indices_1based: np.ndarray
    virtual_positions_lambda: np.ndarray
    full_channel_count: int

    @property
    def active(self) -> bool:
        """Whether the run uses fewer channels than the full sensor."""

        return self.channel_indices.size != int(self.full_channel_count)

    @property
    def channel_count(self) -> int:
        """Number of selected virtual channels."""

        return int(self.channel_indices.size)


def slice_clip(clip: BenchmarkClip, max_frames: int | None) -> BenchmarkClip:
    if max_frames is None or max_frames >= clip.num_frames:
        return clip
    count = max(1, int(max_frames))
    background = clip.background_adc
    if background is not None and background.shape == clip.real_adc.shape:
        background = background[:count]
    return replace(
        clip,
        real_adc=clip.real_adc[:count],
        frames=clip.frames[:count],
        times=None if clip.times is None else clip.times[:count],
        background_adc=background,
    )


def subarray_selection(
    clip: BenchmarkClip,
    config: _configuration.BenchmarkRunConfig,
    *,
    full_channels: int,
) -> SubarraySelection:
    sensor = clip.sensor()
    channel_tx, channel_rx = channel_tx_rx_indices_1based(
        clip,
        sensor,
        full_channels=full_channels,
    )
    positions = channel_virtual_positions_lambda(sensor, channel_tx, channel_rx)
    # Bundles label antennas from 1 internally; selections are zero-based.
    requested_tx = tuple(
        index + 1
        for index in requested_indices_or_all(
            config.selected_tx_indices,
            channel_tx - 1,
            name="selected TX",
        )
    )
    requested_rx = tuple(
        index + 1
        for index in requested_indices_or_all(
            config.selected_rx_indices,
            channel_rx - 1,
            name="selected RX",
        )
    )
    mask = np.isin(channel_tx, np.asarray(requested_tx, dtype=np.int64)) & np.isin(
        channel_rx, np.asarray(requested_rx, dtype=np.int64)
    )
    channel_indices = np.flatnonzero(mask)
    if channel_indices.size == 0:
        raise ValueError(
            "Selected TX/RX indices produced no virtual channels "
            f"(tx={[index - 1 for index in requested_tx]}, "
            f"rx={[index - 1 for index in requested_rx]})"
        )
    return SubarraySelection(
        channel_indices=channel_indices.astype(np.int64),
        tx_indices_1based=tuple(int(v) for v in requested_tx),
        rx_indices_1based=tuple(int(v) for v in requested_rx),
        channel_tx_indices_1based=channel_tx[channel_indices].astype(np.int64),
        channel_rx_indices_1based=channel_rx[channel_indices].astype(np.int64),
        virtual_positions_lambda=positions[channel_indices].astype(float),
        full_channel_count=int(full_channels),
    )


def channel_tx_rx_indices_1based(
    clip: BenchmarkClip,
    sensor,
    *,
    full_channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    # to_sensor already applies legacy slot ordering and stored RX subsets.
    meta = getattr(sensor.hardware, "board_metadata", None) or {}
    num_tx = int(np.max(sensor.hardware.virtual_tx_indices())) + 1
    tx = (
        np.asarray(meta.get("physical_tx_indices", range(num_tx)))[
            sensor.hardware.virtual_tx_indices()
        ]
        + 1
    )
    rx = (
        np.asarray(meta.get("physical_rx_indices", range(sensor.hardware.num_rx)))[
            sensor.hardware.virtual_rx_indices()
        ]
        + 1
    )
    if tx.shape != (int(full_channels),) or rx.shape != (int(full_channels),):
        raise ValueError(
            "Could not derive TX/RX labels for ADC channel axis "
            f"({tx.size}, {rx.size}) != {full_channels}"
        )
    return tx.astype(np.int64), rx.astype(np.int64)


def channel_virtual_positions_lambda(
    sensor,
    channel_tx_1based: np.ndarray,
    channel_rx_1based: np.ndarray,
) -> np.ndarray:
    hardware = sensor.hardware
    meta = hardware.board_metadata or {}
    tx_labels = np.asarray(meta.get("physical_tx_indices", range(hardware.num_tx)))
    rx_labels = np.asarray(meta.get("physical_rx_indices", range(hardware.num_rx)))
    tx = tx_labels[hardware.virtual_tx_indices()] + 1
    rx = rx_labels[hardware.virtual_rx_indices()] + 1
    positions = hardware.virtual_channel_positions_lambda
    if positions is None:
        positions = (
            hardware.tx_positions[hardware.virtual_tx_indices(), 1:]
            + hardware.rx_positions[hardware.virtual_rx_indices(), 1:]
        ) / sensor.fmcw.wavelength
    lookup = {(int(t), int(r)): pos for t, r, pos in zip(tx, rx, positions)}
    return np.asarray(
        [lookup[(int(t), int(r))] for t, r in zip(channel_tx_1based, channel_rx_1based)],
        dtype=float,
    )


def requested_indices_or_all(
    requested: tuple[int, ...] | None,
    available: np.ndarray,
    *,
    name: str,
) -> tuple[int, ...]:
    available_unique = ordered_unique(int(v) for v in np.asarray(available).reshape(-1))
    if requested is None:
        return available_unique
    values = tuple(int(v) for v in requested)
    if not values:
        raise ValueError(f"{name} selection must not be empty")
    missing = sorted(set(values) - set(available_unique))
    if missing:
        raise ValueError(
            f"{name} selection contains unavailable indices {missing}; "
            f"available indices are {list(available_unique)}"
        )
    return values


def ordered_unique(values) -> tuple[int, ...]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        ivalue = int(value)
        if ivalue not in seen:
            seen.add(ivalue)
            out.append(ivalue)
    return tuple(out)


def select_adc_channels(adc: np.ndarray, selection: SubarraySelection) -> np.ndarray:
    x = np.asarray(adc)
    if not selection.active:
        return x
    return x[..., selection.channel_indices]


def select_or_validate_adc_channels(
    adc: np.ndarray,
    selection: SubarraySelection,
    *,
    full_shape: tuple[int, ...],
    selected_shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    x = np.asarray(adc)
    if x.shape == selected_shape:
        return x
    if selection.active and x.shape == full_shape:
        return select_adc_channels(x, selection)
    raise ValueError(
        f"{label} ADC shape must match either selected real ADC shape "
        f"{selected_shape} or full source ADC shape {full_shape}; got {x.shape}"
    )


def select_background_channels(
    background_adc: np.ndarray | None,
    selection: SubarraySelection,
    *,
    full_shape: tuple[int, ...],
) -> np.ndarray | None:
    if background_adc is None or not selection.active:
        return background_adc
    background = np.asarray(background_adc)
    selected_channels = selection.channel_count
    full_channels = int(full_shape[-1])
    if background.shape[-1] == selected_channels:
        return background
    if background.shape[-1] != full_channels:
        return background
    return background[..., selection.channel_indices]


def subarray_metadata(selection: SubarraySelection) -> dict:
    return {
        "enabled": bool(selection.active),
        "full_channel_count": int(selection.full_channel_count),
        "channel_count": int(selection.channel_count),
        "selected_tx_indices": [index - 1 for index in selection.tx_indices_1based],
        "selected_rx_indices": [index - 1 for index in selection.rx_indices_1based],
        "channel_indices": selection.channel_indices.astype(int).tolist(),
        "channel_tx_indices": ((selection.channel_tx_indices_1based - 1).astype(int).tolist()),
        "channel_rx_indices": ((selection.channel_rx_indices_1based - 1).astype(int).tolist()),
        "virtual_positions_lambda": selection.virtual_positions_lambda.tolist(),
        "channel_order": "preserved_from_input_adc_axis",
    }


def write_subarray_adc(
    path: Path,
    real_adc: np.ndarray,
    sim_adcs: dict[str, np.ndarray],
    selection: SubarraySelection,
) -> None:
    payload = {
        "real_adc": np.asarray(real_adc),
        "channel_indices": selection.channel_indices.astype(np.int64),
        "channel_tx_indices": ((selection.channel_tx_indices_1based - 1).astype(np.int64)),
        "channel_rx_indices": ((selection.channel_rx_indices_1based - 1).astype(np.int64)),
        "virtual_positions_lambda": selection.virtual_positions_lambda.astype(float),
    }
    for label, adc in sim_adcs.items():
        payload[f"{label}_adc"] = np.asarray(adc)
    np.savez_compressed(path, **payload)


def index_tuple_token(values: tuple[int, ...] | None) -> str:
    if values is None:
        return ""
    return ",".join(str(int(v)) for v in values)
