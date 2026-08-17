# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Physical TDM slot expansion and public-cube packing helpers."""

from __future__ import annotations

from dataclasses import fields, replace

import numpy as np

from ..radar import RadarSensor
from .types import RadarCube, SensingMetadata


def tdm_expansion_required(radar: RadarSensor) -> bool:
    """Returns whether a radar needs per-transmitter physical slot sampling."""

    return bool(radar.fmcw.tdm_enabled and radar.fmcw.num_tx > 1)


def expanded_tdm_radar(radar: RadarSensor) -> RadarSensor:
    """Returns a simultaneous radar on the equivalent physical-slot timeline."""

    if not tdm_expansion_required(radar):
        return radar
    fmcw = radar.fmcw
    expanded_fmcw = replace(
        fmcw,
        num_chirps_per_frame=fmcw.num_chirps_per_frame * fmcw.num_tx,
        tdm_enabled=False,
    )
    return replace(radar, fmcw=expanded_fmcw)


def _pack_channel_samples(
    values: np.ndarray,
    *,
    num_chirps: int,
    num_tx: int,
    channel_tx_indices: np.ndarray,
) -> np.ndarray:
    """Packs expanded physical-slot values into per-channel slow time."""

    array = np.asarray(values)
    expected_slots = num_chirps * num_tx
    if array.ndim != 4 or array.shape[1] != expected_slots:
        raise ValueError(
            "expanded TDM ADC arrays must have shape "
            "[frames, num_chirps * num_tx, samples, channels]"
        )
    if array.shape[-1] != channel_tx_indices.size:
        raise ValueError("expanded TDM ADC channel count does not match hardware")
    packed = np.empty(
        (array.shape[0], num_chirps, array.shape[2], array.shape[3]),
        dtype=array.dtype,
    )
    for tx_index in range(num_tx):
        channel_indices = np.flatnonzero(channel_tx_indices == tx_index)
        if channel_indices.size:
            packed[..., channel_indices] = array[
                :, tx_index::num_tx, :, channel_indices
            ]
    return packed


def _collapse_slot_array(
    value: np.ndarray,
    *,
    num_chirps: int,
    num_tx: int,
    name: str,
) -> np.ndarray:
    """Collapses per-physical-slot diagnostics to one value per TDM cycle."""

    array = np.asarray(value)
    expected_slots = num_chirps * num_tx
    if array.ndim < 2 or array.shape[1] != expected_slots:
        return array
    grouped = array.reshape(
        (array.shape[0], num_chirps, num_tx, *array.shape[2:])
    )
    if np.issubdtype(array.dtype, np.bool_):
        return np.any(grouped, axis=2)
    if name == "runtime_per_chirp_s":
        return np.sum(grouped, axis=2)
    if "_min_" in name:
        return np.nanmin(grouped, axis=2)
    if "_max_" in name or name.endswith("_counts") or name == "path_counts":
        return np.nanmax(grouped, axis=2).astype(array.dtype, copy=False)
    if np.issubdtype(array.dtype, np.integer):
        return np.max(grouped, axis=2)
    return np.nanmean(grouped, axis=2)


def _expand_slot_array(
    value: np.ndarray,
    *,
    num_chirps: int,
    num_tx: int,
) -> np.ndarray:
    """Repeats a packed frame/chirp diagnostic over physical Tx slots."""

    array = np.asarray(value)
    if array.ndim < 2 or array.shape[1] != num_chirps:
        return array
    expanded = np.repeat(array[:, :, None, ...], num_tx, axis=2)
    return expanded.reshape(
        (array.shape[0], num_chirps * num_tx, *array.shape[2:])
    )


def _tdm_events(events: list[dict], *, num_tx: int) -> list[dict]:
    """Maps expanded physical-chirp event indices to TDM cycle/slot indices."""

    packed_events = []
    for source in events:
        event = dict(source)
        if "chirp" in event:
            physical_chirp = int(event["chirp"])
            event["chirp"] = physical_chirp // num_tx
            event["tx_slot"] = physical_chirp % num_tx
        packed_events.append(event)
    return packed_events


def virtual_channel_times(radar: RadarSensor, num_frames: int) -> np.ndarray:
    """Returns acquisition time coordinates for every virtual channel."""

    fmcw = radar.fmcw
    channel_tx = radar.hardware.virtual_tx_indices()
    result = np.empty(
        (
            int(num_frames),
            fmcw.num_chirps_per_frame,
            radar.hardware.num_virtual_channels,
        ),
        dtype=np.float64,
    )
    for frame in range(int(num_frames)):
        for chirp in range(fmcw.num_chirps_per_frame):
            for channel, tx_index in enumerate(channel_tx):
                result[frame, chirp, channel] = fmcw.tx_chirp_time(
                    frame,
                    chirp,
                    int(tx_index),
                )
    return result


def collapse_tdm_cube(cube: RadarCube, radar: RadarSensor) -> RadarCube:
    """Packs an expanded physical-slot cube into the public virtual ADC cube."""

    if not tdm_expansion_required(radar):
        return cube
    fmcw = radar.fmcw
    num_chirps = fmcw.num_chirps_per_frame
    num_tx = fmcw.num_tx
    channel_tx = radar.hardware.virtual_tx_indices()
    adc = _pack_channel_samples(
        cube.adc,
        num_chirps=num_chirps,
        num_tx=num_tx,
        channel_tx_indices=channel_tx,
    )
    physical_times = np.asarray(cube.times, dtype=np.float64)
    expected_time_shape = (adc.shape[0], num_chirps * num_tx)
    if physical_times.shape != expected_time_shape:
        raise ValueError(
            "expanded TDM times must have shape "
            f"{expected_time_shape}; received {physical_times.shape}"
        )
    times = physical_times[:, ::num_tx]
    channel_times = np.empty(
        (adc.shape[0], num_chirps, adc.shape[-1]),
        dtype=np.float64,
    )
    for channel, tx_index in enumerate(channel_tx):
        channel_times[..., channel] = physical_times[:, int(tx_index)::num_tx]
    metadata_updates: dict[str, object] = {
        "events": _tdm_events(cube.metadata.events, num_tx=num_tx),
        "virtual_channel_times_s": channel_times,
    }
    for item in fields(SensingMetadata):
        if item.name in metadata_updates:
            continue
        value = getattr(cube.metadata, item.name)
        if isinstance(value, np.ndarray):
            metadata_updates[item.name] = _collapse_slot_array(
                value,
                num_chirps=num_chirps,
                num_tx=num_tx,
                name=item.name,
            )
    metadata = replace(cube.metadata, **metadata_updates)

    components = None
    if cube.components is not None:
        components = {}
        for name, value in cube.components.items():
            array = np.asarray(value)
            if array.shape == np.asarray(cube.adc).shape:
                components[name] = _pack_channel_samples(
                    array,
                    num_chirps=num_chirps,
                    num_tx=num_tx,
                    channel_tx_indices=channel_tx,
                )
            else:
                components[name] = _collapse_slot_array(
                    array,
                    num_chirps=num_chirps,
                    num_tx=num_tx,
                    name=name,
                )
    return RadarCube(
        adc=adc,
        times=times,
        metadata=metadata,
        components=components,
    )


def expand_tdm_cube(cube: RadarCube, radar: RadarSensor) -> RadarCube:
    """Expands a packed TDM cube onto its physical transmitter-slot timeline.

    Only the channels belonging to the active transmitter are populated in
    each physical slot. This is the inverse representation used internally by
    cached coupling APIs before they synthesize new channel components.
    """

    if not tdm_expansion_required(radar):
        return cube
    adc = np.asarray(cube.adc)
    if adc.ndim != 4:
        raise ValueError(
            "packed TDM ADC must have shape "
            "[frames, chirps, samples, channels]"
        )
    fmcw = radar.fmcw
    num_chirps = fmcw.num_chirps_per_frame
    num_tx = fmcw.num_tx
    if adc.shape[1] != num_chirps:
        raise ValueError(
            "packed TDM ADC chirp count does not match FMCW configuration"
        )
    channel_tx = radar.hardware.virtual_tx_indices()
    expanded_adc = np.zeros(
        (adc.shape[0], num_chirps * num_tx, adc.shape[2], adc.shape[3]),
        dtype=adc.dtype,
    )
    for tx_index in range(num_tx):
        channel_indices = np.flatnonzero(channel_tx == tx_index)
        if channel_indices.size:
            expanded_adc[:, tx_index::num_tx, :, channel_indices] = adc[
                ..., channel_indices
            ]

    packed_times = np.asarray(cube.times, dtype=np.float64)
    if packed_times.shape != adc.shape[:2]:
        raise ValueError(
            "packed TDM times must have shape "
            f"{adc.shape[:2]}; received {packed_times.shape}"
        )
    slot_offsets = (
        np.arange(num_tx, dtype=np.float64)
        * fmcw.chirp_repetition_time
    )
    times = (packed_times[:, :, None] + slot_offsets).reshape(
        expanded_adc.shape[:2]
    )

    metadata_updates: dict[str, object] = {"virtual_channel_times_s": None}
    for item in fields(SensingMetadata):
        if item.name in metadata_updates or item.name == "events":
            continue
        value = getattr(cube.metadata, item.name)
        if isinstance(value, np.ndarray):
            metadata_updates[item.name] = _expand_slot_array(
                value,
                num_chirps=num_chirps,
                num_tx=num_tx,
            )
    events = []
    for source in cube.metadata.events:
        event = dict(source)
        if "chirp" in event:
            tx_slot = int(event.pop("tx_slot", 0))
            event["chirp"] = int(event["chirp"]) * num_tx + tx_slot
        events.append(event)
    metadata_updates["events"] = events
    metadata = replace(cube.metadata, **metadata_updates)

    components = None
    if cube.components is not None:
        components = {}
        for name, value in cube.components.items():
            array = np.asarray(value)
            if array.shape == adc.shape:
                expanded = np.zeros_like(expanded_adc)
                for tx_index in range(num_tx):
                    channel_indices = np.flatnonzero(channel_tx == tx_index)
                    if channel_indices.size:
                        expanded[:, tx_index::num_tx, :, channel_indices] = array[
                            ..., channel_indices
                        ]
                components[name] = expanded
            else:
                components[name] = _expand_slot_array(
                    array,
                    num_chirps=num_chirps,
                    num_tx=num_tx,
                )
    return RadarCube(
        adc=expanded_adc,
        times=times,
        metadata=metadata,
        components=components,
    )


def collapse_tdm_first_frame_adc(
    adc: np.ndarray,
    radar: RadarSensor,
) -> np.ndarray:
    """Packs an expanded ``[chirp, sample, channel]`` first-frame ADC cube."""

    packed = _pack_channel_samples(
        np.asarray(adc)[None, ...],
        num_chirps=radar.fmcw.num_chirps_per_frame,
        num_tx=radar.fmcw.num_tx,
        channel_tx_indices=radar.hardware.virtual_tx_indices(),
    )
    return packed[0]


def collapse_tdm_diagnostic(
    value: np.ndarray,
    radar: RadarSensor,
    *,
    name: str,
) -> np.ndarray:
    """Collapses a standalone expanded frame/chirp diagnostic array."""

    return _collapse_slot_array(
        value,
        num_chirps=radar.fmcw.num_chirps_per_frame,
        num_tx=radar.fmcw.num_tx,
        name=name,
    )


__all__ = [
    "collapse_tdm_cube",
    "collapse_tdm_diagnostic",
    "collapse_tdm_first_frame_adc",
    "expand_tdm_cube",
    "expanded_tdm_radar",
    "tdm_expansion_required",
    "virtual_channel_times",
]
