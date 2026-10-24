# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Dataset-neutral loading and internal consistency for HERMES bundles."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping
import warnings

import numpy as np

from mmWaveRadar.values import boolean
from mmWaveRadar.naming import validate_tx_schedule
from mmWaveRadar.radar import (
    FMCWConfig,
    RadarHardware,
    RadarSensor,
    get_ti_board_spec,
)

from .paths import portable_relative_path, safe_bundle_file
from .safe_npz import DEFAULT_NPZ_LIMITS, preflight_npz, require_array

HERMES_BUNDLE_PROFILE = "hermes"
_MAX_BUNDLE_JSON_BYTES = 8 * 1024**2
_MAX_BUNDLE_OBJ_BYTES = 64 * 1024**2
_MAX_BUNDLE_ADC_ARTIFACTS = 64
ADC_ARTIFACT_ORIGINS = frozenset({"measurement", "simulation", "background", "processed"})
SUPPORTED_FEATURES = frozenset(
    {"custom_array", "explicit_tdm_slots", "channel_map", "target_material"}
)


class BundleValidationError(ValueError):
    """A semantic failure with a stable machine-readable location and code."""

    def __init__(self, message, *, file="bundle.json", field="", code="bundle.integrity"):
        super().__init__(message)
        self.file, self.field, self.code = file, field, code


def make_bundle_descriptor(
    *,
    primary_origin: str,
    primary_path: str = "radar_adc.npz",
    motion_parameters: str | None = None,
    background_path: str | None = None,
    environment: str | None = None,
) -> dict[str, Any]:
    """Build the common descriptor used by straightforward bundle producers."""

    origin = str(primary_origin)
    if origin not in {"measurement", "simulation"}:
        raise ValueError("primary_origin must be measurement or simulation")
    descriptor: dict[str, Any] = {
        "schema_version": 1,
        "profile": HERMES_BUNDLE_PROFILE,
        "primary_adc": "primary",
        "adcs": {
            "primary": {
                "path": str(primary_path),
                "origin": origin,
            }
        },
    }
    if background_path is not None:
        descriptor["adcs"]["background"] = {
            "path": str(background_path),
            "origin": "background",
        }
    if motion_parameters is not None:
        descriptor["motion"] = {"parameters": str(motion_parameters)}
    if environment is not None:
        descriptor["environment"] = str(environment)
    return descriptor


def read_json(path: Path) -> Any:
    size = path.stat().st_size
    if size > _MAX_BUNDLE_JSON_BYTES:
        raise ValueError(
            f"bundle JSON file is {size:,} bytes; the limit is "
            f"{_MAX_BUNDLE_JSON_BYTES:,} bytes: {path}"
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _first(mapping: Mapping[str, Any], *names: str, default=None):
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _boolean(value: Any, *, name: str) -> bool:
    """Normalize a JSON-style boolean without accepting truthy substitutes."""

    return boolean(name, value)


def _tx_schedule(data: Mapping[str, Any], *, name: str) -> str | None:
    """Read ``tx_schedule``, or the ``tdm_enabled`` flag of older bundles."""

    if data.get("tx_schedule") is not None:
        try:
            return validate_tx_schedule(data["tx_schedule"], name=name)
        except ValueError as exc:
            raise ValueError(str(exc)) from None
    if data.get("tdm_enabled") is not None:
        tdm = _boolean(data["tdm_enabled"], name=f"{name} tdm_enabled")
        return "tdm" if tdm else "simultaneous"
    return None


def _hardware_with_virtual_channel_order(
    hardware: RadarHardware,
    order: str,
) -> RadarHardware:
    """Return hardware whose per-channel metadata follows ``order``."""

    if order not in ("tx_major", "rx_major"):
        raise ValueError("sensor.json virtual_channel_order must be 'tx_major' or 'rx_major'")
    if order == hardware.virtual_channel_order:
        return hardware

    if order == "tx_major":
        desired_tx = np.repeat(
            np.arange(hardware.num_tx, dtype=np.int64),
            hardware.num_rx,
        )
        desired_rx = np.tile(
            np.arange(hardware.num_rx, dtype=np.int64),
            hardware.num_tx,
        )
    else:
        desired_tx = np.tile(
            np.arange(hardware.num_tx, dtype=np.int64),
            hardware.num_rx,
        )
        desired_rx = np.repeat(
            np.arange(hardware.num_rx, dtype=np.int64),
            hardware.num_tx,
        )
    current_pairs = {
        (int(tx_index), int(rx_index)): index
        for index, (tx_index, rx_index) in enumerate(
            zip(
                hardware.virtual_tx_indices(),
                hardware.virtual_rx_indices(),
            )
        )
    }
    permutation = np.asarray(
        [
            current_pairs[(int(tx_index), int(rx_index))]
            for tx_index, rx_index in zip(desired_tx, desired_rx)
        ],
        dtype=np.int64,
    )

    def reordered(value):
        return None if value is None else np.asarray(value)[permutation].copy()

    pattern = hardware.antenna_pattern
    if pattern is not None and hasattr(pattern, "subset"):
        pattern = pattern.subset(permutation)
    labels = None
    if hardware.virtual_channel_labels is not None:
        labels = tuple(hardware.virtual_channel_labels[int(index)] for index in permutation)
    metadata = None if hardware.board_metadata is None else dict(hardware.board_metadata)
    if metadata is not None:
        metadata["virtual_channel_order"] = order
    return RadarHardware(
        name=hardware.name,
        tx_positions=hardware.tx_positions,
        rx_positions=hardware.rx_positions,
        virtual_channel_order=order,
        virtual_channel_labels=labels,
        virtual_channel_positions_lambda=reordered(hardware.virtual_channel_positions_lambda),
        virtual_channel_tx_indices=desired_tx,
        virtual_channel_rx_indices=desired_rx,
        virtual_channel_phase_signs=reordered(hardware.virtual_channel_phase_signs),
        scalar_antenna_gain_dbi=hardware.scalar_antenna_gain_dbi,
        antenna_pattern=pattern,
        board_metadata=metadata,
    )


def _vec3(value: Any, *, name: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain exactly three finite values")
    return (float(array[0]), float(array[1]), float(array[2]))


_MISSING = object()


def _required_fmcw_value(
    data: Mapping[str, Any],
    field: str,
    *names: str,
    default: Any = _MISSING,
):
    value = _first(data, *names, default=None)
    if value is not None:
        return value
    if default is not _MISSING:
        return default
    keys = ", ".join(names)
    raise ValueError(f"FMCW profile missing {field}; expected one of: {keys}")


def _ti_board_defaults(board_model: str | None):
    if board_model is None:
        return None
    try:
        return get_ti_board_spec(board_model)
    except KeyError:
        return None


def _fmcw_from_mapping(
    data: Mapping[str, Any],
    *,
    board_model: str | None = None,
) -> FMCWConfig:
    """Build an FMCW config from plain, unit-suffixed, or TI profile keys."""

    board_spec = _ti_board_defaults(board_model)
    tx_schedule = _tx_schedule(data, name="FMCW profile tx_schedule")
    if tx_schedule is None:
        tdm_virtual_adc = _boolean(
            _first(data, "tdm_virtual_adc", default=False),
            name="FMCW profile tdm_virtual_adc",
        )
        tx_schedule = "tdm" if tdm_virtual_adc else "simultaneous"
    carrier_frequency = _first(
        data,
        "carrier_frequency",
        "carrier_frequency_hz",
    )
    if carrier_frequency is None:
        start_frequency = _first(data, "start_frequency", "start_frequency_hz")
        centre = data.get("center_frequency_hz")
        if start_frequency is not None:
            slope = float(_required_fmcw_value(data, "slope", "slope", "slope_hz_per_s"))
            carrier_frequency = float(start_frequency) + slope * float(
                _first(data, "adc_start_time_s", "adc_start_time", default=0.0)
            )
        elif centre is not None:
            slope = float(_required_fmcw_value(data, "slope", "slope", "slope_hz_per_s"))
            fs = float(
                _required_fmcw_value(
                    data, "sampling_frequency", "sampling_frequency", "sampling_frequency_hz"
                )
            )
            carrier_frequency = float(centre) - slope * int(data["num_adc_samples"]) / (2 * fs)
        elif data.get("design_frequency_hz") is not None:
            carrier_frequency = float(data["design_frequency_hz"])
        elif board_spec is not None:
            carrier_frequency = float(board_spec.design_frequency_hz)
        else:
            raise ValueError(
                "FMCW profile missing carrier frequency; expected "
                "carrier_frequency_hz, center_frequency_hz, start_frequency_hz "
                "with ADC start delay, or a known TI board model"
            )

    num_tx = int(
        _required_fmcw_value(
            data,
            "num_tx",
            "num_tx",
            default=1 if board_spec is None else int(board_spec.num_tx),
        )
    )
    chirp_duration = _required_fmcw_value(
        data,
        "chirp_duration",
        "chirp_duration",
        "chirp_duration_s",
        "ramp_time_s",
        "ramp_end_time_s",
        "effective_chirp_time_s",
    )
    chirp_repetition_time = _first(
        data,
        "chirp_repetition_time",
        "chirp_repetition_time_s",
    )
    if chirp_repetition_time is None:
        idle_time = _first(data, "idle_time", "idle_time_s", default=0.0)
        ramp_time = _first(
            data,
            "ramp_end_time_s",
            "ramp_time_s",
            "chirp_duration_s",
            "chirp_duration",
            default=chirp_duration,
        )
        chirp_repetition_time = float(idle_time) + float(ramp_time)

    num_chirps_per_frame = _first(data, "num_chirps_per_frame")
    per_tx = _first(
        data,
        "num_chirps_per_frame_per_tx",
        "num_chirps_per_tx",
    )
    if tx_schedule == "tdm" and per_tx is not None:
        # Older board profiles use num_chirps_per_frame for the total number
        # of physical Tx slots. FMCWConfig uses the number of slow-time
        # samples available to each transmitter, which those profiles expose
        # explicitly as num_chirps_per_tx.
        num_chirps_per_frame = int(per_tx)
    elif num_chirps_per_frame is None and per_tx is not None:
        num_chirps_per_frame = int(per_tx) * int(num_tx)
    if num_chirps_per_frame is None:
        raise ValueError(
            "FMCW profile missing num_chirps_per_frame or "
            "num_chirps_per_frame_per_tx/num_chirps_per_tx"
        )

    return FMCWConfig(
        carrier_frequency=float(carrier_frequency),
        slope=float(_required_fmcw_value(data, "slope", "slope", "slope_hz_per_s")),
        chirp_duration=float(chirp_duration),
        chirp_repetition_time=float(chirp_repetition_time),
        sampling_frequency=float(
            _required_fmcw_value(
                data,
                "sampling_frequency",
                "sampling_frequency",
                "sampling_frequency_hz",
            )
        ),
        num_adc_samples=int(_required_fmcw_value(data, "num_adc_samples", "num_adc_samples")),
        num_chirps_per_frame=int(num_chirps_per_frame),
        frame_period=float(
            _required_fmcw_value(
                data,
                "frame_period",
                "frame_period",
                "frame_period_s",
            )
        ),
        num_tx=num_tx,
        tx_schedule=tx_schedule,
        adc_mode=str(data.get("adc_mode", "complex1x")),
        if_bandwidth_hz=data.get("if_bandwidth_hz"),
    )


def _normalize_carrier_reference(data: dict[str, Any]) -> tuple[str, ...]:
    """Resolve old waveform metadata without guessing an unlabeled carrier."""
    reference = data.get("carrier_reference")
    if reference not in (None, "first_adc_sample"):
        raise BundleValidationError(
            "Convert carrier to the first ADC sample before importing",
            file="sensor.json",
            field="fmcw.carrier_reference",
            code="carrier.reference",
        )
    start = _first(data, "start_frequency_hz", "start_frequency")
    delay = _first(data, "adc_start_time_s", "adc_start_time")
    carrier = _first(data, "carrier_frequency", "carrier_frequency_hz")
    if start is not None and delay is not None:
        slope = float(_required_fmcw_value(data, "slope", "slope", "slope_hz_per_s"))
        expected = float(start) + slope * float(delay)
        if carrier is not None and not np.isclose(float(carrier), expected, rtol=1e-10, atol=1.0):
            if reference == "first_adc_sample":
                raise BundleValidationError(
                    "First-sample carrier disagrees with start_frequency_hz + slope * adc_start_time_s",
                    file="sensor.json",
                    field="fmcw.carrier_frequency_hz",
                    code="carrier.inconsistent",
                )
            data["carrier_conversion"] = {
                "source_frequency_hz": float(carrier),
                "source_reference": "unspecified",
                "method": "start_frequency_hz + slope_hz_per_s * adc_start_time_s",
            }
            warnings.warn(
                "Legacy carrier converted to the first ADC sample using source waveform fields",
                RuntimeWarning,
                stacklevel=3,
            )
        data["carrier_frequency_hz"] = expected
        data.pop("carrier_frequency", None)
        data["carrier_reference"] = "first_adc_sample"
    elif reference is None and carrier is None and data.get("center_frequency_hz") is not None:
        # An explicitly named sampled-sweep centre can be converted by _fmcw_from_mapping.
        data["carrier_reference"] = "first_adc_sample"
    elif reference is None:
        message = (
            "Carrier reference is unspecified and cannot be derived; re-simulation is disabled. "
            "Supply verified fmcw.carrier_reference: first_adc_sample or source start frequency and ADC delay"
        )
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        return (message,)
    return ()


@dataclass(frozen=True)
class BundleSensorConfig:
    """Radar hardware, pose, and waveform metadata stored in a bundle."""

    board_model: str | None
    fmcw: FMCWConfig
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = "radar"
    pattern_mode: str = "none"
    cosine_3db_beamwidth_deg: float = 60.0
    tx_power_dbm: float | None = None
    virtual_channel_order: str = "tx_major"
    tx_indices_zero_based: tuple[int, ...] | None = None
    rx_indices_zero_based: tuple[int, ...] | None = None
    metadata: Mapping[str, Any] | None = None
    array: Mapping[str, Any] | None = None
    tdm_tx_order: tuple[int, ...] | None = None
    adc_units: str = "counts"
    channel_map: tuple[tuple[int, int], ...] | None = None
    source_fmcw: Mapping[str, Any] | None = None
    timestamp_tolerance_s: float | None = None
    resimulation_issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.virtual_channel_order not in ("tx_major", "rx_major"):
            raise ValueError("virtual_channel_order must be 'tx_major' or 'rx_major'")
        tolerance = self.timestamp_tolerance_s
        if tolerance is not None and (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not np.isfinite(tolerance)
            or tolerance < 0
        ):
            raise BundleValidationError(
                "timestamp_tolerance_s must be finite and non-negative",
                file="sensor.json",
                field="timestamp_tolerance_s",
                code="timing.tolerance",
            )

    @classmethod
    def from_json(cls, path: str | Path) -> "BundleSensorConfig":
        """Load radar hardware, waveform, and pose metadata from ``sensor.json``."""

        return cls.from_mapping(read_json(Path(path)))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BundleSensorConfig":
        """Load a plain sensor mapping without a temporary JSON file."""
        if not isinstance(data, Mapping):
            raise ValueError("sensor.json must contain a JSON object")
        board_value = _first(data, "board_model", "hardware_model")
        if board_value is None and data.get("array") is None:
            raise ValueError("sensor.json requires board_model or array")
        board_model = None if board_value is None else str(board_value)

        if "fmcw" in data:
            if not isinstance(data["fmcw"], Mapping):
                raise ValueError("sensor.json fmcw must be a JSON object")
            fmcw_data = dict(data["fmcw"])
        else:
            profile_key = str(_first(data, "fmcw_profile", "profile", default=""))
            if not profile_key:
                raise ValueError("sensor.json requires fmcw or fmcw_profile/profile")
            defaults = _ti_board_defaults(board_model)
            if defaults is None:
                raise ValueError(
                    f"Unknown board model {board_model!r} cannot supply an FMCW profile"
                )
            profiles = defaults.metadata().get("fmcw_profiles", {})
            if profile_key not in profiles:
                raise ValueError(f"Unknown FMCW profile {profile_key!r} for {board_model!r}")
            fmcw_data = dict(profiles[profile_key])

        radar_data = data.get("radar", {})
        if not isinstance(radar_data, Mapping):
            raise ValueError("sensor.json radar must be a JSON object")
        position = _vec3(
            _first(
                data,
                "position",
                default=radar_data.get("position", (0, 0, 0)),
            ),
            name="radar.position",
        )
        orientation = _vec3(
            _first(
                data,
                "orientation",
                default=radar_data.get("orientation", (0, 0, 0)),
            ),
            name="radar.orientation",
        )
        tx_power = _first(
            data,
            "tx_power_dbm",
            default=radar_data.get("tx_power_dbm"),
        )
        tx_power_value = None if tx_power is None else float(tx_power)
        if tx_power_value is not None and not np.isfinite(tx_power_value):
            raise ValueError("radar.tx_power_dbm must be finite")
        cosine_3db_beamwidth_deg = float(_first(data, "cosine_3db_beamwidth_deg", default=60.0))
        if not np.isfinite(cosine_3db_beamwidth_deg) or not 0.0 < cosine_3db_beamwidth_deg < 180.0:
            raise ValueError(
                "sensor.json cosine_3db_beamwidth_deg must be between 0 and 180 degrees"
            )
        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("sensor.json metadata must be a JSON object")

        # ``tx_schedule`` is the canonical field; bundles written before it
        # carry a ``tdm_enabled`` flag. Older public bundles predate both and
        # declare their packed per-transmitter ADC layout as
        # ``metadata.tdm_virtual_adc``. An explicit FMCW or top-level value
        # takes precedence over that legacy marker.
        explicit_schedule = _tx_schedule(
            fmcw_data, name="sensor.json fmcw.tx_schedule"
        ) or _tx_schedule(data, name="sensor.json tx_schedule")
        legacy_tdm = _first(
            metadata,
            "tdm_virtual_adc",
            default=_first(data, "tdm_virtual_adc", default=None),
        )
        if legacy_tdm is None and metadata.get("source_adc_axes") == [
            "adc_sample",
            "chirp_loop",
            "rx",
            "tx",
        ]:
            legacy_tdm = True
        if explicit_schedule is not None:
            fmcw_data["tx_schedule"] = explicit_schedule
        elif legacy_tdm is not None:
            fmcw_data["tx_schedule"] = (
                "tdm"
                if _boolean(legacy_tdm, name="sensor.json tdm_virtual_adc")
                else "simultaneous"
            )
        if fmcw_data.get("tdm_tx_order") is not None:
            fmcw_data.setdefault("num_tx", len(fmcw_data["tdm_tx_order"]))
        elif board_model is None and isinstance(data.get("array"), Mapping):
            fmcw_data.setdefault("num_tx", len(data["array"]["tx_positions_m"]))

        def antenna_indices(name: str) -> tuple[int, ...] | None:
            value = data.get(name)
            if value is None:
                return None
            if (
                not isinstance(value, (list, tuple))
                or not value
                or any(
                    isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer))
                    for index in value
                )
            ):
                raise ValueError(f"sensor.json {name} must be a non-empty integer array")
            indices = tuple(int(index) for index in value)
            if any(index < 0 for index in indices):
                raise ValueError(f"sensor.json {name} must contain non-negative indices")
            if len(set(indices)) != len(indices):
                raise ValueError(f"sensor.json {name} must not contain duplicates")
            return indices

        carrier_issues = _normalize_carrier_reference(fmcw_data)
        return cls(
            board_model=board_model,
            fmcw=_fmcw_from_mapping(fmcw_data, board_model=board_model),
            position=position,
            orientation=orientation,
            name=str(_first(data, "name", default=radar_data.get("name", "radar"))),
            pattern_mode=str(_first(data, "pattern_mode", default="none")),
            cosine_3db_beamwidth_deg=cosine_3db_beamwidth_deg,
            tx_power_dbm=tx_power_value,
            virtual_channel_order=str(_first(data, "virtual_channel_order", default="tx_major")),
            tx_indices_zero_based=antenna_indices("tx_indices_zero_based"),
            rx_indices_zero_based=antenna_indices("rx_indices_zero_based"),
            metadata=dict(metadata),
            array=data.get("array"),
            tdm_tx_order=(
                None if fmcw_data.get("tdm_tx_order") is None else tuple(fmcw_data["tdm_tx_order"])
            ),
            adc_units=str(data.get("adc_units", "counts")),
            channel_map=(
                None
                if data.get("channel_map") is None
                else tuple(tuple(p) for p in data["channel_map"])
            ),
            source_fmcw=dict(fmcw_data),
            timestamp_tolerance_s=data.get("timestamp_tolerance_s"),
            resimulation_issues=carrier_issues,
        )

    def to_json(self) -> dict[str, Any]:
        """Serialize canonical SI-valued sensor metadata."""
        if self.resimulation_issues:
            raise BundleValidationError(
                "Cannot write unverified carrier metadata: " + "; ".join(self.resimulation_issues),
                file="sensor.json",
                field="fmcw.carrier_reference",
                code="carrier.reference",
            )
        f = self.fmcw
        data = {
            "name": self.name,
            "position": list(self.position),
            "orientation": list(self.orientation),
            "pattern_mode": self.pattern_mode,
            "cosine_3db_beamwidth_deg": self.cosine_3db_beamwidth_deg,
            "virtual_channel_order": self.virtual_channel_order,
            "adc_units": self.adc_units,
            "metadata": dict(self.metadata or {}),
            "fmcw": {
                **dict(self.source_fmcw or {}),
                "carrier_frequency_hz": f.carrier_frequency,
                "carrier_reference": "first_adc_sample",
                "slope_hz_per_s": f.slope,
                "chirp_duration_s": f.chirp_duration,
                "chirp_repetition_time_s": f.chirp_repetition_time,
                "sampling_frequency_hz": f.sampling_frequency,
                "num_adc_samples": f.num_adc_samples,
                "num_chirps_per_frame": f.num_chirps_per_frame,
                "frame_period_s": f.frame_period,
                "num_tx": f.num_tx,
                "tx_schedule": f.tx_schedule,
                "adc_mode": f.adc_mode,
                "if_bandwidth_hz": f.if_bandwidth_hz,
            },
        }
        data["fmcw"].pop("tdm_enabled", None)
        data["fmcw"].pop("carrier_frequency", None)
        for key in (
            "board_model",
            "array",
            "tx_power_dbm",
            "channel_map",
            "timestamp_tolerance_s",
            "tx_indices_zero_based",
            "rx_indices_zero_based",
        ):
            value = getattr(self, key)
            if value is not None:
                data[key] = list(value) if isinstance(value, tuple) else value
        if self.tdm_tx_order is not None:
            data["fmcw"]["tdm_tx_order"] = list(self.tdm_tx_order)

        def plain(value):
            if isinstance(value, Mapping):
                return {k: plain(v) for k, v in value.items()}
            if isinstance(value, (tuple, list, np.ndarray)):
                return [plain(v) for v in value]
            if isinstance(value, np.generic):
                return value.item()
            return value

        return plain(data)

    @classmethod
    def from_ti_cli_config(
        cls, filename: str | Path, *, board_model: str, **pose
    ) -> "BundleSensorConfig":
        """Import the supported TI profile/chirp/frame CLI subset."""
        from mmWaveRadar.radar.ti_cli import parse_ti_cli_config

        fmcw, tx_order = parse_ti_cli_config(
            str(filename), get_ti_board_spec(board_model).design_frequency_hz
        )
        rx_indices = None
        for line in Path(filename).read_text().splitlines():
            tokens = line.split()
            if tokens and tokens[0] == "channelCfg":
                rx_mask, tx_mask = int(tokens[1]), int(tokens[2])
                if rx_mask <= 0 or tx_mask <= 0:
                    raise ValueError("channelCfg must enable RX and TX elements")
                rx_indices = tuple(i for i in range(rx_mask.bit_length()) if rx_mask & (1 << i))
                if tx_order is None:
                    tx_order = tuple(i for i in range(tx_mask.bit_length()) if tx_mask & (1 << i))
                    fmcw = replace(fmcw, num_tx=len(tx_order))
                elif any(not tx_mask & (1 << i) for i in tx_order):
                    raise ValueError("chirpCfg enables a TX disabled by channelCfg")
        provenance = {}
        for line in Path(filename).read_text().splitlines():
            tokens = line.split()
            if tokens and tokens[0] == "profileCfg":
                provenance = {
                    "start_frequency_hz": float(tokens[2]) * 1e9,
                    "adc_start_time_s": float(tokens[4]) * 1e-6,
                    "source_profile_cfg": line.strip(),
                }
        return cls(
            board_model=board_model,
            fmcw=fmcw,
            source_fmcw=provenance,
            tx_indices_zero_based=tx_order,
            rx_indices_zero_based=rx_indices,
            tdm_tx_order=tx_order if fmcw.tx_schedule == "tdm" else None,
            **pose,
        )

    def full_hardware(self) -> RadarHardware:
        """Build the physical array, before selecting stored channels."""
        if self.array is not None:
            a = self.array
            if not isinstance(a, Mapping):
                raise ValueError("sensor.json array must be an object")
            if self.pattern_mode != "none":
                raise ValueError("Explicit arrays currently require pattern_mode='none'")
            pairs = np.asarray(a.get("channels")) if "channels" in a else None
            signs = np.asarray(a["phase_signs"]) if "phase_signs" in a else None
            if signs is not None and not np.all(np.isin(signs, [-1, 1])):
                raise ValueError("array.phase_signs must contain only -1 or +1")
            if pairs is not None and (pairs.ndim != 2 or pairs.shape[1] != 2):
                raise ValueError("array.channels must have shape [C,2]")
            return RadarHardware(
                name=self.board_model or "custom",
                tx_positions=a["tx_positions_m"],
                rx_positions=a["rx_positions_m"],
                virtual_channel_order=self.virtual_channel_order,
                virtual_channel_tx_indices=None if pairs is None else pairs[:, 0],
                virtual_channel_rx_indices=None if pairs is None else pairs[:, 1],
                virtual_channel_phase_signs=signs,
                scalar_antenna_gain_dbi=a.get("gain_dbi", 0.0),
            )
        return _hardware_with_virtual_channel_order(
            RadarHardware.from_ti_board(
                self.board_model,
                pattern_mode=self.pattern_mode,
                cosine_half_power_angle_deg=0.5 * self.cosine_3db_beamwidth_deg,
            ),
            self.virtual_channel_order,
        )

    def slot_order(self) -> np.ndarray:
        """Physical zero-based TX identifiers in acquisition-slot order."""
        raw = self.tdm_tx_order
        if raw is None:
            legacy = (self.metadata or {}).get("tx_to_enable") or (self.metadata or {}).get(
                "tdm_tx_order"
            )
            selected = self.tx_indices_zero_based
            default = (
                selected
                if selected is not None and len(selected) == self.fmcw.num_tx
                else tuple(range(self.fmcw.num_tx))
            )
            raw = tuple(int(v) - 1 for v in legacy) if legacy is not None else default
        values = np.asarray(raw)
        if (
            values.dtype.kind not in "iu"
            or values.shape != (self.fmcw.num_tx,)
            or len(set(values.tolist())) != values.size
            or np.any(values < 0)
            or np.any(values >= self.full_hardware().num_tx)
        ):
            raise ValueError(
                "fmcw.tdm_tx_order must contain unique physical TX indices, one per slot"
            )
        stored = self.tx_indices_zero_based
        if stored is not None and not set(stored).issubset(set(values.tolist())):
            raise ValueError("Stored TX elements must occur in fmcw.tdm_tx_order")
        return values.astype(np.int64)

    def to_sensor(self, *, full_array: bool = False) -> RadarSensor:
        """Build the runtime radar sensor represented by this configuration."""

        hardware = self.full_hardware()
        physical_tx = tuple(range(hardware.num_tx))
        physical_rx = tuple(range(hardware.num_rx))
        if not full_array and self.tx_indices_zero_based is not None:
            hardware = hardware.subset_tx(self.tx_indices_zero_based)
            physical_tx = self.tx_indices_zero_based
        if not full_array and self.rx_indices_zero_based is not None:
            hardware = hardware.subset_rx(self.rx_indices_zero_based)
            physical_rx = self.rx_indices_zero_based
        hardware = replace(
            hardware,
            board_metadata={
                **(hardware.board_metadata or {}),
                "physical_tx_indices": physical_tx,
                "physical_rx_indices": physical_rx,
                **(
                    {"tdm_slot_tx_indices": tuple(self.slot_order().tolist())}
                    if self.fmcw.tx_schedule == "tdm"
                    else {}
                ),
            },
        )
        desired = self.channel_map
        metadata = self.metadata or {}
        legacy_slots = metadata.get("source_adc_axes") == [
            "adc_sample",
            "chirp_loop",
            "rx",
            "tx",
        ] or any("tx_major" in str(axis) for axis in metadata.get("exported_adc_axes", []))
        if not full_array and desired is None and self.fmcw.tx_schedule == "tdm" and legacy_slots:
            desired = [
                (int(t), int(r))
                for t in self.slot_order()
                if int(t) in physical_tx
                for r in physical_rx
            ]
        if not full_array and desired is not None:
            current = list(
                zip(
                    np.asarray(physical_tx)[hardware.virtual_tx_indices()],
                    np.asarray(physical_rx)[hardware.virtual_rx_indices()],
                )
            )
            lookup = {pair: i for i, pair in enumerate(current)}
            if len(desired) != len(current) or set(map(tuple, desired)) != set(current):
                raise ValueError(
                    "channel_map must enumerate each stored physical TX/RX pair exactly once"
                )
            order = np.asarray([lookup[tuple(pair)] for pair in desired])
            hardware = replace(
                hardware,
                virtual_channel_tx_indices=hardware.virtual_tx_indices()[order],
                virtual_channel_rx_indices=hardware.virtual_rx_indices()[order],
                virtual_channel_positions_lambda=(
                    None
                    if hardware.virtual_channel_positions_lambda is None
                    else hardware.virtual_channel_positions_lambda[order]
                ),
                virtual_channel_phase_signs=(
                    None
                    if hardware.virtual_channel_phase_signs is None
                    else hardware.virtual_channel_phase_signs[order]
                ),
                virtual_channel_labels=(
                    None
                    if hardware.virtual_channel_labels is None
                    else tuple(hardware.virtual_channel_labels[i] for i in order)
                ),
                antenna_pattern=(
                    hardware.antenna_pattern.subset(order)
                    if hardware.antenna_pattern is not None
                    and hasattr(hardware.antenna_pattern, "subset")
                    else hardware.antenna_pattern
                ),
            )
        return RadarSensor(
            name=self.name,
            hardware=hardware,
            fmcw=(
                self.fmcw
                if self.fmcw.tx_schedule == "tdm"
                else replace(self.fmcw, num_tx=hardware.num_tx)
            ),
            position=self.position,
            orientation=self.orientation,
            tx_power_dbm=(
                (get_ti_board_spec(self.board_model).tx_power_dbm if self.array is None else 0.0)
                if self.tx_power_dbm is None
                else self.tx_power_dbm
            ),
        )


@dataclass(frozen=True)
class FrameRecord:
    """Frame-ID mapping and body-motion timestamp for one bundle frame."""

    bundle_index: int
    motion_time_s: float
    radar_frame_id: str | None = None
    camera_frame_id: str | None = None
    pose_frame_id: str | None = None
    dataset_frame_id: str | None = None
    metadata: Mapping[str, Any] | None = None

    @property
    def benchmark_index(self) -> int:
        """Compatibility name used by existing benchmark reports."""

        return self.bundle_index

    @classmethod
    def from_mapping(cls, index: int, data: Mapping[str, Any]) -> "FrameRecord":
        if not isinstance(data, Mapping):
            raise ValueError(f"frames.json entry {index} must be a JSON object")
        motion_time = data.get("motion_time_s")
        if motion_time is None:
            raise ValueError(f"frames.json entry {index} must include motion_time_s")
        motion_time_value = float(motion_time)
        if not np.isfinite(motion_time_value):
            raise ValueError(f"frames.json entry {index} motion time must be finite")
        bundle_index = int(_first(data, "bundle_index", "benchmark_index", default=index))
        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"frames.json entry {index} metadata must be an object")
        return cls(
            bundle_index=bundle_index,
            motion_time_s=motion_time_value,
            radar_frame_id=_optional_str(_first(data, "radar_frame_id", "Radar_frameID")),
            camera_frame_id=_optional_str(_first(data, "camera_frame_id", "camera_id")),
            pose_frame_id=_optional_str(_first(data, "pose_frame_id", "pose_id")),
            dataset_frame_id=_optional_str(_first(data, "dataset_frame_id", "frame_id")),
            metadata=dict(metadata),
        )


def load_frames(path: Path) -> list[FrameRecord]:
    data = read_json(path)
    frames = data["frames"] if isinstance(data, Mapping) and "frames" in data else data
    if not isinstance(frames, list) or not frames:
        raise ValueError("frames.json must contain a non-empty list or {'frames': [...]}")
    records = [FrameRecord.from_mapping(index, frame) for index, frame in enumerate(frames)]
    indices = [record.bundle_index for record in records]
    if indices != list(range(len(records))):
        raise ValueError("frames.json bundle/benchmark indices must be contiguous from zero")
    return records


def load_adc(
    path: Path,
    *,
    key: str = "adc",
    allowed_ndim: tuple[int, ...] = (4,),
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load a pickle-free ADC array and validate its optional timestamps."""

    arrays = preflight_npz(path, limits=DEFAULT_NPZ_LIMITS)
    adc_key = key if key in arrays else "adc"
    if adc_key not in arrays:
        raise ValueError(f"{path} must contain an {key!r} or 'adc' array")
    try:
        adc_info = require_array(
            arrays,
            adc_key,
            allowed_ndim=allowed_ndim,
            dtype_kinds=frozenset({"i", "u", "f", "c"}),
        )
    except ValueError as exc:
        if adc_key in arrays and len(arrays[adc_key].shape) not in allowed_ndim:
            expected = (
                "[frames, chirps, samples, channels]"
                if allowed_ndim == (4,)
                else ("[frames, chirps, samples, channels] or [chirps, samples, channels]")
            )
            raise ValueError(f"{path} adc must have shape {expected}") from exc
        raise
    if any(size <= 0 for size in adc_info.shape):
        raise ValueError(f"{path} adc dimensions must all be non-empty")
    if "times" in arrays:
        require_array(
            arrays,
            "times",
            dtype_kinds=frozenset({"i", "u", "f"}),
        )
    archive = np.load(path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError(f"Expected a NumPy NPZ archive: {path}")
    with archive:
        adc_key = key if key in archive.files else "adc"
        if adc_key not in archive.files:
            raise ValueError(f"{path} must contain an {key!r} or 'adc' array")
        adc = np.array(archive[adc_key], copy=True)
        if adc.ndim not in allowed_ndim:
            expected = (
                "[frames, chirps, samples, channels]"
                if allowed_ndim == (4,)
                else "[frames, chirps, samples, channels] or [chirps, samples, channels]"
            )
            raise ValueError(f"{path} adc must have shape {expected}")
        if any(size <= 0 for size in adc.shape):
            raise ValueError(f"{path} adc dimensions must all be non-empty")
        if adc.dtype.kind not in {"i", "u", "f", "c"}:
            raise ValueError(f"{path} adc must use a plain numeric dtype")
        if not np.all(np.isfinite(adc)):
            raise ValueError(f"{path} adc must contain only finite values")
        times = np.array(archive["times"], copy=True) if "times" in archive.files else None
    expected_times_shape = adc.shape[:-2]
    if times is not None:
        if times.shape != expected_times_shape:
            raise ValueError(f"{path} times must have shape {expected_times_shape}")
        if times.dtype.kind not in {"i", "u", "f"} or not np.all(np.isfinite(times)):
            raise ValueError(f"{path} times must contain finite real values")
    return adc, times


def required_bundle_features(descriptor, sensor) -> set[str]:
    """Infer feature gates identically for readers and writers."""
    target = descriptor.get("target") or {}
    if not isinstance(target, Mapping):
        raise ValueError("bundle.json target must be an object")
    return {
        feature
        for enabled, feature in (
            (sensor.array is not None, "custom_array"),
            (sensor.tdm_tx_order is not None, "explicit_tdm_slots"),
            (sensor.channel_map is not None, "channel_map"),
            (bool(target.get("material_parameters")), "target_material"),
        )
        if enabled
    }


def validate_required_features(descriptor, sensor) -> None:
    """Check mandatory interpretations before loading any ADC artifact."""
    missing = required_bundle_features(descriptor, sensor) - set(
        descriptor.get("required_features", [])
    )
    if missing:
        feature = sorted(missing)[0]
        raise BundleValidationError(
            f"bundle.json required_features must declare {feature}",
            field="required_features",
            code="feature.missing",
        )


def normalize_adc_times(times, frame_times, fmcw, *, tolerance_s=None):
    """Rebase clock epochs and check loop timing with jitter/ULP tolerance.

    The default jitter budget is one percent of a loop interval (at least
    1 ns). Quantization of the supplied clocks supplies an additional lower
    bound; a relative tolerance must never grow with the epoch or run length.
    """
    frames = np.asarray(frame_times, dtype=float)
    expected = (
        frames[:, None]
        - frames[0]
        + np.arange(fmcw.num_chirps_per_frame)[None, :] * fmcw.slow_time_interval
    )
    if tolerance_s is None:
        tolerance_s = max(1e-9, 0.01 * fmcw.slow_time_interval)
    if (
        isinstance(tolerance_s, (bool, np.bool_))
        or not np.isfinite(tolerance_s)
        or tolerance_s < 0
    ):
        raise BundleValidationError(
            "timestamp_tolerance_s must be finite and non-negative",
            file="sensor.json",
            field="timestamp_tolerance_s",
            code="timing.tolerance",
        )
    if times is None:
        return expected
    values = np.asarray(times, dtype=float)
    if values.shape != expected.shape:
        raise BundleValidationError(
            "ADC times shape must match frame and loop counts",
            file="radar_adc.npz",
            field="times",
            code="timing.shape",
        )
    # Account for rounding already present in epoch-valued float timestamps.
    ulp = max(float(np.max(np.abs(np.spacing(values)))), float(np.max(np.abs(np.spacing(frames)))))
    relative = values - values[0, 0]
    if not np.allclose(relative, expected, rtol=0, atol=max(float(tolerance_s), 4 * ulp)):
        raise BundleValidationError(
            "ADC timestamps disagree with frames.json/FMCW loop timing",
            file="radar_adc.npz",
            field="times",
            code="timing.inconsistent",
        )
    if np.any(np.diff(relative, axis=1) <= 0) or np.any(np.diff(relative[:, 0]) <= 0):
        raise BundleValidationError(
            "ADC times must increase across loops and frames",
            file="radar_adc.npz",
            field="times",
            code="timing.order",
        )
    return relative


def validate_acquisition_coverage(frame_times, motion_times, fmcw) -> None:
    """Require coverage at every physical chirp start without evaluating SMPL."""
    frames = np.asarray(frame_times, dtype=float)
    motion = np.asarray(motion_times, dtype=float)
    slots = fmcw.num_tx if fmcw.tx_schedule == "tdm" else 1
    stop = (
        float(np.max(frames))
        + (fmcw.num_chirps_per_frame * slots - 1) * fmcw.chirp_repetition_time
    )
    tolerance = max(1e-12, abs(stop) * 1e-12)
    if np.min(frames) < motion[0] - tolerance or stop > motion[-1] + tolerance:
        raise ValueError(
            f"Motion does not cover the full radar acquisition: motion ends at {motion[-1]:.9g}s but the final physical chirp requires {stop:.9g}s"
        )


def _load_motion_times(path: Path) -> np.ndarray:
    """Load the preferred bundle-relative timeline from a motion archive."""

    arrays = preflight_npz(path, limits=DEFAULT_NPZ_LIMITS)
    timing_key = "bundle_times" if "bundle_times" in arrays else "times"
    require_array(
        arrays,
        timing_key,
        allowed_ndim=(1,),
        dtype_kinds=frozenset({"i", "u", "f"}),
        max_elements=10_000_000,
    )
    archive = np.load(path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError(f"Expected a NumPy NPZ archive: {path}")
    with archive:
        timing_key = "bundle_times" if "bundle_times" in archive.files else "times"
        if timing_key not in archive.files:
            raise ValueError(f"{path} must contain times or bundle_times")
        times = np.array(archive[timing_key], copy=True)
    if (
        times.ndim != 1
        or times.size == 0
        or times.dtype.kind not in {"i", "u", "f"}
        or not np.all(np.isfinite(times))
    ):
        raise ValueError(f"{path} {timing_key} must be a non-empty finite vector")
    if times.size > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError(f"{path} {timing_key} must be strictly increasing")
    return np.asarray(times, dtype=float)


def load_bundle_descriptor(root: str | Path) -> dict[str, Any]:
    """Load the unified HERMES Bundle v1 descriptor."""

    root = Path(root)
    path = root / "bundle.json"
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Bundle descriptor was not found: {path}")
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise ValueError("bundle.json must contain a JSON object")
    descriptor = dict(value)
    profile = str(descriptor.get("profile", "")).strip()
    if profile != HERMES_BUNDLE_PROFILE:
        raise ValueError(
            f"Unsupported bundle profile {profile!r}; expected " f"{HERMES_BUNDLE_PROFILE!r}"
        )
    schema_version = descriptor.get("schema_version")
    if type(schema_version) is not int or schema_version not in (1, 2):
        raise ValueError("bundle.json schema_version must be 1 or 2")
    features = descriptor.get("required_features", [])
    if (
        not isinstance(features, list)
        or any(not isinstance(v, str) for v in features)
        or len(set(features)) != len(features)
    ):
        raise ValueError("bundle.json required_features must be a unique string list")
    unknown = set(features) - SUPPORTED_FEATURES
    if unknown:
        raise BundleValidationError(
            f"Unsupported required_features: {sorted(unknown)}",
            field="required_features",
            code="feature.unsupported",
        )
    if features and schema_version < 2:
        raise ValueError("required_features requires schema_version 2 so older readers reject it")
    artifacts = descriptor.get("adcs")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("bundle.json adcs must be a non-empty object")
    if len(artifacts) > _MAX_BUNDLE_ADC_ARTIFACTS:
        raise ValueError(
            "bundle.json adcs contains too many artifacts: "
            f"{len(artifacts):,} > {_MAX_BUNDLE_ADC_ARTIFACTS:,}"
        )
    primary = str(descriptor.get("primary_adc", "")).strip()
    if not primary or primary not in artifacts:
        raise ValueError("bundle.json primary_adc must name an entry in adcs")
    adc_paths: set[str] = set()
    for name, artifact in artifacts.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("bundle.json adcs keys must be non-empty strings")
        if not isinstance(artifact, Mapping):
            raise ValueError(f"bundle.json adcs.{name} must be an object")
        path_value = artifact.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            raise ValueError(f"bundle.json adcs.{name}.path must be a non-empty relative path")
        relative_path = portable_relative_path(path_value, label=f"bundle.json adcs.{name}.path")
        canonical_path = relative_path.as_posix()
        if canonical_path in adc_paths:
            raise ValueError(
                "bundle.json ADC artifacts must reference unique paths: " f"{path_value!r}"
            )
        adc_paths.add(canonical_path)
        origin = str(artifact.get("origin", "")).strip()
        if origin not in ADC_ARTIFACT_ORIGINS:
            raise ValueError(
                f"bundle.json adcs.{name}.origin must be one of " f"{sorted(ADC_ARTIFACT_ORIGINS)}"
            )
    primary_origin = str(artifacts[primary].get("origin", ""))
    if primary_origin not in {"measurement", "simulation"}:
        raise ValueError("bundle.json primary ADC origin must be measurement or simulation")
    return descriptor


_safe_bundle_file = safe_bundle_file  # Compatibility for existing callers.


def load_static_target_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate a static triangular mesh archive."""

    path = Path(path)
    arrays = preflight_npz(path, limits=DEFAULT_NPZ_LIMITS)
    require_array(
        arrays,
        "vertices",
        allowed_ndim=(2, 3),
        dtype_kinds=frozenset({"i", "u", "f"}),
        trailing_shape=(3,),
        max_elements=60_000_000,
    )
    require_array(
        arrays,
        "faces",
        allowed_ndim=(2,),
        dtype_kinds=frozenset({"i", "u"}),
        trailing_shape=(3,),
        max_elements=60_000_000,
    )
    archive = np.load(path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError(f"Expected a NumPy NPZ archive: {path}")
    with archive:
        if "vertices" not in archive.files or "faces" not in archive.files:
            raise ValueError(f"{path} must contain vertices and faces")
        vertices = np.array(archive["vertices"], copy=True)
        faces = np.array(archive["faces"], copy=True)
    if vertices.ndim == 3 and vertices.shape[0] == 1:
        vertices = vertices[0]
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or vertices.shape[0] < 3
        or vertices.dtype.kind not in {"i", "u", "f"}
        or not np.all(np.isfinite(vertices))
    ):
        raise ValueError(f"{path} vertices must have finite shape [V, 3]")
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or faces.shape[0] < 1
        or faces.dtype.kind not in {"i", "u"}
    ):
        raise ValueError(f"{path} faces must have integer shape [F, 3]")
    if np.min(faces) < 0 or np.max(faces) >= vertices.shape[0]:
        raise ValueError(f"{path} faces reference vertices outside vertices")
    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(faces, dtype=np.uint32),
    )


def preflight_motion_archive(path: str | Path) -> dict[str, object]:
    """Preflight AMASS-like motion shapes before any array is allocated."""

    arrays = preflight_npz(path, limits=DEFAULT_NPZ_LIMITS)
    poses = require_array(
        arrays,
        "poses",
        allowed_ndim=(2,),
        dtype_kinds=frozenset({"i", "u", "f"}),
        max_elements=30_000_000,
    )
    trans = require_array(
        arrays,
        "trans",
        allowed_ndim=(2,),
        dtype_kinds=frozenset({"i", "u", "f"}),
        trailing_shape=(3,),
        max_elements=30_000_000,
    )
    require_array(
        arrays,
        "betas",
        allowed_ndim=(1,),
        dtype_kinds=frozenset({"i", "u", "f"}),
        max_elements=10_000,
    )
    if "faces" in arrays:
        require_array(
            arrays,
            "faces",
            allowed_ndim=(2,),
            dtype_kinds=frozenset({"i", "u"}),
            trailing_shape=(3,),
            max_elements=6_000_000,
            max_bytes=32 * 1024**2,
        )
    for name, maximum_bytes in (
        ("model_type", 256),
        ("gender", 256),
        ("smpl_model_dir", 4 * 1024),
    ):
        if name in arrays:
            require_array(
                arrays,
                name,
                allowed_ndim=(0,),
                dtype_kinds=frozenset({"S", "U"}),
                max_elements=1,
                max_bytes=maximum_bytes,
            )
    if poses.shape[0] <= 0 or trans.shape[0] != poses.shape[0]:
        raise ValueError("AMASS-like poses and trans frame counts must match")
    for timing_key in ("times", "bundle_times"):
        if timing_key in arrays:
            timing = require_array(
                arrays,
                timing_key,
                allowed_ndim=(1,),
                dtype_kinds=frozenset({"i", "u", "f"}),
                max_elements=10_000_000,
            )
            if timing.shape != (poses.shape[0],):
                raise ValueError(f"AMASS-like {timing_key} must match the pose frame count")
    return arrays


def preflight_mesh_sequence_archive(path: str | Path) -> dict[str, object]:
    """Preflight an evaluated mesh sequence before NumPy materialization."""

    arrays = preflight_npz(path, limits=DEFAULT_NPZ_LIMITS)
    vertices = require_array(
        arrays,
        "vertices",
        allowed_ndim=(2, 3),
        dtype_kinds=frozenset({"i", "u", "f"}),
        trailing_shape=(3,),
        max_elements=120_000_000,
    )
    require_array(
        arrays,
        "faces",
        allowed_ndim=(2,),
        dtype_kinds=frozenset({"i", "u"}),
        trailing_shape=(3,),
        max_elements=60_000_000,
    )
    if "times" in arrays:
        times = require_array(
            arrays,
            "times",
            allowed_ndim=(1,),
            dtype_kinds=frozenset({"i", "u", "f"}),
            max_elements=10_000_000,
        )
        frame_count = vertices.shape[0] if len(vertices.shape) == 3 else 1
        if times.shape != (frame_count,):
            raise ValueError("mesh sequence times must match its frame count")
    return arrays


@dataclass(frozen=True)
class Bundle:
    """Dataset-neutral bundle consumed by simulation validation."""

    root: Path
    sensor_config: BundleSensorConfig
    real_adc: np.ndarray
    frames: tuple[FrameRecord, ...]
    motion_path: Path | None
    times: np.ndarray | None = None
    background_adc: np.ndarray | None = None
    environment: Mapping[str, Any] | None = None
    descriptor: Mapping[str, Any] | None = None
    profile: str = HERMES_BUNDLE_PROFILE
    data_origin: str = "measurement"
    primary_adc_key: str = "primary"
    adc_paths: Mapping[str, Path] | None = None
    adc_origins: Mapping[str, str] | None = None
    simulated_adc_paths: Mapping[str, Path] | None = None
    target_mesh_path: Path | None = None
    target_obj_path: Path | None = None
    scene_path: Path | None = None
    evaluated_motion_path: Path | None = None
    resimulation_issues: tuple[str, ...] = ()

    @classmethod
    def load(cls, root: str | Path) -> "Bundle":
        """Load the core contract and validate cross-file consistency."""

        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"Bundle directory was not found: {root}")
        descriptor = load_bundle_descriptor(root)
        profile = str(descriptor["profile"])
        sensor = BundleSensorConfig.from_json(root / "sensor.json")
        validate_required_features(descriptor, sensor)
        if sensor.fmcw.tx_schedule == "tdm":
            sensor.slot_order()
        frames = tuple(load_frames(root / "frames.json"))

        artifacts = descriptor["adcs"]
        primary_adc_key = str(descriptor["primary_adc"])
        adc_paths: dict[str, Path] = {}
        adc_origins: dict[str, str] = {}
        loaded_adcs: dict[str, np.ndarray] = {}
        loaded_times: dict[str, np.ndarray | None] = {}
        background = None
        background_keys: list[str] = []
        resolved_adc_artifacts: list[tuple[str, Mapping[str, Any], Path]] = []
        resolved_adc_paths: set[Path] = set()
        for name, artifact in artifacts.items():
            path = safe_bundle_file(
                root,
                artifact.get("path"),
                label=f"bundle.json adcs.{name}.path",
            )
            if path in resolved_adc_paths:
                raise ValueError(
                    "bundle.json ADC artifacts must resolve to unique files: "
                    f"{artifact.get('path')!r}"
                )
            resolved_adc_paths.add(path)
            resolved_adc_artifacts.append((str(name), artifact, path))

        for name, artifact, path in resolved_adc_artifacts:
            origin = str(artifact["origin"])
            adc_value, adc_times = load_adc(
                path,
                allowed_ndim=(3, 4) if origin == "background" else (4,),
            )
            mode = sensor.fmcw.adc_mode
            if mode != "real" and not np.iscomplexobj(adc_value):
                raise BundleValidationError(
                    f"{mode} requires complex I/Q ADC, received {adc_value.dtype}",
                    file=path.name,
                    field="adc",
                    code="adc.mode_dtype",
                )
            if mode == "real" and np.iscomplexobj(adc_value) and np.any(adc_value.imag != 0):
                raise BundleValidationError(
                    "real ADC mode requires zero imaginary samples",
                    file=path.name,
                    field="adc",
                    code="adc.mode_dtype",
                )
            adc_paths[name] = path
            adc_origins[name] = origin
            loaded_adcs[name] = adc_value
            loaded_times[name] = adc_times
            if artifact.get("manifest") is not None:
                safe_bundle_file(
                    root,
                    artifact.get("manifest"),
                    label=f"bundle.json adcs.{name}.manifest",
                )
            if origin == "background":
                background_keys.append(name)
        if len(background_keys) > 1:
            raise ValueError("bundle.json may declare at most one background ADC")

        adc = loaded_adcs[primary_adc_key]
        times = loaded_times[primary_adc_key]
        data_origin = adc_origins[primary_adc_key]
        for name, value in loaded_adcs.items():
            if adc_origins[name] == "background":
                continue
            if value.shape != adc.shape:
                raise ValueError(
                    f"{adc_paths[name].name} ADC shape must match the primary ADC "
                    f"({value.shape} != {adc.shape})"
                )
        if background_keys:
            background = loaded_adcs[background_keys[0]]
            _validate_background_shape(background, adc.shape)

        motion_path: Path | None = None
        evaluated_motion_path: Path | None = None
        target_mesh_path: Path | None = None
        target_obj_path: Path | None = None
        scene_path: Path | None = None
        simulated_adc_paths = {
            name: path
            for name, path in adc_paths.items()
            if adc_origins[name] == "simulation" and name != primary_adc_key
        }

        motion = descriptor.get("motion")
        if motion is not None:
            if not isinstance(motion, Mapping):
                raise ValueError("bundle.json motion must be an object")
            if motion.get("coordinate_frame", "world") != "world":
                raise ValueError(
                    "motion.coordinate_frame must be world; normalize source coordinates before writing"
                )
            if motion.get("parameters") is not None:
                motion_path = safe_bundle_file(
                    root,
                    motion.get("parameters"),
                    label="bundle.json motion.parameters",
                )
                preflight_motion_archive(motion_path)
            if motion.get("evaluated_mesh") is not None:
                evaluated_motion_path = safe_bundle_file(
                    root,
                    motion.get("evaluated_mesh"),
                    label="bundle.json motion.evaluated_mesh",
                )
                preflight_mesh_sequence_archive(evaluated_motion_path)

        target = descriptor.get("target")
        if target is not None:
            if not isinstance(target, Mapping):
                raise ValueError("bundle.json target must be an object")
            material = target.get("material_parameters", {})
            if not isinstance(material, Mapping):
                raise ValueError("target.material_parameters must be an object")
            bounds = {
                "relative_permittivity": (1, None),
                "conductivity": (0, None),
                "thickness": (0, None),
                "scattering_coefficient": (0, 1),
                "xpd_coefficient": (0, 1),
            }
            for key, value in material.items():
                if (
                    key not in bounds
                    or isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not np.isfinite(value)
                    or value < bounds[key][0]
                    or (bounds[key][1] is not None and value > bounds[key][1])
                ):
                    raise ValueError(f"Invalid target.material_parameters.{key}")
            if target.get("mesh") is not None:
                target_mesh_path = safe_bundle_file(
                    root,
                    target.get("mesh"),
                    label="bundle.json target.mesh",
                )
                load_static_target_mesh(target_mesh_path)
            if target.get("obj") is not None:
                target_obj_path = safe_bundle_file(
                    root,
                    target.get("obj"),
                    label="bundle.json target.obj",
                )
                if target_obj_path.stat().st_size > _MAX_BUNDLE_OBJ_BYTES:
                    raise ValueError("bundle target OBJ exceeds the 64 MiB limit")

        scene = descriptor.get("scene")
        if scene is not None:
            if not isinstance(scene, Mapping):
                raise ValueError("bundle.json scene must be an object")
            if scene.get("file") is not None:
                scene_path = safe_bundle_file(
                    root,
                    scene.get("file"),
                    label="bundle.json scene.file",
                )
            assets = scene.get("assets", [])
            if not isinstance(assets, list):
                raise ValueError("bundle.json scene.assets must be an array")
            for index, asset in enumerate(assets):
                safe_bundle_file(
                    root,
                    asset,
                    label=f"bundle.json scene.assets[{index}]",
                )

        for group_name in ("manifests", "diagnostics"):
            group = descriptor.get(group_name, {})
            if not isinstance(group, Mapping):
                raise ValueError(f"bundle.json {group_name} must be an object")
            for name, path_value in group.items():
                safe_bundle_file(
                    root,
                    path_value,
                    label=f"bundle.json {group_name}.{name}",
                )

        environment = None
        environment_value = descriptor.get("environment")
        if environment_value is not None:
            environment_path = safe_bundle_file(
                root,
                environment_value,
                label="bundle.json environment",
            )
            value = read_json(environment_path)
            if not isinstance(value, Mapping):
                raise ValueError(f"{environment_path.name} must contain a JSON object")
            environment = dict(value)

        # Legacy aliases are checked at the same boundary as the descriptor.
        scenes = [] if scene_path is None else [scene_path]
        for key in ("scene_path", "scene_file"):
            if environment is not None and environment.get(key) is not None:
                scenes.append(
                    safe_bundle_file(root, environment[key], label=f"environment.json {key}")
                )
        if scenes:
            if len(set(scenes)) != 1:
                raise ValueError(
                    "Conflicting scene declarations in bundle.json and environment.json"
                )
            scene_path = scenes[0]

        if len(frames) != adc.shape[0]:
            raise ValueError(
                "frames.json length must match the primary ADC frame count "
                f"({len(frames)} != {adc.shape[0]})"
            )
        if int(sensor.fmcw.num_adc_samples) != adc.shape[2]:
            raise ValueError(
                "sensor fmcw.num_adc_samples must match ADC sample axis "
                f"({sensor.fmcw.num_adc_samples} != {adc.shape[2]})"
            )
        if int(sensor.fmcw.num_chirps_per_frame) != adc.shape[1]:
            raise ValueError(
                "sensor fmcw.num_chirps_per_frame must match ADC chirp axis "
                f"({sensor.fmcw.num_chirps_per_frame} != {adc.shape[1]})"
            )
        sensor_channels = sensor.to_sensor().hardware.num_virtual_channels
        if int(sensor_channels) != adc.shape[3]:
            raise ValueError(
                "sensor hardware virtual-channel count must match ADC channel axis "
                f"({sensor_channels} != {adc.shape[3]})"
            )
        frame_times = np.asarray([frame.motion_time_s for frame in frames])
        if frame_times.size > 1 and np.any(np.diff(frame_times) <= 0):
            raise ValueError("frames.json motion times must be strictly increasing")
        resimulation_issues = list(sensor.resimulation_issues)
        for timeline_path in (motion_path, evaluated_motion_path):
            if timeline_path is None:
                continue
            motion_times = _load_motion_times(timeline_path)
            try:
                validate_acquisition_coverage(frame_times, motion_times, sensor.fmcw)
            except ValueError as exc:
                resimulation_issues.append(f"{timeline_path.name}: {exc}")
        if times is not None:
            times = normalize_adc_times(
                times, frame_times, sensor.fmcw, tolerance_s=sensor.timestamp_tolerance_s
            )
        return cls(
            root=root,
            sensor_config=sensor,
            real_adc=adc,
            times=times,
            frames=frames,
            motion_path=motion_path,
            background_adc=background,
            environment=environment,
            descriptor=descriptor,
            profile=profile,
            data_origin=data_origin,
            primary_adc_key=primary_adc_key,
            adc_paths=adc_paths,
            adc_origins=adc_origins,
            simulated_adc_paths=simulated_adc_paths,
            target_mesh_path=target_mesh_path,
            target_obj_path=target_obj_path,
            scene_path=scene_path,
            evaluated_motion_path=evaluated_motion_path,
            resimulation_issues=tuple(resimulation_issues),
        )

    @property
    def num_frames(self) -> int:
        return int(self.real_adc.shape[0])

    @property
    def motion_times_s(self) -> np.ndarray:
        """Body-motion timestamps selected by ``frames.json``."""

        return np.asarray([frame.motion_time_s for frame in self.frames], dtype=float)

    @property
    def can_resimulate(self) -> bool:
        """Whether the bundle declares geometry for a fresh simulation."""

        return not self.resimulation_issues and any(
            path is not None
            for path in (
                self.motion_path,
                self.evaluated_motion_path,
                self.target_mesh_path,
                self.target_obj_path,
            )
        )

    @property
    def capabilities(self) -> tuple[str, ...]:
        """Return capabilities derived from declared bundle resources."""

        values = {"primary-adc"}
        if self.background_adc is not None:
            values.add("background-subtraction")
        if self.simulated_adc_paths:
            values.add("bundled-simulations")
        if self.motion_path is not None:
            values.add("human-motion-resimulation")
        if self.evaluated_motion_path is not None:
            values.add("motion-playback")
        if self.environment is not None:
            values.add("environment-metadata")
        if self.scene_path is not None:
            values.add("scene")
        if self.target_mesh_path is not None:
            values.add("target-geometry")
        return tuple(sorted(values))

    def sensor(self) -> RadarSensor:
        return self.sensor_config.to_sensor()

    def load_bundled_simulation(self, solver: str) -> np.ndarray:
        """Load a declared simulation ADC artifact."""

        paths = self.simulated_adc_paths or {}
        normalized = str(solver).lower()
        if normalized not in paths:
            raise ValueError(f"Bundle does not contain a {normalized.upper()} ADC cube")
        adc, _ = load_adc(paths[normalized])
        return adc


def _validate_background_shape(background: np.ndarray, adc_shape: tuple[int, ...]) -> None:
    if background.shape in {adc_shape, adc_shape[1:], (1, *adc_shape[1:])}:
        return
    raise ValueError("background_adc.npz adc must have shape [F,M,N,C], [M,N,C], or [1,M,N,C]")
