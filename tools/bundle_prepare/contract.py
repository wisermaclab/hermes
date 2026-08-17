# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Dataset-neutral loading and internal consistency for HERMES bundles."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mmWaveRadar.radar import (
    FMCWConfig,
    RadarHardware,
    RadarSensor,
    get_ti_board_spec,
)

from .safe_npz import DEFAULT_NPZ_LIMITS, preflight_npz, require_array


HERMES_BUNDLE_PROFILE = "hermes"
_MAX_BUNDLE_JSON_BYTES = 8 * 1024**2
_MAX_BUNDLE_OBJ_BYTES = 64 * 1024**2
_MAX_BUNDLE_ADC_ARTIFACTS = 64
ADC_ARTIFACT_ORIGINS = frozenset(
    {"measurement", "simulation", "background", "processed"}
)


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

    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _hardware_with_virtual_channel_order(
    hardware: RadarHardware,
    order: str,
) -> RadarHardware:
    """Return hardware whose per-channel metadata follows ``order``."""

    if order not in ("tx_major", "rx_major"):
        raise ValueError(
            "sensor.json virtual_channel_order must be 'tx_major' or "
            "'rx_major'"
        )
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
        labels = tuple(
            hardware.virtual_channel_labels[int(index)]
            for index in permutation
        )
    metadata = (
        None
        if hardware.board_metadata is None
        else dict(hardware.board_metadata)
    )
    if metadata is not None:
        metadata["virtual_channel_order"] = order
    return RadarHardware(
        name=hardware.name,
        tx_positions=hardware.tx_positions,
        rx_positions=hardware.rx_positions,
        virtual_channel_order=order,
        virtual_channel_labels=labels,
        virtual_channel_positions_lambda=reordered(
            hardware.virtual_channel_positions_lambda
        ),
        virtual_channel_tx_indices=desired_tx,
        virtual_channel_rx_indices=desired_rx,
        virtual_channel_phase_signs=reordered(
            hardware.virtual_channel_phase_signs
        ),
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
    tdm_enabled = _boolean(
        _first(
            data,
            "tdm_enabled",
            "tdm_virtual_adc",
            default=False,
        ),
        name="FMCW profile tdm_enabled",
    )
    carrier_frequency = _first(
        data,
        "carrier_frequency",
        "carrier_frequency_hz",
        "center_frequency_hz",
        "design_frequency_hz",
    )
    if carrier_frequency is None:
        start_frequency = _first(data, "start_frequency", "start_frequency_hz")
        bandwidth = _first(data, "bandwidth", "bandwidth_hz")
        if start_frequency is not None and bandwidth is not None:
            carrier_frequency = float(start_frequency) + 0.5 * float(bandwidth)
        elif board_spec is not None:
            carrier_frequency = float(board_spec.design_frequency_hz)
        else:
            raise ValueError(
                "FMCW profile missing carrier frequency; expected "
                "carrier_frequency_hz, center_frequency_hz, start_frequency_hz "
                "+ bandwidth_hz, or a known TI board model"
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
    if tdm_enabled and per_tx is not None:
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
        slope=float(
            _required_fmcw_value(data, "slope", "slope", "slope_hz_per_s")
        ),
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
        num_adc_samples=int(
            _required_fmcw_value(data, "num_adc_samples", "num_adc_samples")
        ),
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
        tdm_enabled=tdm_enabled,
    )


@dataclass(frozen=True)
class BundleSensorConfig:
    """Radar hardware, pose, and waveform metadata stored in a bundle."""

    board_model: str
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

    def __post_init__(self) -> None:
        if self.virtual_channel_order not in ("tx_major", "rx_major"):
            raise ValueError(
                "virtual_channel_order must be 'tx_major' or 'rx_major'"
            )

    @classmethod
    def from_json(cls, path: str | Path) -> "BundleSensorConfig":
        """Load radar hardware, waveform, and pose metadata from ``sensor.json``."""

        data = read_json(Path(path))
        if not isinstance(data, Mapping):
            raise ValueError("sensor.json must contain a JSON object")
        board_value = _first(data, "board_model", "hardware_model")
        if board_value is None:
            raise ValueError("sensor.json requires board_model")
        board_model = str(board_value)

        if "fmcw" in data:
            if not isinstance(data["fmcw"], Mapping):
                raise ValueError("sensor.json fmcw must be a JSON object")
            fmcw_data = dict(data["fmcw"])
        else:
            profile_key = str(
                _first(data, "fmcw_profile", "profile", default="")
            )
            if not profile_key:
                raise ValueError(
                    "sensor.json requires fmcw or fmcw_profile/profile"
                )
            defaults = _ti_board_defaults(board_model)
            if defaults is None:
                raise ValueError(
                    f"Unknown board model {board_model!r} cannot supply an FMCW profile"
                )
            profiles = defaults.metadata().get("fmcw_profiles", {})
            if profile_key not in profiles:
                raise ValueError(
                    f"Unknown FMCW profile {profile_key!r} for {board_model!r}"
                )
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
        cosine_3db_beamwidth_deg = float(
            _first(data, "cosine_3db_beamwidth_deg", default=60.0)
        )
        if (
            not np.isfinite(cosine_3db_beamwidth_deg)
            or not 0.0 < cosine_3db_beamwidth_deg < 180.0
        ):
            raise ValueError(
                "sensor.json cosine_3db_beamwidth_deg must be between "
                "0 and 180 degrees"
            )
        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("sensor.json metadata must be a JSON object")

        # ``tdm_enabled`` is the canonical runtime flag. Existing public
        # bundles predate it and declare their packed per-transmitter ADC
        # layout as ``metadata.tdm_virtual_adc``. An explicit FMCW or
        # top-level value takes precedence over that legacy marker.
        explicit_tdm = _first(
            fmcw_data,
            "tdm_enabled",
            default=_first(data, "tdm_enabled", default=None),
        )
        legacy_tdm = _first(
            metadata,
            "tdm_virtual_adc",
            default=_first(data, "tdm_virtual_adc", default=None),
        )
        if (
            legacy_tdm is None
            and metadata.get("source_adc_axes")
            == ["adc_sample", "chirp_loop", "rx", "tx"]
        ):
            legacy_tdm = True
        resolved_tdm = explicit_tdm if explicit_tdm is not None else legacy_tdm
        if resolved_tdm is not None:
            fmcw_data["tdm_enabled"] = resolved_tdm

        def antenna_indices(name: str) -> tuple[int, ...] | None:
            value = data.get(name)
            if value is None:
                return None
            if (
                not isinstance(value, (list, tuple))
                or not value
                or any(
                    isinstance(index, (bool, np.bool_))
                    or not isinstance(index, (int, np.integer))
                    for index in value
                )
            ):
                raise ValueError(
                    f"sensor.json {name} must be a non-empty integer array"
                )
            indices = tuple(int(index) for index in value)
            if any(index < 0 for index in indices):
                raise ValueError(
                    f"sensor.json {name} must contain non-negative indices"
                )
            if len(set(indices)) != len(indices):
                raise ValueError(
                    f"sensor.json {name} must not contain duplicates"
                )
            return indices

        return cls(
            board_model=board_model,
            fmcw=_fmcw_from_mapping(fmcw_data, board_model=board_model),
            position=position,
            orientation=orientation,
            name=str(_first(data, "name", default=radar_data.get("name", "radar"))),
            pattern_mode=str(_first(data, "pattern_mode", default="none")),
            cosine_3db_beamwidth_deg=cosine_3db_beamwidth_deg,
            tx_power_dbm=tx_power_value,
            virtual_channel_order=str(
                _first(data, "virtual_channel_order", default="tx_major")
            ),
            tx_indices_zero_based=antenna_indices(
                "tx_indices_zero_based"
            ),
            rx_indices_zero_based=antenna_indices(
                "rx_indices_zero_based"
            ),
            metadata=dict(metadata),
        )

    def to_sensor(self) -> RadarSensor:
        """Build the runtime radar sensor represented by this configuration."""

        hardware = RadarHardware.from_ti_board(
            self.board_model,
            pattern_mode=self.pattern_mode,
            cosine_half_power_angle_deg=(
                0.5 * self.cosine_3db_beamwidth_deg
            ),
        )
        hardware = _hardware_with_virtual_channel_order(
            hardware,
            self.virtual_channel_order,
        )
        if self.tx_indices_zero_based is not None:
            hardware = hardware.subset_tx(self.tx_indices_zero_based)
        if self.rx_indices_zero_based is not None:
            hardware = hardware.subset_rx(self.rx_indices_zero_based)
        return RadarSensor(
            name=self.name,
            hardware=hardware,
            fmcw=self.fmcw,
            position=self.position,
            orientation=self.orientation,
            tx_power_dbm=(
                get_ti_board_spec(self.board_model).tx_power_dbm
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
            raise ValueError(
                f"frames.json entry {index} must include motion_time_s"
            )
        motion_time_value = float(motion_time)
        if not np.isfinite(motion_time_value):
            raise ValueError(f"frames.json entry {index} motion time must be finite")
        bundle_index = int(
            _first(data, "bundle_index", "benchmark_index", default=index)
        )
        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"frames.json entry {index} metadata must be an object")
        return cls(
            bundle_index=bundle_index,
            motion_time_s=motion_time_value,
            radar_frame_id=_optional_str(
                _first(data, "radar_frame_id", "Radar_frameID")
            ),
            camera_frame_id=_optional_str(
                _first(data, "camera_frame_id", "camera_id")
            ),
            pose_frame_id=_optional_str(_first(data, "pose_frame_id", "pose_id")),
            dataset_frame_id=_optional_str(
                _first(data, "dataset_frame_id", "frame_id")
            ),
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
                else (
                    "[frames, chirps, samples, channels] or "
                    "[chirps, samples, channels]"
                )
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
        times = (
            np.array(archive["times"], copy=True)
            if "times" in archive.files
            else None
        )
    expected_times_shape = adc.shape[:-2]
    if times is not None:
        if times.shape != expected_times_shape:
            raise ValueError(f"{path} times must have shape {expected_times_shape}")
        if times.dtype.kind not in {"i", "u", "f"} or not np.all(np.isfinite(times)):
            raise ValueError(f"{path} times must contain finite real values")
    return adc, times


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
            f"Unsupported bundle profile {profile!r}; expected "
            f"{HERMES_BUNDLE_PROFILE!r}"
        )
    schema_version = int(descriptor.get("schema_version", 0))
    if schema_version != 1:
        raise ValueError("bundle.json schema_version must be 1")
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
            raise ValueError(
                f"bundle.json adcs.{name}.path must be a non-empty relative path"
            )
        relative_path = Path(path_value)
        if relative_path.is_absolute() or "\\" in path_value or any(
            part in {"", ".", ".."} or ":" in part
            for part in relative_path.parts
        ):
            raise ValueError(
                f"bundle.json adcs.{name}.path must be a safe relative path"
            )
        canonical_path = relative_path.as_posix()
        if canonical_path in adc_paths:
            raise ValueError(
                "bundle.json ADC artifacts must reference unique paths: "
                f"{path_value!r}"
            )
        adc_paths.add(canonical_path)
        origin = str(artifact.get("origin", "")).strip()
        if origin not in ADC_ARTIFACT_ORIGINS:
            raise ValueError(
                f"bundle.json adcs.{name}.origin must be one of "
                f"{sorted(ADC_ARTIFACT_ORIGINS)}"
            )
    primary_origin = str(artifacts[primary].get("origin", ""))
    if primary_origin not in {"measurement", "simulation"}:
        raise ValueError(
            "bundle.json primary ADC origin must be measurement or simulation"
        )
    return descriptor


def _safe_bundle_file(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or "\\" in value or any(
        part in {"", ".", ".."} or ":" in part for part in relative.parts
    ):
        raise ValueError(f"{label} must be a safe bundle-relative path")
    resolved_root = root.resolve(strict=True)
    path = resolved_root / relative
    cursor = resolved_root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(
                f"{label} must not traverse a symlink inside the bundle"
            )
    try:
        resolved_path = path.resolve(strict=True)
        resolved_path.relative_to(resolved_root)
    except FileNotFoundError:
        raise FileNotFoundError(f"Referenced bundle file was not found: {relative}")
    except ValueError as exc:
        raise ValueError(f"{label} escapes the bundle root") from exc
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Referenced bundle file was not found: {relative}")
    return resolved_path


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
                raise ValueError(
                    f"AMASS-like {timing_key} must match the pose frame count"
                )
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

    @classmethod
    def load(cls, root: str | Path) -> "Bundle":
        """Load the core contract and validate cross-file consistency."""

        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"Bundle directory was not found: {root}")
        descriptor = load_bundle_descriptor(root)
        profile = str(descriptor["profile"])
        sensor = BundleSensorConfig.from_json(root / "sensor.json")
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
            path = _safe_bundle_file(
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
            adc_paths[name] = path
            adc_origins[name] = origin
            loaded_adcs[name] = adc_value
            loaded_times[name] = adc_times
            if artifact.get("manifest") is not None:
                _safe_bundle_file(
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
            if motion.get("parameters") is not None:
                motion_path = _safe_bundle_file(
                    root,
                    motion.get("parameters"),
                    label="bundle.json motion.parameters",
                )
                preflight_motion_archive(motion_path)
            if motion.get("evaluated_mesh") is not None:
                evaluated_motion_path = _safe_bundle_file(
                    root,
                    motion.get("evaluated_mesh"),
                    label="bundle.json motion.evaluated_mesh",
                )
                preflight_mesh_sequence_archive(evaluated_motion_path)

        target = descriptor.get("target")
        if target is not None:
            if not isinstance(target, Mapping):
                raise ValueError("bundle.json target must be an object")
            if target.get("mesh") is not None:
                target_mesh_path = _safe_bundle_file(
                    root,
                    target.get("mesh"),
                    label="bundle.json target.mesh",
                )
                load_static_target_mesh(target_mesh_path)
            if target.get("obj") is not None:
                target_obj_path = _safe_bundle_file(
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
                scene_path = _safe_bundle_file(
                    root,
                    scene.get("file"),
                    label="bundle.json scene.file",
                )
            assets = scene.get("assets", [])
            if not isinstance(assets, list):
                raise ValueError("bundle.json scene.assets must be an array")
            for index, asset in enumerate(assets):
                _safe_bundle_file(
                    root,
                    asset,
                    label=f"bundle.json scene.assets[{index}]",
                )

        for group_name in ("manifests", "diagnostics"):
            group = descriptor.get(group_name, {})
            if not isinstance(group, Mapping):
                raise ValueError(f"bundle.json {group_name} must be an object")
            for name, path_value in group.items():
                _safe_bundle_file(
                    root,
                    path_value,
                    label=f"bundle.json {group_name}.{name}",
                )

        environment = None
        environment_value = descriptor.get("environment")
        if environment_value is not None:
            environment_path = _safe_bundle_file(
                root,
                environment_value,
                label="bundle.json environment",
            )
            value = read_json(environment_path)
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"{environment_path.name} must contain a JSON object"
                )
            environment = dict(value)

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
        if motion_path is not None:
            motion_times = _load_motion_times(motion_path)
            if np.any(frame_times < motion_times[0]) or np.any(
                frame_times > motion_times[-1]
            ):
                raise ValueError(
                    "frames.json motion times must lie within the "
                    f"{motion_path.name} timeline"
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
        """Whether the bundle declares parameterized human-motion inputs."""

        return self.motion_path is not None

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
            raise ValueError(
                f"Bundle does not contain a {normalized.upper()} ADC cube"
            )
        adc, _ = load_adc(paths[normalized])
        return adc


def _validate_background_shape(background: np.ndarray, adc_shape: tuple[int, ...]) -> None:
    if background.shape in {adc_shape, adc_shape[1:], (1, *adc_shape[1:])}:
        return
    raise ValueError(
        "background_adc.npz adc must have shape [F,M,N,C], [M,N,C], or [1,M,N,C]"
    )
