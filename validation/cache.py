# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Portable content-addressed simulated ADC caches."""

from __future__ import annotations

import hashlib
from importlib import resources
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
from mmWaveRadar.provenance import capture_provenance
from bundle_prepare.contract import Bundle as BenchmarkClip
from . import channels as _channels
from . import configuration as _configuration
from . import simulation as _simulation


def load_or_simulate_adcs(
    clip: BenchmarkClip,
    config: _configuration.BenchmarkRunConfig,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    if config.simulated_adc_path is not None:
        with np.load(config.simulated_adc_path, allow_pickle=False) as data:
            if "adc" not in data.files:
                raise ValueError(f"{config.simulated_adc_path} must contain adc")
            adc = np.array(data["adc"], copy=True)
        return {"sim": adc}, {
            "sim": {
                "status": "provided",
                "path": str(config.simulated_adc_path),
            }
        }
    if not config.simulate:
        raise ValueError("simulate=False requires simulated_adc_path")
    sim_adcs: dict[str, np.ndarray] = {}
    sim_caches: dict[str, dict] = {}
    for label, solver in labeled_solvers(config.solvers):
        adc, cache = load_or_simulate_cached_adc(
            clip,
            config,
            label=label,
            solver=solver,
        )
        sim_adcs[label] = adc
        sim_caches[label] = cache
    return sim_adcs, sim_caches


def labeled_solvers(solvers: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    solvers = tuple(str(solver) for solver in solvers)
    if len(solvers) == 1:
        return (("sim", solvers[0]),)
    return tuple((solver, solver) for solver in solvers)


def load_or_simulate_cached_adc(
    clip: BenchmarkClip,
    config: _configuration.BenchmarkRunConfig,
    *,
    label: str,
    solver: str,
) -> tuple[np.ndarray, dict]:
    cache_path = simulated_adc_cache_path(config, label)
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                if "adc" in data.files and simulated_adc_cache_matches(
                    data,
                    clip,
                    config,
                    solver=solver,
                ):
                    return np.array(data["adc"], copy=True), {
                        "status": "hit",
                        "path": str(cache_path),
                        "solver": str(solver),
                        "provenance": (
                            json.loads(str(data["provenance_json"].item()))
                            if "provenance_json" in data
                            else None
                        ),
                    }
        except (
            EOFError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
            zipfile.BadZipFile,
        ) as exc:
            if config.progress:
                print(
                    "[validation] rebuilding unreadable simulated-ADC cache "
                    f"{cache_path}: {type(exc).__name__}",
                    flush=True,
                )
    adc = _simulation.simulate_adc(clip, config, solver=solver)
    write_simulated_adc_cache(
        cache_path,
        adc,
        clip,
        config,
        solver=solver,
    )
    return adc, {
        "status": "miss_rebuilt",
        "path": str(cache_path),
        "solver": str(solver),
    }


def simulated_adc_cache_path(config: _configuration.BenchmarkRunConfig, label: str) -> Path:
    return Path(config.out_dir) / f"simulated_adc_{label}.npz"


def simulated_adc_cache_metadata(
    clip: BenchmarkClip,
    config: _configuration.BenchmarkRunConfig,
    *,
    solver: str,
) -> dict[str, object]:
    scene_path = _simulation.resolve_scene_path(clip, config)
    selection = _channels.subarray_selection(
        clip,
        config,
        full_channels=int(np.asarray(clip.real_adc).shape[3]),
    )
    source_shape = tuple(int(v) for v in np.asarray(clip.real_adc).shape)
    adc_shape = (
        source_shape[:-1] + (int(selection.channel_count),) if selection.active else source_shape
    )
    metadata = {
        "cache_schema_version": 9,
        "adc_shape": adc_shape,
        "source_adc_shape": source_shape,
        "board_model": str(clip.sensor_config.board_model),
        "num_frames": int(clip.num_frames),
        "solver": str(solver),
        "scene_path": cache_resource_name(scene_path, clip.root),
        **_simulation.solver_cache_settings(config, solver),
        "ignore_tdm_timing": bool(config.ignore_tdm_timing),
        "amass_sequence_path": cache_resource_name(clip.motion_path, clip.root),
        "selected_tx_indices": _channels.index_tuple_token(config.selected_tx_indices),
        "selected_rx_indices": _channels.index_tuple_token(config.selected_rx_indices),
        "subarray_channel_indices": _channels.index_tuple_token(
            tuple(int(v) for v in selection.channel_indices)
        ),
    }
    metadata["input_fingerprint"] = simulated_adc_input_fingerprint(
        clip,
        config,
        solver=solver,
        scene_path=scene_path,
        selection=selection,
        source_shape=source_shape,
        adc_shape=adc_shape,
    )
    return metadata


def simulated_adc_input_fingerprint(
    clip: BenchmarkClip,
    config: _configuration.BenchmarkRunConfig,
    *,
    solver: str,
    scene_path: str | None,
    selection: _channels.SubarraySelection,
    source_shape: tuple[int, ...],
    adc_shape: tuple[int, ...],
) -> str:
    """Hashes clip inputs and simulation controls that affect cached ADC output."""

    root = Path(clip.root)
    payload = {
        "cache_schema": 9,
        "source_adc_shape": list(source_shape),
        "adc_shape": list(adc_shape),
        "solver": str(solver),
        "scene_path": cache_resource_name(scene_path, root),
        "scene_inputs": scene_input_fingerprint(scene_path),
        "smpl_model": smpl_model_fingerprint(clip, config),
        "ti_digitized_pattern": ti_digitized_pattern_fingerprint(clip),
        "simulation_config": {
            **_simulation.solver_cache_settings(config, solver),
            "ignore_tdm_timing": bool(config.ignore_tdm_timing),
            "selected_tx_indices": _channels.index_tuple_token(config.selected_tx_indices),
            "selected_rx_indices": _channels.index_tuple_token(config.selected_rx_indices),
            "subarray_channel_indices": _channels.index_tuple_token(
                tuple(int(v) for v in selection.channel_indices)
            ),
        },
        "sensor_config": sensor_config_cache_payload(clip),
        "frames": frames_cache_payload(clip),
        "clip_files": {
            "sensor_json": file_fingerprint(root / "sensor.json"),
            "frames_json": file_fingerprint(root / "frames.json"),
            "amass_sequence_npz": file_fingerprint(clip.motion_path),
            "evaluated_motion_npz": file_fingerprint(clip.evaluated_motion_path),
            "target_mesh": file_fingerprint(clip.target_mesh_path),
            "target_obj": file_fingerprint(clip.target_obj_path),
            "environment_json": file_fingerprint(root / "environment.json"),
        },
    }
    return stable_json_hash(payload)


def sensor_config_cache_payload(clip: BenchmarkClip) -> dict[str, object]:
    """Returns deterministic sensor metadata for ADC-cache fingerprinting."""

    sensor = clip.sensor_config
    fmcw = sensor.fmcw
    return {
        "board_model": str(sensor.board_model),
        "name": str(sensor.name),
        "position": [float(v) for v in sensor.position],
        "orientation": [float(v) for v in sensor.orientation],
        "pattern_mode": str(sensor.pattern_mode),
        "tx_power_dbm": (None if sensor.tx_power_dbm is None else float(sensor.tx_power_dbm)),
        "virtual_channel_order": str(sensor.virtual_channel_order),
        "metadata": jsonable(sensor.metadata or {}),
        "array": jsonable(sensor.array),
        "tdm_tx_order": jsonable(sensor.tdm_tx_order),
        "channel_map": jsonable(sensor.channel_map),
        "tx_indices_zero_based": jsonable(sensor.tx_indices_zero_based),
        "rx_indices_zero_based": jsonable(sensor.rx_indices_zero_based),
        "fmcw": {
            "carrier_frequency_hz": float(fmcw.carrier_frequency),
            "slope_hz_per_s": float(fmcw.slope),
            "chirp_duration_s": float(fmcw.chirp_duration),
            "chirp_repetition_time_s": float(fmcw.chirp_repetition_time),
            "sampling_frequency_hz": float(fmcw.sampling_frequency),
            "adc_mode": fmcw.adc_mode,
            "if_bandwidth_hz": fmcw.if_bandwidth_hz,
            "num_adc_samples": int(fmcw.num_adc_samples),
            "num_chirps_per_frame": int(fmcw.num_chirps_per_frame),
            "frame_period_s": float(fmcw.frame_period),
            "num_tx": int(fmcw.num_tx),
            "tx_schedule": str(fmcw.tx_schedule),
        },
    }


def frames_cache_payload(clip: BenchmarkClip) -> list[dict[str, object]]:
    """Returns deterministic frame metadata for ADC-cache fingerprinting."""

    return [
        {
            "benchmark_index": int(frame.benchmark_index),
            "motion_time_s": float(frame.motion_time_s),
            "radar_frame_id": frame.radar_frame_id,
            "camera_frame_id": frame.camera_frame_id,
            "pose_frame_id": frame.pose_frame_id,
            "dataset_frame_id": frame.dataset_frame_id,
            "metadata": jsonable(frame.metadata or {}),
        }
        for frame in clip.frames
    ]


def array_fingerprint(value: np.ndarray) -> dict[str, object]:
    """Returns a shape, dtype, and content hash for a numeric array."""

    arr = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(json.dumps(list(arr.shape)).encode("utf-8"))
    digest.update(arr.view(np.uint8))
    return {
        "dtype": str(arr.dtype),
        "shape": [int(v) for v in arr.shape],
        "sha256": digest.hexdigest(),
    }


def file_fingerprint(path: str | Path | None) -> str:
    """Returns a stable fingerprint for a file path or missing-file state."""

    if path is None:
        return "none"
    candidate = Path(path)
    try:
        candidate.stat()
    except FileNotFoundError:
        return "missing"
    if not candidate.is_file():
        return "not_file"
    digest = hashlib.sha256()
    with candidate.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def cache_resource_name(path: str | Path | None, root: str | Path) -> str:
    """Name a resource independently of the bundle's extraction directory."""
    if path is None:
        return ""
    candidate = Path(path).expanduser().resolve()
    root = Path(root).expanduser().resolve()
    if candidate.is_relative_to(root):
        return candidate.relative_to(root).as_posix()
    return "external:" + candidate.name


def scene_input_fingerprint(path: str | Path | None) -> str:
    """Hashes a scene XML and the local files it references recursively."""

    if path is None:
        return "none"
    root_path = Path(path).expanduser().resolve()
    pending = [root_path]
    visited: set[Path] = set()
    inputs: dict[str, str] = {}
    while pending:
        candidate = pending.pop()
        candidate = candidate.expanduser().resolve()
        if candidate in visited:
            continue
        visited.add(candidate)
        inputs[os.path.relpath(candidate, root_path.parent)] = file_fingerprint(candidate)
        if candidate.suffix.lower() != ".xml" or not candidate.is_file():
            continue
        try:
            document = ET.parse(candidate)
        except (ET.ParseError, OSError):
            continue
        for reference in scene_xml_file_references(document.getroot()):
            referenced_path = Path(reference).expanduser()
            if not referenced_path.is_absolute():
                referenced_path = candidate.parent / referenced_path
            pending.append(referenced_path)
    return stable_json_hash(inputs)


def scene_xml_file_references(root: ET.Element) -> tuple[str, ...]:
    """Returns Mitsuba/Sionna XML filename references in document order."""

    references: list[str] = []
    for element in root.iter():
        filename = element.attrib.get("filename")
        if filename:
            references.append(str(filename))
        if (
            str(element.tag).rsplit("}", 1)[-1] == "string"
            and str(element.attrib.get("name", "")).casefold() == "filename"
            and element.attrib.get("value")
        ):
            references.append(str(element.attrib["value"]))
    return tuple(references)


def smpl_model_fingerprint(
    clip: BenchmarkClip,
    config: _configuration.BenchmarkRunConfig,
) -> str:
    """Hashes the effective SMPL-family model files used by this bundle."""

    if clip.motion_path is None:
        return "not_used"
    model_dir = config.smpl_model_dir
    model_type = "smpl"
    gender = "neutral"
    try:
        with np.load(clip.motion_path, allow_pickle=False) as data:
            model_dir = model_dir or _simulation.optional_npz_string(data, "smpl_model_dir")
            model_type = _simulation.optional_npz_string(data, "model_type") or model_type
            gender = _simulation.optional_npz_string(data, "gender") or gender
    except (OSError, ValueError):
        # The motion archive itself is fingerprinted separately and the normal
        # loader will report malformed motion with its more specific error.
        pass
    if not model_dir:
        return "unconfigured"

    root = Path(model_dir).expanduser().resolve()
    if root.is_file():
        return stable_json_hash({root.name: file_fingerprint(root)})

    model_root = root / str(model_type).casefold()
    if not model_root.is_dir():
        return stable_json_hash({model_root.name: file_fingerprint(model_root)})

    model_files = sorted(
        path
        for path in model_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".pkl", ".npz"}
    )
    gender_token = str(gender).casefold()
    gender_files = [path for path in model_files if gender_token in path.stem.casefold()]
    selected = gender_files or model_files
    if not selected:
        return stable_json_hash({model_root.name: "no_model_files"})
    return stable_json_hash(
        {str(path.relative_to(root)): file_fingerprint(path) for path in selected}
    )


def ti_digitized_pattern_fingerprint(clip: BenchmarkClip) -> str:
    """Hashes the effective optional TI pattern asset when it affects the run."""

    mode = str(clip.sensor_config.pattern_mode).casefold().replace("-", "_")
    if mode != "digitized":
        return "not_applicable"

    from mmWaveRadar.radar import get_ti_board_spec

    spec = get_ti_board_spec(str(clip.sensor_config.board_model))
    if spec.pattern_asset_key is None:
        return "board_has_no_digitized_pattern"

    configured = os.environ.get("MMWAVE_TI_PATTERN_NPZ")
    if configured:
        return file_fingerprint(Path(configured).expanduser().resolve())

    asset = resources.files("mmWaveRadar.radar").joinpath(
        "data",
        "ti_digitized_patterns.npz",
    )
    if not asset.is_file():
        return "packaged_asset_missing"
    digest = hashlib.sha256()
    with asset.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def stable_json_hash(value: object) -> str:
    """Hashes JSON-compatible data after deterministic key ordering."""

    text = json.dumps(
        jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def jsonable(value: object) -> object:
    """Converts common Python/Numpy values into JSON-serializable values."""

    if isinstance(value, Mapping):
        return {
            str(key): jsonable(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def simulated_adc_cache_matches(
    data: np.lib.npyio.NpzFile,
    clip: BenchmarkClip,
    config: _configuration.BenchmarkRunConfig,
    *,
    solver: str,
) -> bool:
    expected = simulated_adc_cache_metadata(
        clip,
        config,
        solver=solver,
    )
    adc = np.asarray(data["adc"])
    if tuple(int(v) for v in adc.shape) != expected["adc_shape"]:
        return False
    if not np.issubdtype(adc.dtype, np.number) or not np.all(np.isfinite(adc)):
        return False
    for key, value in expected.items():
        if key not in data.files:
            return False
        stored = np.asarray(data[key])
        if key == "adc_shape":
            if tuple(int(v) for v in stored.reshape(-1)) != value:
                return False
        elif key == "source_adc_shape":
            if tuple(int(v) for v in stored.reshape(-1)) != value:
                return False
        elif isinstance(value, bool):
            if bool(stored.item()) != value:
                return False
        elif isinstance(value, int):
            if int(stored.item()) != value:
                return False
        else:
            if str(stored.item()) != str(value):
                return False
    return True


def write_simulated_adc_cache(
    path: Path,
    adc: np.ndarray,
    clip: BenchmarkClip,
    config: _configuration.BenchmarkRunConfig,
    *,
    solver: str,
) -> None:
    metadata = simulated_adc_cache_metadata(
        clip,
        config,
        solver=solver,
    )
    metadata["provenance_json"] = json.dumps(
        capture_provenance(
            settings={"solver": solver, **config.effective_solver_settings},
            seed=config.effective_solver_settings["rt_seed"],
            input_hashes={"simulation_inputs": metadata["input_fingerprint"]},
            runtime={"adc_dtype": str(adc.dtype)},
        )
    )
    payload = {"adc": np.asarray(adc)}
    for key, value in metadata.items():
        dtype = np.int64 if key in {"adc_shape", "source_adc_shape"} else None
        payload[key] = np.asarray(value, dtype=dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".npz",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        np.savez_compressed(temporary_path, **payload)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
