# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Headless Dynamic Scenes runner with the GUI's radar and solver defaults."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from ..experiments import (
    PREPARED_ROOM_PRESETS,
    load_amass_motion,
    load_scene_xml,
    motion_source_to_radar_transform,
    radar_frame_count,
    run_human_room_experiment,
)
from ..simulation.subarray import local_array_groups
from ..naming import (
    ARRAY_MODELS,
    HYBRID_CALIBRATIONS,
    RT_HUMAN_BOUNCES,
    SOLVERS,
    TX_SCHEDULES,
    add_removed_options,
    parse_index_spec,
    renamed_setting_message,
    validate_array_groups,
    validate_array_model,
    validate_solver,
)
from ..radar import FMCWConfig, get_ti_board_spec
from ..simulation.physical_optics import default_phase_span_scale_rad
from .app import (
    _PO_INTEGRATION_MODES,
    _RADAR_ACTIVE_ANTENNA_DEFAULTS,
    _SOLVER_SETTINGS_DEFAULTS,
    _SOLVER_SETTINGS_NUMERIC_BOUNDS,
    _default_amass_motion_path,
    _default_smpl_model_dir,
    _human_room_export,
    _human_room_running_status,
)

# CLI units match the GUI; JSON configuration keys use the experiment API's SI units.
_NUMERIC_OPTIONS = (
    ("--yaw", "human_yaw_deg", float, 1.0, "Human yaw [deg]"),
    ("--carrier-ghz", "carrier_frequency_hz", float, 1e9, "Carrier frequency [GHz]"),
    ("--slope-mhz-us", "slope_hz_per_s", float, 1e12, "Chirp slope [MHz/us]"),
    ("--chirp-duration-us", "chirp_duration_s", float, 1e-6, "Chirp duration [us]"),
    ("--chirp-repetition-us", "chirp_repetition_time_s", float, 1e-6, "Chirp repetition [us]"),
    ("--sampling-msps", "sampling_frequency_hz", float, 1e6, "ADC rate [MS/s]"),
    ("--if-bandwidth-mhz", "if_bandwidth_hz", float, 1e6, "IF cap [MHz]; 0 uses board default"),
    ("--adc-samples", "num_adc_samples", int, 1, "ADC samples per chirp"),
    ("--chirps", "num_chirps_per_frame", int, 1, "Chirps per frame (per Tx in TDM)"),
    ("--frame-period-ms", "frame_period_s", float, 1e-3, "Frame period [ms]"),
    ("--beamwidth-deg", "cosine_3db_beamwidth_deg", float, 1.0, "Full cosine 3 dB width [deg]"),
    # Solver settings use their configuration key with dashes.
    (
        "--rt-human-diffuse-coefficient",
        "rt_human_diffuse_coefficient",
        float,
        1.0,
        "Human RT diffuse coefficient",
    ),
    ("--rt-samples-per-source", "rt_samples_per_source", int, 1, "RT rays per source"),
    (
        "--rt-max-paths-per-source",
        "rt_max_paths_per_source",
        int,
        1,
        "RT retained paths per source",
    ),
    ("--rt-max-depth", "rt_max_depth", int, 1, "RT maximum interaction depth"),
    (
        "--rt-refresh-frames",
        "rt_refresh_frames",
        int,
        1,
        "RT retrace every N frames; 0 = every chirp",
    ),
    (
        "--rt-visibility-refresh-frames",
        "rt_visibility_refresh_frames",
        int,
        1,
        "RT path blocking test every N frames; 0 = every chirp",
    ),
    (
        "--rt-visibility-fade-frames",
        "rt_visibility_fade_frames",
        int,
        1,
        "RT blocking fade [frames]; 0 = step",
    ),
    (
        "--po-quadrature-phase-span-scale-rad",
        "po_quadrature_phase_span_scale_rad",
        float,
        1.0,
        "PO phase span [rad]",
    ),
    (
        "--po-curvature-error-rad",
        "po_curvature_error_rad",
        float,
        1.0,
        "PO curvature tolerance [rad]",
    ),
    (
        "--po-quadrature-max-refinement-depth",
        "po_quadrature_max_refinement_depth",
        int,
        1,
        "PO subdivision depth",
    ),
    (
        "--po-quadrature-max-subfaces-per-parent",
        "po_quadrature_max_subfaces_per_parent",
        int,
        1,
        "PO children per parent",
    ),
    (
        "--po-adaptive-visibility-edge-margin",
        "po_adaptive_visibility_edge_margin",
        float,
        1.0,
        "PO grazing cosine margin",
    ),
    (
        "--po-visibility-samples-per-face",
        "po_visibility_samples_per_face",
        int,
        1,
        "PO visibility samples per face",
    ),
    (
        "--po-refresh-frames",
        "po_refresh_frames",
        int,
        1,
        "PO exact re-integration every N frames; 0 = every chirp",
    ),
    (
        "--po-refresh-on-phase-error-rad",
        "po_refresh_on_phase_error_rad",
        float,
        1.0,
        "PO neglected vertex phase-change trigger [rad]",
    ),
    (
        "--po-refresh-on-area-fraction",
        "po_refresh_on_area_fraction",
        float,
        1.0,
        "PO relative area-change trigger",
    ),
    (
        "--po-refresh-on-normal-deg",
        "po_refresh_on_normal_deg",
        float,
        1.0,
        "PO normal-change trigger [deg]",
    ),
    (
        "--po-visibility-refresh-frames",
        "po_visibility_refresh_frames",
        int,
        1,
        "PO facet visibility every N frames; 0 = every chirp",
    ),
    (
        "--po-visibility-fade-frames",
        "po_visibility_fade_frames",
        int,
        1,
        "PO visibility fade [frames]; 0 = step",
    ),
    (
        "--hybrid-reflector-cutoff-db",
        "hybrid_reflector_cutoff_db",
        float,
        1.0,
        "Reflector relative-power cutoff [dB]",
    ),
    (
        "--hybrid-max-reflectors",
        "hybrid_max_reflectors",
        int,
        1,
        "Maximum image-source reflectors",
    ),
)
_REMOVED_OPTIONS = {
    "--po-refresh-on-centroid-m": "--po-refresh-on-phase-error-rad",
    "--modes": "--solvers (po, rt, hybrid_po); the RT retrace cadence is --rt-refresh-frames",
    "--tdm": "--tx-schedule tdm",
    "--no-tdm": "--tx-schedule simultaneous",
    "--coupling": "--hybrid-coupling",
    "--no-coupling": "--no-hybrid-coupling",
    "--human-diffuse": "--rt-human-diffuse-coefficient",
    "--rt-samples": "--rt-samples-per-source",
    "--rt-path-cap": "--rt-max-paths-per-source",
    "--po-integration": "--po-integration-mode",
    "--tx": "--tx-indices, which is zero-based (TI's TX1 is 0)",
    "--rx": "--rx-indices, which is zero-based (TI's RX1 is 0)",
    "--po-phase-span-rad": "--po-quadrature-phase-span-scale-rad",
    "--po-max-depth": "--po-quadrature-max-refinement-depth",
    "--po-max-subfaces": "--po-quadrature-max-subfaces-per-parent",
    "--po-visibility-samples": "--po-visibility-samples-per-face",
    "--coupling-cutoff-db": "--hybrid-reflector-cutoff-db",
    "--coupling-max-reflectors": "--hybrid-max-reflectors",
}
_CHOICES = {
    "room_preset": PREPARED_ROOM_PRESETS,
    "adc_mode": ("complex1x", "complex2x", "real"),
    "antenna_pattern_mode": ("cosine", "none", "digitized"),
    "po_integration_mode": _PO_INTEGRATION_MODES,
    "tx_schedule": TX_SCHEDULES,
    "array_model": ARRAY_MODELS,
    "rt_human_bounces": RT_HUMAN_BOUNCES,
    "hybrid_calibration": HYBRID_CALIBRATIONS,
    "compute_backend": ("auto", "numpy", "torch"),
    "motion_source_handedness": ("right", "left"),
    "motion_source_length_unit": ("m", "cm", "mm"),
}
_INPUT_PATH_KEYS = ("human_motion", "smpl_model_dir", "scene_xml")
_PATH_KEYS = (*_INPUT_PATH_KEYS, "output")
_RUN_CONTROL_KEYS = {"schema_version", "solvers", "output", "quiet", "force"}
_NON_CONFIG_ARGUMENTS = {
    "human_motion",
    "human_motion_filename",
    "scene_xml_filename",
    "solver",
    "cancel_check",
    "progress_callback",
}
_SCHEMA_VERSION = 3
_LEGACY_WINDOW_KEYS = ("begin_frame_index", "end_frame_index")
_LEGACY_WINDOW_MESSAGE = (
    "schema_version 1 selected motion poses with begin_frame_index/end_frame_index "
    "and stretched them over radar frames. Later versions use motion time: set "
    "motion_start_s (seconds after the first pose) and either motion_end_s or "
    "num_frames (radar frames). For pose index i in a motion recorded at R frames "
    f"per second, motion_start_s = i / R. Then set schema_version to {_SCHEMA_VERSION}."
)


def _check_schema(supplied: dict) -> None:
    """Explain what an older configuration must change."""

    version = supplied.get("schema_version", _SCHEMA_VERSION)
    if version == 1 or any(key in supplied for key in _LEGACY_WINDOW_KEYS):
        raise ValueError(_LEGACY_WINDOW_MESSAGE)
    renamed = [
        message
        for key in sorted(supplied)
        if (message := renamed_setting_message(key)) is not None
    ]
    if renamed:
        raise ValueError(
            "configuration uses removed setting names: "
            + "; ".join(renamed)
            + f". Rename them and set schema_version to {_SCHEMA_VERSION}."
        )
    if version == 2:
        raise ValueError(
            f"schema_version 2 predates the solver setting names; set "
            f"schema_version to {_SCHEMA_VERSION}"
        )


def default_config(board_model: str = "IWR6843AOPEVM") -> dict:
    """Return JSON-compatible defaults without building Panel or running a solver."""

    settings = {
        name: parameter.default
        for name, parameter in inspect.signature(run_human_room_experiment).parameters.items()
        if name not in _NON_CONFIG_ARGUMENTS
        and parameter.kind is not inspect.Parameter.VAR_KEYWORD
    }
    spec = get_ti_board_spec(board_model)
    active = _RADAR_ACTIVE_ANTENNA_DEFAULTS.get(spec.key, {})
    motion = _default_amass_motion_path()
    settings.update(
        _SOLVER_SETTINGS_DEFAULTS,
        schema_version=_SCHEMA_VERSION,
        human_motion=str(motion) if motion is not None else None,
        smpl_model_dir=str(_default_smpl_model_dir()),
        board_model=spec.key,
        carrier_frequency_hz=spec.design_frequency_hz,
        selected_tx_indices=list(active.get("tx", range(spec.num_tx))),
        selected_rx_indices=list(active.get("rx", range(spec.num_rx))),
        solvers=["hybrid_po"],
        compute_backend="auto",
        output=str(Path("tutorial-bedroom-bundle.zip").resolve()),
        quiet=False,
        force=False,
    )
    return json.loads(json.dumps(settings))


def _index_group(text: str) -> list[int]:
    """Parses one antenna group such as ``0-2,9``."""

    try:
        return list(parse_index_spec(text))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def build_parser() -> argparse.ArgumentParser:
    """Describe every Dynamic Scenes simulation input and shared Settings field."""

    parser = argparse.ArgumentParser(
        description="Run the GUI's Dynamic Scenes workflow without a browser.",
        argument_default=argparse.SUPPRESS,
        epilog=(
            "Precedence: GUI factory defaults < --config JSON < command-line flags. "
            "Use --dry-run to see all resolved defaults in SI units. A frame is one "
            "radar frame; motion is placed by time in seconds after its first pose. "
            "Antenna indices are zero-based everywhere in HERMES (TI's TX1 is 0)."
        ),
    )
    parser.add_argument(
        "--config", type=Path, metavar="JSON", help="JSON simulation configuration"
    )
    parser.add_argument(
        "--write-config", type=Path, metavar="JSON", help="save the resolved JSON configuration"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print configuration without loading motion or simulating",
    )
    parser.add_argument(
        "--motion",
        dest="human_motion",
        type=Path,
        metavar="NPZ",
        help="mesh or AMASS-like NPZ; default: GUI example",
    )
    parser.add_argument(
        "--scene",
        dest="scene_xml",
        type=Path,
        metavar="XML",
        help="environment XML overriding the tutorial bedroom",
    )
    parser.add_argument("--room-preset", dest="room_preset", choices=PREPARED_ROOM_PRESETS)
    parser.add_argument(
        "--smpl-model-dir",
        dest="smpl_model_dir",
        type=Path,
        metavar="DIR",
        help="licensed model directory for pose-only NPZ",
    )
    parser.add_argument(
        "--position",
        dest="human_position_m",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="initial mesh bounding-box center [m]: +X forward, +Y left, +Z up",
    )
    parser.add_argument(
        "--radar-orientation",
        dest="radar_orientation_deg",
        nargs=3,
        type=float,
        metavar=("YAW", "PITCH", "ROLL"),
        help="radar orientation [deg]; phase center remains (0,0,0)",
    )
    parser.add_argument(
        "--start-s",
        dest="motion_start_s",
        type=float,
        metavar="SECONDS",
        help="motion time at the start of radar frame 0 [s after the first pose]",
    )
    end = parser.add_mutually_exclusive_group()
    end.add_argument(
        "--end-s",
        dest="motion_end_s",
        type=float,
        metavar="SECONDS",
        help="simulate every radar frame whose chirps end by this motion time [s]",
    )
    end.add_argument(
        "--num-frames",
        dest="num_frames",
        type=int,
        metavar="N",
        help="number of radar frames to simulate",
    )
    parser.add_argument(
        "--solvers",
        dest="solvers",
        nargs="+",
        choices=SOLVERS,
        help="solvers to run; the bundle keeps each one",
    )
    parser.add_argument(
        "--board", dest="board_model", help="TI board model; default: IWR6843AOPEVM"
    )
    parser.add_argument(
        "--tx-indices",
        dest="selected_tx_indices",
        nargs="+",
        type=int,
        metavar="N",
        help="active Tx indices, zero-based (e.g. 0 2)",
    )
    parser.add_argument(
        "--rx-indices",
        dest="selected_rx_indices",
        nargs="+",
        type=int,
        metavar="N",
        help="active Rx indices, zero-based (e.g. 0 1 3)",
    )
    parser.add_argument(
        "--tx-schedule",
        dest="tx_schedule",
        choices=TX_SCHEDULES,
        help="Tx antennas transmit together or take turns (TDM-MIMO)",
    )
    parser.add_argument(
        "--array-model",
        dest="array_model",
        choices=ARRAY_MODELS,
        help="phase_center solves once at the array centres; subarray once per Tx/Rx group pair",
    )
    parser.add_argument(
        "--array-tx-groups",
        dest="array_tx_groups",
        nargs="+",
        type=_index_group,
        metavar="GROUP",
        help="Tx groups for --array-model subarray, zero-based (e.g. 0-1 2)",
    )
    parser.add_argument(
        "--array-rx-groups",
        dest="array_rx_groups",
        nargs="+",
        type=_index_group,
        metavar="GROUP",
        help="Rx groups for --array-model subarray, zero-based (e.g. 0-1 2-3)",
    )
    parser.add_argument("--adc-mode", dest="adc_mode", choices=_CHOICES["adc_mode"])
    parser.add_argument(
        "--antenna-pattern", dest="antenna_pattern_mode", choices=_CHOICES["antenna_pattern_mode"]
    )
    parser.add_argument(
        "--po-integration-mode", dest="po_integration_mode", choices=_PO_INTEGRATION_MODES
    )
    parser.add_argument(
        "--rt-human-bounces",
        dest="rt_human_bounces",
        choices=RT_HUMAN_BOUNCES,
        help="keep RT paths that touch the human once, or any number of times",
    )
    parser.add_argument(
        "--po-adaptive-visibility-sampling",
        action=argparse.BooleanOptionalAction,
        help="Sample extra visibility points near edges and grazing faces",
    )
    parser.add_argument(
        "--hybrid-coupling",
        dest="hybrid_coupling",
        action=argparse.BooleanOptionalAction,
        help="Hybrid PO human-room coupling paths",
    )
    parser.add_argument(
        "--hybrid-calibration",
        dest="hybrid_calibration",
        choices=HYBRID_CALIBRATIONS,
        help="Hybrid PO amplitude calibration against RT",
    )
    parser.add_argument("--backend", dest="compute_backend", choices=_CHOICES["compute_backend"])
    axes = tuple(f"{sign}{axis}" for axis in "xyz" for sign in ("+", "-"))
    parser.add_argument(
        "--source-up",
        dest="motion_source_up_axis",
        choices=axes,
        help="manual source up axis; specify both axes (negative axes: --source-up=-y)",
    )
    parser.add_argument("--source-forward", dest="motion_source_forward_axis", choices=axes)
    parser.add_argument(
        "--source-handedness", dest="motion_source_handedness", choices=("right", "left")
    )
    parser.add_argument(
        "--source-units", dest="motion_source_length_unit", choices=("m", "cm", "mm")
    )
    for flag, destination, kind, _scale, help_text in _NUMERIC_OPTIONS:
        parser.add_argument(
            flag,
            dest=destination,
            type=kind,
            metavar="N" if kind is int else "VALUE",
            help=help_text,
        )
    parser.add_argument(
        "--output",
        type=Path,
        metavar="ZIP",
        help="output bundle ZIP; default: tutorial-bedroom-bundle.zip",
    )
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        help="allow replacing output/config files",
    )
    parser.add_argument(
        "--quiet", action=argparse.BooleanOptionalAction, help="hide solver progress"
    )
    add_removed_options(parser, _REMOVED_OPTIONS)
    return parser


def _path(value, base: Path) -> str | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise ValueError("input paths must be strings or null")
    path = Path(value).expanduser()
    return str((base / path).resolve())


def resolve_config(args: argparse.Namespace) -> dict:
    """Merge a strict, versioned JSON configuration and unit-converted CLI values."""

    cli = vars(args)
    supplied = {}
    if "config" in cli:
        path = cli["config"].expanduser().resolve()
        if path.stat().st_size > 1024**2:
            raise ValueError("configuration exceeds the 1 MiB limit")
        supplied = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(supplied, dict):
            raise ValueError("configuration must be a JSON object")
        supplied = dict(supplied)
        for key in _PATH_KEYS:
            if key in supplied:
                supplied[key] = _path(supplied[key], path.parent)
    _check_schema(supplied)
    config = default_config(cli.get("board_model", supplied.get("board_model", "IWR6843AOPEVM")))
    unknown = set(supplied) - set(config)
    if unknown:
        raise ValueError("unknown configuration keys: " + ", ".join(sorted(unknown)))
    config.update(supplied)
    overrides = {key: value for key, value in cli.items() if key in config}
    for _flag, key, _kind, scale, _help in _NUMERIC_OPTIONS:
        if key in overrides:
            overrides[key] *= scale
    for key in _PATH_KEYS:
        if key in overrides:
            overrides[key] = _path(overrides[key], Path.cwd())
    config.update(overrides)
    # A command-line end replaces the file's end, whichever form either uses.
    if "num_frames" in overrides:
        config["motion_end_s"] = None
    elif "motion_end_s" in overrides:
        config["num_frames"] = None
    if config["if_bandwidth_hz"] == 0 and not isinstance(config["if_bandwidth_hz"], bool):
        config["if_bandwidth_hz"] = None
    explicit = {**supplied, **overrides}
    if "po_integration_mode" in explicit and "po_quadrature_phase_span_scale_rad" not in explicit:
        config["po_quadrature_phase_span_scale_rad"] = default_phase_span_scale_rad(
            config["po_integration_mode"]
        )
    _validate_config(config)
    return config


def _validate_config(config: dict) -> None:
    if type(config["schema_version"]) is not int or config["schema_version"] != _SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {_SCHEMA_VERSION}")
    spec = get_ti_board_spec(config["board_model"])
    config["board_model"] = spec.key
    for key, choices in _CHOICES.items():
        if config[key] not in choices:
            raise ValueError(f"{key} must be one of {', '.join(choices)}")
    for key in ("po_adaptive_visibility_sampling", "hybrid_coupling", "quiet", "force"):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    if not isinstance(config["output"], str) or not config["output"]:
        raise ValueError("output must be a nonempty file path")
    for _flag, key, kind, _scale, _help in _NUMERIC_OPTIONS:
        value = config[key]
        if key == "if_bandwidth_hz" and value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (kind is int and type(value) is not int)
        ):
            raise ValueError(f"{key} must be a finite {kind.__name__}")
    for key, (lower, upper, _integer) in _SOLVER_SETTINGS_NUMERIC_BOUNDS.items():
        if not lower <= config[key] <= upper:
            raise ValueError(f"{key} must be in [{lower}, {upper}]")
    validate_array_model(config["array_model"])
    if not 0 <= config["rt_human_diffuse_coefficient"] <= 1:
        raise ValueError("rt_human_diffuse_coefficient must be in [0, 1]")
    if not 0 < config["cosine_3db_beamwidth_deg"] < 180:
        raise ValueError("cosine_3db_beamwidth_deg must be in (0, 180)")
    if config["po_visibility_samples_per_face"] < 1:
        raise ValueError("po_visibility_samples_per_face must be positive")
    for key in ("human_position_m", "radar_orientation_deg"):
        value = config[key]
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 3
            or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in value
            )
        ):
            raise ValueError(f"{key} must contain three finite numbers")
    for key, count in (("selected_tx_indices", spec.num_tx), ("selected_rx_indices", spec.num_rx)):
        indices = config[key]
        if indices is None:
            indices = config[key] = list(range(count))
        if (
            not isinstance(indices, list)
            or not indices
            or any(type(i) is not int or not 0 <= i < count for i in indices)
            or len(set(indices)) != len(indices)
        ):
            raise ValueError(f"{key} must contain unique zero-based indices in [0, {count - 1}]")
    for key, active in (
        ("array_tx_groups", config["selected_tx_indices"]),
        ("array_rx_groups", config["selected_rx_indices"]),
    ):
        groups = validate_array_groups(config[key], name=key)
        if groups is not None:
            if config["array_model"] != "subarray":
                raise ValueError(f"{key} needs array_model 'subarray'")
            local_array_groups(groups, active, name=key)
            config[key] = [list(group) for group in groups]
    start, end, count = config["motion_start_s"], config["motion_end_s"], config["num_frames"]
    if (
        isinstance(start, bool)
        or not isinstance(start, (int, float))
        or not math.isfinite(start)
        or start < 0
    ):
        raise ValueError("motion_start_s must be a nonnegative number of seconds")
    if end is not None and count is not None:
        raise ValueError("set motion_end_s or num_frames, not both")
    if end is not None and (
        isinstance(end, bool)
        or not isinstance(end, (int, float))
        or not math.isfinite(end)
        or end < start
    ):
        raise ValueError("motion_end_s must be a number of seconds no earlier than motion_start_s")
    if count is not None and (type(count) is not int or count <= 0):
        raise ValueError("num_frames must be a positive integer")
    solvers = config["solvers"]
    if not isinstance(solvers, list) or not solvers:
        raise ValueError("solvers must be a nonempty list of: " + ", ".join(SOLVERS))
    config["solvers"] = list(
        dict.fromkeys(validate_solver(solver, name="solvers entry") for solver in solvers)
    )
    up, forward = config["motion_source_up_axis"], config["motion_source_forward_axis"]
    if up is not None or forward is not None:
        motion_source_to_radar_transform(
            up,
            forward,
            handedness=config["motion_source_handedness"],
            length_unit=config["motion_source_length_unit"],
        )
    elif (
        config["motion_source_handedness"] != "right" or config["motion_source_length_unit"] != "m"
    ):
        raise ValueError("manual handedness/units require both source up and forward axes")
    fmcw = FMCWConfig(
        carrier_frequency=config["carrier_frequency_hz"],
        slope=config["slope_hz_per_s"],
        chirp_duration=config["chirp_duration_s"],
        chirp_repetition_time=config["chirp_repetition_time_s"],
        sampling_frequency=config["sampling_frequency_hz"],
        adc_mode=config["adc_mode"],
        if_bandwidth_hz=config["if_bandwidth_hz"],
        num_adc_samples=config["num_adc_samples"],
        num_chirps_per_frame=config["num_chirps_per_frame"],
        frame_period=config["frame_period_s"],
        num_tx=len(config["selected_tx_indices"]),
        tx_schedule=config["tx_schedule"],
    )
    if (
        end is not None
        and radar_frame_count(
            start,
            end,
            frame_period_s=fmcw.frame_period,
            frame_duration_s=fmcw.frame_duration,
        )
        <= 0
    ):
        raise ValueError(
            f"the window {start:g}–{end:g} s is shorter than one radar frame's "
            f"chirps ({1e3 * fmcw.frame_duration:.3f} ms)"
        )


def _window(config: dict) -> tuple[float, float, int]:
    """Return the motion window's start and end [s] and its radar frame count."""

    fmcw = FMCWConfig(
        carrier_frequency=config["carrier_frequency_hz"],
        slope=config["slope_hz_per_s"],
        chirp_duration=config["chirp_duration_s"],
        chirp_repetition_time=config["chirp_repetition_time_s"],
        sampling_frequency=config["sampling_frequency_hz"],
        num_adc_samples=config["num_adc_samples"],
        num_chirps_per_frame=config["num_chirps_per_frame"],
        frame_period=config["frame_period_s"],
        num_tx=len(config["selected_tx_indices"]),
        tx_schedule=config["tx_schedule"],
    )
    start, end, count = config["motion_start_s"], config["motion_end_s"], config["num_frames"]
    if end is None:
        count = 1 if count is None else count
        end = start + (count - 1) * fmcw.frame_period + fmcw.frame_duration
    else:
        count = radar_frame_count(
            start,
            end,
            frame_period_s=fmcw.frame_period,
            frame_duration_s=fmcw.frame_duration,
        )
    return float(start), float(end), int(count)


def _window_summary(config: dict) -> str:
    """Describe the motion window and radar frame count in one line."""

    start, end, count = _window(config)
    return f"motion {start:.3f}–{end:.3f} s · {count} radar frame{'s' * (count != 1)}"


def _check_output(path: Path, *, force: bool, config: dict) -> None:
    if path.is_dir() or (path.exists() and not force):
        raise ValueError(f"output already exists: {path}; choose another file or use --force")
    if any(
        config[key] is not None and path.resolve() == Path(config[key]).resolve()
        for key in _INPUT_PATH_KEYS
    ):
        raise ValueError("an output must not replace an input file or model directory")


def _write_output(path: Path, payload: bytes, *, force: bool) -> None:
    """Publish a complete output atomically, without clobbering by default."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".hermes-scene-", dir=path.parent) as directory:
        temporary = Path(directory) / "output"
        temporary.write_bytes(payload)
        if force:
            temporary.replace(path)
        else:
            os.link(temporary, path)


def main(
    argv: list[str] | None = None,
    *,
    default_config_path: Path | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if default_config_path is not None and not hasattr(args, "config"):
        args.config = default_config_path
    try:
        config = resolve_config(args)
        force = config["force"]
        output = Path(config["output"]).expanduser()
        saved_config = getattr(args, "write_config", None)
        if saved_config is not None:
            saved_config = saved_config.expanduser()
            _check_output(saved_config, force=force, config=config)
        dry_run = getattr(args, "dry_run", False)
        if not dry_run:
            _check_output(output, force=force, config=config)
            if saved_config is not None and saved_config.resolve() == output.resolve():
                raise ValueError("bundle and configuration outputs must be different files")
            if hasattr(args, "config") and args.config.expanduser().resolve() == output.resolve():
                raise ValueError("the bundle output must not replace the input configuration")
    except (OSError, TypeError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    try:
        config_bytes = (json.dumps(config, indent=2, allow_nan=False) + "\n").encode()
        if dry_run:
            if saved_config is not None:
                _write_output(saved_config, config_bytes, force=force)
            print(config_bytes.decode(), end="")
            return 0
        if config["human_motion"] is None:
            raise ValueError("no bundled motion is installed; specify --motion NPZ")
        loaded = load_amass_motion(config["human_motion"], smpl_model_dir=config["smpl_model_dir"])
        duration = float(loaded.sequence.times[-1]) - float(loaded.sequence.times[0])
        window_end = _window(config)[1]
        if duration > 0 and window_end > duration + 1e-9:
            raise ValueError(
                f"the window ends at {window_end:.3f} s but the motion lasts {duration:.3f} s"
            )
        run_arguments = {
            key: value for key, value in config.items() if key not in _RUN_CONTROL_KEYS
        }
        run_arguments["human_motion"] = loaded
        if config["scene_xml"] is not None:
            payload, filename = load_scene_xml(config["scene_xml"])
            run_arguments["scene_xml"] = payload
            run_arguments["scene_xml_filename"] = filename
        solvers = config["solvers"]
        results = {}
        for solver in solvers:
            last_status = None

            def progress(event):
                nonlocal last_status
                if config["quiet"] or "stage" in event:
                    return
                status = _human_room_running_status({**event, "solver": solver}).replace("**", "")
                if status != last_status:
                    print(status, flush=True)
                    last_status = status

            if not config["quiet"]:
                print(f"[{solver}] {_window_summary(config)}", flush=True)
            results[solver] = run_human_room_experiment(
                **run_arguments,
                solver=solver,
                progress_callback=progress,
            )
        primary = results["hybrid_po"] if "hybrid_po" in results else results[solvers[0]]
        bundle = _human_room_export(primary, solver_results=results)
        _write_output(output, bundle.getvalue(), force=force)
        if saved_config is not None:
            _write_output(saved_config, config_bytes, force=force)
        print(f"Saved {output} · solvers: {', '.join(solvers)} · ADC shape: {primary.adc.shape}")
        return 0
    except KeyboardInterrupt:
        print("Simulation interrupted.", file=sys.stderr)
        return 130
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Simulation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
