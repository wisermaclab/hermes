# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Prepare measured or simulated HERMES bundles from JSON or direct arguments."""

from __future__ import annotations

import argparse
from dataclasses import replace
from importlib import resources
import json
from pathlib import Path
import sys

import numpy as np

from .contract import (
    BundleSensorConfig,
    load_adc,
    load_frames,
    make_bundle_descriptor,
    normalize_adc_times,
    read_json,
)
from .writer import BundleWriter

_CONFIG_KEYS = {
    "schema_version",
    "adc",
    "sensor",
    "ti_config",
    "board_model",
    "frames",
    "frame_times_s",
    "start_time_s",
    "frame_period_s",
    "output",
    "origin",
    "background_adc",
    "scene",
    "target_mesh",
    "target_obj",
    "motion_parameters",
    "evaluated_mesh",
    "environment",
    "iq_transform",
    "phase_signs_removed",
    "adc_units",
    "target_material",
    "motion_coordinate_frame",
    "timestamp_tolerance_s",
}


def prepare_from_config(config_path: str | Path, *, output=None, force=False) -> dict:
    """Prepare a bundle; all config paths are relative to the config file."""
    config_path = Path(config_path).expanduser().resolve()
    config = read_json(config_path)
    return _prepare_bundle(
        config, base_dir=config_path.parent, source_config=config_path, output=output, force=force
    )


def _prepare_bundle(
    config, *, base_dir: Path, source_config=None, output=None, force=False
) -> dict:
    """Share preparation and validation across config-file and direct inputs."""
    if (
        not isinstance(config, dict)
        or type(config.get("schema_version")) is not int
        or config["schema_version"] != 1
    ):
        raise ValueError("Preparation config must be an object with schema_version: 1")
    unknown = set(config) - _CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown preparation settings: {sorted(unknown)}")

    def path(key):
        value = config.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{key} must be a nonempty path")
        return (base_dir / Path(value).expanduser()).absolute()

    if ("sensor" in config) == ("ti_config" in config):
        raise ValueError("Specify exactly one of sensor and ti_config")
    if "sensor" in config:
        if "board_model" in config:
            raise ValueError("board_model belongs in sensor.json when sensor is supplied")
        sensor = BundleSensorConfig.from_json(path("sensor"))
    else:
        sensor = BundleSensorConfig.from_ti_cli_config(
            path("ti_config"), board_model=config["board_model"]
        )
    sensor = replace(
        sensor,
        adc_units=config.get("adc_units", sensor.adc_units),
        timestamp_tolerance_s=config.get("timestamp_tolerance_s", sensor.timestamp_tolerance_s),
    )
    hardware = sensor.to_sensor().hardware
    adc, times = load_adc(path("adc"))
    has_motion = any(key in config for key in ("motion_parameters", "evaluated_mesh"))
    if has_motion and not any(
        key in config for key in ("frames", "frame_times_s", "start_time_s")
    ):
        raise ValueError(
            "Motion inputs require explicit frame timing: frames, frame_times_s, or start_time_s on the motion timeline"
        )
    timing_modes = sum(key in config for key in ("frames", "frame_times_s"))
    if timing_modes > 1 or (
        timing_modes and any(k in config for k in ("start_time_s", "frame_period_s"))
    ):
        raise ValueError(
            "Use frames, frame_times_s, or start_time_s/frame_period_s as one timing mode"
        )
    if "frames" in config:
        records = read_json(path("frames"))
        frame_times = np.asarray([r.motion_time_s for r in load_frames(path("frames"))])
    else:
        if "frame_times_s" in config:
            frame_times = np.asarray(config["frame_times_s"], dtype=float)
        elif times is not None and not any(
            k in config for k in ("start_time_s", "frame_period_s")
        ):
            frame_times = times[:, 0] - times[0, 0]
        else:
            frame_times = float(config.get("start_time_s", 0)) + np.arange(adc.shape[0]) * float(
                config.get("frame_period_s", sensor.fmcw.frame_period)
            )
        if frame_times.shape != (adc.shape[0],) or not np.all(np.isfinite(frame_times)):
            raise ValueError("Frame times must contain one finite timestamp per ADC frame")
        records = {
            "frames": [
                {"bundle_index": i, "motion_time_s": float(t)} for i, t in enumerate(frame_times)
            ]
        }
    if len(frame_times) != adc.shape[0]:
        raise ValueError("Frame timing length must match ADC frame count")
    source_origin = None if times is None else float(times[0, 0])
    if times is not None:
        quantization = 4 * max(
            float(np.max(np.abs(np.spacing(times)))),
            float(np.max(np.abs(np.spacing(frame_times)))),
        )
        tolerance = (
            max(1e-9, 0.01 * sensor.fmcw.slow_time_interval)
            if sensor.timestamp_tolerance_s is None
            else sensor.timestamp_tolerance_s
        )
        sensor = replace(sensor, timestamp_tolerance_s=max(tolerance, quantization))
    times = normalize_adc_times(
        times, frame_times, sensor.fmcw, tolerance_s=sensor.timestamp_tolerance_s
    )
    if has_motion and config.get("motion_coordinate_frame") != "world":
        raise ValueError(
            "Motion inputs require explicit motion_coordinate_frame: world; normalize source coordinates upstream"
        )
    if not has_motion and "motion_coordinate_frame" in config:
        raise ValueError("motion_coordinate_frame requires a motion input")
    transform = config.get("iq_transform", "none")
    if transform not in {"none", "conjugate", "swap_iq"}:
        raise ValueError("iq_transform must be none, conjugate or swap_iq")
    phase_removed = config.get("phase_signs_removed", False)
    if type(phase_removed) is not bool:
        raise ValueError("phase_signs_removed must be a boolean")

    def normalize(values):
        if transform == "conjugate":
            values = values.conj()
        elif transform == "swap_iq":
            values = 1j * values.conj()
        if phase_removed and hardware.virtual_channel_phase_signs is not None:
            values = values * hardware.virtual_channel_phase_signs
        return values

    descriptor = make_bundle_descriptor(primary_origin=config.get("origin", "measurement"))
    environment = read_json(path("environment")) if "environment" in config else {}
    if not isinstance(environment, dict):
        raise ValueError("environment must contain an object")
    if any(environment.get(k) is not None for k in ("scene_path", "scene_file")):
        raise ValueError(
            "Use the preparation config's scene input; environment scene aliases are bundle-relative"
        )
    if not isinstance(environment, dict):
        raise ValueError("environment must contain an object")
    output_path = Path(output).expanduser().absolute() if output is not None else path("output")
    inputs = {
        path(k).resolve()
        for k in (
            "adc",
            "sensor",
            "ti_config",
            "frames",
            "background_adc",
            "scene",
            "target_mesh",
            "target_obj",
            "motion_parameters",
            "evaluated_mesh",
            "environment",
            "target_material",
        )
        if k in config
    }
    if source_config is not None:
        inputs.add(source_config)
    if any(output_path.resolve() == p or output_path.resolve() in p.parents for p in inputs):
        raise ValueError("Output must not replace or contain an input file")
    with BundleWriter(output_path, force=force) as writer:
        writer.write_json("sensor.json", sensor.to_json())
        writer.write_json("frames.json", records)
        writer.write_npz("radar_adc.npz", adc=normalize(adc), times=times)
        if "background_adc" in config:
            background, _ = load_adc(path("background_adc"), allowed_ndim=(3, 4))
            writer.write_npz("background_adc.npz", adc=normalize(background))
            descriptor["adcs"]["background"] = {
                "path": "background_adc.npz",
                "origin": "background",
            }
        for setting, group, key, name in (
            ("target_mesh", "target", "mesh", "target_mesh.npz"),
            ("motion_parameters", "motion", "parameters", "amass_sequence.npz"),
            ("evaluated_mesh", "motion", "evaluated_mesh", "motion.npz"),
        ):
            if setting in config:
                writer.copy_file(path(setting), name)
                descriptor.setdefault(group, {})[key] = name
        if "target_obj" in config:
            geometry = writer.add_scene(path("target_obj"))
            descriptor.setdefault("target", {})["obj"] = geometry["file"]
        if "motion" in descriptor:
            descriptor["motion"]["coordinate_frame"] = config["motion_coordinate_frame"]
        if "target_material" in config:
            descriptor.setdefault("target", {})["material_parameters"] = read_json(
                path("target_material")
            )
        if "scene" in config:
            descriptor["scene"] = writer.add_scene(path("scene"))
        if environment:
            descriptor["environment"] = "environment.json"
            writer.write_json("environment.json", environment)
        writer.write_json("bundle.json", descriptor)
        writer.write_json(
            "preparation.json",
            {
                "tool": "mmwave-prepare-bundle",
                "config_schema_version": 1,
                "iq_transform": transform,
                "phase_signs_restored": phase_removed,
                "adc_units": sensor.adc_units,
                "source_adc_timestamp_origin_s": source_origin,
                "timestamp_tolerance_s": sensor.timestamp_tolerance_s,
            },
        )
        report = writer.finish()
    return {**report, "path": str(output_path)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--config", type=Path, help="Preparation JSON; paths are relative to this file"
    )
    mode.add_argument(
        "--adc", type=Path, help="Decoded ADC NPZ; use direct inputs instead of --config"
    )
    mode.add_argument(
        "--example", type=Path, help="Copy the packaged synthetic static-target inputs and config"
    )
    mode.add_argument(
        "--channel-table",
        metavar="BOARD",
        help="Print the catalog's channel geometry and phase signs as JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory/ZIP, required with --adc; overrides config output (relative to CWD)",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing output bundle")

    inputs = parser.add_argument_group("direct inputs (with --adc; paths relative to CWD)")
    sensor = inputs.add_mutually_exclusive_group()
    sensor.add_argument(
        "--sensor", type=Path, help="Sensor JSON with radar settings and channel mapping"
    )
    sensor.add_argument(
        "--ti-config", type=Path, help="TI radar CLI configuration; requires --board-model"
    )
    inputs.add_argument("--board-model", help="Radar board model for --ti-config, e.g. IWR6843AOP")
    for name, help_text in (
        ("scene", "Environment XML or OBJ; keep the declared target separate"),
        ("target_mesh", "Static target NPZ with world-coordinate vertices and faces"),
        ("target_obj", "Static target OBJ"),
        ("target_material", "Target RadioMaterial parameters JSON"),
        ("background_adc", "Optional background ADC NPZ; not subtracted during preparation"),
        ("environment", "Environment metadata JSON"),
        ("motion_parameters", "World-coordinate body motion parameters NPZ"),
        ("evaluated_mesh", "World-coordinate evaluated mesh motion NPZ"),
    ):
        inputs.add_argument("--" + name.replace("_", "-"), type=Path, help=help_text)
    inputs.add_argument(
        "--motion-coordinate-frame",
        choices=["world"],
        help="Required confirmation for either motion input",
    )

    timing = parser.add_argument_group("direct timing")
    timeline = timing.add_mutually_exclusive_group()
    timeline.add_argument("--frames", type=Path, help="Frame records JSON")
    timeline.add_argument(
        "--frame-times-s",
        nargs="+",
        type=float,
        metavar="SECONDS",
        help="One frame-start time per ADC frame",
    )
    timeline.add_argument("--start-time-s", type=float, help="First frame start (default: 0)")
    timing.add_argument(
        "--frame-period-s",
        type=float,
        help="Frame spacing with --start-time-s or a default start of 0",
    )
    timing.add_argument(
        "--timestamp-tolerance-s", type=float, help="Absolute ADC timestamp tolerance"
    )

    conventions = parser.add_argument_group("direct data conventions")
    conventions.add_argument(
        "--origin",
        choices=["measurement", "simulation"],
        help="Primary ADC origin (default: measurement)",
    )
    conventions.add_argument("--adc-units", help="ADC amplitude units, e.g. V or counts")
    conventions.add_argument(
        "--iq-transform",
        choices=["none", "conjugate", "swap_iq"],
        help="Explicit IQ normalization (default: none)",
    )
    conventions.add_argument(
        "--phase-signs-removed",
        action="store_true",
        default=None,
        help="Restore fixed board phase signs removed by the source decoder",
    )
    args = parser.parse_args(argv)
    direct = {
        key: (str(value) if isinstance(value, Path) else value)
        for key in _CONFIG_KEYS - {"schema_version", "output"}
        if (value := getattr(args, key, None)) is not None
    }
    if args.adc is None and direct:
        parser.error(
            "Direct input arguments require --adc and cannot be combined with --config, --example, or --channel-table"
        )
    if args.adc is not None:
        if args.output is None:
            parser.error("--output is required with --adc")
        if args.sensor is None and args.ti_config is None:
            parser.error("--adc requires --sensor or --ti-config with --board-model")
        if args.ti_config is not None and args.board_model is None:
            parser.error("--ti-config requires --board-model")
        if args.board_model is not None and args.ti_config is None:
            parser.error(
                "--board-model is only used with --ti-config; put the board in --sensor JSON otherwise"
            )
        if args.frame_period_s is not None and (
            args.frames is not None or args.frame_times_s is not None
        ):
            parser.error("--frame-period-s cannot be combined with --frames or --frame-times-s")
    try:
        if args.channel_table:
            from mmWaveRadar.radar import get_ti_board_spec

            spec = get_ti_board_spec(args.channel_table)
            print(
                json.dumps(
                    {
                        "board_model": spec.key,
                        "channels": [
                            {
                                "channel": i,
                                "tx": c.tx_index,
                                "rx": c.rx_index,
                                "position_lambda": list(c.position_lambda),
                                "phase_sign": int(c.phase_sign.real),
                            }
                            for i, c in enumerate(spec.virtual_channels)
                        ],
                    },
                    indent=2,
                )
            )
        elif args.example:
            source = resources.files("bundle_prepare").joinpath("data", "static_target")
            args.example.mkdir(parents=True, exist_ok=False)
            for item in source.iterdir():
                if item.is_file():
                    args.example.joinpath(item.name).write_bytes(item.read_bytes())
            print(
                f"Example inputs saved to {args.example}; run mmwave-prepare-bundle --config {args.example / 'prepare.json'}"
            )
        elif args.config is not None:
            result = prepare_from_config(args.config, output=args.output, force=args.force)
            print(json.dumps(result, indent=2))
        else:
            result = _prepare_bundle(
                {"schema_version": 1, **direct},
                base_dir=Path.cwd(),
                output=args.output,
                force=args.force,
            )
            print(json.dumps(result, indent=2))
        return 0
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print(f"Bundle preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
