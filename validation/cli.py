# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Command-line entrypoint for real-radar validation benchmarks."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from bundle_prepare.integrity import check_and_load_bundle
from mmWaveRadar.naming import HYBRID_CALIBRATIONS, add_removed_options, parse_index_spec
from mmWaveRadar.simulation.types import DEFAULT_PO_REFRESH_ON_PHASE_ERROR_RAD
from .results import write_metrics_json
from .runner import (
    DEFAULT_VALIDATION_SOLVERS,
    VALIDATION_SOLVERS,
    BenchmarkRunConfig,
    run_validation_benchmark,
)

_REMOVED_OPTIONS = {
    "--mobility-mode": (
        "--solvers (po, rt, hybrid_po); the RT retrace cadence is "
        "--rt-refresh-frames (0 = every chirp)"
    ),
    "--no-diffuse-reflection": "--no-rt-diffuse-reflection",
    "--refraction": "--rt-refraction",
    "--human-specular-reflection": "--rt-human-specular-reflection",
    "--hybrid-po-calibration": "--hybrid-calibration",
    "--subarray-tx": "--tx-indices, which is zero-based (TI's TX1 is 0)",
    "--subarray-rx": "--rx-indices, which is zero-based (TI's RX1 is 0)",
}


def _frame_count(value: str) -> int:
    count = int(value)
    if count < 0:
        raise argparse.ArgumentTypeError("must be a non-negative frame count")
    return count


class _ValidationHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Help formatter that displays defaults for meaningful CLI options."""

    def _get_help_string(self, action):
        help_text = action.help or ""
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            return help_text
        if (
            action.default is argparse.SUPPRESS
            or action.default is None
            or action.required
            or "%(default)" in help_text
        ):
            return help_text
        if isinstance(action.default, tuple):
            default = " ".join(str(value) for value in action.default)
            return f"{help_text} (default: {default})"
        return super()._get_help_string(action)


def _parse_index_spec(value: str) -> tuple[int, ...]:
    """Parses zero-based indices such as ``0-2,9`` for argparse."""

    try:
        return parse_index_spec(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _parse_args(argv: list[str] | None = None):
    """Parses validation CLI arguments into a benchmark run configuration."""

    parser = argparse.ArgumentParser(
        description="Validate simulator output against a standard real-radar clip.",
        formatter_class=_ValidationHelpFormatter,
        epilog="Configuration sweeps: mmwave-validation-benchmark study --help",
    )
    required = parser.add_argument_group("required inputs")
    required.add_argument(
        "--clip",
        type=Path,
        required=True,
        help="Path to a validation bundle directory or ZIP file.",
    )
    required.add_argument(
        "--out", type=Path, required=True, help="Output directory for metrics, maps, and plots."
    )

    benchmark = parser.add_argument_group("benchmark controls")
    benchmark.add_argument(
        "--suite",
        choices=("smoke", "range_time", "range_doppler", "pose_range", "dsp_maps"),
        default="dsp_maps",
        help="Validation suite to run.",
    )
    benchmark.add_argument(
        "--simulated-adc",
        type=Path,
        default=None,
        help=(
            "Optional NPZ containing simulated adc [F,M,N,C]. "
            "When omitted, the CLI runs simulation."
        ),
    )
    benchmark.add_argument(
        "--max-frames", type=int, default=None, help="Limit the clip to the first N frames."
    )

    simulation = parser.add_argument_group("simulation controls")
    simulation.add_argument(
        "--solver-settings",
        type=Path,
        help="JSON object of public RT/PO/Hybrid/array/backend settings.",
    )
    simulation.add_argument(
        "--target-material",
        type=Path,
        help="JSON object of RadioMaterial parameters for the target.",
    )
    simulation.add_argument(
        "--frozen-human-po-amplitude-gain",
        type=float,
        help="Fixed validation-boundary PO gain; use --hybrid-calibration none.",
    )
    simulation.add_argument(
        "--solvers",
        choices=VALIDATION_SOLVERS,
        default=DEFAULT_VALIDATION_SOLVERS,
        nargs="+",
        help=(
            "One or more solvers run when --simulated-adc is omitted: "
            f"{', '.join(VALIDATION_SOLVERS)}."
        ),
    )
    simulation.add_argument(
        "--rt-refresh-frames",
        type=_frame_count,
        default=1,
        help=(
            "RT retraces at the start of every N radar frames and "
            "phase-updates the traced paths in between; 0 retraces every "
            "chirp."
        ),
    )
    simulation.add_argument(
        "--po-refresh-on-phase-error-rad",
        type=float,
        default=DEFAULT_PO_REFRESH_ON_PHASE_ERROR_RAD,
        help="PO and Hybrid PO neglected vertex phase-change trigger [rad].",
    )
    simulation.add_argument(
        "--scene",
        type=str,
        default=None,
        help=(
            "Sionna scene path used when simulation runs. "
            "Defaults to the clip environment scene, then "
            "an empty scene."
        ),
    )
    simulation.add_argument(
        "--smpl-model-dir",
        type=str,
        default=None,
        help=(
            "SMPL/SMPL-X model directory used to build the "
            "bundle's parameterized body motion for range "
            "metadata and simulation. "
            "Defaults to MMWAVE_SMPL_MODEL_DIR when set."
        ),
    )
    simulation.add_argument(
        "--no-rt-diffuse-reflection", action="store_true", help="Disable RT diffuse reflection."
    )
    simulation.add_argument("--rt-refraction", action="store_true", help="Enable RT refraction.")
    simulation.add_argument(
        "--rt-human-specular-reflection",
        action="store_true",
        help="Enable RT specular reflection on the human.",
    )
    simulation.add_argument(
        "--hybrid-calibration",
        choices=HYBRID_CALIBRATIONS,
        default="rt_first_pose",
        help=(
            "Hybrid PO amplitude calibration. rt_first_pose fits one "
            "amplitude gain from the first-frame RT baseline; rt_sequence "
            "fits from selected path-power samples; none disables "
            "calibration."
        ),
    )
    simulation.add_argument(
        "--ignore-tdm-timing",
        action="store_true",
        help=(
            "For generated simulation of a TDM bundle, treat all selected TX "
            "channels as simultaneous at each full-cycle start. This uses one "
            "simulator run per solver and preserves the full-cycle "
            "slow-time interval, but omits per-slot mesh and phase offsets. "
            "It has no effect for non-TDM bundles or --simulated-adc."
        ),
    )

    channels = parser.add_argument_group("channel selection controls")
    channels.add_argument(
        "--tx-indices",
        type=_parse_index_spec,
        default=None,
        help=(
            "Optional zero-based TX subset, e.g. '0,1,2,9'. Ranges such as "
            "'0-2,9' are accepted. TI and dataset labels count from 1, so "
            "their TX1 is 0."
        ),
    )
    channels.add_argument(
        "--rx-indices",
        type=_parse_index_spec,
        default=None,
        help=(
            "Optional zero-based RX subset, e.g. '4-11'. "
            "Ranges and comma-separated lists are accepted."
        ),
    )

    metrics = parser.add_argument_group("metric and preprocessing controls")
    metrics.add_argument(
        "--background-subtract",
        action="store_true",
        help=(
            "Enable real-ADC background subtraction using background_adc.npz. "
            "Default behavior is no real-ADC background subtraction. The run "
            "fails if this flag is enabled and the clip bundle does not contain "
            "background_adc.npz. Simulated ADC is unchanged."
        ),
    )
    metrics.add_argument(
        "--clutter-removal",
        dest="clutter_removal",
        choices=("none", "mean"),
        default="none",
        help=(
            "DSP-domain slow-time clutter removal used while building real and "
            "simulated range-time/range-Doppler maps. 'mean' subtracts the "
            "per-range-bin/channel slow-time mean after range FFT; 'none' "
            "leaves DSP maps unchanged."
        ),
    )
    output = parser.add_argument_group("output controls")
    output.add_argument("--no-plots", action="store_true", help="Skip PNG diagnostic plots.")
    output.add_argument(
        "--progress",
        action="store_true",
        help="Print simulator progress while generated simulation runs.",
    )
    add_removed_options(parser, _REMOVED_OPTIONS)
    args = parser.parse_args(argv)
    if len(set(args.solvers)) != len(args.solvers):
        parser.error("--solvers values must not contain duplicates")
    return args


def main(argv: list[str] | None = None) -> int:
    """Loads a clip, runs validation, and reports the output metric path."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "study":
        from .study import main as study_main

        return study_main(arguments[1:])
    args = _parse_args(argv)
    config = BenchmarkRunConfig(
        suite=args.suite,
        out_dir=args.out,
        simulated_adc_path=args.simulated_adc,
        max_frames=args.max_frames,
        background_subtract=args.background_subtract,
        write_plots=not args.no_plots,
        simulate=args.simulated_adc is None,
        clutter_removal=args.clutter_removal,
        scene_path=args.scene,
        solvers=tuple(args.solvers),
        rt_refresh_frames=args.rt_refresh_frames,
        po_refresh_on_phase_error_rad=args.po_refresh_on_phase_error_rad,
        rt_diffuse_reflection=not args.no_rt_diffuse_reflection,
        rt_refraction=args.rt_refraction,
        rt_human_specular_reflection=args.rt_human_specular_reflection,
        hybrid_calibration=args.hybrid_calibration,
        ignore_tdm_timing=args.ignore_tdm_timing,
        smpl_model_dir=args.smpl_model_dir,
        selected_tx_indices=args.tx_indices,
        selected_rx_indices=args.rx_indices,
        progress=args.progress,
        solver_settings=(
            json.loads(args.solver_settings.read_text()) if args.solver_settings else {}
        ),
        target_material=(
            json.loads(args.target_material.read_text()) if args.target_material else {}
        ),
        frozen_human_po_amplitude_gain=args.frozen_human_po_amplitude_gain,
    )
    with ExitStack() as stack:
        clip_root = args.clip
        is_archive = args.clip.is_file() and args.clip.suffix.lower() == ".zip"
        if is_archive:
            from bundle_prepare.archives import extract_bundle_zip

            clip_root = Path(stack.enter_context(TemporaryDirectory(prefix="hermes-validation-")))
            extract_bundle_zip(args.clip, clip_root)
        # Check and load once, including all referenced scene/motion resources.
        # Keep extracted assets alive through simulation and plot generation.
        clip, _integrity_report = check_and_load_bundle(clip_root)
        metrics = run_validation_benchmark(clip, config)
        if is_archive:
            metrics["clip"] = str(args.clip)
            if metrics.get("scene_path") is not None:
                scene = Path(metrics["scene_path"]).resolve()
                if scene.is_relative_to(clip_root.resolve()):
                    relative_scene = scene.relative_to(clip_root.resolve())
                    metrics["scene_path"] = f"{args.clip}!/{relative_scene.as_posix()}"
            write_metrics_json(args.out / "metrics.json", metrics)
    print(f"Wrote validation metrics: {args.out / 'metrics.json'}")
    if "range_time" in metrics:
        rt = metrics["range_time"]
        print(
            "Range-time: "
            f"corr={rt['normalized_correlation']:.3f}, "
            f"peak_err={rt['peak_range_error_m']:.3f} m"
        )
    if "range_doppler" in metrics:
        rd = metrics["range_doppler"]
        print(
            "Range-Doppler: "
            f"corr={rd['normalized_correlation']:.3f}, "
            f"peak_err={rd['peak_range_error_m']:.3f} m"
        )
    for label, comparison in metrics.get("comparisons", {}).items():
        if label == metrics.get("primary_simulation"):
            continue
        rt = comparison.get("range_time")
        rd = comparison.get("range_doppler")
        if rt is not None:
            print(
                f"{label.upper()} range-time: "
                f"corr={rt['normalized_correlation']:.3f}, "
                f"peak_err={rt['peak_range_error_m']:.3f} m"
            )
        if rd is not None:
            print(
                f"{label.upper()} range-Doppler: "
                f"corr={rd['normalized_correlation']:.3f}, "
                f"peak_err={rd['peak_range_error_m']:.3f} m"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
