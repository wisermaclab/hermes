# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Command-line entrypoint for real-radar validation benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

from bundle_prepare.contract import Bundle as BenchmarkClip
from bundle_prepare.integrity import check_bundle_integrity
from .runner import (
    DEFAULT_VALIDATION_MOBILITY_MODES,
    VALIDATION_MOBILITY_MODES,
    BenchmarkRunConfig,
    run_validation_benchmark,
)


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
    text = str(value).strip()
    if not text:
        raise argparse.ArgumentTypeError("index list must not be empty")
    out: list[int] = []
    for token in text.replace(" ", "").split(","):
        if not token:
            continue
        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise argparse.ArgumentTypeError(f"invalid index range {token!r}")
            start, stop = (int(parts[0]), int(parts[1]))
            if start <= 0 or stop <= 0:
                raise argparse.ArgumentTypeError("indices are 1-based")
            step = 1 if stop >= start else -1
            out.extend(range(start, stop + step, step))
        else:
            index = int(token)
            if index <= 0:
                raise argparse.ArgumentTypeError("indices are 1-based")
            out.append(index)
    if not out:
        raise argparse.ArgumentTypeError("index list must select at least one value")
    seen: set[int] = set()
    unique: list[int] = []
    for index in out:
        if index not in seen:
            seen.add(index)
            unique.append(index)
    return tuple(unique)


def _parse_args(argv: list[str] | None = None):
    """Parses validation CLI arguments into a benchmark run configuration."""

    parser = argparse.ArgumentParser(
        description="Validate simulator output against a standard real-radar clip.",
        formatter_class=_ValidationHelpFormatter,
    )
    required = parser.add_argument_group("required inputs")
    required.add_argument("--clip", type=Path, required=True,
                          help="Path to a standard validation clip bundle.")
    required.add_argument("--out", type=Path, required=True,
                          help="Output directory for metrics, maps, and plots.")

    benchmark = parser.add_argument_group("benchmark controls")
    benchmark.add_argument("--suite",
                           choices=("smoke", "range_time", "range_doppler",
                                    "pose_range", "dsp_maps"),
                           default="dsp_maps",
                           help="Validation suite to run.")
    benchmark.add_argument("--simulated-adc", type=Path, default=None,
                           help=(
                               "Optional NPZ containing simulated adc [F,M,N,C]. "
                               "When omitted, the CLI runs simulation."
                           ))
    benchmark.add_argument("--max-frames", type=int, default=None,
                           help="Limit the clip to the first N frames.")

    simulation = parser.add_argument_group("simulation controls")
    simulation.add_argument(
        "--mobility-mode",
        choices=VALIDATION_MOBILITY_MODES,
        default=DEFAULT_VALIDATION_MOBILITY_MODES,
        nargs="+",
        help=(
            "One or more simulator mobility modes used when --simulated-adc "
            "is omitted. Supported modes: "
            f"{', '.join(VALIDATION_MOBILITY_MODES)}."
        ),
    )
    simulation.add_argument("--scene", type=str, default=None,
                            help=(
                                "Sionna scene path used when simulation runs. "
                                "Defaults to the clip environment scene, then "
                                "an empty scene."
                            ))
    simulation.add_argument("--smpl-model-dir", type=str, default=None,
                            help=(
                                "SMPL/SMPL-X model directory used to build the "
                                "bundle's parameterized body motion for range "
                                "metadata and simulation. "
                                "Defaults to MMWAVE_SMPL_MODEL_DIR when set."
                            ))
    simulation.add_argument("--no-diffuse-reflection",
                            action="store_true",
                            help="Disable diffuse reflection in generated simulation.")
    simulation.add_argument("--refraction", action="store_true",
                            help="Enable refraction in generated simulation.")
    simulation.add_argument("--human-specular-reflection",
                            action="store_true",
                            help=(
                                "Enable human specular reflection in generated "
                                "simulation."
                            ))
    simulation.add_argument(
        "--hybrid-po-calibration",
        choices=("none", "rt_first_pose", "rt_sequence"),
        default="rt_first_pose",
        help=(
            "RT-reference calibration used by hybrid_static_env_po. "
            "rt_first_pose fits one amplitude gain from the first-frame RT "
            "baseline; rt_sequence fits from selected path-power samples; "
            "none disables calibration."
        ),
    )
    simulation.add_argument(
        "--ignore-tdm-timing",
        action="store_true",
        help=(
            "For generated simulation of a TDM bundle, treat all selected TX "
            "channels as simultaneous at each full-cycle start. This uses one "
            "simulator run per mobility mode and preserves the full-cycle "
            "slow-time interval, but omits per-slot mesh and phase offsets. "
            "It has no effect for non-TDM bundles or --simulated-adc."
        ),
    )

    channels = parser.add_argument_group("channel selection controls")
    channels.add_argument(
        "--subarray-tx",
        type=_parse_index_spec,
        default=None,
        help=(
            "Optional 1-based physical TX subset, e.g. '1,2,3,10'. "
            "Ranges such as '1-3,10' are accepted."
        ),
    )
    channels.add_argument(
        "--subarray-rx",
        type=_parse_index_spec,
        default=None,
        help=(
            "Optional 1-based physical RX subset, e.g. '5-12'. "
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
    output.add_argument("--no-plots", action="store_true",
                        help="Skip PNG diagnostic plots.")
    output.add_argument("--progress", action="store_true",
                        help="Print simulator progress while generated simulation runs.")
    args = parser.parse_args(argv)
    if len(set(args.mobility_mode)) != len(args.mobility_mode):
        parser.error("--mobility-mode values must not contain duplicates")
    return args


def main(argv: list[str] | None = None) -> int:
    """Loads a clip, runs validation, and reports the output metric path."""

    args = _parse_args(argv)
    # Validate the whole portable boundary before loading the numerical subset
    # used by the runner. This catches missing AMASS, camera, and environment
    # references with the same rules as ``mmwave-check-bundle``.
    check_bundle_integrity(args.clip)
    clip = BenchmarkClip.load(args.clip)
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
        mobility_modes=tuple(args.mobility_mode),
        diffuse_reflection=not args.no_diffuse_reflection,
        refraction=args.refraction,
        human_specular_reflection=args.human_specular_reflection,
        hybrid_po_calibration_mode=args.hybrid_po_calibration,
        ignore_tdm_timing=args.ignore_tdm_timing,
        smpl_model_dir=args.smpl_model_dir,
        subarray_tx_indices=args.subarray_tx,
        subarray_rx_indices=args.subarray_rx,
        progress=args.progress,
    )
    metrics = run_validation_benchmark(clip, config)
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
