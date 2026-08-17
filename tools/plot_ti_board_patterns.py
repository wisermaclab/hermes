#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Plot TI board catalog layouts, digitized pattern losses, and gain cuts."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path(tempfile.gettempdir()) / "mmwave_radar_outputs" / "ti_board_patterns"


def main() -> None:
    """Runs catalog inspection and writes the requested board figures."""

    args = build_arg_parser().parse_args()
    ensure_src_path()
    plt = configure_matplotlib(show=args.show)

    from mmWaveRadar.radar import (  # pylint: disable=import-outside-toplevel
        RadarHardware,
        available_ti_boards,
        get_ti_board_spec,
    )

    board_keys = resolve_boards(args.boards, available_ti_boards, get_ti_board_spec)
    output_dir = None if args.no_save else args.output_dir.expanduser().resolve()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    print_catalog_summary(board_keys, get_ti_board_spec)
    saved_paths: list[Path] = []
    for board_key in board_keys:
        if "layout" in args.plots:
            fig, _ = plot_virtual_array(plt, RadarHardware, board_key)
            saved_paths.extend(
                save_figure(fig, output_dir, f"{board_key}_virtual_array", args.format, args.dpi)
            )

        if "pattern" in args.plots:
            figures = plot_pattern_losses(plt, RadarHardware, board_key)
            for figure_index, fig in enumerate(figures):
                suffix = "" if len(figures) == 1 else f"_band{figure_index + 1}"
                saved_paths.extend(
                    save_figure(
                        fig,
                        output_dir,
                        f"{board_key}_digitized_pattern_losses{suffix}",
                        args.format,
                        args.dpi,
                    )
                )

        if "gain" in args.plots:
            fig, _ = plot_channel_gain_cut(
                plt,
                RadarHardware,
                board_key,
                frequency_hz=args.frequency_hz,
            )
            saved_paths.extend(
                save_figure(fig, output_dir, f"{board_key}_channel_gain", args.format, args.dpi)
            )

    if args.inspect_board:
        print_board_details(RadarHardware, args.inspect_board)

    if saved_paths:
        print("\nSaved figures:")
        for path in saved_paths:
            print(f"  {path}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds CLI options for board selection, plot types, and output files."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--boards",
        nargs="+",
        help="TI board keys or aliases. Defaults to every catalog board.",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=("layout", "pattern", "gain"),
        default=("layout", "pattern", "gain"),
        help="Figure groups to produce.",
    )
    parser.add_argument(
        "--inspect-board",
        default="IWR6843AOPEVM",
        help="Board key or alias to print detailed channel metadata for.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated figures.",
    )
    parser.add_argument("--format", default="png", choices=("png", "pdf", "svg"))
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--frequency-hz",
        type=float,
        help="Frequency for channel-gain cuts. Defaults to the mean pattern band.",
    )
    parser.add_argument("--show", action="store_true", help="Display figures interactively.")
    parser.add_argument("--no-save", action="store_true", help="Do not write figures.")
    return parser


def ensure_src_path() -> None:
    """Makes the source checkout importable when the package is not installed."""

    src_path = str(REPO_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def configure_matplotlib(*, show: bool):
    """Configures Matplotlib for batch or interactive plotting."""

    import matplotlib  # pylint: disable=import-outside-toplevel

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    plt.rcParams.update({
        "figure.figsize": (10, 4),
        "axes.grid": True,
        "grid.alpha": 0.25,
    })
    return plt


def resolve_boards(board_args, available_ti_boards, get_ti_board_spec) -> list[str]:
    """Resolves CLI board aliases into canonical catalog keys."""

    if not board_args:
        return list(available_ti_boards())
    return [get_ti_board_spec(board).key for board in board_args]


def save_figure(fig, output_dir: Path | None, stem: str, fmt: str, dpi: int) -> list[Path]:
    """Saves one figure when output is enabled and returns written paths."""

    if output_dir is None:
        return []
    path = output_dir / f"{safe_stem(stem)}.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return [path]


def safe_stem(value: str) -> str:
    """Returns a filesystem-safe filename stem for a board/plot label."""

    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def print_catalog_summary(board_keys, get_ti_board_spec) -> None:
    """Prints compact board metadata before generating figures."""

    print("TI board catalog")
    for board_key in board_keys:
        spec = get_ti_board_spec(board_key)
        signs = np.asarray(spec.channel_phase_signs()).real.astype(int).tolist()
        print(f"{spec.key}: {spec.device}")
        print(
            f"  channels: {spec.num_tx} TX x {spec.num_rx} RX = "
            f"{spec.num_virtual_channels} virtual"
        )
        print(
            f"  RF range: {spec.frequency_range_hz[0] / 1e9:.1f}-"
            f"{spec.frequency_range_hz[1] / 1e9:.1f} GHz"
        )
        print(f"  nominal element gain: {spec.antenna_gain_dbi_per_element:.1f} dBi")
        print(f"  fixed signs: {signs}")
        print()


def tx_color(label: str) -> str:
    """Maps a virtual-channel label such as ``TX1 RX2`` to a plot color."""

    tx = int(label.split()[0].replace("TX", ""))
    return f"C{tx - 1}"


def plot_virtual_array(plt, radar_hardware, board_key: str):
    """Plots documented virtual-channel positions and phase signs."""

    hardware = radar_hardware.from_ti_board(board_key, pattern_mode="none")
    positions = hardware.virtual_channel_positions_lambda
    labels = hardware.virtual_channel_labels
    signs = hardware.virtual_channel_phase_signs.real.astype(int)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for index, (xy, label, sign) in enumerate(zip(positions, labels, signs), start=1):
        marker = "o" if sign > 0 else "s"
        ax.scatter(
            xy[0],
            xy[1],
            s=150,
            c=tx_color(label),
            marker=marker,
            edgecolor="black",
            linewidth=0.8,
        )
        ax.text(xy[0], xy[1] + 0.035, str(index), ha="center", va="bottom", fontsize=8)
    ax.set_title(f"{hardware.name} documented virtual array")
    ax.set_xlabel("horizontal coordinate / lambda")
    ax.set_ylabel("vertical coordinate / lambda")
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="0.55",
                markeredgecolor="black",
                markersize=9,
                label="+1 sign",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="0.55",
                markeredgecolor="black",
                markersize=9,
                label="-1 sign",
            ),
        ],
        loc="best",
    )
    return fig, ax


def plot_pattern_losses(plt, radar_hardware, board_key: str) -> list:
    """Plots digitized azimuth/elevation pattern losses when available."""

    try:
        hardware = radar_hardware.from_ti_board(
            board_key,
            pattern_mode="digitized",
        )
    except FileNotFoundError as exc:
        print(f"{board_key}: {exc} Skipping digitized pattern-loss plots.")
        return []
    pattern = hardware.antenna_pattern
    if pattern is None:
        print(f"{hardware.name}: no digitized pattern asset; skipping pattern-loss plots.")
        return []

    figures = []
    for band_index, band_hz in enumerate(pattern.band_centers_hz):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
        for channel, label in enumerate(pattern.channel_labels):
            color = tx_color(label)
            alpha = 0.35 if pattern.num_virtual_channels > 8 else 0.6
            axes[0].plot(
                pattern.azimuth_angles_deg,
                pattern.azimuth_loss_db[band_index, channel],
                color=color,
                alpha=alpha,
                linewidth=1.2,
            )
            axes[1].plot(
                pattern.elevation_angles_deg,
                pattern.elevation_loss_db[band_index, channel],
                color=color,
                alpha=alpha,
                linewidth=1.2,
            )

        band_label = f"{band_hz / 1e9:.1f} GHz" if len(pattern.band_centers_hz) > 1 else "catalog band"
        axes[0].set_title(f"{hardware.name} azimuth loss, {band_label}")
        axes[1].set_title(f"{hardware.name} elevation loss, {band_label}")
        axes[0].set_xlabel("azimuth angle (deg)")
        axes[1].set_xlabel("elevation angle (deg)")
        axes[0].set_ylabel("normalized pattern loss (dB)")
        axes[0].set_ylim(-35, 2)
        for ax in axes:
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
        fig.tight_layout()
        figures.append(fig)
    return figures


def plot_channel_gain_cut(plt, radar_hardware, board_key: str, frequency_hz: float | None = None):
    """Plots per-channel gain versus azimuth and elevation angle."""

    hardware = radar_hardware.from_ti_board(board_key)
    pattern = hardware.antenna_pattern
    band_centers_hz = getattr(pattern, "band_centers_hz", None)
    if frequency_hz is None and band_centers_hz is not None:
        frequency_hz = float(np.mean(band_centers_hz))

    angles = np.linspace(-80.0, 80.0, 321)
    az_gain = hardware.channel_gain(azimuth_deg=angles, frequency_hz=frequency_hz)
    el_gain = hardware.channel_gain(elevation_deg=angles, frequency_hz=frequency_hz)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for channel, label in enumerate(hardware.virtual_channel_labels):
        color = tx_color(label)
        alpha = 0.35 if hardware.num_virtual_channels > 8 else 0.6
        axes[0].plot(angles, 20.0 * np.log10(np.abs(az_gain[channel])), color=color, alpha=alpha)
        axes[1].plot(angles, 20.0 * np.log10(np.abs(el_gain[channel])), color=color, alpha=alpha)

    scalar = hardware.scalar_antenna_gain_dbi or 0.0
    axes[0].set_title(f"{hardware.name} azimuth coefficient gain")
    axes[1].set_title(f"{hardware.name} elevation coefficient gain")
    axes[0].set_ylabel("|channel multiplier| (dB)")
    axes[0].axhline(scalar, color="black", linewidth=0.8, label="scalar gain")
    axes[1].axhline(scalar, color="black", linewidth=0.8, label="scalar gain")
    for ax in axes:
        ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("angle (deg)")
        ax.legend(loc="lower center")
    fig.tight_layout()
    return fig, axes


def print_board_details(radar_hardware, board_key: str) -> None:
    """Prints full metadata and virtual-channel details for one board."""

    hardware = radar_hardware.from_ti_board(board_key)
    pattern = hardware.antenna_pattern

    print(f"\nDetailed board metadata: {hardware.name}")
    print(json.dumps(hardware.board_metadata, indent=2))
    print("\nVirtual channels:")
    for index, (label, pos, sign) in enumerate(
        zip(
            hardware.virtual_channel_labels,
            hardware.virtual_channel_positions_lambda,
            hardware.virtual_channel_phase_signs,
        ),
        start=1,
    ):
        print(
            f"{index:3d}: {label:10s}  y/lambda={pos[0]:6.1f}  "
            f"z/lambda={pos[1]:6.1f}  sign={sign.real:+.0f}"
        )

    if pattern is not None:
        print("\nPattern model:")
        metadata = getattr(pattern, "metadata", None)
        if metadata is not None:
            print(json.dumps(dict(metadata), indent=2))
        else:
            print(type(pattern).__name__)
            print(
                "half-power angle: "
                f"{float(pattern.half_power_angle_deg):.1f} degrees"
            )


if __name__ == "__main__":
    main()
