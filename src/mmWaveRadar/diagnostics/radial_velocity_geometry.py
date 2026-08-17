# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Geometry-only human radial-velocity diagnostic.

This module intentionally does not run RT/PO field synthesis. It projects mesh
motion onto the equivalent bistatic range-rate direction for a requested radar
pose, making it useful as a quick sanity check for Doppler interpretations.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..radar import FMCWConfig, RadarHardware, RadarSensor
from ..simulation.motion_diagnostics import (
    RadialVelocitySamples,
    mesh_sequence_radial_velocity_samples,
    radial_velocity_summary,
)
from ..targets import MeshSequence, amass_to_mesh_sequence, load_amass_npz


_LIGHT_SPEED_MPS = 299792458.0
_DEFAULT_PERCENTILES = (50.0, 90.0, 95.0, 99.0, 99.9, 100.0)
_DEFAULT_POINT_KINDS = ("face_centroid", "vertex")


@dataclass(frozen=True)
class RadialVelocityDiagnosticConfig:
    """Configuration for geometry-only radial velocity profiling."""

    mesh_sequence_npz: Path | None = None
    amass_npz: Path | None = None
    smpl_model_dir: Path | None = None
    model_type: str = "smpl"
    device: str = "cpu"
    radar_position: tuple[float, float, float] | None = None
    radar_orientation: tuple[float, float, float] | None = None
    look_at: tuple[float, float, float] | None = None
    chest_clearance_m: float = 0.50
    hardware_model: str = "single-bistatic-lambda"
    carrier_frequency_hz: float = 60e9
    tx_positions: np.ndarray | None = None
    rx_positions: np.ndarray | None = None
    point_kinds: tuple[str, ...] = _DEFAULT_POINT_KINDS
    time_start_s: float | None = None
    time_stop_s: float | None = None
    duration_s: float | None = None
    virtual_channel_order: str = "tx_major"
    percentiles: tuple[float, ...] = _DEFAULT_PERCENTILES


@dataclass(frozen=True)
class RadialVelocityDiagnosticResult:
    """Geometry-only radial velocity result for one motion/radar setup."""

    source_path: Path
    mesh_shape: tuple[int, int, int]
    face_count: int
    mesh_time_span_s: tuple[float, float]
    radar_position: np.ndarray
    radar_orientation: np.ndarray
    tx_positions: np.ndarray
    rx_positions: np.ndarray
    virtual_channel_order: str
    summaries: dict[str, dict[str, Any]]
    samples: dict[str, RadialVelocitySamples]


def run_radial_velocity_diagnostic(
    config: RadialVelocityDiagnosticConfig,
) -> RadialVelocityDiagnosticResult:
    """Runs the geometry-only radial velocity diagnostic."""

    mesh_sequence, source_path, amass_data = _load_mesh_sequence(config)
    radar_position, radar_orientation = _resolve_radar_pose(
        mesh_sequence,
        config,
        amass_data,
    )
    radar = _make_radar_sensor(config, radar_position, radar_orientation)
    time_start_s, time_stop_s = _resolve_time_window(config)

    summaries: dict[str, dict[str, Any]] = {}
    samples_by_kind: dict[str, RadialVelocitySamples] = {}
    for point_kind in _normalize_point_kinds(config.point_kinds):
        samples = mesh_sequence_radial_velocity_samples(
            mesh_sequence,
            tx_positions=radar.world_tx_positions(),
            rx_positions=radar.world_rx_positions(),
            point_kind=point_kind,
            time_start_s=time_start_s,
            time_stop_s=time_stop_s,
            virtual_channel_order=radar.hardware.virtual_channel_order,
            virtual_channel_tx_indices=radar.hardware.virtual_tx_indices(),
            virtual_channel_rx_indices=radar.hardware.virtual_rx_indices(),
        )
        samples_by_kind[point_kind] = samples
        summaries[point_kind] = radial_velocity_summary(
            samples,
            percentiles=tuple(config.percentiles),
        )

    times = np.asarray(mesh_sequence.times, dtype=float)
    return RadialVelocityDiagnosticResult(
        source_path=source_path,
        mesh_shape=tuple(int(v) for v in mesh_sequence.vertices.shape),
        face_count=int(mesh_sequence.faces.shape[0]),
        mesh_time_span_s=(float(times[0]), float(times[-1])),
        radar_position=np.asarray(radar.position, dtype=float),
        radar_orientation=np.asarray(radar.orientation, dtype=float),
        tx_positions=radar.world_tx_positions(),
        rx_positions=radar.world_rx_positions(),
        virtual_channel_order=radar.hardware.virtual_channel_order,
        summaries=summaries,
        samples=samples_by_kind,
    )


def format_radial_velocity_report(result: RadialVelocityDiagnosticResult) -> str:
    """Formats a human-readable report for terminal use."""

    lines = [
        "Geometry-only radial velocity diagnostic",
        f"  source: {result.source_path}",
        (
            "  mesh: "
            f"vertices={result.mesh_shape}, faces={result.face_count}, "
            f"time={result.mesh_time_span_s[0]:.3f}.."
            f"{result.mesh_time_span_s[1]:.3f} s"
        ),
        (
            "  radar position: "
            f"{np.array2string(result.radar_position, precision=4)}"
        ),
        (
            "  radar orientation: "
            f"{np.array2string(result.radar_orientation, precision=4)}"
        ),
        (
            "  channels: "
            f"tx={result.tx_positions.shape[0]}, "
            f"rx={result.rx_positions.shape[0]}, "
            f"order={result.virtual_channel_order}"
        ),
    ]

    for point_kind, summary in result.summaries.items():
        abs_pct = summary["abs_percentiles_mps"]
        signed_pct = summary["signed_percentiles_mps"]
        lines.extend([
            "",
            point_kind,
            (
                f"  samples={summary['sample_count']} "
                f"time={summary['time_start_s']:.3f}.."
                f"{summary['time_stop_s']:.3f} s"
            ),
            (
                "  signed min/mean/max [m/s]: "
                f"{summary['signed_min_mps']:.3f}, "
                f"{summary['signed_mean_mps']:.3f}, "
                f"{summary['signed_max_mps']:.3f}"
            ),
            f"  |v| mean [m/s]: {summary['abs_mean_mps']:.3f}",
            (
                "  |v| percentiles [m/s]: "
                + ", ".join(
                    f"p{_format_percentile_key(p)}={value:.3f}"
                    for p, value in abs_pct.items()
                )
            ),
            (
                "  signed percentiles [m/s]: "
                + ", ".join(
                    f"p{_format_percentile_key(p)}={value:.3f}"
                    for p, value in signed_pct.items()
                )
            ),
            (
                "  fraction |v| >= 5/10/20 m/s: "
                f"{summary['fraction_abs_ge_5_mps']:.3e}, "
                f"{summary['fraction_abs_ge_10_mps']:.3e}, "
                f"{summary['fraction_abs_ge_20_mps']:.3e}"
            ),
        ])
    return "\n".join(lines)


def save_radial_velocity_diagnostic_plot(
    result: RadialVelocityDiagnosticResult,
    path: str | Path,
) -> None:
    """Saves histogram and per-time radial-velocity diagnostics as a PNG/PDF."""

    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    point_kinds = tuple(result.samples)
    fig, axes = plt.subplots(
        3,
        len(point_kinds),
        figsize=(6.0 * len(point_kinds), 10.0),
        squeeze=False,
        constrained_layout=True,
    )
    for col, point_kind in enumerate(point_kinds):
        samples = result.samples[point_kind]
        values, weights = _flattened_values_and_weights(samples)
        axes[0, col].hist(
            values,
            bins=160,
            weights=weights,
            density=True,
            color="#3b6ea8",
            alpha=0.85,
        )
        axes[0, col].axvspan(5.0, 20.0, color="#c44e52", alpha=0.12)
        axes[0, col].axvspan(-20.0, -5.0, color="#c44e52", alpha=0.12)
        axes[0, col].set_title(f"{point_kind}: signed")
        axes[0, col].set_xlabel("equiv. radial velocity [m/s]")
        axes[0, col].set_ylabel("weighted density")
        axes[0, col].grid(True, alpha=0.25)

        axes[1, col].hist(
            np.abs(values),
            bins=120,
            weights=weights,
            density=True,
            color="#4f9d69",
            alpha=0.85,
        )
        axes[1, col].axvspan(5.0, 20.0, color="#c44e52", alpha=0.12)
        axes[1, col].set_title(f"{point_kind}: absolute")
        axes[1, col].set_xlabel("|equiv. radial velocity| [m/s]")
        axes[1, col].set_ylabel("weighted density")
        axes[1, col].grid(True, alpha=0.25)

        stats = _per_time_abs_stats(samples)
        for key, style in (("p50", "-"), ("p95", "-"), ("p99", "-"), ("max", "--")):
            axes[2, col].plot(samples.times_s, stats[key], style, label=key)
        axes[2, col].axhspan(5.0, 20.0, color="#c44e52", alpha=0.12)
        axes[2, col].set_title(f"{point_kind}: time profile")
        axes[2, col].set_xlabel("time [s]")
        axes[2, col].set_ylabel("|equiv. radial velocity| [m/s]")
        axes[2, col].grid(True, alpha=0.25)
        axes[2, col].legend(loc="best")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_radial_velocity_samples_npz(
    result: RadialVelocityDiagnosticResult,
    path: str | Path,
) -> None:
    """Saves diagnostic samples and summaries for later offline analysis."""

    payload: dict[str, Any] = {
        "source_path": np.asarray(str(result.source_path)),
        "mesh_shape": np.asarray(result.mesh_shape, dtype=np.int64),
        "face_count": np.asarray(result.face_count, dtype=np.int64),
        "mesh_time_span_s": np.asarray(result.mesh_time_span_s, dtype=float),
        "radar_position": np.asarray(result.radar_position, dtype=float),
        "radar_orientation": np.asarray(result.radar_orientation, dtype=float),
        "tx_positions": np.asarray(result.tx_positions, dtype=float),
        "rx_positions": np.asarray(result.rx_positions, dtype=float),
        "virtual_channel_order": np.asarray(result.virtual_channel_order),
        "point_kinds": np.asarray(tuple(result.samples)),
        "summary_json": np.asarray(json.dumps(result.summaries, sort_keys=True)),
    }
    for point_kind, samples in result.samples.items():
        key = point_kind.replace("-", "_")
        payload[f"{key}_times_s"] = np.asarray(samples.times_s, dtype=float)
        payload[f"{key}_radial_velocity_mps"] = np.asarray(
            samples.radial_velocity_mps,
            dtype=float,
        )
        payload[f"{key}_weights"] = np.asarray(samples.weights, dtype=float)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    tx_positions = args.tx_positions
    rx_positions = args.rx_positions
    if (tx_positions is None) != (rx_positions is None):
        parser.error("--tx-positions and --rx-positions must be supplied together")

    smpl_model_dir = Path(args.smpl_model_dir).expanduser() if args.smpl_model_dir else None
    if smpl_model_dir is None and args.amass_npz:
        env_dir = os.environ.get("MMWAVE_SMPL_MODEL_DIR")
        if env_dir:
            smpl_model_dir = Path(env_dir).expanduser()

    config = RadialVelocityDiagnosticConfig(
        mesh_sequence_npz=Path(args.mesh_sequence_npz).expanduser()
        if args.mesh_sequence_npz
        else None,
        amass_npz=Path(args.amass_npz).expanduser() if args.amass_npz else None,
        smpl_model_dir=smpl_model_dir,
        model_type=args.model_type,
        device=args.device,
        radar_position=args.radar_position,
        radar_orientation=args.radar_orientation,
        look_at=args.look_at,
        chest_clearance_m=float(args.chest_clearance_m),
        hardware_model=args.hardware_model,
        carrier_frequency_hz=float(args.carrier_frequency_hz),
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        point_kinds=_normalize_point_kinds(tuple(args.point_kind or ("both",))),
        time_start_s=args.time_start_s,
        time_stop_s=args.time_stop_s,
        duration_s=args.duration_s,
        virtual_channel_order=args.virtual_channel_order,
    )
    result = run_radial_velocity_diagnostic(config)
    print(format_radial_velocity_report(result))

    if args.json:
        write_radial_velocity_json_report(result, args.json)
    if args.samples_npz:
        save_radial_velocity_samples_npz(result, args.samples_npz)
    if args.plot:
        save_radial_velocity_diagnostic_plot(result, args.plot)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Geometry-only human radial velocity diagnostic for AMASS/SMPL or "
            "precomputed mesh sequences."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--mesh-sequence-npz",
        help="NPZ with vertices, faces, and times arrays.",
    )
    source.add_argument("--amass-npz", help="AMASS/SMPL motion NPZ.")
    parser.add_argument(
        "--smpl-model-dir",
        help=(
            "SMPL model directory for --amass-npz. Defaults to "
            "MMWAVE_SMPL_MODEL_DIR when set."
        ),
    )
    parser.add_argument(
        "--model-type",
        default="smpl",
        choices=("smpl", "smplh", "smplx"),
        help="SMPL-family model type for --amass-npz.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device for AMASS/SMPL.")

    parser.add_argument(
        "--radar-position",
        type=parse_vector3,
        help=(
            "Radar phase-center position as x,y,z [m]. If omitted, the tool "
            "places the radar in front of the first-frame chest estimate."
        ),
    )
    parser.add_argument(
        "--radar-orientation",
        type=parse_vector3,
        help="Radar yaw,pitch,roll in radians. Overrides --look-at.",
    )
    parser.add_argument(
        "--look-at",
        type=parse_vector3,
        help="Point x,y,z [m] used to orient the radar when orientation is omitted.",
    )
    parser.add_argument(
        "--chest-clearance-m",
        type=float,
        default=0.50,
        help="Clearance used by automatic chest-facing radar placement.",
    )
    parser.add_argument(
        "--hardware-model",
        default="single-bistatic-lambda",
        help=(
            "Hardware model. Use single-bistatic-lambda or a RadarHardware "
            "xWR68xx model such as IWR6843AOP."
        ),
    )
    parser.add_argument(
        "--carrier-frequency-hz",
        type=float,
        default=60e9,
        help="Carrier frequency used for the default single-bistatic separation.",
    )
    parser.add_argument(
        "--tx-positions",
        type=parse_positions_matrix,
        help=(
            "Explicit local Tx positions in meters, e.g. '0,0,0' or "
            "'0,0,0;0,0.002,0'. Requires --rx-positions."
        ),
    )
    parser.add_argument(
        "--rx-positions",
        type=parse_positions_matrix,
        help="Explicit local Rx positions in meters. Requires --tx-positions.",
    )
    parser.add_argument(
        "--virtual-channel-order",
        default="tx_major",
        choices=("tx_major", "rx_major"),
    )

    parser.add_argument(
        "--point-kind",
        action="append",
        choices=("face_centroid", "vertex", "both"),
        help="Point set to profile. Repeatable; defaults to both.",
    )
    parser.add_argument("--time-start-s", type=float, default=None)
    parser.add_argument("--time-stop-s", type=float, default=None)
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Duration from --time-start-s. Mutually exclusive with --time-stop-s.",
    )
    parser.add_argument("--json", help="Optional JSON report path.")
    parser.add_argument("--samples-npz", help="Optional NPZ sample output path.")
    parser.add_argument("--plot", help="Optional PNG/PDF plot output path.")
    return parser


def write_radial_velocity_json_report(
    result: RadialVelocityDiagnosticResult,
    path: str | Path,
) -> None:
    """Writes a compact JSON diagnostic report."""

    report = {
        "source_path": str(result.source_path),
        "mesh_shape": list(result.mesh_shape),
        "face_count": result.face_count,
        "mesh_time_span_s": list(result.mesh_time_span_s),
        "radar_position": np.asarray(result.radar_position, dtype=float).tolist(),
        "radar_orientation": np.asarray(result.radar_orientation, dtype=float).tolist(),
        "tx_positions": np.asarray(result.tx_positions, dtype=float).tolist(),
        "rx_positions": np.asarray(result.rx_positions, dtype=float).tolist(),
        "virtual_channel_order": result.virtual_channel_order,
        "summaries": result.summaries,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def parse_vector3(value: str) -> tuple[float, float, float]:
    """Parses a comma-separated 3-vector."""

    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected x,y,z, got {value!r}")
    try:
        return tuple(float(part) for part in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected numeric x,y,z, got {value!r}") from exc


def parse_positions_matrix(value: str) -> np.ndarray:
    """Parses semicolon-separated 3-vectors into an ``[N, 3]`` array."""

    rows = [row.strip() for row in str(value).split(";") if row.strip()]
    if not rows:
        raise argparse.ArgumentTypeError("expected at least one x,y,z row")
    return np.asarray([parse_vector3(row) for row in rows], dtype=float)


def _load_mesh_sequence(
    config: RadialVelocityDiagnosticConfig,
) -> tuple[MeshSequence, Path, Mapping[str, Any] | None]:
    """Loads exactly one mesh source and returns optional AMASS metadata."""

    has_mesh = config.mesh_sequence_npz is not None
    has_amass = config.amass_npz is not None
    if has_mesh == has_amass:
        raise ValueError("Provide exactly one of mesh_sequence_npz or amass_npz")

    if config.mesh_sequence_npz is not None:
        path = Path(config.mesh_sequence_npz).expanduser()
        return MeshSequence.from_npz(str(path)), path, None

    if config.smpl_model_dir is None:
        raise ValueError(
            "smpl_model_dir is required for AMASS input unless "
            "MMWAVE_SMPL_MODEL_DIR is set in the CLI."
        )
    path = Path(config.amass_npz).expanduser()  # type: ignore[arg-type]
    smpl_model_dir = Path(config.smpl_model_dir).expanduser()
    amass_data = load_amass_npz(str(path))
    mesh_sequence = amass_to_mesh_sequence(
        str(path),
        str(smpl_model_dir),
        model_type=config.model_type,
        device=config.device,
    )
    return mesh_sequence, path, amass_data


def _resolve_radar_pose(
    mesh_sequence: MeshSequence,
    config: RadialVelocityDiagnosticConfig,
    amass_data: Mapping[str, Any] | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Resolves explicit, look-at, or chest-facing radar pose settings."""

    if config.radar_position is None:
        radar_position, radar_orientation, _, _ = radar_pose_in_front_of_chest(
            mesh_sequence,
            clearance_m=config.chest_clearance_m,
            amass_data=amass_data,
        )
    else:
        radar_position = tuple(float(v) for v in config.radar_position)
        if config.radar_orientation is not None:
            radar_orientation = tuple(float(v) for v in config.radar_orientation)
        elif config.look_at is not None:
            radar_orientation = _orientation_toward(config.look_at, radar_position)
        else:
            radar_orientation = (0.0, 0.0, 0.0)

    if config.radar_orientation is not None:
        radar_orientation = tuple(float(v) for v in config.radar_orientation)
    elif config.look_at is not None:
        radar_orientation = _orientation_toward(config.look_at, radar_position)

    return (
        tuple(float(v) for v in radar_position),
        tuple(float(v) for v in radar_orientation),
    )


def radar_pose_in_front_of_chest(
    mesh_sequence: MeshSequence,
    *,
    clearance_m: float = 0.50,
    amass_data: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, tuple[float, float, float], np.ndarray, np.ndarray]:
    """Estimates a first-frame chest-facing radar pose."""

    vertices = mesh_sequence.vertices_at(float(mesh_sequence.times[0]))
    chest = _smpl_chest_front(vertices, amass_data)
    if chest is None:
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        chest_front_point = np.array(
            [mins[0], 0.5 * (mins[1] + maxs[1]), 0.65 * maxs[2]],
            dtype=float,
        )
        front_dir = np.array([-1.0, 0.0, 0.0], dtype=float)
    else:
        chest_front_point, front_dir = chest
    radar_position = chest_front_point + float(clearance_m) * front_dir
    radar_orientation = _orientation_toward(chest_front_point, radar_position)
    return radar_position, radar_orientation, chest_front_point, front_dir


def _make_radar_sensor(
    config: RadialVelocityDiagnosticConfig,
    radar_position: tuple[float, float, float],
    radar_orientation: tuple[float, float, float],
) -> RadarSensor:
    """Builds the diagnostic radar sensor using configured pose and hardware."""

    hardware = _make_hardware(config)
    fmcw = FMCWConfig(
        carrier_frequency=float(config.carrier_frequency_hz),
        slope=68e12,
        chirp_duration=58e-6,
        chirp_repetition_time=65e-6,
        sampling_frequency=4.5e6,
        num_adc_samples=225,
        num_chirps_per_frame=64,
        frame_period=50e-3,
        num_tx=hardware.num_tx,
    )
    return RadarSensor(
        name="radar",
        position=radar_position,
        orientation=radar_orientation,
        hardware=hardware,
        fmcw=fmcw,
    )


def _make_hardware(config: RadialVelocityDiagnosticConfig) -> RadarHardware:
    """Resolves custom, synthetic, or catalog radar hardware for diagnostics."""

    if config.tx_positions is not None or config.rx_positions is not None:
        if config.tx_positions is None or config.rx_positions is None:
            raise ValueError("tx_positions and rx_positions must be supplied together")
        return RadarHardware.from_positions(
            config.tx_positions,
            config.rx_positions,
            name="custom",
            virtual_channel_order=config.virtual_channel_order,
        )

    if config.hardware_model == "single-bistatic-lambda":
        wavelength = _LIGHT_SPEED_MPS / float(config.carrier_frequency_hz)
        return RadarHardware.from_positions(
            tx_positions=[[0.0, -0.5 * wavelength, 0.0]],
            rx_positions=[[0.0, 0.5 * wavelength, 0.0]],
            name="single-bistatic-lambda",
            virtual_channel_order=config.virtual_channel_order,
        )

    hardware = RadarHardware.from_xwr68xx(config.hardware_model)
    return RadarHardware(
        name=hardware.name,
        tx_positions=hardware.tx_positions,
        rx_positions=hardware.rx_positions,
        virtual_channel_order=config.virtual_channel_order,
    )


def _resolve_time_window(
    config: RadialVelocityDiagnosticConfig,
) -> tuple[float | None, float | None]:
    """Converts start/stop or start/duration options into a time interval."""

    if config.time_stop_s is not None and config.duration_s is not None:
        raise ValueError("time_stop_s and duration_s are mutually exclusive")
    time_start_s = config.time_start_s
    if config.duration_s is None:
        return time_start_s, config.time_stop_s
    if config.duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    start = 0.0 if time_start_s is None else float(time_start_s)
    return start, start + float(config.duration_s)


def _normalize_point_kinds(point_kinds: Sequence[str]) -> tuple[str, ...]:
    """Normalizes requested radial-velocity point families."""

    if not point_kinds or "both" in point_kinds:
        return _DEFAULT_POINT_KINDS
    valid = set(_DEFAULT_POINT_KINDS)
    normalized = []
    for point_kind in point_kinds:
        if point_kind not in valid:
            raise ValueError("point_kind must be 'face_centroid', 'vertex', or 'both'")
        if point_kind not in normalized:
            normalized.append(point_kind)
    return tuple(normalized)


def _orientation_toward(
    target_point: Sequence[float],
    radar_position: Sequence[float],
) -> tuple[float, float, float]:
    """Returns Euler angles that point the radar boresight at ``target_point``."""

    direction = np.asarray(target_point, dtype=float) - np.asarray(
        radar_position,
        dtype=float,
    )
    norm = np.linalg.norm(direction)
    if norm <= 0.0:
        raise ValueError("target point must differ from radar position")
    direction /= norm
    theta = np.arccos(np.clip(direction[2], -1.0, 1.0))
    phi = np.arctan2(direction[1], direction[0])
    return (float(phi), float(theta - 0.5 * np.pi), 0.0)


def _smpl_chest_front(
    vertices: np.ndarray,
    amass_data: Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Estimates a chest-front point and horizontal front direction from markers."""

    markers = _amass_marker_dict(amass_data)
    if not markers:
        return None
    sternum = _mean_marker(markers, ("STRN", "CLAV"))
    front_ref = _mean_marker(markers, ("STRN", "LFSH", "RFSH"))
    back_ref = _mean_marker(markers, ("TOPBACK", "MIDBACK", "LBSH", "RBSH"))
    if sternum is None or front_ref is None or back_ref is None:
        return None
    front_xy = front_ref[:2] - back_ref[:2]
    front_norm = np.linalg.norm(front_xy)
    if front_norm < 1e-6:
        return None
    front_dir_xy = front_xy / front_norm
    lateral_dir_xy = np.array([-front_dir_xy[1], front_dir_xy[0]])
    z_span = max(float(vertices[:, 2].max() - vertices[:, 2].min()), 1e-6)
    lateral = vertices[:, :2] @ lateral_dir_xy
    lateral_center = float(sternum[:2] @ lateral_dir_xy)
    torso = (
        (np.abs(vertices[:, 2] - sternum[2]) <= max(0.12, 0.12 * z_span))
        & (np.abs(lateral - lateral_center) <= 0.30)
    )
    if np.count_nonzero(torso) < 16:
        torso = np.ones(vertices.shape[0], dtype=bool)
    proj = vertices[:, :2] @ front_dir_xy
    front_surface_proj = float(np.max(proj[torso]))
    sternum_proj = float(sternum[:2] @ front_dir_xy)
    chest_front_xy = sternum[:2] + (front_surface_proj - sternum_proj) * front_dir_xy
    chest_front_point = np.array(
        [chest_front_xy[0], chest_front_xy[1], sternum[2]],
        dtype=float,
    )
    front_dir = np.array([front_dir_xy[0], front_dir_xy[1], 0.0], dtype=float)
    return chest_front_point, front_dir


def _amass_marker_dict(amass_data: Mapping[str, Any] | None) -> dict[str, np.ndarray]:
    """Returns first-frame AMASS marker positions keyed by marker label."""

    if amass_data is None:
        return {}
    if "marker_data" not in amass_data or "marker_labels" not in amass_data:
        return {}
    labels = [
        label.decode("utf-8") if isinstance(label, bytes) else str(label)
        for label in amass_data["marker_labels"]
    ]
    points = np.asarray(amass_data["marker_data"], dtype=float)[0]
    return dict(zip(labels, points))


def _mean_marker(
    markers: Mapping[str, np.ndarray],
    names: Sequence[str],
) -> np.ndarray | None:
    """Averages available markers from ``names`` or returns ``None``."""

    points = [markers[name] for name in names if name in markers]
    if not points:
        return None
    return np.mean(np.asarray(points, dtype=float), axis=0)


def _flattened_values_and_weights(samples: RadialVelocitySamples) -> tuple[np.ndarray, np.ndarray]:
    """Flattens radial-velocity samples and repeats face/vertex weights per channel."""

    values = np.asarray(samples.radial_velocity_mps, dtype=float).reshape(-1)
    channel_count = int(samples.radial_velocity_mps.shape[-1])
    weights = np.repeat(np.asarray(samples.weights, dtype=float).reshape(-1), channel_count)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    return values[finite], weights[finite]


def _per_time_abs_stats(samples: RadialVelocitySamples) -> dict[str, np.ndarray]:
    """Computes per-time absolute radial-velocity percentiles and maxima."""

    values = np.abs(np.asarray(samples.radial_velocity_mps, dtype=float))
    flat = values.reshape((values.shape[0], -1))
    return {
        "p50": np.percentile(flat, 50.0, axis=1),
        "p95": np.percentile(flat, 95.0, axis=1),
        "p99": np.percentile(flat, 99.0, axis=1),
        "max": np.max(flat, axis=1),
    }


def _format_percentile_key(value: Any) -> str:
    """Formats percentile values as stable compact dictionary keys."""

    return f"{float(value):g}"


if __name__ == "__main__":
    raise SystemExit(main())
