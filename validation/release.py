# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Release verification: native solver outputs, immutable references and reports."""

import argparse
import json
from pathlib import Path

import numpy as np

from bundle_prepare.contract import Bundle
from mmWaveRadar.provenance import directory_hashes, software_provenance
from .configuration import BenchmarkRunConfig
from .release_reference import compare_reference, freeze_reference, power_features
from .settings import resolve_settings
from .study import safe_name
from .study_metrics import compare
from .study_runtime import file_hash, launch_worker, stable_hash, write_json
from .uncertainty import seed_mean_interval


def prepare_pec_plate(root):
    """Data-free static fixture; its zero ADC is never called a measurement."""
    from bundle_prepare.writer import BundleWriter
    from mmWaveRadar.experiments.physics_microscope import make_rectangular_plate_mesh

    vertices, faces = make_rectangular_plate_mesh(
        range_m=2.0, width_m=0.152, height_m=0.152, max_edge_m=0.02
    )
    with BundleWriter(root) as writer:
        writer.write_json(
            "bundle.json",
            {
                "schema_version": 1,
                "profile": "hermes",
                "primary_adc": "primary",
                "adcs": {
                    "primary": {"path": "radar_adc.npz", "origin": "simulation", "solver": "po"}
                },
                "target": {"mesh": "target_mesh.npz"},
                "scene": {"file": "scene.xml"},
            },
        )
        writer.write_npz("target_mesh.npz", vertices=vertices, faces=faces)
        writer.write_npz("radar_adc.npz", adc=np.zeros((1, 1, 64, 12), dtype=np.complex128))
        writer.write_json(
            "frames.json", {"frames": [{"benchmark_index": 0, "motion_time_s": 0.0}]}
        )
        writer.writestr("scene.xml", '<scene version="3.0.0"/>')
        writer.write_json(
            "sensor.json",
            {
                "board_model": "IWR6843AOPEVM",
                "pattern_mode": "none",
                "fmcw": {
                    "carrier_frequency_hz": 60.476e9,
                    "carrier_reference": "first_adc_sample",
                    "slope_hz_per_s": 1e12,
                    "chirp_duration_s": 40e-6,
                    "chirp_repetition_time_s": 50e-6,
                    "sampling_frequency_hz": 2e6,
                    "num_adc_samples": 64,
                    "num_chirps_per_frame": 1,
                    "frame_period_s": 0.05,
                    "num_tx": 3,
                    "tx_schedule": "tdm",
                    "tdm_tx_order": [0, 1, 2],
                },
            },
        )
    return root


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("Release configuration requires schema_version=1")
    seeds = config.get("seeds", list(range(8)))
    if (
        len(seeds) < 5
        or len(set(seeds)) != len(seeds)
        or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds)
    ):
        raise ValueError("Release runs require at least five distinct nonnegative integer seeds")
    config["seeds"] = seeds
    config["settings"] = resolve_settings(
        {
            "compute_backend": "numpy",
            "compute_precision": "float64",
            "adc_compute_precision": "float64",
            "hybrid_calibration": "none",
            **config.get("settings", {}),
        }
    )
    fixtures = config["fixtures"]
    names = [safe_name(f["id"]) for f in fixtures]
    if not names or len(names) != len(set(names)):
        raise ValueError("Release fixtures need unique identifiers")
    for fixture in fixtures:
        if fixture.get("kind") not in {"synthetic", "measurement"}:
            raise ValueError("Fixture kind must be synthetic or measurement")
        solvers = fixture.setdefault("solvers", ["po", "rt", "hybrid_po"])
        if (
            not solvers
            or len(set(solvers)) != len(solvers)
            or set(solvers) - {"po", "rt", "hybrid_po"}
        ):
            raise ValueError("Invalid release solver list")
        if fixture.get("bundle"):
            fixture["bundle"] = str((path.parent / fixture["bundle"]).resolve())
        elif fixture.get("generate") != "pec_plate":
            raise ValueError("Fixture requires a bundle or generate=pec_plate")
    return config


def run_suite(config, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Run output already exists; use a fresh directory")
    output.mkdir(parents=True)
    protocol = {
        "schema_version": 1,
        "seeds": config["seeds"],
        "settings": config["settings"],
        "fixtures": [],
    }
    records = []
    provenance = software_provenance()
    for fixture in config["fixtures"]:
        name = fixture["id"]
        bundle = fixture.get("bundle") or prepare_pec_plate(output / "inputs" / name)
        clip = Bundle.load(bundle)
        files = directory_hashes(clip.root, exclude=("provenance.json",))
        protocol["fixtures"].append(
            {
                "id": name,
                "kind": fixture["kind"],
                "input_hashes": files,
                "solvers": fixture["solvers"],
                "target_material": fixture.get("target_material", {}),
                "angle_deg": fixture.get("angle_deg"),
                "angular_study": fixture.get("angular_study"),
                "range_gate_m": fixture.get("range_gate_m"),
            }
        )
        if (
            fixture["kind"] == "measurement"
            and clip.descriptor["adcs"][clip.descriptor["primary_adc"]].get("origin")
            != "measurement"
        ):
            raise ValueError(f"{name}: primary ADC is not declared as measurement")
        for solver in fixture["solvers"]:
            for seed in ([config["seeds"][0]] if solver == "po" else config["seeds"]):
                print(f"[release] {name} / {solver} / seed {seed}", flush=True)
                folder = output / name / solver / str(seed)
                settings = {**config["settings"], "rt_seed": seed}
                runtime = launch_worker(
                    {
                        "directory": str(folder),
                        "phase": "accuracy",
                        "bundle": str(bundle),
                        "solver": solver,
                        "settings": settings,
                        "target_material": fixture.get("target_material", {}),
                    },
                    cpu_reference=True,
                )
                adc_path = folder / "adc.npz"
                record = {
                    "fixture": name,
                    "kind": fixture["kind"],
                    "solver": solver,
                    "seed": seed,
                    "adc": adc_path.relative_to(output).as_posix(),
                    "adc_sha256": file_hash(adc_path),
                    "runtime": runtime,
                    "angle_deg": fixture.get("angle_deg"),
                    "angular_study": fixture.get("angular_study"),
                }
                with np.load(adc_path, allow_pickle=False) as arrays:
                    adc = arrays["adc"]
                    if fixture["kind"] == "measurement":
                        record["measurement"], _ = compare(
                            clip,
                            BenchmarkRunConfig(
                                solvers=(solver,),
                                solver_settings=settings,
                                hybrid_calibration="none",
                            ),
                            adc,
                            clip.real_adc,
                            coherent=False,
                            absolute_power=False,
                        )
                    if fixture.get("angular_study"):
                        record["peak_power"] = gated_peak_power(
                            adc, clip.sensor_config.fmcw, fixture["range_gate_m"]
                        )
                        record["measured_peak_power"] = gated_peak_power(
                            clip.real_adc, clip.sensor_config.fmcw, fixture["range_gate_m"]
                        )
                records.append(record)
    # Release compatibility excludes source version deliberately, but includes dependencies,
    # native tracing variant, backend and processing settings. Hashes are path independent.
    protocol["dependencies"] = dict(provenance["dependencies"])
    protocol["dependencies"].pop("hermes-radar-sim", None)
    protocol["execution"] = [
        {k: r["runtime"][k] for k in ("resolved_backends", "accuracy_cpu_threads", "adc_dtype")}
        for r in records
    ]
    result = {
        "schema_version": 1,
        "provenance": provenance,
        "protocol": protocol,
        "protocol_sha256": stable_hash(protocol),
        "records": records,
    }
    write_json(output / "run.json", result)
    write_report(output)
    return result


def gated_peak_power(adc, fmcw, gate):
    """Copper-plate convention: mean of per-pair peak powers, Blackman / 1024 FFT."""
    from scipy.constants import c

    spectrum = np.fft.fft(adc * np.blackman(adc.shape[2])[None, None, :, None], n=1024, axis=2)
    ranges = np.fft.fftfreq(1024, 1 / fmcw.sampling_frequency) * c / (2 * fmcw.slope)
    selected = (ranges >= gate[0]) & (ranges <= gate[1])
    if not np.any(selected):
        raise ValueError("Range gate contains no FFT bins")
    return float(np.mean(np.max(abs(spectrum[:, :, selected]) ** 2, axis=2)))


def write_report(output, *, comparison=None, analytic=None, rt_convergence=None):
    output = Path(output)
    run = json.loads((output / "run.json").read_text())
    from .release_reference import grouped_records, checked_adc

    summaries = []
    source_snapshots = sorted(
        {
            r.get("runtime", {}).get("provenance", {}).get("source_sha256")
            for r in run["records"]
            if r.get("runtime", {}).get("provenance", {}).get("source_sha256")
        }
    )
    lines = [
        f"# HERMES {run['provenance']['version']} validation report",
        "",
        f"Commit: `{run['provenance']['git_commit']}`; dirty checkout: {run['provenance']['git_dirty']}.",
        "",
        "Regression, analytic verification and measurement agreement are separate claims.",
        "",
        "| Fixture | Kind | Solver | Seeds | Mean power | 95% CI for seed mean | Measurement range-time relative L2 |",
        "| --- | --- | --- | ---: | ---: | --- | ---: |",
    ]
    angular = {}
    if len(source_snapshots) > 1:
        lines += [
            "",
            f"Development evidence spans {len(source_snapshots)} worker source snapshots. It cannot be frozen as a release reference; per-seed source hashes are retained.",
            "",
        ]
    for (fixture, solver), records in grouped_records(run).items():
        powers = [float(np.mean(power_features(checked_adc(output, r)))) for r in records]
        interval = seed_mean_interval(powers)
        measurement = [
            r["measurement"]["range_time"]["relative_l2"] for r in records if r.get("measurement")
        ]
        error = seed_mean_interval(measurement) if measurement else None
        summaries.append(
            {
                "fixture": fixture,
                "solver": solver,
                "kind": records[0]["kind"],
                "mean_linear_power": interval,
                "measurement_range_time_error": error,
                "per_seed_metrics": [
                    {
                        "seed": r["seed"],
                        "measurement": r.get("measurement"),
                        "adc_sha256": r["adc_sha256"],
                        "producer_source_sha256": r.get("runtime", {})
                        .get("provenance", {})
                        .get("source_sha256"),
                        "warnings": r.get("runtime", {}).get("warnings", []),
                    }
                    for r in records
                ],
            }
        )
        ci = (
            f"[{float(interval['lower']):.5g}, {float(interval['upper']):.5g}]"
            if interval["lower"] is not None
            else "not estimated"
        )
        lines.append(
            f"| {fixture} | {records[0]['kind']} | {solver} | {len(records)} | {float(interval['mean']):.5g} | {ci} | {float(error['mean']) if error else 'not measured'} |"
        )
        if records[0].get("angular_study"):
            angular.setdefault((records[0]["angular_study"], solver), []).append(
                {
                    "angle_deg": records[0]["angle_deg"],
                    "simulated_power": float(np.mean([r["peak_power"] for r in records])),
                    "measured_power": records[0]["measured_peak_power"],
                }
            )
    angular_rows = []
    for (name, solver), rows in angular.items():
        zero = next((r for r in rows if r["angle_deg"] == 0), None)
        if zero is None:
            raise ValueError("Angular studies need a zero-degree fixture")
        errors = []
        missing = []
        for row in sorted(rows, key=lambda r: r["angle_deg"]):
            for kind in ("measured", "simulated"):
                row[f"{kind}_db_relative_0"] = (
                    float(10 * np.log10(row[f"{kind}_power"] / zero[f"{kind}_power"]))
                    if min(row[f"{kind}_power"], zero[f"{kind}_power"]) > 0
                    else None
                )
            if row["measured_db_relative_0"] is None or row["simulated_db_relative_0"] is None:
                missing.append(row["angle_deg"])
            else:
                errors.append(row["simulated_db_relative_0"] - row["measured_db_relative_0"])
        rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else None
        angular_rows.append(
            {
                "study": name,
                "solver": solver,
                "angular_response_rmse_db": rmse if not missing else None,
                "detected_angle_rmse_db": rmse,
                "missing_return_angles_deg": missing,
                "angles": rows,
            }
        )
        value = (
            f"{rmse:.4g} dB"
            if not missing
            else f"not estimable over all angles (zero return at {missing})"
        )
        lines += [
            "",
            f"{name}, {solver}: relative angular-response RMSE = {value}; one alignment at 0°. Absolute gain is not validated.",
        ]
    result = {
        "provenance": run["provenance"],
        "protocol": run["protocol"],
        "protocol_sha256": run["protocol_sha256"],
        "worker_source_sha256s": source_snapshots,
        "fixtures": summaries,
        "angular_studies": angular_rows,
        "regression": comparison,
        "analytic": analytic,
        "rt_convergence": rt_convergence,
    }
    lines += [
        "",
        "Confidence intervals describe simulation-seed means only (pointwise Student-t). They exclude experimental uncertainty and model bias.",
        "",
        f"Release regression: {('PASS' if comparison['passed'] else 'FAIL') if comparison else 'not run against a frozen release'}.",
        f"Analytic verification: {'included in report.json' if analytic else 'not supplied'}.",
    ]
    lines.append(
        f"RT ray convergence: {'included in report.json' if rt_convergence else 'not supplied'}."
    )
    write_json(output / "report.json", result)
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    run = actions.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    freeze = actions.add_parser("freeze")
    freeze.add_argument("--run", type=Path, required=True)
    freeze.add_argument("--reference", type=Path, required=True)
    freeze.add_argument("--release", required=True)
    compare_parser = actions.add_parser("compare")
    compare_parser.add_argument("--run", type=Path, required=True)
    compare_parser.add_argument("--reference", type=Path, required=True)
    report = actions.add_parser("report")
    report.add_argument("--run", type=Path, required=True)
    report.add_argument("--analytic", type=Path)
    report.add_argument("--rt-convergence", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "run":
            run_suite(load_config(args.config), args.output)
        elif args.action == "freeze":
            freeze_reference(args.run, args.reference, args.release)
        elif args.action == "compare":
            comparison = compare_reference(args.reference, args.run)
            write_json(args.run / "comparison.json", comparison)
            write_report(args.run, comparison=comparison)
            return 0 if comparison["passed"] else 1
        else:
            comparison_path = args.run / "comparison.json"
            write_report(
                args.run,
                comparison=(
                    json.loads(comparison_path.read_text()) if comparison_path.exists() else None
                ),
                analytic=json.loads(args.analytic.read_text()) if args.analytic else None,
                rt_convergence=(
                    json.loads(args.rt_convergence.read_text()) if args.rt_convergence else None
                ),
            )
    except (KeyError, ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"release validation: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
