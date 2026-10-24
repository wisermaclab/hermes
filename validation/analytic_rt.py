# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""RT ray-density diagnostics against ideal PEC far-field RCS references."""

import argparse
from pathlib import Path
import warnings

import numpy as np
from scipy.constants import c

from mmWaveRadar.provenance import capture_provenance
from .analytic import plate_mesh, plate_rcs, square_trihedral_rcs
from .study_runtime import write_json
from .uncertainty import seed_mean_interval


def run_rt_convergence(output, *, ray_counts=(1000, 10000, 100000, 1000000), seeds=range(5)):
    """Report misses and disagreement as observations, never fit an RCS gain.

    Geometrical-optics RT does not implement finite-aperture diffraction. Its
    equivalent RCS need not approach the PO formula as ray density increases.
    """
    from mmWaveRadar.experiments.physics_microscope import make_trihedral_corner_mesh
    from mmWaveRadar.radar import FMCWConfig, RadarHardware, RadarSensor
    from mmWaveRadar.simulation import MmWaveRadarSimulator
    from mmWaveRadar.targets import MeshSequence, MeshTarget
    from sionna.rt import load_scene, RadioMaterial
    import drjit as dr

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seeds = tuple(seeds)
    if len(seeds) < 5 or len(set(seeds)) != len(seeds):
        raise ValueError("RT convergence needs at least five unique seeds")
    frequency, distance, side = 60e9, 5.0, 0.05
    fmcw = FMCWConfig(frequency, 1e9, 40e-6, 50e-6, 1e6, 8, 1, 0.05)
    hardware = RadarHardware(
        "unit-gain", np.zeros((1, 3)), np.zeros((1, 3)), scalar_antenna_gain_dbi=0
    )
    radar = RadarSensor("verification", (0, 0, 0), (0, 0, 0), hardware, fmcw, tx_power_dbm=0)
    geometries = {
        "pec_plate": (
            *plate_mesh(side, side, distance, 0, 4),
            float(plate_rcs(side, side, frequency)),
        ),
        "square_trihedral": (
            *make_trihedral_corner_mesh(range_m=distance, edge_m=side, max_edge_m=side / 4),
            float(square_trihedral_rcs(side, frequency)),
        ),
    }
    rows = []
    for name, (vertices, faces, reference) in geometries.items():
        for rays in ray_counts:
            values, counts, provenance = [], [], None
            messages = []
            for seed in seeds:
                simulator = MmWaveRadarSimulator(
                    solver="rt",
                    rt_human_bounces="any",
                    rt_human_specular_reflection=True,
                    rt_diffuse_reflection=False,
                    rt_max_depth=3,
                    rt_samples_per_source=rays,
                    rt_max_paths_per_source=10000,
                    rt_seed=seed,
                    compute_backend="numpy",
                    compute_precision="float64",
                )
                dr.set_thread_count(1)
                target = MeshTarget(
                    name,
                    MeshSequence(vertices[None], faces, np.array([0.0])),
                    RadioMaterial(
                        "pec-proxy",
                        relative_permittivity=1,
                        conductivity=1e12,
                        scattering_coefficient=0,
                    ),
                )
                with warnings.catch_warnings(record=True) as caught:
                    cube = simulator.run(load_scene(), radar, [target])
                messages.extend(str(w.message) for w in caught)
                values.append(
                    float(
                        np.mean(abs(cube.adc) ** 2)
                        * (4 * np.pi) ** 3
                        * distance**4
                        / (c / frequency) ** 2
                    )
                )
                counts.append(int(np.max(cube.metadata.path_counts)))
                provenance = cube.metadata.provenance
            interval = seed_mean_interval(values)
            rows.append(
                {
                    "target": name,
                    "rays": rays,
                    "seed_rcs_m2": values,
                    "equivalent_rcs_m2": interval,
                    "reference_rcs_m2": reference,
                    "relative_error": abs(float(interval["mean"]) / reference - 1),
                    "path_counts": counts,
                    "missed_seed_count": sum(x == 0 for x in counts),
                    "warnings": sorted(set(messages)),
                    "native_provenance": provenance,
                }
            )
    result = {
        "provenance": capture_provenance(
            settings={
                "rays": ray_counts,
                "seeds": seeds,
                "range_m": distance,
                "side_m": side,
                "frequency_hz": frequency,
            },
            runtime={"backend": "native Mitsuba LLVM", "adc_precision": "float64"},
        ),
        "status": "diagnostic; no automatic RCS accuracy pass",
        "rows": rows,
        "interpretation": "RT uses a high-conductivity PEC proxy and geometric paths. Finite-aperture diffraction and PO exit-aperture integration are absent. Ray convergence alone cannot correct this model difference. An all-zero interval with missed paths describes sampler failure, not certainty of zero physical scattering.",
    }
    write_json(output / "rt_convergence.json", result)
    lines = [
        "# RT ray-density diagnostics",
        "",
        result["interpretation"],
        "",
        "| Target | Rays/source | Seeds | Mean equivalent RCS m² | 95% seed-mean CI | PO reference m² | Relative error | Missed seeds |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        interval = row["equivalent_rcs_m2"]
        lines.append(
            f"| {row['target']} | {row['rays']} | {len(seeds)} | {float(interval['mean']):.5g} | [{float(interval['lower']):.5g}, {float(interval['upper']):.5g}] | {row['reference_rcs_m2']:.5g} | {row['relative_error']:.4g} | {row['missed_seed_count']} |"
        )
    (output / "rt_convergence.md").write_text("\n".join(lines) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run_rt_convergence(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
