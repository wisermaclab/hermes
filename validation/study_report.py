# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Static scientific figures for the two separate benchmark tracks."""

from collections import defaultdict
import json

import numpy as np


def pareto_indices(times, errors):
    """Indices whose accuracy/runtime is not dominated by another point."""
    keep = []
    best = np.inf
    for i in sorted(range(len(times)), key=lambda i: (times[i], errors[i])):
        if errors[i] < best:
            keep.append(i)
            best = errors[i]
    return keep


def plot_summary(directory, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["case"],
                row["experiment"],
                row["solver"],
                row["track"],
                row["comparison_context"],
            )
        ].append(row)
    output = directory / "figures"
    output.mkdir(exist_ok=True)
    for (case, experiment, solver, track, context), points in grouped.items():
        stem = f"{case}-{experiment}-{solver}-{context[:8]}"
        points.sort(key=lambda p: (bool(p["changes"]), json.dumps(p["changes"], sort_keys=True)))
        fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")
        labels = [
            json.dumps(p["changes"], sort_keys=True) if p["changes"] else "Baseline"
            for p in points
        ]
        if track == "numerical":
            times = [p["median_seconds"] for p in points]
            errors = [p["numerical_error"] for p in points]
            valid = [i for i, e in enumerate(errors) if e is not None]
            times, errors = [times[i] for i in valid], [errors[i] for i in valid]
            ax.scatter(times, errors, s=45)
            for i, t, e in zip(valid, times, errors):
                right = t > np.median(times)
                ax.annotate(
                    labels[i],
                    (t, e),
                    xytext=(-4 if right else 4, 5),
                    ha="right" if right else "left",
                    textcoords="offset points",
                    fontsize=7,
                )
            frontier = pareto_indices(times, errors)
            ax.plot([times[i] for i in frontier], [errors[i] for i in frontier], "--", alpha=0.6)
            ax.set_xlabel("Warmed native simulator time [s], median across seeds")
            ax.set_ylabel("Seed-mean range–time power error / CPU reference")
            ax.set_title(f"{case} · {solver} · numerical tradeoff")
        else:
            y = [
                p["measurement_seed_mean_power"]["range_time"]["normalized_correlation"]
                for p in points
            ]
            ax.bar(np.arange(len(points)), y, color="#267d9b")
            ax.set_xticks(np.arange(len(points)), labels, rotation=25, ha="right", fontsize=8)
            ax.set_ylabel("Measured / simulated range–time power correlation")
            ax.set_title(f"{case} · {solver} · measurement agreement")
        ax.grid(alpha=0.2)
        fig.savefig(output / f"{stem}.png", dpi=160)
        plt.close(fig)

        motion_keys = {
            "rt_refresh_frames",
            "rt_refresh_on_displacement",
            "po_refresh_frames",
            "po_refresh_on_phase_error_rad",
        }
        if track == "numerical" and any(set(p["changes"]) & motion_keys for p in points):
            axis = "rt" if solver == "rt" else "po"
            fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")
            midpoint = np.median(
                [p["motion_axis"][axis]["scheduled_max_vertex_drift_mm"] for p in points]
            )
            for label, point in zip(labels, points):
                error = point["numerical_error"]
                if error is None:
                    continue
                drift = point["motion_axis"][axis]["scheduled_max_vertex_drift_mm"]
                ax.scatter(drift, error, s=45)
                right = drift > midpoint
                ax.annotate(
                    label,
                    (drift, error),
                    xytext=(-4 if right else 4, 5),
                    ha="right" if right else "left",
                    textcoords="offset points",
                    fontsize=7,
                )
            ax.set(
                xlabel="Maximum vertex displacement per scheduled refresh [mm]",
                ylabel="Seed-mean range–time power error / CPU reference",
                title=f"{case} · {solver} · temporal error",
            )
            ax.grid(alpha=0.2)
            fig.savefig(output / f"{stem}-displacement.png", dpi=160)
            plt.close(fig)

        # Average powers over seeds; do not average complex stochastic returns.
        rmean, emean, ranges = None, None, None
        for i, record in enumerate(points[0]["records"], 1):
            with np.load(
                (directory / record).parent / "measurement_maps.npz", allow_pickle=False
            ) as data:
                r, e = data["reference_range_time"].mean(0), data["candidate_range_time"].mean(0)
                ranges = data["ranges_m"]
                if rmean is None:
                    rmean, emean = r.copy(), e.copy()
                else:
                    rmean += (r - rmean) / i
                    emean += (e - emean) / i
        if rmean is not None:
            fig, ax = plt.subplots(figsize=(7.5, 4.3), layout="constrained")
            scale = max(float(np.max(rmean)), 1e-30)
            ax.plot(ranges, 10 * np.log10(np.maximum(rmean / scale, 1e-12)), label="Measurement")
            ax.plot(
                ranges,
                10 * np.log10(np.maximum(emean / scale, 1e-12)),
                label="Baseline simulation",
            )
            ax.set(
                xlabel="Range [m]",
                ylabel="Power [dB relative to measured peak]",
                title=f"{case} · {solver} · seed-mean range profile",
            )
            ax.legend()
            ax.grid(alpha=0.2)
            fig.savefig(output / f"{stem}-range-profile.png", dpi=160)
            plt.close(fig)
