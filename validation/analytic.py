# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Independent far-field PEC references and native PO mesh-convergence studies."""

import argparse
from pathlib import Path

import numpy as np
from scipy.constants import c
from scipy.spatial import ConvexHull

from mmWaveRadar.provenance import capture_provenance
from .study_runtime import write_json


def plate_rcs(width_m, height_m, frequency_hz, aspect_deg=0.0):
    """Principal-plane, co-polar far-field PO RCS (no edge diffraction)."""
    wavelength = c / frequency_hz
    theta = np.deg2rad(aspect_deg)
    return (
        4
        * np.pi
        * (width_m * height_m / wavelength) ** 2
        * np.cos(theta) ** 2
        * np.sinc(2 * width_m / wavelength * np.sin(theta)) ** 2
    )


def sphere_optical_rcs(radius_m):
    """High-frequency limit, ka >> 1; not an exact Mie resonance reference."""
    return np.pi * radius_m**2


def square_trihedral_rcs(side_m, frequency_hz):
    """Boresight high-frequency triple-reflection RCS for three square faces."""
    return 12 * np.pi * side_m**4 / (c / frequency_hz) ** 2


def plate_mesh(width, height, distance, aspect, divisions):
    y, z = np.meshgrid(
        np.linspace(-width / 2, width / 2, divisions + 1),
        np.linspace(-height / 2, height / 2, divisions + 1),
        indexing="ij",
    )
    theta = np.deg2rad(aspect)
    vertices = np.column_stack(
        (distance - y.ravel() * np.sin(theta), y.ravel() * np.cos(theta), z.ravel())
    )
    faces = []
    for i in range(divisions):
        for j in range(divisions):
            a = i * (divisions + 1) + j
            faces.extend(
                ((a, a + 1, a + divisions + 2), (a, a + divisions + 2, a + divisions + 1))
            )
    return vertices, np.array(faces, dtype=np.uint32)


def sphere_mesh(radius, distance, points):
    # Convex hull of deterministic Fibonacci nodes: independent of the solver.
    i = np.arange(points)
    z = 1 - 2 * (i + 0.5) / points
    phi = i * np.pi * (3 - np.sqrt(5))
    nodes = (
        np.column_stack((np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), z))
        * radius
    )
    faces = ConvexHull(nodes).simplices.copy()
    triangles = nodes[faces]
    inward = (
        np.einsum(
            "ij,ij->i",
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            triangles.mean(1),
        )
        < 0
    )
    faces[inward] = faces[inward][:, [0, 2, 1]]
    return nodes + [distance, 0, 0], faces


def native_po_rcs(vertices, faces, frequency_hz, distance, integration_mode):
    from mmWaveRadar.simulation import PhysicalOpticsMaterial, po_channel

    channel = po_channel(
        vertices,
        faces,
        np.zeros((1, 3)),
        np.zeros((1, 3)),
        frequency_hz,
        PhysicalOpticsMaterial(1.0, 0.0, model="pec"),
        visibility="none",
        po_integration_mode=integration_mode,
        compute_backend="numpy",
        compute_precision="float64",
    )
    h = np.sum(channel.phase_center_contributions)
    # Invert the unit-gain monostatic radar equation, with no fitted scale.
    return float(abs(h) ** 2 * (4 * np.pi) ** 3 * distance**4 / (c / frequency_hz) ** 2)


def run_verification(output, *, divisions=(1, 2, 4, 8, 16), sphere_points=(512, 2048, 8192)):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frequency, distance, width, height = 60e9, 1000.0, 0.152, 0.152
    rows = []
    for angle in (0.0, 0.5, 1.0):
        expected = float(plate_rcs(width, height, frequency, angle))
        for mode in ("face_centroid", "parent_face_far_field_analytic"):
            for n in divisions:
                vertices, faces = plate_mesh(width, height, distance, angle, n)
                actual = native_po_rcs(vertices, faces, frequency, distance, mode)
                rows.append(
                    {
                        "target": "pec_plate",
                        "aspect_deg": angle,
                        "solver": "po",
                        "mode": mode,
                        "faces": len(faces),
                        "rcs_m2": actual,
                        "reference_rcs_m2": expected,
                        "relative_error": abs(actual / expected - 1),
                        "reference": "far-field physical optics closed form",
                    }
                )
    radius = 0.015
    for count in sphere_points:
        vertices, faces = sphere_mesh(radius, distance, count)
        actual = native_po_rcs(vertices, faces, frequency, distance, "face_centroid")
        expected = sphere_optical_rcs(radius)
        rows.append(
            {
                "target": "pec_sphere",
                "solver": "po",
                "mode": "face_centroid",
                "faces": len(faces),
                "rcs_m2": actual,
                "reference_rcs_m2": expected,
                "relative_error": abs(actual / expected - 1),
                "reference": "optical limit only; includes finite-ka model discrepancy",
                "ka": 2 * np.pi * radius * frequency / c,
            }
        )
    plate_final = [
        r
        for r in rows
        if r["target"] == "pec_plate"
        and r["mode"] == "parent_face_far_field_analytic"
        and r["faces"] == 2 * max(divisions) ** 2
    ]
    result = {
        "provenance": capture_provenance(
            settings={
                "frequency_hz": frequency,
                "range_m": distance,
                "plate_width_m": width,
                "plate_height_m": height,
                "sphere_radius_m": radius,
                "divisions": divisions,
                "sphere_points": sphere_points,
            },
            seed=None,
            runtime={"backend": "numpy", "precision": "float64"},
        ),
        "passed": all(r["relative_error"] < 0.01 for r in plate_final),
        "rows": rows,
        "corner_reflector": {
            "reference_rcs_m2": square_trihedral_rcs(0.152, frequency),
            "status": "not verified",
            "reason": "Native PO is single-bounce; triple reflections need a separate RT/SBR verification. No analytic corner pass is claimed.",
        },
        "sources": [
            "https://www.ll.mit.edu/media/6956",
            "https://ntrs.nasa.gov/citations/19940020143",
        ],
    }
    write_json(output / "analytic.json", result)
    lines = [
        "# PEC analytic verification",
        "",
        "Far-field, unit gains, ideal PEC; no gain fitting. Plate RCS is a closed-form PO reference. Sphere values compare to the high-frequency optical limit.",
        "",
        "| Target | Aspect | Integration | Faces | RCS m² | Reference m² | Relative error |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['target']} | {r.get('aspect_deg', '—')} | {r['mode']} | {r['faces']} | {r['rcs_m2']:.6g} | {r['reference_rcs_m2']:.6g} | {r['relative_error']:.4g} |"
        )
    lines += ["", result["corner_reflector"]["reason"]]
    (output / "analytic.md").write_text("\n".join(lines) + "\n")
    plot_convergence(output, rows)
    return result


def plot_convergence(output, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    groups = {}
    for row in rows:
        groups.setdefault((row["target"], row.get("aspect_deg"), row["mode"]), []).append(row)
    for key, points in groups.items():
        ax.loglog(
            [r["faces"] for r in points],
            [max(r["relative_error"], 1e-14) for r in points],
            ".-",
            label=" / ".join(str(k) for k in key if k is not None),
        )
    ax.set(
        xlabel="Mesh faces",
        ylabel="Relative RCS error",
        title="PEC mesh convergence (no measurement inputs)",
    )
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.savefig(output / "convergence.png", dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return 0 if run_verification(args.output)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
