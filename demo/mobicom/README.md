# MobiCom demo: PEC plate verification

Open [PEC-Plate.ipynb](PEC-Plate.ipynb) for the interactive version, or use the CLI below.

This data-free example checks PO integration separately from the measured copper
plate. It uses an ideal 152 × 152 mm PEC plate, 60 GHz, a monostatic unit-gain
sensor and a 1 km far-field separation. The distance avoids mistaking finite-range
Fresnel effects near an angular null for integration error.

From the repository root, in an environment where HERMES is installed
(`python -m pip install -e .`):

```bash
python -m validation.analytic --output outputs/mobicom/pec-plate
```

Open analytic.md and convergence.png in the output directory. The report compares
centroid and analytic-facet integration at increasing mesh densities and at 0°,
0.5° and 1° aspect, without fitting an amplitude. The principal-plane far-field
reference is

σ(θ) = 4π(wh/λ)² cos²θ sinc²(2w sinθ/λ), with sinc(x) = sin(πx)/(πx).

The same command includes a sphere optical-limit mesh study. For plate and
trihedral ray-density diagnostics:

```bash
python -m validation.analytic_rt --output outputs/mobicom/rt-convergence
```

RT misses and differences from finite-aperture PEC formulas are published as
observations. They are not hidden by gain alignment. See the
[verification guide](../../docs/scientific_validation.md) for release regression
and the separate, measured copper-plate comparison.
