# Demo Notebooks

These notebooks are interactive workflows, not benchmark definitions. Keep them
output-free in the repository; generated cubes, figures, and cache files should
stay outside the repository, preferably under the OS-specific temporary
directory returned by Python's `tempfile.gettempdir()`.

For the packaged local web application, install the `gui` extra and run
`hermes-gui`. Unlike the notebooks, the GUI also exposes static plate,
trihedral-corner, and uploaded-human RT/PO comparisons; Dynamic Scenes using a
prepared room or uploaded XML with AMASS-like motion and a separately licensed
SMPL model; the versioned experiment manifest; global TI hardware, Tx/Rx
subset, FMCW, and antenna-pattern settings; separate fixed-origin radar
orientation for static and dynamic runs; on-demand RT-path/PO-facet
diagnostics for both static targets and selected dynamic frames; and
unified measurement/simulation bundle workflows.

## Suggested Order

1. `Point-Targets.ipynb` - analytic point-target ADC synthesis, DSP basics, and
   TI board virtual-array angle-map checks.
2. `Human-Only-PO.ipynb` - moving-human physical optics.
3. `Full-RT.ipynb` - full-scene RT retracing and coherent-bank comparison.
4. `Hybrid-PO.ipynb` - static-environment RT plus moving-human PO assembly.

Install the notebook dependencies using the
[top-level installation instructions](../README.md#installation), then launch
Jupyter from the repository root so relative data and model paths resolve
consistently:

```bash
jupyter lab demo/Point-Targets.ipynb
```

The point-target notebook needs no external data. The three human-motion
notebooks also require the `smpl` optional dependencies and a separately
obtained SMPL-family model.

## Notebook Roles

| Notebook | Role | External Data |
| --- | --- | --- |
| `Point-Targets.ipynb` | Small analytic FMCW/DSP sanity check plus TI board virtual-array angle maps without scene geometry. | No |
| `Human-Only-PO.ipynb` | Runs human-only PO and incremental PO comparisons. | Included CMU motion or external AMASS motion; external SMPL model |
| `Full-RT.ipynb` | Compares per-chirp RT retracing with coherent path-bank updates, then illustrates the effect of maximum interaction depth. | Included CMU motion or external AMASS motion; external SMPL model |
| `Hybrid-PO.ipynb` | Recombines static RT, human PO, blocking, and coupling components. | Included CMU motion or external AMASS motion; external SMPL model |

## Maintenance Notes

- Keep notebook outputs and execution counts stripped before committing.
- Avoid absolute local paths in committed notebook source.
- Keep generated artifacts outside the repository. The notebooks retain
  results in memory and, by default, direct Matplotlib's cache to
  `tempfile.gettempdir()`.
- From a source checkout, set `MMWAVE_AMASS_NPZ` to
  `data/AMASS/walking_poses_cmu_105_02.npz` to use the included walking motion.
  Set it to an external archive, or use `MMWAVE_DATASET_ROOT`, for any other
  motion. Obtain the licensed SMPL-family model separately and put it under the
  ignored `models/smpl_models/` directory; use `MMWAVE_SMPL_MODEL_DIR` only to
  override that default location.
- TI board presets use the data-free `cosine30` pattern by default. Set
  `MMWAVE_TI_PATTERN_NPZ` only when you have an authorized compatible
  digitized-pattern archive.
- Use `tools/plot_ti_board_patterns.py` for TI board catalog and pattern
  inspection.
- Use `python -m mmWaveRadar.diagnostics.radial_velocity_geometry` for
  geometry-only radial-velocity diagnostics.
- Put reusable batch measurements in `benchmarks/`, not in notebooks.
- Put real-radar dataset comparisons in `validation/`, not in notebooks.
