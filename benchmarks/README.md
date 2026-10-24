# Benchmarks

This directory contains performance and behavior benchmarks for the simulator.
The [6843AOP configuration study](configuration_tradeoffs/README.md) combines
numerical convergence sweeps with your own measured captures through the existing
`validation/` runner, and separates results by hardware platform. The other
benchmarks below use synthetic inputs.

The source checkout includes the CMU walking motion at
`data/AMASS/walking_poses_cmu_105_02.npz`. Other motion files, every SMPL model,
and generated mesh-sequence NPZ files are local artifacts. Benchmark output
directories default under `tempfile.gettempdir()`. Pass `--output-dir` when you
want a specific location.

Obtain the licensed SMPL model separately and place it under the ignored
`models/smpl_models/` directory. The conversion and human benchmark tools use
that location by default. Set `MMWAVE_SMPL_MODEL_DIR` or pass
`--smpl-model-dir` to use another private location.

## Human PO

`human_po/run.py` benchmarks the PO solver on a human mesh from a precomputed
`MeshSequence` NPZ containing `vertices`, `faces`, and `times`.

Use it for:

- A quick PO runtime smoke test with `--mode single`
- Incremental-PO speed/error tradeoff measurements with `--mode sweep`
- Tuning `po_visibility_refresh_frames`, `po_refresh_on_phase_error_rad`,
  backend, and precision

Create a mesh sequence from an AMASS motion file:

```bash
python tools/amass_to_mesh_sequence.py \
  --amass-npz data/AMASS/walking_poses_cmu_105_02.npz \
  --output-npz /path/to/walking_mesh_sequence.npz \
  --num-frames 120 \
  --force
```

Run a single PO benchmark (exact PO every chirp by default; add
`--po-refresh-frames 1` for one exact solve per frame with phase updates in
between):

```bash
python benchmarks/human_po/run.py \
  --mode single \
  --mesh-npz /path/to/walking_mesh_sequence.npz \
  --num-frames 10
```

Run the incremental PO sweep:

```bash
python benchmarks/human_po/run.py \
  --mode sweep \
  --mesh-npz /path/to/walking_mesh_sequence.npz \
  --duration-s 5 \
  --plot
```

Sweep outputs include `summary.json`, `summary.csv`, cached baseline and
incremental ADC cubes, and diagnostic plots when `--plot` is enabled.

### What The Incremental PO Sweep Measures

The sweep is a regression and tuning benchmark for incremental PO. It first
runs an exact PO baseline (`po_refresh_frames=0`, exact at every chirp). That
baseline is treated as the reference ADC cube for both runtime and
numerical-error comparisons.

Each sweep setting reruns the same mesh sequence with exact solves at frame
starts by default. Between them phase, delay, and spreading follow the motion,
and facets whose neglected vertex phase change exceeds the trigger are
integrated again. The sweep
changes these controls:

- `V`: `po_visibility_refresh_frames`, the visibility cadence in radar frames
  (`0` = every chirp).
- `po_refresh_on_phase_error_rad`: wavenumber times the largest change in a
  vertex's two-way path length relative to its facet centre, measured since
  the last exact solve. The default is 0.1 rad.

The built-in settings are:

| Label | V | Phase-error trigger [rad] |
| --- | ---: | ---: |
| `v0 phase 0.1 rad` | 0 | 0.1 |
| `v1 phase 0.1 rad` | 1 | 0.1 |
| `v2 phase 0.1 rad` | 2 | 0.1 |
| `v0 phase 0.3 rad` | 0 | 0.3 |
| `v1 phase 0.5 rad` | 1 | 0.5 |
| `v2 phase 1.0 rad` | 2 | 1.0 |

Keep `po_refresh_frames=1` as the default. On the tutorial walk, a 4-frame
cadence stays within 0.4% of exact per-chirp PO with the 0.1 rad trigger and
reaches 5% at 0.5 rad. The trigger does not cover visibility changes, so
compare against exact per-chirp integration for each scene.

You can replace the built-in sweep with repeated `--setting` values:

```bash
python benchmarks/human_po/run.py \
  --mode sweep \
  --mesh-npz /path/to/walking_mesh_sequence.npz \
  --setting "v1 custom:1:0.1" \
  --setting "v4 custom:4:0.3"
```

The setting format is:

```text
label:visibility_frames:phase_error_rad
```

The summary table and CSV report:

- `speedup`: full-PO baseline wall time divided by incremental wall time.
- `adc_relative_error`: relative L2 error of the complex ADC cube.
- `range_profile_relative_error`: relative L2 error after range processing.
- `path_count_delta_*`: min/mean/max path-count difference from baseline.
- `recomputed_mean` and `recomputed_max`: how many faces were fully recomputed.
- `phase_updated_mean`: how many faces reused visibility/amplitude but updated
  phase.
- `full_refresh_count` and `visibility_refresh_count`: how often the
  incremental path fell back to broader refresh work.

Use the sweep to choose a setting that gives useful speedup while keeping ADC
and range-profile error within the tolerance needed for the experiment. The
plot outputs are intended for that tradeoff review: speedup vs error,
range-time, range-Doppler, and Doppler-time comparisons against the baseline.

To override the default output directory:

```bash
python benchmarks/human_po/run.py \
  --mode sweep \
  --mesh-npz /path/to/walking_mesh_sequence.npz \
  --output-dir /path/to/po_benchmark_output
```

## Human RT

`human_rt/run.py` benchmarks the RT solver on an AMASS sequence. It converts
AMASS/SMPL motion internally, places a single virtual radar channel in front of
the subject, and runs one RT pass per requested `--rt-refresh-frames` cadence.

Use it for:

- RT runtime smoke tests
- Ray/path-budget studies with `--rt-samples-per-source` and
  `--rt-max-paths-per-source`
- Comparing retracing every chirp against retracing every N frames with
  coherent path updates in between
- Checking effects of `--rt-max-depth`, retrace cadence,
  `--rt-diffuse-reflection`, and `--rt-human-bounces`

Run a short benchmark that retraces every chirp (the default):

```bash
python benchmarks/human_rt/run.py \
  --num-frames 1 \
  --rt-samples-per-source 20000 \
  --rt-max-paths-per-source 20000
```

Compare retracing every chirp with retracing once per frame under the same ray
budget:

```bash
python benchmarks/human_rt/run.py \
  --num-frames 2 \
  --rt-refresh-frames 0 1 \
  --rt-samples-per-source 20000 \
  --rt-max-paths-per-source 20000
```

`--rt-refresh-frames` takes one or more cadences in radar frames; `0` retraces
every chirp. `--rt-max-depth`, `--rt-samples-per-source`, and
`--rt-max-paths-per-source` are ordinary scalar parameters for the RT trace, so
all requested cadences share them and their runtime and path-count summaries
can be compared directly.

The RT runner uses the bundled CMU walking fixture unless `--motion-npz`,
`MMWAVE_AMASS_NPZ`, or `MMWAVE_DATASET_ROOT` selects another compatible file.

RT-specific controls:

- `--coherent-transition` / `--no-coherent-transition`: path matching used by
  retrace transition synthesis and diagnostics.
- `--coherent-crossfade` / `--no-coherent-crossfade`: force crossfade behavior
  for coherent-bank transitions; omit it to use the simulator policy.
- `--coherent-match-delay-tolerance-fraction` and
  `--coherent-match-separation-fraction`: matching tolerances for persistent
  coherent RT paths.

The RT runner prints timing and path-count diagnostics and writes `summary.json`
under its output directory. The JSON has a `runs` list with one entry per
`--rt-refresh-frames` cadence. Pass `--output-dir` to choose that directory, or `--json` to write the
summary to a specific file.

## Choosing A Runner

- Use `human_po/run.py --mode single` for a fast PO sanity check.
- Use `human_po/run.py --mode sweep` before changing incremental PO behavior.
- Use `human_rt/run.py` when the experiment is about RT path tracing, ray
  budget, diffuse reflection, retrace cadence, or human-bounce behavior.
- Use `validation/` instead when comparing simulation against real radar clips.
