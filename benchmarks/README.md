# Benchmarks

This directory contains synthetic performance and behavior benchmarks for the
simulator. These are not real-radar validation suites; use `validation/` when
the goal is to compare simulated ADC against captured radar clips.

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

`human_po/run.py` benchmarks human-only physical optics on a precomputed
`MeshSequence` NPZ containing `vertices`, `faces`, and `times`.

Use it for:

- A quick PO runtime smoke test with `--mode single`
- Incremental-PO speed/error tradeoff measurements with `--mode sweep`
- Tuning visibility refresh, geometry-change thresholds, backend, and precision

Create a mesh sequence from an AMASS motion file:

```bash
python tools/amass_to_mesh_sequence.py \
  --amass-npz data/AMASS/walking_poses_cmu_105_02.npz \
  --output-npz /path/to/walking_mesh_sequence.npz \
  --num-frames 120 \
  --force
```

Run a single PO benchmark:

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

The sweep is a regression and tuning benchmark for the incremental
human-only PO path. It first runs a full PO baseline with incremental updates
disabled. That baseline is treated as the reference ADC cube for both runtime
and numerical-error comparisons.

Each sweep setting then reruns the same mesh sequence with incremental updates
enabled. Instead of recomputing PO visibility and facet contributions for every
face on every chirp, the incremental path updates only faces whose geometry has
changed enough since the last refresh. The sweep changes these controls:

- `K`: visibility refresh period in chirps.
- `normal_threshold_deg`: face-normal change threshold.
- `centroid_displacement_threshold_m`: face-centroid displacement threshold.
- `area_relative_threshold`: relative face-area change threshold.

The built-in settings are:

| Label | K | Normal | Centroid | Area |
| --- | ---: | ---: | ---: | ---: |
| `k8 nominal n5 d2mm a5pct` | 8 | 5 deg | 0.002 m | 0.05 |
| `k16 nominal n5 d2mm a5pct` | 16 | 5 deg | 0.002 m | 0.05 |
| `k32 nominal n5 d2mm a5pct` | 32 | 5 deg | 0.002 m | 0.05 |
| `k64 nominal n5 d2mm a5pct` | 64 | 5 deg | 0.002 m | 0.05 |
| `k32 loose n15 d10mm a20pct` | 32 | 15 deg | 0.010 m | 0.20 |
| `k64 loose n20 d20mm a30pct` | 64 | 20 deg | 0.020 m | 0.30 |
| `k128 aggressive n30 d50mm a50pct` | 128 | 30 deg | 0.050 m | 0.50 |

You can replace the built-in sweep with repeated `--setting` values:

```bash
python benchmarks/human_po/run.py \
  --mode sweep \
  --mesh-npz /path/to/walking_mesh_sequence.npz \
  --setting "k32 custom:32:10:0.005:0.10" \
  --setting "k96 custom:96:20:0.020:0.25"
```

The setting format is:

```text
label:K:normal_deg:centroid_m:area_rel
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

`human_rt/run.py` benchmarks RT sensing on an AMASS sequence. It converts
AMASS/SMPL motion internally, places a single virtual radar channel in front of
the subject, and can run either `rt_retrace` or `rt_coherent_bank`.

Use it for:

- RT runtime smoke tests
- Ray/path-budget studies with `--samples-per-src` and
  `--max-num-paths-per-src`
- Comparing full RT retracing against coherent path-bank updates
- Checking effects of `--max-depth`, retrace cadence,
  `--diffuse-reflection`, and `--coupling-mode`

Run a short full-RT retrace benchmark:

```bash
python benchmarks/human_rt/run.py \
  --num-frames 1 \
  --samples-per-src 20000 \
  --max-num-paths-per-src 20000
```

Compare full retrace with coherent bank under the same default ray budget:

```bash
python benchmarks/human_rt/run.py \
  --num-frames 2 \
  --mode both \
  --samples-per-src 20000 \
  --max-num-paths-per-src 20000
```

Run coherent bank with a different retrace cadence:

```bash
python benchmarks/human_rt/run.py \
  --num-frames 2 \
  --mode rt_coherent_bank \
  --max-depth 2 \
  --samples-per-src 20000 \
  --max-num-paths-per-src 20000 \
  --coherent-bank-retrace-period-chirps 1
```

`--mode` is `rt_retrace`, `rt_coherent_bank`, or `both`. `--max-depth`,
`--samples-per-src`, and `--max-num-paths-per-src` are ordinary scalar
parameters for the RT trace. `--mode both` runs the two RT mobility modes with
the same trace parameters so their runtime and path-count summaries can be
compared directly.

The RT runner uses the bundled CMU walking fixture unless `--motion-npz`,
`MMWAVE_AMASS_NPZ`, or `MMWAVE_DATASET_ROOT` selects another compatible file.

RT-specific controls:

- `--coherent-bank-retrace-period-chirps`: retrace cadence for
  `rt_coherent_bank`. Omit it, or pass `frame`, `default`, or `none`, for one
  retrace per frame. An integer period such as `1`, `2`, or `8` retraces at
  that chirp cadence; it must divide `--num-chirps-per-frame`.
- `--coherent-transition` / `--no-coherent-transition`: path matching used by
  `rt_coherent_bank` transition synthesis and diagnostics.
- `--coherent-crossfade` / `--no-coherent-crossfade`: force crossfade behavior
  for coherent-bank transitions; omit it to use the simulator policy.
- `--coherent-match-delay-tolerance-fraction` and
  `--coherent-match-separation-fraction`: matching tolerances for persistent
  coherent RT paths.

The RT runner prints timing and path-count diagnostics and writes `summary.json`
under its output directory. The JSON has a `runs` list with one entry per RT
mode. Pass `--output-dir` to choose that directory, or `--json` to write the
summary to a specific file.

## Choosing A Runner

- Use `human_po/run.py --mode single` for a fast PO sanity check.
- Use `human_po/run.py --mode sweep` before changing incremental PO behavior.
- Use `human_rt/run.py` when the experiment is about RT path tracing, ray
  budget, diffuse reflection, or coupling behavior.
- Use `validation/` instead when comparing simulation against real radar clips.
