# Configuration tradeoffs on 6843AOP captures

This suite drives the existing validation runner and native PO, RT and Hybrid PO
simulators. It changes no simulation algorithms. It separates **numerical
approximation error** against shared CPU references from **physical agreement**
with your own IWR6843AOPEVM captures. RT-Pose is not an input.

The editable configuration is
[`configs/benchmark_6843aop.json`](../../configs/benchmark_6843aop.json).
Measurement data are intentionally not included. Planning works before they arrive;
running requires prepared bundles. Synthetic test fixtures verify the harness,
not measurement accuracy.

## Inputs

The example looks for these directories, relative to the repository root:

```text
data/measurements/6843aop/
  plate/
  stationary-human/
  walking-human/
  calibration-human/     # separate held-out recording/poses
```

Each directory is a [HERMES bundle](../../validation/BUNDLE_SPEC.md) with measured
complex ADC `[frame, loop, sample, virtual_channel]`, matching waveform/array/Tx
order, frame timestamps and scene/target geometry. Static targets can use a mesh
NPZ or OBJ; humans can use an evaluated mesh sequence or motion parameters plus
a separately licensed body model. Walking geometry must cover the full physical
acquisition interval, including TDM slots. Materials can be set per case.
The runner validates the declared board model rather than silently substituting
cascade geometry.

Add cases for each subject, placement and trajectory, with identifying metadata.
The three example cases are placeholders, not a sufficient population study.
Keep evaluation poses separate from calibration poses. Background subtraction
defaults off; subtracting a measured empty room while retaining the simulated
room changes the comparison and needs an explicit experimental choice.

## Commands

Run from the repository root using the environment with HERMES dependencies:

```bash
python tools/run_benchmark_study.py --plan --platform apple-m4-pro
```

`plan.json` lists resolved settings, jobs, missing inputs and timing lower bounds.
Use a restricted pilot before launching the whole matrix:

```bash
python tools/run_benchmark_study.py --references --case plate --experiment po-integration
python tools/run_benchmark_study.py --run --case plate --experiment po-integration --platform apple-m4-pro
```

Generate references for all numerical experiments with `--references`. Fit human
gain on the held-out capture before physical Hybrid PO comparisons:

```bash
python tools/run_benchmark_study.py --calibrate --case stationary-human --experiment baseline
python tools/run_benchmark_study.py --calibrate --case walking-human --experiment baseline
python tools/run_benchmark_study.py --run --platform apple-m4-pro
```

On the actual RTX 5090 machine, use the same checkout, bundles and reference
artifacts. `--platform` labels a directory; it does not select or emulate hardware.
The resolved devices and package versions are recorded independently of that label.

```bash
python tools/run_benchmark_study.py --run --platform rtx-5090 --reference-root /path/to/shared/references
python tools/run_benchmark_study.py --report --platform apple-m4-pro --peer-platform /path/to/rtx-5090
```

The regular entry point also accepts
`mmwave-validation-benchmark study --config configs/benchmark_6843aop.json ...`.
Repeat `--case` or `--experiment` to select several. `--reference-root` also
relocates automatically assigned frozen-gain files. Copy the entire shared
reference directory, including calibration records, between machines.

## Study design

The default sweep design changes one parameter at a time, includes a baseline,
and deduplicates identical effective settings within each experiment. Set
`design: "grid"` for selected interactions. Seeds are paired between numerical
candidates and references. Deterministic PO runs once; RT and Hybrid PO use five
seeds by default. Static cases skip temporal sweeps unless `has_motion` is set.

The numerical track covers RT rays and path caps, interaction depth, human
bounces, chirp/frame cadence, displacement and PO phase-error triggers, integration
mode, adaptive tolerances, visibility samples/cadence and reflector truncation.
The `visibility` experiment crosses 1, 4 and 16 samples per face with adaptive
visibility sampling on and off, for both PO and Hybrid PO. It measures timing
and numerical error against a uniform 16-sample reference. Adaptive sampling is
the PO default (grazing cosine margin 0.15); it can miss partial occlusion inside
a face when centroid and neighboring-face checks do not flag a boundary.
Compare on/off results at the same sample count, particularly for walking
sequences near occlusion boundaries. At one sample the switch has no effect.
`0` refresh frames means an exact solve each chirp; `1` means once each frame,
with the existing incremental updates between refreshes. One frame remains the
baseline. Longer cadences can produce 10–50% error with current triggers; the
suite measures that error rather than treating a trigger as an accuracy guarantee.
The `po-motion` experiment also varies area-change (2% and 30%) and
normal-change (1 and 20 degrees) limits. Native stage counts record faces
re-integrated versus phase-updated to quantify their runtime cost.
Temporal outputs include scheduled displacement per refresh in millimetres and
wavelengths. Early adaptive refreshes can reduce actual drift.

Phase-centre mode is the default for the AOP study. The optional `subarray-check`
experiment is disabled. If enabled, use spatially compact groups; singleton groups
are the per-element reference through the existing subarray API. Physical runs
reject native aperture-validity warnings by default (`require_valid_array`).
For physical TDM the existing adapter runs each physical Tx separately, so its
active Tx grouping is a singleton. `ignore_tdm_timing: false` preserves slot
timing and walking Doppler; record and justify any change.

The example physical experiments compare solver, coupling and diffuse-scattering
choices at fixed candidate numerical settings. First inspect the numerical
convergence results, then choose settings whose approximation error is below the
model difference of interest. A million rays, depth four, or a tight PO tolerance
is a reference configuration, not proof of convergence. Inspect native refinement
warnings and path-cap indicators. Post-filter path counts cannot prove that a cap
never truncated paths. Existing analytic plate/sphere tools remain separate;
agreement with an exact integral and agreement with Mie scattering answer
different questions.

## Calibration and accuracy

Numerical experiments require both native calibrations to be `none`. CPU
references force NumPy float64 PO and ADC computation and the LLVM Mitsuba
variant. RT tracing retains the native Mitsuba precision; this is recorded and
is not described as float64 ray tracing. Copy the identical reference ADC files
to both machines. Cross-platform output records reference checksums alongside
the M4-versus-5090 ADC difference; each run also records its own reference error.

`--calibrate` uses native `rt_first_pose` calibration on the **first frame** of a
separate held-out bundle. It pools human RT and raw human PO reference powers
over physical Tx slots and seeds, then saves
`sqrt(mean(RT power) / mean(PO power))`. This frozen amplitude gain is applied at
the validation boundary to human PO and both coupling contributions. Environment
RT retains its scale. Evaluation runs use native calibration `none`; they do not
fit gain to the evaluated frames. Calibration input/code hashes are checked.
Native runtime metadata identifies the calibration aggregation convention.

Wideband PO frequency grids follow `compute_precision`: float32 uses a 1e-8
Chebyshev-tail tolerance and float64 uses 1e-14. These tolerances scale with the
sum of leaf magnitudes; they are not relative-error guarantees for a cancelled
ADC signal. CPU float64 references therefore also check the reduced float32
grid. Sparse synthesis skips noncontributing columns while IF-filter counts
remain in the original path domain.

Exact wideband PO integration also follows `compute_backend` and
`compute_precision`. Torch evaluates analytic simplex integrals and reduces
quadrature/analytic leaves on the chosen device; clipped IF-edge integrals use
the same backend. Geometry preparation, clipping polygons, and persistent
response storage remain on the host. `auto` uses a supported accelerator or
NumPy; `numpy` remains the CPU reference. Device timings must include the
integration stages and transfers, rather than reporting kernel-only speedups.

This RT-to-PO gain is distinct from radar-to-simulator absolute calibration.
Shape correlation, normalized errors, peak range and Doppler errors are reported
without claiming absolute radiometric calibration. Absolute power error requires
`absolute_power_calibrated: true`, a fixed `measurement_amplitude_gain` applied
to simulated ADC, and a documented `measurement_calibration_source`. There is no
per-candidate fit to measurement.

Metrics cover range-time, range-Doppler, Doppler-time and angle power maps.
Coherent ADC error is restricted to numerical deterministic/specular comparisons.
Stochastic comparisons average **powers over seeds**, never complex returns.
Reports contain seed dispersion, timing tables, numerical Pareto plots and
measured/simulated range profiles. A seed IQR is not a population confidence interval.

## Timing, provenance and resumability

Each job uses separate accuracy and timing subprocesses. Accuracy/reference LLVM
execution uses one CPU thread for reproducibility; timing uses the normal runtime
thread configuration. Timers synchronize devices, and artifact compression and
DSP scoring stay outside the timed region. Native `simulator.run` time is the
primary metric; whole validation-pipeline time is retained separately. Nested
native stage profiles describe the last repetition and are not disjoint totals.

The first in-process execution is recorded separately and counts as the first
warmup. It preserves existing disk/JIT caches, so it is not an empty-cache cold
start. Warmed runs execute the solver at least five times and for at least ten
seconds, capped at 1,000 repetitions. Short plate workloads therefore receive
more repetitions. Median, IQR, individual samples and whether the duration target
was met are saved. Adjust these limits for pilots. Accuracy and timing may use
different CPU thread counts; both are explicit in worker records.

References are reusable only with matching code, inputs and settings, and their
ADC checksums are verified. Timed work never reads cached ADC. `--resume` retains
completed job records, including their original timing observations; omit it for
fresh timing repetitions. Timing-policy or environment changes create separate
execution records. Reports keep different code/input/environment/protocol cohorts
separate. Provenance includes the Git commit, dirty Python-source hashes, input
hashes, actual devices, precision, stage counts, memory and warnings.

```text
benchmarks/configuration_tradeoffs/results/6843aop-tradeoffs/
  references/
    cpu-float64/<case>/<configuration>/
    calibration/<case>.json
  apple-m4-pro/
    plan.json
    runs/<case>/<experiment>/<configuration>-<environment>/
      accuracy/{adc.npz,worker_result.json,worker.log}
      timing/{worker_result.json,worker.log}
      result.json
      measurement_maps.npz
      numerical_maps.npz
    summary.json
    summary.csv
    figures/
    cross_platform.json
  rtx-5090/
    ...
```

No full-suite runtime estimate or measurement result is reported before a pilot
and the actual captures exist. Verification fixtures are labelled synthetic.
