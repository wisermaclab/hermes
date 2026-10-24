# Validation

`validation/` contains the real-radar validation harness. It loads a standard
clip bundle, derives DSP maps from measured ADC, optionally runs simulator
solvers for the same clip, and writes comparable metrics, maps, and plots.

Use this directory when the question is:

- Does simulated ADC match a captured radar clip?
- Do range-time or range-Doppler maps align with the measured data?
- Does the PO, RT, or Hybrid PO solver perform better for a specific clip?

Use `benchmarks/` for synthetic runtime and behavior benchmarks. For configurable
numerical/measurement sweeps and M4 Pro versus RTX 5090 timing, see the
[6843AOP configuration study](../benchmarks/configuration_tradeoffs/README.md).
It uses this runner and the existing simulation entry points. Run
`mmwave-validation-benchmark study --config configs/benchmark_6843aop.json --plan`
from the repository root, or use `tools/run_benchmark_study.py` without installing.

Bundle construction, loading, and structural integrity are owned separately by
`tools/bundle_prepare/`. The validation CLI runs the same complete integrity
check as `mmwave-check-bundle` before scientific validation; the standalone
command remains useful for checking a bundle without running metrics.

The runner delegates channel selection, simulation/TDM, caching and DSP products
to focused modules. See [code organization](../docs/code_organization.md) for
their import paths and compatibility aliases.

## Universal Bundle Contract

The bundle layout is the boundary between dataset preparation and validation.
The validation package consumes the required JSON/NPZ files without importing
the source dataset, its reader, or its conversion pipeline. Consequently, it
can validate any dataset that is expressed through this contract; dataset
names are provenance metadata, not dispatch keys. New datasets do not require
changes to the validation runner.

The normative, versioned public contract is
[HERMES Bundle contract](BUNDLE_SPEC.md). It defines one `hermes` profile for measured
and simulated ADC artifacts, parameterized or evaluated motion, target
geometry, scene metadata, manifests, and diagnostics. Machine-readable JSON
Schemas for the bundle sidecars live under [`validation/schema/`](schema/).

Raw datasets and unreviewed conversion pipelines can remain external. Check
every produced bundle at the boundary:

```bash
mmwave-check-bundle /path/to/prepared_bundle
```

### Reference adapters and fixtures

HERMES provides reviewed adapters for RT-Pose and mmRadPose because they are
public mmWave pose datasets used in this project's validation work. They also
exercise meaningfully different contracts: RT-Pose uses a cascade radar with a
large TX-major virtual array and synchronized cameras, while mmRadPose uses a
12-channel AOP layout without camera imagery. These adapters demonstrate how to
translate source-specific timing, coordinates, and channel order into the
common bundle. They are not an exhaustive support list.

Both adapters consume a separately generated, pickle-free AMASS-like base-SMPL
archive. [SparseSMPLFit](https://github.com/wisermaclab/SparseSMPLFit) can
optionally produce a compatible archive but is not bundled with or invoked by
HERMES. Dataset-specific mapping remains inside the adapter; the validation
package sees only the completed bundle. The RT-Pose adapter also reads the
selected measured cascade frame automatically, so ADC
input paths or decoder commands are not part of its public CLI. The same call
also reuses supplied geometry or reconstructs and fits it from synchronized
LiDAR/stereo inputs; each completed bundle contains its own scene assets.
The mmRadPose adapter similarly discovers its parsed radar cube and alignment
and can reuse geometry or fit a laser cloud registered to the radar frame. Its
environment path has no stereo/camera input; absent laser or fitted geometry,
it writes an explicit empty environment.

The source checkout also contains two deliberately limited, runnable bundles
under `data/validation_bundles/`. Each has one measured ADC frame and a
three-sample neutralized `amass_sequence.npz` window covering the acquisition
interval. The archives use neutral gender and zero betas and do not include
model files, so mesh-generating validation needs a separately licensed neutral
SMPL model. The RT-Pose example includes synchronized
stereo setup images with opaque face masks; the mmRadPose source has no
camera image. See each fixture's `NOTICE.md` before redistribution.

## Bundle resources

Follow the normative [bundle specification](BUNDLE_SPEC.md) for the descriptor,
sensor, ADC axes, scene assets, timing, coordinate/phase conventions, and
optional static or moving targets. Motion is optional: static targets and
evaluated motion do not require SMPL. Parameterized human motion requires a
separately licensed model when meshes are evaluated.

Use `mmwave-prepare-bundle --config prepare.json` to package normalized inputs.
The [measurement tutorial](../tools/bundle_prepare/BRING_YOUR_OWN.md) includes
an installed, redistributable synthetic static-target example and the measured
conformance milestone. Check directories or ZIPs with
`mmwave-check-bundle PATH --all-errors --json`; add `--diagnose OUTPUT_DIR`
for warning-only capture plots. The validation CLI's `--clip` argument accepts
either a bundle directory or a ZIP file.

## Run Validation

Run simulation and compare it with the measurements directly from a ZIP:

```bash
mmwave-validation-benchmark \
  --clip /path/to/capture-bundle.zip \
  --solvers po hybrid_po \
  --suite dsp_maps \
  --out /path/to/validation-results \
  --progress
```

Without `--simulated-adc`, the runner generates ADC using the bundled sensor,
scene, target geometry or motion, and target material, then compares it with
the primary measured ADC. Outputs include `metrics.json`, `maps.npz`, simulated
ADC caches and comparison plots. ZIP resources are extracted to a temporary
directory for the run and removed afterwards; the source ZIP is unchanged.
In `metrics.json`, bundled scene paths use `capture-bundle.zip!/path/in/bundle`
to identify the scene without referring to the temporary directory.

Write outputs to a temporary directory:

```bash
OUT_DIR="$(python -c 'import tempfile; print(tempfile.gettempdir())')/validation_run"

python -m validation.cli \
  --clip /path/to/clip_bundle \
  --suite dsp_maps \
  --out "$OUT_DIR"
```

The default solvers are `po` and `rt`. To run specific solvers, here PO and
an RT that retraces every chirp:

```bash
python -m validation.cli \
  --clip /path/to/clip_bundle \
  --suite dsp_maps \
  --out "$OUT_DIR" \
  --solvers po rt \
  --rt-refresh-frames 0
```

To validate a precomputed simulation cube without running simulation:

```bash
python -m validation.cli \
  --clip /path/to/clip_bundle \
  --simulated-adc /path/to/simulated_adc.npz \
  --suite range_time \
  --out "$OUT_DIR"
```

Parameterized motion always needs SMPL model files for mesh-derived range
metadata, including when `--simulated-adc` is supplied. Pass them explicitly:

```bash
python -m validation.cli \
  --clip /path/to/clip_bundle \
  --out "$OUT_DIR" \
  --smpl-model-dir "$MMWAVE_SMPL_MODEL_DIR"
```

## Suites

- `smoke`: minimal range-time and range-Doppler metric path.
- `range_time`: range-time map comparison only.
- `range_doppler`: range-Doppler map comparison only.
- `pose_range`: mesh-derived range summary without full DSP map comparison.
- `dsp_maps`: full default suite, including range-time, range-Doppler,
  pose-range metadata, and angle-map diagnostics when possible.

## Solvers

Supported `--solvers` values are the solver names shared by every HERMES entry
point:

- `po`: physical optics on the human only; the room and human-room coupling
  are ignored.
- `rt`: ray tracing of the whole scene. `--rt-refresh-frames N` retraces at the
  start of every N radar frames and phase-updates the traced paths in between;
  `0` retraces every chirp. The default is 1.
- `hybrid_po`: RT for the static room and PO for the human, with body blocking
  and human-room coupling.

When more than one solver is selected, the runner writes per-solver comparisons
and uses a primary simulation label for top-level summary fields.

### Hybrid PO Calibration

Validation enables RT-reference calibration for `hybrid_po` by default. The
default `--hybrid-calibration rt_first_pose` runs a full-scene
RT baseline for the first frame, extracts its single-human-only reference, and
fits one amplitude gain between that reference and the direct-human PO ADC.
Power is averaged across all chirps, virtual channels, and ADC samples of the
first simulated radar frame: `gain = sqrt(P_rt_human / P_po_human)`. This is
first-frame averaging, rather than a fit using only the first chirp. The same
gain is reused for every frame and chirp, scaling direct-human PO and both
human-environment coupling orders; the static-environment RT component is not
rescaled. The fit excludes static-room and coupling returns and is not repeated
on later chirps. Calibration uses ADC RMS power and therefore does not require
individual range-profile peaks to be identical.

This calibration gain is separate from the radar antenna pattern. Direct-human
PO and RT paths use the configured hardware gain and pattern. The experimental
human-environment coupling terms currently retain Tx/Rx geometry and array
phase but do not apply direction-dependent pattern loss, because their radar
departure and arrival directions differ and the supported TI pattern assets
are combined monostatic Tx/Rx cuts.

The available policies are:

- `rt_first_pose` (default): fit one amplitude gain from the first-frame RT
  and human-only PO ADC powers, averaged over the whole frame and reused for
  the entire run.
- `rt_sequence`: fit from selected RT/PO path-power samples across the
  available sequence.
- `none`: skip the RT calibration baseline and leave PO gain at unity.

Calibration adds an RT run to each hybrid simulation invocation. With
`--progress`, enabled calibration prints explicit `starting full-scene RT
baseline` and `finished full-scene RT baseline` messages. The calibration mode
is included in `metrics.json` and the simulated-ADC cache fingerprint, so
changing it rebuilds an incompatible cached hybrid cube.

## Channel Selection

Use the zero-based TX/RX selectors `--tx-indices` and `--rx-indices` to
validate a subarray while preserving channel ordering from the source ADC.
Index `i` is the physical label `TX(i+1)` or `RX(i+1)`:

```bash
python -m validation.cli \
  --clip /path/to/clip_bundle \
  --out "$OUT_DIR" \
  --tx-indices 0-2 \
  --rx-indices 0,1,2,3
```

Prepared RT-Pose bundles normalize the source MIMO receiver permutation to
physical `RX1..RX16` order. A compact horizontal cascade selection is:

```bash
python -m validation.cli \
  --clip /path/to/prepared_rtpose_bundle \
  --out "$OUT_DIR" \
  --tx-indices 0-2,9 \
  --rx-indices 4-11
```

This selects the physical labels TX1-TX3 and TX10 and RX5-RX12. RT-Pose's RX labels are not spatially
ordered; the selected RX set becomes `9,10,11,12,5,6,7,8` when sorted by
horizontal position and has half-wavelength spacing. The selected TX set has
half-wavelength horizontal projections but differing elevations. The runner
applies the same output-channel mask to measured and simulated ADC. See the
RT-Pose bundle-preparation README for the complete channel convention.

### TDM Timing Policies

`--ignore-tdm-timing` controls generated simulation of bundles whose sensor
metadata declares TDM virtual ADC. It does not change a non-TDM bundle and does
not alter ADC loaded through `--simulated-adc`.

Let `t_frame[f]` be the bundle's body-motion timestamp for frame `f`, `m` the
slow-time chirp-loop index, `s` the physical TX slot, `N_TX` the number of slots
in the full source schedule, and `T_chirp` the physical chirp repetition time.
The two policies are:

| Behavior | Default (without the flag) | With `--ignore-tdm-timing` |
|---|---|---|
| Simulator calls per solver | One per selected physical TX | One total |
| Mesh timestamp | `t_frame[f] + (m*N_TX + s)*T_chirp` | `t_frame[f] + m*N_TX*T_chirp` for every selected TX |
| Mesh update | Vertices are sampled/interpolated separately at every selected slot time; triangle indices stay fixed | Vertices are sampled/interpolated once per loop; every selected TX sees the same pose |
| Simulated hardware | One selected TX and the complete physical RX array per call; only requested RX outputs are retained | All selected TX and only selected RX elements in one sensor |
| PO | A separate PO sequence per selected TX; the PO surface term uses that TX and the full-RX phase center, followed by per-RX phase, spreading, delay, and hardware corrections | One PO sequence; the PO surface term uses the mean selected-TX and mean selected-RX phase centers, followed by per-virtual-channel phase, spreading, delay, and hardware corrections |
| RT | A separate multi-RX path bank per selected TX; the configured retrace/update cadence runs independently for each slot | One joint selected-TX/selected-RX path bank; the same retrace/update cadence runs once for the joint bank |
| Hybrid PO calibration | Each selected-TX hybrid invocation fits its own RT-reference gain for that slot sensor | The one joint hybrid invocation fits one gain for the selected TX/RX sensor |
| Slow-time/Doppler PRI | Full source TDM-cycle interval, `N_TX*T_chirp` | The same full-cycle interval; it is not shortened to the selected-TX count |
| Output channel order | Reassembled in the measured ADC's source order | Reordered to the same measured ADC source order |

The PO phase-center projection is an existing virtual-array approximation: PO
surface currents, visibility, incidence angle, and polarization are evaluated
at the applicable TX/RX phase centers. Each retained virtual channel then uses
its physical TX-to-face plus face-to-RX path length for carrier phase,
free-space spreading, and beat delay. The flag changes which antennas share
the common PO phase-center evaluation as shown in the table; it does not merely
remove a phase factor after a default TDM run.

For `rt`, "one run" does not mean one slow-time calculation. The joint path
bank is still updated at every chirp loop, and retracing follows
`--rt-refresh-frames`. With the default one-frame cadence, a one-frame clip
normally has one joint trace followed by coherent updates across all 64 loops;
with `--rt-refresh-frames 0` every loop is retraced.

For the RT-Pose profile (`N_TX=12`, `T_chirp=65 us`), both policies retain a
`780 us` slow-time interval. Without the flag, the bundled reverse schedule
places TX10, TX3, TX2, and TX1 at offsets `130 us`, `585 us`, `650 us`, and
`715 us`, respectively. With the flag, all four use the cycle-start timestamp.
In either case, every slow-time channel response synthesizes the bundle's 256
fast-time ADC samples.

To select the simultaneous approximation:

```bash
python -m validation.cli \
  --clip /path/to/prepared_rtpose_bundle \
  --out "$OUT_DIR" \
  --tx-indices 0-2,9 \
  --rx-indices 0-3 \
  --ignore-tdm-timing
```

The timing policy is included in `metrics.json` and the simulated-ADC cache
fingerprint. Switching the flag while reusing an output directory invalidates
and rebuilds `simulated_adc_<label>.npz`; the two variants do not coexist under
different filenames.

The core simulator supports consecutive slots in local hardware TX order.
Passing `bundle.sensor()` directly to it with omitted or reordered slots now
raises an actionable error. Use this bundle-aware validation runner for those
schedules; its per-slot path preserves physical TX identity and acquisition time.

Simulation caches use bundle-relative resource names and content hashes, so
running the same ZIP again with the same output directory reuses the cache
despite a different temporary extraction directory. Changing scene dependencies,
motion, sensor settings, model files, or solver controls invalidates the cache.
Older cache-schema versions are rebuilt once after this update.

## Outputs

The runner writes:

- `metrics.json`: suite metadata, clip metadata, raw diagnostics, and metrics.
- `maps.npz`: real and simulated DSP maps and axes.
- `simulated_adc_<label>.npz`: cached simulation cubes when simulation runs.
- `subarray_adc.npz`: selected ADC channels when subarray selection is active.
- PNG diagnostic plots unless `--no-plots` is passed.

Plot filenames contain every simulated solver in command-line order. For
example, `--solvers po rt` writes `range_time_real_vs_po_vs_rt.png` and
corresponding range-profile, range-Doppler, and angle-FFT files. Panel titles
and range-profile legend entries use the solver labels (PO, RT, Hybrid PO). A
user-supplied `--simulated-adc` retains the shorter `real_vs_sim` filename
convention because no solver was run.

Primary metrics include normalized correlation, peak range error, and energy
ratios inside the mesh-derived range gate.

The metrics are descriptive; the runner does not impose a universal accuracy
threshold. In particular, `--suite smoke` verifies that the bundle, simulator,
and DSP pipeline execute together—it is not by itself a physical-fidelity
certification. Quantitative fidelity claims should define dataset- and
scenario-specific acceptance thresholds on held-out frames.

## Useful Flags

- `--max-frames N`: limit runtime while debugging a bundle.
- `--background-subtract`: turn ON real-ADC background subtraction using
  `background_adc.npz`. Default behavior is no real-ADC background subtraction.
  The run fails if this flag is enabled and the clip bundle does not contain
  `background_adc.npz`. Simulated ADC is unchanged.
- `--clutter-removal none|mean`: control DSP-domain slow-time clutter
  removal while building real and simulated range-time/range-Doppler maps.
  `mean` subtracts the per-range-bin/channel slow-time mean after range FFT;
  `none` leaves DSP maps unchanged.
- `--scene /path/to/scene.xml`: override the clip environment scene.
- `--rt-refresh-frames N`: RT retrace cadence in radar frames; `0` retraces
  every chirp.
- `--no-rt-diffuse-reflection`: disable RT diffuse reflection.
- `--rt-refraction`: enable RT refraction.
- `--rt-human-specular-reflection`: enable RT specular reflection on the human.
- `--hybrid-calibration none|rt_first_pose|rt_sequence`: select the
  RT-reference PO calibration used by `hybrid_po`; the default is
  `rt_first_pose`.
- `--no-plots`: skip PNG output for faster CI-style runs.
