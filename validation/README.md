# Validation

`validation/` contains the real-radar validation harness. It loads a standard
clip bundle, derives DSP maps from measured ADC, optionally runs simulator
modes for the same clip, and writes comparable metrics, maps, and plots.

Use this directory when the question is:

- Does simulated ADC match a captured radar clip?
- Do range-time or range-Doppler maps align with the measured data?
- Does a PO, RT, or hybrid mobility mode perform better for a specific clip?

Use `benchmarks/` instead for synthetic runtime and behavior benchmarks that do
not compare against measured radar ADC.

Bundle construction, loading, and structural integrity are owned separately by
`tools/bundle_prepare/`. The validation CLI runs the same complete integrity
check as `mmwave-check-bundle` before scientific validation; the standalone
command remains useful for checking a bundle without running metrics.

## Universal Bundle Contract

The bundle layout is the boundary between dataset preparation and validation.
The validation package consumes the required JSON/NPZ files without importing
the source dataset, its reader, or its conversion pipeline. Consequently, it
can validate any dataset that is expressed through this contract; dataset
names are provenance metadata, not dispatch keys. New datasets do not require
changes to the validation runner.

The normative, versioned public contract is
[HERMES Bundle v1](BUNDLE_SPEC.md). It defines one `hermes` profile for measured
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

## Human-Measurement Clip Layout

A validation clip is a directory containing:

- `sensor.json`
- `radar_adc.npz`
- `frames.json`
- `amass_sequence.npz`
- optional `background_adc.npz`
- optional `environment.json`
- optional `benchmark_metadata.json`
- optional `bundle_summary.json`

Required array shapes:

- `radar_adc.npz`: `adc [frames, chirps, adc_samples, virtual_channels]`
- optional `radar_adc.npz/times`: `[frames, chirps]`
- optional `background_adc.npz/adc`: `[F,M,N,C]`, `[M,N,C]`, or `[1,M,N,C]`

`amass_sequence.npz` is the required and only body-motion source. Its required
pickle-free arrays are:

- finite `poses [T,3*K]`, with `T > 0`;
- finite `trans [T,3]`;
- finite `betas [B]`, with `B >= 10`;
- strictly increasing finite source `times [T]`; and
- integer base-SMPL `faces [F,3]`, with indices in `[0,6890)`.

When `bundle_times [T]` is present, it must also be strictly increasing and is
the timeline used by bundle loading, simulation, and `frames.json`. Otherwise,
`times` is used. Reference-adapter outputs additionally retain `gender`,
`model_type`, `mocap_framerate`, frame IDs, coordinate convention, output
transform, selected-frame, and source-provenance fields. RT-Pose prepared
bundles also include:

- `source_capture_times [T]`: absolute source capture times;
- `bundle_times [T]`: motion times relative to the selected radar frame at
  `t=0`, used by validation and simulation;
- `radar_frame_ids [T]` and `radar_capture_times [T]`; and
- source indices, selected-frame identifiers, topology, and output-transform
  metadata.

These values are sliced directly from the selected source segment without
refitting. Meshes are generated from them and a separately licensed SMPL model
when simulation is run. `amass_sequence.npz` is the only accepted body-motion
archive; baked mesh files and alternate parameter filenames are not loaded.

`sensor.json` must define the board model and FMCW settings. It can use
`board_model` or `hardware_model`, and either an explicit `fmcw` object or a
known board `fmcw_profile`/`profile`. The optional boolean
`fmcw.tdm_enabled` defaults to false and is the canonical declaration that
the channelized ADC stores per-transmitter TDM slow-time samples.

`frames.json` can be a list of frame records or `{"frames": [...]}`. Its length
must match the ADC frame count. Every record must include finite
`motion_time_s` within the selected AMASS timeline and may include source frame
IDs and metadata. Bundle or benchmark indices must be contiguous from zero.
Simulation anchors every radar frame at its explicit `motion_time_s` and adds
the physical chirp offset within that frame. The AMASS window must therefore
extend through the final chirp of every selected radar acquisition; validation
rejects a motion window that would otherwise be clamped at its endpoint.

`environment.json` can provide `scene_path`; the runner uses it when simulation
runs and `--scene` is not supplied. `environment_mesh_sequence`, when present,
is optional static scene geometry; it is not a body-motion alternative to
`amass_sequence.npz`.

## Run Validation

Write outputs to a temporary directory:

```bash
OUT_DIR="$(python -c 'import tempfile; print(tempfile.gettempdir())')/validation_run"

python -m validation.cli \
  --clip /path/to/clip_bundle \
  --suite dsp_maps \
  --out "$OUT_DIR"
```

The default simulation modes are `human_only_po` and `rt_coherent_bank`. To run
specific modes:

```bash
python -m validation.cli \
  --clip /path/to/clip_bundle \
  --suite dsp_maps \
  --out "$OUT_DIR" \
  --mobility-mode human_only_po rt_retrace
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

## Simulation Modes

Supported `--mobility-mode` values are:

- `human_only_po`
- `rt_retrace`
- `rt_coherent_bank`
- `hybrid_static_env_po`

When more than one mode is selected, the runner writes per-mode comparisons and
uses a primary simulation label for top-level summary fields.

### Hybrid PO Calibration

Validation enables RT-reference calibration for `hybrid_static_env_po` by
default. The default `--hybrid-po-calibration rt_first_pose` runs a full-scene
RT baseline for the first frame, extracts its single-human-only reference, and
fits an amplitude gain between that reference and the direct-human PO ADC. The
gain is applied to direct-human PO and to both human-environment coupling
orders; the static-environment RT component is not rescaled. Calibration fits
ADC RMS power and therefore does not require individual range-profile peaks to
be identical.

This calibration gain is separate from the radar antenna pattern. Direct-human
PO and RT paths use the configured hardware gain and pattern. The experimental
human-environment coupling terms currently retain Tx/Rx geometry and array
phase but do not apply direction-dependent pattern loss, because their radar
departure and arrival directions differ and the supported TI pattern assets
are combined monostatic Tx/Rx cuts.

The available policies are:

- `rt_first_pose` (default): fit one amplitude gain from the first-frame RT
  baseline.
- `rt_sequence`: fit from selected RT/PO path-power samples across the
  available sequence.
- `none`: skip the RT calibration baseline and leave PO gain at unity.

Calibration adds an RT run to each hybrid simulation invocation. With
`--progress`, enabled calibration prints explicit `starting full-scene RT
baseline` and `finished full-scene RT baseline` messages. The calibration mode
is included in `metrics.json` and the simulated-ADC cache fingerprint, so
changing it rebuilds an incompatible cached hybrid cube.

## Channel Selection

Use 1-based TX/RX selectors to validate a subarray while preserving channel
ordering from the source ADC:

```bash
python -m validation.cli \
  --clip /path/to/clip_bundle \
  --out "$OUT_DIR" \
  --subarray-tx 1-3 \
  --subarray-rx 1,2,3,4
```

Prepared RT-Pose bundles normalize the source MIMO receiver permutation to
physical `RX1..RX16` order. A compact horizontal cascade selection is:

```bash
python -m validation.cli \
  --clip /path/to/prepared_rtpose_bundle \
  --out "$OUT_DIR" \
  --subarray-tx 1-3,10 \
  --subarray-rx 5-12
```

These are physical antenna labels. RT-Pose's RX labels are not spatially
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
| Simulator calls per mobility mode | One per selected physical TX | One total |
| Mesh timestamp | `t_frame[f] + (m*N_TX + s)*T_chirp` | `t_frame[f] + m*N_TX*T_chirp` for every selected TX |
| Mesh update | Vertices are sampled/interpolated separately at every selected slot time; triangle indices stay fixed | Vertices are sampled/interpolated once per loop; every selected TX sees the same pose |
| Simulated hardware | One selected TX and the complete physical RX array per call; only requested RX outputs are retained | All selected TX and only selected RX elements in one sensor |
| Human-only PO | A separate PO sequence per selected TX; the PO surface term uses that TX and the full-RX phase center, followed by per-RX phase, spreading, delay, and hardware corrections | One PO sequence; the PO surface term uses the mean selected-TX and mean selected-RX phase centers, followed by per-virtual-channel phase, spreading, delay, and hardware corrections |
| Coherent RT | A separate multi-RX path bank per selected TX; the configured retrace/update policy runs independently for each slot | One joint selected-TX/selected-RX path bank; the same retrace/update policy runs once for the joint bank |
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

For `rt_coherent_bank`, "one run" does not mean one slow-time calculation. The
joint path bank is still updated at every chirp loop, and retracing follows the
normal configuration. With the default one-frame retrace schedule, a one-frame
clip normally has one joint trace followed by coherent updates across all 64
loops. `rt_retrace` likewise retains its own retrace policy.

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
  --subarray-tx 1-3,10 \
  --subarray-rx 1-4 \
  --ignore-tdm-timing
```

The timing policy is included in `metrics.json` and the simulated-ADC cache
fingerprint. Switching the flag while reusing an output directory invalidates
and rebuilds `simulated_adc_<label>.npz`; the two variants do not coexist under
different filenames.

## Outputs

The runner writes:

- `metrics.json`: suite metadata, clip metadata, raw diagnostics, and metrics.
- `maps.npz`: real and simulated DSP maps and axes.
- `simulated_adc_<label>.npz`: cached simulation cubes when simulation runs.
- `subarray_adc.npz`: selected ADC channels when subarray selection is active.
- PNG diagnostic plots unless `--no-plots` is passed.

Plot filenames contain every simulated mobility mode in command-line order.
For example, `--mobility-mode human_only_po rt_retrace` writes
`range_time_real_vs_human_only_po_vs_rt_retrace.png` and corresponding
range-profile, range-Doppler, and angle-FFT files. Panel titles and range-profile
legend entries use the same complete mobility-mode names. A user-supplied
`--simulated-adc` retains the shorter `real_vs_sim` filename convention because
no simulator mobility mode was run.

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
- `--no-diffuse-reflection`: disable diffuse reflection for generated RT.
- `--human-specular-reflection`: enable human specular reflection.
- `--hybrid-po-calibration none|rt_first_pose|rt_sequence`: select the
  RT-reference PO calibration used by `hybrid_static_env_po`; the default is
  `rt_first_pose`.
- `--no-plots`: skip PNG output for faster CI-style runs.
