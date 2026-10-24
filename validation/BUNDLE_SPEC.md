# HERMES Bundle contract (v2 with v1 read compatibility)

Status: public compatibility contract for HERMES 0.2.0.

Newly prepared bundles also include optional `provenance.json`: software and
dependency versions, execution metadata when known, input checksums and upstream
provenance. This additive record does not change existing schema field meanings.

A HERMES bundle is a portable directory or ZIP containing one or more raw FMCW
ADC artifacts, a shared sensor configuration and frame timeline, and optional
scene, geometry, motion, manifest, and diagnostic resources. Measurement and
simulation bundles use the same organization and loader.

## Unified profile

Every bundle declares `profile: hermes` in a required `bundle.json`. Data origin
belongs to each ADC artifact, not to the bundle profile. This lets one bundle
contain a measured capture, background ADC, and several simulated candidates
without changing formats or loader branches.

Every bundle requires:

| Resource | Purpose |
| --- | --- |
| `bundle.json` | Artifact index, primary ADC selection, and optional resource map. |
| `sensor.json` | Radar hardware, pose, waveform, channel order, and acquisition metadata. |
| `frames.json` | One record per primary ADC frame, including its motion timestamp. |
| Primary ADC NPZ | The ADC artifact named by `bundle.json/primary_adc`. |

File names other than the three JSON sidecars are not semantically special;
the descriptor supplies their safe bundle-relative paths.

## Versions and required features

Existing `schema_version: 1` bundles remain readable. Writers use version 2
when they emit explicit channel maps, physical TDM slot lists, custom arrays, or embedded target materials.
Version 2 declares `required_features`, a unique list of `custom_array`,
`explicit_tdm_slots`, `channel_map`, and/or `target_material`. Readers reject unknown features and
require the matching declaration whenever these fields are used. Feature checks
happen before ADC allocation. An older v1 reader rejects version 2 rather than
silently ignoring array geometry while using a retained board name. A feature
list alone cannot protect an old reader that never implemented feature checks.

## Conventions

All scene and evaluated geometry is in a right-handed world frame, in metres.
A sensor's local +X is boresight, +Y is left, and +Z is up. `position` translates
the radar phase centre into the world. `orientation` is active yaw, pitch, roll
in radians with local-to-world rotation `Rz(yaw) @ Ry(pitch) @ Rx(roll)`.
Consequently positive pitch points local boresight down. Positive azimuth turns
from +X toward +Y; positive elevation points toward +Z. Parameterized motion
must declare or default `motion.coordinate_frame: "world"`; adapters normalize
source axes/handedness/units and retain the applied transform as provenance.
The current reader rejects other coordinate frames instead of guessing.

The dechirped mixer is `tx(t) * conj(rx(t))`. For physical coefficient `a`
including propagation phase `exp(-j*2*pi*fc*tau)`, stored ADC is
`conj(a) * exp(+j*2*pi*slope*tau*t_fast)`. With positive slope, a stationary
positive-range target therefore has positive beat frequency. Swapping I/Q is
`j*conj(adc)` and changes this convention. Producers normalize to it and record
any IQ transformation; consumers do not infer the mixer from the samples.

`carrier_frequency_hz` is the instantaneous frequency at the **first ADC
sample**, with `fmcw.carrier_reference: "first_adc_sample"`. From a TI profile,
`fc = start_frequency_hz + slope_hz_per_s * adc_start_time_s`. Synthesis uses
`t_fast[n] = n/fs`; wideband PO uses offsets `slope*t_fast[n]` from this same
reference. Wavelength and propagation/array phase use this frequency. A
sampled-sweep centre must be converted before importing; using it with this
time axis introduces a path-dependent phase error. Source waveform fields
(including start frequency and ADC start delay) are preserved in `fmcw`.
The legacy `center_frequency_hz` input is converted by subtracting
`slope*num_adc_samples/(2*sampling_frequency)`; `start_frequency_hz` uses the
declared ADC start delay. When both start frequency and ADC delay are present,
an explicit first-sample carrier must agree with them (within 1 Hz plus
relative floating-point tolerance). For unlabeled legacy carriers, those
source fields establish the first-sample frequency; conversion retains the
old value in `fmcw.carrier_conversion` and emits a warning. Without enough
source information, an unlabeled carrier remains available for playback/DSP,
but warns and disables re-simulation. New writers refuse that ambiguity and
emit the explicit first-sample reference. A nominal approximation must be
explicitly declared as such in waveform provenance.

Complex ADC modes require a complex I/Q array. Real mode accepts real arrays
or complex storage with an exactly zero imaginary component. The checker does
not reconstruct quadrature data or silently change the sampling mode.

`adc_units` identifies counts, volts or another explicitly named amplitude
unit (legacy default: `counts`). Samples are not automatically amplitude
normalized. Correlation metrics can ignore a constant scale; energy ratios and
absolute comparisons cannot. Frequency, time, translation and angular units
are Hz, seconds, metres and radians unless a field explicitly names otherwise.

Stored ADC retains fixed board channel phase signs. Simulation applies them
once to path coefficients; angle processing removes them once. A producer
whose decoder removed these signs restores them before writing the canonical
ADC. `array.phase_signs` represents fixed real +1/-1 signs, not arbitrary
complex calibration coefficients. Calibration beyond these signs needs its
own explicitly defined convention and provenance.

A board's channel table can be generated directly from the catalog:
`mmwave-prepare-bundle --channel-table BOARD`. It reports zero-based channel,
TX and RX indices, virtual position in wavelengths and fixed phase sign.
`channel_map [C,2]`, when supplied, is the authoritative physical zero-based
`(TX,RX)` table for the stored ADC axis. It enumerates every pair in the selected
TX/RX Cartesian product exactly once; arbitrary sparse products are not yet
supported. Otherwise the named `virtual_channel_order` applies. Legacy adapter
slot-packed layouts are normalized by the common writer into a channel map.

For TDM, `fmcw.tdm_tx_order` is the unique physical zero-based TX index list in
acquisition-slot order. Its length equals the legacy `fmcw.num_tx` slot count.
Slot `s` in loop `m` starts at `frame_start + (m*len(tdm_tx_order)+s)*PRI`.
Stored TX selection and channel order are independent of this list. Keeping a
subset of channels never shortens the acquisition cycle or the Doppler PRI.
`Bundle.sensor().fmcw` preserves this acquisition slot count even when the
runtime hardware contains fewer stored TX elements. Ignoring within-cycle
TDM timing preserves the same full loop interval.
The one-based metadata aliases `tx_to_enable` and `tdm_tx_order` are read only
for legacy bundles. New producers must not use those aliases as the contract.

Static `target.material_parameters` optionally embeds numeric `relative_permittivity`,
`conductivity`, `thickness`, `scattering_coefficient`, and `xpd_coefficient`.
Validation uses this material unless the user supplies a material override.
Relative permittivity is at least 1, conductivity/thickness are nonnegative,
and scattering/XPD coefficients lie in [0,1]. All values must be finite.

## Explicit arrays

A sensor supplies a board preset or an `array` object. When both are present,
the explicit array is authoritative and the board name is descriptive metadata.
The array contains `tx_positions_m [TX,3]`, `rx_positions_m [RX,3]` in the radar
local frame, optional `channels [TX*RX,2]` in physical zero-based index order,
optional `phase_signs [TX*RX]`, and optional combined voltage-gain `gain_dbi`.
Positions must be finite and channels must enumerate the full physical product
exactly once. Use `tx_indices_zero_based`/`rx_indices_zero_based` to select the
stored product, and `channel_map` to declare its order. Explicit arrays
currently require `pattern_mode: "none"`; default combined gain and TX power
are 0 dB and 0 dBm. No TI pattern or board gain is silently inherited.

## Descriptor

`primary_adc` names one entry in `adcs`. Each ADC entry declares a path and one
of these origins:

| Origin | Meaning |
| --- | --- |
| `measurement` | ADC captured by radar hardware. |
| `simulation` | ADC synthesized by a declared solver or simulation mode. |
| `background` | Background reference used for optional subtraction. |
| `processed` | Derived ADC that is no longer a raw capture or solver output. |

The primary artifact must have origin `measurement` or `simulation`. A bundle
may declare at most one background artifact. Every non-background ADC must have
the same `[F,M,N,C]` shape as the primary ADC.

Example containing measured and simulated data:

```json
{
  "schema_version": 1,
  "profile": "hermes",
  "primary_adc": "capture",
  "adcs": {
    "capture": {
      "path": "radar_adc.npz",
      "origin": "measurement"
    },
    "background": {
      "path": "background_adc.npz",
      "origin": "background"
    },
    "full_rt": {
      "path": "simulated_adc_full_rt.npz",
      "origin": "simulation",
      "solver": "rt",
      "simulation_mode": "full_rt"
    }
  },
  "motion": {
    "parameters": "amass_sequence.npz",
    "evaluated_mesh": "human_motion.npz",
    "time_window": {
      "start_s": 0.5,
      "end_s": 1.5
    }
  },
  "scene": {
    "file": "scene.xml",
    "assets": ["target.obj"]
  },
  "environment": "environment.json",
  "manifests": {
    "full_rt": "experiment_manifest_full_rt.json"
  },
  "diagnostics": {
    "solver": "solver_diagnostics.npz"
  }
}
```

The optional resource groups are capability declarations:

- `motion.parameters` identifies pickle-free AMASS-like parameters that can be
  evaluated again when the required licensed model is supplied.
- `motion.evaluated_mesh` identifies an already evaluated mesh sequence for
  inspection and playback.
- `target.mesh` and `target.obj` identify transformed target geometry.
- `scene.file` and `scene.assets` identify the portable scene envelope and its
  declared assets.
- `environment` identifies coordinate transforms, placement, and other scene
  metadata.
- `manifests` and `diagnostics` index reproducibility and diagnostic sidecars.

Readers determine capabilities from these resources. For example, the GUI
offers human-motion re-simulation when `motion.parameters` is present; it does
not infer that capability from the bundle's primary ADC origin.

## Sensor and frame metadata

In `sensor.json`, `position` is in metres and `orientation` is radar
yaw/pitch/roll in radians. Producers may also include the degree-valued
`orientation_deg` mirror. `pattern_mode` identifies the selected digitized,
omnidirectional (`none`), or cosine response; cosine bundles record the full
power-pattern width in `cosine_3db_beamwidth_deg`. Producers using only a
subset of the named board record nonempty, unique, zero-based element lists in
`tx_indices_zero_based` and `rx_indices_zero_based`.

The canonical transmit-timing declaration is `fmcw.tx_schedule`, either
`"simultaneous"` (the default) or `"tdm"`. With `"simultaneous"`, every
selected Tx/Rx virtual channel is sampled at once and consecutive slow-time
samples are one chirp repetition apart. With `"tdm"`, transmitters occupy
consecutive physical chirp slots; samples for one Tx/Rx pair are therefore
separated by `num_tx * chirp_repetition_time_s`. In that mode,
`num_chirps_per_frame` is the slow-time sample count per transmitter. Readers
also accept the boolean `fmcw.tdm_enabled` written by earlier HERMES
releases, the `metadata.tdm_virtual_adc: true` marker, and the legacy
`source_adc_axes` TDM layout as compatibility inputs; new producers write
`tx_schedule` while retaining the legacy marker where needed by older tools.

`frames.json` contains a list, or an object with a `frames` list. Its length
must equal the primary ADC frame count and its bundle indices are contiguous
from zero. When parameterized motion is declared, every `motion_time_s` must
lie on that motion's preferred timeline. Dynamic exports also record the
selected motion time window as `motion.time_window`, in seconds after the
motion's first pose; each frame's `motion_time_s` is that radar frame's start,
and the motion advances in real time within the frame. Earlier producers may
instead declare a pose-index `motion.frame_window`, which readers accept.

## ADC arrays

The sensor's `fmcw.adc_mode` may be `complex1x` (default), `complex2x`, or
`real`. Declare the acquisition mode explicitly for hardware captures; complex
array storage alone does not distinguish Complex 1x from Complex 2x. Optional
`fmcw.if_bandwidth_hz` caps the simulated receiver IF bandwidth. Synthesized
returns outside the usable IF band are removed before sampling; the range FFT
retains all bins for Complex 1x and half for Complex 2x/real.

Every ADC NPZ must load with `numpy.load(..., allow_pickle=False)` and contain:

| Key | Shape | Meaning |
| --- | --- | --- |
| `adc` | `[F,M,N,C]` | Raw complex ADC: frames, chirps, fast-time samples, virtual channels. |
| `times` | `[F,M]`, optional | Chirp timestamps in seconds relative to bundle time zero. |

All values must be finite. `N` and `M` must equal the sensor FMCW sample and
chirp counts. `C` must equal the selected virtual-channel count. The channel
axis follows `virtual_channel_order`; readers never guess or reorder an
undocumented convention.

A background ADC may have shape `[F,M,N,C]`, `[M,N,C]`, or `[1,M,N,C]`.

## Geometry and human motion

A target mesh NPZ contains finite `vertices [V,3]` or `[1,V,3]`, integer
`faces [P,3]`, and optional `times [1]`. An evaluated motion mesh uses
`vertices [T,V,3]`, `faces [P,3]`, and `times [T]`.

An AMASS-like parameter archive contains:

| Key | Shape | Meaning |
| --- | --- | --- |
| `poses` | `[T,3*K]` | Finite axis-angle base-SMPL pose parameters. |
| `trans` | `[T,3]` | Finite body translation in metres. |
| `betas` | `[B]`, `B >= 10` | Finite body-shape parameters. |
| `times` | `[T]` | Strictly increasing source times in seconds. |
| `faces` | `[P,3]` | Integer base-SMPL topology with indices in `[0,6890)`. |
| `bundle_times` | `[T]`, optional | Strictly increasing bundle-relative motion timeline. |

When `bundle_times` exists, it is authoritative. Licensed SMPL-family model
files are never embedded. Dynamic GUI exports store `human_position_m [3]` (metres) and
`human_yaw_deg` (degrees) in `environment.json`; re-simulation applies that
placement to parameterized motion only. Evaluated meshes already include world
placement. Both motion timelines must cover the final physical chirp start,
even for a single radar frame; checking this never needs a licensed model.

## Scene resources and timestamp consistency

`bundle.json/scene.file` is canonical and describes the static environment.
Declared target/motion resources are inserted separately by simulation; do not
embed a second copy of that target in the environment scene. Legacy `environment.json/scene_path`
and `scene_file` are contained, checked aliases. Multiple declarations must
resolve to the same file. `sensor.metadata.scene_path` is provenance only and
is never used to open a scene. An explicit validation `--scene` is a user
chosen override outside the bundle contract. Scene filename dependencies must
remain inside the bundle; XML includes are unsupported.

Optional environment references include `scene_path`, `scene_file`,
`environment_mesh_sequence`, `environment_obj`, `motion_parameters`,
`motion_mesh`, and the recognized geometry paths within `geometry_outputs`.
The optional legacy `benchmark_metadata.bundle_contents.camera_images` extension
is a stereo PNG pair keyed by `left` and `right`; it is not required for a
bundle and makes no assertion about other camera arrangements.

`frames.json/motion_time_s` is the frame start on the preferred motion timeline.
ADC `times [F,M]` are chirp/loop starts relative to the first radar frame. In
version 2, they must equal `motion_time_s[f] - motion_time_s[0] +
m*slow_time_interval`; the first entry is zero. A TDM timestamp denotes the loop
start, with physical slot offsets supplied by `tdm_tx_order`. A preparation
config rebases source ADC timestamps and rejects inconsistent frame/chirp times.
This check applies to v1 and v2, and accepts absolute clock epochs. The default
jitter tolerance is one percent of the loop interval, with a 1 ns minimum;
`sensor.timestamp_tolerance_s` (or the preparation setting of the same name)
sets an explicit tolerance. Four ULPs of the source clocks supply a precision
floor. Comparisons use an absolute tolerance after rebasing, never a relative
tolerance that grows with the epoch or run length. Accepted jitter is retained.

Motion that does not cover the final physical chirp remains loadable for ADC
display and playback. It sets `can_resimulate` to false, with reasons in
`Bundle.resimulation_issues` and the integrity report. Fresh simulation still
requires full acquisition coverage. Legacy GUI scenes that embed the same
declared `target.obj` have that copy removed from the environment at simulation
time, before inserting the separate target; source files remain unchanged.
Version 1 retains its historical timestamp acceptance for reading older exports.

## JSON and path safety

Normative v1/v2 schemas live under [`schema/`](schema/). All descriptor paths are
safe bundle-relative paths. Absolute paths, backslash paths, `..` traversal,
escaping symlinks, and references outside the bundle are invalid. ZIP readers
apply the same checks before extraction.

## Loading and validation

Validate a directory:

```bash
mmwave-check-bundle /path/to/bundle --all-errors --json
```

Load a directory or GUI-exported ZIP:

```python
from mmWaveRadar.measurements import load_bundle

bundle = load_bundle("hermes-bundle.zip")
print(bundle.data_origin, bundle.available_simulations)
candidate = bundle.load_bundled_simulation("full_rt")
comparison = bundle.compare_adc(candidate)
```

The loader computes a SHA-256 fingerprint over each regular file and its
bundle-relative path. Structural validation establishes internal consistency;
it does not certify physical fidelity, synchronization accuracy, calibration,
provenance claims, or redistribution rights.

## Compatibility policy

- HERMES 0.2.0 uses the `hermes` profile, with schema version 1 read compatibility
  and schema version 2 feature gates. Bundle schema versions are independent of
  the HERMES package version.
- Readers preserve declared axes, units, timing, origins, and channel order.
- New optional metadata is backward compatible.
- A new required field or changed field meaning requires a new major bundle
  schema version.
- Dataset names are provenance values, never loader dispatch keys.
