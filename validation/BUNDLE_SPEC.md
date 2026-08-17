# HERMES Bundle v1

Status: public compatibility contract for the unreleased HERMES 0.1 release.

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
    "frame_window": {
      "begin_frame_index": 20,
      "end_frame_index": 40
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

The canonical transmit-timing declaration is the boolean
`fmcw.tdm_enabled`, defaulting to `false`. When false, every selected Tx/Rx
virtual channel is sampled simultaneously and consecutive slow-time samples
are one chirp repetition apart. When true, transmitters occupy consecutive
physical chirp slots; samples for one Tx/Rx pair are therefore separated by
`num_tx * chirp_repetition_time_s`. In that mode,
`num_chirps_per_frame` is the slow-time sample count per transmitter. Readers
also accept the earlier `metadata.tdm_virtual_adc: true` marker and the legacy
`source_adc_axes` TDM layout as compatibility inputs; new producers write the
canonical field while retaining the legacy marker where needed by older tools.

`frames.json` contains a list, or an object with a `frames` list. Its length
must equal the primary ADC frame count and its bundle indices are contiguous
from zero. When parameterized motion is declared, every `motion_time_s` must
lie on that motion's preferred timeline. Dynamic exports also record the
source motion-frame index and selected inclusive frame window.

## ADC arrays

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
files are never embedded. Dynamic GUI exports store radar-relative human
position and yaw in `environment.json`; re-simulation applies those transforms
to the declared parameterized motion.

## JSON and path safety

Normative schemas live under [`schema/`](schema/). All descriptor paths are
safe bundle-relative paths. Absolute paths, backslash paths, `..` traversal,
escaping symlinks, and references outside the bundle are invalid. ZIP readers
apply the same checks before extraction.

## Loading and validation

Validate a directory:

```bash
mmwave-check-bundle /path/to/bundle
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

- HERMES 0.1 has one Bundle v1 profile: `hermes`.
- Readers preserve declared axes, units, timing, origins, and channel order.
- New optional metadata is backward compatible.
- A new required field or changed meaning requires a future major schema
  version after the 0.1 contract is released.
- Dataset names are provenance values, never loader dispatch keys.
