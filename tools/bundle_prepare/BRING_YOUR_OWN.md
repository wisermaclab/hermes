# Prepare your own measurement bundle

Install HERMES, then use the installed command; a source checkout and the GUI
are not required. Bundle preparation does not invoke a solver or SMPL.

**Create a measurement bundle from the command line:**

```bash
mmwave-prepare-bundle \
  --adc /path/to/capture/capture_adc.npz \
  --sensor /path/to/capture/sensor.json \
  --scene /path/to/capture/scene.xml \
  --target-mesh /path/to/capture/plate.npz \
  --target-material /path/to/capture/material.json \
  --output /path/to/capture/capture-bundle.zip
```

This reads your decoded ADC, sensor settings and optional scene/target inputs,
checks them, and writes the self-contained directory or ZIP named by `--output`.
Direct paths are relative to the current directory. Direct mode requires
`--adc`, `--output`, and either `--sensor` or `--ti-config` with `--board-model`;
scene, target and material inputs are optional. The ADC origin defaults to
`measurement`; use `--origin simulation` for synthetic inputs.

Alternatively, use `--config /path/to/capture/prepare.json` as described
[below](#replace-the-synthetic-inputs-with-a-capture). Choose direct inputs or
`--config`; `--output` and `--force` work with both. Use `--help` for all options.

## Try the packaged synthetic example

```bash
mmwave-prepare-bundle --example /tmp/hermes-static-example
mmwave-prepare-bundle --config /tmp/hermes-static-example/prepare.json
mmwave-check-bundle /tmp/hermes-static-example/static-target.zip --all-errors --json
mmwave-check-bundle /tmp/hermes-static-example/static-target.zip \
  --diagnose /tmp/hermes-static-diagnostics --reference-position 2 0.5 0.3
```

The inputs include analytic ADC, a square plate mesh centred at `(2, 0.5, 0.3)`
metres, an empty environment scene, and sensor metadata. They contain no
participant data or licensed body model. The ADC is labelled `simulation`:
it is a point-return reference with exact element-to-target distances, rather
than a measured capture or a PO solution of the plate. Its amplitudes therefore
need not match PO or RT. See the included `NOTICE.txt`.

Run simulation validation directly from the ZIP:

```bash
mmwave-validation-benchmark --clip /tmp/hermes-static-example/static-target.zip \
  --solvers po rt --rt-human-specular-reflection --suite smoke --out /tmp/hermes-static-validation
```

Preparation, checking and validation accept portable ZIPs or bundle directories.
Validation extracts ZIP assets temporarily for the run and leaves the source
archive unchanged. The Python `mmWaveRadar.measurements.load_bundle()` API also
loads either representation.

## Replace the synthetic inputs with a capture

Decode your recorder's raw files into a pickle-free NumPy archive with numeric
`adc [frames, chirps_per_tx, samples, virtual_channels]`. Decode packet layout,
endianness, sign extension, dropped packets, I/Q packing, and synchronization
in your acquisition adapter. The preparation command does not guess them.

Use the direct arguments above, or save the following as `prepare.json` next
to the normalized inputs. This config example describes two measured frames;
adjust the frame times to match your capture:

```json
{
  "schema_version": 1,
  "adc": "capture_adc.npz",
  "sensor": "sensor.json",
  "origin": "measurement",
  "frame_times_s": [0.0, 0.05],
  "scene": "room/scene.xml",
  "target_mesh": "plate.npz",
  "output": "capture-bundle.zip"
}
```

Then **create the bundle and check it**:

```bash
mmwave-prepare-bundle --config /path/to/capture/prepare.json
mmwave-check-bundle /path/to/capture/capture-bundle.zip --all-errors --json
```

Then **simulate and compare with the measured ADC** using the bundled sensor,
scene and target:

```bash
mmwave-validation-benchmark --clip /path/to/capture/capture-bundle.zip \
  --solvers po hybrid_po --out /path/to/capture/validation-results --progress
```

This writes metrics, DSP maps, generated ADC and comparison plots. Evaluated
human meshes need no body-model files; parameterized human motion requires
`--smpl-model-dir /path/to/licensed/models` or `MMWAVE_SMPL_MODEL_DIR`.

If you are running from a source checkout without the installed command, use
the Python launcher from the directory containing `pyproject.toml`:

```bash
python tools/prepare_bundle.py --config /path/to/capture/prepare.json
PYTHONPATH=src:tools python -m bundle_prepare.integrity \
  /path/to/capture/capture-bundle.zip --all-errors --json
PYTHONPATH=src:tools:. python -m validation.cli \
  --clip /path/to/capture/capture-bundle.zip \
  --solvers po hybrid_po --out /path/to/capture/validation-results --progress
```

Preparation expects decoded numeric ADC; it does not decode recorder
formats such as KVNL or DCA1000 binary files. For the JSON workflow, replace
the direct inputs with `--config /path/to/capture/prepare.json`.

All paths in this config, including `output`, are relative to the config file.
The optional CLI `--output` override is relative to the current directory.
Existing outputs are retained unless `--force` is supplied; a failed check
leaves the old output intact. A destination without `.zip` produces a directory.

Specify **one sensor source**: `sensor`, or `ti_config` plus `board_model`.
For a real acquisition example, use the tutorial's
[TI IWR6843AOP metal-plate sensor JSON](../../docs/tutorial/examples/metal_plate_sensor.json).
It records the plate capture's waveform and explicitly identifies the assumed
TX order, frame period and I/Q/phase conventions; adapt those settings to your
capture. See the [tutorial explanation](../../docs/tutorial/index.html#prepare-bundle)
for the first-sample carrier calculation and channel layout.
The TI importer supports the existing `adcCfg`, `channelCfg`, `profileCfg`,
`chirpCfg`, and `frameCfg` subset, with constant simultaneous masks or repeated
single-TX TDM cycles. Unsupported waveform variations are rejected. It computes
the reference carrier at the first ADC sample from start frequency, ADC start
time and slope. Source waveform fields are retained in `sensor.json`. Inspect
the output against the actual acquisition configuration.

Specify **one timing source**:

- `frames`: a frames JSON sidecar, preserving source frame identifiers;
- `frame_times_s`: one frame-start time per ADC frame; or
- `start_time_s` and optional `frame_period_s` (defaults to the sensor period).

The direct equivalents are `--frames records.json`, `--frame-times-s 0 0.05`,
or `--start-time-s 0 --frame-period-s 0.05`. Frame-period arguments cannot be
combined with frame records or explicit frame times.

When motion is supplied, choose an explicit timing source, including an explicit
`start_time_s` if using regular spacing. Frame times must use the motion archive's
timeline; for epoch-valued motion, give epoch-valued frame times. The CLI never
silently rebases motion. Without motion and with no timing source, supplied ADC
`times [F,M]` determine relative frame starts; otherwise frame starts default to
zero plus the sensor frame period. Stored ADC acquisition times are rebased to
the first frame. Supplied times must agree with
frame starts and FMCW loop intervals within the declared timestamp tolerance
(default: one percent of the loop interval, at least 1 ns, with source-clock
precision accounted for). Accepted timestamp jitter is retained. For TDM they identify loop starts; TX slot
offsets are represented separately. New TDM inputs must declare the physical
`fmcw.tdm_tx_order`; the writer does not guess it. Motion archives must already be aligned to
the `frames.json` motion timeline, including the final physical chirp.

Optional config inputs are `background_adc`, `environment`, `target_mesh`,
`target_obj`, `target_material`, `motion_parameters`, and `evaluated_mesh`.
Their direct flags use hyphens, for example `--background-adc background.npz`,
`--target-material material.json`, or `--evaluated-mesh motion.npz`. Motion
also requires `--motion-coordinate-frame world` in direct mode.
`target_material` names a numeric RadioMaterial JSON object; it is embedded in
the descriptor, so validation needs no external material file. The example
declares aluminium-like conductivity. Enable `--rt-human-specular-reflection`
for specular target returns (the historical flag also applies to static targets). Body parameters are
required only when re-evaluation through SMPL is wanted. A static target needs
no body model. Meshes and evaluated motion use world coordinates in metres;
parameterized motion must use the conventions in the bundle specification.
If either motion input is supplied, explicitly set
`motion_coordinate_frame: "world"` after normalizing its coordinates; the CLI
does not perform or infer a coordinate transform. Preparation rejects incomplete
motion coverage before publication, including with `--force`. Older bundles
with incomplete motion remain readable for playback, with re-simulation disabled.

Sensor inputs must establish `fmcw.carrier_reference: "first_adc_sample"` or
provide start frequency and ADC start delay from which it can be derived.
An unlabeled carrier without that provenance cannot be written as a new bundle.
ADC arrays must match the declared sampling mode: complex I/Q for `complex1x`
or `complex2x`, and real samples for `real` (zero-imaginary complex storage is
also accepted). Preparation does not recover missing I/Q information.
Use `scene` for scene import, rather than external scene paths inside the
environment sidecar. XML, OBJ, MTL and texture dependencies inside the source
scene directory are copied and their references rewritten. Remote resources,
escaping references and XML includes are rejected. Keep the environment scene
separate from the declared target to avoid duplicating it during simulation.
Assets are limited to 256 MiB each and scene XML to 2 MiB.

Stored ADC follows the canonical mixer and board phase-sign conventions in
[the specification](../../validation/BUNDLE_SPEC.md). If your decoder emits a
different IQ convention, explicitly select `iq_transform: "conjugate"` or
`"swap_iq"`. If it already removed fixed board signs, set
`phase_signs_removed: true` to restore them before bundling. The same
normalization is applied to a supplied background ADC, so it must originate
from the same decoder convention. These transformations are recorded in
`preparation.json`. Declare `adc_units`; no amplitude normalization is guessed.
Direct mode accepts `--adc-units`, `--iq-transform`, `--phase-signs-removed`,
and `--timestamp-tolerance-s` for the same conventions and timestamp checks.

Inspect board channels without reading implementation code:

```bash
mmwave-prepare-bundle --channel-table IWR6843AOP
```

## Use the Python writer

```python
from bundle_prepare import BundleSensorConfig, BundleWriter
from bundle_prepare.contract import make_bundle_descriptor

sensor = BundleSensorConfig.from_json("sensor.json")
with BundleWriter("capture.zip") as writer:
    writer.write_json("sensor.json", sensor.to_json())
    writer.write_json("frames.json", [{"motion_time_s": 0.0}])
    writer.write_npz("radar_adc.npz", adc=adc)  # numeric [1,M,N,C]
    descriptor = make_bundle_descriptor(primary_origin="measurement")
    descriptor["scene"] = writer.add_scene("room/scene.xml")
    writer.copy_file("plate.npz", "plate.npz")
    descriptor["target"] = {"mesh": "plate.npz"}
    writer.write_json("bundle.json", descriptor)
```

The context manager stages, canonicalizes, checks and publishes. With no
destination, `writer.to_bytes()` returns a checked ZIP for a download. It does
not require importing the GUI.

## Measured conformance release milestone

The synthetic example exercises packaging and reproducible numerical
conventions. Verified measurement ingestion additionally requires a
redistributable static-target capture with independently recorded radar
settings and target positions. Acquire at least two positions with nonzero
azimuth and elevation, retain raw/decoder provenance, and run diagnostics with
`--reference-position X Y Z` for each bundle. Confirm range within the chosen
resolution tolerance and both angle signs. Compare deliberately conjugated
and selected permuted-channel variants, documenting which errors the geometry
can constrain. No single target identifies every channel permutation.

This measured fixture is an explicit remaining release requirement for claims
of verified physical capture ingestion; the synthetic data must never be
relabeled as a measurement. Diagnostics report warnings and cannot establish
calibration or physical fidelity by themselves.
