# RT-Pose bundle adapter

This source-checkout tool prepares and validates one HERMES frame bundle from
a separately obtained RT-Pose dataset. It does not download or redistribute
RT-Pose, fitted participant data, or SMPL models, and it does not fit SMPL
parameters.

The five preparation inputs are:

- the RT-Pose dataset directory;
- the RT-Pose sequence ID;
- the measured radar frame ID;
- a pickle-free AMASS-like base-SMPL sequence; and
- a separately licensed SMPL model directory used to evaluate the selected
  pose for foreground removal and verify its topology. The evaluated mesh is
  transient and is not stored in the bundle.

The dataset and model directories may be supplied through the environment:

```bash
export RTPOSE_ROOT=/path/to/your/RT-Pose-checkout
export MMWAVE_SMPL_MODEL_DIR=/path/to/your/licensed/smpl_models
```

Then prepare one bundle with:

```bash
python tools/bundle_prepare/rtpose/prepare_rtpose_bundle.py \
  --sequence 176 \
  --radar-frame-id 20
```

By default, the bundle is written below the operating system's temporary
directory at
`hermes-validation-bundles/rtpose/sequence_<seq>/bundles/`. Pass
`--output-root /path/to/private/bundles` to retain it elsewhere. Outputs are
participant-derived and should not be written into the source checkout.

Use `--dataset-dir` and `--smpl-model-dir` instead of the environment variables
when desired. `--output-root` changes the bundle destination, and `--force`
replaces a bundle previously created by this tool. `--dataset-dir` accepts
either the RT-Pose checkout root or its `Data` directory. When `--amass-npz` is
omitted, the adapter searches `Data/sequences/<sequence>/smpl`. If a
[SparseSMPLFit](https://github.com/wisermaclab/SparseSMPLFit) `segments.json`
manifest is present, it resolves the requested
radar frame through `Train.json` and selects the unique segment whose canonical
`frame_ids` contain the synchronized pose/camera frame. Without a manifest it
applies the same frame-membership test to discovered `.npz` archives. A frame
excluded from all segments as bad is rejected rather than interpolated;
overlapping candidates require an explicit `--amass-npz`. No ADC input is required:
the adapter decodes the selected frame from RT-Pose's measured AWR2243 cascade
capture. If the standard dataset-local
`Data/sequences/<sequence>/radar/npy_raw/frameNNNN_adc.npz` already exists, it
is validated and reused as an optimization. The source capture and cache are
never modified.

The adapter also reads `Data/filemeta.txt` and records the sequence activity,
location, clean/cluttered scenario, occlusion, and source subject type. Using
the pose/camera frame synchronized through `Train.json`, it copies the original
stereo PNGs into `camera/left/` and `camera/right/` in the prepared bundle.
These user-local images are copied byte-for-byte and are not face-obfuscated;
the generated metadata states that explicitly. The tool does not add downloaded
RT-Pose data to this repository automatically.

## Cascade channel ordering

RT-Pose's calibrated MIMO-processing cube orders physical receivers as:

```text
RX13, RX14, RX15, RX16, RX1, RX2, RX3, RX4,
RX9, RX10, RX11, RX12, RX5, RX6, RX7, RX8
```

That permutation belongs to RT-Pose preprocessing; it does not redefine the
physical AWR2243 receiver labels. Before writing `radar_adc.npz`, the adapter
reorders the receiver axis into ascending physical order `RX1` through `RX16`.
Every exported virtual-channel block therefore follows this bundle convention:

```text
recorded physical TX slot, then physical RX1 ... RX16
```

The TX blocks remain in the measured TDM schedule, normally
`TX12, TX11, ..., TX1`, because their slot order determines chirp timing. The
bundle records that schedule as `tx_to_enable`. Its `sensor.json` also records
`source_rx_order`, `exported_rx_order`, and `rx_order` so the conversion is
auditable. A dataset-local ADC cache must declare its physical RX order (or use
the older unambiguous `rx_mimo_order` layout name); an ambiguous cache is
rejected rather than guessed.

For a smaller horizontal aperture, the validation CLI can select physical
antenna labels:

```bash
python -m validation.cli \
  --clip /path/to/prepared_rtpose_bundle \
  --subarray-tx 1-3,10 \
  --subarray-rx 5-12 \
  --suite dsp_maps \
  --out /tmp/rtpose_subarray_validation
```

The receiver labels are not spatially ordered. Sorted by horizontal position,
the selected set is `RX9, RX10, RX11, RX12, RX5, RX6, RX7, RX8`, located at
`23, 23.5, ..., 26.5` wavelengths and therefore spaced by half a wavelength.
The selected TX labels have horizontal coordinates `5.5, 5, 4.5, 4`
wavelengths for `TX1, TX2, TX3, TX10`. They are half-wavelength-spaced only in
horizontal projection; their elevations differ, so this is not a collinear
half-wavelength physical TX array. Validation applies this selection to both
measured and generated ADC and simulates only the selected physical TX slots
and RX channels. It retains their original positions in the full TDM schedule,
so channel order, chirp timing, and Doppler PRI remain consistent with the
capture.

When several sequences share the same radar-coordinate environment, pass its
previously fitted directory (or a scene XML or mesh NPZ) with:

```bash
  --environment-geometry /path/to/shared/environment_geometry
```

The supplied assets are validated and copied into the new bundle; the bundle
never depends on that external path after preparation.

## AMASS-like motion contract

The motion sequence may come from any compatible producer.
[SparseSMPLFit](https://github.com/wisermaclab/SparseSMPLFit) is
one optional producer; it is not bundled with or invoked by HERMES. Required
archive members are:

```text
poses:              (T, 72), finite axis-angle values
trans:              (T, 3), finite translations in metres
betas:              (10,), finite shape coefficients
gender:             scalar neutral, male, or female
world_up_axis:      scalar "z"
world_forward_axis: scalar "-y"
frame_ids:          (T,), strictly increasing RT-Pose pose/camera frame keys
pose timing:        strictly increasing times (T,) in seconds, or a positive
                    scalar mocap_framerate in Hz
```

The pose and translation arrays must already use canonical AMASS world axes:
`+Z` is up, `-Y` is forward, and `+X` points toward the subject's left. The
adapter converts that world to RT-Pose radar coordinates with
`[x, y, z] -> [-y, x, z]`, yielding `[range_forward, lateral_left, up]`.
Native SMPL `+Y`-up/`+Z`-forward arrays are rejected rather than interpreted
ambiguously.

All members must load with `numpy.load(..., allow_pickle=False)`; object arrays
are rejected. The adapter reads RT-Pose `Train.json`, automatically identifies
the participant whose label mapping matches the supplied pose frames and
selected radar frame, and maps `frame_ids` to `Radar_frameID`. Radar IDs and
timestamps therefore do not need to be present in the producer's source
segment. Bundle preparation adds the synchronized radar IDs and timestamps to
the output `amass_sequence.npz`.

## Automatic environment preparation

Environment preparation is part of the same `prepare_rtpose_bundle.py` call.
The adapter uses the first available source in this order:

1. geometry passed through `--environment-geometry`;
2. fitted geometry under
   `Data/sequences/<sequence>/map/environment_geometry/`;
3. a standard previously reconstructed background LiDAR PLY; or
4. a new reconstruction from the selected frame's LiDAR and synchronized
   stereo images.

The final two paths invoke the bundled environment fitter automatically. New
reconstruction uses the already baked center mesh for foreground removal, so
it does not require legacy fitted-joint fields or pickle loading. Intermediate
point clouds remain temporary. The fitted XML, OBJ, numeric mesh NPZ, JSON,
preview, and sanitized reconstruction statistics are copied into
`environment_geometry/`, making the final bundle self-contained. Scene
construction is dataset-specific preprocessing, not part of the universal
bundle contract.

The resulting fidelity-oriented bundle contains measured ADC, the aligned
canonical `amass_sequence.npz` motion window, and synchronized original stereo
images. It does not store a duplicate body-motion archive. The AMASS-like
archive directly copies
poses, translations, betas, gender, coordinate convention, frame rate, and
available producer provenance from the selected fitted segment; no refitting is
performed. It records three timelines explicitly:

- `times`: original segment-local AMASS sample times;
- `source_capture_times`: absolute source capture times; and
- `bundle_times`: times relative to the selected radar frame (`t=0`).

It also records synchronized `radar_frame_ids`, `radar_capture_times`, mesh
topology, and the RT-Pose output transform used by simulation. Validation
generates the mesh in memory from these parameters and a user-provided licensed
SMPL model. Producing a limited public fixture—for example by removing body
parameters, substituting neutral shape, or redacting imagery—is a separate,
explicit release-preparation process.

Prepared bundles contain participant measurements and fitted body parameters.
Keep them private unless the source terms and participant consent permit the
intended redistribution. See
[`THIRD_PARTY_NOTICES.md`](../../../THIRD_PARTY_NOTICES.md).
