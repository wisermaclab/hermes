# mmRadPose bundle adapter

This source-checkout tool prepares and validates one standard HERMES bundle
from a separately obtained mmRadPose trial. It does not download or
redistribute mmRadPose data, fit SMPL parameters, or provide SMPL models.

The interface has five domain inputs:

1. mmRadPose dataset directory;
2. trial ID;
3. radar frame ID;
4. pickle-free AMASS-like base-SMPL sequence; and
5. licensed SMPL model directory.

The two directories may be set once:

```bash
export MMRADARPOSE_ROOT=/path/to/mmRadarPose
export MMWAVE_SMPL_MODEL_DIR=/path/to/licensed/smpl_models
```

Then prepare one frame:

```bash
python tools/bundle_prepare/mmradarpose/prepare_mmradarpose_bundle.py \
  --trial p1_an0_ac4_r0 \
  --radar-frame-id 240 \
  --amass-npz /path/to/amass_like_base_smpl_sequence.npz
```

By default, the bundle is written below the operating system's temporary
directory at
`hermes-validation-bundles/mmradarpose/<trial>/bundles/`. Pass
`--output-root /path/to/private/bundles` to retain it elsewhere. Outputs are
participant-derived and should not be written into the source checkout.

The adapter discovers these trial inputs automatically:

```text
radar/data_cube_parsed_<trial>.npz
radar_to_smpl_alignment_inferred.json
```

It uses the radar frame ID as the exact motion `frame_ids` key, selects the
small motion window needed to cover the physical radar acquisition, and
exports the IWR6843AOPEVM ADC with 12 virtual channels in TX-major order. It
writes `amass_sequence.npz` with source and radar-relative timing. It stores no
duplicate body-motion archive; validation evaluates SMPL in memory. Bundle
naming and motion-window selection are deterministic. Diagnostic
skeleton, radar-target point cloud, and source-matrix transfers are
intentionally outside this preparation path.

## Motion input

The motion sequence may come from any compatible producer.
[SparseSMPLFit](https://github.com/wisermaclab/SparseSMPLFit) is
one optional producer; HERMES neither bundles nor invokes it. The archive
contract is:

```text
poses:              (T, 72), finite axis-angle values
trans:              (T, 3), finite translations in metres
betas:              (10,), finite shape coefficients
gender:             scalar neutral, male, or female
world_up_axis:      scalar "z"
world_forward_axis: scalar "-y"
frame_ids:          (T,), strictly increasing mmRadPose trial-frame keys
pose timing:        strictly increasing times (T,) in seconds, or a positive
                    scalar mocap_framerate in Hz
profile:            optional scalar "mmradarpose"
source_sequence_id: optional scalar matching the requested trial
```

The motion must use canonical AMASS world axes (`+Z` up, `-Y` forward, and
`+X` subject-left). The adapter composes the dataset's radar-to-native-SMPL
calibration with the native-SMPL-to-AMASS serializer rotation before applying
translation or baking meshes. Native SMPL `+Y`-up/`+Z`-forward arrays are
rejected rather than silently passed through the AMASS path.

Radar timestamps, channel metadata, and the radar-to-body transform belong to
the adapter and do not need to be embedded in this archive. Every NPZ member is
loaded with `allow_pickle=False`; unsafe dtypes, invalid timing, inconsistent
provenance, and non-finite values are rejected.

The prepared `amass_sequence.npz` preserves the selected source `times`, adds
matching absolute `source_capture_times`, and records `bundle_times` relative
to the selected radar frame at `t=0`. Poses, translations, betas, gender, and
producer provenance are copied without refitting. Validation uses
`bundle_times` and evaluates SMPL in memory with the user-provided model.

## Laser-only environment preparation

mmRadPose has no synchronized stereo input in this workflow. Environment
geometry comes only from laser measurements. The same command applies this
precedence:

1. `--environment-geometry`, if supplied;
2. a standard dataset-local `environment_geometry/` directory; or
3. a standard environment/background/map cloud under `laser/`, `lidar/`, or
   `environment/`.

`--environment-geometry` may name a reusable fitted geometry directory, scene
XML, static mesh NPZ, or raw `.pcd`, ASCII `.ply`, `.npy`, `.npz`, `.xyz`,
`.txt`, or `.csv` laser cloud. A raw cloud must already be registered to the
mmRadPose radar frame:

```text
x = lateral, y = range/boresight away from radar, z = vertical
```

The adapter removes points overlapping the selected body, fits simple room and
furniture surfaces, converts the result back to the mmRadPose axes, and puts
portable JSON, OBJ, mesh NPZ, and Mitsuba/Sionna XML assets in each bundle. It
does not copy the raw laser cloud. If no geometry or laser input is available,
the bundle remains valid and records an empty environment explicitly.

The dataset's `pointcloud/.../targetlist_64.npy` files are radar detections, not
laser geometry, and are never auto-selected by this stage.

The completed bundle is staged, checked by the dataset-neutral bundle-integrity
checker, and moved into place only after that check succeeds. Prepared outputs
contain participant measurements and fitted body parameters; keep them private
unless the applicable dataset, consent, and body-model terms allow
redistribution. See
[`THIRD_PARTY_NOTICES.md`](../../../THIRD_PARTY_NOTICES.md).
