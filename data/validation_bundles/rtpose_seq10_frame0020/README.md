# RT-Pose sequence 10 validation fixture

This limited fixture contains RT-Pose sequence 10, radar frame `000020`,
synchronized camera/pose frame `000056`, and a three-sample neutralized motion
window. It is a source-checkout interoperability example, not an accuracy claim
or a redistribution of the RT-Pose dataset.

The runnable bundle contains:

- a unified artifact index in `bundle.json`;
- one measured AWR2243-cascade ADC frame in
  `radar_adc.npz`, shaped `[1, 64, 256, 192]`;
- the sensor and physical TDM schedule in `sensor.json`;
- one frame mapping in `frames.json`;
- a pickle-free base-SMPL motion window in `amass_sequence.npz`;
- face-occluded left/right setup images; and
- a simplified, self-contained static scene in
  `environment_geometry/environment_scene.xml`.

The public motion archive retains pose, translation, timing, coordinate, and
topology fields. Participant-specific shape coefficients, gender, subject ID,
and fitted root-joint offset were removed: `betas` is all zeros and `gender` is
`neutral`. It does not contain a body-model file or baked human mesh. A
separately licensed neutral SMPL model is required to generate the human mesh.

Run the integrity check from the public source checkout:

```bash
python -m bundle_prepare.integrity \
  data/validation_bundles/rtpose_seq10_frame0020
```

For the compact horizontal subarray used by the validation examples:

```bash
python -m validation.cli \
  --clip data/validation_bundles/rtpose_seq10_frame0020 \
  --out /tmp/rtpose_seq10_frame0020 \
  --smpl-model-dir /path/to/licensed/smpl_models \
  --subarray-tx 1-3,10 \
  --subarray-rx 5-12 \
  --ignore-tdm-timing \
  --progress
```

The ADC channel axis preserves measured TDM TX-block order
`TX12, TX11, ..., TX1`; receivers within each block were converted from the
RT-Pose MIMO-processing order to physical `RX1, ..., RX16`. Validation selects
channels by physical antenna label and keeps output channels in the measured
ADC-axis order. See [`validation/README.md`](../../../validation/README.md) for
the exact default and `--ignore-tdm-timing` simulation policies.

The fixture is excluded from the Python wheel and source distribution. Its
upstream license, attribution, modifications, privacy limitations, and scene
material notes are in [`NOTICE.md`](NOTICE.md).
