# mmRadPose limited validation fixture

This source-checkout fixture selects mmRadPose trial `p1_an0_ac4_r0`, action
4 (“Bicep curls”), frame `000240`. The upstream dataset contains no camera image
for this capture. This directory is a runnable current-schema HERMES validation
bundle and a limited interoperability example.

Contents:

- `bundle.json`: unified HERMES artifact index declaring the measured primary
  ADC, parameterized motion, and environment metadata;
- `radar_adc.npz`: measured ADC, shape `[1,128,64,12]`, representing the
  IWR6843AOPEVM's 3-TX by 4-RX channels flattened in TX-major order;
- `amass_sequence.npz`: three AMASS-like base-SMPL motion samples centered on
  the radar frame, with `bundle_times` spanning approximately
  `-0.0667`–`0.0667` seconds;
- `sensor.json`, `frames.json`, and `benchmark_metadata.json`: portable timing,
  coordinate, alignment, and waveform metadata; and
- `environment.json`: an explicit empty simulator environment.

The motion archive stores poses, translations, base-SMPL topology, timing, and
coordinate metadata. Its ten shape coefficients are zero and its gender is
neutral; those release-time substitutions are not participant-specific shape.
No SMPL model, baked body mesh, skeleton, or point cloud is included. Mesh
generation therefore requires a separately licensed neutral SMPL model.

See `NOTICE.md` for the CC BY-SA 4.0 terms and attribution that apply to this
fixture.
