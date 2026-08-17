# RT-Pose fixture notice

This directory contains an adapted third-party data fixture, not project-owned
HERMES data. The fixture assets are expressly excluded from the HERMES PolyForm
code license and the CC BY-NC project-owned documentation/data license. This
HERMES-authored notice is project documentation under CC BY-NC 4.0, except for
reproduced third-party text, but that license does not extend to the assets it
describes.

## Source and license

Source: **RT-Pose: A 4D Radar Tensor-based 3D Human Pose Estimation and
Localization Benchmark**, sequence 10, radar frame `000020`, synchronized
camera/pose frame `000056`, action “Walk”.

- Official dataset: <https://huggingface.co/datasets/uwipl/RT-Pose>
- Official processing code: <https://github.com/ipl-uw/RT-POSE>
- Dataset license:
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- Paper: <https://arxiv.org/abs/2407.13930>

Creators named by the upstream citation: Yuan-Hao Ho, Jen-Hao Cheng, Sheng Yao
Kuan, Zhongyu Jiang, Wenhao Chai, Hsiang-Wei Huang, Chih-Lung Lin, and Jenq-Neng
Hwang.

Contains adapted material from RT-Pose by those creators, licensed under CC
BY-NC-SA 4.0. HERMES selected and adapted the material as described below. No
endorsement is implied. Redistribution and adaptations must retain attribution,
identify changes, link or provide the license, remain NonCommercial, and use
the required ShareAlike terms.

## Changes

HERMES made these changes in July 2026:

- selected sequence 10 radar frame `000020`, synchronized camera/pose frame
  `000056`, and adjacent fitted-motion samples;
- decoded the calibrated AWR2243 cascade capture, converted it to `complex64`,
  separated physical TX slots into TX-major virtual-channel blocks, and
  reordered receivers within each block to physical `RX1` through `RX16`;
- retained the measured physical TDM TX schedule `TX12` through `TX1` in
  metadata;
- copied pose, translation, timing, coordinate, and producer-provenance fields
  into a pickle-free three-sample AMASS-like base-SMPL archive;
- replaced participant-specific SMPL betas and gender with zero betas and
  neutral gender, and removed subject ID and fitted root-joint-offset fields;
- retained base-SMPL triangle indices but omitted SMPL model files, blendshape
  bases, baked human meshes, skeletons, LiDAR point clouds, and source spatial
  matrices;
- irreversibly covered the visible face in each synchronized stereo image with
  an opaque mask and omitted the original images; and
- reduced fitted background geometry to one self-contained XML scene of simple
  cuboids, omitting the source point cloud, fitted OBJ/mesh, surface-detail JSON,
  and background-point preview.

The shareable animated-body contribution is attributed under the
[SMPL-Body CC BY 4.0 terms](https://smpl.is.tue.mpg.de/bodylicense.html): SMPL
was used for character animation courtesy of the Max Planck Institute for
Intelligent Systems. See Loper et al., “SMPL: A Skinned Multi-Person Linear
Model” (2015). No SMPL model or shape-blend basis is distributed.

## Scene materials

The scene XML selects Sionna RT's `itu-radio-material` implementation by the
common material names `wood`, `concrete`, and `chipboard`. Sionna supplies the
frequency-dependent radio parameters at runtime. The thickness, scattering,
and cross-polarization-discrimination coefficients in this XML are HERMES
modeling choices; the fixture does not contain an ITU recommendation, an
extracted standards table, or copied permittivity/conductivity coefficients.

## Privacy and other rights

The fixture remains derived from a human-participant dataset. Face masking and
body-shape neutralization reduce identifiability but do not turn the measured
radar or motion into synthetic data. CC BY-NC-SA 4.0 does not itself grant
privacy, publicity, trademark, patent, or other personality rights. Do not try
to identify the participant, imply participant or creator endorsement, or use
the fixture outside the rights granted by the upstream license and applicable
law.

The fixture is provided as-is and without warranties, subject to the disclaimer
and limitation of liability in CC BY-NC-SA 4.0.
