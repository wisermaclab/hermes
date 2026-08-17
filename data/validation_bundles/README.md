# Limited real-radar validation fixtures

This directory contains two reviewed, byte-pinned source-checkout fixtures for
learning the HERMES measured-data layout:

- `rtpose_seq10_frame0020`: RT-Pose radar frame 20 with a neutralized
  AMASS-like motion window, face-occluded stereo setup images, and a simplified
  self-contained environment scene.
- `mmradarpose_p1_an0_ac4_r0_frame0240`: mmRadPose bicep-curl frame 240 with
  a neutralized AMASS-like motion window and no camera imagery.

Each fixture is a runnable bundle under the current validation contract. It
uses the same unified `bundle.json` profile as simulated exports and contains
one radar capture and a three-sample `amass_sequence.npz` window whose
`bundle_times` cover the physical radar acquisition. To avoid distributing a
participant-specific shape, the stored betas are zero and gender is neutral.
Neither fixture contains body-model files, baked body meshes, skeletons, point
clouds, or source spatial matrices. A separately licensed neutral SMPL model is
therefore required when validation must generate body meshes. These fixtures
are limited interoperability examples, not accuracy claims or complete dataset
distributions.

The fixture assets are third-party-derived data rather than project-owned
HERMES data. They are expressly excluded from both HERMES project licenses:
neither the PolyForm code license nor the CC BY-NC project-owned
documentation/data license applies to those assets. HERMES-authored README and
NOTICE prose is project documentation under CC BY-NC 4.0, except for reproduced
third-party text, but that license does not extend to the assets it describes.
The fixture-specific `NOTICE.md` files identify the controlling upstream terms,
attribution, modifications, and remaining privacy/publicity considerations.
The fixtures are excluded from the HERMES wheel and Python source distribution.
