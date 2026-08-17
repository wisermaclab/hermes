# Source-checkout motion fixture

This directory contains one reviewed, byte-pinned motion fixture for
source-checkout examples. It is not installed as package data and is not
included in the HERMES wheel or Python source distribution (`sdist`). All other
motion datasets and all SMPL-family model files must be obtained separately
from their official providers.

The NPZ is third-party-derived data rather than project-owned HERMES data. It is
expressly excluded from both HERMES project licenses: neither the PolyForm code
license nor the CC BY-NC project-owned documentation/data license relicenses it
or replaces its upstream terms. This HERMES-authored README is project
documentation under CC BY-NC 4.0, but that license does not extend to the NPZ.

## `walking_poses_cmu_105_02.npz`

This AMASS-style base-SMPL parameter archive was converted from CMU Motion
Capture Database subject 105, trial 02. Its embedded provenance identifies the
CMU ASF, AMC, and C3D sources; the pose data comes directly from the ASF/AMC
joint rotations. For public distribution, the archive uses the neutral SMPL
model selector and ten zero `betas`; it does not distribute the fitted,
subject-specific SMPL shape coefficients or gender.
[SparseSMPLFit](https://github.com/wisermaclab/SparseSMPLFit) converts the CMU
Y-up motion to the Z-up AMASS world convention. This fixture retains all 1,748
source frames at 120 Hz and preserves their original frame identifiers. The GUI
uses a source-frame stride for a responsive 20 Hz preview without modifying the
motion archive or the frame indices used for simulation.
The `source_retarget_metrics` and `retarget_metrics` fields describe the same
1,748-frame conversion; `source_retarget_frame_count=1748` and
`output_frame_count=1748` make that scope explicit.
The archive does not contain marker trajectories, mesh topology, or an
SMPL-family model file.

The [CMU Motion Capture Database FAQ](https://mocap.cs.cmu.edu/faqs.php)
permits copying, modification, and redistribution of the motion-capture data
without permission. The [database homepage](https://mocap.cs.cmu.edu/) says
the data may be included in commercial products but may not be resold directly,
even in converted form, and requests this acknowledgement:

> The data used in this project was obtained from mocap.cs.cmu.edu. The
> database was created with funding from NSF EIA-0196217.

The neutral animation's SMPL-Body attribution and CC BY 4.0 notice are recorded
in [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md).

- Size: `807,203` bytes
- SHA-256: `a978f1b6dcd349e358cce922c767a07bf7bf86d044bcbf10a071945fe55d2c12`

The pickle-free archive embeds the source filenames, conversion settings, and
these source-file pins so the input provenance can be checked independently:

| CMU source file | Size | SHA-256 |
| --- | ---: | --- |
| `105.asf` | `7,274` bytes | `de06a1ee5d917e4bd23461e22e49b43591ed8d9bd84a92d6b6927f423a972b30` |
| `105_02.amc.txt` | `1,395,317` bytes | `e9fb90488a44dcd2d2480022e320bd75e5cc71edcb7dc48cf316aec10e6c9074` |
| `105_02.c3d` | `1,152,832` bytes | `cd9490c1408b99dd6838fe7bb3fe12c18d375372bbf81581ae61f3531fe6f206` |

The conversion used
[SparseSMPLFit](https://github.com/wisermaclab/SparseSMPLFit)'s direct
profile-based conversion path, followed by the public neutral-shape
substitution. The archive records SparseSMPLFit version `0.1.0`, the source
shape-estimation method separately from the released zero-shape selector, and
only source basenames rather than machine-local paths.

## Using the fixture

From a source checkout, select the CMU walking motion explicitly:

```bash
export MMWAVE_AMASS_NPZ="$PWD/data/AMASS/walking_poses_cmu_105_02.npz"
```

Mesh generation still requires separately obtained, appropriately licensed
SMPL-family model files. Put them under the ignored `models/smpl_models/`
directory, or set `MMWAVE_SMPL_MODEL_DIR` to another private location. To use
any other motion, point `MMWAVE_AMASS_NPZ` or `MMWAVE_DATASET_ROOT` at a
user-provided dataset outside the repository.
