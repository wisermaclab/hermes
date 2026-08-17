# mmRadPose fixture notice

This directory contains an adapted third-party data fixture, not project-owned
HERMES data. The fixture assets are expressly excluded from the HERMES PolyForm
code license and the CC BY-NC project-owned documentation/data license. This
HERMES-authored notice is project documentation under CC BY-NC 4.0, except for
reproduced third-party text, but that license does not extend to the assets it
describes.

## Source and license

Source: **mmRadPose**, version 1, trial `p1_an0_ac4_r0`, action 4 (“Bicep
curls”), frame `000240`.

- Official record: <https://zenodo.org/records/14738837>
- DOI: <https://doi.org/10.5281/zenodo.14738837>
- Dataset license: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- Paper DOI: <https://doi.org/10.1109/JMW.2025.3535525>

Creators: Lukas Engel, Jonas Müller, Eduardo Javier Feria Rendon, Eva Dorschky,
and Daniel Krauss.

Contains adapted material from mmRadPose by those creators, licensed under CC
BY-SA 4.0. HERMES selected and adapted the material as described below. No
endorsement is implied. Redistribution and adaptations must retain attribution,
identify changes, provide the license, and use the required ShareAlike terms.

## Changes

HERMES made these changes in July 2026:

- selected radar frame 240 and adjacent motion samples from trial
  `p1_an0_ac4_r0`;
- exported the measured IWR6843AOPEVM ADC as 12 TX-major virtual channels;
- applied the recorded inferred radar-to-SMPL coordinate alignment;
- fitted source skeleton motion separately with
  [SparseSMPLFit](https://github.com/wisermaclab/SparseSMPLFit);
- replaced fitted body shape and gender with neutral SMPL and zero betas;
- retained a three-sample AMASS-like window containing fitted poses,
  translations, timing, coordinate metadata, and base-SMPL topology; and
- omitted SMPL model files, baked body meshes, skeletons, and point clouds.

The shareable SMPL animated-body contribution is attributed under the
[SMPL-Body CC BY 4.0 terms](https://smpl.is.tue.mpg.de/bodylicense.html): SMPL
was used for character animation courtesy of the Max Planck Institute for
Intelligent Systems. See Loper et al., “SMPL: A Skinned Multi-Person Linear
Model” (2015). No SMPL model or blendshape basis is distributed; the included
AMASS-like archive contains only motion parameters, neutralized shape metadata,
timing, coordinate metadata, and triangle indices.

## Participant research

The associated paper reports 12 healthy participants, FAU ethics approval
`#22-437-B`, and written informed consent. The exact consent form is not part of
this fixture. The radar and fitted motion remain participant-derived; CC BY-SA
4.0 does not itself grant privacy, publicity, or other personality rights. This
fixture grants no endorsement or permission to identify a participant.
