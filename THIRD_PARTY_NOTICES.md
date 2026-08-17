# Third-Party Notices

This file describes third-party software and optional external inputs associated
with the public HERMES release. It is informational and does not change the
license terms of HERMES or any third-party material.

## Public Release Boundary

The public Git source checkout contains project source, documentation,
output-free examples, one reviewed motion fixture, and two limited real-radar
validation fixtures described below. The Python source distribution (`sdist`)
and wheel omit all three data examples. Apart from these exact byte-pinned
source-checkout exceptions, the public release does not contain:

- standalone SMPL, SMPL-H, SMPL-X, or DMPL model files such as templates,
  blend weights, and learned bases;
- other real-radar, camera, point-cloud, or participant-derived captures;
- vendor or standards-body PDFs, extracted figures, or material-property tables;
- digitized TI antenna-pattern archives; or
- other generated meshes, simulation results, checkpoints, or caches.

Those materials must not be inferred to be covered by either HERMES project
license. Users must obtain any optional input from its official provider and
comply with the provider's terms.

### Explicit exclusions from the HERMES project licenses

The following bundled material is expressly excluded from the PolyForm
Noncommercial 1.0.0 code grant and the CC BY-NC 4.0 project-owned
documentation/data grant:

- `data/AMASS/walking_poses_cmu_105_02.npz`, which remains subject to the CMU
  source-motion terms and the recorded SMPL-Body attribution;
- the third-party and adapted data assets below
  `data/validation_bundles/rtpose_seq10_frame0020/`, which remain subject to the
  recorded RT-Pose CC BY-NC-SA 4.0 terms, SMPL-Body attribution, and other
  rights described by its `NOTICE.md`;
- the third-party and adapted data assets below
  `data/validation_bundles/mmradarpose_p1_an0_ac4_r0_frame0240/`, which remain
  subject to the recorded mmRadPose CC BY-SA 4.0 terms, SMPL-Body attribution,
  and other rights described by its `NOTICE.md`; and
- the Texas Instruments-derived portions of
  `tools/bundle_prepare/rtpose/rtpose_adc.py`, which remain under the retained
  TI redistribution terms in that file and below. The separable HERMES-authored
  portions of that file remain under PolyForm Noncommercial 1.0.0.

Installed dependencies are not bundled and remain under their respective
upstream licenses. The project licenses also do not cover optional external
inputs, locally installed models, or outputs whose rights derive from those
inputs. Where more than one set of terms applies, users must comply with all of
them; a HERMES license does not replace or broaden an upstream grant.
HERMES-authored README and NOTICE prose remains project documentation under CC
BY-NC 4.0, except for any reproduced third-party text. Licensing that prose
does not relicense the assets it describes.

## Included Source-checkout Motion Fixture

The file below is included in the Git source checkout for examples and format
validation. It is not installed as package data and does not appear in the
HERMES wheel or Python `sdist`. Exact byte-level pins prevent an unnoticed
replacement from passing the public-release audit. See
[`data/AMASS/README.md`](data/AMASS/README.md) for its detailed contents and
usage.

| File | Provenance and terms | Size and SHA-256 |
| --- | --- | --- |
| `data/AMASS/walking_poses_cmu_105_02.npz` | AMASS-style base-SMPL parameters converted from CMU Motion Capture Database subject 105, trial 02. It contains all 1,748 derived pose and translation frames at 120 Hz, neutral/zero shape fields for format compatibility, and no SMPL model file. Subject-specific SMPL shape coefficients and gender are not distributed. The CMU database terms apply to its source motion. | `807,203` bytes; `a978f1b6dcd349e358cce922c767a07bf7bf86d044bcbf10a071945fe55d2c12` |

The [CMU Motion Capture Database FAQ](https://mocap.cs.cmu.edu/faqs.php)
permits copying, modification, and redistribution of its motion-capture data
without permission. The [database homepage](https://mocap.cs.cmu.edu/) says
the data may be included in commercial products but may not be resold directly,
even in converted form, and requests the following acknowledgement:

> The data used in this project was obtained from mocap.cs.cmu.edu. The
> database was created with funding from NSF EIA-0196217.

The [SMPL-Body license](https://smpl.is.tue.mpg.de/bodylicense.html) covers the
shareable body subset, including animated bodies and skeleton rigs, under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), while excluding the
shape blendshapes and tools distributed under the separate SMPL model license.
This fixture keeps only neutral base-SMPL pose/translation parameters and zero
shape fields. SMPL-Body attribution: SMPL was used for character animation
courtesy of the Max Planck Institute for Intelligent Systems; see Loper et al.,
“SMPL: A Skinned Multi-Person Linear Model” (2015). The fixture is a modified,
CMU-derived animation in the Z-up AMASS-style coordinate convention.

## Python Dependencies

The packages below are declared in `pyproject.toml`. They are installed from
upstream package indexes and are not vendored. Their own licenses apply to the
versions users install.

| Dependency | Use | Distribution status |
| --- | --- | --- |
| `sionna-rt` | Ray-tracing backend for RT and hybrid simulations. | Not vendored. |
| `mitsuba` | Rendering and ray-tracing runtime used by Sionna RT workflows. | Not vendored. |
| `drjit` | Array/JIT runtime used by Mitsuba and Sionna RT. | Not vendored. |
| `numpy` | Numerical arrays and file formats. | Not vendored. |
| `scipy` | Signal-processing and numerical helpers. | Not vendored. |
| `matplotlib` | Plotting and diagnostic figures. | Not vendored. |
| `Pillow` | Preview-image generation for optional environment fitting. | Not vendored. |
| `typing-extensions` | Python typing backports. | Not vendored. |
| `jsonschema` | Draft 2020-12 validation of the normative HERMES Bundle v1 contract. | Not vendored. |
| `torch`, `smplx` | Optional AMASS/SMPL conversion support. | Optional; model data is not included. |
| `jupyter`, `ipywidgets`, `pythreejs` | Optional notebook workflows. | Optional, not vendored. |
| `panel`, `plotly` | Optional local GUI framework and interactive plotting. | Optional, not vendored. |
| `pytest`, `pylint` | Test and development tools. | Development only, not vendored. |

## Optional Motion And Body-Model Inputs

HERMES can operate on user-provided AMASS-compatible motion archives and
SMPL-family models. AMASS aggregates sequences from multiple source datasets, so
the applicable terms may depend on the selected sequence as well as AMASS.
Obtain AMASS data through the [official AMASS
site](https://amass.is.tue.mpg.de/) and body models through the official
[SMPL](https://smpl.is.tue.mpg.de/) or
[SMPL-X](https://smpl-x.is.tue.mpg.de/) site. Never redistribute standalone
body-model files or unreviewed restricted sequences through HERMES.
Licensed users may store their own model files under the Git-ignored
`models/smpl_models/` directory; that local convention does not make the files
part of the repository or a HERMES distribution.

The included CMU conversion is the only standalone motion-data exception. The
two limited validation bundles described below each contain their own reviewed,
neutralized three-sample motion window. Every other AMASS or AMASS-compatible
sequence remains an external, user-provided input obtained under its applicable
upstream terms.

## Included Limited Real-Radar Fixtures

Two deliberately limited, byte-pinned validation bundles are included under
[`data/validation_bundles/`](data/validation_bundles/). They demonstrate the
measured-ADC bundle API and are excluded from the wheel and Python `sdist`.
Each contains a pickle-free, three-sample base-SMPL motion window with zero
betas and neutral gender. Neither contains SMPL model files, fitted participant
shape or gender, baked human meshes, skeletons, point clouds, or source spatial
matrices. A separately licensed neutral SMPL model is required to generate the
human mesh at simulation time.

The RT-Pose fixture contains one AWR2243 radar frame plus synchronized stereo
setup images with the participant's face irreversibly covered by an opaque
mask. It also contains one simplified static-scene XML derived from upstream
background geometry; the raw point cloud and fitter intermediates are omitted.
It is adapted under CC BY-NC-SA 4.0. The mmRadPose fixture contains one
IWR6843AOPEVM radar frame and no image; it is adapted under CC BY-SA 4.0.
The fixture-specific `NOTICE.md` files provide creator attribution, exact
source records, modification notices, ShareAlike/NonCommercial obligations
where applicable, SMPL-Body attribution, and privacy/publicity cautions.
The HERMES project licenses do not replace those terms.

## Optional Real-Radar Inputs

The standard validation-bundle schema and source-checkout adapters are project
code. No full dataset or
[SparseSMPLFit](https://github.com/wisermaclab/SparseSMPLFit) checkout is
included. The RT-Pose
adapter includes project code that reads the dataset's measured AWR2243
cascade files and applies the dataset-provided calibration; the capture and
calibration files remain external inputs. Its environment stage may also read
the selected RT-Pose LiDAR and stereo-camera frames to reconstruct a temporary
background cloud; it copies fitted geometry rather than those source frames
into the completed bundle. Beyond the two limited fixtures above, the adapters
operate on separately obtained data such as
[RT-Pose](https://huggingface.co/datasets/uwipl/RT-Pose) and
[mmRadPose](https://zenodo.org/records/14738837). Users must review
the terms attached to the exact dataset version they obtain. Input data and
resulting radar, skeleton, point-cloud, image, fitted-body, and mesh artifacts
remain subject to the applicable license, attribution, privacy/consent, and
body-model terms; HERMES does not grant redistribution rights for them.

RT-Pose bundle preparation consumes a pickle-free AMASS-like base-SMPL
sequence from any compatible producer. The archive supplies pose parameters,
pose frame IDs, and pose timing; HERMES reads `Train.json` to obtain
`Radar_frameID` values and aligns those poses with RT-Pose's 10 Hz radar
timeline. The archive need not contain radar IDs or radar timestamps.
mmRadPose preparation uses the same pose-only contract. HERMES selects the
matching trial frame, applies the dataset-local radar-to-SMPL alignment,
and converts the parsed source radar matrix to the standard bundle channel
layout. Its optional environment stage uses only fitted geometry or a laser
cloud already registered to the radar frame; it does not read camera data or
copy the raw laser cloud into the bundle.
[SparseSMPLFit](https://github.com/wisermaclab/SparseSMPLFit) can optionally
produce compatible motion archives, but HERMES neither bundles nor invokes it.
These operations do not grant redistribution rights for the inputs or outputs.

## TI Board Metadata And Optional Patterns

HERMES contains project code implementing TI board presets from publicly
available product information. Texas Instruments source PDFs and figures are not
included. TI names and marks belong to their respective owner; their mention
does not imply endorsement.

The built-in `cosine30` antenna model is synthetic and is the default.
Digitized patterns are an optional user-side input selected explicitly with
`pattern_mode="digitized"` and `MMWAVE_TI_PATTERN_NPZ`. HERMES does not
provide or license such an archive.

## TI Cascade Calibration Software

`tools/bundle_prepare/rtpose/rtpose_adc.py` includes Python adaptations of
Texas Instruments' 2018 mmWave cascade ADC decoding and calibration logic,
obtained through the official RT-Pose processing-code distribution. The source
file retains this notice and the following terms:

Copyright (C) 2018 Texas Instruments Incorporated - http://www.ti.com/

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
- Neither the name of Texas Instruments Incorporated nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Project Licenses

Project-owned HERMES code is licensed under PolyForm Noncommercial 1.0.0; see
`LICENSE`. Project-owned documentation and data are licensed under CC BY-NC
4.0; see `LICENSE-DOCS-DATA`. `NOTICE` contains the required PolyForm notice,
and `COMMERCIAL-LICENSING.md` gives the contact for separate commercial
licensing. These grants apply only to material for which the HERMES rights
holder can grant the relevant rights. A user's obligations for excluded
third-party material, external inputs, and installed dependencies remain
separate.
