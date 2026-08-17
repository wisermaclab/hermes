# HERMES

HERMES (Human-Environment Radar ModEling and Simulation) provides mmWave FMCW
radar simulation for moving mesh targets. The Python package is imported as
`mmWaveRadar` and uses Sionna RT, Mitsuba, and Dr.Jit as installed dependencies.

HERMES includes:

- FMCW timing, TI radar layouts, ADC synthesis, DSP, and visualization;
- moving SMPL-family mesh targets and AMASS-compatible motion;
- human-only physical optics (PO), full ray tracing (RT), and hybrid static-
  environment RT plus moving-human PO; and
- dataset-neutral comparison of simulated and measured radar products.

## Installation

Create an environment and install the published core package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install "hermes-radar-sim[notebook]"
```

For development from a cloned source checkout, install the editable project
and its tests instead:

```bash
python -m pip install -e ".[test,notebook]"
```

The distribution name is `hermes-radar-sim`; the import package remains
`mmWaveRadar` for API compatibility.

### Runtime platforms

The HERMES package itself contains Python code, but it is not advertised as
OS-independent because its simulation runtime depends on the pinned native
Mitsuba and Dr.Jit distributions. A platform can run the simulator only when
`pip` can install those dependencies for its Python and CPU architecture and
Mitsuba can load a suitable execution variant.

HERMES prefers `cuda_ad_mono_polarized` when a compatible NVIDIA CUDA/OptiX
runtime is available and otherwise uses the CPU
`llvm_ad_mono_polarized` variant included by the upstream Mitsuba package.
Public CI currently installs and tests HERMES on Ubuntu with Python 3.10 and
3.12. macOS and Windows are best-effort platforms until equivalent CI jobs are
added; consult the pinned Mitsuba and Dr.Jit release files before choosing a
Python version or machine architecture.

Point-target, radar-hardware, DSP, and precomputed-mesh workflows do not require
SMPL. For AMASS conversion or SMPL-family mesh generation, install the optional
dependencies:

```bash
python -m pip install "hermes-radar-sim[smpl]"
```

Obtain SMPL-family model files from the official
[SMPL](https://smpl.is.tue.mpg.de/) or
[SMPL-X](https://smpl-x.is.tue.mpg.de/) provider under its applicable license.
Neither the HERMES public license nor a separate commercial HERMES license
grants rights to the optional `smplx` software or SMPL-family model files.
Place them in the ignored source-checkout directory below, or set
`MMWAVE_SMPL_MODEL_DIR` to a private location:

```text
models/smpl_models/
  smpl/SMPL_FEMALE.pkl
  smpl/SMPL_MALE.pkl
  smpl/SMPL_NEUTRAL.pkl
```

Do not commit or redistribute these model files.

## Examples

The notebooks are source-checkout examples. Clone the public repository, enter
its root, and start with the data-free point-target notebook:

```bash
git clone https://github.com/wisermaclab/hermes.git
cd hermes
jupyter lab demo/Point-Targets.ipynb
```

The focused notebooks are:

- [Point Targets](https://github.com/wisermaclab/hermes/blob/main/demo/Point-Targets.ipynb): analytic ADC synthesis, DSP, and
  virtual-array angle maps;
- [Human-Only PO](https://github.com/wisermaclab/hermes/blob/main/demo/Human-Only-PO.ipynb): moving-human physical optics;
- [Full RT](https://github.com/wisermaclab/hermes/blob/main/demo/Full-RT.ipynb): retracing and coherent path-bank baselines;
- [Hybrid PO](https://github.com/wisermaclab/hermes/blob/main/demo/Hybrid-PO.ipynb): static-environment RT plus moving-human
  PO.

The human-motion notebooks require a separately licensed SMPL model. See the
[demo guide](https://github.com/wisermaclab/hermes/blob/main/demo/README.md) for
their inputs and expected runtimes.

## Interactive GUI

Install the optional local web interface:

```bash
python -m pip install "hermes-radar-sim[gui]"
hermes-gui
```

Dynamic Scenes additionally needs the SMPL adapters:

```bash
python -m pip install "hermes-radar-sim[gui,smpl]"
```

The GUI has three top-level workflows: **Static Target**, **Dynamic Scenes**,
and **Bundle Comparison**. The header Settings dialog holds the TI
hardware, active Tx/Rx antenna elements, FMCW waveform, antenna pattern, and
solver budgets shared by subsequent runs. The radar phase center is fixed at
`(0, 0, 0)`; all scene and target positions are expressed relative to it.
Static and dynamic runs keep separate radar yaw, pitch, and roll controls.

- **Static Target** runs both RT and PO for a plate, trihedral corner
  reflector, or uploaded human `.obj`/`.npz` mesh. Cartesian target position,
  geometry, orientation, material, and diffuse-reflection controls recompute
  the absolute-power RT/PO range and angle views. The header settings menu
  controls adaptive parent-facet PO integration plus RT ray, path, and depth
  budgets, and saves them in browser-local storage. Its local radar orientation
  rotates the fixed-origin sensor used by both solvers. Settings selects a
  digitized TI profile, omnidirectional response, or cosine response with a
  user-specified full 3 dB beamwidth. It can also simulate any nonempty subset
  of the selected board's Tx and Rx elements. TDM is off by default, so every
  selected Tx/Rx channel is sampled simultaneously once per chirp repetition;
  enabling TDM evaluates each transmitter in its physical chirp slot and uses
  the full Tx cycle as the per-channel slow-time interval. Plain mouse drag
  orbits the camera; Shift-drag previews physical
  target yaw/pitch and triggers one RT/PO run on release. Closed human meshes
  are normalized to outward face winding at import; their scene view shades
  PO power and highlights RT-hit faces plus RT/PO overlap. An on-demand
  diagnostics view overlays the strongest
  target-touching RT paths and strongest PO facets, with RT interaction-depth
  plots and compact path/facet counts.
- **Dynamic Scenes** uses either the prepared bedroom or an uploaded
  Sionna/Mitsuba scene XML. It accepts pickle-free AMASS-like motion parameters
  (`poses`, `trans`, `betas`, and `times`, `bundle_times`, or
  `mocap_framerate`) and evaluates them with a separately licensed SMPL-family
  model. The scene preview provides synchronized frame-slider and playback
  controls; playback updates geometry only and does not rerun the solver.
  High-rate motion is displayed at a source-dividing rate near 20--25 Hz, so
  playback follows real motion time without flooding the browser. Orbit and
  zoom gestures remain usable during playback; every mesh update carries the
  live Plotly camera, while redraws are coalesced during each gesture and catch
  up to the newest frame afterward. The
  simulation controls select any combination of full per-chirp RT, coherent RT,
  human-only PO, and Hybrid PO over an inclusive begin/end motion-frame window
  bounded by the loaded sequence. A newly loaded motion defaults to a one-frame
  window, and the end-frame label shows the maximum available source index.
  Four square checkboxes select the modes; each has a question-mark tooltip
  describing its solver policy.
  **Start** launches them and **Stop** cancels at the next safe RT/PO solver
  boundary. While active, the status reports only the current mode and frame.
  Results are grouped by product rather than by mode: one range-
  profile comparison overlays all completed modes, while the range-time and
  frame-selectable range-Doppler pages show every selected mode together. The
  result-frame selector is one-based (`1` through the simulated frame count)
  and has explicit Previous/Next frame actions for short runs.
  Switching product tabs restores those plots' default axes. Hybrid PO always
  includes human blocking and human/environment coupling, and its PO and
  coupling components are placed on the coherent-RT power scale by the
  sequence calibration stage. The
  controls also expose human placement, yaw, and diffuse coefficient.
  The prepared bedroom uses a tightly bounded, open-front shell with furniture
  outside the included CMU walking trial and modest rough-wall scattering for
  visible room returns. After a run, the scene page can retrace
  the displayed simulated frame on demand: Full or Coherent RT enables the top-
  K target-touching ray overlay, while Human-only or Hybrid PO colors the human
  facets by PO power. These compact overlays do not retain Sionna's full path
  object during normal simulations.
  Arbitrary uploaded XML is used by
  the solver but is not rebuilt in
  the lightweight Plotly preview; a browser-uploaded XML should be
  self-contained because referenced assets are not uploaded with it.
  Component range profiles distinguish direct human PO, blocked static-
  environment RT, and the two coupling directions. Public installs include the
  prepared room and a redistributable AMASS-like example, but no licensed SMPL
  model. This workflow has its own radar-orientation controls and uses the
  global hardware, antenna, and waveform Settings.
- **Bundle Comparison** loads and validates a
  [HERMES Bundle v1](https://github.com/wisermaclab/hermes/blob/main/validation/BUNDLE_SPEC.md) directory or GUI-exported ZIP,
  displays its primary scene, ADC, range profile, range-time, and range-Doppler
  products in product-specific tabs, and either runs a bundle-configured human
  RT/PO simulation or loads previously saved ADC from the bundle, an external
  archive, or another HERMES bundle. A fresh run may select several simulation
  modes, plus an optional channel-0-only path that simulates the corresponding
  physical TX/RX pair and compares it with primary channel 0. Candidate ADC,
  range-time, and range-Doppler plots appear beside their primary counterparts;
  every primary and candidate range profile is overlaid in one shared plot.
  The scene remains primary-only, and object hover labels include the referenced
  material. Parameter-only
  AMASS bundles evaluate the human with the configured licensed SMPL model and
  display it at the primary bundle frame time; bundles with evaluated motion
  use that mesh directly. A fresh run can be stopped after confirmation at the
  next safe solver boundary. The compact download action beside the run button
  exports one completed raw ADC as NPZ or several mode results together as a
  ZIP of NPZ archives. Inputs use one combined drag-and-drop/click-to-browse
  control; folders are supported when dropped. Theme changes preserve the
  selected static-target controls and mesh upload; shared radar settings;
  dynamic-scene controls, motion and XML uploads, displayed frame, selected
  result product, and completed static/dynamic solver products. They also
  preserve the selected bundle path, processing controls, comparison source
  and modes, channel-0 setting, external ADC path, active product tab, and
  automatically reload filesystem-backed bundles and saved comparisons.

GUI exports and measurement fixtures use the same public `hermes` profile.
Each ADC artifact declares whether it is a measurement, simulation, background,
or processed cube. Exports include primary raw ADC, separate RT and PO
comparison cubes, target geometry, scene XML and its referenced OBJ,
sensor/frame metadata, diagnostics, and reproducible experiment manifests.
Static Target bundles store separate RT and PO cubes.
Dynamic Scenes bundles store the preferred selected mode as primary data and
retain every selected mode as `simulated_adc_<mode>.npz`; HERMES RT/PO aliases
remain available for the comparison screen. They also store the evaluated
motion as `human_motion.npz`. AMASS-driven exports retain
the source parameters as `amass_sequence.npz`; the inclusive source-frame
window and per-frame source indices are recorded in bundle metadata. Licensed
model files are never embedded. Uploaded XML is preserved as `scene.xml`. Both
workflows store the
selected zero-based Tx/Rx board-element indices in `sensor.json`, allowing the
same loader to reconstruct the virtual array regardless of ADC origin.
The validation runner's `simulated_adc_<label>.npz` outputs remain compatible
as external comparison inputs.

The rough PEC RT preset uses a diffuse scattering coefficient and
backscattering-pattern mixture parameter of `0.20`; the smooth aluminum preset
uses `0.0` for both, and concrete uses the heuristic value `0.35`. The
human-mesh coefficient is user controlled and defaults to `0.35`. The
diffuse-energy coefficient and backscattering-pattern mixture parameter are
stored separately. These are RT rough-surface terms; PO uses the selected
conductivity/permittivity material model without them.

Sionna RT 2.0.1 internally discards propagation paths containing segments
shorter than 1 cm. This safeguard improves numerical robustness by rejecting
near-degenerate paths. In trihedral corner-reflector simulations, however,
physically valid triple-bounce paths can contain face-to-face segments below
this threshold, particularly with millimeter-scale MIMO antenna spacing.
Consequently, Sionna RT may report no specular return from the reflector even
with sufficient samples and `max_depth >= 3`. The PO calculation is unaffected.
HERMES retains Sionna's upstream behavior rather than overriding the internal
threshold.

The server binds to `127.0.0.1` by default. The official launcher prints a
random per-launch password and requires it before Bokeh constructs a GUI
session. Its signed authentication cookie is `HttpOnly` and `SameSite=Strict`;
cross-site document/login requests fail closed before session construction,
and GUI responses deny framing. This protects a loopback server from hostile
web pages that try to create expensive sessions, but it is not TLS or a
multi-user identity system.

Normal operation is loopback-only. The launcher rejects non-loopback
`--address` values unless an operator deliberately enables
`--unsafe-allow-remote`. In that mode HERMES prints an additional warning,
locks browser-controlled server paths, uses only the server-configured
`MMWAVE_SMPL_MODEL_DIR`, and disables custom scene XML. Use the override only
on a trusted, access-controlled network; do not expose it directly to the
public internet or transmit its password over an untrusted network.
Wildcard binds additionally require an explicit browser-origin allowlist, for
example:

```bash
hermes-gui --address 0.0.0.0 --port 5006 --no-browser \
  --unsafe-allow-remote --allow-websocket-origin radar-lab.example:5006
```

Repeat `--allow-websocket-origin HOST[:PORT]` when several concrete origins
are needed; `*`, URL schemes, paths, and wildcard addresses are rejected. If
you embed Panel directly, or place HERMES behind a reverse proxy or tunnel,
call `build_app(remote_access=True)`. The official CLI infers this restriction
from its bind address, but direct application embedding cannot infer whether a
browser is remote. Direct Panel embedding must also supply authentication and
equivalent cross-site/framing protections; those launcher-layer controls are
not installed by `build_app` itself.

Browser uploads are bounded: individual dropped bundle/ADC files are limited
to 256 MiB, a dropped group to 512 MiB and 1,024 files, static meshes and
AMASS-like motion to 64 MiB each, and scene XML to 2 MiB. Check dependencies
and source-checkout fixture availability with:

```bash
hermes-gui --self-check
```

SMPL model files are required only when Dynamic Scenes evaluates AMASS-like
motion. Configure the licensed model directory in that tab or through
`MMWAVE_SMPL_MODEL_DIR`. Static Target mesh simulation and measurement
comparison do not require it. The public release does not contain a licensed
body model. Dynamic exports contain the source AMASS-like parameters and the
evaluated mesh sequence used in the result; redistribution rights remain the
exporter's responsibility.

## Data and external models

The source checkout contains three narrowly reviewed examples:

- one [CMU-derived walking fixture](https://github.com/wisermaclab/hermes/blob/main/data/AMASS/README.md);
- one limited bundle derived from [RT-Pose][rt-pose]; and
- one limited bundle derived from [mmRadPose][mmradpose].

The two radar bundles are documented in the
[validation-fixture guide](https://github.com/wisermaclab/hermes/blob/main/data/validation_bundles/README.md). Their motion
windows use neutral gender and zero betas; they contain no body-model files or
baked human meshes. The [RT-Pose][rt-pose] setup images use opaque face masks.
These fixtures are excluded from the wheel and Python source distribution
(`sdist`).

All three examples retain their recorded upstream terms and are explicitly
excluded from the HERMES project-license grants. See
[Third-Party Notices](https://github.com/wisermaclab/hermes/blob/main/THIRD_PARTY_NOTICES.md) for provenance, exact boundaries,
and optional external inputs.

For other workflows, supply authorized local paths:

- `MMWAVE_AMASS_NPZ` or `MMWAVE_DATASET_ROOT` for motion;
- `MMWAVE_SMPL_MODEL_DIR` for SMPL-family model files; and
- `MMWAVE_TI_PATTERN_NPZ` for an optional digitized TI antenna-pattern archive.

The GUI's built-in deterministic cosine antenna model requires no external
pattern data and defaults to a 60° full 3 dB beamwidth (30° off-boresight
half-power angle). Selecting the digitized-profile option requires the optional
pattern archive.

## Real-radar validation

HERMES separates dataset-specific bundle preparation from scientific
validation. A completed bundle contains measured ADC, radar configuration,
frame timing, and parameterized body motion; it may also reference a static
environment scene. The [validation guide](https://github.com/wisermaclab/hermes/blob/main/validation/README.md) defines the
file contract, simulation modes, channel selection, TDM timing, metrics, and
outputs.

Run validation with an explicit output directory:

```bash
python -m validation.cli \
  --clip /path/to/clip_bundle \
  --suite dsp_maps \
  --out /tmp/hermes_validation \
  --progress
```

Without `--simulated-adc`, validation runs the selected simulation modes. The
main choices are `human_only_po`, `rt_retrace`, `rt_coherent_bank`, and
`hybrid_static_env_po`:

```bash
python -m validation.cli \
  --clip /path/to/clip_bundle \
  --out /tmp/hermes_validation \
  --mobility-mode human_only_po hybrid_static_env_po
```

TX/RX selectors use 1-based physical antenna labels. By default, TDM
simulation evaluates each selected TX at its physical slot time. Add
`--ignore-tdm-timing` to evaluate the selected array jointly at the TDM-cycle
start while retaining the full-cycle slow-time interval. See
[TDM timing policies](https://github.com/wisermaclab/hermes/blob/main/validation/README.md#tdm-timing-policies) for the exact
PO/RT, phase, channel-ordering, and cache behavior.

Check bundle structure without running a simulation:

```bash
mmwave-check-bundle /path/to/clip_bundle
```

### Reference adapters

The public source checkout includes a reviewed
[RT-Pose adapter](https://github.com/wisermaclab/hermes/blob/main/tools/bundle_prepare/rtpose/prepare_rtpose_bundle.py) and
[mmRadPose adapter](https://github.com/wisermaclab/hermes/blob/main/tools/bundle_prepare/mmradarpose/prepare_mmradarpose_bundle.py).
They demonstrate the same universal contract with different radar layouts and
synchronization metadata; validation is not limited to those datasets.

The [RT-Pose][rt-pose] adapter reads the AWR2243 cascade capture and can prepare
background geometry from its synchronized LiDAR/stereo inputs. The
[mmRadPose][mmradpose] adapter exports the IWR6843AOPEVM radar layout and uses
laser-only environment input when available. Neither adapter downloads a
dataset.

Both adapters consume a pickle-free AMASS-like base-SMPL motion archive.
[SparseSMPLFit](https://github.com/wisermaclab/SparseSMPLFit) is one optional
producer; it is a separate project and is not bundled or invoked by HERMES.

Start with the [bundle-preparation guide](https://github.com/wisermaclab/hermes/blob/main/tools/bundle_prepare/README.md) or the
adapter help:

```bash
python tools/bundle_prepare/rtpose/prepare_rtpose_bundle.py --help
python tools/bundle_prepare/mmradarpose/prepare_mmradarpose_bundle.py --help
```

## TI board presets

Built-in presets cover MMWCAS-RF-EVM/AWR2243 cascade, AWRL6844EVM,
IWR6843ISK, and IWR6843AOPEVM:

```python
from mmWaveRadar.radar import FMCWConfig, RadarSensor

fmcw = FMCWConfig(
    carrier_frequency=60e9,
    slope=68e12,
    chirp_duration=58e-6,
    chirp_repetition_time=65e-6,
    sampling_frequency=4.5e6,
    num_adc_samples=225,
    num_chirps_per_frame=96,
    frame_period=50e-3,
)
radar = RadarSensor.from_ti_board("IWR6843AOPEVM", fmcw=fmcw)
```

The default `pattern_mode="cosine30"` is data-free. Use `"none"` for scalar
gain and phase signs only, or `"digitized"` with an authorized archive selected
through `MMWAVE_TI_PATTERN_NPZ`.

## License

- Project-owned code: [PolyForm Noncommercial 1.0.0](https://github.com/wisermaclab/hermes/blob/main/LICENSE).
- Project-owned documentation and data:
  [CC BY-NC 4.0](https://github.com/wisermaclab/hermes/blob/main/LICENSE-DOCS-DATA).
- Bundled third-party material: its recorded upstream terms, as listed in
  [Third-Party Notices](https://github.com/wisermaclab/hermes/blob/main/THIRD_PARTY_NOTICES.md).

The governance-only bootstrap commit `0b3694bb82dbf9a220c3942d032901918ca3fea7`
was originally published under GPL-3.0-only, and that historical grant remains
valid for copies of that commit. Starting with the first simulator release,
project-owned governance code and configuration are also offered under
PolyForm Noncommercial 1.0.0, while project-owned governance documentation is
also offered under CC BY-NC 4.0, as marked by SPDX headers. See [NOTICE](NOTICE)
for the transition record.

For a separate commercial license, contact **Rong Zheng** at
**rzheng@mcmaster.ca**. See [Commercial Licensing](https://github.com/wisermaclab/hermes/blob/main/COMMERCIAL-LICENSING.md).

## Development

```bash
python -m pip install -e ".[dev,notebook,smpl]"
python -m compileall -q src/mmWaveRadar validation tools benchmarks
python -m pytest tests/unit
```

See [Contributing](https://github.com/wisermaclab/hermes/blob/main/CONTRIBUTING.md),
the [benchmark guide](https://github.com/wisermaclab/hermes/blob/main/benchmarks/README.md),
and the [developer API](https://github.com/wisermaclab/hermes/blob/main/docs/api/index.html)
for more detail.

## Layout

```text
src/mmWaveRadar/        # package source
validation/             # dataset-neutral validation
tests/unit/             # focused unit tests
benchmarks/             # performance and comparison scripts
tools/bundle_prepare/   # validation-bundle adapters
demo/                   # focused notebooks
docs/api/               # generated API reference
```

[rt-pose]: https://huggingface.co/datasets/uwipl/RT-Pose
[mmradpose]: https://zenodo.org/records/14738837
