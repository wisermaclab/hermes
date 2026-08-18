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
3.12. HERMES has also been tested manually on macOS with Apple M4 processors.
macOS remains manually tested rather than CI-enforced, and Windows remains a
best-effort platform; consult the pinned Mitsuba and Dr.Jit release files before
choosing a Python version or machine architecture.

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

The GUI provides three workflows:

- **Static Target** compares RT and PO for plates, corner reflectors, and
  uploaded meshes, with range, angle, and path/facet diagnostics.
- **Dynamic Scenes** runs full or coherent RT, human-only PO, and Hybrid PO for
  a prepared or uploaded room with AMASS-compatible motion.
- **Bundle Comparison** validates a
  [HERMES Bundle v1](https://github.com/wisermaclab/hermes/blob/main/validation/BUNDLE_SPEC.md),
  explores its ADC and DSP products, and compares them with fresh or saved
  simulations.

The Settings dialog configures TI hardware, active Tx/Rx elements, the FMCW
waveform, antenna response, and solver budgets. The radar phase center remains
at `(0, 0, 0)`; scene and target positions are relative to it.

GUI exports are portable HERMES Bundle v1 archives containing the ADC products,
sensor and frame metadata, scene inputs, diagnostics, and reproducible
experiment manifests. Dynamic exports retain each selected simulation mode;
licensed model files are never embedded.

The launcher binds to `127.0.0.1` by default and protects each launch with a
random password. Remote binding requires the explicit `--unsafe-allow-remote`
override and an origin allowlist. Use it only on a trusted, access-controlled
network: the launcher is not a TLS endpoint or multi-user identity system.

Check optional dependencies and source-checkout fixtures with:

```bash
hermes-gui --self-check
```

Dynamic Scenes requires an authorized SMPL-family model configured in the GUI
or through `MMWAVE_SMPL_MODEL_DIR`. Static Target and measurement-only bundle
comparison do not. The public release contains no licensed body model.

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
