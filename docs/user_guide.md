# HERMES user guide

Detailed runtime, solver, GUI, and measurement reference. Start with the
[simulation tutorial](https://wisermaclab.github.io/hermes/docs/tutorial/)
for guided workflows or the [README](../README.md) for installation.
Commands below run from the repository root.

HERMES (Human-Environment Radar ModEling and Simulation) provides mmWave FMCW
radar simulation for moving mesh targets. The Python package is imported as
`mmWaveRadar` and uses Sionna RT, Mitsuba, and Dr.Jit as installed dependencies.

HERMES includes:

- FMCW timing, TI radar layouts, ADC synthesis, DSP, and visualization;
- moving SMPL-family mesh targets and AMASS-compatible motion;
- three solvers: physical optics on the human (`po`), ray tracing of the
  whole scene (`rt`), and static-environment RT plus moving-human PO
  (`hybrid_po`); and
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
With PyTorch installed, `compute_backend="auto"` also accelerates ADC synthesis,
including wideband PO and Hybrid PO, on an available CUDA device or supported
Apple MPS device. It falls back to NumPy without a supported accelerator;
`compute_backend="numpy"` explicitly selects the CPU for these operations.

ADC synthesis, physical optics, and coherent RT updates share one backend
policy in `simulation/compute.py`:

| `compute_backend` | Array computation |
| --- | --- |
| `auto` | PyTorch on CUDA, then capability-checked MPS; otherwise NumPy on CPU |
| `numpy` | NumPy on CPU, without loading PyTorch |
| `torch` | PyTorch on a supported accelerator, otherwise PyTorch on CPU; requires PyTorch |

`MMWAVE_TORCH_DEVICE=cpu`, `cuda:0`, or `mps` overrides device selection for
these kernels. An unavailable device or unsupported precision raises a clear
error; `numpy` ignores this override. MPS requires float32 and supported complex
operations. Coherent RT updates follow `compute_precision`, including float32
for MPS; float64 remains available for large-coordinate scenes or sensitive
phase comparisons. `adc_compute_backend` and
`adc_compute_precision` inherit the simulator's settings when omitted, as do
the corresponding `POConfig` fields. The scene CLI/GUI use `auto`; low-level
Python APIs retain their NumPy defaults. Runtime summaries report the resolved
`auto` backend/device for each precision.

Mitsuba ray tracing uses its own process-wide variant, independently of these
array settings. Set `MMWAVE_MITSUBA_VARIANT=llvm_ad_mono_polarized` before
initializing Mitsuba to force CPU ray tracing. SMPL preprocessing accepts a
separate device argument; `select_torch_device()` uses the same device preference
and environment override when a preprocessing caller requests automatic selection.

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

Ensure that jupyter is running within the virtual environment

```bash
$ source .venv/bin/activate
$ realpath --relative-to . `which pip`  # check that the right `pip` is in use
.venv/bin/pip
$ pip install jupyter
$ pip install jupyter_collaboration  # needed due to a bug in jupyter as of this writing
$ source .venv/bin/activate  # reactivate the environment for jupyter
$ realpath --relative-to . `which jupyter`  # check that the right `jupyter` is in use
.venv/bin/jupyter
```

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
- [PO](https://github.com/wisermaclab/hermes/blob/main/demo/PO.ipynb): the PO solver on a moving human;
- [RT](https://github.com/wisermaclab/hermes/blob/main/demo/RT.ipynb): the RT solver, retracing every chirp versus
  once per frame;
- [Hybrid PO](https://github.com/wisermaclab/hermes/blob/main/demo/Hybrid-PO.ipynb): static-environment RT plus moving-human
  PO.

The human-motion notebooks require a separately licensed SMPL model. See the
[demo guide](https://github.com/wisermaclab/hermes/blob/main/demo/README.md) for
their inputs and expected runtimes.

## Python experiment API

Start new Python integrations with the grouped settings APIs. For an evaluated
human mesh in the empty room:

```python
from dataclasses import replace
from mmWaveRadar.experiments import (
    HumanMotionSettings, HumanRoomSettings, HumanRoomSolverSettings,
    RoomSceneSettings, simulate_human_room,
)

settings = HumanRoomSettings(
    motion=HumanMotionSettings(
        human_motion="/path/to/human_motion.npz",
        human_position_m=None,  # preserve motion-file coordinates
        num_frames=2,
    ),
    scene=RoomSceneSettings(room_preset="empty_room"),
    solver=HumanRoomSolverSettings(solver="po"),
)
result = simulate_human_room(settings)
next_settings = replace(settings, radar=replace(settings.radar, num_adc_samples=128))
```

For a static plate, use `simulate_static_target(StaticTargetSettings(...))`:

```python
from mmWaveRadar.experiments import (
    StaticTargetGeometrySettings, StaticTargetSettings, simulate_static_target,
)

result = simulate_static_target(StaticTargetSettings(
    target=StaticTargetGeometrySettings(
        target_type="plate", range_m=0.668, width_m=0.152, height_m=0.152,
        material_name="pec",
    ),
))
```

This uses the perfect-conductor plate preset; bundle preparation can carry
numeric copper material parameters for the measurement comparison.
Settings group motion/geometry, scene, radar and solver options. They are frozen
containers; use `dataclasses.replace` for a changed run. The flat
`run_human_room_experiment` and `run_static_target_experiment` functions remain
supported compatibility entry points. See the [tutorial](../docs/tutorial/index.html#python)
and [settings API guide](../docs/code_organization.md#grouped-settings-apis).

## Solvers and settings

The GUI, the headless runner and its JSON files, the Python API, validation, and
saved bundles all use the same names (defined in `mmWaveRadar.naming`).

| Solver | What it simulates |
|---|---|
| `po` | Physical optics on the human only; the room and human-room coupling are ignored. |
| `rt` | Ray tracing of the whole scene. |
| `hybrid_po` | RT for the static room, PO for the human, body blocking of room paths (always on), and human-room coupling. |

Settings that apply to every solver have no prefix: `array_model`
(`phase_center` or `subarray`, with `array_tx_groups` and `array_rx_groups`)
and `tx_schedule` (`simultaneous` or `tdm`). The rest are prefixed by where they
apply: `rt_*` wherever RT runs (the RT solver and the Hybrid PO room), `po_*`
wherever PO runs (the PO solver and the Hybrid PO body and coupling), and
`hybrid_*` only for Hybrid PO.

Temporal cadences are counted in whole radar frames, and updates happen at frame
starts; `0` means every chirp.

| Setting | Meaning |
|---|---|
| `rt_refresh_frames` | Retrace every N frames; in between, traced paths ride the moving mesh with coherent phase and delay updates. |
| `rt_visibility_refresh_frames`, `rt_visibility_fade_frames` | Re-test known paths for blocking by the body between retraces, and the fade length. Hybrid PO room blocking uses the same cadence. |
| `po_refresh_frames` | Exact PO integration every N frames; in between, phase, delay, and spreading follow the motion. Hybrid PO coupling uses the same cadence. |
| `po_visibility_refresh_frames`, `po_visibility_fade_frames` | Facet self-occlusion cadence and fade length. |

`po_refresh_on_phase_error_rad` (default **0.1 rad**) re-integrates a facet
when `k * max_vertex(abs(ΔL_vertex - ΔL_centre))` exceeds the threshold,
measured since that facet's last exact solve. `L` is the Tx–point–Rx path
length and `k = 2π / wavelength`. The centre's change is already included in
the incremental phase update. Direct PO and coupling use this same trigger;
coupling evaluates `L` with its image-source Tx/Rx positions. Configure it in
JSON, with `--po-refresh-on-phase-error-rad`, or in Settings → PO temporal
cadence. A facet also refreshes when its area changes by more than 30% or its
normal changes by more than 20 degrees since its last exact solve. These configurable
guards catch amplitude and polarization changes below the phase threshold.

`rt_refresh_on_displacement` optionally adds an RT retrace when any vertex
moves more than the configured distance **in metres** since the last trace.
RT retraces on the frame cadence **or** on this drift; an early trace does
not reset the frame schedule. `None` disables the drift trigger, and metadata
records its threshold only when enabled.

Keep the **one-frame refresh default** for RT and PO. On the tutorial walk
(4 frames, compared with exact solves at every chirp), PO stays within 0.3% at
the one-frame default and within 0.4% at a 4-frame cadence with the 0.1 rad
trigger (5% at 0.5 rad). RT with specular paths only differs by **12–26% per
frame** at the one-frame default and by up to 100% in one frame at a 4-frame
cadence, because paths that appear between retraces are missed. The phase
trigger does not cover visibility changes, and these numbers depend on motion
and geometry; validate longer cadences against exact per-chirp solves for the
scene of interest.

RT is **not bit-reproducible** on multithreaded CPUs, even with a fixed
`rt_seed`: a few weak diffuse-then-specular paths change between runs, moving
the room components by about 0.3% (relative L2). PO and coupling components
are unaffected. Call `drjit.set_thread_count(1)` before running for identical
results; see [docs/rt_reproducibility.md](../docs/rt_reproducibility.md) for the
cause, measurements, and the path-cap hazard when raising
`rt_samples_per_source`.

For release evidence, use the [scientific validation guide](../docs/scientific_validation.md):
it separates frozen-output regressions, PEC analytic/convergence checks and measured
copper-plate agreement. See [model limitations](../docs/limitations.md),
[API/schema stability](../docs/stability.md) and [citation metadata](../CITATION.cff).

PO refresh settings are grouped in the GUI Settings panel and in
[`configs/human_room.json`](../configs/human_room.json), which can be edited
and passed to `tools/run_human_room.py --config`. Python callers use the same
names in `POConfig` or `run_human_room_experiment`:

- `po_refresh_on_phase_error_rad`: neglected vertex phase threshold (default 0.1 rad).
- `po_refresh_on_area_fraction`: relative area change since the last exact solve
  (default 0.30, meaning 30%).
- `po_refresh_on_normal_deg`: face-normal rotation since the last exact solve
  (default 20 degrees).

Any exceeded threshold re-integrates the face, for direct and reflected PO.
The Settings panel also exposes visibility sample count, adaptive visibility
sampling, and its grazing cosine margin. JSON keys and CLI flags use the same
names (replace underscores with hyphens for flags). `POConfig` owns the PO
threshold and adaptive visibility defaults; the GUI reads the public entry
point defaults rather than maintaining separate values.

`rt_human_bounces` (`single` by default, or `any`) limits
how many times an RT path may touch the human. Hybrid PO calibrates the human
PO amplitude to RT with `hybrid_calibration` (`rt_first_pose` by default,
`rt_sequence`, or `none`).

`rt_first_pose` fits one amplitude gain from the **first simulated radar
frame**, averaging ADC power across all chirps, virtual channels, and ADC
samples in that frame. The comparison uses single-human-only RT returns and
human-only PO, excluding static-room and human-environment coupling returns:
`gain = sqrt(mean(abs(adc_rt_human)**2) / mean(abs(adc_po_human)**2))`.
This gain is reused for every frame and chirp, scaling direct-human PO and
both human-environment coupling orders; static-environment RT is unchanged.
Calibration is not repeated on later chirps. With the default calibration RT
cadence, the reference is traced at the first chirp and its cached paths are
updated coherently for the remaining chirps of that frame.

With `array_model="phase_center"`, each solver works once at the Tx and Rx
array centers and adds per-channel offsets: PO gives each virtual channel its
own phase, delay, and spreading; RT gives each its own phase and shares one
delay per path. A run warns (`PhaseCenterApproximationWarning`) when the
aperture is too large for that at the target range: RT wavefront curvature
`π(d_tx² + d_rx²)/(λR)` above π/8, or a PO facet phase error
`k·a·(d_tx + d_rx)/R` above 1 rad. The warning suggests how many groups to use.

`array_model="subarray"` splits the arrays instead. `array_tx_groups` and
`array_rx_groups` list groups of zero-based antenna indices, numbered like
`selected_tx_indices`, and every active antenna must be in exactly one group
(`None` is one group). Group antennas that sit next to each other: index
order is not position order on every board (the cascade's is not), and the
warning measures each group's widest element offset. Each pair of a Tx group
and an Rx group is solved at
the centers of its two groups, and its channels are written back in the array's
order. One antenna per group is element-wise solving. Run time grows with the
number of pairs. Calibrated PO parts share one gain across the array:
`rt_first_pose` is refitted from every pair's reference powers, and
`rt_sequence` uses the first pair's fit.

```bash
# The cascade board's default active antennas are TX0-2, TX9 and RX4-7.
hermes-human-room --board MMWCAS_RF_EVM --array-model subarray \
  --array-tx-groups 0-2 9 --array-rx-groups 4-5 6-7 --dry-run
```

In the GUI, use Settings › Solver › MIMO array, with groups written as
`0-2; 9`.

Earlier names (`full_rt`, `coherent_rt`, `human_only_po`, `mobility_mode`,
`tdm_enabled`, `coupling_*`, `--modes`, `--mobility-mode`, ...) raise an error
naming their replacement. Saved bundles that record `tdm_enabled` are still
read.

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
- **Dynamic Scenes** runs the RT, PO, and Hybrid PO solvers for a prepared or
  uploaded room with AMASS-compatible poses or evaluated mesh motion.
- **Bundle Comparison** validates a
  [HERMES Bundle v1](https://github.com/wisermaclab/hermes/blob/main/validation/BUNDLE_SPEC.md),
  explores its ADC and DSP products, and compares them with fresh or saved
  simulations.

The Settings dialog configures TI hardware, active Tx/Rx elements, the Tx
schedule, the FMCW waveform, antenna response, solver budgets and temporal
cadences, and human motion coordinates. The radar phase center remains
at `(0, 0, 0)`; scene and target positions are relative to it.
Edits are staged until **Apply**; **Cancel**, Escape, or dismissing the dialog
restores the previous values. **Apply & rerun this page** also starts the active
workflow. Each settings group names the workflows that use it. Bundle Comparison
uses the radar and solver settings in its primary bundle, independently of this
dialog. The radar summary reports bandwidth over the sampled ADC window and its
corresponding range resolution.

Each workflow remembers the settings behind its last completed results. Editing
those settings keeps the plots visible and shows a banner naming the changes,
with **Rerun** and **Revert** actions. A dot on the workflow tab marks stale
results, and exports remain disabled until the settings match or a new run
finishes. Theme reloads retain completed results and their settings in a bounded
server-memory cache; restarting the server clears that cache.
Plot interaction tips and diagnostic controls remain visible alongside the plots;
the Static Target scene stays visible while scrolling through its controls.
Each tab fits its active content. Dynamic Scenes opens on **Scene and motion**.
Its three solvers appear as checkboxes. **Start [s]** and **End [s]**
choose the motion time window, in seconds after the motion's first pose. Radar
frame 0 starts at the start time, the motion then plays in real time, and every
radar frame whose chirps finish by the end time is simulated; the window shows
the resulting frame count. Each chirp samples the pose at its own instant, so
Doppler follows the recorded motion speed for any frame period. Playback shows
the motion time of the previewed pose, and the ADC tab provides radar frame,
chirp and channel selection; radar frame indices start at zero. Run metadata
shows runtime, path/facet counts and calibration status; the largest measured
stages appear in a sorted table, with the full profile under **Advanced**.

The default **PO facet integration** mode is **Adaptive analytic parent facet**.
It splits a triangle into four children when either its centroid-linearized
phase span exceeds **6 rad** or its edge-midpoint curvature error exceeds
**0.005 rad**. The curvature test uses `kc/3` times the largest midpoint path
interpolation error over all edges and Tx/Rx combinations. Each leaf uses the
existing exact linear-phase triangle integral. Children inherit the original
facet's visibility weight and sum coherently under its persistent identity.
Both tolerances are configurable in **Settings → Solver settings**. The GUI
remembers a separate phase span for each integration mode across browser
and theme reloads; quadrature initially uses **1 rad**.
The existing depth **3** and **64 children per parent** budgets still apply; the
GUI reports when these limits leave facets above a tolerance. The earlier
quadrature and Fraunhofer analytic modes remain available for comparisons.

Hybrid PO uses shared static-blocking and environment-coupling passes in the
simulator, the GUI/notebook pipeline, and cached coupling recomputation.
Recomputation follows the same configured visibility-refresh schedule as a
fresh run and checks cancellation between chirps and after each coupling solve.
Direct-human PO and both coupling directions share the same incremental facet
updater. Coupling reuses the selected image-source bank and caches surface
contributions separately for each reflector and direction. Each chirp updates
phase, spreading, and delay; new facets and facets exceeding the configured
`po_refresh_on_phase_error_rad` threshold are reintegrated.
Periodic exact refreshes follow `po_refresh_frames`, and facets near the
receiver IF cutoff retain exact integration for clipping. As with direct-human
incremental PO, reuse approximates changes in scattering between exact solves.
Runtime counts include `hybrid_coupling_recomputed_faces` and
`hybrid_coupling_phase_updated_faces` when coupling reuse is enabled.

In **Settings → Solver settings → Hybrid PO**, the
power cutoff defaults to **−20 dB relative to the strongest candidate** and
the maximum count defaults to **4** (adjustable from 1 to 4). Candidates come
from scene geometry and are ranked by estimated coupling power toward the
human motion volume, including reflection loss, distance, antenna response,
and static obstruction. The cutoff is applied first, then the count cap;
selection stays fixed throughout the run. This estimate is a selection proxy
and does not bound the final coherent PO signal power.

GUI exports are portable HERMES Bundle v1 archives containing the ADC products,
sensor and frame metadata, scene inputs, diagnostics, and reproducible
experiment manifests. Dynamic exports retain each selected solver's ADC as
`simulated_adc_<solver>.npz`, and Hybrid PO's separable terms (human PO,
blocked room, and both coupling directions) in `hybrid_po_components.npz`;
licensed model files are never embedded.

The launcher binds to `127.0.0.1` by default and protects each launch with a
random password. Remote binding requires the explicit `--unsafe-allow-remote`
override and an origin allowlist. Use it only on a trusted, access-controlled
network: the launcher is not a TLS endpoint or multi-user identity system.

Check optional dependencies and source-checkout fixtures with:

```bash
hermes-gui --self-check
```

Dynamic Scenes accepts evaluated mesh motion archives containing `vertices`
(`[frames, vertices, 3]`), `faces` (`[triangles, 3]`), and `times` (`[frames]`).
Supplied meshes take precedence over any pose parameters in the same archive
and do not require a body model. Pose-only archives require an authorized
SMPL-family model configured in the GUI or through `MMWAVE_SMPL_MODEL_DIR`.
Static Target and measurement-only bundle comparison do not require one.
The public release contains no licensed body model. Uploading motion loads its
preview automatically. **Load default motion** is shown only before motion is
loaded and uses the bundled example. Both scene pages group position and
orientation into compact numeric rows with up/down controls.

Range-profile views show the mean linear power across active Tx/Rx channels
on one plot, with solver curves overlaid. Each profile averages
chirps within the selected frame. Multi-frame results provide previous/next
controls and a frame selector; frames are not averaged together. Values below
−80 dB are hidden. Simulated profiles retain their absolute scale; bundle
comparisons normalize the channel mean using one peak reference across frames
of each dataset, so differences between frames remain visible. Each simulated
mode also has a separate plot overlaying its individual Tx/Rx channels. Hybrid
PO's channel plot follows its averaged component plot.

Dynamic Scenes defaults to **Empty room**, with the tutorial's metal floor
and plasterboard walls and no furniture. The radar is at `(0, 0, 0)` m and the
floor surface is at `z = -1.035` m, placing the radar 1035 mm above it. A standalone
copy is available at [`data/scenes/empty_room.xml`](../data/scenes/empty_room.xml).
The furnished **Tutorial bedroom** remains available in the scene selector.
**Human position → Use motion-file coordinates** preserves the file's positions
and trajectory after coordinate conversion. Choose **Place manually** to set the
first pose's bounding-box centre with the X/Y/Z controls.
Their limits follow the scene's floor footprint and height and update when the
scene changes. These limits apply to manual placement; file positions are preserved.
The preview time and simulation start/end controls use seconds. Changes to the
room definition in `tutorial_support.py` require restarting the GUI process.

In **Settings → Solver settings → Human motion coordinates**, either keep the
file's `output_axes`/`output_signs` mapping or explicitly select its source up
axis, forward axis, handedness, and evaluated mesh units. The manual selection
overrides the file mapping and converts evaluated vertices to the radar frame
(+X forward, +Y left, +Z up, metres) before applying human yaw and position.
The preview and simulation use the same conversion, which is recorded in the
exported experiment metadata. The source file is not modified. Local joint
rotations are not permuted; SMPL is evaluated first, and its translations are
expected in metres.

### Headless scene runner

Keep the complete run setup in **`configs/human_room.json`**. It contains motion
and scene paths, human placement,
the motion time window, solvers, all radar and PO/RT/Hybrid PO settings, and the output
path, logging and overwrite policy. Its supplied values use the furnished
tutorial bedroom and explicit human placement; the GUI starts with the empty room
and motion-file positions.

Edit that file, then run from the repository root:

```bash
python tools/run_human_room.py
```

The source-checkout script reads `configs/human_room.json` automatically,
independently of the working directory. Select another complete or partial JSON
file with `--config`:

```bash
python tools/run_human_room.py --config /path/to/run.json
```

After installing/updating the package, `hermes-human-room --config /path/to/run.json`
or `python -m mmWaveRadar.gui.scene_cli --config /path/to/run.json` runs the same
workflow. These installed entry points use GUI defaults when no file is supplied.
The source-checkout wrapper instead selects the checked-in JSON by default.

JSON uses the experiment API's parameter names and SI units. **All paths,
including `output`, are relative to the JSON file**, not the working directory.
Useful fields to edit are:

- `human_motion`: an evaluated mesh or AMASS-like NPZ. The example uses the GUI's
  bundled AMASS motion and `smpl_model_dir`; mesh NPZ inputs need no body model.
  Set the model directory here when it differs from the supplied checkout path.
- `scene_xml`: `null` for the tutorial bedroom, or a custom environment XML path.
  This replaces only the environment, not the human or radar configuration.
- `human_position_m`, `human_yaw_deg`, and `solvers`: placement and the solvers
  to run (`po`, `rt`, `hybrid_po`).
- `motion_start_s`, and either `motion_end_s` or `num_frames`: the motion time
  window. A frame always means one radar frame. `motion_start_s` is the motion
  time, in seconds after the first pose, at which radar frame 0 starts. The run
  covers every radar frame whose chirps finish by `motion_end_s`, or exactly
  `num_frames` radar frames; with both `null` it simulates one frame.
- `rt_*`, `po_*`, and `hybrid_*`: solver settings and cadences, as described in
  [Solvers and settings](#solvers-and-settings).
- `board_model`, `carrier_frequency_hz`, `slope_hz_per_s`, sampling/chirp fields,
  `selected_tx_indices`, `selected_rx_indices`, and `tx_schedule`: radar setup.
  Antenna indices are **zero-based** here and in every front end. All explicitly configured values
  are retained when another field, such as the board model, is changed.
- `output`, `quiet`, and `force`: bundle ZIP path, progress logging, and whether
  an existing output may be replaced. The example protects outputs (`force: false`).

Inspect the effective settings without loading motion or running a solver, or
save a fully resolved copy of them:

```bash
python tools/run_human_room.py --dry-run
python tools/run_human_room.py --dry-run --write-config settings.json
```

Optional command-line flags still override individual file settings. Their paths
are relative to the working directory, and their numeric units match the GUI.
Precedence is **GUI fallback defaults → selected JSON → command-line flags**:

```bash
python tools/run_human_room.py \
  --config /path/to/run.json --start-s 3.0 --end-s 4.0 --output results/walk-3s.zip
```

The radar phase center stays at `(0, 0, 0)`, with +X forward, +Y left, +Z up.
The human position anchors its initial mesh bounding-box center; later poses
retain their motion relative to it. Orientations use degrees. Solver flags take
the setting name with dashes (for example `--solvers rt po`,
`--rt-refresh-frames 0`, `--tx-schedule tdm`, `--no-hybrid-coupling`); unit
flags such as `--carrier-ghz` keep the GUI's units. The motion's file-axis
mapping is retained unless both manual source axes are supplied. `--help` lists
all individual overrides.

XML uses the existing GUI loader: referenced assets must be accessible, and a
portable exported scene should be self-contained. The output ZIP includes all
selected solvers and opens with `hermes-gui --bundle FILE.zip`. Bundles contain
human meshes; redistribute them only with the necessary rights. Resolved JSON
contains local input/model paths and is intended for local reuse.

Configuration files use `schema_version` 3. Version 1 files used pose-index
`begin_frame_index`/`end_frame_index` windows, and version 2 files used the
earlier setting names; both are rejected with the names to change.

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
file contract, solvers, channel selection, TDM timing, metrics, and
outputs.

Run simulation and measurement comparison with an explicit output directory.
`--clip` accepts a self-contained ZIP or a bundle directory:

```bash
python -m validation.cli \
  --clip /path/to/capture-bundle.zip \
  --suite dsp_maps \
  --out /tmp/hermes_validation \
  --progress
```

Without `--simulated-adc`, validation runs the selected solvers (`po` and `rt`
by default) using the bundled sensor, scene and target. It writes metrics, DSP
maps, simulated ADC and comparison plots, leaving the source ZIP unchanged:

```bash
python -m validation.cli \
  --clip /path/to/capture-bundle.zip \
  --out /tmp/hermes_validation \
  --solvers po hybrid_po
```

TX/RX selectors (`--tx-indices`, `--rx-indices`) are zero-based, as everywhere
in HERMES, so a dataset's TX1 is index 0. By default, TDM
simulation evaluates each selected TX at its physical slot time. Add
`--ignore-tdm-timing` to evaluate the selected array jointly at the TDM-cycle
start while retaining the full-cycle slow-time interval. See
[TDM timing policies](https://github.com/wisermaclab/hermes/blob/main/validation/README.md#tdm-timing-policies) for the exact
PO/RT, phase, channel-ordering, and cache behavior.

Prepare a portable directory or ZIP from normalized ADC, radar configuration,
frame timing, and optional scene/target/motion inputs:

```bash
mmwave-prepare-bundle \
  --adc /path/to/capture_adc.npz --sensor /path/to/sensor.json \
  --scene /path/to/scene.xml --target-mesh /path/to/plate.npz \
  --output /path/to/capture-bundle.zip
```

Use `--config /path/to/prepare.json` instead of direct inputs for a JSON-based
setup. Direct paths are relative to the current directory; config paths are
relative to the JSON file. Both forms accept `--output` and `--force`.

The [measurement tutorial](../tools/bundle_prepare/BRING_YOUR_OWN.md) includes an
installed synthetic static-target example, a config reference, and the Python
writer API. Custom arrays and explicit channel/TDM maps use version 2 feature
gates; existing version 1 bundles remain readable.

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

`FMCWConfig.adc_mode` defaults to `"complex1x"`. Range FFTs retain all
`NFFT` bins in that mode, ordered by increasing range even for a negative
chirp slope. Set `"complex2x"` or `"real"` for half-band processing.
Synthesis applies an ideal IF cutoff before sampling: 0.9 times the sample
rate for Complex 1x, or 0.45 for Complex 2x/real, capped by the board's analog
bandwidth and optional `if_bandwidth_hz`. These usable-band limits follow
the [TI interface specification](https://dr-download.ti.com/software-development/firmware/MD-074SwD074J/02.02.04.00/mmWave-Radar-Interface-Control.pdf#page=123).
ADC mode and IF cap are also available under the GUI's FMCW settings.

`cube.metadata.adc_if_filter` records tested, rejected, and partially filtered
path-channel evaluations. For delay-resolved PO these counts refer to original
parent facets; analytic facets crossing the cutoff are clipped and integrated
analytically before coherent summation. Counts cover synthesis calls, including
diagnostic/reference products, rather than distinct physical reflectors.
Low-level `synthesize_adc_from_paths(..., diagnostics=counts)` can accumulate
the same information in a caller-owned dictionary. Existing sampled ADC cannot
be unaliased by this filter; it applies during synthesis.

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
also offered under CC BY-NC 4.0, as marked by SPDX headers. See [NOTICE](../NOTICE)
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
