# Code organization

Shared code lives with the contract or operation it implements. Existing import
aliases remain available while internal callers use the modules below.

| Responsibility | Module |
| --- | --- |
| Raw portable names, contained files and destinations | `bundle_prepare.paths` |
| Resource-bounded ZIP extraction | `bundle_prepare.archives.extract_bundle_zip` |
| Common archive/SMPL validation and adapter entry points | `bundle_prepare.adapters` |
| Bundle feature requirements | `bundle_prepare.contract.required_bundle_features` |
| Yaw, pitch and roll rotation in radians or degrees | `mmWaveRadar.geometry.rotation_matrix` |
| TI CLI waveform parsing | `mmWaveRadar.radar.ti_cli.parse_ti_cli_config` |
| Model and motion defaults | `mmWaveRadar.defaults` |
| Scalar validation | `mmWaveRadar.values` |
| Snapshot/steering array shapes and waveform lookup | `mmWaveRadar.dsp.arrays` |
| Coherent sums of valid path coefficients | `mmWaveRadar.simulation.path_products` |
| Experiment radar construction | `mmWaveRadar.experiments.radar` |

Path checks validate the original string before `Path` can normalize it. ZIP
directory names and Windows scene-asset names have explicit additional policies.
Filesystem checks reject symlink components, including pre-existing destination
links during extraction. A GUI's permission to read a server path is a separate
authorization policy.

## Mesh sequences

`targets.mesh.MeshSequenceBase` derives counts and displacement from the mesh
sequence protocol. `MeshSequenceView` supplies sampled vertices for lazy wrappers.
`PlacedMeshSequence` handles yaw, translation and outward winding. Its default
preserves the source timeline; GUI placement opts into rebasing to the first pose.

`TimeWindowMeshSequence` supplies offset evaluation. The source-knot window,
radar-frame window and hybrid two-endpoint window keep their sampling policies.
They cannot be substituted for one another: interpolation knots and physical
chirp times affect the simulated geometry. `MeshSequenceTimeWindow` and
`RigidTransformMeshSequence` remain importable from `tutorial_support`.

## Validation

`validation.runner` orchestrates the benchmark and retains its previous import
paths. New callers should import the operation from its owner:

- `configuration`: run settings and their validation;
- `channels`: physical channel selection and ADC slicing;
- `simulation`: scene/motion loading and slot-aware TDM simulation;
- `cache`: content fingerprints and cached ADC;
- `dsp_maps`: DSP products and comparison plots.

Dataset identity, frame mapping and coordinate alignment remain in the adapters.
Only archive decoding, common SMPL shape/value checks and bundle publication are
shared. The adapters deliberately retain their different beta-shape and timing
requirements.

## GUI and solvers

`gui.app.build_app` delegates widget construction to `gui.assembly.AppBuilder`,
whose ordered methods build controls, plots, layouts and session wiring.
It connects seven workflow classes under
`gui.workflows`: application lifecycle, settings, radar, static targets, human
motion, bundles and session persistence. Each session owns a closed, slotted
`ApplicationContext`. Each workflow accepts its own typed protocol in
`gui.workflows.contexts`, declaring its controls, state and callbacks. The
context implements their union without a dynamic attribute fallback; a
misspelled assembly assignment raises immediately. Workflow fixtures can still
supply only the controls exercised by a callback. CI checks workflow and
assembly types with mypy.
Callbacks retain their names and registration order. Browser scripts live in
`gui.interaction_scripts`; the two scene components share the Plotly lookup.
Results display and bundle export also have separate construction stages.

The RT path-bank engine separates anchor scheduling, tracing, visibility,
transition matching, diagnostics and chirp synthesis. PO has mesh, channel,
recording and calibration stages; incremental PO separates geometry/visibility,
path geometry, facet updates and state publication. Coupling separates geometry,
centre channels, projection and wideband/hardware application. `_POChannelInputs`
groups fixed geometry, visibility scenes and numerical settings for exact and
incremental PO. Each pass owns its temporary state; the simulator owns its
configuration and cancellation hooks.

## Grouped settings APIs

| Entry point | Settings |
| --- | --- |
| `experiments.simulate_human_room(settings)` | `HumanRoomSettings`: motion, scene, radar, solver |
| `experiments.simulate_static_target(settings)` | `StaticTargetSettings`: target, radar, solver |
| `simulation.run_hybrid_po_pipeline(scene=..., radar=..., target=..., settings=...)` | `HybridPipelineSettings`: frames, array, RT, physics, execution |
| `simulation.evaluate_po_channel(vertices, faces, settings=...)` | `POChannelSettings`: array/material, visibility, integration, compute |

Experiment settings are exported by `mmWaveRadar.experiments` and declared in
`experiments.settings`. Pipeline/channel settings are exported by
`mmWaveRadar.simulation` and declared in `simulation.options`.
The existing `run_human_room_experiment`, `run_static_target_experiment`,
`run_calibrated_hybrid_po` and `po_channel` signatures remain supported. Their
adapters construct settings and call the same staged implementation.

For a human in the empty room:

```python
from dataclasses import replace
from mmWaveRadar.experiments import (
    HumanMotionSettings, HumanRoomSettings, HumanRoomSolverSettings,
    RadarSettings, RoomSceneSettings, simulate_human_room,
)

settings = HumanRoomSettings(
    motion=HumanMotionSettings(
        human_motion="/path/to/mesh_sequence.npz",
        human_position_m=None,  # retain the coordinates stored in the file
        motion_start_s=0.0,
        num_frames=2,
    ),
    scene=RoomSceneSettings(room_preset="empty_room"),
    radar=RadarSettings(num_adc_samples=225, num_chirps_per_frame=16),
    solver=HumanRoomSolverSettings(solver="po"),
)
result = simulate_human_room(settings)
next_settings = replace(settings, radar=replace(settings.radar, num_adc_samples=128))
```

Use `dataclasses.replace` to change one group. Settings are frozen containers;
arrays, motion and scene objects stay referenced rather than being deep-copied.
`settings.as_keywords()` provides the equivalent flat keywords for integrations
that still use a legacy entry point. Defaults and every forwarded option are
covered by compatibility tests; numerical tests exercise channel order, motion
time, wideband PO and TDM.

`RadarSettings` and `build_radar` are shared by the human-room and static-target
experiments. Chest geometry, markers and radar placement in benchmark scripts
use `tutorial_support`.

DSP reductions retain their axis contracts. Imaging accepts empty snapshot
matrices and one channel; covariance estimation requires at least one snapshot
and two channels. A human-room profile retains frames and channels and averages
chirps. Validation's map profile averages the axes preceding its final range
axis. These operations remain distinct.
