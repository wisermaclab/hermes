# Validation-bundle preparation

This directory contains two reviewed reference adapters for preparing standard
HERMES bundles from separately obtained datasets:

- `rtpose/prepare_rtpose_bundle.py`
- `rtpose/rtpose_adc.py`
- `rtpose/reconstruct_rtpose_background.py`
- `rtpose/fit_rtpose_environment_geometry.py`
- `mmradarpose/prepare_mmradarpose_bundle.py`
- `mmradarpose/mmradarpose_environment.py`

Dataset timing, coordinate transforms, ADC layout, frame mapping, and source
validation remain explicit in each adapter.

The dataset-neutral contract is implemented in `contract.py`, and
`integrity.py` checks a completed bundle without invoking simulation or
comparison metrics. Both adapters run this checker before publishing their
staged output. It is also available directly:

```bash
mmwave-check-bundle /path/to/prepared_bundle
```

The RT-Pose adapter automatically reads the selected measured AWR2243 cascade
frame and produces the bundle's ADC cube; ADC paths and decoder programs are
not command-line inputs. In the same call it reuses supplied or dataset-local
geometry, or reconstructs a background from synchronized LiDAR/stereo inputs
and invokes the bundled fitter. The fitter turns that cloud into simple floor,
wall, and furniture scatterers plus a portable Mitsuba/Sionna scene. This
remains dataset-specific preprocessing and is not part of the universal
validation-bundle contract.

RT-Pose's source MIMO receiver permutation is normalized inside its adapter.
Completed bundles preserve the measured physical TX-slot schedule but store
physical RX labels in ascending `RX1..RX16` order within every TX-major block.
The source and exported RX orders are retained in `sensor.json` for provenance.
See `rtpose/README.md` for the exact permutation and the documented compact
half-wavelength horizontal subarray.

The mmRadPose adapter has the same compact five-input shape. It discovers the
parsed radar cube and alignment, exports one measured ADC frame, and writes a
canonical `amass_sequence.npz` motion window. Its environment stage is
laser-only: it reuses fitted geometry or fits a laser cloud already registered
to the mmRadPose radar frame. It never reads camera imagery or mistakes the
dataset's radar target-list point clouds for laser geometry.

RT-Pose and mmRadPose are included because they are public mmWave pose
datasets used in HERMES validation and demonstrate different radar/channel and
synchronization layouts. They are not the only datasets the validation package
can consume. For another dataset, any external or project-specific converter
may be used as long as it writes the bundle contract documented in
`validation/README.md` and the result passes `mmwave-check-bundle`.

Both adapters have a deliberately compact, fidelity-oriented interface and
write only canonical `amass_sequence.npz` body motion with source and
radar-relative timing. Simulation generates meshes from those parameters and a
licensed SMPL model when needed. Validation does not read a separate baked-mesh
body-motion archive or alternate parameter filename.
Preparing a reduced public fixture is a separate release step, because removing
body parameters or changing shape does not by itself grant redistribution
rights for measured radar, source motion, images, or other participant-derived
data.
