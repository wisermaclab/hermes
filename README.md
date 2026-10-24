# HERMES

**Human-Environment Radar ModEling and Simulation** simulates mmWave FMCW radar
returns from people, static targets, and indoor environments. It provides a
browser GUI, command-line tools, and Python APIs for generating ADC data,
visualizing radar products, and comparing simulations with measurements.

**[Read the tutorial](https://wisermaclab.github.io/hermes/docs/tutorial/)**
· [Notebook examples](demo/README.md)
· [API reference](https://wisermaclab.github.io/hermes/docs/api/)

## Simulation solvers

| Solver | Model |
| --- | --- |
| **PO** (`po`) | Physical optics on the target. |
| **RT** (`rt`) | Ray tracing of the complete scene. |
| **Hybrid PO** (`hybrid_po`) | Static-environment ray tracing combined with target PO, body blocking, and target–environment coupling. |

Radar configuration includes FMCW waveforms, TI board presets, antenna selection,
and TDM timing. Outputs include ADC cubes, range profiles, range–Doppler maps,
and portable experiment bundles.

## Installation

From a source checkout, using Python 3.10 or later:

```bash
git clone https://github.com/wisermaclab/hermes.git
cd hermes
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[gui]"
hermes-gui
```

The package imports as `mmWaveRadar`. The simulation runtime uses Sionna RT,
Mitsuba, and Dr.Jit, with CPU or compatible NVIDIA GPU ray tracing. See the
[runtime requirements](docs/user_guide.md#runtime-platforms) for platform support
and optional acceleration.

Static targets and evaluated mesh motion need no SMPL body model. Pose-based
human motion additionally requires the `[smpl]` dependencies and a separately
licensed model; see the
[installation tutorial](https://wisermaclab.github.io/hermes/docs/tutorial/#install).

## Choose a workflow

- **Explore interactively:** launch `hermes-gui` and follow the
  [GUI tutorial](https://wisermaclab.github.io/hermes/docs/tutorial/#gui).
- **Run repeatable experiments:** use a JSON configuration with
  [the headless CLI](https://wisermaclab.github.io/hermes/docs/tutorial/#cli)
  or the [Python API](https://wisermaclab.github.io/hermes/docs/tutorial/#python).
- **Compare with measurements:** follow
  [bundle preparation](https://wisermaclab.github.io/hermes/docs/tutorial/#prepare-bundle)
  and [simulation validation](https://wisermaclab.github.io/hermes/docs/tutorial/#validation).
- **Try a data-free example:** open the
  [point-target notebook](demo/Point-Targets.ipynb) or
  [PEC-plate demo](demo/mobicom/README.md).

## Validation and reference

The [scientific validation guide](docs/scientific_validation.md) separates
analytic convergence checks, release regressions, and measured copper-plate
comparisons. Review the [0.2.0 validation report](docs/releases/0.2.0.md) and
[solver limitations](docs/limitations.md) when interpreting results.

Detailed settings and runtime behavior are in the [user guide](docs/user_guide.md).
See also [API and bundle stability](docs/stability.md),
[contributing](CONTRIBUTING.md), and [issue reporting](https://github.com/wisermaclab/hermes/issues).

## Citation

If you use HERMES in your research, please cite our demo paper:

Rong Zheng and Mihail Georgiev. 2026.
[Demo: HERMES: Human-Environment Radar ModEling and Simulation for mmWave FMCW radars](https://dl.acm.org/doi/10.1145/3795866.3848430).
In *Proceedings of the 32nd Annual International Conference on Mobile Computing
and Networking (MobiCom '26)*, 1867–1869. ACM.

```bibtex
@inproceedings{zheng2026hermes,
  author    = {Zheng, Rong and Georgiev, Mihail},
  title     = {{Demo: HERMES: Human-Environment Radar ModEling and Simulation
                for mmWave FMCW radars}},
  booktitle = {Proceedings of the 32nd Annual International Conference
               on Mobile Computing and Networking},
  series    = {MobiCom '26},
  year      = {2026},
  pages     = {1867--1869},
  publisher = {ACM},
  doi       = {10.1145/3795866.3848430},
  url       = {https://dl.acm.org/doi/10.1145/3795866.3848430}
}
```

For the software version used in your work, see [CITATION.cff](CITATION.cff).

## License

Project-owned code uses [PolyForm Noncommercial 1.0.0](LICENSE); project-owned
documentation and data use [CC BY-NC 4.0](LICENSE-DOCS-DATA). Bundled third-party
examples and external body models retain their separate terms; see
[Third-Party Notices](THIRD_PARTY_NOTICES.md).

For commercial licensing, contact **Rong Zheng** at **rzheng@mcmaster.ca**.
See [Commercial Licensing](COMMERCIAL-LICENSING.md).
