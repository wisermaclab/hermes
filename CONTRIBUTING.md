# Contributing

Thank you for your interest in contributing to HERMES.

## Maintainer

HERMES is maintained by Rong Zheng and the Wireless System Research Group at
McMaster University. Use the repository
[issue tracker](https://github.com/wisermaclab/hermes/issues) for public bug
reports, feature proposals, and contributor questions.

The public `wisermaclab/hermes` repository is a reviewed snapshot of the private
authoritative development repository. Public pull requests are welcome. A
maintainer imports accepted changes into the authoritative source, preserving
authorship, and the next audited synchronization updates the public snapshot.
Files under `.github/` and `SECURITY.md` are maintained directly in the public
repository; other released files are synchronized from the authoritative source.

## Development Workflow

1. Open an issue or discussion before starting large changes.
2. Keep pull requests focused on one feature, bug fix, or documentation update.
3. Use clear commit messages that describe the user-visible or developer-visible
   change.
4. Do not commit local datasets, generated outputs, model files, cache
   directories, or machine-specific paths. The exact CMU conversion documented
   in `data/AMASS/README.md` and the two limited bundles documented in
   `data/validation_bundles/` are reviewed, byte-pinned source-checkout
   exceptions. Do not modify, replace, or add to them without a maintainer-led
   provenance, licensing, privacy, integrity-pin, packaging, and release-audit
   review. Other public examples should generate synthetic inputs or clearly
   request user-provided external data.

## Code Style

- Follow the existing module structure and naming conventions.
- Keep public APIs typed and documented with concise docstrings.
- Prefer small, focused helpers over broad refactors.
- Add comments only where they clarify non-obvious radar, DSP, simulation, or
  dataset assumptions.
- Keep Python source headers consistent with the project SPDX and copyright
  headers.

## Tests

Run the relevant tests before submitting changes:

```bash
pytest -q tests/unit
```

For release-facing changes, also run:

```bash
git diff --check
python -m compileall -q src validation tools benchmarks
```

Notebook files under `demo/` should be committed without execution counts or
cell outputs.

## Documentation

- Update `README.md` when setup, workflow, or public behavior changes.
- Update `docs/api/index.html` when public APIs change.
- Regenerate the API reference with
  `python tools/generate_api_docs.py --date "Month D, YYYY"` after adding or
  changing public classes, functions, or modules.
- Update `THIRD_PARTY_NOTICES.md` when adding or changing third-party
  dependencies, references, generated assets, or required external inputs.
- Keep raw datasets and unreviewed conversion pipelines outside this
  repository. Changes to the reviewed source-checkout adapters require
  synthetic tests, pickle-free input handling, portable metadata, and an
  update to the third-party boundary when applicable.
- Treat the validation bundle as the dataset-neutral interface. A new dataset
  can use an external converter and does not need a built-in adapter; if a new
  reference adapter is proposed, justify its general value and keep all source
  assumptions outside the validation runner.
- Keep SMPL fitting outside HERMES. The public adapters accept a pickle-free
  AMASS-like base-SMPL archive containing `poses`, `trans`, `betas`, `gender`,
  `frame_ids`, and archive-supplied pose timing.
  [SparseSMPLFit](https://github.com/wisermaclab/SparseSMPLFit) is one optional
  producer of this contract.
- Keep dataset-specific mapping in HERMES: the RT-Pose adapter owns
  `Train.json`, `Radar_frameID`, and 10 Hz radar-time alignment; the
  mmRadPose adapter owns trial-frame mapping, explicit radar-to-SMPL
  alignment, and 4-by-4 source to 12-channel AOP radar export. Do not require
  radar IDs or radar timestamps in the motion archive.

## Licensing

By contributing code, you agree to provide the contribution under the PolyForm
Noncommercial License 1.0.0. By contributing project-owned documentation or
data, you agree to provide the contribution under CC BY-NC 4.0. Contributions
that mix categories must make their applicable terms clear. Your copyright
ownership is not transferred, and these contribution terms alone do not
authorize HERMES maintainers to relicense your contribution commercially. A
maintainer may request a separate written contribution agreement before
accepting a change that is intended to be available under the HERMES commercial
license.

The documented CMU-derived motion fixture and the two exact, byte-pinned
real-radar validation fixtures are outside the project licenses and remain
under their recorded upstream terms. These are narrow source-checkout
exceptions, not permission to add other dataset material. Do not submit code,
data, or generated assets unless you have the right to contribute them under
terms compatible with this repository and have documented any applicable
attribution, ShareAlike, NonCommercial, privacy, consent, patent, and
body-model requirements.
