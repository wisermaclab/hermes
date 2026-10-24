# Scientific verification and release validation

Three questions have separate evidence:

* **Regression:** does this version reproduce a frozen release's calculation?
* **Analytic verification:** does a solver approach a known solution in its valid regime?
* **Measurement validation:** how closely does a configured model reproduce an
  independent measurement? A synthetic fixture cannot answer this.

The [0.2.0 development report](releases/0.2.0.md) records the current analytic,
ray-density and measured copper-plate results, including failed or unverified cases.

## Freeze and compare releases

Each published release attaches an evidence archive containing its frozen
reference arrays. To check a candidate checkout against a release, extract that
archive, then run from the repository root with the same inputs, environment and
configuration:

```bash
python tools/release_validation.py run \
  --config configs/release_validation.json \
  --output outputs/release-validation/candidate
python tools/release_validation.py compare \
  --run outputs/release-validation/candidate \
  --reference /path/to/hermes-0.2.0-evidence/reference
```

**Maintainers** freeze a release's reference from a reviewed run of that release:

```bash
python tools/release_validation.py run \
  --config configs/release_validation.json \
  --output outputs/release-validation/0.2.0
python tools/release_validation.py freeze \
  --run outputs/release-validation/0.2.0 \
  --reference /path/to/references/0.2.0 --release 0.2.0
```

Output and reference directories must be new. Freeze never overwrites a reference.
Reference arrays are never stored in Git; keep them with the release's evidence
archive and in a durable copy.
Workers in one frozen run must share one source snapshot. Reports disclose mixed
development snapshots, and freezing/comparison rejects them.
Checksums detect damaged arrays. Inputs, solver settings, seeds, dependencies and
resolved backends must match. The source version is deliberately excluded from
protocol matching so different releases can be compared.

PO compares complex ADC with rtol=1e-10, atol=1e-16 in the CPU float64 protocol.
RT and Hybrid PO compare **mean linear range-time power** across at least five
seeds. Their tolerance is frozen from the largest observed pairwise seed-to-seed
relative L2 power difference, with a 1e-6 numerical floor. This empirical envelope
has no guaranteed coverage for unseen seeds; a large band weakens the gate and
must be reviewed. It is not a confidence interval. Stochastic complex fields are
never averaged.

Reports retain fixture/solver metrics, provenance, seed counts, power means and
95% Student-t intervals. These pointwise intervals assume independent simulation
seeds and approximately normal seed means. They exclude model bias and
measurement uncertainty. A single PO run has no estimated seed interval.
Configuration-study reports retain IQR dispersion and also include intervals for
seed-mean measurement errors.

CI runs native synthetic fixtures and the analytic suite, and uploads reports.
Set the Actions variable HERMES_REFERENCE_RUN_ID to the reviewed workflow run
whose hermes-release-reference artifact contains the frozen Linux reference.
Ordinary CI explicitly reports an unconfigured baseline; tagged release runs fail
without one. A manually dispatched freeze produces an artifact and does not
replace the selected baseline. Artifact retention is finite: keep a durable copy.
Private measured fixtures run locally or on an authorized private runner.
Published releases freeze their own reference after the comparison passes and
attach an evidence archive containing the reports, figures and reference arrays.

## Analytic and ray-density verification

From the repository root, with HERMES installed (`python -m pip install -e .`):

```bash
python -m validation.analytic --output outputs/release-validation/analytic
python -m validation.analytic_rt --output outputs/release-validation/rt-convergence
python tools/release_validation.py report \
  --run outputs/release-validation/candidate \
  --analytic outputs/release-validation/analytic/analytic.json
```

The plate sweep compares native centroid and analytic-facet integration against
an independent far-field PEC formula, without fitting gain. It writes JSON,
Markdown and a standalone convergence plot. Sphere density is compared to the
high-frequency optical limit, not an exact Mie resonance solution.

The RT diagnostic sweeps 1,000 to 1,000,000 rays/source and five seeds for a plate
and square trihedral, enabling up to three reflections. It records missed paths,
equivalent RCS and confidence intervals. Native RT lacks finite-aperture
diffraction and PO exit-aperture integration: higher ray density need not recover
the PEC formula. This diagnostic reports discrepancies, not an automatic RCS pass.
Direct single-bounce PO cannot verify a corner's triple reflection.
See the [MobiCom PEC-plate demo](../demo/mobicom/README.md) and the broader
[configuration study](../benchmarks/configuration_tradeoffs/README.md).

## Measured copper plate

Add one fixture per angle to a local configuration. Paths resolve relative to
the configuration file:

```json
{
  "id": "copper-00deg",
  "kind": "measurement",
  "bundle": "/path/to/copper-plate/plate_00deg",
  "solvers": ["po", "rt", "hybrid_po"],
  "angular_study": "copper-plate",
  "angle_deg": 0,
  "range_gate_m": [0.5, 0.8],
  "target_material": {
    "relative_permittivity": 1,
    "conductivity": 58000000,
    "thickness": 0.001,
    "scattering_coefficient": 0
  }
}
```

Include all measured angles 0–30°. The native comparison retains per-fixture,
per-solver range-time, range-Doppler and angle metrics. The angular study uses a
1024-point Blackman FFT, a per-channel peak power in the declared range gate, then
a mean across pairs, loops and frames. It aligns each simulated curve once at 0°,
retains that offset at other angles and reports angular-response RMSE in dB.
It does **not** validate absolute channel gain. Raw ADC remains unchanged.
Document acquisition and geometry uncertainties, including unverified TX order.

## Provenance

Existing experiment manifests describe scene, radar, solver controls, seed and
measurement identity. Benchmark studies already capture input/source hashes,
versions and devices. Portable output provenance adds:

* HERMES version, commit, dirty flag and source hash;
* Sionna RT, Mitsuba, Dr.Jit, NumPy and PyTorch distribution versions;
* Mitsuba variant, resolved PO/ADC backend, precision and seed;
* content hashes independent of local path names.

Native RadarCube metadata and saved cube NPZs retain provenance. Input hashes
cover evaluated motion knots, waveform, array/pose and native scene parameters.
Prepared and GUI bundles contain provenance.json with all packaged file hashes
and experiment manifests. Validation caches retain their producer provenance;
metric reports record the validation execution separately. Unknown fields remain
null. Wheels without a source checkout do not claim a Git commit. Retain the
input bundle and report alongside derived maps/figures, which are not standalone
experiments. Hashing evaluated motion can add preparation time for lazy sequences.

See [limitations](limitations.md) and [stability/citation](stability.md).
