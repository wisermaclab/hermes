# Versioning, compatibility and citation

HERMES uses package versions independently of bundle and experiment-manifest
schema versions. The current package is 0.2.0.

During 0.x development, patch releases preserve the documented Python API;
minor releases may introduce announced breaking changes. Changes and migrations
belong in CHANGELOG.md. Grouped-settings functions and their exported settings
classes are the preferred Python entry points. Underscore-prefixed helpers,
GUI internals, research scripts and numerical cache formats are implementation
details. After 1.0, incompatible documented API changes require a major version.
Numerical results may change after a physics correction even when signatures
remain compatible; release validation reports describe those changes.

The [bundle compatibility contract](../validation/BUNDLE_SPEC.md) applies to
released schema versions. Required-field additions or changed field meanings
require a new major schema version. Readers retain supported older versions for
inspection/playback. If missing metadata makes re-simulation ambiguous, readers
explain why it is disabled instead of silently assigning new semantics. Writers
must produce complete, unambiguous current conventions. New optional provenance
fields do not change the meaning of existing fields.

## Citation and release archive

[CITATION.cff](../CITATION.cff) contains citation metadata. Cite the exact
version and, when available, the **version-specific Zenodo DOI**. For a modified
checkout also record the commit, dirty-source hash and validation report.
No DOI has been assigned by this checkout; a DOI must not be invented.

The repository owner enables wisermaclab/hermes in the
[Zenodo GitHub integration](https://help.zenodo.org/docs/github/enable-repository/)
once. Subsequent GitHub releases are archived with their own DOI. Include the
validation report and its configuration with the release; retain reference ADC
archives outside Git. Verify the archive and version DOI after publication, then
add the actual DOI to that release's citation metadata. Existing GitHub releases
are not automatically backfilled by merely enabling the integration.

Zenodo supports CITATION.cff; a second .zenodo.json is deliberately omitted
to avoid overriding it with divergent metadata. See
[Zenodo's metadata precedence](https://help.zenodo.org/docs/github/describe-software/).

Before publishing a release:

1. Update the package, changelog, tutorial and citation version/date together.
2. Run analytic verification, the frozen-release regression and the measured
   fixture suite. Review failures and unverified cases; do not silently refresh
   the reference to make a failing comparison pass.
3. Attach the JSON/Markdown report, configuration, convergence figures and
   provenance. Store frozen ADC outside Git, in the release's evidence archive
   and an external durable artifact archive, with checksums.
4. Publish the reviewed GitHub release and verify its Zenodo version DOI.

The scientific-validation workflow attaches its evidence archive to a published
release only after its jobs pass. Enabling Zenodo and selecting a reviewed CI
reference artifact are repository-owner configuration steps.

These practices improve scientific reproducibility. They do not establish JOSS
eligibility: [JOSS requires an OSI-approved software license](https://joss.readthedocs.io/en/latest/review_checklist.html),
while HERMES currently uses PolyForm Noncommercial. The license is unchanged.
