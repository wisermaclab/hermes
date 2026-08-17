# Changelog

All notable changes to HERMES will be recorded in this file.

## 0.1.0 - 2026-08-17

### Added

- Added the local HERMES web GUI with reactive plate, trihedral-corner, and
  uploaded-human controls; RT/PO range-profile overlays; deterministic
  manifests; bundle-based comparison; separate simulation and radar-
  configuration tabs; editable FMCW/TDM settings; and distinct mouse-camera
  versus Shift-drag target manipulation. The simulation explorer includes
  shared-scale RT/PO 2-D angle-FFT maps with an interactive range-bin interval
  initialized to the target-centroid range. Radar yaw, pitch, and roll now
  drive both solvers and the scene pose indicators; antenna selection supports
  optional digitized TI profiles, an omnidirectional response, and a cosine
  response with configurable full 3 dB beamwidth. Cartesian target-position
  controls move the target anchor in range, lateral, and vertical coordinates
  and update RT/PO, scene manipulation, angle-bin selection, and exports.
- Simplified the GUI to three top-level workflows: Static Target, Dynamic
  Scenes, and Bundle Comparison. Moved shared TI hardware, active Tx/Rx
  element selection, FMCW waveform, antenna pattern, and solver controls into
  the header Settings dialog while keeping independent fixed-origin radar
  orientation controls in the two simulation workflows. Corrected the PEC
  preset label to identify ideal PO with rough/diffuse RT behavior.
- Made board changes refresh the active Tx/Rx element choices immediately.
  Dynamic Scenes now accepts an optional static-environment XML and uses
  AMASS-like pose archives, evaluated lazily through a separately licensed
  SMPL-family model, instead of accepting baked mesh sequences in the UI.
  Its preview now has synchronized numeric frame-entry and playback controls,
  with compact top-ray and overlay actions on the same row. Square
  mode checkboxes accept an inclusive source-motion frame window and any
  combination of full RT, coherent RT, human-only PO, and RT-calibrated Hybrid
  PO. Start/Stop controls provide cooperative cancellation at safe solver
  boundaries and report the active mode plus one-based frame progress. Mode
  descriptions moved into per-checkbox question-mark tooltips. Newly loaded
  motion defaults to one simulated frame, with the maximum source-frame index
  shown in the end-frame label. Results are grouped into resettable range-
  profile, range-time, and frame-selectable range-Doppler pages, with all
  selected modes shown on the same product page; Range-Doppler navigation is
  one-based and now includes explicit Previous/Next actions. Completed dynamic
  RT runs add an aggregated path-segment histogram, while Hybrid PO adds one
  shared-scale plot for human-only, human-blocked environment RT, human-to-
  environment, and environment-to-human range profiles. Completed runs can
  retrace the displayed simulated
  frame on demand to overlay top-K
  target-touching RT rays for selected RT modes and PO face-power shading for
  selected PO modes. The prepared bedroom shell is tightly bounded around the
  included walking trial, keeps the radar at its open edge, moves furniture
  outside the swept walking corridor, and uses modest rough-wall scattering
  for visible room returns. Dynamic exports preserve every selected mode's ADC,
  uploaded XML, selected-frame provenance, and source AMASS parameters alongside
  evaluated motion geometry.
- Added header-level solver settings for adaptive parent-facet PO quadrature
  and RT ray/path/depth budgets. Static GUI simulations now use bounded
  parent-face quadrature by default, persist the last browser-local solver
  configuration, and record all solver controls in exports.
- Bundle Comparison now supports selecting multiple simulation modes in one
  run, an optional channel-0-only TX/RX simulation, overlaid primary/candidate
  range profiles, material-aware scene hover labels, and compact single- or
  multi-mode ADC export. Its stop-confirmation buttons now use server-backed
  click events so confirmed cancellation reaches the active solver. Bundle
  paths, comparison controls, product navigation, and reloadable saved results
  now survive the page reload used to switch between dark and light themes.
- Theme switching now also preserves static-target geometry, pose, material,
  radar controls, uploaded human meshes, dynamic-scene placement and frame
  controls, AMASS and XML uploads, selected dynamic product/frame, and
  completed static/dynamic solver results. Small controls use session storage,
  uploads use browser-local IndexedDB, and large solver objects remain in a
  bounded server-side reload cache rather than being serialized into the
  browser.
- Added a Dynamic Scenes GUI scenario with a prepared bedroom or uploaded XML,
  AMASS-like motion, static-environment RT, dynamic-human
  PO, blocking/coupling component profiles, and loadable hybrid bundles.
  Static Target and Dynamic Scenes share fixed-origin TI hardware, FMCW, and
  antenna-pattern settings but keep separate radar orientations. Added on-demand
  strongest-path RT overlays, interaction-depth statistics, and strongest-
  facet PO diagnostics.
- Published the unified HERMES Bundle v1 `hermes` profile plus JSON Schemas for
  bundle, sensor, frame, and environment metadata. Measurements and simulations
  share one artifact-driven loader; GUI exports include raw RT/PO ADC,
  transformed target geometry, motion inputs, and scene XML.
- Added validation `--ignore-tdm-timing` support for one simultaneous selected-
  subarray simulation per mobility mode while retaining the full-cycle Doppler
  PRI.
- Enabled first-pose RT-reference calibration by default for validation
  `hybrid_static_env_po` runs, with an explicit calibration-policy option and
  cache invalidation when the policy changes.
- Added a documented, byte-pinned source-checkout motion fixture converted
  directly from CMU subject 105, trial 02. The fixture is not included in the
  wheel or Python source distribution.
- Initial HERMES public-release package metadata using the `hermes-radar-sim`
  distribution name and the existing `mmWaveRadar` import package.
- Public developer API documentation, third-party notices, dataset-neutral
  validation helpers, synthetic benchmark runners, analytic point-target
  synthesis helpers, and focused unit tests.
- GitHub Actions CI for whitespace checks, Python compilation, notebook-output
  checks, unit tests, packaging, and the audited public-tree boundary.
- Release-hardened source-checkout bundle adapters for RT-Pose and mmRadPose.
  Both consume a separately generated AMASS-like base-SMPL archive, keep
  dataset timing and radar mapping in the adapter, write portable metadata,
  and validate the completed bundle before publication.
  [SparseSMPLFit](https://github.com/wisermaclab/SparseSMPLFit) is one
  optional producer of the motion archive.
- Separated dataset-neutral bundle loading and integrity checks into
  `tools/bundle_prepare`. The `mmwave-check-bundle` command now verifies bundle
  structure and cross-file consistency independently of simulator validation.
- Added one limited, byte-pinned validation bundle for each provided reference
  adapter.
  Both contain measured ADC and a neutralized three-sample AMASS-like motion
  window but no SMPL model files or baked human meshes; the RT-Pose stereo
  images use opaque face masks and its limited scene retains only runnable XML.
- Added the RT-Pose environment fitter to the source-checkout bundle tools and
  made the RT-Pose adapter invoke this reviewed implementation directly instead
  of accepting an arbitrary environment-fitting program.
- Added a HERMES RT-Pose AWR2243 cascade reader, including TI-licensed adapted
  decoding and calibration portions, so bundle preparation produces the
  selected measured ADC cube without an ADC path, cache, or decoder argument.
- Integrated RT-Pose LiDAR/stereo background reconstruction and environment
  fitting into the bundle preparer. A reusable geometry directory, scene XML,
  or mesh NPZ may be supplied once for sequences recorded in the same
  radar-coordinate environment; every resulting bundle receives its own copy.
- Added laser-only mmRadPose environment preparation. The adapter can reuse
  fitted scene assets or fit a registered laser cloud, converts fitted geometry
  back to the mmRadPose radar axes, and never reads camera imagery.

### Fixed

- Added an explicit, default-off TDM acquisition policy. Simultaneous mode now
  retains every selected Tx/Rx channel at one-chirp slow-time spacing, while
  TDM mode evaluates physical transmitter slots, records per-channel sample
  times, and uses the full transmitter-cycle Doppler PRI.
- Propagated independent local-y and local-z wavelength-normalized URA spacing
  through aperture gridding, angle FFT axes, point clouds, and GUI products.
  Two-dimensional plots now use direction-cosine axes with coupled physical
  azimuth/elevation readouts instead of mislabeling off-plane angles. Grid
  metadata now consistently uses `y_lambda` for the horizontal/local-y axis
  and `z_lambda` for the vertical/local-z axis; this replaces the earlier
  ambiguous use of `y_lambda` for the vertical axis.
- Made the GUI launcher loopback-only unless an explicit unsafe remote override
  is supplied. Remote sessions cannot select arbitrary server paths, SMPL
  pickle roots, or custom scene XML, and browser uploads now have enforced
  file-count and byte limits.
- Clarified that the HERMES wheel is OS-independent while its simulator runtime
  depends on platform-specific Mitsuba and Dr.Jit distributions. Documented the
  CUDA preference, LLVM CPU fallback, current Ubuntu/Python CI coverage, and
  best-effort status of other operating systems.
- Documented the governance-only GPL bootstrap history and the current
  PolyForm/CC licensing of target-owned public-repository governance files.
- Preserved Dynamic Scenes mode selections and completed simulation products
  across light/dark theme reloads by transferring the live solver, mode, and
  result-cache state and keeping the dynamic result panes mounted.
- Kept Dynamic Scenes camera orbit and zoom responsive during motion playback
  by carrying the live camera through every mesh update and coalescing redraws
  while a camera gesture is active. High-rate motion previews are decimated to
  a source-dividing display rate near 20--25 Hz without changing simulation
  frame indices or timing.
- Made the light GUI theme fully legible with a darker interactive accent,
  opaque explanatory/status text, theme-aware scientific trace colors, and
  explicit high-contrast grids, axes, labels, and annotations in both 2-D and
  3-D Plotly views. The existing dark visual treatment is unchanged.
- Normalized closed static-human meshes to outward face winding before either
  solver uses them, fixing reversed PO incidence and renderer lighting after
  handedness-changing coordinate conversions. Static runs now retain compact
  RT face-hit counts, highlight RT-only and RT/PO-overlap facets in the human
  scene, and export those counts with solver diagnostics. Removed the duplicate
  strongest-path and strongest-facet Markdown tables from the GUI.
- Restored the rough PEC RT preset while keeping smooth aluminum specular-only,
  and separated the diffuse-energy coefficient from the
  backscattering-pattern mixture parameter.
- Removed the template-level busy spinner and restored persisted PO/RT solver
  settings through validated, versioned browser storage.
- Bound the Human-Only PO notebook and benchmark FMCW transmitter count to
  their single-channel hardware, preventing the default radar configuration
  from failing validation.
- Corrected negative-slope range FFTs, magnitude-domain CA-CFAR, TI CLI chirp
  scheduling, antenna-angle broadcasting, and existing scene-device updates.
- Corrected multi-chirp angle processing, transmitter-aware Doppler PRI
  defaults, point-target cache dimensions, and SMPL-H/SMPL-X pose slicing.
- Added strict validation for public waveform, radar, mesh, PO, and ADC inputs.
- Made benchmark NPZ loading pickle-free and corrected 3-D background-capture
  support.
- Made dataset-adapter NPZ loading pickle-free and prevented trial and bundle
  path traversal. RT-Pose preparation writes only the completed bundle and
  never modifies its measured capture or optional standard dataset-local cache.
- Removed machine-local paths and outputs from release notebooks and fixtures.

### Changed

- Accelerated adaptive PO without changing its numerical result by caching
  Tx-independent quadrature distances, fixed human-mesh face adjacency,
  immutable ADC fast-time tensors, and sorted incremental face-bank lookups,
  and by skipping unused aggregation progress accounting when no callback is
  installed.
- Made simulation-package exports lazy so geometry-only diagnostics do not load
  the Mitsuba/Sionna runtime stack.
- Hardened packaging metadata, CI release checks, API links, command-line error
  reporting, and third-party release documentation.
- Made the synthetic `cosine30` TI antenna model the default and moved
  digitized pattern data behind the optional `MMWAVE_TI_PATTERN_NPZ` input.
- Defined a deterministic, audited export boundary for the public
  `wisermaclab/hermes` mirror. Only the documented, exact-hash CMU conversion
  and two limited validation bundles may cross that boundary; all other
  third-party datasets and binary assets remain outside the public release or
  with their upstream providers.
- Dataset source files remain external; source-checkout preparation utilities
  operate only on user-provided local datasets.
- Kept SMPL fitting separate from HERMES. The RT-Pose adapter accepts an
  AMASS-like pose sequence from any compatible producer, reads frame mapping
  from RT-Pose `Train.json`, and owns alignment between archive-derived pose
  timing and the dataset's 10 Hz radar timing.
  [SparseSMPLFit](https://github.com/wisermaclab/SparseSMPLFit) is supported but
  not required.
- RT-Pose bundle preparation now selects the fitted segment containing the
  synchronized pose frame and writes one canonical `amass_sequence.npz`
  window. Poses, translations, betas, gender, coordinate fields, and producer
  provenance are copied without refitting; segment-local, absolute source, and
  radar-centered bundle timelines are retained explicitly. This is now the
  sole parameterized body-motion archive accepted by validation.
- RT-Pose preparation now imports activity and capture descriptors from
  `Data/filemeta.txt` and copies the synchronized original left/right camera
  PNGs into user-generated bundles, with explicit non-obfuscation metadata.
  The checkout root and its `Data` directory are both accepted as input.
- The mmRadPose adapter now consumes the same pose-only handoff, applies an
  automatically discovered dataset-local radar-to-SMPL alignment, and exports
  the parsed 4-by-4 source radar matrix in IWR6843AOPEVM 12-channel TX-major
  order.
- Kept RT-Pose and mmRadPose as reviewed reference adapters for the universal
  validation-bundle contract; removed redundant sequence and visualization
  utilities, and kept a single shared dataset-neutral mesh-baking helper behind
  the adapters.
- Reduced the RT-Pose preparation CLI to its five domain inputs plus output
  location and safe replacement controls. Participant mapping, ADC decoding,
  bundle naming, motion-window selection, and environment discovery are now
  deterministic.
- Reduced the mmRadPose preparation CLI to the corresponding five domain
  inputs plus optional reusable/laser geometry, output location, and safe
  replacement. Radar/alignment discovery, frame selection, bundle naming, and
  motion-window selection are deterministic; diagnostic artifact transfers and
  manual research controls were removed.
- Removed bundle-format loading from the scientific `validation` package.
  Validation now consumes the shared preparation contract and remains focused
  on measured-versus-simulated metrics and reports.
- README and bundled metadata normalized to avoid user-local filesystem paths.
- Adopted a split noncommercial license model: project-owned code uses
  PolyForm Noncommercial 1.0.0, project-owned documentation and data use
  CC BY-NC 4.0, and bundled third-party material remains expressly excluded
  under its recorded upstream terms. Commercial licensing inquiries are
  directed to Rong Zheng at `rzheng@mcmaster.ca`.
