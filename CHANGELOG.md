# Changelog

All notable changes to HERMES will be recorded in this file.

## 0.2.0 - 2026-10-23

- Add immutable release-reference comparisons, analytic PEC plate/sphere mesh
  studies, plate/corner RT ray-density diagnostics, per-seed uncertainty and
  per-fixture release reports, including relative copper-plate angular response.
- Preserve software, execution and input-hash provenance in native cubes,
  profiling archives, prepared bundles, validation caches and reports.
- Add citation metadata, scientific limitations, API/schema stability and a
  scientific-validation CI workflow. Generated references remain outside Git.
- Expand the simulation tutorial with scene selection, measurement bundle
  preparation and a TI copper-plate sensor example; improve navigation,
  dark-mode styling and layout on narrow screens.
- Declare typed dependencies for each GUI workflow and reject undeclared
  context fields during assembly; check the workflow contracts with mypy in CI.
- Share benchmark summaries, adapter publication/destinations, tensor conversion
  and endpoint mesh windows. Fold adjacent same-line string literals and retain
  all-NaN TDM diagnostics without missing-data warnings.
- Lead the Python tutorial and README with grouped experiment settings APIs.
- Add grouped settings APIs for human-room and static-target experiments,
  calibrated hybrid pipelines and PO channels. Existing keyword functions
  forward every option through compatibility adapters.
- Split GUI assembly and results/export construction into ordered stages;
  separate browser scripts from Python constructors. Extract per-run RT,
  PO, incremental-PO and coupling stages without changing their equations.
- Pin and configure pylint, run it in CI, and clean stale imports/type annotations.
- Consolidate portable path and archive checks, mesh-sequence views, adapter
  validation, radar construction, placement/default helpers and scalar/DSP
  operations while retaining their distinct timing and axis policies.
- Split validation into configuration, channels, simulation, cache and DSP-map
  modules. Move GUI callbacks into explicit session workflow classes and share
  scene-interaction JavaScript. Retain existing runner imports and keyword APIs.
- Extract RT scheduling/tracing/matching and PO calibration stages; group PO
  channel settings for the exact and incremental paths.
- Pin Black with a 99-character line length and check formatting in CI.
- Resolve legacy carrier references from source waveform fields, retaining
  conversion provenance; ambiguous carriers warn and disable re-simulation.
  New writers reject unverified carriers and incomplete motion, and motion
  preparation requires explicit frame timing. Check ADC sampling-mode/dtype
  consistency and document mmRadPose's nominal carrier approximation.
- Reuse simulation caches across ZIP extractions using relative resource names
  and content hashes. Reject omitted or reordered TDM slots in the core simulator
  with guidance to use the slot-aware validation runner.
- Keep GUI range profiles on a shared plot showing channel-mean power, with
  frame selection. Average chirps within each frame and hide values below
  −80 dB without drawing a clipped floor. Bundle views retain relative levels
  between frames. Add a per-channel plot for each simulated mode, with Hybrid
  PO's channels immediately below its averaged components.
- Fixed empty Dynamic Scenes content at launch when restoring the selected tab,
  including resizing the scene preview when it becomes visible. Preview pose
  controls now use seconds and interpolate between source poses.
- Made an empty tutorial room the GUI default, with a metal floor and
  plasterboard wall materials with the radar at the origin, 1035 mm above the
  floor. Motion loading now preserves file positions by default in the GUI,
  with manual placement available and X/Y/Z limits derived from scene dimensions.
- Added `mmwave-prepare-bundle` with direct ADC/sensor/scene/target arguments
  or `--config`, and the public GUI-independent
  `BundleWriter`, including portable scene assets and checked directory/ZIP
  publication. GUI exporters and reference adapters share the writer.
- Accept bundle ZIPs directly in `mmwave-validation-benchmark --clip` for
  simulation and measurement comparison, with checked temporary extraction,
  cleanup after the run, and source-archive provenance in the metrics.
- Added version 2 feature gates for explicit arrays, channel maps and physical
  TDM slot order while retaining version 1 read compatibility. Documented IQ,
  phase-sign, coordinate, amplitude and sampled-carrier conventions.
- Report full-acquisition motion coverage during loading, retaining playback
  while disabling re-simulation for incomplete motion. Checked scene
  aliases, removed the sensor-metadata scene fallback, and added structured
  `--all-errors` output plus warning-only `--diagnose` plots.
- Corrected TI CLI and RT-Pose carrier import to reference the first ADC sample
  (`start_frequency + slope * adc_start_time`), matching synthesis and wideband
  PO. This changes wavelength/phase/Doppler results from prior imports.
- Preserve acquisition-loop timing for stored TX subsets and for ignored TDM
  slot offsets. Preserve waveform provenance during writing and pre-existing
  empty destinations after failed publication. Accept epoch-valued and jittered
  timestamps in both bundle versions, with a declared absolute tolerance.
- Unified angle-map phase-sign removal and corrected GUI evaluated-motion
  exports to retain interpolation knots and the final physical chirp. Environment
  exports keep targets separate to prevent duplicate geometry on re-simulation.
  Thin static-target range gates retain overlapping FFT cells.
- Packaged an original synthetic static-target example and measurement/adapter
  tutorials; a redistributable measured static-target conformance capture
  remains a stated release milestone.

- Documented Sionna RT run-to-run variation on multithreaded CPUs in
  `docs/rt_reproducibility.md`: thread-order-dependent diffuse-then-specular
  paths, a one-thread workaround, and path-cap saturation when raising
  `rt_samples_per_source`.
- Reused physical Rx/Tx reflection-feasibility checks in active-set coupling
  projection before expanding to virtual channels. Finite-wall masks, phase,
  spreading, delays, and channel ordering retain their existing behavior.
- Fixed MPS backend selection on Torch builds missing the native complex
  cross-product kernel. Complex PO cross products now use equivalent
  elementwise operations on MPS, allowing supported float32 runs to select
  the accelerator instead of falling back to NumPy.
- Added Torch exact wideband PO integration for quadrature and analytic modes,
  including clipped-simplex IF responses. Analytic node integrals and
  leaf-to-parent reductions stay on the selected device; retained geometry and
  persistent response storage preserve incremental reuse. Integration follows
  `compute_backend`/`compute_precision`, with NumPy as the reference/fallback.
- Accelerated wideband PO synthesis by uploading and mixing only contributing
  parent-path columns, preserving original path IDs for partial IF clipping.
  Analytic integration now prepares triangle geometry once per chunk and batches
  frequency nodes. Coupling reflection geometry and cosine antenna patterns
  are evaluated per physical antenna; channel-specific patterns retain their
  individual curves and virtual-channel mapping.
- Matched PO response-grid Chebyshev tail tolerances to `compute_precision`:
  1e-8 for float32, 1e-14 for float64. Float32 results can change slightly as
  fewer frequency nodes are integrated; float64 keeps the existing tolerance.
- Unified RT baseline and solver execution, including human-bounce filtering,
  moving-body visibility refreshes, and fades. Removed the duplicate baseline
  loops and skipped unused cross-frame crossfade updates.
- Added last-exact area and normal guards to incremental PO, retained adequate
  wideband frequency grids across small mesh changes, and limited coupling
  acceleration-structure refits to human-visibility refreshes.
- Shared subarray execution and array restoration across Hybrid PO entry
  points, with one full-array reflector selection per run. Calibrated Hybrid
  PO now rejects groups without subarray mode, uses consistent PO defaults,
  and honors explicit visibility sample overrides. Cached coupling requires
  an explicit HybridPOConfig; legacy rt_baseline caches load as RT.
- Made coherent RT updates and transition matching honor `compute_precision`
  instead of forcing float64, allowing capability-checked float32 MPS execution.
  Coherent-bank tensors are cached separately for each device and precision.
- Centralized array backend/device selection, complex capability probes, and
  dtypes in `simulation/compute.py`. ADC, PO, wideband PO, and coherent RT honor
  the same device override; explicit Torch can use CPU, CUDA indices are
  preserved, and invalid settings fail early. Runtime summaries report the
  resolved `auto` backend and device for each precision.
- Made wideband PO and Hybrid PO ADC synthesis honor `compute_backend`, using
  PyTorch on an available accelerator in `auto` mode and retaining the NumPy
  CPU fallback, coherent IF-edge clipping, and bounded working buffers.
- Added `po_refresh_on_phase_error_rad` (default 0.1 rad) alongside configurable
  last-exact area and normal guards, `po_refresh_on_area_fraction` (default
  0.30, or 30%) and `po_refresh_on_normal_deg` (default 20 degrees). Direct-human
  PO and image-source coupling share these triggers, exposed in Python, CLI,
  JSON, and GUI settings. Removed the centroid-displacement trigger.
- Enabled adaptive PO visibility sampling by default in `POConfig` and
  `po_channel`, with grazing cosine margin 0.15. Multi-sample visibility results
  and benchmark numbers may change; set `po_adaptive_visibility_sampling=False`
  for uniform sampling. The benchmark suite compares both modes.
- Removed `human_multi_touch` as a calibration reference. Existing callers
  selecting it now receive a validation error; RT multi-touch diagnostics remain.
- Fixed human-room PO diagnostics to read adaptive visibility settings from
  the run manifest, with configuration defaults for older manifests.
- Made RT displacement-triggered retraces additive to the frame cadence.
  `rt_refresh_on_displacement` now uses metres, rather than wavelength
  fractions; inactive triggers omit the threshold from event metadata.
- Kept one-frame RT/PO refresh defaults and documented the cadence errors
  measured on the tutorial walk: PO within 0.4% at four frames, specular RT
  12–26% per frame at one frame and up to 100% at four.
- Antenna indices are now zero-based in every front end. The scene CLI's
  one-based `--tx`/`--rx` became `--tx-indices`/`--rx-indices`, validation's
  `--subarray-tx`/`--subarray-rx` became `--tx-indices`/`--rx-indices`,
  `BenchmarkRunConfig.subarray_*_indices` became `selected_*_indices`, and the
  GUI lists `TX0`, `TX1`, .... The removed flags stop with an error naming the
  replacement. Validation metrics and `subarray_adc.npz` report zero-based
  `selected_*_indices` and `channel_*_indices`; the simulated-ADC cache schema
  is now 7.
- Fixed independent coupling visibility cadence and GUI result fingerprints
  for Bundle Comparison and Hybrid-only settings.
- Added `array_model="subarray"`, which replaces the reserved `element` model.
  `array_tx_groups` and `array_rx_groups` split the active antennas into
  zero-based groups; each Tx-group/Rx-group pair is solved at its own centers
  by every solver and front end (Python, JSON/CLI `--array-tx-groups`/
  `--array-rx-groups`, GUI MIMO array settings), and the channels are merged
  back into one cube. Calibrated PO parts share one gain across the pairs.
  `PhaseCenterApproximationWarning` now suggests group counts.
- RT now traces from the Tx- and Rx-array centers instead of the board
  origin, matching PO, the RT path bank, and the documented phase-center
  model. Specular RT results move slightly (0.2% on the tutorial IWR6843AOP
  run); with diffuse scattering, the sampled rays change as with a new seed.
  Radar traces no longer include the direct Tx-to-Rx path, which appears once
  the two centers differ.
- Fixed RT path diagnostics failing on scenes with no RT paths.

## 0.1.0 - 2026-08-17

### Added

- Added the headless Dynamic Scenes runner (`hermes-human-room` or
  `tools/run_human_room.py`) with current GUI radar/solver defaults, full
  simulation-setting overrides, JSON configuration save/load, custom XML scenes,
  human placement, zero-based frame windows, multiple modes, and protected,
  GUI-loadable bundle export. The source script reads the complete
  `configs/human_room.json` setup by default, including output and logging options.
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
  subarray simulation per solver while retaining the full-cycle Doppler
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

- Fixed authenticated Chrome WebSocket connections when Fetch Metadata is
  omitted on an upgrade, while retaining signed-cookie and Origin checks.
- Fixed the theme button's unmounted path references and restoration callbacks
  that could invalidate cached products. Completed workflow results now retain
  settings fingerprints, remain visible after edits, and show named changes
  with Rerun/Revert actions and a stale marker on the workflow tab.
- Staged radar and solver edits behind one Apply/Cancel action, added an optional
  active-page rerun, and kept human motion coordinates in Settings with the same
  Apply/Cancel behavior. Dynamic simulation modes appear in two rows. Bundle
  Comparison now explicitly identifies its bundle-owned solver settings and
  retains selected simulation modes across theme reloads.
- Corrected the radar summary to use sampled bandwidth, disabled Start until
  motion is loaded, made idle Stop controls neutral, and cleared progress panes
  on failures without double-escaping error messages.
- Kept Start disabled during dynamic solver overlays and rejected overlapping
  dynamic work. Stopping before the first mode completes now replaces the
  progress pane with a stopped message while preserving any previous products.
- Saved ADC frame, chirp, and channel selections immediately and captured their
  current browser values before a theme reload.
- Made Scene and motion the Dynamic Scenes landing tab on page loads and theme
  reloads, while retaining completed products in Simulation results.
- Restored the original workflow layout, control widths, directly visible tips
  and overlay controls, and expanded, stacked Bundle Comparison sections. Sized tabs to
  active content. Improved both themes,
  section headings, 3-D axis labels, numeric precision and icon accessibility;
  styled the launcher login page to match the GUI.
- Redesigned the launch screen with an animated radar-scene illustration,
  responsive HERMES branding, keyboard-friendly sign-in, and inline password
  errors. The illustration respects reduced-motion preferences.
- Added ordered playback controls with a frame counter, ADC frame/chirp/channel
  selectors, readable run summaries with advanced JSON, and actionable refinement
  warnings.
- Fixed Tabler playback icon sizing and Reverse orientation, redrew stale tab
  headers, labelled stale results as LAST RUN (out of date), and standardized
  frame selectors and counters on zero-based indices. Added dark secondary
  buttons, a secondary motion-preview action, sorted stage timings, and a sticky
  Static Target scene.
- Grouped position and orientation into compact numeric rows on both scene
  pages, tightened upload and mode help spacing, and inset mode checkboxes.
  Reverse now matches the outline Play icon. Uploaded motion previews
  automatically; the default-motion action disappears once motion is loaded.
- Put all three workflows on one layout. A control rail with its own scroll
  keeps every control visible in headed groups, with a jump bar to each group.
  Beside it, a sticky run bar shows the page title and help, the run status,
  the Start/Stop or higher-fidelity actions, and the stale-result banner with
  Rerun/Revert above the plots. Bundle Comparison steps moved into the rail, so
  its plots start at the top. Narrow windows stack the rail above the run bar
  and plots.
- Reorganized the Settings dialog into Radar configuration, Solver settings,
  and Motion coordinates tabs. Settings are grouped into headed sections laid
  out in columns, each noting the workflows it affects. A Derived card keeps the
  channel count, bandwidth, range resolution, and maximum range beside the
  radar form. Apply, Cancel, and Apply & rerun stay pinned to the bottom of the
  dialog. The dialog keeps one size on every tab, so switching tabs no longer
  moves it, and a taller tab scrolls inside it.
- Fixed an intermittent theme-switch failure that dropped restored Dynamic
  Scenes result tabs and stale-result banners with a browser "reference isn't
  known" error. The browser reports its saved session before the page is ready.
  Panel then holds the restored result views unsent, and a browser message in
  that window made Bokeh treat them as already delivered. Saved sessions and
  uploads are now restored once the page is ready.

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

- Dynamic Scenes, the headless runner, and `run_human_room_experiment()` now
  select motion by time. `motion_start_s` aligns radar frame 0 with a motion
  time, the motion plays in real time, and the run covers every radar frame
  whose chirps finish by `motion_end_s` (or `num_frames` radar frames). The
  former pose-index window mapped one motion frame onto each radar frame, which
  slowed motion and Doppler by the ratio of the frame period to the motion
  frame interval (six times for 120 fps motion at 50 ms). The GUI shows
  **Start [s]** and **End [s]** with the resulting radar frame count; the CLI
  replaces `--frame`, `--begin-frame`, and `--end-frame` with `--start-s` and
  `--end-s`, and `--num-frames` now counts radar frames. JSON configurations
  move to `schema_version` 2, and version-1 files are rejected with conversion
  instructions. Manifests record `motion_start_s`, `motion_end_s`,
  `motion_pose_count`, and `motion_duration_s`; exported bundles record
  `motion.time_window`, and each `frames.json` entry gives the radar frame's
  start on the motion's own timeline.

- Unified solver and setting names across the GUI, headless runner and JSON,
  Python API, validation, benchmarks, notebooks, and bundles
  (`mmWaveRadar.naming`). The solvers are `po`, `rt`, and `hybrid_po`, which
  replace `human_only_po`/`target_po`, `full_rt`/`coherent_rt`/`rt_retrace`/
  `rt_coherent_bank`, and `hybrid_static_env_po`. Unprefixed settings apply to
  every solver (`array_model`, `tx_schedule`); `rt_*` settings apply wherever RT
  runs, `po_*` wherever PO runs, and `hybrid_*` only to Hybrid PO. Removed
  names raise an error naming their replacement, including the old CLI flags.
- Temporal cadences are counted in radar frames and applied at frame starts:
  `rt_refresh_frames` (replaces the RT mobility modes and retrace-period
  options; `0` retraces every chirp), `po_refresh_frames` (replaces
  `incremental_update` and `incremental_full_refresh_chirps`), and
  `po_visibility_refresh_frames`/`po_visibility_fade_frames`. Hybrid PO coupling
  follows the PO cadence. The RT path-freezing mode was removed: between
  retraces, paths are always phase-updated on the moving mesh.
- Added RT path-visibility refresh between retraces
  (`rt_visibility_refresh_frames`, `rt_visibility_fade_frames`). Hybrid PO body
  blocking of room paths is always on and follows the same cadence.
- `tx_schedule` (`simultaneous` or `tdm`) replaces the `tdm_enabled` flag.
  Bundles now write `tx_schedule`; readers still accept `tdm_enabled` and the
  legacy TDM markers.
- Added `array_model` (`phase_center`; `element` is reserved and raises
  `NotImplementedError`). Phase-center runs emit
  `PhaseCenterApproximationWarning` when the RT curvature or PO facet phase
  error for the aperture exceeds its limit.
- `rt_human_bounces` (`single` or `any`) replaces `coupling_mode`.
  `hybrid_calibration` (default `rt_first_pose`) replaces the PO calibration
  options of Hybrid PO in every entry point; the PO solver keeps
  `POConfig.po_calibration` (default `none`).
- Renamed `HumanPOMobilityConfig` to `POConfig` and `HybridStaticEnvPOConfig`
  to `HybridPOConfig`, with `po_*`/`hybrid_*` fields. `SensingMetadata.mode` is
  now `SensingMetadata.solver` (schema 2). Experiment manifests move to schema 2
  with a `solver.solvers` list; Static Target records `["rt", "po"]`. Headless
  JSON configurations move to `schema_version` 3 with a `solvers` list.
  GUI bundle exports key each ADC by solver (`simulated_adc_<solver>.npz`) and
  store the Hybrid PO components in `hybrid_po_components.npz`, replacing the
  `simulated_adc_rt.npz`/`simulated_adc_po.npz` component aliases.
- Renamed the remaining helpers and diagnostics that used the earlier solver
  names. The full-scene RT reference is `run_rt_baseline()` returning
  `RTBaselineResult` (was `run_full_rt_mobility_baseline()` and
  `FullRTMobilityBaselineResult`; the `run_rt_mobility_baseline` alias was
  removed). Runtime-profile stages are `rt_baseline_*` for that reference and
  `rt_path_bank_*` for the RT solver's path bank, without the earlier duplicate
  `full_rt_*`/`rt_*` aliases. The RT schedule profiler now works in
  `rt_refresh_frames` (`rt_retrace_schedule_indices()`,
  `recommend_rt_refresh_frames()`), and prepared bundles key their
  `evaluation_notes` by solver.
- Renamed the demo notebooks `Human-Only-PO.ipynb` and `Full-RT.ipynb` to
  `PO.ipynb` and `RT.ipynb`, and the `human_only_rt_po_range_profile`
  benchmark to `rt_po_range_profile`.

- Reused the direct-human PO incremental facet updater for both image-source
  coupling directions. Coupling now caches surface terms per reflector,
  updates phase/spreading/delay each chirp, and shares geometry-threshold and
  receiver IF-edge refresh rules with direct-human PO. Runtime metadata counts
  reintegrated and phase-updated coupling facets.
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
