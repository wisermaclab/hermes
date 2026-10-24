# Model validity and limitations

HERMES 0.2.0 models idealized FMCW propagation and surface scattering. A numerically
converged result can still disagree with a measurement because of model error,
scene uncertainty or hardware effects. Regression agreement checks whether a
release changed the calculation; it does not establish physical accuracy.

| Solver or component | Assumptions and practical limits | Checks to report |
| --- | --- | --- |
| PO | Locally planar illuminated surfaces with a high-frequency physical-optics current approximation. The direct-target solver is single-bounce. It omits edge diffraction, creeping waves, resonances and internal multiple scattering. A corner reflector's triple bounce is not a direct-PO validation target. | Mesh density, integration mode, phase/curvature tolerances and visibility convergence. Compare a PEC plate to the far-field PO integral; distinguish this from a full-wave reference. |
| RT | Geometrical-optics paths and configured interaction types. Finite ray sampling can miss small targets and higher-order paths. Increasing maximum depth or the path cap alone does not establish convergence. | Ray-count sweeps, path caps, several independent seeds and uncertainty on seed means. Specular paths can be deterministic while diffuse contributions vary. |
| Hybrid PO | Environment RT plus target PO, blocking and selected coupling paths. Reflector selection and coupling truncation omit paths. An RT-derived gain transfers the assumptions and noise of that RT reference. | Disable calibration for numerical verification. Freeze gains fitted on held-out calibration poses before measurement evaluation. Report reflector caps, coupling options and the calibration record. |
| Materials | Ideal PEC, interface or slab approximations do not describe all roughness, coatings, anisotropy or layering. Nominal permittivity/conductivity may be inaccurate at the acquisition frequency. | State material model, frequency, thickness, parameter source and frequency dependence. Do not interpret a fitted gain as a material measurement. |
| Array | Phase-center mode shares a surface calculation across the aperture. Close targets or large apertures violate that approximation. | Retain runtime aperture warnings; compare compact subarrays or singleton element groups at the same settings. |
| Time and motion | Piecewise interpolation and refresh schedules approximate continuous motion. Target geometry must cover the complete acquisition, including physical TDM slots. | Preserve original timestamps and TX order. Sweep refresh cadence and displacement/phase triggers; include Doppler and phase checks. |
| Receiver | Ideal sampling and a brick-wall IF limit. Analog filtering, phase noise, IQ imbalance, gain variation, quantization and saturation are not generally characterized by the simulator. | Document IQ and timing conventions, units, board settings, channel calibration and experimental uncertainty. |

The closed-form plate comparison assumes a far-field observer and unit antenna
gains. The copper-plate experiment is at approximately 0.668 m and therefore must
use the finite-distance model, not the far-field plate formula. Its curves are
aligned once at 0° and evaluate relative angular response, not absolute gain.

The sphere's optical limit, σ = πa², is an asymptotic reference for ka ≫ 1; it is
not the exact Mie solution in the Rayleigh or resonance regions. Trihedral
formulas require the matching face geometry, boresight and multiple reflections.
The verification report explicitly identifies unsupported or unverified cases.

See [release verification](scientific_validation.md),
[RT reproducibility](rt_reproducibility.md), the
[MIT Lincoln Laboratory radar lectures](https://www.ll.mit.edu/media/6956), and
[NASA's PO/equivalent-current trihedral study](https://ntrs.nasa.gov/citations/19940020143).
