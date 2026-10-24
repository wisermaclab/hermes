# RT reproducibility with Sionna RT

Sionna RT path tracing is **not bit-reproducible** when Mitsuba runs on
multiple CPU threads, even with a fixed `rt_seed`. Two identical runs return
the same strong paths, but a handful of weak paths that bounce diffusely and
then specularly differ from run to run. With one CPU thread the trace is
identical across runs and processes.

This affects everything HERMES computes by ray tracing: the `rt` solver and the
Hybrid PO room components `static_environment_unblocked` and
`static_environment_blocked`. PO, coupling, and the PO calibration gain are
effectively unaffected.

Measured on 2026-10-20 with sionna-rt 2.0.1, Mitsuba 3.8.0
(`llvm_ad_mono_polarized`), Dr.Jit 1.3.1, macOS 15.4 on an Apple M4 Pro
(14 LLVM threads), using [`configs/human_room.json`](../configs/human_room.json)
for one frame with `compute_backend="numpy"`.

## Symptom

Running the same configuration twice in one process:

| Component | Relative L2 difference between two identical runs |
|---|---|
| `human_po`, `human_env`, `env_human` | 0 to 1e-8 |
| `static_environment_unblocked` | 3e-3 |
| `static_environment_blocked` | 2.5e-2 |

The absolute error is the same in both room components (about 3.4e-5). The
blocked component is smaller, so its relative error is larger.

## What varies

Comparing the room trace (human detached) between runs:

- About 1,404 of about 1,410 paths are **bit-identical** in delay and in the
  coefficients for every Tx/Rx antenna pair.
- The order in which paths are returned differs, from the first path on. This
  does not change the synthesized ADC.
- Between 5 and 16 paths are present in one run and missing in another. All of
  them start with a diffuse bounce followed by one or more specular bounces
  (`D+S`, `D+S+S`, `S+D+S`), at 35–62 ns (about 5–9 m). Together they carry
  less than 1e-6 of the room path power.
- Because they add coherently, those paths move the per-channel coherent
  sum of path coefficients by up to 7.5e-4. That is enough to produce the
  ADC differences above.

## Cause

Sionna's shooting-and-bouncing candidate generator
(`SBCandidateGenerator._shoot_and_bounce` in
`sionna/rt/path_solvers/sb_candidate_generator.py`) stores a specular chain
only the first time it is seen. Each ray thread hashes its chain and
increments a shared counter table with an atomic `scatter_inc`. A chain is
kept only if the counter was zero before the increment.

After a diffuse bounce, many diffuse rays can feed the same follow-on specular
chain. Only the first thread to register that chain stores its path, so
**which diffuse scatter point is kept depends on thread scheduling**. A
different scatter point gives a slightly different delay and coefficient.
`rt_seed` fixes each ray's random samples but not the order in which threads
reach the shared table.

## Measurements

Four repeated room traces per setting:

| Threads | `rt_max_paths_per_source` | `rt_samples_per_source` | Paths | Bit-identical | Max per-channel coherent-sum difference |
|---|---|---|---|---|---|
| 14 | 5,000 | 100,000 | 1,408 | No (10–16 paths differ) | 7.5e-4 |
| **1** | 5,000 | 100,000 | 1,407 | **Yes** | **0** |
| 14 | 50,000 | 100,000 | 1,408 | No | 4.1e-4 |
| 14 | 500,000 | 100,000 | 1,407–1,408 | No | 6.9e-4 |
| 14 | 5,000 | 1,000,000 | 4,800–4,804 | No (750–1,090 paths differ) | 2.0e-2 |

- With one thread, the trace was also identical in two separate processes.
  With 14 threads, it was not.
- Raising the path cap does not remove the variation, so the cap is not the
  cause at the default sample count.
- A single-threaded trace of this room took 45 ms, compared with 20 ms on 14
  threads.

### Path-cap saturation

The last row shows a separate hazard. With 1,000,000 rays per source and the
default cap of 5,000, the trace returns about 4,800 paths, close to the cap.
Paths beyond the cap are silently dropped, and which ones survive also depends
on thread order, so the variation grows about 30-fold. When increasing
`rt_samples_per_source`, keep `rt_max_paths_per_source` well above the number
of paths returned.

## Recommendations

- **Reproducible CPU runs.** Restrict Dr.Jit to one thread before tracing:

  ```python
  import drjit as dr

  dr.set_thread_count(1)
  ```

  The setting applies to the whole process, including Mitsuba visibility
  rays used by PO and blocking, so those also run on one thread. HERMES does
  not set it automatically. The
  [configuration trade-off study](../benchmarks/configuration_tradeoffs/README.md)
  already runs its accuracy jobs on one LLVM thread for this reason.
- **CUDA.** The thread-count setting only applies to the LLVM (CPU) variant.
  The same race happens among GPU threads. GPU runs were not tested, but they
  should be expected to vary run to run.
- **Comparisons and tests.** Do not expect bit-identical RT or room
  components from multithreaded runs. Allow about 0.5% relative L2 on
  `static_environment_*` and RT-solver ADC, or average powers over seeds as the
  configuration trade-off study does. PO and coupling components can be compared
  tightly.
- **Path cap.** Check that the number of traced paths stays well below
  `rt_max_paths_per_source`, especially after raising `rt_samples_per_source`.

## Reproducing

```python
import json
import numpy as np
from mmWaveRadar.experiments.human_room import run_human_room_experiment

config = json.load(open("configs/human_room.json"))
for key in ("schema_version", "output", "quiet", "force", "solvers",
            "num_frames", "motion_end_s"):
    config.pop(key, None)
config.update(human_motion="data/AMASS/walking_poses_cmu_105_02.npz",
              smpl_model_dir="models/smpl_models", solver="hybrid_po",
              num_frames=1, compute_backend="numpy")

first, second = (run_human_room_experiment(**config) for _ in range(2))
for name in sorted(first.components):
    a = np.asarray(first.components[name])
    b = np.asarray(second.components[name])
    print(name, np.linalg.norm(b - a) / np.linalg.norm(a))
```

Run it from the `public/` directory. With 14 threads the room components
differ as in the table above. Calling `drjit.set_thread_count(1)` before the
runs makes every component, including both room components, identical.
