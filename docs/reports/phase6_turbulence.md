# Phase 6 Step 5a: The Turbulence Model and the Wall Functions (prompt 94)

**Date:** 2026-11-25
**Prompt:** 94 (`docs/prompts/cc-prompt-94-turbulence.md`)
**Branch:** `phase6/turbulence`
**Status:** predictions committed, before any run.

This is the first half of Phase 6 step 5 (`docs/PROJECT_PLAN.md`, Phase 6, the steps table;
ADR-012 A to D as built and amended; ADR-014 J; ADR-017 C, F and G): `KEpsilonModel.step` on the
steady path and `TurbulenceBoundary`, in the core's `turbulence` and `walls` modules. Nothing under
`src/` or `configs/` changes. The true-time step of VAL-015 (a float `dt`) is tier 2's and stays on
the reference (ADR-017 F); the core refuses it.

## The predictions

Set by the orchestrator in prompt 94 on 2026-11-25, and committed alone before any run.

- (a) Every block of the first kind equals the reference bit for bit on every planted and captured
  input, the UMIST advection of k and eps included.
- (b) Every captured CG solve of k and eps stops at the reference's count, its field within 1e-12
  of its largest magnitude, and no captured step falls back where the reference does not.
- (c) The D = 3 instantiation builds and its tests pass.
- (d) A turbulence step with its conditions on the 320x120 state is at least five times faster than
  the reference's, same session.

## How they are scored (the builder, before any run)

The core is built under the project's flags (`CLAUDE.md`, C++ Coding Standards) and called
through its binding under WSL2 with the Linux environment, one BLAS thread. "The reference" is
`KEpsilonModel` and `TurbulenceBoundary` at `origin/main` (1bca2d8), unchanged, as this
environment's NumPy 2.4.4 runs them.

- **(a).** The blocks of the first kind are everything elementwise: in `turbulence` the eddy
  viscosity, the strain squared and the corner gradients it averages, the production, RNG's R, the
  two growths and the two decays, the pseudo-time step, each quantity's advected field and its
  explicit field, and the two pairs of conductances; in `walls` the wall viscosity on both
  components' wall stencils, the held eps and the given production, the four inflow arrays and the
  four edges' tangential velocities; and the set-up's fixed arrays and scalars (the wall faces and
  cells, the inlet values, the initial values, `log_law_floor`). Holds when each equals the
  reference's in every bit (a float compared by its 64-bit pattern, so that 0.0 and -0.0 differ) on
  every planted case of the parity module and every captured call. One differing bit is a failure,
  whatever its cause.
- **(b).** For each captured CG solve, of k and of eps, the core's solve of the captured system stops
  at the reference's count, and the largest difference of the field over the reference's largest
  magnitude is at most 1e-12; and no solve of the core's falls back where the reference's did not. A
  count that differs is reported apart from a field that differs. The whole step's own differences
  (k, eps and nu_t, each over its largest magnitude) are reported beside it and not scored
  (REQ-N03 as amended 2026-11-25).
- **(c).** The GoogleTest program of three directions (`core/tests/test_three_directions.cpp`)
  compiles with `turbulence` and `walls` instantiated at D = 3 and every test of it passes.
- **(d).** On the stored 320x120 product state (`results/builder81/product/equipment_dust.h5`), the
  arguments of the first outer iteration of a solve warm-started from it: the median of repeated
  `conditions` then `step` calls of the reference over the median of repeated core calls through the
  binding doing the same, both in one process, after unrecorded warm-up calls. Three sessions, each
  recording the machine's state as prompt 90's probe records it. Session 1 is scored; 2 and 3 are
  reported beside it. The CG solves are timed apart and reported. Holds when the ratio is at least
  5.0 in session 1.

A failed prediction is a finding, not a stop.
