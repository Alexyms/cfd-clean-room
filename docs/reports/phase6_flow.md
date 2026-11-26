# Phase 6 Step 5b: The Stopping Rule and the Outer Iteration (prompt 95)

**Date:** 2026-11-25
**Prompt:** 95 (`docs/prompts/cc-prompt-95-outer-iteration.md`)
**Branch:** `phase6/flow`
**Status:** predictions committed, before any run.

This is the second half of Phase 6 step 5 (`docs/PROJECT_PLAN.md`, Phase 6, the steps table;
ADR-011 G and ADR-012 D and E as built and amended; ADR-014 A and D; ADR-017 B, C, F, G and H):
`ErrorEstimateRule`, `viscosity_scale` and the imbalance summary in the core's `stopping` module,
and `StaggeredSolver.solve_steady`'s outer iteration in its `flow` module, with the fields resident
and the reference's set-up, refusals, checkpoints and log lines driving it from Python
(`tests/core_driver.py`). Nothing under `src/` or `configs/` changes.

## The predictions

Set by the orchestrator in prompt 95, and committed alone before any run.

- (a) Every block of the first kind equals the reference bit for bit on every planted and captured
  input, the median and the imbalance sums included.
- (b) Every captured estimate is within 1e-11 of the reference's, relative, and no captured window
  has the two estimates on opposite sides of `iteration_error_tol`.
- (c) Every whole solve stops for the reference's reason at the reference's outer count, not one
  apart, and every flow field is inside the measured spread (ADR-017 H's first tier).
- (d) The D = 3 instantiation builds and its tests pass.
- (e) The 160x60 product solve from rest is at least four times faster than the reference's, same
  session.

## How they are scored (the builder, before any run)

The core is built under the project's flags (`CLAUDE.md`, C++ Coding Standards) and called
through its binding under WSL2 with the Linux environment, one BLAS thread, NumPy 2.4.4 with its
AVX-512 loops off (`NPY_DISABLE_CPU_FEATURES=X86_V4`) in every parity run and every whole solve of
the reference (Alex, prompt 95). "The reference" is `src/stopping.py` and
`StaggeredSolver.solve_steady` at `origin/main` (e320545), unchanged.

- **(a).** The blocks of the first kind are: in `stopping`, the rule's verdict and the readings it
  records given the same steps, scales and imbalance (each condition, the window, `restore`, the
  version), `x @ x` of the rate fit, the logarithms of the window and the exponential of the
  slope, `viscosity_scale` (the median of nu_t over the FLUID cells, plus nu) and the imbalance
  summary (the worst absolute imbalance and the two sums, in NumPy's order); in `flow`, every
  elementwise write of an outer iteration (the outlets' extrapolation, the effective viscosity,
  the eddy viscosity's relaxation, the cell-centred velocities) and the max-norms (the velocity
  step, the viscosity step, the residual). A captured input is one the probe kept from the
  reference's own outer iterations (section 3 of the prompt); a planted one is the parity
  module's. Holds when each output equals the reference's in every bit (a float compared by its
  64-bit pattern) on every one. One differing bit is a failure, whatever its cause.
- **(b).** For each captured outer iteration whose window is full, the core's estimate formed from
  the reference's window (velocity, and with condition (e) on the viscosity's) against the
  reference's estimate: the relative difference `|core - reference| / |reference|`, two infinities
  equal. Holds when every one is at most 1e-11, and when no captured iteration has one estimate
  below `iteration_error_tol` and the other at or above it.
- **(c).** The whole solves are the four the prompt names: the 80x30 and 160x60 product room from
  rest, the 40x40 cavity from rest, and the door leak's open state warm-started from its closed
  one on 80x30, each side from its own closed state, as a run on that side would. For each: the core's stop reason and outer count equal the reference's, and each
  flow field's largest difference over its largest magnitude (u, v, p, and k, eps, nu_t where the
  solve is coupled) is at most that field's measured spread, the largest P2 measured for it
  (`docs/reports/probe85/tables85.md`, P2: u 1.2e-10, v 6.1e-11, p 9.5e-11, k 7.8e-10, eps
  7.5e-10, nu_t 6.7e-10). An outer count one apart, or a field between the spread and its limit,
  fails (c) as stated, and is then read against ADR-017 H's three tiers and item 3 for the gate.
- **(d).** The GoogleTest program of three directions (`core/tests/test_three_directions.cpp`)
  compiles with `stopping` and `flow`'s per-direction parts instantiated at D = 3, and every test
  of it passes.
- **(e).** The 160x60 product room's solve from rest, the reference's `solve_steady` and the
  core's driven solve, in one process, one after the other, the wall time of each whole solve.
  Three sessions, each recording the machine's state as prompt 90's probe records it. Session 1
  is scored; 2 and 3 are reported beside it, with the time in each stage. Holds when the
  reference's time over the core's is at least 4.0 in session 1.

A failed prediction is a finding, not a stop.
