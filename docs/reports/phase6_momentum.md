# Phase 6 Step 4a: The Momentum Predictor (prompt 92)

**Date:** 2026-11-23
**Prompt:** 92 (`docs/prompts/cc-prompt-92-momentum.md`)
**Branch:** `phase6/momentum`
**Status:** predictions committed, before any run.

This is the first half of Phase 6 step 4 (`docs/PROJECT_PLAN.md`, Phase 6, the steps table;
ADR-017 B, C, F and G): `MomentumPredictor.predict` on the steady path in the core's `momentum`
module. The transient term (ADR-014 decision 3) is tier 2's and stays on the reference
(ADR-017 F), so the core refuses a `transient` argument. Nothing under `src/` or `configs/`
changes.

## The predictions

Set by the orchestrator in prompt 92 on 2026-11-23, and committed alone before any run.

- (a) Every planted and captured `predict` call equals the reference's bit for bit: u*, v* and
  both diagonals.
- (b) The set-up's fixed quantities equal the reference's per-call values bit for bit on every
  mesh.
- (c) `predict` on the 320x120 state is at least five times faster per call than the reference's,
  same session.

## How they are scored (the builder, before any run)

The core is built under the project's flags (`CLAUDE.md`, C++ Coding Standards) and called
through its binding under WSL2 with the Linux environment, one BLAS thread. "The reference" is
`MomentumPredictor` at `origin/main` (cb161c7), unchanged.

- **(a).** Holds when, for every planted case of the parity module and every captured call of
  section 3 of the prompt, u*, v*, a_P of u and a_P of v equal the reference's in every bit (a
  float compared by its 64-bit pattern, so that 0.0 and -0.0 differ). The planted and captured
  cases include the scalar path, a viscosity field, a wall viscosity, vents, one sweep and
  several. A planted case with a NaN in its input is scored on the positions of the NaNs and the
  bits of every other value, since a NaN's payload is the processor's. One differing bit is a
  failure.
- **(b).** The fixed quantities are the ones the core computes once at construction where the
  reference computes them in every call or holds them from its own construction: the QUICK
  weights of both directions with their boundary forms at the domain's ends and at obstacles,
  the obstacle walls' faces, distances and the unknowns they bound, the wall faces an unknown
  reads, the viscosity means' fractions, and the scalar path's conductances. Each is read from
  the core and compared with the value the reference forms inside a call (its functions' own
  local values, read as the function returns), on every parity mesh. Holds when no bit differs.
- **(c).** On the stored 320x120 product state (`results/builder81/product/equipment_dust.h5`),
  the arguments of the first outer iteration of a solve warm-started from it: the median of
  repeated `predict` calls over the median of repeated core calls through the binding, both in
  one process, after unrecorded warm-up calls, `mu_eff` and `wall_mu` given to both. Neither
  time includes `stencil_viscosity`. Three sessions, each recording the machine's state as
  prompt 90's probe records it. Run 1 is scored; runs 2 and 3 are reported beside it. Holds when
  the ratio is at least 5.0 in run 1.

A failed prediction is a finding, not a stop.
