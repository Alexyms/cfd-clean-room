# Phase 6 Step 2b: the Implicit Solve (prompt 88)

**Date:** 2026-11-19
**Prompt:** 88 (`docs/prompts/cc-prompt-88-scalar-implicit.md`)
**Branch:** `phase6/scalar-implicit`
**Status:** predictions committed; the port and the runs follow.

This is the second half of Phase 6 step 2 (`docs/PROJECT_PLAN.md`, Phase 6, the steps table;
ADR-017 C, F and G). It ports `src/scalar_scheme.py`'s `implicit_step` to the core's `scalar`
module: the set-up and its refusals, ADR-016's acceptance, the Jacobi solve with held cells,
the diagonal-preconditioned conjugate gradient solve with its restart, the fallback to Jacobi,
and the result. Nothing under `src/` or `configs/` changes.

## The predictions

Set by the orchestrator in prompt 88 on 2026-11-19, and committed alone before any run.

- (a) Every Jacobi solve, planted and captured, equals the reference bit for bit: field, sweeps,
  converged and leftover.
- (b) Every captured CG solve stops at the reference's iteration count, and its field is within
  1e-12 of the reference's largest magnitude.
- (c) On the developed 320x120 frame the core's Jacobi solve is at least ten times faster per call
  than the reference's.

## How they are scored (the builder, before the runs)

- **(a).** Bit for bit means the 64-bit patterns are equal, so 0.0 and -0.0 differ. The outputs
  compared are the field (every cell), the sweep count (`iterations`), `converged` and the
  leftover, and with them `method`, `fallbacks` and `fallback_sweeps`. A Jacobi solve is any
  call with `method="jacobi"`, and the Jacobi fallback of a CG call, whose field, sweeps and
  leftover are the Jacobi path's from `C*`. The planted inputs are the parity tests'
  (`tests/test_core_parity.py`); the captured inputs are every call the reference's transport
  march made in the probe's runs. The core is called through `_cfdcore` on the same arrays as
  the reference. The prediction holds when no output of any Jacobi solve differs; one differing
  bit is a failure, reported with the case.
- **(b).** For each captured k or eps call (`method="cg"`), the reference's `iterations` against
  the core's, and the field's largest absolute difference over all cells divided by the largest
  magnitude of the reference's field. The prediction holds when every captured CG call has the
  same count and a ratio at or below 1e-12. `converged`, `fallbacks` and the leftover are
  recorded beside it and do not change the score.
- **(c).** Per call means the median wall time of repeated calls, one thread, the core through
  the binding (its argument checks included, its scratch allocated once before the timing) and
  the reference's `implicit_step` as the transport solver calls it (Jacobi, with ADR-016's
  bound). The inputs: each class's last frame of the stored 320x120 dust release
  (`results/builder81/product/equipment_dust.h5`) as `C*`, with the volume, step, conductances,
  sink, SOLID mask, tolerance and cap of that class's last captured call, and the bound
  rescaled to the supply booked at the last frame (the bound is proportional to it). The
  prediction holds when the reference's median is at least ten times the core's for every
  class, in the first of three timed runs; the other two are reported beside it and do not
  change the score.
