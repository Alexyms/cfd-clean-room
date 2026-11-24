# Phase 6 Step 4b: The Pressure Correction (prompt 93)

**Date:** 2026-11-24
**Prompt:** 93 (`docs/prompts/cc-prompt-93-pressure.md`)
**Branch:** `phase6/pressure`
**Status:** predictions committed, before any run.

This is the second half of Phase 6 step 4 (`docs/PROJECT_PLAN.md`, Phase 6, the steps table;
ADR-017 C, F and G; ADR-013 as amended 2026-10-14): `PressureCorrector` and `Multigrid` in the
core's `pressure` and `multigrid` modules. Nothing under `src/` or `configs/` changes.

## The predictions

Set by the orchestrator in prompt 93 on 2026-11-23, and committed alone before any run.

- (a) Every block of the first kind equals the reference bit for bit on every planted and captured
  input.
- (b) The core's pseudo-inverse is within 1e-12 of NumPy's on every planted and captured coarsest
  system, the singular ones included.
- (c) Every captured correction stops at the reference's CG iteration count, and its corrected u,
  v and p are within 1e-12 of their largest magnitudes.
- (d) A correction on the 320x120 state is at least five times faster than the reference's, same
  session.

## How they are scored (the builder, before any run)

The core is built under the project's flags (`CLAUDE.md`, C++ Coding Standards) and called
through its binding under WSL2 with the Linux environment, one BLAS thread. "The reference" is
`PressureCorrector` and `Multigrid` at `origin/main` (16bd048), unchanged.

- **(a).** The blocks of the first kind are: the five p' coefficients, the mass imbalance, the
  right-hand side (projected on a closed domain), `apply_operator`, the diagonal preconditioner,
  every level's five arrays and smoother, a weighted Jacobi sweep, `restrict`, `add_prolonged`,
  the projection of a preconditioned vector, the coarsest level's dense matrix, and the corrected
  u, v and p given p'. Holds when each equals the reference's in every bit (a float compared by
  its 64-bit pattern, so that 0.0 and -0.0 differ) on every planted case of the parity module and
  every captured correction. A case with a NaN in its input is scored on the NaNs' positions and
  the bits of every other value. One differing bit is a failure.
- **(b).** For each coarsest system, the largest difference between the core's pseudo-inverse and
  `np.linalg.pinv(matrix, hermitian=True)`, over the largest magnitude of NumPy's, is at most
  1e-12. One system beyond it is a failure.
- **(c).** For each captured correction, the core's CG count equals the reference's, and the
  largest difference of u, of v and of p, each over the largest magnitude of the reference's field
  of that name, is at most 1e-12. A whole V-cycle on the captured right-hand side is read on the
  same scale and reported beside it. One correction that misses is a failure; a count that differs
  is reported apart from a field that differs.
- **(d).** On the stored 320x120 product state (`results/builder81/product/equipment_dust.h5`),
  the arguments of the first outer iteration of a solve warm-started from it: the median of
  repeated `correct` calls over the median of repeated core calls through the binding, both in
  one process, after unrecorded warm-up calls. Three sessions, each recording the machine's state
  as prompt 90's probe records it. Session 1 is scored; 2 and 3 are reported beside it. Holds when
  the ratio is at least 5.0 in session 1.

A failed prediction is a finding, not a stop.
