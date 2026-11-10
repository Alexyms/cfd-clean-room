# The Budget Residual's Mechanism (Phase 4 Condition 1)

**Date:** 2026-11-10
**Prompt:** 77 (`docs/prompts/cc-prompt-77-budget-residual.md`)
**Branch:** `phase4/condition1-residual`
**Status:** predictions committed before any run; the measurements follow in a later commit.

Phase 4's gate passed on condition that the particle budget's closure residual
(`initial + inflow + source - outflow - deposited - current`, table B1 of
`docs/reports/phase4_scenarios_report.md`) is explained before Phase 6's finer grids. This
pass finds the mechanism by splitting each step's increment to the residual into the parts of
`TransportSolver.solve_timestep` that make it. It changes nothing under `src/`.

## The predictions

Set by the orchestrator in prompt 77 on 2026-11-10, before any run of this pass, and copied
here as stated. Run names are those of the prompt's table: R1 to R4 the four reference
scenarios on 80x30 at the file's `diffusion_tol` 1e-8, R5 and R6 the equipment dust and the
clean-out on 80x30 at 1e-6, R7 and R8 the same at 1e-10, R9 the equipment dust on 160x60 and
R10 on 320x120, both at 1e-8. T1 to T6 are the terms of section 1 of the prompt: T1 between
steps, T2 advection, T3 the SOLID zeroing after advection, T4 the source, T5 the implicit
solve's leftover (`dt` times the sum of the final Jacobi residual over the non-SOLID cells),
T6 the deposition booking.

- (a) **Closure.** In every run and class, T1 to T6 summed over the run equal the budget's residual
  at every stored frame, within 1e-6 of the largest residual magnitude over that run.
- (b) **The implicit leftover carries it.** At `diffusion_tol` 1e-8 (R1 to R4, R9, R10), at the
  run's end and at its worst stored frame, T5 is within 1% of the residual in every class, and T1 to
  T4 and T6 together are under 1% of it.
- (c) **Both signs, one mechanism.** T5's cumulative sum has the residual's sign at the end of R1 to
  R4: the door leak's positive, the other three negative. On R10 the 0.1 um class's cumulative T5
  changes sign in the same frame intervals as the residual (10.5 to 10.9 s and 25.0 to 25.4 s).
- (d) **Dose response.** On 80x30, in the dust release and the clean-out and every class, each
  decade of `diffusion_tol` moves the run's end residual by a factor between 3 and 30, smaller as the
  tolerance tightens. Under any mechanism other than T5 the residual moves by under 10% across the
  three tolerances.

The prompt's stop conditions: the in-process calibration (a wrapped and an unwrapped 2,000-step
window of the 80x30 dust release in one process) is not bitwise; (a) fails in any run; a solve
at 1e-8 reaches the 200-sweep cap; the wall-clock estimate exceeds the 3-hour budget. A failed
prediction other than (a) is a finding, not a stop.
