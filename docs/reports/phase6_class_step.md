# Phase 6 Step 3a: One Class Step in the Core (prompt 90)

**Date:** 2026-11-20
**Prompt:** 90 (`docs/prompts/cc-prompt-90-class-step.md`)
**Branch:** `phase6/class-step`
**Status:** predictions committed; nothing built or run yet.

This is the first half of Phase 6 step 3 (`docs/PROJECT_PLAN.md`, Phase 6, the steps table;
ADR-017 B, C, E, F and G): one particle class advanced one time step in the core's `transport`
module, as `TransportSolver.solve_timestep` advances it, with the budget booked as
`MassBudget` books it and the breach tested as ADR-016 tests it. Python keeps the
configuration, the interval plan and the events, and reads the budgets. Nothing under `src/`
or `configs/` changes.

## The predictions

Set by the orchestrator in prompt 90 on 2026-11-20, and committed alone before any run.

- (a) Every captured and planted class step equals the reference's bit for bit: the field, every
  budget term, the sweeps and the leftover, vents and recirculation included.
- (b) The interval set-up's quantities equal the reference's per-step values bit for bit on every
  mesh.
- (c) A class step on the developed 320x120 frame is at least five times faster than the
  reference's, same session.

## How they are scored (the builder, before any run)

The core is built under the project's flags (`CLAUDE.md`, C++ Coding Standards) and called
through its binding under WSL2 with the Linux environment, one BLAS thread. "The reference" is
`TransportSolver.solve_timestep` at `origin/main` (173496d), unchanged.

- **(a).** A class step's outputs are: the returned field; the budget's ten terms after the step
  (`initial`, `inflow`, `outflow`, `source`, the four `deposited` slots, `current`,
  `leftover`); the step's records (`last_diffusion_sweeps`, `diffusion_converged`,
  `last_leftover`, `last_leftover_bound`); and, where the reference raises
  `ConservationBreachError`, the error's four arguments with the budget it leaves. The
  prediction holds when, for every planted case of the parity module and every captured call of
  section 3, the core's outputs equal the reference's in every bit (a float compared by its
  64-bit pattern, so that 0.0 and -0.0 differ), and the captured and planted cases include at
  least one step with vents and a recirculating discharge. One differing bit is a failure.
- **(b).** The interval set-up's quantities are the ones the core computes once per interval
  where the reference computes them every class step: per class, the advecting volume fluxes
  (the reference's `u_adv * area`), `stable_dt`, the conductances, the deposition per face times
  `dt`, `dt` times the source rate and the source's total, the recirculating intakes' drawn flux,
  and the masks the step reads (the domain faces' inward and outward ones, the vents' sides);
  and, for the interval, the face eddy viscosity. Each is read from the core and compared with
  the value the reference forms inside `solve_timestep` on the same inputs, on every parity mesh
  (the five of prompt 87 and its three vent meshes) and at every captured call. The prediction
  holds when no bit differs on any of them.
- **(c).** On the stored 320x120 dust release's last frame (prompt 87's developed fields), each
  class stepped once under its last interval's conditions: the median of repeated
  `solve_timestep` calls over the median of repeated core steps, both in one process, after
  unrecorded warm-up calls. The core's step is the binding's `step` call with the interval set
  up beforehand; the reference's is the whole call, its per-step checks included, since that is
  what the march calls. Three sessions, each recording the machine's state as prompt 89's probe
  records it (Windows' busiest processes, the power state, the processor's performance and a
  NumPy calibration). Run 1 is scored; runs 2 and 3 are reported beside it. Holds when the ratio
  is at least 5.0 for every class in run 1.

A failed prediction is a finding, not a stop.
