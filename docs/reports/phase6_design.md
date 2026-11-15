# The Phase 6 Design Pass (prompt 85)

**Date:** 2026-11-15
**Prompt:** 85 (`docs/prompts/cc-prompt-85-phase6-design.md`)
**Branch:** `phase6/design`
**Status:** predictions committed before any run of this pass.

## The predictions

Set by the orchestrator in prompt 85 on 2026-11-15, and committed alone before the probe runs.

- (a) Scaling a tolerance by 1% moves no detection latency by more than one time step on either
  grid.
- (b) The 1e-15 inflow perturbation leaves every decision output unchanged and moves no field by
  more than 1e-10 relative.
- (c) The tolerance changes move fields by about the tolerances' size: no field moves by more
  than 1e-6 relative.

## How they will be scored (the builder, before the runs)

The reference is main's code and configuration unchanged on each grid. A variant is one change
to it: `solver.pressure_rtol` or `transport.diffusion_tol` times 0.99 or 1.01 (the four
tolerance variants), every velocity inlet's velocity times 1 + 1e-15 (the inflow variant), and
the reduction this pass chooses. Each variant is set against the reference on the same grid
and scenario.

- **A detection latency** is every one the system reports or a search reads: the file's four
  sensors' latencies from each event's onset per channel (the monitor), every non-SOLID cell's
  first time at or above per event and channel (the arrival maps, which are the latency of a
  sensor at that cell), and each searched layout's latency per unit. Two latencies are compared
  in steps: the difference in seconds over the step of the interval the later of the two falls
  in. A latency reached in one run and never in the other has moved by more than one step.
  (a) holds when no latency of any of the four tolerance variants moves by more than one step.
- **A decision output** for (b) is: every detection latency as above, compared by its step;
  every cell's clear time and the room's clear time per all-clear unit
  (`placement.clear_units`, frame-resolved), compared by its frame; the searched layouts' cells
  under both rules and every candidate pool, and the order of the four layouts (the file's,
  the surface search, the product layout, the free optimum) under each rule; and each class's
  verdict against REQ-T05 (the accumulated leftover and the residual within 1e-4 of the
  supply). "Unchanged" is equal by those comparisons; times themselves can move in their last
  bits, since the step follows the flow.
- **A field's relative difference** is the largest absolute difference over the cells over the
  largest magnitude of the reference's field: per flow state for u, v, p, k, eps and nu_t, and
  per class over every stored frame (the largest difference over the frames over the largest
  value over the frames). (b) and (c) are scored on the largest of these over every field,
  state, class, scenario and grid of the variants they name.

## A deviation known before the runs

The prompt names the dust release and the door leak on 80x30 and 160x60. The door leak's
open-door flow state does not converge on 160x60 at the file's inputs: it stopped at the
15,000 cap with a repeating residual in prompt 72 (`docs/reports/phase5_sensitivity.md`,
section 1.2) and again in prompt 79 (`docs/reports/rerun79.md`, F2), so that run cannot be made.
This pass runs the door leak on 80x30 only and adds the filter breach on both grids, so each
grid has a scenario with a boundary change beside the dust release, and the 80x30 placement
reads the same three runs that prompts 75, 79 and 81 searched on 80x30.
