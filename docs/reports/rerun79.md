# The Product Grid Again on the Budgeted Solve (prompt 79)

**Date:** 2026-11-12
**Prompt:** 79 (`docs/prompts/cc-prompt-79-rerun-budgeted.md`)
**Branch:** `phase5/rerun-budgeted`
**Status:** predictions committed before any run. The runs, the tables and the findings
follow in later commits.

Prompt 78 (ADR-016) booked the implicit transport solve's leftover and budgeted it per step.
Every transport output made before that change is superseded wherever a step's share binds
(`docs/reports/conservation_budget.md`, section 7). This pass makes those runs again on main's
files, with the flow states reused from the runs they replace, and re-scores the tables built
on them with no change to any rule.

## The predictions

Set by the orchestrator in prompt 79 on 2026-11-12, before any run of this pass, and committed
alone before the first run.

- (a) Every run's budget, the leftover booked, closes within 1e-12 of its supply at every stored
  frame and class; every accumulated leftover stays within `f L S(t)` at every step; no run is
  refused.
- (b) Every flow state that converged before reproduces by its fingerprint; the two that did not
  converge end as they did in prompt 72.
- (c) On 320x120, every detection latency and every all-clear reading equals the Phase 5 gate's
  to within one time step, and the layouts' ranking at every sensor count is unchanged, on
  320x120 and on 80x30.
- (d) The sweep's conclusion holds: no sensor's cell is reached by any plume at any arm that
  converges, and every contour distance is within one cell of prompt 72's.
- (e) The 320x120 dust release's accumulated leftover at the end is below 1e-6 of its supply
  (4.36e-6 before the budget).
- (f) The clean-out's room recovery time and free-area 100:1 time are within 0.1% of the old
  values on each of the three grids.

The prompt's stop condition: a flow state that converged before does not reproduce stops the
queue, since the flow solve is not supposed to have changed. Anything else that goes wrong in
one run (a refusal, a crash, a missing input) is recorded, and the queue goes on with the rest.
