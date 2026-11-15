# Where a Run's Time Goes (prompt 84)

**Date:** 2026-11-15
**Prompt:** 84 (`docs/prompts/cc-prompt-84-profile.md`)
**Branch:** `phase6/profile`
**Status:** predictions committed; measurements not yet made.

Phase 6 ports the solver to C++ and then CUDA. The first design decision is how much moves: only
the hot inner loops, with Python driving every step, or the whole step, with the fields resident
on the GPU. This pass measures where a scenario run's time goes now, stage by stage, on the
equipment dust release on 80x30, 160x60 and 320x120, and counts each stage's memory traffic and
arithmetic from the code. Nothing under `src/` changes; the probe is `docs/reports/probe84/`.

## The predictions

Set by the orchestrator in prompt 84 on 2026-11-15, and committed alone before any run of this
pass.

- (a) On 320x120 the march takes more than 70% of a scenario run's wall time, flow solve
  included, and within the march the implicit solve is the largest single stage.
- (b) Within an outer iteration of the flow solve, the pressure correction takes more than half.
- (c) The cost per cell per step on 80x30 is at least five times that on 320x120 (Python's fixed
  cost per call dominates the small grid).
- (d) Every stage's arithmetic intensity is below one operation per byte, so every stage is
  memory-bound on the GPU in double precision.
- (e) The monitor, the arrival maps and the frame writes together take under 10% of the march.

The prompt's stop conditions: the calibration is not bitwise, or the wrappers' overhead is over
5% of a window's wall time. A failed prediction other than that is a finding.
