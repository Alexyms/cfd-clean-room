# What ISO's Cumulative Counts Do to Detection (prompt 80)

**Date:** 2026-11-14
**Prompt:** 80 (`docs/prompts/cc-prompt-80-cumulative-counts.md`)
**Branch:** `phase5/cumulative-counts`
**Status:** predictions committed before any run or read of the stored frames. The probe, the
tables and the findings follow in later commits.

ISO 14644-1's limits are cumulative: the limit at 0.3 um counts every particle at or above
0.3 um, and a real counter's size channels report those cumulative counts. The monitor and the
arrival maps compare each size class's own concentration with the limit for its size, which
ADR-015 never examined. This pass measures what the cumulative reading does to detection
latency and to the layouts' ranking, on the stored 320x120 frames of prompt 79 and on short
step-resolved reruns of prompt 79's three 80x30 placement runs. Nothing under `src/` changes.

## The predictions

Set by the orchestrator in prompt 80 on 2026-11-14, and committed alone before any run of this
pass or any read of its inputs.

- (a) The file's four sensors still detect no unit on either grid.
- (b) On 320x120 the product layout's worst-case latency is unchanged or earlier by at most one
  frame (0.36 s), because its worst unit is already the breach's, set by how far the plume must
  travel rather than by how much it carries.
- (c) The order of the four layouts by worst case is unchanged on both grids.
- (d) On 80x30, at step resolution, at least one unit's latency at the searched layout falls by
  more than 10%.

The prompt's stop conditions: the 80x30 calibration (the probe run's frames, budgets, readings
and per-class arrival maps against prompt 79's run) is not bitwise; the structural check
(a channel's cumulative map is never later than its class's, and the largest class's maps are
identical) fails on the real maps; or a cumulative latency is later than its per-class one
anywhere. A failed prediction is a finding, not a stop.
