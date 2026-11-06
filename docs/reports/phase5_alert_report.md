# Phase 5 step 5: the latency report

**Date:** 2026-11-05 (the machine's local date)
**Branch:** `phase5/step5-gate` (prompt 76, `docs/prompts/cc-prompt-76-step5-gate.md`)
**What this is:** ADR-015 section E: the four reference scenarios on 320x120 with their arrival
maps and alert tables, the layouts scored and searched on them, and what the ranking can claim.
The runs are made by `docs/reports/probe76/run76.sh`, the records stay in `results/builder76/`
(untracked, this machine), and every table is generated from the run files by
`docs/reports/probe76/tables76.py`, whose `--check` regenerates and compares.

## Predictions, committed before the runs

The orchestrator's (2026-11-05, prompt 76), copied verbatim:

- (i) On 320x120 the file's four sensors detect none of the event scenarios' units.
- (j) With the doorway rule of item 0, no mountable candidate detects a door unit on 320x120
  either: the corridor air is taken by the first return before it reaches a wall a sensor can
  be mounted on.
- (k) The searched mountable four detect every breach and dust unit on 320x120, with a worst
  case under 3 s.

<!-- tables76 reload begin -->
<!-- tables76 reload end -->

<!-- tables76 flows begin -->
<!-- tables76 flows end -->

<!-- tables76 states begin -->
<!-- tables76 states end -->

<!-- tables76 march begin -->
<!-- tables76 march end -->

<!-- tables76 budgets begin -->
<!-- tables76 budgets end -->

<!-- tables76 alerts begin -->
<!-- tables76 alerts end -->

<!-- tables76 layouts begin -->
<!-- tables76 layouts end -->

<!-- tables76 curve begin -->
<!-- tables76 curve end -->

<!-- tables76 headline begin -->
<!-- tables76 headline end -->

<!-- tables76 units begin -->
<!-- tables76 units end -->

<!-- tables76 recovery begin -->
<!-- tables76 recovery end -->

<!-- tables76 stability begin -->
<!-- tables76 stability end -->

<!-- tables76 arms begin -->
<!-- tables76 arms end -->

<!-- tables76 predictions begin -->
<!-- tables76 predictions end -->
