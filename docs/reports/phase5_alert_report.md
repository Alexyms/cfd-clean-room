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
