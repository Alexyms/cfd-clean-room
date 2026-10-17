# ADR-014 Step 0: The Housekeeping, the Warm Start and the Steady Checkpoint

**Date:** 2026-10-16
**Tree:** branch `phase4/step0-warm-start` from main at 1c58700 (PR 87, ADR-014 merged as
Proposed). The commits are listed in section 1 and in the pull request body.
**Instruments:** `docs/reports/probe60/` (`hash60.py`, `warm60.py`, `tables60.py`); records in
`results/builder60/` (untracked).
**Order:** this report's predictions (section 2) were committed before the two product runs of
section 4 and the hash runs of section 3 were started.

## 1. What the step built

ADR-014 section G, step 0, as prompt 60 sets it out:

- Commit 0, the records: ADR-014 accepted by Alex on 2026-10-16, each decision's recommendation
  as written; the amendment of the same date, the equipment vents as airflow only, as step 0b
  of section G and the design text of a new section J, with its limit in section H; SYSTEM.md's
  planned contracts; STATUS; PROJECT_PLAN's Phase 4 rows.
- The housekeeping batch of section I: the constants of tests 55 and 56 pinned;
  `largest_inlet_eddy_viscosity` removed with its tests and contract line; the docstring
  summaries over 88 characters shortened in the files the batch and the step touch;
  `field43.py`'s git lookup moved out of import; `scripts/preflight.py` added; the pipes inside
  table cells of the four ECR-002 reports escaped, at the reports and at the generator; the 16
  local branches merged into main deleted (the remote ones section I counted were already gone).
- The warm start: `FlowState` and `solve_steady(initial=...)` (ADR-014 A).
- The steady checkpoint and its bitwise continuation: `SteadyCheckpoint`,
  `solve_steady(resume=...)`, `ErrorEstimateRule.window` and `restore`, and the
  `/steady_checkpoint/` group of the run file in a new `src/io_manager.py` (ADR-014 D).

## 2. Predictions

**Orchestrator (prompt 60, 2026-10-16), as given.**

- (a) Every hash of record is unchanged with `initial=None`.
- (b) The self-warm-started product solve stops within one rule window (100 outer iterations)
  of its start.
- (c) The open-from-closed re-solve through the argument matches T1's warm arm to the outer
  iteration and its faces bitwise.

**Builder (2026-10-16, before the runs, scored apart).**

- (a) Holds. Nothing in the loop changed for a solve with neither keyword; the start from rest
  and a warm start from a state of rest give the same bits on both test fixtures
  (`tests/test_warm_start.py`), which a change on the shared path would break.
- (b) Holds at exactly 100: the rule starts fresh, so it cannot stop before its window is
  full, and both fixtures, warm-started from their converged states, stop at the first full
  window. The rebuilt nu_t differs from the relaxed one by less than condition (e)'s
  tolerance at a converged stop, so the first full window meets (e) as well.
- (c) Holds bitwise: the argument copies the same arrays the probe's patch returned and passes
  k and eps through the same `KEpsilonModel.state`, which `tests/test_warm_start.py` pins on
  the 40x15 room with a changed supply.

## 3. Prediction (a): the hashes of record

To be filled from the records.

## 4. Item 6: the warm start on the product room

To be filled from the records.

## 5. The tests and the planted defects

To be filled from the records.

## 6. What step 3 inherits

To be filled.

## 7. Deviations from the design text

To be filled.
