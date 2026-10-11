# ECR-002 Step 7: Condition (e)'s Scale, and VAL-017 as a Harness Row

**Date:** 2026-10-11
**Tree:** branch `feature/ecr002-rule-e-scale` from main at e281629 (prompt 49 merged as PR 76).
Commit 0 records Alex's decisions of 2026-10-11 (ADR-012 decision 6 and section E, amended).
Commit A builds the scale: `ErrorEstimateRule(..., viscosity_condition=True)` takes (e)'s scale
with every step, and the staggered solver passes the molecular nu plus the median of nu_t over
the FLUID cells of the iterate just computed (`src/stopping.py`, `viscosity_scale`); rule version
5. Commit B runs CI's suite once. Commit C moves the VAL-017 run to the harness row
`val017_216x72`, scored by the allowance (`validation.annex20.score_lines`,
`scripts/benchmark.py`).
**Instruments:** `docs/reports/probe50/`: `plant50.py` (the planted defects of commits A and C),
`check50.py` (the laminar hashes), and, after this section was committed, the runs and the
tables. Raw output under `results/builder50/` (untracked).
**Order:** sections 1 to 4 were committed before any run of items 6 and 7, this file alone.
Section 5 onward was written after the runs.
**Policy:** `docs/REVIEW_POLICY.md`, "What blocks". The numbers live in generated tables; the
sentences point at them.

## 1. The questions

Whether the stopping rule can stop a ventilated room whose inlet turbulence is weak. Prompt 49
showed it cannot while condition (e) is scaled by the inlet's eddy viscosity (2.2 nu on Annex
20) and the room makes several hundred nu (`docs/reports/ecr002_val017_annex20.md`, section
5.2). VAL-019 and VAL-020 have weak or no inlet turbulence too. With the scale changed to the
room's median: does the Annex 20 room stop, at which k and eps tolerance, and does it pass the
allowance; and does the product room's stop move?

## 2. Method (fixed before the runs)

### 2.1 Item 6: the Annex 20 room

The harness row `val017_216x72` as `scripts/benchmark.py` runs it (`run_case`), on the committed
case file, under rule version 5, twice: once at the file's `turbulence.tol` (1e-10) and once at
1e-13, each with `max_simple_iter` 20,000 and every other setting the file's (the k and eps
`max_iter` stays 500; prompt 49's 1e-13 diagnostics used 2,000). The run is from rest, one BLAS
thread, the rule replaced in the probe by a recording subclass that asks for the imbalance every
outer iteration and keeps the readings and the scales it was passed (the decision is the
committed rule's, on the same readings). Reported for each: whether it stops, the outer count,
the readings at the stop and the iteration from which each condition holds, the RMS on each line
against the measurements beside the allowance, and the largest difference of the four lines
from prompt 49's records of the same case (the 1e-13 diagnostic at its 12,000 cap and the
file's row at its 10,000 cap). If only the tighter tolerance stops, the case file takes
`turbulence.tol` 1e-13 and `max_simple_iter` the next round thousand above the stop plus 20%,
and the harness row is then recorded on the committed file.

### 2.2 Item 7: the product room

Prompt 48's standard upwind rows on 80x30 and 160x60, through prompt 48's runner (prompt 47's
`control` arm: the committed solver on `configs/clean_room_default.yaml` regridded, with prompt
46's product settings), each with its own settings, under rule version 5. Reported: the stop and
the outer count against prompt 48's (1,046 and 2,003), the scale at the stop against the inlet
scale those rows ran under, and the largest difference of the stopped face velocities from prompt
48's stopped faces over the velocity scale (the largest prescribed boundary velocity). 320x120
waits for step 8.

### 2.3 Item 3: unchanged by construction

VAL-001, its stretched twin and VAL-002 (80x80) through the committed solver on main (a worktree
at e281629) and on commit A: the stop, the outer count and the face hash, equal or not
(`check50.py`, prompt 48's measure). VAL-016 through its tests at commit A.

## 3. Predictions

### 3.1 Orchestrator (2026-10-11; committed before item 6's runs)

- (a) Under the file's `turbulence.tol` (1e-10), 216x72 still does not stop: the rate fit goes
  infinite once the nu_t step floors at the k and eps solves' tolerance, whatever the scale.
- (b) At 1e-13 it stops, between 11,200 and 12,500 outer iterations. Prompt 49's record reading
  put the stop at 11,206 under nu plus the room's largest nu_t; the median is about 2.4 times
  smaller, which delays the stop by roughly ln(2.4) / -ln(rho_nu), a few hundred iterations at
  rho_nu near 0.998.
- (c) The stopped run's line values differ from prompt 49's record by under 1e-4 u0, and both
  scored lines pass the allowance (RMS 0.059 against 0.070 at x = H, 0.080 against 0.136 on the
  jet line).
- (d) The product rows still stop. Their inlet eddy viscosity (5%, 0.1 m) is of the order of the
  room's median, so (e) changes little: the outer count moves by under 20%, and the stopped fields
  differ by under 1e-3 of the velocity scale.

What each outcome means:

- (b) holds: validation cases with weak inlet turbulence carry `turbulence.tol` 1e-13, and the
  product keeps its own until step 8 measures it.
- (b) fails, no stop at either tolerance: (e)'s rate fit itself is the problem, not its scale, and
  Alex decides between a raw-step clause for (e) and dropping (e) for these cases.
- (d) fails: the product's stop moves enough to matter, and step 8 sets its tolerance from a
  measurement before VAL-018.

### 3.2 Builder (2026-10-11, before item 6's runs; scored apart)

Read from prompt 49's and prompt 48's records (their per-iteration velocity and nu_t steps) by
replaying the rule's two rate fits under a scale held at nu plus the stopped field's FLUID-cell
median; the replay reproduces the stops the records hold under the scales they ran (1,046 and
2,003 under the inlet scale, 11,206 and 11,160 under the room's largest nu_t). Two things the
replay cannot carry: the median moves during the run (it is taken at the end), and prompt 49's
1e-13 rows ran the k and eps solves with `max_iter` 2,000, whose k solve exceeded 500 sweeps in
372 outer iterations, all between 2,848 and 3,226; the file's 500 truncates those.

- (a) Holds. The nu_t step at the file's tolerance floors at 1.9e-10 m^2/s, and (e) under the
  median scale needs the step below about 1.2e-11 while the steps still contract at 0.998; the
  floor is sixteen times higher.
- (b) Holds, at about 11,440 (the replay gives 11,441 under upwind; umist 11,439), within 150 of
  it if the truncated k solves around outer 3,000 leave the late iteration as it was. The FLUID
  median is 380 nu at the stop, not the 367 of report 49's table, which took the median over
  every non-SOLID cell.
- (c) Holds on both clauses: the stop falls 560 iterations before prompt 49's 12,000 cap with
  (a)'s estimate already below 1e-6 of u0 there, so the lines differ from that record by under
  1e-6 u0; the RMS values are prompt 49's to three figures.
- (d) Holds. The product's FLUID median is 28 nu on 80x30 and 26 nu on 160x60 against the inlet
  scale's 17 nu, so (e) holds sooner: the replay gives 1,021 and 1,954, about 2.4% fewer, and the
  fields differ by the iteration error left at the earlier stop, under 1e-6 of the velocity
  scale.

## 4. Stop conditions (from the prompt)

- A laminar hash moves.
- Item 6 does not stop at either tolerance within 20,000 iterations: finish items 6 and 7,
  report, stop.
- A positivity error, or a capped pressure correction.
