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

## 5. Results (written after the runs)

The four runs of items 6 and 7 were launched together on 2026-10-11 at 16:05 (`run50.sh flows`,
one BLAS thread each, on commit 68ac2eb with the probe untracked); the product rows finished in
minutes, the Annex 20 rows by 16:57. The harness row of section 5.2 ran afterwards, alone, in a
worktree of the commit that set the case file's cap. Wall times beside other runs are not
measurements.

### 5.1 Item 6: the Annex 20 room under rule version 5

<!-- tables50 annex begin -->
| Run | k, eps tol | Cap | Stop | Outer | Rule version | Readings at stop (a), (b), (c), (d), (e) | Holds from (a), (b), (c), (d), (e) | (e)'s scale at stop / nu; FLUID nu_t / nu: median, largest | k, eps sweeps (mean); outer iterations with a capped k or eps solve | Pressure cap hits | Wall (s) | Face hash |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| file | 1e-10 | 20,000 | error_estimate_and_continuity | 11,441 | 5 | 4.3e-07, 9.5e-16, 7.4e-12, 3e-18, 1e-06 | 11,159, 2,549, 2,585, 1, 11,441 | 380; 379, 898 | 276.2, 184.4; 0 | 0 | 2773 | 719c37d8bdfe75cf |
| tol1e-13 | 1e-13 | 20,000 | error_estimate_and_continuity | 11,441 | 5 | 4.3e-07, 9.5e-16, 7.4e-12, 2e-18, 1e-06 | 11,159, 2,549, 2,584, 1, 11,441 | 380; 379, 898 | 389.9, 278.6; 374 | 0 | 3160 | 64540f2b54c0cad7 |
<!-- tables50 annex end -->

Both rows stop by the rule, at the same outer iteration, with the same readings to the figures
shown and condition (e) the last to hold (the holds-from column). The k and eps tolerance does
not decide the stop. At 1e-13 the k solve averages more sweeps and reaches its 500-sweep cap in
a few hundred outer iterations (the sweeps column), and the stop and the lines are the same:

<!-- tables50 lines begin -->
| Run | Line | RMS against the data, u / u0 (stations) | Allowance (published RMS) | Scored | Verdict | Largest difference from prompt 49's 1e-13 record (12,000 cap), u0 | From prompt 49's file row (10,000 cap), u0 |
|---|---|---|---|---|---|---|---|
| file | x = H (3.0 m) | 0.0593 (25) | 0.0703 | scored | pass | 1.2e-07 | 1.1e-06 |
| file | x = 2H (6.0 m) | 0.0368 (25) | 0.0933 | reported | - | 1.5e-07 | 2.6e-07 |
| file | y = 2.916 m, the ceiling jet | 0.0797 (28) | 0.1356 | scored | pass | 2.5e-07 | 8.8e-07 |
| file | y = 0.084 m, the floor | 0.0579 (28) | 0.0522 | reported | - | 1.5e-07 | 1.1e-06 |
| file | the row | value 0.8440 | 1 | - | pass | - | - |
| tol1e-13 | x = H (3.0 m) | 0.0593 (25) | 0.0703 | scored | pass | 8.2e-08 | 1.1e-06 |
| tol1e-13 | x = 2H (6.0 m) | 0.0368 (25) | 0.0933 | reported | - | 1.1e-07 | 2.4e-07 |
| tol1e-13 | y = 2.916 m, the ceiling jet | 0.0797 (28) | 0.1356 | scored | pass | 2.1e-07 | 9.3e-07 |
| tol1e-13 | y = 0.084 m, the floor | 0.0579 (28) | 0.0522 | reported | - | 1.1e-07 | 1.1e-06 |
| tol1e-13 | the row | value 0.8440 | 1 | - | pass | - | - |
<!-- tables50 lines end -->

Both scored lines pass, x = H with less margin than the jet line (the row's value is the larger
ratio of RMS to allowance). The floor line's RMS is above the published one, as report 49
found it; it is reported, not scored. The lines sit within a few tenths of a millionth of u0 of
prompt 49's 1e-13 record and within about a millionth of its file row at the 10,000 cap.

**Report 49's floor was the cap, not the tolerance.** The 1e-10 row is prompt 49's file row
continued: over the first 10,000 outer iterations the steps the rule received, velocity and
eddy viscosity, are bit for bit the ones prompt 49's recording rule kept
(`results/builder49/standard_216x72_upwind.json`, `readings`). Report 49 (section 5.2, the
condition (e) table) read the nu_t step at its cap as a floor at the k and eps solves'
tolerance, about 1.9e-10 m^2/s, and concluded that under the file's tolerance the rule could not
stop under any scale. The continued run shows a pause, not a floor: over the last hundred
iterations before 10,000 the step's per-iteration ratio is about 1.00002, and from there it
contracts again (about 0.9986 per iteration over 10,500 to 10,600, 0.9974 after 11,000) to about
1.7e-11 at the stop, below the value report 49 called the floor by a factor of ten. The rule's
rate fit was infinite across the pause, which is what report 49 saw at its cap. What stopped
prompt 49's rows was the 10,000 cap and (e)'s scale together, not the tolerance; the tighter
tolerance changed nothing that reached the stop.

The case file therefore keeps `turbulence.tol` 1e-10. The prompt foresaw changing the file only
if the tighter tolerance alone stopped; under the file's tolerance the row stops at 11,441, above
the file's 10,000 cap, so the cap went to 14,000 by the rule item 6 gives for it (the next round
thousand above the stop plus 20%), and nothing else in the file changed. That is a deviation from
the prompt's wording, made so the committed case stops by its rule; Alex may prefer another cap.

### 5.2 The harness row

`python scripts/benchmark.py --cases val017_216x72 --repeats 1` on the committed case file, in a
clean worktree of 7be518d, alone, appended to `benchmarks/results.jsonl`:

<!-- tables50 harness begin -->
| Commit | Dirty | Stop | Outer | Rule version | k, eps tol; max_simple_iter | RMS / allowance: x = H, jet | Value | Verdict | Pressure cap hits | Wall (s) | Processes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 7be518d | no | error_estimate_and_continuity | 11,441 | 5 | 1e-10, 14,000 | 0.0593 / 0.0703, 0.0797 / 0.1356 | 0.8440 | pass | 0 | 4934 | 1 |
<!-- tables50 harness end -->

The row stops where the probe rows did, with the same lines and verdicts: the cap of 14,000 is
not reached, so the committed case reproduces the measured stop. Its wall time is not a
measurement of the solver: the process ran alone but was given only part of a core for most of
its run (about 29 minutes of processor time in its first 73 minutes of wall time).

### 5.3 Item 7: the product room

<!-- tables50 product begin -->
| Grid | Stop: prompt 48 / now | Outer | Change | Rule version | (e)'s scale at stop / nu | Now, holds from (a), (b), (c), (d), (e) | Largest face difference / velocity scale | Pressure cap hits | Face hash |
|---|---|---|---|---|---|---|---|---|---|
| 80x30 | error_estimate_and_continuity / error_estimate_and_continuity | 1,046 / 1,021 | -2.4% | 4 / 5 | 17.4 / 28.3 | 788, 192, 181, 1, 1,021 | 6.1e-09 | 0 / 0 | 28f437c84e3d5057 / 7300477dfade4b89 |
| 160x60 | error_estimate_and_continuity / error_estimate_and_continuity | 2,003 / 1,954 | -2.4% | 4 / 5 | 17.4 / 26.2 | 1,432, 398, 374, 1, 1,954 | 7e-09 | 0 / 0 | 73f5eaded654a1a7 / a44f891cef2d5a29 |
<!-- tables50 product end -->

Both rows still stop, about 2.4% sooner, condition (e) still the last to hold: the room's
FLUID-cell median is above the inlet scale they ran under, so (e) holds a few dozen iterations
earlier. The stopped faces differ from prompt 48's by a few billionths of the velocity scale.
No pressure correction reached its cap and no run raised a positivity error.

### 5.4 Item 3 and the planted defects

<!-- tables50 laminar begin -->
| Case | Stop: main / commit A | Outer | Face hash | Equal |
|---|---|---|---|---|
| val001_80x40 | error_estimate_and_continuity / error_estimate_and_continuity | 1,559 / 1,559 | c7d88d919e1141a5 / c7d88d919e1141a5 | yes |
| val001_80x40_stretched | error_estimate_and_continuity / error_estimate_and_continuity | 1,124 / 1,124 | 9838d632f181d90f / 9838d632f181d90f | yes |
| val002_80x80 | error_estimate_and_continuity / error_estimate_and_continuity | 12,814 / 12,814 | a8a61e28120e3f44 / a8a61e28120e3f44 | yes |
<!-- tables50 laminar end -->

VAL-016 passes at commit A with the version it pins now 5 (`tests/test_turbulent_channel.py`,
five of five). The planted defects of commits A and C (`plant50.py`, each one line of a scratch
copy, the rule's, the coupled solve's, the Annex 20 and the harness tests run):

<!-- tables50 plants begin -->
| Plant | pytest exit | Caught | Failing tests |
|---|---|---|---|
| (none, the control) | 0 | - | - |
| the maximum in place of the median | 1 | yes | 3: test_condition_e_scales_by_the_median_of_the_iterate, test_one_cells_overshoot_leaves_the_stop_unchanged, test_the_scale_is_the_median_of_the_fluid_cells |
| the median over every cell | 1 | yes | 2: test_condition_e_scales_by_the_median_of_the_iterate, test_the_scale_is_the_median_of_the_fluid_cells |
| nu left out of the scale | 1 | yes | 2: test_condition_e_scales_by_the_median_of_the_iterate, test_the_scale_is_the_median_of_the_fluid_cells |
| the first iteration's median kept (solver) | 1 | yes | 1: test_condition_e_scales_by_the_median_of_the_iterate |
| the inlet scale restored (solver) | 1 | yes | 1: test_condition_e_scales_by_the_median_of_the_iterate |
| the first scale kept (rule) | 1 | yes | 1: test_each_step_is_estimated_over_the_scale_passed_with_it |
| the scale not applied (rule) | 1 | yes | 6: test_a_room_sized_scale_stops_where_one_165_times_smaller_does_not, test_condition_e_is_the_estimate_on_the_eddy_viscosity, test_each_step_is_estimated_over_the_scale_passed_with_it, test_one_cells_overshoot_leaves_the_stop_unchanged |
| version 4 with (e) | 1 | yes | 3: test_the_annex20_row_is_scored_by_the_allowance, test_the_rule_version_is_recorded, test_the_version_is_the_rules |
| the per-step scale unchecked | 1 | yes | 5: test_a_nu_scale_not_positive_and_finite_is_rejected |
| a missing scale accepted | 1 | yes | 2: test_a_laminar_rule_refuses_a_viscosity_step_or_scale, test_condition_e_needs_the_step_and_its_scale |
| a truthy flag accepted | 1 | yes | 3: test_viscosity_condition_must_be_a_bool |
| the comparison reversed | 1 | yes | 7: test_a_line_passes_at_its_allowance_and_fails_above_it, test_the_annex20_row_is_scored_by_the_allowance, test_the_annex20_value_and_pass_read_the_scored_lines_only, test_the_measurements_pass_with_no_error, test_the_rms_is_over_every_station_of_the_line, test_the_unscored_lines_never_fail |
| the RMS over the published line's stations only | 1 | yes | 1: test_the_rms_is_over_every_station_of_the_line |
| a non-finite value accepted | 1 | yes | 1: test_a_missing_short_or_non_finite_line_is_refused |
| x = 2H scored | 1 | yes | 3: test_the_allowance_is_the_published_rms_the_decision_cites, test_the_annex20_value_and_pass_read_the_scored_lines_only, test_the_unscored_lines_never_fail |
| the smaller ratio as the row's value | 1 | yes | 1: test_the_annex20_row_is_scored_by_the_allowance |
| every line in the row's pass | 1 | yes | 1: test_the_annex20_value_and_pass_read_the_scored_lines_only |
| the turbulence section left out of the params | 1 | yes | 1: test_the_annex20_row_is_scored_by_the_allowance |
<!-- tables50 plants end -->

One plant survived its first run, every line in the row's value and pass: the short harness run
cannot show it, its scored lines being its worst. A unit test with the sampler replaced was added
(`test_the_annex20_value_and_pass_read_the_scored_lines_only`) and the plant is caught.

## 6. Predictions against measurement

<!-- tables50 predictions begin -->
| Prediction | Measured | Holds |
|---|---|---|
| (a) at 1e-10, 216x72 does not stop | error_estimate_and_continuity at 11,441 | no |
| (b) at 1e-13 it stops between 11,200 and 12,500 | error_estimate_and_continuity at 11,441 | yes |
| (c) lines within 1e-4 u0 of prompt 49's record; both scored lines pass | largest line difference from the 1e-13 record 2.1e-07; x3 0.059 against 0.070, y2916 0.080 against 0.136 | yes |
| (d) product rows stop; outer count within 20%; fields within 1e-3 | 80x30: error_estimate_and_continuity, -2.4%, 6.1e-09; 160x60: error_estimate_and_continuity, -2.4%, 7e-09 | yes |
<!-- tables50 predictions end -->

Orchestrator's (section 3.1):

- (a) fails. The row stops under the file's tolerance, for the reason section 5.1 gives: the
  nu_t step did not floor at the k and eps tolerance; prompt 49's cap fell in a pause of its
  contraction.
- (b) holds: 11,441, inside 11,200 to 12,500.
- (c) holds: the lines are within 2.1e-7 u0 of prompt 49's 1e-13 record and both scored lines
  pass (the lines table).
- (d) holds: both product rows stop, 2.4% sooner, their faces within 7e-9 of the velocity scale.

The outcome written for (b) holding, "validation cases with weak inlet turbulence carry
`turbulence.tol` 1e-13", rested on (a): it assumed the file's tolerance could not stop. With (a)
failed, the tighter tolerance buys nothing on this case and costs k sweeps and capped k solves,
so this report does not apply it; VAL-019 and VAL-020 are the next cases to say whether any case
needs it (section 7).

Builder's (section 3.2), scored apart: (a) wrong, the same misreading of report 49's floor;
(b) right, the replay's 11,441 exactly, and the stop does not depend on the truncated k solves
of prompt 49's 1e-13 run; (c) right; (d) right, the replay's 1,021 and 1,954 exactly.

## 7. What this implies

Questions, not decisions.

1. **The Annex 20 cap.** `max_simple_iter` is 14,000 in the case file (section 5.1), set by the
   rule item 6 gave for a tolerance change. Keep it, or another value?
2. **The k and eps tolerance for weak-inlet cases.** On Annex 20 the file's 1e-10 stops where
   1e-13 does. Does VAL-019 keep its file's tolerance until a run of its own says otherwise?
3. **Report 49's section 5.2 and its condition (e) table** state a floor that section 5.1 shows
   was a pause. Report 49 is merged; does it take a dated note pointing here?
4. **`TurbulenceBoundary.largest_inlet_eddy_viscosity`** is no longer called by the solver; this
   pass could not change `src/turbulence.py`. Keep it for reports, or retire it in step 8's pull
   request?

## 8. Reproducing

From the repository root, `.venv` active, prompt 48's and 49's records under
`results/builder48/` and `results/builder49/` (untracked; their reports say how to remake them):

```
python docs/reports/probe50/plant50.py "$(pwd)/.venv/Scripts/python.exe"   # the planted defects
git worktree add --detach results/builder50/base_tree e281629
bash docs/reports/probe50/run50.sh laminar base results/builder50/base_tree
bash docs/reports/probe50/run50.sh laminar built                    # on commit A
python docs/reports/probe50/check50.py report --base base --built built
bash docs/reports/probe50/run50.sh flows                            # items 6 and 7, about an hour
bash docs/reports/probe50/run50.sh compare
python scripts/benchmark.py --cases val017_216x72 --repeats 1 --sample-every 100   # the harness row
python docs/reports/probe50/tables50.py --check                     # the report's tables against the records
```
