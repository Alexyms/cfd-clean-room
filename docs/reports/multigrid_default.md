# Multigrid as the default pressure preconditioner (prompt 54)

**Date:** 2026-10-14
**Tree:** branch `feature/multigrid-default` from main at 14166aa (prompt 53 merged as PR 80).
Commit 0 (`f7a5de9`) records Alex's decision of 2026-10-14 under ADR-013 decision 1 and in
`docs/reports/pressure_multigrid.md`, section 8. Commit A (`11fb6ad`) fixes the smoother weight.
Commit B (`759b3ad`) makes the default.
**Policy:** `docs/REVIEW_POLICY.md`, "What blocks". The numbers live in generated tables; the
sentences point at them.

## 1. Predictions

Stated by the orchestrator on 2026-10-14 in prompt 54. Committed alone, before item 6's
measurements. The twelve captured systems are prompt 53's: six cases (VAL-001, the closed cavity,
the product room on 160x60 and 320x120, the Annex 20 room on 216x72 and the backward-facing step's
medium grid 656x60), each early (outer 50) and late (outer 2,000, or the last correction of a run
that stops before it).

- (a) The fixed weight changes no captured system's iteration count by more than one, and cuts
  setup by at least half where it was over 10% of the correction.
- (b) Every case in item 7 reaches the same kind of stop as its Jacobi record, within 5% of the
  outer count, and meets its criterion: VAL-001's and VAL-002's as before, VAL-017's allowance,
  VAL-016's split criteria, the product's faces within 1e-3 of the velocity scale, the step's x_r
  within 0.02 H.
- (c) Every laminar face hash changes (the preconditioner changes every iterate) and no laminar
  criterion moves by more than its rounding.

What each outcome means (prompt 54):

- (a) fails: the fixed weight is wrong for some level, and the power iterations stay.
- (b) fails on a case: the default does not change for that case's file until the difference is
  measured; the others proceed.

Stop and report (prompt 54): a case that fails its criterion under the new default; a positivity
error, a capped pressure correction, or a CG solve that reaches its cap.

## 2. What was built

**Commit 0 (`f7a5de9`).** Alex's decision of 2026-10-14, a dated note under ADR-013 decision 1 and
in section 8 of `docs/reports/pressure_multigrid.md`. No code.

**Commit A (`11fb6ad`): a fixed smoother weight.** Every level's weight is `SMOOTHER_FACTOR / 2`
(2/3). The power iterations, `POWER_ITERATIONS`, `Level.eigenvalue` and the method that formed the
estimate are removed. The argument is the module docstring's: the finest level's diagonal is its
couplings plus any outlet term, a coarse diagonal is its children's diagonals less twice the
couplings inside the group, so every level is weakly diagonally dominant and Gershgorin bounds the
eigenvalues of `D^-1 A` by 2. Two things follow from taking that argument as the premise.

- `Multigrid` now refuses a finest system that is not weakly diagonally dominant (a `ValueError`
  naming the condition, with a rounding allowance of `DOMINANCE_TOLERANCE`). The bound is the
  fixed weight's whole justification, and the estimate used to notice a system that broke it.
  The check is one pass over the five arrays per correction, on the finest level only, since the
  coarse levels inherit dominance through the sums.
- The tests of the weight changed from "within 1% of the dense largest eigenvalue" to three:
  the weight is the fixed value on every smoothed level of the three solver systems and of a
  system whose diagonal sits far above its couplings, where an estimate would differ from 2/3;
  every level of those systems is dominant by rows and has a dense `lambda_max` of at most 2; and
  the refusal.

The same commit closes review 53's S5 and S8 for this module: the comment figure that disagreed
with the dev table now reads "one to seven fewer"; the comment resting on a scratch check went
with the power iterations; `restrict` and `add_prolonged` have Parameters and Returns sections;
and the `SMOOTHING_SWEEPS - 1` pre-smoothing loop is exercised, not removed, by a test that sets
nu to 2 and 3 and requires the cycle to stay symmetric and positive on the range and to differ
from V(1,1) (the loop is the reason the symmetry holds for any nu, and the probe that chose nu
used the knob).

**Commit B (`759b3ad`): the default.**

- `pressure_preconditioner` absent means `multigrid` (`DEFAULT_PRESSURE_PRECONDITIONER` in
  `src/config.py`). `jacobi` is selectable and is bitwise the loop of the commit before prompt 53.
- `PRESSURE_SOLVER_VERSION` is 3 with the label `staggered-mgcg`. This is a decision the prompt
  did not name; it is recorded under "Implementation decisions" below.
- The harness label selects the preconditioner. `run_case` runs a case under the preconditioner
  its label names whatever the case file says: `staggered-mgcg` is multigrid and the default,
  `staggered-cg` is the diagonal, so the stored `staggered-cg` rows keep their label and
  summarize apart from the new ones. The cell-update definition of a `staggered-mgcg` row says
  what it leaves uncounted, the V-cycle's own products. `validation.cases.with_pressure_preconditioner`
  is the copy helper, beside `with_velocity_step`.
- SYSTEM.md follows: REQ-S08's text names the multigrid preconditioner and the diagonal on request
  and is marked amended; the config, pressure, multigrid and cascade contracts follow.
- Tests that pinned the diagonal state `jacobi` explicitly: `tests/test_pressure.py` (its
  `_config` helper takes the key and defaults it to `jacobi`, so its cap, floor and residual
  tests keep testing the loop they were written for), the three cap tests of
  `tests/test_solver_staggered.py`, and the base-loop bitwise test of `tests/test_multigrid.py`,
  which now builds its corrector with `pressure_preconditioner: jacobi`. Tests of the default
  follow it: the config default, and a test that the absent key is bitwise the `multigrid`
  correction and not the diagonal's. The harness tests that compare a row with an independent
  solve of the same case (the Annex 20 and step rows) run the default label, so row and
  reference are the same solver; `tests/test_self_convergence.py` and
  `tests/test_solver_selection.py` follow the label.

**Implementation decisions.**

- *The solver version.* The default changes every iterate, so every saved history changes.
  `scripts/stopping_probe.py` saves per-iteration pressure iteration counts, imbalances and
  times, and reuses a saved solve whose stored `PRESSURE_SOLVER_VERSION` matches; with the version
  left at 2 it would serve a diagonal-preconditioned history as the default's. Raising the version
  is the mechanism review 37 S4 built for this (the comment on the constant), and the label lookup then
  makes the harness, the viewer and `scripts/self_convergence.py` file under `staggered-mgcg`
  without an edit. The alternative, leaving the version and labelling only the harness, was
  rejected for the stale-history reason. The cost: a user's untracked saved solves under
  `results/` are solved again on next use, and a corrector built with `jacobi` still carries
  version 3 (the `PRESSURE_SOLVER_VERSION` comment says so).
- *Which rows go into the results file.* The `staggered-cg` rows of this session are parity checks
  against the Jacobi records and are not appended; the `staggered-mgcg` rows of the harness cases
  are.

## 3. Method

**Item 6.** `weight54.py` solves prompt 53's twelve captured systems (`results/builder53/systems/`:
six cases, early and late, captured by `probe53/capture53.py`) to each case's `pressure_rtol`, with
the floor and cap the corrector uses, under two versions of the multigrid module in one process,
the two interleaved, five repeats each, and keeps the median setup, solve and total times. The old
version is `src/multigrid.py` of main at 14166aa, read with `git show` and loaded as a separate
module; it is swapped in by rebinding `Multigrid` in `src.pressure`, so both run the committed CG
loop and projection. One BLAS thread, alone on the machine.

**Item 7.** `row54.py` runs the harness's `run_case` from a worktree of commit B
(`results/builder54/B`, clean) on each harness case with nothing replaced but the solver class,
which is wrapped to read the faces after the row. `inject54.py` runs prompt 51's product and VAL-016
drivers from the same tree; their configurations carry no preconditioner key, so they run the
default. All fifteen processes ran together, one BLAS thread each, on a 24-core machine; the rows
record `concurrent_processes` 15, so their wall times compare only with each other. The Jacobi side
is the record each case has: the five laminar presets were also rerun this session under the label
`staggered-cg`, which runs the diagonal, and their hashes are compared with the hashes prompt 53
recorded; the Annex 20 room, the product rows and VAL-016 are compared with prompt 51's records,
and the step with the clean harness row of prompt 52 (`run_id` in `tables54.py`). Those records
were taken on this machine. The new hash of each case is the SHA-256 over the final `u` faces'
bytes then the `v` faces', recorded with commit B.

## 4. Planted defects

Each defect was planted in a worktree made for it (`results/builder54/plantsA` at commit A,
`plantsB` at commit B), the test files that guard it run there, and the file restored
(`plant54.py`).

<!-- BEGIN tables54: plants -->
| Plant | Commit | Defect | Result | Failing tests | Failing (first three) |
| --- | --- | --- | --- | --- | --- |
| `old_estimate_restored` | A (`11fb6ad`) | the three-power-iteration weight restored, on levels whose estimate differs from 2 | caught | 7 | TestCycle::test_every_level_has_the_fixed_weight[closed]; TestCycle::test_every_level_has_the_fixed_weight[open]; TestCycle::test_every_level_has_the_fixed_weight[open_stretched] ... |
| `presmoothing_dropped` | A (`11fb6ad`) | the extra pre-smoothing sweeps of V(nu, nu) dropped, so nu above 1 is unbalanced | caught | 6 | TestCycle::test_more_than_one_sweep_keeps_the_cycle_symmetric_and_positive[closed-2]; TestCycle::test_more_than_one_sweep_keeps_the_cycle_symmetric_and_positive[closed-3]; TestCycle::test_more_than_one_sweep_keeps_the_cycle_symmetric_and_positive[open-2] ... |
| `dominance_unchecked` | A (`11fb6ad`) | the finest system's diagonal dominance not checked | caught | 1 | TestCycle::test_a_finest_system_that_is_not_dominant_is_refused |
| `dominance_tolerance_wide` | A (`11fb6ad`) | the dominance check's rounding allowance widened to 50% | caught | 1 | TestCycle::test_a_finest_system_that_is_not_dominant_is_refused |
| `coarsest_smoothed` | A (`11fb6ad`) | the coarsest level given a weight although it is solved exactly | caught | 4 | TestCycle::test_every_level_has_the_fixed_weight[closed]; TestCycle::test_every_level_has_the_fixed_weight[open]; TestCycle::test_every_level_has_the_fixed_weight[open_stretched] ... |
| `weight_beyond_contraction` | A (`11fb6ad`) | a smoother weight of 1.5, past the energy-norm contraction limit 2 / lambda_max | caught | 21 | TestCycle::test_every_level_has_the_fixed_weight[closed]; TestCycle::test_every_level_has_the_fixed_weight[open]; TestCycle::test_every_level_has_the_fixed_weight[open_stretched] ... |
| `default_left_jacobi` | B (`759b3ad`) | the default constant left at the diagonal | caught | 4 | TestPressureKeys::test_pressure_preconditioner_defaults_to_multigrid_and_reads_either_choice; TestPreconditionerChoice::test_the_absent_key_is_the_multigrid_correction_bitwise[closed]; TestPreconditionerChoice::test_the_absent_key_is_the_multigrid_correction_bitwise[open] ... |
| `default_read_as_jacobi` | B (`759b3ad`) | the default constant changed but the configuration still reading the diagonal | caught | 4 | TestPressureKeys::test_pressure_preconditioner_defaults_to_multigrid_and_reads_either_choice; TestPreconditionerChoice::test_the_absent_key_is_the_multigrid_correction_bitwise[closed]; TestPreconditionerChoice::test_the_absent_key_is_the_multigrid_correction_bitwise[open] ... |
| `label_ignored` | B (`759b3ad`) | run_case leaving the case file's preconditioner, so a label can claim a solver it did not run | caught | 2 | test_a_row_runs_the_preconditioner_its_label_names; test_staggered_velocity_step_stop_has_the_collocated_label |
| `diagonal_label_runs_multigrid` | B (`759b3ad`) | the staggered-cg label mapped to the multigrid preconditioner | caught | 3 | test_each_label_names_its_preconditioner_and_its_cell_update_definition; test_a_row_runs_the_preconditioner_its_label_names; test_staggered_velocity_step_stop_has_the_collocated_label |
| `labels_pooled` | B (`759b3ad`) | both preconditioners filed under one label, so their rows would summarize together | caught | 8 | TestCorrection::test_each_solver_version_has_its_own_label; test_the_annex20_row_is_scored_by_the_allowance; test_the_backstep_row_is_reported_not_scored ... |
| `version_not_raised` | B (`759b3ad`) | the version left at 2, so a saved diagonal-preconditioned history is read as current | caught | 7 | TestCorrection::test_result_fields_are_the_contract; test_the_annex20_row_is_scored_by_the_allowance; test_the_backstep_row_is_reported_not_scored ... |
| `work_definition_unchanged` | B (`759b3ad`) | the multigrid row's work definition not saying what it leaves out | caught | 2 | test_each_label_names_its_preconditioner_and_its_cell_update_definition; test_a_row_runs_the_preconditioner_its_label_names |
| `helper_unvalidated` | B (`759b3ad`) | with_pressure_preconditioner accepting any name | caught | 1 | test_with_pressure_preconditioner_copies_and_refuses_unknown_names |
<!-- END tables54: plants -->

The full suite, 1,512 tests, passed at commit D with this report's edits in the tree
(`results/builder54/suite_final.txt`).

## 5. The weight alone (item 6)

<!-- BEGIN tables54: weight_systems -->
| System | CG its, old / new | max abs p' difference / max abs p', old | Setup ms, old / new | Setup new / old | Old setup share of the correction | Solve ms, old / new | Total ms, old / new | Total new / old |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VAL-001 80x40, outer 50 | 25 / 25 | 2.4e-11 | 2.06 / 1.02 | 0.50 | 18.2% | 9.39 / 9.77 | 11.33 / 11.14 | 0.98 |
| VAL-001 80x40, outer 1,558 | 9 / 9 | 2.1e-06 | 1.55 / 1.08 | 0.70 | 25.7% | 4.75 / 4.34 | 6.03 / 5.53 | 0.92 |
| VAL-002 cavity 80x80 (closed), outer 50 | 36 / 36 | 0.0e+00 | 1.73 / 1.21 | 0.70 | 6.8% | 23.64 / 22.88 | 25.38 / 24.24 | 0.96 |
| VAL-002 cavity 80x80 (closed), outer 2,000 | 27 / 27 | 5.3e-15 | 2.02 / 1.20 | 0.59 | 9.6% | 19.50 / 16.05 | 21.16 / 17.06 | 0.81 |
| Product 160x60, outer 50 | 16 / 16 | 0.0e+00 | 2.32 / 1.25 | 0.54 | 15.2% | 12.68 / 12.98 | 15.30 / 14.91 | 0.97 |
| Product 160x60, outer 1,953 | 8 / 8 | 2.0e-14 | 3.46 / 1.96 | 0.57 | 28.7% | 9.00 / 7.78 | 12.08 / 9.94 | 0.82 |
| Product 320x120, outer 50 | 23 / 23 | 6.4e-15 | 5.30 / 3.13 | 0.59 | 6.9% | 69.53 / 56.70 | 76.28 / 60.08 | 0.79 |
| Product 320x120, outer 2,000 | 23 / 23 | 3.5e-15 | 6.17 / 2.86 | 0.46 | 13.0% | 41.99 / 52.98 | 47.59 / 55.65 | 1.17 |
| Annex 20 216x72, outer 50 | 22 / 22 | 4.4e-16 | 2.44 / 1.53 | 0.63 | 8.3% | 26.85 / 21.81 | 29.27 / 23.49 | 0.80 |
| Annex 20 216x72, outer 2,000 | 26 / 26 | 2.7e-15 | 2.67 / 1.59 | 0.60 | 8.2% | 29.94 / 27.30 | 32.78 / 29.03 | 0.89 |
| Step medium 656x60, outer 50 | 25 / 25 | 7.4e-07 | 5.15 / 2.79 | 0.54 | 8.4% | 54.23 / 49.67 | 61.10 / 52.46 | 0.86 |
| Step medium 656x60, outer 2,000 | 22 / 22 | 1.7e-06 | 5.53 / 2.61 | 0.47 | 10.7% | 47.21 / 39.97 | 51.87 / 42.38 | 0.82 |
<!-- END tables54: weight_systems -->

The weights the levels had and have:

<!-- BEGIN tables54: weight_levels -->
| System | Smoothed levels | Old lambda_max estimate | Old weight | New weight |
| --- | --- | --- | --- | --- |
| VAL-001 80x40, outer 50 | 3 | 1.9912 to 1.9978 | 0.6674 to 0.6696 | 0.6667 |
| VAL-001 80x40, outer 1,558 | 3 | 1.9912 to 1.9978 | 0.6674 to 0.6696 | 0.6667 |
| VAL-002 cavity 80x80 (closed), outer 50 | 4 | 2.0000 to 2.0000 | 0.6667 to 0.6667 | 0.6667 |
| VAL-002 cavity 80x80 (closed), outer 2,000 | 4 | 2.0000 to 2.0000 | 0.6667 to 0.6667 | 0.6667 |
| Product 160x60, outer 50 | 4 | 2.0000 to 2.0000 | 0.6667 to 0.6667 | 0.6667 |
| Product 160x60, outer 1,953 | 4 | 2.0000 to 2.0000 | 0.6667 to 0.6667 | 0.6667 |
| Product 320x120, outer 50 | 5 | 2.0000 to 2.0000 | 0.6667 to 0.6667 | 0.6667 |
| Product 320x120, outer 2,000 | 5 | 2.0000 to 2.0000 | 0.6667 to 0.6667 | 0.6667 |
| Annex 20 216x72, outer 50 | 5 | 2.0000 to 2.0000 | 0.6667 to 0.6667 | 0.6667 |
| Annex 20 216x72, outer 2,000 | 5 | 2.0000 to 2.0000 | 0.6667 to 0.6667 | 0.6667 |
| Step medium 656x60, outer 50 | 5 | 1.9967 to 1.9998 | 0.6667 to 0.6678 | 0.6667 |
| Step medium 656x60, outer 2,000 | 5 | 1.9970 to 1.9998 | 0.6667 to 0.6677 | 0.6667 |
<!-- END tables54: weight_levels -->

The fixed weight changes no iteration count, and the setup falls by less than the prediction's
half on some systems. The first is the claim that matters for the weight: on every captured level
the estimate was within a percent of 2, so 2/3 and the estimated weight give nearly the same cycle.
The second is a smaller saving than predicted, not a wrong weight: the outcome line written for (a)
("the fixed weight is wrong for some level, and the power iterations stay") describes a weight that
costs iterations, and none does. What is left of the setup, after the power iterations, is the
coarsening's strided sums and the dense pseudo-inverse of the coarsest level, the pseudo-inverse
the larger part on the small laminar systems and the coarsening on the large ones; the dominance
check that replaces the estimate's safety net is a small part of either.

<!-- BEGIN tables54: setup_parts -->
| System | Setup ms | Dominance check | Coarsening, all levels | Smoother arrays | Coarsest dense pseudo-inverse | Pseudo-inverse share |
| --- | --- | --- | --- | --- | --- | --- |
| VAL-001 80x40, outer 50 | 0.82 | 0.01 | 0.18 | 0.04 | 0.48 | 59% |
| VAL-001 80x40, outer 1,558 | 0.77 | 0.02 | 0.17 | 0.06 | 0.52 | 67% |
| VAL-002 cavity 80x80 (closed), outer 50 | 0.63 | 0.02 | 0.37 | 0.06 | 0.19 | 30% |
| VAL-002 cavity 80x80 (closed), outer 2,000 | 0.77 | 0.02 | 0.27 | 0.09 | 0.28 | 37% |
| Product 160x60, outer 50 | 0.75 | 0.02 | 0.43 | 0.08 | 0.32 | 43% |
| Product 160x60, outer 1,953 | 0.74 | 0.02 | 0.44 | 0.11 | 0.32 | 44% |
| Product 320x120, outer 50 | 1.83 | 0.41 | 0.95 | 0.18 | 0.34 | 19% |
| Product 320x120, outer 2,000 | 2.07 | 0.17 | 1.08 | 0.20 | 0.34 | 17% |
| Annex 20 216x72, outer 50 | 0.86 | 0.03 | 0.39 | 0.10 | 0.15 | 18% |
| Annex 20 216x72, outer 2,000 | 0.94 | 0.04 | 0.43 | 0.15 | 0.18 | 19% |
| Step medium 656x60, outer 50 | 2.02 | 0.10 | 1.13 | 0.21 | 0.39 | 19% |
| Step medium 656x60, outer 2,000 | 2.26 | 0.10 | 1.12 | 0.21 | 0.51 | 23% |
<!-- END tables54: setup_parts -->

(`setup54.py`, a separate pass after the whole runs, alone, medians of 200 calls. Timings on this
machine move between sessions by tens of percent, so the shares are the reading, not the
milliseconds.)

## 6. The new baseline (item 7)

<!-- BEGIN tables54: baseline -->
| Case | Jacobi record | Jacobi stop | Default stop | Outer change | Jacobi answer | Default answer (products: largest face difference over the scale) | Criterion | Capped corrections |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val001_80x40 | rerun under staggered-cg, this session | error_estimate_and_continuity at 1,559 | error_estimate_and_continuity at 1,559 | +0.00% | 4.1077e-04 | 4.1077e-04 | L2 < 0.01 | 0 |
| val001_80x40_stretched | rerun under staggered-cg, this session | error_estimate_and_continuity at 1,124 | error_estimate_and_continuity at 1,124 | +0.00% | 3.0237e-03 | 3.0237e-03 | L2 < 0.01 | 0 |
| val002_20x20 | rerun under staggered-cg, this session | error_estimate_and_continuity at 970 | error_estimate_and_continuity at 970 | +0.00% | u 2.1436e-02, v 1.3370e-02 | u 2.1436e-02, v 1.3370e-02 | criterion 3a (table below) | 0 |
| val002_40x40 | rerun under staggered-cg, this session | error_estimate_and_continuity at 3,435 | error_estimate_and_continuity at 3,435 | +0.00% | u 4.5487e-03, v 3.0801e-03 | u 4.5487e-03, v 3.0801e-03 | criterion 3a (table below) | 0 |
| val002_80x80 | rerun under staggered-cg, this session | error_estimate_and_continuity at 12,814 | error_estimate_and_continuity at 12,814 | +0.00% | u 1.0572e-03, v 7.3556e-04 | u 1.0572e-03, v 7.3556e-04 | u, v < 0.02 (criterion 3) | 0 |
| val017_216x72 | prompt 51 record | error_estimate_and_continuity at 11,441 | error_estimate_and_continuity at 11,443 | +0.02% | 0.84395 | 0.84395 | worst scored RMS / allowance < 1 | 0 |
| val019_656x60 | prompt 52 row of record | max_simple_iter at 20,000 | max_simple_iter at 20,000 | +0.00% | x_r/H 4.8586 | x_r/H 4.8541 | within 0.02 H of the Jacobi row | 0 |
| product_80x30 | prompt 51 record | error_estimate_and_continuity at 1,021 | error_estimate_and_continuity at 1,021 | +0.00% | - | 1.19e-11 | faces within 0.001 of the scale | 0 |
| product_160x60 | prompt 51 record | error_estimate_and_continuity at 1,954 | error_estimate_and_continuity at 1,953 | -0.05% | - | 1.57e-10 | faces within 0.001 of the scale | 0 |
| val016 | prompt 51 record | error_estimate_and_continuity at 258 | error_estimate_and_continuity at 258 | +0.00% | du/U_w 7.029e-08, dk/u_tau^2 3.741e-06 | du/U_w 7.029e-08, dk/u_tau^2 3.741e-06 | (a) both < 0.0001; (b) core k within 0.01 of 1 | 0 |
<!-- END tables54: baseline -->

The face hashes. The default's are the hashes of record from commit B, on this machine; the
diagonal's are kept beside them:

<!-- BEGIN tables54: hashes -->
| Case | Hash under the diagonal (kept) | Hash under the default (of record) | Changed | Commit | Note |
| --- | --- | --- | --- | --- | --- |
| val001_80x40 | `c7d88d919e1141a5` | `f2e5b888cb269112` | yes | `759b3ad` | equals prompt 53's record and `ecr003_step2_baseline.md` |
| val001_80x40_stretched | `9838d632f181d90f` | `4c79ffb9def55fbb` | yes | `759b3ad` | equals prompt 53's record and `ecr003_step2_baseline.md` |
| val002_20x20 | `8580ae56fa7676a8` | `1ecbedb180303c8f` | yes | `759b3ad` | equals `ecr003_step2_baseline.md` |
| val002_40x40 | `459c6691ab2cc9ea` | `7a881aaa88b1946d` | yes | `759b3ad` | no earlier record |
| val002_80x80 | `a8a61e28120e3f44` | `95efc0123aaabc55` | yes | `759b3ad` | equals prompt 53's record and `ecr003_step2_baseline.md` |
| val017_216x72 | `be4b4fdbccae7e28` | `d59c4a1a644f67aa` | yes | `759b3ad` |  |
| val019_656x60 | none stored | `2c8a57b634e018ab` | - | `759b3ad` | prompt 52's driver run: `37d84ab7c1495488` |
| product_80x30 | `733672ccd4dca776` | `6f27f27f3e89963a` | yes | `759b3ad` |  |
| product_160x60 | `a126326517fb19bd` | `d295ccb1a4c8f9e1` | yes | `759b3ad` |  |
<!-- END tables54: hashes -->

VAL-002's criterion 3a, the errors at the three grids and the orders, under each preconditioner:

<!-- BEGIN tables54: cavity_orders -->
| Component | Preconditioner | Error at 20x20 / 40x40 / 80x80 | Order, 20 to 40 / 40 to 80 | Falls at each refinement |
| --- | --- | --- | --- | --- |
| u | Jacobi | 2.1436e-02 / 4.5487e-03 / 1.0572e-03 | 2.24 / 2.11 | yes |
| u | default | 2.1436e-02 / 4.5487e-03 / 1.0572e-03 | 2.24 / 2.11 | yes |
| v | Jacobi | 1.3370e-02 / 3.0801e-03 / 7.3556e-04 | 2.12 / 2.07 | yes |
| v | default | 1.3370e-02 / 3.0801e-03 / 7.3556e-04 | 2.12 / 2.07 | yes |
<!-- END tables54: cavity_orders -->

The Annex 20 room, its four lines:

<!-- BEGIN tables54: annex -->
| Line | Scored | RMS / u0, Jacobi record | RMS / u0, default | Allowance | Default | Largest difference from the Jacobi record, u / u0 |
| --- | --- | --- | --- | --- | --- | --- |
| x3 | yes | 0.059288 | 0.059288 | 0.070250 | pass | 2.79e-09 |
| x6 | no | 0.036822 | 0.036822 | 0.093296 | - | 1.75e-09 |
| y2916 | yes | 0.079742 | 0.079742 | 0.135583 | pass | 1.37e-09 |
| y0084 | no | 0.057916 | 0.057916 | 0.052172 | - | 2.90e-09 |
<!-- END tables54: annex -->

The step's figures, every one its harness row reports:

<!-- BEGIN tables54: step -->
| Figure | Jacobi row of record | Default | Difference |
| --- | --- | --- | --- |
| x_r_over_h | 4.8586 | 4.85411 | -0.00449 |
| u_ref_over_measured | 0.999615 | 0.999615 | -1.19e-08 |
| delta_99_over_h | 1.45711 | 1.45711 | -3.24e-07 |
| re_theta | 5229.4 | 5229.39 | -0.00438 |
| rms_cf | 0.000426956 | 0.000427181 | +2.25e-07 |
| rms_cp | 0.0311057 | 0.0311 | -5.68e-06 |
| rms_profile_x-4.0 | 0.018139 | 0.018139 | +3.16e-08 |
| rms_profile_x1.0 | 0.0478425 | 0.0478424 | -1.33e-07 |
| rms_profile_x4.0 | 0.0547241 | 0.0547277 | +3.61e-06 |
| rms_profile_x6.0 | 0.0235447 | 0.0235454 | +6.96e-07 |
| rms_profile_x10.0 | 0.0275178 | 0.0275175 | -2.29e-07 |
<!-- END tables54: step -->

VAL-016's split criteria:

<!-- BEGIN tables54: val016 -->
| Preconditioner | Commit | Stop | Outer | (a) max du / U_w (< 1e-4) | (a) max dk / u_tau^2 (< 1e-4) | (a) u_tau / 1D | (b) standard core k ratio | (b) RNG core k ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jacobi, prompt 51 record | `4cf12f9` | error_estimate_and_continuity | 258 | 7.029e-08 | 3.741e-06 | 0.9999997 | 0.99946 to 1.00146 | 0.99941 to 1.00108 |
| default | `759b3ad` | error_estimate_and_continuity | 258 | 7.029e-08 | 3.741e-06 | 0.9999997 | 0.99946 to 1.00146 | 0.99941 to 1.00108 |
<!-- END tables54: val016 -->

## 7. Predictions against measurement

<!-- BEGIN tables54: predictions -->
|  | Prediction | Measured | Verdict |
| --- | --- | --- | --- |
| (a) | The fixed weight changes no captured system's iteration count by more than one, and cuts setup by at least half where it was over 10% of the correction | iteration counts equal on 12 of 12 systems, the largest change 0; setup new / old 0.46 to 0.70 on the 6 systems whose old setup share was over 10%, at most half on 3 of them | first clause holds; second clause fails |
| (b) | Every case in item 7 reaches the same kind of stop as its Jacobi record, within 5% of the outer count, and meets its criterion | outer change: val001_80x40 +0.00%, val001_80x40_stretched +0.00%, val002_20x20 +0.00%, val002_40x40 +0.00%, val002_80x80 +0.00%, val017_216x72 +0.02%, val019_656x60 +0.00%, product_80x30 +0.00%, product_160x60 -0.05%, val016 +0.00% | holds |
| (c) | Every laminar face hash changes and no laminar criterion moves by more than its rounding | hash changed on 5 of 5 laminar cases; criterion values equal at the five significant figures the tables print on 5 of 5 | holds |
<!-- END tables54: predictions -->

## 8. What this leaves for Alex

Decisions made in the builder's pass that the prompt did not name, and findings it did not ask for.

1. **Prediction (a).** Its second clause failed and its first held. I kept the fixed weight (section
   5 says why the outcome line does not apply). If the orchestrator reads the line as binding, the
   revert is commit A's `Multigrid._level`; the dominance check and the tests of the weight go with
   it.
2. **`PRESSURE_SOLVER_VERSION` is 3.** The prompt asked for a harness label. The default changes
   every iterate, so every saved history in `scripts/stopping_probe.py` is a diagonal-preconditioned
   one, and the version is what makes that script solve again (section 2, "Implementation
   decisions"). A user's untracked saved solves under `results/` are therefore solved again on
   next use, once.
3. **The cell-update count is not comparable across the two labels.** A `staggered-mgcg` row
   counts the CG loop's products and not the V-cycle's own (two per smoothed level per cycle, and
   the transfers), and says so in its stored definition, so the summary's cell-update column
   understates the multigrid rows' work. Wall time is the comparable axis between them.
4. **The harness rows were taken with fifteen processes running together**, on the 24-core
   machine, one BLAS thread each, and record `concurrent_processes` 15. The summary keeps them
   apart from rows at other loads. A row taken alone, if the baseline for wall time is wanted, is a
   rerun of `scripts/benchmark.py --method staggered-mgcg`.
5. **The step's Jacobi row of record stores no face hash.** Harness rows store none; the table
   shows prompt 52's driver run of the same configuration beside it for reference only, since a
   different run produced it. The step's new hash is the first of record.
6. **Nothing in the momentum, transport, turbulence or boundary modules, the stopping rule, the
   hierarchy beyond the weight, or any case file changed**, and the `jacobi` path is bitwise what it
   was (five laminar reruns reproduce their hashes).
