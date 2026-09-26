# VAL-002 revalidation on the staggered solver (ECR-001 step 8)

**Date:** 2026-09-25
**Context:** ECR-001 step 8, acceptance criteria 3 and 3a, scored against `marchi_2009_re100`
(the amendment of 2026-09-24), with the cavity case moved to the `error_estimate` rule.
**Instruments:** `scripts/benchmark.py` (three rows in `benchmarks/results.jsonl`),
`tests/test_lid_cavity.py`, and one-offs under `results/builder26/` (section 6).

**Answers.** Criterion 3 passes: on 80x80 the worst error against Marchi is 1.057e-3 of the lid
speed, 18.9 times inside 2%. Criterion 3a passes: u and v each fall at every refinement, at
observed orders 2.24 and 2.11 (u) and 2.12 and 2.07 (v). Every solve stopped by
`error_estimate_and_continuity`, none at its cap. Against Ghia on the same fields neither
component falls monotonically.

## 1. Criteria 3 and 3a

Rows at commit f3561ba, clean tree, one process, `error_estimate` at 1e-6 and 1e-10, cap 20000.
Metric `max_normalized_centerline_error_cubic`, reference `marchi_2009_re100`.

| MEASURED | Outer | Row s | u | v | Row run_id |
|---|---|---|---|---|---|
| 20x20 | 1370 | 12.9 | 2.144e-2 | 1.337e-2 | 5129231b99824b52ac294f55b3154a58 |
| 40x40 | 3849 | 120.2 | 4.548e-3 | 3.080e-3 | 6e1cf3fe9fcf4e268e6763491b4efe52 |
| 80x80 | 12849 | 426.5 | 1.057e-3 | 7.356e-4 | 0d0d7efa88c84cfe966af7436a3fa6a8 |
| Orders | | | 2.24, 2.11 | 2.12, 2.07 | |

u is the larger component on every grid. Its worst station is y = 0.9375, the one nearest the
lid, on all three grids, with the profile below Marchi's; v's is x = 0.75 at 20x20 and 0.6875
after, above Marchi's. CI runs `test_lid_driven_cavity_staggered_val002` on the case file's 40x40
(decision 4): it passed in 56.3 s alone and 72.1 s in the full suite beside another solve.

**Ghia beside, no threshold.** `cavity_true_centerline_errors` on the same fields: u 8.90e-3,
3.99e-3, 4.81e-3 (falls, then rises); v 6.49e-3, 8.25e-3, 8.94e-3 (rises at both steps). At 80x80
the worst Ghia errors are at y = 0.8516 in u and x = 0.8594 in v, the jet by the right wall, where
`docs/reports/cavity_reference_marchi.md` places Ghia's own error. Scored against Ghia, criterion
3a would fail this solution in both components; scored against Marchi it passes.

## 2. What the metric reads

The metric reads the cell-centered field on the true centerlines, the profiles of the r2
metric, and interpolates between nodes by the cubic through the four nearest, as
`self_convergence.marchi_comparison` did. The prediction's figures came from a different
instrument: the evidence report read the staggered faces themselves (`face_profiles`). On the
same fields that instrument gives u 1.562e-2, 3.970e-3, 9.118e-4 and v 1.169e-2, 2.631e-3,
6.054e-4, the evidence report's series to the three digits it gives. The metric reads 37%, 15%
and 16% higher in u, and 14%, 17% and 22% in v. On an even grid the metric's profile on x = 0.5
is the mean of two cell means, the face value plus h^2 / 4 times the second x-derivative of u
(INFERRED), so the difference is second order: in u it falls by 10 and then 4.0 times. It is
largest by the lid. The metric is solver-independent on purpose, and like the faces it converges
at second order.

## 3. Prediction check

Against the orchestrator's prediction, written before the run.
- Criterion 3, about 9.1e-4, passing by about 22 times: passes, but **missed** in size: 1.057e-3,
  18.9 times. The 9.1e-4 is the faces instrument's figure (section 2); on this field it reads
  9.118e-4.
- Criterion 3a series about 1.6e-2, 4.0e-3, 9.1e-4, orders about 2.0 and 2.1: **missed** in size
  for the same reason: 2.144e-2, 4.548e-3, 1.057e-3, orders 2.24 and 2.11. Each component falls
  monotonically: matched. Neither falls unevenly: u 2.24 and 2.11, v 2.12 and 2.07.
- Against Ghia, not monotone in v, 80x80 worst near 0.9% in the jet by the right wall: matched,
  0.894% at x = 0.8594. Not predicted: u is not monotone either.
- 80x80 stops by the rule near 12849 under the new cap: matched, 12849 of 20000.
- The staggered 40x40 CI test takes about a minute: matched, 56.3 s alone.

## 4. What the switch reached (findings)

- **F1, stop hit.** With the case file switched, the full suite failed six tests. Three were on
  decision 2's list (`test_solver_staggered.py:240`, one per cavity preset). Three were not:
  `test_benchmark.py` `test_staggered_velocity_step_stop_has_the_collocated_label`, and in
  `test_solver_staggered.py` `test_quiescent_closed_box_stays_at_rest_and_stops_at_once` (no
  velocity scale, so `error_estimate` refuses at construction) and
  `test_stop_is_reported_and_reset_at_the_start_of_each_solve`. Alex decided that the `_case` and
  `_channel` fixtures pin `velocity_step`, as `stopping_probe.case_config` does, that a test
  wanting `error_estimate` asks through `_ruled`, and that the label test wraps its loader in
  `with_velocity_step`. No assertion changed.
- **F2.** Two of decision 2's entries needed nothing: `test_benchmark.py:303` and
  `test_solver_staggered.py:255` build the collocated solver from the channel, not the cavity.
- **F3, rules that changed.** From `velocity_step` to `error_estimate` with no edit: the staggered
  parametrizations of `test_solver_selection.py` `test_harness_method_selects_the_solver` and
  `test_viewer_method_selects_the_solver` (a 6x6 cavity solved through each script), and the
  staggered solver `TestReferenceVelocity` builds for the three cavity presets (built, not
  solved). From `error_estimate` back to `velocity_step`: the two channel tests of review 25 S9,
  `test_dirichlet_faces_hold_what_apply_normal_velocity_wrote[channel]` and
  `test_outlet_faces_are_extrapolated_before_every_prediction`. Kept on `velocity_step` by a pin:
  every `_case` caller in `test_solver_staggered.py`, the harness label test and the collocated
  VAL-002 test.
- **F4, the S5 seam.** `load_preset` moved the harness's loader choice, so
  `test_harness_row_takes_the_cap_from_the_solver` and
  `test_harness_builds_a_wall_clustered_preset_on_its_clustered_mesh` patch `load_preset` and
  `validation.cases.WALL_CLUSTERED_GRIDS`, assertions unchanged (Alex). With either patch removed
  the test fails (section 6).
- **F5.** Decision 3 moves the harness to the new metric, so
  `test_harness_scores_the_cavity_on_the_true_centerlines` now asserts its name. This test was not
  on the cascade list, and the decision required the edit.
- **F6.** The rows were taken at commit 2 and are committed here, in commit 3: a row cannot name
  the commit that holds it.
- **F7.** The 40x40 row took 120.2 s. A re-solve of the same work (776871 pressure sweeps, equal
  to the row's) took 68.4 s, and the CI test 56.3 s. The cause is throughput on that run, not in
  the tree (UNKNOWN).
- **F8.** Review 25 S1's SYSTEM.md row also lists `scripts/self_convergence.py`, which consumes
  the solver's public shape as `val001_order.py` does.

**Collocated unchanged.** A collocated harness row on a 16x16 cavity at decedb9 and at 1085f28:
u, v and p are bitwise equal, and so is every field of the row except run id, time stamp, commit,
wall times, `params.max_simple_iter` (10000 to 20000, decision 1; both stop at 234) and `grid`,
which adds the axis clustering (test 25 T2).

## 5. What this establishes and what it does not

Established: criteria 3 and 3a pass on the staggered solver under `error_estimate`, against
Marchi, each solve stopping by the rule; the metric converges at second order and reads 15% to
22% above the face values at 40x40 and 80x80; Ghia's table would fail this solution under 3a;
the collocated results and the saved self-convergence fields are untouched. Planted defects: 13
of 13 caught.

Not established: timing beyond single runs (F7); 3a on grids other than 20, 40 and 80. Not
covered by any test: the metric's tests put every station on a node, so replacing the cubic with
linear interpolation passes them; the `_channel` pin changes no assertion's outcome (section 6).
REQ-S03's text and the PROJECT_PLAN gate row change in step 9. Premises: all seven held.

## 6. How each number was taken

- Section 1: the three rows; `results/builder26/ghia_beside.py` re-solves each grid as the harness
  does. Its Marchi values equal the rows' with `==`, as do the outer counts and pressure sweeps,
  so its Ghia values, worst stations and face values (section 2) are on the rows' fields.
  `ghia_beside.json`. CI test times: `suite_commit2.txt` and a lone run.
- Section 4: `probe_suite_cavity_switch.txt` (F1); `sc_bitwise_control.py`: fresh 20x20 staggered
  and collocated `solve_and_save` equal the saved fields bitwise, 629 and 380 outer;
  `collocated_row.py` on a worktree of decedb9 and on 1085f28, compared by `compare_rows.py`.
- Planted defects: `mutate.py` and `mutations.log`: 13 caught, and 2 informational, linear
  interpolation and an unpinned `_channel`, not caught.
