# VAL-001 revalidation on the staggered solver (ECR-001 step 7)

**Date:** 2026-09-25
**Context:** ECR-001 step 7, acceptance criteria 1, 2 and 4, with the channel case moved to the
`error_estimate` stopping rule.
**Instruments:** `tests/test_poiseuille.py`, `scripts/benchmark.py` (two rows in
`benchmarks/results.jsonl`), `scripts/val001_order.py` (`results/val001_order/summary.json`,
gitignored). One-offs under `results/builder25/` (section 6).

**Answers.** Criterion 1 passes: 4.104e-4 on 80x40 uniform, 24 times inside 1%. Criterion 2
passes: 3.024e-3 on 80x40 with y clustered to a wall cell of 0.1 H / ny, 3.3 times inside. Criterion
4 passes: the reference-free order on 40x20, 80x40 and 160x80 is 1.99. Every solve stopped by
`error_estimate_and_continuity`, none at its cap.

## 1. Criteria 1 and 2

| MEASURED | Mesh | Outer | Row s (test s) | Worst imbalance | Metric | Row run_id |
|---|---|---|---|---|---|---|
| Criterion 1 | 80x40 uniform | 3154 | 100.9 (101.5) | 1.11e-11 | 4.104e-4 | b8a2f3dfe8084cc7ba4e26c11023f6fd |
| Criterion 2 | 80x40, y clustered | 1523 | 33.9 (34.8) | 1.97e-11 | 3.024e-3 | d2a57fe1b1794c16b8e366b6af49c590 |

Both rows: commit 0ca4ace, clean tree, one process, `params.stopping_rule` `error_estimate` at
tolerances 1e-6 and 1e-10. The collocated test, unchanged at 2.5% under `velocity_step`, took
155.1 s and scored 2.036e-2, as its stored rows do. Test times are one run each, alone; section 9 of
`docs/reports/stopping_rule_evidence.md` puts single-run spread at about 10%.

**The criterion 2 mesh.** Wall cell 0.00125 at both walls (0.1 H / ny, H = 0.5), ratio derived
1.2057, largest cell 3.49 times the uniform 0.0125, x uniform. It is the preset
`val001_80x40_stretched`, built from the case file by `validation.cases.load_wall_clustered`.

**Where the criterion 2 error comes from.** With v = 0 and u independent of x the momentum stencil
reduces to its y diffusion with the half-cell wall distance, and that one-dimensional system can be
solved directly (`results/builder25/developed_profile.py`). Its control: on the uniform mesh it
gives 5.574e-4 and 2.152e-3 at ny = 40 and 20, the closed form of the evidence report, section 3.
On the clustered mesh the same fully developed profile scores 2.893e-3 by the metric, 96% of the
solve's 3.024e-3 (MEASURED). So the criterion 2 number is the clustered stencil's own error on the
developed flow, not iteration error or development. On the clustered family (0.1 H / ny at every
ny) it falls 1.143e-2, 2.893e-3, 7.14e-4 from ny = 20 to 80, orders 1.98 and 2.02: second order,
and above 1% at 40x20, so criterion 2 holds at the 80x40 it names and would not at half that.
INFERRED, not separated: on a geometric mesh the midpoint of two centres sits (h_N - h_P) / 4 off
their face, which changes the diffusion of a parabola by (r - 1)^2 / (4 r), 0.88% at r = 1.2057,
while the wall stencil's error, all of the uniform grid's, shrinks with the wall cell. Where two
neighbouring cells are equal, as at the centre, the stencil is exact for a parabola. The metric
sums rows unweighted by `dy_cell`, so on this mesh it weights the wall region more than the
uniform grid does; criterion 2 is judged by the metric as written.

## 2. Criterion 4: the order

Decision 5 (ECR-001 section 9, note under criterion 4): judged by the reference-free order. Each
profile is u at x = L/2 exactly, the mean of the two columns either side of that face; fine
profiles are restricted to the coarse rows by averaging pairs; p = log2(||d1|| / ||R d2||).

**Control first.** Synthetic P + A sin(4 pi x / L)(1 + s) + U h^q G through the same functions,
q = 2 and 1: worst |p - q| 0.0004 and 0.016, against a stop at 0.05. Planted in the script, a
restriction that samples every other row and an L/2 station read off column nx // 2 each stop it
(section 5).

| MEASURED | 40x20 | 80x40 | 160x80 | Orders |
|---|---|---|---|---|
| Outer, seconds | 880, 22 | 3154, 102 | 10009, 676 | |
| Reference-free, RMS (max) | | | | **1.993** (1.991) |
| L2 vs parabola at L/2, every row | 2.198e-3 | 4.541e-4 | 1.571e-4 | 2.28, 1.53 |
| L2 vs parabola at 3L/4, every row | 2.272e-3 | 5.420e-4 | 1.164e-4 | 2.07, 2.22 |
| L/2 minus 3L/4, over the parabola's norm | 2.11e-4 | 1.92e-4 | 1.93e-4 | |
| Metric (column nx // 2, FLUID rows) | 1.999e-3 | 4.104e-4 | 1.570e-4 | |

**The development floor.** The profile at L/2 differs from the one at 3L/4 by 1.9e-4 of the
parabola's norm on every grid from 80x40, flat under refinement, as the evidence report found on
the metric's rows and columns (1.13e-4 and 1.07e-4, section 3). That difference is the flow still
developing, which refinement does not remove, and at 160x80 it is larger than the whole error
against the parabola at L/2. So the order against the parabola at L/2 falls from 2.28 to 1.53: it
measures the approach to that floor, not the scheme. At 3L/4, further developed, it stays near 2.

**Limit of the instrument.** The two-column station and the pair restriction each carry an O(h^2)
error of their own. On a parabola the restriction misses by P'' h^2 / 32, 9.4e-5 at 40x20, about
half the signal (d1 RMS 1.69e-4), so the instrument pulls a lower order toward 2: synthetic q = 1.5
and 1.8 at about twice the real signal read 1.62 and 1.85. On these fields the pull is absent. With
the parabola subtracted from every grid first, which leaves the differences alone and moves the
restriction onto the small f - P, the order is 1.996 against the judged 1.993
(`results/builder25/restriction_bias.json`). The instrument cannot show an order above 2.

## 3. Prediction check

Against the orchestrator's prediction, written before the run.
- Criterion 1 about 4.1e-4, passing by about 24 times: matched, 4.104e-4, 24.4 times.
- Criterion 2 passes below 1%: matched. Its metric from 2e-4 to 2e-3: **missed**, 3.024e-3. It
  rose 7.4 times from uniform; section 1 traces 96% of it to the developed-flow stencil error.
- Reference-free order from 1.8 to 2.3: matched, 1.993.
- Against the parabola at L/2 well away from 2: matched at the fine pair, 1.53 (2.28 at the coarse
  pair). At 3L/4 nearer 2: matched, 2.07 and 2.22.
- 160x80 under 45 minutes, not at its cap: matched, 676 s and 10009 of 20000.

## 4. What the switch reached (findings)

Moving the case file to `error_estimate` (decision 1) reached callers the plan did not list.
- **F1, stop hit.** `tests/test_solver_staggered.py`, the channel reference-velocity test, builds
  the collocated solver from the channel case, which now raises. Alex decided on one helper,
  `validation.cases.with_velocity_step`, used by that test, the collocated VAL-001 test, the
  harness's collocated path and the viewer, with no assertion changed. No other existing test
  builds the collocated solver from the channel.
- **F2.** `scripts/stopping_probe.py` `case_config` without a rule read the case file's. A fresh
  run would have solved its 1e-11 and 1e-13 truths under `error_estimate`, stopping once the
  estimated iteration error reached 1e-6 of U. It now pins `velocity_step` when no rule is named.
  The saved truths are unaffected.
- **F3.** `scripts/view_field.py` defaults to the collocated solver, which would now raise on
  VAL-001, and it would have solved the stretched preset uniform under its name; its `streamplot`
  needs evenly spaced points besides. It gives its collocated solve `velocity_step` and refuses a
  wall-clustered preset.
- **F4.** The planted defect "spacing 0.1 L / nx" gives 0.00125 at 80x40 too (L / nx = H / ny on
  this domain for every 2:1 grid), so the preset's own test cannot see it; an 80x20 case does.

These carry the size overrun: wiring 137 added lines against about 60, tests 328 against about
150. It follows from decision 1's reach, which the size estimate did not allow for. The order
script is 186 lines against about 150 (its docstring, control and reuse check), and this report
148 against about 120 (the findings above).

**One label.** A velocity-step stop is `residual_below_tol` for both solvers, the label all 34
stored rows carry; the staggered solver's own `velocity_step_below_tol` is mapped in the harness.
The other direction would have changed every new collocated row. The summary keys on
`params.stopping_rule`, absent read as `velocity_step`, and notes a case with rows under both
(review 24 S3, GitHub issue 38).

**Collocated rows unchanged.** A collocated harness row on the channel at 16x8, at 078bd32 and at
0ca4ace: u, v and p bitwise equal, and every field of the row equal except its run id, time stamp,
commit and wall times (`results/builder25/compare_rows.log`).

## 5. What this establishes and what it does not

Established: criteria 1, 2 and 4 pass on the staggered solver under `error_estimate`, each solve
stopping by the rule; the criterion 2 error is the clustered stencil's error on developed flow and
second order on its own family; the collocated results are untouched. Planted defects: 14 of 14
caught (`results/builder25/mutations.log`).

Not established: an order above 2 (section 2); how the criterion 2 error splits between the face
offset and the wall stencil; criterion 2 at any other ny; timing beyond single runs. REQ-S02's text
changes in step 9; the cavity, its cap and its metric are step 8. All seven of the prompt's
premises held. One reading to note: the metric reads column nx // 2, centred at L/2 + dx/2, so
criteria 1 and 2 are scored half a cell downstream of L/2; the order study reads L/2 exactly.

## 6. How each number was taken

- Section 1: the two rows; `results/builder25/val001_tests.log` for the test times; the criterion
  2 imbalance from a re-solve with the same outer count and metric,
  `results/builder25/stretched_imbalance.log`; `developed_profile.json` for the developed profiles.
- Section 2: `results/val001_order/summary.json` from `python scripts/val001_order.py`; a rerun
  reused all three solves in 0.6 s. The restriction check: `results/builder25/restriction_bias.py`.
- Section 4: `results/builder25/mutate.py` and `mutations.log`; `collocated_row.py` run on a
  worktree of 078bd32 and on 0ca4ace, compared by `compare_rows.py`.
