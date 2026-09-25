# What the stopping rule should measure: evidence

**Date:** 2026-09-24
**Context:** the STATUS question of whether the stopping rule needs a continuity term.
Sections 1 to 8 measure and do not choose a rule; section 9 checks the rule chosen from them.
**Instrument:** `python scripts/stopping_probe.py`. Every value is from
`results/stopping_probe/summary.json` (gitignored) unless section 8 or 9 says otherwise.

**Answers.** (1) 82% of VAL-001's stored 2.3e-3 is iteration error; the rest is the
half-cell wall stencil's discretization error, less some development at x = L/2. (2) Step
times rho_hat / (1 - rho_hat) is within 0.77 to 1.34 of the true error on the cavity from
1e-5 to 1e-10. On the channel it fails below 1e-8 (40x20) and 1e-9 (80x40), where the error
left is a flux drift only the imbalance shows. (3) The imbalance crosses 1e-10 only when the
velocity iteration error is far below every discretization error. (4) At 1e-6 the iteration
error exceeds the discretization error on the 80x80 cavity and on VAL-001 80x40.

## 1. Method

Each case was solved once with its committed settings, except `convergence_tol` (1e-11, in
memory) and the cap (20000 outer on the channel, 40000 on the cavity). The field at 1e-11 is
the truth; u and v were kept when the residual first fell below each quarter decade from 1e-5.
The corrector's `correct` was wrapped on the instance to record the worst per-cell imbalance
and the sweeps; `src/` is unchanged. True error: the largest abs difference from the truth
over FLUID cells, u or v. rho_hat: exp of the least-squares slope of log(residual) over the
last W = 100 outer iterations, repeated at 50 and 200. Discretization error: the truth's
`poiseuille_l2_error`, or the truth against `marchi_2009_re100` at its 30 stations through
`face_profiles` and `lagrange`. Iteration error in those terms is a norm of the snapshot's
difference from the truth: on the cavity the largest difference at the 30 stations, on the
channel the L2 of u_snap - u_truth on the metric's column over the metric's u_ref norm.

## 2. Controls

- **Same computation.** Unwrapped solves at 1e-6 equal the wrapped snapshots bitwise: cavity
  20x20 (629 outer) and VAL-001 80x40 (570 outer). The VAL-001 metric, 0.002306261116980951,
  equals the stored row with `==`. That row's commit, 0e7f5b0, is a pre-rebase commit, and
  `src/` at HEAD is identical to `src/` at 0e7f5b0.
- **Estimator.** The tests recover rho from geometric histories, from a quadratic log
  (a + 2 b m), and from a history exactly W long. A window one entry long, short or shifted,
  a fit to raw values, and a guard that refuses a history of W entries each fail a test,
  planted in the script.
- **True error.** 2^-20 added to or taken from a FLUID cell of u or v reads as exactly
  2^-20, and in the BOUNDARY ring as zero. Dropping the FLUID mask, v, or `abs` fails a test.

## 3. Question 1: VAL-001's 2.3e-3

**The premise failed on reading.** `src/momentum.py` gives the wall row the diffusive flux
(u_0 - 0) / (dy / 2), which a parabola does not satisfy. With the interior three-point
difference exact for a parabola, the fully developed discrete profile is (INFERRED, by hand)

    u_j = C [y_j (H - y_j) + dy^2 / 4],   C = 6 U / (H^2 (1 + 2 / ny^2))

so u_num / u_ref = (1 + dy^2 / (4 y (H - y))) / (1 + 2 / ny^2): below 1 at mid-channel,
rising toward the walls, not flat. The truth meets it downstream: the difference falls to
1.5e-6 and 1.2e-6 at the last FLUID column (the table's last row). A one-off 16x8 run, not
committed, gave L2 0.01288 against the truth's 0.01290, where 1/(2 ny^2) gives 0.0078.
Closed form from ny = 20 to 40: L2 2.152e-3 to 5.574e-4, ratio 3.86, observed order 1.95.

| MEASURED | 40x20 | 80x40 |
|---|---|---|
| metric at the committed 1e-6; of it, iteration error (it minus the truth's) | 2.134e-3; 1.35e-4 | 2.306e-3; 1.90e-3 |
| truth at x = L/4, L/2 (the metric), 3L/4 | 3.16e-2, 1.999e-3, 2.112e-3 | 3.27e-2, 4.108e-4, 5.183e-4 |
| closed form; prompt's 1/(2 ny^2) | 2.152e-3; 1.25e-3 | 5.574e-4; 3.13e-4 |
| u_num / u_ref at L/2, min to max | 0.99760 to 1.00297 | 0.99947 to 1.00158 |
| max abs(truth - closed form) in u at L/2, 3L/4, last FLUID column | 4.3e-5, 1.2e-5, 1.5e-6 | 4.4e-5, 1.2e-5, 1.2e-6 |

The truth approaches the closed form along the channel, identically on both grids, so the
closed form is the fully developed discrete solution and the gap at L/2 is the flow's own
development, which refinement does not remove. It lowers the metric at L/2. From 40x20 to
80x40 the truth at L/2 gives ratio 4.87, order 2.28; at 3L/4, 4.07, order 2.03.

**Composition, MEASURED, outside the readings.** Iteration error is 82% of the stored
2.3e-3 at 80x40 and 6% of 2.13e-3 at 40x20; the metric settles at 4.1e-4 from 1e-8.
**Readings.** R1 not matched. The truth is 1.6 and 1.3 times the prompt's prediction, and
u_num / u_ref spreads 5.4e-3 and 2.1e-3 rather than 1e-4. Against the closed form it is 7%
and 26% below, and the ratio is still not flat. R2 matched: L/2 and 3L/4 differ by 1.13e-4
and 1.07e-4. R3 matched: the ratio is highest at the wall-adjacent FLUID row, which points
at the wall stencil, as the closed form shows. R4 not matched: both grids reached 1e-11.

## 4. Question 2: the estimate step times rho_hat / (1 - rho_hat)

| Estimate / true error, W = 100 | 1e-5 to 1e-8 | 1e-5 to 1e-10 | first outside [0.5, 2] |
|---|---|---|---|
| Cavity 20x20 | 0.77 to 1.00 | 0.77 to 1.26 | 3.2e-11 (2.10) |
| Cavity 40x40 | 0.92 to 1.01 | 0.90 to 1.34 | 5.6e-11 (3.23) |
| Cavity 80x80 | 0.98 to 1.02 | 0.94 to 1.29 | 1.8e-11 (2.90) |
| VAL-001 40x20 | 0.47 to 1.31 | 0.08 to 9.80 | 1e-8 (0.47, 2 sweeps) |
| VAL-001 80x40 | 1.00 to 1.24 | 0.15 to 1.74 | 1e-9 (0.31, 1 sweep) |

**Window.** From 1e-5 to 1e-8, halving or doubling W moves no ratio by more than 0.015
(0.095 on VAL-001 80x40 near 1e-5). Off the geometric regime the three differ by up to 4x
(80x40 at 5.6e-10: 1.86, 1.25, 0.43); on 40x20's first four snapshots W = 200 has no rate.
**Below 1e-10 on the cavity** four snapshots leave [0.5, 2]. At 20x20 3.2e-11 and 40x40 and
80x80 1.8e-11 the truth's own error by rate (3.9e-9, 5.5e-9, 4.7e-8) exceeds the snapshot's,
so they do not test the estimate. The miss at 40x40 5.6e-11 is unexplained: adding the
truth's whole error still leaves 2.27. The script no longer writes `ratio_net`: a difference
of two estimates, it goes negative there, and this report does not use it.

**The channel failure.** It begins as the pressure solve falls to one or two sweeps per
outer iteration. The true error then levels off at 1.1e-6 to 1.5e-6 on both grids while the
residual falls, with its maximum at the last FLUID column (`worst_cell`). A flux drift
predicts it: worst imbalance times the cells upstream, (nx - 1) ny, over rho H. That is
1.38e-6 against a true 1.30e-6 at 80x40 1e-9, and 1.35e-6 against 1.25e-6 at 40x20 1.78e-9.
The true error is 0.70 to 1.07 times the drift from 1.8e-9 to the truth at 80x40, and 0.85
to 0.97 from 1e-8 to 5.6e-11 at 40x20 except 0.55 at 1e-9 (0.29 and 0.07 at the last two).
The rate estimate falls to 0.04. On the closed cavity the imbalance sums to zero.

**Reading: E1 on the cavity from 1e-5 to 1e-10**, with the 40x40 miss at 5.6e-11
unexplained. **E2** on the channel: a change of rate once the inner solve drops to one or
two sweeps, at an iteration error of about 1.3e-5 of the inlet speed.

## 5. Question 3: where the imbalance crosses 1e-10

| MEASURED | at 1e-6 | first below: outer, residual | true error / velocity either side | discretization |
|---|---|---|---|---|
| Cavity 20x20 | 2.98e-9 | 1370, 2.6e-10 | 2.7e-8 to 1.5e-8 | 1.56e-2 |
| Cavity 40x40 | 8.19e-10 | 3849, 1.0e-9 | 5.3e-7 to 2.9e-7 | 3.97e-3 |
| Cavity 80x80 | 2.18e-10 | 11276, 4.4e-9 | 5.8e-6 to 3.3e-6 | 9.11e-4 |
| VAL-001 40x20 | 8.93e-10 | 767, 1.2e-9 | 1.25e-5 to 3.9e-7 | 2.00e-3 |
| VAL-001 80x40 | 2.31e-10 | 1805, 6.2e-10 | 1.1e-5 to 2.0e-6 | 4.11e-4 |

From 1e-5 the imbalance stays within 3x of its 1e-6 value (step 6's) until the pressure
solve falls to one or two sweeps, then falls with the residual and stays below 1e-10. INFERRED:
the plateau is what the inner solve leaves at `pressure_tol`, not the outer iteration.

## 6. Question 4: iteration error against discretization error, and cost

Iteration error in the metric's terms (section 1) over the discretization error, and the
first snapshot below a tenth by it and by the prompt's true error (`true_error_rel`):

| Case | Discretization | 1e-5 | 1e-6 | 1e-7 | 1e-8 | 1e-9 | Tenth: metric | Tenth: true error |
|---|---|---|---|---|---|---|---|---|
| Cavity 20x20 | 0.0156 | 0.048 | 0.0048 | 0.0005 | 6.4e-5 | 6.1e-6 | 1e-5 or looser | 1e-5 or looser |
| Cavity 40x40 | 0.00397 | 0.68 | 0.068 | 0.0068 | 0.00072 | 7.2e-5 | 1e-6 | 1e-6 |
| Cavity 80x80 | 0.000911 | 11 | 1.1 | 0.11 | 0.011 | 0.0011 | 5.6e-8 | 5.6e-8 |
| VAL-001 40x20 | 0.00200 | 1.3 | 0.11 | 0.0079 | 0.0030 | 6.2e-5 | 5.6e-7 | 3.2e-7 |
| VAL-001 80x40 | 0.000411 | 55 | 5.1 | 0.42 | 0.032 | 0.016 | 1.8e-8 | 1e-8 |

On the cavity the two measures agree. On the channel the true error crosses a quarter decade
later in the table (about 0.4 decades interpolated): it is a maximum over every FLUID cell,
not a root mean square over one column.
Outer iterations and seconds to reach each residual, one run each:

| Case | 1e-5 | 1e-6 | 1e-7 | 1e-8 | 1e-9 | 1e-10 | 1e-11 |
|---|---|---|---|---|---|---|---|
| Cavity 20x20 | 449, 10 | 629, 11 | 809, 12 | 989, 12 | 1243, 13 | 1469, 13 | 1786, 13 |
| Cavity 40x40 | 1257, 42 | 1891, 51 | 2525, 53 | 3158, 55 | 3856, 56 | 4553, 57 | 5384, 59 |
| Cavity 80x80 | 3353, 209 | 5728, 300 | 8083, 322 | 10437, 335 | 12840, 346 | 15275, 357 | 17815, 368 |
| VAL-001 40x20 | 142, 15 | 213, 19 | 284, 19 | 354, 19 | 800, 20 | 958, 20 | 1532, 21 |
| VAL-001 80x40 | 298, 49 | 570, 83 | 841, 88 | 1113, 89 | 1371, 90 | 2331, 93 | 4282, 97 |

To 1e-8 a decade costs a steady 180, 634 and about 2360 outer iterations on the cavity, 71
and 272 on the channel. Against the 1e-7 to 1e-8 decade, the three below it cost 41%, 26%
and 76% more at 20x20, 10%, 10%, 31% at 40x40 and 2%, 3%, 8% at 80x80; the channel's rate
changes. Seconds per decade fall with the sweeps: 91, 22, 14, 11 s on the 80x80 cavity.
The truths' own error: 3.9e-9, 5.5e-9 and 4.7e-8 on the cavity by rate, near 1e-8 and 3e-8
on the channel by the drift (INFERRED).

## 7. What this establishes and what it does not

The committed 1e-6 leaves iteration error that dominates VAL-001 80x40's stored metric and
matches the 80x80 cavity's discretization error. The rate estimate holds while the iteration
is geometric, whatever the window. On the open channel the error that outlasts the velocity
step is flux drift, which the imbalance measures and the step does not. Tightening past 1e-6
costs outer iterations steadily but little time. The evidence supports two quantities: the
rate estimate, and the imbalance accumulated along the flow.
Not established: which rule to adopt; 160x80, stretched meshes or another `pressure_tol`;
that `pressure_tol` sets the plateau; the channel truth's own error beyond the drift; timing
beyond one run (the unwrapped controls took 11.9 s and 85.5 s to 1e-6, wrapped 11.4 and 83.2).

**Premises and stops.** The premise table's 1 / (1 - rho) of 89 at 20x20 should read 79
(rho_hat at 1e-6: 78.7; 276 and 1026 at 40 and 80); the others checked held; 160x80 was not
built. No stop was reached. Solves took 657 s: 97 s for the two controls, 559 s for the five
truths. The pressure solve sat at its cap (2000, 500) at the channel's and 80x80's 1e-5.

## 8. How each number was taken

- Sections 2 to 7: `summary.json` and `run.log` from `python scripts/stopping_probe.py`;
  the question 1 split is the 1e-6 snapshot's `l2` minus the truth's `l2`.
- The drift in section 4: hand arithmetic on `summary.json`, the snapshot's `imbalance`
  times (nx - 1) ny, over rho H with rho = 1 and H = 0.5, set against its `true_error`.
- One-offs, not committed: the 16x8 check (this script with `CASES` set to the cavity at
  8x8 and VAL-001 at 16x8), and the channel's late error growing from inlet to outlet
  (`worst_cell` records only the maximum's cell). Planted defects: `results/builder23/` and
  `results/builder23b/`.

## 9. The rule as built

Alex chose the rule on 2026-09-24 and `src/stopping.py` holds it. Under `stopping_rule:
error_estimate` a solve stops when (a) step times rho_hat / (1 - rho_hat) over the largest
prescribed boundary velocity is below `iteration_error_tol` (1e-6), rho_hat fitted over the
last 100 steps, (b) the worst per-cell imbalance is below `mass_imbalance_tol` (1e-10), and
(c) the summed absolute imbalance over rho times the inflow F (closed: rho times the lid
speed times the side) is below `iteration_error_tol`. The cap is not convergence. (c) came in
the review round: the first build had only (a) and (b), and (b) alone lets the through-flow
drift by up to 1e-10 (nx - 1) ny / (rho H U), 6.3e-6 of the inlet speed at 80x40 and four
times that per refinement. The flux through any cross-section differs from the inflow by at
most the summed imbalance on one side of it, so (c) bounds the drift on any grid.
`velocity_step` stays the default: under it the 20x20 cavity equals
`results/self_convergence/staggered-jacobi_20.npz` bitwise, 629 outer, on main and here.

**Method.** `python scripts/stopping_probe.py --verify-rule` solves each case once under
`error_estimate` at those defaults, set in memory with the truth solve's cap (40000, 20000),
the corrector wrapped as in section 1, and writes `verify_rule.json`. True error: section 1's,
over the velocity scale (lid 1.0, inlet 0.1). The channel is read against a truth solved to
1e-13 (7306 and 2671 outer, worst imbalance 8.6e-14 and 4.7e-14 at 80x40 and 40x20): the
section 1 truths sit 2.25e-7 and 9.0e-8 of U from it, their own flux drift, as test 24
found. The cavity truths stand. Each condition is dated from the start of its final unbroken
run. The script raises unless a fresh rule replaying the saved history stops at the solver's
iteration and no earlier, and the recorded imbalance at the stop is the returned field's; it
re-solves when the saved rule parameters differ. The default rule's figures are the section 1
snapshot at 1e-6.

| MEASURED | Outer (default) | Seconds (default) | (a), (b), (c) from | Last | True error / U | Imbalance | Summed / F | Metric (truth) |
|---|---|---|---|---|---|---|---|---|
| Cavity 20x20 | 1370 (629) | 13.1 (11.4) | 973, 1370, 266 | (b) | 2.26e-8 | 9.92e-11 | 1.2e-8 | |
| Cavity 40x40 | 3849 (1891) | 54.3 (50.8) | 3463, 3849, 946 | (b) | 2.97e-7 | 9.98e-11 | 4.9e-8 | |
| Cavity 80x80 | 12849 (5728) | 338.1 (299.9) | 12849, 11276, 2946 | (a) | 1.02e-6 | 2.34e-11 | 4.9e-8 | |
| VAL-001 40x20 | 880 (213) | 20.3 (18.7) | 880, 767, 748 | (a) | 6.67e-7 | 5.33e-11 | 5.5e-7 | 1.9992e-3 (1.9989e-3) |
| VAL-001 80x40 | 3154 (570) | 96.1 (83.2) | 3154, 1805, 2779 | (a) | 5.44e-7 | 1.11e-11 | 4.6e-7 | 4.104e-4 (4.107e-4) |

Every case stops with `error_estimate_and_continuity`, in 1.07 to 1.16 times the default
rule's wall time. Four fields are bitwise those of the first build; VAL-001 80x40 has the
same residual history to the first build's stop at 2286 and runs on to 3154. Its 96.1 s
against the first build's 101.5 s for fewer iterations shows single-run timings move by
about 10%.

**Prediction check**, against the prediction written before this run. The three cavity stops
unchanged at 1370, 3849 and 12849 with identical fields: matched. VAL-001 40x20 unchanged at
880: matched. VAL-001 80x40 later, within 2300 to 3500: matched, 3154. Its true error at most
1.5e-6 of U against the new truth: matched, 5.44e-7. Its wall time at most 1.5 times 101.5 s:
matched. The first build's own check had missed twice on the channel: VAL-001 80x40 stopped
with 2.76e-6 of U against the old truth (2.99e-6 against the new), and (a) rather than (b)
was met last on both channels.

**Reading.** (c) never binds on the cavity: it holds from long before the stop, and the
summed imbalance there is 20 to 80 times below its bound. On VAL-001 80x40 it moved the
stop: at 2286, where the first build stopped, (a) and (b) held and the summed imbalance was
2.3e-6 of F. (c) held from 1948 to 2082 and then broke: the summed imbalance rose to 2.4e-6
of F at 2371 while no cell passed 6.0e-11. (a) held from 2286 to 2434. The residual rose 4.0
times from its minimum at 2377 to its peak at 2809, with no estimate from 2443 to 2858. (c)
held again from 2779 and (a) from 3154. The true error fell from 2.99e-6 to 5.44e-7 of U,
and the metric is 0.07% below the truth's. The estimate at the stop is 1.83 times the true
error at 80x40 and 1.48 at 40x20, and 0.98 to 1.03 on the cavity. VAL-001 40x20 met (c)
before (a), so (c) changed nothing there.

**For step 7.** (c) bounds the channel's flux drift at `iteration_error_tol` of F on any grid
(arithmetic); (b) stays ECR-001 criterion 6 as written. The 80x80 cavity needs 12849 outer
iterations where the committed cap is 10000: with the committed file it would stop at the cap,
reported as not converged. Planted defects for the rule's tests: `results/builder24/` and
`results/builder24b/`.
