# k and eps by conjugate gradients (ECR-002 step 7, prompt 51)

**Date:** 2026-10-12
**Tree:** branch `feature/turbulence-cg-solve` from main at 06aee2b (prompt 50 merged as PR 77).
Commit A (`2b3fc89`) solves k and eps by diagonal-preconditioned conjugate gradients with a
Jacobi fallback (`src/scalar_scheme.py`, `src/turbulence.py`). Commit B (`cc6996e`) is section 1
alone. The commits C (`4776657`, `4cf12f9` and the probe's later files) are the probe. Commit D
is this report's results and the records.
**Instruments:** `docs/reports/probe51/`: `plant51.py` (commit A's planted defects),
`check51.py` (the laminar hashes and the Jacobi battery), `annex51.py` (item 5),
`product51.py` and `val016_51.py` (item 6), `stop51.py` (section 3), `loop51.py` (section 5),
`tables51.py` (every table below, spliced from the records), `run51.sh` (the stages). Raw
output under `results/builder51/` (untracked).
**Order:** section 1 was committed alone before any run of item 5. Item 4 ran at commit A,
before it. Sections 2 onward were written after the runs.
**Policy:** `docs/REVIEW_POLICY.md`, "What blocks". The numbers live in generated tables; the
sentences point at them.

## 1. Predictions

Stated by the orchestrator on 2026-10-12 in prompt 51, with (f) added by Alex
on 2026-10-12 with the fallback decision. Committed alone, before item 5's
runs. A row is the Annex 20 harness row `val017_216x72` or one of prompt
48's product rows on 80x30 and 160x60.

- (a) The laminar hashes, the pressure loop and the transport gate tests are unchanged bit for bit.
- (b) The mean CG iterations per k or eps solve over the Annex 20 run are under 60, against 272 and 182 Jacobi sweeps.
- (c) Seconds per outer iteration over the whole Annex 20 run fall by at least 40%.
- (d) The Annex 20 row stops within 2% of main's outer count, both scored lines pass, and its line values differ from main's by under 1e-4 u0.
- (e) No positivity error in any run.
- (f) Zero fallbacks on the Annex 20 and product rows.

What each outcome means (prompt 51):

- (c) holds: the remaining step 7 cases and step 8 run on this solve, and
  the pressure correction becomes the largest cost; whether to take
  ADR-013's named successor then goes to Alex with this report's numbers.
- (c) fails while (b) holds: the CG loop's own cost per iteration eats the
  gain, and the next pass looks at the loop, not the method.
- (e) fails: the step stops and reports the cell, the iteration and the
  field. No clipping is added.

Item 4's checks of (a) were run at commit A before this commit and are
reported with the rest.

## 2. What was built (commit A)

`scalar_scheme.implicit_step` takes a method: `jacobi`, the default and the transport
solver's, or `cg`, diagonal-preconditioned conjugate gradients, the k-epsilon model's.

- **The same solve.** The `cg` path starts from C*, as Jacobi does. SOLID and held cells
  are not unknowns: their values stay fixed and their couplings act on the right-hand
  side, which is the system Jacobi solves. The stop is Jacobi's: the largest absolute
  residual over the solved cells, at most `tol` times the largest |b|. It is tested on
  the recursive residual each iteration and confirmed on the true residual, and a failed
  confirmation restarts from the true residual, as `pressure.py`'s loop does.
  `max_iter` counts CG iterations, so `turbulence.tol` and `turbulence.max_iter` keep
  their meaning.
- **The fallback (Alex, 2026-10-12).** A CG solve that reaches its cap, or ends with a
  value at or below zero in a cell it solves for, is discarded. Jacobi then solves from C*
  with the same tolerance and cap. The result is the `jacobi` path's, bit for bit, so
  every result keeps Jacobi's sign guarantee. Section 3 has the case that led to it.
- **The record.** `ImplicitResult` carries `method`, `iterations`, `converged`,
  `fallbacks` and `fallback_sweeps`. The model's `last_sweeps` is now `last_iterations`,
  with `last_fallbacks` beside it, and a fallback logs a warning. The probes of prompts
  46, 47 and 50 read `last_sweeps` and no longer run against this tree. They are left as
  they are, as their own commits measured them (Alex, 2026-10-12).
- **Not shared with the pressure loop.** `pressure.py`'s loop starts from zero and stops
  on a 2-norm with an absolute floor. Sharing it would mean making its start and its
  stop parameters, which changes the pressure module, and this change leaves that module
  alone. The scalar loop is a second loop of the same shape.
- **One BLAS thread.** On the Annex 20 mesh (15,552 cells) a dot product on this
  machine's 24-thread OpenBLAS pool costs many times its one-thread cost, the measurement
  behind `pressure.py`'s `PRESSURE_BLAS_THREADS` repeated for this vector length
  (`results/builder51/bench_dot.py`). The CG loop runs inside its own one-thread
  limit, so a run that does not pin the thread count is not slowed by it.
- **Callers.** The turbulence model names `CONJUGATE_GRADIENT`. The transport solver names
  `JACOBI`: its concentrations span many decades down to zero, and Jacobi's iterates keep
  their sign. A change there is a separate decision.

## 3. The first pass's stop, and the decision

The first pass built `cg` without a fallback. On it, `tests/test_turbulence.py`'s long
positivity run raised a PositivityError in both standard cases: the random field from the
patchy start, 300 pseudo-time steps, the frozen strain growing k without a steady state.
The prompt lists a positivity error as a stop, so the pass stopped. `stop51.py` replays
that pass and assembles the k system at chosen steps:

<!-- BEGIN tables51: stop -->
| Scheme | Step | Largest G / smallest V/dt | Condition number | Dense solve, least k | Jacobi, least k (its, stop) | CG without fallback, least k (its, stop) | Committed CG, least k |
| --- | --- | --- | --- | --- | --- | --- | --- |
| upwind | 4 | 8.6e+03 | 5.7e+03 | 6.273e-03 | 7.346e-03 (500, cap) | 6.273e-03 (139, conv) | 6.273e-03 (fallback 0) |
| upwind | 17 | 6.9e+06 | 2.5e+06 | 1.882e+06 | 1.907e+06 (500, cap) | 1.882e+06 (500, cap) | 1.907e+06 (fallback 1) |
| upwind | 60 | 1.4e+10 | 4.8e+09 | 4.414e+09 | 4.700e+09 (500, cap) | 4.414e+09 (500, cap) | 4.700e+09 (fallback 1) |
| upwind | 120 | 3.0e+14 | 1.1e+14 | 9.704e+13 | 1.033e+14 (500, cap) | 9.706e+13 (500, cap) | 1.033e+14 (fallback 1) |
| upwind | 151 (raises) | 1.9e+17 | 7.1e+15 | 1.090e+17 | 1.073e+17 (500, cap) | -3.525e+16 (500, cap) | 1.073e+17 (fallback 1) |
| umist | 4 | 8.7e+03 | 5.4e+03 | 8.427e-04 | 1.111e-03 (500, cap) | 8.427e-04 (139, conv) | 8.427e-04 (fallback 0) |
| umist | 17 | 7.0e+06 | 2.6e+06 | 1.894e+06 | 1.920e+06 (500, cap) | 1.894e+06 (174, conv) | 1.894e+06 (fallback 0) |
| umist | 60 | 1.4e+10 | 4.8e+09 | 4.441e+09 | 4.729e+09 (500, cap) | 4.441e+09 (500, cap) | 4.729e+09 (fallback 1) |
| umist | 120 | 3.0e+14 | 1.1e+14 | 9.747e+13 | 1.038e+14 (500, cap) | 9.741e+13 (500, cap) | 1.038e+14 (fallback 1) |
| umist | 151 (raises) | 2.8e+17 | 1.4e+16 | 6.430e+16 | 7.903e+16 (500, cap) | -2.568e+18 (500, cap) | 7.903e+16 (fallback 1) |
<!-- END tables51: stop -->

By step 60 both methods reach the cap under both schemes. There the stop sits below the
rounding of the residual itself, since the conductances outweigh V/dt by orders, so neither
method can meet it. While the condition number stays below the reciprocal of double
precision's epsilon, the capped CG iterate matches the dense solve. Past it, where the dense
solve is itself at the edge of what double precision resolves, the CG iterate is negative in
every cell, while Jacobi, a non-negative
combination at every sweep, stays positive. The RNG cases converge every solve over the 300
steps under CG, so the old test's own premise, that its solves are cut off, no longer held
for them.

Alex's decision of 2026-10-12: fall back to Jacobi from C* on the cap or on any value at or
below zero, count the fallbacks, and add prediction (f). The long-run test was split: the
standard cases pin positivity through the fallback (every truncated step falls back, and k
and eps stay positive over all 300 steps), and the RNG cases are a separate test that every
solve converges and stays positive.

## 4. Unchanged by construction, and the planted defects (items 3 and 4)

Item 4, measured at commit A (`2b3fc89`) against a worktree of main at the branch point
(`06aee2b`), one machine. The laminar cases are step 4's measure (the committed solver on the
preset, SHA-256 over the stopped faces). The Jacobi battery is 240 systems through
`implicit_step` with no method named: conductances from 1e-3 to 1e3, steps from 1e-6 to
1e6 s, one step or one per cell, with and without a sink and held cells, caps from 1 to
5,000 sweeps, every field's bytes and every count and flag hashed in order
(`check51.py`).

<!-- BEGIN tables51: item4 -->
| Case | Stop (main / A) | Count (main / A) | Hash, 16 hex (main / A) | Bitwise |
| --- | --- | --- | --- | --- |
| val001_80x40 | error_estimate_and_continuity / error_estimate_and_continuity | 1,559 / 1,559 | `c7d88d919e1141a5` / `c7d88d919e1141a5` | yes |
| val001_80x40_stretched | error_estimate_and_continuity / error_estimate_and_continuity | 1,124 / 1,124 | `9838d632f181d90f` / `9838d632f181d90f` | yes |
| val002_80x80 | error_estimate_and_continuity / error_estimate_and_continuity | 12,814 / 12,814 | `a8a61e28120e3f44` / `a8a61e28120e3f44` | yes |
| Jacobi battery (240 systems) | 139 capped / 139 capped | 69,305 / 69,305 sweeps | `74c989431260d90e` / `74c989431260d90e` | yes |
<!-- END tables51: item4 -->

`src/pressure.py` is not in the branch's diff, and the laminar runs pass through it, so its
results are main's. The transport gate tests (VAL-003, VAL-004, VAL-007, VAL-012, VAL-013,
VAL-014) pass unchanged, with the 730 tests of the 18 files that import what commit A
changed.

Each plant below was applied to the committed source, the guarding test files run, and the
source restored (`plant51.py`):

<!-- BEGIN tables51: plants -->
| Plant | What it stands for | Outcome | Failing tests |
| --- | --- | --- | --- |
| `two_norm_stop` | CG's own 2-norm stop, \|\|r\|\| <= tol \|\|b\|\|, in place of the max-norm one | caught | `TestConjugateGradient::test_the_stop_is_the_largest_residual_over_the_largest_rhs` |
| `held_unknown` | a held cell left as an unknown under CG | caught | `TestConjugateGradient::test_cg_and_jacobi_agree_and_both_match_a_dense_solve`; `TestConjugateGradient::test_a_held_cell_keeps_its_value_exactly`; `TestBoundaries::test_held_eps_keeps_its_value_and_its_neighbours_read_it` |
| `transport_on_cg` | the transport solver on CG | caught | `TestConjugateGradient::test_the_transport_solver_asks_for_jacobi` |
| `turbulence_on_jacobi` | the model's k solve left on the default, Jacobi | caught | `TestPositivity::test_a_long_standard_run_stays_positive_through_the_fallback[upwind]`; `TestPositivity::test_a_long_standard_run_stays_positive_through_the_fallback[umist]`; `TestPositivity::test_a_long_rng_run_converges_and_stays_positive[upwind]`; and 2 more |
| `fallback_from_cg_iterate` | the fallback restarts from the CG iterate, not C* | caught | `TestConjugateGradient::test_the_cap_is_honoured_and_falls_back`; `TestConjugateGradient::test_a_negative_converged_cg_value_falls_back_to_jacobi_from_c_star`; `TestConjugateGradient::test_a_zero_converged_cg_value_falls_back_too`; and 2 more |
| `fallback_on_cap_only` | the fallback on the cap only, not on a non-positive value | caught | `TestConjugateGradient::test_a_negative_converged_cg_value_falls_back_to_jacobi_from_c_star`; `TestConjugateGradient::test_a_zero_converged_cg_value_falls_back_too` |
| `fallback_below_zero_only` | the fallback on a negative value only, a zero kept | caught | `TestConjugateGradient::test_a_zero_converged_cg_value_falls_back_too` |
| `no_fallback` | no fallback at all, the CG iterate returned whatever it is | caught | `TestConjugateGradient::test_the_cap_is_honoured_and_falls_back`; `TestConjugateGradient::test_a_negative_converged_cg_value_falls_back_to_jacobi_from_c_star`; `TestConjugateGradient::test_a_zero_converged_cg_value_falls_back_too`; and 2 more |
<!-- END tables51: plants -->

## 5. Cost: the Annex 20 row (item 5)

The harness row `val017_216x72` on main (a worktree at `06aee2b`) and on this branch (a
clean worktree at `4cf12f9`, whose `src/` is commit A's), launched together, with nothing
else running on the machine, one BLAS thread each, the harness's `concurrent` set to 2. The
branch's row is appended to `benchmarks/results.jsonl`. The k and eps counts are means over
every outer iteration's two solves: Jacobi sweeps on main, CG iterations on the branch
(`annex51.py`).

<!-- BEGIN tables51: annex_cost -->
| Row | Stop | Outer | Wall s | s per outer | k its per solve | eps its per solve | CG its per p' solve | p' caps | Fallbacks | x3 RMS / allowance | y2916 RMS / allowance | Scored lines |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main (Jacobi), `06aee2b` | error_estimate_and_continuity | 11,441 | 2,782 | 0.2432 | 276.2 | 184.4 | 505.4 | 0 | 0 | 0.0593 / 0.0703 | 0.0797 / 0.1356 | pass |
| branch (CG), `4cf12f9` | error_estimate_and_continuity | 11,441 | 2,034 | 0.1778 | 47.8 | 37.1 | 505.4 | 0 | 0 | 0.0593 / 0.0703 | 0.0797 / 0.1356 | pass |
<!-- END tables51: annex_cost -->

The harness times each stage of every outer iteration (`StaggeredSolver.stage_seconds`):

<!-- BEGIN tables51: annex_stages -->
| Stage | Main s | Main s per outer | Main share | Branch s | Branch s per outer | Branch share | Change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| momentum | 222 | 0.0194 | 8.0% | 224 | 0.0196 | 11.0% | +0.6% |
| pressure | 1,296 | 0.1133 | 46.6% | 1,338 | 0.1170 | 65.8% | +3.3% |
| correct | 4 | 0.0004 | 0.2% | 5 | 0.0004 | 0.2% | +3.6% |
| turbulence | 1,257 | 0.1099 | 45.2% | 465 | 0.0407 | 22.9% | -63.0% |
| the turbulence stage's fall that a 40% fall of the whole run needs |  |  |  |  |  |  | -89% |
<!-- END tables51: annex_stages -->

One iteration of each loop, isolated on a field of the Annex 20 row's size, by part
(`loop51.py`, run alone after the rows):

<!-- BEGIN tables51: loop -->
| Per iteration, 216 x 72, one thread | ms |
| --- | --- |
| Jacobi sweep | 0.1344 |
| CG iteration | 0.1767 |
| CG part: product | 0.1061 |
| CG part: of which masked zeroing | 0.0017 |
| CG part: two dot products | 0.0092 |
| CG part: largest absolute residual | 0.0094 |
| CG part: updates of c and r | 0.0195 |
| CG part: preconditioner and direction | 0.0121 |
| CG iteration over Jacobi sweep | 1.31 |
<!-- END tables51: loop -->

## 6. The answers (item 6)

### 6.1 The Annex 20 row

<!-- BEGIN tables51: annex_answers -->
| Quantity | Value |
| --- | --- |
| Outer count, branch against main | +0.00% |
| Seconds per outer iteration, branch against main | -26.9% |
| Largest difference on line x3, u / u0 | 3.64e-08 |
| Largest difference on line x6, u / u0 | 3.21e-08 |
| Largest difference on line y2916, u / u0 | 4.01e-08 |
| Largest difference on line y0084, u / u0 | 3.65e-08 |
| Largest line difference, u / u0 | 4.01e-08 |
<!-- END tables51: annex_answers -->

### 6.2 The product rows

Prompt 48's standard upwind rows, run with each tree's committed solver and rule on
prompt 47's configuration builder; main's and the branch's 80x30 and 160x60 runs were
launched together, before the Annex 20 pair (`product51.py`).

<!-- BEGIN tables51: products -->
| Grid | Row | Stop | Outer | Prompt 50's outer | Wall s | s per outer | k its per solve | eps its per solve | Fallbacks | k/eps caps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 80x30 | main (Jacobi), `06aee2b` | error_estimate_and_continuity | 1,021 | 1,021 | 36 | 0.0348 | 19.4 | 15.7 | 0 | 0 |
| 80x30 | branch (CG), `4776657` | error_estimate_and_continuity | 1,021 | 1,021 | 36 | 0.0349 | 11.3 | 9.6 | 0 | 0 |
| 160x60 | main (Jacobi), `06aee2b` | error_estimate_and_continuity | 1,954 | 1,954 | 206 | 0.1055 | 37.3 | 28.2 | 0 | 0 |
| 160x60 | branch (CG), `4776657` | error_estimate_and_continuity | 1,954 | 1,954 | 201 | 0.1028 | 18.0 | 16.0 | 0 | 0 |
<!-- END tables51: products -->

<!-- BEGIN tables51: product_faces -->
| Grid | Velocity scale, m/s | Main's faces bitwise prompt 50's | Largest face difference, branch vs main, over the scale | Same, branch vs prompt 50 | s per outer, branch vs main |
| --- | --- | --- | --- | --- | --- |
| 80x30 | 0.45 | yes | 2.45e-09 | 2.45e-09 | +0.3% |
| 160x60 | 0.45 | yes | 1.87e-08 | 1.87e-08 | -2.5% |
<!-- END tables51: product_faces -->

### 6.3 VAL-016 under CG

The 12-row standard Couette solve of `tests/test_turbulent_channel.py`, measured as its
tests measure it, on main and on the branch (`val016_51.py`). Criterion (b) runs the
one-dimensional reference alone, which imports nothing from `src/`, so it cannot move; it
is listed so the criteria are whole.

<!-- BEGIN tables51: val016 -->
| Tree | Stop | Outer | k / eps its per solve | Fallbacks | (a) max du / U_w (< 1e-4) | (a) max dk / u_tau^2 (< 1e-4) | (a) u_tau / 1D | (b) standard core k ratio | (b) RNG core k ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main (Jacobi), `06aee2b` | error_estimate_and_continuity | 258 | 30.0 / 23.2 | 0 | 7.03e-08 | 3.74e-06 | 1.000000 | 0.99946 to 1.00146 | 0.99941 to 1.00108 |
| branch (CG), `4cf12f9` | error_estimate_and_continuity | 258 | 17.4 / 14.7 | 0 | 7.03e-08 | 3.74e-06 | 1.000000 | 0.99946 to 1.00146 | 0.99941 to 1.00108 |
<!-- END tables51: val016 -->

## 7. Predictions against measurement

<!-- BEGIN tables51: predictions -->
|  | Prediction | Measured | Verdict |
| --- | --- | --- | --- |
| (a) | laminar hashes, pressure loop and transport gate unchanged bit for bit | all three laminar cases and the 240-system Jacobi battery bitwise main's | holds |
| (b) | mean CG iterations per k or eps solve under 60 (Jacobi 272 and 182) | k 47.8, eps 37.1 (this run's Jacobi: 276.2, 184.4) | holds |
| (c) | seconds per outer iteration fall by at least 40% | 26.9% lower | FAILS |
| (d) | stop within 2% of main's count, both scored lines pass, lines within 1e-4 u0 | count +0.00%, scored lines pass, largest line difference 4.0e-08 u0 | holds |
| (e) | no positivity error in any run | none | holds |
| (f) | zero fallbacks on the Annex 20 and product rows | 0 | holds |
| plants | every planted defect fails a test | 8 of 8 |  |
<!-- END tables51: predictions -->

## 8. What this implies

Prediction (c) fails and (b) holds, which the prompt reads as "the CG loop's own cost per
iteration eats the gain, and the next pass looks at the loop, not the method". The stage
times in section 5 contradict the premise behind that reading, so this section reports
them first and gives the question to Alex.

- **The premise.** The prompt took the k and eps solves as about half of an outer
  iteration and the pressure correction as about 40%, from a one-thread cProfile of
  prompt 49's run. The harness times its stages on every row. On main's row, the pressure
  stage is the larger of the two, and the whole turbulence stage, the solves together with
  the advection, the terms, the conductances and the checks, is a little under half of the
  run. A 40% fall of the run needs the turbulence stage to fall by the amount in the
  stages table's last row. That is more than the solves could give even at no cost,
  because the stage's other work stays.
- **The loop.** In isolation a CG iteration costs about a third more than a Jacobi sweep
  (the loop table). The extra is the two dot products, the largest-residual check and the
  two vector updates, the usual price of CG over Jacobi, and the product dominates both.
  The loop is not where the shortfall comes from. A pass that tunes it would at best take a
  part of the turbulence stage's remainder.
- **What changed.** The k and eps solves take a fraction of their Jacobi count (prediction
  (b)). The turbulence stage fell by the change in the stages table, and the row by the
  change in the answers table. The stop, the scored lines and the fields are unchanged to
  the line differences of section 6.1. No solve reached its cap or fell back on any row
  (predictions (e) and (f)).
- **Where the cost is now.** The pressure correction is the largest stage on the branch's
  row, the larger part of it (the stages table). That is the question the prompt gives to the outcome where
  (c) holds: whether to take ADR-013's named successor, geometric multigrid preconditioning
  of the pressure solve.

Questions for Alex, not decisions:

1. **The next pass.** By the prompt's outcome table, (c) failing sends the next pass to the
   CG loop. By the stage times, the remaining cost is the pressure correction's. Which
   does the next pass take?
2. **Issue 58.** Its question was what an unconverged k or eps solve does. Now `max_iter`
   counts CG iterations, and a CG solve that reaches it falls back to Jacobi from C*, which
   returns with a warning as before, now as a Jacobi iterate. No row in this report reached
   the cap. Whether a capped solve should raise or count against the stopping rule is still
   open. This note is in the records (STATUS); it has not been posted on the issue. Post it
   there?
3. **The transport solver** stays on Jacobi, as the prompt set. Its gate rows are small and
   their Jacobi solves are not a measured cost, so no change is proposed.

## 9. Reproducing

From the repository root, `.venv` active, prompt 50's product records under
`results/builder50/` (untracked; its report says how to remake them):

```
git worktree add --detach results/builder51/base 06aee2b      # main at the branch point
git worktree add --detach results/builder51/cg 4cf12f9        # a clean tree of the branch
bash docs/reports/probe51/run51.sh plants      # commit A's plants, alone, about ten minutes
bash docs/reports/probe51/run51.sh laminar     # item 4, about eight minutes
bash docs/reports/probe51/run51.sh products    # item 6's product rows, about four minutes
bash docs/reports/probe51/run51.sh annex       # item 5, alone, about 47 minutes
bash docs/reports/probe51/run51.sh val016      # item 6, VAL-016
bash docs/reports/probe51/run51.sh stop        # section 3's replay
python docs/reports/probe51/loop51.py          # section 5's loop table, alone
bash docs/reports/probe51/run51.sh compare
bash docs/reports/probe51/run51.sh tables      # every table, spliced from the records
```

The Annex 20 rows were launched together, so their `concurrent` is 2. Prompt 50's stored
row of the same case was timed on another occasion and is not a timing reference for this
report, which compares only rows launched together.
