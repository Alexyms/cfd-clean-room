# A multigrid preconditioner for the pressure correction (ADR-013's successor, prompt 53)

**Date:** 2026-10-14
**Tree:** branch `feature/pressure-multigrid` from main at 13b0b9a (prompt 52 merged as PR 79).
Commit 0 (`8ee82d7`) records Alex's decision of 2026-10-14 under ADR-013 decisions 1 and 2.
Commit A (`5c5eff4`) builds the preconditioner (`src/multigrid.py`, `src/pressure.py`,
`src/config.py`). Commit B (`2a178a7`) is section 1 alone. Commit C (`dc97f54` and the probe's
later files) is the probe; commit D this report's results and the records.
**Instruments:** `docs/reports/probe53/`: `capture53.py` (item 6's systems), `common53.py` (a
captured system and one timed solve), `dev53.py` (nu and the power iterations), `plant53.py`
(commit A's planted defects), `check53.py` (the laminar cases), `measure53.py` (item 6),
`anatomy53.py` (a cycle's time by level), `row53.py` (the harness rows), `inject53.py` (prompt
51's product and VAL-016 drivers under a chosen preconditioner), `tables53.py` (every table
below, from the records), `run53.sh` (the stages). Raw output under `results/builder53/`
(untracked).
**Order:** item 5 and the plants ran at commit A, before commit B; section 1 was committed alone
before item 6's measurements. Sections 2 onward were written after the runs.
**Policy:** `docs/REVIEW_POLICY.md`, "What blocks". The numbers live in generated tables; the
sentences point at them.

## 1. Predictions

Stated by the orchestrator on 2026-10-14 in prompt 53. Committed alone, before item 6's
measurements. A captured system is one of the ten p' systems of item 6: VAL-001, the closed
cavity, the product room on 160x60 and 320x120, the Annex 20 room on 216x72 and the backward-facing
step's medium grid 656x60, each early (outer 50) and late (outer 2,000, or the last correction of
a run that stops before it).

- (a) Under the default every laminar hash and the pressure loop's results are the base's.
- (b) MG-PCG takes 10 to 40 iterations per correction on every captured system, and its count
  grows by less than a factor of two from the product's 160x60 to 320x120 (Jacobi-PCG's grows by
  about 1.9 per doubling, ADR-013 section A).
- (c) Setup is under 10% of a multigrid correction's time on every captured system.
- (d) A correction's total time falls at least fivefold on the step's medium grid and at least
  threefold on Annex 20 and the product's 320x120.
- (e) The whole runs reach the same stop as their `jacobi` records, within 5% of the outer count,
  with answers within the same tolerances prompt 51 used (Annex 20 lines within 1e-4 u0, product
  faces within 1e-3 of the velocity scale, the step's x_r within 0.02 H).
- (f) The step's medium grid runs at least three times faster per outer iteration alone.

What each outcome means (prompt 53):

- (d) and (f) hold: Alex decides on making `multigrid` the default, with VAL-020 and step 8 then
  run on it, and the laminar hashes rebaselined once in that change.
- (b) fails on the step only: the anisotropy defeats the point smoother; line smoothing or
  semi-coarsening is the next question, and the default changes for the uniform-grid cases first.
- (e) fails: a different stopped field at the same `pressure_rtol` means the two preconditioners
  leave different errors at the same residual (ADR-013 section 7.4); the report measures by how
  much before anything is decided.

Stop and report (prompt 53): a laminar hash that moves under the default; MG-PCG missing
`pressure_rtol` within the cap on a captured system, or taking more than 100 iterations on one;
a positivity error or a capped pressure correction in a whole run.

## 2. What was built (commit A)

`src/multigrid.py` builds an additive-correction multigrid hierarchy from a correction's
coefficients and applies one V-cycle to a residual; `src/pressure.py` hands it to the same
conjugate gradient loop as its preconditioner when the solver key `pressure_preconditioner` is
`multigrid`. Absent, the key is `jacobi` and nothing of the module runs.

- **The hierarchy.** Each coarser level groups 2x2 cells; a lone cell at an odd edge is its own
  group. Prolongation is piecewise constant and restriction its transpose, a sum. With that pair
  the Galerkin operator `P^T A P` is a five-point stencil again: a coarse face coefficient is the
  sum of the fine ones across that face, and a coarse diagonal the children's diagonals minus
  twice the couplings inside the group. The setup is those strided sums and the power iterations
  below: no probing and no rediscretization, so SOLID cells, obstacles, the outlet terms and a
  closed domain's constant null space reach every level through the sums. A coarse cell has an
  equation when a child has one. Coarsening stops at 64 cells or at a side of one cell, and the
  coarsest level is solved by a dense pseudo-inverse, the minimum-norm solution on a closed
  domain.
- **The cycle.** V(1,1) from zero: one weighted Jacobi sweep, the coarse correction, one sweep
  with the same weight. The same sweep before and after makes the cycle a symmetric operator, and
  positive definite on the range while the sweep contracts in the energy norm. Each level's
  weight is `(4/3) / lambda`, lambda the largest eigenvalue of `D^-1 A` estimated by three power
  iterations from the checkerboard. The estimate is a Rayleigh quotient, a lower bound, so the
  weight stays inside the contraction limit `2 / lambda_max` for any estimate above two thirds of
  the true value. Every operation is a whole-array expression on every level.
- **The loop.** `conjugate_gradient(apply, precondition, f, rtol, floor, max_iter)` takes the
  preconditioner as a callable. `jacobi_preconditioner` returns the product with the inverse
  diagonal the loop formed before, the same operation in the same place, so the default path is
  the base's to the bit (section 4). The stop, `pressure_rtol`, the rounding floor and the cap
  mean the same under either preconditioner. The hierarchy is built afresh in every correction,
  since the coefficients change every outer iteration.
- **The closed domain.** Under multigrid the preconditioned vector has its mean over the cells
  with an equation removed, beside the existing projection of the right-hand side. A V-cycle's
  coarse solve can add a constant the singular operator does not see; ADR-013's probe saw pyamg's
  CG fail on the cavity, by that report's reading for this reason (`pressure_solver_ecr003.md`,
  section 7.5). The product room and the Annex 20 room are closed domains too: their outlets are
  fixed-flow, so no face holds `p' = 0`.
- **The key.** `solver.pressure_preconditioner`, `jacobi` or `multigrid`, absent meaning
  `jacobi`, anything else refused at load, in `SOLVER_KEYS`, so every new harness row records it.

**The choices, and why.** nu and the power-iteration count were chosen on two systems outside
the measurement set, the product room on 80x30 and the step's coarse grid 410x40 at outer 50 and
300 (`dev53.py`). V(2,2) and V(3,3) took one to seven fewer CG iterations than V(1,1) and more
time per correction on every system: every sweep is a full product on every level. In a scratch
check on the same systems, not kept as a record, two power iterations came within a third of a
percent of a 200-iteration estimate on every level, so three leave a margin; the committed test of
the weights holds three to within 1% of the dense largest eigenvalue. The choice was made on an
earlier draft of the cycle whose transfers were slower; the table below is `dev53.py` rerun on the
committed code, alone on the machine.

<!-- BEGIN tables53: dev -->
| System | Preconditioner | CG its | Setup ms | Solve ms | Total ms |
| --- | --- | --- | --- | --- | --- |
| product80_early | Jacobi-PCG | 284 | 0.0 | 14.6 | 14.7 |
|  | V(1,1), 2 power its | 14 | 1.7 | 5.2 | 6.9 |
|  | V(1,1), 3 power its | 14 | 1.2 | 4.8 | 6.0 |
|  | V(1,1), 6 power its | 14 | 1.5 | 4.8 | 6.2 |
|  | V(2,2), 2 power its | 12 | 1.0 | 6.4 | 7.4 |
|  | V(2,2), 3 power its | 12 | 1.2 | 6.9 | 8.2 |
|  | V(2,2), 6 power its | 12 | 1.9 | 7.6 | 9.6 |
|  | V(3,3), 2 power its | 12 | 1.2 | 8.1 | 9.3 |
|  | V(3,3), 3 power its | 12 | 1.3 | 8.4 | 9.7 |
|  | V(3,3), 6 power its | 12 | 1.6 | 8.7 | 10.3 |
| product80_late | Jacobi-PCG | 272 | 0.0 | 14.4 | 14.5 |
|  | V(1,1), 2 power its | 13 | 1.1 | 4.5 | 5.6 |
|  | V(1,1), 3 power its | 13 | 1.1 | 4.4 | 5.5 |
|  | V(1,1), 6 power its | 13 | 1.4 | 4.4 | 5.8 |
|  | V(2,2), 2 power its | 12 | 1.0 | 6.1 | 7.1 |
|  | V(2,2), 3 power its | 12 | 1.2 | 6.1 | 7.3 |
|  | V(2,2), 6 power its | 12 | 1.6 | 6.1 | 7.7 |
|  | V(3,3), 2 power its | 12 | 1.0 | 8.0 | 9.1 |
|  | V(3,3), 3 power its | 12 | 1.0 | 8.0 | 9.1 |
|  | V(3,3), 6 power its | 12 | 1.5 | 8.1 | 9.6 |
| step410_early | Jacobi-PCG | 655 | 0.2 | 105.0 | 105.2 |
|  | V(1,1), 2 power its | 27 | 2.6 | 23.5 | 26.1 |
|  | V(1,1), 3 power its | 27 | 3.1 | 23.3 | 26.5 |
|  | V(1,1), 6 power its | 27 | 3.5 | 24.2 | 27.7 |
|  | V(2,2), 2 power its | 22 | 2.7 | 32.1 | 34.7 |
|  | V(2,2), 3 power its | 22 | 2.9 | 29.9 | 32.8 |
|  | V(2,2), 6 power its | 22 | 3.5 | 30.5 | 34.0 |
|  | V(3,3), 2 power its | 20 | 2.6 | 36.5 | 39.1 |
|  | V(3,3), 3 power its | 20 | 2.7 | 36.5 | 39.1 |
|  | V(3,3), 6 power its | 20 | 3.8 | 37.5 | 41.3 |
| step410_late | Jacobi-PCG | 1009 | 0.1 | 159.6 | 159.7 |
|  | V(1,1), 2 power its | 28 | 3.4 | 30.2 | 33.6 |
|  | V(1,1), 3 power its | 28 | 3.2 | 30.4 | 33.6 |
|  | V(1,1), 6 power its | 28 | 4.5 | 29.5 | 34.0 |
|  | V(2,2), 2 power its | 24 | 2.8 | 42.6 | 45.4 |
|  | V(2,2), 3 power its | 24 | 3.2 | 42.3 | 45.5 |
|  | V(2,2), 6 power its | 24 | 6.4 | 46.6 | 53.0 |
|  | V(3,3), 2 power its | 23 | 3.6 | 51.5 | 55.1 |
|  | V(3,3), 3 power its | 23 | 4.1 | 54.9 | 59.0 |
|  | V(3,3), 6 power its | 23 | 7.0 | 59.4 | 66.5 |
<!-- END tables53: dev -->

**A premise that did not hold.** The prompt asked for the weight from a power iteration "since a
summed operator need not be diagonally dominant". With piecewise-constant prolongation it is. A
fine diagonal is its neighbour couplings plus any outlet term; summed over a group, each coupling
inside the group appears twice, and the Galerkin diagonal subtracts both, which leaves the
couplings leaving the group plus the outlet terms: at least the sum of the magnitudes of the
coarse off-diagonals, which are non-positive. Every level is weakly diagonally dominant, so
`lambda_max(D^-1 A) <= 2` by Gershgorin. The probe of ADR-013 used bilinear prolongation, whose
Galerkin operator need not be. The estimates on every smoothed level of every captured system lie
in the hierarchy table of section 5; a fixed weight of 2/3 would give nearly the same cycle and
drop the power iterations from the setup. Built as asked; the fixed weight is a question for the
default's pull request.

**What the tests pin** (`tests/test_multigrid.py`, beside the additions to `tests/test_config.py`
and the callable in `tests/test_pressure.py`). On a 27x19 channel with an obstacle (open, uniform
and stretched) and a closed cavity with the same obstacle, odd on both sides so the lone edge
cells occur: every coarse operator equals `P^T A P` formed densely with P written out cell by
cell; the coarse equation mask; restriction as prolongation's transpose on every parity; where
coarsening stops; each weight against the dense largest eigenvalue; the cycle symmetric on random
vectors and as a dense matrix; the preconditioner positive on the range; zero at SOLID cells; the
closed-domain projection; MG-PCG and Jacobi-PCG to the same relative residual, MG within the
tolerance's worth of a dense solve, directly and through `correct`; the VAL-002 case file's
closed system at its first correction and at outer 100; and the default correction against a copy
of the base loop, bitwise, at two levels. The config tests pin the default, both choices and six
refusals.

## 3. Method

**Captured systems (item 6).** `capture53.py` runs each case from rest under its committed
settings, the default preconditioner included, from a worktree of commit 0 (`8ee82d7`,
`results/builder53/base`), and saves the p' system the corrector is handed at outer iteration 50
and at outer iteration 2,000, built by the corrector's own `coefficients` and `mass_imbalance`.
A run that stops before 2,000 saves its last correction instead, the converged state: VAL-001 at
1,558 and the product's 160x60 at 1,953. The cases are VAL-001 80x40, the closed cavity VAL-002
80x80, prompt 48's standard upwind product row on 160x60 and 320x120 (prompt 47's configuration
builder), the Annex 20 harness case 216x72 and the step's medium harness case 656x60. Each is
solved as the corrector solves it: the right-hand side projected on a closed domain, the case's
`pressure_rtol` (1e-8 for the laminar cases, 1e-4 for the turbulent ones), the floor `1e-13 F`
and the cap 5,000. On three late systems, the converged states of VAL-001 and the 160x60 product
and the cavity at outer 2,000, the floor is above `pressure_rtol ||f||` and ends the correction
under either preconditioner; the tables mark them.

**Timing (item 6).** `measure53.py` solves every system under both preconditioners in one
process, the two interleaved, five times each, and keeps the medians of the setup (building the
preconditioner), the solve and their total. One BLAS thread; nothing else ran on the machine. It
ran at commit B, after the predictions were committed.

**Whole runs (item 7).** Each run is the committed solver under its committed settings with the
key set to `multigrid` and nothing else changed: `row53.py` for the harness rows (the Annex 20 and
step cases, `run_case` with `load_preset` returning the case with the key set), `inject53.py` for
prompt 48's product rows and VAL-016 (prompt 51's drivers, with every configuration they build
given the key) and `check53.py` for VAL-001, its twin and VAL-002. They ran one after another
from a worktree of commit A (`results/builder53/A`), each alone on the machine, so seconds per
outer iteration compare, with three short exceptions: a 20-second smoke run of `inject53.py` and a
one-second test file ran during the step's 48-minute row, and a ten-second isolated timing check
ran during the 21-second product 80x30 multigrid row. The Jacobi records are prompt 51's (`results/builder51/`: the Annex 20
row ran beside one other run, the product rows four together) and, for the step, prompt 52's
clean harness row, alone. The Annex 20 and product rows were also rerun under `jacobi` alone in
the same sequence, so their times compare like for like. The multigrid rows are not appended to
`benchmarks/results.jsonl`: the default stays the diagonal, and a row under another
preconditioner there would be read as the case's latest.

## 4. Unchanged by construction (item 5), and the planted defects

<!-- BEGIN tables53: laminar -->
| Case | Base | Commit A | Base stop | A stop | Base face hash | A face hash | Equal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val001_80x40 | `8ee82d7` | `5c5eff4` | error_estimate_and_continuity at 1,559 | error_estimate_and_continuity at 1,559 | `c7d88d919e1141a5` | `c7d88d919e1141a5` | yes |
| val001_80x40_stretched | `8ee82d7` | `5c5eff4` | error_estimate_and_continuity at 1,124 | error_estimate_and_continuity at 1,124 | `9838d632f181d90f` | `9838d632f181d90f` | yes |
| val002_80x80 | `8ee82d7` | `5c5eff4` | error_estimate_and_continuity at 12,814 | error_estimate_and_continuity at 12,814 | `a8a61e28120e3f44` | `a8a61e28120e3f44` | yes |
<!-- END tables53: laminar -->

The full suite, 1,491 tests, passed at commit A and again at this report's commit, the transport and turbulence tests among them.
Each defect below was planted in a worktree of commit A made for it, the tests run, and the file
restored (`plant53.py`).

<!-- BEGIN tables53: plants -->
| Plant | Defect | Result | Failing tests | Failing (first three) |
| --- | --- | --- | --- | --- |
| `inside_coupling_once` | a coupling inside a group subtracted from the coarse diagonal once, not twice | caught | 8 | TestCoarseOperator::test_each_level_is_the_dense_galerkin_product; TestCycle::test_weights_come_from_a_lower_bound_close_to_the_largest_eigenvalue; TestMultigridSolve::test_reaches_the_level_jacobi_reaches_and_the_dense_solution |
| `inside_coupling_kept` | a coupling inside a group not subtracted from the coarse diagonal | caught | 8 | TestCoarseOperator::test_each_level_is_the_dense_galerkin_product; TestCycle::test_weights_come_from_a_lower_bound_close_to_the_largest_eigenvalue; TestMultigridSolve::test_reaches_the_level_jacobi_reaches_and_the_dense_solution |
| `east_face_from_inside` | the coarse east coefficient summed from the couplings inside the group | caught | 6 | TestCoarseOperator::test_each_level_is_the_dense_galerkin_product; TestCycle::test_the_cycle_is_symmetric |
| `no_post_smoothing` | post-smoothing omitted | caught | 11 | TestCycle::test_the_cycle_is_symmetric; TestMultigridSolve::test_reaches_the_level_jacobi_reaches_and_the_dense_solution; TestMultigridSolve::test_the_corrector_under_multigrid_meets_its_stop ... |
| `weight_beyond_contraction` | a smoother weight of 1.5, past the energy-norm contraction limit 2 / lambda_max | caught | 14 | TestCycle::test_the_preconditioner_is_positive_on_the_range; TestCycle::test_weights_come_from_a_lower_bound_close_to_the_largest_eigenvalue; TestMultigridSolve::test_reaches_the_level_jacobi_reaches_and_the_dense_solution ... |
| `no_closed_projection` | the closed domain's preconditioned vector not projected onto the range | caught | 1 | TestCycle::test_the_closed_domain_projection_removes_the_mean |
| `default_multigrid` | the default preconditioner multigrid | caught | 12 | TestDefaultPath::test_the_default_correction_is_the_base_loop_bitwise; TestMultigridSolve::test_the_corrector_under_multigrid_meets_its_stop; TestMultigridSolve::test_the_val002_cavity_system_is_solved_under_multigrid ... |
| `jacobi_divides` | the diagonal preconditioner as a division, the same values to rounding but not to the bit | caught | 6 | TestDefaultPath::test_the_default_correction_is_the_base_loop_bitwise |
<!-- END tables53: plants -->

## 5. One correction (item 6)

<!-- BEGIN tables53: per_correction -->
| System | Stop set by | Jacobi its | Jacobi ms | MG its | MG setup ms | MG solve ms | MG total ms | Setup share | Jacobi / MG time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VAL-001 80x40, outer 50 | pressure_rtol | 237 | 14.7 | 25 | 1.7 | 9.6 | 11.4 | 15.2% | 1.3 |
| VAL-001 80x40, outer 1,558 | floor | 146 | 8.5 | 9 | 1.4 | 3.3 | 4.7 | 30.3% | 1.8 |
| VAL-002 cavity 80x80 (closed), outer 50 | pressure_rtol | 414 | 34.7 | 36 | 1.5 | 20.5 | 22.3 | 6.9% | 1.6 |
| VAL-002 cavity 80x80 (closed), outer 2,000 | floor | 352 | 32.2 | 27 | 1.7 | 16.0 | 17.8 | 9.4% | 1.8 |
| Product 160x60, outer 50 | pressure_rtol | 402 | 43.7 | 16 | 2.2 | 11.0 | 13.3 | 16.8% | 3.3 |
| Product 160x60, outer 1,953 | floor | 101 | 11.5 | 8 | 2.3 | 5.8 | 8.2 | 28.2% | 1.4 |
| Product 320x120, outer 50 | pressure_rtol | 930 | 350.4 | 23 | 4.8 | 74.4 | 81.3 | 6.0% | 4.3 |
| Product 320x120, outer 2,000 | pressure_rtol | 907 | 481.4 | 23 | 8.5 | 81.6 | 88.3 | 9.6% | 5.5 |
| Annex 20 216x72, outer 50 | pressure_rtol | 419 | 92.5 | 22 | 3.8 | 30.5 | 34.3 | 11.0% | 2.7 |
| Annex 20 216x72, outer 2,000 | pressure_rtol | 500 | 112.5 | 26 | 3.7 | 36.4 | 40.3 | 9.1% | 2.8 |
| Step medium 656x60, outer 50 | pressure_rtol | 919 | 480.6 | 25 | 6.8 | 87.9 | 92.7 | 7.3% | 5.2 |
| Step medium 656x60, outer 2,000 | pressure_rtol | 1405 | 737.8 | 22 | 6.5 | 70.0 | 76.5 | 8.5% | 9.6 |
<!-- END tables53: per_correction -->

<!-- BEGIN tables53: per_iteration -->
| System | Jacobi ms per it | MG ms per it | MG / Jacobi per it | Jacobi / MG its | Jacobi residual at exit | MG residual at exit | max \|p'_MG - p'_J\| / max \|p'_J\| |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VAL-001 80x40, outer 50 | 0.062 | 0.386 | 6.2 | 9.5 | 9.32e-09 | 6.31e-09 | 2.8e-09 |
| VAL-001 80x40, outer 1,558 | 0.058 | 0.363 | 6.3 | 16.2 | 6.27e-03 | 3.90e-03 | 7.8e-04 |
| VAL-002 cavity 80x80 (closed), outer 50 | 0.084 | 0.569 | 6.8 | 11.5 | 9.61e-09 | 6.36e-09 | 1.6e-09 |
| VAL-002 cavity 80x80 (closed), outer 2,000 | 0.091 | 0.592 | 6.5 | 13.0 | 1.80e-06 | 1.47e-06 | 6.7e-07 |
| Product 160x60, outer 50 | 0.109 | 0.690 | 6.4 | 25.1 | 9.70e-05 | 8.39e-05 | 1.2e-03 |
| Product 160x60, outer 1,953 | 0.113 | 0.722 | 6.4 | 12.6 | 2.02e-02 | 1.70e-02 | 3.1e-01 |
| Product 320x120, outer 50 | 0.376 | 3.236 | 8.6 | 40.4 | 9.98e-05 | 9.79e-05 | 9.6e-04 |
| Product 320x120, outer 2,000 | 0.531 | 3.548 | 6.7 | 39.4 | 9.79e-05 | 9.24e-05 | 2.2e-04 |
| Annex 20 216x72, outer 50 | 0.221 | 1.387 | 6.3 | 19.0 | 9.90e-05 | 9.73e-05 | 2.2e-04 |
| Annex 20 216x72, outer 2,000 | 0.225 | 1.401 | 6.2 | 19.2 | 9.98e-05 | 8.54e-05 | 4.6e-04 |
| Step medium 656x60, outer 50 | 0.523 | 3.516 | 6.7 | 36.8 | 9.99e-05 | 9.58e-05 | 2.2e-02 |
| Step medium 656x60, outer 2,000 | 0.525 | 3.183 | 6.1 | 63.9 | 9.97e-05 | 9.55e-05 | 1.4e-03 |
<!-- END tables53: per_iteration -->

<!-- BEGIN tables53: hierarchy -->
| System | Levels (nx x ny), coarsest last | lambda_max estimate | Weight omega |
| --- | --- | --- | --- |
| VAL-001 80x40, outer 50 | 80x40 > 40x20 > 20x10 > 10x5 | 1.9912 to 1.9978 | 0.6674 to 0.6696 |
| VAL-001 80x40, outer 1,558 | 80x40 > 40x20 > 20x10 > 10x5 | 1.9912 to 1.9978 | 0.6674 to 0.6696 |
| VAL-002 cavity 80x80 (closed), outer 50 | 80x80 > 40x40 > 20x20 > 10x10 > 5x5 | 2.0000 to 2.0000 | 0.6667 to 0.6667 |
| VAL-002 cavity 80x80 (closed), outer 2,000 | 80x80 > 40x40 > 20x20 > 10x10 > 5x5 | 2.0000 to 2.0000 | 0.6667 to 0.6667 |
| Product 160x60, outer 50 | 160x60 > 80x30 > 40x15 > 20x8 > 10x4 | 2.0000 to 2.0000 | 0.6667 to 0.6667 |
| Product 160x60, outer 1,953 | 160x60 > 80x30 > 40x15 > 20x8 > 10x4 | 2.0000 to 2.0000 | 0.6667 to 0.6667 |
| Product 320x120, outer 50 | 320x120 > 160x60 > 80x30 > 40x15 > 20x8 > 10x4 | 2.0000 to 2.0000 | 0.6667 to 0.6667 |
| Product 320x120, outer 2,000 | 320x120 > 160x60 > 80x30 > 40x15 > 20x8 > 10x4 | 2.0000 to 2.0000 | 0.6667 to 0.6667 |
| Annex 20 216x72, outer 50 | 216x72 > 108x36 > 54x18 > 27x9 > 14x5 > 7x3 | 2.0000 to 2.0000 | 0.6667 to 0.6667 |
| Annex 20 216x72, outer 2,000 | 216x72 > 108x36 > 54x18 > 27x9 > 14x5 > 7x3 | 2.0000 to 2.0000 | 0.6667 to 0.6667 |
| Step medium 656x60, outer 50 | 656x60 > 328x30 > 164x15 > 82x8 > 41x4 > 21x2 | 1.9967 to 1.9998 | 0.6667 to 0.6678 |
| Step medium 656x60, outer 2,000 | 656x60 > 328x30 > 164x15 > 82x8 > 41x4 > 21x2 | 1.9970 to 1.9998 | 0.6667 to 0.6677 |
<!-- END tables53: hierarchy -->

MG-PCG needs many times fewer iterations than Jacobi-PCG, and each of its iterations costs
several times one of Jacobi's. The gain in a correction's time is the first factor divided by the
second, so it is largest where Jacobi needs the most iterations, the step's medium grid and the
product's 320x120, and smallest on the small laminar systems, where Jacobi needs a few hundred
cheap iterations. No captured system came near a stop condition: no cap, and no more than 36
iterations.

At the same relative residual the two leave p' errors of different shape (ADR-013 section 7.4:
the diagonal leaves the smooth part). The last column of the second table is the difference of
the two solutions over the largest p'.

**Twice.** `measure53.py` ran a second time at the end of the session, after the whole runs, to
size how far this machine's timings move between sessions. The iteration counts are the same;
the times move by up to about a quarter, and the ratios by less.

<!-- BEGIN tables53: passes -->
| System | Jacobi ms, pass 1 / 2 | MG ms, pass 1 / 2 | Setup share, pass 1 / 2 | Jacobi / MG time, pass 1 / 2 |
| --- | --- | --- | --- | --- |
| VAL-001 80x40, outer 50 | 14.7 / 13.4 | 11.4 / 10.1 | 15.2% / 13.1% | 1.3 / 1.3 |
| VAL-001 80x40, outer 1,558 | 8.5 / 10.1 | 4.7 / 5.7 | 30.3% / 27.5% | 1.8 / 1.8 |
| VAL-002 cavity 80x80 (closed), outer 50 | 34.7 / 34.9 | 22.3 / 22.7 | 6.9% / 7.6% | 1.6 / 1.5 |
| VAL-002 cavity 80x80 (closed), outer 2,000 | 32.2 / 29.7 | 17.8 / 17.4 | 9.4% / 9.3% | 1.8 / 1.7 |
| Product 160x60, outer 50 | 43.7 / 43.8 | 13.3 / 14.2 | 16.8% / 17.5% | 3.3 / 3.1 |
| Product 160x60, outer 1,953 | 11.5 / 11.2 | 8.2 / 7.8 | 28.2% / 28.3% | 1.4 / 1.4 |
| Product 320x120, outer 50 | 350.4 / 389.3 | 81.3 / 72.1 | 6.0% / 8.6% | 4.3 / 5.4 |
| Product 320x120, outer 2,000 | 481.4 / 397.6 | 88.3 / 70.0 | 9.6% / 8.2% | 5.5 / 5.7 |
| Annex 20 216x72, outer 50 | 92.5 / 75.4 | 34.3 / 30.3 | 11.0% / 11.9% | 2.7 / 2.5 |
| Annex 20 216x72, outer 2,000 | 112.5 / 98.6 | 40.3 / 35.4 | 9.1% / 11.6% | 2.8 / 2.8 |
| Step medium 656x60, outer 50 | 480.6 / 374.8 | 92.7 / 66.3 | 7.3% / 9.3% | 5.2 / 5.7 |
| Step medium 656x60, outer 2,000 | 737.8 / 548.8 | 76.5 / 65.6 | 8.5% / 8.2% | 9.6 / 8.4 |
<!-- END tables53: passes -->

**Where a cycle's time goes** (`anatomy53.py`, the two late turbulent systems, alone). The finest
level forms two products per cycle and costs a little under four products' time with its
transfers and smoothing. The coarser levels hold a third of the finest level's cells between
them, yet cost about as much again on the Annex 20 room and a third as much on the step: their
arrays are small enough that each NumPy call's fixed cost, not the arithmetic, sets their time.
That overhead, more than the method's arithmetic, is why one MG-PCG iteration costs as many
Jacobi-PCG iterations as it does; a compiled or GPU cycle (Phase 6) would remove most of it, and so might stopping the
coarsening earlier with a larger dense coarsest solve. Neither was tried.

<!-- BEGIN tables53: anatomy -->
| System | Part | ms | Products in one cycle |
| --- | --- | --- | --- |
| Annex 20 216x72, outer 2,000 | one product, finest | 0.085 |  |
|  | one Jacobi-PCG iteration | 0.152 |  |
|  | one V-cycle | 0.672 |  |
|  | of which level 216x72 | 0.318 | 2 |
|  | of which level 108x36 | 0.151 | 2 |
|  | of which level 54x18 | 0.072 | 2 |
|  | of which level 27x9 | 0.091 | 2 |
|  | of which level 14x5 | 0.063 | 2 |
|  | of which level 7x3 | 0.003 | 0 |
| Step medium 656x60, outer 2,000 | one product, finest | 0.365 |  |
|  | one Jacobi-PCG iteration | 0.433 |  |
|  | one V-cycle | 2.384 |  |
|  | of which level 656x60 | 1.332 | 2 |
|  | of which level 328x30 | 0.184 | 2 |
|  | of which level 164x15 | 0.125 | 2 |
|  | of which level 82x8 | 0.114 | 2 |
|  | of which level 41x4 | 0.078 | 2 |
|  | of which level 21x2 | 0.003 | 0 |
<!-- END tables53: anatomy -->

## 6. Whole runs (item 7)

Each run under multigrid against its Jacobi record. No run had a capped pressure correction, a
positivity error or a k-epsilon fallback.

<!-- BEGIN tables53: whole -->
| Run | Preconditioner | Record | Stop | Outer | Wall s | s per outer | Pressure s per outer | Pressure share | CG its per correction | Capped corrections |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Step medium 656x60 | jacobi | prompt 52 row of record, alone | max_simple_iter | 20,000 | 13,475 | 0.6737 | 0.5826 | 86% | 1347.1 | 0 |
| Step medium 656x60 | multigrid | this branch, alone | max_simple_iter | 20,000 | 2,869 | 0.1434 | 0.0540 | 38% | 22.5 | 0 |
| Annex 20 216x72 | jacobi | prompt 51 record, beside one other run | error_estimate_and_continuity | 11,441 | 2,034 | 0.1778 | 0.1170 | 66% | 505.4 | 0 |
| Annex 20 216x72 | jacobi | this branch, alone | error_estimate_and_continuity | 11,441 | 2,794 | 0.2442 | 0.1970 | 81% | 505.4 | 0 |
| Annex 20 216x72 | multigrid | this branch, alone | error_estimate_and_continuity | 11,443 | 890 | 0.0778 | 0.0315 | 40% | 25.0 | 0 |
| Product 80x30 | jacobi | prompt 51 record, four runs together | error_estimate_and_continuity | 1,021 | 36 | 0.0349 |  |  |  |  |
| Product 80x30 | jacobi | this branch, alone | error_estimate_and_continuity | 1,021 | 29 | 0.0288 | 0.0172 | 60% |  | 0 |
| Product 80x30 | multigrid | this branch, alone | error_estimate_and_continuity | 1,021 | 21 | 0.0210 | 0.0085 | 40% |  | 0 |
| Product 160x60 | jacobi | prompt 51 record, four runs together | error_estimate_and_continuity | 1,954 | 201 | 0.1028 |  |  |  |  |
| Product 160x60 | jacobi | this branch, alone | error_estimate_and_continuity | 1,954 | 161 | 0.0826 | 0.0533 | 65% |  | 0 |
| Product 160x60 | multigrid | this branch, alone | error_estimate_and_continuity | 1,953 | 88 | 0.0450 | 0.0167 | 37% |  | 0 |
| val001_80x40 | jacobi | commit A, beside five other runs | error_estimate_and_continuity | 1,559 | 36 | 0.0232 | 0.0195 | 84% | 198.4 | 0 |
| val001_80x40 | multigrid | this branch, alone | error_estimate_and_continuity | 1,559 | 20 | 0.0127 | 0.0100 | 79% | 17.6 | 0 |
| val001_80x40_stretched | jacobi | commit A, beside five other runs | error_estimate_and_continuity | 1,124 | 25 | 0.0219 | 0.0183 | 83% | 186.5 | 0 |
| val001_80x40_stretched | multigrid | this branch, alone | error_estimate_and_continuity | 1,124 | 16 | 0.0145 | 0.0117 | 81% | 21.2 | 0 |
| val002_80x80 | jacobi | commit A, beside five other runs | error_estimate_and_continuity | 12,814 | 479 | 0.0374 | 0.0328 | 88% | 263.0 | 0 |
| val002_80x80 | multigrid | this branch, alone | error_estimate_and_continuity | 12,814 | 255 | 0.0199 | 0.0161 | 81% | 19.7 | 0 |
<!-- END tables53: whole -->

### 6.1 The answers

Every run stops where its Jacobi record stops, by the same rule, within two outer iterations,
and every answer moves by far less than prompt 51's tolerances: by orders of magnitude on the
Annex 20 lines and the product's faces, by about a fifth of the tolerance on the step's x_r
(0.004 against 0.02 H; review 53 S3). The two preconditioners leave
different p' errors in each correction (section 5); over a whole run they leave no trace in the
answers beyond the differences tabled here.

The Annex 20 room, its four lines (the Jacobi rerun alone is bitwise the record, the same face
hash, so the two difference columns agree):

<!-- BEGIN tables53: annex -->
| Line | Scored | RMS / u0, Jacobi record | RMS / u0, multigrid | Allowance | Multigrid | Largest difference from the Jacobi record, u / u0 | From the Jacobi rerun |
| --- | --- | --- | --- | --- | --- | --- | --- |
| x3 | yes | 0.059288 | 0.059288 | 0.070250 | pass | 2.79e-09 | 2.79e-09 |
| x6 | no | 0.036822 | 0.036822 | 0.093296 | - | 1.75e-09 | 1.75e-09 |
| y2916 | yes | 0.079742 | 0.079742 | 0.135583 | pass | 1.37e-09 | 1.37e-09 |
| y0084 | no | 0.057916 | 0.057916 | 0.052172 | - | 2.90e-09 | 2.90e-09 |
<!-- END tables53: annex -->

The product rows of prompt 48, the faces over the velocity scale:

<!-- BEGIN tables53: products -->
| Grid | Jacobi record | Multigrid | Outer change | Velocity scale, m/s | Largest face difference over the scale, MG vs record | Jacobi rerun bitwise the record | MG vs rerun |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 80x30 | error_estimate_and_continuity at 1,021 | error_estimate_and_continuity at 1,021 | +0.00% | 0.45 | 1.19e-11 | yes | 1.19e-11 |
| 160x60 | error_estimate_and_continuity at 1,954 | error_estimate_and_continuity at 1,953 | -0.05% | 0.45 | 1.57e-10 | yes | 1.57e-10 |
<!-- END tables53: products -->

VAL-016's split criteria:

<!-- BEGIN tables53: val016 -->
| Preconditioner | Commit | Stop | Outer | (a) max du / U_w (< 1e-4) | (a) max dk / u_tau^2 (< 1e-4) | (a) u_tau / 1D | (b) standard core k ratio | (b) RNG core k ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jacobi, prompt 51 record | `4cf12f9` | error_estimate_and_continuity | 258 | 7.029e-08 | 3.741e-06 | 0.9999997 | 0.99946 to 1.00146 | 0.99941 to 1.00108 |
| multigrid | `5c5eff4` | error_estimate_and_continuity | 258 | 7.029e-08 | 3.741e-06 | 0.9999997 | 0.99946 to 1.00146 | 0.99941 to 1.00108 |
<!-- END tables53: val016 -->

The step's medium grid, every figure its harness row reports (both rows at the 20,000 cap, where
the medium grid cycles at the reattachment cell; prompt 52's report, section 7, item 3):

<!-- BEGIN tables53: step -->
| Figure | Jacobi row of record | Multigrid | Difference |
| --- | --- | --- | --- |
| x_r_over_h | 4.8586 | 4.85412 | -0.00448 |
| u_ref_over_measured | 0.999615 | 0.999615 | -1.19e-08 |
| delta_99_over_h | 1.45711 | 1.45711 | -3.23e-07 |
| re_theta | 5229.4 | 5229.39 | -0.00437 |
| rms_cf | 0.000426956 | 0.00042718 | +2.25e-07 |
| rms_cp | 0.0311057 | 0.0311 | -5.67e-06 |
| rms_profile_x-4.0 | 0.018139 | 0.018139 | +3.15e-08 |
| rms_profile_x1.0 | 0.0478425 | 0.0478424 | -1.33e-07 |
| rms_profile_x4.0 | 0.0547241 | 0.0547277 | +3.61e-06 |
| rms_profile_x6.0 | 0.0235447 | 0.0235454 | +6.96e-07 |
| rms_profile_x10.0 | 0.0275178 | 0.0275175 | -2.29e-07 |
<!-- END tables53: step -->

VAL-001, its stretched twin and VAL-002 under multigrid, against their criteria (their hashes
differ from the default's, as expected):

<!-- BEGIN tables53: laminar_mg -->
| Case | Jacobi (commit A) | Multigrid | Jacobi value | Multigrid value | Criterion | Multigrid |
| --- | --- | --- | --- | --- | --- | --- |
| val001_80x40 | error_estimate_and_continuity at 1,559 | error_estimate_and_continuity at 1,559 | 4.1077e-04 | 4.1077e-04 | < 0.01 | pass |
| val001_80x40_stretched | error_estimate_and_continuity at 1,124 | error_estimate_and_continuity at 1,124 | 3.0237e-03 | 3.0237e-03 | < 0.01 | pass |
| val002_80x80 | error_estimate_and_continuity at 12,814 | error_estimate_and_continuity at 12,814 | u 1.0572e-03, v 7.3556e-04 | u 1.0572e-03, v 7.3556e-04 | < 0.02 | pass |
<!-- END tables53: laminar_mg -->

### 6.2 The time

<!-- BEGIN tables53: gains -->
| Run | Jacobi record | s per outer, Jacobi / MG | Pressure stage, Jacobi / MG | Jacobi ms per CG iteration in the run | MG ms per CG iteration in the run | Item 6, late system, Jacobi / MG time, pass 1 / 2 |
| --- | --- | --- | --- | --- | --- | --- |
| Step medium 656x60 | prompt 52 row of record, alone | 4.70 | 10.79 | 0.433 | 2.402 | 9.6 / 8.4 |
| Annex 20 216x72 | this branch, alone | 3.14 | 6.26 | 0.390 | 1.261 | 2.8 / 2.8 |
| Product 80x30 | this branch, alone | 1.37 | 2.03 |  |  |  |
| Product 160x60 | this branch, alone | 1.83 | 3.19 |  |  | 1.4 / 1.4 |
| val001_80x40 | commit A, beside five other runs | 1.83 | 1.94 | 0.098 | 0.572 | 1.8 / 1.8 |
| val001_80x40_stretched | commit A, beside five other runs | 1.52 | 1.56 | 0.098 | 0.555 |  |
| val002_80x80 | commit A, beside five other runs | 1.88 | 2.04 | 0.125 | 0.817 | 1.8 / 1.7 |
<!-- END tables53: gains -->

Two cautions on reading it.

- **Sessions differ.** The Annex 20 Jacobi row rerun alone in this session is bitwise prompt 51's
  record, yet slower per outer iteration than that record, which ran beside another run two days
  earlier: its momentum and turbulence stages ran faster, its pressure stage slower, at the same
  iteration count. An isolated Jacobi-PCG iteration on the late Annex 20 system took 0.225 ms in
  the first pass of `measure53.py` and 0.152 ms in `anatomy53.py` hours later (section 5's
  per-iteration and anatomy tables). So seconds per outer
  iteration from different sessions carry tens of percent. The step's comparison is across
  sessions (its Jacobi row ran early on 2026-10-14, its multigrid row this afternoon); the Annex 20
  and product comparisons are within one sequence.
- **In a run is not isolated.** On the Annex 20 room a Jacobi-PCG iteration inside the whole run
  costs well over its isolated time (the table's in-run column against section 5), while a
  multigrid iteration costs about its isolated time, so the run gains more than item 6 predicts:
  the pressure stage falls more than the per-correction ratio there. On the step the in-run and
isolated costs per iteration are close (test 53 S3).
  Why the Jacobi loop runs slower inside this run was not measured.

The pressure correction stops being most of the run: under multigrid it is 37% to 40% of the
turbulent runs (the whole table's pressure share), against 60% to 86% under Jacobi. On the
laminar cases, where Jacobi needs a few hundred iterations, it stays the larger stage either way.

## 7. Predictions against measurement

<!-- BEGIN tables53: predictions -->
|  | Prediction | Measured | Verdict |
| --- | --- | --- | --- |
| (a) | Under the default every laminar hash and the pressure loop's results are the base's | 3 of 3 laminar cases stop at the base's count with its face hash; the default correction equals the base loop to the bit on six test systems (`TestDefaultPath`); tests/test_pressure.py passes with the callable in place of the array | holds |
| (b) | MG-PCG 10 to 40 iterations per correction on every captured system; under 2x growth from 160x60 to 320x120 | 8 to 36 on all ten (16 to 36 where pressure_rtol sets the stop; the counts below 10 are floor-stopped converged states); growth 1.44x early, 2.88x late (160x60's late system is floor-stopped, 320x120's is not) | fails as worded, below 10 on two floor-stopped systems; growth holds on the like-for-like early pair |
| (c) | Setup under 10% of a multigrid correction's time on every captured system | 6.0% to 30.3% over both passes; at or above 10% in a pass on 6 of 12: Annex 20 216x72, outer 2,000, Annex 20 216x72, outer 50, Product 160x60, outer 1,953, Product 160x60, outer 50, VAL-001 80x40, outer 1,558, VAL-001 80x40, outer 50 | fails |
| (d) | Correction time falls at least 5x on the step's medium grid, 3x on Annex 20 and the product's 320x120 | The smaller of the two passes: step 5.2x early, 8.4x late; Annex 20 2.5x early, 2.8x late; 320x120 4.3x early, 5.5x late | holds on the step and 320x120, fails on Annex 20 |
| (e) | Whole runs reach the Jacobi records' stop within 5% of the outer count, answers within prompt 51's tolerances | Annex 20 outer +0.02%, lines within 2.9e-09 u0; product 80x30 outer +0.00%, faces within 1.2e-11 of the scale; product 160x60 outer -0.05%, faces within 1.6e-10 of the scale; step outer +0.00% (both at the cap), x_r -0.004 H; VAL-016 outer 258 against 258 | holds |
| (f) | The step's medium grid at least 3x faster per outer iteration alone | 0.674 s against 0.143 s, 4.70x | holds |
<!-- END tables53: predictions -->

## 8. What this implies, and the question of the default

Questions for Alex, with what the measurements say about each; not decisions.

1. **The default.** By the prompt's reading of its outcomes, (d) and (f) holding means the default
   is Alex's to change. (f) holds with room for the session drift of section 6.2. (d) holds on the
   step and the product's 320x120 and fails on the Annex 20 room, where Jacobi-PCG needs only
   about 500 iterations and a correction falls under three times, though its whole run falls three
   times and its pressure stage more (section 6.2). Every whole run measured reaches its Jacobi
   stop and answers (e), so on these runs the change moves no answer by more than prompt 51's
   tolerances; it would move every laminar face hash, which section 4 shows does not happen
   under the default, and the rebaseline would be taken once in that change. Three ways to put it:
   - (1) `multigrid` the default for every case, the laminar hashes rebaselined once. Section
     6.2's table gives each run's gain per outer iteration; the step's medium grid gains most.
     The fine grid was not run under multigrid; its Jacobi corrections need more iterations than
     the medium grid's (prompt 52's report, section 7, item 5), so it would likely gain more,
     unmeasured.
   - (2) `multigrid` in the turbulent case files and the product configuration only, the
     validation cases on the diagonal. No laminar hash moves; two preconditioners stay in use by
     default, two paths to keep, test and document, which ADR-013 decision 4's option (3)
     weighed against.
   - (3) Leave the default; name `multigrid` per run. VAL-020 and step 8 then choose.
   The prompt's (b)-fails-on-the-step outcome did not occur: the stretched step needs no more
   iterations than the uniform rooms, so line smoothing or semi-coarsening is not needed for it.
2. **The setup.** (c) fails on six of the twelve systems, wherever the solve is short next to
   building the hierarchy: the small laminar systems, the floor-stopped converged states, the
   160x60 product and, just over the line, the Annex 20 room. The power iterations are part of
   the setup and buy nothing measurable here: every smoothed level's estimate is within a percent
   of 2, and the coarse operators' diagonal dominance (section 2) bounds the true value by 2. A
   fixed weight of 2/3 would drop them. Whether to make that change, and whether to
   reuse a hierarchy for a few outer iterations (ADR-013's report found a stale one costly at 99
   outer iterations; a lag of a few was not measured), are for the default's pull request.
3. **The cycle's cost.** One MG-PCG iteration costs six to nine Jacobi-PCG iterations (section
   5's per-iteration table), more than its arithmetic accounts for, and much of the difference is
   NumPy's per-call cost on the small coarse levels (section 5's anatomy). That is Phase 6's to remove, where the cycle becomes kernels: the
   restriction, the prolongation and the smoother are each one thread per cell, and the coarsest
   solve a small dense product. PROJECT_PLAN's Phase 6 kernel set would grow by those if the
   default changes.
