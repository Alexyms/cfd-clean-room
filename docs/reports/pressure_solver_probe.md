# Pressure Solver Probe: Jacobi Convergence in the Collocated SIMPLE Solver

*Erratum 2026-09-22: section 5 calls the wall leak "the same wall-treatment defect ECR-001 identified for the v-velocity error". That v error was an artifact: the Ghia v reference in use until 2026-09-22 was not Ghia's Table II and failed mass conservation along the centerline; see the ECR-001 erratum, section 12, and reference `ghia_1982_re100_r2`. The tables below are unchanged. The wall-leak measurements here use no reference data and stand.*

**Date:** 2026-09-19
**Branch:** `scratch/pressure-solver-probe` (instrumentation only, not for merge)
**Baseline:** `main` at `ce097db`
**Runner:** `scripts/pressure_solver_probe.py` on this branch. Raw JSON records are in
`results/probe/` (gitignored).

## 1. Verdict

Hypothesis H1 is contradicted.

The Jacobi pressure solve does hit its sweep cap on essentially every SIMPLE iteration
(P1 confirmed) and exits with an update norm three to five orders of magnitude above
`pressure_tol` (P2 confirmed). But the number of outer SIMPLE iterations to convergence
does not depend on the cap at all. At 40x40, caps of 50, 200, 500 and 1000 all converge
in 1571 or 1572 outer iterations (P3 contradicted). The outer loop is not compensating
for an unsolved `p_prime`. The wall time is set by the outer iteration count, which is
governed by the SIMPLE coupling and under-relaxation, multiplied by a per-iteration cost
that is almost entirely wasted Jacobi sweeps.

The reason the cap is always hit is different from the one H1 assumed. For the closed
cavity the pressure-correction system is singular (all boundaries are Neumann) and its
right-hand side is not compatible: the mass imbalance summed over the domain is
nonzero, because the collocated ghost-cell wall treatment lets mass leak through walls
and lid. A singular system with an incompatible right-hand side has no solution. Jacobi
then drifts by a constant each sweep, the update norm settles at that drift value, and
no number of sweeps ever reaches the tolerance. With the cap removed and a bound of
100,000 sweeps, every captured cavity system after iteration 0 hit the bound with the
update norm flat to four significant figures. The velocity correction only sees
gradients of `p_prime`, so the drift is harmless, and the number of sweeps beyond the
first few hundred changes nothing.

The prompt's premise about the configuration also needs correcting: the two validation
tests do not use the YAML defaults. VAL-001 runs with a cap of 2000 and tolerance 1e-8,
and VAL-002 runs with a cap of 500, tolerance 1e-8 and `alpha_velocity` 0.5. All runs
below use the test settings unless stated, since those produced the observed runtime.

Recommendation: REQ-S08 should still be revisited in the rebuild, but for reasons the
data supports rather than the reason H1 proposed. See section 7.

## 2. Hypothesis and predictions, as stated before measuring

**H1.** The pressure correction never converges. Every SIMPLE iteration receives a
substantially unsolved `p_prime`, and the outer loop compensates with a very large
iteration count. The 25-minute VAL-002 runtime is a consequence of an inadequate inner
linear solve, not of insufficient arithmetic throughput.

| ID | Prediction | Verdict |
|----|------------|---------|
| P1 | At 40x40 and 80x80 the Jacobi loop hits the sweep cap on essentially every SIMPLE iteration. | Confirmed (100.0 percent of iterations at both sizes) |
| P2 | The Jacobi update norm at the moment the cap is reached is orders of magnitude above `pressure_tol`. | Confirmed (median 1.0e4 x tol at 40x40, 2.5e3 x tol at 80x80) |
| P3 | Raising `max_pressure_iter` to 5000 at 40x40 reduces the outer SIMPLE iteration count by a factor of three or more. | Contradicted (1571 outer iterations at every cap from 50 to 5000) |
| P4 | The pressure solve accounts for more than 90 percent of wall time. | Confirmed (96.9 to 99.4 percent), but as the prompt noted this alone proves nothing |
| P5 | Sweeps required to reach `pressure_tol` grow roughly as N squared across 20, 40 and 80 cells per side. | Confirmed for the one solvable case per grid (iteration 0): 1159, 4700, 17970 sweeps, ratios 4.06 and 3.82 |

P1, P2 and P5 are confirmed and P3 is contradicted. P1 and P2 are true for a reason
unrelated to H1 (section 5.3), and P3 is the discriminating test. H1 is dead.

## 3. Method

### 3.1 Instrumentation

On the scratch branch, `NavierStokesSolver` records per SIMPLE iteration:

- Jacobi sweeps performed and whether the cap was reached.
- `final_diff`: the max-norm update between the last two sweeps. This is what the
  existing code compares against `pressure_tol`. It is not a residual.
- `final_resid`: max-norm residual of the linear system `a_P p' = sum(a_nb p'_nb) - b`
  over interior fluid cells, evaluated after the final ghost-cell update.
- `initial_resid`: max-norm of the right-hand side `b` (the mass imbalance), which is
  the residual at `p' = 0`.
- `sum b` and `sum |b|` over fluid cells. For a Neumann problem `sum b` must be zero
  for a solution to exist.
- Wall-clock timers around momentum (steps 1 to 4), face flux (5 to 7), pressure solve
  (8), velocity correction and convergence check (9 to 11), and separately the
  per-sweep `apply_pressure_bc` call inside the Jacobi loop.

For chosen SIMPLE iterations the solver also copies `(mass_imbalance, d)` so the same
linear system can be re-solved afterwards with the cap raised to 100,000.

No solver logic was changed. `tests/test_solver_ns.py` passes on the instrumented
branch (17 passed).

### 3.2 Environment

AMD Ryzen AI 9 HX 370, 24 logical processors, Windows 11, Python 3.13.3, NumPy 2.4.4.
Runs were executed as three concurrent single-threaded processes. Timer fractions are
unaffected; absolute wall times may be slightly inflated relative to a quiet machine.
The 80x80 cavity took 11.7 minutes here versus the 25 minutes observed on CI.

### 3.3 Runs

| Run | Case | Grid | Cap | p_tol | alpha_u | Notes |
|-----|------|------|-----|-------|---------|-------|
| poiseuille_80x40_cap2000 | VAL-001 | 80x40 | 2000 | 1e-8 | 0.7 | test settings |
| cavity_20_cap500 | VAL-002 | 20x20 | 500 | 1e-8 | 0.5 | test settings |
| cavity_40_cap500 | VAL-002 | 40x40 | 500 | 1e-8 | 0.5 | test settings |
| cavity_80_cap500 | VAL-002 | 80x80 | 500 | 1e-8 | 0.5 | test settings, run once |
| cavity_40_cap50 | VAL-002 | 40x40 | 50 | 1e-8 | 0.5 | cap sweep |
| cavity_40_cap200 | VAL-002 | 40x40 | 200 | 1e-8 | 0.5 | cap sweep |
| cavity_40_cap1000 | VAL-002 | 40x40 | 1000 | 1e-8 | 0.5 | cap sweep |
| cavity_40_cap5000 | VAL-002 | 40x40 | 5000 | 1e-8 | 0.5 | cap sweep |
| cavity_40_yamldefaults | VAL-002 | 40x40 | 200 | 1e-6 | 0.7 | YAML solver defaults, max_simple_iter raised to 10000 so it can converge |

Uncapped re-solves were done for the first four runs at captured iterations 0, 50, 100
and 1000 (1000 does not exist for the 20x20 run, which converged in 380 iterations).

## 4. Raw measurements

### Table A: run summary

"Cap hit" counts SIMPLE iterations whose Jacobi loop exhausted `max_pressure_iter`.
"Pressure %" is the share of wall time in the pressure solve; "BC loop %" is the share
of total wall time spent inside the per-sweep `apply_pressure_bc` Python loop, which is
part of the pressure share.

| Run | Outer iters | Converged | Wall (s) | Total sweeps | Sweeps per outer min/med/max | Cap hit | Pressure % | BC loop % |
|---|---|---|---|---|---|---|---|---|
| poiseuille_80x40_cap2000 | 551 | yes | 178.3 | 963450 | 517/2000/2000 | 396/551 (71.9%) | 99.4 | 47.1 |
| cavity_20_cap500 | 380 | yes | 16.3 | 190000 | 500/500/500 | 380/380 (100%) | 97.6 | 32.0 |
| cavity_40_cap500 | 1571 | yes | 108.5 | 785500 | 500/500/500 | 1571/1571 (100%) | 97.7 | 41.7 |
| cavity_80_cap500 | 5288 | yes | 703.9 | 2644000 | 500/500/500 | 5288/5288 (100%) | 96.9 | 43.9 |
| cavity_40_cap50 | 1572 | yes | 13.1 | 78600 | 50/50/50 | 1572/1572 (100%) | 82.6 | 33.3 |
| cavity_40_cap200 | 1571 | yes | 45.5 | 314200 | 200/200/200 | 1571/1571 (100%) | 94.6 | 40.6 |
| cavity_40_cap1000 | 1571 | yes | 218.4 | 1571000 | 1000/1000/1000 | 1571/1571 (100%) | 98.8 | 42.7 |
| cavity_40_cap5000 | 1571 | yes | 970.0 | 7854700 | 4700/5000/5000 | 1570/1571 (99.9%) | 99.8 | 42.0 |
| cavity_40_yamldefaults | 1178 | yes | 33.1 | 235600 | 200/200/200 | 1178/1178 (100%) | 94.6 | 40.5 |

### Table B: Jacobi state at exit, per SIMPLE iteration (min / median / max over the run)

"Residual reduction" is `final_resid / initial_resid`. A value of 1 means the sweeps
made no progress on the residual at all.

| Run | final_diff / p_tol | final residual | initial residual | residual reduction |
|---|---|---|---|---|
| poiseuille_80x40_cap2000 | 1 / 3.09 / 9.49e4 | 1.56e-10 / 4.83e-10 / 1.49e-05 | 4.92e-10 / 5.81e-09 / 4.42e-04 | 0.0103 / 0.0891 / 0.674 |
| cavity_20_cap500 | 158 / 3.59e4 / 3.6e4 | 3.92e-07 / 9.01e-05 / 9.02e-05 | 8.99e-05 / 9.02e-05 / 2.19e-03 | 0.000282 / 1 / 1 |
| cavity_40_cap500 | 2.51e3 / 1.02e4 / 1.02e4 | 1.57e-06 / 6.46e-06 / 6.46e-06 | 6.43e-06 / 6.46e-06 / 1.10e-03 | 0.00225 / 1 / 1 |
| cavity_80_cap500 | 2.47e3 / 2.47e3 / 2.85e4 | 3.88e-07 / 3.88e-07 / 4.44e-06 | 3.88e-07 / 3.88e-07 / 5.48e-04 | 0.00254 / 1 / 1 |
| cavity_40_cap50 | 1.02e4 / 1.03e4 / 1.43e5 | 6.42e-06 / 6.46e-06 / 8.79e-05 | 6.43e-06 / 6.46e-06 / 1.12e-03 | 0.0247 / 1 / 1 |
| cavity_40_cap200 | 6.62e3 / 1.02e4 / 3.23e4 | 4.14e-06 / 6.46e-06 / 2.00e-05 | 6.43e-06 / 6.46e-06 / 1.10e-03 | 0.00631 / 1 / 1 |
| cavity_40_cap1000 | 759 / 1.02e4 / 1.02e4 | 4.74e-07 / 6.46e-06 / 6.46e-06 | 6.43e-06 / 6.46e-06 / 1.10e-03 | 0.000682 / 1 / 1 |
| cavity_40_cap5000 | 0.999 / 1.02e4 / 1.02e4 | 6.24e-10 / 6.46e-06 / 6.46e-06 | 6.43e-06 / 6.46e-06 / 1.09e-03 | 8.98e-07 / 1 / 1 |
| cavity_40_yamldefaults | 85.2 / 102 / 422 | 5.32e-06 / 6.46e-06 / 2.61e-05 | 6.43e-06 / 6.46e-06 / 1.40e-03 | 0.00631 / 1 / 1 |

### Table C: stage timers (seconds)

| Run | momentum | flux | pressure | of which BC loop | correct | total |
|---|---|---|---|---|---|---|
| poiseuille_80x40_cap2000 | 0.5 | 0.3 | 177.1 | 84.0 | 0.4 | 178.3 |
| cavity_20_cap500 | 0.2 | 0.1 | 15.9 | 5.2 | 0.1 | 16.3 |
| cavity_40_cap500 | 1.1 | 0.6 | 106.0 | 45.3 | 0.8 | 108.5 |
| cavity_80_cap500 | 9.6 | 6.8 | 681.9 | 308.7 | 5.5 | 703.9 |
| cavity_40_cap50 | 1.0 | 0.5 | 10.8 | 4.3 | 0.7 | 13.1 |
| cavity_40_cap200 | 1.1 | 0.6 | 43.0 | 18.4 | 0.8 | 45.5 |
| cavity_40_cap1000 | 1.1 | 0.6 | 215.9 | 93.2 | 0.8 | 218.4 |
| cavity_40_cap5000 | 1.0 | 0.6 | 967.6 | 407.1 | 0.7 | 970.0 |
| cavity_40_yamldefaults | 0.8 | 0.4 | 31.3 | 13.4 | 0.6 | 33.1 |

Per-sweep cost, from the uncapped re-solves: about 90 us at 20x20, 132 us at 40x40,
185 us at 80x40 and 248 us at 80x80. The Python loop over boundary entries in
`apply_pressure_bc` runs once per sweep and accounts for 32 to 47 percent of total
wall time.

### Table D: uncapped re-solves of captured pressure systems (bound 100,000 sweeps)

Iteration 0 starts from a zero velocity field, so its right-hand side sums to zero
exactly. Later iterations do not.

| Run | Captured iter | Sweeps | Bound hit | final_diff | final residual | initial residual | sum b | sum abs b | Last three update norms |
|---|---|---|---|---|---|---|---|---|---|
| poiseuille_80x40 | 0 | 100000 | YES | 4.27e-08 | 6.70e-10 | 4.42e-04 | -1.96e-02 | 1.96e-02 | 4.272e-08, 4.271e-08, 4.271e-08 |
| poiseuille_80x40 | 50 | 74608 | no | 1.00e-08 | 1.56e-10 | 4.79e-07 | 4.74e-04 | 4.74e-04 | 1.000e-08, 1.000e-08, 9.999e-09 |
| poiseuille_80x40 | 100 | 34049 | no | 1.00e-08 | 1.56e-10 | 6.54e-08 | 8.24e-06 | 8.88e-06 | 1.000e-08, 1.000e-08, 1.000e-08 |
| cavity_20 | 0 | 1159 | no | 9.96e-09 | 2.47e-09 | 1.39e-03 | 0 | 2.78e-03 | 1.011e-08, 1.004e-08, 9.960e-09 |
| cavity_20 | 50 | 100000 | YES | 3.42e-04 | 8.57e-05 | 1.10e-04 | 2.75e-02 | 2.75e-02 | 3.415e-04, 3.415e-04, 3.415e-04 |
| cavity_20 | 100 | 100000 | YES | 3.56e-04 | 8.93e-05 | 9.20e-05 | 2.87e-02 | 2.87e-02 | 3.559e-04, 3.559e-04, 3.559e-04 |
| cavity_40 | 0 | 4700 | no | 9.99e-09 | 6.24e-10 | 6.94e-04 | 0 | 1.39e-03 | 1.003e-08, 1.001e-08, 9.993e-09 |
| cavity_40 | 50 | 100000 | YES | 9.16e-05 | 5.79e-06 | 1.96e-05 | 8.26e-03 | 8.44e-03 | 9.157e-05, 9.157e-05, 9.157e-05 |
| cavity_40 | 100 | 100000 | YES | 9.87e-05 | 6.23e-06 | 8.68e-06 | 8.90e-03 | 8.90e-03 | 9.866e-05, 9.866e-05, 9.866e-05 |
| cavity_40 | 1000 | 100000 | YES | 1.02e-04 | 6.46e-06 | 6.46e-06 | 9.23e-03 | 9.23e-03 | 1.022e-04, 1.022e-04, 1.022e-04 |
| cavity_80 | 0 | 17970 | no | 1.00e-08 | 1.56e-10 | 3.47e-04 | 0 | 6.94e-04 | 1.001e-08, 1.000e-08, 9.999e-09 |
| cavity_80 | 50 | 100000 | YES | 2.25e-05 | 3.53e-07 | 6.47e-06 | 2.14e-03 | 2.28e-03 | 2.250e-05, 2.250e-05, 2.250e-05 |
| cavity_80 | 100 | 100000 | YES | 2.40e-05 | 3.76e-07 | 1.43e-06 | 2.28e-03 | 2.30e-03 | 2.395e-05, 2.395e-05, 2.395e-05 |
| cavity_80 | 1000 | 100000 | YES | 2.47e-05 | 3.89e-07 | 3.99e-07 | 2.35e-03 | 2.35e-03 | 2.475e-05, 2.475e-05, 2.475e-05 |

### Table E: net mass source `sum b` over the run (cavity, cap 500)

| Grid | it 0 | it 1 | it 10 | it 50 | it 100 | it 500 | it 1000 | final |
|---|---|---|---|---|---|---|---|---|
| 20x20 | 0 | 2.55e-04 | 1.29e-02 | 2.75e-02 | 2.87e-02 | n/a | n/a | 2.90e-02 (it 379) |
| 40x40 | 0 | 6.40e-05 | 3.57e-03 | 8.26e-03 | 8.90e-03 | 9.22e-03 | 9.23e-03 | 9.23e-03 (it 1570) |
| 80x80 | 0 | 1.60e-05 | 9.25e-04 | 2.14e-03 | 2.28e-03 | 2.35e-03 | 2.35e-03 | 2.35e-03 (it 5287) |

The 40x40 cap 50, 200 and 1000 runs reproduce the 40x40 row to three significant
figures at every listed iteration.

### Table F: outer residual history (cavity 80x80, cap 500)

| it 0 | it 10 | it 50 | it 100 | it 200 | it 500 | it 1000 | it 2000 | it 3000 | it 4000 | it 5287 |
|---|---|---|---|---|---|---|---|---|---|---|
| 8.33e-02 | 1.62e-02 | 4.17e-03 | 2.10e-03 | 1.05e-03 | 3.59e-04 | 1.26e-04 | 3.33e-05 | 1.11e-05 | 3.86e-06 | 9.99e-07 |

### Table G: converged fields versus cap (cavity 40x40)

Separate runs with the converged `u`, `v`, `p` saved. Differences are max-norm against
the cap 500 run. Reference magnitudes: `max |u| = 0.922`, `max |v| = 0.402`,
`max |p| = 0.572`.

| Cap | Outer iters | Wall (s) | max diff u | max diff v | max diff p |
|---|---|---|---|---|---|
| 5 | 1575 | 3.0 | 1.40e-05 | 1.17e-05 | 1.03e-05 |
| 50 | 1572 | 12.6 | 4.15e-07 | 5.03e-07 | 1.71e-07 |
| 500 | 1571 | 103.5 | reference | reference | reference |

Five Jacobi sweeps per SIMPLE iteration give the same converged solution as 500 to
within 1.5e-5 absolute, at 3 percent of the wall time.

## 5. Analysis

### 5.1 P3, the discriminating test

At 40x40 the outer SIMPLE iteration count is 1571 or 1572 for every cap tried. The
per-iteration outer residual histories are identical to three significant figures
across caps. Wall time scales linearly with the cap because every sweep up to the cap
is executed: 13.1 s at cap 50, 45.5 s at 200, 108.5 s at 500, 218.4 s at 1000, 970.0 s
at 5000. The only cavity iteration that ever converged before a cap was iteration 0
of the cap 5000 run, at 4700 sweeps, the same count as the uncapped re-solve. Beyond
the first few sweeps of any given SIMPLE iteration the extra sweeps do no work that
the outer loop can see. Table G makes this concrete: a cap of 5 reproduces the cap 500
converged fields to 1.5e-5 absolute in 3 percent of the wall time, and a cap of 50
reproduces them to 5e-7.

Changing `alpha_velocity` from 0.5 to 0.7 (the YAML default) cut the outer count from
1571 to 1178 with the same cap. That is the kind of sensitivity H1 predicted for the
cap and did not find. The outer count also grows with the grid (380, 1571, 5288 for
20, 40, 80 cells per side), which is the usual behaviour of SIMPLE with a fixed
under-relaxation factor and an iterate-change convergence criterion, not a symptom of
an unsolved inner system.

### 5.2 P1 and P2 are true, for the wrong reason

The cap is hit on 100 percent of cavity iterations and the update norm sits at 1e3 to
1e4 times `pressure_tol`. H1 reads this as "not enough sweeps". Table D shows it is
"no number of sweeps". With the cap raised to 100,000, every captured cavity system
after iteration 0 exits with the update norm flat to four significant figures for the
last several thousand sweeps (the full traces are in the JSON records; the last three
values are in the table). The iteration is not converging slowly. It has reached a
fixed drift and will stay there.

The residual tells the same story from the other side. Table B shows a median residual
reduction of exactly 1 for every cavity run: after 500 sweeps the max-norm residual is
the same as at `p' = 0`. The pressure solve is not partially solving the system, it is
not solving it at all, and yet the outer loop converges.

### 5.3 Why: a singular system with an incompatible right-hand side

All cavity boundaries are Neumann for `p'` (`apply_pressure_bc` copies the interior
value into the ghost cell for walls and velocity inlets). The pressure-correction
matrix therefore has the constant vector in its null space, and a solution exists only
if `sum b = 0` over the fluid cells. Table E shows `sum b` is zero at iteration 0 (zero
velocity field), grows within a few iterations, and settles at a nonzero plateau that
the outer loop never removes: 2.90e-2 at 20x20, 9.23e-3 at 40x40, 2.35e-3 at 80x80.

Summing the discrete divergence over the domain telescopes to the net mass flux through
the boundary faces, so a nonzero `sum b` means mass is crossing the walls. The face
flux at a wall-adjacent face is `0.5 * (v_interior + v_ghost)` with
`v_ghost = v_interior / 3` for a no-slip wall, which is `(2/3) v_interior`, not zero.
The collocated ghost-cell wall treatment does not enforce zero normal mass flux at the
wall. This is the same wall-treatment defect ECR-001 identified for the v-velocity
error, seen here through the continuity equation.

Two further observations from Table D and Table B:

- At convergence `b` is uniform. At 40x40, `max |b| = 6.46e-6` and there are 38 x 38 =
  1444 interior fluid cells; `1444 * 6.46e-6 = 9.33e-3`, which matches `sum b = 9.23e-3`
  to within 1 percent. The same holds at 20x20 and 80x80. A uniform `b` lies entirely
  in the incompatible direction, so the non-constant part of `p'` is zero and the
  velocity correction is zero. The outer loop has converged to a state where the
  pressure correction step is a no-op and the domain carries a uniform residual mass
  source.
- Jacobi on this system adds a constant `mean(b / a_P)` to `p'` every sweep. The
  measured plateau update norms (1.02e-4 at 40x40, 2.47e-5 at 80x80) are that drift.
  The velocity correction takes gradients of `p'`, so the drift is invisible to it, and
  so is the cap.

### 5.4 The Poiseuille case is different and still slow

VAL-001 has a pressure outlet, so the system is nonsingular and Jacobi does converge.
The cap of 2000 was hit on 72 percent of iterations (the early ones); by the end each
SIMPLE iteration converged in 517 to 812 sweeps. From a cold start the captured
systems needed 34,049 sweeps at iteration 100, 74,608 at iteration 50, and more than
100,000 at iteration 0 (bound hit with the update norm at 4.3e-8, four times the
tolerance, residual reduced by six orders of magnitude). The prompt's asymptotic
estimate, roughly `2 N^2 / pi^2` sweeps per e-fold, gives about 1300 sweeps per e-fold
for N = 80 and 18 e-folds for eight orders of magnitude, or about 23,000 sweeps, so
the measured counts are in the expected range for a Dirichlet-Neumann problem of this
length. This is the genuine Jacobi cost H1 was worried about, and it is real. Whether
the 2000-sweep cap affects the VAL-001 outer count was not tested; the prompt limited
the cap sweep to the cavity. It is an open question.

### 5.5 P5

For the one solvable system per grid (iteration 0, `sum b = 0`), sweeps to reach the
tolerance were 1159, 4700 and 17970 at 20, 40 and 80 cells per side. Successive ratios
are 4.06 and 3.82 against 4.0 for pure N-squared scaling. The prompt's estimate of
"on the order of 18,000 sweeps" for six orders of magnitude at 80x80 is matched almost
exactly (the criterion here is an absolute update norm of 1e-8 from a starting update
of order 1e-4, which is about the same decade count).

### 5.6 Where the wall time goes

P4 holds: 97 to 99 percent of wall time is the pressure solve. Inside it, the Python
loop in `apply_pressure_bc`, executed once per Jacobi sweep, accounts for 32 to 47
percent of total wall time on its own, rising with grid size because the number of
boundary entries grows with the perimeter. The 80x80 cavity spent 309 of its 704
seconds in that loop. The remaining pressure time is the vectorized sweep itself plus
the per-sweep max-norm reduction with boolean indexing.

The 25-minute runtime therefore decomposes as: 5288 outer iterations (set by SIMPLE
coupling and under-relaxation, independent of the inner solve) times 500 sweeps (all
of them wasted after the first few tens, since the system cannot converge) times about
250 us per sweep (nearly half of it a Python boundary loop). Any one of the three
factors could be attacked. H1 pointed at the wrong one.

## 6. What this probe did not establish

- It did not test whether the VAL-001 outer count is sensitive to the cap. The cap
  sweep was specified for the cavity only.
- It did not identify what does control the cavity's outer count beyond the
  under-relaxation sensitivity in Table A. The outer convergence criterion is the
  max-norm velocity change between iterations, scaled by a reference velocity. An
  iterate-change criterion with heavy under-relaxation stops when the iteration slows
  down, not when the equations are satisfied. That is consistent with the data but was
  not isolated as a variable.
- Timings were taken with three processes sharing one machine. Fractions are reliable;
  absolute seconds are indicative.
- The 80x80 cavity was run once, as instructed.

## 7. Recommendation on REQ-S08

REQ-S08 ("the pressure correction equation shall be solved using Jacobi iteration")
should be revisited during the ECR-001 rebuild. Not because H1 was right, but because
of three things the data does show.

1. **The N-squared cost is real and the rebuild will expose it.** Today the cavity's
   inner solve is a no-op because the system is inconsistent, so the cap is harmless.
   ECR-001 replaces the collocated ghost-cell walls with a staggered arrangement that
   enforces discrete continuity exactly. That makes the right-hand side compatible,
   which means the inner system becomes solvable, which means the outer loop will start
   to depend on how well it is solved. At that point the measured 18,000 sweeps at 80x80
   for one accurate solve (Table D, iteration 0) becomes the binding cost, and the
   pathology H1 described would appear for the first time. The production grid is
   200x75. Jacobi is the wrong algorithm to inherit into a discretization that finally
   makes the pressure solve matter. REQ-S08 should be rewritten as a requirement on
   solver behaviour (residual reduction per outer iteration, or a tolerance relative to
   the initial residual) and leave the algorithm open. Candidates that vectorize in
   NumPy and map to one-thread-per-cell CUDA at least as well as Jacobi include
   red-black Gauss-Seidel with over-relaxation, preconditioned conjugate gradient, and
   geometric multigrid. Any of them removes the N-squared factor or reduces its
   constant by an order of magnitude or more.

2. **The convergence test must be residual based and the system must be made
   consistent.** The current exit test compares the update norm against an absolute
   tolerance. For a singular system it can never be met, and for a nonsingular one it
   is not a residual. The rebuild should test a residual norm relative to the initial
   residual, and for closed domains should either pin `p'` at the reference cell inside
   the inner solve or project the right-hand side to zero mean, with the net mass
   imbalance reported as a diagnostic rather than silently absorbed.

3. **The per-sweep boundary loop must not survive.** Nearly half the wall time is a
   Python loop over boundary entries executed every sweep. Whatever solver the rebuild
   uses, ghost-cell updates need to be vectorized index operations or folded into the
   matrix coefficients. This is a throughput fix and is independent of the algorithm
   choice, but it is the single largest cost in the current code.

Separately from REQ-S08, the outer loop deserves its own probe. The outer count is
insensitive to the inner solve and sensitive to `alpha_velocity`, and it grows with the
grid. That is where the 25 minutes actually come from today.

No solver code, configuration default, or requirement was changed by this probe.
