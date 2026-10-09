# ECR-002 Step 5: The Convergence Measurement

**Date:** 2026-10-09
**Tree:** branch `docs/ecr002-step5-convergence` from main at 4391252. Nothing under `src/`,
`validation/`, `configs/` or `tests/` changes: every run goes through the committed
`StaggeredSolver` and `solve_steady(eddy_viscosity=...)`, with probe code only for the measuring
devices and the one counterfactual of measurement 4.
**Instruments:** `docs/reports/probe44/` (committed): `conv44.py` (the rooms, the fields, the runner
and the cavity), `tables44.py` (the comparisons and the tables) and the launcher `run44.sh`. Raw
output under `results/builder44/` (untracked).
**Order:** sections 1 to 4 were committed before any run (the first commit of the branch). Sections
5 onward were written after the runs. Each run log's first line carries its start time.

## 1. The question

Step 6 will couple k-epsilon to the flow. Before it is built we need to know whether the room's
outer iteration converges at the viscosities k-epsilon will hand it, on the grids we will run.
Three things have changed since step 0 asked this on 2026-10-04: the outlets are fixed-flow (step
3), the pressure is solved by CG (ECR-003), and momentum takes a viscosity field with an exact wall
stencil at the obstacles (step 4). Step 0's answer ("one sweep repels; ten converge") was measured
under the old outlets and the weighted Jacobi correction on 40x15 only. Step 4 then showed that the
40x15 answer is dominated by discretisation error: changing the advection at a few obstacle
corners moved the converged velocity by up to 0.29 m/s, half of it far from the obstacles.

Four questions, on 40x15, 80x30 and 200x75:

1. Does the room converge across k-epsilon's range of effective viscosity, uniform and
   non-uniform, with one momentum sweep, or does it need ten, or neither? (ECR-002 section 8
   step 5.)
2. Does the converged velocity settle as the grid refines? If 80x30 and 200x75 disagree as much as
   40x15 and 200x75, no grid we can afford resolves the room.
3. Does step 4's corner rule (upwind at half-open obstacle faces) still slow the outer loop on the
   finer grids, where corners are a smaller share of the room?
4. Can `pressure_rtol` relax from 1e-8?

Plus two rows the ECR asks for: the laminar room at real air (ADR-013 decision 6's finding that
the room converging on 40x15 converged on neither finer grid), and the lid-driven cavity at Re
1,000 (a known steady answer at a cell Reynolds number of tens).

## 2. Method (fixed before the runs)

### 2.1 The room and the settings

The product room is `configs/clean_room_default.yaml` regridded through `product_raw` of
`docs/reports/probe41/outlet41.py`: the ladder's solver keys, `alpha_velocity` 0.5,
`max_pressure_iter` 5000, `pressure_rtol` 1e-8 unless a row states otherwise. The outlets are the
committed fixed-flow outlets (the hood at 0.5 m/s, the four returns sharing the remainder). The
sweep count is the committed `solver.momentum_sweeps` key, 1 or 10.

Stopping: `stopping_rule: error_estimate` with `stopping_for`'s tolerances for each grid
(`iteration_error_tol` 1e-6 and ADR-011 G's per-cell bound `mass_imbalance_tol = 1e-4 rho V_min /
t_end`, as the 80x30 drift case used; on 40x15 that is 8.0e-8, on 80x30 2.0e-8 and on 200x75
3.2e-9 kg/s per metre of depth). A converged run stops by `error_estimate_and_continuity`,
criterion 10's stop. The run also records the first outer iteration at which the committed
`velocity_step` rule would have stopped (the residual below `convergence_tol` 1e-6 on an iteration
whose correction did not reach the cap), so the 40x15 rows can be read against step 4's ladder,
which stopped at 728 (Re 895) and 2,115 (Re 8,950) under that rule.

Caps: 5,000 outer iterations on 40x15 and 80x30, 10,000 on 200x75. Divergence: a cell-centred
speed above 100 m/s, or a non-finite field. From rest in every run.

Each run records: the commit, grid, field, sweep count, `pressure_rtol`, stop reason and outer
count, the residual history (the solver's velocity-step residual, every iteration), the estimated
iteration error over the velocity scale (the rule's `estimate_history`), the largest cell-centred
speed and its cell every iteration, CG iterations per correction (every one; the mean and the
largest are reported), the number of corrections that reached the cap, the wall time, and the face
hash of the final faces (section 2.3 of `docs/reports/ecr003_step2_baseline.md`, SHA-256 over u's
bytes then v's). The cell-centred fields, the pressure and the faces are kept in an `.npz`.

**Classification.** Each run is classified at its end by these rules in this order, step 0's
section 2.6 with the prompt's four classes:

- **diverged**: the run stopped because the largest speed passed 100 m/s, or a field went
  non-finite;
- **converged**: the solver's own stop, `error_estimate_and_continuity`;
- **growing**: the run reached its cap with the largest speed at the end above 5 m/s (step 0's
  "grown"; the room's largest converged speed is about 3.3 m/s on 40x15);
- **bounded and not converged**: the run reached its cap with the largest speed under 5 m/s. Step
  0's sub-classes are reported beside it: *falling* (over the last 500 iterations the residual ends
  within 10% of the window's least value and below half the window's first), *stalled* (the
  residual's largest and least over the window within a factor of two), or *neither*.

**Machine and threads.** AMD Ryzen AI 9 HX 370 (12 cores, 24 logical processors), 64 GB, Windows
11, Python 3.13, NumPy 2.4 with OpenBLAS 0.3.31. The CG solve runs under the code's limit of one
BLAS thread (`PRESSURE_BLAS_THREADS` in `src/pressure.py`, the cleanup pull request), and every
run's process also sets `OPENBLAS_NUM_THREADS=1` before NumPy loads, so no BLAS call in the run is
threaded. Runs go in parallel processes; section 5.1 states how many at once. Wall times taken
beside other runs are not like for like with the baseline report's single-process figures.

### 2.2 The fields

The effective kinematic viscosity is air's (1.81e-5 / 1.2 = 1.508e-5 m^2/s) plus
`eddy_viscosity`. Uniform fields set `eddy_viscosity` so that the sum equals the stated value.

| Name | Field | Effective viscosity (m^2/s) | `eddy_viscosity` |
|---|---|---|---|
| U1 | uniform | 1.5e-3 (top of the range; step 4's Re 895 rung) | 1.4849e-3 |
| U2 | uniform | 1.5e-4 (step 4's Re 8,950 rung) | 1.3492e-4 |
| U3 | uniform | 6.5e-5 (bottom of the range) | 4.9917e-5 |
| Z2, Z3, Z4 | step 0's zero-equation field, scaled to core medians 1.5e-3, 5e-4 and 1.5e-4 | as step 0 | `s nu_t0` |
| L | none | air (laminar) | None |

U1 and U2 are not quite step 4's rungs: the ladder scaled the molecular viscosity by 100 and 10
(1.508e-3 and 1.508e-4 m^2/s); here the molecular viscosity stays air's and the field makes up the
difference to 1.5e-3 and 1.5e-4 exactly, so the two differ by 0.5% in the viscosity and in where
it enters (a uniform `eddy_viscosity` goes through the field path of `MomentumPredictor.predict`,
which is bitwise the scalar path's arithmetic for a uniform field; step 4's `frozen` comparison).
The 40x15 rows therefore read against the ladder's counts to within that difference, not to the
iteration.

**The Z fields.** Built on each grid from that grid's own base field: the room at a thousand times
air's viscosity (Re 90) with ten sweeps, `error_estimate` with the grid's tolerances, to the stop
(the 80x30 drift case of `fixed42.py`, on each grid). From the base field's cell-centred velocity,
`base34.py`'s construction (`docs/reports/ecr002_step0_frozen_viscosity.md`, appendix A):
`nu_t0 = 0.03874 V L`, V the cell-centred speed and L the distance from the cell centre to the
nearest domain edge or SOLID cell (the staircase distance, which the step 0 rungs took, section 3
there), SOLID cells zero. The core is the non-SOLID cells whose centre lies above the equipment
tops, y > 2.0 m. Step 0's scaling: `s = target / median(nu_t0 over the core)`, and the run's
`eddy_viscosity` is `s nu_t0`, so the core median of the eddy viscosity equals the stated value and
the effective viscosity's core median is that plus air's. If a grid's base field does not converge
within its cap, that grid's Z rows are not run and section 5 says so.

### 2.3 Measurements

1. **The matrix.** Every field on every grid at one sweep. Ten sweeps for every field and grid
   where one sweep does not converge, for U1 on every grid whether or not it does (the cost
   comparison), and for L on every grid (ECR-003 criterion 3's setting).
2. **Same answer, different path.** Where one and ten sweeps both converge, the largest cell-centred
   velocity difference between the two converged fields (`max |du|` and `max |dv|` over non-SOLID
   cells), against the stopping tolerance expressed in m/s (`iteration_error_tol` times the
   velocity scale, 1e-6 x 0.45 = 4.5e-7 m/s).
3. **Grid convergence.** For each field converged on all three grids: u and v interpolated
   bilinearly from the cell-centred field of each grid to common points, SOLID cells at zero
   velocity (the wall value). The points: the configuration's four `sensors` (near_door (1.0,
   1.5), above_gap_1 (2.7, 2.5), above_gap_2 (4.8, 2.5), hood_entry (6.2, 1.2)); a vertical line
   at x = 2.7 m, the gap between the server rack (x 1.5 to 2.3) and the litho tool (3.2 to 4.5)
   through which the supply jet descends to return 2, y from 0.1 to 2.9 m every 0.02 m; and a
   horizontal line at y = 1.2 m, the hood entry's height, x from 0.1 to 7.9 m every 0.02 m. Points
   inside a configured obstacle rectangle are excluded, and the lines stop 0.1 m (the coarse
   grid's half cell) short of the domain edges so that no point lies outside the coarse grid's
   centre lattice. Reported: the largest and the RMS difference of 40x15 against 200x75 and of
   80x30 against 200x75, per component, over all points and over the points more than 0.2 m
   (the coarse grid's cell) from any obstacle rectangle, since the SOLID staircase differs between
   grids, and where the largest sits.
4. **The corner rule.** U1 and U2 on 80x30 and 200x75, at the sweep count that converged in
   measurement 1, with the corner QUICK zeroing removed through `tail43.py`'s
   `corner_free_class` (the committed `_deferred_correction` with the two lines that zero the
   QUICK correction at obstacle faces taken out; the diffusive half of the stencil and the boundary
   form's far nodes stay). Outer counts beside the committed rule's, and the converged-field
   difference as in measurement 2.
5. **The pressure tolerance.** U1 and U2 on 80x30 and 200x75 at `pressure_rtol` 1e-4 and 1e-2,
   same sweep count. Stop, count, CG iterations, and the face difference (`max |du|`, `max |dv|`
   over the face arrays) from the 1e-8 run at its stop.
6. **The cavity at Re 1,000.** `configs/validation_cavity.yaml` with the viscosity 1e-3 (rho 1, lid
   1 m/s, side 1 m), 40x40 and 80x80, one sweep, VAL-002's stopping keys as the file states them
   (`error_estimate`, `iteration_error_tol` 1e-6, `mass_imbalance_tol` 1e-10), cap 40,000.
   Reported: convergence, the count, and the centreline extremes (u on the vertical centreline, v
   on the horizontal) with their positions, from `cavity_true_centerline_profiles` of
   `validation/metrics.py` (the profiles interpolated to the true centrelines, walls appended),
   each extreme as the discrete sample and as the vertex of the parabola through that sample and
   its two neighbours. For orientation only, the orchestrator recalls Botella and Peyret (1998):
   u_min -0.38857 at y 0.1717, v_max 0.37694 at x 0.1578, v_min -0.52708 at x 0.9092. These are
   from memory, not from the paper, and no claim here rests on them.

## 3. Predictions (orchestrator, 2026-10-09; committed before any run)

- (a) U1 converges at one sweep on all three grids. Its outer count on 200x75 is between two and
  ten times its 40x15 count.
- (b) U2 converges at one sweep on 40x15 and 80x30. On 200x75 it needs ten sweeps.
- (c) U3 does not converge at one sweep on 200x75. Ten sweeps on 200x75: no prediction.
- (d) Each Z field converges wherever the uniform field at its median does.
- (e) L converges on 40x15 at ten sweeps and on neither 80x30 nor 200x75: ADR-013 decision 6's
  finding stands under fixed-flow outlets, because a steady laminar room at real air is not a
  stable answer of the equations, whatever the outlets.
- (f) Where one and ten sweeps both converge, their fields agree to within ten times the stopping
  tolerance.
- (g) For U1, the largest difference of 40x15 against 200x75 exceeds 0.1 m/s, and that of 80x30
  against 200x75 is under half of it.
- (h) The corner rule raises U1's count on 200x75 by under 20% (on 40x15 it doubled it).
- (i) `pressure_rtol` 1e-2 stops at the same count as 1e-8, within 1%, for U1 and U2 on both
  finer grids.
- (j) The cavity converges at one sweep on both grids; on 80x80 its extremes lie within 5% of the
  values above.

### 3.1 Builder's predictions (2026-10-09, before any run)

Where I differ from the orchestrator, or add a number:

- (a) Agree. The 200x75 count three to six times the 40x15 count: the drift case's count barely
  moved from 40x15 to 80x30 (ten sweeps), but one-sweep counts grew with refinement in the ECR-003
  report.
- (b) U2 at one sweep on 80x30: bounded and not converged, not converged. The cell Peclet number
  halves from 40x15 to 80x30 but the corner rule's QUICK half, which carried the whole rise at Re
  8,950 on 40x15, acts at twice as many corner faces.
- (c) U3 with ten sweeps on 200x75: converges. Step 0's ten-sweep result held at the bottom of the
  range on 40x15 at nearly the same count as the top.
- (e) Agree on 40x15. On 80x30 and 200x75 with ten sweeps: bounded and not converged, not growing.
- (f) Agree, with the largest difference under 1e-5 m/s.
- (g) Agree. The largest difference sits at an obstacle corner or beside a return, not at a sensor.
- (h) Agree: under 20% on 200x75, and under 50% on 80x30.
- (i) Holds for U1 on both grids; fails for U2 on 200x75, where the looser correction changes the
  path enough to move the count by more than 1%. ADR-013 B's finding that 1e-2 never met the
  per-cell condition was taken where the outlets held a standing imbalance; the fixed-flow outlets
  removed it (D0 at 1e-2 stopped at the same 177 as 1e-8).
- (j) Both converge; on 80x80 u_min and v_max within 5% of the recalled values and v_min within
  10%, since the right-wall jet is where the Re 100 cavity's error was largest.

### 3.2 What each outcome means

- Every field in the range converges on 200x75 at one sweep: step 6 needs no convergence aid.
- Some converge only at ten sweeps: step 6 runs with the sweep count measured, and its cost is the
  wall time recorded.
- Some converge at neither: step 6 needs a stronger aid (step 0 section 6.6's ranking: continuation
  in viscosity, pseudo-transient continuation), and that goes to Alex before step 6 opens.
- (g) fails with 80x30 as far from 200x75 as 40x15 is: 200x75 is not shown to resolve the room
  either, and VAL-018's grid needs a decision.
- (i) holds: ADR-013 decision 3's tight default can be revisited for this room.
- (e) fails, so L converges on a finer grid: ADR-013 decision 6's finding was the old outlets'.

## 4. Stop conditions (from the prompt)

The work stops and reports, before running the rest, if: U1 on 40x15 at one sweep does not converge
(step 4's ladder converged there, so the setup would differ from the one measured); a pressure
solve fails in a way no row predicts (CG at its cap on every correction, or a refusal); or the whole
matrix would exceed about eight hours of wall time on this machine, in which case the estimate and a
reduced matrix are reported first.

## 5. Results (written after the runs)

### 5.1 Order, machine and cost

**Order.** Commit 1bdb2bf ("docs: record the step 5 convergence predictions before any run") is
dated 2026-10-09 08:23:11 -0700. The first runs, four twenty-iteration timing probes, started at
08:26:13; the matrix's first rows at 08:26:40. The scripts were committed at 08:30:41 (23e2940)
with `conv44.py` as it ran: its last edit, a lint fix, preceded the timing probes, and the runs
read the committed text. `tables44.py` was edited after the runs started (it reads records and
runs nothing) and is committed as used.

**Cost and the thread setting.** Each record carries threadpoolctl's view of the loaded BLAS:
`openblas 0.3.31.188.0`, one thread, in every run, with `OPENBLAS_NUM_THREADS=1` in the process
environment and the pressure solve under `PRESSURE_BLAS_THREADS`. The timing probes, five
processes at once, gave 0.18 s per outer iteration on 200x75 at one sweep and 0.19 at ten (the
baseline report's single-process figure is 0.18 s at one sweep), 0.027 s on 80x30 and 0.012 s on
40x15. With the matrix's rows running eight to fourteen at once the 200x75 rows took 0.22 to
0.25 s per outer iteration, so the wall times below are for a loaded machine. The whole matrix,
with the supplementary rows, ran in 58 minutes of wall time between 08:26 and 09:24,
well under the eight-hour bound, so no reduced matrix was needed. The longest single run is the
cavity on 80x80, 901 s for 16,668 outer iterations.

### 5.2 The base fields and the Z fields

The room at a thousand times air's viscosity with ten sweeps converges by the solver's own stop
on every grid. The 80x30 stop at 176 is the drift case's count at step 4's commit B2 (prompt 43's
pull request).

| Grid | Stop | Outer | velocity_step at | Fluid, core cells | nu_t0 core: 5th, 25th, median, 75th, 95th (m^2/s) | Largest | Core median / nu_air | s for Z2, Z3, Z4 | Largest speed, cell |
|---|---|---|---|---|---|---|---|---|---|
| 40x15 | error_estimate_and_continuity | 100 | 61 | 420, 200 | 0.00042, 0.00176, 0.00537, 0.0092, 0.0157 | 0.0262 | 355.9 | 0.2795, 0.09315, 0.02795 | 1.5, (6.1, 0.9) |
| 80x30 | error_estimate_and_continuity | 176 | 103 | 1752, 800 | 0.000152, 0.0025, 0.00508, 0.00851, 0.0151 | 0.0274 | 336.8 | 0.2953, 0.09842, 0.02953 | 1.35, (6.05, 0.55) |
| 200x75 | error_estimate_and_continuity | 745 | 331 | 10860, 5000 | 0.000113, 0.00213, 0.00525, 0.00862, 0.0151 | 0.0291 | 348.2 | 0.2856, 0.09519, 0.02856 | 1.34, (6.02, 0.3) |

The core median of nu_t0 is 5.1e-3 to 5.4e-3 m^2/s on the three grids, 337 to 356 times air's,
against step 0's 8.12e-3 on 40x15 under the T3 outlets after 1,000 unconverged iterations: the
fixed-flow outlets give a slower room (largest speed 1.34 to 1.5 m/s against 3.66 then), so the
field is smaller, and its shape is the same construction. The scales s that bring the core median
to 1.5e-3, 5e-4 and 1.5e-4 m^2/s are within 6% across the grids, so the Z fields are the same
field to that accuracy, resolved on each grid. Every grid's Z rows were run.

**The outlets by grid.** The fixed-flow returns share the remainder of the discrete inflow at one
face velocity, and both depend on how the configured segments fall on the grid: the inflow is
3.24 m^2/s on 40x15 (the supply covers 7.2 m of ceiling) and 3.15 and 3.17 on 80x30 and 200x75
(7.0 and 7.04 m); the returns cover 2.6 m on 40x15 and 2.2 and 2.24 m on the finer grids, so the
return face velocity is 1.054 m/s on 40x15 against 1.227 and 1.209 m/s. The hood holds 0.5 m/s on
every grid. The 40x15 room therefore differs from the finer rooms in its boundary data by 15% at
the returns, which measurement 3 should be read with.

### 5.3 Measurement 1: the matrix

Every row, classified by section 2.1's rules. Rows with `_cap15k` are the two supplementary
reruns past the prompt's cap (section 5.3.1); `_s50` rows are the fifty-sweep diagnostics
(section 5.3.3). The estimate column is the rule's estimate over the velocity scale at the end
(`inf` where the step was not falling). The face hash is the first sixteen hex digits.

| Run | Class | Stop | Outer | velocity_step at | Residual: least (at), end | Estimate at end | Largest speed at end (m/s), cell | Peak speed | CG per correction: mean, largest | Cap hits | Wall (s), per outer | Face hash |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U1_40x15_s1 | converged | error_estimate_and_continuity | 1,535 | 706 | 2.63e-10 (1,535), 2.63e-10 | 9.68e-07 | 1.57, (6.1, 0.9) | 1.67 | 155, 165 | 0 | 25, 0.016 | c6fa1286c072aee6 |
| U1_40x15_s10 | converged | error_estimate_and_continuity | 333 | 178 | 1.08e-09 (333), 1.08e-09 | 9.29e-07 | 1.57, (6.1, 0.9) | 1.57 | 159, 164 | 0 | 7, 0.020 | f49ac776a40f0a77 |
| U1_80x30_s1 | bounded (neither) | max_simple_iter | 5,000 | - | 8.04e-04 (3,010), 1.75e-03 | 50.4 | 1.4, (0.85, 0.55) | 2.22 | 337, 348 | 0 | 186, 0.037 | 9573067e591a0004 |
| U1_80x30_s10 | converged | error_estimate_and_continuity | 745 | 350 | 2.92e-10 (745), 2.92e-10 | 9.89e-07 | 1.4, (2.75, 1.25) | 1.49 | 310, 348 | 0 | 29, 0.039 | 1026f56dbff915d9 |
| U1_200x75_s1 | diverged | diverged | 824 | - | 5.44e-04 (58), 3.88e-01 | inf | 105, (1.26, 0.98) | 105 | 802, 855 | 0 | 185, 0.224 | 75a633a50837cb82 |
| U1_200x75_s10 | converged | error_estimate_and_continuity | 2,329 | 1,155 | 5.94e-11 (2,329), 5.94e-11 | 8.85e-07 | 1.45, (0.5, 0.02) | 1.63 | 735, 862 | 0 | 524, 0.225 | 03f52267f647d0d9 |
| U2_40x15_s1 | bounded (falling) | max_simple_iter | 5,000 | 2,132 | 9.08e-11 (4,995), 9.46e-11 | 1.05e-06 | 1.59, (6.1, 0.9) | 1.76 | 152, 170 | 0 | 79, 0.016 | 19c9f3f0046d5ac5 |
| U2_40x15_s1_cap15k | converged | error_estimate_and_continuity | 5,012 | 2,132 | 8.75e-11 (5,012), 8.75e-11 | 9.84e-07 | 1.59, (6.1, 0.9) | 1.76 | 152, 170 | 0 | 83, 0.016 | bf2ac133577f072d |
| U2_40x15_s10 | converged | error_estimate_and_continuity | 334 | 193 | 1.12e-09 (334), 1.12e-09 | 8.73e-07 | 1.59, (6.1, 0.9) | 1.84 | 162, 167 | 0 | 6, 0.018 | 85a9481722c4c0ce |
| U2_80x30_s1 | diverged | diverged | 767 | - | 2.42e-03 (20), 7.72e-01 | inf | 100, (1.05, 0.85) | 100 | 344, 370 | 0 | 30, 0.039 | 89f76afdc9e9a07e |
| U2_80x30_s10 | bounded (neither) | max_simple_iter | 5,000 | - | 7.09e-04 (140), 2.22e-03 | inf | 1.46, (2.75, 0.35) | 1.72 | 357, 370 | 0 | 216, 0.043 | f972cd575f6c8f2e |
| U2_80x30_s50 | bounded (neither) | max_simple_iter | 5,000 | - | 7.11e-04 (139), 2.08e-03 | inf | 1.45, (2.75, 0.35) | 1.72 | 357, 370 | 0 | 271, 0.054 | cde489a8e5d5f0b6 |
| U2_200x75_s1 | diverged | diverged | 440 | - | 1.19e-03 (26), 2.79e-01 | inf | 100, (7.26, 1.5) | 100 | 848, 882 | 0 | 108, 0.245 | 8326788db5899c88 |
| U2_200x75_s10 | bounded (neither) | max_simple_iter | 10,000 | - | 7.34e-04 (296), 1.67e-03 | inf | 1.82, (0.86, 0.78) | 2.09 | 920, 953 | 0 | 2798, 0.280 | 3384fa917db2582a |
| U3_40x15_s1 | bounded (neither) | max_simple_iter | 5,000 | 4,187 | 2.61e-07 (4,998), 3.12e-07 | 0.00651 | 1.59, (6.1, 0.9) | 3.35 | 164, 172 | 0 | 84, 0.017 | a93a1058b6de1d39 |
| U3_40x15_s1_cap15k | converged | error_estimate_and_continuity | 10,254 | 4,187 | 4.43e-11 (10,254), 4.43e-11 | 9.79e-07 | 1.59, (6.1, 0.9) | 3.35 | 151, 172 | 0 | 167, 0.016 | 2254bad879b6702d |
| U3_40x15_s10 | converged | error_estimate_and_continuity | 344 | 187 | 1.18e-09 (344), 1.18e-09 | 9.44e-07 | 1.59, (6.1, 0.9) | 4.24 | 162, 167 | 0 | 6, 0.018 | 8e731bbb49b737a6 |
| U3_80x30_s1 | diverged | diverged | 618 | - | 2.91e-03 (30), 7.54e-01 | inf | 104, (6.85, 1.75) | 104 | 342, 375 | 0 | 24, 0.039 | 00fdb5fadd015a0b |
| U3_80x30_s10 | bounded (neither) | max_simple_iter | 5,000 | - | 3.21e-04 (149), 3.74e-03 | inf | 1.48, (2.75, 0.35) | 2.18 | 358, 373 | 0 | 218, 0.044 | 1c14462c83bb3dd9 |
| U3_80x30_s50 | bounded (neither) | max_simple_iter | 5,000 | - | 3.30e-04 (149), 2.12e-03 | inf | 1.48, (2.75, 0.35) | 2.18 | 358, 374 | 0 | 271, 0.054 | e4d1a2050ae7c2ee |
| U3_200x75_s1 | diverged | diverged | 463 | - | 1.35e-03 (23), 3.35e-01 | inf | 101, (1.42, 0.98) | 101 | 863, 926 | 0 | 115, 0.249 | ec5092f7f7cd24cf |
| U3_200x75_s10 | bounded (neither) | max_simple_iter | 10,000 | - | 7.55e-04 (5,023), 2.34e-03 | 33 | 1.92, (2.66, 0.66) | 2.12 | 918, 952 | 0 | 2778, 0.278 | 084da2c28b5f3682 |
| Z2_40x15_s1 | converged | error_estimate_and_continuity | 502 | 301 | 9.36e-10 (502), 9.36e-10 | 9.6e-07 | 1.56, (6.1, 0.9) | 2.16 | 161, 172 | 0 | 8, 0.016 | c739682a91ef4616 |
| Z2_80x30_s1 | bounded (neither) | max_simple_iter | 5,000 | - | 1.93e-03 (431), 3.42e-03 | 19.6 | 1.45, (2.65, 0.75) | 3.84 | 355, 388 | 0 | 202, 0.040 | cc7640d888b5b613 |
| Z2_80x30_s10 | converged | error_estimate_and_continuity | 630 | 342 | 3.74e-10 (630), 3.74e-10 | 9.76e-07 | 1.45, (2.65, 0.75) | 3.85 | 343, 387 | 0 | 25, 0.040 | 9ca5b7c3b045bcaf |
| Z2_200x75_s1 | diverged | diverged | 574 | - | 1.62e-02 (48), 2.96e-01 | inf | 100, (0.46, 0.3) | 100 | 785, 1068 | 0 | 138, 0.241 | 59af481f26298307 |
| Z2_200x75_s10 | bounded (neither) | max_simple_iter | 10,000 | - | 2.45e-04 (449), 7.81e-04 | 33.9 | 1.6, (0.5, 0.02) | 7.86 | 928, 1066 | 0 | 2759, 0.276 | 269995326a5d0cd0 |
| Z3_40x15_s1 | converged | error_estimate_and_continuity | 1,313 | 705 | 3.24e-10 (1,313), 3.24e-10 | 8.93e-07 | 1.58, (6.1, 0.9) | 2.17 | 159, 173 | 0 | 21, 0.016 | 9d91db9c5554c76a |
| Z3_80x30_s1 | diverged | diverged | 1,039 | - | 1.09e-02 (43), 7.50e-01 | inf | 100, (0.55, 0.85) | 100 | 346, 377 | 0 | 39, 0.038 | 94b2b8d6c1e47067 |
| Z3_80x30_s10 | converged | error_estimate_and_continuity | 758 | 490 | 4.66e-10 (758), 4.66e-10 | 9.86e-07 | 1.46, (2.65, 0.65) | 3.58 | 350, 380 | 0 | 31, 0.041 | f6129586046aa431 |
| Z3_200x75_s1 | diverged | diverged | 522 | - | 1.35e-02 (5), 3.27e-01 | inf | 100, (1.14, 1.26) | 100 | 796, 1020 | 0 | 128, 0.246 | 6fa7b0621e5ee832 |
| Z3_200x75_s10 | bounded (neither) | max_simple_iter | 10,000 | - | 5.07e-04 (1,808), 3.67e-03 | inf | 1.59, (2.74, 1.14) | 6.29 | 926, 1017 | 0 | 2754, 0.275 | 13f0dc1bf7a5f036 |
| Z4_40x15_s1 | converged | error_estimate_and_continuity | 3,528 | 1,531 | 1.22e-10 (3,528), 1.22e-10 | 9.77e-07 | 1.59, (6.1, 0.9) | 3.05 | 155, 172 | 0 | 57, 0.016 | b1aca42e5c4497f6 |
| Z4_80x30_s1 | diverged | diverged | 785 | - | 1.24e-02 (41), 5.02e-01 | inf | 103, (1.05, 0.65) | 103 | 337, 363 | 0 | 29, 0.037 | 4ac3205311b8db79 |
| Z4_80x30_s10 | bounded (neither) | max_simple_iter | 5,000 | - | 7.48e-04 (106), 1.43e-03 | inf | 1.47, (2.75, 0.35) | 2.98 | 361, 369 | 0 | 213, 0.043 | 0492dc9d4bc12405 |
| Z4_200x75_s1 | diverged | diverged | 410 | - | 8.34e-03 (13), 2.85e-01 | inf | 101, (0.74, 0.54) | 101 | 824, 943 | 0 | 104, 0.253 | 9de0990e09efb6ed |
| Z4_200x75_s10 | bounded (neither) | max_simple_iter | 10,000 | - | 6.16e-04 (9,898), 4.95e-03 | inf | 1.72, (2.7, 0.18) | 5 | 916, 950 | 0 | 2735, 0.273 | 63438ff48d16c449 |
| L_40x15_s1 | bounded (neither) | max_simple_iter | 5,000 | - | 3.38e-02 (22), 6.97e-02 | inf | 4.84, (6.5, 1.7) | 14.4 | 167, 178 | 0 | 81, 0.016 | d8560e54a5c2f789 |
| L_40x15_s10 | converged | error_estimate_and_continuity | 373 | 217 | 1.13e-09 (373), 1.13e-09 | 9.33e-07 | 1.59, (6.1, 0.9) | 18.4 | 163, 171 | 0 | 7, 0.019 | bfc5c54f23762883 |
| L_80x30_s1 | diverged | diverged | 639 | - | 1.62e-02 (19), 7.25e-01 | inf | 102, (6.25, 1.05) | 102 | 338, 362 | 0 | 24, 0.038 | d42a1895f16d892c |
| L_80x30_s10 | bounded (neither) | max_simple_iter | 5,000 | - | 4.36e-04 (2,938), 3.81e-03 | inf | 1.49, (2.75, 0.35) | 9.8 | 360, 370 | 0 | 208, 0.042 | dbe6b85a3609aa07 |
| L_80x30_s50 | bounded (neither) | max_simple_iter | 5,000 | - | 4.14e-04 (4,894), 1.89e-03 | inf | 1.49, (2.75, 0.35) | 9.8 | 360, 371 | 0 | 269, 0.054 | e39d0123d5080e32 |
| L_200x75_s1 | diverged | diverged | 423 | - | 2.39e-03 (15), 3.07e-01 | inf | 101, (2.9, 1.54) | 101 | 865, 926 | 0 | 105, 0.248 | 79f011662f8945a7 |
| L_200x75_s10 | bounded (neither) | max_simple_iter | 10,000 | - | 7.36e-04 (7,178), 6.27e-03 | inf | 2.11, (1.02, 0.02) | 3.79 | 914, 951 | 0 | 2732, 0.273 | 8ab944bff35cee8a |
| U1_80x30_s10_r1e-2 | converged | error_estimate_and_continuity | 764 | 339 | 2.74e-10 (764), 2.74e-10 | 9.76e-07 | 1.4, (2.75, 1.25) | 1.49 | 37, 182 | 0 | 10, 0.013 | 89f263af42c0982f |
| U1_80x30_s10_r1e-4 | converged | error_estimate_and_continuity | 745 | 350 | 2.94e-10 (745), 2.94e-10 | 9.95e-07 | 1.4, (2.75, 1.25) | 1.49 | 161, 269 | 0 | 19, 0.025 | ed444bc6339ced98 |
| U1_200x75_s10_r1e-2 | converged | error_estimate_and_continuity | 3,981 | 1,622 | 3.67e-11 (3,960), 4.01e-11 | 9.66e-07 | 1.45, (0.5, 0.02) | 1.63 | 83, 440 | 0 | 225, 0.057 | dcc325b7271e748b |
| U1_200x75_s10_r1e-4 | converged | error_estimate_and_continuity | 2,329 | 1,155 | 5.99e-11 (2,329), 5.99e-11 | 8.88e-07 | 1.45, (0.5, 0.02) | 1.63 | 445, 666 | 0 | 357, 0.153 | 4bc06d582d475542 |
| U2_80x30_s10_r1e-2 | bounded (neither) | max_simple_iter | 5,000 | - | 6.28e-04 (126), 2.32e-03 | inf | 1.5, (2.75, 0.45) | 1.72 | 31, 211 | 0 | 66, 0.013 | 8998159068ac6e64 |
| U2_80x30_s10_r1e-4 | bounded (neither) | max_simple_iter | 5,000 | - | 7.10e-04 (140), 1.68e-03 | 402 | 1.46, (2.75, 0.35) | 1.72 | 217, 300 | 0 | 152, 0.030 | 5a63e4169a7d91f5 |
| U2_200x75_s10_r1e-2 | bounded (neither) | max_simple_iter | 10,000 | - | 6.82e-04 (6,019), 5.12e-03 | 417 | 1.85, (2.7, 0.82) | 2.15 | 78, 419 | 0 | 539, 0.054 | 001366a15d9198e1 |
| U2_200x75_s10_r1e-4 | bounded (neither) | max_simple_iter | 10,000 | - | 7.34e-04 (296), 3.75e-03 | inf | 1.98, (1.22, 0.02) | 2.16 | 548, 719 | 0 | 1793, 0.179 | 4a7f8bd0810a2cfa |
| U1_80x30_s10_cf | converged | error_estimate_and_continuity | 713 | 329 | 2.93e-10 (713), 2.93e-10 | 9.63e-07 | 1.38, (2.75, 1.15) | 1.46 | 310, 347 | 0 | 28, 0.039 | f9271762e86834db |
| U1_200x75_s10_cf | converged | error_estimate_and_continuity | 2,203 | 1,134 | 7.97e-11 (2,196), 8.39e-11 | 9.49e-07 | 1.44, (0.5, 0.02) | 1.62 | 745, 863 | 0 | 511, 0.232 | dd916e4f54b065fe |
| U2_80x30_s10_cf | converged | error_estimate_and_continuity | 2,293 | 982 | 9.67e-11 (2,293), 9.67e-11 | 9.92e-07 | 1.47, (2.75, 0.45) | 1.7 | 343, 371 | 0 | 95, 0.041 | 75798ea337563c50 |
| U2_200x75_s10_cf | bounded (neither) | max_simple_iter | 10,000 | - | 5.57e-04 (964), 2.59e-03 | inf | 1.7, (0.9, 1.14) | 2.08 | 919, 949 | 0 | 2633, 0.263 | 0d4ced3f4d1b94b3 |
| U3_80x30_s10_cf | bounded (neither) | max_simple_iter | 5,000 | - | 3.90e-04 (207), 4.53e-04 | inf | 1.45, (2.75, 0.45) | 2.18 | 362, 373 | 0 | 214, 0.043 | f9356bae4d3b8c4a |
| Z4_80x30_s10_cf | bounded (neither) | max_simple_iter | 5,000 | - | 7.77e-04 (111), 2.13e-03 | inf | 1.47, (2.75, 0.45) | 2.98 | 363, 370 | 0 | 216, 0.043 | d7d930e3aa77c5d1 |
| L_80x30_s10_cf | bounded (neither) | max_simple_iter | 5,000 | - | 4.12e-04 (399), 6.17e-04 | 34.3 | 1.46, (2.75, 0.45) | 9.8 | 365, 378 | 0 | 211, 0.042 | f08a39cbe905c3dc |

**40x15.** Every field converges at one sweep except as the cap cuts it: U1 at 1,535, Z2 at 502,
Z3 at 1,313, Z4 at 3,528; U2 and U3 reach the prompt's cap of 5,000 with the residual still
falling at about 0.997 per iteration (U2's estimate was 1.05e-6 at outer 5,000, just above the
tolerance), and the supplementary reruns converge them at 5,012 and 10,254. The velocity-step
stops, 706 at U1 and 2,132 at U2, are step 4's ladder's 728 and 2,115 to within the 0.5%
viscosity difference of section 2.2; U3's is 4,187. Ten sweeps converge every field in 333 to
373 outer iterations, nearly one count across the range and laminar, as step 0 found under the
old outlets (1,205 to 1,240 there). L at one sweep is the exception: bounded by the rule, with
the largest speed between 2.6 and 14.4 m/s through the run, the residual between 3e-2 and 6e-1,
and no trend: the oscillation step 0's section 7.6 described, under the new outlets.

**80x30.** One sweep converges nothing. U2, U3, Z3, Z4 and L diverge (past 100 m/s between
outer 618 and 1,039); U1 and Z2 stay bounded to the cap with the residual between 1e-3 and 1e-2
and the largest speed between 1.4 and 2.2 m/s. Ten sweeps converge U1 (745), Z2 (630) and Z3
(758), and leave U2, U3, Z4 and L bounded to the cap: in each the residual settles after about
150 iterations into an oscillation between about 1e-3 and 4e-3 (median 2.1e-3 over the last 500)
while the largest speed is steady to 1e-3 m/s at (2.75, 0.35), the cell above return 2. Fifty
sweeps change nothing in the three of those rows run at fifty, U2, U3 and L: the same tails to
within a few percent (section 5.3.3). So on
this grid the sweep aid holds down to Z3's mixing (core median 5e-4 m^2/s) and fails below it,
whatever the sweep count.

**200x75.** One sweep diverges every field, between outer 410 and 824; U1's residual never falls
below 5.4e-4 (at outer 58) and its largest speed passes 10 m/s by outer 500. Ten sweeps converge
U1 at 2,329 (velocity_step at 1,155) and nothing else. Z2 stays bounded to the 10,000 cap with
the residual's median per thousand iterations between 5.7e-4 and 7.1e-4 from the first thousand
to the last, no trend, and the largest speed steady to 7e-3 m/s at (0.5, 0.02), the supply's
first cell; its least residual, 2.45e-4, came at outer 449. U2, U3, Z3, Z4 and L stay bounded
with the residual between about 1e-3 and 7e-3 (medians per thousand 1.7e-3 to 4.1e-3, no trend
after the first thousand) and the largest speed swinging between 1.6 and 2.1 m/s at (1.22,
0.02), the last floor cell of return 1: a wider swing than on 80x30, where the largest speed was
steady. The bounded 200x75 rows took 2,730 to 2,800 s each at 0.27 to 0.28 s per outer
iteration, ten processes at once.

#### 5.3.1 The two rows past the cap

U2 and U3 on 40x15 at one sweep were falling at the cap, so they were rerun with the cap at
15,000 and nothing else changed (`--cap 15000 --tag cap15k`). Their first 5,000 iterations
reproduce the capped runs' residual histories bit for bit (the records' histories are equal over
that range), and they converge at 5,012 and 10,254. The rows are supplementary: the matrix's
classification stands on the prompt's caps, and these say what the caps cut.

#### 5.3.2 The velocity-step stop against the rule's

In every converged product row the rule's stop comes 1.55 to 2.45 times later than the
velocity-step stop would have (2.4 and 3.0 on the cavity): the step falls below 1e-6 of the
reference velocity about halfway, and the rule waits for its estimate, `step rho / (1 - rho)` over
0.45 m/s with rho about 0.997, to fall below 1e-6. At one sweep on 40x15 that is the difference
between 706 and 1,535 (U1) and between 4,187 and 10,254 (U3). The continuity conditions never bind
in a converged row: the worst per-cell imbalance at every stop at `pressure_rtol` 1e-8 is below
2.5e-13 kg/s per metre of depth against tolerances of 8e-8 to 3.2e-9 (2.4e-11 at most in the 1e-2
rows), and the signed sum below 2e-14.

#### 5.3.3 Fifty sweeps on 80x30

Prompt 34b's rule: a failure at ten is ambiguous between "ten is too few" and "sweeps do not
help at that mixing" until fifty is run. U2, U3 and L on 80x30 were run at fifty sweeps
(supplementary, 0.054 s per outer iteration against 0.043 at ten). All three reach the cap in the
same state as at ten, to within a few percent: residual median over the last 500 iterations
2.08e-3, 2.07e-3 and 2.09e-3 against 2.10e-3, 2.14e-3 and 2.18e-3, least residual over the run
7.11e-4, 3.30e-4 and 4.14e-4 against 7.09e-4, 3.21e-4 and 4.36e-4, the largest speed steady at
the same cell. The momentum
solve is exhausted at ten; what remains is not the inner solve.

### 5.4 Measurement 2: same answer, different path

Where one and ten sweeps both converge (the three uniform fields on 40x15, U2 and U3 through
the reruns past the cap):

| Field | Grid | Outer: one, ten | Wall (s): one, ten | max |du|, |dv| (m/s) | Largest speed difference, at | RMS, median | Largest / tolerance (4.5e-7 m/s) |
|---|---|---|---|---|---|---|---|
| U1 | 40x15 | 1,535, 333 | 25, 7 | 1.31e-07, 2.50e-07 | 2.51e-07, (1.3, 1.1) | 2.55e-08, 1.70e-13 | 0.557 |
| U2 | 40x15 | 5,012, 334 | 83, 6 | 1.56e-07, 1.80e-07 | 1.81e-07, (1.3, 0.5) | 2.81e-08, 4.63e-12 | 0.403 |
| U3 | 40x15 | 10,254, 344 | 167, 6 | 6.57e-08, 1.14e-07 | 1.17e-07, (1.1, 1.3) | 1.51e-08, 2.58e-12 | 0.26 |

The two paths land within the stopping tolerance of each other: the largest cell-centred
difference is 0.26 to 0.56 of 4.5e-7 m/s, the RMS a twentieth of that, and the median 1e-13 to
5e-12 m/s, so the fields are the same discrete solution to the rule's accuracy, differing at a
few cells beside the server rack's top. Prediction (f) asked for ten times the tolerance.

### 5.5 Measurement 3: grid convergence

U1 is the one field converged on all three grids at one sweep count (ten: 333, 745 and 2,329). No
other field converged on all three grids (Z2 converges on 40x15 and 80x30 only), so the
measurement has one field. The points are section 2.3's; "far" excludes points within 0.2 m of an
obstacle rectangle.

| Field | Points | Coarse grid | Comp. | Points (far) | Largest |diff| (m/s), at | RMS | Largest, far from obstacles, at | RMS far | Scale: fine max |comp.| |
|---|---|---|---|---|---|---|---|---|---|
| U1 | sensors | 40x15 | u | 4 (4) | 0.205, (1.00, 1.50) | 0.169 | 0.205, (1.00, 1.50) | 0.169 | 0.405 |
| U1 | sensors | 40x15 | v | 4 (4) | 0.527, (1.00, 1.50) | 0.3 | 0.527, (1.00, 1.50) | 0.3 | 1.25 |
| U1 | sensors | 80x30 | u | 4 (4) | 0.0705, (1.00, 1.50) | 0.0359 | 0.0705, (1.00, 1.50) | 0.0359 | 0.405 |
| U1 | sensors | 80x30 | v | 4 (4) | 0.194, (1.00, 1.50) | 0.0981 | 0.194, (1.00, 1.50) | 0.0981 | 1.25 |
| U1 | vertical x=2.7 | 40x15 | u | 141 (141) | 0.292, (2.70, 2.10) | 0.141 | 0.292, (2.70, 2.10) | 0.141 | 0.297 |
| U1 | vertical x=2.7 | 40x15 | v | 141 (141) | 1.26, (2.70, 0.10) | 0.283 | 1.26, (2.70, 0.10) | 0.283 | 1.33 |
| U1 | vertical x=2.7 | 80x30 | u | 141 (141) | 0.085, (2.70, 0.16) | 0.0307 | 0.085, (2.70, 0.16) | 0.0307 | 0.297 |
| U1 | vertical x=2.7 | 80x30 | v | 141 (141) | 0.606, (2.70, 0.10) | 0.0796 | 0.606, (2.70, 0.10) | 0.0796 | 1.33 |
| U1 | horizontal y=1.2 | 40x15 | u | 256 (196) | 0.479, (7.90, 1.20) | 0.075 | 0.479, (7.90, 1.20) | 0.0842 | 0.479 |
| U1 | horizontal y=1.2 | 40x15 | v | 256 (196) | 0.646, (4.98, 1.20) | 0.242 | 0.579, (2.98, 1.20) | 0.198 | 1.36 |
| U1 | horizontal y=1.2 | 80x30 | u | 256 (196) | 0.252, (7.90, 1.20) | 0.0233 | 0.252, (7.90, 1.20) | 0.0246 | 0.479 |
| U1 | horizontal y=1.2 | 80x30 | v | 256 (196) | 0.439, (2.30, 1.20) | 0.0897 | 0.245, (0.10, 1.20) | 0.0763 | 1.36 |

| Field | Sensor (x, y) | u: 40x15, 80x30, 200x75 (m/s) | v: 40x15, 80x30, 200x75 (m/s) |
|---|---|---|---|
| U1 | near_door (1.0, 1.5) | -0.1996, -0.3344, -0.4050 | -0.7232, -1.0561, -1.2499 |
| U1 | above_gap_1 (2.7, 2.5) | -0.0545, -0.1891, -0.2013 | -0.4950, -0.5097, -0.4959 |
| U1 | above_gap_2 (4.8, 2.5) | +0.3256, +0.1340, +0.1368 | -0.4713, -0.4451, -0.4514 |
| U1 | hood_entry (6.2, 1.2) | -0.3862, -0.2677, -0.2639 | -1.1654, -0.9030, -0.8772 |

Reading. The largest differences are large: 40x15 against 200x75 differs by up to 1.26 m/s in v at
the foot of the vertical line, (2.7, 0.1), the cell above return 2, and by 0.53 m/s in v at the
near_door sensor; 80x30 against 200x75 by 0.61 and 0.19 m/s at the same two places. Three things
are in those numbers. First, the boundary data: the 40x15 return face velocity is 1.054 m/s
against 1.209 on 200x75 (section 5.2), and the foot of the vertical line sits on return 2. Second,
the SOLID staircase: the server rack spans x 1.4 to 2.4 on 40x15, 1.5 to 2.3 on 80x30 and 1.48 to
2.32 on 200x75, and the litho tool 3.2 to 4.6, 3.2 to 4.5 and 3.2 to 4.52, so the gap the vertical
line runs down is 0.8 m wide on 40x15 and 0.9 and 0.88 m on the finer grids, with return 2
(configured 2.4 to 2.95) covering 0.6, 0.6 and 0.56 m of its floor. Third, the flow beside the
door: the near_door sensor at (1.0, 1.5) reads v of -0.72, -1.06 and -1.25 m/s on the three grids,
a downward jet along the door wall that the coarse grids under-resolve, and it is 0.5 m from any
obstacle, so it is neither of the first two. The RMS differences over the lines fall by a factor
of 2.7 to 4.7 from 40x15 to 80x30 against 200x75, about the factor of 4 a second-order scheme
gives for a halved cell; the largest differences fall by 1.5 to 3.4. At the sensors the near_door
difference falls by 2.7 (v) and 2.9 (u); at the other three sensors 80x30 is within 0.03 m/s of
200x75 in both components, where 40x15 is 0.12 to 0.29 m/s off. On the "far" subsets the picture
is the same.

So 80x30 is closer to 200x75 than 40x15 is, by about the factor a second-order scheme gives,
and at three sensors of four it is within 0.03 m/s; but at the fourth, in the jet along the
door wall, it is 0.19 m/s off, and on the lines up to 0.61 m/s, against a supply of 0.45 m/s and
a largest speed of 1.45. Whether 200x75 itself is resolved is not
measured here (it would need a finer grid); the differences between 80x30 and 200x75 say that it
is not resolved to better than about 0.1 to 0.2 m/s at the sensors.

### 5.6 Measurement 4: the corner rule

U1 and U2 on 80x30 and 200x75 at ten sweeps, the count that converged U1 on both grids; no
count converged U2 on either, so its rows compare the bounded behaviour. The corner-free
predictor is `tail43.corner_free_class()`.

| Run | Setting | Class: this, committed | Outer: this, committed (ratio) | velocity_step at: this, committed | CG mean, largest: this; committed | Wall (s): this, committed | Faces: max |du|, |dv| (m/s) | Cells: largest speed difference, at; RMS |
|---|---|---|---|---|---|---|---|---|
| L_80x30_s10_cf | corner-free | bounded neither, bounded neither | 5,000, 5,000 (1.000) | -, - | 365, 378; 360, 370 | 211, 208 | - | - |
| U1_200x75_s10_cf | corner-free | converged, converged | 2,203, 2,329 (0.946) | 1,134, 1,155 | 745, 863; 735, 862 | 511, 524 | 7.17e-02, 1.05e-01 | 1.09e-01, (6.34, 0.82); 1.05e-02 |
| U1_80x30_s10_cf | corner-free | converged, converged | 713, 745 (0.957) | 329, 350 | 310, 347; 310, 348 | 28, 29 | 7.34e-02, 1.43e-01 | 1.30e-01, (3.05, 1.75); 2.01e-02 |
| U2_200x75_s10_cf | corner-free | bounded neither, bounded neither | 10,000, 10,000 (1.000) | -, - | 919, 949; 920, 953 | 2633, 2798 | - | - |
| U2_80x30_s10_cf | corner-free | converged, bounded neither | 2,293, 5,000 (0.459) | 982, - | 343, 371; 357, 370 | 95, 216 | - | - |
| U3_80x30_s10_cf | corner-free | bounded neither, bounded neither | 5,000, 5,000 (1.000) | -, - | 362, 373; 358, 373 | 214, 218 | - | - |
| Z4_80x30_s10_cf | corner-free | bounded neither, bounded neither | 5,000, 5,000 (1.000) | -, - | 363, 370; 361, 369 | 216, 213 | - | - |

**U1.** The corner rule costs nothing on the finer grids: with it removed the count falls from
745 to 713 on 80x30 and from 2,329 to 2,203 on 200x75, 4% and 5%, against the doubling on 40x15
(prompt 43b: 372 to 728). The converged fields differ by up to 0.13 m/s on 80x30, at (3.05,
1.75) beside the litho tool's left face, and 0.11 m/s on 200x75, at (6.34, 0.82) beside the hood
bench's corner; the RMS difference halves from 80x30 to 200x75 (0.020 to 0.0105 m/s), as a
first-order difference at the corners would. The corner rule's effect on the answer is therefore
of the same size as the 80x30-to-200x75 grid difference at the sensors, and it decreases only at
first order.

**U2.** Here the rule decides the outcome. With the committed rule U2 at ten sweeps is bounded to
the cap on 80x30 (section 5.3); with the corner QUICK zeroing removed it converges at 2,293
(velocity_step at 982). On 200x75 removing the rule does not converge U2 at ten sweeps: bounded to
the cap either way, with the same residual level (medians per thousand 3.6e-3 to 4.0e-3 without
the rule, 3.7e-3 to 4.0e-3 with it) and the same swinging cell. On 40x15 at one sweep the finding
was the reverse (prompt 43b: the committed rule converged Re 8,950 at 2,115 and the corner-free
run never did). The corner treatment is a first-order choice at a few faces that changes whether
the iteration converges at the middle of the range, in one direction on the coarse grid and the
other on the finer one.

**The other bounded rows without the rule.** Given U2's result, U3, L and Z4 on 80x30 at ten
sweeps were also run without the corner rule (supplementary). None converges: all three reach
the cap bounded, with the least residual over the run 3.9e-4 (U3), 4.1e-4 (L) and 7.8e-4 (Z4)
against 3.2e-4, 4.4e-4 and 7.5e-4 with the rule, the same oscillation at the same cell. The
corner rule's QUICK half decides U2 and nothing below it on this grid.

### 5.7 Measurement 5: the pressure tolerance

U1 and U2 on 80x30 and 200x75 at ten sweeps, `pressure_rtol` 1e-4 and 1e-2 against 1e-8.

| Run | pressure_rtol | Class: this, committed | Outer: this, committed (ratio) | velocity_step at: this, committed | CG mean, largest: this; committed | Wall (s): this, committed | Faces: max |du|, |dv| (m/s) | Cells: largest speed difference, at; RMS |
|---|---|---|---|---|---|---|---|---|
| U1_200x75_s10_r1e-2 | 1e-2 | converged, converged | 3,981, 2,329 (1.709) | 1,622, 1,155 | 83, 440; 735, 862 | 225, 524 | 2.87e-08, 3.04e-08 | 3.31e-08, (1.18, 0.14); 2.62e-09 |
| U1_200x75_s10_r1e-4 | 1e-4 | converged, converged | 2,329, 2,329 (1.000) | 1,155, 1,155 | 445, 666; 735, 862 | 357, 524 | 3.07e-10, 3.20e-10 | 4.02e-10, (1.1, 0.58); 2.90e-11 |
| U1_80x30_s10_r1e-2 | 1e-2 | converged, converged | 764, 745 (1.026) | 339, 350 | 37, 182; 310, 348 | 10, 29 | 1.09e-08, 1.11e-08 | 1.33e-08, (1.05, 0.65); 1.26e-09 |
| U1_80x30_s10_r1e-4 | 1e-4 | converged, converged | 745, 745 (1.000) | 350, 350 | 161, 269; 310, 348 | 19, 29 | 3.75e-10, 4.36e-10 | 4.91e-10, (1.05, 0.65); 4.78e-11 |
| U2_200x75_s10_r1e-2 | 1e-2 | bounded neither, bounded neither | 10,000, 10,000 (1.000) | -, - | 78, 419; 920, 953 | 539, 2798 | - | - |
| U2_200x75_s10_r1e-4 | 1e-4 | bounded neither, bounded neither | 10,000, 10,000 (1.000) | -, - | 548, 719; 920, 953 | 1793, 2798 | - | - |
| U2_80x30_s10_r1e-2 | 1e-2 | bounded neither, bounded neither | 5,000, 5,000 (1.000) | -, - | 31, 211; 357, 370 | 66, 216 | - | - |
| U2_80x30_s10_r1e-4 | 1e-4 | bounded neither, bounded neither | 5,000, 5,000 (1.000) | -, - | 217, 300; 357, 370 | 152, 216 | - | - |

**1e-4** reproduces 1e-8 on both grids for U1: the same outer count to the iteration (745 and
2,329), the same velocity-step stop, the final faces within 4.4e-10 m/s, at 161 against 310 CG
iterations per correction on 80x30 and 445 against 735 on 200x75, so 32% less wall time on 200x75.
**1e-2** stops at 764 against 745 on 80x30 (2.6% more) and at 3,981 against 2,329 on 200x75 (71%
more), the faces within 3e-8 m/s of the 1e-8 fields, with 37 and 83 CG iterations per correction;
on 200x75 the run still took 225 s against 524, because a correction at 1e-2 costs a ninth of one
at 1e-8 and the outer loop took less than twice as many. U2 converges at none of the three
tolerances on either grid; its rows are bounded to the cap with the same tails (on 200x75 the
residual's median per thousand is 3.5e-3 to 4.0e-3 at every tolerance), so the pressure tolerance
neither causes nor cures that, and the 1e-2 row costs a fifth of the 1e-8 row's wall time (539
against 2,798 s) for the same non-answer.

ADR-013 B's table found 1e-2 and 1e-4 never met the per-cell condition on the 80x30 room at Re 90
under the old outlets, where the outlets held a standing imbalance that the loose correction left
in the faces; with the fixed-flow outlets there is no standing imbalance (the worst cell at every
stop is below 2.5e-11, and below 2.5e-13 at 1e-8), and the loose corrections meet every condition.
What 1e-2 changes is the path: the outer loop takes more iterations to the same answer on the fine
grid.

### 5.8 Measurement 6: the cavity at Re 1,000

| Grid | Stop | Outer | velocity_step at | Wall (s) | CG mean, largest | u_min (sample at y; parabola at y) | v_max (sample at x; parabola at x) | v_min (sample at x; parabola at x) | Parabola against recalled: u_min, v_max, v_min |
|---|---|---|---|---|---|---|---|---|---|
| 40x40 | error_estimate_and_continuity | 6,039 | 2,516 | 118 | 181, 222 | -0.35918 at 0.1875; -0.35924 at 0.1855 | 0.34792 at 0.1625; 0.34798 at 0.1650 | -0.48227 at 0.9125; -0.48781 at 0.9035 | +7.5%, -7.7%, +7.5% |
| 80x80 | error_estimate_and_continuity | 16,668 | 5,480 | 901 | 310, 433 | -0.37858 at 0.1688; -0.37924 at 0.1749 | 0.36759 at 0.1563; 0.36771 at 0.1596 | -0.51372 at 0.9062; -0.51384 at 0.9076 | +2.4%, -2.4%, +2.5% |

Both grids converge at one sweep by the solver's own stop: 6,039 outer iterations on 40x40 and
16,668 on 80x80 (velocity_step at 2,516 and 5,480), with no correction at the cap. The
centreline extremes on 80x80, as the parabola through the sample and its neighbours gives them,
are u_min -0.3792 at y 0.175, v_max 0.3677 at x 0.160 and v_min -0.5138 at x 0.908. Against the
values the orchestrator recalled (u_min -0.38857 at 0.1717, v_max 0.37694 at 0.1578, v_min
-0.52708 at 0.9092, stated from memory and not checked against the paper) they are 2.4% to 2.5%
short in magnitude and within 0.004 in position; the 40x40 extremes are 7.5% to 7.7% short. The
80x80 differences are 3.1 to 3.2 times smaller than the 40x40 ones, a little better than second
order, which is what a second-order scheme short of the fine limit gives. The cavity's cell
Reynolds number on 80x80 is 12.5. No claim here rests on the recalled values; they orient the
reader, and the grid-to-grid ratio is the measurement.

## 6. Predictions against the measurement

| Prediction | Measured |
|---|---|
| (a) U1 converges at one sweep on all three grids; 200x75 count two to ten times 40x15's | **Fails.** U1 converges at one sweep on 40x15 only (1,535); on 80x30 it is bounded to the cap and on 200x75 it diverges at 824. At ten sweeps it converges on all three, 333, 745 and 2,329: the 200x75 count is 7.0 times the 40x15 count at ten sweeps |
| (b) U2 converges at one sweep on 40x15 and 80x30; ten on 200x75 | **Fails.** One sweep: 40x15 only (5,012, past the prompt's cap); 80x30 diverges at 767. Ten sweeps: bounded to the cap on 80x30 (5,000) and on 200x75 (10,000) |
| (c) U3 does not converge at one sweep on 200x75; ten: no prediction | Holds on the first part: diverges at 463. Ten sweeps: bounded to the cap at 10,000, the residual between 1e-3 and 7e-3 |
| (d) Each Z field converges wherever the uniform field at its median does | **Fails on 200x75.** Z2 converges where U1 does on 40x15 (one sweep) and 80x30 (ten), and not on 200x75, where U1 converges at ten sweeps and Z2 stays bounded with its residual steady at 6e-4 and its largest speed steady to 7e-3 m/s. Z4 follows U2 everywhere (one sweep on 40x15; bounded at ten on both finer grids). Z3 has no uniform twin: it converges at ten sweeps on 80x30 where Z4 does not, and is bounded on 200x75. On 40x15 at one sweep the Z fields converge in fewer iterations than their uniform twins (502 against 1,535; 3,528 against 5,012) |
| (e) L converges on 40x15 at ten sweeps and on neither finer grid | **Holds.** 373 on 40x15; bounded to the cap on 80x30 at ten and fifty sweeps; bounded to the cap at 10,000 at ten sweeps on 200x75. ADR-013 decision 6's finding stands under the fixed-flow outlets |
| (f) One and ten sweeps agree within ten times the tolerance | **Holds**, within the tolerance itself: 0.26 to 0.56 of 4.5e-7 m/s |
| (g) U1: 40x15 against 200x75 exceeds 0.1 m/s; 80x30 against 200x75 under half of it | First part holds (0.53 m/s at a sensor, 1.26 on a line). Second part holds for the RMS differences (ratios 0.21 to 0.37) and at the sensors (0.34 to 0.37), fails for the largest differences on the lines (ratios 0.48 to 0.68), which sit on return 2 and beside the hood |
| (h) The corner rule raises U1's count on 200x75 by under 20% | **Holds:** 5.7% (2,329 against 2,203); 4.5% on 80x30 |
| (i) 1e-2 stops at the same count as 1e-8 within 1%, U1 and U2, both finer grids | **Fails.** U1: 2.6% more on 80x30, 71% more on 200x75. U2 converges at neither tolerance. 1e-4, not predicted, stops at the same count to the iteration on both grids |
| (j) The cavity converges at one sweep on both grids; 80x80 extremes within 5% of the recalled values | **Holds:** 6,039 and 16,668; 2.4% to 2.5% from the recalled values |

Builder's predictions (section 3.1): (a) failed with the orchestrator's; (b) the 80x30 one-sweep
prediction (bounded) was wrong in kind (diverged), and ten sweeps did not converge it; (c) wrong:
U3 at ten sweeps on 200x75 is bounded, not converged; (e) held, bounded and not growing; (f) held,
with the largest difference 2.5e-7 m/s, under the 1e-5 I gave; (g) the largest differences sit
beside a return and the hood, and at the near_door sensor, not at an obstacle corner; (h) held on
both grids; (i) wrong for U1 on 200x75, where 1e-2 moved the count by 71%, and the reason I gave
for 1e-2 meeting the conditions (no standing imbalance) is what the records show; (j) held, v_min
within 2.5%, not the 10% I allowed.

## 7. What this implies

Stated as the prompt asks, without deciding.

**Which outcome holds, and what step 6 needs.** None of section 3.2's three outcomes holds cleanly
on 200x75. With one sweep nothing converges on either finer grid, so the first outcome (no aid) is
out. With ten sweeps the top of the range converges on 80x30 (U1 and Z2) and, as the uniform field
only, on 200x75 (U1 at 2,329; Z2 bounded); Z3 converges on 80x30 and not on 200x75; the middle and
bottom (U2, U3, Z4) and laminar air converge on neither finer grid, on 80x30 at ten and fifty
sweeps. So on 80x30 the second outcome (run with the sweep count measured) holds for effective
viscosities down to about 5e-4 m^2/s and the third (a stronger aid) below it; on 200x75, the grid
the plan scores VAL-018 on, the second holds for the uniform top of the range only and the third
for everything else, the non-uniform Z2 included. Two things qualify that. The corner rule decides
U2's convergence on 80x30: without its QUICK half U2 converges at ten sweeps (2,293), but U3, L
and Z4 stay bounded without it. And a k-epsilon field is not a uniform one: the model's mu_t is
large in the shear layers and small in the core, Z2's shape, and on 40x15 and 80x30 the Z fields
converge where their uniform twins do or better, while on 200x75 Z2 does not where U1 does, so the
field's shape cuts both ways. What step 6 needs is therefore a question for Alex with three parts:
whether ten sweeps is the setting for the coupled solve (it is the only count that converged
anything on the finer grids, and costs 5% of the outer iteration); whether the corner rule's QUICK
half is kept, given that it moves the answer by 0.1 m/s at obstacle corners on 200x75, costs
nothing in count at U1, and decides convergence at U2 in opposite directions on 40x15 and 80x30;
and whether a stronger aid (step 0 section 6.6's continuation in viscosity or pseudo-transient
continuation) is built before step 6 for the lower half of the range, or whether the coupled solve
is first tried at ten sweeps and its own field, since the model's field has not been shown to
behave as a uniform one. The bounded rows are a bounded oscillation at a residual of 1e-3 to 4e-3,
with the largest speed steady to 1e-3 m/s on 80x30 and swinging between 1.6 and 2.1 m/s beside
return 1 on 200x75; the records do not locate the cell that moves, which a probe would need to
before an aid is chosen.

**Whether VAL-018 (criterion 10) can stand as written, and on which grid.** VAL-018 asks the
product configuration to stop by `error_estimate_and_continuity` within its cap. On 200x75 the
one row that converged stopped by that rule, U1 at ten sweeps at 2,329 outer iterations, about 9
minutes on this machine beside nine other runs, so the criterion is meetable there at the
uniform top of the range with ten sweeps. Below the top, and for the non-uniform Z2 at the top,
it is not met at any sweep count measured within 10,000 outer iterations (46 minutes). Whether the
criterion stands as written therefore depends on what the coupled field's effective viscosity
turns out to be, which step 6 measures; the criterion's text does not need to change for that,
but its cap does: 5,000 outer iterations on 40x15 cut two one-sweep rows that converge at 5,012
and 10,254, and the rule's stop comes about twice as late as the velocity-step stop in every
row, so a cap set from velocity-step experience is too low by that factor. On which grid: the
measurement says 80x30 is not resolved to better than 0.2 m/s at a sensor against 200x75, and
200x75 is not shown to be resolved; 40x15 is a different room at the returns (section 5.2). If
VAL-018 is scored on 200x75, as the plan says, its count and cost are the ones above; if on
80x30 for cost, the grid difference is the size of the quantity measured.

**Whether `pressure_rtol` can relax.** From 1e-8 to 1e-4, yes, on this room: the same outer
count to the iteration, the faces within 4.4e-10 m/s, a third less wall time on 200x75. To 1e-2:
the same answer within 3e-8 m/s, but 71% more outer iterations on 200x75 and a cheaper run
overall. ADR-013 decision 3's reason for 1e-8, the standing imbalance the loose correction left
at the outlets, is gone with the fixed-flow outlets. What the measurement does not say is
whether 1e-4 holds under the coupled solve, where the field changes every outer iteration; a
decision would be for the product room as configured, and a row under step 6's solver would
confirm it.

**What the corner rule costs on the grids that matter.** In outer count, 4% to 5% at U1 on
80x30 and 200x75, against a doubling on 40x15. In the answer, 0.11 to 0.13 m/s at obstacle
corners, 0.01 to 0.02 m/s RMS, halving with the cell size: first order, and on 200x75 of the
same size as the grid difference at the sensors. In convergence, it decides U2 on 80x30 (and
the reverse on 40x15), and nothing below U2 on 80x30. The question is whether a rule chosen for
conservativeness at a few faces should decide the iteration's convergence, and whether its
first-order footprint is acceptable in a VAL-018 scored at the sensors, which are 0.4 to 0.6 m
from the nearest obstacle rectangle.

## 8. What this does not settle

- The bounded rows' mechanism: where the oscillation sits and what it responds to. The records
  keep the residual and the largest speed, not the cell of the largest change.
- Whether 200x75 resolves the room: a finer grid was not run.
- Whether the coupled k-epsilon field converges where a frozen field does: step 6's measurement.
- Whether the Z3 result (converges at ten sweeps on 80x30 where Z4 does not) marks a threshold
  in the mixing or in the field's shape: no uniform field at 5e-4 m^2/s was run.
- The cavity extremes against a sourced reference: the recalled values were not checked against
  Botella and Peyret (1998).
