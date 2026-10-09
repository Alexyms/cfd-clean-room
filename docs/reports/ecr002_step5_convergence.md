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

To be written after the runs.
