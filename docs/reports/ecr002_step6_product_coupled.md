# ECR-002 Step 6: The Product Room Under the Coupled Model

**Date:** 2026-10-09
**Tree:** branch `docs/ecr002-coupled-product` from main at 3bd8146. Nothing under `src/`,
`validation/`, `configs/` or `tests/` changes: every run goes through the committed
`StaggeredSolver` with the configuration's turbulence section, and the committed
`TransportSolver` with the converged flow handed to it, the product configuration overridden in
the probe as step 5's was.
**Instruments:** `docs/reports/probe46/` (committed): `coupled46.py` (the room, the runner, the
timing probe), `transport46.py` (measurement 4), `tables46.py` (the tables), `figures46.py` (the
figures), `bounded46.py` (step 5's `bounded44.py` pointed at this step's records), `assemble46.py`
(sections 5 to 8 of this file from the generated tables) and the launcher `run46.sh`. Raw output under `results/builder46/` (untracked). The figures are the small PNGs
beside this report.
**Order:** sections 1 to 4 were committed before any run (the first commit of the branch, this
file alone). The scripts followed in later commits. Sections 5 onward were written after the
runs. Each run log's first line carries its start time.
**Policy:** `docs/REVIEW_POLICY.md`, "What blocks". The numbers live in the tables
`tables46.py` generates from the records; the sentences point at the tables.

## 1. The question

Step 5 found that frozen eddy-viscosity fields do not let the room converge on the finer grids:
with ten momentum sweeps the uniform top of k-epsilon's range converges on 200x75 and nothing
else there, and nothing below about 5e-4 m^2/s on 80x30 at any sweep count measured. Alex
decided (ADR-012 D's note of 2026-10-09) to try the coupled model before building any
convergence aid, on the argument that the real eddy viscosity responds to the flow and damps the
disturbances a frozen one cannot. This measurement tests that argument. If the room converges, it
asks the product's question for the first time: where do particles go, and do those answers hold
still when the grid or the model variant changes? The product's purpose is comparative
(`docs/SYSTEM.md` section 1; ECR-002 criterion 10's note): hotspots and rankings, not absolute
counts.

Four measurements, in the order the prompt gives them:

1. Does the coupled room converge, with either variant, on 40x15, 80x30 and 200x75?
2. How does the inlet's turbulence setting move the core's eddy viscosity and the convergence?
3. What does the converged room look like: the flow, the eddy viscosity, k, and y+ at the walls?
4. Where do particles go, and does the answer agree between grids and between variants?

## 2. Method (fixed before the runs)

### 2.1 The room and the settings

The product room is `configs/clean_room_default.yaml` regridded, with these keys overridden in
the probe (Alex's decisions of 2026-10-09, ADR-012 D's note and ECR-002 section 8 step 8):

- `turbulence`: `model: k_epsilon`, `variant` standard or rng, `wall_treatment:
  scalable_wall_functions`, `cfl_number` 0.25, `alpha_turbulence` 0.7, `max_iter` 500, `tol`
  1e-10. The last three are VAL-016's (`tests/couette_reference.py`, `case_raw`); 0.25 is the
  Courant number every VAL-016 run used, since 0.5 locked the coupled iteration into a limit
  cycle (ADR-012 C's note).
- `hepa_supply`: `turbulence_intensity` 0.05 and `dissipation_length` 0.1 m, the baseline;
  measurement 2 varies them.
- `solver`: `stopping_rule: error_estimate` with step 5's tolerances (`iteration_error_tol` 1e-6
  and `mass_imbalance_tol = 1e-4 rho V_min / t_end`, ADR-011 G's formula: 8.0e-8 on 40x15, 2.0e-8
  on 80x30, 3.2e-9 kg/s per metre of depth on 200x75); `momentum_sweeps` 10; `pressure_rtol` 1e-4
  (one check row at 1e-8); `max_simple_iter` 10,000 on every grid; `alpha_velocity` 0.5;
  `max_pressure_iter` 5,000 and `alpha_pressure` 0.3 as the file states them.
- Everything else as the file states it: the fixed-flow outlets, the obstacles, the sensors, the
  particle classes, `transport` (`cfl_number` 0.1, UMIST, `turbulent_schmidt` 0.7).

Each record carries the full configuration mapping the run was built from, the commit, the
start time, the BLAS thread setting and the wall time. From rest in every run: the solver's
initial state, k and eps uniform at the inlet's values.

Each run records, per outer iteration: the solver's residual, the CG iterations of the pressure
correction and whether it reached its cap, the largest cell-centred speed and its cell, the
velocity step and the eddy-viscosity step the stopping rule read, the rule's two estimates
(conditions (a) and (e)), and the three continuity readings (conditions (b), (c), (d)), asked of
the corrector every iteration by a recording subclass of the rule (the probe's one substitution,
as VAL-016's `couette45.py` made it; the rule's decision is unchanged). With `--locate`, the cell
of the largest change of each velocity component and of nu_t between iterates. At the end: the
stop, the outer count, the five readings at the stop, the face hash (SHA-256 over u's bytes then
v's, `docs/reports/ecr003_step2_baseline.md` section 2.3), the iteration from which each of the
five conditions holds to the end, and the fields (faces, cell-centred velocity, pressure, k, eps,
nu_t, the y+ of every wall node) in an `.npz`.

**Classification.** Step 5's rule, in its order (`docs/reports/ecr002_step5_convergence.md`,
section 2.1, with round 2's window amendment): *diverged* (a cell-centred speed past 100 m/s or
a non-finite field); *converged* (the solver's own stop, `error_estimate_and_continuity`);
*growing* (at the cap with the median of the largest speed over the last 500 iterations above
5 m/s); *bounded and not converged* (at the cap with that median under 5 m/s), with step 0's
sub-classes falling, stalled or neither over the last 500 iterations. A `PositivityError` is its
own outcome, reported with the iteration, the cell and the run's pressure cap hits, as the
prompt asks.

**Machine and threads.** AMD Ryzen AI 9 HX 370 (12 cores, 24 logical processors), 64 GB,
Windows 11, Python 3.13, NumPy 2.4 with OpenBLAS 0.3.31. Every run sets
`OPENBLAS_NUM_THREADS=1` before NumPy loads and the CG solve runs under the code's own limit. The
flow runs go in parallel processes, so wall times are not like for like with single-process
figures; section 5.1 states how many ran at once.

**Cost before the set.** A timing probe of 20 outer iterations per grid (standard variant) runs
first. If it projects the whole set past about eight hours at the 10,000 cap, the estimate and a
reduced set are reported before anything else runs.

### 2.2 Measurements

1. **Does it converge.** Both variants on the three grids, six runs, plus the check row 200x75
   standard at `pressure_rtol` 1e-8. Every run classified. For a run that does not converge,
   `bounded44.py`'s characterisation of its residual and largest-speed tails (the least, median
   and largest; the amplitude as the 95th over the 5th percentile; the drift; `compare34b`'s
   period and the strongest recurrence), and for one such run per grid a rerun with `--locate`
   and `bounded44.py where`'s region of largest change.
2. **The inlet's turbulence.** 80x30 standard at two more inlet settings: intensity 0.02 with
   dissipation length 0.05 m, and 0.10 with 0.30 m. Convergence, and the core's nu_t / nu beside
   the baseline's.
3. **What the converged room looks like.** For every converged run, one figure of three panels:
   streamlines coloured by speed on the cell-centred field, nu_t / nu on a log scale, and k. One
   table row per run: the outer count and stop, the five readings at the stop, the wall time,
   nu_t / nu in the core (median and 95th percentile), and y+ at the first wall nodes (median,
   least and largest, with the share below 11.53). The core is step 5's: the non-SOLID cells
   whose centre lies above the equipment tops, y > 2.0 m. nu is air's, mu / rho = 1.508e-5
   m^2/s. y+ is the quantity the wall functions evaluate (ADR-012 B): for every wall cell and
   each of its wall sides, `y* = C_mu^(1/4) k_P^(1/2) y_P / nu` with k_P the cell's k and y_P the
   distance from its centre to that wall, read from the solver's `TurbulenceBoundary` (its wall
   sides, as its `conditions` reads them). 11.53 is the scalable form's floor `Y_STAR_FLOOR`. The
   share is also given per surface kind: domain walls, obstacle tops, obstacle sides.
4. **Where particles go**, only if both variants converge on 80x30 and on 200x75; otherwise
   skipped and said so. Method:
   - *The source.* An operator at working height in the gap above return 2: the cells whose
     centres lie in the 0.2 m square [2.6, 2.8] x [0.9, 1.1] m (closed), the rate spread
     uniformly by volume so the emission totals Q = 1.0e4 particles per second per metre of
     depth for each class. On 80x30 that is four 0.1 m cells, on 200x75 twenty-five 0.04 m
     cells, the same square. Continuous, from a zero field.
   - *The classes.* 0.5 and 5 micrometres, indices 2 and 4 of the configuration's five. Nothing
     comes in through the supply (it is HEPA filtered with no concentration key).
   - *The flow.* The converged run's face velocities, frozen, and its eddy viscosity (the
     solver's `turbulence_state.nu_t`, the under-relaxed field the momentum ran on) passed to
     `solve_timestep` as `eddy_viscosity`, with the configuration's `turbulent_schmidt` 0.7.
   - *The step.* The configuration's `cfl_number` 0.1; dt the smaller of the two classes'
     `stable_dt` on that run's faces, one dt for both classes. The scheme's fixed point does not
     depend on dt (the explicit advection and the implicit diffusion cancel it at a steady
     field), so the step sets the march's length, not its answer.
   - *The steady rule.* Over every window of 1.0 s of simulated time: the in-domain count's
     relative change, each sensor's change relative to the largest sensor reading, and the
     removal rate (the budget's outflow and deposited increments over the window) against Q.
     Steady when the first two are below 1e-4 and the removal is within 1e-3 of Q. Cap 600 s of
     simulated time; the record states the time reached and which stopped it.
   - *Reported per run and class.* The concentration at the four sensors, interpolated
     bilinearly from the cell centres with SOLID cells at zero (step 5's measurement 3). The
     deposition rate per surface at the steady field, `v_d A_f C_P` per depositing face (the
     rule `_book_deposition` applies), summed over named surfaces: each contiguous run of
     depositing floor faces between the returns and the obstacle footprints, named by its x
     interval; each obstacle top; each obstacle side; the left wall; the right wall; the
     ceiling outside the supply. Checked against the budget: the sum over faces against the
     deposited increment of the last window over its length. The five locations of largest
     deposition: every depositing surface cut into 0.2 m segments (the coarse grid's cell, the
     same bins on every grid; floors and tops along x, walls and sides along y), the rate per
     segment, the five largest. One picture per run of the concentration of each class.
   - *The comparative check.* The five hotspots and the sensor order (descending) of 80x30
     against 200x75, standard; and of standard against RNG on 200x75. Reported as the hotspots
     in common (the segments both top fives contain) and the two sensor orders, not as
     percentages of absolute values.

## 3. Predictions (orchestrator, 2026-10-09; committed before any run)

- (a) The standard variant converges on all three grids. The coupled field damps what the
  frozen fields could not.
- (b) RNG converges on 40x15 and 80x30. On 200x75: no prediction.
- (c) The core's median nu_t / nu at the baseline inlet lies between 10 and 100. The inlet alone
  gives about 16 (k = 1.5 (0.05 x 0.45)^2, eps = k^1.5 / 0.1); the equipment tops add
  production.
- (d) First-node y+ on 200x75 lies between about 5 and 100, below 11.53 near the stagnation
  points on the equipment tops and in slow corners (ADR-012 B's table).
- (e) `pressure_rtol` 1e-4 and 1e-8 stop at the same outer count on 200x75.
- (f) For 5 micrometres, the largest deposition sits on the floor nearest the source on both
  grids and both variants. For 0.5 micrometres the four sensors keep the same order between
  80x30 and 200x75.

### 3.1 What each outcome means (orchestrator)

- (a) fails on 200x75: the coupled field does not rescue the fine grid, and the convergence aid
  (step 0 section 6.6's ranking) goes to Alex before anything else.
- (f) fails between grids: 80x30 cannot be used for comparisons and 200x75 is not shown to be
  enough either; the product's comparative claim needs a finer grid or a different
  discretisation.
- (f) fails between variants only: the model choice decides the answer, and VAL-020 (the
  impinging jet) becomes the deciding validation.

### 3.2 Builder's predictions (2026-10-09, before any run)

Written beside the orchestrator's so they can be scored apart. (a) The standard variant
converges on 40x15 and 80x30; on 200x75 I expect it bounded, not converged: step 5's Z2, the
non-uniform field at the top of the range, stayed bounded there at ten sweeps, and the coupled
field's shape is Z2's, large in the shear layers and small in the core. (b) RNG follows the
standard variant grid for grid. (c) Holds, nearer 16 than 100: the core is the supply's air, and
the tops' production stays in the shear layers below it. (d) The largest y+ lies at the returns,
above 70; the share below 11.53 is between a fifth and a half of the wall nodes, most of them
on the equipment tops and the ceiling outside the supply. (e) Holds to within ten outer
iterations rather than to the iteration: the coupled solve adds a field that changes every
iteration, and step 5's bitwise agreement was of a frozen one. (f) The 5 micrometre hotspot is
the floor of the gap above return 2 on both grids and both variants; the 0.5 micrometre sensor
order has above_gap_1 first on both grids, and I expect the two sensors farthest from the source
(hood_entry and above_gap_2) to swap between grids.

## 4. Stop conditions (from the prompt)

- A positivity error: the iteration, the cell, and whether that run's pressure corrections hit
  their cap are reported, and the run is its own outcome in the table.
- Nothing converges on 200x75 with either variant: measurement 1's characterisation is finished,
  measurements 2 to 4 are skipped, the report is written, and the work stops.
- The whole run set would exceed about eight hours: the estimate and a reduced set are reported
  first.

## 5. Results (written after the runs)

### 5.1 Order, machine and cost

The timing probes ran first (20:58), three processes at once, 20 outer iterations each:

| Grid | Seconds per outer | CG per correction (mean) | k, eps sweeps (mean) | Share of wall: pressure, turbulence | Minutes at the 10,000 cap |
|---|---|---|---|---|---|
| 40x15 | 0.014 | 134 | 5.0, 3.0 | 0.45, 0.24 | 2 |
| 80x30 | 0.028 | 268 | 5.6, 5.1 | 0.59, 0.19 | 5 |
| 200x75 | 0.151 | 651 | 12.6, 9.2 | 0.73, 0.15 | 25 |

The projection of the full set at the cap was under an hour of wall time in parallel, so the
set ran as planned. The seven matrix rows and the two inlet rows were launched together at
20:59:09 (nine processes, one BLAS thread each); the three located reruns and the three
transport marches followed as rows ended, so at most eleven processes ran at once on twelve
cores. The sum of the nine flow rows' wall times is 59 minutes (58.9); the longest single row was RNG on
200x75 to the cap, 21.5 minutes. The wall times in the tables were taken beside other runs.

Three deviations from section 2, none of which changes a measurement. (1) The scripts were
committed in three commits after the predictions, not one. (2) On 200x75 the source square
covers 30 cells of 0.04 m (0.048 m^2), not the 25 section 2.2 states: x = 2.60 is a cell face
there and y = 0.90 a cell centre, so the closed square takes five columns and six rows. On
80x30 it is the four cells stated (0.040 m^2) and on 40x15 two cells (0.080 m^2). The emission
is the same on every grid; the records carry the coverage. (3) The transport march ran as a
supplement on the converged rows (section 5.5) after the prompt's condition for measurement 4
failed, and the segment key of a floor bin was changed to "floor" alone before any
comparison was scored, since a floor piece's name carries its x interval, which the staircase
moves between grids (the `rescore` mode of `transport46.py`; the march was not repeated).

### 5.2 Measurement 1: the matrix

Every flow row, classified by section 2.1's rule. The readings columns give the five
conditions at the stop, (a) and (e) the rule's estimates over their scales, (b) and (d) in kg/s
per metre of depth against `mass_imbalance_tol`, (c) over the flux scale; "holds from" is the
first outer iteration from which each condition holds to the end. Rows ending in `_loc` are the
located reruns, which reproduce their originals' stops, counts and face hashes bit for bit.

| Run | Class | Stop | Outer | Residual: least (at), end | Readings at stop (a), (b), (c), (d), (e) | Holds from (a), (b), (c), (d), (e) | Largest speed at end (m/s), cell | Peak speed | CG per correction: mean, largest | Cap hits | k, eps sweeps (mean) | Wall (s), per outer | Face hash |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| standard_40x15 | bounded (stalled) | max_simple_iter | 10,000 | 2.21e-07 (9,632), 3.64e-07 | -, 2.7e-10, 3.3e-09, 2.1e-15, 7 | -, 102, 94, 1, - | 1.58, (6.1, 0.9) | 1.58 | 117, 142 | 0 | 7.0, 6.0 | 192, 0.019 | 50c87381d66deb9e |
| standard_80x30 | bounded (neither) | max_simple_iter | 10,000 | 7.43e-07 (5,352), 9.32e-07 | 0.021, 6.9e-10, 3.9e-08, 2e-15, 40 | -, 187, 164, 1, - | 1.4, (2.75, 1.35) | 1.62 | 223, 287 | 0 | 23.8, 18.9 | 373, 0.037 | 6ab5e2c7d5a1fa0c |
| standard_200x75 | converged | error_estimate_and_continuity | 3,421 | 2.10e-12 (3,420), 2.10e-12 | 9.2e-08, 8.1e-14, 3.2e-12, 1.7e-14, 1e-06 | 2,826, 487, 435, 1, 3,421 | 1.53, (0.5, 0.02) | 1.81 | 455, 714 | 0 | 50.2, 37.8 | 576, 0.168 | 218c8d50888c26ab |
| rng_40x15 | converged | error_estimate_and_continuity | 694 | 8.15e-12 (694), 8.15e-12 | 1.1e-08, 1.1e-13, 1.4e-12, 2.2e-15, 9.9e-07 | 517, 127, 95, 1, 694 | 1.59, (6.1, 0.9) | 1.59 | 126, 145 | 0 | 6.2, 5.7 | 13, 0.019 | 64b4c4faae0c3449 |
| rng_80x30 | converged | error_estimate_and_continuity | 1,407 | 5.12e-12 (1,407), 5.12e-12 | 2.4e-08, 7.2e-14, 1.9e-12, 2e-15, 9.9e-07 | 1,156, 418, 351, 1, 1,407 | 1.42, (2.75, 1.35) | 1.72 | 226, 288 | 0 | 19.0, 16.9 | 53, 0.037 | 1802964a692cdd79 |
| rng_200x75 | bounded (stalled) | max_simple_iter | 10,000 | 3.60e-04 (750), 5.62e-04 | -, 1.1e-07, 1.8e-05, 1.7e-14, - | -, -, -, 1, - | 1.59, (0.5, 0.02) | 2.01 | 277, 718 | 0 | 45.5, 38.8 | 1291, 0.129 | 7ccc20b8b9e4379a |
| standard_40x15_loc | bounded (stalled) | max_simple_iter | 10,000 | 2.21e-07 (9,632), 3.64e-07 | -, 2.7e-10, 3.3e-09, 2.1e-15, 7 | -, 102, 94, 1, - | 1.58, (6.1, 0.9) | 1.58 | 117, 142 | 0 | 7.0, 6.0 | 190, 0.019 | 50c87381d66deb9e |
| standard_80x30_i0.02_l0.05 | bounded (neither) | max_simple_iter | 10,000 | 1.25e-06 (652), 3.12e-06 | 0.53, 3.7e-09, 9.3e-08, 2e-15, - | -, 238, 200, 1, - | 1.41, (2.75, 1.35) | 2.24 | 181, 302 | 0 | 22.8, 17.8 | 335, 0.034 | c78d4ce7c04d5e3b |
| standard_80x30_i0.1_l0.3 | converged | error_estimate_and_continuity | 743 | 1.11e-11 (743), 1.11e-11 | 1.8e-08, 1e-13, 2.5e-12, 2e-15, 9.7e-07 | 645, 178, 153, 1, 743 | 1.34, (2.75, 1.45) | 1.42 | 251, 280 | 0 | 22.1, 17.4 | 29, 0.039 | 23e7bf1818d6acc9 |
| standard_80x30_loc | bounded (neither) | max_simple_iter | 10,000 | 7.43e-07 (5,352), 9.32e-07 | 0.021, 6.9e-10, 3.9e-08, 2e-15, 40 | -, 187, 164, 1, - | 1.4, (2.75, 1.35) | 1.62 | 223, 287 | 0 | 23.8, 18.9 | 368, 0.037 | 6ab5e2c7d5a1fa0c |
| standard_200x75_r1e-8 | converged | error_estimate_and_continuity | 3,421 | 2.09e-12 (3,421), 2.09e-12 | 9.2e-08, 7.1e-14, 2.7e-12, 1.7e-14, 1e-06 | 2,826, 1, 1, 1, 3,421 | 1.53, (0.5, 0.02) | 1.81 | 581, 888 | 0 | 50.2, 37.8 | 670, 0.196 | 73537caa33389fa3 |
| rng_200x75_loc | bounded (stalled) | max_simple_iter | 10,000 | 3.60e-04 (750), 5.62e-04 | -, 1.1e-07, 1.8e-05, 1.7e-14, - | -, -, -, 1, - | 1.59, (0.5, 0.02) | 2.01 | 277, 718 | 0 | 45.5, 38.8 | 1240, 0.124 | 7ccc20b8b9e4379a |

The six baseline rows split by variant and grid in opposite directions. The standard variant
converges on 200x75 only; RNG converges on 40x15 and 80x30 only. No row diverged, grew or
raised a positivity error, and no pressure correction or k and eps solve reached its cap.

The bounded rows are of two kinds, and the bounded table below gives their tails (the last
2,000 outer iterations) with `bounded44.py`'s measures and, for the located reruns, the regions
of largest change. The standard rows on 40x15 and 80x30 sit in small cycles: their velocity
step is 1e-5 to 2e-4 of the velocity scale (the `velocity_step` reading against 0.45 m/s), the
per-cell imbalance is under `mass_imbalance_tol` throughout the tail (conditions (b), (c) and
(d) hold from the iterations the matrix gives), and only (a) and (e) refuse: the step neither
falls nor grows, so the fitted rate is not below one and the estimate is infinite. The cycles
are periodic in the residual (the period column), and the largest change sits in the shear
layer off the server rack's top corner above the gap (40x15), and at return 2's first cells and
the rack's top corner (80x30). The RNG row on 200x75 is a different kind: its velocity step is
about a tenth of the scale, its per-cell imbalance is above the tolerance by a factor of about
thirty, and its recurrence is weak at short lags and strongest at 75 iterations. Its largest
change sits in every tail iteration in the gap between the rack and the litho tool, beside
the litho tool's west face between 1.3 and 1.7 m up (the located row's cells), the shear
layer of the air turning down into return 2, and the largest nu_t change sits on the same
face a little lower.

| Run | Outer, tail | Residual: least, median, largest | Amplitude p95/p5 | Drift (log10 per 1,000) | Period (height); strongest (height) | Largest speed: least, largest; cells | Located: share by region (largest of u, v); cells | Largest nu_t change: share by region; cells; size (m^2/s) |
|---|---|---|---|---|---|---|---|---|
| rng_200x75 | 10,000, 2,000 | 4.57e-04, 5.30e-04, 5.63e-04 | 1.19 | 4.69e-05 | 9 (0.61); 75 (0.96) | 1.59, 1.59; (0.5, 0.02) | - | - |
| rng_200x75_loc | 10,000, 2,000 | 4.57e-04, 5.30e-04, 5.63e-04 | 1.19 | 4.69e-05 | 9 (0.61); 75 (0.96) | 1.59, 1.59; (0.5, 0.02) | gap between equipment 1.00; (2.94, 1.54), (2.9, 1.42), (2.98, 1.66) | gap between equipment 1.00; (2.98, 1.06), (2.98, 1.1), (2.98, 1.02); 7.3e-05 |
| standard_40x15 | 10,000, 2,000 | 2.21e-07, 3.11e-07, 4.00e-07 | 1.73 | -3.59e-05 | 21 (0.71); 42 (0.97) | 1.58, 1.58; (6.1, 0.9) | - | - |
| standard_40x15_loc | 10,000, 2,000 | 2.21e-07, 3.11e-07, 4.00e-07 | 1.73 | -3.59e-05 | 21 (0.71); 42 (0.97) | 1.58, 1.58; (6.1, 0.9) | core above equipment 0.63; gap between equipment 0.30; server_rack face 0.07; (2.7, 2.1), (2.7, 1.9), (2.5, 0.7) | gap between equipment 0.74; core above equipment 0.26; (2.7, 1.9), (2.7, 2.1), (2.7, 1.3); 2.3e-06 |
| standard_80x30 | 10,000, 2,000 | 7.46e-07, 1.32e-06, 1.96e-06 | 1.99 | -0.00203 | 28 (0.33); 342 (0.68) | 1.4, 1.4; (2.75, 1.35) | - | - |
| standard_80x30_i0.02_l0.05 | 10,000, 2,000 | 1.25e-06, 2.96e-06, 4.81e-06 | 3.3 | -0.00364 | 51 (0.97); 51 (0.97) | 1.41, 1.41; (2.75, 1.35) | - | - |
| standard_80x30_loc | 10,000, 2,000 | 7.46e-07, 1.32e-06, 1.96e-06 | 1.99 | -0.00203 | 28 (0.33); 342 (0.68) | 1.4, 1.4; (2.75, 1.35) | return 2 0.43; server_rack top 0.27; etch_chamber top 0.13; (2.35, 0.15), (2.35, 2.05), (4.95, 2.05) | server_rack face 0.44; server_rack top 0.44; etch_chamber top 0.10; (2.35, 2.05), (4.95, 2.05), (2.35, 1.85); 9.7e-06 |

### 5.3 Measurement 2: the inlet's turbulence

| Run | Intensity, length (m) | Inlet k (m^2/s^2), eps (m^2/s^3) | Inlet nu_t / nu | Class | Outer | Core nu_t / nu: median, 95th | y+: median, share below 11.53 | Largest speed, cell |
|---|---|---|---|---|---|---|---|---|
| standard_80x30 | 0.05, 0.1 | 0.000759, 0.000209 | 16.4 | bounded (neither) | 10,000 | 17.2, 58.8 | 111, 0.023 | 1.4, (2.75, 1.35) |
| standard_80x30_i0.02_l0.05 | 0.02, 0.05 | 0.000122, 2.68e-05 | 3.29 | bounded (neither) | 10,000 | 3.5, 50.1 | 81.7, 0.027 | 1.41, (2.75, 1.35) |
| standard_80x30_i0.1_l0.3 | 0.1, 0.3 | 0.00304, 0.000558 | 98.7 | converged | 743 | 103, 281 | 209, 0.023 | 1.34, (2.75, 1.45) |
| standard_80x30_loc | 0.05, 0.1 | 0.000759, 0.000209 | 16.4 | bounded (neither) | 10,000 | 17.2, 58.8 | 111, 0.023 | 1.4, (2.75, 1.35) |

The three inlet settings on 80x30 with the standard variant: the strongest converges, the
baseline and the weakest reach the cap. The core's nu_t / nu follows the inlet's value to within
a few percent at the median in every row, and the 95th percentile sits two to fifteen times
above it where the shear layers reach into the core.

### 5.4 Measurement 3: what the converged room looks like

| Run | Outer, stop | Readings at stop (a), (b), (c), (d), (e) | Wall (s) | Inlet nu_t / nu | Core nu_t / nu: median, 95th | Core nu_t / nu: 5th, 25th, 75th | y+ nodes | y+: median, least, largest | Share below 11.53: all; domain, tops, sides | Figure |
|---|---|---|---|---|---|---|---|---|---|---|
| standard_200x75 | 3,421, error_estimate_and_continuity | 9.2e-08, 8.1e-14, 3.2e-12, 1.7e-14, 1e-06 | 576 | 16.4 | 17, 87.6 | 16.2, 16.5, 20.8 | 641 | 65.4, 2.83, 271 | 0.045; 0.13, 0, 0.012 | ecr002_step6_standard_200x75.png |
| rng_40x15 | 694, error_estimate_and_continuity | 1.1e-08, 1.1e-13, 1.4e-12, 2.2e-15, 9.9e-07 | 13 | 15.4 | 16.1, 41.1 | 13.6, 15, 17.7 | 126 | 115, 21.9, 372 | 0; 0, 0, 0 | ecr002_step6_rng_40x15.png |
| rng_80x30 | 1,407, error_estimate_and_continuity | 2.4e-08, 7.2e-14, 1.9e-12, 2e-15, 9.9e-07 | 53 | 15.4 | 15.7, 42.4 | 13.6, 15.1, 17.2 | 257 | 81.3, 6.86, 392 | 0.023; 0.062, 0, 0.0072 | ecr002_step6_rng_80x30.png |
| standard_80x30_i0.1_l0.3 | 743, error_estimate_and_continuity | 1.8e-08, 1e-13, 2.5e-12, 2e-15, 9.7e-07 | 29 | 98.7 | 103, 281 | 66.5, 99, 123 | 257 | 209, 6.95, 555 | 0.023; 0.062, 0, 0.0072 | ecr002_step6_standard_80x30_i0.1_l0.3.png |
| standard_200x75_r1e-8 | 3,421, error_estimate_and_continuity | 9.2e-08, 7.1e-14, 2.7e-12, 1.7e-14, 1e-06 | 670 | 16.4 | 17, 87.6 | 16.2, 16.5, 20.8 | 641 | 65.4, 2.83, 271 | 0.045; 0.13, 0, 0.012 | ecr002_step6_standard_200x75_r1e-8.png |

The figures named in the last column are beside this report, one per converged row: the
streamlines coloured by speed, nu_t / nu on a log scale and k. The check row at
`pressure_rtol` 1e-8 against the 1e-4 row on 200x75:

| Rows (1e-4, 1e-8) | Outer | Stop | CG per correction (mean) | Wall (s) | Faces: max |du|, max |dv| (m/s) | Cells: max speed difference (m/s) | nu_t: max |difference| (m^2/s), max relative | k: max relative difference |
|---|---|---|---|---|---|---|---|---|
| standard_200x75, standard_200x75_r1e-8 | 3,421, 3,421 | error_estimate_and_continuity, error_estimate_and_continuity | 455, 581 | 576, 670 | 1.05e-11, 9.78e-12 | 1.02e-11 | 6.66e-14, 1.32e-10 | 2.50e-10 |

### 5.5 Measurement 4: where particles go

**Skipped by the prompt's condition.** Measurement 4 was to run only if both variants converge
on 80x30 and on 200x75. Neither grid has both: the standard variant did not converge on 80x30
and RNG did not converge on 200x75 (section 5.2). The two comparisons the prompt names, 80x30
against 200x75 with the standard variant and standard against RNG on 200x75, therefore have no
pair of converged records and are not scored; the comparative table below marks them skipped.

**The supplementary marches.** So that the instrument and the source's behaviour are known
before the next prompt, the march of section 2.2 was run on the three converged baseline rows
(RNG on 40x15 and 80x30, standard on 200x75). The tables below are what the method reports;
the two supplementary pairs are the ones those rows allow, a grid pair within RNG and a cross
pair between RNG on 80x30 and standard on 200x75, and neither is a pair the prompt asked for.

| Run, class (um) | Stop, t (s) | Last window: total, sensor change; removal / Q | Deposition / Q (faces; budget), outflow / Q | Budget residual | Sensors: near_door, above_gap_1, above_gap_2, hood_entry (per m^3) | Sensor order |
|---|---|---|---|---|---|---|
| rng_40x15, 0.5 | steady, 61 | 0, 9.8e-05; 1 | 2.817e-08 (2.817e-08), 1 | 2e-09 | 1.503e-16, 6.787e-18, 8.214e-27, 1.58e-24 | near_door > above_gap_1 > hood_entry > above_gap_2 |
| rng_40x15, 5 | steady, 61 | 0, 9.4e-05; 1 | 2.196e-07 (2.196e-07), 1 | 2e-09 | 1.466e-16, 6.725e-18, 7.881e-27, 1.504e-24 | near_door > above_gap_1 > hood_entry > above_gap_2 |
| rng_80x30, 0.5 | steady, 71 | 1.2e-05, 9e-05; 0.99999 | 2.13e-06 (2.13e-06), 1 | 5.9e-08 | 9.701e-20, 2.598e-12, 6.974e-32, 6.951e-35 | above_gap_1 > near_door > above_gap_2 > hood_entry |
| rng_80x30, 5 | steady, 71 | 1.2e-05, 9e-05; 0.99999 | 0.0001567 (0.0001567), 0.9998 | 5.9e-08 | 9.052e-20, 2.518e-12, 6.397e-32, 6.258e-35 | above_gap_1 > near_door > above_gap_2 > hood_entry |
| standard_200x75, 0.5 | steady, 47 | 2.9e-05, 9.7e-05; 1 | 3.284e-06 (3.284e-06), 1 | 1.3e-06 | 1.899e-33, 2.817e-27, 4.658e-56, 1.034e-57 | above_gap_1 > near_door > above_gap_2 > hood_entry |
| standard_200x75, 5 | steady, 47 | 2.8e-07, 9.6e-05; 1 | 0.0002456 (0.0002456), 0.9998 | 1.3e-06 | 1.719e-33, 2.622e-27, 4.025e-56, 8.707e-58 | above_gap_1 > near_door > above_gap_2 > hood_entry |

Deposition per surface, share of the deposition:
| Surface | rng_40x15, 0.5 um | rng_40x15, 5 um | rng_80x30, 0.5 um | rng_80x30, 5 um | standard_200x75, 0.5 um | standard_200x75, 5 um |
|---|---|---|---|---|---|---|
| server_rack east | 0.898 | 0.00888 | 3.16e-07 | 3.31e-10 | 1.81e-09 | 1.86e-12 |
| floor 3.00-3.20 | 0.101 | 0.991 | 0.962 | 1 | - | - |
| litho_tool west | 0.00126 | 1.24e-05 | 0.0375 | 3.91e-05 | 0.0221 | 2.27e-05 |
| server_rack top | 6.45e-16 | 6.29e-15 | 1.81e-21 | 1.8e-21 | 2.16e-35 | 2.08e-35 |
| litho_tool top | 2.8e-18 | 2.72e-17 | 1.91e-07 | 1.95e-07 | 2.58e-07 | 2.6e-07 |
| floor 1.20-1.40 | 9.38e-19 | 9.02e-18 | - | - | - | - |
| server_rack west | 6.84e-20 | 6.62e-22 | 6.56e-25 | 6.43e-28 | 6.7e-39 | 6.28e-42 |
| floor 0.00-0.40 | 9.81e-22 | 9.29e-21 | - | - | - | - |
| left wall | 4.49e-24 | 4.25e-26 | 1.5e-35 | 1.42e-38 | 1.2e-55 | 1.08e-58 |
| etch_chamber top | 9.63e-25 | 9.09e-24 | 1.08e-34 | 1.05e-34 | 1.69e-52 | 1.54e-52 |
| litho_tool east | 4.79e-25 | 4.57e-27 | 5.25e-26 | 5.16e-29 | 1.02e-36 | 9.5e-40 |
| etch_chamber west | 2.78e-25 | 2.65e-27 | 1.55e-28 | 1.51e-31 | 6.03e-41 | 5.59e-44 |
| etch_chamber east | 3.5e-27 | 3.31e-29 | 5.46e-38 | 5.2e-41 | 4.4e-57 | 3.88e-60 |
| hood_bench west | 5.23e-29 | 4.93e-31 | 3.61e-43 | 3.42e-46 | 9.6e-65 | 8.36e-68 |
| hood_bench top | 1.33e-29 | 1.23e-28 | 7.08e-46 | 6.66e-46 | 1.67e-69 | 1.41e-69 |
| ceiling | 8.41e-32 | 7.66e-34 | 7.31e-47 | 6.51e-50 | 4.83e-83 | 3.82e-86 |
| floor 7.60-8.00 | 1.11e-34 | 1.3e-33 | 9.61e-54 | 1.17e-53 | 8.62e-82 | 8.08e-82 |
| hood_bench east | 3.04e-35 | 2.81e-37 | 2.04e-54 | 1.96e-57 | 9.02e-83 | 7.56e-86 |
| right wall | 2.15e-35 | 1.99e-37 | 3.32e-54 | 3.13e-57 | 1.08e-82 | 8.75e-86 |
| floor 2.30-2.40 | - | - | 1.54e-05 | 1.6e-05 | - | - |
| floor 1.20-1.50 | - | - | 1.5e-23 | 1.46e-23 | - | - |
| floor 4.50-4.60 | - | - | 3.79e-25 | 3.7e-25 | - | - |
| floor 4.90-5.00 | - | - | 2.93e-27 | 2.85e-27 | - | - |
| floor 0.00-0.50 | - | - | 7.09e-31 | 6.86e-31 | - | - |
| floor 5.60-5.70 | - | - | 3.84e-37 | 3.63e-37 | - | - |
| floor 6.30-6.40 | - | - | 9.56e-42 | 8.99e-42 | - | - |
| floor 2.96-3.20 | - | - | - | - | 0.978 | 1 |
| floor 2.32-2.40 | - | - | - | - | 2.04e-07 | 2.08e-07 |
| floor 4.52-4.60 | - | - | - | - | 4.74e-36 | 4.4e-36 |
| floor 1.24-1.48 | - | - | - | - | 1.15e-37 | 1.07e-37 |
| floor 4.92-5.00 | - | - | - | - | 2.42e-39 | 2.23e-39 |
| floor 0.00-0.48 | - | - | - | - | 1.09e-49 | 1e-49 |
| floor 5.60-5.72 | - | - | - | - | 3.08e-56 | 2.7e-56 |
| floor 6.32-6.40 | - | - | - | - | 4.08e-63 | 3.54e-63 |

Deposition per surface, rate (per s per m depth):
| Surface | rng_40x15, 0.5 um | rng_40x15, 5 um | rng_80x30, 0.5 um | rng_80x30, 5 um | standard_200x75, 0.5 um | standard_200x75, 5 um |
|---|---|---|---|---|---|---|
| server_rack east | 0.000253 | 1.95e-05 | 6.73e-09 | 5.19e-10 | 5.94e-11 | 4.56e-12 |
| floor 3.00-3.20 | 2.85e-05 | 0.00218 | 0.0205 | 1.57 | - | - |
| litho_tool west | 3.55e-07 | 2.73e-08 | 0.000799 | 6.13e-05 | 0.000726 | 5.57e-05 |
| server_rack top | 1.82e-19 | 1.38e-17 | 3.85e-23 | 2.82e-21 | 7.09e-37 | 5.11e-35 |
| litho_tool top | 7.89e-22 | 5.98e-20 | 4.07e-09 | 3.06e-07 | 8.48e-09 | 6.38e-07 |
| floor 1.20-1.40 | 2.64e-22 | 1.98e-20 | - | - | - | - |
| server_rack west | 1.93e-23 | 1.45e-24 | 1.4e-26 | 1.01e-27 | 2.2e-40 | 1.54e-41 |
| floor 0.00-0.40 | 2.76e-25 | 2.04e-23 | - | - | - | - |
| left wall | 1.26e-27 | 9.33e-29 | 3.19e-37 | 2.23e-38 | 3.95e-57 | 2.65e-58 |
| etch_chamber top | 2.71e-28 | 2e-26 | 2.31e-36 | 1.64e-34 | 5.56e-54 | 3.78e-52 |
| litho_tool east | 1.35e-28 | 1e-29 | 1.12e-27 | 8.09e-29 | 3.34e-38 | 2.33e-39 |
| etch_chamber west | 7.83e-29 | 5.83e-30 | 3.3e-30 | 2.37e-31 | 1.98e-42 | 1.37e-43 |
| etch_chamber east | 9.86e-31 | 7.26e-32 | 1.16e-39 | 8.15e-41 | 1.44e-58 | 9.53e-60 |
| hood_bench west | 1.47e-32 | 1.08e-33 | 7.7e-45 | 5.35e-46 | 3.15e-66 | 2.05e-67 |
| hood_bench top | 3.76e-33 | 2.69e-31 | 1.51e-47 | 1.04e-45 | 5.47e-71 | 3.47e-69 |
| ceiling | 2.37e-35 | 1.68e-36 | 1.56e-48 | 1.02e-49 | 1.59e-84 | 9.39e-86 |
| floor 7.60-8.00 | 3.12e-38 | 2.85e-36 | 2.05e-55 | 1.83e-53 | 2.83e-83 | 1.98e-81 |
| hood_bench east | 8.58e-39 | 6.17e-40 | 4.34e-56 | 3.07e-57 | 2.96e-84 | 1.86e-85 |
| right wall | 6.05e-39 | 4.36e-40 | 7.08e-56 | 4.9e-57 | 3.54e-84 | 2.15e-85 |
| floor 2.30-2.40 | - | - | 3.28e-07 | 2.51e-05 | - | - |
| floor 1.20-1.50 | - | - | 3.19e-25 | 2.28e-23 | - | - |
| floor 4.50-4.60 | - | - | 8.07e-27 | 5.8e-25 | - | - |
| floor 4.90-5.00 | - | - | 6.25e-29 | 4.46e-27 | - | - |
| floor 0.00-0.50 | - | - | 1.51e-32 | 1.07e-30 | - | - |
| floor 5.60-5.70 | - | - | 8.17e-39 | 5.68e-37 | - | - |
| floor 6.30-6.40 | - | - | 2.04e-43 | 1.41e-41 | - | - |
| floor 2.96-3.20 | - | - | - | - | 0.0321 | 2.46 |
| floor 2.32-2.40 | - | - | - | - | 6.7e-09 | 5.12e-07 |
| floor 4.52-4.60 | - | - | - | - | 1.56e-37 | 1.08e-35 |
| floor 1.24-1.48 | - | - | - | - | 3.79e-39 | 2.63e-37 |
| floor 4.92-5.00 | - | - | - | - | 7.96e-41 | 5.49e-39 |
| floor 0.00-0.48 | - | - | - | - | 3.57e-51 | 2.46e-49 |
| floor 5.60-5.72 | - | - | - | - | 1.01e-57 | 6.63e-56 |
| floor 6.32-6.40 | - | - | - | - | 1.34e-64 | 8.69e-63 |

The five segments of largest deposition (surface | bin start), rate per s per m depth:
| Run, class (um) | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| rng_40x15, 0.5 | server_rack east | y 0.0 (0.00011) | server_rack east | y 0.2 (5.51e-05) | server_rack east | y 0.4 (4.58e-05) | floor | x 3.0 (2.85e-05) | server_rack east | y 0.6 (2.53e-05) |
| rng_40x15, 5 | floor | x 3.0 (0.00218) | server_rack east | y 0.0 (8.47e-06) | server_rack east | y 0.2 (4.25e-06) | server_rack east | y 0.4 (3.53e-06) | server_rack east | y 0.6 (1.95e-06) |
| rng_80x30, 0.5 | floor | x 3.0 (0.0205) | litho_tool west | y 0.0 (0.000125) | litho_tool west | y 0.2 (0.000114) | litho_tool west | y 0.4 (9.94e-05) | litho_tool west | y 0.6 (8.77e-05) |
| rng_80x30, 5 | floor | x 3.0 (1.57) | floor | x 2.2 (2.51e-05) | litho_tool west | y 0.0 (9.57e-06) | litho_tool west | y 0.2 (8.73e-06) | litho_tool west | y 0.4 (7.63e-06) |
| standard_200x75, 0.5 | floor | x 3.0 (0.0265) | floor | x 2.8 (0.00561) | litho_tool west | y 0.0 (0.000162) | litho_tool west | y 0.2 (0.000146) | litho_tool west | y 0.4 (0.000115) |
| standard_200x75, 5 | floor | x 3.0 (2.03) | floor | x 2.8 (0.429) | litho_tool west | y 0.0 (1.24e-05) | litho_tool west | y 0.2 (1.12e-05) | litho_tool west | y 0.4 (8.83e-06) |

| Pair, class (um) | Hotspots in common (of 5) | In common | Only in the first | Only in the second | Sensor order, first | Sensor order, second | Orders agree |
|---|---|---|---|---|---|---|---|
| grids, standard | skipped: a record is missing |  |  |  |  |  |  |
| variants, 200x75 | skipped: a record is missing |  |  |  |  |  |  |
| supplementary: grids, rng, 0.5 | 1 | floor | x 3.0 | server_rack east | y 0.0; server_rack east | y 0.2; server_rack east | y 0.4; server_rack east | y 0.6 | litho_tool west | y 0.0; litho_tool west | y 0.2; litho_tool west | y 0.4; litho_tool west | y 0.6 | near_door > above_gap_1 > hood_entry > above_gap_2 | above_gap_1 > near_door > above_gap_2 > hood_entry | no |
| supplementary: grids, rng, 5 | 1 | floor | x 3.0 | server_rack east | y 0.0; server_rack east | y 0.2; server_rack east | y 0.4; server_rack east | y 0.6 | floor | x 2.2; litho_tool west | y 0.0; litho_tool west | y 0.2; litho_tool west | y 0.4 | near_door > above_gap_1 > hood_entry > above_gap_2 | above_gap_1 > near_door > above_gap_2 > hood_entry | no |
| supplementary: rng 80x30 against standard 200x75, 0.5 | 4 | floor | x 3.0; litho_tool west | y 0.0; litho_tool west | y 0.2; litho_tool west | y 0.4 | litho_tool west | y 0.6 | floor | x 2.8 | above_gap_1 > near_door > above_gap_2 > hood_entry | above_gap_1 > near_door > above_gap_2 > hood_entry | yes |
| supplementary: rng 80x30 against standard 200x75, 5 | 4 | floor | x 3.0; litho_tool west | y 0.0; litho_tool west | y 0.2; litho_tool west | y 0.4 | floor | x 2.2 | floor | x 2.8 | above_gap_1 > near_door > above_gap_2 > hood_entry | above_gap_1 > near_door > above_gap_2 > hood_entry | yes |

What the tables show. The source sits in return 2's capture zone: every march reaches its
steady rule within about a minute of simulated time, the removal rate equals the emission to
the rule's tolerance, and the outflow carries all but about 2e-4 of it for 5 micrometres and about 1.5e-5 for
0.5 micrometres on the finer grids (on 200x75 the 0.5 micrometre outflow still exceeds the
emission by 3e-5 at the stop, the plume draining; `tables_transport.json`). The deposition that does occur lies on the gap's
floor beside return 2 and on the two faces that bound the gap; on 40x15 the rack's east face
takes most of the 0.5 micrometre deposition, on the finer grids the litho tool's west face
and the floor piece east of the return. The four sensors read nothing: their largest value is
1e-12 per cubic metre on 80x30 and 1e-27 on 200x75 against a source-cell concentration of
1e4 to 1e5, so the sensor orders in the tables are the scheme's tails, not transport, and
their agreement or disagreement between rows means nothing. The concentration pictures
(`ecr002_step6_<run>_concentration.png`) show the plume held in the gap: it rises from the source along the litho tool's west face to
about the equipment tops' height (y about 2 m) in the small upward eddy there, and the gap's
downflow returns it to return 2 (test 46 found the earlier "confined below the source" contradicted
by the saved fields).

## 6. Predictions against the measurement

| Prediction | Measured |
|---|---|
| (a) The standard variant converges on all three grids | **Fails.** It converges on 200x75 (3,421) and reaches the cap on 40x15 and 80x30, bounded in small cycles (section 5.2) |
| (b) RNG converges on 40x15 and 80x30; 200x75 no prediction | **Holds** on the two grids predicted (694 and 1,407). On 200x75 it is bounded at the cap |
| (c) The core's median nu_t / nu at the baseline inlet between 10 and 100, about 16 from the inlet alone | **Holds.** The converged table gives 16 to 17 on every converged baseline row, the inlet's value; the tops' production shows in the 95th percentile, not the median |
| (d) First-node y+ on 200x75 between about 5 and 100, below 11.53 near the tops' stagnation points and in slow corners | **Fails** in range and in place. The median is 65 and the range 2.8 to 271, wider than predicted at both ends; the share below 11.53 is 4.5% of the nodes, all on the domain walls and the obstacle sides, none on the equipment tops (the by-kind shares in the converged table) |
| (e) `pressure_rtol` 1e-4 and 1e-8 stop at the same outer count on 200x75 | **Holds** to the iteration, 3,421, with the faces within 1.1e-11 m/s and nu_t within 1.3e-10 relative (the rtol table) |
| (f) 5 micrometres: the largest deposition on the floor nearest the source on both grids and both variants; 0.5 micrometres: the sensors keep their order between 80x30 and 200x75 | **Not scored as asked** (section 5.5). On the rows that converged the 5 micrometre hotspot is the gap's floor beside return 2 on every grid, and the 0.5 micrometre sensor order agrees between RNG 80x30 and standard 200x75, but the sensors read the scheme's tails, so the agreement carries no meaning |

Builder's predictions (section 3.2): (a) failed on all three grids: I predicted the standard
variant converged on 40x15 and 80x30 and bounded on 200x75, and it is the reverse (corrected
after test 46, which found this scored as held); (b) failed, RNG does not follow the standard
variant grid for grid but mirrors it; (c) held, nearer 16; (d) the largest y+ is at return 1's
corner, not above 70 only but 271, and the share below the floor is 4.5%, below the fifth I
gave, and on the walls and sides rather than the tops and ceiling; (e) held to the iteration,
better than the ten I allowed; (f) the 5 micrometre hotspot held, the sensor swap was not
measurable.

## 7. What this implies

Stated as the prompt asks, without deciding.

**The coupled field converges the fine grid, with the standard variant.** Section 3.1's first
outcome (the aid before anything else) does not apply as written: on 200x75, the grid the plan
scores VAL-018 on, the standard variant converges by the rule in 3,421 outer iterations and
about ten minutes, where step 5's frozen Z2 field stayed bounded. The question the coarse grids
raise is a different one. The standard rows on 40x15 and 80x30 are converged in every reading
but the rule's two estimates: their velocity step is 1e-5 to 1e-4 of the scale, steady, and
continuity holds to the tolerance. They are small limit cycles of the kind ADR-012 C's note
found at `cfl_number` 0.5 on the Couette channel, here at 0.25, located at the rack's top
corner and return 2's first cells. Whether the rule should stop on such a row (a bound on the
step in m/s beside the estimate, as the velocity-step rule once was), whether `cfl_number`
below 0.25 or a different `alpha_turbulence` removes the cycle, or whether the cycle is left
as a coarse-grid finding since the product grid converges, is a question for Alex.

**RNG on 200x75 is not that case.** Its velocity step is a tenth of the scale and its per-cell
imbalance thirty times the tolerance, a real oscillation of the flow the ten-sweep iteration
circles. Section 3.1's first outcome applies to RNG on the product grid: the aid of step 0's
ranking (continuation in viscosity, pseudo-transient continuation), or the decision that
VAL-018 runs the standard variant and RNG stays a coarse-grid comparison until VAL-020 tells
them apart.

**The inlet's turbulence decides convergence on 80x30.** The strongest setting converges the
standard variant where the baseline and the weakest do not, and the core's eddy viscosity is
the inlet's at every setting. The supply's intensity and dissipation length are design inputs
with no measurement behind them (ADR-012 C); whether the product configuration's values should
be chosen for the room's physics or for the iteration's convergence is a question the
sensitivity pair of step 8 was meant to inform, and this measurement says the two cannot be
separated on 80x30.

**The product's question cannot be asked at this source.** The operator in the gap above
return 2 emits into the return's capture zone; everything leaves through the floor, and the
sensors read the scheme's tails. The comparative criterion of VAL-018 needs a source whose
plume reaches the sensors, or sensors placed where a plume goes: on the tops, in the gap, at
the returns. Whether the source moves (to the aisle before the door, or above an equipment
top), whether the sensors move, or whether the criterion is scored on deposition alone, is a
question for Alex before step 8. The deposition instrument itself is ready: the per-surface
rates sum to the budget's to the march's tolerance, and the 0.2 m bins compare across grids
once keyed by surface kind.

**What the check row settles.** `pressure_rtol` 1e-4 reproduces 1e-8's count to the iteration
under the coupled solve, with the faces and the eddy viscosity equal to rounding, so step 5's
finding holds with the field changing every iteration; the 1e-4 row costs 22% fewer CG
iterations per correction. ADR-013 decision 3's amendment stands on this row.

## 8. What this does not settle

- Whether the standard rows' cycles on 40x15 and 80x30 go away at a smaller `cfl_number` or
  another `alpha_turbulence`: one setting was run.
- Whether RNG's oscillation on 200x75 is the iteration's or the flow's: located at the litho
  tool's west face, not probed.
- Where particles go once the source is outside a return's capture zone: the supplementary
  marches say only that this source is inside one.
- The strain overshoot beside the walls (ADR-012 C's note) in the room: k on the equipment
  tops is not compared with anything.
- The transport march's Courant number: the configuration's 0.1 was used; the fixed point
  does not depend on it, and a larger value would only shorten the march.
