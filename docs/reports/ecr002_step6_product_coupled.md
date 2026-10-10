# ECR-002 Step 6: The Product Room Under the Coupled Model

**Date:** 2026-10-09
**Tree:** branch `docs/ecr002-coupled-product` from main at 3bd8146. Nothing under `src/`,
`validation/`, `configs/` or `tests/` changes: every run goes through the committed
`StaggeredSolver` with the configuration's turbulence section, and the committed
`TransportSolver` with the converged flow handed to it, the product configuration overridden in
the probe as step 5's was.
**Instruments:** `docs/reports/probe46/` (committed): `coupled46.py` (the room, the runner, the
timing probe), `transport46.py` (measurement 4), `tables46.py` (the tables), `figures46.py` (the
figures), `bounded46.py` (step 5's `bounded44.py` pointed at this step's records) and the launcher
`run46.sh`. Raw output under `results/builder46/` (untracked). The figures are the small PNGs
beside this report.
**Order:** sections 1 to 4 were committed before any run (the first commit of the branch, this
file alone). The scripts followed in a second commit. Sections 5 onward were written after the
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
