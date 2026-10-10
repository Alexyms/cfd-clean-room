# ECR-002 Step 6: Two Converged Grids and the Comparative Check

**Date:** 2026-10-10
**Tree:** branch `docs/ecr002-comparative-check` from main at 714d6de. Nothing under `src/`,
`validation/`, `configs/` or `tests/` changes. Every flow run goes through the committed
`StaggeredSolver` with the configuration's turbulence section, and every particle march through
the committed `TransportSolver` with the converged flow handed to it. The three counterfactuals
of measurement 1 are probe-side subclasses or configuration keys, named in section 2.3.
**Instruments:** `docs/reports/probe47/` (committed after this section, before any run):
`coupled47.py` (the room on the new grids, the runner, the counterfactuals, the limiter
diagnostic), `transport47.py` (the sources, the discrimination check, the march),
`tables47.py` (the tables), `figures47.py` (the figures), `rooms47.py` (which room positions
each grid represents exactly) and the launcher `run47.sh`. Raw output under
`results/builder47/` (untracked). The figures are the PNGs beside this report.
**Order:** sections 1 to 4 were committed before any run, this file alone. The scripts followed
in later commits. Section 5 onward was written after the runs. Each run log's first line carries
its start time.
**Policy:** `docs/REVIEW_POLICY.md`, "What blocks". The numbers live in the tables
`tables47.py` generates from the records; the sentences point at the tables.

## 1. The question

Can the product's comparative claim be tested, and does it hold? The claim (`docs/SYSTEM.md`
section 1; ECR-002 criterion 10's note) is that the deposition hotspots and the sensor ranking
are stable under grid refinement and under the choice of k-epsilon variant. Testing it needs one
model converged on two grids that represent the same room, and particle sources whose plumes
reach the room. Prompt 46 had neither: the standard variant converged only on 200x75, RNG only
on the coarse grids, and the one source sat over a return that captured everything
(`docs/reports/ecr002_step6_product_coupled.md`, sections 5.2 and 5.5).

The orchestrator found that every position in the product room (the supply's and the returns'
ends, the door's top, the hood's ends, every obstacle edge and height) is a multiple of 0.05 m,
so 160x60 (5 cm cells) and 320x120 (2.5 cm cells) represent the room exactly, where 200x75 and
80x30 round some positions to whole faces and every grid comparison so far mixed resolution with
slightly different rooms. `rooms47.py` checks that claim per grid (section 5.1). This
measurement uses the exact pair.

Five measurements, in the prompt's order:

1. The coarse-grid cycle on 80x30 (standard variant), diagnosed by three counterfactuals.
2. The exact grid pair: standard and RNG on 160x60 and 320x120.
3. Where particles go, from three sources, on every converged run of measurement 2.
4. The figures, with the room's boundaries marked.
5. The comparative check between grids and between variants.

## 2. Method (fixed before the runs)

### 2.1 The room and the settings

Prompt 46's (its report, section 2.1), on the grids named below: `configs/clean_room_default.yaml`
regridded; `turbulence` with `model: k_epsilon`, `variant` standard or rng, `wall_treatment:
scalable_wall_functions`, `cfl_number` 0.25, `alpha_turbulence` 0.7, `max_iter` 500, `tol` 1e-10;
the baseline inlet (`turbulence_intensity` 0.05, `dissipation_length` 0.1 m); `solver` with
`stopping_rule: error_estimate` at step 5's tolerances (`iteration_error_tol` 1e-6 and
`mass_imbalance_tol = 1e-4 rho V_min / t_end`, ADR-011 G's formula), `momentum_sweeps` 10,
`pressure_rtol` 1e-4, `max_simple_iter` 10,000, `alpha_velocity` 0.5; everything else as the
file states it. From rest in every run. Each record carries its full configuration mapping, the
commit, the start time, the BLAS thread setting (one thread) and the wall time, and per outer
iteration what prompt 46's records carried: the residual, the CG count, the largest speed and
its cell, the velocity and eddy-viscosity steps, the rule's two estimates and the three
continuity readings (the recording subclass of the rule, `coupled46.RecordingRule`, the rule's
decision unchanged), and the cell of the largest change of u, v and nu_t (`--locate` on every
row; prompt 46's located reruns reproduced their originals bit for bit, so the wrapper is not a
change to the solve).

**Classification.** Step 5's rule in its order (`docs/reports/ecr002_step5_convergence.md`,
section 2.1, with round 2's window amendment): diverged (a speed past 100 m/s or a non-finite
field); converged (the solver's own stop, `error_estimate_and_continuity`); growing (at the cap
with the median largest speed over the last 500 iterations above 5 m/s); bounded and not
converged (at the cap under 5 m/s), with step 0's sub-classes falling, stalled or neither. A
`PositivityError` is its own outcome, reported with the iteration, the cell and the run's
pressure cap hits.

**Machine and threads.** The machine of prompt 46 (AMD Ryzen AI 9 HX 370, 12 cores, 64 GB,
Windows 11, Python 3.13, NumPy 2.4, OpenBLAS 0.3.31), `OPENBLAS_NUM_THREADS=1` in every process,
the flow runs in parallel processes, so wall times are not like for like with single-process
figures; section 5.1 says how many ran at once.

**Cost before the set.** A timing probe of 20 outer iterations on 160x60 and 320x120 (standard)
runs first. If it projects the set past about eight hours at the cap, the estimate and a reduced
set are reported before anything else runs.

### 2.2 The grids

| Grid | Cell (m) | Cells | Use |
|---|---|---|---|
| 80x30 | 0.10 | 2,400 | measurement 1, the cycle prompt 46 found |
| 160x60 | 0.05 | 9,600 | the exact pair, coarse |
| 320x120 | 0.025 | 38,400 | the exact pair, fine |

`rooms47.py` lists, for 80x30, 160x60, 200x75 and 320x120, every x and y position the
configuration states (the boundary segments' ends, the obstacles' edges and heights, the
sensors) and whether it lies on a cell face; the count of positions a grid rounds is the
check of section 1's claim.

### 2.3 Measurement 1: the coarse-grid cycle, diagnosed

80x30, standard variant, one run each, through probe-side substitutions on the committed
solver, nothing under `src/` changed:

- **control**: the baseline row again, expected to reproduce prompt 46's `standard_80x30` bit
  for bit (face hash `6ab5e2c7d5a1fa0c`, 10,000 outer iterations, bounded).
- **upwind**: k and eps advected by plain upwind in place of the UMIST-limited QUICK face value.
  A subclass of `KEpsilonModel` whose `_advect` passes `upwind=True` to the shared
  `advective_flux`; the momentum scheme, the production, the diffusion and the wall functions
  are unchanged.
- **corner**: momentum's corner rule removed, `tail43.corner_free_class()`'s predictor
  substituted for the solver's (the two lines that zero the QUICK correction at obstacle faces
  taken out of `_deferred_correction`; ADR-012 D's note of 2026-10-08 on the step 4 corner rule).
  The k and eps step is unchanged.
- **alpha**: `alpha_turbulence` 0.35 in place of 0.7, a configuration key.

Each is classified by step 5's rule. Where one converges, the change it made names the
mechanism the cycle lives in. Where none does, the control's `--locate` region
(`bounded44.py where`) and `bounded44.py history`'s recurrence are reported for the baseline
row and the diagnosis stops there.

**The limiter diagnostic.** Prediction (a) names a mechanism: the limiter switching branch near
the rack's corner, iteration to iteration. One more run of the baseline row (**limiter**) records
it directly. `scalar_scheme.limited_face_values` is wrapped, in that process only, to keep for
every face the branch of `psi = max(0, min(2r, (1 + 3r) / 4, psi_quick, 2))` that set the face
value (the clamp at zero, one of the four terms, or the `C_C` fallback where the downstream
difference is zero), and `turbulence.advective_flux` to keep the sign of the flux, for each of
the four advection calls of a step (k along x and y, eps along x and y). Per outer iteration the
record keeps the number of faces whose branch changed from the previous iteration and the
number whose flux sign changed; over the last 2,000 iterations, the per-face count of branch
changes, so the faces that switch most can be placed in the room. The wrapper returns the
committed function's value, so the run is expected to reproduce the control's face hash.

### 2.4 Measurement 2: the exact grid pair

Standard and RNG on 160x60 and 320x120, four runs, classified as above. For every converged run:
the three-panel figure of prompt 46 (streamlines coloured by speed, nu_t / nu on a log scale, k)
with the boundaries marked as section 2.7 says, and one table row with the outer count, the five
readings at the stop, the wall time, the core's nu_t / nu (median and 95th percentile; the core
is step 5's, the non-SOLID cells above the equipment tops, y > 2.0 m) and y+ at the wall nodes
(median, least, largest, the share below 11.53), the quantities `coupled46.py` computes. For a
bounded run, `bounded44.py history`'s characterisation and `bounded44.py where`'s region.

### 2.5 Measurement 3: where particles go

On every converged run of measurement 2, three sources, each a separate march:

| Source | Where | Square (closed), m |
|---|---|---|
| S1 | an operator in the recirculation roll beside the door wall | [0.75, 0.85] x [1.15, 1.25] |
| S2 | a process emission just above the litho tool's top | [3.80, 3.90] x [2.00, 2.10] |
| S3 | an operator at the hood bench | [6.15, 6.25] x [1.15, 1.25] |

Each square is 0.1 m, centred on the prompt's point, its edges on cell faces of both grids: four
cells on 160x60, sixteen on 320x120, the same square. The emission is Q = 1.0e4 particles per
second per metre of depth for each class, spread uniformly by volume over the square's cells,
continuous, from a zero field. The sensor `hood_entry` (6.2, 1.2) lies inside S3's square, so
for S3 it reads the source itself and is not a discriminating sensor.

**The discrimination check**, first, on 160x60 with the standard variant (with RNG if the
standard variant does not converge there and RNG does). A source discriminates when its steady
plume puts a concentration above the floor at two or more sensors, or on two or more surfaces,
where

- the floor is 1e-6 of the source-cell concentration, the largest steady concentration among
  the source's own cells;
- a sensor's concentration is the bilinear interpolation from the cell centres with SOLID cells
  at zero (`transport46.interpolated`, step 5's rule); a sensor inside the source square does not
  count;
- a surface's concentration is the largest C_P over the surface's depositing faces (the named
  surfaces of `transport46.surface_names`: floor pieces, obstacle tops, obstacle sides, the
  walls, the ceiling); a surface the source square touches does not count. For S2 that excludes
  the litho tool's top, which its square sits on.

A source that fails is reported and moved once, by this rule: 0.5 m upward (y + 0.5), keeping x,
out of the gap's capture zone and into the supply's downflow; if that would put the square
inside an obstacle or above the ceiling, 0.5 m in x away from the nearest return instead. The
moved source is checked again and reported either way; it is then the source every other run
uses.

**The march.** Prompt 46's (its report, section 2.2, measurement 4), with one change stated
here: the converged run's face velocities frozen and its eddy viscosity (the solver's
`turbulence_state.nu_t`) passed to `solve_timestep` as `eddy_viscosity` with the configuration's
`turbulent_schmidt` 0.7; the two classes 0.5 and 5 micrometres (indices 2 and 4); one dt for
both classes, the smaller of their `stable_dt`, rounded down so that a 1.0 s window is a whole
number of steps; steady when over a window the in-domain count's relative change and each
sensor's change relative to the largest sensor reading are below 1e-4 and the removal rate is
within 1e-3 of Q; cap 600 s of simulated time. The change: the march's Courant number is 0.4
(`transport.cfl_number`, overridden in the probe's configuration mapping; the bound is 0.5)
where prompt 46 used the file's 0.1, because a 320x120 march at 0.1 is about four times a 200x75
march's cost and the set has up to twelve. The scheme's fixed point does not depend on dt (the
explicit advection and the implicit diffusion cancel it at a steady field), so the step sets the
march's length, not its answer; one check march, S1 on 160x60 at 0.1, is run beside the 0.4 row
and the two steady fields compared (the largest relative difference of the sensors and of the
per-surface deposition).

**Reported per run, source and class**: the sensors above the floor, in descending order; the
deposition rate per named surface (`v_d A_f C_P` per depositing face, summed; checked against
the budget's deposited increment over the last window); the five largest 0.2 m segments (every
depositing surface cut into 0.2 m bins, floors and tops along x, walls and sides along y, keyed
by surface kind so grids compare; prompt 46's `rescore` keying); one concentration figure per
run and source with a deposition panel (section 2.7).

### 2.6 Measurement 5: the comparative check

For each source and class:

- **between grids** (standard variant): the hotspots in common among the five largest segments
  on 160x60 and 320x120, the largest one's surface and location on each, and the order of the
  sensors above the floor on each;
- **between variants**: the same between standard and RNG on every grid where both converge.

Counts and orders, not percentages of absolute values.

### 2.7 Figures

Every figure marks the supply (a blue bar along the ceiling with downward arrows), the returns
(red bars on the floor), the hood (an orange bar on the right wall with an outward arrow), the
door (a bar on the left wall over its height), the obstacles (grey), the sensors (labelled by
name) and, on the concentration figures, the source square. The deposition panel draws each
0.2 m segment's deposition as a bar standing on its surface (up from a floor or an obstacle
top, down from the ceiling, into the room from a wall or an obstacle side), its length
proportional to the segment's rate, one colour per class, with the scale stated in the panel's
title: the bar length that stands for the largest segment's rate, in particles per second per
metre of depth.

## 3. Predictions

### 3.1 Orchestrator (2026-10-10; committed before any run)

- (a) Upwind advection of k and eps converges the coarse-grid cycle; the corner rule's removal
  and the halved `alpha_turbulence` do not. The cycle is a limiter switching branch near the
  rack's corner, iteration to iteration.
- (b) The standard model converges on 160x60 and on 320x120.
- (c) RNG converges on 160x60. On 320x120: no prediction.
- (d) S1 and S3 pass the discrimination check on the first placement; S2 is carried into the
  nearest gap's return before reaching any sensor and needs moving.
- (e) Between grids, for every source that passes the check: the largest deposition location is
  the same surface, at least three of the five hotspots are shared, and the sensor order is the
  same.
- (f) Between variants, where both converge: the largest deposition location is the same
  surface, and at least three of the five hotspots are shared.

What each outcome means, written with the predictions:

- (b) fails: no grid pair with an exact room converges, and the convergence aid goes to Alex.
- (e) holds: the comparative claim stands under refinement on this room, and the product grid
  should be one of the exact pair, not 200x75. 160x60 if it agrees with 320x120, since it costs
  a fraction.
- (e) fails: the hotspots move with resolution, and the comparative claim needs a finer grid or
  a better near-wall treatment (ADR-012's known second-cell overshoot).
- (f) fails while (e) holds: the variant decides the answer, and VAL-020 (the impinging jet)
  decides between them.

### 3.2 Builder (2026-10-10, before any run; scored apart)

- (a) Upwind converges and the other two do not, as predicted; the limiter diagnostic finds
  branch switches every iteration of the tail, most of them on faces within 0.2 m of the server
  rack's top corner and return 2, and flux sign changes on a handful of faces at most. The
  halved `alpha_turbulence` shrinks the tail's velocity step by about half without removing it.
- (b) The standard variant converges on 320x120 (its cells are finer than 200x75's, where it
  converged) and I expect it bounded on 160x60, between 80x30 where it stalls and 200x75 where
  it converges; if it converges there, the count is above 200x75's 3,421.
- (c) RNG converges on 160x60 and stalls on 320x120 as it did on 200x75.
- (d) S1 and S3 pass. S2 passes through the surfaces clause, not the sensors: its plume runs off
  the tool's top into the gap and deposits on the gap's faces and floor, so it does not need
  moving, against the orchestrator's (d). No sensor other than `near_door` (for S1) and
  `hood_entry` (inside S3) reads above the floor from any source.
- (e) Holds for the largest location and for three of five hotspots for S1 and S3; the sensor
  order holds trivially where one sensor reads. For S2 the five hotspots split between the gap's
  two faces and fewer than three are shared.
- (f) Not scorable on 320x120 if (c)'s second half holds; on 160x60 the largest location agrees
  and three of five hotspots are shared.
- The 0.1 against 0.4 Courant check agrees to the march's tolerance (1e-4 relative at the
  sensors and surfaces).

## 4. Stop conditions (from the prompt)

- A positivity error or a capped pressure correction: the iteration, the cell and the run are
  reported, and the run is its own outcome in the table.
- Neither model converges on either exact grid: measurement 2's characterisation is finished,
  measurements 3 and 5 are skipped, the report is written.
- The run set would exceed about eight hours: the estimate and a reduced set are reported first
  (320x120 is about 2.6 times 200x75's cells).

## 5. Results (written after the runs)

### 5.1 Order, cost and the rooms

`rooms47.py` on the four grids. Section 1's claim holds for the exact pair; 200x75 rounds
fourteen positions, not the twelve the prompt counted (the supply's two ends are among them):

<!-- tables47 rooms begin -->
| Grid | Cell dx, dy (m) | Positions on faces | Rounded | Rounded positions |
|---|---|---|---|---|
| 80x30 | 0.1, 0.1 | 36 of 38 | 2 | floor_return_1.x_end 1.25; floor_return_2.x_end 2.95 |
| 160x60 | 0.05, 0.05 | 38 of 38 | 0 | - |
| 200x75 | 0.04, 0.04 | 24 of 38 | 14 | hepa_supply.x_start 0.5; hepa_supply.x_end 7.5; door.y_end 2.1; floor_return_1.x_start 0.5; floor_return_1.x_end 1.25; floor_return_2.x_end 2.95; floor_return_3.x_end 4.9; floor_return_4.x_start 5.7; floor_return_4.x_end 6.3; hood_exhaust.y_start 0.9; server_rack.x_start 1.5; server_rack.x_end 2.3; litho_tool.x_end 4.5; hood_bench.y_end 0.9 |
| 320x120 | 0.025, 0.025 | 38 of 38 | 0 | - |
<!-- tables47 rooms end -->

The timing probes ran first (09:48), 20 outer iterations each, two processes at once:

<!-- tables47 timing begin -->
| Grid | Outer | Seconds per outer | CG per correction (mean) | k, eps sweeps (mean) | Share of wall: pressure, turbulence | Minutes at the 10,000 cap |
|---|---|---|---|---|---|---|
| 160x60 | 20 | 0.104 | 540 | 9.6, 7.5 | 0.68, 0.17 | 17 |
| 320x120 | 20 | 0.561 | 974 | 26.2, 16.5 | 0.78, 0.13 | 93 |
<!-- tables47 timing end -->

The projection at the cap was under two hours of wall time for the longest row, so the set
ran as planned. The five 80x30 rows of measurement 1 were launched together at 09:48:42 and the
four rows of measurement 2 at 09:48:54, nine processes at once on twelve cores; the marches of
measurement 3 followed as rows converged (section 5.4). The wall times in the tables were taken
beside other runs.

### 5.2 Measurement 1: the coarse-grid cycle, diagnosed

<!-- tables47 cycle begin -->
| Run (arm) | Change | Class | Outer | Residual at end | Readings at stop (a), (b), (c), (d), (e) | Holds from (a), (b), (c), (d), (e) | Tail velocity step: median, largest (over 0.45 m/s) | Face hash (equals prompt 46's row) |
|---|---|---|---|---|---|---|---|---|
| standard_80x30 (control) | none | bounded (neither) | 10,000 | 9.32e-07 | 0.021, 6.9e-10, 3.9e-08, 2e-15, 40 | -, 187, 164, 1, - | 9.2e-05, 0.00014 | 6ab5e2c7d5a1fa0c (yes) |
| standard_80x30_upwind (upwind) | k and eps advected by upwind | converged | 1,046 | 1.68e-12 | 5.9e-09, 6.8e-14, 2.2e-12, 2e-15, 9.8e-07 | 788, 192, 181, 1, 1,046 | 1.4e-06, 0.045 | 28f437c84e3d5057 (no) |
| standard_80x30_corner (corner) | momentum corner rule removed | bounded (neither) | 10,000 | 1.47e-06 | -, 1.2e-09, 5e-08, 2e-15, 6.9e+02 | -, 183, 162, 1, - | 9e-05, 0.00012 | f0a8cb0764883cbf (no) |
| standard_80x30_alpha (alpha) | alpha_turbulence 0.35 | bounded (neither) | 10,000 | 1.04e-06 | 0.021, 7.9e-10, 4.5e-08, 2e-15, 68 | -, 193, 168, 1, - | 7.9e-05, 0.00013 | e1a44d7f25d3bdb6 (no) |
| standard_80x30_limiter (limiter) | none (branches recorded) | bounded (neither) | 10,000 | 9.32e-07 | 0.021, 6.9e-10, 3.9e-08, 2e-15, 40 | -, 187, 164, 1, - | 9.2e-05, 0.00014 | 6ab5e2c7d5a1fa0c (yes) |
<!-- tables47 cycle end -->

The control reproduces prompt 46's row bit for bit (the face hash column) and the limiter
diagnostic's row, which wraps two functions without changing their values, reproduces it too.
Upwind advection of k and eps converges the row by the rule; the corner rule's removal and the
halved `alpha_turbulence` leave it bounded at the cap with the same small velocity step
(the tail column: about 1e-4 of the scale at the largest) and the same refusal of conditions
(a) and (e). The limiter diagnostic's record of the branch switches:

<!-- tables47 limiter begin -->
| Run | Faces per iteration (4 calls) | Tail iterations | Branch switches per iteration: least, median, largest; iterations with none | Flux sign switches per iteration: least, median, largest; iterations with none | Median switches per call (k x, k y, eps x, eps y) | Faces that switched at least once in the tail | Faces switching most (call, x, y, share of tail) | Regions of those faces | Branch shares at the end (zero, 2r, (1+3r)/4, quick, 2, c_c) |
|---|---|---|---|---|---|---|---|---|---|
| standard_80x30_limiter | 9,380 | 2,000 | 3, 22, 45; 0 | 0, 0, 0; 2000 | 1, 11, 1, 8 | 234 | k_y (2.55, 1) 0.21; k_y (2.55, 0.6) 0.21; k_y (2.35, 1.3) 0.20; eps_y (2.35, 1.3) 0.20; k_y (2.35, 1.7) 0.20; k_y (2.55, 0.8) 0.20; eps_y (2.35, 1.7) 0.20; k_y (2.55, 0.9) 0.19 | etch_chamber face 8; server_rack face 7; gap between equipment 5 | 1,496, 689, 2,416, 2,243, 160, 2,376 |

Branch and sign switches per iteration, medians per thousand iterations:
| Run | 1 | 1,001 | 2,001 | 3,001 | 4,001 | 5,001 | 6,001 | 7,001 | 8,001 | 9,001 |
|---|---|---|---|---|---|---|---|---|---|---|
| standard_80x30_limiter (branch) | 24 | 22 | 22 | 22 | 22 | 22 | 22 | 22 | 22 | 22 |
| standard_80x30_limiter (sign) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
<!-- tables47 limiter end -->

The bounded rows' tails, with the regions of largest change:

<!-- tables47 bounded begin -->
| Run | Outer, tail | Residual: least, median, largest | Amplitude p95/p5 | Drift (log10 per 1,000) | Period (height); strongest (height) | Largest speed: least, largest; cells | Located: share by region (largest of u, v); cells | Largest nu_t change: share by region; cells; size (m^2/s) |
|---|---|---|---|---|---|---|---|---|
| rng_160x60 | 10,000, 2,000 | 4.82e-04, 5.53e-04, 6.05e-04 | 1.18 | 0.000612 | 10 (0.24); 19 (0.99) | 1.56, 1.56; (0.525, 0.025) | gap between equipment 1.00; (2.975, 1.675), (2.925, 1.475), (2.875, 1.325) | gap between equipment 1.00; (2.975, 0.875), (2.975, 1.125), (2.975, 0.925); 8.1e-05 |
| rng_320x120 | 10,000, 2,000 | 2.69e-04, 2.89e-04, 3.09e-04 | 1.09 | 0.000393 | 9 (0.31); 19 (0.73) | 1.74, 1.74; (0.5125, 0.0125) | gap between equipment 0.54; hood_bench face 0.35; hood_bench top 0.11; (6.2625, 0.6375), (2.9125, 1.5375), (2.9375, 1.6125) | gap between equipment 0.78; hood_bench face 0.21; etch_chamber face 0.00; (2.9625, 1.2125), (2.9625, 1.2625), (2.9625, 1.1875); 4.5e-05 |
| standard_320x120 | 10,000, 2,000 | 6.24e-08, 8.72e-08, 9.93e-08 | 1.5 | 0.000454 | 29 (0.8); 57 (0.97) | 1.64, 1.64; (0.5125, 0.0125) | litho_tool top 1.00; (4.5125, 1.9375), (4.5125, 1.9125), (4.5125, 1.8875) | litho_tool top 1.00; (4.5125, 2.0375), (4.5375, 2.0125), (4.5625, 2.0125); 1.1e-06 |
| standard_80x30 | 10,000, 2,000 | 7.46e-07, 1.32e-06, 1.96e-06 | 1.99 | -0.00203 | 28 (0.33); 342 (0.68) | 1.4, 1.4; (2.75, 1.35) | return 2 0.43; server_rack top 0.27; etch_chamber top 0.13; (2.35, 0.15), (2.35, 2.05), (4.95, 2.05) | server_rack face 0.44; server_rack top 0.44; etch_chamber top 0.10; (2.35, 2.05), (4.95, 2.05), (2.35, 1.85); 9.7e-06 |
| standard_80x30_alpha | 10,000, 2,000 | 7.74e-07, 1.13e-06, 1.91e-06 | 1.77 | -0.000466 | 32 (0.4); 192 (0.59) | 1.4, 1.4; (2.75, 1.35) | return 2 0.33; etch_chamber face 0.22; etch_chamber top 0.15; (2.35, 0.15), (4.95, 2.05), (4.85, 0.15) | server_rack top 0.42; server_rack face 0.38; etch_chamber top 0.15; (2.35, 2.05), (4.95, 2.05), (2.45, 2.05); 8.4e-06 |
| standard_80x30_corner | 10,000, 2,000 | 7.28e-07, 1.28e-06, 1.76e-06 | 1.93 | 0.00261 | 28 (0.23); 408 (0.76) | 1.39, 1.39; (2.75, 1.35) | server_rack top 0.34; return 2 0.31; etch_chamber top 0.16; (2.25, 2.15), (2.35, 0.15), (4.95, 2.05) | server_rack face 0.57; server_rack top 0.41; etch_chamber top 0.02; (2.35, 2.05), (2.35, 1.85), (2.35, 1.95); 1e-05 |
| standard_80x30_limiter | 10,000, 2,000 | 7.46e-07, 1.32e-06, 1.96e-06 | 1.99 | -0.00203 | 28 (0.33); 342 (0.68) | 1.4, 1.4; (2.75, 1.35) | return 2 0.43; server_rack top 0.27; etch_chamber top 0.13; (2.35, 0.15), (2.35, 2.05), (4.95, 2.05) | server_rack face 0.44; server_rack top 0.44; etch_chamber top 0.10; (2.35, 2.05), (4.95, 2.05), (2.35, 1.85); 9.7e-06 |
<!-- tables47 bounded end -->

### 5.3 Measurement 2: the exact grid pair

The matrix of every run and the converged rows' readings (added after test 47, which found the
section cited but missing; the tables are the generated ones, unchanged):

<!-- tables47 matrix begin -->
| Run | Class | Stop | Outer | Residual: least (at), end | Readings at stop (a), (b), (c), (d), (e) | Holds from (a), (b), (c), (d), (e) | Largest speed at end (m/s), cell | Peak speed | CG per correction: mean, largest | Cap hits | k, eps sweeps (mean) | Wall (s), per outer | Face hash |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| standard_80x30 | bounded (neither) | max_simple_iter | 10,000 | 7.43e-07 (5,352), 9.32e-07 | 0.021, 6.9e-10, 3.9e-08, 2e-15, 40 | -, 187, 164, 1, - | 1.4, (2.75, 1.35) | 1.62 | 223, 287 | 0 | 23.8, 18.9 | 384, 0.038 | 6ab5e2c7d5a1fa0c |
| standard_80x30_upwind | converged | error_estimate_and_continuity | 1,046 | 1.68e-12 (1,046), 1.68e-12 | 5.9e-09, 6.8e-14, 2.2e-12, 2e-15, 9.8e-07 | 788, 192, 181, 1, 1,046 | 1.41, (2.75, 1.35) | 1.63 | 253, 287 | 0 | 19.5, 15.8 | 39, 0.038 | 28f437c84e3d5057 |
| standard_80x30_corner | bounded (neither) | max_simple_iter | 10,000 | 7.28e-07 (8,827), 1.47e-06 | -, 1.2e-09, 5e-08, 2e-15, 6.9e+02 | -, 183, 162, 1, - | 1.39, (2.75, 1.35) | 1.57 | 222, 287 | 0 | 24.8, 19.8 | 388, 0.039 | f0a8cb0764883cbf |
| standard_80x30_alpha | bounded (neither) | max_simple_iter | 10,000 | 7.73e-07 (924), 1.04e-06 | 0.021, 7.9e-10, 4.5e-08, 2e-15, 68 | -, 193, 168, 1, - | 1.4, (2.75, 1.35) | 1.63 | 226, 284 | 0 | 23.8, 18.8 | 388, 0.039 | e1a44d7f25d3bdb6 |
| standard_80x30_limiter | bounded (neither) | max_simple_iter | 10,000 | 7.43e-07 (5,352), 9.32e-07 | 0.021, 6.9e-10, 3.9e-08, 2e-15, 40 | -, 187, 164, 1, - | 1.4, (2.75, 1.35) | 1.62 | 223, 287 | 0 | 23.8, 18.9 | 413, 0.041 | 6ab5e2c7d5a1fa0c |
| standard_160x60 | converged | error_estimate_and_continuity | 2,732 | 3.13e-13 (2,732), 3.13e-13 | 6.5e-09, 1.2e-13, 3.1e-12, 5.5e-15, 1e-06 | 1,980, 397, 340, 1, 2,732 | 1.5, (0.525, 0.025) | 1.68 | 357, 574 | 0 | 45.5, 35.1 | 310, 0.114 | cea39bd670cf68a6 |
| standard_160x60_upwind | converged | error_estimate_and_continuity | 2,003 | 5.49e-13 (2,002), 5.49e-13 | 9.4e-09, 5.8e-14, 3.4e-12, 5.5e-15, 1e-06 | 1,432, 398, 374, 1, 2,003 | 1.54, (0.525, 0.025) | 1.75 | 399, 574 | 0 | 37.4, 28.2 | 230, 0.115 | 73f5eaded654a1a7 |
| rng_160x60 | bounded (stalled) | max_simple_iter | 10,000 | 3.74e-04 (455), 5.72e-04 | 3.6e+04, 1.7e-07, 1.4e-05, 5.5e-15, 7.6e+03 | -, -, -, 1, - | 1.56, (0.525, 0.025) | 1.9 | 303, 575 | 0 | 36.2, 31.3 | 1030, 0.103 | 9c7e473bb0c71fc2 |
| standard_320x120 | bounded (stalled) | max_simple_iter | 10,000 | 6.24e-08 (4,943), 9.29e-08 | 0.18, 2.2e-11, 6.5e-09, 8.5e-15, - | -, 1,068, 799, 1, - | 1.64, (0.5125, 0.0125) | 2.13 | 590, 1118 | 0 | 101.5, 74.2 | 5331, 0.533 | e12d5b6cbf84ddad |
| standard_320x120_upwind | converged | error_estimate_and_continuity | 4,472 | 1.94e-13 (4,472), 1.94e-13 | 1.2e-08, 1.4e-13, 3.7e-12, 8.6e-15, 1e-06 | 3,530, 1,068, 762, 1, 4,472 | 1.69, (0.5125, 0.0125) | 2.13 | 713, 1118 | 0 | 87.8, 63.7 | 2556, 0.572 | a055b8e30d9b00ca |
| rng_320x120 | bounded (stalled) | max_simple_iter | 10,000 | 2.36e-04 (1,532), 2.80e-04 | 7.5e+02, 6.4e-08, 2.2e-05, 8.6e-15, 2.1e+03 | -, -, -, 1, - | 1.74, (0.5125, 0.0125) | 2.41 | 351, 1121 | 0 | 77.7, 65.5 | 3942, 0.394 | ac1505efddd8ec95 |
<!-- tables47 matrix end -->

<!-- tables47 converged_rows begin -->
| Run | Outer, stop | Readings at stop (a), (b), (c), (d), (e) | Holds from (a), (b), (c), (d), (e) | Wall (s) | Inlet nu_t / nu | Core nu_t / nu: median, 95th | y+ nodes | y+: median, least, largest | Share below 11.53: all; domain, tops, sides | Figure |
|---|---|---|---|---|---|---|---|---|---|---|
| standard_80x30_upwind | 1,046, error_estimate_and_continuity | 5.9e-09, 6.8e-14, 2.2e-12, 2e-15, 9.8e-07 | 788, 192, 181, 1, 1,046 | 39 | 16.4 | 17.4, 56.5 | 257 | 105, 8.19, 536 | 0.023; 0.062, 0, 0.0072 | ecr002_step6_standard_80x30_upwind.png |
| standard_160x60 | 2,732, error_estimate_and_continuity | 6.5e-09, 1.2e-13, 3.1e-12, 5.5e-15, 1e-06 | 1,980, 397, 340, 1, 2,732 | 310 | 16.4 | 17, 93.1 | 514 | 79.6, 3.55, 329 | 0.037; 0.094, 0, 0.014 | ecr002_step6_standard_160x60.png |
| standard_160x60_upwind | 2,003, error_estimate_and_continuity | 9.4e-09, 5.8e-14, 3.4e-12, 5.5e-15, 1e-06 | 1,432, 398, 374, 1, 2,003 | 230 | 16.4 | 17.1, 58 | 514 | 61.8, 4.08, 296 | 0.035; 0.094, 0, 0.011 | ecr002_step6_standard_160x60_upwind.png |
| standard_320x120_upwind | 4,472, error_estimate_and_continuity | 1.2e-08, 1.4e-13, 3.7e-12, 8.6e-15, 1e-06 | 3,530, 1,068, 762, 1, 4,472 | 2556 | 16.4 | 17, 56.2 | 1,028 | 34.3, 2.11, 167 | 0.096; 0.24, 0, 0.038 | ecr002_step6_standard_320x120_upwind.png |
<!-- tables47 converged_rows end -->

What the tables say. The standard variant converges on 160x60 and does not converge on
320x120. On 320x120 its residual fell as on 160x60 to about outer 1,800 and then sat between
6e-8 and 1e-7 to the cap (the bounded table's amplitude and drift), a cycle of the kind the
80x30 control shows at 1e-6: continuity met (conditions (b), (c) and (d) hold from the
iterations the matrix gives), the velocity step steady, conditions (a) and (e) refusing. It
recurs at 29 and 57 outer iterations (the bounded table's period and strongest columns) and
its largest change of velocity and of nu_t sits in every tail iteration at the litho tool's
east top corner (x 4.51 to 4.56, y 1.89 to 2.04), the shear layer leaving the corner into the
gap above return 3. RNG is bounded on both exact grids as it was on 200x75 (prompt 46): its velocity step
is about a twelfth of the scale, its per-cell imbalance tens of times its tolerance (the matrix
table's (b) reading against `mass_imbalance_tol`, 5.0e-9 on 160x60 and 1.25e-9 on 320x120), and
the recurrence is clean at a period of 19 outer iterations on both grids (the bounded table),
located in the gap above return 2 between 0.9 and 1.7 m up, a quarter metre from the litho
tool's west face.

**The supplementary pair.** When the standard row's residual on 320x120 had sat flat for 900
iterations (outer 1,800 to 2,700), the one arm of measurement 1 that converged the 80x30 cycle,
upwind advection of k and eps, was run on both exact grids (`run47.sh pair-upwind`, launched
10:17), beside the prompt's rows and labelled apart from them. It converges on both: 2,003
outer iterations on 160x60 and 4,472 on 320x120, against 2,732 for the committed scheme on
160x60. The three converged rooms (standard 160x60, upwind 160x60, upwind 320x120) have the
same core eddy viscosity (the converged table's median) and the same flow pattern (the
figures); their y+ medians fall with the cell size as the first node moves toward the wall,
and the share of nodes below the scalable floor rises to about a tenth on 320x120, a quarter
of the domain-wall nodes. Measurements 3 and 5 run on the committed scheme where it converged
(160x60) and on the upwind pair, and the comparison between grids the prompt asked for is
scored on the upwind pair, with the comparison between the two schemes on 160x60 beside it so
the reader can see what the scheme changes.

### 5.4 Measurement 3: where particles go

**The discrimination check**, on the committed scheme's converged 160x60 room:

<!-- tables47 checks begin -->
| Source (record) | Square (m) | Class (um) | Source-cell C, floor (per m^3) | Sensors above the floor (excluding one inside the square) | Surfaces above the floor (excluding ones the square touches) | Discriminates | Verdict; move |
|---|---|---|---|---|---|---|---|
| S3 (standard_160x60) | [6.15, 6.25, 1.15, 1.25] | 0.5 | 1.04e+05, 0.104 | none | floor 1.6e+03, hood_bench west 1.5e+03, etch_chamber east 44 | yes | passes |
|  |  | 5 | 1.04e+05, 0.104 | none | floor 1.6e+03, hood_bench west 1.5e+03, etch_chamber east 43 | yes |  |
| S2 (standard_160x60) | [3.8, 3.9, 2.0, 2.1] | 0.5 | 6.84e+05, 0.684 | none | litho_tool west 1.2e+05, floor 4.9e+04, litho_tool east 12 | yes | passes |
|  |  | 5 | 6.83e+05, 0.683 | none | litho_tool west 1.2e+05, floor 4.8e+04, litho_tool east 12 | yes |  |
| S1 (standard_160x60) | [0.75, 0.85, 1.15, 1.25] | 0.5 | 7.47e+04, 0.0747 | near_door 4.7e+02 | floor 9.7e+03, server_rack west 8.9e+03 | yes | passes |
|  |  | 5 | 7.46e+04, 0.0746 | near_door 4.6e+02 | floor 9.7e+03, server_rack west 8.9e+03 | yes |  |
<!-- tables47 checks end -->

All three sources pass on the first placement, so none moves and `sources.json` was never
written. S1 is the only source a sensor reads: `near_door`, 0.3 m above it in the column that
descends to return 1. S2 and S3 pass through the surfaces clause alone, with no sensor above
the floor; `hood_entry` lies inside S3's square and is excluded by the method. Every plume ends
in a return: S1's in return 1, S2's in return 2 after running down the litho tool's west face,
S3's in return 4 directly below it, and the deposition is a small part of the emission (the
transport table's deposition over Q column). The check as the prompt specifies it is met;
whether a source whose plume a return captures within a metre is the source VAL-018 wants is a
question for section 7.

**The Courant check**, S1 on the same room at 0.1 and 0.4:

<!-- tables47 cfl begin -->
| Rows (0.4, check), class (um) | Courant | dt (s) | Steady at (s) | Wall (s) | Sensors: largest difference over the largest reading | Surfaces: largest difference over the largest rate | Hotspots equal | Sensor order equal |
|---|---|---|---|---|---|---|---|---|
| transport_standard_160x60_S1, transport_standard_160x60_S1_cfl0.1, 0.5 | 0.4, 0.1 | 0.008, 0.002 | 68, 68 | 150, 549 | 8.53e-06 | 6.00e-06 | yes | yes |
| transport_standard_160x60_S1, transport_standard_160x60_S1_cfl0.1, 5 | 0.4, 0.1 | 0.008, 0.002 | 68, 68 | 150, 549 | 8.56e-06 | 6.02e-06 | yes | yes |
<!-- tables47 cfl end -->

The two marches stop at the same simulated time with the sensors and the per-surface rates
within 1e-5 of each other relative to their largest values, the same five hotspots and the
same sensor order, at a quarter of the wall time; the 0.4 marches stand.

**The marches**, every converged room, every source, both classes:

<!-- tables47 transport begin -->
| Run, class (um) | Stop, t (s) | Last window: total, sensor change; removal / Q | Deposition / Q (faces; budget), outflow / Q | Budget residual | Source-cell C, floor | Sensors above the floor, in order (per m^3) |
|---|---|---|---|---|---|---|
| standard_160x60_S1, 0.5 | steady, 68 | 1.6e-05, 9.8e-05; 0.99998 | 2.405e-06 (2.405e-06), 1 | 1.8e-07 | 7.47e+04, 0.0747 | near_door (468) |
| standard_160x60_S1, 5 | steady, 68 | 1.6e-05, 9.7e-05; 0.99998 | 0.0001778 (0.0001778), 0.9998 | 1.8e-07 | 7.46e+04, 0.0746 | near_door (459) |
| standard_160x60_S2, 0.5 | steady, 43 | 9.3e-05, 1.9e-06; 0.99934 | 0.0002976 (0.0002976), 0.999 | -2.8e-06 | 6.84e+05, 0.684 | none |
| standard_160x60_S2, 5 | steady, 43 | 9.2e-05, 1.9e-06; 0.99937 | 0.02256 (0.02256), 0.9768 | -2.8e-06 | 6.83e+05, 0.683 | none |
| standard_160x60_S3, 0.5 | steady, 10 | 9.1e-05, 2.5e-12; 0.99991 | 1.688e-07 (1.68e-07), 0.9999 | 1.4e-07 | 1.04e+05, 0.104 | none |
| standard_160x60_S3, 5 | steady, 10 | 9e-05, 1.9e-12; 0.99991 | 1.258e-05 (1.253e-05), 0.9999 | 1.4e-07 | 1.04e+05, 0.104 | none |
| standard_160x60_upwind_S1, 0.5 | steady, 84 | 1.6e-05, 9.6e-05; 0.99998 | 1.818e-06 (1.818e-06), 1 | 3.4e-07 | 7.69e+04, 0.0769 | near_door (400) |
| standard_160x60_upwind_S1, 5 | steady, 84 | 1.6e-05, 9.6e-05; 0.99998 | 0.0001338 (0.0001338), 0.9998 | 3.4e-07 | 7.69e+04, 0.0769 | near_door (392) |
| standard_160x60_upwind_S2, 0.5 | steady, 51 | 9.2e-05, 1.7e-06; 0.99933 | 0.0003003 (0.0003003), 0.999 | -3.1e-06 | 6.85e+05, 0.685 | none |
| standard_160x60_upwind_S2, 5 | steady, 51 | 9e-05, 1.7e-06; 0.99935 | 0.02276 (0.02276), 0.9766 | -3.7e-06 | 6.84e+05, 0.684 | none |
| standard_160x60_upwind_S3, 0.5 | steady, 10 | 9.6e-05, 3.6e-12; 0.9999 | 8.021e-08 (7.962e-08), 0.9999 | 2.5e-07 | 1.03e+05, 0.103 | none |
| standard_160x60_upwind_S3, 5 | steady, 10 | 9.6e-05, 3.6e-12; 0.9999 | 5.965e-06 (5.922e-06), 0.9999 | 2.5e-07 | 1.03e+05, 0.103 | none |
| standard_320x120_upwind_S1, 0.5 | steady, 91 | 7.1e-06, 9.5e-05; 0.99999 | 7.668e-07 (7.668e-07), 1 | 2.1e-07 | 7.85e+04, 0.0785 | near_door (93.6) |
| standard_320x120_upwind_S1, 5 | steady, 91 | 7e-06, 9.5e-05; 0.99999 | 5.64e-05 (5.64e-05), 0.9999 | 2.1e-07 | 7.84e+04, 0.0784 | near_door (91) |
| standard_320x120_upwind_S2, 0.5 | steady, 55 | 9.6e-05, 1.2e-08; 0.99923 | 0.0003684 (0.0003684), 0.9989 | -5.6e-06 | 7.29e+05, 0.729 | none |
| standard_320x120_upwind_S2, 5 | steady, 55 | 9.4e-05, 1.3e-08; 0.99926 | 0.02792 (0.02792), 0.9713 | -5.5e-06 | 7.28e+05, 0.728 | none |
| standard_320x120_upwind_S3, 0.5 | steady, 4 | 6.5e-05, 3.8e-16; 0.99993 | 1.232e-08 (1.194e-08), 0.9999 | 1.3e-07 | 1.09e+05, 0.109 | none |
| standard_320x120_upwind_S3, 5 | steady, 4 | 6.4e-05, 5.1e-16; 0.99993 | 9.266e-07 (8.992e-07), 0.9999 | 1.3e-07 | 1.09e+05, 0.109 | none |

Deposition per surface, source S1, share of the deposition total (surfaces with at least 1e-06 of it in some column):
| Surface | standard_160x60_S1, 0.5 um | standard_160x60_S1, 5 um | standard_160x60_upwind_S1, 0.5 um | standard_160x60_upwind_S1, 5 um | standard_320x120_upwind_S1, 0.5 um | standard_320x120_upwind_S1, 5 um |
|---|---|---|---|---|---|---|
| floor 1.25-1.50 | 0.968 | 1 | 0.965 | 1 | 0.967 | 1 |
| server_rack west | 0.0317 | 3.29e-05 | 0.035 | 3.64e-05 | 0.0331 | 3.44e-05 |
| server_rack top | 8.75e-07 | 8.94e-07 | 1.01e-06 | 1.04e-06 | 7.64e-07 | 7.83e-07 |

Deposition per surface, source S2, share of the deposition total (surfaces with at least 1e-06 of it in some column):
| Surface | standard_160x60_S2, 0.5 um | standard_160x60_S2, 5 um | standard_160x60_upwind_S2, 0.5 um | standard_160x60_upwind_S2, 5 um | standard_320x120_upwind_S2, 0.5 um | standard_320x120_upwind_S2, 5 um |
|---|---|---|---|---|---|---|
| litho_tool top | 0.956 | 0.959 | 0.953 | 0.956 | 0.955 | 0.958 |
| floor 2.95-3.20 | 0.0418 | 0.0414 | 0.0448 | 0.0444 | 0.0427 | 0.042 |
| litho_tool west | 0.00257 | 2.56e-06 | 0.00263 | 2.62e-06 | 0.00261 | 2.59e-06 |
| floor 4.50-4.60 | 1.68e-06 | 1.6e-06 | 9.24e-07 | 8.8e-07 | 4.25e-07 | 3.96e-07 |

Deposition per surface, source S3, share of the deposition total (surfaces with at least 1e-06 of it in some column):
| Surface | standard_160x60_S3, 0.5 um | standard_160x60_S3, 5 um | standard_160x60_upwind_S3, 0.5 um | standard_160x60_upwind_S3, 5 um | standard_320x120_upwind_S3, 0.5 um | standard_320x120_upwind_S3, 5 um |
|---|---|---|---|---|---|---|
| floor 6.30-6.40 | 0.94 | 0.967 | 0.949 | 0.98 | 0.977 | 0.999 |
| floor 5.60-5.70 | 0.032 | 0.0326 | 0.0195 | 0.0199 | 0.000617 | 0.000618 |
| hood_bench west | 0.0281 | 2.91e-05 | 0.0311 | 3.23e-05 | 0.0219 | 2.25e-05 |
| etch_chamber east | 0.000343 | 3.51e-07 | 0.000183 | 1.87e-07 | 2.95e-06 | 2.96e-09 |
| hood_bench top | 2.43e-06 | 2.5e-06 | 2.2e-06 | 2.26e-06 | 5.05e-07 | 5.09e-07 |

The five segments of largest deposition (surface | bin start), rate per s per m depth:
| Run, class (um) | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| standard_160x60_S1, 0.5 | floor | x 1.2 (0.0142) | floor | x 1.4 (0.00908) | server_rack west | y 0.0 (0.000111) | server_rack west | y 0.2 (0.000103) | server_rack west | y 0.4 (9.19e-05) |
| standard_160x60_S1, 5 | floor | x 1.2 (1.09) | floor | x 1.4 (0.693) | server_rack west | y 0.0 (8.55e-06) | server_rack west | y 0.2 (7.94e-06) | server_rack west | y 0.4 (7.05e-06) |
| standard_160x60_S2, 0.5 | litho_tool top | x 3.6 (1.12) | litho_tool top | x 3.4 (0.745) | litho_tool top | x 3.8 (0.497) | litho_tool top | x 3.2 (0.478) | floor | x 3.0 (0.0998) |
| standard_160x60_S2, 5 | litho_tool top | x 3.6 (85.7) | litho_tool top | x 3.4 (56.4) | litho_tool top | x 3.8 (38) | litho_tool top | x 3.2 (36) | floor | x 3.0 (7.49) |
| standard_160x60_S3, 0.5 | floor | x 6.2 (0.00159) | floor | x 5.6 (5.4e-05) | hood_bench west | y 0.0 (1.8e-05) | hood_bench west | y 0.2 (1.23e-05) | hood_bench west | y 0.4 (8.77e-06) |
| standard_160x60_S3, 5 | floor | x 6.2 (0.122) | floor | x 5.6 (0.0041) | hood_bench west | y 0.0 (1.39e-06) | hood_bench west | y 0.2 (9.52e-07) | hood_bench west | y 0.4 (6.76e-07) |
| standard_160x60_upwind_S1, 0.5 | floor | x 1.2 (0.0108) | floor | x 1.4 (0.0068) | server_rack west | y 0.0 (8.34e-05) | server_rack west | y 0.2 (7.78e-05) | server_rack west | y 0.4 (7.16e-05) |
| standard_160x60_upwind_S1, 5 | floor | x 1.2 (0.82) | floor | x 1.4 (0.518) | server_rack west | y 0.0 (6.4e-06) | server_rack west | y 0.2 (5.97e-06) | server_rack west | y 0.4 (5.49e-06) |
| standard_160x60_upwind_S2, 0.5 | litho_tool top | x 3.6 (1.13) | litho_tool top | x 3.4 (0.753) | litho_tool top | x 3.8 (0.495) | litho_tool top | x 3.2 (0.482) | floor | x 3.0 (0.108) |
| standard_160x60_upwind_S2, 5 | litho_tool top | x 3.6 (86.2) | litho_tool top | x 3.4 (57) | litho_tool top | x 3.8 (37.9) | litho_tool top | x 3.2 (36.3) | floor | x 3.0 (8.1) |
| standard_160x60_upwind_S3, 0.5 | floor | x 6.2 (0.000761) | floor | x 5.6 (1.56e-05) | hood_bench west | y 0.0 (8.49e-06) | hood_bench west | y 0.2 (6.35e-06) | hood_bench west | y 0.4 (5.06e-06) |
| standard_160x60_upwind_S3, 5 | floor | x 6.2 (0.0585) | floor | x 5.6 (0.00119) | hood_bench west | y 0.0 (6.56e-07) | hood_bench west | y 0.2 (4.9e-07) | hood_bench west | y 0.4 (3.9e-07) |
| standard_320x120_upwind_S1, 0.5 | floor | x 1.2 (0.00456) | floor | x 1.4 (0.00285) | server_rack west | y 0.0 (3.46e-05) | server_rack west | y 0.2 (3.2e-05) | server_rack west | y 0.4 (2.89e-05) |
| standard_320x120_upwind_S1, 5 | floor | x 1.2 (0.347) | floor | x 1.4 (0.217) | server_rack west | y 0.0 (2.65e-06) | server_rack west | y 0.2 (2.44e-06) | server_rack west | y 0.4 (2.21e-06) |
| standard_320x120_upwind_S2, 0.5 | litho_tool top | x 3.6 (1.34) | litho_tool top | x 3.4 (1.01) | litho_tool top | x 3.2 (0.715) | litho_tool top | x 3.8 (0.451) | floor | x 3.0 (0.126) |
| standard_320x120_upwind_S2, 5 | litho_tool top | x 3.6 (102) | litho_tool top | x 3.4 (76.9) | litho_tool top | x 3.2 (53.8) | litho_tool top | x 3.8 (34.6) | floor | x 3.0 (9.42) |
| standard_320x120_upwind_S3, 0.5 | floor | x 6.2 (0.00012) | hood_bench west | y 0.0 (1.29e-06) | hood_bench west | y 0.2 (7.32e-07) | hood_bench west | y 0.4 (3.93e-07) | hood_bench west | y 0.6 (2.24e-07) |
| standard_320x120_upwind_S3, 5 | floor | x 6.2 (0.00926) | floor | x 5.6 (5.72e-06) | hood_bench west | y 0.0 (9.97e-08) | hood_bench west | y 0.2 (5.65e-08) | hood_bench west | y 0.4 (3.03e-08) |
<!-- tables47 transport end -->

The figures `ecr002_step6_<room>_<source>_concentration.png` beside this report show each
class's concentration on a log scale down to a millionth of its peak, with the deposition
panel. The panel's two classes share one bar scale, the largest segment's rate, and the 5
micrometre class deposits about 75 times the 0.5 micrometre class on the same faces
(the hotspot table's rates), so the 0.5 micrometre bars stand at about a hundredth of the 5
micrometre bars' length; the hotspot table carries both classes' values.

### 5.5 Measurement 5: the comparative check

<!-- tables47 compare begin -->
| Pair, source, class (um) | Hotspots in common (of 5) | In common | Largest: first | Largest: second | Same surface | Sensor order above the floor: first | Second | Orders equal |
|---|---|---|---|---|---|---|---|---|
| supplementary: grids, upwind, S1, 0.5 | 5 | floor | x 1.2; floor | x 1.4; server_rack west | y 0.0; server_rack west | y 0.2; server_rack west | y 0.4 | floor | x 1.2 | floor | x 1.2 | yes | near_door | near_door | yes |
| supplementary: grids, upwind, S1, 5 | 5 | floor | x 1.2; floor | x 1.4; server_rack west | y 0.0; server_rack west | y 0.2; server_rack west | y 0.4 | floor | x 1.2 | floor | x 1.2 | yes | near_door | near_door | yes |
| supplementary: schemes, 160x60, S1, 0.5 | 5 | floor | x 1.2; floor | x 1.4; server_rack west | y 0.0; server_rack west | y 0.2; server_rack west | y 0.4 | floor | x 1.2 | floor | x 1.2 | yes | near_door | near_door | yes |
| supplementary: schemes, 160x60, S1, 5 | 5 | floor | x 1.2; floor | x 1.4; server_rack west | y 0.0; server_rack west | y 0.2; server_rack west | y 0.4 | floor | x 1.2 | floor | x 1.2 | yes | near_door | near_door | yes |
| supplementary: grids, upwind, S2, 0.5 | 5 | litho_tool top | x 3.6; litho_tool top | x 3.4; litho_tool top | x 3.8; litho_tool top | x 3.2; floor | x 3.0 | litho_tool top | x 3.6 | litho_tool top | x 3.6 | yes | none | none | yes |
| supplementary: grids, upwind, S2, 5 | 5 | litho_tool top | x 3.6; litho_tool top | x 3.4; litho_tool top | x 3.8; litho_tool top | x 3.2; floor | x 3.0 | litho_tool top | x 3.6 | litho_tool top | x 3.6 | yes | none | none | yes |
| supplementary: schemes, 160x60, S2, 0.5 | 5 | litho_tool top | x 3.6; litho_tool top | x 3.4; litho_tool top | x 3.8; litho_tool top | x 3.2; floor | x 3.0 | litho_tool top | x 3.6 | litho_tool top | x 3.6 | yes | none | none | yes |
| supplementary: schemes, 160x60, S2, 5 | 5 | litho_tool top | x 3.6; litho_tool top | x 3.4; litho_tool top | x 3.8; litho_tool top | x 3.2; floor | x 3.0 | litho_tool top | x 3.6 | litho_tool top | x 3.6 | yes | none | none | yes |
| supplementary: grids, upwind, S3, 0.5 | 4 | floor | x 6.2; hood_bench west | y 0.0; hood_bench west | y 0.2; hood_bench west | y 0.4 | floor | x 6.2 | floor | x 6.2 | yes | none | none | yes |
| supplementary: grids, upwind, S3, 5 | 5 | floor | x 6.2; floor | x 5.6; hood_bench west | y 0.0; hood_bench west | y 0.2; hood_bench west | y 0.4 | floor | x 6.2 | floor | x 6.2 | yes | none | none | yes |
| supplementary: schemes, 160x60, S3, 0.5 | 5 | floor | x 6.2; floor | x 5.6; hood_bench west | y 0.0; hood_bench west | y 0.2; hood_bench west | y 0.4 | floor | x 6.2 | floor | x 6.2 | yes | none | none | yes |
| supplementary: schemes, 160x60, S3, 5 | 5 | floor | x 6.2; floor | x 5.6; hood_bench west | y 0.0; hood_bench west | y 0.2; hood_bench west | y 0.4 | floor | x 6.2 | floor | x 6.2 | yes | none | none | yes |
<!-- tables47 compare end -->

The prompt's two comparisons as written: between grids with the standard variant, no pair
(the standard row is bounded on 320x120); between variants, no pair on either grid (RNG is
bounded on both). The table scores the pairs the records allow: between grids on the upwind
pair, and between the two schemes on 160x60 and on 320x120 where both converged.

## 6. Predictions against the measurement

| Prediction | Measured |
|---|---|
| (a) Upwind advection of k and eps converges the coarse-grid cycle; the corner rule's removal and the halved `alpha_turbulence` do not. The cycle is a limiter switching branch near the rack's corner, iteration to iteration | **Holds.** Upwind converges (1,046 outer iterations); the other two arms are bounded at the cap with the control's velocity step. The limiter diagnostic finds branch switches in every tail iteration (the "least" column is never zero) and no flux sign change, on the vertical faces beside the rack's east face and the etch chamber's west face in the gaps, along the faces' height rather than at the rack's corner alone (section 5.2). The switching is shown to accompany the cycle, not to cause it: test 47 froze the limiter's branches from outer 4,000 and the run failed positivity at 4,023, so the counterfactual that would settle cause could not be run. What the decision rests on is measured directly: upwind converges and moves no hotspot (section 6.1) |
| (b) The standard model converges on 160x60 and on 320x120 | **Fails on 320x120.** Converges on 160x60 (2,732); bounded on 320x120 from about outer 1,800 at a residual of 6e-8 to 1e-7 (section 5.3) |
| (c) RNG converges on 160x60; 320x120 no prediction | **Fails.** Bounded on both, as on 200x75, a 19-iteration cycle in the gap above return 2 (the bounded table) |
| (d) S1 and S3 pass the check on the first placement; S2 is carried into the nearest gap's return and needs moving | **Holds for S1 and S3, fails for S2.** S2 is carried into return 2 as predicted, but it passes the check through the surfaces clause (the litho tool's west face and the floor above the floor) and is not moved (the checks table) |
| (e) Between grids, for every source that passes: the largest deposition location is the same surface, at least three of five hotspots shared, the sensor order the same | **Not scorable as written** (no converged standard pair). On the upwind pair: section 6.1 |
| (f) Between variants, where both converge: the largest location the same surface, three of five hotspots shared | **Not scorable.** RNG converges on neither exact grid |

### 6.1 The comparative check on the pairs the records allow

The sentences point at the compare table's rows.

**Between grids, on the upwind pair (160x60 against 320x120), for every source and class:**
the largest deposition location is the same segment, not only the same surface (the "same
surface" column is yes in every row, and the two "largest" columns name the same segment); the
five hotspots are shared five of five for S1 and S2 in both classes and for S3 at 5
micrometres, and four of five for S3 at 0.5 micrometres (the floor segment at x 5.6, beside
return 4, drops out of the top five on the finer grid); and the sensor order above the floor is
the same (`near_door` alone for S1, no sensor for S2 and S3). Read against prediction (e)'s
three clauses, every clause holds on this pair for every source, with the sensor clause holding
on one sensor or none.

**Between the two schemes on 160x60 (the committed limited QUICK value against upwind for k and
eps), for every source and class:** the five hotspots are the same five, the largest is the
same segment, and the sensor order is the same. The change that converges the iteration does
not move a hotspot on this grid.

**Between variants:** no row; RNG converged on neither grid.

**What agrees and what does not.** The hotspots' locations and the sensor order hold between
grids; their amplitudes do not. The `near_door` reading falls 4.3 times from 160x60 to 320x120,
and S3's largest floor bin 6.3 times (test 47). The comparative claim (where particles collect,
and in what order) stands on this room; an absolute deposition figure would not. S2's agreement
is the weakest evidence of the three: its source sits on the litho tool's top, so most of its
hotspots are on that one surface and would agree on almost any grid.

### 6.2 Builder's predictions (section 3.2), scored

(a) held in both clauses, and the diagnostic's placement (faces beside the rack and in the
gaps, no sign changes) is as written; the halved `alpha_turbulence` did not shrink the tail's
velocity step by half (the cycle table's tail column). (b) was wrong in both halves: the
standard variant converged on 160x60, where I expected it bounded, and stalled on 320x120,
where I expected it to converge; the 160x60 count is below 200x75's, not above. (c) was wrong
on 160x60 (bounded, not converged) and right on 320x120. (d) held: S2 passed through the
surfaces clause and no sensor other than `near_door` and the one inside S3 reads above the
floor. (e) and (f) are scored in section 6.1 on the pairs that exist. The Courant check held
to 1e-5, better than the 1e-4 allowed.

## 7. What this implies

Stated as questions, as the prompt asks; no decision is taken here.

- **Does the product grid move to the exact pair?** 200x75 rounds fourteen stated positions to
  whole faces and 160x60 and 320x120 round none (section 5.1), so a comparison on the exact
  pair is of one room at two resolutions where every comparison so far was of two rooms. The
  committed scheme converges on 160x60 in fewer outer iterations than on 200x75; whether
  VAL-018's grid should be 160x60, with 320x120 as its refinement check, is the question the
  prompt names, and the answer depends on the next one.
- **Does the k and eps advection change?** The one change that converges the 80x30 cycle is
  upwind advection of k and eps, and the same change converges 320x120 where the committed
  scheme does not; the mechanism is the limiter switching branch in the shear layers beside the
  obstacle faces (section 5.2). ADR-012 C chose the limited QUICK value over upwind for its
  accuracy in thin shear layers and named upwind's numerical diffusion as the cost. The three
  converged rooms show the same core eddy viscosity and flow pattern under either scheme, and
  on 160x60 the two schemes give the same five hotspots for every source and class (the
  compare table, the "schemes" rows). Whether k and eps take upwind (the transport section's
  `advection_scheme` key, which the turbulence section does not yet carry), whether the
  limiter is frozen once the residual is small, or whether the cycle is accepted as a stopping
  question (a bound on the step in m/s beside the estimate, as prompt 46's section 7 asked) is
  a decision for Alex before step 8.
- **Does the comparative claim hold?** On the pair that converged, section 6.1 says what held
  and what did not, source by source. The sources the prompt placed all end in a return within
  a metre, so the hotspots are the footprints of plumes on their way to a return, and only one
  sensor reads any of them. The claim VAL-018 makes is about hotspots and rankings; a ranking
  of sensors needs sensors in plumes, and the sensor rows of the compare table are nearly
  empty. Whether the product's sensors move (into the gaps, onto the tops, before the returns)
  or whether VAL-018 is scored on deposition alone is the question prompt 46's section 7
  asked and this measurement sharpens.
- **Is RNG a product variant?** It converges on no exact grid and on no grid finer than 80x30
  under this iteration, in a clean 19-iteration cycle in the gap above return 2. The
  between-variants half of VAL-018's criterion cannot be scored until it does, or until VAL-020
  decides between the variants on a case that converges.
- **Does the near-wall treatment matter at 320x120?** A tenth of the wall nodes sit below the
  scalable floor there, a quarter of the domain-wall nodes (the converged table). ADR-012's
  second-cell overshoot and the floor's clamp both grow in reach as the first node moves toward
  the wall; whether the hotspot differences between the grids, where there are any, come from
  that is not measured here.

## 8. What this does not settle

- Whether the committed scheme converges on 320x120 at a smaller `cfl_number` or more momentum
  sweeps: one setting was run, as the prompt fixed it.
- Whether upwind advection of k and eps changes the converged k field where it matters (the
  shear layers): the core medians agree and the hotspots agree on 160x60; the shear layers'
  profiles are not compared.
- Whether the sources the prompt placed are the product's: all three sit in capture zones.
- RNG's cycle: located and timed, not probed.
- The 0.1 against 0.4 Courant check was run on one source and one room.
