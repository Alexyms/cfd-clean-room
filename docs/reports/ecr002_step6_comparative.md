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
