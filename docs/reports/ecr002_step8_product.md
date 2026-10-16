# ECR-002 Step 8: VAL-018, the Product Room and the Comparative Check

**Date:** 2026-10-15 (the machine's local date; the prompt dates Alex's decisions and the
orchestrator's predictions 2026-10-16)
**Tree:** branch `feature/ecr002-step8-product` from main at d1866b9 (prompt 55 merged). Nothing
under `src/` changes. Commit 0 records Alex's three decisions (section 2.1); commit A moves the
product configuration into `configs/clean_room_default.yaml` with the tests that read it; commit B
adds the harness family `product` and the preset `val018_320x120`; commit C adds the two layout
files and `tests/test_layouts.py`. Every flow run goes through the committed `StaggeredSolver` on
a committed configuration file regridded, and every particle march through the committed
`TransportSolver` with the converged flow handed to it.
**Instruments:** `docs/reports/probe56/` (committed after this section, before any run):
`coupled56.py` (the layouts on the exact grids, prompt 47's runner), `transport56.py` (prompt
47's march with the turbulent Schmidt number a parameter), `tables56.py` (the tables, the score,
the ranking, the margins, the hotspot comparison and the predictions' verdicts from the records),
`figures56.py` (the figures), `bounded56.py`, `assemble56.py` and the launcher `run56.sh`. Raw
output under `results/builder56/` (untracked). The figures are the PNGs beside this report.
**Order:** sections 1 to 4 were committed before any run, this file alone (commit D). The scripts
followed (commit E). Section 5 onward was written after the runs. Each run log's first line
carries its start time, and every record its `concurrent_processes`.
**Policy:** `docs/REVIEW_POLICY.md`, "What blocks". The numbers live in the tables `tables56.py`
generates from the records; the sentences point at the tables.

## 1. The question

Does the tool's purpose hold? `docs/SYSTEM.md` section 1 states it: the tool says where particles
collect in a room and how a layout change moves that, and does not defend absolute counts. ECR-002
criterion 10's note sets what VAL-018 asks for in this step: the deposition hotspots and the
ranking of layouts stable under grid refinement (160x60 against 320x120) and across the turbulent
Schmidt number's range (0.2 to 1.3). The variant clause is closed (decision 1, section 2.1). This
is the last measurement the Phase 3 gate needs.

The step is built on prompts 46 to 48. Prompt 47 found that 160x60 and 320x120 represent every
stated position of the room exactly, and that on that pair, with k and eps advected by upwind, the
standard variant converges on both grids and the five deposition hotspots of each of three sources
agree between the grids (`docs/reports/ecr002_step6_comparative.md`, sections 5.1, 5.3 and 6.1).
Prompt 48 made upwind the default and reproduced those runs bit for bit
(`docs/reports/ecr002_step6_variants.md`). What neither did is compare layouts: one room at two
resolutions, not two rooms. This step adds the layouts, defines a score before any run, and asks
whether the ranking the score gives survives the grid and the Schmidt number.

Three measurements:

1. The flows: three layouts on two grids, classified by step 5's rule; VAL-018's first clause (the
   room as it is on 320x120 stops by the rule within its cap) as a harness row.
2. The particles: prompt 47's three sources and two classes on every converged run, at Sc_t 0.7 on
   both grids and at 0.2 and 1.3 on 320x120.
3. The comparative check: the ranking, the hotspots and the margins.

## 2. Method (fixed before the runs)

### 2.1 The decisions this step runs under (Alex, recorded in commit 0)

1. **The product runs the standard variant.** VAL-020 (`docs/reports/ecr002_val020_impinging_jet.md`):
   RNG matched the measured turbulence at the impingement point where the standard model reads 26
   to 35% high, but RNG's free jet was worse (a potential core of 10.1 widths against the measured
   4.8; the standard model's 6.5 to 7.6) and RNG has not converged on the product's grids (prompts
   47 and 48). The caveat the product carries: turbulence over the equipment tops reads about 30%
   high, which inflates turbulent mixing and absolute deposition there by a similar factor in every
   layout, so layout comparisons stand while absolute tool-top deposition reads high. VAL-018's
   variant clause is closed. Recorded in ADR-012 decision 2 and section A, ECR-002 criterion 10
   and SYSTEM.md section 1.
2. **VAL-019 and VAL-020 are report-only validation rows.** Neither has a readable published
   standard k-epsilon reference on its case, and neither run settles cleanly (the step's
   reattachment cell and the jet's outlet cycle), so an allowance set now would be fitted to the
   result. Each row records its known departures and the harness rows pin them against regression.
   Recorded in ECR-002 criteria 8 and 13, ADR-012 G and the PROJECT_PLAN rows, status REPORTED.
3. **The product's configuration** is `configs/clean_room_default.yaml` (commit A): 320x120, the
   turbulence section (standard, `upwind`, `cfl_number` 0.25, `alpha_turbulence` 0.7, the k and eps
   solves to 1e-10 within 500 iterations, the inlet at 5% and 0.1 m), `stopping_rule:
   error_estimate` with `iteration_error_tol` 1e-6 and `mass_imbalance_tol` 1.25e-9 (ADR-011 G's
   formula, 1e-4 rho V_min / t_end, at the file's own cell volume), `momentum_sweeps` 10,
   `pressure_rtol` 1e-4, `max_simple_iter` 10,000, `alpha_velocity` 0.5 and the multigrid
   preconditioner stated, every value as prompts 46 to 48 ran it. The file names each key's
   source beside it, and `tests/test_config.py` pins them.

### 2.2 The room, the run and the classification

Every flow run is a layout file regridded (`domain.nx`, `domain.ny` set; on 160x60
`mass_imbalance_tol` recomputed by the same formula, 5.0e-9; on 320x120 the file's own value,
checked against the formula) and nothing else, through prompt 47's runner (`coupled47.run_room`,
its `control` arm, nothing substituted): the committed `StaggeredSolver` from rest, the cell of the
largest change of u, v and nu_t recorded every outer iteration, one BLAS thread per process. Each
record carries its full configuration mapping, the commit, the start time, the wall time, the
number of processes it shared the machine with (`concurrent_processes`), and per outer iteration
the residual, the CG count, the largest speed and its cell, the velocity and eddy-viscosity steps,
the rule's two estimates and the three continuity readings.

**Classification.** Step 5's rule in its order (`docs/reports/ecr002_step5_convergence.md`, section
2.1, with round 2's window amendment), as prompt 47 applied it: diverged (a speed past 100 m/s or
a non-finite field); converged (the solver's own stop, `error_estimate_and_continuity`); growing
(at the cap with the median largest speed over the last 500 iterations above 5 m/s); bounded and
not converged (at the cap under 5 m/s), with step 0's sub-classes falling, stalled or neither. A
`PositivityError` and a capped pressure correction are each their own outcome and a stop condition
(section 4).

**Machine and threads.** AMD Ryzen AI 9 HX 370 (12 cores, 24 threads), 64 GB, Windows 11, Python
3.13, `OPENBLAS_NUM_THREADS=1` in every process. Another builder (prompt 57, VAL-021) may run on
the machine at the same time, so every timing states `concurrent_processes` and no wall time here
is like for like with a single-process figure.

### 2.3 The layouts

Every position in every layout is a multiple of 0.05 m, so 160x60 (5 cm cells) and 320x120 (2.5 cm
cells) represent each layout exactly (`tests/test_layouts.py` checks every stated position). Each
layout is a configuration file beside the default, loaded the same way, generated from the default
so that nothing but the stated keys differs.

| Layout | File | The change | What the file states |
|---|---|---|---|
| L0 | `configs/clean_room_default.yaml` | none: the room as it is | the supply over x 0.5 to 7.5 m at 0.45 m/s (3.15 m^2/s per metre of depth); four floor returns stating no velocity, which share what the hood leaves at one face velocity (2.70 m^2/s over 2.20 m, 1.227 m/s); the hood at 0.5 m/s over 0.9 m (0.45 m^2/s) |
| L1 | `configs/clean_room_l1_return3_closed.yaml` | a subtle change a designer would weigh: floor return 3 (x 4.6 to 4.9 m, between the litho tool and the etch chamber) closed, its flow shared by returns 2 and 4, the total extract unchanged | `floor_return_3` removed; `floor_return_1` states 2.7 / 2.2 m/s, the share L0 resolves, so it keeps its flow (0.920 m^2/s); returns 2 and 4 state none and share the rest (1.780 m^2/s over 1.15 m, 1.547 m/s) |
| L2 | `configs/clean_room_l2_left_supply.yaml` | a deliberately poor design: the HEPA supply over the left half of the ceiling only, at the same total supply flow | `hepa_supply` over x 0.5 to 4.0 m (the default's margin at the wall kept) at 0.9 m/s, 3.15 m^2/s; the returns share the same remainder, so every outlet's flow is L0's |

The flows the boundary layer resolves on each grid are the layouts table of section 5.1, generated
from the configurations; `tests/test_layouts.py` asserts them on both exact grids. L1 and L2 keep
the sources, the sensors, the obstacles, the door, the hood and every solver and turbulence key.

### 2.4 Measurement 1: the flows

Six runs, each layout on 160x60 and 320x120, launched together, classified as above. For every
converged run: the three-panel figure of prompt 47 (streamlines coloured by speed, nu_t / nu on a
log scale, k, the room's boundaries marked as section 2.7 says) and one row of the converged table
with the outer count, the five readings at the stop, the wall time, the core's nu_t / nu (the
non-SOLID cells above the equipment tops, y > 2.0 m) and y+ at the wall nodes. For a bounded run,
`bounded44.py history`'s characterisation and `bounded44.py where`'s region.

**VAL-018's first clause** (ECR-002 criterion 10): L0 on 320x120 stops by
`error_estimate_and_continuity` within its cap. The harness row `val018_320x120`
(`scripts/benchmark.py --cases val018_320x120 --repeats 1`), the preset loading
`configs/clean_room_default.yaml` as it stands, records the stop, the outer count over the cap as
its value, the pass (the stop by the rule within the cap) and the readings; the probe's own L0
320x120 record is expected to stop at the same outer count, since both run the same file through
the same solver.

**Cost before the set.** A timing probe of 20 outer iterations of L0 on each grid runs first, two
processes at once. The projection at the cap for the six runs, plus the marches at prompt 47's
costs, is set against the eight-hour stop condition before anything else is launched; if it
exceeds eight hours, the estimate and a reduced set are reported first.

### 2.5 Measurement 2: the particles

On every converged run, prompt 47's three sources (its report, section 2.5), unchanged:

| Source | Where | Square (closed), m |
|---|---|---|
| S1 | an operator in the recirculation roll beside the door wall | [0.75, 0.85] x [1.15, 1.25] |
| S2 | a process emission just above the litho tool's top | [3.80, 3.90] x [2.00, 2.10] |
| S3 | an operator at the hood bench | [6.15, 6.25] x [1.15, 1.25] |

Each square is 0.1 m, its edges on cell faces of both grids: four cells on 160x60, sixteen on
320x120. The emission is Q = 1.0e4 particles per second per metre of depth for each class, spread
uniformly by volume over the square's cells, continuous, from a zero field. S2's square sits on the
litho tool's top, and the sensor `hood_entry` lies inside S3's square. In L2 the supply ends at
x 4.0 m, above S2's square.

**The march** is prompt 47's (its section 2.5), through `transport56.py`: the converged run's face
velocities frozen and its eddy viscosity passed to `solve_timestep` as `eddy_viscosity`; the two
classes 0.5 and 5 micrometres (indices 2 and 4); one dt for both classes, the smaller of their
`stable_dt`, rounded down so that a 1.0 s window is a whole number of steps; steady when over a
window the in-domain count's relative change and each sensor's change relative to the largest
sensor reading are below 1e-4 and the removal rate is within 1e-3 of Q; cap 600 s of simulated
time; the transport Courant number 0.4 (prompt 47 checked it against 0.1 to 1e-5). One change:
the turbulent Schmidt number is a parameter of the march, set in the configuration mapping the
march builds (`transport.turbulent_schmidt`), so that one converged flow serves every Sc_t. The
set: Sc_t 0.7 on every converged run; Sc_t 0.2 and 1.3 on the converged 320x120 runs as well.
Three sources, three layouts, two grids and three Schmidt numbers make up to 36 marches, each of
both classes.

**Reported per run, source and class** (prompt 47's quantities, the scoring `transport46.score`):
the stop and the steady time; the budget (the deposition and the outflow as shares of Q, the
residual); the sensors above the floor (1e-6 of the source-cell concentration) in descending order;
the deposition rate per named surface (`v_d A_f C_P` per depositing face, summed) and each
surface's share of the deposition; the five largest 0.2 m segments (every depositing surface cut
into 0.2 m bins, floors and tops along x, walls and sides along y, keyed by surface kind so grids
compare); the share of the emission exhausted (the outflow over Q) and deposited.

### 2.6 Measurement 3: the comparative check

**The score, defined here before any run.** For a layout on a grid at a Schmidt number, per class:

    score = sum over the three sources of (deposition rate on the litho tool's top
                                            + deposition rate on the etch chamber's top)

in particles per second per metre of depth at the march's steady state, the surfaces being the
named surfaces `litho_tool top` and `etch_chamber top` of `transport46.surface_names` (the
depositing faces along each obstacle's top edge). Lower is better: the process surfaces are where
a particle does harm. The sum over sources weights each source equally, by its emission Q. S2 sits
on the litho tool's top, so its deposition there dominates the score in every layout; the score
says how much of what the three sources emit ends on a process surface, not which source put it
there, and the per-source terms are tabled beside it.

**Ranking.** The order of L0, L1 and L2 by the score, ascending, on 160x60 and on 320x120 at
Sc_t 0.7, and on 320x120 at Sc_t 0.2, 0.7 and 1.3, for each class: ten orderings in all.

**Hotspots.** Per layout, source and class: the largest deposition location (the 0.2 m segment
with the largest rate) and the count of segments shared among the five largest, between the two
grids at Sc_t 0.7, and on 320x120 between Sc_t 0.7 and 0.2 and between 0.7 and 1.3. Counts and
segment names, not percentages of absolute values, as prompt 47 scored them.

**Margins.** For each ordering, the score difference between neighbouring layouts in the ranking
(first against second, second against third), set against the change the grid makes to each of
the two scores (the absolute difference between the layout's 320x120 and 160x60 scores at Sc_t
0.7) and, on 320x120, the change the Schmidt number makes (the larger of the absolute differences
between the layout's score at 0.2 and at 1.3 and its score at 0.7). An order between two layouts
is **robust** when the difference exceeds the grid's change to both scores and, on 320x120, the
Schmidt number's change to both; otherwise it holds by a hair and the table says so. Each
difference is also given relative to the larger of the two scores.

### 2.7 Figures

Every figure marks the supply (a blue bar along the ceiling with downward arrows, over the
layout's own range), the returns (red bars on the floor, those the layout has), the hood, the
door, the obstacles (grey), the sensors (labelled) and, on the concentration panels, the source
square; prompt 47's drawing code, unchanged, with this step's file names.

**The comparison figure**, one per class (`ecr002_step8_compare_0.5um.png` and
`ecr002_step8_compare_5um.png`): the three layouts side by side on 320x120 at Sc_t 0.7, three rows.
The flow (streamlines coloured by speed, one colour scale across the layouts); the plume of the
litho-top source S2 for that class (concentration on a log scale, one colour scale across the
layouts, down to a millionth of the largest peak); the deposition bars (every 0.2 m segment's
deposition summed over the three sources, as a bar standing on its surface, one bar scale across
the layouts, the scale stated in the title: the bar length that stands for the largest segment's
rate, in particles per second per metre of depth). Each deposition panel's title carries the
layout's score for that class. These are the product's first "this layout against that one"
picture.

### 2.8 The harness row and the records

The harness row `val018_320x120` is appended to `benchmarks/results.jsonl` with its
`concurrent_processes`. The probe's records (`results/builder56/*.json`, `*.npz`) are untracked;
the tables `tables56.py` generates from them are spliced into section 5 by `assemble56.py`
between marker comments, and every printed number is kept beside the records as
`tables_MODE.json`.

## 3. Predictions

### 3.1 Orchestrator (committed before any run; the prompt dates them 2026-10-16)

- (a) L0 on 320x120 stops by the rule within 10,000 iterations; L1 and L2 converge on both grids.
- (b) The ranking is L0 or L1 first and L2 last, on both grids and at every Sc_t, for both classes.
- (c) L0 and L1 are close: their score difference is smaller than the grid's change to either
  score for at least one class, so their order is not robust, and the report says so.
- (d) For every layout, source and class, the largest deposition location is the same segment on
  both grids and at Sc_t 0.7 and 1.3, and at least three of the five largest are shared; Sc_t 0.2
  moves more than the others.

What each outcome means, written with the predictions:

- (b) holds and (c) holds: the tool separates a good design from a poor one robustly, and calls
  near-equal designs near-equal rather than inventing an order. That is the comparative claim,
  stated at its honest resolution.
- (b) fails: the comparative claim does not yet hold on this room; the Phase 3 gate report says so
  and what would be needed.
- (d) fails at Sc_t 0.2 only: the hotspots depend on the turbulent diffusivity at the low end of
  its range, recorded as a sensitivity.

How (c) is scored: "smaller than the grid's change to either score" is read as smaller than the
grid's change to each of the two scores (the margin table's robustness rule of section 2.6); the
weaker reading, smaller than the change to at least one of them, is reported beside it.

### 3.2 Builder (2026-10-15, before any run; scored apart)

- (a) Holds. L0 on 320x120 stops near prompt 48's 4,472 outer iterations (the multigrid
  preconditioner moved no stop of prompt 54's set by more than 0.05%); L1 converges on both grids
  at counts within a few hundred of L0's; L2 converges on both but takes more outer iterations
  than L0 on each grid, since the supply at 0.9 m/s over half the ceiling drives a stronger
  recirculation on the right half.
- (b) Holds, with L2 last by a wide margin for both classes: the supply ends above S2's square,
  so the downflow at twice the velocity presses S2's plume onto the litho tool's top, and the
  right half of the room, with no supply above it, recirculates S3's plume over the etch chamber.
- (c) Holds for both classes: return 3 sits on the litho tool's east side, and prompt 47 found
  S2's plume leaving the top on the west, into return 2, so closing return 3 changes little at
  the surfaces the score reads. I expect L1 a little worse than L0 (the east gap's extraction gone
  leaves more of what reaches it to settle on the etch chamber's top), by less than the grid's
  change.
- (d) Holds between the grids and at Sc_t 1.3 for every row; at Sc_t 0.2 the largest location
  still agrees for S2 (its square is on the surface) and fewer than three of five are shared for
  at least one of S1 and S3, whose hotspots are a floor segment and a run of wall segments at
  rates orders apart.
- The timing probe projects the set under eight hours; the 320x120 marches at Sc_t 0.2 take the
  longest, the diffusivity at 3.5 times the 0.7 value spreading the plumes further before they
  reach a return.

## 4. Stop conditions (from the prompt)

- A layout needs a `src/` change: none does (commit C).
- L0 on 320x120 does not stop by the rule within its cap: the characterisation is finished,
  measurements 2 and 3 run on the converged runs only, and the report says so.
- A positivity error, a capped pressure correction, or an outlet that drifts: the run is its own
  outcome in the matrix, with the iteration and the cell, and the report stops there.
- The run set would exceed eight hours: the estimate and a reduced set are reported before
  anything is launched.

## 5. Results (written after the runs)

### 5.1 Order, cost and the layouts

<!-- tables56 timing begin -->
| Grid | Outer | Seconds per outer | CG per correction (mean) | k, eps iterations (mean) | Share of wall: pressure, turbulence | Minutes at the 10,000 cap | Concurrent processes |
|---|---|---|---|---|---|---|---|
| 160x60 | 20 | 0.044 | 19 | 6.8, 5.8 | 0.39, 0.30 | 7 | 2 |
| 320x120 | 20 | 0.148 | 23 | 12.9, 9.9 | 0.38, 0.33 | 25 | 2 |
<!-- tables56 timing end -->

The timing probes ran first (17:43, two processes). Under the multigrid preconditioner the
product grid costs a third of what prompt 47 measured per outer iteration (its 0.561 s against
the timing table's figure), so the projection at the cap was under half an hour for the longest
row alone and the set ran as planned: the six flow runs and the harness row were launched
together at 17:44:14 (the records' start times; the timing probes at 17:43:17), seven processes on twelve cores, the marches as rows converged (section
5.3). The wall times in the tables were taken beside other runs and are not like for like: the
machine has two kinds of core, and every 320x120 row shows the same pause, the stretch from outer
3,500 to 3,900 taking about 1,400 s where the stretches before it took about 75 s (the logs), a
machine-wide event between about 18:00 and 18:18, so the 320x120 rows' seconds per outer
iteration are inflated by about 2.5 against their first 3,500 iterations.

The full suite ran once at the last code commit (9cfe5e7) before this section was committed:
1,565 passed in 363 s (`results/builder56/full_suite.txt`); ruff and the system-map check pass at
every commit.

The layouts' flows as the boundary layer resolves them (the test of `tests/test_layouts.py`,
generated here from the configurations):

<!-- tables56 layouts begin -->
| Layout | Change | Supply: x range (m), velocity (m/s), flow (m^2/s) | Return 1: range, velocity, flow | Return 2 | Return 3 | Return 4 | Hood: y range, velocity, flow | Total extract (m^2/s) | Positions rounded on 160x60, 320x120 |
|---|---|---|---|---|---|---|---|---|---|
| L0 | the room as it is | 0.5 to 7.5, 0.45, 3.15 | 0.5 to 1.25, 1.227, 0.9205 | 2.4 to 2.95, 1.227, 0.675 | 4.6 to 4.9, 1.227, 0.3682 | 5.7 to 6.3, 1.227, 0.7364 | 0.9 to 1.8, 0.5 (stated), 0.45 | 3.15 | 0, 0 |
| L1 | return 3 closed, its flow to returns 2 and 4 | 0.5 to 7.5, 0.45, 3.15 | 0.5 to 1.25, 1.227 (stated), 0.9205 | 2.4 to 2.95, 1.547, 0.8511 | closed | 5.7 to 6.3, 1.547, 0.9285 | 0.9 to 1.8, 0.5 (stated), 0.45 | 3.15 | 0, 0 |
| L2 | the supply over x 0.5 to 4.0 m only, at 0.9 m/s | 0.5 to 4, 0.9, 3.15 | 0.5 to 1.25, 1.227, 0.9205 | 2.4 to 2.95, 1.227, 0.675 | 4.6 to 4.9, 1.227, 0.3682 | 5.7 to 6.3, 1.227, 0.7364 | 0.9 to 1.8, 0.5 (stated), 0.45 | 3.15 | 0, 0 |
<!-- tables56 layouts end -->

### 5.2 Measurement 1: the flows

<!-- tables56 matrix begin -->
| Run | Class | Stop | Outer (cap) | Residual: least (at), end | Readings at stop (a), (b), (c), (d), (e) | Holds from (a), (b), (c), (d), (e) | Largest speed at end (m/s), cell | Peak speed | CG per correction: mean, largest | Cap hits | k, eps iterations (mean) | Wall (s), per outer; processes | Face hash |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L0_160x60 | converged | error_estimate_and_continuity | 1,953 (10,000) | 8.27e-13 (1,953), 8.27e-13 | 1.4e-08, 7.9e-14, 9e-13, 5.5e-15, 1e-06 | 1,432, 388, 323, 1, 1,953 | 1.54, (0.525, 0.025) | 1.75 | 15, 21 | 0 | 18.0, 16.0 | 117, 0.060; 7 | d295ccb1a4c8f9e1 |
| L1_160x60 | converged | error_estimate_and_continuity | 1,938 (10,000) | 1.15e-12 (1,938), 1.15e-12 | 1.7e-08, 5.7e-14, 1.5e-12, 5.7e-15, 9.9e-07 | 1,506, 399, 333, 1, 1,938 | 1.87, (6.125, 0.575) | 2.34 | 17, 21 | 0 | 18.0, 15.1 | 119, 0.061; 7 | 757156af38d1d8da |
| L2_160x60 | bounded (neither) | max_simple_iter | 10,000 (10,000) | 1.25e-05 (8,660), 6.29e-05 | -, 2.7e-08, 4.1e-07, 2.6e-15, - | -, -, 9,997, 1, - | 3.44, (4.975, 1.575) | 4.05 | 16, 21 | 0 | 21.8, 18.8 | 613, 0.061; 7 | c36875d3757297e7 |
| L0_320x120 | converged | error_estimate_and_continuity | 4,389 (10,000) | 2.88e-13 (4,388), 2.96e-13 | 1.7e-08, 6.5e-14, 1.4e-12, 8.6e-15, 9.9e-07 | 3,530, 1,087, 582, 1, 4,389 | 1.69, (0.5125, 0.0125) | 2.13 | 20, 30 | 0 | 27.7, 23.1 | 2151, 0.490; 7 | cd0701fd8aab916d |
| L1_320x120 | converged | error_estimate_and_continuity | 4,394 (10,000) | 2.90e-13 (4,394), 2.90e-13 | 1.7e-08, 6.2e-14, 2.5e-12, 8e-15, 1e-06 | 3,536, 1,086, 712, 1, 4,394 | 1.89, (6.2125, 0.7375) | 2.68 | 20, 31 | 0 | 27.0, 22.1 | 2146, 0.488; 7 | b9e128770b1ac7dd |
| L2_320x120 | bounded (stalled) | max_simple_iter | 10,000 (10,000) | 2.83e-12 (10,000), 2.83e-12 | 4.8e-07, 7.5e-14, 3.5e-12, 6.1e-15, 2.6e-06 | 9,515, 1,737, 1,007, 1, - | 3.54, (6.1625, 0.7875) | 4.14 | 25, 32 | 0 | 30.8, 25.7 | 4018, 0.402; 7 | 32aac83bf1223d5c |
| L2_320x120_cap12000 | converged | error_estimate_and_continuity | 10,794 (12,000) | 1.36e-12 (10,794), 1.36e-12 | 1.9e-07, 7.9e-14, 2.7e-12, 6.1e-15, 1e-06 | 9,515, 1,737, 1,007, 1, 10,794 | 3.54, (6.1625, 0.7875) | 4.14 | 24, 32 | 0 | 30.8, 25.7 | 2285, 0.212; 5 | 549d6a7f4533662f |
<!-- tables56 matrix end -->

<!-- tables56 converged begin -->
| Run | Outer, stop | Readings at stop (a), (b), (c), (d), (e) | Holds from (a), (b), (c), (d), (e) | Wall (s); processes | Core nu_t / nu: median, 95th | y+ nodes | y+: median, least, largest | Share below 11.53: all; domain, tops, sides | Figure |
|---|---|---|---|---|---|---|---|---|---|
| L0_160x60 | 1,953, error_estimate_and_continuity | 1.4e-08, 7.9e-14, 9e-13, 5.5e-15, 1e-06 | 1,432, 388, 323, 1, 1,953 | 117; 7 | 17.1, 58 | 514 | 61.8, 4.08, 296 | 0.035; 0.094, 0, 0.011 | ecr002_step8_L0_160x60.png |
| L1_160x60 | 1,938, error_estimate_and_continuity | 1.7e-08, 5.7e-14, 1.5e-12, 5.7e-15, 9.9e-07 | 1,506, 399, 333, 1, 1,938 | 119; 7 | 18.2, 59.3 | 520 | 54.3, 0.157, 384 | 0.18; 0.15, 0, 0.25 | ecr002_step8_L1_160x60.png |
| L0_320x120 | 4,389, error_estimate_and_continuity | 1.7e-08, 6.5e-14, 1.4e-12, 8.6e-15, 9.9e-07 | 3,530, 1,087, 582, 1, 4,389 | 2151; 7 | 17, 56.2 | 1,028 | 34.3, 2.11, 167 | 0.096; 0.24, 0, 0.038 | ecr002_step8_L0_320x120.png |
| L1_320x120 | 4,394, error_estimate_and_continuity | 1.7e-08, 6.2e-14, 2.5e-12, 8e-15, 1e-06 | 3,536, 1,086, 712, 1, 4,394 | 2146; 7 | 17.9, 57.7 | 1,040 | 27.5, 0.207, 214 | 0.26; 0.3, 0, 0.31 | ecr002_step8_L1_320x120.png |
| L2_320x120_cap12000 | 10,794, error_estimate_and_continuity | 1.9e-07, 7.9e-14, 2.7e-12, 6.1e-15, 1e-06 | 9,515, 1,737, 1,007, 1, 10,794 | 2285; 5 | 58.7, 746 | 1,168 | 59.9, 6.25, 321 | 0.033; 0.076, 0, 0.0054 | ecr002_step8_L2_320x120_cap12000.png |
<!-- tables56 converged end -->

L0 and L1 stop by the rule on both grids, at nearly the same outer counts (the matrix: 1,953 and
1,938 on 160x60, 4,389 and 4,394 on 320x120), condition (e) the last to hold in every row. VAL-018's
first clause holds: L0 on 320x120 stops by `error_estimate_and_continuity` at 4,389 outer
iterations, and the harness row `val018_320x120` (`benchmarks/results.jsonl`, taken beside the
same six runs, `concurrent_processes` 7) stops at the same count with `passed` true. Prompt 48's
count on the same room under the diagonal preconditioner and the Jacobi k and eps solves was
4,472; the field is the same room (the converged table's core eddy viscosity and y+ against
prompt 48's converged table). L1's room differs from L0's where the change is: its largest speed
sits beside the hood bench (the matrix's cell column) where L0's is at the door's foot, and its
first nodes in the gap between the litho tool and the etch chamber, now without a return, fall
below the scalable floor (the converged table's y+ least and share columns).

L2 stops by the rule on neither grid, and the two failures are different things (the bounded
table). On 160x60 it is a cycle: the residual between 1e-5 and 9e-5 over the whole tail with a
period of 4 to 8 outer iterations, the velocity step 4e-3 m/s (the matrix's readings, against the
0.9 m/s scale), located in the cells beside the door wall below the door's top (x 0.025 to 0.075 m,
y 0.45 to 0.7 m; the bounded table's region label is bounded44's, which has no door-wall region,
and the cells it lists are these), where the supply's return flow along the left wall meets the
floor; conditions (a) and (e) refuse throughout and (b) holds only in the last iterations. On 320x120 the field is
still: the residual 2.8e-12 and falling (the drift column, -0.41 decades per thousand), the
velocity step 4e-10 m/s, conditions (a) to (d) holding from outer 9,515 at the latest, and only
condition (e) refusing, its estimate of the eddy-viscosity error 2.6e-6 against the 1e-6
tolerance and falling by 0.13% per iteration, which puts its stop about 800 iterations past the
cap. The largest changes sit in the supply row and the core above the equipment on the right
half of the room, under no supply.

**The supplement.** The prompt's stop condition covers L0, which stopped. For L2 the comparison
the step asks for needs a room that stopped by the rule, so one labelled run beyond the file's
configuration was made: L2 on 320x120 again from rest with `max_simple_iter` 12,000
(`L2_320x120_cap12000`), which stops by the rule at 10,794 outer iterations (the matrix), condition (e) the last to hold as in every converged row, the velocity step then 2e-10 m/s; it is the same room as the capped run, continued about 800
iterations further. Its marches, and the 160x60 cap field's marches at
Sc_t 0.7 (`--at-cap`, the label `_cap`), are the rows marked L2* in the tables of sections 5.3 and
5.4; a starred row is not a converged product-configuration run, and the ranking is read with
that in view.

<!-- tables56 bounded begin -->
| Run | Outer, tail | Residual: least, median, largest | Amplitude p95/p5 | Drift (log10 per 1,000) | Period (height); strongest (height) | Largest speed: least, largest; cells | Located: share by region (largest of u, v); cells | Largest nu_t change: share by region; cells; size (m^2/s) |
|---|---|---|---|---|---|---|---|---|
| L2_160x60 | 10,000, 2,000 | 1.25e-05, 4.58e-05, 8.98e-05 | 5.91 | 0.000522 | 4 (0.56); 8 (0.97) | 3.44, 3.44; (4.975, 1.575) | gap between equipment 1.00; (0.025, 0.525), (0.025, 0.475), (0.025, 0.675) | gap between equipment 1.00; (0.025, 0.475), (0.075, 0.475), (0.075, 0.525); 0.00069 |
| L2_320x120 | 10,000, 2,000 | 2.83e-12, 4.95e-12, 2.64e-11 | 6.88 | -0.411 | - (-); 630 (-0.00039) | 3.54, 3.54; (6.1625, 0.7875) | supply row 0.62; core above equipment 0.38; (6.6875, 2.9625), (7.2375, 2.7375), (7.6875, 2.1875) | core above equipment 1.00; (7.2625, 2.0875), (7.2375, 2.0875), (7.2125, 2.1375); 4.1e-11 |
<!-- tables56 bounded end -->

### 5.3 Measurement 2: the particles

<!-- tables56 transport begin -->
| Run, class (um) | Stop, t (s) | Last window: total, sensor change; removal / Q | Exhausted / Q, deposited / Q (faces; budget) | Budget residual | Source-cell C, floor | Sensors above the floor, in order (per m^3) | Wall (s); processes |
|---|---|---|---|---|---|---|---|
| L0_160x60_S1_sc0.7, 0.5 | steady, 84 | 1.6e-05, 9.6e-05; 0.99998 | 1, 1.818e-06 (1.818e-06) | 3.4e-07 | 7.69e+04, 0.0769 | near_door (400) | 171; 9 |
| L0_160x60_S1_sc0.7, 5 | steady, 84 | 1.6e-05, 9.6e-05; 0.99998 | 0.9998, 0.0001338 (0.0001338) | 3.4e-07 | 7.69e+04, 0.0769 | near_door (392) | 171; 9 |
| L0_160x60_S2_sc0.7, 0.5 | steady, 51 | 9.2e-05, 1.7e-06; 0.99933 | 0.999, 0.0003003 (0.0003003) | -3.1e-06 | 6.85e+05, 0.685 | none | 112; 9 |
| L0_160x60_S2_sc0.7, 5 | steady, 51 | 9e-05, 1.7e-06; 0.99935 | 0.9766, 0.02276 (0.02276) | -3.7e-06 | 6.84e+05, 0.684 | none | 112; 9 |
| L0_160x60_S3_sc0.7, 0.5 | steady, 10 | 9.6e-05, 3.6e-12; 0.9999 | 0.9999, 8.021e-08 (7.962e-08) | 2.5e-07 | 1.03e+05, 0.103 | none | 21; 9 |
| L0_160x60_S3_sc0.7, 5 | steady, 10 | 9.6e-05, 3.6e-12; 0.9999 | 0.9999, 5.965e-06 (5.922e-06) | 2.5e-07 | 1.03e+05, 0.103 | none | 21; 9 |
| L0_320x120_S1_sc0.2, 0.5 | steady, 59 | 1.4e-05, 8.8e-05; 0.99998 | 1, 3.399e-06 (3.399e-06) | 6e-07 | 7.77e+04, 0.0777 | near_door (599) | 1502; 20 |
| L0_320x120_S1_sc0.2, 5 | steady, 59 | 1.4e-05, 8.8e-05; 0.99998 | 0.9997, 0.000252 (0.000252) | 6e-07 | 7.77e+04, 0.0777 | near_door (588) | 1502; 20 |
| L0_320x120_S1_sc0.7, 0.5 | steady, 91 | 7.1e-06, 9.5e-05; 0.99999 | 1, 7.668e-07 (7.668e-07) | 2.1e-07 | 7.85e+04, 0.0785 | near_door (93.6) | 1774; 20 |
| L0_320x120_S1_sc0.7, 5 | steady, 91 | 7e-06, 9.5e-05; 0.99999 | 0.9999, 5.64e-05 (5.64e-05) | 2.1e-07 | 7.84e+04, 0.0784 | near_door (91) | 1774; 20 |
| L0_320x120_S1_sc1.3, 0.5 | steady, 112 | 2.4e-06, 9.6e-05; 1 | 1, 1.965e-07 (1.965e-07) | 4.5e-07 | 7.94e+04, 0.0794 | near_door (18.2) | 1879; 20 |
| L0_320x120_S1_sc1.3, 5 | steady, 112 | 2.4e-06, 9.5e-05; 1 | 1, 1.438e-05 (1.438e-05) | 4.5e-07 | 7.94e+04, 0.0794 | near_door (17.6) | 1879; 20 |
| L0_320x120_S2_sc0.2, 0.5 | steady, 33 | 9.4e-05, 9.9e-07; 0.99936 | 0.9991, 0.0002845 (0.0002845) | -3.1e-06 | 6.9e+05, 0.69 | none | 855; 20 |
| L0_320x120_S2_sc0.2, 5 | steady, 33 | 9.2e-05, 9.5e-07; 0.99939 | 0.9778, 0.02159 (0.02159) | -4.5e-06 | 6.88e+05, 0.688 | none | 855; 20 |
| L0_320x120_S2_sc0.7, 0.5 | steady, 55 | 9.6e-05, 1.2e-08; 0.99923 | 0.9989, 0.0003684 (0.0003684) | -5.6e-06 | 7.29e+05, 0.729 | none | 1156; 20 |
| L0_320x120_S2_sc0.7, 5 | steady, 55 | 9.4e-05, 1.3e-08; 0.99926 | 0.9713, 0.02792 (0.02792) | -5.5e-06 | 7.28e+05, 0.728 | none | 1156; 20 |
| L0_320x120_S2_sc1.3, 0.5 | steady, 70 | 9.7e-05, 1.2e-08; 0.99915 | 0.9988, 0.0003957 (0.0003957) | -8.9e-06 | 7.29e+05, 0.729 | none | 1314; 20 |
| L0_320x120_S2_sc1.3, 5 | steady, 70 | 9.6e-05, 1.2e-08; 0.99918 | 0.9692, 0.02998 (0.02998) | -8.7e-06 | 7.29e+05, 0.729 | none | 1314; 20 |
| L0_320x120_S3_sc0.2, 0.5 | steady, 8 | 4.7e-05, 2.5e-16; 0.99995 | 0.9999, 3.812e-07 (3.81e-07) | 5.9e-07 | 1.04e+05, 0.104 | none | 226; 20 |
| L0_320x120_S3_sc0.2, 5 | steady, 8 | 4.7e-05, 2.5e-16; 0.99995 | 0.9999, 2.847e-05 (2.846e-05) | 5.9e-07 | 1.04e+05, 0.104 | none | 226; 20 |
| L0_320x120_S3_sc0.7, 0.5 | steady, 4 | 6.5e-05, 5.1e-16; 0.99993 | 0.9999, 1.232e-08 (1.194e-08) | 1.3e-07 | 1.09e+05, 0.109 | none | 102; 20 |
| L0_320x120_S3_sc0.7, 5 | steady, 4 | 6.4e-05, 7.7e-16; 0.99993 | 0.9999, 9.266e-07 (8.992e-07) | 1.3e-07 | 1.09e+05, 0.109 | none | 102; 20 |
| L0_320x120_S3_sc1.3, 0.5 | steady, 3 | 7.9e-06, 2.5e-128; 0.99999 | 1, 8.787e-10 (7.797e-10) | 2.1e-07 | 1.1e+05, 0.11 | none | 75; 20 |
| L0_320x120_S3_sc1.3, 5 | steady, 3 | 7.9e-06, 2.2e-128; 0.99999 | 1, 6.644e-08 (5.92e-08) | 2.1e-07 | 1.1e+05, 0.11 | none | 75; 20 |
| L1_160x60_S1_sc0.7, 0.5 | steady, 84 | 1.7e-05, 9.6e-05; 0.99998 | 1, 1.849e-06 (1.849e-06) | 2.4e-07 | 7.68e+04, 0.0768 | near_door (417) | 183; 11 |
| L1_160x60_S1_sc0.7, 5 | steady, 84 | 1.7e-05, 9.6e-05; 0.99998 | 0.9998, 0.000136 (0.000136) | 2.4e-07 | 7.68e+04, 0.0768 | near_door (408) | 183; 11 |
| L1_160x60_S2_sc0.7, 0.5 | steady, 46 | 8.9e-05, 7.8e-05; 0.99953 | 0.9994, 0.0001423 (0.0001423) | -8.7e-07 | 2.36e+05, 0.236 | none | 104; 11 |
| L1_160x60_S2_sc0.7, 5 | steady, 46 | 8.8e-05, 6.5e-05; 0.99954 | 0.9887, 0.01083 (0.01083) | -8.9e-07 | 2.36e+05, 0.236 | none | 104; 11 |
| L1_160x60_S3_sc0.7, 0.5 | steady, 9 | 9.7e-05, 5e-12; 0.99992 | 0.9999, 7.655e-08 (7.618e-08) | 1.3e-07 | 7.51e+04, 0.0751 | none | 22; 11 |
| L1_160x60_S3_sc0.7, 5 | steady, 9 | 9.6e-05, 4.9e-12; 0.99993 | 0.9999, 5.696e-06 (5.668e-06) | 1.3e-07 | 7.5e+04, 0.075 | none | 22; 11 |
| L1_320x120_S1_sc0.2, 0.5 | steady, 59 | 1.4e-05, 8.8e-05; 0.99998 | 1, 3.442e-06 (3.442e-06) | 5.8e-07 | 7.77e+04, 0.0777 | near_door (623) | 1495; 20 |
| L1_320x120_S1_sc0.2, 5 | steady, 59 | 1.4e-05, 8.8e-05; 0.99998 | 0.9997, 0.0002551 (0.0002551) | 5.9e-07 | 7.76e+04, 0.0776 | near_door (611) | 1495; 20 |
| L1_320x120_S1_sc0.7, 0.5 | steady, 91 | 7.3e-06, 9.4e-05; 0.99999 | 1, 7.904e-07 (7.904e-07) | 2.1e-07 | 7.84e+04, 0.0784 | near_door (100) | 1793; 20 |
| L1_320x120_S1_sc0.7, 5 | steady, 91 | 7.2e-06, 9.4e-05; 0.99999 | 0.9999, 5.813e-05 (5.813e-05) | 2.1e-07 | 7.84e+04, 0.0784 | near_door (97.4) | 1793; 20 |
| L1_320x120_S1_sc1.3, 0.5 | steady, 112 | 2.6e-06, 9.6e-05; 1 | 1, 2.062e-07 (2.062e-07) | 4.6e-07 | 7.94e+04, 0.0794 | near_door (20) | 1864; 20 |
| L1_320x120_S1_sc1.3, 5 | steady, 112 | 2.5e-06, 9.6e-05; 1 | 1, 1.509e-05 (1.509e-05) | 4.6e-07 | 7.93e+04, 0.0793 | near_door (19.3) | 1864; 20 |
| L1_320x120_S2_sc0.2, 0.5 | steady, 249 | 2.9e-16, 9.9e-05; 1 | 0.9999, 0.000137 (0.000137) | -3.3e-06 | 2.39e+05, 0.239 | none | 4819; 20 |
| L1_320x120_S2_sc0.2, 5 | steady, 249 | 5.9e-16, 0.0001; 1 | 0.9896, 0.01044 (0.01044) | -3.3e-06 | 2.39e+05, 0.239 | none | 4819; 20 |
| L1_320x120_S2_sc0.7, 0.5 | steady, 360 | 0, 0.0001; 1 | 0.9998, 0.0001598 (0.0001598) | -2.1e-06 | 2.38e+05, 0.238 | none | 5381; 20 |
| L1_320x120_S2_sc0.7, 5 | steady, 360 | 0, 0.0001; 1 | 0.9878, 0.01218 (0.01218) | -2.1e-06 | 2.38e+05, 0.238 | none | 5381; 20 |
| L1_320x120_S2_sc1.3, 0.5 | steady, 437 | 0, 0.0001; 1 | 0.9998, 0.0001662 (0.0001662) | -9.5e-07 | 2.38e+05, 0.238 | none | 5848; 20 |
| L1_320x120_S2_sc1.3, 5 | steady, 419 | 0, 0.0001; 1 | 0.9873, 0.01266 (0.01266) | -9.4e-07 | 2.38e+05, 0.238 | none | 5848; 20 |
| L1_320x120_S3_sc0.2, 0.5 | steady, 7 | 5.3e-05, 9.6e-13; 0.99996 | 1, 3.887e-07 (3.885e-07) | 4.7e-07 | 7.51e+04, 0.0751 | none | 220; 20 |
| L1_320x120_S3_sc0.2, 5 | steady, 7 | 5.3e-05, 9.3e-13; 0.99996 | 0.9999, 2.902e-05 (2.901e-05) | 4.7e-07 | 7.51e+04, 0.0751 | none | 220; 20 |
| L1_320x120_S3_sc0.7, 0.5 | steady, 4 | 9.2e-05, 7e-16; 0.99993 | 0.9999, 1.23e-08 (1.205e-08) | 1.2e-07 | 7.92e+04, 0.0792 | none | 105; 20 |
| L1_320x120_S3_sc0.7, 5 | steady, 4 | 9.1e-05, 1.8e-16; 0.99993 | 0.9999, 9.207e-07 (9.033e-07) | 1.2e-07 | 7.91e+04, 0.0791 | none | 105; 20 |
| L1_320x120_S3_sc1.3, 0.5 | steady, 2 | 3.2e-05, 8e-13; 0.99998 | 1, 8.177e-10 (5.714e-10) | 8e-08 | 7.98e+04, 0.0798 | none | 52; 20 |
| L1_320x120_S3_sc1.3, 5 | steady, 2 | 3e-05, 7.9e-13; 0.99998 | 1, 6.192e-08 (4.353e-08) | 8e-08 | 7.97e+04, 0.0797 | none | 52; 20 |
| L2_160x60_S1_sc0.7_cap, 0.5 | steady, 3 | 3.6e-08, 3.2e-10; 1 | 1, 2.534e-10 (2.269e-10) | 2.9e-07 | 1.11e+05, 0.111 | none | 12; 5 |
| L2_160x60_S1_sc0.7_cap, 5 | steady, 3 | 5.7e-08, 3e-10; 1 | 1, 1.941e-08 (1.739e-08) | 2.9e-07 | 1.11e+05, 0.111 | none | 12; 5 |
| L2_160x60_S2_sc0.7_cap, 0.5 | steady, 35 | 3.3e-05, 9.8e-05; 0.99989 | 0.9998, 4.017e-05 (4.017e-05) | -2.9e-07 | 4.77e+04, 0.0477 | hood_entry (1.71e+03) | 151; 5 |
| L2_160x60_S2_sc0.7_cap, 5 | steady, 35 | 3.3e-05, 9.8e-05; 0.99989 | 0.9969, 0.003029 (0.003029) | -2.9e-07 | 4.77e+04, 0.0477 | hood_entry (1.69e+03) | 151; 5 |
| L2_160x60_S3_sc0.7_cap, 0.5 | steady, 38 | 9.7e-05, 5.3e-05; 0.9996 | 0.9996, 3.694e-06 (3.693e-06) | -2.4e-06 | 9.9e+04, 0.099 | none | 158; 5 |
| L2_160x60_S3_sc0.7_cap, 5 | steady, 38 | 9.7e-05, 5.3e-05; 0.9996 | 0.9994, 0.0002445 (0.0002445) | -2.4e-06 | 9.89e+04, 0.0989 | none | 158; 5 |
| L2_320x120_cap12000_S1_sc0.2, 0.5 | cap, 100 | 2.2e-09, 0.00014; 1 | 1, 9.089e-08 (9.089e-08) | 1.2e-06 | 1.11e+05, 0.111 | none | 1679; 1 |
| L2_320x120_cap12000_S1_sc0.2, 5 | cap, 100 | 2.1e-09, 0.00014; 1 | 1, 6.909e-06 (6.909e-06) | 1.2e-06 | 1.1e+05, 0.11 | none | 1679; 1 |
| L2_320x120_cap12000_S1_sc0.7, 0.5 | steady, 3 | 4.5e-08, 4.6e-13; 1 | 1, 1.584e-11 (1.447e-11) | 6.8e-08 | 1.13e+05, 0.113 | none | 89; 12 |
| L2_320x120_cap12000_S1_sc0.7, 5 | steady, 3 | 4.5e-08, 4e-13; 1 | 1, 1.211e-09 (1.107e-09) | 6.8e-08 | 1.13e+05, 0.113 | none | 89; 12 |
| L2_320x120_cap12000_S1_sc1.3, 0.5 | steady, 3 | 1.7e-08, 5e-14; 1 | 1, 4.403e-14 (4.25e-14) | 3.2e-08 | 1.13e+05, 0.113 | none | 86; 12 |
| L2_320x120_cap12000_S1_sc1.3, 5 | steady, 3 | 1.7e-08, 4.2e-14; 1 | 1, 3.369e-12 (3.253e-12) | 3.2e-08 | 1.13e+05, 0.113 | none | 86; 12 |
| L2_320x120_cap12000_S2_sc0.2, 0.5 | steady, 28 | 8e-05, 8.7e-05; 0.99974 | 0.9997, 3.995e-05 (3.996e-05) | -9.1e-09 | 4.91e+04, 0.0491 | hood_entry (1.76e+03) | 952; 12 |
| L2_320x120_cap12000_S2_sc0.2, 5 | steady, 28 | 7.8e-05, 9.1e-05; 0.99975 | 0.9967, 0.003015 (0.003015) | -7.5e-09 | 4.91e+04, 0.0491 | hood_entry (1.74e+03) | 952; 12 |
| L2_320x120_cap12000_S2_sc0.7, 0.5 | steady, 46 | 9.5e-06, 8.7e-05; 0.99997 | 0.9999, 3.924e-05 (3.923e-05) | -4.4e-07 | 4.94e+04, 0.0494 | hood_entry (738) | 1216; 12 |
| L2_320x120_cap12000_S2_sc0.7, 5 | steady, 45 | 1.1e-05, 0.0001; 0.99996 | 0.997, 0.002962 (0.002962) | -4.3e-07 | 4.95e+04, 0.0495 | hood_entry (725) | 1216; 12 |
| L2_320x120_cap12000_S2_sc1.3, 0.5 | steady, 59 | 6.3e-06, 9e-05; 0.99998 | 0.9999, 3.893e-05 (3.892e-05) | -1.3e-06 | 4.95e+04, 0.0495 | hood_entry (533) | 1468; 12 |
| L2_320x120_cap12000_S2_sc1.3, 5 | steady, 59 | 6.5e-06, 9.3e-05; 0.99998 | 0.997, 0.002939 (0.002939) | -1.3e-06 | 4.95e+04, 0.0495 | hood_entry (522) | 1468; 12 |
| L2_320x120_cap12000_S3_sc0.2, 0.5 | steady, 30 | 8.4e-05, 1.7e-05; 0.99974 | 0.9997, 3.298e-06 (3.297e-06) | -1.6e-06 | 7.83e+04, 0.0783 | none | 1123; 12 |
| L2_320x120_cap12000_S3_sc0.2, 5 | steady, 30 | 8.3e-05, 1.7e-05; 0.99974 | 0.9995, 0.0002244 (0.0002244) | -1.6e-06 | 7.82e+04, 0.0782 | none | 1123; 12 |
| L2_320x120_cap12000_S3_sc0.7, 0.5 | steady, 45 | 9.3e-05, 5.6e-05; 0.99953 | 0.9995, 3.94e-06 (3.94e-06) | -2.2e-06 | 1.29e+05, 0.129 | none | 1224; 12 |
| L2_320x120_cap12000_S3_sc0.7, 5 | steady, 45 | 9.3e-05, 5.6e-05; 0.99954 | 0.9993, 0.0002572 (0.0002572) | -2.2e-06 | 1.29e+05, 0.129 | none | 1224; 12 |
| L2_320x120_cap12000_S3_sc1.3, 0.5 | steady, 58 | 9.2e-05, 6.4e-05; 0.99939 | 0.9994, 5.033e-06 (5.032e-06) | -1.3e-06 | 1.6e+05, 0.16 | none | 1425; 12 |
| L2_320x120_cap12000_S3_sc1.3, 5 | steady, 58 | 9.2e-05, 6.3e-05; 0.99939 | 0.9991, 0.0003283 (0.0003283) | -1.3e-06 | 1.6e+05, 0.16 | none | 1425; 12 |

Deposition per surface, source S1, 0.5 um, share of the deposition total (surfaces with at least 1e-06 of it in some column):
| Surface | L0 160x60 Sc 0.7 | L0 320x120 Sc 0.2 | L0 320x120 Sc 0.7 | L0 320x120 Sc 1.3 | L1 160x60 Sc 0.7 | L1 320x120 Sc 0.2 | L1 320x120 Sc 0.7 | L1 320x120 Sc 1.3 | L2* 160x60 Sc 0.7 | L2* 320x120 Sc 0.2 | L2* 320x120 Sc 0.7 | L2* 320x120 Sc 1.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| floor 1.25-1.50 | 0.965 | 0.97 | 0.967 | 0.966 | 0.965 | 0.97 | 0.967 | 0.966 | 3.69e-06 | 1.19e-05 | 7.73e-10 | 1.17e-11 |
| server_rack west | 0.035 | 0.0303 | 0.0331 | 0.0337 | 0.0351 | 0.0304 | 0.0333 | 0.0338 | 2.28e-09 | 2.69e-08 | 9.1e-14 | 7.85e-17 |
| server_rack top | 1.01e-06 | 4.13e-06 | 7.64e-07 | 2.47e-07 | 1.03e-06 | 4.26e-06 | 7.85e-07 | 2.54e-07 | 4.37e-34 | 2.34e-25 | 1.95e-45 | 6.82e-56 |
| floor 0.00-0.50 | 6.73e-11 | 2.84e-09 | 5.29e-16 | 1.26e-18 | 6.32e-11 | 2.6e-09 | 4.72e-16 | 1.11e-18 | 1 | 0.996 | 1 | 1 |
| left wall | 1.28e-16 | 5.21e-14 | 4.16e-24 | 4.46e-29 | 1.19e-16 | 4.73e-14 | 3.69e-24 | 3.89e-29 | 9.89e-05 | 0.00372 | 3.7e-05 | 1.1e-06 |
| ceiling | 6.11e-43 | 6.5e-38 | 2.53e-69 | 5.48e-84 | 6.08e-43 | 6.11e-38 | 2.53e-69 | 5.61e-84 | 8.55e-27 | 1.31e-05 | 9.24e-36 | 5.62e-50 |

Deposition per surface, source S1, 5 um, share of the deposition total (surfaces with at least 1e-06 of it in some column):
| Surface | L0 160x60 Sc 0.7 | L0 320x120 Sc 0.2 | L0 320x120 Sc 0.7 | L0 320x120 Sc 1.3 | L1 160x60 Sc 0.7 | L1 320x120 Sc 0.2 | L1 320x120 Sc 0.7 | L1 320x120 Sc 1.3 | L2* 160x60 Sc 0.7 | L2* 320x120 Sc 0.2 | L2* 320x120 Sc 0.7 | L2* 320x120 Sc 1.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| floor 1.25-1.50 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 3.68e-06 | 1.19e-05 | 7.71e-10 | 1.17e-11 |
| server_rack west | 3.64e-05 | 3.13e-05 | 3.44e-05 | 3.5e-05 | 3.66e-05 | 3.15e-05 | 3.46e-05 | 3.51e-05 | 2.28e-12 | 2.69e-11 | 9.08e-17 | 7.83e-20 |
| server_rack top | 1.04e-06 | 4.19e-06 | 7.83e-07 | 2.53e-07 | 1.06e-06 | 4.32e-06 | 8.05e-07 | 2.6e-07 | 4.15e-34 | 2.18e-25 | 1.76e-45 | 6.08e-56 |
| floor 0.00-0.50 | 7.05e-11 | 2.95e-09 | 5.57e-16 | 1.34e-18 | 6.63e-11 | 2.7e-09 | 4.97e-16 | 1.17e-18 | 1 | 1 | 1 | 1 |
| left wall | 1.31e-19 | 5.3e-17 | 4.22e-27 | 4.52e-32 | 1.22e-19 | 4.82e-17 | 3.74e-27 | 3.94e-32 | 9.92e-08 | 3.74e-06 | 3.71e-08 | 1.1e-09 |

Deposition per surface, source S2, 0.5 um, share of the deposition total (surfaces with at least 1e-06 of it in some column):
| Surface | L0 160x60 Sc 0.7 | L0 320x120 Sc 0.2 | L0 320x120 Sc 0.7 | L0 320x120 Sc 1.3 | L1 160x60 Sc 0.7 | L1 320x120 Sc 0.2 | L1 320x120 Sc 0.7 | L1 320x120 Sc 1.3 | L2* 160x60 Sc 0.7 | L2* 320x120 Sc 0.2 | L2* 320x120 Sc 0.7 | L2* 320x120 Sc 1.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| litho_tool top | 0.953 | 0.96 | 0.955 | 0.949 | 0.925 | 0.934 | 0.919 | 0.906 | 0.78 | 0.781 | 0.819 | 0.83 |
| floor 2.95-3.20 | 0.0448 | 0.0378 | 0.0427 | 0.0484 | 0.0714 | 0.0618 | 0.077 | 0.0891 | 2.96e-33 | 1.31e-41 | 5.98e-57 | 1.4e-64 |
| litho_tool west | 0.00263 | 0.00251 | 0.00261 | 0.0028 | 0.00408 | 0.00398 | 0.00448 | 0.00493 | 6.88e-34 | 3.86e-42 | 1.3e-56 | 1.21e-63 |
| floor 4.50-4.60 | 9.24e-07 | 5.76e-05 | 4.25e-07 | 2.17e-08 | - | - | - | - | 0.0661 | 0.0638 | 0.0748 | 0.0841 |
| litho_tool east | 1.51e-07 | 1.09e-05 | 7.85e-08 | 3.78e-09 | 1.59e-19 | 1.51e-15 | 2.78e-25 | 1.39e-30 | 0.00875 | 0.00856 | 0.0101 | 0.0111 |
| etch_chamber top | 1.39e-22 | 1.98e-19 | 9.82e-34 | 1.66e-40 | 2.02e-17 | 5.08e-14 | 8.75e-24 | 4.05e-29 | 0.0696 | 0.0664 | 0.0315 | 0.0233 |
| floor 4.90-5.00 | 1.54e-10 | 3.66e-07 | 1.86e-13 | 2.92e-17 | - | - | - | - | 0.0603 | 0.0632 | 0.0583 | 0.0469 |
| etch_chamber west | 3.09e-12 | 7.36e-09 | 1.65e-15 | 1.51e-19 | 3.02e-19 | 8.89e-16 | 1.69e-25 | 9.13e-31 | 0.00655 | 0.00621 | 0.00445 | 0.00348 |
| floor 5.60-5.70 | 6.64e-26 | 4.66e-22 | 1.24e-37 | 2.82e-45 | 5.76e-19 | 1.32e-15 | 2.04e-25 | 9.86e-31 | 0.00309 | 0.00288 | 0.0013 | 0.00102 |
| hood_bench top | 1.84e-39 | 1.24e-32 | 1.41e-55 | 6.9e-67 | 3.48e-31 | 1.1e-24 | 2.34e-41 | 6.31e-50 | 0.00299 | 0.00507 | 9.49e-05 | 6.02e-06 |
| floor 6.30-6.40 | 3.43e-33 | 4.37e-27 | 1.74e-49 | 1.67e-60 | 3.98e-25 | 1.59e-19 | 1.9e-35 | 1.05e-43 | 0.00148 | 0.00216 | 0.000259 | 7.08e-05 |
| etch_chamber east | 1.05e-26 | 8.77e-23 | 2.17e-38 | 4.66e-46 | 9.08e-20 | 2.41e-16 | 3.5e-26 | 1.6e-31 | 0.000457 | 0.000442 | 0.000193 | 0.000145 |
| floor 7.60-8.00 | 3.46e-50 | 4.29e-41 | 6.1e-72 | 1.79e-87 | 2.21e-42 | 7.59e-31 | 2.36e-58 | 1.27e-71 | 0.000262 | 0.000536 | 8.46e-06 | 4.71e-07 |
| hood_bench west | 1.13e-34 | 8.56e-29 | 3.75e-51 | 4.07e-62 | 1.45e-26 | 3.76e-21 | 4.89e-37 | 3.08e-45 | 7.66e-05 | 0.000108 | 1.29e-05 | 3.52e-06 |
| right wall | 1.78e-51 | 4.61e-42 | 2.35e-73 | 5.63e-89 | 1.54e-43 | 1.1e-32 | 3.5e-60 | 1.89e-73 | 7.33e-06 | 2.31e-05 | 1.9e-07 | 8.31e-09 |
| hood_bench east | 1.78e-51 | 3.99e-42 | 2.25e-73 | 5.57e-89 | 1.52e-43 | 1.12e-32 | 3.55e-60 | 1.91e-73 | 5.9e-06 | 1.05e-05 | 1.54e-07 | 8.28e-09 |
| ceiling | 6.22e-71 | 2.3e-61 | 7.38e-108 | 7.56e-127 | 9.17e-70 | 4.64e-59 | 2.8e-108 | 4.99e-115 | 2.91e-06 | 1.96e-05 | 7.63e-08 | 6.87e-10 |

Deposition per surface, source S2, 5 um, share of the deposition total (surfaces with at least 1e-06 of it in some column):
| Surface | L0 160x60 Sc 0.7 | L0 320x120 Sc 0.2 | L0 320x120 Sc 0.7 | L0 320x120 Sc 1.3 | L1 160x60 Sc 0.7 | L1 320x120 Sc 0.2 | L1 320x120 Sc 0.7 | L1 320x120 Sc 1.3 | L2* 160x60 Sc 0.7 | L2* 320x120 Sc 0.2 | L2* 320x120 Sc 0.7 | L2* 320x120 Sc 1.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| litho_tool top | 0.956 | 0.963 | 0.958 | 0.952 | 0.929 | 0.938 | 0.923 | 0.911 | 0.794 | 0.794 | 0.832 | 0.843 |
| floor 2.95-3.20 | 0.0444 | 0.0374 | 0.042 | 0.0476 | 0.0712 | 0.0616 | 0.0767 | 0.0887 | 3.04e-33 | 1.34e-41 | 6.16e-57 | 1.44e-64 |
| litho_tool west | 2.62e-06 | 2.5e-06 | 2.59e-06 | 2.78e-06 | 4.1e-06 | 4e-06 | 4.49e-06 | 4.94e-06 | 7.07e-37 | 3.97e-45 | 1.33e-59 | 1.24e-66 |
| floor 4.50-4.60 | 8.8e-07 | 5.59e-05 | 3.96e-07 | 1.98e-08 | - | - | - | - | 0.0671 | 0.0647 | 0.0758 | 0.0853 |
| etch_chamber top | 1.29e-22 | 1.84e-19 | 8.74e-34 | 1.46e-40 | 1.91e-17 | 4.91e-14 | 8.12e-24 | 3.66e-29 | 0.0702 | 0.0669 | 0.0315 | 0.0232 |
| floor 4.90-5.00 | 1.46e-10 | 3.52e-07 | 1.71e-13 | 2.62e-17 | - | - | - | - | 0.0612 | 0.0641 | 0.059 | 0.0474 |
| floor 5.60-5.70 | 5.98e-26 | 4.21e-22 | 1.04e-37 | 2.32e-45 | 5.4e-19 | 1.27e-15 | 1.86e-25 | 8.74e-31 | 0.00311 | 0.00289 | 0.0013 | 0.00101 |
| hood_bench top | 1.6e-39 | 1.09e-32 | 1.16e-55 | 5.48e-67 | 3.18e-31 | 1.03e-24 | 2.08e-41 | 5.43e-50 | 0.00298 | 0.00507 | 9.33e-05 | 5.86e-06 |
| floor 6.30-6.40 | 3.05e-33 | 3.9e-27 | 1.44e-49 | 1.34e-60 | 3.69e-25 | 1.51e-19 | 1.71e-35 | 9.15e-44 | 0.00149 | 0.00217 | 0.000258 | 7e-05 |
| floor 7.60-8.00 | 3.39e-50 | 4.08e-41 | 5.2e-72 | 1.47e-87 | 2.32e-42 | 6.86e-31 | 1.96e-58 | 9.81e-72 | 0.00027 | 0.000544 | 8.36e-06 | 4.68e-07 |
| litho_tool east | 1.44e-10 | 1.07e-08 | 7.37e-11 | 3.47e-12 | 1.67e-22 | 1.67e-18 | 3.24e-28 | 1.64e-33 | 8.94e-06 | 8.74e-06 | 1.03e-05 | 1.13e-05 |
| etch_chamber west | 2.93e-15 | 7.12e-12 | 1.52e-18 | 1.36e-22 | 3.07e-22 | 9.17e-19 | 1.78e-28 | 9.86e-34 | 6.68e-06 | 6.33e-06 | 4.52e-06 | 3.53e-06 |

Deposition per surface, source S3, 0.5 um, share of the deposition total (surfaces with at least 1e-06 of it in some column):
| Surface | L0 160x60 Sc 0.7 | L0 320x120 Sc 0.2 | L0 320x120 Sc 0.7 | L0 320x120 Sc 1.3 | L1 160x60 Sc 0.7 | L1 320x120 Sc 0.2 | L1 320x120 Sc 0.7 | L1 320x120 Sc 1.3 | L2* 160x60 Sc 0.7 | L2* 320x120 Sc 0.2 | L2* 320x120 Sc 0.7 | L2* 320x120 Sc 1.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| floor 6.30-6.40 | 0.949 | 0.904 | 0.977 | 0.982 | 0.815 | 0.747 | 0.954 | 0.981 | 0.119 | 0.278 | 0.0443 | 0.00823 |
| hood_bench west | 0.0311 | 0.0256 | 0.0219 | 0.0184 | 0.0278 | 0.0225 | 0.0251 | 0.0162 | 0.00547 | 0.0118 | 0.00201 | 0.000377 |
| floor 5.60-5.70 | 0.0195 | 0.0699 | 0.000617 | 2.48e-05 | 0.155 | 0.227 | 0.0206 | 0.0024 | 0.719 | 0.557 | 0.806 | 0.842 |
| etch_chamber east | 0.000183 | 0.000843 | 2.95e-06 | 8.01e-08 | 0.00182 | 0.00365 | 0.000121 | 9.06e-06 | 0.131 | 0.101 | 0.147 | 0.149 |
| hood_bench top | 2.2e-06 | 8.19e-06 | 5.05e-07 | 4.95e-08 | 1.84e-06 | 8.44e-06 | 9.57e-07 | 1.26e-08 | 0.0236 | 0.0489 | 0.000742 | 1.56e-05 |
| floor 7.60-8.00 | 4e-33 | 1.08e-32 | 4.13e-97 | 7.11e-130 | 2.48e-35 | 1.26e-35 | 2.61e-92 | 7.99e-142 | 0.00149 | 0.00356 | 3.9e-05 | 6.73e-07 |
| right wall | 2.33e-18 | 1.66e-15 | 2.18e-39 | 5.37e-70 | 1.99e-19 | 3.04e-16 | 1.14e-32 | 6.38e-81 | 4.15e-05 | 0.000147 | 9e-07 | 1.26e-08 |
| hood_bench east | 5.14e-19 | 4.39e-16 | 2.28e-29 | 4.28e-52 | 4.7e-20 | 9.36e-17 | 2.3e-29 | 3.66e-60 | 3.59e-05 | 7.23e-05 | 7.67e-07 | 1.27e-08 |
| etch_chamber top | 4.25e-13 | 2e-11 | 2.76e-20 | 9.23e-32 | 1.32e-10 | 2.77e-09 | 1.61e-15 | 4.55e-33 | 1.4e-05 | 5.46e-05 | 1.02e-05 | 3.16e-06 |
| ceiling | 7.07e-58 | 1.56e-50 | 6.29e-133 | 6.34e-184 | 1.12e-60 | 1.2e-52 | 1.58e-129 | 2.13e-202 | 1.23e-05 | 0.000113 | 2.81e-07 | 8.22e-10 |

Deposition per surface, source S3, 5 um, share of the deposition total (surfaces with at least 1e-06 of it in some column):
| Surface | L0 160x60 Sc 0.7 | L0 320x120 Sc 0.2 | L0 320x120 Sc 0.7 | L0 320x120 Sc 1.3 | L1 160x60 Sc 0.7 | L1 320x120 Sc 0.2 | L1 320x120 Sc 0.7 | L1 320x120 Sc 1.3 | L2* 160x60 Sc 0.7 | L2* 320x120 Sc 0.2 | L2* 320x120 Sc 0.7 | L2* 320x120 Sc 1.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| floor 6.30-6.40 | 0.98 | 0.929 | 0.999 | 1 | 0.841 | 0.768 | 0.979 | 0.998 | 0.138 | 0.313 | 0.0519 | 0.00962 |
| floor 5.60-5.70 | 0.0199 | 0.0712 | 0.000618 | 2.47e-05 | 0.159 | 0.232 | 0.0209 | 0.0024 | 0.833 | 0.628 | 0.947 | 0.99 |
| hood_bench west | 3.23e-05 | 2.65e-05 | 2.25e-05 | 1.88e-05 | 2.89e-05 | 2.33e-05 | 2.58e-05 | 1.65e-05 | 6.37e-06 | 1.33e-05 | 2.37e-06 | 4.43e-07 |
| hood_bench top | 2.26e-06 | 8.34e-06 | 5.09e-07 | 4.95e-08 | 1.89e-06 | 8.62e-06 | 9.72e-07 | 1.27e-08 | 0.0271 | 0.0546 | 0.000856 | 1.79e-05 |
| etch_chamber east | 1.87e-07 | 8.62e-07 | 2.96e-09 | 7.98e-11 | 1.87e-06 | 3.75e-06 | 1.23e-07 | 9.08e-09 | 0.000153 | 0.000114 | 0.000174 | 0.000176 |
| floor 7.60-8.00 | 4.9e-33 | 1.29e-32 | 4.59e-97 | 7.76e-130 | 3.1e-35 | 1.5e-35 | 2.85e-92 | 8.66e-142 | 0.00176 | 0.00404 | 4.6e-05 | 7.88e-07 |
| etch_chamber top | 4.23e-13 | 1.98e-11 | 2.6e-20 | 8.38e-32 | 1.32e-10 | 2.78e-09 | 1.56e-15 | 4.31e-33 | 1.62e-05 | 6.12e-05 | 1.19e-05 | 3.71e-06 |

The five segments of largest deposition (surface | bin start), rate per s per m depth:
| Run, class (um) | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| L0_160x60_S1_sc0.7, 0.5 | floor | x 1.2 (0.0108) | floor | x 1.4 (0.0068) | server_rack west | y 0.0 (8.34e-05) | server_rack west | y 0.2 (7.78e-05) | server_rack west | y 0.4 (7.16e-05) |
| L0_160x60_S1_sc0.7, 5 | floor | x 1.2 (0.82) | floor | x 1.4 (0.518) | server_rack west | y 0.0 (6.4e-06) | server_rack west | y 0.2 (5.97e-06) | server_rack west | y 0.4 (5.49e-06) |
| L0_160x60_S2_sc0.7, 0.5 | litho_tool top | x 3.6 (1.13) | litho_tool top | x 3.4 (0.753) | litho_tool top | x 3.8 (0.495) | litho_tool top | x 3.2 (0.482) | floor | x 3.0 (0.108) |
| L0_160x60_S2_sc0.7, 5 | litho_tool top | x 3.6 (86.2) | litho_tool top | x 3.4 (57) | litho_tool top | x 3.8 (37.9) | litho_tool top | x 3.2 (36.3) | floor | x 3.0 (8.1) |
| L0_160x60_S3_sc0.7, 0.5 | floor | x 6.2 (0.000761) | floor | x 5.6 (1.56e-05) | hood_bench west | y 0.0 (8.49e-06) | hood_bench west | y 0.2 (6.35e-06) | hood_bench west | y 0.4 (5.06e-06) |
| L0_160x60_S3_sc0.7, 5 | floor | x 6.2 (0.0585) | floor | x 5.6 (0.00119) | hood_bench west | y 0.0 (6.56e-07) | hood_bench west | y 0.2 (4.9e-07) | hood_bench west | y 0.4 (3.9e-07) |
| L0_320x120_S1_sc0.2, 0.5 | floor | x 1.2 (0.0201) | floor | x 1.4 (0.0129) | server_rack west | y 0.0 (0.000159) | server_rack west | y 0.2 (0.000149) | server_rack west | y 0.4 (0.000133) |
| L0_320x120_S1_sc0.2, 5 | floor | x 1.2 (1.53) | floor | x 1.4 (0.986) | server_rack west | y 0.0 (1.22e-05) | server_rack west | y 0.2 (1.14e-05) | server_rack top | x 1.4 (1.06e-05) |
| L0_320x120_S1_sc0.7, 0.5 | floor | x 1.2 (0.00456) | floor | x 1.4 (0.00285) | server_rack west | y 0.0 (3.46e-05) | server_rack west | y 0.2 (3.2e-05) | server_rack west | y 0.4 (2.89e-05) |
| L0_320x120_S1_sc0.7, 5 | floor | x 1.2 (0.347) | floor | x 1.4 (0.217) | server_rack west | y 0.0 (2.65e-06) | server_rack west | y 0.2 (2.44e-06) | server_rack west | y 0.4 (2.21e-06) |
| L0_320x120_S1_sc1.3, 0.5 | floor | x 1.2 (0.00118) | floor | x 1.4 (0.00072) | server_rack west | y 0.0 (8.65e-06) | server_rack west | y 0.2 (7.94e-06) | server_rack west | y 0.4 (7.32e-06) |
| L0_320x120_S1_sc1.3, 5 | floor | x 1.2 (0.0893) | floor | x 1.4 (0.0545) | server_rack west | y 0.0 (6.58e-07) | server_rack west | y 0.2 (6.04e-07) | server_rack west | y 0.4 (5.57e-07) |
| L0_320x120_S2_sc0.2, 0.5 | litho_tool top | x 3.6 (1.07) | litho_tool top | x 3.4 (0.676) | litho_tool top | x 3.8 (0.513) | litho_tool top | x 3.2 (0.455) | floor | x 3.0 (0.0863) |
| L0_320x120_S2_sc0.2, 5 | litho_tool top | x 3.6 (81.7) | litho_tool top | x 3.4 (51.2) | litho_tool top | x 3.8 (39.3) | litho_tool top | x 3.2 (34.3) | floor | x 3.0 (6.48) |
| L0_320x120_S2_sc0.7, 0.5 | litho_tool top | x 3.6 (1.34) | litho_tool top | x 3.4 (1.01) | litho_tool top | x 3.2 (0.715) | litho_tool top | x 3.8 (0.451) | floor | x 3.0 (0.126) |
| L0_320x120_S2_sc0.7, 5 | litho_tool top | x 3.6 (102) | litho_tool top | x 3.4 (76.9) | litho_tool top | x 3.2 (53.8) | litho_tool top | x 3.8 (34.6) | floor | x 3.0 (9.42) |
| L0_320x120_S2_sc1.3, 0.5 | litho_tool top | x 3.6 (1.39) | litho_tool top | x 3.4 (1.12) | litho_tool top | x 3.2 (0.805) | litho_tool top | x 3.8 (0.44) | floor | x 3.0 (0.154) |
| L0_320x120_S2_sc1.3, 5 | litho_tool top | x 3.6 (106) | litho_tool top | x 3.4 (85.1) | litho_tool top | x 3.2 (60.5) | litho_tool top | x 3.8 (33.8) | floor | x 3.0 (11.4) |
| L0_320x120_S3_sc0.2, 0.5 | floor | x 6.2 (0.00344) | floor | x 5.6 (0.000266) | hood_bench west | y 0.0 (3.98e-05) | hood_bench west | y 0.2 (2.85e-05) | hood_bench west | y 0.4 (1.78e-05) |
| L0_320x120_S3_sc0.2, 5 | floor | x 6.2 (0.264) | floor | x 5.6 (0.0203) | hood_bench west | y 0.0 (3.07e-06) | hood_bench top | x 6.4 (2.37e-06) | hood_bench west | y 0.2 (2.2e-06) |
| L0_320x120_S3_sc0.7, 0.5 | floor | x 6.2 (0.00012) | hood_bench west | y 0.0 (1.29e-06) | hood_bench west | y 0.2 (7.32e-07) | hood_bench west | y 0.4 (3.93e-07) | hood_bench west | y 0.6 (2.24e-07) |
| L0_320x120_S3_sc0.7, 5 | floor | x 6.2 (0.00926) | floor | x 5.6 (5.72e-06) | hood_bench west | y 0.0 (9.97e-08) | hood_bench west | y 0.2 (5.65e-08) | hood_bench west | y 0.4 (3.03e-08) |
| L0_320x120_S3_sc1.3, 0.5 | floor | x 6.2 (8.63e-06) | hood_bench west | y 0.0 (8.29e-08) | hood_bench west | y 0.2 (4.33e-08) | hood_bench west | y 0.4 (2.23e-08) | hood_bench west | y 0.6 (1.09e-08) |
| L0_320x120_S3_sc1.3, 5 | floor | x 6.2 (0.000664) | floor | x 5.6 (1.64e-08) | hood_bench west | y 0.0 (6.42e-09) | hood_bench west | y 0.2 (3.35e-09) | hood_bench west | y 0.4 (1.72e-09) |
| L1_160x60_S1_sc0.7, 0.5 | floor | x 1.2 (0.0109) | floor | x 1.4 (0.00691) | server_rack west | y 0.0 (8.48e-05) | server_rack west | y 0.2 (7.92e-05) | server_rack west | y 0.4 (7.3e-05) |
| L1_160x60_S1_sc0.7, 5 | floor | x 1.2 (0.833) | floor | x 1.4 (0.527) | server_rack west | y 0.0 (6.5e-06) | server_rack west | y 0.2 (6.07e-06) | server_rack west | y 0.4 (5.59e-06) |
| L1_160x60_S2_sc0.7, 0.5 | litho_tool top | x 3.6 (0.458) | litho_tool top | x 3.4 (0.394) | litho_tool top | x 3.2 (0.305) | litho_tool top | x 3.8 (0.159) | floor | x 3.0 (0.0815) |
| L1_160x60_S2_sc0.7, 5 | litho_tool top | x 3.6 (35.1) | litho_tool top | x 3.4 (30.1) | litho_tool top | x 3.2 (23.2) | litho_tool top | x 3.8 (12.2) | floor | x 3.0 (6.19) |
| L1_160x60_S3_sc0.7, 0.5 | floor | x 6.2 (0.000624) | floor | x 5.6 (0.000119) | hood_bench west | y 0.0 (6.94e-06) | hood_bench west | y 0.2 (5.37e-06) | hood_bench west | y 0.4 (4.44e-06) |
| L1_160x60_S3_sc0.7, 5 | floor | x 6.2 (0.0479) | floor | x 5.6 (0.00906) | hood_bench west | y 0.0 (5.36e-07) | hood_bench west | y 0.2 (4.15e-07) | hood_bench west | y 0.4 (3.43e-07) |
| L1_320x120_S1_sc0.2, 0.5 | floor | x 1.2 (0.0203) | floor | x 1.4 (0.0131) | server_rack west | y 0.0 (0.000161) | server_rack west | y 0.2 (0.000151) | server_rack west | y 0.4 (0.000135) |
| L1_320x120_S1_sc0.2, 5 | floor | x 1.2 (1.55) | floor | x 1.4 (0.999) | server_rack west | y 0.0 (1.24e-05) | server_rack west | y 0.2 (1.16e-05) | server_rack top | x 1.4 (1.1e-05) |
| L1_320x120_S1_sc0.7, 0.5 | floor | x 1.2 (0.0047) | floor | x 1.4 (0.00294) | server_rack west | y 0.0 (3.57e-05) | server_rack west | y 0.2 (3.3e-05) | server_rack west | y 0.4 (2.99e-05) |
| L1_320x120_S1_sc0.7, 5 | floor | x 1.2 (0.358) | floor | x 1.4 (0.224) | server_rack west | y 0.0 (2.73e-06) | server_rack west | y 0.2 (2.52e-06) | server_rack west | y 0.4 (2.28e-06) |
| L1_320x120_S1_sc1.3, 0.5 | floor | x 1.2 (0.00124) | floor | x 1.4 (0.000755) | server_rack west | y 0.0 (9.08e-06) | server_rack west | y 0.2 (8.34e-06) | server_rack west | y 0.4 (7.7e-06) |
| L1_320x120_S1_sc1.3, 5 | floor | x 1.2 (0.0937) | floor | x 1.4 (0.0572) | server_rack west | y 0.0 (6.91e-07) | server_rack west | y 0.2 (6.35e-07) | server_rack west | y 0.4 (5.86e-07) |
| L1_320x120_S2_sc0.2, 0.5 | litho_tool top | x 3.6 (0.456) | litho_tool top | x 3.4 (0.383) | litho_tool top | x 3.2 (0.302) | litho_tool top | x 3.8 (0.139) | floor | x 3.0 (0.068) |
| L1_320x120_S2_sc0.2, 5 | litho_tool top | x 3.6 (35) | litho_tool top | x 3.4 (29.3) | litho_tool top | x 3.2 (23.1) | litho_tool top | x 3.8 (10.7) | floor | x 3.0 (5.17) |
| L1_320x120_S2_sc0.7, 0.5 | litho_tool top | x 3.6 (0.478) | litho_tool top | x 3.4 (0.456) | litho_tool top | x 3.2 (0.399) | litho_tool top | x 3.8 (0.135) | floor | x 3.0 (0.0988) |
| L1_320x120_S2_sc0.7, 5 | litho_tool top | x 3.6 (36.7) | litho_tool top | x 3.4 (34.9) | litho_tool top | x 3.2 (30.5) | litho_tool top | x 3.8 (10.3) | floor | x 3.0 (7.49) |
| L1_320x120_S2_sc1.3, 0.5 | litho_tool top | x 3.6 (0.48) | litho_tool top | x 3.4 (0.468) | litho_tool top | x 3.2 (0.423) | litho_tool top | x 3.8 (0.134) | floor | x 3.0 (0.119) |
| L1_320x120_S2_sc1.3, 5 | litho_tool top | x 3.6 (36.9) | litho_tool top | x 3.4 (35.9) | litho_tool top | x 3.2 (32.3) | litho_tool top | x 3.8 (10.3) | floor | x 3.0 (9.01) |
| L1_320x120_S3_sc0.2, 0.5 | floor | x 6.2 (0.0029) | floor | x 5.6 (0.000883) | hood_bench west | y 0.0 (3.38e-05) | hood_bench west | y 0.2 (2.54e-05) | hood_bench west | y 0.4 (1.68e-05) |
| L1_320x120_S3_sc0.2, 5 | floor | x 6.2 (0.223) | floor | x 5.6 (0.0674) | hood_bench west | y 0.0 (2.61e-06) | hood_bench top | x 6.4 (2.5e-06) | hood_bench west | y 0.2 (1.96e-06) |
| L1_320x120_S3_sc0.7, 0.5 | floor | x 6.2 (0.000117) | floor | x 5.6 (2.54e-06) | hood_bench west | y 0.0 (1.29e-06) | hood_bench west | y 0.2 (8.3e-07) | hood_bench west | y 0.4 (5.18e-07) |
| L1_320x120_S3_sc0.7, 5 | floor | x 6.2 (0.00901) | floor | x 5.6 (0.000192) | hood_bench west | y 0.0 (9.96e-08) | hood_bench west | y 0.2 (6.41e-08) | hood_bench west | y 0.4 (3.99e-08) |
| L1_320x120_S3_sc1.3, 0.5 | floor | x 6.2 (8.02e-06) | hood_bench west | y 0.0 (7.13e-08) | hood_bench west | y 0.2 (3.69e-08) | floor | x 5.6 (1.96e-08) | hood_bench west | y 0.4 (1.72e-08) |
| L1_320x120_S3_sc1.3, 5 | floor | x 6.2 (0.000618) | floor | x 5.6 (1.48e-06) | hood_bench west | y 0.0 (5.52e-09) | hood_bench west | y 0.2 (2.86e-09) | hood_bench west | y 0.4 (1.33e-09) |
| L2_160x60_S1_sc0.7_cap, 0.5 | floor | x 0.4 (1.77e-06) | floor | x 0.2 (6.8e-07) | floor | x 0.0 (8.03e-08) | left wall | y 0.0 (1.17e-10) | left wall | y 0.2 (8.52e-11) |
| L2_160x60_S1_sc0.7_cap, 5 | floor | x 0.4 (0.000136) | floor | x 0.2 (5.2e-05) | floor | x 0.0 (6.18e-06) | floor | x 1.2 (6.49e-10) | floor | x 1.4 (6.53e-11) |
| L2_160x60_S2_sc0.7_cap, 0.5 | litho_tool top | x 4.0 (0.0954) | litho_tool top | x 4.2 (0.093) | litho_tool top | x 3.8 (0.0799) | litho_tool top | x 4.4 (0.0451) | floor | x 4.4 (0.0266) |
| L2_160x60_S2_sc0.7_cap, 5 | litho_tool top | x 4.0 (7.32) | litho_tool top | x 4.2 (7.14) | litho_tool top | x 3.8 (6.13) | litho_tool top | x 4.4 (3.45) | floor | x 4.4 (2.03) |
| L2_160x60_S3_sc0.7_cap, 0.5 | floor | x 5.6 (0.0265) | floor | x 6.2 (0.00441) | etch_chamber east | y 1.6 (0.000575) | etch_chamber east | y 1.4 (0.000568) | etch_chamber east | y 1.2 (0.000555) |
| L2_160x60_S3_sc0.7_cap, 5 | floor | x 5.6 (2.04) | floor | x 6.2 (0.337) | hood_bench top | x 6.4 (0.0233) | hood_bench top | x 6.6 (0.0124) | hood_bench top | x 6.8 (0.00924) |
| L2_320x120_cap12000_S1_sc0.2, 0.5 | floor | x 0.2 (0.000353) | floor | x 0.4 (0.000305) | floor | x 0.0 (0.000248) | left wall | y 0.0 (1.39e-06) | left wall | y 0.2 (9.91e-07) |
| L2_320x120_cap12000_S1_sc0.2, 5 | floor | x 0.2 (0.027) | floor | x 0.4 (0.0233) | floor | x 0.0 (0.0189) | floor | x 1.2 (6.31e-07) | floor | x 1.4 (1.92e-07) |
| L2_320x120_cap12000_S1_sc0.7, 0.5 | floor | x 0.4 (1.19e-07) | floor | x 0.2 (3.64e-08) | floor | x 0.0 (3.37e-09) | left wall | y 0.0 (3.38e-12) | left wall | y 0.2 (2.08e-12) |
| L2_320x120_cap12000_S1_sc0.7, 5 | floor | x 0.4 (9.07e-06) | floor | x 0.2 (2.78e-06) | floor | x 0.0 (2.58e-07) | left wall | y 0.0 (2.6e-13) | left wall | y 0.2 (1.59e-13) |
| L2_320x120_cap12000_S1_sc1.3, 0.5 | floor | x 0.4 (4e-10) | floor | x 0.2 (3.9e-11) | floor | x 0.0 (1.21e-12) | left wall | y 0.0 (3.04e-16) | left wall | y 0.2 (1.63e-16) |
| L2_320x120_cap12000_S1_sc1.3, 5 | floor | x 0.4 (3.06e-08) | floor | x 0.2 (2.98e-09) | floor | x 0.0 (9.39e-11) | left wall | y 0.0 (2.34e-17) | left wall | y 0.2 (1.24e-17) |
| L2_320x120_cap12000_S2_sc0.2, 0.5 | litho_tool top | x 4.0 (0.0961) | litho_tool top | x 4.2 (0.0935) | litho_tool top | x 3.8 (0.077) | litho_tool top | x 4.4 (0.0452) | floor | x 4.4 (0.0255) |
| L2_320x120_cap12000_S2_sc0.2, 5 | litho_tool top | x 4.0 (7.39) | litho_tool top | x 4.2 (7.15) | litho_tool top | x 3.8 (5.9) | litho_tool top | x 4.4 (3.49) | floor | x 4.4 (1.95) |
| L2_320x120_cap12000_S2_sc0.7, 0.5 | litho_tool top | x 4.0 (0.0985) | litho_tool top | x 4.2 (0.0972) | litho_tool top | x 3.8 (0.0774) | litho_tool top | x 4.4 (0.0482) | floor | x 4.4 (0.0294) |
| L2_320x120_cap12000_S2_sc0.7, 5 | litho_tool top | x 4.0 (7.56) | litho_tool top | x 4.2 (7.45) | litho_tool top | x 3.8 (5.93) | litho_tool top | x 4.4 (3.7) | floor | x 4.4 (2.25) |
| L2_320x120_cap12000_S2_sc1.3, 0.5 | litho_tool top | x 4.0 (0.0988) | litho_tool top | x 4.2 (0.0981) | litho_tool top | x 3.8 (0.0775) | litho_tool top | x 4.4 (0.0486) | floor | x 4.4 (0.0328) |
| L2_320x120_cap12000_S2_sc1.3, 5 | litho_tool top | x 4.0 (7.58) | litho_tool top | x 4.2 (7.52) | litho_tool top | x 3.8 (5.94) | litho_tool top | x 4.4 (3.72) | floor | x 4.4 (2.51) |
| L2_320x120_cap12000_S3_sc0.2, 0.5 | floor | x 5.6 (0.0184) | floor | x 6.2 (0.00916) | hood_bench top | x 6.4 (0.000565) | etch_chamber east | y 1.4 (0.000407) | etch_chamber east | y 1.6 (0.0004) |
| L2_320x120_cap12000_S3_sc0.2, 5 | floor | x 5.6 (1.41) | floor | x 6.2 (0.702) | hood_bench top | x 6.4 (0.043) | hood_bench top | x 6.6 (0.0244) | hood_bench top | x 6.8 (0.0176) |
| L2_320x120_cap12000_S3_sc0.7, 0.5 | floor | x 5.6 (0.0317) | floor | x 6.2 (0.00175) | etch_chamber east | y 1.6 (0.000688) | etch_chamber east | y 1.4 (0.000676) | etch_chamber east | y 1.2 (0.000658) |
| L2_320x120_cap12000_S3_sc0.7, 5 | floor | x 5.6 (2.44) | floor | x 6.2 (0.134) | hood_bench top | x 6.4 (0.00101) | hood_bench top | x 6.6 (0.000395) | hood_bench top | x 6.8 (0.000261) |
| L2_320x120_cap12000_S3_sc1.3, 0.5 | floor | x 5.6 (0.0424) | etch_chamber east | y 1.6 (0.000876) | etch_chamber east | y 1.4 (0.00086) | etch_chamber east | y 1.8 (0.000845) | etch_chamber east | y 1.2 (0.000839) |
| L2_320x120_cap12000_S3_sc1.3, 5 | floor | x 5.6 (3.25) | floor | x 6.2 (0.0316) | etch_chamber east | y 1.6 (6.76e-05) | etch_chamber east | y 1.4 (6.64e-05) | etch_chamber east | y 1.8 (6.52e-05) |
<!-- tables56 transport end -->

Thirty-six marches: prompt 47's three sources at Sc_t 0.7 on the four converged
product-configuration rooms and at 0.2 and 1.3 on the two converged 320x120 rooms (24), and the
L2* rows (the nine on the cap-12,000 room, the three on the 160x60 cap field at 0.7). Every march
but one reaches the steady rule before the 600 s cap; the three S2 marches on L1 320x120 take 249
to 437 s of simulated time to do so, their field steady long before (the last-window column's
total change zero) while the sensor clause waited on readings at the floor of the scheme. The one
exception is S1 on the L2* 320x120 room at Sc_t 0.2, a source whose deposition there is 1e-7 of
its emission and whose sensors read noise: its first march had crawled to 70 s of simulated time
in 40 minutes with the sensor clause at 1.5e-3 and the total change at 2e-8, so it was stopped and
run again with the cap at 100 s (`--cap-seconds 100`), where it ends as "cap" with the total
change 2e-9 and the sensor clause at 1.4e-4 (the transport table); its score term is below 1e-35 of the
others and its hotspots are read with that in view. The budget closes in every row (the residual
column), and the faces' deposition agrees with the budget's increment.

Where the emission goes, per source. S1 (the operator beside the door) and S3 (the operator at
the hood bench) are captured by returns 1 and 4 in L0 and L1 as prompt 47 found, and their
deposition on the process surfaces is nil in every layout (the score table's S1 and S3 columns,
1e-14 of the emission and below); in L2 S1's whole plume descends the door wall to return 1 (its
hotspots the floor at x 0.0 to 0.4 m) and S3's crosses the etch chamber's east face to the floor
at x 5.6 m, both still nil on the tops. S2 (the process emission on the litho tool's top) is the
score. Its deposition on that top is 92 to 96% of what it deposits in L0 and L1 and 78 to 83% in
L2 (the S2 per-surface tables), the rest on the gap floor it leaves by: the west gap (floor 2.95
to 3.20 m, toward return 2) in L0 and L1, the east gap (floor 4.50 to 5.00 m) and the etch
chamber's top in L2. What each layout changes is the concentration the source cell sits in (the
transport table's source-cell column, 320x120 at 0.7): 7.3e5 per m^3 in L0, 2.4e5 in L1, 4.9e4 in
L2. In L1 the east gap no longer draws, so the supply air over the litho tool sweeps west over
the source at return 2's larger share and dilutes it; in L2 the supply's edge at x 4.0 m, at twice
the velocity, sweeps the source east across the etch chamber to the hood, which exhausts it (the
exhausted column, 0.9999), past the hood entry: `hood_entry` reads S2's plume in L2 at 738 per
m^3 (the sensors column) where no sensor reads S2 in L0 or L1. In every layout almost all of each
emission is exhausted (the exhausted and deposited columns: S2's deposited share 3.7e-4 of Q at
0.5 um and 2.8% at 5 um in L0, 1.6e-4 and 1.2% in L1, 3.9e-5 and 0.3% in L2).

### 5.4 Measurement 3: the comparative check

<!-- tables56 score begin -->
L2* is supplementary: on 320x120 the rerun with the cap at 12,000, which stops by the rule; on 160x60 the field at the cap, a cycle the rule does not stop on.

The layout score, 0.5 um: deposition on the litho tool's top and the etch chamber's top, particles per s per m depth, per source and summed (lower is better):
| Layout, grid, Sc_t | S1: litho top, etch top, sum | S2: litho top, etch top, sum | S3: litho top, etch top, sum | Score |
|---|---|---|---|---|
| L0, 160x60, 0.7 | 3.296e-54, 1.706e-97, 3.296e-54 | 2.86, 4.16e-22, 2.86 | 1.154e-50, 3.412e-16, 3.412e-16 | 2.86 |
| L0, 320x120, 0.2 | 1.021e-50, 3.528e-90, 1.021e-50 | 2.731, 5.639e-19, 2.731 | 1.085e-44, 7.632e-14, 7.632e-14 | 2.731 |
| L0, 320x120, 0.7 | 2.358e-77, 1.672e-145, 2.358e-77 | 3.517, 3.617e-33, 3.517 | 1.041e-90, 3.406e-24, 3.406e-24 | 3.517 |
| L0, 320x120, 1.3 | 3.893e-92, 1.422e-174, 3.893e-92 | 3.755, 6.572e-40, 3.755 | 1.209e-122, 8.109e-37, 8.109e-37 | 3.755 |
| L1, 160x60, 0.7 | 4.11e-53, 3.073e-94, 4.11e-53 | 1.316, 2.873e-17, 1.316 | 3.89e-52, 1.007e-13, 1.007e-13 | 1.316 |
| L1, 320x120, 0.2 | 3.214e-49, 7.15e-90, 3.214e-49 | 1.28, 6.957e-14, 1.28 | 1.037e-50, 1.078e-11, 1.078e-11 | 1.28 |
| L1, 320x120, 0.7 | 4.141e-75, 1.889e-140, 4.141e-75 | 1.468, 1.398e-23, 1.468 | 6.853e-87, 1.983e-19, 1.983e-19 | 1.468 |
| L1, 320x120, 1.3 | 1.637e-89, 7.733e-168, 1.637e-89 | 1.506, 6.738e-29, 1.506 | 2.117e-125, 3.717e-38, 3.717e-38 | 1.506 |
| L2*, 160x60, 0.7 | 2.822e-64, 3.637e-114, 2.822e-64 | 0.3134, 0.02797, 0.3413 | 3.672e-34, 5.171e-07, 5.171e-07 | 0.3413 |
| L2*, 320x120, 0.2 | 4.224e-36, 6.403e-38, 4.288e-36 | 0.3119, 0.02653, 0.3384 | 2.594e-29, 1.799e-06, 1.799e-06 | 0.3384 |
| L2*, 320x120, 0.7 | 6.064e-98, 3.592e-119, 6.064e-98 | 0.3214, 0.01235, 0.3337 | 1.157e-47, 4.016e-07, 4.016e-07 | 0.3337 |
| L2*, 320x120, 1.3 | 1.438e-115, 5.691e-129, 1.438e-115 | 0.323, 0.009062, 0.3321 | 1.941e-58, 1.591e-07, 1.591e-07 | 0.3321 |

The layout score, 5 um: deposition on the litho tool's top and the etch chamber's top, particles per s per m depth, per source and summed (lower is better):
| Layout, grid, Sc_t | S1: litho top, etch top, sum | S2: litho top, etch top, sum | S3: litho top, etch top, sum | Score |
|---|---|---|---|---|
| L0, 160x60, 0.7 | 2.372e-52, 1.122e-95, 2.372e-52 | 217.5, 2.944e-20, 217.5 | 7.934e-49, 2.525e-14, 2.525e-14 | 217.5 |
| L0, 320x120, 0.2 | 7.294e-49, 2.285e-88, 7.294e-49 | 207.8, 3.973e-17, 207.8 | 7.593e-43, 5.624e-12, 5.624e-12 | 207.8 |
| L0, 320x120, 0.7 | 1.665e-75, 1.013e-143, 1.665e-75 | 267.4, 2.441e-31, 267.4 | 6.982e-89, 2.406e-22, 2.406e-22 | 267.4 |
| L0, 320x120, 1.3 | 2.712e-90, 8.33e-173, 2.712e-90 | 285.5, 4.373e-38, 285.5 | 8.303e-121, 5.569e-35, 5.569e-35 | 285.5 |
| L1, 160x60, 0.7 | 2.998e-51, 2.089e-92, 2.998e-51 | 100.6, 2.07e-15, 100.6 | 2.826e-50, 7.534e-12, 7.534e-12 | 100.6 |
| L1, 320x120, 0.2 | 2.316e-47, 4.917e-88, 2.316e-47 | 97.98, 5.126e-12, 97.98 | 7.58e-49, 8.055e-10, 8.055e-10 | 97.98 |
| L1, 320x120, 0.7 | 2.955e-73, 1.227e-138, 2.955e-73 | 112.4, 9.893e-22, 112.4 | 4.863e-85, 1.434e-17, 1.434e-17 | 112.4 |
| L1, 320x120, 1.3 | 1.151e-87, 4.806e-166, 1.151e-87 | 115.4, 4.632e-27, 115.4 | 1.553e-123, 2.668e-36, 2.668e-36 | 115.4 |
| L2*, 160x60, 0.7 | 2.087e-62, 2.648e-112, 2.087e-62 | 24.04, 2.127, 26.17 | 2.565e-32, 3.953e-05, 3.953e-05 | 26.17 |
| L2*, 320x120, 0.2 | 2.844e-34, 4.25e-36, 2.887e-34 | 23.93, 2.018, 25.95 | 1.905e-27, 0.0001373, 0.0001373 | 25.95 |
| L2*, 320x120, 0.7 | 4.144e-96, 2.547e-117, 4.144e-96 | 24.64, 0.9334, 25.58 | 7.805e-46, 3.07e-05, 3.07e-05 | 25.58 |
| L2*, 320x120, 1.3 | 9.992e-114, 4.407e-127, 9.992e-114 | 24.77, 0.683, 25.46 | 1.227e-56, 1.217e-05, 1.217e-05 | 25.46 |
<!-- tables56 score end -->

<!-- tables56 ranking begin -->
L2* is supplementary: on 320x120 the rerun with the cap at 12,000, which stops by the rule; on 160x60 the field at the cap, a cycle the rule does not stop on.

| Grid, Sc_t, class (um) | Ranking (best first) | Scores L0, L1, L2 | L2 last | L0 or L1 first |
|---|---|---|---|---|
| 160x60, 0.7, 0.5 | L2* < L1 < L0 | 2.86, 1.316, 0.3413 | no | no |
| 160x60, 0.7, 5 | L2* < L1 < L0 | 217.5, 100.6, 26.17 | no | no |
| 320x120, 0.2, 0.5 | L2* < L1 < L0 | 2.731, 1.28, 0.3384 | no | no |
| 320x120, 0.2, 5 | L2* < L1 < L0 | 207.8, 97.98, 25.95 | no | no |
| 320x120, 0.7, 0.5 | L2* < L1 < L0 | 3.517, 1.468, 0.3337 | no | no |
| 320x120, 0.7, 5 | L2* < L1 < L0 | 267.4, 112.4, 25.58 | no | no |
| 320x120, 1.3, 0.5 | L2* < L1 < L0 | 3.755, 1.506, 0.3321 | no | no |
| 320x120, 1.3, 5 | L2* < L1 < L0 | 285.5, 115.4, 25.46 | no | no |
<!-- tables56 ranking end -->

<!-- tables56 margins begin -->
L2* is supplementary: on 320x120 the rerun with the cap at 12,000, which stops by the rule; on 160x60 the field at the cap, a cycle the rule does not stop on.

| Grid, Sc_t, class (um) | Pair (better, worse) | Scores | Difference; relative to the larger | Grid's change to each score (at Sc_t 0.7) | Sc_t's change to each score (320x120, largest of 0.2 and 1.3 against 0.7) | Robust (exceeds every change) |
|---|---|---|---|---|---|---|
| 160x60, 0.7, 0.5 | L2*, L1 | 0.3413, 1.316 | 0.9743; 0.741 | 0.00764, 0.152 | - | yes |
| 160x60, 0.7, 0.5 | L1, L0 | 1.316, 2.86 | 1.545; 0.54 | 0.152, 0.656 | - | yes |
| 160x60, 0.7, 5 | L2*, L1 | 26.17, 100.6 | 74.46; 0.74 | 0.588, 11.8 | - | yes |
| 160x60, 0.7, 5 | L1, L0 | 100.6, 217.5 | 116.8; 0.537 | 11.8, 50 | - | yes |
| 320x120, 0.2, 0.5 | L2*, L1 | 0.3384, 1.28 | 0.9418; 0.736 | 0.00764, 0.152 | 0.00469, 0.188 | yes |
| 320x120, 0.2, 0.5 | L1, L0 | 1.28, 2.731 | 1.45; 0.531 | 0.152, 0.656 | 0.188, 0.786 | yes |
| 320x120, 0.2, 5 | L2*, L1 | 25.95, 97.98 | 72.03; 0.735 | 0.588, 11.8 | 0.372, 14.5 | yes |
| 320x120, 0.2, 5 | L1, L0 | 97.98, 207.8 | 109.8; 0.528 | 11.8, 50 | 14.5, 59.7 | yes |
| 320x120, 0.7, 0.5 | L2*, L1 | 0.3337, 1.468 | 1.134; 0.773 | 0.00764, 0.152 | 0.00469, 0.188 | yes |
| 320x120, 0.7, 0.5 | L1, L0 | 1.468, 3.517 | 2.049; 0.583 | 0.152, 0.656 | 0.188, 0.786 | yes |
| 320x120, 0.7, 5 | L2*, L1 | 25.58, 112.4 | 86.86; 0.773 | 0.588, 11.8 | 0.372, 14.5 | yes |
| 320x120, 0.7, 5 | L1, L0 | 112.4, 267.4 | 155; 0.58 | 11.8, 50 | 14.5, 59.7 | yes |
| 320x120, 1.3, 0.5 | L2*, L1 | 0.3321, 1.506 | 1.174; 0.78 | 0.00764, 0.152 | 0.00469, 0.188 | yes |
| 320x120, 1.3, 0.5 | L1, L0 | 1.506, 3.755 | 2.249; 0.599 | 0.152, 0.656 | 0.188, 0.786 | yes |
| 320x120, 1.3, 5 | L2*, L1 | 25.46, 115.4 | 89.92; 0.779 | 0.588, 11.8 | 0.372, 14.5 | yes |
| 320x120, 1.3, 5 | L1, L0 | 115.4, 285.5 | 170.2; 0.596 | 11.8, 50 | 14.5, 59.7 | yes |
<!-- tables56 margins end -->

<!-- tables56 hotspots begin -->
L2* is supplementary: on 320x120 the rerun with the cap at 12,000, which stops by the rule; on 160x60 the field at the cap, a cycle the rule does not stop on.

| Layout, source, class (um) | Largest on 320x120, Sc_t 0.7 | 160x60: largest; same segment; shared of 5 | Sc_t 0.2: largest; same segment; shared of 5 | Sc_t 1.3: largest; same segment; shared of 5 | Sensors above the floor (320x120, 0.7); order equal on 160x60, 0.2, 1.3 |
|---|---|---|---|---|---|
| L0, S1, 0.5 | floor | x 1.2 | floor | x 1.2; yes; 5 | floor | x 1.2; yes; 5 | floor | x 1.2; yes; 5 | near_door; yes, yes, yes |
| L0, S1, 5 | floor | x 1.2 | floor | x 1.2; yes; 5 | floor | x 1.2; yes; 4 | floor | x 1.2; yes; 5 | near_door; yes, yes, yes |
| L0, S2, 0.5 | litho_tool top | x 3.6 | litho_tool top | x 3.6; yes; 5 | litho_tool top | x 3.6; yes; 5 | litho_tool top | x 3.6; yes; 5 | none; yes, yes, yes |
| L0, S2, 5 | litho_tool top | x 3.6 | litho_tool top | x 3.6; yes; 5 | litho_tool top | x 3.6; yes; 5 | litho_tool top | x 3.6; yes; 5 | none; yes, yes, yes |
| L0, S3, 0.5 | floor | x 6.2 | floor | x 6.2; yes; 4 | floor | x 6.2; yes; 4 | floor | x 6.2; yes; 5 | none; yes, yes, yes |
| L0, S3, 5 | floor | x 6.2 | floor | x 6.2; yes; 5 | floor | x 6.2; yes; 4 | floor | x 6.2; yes; 5 | none; yes, yes, yes |
| L1, S1, 0.5 | floor | x 1.2 | floor | x 1.2; yes; 5 | floor | x 1.2; yes; 5 | floor | x 1.2; yes; 5 | near_door; yes, yes, yes |
| L1, S1, 5 | floor | x 1.2 | floor | x 1.2; yes; 5 | floor | x 1.2; yes; 4 | floor | x 1.2; yes; 5 | near_door; yes, yes, yes |
| L1, S2, 0.5 | litho_tool top | x 3.6 | litho_tool top | x 3.6; yes; 5 | litho_tool top | x 3.6; yes; 5 | litho_tool top | x 3.6; yes; 5 | none; yes, yes, yes |
| L1, S2, 5 | litho_tool top | x 3.6 | litho_tool top | x 3.6; yes; 5 | litho_tool top | x 3.6; yes; 5 | litho_tool top | x 3.6; yes; 5 | none; yes, yes, yes |
| L1, S3, 0.5 | floor | x 6.2 | floor | x 6.2; yes; 5 | floor | x 6.2; yes; 5 | floor | x 6.2; yes; 5 | none; yes, yes, yes |
| L1, S3, 5 | floor | x 6.2 | floor | x 6.2; yes; 5 | floor | x 6.2; yes; 4 | floor | x 6.2; yes; 5 | none; yes, yes, yes |
| L2*, S1, 0.5 | floor | x 0.4 | floor | x 0.4; yes; 5 | floor | x 0.2; no; 5 | floor | x 0.4; yes; 5 | none; yes, yes, yes |
| L2*, S1, 5 | floor | x 0.4 | floor | x 0.4; yes; 3 | floor | x 0.2; no; 3 | floor | x 0.4; yes; 5 | none; yes, yes, yes |
| L2*, S2, 0.5 | litho_tool top | x 4.0 | litho_tool top | x 4.0; yes; 5 | litho_tool top | x 4.0; yes; 5 | litho_tool top | x 4.0; yes; 5 | hood_entry; yes, yes, yes |
| L2*, S2, 5 | litho_tool top | x 4.0 | litho_tool top | x 4.0; yes; 5 | litho_tool top | x 4.0; yes; 5 | litho_tool top | x 4.0; yes; 5 | hood_entry; yes, yes, yes |
| L2*, S3, 0.5 | floor | x 5.6 | floor | x 5.6; yes; 5 | floor | x 5.6; yes; 4 | floor | x 5.6; yes; 4 | none; yes, yes, yes |
| L2*, S3, 5 | floor | x 5.6 | floor | x 5.6; yes; 5 | floor | x 5.6; yes; 5 | floor | x 5.6; yes; 2 | none; yes, yes, yes |

Per comparison, over the rows (layout, source, class):
| Comparison | Rows | Largest the same segment | Largest the same surface | At least three of five shared | Shared: fewest, median | Segments not shared, summed |
|---|---|---|---|---|---|---|
| grids | 18 | 18 | 18 | 18 | 3, 5 | 3 |
| sc0.2 | 18 | 16 | 18 | 18 | 3, 5 | 8 |
| sc1.3 | 18 | 18 | 18 | 17 | 2, 5 | 4 |
<!-- tables56 hotspots end -->

**The ranking** is the same in every ordering the records allow: L2* < L1 < L0, the room as it
is last, for both classes, on both grids and at every Schmidt number (the ranking table). The
score is S2's deposition on the litho tool's top (section 5.3), and the layouts order by how much
of that emission the air over the top carries away. **The margins** are wide: L1's score is 53 to
60% below L0's and L2*'s 74 to 78% below L1's (the margin table's relative column), and every
neighbouring difference exceeds the grid's change to both scores and the Schmidt number's change
to both, so every order is robust by section 2.6's rule. The grid moves L0's score by 19% (0.66
of 3.5 at 0.5 um) and L1's by 10%, and the Schmidt number moves L0's by 22% and L1's by 13%,
against differences of 2.0 and 1.1 between neighbours. **The hotspots** agree: the largest
deposition location is the same segment on both grids and at Sc_t 1.3 in all 18 rows (the hotspot
summary; 16 of 18 at Sc_t 0.2), at least three of the five largest are shared in 18 of 18 between
grids and 17 of 18 at Sc_t 1.3 (the exception L2*'s S3 at 5 um at 1.3, 2 of 5, a floor segment and hood-bench top
segments at rates four orders apart), and Sc_t 0.2 moves the most segments (8 not shared against 3
between grids and 4 at 1.3) and the largest location in two rows, both L2*'s S1, the capped march
of section 5.3, whose largest floor segment moves from x 0.4 to x 0.2 m; the sensor orders are
equal in every comparison. The hotspots differ between layouts as the flow does: S2's largest segment is the litho
top at x 3.6 m in L0 and L1 and at x 4.0 m, the supply's edge, in L2*.

### 5.5 The figures

`ecr002_step8_<layout>_<grid>.png` beside this report, for the five rooms that stopped by the
rule (L2's is the cap-12,000 room): the streamlines, nu_t / nu and k with the boundaries marked.
L1's shows the gap between the litho tool and the etch chamber as a slow recirculation with
nu_t / nu near 0.1 where L0's gap descends to return 3; L2's shows the supply's edge at x 4.0 m
turning east over the litho tool and the right half of the room recirculating from the hood.

The comparison figures `ecr002_step8_compare_0.5um.png` and `ecr002_step8_compare_5um.png`: the
three layouts on 320x120 at Sc_t 0.7 side by side, the flow, S2's plume on one colour scale, and
the deposition bars summed over the three sources on one bar scale, each deposition panel titled
with the layout's score. The plume panels are the mechanism of section 5.3 in one row: S2's cloud
over the litho tool shrinks from L0 to L1 and in L2 streams east to the hood. The concentration
figures of all 36 marches (prompt 47's format) are under `results/builder56/figures/`, untracked.

## 6. Predictions against the measurement

<!-- tables56 verdict begin -->
| Prediction | Holds | From the records |
|---|---|---|
| (a) L0 320x120 stops within the cap; L1 and L2 converge on both grids | no | L0_320x120_stops_within_cap True; L1_160x60_converges True; L1_320x120_converges True; L2_160x60_converges False; L2_320x120_converges False; supplement_L2_320x120_cap12000_stops_by_the_rule True |
| (b) L0 or L1 first and L2 last in every ordering | no | 8 of 8 orderings; L2 last in 0, L0 or L1 first in 0; failing: ['160x60, 0.7, class 2: L2* < L1 < L0', '160x60, 0.7, class 4: L2* < L1 < L0', '320x120, 0.2, class 2: L2* < L1 < L0', '320x120, 0.2, class 4: L2* < L1 < L0', '320x120, 0.7, class 2: L2* < L1 < L0', '320x120, 0.7, class 4: L2* < L1 < L0', '320x120, 1.3, class 2: L2* < L1 < L0', '320x120, 1.3, class 4: L2* < L1 < L0']; orderings whose L2 row is supplementary (L2*): 8 |
| (c) L0 and L1's difference smaller than the grid's change to either score, for at least one class (320x120) | strict no, weak no | class 2 160x60: L1 < L0, difference 1.545, grid change L0 0.6564, L1 0.1525; class 2 320x120: L1 < L0, difference 2.049, grid change L0 0.6564, L1 0.1525; class 4 160x60: L1 < L0, difference 116.8, grid change L0 49.99, L1 11.81; class 4 320x120: L1 < L0, difference 155, grid change L0 49.99, L1 11.81 |
| (d) largest the same segment and three of five shared between grids and at Sc_t 1.3 for every row; Sc_t 0.2 moves more | no | grids: 18 rows, largest same 18, three shared 18, not shared 3; sc1.3: 18 rows, largest same 18, three shared 17, not shared 4; sc0.2: 18 rows, largest same 16, three shared 18, not shared 8; failing: ['L2, S3, class 4: largest floor | x 5.6 against floor | x 5.6, shared 2'] |
<!-- tables56 verdict end -->

| Prediction | Measured |
|---|---|
| (a) L0 on 320x120 stops by the rule within 10,000 iterations; L1 and L2 converge on both grids | **Holds for L0 and L1, fails for L2.** L0 stops at 4,389 and L1 at 4,394 on 320x120, at 1,953 and 1,938 on 160x60. L2 stops on neither grid within the cap: a cycle beside the door wall on 160x60, and on 320x120 a still field that only condition (e) refuses, which stops by the rule at 10,794 when the cap is raised (section 5.2) |
| (b) The ranking is L0 or L1 first and L2 last, on both grids and at every Sc_t, for both classes | **Fails in every ordering, and robustly.** The order is L2* < L1 < L0 in all eight orderings (the ranking table), L2's row supplementary in each, every margin exceeding every grid and Schmidt-number change (the margin table). The "deliberately poor" layout scores best because the score counts what stays on the process surfaces, and L2's supply edge sweeps the litho-top emission off the top and across the room to the hood; the room as it is scores worst because its air over the litho tool is the stillest |
| (c) L0 and L1 are close: their score difference smaller than the grid's change to either score for at least one class, so their order is not robust | **Fails under both readings.** L1's score is 53 to 60% below L0's; the difference (2.0 at 0.5 um, 155 at 5 um on 320x120) is three times the grid's change to L0's score and thirteen times its change to L1's (the verdict table's (c) row), and the Schmidt number's change is smaller still. Closing return 3 is not a subtle change at the litho tool's top: it halves the concentration the source sits in |
| (d) For every layout, source and class the largest deposition location is the same segment on both grids and at Sc_t 0.7 and 1.3, at least three of five shared; Sc_t 0.2 moves more | **Holds on every product-configuration row; one supplementary row short of three shared.** The largest location is the same segment in all 18 rows between grids and at 1.3; three of five are shared in all 12 L0 and L1 rows and in 5 of 6 L2* rows (the exception S3 at 5 um at Sc_t 1.3, 2 of 5); Sc_t 0.2 moves the most segments (8, against 3 and 4) and the largest location in the two L2* S1 rows, the capped march of section 5.3 (the hotspot summary) |

Read against the outcomes written with the predictions: (b) fails, so by that reading "the
comparative claim does not yet hold on this room". What the records say is narrower and is put
as questions in section 7: the tool ranks the layouts the same way on every grid and at every
Schmidt number, with margins far outside the numerical changes, so the ranking is stable, which
is what criterion 10 asks; what failed is the prediction of which layout the score would favour,
and the score the orchestrator chose rewards a layout that moves the emission to the operator at
the hood. (d) holds where the method applies without supplement.

### 6.1 Builder's predictions (section 3.2), scored

(a) was wrong about L2 on both grids (I expected it to converge, slower than L0) and right about
L0 and L1, whose counts are within 5 of each other rather than hundreds apart. (b) was wrong
with the orchestrator: I expected the doubled downflow to press S2's plume onto the top, and it
sweeps it off. (c) was wrong: closing return 3 changes the air over the litho top by a factor
of three in concentration, not "little", because the east gap's draw was part of what held it
still. (d) held as written for S2 and for the L0 and L1 rows; my expectation that Sc_t 0.2 would
leave fewer than three of five shared for S1 or S3 did not happen (four or five shared in every
L0 and L1 row). The 320x120 marches at Sc_t 0.2 were not the longest: the plumes reach a return
sooner, not later, with more diffusivity. The timing projection held, with the machine pause of section 5.1 inflating the
320x120 rows.

## 7. What this implies

Stated as questions, as the prompt asks; no decision is taken here.

- **Does the comparative claim hold, and at what resolution?** The ranking and the hotspots are
  stable under grid refinement and across the Schmidt number's range, with every margin several
  times the numerical change (section 5.4), on the three layouts and the score this step fixed.
  That is criterion 10's stability clause, met. The claim VAL-018 can make from it is that the
  tool orders layouts by this score robustly and places hotspots on the same 0.2 m segments on
  both grids; it is not that the ordering agrees with a designer's intuition, which it did not,
  and the resolution at which it holds is a score difference of about 50% between layouts against
  numerical changes of 10 to 20%. Whether a difference of 20% between two layouts would be called
  robust is not measured: no pair of layouts here is that close.
- **Is the score the product's score?** The score counts deposition on the process surfaces and
  nothing else, so it rewards L2 for carrying the litho-top emission across the room to the hood,
  past the hood entry, where a sensor reads 738 per m^3 and no sensor reads anything in L0 or L1
  (section 5.3). A stakeholder's score would count exposure at the operator positions and
  deposition on the process surfaces together, or rank each separately; with the sensors in the
  plumes (the question prompts 46 and 47 asked), the sensor ranking criterion 10's first wording
  named becomes scorable. Which score the Phase 3 gate report states the comparative claim on is
  Alex's.
- **Does L2's non-convergence matter?** The "poor" layout stops by the rule on 320x120 only with
  the cap raised 8% (condition (e) alone refusing, the field still), and on 160x60 not at all (a
  cycle beside the door wall at 0.4% of the velocity scale). The product configuration's cap is
  the file's 10,000; whether it rises to cover a layout like L2, whether a cycle of this size is a
  stopping question (prompt 46's section 7), or whether a layout that does not converge is
  reported as such and left out of the ranking, is a decision for the gate report. The ranking
  here is read with L2's rows marked.
- **Where does the absolute caveat bite?** Decision 1's caveat (turbulence over the tops about
  30% high) inflates absolute deposition on the tops in every layout alike; the ranking stands on
  ratios of 2 to 4 between layouts, far outside it. The absolute shares (S2's 2.8% deposited at
  5 um in L0) are not defended.
- **What does the grid change?** The score moves 10 to 19% between 160x60 and 320x120 (the margin
  table's grid column), the hotspots do not. A comparison finer than that needs the finer exact
  grid (640x240, Phase 6) or the near-wall treatment ADR-012 names; the step does not say which.

## 8. What this does not settle

- The L2 rows are supplementary (section 5.2); no product-configuration run of L2 stopped by
  the rule, and the 160x60 L2 marches read a cycling field.
- One score, chosen before the runs, was measured; a score that counts the sensors is not.
- The L2* S1 march at Sc_t 0.2 ends at a 100 s cap, not by the steady rule (section 5.3).
- The sources are prompt 47's; S1 and S3 contribute nothing to the score in any layout, so the
  ranking is S2's alone.
- The 0.1 against 0.4 Courant check of prompt 47 was not repeated on these rooms.
- Wall times: the machine pause and the two core types (section 5.1) make the 320x120 rows'
  seconds per outer iteration incomparable with the timing probes; the records carry what was
  measured.
