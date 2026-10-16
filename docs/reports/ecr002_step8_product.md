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
