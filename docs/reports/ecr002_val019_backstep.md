# ECR-002 Step 7: VAL-019, the Backward-Facing Step

**Date:** 2026-10-12
**Tree:** branch `feature/ecr002-val019-backstep` from main at e760332 (prompt 51 merged as PR 78).
Commit 0 (a2cc20e) records Alex's two decisions (ADR-012 G (ii) and ECR-002 section 8 step 7,
notes of 2026-10-12): the inflow boundary layer grown in a long upstream channel whose length a
precursor run sets, and the step run on the current pressure solve. Commit A (6a606b5) adds the
case (`configs/validation_backstep.yaml`), its loader (`validation.cases.load_backstep`, the
presets `val019_410x40`, `val019_656x60` and `val019_984x90`), the data with its provenance and the
comparison code (`validation/backstep.py`, `validation/data/driver_seegmiller_1985/`), the harness
family (`scripts/benchmark.py`, reported and not scored) and their tests
(`tests/test_backward_step.py`). Nothing under `src/` changes. Every run goes through the committed
`StaggeredSolver` on the case as the suite loads it.
**Instruments:** `docs/reports/probe52/`: `data52.py` (the data tables printed for the module),
`backstep52.py` (the runs and the precursor, through prompt 46's runner with four probe-side
additions its docstring lists), `precursor52.py` (the layer along the precursor channel),
`tables52.py` (the tables), `figures52.py` (the figures) and the launcher `run52.sh`. Raw output
under `results/builder52/` (untracked).
**Order:** sections 1 to 4 were committed before any run of item 4, this file alone. The
precursor (item 2) ran before them, since its result is part of the case. Section 5 onward was
written after the runs.
**Policy:** `docs/REVIEW_POLICY.md`, "What blocks". The numbers live in the tables `tables52.py`
generates from the records; the sentences point at the tables. No threshold is set here: the
allowance is Alex's (ADR-012 G (ii), decision 6).

## 1. The questions

Three, from the prompt. Does the solver reproduce turbulent separation and reattachment, the flow
type behind every equipment edge in the product room, as well as a published k-epsilon run does?
What allowance does Alex set for VAL-019, which ADR-012 G (ii) leaves OPEN until the first coupled
results? And, if RNG converges here, how do the two variants rank on separation (ADR-012 A)?

## 2. Method (fixed before the runs)

### 2.1 The case

Driver and Seegmiller (1985) as the NASA Turbulence Modeling Resource poses it (ADR-012 G (ii),
source 13; `validation/data/driver_seegmiller_1985/PROVENANCE.md`). Step height H = 0.0127 m. The
upstream channel is 8 H tall (y from H to 9 H, the step a SOLID block under it for x < 0), the
downstream channel 9 H (expansion ratio 1.125), both walls straight, the step's corner at (0, H)
on a grid line in both axes. x runs from -L_up to +50 H, the outlet where the reference grids have
it: a pressure outlet, the outflow straight through the edge and not the sideways entry under
which the copy rule drifted (issue 61). The flow: U_ref 44.2 m/s at the channel centre at x/H = -4
and nu = U_ref H / 36,000 (Re_H 36,000; Mach 0.128 treated as incompressible), the density air's
round 1.2 kg/m^3, on which nothing compared depends. The inlet is uniform at the speed the
precursor sets (section 2.3) so that the computed centreline at x/H = -4 is U_ref within 1%, with
a turbulence intensity of 1% and a dissipation length of H / 10 (eps = k^(3/2) / l_e, ADR-012 C):
low, as the prompt asks; the data's own freestream at x/H = -4 carries about 1.8% in u'. Every
comparison normalises by the computed centreline speed at x/H = -4, as the data normalise by
U_ref measured there.

The model: the standard variant, the turbulence section as VAL-017 runs it (rule version 5,
`cfl_number` 0.25, `momentum_sweeps` 10, k and eps by conjugate gradients at `tol` 1e-10 with a
sweep cap of 500), `pressure_rtol` 1e-4, `alpha_velocity` 0.5, `max_simple_iter` 20,000. The
mass-imbalance tolerance is ADR-011 G's bound on the finest grid's smallest cell (3.01e-12 kg/s),
one value for the three grids.

### 2.2 The data and its provenance

The three files the Resource publishes for the case, fetched unchanged on 2026-10-12 and
committed with their addresses and SHA-256 checksums (`PROVENANCE.md` beside them): the profiles
of u / U_ref and 1000 u'v' / U_ref^2 at x/H = -4 (17 heights, from y/H 1.1 above the step's top),
1, 4, 6 and 10 (28 heights each, from y/H 0.1); the skin friction along the bottom wall at 20
stations (x/H -3.956 to 35.994, with its error); the pressure coefficient at 33 stations on the
bottom wall and 19 on the top, both the column referenced to x/H = -4 and the column shifted so Cp
is zero near x/H = 40 (its last bottom-wall station, 37.5, reads exactly zero). The reattachment
length is 6.26 +/- 0.10 H by laser oil-flow interferometry; the reader of section 2.5 applied to the
measured Cf itself gives 6.28, inside that bracket (`tests/test_backward_step.py`). The values were
printed into `validation/backstep.py` by `data52.py`; nothing was typed, and the test parses the
files on disk against the module.

The published k-epsilon results beside the data are the NPARC Alliance Validation Archive's
(read 2026-10-12): Chien k-epsilon x_r / H 5.30 in WIND and 5.31 in NPARC, 5.55 with WIND's
variable C_mu, and Menter SST 6.43, on a 238 x 185 grid from x/H = -105 to +50 clustered to y+ = 1.
The Chien model is a low-Reynolds-number k-epsilon integrated to the wall, not the standard model
with wall functions this solver runs; every table that cites those numbers says so.

### 2.3 Item 2: the precursor

The upstream channel alone, on the case's own y grid (the medium clustering of section 2.4, the
step's block spanning the whole 160 H length so the channel's floor is the block's top), 0.5 H
cells in x, a uniform inlet at 42.0 m/s with the intensity and length above, a pressure outlet,
the same solver keys. Read at every face column: the centreline speed, the bottom wall's delta_99
(the height at which u first reaches 99% of the local centreline speed, linear between rows) and
Re_theta (the trapezoid momentum thickness from the wall to mid-channel, times the local
centreline speed over nu). The prompt's rule, L_up = the distance at which the layer reaches 1.5 H
and Re_theta 5,000, plus 4 H, rounded up to a whole H, names two quantities that the computed
layer reaches at different distances, so each candidate length was also scored by the RMS of
u / U_centre against the data's 17 heights at x/H = -4, decision 1's stated purpose being the
measured layer there.

<!-- tables52 precursor begin -->
| x/H from the inlet | U_centre / U_in | delta_99 / H | Re_theta |
|---|---|---|---|
| 10 | 1.006 | 0.169 | 372.6 |
| 20 | 1.012 | 0.304 | 1014 |
| 30 | 1.018 | 0.448 | 1625 |
| 40 | 1.023 | 0.599 | 2202 |
| 50 | 1.028 | 0.726 | 2750 |
| 60 | 1.033 | 0.865 | 3277 |
| 70 | 1.038 | 1 | 3787 |
| 80 | 1.043 | 1.14 | 4283 |
| 90 | 1.047 | 1.28 | 4768 |
| 100 | 1.052 | 1.39 | 5242 |
| 110 | 1.056 | 1.5 | 5707 |
| 120 | 1.06 | 1.64 | 6164 |
| 130 | 1.064 | 1.76 | 6614 |
| 140 | 1.068 | 1.86 | 7057 |
| 150 | 1.072 | 1.99 | 7494 |

| Rule | Reached at x/H | L_up / H (plus 4 H, rounded up) | U_centre / U_in at x/H = -4 | U_in for U_ref (m/s) | RMS of u / U_centre against the x/H = -4 data | Largest |
|---|---|---|---|---|---|---|
| delta_99 = 1.5 H | 109.9 | 114 | 1.056 | 41.86 | 0.0146 | 0.0478 |
| Re_theta = 5,000 | 94.87 | 99 | 1.049 | 42.12 | 0.0174 | 0.0318 |

Precursor: stop error_estimate_and_continuity after 1,654 outer iterations, 518 s; inlet 42.0 m/s.
<!-- tables52 precursor end -->

The precursor stopped by the rule (first table's note), with the mean pressure moving by 3e-7 Pa
per outer iteration over its last 200: the pressure outlet does not drift on this arrangement. The
case takes the delta_99 rule, L_up = 114 H, the quantity the prompt defines and the candidate with
the smaller RMS against the measured profile (second table); the Re_theta rule's 99 H and both
candidates' numbers stay in the table for Alex. At 110 H the computed layer's Re_theta is 5,707
against the data's 5,000 at the same delta_99: the computed layer is fuller than the measured one,
a property of the model and the uniform inlet, not of the length. The inlet speed is 41.862 m/s,
the precursor's centreline ratio at x/H = -4 of that length (1.0558) under U_ref; whether the step
case itself makes U_ref there within 1% is section 5's first check.

### 2.4 The grids, and a deviation from the prompt

The prompt asks for tensor-product grids stretched toward the walls, the step's corner, x = 0 and
the reattachment region, and coarse in the upstream channel and toward the outlet. The committed
mesh (`src/mesh.py`) stretches each axis geometrically toward both domain walls only, and nothing
under `src/` changes in this step, so the grids are what the mesh can express and the harness can
load from a case file: x uniform, and y clustered toward the floor and the ceiling with the face
ny / 6 from the floor pinned on the step height (`load_backstep`, the face found by bisection on
the wall spacing). The step's face at x = 0 is on an x face when nx is a multiple of 82 (114 of
164 step heights). Three levels, each about 1.5 times the last per axis:

| Preset | Cells | dx / H | Wall cell / H | Cell at the step height / H | First-node y+ at Cf 2e-3 (floor), 2.9e-3 (step top) |
|---|---|---|---|---|---|
| val019_410x40 (coarse) | 16,400 | 0.400 | 0.118 | 0.181 | 67, 124 |
| val019_656x60 (medium) | 39,360 | 0.250 | 0.0845 | 0.121 | 48, 83 |
| val019_984x90 (fine) | 88,560 | 0.167 | 0.0560 | 0.080 | 32, 55 |

The y+ column is the flat-plate estimate before any field; the runs report the wall functions'
own y* on the three walls against the scalable floor (11.53). The medium grid has 39,000 cells
rather than the 30,000 asked, because the uniform x spends 70% of its cells in the upstream
channel. What the deviation costs: the reattachment is read between faces 0.25 H apart on the
medium grid and 0.167 H on the fine, and the cells at the step's corner are 0.25 H by 0.12 H rather
than the finer ones a stretched x would give there. A mesh that takes a face list per axis, or
clusters toward interior lines, is a `src/` change for Alex to decide; section 7 asks it.

### 2.5 The runs and the comparison

Item 4, in parallel, one BLAS thread each: standard with `upwind` (the default, key absent) on the
three grids; standard with `umist` (the key stated) on the medium grid; RNG with `upwind` on the
medium grid, reported whatever it does. Each run is prompt 46's runner on the committed solver,
from rest, the cell of the largest change recorded every outer iteration, the mean pressure over
the non-SOLID cells kept every outer iteration (the outlet probe's drift reading, prompt 41),
classified by step 5's rule as prompts 46 to 51 applied it (converged, bounded with its sub-class,
growing, diverged, positivity error). Item 3's harness row, `val019_656x60`, runs beside them
through `scripts/benchmark.py`, reported and not scored. Each converged run gets the marked figure:
the whole domain's streamlines with the inlet, the outlet and the step marked, then x/H -5 to 25
in three panels (streamlines coloured by speed, nu_t / nu on a log scale, k) with the
reattachment point marked on the floor.

Item 5, for every converged run (`validation.backstep.evaluate`, the same code the harness row
stores):

- x_r / H from the sign change of the bottom-wall shear. The shear is the wall function's at
  every u face of the floor row, `tau_w = mu_w u_P / y_P` with mu_w the scalable wall function on
  k_P (the mean of the two cells the face lies between) and y_P half the row's height, so its sign
  is u_P's; x_r is the zero of the straight line through the last pair of neighbouring faces
  whose shear turns from negative to positive, the last pair because a corner eddy can put an
  earlier crossing near x/H = 0.
- The inflow check at x/H = -4: the computed centreline speed over U_ref, the profile of u / U_ref
  against the 17 measured heights, delta_99 and Re_theta read as the precursor read them.
- u / U_ref at x/H = 1, 4, 6 and 10 against the data: the largest difference and the RMS over
  each station's 28 heights, sampled bilinearly from the final u faces (VAL-017's sampler), one
  figure per station.
- Cf and Cp along the bottom wall against the data at its stations, linear between the faces (Cf)
  or the wall cells (Cp); the computed Cp is shifted to zero at x/H = 37.5, the station at which
  the data's shifted column is zero, so both lines share the data's datum. One figure each; the
  top wall's Cp is drawn beside the bottom's, unscored.
- The resolution study: x_r on the three grids and its observed order over the ratio 1.5 if the
  differences are monotone, otherwise the spread; and medium against fine. The two schemes
  against each other on the medium grid: x_r, and at every station the largest difference between
  them over the larger of their distances from the data (VAL-017's test).

## 3. Predictions

### 3.1 Orchestrator (prompt 52, dated 2026-10-13 there; committed before item 4's runs)

- (a) The precursor puts L_up between 60 H and 105 H. A flat-plate turbulent layer at this speed
  reaches 1.5 H after roughly 80 H; the channel's acceleration and the inlet's low turbulence move
  it, not by a factor of two.
- (b) The standard model with wall functions reattaches short: x_r / H between 5.0 and 5.9 on the
  medium grid. The Chien k-epsilon runs give 5.3, and [12] reports the standard model about 15%
  short on a similar step.
- (c) The resolution study: medium and fine differ in x_r by under 0.2 H.
- (d) The two schemes differ in x_r by under 0.2 H, and on every station by less than half the
  larger distance from the data (VAL-017's test).
- (e) Upstream of reattachment the profiles at x/H = 1 and 4 sit within 0.1 U_ref of the data
  outside the recirculation; the largest departures are in the shear layer and the reversed flow.
- (f) RNG: no prediction on convergence. If it converges, it reattaches longer than standard.

What each outcome means:

- (b) holds: the solver behaves as a published k-epsilon does on separation, and the allowance can
  be set from the NPARC values as VAL-017's was from Rong and Nielsen.
- (b) fails long (above 6.3): suspect the wall functions at separation or the grid, before the
  model; the resolution study says which.
- (c) fails: the medium grid is not converged enough to score, and the harness row moves to the
  fine grid.
- (f) RNG converges and reattaches nearer 6.26: evidence for VAL-020's question, reported, not
  decided here.

Prediction (a) is already decided by section 2.3: L_up is 114 H, above the band, and the Re_theta
rule's 99 H inside it. Section 6 scores it with the rest. The scoring of (e) reads "outside the
recirculation" as the heights at or above the step's top, y/H >= 1, at x/H = 1 and 4.

### 3.2 Builder (2026-10-12, before item 4's runs; scored apart)

- (a) Fails on the delta_99 rule, holds on the Re_theta rule: the precursor is run, and the two
  quantities the rule names are reached 15 H apart because the computed layer carries more
  momentum deficit per unit thickness than the measured one.
- (b) Holds, near the band's upper half: 5.5 to 5.9. The standard model with wall functions on a
  first node at y+ 48 misses the near-wall part of the reversed flow, which shortens the bubble
  less than the Chien model's wall-integrated dissipation does; the first-node spacing of the
  medium grid is coarser than any grid [12] reports.
- (c) Holds narrowly: the coarse grid's x_r sits further from the medium's than the medium's from
  the fine's, by about the ratio of the grid steps, and the medium-fine difference lands near
  0.1 H.
- (d) Holds in x_r and at x/H = 4, 6 and 10; the x/H = 1 station is the one where the schemes can
  differ most, in the shear layer leaving the corner, and I expect it to hold there by less than a
  factor of two.
- (e) Holds at x/H = 4 and fails at x/H = 1 by a few heights just above the step's top (y/H 1.0 to
  1.3), where the measured shear layer is thinner than the model's on a 0.12 H cell.
- (f) RNG converges on this case: no obstacle corner inside a room for the cycle of prompts 46 to
  48 to sit at, and the one corner here is the step's, where the flow leaves rather than strikes.
  It reattaches longer than standard, by 0.3 to 0.8 H.

## 4. Stop conditions (from the prompt)

- The data cannot be fetched, or its stations not matched (fetched; matched, section 2.2).
- The outlet drifts, a positivity error, or a capped pressure correction.
- The run set would exceed about eight hours on the machine. From the 100-iteration timing probes
  (0.22, 0.81 and 2.17 s per outer iteration on the coarse, medium and fine grids, three processes
  together), the fine grid costs 3 h per 5,000 outer iterations and 12 h at the cap; the set is
  launched whole and the fine run's projection is read from its first hundreds of iterations, the
  report saying so if it is stopped.

## 5. Results (written after the runs)

### 5.1 Order and cost

The five rows of item 4 and the harness row were launched together on 2026-10-12 at 19:27 on commit
597315c (`run52.sh flows`, then `run52.sh row 6`), one BLAS thread each on twenty-four cores. The
coarse run and the RNG run ended within twelve minutes; the two medium runs and the harness row
reached the 20,000 cap on 2026-10-13 at 14:40, and the fine run on 2026-10-13 at 20:21. The machine
slept during the night, so the wall times of every run that spanned it (the timing table's medium
and fine rows, and the harness row's 68,672 s) are not measurements; the 100-iteration probes
before the runs gave 0.22, 0.81 and 2.17 s per outer iteration on the coarse, medium and fine
grids, three processes together, and the medium runs took about 1.0 s per outer iteration while six
processes ran. The harness row was run a second time, alone beside the fine run (`run52.sh row 2`,
from 2026-10-13 14:45), because the first row's record carries `git_dirty` true: the figure script
`docs/reports/probe52/figures52.py` was edited (the streamline resampling, d972311) while the row
ran, and the harness reads the tree when the solve ends. No file under `src/`, `validation/`,
`configs/` or `scripts/` differed. Section 5.8 reads both rows.

<!-- tables52 timing begin -->
| Run | Outer | Wall (s) | Wall (h) | Per outer (s) | CG per correction (mean) | Stage seconds |
|---|---|---|---|---|---|---|
| standard 410x40 (coarse) upwind | 2,184 | 645.1 | 0.179 | 0.295 | 864 | momentum 62, pressure 508, correct 1, turbulence 71 |
| standard 656x60 (medium) upwind | 20,000 | 6.907e+04 | 19.2 | 3.45 | 1.35e+03 | momentum 2359, pressure 65268, correct 26, turbulence 1364 |
| standard 656x60 (medium) umist | 20,000 | 6.913e+04 | 19.2 | 3.46 | 1.32e+03 | momentum 22075, pressure 45268, correct 23, turbulence 1719 |
| rng 656x60 (medium) upwind | 403 | 378 | 0.105 | 0.938 | 1.27e+03 | momentum 24, pressure 326, correct 1, turbulence 26 |
| standard 984x90 (fine) upwind | 20,000 | 8.956e+04 | 24.9 | 4.48 | 1.36e+03 | momentum 3989, pressure 82596, correct 36, turbulence 2864 |
<!-- tables52 timing end -->

The pressure correction is most of every run's time (the stage column), as prompt 51's report
found on the Annex 20 row: on the medium grid it takes 1,300 to 1,600 conjugate-gradient
iterations per correction at `pressure_rtol` 1e-4, the k and eps solves under twenty each.

### 5.2 Item 4: the runs

<!-- tables52 matrix begin -->
| Run | Class | Stop | Outer | Residual: least (at), end | Readings at stop (a), (b), (c), (d), (e) | Holds from (a), (b), (c), (d), (e) | Largest speed at end (m/s), cell | CG per correction: mean, largest | Cap hits | k, eps CG iterations (mean); capped | Wall (s), per outer | Face hash |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| standard 410x40 (coarse) upwind | converged | error_estimate_and_continuity | 2,184 | 2.7e-13 (2,183), 2.7e-13 | 1.7e-10, 6e-14, 5.3e-12, 1.9e-12, 1e-06 | 1,917, 1,294, 521, 2,184, 2,184 | 44.8, (1.4503, 0.034) | 864, 1,097 | 0 | 14.8, 13.7; 0 | 645.1, 0.295 | 9620080dabf02a15 |
| standard 656x60 (medium) upwind | bounded (neither) | max_simple_iter | 20,000 | 3.92e-07 (1,959), 1.75e-05 | 2.5, 2e-09, 1.1e-06, 3.2e-08, 3.9e+04 | -, -, -, -, - | 44.8, (1.4526, 0.0343) | 1.35e+03, 1,585 | 0 | 18.9, 16.6; 0 | 6.907e+04, 3.45 | 37d84ab7c1495488 |
| standard 656x60 (medium) umist | bounded (neither) | max_simple_iter | 20,000 | 1.51e-06 (1,438), 1.98e-05 | -, 5.1e-09, 1.3e-06, 4.1e-08, 3.8e+04 | -, -, -, -, - | 44.8, (1.4526, 0.0343) | 1.32e+03, 1,599 | 0 | 19, 16.9; 0 | 6.913e+04, 3.46 | c5f925a497808a0c |
| rng 656x60 (medium) upwind | diverged | diverged | 403 | 0.000981 (42), 1.16 | -, 2.8e-05, 0.014, 0.0026, 2.3e+03 | -, -, -, -, - | 1.54e+03, (2.078, 0.1052) | 1.27e+03, 1,611 | 0 | 11, 9.93; 0 | 378, 0.938 | - |
| standard 984x90 (fine) upwind | bounded (neither) | max_simple_iter | 20,000 | 3.14e-06 (6,446), 3.97e-05 | 2.3, 6.5e-09, 3.3e-06, 1.7e-08, 2.1e+05 | -, -, -, -, - | 44.8, (1.451, 0.0347) | 1.36e+03, 2,380 | 0 | 24, 20.2; 0 | 8.956e+04, 4.48 | d7fb3e9ab037301b |
<!-- tables52 matrix end -->

The coarse run converges by the rule, condition (e) the last to hold. The two medium runs are
bounded at the cap in a cycle, neither falling nor stalled by step 5's sub-classes: the residual
reaches its least near outer iteration 1,500 to 2,000 (4e-7 under upwind, 1.5e-6 under umist) and
then moves between about 5e-6 and 4e-5 for the remaining 18,000 iterations, with conditions (b),
(c) and (d) met at the stop and (a) and (e) never holding (the holds-from column). The cell of the
largest velocity change is the floor cell at x/H 4.9 in 87% of the last 2,000 iterations under
either scheme, the cell the reattachment point falls in (section 5.4), with a step of about 0.08
m/s (1.7e-3 of U_ref) per iteration; the largest eddy-viscosity change is in the cell above it. The
reattachment point moves between neighbouring faces and the iteration does not settle on one. The
fine grid is the same: its residual reaches 3e-6 near iteration 6,400 and cycles between 4e-6 and
7e-5 to the cap, the largest velocity change in the floor cell at x/H 5.1 (the reattachment cell
again) in 72% of the last 2,000 iterations, with a step of about 0.12 m/s. No run hit the pressure
cap, no k or eps solve fell back, and no positivity error was raised. The mean pressure of every
standard run moves by under 1e-3 Pa per outer iteration over its last 200 (the walls table, against
a dynamic pressure of 1,170 Pa): the pressure outlet does not drift on this arrangement, so the
prompt's second stop condition is not met.

RNG diverged at outer iteration 403. Its residual fell to 1e-3 by iteration 42 and the largest
speed settled at 63 to 66 m/s, where the standard runs' also sat at that stage; from iteration
350 the largest speed sits in the last cell column, at the outlet, near the top wall (y/H 8.7)
and then the bottom (y/H 2.5), reaching 124 m/s by 380 and 1,537 m/s at 402, while the mean
pressure walks by 12 Pa per iteration over the last 200. The divergence is at the pressure outlet
under RNG's viscosity, not at the step; the standard variant on the same grid, scheme and outlet
shows no walk. It is reported, as the prompt asks, and not pursued here.

### 5.3 The inflow and the walls

<!-- tables52 walls begin -->
| Run | Class | U_centre at x/H = -4 (m/s) | Over U_ref | delta_99 / H | Re_theta | y+ floor downstream of x_r: median (at x/H = 10) | y+ step top: median (at x/H = -4) | y+ ceiling upstream: median (at x/H = -4) | Scalable floor | Mean pressure drift, last 200 outer (Pa per iteration) | nu_t / nu: median | largest |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| standard 410x40 (coarse) upwind | converged | 44.14 | 0.9987 | 1.39 | 5024 | 77.2 (91.4) | 127 (126) | 83 (81) | 11.53 | -5e-09 | 3.53 | 631.9 |
| standard 656x60 (medium) upwind | bounded (neither) | 44.18 | 0.9996 | 1.46 | 5229 | 53.9 (63.4) | 85.5 (84.3) | 59.7 (58) | 11.53 | -0.00034 | 3.49 | 587.1 |
| standard 656x60 (medium) umist | bounded (neither) | 44.18 | 0.9996 | 1.46 | 5225 | 54.4 (64.8) | 85.6 (84.3) | 59.7 (58.1) | 11.53 | 0.00048 | 3.5 | 614.9 |
| standard 984x90 (fine) upwind | bounded (neither) | 44.2 | 1 | 1.49 | 5353 | 34.8 (40.5) | 56.9 (55.9) | 39.6 (38.4) | 11.53 | -0.00035 | 3.5 | 562 |
<!-- tables52 walls end -->

The computed centreline speed at x/H = -4 is U_ref within 0.2% on every compared grid (the
precursor's inlet speed carries over to the step case), so the normalisation is the data's. The
layer at x/H = -4 is 1.39 H thick with Re_theta 5,024 on the coarse grid, 1.46 H with 5,229 on
the medium and 1.49 H with 5,353 on the fine, against the data's 1.5 H and 5,000: the coarser
grids thin the layer a little from the precursor's 1.5 H at the same distance, and every grid's
layer carries more momentum deficit than the measured one at its thickness. The first-node y+
sits inside the log layer on every wall of every grid, 77 to 127 on the coarse, 54 to 86 on the
medium and 35 to 57 on the fine (the floor downstream of reattachment, the ceiling and the step's
top in that order); nothing is near the scalable floor. The eddy viscosity's median over the
domain is the inlet's (3.5 nu, the freestream of a uniform inlet with I 1% and l_e H / 10) and
its largest 560 to 630 nu, in the shear layer and the floor's recovering layer (the field
figures).

### 5.4 Item 5: the reattachment length and the resolution study

<!-- tables52 reattachment begin -->
| Run | Class | x_r / H | Minus measured (H) | Relative |
|---|---|---|---|---|
| standard 410x40 (coarse) upwind | converged | 4.509 | -1.75 | -28.0% |
| standard 656x60 (medium) upwind | bounded (neither) | 4.859 | -1.4 | -22.4% |
| standard 656x60 (medium) umist | bounded (neither) | 4.896 | -1.36 | -21.8% |
| standard 984x90 (fine) upwind | bounded (neither) | 5.087 | -1.17 | -18.7% |
| Driver and Seegmiller (1985), measured | - | 6.26 +/- 0.1 | 0 | 0 |
| NPARC archive: NPARC Chien k-epsilon (low-Re, to the wall) | published | 5.31 | -0.95 | -15.2% |
| NPARC archive: WIND Chien k-epsilon (low-Re, to the wall) | published | 5.3 | -0.96 | -15.3% |
| NPARC archive: WIND Chien k-epsilon, variable C_mu (low-Re) | published | 5.55 | -0.71 | -11.3% |
| NPARC archive: WIND Menter SST | published | 6.43 | 0.17 | +2.7% |

The Chien k-epsilon rows are low-Reynolds-number k-epsilon runs integrated to the wall on a y+ = 1 grid, not the standard model with wall functions this solver runs.

| Grid | Class | x_r / H (standard, upwind) |
|---|---|---|
| 410x40 (coarse) | converged | 4.509 |
| 656x60 (medium) | bounded (neither) | 4.859 |
| 984x90 (fine) | bounded (neither) | 5.087 |
| extrapolated (observed order, ratio 1.5) | - | 5.518 |

Observed order of x_r over the three grids (ratio 1.5 per axis): 1.05; spread 0.578 H; medium against fine 0.228 H.
<!-- tables52 reattachment end -->

The standard model with wall functions reattaches short on every grid: 28% short of the measured
6.26 H on the coarse grid and 22% on the medium (first table), beside the NPARC archive's
low-Reynolds Chien k-epsilon at 15% short and SST at 3% long. The coarse-to-medium change is +0.35
H; the medium-to-fine change +0.23 H (second table), monotone, an observed order of 1.05 over the
ratio 1.5, a spread of 0.58 H over the three grids, and an order-one extrapolation (the table's
last line) that puts the grid-converged length near 5.5 H, still 12% short of the measurement and
near the Chien model's variable-C_mu value. The coarse grid's reading is between faces 0.4 H apart
and the fine grid's 0.17 H; the length grows with refinement because the finer grids resolve more
of the reversed flow under the shear layer (the fine grid's largest differences at x/H = 1 and 4
are 0.03 of U_ref smaller than the medium's, section 5.5).

### 5.5 The profiles

<!-- tables52 profiles begin -->
| Run | Class | x/H = -4.0: largest / RMS of u / U_ref (heights) | x/H = 1.0: largest / RMS of u / U_ref (heights) | x/H = 4.0: largest / RMS of u / U_ref (heights) | x/H = 6.0: largest / RMS of u / U_ref (heights) | x/H = 10.0: largest / RMS of u / U_ref (heights) |
|---|---|---|---|---|---|---|
| standard 410x40 (coarse) upwind | converged | 0.0542 / 0.0187 (17) | 0.173 / 0.0603 (28) | 0.19 / 0.0685 (28) | 0.0505 / 0.0248 (28) | 0.0628 / 0.0269 (28) |
| standard 656x60 (medium) upwind | bounded (neither) | 0.0531 / 0.0181 (17) | 0.137 / 0.0478 (28) | 0.148 / 0.0547 (28) | 0.0476 / 0.0235 (28) | 0.0548 / 0.0275 (28) |
| standard 656x60 (medium) umist | bounded (neither) | 0.053 / 0.0183 (17) | 0.126 / 0.0439 (28) | 0.147 / 0.0544 (28) | 0.0485 / 0.023 (28) | 0.0558 / 0.0249 (28) |
| standard 984x90 (fine) upwind | bounded (neither) | 0.0523 / 0.018 (17) | 0.105 / 0.0377 (28) | 0.114 / 0.0449 (28) | 0.0554 / 0.0269 (28) | 0.0545 / 0.0291 (28) |
<!-- tables52 profiles end -->

Figures: `ecr002_val019_profile_x-4.png`, `_x1.png`, `_x4.png`, `_x6.png`, `_x10.png`. The inflow
profile (x/H = -4) matches the data to 0.018 RMS of U_ref on every grid, the largest difference
0.053 at the lowest height (y/H 1.1), where the computed layer is fuller than the measured one.
Downstream the largest departures are in the reversed flow and the shear layer at x/H = 1 and 4
(0.11 to 0.19 of U_ref across the grids, the measured reversed flow reaching -0.17 U_ref at x/H = 4
where the computed reaches -0.05), and at x/H = 6, which the data have inside the bubble and the
model has past reattachment, the profiles agree to 0.05 because the measured reversed flow is
already weak there. Above the step's top the computed and measured profiles agree to 0.05 of U_ref
at every station.

### 5.6 Cf and Cp along the bottom wall

<!-- tables52 coefficients begin -->
| Run | Class | Cf: largest difference | RMS | stations | Cp (shifted to zero at x/H = 37.5): largest difference | RMS | stations |
|---|---|---|---|---|---|---|---|
| standard 410x40 (coarse) upwind | converged | 0.00129 | 0.000524 | 20 | 0.105 | 0.0389 | 33 |
| standard 656x60 (medium) upwind | bounded (neither) | 0.00107 | 0.000427 | 20 | 0.0827 | 0.0311 | 33 |
| standard 656x60 (medium) umist | bounded (neither) | 0.00103 | 0.000435 | 20 | 0.0813 | 0.0306 | 33 |
| standard 984x90 (fine) upwind | bounded (neither) | 0.000808 | 0.000345 | 20 | 0.0627 | 0.0249 | 33 |
<!-- tables52 coefficients end -->

Figures: `ecr002_val019_cf.png` and `ecr002_val019_cp.png`. The computed skin friction follows
the data upstream (3e-3 before the step), reaches its most negative value at x/H 2 to 3 as the
data do, at a similar depth, and then crosses zero 1.4 to 1.8 H before the data and recovers
earlier, to the data's level by x/H 20 (the Cf figure). The pressure coefficient's recovery is
the same story shifted: the computed minimum is deeper and the rise to the plateau about 1.5 H
upstream of the data's, and the plateau itself, the datum, agrees. The top wall's Cp (thin
lines) agrees with its data to the figure's width.

### 5.7 The two schemes

<!-- tables52 schemes begin -->
| x/H | Largest difference between the schemes, u / U_ref | RMS | upwind's largest from the data | umist's | Ratio to the larger | Under 0.5 |
|---|---|---|---|---|---|---|
| -4 | 0.000544 | 0.000347 | 0.0531 | 0.053 | 0.0102 | holds |
| 1 | 0.0145 | 0.00572 | 0.137 | 0.126 | 0.105 | holds |
| 4 | 0.00772 | 0.0025 | 0.148 | 0.147 | 0.0523 | holds |
| 6 | 0.00497 | 0.00209 | 0.0476 | 0.0485 | 0.103 | holds |
| 10 | 0.00699 | 0.00349 | 0.0548 | 0.0558 | 0.125 | holds |

x_r / H: upwind 4.859, umist 4.896, difference 0.0377 H (prediction (d) asks under 0.2 H).
<!-- tables52 schemes end -->

Under upwind and umist advection of k and eps the medium grid gives the same reattachment length
to 0.04 H and the same profiles to 0.015 of U_ref, a tenth of either scheme's distance from the
data at every station: the scheme is not where the model's error is, as VAL-017 found for the
room.

### 5.8 Item 3: the harness row

`val019_656x60` (`scripts/benchmark.py`, reported and not scored) stops at the cap, as the probe's
medium run does, and its record holds the same comparison to the printed digits: x_r / H 4.859, the
centreline 0.9996 U_ref, delta_99 1.46 H, Re_theta 5,229, the profiles' RMS 0.018, 0.048, 0.055,
0.024 and 0.028 at x/H = -4, 1, 4, 6 and 10, Cf RMS 4.3e-4, Cp RMS 0.031 (`benchmarks/
results.jsonl`, the rows with case `val019_656x60`). Its value, the reattachment length's relative
departure, is 0.224, and `passed` is None. The first row (commit 597315c, `concurrent_processes` 6)
carries `git_dirty` true for the reason section 5.1 gives; the second (commit d972311,
`concurrent_processes` 2) carries it too, because the report and the notes of this step were
written in the tree while it ran. The two rows' values and components agree to every printed digit
(the solve is bit for bit the same), and the probe's medium record on 597315c gives the same x_r; a
third row, run in a worktree of d972311 with nothing changing, is appended after them (section
5.9).

### 5.9 The third harness row

THIRD-ROW-PENDING

## 6. Predictions against measurement

<!-- tables52 predictions begin -->
| Prediction | Measured | Verdict |
|---|---|---|
| (a) | L_up = 114 H (delta_99 rule); Re_theta rule 99 H | fails |
| (b) | x_r / H = 4.859 on the medium grid (bounded (neither)); band (5.0, 5.9) | fails short |
| (c) | medium 4.859, fine 5.087: 0.228 H | fails |
| (d) | x_r differs by 0.038 H; ratios 1.0: 0.11, 4.0: 0.05, 6.0: 0.10, 10.0: 0.13 | holds |
| (e) | largest |u / U_ref - data| at y/H >= 1.0: x/H = 1.0: 0.051, x/H = 4.0: 0.048 | holds |
| (f) | RNG diverged at 403 outer iterations; no prediction on convergence | not scorable (did not converge) |
<!-- tables52 predictions end -->

- (a) fails: the precursor's delta_99 rule puts L_up at 114 H, above the band; the Re_theta rule
  would have put it inside (99 H). A flat-plate estimate undershoots the channel's layer growth
  under a uniform inlet with 1% turbulence by about a third. - (b) fails short: 4.86 H on the
  medium grid, under the band's 5.0, and 4.51 on the coarse. The reattachment is shorter than the
  Chien model's 5.3, not longer, so the outcome written with the prediction ("suspect the wall
  functions at separation or the grid, before the model") does not apply as written; the model
  reattaches short of a published low-Reynolds k-epsilon by a further 0.4 H, on a grid whose first
  node is in the log layer everywhere. - (c) fails, narrowly: medium and fine differ by 0.23 H
  against the 0.2 H asked. By the outcome written with the prediction the medium grid is not
  converged enough to score and the harness row would move to the fine grid, at four times the cost
  (the timing table); the report leaves that with Alex (section 7). - (d) holds: the schemes differ
  by 0.04 H in x_r and by a tenth of the model's distance from the data at every station. - (e)
  holds: above the step's top the profiles at x/H = 1 and 4 are within 0.051 of U_ref of the data;
  the largest departures are in the reversed flow and the shear layer, as predicted. - (f) not
  scorable: RNG diverged at the outlet (section 5.2).

The builder's predictions (section 3.2), scored apart: (a) as written, both clauses; (b) fails, the
length below the band rather than in its upper half; (c) fails as the orchestrator's does, the
medium-fine difference 0.23 H not 0.1; (d) holds, and the x/H = 1 station is the one where the
schemes differ most, as expected; (e) wrong at x/H = 1, which holds; (f) wrong, RNG diverged.

## 7. What this implies

Questions, not decisions; the numbers are in the tables above.

1. **The allowance for VAL-019.** The one standard k-epsilon result with wall functions on this
   step is this one: 4.86 H on the medium grid, 22% short, with the profiles above the step's top
   within 0.05 U_ref and the reversed flow weaker than measured. The sourced references are not the
   same model: the NPARC archive's Chien runs integrate to the wall (15% short), and [12]'s
   standard model with wall functions on the Kim, Kline and Johnston step was 15% short of a
   measured 7.1 H. One allowance Alex might set from these: x_r / H no shorter than the published
   standard-with-wall-functions shortfall, 15%, which this result fails by 7 points; another, the
   NPARC k-epsilon values as VAL-017's allowance was set from Rong and Nielsen's, which it also
   fails; a third, a reported-only case with the profiles above the step scored (0.05 U_ref), which
   it would pass. The report takes no position; the case stays OPEN. 2. **The grid.** The committed
   mesh cannot cluster toward the step's corner or the reattachment region (section 2.4), so the
   medium grid's x spacing there is 0.25 H and the coarse-to-medium change in x_r is 0.35 H. The
   three grids give 4.51, 4.86 and 5.09 H at first order, the extrapolated length near 5.5 H, so
   about a third of the medium grid's shortfall is the grid and two thirds the model. A mesh that
   takes a face list per axis, or clusters toward interior lines, is a small `src/` change; whether
   VAL-019 is worth it before VAL-020 is Alex's. 3. **The stop.** The medium grid does not stop by
   the rule: the iteration cycles at the reattachment cell with a velocity step of 1.7e-3 U_ref,
   the same kind of bounded cycle the product room showed at an obstacle corner (prompts 46 and
   47), while the coarse grid converges. The comparison is made at the cap, labelled, as VAL-017's
   375x125 was; the differences read (1.4 H in x_r, 0.05 to 0.19 in the profiles) are far above the
   cycle's amplitude. Whether a stop on a bounded cycle is wanted for the validation cases is the
   open question prompt 47 raised. 4. **The variants.** RNG diverged at the pressure outlet, so
   ADR-012 A's ranking on separation has no pair here; the standard variant's own ranking against
   the data is the tables'. If VAL-020 favours RNG, its outlet behaviour needs its own probe before
   the step is rerun with it. 5. **The cost, for decision 2.** The medium grid's 20,000 iterations
   took 69,000 s beside five other runs (about 1.0 s per outer iteration when six ran, 0.81 alone,
   95% of it the pressure correction), and the fine grid's 20,000 took 89,600 s (4.5 s per outer
   iteration beside the others, 2.2 alone), 92% of it the pressure correction at 1,400 to 2,400
   conjugate-gradient iterations per correction. The multigrid successor (ADR-013) is the question
   Alex deferred to this timing and the product's.

## 8. Figures

- `ecr002_val019_standard_410x40_upwind.png`, `_656x60_upwind.png`, `_656x60_umist.png` and
  `_984x90_upwind.png`: the marked figure of each compared run (the whole domain; x/H -5 to 25 in
  three panels; the step, inlet, outlet and reattachment point marked). -
  `ecr002_val019_profile_x-4.png`, `_x1.png`, `_x4.png`, `_x6.png`, `_x10.png`: the profiles. -
  `ecr002_val019_cf.png`, `ecr002_val019_cp.png`: the wall coefficients. -
  `ecr002_val019_precursor.png`: the precursor's layer along the channel.