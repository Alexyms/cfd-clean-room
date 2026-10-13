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
