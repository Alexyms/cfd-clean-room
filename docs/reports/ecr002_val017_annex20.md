# ECR-002 Step 7: VAL-017, the Annex 20 Room

**Date:** 2026-10-11
**Tree:** branch `feature/ecr002-val017-annex20` from main at 17342cc. Commit 0 records Alex's
decision of 2026-10-11 (ADR-012 decision 2: the product runs the standard variant until VAL-020).
Commit A adds the case (`configs/validation_annex20.yaml`), its loader (`validation.cases.load_annex20`,
the presets `val017_216x72` and `val017_375x125`), the data with its provenance and the comparison
code (`validation/annex20.py`) and their tests (`tests/test_annex20.py`). Nothing under `src/`
changes. Every run goes through the committed `StaggeredSolver` on the case as the suite loads it.
**Instruments:** `docs/reports/probe49/`: `digitize49.py` (Rong and Nielsen's curves, read from the
PDF's embedded images), `data49.py` (the data tables printed for the module), `annex49.py` (the
runs, through prompt 46's runner), `tables49.py` (the tables), `figures49.py` (the figures) and
the launcher `run49.sh`. Raw output under `results/builder49/` (untracked).
**Order:** sections 1 to 4 were committed before any run, this file alone. Section 5 onward was
written after the runs.
**Policy:** `docs/REVIEW_POLICY.md`, "What blocks". The numbers live in the tables `tables49.py`
generates from the records; the sentences point at the tables. No threshold is set here: the
allowance is Alex's (ADR-012 decision 6).

## 1. The questions

Two, from the prompt. Is upwind advection of k and eps acceptable in a ventilated room? It is
acceptable if the difference between the two schemes is small beside the model's own difference
from measurement. And what allowance does Alex set for VAL-017? ADR-012 G (iii) leaves the
threshold open until the first coupled results, and this is that result. The case is the
validation nearest the product: a mechanically ventilated room with a wall jet.

## 2. Method (fixed before the runs)

### 2.1 The case

ADR-012 G (iii): L 9.0 m by H 3.0 m; a slot inlet h = 0.168 m at the top of the left wall; an
outlet t = 0.48 m at the foot of the right wall; u0 0.455 m/s, nu 15.3e-6 m^2/s (Re 5,000 on h);
inlet k and eps from I = 0.04 and l_e = h / 10 in the design's convention (eps = k^(3/2) / l_e,
ADR-012 C). The standard variant, and the turbulence section otherwise as the product runs it:
`cfl_number` 0.25, `alpha_turbulence` 0.7, `error_estimate`, `momentum_sweeps` 10,
`pressure_rtol` 1e-4, `max_simple_iter` 10,000, `alpha_velocity` 0.5 (prompt 46's product
setting). The density is air's round 1.2 kg/m^3; nothing compared depends on it. The outlet is a
`fixed_flow_outlet` stating no velocity, so it takes the whole inflow at one face velocity; the
air reaches it coming down the right wall, sideways into the outlet cells, the arrangement under
which the pressure outlet's copy drifted (issue 61). The mass-imbalance tolerance is ADR-011 G's
bound on the finer grid (1.152e-9 kg/s), one value for both grids.

Two grids, the grid sensitivity ADR-012 G (v) asks for. 216x72 (0.0417 m cells) is the ADR's
grid: the slot covers 4.03 cells and its edge falls off a face, so the loader scales the inlet
velocity (to 0.45864 m/s) so the four covered faces carry u0 h exactly; u0 stays the
normalisation. 375x125 (0.024 m) is the coarsest grid on which the slot (7 faces) and the outlet
(20 faces) both fall on whole faces, and the velocity stays u0. On 216x72 the outlet's twelve
covered faces span 0.50 m, not 0.48. Each grid's first-node y+ is reported against the scalable
floor (11.53). The ADR's estimate for 216x72 is 27 along the jet.

### 2.2 The data and its provenance

The measurements are the Aalborg benchmark workbook's symmetry plane, Re 5,000 (Restivo 1979,
digitized in 2005 from Nielsen 1990's figures 5 and 7; the address, checksum and licence note in
`docs/reports/product_case_reynolds.md`, section 6.1), the mean u / u0 of the four lines the
specification names, read by `digitize49.py` and printed into `validation/annex20.py` by
`data49.py`, nothing typed. The specification measures y down from the ceiling; the module
stores the lines in the solver's frame (y up from the floor), so the specification's y = h/2 is
the ceiling-jet line at y = 2.916 m and its y = H - h/2 the floor line at y = 0.084 m. The
stored vertical profiles carry 1.17 (x = H) and 0.62 (x = 2H) of u0 h with the no-slip closure,
the numbers section 6.3 of that report found, which `tests/test_annex20.py` pins. The workbook
file itself is not committed (its portal terms forbid redistribution); the values are, with the
provenance in the module's docstring.

The published standard k-epsilon prediction is Rong and Nielsen (2008), DCE Technical Report 46
(ADR-012 source 10, checksum there), figures 5(a), 5(b), 9(a) and 9(b): raster images embedded in
the PDF (265 x 404 to 705 x 294 pixels), extracted at their native size, the axes calibrated from
the tick marks on the axis box, the red line traced by continuity (the legend's segment refused)
and resampled at the measurement stations. One pixel is 0.008 m and 0.008 u/u0 on the vertical
lines, 0.014 to 0.023 m and 0.004 to 0.012 u/u0 on the horizontal ones. The check is the
figures' own measurement circles mapped through the same calibration against the workbook:
within 0.007 u/u0 wherever a circle matches a station, and on the vertical lines the circles sit
0.03 to 0.04 m from the workbook's stations in y, the figure's plotting rather than the
calibration (tick residuals below 0.005 m there), which puts about 0.05 u/u0 of uncertainty on
the published values in the jet, where du/dy is largest. Three stations the traced line does not
reach hold None. The departure of the published prediction from the data, line by line, is in
the module and in section 5.3's tables.

### 2.3 The runs

Item 3: standard with k and eps advected by `upwind` (the default, the key absent) and by
`umist` (the key stated), on both grids, four runs; plus RNG under the default on 216x72,
reported whatever it does. Each run is prompt 46's runner on the committed solver, from rest, one
BLAS thread, the cell of the largest change recorded every outer iteration; classified by step
5's rule as prompts 46 to 48 applied it (converged, bounded with its sub-class, growing,
diverged, positivity error). Each converged run gets the marked three-panel figure (streamlines
coloured by speed, nu_t / nu on a log scale, k; the slot and the outlet marked).

### 2.4 The comparison

For every converged run, u / u0 at the measurement stations of the four lines, sampled bilinearly
from the final u faces (`validation.annex20.sample_lines`): the largest difference and the RMS
on each line against the data, and against the published prediction; one figure per line with
the measured points, the runs and the published curve. Then two smaller comparisons at the same
stations: the two schemes against each other on each grid, and the two grids against each other
under each scheme.

## 3. Predictions

### 3.1 Orchestrator (2026-10-11; committed before any run)

- (a) Both schemes converge on both grids.
- (b) The wall jet along the ceiling (the line y = H - h/2) agrees with the data to within 0.1 u0
  up to x/H 2. The jet is the easy part for k-epsilon.
- (c) The return flow along the floor (the line y = h/2) carries the right sign everywhere, but
  its magnitude departs from the data by more than the jet's. ADR-012 notes that the symmetry
  plane carries less flux at x/H 2 than any 2D solution can.
- (d) On every line, the largest difference between the two schemes is under half the largest
  difference between either scheme and the data: the scheme choice is smaller than the model's
  error.
- (e) The two grids differ by less than either differs from the data.

What each outcome means:

- (d) holds: upwind is acceptable for rooms, and the product's default stands.
- (d) fails: upwind's smearing is comparable to the model's error, and the product's scheme
  becomes a question for Alex.
- (b) or (c) far off: the threshold Alex sets should be judged against Rong and Nielsen's
  published standard k-epsilon departure, not against the data alone.

### 3.2 Builder (2026-10-11, before any run; scored apart)

- (a) Holds. The room has no obstacle corner for the cycle of prompts 46 and 47 to sit at, the
  cell Reynolds numbers are below the product's, and the outlet arrangement is the one the
  fixed-flow outlets were built for.
- (b) Holds on 375x125 and fails on 216x72 at one or two stations: the jet's first node on the
  coarse grid sits at y+ near the floor's edge of the log layer, and the second-cell strain
  overshoot of ADR-012 C's note thickens the coarse jet faster than the measured one; the finer
  grid halves both.
- (c) Holds: the published standard k-epsilon floor line sits 0.05 u0 below the data along most
  of its length (section 2.2), and this model shares its closure.
- (d) Holds on the vertical lines and the floor, and is closest on the jet line, where the
  two schemes differ most in the shear layer under the jet; I expect it to hold there too, by
  less than a factor of two.
- (e) Holds everywhere except the jet line near the end wall (x > 8 m), where the measured
  corner eddy is what the model misses on either grid and the grids differ by about as much as
  the finer one differs from the data.

## 4. Stop conditions (from the prompt)

- The data cannot be obtained, or its four lines cannot be matched to the specification's.
- A positivity error, or a capped pressure correction.
- Neither scheme converges on either grid: report the characterisation and stop.

## 5. Results (written after the runs)

### 5.1 Order and cost

The five rows of item 3 were launched together on 2026-10-10 at 22:16 (`run49.sh flows`), one
BLAS thread each on twenty-four cores; the two 216x72 diagnostic rows of section 5.2 followed on
2026-10-11 beside them, and two 375x125 diagnostics with the tighter tolerance were started and
stopped unfinished for their cost (no record). The wall times in the tables were taken beside
other runs and beside the suite's VAL-017 test, and the two 375x125 rows of item 3 span a
night during which the machine slept, so their wall time and seconds per outer iteration are
not measurements; the twenty-iteration timing before the runs gave 0.11 s per outer iteration on
216x72 and 0.42 s on 375x125, alone.

### 5.2 Item 3: the runs

<!-- tables49 matrix begin -->
| Run | Class | Stop | Outer | Residual: least (at), end | Readings at stop (a), (b), (c), (d), (e) | Holds from (a), (b), (c), (d), (e) | Largest speed at end (m/s), cell | CG per correction: mean, largest | Cap hits | k, eps sweeps (mean) | Wall (s), per outer | Face hash |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| standard_216x72_upwind | bounded (falling) | max_simple_iter | 10,000 | 6.87e-10 (10,000), 6.87e-10 | 1.3e-06, 4.8e-15, 9.9e-12, 1.8e-18, - | -, 2,549, 2,585, 1, - | 0.47, (1.0625, 2.9375) | 506, 561 | 0 | 271.8, 181.5 | 3449, 0.345 | e0abd8ec606da62b |
| standard_216x72_upwind_tol1e-13 | bounded (falling) | max_simple_iter | 12,000 | 4.65e-11 (12,000), 4.65e-11 | 6.1e-08, 2.4e-15, 6.5e-12, 7.7e-19, - | 11,159, 2,549, 2,584, 1, - | 0.47, (1.0625, 2.9375) | 502, 561 | 0 | 392.5, 280.0 | 3443, 0.287 | dc4ed49e5598ec21 |
| standard_216x72_umist | bounded (falling) | max_simple_iter | 10,000 | 6.35e-10 (10,000), 6.35e-10 | 7.2e-07, 2.2e-15, 9.2e-12, 2.3e-18, - | 9,934, 2,532, 2,568, 1, - | 0.469, (1.1875, 2.9375) | 505, 562 | 0 | 277.3, 185.9 | 3577, 0.358 | 11d0922bdc7e09e9 |
| standard_216x72_umist_tol1e-13 | bounded (falling) | max_simple_iter | 12,000 | 3.87e-11 (12,000), 3.87e-11 | 6.7e-08, 2.1e-15, 6.8e-12, 2.5e-18, 1.7e-05 | 11,145, 2,532, 2,568, 1, - | 0.469, (1.1875, 2.9375) | 498, 562 | 0 | 399.1, 285.2 | 3506, 0.292 | 615f7e9011a637a1 |
| rng_216x72_upwind | bounded (stalled) | max_simple_iter | 10,000 | 2.26e-07 (10,000), 2.26e-07 | 0.0016, 3.8e-13, 3e-09, 3.2e-18, 0.25 | -, 5,500, 5,534, 1, - | 0.47, (1.4375, 2.9375) | 480, 568 | 0 | 244.8, 195.9 | 3397, 0.340 | a7f3fcc58f9f5b37 |
| standard_375x125_upwind | bounded (stalled) | max_simple_iter | 10,000 | 1.05e-06 (10,000), 1.05e-06 | 0.011, 2.5e-12, 2.1e-08, 7.8e-19, 4.5 | -, 6,482, 6,540, 1, - | 0.465, (0.636, 2.964) | 840, 1004 | 0 | 415.9, 351.2 | 37904, 3.790 | 58133c4dba3caecf |
| standard_375x125_upwind_sweeps2000 | bounded (stalled) | max_simple_iter | 10,000 | 1.05e-06 (10,000), 1.05e-06 | 0.011, 2.5e-12, 2.1e-08, 2.2e-18, 4.5 | -, 6,413, 6,449, 1, - | 0.465, (0.636, 2.964) | 843, 1004 | 0 | 558.9, 358.3 | 12005, 1.201 | 3c54b7f9c36922c9 |
| standard_375x125_umist | bounded (stalled) | max_simple_iter | 10,000 | 1.23e-06 (10,000), 1.23e-06 | 0.016, 2.6e-12, 2.6e-08, 1.6e-18, 5.8 | -, 6,256, 6,314, 1, - | 0.464, (0.636, 2.964) | 850, 961 | 0 | 409.5, 348.0 | 38032, 3.803 | f673061631392c20 |
| standard_375x125_umist_sweeps2000 | bounded (stalled) | max_simple_iter | 10,000 | 1.23e-06 (10,000), 1.23e-06 | 0.016, 2.6e-12, 2.6e-08, 2.5e-18, 5.8 | -, 6,203, 6,344, 1, - | 0.464, (0.636, 2.964) | 847, 961 | 0 | 551.9, 355.3 | 12144, 1.214 | dcf2fa9266c62442 |
<!-- tables49 matrix end -->

No run diverged, raised a positivity error or reached the pressure cap (the cap-hits column).
None stopped by the rule either. On 216x72 the standard rows reach the cap with the residual
still falling, the class step 5's rule names bounded (falling), conditions (b), (c) and (d)
holding from the first quarter of the run and (a) holding, or at its threshold, at the cap (the
matrix's readings and holds-from columns). On 375x125 they reach the cap stalled: the implicit k
and eps solves hit their 500-sweep cap from the middle of the run on (the k and eps sweeps
column; the logs mark the first at outer 2,300 for upwind), the first coupled runs to do so
(GitHub issue 58's case), and from there the residual falls by about half per thousand
iterations, with (b), (c) and (d) met and (a) four orders above its tolerance at the cap (0.011
and 0.016, about a hundredth of u0, against 1e-6). RNG on
216x72 is bounded (stalled), as it was on the product's exact grids (prompt 48). Two further
375x125 rows with the sweep cap raised to 2,000 (`_sweeps2000`, everything else the file's)
were started after the first pair finished; section 5.5 says what became of them.

**Why the 216x72 rows do not stop.** Condition (e), the eddy viscosity's estimated iteration
error over `nu_scale`, never holds, and the records say why:

<!-- tables49 rule_e begin -->
| Run | Class, outer | k, eps tol | nu_scale (m^2/s); inlet nu_t / nu; room's largest nu_t / nu | nu_t step at the end: floor (m^2/s), over tol x nu_scale; step ratio | (e): last finite estimate (at); (a) at the stop, holds from | (e) first holds at: rule's scale; nu + the room's largest nu_t | Would stop at (all five): rule's scale; the room's |
|---|---|---|---|---|---|---|---|
| standard_216x72_upwind | bounded (falling), 10,000 | - | 4.93e-05; 2.22; 898 | 1.9e-10, 3.8; 1.0002 | 0.19 (9,705); 1.3e-06, - | -; - | -; - |
| standard_216x72_upwind_tol1e-13 | bounded (falling), 12,000 | 1e-13 | 4.93e-05; 2.22; 898 | 4.9e-12, 0.099; 0.9976 | 0.0024 (11,963); 6.1e-08, 11,159 | -; 11,206 | -; 11,206 |
| standard_216x72_umist | bounded (falling), 10,000 | - | 4.93e-05; 2.22; 910 | 1e-10, 2.1; 1.0003 | 0.098 (9,824); 7.2e-07, 9,934 | -; - | -; - |
| standard_216x72_umist_tol1e-13 | bounded (falling), 12,000 | 1e-13 | 4.93e-05; 2.22; 910 | 3.5e-12, 0.07; 0.9963 | 1.7e-05 (12,000); 6.7e-08, 11,145 | -; 11,160 | -; 11,160 |
| rng_216x72_upwind | bounded (stalled), 10,000 | - | 4.72e-05; 2.08; 734 | 3.5e-08, 7.4e+02; 0.9978 | 0.25 (10,000); 0.0016, - | -; - | -; - |
| standard_375x125_upwind | bounded (stalled), 10,000 | - | 4.9e-05; 2.2; 880 | 3.1e-07, 6.4e+03; 0.9988 | 4.5 (10,000); 0.011, - | -; - | -; - |
| standard_375x125_upwind_sweeps2000 | bounded (stalled), 10,000 | 1e-10 | 4.9e-05; 2.2; 880 | 3.1e-07, 6.3e+03; 0.9988 | 4.5 (10,000); 0.011, - | -; - | -; - |
| standard_375x125_umist | bounded (stalled), 10,000 | - | 4.9e-05; 2.2; 886 | 3.6e-07, 7.3e+03; 0.9989 | 5.8 (10,000); 0.016, - | -; - | -; - |
| standard_375x125_umist_sweeps2000 | bounded (stalled), 10,000 | 1e-10 | 4.9e-05; 2.2; 886 | 3.6e-07, 7.3e+03; 0.9989 | 5.8 (10,000); 0.016, - | -; - | -; - |
<!-- tables49 rule_e end -->

The largest change of nu_t per outer iteration stops falling before the end of each run: it
floors at a few times the k and eps solves' tolerance times the scale (the floor column), so the
rate fit (e) rests on finds no contraction and its estimate is infinite from there (the last
finite estimate column). The scale is the molecular nu plus the inlet's eddy viscosity (ADR-012
E). On this case that is about twice nu, because the specification's inlet turbulence is weak,
while the room's shear layer makes several hundred nu (the scale column): (e) asks nu_t to settle
to a millionth of the inlet's value, a billionth of the room's. The two labelled diagnostics,
the 216x72 rows rerun with the k and eps solve tolerance a thousand times tighter (`_tol1e-13`,
`max_iter` 2000, the cap 12,000), lower the floor by about forty times and still do not stop:
the step then sits below the tolerance as a raw step, not as an estimate. Read from the same
records, the rule would have stopped at the table's last column under a scale of nu plus the
room's largest nu_t, with the tighter tolerance; under the file's tolerance it would not, since
(a) closes only where the nu_t step has already floored. Both are readings of the records, not
built changes: (e)'s scale is ADR-012 E's, and the decision is Alex's (section 7). The stricter
tolerance moves no line value by more than the rounding the test pins (section 5.4's table, the
`_tol1e-13` rows against their untagged twins).

The prompt's third stop condition, neither scheme converging on either grid, is met as the rule
classifies the runs. The comparison below is reported all the same, each row labelled by its
class, with the remaining iteration error stated beside it. Reading (a) is the estimated
iteration error of the velocity over the velocity scale, a field maximum. On 216x72 it is about
1e-6 of u0, far below every difference read. On 375x125 it is 0.011 to 0.016 of u0: an order
below the grid differences on the jet line, but of the same order as those on the vertical
lines and above the RMS grid difference on x = 2H (section 5.4). The scheme comparison (d) and
the grid comparison (e) hold with more than that margin everywhere except (e) under umist on
the jet line, where the margin is about twice the estimate (test 49, 2026-10-11). Whether the
error at the vertical lines' stations is below the grid differences there, rather than only
its field maximum, is not measured: the estimate's cell sits at the end-wall corner.

The rows compared, their readings and the first-node y+ (the ADR's estimate for 216x72 along
the jet was 27):

<!-- tables49 converged begin -->
| Run | Class | Outer, wall (s) | Readings at stop (a), (b), (c), (d), (e) | Holds from (a), (b), (c), (d), (e) | Inlet nu_t / nu | Room nu_t / nu: median, 95th, largest | y+ nodes; median, least, largest; share below 11.53 | Ceiling y+ for H < x < 2H: median (least, largest) | Floor y+ for H < x < 2H: median (least, largest) | Figure |
|---|---|---|---|---|---|---|---|---|---|---|
| standard_216x72_upwind | bounded (falling) | 10,000, 3449 | 1.3e-06, 4.8e-15, 9.9e-12, 1.8e-18, - | -, 2,549, 2,585, 1, - | 2.22 | 367, 821, 898 | 560; 22.6, 2.32, 46; 0.23 | 33 (29.5, 35.8) | 18 (14.7, 21.2) | ecr002_val017_standard_216x72_upwind.png |
| standard_216x72_upwind_tol1e-13 | bounded (falling) | 12,000, 3443 | 6.1e-08, 2.4e-15, 6.5e-12, 7.7e-19, - | 11,159, 2,549, 2,584, 1, - | 2.22 | 367, 821, 898 | 560; 22.6, 2.32, 46; 0.23 | 33 (29.5, 35.8) | 18 (14.7, 21.2) | ecr002_val017_standard_216x72_upwind_tol1e-13.png |
| standard_216x72_umist | bounded (falling) | 10,000, 3577 | 7.2e-07, 2.2e-15, 9.2e-12, 2.3e-18, - | 9,934, 2,532, 2,568, 1, - | 2.22 | 376, 832, 910 | 560; 22.6, 2.11, 48.2; 0.24 | 33.2 (29.6, 35.5) | 17.9 (14.5, 21.3) | ecr002_val017_standard_216x72_umist.png |
| standard_216x72_umist_tol1e-13 | bounded (falling) | 12,000, 3506 | 6.7e-08, 2.1e-15, 6.8e-12, 2.5e-18, 1.7e-05 | 11,145, 2,532, 2,568, 1, - | 2.22 | 376, 832, 910 | 560; 22.6, 2.11, 48.2; 0.24 | 33.2 (29.6, 35.5) | 17.9 (14.5, 21.3) | ecr002_val017_standard_216x72_umist_tol1e-13.png |
| rng_216x72_upwind | bounded (stalled) | 10,000, 3397 | 0.0016, 3.8e-13, 3e-09, 3.2e-18, 0.25 | -, 5,500, 5,534, 1, - | 2.08 | 320, 689, 734 | 560; 23.5, 2.76, 35.8; 0.21 | 34 (31.5, 34.9) | 19.2 (16.4, 21.9) | ecr002_val017_rng_216x72_upwind.png |
| standard_375x125_upwind | bounded (stalled) | 10,000, 37904 | 0.011, 2.5e-12, 2.1e-08, 7.8e-19, 4.5 | -, 6,482, 6,540, 1, - | 2.2 | 355, 803, 880 | 973; 12.7, 0.816, 26.2; 0.45 | 18.5 (16.7, 19.2) | 9.55 (7.54, 11.6) | ecr002_val017_standard_375x125_upwind.png |
| standard_375x125_upwind_sweeps2000 | bounded (stalled) | 10,000, 12005 | 0.011, 2.5e-12, 2.1e-08, 2.2e-18, 4.5 | -, 6,413, 6,449, 1, - | 2.2 | 355, 803, 880 | 973; 12.7, 0.816, 26.2; 0.45 | 18.5 (16.7, 19.2) | 9.55 (7.54, 11.6) | ecr002_val017_standard_375x125_upwind_sweeps2000.png |
| standard_375x125_umist | bounded (stalled) | 10,000, 38032 | 0.016, 2.6e-12, 2.6e-08, 1.6e-18, 5.8 | -, 6,256, 6,314, 1, - | 2.2 | 363, 809, 886 | 973; 12.7, 0.776, 27; 0.45 | 18.5 (16.7, 19.1) | 9.5 (7.49, 11.6) | ecr002_val017_standard_375x125_umist.png |
| standard_375x125_umist_sweeps2000 | bounded (stalled) | 10,000, 12144 | 0.016, 2.6e-12, 2.6e-08, 2.5e-18, 5.8 | -, 6,203, 6,344, 1, - | 2.2 | 363, 809, 886 | 973; 12.7, 0.776, 27; 0.45 | 18.5 (16.7, 19.1) | 9.5 (7.49, 11.6) | ecr002_val017_standard_375x125_umist_sweeps2000.png |
<!-- tables49 converged end -->

Along the ceiling jet between x = H and 2H the coarse grid's first node sits in the log layer and
the fine grid's at its lower edge (the ceiling column against the scalable floor of 11.53); on
the floor under the return flow both sit lower, the fine grid's below the floor. The figures,
one per compared row, `ecr002_val017_<row>.png` (streamlines coloured by speed, nu_t / nu, k;
the slot, the outlet and the four lines marked). The jet runs along the ceiling to the end wall,
turns down it into the outlet and feeds one room-filling eddy; the corner eddy under the ceiling
at the end wall, which the measurements show (section 5.3), is small on every grid and smaller
on the coarse one.

### 5.3 Item 1: the published prediction's own departure

<!-- tables49 published begin -->
| Line | Stations (held by the published line) | Published minus measured: largest |d| (at, m) | RMS | Mean |
|---|---|---|---|---|
| x = H (3.0 m) | 25 (25) | 0.149 (2.73) | 0.070 | -0.032 |
| x = 2H (6.0 m) | 25 (24) | 0.185 (0.12) | 0.093 | +0.059 |
| y = H - h/2 (2.916 m), the ceiling jet | 28 (27) | 0.366 (8.60) | 0.136 | +0.071 |
| y = h/2 (0.084 m), the floor | 28 (27) | 0.070 (8.94) | 0.052 | -0.041 |
<!-- tables49 published end -->

Rong and Nielsen's standard k-epsilon, digitized as section 2.2 describes, departs most from the
data on the ceiling-jet line at the end wall, where the measured corner eddy reverses the flow
and the published line does not (`ecr002_val017_line_y2916.png`), and least along the floor,
where it sits a constant few hundredths of u0 below the data over most of the length
(`ecr002_val017_line_y0084.png`).

### 5.4 Item 4: the comparison

<!-- tables49 lines begin -->
| Run | Class | Line | Against the data: largest |d| (at, m) | RMS | Mean | Against the published line: largest |d| | RMS | Stations |
|---|---|---|---|---|---|---|---|---|
| standard_216x72_upwind | bounded (falling) | x = H (3.0 m) | 0.123 (2.77) | 0.059 | -0.036 | 0.053 | 0.022 | 25 (25) |
| standard_216x72_upwind | bounded (falling) | x = 2H (6.0 m) | 0.074 (0.88) | 0.037 | +0.008 | 0.203 | 0.087 | 25 (24) |
| standard_216x72_upwind | bounded (falling) | y = H - h/2 (2.916 m), the ceiling jet | 0.179 (8.60) | 0.080 | -0.020 | 0.238 | 0.121 | 28 (27) |
| standard_216x72_upwind | bounded (falling) | y = h/2 (0.084 m), the floor | 0.086 (8.94) | 0.058 | -0.043 | 0.021 | 0.011 | 28 (27) |
| standard_216x72_upwind_tol1e-13 | bounded (falling) | x = H (3.0 m) | 0.123 (2.77) | 0.059 | -0.036 | 0.053 | 0.022 | 25 (25) |
| standard_216x72_upwind_tol1e-13 | bounded (falling) | x = 2H (6.0 m) | 0.074 (0.88) | 0.037 | +0.008 | 0.203 | 0.087 | 25 (24) |
| standard_216x72_upwind_tol1e-13 | bounded (falling) | y = H - h/2 (2.916 m), the ceiling jet | 0.179 (8.60) | 0.080 | -0.020 | 0.238 | 0.121 | 28 (27) |
| standard_216x72_upwind_tol1e-13 | bounded (falling) | y = h/2 (0.084 m), the floor | 0.086 (8.94) | 0.058 | -0.043 | 0.021 | 0.011 | 28 (27) |
| standard_216x72_umist | bounded (falling) | x = H (3.0 m) | 0.126 (2.73) | 0.060 | -0.035 | 0.046 | 0.019 | 25 (25) |
| standard_216x72_umist | bounded (falling) | x = 2H (6.0 m) | 0.075 (0.88) | 0.037 | +0.008 | 0.199 | 0.085 | 25 (24) |
| standard_216x72_umist | bounded (falling) | y = H - h/2 (2.916 m), the ceiling jet | 0.193 (8.03) | 0.088 | -0.029 | 0.273 | 0.138 | 28 (27) |
| standard_216x72_umist | bounded (falling) | y = h/2 (0.084 m), the floor | 0.088 (8.94) | 0.056 | -0.040 | 0.029 | 0.011 | 28 (27) |
| standard_216x72_umist_tol1e-13 | bounded (falling) | x = H (3.0 m) | 0.126 (2.73) | 0.060 | -0.035 | 0.046 | 0.019 | 25 (25) |
| standard_216x72_umist_tol1e-13 | bounded (falling) | x = 2H (6.0 m) | 0.075 (0.88) | 0.037 | +0.008 | 0.199 | 0.085 | 25 (24) |
| standard_216x72_umist_tol1e-13 | bounded (falling) | y = H - h/2 (2.916 m), the ceiling jet | 0.193 (8.03) | 0.088 | -0.029 | 0.273 | 0.138 | 28 (27) |
| standard_216x72_umist_tol1e-13 | bounded (falling) | y = h/2 (0.084 m), the floor | 0.088 (8.94) | 0.056 | -0.040 | 0.029 | 0.011 | 28 (27) |
| rng_216x72_upwind | bounded (stalled) | x = H (3.0 m) | 0.139 (2.73) | 0.080 | -0.042 | 0.059 | 0.033 | 25 (25) |
| rng_216x72_upwind | bounded (stalled) | x = 2H (6.0 m) | 0.093 (0.08) | 0.048 | +0.007 | 0.270 | 0.117 | 25 (24) |
| rng_216x72_upwind | bounded (stalled) | y = H - h/2 (2.916 m), the ceiling jet | 0.232 (8.03) | 0.099 | -0.014 | 0.313 | 0.162 | 28 (27) |
| rng_216x72_upwind | bounded (stalled) | y = h/2 (0.084 m), the floor | 0.120 (7.56) | 0.089 | -0.080 | 0.071 | 0.046 | 28 (27) |
| standard_375x125_upwind | bounded (stalled) | x = H (3.0 m) | 0.137 (2.73) | 0.066 | -0.036 | 0.023 | 0.010 | 25 (25) |
| standard_375x125_upwind | bounded (stalled) | x = 2H (6.0 m) | 0.076 (0.88) | 0.036 | +0.011 | 0.197 | 0.084 | 25 (24) |
| standard_375x125_upwind | bounded (stalled) | y = H - h/2 (2.916 m), the ceiling jet | 0.201 (8.60) | 0.069 | +0.013 | 0.176 | 0.082 | 28 (27) |
| standard_375x125_upwind | bounded (stalled) | y = h/2 (0.084 m), the floor | 0.099 (8.94) | 0.055 | -0.039 | 0.029 | 0.009 | 28 (27) |
| standard_375x125_upwind_sweeps2000 | bounded (stalled) | x = H (3.0 m) | 0.137 (2.73) | 0.066 | -0.036 | 0.023 | 0.010 | 25 (25) |
| standard_375x125_upwind_sweeps2000 | bounded (stalled) | x = 2H (6.0 m) | 0.076 (0.88) | 0.036 | +0.011 | 0.197 | 0.084 | 25 (24) |
| standard_375x125_upwind_sweeps2000 | bounded (stalled) | y = H - h/2 (2.916 m), the ceiling jet | 0.201 (8.60) | 0.069 | +0.013 | 0.176 | 0.082 | 28 (27) |
| standard_375x125_upwind_sweeps2000 | bounded (stalled) | y = h/2 (0.084 m), the floor | 0.099 (8.94) | 0.055 | -0.039 | 0.029 | 0.009 | 28 (27) |
| standard_375x125_umist | bounded (stalled) | x = H (3.0 m) | 0.140 (2.73) | 0.067 | -0.035 | 0.020 | 0.009 | 25 (25) |
| standard_375x125_umist | bounded (stalled) | x = 2H (6.0 m) | 0.076 (0.88) | 0.036 | +0.011 | 0.194 | 0.082 | 25 (24) |
| standard_375x125_umist | bounded (stalled) | y = H - h/2 (2.916 m), the ceiling jet | 0.188 (8.60) | 0.065 | +0.008 | 0.199 | 0.092 | 28 (27) |
| standard_375x125_umist | bounded (stalled) | y = h/2 (0.084 m), the floor | 0.102 (8.94) | 0.055 | -0.037 | 0.031 | 0.011 | 28 (27) |
| standard_375x125_umist_sweeps2000 | bounded (stalled) | x = H (3.0 m) | 0.140 (2.73) | 0.067 | -0.035 | 0.020 | 0.009 | 25 (25) |
| standard_375x125_umist_sweeps2000 | bounded (stalled) | x = 2H (6.0 m) | 0.076 (0.88) | 0.036 | +0.011 | 0.194 | 0.082 | 25 (24) |
| standard_375x125_umist_sweeps2000 | bounded (stalled) | y = H - h/2 (2.916 m), the ceiling jet | 0.188 (8.60) | 0.065 | +0.008 | 0.199 | 0.092 | 28 (27) |
| standard_375x125_umist_sweeps2000 | bounded (stalled) | y = h/2 (0.084 m), the floor | 0.102 (8.94) | 0.055 | -0.037 | 0.031 | 0.011 | 28 (27) |
<!-- tables49 lines end -->

The figures, one per line, with the measured points, every compared row and the digitized
published line: `ecr002_val017_line_x3.png`, `ecr002_val017_line_x6.png`,
`ecr002_val017_line_y2916.png`, `ecr002_val017_line_y0084.png`.

On three of the four lines this solver's standard variant sits nearer the data than the
published standard k-epsilon does, and on the floor line the two predictions sit nearer each
other than either sits to the data (the "against the published line" columns): the floor's
departure is the model's, shared by the two codes. On x = 2H the published line leaves the
return flow by the floor at half the measured strength (its wall cells), where this solver
follows the measurements to the floor (`ecr002_val017_line_x6.png`). The largest differences
are at the end wall on the two horizontal lines, in the corner the measurements show reversed,
at the jet's lower edge on x = H, and in the return flow 0.88 m above the floor on x = 2H.

**The two schemes against each other, on each grid:**

<!-- tables49 schemes begin -->
| Pair | Line | Between upwind and umist: largest |d| | RMS | upwind against the data: largest |d|, RMS | umist against the data: largest |d|, RMS | Ratio: between over the larger against the data |
|---|---|---|---|---|---|---|
| schemes, 216x72 | x = H (3.0 m) | 0.009 | 0.004 | 0.123, 0.059 | 0.126, 0.060 | 0.08 |
| schemes, 216x72 | x = 2H (6.0 m) | 0.004 | 0.002 | 0.074, 0.037 | 0.075, 0.037 | 0.05 |
| schemes, 216x72 | y = H - h/2 (2.916 m), the ceiling jet | 0.050 | 0.020 | 0.179, 0.080 | 0.193, 0.088 | 0.26 |
| schemes, 216x72 | y = h/2 (0.084 m), the floor | 0.008 | 0.004 | 0.086, 0.058 | 0.088, 0.056 | 0.09 |
| schemes, 375x125 | x = H (3.0 m) | 0.006 | 0.002 | 0.137, 0.066 | 0.140, 0.067 | 0.04 |
| schemes, 375x125 | x = 2H (6.0 m) | 0.003 | 0.001 | 0.076, 0.036 | 0.076, 0.036 | 0.04 |
| schemes, 375x125 | y = H - h/2 (2.916 m), the ceiling jet | 0.029 | 0.011 | 0.201, 0.069 | 0.188, 0.065 | 0.14 |
| schemes, 375x125 | y = h/2 (0.084 m), the floor | 0.006 | 0.002 | 0.099, 0.055 | 0.102, 0.055 | 0.06 |
<!-- tables49 schemes end -->

**The two grids against each other, under each scheme:**

<!-- tables49 grids begin -->
| Pair | Line | Between 216x72 and 375x125: largest |d| | RMS | 216x72 against the data: largest |d|, RMS | 375x125 against the data: largest |d|, RMS | Ratio: between over the larger against the data |
|---|---|---|---|---|---|---|
| grids, upwind | x = H (3.0 m) | 0.030 | 0.014 | 0.123, 0.059 | 0.137, 0.066 | 0.22 |
| grids, upwind | x = 2H (6.0 m) | 0.024 | 0.007 | 0.074, 0.037 | 0.076, 0.036 | 0.32 |
| grids, upwind | y = H - h/2 (2.916 m), the ceiling jet | 0.122 | 0.048 | 0.179, 0.080 | 0.201, 0.069 | 0.61 |
| grids, upwind | y = h/2 (0.084 m), the floor | 0.014 | 0.006 | 0.086, 0.058 | 0.099, 0.055 | 0.14 |
| grids, umist | x = H (3.0 m) | 0.026 | 0.011 | 0.126, 0.060 | 0.140, 0.067 | 0.18 |
| grids, umist | x = 2H (6.0 m) | 0.023 | 0.007 | 0.075, 0.037 | 0.076, 0.036 | 0.31 |
| grids, umist | y = H - h/2 (2.916 m), the ceiling jet | 0.157 | 0.058 | 0.193, 0.088 | 0.188, 0.065 | 0.82 |
| grids, umist | y = h/2 (0.084 m), the floor | 0.013 | 0.005 | 0.088, 0.056 | 0.102, 0.055 | 0.13 |
<!-- tables49 grids end -->

### 5.5 The fine-grid rows with the sweep cap raised

The two `_sweeps2000` rows (375x125, `max_iter` 2000, everything else the file's) are in the
tables above under their tag. They stop in the same state as the capped pair: the same class,
the same residual and readings at the cap to three figures (the matrix), the same floor of the
nu_t step (the condition (e) table), and every line value within the rounding the test pins of
the capped row's (section 5.4's table; the largest difference over the four lines is below 2e-6
u/u0 for either scheme). The k solve, free of the 500-sweep cap, takes several hundred sweeps
more on average and still reaches 2,000 at a few dozen outer iterations (the sweeps column, and
the scalar cap-hit counts in the records). So the sweep cap is not what stalls the fine grid:
the fine grid converges at about half the residual per thousand outer iterations under either
cap, and reaches the file's cap with an estimated iteration error of a hundredth of u0 (the
matrix's (a) column), which is what the grid comparison of section 5.4 carries. Where the k
and eps solves do reach their cap the fields do not move by more than that rounding, which is
the answer issue 58 was waiting for on this case.

## 6. Predictions against measurement

<!-- tables49 predictions begin -->
| Prediction | Measured | Holds |
|---|---|---|
| (a) both schemes converge on both grids | standard_216x72_upwind: bounded (falling); standard_216x72_umist: bounded (falling); standard_375x125_upwind: bounded (stalled); standard_375x125_umist: bounded (stalled) | no |
| (b) the jet within 0.1 u0 up to x/H 2 | standard_216x72_upwind (bounded (falling)): 0.074 at 0.01 m; standard_216x72_upwind_tol1e-13 (bounded (falling)): 0.074 at 0.01 m; standard_216x72_umist (bounded (falling)): 0.073 at 0.01 m; standard_216x72_umist_tol1e-13 (bounded (falling)): 0.073 at 0.01 m; rng_216x72_upwind (bounded (stalled)): 0.073 at 0.01 m; standard_375x125_upwind (bounded (stalled)): 0.081 at 0.01 m; standard_375x125_upwind_sweeps2000 (bounded (stalled)): 0.081 at 0.01 m; standard_375x125_umist (bounded (stalled)): 0.081 at 0.01 m; standard_375x125_umist_sweeps2000 (bounded (stalled)): 0.081 at 0.01 m | yes |
| (c) the floor's sign right everywhere, its departure above the jet's | standard_216x72_upwind (bounded (falling)): 22 of 28 signs, floor 0.086 against jet 0.179; standard_216x72_upwind_tol1e-13 (bounded (falling)): 22 of 28 signs, floor 0.086 against jet 0.179; standard_216x72_umist (bounded (falling)): 22 of 28 signs, floor 0.088 against jet 0.193; standard_216x72_umist_tol1e-13 (bounded (falling)): 22 of 28 signs, floor 0.088 against jet 0.193; rng_216x72_upwind (bounded (stalled)): 22 of 28 signs, floor 0.120 against jet 0.232; standard_375x125_upwind (bounded (stalled)): 21 of 28 signs, floor 0.099 against jet 0.201; standard_375x125_upwind_sweeps2000 (bounded (stalled)): 21 of 28 signs, floor 0.099 against jet 0.201; standard_375x125_umist (bounded (stalled)): 21 of 28 signs, floor 0.102 against jet 0.188; standard_375x125_umist_sweeps2000 (bounded (stalled)): 21 of 28 signs, floor 0.102 against jet 0.188 | no |
| (d) schemes differ by under half the larger distance from the data, every line | schemes, 216x72 x3: 0.009 against 0.5 x 0.126; schemes, 216x72 x6: 0.004 against 0.5 x 0.075; schemes, 216x72 y2916: 0.050 against 0.5 x 0.193; schemes, 216x72 y0084: 0.008 against 0.5 x 0.088; schemes, 375x125 x3: 0.006 against 0.5 x 0.140; schemes, 375x125 x6: 0.003 against 0.5 x 0.076; schemes, 375x125 y2916: 0.029 against 0.5 x 0.201; schemes, 375x125 y0084: 0.006 against 0.5 x 0.102 | yes |
| (e) grids differ by less than either differs from the data, every line | grids, upwind x3: 0.030 against 0.123; grids, upwind x6: 0.024 against 0.074; grids, upwind y2916: 0.122 against 0.179; grids, upwind y0084: 0.014 against 0.086; grids, umist x3: 0.026 against 0.126; grids, umist x6: 0.023 against 0.075; grids, umist y2916: 0.157 against 0.188; grids, umist y0084: 0.013 against 0.088 | yes |
<!-- tables49 predictions end -->

Orchestrator's, as written in section 3.1:

- (a) fails as the rule classifies the runs: no standard row stops by the rule on either grid,
  for the reason section 5.2 gives, every one of them falling at the cap.
- (b) holds on every compared row: the jet line is within 0.1 u0 of the data up to x/H 2 (the
  table's measured column), and its worst station there is the first one, at the slot's lip.
- (c) fails on both clauses as the table scores it, over the whole of each line. The sign
  clause: at the stations nearest the inlet wall, where the measured floor velocity is slightly
  forward and the computed (and the published) slightly backward, the signs differ, with both
  magnitudes small (`ecr002_val017_line_y0084.png`); 22 of 28 signs agree on 216x72 and 21 of 28
  on 375x125. The magnitude clause: the floor's largest departure (0.086 to 0.120) is below the
  jet line's (0.179 to 0.232) on every row, the jet's end-wall corner being the larger, and the
  RMS agrees. The prediction named no metric or extent; scored up to x/H 2, as (b) is, the
  floor's RMS departure is about twice the jet's and the magnitude clause would hold (test 49).
- (d) holds on every line and both grids, with margin: the schemes differ by a small fraction of
  either's distance from the data (the ratio column of section 5.4's schemes table), most on the
  jet line, least on the vertical lines.
- (e) holds on every line under both schemes, with the jet line the closest (the ratio column
  of section 5.4's grids table), scored from the fine-grid rows as they stand, stalled at the cap
  with an estimated iteration error of a hundredth of u0 (the matrix's (a) column), an order
  below the grid differences on the jet and of their order on the vertical lines.

Builder's (section 3.2), scored apart: (a) wrong, for a reason the builder did not consider,
the stopping rule's scale rather than the physics; (b) wrong on 216x72, which held where the
builder expected one or two stations out; (c) wrong on both clauses over the whole lines (on the
magnitude clause, right only if the jet is scored up to x/H 2); (d) right, with the jet line the closest as expected; (e) wrong at the end wall, where
the grids differ most but still by less than either differs from the data.

## 7. What this implies

Questions, not decisions.

1. **Upwind for rooms.** (d) holds with margin on both grids: the scheme moves the lines by a
   fraction of the model's own error, and the product's default stands. Does step 7's other case
   (VAL-019, the step) need the pair run as well, or is this result enough for rooms?
2. **The allowance for VAL-017.** ADR-012 G (iii)'s route is the published standard k-epsilon
   departure, line by line (section 5.3's table), on the lines decision 6 scores (x = H and the
   ceiling jet), with the symmetry-plane shortfall keeping x = 2H unscored. On both scored lines
   this solver's departure is below the published one (section 5.4's table), so an allowance of
   "no worse than Rong and Nielsen" passes as it stands; the alternative is a fixed band in u / u0
   on each scored line, which the same table gives the basis for. Three cautions: the published
   values near the jet carry the digitizing uncertainty of section 2.2; the floor line, which the
   two codes agree on and the data do not, is the model's departure and would fail any band
   tighter than the published one; and the scored lines' worst stations are the end-wall corner,
   where every 2D k-epsilon misses the measured eddy.
3. **Condition (e)'s scale.** The rule cannot stop this case because (e) is scaled by the inlet's
   eddy viscosity, which here is far below the room's (section 5.2). The product's inlet (5%,
   0.1 m) gave an inlet eddy viscosity of the core's order, which is why the product rows
   stopped; a HEPA ceiling with a weaker stated turbulence would behave as this room does. The
   records show the rule stopping under nu plus the room's largest nu_t with a tighter k and eps
   tolerance, and under neither change alone. Which scale, and which tolerance, is a change to
   ADR-012 E and `src/stopping.py` for Alex; until then the VAL-017 test pins the run's record at
   the cap and expects its convergence clause to fail (strict xfail).
4. **The k and eps sweep cap, and the fine grid's slowness.** On 375x125 the implicit solves
   reach `max_iter` 500 from mid-run (issue 58's open decision); with the cap at 2,000 the
   fields and the readings are the same to the test's rounding (section 5.5), so the cap costs
   nothing on this case and the decision can rest on that. What the fine grid does show is a
   coupled iteration that converges at half the rate per outer iteration of the coarse one and
   is far from the rule's stop at 10,000 iterations; the product's 320x120 grid is as fine. A
   cap of several times 10,000, or a convergence aid, is a question for step 8.
5. **The test's cost.** The VAL-017 test runs the case to its cap, about eighteen minutes alone
   on this machine, in the validation tier that CI runs. Does it stay there, or become a harness
   row as the 80x80 cavity and VAL-018 are?
6. **Two dimensions at x = 2H.** The solver follows the symmetry-plane profile there more closely
   than the flux argument of section 2.2 expects on average; the integrated shortfall is spread
   over the stations rather than concentrated in the return flow. Unscored, as decided.

## 8. Reproducing

From the repository root, `.venv` active, `results/builder33/sources/` and
`results/builder33b/sources/` holding the workbook and the PDF named in section 2.2 (untracked;
addresses and checksums there):

```
python docs/reports/probe49/digitize49.py curves      # the published line (probe venv: xlrd, pymupdf, cv2)
python docs/reports/probe49/data49.py                 # the tables of validation/annex20.py
bash docs/reports/probe49/run49.sh flows              # the five rows (an hour on 216x72, two on 375x125)
python docs/reports/probe49/annex49.py run 216x72 --tol 1e-13 --max-iter 2000 --cap 12000 --tag tol1e-13
python docs/reports/probe49/annex49.py run 216x72 --scheme umist --tol 1e-13 --max-iter 2000 --cap 12000 --tag tol1e-13
bash docs/reports/probe49/run49.sh tables             # every table, to results/builder49/tables_*.json
bash docs/reports/probe49/run49.sh figures            # every figure, beside this report
python docs/reports/probe49/assemble49.py --check     # the report's tables against the records
pytest tests/test_annex20.py -v                       # the unit tests, then the VAL-017 run
```
