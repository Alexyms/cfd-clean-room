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
