# ECR-002 Step 7: VAL-020, the Plane Impinging Jet

**Date:** 2026-10-15
**Tree:** branch `feature/ecr002-val020-impinging-jet` from main at 8acb475 (prompt 54 merged as PR 81;
multigrid is the default pressure preconditioner). Commit A (5749ce4) adds the case
(`configs/validation_impinging_jet.yaml`), its loader (`validation.cases.load_impinging_jet`, the presets
`val020_272x90` and `val020_408x135`) and the instruments; commit B (6b8e89d, after the predictions
and before any run was read) adds the digitised data with its provenance and the comparison code
(`validation/impinging_jet.py`), the harness family (`scripts/benchmark.py`, reported and not scored) and
their tests (`tests/test_impinging_jet.py`, `tests/impinging_jet_reference.py`). Nothing under `src/`
changes. Every run goes through the committed `StaggeredSolver` on the case as the suite loads it.
**Instruments:** `docs/reports/probe55/`: `digitize55.py` (the paper's figures to numbers), `data55.py`
(the tuples printed for the module), `impinging55.py` (the runs, through prompt 46's runner with prompt
52's probe-side additions), `tables55.py` (the tables), `figures55.py` (the figures) and the launcher
`run55.sh`. Raw output under `results/builder55/` (untracked).
**Order:** sections 1 to 4 were committed before any run of item 3, this file alone. Section 5 onward
was written after the runs.
**Policy:** `docs/REVIEW_POLICY.md`, "What blocks". The numbers live in the tables `tables55.py`
generates from the records; the sentences point at the tables. No threshold is set here: the allowance
is Alex's (ECR-002 criterion 13, OPEN).

## 1. The questions

Three, from the prompt. Which k-epsilon variant should the product run where downflow strikes the
equipment tops (ADR-012 A and decision 2; VAL-020 was named to decide it)? RNG has converged on no
coupled case so far (the product's exact grids, prompts 47 and 48; the step, prompt 52, diverging at the
outlet); if it does not converge here either, that is the answer for practical purposes. What allowance
does Alex set for VAL-020? And how does the solver behave at a stagnation point, where the standard
model is known to overpredict the turbulence (the stagnation anomaly), which is the equipment-top
regime?

## 2. Method (fixed before the runs)

### 2.1 The source and the case

Khayrullina, van Hooff, Blocken and van Heijst (2017), "PIV measurements of isothermal plane turbulent
impinging jets at moderate Reynolds numbers", Exp Fluids 58(4):31, DOI 10.1007/s00348-017-2315-0 (CC
BY). The file is `results/builder55/sources/khayrullina_2017.pdf`, SHA-256
c9d08bb01f1bc8ddff6dee2b5d437a3f0184dc77347b457bd29e8538ac05421d, fetched 2026-10-15 from
https://link.springer.com/content/pdf/10.1007/s00348-017-2315-0.pdf (the TU/e address the prompt names
answered 403; the Springer copy is the publisher's open-access file, 14 pages, 5,780,365 bytes).

The setup (its section 2): a water channel; a nozzle with a smooth contraction and a rectangular section
upstream of the exit, w_jet 16 mm or 8 mm wide and 300 mm deep (aspect ratio 18.75 or 37.5), flush with
the channel's top plate; the plate h_jet = 360 mm below (h_jet / w_jet 22.5 or 45); water, with the jet
Reynolds number Re = V0 w_jet / nu from 7,200 to 13,500 (Re 7,200 at 8 mm is V0 0.9 m/s, its section 3,
so nu = 1.0e-6 m^2/s); 2D PIV in the vertical centreplane, 5,000 uncorrelated samples per set, three
regions of interest (the whole plane 192 x 360 mm, the nozzle region 128 x 192 mm, the impingement
region 128 x 176 mm), the top 40 mm hidden by a beam of the channel. Fluid leaves along the channel,
whose length the paper does not state; it reports recirculation at the top of the channel with a lateral
speed up to 0.08 V0 (its section 4.2).

The case is its case I: Re 8,000, w_jet 16 mm, h_jet / w_jet 22.5, V0 = 0.5 m/s. It is the H/W 22.5 case
the same authors' RANS comparison also ran (Khayrullina, van Hooff, Blocken and van Heijst 2019, Eur J
Mech B/Fluids 75:228-243, DOI 10.1016/j.euromechflu.2018.10.003: cases at Re 8,000 and 13,000). That
paper is paywalled; its abstract (read 2026-10-15 on the Heriot-Watt research portal) is what this
report has of it: five steady RANS models (standard, realisable and RNG k-epsilon, SST k-omega, a
Reynolds stress model); SST best for the velocities and the turbulent kinetic energy near the nozzle at
both Re; RNG best for the potential core length in case I; RNG and realisable best for the centreline
velocities and TKE in case I, standard and RNG in case II; realisable best for the velocity overall,
realisable and RNG for TKE overall; the standard model most accurate for the jet decay rate at both Re.
No figure of it was available to digitise, so no published k-epsilon line stands beside the data here.

The inlet (the paper's figure 3): at y = 2 w_jet the mean profile is a top-hat with a displacement
thickness of 0.05 to 0.06 w_jet and a shape factor of 1.6 to 2 (turbulent); the turbulence intensity
I_uv = sqrt(u'^2 + v'^2) / V0 is about 0.05 in the core and 0.4 in the shear layers. The paper states
no profile at the exit itself, so the case takes the prompt's rule: a top-hat at V0 through the 16 mm
slot in the top wall, k from the core intensity and a dissipation length of w_jet / 10. The solver's
intensity is per component (k = 1.5 (I V0)^2, `src/turbulence.py`), while I_uv sums two measured
components; with the third taken as their mean, k = 0.75 I_uv^2 V0^2 = 4.69e-4 m^2/s^2, which the
file's I = 0.05 / sqrt(2) = 0.0354 gives. eps = k^(3/2) / l_e with l_e = 1.6 mm (ADR-012 C).

The enclosure. The top wall is the channel's top plate with the nozzle flush in it (the paper's figure
1), the plate the floor, both walls with the scalable wall functions. The paper does not state the
channel's length or where the water leaves, so the choice (section 2.4 checks it): the domain is 68
w_jet wide (1.088 m) with the nozzle at its centre, 33.5 w_jet (1.49 h_jet) from the nozzle's edge to
each side, and each side is a fixed-flow outlet over its whole height taking half the inflow at one
face velocity (V0 w_jet / 2 h_jet = 0.011 m/s; ADR-012 D as amended 2026-10-08). A pressure outlet
would take the wall jet out along the plate and the entrainment in above it, the sideways and reversed
arrangement under which the copy rule drifted (issue 61) and at which RNG diverged on the step (prompt
52). The frame is the solver's, y up from the plate; the paper's y runs down from the nozzle, and
`validation/impinging_jet.py` converts.

The fluid: water, rho 998 kg/m^3, mu = rho nu, nu 1.0e-6 m^2/s; nothing compared depends on rho. The
model and the solver keys are VAL-019's: the standard variant, k and eps advected by the default
(upwind), rule version 5, `cfl_number` 0.25, `momentum_sweeps` 10, k and eps by conjugate gradients at
`tol` 1e-10 with a sweep cap of 500, `pressure_rtol` 1e-4, `alpha_velocity` 0.5, `max_simple_iter`
20,000, multigrid the pressure preconditioner (the default since prompt 54). The mass-imbalance
tolerance is ADR-011 G's bound on the finer grid's smallest cell (5.91e-9 kg/s), one value for both
grids.

### 2.2 The data and its provenance

Every value is read from the paper's raster figures by `digitize55.py`, which calibrates each panel on
its gridlines and axis box (a linear map per axis fitted by least squares; the residuals are in the
record `results/builder55/khayrullina_2017_digitized.json` and printed by `data55.py`), and checked
against the numbers the paper states in its text. The values were printed into the module by
`data55.py`; nothing was typed, and `tests/test_impinging_jet.py` pins them by a hash.

- *The centreline velocity V_0,y / V0 against y / h_jet* (figure 4(a), a 1772 x 862 JPEG, panel (a)
  557 x 742 pixels; one pixel is 0.0022 V0 and 0.0013 h_jet; fit residuals 0.0005 and 0.0010). The
  three 16 mm series (Re 8,000 open circles, 10,000 filled squares, 13,000 crosses, all cyan) are
  plotted on top of one another and the paper says the distributions are identical for one nozzle, so
  each cyan blob is one station: its centroid is the value and its half-extent (0.008 to 0.012 V0) the
  uncertainty. 51 stations from y / h_jet 0.06 to 0.99; the cyan annotations (the "4.8 w_jet" arrow and
  text, the "0.12 h_jet" arrow and text) are excluded by rectangle. Three stations between 0.87 and 0.91
  lie under the orange dashed line and are not read.
- *The centreline turbulence intensity I_uv against y / h_jet* (figure 6, 969 x 1163, one pixel 0.0006
  I_uv and 0.0010 h_jet). The three 16 mm series are offset by up to 0.02 here, so the open circle (Re
  8,000) is read from the hole it encloses wherever that hole is found (30 stations) and the blob's
  centroid, the mean of the three, where it is not (10 stations, flagged in the module). 40 stations from
  0.11 to 0.99.
- *The half-width delta_0.5 against y / w_jet* (figure 8(a), 1500 x 1914 PNG: the white dots at
  2 delta_0.5 on the contour map, 11 pairs from y / w_jet 2.6 to 19.3, one pixel 0.028 w_jet; the half-
  width is half the distance between the left and right dot of a pair, so an offset of the jet axis
  cancels; the dots sit 0.03 w_jet right of the axis).
- *The lateral mean-velocity profile V / V_0,y against x / delta_0.5 at y' / w_jet = 4.8* (figure 5(a),
  one pixel 0.0055 in x / delta_0.5 and 0.0027 in V / V_0,y), y' from the virtual origin at y0 = -0.8
  w_jet, so y = 4.0 w_jet, y / h_jet 0.178: the green filled circles, found by eroding the green mask
  until only the filled markers' cores remain, 19 markers. The other seven heights the figure holds
  (2.8 and 18.4 are plain lines; 5.9 and 7.9 open triangles, 12.1 open circles, 14.9 crosses, 20.1 open
  squares) could not be digitised: the open markers are seven pixels across with the curves of their
  own colour drawn through them and the other curves crossing them, and neither a correlation with the
  legend's symbol nor the hole a marker encloses found more than a few of each. For those heights the
  paper's own fit stands in, V / V_0,y = exp(-0.693 (x / delta_0.5)^2), which it states the profiles
  follow for 5.9 <= y' / w_jet <= 20.1; every number resting on it says so.
- *The lateral I_uv profile at y' / w_jet = 4.8* (figure 7(a), 24 markers, the same reading). The other
  heights as above.
- The wall jet along the plate is not given as a profile (the contour maps of figures 8 and 9 only), so
  it is not compared.

The checks against the paper's text (the record's `checks`): the potential core, the first station with
V_0,y / V0 below 0.98, is 4.84 w_jet (stated 4.8); the half-velocity point is y / h_jet 0.82 (stated
"about 0.85"); the centreline I_uv peaks at 0.206 at y / h_jet 0.545 (stated 0.21 at 0.55); the
half-width at the first dot pair, y / w_jet 2.6, is 0.47 w_jet (the nozzle's 0.5) and its slope over
the paper's self-similar range (y / w_jet 7.5 to 17) is 0.10 (stated 0.10). The Gaussian against the
digitised 4.8 profile differs by 0.076 RMS, as the figure shows for a height below the self-similar
range. The digitising uncertainty is therefore about 0.01 V0 on the centreline, 0.004 in I_uv, 0.03
w_jet in the half-width and 0.01 on the lateral profile, all well inside the prompt's 0.05 V0.

What the 2D PIV gives for k. I_uv^2 V0^2 = u'^2 + v'^2, the in-plane components only; the model's k is
the half-sum of three. The module stores I_uv and compares the model's k / V0^2 with 0.75 I_uv^2 (the
out-of-plane component taken as the mean of the two measured) and reports 0.5 I_uv^2 (the in-plane
half-sum, a lower bound) beside it. Near the stagnation point the out-of-plane component is the one the
plate stretches (the paper's section 4.1), so the estimate there is the less certain.

### 2.3 The grids

Two presets, 272 x 90 (coarse) and 408 x 135 (fine), 1.5 times apart per axis. x is uniform at 4.0 mm
and 2.67 mm (4 and 6 cells across the nozzle): the mesh clusters only toward the domain's walls, so in
x it could cluster toward the outlets and not toward the nozzle at the centre, and the resolution the
nozzle needs is paid over the whole width. In y the mesh's symmetric clustering puts the fine cells
where the case needs them, at the plate and at the top wall the nozzle sits in, and the loader derives
the wall cell as half the uniform spacing (2.0 mm and 1.33 mm; ratios 1.029 and 1.019). First-node y+
on the plate is reported against the scalable floor, 11.53 (ADR-012 B).

Timing probes (100 outer iterations, both running together, one BLAS thread each): 0.116 s per outer
iteration on the coarse grid and 0.228 s on the fine, so the cap is 39 and 76 minutes alone and the
set of section 2.4 is inside the prompt's eight hours by a wide margin.

### 2.4 The runs and the comparison

Item 3, in parallel: the standard variant under `upwind` on both grids, under `umist` on the fine grid,
and RNG under `upwind` on the fine grid, reported whatever it does; the width check of item 2, the
coarse grid made 100 w_jet wide (50 w_jet per side, 2.2 h_jet) at the same cell size, 400 x 90; and the
harness row `val020_272x90`. Each run is classified by step 5's rule (`tables46.classify`): converged by
`error_estimate_and_continuity`, diverged, a positivity error, growing, or bounded at the cap with its
sub-class. A bounded run's fields enter the comparison labelled as such.

For every compared run, `validation.impinging_jet.evaluate` reads: the centreline V_0,y / V0 at the 51
data stations, with the largest and RMS difference over the whole line, over the free-jet part (y /
h_jet <= 0.70 from the nozzle, the paper's range) and over the impingement region (y / h_jet > 0.88);
the potential core length (the first station below 0.98 V0, interpolated), the half-velocity height and
the decay rate K_u (the slope of (V0 / V_0,y)^2 against (y - y0) / w_jet over y / h_jet 0.35 to 0.75)
beside the data's; the half-width delta_0.5 / w_jet at the 11 dot heights and its spreading rate K_y;
the self-similar lateral profile V / V_0,y at the digitised 4.8 stations (largest and RMS) and at the
Gaussian's at the six other heights; k / V0^2 along the centreline at the 40 I_uv stations against
0.75 I_uv^2; k at the stagnation point, three readings: the model's k in the first cell above the
plate on the axis, the model's largest k on the axis within the impingement region, and the model's k
at the data's nearest station to the plate (y / h_jet 0.99) against the data's estimate there; and y*
of the first node on the plate, the median within 10 w_jet of the axis and the value under the axis.
The two schemes against each other and the two grids against each other on every quantity, as VAL-017's
section 5.3. The side-distance check: the 100 w_jet run against the 68 w_jet run on the centreline, the
half-width and the lateral profile.

## 3. Predictions

### 3.1 Orchestrator (prompt 55, dated 2026-10-15; committed before item 3's runs)

- (a) Standard converges on both grids.
- (b) RNG: no prediction on convergence. If it converges, its centreline k at the stagnation point is
  lower than standard's.
- (c) Standard's centreline velocity in the free-jet part (above about 70% of the nozzle-to-plate
  distance from the plate, where the paper says the jet behaves as a free jet) is within 0.1 U_0 of the
  data, U_0 the exit velocity.
- (d) Standard overpredicts k at the stagnation point against the PIV estimate by more than 50% (the
  stagnation anomaly).
- (e) The schemes differ by under half the larger distance from the data on every quantity, and the
  grids by less than either differs from the data.

What each outcome means (the prompt's words): RNG fails to converge: the product stays standard;
VAL-018's variant clause is closed as "RNG not runnable in this solver on four cases", and probing RNG's
cycle becomes a Phase 6 question. RNG converges and (b) holds with its stagnation k nearer the data:
evidence for RNG at the equipment tops, weighed by Alex against its non-convergence elsewhere. (d)
holds: the product's turbulence over equipment tops is overstated, which overstates mixing and
deposition by turbulent diffusion there; the comparative claim needs that caveat.

A note on (c)'s range. The paper says the jet behaves as a free jet "until 70-75% of the total jet
height" measured from the nozzle, that is over the upper 70% of the distance. The prompt's parenthesis
reads the other way (the 30% nearest the nozzle). Section 6 scores (c) on the paper's range, y / h_jet
<= 0.70 from the nozzle, and reports the narrower reading beside it.

### 3.2 Builder (2026-10-15, before item 3's runs; scored apart)

- (f) The side-distance check: the 100 w_jet run's centreline differs from the 68 w_jet run's by less
  than 0.02 V0 at every station and its half-width by less than 0.05 w_jet at every dot height.
- (g) The standard model with a top-hat inlet at I 0.035 grows its shear layers more slowly than the
  measured nozzle does, so its potential core is longer than the data's 4.8 w_jet, between 6 and 9
  w_jet, and its centreline lies above the data over the free-jet part.
- (h) Its spreading rate K_y lies between 0.10 and 0.12 (the plane-jet overprediction of the standard
  model).
- (i) The first-node y+ on the plate under the wall jet lies between 5 and 30 on the coarse grid, so
  the scalable floor is active under part of the plate on the fine grid.
- (j) RNG does not converge (bounded or diverged).

## 4. Stop conditions (from the prompt)

The paper could not be fetched or its figures not digitised to 0.05 V0: not met (section 2.2). A
positivity error, a capped pressure correction or a drifting outlet on a standard run: watched in
every record (the positivity field, `cap_hits` and the mean-pressure history). The set exceeding eight
hours: not met (section 2.3).

## 5. Results (written after the runs)

Every number below is a table's, generated by `tables55.py` from the records under
`results/builder55/` and spliced in by `assemble55.py`; a sentence restates nothing a table does not
hold. The five runs were launched together on 2026-10-15 at 09:49 in a worktree at commit 8b13416
(the predictions), one BLAS thread each, with the harness row running beside them in a second
worktree at 6b8e89d; the records carry their commits.

### 5.1 Item 3: the runs

<!-- tables55 matrix begin -->
| Run | Class | Stop | Outer | Residual: least (at), end | Readings at stop (a), (b), (c), (d), (e) | Holds from (a), (b), (c), (d), (e) | Largest speed at end (m/s), cell | CG per correction: mean, largest | Cap hits | k, eps CG iterations (mean); capped | p_mean drift (Pa per outer, last 200) | Wall (s), per outer | Face hash |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| standard 272x90 (coarse) upwind | bounded (neither) | max_simple_iter | 20,000 | 4.11e-05 (1,763), 0.000124 | -, 1.4e-07, 4.4e-06, 2.2e-16, 3.5e+02 | -, -, -, 1, - | 0.507, (0.542, 0.3359) | 17.9, 32 | 0 | 41.2, 33.1; 0 | 7e-07 | 8181, 0.409 | f53650e3b8e3d60a |
| standard 408x135 (fine) upwind | bounded (stalled) | max_simple_iter | 20,000 | 4.64e-05 (2,045), 7.12e-05 | -, 7.2e-08, 3.3e-06, 4.5e-16, - | -, -, -, 1, - | 0.505, (0.54, 0.3447) | 23, 44 | 0 | 52.1, 41.4; 0 | 1.6e-06 | 1.148e+04, 0.574 | 1594e89c8dca8509 |
| standard 408x135 (fine) umist | bounded (stalled) | max_simple_iter | 20,000 | 4.89e-05 (2,020), 7.19e-05 | -, 5.2e-08, 1.9e-06, 5.1e-16, - | -, -, -, 1, - | 0.506, (0.548, 0.343) | 23.1, 39 | 0 | 52.9, 42.5; 0 | -2.9e-07 | 1.201e+04, 0.6 | 9027d3ffc01cf347 |
| rng 408x135 (fine) upwind | bounded (neither) | max_simple_iter | 20,000 | 1.8e-06 (19,981), 2.71e-06 | 0.0078, 8.8e-10, 5.2e-08, 4.6e-16, 1.5 | -, 19,613, 18,076, 1, - | 0.502, (0.5453, 0.2416) | 29.2, 41 | 0 | 48.2, 41.6; 0 | 1e-05 | 1.197e+04, 0.598 | 971d55e89bfa236c |
| standard 272x90 (coarse) upwind, 100 w wide | bounded (neither) | max_simple_iter | 20,000 | 7.92e-05 (4,434), 0.000417 | 9.4, 2.3e-07, 8.8e-06, 4.2e-16, 3.5e+02 | -, -, -, 1, - | 0.508, (0.798, 0.3332) | 20, 32 | 0 | 48.6, 36.8; 0 | 0.00013 | 9362, 0.468 | 0a062b366b2e67dd |
<!-- tables55 matrix end -->

No run converged by the rule; all five reached the 20,000 cap bounded. The standard runs sit in a
cycle whose residual stays between 5e-5 and 1.5e-4 from about the 2,000th outer iteration, with
condition (a)'s estimate undefined (the velocity step does not contract), (b) and (c) an order of
magnitude above their tolerances and (d) met throughout. RNG's residual falls monotonically to
2.7e-6 at the cap, below every standard run's, with (b) met from 19,613 and (c) from 18,076; its
velocity-step and viscosity-step estimates, 0.008 and 1.5 against the tolerance 1e-6, keep (a) and
(e) from holding, so it is bounded under the classification, falling rather than cycling. No pressure correction hit its cap, no positivity error was raised, and the outlets did
not drift: the mean pressure moves by at most 1.3e-4 Pa per outer iteration over the last 200 (the
width check), against a dynamic pressure of 125 Pa, five orders below the drift the copy rule
showed on the product room (12 Pa per iteration, prompt 41). The three stop conditions of section
4 were not met.

Where the cycle lives:

<!-- tables55 cycle begin -->
| Run | Class | du: share beside an outlet / in the jet; median place (from the side, from the top); step median, largest (m/s) | dv: the same | d nu_t: the same (m^2/s) |
|---|---|---|---|---|
| standard 272x90 (coarse) upwind | bounded (neither) | 1.00 / 0.00; side 3.4 w, top 0.1 w; 0.00037, 0.00054 | 1.00 / 0.00; side 3.4 w, top 0.2 w; 0.00013, 0.00015 | 1.00 / 0.00; side 3.4 w, top 0.2 w; 4.4e-05, 7.4e-05 |
| standard 408x135 (fine) upwind | bounded (stalled) | 1.00 / 0.00; side 3.4 w, top 0.0 w; 0.00039, 0.00054 | 1.00 / 0.00; side 3.2 w, top 0.1 w; 0.00012, 0.00016 | 1.00 / 0.00; side 3.4 w, top 0.1 w; 4.2e-05, 6.1e-05 |
| standard 408x135 (fine) umist | bounded (stalled) | 1.00 / 0.00; side 3.4 w, top 0.0 w; 0.00041, 0.00054 | 1.00 / 0.00; side 3.2 w, top 0.1 w; 0.00014, 0.00017 | 1.00 / 0.00; side 3.4 w, top 0.1 w; 4.6e-05, 6.6e-05 |
| rng 408x135 (fine) upwind | bounded (neither) | 1.00 / 0.00; side 3.1 w, top 0.0 w; 5.6e-05, 0.00015 | 1.00 / 0.00; side 3.1 w, top 0.1 w; 1.5e-05, 3.7e-05 | 1.00 / 0.00; side 3.2 w, top 0.1 w; 3.7e-06, 1.1e-05 |
| standard 272x90 (coarse) upwind, 100 w wide | bounded (neither) | 1.00 / 0.00; side 3.1 w, top 22.4 w; 0.0019, 0.0027 | 1.00 / 0.00; side 3.9 w, top 22.3 w; 0.00021, 0.00033 | 1.00 / 0.00; side 3.6 w, top 0.2 w; 5.3e-05, 8.4e-05 |
<!-- tables55 cycle end -->

In every bounded run the cell of the largest change of u, v and nu_t over the last 2,000
iterations is beside a fixed-flow outlet, 3 to 4 w_jet from the side wall along the top wall (the
68 w runs) or along the plate (the 100 w run), where the recirculation that fills the domain
(figure 8) turns into the outlet's uniform outflow. None of the 2,000 samples lies within 6 w_jet
of the axis. The jet itself is not what keeps the rule from stopping; the same kind of corner cycle
appeared beside the product room's outlets (prompts 46 and 47) and at the step's outlet (prompt
52). The compared fields are the cap's, labelled "at the cap" in every table.

### 5.2 Item 2: the side distance

<!-- tables55 width begin -->
| Quantity | 68 w minus 100 w, largest over the stations |
|---|---|
| centreline V_0,y / V0 (51 stations) | 0.0309 (RMS 0.0194) |
| half-width delta_0.5 / w (11 heights) | 0.073 |
| lateral V / V_0,y at 4.8 (19 stations) | 0.0176 |
| k / V0^2 on the axis (40 stations) | 0.002519 |
| k / V0^2 at the stagnation station | 0.03714 vs 0.0384 |
| potential core (w) | 6.53 vs 6.46 |
| fields | at the cap, bounded (neither) vs at the cap, bounded (neither) |
<!-- tables55 width end -->

The 100 w_jet domain moves the centreline by up to 0.031 V0 (RMS 0.019) and the half-width by up to
0.073 w_jet, both towards the data (the width check's centreline RMS against the data is 0.050
against the 68 w run's 0.066, section 5.3). The choice of section 2.1 therefore does move the jet
at the few-percent level: the recirculation cells the confined domain forms (figure 8, the whole
domain) induce a downward co-flow beside the jet, visible in the lateral profiles' tails (section
5.5), and a wider channel weakens it. The stagnation k and the potential core move by about 1%.
The case file keeps the 68 w_jet domain, which is what every table reports; section 7 asks whether
to widen it.

### 5.3 Item 4: the centreline

<!-- tables55 centreline begin -->
| Run | Fields | Potential core (w) | Half-velocity (y/h) | K_u | Whole line: largest, RMS, mean (stations) | Free jet y/h <= 0.70 | Nearest the nozzle y/h <= 0.30 | Impingement y/h > 0.88 |
|---|---|---|---|---|---|---|---|---|
| data (digitised) | - | 4.82 | 0.821 | 0.195 | - | - | - | - |
| paper's text | - | 4.8 | 0.85 | 0.14 to 0.15 | - | - | - | - |
| standard 272x90 (coarse) upwind | at the cap, bounded (neither) | 6.53 | 0.858 | 0.118 | 0.0996, 0.0665, 0.06 (51) | 0.0996, 0.0671, 0.064 (39) | 0.0619, 0.0463, 0.045 (16) | 0.0318, 0.0177, -0.0082 (4) |
| standard 408x135 (fine) upwind | at the cap, bounded (stalled) | 7.56 | 0.873 | 0.109 | 0.134, 0.0956, 0.084 (51) | 0.134, 0.0957, 0.085 (39) | 0.0938, 0.0458, 0.037 (16) | 0.0401, 0.0317, 0.015 (4) |
| standard 408x135 (fine) umist | at the cap, bounded (stalled) | 7.53 | 0.868 | 0.115 | 0.123, 0.0889, 0.078 (51) | 0.123, 0.0904, 0.081 (39) | 0.0956, 0.0475, 0.039 (16) | 0.0306, 0.0237, 0.0052 (4) |
| rng 408x135 (fine) upwind | at the cap, bounded (neither) | 10.1 | 0.892 | 0.092 | 0.213, 0.151, 0.12 (51) | 0.213, 0.152, 0.12 (39) | 0.0955, 0.0367, 0.02 (16) | 0.0946, 0.0722, 0.054 (4) |
| standard 272x90 (coarse) upwind, 100 w wide | at the cap, bounded (neither) | 6.46 | 0.846 | 0.141 | 0.0689, 0.0503, 0.045 (51) | 0.0689, 0.0522, 0.051 (39) | 0.0587, 0.0462, 0.045 (16) | 0.0334, 0.0228, -0.021 (4) |
<!-- tables55 centreline end -->

The standard model with the top-hat inlet keeps its potential core 6.5 w_jet (coarse) and 7.6 w_jet
(fine) long against the data's 4.8, and from there decays at a rate K_u of 0.11 to 0.12 against the
digitised line's 0.195, so its centreline lies above the data by 0.064 to 0.085 V0 on average over the
free-jet part and by at most 0.10 V0 (coarse) or 0.13 V0 (fine) anywhere; it reaches half speed at
y / h_jet 0.86 to 0.87 against the data's 0.82. In the impingement region it is within 0.03 to 0.04
V0. RNG keeps a 10 w_jet core and lies 0.21 V0 above the data at worst. On K_u: the paper's text
gives 0.14 to 0.15 for the 16 mm nozzle, but its figure 4(b) read as a slope over the self-similar
range gives 0.20, and read as the ratio the paper defines gives 0.16 to 0.18; the table reports the
digitised slope, with the text's number beside it, and the discrepancy is the paper's own.

### 5.4 The half-width

<!-- tables55 halfwidth begin -->
| Run | Fields | y/w 2.6 | y/w 4.1 | y/w 5.7 | y/w 7.2 | y/w 8.7 | y/w 10.2 | y/w 11.7 | y/w 14.7 | y/w 16.2 | y/w 17.7 | y/w 19.3 | Largest, RMS | K_y |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| data (dots) | - | 0.469 | 0.553 | 0.619 | 0.73 | 0.853 | 0.989 | 1.13 | 1.44 | 1.59 | 1.79 | 2.07 | - | 0.0992 |
| paper's text | - | - | - | - | - | - | - | - | - | - | - | - | - | 0.1 |
| standard 272x90 (coarse) upwind | at the cap, bounded (neither) | 0.573 | 0.639 | 0.723 | 0.846 | 1.01 | 1.18 | 1.34 | 1.65 | 1.79 | 1.93 | 2.14 | 0.218, 0.154 | 0.104 |
| standard 408x135 (fine) upwind | at the cap, bounded (stalled) | 0.591 | 0.638 | 0.69 | 0.773 | 0.9 | 1.05 | 1.2 | 1.49 | 1.61 | 1.74 | 1.93 | 0.134, 0.0759 | 0.0958 |
| standard 408x135 (fine) umist | at the cap, bounded (stalled) | 0.591 | 0.638 | 0.69 | 0.777 | 0.912 | 1.07 | 1.22 | 1.52 | 1.66 | 1.8 | 2 | 0.122, 0.0776 | 0.1 |
| rng 408x135 (fine) upwind | at the cap, bounded (neither) | 0.543 | 0.588 | 0.646 | 0.701 | 0.781 | 0.905 | 1.06 | 1.36 | 1.48 | 1.6 | 1.76 | 0.302, 0.125 | 0.0958 |
| standard 272x90 (coarse) upwind, 100 w wide | at the cap, bounded (neither) | 0.572 | 0.635 | 0.706 | 0.813 | 0.957 | 1.11 | 1.27 | 1.58 | 1.74 | 1.91 | 2.16 | 0.152, 0.115 | 0.105 |
<!-- tables55 halfwidth end -->

The computed half-width is 0.10 to 0.12 w_jet above the dots at the nozzle (0.57 to 0.59 against
0.47, a four- or six-cell nozzle resolving the exit profile) and follows the dots' growth from there:
the spreading rate K_y is 0.096 to 0.105 on the standard runs against the data's 0.099 and the text's
0.10, inside the digitising uncertainty. The coarse grid spreads a little faster than the fine over
the first half of the jet and is up to 0.22 w_jet above the dots; the fine grid is within 0.13
w_jet of the dots everywhere.

### 5.5 The lateral profiles

<!-- tables55 lateral begin -->
| Run | Fields | y'/w 4.8 (digitised) | y'/w 5.9 (Gaussian) | y'/w 7.9 (Gaussian) | y'/w 12.1 (Gaussian) | y'/w 14.9 (Gaussian) | y'/w 20.1 (Gaussian) | I_uv at 4.8 |
|---|---|---|---|---|---|---|---|---|
| standard 272x90 (coarse) upwind | at the cap, bounded (neither) | 0.19, 0.132; delta 0.633 w | 0.114, 0.0822; delta 0.688 w | 0.157, 0.0989; delta 0.837 w | 0.203, 0.107; delta 1.3 w | 0.194, 0.101; delta 1.6 w | 0.0945, 0.0541; delta 2.15 w | 0.0929, 0.045 |
| standard 408x135 (fine) upwind | at the cap, bounded (stalled) | 0.166, 0.109; delta 0.636 w | 0.106, 0.0654; delta 0.667 w | 0.15, 0.0939; delta 0.766 w | 0.196, 0.104; delta 1.16 w | 0.191, 0.101; delta 1.43 w | 0.0967, 0.0558; delta 1.94 w | 0.0728, 0.0404 |
| standard 408x135 (fine) umist | at the cap, bounded (stalled) | 0.176, 0.116; delta 0.635 w | 0.111, 0.0703; delta 0.667 w | 0.151, 0.097; delta 0.77 w | 0.194, 0.103; delta 1.19 w | 0.188, 0.0988; delta 1.47 w | 0.0947, 0.0547; delta 2 w | 0.0702, 0.0402 |
| rng 408x135 (fine) upwind | at the cap, bounded (neither) | 0.0951, 0.0569; delta 0.583 w | 0.091, 0.0423; delta 0.627 w | 0.179, 0.0943; delta 0.697 w | 0.25, 0.134; delta 1.02 w | 0.245, 0.128; delta 1.31 w | 0.111, 0.0623; delta 1.77 w | 0.0871, 0.0568 |
| standard 272x90 (coarse) upwind, 100 w wide | at the cap, bounded (neither) | 0.183, 0.126; delta 0.63 w | 0.105, 0.0739; delta 0.678 w | 0.116, 0.0845; delta 0.805 w | 0.115, 0.0681; delta 1.23 w | 0.11, 0.0624; delta 1.52 w | 0.0704, 0.0434; delta 2.17 w | 0.101, 0.0471 |
<!-- tables55 lateral end -->

At the one digitised height, y' / w_jet 4.8 (y / h_jet 0.178), the measured profile is still
flat-topped (the figure's green circles) while the computed one is already rounded, so the RMS is
0.11 to 0.13 with the largest difference 0.17 to 0.19 at the shoulders; RNG, with its longer core,
is flatter there and within 0.10. At the five Gaussian heights the computed profiles match the fit
within 0.1 over |x / delta_0.5| <= 1 and lie above it in the tails, by 0.1 to 0.2 of V_0,y at
|x / delta_0.5| = 2, where the confined domain's recirculation carries a downward co-flow beside the
jet that the open channel of the experiment does not (the 100 w run's tails are lower, section 5.2;
the paper's figure 8(c) shows the ambient nearly at rest). Those numbers rest on the paper's fit,
not on digitised measurements (section 2.2). The lateral I_uv profile at 4.8 is matched within 0.06
RMS by every run.

### 5.6 k on the axis, the stagnation point and the plate

<!-- tables55 turbulence begin -->
| Run | Fields | k/V0^2 on the axis vs 0.75 I_uv^2 over the ring stations: largest, RMS, mean (stations) | Median ratio model / estimate (ring stations) | Stagnation, the last ring station: model vs estimate (k/V0^2) | Stagnation, the last station (a centroid): model vs estimate | First cell above the plate | Largest on the axis in the impingement region | Plate y+: median within 10 w, at the axis, largest; floor |
|---|---|---|---|---|---|---|---|---|
| standard 272x90 (coarse) upwind | at the cap, bounded (neither) | 0.0117, 0.007, -0.0055 (30) | 0.711 | 0.0371 vs 0.0296 (in-plane 0.0197) at y/h 0.973; ratios 1.26, 1.88 | 0.0236 vs 0.0259 at y/h 0.995; ratios 0.912, 1.37 | 0.0127 | 0.0376 at y/h 0.98 | 26.1, 30.9, 31.2; floor 11.53 |
| standard 408x135 (fine) upwind | at the cap, bounded (stalled) | 0.0146, 0.00927, -0.0074 (30) | 0.644 | 0.038 vs 0.0296 (in-plane 0.0197) at y/h 0.973; ratios 1.28, 1.93 | 0.0346 vs 0.0259 at y/h 0.995; ratios 1.34, 2.01 | 0.0119 | 0.0389 at y/h 0.983 | 17.2, 19.9, 20.3; floor 11.53 |
| standard 408x135 (fine) umist | at the cap, bounded (stalled) | 0.0136, 0.00883, -0.0068 (30) | 0.675 | 0.0399 vs 0.0296 (in-plane 0.0197) at y/h 0.973; ratios 1.35, 2.02 | 0.0362 vs 0.0259 at y/h 0.995; ratios 1.4, 2.1 | 0.012 | 0.0407 at y/h 0.979 | 17.2, 20, 20.5; floor 11.53 |
| rng 408x135 (fine) upwind | at the cap, bounded (neither) | 0.0237, 0.0115, -0.0092 (30) | 0.618 | 0.03 vs 0.0296 (in-plane 0.0197) at y/h 0.973; ratios 1.01, 1.52 | 0.0262 vs 0.0259 at y/h 0.995; ratios 1.01, 1.52 | 0.0108 | 0.0306 at y/h 0.979 | 17.1, 18.7, 19.3; floor 11.53 |
| standard 272x90 (coarse) upwind, 100 w wide | at the cap, bounded (neither) | 0.0101, 0.00624, -0.0043 (30) | 0.773 | 0.0384 vs 0.0296 (in-plane 0.0197) at y/h 0.973; ratios 1.3, 1.95 | 0.0239 vs 0.0259 at y/h 0.995; ratios 0.924, 1.39 | 0.0127 | 0.0386 at y/h 0.98 | 25.7, 30.9, 31.2; floor 11.53 |
<!-- tables55 turbulence end -->

Along the free jet the model's k on the axis is below the PIV estimate 0.75 I_uv^2 V0^2: the median
ratio is 0.64 to 0.71 for the standard runs and 0.62 for RNG over the 30 stations that hold the Re
8,000 circle's own value (the ten centroid stations, which carry the other two series' symbols up to
0.02 above it in I_uv, are reported beside them with the suffix `_all`). The computed k rises
through the impingement region to a maximum on the axis at y / h_jet 0.98 of 0.038 to 0.041 V0^2
(standard) and 0.031 (RNG). At the last ring station, y / h_jet 0.973, the standard model reads
0.037 to 0.040 against the estimate 0.030: 1.26 to 1.35 times it, or 1.9 to 2.0 times the in-plane
half-sum. RNG reads 0.030 there, 1.01 times the estimate. At the data's last station, y / h_jet
0.995, a centroid station 2 mm above the plate, the fine standard runs read 1.34 to 1.40 times the
estimate and the coarse 0.91, the first cell above the plate being 0.012 to 0.013 on every grid.
The plate's first node is above the scalable floor on both grids: y+ median 26 (coarse) and 17
(fine) within 10 w_jet of the axis, 31 and 20 at the axis.

### 5.7 The two schemes and the two grids

<!-- tables55 schemes begin -->
| Quantity | fine upwind | fine umist | |difference| | first from data | second from data | difference / larger distance |
|---|---|---|---|---|---|---|
| centreline_rms_all | 0.09557 | 0.08894 | 0.00663 | 0.0956 | 0.0889 | 0.0694 |
| centreline_rms_free_jet | 0.09575 | 0.09045 | 0.0053 | 0.0957 | 0.0904 | 0.0553 |
| centreline_rms_impingement | 0.03166 | 0.02375 | 0.00791 | 0.0317 | 0.0237 | 0.25 |
| potential_core_w | 7.558 | 7.533 | 0.025 | 2.73 | 2.71 | 0.00915 |
| decay_rate | 0.1087 | 0.1155 | 0.00677 | 0.0865 | 0.0797 | 0.0783 |
| half_width_rms | 0.07588 | 0.0776 | 0.00173 | 0.0759 | 0.0776 | 0.0222 |
| spreading_rate | 0.09582 | 0.1002 | 0.00436 | 0.00339 | 0.000968 | 1.29 |
| k_axis_rms | 0.009266 | 0.008827 | 0.00044 | 0.00927 | 0.00883 | 0.0474 |
| stagnation_k | 0.03801 | 0.03988 | 0.00187 | 0.0121 | 0.014 | 0.133 |
| lateral_4.8_rms | 0.1094 | 0.1164 | 0.00693 | 0.109 | 0.116 | 0.0595 |
| lateral_5.9_rms | 0.06542 | 0.07034 | 0.00492 | 0.0654 | 0.0703 | 0.07 |
| lateral_7.9_rms | 0.09386 | 0.09704 | 0.00318 | 0.0939 | 0.097 | 0.0327 |
| lateral_12.1_rms | 0.1044 | 0.1033 | 0.00111 | 0.104 | 0.103 | 0.0106 |
| lateral_14.9_rms | 0.1005 | 0.09878 | 0.00173 | 0.101 | 0.0988 | 0.0172 |
| lateral_20.1_rms | 0.05584 | 0.05469 | 0.00115 | 0.0558 | 0.0547 | 0.0206 |
<!-- tables55 schemes end -->

<!-- tables55 grids begin -->
| Quantity | coarse upwind | fine upwind | |difference| | first from data | second from data | difference / larger distance |
|---|---|---|---|---|---|---|
| centreline_rms_all | 0.06646 | 0.09557 | 0.0291 | 0.0665 | 0.0956 | 0.305 |
| centreline_rms_free_jet | 0.0671 | 0.09575 | 0.0286 | 0.0671 | 0.0957 | 0.299 |
| centreline_rms_impingement | 0.01769 | 0.03166 | 0.014 | 0.0177 | 0.0317 | 0.441 |
| potential_core_w | 6.528 | 7.558 | 1.03 | 1.7 | 2.73 | 0.377 |
| decay_rate | 0.1181 | 0.1087 | 0.00942 | 0.0771 | 0.0865 | 0.109 |
| half_width_rms | 0.1538 | 0.07588 | 0.0779 | 0.154 | 0.0759 | 0.507 |
| spreading_rate | 0.1045 | 0.09582 | 0.00865 | 0.00526 | 0.00339 | 1.65 |
| k_axis_rms | 0.006999 | 0.009266 | 0.00227 | 0.007 | 0.00927 | 0.245 |
| stagnation_k | 0.03714 | 0.03801 | 0.000868 | 0.0113 | 0.0121 | 0.0715 |
| lateral_4.8_rms | 0.1323 | 0.1094 | 0.0228 | 0.132 | 0.109 | 0.173 |
| lateral_5.9_rms | 0.08219 | 0.06542 | 0.0168 | 0.0822 | 0.0654 | 0.204 |
| lateral_7.9_rms | 0.09886 | 0.09386 | 0.005 | 0.0989 | 0.0939 | 0.0505 |
| lateral_12.1_rms | 0.1068 | 0.1044 | 0.0024 | 0.107 | 0.104 | 0.0225 |
| lateral_14.9_rms | 0.1013 | 0.1005 | 0.000815 | 0.101 | 0.101 | 0.00805 |
| lateral_20.1_rms | 0.05412 | 0.05584 | 0.00172 | 0.0541 | 0.0558 | 0.0308 |
<!-- tables55 grids end -->

Upwind and UMIST on the fine grid differ by a quarter or less of the larger distance from the data
on every quantity but the spreading rate, where both are within 0.005 of the data and the ratio has
no meaning. The two grids differ by less than the larger distance from the data on every quantity
but the spreading rate (the same); the share reaches 0.51 on the half-width RMS and 0.44 on the
impingement-region RMS, and on the centreline and the half-width the coarse grid is the nearer to
the data, by 0.03 V0 RMS and 0.08 w_jet RMS. At the ring station the two grids' stagnation k differ
by 2%; at the centroid station 2 mm above the plate they differ by 47%, the fine grid's first row of
cells reading the rise toward the plate's k maximum that the coarse grid's does not resolve.

### 5.8 The harness row

The row `val020_272x90` (run id fab5e311, 2026-10-15, commit 6b8e89d, clean, `staggered-mgcg`, six
concurrent processes) stops at the cap after 20,000 outer iterations in 8,328 s and carries the whole
comparison of `validation.impinging_jet.evaluate`; its value, the centreline's RMS 0.06646, and its
components equal the coarse record's (section 5.3) to every printed digit, the two having gone
through the same code on the same case. Its `stagnation_k_ratio_isotropic` component is the
centroid-station ratio of section 5.6 (0.912); the ring-station component was added to the family
after the row was taken (commit D), so the stored row lacks it and a rerun would carry it.

### 5.9 Cost

<!-- tables55 timing begin -->
| Run | Cells | Outer | Wall (s) | Per outer (s) | Hours | Started | Stages (s) |
|---|---|---|---|---|---|---|---|
| standard 272x90 (coarse) upwind | 24,480 | 20,000 | 8181 | 0.409 | 2.27 | 2026-10-15T09:49:17 | momentum 626, pressure 6415, correct 21, turbulence 1088 |
| standard 408x135 (fine) upwind | 55,080 | 20,000 | 1.148e+04 | 0.574 | 3.19 | 2026-10-15T09:49:17 | momentum 6623, pressure 2042, correct 24, turbulence 2735 |
| standard 408x135 (fine) umist | 55,080 | 20,000 | 1.201e+04 | 0.6 | 3.34 | 2026-10-15T09:49:17 | momentum 1123, pressure 7625, correct 26, turbulence 3180 |
| rng 408x135 (fine) upwind | 55,080 | 20,000 | 1.197e+04 | 0.598 | 3.32 | 2026-10-15T09:49:17 | momentum 1054, pressure 8116, correct 23, turbulence 2722 |
| standard 272x90 (coarse) upwind, 100 w wide | 36,000 | 20,000 | 9362 | 0.468 | 2.6 | 2026-10-15T09:49:17 | momentum 803, pressure 6832, correct 18, turbulence 1670 |
<!-- tables55 timing end -->

Six processes together ran at 3.5 times the timing probes' seconds per outer iteration (0.41 to 0.60
s against 0.12 and 0.23), the pressure stage taking most of the coarse runs' time and the momentum
stage most of the fine upwind run's. The set took 3.3 hours of wall time.

## 6. Predictions against measurement

<!-- tables55 predictions begin -->
| Prediction | Claim | Verdict | Evidence |
|---|---|---|---|
| (a) | standard converges on both grids | FAILS | standard_272x90_upwind: max_simple_iter (bounded (neither)) at 20,000; standard_408x135_upwind: max_simple_iter (bounded (stalled)) at 20,000 |
| (b) | if RNG converges, its stagnation k is below standard's | NOT SCORABLE | RNG: max_simple_iter (bounded (neither)) at 20,000 |
| (c) | standard within 0.1 V0 of the data over the free-jet part (standard_272x90_upwind) | HOLDS | largest |difference| 0.100 over y/h <= 0.70 (39 stations); 0.062 over y/h <= 0.30 (16) |
| (c) | standard within 0.1 V0 of the data over the free-jet part (standard_408x135_upwind) | FAILS | largest |difference| 0.134 over y/h <= 0.70 (39 stations); 0.094 over y/h <= 0.30 (16) |
| (d) | standard overpredicts the stagnation k by more than 50% (standard_272x90_upwind) | FAILS | model 0.03714 vs estimate 0.02958 at the last ring station y/h 0.9731: ratio 1.256 (in-plane 1.883); at the last station y/h 0.9945 (a centroid) 0.912; first cell 0.01271, largest in the impingement region 0.03763 at y/h 0.980 |
| (d) | standard overpredicts the stagnation k by more than 50% (standard_408x135_upwind) | FAILS | model 0.03801 vs estimate 0.02958 at the last ring station y/h 0.9731: ratio 1.285 (in-plane 1.927); at the last station y/h 0.9945 (a centroid) 1.338; first cell 0.0119, largest in the impingement region 0.03894 at y/h 0.983 |
| (e) | the schemes differ by under 0.5 of the larger distance from the data on every quantity | FAILS | largest share 1.285; failing: spreading_rate 1.29 |
| (e) | the grids differ by under 1 of the larger distance from the data on every quantity | FAILS | largest share 1.645; failing: spreading_rate 1.65 |
| (f) | the 100 w domain moves the centreline by under 0.02 V0 and the half-width by under 0.05 w | FAILS | centreline 0.0309, half-width 0.07301478735694089 |
| (g) | potential core 6 to 9 w and the centreline above the data over the free-jet part (standard_272x90_upwind) | HOLDS | core 6.528475653648808, mean difference over the free-jet part +0.0640 |
| (g) | potential core 6 to 9 w and the centreline above the data over the free-jet part (standard_408x135_upwind) | HOLDS | core 7.558419778162976, mean difference over the free-jet part +0.0849 |
| (h) | K_y between 0.10 and 0.12 (standard_272x90_upwind) | HOLDS | K_y 0.10446805046768806 |
| (h) | K_y between 0.10 and 0.12 (standard_408x135_upwind) | FAILS | K_y 0.09581789142980354 |
| (i) | the plate's first-node y+ between 5 and 30 on the coarse grid | HOLDS | coarse median 26.08, at the axis 30.88, largest 31.23; fine median 17.15, largest 20.34; floor 11.53 |
| (j) | RNG does not converge | HOLDS | max_simple_iter (bounded (neither)) at 20,000 |
<!-- tables55 predictions end -->

- (a) fails: neither standard run stops by the rule; both are bounded at the cap in a cycle beside
  the outlets (section 5.1).
- (b) is not scorable as written, RNG not having converged. Its content, scored on the cap's
  fields: RNG's stagnation k at the last ring station (0.030 V0^2) is below standard's (0.038 on the
  same grid) and within 1% of the PIV estimate.
- (c) holds on the coarse grid (largest difference 0.100 V0 over y / h_jet <= 0.70, at the limit)
  and fails on the fine (0.134). On the prompt's narrower reading (the 30% nearest the nozzle) it
  holds on both (0.062 and 0.094).
- (d) fails: the standard model's k at the last ring station is 1.26 to 1.29 times the isotropic
  estimate, not 1.5; against the in-plane half-sum it is 1.9. The anomaly is there, at a quarter to
  a third rather than a half, with the caveat of section 2.2 on what the 2D PIV gives.
- (e) fails on the spreading rate alone, for the schemes and for the grids (both runs within 0.005
  of the data in each pair, the share undefined in substance); it holds on every other quantity.
- (f) fails: the wider domain moves the centreline by 0.031 V0 and the half-width by 0.073 w_jet,
  both towards the data.
- (g) holds on both grids: cores of 6.5 and 7.6 w_jet, the centreline above the data by 0.064 to
  0.085 V0 on average over the free-jet part.
- (h) holds on the coarse grid (K_y 0.104) and fails by 0.004 on the fine (0.096).
- (i) holds: y+ 26 on the coarse grid's plate, 17 on the fine, both above the floor.
- (j) holds: RNG does not converge within the cap, though it is the one run whose residual is still
  falling at the cap.

## 7. What this implies

Questions for Alex; no decision is taken here.

1. *The variant.* RNG did not converge within the cap on this case either, so by the outcome the
   prompt wrote the product stays standard and VAL-018's variant clause closes as "RNG not runnable
   in this solver on four cases". Two facts cut the other way and should be weighed: RNG's residual
   falls monotonically to 2.7e-6 at the cap, four orders below the standard runs' cycle, with (b)
   and (c) met, so a longer cap might have stopped it by the rule; and on the cap's fields RNG's k at
   the stagnation point is within 1% of the PIV estimate where standard's is 26 to 35% above it,
   although RNG's centreline is the worse of the two by a wide margin (a 10 w_jet core, 0.21 V0 off
   at worst). Does the variant decision stand on the cap as run, or does RNG get a longer cap on
   this case before it is closed?
2. *The allowance.* The standard model's centreline is within 0.10 V0 (coarse) and 0.13 V0 (fine)
   of the data everywhere, within 0.067 and 0.096 RMS; its spreading rate is within 0.005 of the
   data's; its stagnation k is 1.3 times the estimate. Which of these carries the allowance, on which
   grid, and at the cap or only on a converged field?
3. *The stop.* Every run is bounded by a cycle beside the fixed-flow outlets, 3 to 4 w_jet from a
   side wall, with the jet still. The same corner cycle has now appeared on three cases. Is the cap
   the stop for this case (the record then says "at the cap" as here), or does the cycle at the
   outlets get its own treatment (ADR-012 E's rule, or the outlet's corner cells)?
4. *The domain.* The 68 w_jet width moves the centreline by 0.03 V0 against a 100 w_jet width, and the
   confinement's co-flow lifts the lateral tails. Should the case be widened (36,000 cells for 100
   w_jet on the coarse spacing, 2.6 hours at the cap beside other runs), or the outlets placed
   elsewhere (the top wall, as a channel open above would have them)?
5. *The equipment-top caveat.* Prediction (d) holds at a quarter to a third: the standard model
   overstates the turbulence at the stagnation point by 26 to 35% against the isotropic estimate
   and by about 90% against the in-plane half-sum, and understates it along the free jet by a
   third. The comparative claim of VAL-018 carries that caveat either way; the size depends on what
   the third component is taken to be.
6. *The data.* The lateral profiles at five of the six heights rest on the paper's Gaussian fit,
   not on digitised measurements (section 2.2); the near-nozzle profile at y = 2 w_jet (figure 3)
   was not digitised either. If the lateral comparison is to carry an allowance, the open-marker
   curves need another reading (a request to the authors for the PIV fields would settle it).

## 8. Figures

- `ecr002_val020_standard_272x90_upwind.png`, `ecr002_val020_standard_408x135_upwind.png`,
  `ecr002_val020_standard_408x135_umist.png`, `ecr002_val020_rng_408x135_upwind.png`,
  `ecr002_val020_standard_272x90_upwind_w100.png`: the marked figure of each run (streamlines over
  the whole domain and over the jet, nu_t / nu, k / V0^2, the readings in the titles).
- `ecr002_val020_centreline.png`: V_0,y / V0 and k / V0^2 on the axis, every run against the data.
- `ecr002_val020_halfwidth.png`: delta_0.5 / w_jet against the dots.
- `ecr002_val020_lateral.png`: the self-similar profiles at the six heights, the digitised circles
  at 4.8 and the Gaussian fit at every height.
