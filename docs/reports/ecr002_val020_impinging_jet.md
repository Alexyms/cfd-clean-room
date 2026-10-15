# ECR-002 Step 7: VAL-020, the Plane Impinging Jet

**Date:** 2026-10-15
**Tree:** branch `feature/ecr002-val020-impinging-jet` from main at 8acb475 (prompt 54 merged as PR 81;
multigrid is the default pressure preconditioner). Commit A adds the case
(`configs/validation_impinging_jet.yaml`), its loader (`validation.cases.load_impinging_jet`, the presets
`val020_272x90` and `val020_408x135`), the digitised data with its provenance and the comparison code
(`validation/impinging_jet.py`), the harness family (`scripts/benchmark.py`, reported and not scored) and
their tests (`tests/test_impinging_jet.py`). Nothing under `src/` changes. Every run goes through the
committed `StaggeredSolver` on the case as the suite loads it.
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
