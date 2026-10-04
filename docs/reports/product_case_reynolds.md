# The Product Room's Reynolds Number and the Laminar Solver

**Date:** 2026-10-04
**Tree:** main at 02ec787, unchanged; the probes import `src/` and change nothing.
**Instruments:** `results/builder33/` (untracked). `probe33.py`, `cost33.py`, `annex20_flux.py`,
`digitize_fig.py`, `summarize.py` and `calc33.py` (the arithmetic ADR-012 quotes) are carried
verbatim in the appendices, so this report stands without them.
**Measured values:** every figure in sections 2, 4, 5 and 6 is from the builder's runs on
2026-10-04. Where the orchestrator measured the same quantity first, its figure is beside mine
and named as its.

## 1. Summary

The product room runs at a Reynolds number of 89,500 on its height and a cell Reynolds number of
1,190 on its 0.04 m mesh. The flow solver was validated at 5 and 100. As committed, the solve
diverges by outer iteration 58. On a 40x15 copy of the room with the pressure solved toward 1e-8,
the real viscosity diverges, as does ten times it; a hundred times stalls; a thousand times
converges. Removing the four obstacles does not stop the divergence. With heavier
under-relaxation (alpha_velocity 0.2) the runs have not diverged, nor converged: the residual of
the 40x15 run falls to 4.8e-6 by outer 2,809 and then rises again, never reaching the stopping
tolerance in 3,000 iterations (section 4, row 7), and the 200x75 run plateaus near 6e-4 over
300.

The orchestrator's rows reproduce to every digit where the settings are the same. One premise
does not: in the real-air 40x15 run the pressure solve reached its 40,000-sweep cap in 30 of the
first 121 iterations, first at outer 3, not only at outer 120. The Re 8,950 control, whose
pressure corrections all reached their tolerance until outer 220 while its divergence began near
200, carries the claim that the cap is not the cause.

One pressure correction on the product mesh needs 27,408 sweeps to the committed tolerance and
112,519 to 1e-8, against the committed cap of 200; the count quadruples as the cells per side
double.

Item 0, the Annex 20 benchmark: the specification and the measured profiles were obtained from
the primary source. The measured symmetry-plane profile at x/H = 1.0 carries 1.02 to 1.32 of the
inlet flux, depending on how the readings are closed to the walls. At x/H = 2.0 it carries 0.60
to 0.64 whatever the closure, while the plane z/W = 0.4 at the same section carries 1.11 to 1.13.
The model room was three-dimensional at x/H = 2.0 by about half the inlet flux between those two
planes, which is what a two-dimensional comparison there cannot match (section 6).

## 2. The Reynolds numbers

| Case | rho (kg/m^3) | U (m/s) | Length (m) | mu (Pa s) | Re = rho U L / mu | Cell (m) | Cell Re |
|---|---|---|---|---|---|---|---|
| Product room, height | 1.2 | 0.45 | 3.0 | 1.81e-5 | 89,503 | 0.04 (200x75) | 1,193 |
| Product room, 40x15 probe grid | 1.2 | 0.45 | 3.0 | 1.81e-5 | 89,503 | 0.2 | 5,967 |
| VAL-001 channel, height | 1.0 | 0.1 | 0.5 | 0.01 | 5 | 0.0125 (80x40) | 0.125 |
| VAL-002 cavity | 1.0 | 1.0 | 1.0 | 0.01 | 100 | 0.025 (40x40) | 2.5 |
| Annex 20 room, slot height | (nu 15.3e-6 m^2/s) | 0.455 | 0.168 | | 4,996 | | |
| Annex 20 room, height | (nu 15.3e-6 m^2/s) | 0.455 | 3.0 | | 89,216 | | |

The product room's arithmetic: `1.2 x 0.45 x 3.0 / 1.81e-5 = 1.62 / 1.81e-5 = 89,503`; with the
cell width in place of the height, `1.2 x 0.45 x 0.04 / 1.81e-5 = 1,193`. The supply speed is
the inlet's 0.45 m/s, the room's largest prescribed velocity. The Annex 20 room's Reynolds number
on its height is the product room's within 0.4%.

## 3. What "laminar" means here

ADR-004, as SYSTEM.md section 6 records it: "Clean rooms are engineered for laminar flow. Re well
below transition. Physically correct, not a shortcut." In clean-room practice "laminar flow"
names a supply that moves in one direction at low turbulence intensity, which the room's
ceiling-wide HEPA supply is. In the Navier-Stokes sense the flow is set by the Reynolds number,
and at 89,500 a room with shear layers at the supply's edges, wakes behind four obstacles and a
supply striking their tops is not below transition. ADR-004 used the first meaning to claim the
second.

## 4. The probes

**Method.** `probe33.py` (appendix A) loads `configs/clean_room_default.yaml`, overrides the
grid, multiplies the viscosity, optionally removes the obstacles, sets the outer cap, the
velocity under-relaxation and the pressure sweep cap and tolerance, and runs the staggered solver
under the committed `velocity_step` rule from rest. Each outer iteration it records the solver's
own residual (the largest change of the cell-centred velocity over the reference velocity
`F / (rho h)`, which is 79.2 m/s on 200x75 and 16.2 m/s on 40x15), the pressure sweeps, the
largest cell-centred speed, and the worst per-cell and signed domain-sum mass imbalance of the
corrected faces. It wraps the corrector to keep those faces, a probe's use of a private
attribute. Outer iterations are counted from zero. The obstacle-free and Reynolds controls, row 7's
length and row 8 are the builder's additions. On 40x15 the converging Re 90 run's largest speed
settles near 3.8 m/s, the coarse grid's outlet jets, so a speed below about 5 m/s there is not a
sign of divergence; growth to tens and hundreds of m/s is.

| # | Run | Builder's result | Orchestrator's row |
|---|---|---|---|
| 1 | 200x75 as committed: alpha_u 0.7, pressure cap 200, tol 1e-6; 500 outers | Diverges. Largest speed 9.24 m/s at outer 58 (9.19 at 59), signed sum -1.095 kg/s at 59 of a 3.80 kg/s through-flow; 103 m/s at 98; 2.4e11 m/s at 499. Pressure at its cap every iteration. | "diverges: max speed 9.0 m/s at outer 59, signed mass sum -1.1 kg/s" |
| 2 | 200x75, cap 5,000, tol 1e-8, alpha_u 0.5; 150 outers | Grows. 2.54 m/s at outer 69; residual least at outer 14 (2.2e-3), rising to 2.7e-2 at 149; 7.8 m/s at 149. Pressure at its cap every iteration. | "grows: 2.47 m/s at outer 69, residual rising after 25; pressure at its cap every iteration" |
| 3 | 40x15, real air, cap 40,000, tol 1e-8, alpha_u 0.5; 250 outers | Diverges. 26.14 m/s and residual 5.22e-1 at outer 120; 293 m/s at 249. Pressure at its cap in 30 of the first 121 iterations, first at outer 3. | "diverges: 26.1 m/s at outer 120, residual 5.2e-1"; cap "reached only at outer 120" |
| 4 | 40x15, viscosity x1000 (Re 90), as row 3; 1,000 outers | Converging. Residual 5.27e-2 at 0, 1.83e-4 at 249, 2.65e-5 at 999, falling throughout; pressure never at its cap; largest speed 3.8 m/s. | "converges: residual 5.3e-2 to 1.8e-4 by outer 250" |
| 5 | 40x15, real air, alpha_u 0.2, as row 3; 500 outers | Not diverged. 2.25 m/s and residual 5.23e-3 at outer 75; residual 1.3e-4 at 499, the last 100 between 1.2e-4 and 6.5e-4; largest speed 3.3 m/s. | "not diverged at outer 75 (2.25 m/s, residual 5.2e-3); too short to call" |
| 6 | 40x15, viscosity x10 (Re 8,950), as row 3; 500 outers | Diverges. Speed past 5 m/s at outer 202, 102 m/s at 295, 2.2e6 m/s at 499. Pressure first at its cap at outer 220. | (builder's control) |
| 7 | 40x15, real air, alpha_u 0.2; 3,000 outers (row 5 continued) | Stalls without diverging. Residual 3.6e-5 at outer 1,000, least 4.8e-6 at 2,809, then rising: between 7.3e-6 and 1.5e-5 over the last 100; never below the 1e-6 tolerance; largest speed 3.87 m/s, the Re 90 run's level; pressure at its cap in 19 of 3,000. | (builder's control) |
| 8 | 200x75, real air, alpha_u 0.2, cap 5,000, tol 1e-8; 300 outers | Not diverged. Residual between 4.4e-4 and 6.1e-4 over the last 100; largest speed 2.3 m/s; signed sum -7.2e-2 kg/s, since a 5,000-sweep correction is not converged on this grid (section 5). | (builder's control) |
| 9 | 40x15, viscosity x100 (Re 895), as row 3; 1,000 outers | Stalls. Residual least 6.9e-4 at outer 589, then between 1.4e-3 and 4.9e-3 over the last 100; largest speed 3.7 m/s; pressure never at its cap. | (builder's control) |
| 10 | 40x15, real air, obstacles removed, as row 3; 250 outers | Diverges. 20.7 m/s at outer 115, 147 m/s at 249. | (builder's control) |

The rows that share the orchestrator's settings reproduce its figures exactly, as a deterministic
solver should (row 3: 26.14 m/s and 5.22e-1 at outer 120; row 5: 2.25 m/s and 5.23e-3 at outer
75; row 4's residuals). The speeds in rows 1 and 2 differ in the second figure because the
orchestrator's first probe printed the largest |u| and |v| separately, where `probe33.py` takes
the largest speed; the signed sum in row 1 agrees. The one premise that does not reproduce
is row 3's: its pressure solve did not converge at every iteration. Row 6 is the control that
replaces it: at Re 8,950 every correction reached 1e-8 within its cap until outer 220, and the
speed was already past 5 m/s at 202 and rising.

**What the probes show.** At the room's Reynolds number the solver, as committed and with the
pressure solved further, does not converge; the same geometry, boundaries, outlets and initial
field converge at Re 90, stall at 895 and diverge at 8,950; the divergence needs neither an
unconverged pressure solve (row 6) nor the obstacles (row 10). Heavier under-relaxation keeps the
iterate bounded on both grids over the lengths run.

**What they do not show.** That no steady laminar solution exists: rows 5, 7 and 8 have not
diverged, and under-relaxation changes the path to a fixed point, not the fixed point. Row 7
came within a factor of five of the stopping tolerance before its residual turned upward, as row
9's did at Re 895; whether it would then converge, settle into an oscillation or grow beyond
3,000 iterations is not known. Had it converged, it would be a fixed point of the laminar
discrete equations at a cell Reynolds number of 5,967. Nor do they show what the field would
mean if found; that argument is ECR-002 section 2, not a measurement.

**Controls the probes lack.** (a) The pressure outlets: each extrapolates its face velocity from
the interior and corrects it against p' = 0, and reversed flow through an outlet is a known
destabilizer at high Reynolds numbers; no run isolates the outlets or the hood exhaust on the
right wall. (b) The initial field: every run starts from rest with the supply switched on; none
continues from the Re 90 field toward higher Reynolds numbers. (c) A case with a known steady
answer at a cell Reynolds number of tens to hundreds, such as the lid-driven cavity at Re 1,000,
which would separate what the solver can do from what the room does. (d) Grid refinement at a
fixed Reynolds number: real air ran on 40x15 and 200x75 only. (e) A time-accurate solve: an
unsteady flow appears to a steady iteration as one that does not converge, and the solver has no
time-accurate mode.

## 5. The pressure solve's cost

`cost33.py` (appendix B) runs the first outer iteration only, with the sweep cap at 5,000,000,
and records the sweeps the correction took to its tolerance. The product cases keep the
committed alpha_velocity 0.7; the Annex 20 case is that room as section 6 specifies it, on a
uniform mesh.

| Grid | Tolerance (Pa) | Sweeps | Seconds | ms per sweep |
|---|---|---|---|---|
| product 40x15 | 1e-8 | 2,553 | 0.13 | 0.05 |
| product 100x38 | 1e-8 | 28,456 | 3.0 | 0.11 |
| product 200x75 | 1e-8 | 112,519 | 32.0 | 0.28 |
| product 200x75 | 1e-6 (committed) | 27,408 | 7.7 | 0.28 |
| Annex 20, 90x30 | 1e-8 | 64,113 | 6.4 | 0.10 |
| Annex 20, 180x60 | 1e-8 | 211,592 | 45.2 | 0.21 |

From 100x38 to 200x75 the count rises 3.95 times and from 90x30 to 180x60 3.30 times, ADR-010's
N^2 growth. The committed cap of 200 is 137 times short of one correction at the committed
tolerance, which is why row 1's continuity is lost from the first iteration (signed sum -3.80
kg/s at outer 0, the whole through-flow). In row 4 the correction's count fell from 9,748 to
4,048 over 1,000 iterations. ADR-012 section H extrapolates these to a converged product solve.

## 6. Item 0: the Annex 20 benchmark

### 6.1 Sources obtained

| Source | Location | SHA-256 |
|---|---|---|
| Nielsen, P. V. (1990), "Specification of a Two-Dimensional Test Case", IEA Annex 20 Research Item 1.45, Aalborg University R9040, the portal's publisher's PDF (23 pages with the portal's cover) | https://vbn.aau.dk/ws/files/197503356/Specification_of_a_TwoDimensional_Test_Case.pdf | 9d963abeded02bffd0ead10ae3104209899857e8d2514007771f5c32837c84b5 |
| The same report, the benchmark page's scan ("Benchmark test report") | https://www.aaudxp-cms.aau.dk/media/v0xnnqtw/592301_2d_benchmark_test.pdf | 40b4af4ad2f59051ee22cbc161c054125a45156078d5989051ddea9866652138 |
| "Laser-doppler measurements of the isothermal two dimensional test case", Excel workbook, the benchmark page's "Benchmark test measurements" | https://www.aaudxp-cms.aau.dk/media/udffy5eg/592302_2d_-benchmark_test_measurements_new.xls | e6dc80485371e3d076cf0677202958df07d07e9207b787053ea68313bc745a8e |
| Nielsen, Rong and Olmedo (2010), Clima 2010 | https://homes.civil.aau.dk/pvn/cfd-benchmarks/two_d_literature/2005_2010/Nielsen_PV_Rong_L_Olmedo_I_2010.pdf | a9e9b7346361d979441fd305817652b507bde18c4c1757f0ec6536140e18b7c1 |
| Zhai, Zhang, Zhang and Chen (2007), HVAC&R Research 13(6), Part 1 | https://engineering.purdue.edu/~yanchen/paper/2007-8.pdf | d480b5252191b8c3fc59030cbf0cafb9d4a1debdd7b096b4c251dd8229c8fc51 |

The benchmark page is https://www.en.build.aau.dk/web/cfd-benchmarks/two-dimensional-benchmark-test;
the older index at homes.civil.aau.dk/pvn/cfd-benchmarks/ returned 403, and the 2010 paper's
address, www.cfd-benchmarks.com, was not tried. Downloaded 2026-10-04.

**The data file is not saved under `validation/data/`.** The workbook carries no licence
statement. Its document properties name its author as Arnkell J. Petersen, created 2005-11-24,
last saved by Peter V. Nielsen 2006-08-10, and its first rows say the values "are taken from the
'Specifcation of the two dimensional test case' by Peter V. Nielsen, november 1990": a
digitization of the report's figure 5. The report's portal terms allow downloading and printing
one copy for private study and forbid further distribution. Neither permits redistribution in a
public repository, so the file stays where it is, identified by the address and checksum above.

### 6.2 The specification

Page numbers are the report's own (R9040); the portal PDF puts report page n at file page n + 5.

| Quantity | Value | Source |
|---|---|---|
| Geometry ratios | L/H = 3.0, h/H = 0.056, t/H = 0.16 | R9040 p. 2, eq. (1) |
| Dimensions | H = 3.0 m, L = 9.0 m, h = 0.168 m, t = 0.48 m | p. 2, eq. (2) |
| Inlet | a slot of height h along the side wall directly under the ceiling, across the full width | p. 1, figure 1; p. 2 |
| Outlet | a slot of height t at the foot of the opposite wall, across the full width | p. 1, figure 1 |
| Reynolds number | Re = h u0 / nu = 5000 | p. 2, eq. (3) |
| Inlet velocity | u0 = 0.455 m/s at nu = 15.3e-6 m^2/s, 20 C | p. 2, eq. (4) |
| Inlet turbulence | k0 = 1.5 (0.04 u0)^2, eps0 = k0^1.5 / l0, l0 = h / 10, "a turbulence intensity of 4%"; varying l0 "within a reasonable level shows only a very small influence" | p. 2, eqs. (5) to (7) |
| Comparison lines | x = H and x = 2H; y = h/2 and y = H - h/2 | p. 3, eqs. (8) and (9) |
| Coordinates | origin at the top of the inlet wall, x along the room, y downward from the ceiling | p. 1, figure 1 |
| Measurements | laser-Doppler, Restivo (1979), in a model with W/H = 1.0 and H = 89.3 mm; mean and rms u at z/W = 0 (figure 5) and z/W = 0.4 (figure 6) | p. 5; pp. 8 and 9 |
| Flow regime | "fully turbulent for a Reynolds number of 5000" (figure 3) | p. 5 |
| Three-dimensionality | "Comparisons between mean velocity in figures 5 and 6 show that the flow is slightly three-dimensional" | p. 5 |

The 2010 paper repeats the ratios, the Reynolds number, the inlet turbulence and the W/H = 1.0
measurement model (its introduction and eq. (1)). The inlet values in SI:
`k0 = 1.5 (0.04 x 0.455)^2 = 4.969e-4 m^2/s^2`, `l0 = 0.0168 m`, `eps0 = k0^1.5 / l0 = 6.59e-4
m^2/s^3`, the specification's form without the factor C_mu^(3/4) some codes use (which would give
1.08e-4); the eddy viscosity they imply is 2.2 times the molecular one.

### 6.3 The flux check

In a steady two-dimensional room the net volume flux through any vertical section between the
inlet and the outlet equals the inlet's, u0 h. In the specification's units the integral of
u / u0 over y / H from 0 to 1 must equal h / H = 0.056. The workbook gives 25 readings at each
section, the outermost 0.0101 to 0.0236 H from the ceiling and 0.0134 to 0.0256 H from the floor,
so the strips between the outermost readings and the walls are closed three ways: u falling
linearly to zero at the wall (no slip), u held at the outermost reading, and the measured span
alone. `annex20_flux.py` (appendix C):

| Section | Closure | Net / (u0 h) | Forward / (u0 h) | Reverse / (u0 h) |
|---|---|---|---|---|
| x/H = 1.0 | no slip | 1.17 | 2.07 | -0.90 |
| x/H = 1.0 | held | 1.32 | 2.23 | -0.92 |
| x/H = 1.0 | span only | 1.02 | 1.91 | -0.89 |
| x/H = 2.0 | no slip | 0.62 | 2.46 | -1.84 |
| x/H = 2.0 | held | 0.60 | 2.52 | -1.91 |
| x/H = 2.0 | span only | 0.64 | 2.41 | -1.77 |

The net flux is the difference of a forward and a return flux each two to four times larger, so
a uniform error of 0.01 u0 in the readings moves the ratio by 0.18; at x/H = 1.0 most of the
spread between closures is the strip under the ceiling, where the wall jet's peak sits 0.04 H
from the wall. At x/H = 2.0 the closures agree and the profile carries six tenths of the inlet
flux.

**Is the workbook the figure?** `digitize_fig.py` (appendix D) digitizes figures 5 and 6 from the
portal PDF at 300 dpi: circles found by a Hough transform, triangles (the rms readings) rejected
by matching templates cut from the page, the axes calibrated from the frame lines and every
labelled tick. At x/H = 2.0 it finds all 25 of the workbook's points and agrees with them within
0.008 u0 in u (mean difference -0.001 u0) and 0.003 H in y, and gives a net flux of 0.64 (no
slip). At x/H = 1.0 it finds 21 of 25, missing four where circles overlap, and gives 1.21. The
workbook is a faithful reading of figure 5.

**The second plane.** The same digitization of figure 6, the plane z/W = 0.4, a tenth of the
width from the side wall:

| Section | z/W = 0 (workbook) | z/W = 0 (digitized) | z/W = 0.4 (digitized) |
|---|---|---|---|
| x/H = 1.0, no slip | 1.17 | 1.21 | 1.11 |
| x/H = 1.0, held | 1.32 | 1.35 | 1.25 |
| x/H = 2.0, no slip | 0.62 | 0.64 | 1.12 |
| x/H = 2.0, held | 0.60 | 0.61 | 1.13 |

At x/H = 2.0 the two planes differ by half the inlet flux. In a three-dimensional room only the
integral over the width must carry u0 h W; the symmetry plane's shortfall and the outer plane's
excess are air moving across the width between the sections, which the specification calls
"slightly three-dimensional". The 2010 paper's three-dimensional k-omega prediction has a large
difference between the middle plane and a plane 0.1 W from the side wall, and the paper notes
that the measurements "do not show this large difference" (the text above its figure 10). That
is consistent with the flux check: point by point the two planes' profiles differ by a few
hundredths of u0, which the eye does not call large, and those hundredths integrate to half the
inlet flux because the net flux is the small difference of a forward and a return flux each two
to four times larger. The prediction was that x/H = 1.0
would come within about 10%, with three-dimensionality the likely cause of any departure. It
came within 2% to 32% depending on the closure, 17% with no slip; the larger departure is at
x/H = 2.0, and three-dimensionality accounts for it, measured rather than assumed.

### 6.4 What it implies for a two-dimensional comparison

A two-dimensional solution carries u0 h through every section, so at x/H = 2.0 it cannot match
the symmetry plane closer than the shortfall, 0.38 x 0.056 = 0.021 in the integral of u / u0 over
y / H: 0.021 u0 if spread over the height, about 0.04 u0 if it sits in the return flow below the
zero crossing at y / H = 0.46. That is a floor on any two-dimensional model's agreement there,
whatever its turbulence model, and it is consistent with the 2010 paper's remark that a
two-dimensional low-Reynolds k-epsilon prediction leaves the counter flow "slightly
underestimated" (under its figure 5). At x/H = 1.0 the two planes agree within 0.10 of the inlet
flux and both sit near or above it, so a comparison there is limited by the closures, not by the
room's three-dimensionality. The data therefore support a two-dimensional comparison at x/H =
1.0 and along the jet under the ceiling, and not a scored one at x/H = 2.0 against the symmetry
plane alone. The criterion is ADR-012's decision 5 and stays OPEN.

## 7. Reproducing

From `results/builder33/`, with the repository's `.venv` for the probes and a separate
environment holding `xlrd`, `pymupdf` and `opencv-python-headless` for item 0 (none of them
project dependencies):

    python probe33.py p1_committed 200 75 1 500 0 0 0
    python probe33.py p2_cap5000 200 75 1 150 0.5 5000 1e-8
    python probe33.py p3_real40 40 15 1 250 0.5 40000 1e-8
    python probe33.py p4_mu1000 40 15 1000 1000 0.5 40000 1e-8
    python probe33.py p5_av02 40 15 1 500 0.2 40000 1e-8
    python probe33.py c2_mu10 40 15 10 500 0.5 40000 1e-8
    python probe33.py p5b_av02_long 40 15 1 3000 0.2 40000 1e-8
    python probe33.py p6_av02_200 200 75 1 300 0.2 5000 1e-8
    python probe33.py c1_mu100 40 15 100 1000 0.5 40000 1e-8
    python probe33.py c3_noobs 40 15 1 250 0.5 40000 1e-8 noobs
    python cost33.py k_prod200_e8 product 200 75 1e-8      (and the other five rows of section 5)
    python annex20_flux.py sources/annex20_2d_measurements.xls
    python digitize_fig.py sources/nielsen1990_spec.pdf sources/annex20_2d_measurements.xls

Rows 1 to 10 of section 4 are, in order, p1, p2, p3, p4, p5, c2, p5b, p6, c1 and c3. Each run
writes its full per-iteration history to `NAME.json`; `summarize.py` prints the figures quoted
above (appendix E). The runs took 36 s (p1) to 39 min (p5b), eight of them in parallel on one
machine.

## Appendix A: probe33.py

```python
"""Builder probe, prompt 33: does the steady solver settle on the product room?

Usage: python probe33.py NAME NX NY MU_FACTOR N_OUTER ALPHA_U MAX_P_ITER P_TOL [noobs]

Loads configs/clean_room_default.yaml, overrides the grid, multiplies the
viscosity by MU_FACTOR (and with ``noobs`` removes the four obstacles), and
runs N_OUTER outer iterations of the staggered
solver under the velocity_step rule. A value of 0 for ALPHA_U, MAX_P_ITER or
P_TOL keeps the committed value. Every outer iteration records the residual
(the solver's own), the pressure sweeps, the largest cell-centred speed, the
worst per-cell mass imbalance and the signed domain sum of the corrected
faces. The run stops early only if a field goes non-finite. Writes NAME.json
beside this file.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.boundary_staggered import StaggeredBoundary  # noqa: E402
from src.config import SimConfig  # noqa: E402
from src.mesh import Mesh  # noqa: E402
from src.solver_staggered import StaggeredSolver  # noqa: E402


class NonFinite(Exception):
    """Raised from the callback when the field stops being finite."""


def main() -> None:
    name = sys.argv[1]
    nx, ny = int(sys.argv[2]), int(sys.argv[3])
    mu_factor, n_outer = float(sys.argv[4]), int(sys.argv[5])
    alpha_u, max_p, p_tol = float(sys.argv[6]), int(sys.argv[7]), float(sys.argv[8])

    raw = yaml.safe_load((ROOT / "configs/clean_room_default.yaml").read_text())
    raw["domain"]["nx"], raw["domain"]["ny"] = nx, ny
    raw["fluid"]["viscosity"] *= mu_factor
    if len(sys.argv) > 9 and sys.argv[9] == "noobs":
        raw["obstacles"] = []
    solver_block = raw["solver"]
    solver_block["max_simple_iter"] = n_outer
    if alpha_u:
        solver_block["alpha_velocity"] = alpha_u
    if max_p:
        solver_block["max_pressure_iter"] = max_p
    if p_tol:
        solver_block["pressure_tol"] = p_tol
    cfg = SimConfig.from_dict(raw)
    mesh = Mesh(cfg)
    boundary = StaggeredBoundary(mesh, cfg)
    solver = StaggeredSolver(mesh, cfg, boundary)

    # Keep the corrected faces of every outer iteration so the imbalance can be
    # read from them; the callback itself sees only the cell means.
    corrector = solver._corrector
    original = corrector.correct
    latest: dict[str, np.ndarray] = {}

    def correct(prediction, p):  # type: ignore[no-untyped-def]
        result = original(prediction, p)
        latest["u"], latest["v"] = result.u, result.v
        return result

    corrector.correct = correct  # type: ignore[method-assign]

    supply = 0.45
    height = cfg.room_height
    reynolds = cfg.rho * supply * height / cfg.mu
    cell_re = cfg.rho * supply * (height / ny) / cfg.mu
    rec: dict[str, list[float]] = {
        "residual": [], "sweeps": [], "max_speed": [], "worst": [], "signed": []
    }
    t0 = time.perf_counter()

    def callback(state) -> None:  # type: ignore[no-untyped-def]
        speed = float(np.max(np.hypot(state.u, state.v)))
        imbalance = corrector.mass_imbalance(latest["u"], latest["v"])
        rec["residual"].append(float(state.residual))
        rec["sweeps"].append(int(state.pressure_sweeps))
        rec["max_speed"].append(speed)
        rec["worst"].append(float(np.max(np.abs(imbalance))))
        rec["signed"].append(float(np.sum(imbalance)))
        it = state.iteration
        if it % 25 == 0 or it == n_outer - 1:
            print(
                f"{name} it {it:5d} res {state.residual:.3e} sweeps "
                f"{state.pressure_sweeps:6d} max|U| {speed:.4g} worst "
                f"{rec['worst'][-1]:.3e} signed {rec['signed'][-1]:.3e} "
                f"t {time.perf_counter() - t0:7.1f}s",
                flush=True,
            )
        if not math.isfinite(speed):
            raise NonFinite

    stop = None
    try:
        solver.solve_steady(on_iteration=callback)
        stop = solver.stop_reason
    except NonFinite:
        stop = "non_finite"
    out = {
        "name": name,
        "nx": nx,
        "ny": ny,
        "mu": cfg.mu,
        "rho": cfg.rho,
        "reynolds": reynolds,
        "cell_reynolds": cell_re,
        "alpha_velocity": cfg.alpha_velocity,
        "max_pressure_iter": cfg.max_pressure_iter,
        "pressure_tol": cfg.pressure_tol,
        "reference_velocity": solver.reference_velocity,
        "through_flow": cfg.rho * boundary.get_total_inlet_flux(),
        "stop": stop,
        "seconds": time.perf_counter() - t0,
        **rec,
    }
    (Path(__file__).parent / f"{name}.json").write_text(json.dumps(out))
    print(f"{name}: Re {reynolds:.0f}, cell Re {cell_re:.1f}, stop {stop}", flush=True)


if __name__ == "__main__":
    main()
```

## Appendix B: cost33.py

```python
"""Builder probe, prompt 33, section H: sweeps of one pressure correction.

Usage: python cost33.py NAME CASE NX NY P_TOL [STRETCH_Y]

CASE is "product" (configs/clean_room_default.yaml, regridded) or "annex20"
(the IEA Annex 20 room, Nielsen 1990: L 9.0 m, H 3.0 m, slot h 0.168 m under
the ceiling on the left wall at u0 0.455 m/s, outlet t 0.48 m at the foot of
the right wall, no obstacles, air). Runs the first outer iteration only, with
the sweep cap at 5,000,000, and records the sweeps that correction took to
reach P_TOL and its wall time. STRETCH_Y, if given, clusters y toward both
walls to that wall-cell width (m). Writes NAME.json beside this file.
"""

import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.boundary_staggered import StaggeredBoundary  # noqa: E402
from src.config import SimConfig  # noqa: E402
from src.mesh import Mesh  # noqa: E402
from src.solver_staggered import StaggeredSolver  # noqa: E402


def annex20(raw: dict) -> dict:
    """The Annex 20 2D room on the product file's particle and solver blocks."""
    raw["domain"].update(width=9.0, height=3.0)
    raw["fluid"].update(density=1.2, viscosity=1.2 * 15.3e-6)
    raw["boundaries"] = {
        "slot": {
            "type": "velocity_inlet", "location": "left",
            "y_start": 3.0 - 0.168, "y_end": 3.0, "velocity": 0.455,
        },
        "outlet": {
            "type": "pressure_outlet", "location": "right",
            "y_start": 0.0, "y_end": 0.48,
        },
    }
    raw["obstacles"] = []
    return raw


def main() -> None:
    name, case = sys.argv[1], sys.argv[2]
    nx, ny, p_tol = int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
    raw = yaml.safe_load((ROOT / "configs/clean_room_default.yaml").read_text())
    if case == "annex20":
        raw = annex20(raw)
    raw["domain"]["nx"], raw["domain"]["ny"] = nx, ny
    if len(sys.argv) > 6:
        raw["mesh"]["y"] = {"min_wall_spacing": float(sys.argv[6])}
    raw["solver"].update(max_simple_iter=1, max_pressure_iter=5_000_000, pressure_tol=p_tol)
    cfg = SimConfig.from_dict(raw)
    mesh = Mesh(cfg)
    solver = StaggeredSolver(mesh, cfg, StaggeredBoundary(mesh, cfg))
    t0 = time.perf_counter()
    solver.solve_steady()
    seconds = time.perf_counter() - t0
    out = {
        "name": name, "case": case, "nx": nx, "ny": ny, "pressure_tol": p_tol,
        "sweeps": solver.last_pressure_sweeps, "seconds": seconds,
        "pressure_seconds": solver.stage_seconds["pressure"],
        "min_dy": float(mesh.dy_cell.min()), "stretch_ratio_y": float(mesh.stretch_ratio_y),
    }
    (Path(__file__).parent / f"{name}.json").write_text(json.dumps(out))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
```

## Appendix C: annex20_flux.py

```python
"""Builder check, prompt 33, item 0: the Annex 20 profiles against the inlet flux.

Reads the Aalborg measurement workbook (symmetry plane, Re 5000, digitized in
2005 from Nielsen 1990, figure 5) and integrates u/u0 over the height at
x/H = 1.0 and 2.0. y is measured down from the ceiling, as in the
specification. In a two-dimensional steady room the net volume flux through
any vertical section between the inlet and the outlet is u0 h, so the
integral of u/u0 over y/H from 0 to 1 should be h/H = 0.056.

The profiles stop short of the two walls, so the end strips are closed three
ways: u falling linearly to zero at the wall (no slip), u held at the
outermost measured value up to the wall, and the measured span alone. Needs
xlrd (not a project dependency). Usage: python annex20_flux.py WORKBOOK
"""

import json
import sys

import numpy as np
import xlrd

H_OVER_H = 0.056


def column(sheet: xlrd.sheet.Sheet, col: int, rows: range) -> np.ndarray:
    """One numeric column over the given rows."""
    return np.array([float(sheet.cell_value(r, col)) for r in rows])


def integrals(y: np.ndarray, u: np.ndarray) -> dict[str, float]:
    """Net, forward and reverse flux of one profile under three end closures."""
    out: dict[str, float] = {}
    closures = {
        "no_slip": (np.r_[0.0, y, 1.0], np.r_[0.0, u, 0.0]),
        "held": (np.r_[0.0, y, 1.0], np.r_[u[0], u, u[-1]]),
        "span": (y, u),
    }
    for name, (yy, uu) in closures.items():
        out[name] = float(np.trapezoid(uu, yy))
        out[name + "_forward"] = float(np.trapezoid(np.maximum(uu, 0.0), yy))
        out[name + "_reverse"] = float(np.trapezoid(np.minimum(uu, 0.0), yy))
    return out


def main() -> None:
    sheet = xlrd.open_workbook(sys.argv[1]).sheet_by_index(0)
    rows = range(11, 36)
    report = {}
    for label, ycol, ucol in (("x/H=1.0", 1, 2), ("x/H=2.0", 5, 6)):
        y, u = column(sheet, ycol, rows), column(sheet, ucol, rows)
        assert np.all(np.diff(y) > 0.0), "y must increase down the sheet"
        found = integrals(y, u)
        found["ratio_no_slip"] = found["no_slip"] / H_OVER_H
        found["ratio_held"] = found["held"] / H_OVER_H
        found["ratio_span"] = found["span"] / H_OVER_H
        found["points"] = int(y.size)
        found["y_first"], found["y_last"] = float(y[0]), float(y[-1])
        k = int(np.argmax(u < 0.0))
        found["zero_crossing"] = float(
            y[k - 1] + (y[k] - y[k - 1]) * u[k - 1] / (u[k - 1] - u[k])
        )
        report[label] = found
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

## Appendix D: digitize_fig.py

```python
"""Builder check, prompt 33, item 0: digitize the u/u0 circles of figures 5 and 6.

Nielsen (1990), R9040: figure 5 is the symmetry plane z/W = 0 and figure 6 the
plane z/W = 0.4, pages 13 and 14 of the Aalborg portal PDF, rendered here at
300 dpi. The figures are printed rotated: the page's horizontal axis is the
room height, ceiling on the left frame line and floor on the right, and u/u0
runs up the page from each profile's zero line, 0.2 per tick.

Circles are found by the Hough transform on the dark mask with the long frame
and zero lines removed, and kept if their circumference is dark, their centre
light, and a circle cut from the page fits them better than a triangle cut
from the page, which rejects the rms triangles. A profile's points are the
circles in its band of rows whose sign fits the profile: neither profile is
negative above mid-height, which removes the legend and label glyphs and the
other profile's points where the two bands overlap.

Usage: python digitize_fig.py SPEC_PDF WORKBOOK. Prints the net flux of each
profile over u0 h under three closures and, for figure 5, the digitized points
against the workbook's. Needs pymupdf, opencv-python-headless and xlrd, none of
them project dependencies.
"""

import json
import sys

import cv2
import numpy as np
import pymupdf
import xlrd

H_OVER_H = 0.056
DPI = 300
# Calibration read off each page at 300 dpi: the frame columns (ceiling,
# floor) and, for each profile, the zero line's row at the ceiling frame and at
# the floor frame (the scan is skewed by 4 to 6 px across the height). Every
# labelled tick on both frames of both figures sits 209 to 216.5 px from the
# next, 214 on average, so one scale serves all four profiles.
PX_PER_UNIT = 214.0 / 0.2
PAGES = {
    "fig5": {
        "page": 13, "ceiling": 729.0, "floor": 1805.0,
        "x/H=2.0": (1256.0, 1250.0), "x/H=1.0": (2323.0, 2317.0),
    },
    "fig6": {
        "page": 14, "ceiling": 663.0, "floor": 1740.0,
        "x/H=2.0": (1263.0, 1259.0), "x/H=1.0": (2329.0, 2326.0),
    },
}


def render(doc: pymupdf.Document, page: int) -> np.ndarray:
    """One page as an 8-bit grey image at DPI."""
    pix = doc[page - 1].get_pixmap(dpi=DPI, colorspace=pymupdf.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()


def long_lines(dark: np.ndarray) -> np.ndarray:
    """Mask of the long horizontal and vertical lines."""
    horiz = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((1, 121), np.uint8))
    vert = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((121, 1), np.uint8))
    return cv2.dilate(horiz | vert, np.ones((3, 3), np.uint8))


def circles(dark: np.ndarray) -> np.ndarray:
    """Centres (col, row) of open circles that pass the ring test."""
    clean = dark & (1 - long_lines(dark))
    img = cv2.GaussianBlur((255 - 255 * clean).astype(np.uint8), (5, 5), 1.0)
    found = cv2.HoughCircles(
        img, cv2.HOUGH_GRADIENT, dp=1, minDist=9, param1=80, param2=14,
        minRadius=7, maxRadius=13,
    )
    if found is None:
        return np.zeros((0, 2))
    kept = []
    angles = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    rows, cols = dark.shape
    for cx, cy, r in found[0]:
        ring = []
        for dr in (-1.5, 0.0, 1.5):
            xs = np.clip(np.round(cx + (r + dr) * np.cos(angles)).astype(int), 0, cols - 1)
            ys = np.clip(np.round(cy + (r + dr) * np.sin(angles)).astype(int), 0, rows - 1)
            ring.append(dark[ys, xs])
        on_ring = np.max(np.array(ring), axis=0).mean()
        core = dark[int(cy) - 2 : int(cy) + 3, int(cx) - 2 : int(cx) + 3].mean()
        if on_ring > 0.8 and core < 0.2:
            kept.append((cx, cy))
    return np.array(kept)


def template(gray: np.ndarray, col: float, row: float, half: int = 14) -> np.ndarray:
    """A marker template cut from the page around (col, row)."""
    c, r = int(round(col)), int(round(row))
    return gray[r - half : r + half + 1, c - half : c + half + 1].astype(np.float32)


def is_circle(gray: np.ndarray, col: float, row: float, ring: np.ndarray, tri: np.ndarray) -> bool:
    """True where the circle template fits better than the triangle template."""
    half = ring.shape[0] // 2
    c, r = int(round(col)), int(round(row))
    win = gray[r - half - 4 : r + half + 5, c - half - 4 : c + half + 5].astype(np.float32)
    if win.shape[0] < ring.shape[0] or win.shape[1] < ring.shape[1]:
        return False
    s_ring = float(cv2.matchTemplate(win, ring, cv2.TM_CCOEFF_NORMED).max())
    s_tri = float(cv2.matchTemplate(win, tri, cv2.TM_CCOEFF_NORMED).max())
    return s_ring > 0.5 and s_ring > s_tri


def flux(y: np.ndarray, u: np.ndarray) -> dict[str, float]:
    """Net flux under the no-slip and held closures, and over the span alone."""
    order = np.argsort(y)
    y, u = y[order], u[order]
    return {
        "no_slip": float(np.trapezoid(np.r_[0.0, u, 0.0], np.r_[0.0, y, 1.0])),
        "held": float(np.trapezoid(np.r_[u[0], u, u[-1]], np.r_[0.0, y, 1.0])),
        "span": float(np.trapezoid(u, y)),
        "points": int(y.size),
    }


def main() -> None:
    doc = pymupdf.open(sys.argv[1])
    sheet = xlrd.open_workbook(sys.argv[2]).sheet_by_index(0)
    workbook = {
        label: np.array([[sheet.cell_value(r, yc), sheet.cell_value(r, uc)] for r in range(11, 36)])
        for label, yc, uc in (("x/H=1.0", 1, 2), ("x/H=2.0", 5, 6))
    }
    # One clean circle and one clean triangle from figure 5, x/H = 2.0
    # (bounding boxes 24 x 24 at (1021, 1050) and 29 x 25 at (1250, 1098)).
    first = render(doc, 13)
    ring_t = template(first, 1032.5, 1061.5)
    tri_t = template(first, 1264.0, 1110.0)
    report: dict[str, dict] = {}
    for fig, cal in PAGES.items():
        gray = render(doc, cal["page"])
        dark = (gray < 128).astype(np.uint8)
        found = np.array([p for p in circles(dark) if is_circle(gray, p[0], p[1], ring_t, tri_t)])
        report[fig] = {"circles": int(found.shape[0])}
        y_all = (found[:, 0] - cal["ceiling"]) / (cal["floor"] - cal["ceiling"])
        for label in ("x/H=2.0", "x/H=1.0"):
            z_ceiling, z_floor = cal[label]
            zero = z_ceiling + (z_floor - z_ceiling) * y_all
            u_all = (zero - found[:, 1]) / PX_PER_UNIT
            inside = (y_all > 0.0) & (y_all < 1.0) & (u_all > -0.45) & (u_all < 0.85)
            inside &= ~((u_all < 0.0) & (y_all < 0.4))
            if label == "x/H=1.0":
                inside &= ~((u_all > 0.3) & (y_all > 0.3))
            y, u = y_all[inside], u_all[inside]
            entry: dict[str, object] = flux(y, u)
            entry["ratio_no_slip"] = entry["no_slip"] / H_OVER_H
            entry["ratio_held"] = entry["held"] / H_OVER_H
            entry["ratio_span"] = entry["span"] / H_OVER_H
            entry["profile"] = sorted(zip(y.round(4).tolist(), u.round(4).tolist()))
            if fig == "fig5":
                # Each workbook point against the nearest digitized circle.
                d = np.array([
                    [yw - y[k], uw - u[k]]
                    for yw, uw in workbook[label]
                    for k in [int(np.argmin((y - yw) ** 2 + (u - uw) ** 2))]
                ])
                entry["vs_workbook_du_mean"] = float(d[:, 1].mean())
                entry["vs_workbook_du_max"] = float(np.abs(d[:, 1]).max())
                entry["vs_workbook_dy_max"] = float(np.abs(d[:, 0]).max())
            report[fig][label] = entry
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
```

## Appendix E: summarize.py

```python
"""Summarize the prompt 33 probe histories written by probe33.py."""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
names = sys.argv[1:] or sorted(p.stem for p in HERE.glob("[pc][0-9]_*.json"))
for name in names:
    r = json.loads((HERE / f"{name}.json").read_text())
    res = np.array(r["residual"])
    spd = np.array(r["max_speed"])
    sw = np.array(r["sweeps"])
    signed = np.array(r["signed"])
    worst = np.array(r["worst"])
    n = res.size
    print(f"== {name}: {r['nx']}x{r['ny']} Re {r['reynolds']:.0f} cell Re {r['cell_reynolds']:.1f} "
          f"alpha_u {r['alpha_velocity']} cap {r['max_pressure_iter']} ptol {r['pressure_tol']} "
          f"ref vel {r['reference_velocity']:.4g} through-flow {r['through_flow']:.4g} kg/s "
          f"stop {r['stop']} n {n} t {r['seconds']:.0f}s")
    marks = sorted({0, 24, 49, 59, 69, 74, 99, 119, 149, 199, 249, 299, 399, 499, 749, 999} & set(range(n)) | {n - 1})
    for i in marks:
        print(f"   it {i:5d} res {res[i]:.3e} max|U| {spd[i]:.4g} sweeps {sw[i]:6d} "
              f"worst {worst[i]:.3e} signed {signed[i]:.3e}")
    k = int(np.argmin(res))
    print(f"   min residual {res[k]:.3e} at it {k}; at cap in {int(np.sum(sw >= r['max_pressure_iter']))} of {n}")
    for v in (5.0, 9.0, 20.0, 100.0):
        hit = np.where(spd > v)[0]
        if hit.size:
            print(f"   first max|U| > {v:g} m/s at it {hit[0]} ({spd[hit[0]]:.4g})")
    tail = res[max(0, n - 100):]
    print(f"   last 100: residual min {tail.min():.3e} max {tail.max():.3e} "
          f"median {np.median(tail):.3e}; max|U| {spd[max(0, n - 100):].min():.4g} to {spd[max(0, n - 100):].max():.4g}")
```

## Appendix F: calc33.py

The arithmetic ADR-012 quotes as its source 9: the Reynolds numbers, the Annex 20 inlet values,
the friction-velocity and y+ estimates, the implied kappa, the eddy-viscosity scale and the
particle relaxation time.

```python
"""Builder arithmetic, prompt 33: the numbers ADR-012 and the report quote."""

import json
import math
from pathlib import Path

C_MU, KAPPA, E_WALL = 0.09, 0.41, 9.793
out: dict[str, object] = {}

# Product room, configs/clean_room_default.yaml.
rho, mu, U, H, W = 1.2, 1.81e-5, 0.45, 3.0, 8.0
nu = mu / rho
out["product"] = {
    "nu": nu,
    "Re_H": rho * U * H / mu,
    "cell_Re_200x75": rho * U * (W / 200) / mu,
    "cell_Re_40x15": rho * U * (W / 40) / mu,
    "cell_Re_100x38": rho * U * (W / 100) / mu,
}
# Validation cases.
out["val001"] = {"Re_H": 1.0 * 0.1 * 0.5 / 0.01, "cell_Re_80x40": 1.0 * 0.1 * (0.5 / 40) / 0.01}
out["val002"] = {"Re": 1.0 * 1.0 * 1.0 / 0.01, "cell_Re_40x40": 0.025 / 0.01, "cell_Re_80x80": 0.0125 / 0.01}

# Annex 20, Nielsen (1990) R9040, eqs (1) to (7).
h, Ha, u0, nua = 0.168, 3.0, 0.455, 15.3e-6
k0 = 1.5 * (0.04 * u0) ** 2
l0 = h / 10.0
eps0 = k0**1.5 / l0
out["annex20"] = {
    "Re_h": h * u0 / nua,
    "Re_H": Ha * u0 / nua,
    "k0": k0,
    "l0": l0,
    "eps0_spec": eps0,
    "eps0_with_cmu34": C_MU**0.75 * k0**1.5 / l0,
    "nu_t0_over_nu": C_MU * k0**2 / eps0 / nua,
    "inlet_flux_m2s": u0 * h,
}

# Friction velocity estimates on surfaces of the product room. U is the speed
# outside the wall layer, x the run length along the surface.
def u_tau(U: float, x: float, law: str) -> float:
    re_x = U * x / nu
    cf = 0.0592 * re_x**-0.2 if law == "turbulent" else 0.664 * re_x**-0.5
    return U * math.sqrt(cf / 2.0)


cases = []
for U_e in (0.1, 0.2, 0.45):
    for x in (0.3, 0.6, 1.3, 3.0):
        cases.append({
            "U": U_e, "x": x, "Re_x": U_e * x / nu,
            "u_tau_turb": u_tau(U_e, x, "turbulent"),
            "u_tau_lam": u_tau(U_e, x, "laminar"),
        })
out["u_tau_cases"] = cases

def y_plus(ut: float, y: float, nu_: float) -> float:
    return ut * y / nu_


out["y_plus_product_first_node_0p02m"] = {
    f"{ut:g}": y_plus(ut, 0.02, nu) for ut in (0.005, 0.01, 0.02, 0.03, 0.05)
}
# Annex 20 grids: first tangential node at half the wall cell.
out["y_plus_annex20"] = {
    f"cell {dy:g} m": {f"{ut:g}": y_plus(ut, dy / 2.0, nua) for ut in (0.01, 0.02, 0.03)}
    for dy in (0.021, 0.042, 0.05)
}
# Wall cell for y+ = 1 at the first node, and the log-law lower edge 30.
out["wall_cell_for_yplus"] = {
    f"{ut:g}": {"y+=1": 2.0 * nu / ut, "y+=11.06": 2.0 * 11.06 * nu / ut, "y+=30": 2.0 * 30.0 * nu / ut}
    for ut in (0.01, 0.02, 0.03)
}

# Standard k-epsilon log layer: kappa implied by the constants (Launder and
# Spalding 1974): kappa^2 = (C2 - C1) sigma_eps sqrt(C_mu).
C1, C2, SIG_E = 1.44, 1.92, 1.3
out["kappa_implied_standard"] = math.sqrt((C2 - C1) * SIG_E * math.sqrt(C_MU))

# Eddy viscosity scale in the room for intensities 2% to 10% of the supply and
# length scales 0.05 to 0.3 m: nu_t = C_mu^(1/4) sqrt(k) l with k = 1.5 (I U)^2.
scale = {}
for I in (0.02, 0.05, 0.1):
    k = 1.5 * (I * U) ** 2
    for ell in (0.05, 0.1, 0.3):
        scale[f"I={I:g}, l={ell:g}"] = C_MU**0.25 * math.sqrt(k) * ell
out["nu_t_scale"] = scale
out["cell_Pe_with_nu_t_1e-3"] = U * 0.04 / 1e-3

# Stokes number of the 5 um class against a Kolmogorov time for eps 1e-4.
tau_p = 1000.0 * (5e-6) ** 2 * 1.034 / (18.0 * mu)
out["tau_p_5um"] = tau_p
out["kolmogorov_time_eps1e-4"] = math.sqrt(nu / 1e-4)

(Path(__file__).parent / "calc33.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
```
