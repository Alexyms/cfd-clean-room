# The Product Room's Reynolds Number and the Laminar Solver

**Date:** 2026-10-04
**Tree:** main at 02ec787, unchanged; the probes import `src/` and change nothing.
**Instruments:** `results/builder33/` and `results/builder33b/` (untracked). `probe33.py`,
`cost33.py`, `annex20_flux.py`, `digitize_fig.py`, `summarize.py` and `calc33.py` (prompt 33), and
`outlet33b.py`, `summarize33b.py`, `oned_keps.py`, `calc33b.py`, `table33b.py` and the launchers
(prompt 33b) are carried verbatim in the appendices, so this report stands without them.
**Measured values:** every figure in sections 2, 4, 5, 6 and 8 is from the builder's runs on
2026-10-04. Where the orchestrator measured the same quantity first, its figure is beside mine
and named as its.

## 1. Summary

The product room runs at a Reynolds number of 89,500 on its height and a cell Reynolds number of
1,190 on its 0.04 m mesh. The flow solver was validated at 5 and 100. As committed, the solve
diverges by outer iteration 58. On a 40x15 copy of the room with the pressure solved toward 1e-8,
the real viscosity diverges, as does ten times it; a hundred times stalls and, run
long enough, diverges; at a thousand times the residual falls without converging in 3,000
iterations. Removing the four obstacles does not stop the divergence. With heavier
under-relaxation (alpha_velocity 0.2) the runs have not diverged, nor converged: the residual of
the 40x15 run falls to 4.8e-6 by outer 2,809 and then rises again, never reaching the stopping
tolerance in 3,000 iterations (section 4, row 7), and the 200x75 run plateaus near 6e-4 over
300, a change of 0.048 m/s per outer iteration (residuals divide the largest velocity change by
the reference velocity, 79.2 m/s on 200x75 and 16.2 m/s on 40x15).

Section 8 measures the outlets, which today let air in with no condition of its own. Holding
inward-turning outlet faces closed, fixing the hood exhaust's flow at 0.5 m/s, or both, moves and
delays the divergence and converges nothing above Re 895: at Re 8,950 today's divergence sits at
the hood and the fixed-flow hood removes it there, the floor returns then diverging instead; at
the real viscosity the product mesh diverges at outer 427 with the two together against 267 with
today's outlets, and its growth starts inside the room before any outlet face reverses. Air drawn
in through the outlets is part of how the solve diverges, not why it fails to converge.

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
| 4 | 40x15, viscosity x1000 (Re 90), as row 3; 1,000 outers | Falling, not converged. Residual 5.27e-2 at 0, 1.83e-4 at 249, 2.65e-5 at 999 (4.3e-4 m/s), above the 1e-6 tolerance after 1,000 iterations and still falling; pressure never at its cap; largest speed 3.8 m/s. | "converges: residual 5.3e-2 to 1.8e-4 by outer 250" |
| 5 | 40x15, real air, alpha_u 0.2, as row 3; 500 outers | Not diverged. 2.25 m/s and residual 5.23e-3 at outer 75; residual 1.3e-4 at 499, the last 100 between 1.2e-4 and 6.5e-4; largest speed 3.3 m/s. | "not diverged at outer 75 (2.25 m/s, residual 5.2e-3); too short to call" |
| 6 | 40x15, viscosity x10 (Re 8,950), as row 3; 500 outers | Diverges. Speed past 5 m/s at outer 202, 102 m/s at 295, 2.2e6 m/s at 499. Pressure first at its cap at outer 220. | (builder's control) |
| 7 | 40x15, real air, alpha_u 0.2; 3,000 outers (row 5 continued) | Stalls without diverging. Residual 3.6e-5 at outer 1,000, least 4.8e-6 at 2,809, then rising: between 7.3e-6 and 1.5e-5 over the last 100; never below the 1e-6 tolerance; largest speed 3.87 m/s, the Re 90 run's level; pressure at its cap in 19 of 3,000. | (builder's control) |
| 8 | 200x75, real air, alpha_u 0.2, cap 5,000, tol 1e-8; 300 outers | Not diverged. Residual between 4.4e-4 and 6.1e-4 over the last 100; largest speed 2.3 m/s; signed sum -7.2e-2 kg/s, since a 5,000-sweep correction is not converged on this grid (section 5). | (builder's control) |
| 9 | 40x15, viscosity x100 (Re 895), as row 3; 1,000 outers | Stalls over the 1,000 run. Residual least 6.9e-4 at outer 589, then between 1.4e-3 and 4.9e-3 over the last 100; largest speed 3.7 m/s; pressure never at its cap. Continued in prompt 33b it diverges at outer 1,633: the residual rises from about 950, the hood draws air in from 1,188, and the speed passes 5 m/s at 1,277 over return 4 and ends at the hood (section 8). | (builder's control) |
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
field fall without converging at Re 90, stall and then diverge at 895, and diverge at 8,950; the divergence needs neither an
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
destabilizer at high Reynolds numbers; no run here isolates the outlets or the hood exhaust on
the right wall (section 8 does). (b) The initial field: every run starts from rest with the
supply switched on; none continues from the Re 90 field toward higher Reynolds numbers. (c) A
case with a known steady answer at a cell Reynolds number of tens to hundreds, such as the
lid-driven cavity at Re 1,000, which would separate what the solver can do from what the room
does. (d) Grid refinement at a fixed Reynolds number: real air ran on 40x15 and 200x75 only. (e)
A time-accurate solve: an unsteady flow appears to a steady iteration as one that does not
converge, and the solver has no time-accurate mode.

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
width from the side wall. At x/H = 2.0 it found 22 of the plane's circles; the three it missed sit
under rms triangles or overlap one, and the tester's second digitization (test 33, check 13),
whose union with this one has the traverse's 25 points, gives 1.115 (no slip):

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
that "The measured flow in the benchmark, [1], does not show this large difference" (file page
7, above its figure 10). That
is consistent with the flux check: point by point the two planes' profiles differ by a few
hundredths of u0, which the eye does not call large, and those hundredths integrate to half the
inlet flux because the net flux is the small difference of a forward and a return flux each two
to four times larger. That the symmetry plane's shortfall is air moving across the width is
inferred from two measured planes, not measured: two planes cannot close the width integral
(their straight-line mean over the inner 0.4 W is 0.87 of u0 h, so the outer tenth on each side
would have to carry about 1.5), and at x/H = 1.0 both planes sit above the inlet flux. The
specification's figure 10 (report page 13) supports the reading independently: hot-wire profiles
at x/H = 2.0, z/W = 0 and Re 7,100 in models of W/H = 4.7, 1.0 and 0.5 agree in the ceiling jet,
and the floor return strengthens as the model narrows; Nielsen's note beside it reads "The
measurements suggest three-dimensional flow but it is not possible to exclude some influence
from the probe support due to the small width of the model." The prediction was that x/H = 1.0
would come within about 10%, with three-dimensionality the likely cause of any departure. It
came within 2% to 32% depending on the closure, 17% with no slip; the larger departure is at
x/H = 2.0, and three-dimensionality is the reading the two planes and figure 10 support.

### 6.4 What it implies for a two-dimensional comparison

A two-dimensional solution carries u0 h through every section, so at x/H = 2.0 it cannot match
the symmetry plane closer than the shortfall, 0.38 x 0.056 = 0.021 in the integral of u / u0 over
y / H: 0.021 u0 if spread over the height, about 0.04 u0 if it sits in the return flow below the
zero crossing at y / H = 0.46. That is a floor on any two-dimensional model's agreement there,
whatever its turbulence model, and it is consistent with the 2010 paper's remark that a
two-dimensional low-Reynolds k-epsilon prediction leaves the counter flow "slightly
underestimated" (the paragraph introducing its figure 5, file page 4). At x/H = 1.0 the two
planes agree within 0.10 of the inlet flux and both sit near or above it, so a comparison there
is limited by the closures, not by the room's three-dimensionality. The data therefore support a
two-dimensional comparison at x/H = 1.0 and along the jet under the ceiling, and not a scored one
at x/H = 2.0 against the symmetry plane alone. The criterion is ADR-012's decision 6 and stays
OPEN.

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

Section 8, from `results/builder33b/`: `sh run_ladder.sh` (D1, D2 and the sixteen ladder runs),
`sh run_real.sh` (R_T3, R_T2) and `sh run_sens.sh` (the four hood-sensitivity runs), each line of
the form `python outlet33b.py NAME NX NY MU_FACTOR N_OUTER ALPHA_U MAX_P_ITER P_TOL TREATMENT
[HOOD]`; then `python summarize33b.py` for the runs that wrote a history and `python table33b.py`
for every run's state (appendices G, H and K). A run writes `NAME.json` when it stops on its own;
the runs stopped by cost left only `NAME.log`. The 1D k-epsilon references and the corrected
arithmetic: `python oned_keps.py` and `python calc33b.py` (appendices I and J).

## 8. The outlets (prompt 33b)

The premise review of this report (docs/prompts/premise-33.md, B1) located the divergence of
rows 3 and 6 at pressure outlets drawing air into the room: the floor returns at the real
viscosity, the hood exhaust at ten times it, and no reversed outlet face in the run that
converges. Today's outlet layer extrapolates each outlet face's velocity from the interior
whatever its sign and corrects it against p' = 0 (`StaggeredSolver._extrapolate_outlets`,
`PressureCorrector._face_d`), so air entering through an outlet has no condition of its own.
This section measures how much of the divergence that explains.

### 8.1 What each outcome means (written before the runs)

- **A treatment converges the laminar room at real air.** The outlets were the instability.
  Turbulence is then still required for the particle physics, since turbulent mixing spreads
  particles about a million times faster than Brownian motion (ADR-012 F), but not for
  convergence.
- **Treatments converge at Re 8,950 but not at real air.** The outlets are part of the
  instability and the Reynolds number the rest.
- **Nothing changes.** The reversal was a symptom, and the attribution to the outlets is
  refuted by measurement.

The orchestrator's prediction, recorded in prompt 33b before the runs: the real-room diagnostic
shows the growth at reversed floor-return faces; T1 removes the divergence at Re 8,950 and slows
it at real air without converging; T2 alone helps only at the hood; T3 converges at 8,950 and
stalls at real air, the second outcome.

### 8.2 Method

`outlet33b.py` (appendix G) runs the staggered solver through a subclass that overrides
`_extrapolate_outlets`, the hook the solver calls before every prediction; `src/` is unchanged.
After today's extrapolation it applies one of four treatments:

- **T0**, today's outlets, unchanged. Each T0 run reproduces the matching run of section 4
  bit for bit in every residual over the iterations both ran: D1 row 1 over 99, D2 row 2 over
  150, and the Re 90, 895, 8,950 and real-air rungs rows 4, 9, 6 and 3 over 1,000, 1,000, 296
  and 220.
- **T1**, a backflow condition at every pressure outlet: a face whose extrapolated velocity
  points into the room is held at zero normal velocity for that outer iteration and taken out of
  the pressure correction's outlet mask (`PressureCorrector._out_bottom`, `_out_right`), so for
  its normal component it is a wall for that iteration, its tangential condition staying the
  pressure outlet's zero gradient (appendix G's "a wall" means the same); OpenFOAM's
  `inletOutlet` switch.
- **T2**, the hood exhaust as a fixed-flow exhaust: its faces hold 0.5 m/s outward and leave the
  outlet mask, so their normal component is a Dirichlet velocity like an inlet's; their
  tangential condition stays the pressure outlet's zero gradient, where a built segment holds it
  at zero (test 33b, B2; section 8.6). The floor returns as today. On
  40x15 the hood covers 5 faces (1.0 m), on 200x75 23 faces (0.92 m), against the 0.9 m opening.
- **T3**, T1 at the floor returns and T2 at the hood.

Each outer iteration records the solver's residual and the step it stands for in m/s, the
pressure sweeps, the largest cell-centred speed and the cell centre it sits at, the signed and
worst mass imbalance, and per outlet segment the faces whose corrected velocity points into the
room and the faces T1 held closed. A run stops when the speed passes 100 m/s (diverged), at the
solver's own stop, or at the outer cap. The ladder ran 40x15 with the pressure toward 1e-8 (cap
40,000) and alpha_velocity 0.5, as prompt 33's rows 3, 4, 6 and 9 did, up to 3,000 outer
iterations. Sixteen ladder runs, two real-room diagnostics, two real-room treatment runs and four
hood-sensitivity runs ran in parallel on one machine. The runs marked cut were stopped together,
about 25 minutes after the ladder started, one stop for cost: the slowest, the fixed-flow hood at
real air or Re 8,950, ran at about 2.7 s per iteration with the pressure at its cap, while T2 and
T3 at Re 90 and 895 never reached the cap and ran at 0.9 to 1.5 s. Their state is read from the
log, which prints every 25th iteration. On 40x15 the floor returns have
4, 3, 2 and 4 faces and the hood 5; on 200x75 19, 14, 8, 15 and 23.

### 8.3 The real room: where the growth sits

| Run | First past 5 m/s | First outlet face reversed | First past 20 m/s | End |
|---|---|---|---|---|
| D1, as committed (cap 200) | outer 46 at (5.82, 2.10) | 55 | 90 at (6.06, 0.02), over return 4 | diverged at 99 at (1.22, 0.02), over return 1; 23 faces reversed |
| D2, cap 5,000, tol 1e-8, alpha 0.5 | 119 at (7.62, 2.46) | 130 | 183 at (6.90, 1.94) | diverged at 267 at (2.66, 0.02), over return 2; 39 reversed |
| R_T2, D2 with T2 | 135 at (0.30, 2.34) | 136 | 166 at (4.66, 0.02), over return 3 | diverged at 292 at (2.50, 0.02), over return 2; 27 reversed |
| R_T3, D2 with T3 | 135 at (0.30, 2.34) | 139 (closing from 120) | 226 at (2.70, 2.06) | diverged at 427 at (6.06, 1.02); 13 reversed, 28 held closed |

Reading. The orchestrator's prediction, that the real-room diagnostic shows the growth at
reversed floor-return faces, holds for where the divergence ends and not for where it starts. In
D1, with continuity lost from the first iteration, the speed first grows at the entrance of the
gap between the etch chamber and the hood bench at the equipment-top height, then at the floor
over returns 4 and 1, the first face reversing between the two. In D2 it grows from 1.5 to 4 m/s
over the first 114 iterations at the supply's right-hand end (x = 7.5 m, y = 2.5 m) and above
the hood bench, with no outlet face reversed until outer 130; from 150 the fastest cell is on the
floor over returns 1 and 2, which carry the reversed faces. With the hood at a fixed flow (R_T2,
R_T3) the growth first shows at the supply's left end, (0.3, 2.34), at the same time as the
first reversal, and the run still diverges: at 292 under T2 and 427 under T3, against 267 under
today's outlets. On the product mesh, then, air drawn in through the openings arrives with the
divergence and speeds it up, the fixed-flow hood and the closed faces delay it, and none of the
treatments stops it. These runs stopped each pressure correction at its 5,000-sweep cap; with the
pressure solved to the committed 1e-6 at every iteration, test 33b finds the same start of growth
(section 8.6).

### 8.4 The ladder

40x15, the pressure toward 1e-8, alpha_velocity 0.5. Residuals are the solver's (section 4); a
run cut by cost gives its state at the last logged iteration, and its least residual is over the
logged iterations. Reversed faces are those whose corrected velocity points into the room, the
most at one time on returns 1 to 4 and the hood among the logged iterations, every 25th (counted
at every iteration in test 33b for two runs, in brackets); under T1 and T3 they are faces the extrapolation
left open and the correction turned inward. The cells named: (4.7, 2.1) is over the litho tool's
right corner, where the converging runs' fastest air passes down to return 3; (6.1, y) is the gap
between the etch chamber and the hood bench, over return 4; (7.7, 1.1) is the cell in front of
the hood.

| Re | Outlets | Outer run | End | Residual: least, at end | Largest speed at end (m/s), cell | Reversed faces, most at once |
|---|---|---|---|---|---|---|
| 90 | T0 | 3,000 | cap, falling | 4.95e-6, 4.95e-6 | 4.15, (4.7, 2.1) | 0, 0, 0, 0, 5: the hood from outer 1,228 |
| 90 | T1 | 3,000 | cap, falling | 1.93e-6, 1.93e-6 | 4.11, (4.7, 2.1) | none; hood faces held closed from 1,167, four at the end |
| 90 | T2 | 1,676 | cut, falling | 6.24e-6, 6.24e-6 | 3.78, (4.7, 2.1) | none |
| 90 | T3 | 1,701 | cut, falling; the same history as T2, no floor face turning inward | 6.03e-6, 6.03e-6 | 3.79, (4.7, 2.1) | none |
| 895 | T0 | 1,633 | diverged | 6.93e-4 at 589, 7.1e-1 | 102, (7.7, 1.1) | 0, 1, 0, 1, 5: return 2 from 384, the hood from 1,188, return 4 from 1,323 |
| 895 | T1 | 1,976 | cut, growing | 1.8e-3, 1.1e-1 | 8.4 (12.4 at 1,950), (6.1, 1.3) | 3, 1, 1, 1, 2 |
| 895 | T2 | 1,001 | cut, bounded | 3.0e-3, 6.9e-3 | 3.63, (4.7, 2.1) | 0, 1, 0, 0, 0 |
| 895 | T3 | 1,001 | cut, bounded | 2.3e-3, 4.7e-3 | 3.48, (4.7, 2.1) | none |
| 8,950 | T0 | 296 | diverged | 4.7e-3, 7.9e-1 | 102, (7.7, 1.1) | 0, 0, 0, 1, 5: the hood from 102 |
| 8,950 | T1 | 926 | cut, growing | 6.4e-3, 1.5e-1 | 13.5 (17.0 at 850), (6.3, 1.1) | 2, 3, 1, 1, 4 |
| 8,950 | T2 | 576 | cut while diverging | 7.7e-3, 8.4e-1 | 66.8, (2.9, 0.1) over return 2 | 4, 3, 0, 1, 0 |
| 8,950 | T3 | 551 | cut, growing and oscillating | 7.7e-3, 1.5e-1 | 8.5 (4.8 at 300, 8.1 to 14.4 from 400), (6.1, 0.9) | 2, 2, 1, 1, 0 (3, 3, 2, 2, 0) |
| 89,500 | T0 | 220 | diverged | 3.1e-2, 1.2 | 104, (0.7, 0.3) over return 1 | 3, 3, 0, 2, 5 |
| 89,500 | T1 | 676 | cut, bounded | 5.9e-2, 1.4e-1 | 11.8 (10 to 16 from 250), (6.1, 1.3) | 2, 2, 1, 1, 2 |
| 89,500 | T2 | 340 | diverged | 3.3e-2, 9.9e-1 | 102, (2.5, 0.3) over return 2 | 3, 3, 0, 2, 0 |
| 89,500 | T3 | 576 | cut, growing | 5.3e-2, 2.1e-1 | 17.1 (3.8 at 100), (7.3, 1.1) | 2, 2, 1, 1, 0 (3, 3, 1, 2, 0 to outer 700) |

Reading.

*Re 90.* Every treatment's residual falls, slowly, as today's outlets' does; no treatment made
the control diverge. Over 3,000 iterations today's hood turns inward from outer 1,228 until all
five faces draw air in, and the residual goes on falling through it: a reversed outlet face is not
by itself a divergence. Holding those faces closed (T1) or fixing the hood's flow (T2, T3) lowers
the residual faster: 1.0e-5 and 6.2e-6 at outer 1,675, against 1.27e-5 with today's outlets.

*Re 895.* Today's outlets stall from about outer 600 and diverge at 1,633. The residual rises
from about 950; the hood draws air in from 1,188; the speed first grows on the floor over return 4
(3.9 m/s at (6.1, 0.1) at 1,200), passes 5 m/s at 1,277 higher in the same gap, and ends at the
hood's cell, past 20 m/s at 1,555. T1 departs earlier and grows more slowly: residual rising from
about 650, past 5 m/s near 1,100, 8 to 12 m/s and rising when cut at 1,976. T2 and T3 stayed at
the converging runs' speed, 3.5 to 3.6 m/s, with residuals between 2.3e-3 and 7.1e-3 from outer
100 until they were cut at 1,001. Today's outlets were in the same state at that point (3.74 m/s,
residual 4.9e-3 and rising), and their speed left that level only from about 1,200, so whether a
fixed-flow hood removes the Re 895 divergence was not measured here. Test 33b continued both
runs: the fixed-flow hood delays the departure and does not remove it (section 8.6).

*Re 8,950.* Today's hood draws air in from outer 102 and the run diverges there at 296, the
premise review's reading. Fixing the hood's flow (T2) removes that: the hood has no reversed face,
and the run diverges over returns 1 and 2 instead, past 20 m/s by 500 and at 67 m/s at 576.
Holding the reversed faces closed, alone (T1) or with the fixed hood (T3), keeps the speed from
running away over the iterations run, 926 and 551, and neither converges: T1 grows to 17 m/s by
850 in the gap over return 4, T3 rises from 4.8 m/s at 300 to 14.4 at 475 and then swings between
8.1 and 14.4 m/s, residuals 0.1 to 0.35.

*Real air.* Every run is far from the fixed point from its first iteration, which reaches 14 m/s
at a corner of the supply: the left under today's outlets, the right, (7.5, 2.9), with the hood's
flow fixed (T2, T3; test 33b). Today's outlets diverge at 220 over return 1 and T2 at 340 over
return 2; T1 holds between 10 and 16 m/s from 250 to its cut at 676; T3 grows from 3.8 m/s at 100
to 17 m/s at its cut at 576, and continued in test 33b holds at 10 to 20 m/s to outer 700
without diverging (section 8.6). T1's and T3's residuals stay between 0.06 and 0.25. Heavier
under-relaxation with today's outlets (section 4, rows 5 and 7, alpha_velocity 0.2) did more than
any outlet treatment here: 3.9 m/s and a residual down to 4.8e-6.

*The hood's set flow.* T3 at 0.4 and 0.6 m/s against 0.5, at outer 250 (the sensitivity runs were
cut at 251):

| Re | Hood 0.4 m/s | Hood 0.5 m/s | Hood 0.6 m/s |
|---|---|---|---|
| 8,950 | 3.77 m/s, residual 4.1e-2 | 4.06 m/s, 4.7e-2 | 5.33 m/s, 7.5e-2 |
| 89,500 | 6.95 m/s, 1.1e-1 | 8.48 m/s, 9.6e-2 | 6.77 m/s, 9.9e-2 |

At Re 8,950 the speed and the residual rise with the set flow; at real air they do not order. No
setting is near converging at outer 250, so the result does not hang on the value 0.5.

### 8.5 Which outcome

None of the three as written. No treatment converges the room at real air or at Re 8,950, so
neither of the first two happened. Nor did nothing change: the treatments move the divergence and
delay it. The fixed-flow hood removes the hood-located divergence at Re 8,950 and delays the one at
real air (outer 340 against 220 on 40x15, 292 against 267 on 200x75); holding the floor returns'
reversed faces closed as well delays it further on 200x75 (427) and keeps the speed bounded over
the iterations run on 40x15. The measurement is closest to the third outcome with that
qualification. Air drawn in through the outlets is part of how the divergence grows, and at Re
8,950 under today's outlets it is where the divergence sits. It is not why the iteration fails to
converge: with the hood's flow fixed and every face the extrapolation turns inward held closed,
the runs still do not converge above Re 895, and on the product mesh the growth starts inside the
room, at the supply's ends and above the hood bench, before any outlet face reverses (D2). Premise
review B1's location of the divergence is confirmed at Re 8,950 and, for where it ends, at real
air; its attribution of the non-convergence to the outlets is refuted by measurement. The Re 895
rung, where the hood reverses a few iterations before the growth begins, was left open by the
cost: T2 and T3 were cut before today's outlets departed. Test 33b closed it (section 8.6): the
fixed-flow hood delays the departure by about 900 to 1,000 iterations and does not remove it.

What this leaves for ECR-002: the outlets need a condition for entering air before any coupled
solve, because today's let the divergence concentrate at the hood and the returns, and a turbulent
solve would inherit that; and convergence at the room's effective viscosity is not bought by the
outlets, so it stays the hypothesis ECR-002 step 5 measures.

**Predictions against the measurement.**

| Orchestrator's prediction | Measured |
|---|---|
| The real-room diagnostic shows the growth at reversed floor-return faces | Held for where it ends, missed for where it starts: in D2 the speed grows for 114 iterations at the supply's right end and above the hood bench before any face reverses at 130 |
| T1 removes the divergence at Re 8,950 | Missed: not diverged at its cut, 926, but growing, 17 m/s at 850, residual 0.15 to 0.35 |
| T1 slows it at real air without converging | Held: 10 to 16 m/s from 250 to 676, not converging |
| T2 alone helps only at the hood | Held: at Re 8,950 the hood's divergence goes and the floor returns diverge instead; at real air the divergence is delayed from 220 to 340 |
| T3 converges at Re 8,950 | Missed: it rises to 14.4 m/s at 475 and swings between 8.1 and 14.4 m/s, residual 0.15 at its cut, 551 |
| T3 stalls at real air | Missed on 200x75, where it diverges at 427; on 40x15 it grows slowly, 17 m/s at its cut, 576 |
| The outcome is "outlets are part, the Reynolds number the rest" | Missed: no treatment converges Re 8,950 |

**Cut by cost.** Fourteen runs were stopped together before 3,000 iterations, about 25 minutes
after the ladder started, one stop for cost (section 8.2): T2 and T3 at Re 90 (1,676 and 1,701), T1 at Re 895 (1,976), T2
and T3 at Re 895 (1,001), T1, T2 and T3 at Re 8,950 (926, 576, 551), T1 and T3 at real air (676,
576), and the four hood-sensitivity runs at 251. None of them had converged. Test 33b continued
the runs whose cut left a question open (section 8.6).

### 8.6 Continued in test 33b

`/cfd-test 33b` (`docs/prompts/test-33b.md`) reimplemented T0, T1 and T3, reproduced the histories
above bit for bit wherever both exist, and continued the runs the cost cut. What it adds:

- Re 895 with the hood's flow fixed. The residual rises from outer 103 (T2) or 601 (T3); the speed
  leaves 3.5 m/s near 2,000 and passes 5 m/s at 2,165 (T2) and 2,284 (T3), against 1,277 under
  today's outlets, the growth starting in the gap over return 4 as today's did. The fixed-flow
  hood delays the Re 895 departure by about 900 to 1,000 iterations and does not remove it.
- T2 at Re 8,950 diverges at outer 615, over return 2.
- T3 at Re 8,950 and at real air, and T1 at Re 8,950, run to outer 700 or 1,000: none diverges;
  each grows to 10 to 20 m/s, four or five times the speed of the runs that converge on this
  grid, and holds there, residuals 0.1 to 0.35. That is not convergence.
- Reversed faces counted at every iteration rather than every 25th: T3 at Re 8,950 has at most 3,
  3, 2 and 2 faces of returns 1 to 4 inward at once, T3 at real air 3, 3, 1 and 2 by outer 700.
- The 200x75 room under today's outlets with the pressure solved to the committed 1e-6 at every
  iteration, where section 8.3's runs stopped each correction at 5,000 sweeps: the speed grows
  inside the room from 1.46 m/s at outer 13 to 5.63 at 116 before the first outlet face reverses,
  the hood's at 116. Section 8.3's reading does not hang on the cap.
- The hood's tangential condition. The probes kept the hood's tangential velocity at the pressure
  outlet's zero gradient. Holding it at zero instead, as a built segment does, moves T3's residual
  at Re 8,950 by 6.5e-8 relative at outer 0 and 9.3e-7 at 25, with a different sweep count, so
  ECR-002 criterion 6 compares against the probe rerun that way.

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

## Appendix G: outlet33b.py

The outlet probe of section 8.

```python
"""Builder probe, prompt 33b: the product room's outlets under three treatments.

Usage: python outlet33b.py NAME NX NY MU_FACTOR N_OUTER ALPHA_U MAX_P_ITER P_TOL TREATMENT [HOOD]

Loads configs/clean_room_default.yaml, overrides the grid, multiplies the
viscosity, sets the outer cap, the velocity under-relaxation and the pressure
sweep cap and tolerance (0 keeps the committed value), and runs the staggered
solver under the committed velocity_step rule from rest with one outlet
treatment, applied by a subclass in this script; src/ is not changed.

T0  today's outlets: every pressure-outlet face takes its interior
    neighbour's velocity before the prediction and is corrected against
    p' = 0, whatever its sign.
T1  a backflow condition at every pressure outlet: after today's
    extrapolation, a face whose velocity points into the room is held at zero
    normal velocity for that outer iteration and taken out of the pressure
    correction's outlet mask, so it is a wall for that iteration (the
    inflow-outflow switch OpenFOAM ships as inletOutlet).
T2  the hood exhaust as a fixed-flow exhaust: its faces hold HOOD m/s outward
    (default 0.5) and leave the outlet mask; the floor returns as today.
T3  T1 at the floor returns and T2 at the hood.

Each outer iteration records the solver's residual, the step in m/s (the
residual times the reference velocity), the pressure sweeps, the largest
cell-centred speed and the cell centre it sits at, the worst per-cell and
signed domain-sum mass imbalance of the corrected faces, and per outlet
segment the faces whose corrected velocity points into the room and, under
T1 and T3, the faces held closed. A run stops at the solver's own stop, at a
non-finite field, or when the largest speed passes 100 m/s, which is called
diverged. Writes NAME.json beside this file.
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

from src.boundary_registry import BoundaryRegistry  # noqa: E402
from src.boundary_staggered import StaggeredBoundary  # noqa: E402
from src.config import SimConfig  # noqa: E402
from src.mesh import Mesh  # noqa: E402
from src.solver_staggered import StaggeredSolver  # noqa: E402
from src.staggered import edge_cell_inputs  # noqa: E402

DIVERGED_SPEED = 100.0


class Stop(Exception):
    """Raised from the callback to end a run that has diverged."""


def segment_names(mesh: Mesh, cfg: SimConfig, edge: str) -> np.ndarray:
    """The covering segment's name for every face along an edge, '' for none."""
    coords, solid = edge_cell_inputs(mesh, edge)
    cover = BoundaryRegistry(cfg).coverage_along(edge, coords, solid)
    return np.array([c.name or "" for c in cover])


class OutletSolver(StaggeredSolver):
    """The staggered solver with an outlet treatment applied before each prediction."""

    def __init__(self, mesh, cfg, boundary, treatment: str, hood_speed: float, hood) -> None:  # type: ignore[no-untyped-def]
        super().__init__(mesh, cfg, boundary)
        self.treatment = treatment
        self.hood_speed = hood_speed
        self.hood = hood
        self.base_bottom = self._outlets["bottom"].is_outlet.copy()
        self.base_right = self._outlets["right"].is_outlet.copy()
        self.closed_bottom = np.zeros_like(self.base_bottom)
        self.closed_right = np.zeros_like(self.base_right)

    def _extrapolate_outlets(self, u: np.ndarray, v: np.ndarray) -> None:
        super()._extrapolate_outlets(u, v)
        if self.treatment == "T0":
            return
        bottom = self.base_bottom.copy()
        right = self.base_right.copy()
        if self.treatment in ("T2", "T3"):
            u[self.hood, -1] = self.hood_speed
            right &= ~self.hood
        if self.treatment in ("T1", "T3"):
            # Into the room is +v on the floor and -u on the right wall.
            self.closed_bottom = bottom & (v[0, :] > 0.0)
            self.closed_right = right & (u[:, -1] < 0.0)
            v[0, self.closed_bottom] = 0.0
            u[self.closed_right, -1] = 0.0
            bottom &= ~self.closed_bottom
            right &= ~self.closed_right
        self._corrector._out_bottom = bottom
        self._corrector._out_right = right


def main() -> None:
    name = sys.argv[1]
    nx, ny = int(sys.argv[2]), int(sys.argv[3])
    mu_factor, n_outer = float(sys.argv[4]), int(sys.argv[5])
    alpha_u, max_p, p_tol = float(sys.argv[6]), int(sys.argv[7]), float(sys.argv[8])
    treatment = sys.argv[9]
    hood_speed = float(sys.argv[10]) if len(sys.argv) > 10 else 0.5

    raw = yaml.safe_load((ROOT / "configs/clean_room_default.yaml").read_text())
    raw["domain"]["nx"], raw["domain"]["ny"] = nx, ny
    raw["fluid"]["viscosity"] *= mu_factor
    block = raw["solver"]
    block["max_simple_iter"] = n_outer
    if alpha_u:
        block["alpha_velocity"] = alpha_u
    if max_p:
        block["max_pressure_iter"] = max_p
    if p_tol:
        block["pressure_tol"] = p_tol
    cfg = SimConfig.from_dict(raw)
    mesh = Mesh(cfg)
    boundary = StaggeredBoundary(mesh, cfg)
    bottom_names = segment_names(mesh, cfg, "bottom")
    right_names = segment_names(mesh, cfg, "right")
    hood = right_names == "hood_exhaust"
    solver = OutletSolver(mesh, cfg, boundary, treatment, hood_speed, hood)

    corrector = solver._corrector
    original = corrector.correct
    latest: dict[str, np.ndarray] = {}

    def correct(prediction, p):  # type: ignore[no-untyped-def]
        result = original(prediction, p)
        latest["u"], latest["v"] = result.u, result.v
        return result

    corrector.correct = correct  # type: ignore[method-assign]

    segments = sorted({n for n in bottom_names if n.startswith("floor_return")}) + ["hood_exhaust"]
    masks = {n: ("bottom", bottom_names == n) for n in segments if n != "hood_exhaust"}
    masks["hood_exhaust"] = ("right", hood)
    xc, yc = np.meshgrid(mesh.xc, mesh.yc)
    ref = solver.reference_velocity
    rec: dict[str, list] = {
        "residual": [], "step": [], "sweeps": [], "max_speed": [], "at": [],
        "worst": [], "signed": [], "inflow": [], "closed": [],
    }
    t0 = time.perf_counter()

    def callback(state) -> None:  # type: ignore[no-untyped-def]
        speed = np.hypot(state.u, state.v)
        k = int(np.argmax(speed))
        top = float(speed.flat[k])
        u, v = latest["u"], latest["v"]
        imbalance = corrector.mass_imbalance(u, v)
        inflow, closed = [], []
        for n in segments:
            edge, m = masks[n]
            if edge == "bottom":
                inflow.append(int(np.sum(v[0, m] > 0.0)))
                closed.append(int(np.sum(solver.closed_bottom[m])))
            else:
                inflow.append(int(np.sum(u[m, -1] < 0.0)))
                closed.append(int(np.sum(solver.closed_right[m])))
        rec["residual"].append(float(state.residual))
        rec["step"].append(float(state.residual) * ref)
        rec["sweeps"].append(int(state.pressure_sweeps))
        rec["max_speed"].append(top)
        rec["at"].append((round(float(xc.flat[k]), 3), round(float(yc.flat[k]), 3)))
        rec["worst"].append(float(np.max(np.abs(imbalance))))
        rec["signed"].append(float(np.sum(imbalance)))
        rec["inflow"].append(inflow)
        rec["closed"].append(closed)
        it = state.iteration
        if it % 25 == 0 or it == n_outer - 1:
            print(
                f"{name} it {it:5d} res {state.residual:.3e} step {rec['step'][-1]:.2e} "
                f"sweeps {state.pressure_sweeps:6d} max|U| {top:.4g} at {rec['at'][-1]} "
                f"inflow {inflow} closed {closed} t {time.perf_counter() - t0:7.1f}s",
                flush=True,
            )
        if not math.isfinite(top) or top > DIVERGED_SPEED:
            raise Stop

    try:
        solver.solve_steady(on_iteration=callback)
        stop = solver.stop_reason
    except Stop:
        stop = "diverged"
    out = {
        "name": name, "nx": nx, "ny": ny, "mu": cfg.mu, "treatment": treatment,
        "hood_speed": hood_speed if treatment in ("T2", "T3") else None,
        "hood_faces": int(hood.sum()), "segments": segments,
        "faces_per_segment": [int(masks[n][1].sum()) for n in segments],
        "reynolds": cfg.rho * 0.45 * cfg.room_height / cfg.mu,
        "alpha_velocity": cfg.alpha_velocity, "max_pressure_iter": cfg.max_pressure_iter,
        "pressure_tol": cfg.pressure_tol, "reference_velocity": ref,
        "stop": stop, "seconds": time.perf_counter() - t0, **rec,
    }
    (Path(__file__).parent / f"{name}.json").write_text(json.dumps(out))
    print(f"{name}: Re {out['reynolds']:.0f} {treatment} stop {stop} after {len(rec['residual'])}", flush=True)


if __name__ == "__main__":
    main()
```

## Appendix H: summarize33b.py

Reads the outlet probe's histories and prints the figures section 8 quotes.

```python
"""Summarize the prompt 33b outlet runs written by outlet33b.py."""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
names = sys.argv[1:] or sorted(p.stem for p in HERE.glob("[DLRH]*.json"))
for name in names:
    r = json.loads((HERE / f"{name}.json").read_text())
    res = np.array(r["residual"])
    step = np.array(r["step"])
    spd = np.array(r["max_speed"])
    inflow = np.array(r["inflow"])
    closed = np.array(r["closed"])
    n = res.size
    k = int(np.argmin(res))
    tail = slice(max(0, n - 100), n)
    first_in = [int(np.argmax(inflow[:, j] > 0)) if inflow[:, j].any() else None for j in range(inflow.shape[1])]
    first_closed = [int(np.argmax(closed[:, j] > 0)) if closed[:, j].any() else None for j in range(closed.shape[1])]
    grow = np.where(spd > 20.0)[0]
    print(f"== {name}: Re {r['reynolds']:.0f} {r['treatment']} hood {r['hood_speed']} "
          f"{r['nx']}x{r['ny']} stop {r['stop']} after {n} ({r['seconds']:.0f} s)")
    print(f"   residual least {res[k]:.2e} at {k}; last 100 {res[tail].min():.2e} to {res[tail].max():.2e}; "
          f"step at end {step[-1]:.2e} m/s; speed at end {spd[-1]:.3g} at {r['at'][-1]}")
    print(f"   segments {r['segments']} faces {r['faces_per_segment']}")
    print(f"   first inflow face per segment {first_in}; most at once {inflow.max(axis=0).tolist()}; at end {inflow[-1].tolist()}")
    print(f"   first closed per segment {first_closed}; most at once {closed.max(axis=0).tolist()}; at end {closed[-1].tolist()}")
    if grow.size:
        g = int(grow[0])
        print(f"   speed past 20 m/s at {g} at {r['at'][g]}, inflow then {inflow[g].tolist()}, closed {closed[g].tolist()}")
    marks = sorted({0, 99, 249, 499, 999, 1999, n - 1} & set(range(n)))
    print("   " + "; ".join(f"{i}: res {res[i]:.1e} spd {spd[i]:.3g}" for i in marks))
```

## Appendix I: oned_keps.py

The one-dimensional standard k-epsilon solve ADR-012 cites as its source 15 (premise review B3).

```python
"""Builder check, prompt 33b, premise B3: the standard k-epsilon model in one dimension.

Solves the fully developed, high-Reynolds standard k-epsilon equations across
a plane channel (half height delta, symmetry at the centre) and across plane
Couette flow (walls at 0 and h, constant stress), with the wall-function
values imposed at the first node y0 in the log layer:

    k(y0) = u_tau^2 / sqrt(C_mu),  eps(y0) = u_tau^3 / (kappa y0),  kappa 0.41.

The momentum equation integrates once to the local stress, so
(nu + nu_t) du/dy = u_tau^2 tau(y) / tau_w with tau / tau_w = 1 - y/delta
(channel) or 1 (Couette); k and eps are marched in pseudo-time to steady
state with implicit diffusion. Prints, over the window VAL-016 used
(y+ 30 to 0.2 of the half height in wall units), k / (u_tau^2 / sqrt(C_mu))
and the least-squares slope of u+ against ln y+ over 1 / kappa_m.
Usage: python oned_keps.py
"""

import json
import math
from pathlib import Path

import numpy as np

C_MU, C1, C2, SK, SE = 0.09, 1.44, 1.92, 1.0, 1.3
KAPPA_WF = 0.41
KAPPA_M = math.sqrt((C2 - C1) * SE * math.sqrt(C_MU))
NU = 1.508e-5


def solve(kind: str, half: float, u_tau: float, y0_plus: float = 30.0, n: int = 400) -> dict:
    """Steady k and eps across [y0, half] (channel) or [y0, 2 half - y0] (Couette)."""
    y0 = y0_plus * NU / u_tau
    top = half if kind == "channel" else 2.0 * half - y0
    # Geometric points clustered toward y0 (and toward the far wall for Couette).
    s = np.linspace(0.0, 1.0, n)
    if kind == "channel":
        y = y0 + (top - y0) * (np.exp(3.0 * s) - 1.0) / (math.exp(3.0) - 1.0)
    else:
        t = 0.5 * (1.0 - np.cos(np.pi * s))
        y = y0 + (top - y0) * t
    stress = (1.0 - y / half) if kind == "channel" else np.ones_like(y)
    k = np.full(n, u_tau**2 / math.sqrt(C_MU))
    eps = u_tau**3 / (KAPPA_WF * np.minimum(y, (2.0 * half - y) if kind == "couette" else y))
    k_wall = u_tau**2 / math.sqrt(C_MU)
    for _ in range(200000):
        nut = C_MU * k**2 / eps
        dudy = u_tau**2 * stress / (NU + nut)
        prod = nut * dudy**2
        k_new = _implicit(y, k, NU + nut / SK, prod, eps / k, k_wall, kind)
        e_bc = u_tau**3 / (KAPPA_WF * y0)
        eps_new = _implicit(y, eps, NU + nut / SE, C1 * eps / k * prod, C2 * eps / k, e_bc, kind)
        change = max(np.max(np.abs(k_new - k) / k), np.max(np.abs(eps_new - eps) / eps))
        k, eps = k_new, eps_new
        if change < 1e-11:
            break
    nut = C_MU * k**2 / eps
    dudy = u_tau**2 * stress / (NU + nut)
    u = u_tau * (math.log(9.793 * y0_plus) / KAPPA_WF) + np.concatenate(
        ([0.0], np.cumsum(0.5 * (dudy[1:] + dudy[:-1]) * np.diff(y)))
    )
    yp, up = y * u_tau / NU, u / u_tau
    window = (yp >= 30.0) & (y <= 0.2 * half)
    slope = np.polyfit(np.log(yp[window]), up[window], 1)[0]
    kr = k[window] / (u_tau**2 / math.sqrt(C_MU))
    core = (y > 0.2 * half) & (y < (half if kind == "channel" else 1.8 * half))
    return {
        "kind": kind, "half": half, "u_tau": u_tau, "re_tau": u_tau * half / NU,
        "window_y_over_half": [float(y[window][0] / half), float(y[window][-1] / half)],
        "window_points": int(window.sum()),
        "slope_over_1_kappa_m": float(slope * KAPPA_M),
        "k_ratio_window": [float(kr.min()), float(kr.max())],
        "k_ratio_core": [float(k[core].min() / (u_tau**2 / math.sqrt(C_MU))),
                         float(k[core].max() / (u_tau**2 / math.sqrt(C_MU)))],
        "iterations_change": float(change),
    }


def _implicit(y, phi, gamma, source, decay, wall_value, kind):  # type: ignore[no-untyped-def]
    """One backward Euler pseudo-time step, large step, decay in the diagonal."""
    n = y.size
    dt = 1e3
    g = 0.5 * (gamma[1:] + gamma[:-1]) / np.diff(y)  # face conductances
    vol = np.empty(n)
    vol[1:-1] = 0.5 * (y[2:] - y[:-2])
    vol[0] = vol[-1] = 0.5 * (y[1] - y[0])
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    rhs = phi * vol / dt + source * vol
    diag = vol / dt + decay * vol
    a[1:] = -g
    c[:-1] = -g
    diag[1:] += g
    diag[:-1] += g
    # Dirichlet at the first node; at the far end symmetry (channel) or Dirichlet (Couette).
    diag[0], c[0], rhs[0] = 1.0, 0.0, wall_value
    if kind == "couette":
        diag[-1], a[-1], rhs[-1] = 1.0, 0.0, wall_value
    # Thomas algorithm.
    cp = np.zeros(n)
    dp = np.zeros(n)
    cp[0] = c[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (rhs[i] - a[i] * dp[i - 1]) / m
    out = np.zeros(n)
    out[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        out[i] = dp[i] - cp[i] * out[i + 1]
    return out


def main() -> None:
    results = []
    # ADR-012's channel: H 0.3 m, so the half height is 0.15 m; Dean gives u_tau 0.073 m/s.
    for u_tau in (0.073, 0.3):
        results.append(solve("channel", 0.15, u_tau))
    # Plane Couette flow over the same gap, at two friction velocities.
    for u_tau in (0.073, 0.3):
        results.append(solve("couette", 0.15, u_tau))
    text = json.dumps(results, indent=1)
    (Path(__file__).parent / "oned_keps.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
```

## Appendix J: calc33b.py

The corrected arithmetic ADR-012 cites as its source 14.

```python
"""Builder arithmetic, prompt 33b: the corrected numbers ADR-012 quotes."""

import json
import math
from pathlib import Path

C_MU, C2 = 0.09, 1.92
NU = 1.81e-5 / 1.2
U, DX = 0.45, 0.04
out: dict[str, object] = {}

# B2: the eddy viscosity in the design's own convention, eps = k^1.5 / l_e
# (Nielsen 1990, eq. 6), so nu_t = C_mu k^2 / eps = C_mu k^0.5 l_e.
table = {}
for intensity in (0.02, 0.05, 0.1):
    k = 1.5 * (intensity * U) ** 2
    for length in (0.05, 0.1, 0.3):
        nut = C_MU * math.sqrt(k) * length
        # Decay from the ceiling to the obstacle tops, 1.0 m at the supply speed:
        # nu_t ~ (1 + t / T)^(1 - n), n = 1 / (C_2 - 1), T = n l_e / sqrt(k).
        n = 1.0 / (C2 - 1.0)
        t_top = 1.0 / U
        decay = (1.0 + t_top / (n * length / math.sqrt(k))) ** (1.0 - n)
        table[f"I={intensity:g}, l_e={length:g}"] = {
            "nu_t": nut,
            "nu_t_at_obstacle_tops": nut * decay,
            "cell_Pe_with_nu": U * DX / (NU + nut),
            "cell_Pe_nu_t_alone": U * DX / nut,
        }
out["core_eddy_viscosity_nielsen_convention"] = table
out["factor_between_conventions"] = 1.0 / C_MU**0.75

# S1: the scalable floor consistent with kappa 0.41 and E 9.793, where the
# linear law u+ = y+ meets the log law u+ = ln(E y+) / kappa.
kappa, e_wall = 0.41, 9.793
y = 11.0
for _ in range(100):
    y = math.log(e_wall * y) / kappa
out["scalable_floor_kappa041_E9793"] = y
out["log_law_at_11.06"] = math.log(e_wall * 11.06) / kappa

# S9: y+ at the first node next to the outlet flows, 1 m/s over 2 m.
re_x = 1.0 * 2.0 / NU
u_tau = 1.0 * math.sqrt(0.0592 * re_x**-0.2 / 2.0)
out["outlet_flow_u_tau"] = u_tau
out["outlet_flow_y_plus"] = u_tau * 0.02 / NU

# B2: the residual's reference velocity on 200x75 and what 6e-4 means in m/s.
out["row8_step_m_per_s"] = 6e-4 * 79.2

(Path(__file__).parent / "calc33b.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
```

## Appendix K: table33b.py and the launchers

`table33b.py` gives section 8's ladder states, from a run's history or, for a run stopped by
cost before it wrote one, from its log. The three launchers started the runs.

```python
"""Section 8's tables, from the outlet probe's histories or, for a run stopped by
cost before it wrote one, from its log, which prints every 25th outer iteration."""

import json
import re
from pathlib import Path

HERE = Path(__file__).parent
LINE = re.compile(
    r"it\s+(\d+) res (\S+) step (\S+) sweeps\s+(\d+) max\|U\| (\S+) at \(([^)]*)\) "
    r"inflow \[([^\]]*)\] closed \[([^\]]*)\]"
)


def record(name: str) -> dict:
    """The run's state: from NAME.json when written, else from NAME.log."""
    js = HERE / f"{name}.json"
    if js.exists():
        r = json.loads(js.read_text())
        n = len(r["residual"])
        return {
            "outer": n, "stop": r["stop"], "res": r["residual"][-1],
            "least": min(r["residual"]), "speed": r["max_speed"][-1],
            "at": tuple(r["at"][-1]),
            "most_in": [max(c) for c in zip(*r["inflow"])],
            "first_in": next((i for i, c in enumerate(r["inflow"]) if sum(c)), None),
        }
    rows = [LINE.search(line) for line in (HERE / f"{name}.log").read_text().splitlines()]
    rows = [m for m in rows if m]
    last = rows[-1]
    inflows = [[int(x) for x in m.group(7).split(",")] for m in rows]
    return {
        "outer": int(last.group(1)) + 1, "stop": "cut by cost (log)",
        "res": float(last.group(2)), "least": min(float(m.group(2)) for m in rows),
        "speed": float(last.group(5)), "at": tuple(float(v) for v in last.group(6).split(",")),
        "most_in": [max(c) for c in zip(*inflows)],
        "first_in": next((int(m.group(1)) for m, c in zip(rows, inflows) if sum(c)), None),
    }


def main() -> None:
    names = sorted(p.stem for p in HERE.glob("[DLRH]*.log"))
    for name in names:
        r = record(name)
        print(
            f"{name:16s} stop {r['stop']:24s} outer {r['outer']:5d} res {r['res']:.2e} "
            f"least {r['least']:.2e} speed {r['speed']:.3g} at {r['at']} "
            f"most reversed {r['most_in']} first reversed at {r['first_in']}"
        )


if __name__ == "__main__":
    main()
```

`run_ladder.sh`:

```sh
#!/bin/sh
# Prompt 33b: the real-room diagnostic and the 40x15 outlet ladder, in parallel.
PY=../../.venv/Scripts/python
$PY outlet33b.py D1_committed 200 75 1 500 0 0 0 T0 > D1_committed.log 2>&1 &
$PY outlet33b.py D2_cap5000 200 75 1 500 0.5 5000 1e-8 T0 > D2_cap5000.log 2>&1 &
for m in 1000 100 10 1; do
  for t in T0 T1 T2 T3; do
    $PY outlet33b.py L${m}_$t 40 15 $m 3000 0.5 40000 1e-8 $t > L${m}_$t.log 2>&1 &
  done
done
wait
```

`run_real.sh`:

```sh
#!/bin/sh
# Prompt 33b: the real room, 200x75 real air, at D2's pressure settings, with T3 and T2.
PY=../../.venv/Scripts/python
$PY outlet33b.py R_T3 200 75 1 2000 0.5 5000 1e-8 T3 > R_T3.log 2>&1 &
$PY outlet33b.py R_T2 200 75 1 2000 0.5 5000 1e-8 T2 > R_T2.log 2>&1 &
wait
```

`run_sens.sh`:

```sh
#!/bin/sh
# Prompt 33b: the hood face-velocity sensitivity pair under T3, at Re 8,950 and real air.
PY=../../.venv/Scripts/python
for m in 10 1; do
  for h in 0.4 0.6; do
    $PY outlet33b.py H${m}_T3_$h 40 15 $m 3000 0.5 40000 1e-8 T3 $h > H${m}_T3_$h.log 2>&1 &
  done
done
wait
```
