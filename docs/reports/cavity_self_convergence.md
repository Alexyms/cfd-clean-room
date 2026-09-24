# Lid-driven cavity: each solver's order measured against itself

**Date:** 2026-09-23
**Context:** why the staggered solver's distance to Ghia et al. (reference
`ghia_1982_re100_r2`) falls by only about 1.45 per halving of h
**Instrument:** `scripts/self_convergence.py`, fields saved under `results/self_convergence/`
(gitignored). Every number below comes from that script's `summary.json` unless stated.
Provenance: MEASURED is read off the instrument, INFERRED follows from measured values by a
stated argument, HYPOTHESISED is neither, and UNKNOWN names what would settle it.

## 1. The control

The whole pipeline (2x2-block restriction, differencing, max and L2 norms, every
functional) was run first on synthetic fields F + C h^q G at the cell centers of the same
three grids, with C = 1 and known q. MEASURED:

| q | worst abs(p - q) over 14 estimates |
|---|---|
| 2 | 0.0139 |
| 1 | 0.0112 |
| 0.5 | 0.0215 |

The 14 estimates are u, v and p in both norms, the centerline extrema and the whole-domain
and regional kinetic energies. All are within the 0.05 tolerance, so the instrument
reports 1 and 0.5 when they are true, not only 2.

The synthetic pair is chosen so that the contamination terms the pipeline itself carries
stay below the tolerance:

- the restriction's own O(h^2) term, h_f^2/8 times the Laplacian of F;
- the quadratic term of the energy;
- the shift of an extremum under the error.

F has low curvature (amplitude 0.1) and an offset of 4 so that the energy is dominated by
the cross term. G is flat across each centerline pair, so the extrema respond linearly.
Centerline extrema are refined by the parabola through three samples, because a bare sample
extremum carries an O(h^2) error that is not a power law in h.

## 2. The six solves

Committed case settings, uniform meshes. MEASURED:

| Solver | 20x20 | 40x40 | 80x80 |
|---|---|---|---|
| collocated, outer iterations / s | 380 / 14 | 1571 / 98 | 5288 / 604 |
| staggered, outer iterations / s | 629 / 13 | 1891 / 60 | 5728 / 327 |

Every outer count equals the stored harness row for the same case and method.

## 3. Observed orders

d1 = R(f40) - f20 and d2 = R(f80) - f40, compared on the 20x20 grid; p = log2(||d1|| /
||R(d2)||). Pressure has its domain mean removed, because each grid pins it at a different
point. Functionals: p = log2((F20 - F40) / (F40 - F80)). MEASURED:

| Estimate | collocated | staggered |
|---|---|---|
| u, max / L2 | 0.60 / 1.07 | 1.58 / 1.79 |
| v, max / L2 | 0.66 / 1.05 | 1.07 / 1.65 |
| p, max / L2 | 0.43 / 0.70 | 0.65 / 0.60 |
| min u on x = 0.5 | 1.14 | 2.26 |
| max v on y = 0.5 | 0.79 | 2.80 |
| min v on y = 0.5 | 1.05 | 3.09 |
| kinetic energy, whole domain | 0.09 | 1.78 |
| kinetic energy inside [0.05, 0.95]^2 | 0.49 | 2.11 |
| kinetic energy in the 0.05 wall strip | 2.08 | 1.44 |
| kinetic energy inside [0.1, 0.9]^2 | 0.67 | 2.27 |
| kinetic energy in the 0.1 wall strip | not monotone | 1.54 |

No difference grows under refinement: ||R(d2)|| < ||d1|| for every field and norm of both
solvers.

**The whole-domain kinetic energy is not a valid order instrument here.** For the
collocated solver the interior energy rises with refinement and the wall-strip energy
falls. Their sum changes by an almost constant amount, 0.0034 then 0.0032, and returns an
order near zero that belongs to neither part. The regional energies are the result.

The estimates for one solver disagree with each other by more than 0.5 in order: 0.6 to
3.1 for the staggered solver, 0.1 to 2.1 for the collocated one. For the staggered solver
the split is systematic. The quantities away from the lid corners (centerline extrema,
interior energy) converge at 2.1 to 3.1, and the field norms, which the corners dominate
(section 4), at 0.6 to 1.8.

## 4. Where the difference lives

The share of the squared L2 norm of d2 on the 40x40 grid. The regions are:

- the four cells nearest each top corner (the 2x2 blocks);
- the first two cells along each wall, without those blocks;
- the interior.

The band is also split by wall, with the side walls halved at y = 0.5, and one more
region, cells within 0.15 of a top corner, cuts across the others. A difference spread
evenly would give the uniform column. MEASURED:

| Region | uniform | coll. u | coll. v | coll. p | stag. u | stag. v | stag. p |
|---|---|---|---|---|---|---|---|
| top corner blocks | 0.005 | 0.340 | 0.060 | 0.852 | 0.369 | 0.446 | 0.989 |
| wall band | 0.185 | 0.202 | 0.136 | 0.063 | 0.445 | 0.330 | 0.008 |
| interior | 0.810 | 0.458 | 0.803 | 0.085 | 0.186 | 0.224 | 0.004 |
| band: lid | 0.045 | 0.125 | 0.008 | 0.025 | 0.389 | 0.051 | 0.002 |
| band: side walls, upper half | 0.045 | 0.076 | 0.127 | 0.032 | 0.055 | 0.279 | 0.005 |
| band: side walls, lower half | 0.045 | 0.000 | 0.001 | 0.002 | 0.000 | 0.000 | 0.000 |
| band: floor | 0.050 | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 |
| within 0.15 of a top corner | 0.035 | 0.615 | 0.223 | 0.916 | 0.833 | 0.839 | 0.998 |

![staggered solver, |d2|](cavity_self_convergence_staggered-jacobi.png)

![collocated solver, |d2|](cavity_self_convergence_collocated-jacobi.png)

The staggered solver's difference is 83% to 100% within 0.15 of the two top corners, a
region holding 3.5% of the cells, and nothing along the floor or the lower side walls. The
collocated solver's v difference is spread through the interior at the uniform share.

## 5. The distance to Ghia, taken apart

**Erratum, 2026-09-23:** the half-cell offset this section describes is fixed under a new
metric name, `max_normalized_centerline_error_r2`; see section 9. The text below is
unchanged.

`validation/metrics.py` takes the u profile from cell column nx // 2 and the v profile from
row ny // 2. Both centers sit at 0.5 + h/2, half a cell off the centerlines Ghia's tables
describe, so every stored cavity error carries a positional error of order h.

The staggered solver stores u on the face line x = 0.5 and v on y = 0.5. Its faces were
recovered exactly from the saved cell means, because each cell value is the mean of its two
faces and the wall face is zero: the residual left on the far wall is below 1e-15. The same
errors were then taken on the centerlines, everything else as the metric does it. The
collocated solver has no faces, so its centerline profile is the mean of the two cells
either side. MEASURED:

| Solver | Grid | metric u | on centerline u | metric v | on centerline v |
|---|---|---|---|---|---|
| staggered | 20x20 | 0.01632 | 0.00824 | 0.02936 | 0.00624 |
| staggered | 40x40 | 0.01113 | 0.00429 | 0.02238 | 0.00764 |
| staggered | 80x80 | 0.00770 | 0.00409 | 0.01536 | 0.00802 |
| collocated | 20x20 | 0.10602 | 0.10981 | 0.14622 | 0.15557 |
| collocated | 40x40 | 0.04077 | 0.04408 | 0.05114 | 0.06061 |
| collocated | 80x80 | 0.01342 | 0.01394 | 0.00981 | 0.01555 |

For the staggered solver the offset adds 0.023, 0.015 and 0.007 to the v error, roughly
halving with h, so it is the larger part of the metric's value at 20x20 and 40x40 and half
of it at 80x80. What remains on the centerline is a floor. From 40x40 to 80x80 it moves
from 0.0043 to 0.0041 in u and from 0.0076 to 0.0080 in v, while the solver's own centerline
extrema change by 0.0007 to 0.0016 over the same step. The floor sits at the same stations
on both grids: v more negative than Ghia by 0.005 to 0.008 at x = 0.8047, 0.8594 and
0.9063, the descending jet by the right wall, and u high by 0.004 at y = 0.8516. It does
not shrink fourfold from 40x40 to 80x80, so it is not the metric's linear interpolation
between samples.

The offset happens to flatter the collocated solver. On the centerline its 80x80 errors
are u 0.0139 and v 0.0156, both still below 2%.

## 6. The collocated solver is pre-asymptotic on these grids

The collocated momentum equation uses the hybrid scheme: a face is upwind when
abs(F) / D >= 2, where F / D = rho abs(u_face) h / mu and u_face is the two-cell mean, as
the solver forms it. From the saved converged fields, MEASURED:

| Grid | x-faces upwind | y-faces upwind | lowest upwind cell row, y |
|---|---|---|---|
| 20x20 | 8.2% | 0 | 0.925 (top two rows) |
| 40x40 | 2.2% | 0 | 0.9875 (lid row) |
| 80x80 | 0 | 0 | none |

![collocated upwind cells](cavity_self_convergence_upwind.png)

The switch region changes between 40x40 and 80x80, from the lid row to nothing, so the
three grids mix a first-order and a second-order treatment of the layer that drives the
flow. INFERRED: a three-grid estimate straddling that change is not an asymptotic order.
That reconciles the collocated solver's low self-convergence (0.4 to 1.1) with its distance
to Ghia falling about three times per halving. The same regional energy, measured against
the staggered solver's interior value, converges at 0.9 then 1.4, rising as the switch
region shrinks. The collocated ghost-cell walls, O(h) at the wall (ADR-008), are a second
first-order source that this measurement does not separate (HYPOTHESISED).

## 7. The candidates, for the staggered solver

1. **Lid-corner singularity.** SUPPORTED for the field norms, MEASURED: 83% to 100% of the
   squared grid-to-grid difference lies within 0.15 of the top corners, which is why u, v
   and p converge at 0.6 to 1.8 in the norms. It is not what slows the approach to Ghia.
   The comparison samples the centerlines, away from the corners, and the centerline
   extrema converge at 2.3 to 3.1.
2. **Wall boundary treatment.** REJECTED, MEASURED: the floor and the lower halves of the
   side walls carry 0.000 of the squared difference against a uniform share of 0.095. The
   band's share is the corner neighbourhood: the lid band and the upper side walls.
3. **Interior scheme.** REJECTED, MEASURED: the interior carries 0.19 to 0.22 against a
   uniform 0.81, and the interior energy converges at 2.11 and 2.27.
4. **The reference floor.** SUPPORTED in part. MEASURED: on the true centerline the
   staggered solver's distance to Ghia stops falling between 40x40 and 80x80, at 0.004 in u
   and 0.008 in v, while its own centerline values change by about 0.001. UNKNOWN: whether
   that floor is Ghia's error or a consistent error in this solver's limit. Self-convergence
   cannot tell a scheme converging to the wrong answer from one converging to the right
   one. Two observations would settle it: an independent high-accuracy Re = 100 solution
   compared at x = 0.8047 to 0.9063 and y = 0.8516, or a 160x160 staggered solve showing
   whether the gap there stays fixed.
5. **OTHER: the metric's half-cell offset.** SUPPORTED, MEASURED: the metric samples at
   0.5 + h/2, which adds a first-order error, 0.023, 0.015 and 0.007 in the staggered v
   error. INFERRED: together with the floor it accounts for the metric's apparent order of
   about 0.5, a first-order term shrinking onto a constant.

**The fix the evidence points at, described and not made.** `validation/metrics.py` should
take the profiles on x = 0.5 and y = 0.5, from the faces where a solver has them and from
the two-cell mean where it does not. That changes every stored cavity error, so it wants a
new metric name, as the Ghia table correction took a new reference name, and a change of
its own.

## 8. How each number was taken

- **Control, orders, functionals, shares, maps, upwind faces and centerline errors:**
  `python scripts/self_convergence.py` on this branch, which runs the control, then solves
  or loads the six fields, then analyses them.
- **Maps:** images of abs(d2) on the 40x40 grid, one per solver, and the upwind-cell map for
  the collocated solver.
- **The collocated energy measured against the staggered interior value (section 6):** a
  hand calculation from the `ke_inside_0.05` values. The staggered 80x80 value extrapolated
  with its own order of 2.11 is 0.0202, against which the collocated errors are 0.0107,
  0.0057 and 0.0022.
- **The stations of the floor (section 5):** a one-off evaluation of the same centerline
  errors per Ghia station, not committed.

## 9. The true centerline, and whether the floor is Ghia's

**Date:** 2026-09-23, revised the same day after test 21 (section 9.3). Every number in this
section comes from `python scripts/self_convergence.py --extrapolate` or from the tests
named below. `--extrapolate` reads saved fields only: the six at the case's
`convergence_tol` of 1e-6, and the three staggered fields that
`python scripts/self_convergence.py --solve-tight` continued to 1e-9 (section 9.2).

### 9.1 The metric on the true centerline

`validation.metrics.cavity_true_centerline_errors` reports the metric
`max_normalized_centerline_error_r2` against the same reference, `ghia_1982_re100_r2`. Its
profiles interpolate linearly in x between the two cell columns whose centers bracket
x = 0.5, and in y between the two rows bracketing y = 0.5. On an even uniform grid that is
the mean of the two middle columns, and on an odd grid it is the middle column. As before,
only FLUID cells are sampled and the wall values are appended, so the only change is the
sampling position. The midlines and the wall positions both come from the mesh. The old
function and its name are unchanged, and the stored rows keep their meaning. The harness,
the viewer and VAL-002 now use the new metric, and a test guards each of the three
switches.

One property of both metrics is easy to miss. The cavity's outer ring of cells is typed
BOUNDARY, not FLUID, so both skip the wall-adjacent cell and interpolate straight from the
second cell to the wall value. At 40x40 the largest new-metric errors sit at interior
stations (y = 0.8516, x = 0.8594), so this does not set the metric's value there. MEASURED.

The tests that show the offset is gone, MEASURED:

- **Linear field, with a control.** For u = A + B x and v = A + B y on a 16x16 grid, the
  new sampling reads A + B/2 to 1e-14. The old one misses by B h/2, 0.025 here, to 1e-14
  (`tests/test_validation.py`, `TestTrueCenterlineMetric`).
- **Odd and stretched grids.** At 21x21 the new sampling equals the middle column exactly,
  and equals the old one. On meshes stretched at ratio 1.1, at 15 and 16 cells, the linear
  field reads A + B/2 to 1e-14. The mesh module stretches symmetrically, so its middle pair
  always weighs one half. A separate check on an unequal pair gives the weight from the
  distances, 0.2.
- **Agreement with the staggered faces.** `centerline_faces` recovers the staggered
  solver's faces on x = 0.5 and y = 0.5 exactly. The largest gap between the new metric's
  profile and those faces is below.

| Grid | gap in u | gap in v |
|---|---|---|
| 20x20 | 2.263e-3 | 2.528e-3 |
| 40x40 | 6.094e-4 | 6.249e-4 |
| 80x80 | 1.538e-4 | 1.557e-4 |

The gap's orders are 1.89 then 1.99 in u and 2.02 then 2.01 in v. The same gap for the old
sampling runs at 0.81 then 0.93 in u and 1.00 then 1.01 in v, which is the first-order
control. At 40x40 the largest gap is at y = 0.9125 in u and x = 0.9125 in v. On smooth
synthetic faces the two orders are 2.00 and 0.97 to 0.99 (`tests/test_self_convergence.py`).

The metric on the saved fields (`convergence_tol` 1e-6), MEASURED. The first two columns
are the stored metric, repeated from section 5:

| Solver | Grid | offset u | offset v | true centerline u | true centerline v |
|---|---|---|---|---|---|
| staggered | 20x20 | 0.01632 | 0.02936 | 0.00894 | 0.00656 |
| staggered | 40x40 | 0.01113 | 0.02238 | 0.00376 | 0.00803 |
| staggered | 80x80 | 0.00770 | 0.01536 | 0.00396 | 0.00813 |
| collocated | 20x20 | 0.10602 | 0.14622 | 0.10981 | 0.15557 |
| collocated | 40x40 | 0.04077 | 0.05114 | 0.04408 | 0.06061 |
| collocated | 80x80 | 0.01342 | 0.00981 | 0.01394 | 0.01555 |

The collocated values equal section 5's centerline columns, which used the same two-cell
mean. The staggered values differ from section 5's by no more than the second-order face
gap above.

**The staggered series is not monotone, which bears on ECR-001 criterion 3a.** MEASURED,
true-centerline metric, staggered solver at 20x20, 40x40 and 80x80:

| Fields | u | v |
|---|---|---|
| saved, `convergence_tol` 1e-6 | 0.00894, 0.00376, 0.00396 | 0.00656, 0.00803, 0.00813 |
| continued to 1e-9 (section 9.2) | 0.00890, 0.00399, 0.00481 | 0.00649, 0.00825, 0.00894 |

On both, u falls and then rises, and v rises at each refinement. Criterion 3a requires the
u and v errors each to decrease monotonically across 20x20, 40x40 and 80x80. From step 8
the harness scores VAL-002 with this metric, so on these fields the staggered solver does
not meet 3a as written. The collocated series falls in both components on the saved
fields; its iteration error was not measured. ECR-001 is not edited here.

VAL-002 was run under the new metric and still xfails. The collocated solver at 40x40 has u
0.04408 and v 0.06061, both above 2%, as they were under the old metric.

The old and new metrics never disagree by more than the offset explains. On an even
uniform grid their difference at each row is exactly half the step between the two middle
columns. The largest station difference halves with h for the staggered solver: 0.0139,
0.0074 and 0.0037 in u, and 0.0271, 0.0144 and 0.0072 in v. For the collocated solver the v
difference does not halve from 20x20 to 40x40 (0.0115, 0.0110, 0.0067). Its gradient across
y = 0.5 at 20x20 differs from the 80x80 gradient by as much as the offset term itself,
which fits section 6's finding that it is not yet asymptotic.

### 9.2 Iteration error in the saved fields

The solves stop when the largest velocity change per outer iteration, over the reference
velocity, falls below `convergence_tol`, 1e-6 for the cavity. That is not a bound on the
distance to the converged answer. `--solve-tight` continues each staggered solve to 1e-9,
with the tolerance and the iteration cap set in memory and the case file unchanged. It
keeps u and v the first time the residual falls below 1e-6, 1e-7, 1e-8 and 1e-9.

**The control:** at every grid the 1e-6 snapshot is bitwise identical to the saved field,
so the tight solve is the saved computation continued. The script stops if it is not.

MEASURED:

| Grid | outer iterations at 1e-6 / 1e-7 / 1e-8 / 1e-9 | largest change in u, 1e-6 to 1e-9 | in v | wall time to 1e-9 |
|---|---|---|---|---|
| 20x20 | 629 / 809 / 989 / 1243 | 7.77e-5 | 7.66e-5 | 17 s |
| 40x40 | 1891 / 2525 / 3158 / 3856 | 2.69e-4 | 2.74e-4 | 75 s |
| 80x80 | 5728 / 8083 / 10437 / 12840 | 9.94e-4 | 1.02e-3 | 465 s |

The iteration error left at 1e-6 grows 3.5 to 3.7 times per halving of h and reaches about
1e-3 at 80x80, the same size as the grid-to-grid steps near the floor stations. Test 21
measured the same values independently (`docs/prompts/test-21.md`, C20), and its 1e-8 and
1e-9 fields are bitwise identical to these. Sections 1 to 8 were computed from the 1e-6
fields and are left as written. Test 21, C23, reran section 3's orders on the converged
fields and found that the functional orders above 2 fall toward 2.

### 9.3 Pointwise extrapolation of the staggered profile

**First reading, withdrawn 2026-09-23.** Part B was first run on the saved 1e-6 fields. It
found 2 of 30 stations second order, and 9 whose steps reversed, and concluded that
extrapolation was not licensed and that three grids had not reached the asymptotic range.
Test 21 (`docs/prompts/test-21.md`, C20) showed that conclusion comes from iteration error.
Every reversed step was 2.4e-4 or smaller, below the 40x40 fields' own iteration error. On
the same three grids converged further, nothing reverses. That reading also said the
cubic and quintic interpolations license the same stations. On the 1e-6 fields they
license 2 and 1. The reading is withdrawn. What follows replaces it, on the 1e-9 fields.

**Method.** The profiles are the staggered solver's exact faces on the true centerlines at
20, 40 and 80, every row including the wall-adjacent cells, with the wall values appended
at the mesh's walls and the whole profile divided by the lid speed from the case. Each is
interpolated onto Ghia's stations by the cubic through the four nearest nodes, which errs
at O(h^4), two orders above the h^2 term being extrapolated. The same is done with the
quintic through six nodes. The difference between the two estimates the cubic's
interpolation error; it does not bound it. MEASURED on the 1e-9 fields, the difference in
value is at most:

- in u: 1.7e-3, 2.6e-4 and 2.6e-5 at 20, 40 and 80 (y = 0.9531, 0.9766, 0.9766);
- in v: 3.9e-4, 2.1e-5 and 1.1e-6 (x = 0.9063, 0.9688, 0.8594).

It moves the orders by up to 0.37 in u (y = 0.9766) and 0.46 in v (x = 0.9063), more than
the band's half-width. So near the lid and in the jet the classification depends on the
interpolation. Both are reported, and no order is undefined under one and not the other.

The order at each station is log2((f20 - f40) / (f40 - f80)), undefined where the two steps
have opposite signs. A station counts as second order when abs(p - 2) < 0.25. That band was
fixed before any field was read: inside it the Richardson factor 1 / (2^p - 1) stays within
27% of the 1/3 applied. Only there is the extrapolated value f80 + (f80 - f40) / 3 used.

**The control, which ran first.** MEASURED: a profile with a known cubic limit plus h^2
times a cubic returns order 2 and the limit to 1e-13. The same limit plus h times the cubic
returns order 1 at every station and licenses none. A profile shifted by 1e-6 from the
limit the control expects stops the script.

**The orders have settled.** MEASURED, stations second order under the cubic and the
quintic, and undefined orders, at each snapshot:

| Fields | second order, cubic | second order, quintic | undefined |
|---|---|---|---|
| 1e-6 | 2 / 30 | 1 / 30 | 9 / 30 |
| 1e-7 | 15 / 30 | 12 / 30 | 0 / 30 |
| 1e-8 | 16 / 30 | 17 / 30 | 0 / 30 |
| 1e-9 | 16 / 30 | 18 / 30 | 0 / 30 |

From 1e-8 to 1e-9 no station's cubic order moves by more than 0.039 in u (y = 0.0547) or
0.028 in v (x = 0.5000), and none becomes undefined or defined. These settling figures,
and `settling` in `extrapolation.json`, cover the cubic orders only. Under the quintic one
classification flips between 1e-8 and 1e-9, and it is the change from 17 to 18 in the
table: u at y = 0.9609, whose quintic order moves from 2.2510 to 2.2498 across the band
edge. The 2.25 printed for that station below is 2.2498, inside the band.

**The orders on the 1e-9 fields, MEASURED.** "yes" marks a second-order station. An
asterisk marks a station whose classification differs between the two interpolations. The
extrapolated value is given where that interpolation licenses it.

| y (u profile) | order, cubic | order, quintic | f80 - Ghia | f80 - f40 | extrapolated - Ghia, cubic | quintic |
|---|---|---|---|---|---|---|
| 0.9766 | 3.27 | 2.91 | +0.0021 | +0.00098 | - | - |
| 0.9688 | 2.59 | 2.48 | +0.0027 | +0.00169 | - | - |
| 0.9609 | 2.11 yes | 2.25 yes | +0.0025 | +0.00239 | +0.0033 | +0.0034 |
| 0.9531 | 2.07 yes | 2.14 yes | +0.0031 | +0.00258 | +0.0039 | +0.0040 |
| 0.8516 | 1.98 yes | 1.98 yes | +0.0045 | +0.00180 | +0.0051 | +0.0051 |
| 0.7344 | 2.05 yes | 1.98 yes | +0.0007 | +0.00056 | +0.0009 | +0.0009 |
| 0.6172 | 1.98 yes | 1.97 yes | -0.0022 | -0.00058 | -0.0024 | -0.0024 |
| 0.5000 | 2.04 yes | 2.03 yes | -0.0028 | -0.00157 | -0.0034 | -0.0034 |
| 0.4531 | 2.06 yes | 2.06 yes | -0.0025 | -0.00173 | -0.0031 | -0.0031 |
| 0.2813 | 2.18 yes | 2.18 yes | -0.0007 | -0.00115 | -0.0011 | -0.0011 |
| 0.1719 | 2.33 | 2.32 | -0.0001 | -0.00055 | - | - |
| 0.1016 | 2.49 | 2.55 | -0.0000 | -0.00025 | - | - |
| 0.0703 | 2.65 | 2.70 | +0.0011 | -0.00012 | - | - |
| 0.0625 | 2.52 | 2.73 | -0.0000 | -0.00009 | - | - |
| 0.0547 | 2.43 | 2.54 | -0.0001 | -0.00006 | - | - |

| x (v profile) | order, cubic | order, quintic | f80 - Ghia | f80 - f40 | extrapolated - Ghia, cubic | quintic |
|---|---|---|---|---|---|---|
| 0.9688 | 2.12 yes | 2.13 yes | -0.0031 | -0.00024 | -0.0032 | -0.0032 |
| 0.9609 | 2.64 | 2.36 | -0.0040 | -0.00018 | - | - |
| 0.9531 | 2.69 | 2.43 | -0.0047 | -0.00016 | - | - |
| 0.9453 | 2.64 | 2.27 | -0.0054 | -0.00015 | - | - |
| 0.9063 * | 2.49 | 2.03 yes | -0.0079 | -0.00021 | - | -0.0080 |
| 0.8594 | 2.42 | 2.33 | -0.0091 | -0.00065 | - | - |
| 0.8047 * | 2.26 | 2.24 yes | -0.0078 | -0.00150 | - | -0.0083 |
| 0.5000 | 3.07 | 3.02 | +0.0030 | +0.00004 | - | - |
| 0.2344 | 2.11 yes | 2.11 yes | +0.0038 | +0.00162 | +0.0043 | +0.0043 |
| 0.2266 | 2.11 yes | 2.11 yes | +0.0038 | +0.00162 | +0.0043 | +0.0043 |
| 0.1563 | 2.10 yes | 2.11 yes | +0.0036 | +0.00152 | +0.0041 | +0.0041 |
| 0.0938 | 2.13 yes | 2.19 yes | +0.0030 | +0.00104 | +0.0033 | +0.0033 |
| 0.0781 | 2.15 yes | 2.17 yes | +0.0026 | +0.00084 | +0.0029 | +0.0029 |
| 0.0703 | 2.09 yes | 2.14 yes | +0.0025 | +0.00072 | +0.0027 | +0.0027 |
| 0.0625 | 1.95 yes | 2.09 yes | +0.0023 | +0.00060 | +0.0025 | +0.0025 |

**The floor stations**, the four section 5 found, MEASURED. The order-2 value is
f80 + (f80 - f40) / 3 under the cubic, given whether or not the station is licensed:

| Station | order, cubic / quintic | f80 - Ghia | f80 - f40 | order-2 value - Ghia |
|---|---|---|---|---|
| u, y = 0.8516 | 1.98 / 1.98 | +0.0045 | +0.00180 | +0.0051 |
| v, x = 0.9063 | 2.49 / 2.03 | -0.0079 | -0.00021 | -0.0080 |
| v, x = 0.8594 | 2.42 / 2.33 | -0.0091 | -0.00065 | -0.0093 |
| v, x = 0.8047 | 2.26 / 2.24 | -0.0078 | -0.00150 | -0.0083 |

These agree with test 21's table to within 1.3e-5, which is inside its four-decimal
rounding.

**The answer, INFERRED.** At all four floor stations the order is between 1.98 and 2.49
under either interpolation. Nothing reverses, and each step from 40x40 to 80x80 has the
sign of the distance from Ghia. For any order in that range the Richardson correction is
between 0.22 and 0.34 of the last step, which is at most 6.1e-4 here. So the converged
staggered solution approaches values that differ from Ghia by about 0.5% of the lid speed
in u at y = 0.8516, and by 0.8 to 0.9% in v at the jet stations by the right wall, x =
0.8047 to 0.9063. Refinement moves the solution away from Ghia there, not toward it. At
the other licensed stations the extrapolated values differ from Ghia by 0.09% to 0.43%.

This does not say Ghia is wrong. Whether the gap is Ghia's or a systematic error of this
scheme cannot be settled by this solver's own refinement. It needs an independent
reference, and the planned one is Marchi, Suero and Araki (2009). Of the two observations
section 7, candidate 4 names, that is the one that remains. A 160x160 solve would add a
grid, but on converged fields the three existing grids already put 16 of the 30 stations
inside the band around order 2 (18 under the quintic), with the cubic orders from 1.95 to
3.27 and none reversed. Further refinement of this scheme cannot tell its own limit from
Ghia's.

### 9.4 How each number in this section was taken

- **Tight solves and their control:** `python scripts/self_convergence.py --solve-tight`,
  which writes `results/self_convergence/staggered-jacobi_<n>_tol1e-9.npz` and stops if the
  1e-6 snapshot differs from the saved field.
- **Metric values, face gaps, iteration error, station orders, interpolation estimates and
  settling:** `python scripts/self_convergence.py --extrapolate`, which writes
  `results/self_convergence/extrapolation.json`.
- **Gap orders:** `pytest tests/test_self_convergence.py -s`, which prints them when the
  saved fields are present.
- **VAL-002:** `pytest tests/test_lid_cavity.py -s`.
- **Old against new per station, and the 40x40 station locations:** one-off evaluations of
  the two profile functions on the saved fields, not committed.
- **Agreement with test 21:** a one-off comparison of these fields and floor values with
  `results/tester21/`, not committed.
