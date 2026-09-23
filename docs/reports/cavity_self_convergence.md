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
