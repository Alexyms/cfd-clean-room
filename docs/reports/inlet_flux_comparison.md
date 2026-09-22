# Inlet flux on VAL-001: staggered against collocated

**Date:** 2026-09-21
**Context:** ECR-001 step 3, `src/boundary_staggered.py` against `src/boundary.py`
**Case:** `val001_80x40`, Poiseuille channel, inlet velocity 0.1 on the left edge, height 0.5

Three quantities that all claim to be the inlet volumetric flux of the same
configuration, per unit depth, rho = 1.

| Quantity | Source | Value |
|---|---|---|
| Exact, U times H | configuration | 5.0000e-02 |
| Staggered `get_inlet_flux("inlet")` | sum of face velocity times face width over the 40 left faces | 5.0000e-02 (to rounding) |
| Collocated `get_inlet_flux("inlet")` | sum over the left-edge BOUNDARY entries | 4.7500e-02 |

The collocated value is short by exactly two cells, 2 x 0.1 x dy. Its edge map
gives corner ring cells to the bottom and top edges (`_identify_edge` in
`src/boundary.py`), so the left inlet has 38 entries rather than 40 and the two
corner cells are no-slip walls. On the staggered layout every left face is a
storage location and the inlet spans the full edge. Neither function reads a
velocity field, so this difference is an accounting difference in what the two
layers call the inlet, not a measurement of mass crossing a wall. It is pinned
by `tests/test_boundary_staggered.py::TestSeededCases`.

## The leak, measured through the boundary faces

The measurement the comparison was expected to give comes from the converged
collocated field instead. Using the solver's own residual-monitoring face
interpolation (`_simple_face_fluxes`) on the VAL-001 solution at commit 262928b,
551 outer iterations, the mass flux through the faces that bound the collocated
fluid region (cells 1 to n-2 on each axis) is:

| Face | Flux |
|---|---|
| Inflow through x = dx | 4.7354e-02 |
| Outflow through x = L - dx | 4.8826e-02 |
| Through the bottom wall face, y = dy, into the fluid | 7.3547e-04 |
| Through the top wall face, y = H - dy, into the fluid | 7.3547e-04 |
| Net imbalance (out - in - walls) | 3.06e-07 |

The two walls each admit 7.35e-04, together 1.47e-03, which is 3.1 percent of
the inflow and is what the outflow exceeds the inflow by. This is the wall leak
of `docs/reports/pressure_solver_probe.md` seen a third way, through the faces
rather than through the divergence sum, on the open-domain case rather than the
cavity. The inflow itself, 4.7354e-02, is 0.3 percent below the 4.75e-02 the
ghost values prescribe, because the collocated face flux at x = dx averages the
ghost and first interior cells rather than reading the wall.

On the staggered grid the wall faces are storage locations set to zero exactly,
so both wall rows of this table are zero by construction (ECR-001 acceptance
criterion 6).

The script that produced the face-flux table is not committed; it solves the
case once with `NavierStokesSolver.solve_steady` and sums the four bounding
rows and columns of the flux arrays. The lid-driven cavity cases have no inlet
and both layers report zero flux for them.
