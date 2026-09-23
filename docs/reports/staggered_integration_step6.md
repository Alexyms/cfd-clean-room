# The staggered solver's first flow fields: ECR-001 step 6

**Date:** 2026-09-22
**Context:** ECR-001 step 6, `src/solver_staggered.py` run alongside the unchanged
collocated solver, before any validation test moves to it (steps 7 and 8)
**Instruments:** the benchmark harness for iterations, work, error and time; a one-off
script, not committed, for the quantities the harness does not record (section 7)

## 1. Per case, collocated against staggered

Same case files, same machine. The collocated figures are the stored rows in
`benchmarks/results.jsonl` (six per case, commits 28dc0c1 and 4063813), reproduced at this
branch's code commit 0e7f5b0 to every digit in accuracy, outer iterations, inner sweeps and
cell updates. The staggered figures are the three rows per case this branch appended,
`method` `staggered-jacobi`, commit 0e7f5b0, `git_dirty` false, one process on the
machine. Wall time is the median of the rows taken at one process.

| Case | Solver | Outer iterations | Cell updates | Error (unchanged metric) | Wall s |
|---|---|---|---|---|---|
| val001_80x40 | collocated | 551 | 2.859e9 | 0.02036 L2 | 151.7 |
| val001_80x40 | staggered | 570 | 3.186e9 | 0.00231 L2 | 104.9 |
| val002_20x20 | collocated | 380 | 6.181e7 | u 0.1060, v 0.1765 | 13.5 |
| val002_20x20 | staggered | 629 | 9.122e7 | u 0.0163, v 0.2209 | 14.5 |
| val002_40x40 | collocated | 1571 | 1.139e9 | u 0.0408, v 0.1893 | 86.7 |
| val002_40x40 | staggered | 1891 | 1.208e9 | u 0.0111, v 0.2223 | 65.6 |

Every staggered case converged within `max_simple_iter`. No residual history grows
persistently: the cavity trajectories fall at every sample, and the channel's rises twice
between iterations 40 and 70, by at most a factor of 1.2, then falls to the tolerance.
The cell updates are not the same count on the two solvers: the collocated one counts two
momentum sweeps and each pressure sweep over the FLUID cells, the staggered one the
unknown faces once and each pressure sweep over every cell with a pressure equation, the
BOUNDARY ring included (the `cell_update_definition` of each row).

On the channel the staggered error is 0.11 of the collocated one and below the 1% ECR-001
sets for VAL-001. In u on the cavity it is 0.15 of it at 20x20 and 0.27 at 40x40. In v on
the cavity it is worse on both grids and does not improve with refinement; section 4 shows
that this is the reference data, not the solver.

The staggered cavity needs more outer iterations than the collocated one, 1.7 times at
20x20 and 1.2 times at 40x40; the channel needs about the same. It also counts more cell
updates on every case, yet runs in less wall time on the channel and the 40x40 cavity.
The pressure solve is the cost: it averages 87% of the case file's sweep cap per outer
iteration on the channel, 72% and 79% on the two cavities, and takes 92% to 98% of the
staggered wall time (`inner_sweeps` and `time.stages` of the rows).

## 2. Continuity against acceptance criterion 6

Criterion 6 asks for a per-cell imbalance below 1e-10 and a domain sum below the same
bound. Measured on the field each staggered solve returned, `last_mass_imbalance`, which
is `PressureCorrector.mass_imbalance` of the returned face velocities.

| Case | max abs per-cell imbalance | Domain sum | Solver said converged |
|---|---|---|---|
| val001_80x40 | 2.31e-10 | 4.59e-07 | yes |
| val002_20x20 | 2.98e-09 | -2.81e-18 | yes |
| val002_40x40 | 8.19e-10 | 2.49e-18 | yes |

On the closed cavity the domain sum is rounding, as it is at every outer iteration of the
solve (`tests/test_solver_staggered.py`,
`test_closed_domain_imbalance_sums_to_zero_at_every_iteration`). On the channel the domain
sum is the net mass flux through the boundary, outflow minus inflow, 9.2e-6 of the inflow
of 5.0e-2.

The per-cell imbalance is above the bound on all three cases, by a factor of 2 to 30, and
the solver declared convergence on all three. This is the false-convergence risk the
capped inner solve allows: a capped correction makes a small step, and a rule that reads
the velocity change can come to rest before every cell conserves mass. The imbalance is
still falling with refinement and is four orders of magnitude below the collocated
solver's uniform residual source on the same grids (9.02e-5 and 6.46e-6,
`docs/reports/pressure_solver_probe.md`, table B). The stopping rule was left as the
collocated one so the outer counts compare; whether it needs a continuity term is step 7's
decision.

## 3. What the unchanged metric discards

`validation/metrics.py` samples FLUID cells only and appends the wall values. On the
staggered grid the BOUNDARY ring is ordinary solution cells, so the metric drops the
solver's nearest-wall data and interpolates linearly from the wall to the second cell
instead. Recomputed with the ring kept, everything else as the metric does it:

| Case | Component | Unchanged metric | Ring included |
|---|---|---|---|
| val001_80x40 | u, L2 | 0.002306 | 0.002316 |
| val002_20x20 | u | 0.01632 | 0.01472 |
| val002_20x20 | v | 0.2209 | 0.2209 |
| val002_40x40 | u | 0.01113 | 0.01113 |
| val002_40x40 | v | 0.2223 | 0.2223 |

The effect is small. Only the 20x20 u error moves, by a tenth. The cavity maxima sit in the
interior, where the ring does not enter; on the channel the ring adds two cells to the
thirty-eight the L2 sum already has. Decision 5 costs little on these grids.

## 4. The Ghia v reference in the metric is not Ghia's table

The v errors above, collocated and staggered, are measured against `GHIA_V_X` and
`GHIA_V_VAL` in `validation/metrics.py`, and those are not the Re = 100 values of Ghia,
Ghia and Shin (1982), Table II. Its sixteen stations are exactly the u table's y stations
(Table I) without 0.9766. Of its sixteen values, ten are Table II values but only five of
those sit at their Table II station (the two walls and the three stations nearest the
right wall); the other six, among them v(0.5) = -0.11477 where Table II gives +0.05454,
are not in Table II at all. It has no station at x = 0.2344 or 0.8047, where Table II's
extremes, +0.17527 and -0.24533, lie. The u table matches Table I exactly.

Mass conservation shows which is wrong without appeal to either source. The net vertical
flux through any horizontal line of a closed cavity is zero, so the v profile along
y = 0.5 must integrate to zero. By the trapezoidal rule over the table's own stations, the
metric's v table integrates to -0.0947; Table II integrates to -0.0004; the metric's u
table, by the same test along x = 0.5, to 0.0073. The Table II values used here are those
of a public transcription (https://gist.github.com/ivan-pi/caa6c6737d36a9140fbcf2ea59c78b3c).
No second transcription was checked; the conservation test does not depend on either.

Against Table II, with the metric's sampling otherwise unchanged:

| Case | Solver | v, stored reference | v, Table II |
|---|---|---|---|
| val002_20x20 | collocated | 0.1765 | 0.1462 |
| val002_20x20 | staggered | 0.2209 | 0.0294 |
| val002_40x40 | collocated | 0.1893 | 0.0511 |
| val002_40x40 | staggered | 0.2223 | 0.0224 |

Against the published profile the staggered v error is a fifth to a half of the
collocated one and falls with refinement, and the collocated v error falls too, from 0.146
to 0.051 between 20 and 40 cells. The ECR-001 problem statement's v series, 0.1765,
0.1893, 0.2072, rising under refinement, was measured against the stored table; so are
acceptance criterion 3a's baseline and the VAL-002 test and its xfail
(`tests/test_lid_cavity.py` uses `cavity_centerline_errors`). The metric is not changed here:
decision 5 keeps it the instrument the baseline rows were measured with, and
`validation/` is outside this step. Correcting it is a change of its own, and it moves
every stored v error at once.

## 5. The reference velocity on the channel

The stopping rule is the collocated one in definition, divided by `F_ref / (rho h)`. On the
cavity the two solvers' reference velocities are equal with `==`. On the channel they
cannot be: the collocated inlet flux is 4.75e-2, two corner cells short
(`docs/reports/inlet_flux_comparison.md`), and the staggered one is the exact 5.00e-2, so
the staggered reference velocity is 40/38 of the collocated one. Replaying the collocated
VAL-001 residual history with its residual scaled by 38/40, it would have stopped at 545
outer iterations instead of 551. That is the size of the difference the reference alone
makes on this case.

## 6. Other observations

- The corrector writes the pressure outlet faces, correcting them against p' = 0; only
  wall and inlet faces are never written. Following the momentum contract, the solver
  extrapolates each outlet face from its interior neighbour before every prediction, and
  the returned outlet faces are the corrected ones. After a solve every wall and inlet
  face still holds exactly what `apply_normal_velocity` wrote.
- `src/solver_staggered.py` imports `IterationState` from `src/solver_ns.py`, so the
  harness callback type is literally the same for both solvers. The class has to move when
  the collocated solver is retired.

## 7. How each number was taken

- Outer iterations, cell updates, errors under the unchanged metric and wall times: the
  harness rows, `python scripts/benchmark.py --method staggered-jacobi --repeats 3` on
  commit 0e7f5b0 with a clean tree, and the stored collocated rows. The collocated rows
  were reproduced at the same commit into a scratch file under `results/`, not into the
  results file.
- Per-cell imbalance and domain sum: `last_mass_imbalance` of a `StaggeredSolver` solve of
  each case, in a scratch script. The solve is deterministic and returned the harness rows'
  outer iteration counts and errors.
- Ring-inclusive errors: the same script, rebuilding the metric's profiles with every row
  or column of the centerline kept instead of the FLUID cells only. With the stored table
  and the ring excluded it reproduces the metric's value to the bit.
- Table II errors: the same construction with the published v table in place of the
  stored one; the collocated fields for this column were solved in the same script with
  `NavierStokesSolver`.
- The reference velocity replay: one collocated VAL-001 solve, recording
  `residual_history` and finding the first iteration at which 38/40 of it falls below
  `convergence_tol`.
