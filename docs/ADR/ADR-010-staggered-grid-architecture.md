# ADR-010: Solver Architecture V2: Staggered Grid, Non-Uniform Mesh, QUICK Advection

## Status
Accepted. Written 2026-09-30 at ECR-001 step 9, after steps 1 to 8 were merged, so it records
what was built rather than what was planned; revised 2026-10-01 for the fourth condition of the
stopping rule. Supersedes ADR-008. ECR-001 holds the decision to rebuild and its acceptance
criteria. Decision 4, the weighted Jacobi sweep, and REQ-S08's clarification of 2026-09-22 are
superseded by ADR-013 (ECR-003, accepted 2026-10-06): the pressure correction is solved by
Jacobi-preconditioned conjugate gradients and REQ-S08 is amended. The step 5 report's evidence
for the weight stands as history. Bracketed numbers point at the sources listed at the end.

## Context
The collocated SIMPLE solver (`src/solver_ns.py`, Rhie-Chow face fluxes, hybrid advection,
ghost-cell walls under ADR-008) let mass cross its walls. The ghost value `v_interior / 3` puts
a face flux of `(2/3) v_interior` through a no-slip wall. In the closed cavity the domain sum of
the mass imbalance settled at 2.90e-2, 9.23e-3 and 2.35e-3 at 20, 40 and 80 cells per side, and
the per-cell imbalance at 9.02e-5, 6.46e-6 and 3.88e-7, uniform over the domain [1]. The
pressure correction system was then singular with an incompatible right-hand side, and at
convergence the correction did nothing [1]. On the open channel the two walls admitted 1.47e-3
together, 3.1% of the inflow [2]. VAL-001 measured 2.04% on 80x40, against a 1% that ADR-008
relaxed to 2.5%.

ECR-001 also cited a 20% VAL-002 v error, measured against a corrupted Ghia v table and since
withdrawn (ECR-001 section 12). The wall leak stands on measurements that read no reference
data, and it is the reason for the layout.

## Decision

**1. Staggered (MAC) layout (REQ-S07).** p at cell centers [ny, nx], u on vertical faces
[ny, nx+1], v on horizontal faces [ny+1, nx] (`src/staggered.py`). Every wall face is a storage
location written to zero, so the discrete divergence of u* summed over a closed domain
telescopes to zero. Measured on the cavity at 20, 40 and 80: -8.7e-19, 2.2e-18 and -3.3e-19,
against the collocated 2.90e-2, 9.23e-3 and 2.35e-3 [3]. No compatibility correction is applied
anywhere. The public return stays cell-centered [ny, nx], each cell the plain mean of its two
faces, so every consumer of `solve_steady` reads the shape it read before. The staggered solver
was built alongside the collocated one (`src/solver_staggered.py`), so both run from one commit
and the harness compares them by its `--method` label.

**2. Direct Dirichlet imposition and the shared boundary registry (REQ-S12, REQ-S12.1).**
`src/boundary_staggered.py` writes the normal component exactly into its domain-face storage.
The tangential component has no storage location on the wall: its wall value and the
half-cell wall distance reach the momentum stencil as data, and the wall shear is
`(phi_P - phi_wall) / wall_distance`. Nothing is stored outside the domain. The inlet flux
is the exact sum over the inlet faces, 5.0000e-2 on VAL-001, where the collocated layer
prescribes 4.7500e-2 because its edge map gives the corner cells to the walls [2]. Which
segment covers a point, its type and its prescribed velocity are interpreted once, in
`src/boundary_registry.py`, and both layers read it, so the two cannot drift while both exist.
The wall stencil is second order: the fully developed discrete channel profile has L2 error
2.152e-3 and 5.574e-4 at ny = 20 and 40, observed order 1.95 [4, section 3].

**3. QUICK by deferred correction over an upwind implicit matrix (REQ-S09).** QUICK's
upstream coefficient is negative, which breaks the diagonal dominance a Jacobi sweep needs.
`src/momentum.py` assembles the implicit matrix with first-order upwind and carries the QUICK
minus upwind advective flux, evaluated on the current field, as an explicit source. At
convergence the solution is the QUICK one. The face value is the quadratic through the two
bounding nodes and the next upstream node, with Lagrange weights from the actual node
positions, so the stencil is exact on a stretched mesh. Where the far-upstream node is outside
the domain, Leonard's (1979) boundary form fits the quadratic through the boundary value at
its physical location. One under-relaxed Jacobi sweep per outer iteration; the collocated
probe found extra momentum sweeps cut its outer count by 1.8 times and then stopped helping
[5]. Measured order: against the solver itself, the cavity's centerline extrema and interior
energy converge at 2.1 to 3.1 on the saved fields, falling toward 2 on fields converged
further [6, sections 3 and 9.2]; VAL-001 at 1.992 without a reference [7], and VAL-002
against Marchi at 2.24 and 2.11 in u, 2.12 and 2.07 in v [8].

**4. Weighted Jacobi at w = 2/3 (REQ-S08, clarified 2026-09-22, not amended).** On a closed
domain every row has a_P equal to its neighbour sum and the grid is bipartite, so the
checkerboard is an exact eigenvector of the plain Jacobi update with eigenvalue -1 and never
decays. Undamped, no cavity correction reached `pressure_tol` in 200,000 sweeps [3, sections
2 and 3]. The weighted sweep `p_new = (1 - w) p_old + w J` maps -1 to 1 - 2w = -1/3 and keeps
the per-cell, previous-iterate update REQ-S08 asks for. The cavity then converges, in 2,563,
10,044 and 38,428 sweeps at 20, 40 and 80, quadrupling as h halves; the channel's single
correction rose from 126,277 to 183,134 sweeps, a factor of 1.450 against the 1.5 the weight
predicts for the slowest mode [3, section 5]. A weight of 0.95 needs about a quarter fewer
sweeps; 2/3, the textbook smoother weight, was kept as a module constant, not a configuration key.

**5. The `error_estimate` stopping rule (REQ-S01 and REQ-S04, clarified 2026-09-24 and
2026-09-30).** Built in `src/stopping.py`, opt-in by the solver key `stopping_rule`, and named
by both validation case files. The velocity-step rule stays the default and reproduces earlier
fields bitwise. The rule carries a version, `RULE_VERSION`, stored with every saved solve and
in the params of every `error_estimate` harness row; the four-condition rule is version 3. A
solve stops when all four hold, and reaching the cap is reported as not converged [4, sections
9 and 10]:

- (a) the velocity step times rho_hat / (1 - rho_hat), over the largest prescribed boundary
  velocity, is below `iteration_error_tol` (1e-6), rho_hat fitted over the last 100 steps. A
  small step is not a small error. At the velocity-step rule's 1e-6, iteration error was 82% of
  VAL-001 80x40's stored 2.3e-3, and on the 80x80 cavity it was 1.1 times the discretization
  error [4, sections 3 and 6]. On the cavity the estimate lies within 0.77 to 1.34 of the true
  error from 1e-5 to 1e-10 [4, section 4].
- (b) the worst per-cell imbalance is below `mass_imbalance_tol` (1e-10, absolute), ECR-001
  criterion 6's per-cell clause. The velocity-step rule had stopped all three step 6 cases 2 to
  30 times above it, with the solver reporting convergence [9, section 2].
- (c) the summed absolute imbalance over rho times the inflow (closed: rho times the velocity
  scale times the longer side) is below `iteration_error_tol`. On the open channel, once the
  pressure solve drops to one or two sweeps, the error left is a drift of the through-flow that
  the velocity step does not see, and (b) alone lets it grow with the cells upstream, 6.3e-6 of
  the inlet speed at 80x40 and four times that per refinement. (c) bounds it on any grid. It
  moved the VAL-001 80x40 stop from 2286 to 3154 outer iterations and the true error from
  2.99e-6 to 5.44e-7 of the inlet speed [4, section 9].
- (d) the absolute value of the signed imbalance summed over the domain, the net mass flux out
  of it, is below `mass_imbalance_tol`: criterion 6's domain-sum clause, which (b) and (c) did
  not check (review 27). On the closed cavity it holds from the first outer iteration and the
  cavity stops are unchanged. On the open channel the net outflow decays as an oscillation
  about zero, and (d) is met at one of its zero crossings, not where the oscillation has
  settled: after the 80x40 stop at 3988 the net outflow reaches 8.7e-9 again, and of the last
  200 iterations only the stop is below 1e-10. Alex accepted that on 2026-10-01, with the
  returned field meeting criterion 6 as written and (a) bounding its accuracy: the true error
  at the 40x20 and 80x40 stops is 2.0e-8 and 6.1e-8 of the inlet speed, 34 and 9 times less
  than under the three conditions [4, section 10]. *Note 2026-10-07:* the oscillation belonged to
  the weighted Jacobi correction; under ECR-003's conjugate gradients (d) holds from the first
  outer iteration on both 80x40 channels (For Phase 3, below).

The three-condition rule cost 1.07 to 1.16 times the default rule's wall time on the five
validation solves [4, section 9]; (d) moved every channel stop later, 1.26 to 1.58 times the
three-condition outer count, and the cavity stops not at all [7, addendum]. The 80x80 cavity
needs 12849 outer iterations, so its case file's cap is 20000 [8].

**6. Non-uniform mesh (REQ-S11).** Each axis may be clustered toward both walls by a constant
geometric ratio, mirrored about the midpoint (`src/mesh.py`). At a fixed cell count the ratio
and the wall-adjacent width determine each other, so the configuration names one and the mesh
derives the other. ECR-001 criterion 2 fixes the wall cell at 0.1 H / ny; at 80x40 the derived
ratio is 1.2057 [7]. The measured cost on this stencil: the clustered 80x40 channel scores
3.024e-3 against the uniform grid's 4.104e-4, 7.4 times higher at the same cell count. 96% of
it, 2.893e-3, is the clustered stencil's own error on the fully developed flow, which falls at
order 1.98 then 2.02 on the clustered family and is above 1% at 40x20 [7, section 1]. A
geometric mesh puts the midpoint of two centers (h_N - h_P) / 4 off their face, which changes
the diffusion of a parabola by (r - 1)^2 / (4 r), 0.88% at r = 1.2057 (INFERRED) [7, section 1];
how the error splits between that offset and the wall stencil was not separated [7, section 5].

## Validation results

| ECR-001 criterion | Measured on the staggered solver | Source |
|---|---|---|
| 1. VAL-001 < 1% L2, 80x40 uniform | 4.107e-4 (4.104e-4 under the three-condition rule) | [7] |
| 2. VAL-001 < 1% L2, 80x40 clustered | 3.024e-3 | [7] |
| 3. VAL-002 < 2%, 80x80, vs `marchi_2009_re100` | u 1.057e-3, v 7.356e-4 of the lid speed | [8] |
| 3a. Falls at 20, 40, 80 | u 2.144e-2, 4.548e-3, 1.057e-3; v 1.337e-2, 3.080e-3, 7.356e-4 | [8] |
| 4. Order >= 1.8, VAL-001 | 1.992 (1.993), reference-free, 40x20 to 160x80 | [7] |
| 6. Per cell < 1e-10 at the stop | cavity 9.92e-11, 9.98e-11, 2.34e-11; channel 2.98e-12, 1.10e-12, 1.40e-13 at 40x20, 80x40, 160x80; clustered 2.61e-12 | [4, section 10], [7, addendum] |
| 6. Signed domain sum < 1e-10 at the stop | cavity -3.1e-18, -1.8e-18, -1.1e-18; channel -8.2e-11, 5.6e-11, 9.5e-11; clustered -9.0e-11 | [4, section 10] |

Every solve stopped by `error_estimate_and_continuity`, none at its cap [7, 8]. Criterion 5
(the Phase 1 validation tests) is in the Phase 2 report. Criterion 6 is met on every case, per
cell and in the signed domain sum, by condition (d) [4, section 10]. On the open channel (d) is
met at a zero crossing of a decaying oscillation of the net outflow, not where the oscillation
has settled; Alex accepted that on 2026-10-01, with the returned field meeting the criterion as
written and condition (a) bounding its accuracy. This solver has no outflow correction, which
would make the signed sum zero by construction (decision 5, and Consequences). The criterion's
baseline and its "by construction" argument are the closed cavity's.

**The references.** VAL-001 is scored against the analytical parabola, but criterion 4 is judged
against no reference: at x = L/2 the flow is still developing, and the profile there differs
from the one at 3L/4 by 1.9e-4 of the parabola's norm on every grid from 80x40 [7, section 2].
VAL-002 was specified against Ghia et al. (1982). Its v table in the repository until
2026-09-22 was not Ghia's Table II and failed mass conservation along the centerline; it was
replaced as `ghia_1982_re100_r2` (ECR-001 section 12). Against that table the staggered solution
converges to values away from Ghia's by about 0.005 of the lid speed in u on the vertical
centerline near y = 0.85, and by 0.008 to 0.009 in v at the jet stations by the right wall
[6, section 9]. Marchi, Suero and Araki (2009), co-located central
differences on grids to 1024x1024 with Richardson extrapolation, shares no discretization with
this solver. Extrapolated from 80x80 and 100x100, the staggered solution lies within 1.9e-5 of
Marchi at all 30 of its points, and Ghia's table differs from Marchi by the gap [10]. Alex
decided on 2026-09-24 to score VAL-002 and criterion 3a against Marchi, with Ghia reported
beside it unscored. Against Ghia the same fields read u 8.90e-3, 3.99e-3, 4.81e-3 and v 6.49e-3,
8.25e-3, 8.94e-3, not falling monotonically in either component [8].

## Planned against built

| ECR-001 planned | Built | Why, and where measured |
|---|---|---|
| Rewrite `solver_ns.py` (7.1) | `solver_staggered.py` alongside it; the collocated solver kept | Both from one commit, so before and after compare [9]. Retirement is Alex's open decision |
| Rewrite `boundary.py`, ghost cells removed (7.1) | `boundary_staggered.py` new, `boundary_registry.py` extracted, `boundary.py` kept | One configuration reading for two layers (REQ-S12.1) |
| Stretching per wall by spacing and ratio; default config clustered (7.1, REQ-S11) | Per axis, mirrored, either quantity, the other derived; default config uniform | One free parameter at a fixed count (criterion 2 note); REQ-S11 amended; product mesh is Phase 3's |
| QUICK boundary stencils from Ferziger and Peric ch. 4 (4) | Leonard's (1979) appendix form; deferred correction over upwind | The negative coefficient and Jacobi's diagonal dominance (decision 3) |
| Pressure "integrated with existing Jacobi", REQ-S08 unchanged (5.2, 8) | Weighted Jacobi, w = 2/3; REQ-S08 clarified | Exact -1 eigenvalue on the closed domain [3] |
| Step 6 "convergence tuning" (8) | `error_estimate` rule, four conditions (version 3); REQ-S01, S04 clarified | Iteration error and flux drift the step cannot see, and criterion 6's domain sum [4] |
| Continuity "exactly" (4) | Closed-domain sum exact to rounding; per cell and signed domain sum below 1e-10 at every stop | The inner solve stops at `pressure_tol`; the rule enforces (b) and (d) [4] |
| VAL-002 against Ghia (5.2, criterion 3) | Against Marchi, Ghia unscored; metric `max_normalized_centerline_error_cubic` | Corrupted v table (section 12); Ghia's own error [10] |
| Criterion 2: ratio 1.05 and spacing 0.1 L/ny | Spacing 0.1 H/ny fixed, ratio 1.2057 derived | Amendment of 2026-09-24, ECR-001 criterion 2; measured in [7] |
| Criterion 4 order on VAL-001 | Reference-free; orders against the parabola reported beside | Development floor at L/2 [7] |
| Tests rewritten or tightened in place (7.2) | Staggered tests added (1%, 2%, own module files); collocated tests kept at 2.5% and xfail | Collocated kept as baseline; VAL-002 runs at 40x40 in CI, 80x80 judged from its row [8] |
| ADR-010 in the rebuild PR (6) | Written at step 9 | To record the build, not the plan |

Addendum 2026-10-02: the open decision in the first two rows is made. The collocated solver
and `boundary.py` were retired in PR 29; tag `collocated-final` on 98f8b1f holds them.

## Consequences

**Positive.** The pressure correction is a compatible system on a closed domain and does work
at every outer iteration. Wall and inlet faces hold exactly what the configuration prescribes.
Both validation cases converge at second order under refinement. A converged solve now carries
an error estimate and a continuity bound rather than a small last step.

**Negative.** The pressure solve is the cost: 92% to 98% of the staggered wall time at step 6
[9], with sweeps growing as N^2 [3]. Under `error_estimate` the 80x80 cavity runs 12849 outer
iterations, against 5728 under the old rule [4, section 9], and condition (d) moved the channel
stops 1.26 to 1.58 times later [7, addendum]. The returned cell means are O(h^2)
from the faces: on the cavity the metric reads 15% to 22% above the face values at 40x40 and
80x80 [8, section 2]. Two solvers and two boundary layers coexist, and `IterationState`, which
both use, is defined in `solver_ns.py` [9, section 6].
Addendum: retired 2026-10-02, PR 29; tag `collocated-final`. One solver and one boundary layer
remain, and `IterationState` is defined in `stopping.py`.

**For Phase 3.**
- *An outer iteration that adapts rather than overshoots.* On the open channel the net outflow
  decays as an oscillation about zero, an underdamped mode of the outer loop under fixed
  under-relaxation, which is why (d) is met at a zero crossing [4, section 10]. Alex's direction
  is an adaptive iteration that does not overshoot: an outflow correction that removes the
  net-outflow mode, or under-relaxation that adapts to the damping the solver observes. A Phase
  3 design question (`docs/STATUS.md`, open questions); nothing is built.
  *Note 2026-10-07 (ECR-003 step 2):* the oscillation was the weighted Jacobi correction's, not
  the outer loop's. That correction's 1e-8 Pa stop left a median 99.9% of u*'s imbalance in the
  corrected faces on both channels, so the net outflow was whatever the outer iteration had not
  yet removed. With the correction solved by conjugate gradients (ADR-013) the faces balance to
  rounding at every outer iteration: the signed domain sum stays below 1.8e-11 throughout,
  conditions (b) to (d) hold from the first outer iteration, and (a) alone sets the stop, at 1,559
  and 1,124 outer iterations against the 3,988 and 2,253 above
  (`docs/reports/ecr003_step2_baseline.md`, section 6). The decision on (d) of 2026-10-01 stands as
  history; under ECR-003 it does not bind on these cases.
- *Mass conservation is a particle-source concern.* A per-cell velocity imbalance is a source or
  sink of particle mass in the transport equation, and REQ-T05 asks for 0.01%. Continuity holds
  on the staggered faces; the returned cell-centered field is their average and does not carry
  it (INFERRED). How the transport solver reads face fluxes is a Phase 3 interface decision.
- *The stretching cost.* Deposition (REQ-T09) depends on the wall-adjacent spacing, which argues
  for clustering, and on this stencil clustering at a ratio near 1.2 cost 7.4 times the error at
  a fixed cell count [7]. The cost should be measured on the product case before a mesh is
  chosen. Only per-axis, mirrored clustering exists.
- *Density is not 1 in the product configuration.* Every validation case has rho = 1 [3, 4].
  `mass_imbalance_tol` is absolute, in kg/s per unit depth, so at air's density and the
  room's scale the same 1e-10 is a different fraction of the through-flow; conditions (a) and
  (c) are relative and carry over. The bound needs restating for the product case.
- *Obstacles.* A face of a SOLID cell is a fixed zero at its storage location, with no
  half-cell wall distance; obstacle accuracy was not a Phase 2 target (`src/momentum.py`).

**For Phase 6.** The CUDA port targets this solver, and REQ-N03's NumPy reference is now
`solver_staggered.py`. Weighted Jacobi keeps one thread per cell. A GPU shortens each sweep, not
their number, which quadruples per halving of h [3], on a production grid of 200x75 [1, section
7]; a faster algorithm would change REQ-S08, which this ADR does not do. ECR-003 did, on
2026-10-06: ADR-013 replaces the sweep by conjugate gradients, one five-point product and a
diagonal scaling per cell plus three reductions per iteration, and the CUDA kernel of Phase 6
becomes a CG kernel.

## Alternatives Considered
Within the build, decisions 4 and 5 and the references paragraph name theirs. At the level of
the ECR, options A (relax REQ-S03) and B (fix within the collocated layout) were rejected in
ECR-001 section 3. Option A's rejection leaned on the withdrawn v error; the wall leak alone
still rules out a scheme whose closed-domain pressure system has no solution [1].

## Sources
Each figure above is from one of these, by the section given.

1. `docs/reports/pressure_solver_probe.md`: Tables B and E, sections 5.3 and 7.
2. `docs/reports/inlet_flux_comparison.md`.
3. `docs/reports/pressure_correction_step5.md`: sections 1 to 3 and 5 (the addendum).
4. `docs/reports/stopping_rule_evidence.md`: sections 3 to 6, 9 and 10.
5. `docs/reports/momentum_sweep_probe.md`.
6. `docs/reports/cavity_self_convergence.md`: section 3 and section 9.
7. `docs/reports/val001_revalidation_step7.md`: sections 1 to 3 and 5, and the addendum of
   2026-09-30; rows b8a2f3df and d2a57fe1 (three conditions), f56fce25 and 43b02c72 (version 3)
   in `benchmarks/results.jsonl`.
8. `docs/reports/val002_revalidation_step8.md`: sections 1 and 2; rows 5129231b, 6e1cf3fe and
   0d0d7efa in `benchmarks/results.jsonl`.
9. `docs/reports/staggered_integration_step6.md`: sections 1, 2 and 6.
10. `docs/reports/cavity_reference_marchi.md`: sections 4 and 5.

The ADR-008 figures (2.04%, 2.5%) are from `docs/ADR/ADR-008-collocated-ghost-cell-walls.md`;
the Ghia correction is ECR-001 section 12.
