# CFD Clean Room: Project Status

This is the single current statement of where the project stands. It is rewritten rather
than appended to, so there is never a question of which version is authoritative; `git log`
on this file gives the sequence and the tree state at each revision.

It states conclusions and points at evidence. It does not reproduce measured values. Numbers
live in `benchmarks/results.jsonl`, which the harness writes, and in the reports under
`docs/reports/`. Structure lives in `docs/SYSTEM.md` section 3, which regenerates from the
source tree. Prose that restates either creates a second copy that can drift, and this
project has already lost five months to exactly that.

---

## Where the project stands

Phases 0 and 1 are complete. Phase 2 is in progress: the Navier-Stokes solver exists and
runs, VAL-001 passes on the collocated solver at its current criterion and on the staggered
solver at the rebuild's 1%, VAL-002 is marked xfail because the collocated solver fails it at
the 40x40 CI grid, and the approved engineering change request to rebuild the solver is seven
steps into its nine-step plan. Phases 3 through 7 have not begun.

`docs/PROJECT_PLAN.md` holds the phase detail, deliverables and validation gates.

## The open engineering change

ECR-001 replaces the collocated grid with Rhie-Chow interpolation by a staggered MAC
arrangement with non-uniform mesh support and QUICK advection. The change request itself is
the source of truth for scope, requirement edits, acceptance criteria and the implementation
plan: `docs/ECR/ECR-001-solver-architecture-rebuild.md`.

The rebuild decomposes into nine increments, each reviewable in isolation. Steps 1 and 2,
the mesh extension and the staggered field layout, and step 3, boundary conditions imposed
directly on the staggered components, are implemented. Step 3 split the boundary module in
two layers over one shared interpretation of the configuration (`src/boundary_registry.py`,
REQ-S12.1): the collocated layer is unchanged in interface and output, and the staggered
layer writes the normal components exactly and hands the tangential and pressure
conditions to steps 4 and 5 as data rather than as a mirrored value outside the domain.
Step 4 added the momentum predictor (`src/momentum.py`): QUICK advection carried as a
deferred-correction source over a first-order upwind implicit matrix, so the Jacobi
sweep keeps its diagonal dominance while the converged answer is the QUICK one. It
consumes step 3's tangential data without modification and returns the un-relaxed
momentum diagonals as the contract step 5 builds the pressure correction on.
Step 5 added the pressure correction (`src/pressure.py`). Its right-hand side is the
discrete divergence of u* read straight off the stored face velocities, and on the closed
cavity it sums to zero to rounding at 20, 40 and 80 cells per side, where the collocated
solver leaked 2.90e-2, 9.23e-3 and 2.35e-3. That is acceptance criterion 6 measured
directly, and it is the number the rebuild was undertaken to obtain. Step 3's
`pressure_outlets` contract was consumed unchanged.

REQ-S08 was deliberately left as written in step 5. Changing the pressure solver in the
same branch as the grid layout would make any improvement unattributable, and the baseline
comparison is the purpose of the rebuild; the cost was measured instead. The measurement
showed more than cost: on a closed domain undamped Jacobi has an exact eigenvalue of -1 on
the checkerboard mode, so it does not converge at all, and the two-cell case overshoots or
does nothing depending on the parity of the sweep cap. REQ-S08 has since been clarified
rather than amended, in a separate change so the weighting is attributable by itself:
the Jacobi update is weighted by two thirds, which keeps the requirement's architectural
content, the data-parallel per-cell update, and moves the -1 to -1/3. The closed-cavity
correction now converges. The weight is a constant in `src/pressure.py`, not a
configuration key. What it costs on the slow modes is measured in
`docs/reports/pressure_correction_step5.md`, section 5.
Step 6 put the three modules in one outer loop, `src/solver_staggered.py`, built
alongside the collocated solver rather than in its place: `src/solver_ns.py` is
unchanged, both run from one commit, and the harness `--method` label now selects which
one runs, so a row can no longer claim a solver it did not run. The collocated rows still
reproduce, and retiring the collocated solver moves to a later step. The staggered solver
converges on all three default cases and is more accurate than the collocated one on the
channel and in u on the cavity. On every case it stops with a per-cell mass imbalance
above acceptance criterion 6's bound: the stopping rule reads the velocity change, and
capped corrections make small steps before mass is conserved. Whether the rule needs a
continuity term is step 7's decision. The measurements, and what the unchanged metric
discards, are in `docs/reports/staggered_integration_step6.md`.
Ahead of step 7, `src/stopping.py` adds the rule step 7 will use, `error_estimate`: a solve
stops when the iteration error estimated from the velocity step and its own geometric rate,
over the largest prescribed boundary velocity, the worst per-cell mass imbalance, and the
summed imbalance over the through-flow are all below their tolerances, and reaching the
iteration cap is not convergence. It is opt-in by a solver configuration key, and the solver
block now rejects any key it does not know. The velocity-step rule stays the default and
reproduces the earlier fields bitwise, and the collocated solver refuses the new rule because
its walls leak. On both validation cases it stops with the imbalance below criterion 6's
bound, and on the cavity it leaves about the iteration error it estimates. On the open
channel the per-cell tolerance alone bounds the flux drift the rule can leave only loosely,
and more loosely under refinement, so the summed condition bounds that drift on any grid
(`docs/reports/stopping_rule_evidence.md`, section 9).
Step 7 revalidated VAL-001 on the staggered solver. The channel case file now names the
`error_estimate` rule, and every staggered VAL-001 solve stops by it. ECR-001 acceptance
criteria 1, 2 and 4 pass: below 1% on the uniform 80x40 grid and on the same grid clustered
toward the walls, and at second order under refinement judged without a reference, because
the parabola is not the exact answer at the channel midpoint while the flow still develops.
The clustered grid's error is its stencil's own error on the developed flow. The collocated
solver refuses the new rule and keeps the old one through one helper, with its results
unchanged; moving the case file reached two scripts beyond the plan, the stopping probe and
the viewer, which now name their rule. See `docs/reports/val001_revalidation_step7.md`.
ADR-010 is deliberately deferred to the end so it records what was built rather than what
was planned.

Four amendments to ECR-001 landed on 2026-09-20, before implementation. The discrete
continuity baseline for the collocated scheme is recorded under acceptance criterion 6, which
is now a quantified before-and-after rather than a sanity check. Criterion 3a requires the
cavity error to fall monotonically under refinement in both components; the v half of the
baseline it records was measured against a corrupted reference and is corrected in the
ECR-001 erratum. The problem statement carries the
continuity measurement as a second, independent confirmation of the wall-treatment defect.
And the `solve_steady` callback, sweep count and stage timing that the benchmark harness
depends on are recorded as an interface obligation on the rebuilt solver.

## What the diagnostics established

Three findings, each recorded with its measurements in the linked report.

The collocated ghost-cell wall treatment does not enforce zero normal mass flux, so mass
crosses the walls. In a closed domain this makes the pressure-correction system singular with
an incompatible right-hand side, which has no solution at all. Jacobi responds by drifting
uniformly, the velocity correction reads only gradients, and the drift is therefore
invisible. At convergence the pressure correction is a no-op and the domain carries a
uniform residual mass source. See `docs/reports/pressure_solver_probe.md`.

Neither inner solve governs the outer iteration count. Raising the pressure sweep cap changes
nothing. Raising the momentum sweeps buys a modest improvement that saturates quickly. What
remains belongs to the pressure-velocity coupling or to the iterate-change convergence
criterion, and is not yet identified. See `docs/reports/pressure_solver_probe.md` and
`docs/reports/momentum_sweep_probe.md`.

It was believed that the lid-driven cavity error diverges under grid refinement in the
v-component while the u-component converges, a directional defect tied to the wall
treatment above. That is false. The Ghia v reference the metric used from 2026-04-16 to
2026-09-22 was not Ghia's Table II and failed mass conservation along the centerline, so
every v error measured against it, including the series that opens ECR-001, measured the
distance from the wrong profile. Against the published table, now the reference
`ghia_1982_re100_r2`, the collocated v error falls under refinement as its u error does.
The wall leak above stands on its own measurement, which uses no reference data. See the
erratum in `docs/ECR/ECR-001-solver-architecture-rebuild.md`, section 12, and the r2 rows
in `benchmarks/results.jsonl`. Against the same table the staggered solver's v error is
below the collocated one's on the coarse grids but falls more slowly, and at 80x80 it is the
higher of the two. Measured against itself it converges at second order or better away from
the lid corners. Its slow approach to Ghia was mostly the metric's own sampling, half a cell
off the centerlines. The harness, the viewer and VAL-002 now sample on the centerlines under
a new metric name, `max_normalized_centerline_error_r2`, and the stored rows keep the old
one. On the true centerlines, and on fields converged well past the case's stopping
tolerance, the staggered solution converges toward values up to about 1% of the lid speed
from Ghia's in the jet by the right wall, and refinement moves it away from Ghia there.
That gap is Ghia's (INFERRED). Against an independent reference, Marchi, Suero and Araki
(2009), read from the paper's text layer and checked against its own published mass flow,
the extrapolated staggered solution agrees at all thirty of its points to a small fraction
of the gap, and Ghia's table differs from it by the gap. Under the new metric the staggered
error series is not monotone, which bears on ECR-001 criterion 3a. The collocated solver is
not yet asymptotic on these grids. See `docs/reports/cavity_self_convergence.md` and
`docs/reports/cavity_reference_marchi.md`.

The inlet flux the collocated layer prescribes on VAL-001 is short of the exact value by
two rows of cells, because its edge map hands the corner ring cells to the top and bottom
walls, and the flux its converged field actually carries through the walls is a third,
independent measurement of the leak. The staggered layer's inlet flux is the exact sum over
domain faces. See `docs/reports/inlet_flux_comparison.md`.

Separately, a validation figure carried in the earlier session handoffs was checked against
the code and found never to have been true of any committed state. The requirement history in
`docs/SYSTEM.md` and the rationale in `docs/ADR/ADR-008-collocated-ghost-cell-walls.md` have
been corrected. REQ-S02's threshold is unchanged; only its recorded justification moved.

## Tooling

`scripts/self_convergence.py` measures each solver's order on the cavity against itself, with
no reference, after a control on synthetic fields of known order. With `--extrapolate` it
extrapolates the staggered centerline profiles pointwise from the saved fields, after a
control of its own, where the observed order licenses it. With `--marchi` it sets the same
extrapolation against the independent reference `marchi_2009_re100`, which ECR-001
criteria 3 and 3a score against since the 2026-09-24 amendment; the metric moves to it in
step 8.

`scripts/stopping_probe.py` solves both validation cases far past their tolerance with the
pressure correction observed, and measures the iteration error each tolerance leaves, its
estimate from the residual's own rate, and the per-cell mass imbalance. With `--verify-rule`
it solves the same cases under the `error_estimate` rule and reads each stop against them.

`scripts/val001_order.py` measures VAL-001's order under uniform refinement on the staggered
solver, with no reference, after a control on synthetic fields of known order, and reports the
orders against the parabola beside it.

`scripts/gen_system_map.py` regenerates sections of `docs/SYSTEM.md` from the source tree by
AST parsing, never by importing. CI runs it with `--check`, so a change under `src/` that
leaves the document stale fails the build. Ported from the Agora project; the fork point is
recorded in its docstring.

`scripts/benchmark.py` appends one record per solver run to `benchmarks/results.jsonl`,
append-only and never rewritten. Each record carries accuracy against a named reference, work
in hardware-independent cell updates, wall time, and a trajectory of error against cumulative
work sampled during the solve. The three axes are separate on purpose: the rebuild changes
accuracy, the linear solver choice changes work, and a GPU port changes throughput, and a
record of wall time alone could not attribute an improvement to any of them.

`scripts/view_field.py` renders streamlines, pressure and the cavity centerline profiles
against the Ghia reference. A development instrument for reading the field during the
rebuild, not a deliverable; Phase 7 owns presentation-quality output.

`validation/` holds the case configurations, reference data and error metrics, imported by
both the test suite and the harness so the two cannot measure different things. Case
parameters live in committed YAML under `configs/`.

## Conventions

Feature branches off main, one pull request per reviewable increment, rebase and merge.
Every pull request runs ruff and the test suite (`.github/workflows/ci.yml`). Review and
test happen before the pull request, each in a fresh Claude Code session:
`/cfd-review NN` and `/cfd-test NN`, tracked in `.claude/commands/`, write
`docs/prompts/review-NN.md` and `test-NN.md`. A builder-fix pass follows if they find
defects. A round that finds at least as many findings as the one before it stops for a
decision, and prospective findings go to a GitHub issue. The reports are posted on the
pull request as its record. Both commands apply `docs/REVIEW_POLICY.md`. The review
Action that ran once on each pull request was removed on 2026-09-23: on its last three
pull requests it ran for about a minute and never reached its review stage. The
hand-rolled pipeline before it was removed on 2026-09-21. Task prompts live in
`docs/prompts/` and are not tracked. Reports live in `docs/reports/` and are.

Measured values belong in the file the instrument wrote. A document may state what a
measurement showed; it should not restate the measurement.

## Next

The rebuild continues at step 8, VAL-002 revalidation on the staggered solver. It judges
VAL-002 and criterion 3a against `marchi_2009_re100` on the true centerlines, with
`ghia_1982_re100_r2` reported beside it and no threshold (the ECR-001 amendment of
2026-09-24), and changes the metric and the VAL-002 test to match. Switching the cavity case
to the `error_estimate` rule, as step 7 did the channel, meets one known finding: the 80x80
cavity needs more outer iterations under the rule than the committed cap allows
(`docs/reports/stopping_rule_evidence.md`, section 9). The collocated solver needs
`validation.cases.with_velocity_step` wherever it is built from a case that names the rule.

With the review Action removed, its repository secret and the GitHub App it used are
still installed. Removing them is Alex's, after merge.

## Open questions

What sets the outer iteration count once the pressure correction is active rather than inert.
Not pursued further on the current solver, because the staggered rebuild makes the system
consistent and changes that regime in kind. The harness records the outer count on every run,
so the rebuild will surface it without a dedicated probe.

Closed 2026-09-22: how REQ-S08 should be amended. It was clarified, not amended. Weighted
Jacobi with w = 2/3 keeps the data-parallel per-cell update and maps the closed-domain
system's exact -1 eigenvalue to -1/3, and the closed cavity now converges. The rationale
is recorded in the requirement in `docs/SYSTEM.md`; the evidence is in
`docs/reports/pressure_correction_step5.md`, sections 3 and 5.

Whether VAL-002 can return to the full grid in CI once the rebuild lands.

Closed 2026-09-24: whether the stopping rule needs a continuity term. It does.
`docs/reports/stopping_rule_evidence.md` found that the committed tolerance leaves iteration
error as large as the discretization error on the finer grids, that an estimate from the
residual's own rate tracks that error while the iteration is geometric, and that on the
open channel the error left once the pressure solve falls to a sweep or two per outer
iteration is a flux drift that only the mass imbalance shows. On that evidence Alex decided
on a rule with two conditions and no pass at the cap: the estimated iteration error over a
physical velocity scale, and the worst per-cell mass imbalance, each below its own
tolerance. The review of the first build added a third, the summed imbalance over the
through-flow: the per-cell bound alone lets the flux drift grow with the cells upstream,
which the continuity condition was chosen to prevent. It is `error_estimate` in
`src/stopping.py`, opt-in, with the velocity-step rule the default; REQ-S01 and REQ-S04 are
clarified for it, not amended, and the validation cases switch to it in step 7. Section 9 of
the report checks that it stops where it should.

Closed 2026-09-22: what the VAL-002 v-component findings become against the published
Ghia table. The v reference was replaced by Table II as `ghia_1982_re100_r2`; the
collocated v error falls under refinement, and the v refinement argument in ECR-001 does
not hold. See the ECR-001 erratum, section 12.

Closed 2026-09-23: why the staggered solver's cavity error falls slowly toward Ghia.
Not the scheme: against itself it converges at second order or better away from the lid
corners. The metric's half-cell offset accounted for most of the slope, and is closed: the
metric now samples on the centerlines. On fields converged to 1e-9 the converged staggered
solution approaches values up to about 1% of the lid speed from Ghia's in the jet by the
right wall (`docs/reports/cavity_self_convergence.md`, section 9.3). That remaining gap is
Ghia's (INFERRED). At every one of the thirty points of Marchi, Suero and Araki (2009),
the solution's extrapolated values lie a small fraction of the gap from theirs, and Ghia's
table differs from theirs by the gap, with or without the solver's field used to compare
them
(`docs/reports/cavity_reference_marchi.md`).

Closed 2026-09-24: which reference ECR-001 criterion 3a and VAL-002 score the cavity
against. Alex decided on `marchi_2009_re100`, with `ghia_1982_re100_r2` reported beside it
and no threshold. Against Ghia, a correctly converging scheme meets a floor set by Ghia's
own error in the jet, below VAL-002's threshold but enough to make the error series rise
under refinement, which criterion 3a forbids (`docs/reports/cavity_reference_marchi.md`,
section 5). The decision is an amendment under criteria 3 and 3a in ECR-001 section 9. The
metric and the VAL-002 test move to Marchi in step 8; until then they read Ghia.

Closed 2026-09-24: which mesh quantity ECR-001 acceptance criterion 2 fixes. At a given
cell count the geometric ratio and the wall spacing determine each other, so the criterion
could not hold both. Alex fixed the wall spacing and derives the ratio from it: the stricter
test of the stretched stencils, and the quantity Phase 3 deposition depends on. The decision
is an amendment under criterion 2 in ECR-001 section 9.

## Superseded

The session handoff documents held outside this repository are historical records of
individual working sessions. This file supersedes them. At least one contains a validation
figure since shown to be false, so prefer the reports and the results file over any figure
quoted in them.
