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

Phases 0 and 1 are complete. Phase 2's Navier-Stokes solver is built and validated. ECR-001,
the engineering change request that rebuilt it on a staggered grid, closed on 2026-09-30, and
the phase gate is recorded in `docs/reports/phase2_navier_stokes_report.md`. VAL-001 and
VAL-002 pass on the staggered solver at their amended criteria. The collocated solver it
replaced as the solver of record still runs as the benchmark harness's baseline, and VAL-002
stays marked xfail on it. Phases 3 through 7 have not begun.

`docs/PROJECT_PLAN.md` holds the phase detail, deliverables and validation gates.

## The engineering change, closed

ECR-001 replaced the collocated grid with Rhie-Chow interpolation by a staggered MAC
arrangement with non-uniform mesh support and QUICK advection. The change request holds the
decision, its amendments and its acceptance criteria:
`docs/ECR/ECR-001-solver-architecture-rebuild.md`. ADR-010 holds what was built, each decision
with the report that measured it, and a table of where the build departed from the plan:
`docs/ADR/ADR-010-staggered-grid-architecture.md`.

The rebuild ran in nine steps, each reviewed before it merged: the mesh, the staggered
layout, boundary conditions imposed directly on the staggered faces over one shared reading of
the configuration, the momentum predictor, the pressure correction, their integration as one
solver, VAL-001, VAL-002, and the records. The staggered solver was built beside the
collocated one, so both run from one commit and every comparison is a before-and-after on the
same tree. On a closed domain its pressure correction is a solvable
system, which the collocated one was not, because the collocated walls leak mass.

Four decisions the plan did not anticipate were made during the build, each recorded where
it was made. The pressure Jacobi sweep is weighted by two thirds, because the plain sweep has
an exact -1 eigenvalue on a closed domain and never converges there (REQ-S08, clarified). The
validation cases stop by an `error_estimate` rule that bounds the iteration error and the mass
imbalance rather than the last velocity step, because the step rule left iteration error as
large as the discretization error and could not see a flux drift on the open channel (REQ-S01
and REQ-S04, clarified). The cavity is scored against Marchi, Suero and Araki (2009) rather
than Ghia et al. (1982), whose table carries an error of its own that sets a floor under a
correct scheme (REQ-S03, amended). And after step 8 the rule gained a fourth condition, the
second clause of acceptance criterion 6, which nothing had checked: the signed mass imbalance
summed over the domain, the net outflow, below the per-cell bound (REQ-S04, clarified again).
The cavity stops are unchanged. The channel stops later, with less iteration error left, and
there the condition is met at a zero crossing of a decaying oscillation of the net outflow, not
where the oscillation has settled: between crossings the outflow is well above the bound. Alex
accepted that as built on 2026-10-01: the returned field meets the criterion as written, and
the estimated-error condition bounds its accuracy. Harness rows now record the rule's version,
so the summary keeps the rule's versions apart (`docs/reports/stopping_rule_evidence.md`,
section 10).

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
off the centerlines. From 2026-09-23 until step 8 the harness, the viewer and VAL-002 sampled
on the centerlines under a new metric name, `max_normalized_centerline_error_r2`, and the
stored rows keep the old one; the viewer and the collocated VAL-002 test still use the r2
metric. On the true centerlines, and on fields converged well past the case's stopping
tolerance, the staggered solution converges toward values up to about 1% of the lid speed
from Ghia's in the jet by the right wall, and refinement moves it away from Ghia there.
That gap is Ghia's (INFERRED). Against an independent reference, Marchi, Suero and Araki
(2009), read from the paper's text layer and checked against its own published mass flow,
the extrapolated staggered solution agrees at all thirty of its points to a small fraction
of the gap, and Ghia's table differs from it by the gap. Under the r2 metric, against Ghia,
the staggered error series is not monotone, which is why criterion 3a moved to Marchi. The
collocated solver is not yet asymptotic on these grids. See
`docs/reports/cavity_self_convergence.md` and `docs/reports/cavity_reference_marchi.md`.

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
criteria 3 and 3a score against since the 2026-09-24 amendment, and which the VAL-002 metric
reads since step 8.

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

Phase 3, the transport solver, after one decision that comes first. Whether to retire the
collocated solver is Alex's: keep `src/solver_ns.py` as the harness's before-and-after
baseline, or retire it, `src/boundary.py` and their tests in a pull request of its own. Either
way, `IterationState`, which both solvers use, is defined in `src/solver_ns.py` today.

ADR-010 lists what Phase 3 inherits from the solver. Continuity is enforced on the staggered
faces, and the cell-centered fields the solver returns are their averages, so how the
transport solver reads face fluxes is an interface decision. The per-cell mass imbalance
bound is absolute and was set on cases at unit density, so it needs restating for the product
configuration. And wall clustering, which deposition wants, cost accuracy at a fixed cell
count on the VAL-001 stencil, so the product mesh should be measured before it is chosen.

Deferred findings from earlier pull requests are open as GitHub issues 33, 36, 38, 40 and 42.

With the review Action removed, its repository secret and the GitHub App it used are
still installed. Removing them is Alex's, after merge.

## Open questions

Whether the cavity centerline metrics should keep the wall-adjacent ring rows. They drop
them today. At 20x20 that moves the Marchi u value; at 80x80, where criterion 3 is judged, it
does not, and no recorded verdict changes. GitHub issue 42 has the measurements and three
options; two of them need a new metric name, and one waits on the retirement decision.

What sets the outer iteration count once the pressure correction is active rather than inert.
Not pursued further on the current solver, because the staggered rebuild makes the system
consistent and changes that regime in kind. The harness records the outer count on every run,
so the rebuild will surface it without a dedicated probe.

An outer iteration that adapts rather than overshoots, a Phase 3 design question recorded
2026-10-01. On the open channel the net mass outflow decays as an oscillation about zero, an
underdamped mode of the outer loop under fixed under-relaxation, and that is why the stopping
rule's domain-sum condition is met at a zero crossing (`docs/reports/stopping_rule_evidence.md`,
section 10). Alex's stated direction is an adaptive iteration that does not overshoot. There
are two candidate fixes: an outflow correction that removes the net-outflow mode, or
under-relaxation that adapts to the damping the solver observes. Nothing is built yet.

Closed 2026-10-01: whether the domain-sum condition should hold over a window, or bound the
oscillation's envelope, rather than be met at a zero crossing. It should not. Alex and the
orchestrator accepted it as built: the returned field meets ECR-001 criterion 6 as written,
and the estimated-error condition bounds its accuracy. Section 10 of the evidence report
records the envelope it leaves.

Closed 2026-09-22: how REQ-S08 should be amended. It was clarified, not amended. Weighted
Jacobi with w = 2/3 keeps the data-parallel per-cell update and maps the closed-domain
system's exact -1 eigenvalue to -1/3, and the closed cavity now converges. The rationale
is recorded in the requirement in `docs/SYSTEM.md`; the evidence is in
`docs/reports/pressure_correction_step5.md`, sections 3 and 5.

Closed 2026-09-25: whether VAL-002 can return to the full grid in CI once the rebuild lands.
It does not need to. CI runs the staggered VAL-002 on the case file's 40x40 in about a minute,
below 2% in both components; ECR-001 criterion 3's 80x80 needs more than six minutes and is
judged from a harness row and the step 8 report, not in CI.

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
metric and the VAL-002 test moved to Marchi in step 8.

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
