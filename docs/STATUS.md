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
the engineering change request that rebuilt it on a staggered grid, closed on 2026-10-01, and
the phase gate is recorded in `docs/reports/phase2_navier_stokes_report.md`. VAL-001 and
VAL-002 pass on the staggered solver at their amended criteria. The collocated solver it
replaced was retired on 2026-10-02 in the first Phase 3 pull request: deleted from the tree,
kept at the annotated tag `collocated-final`, its stored harness rows kept (`docs/SYSTEM.md`,
section 4, Retired modules). Phase 3 is most of the way through: the face-velocity interface
the transport solver reads (REQ-S13), the transport configuration and the concentration
boundary module were built in PR 31, and the transport solver itself in PR 32, with all seven
of its gate rows measured and passing (VAL-003, VAL-004 in two rows, VAL-007, VAL-012, VAL-013,
VAL-014). What remains of the phase is the product configuration's move to `error_estimate` and
the gate report with ADR-011's planned-against-built table. Phases 4 through 7 have not begun.

The product room itself has never been solved, and on 2026-10-04 it was found that the laminar
flow solver does not solve it. Its Reynolds number on the room height is
about ninety thousand and its cell Reynolds number on the product mesh about twelve hundred; the
solver was validated at 5 and 100. As committed the solve diverges within sixty outer
iterations, and on a coarse copy of the room it diverges at the real viscosity, at ten times it
and, given long enough, at a hundred times, while at a thousand times its residual keeps
falling; heavier under-relaxation keeps it bounded without converging. The pressure outlets let
air into the room with no condition of its own, and the outlet measurement of prompt 33b (the
report, section 8) found air entering through them in every run that diverges, the divergence
sitting at the hood exhaust at ten times the viscosity. Giving the entering air a condition, or
fixing the hood's flow, moves and delays the divergence and converges nothing above a hundred
times the viscosity, and on the product mesh the growth starts inside the room before any outlet
face reverses. The outlets are part of how the solve diverges, not why it does not converge.
Those runs used the solver's one momentum sweep per outer iteration; with ten, the laminar
coarse room converges at real air and at ten times its viscosity (test 34b, step 0's report
section 7.6), so on that grid the non-convergence was the one-sweep iteration repelling a steady
solution that exists, not the Reynolds number. The flow is turbulent all the same, and k-epsilon
stays decided for the particles' mixing. ADR-004's "laminar flow" is the clean-room sense,
a unidirectional supply, not the Navier-Stokes one. The evidence, with the controls the probes
lack, is `docs/reports/product_case_reynolds.md`. Alex decided the same day to add a k-epsilon
turbulence model: ECR-002 (`docs/ECR/ECR-002-turbulence-model.md`) is the change request and
ADR-012 (`docs/ADR/ADR-012-turbulence-model.md`) its design, both accepted by Alex on
2026-10-04 with their decisions; ECR-002 fixes the outlets and probes convergence at a realistic
effective viscosity before anything is built (step 0), then measures it with the built code
before its first coupled solve, since the model's convergence is a hypothesis, not a given.
The product case, and with it Phase 3's last deliverable, waits for that change. The seven
transport gate rows stand: they were judged on prescribed or laminar face fields, and the scheme
does not change. A second finding of `docs/reports/product_case_reynolds.md`: one pressure
correction on the product mesh needs about a hundred and forty times the committed sweep cap at
the committed tolerance, growing as the square of the cells per side, so the pressure solve has
to change too (ADR-012, decision 5).

Step 0 ran on 2026-10-04 on the coarse copy of the room, with a frozen eddy viscosity shaped
like the indoor zero-equation model's (`docs/reports/ecr002_step0_frozen_viscosity.md`). At that
model's own size, several times above what k-epsilon will produce, the room converges; scaled
into k-epsilon's range it does not with the solver's one momentum sweep per outer iteration, and
with ten sweeps it converges at the top of that range. There the steady solution is a fixed
point the one-sweep iteration repels and ten sweeps make stable (test 34). Whether ten sweeps do
the same in the middle and at the bottom of the range is prompt 34b's measurement. It ran on
2026-10-05: with ten sweeps the middle and the bottom of the range converge as the top does, so
the sweep aid holds across k-epsilon's range on the coarse room (the report, section 7).

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
stored rows keep the old one; the viewer still uses the r2 metric, as the collocated VAL-002
test did until its retirement. On the true centerlines, and on fields converged well past
the case's stopping tolerance, the staggered solution converges toward values up to about 1%
of the lid speed from Ghia's in the jet by the right wall, and refinement moves it away from
Ghia there.
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
been corrected. That correction left REQ-S02's threshold at 2.5% and moved only its recorded
justification; ECR-001 step 9 then amended the threshold to 1% for the staggered solver
(`docs/SYSTEM.md`, section 2.1).

## Tooling

`scripts/self_convergence.py` measures the staggered solver's order on the cavity against
itself, with no reference, after a control on synthetic fields of known order; the collocated
orders it measured before the retirement are in its report. With `--extrapolate` it
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

Phase 3, the transport solver. The collocated solver was retired on 2026-10-02 (PR 29), and
the design is written and decided: `docs/ADR/ADR-011-transport-solver-architecture.md`,
Accepted at merge (PR 30), is what every Phase 3 module is built from. It answers the three
questions ADR-010 left to Phase 3. The transport solver reads the staggered face velocities,
which the NS solver will expose under REQ-S13; the per-cell imbalance bound becomes a drift
rate of a uniform haze, tested as VAL-012 under REQ-T11; the product mesh is still to be
measured before it is chosen.

Alex decided the design's three open questions on 2026-10-03, with seven further points its
review and test raised; the ten decisions are listed at the top of ADR-011. The face scheme is
QUICK bounded by the UMIST limiter under forward Euler at a Courant number of at most 1/2, so
the field is non-negative and bounded (REQ-T12, new). VAL-004 has two rows, an oblique channel
pulse and a rotating puff, with thresholds set from prototype measurements at the product
case's Courant number of 0.1. The supply is clean by default and particles enter as sources
through the solver's new `sources` argument. The sealed-box case VAL-014 guards the settling
and deposition composition, after the test found settling counted twice on obstacle tops in the
proposed text.

The first build (PR 31) laid what the solver stands on, in the order the design gives them:
`StaggeredSolver.face_velocities`, the read-only faces continuity was enforced on (REQ-S13,
re-measured on VAL-001 40x20 from the faces alone); the `transport` configuration section and
the three segment keys, with the product supply marked `hepa_filtered`; and
`src/boundary_concentration.py`, which derives every face's scalar condition from the same
call the velocity layer makes, `BoundaryRegistry.coverage_along` on the coordinates and SOLID
mask `staggered.edge_cell_inputs` derives once, so neither layer decides coverage or its inputs
on its own; a test checks the two layers' inlet sets are equal on every committed configuration
and on two with an obstacle on an edge under an inlet. The fix pass on review 31 and test 31
added three load rules Alex decided on 2026-10-03: unknown segment keys, overlapping segments
on one edge, and a concentration on a zero-normal inlet (a lid, a wall to the scalar layer)
each fail the load.

The second build (PR 32) is `src/solver_transport.py` and the seven Phase 3 test files, each
from the ADR's sections: QUICK's face value bounded by the UMIST limiter under forward Euler
at the configured Courant number, implicit diffusion and deposition by Jacobi, settling on the
interior faces only, sources, one budget per class written by the solver alone, and the
field-history writer for Phase 7. Before any gate test was written the real solver was run on
VAL-004's two rows against the prototype that set their thresholds, and reproduced its figures
to four figures (`results/builder32/item0.md`, untracked). Every gate row then passed at its
criterion, with the measured value beside it in `docs/PROJECT_PLAN.md`: the diffusion error is
a fifth of the criterion at a step where the time error is below the spatial one, the two pulses
keep 82% and 75% of their peak with no cell below zero, the uniform haze drifts at 0.15 of the
bound REQ-T11 states, the Smith-Hutton front stays inside its inlet's bounds at every step over
20 s, the budget closes to rounding on a random face field and on the VAL-001 faces, and the
sealed box deposits exactly `v_s C_0 W T`. The build found one error in the design: ADR-011 H's
doubled-floor control for VAL-014 cannot double the deposit under the implicit sink the ADR
itself chooses, because the floor row is then no longer stationary and the deposit is set by the
settling supply from above. Alex dropped the clause on 2026-10-03 (ADR-011 H as amended); the
composition it was meant to guard is guarded by planting the settling increment on the floor face
instead, which fails the exact line by half. The fix pass on review 32 and test 32 added the
tests five documented solver behaviours lacked, refused NaN and bool inputs before any
arithmetic, and moved VAL-003's gate step to where the measured error split puts the time error
below the spatial one. Next: the product configuration's move to `error_estimate` when the
product case is measured, and the Phase 3 gate report with ADR-011's planned-against-built
table.

The product case now waits for ECR-002. Before ADR-012 named a turbulent validation case, the
Annex 20 benchmark's data were checked against the property they cannot fake: in a flat room
every vertical section carries the inlet's flow. The measured middle plane carries about the
inlet's flow at the first section and six tenths of it at the second, while the measured plane
near the side wall carries a tenth more than all of it there; that the model room was
three-dimensional there is inferred from the two planes and the specification's hot-wire profiles
(the report, section 6). The design scores the room where that does not bite, against the one
published standard k-epsilon prediction on the same lines, its threshold left to Alex; the
backward-facing step joins the validation, its threshold also OPEN for want of a sourced range.
The premise review of ECR-002 and ADR-012 and `/cfd-test 33` ran, the fix pass of prompt 33b
answered them, and `/cfd-test 33b`'s findings, all in the text, were applied directly. Alex took
the design's decisions and accepted the request on 2026-10-04. Step 0, the risk-retirement
probe, ran the same day, and its pull request carries ECR-002's requirement and scope text into
`docs/SYSTEM.md`. Section 7 of its report (prompt 34b) found ten momentum sweeps converge the
room across k-epsilon's range. Step 1, k and eps on a prescribed face field, is built on
`feature/ecr002-k-epsilon-scalar` (prompt 35): the transport scheme shared through
`src/scalar_scheme.py` with the transport gate unchanged to the bit, the `turbulence`
configuration section, and `src/turbulence.py` for both variants, VAL-015 passing
(`docs/PROJECT_PLAN.md`). Review 35 and test 35 found its tests one-directional and
`nu_t` unchecked; the fix pass of prompt 35b answers them on
`feature/ecr002-k-epsilon-scalar`. Next: `/cfd-test 35b`, the pull request, then step 2,
and ECR-003, the pressure solve, which steps 1 to 4 do not wait for.

Deferred findings from earlier pull requests are open as GitHub issues 33, 36, 38, 40 and 42.

With the review Action removed, its repository secret and the GitHub App it used are
still installed. Removing them is Alex's, after merge.

## Open questions

Step 0's question is answered (step 0's outcome B, then section 7's outcome A,
`docs/reports/ecr002_step0_frozen_viscosity.md`): on the coarse room ten momentum sweeps per outer iteration converge the room across
k-epsilon's range, where one sweep does not. A configurable momentum sweep count
(default 1, so the laminar results stay bitwise) is built with step 4's viscosity field and used
from step 5 on. Step 1 settled RNG's C_mu at 0.0845 (Alex, 2026-10-05) and found that a
wall cell's held eps must follow k, so step 6 rebuilds the wall conditions every outer
iteration (`docs/SYSTEM.md`, the turbulence.py contract). ECR-002's requirement and scope text entered `docs/SYSTEM.md` in step 0's pull
request, since that edit moves the requirement register `tests/test_system_map.py` pins. The
Annex 20 and backward-facing step thresholds stay OPEN by decision until the first coupled
results exist.

Closed 2026-10-03: the three questions ADR-011 left open, the face concentration scheme and
with it positivity, a number for VAL-004's "shape preserved", and the HEPA supply
concentration with or without recirculation. Alex decided on 2026-10-03: a limited QUICK under
forward Euler at Courant number at most 1/2, with positivity a requirement (REQ-T12); measured
thresholds for VAL-004 in two rows; a clean supply with no recirculation and sources as the
entry path. The decisions, with the alternatives rejected, are listed at the top of the ADR.

Whether the cavity centerline metrics should keep the wall-adjacent ring rows. They drop
them today. At 20x20 that moves the Marchi u value; at 80x80, where criterion 3 is judged, it
does not, and no recorded verdict changes. GitHub issue 42 has the measurements and three
options; two of them need a new metric name, and the one that waited on the retirement
decision has been open to choose since 2026-10-02.

What sets the outer iteration count once the pressure correction is active rather than inert.
Asked of the collocated solver and not pursued there, because the rebuild changed that regime
in kind. Open on the staggered solver, with no dedicated probe. What its records show so far:
on the channel the count is set by the net outflow's underdamped oscillation, the next
question; on the cavity the condition met last is the per-cell imbalance at 20x20 and 40x40
and the estimated iteration error at 80x80 (`docs/reports/stopping_rule_evidence.md`, section
10). The harness records the outer count on every run.

An outer iteration that adapts rather than overshoots, a Phase 3 design question recorded
2026-10-01. On the open channel the net mass outflow decays as an oscillation about zero, an
underdamped mode of the outer loop under fixed under-relaxation, and that is why the stopping
rule's domain-sum condition is met at a zero crossing (`docs/reports/stopping_rule_evidence.md`,
section 10). Alex's stated direction is an adaptive iteration that does not overshoot. There
are two candidate fixes: an outflow correction that removes the net-outflow mode, or
under-relaxation that adapts to the damping the solver observes. Nothing is built yet.
Deferred on 2026-10-02 to a later efficiency and clean-up pass; it is not Phase 3 work.

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
