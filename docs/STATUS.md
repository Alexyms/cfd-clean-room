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
runs, VAL-001 passes against its current criterion, VAL-002 is marked xfail against a
documented defect, and the approved engineering change request to rebuild the solver is
four steps into its eight-step plan. Phases 3 through 7 have not begun.

`docs/PROJECT_PLAN.md` holds the phase detail, deliverables and validation gates.

## The open engineering change

ECR-001 replaces the collocated grid with Rhie-Chow interpolation by a staggered MAC
arrangement with non-uniform mesh support and QUICK advection. The change request itself is
the source of truth for scope, requirement edits, acceptance criteria and the implementation
plan: `docs/ECR/ECR-001-solver-architecture-rebuild.md`.

The rebuild decomposes into eight increments, each reviewable in isolation. Steps 1 and 2,
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
The solver itself is untouched so far and the harness rows reproduce at every step.
ADR-010 is deliberately deferred to the end so it records what was built rather than what
was planned.

Four amendments to ECR-001 landed on 2026-09-20, before implementation. The discrete
continuity baseline for the collocated scheme is recorded under acceptance criterion 6, which
is now a quantified before-and-after rather than a sanity check. Criterion 3a requires the
cavity error to fall monotonically under refinement in both components, with the measured
series that moves the wrong way in v as the baseline. The problem statement carries the
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

The lid-driven cavity error diverges under grid refinement in the v-component while the
u-component converges. A scheme that improves in one direction and degrades in the other has
a directional defect rather than a resolution shortfall, and this is consistent with the wall
treatment above: the u-field is driven directly by the lid boundary condition, while the
v-field depends on a continuity constraint that the inert pressure correction never enforces.
Recorded in `benchmarks/results.jsonl`.

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
Every pull request runs ruff and the test suite (`.github/workflows/ci.yml`). A pull
request gets one automated review when it opens or leaves draft, posted on the pull
request as a record rather than a gate and run on the Claude subscription
(`.github/workflows/review.yml`). Review iteration happens in a Claude Code context in
VS Code, the only reviewer that can check whether the previous round's findings were
fixed. Both apply `docs/REVIEW_POLICY.md`. The hand-rolled review pipeline that
preceded this was removed on 2026-09-21; its five review rounds on one pull request
were almost entirely about the pipeline itself. Task prompts live in `docs/prompts/`
and are not tracked. Reports live in `docs/reports/` and are.

Measured values belong in the file the instrument wrote. A document may state what a
measurement showed; it should not restate the measurement.

## Next

Clear the outstanding branch stack bottom up through review. The rebuild continues at step
5, the pressure correction on the staggered grid, which consumes the momentum diagonals
step 4 returns and the pressure outlet data step 3 exposes.

## Open questions

What sets the outer iteration count once the pressure correction is active rather than inert.
Not pursued further on the current solver, because the staggered rebuild makes the system
consistent and changes that regime in kind. The harness records the outer count on every run,
so the rebuild will surface it without a dedicated probe.

Whether REQ-S08 should be rewritten as a requirement on solver behaviour, stated as residual
reduction, rather than mandating Jacobi by name. The N-squared cost of Jacobi is real and
currently hidden by the inner solve being a no-op; the rebuild will expose it.

Whether VAL-002 can return to the full grid in CI once the rebuild lands.

Which mesh quantity ECR-001 acceptance criterion 2 fixes. At a given cell count the
geometric ratio and the wall spacing determine each other, so the criterion's ratio of 1.05
and wall spacing of 0.1 of uniform cannot both hold on the 80x40 grid. The mesh module accepts
either one with the other derived; the criterion must name one before step 7.

## Superseded

The session handoff documents held outside this repository are historical records of
individual working sessions. This file supersedes them. At least one contains a validation
figure since shown to be false, so prefer the reports and the results file over any figure
quoted in them.
