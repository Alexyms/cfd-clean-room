# Phase 2 Completion Report: Navier-Stokes Solver

**Date:** 2026-10-01
**Phase Gate Verdict:** PASS
**Branches Merged:** 33 pull requests, #6 to #43, and this one (`docs/ecr-001-close-phase2`):
- Collocated solver and validation: #6, #7, #9, #10. ECR-001: #11 (approved), #15 (amended).
- ECR-001 steps: #16 (1, 2), #25 (3), #26 (4), #27 and #28 (5), #29 (6), #35 and #37 (the
  stopping rule), #39 (7), #41 (8), #43 (the rule's fourth condition, criterion 6's domain
  sum). References and metrics: #30, #31, #32, #34.
- Instruments: #12, #13, #14, #23. Review and CI tooling: #8, #17 to #22, #24.

Bracketed numbers point at the sources at the end. The test and coverage figures are from one
run at 0017e34, this branch's third commit, the rebase of the step 9 documents onto #43. The
commits after it change `docs/`, `README.md` and one annotation in `tests/test_validation.py`,
whose 39 tests were rerun at 12eac95. The lint and map checks are from 12eac95, the commit
before this one.

## Scope

### Planned
`docs/PROJECT_PLAN.md` as first written (eb3adf9, 2026-04-15): the SIMPLE solver in pure NumPy
(`src/solver_ns.py`), its boundary conditions (`src/boundary.py`) and tests, and the gate
VAL-001 (Poiseuille, L2 < 1% against the parabola) and VAL-002 (cavity, within 2% of Ghia et
al.). On 2026-04-16 VAL-001 was set on 80x40 and relaxed to 2.5% by ADR-008 (af71bf9).
ECR-001, approved 2026-04-16, replanned the solver as a staggered rebuild in nine steps.

### Delivered
- The collocated solver, as planned, which met VAL-001 at 2.5% and failed VAL-002.
- ECR-001, closed at step 9: a staggered SIMPLE solver (`src/solver_staggered.py`) over the MAC
  layout, direct wall imposition on a shared boundary registry, QUICK by deferred correction,
  weighted Jacobi, per-axis wall clustering and the `error_estimate` stopping rule, four
  conditions at version 3 since #43. ADR-010 records it, with a table of planned against built.
- VAL-001 at < 1% and VAL-002 at < 2% against `marchi_2009_re100`, both on the staggered
  solver, with every ECR-001 acceptance criterion demonstrated (ADR-010, Validation results),
  after the Ghia v reference was corrected (ECR-001 section 12) and Marchi's added [7].
  Criterion 6 is met on every case, per cell and in the signed domain sum, by condition (d) of
  the rule. On the open channel (d) is met at a zero crossing of a decaying oscillation of the
  net outflow, not where it has settled; Alex accepted that on 2026-10-01, with the returned
  field meeting the criterion as written and condition (a) bounding its accuracy [3, section
  10; ECR-001 section 9].
- Added without plan entries, each to measure the solver: the benchmark harness and its results
  file, the `validation/` package the tests and harness share, the system map generator, the
  field viewer, and the self-convergence, stopping-probe and VAL-001 order scripts.

### Deferred
- **The collocated solver's retirement, Alex's decision.** Option 1: keep `src/solver_ns.py`,
  `src/boundary.py` and their tests as the harness's before-and-after baseline. They cost
  upkeep, and `IterationState`, which both solvers use, stays in `solver_ns.py`. Option 2:
  retire the three in a pull request of their own and move `IterationState`. The stored
  collocated rows remain, but no new baseline row can be taken. PROJECT_PLAN marks them RETAINED.
- **An outer iteration that adapts rather than overshoots**, Alex's stated direction, a Phase 3
  design question: an outflow correction that removes the channel's net-outflow mode, or
  under-relaxation that adapts to the damping the solver observes (ADR-010, For Phase 3;
  `docs/STATUS.md`, open questions). Nothing is built.
- **Whether the cavity metrics keep the wall-adjacent ring rows**, GitHub issue 42. No verdict
  moves either way [6, section 2].
- **Deferred review and test findings:** GitHub issues 33, 36, 38, 40 and 42.
- **To Phase 3**, from ADR-010: how the transport solver reads the staggered face fluxes;
  restating the absolute per-cell imbalance bound at the product density; measuring the
  clustering cost on the product case. One-wall and interior clustering are not built (REQ-S11
  as amended).
- **To Phases 4 and 6, as planned:** `solve_timestep` and the CUDA kernels. A faster pressure
  algorithm would change REQ-S08 and is not proposed here.

## Test Results

### Unit Tests
- Tests run: 520
- Passed: 520
- Failed: 0
- Skipped: 0

### Integration Tests
- Tests run: 52
- Passed: 52
- Failed: 0

### Validation Tests
26 tests carry the validation marker: 25 passed, 1 xfailed. The ECR-001 order
and refinement criteria are measured by scripts and harness rows, not by pytest.

| ID      | Description | Criterion | Result | Value |
|---------|-------------|-----------|--------|-------|
| VAL-001 | Poiseuille, staggered, 80x40 uniform | L2 < 1% (REQ-S02) | PASS | 4.107e-4 [5, addendum] |
| VAL-001 | Poiseuille, staggered, 80x40 clustered to 0.1 H / ny | L2 < 1% (ECR-001 crit. 2) | PASS | 3.024e-3 [5, addendum] |
| VAL-001 | Order, staggered, 40x20 to 160x80 (script) | >= 1.8 (crit. 4) | PASS | 1.992 [5, addendum] |
| VAL-001 | Poiseuille, collocated baseline | L2 < 2.5% (ADR-008) | PASS | 2.036e-2 [5] |
| VAL-002 | Cavity, staggered, 40x40 (CI test; harness row value) | max < 2% of lid vs Marchi (REQ-S03) | PASS | u 4.548e-3, v 3.080e-3 [6] |
| VAL-002 | Cavity, staggered, 80x80 (harness row) | max < 2% vs Marchi (crit. 3) | PASS | u 1.057e-3, v 7.356e-4 [6] |
| VAL-002 | Falls at 20, 40, 80 (harness rows) | monotone in u and v (crit. 3a) | PASS | orders u 2.24, 2.11; v 2.12, 2.07 [6] |
| VAL-002 | Cavity, collocated baseline, 40x40 | max < 2% vs `ghia_1982_re100_r2` | XFAIL | u 0.0441, v 0.0606 [4] |
| VAL-005, 006, 010, 011 | Phase 1 particle physics, 21 tests | pass (ECR-001 crit. 5); the tests' own tolerance is 0.1% | PASS | as Phase 1 |

The VAL-001 staggered values are the rows taken under the four-condition rule; the cavity stops
are unchanged by it [3, section 10].

### Coverage
- Line coverage: 98.2% (1649 of 1680 statements), against Phase 1's 95%.
- Branch coverage: 91.5% (443 of 484 branches). Phase 1 did not measure it.
- Uncovered: 22 lines of `config.py`, guards for malformed YAML, and the unreachable HEPA
  `RuntimeError` in `particles.py`, both as in Phase 1; an edge-row fallback in `boundary.py`
  (2 lines); a ratio-1 branch and two range guards in `mesh.py`; one line each in `staggered.py`,
  `boundary_staggered.py` and `solver_ns.py`. The four newest solver modules, `solver_staggered`,
  `momentum`, `pressure` and `stopping`, have no uncovered line.

## Phase Gate Criteria

1. **All planned validation tests pass.** Holds. VAL-001 and VAL-002 pass on the staggered
   solver at their amended criteria, and Phase 1's VAL-005, 006, 010 and 011 pass. The one
   xfail is the collocated solver's VAL-002, kept as the baseline's record; since this step the
   gate's VAL-002 is the staggered solver's (REQ-S03).
2. **All unit and integration tests pass.** Holds: 520 and 52, none failed.
3. **No ruff errors.** Holds: `ruff format --check .` reports 47 files already formatted, and
   `ruff check .` passes, at 12eac95.
4. **Coverage does not decrease.** Holds: 98.2% line coverage against 95%.
5. **SYSTEM.md is current.** Holds, on two checks. `scripts/gen_system_map.py --check` reports
   that SYSTEM.md's generated regions match the source tree at 12eac95. The hand-authored
   section 3.2 cascade rows and the section 4 headings, which that check does not read, were
   compared row by row with the generated import graph in this branch (review 27 B2): the rows
   for `config.py`, `mesh.py`, `solver_ns.py`, `solver_staggered.py` and `csolver/` and the
   headings for `mesh.py`, `solver_ns.py` and `solver_staggered.py` were brought in line, a
   `constants.py` row added, and every other row agrees with the graph. This branch also amends
   REQ-S02, S03 and S11, corrects REQ-S04's verification, and stops describing the collocated
   solver as retired in a later step.
6. **The report is filled out and committed.** Holds with this commit. Two of its sections are
   drafts for Alex's review, to be read before this pull request merges.

## Implementation Decisions

*Draft for Alex's review.* Each is recorded where it was made; ADR-010 holds the formal record.

1. **The staggered solver was built beside the collocated one** (step 6), not as a rewrite of
   `solver_ns.py` (ECR-001 section 7.1). Both run from one commit, so each claim is a
   before-and-after on one tree. The cost is two solvers and the retirement decision above.
2. **One boundary registry for both layers** (step 3, REQ-S12.1). The alternative, a staggered
   copy of the collocated interpretation, could drift while both exist.
3. **QUICK by deferred correction over an upwind matrix** (step 4). QUICK in the implicit matrix
   would break the diagonal dominance the Jacobi sweep needs (ADR-010, decision 3).
4. **Weighted Jacobi at 2/3, REQ-S08 clarified rather than amended** (#28). Plain Jacobi cannot
   converge on a closed domain; a different algorithm would amend REQ-S08; a weight near 1
   saves about a quarter of the sweeps [2, section 5]. A pull request of its own kept the
   change attributable apart from the layout.
5. **The `error_estimate` rule, opt-in, with the velocity-step rule the default** (#35, #37).
   A tighter step tolerance costs outer iterations and still misses the channel's flux drift
   [3]. The third condition, the summed imbalance, came from the review of the first build.
6. **VAL-002 scored against Marchi, Ghia unscored; criterion 2 fixes the wall spacing; criterion
   4 judged without a reference** (2026-09-24, 2026-09-25). Against Ghia a correct scheme meets a
   floor of Ghia's own error [7, section 5], and the flow at L/2 is still developing [5, section 2].
7. **Criterion 6's domain-sum clause checked by the rule, and the zero crossing accepted**
   (#43, 2026-10-01). Review 27 found that no condition checked the clause. Condition (d)
   checks it with no new key, and on the channel it is met at a zero crossing of the net
   outflow's decaying oscillation. Alex accepted that as built, with condition (a) bounding
   the field's accuracy, rather than a window, an envelope condition or an outflow correction
   [3, section 10].
8. **Measure before rebuilding.** The system map, the harness and the validation package (#12
   to #14) and the probes came before step 1, so every step had a baseline. The pressure probe
   found the sweep cap irrelevant to the outer count and found the wall leak that motivates the
   layout [1].
9. **Review moved before the pull request** (2026-09-23): `/cfd-review` and `/cfd-test` in
   fresh sessions, after the review Action stopped reaching its review stage.

## Deviations from Architecture

- **Modules.** Seven new: `staggered`, `boundary_registry`, `boundary_staggered`, `momentum`,
  `pressure`, `solver_staggered`, `stopping`. ECR-001 planned rewrites of `solver_ns.py` and
  `boundary.py`; both were kept. SYSTEM.md sections 3 and 4 carry each contract.
- **Interfaces.** `solve_steady` gained an `on_iteration` callback, and the solvers the
  `last_pressure_sweeps` and `stage_seconds` attributes, for the harness. `StaggeredSolver`
  carries the collocated contract's `solve_steady`, `residual_history`, `reference_velocity`,
  `last_pressure_sweeps`, `stage_seconds` and `last_mass_imbalance`, plus `converged`,
  `stop_reason` and `flux_scale`; it does not define `compute_residual` or `solve_timestep`,
  the second Phase 4's. `NavierStokesSolver` refuses `error_estimate`. The rule's imbalance
  callable returns an `ImbalanceSummary`, and `RULE_VERSION` is stored with saved solves and
  harness rows (#43).
- **Configuration.** An optional `mesh` section (per-axis ratio or wall spacing), and solver keys
  `stopping_rule`, `iteration_error_tol` and `mass_imbalance_tol`; unknown solver keys raise.
- **Requirements.** REQ-S07 and S09 replaced (ECR-001); S11 and S12 added, S12.1 derived; S08,
  S01 and S04 clarified; S02, S03 and S11 amended at step 9.
- **ECR-001 criterion 6 on the open channel.** Met as written, per cell and in the signed domain
  sum, at a zero crossing of the net outflow's decaying oscillation rather than where it has
  settled, as the Delivered section states and ECR-001 section 9 records. Removing the
  oscillation is the Phase 3 question above.

Each change was recorded in the SYSTEM.md Document History when it was made.

## Lessons Learned

*Draft for Alex's review.*

- **Check reference data by a property its source cannot fake.** The corrupted Ghia v table
  entered with the VAL-002 test on 2026-04-16 and carried ECR-001's problem statement until a
  mass-conservation check on the table found it (ECR-001 section 12). Marchi's table was checked
  against its own published mass flow before it was used [7].
- **A recorded figure comes from the file its instrument wrote.** The 1.54% once recorded for
  VAL-001 was never reproducible; the measured value is 2.04% (ADR-008).
- **A small last step is not a converged answer.** At the old tolerance the iteration error was
  as large as the discretization error or larger on the finest grids [3, section 6].
- **Each clause of a criterion needs its own check.** Criterion 6's domain-sum clause went
  unchecked from step 5 to step 8, until review 27 read the criterion against the rule. The
  check then showed the channel's net outflow oscillating about zero, which nothing had seen
  [3, section 10].
- **Measuring first paid off.** The probes overturned the first hypothesis about run time and
  found the wall leak, which outlived the withdrawn v error as the reason for the rebuild [1].
- **What the metric reads was harder than the solver.** The half-cell offset [4], cell means
  against faces and the dropped ring row [6] each moved stored numbers, and a stencil shifted by
  one node passed every test until this branch (test 26b T1).

## Sources

1. `docs/reports/pressure_solver_probe.md`
2. `docs/reports/pressure_correction_step5.md`
3. `docs/reports/stopping_rule_evidence.md`, sections 6, 9 and 10
4. `docs/reports/cavity_self_convergence.md`, sections 5 and 9.1
5. `docs/reports/val001_revalidation_step7.md`, its addendum of 2026-09-30, and its rows in
   `benchmarks/results.jsonl`
6. `docs/reports/val002_revalidation_step8.md` and its rows in `benchmarks/results.jsonl`
7. `docs/reports/cavity_reference_marchi.md`

ADR-010, ADR-008 and ECR-001 are cited by name. Tests and coverage: `pytest tests/ -v
--cov=src --cov-branch` with JUnit and JSON output, at 0017e34, with `tests/test_validation.py`
rerun at 12eac95. Lint and map: `ruff format --check .`, `ruff check .` and
`scripts/gen_system_map.py --check`, at 12eac95. Pull requests: `gh pr list --state merged`.
