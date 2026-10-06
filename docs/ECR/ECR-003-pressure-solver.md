# ECR-003: The Pressure Correction Solver

**Project:** CFD Clean Room Simulation
**Change Request ID:** ECR-003
**Status:** Proposed 2026-10-06, with ADR-013, from the measurements in `docs/reports/pressure_solver_ecr003.md`. A premise review and `/cfd-test 36` follow. ADR-012 decision 5 (Alex, 2026-10-04) asked for this request, to land before ECR-002 step 5.
**Author:** Alex Moroz-Smietana (drafted in the builder session of prompt 36)
**Approver(s):** Alex Moroz-Smietana, Claude (pair)
**Date Raised:** 2026-10-06
**Phase Affected:** Phase 2 (the Navier-Stokes solver's pressure correction, REQ-S08) and, through ECR-002's dependency on it, Phase 3's product case. Every laminar result changes beyond rounding, so the laminar baseline ECR-002 criterion 1 compares against is retaken under this change.

---

## 1. Problem Statement

The pressure correction is solved by weighted Jacobi iteration (REQ-S08), stopped when the largest
change of p' in one sweep falls below `pressure_tol` pascals or after `max_pressure_iter` sweeps.
Measured on the systems the product room builds on 200x75 under the T3 outlets with ten momentum
sweeps per outer iteration (the report, sections 6 and 7):

- The systems are symmetric and positive definite, and stiff: the smallest eigenvalue of `D^-1 A`
  is 3.7e-6 to 1.5e-5, so the sweep's slowest mode shrinks by 1 - 2.5e-6 to 1 - 1e-5 per sweep.
- One correction to the committed 1e-6 Pa takes 28,000 to 378,000 sweeps, 6 to 79 s, and reaches a
  relative residual of only 1e-3 to 3e-3. Reaching 1e-6 relative takes 760,000 to 3.6 million
  sweeps. The product report's 27,408 sweeps (its section 5) are reproduced exactly on its own
  system, the room as committed at outer 0; the T3 room with ten sweeps builds stiffer ones.
- A steady product solve at today's stop would take 10 to 42 hours over the 3,000 to 13,000 outer
  iterations ADR-012 H assumed (the report, section 9).
- Today's stop measures no fixed accuracy. Its quantity, the largest weighted change in one sweep,
  falls below the tolerance after one to three sweeps once the right-hand side is small, and then
  leaves it almost untouched (relative residual 0.95 on the channel at outer 569 and the cavity at
  outer 1,000). On a large right-hand side it runs to 3e-5.
- At the committed cap of 200 sweeps the 80x30 room freezes: the velocity stops changing while the
  corrected faces carry 47% of the supply as net imbalance, and the velocity-step rule calls that
  converged (the report, section 8.2).

## 2. Root Cause Summary

**Jacobi's rate is set by the slowest mode, which the product's systems make very slow.** Each sweep
passes information one cell; the slowest mode of a mostly Neumann domain with a few small outlet
patches has `lambda_min(D^-1 A)` near 1e-5 on 200x75, so a correction needs hundreds of thousands of
sweeps. This is ADR-010's N^2 growth, measured.

**The stop is not a measure of the error.** A per-sweep change in pascals is the slow mode's error
times `(2/3) lambda_min` once that mode dominates, and the right-hand side's own size before. It
means 3e-5 on one system and 0.95 on another, and a cap stops it wherever it is.

**How tightly the correction must be solved was not known.** The report's measurement 2 answers it:
a relative residual of 1e-2 keeps the outer iteration count with every solver tried, but not the
converged field, and not the stopping rule's continuity conditions where the outlets hold a standing
imbalance (section 3 below).

## 3. Options Considered

Measured on one correction (the report, section 7) and in the outer loop (section 8), each against
the exact correction (SciPy's SuperLU, a reference). The relative residual `||b + A p'||_2 / ||b||_2`
is the measure every candidate stopped on; it is also the corrected faces' mass imbalance relative
to u*'s, cell by cell (the report, section 2.4).

### 3.1 Option A: Weighted Jacobi with a relative-residual stop

Keeps REQ-S08's text. At 1e-8 it needs millions of sweeps per correction on 200x75 (5 million did
not reach it on the outer-1 system). **Not taken.**

### 3.2 Option B: Jacobi-preconditioned conjugate gradients, NumPy

One five-point product, the diagonal preconditioner and three reductions per iteration; no setup.
On 200x75: 852 to 861 iterations and 0.16 s to 1e-8, 0.13 s to 1e-4, about 0.05 s to 1e-2. Converges on
every captured system, the singular cavity included, with the right-hand side projected onto the
range. A steady product solve at 1e-8: 9 to 39 minutes over the outer range. Iterations grow with
the cells per side (5.3 times from 40x15 to 200x75). **Selected** (ADR-013 decision 1, ranked first).

### 3.3 Option C: Geometric multigrid with Galerkin coarse operators, NumPy

V-cycles with the present weighted sweep as smoother, standalone or as the preconditioner of CG.
Three to four cycles to 1e-2 and 10 to 11 CG iterations to 1e-8 on 200x75, but 0.23 s per
correction, of which 120 to 175 ms is the probe's setup in Python. The best scaling under
refinement; needs its setup written as a stencil formula or in C to compete. **The successor** if a
finer mesh makes B too slow (ADR-013 decision 1, ranked fourth today).

### 3.4 Option D: Algebraic multigrid, pyamg

Ruge-Stuben with CG: 0.07 s per correction to 1e-8 on 200x75, 4.5 to 20 minutes per steady solve.
Adds SciPy and pyamg as runtime dependencies; its setup is sequential; its preconditioned CG fails
on the singular cavity at tight levels under its defaults (the report, section 7.5). **Not taken**
(ADR-013 decisions 1 and 2, ranked third).

### 3.5 Option E: A sparse direct solve, SciPy SuperLU

Not one of the candidates ADR-012 H named; measured as the reference. Exact, 0.034 s per correction
on 200x75, 2.5 to 11 minutes per steady solve, with no tolerance to choose. Adds SciPy as a runtime
dependency and gives up the per-cell data parallelism REQ-S08 was written for. **Not taken**
(ADR-013 decisions 1 and 2, ranked second).

## 4. Recommended Change

Replace the weighted Jacobi loop in `PressureCorrector.correct` by conjugate gradients preconditioned
with the diagonal a_P, from p' = 0, stopped when the relative residual `||b + A p'||_2 / ||b||_2`
falls below a new key `pressure_rtol` (default 1e-8), when the residual's 2-norm falls below a
rounding floor, or at `max_pressure_iter` iterations. On a closed domain the right-hand side is
projected onto the range before the solve and p' pinned after, as now. NumPy only; no new runtime
dependency. Everything else in the correction (the coefficients, the right-hand side, the outlet
faces, the velocity and pressure update) is unchanged.

The default is tight because measurement 2 found looser levels unsafe in three ways (the report,
section 8.4): 1e-1 diverges with this solver, 1e-2 converges the 40x15 room to a different steady
state 0.041 m/s away at one cell, and in the 80x30 room at Re 90, where the outlets hold a standing
imbalance, no level looser than about 2.5e-7 lets the stopping rule's per-cell continuity condition
hold. A tight level costs little with CG: on 200x75 1e-8 costs 1.2 times 1e-4.

## 5. Requirement Changes

### 5.1 Modified requirements

| ID | Current text | Proposed text | Reason | Verified by |
|----|-------------|---------------|--------|-------------|
| REQ-S08 | The pressure correction equation shall be solved using Jacobi iteration. (Clarified 2026-09-22: weighted Jacobi with w = 2/3 satisfies it.) | The pressure correction equation shall be solved by the conjugate gradient method preconditioned by its diagonal, from p' = 0, to a relative residual `||b + A p'||_2 <= pressure_rtol ||b||_2`, where b is the discrete divergence of u* and the residual is the corrected faces' mass imbalance, or to the configured iteration cap. On a closed domain the right-hand side is projected onto the range of the operator before the solve. | The requirement's rationale was per-cell data parallelism for the GPU: each Jacobi sweep updates every cell from the previous iterate, one thread per cell. A CG iteration keeps that for its two per-cell operations, the five-point product and the diagonal preconditioner, and adds three global reductions, which GPU libraries provide; the single array operation per iteration is traded for them. Weighted Jacobi needs 28,000 to 378,000 sweeps per correction on the product mesh to reach a relative residual of 1e-3 to 3e-3; CG reaches 1e-8 in about 860 iterations, 0.16 s (`docs/reports/pressure_solver_ecr003.md`, sections 7 and 9). | tests/test_pressure.py: CG against a dense solve on open and closed systems, the stop, the cap; VAL-001 and VAL-002 under the change |
| REQ-S04 | (unchanged text) | No change to the text. Clarified: the corrected faces' per-cell imbalance equals the residual of the p' equation, so `pressure_rtol` bounds it relative to u*'s, and the stopping rule's per-cell condition can hold only where `pressure_rtol ||b||` is below `mass_imbalance_tol`; the default 1e-8 met it on every case measured. | The standing imbalance at the outlets of the report's section 8.3. | The report, section 8; tests/test_pressure.py (the identity) |

### 5.2 Unchanged requirements

| ID | Note |
|----|------|
| REQ-S05 | SIMPLE stays; only the inner solve changes. |
| REQ-S06 | The NumPy implementation is the reference; the CUDA path of Phase 6 implements the same CG loop. |
| REQ-S01, S02, S03 | Unchanged texts; VAL-001 and VAL-002 are retaken under the change (criterion 2). |
| REQ-N03 | Equivalence of the CUDA loop to the NumPy one at 1e-10; the order of the reductions differs on a GPU, so the equivalence is judged at the default tolerance (section 10). |
| REQ-C01, C02 | `pressure_rtol` is configured and validated as every key is (type, range, NaN, bool); `pressure_tol` is refused with a message naming the change (ADR-013 decision 4). |

### 5.3 New requirements

None. The stop's definition moves into REQ-S08.

## 6. ADR Changes

| ADR | Action | Details |
|-----|--------|---------|
| ADR-013 | New | "Pressure Correction Solver: Jacobi-Preconditioned Conjugate Gradients." The design, Proposed, with its decisions for Alex first. |
| ADR-010 | Amend (note) | Decision on the weighted Jacobi sweep and REQ-S08's clarification of 2026-09-22 superseded by REQ-S08 as amended; the step 5 report's evidence for the weight stands as history. |
| ADR-012 | Note | Decision 5 answered: ECR-003, conjugate gradients (option B), measured against multigrid and two library methods. Section H's cost table is retaken in the report, section 9. |

## 7. Affected Artifacts

### 7.1 Source code

| Artifact | Impact |
|----------|--------|
| `src/pressure.py` | `correct` solves by Jacobi-preconditioned CG; `JACOBI_WEIGHT` and the weighted `sweep` leave the correction (kept or removed, ADR-013 decision 4); the stop reads `pressure_rtol`; `PressureCorrection.sweeps` becomes the CG iteration count, renamed `iterations` (cascade below). |
| `src/config.py` | `pressure_rtol` in (0, 1), validated; `pressure_tol` refused; `max_pressure_iter` the CG cap, its default raised. |
| `src/solver_staggered.py` | `last_pressure_sweeps` follows the rename; nothing else. |
| `src/stopping.py` | `IterationState.pressure_sweeps` follows the rename. |
| `configs/*.yaml`, `validation/transport_cases.py` | `pressure_rtol` in place of `pressure_tol`; the caps re-set (the validation files' 500 and 2,000 Jacobi sweeps mean nothing for CG). |
| `scripts/benchmark.py` | A new method label for the staggered solver with CG (the stored `staggered-jacobi` rows keep theirs) and its work definition: a CG iteration is one stencil evaluation per cell with an equation, plus vector operations; recorded with every row as `CELL_UPDATE_DEFINITIONS` records it today. |
| `scripts/stopping_probe.py` | Reads the configuration's pressure keys; follows the rename. |

### 7.2 Tests

| Artifact | Impact |
|----------|--------|
| `tests/test_pressure.py` | CG against `numpy.linalg.solve` on small open and closed systems; the relative residual reached and the cap; the identity between the residual and the corrected faces' imbalance; the projection on a closed domain. The Jacobi-specific tests (the -1 eigenvalue, the weight) are removed or kept with the sweep (ADR-013 decision 4). |
| `tests/test_config.py` | The key, its range and refusals; `pressure_tol` refused. |
| `tests/test_solver_staggered.py`, `tests/test_solver_selection.py`, `tests/test_validation.py`, `tests/test_benchmark_work.py`, `tests/test_stopping.py` | The rename and the work definition; the validation cases' configured caps. |
| `tests/test_conservation.py`, `tests/test_constancy.py` | Read VAL-001's face velocities from a solve, which change beyond rounding; the transport criteria are re-checked, not re-set. |

### 7.3 Documentation

| Artifact | Impact |
|----------|--------|
| `docs/SYSTEM.md` | REQ-S08 amended and REQ-S04 clarified (section 5); the `pressure.py` contract; cascade rows; generated regions regenerate. |
| `docs/PROJECT_PLAN.md`, `docs/STATUS.md` | The request and its steps; ECR-002 step 5's dependency met on closure. |
| `docs/ADR/ADR-013-pressure-solver.md` | Create (this pass, Proposed). |
| `docs/reports/pressure_solver_ecr003.md` | Create (this pass): the evidence. |

### 7.4 Cascade impact

From SYSTEM.md section 3: `config.py -> pressure`: the new key and the refused one. `pressure.py ->
solver_staggered`: `PressureCorrection` keeps its shapes; its count field is renamed.
`solver_staggered.py -> the harness, the viewer, scripts/stopping_probe.py, tests/test_conservation.py,
tests/test_constancy.py`: `last_pressure_sweeps` renamed; every face field changes beyond rounding.
`stopping.py -> solver_staggered, scripts/benchmark.py, scripts/self_convergence.py,
scripts/stopping_probe.py, scripts/val001_order.py`: `IterationState`'s field renamed. Cross-cutting:
array layout, units and coordinates unchanged. The CUDA port (Phase 6) targets the CG loop.

## 8. Implementation Plan

Each step is one pull request through `/cfd-review` and `/cfd-test`.

| Step | Deliverable | Depends on |
|------|-------------|------------|
| 1 | `src/pressure.py`: the CG solve and its stop; `src/config.py`: `pressure_rtol`, the refused `pressure_tol`, the cap's default; every configuration file and the transport stand-ins moved to the new key; the count field renamed through `solver_staggered.py` and `stopping.py`. Unit tests of section 7.2. The 40x15 room at real air and the 80x30 room at Re 90 run with the built code and compared with the report's `pcg_1e-8` runs (criterion 3). | ADR-013 decisions 1, 3, 4 |
| 2 | The laminar baseline retaken (ECR-002 criterion 1): the harness's new method label and work definition; `val001_80x40`, `val001_80x40_stretched` and `val002_80x80` rerun, their rows the new baseline; VAL-001 and VAL-002 against their criteria; the transport gate tests re-checked. | step 1 |
| 3 | Records: ADR-013 accepted with planned against built, SYSTEM.md, PROJECT_PLAN.md, STATUS.md, this request closed. | steps 1, 2 |

ECR-002 step 5, the first that solves the 200x75 room to tolerance, waits for step 2.

## 9. Acceptance Criteria

1. **The solve.** `PressureCorrector.correct` solves by Jacobi-preconditioned CG; against a dense
   solve on small open and closed systems the result agrees to the configured tolerance; the stop
   and the cap behave as REQ-S08 states; the corrected faces' imbalance equals the residual to
   rounding. Method: test.
2. **The laminar baseline.** `val001_80x40`, `val001_80x40_stretched` and `val002_80x80` meet REQ-S02
   and REQ-S03 under the change and stop by `error_estimate_and_continuity`; their harness rows under
   the new method label are ECR-002 criterion 1's baseline from then on (ECR-002 section 9, criterion
   1, "if ECR-003 lands first"). The transport gate tests pass. Method: test and harness rows.
3. **The measured cases reproduce.** With the built code at the default: the 40x15 room at real air
   under T3 with ten momentum sweeps meets the velocity-step stop at 1,209 and the error-estimate stop
   at 2,822 outer iterations, within 1%; the 80x30 room at a thousand times air's viscosity at 233 and
   588. Both through a probe-only subclass for T3 and the sweeps, as the report's were. Method: report.
4. **The cost.** One correction on each of the report's three captured 200x75 systems at the default
   in under 0.5 s on the report's machine, or the harness's equivalent on another. Method: report.
5. **Records.** SYSTEM.md, PROJECT_PLAN.md, STATUS.md, ADR-013 (accepted, with planned against built)
   and this request updated and committed. Method: inspection.
6. **Review.** Each step's pull request carries its `/cfd-review` and `/cfd-test` reports. Method:
   inspection.

## 10. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| CG's iterations grow with the cells per side, so a correction costs about four times more per refinement in each direction | Certain | Medium on a finer mesh | MG-PCG is the measured successor (ADR-013 decision 1): the same CG loop, a multigrid preconditioner, its setup rewritten |
| Changing the solver changes laminar results beyond rounding: the 40x15 room has a nearly neutral direction at one cell along which converged runs land differently with the path (up to 4.8e-4 m/s with the exact correction and another sweep count or under-relaxation) | Measured | Low | The baseline is retaken (criterion 2); the report gives the floor |
| The outlets hold a standing imbalance under the committed treatment (the pressure rising each outer iteration), which a loose correction can never clear | Measured on 80x30 at Re 90 | Medium | The default 1e-8; the outlet treatment itself is ECR-002 step 3's |
| A GPU's reductions sum in another order, so the CUDA loop's iterates differ in the last bits from NumPy's (REQ-N03) | Certain in Phase 6 | Low | Equivalence judged on the corrected faces at the default tolerance |
| `max_pressure_iter` reached silently, as the committed cap of 200 is today | Medium | High (the frozen state of the report's section 8.2) | The default cap set from measurement with margin; the count per correction recorded by the harness; `error_estimate`'s continuity conditions as the guard |
| The product's outer count is unknown (no laminar solve converged on 80x30 or 200x75) | Certain | Medium | ECR-002 step 5 measures it; the cost per 1,000 outer iterations is known (the report, section 9) |

## 11. Approval

By signing below, approvers confirm that the problem statement is accurate, the selected option
is appropriate given the alternatives, the requirement and ADR changes are consistent with the
architecture, and the acceptance criteria are sufficient to close the change.

| Role | Name | Approval | Date |
|------|------|----------|------|
| Author | Alex Moroz-Smietana | Pending | |
| Reviewer | Claude | Pending: premise review and `/cfd-test 36` | |

---

## Document History

| Date | Change | Author |
|------|--------|--------|
| 2026-10-06 | Proposed, with ADR-013 and the evidence report `docs/reports/pressure_solver_ecr003.md`. ADR-013's decisions open. Premise review and `/cfd-test 36` to follow. | Alex Moroz-Smietana |
