# ECR-003: The Pressure Correction Solver

**Project:** CFD Clean Room Simulation
**Change Request ID:** ECR-003
**Status:** Accepted 2026-10-06 by Alex, with ADR-013's six decisions each as ranked first; step 1 of section 8 opened the same day (branch `feature/ecr003-pressure-cg`), and its first commit enters section 5's text in `docs/SYSTEM.md`. Proposed 2026-10-06, with ADR-013, from the measurements in `docs/reports/pressure_solver_ecr003.md`. Revised the same day by the fix pass of prompt 36b on premise review 36 and `/cfd-test 36`, with the runs it cites added to the report as section 12. ADR-012 decision 5 (Alex, 2026-10-04) asked for this request, to land before ECR-002 step 5.
**Author:** Alex Moroz-Smietana (drafted in the builder sessions of prompts 36 and 36b)
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
- A cap that binds is silent, and the velocity-step rule cannot see it. In the probe room (T3
  outlets, ten momentum sweeps, alpha_velocity 0.5) the committed cap of 200 sweeps freezes the 80x30
  room: the velocity stops changing while the corrected faces carry 47% of the supply as net
  imbalance, and the velocity-step rule calls that converged (the report, section 8.2). The
  committed configuration itself (T0, one momentum sweep, alpha_velocity 0.7) does not get that far:
  it goes non-finite at outer 7,635 on 80x30 and never meets the velocity step on 200x75 (the report,
  section 12.5).

## 2. Root Cause Summary

**Jacobi's rate is set by the slowest mode, which the product's systems make very slow.** Each sweep
passes information one cell; the slowest mode of a mostly Neumann domain with a few small outlet
patches has `lambda_min(D^-1 A)` near 1e-5 on 200x75, so a correction needs hundreds of thousands of
sweeps. This is ADR-010's N^2 growth, measured.

**The stop is not a measure of the error.** A per-sweep change in pascals is the slow mode's error
times `(2/3) lambda_min` once that mode dominates, and the right-hand side's own size before. It
means 3e-5 on one system and 0.95 on another, and a cap stops it wherever it is, without a trace.

**How tightly the correction must be solved was not known.** The report's measurement 2 and the
controls of its section 12 answer it. A relative residual of 1e-2 keeps the outer iteration count
with every solver tried. From rest, looser levels with CG derail the start (1e-1 diverges), and 1e-2
reaches a different steady solution of the coarse 40x15 room, which has at least three. After 100
tight corrections 1e-1 and 1e-2 converge within 3% of the exact correction's count and near its solution;
3e-1 still diverges. Where the committed outlets hold a standing imbalance, the stopping rule's
per-cell continuity needs a tight correction at the end of the solve (section 3 below; the report,
sections 8 and 12).

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
the cells per side, 1.9 times per doubling (861 on 200x75, 1,643 on 400x150), so a correction costs
about 7.5 times as much on a mesh twice as fine each way (0.15 s to 1.1 s; the report, section 12.1).
**Selected** (ADR-013 decision 1, ranked first).

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

The alternatives not measured (red-black Gauss-Seidel and SOR, Chebyshev-accelerated Jacobi, CG
preconditioned by an incomplete Cholesky factorization) are listed in ADR-013 with the reason each
is set aside.

## 4. Recommended Change

Replace the weighted Jacobi loop in `PressureCorrector.correct` by conjugate gradients preconditioned
with the diagonal a_P, from p' = 0. A correction stops at the first of:

- the relative residual `||r||_2 / ||b||_2`, with `r = b + A p'`, below a new key `pressure_rtol`
  (default 1e-8);
- the residual's 2-norm below a rounding floor, `1e-13 F`, with F the stopping rule's flux scale (rho
  times the inflow, or on a closed domain rho times the largest prescribed boundary velocity times
  the longer side) and 1e-13 a module constant in `pressure.py`;
- `max_pressure_iter` iterations.

The first two are checked on the recursive residual and confirmed on the true one, formed once more
at exit. A correction that stops at the cap reports it (`PressureCorrection.reached_cap`); the
solver counts such corrections in each solve, logs the first and records the count. Under
velocity_step an outer iteration whose correction reached the cap does not stop the solve, so a run
the cap freezes ends at `max_simple_iter`, unconverged, with the count saying why; under
error_estimate the continuity conditions already refuse that state. On a closed domain the
right-hand side is projected onto the range before the solve, the stop reads the projected residual,
p' is pinned after, as now, and the constructor refuses a closed domain whose cells with an equation
are not one connected component. NumPy only; no new runtime dependency. Everything else in the
correction (the coefficients, the right-hand side, the outlet faces, the velocity and pressure
update) is unchanged.

The default 1e-8 rests on three measured reasons (ADR-013 decision 3; the report, sections 8 and 12):

1. **The start from rest.** At 1e-1 CG's first corrections leave 85% of the supply unbalanced and the
   40x15 and 80x30 rooms at real air diverge within three outer iterations. A tight start removes
   that for 1e-1 (2,739 outer iterations to the error-estimate stop, against direct's 2,822) but not
   for 3e-1, which diverges 25 outer iterations after the switch.
2. **Reproducibility.** The coarse laminar room has at least three steady solutions, two of them
   4.8e-4 and 0.041 m/s from the exact correction's at one cell, each a fixed point of the exact
   iteration with the same return faces shut.
   A looser correction from rest, like a different under-relaxation, can reach another of them; 1e-2
   reached the 0.041 m/s one. 1e-8 reaches the exact correction's, to 2.4e-8 m/s. That is
   reproducibility, not accuracy: none of the solutions is wrong.
3. **The standing imbalance at the outlets.** In the 80x30 room at Re 90 the committed outlet
   treatment returns the same imbalance every outer iteration while the pressure climbs, so the
   stopping rule's per-cell condition holds only if every correction leaves less than about 3e-7 of
   it. 1e-8 met it under both the committed outlets and T3; 1e-2 and 1e-4 never did. That is the
   committed outlet treatment's defect, which ECR-002 step 3 rebuilds.

A tight level costs little against a sweep, but not nothing against a looser level: on 200x75 1e-8
costs 1.2 times 1e-4 per correction, 2.7 times 1e-2 per outer iteration, and about nine times 1e-1
(the report, sections 7.2 and 9). The options ADR-013 decision 3 ranks below it, a relative level
with an absolute guard tied to `mass_imbalance_tol` and a tight start before a looser level, are
not measured in the outer loop or measured on one grid only, and the second alone fails the
standing-imbalance room. The default may be revisited after ECR-002 step 3 shows whether the
standing imbalance survives the rebuilt outlets.

## 5. Requirement Changes

### 5.1 Modified requirements

| ID | Current text | Proposed text | Reason | Verified by |
|----|-------------|---------------|--------|-------------|
| REQ-S08 | The pressure correction equation shall be solved using Jacobi iteration. (Clarified 2026-09-22: weighted Jacobi with w = 2/3 satisfies it.) | The pressure correction equation shall be solved by the conjugate gradient method preconditioned by its diagonal, from p' = 0, until the residual r = b + A p', where b is the discrete divergence of u* and r is the corrected faces' mass imbalance, meets `||r||_2 <= pressure_rtol ||b||_2` or a rounding floor `||r||_2 <= 1e-13 F` (F the stopping rule's flux scale), confirmed on the true residual at exit, or until the configured iteration cap, which the correction reports. On a closed domain b is projected onto the range of the operator before the solve and the stop reads the projected residual. | The requirement's rationale was per-cell data parallelism for the GPU: each Jacobi sweep updates every cell from the previous iterate, one thread per cell. A CG iteration keeps that for its two per-cell operations, the five-point product and the diagonal preconditioner, and adds three global reductions, which GPU libraries provide; the single array operation per iteration is traded for them. Weighted Jacobi needs 28,000 to 378,000 sweeps per correction on the product mesh to reach a relative residual of 1e-3 to 3e-3; CG reaches 1e-8 in about 860 iterations, 0.16 s (`docs/reports/pressure_solver_ecr003.md`, sections 7 and 9). The floor is in the text because on a closed domain it, not the relative level, can end a correction (section 12.2 there). | tests/test_pressure.py: CG against a dense solve on open and closed systems, the stop, the floor, the true-residual check, the cap and its flag; VAL-001 and VAL-002 under the change |
| REQ-S04 | (unchanged text) | No change to the text. Clarified: the corrected faces' per-cell imbalance equals the residual of the p' equation, so `pressure_rtol` bounds it relative to u*'s, and the stopping rule's per-cell condition can hold only where `pressure_rtol ||b||` is below `mass_imbalance_tol`. Under ADR-011 G that tolerance shrinks with the smallest cell volume (3.2e-9 kg/s per metre on 200x75, 8e-10 on 400x150), and the standing `||b||` at the product's converged state is not measured; the default 1e-8 met the condition on every case measured. | The standing imbalance at the outlets of the report's sections 8.3 and 12.4. | The report, sections 8 and 12; tests/test_pressure.py (the identity) |

### 5.2 Unchanged requirements

| ID | Note |
|----|------|
| REQ-S05 | SIMPLE stays; only the inner solve changes. |
| REQ-S06 | The NumPy implementation is the reference; the CUDA path of Phase 6 implements the same CG loop. |
| REQ-S01, S02, S03 | Unchanged texts; VAL-001 and VAL-002 are retaken under the change (criterion 2). |
| REQ-N03 | Unchanged, text and criterion: 1e-10 element-wise on every output array. A different summation order in CG's three sums moved a correction's faces by at most 1.6e-12 m/s and the converged 40x15 field by 1.1e-14 m/s, with the same iteration and outer counts (the report, section 12.2). The narrower risk, two loops stopping one iteration apart on a borderline correction, is handled by how Phase 6's test is run (section 10). |
| REQ-C01, C02 | `pressure_rtol` is configured and validated as every key is (type, range, NaN, bool); `pressure_tol` is refused with a message naming the change (ADR-013 decision 4). `max_pressure_iter` stays a required key. |

### 5.3 New requirements

None. The stop's definition moves into REQ-S08.

## 6. ADR Changes

| ADR | Action | Details |
|-----|--------|---------|
| ADR-013 | New | "Pressure Correction Solver: Jacobi-Preconditioned Conjugate Gradients." The design, Proposed, with its decisions for Alex first. |
| ADR-010 | Amend (note) | Decision on the weighted Jacobi sweep and REQ-S08's clarification of 2026-09-22 superseded by REQ-S08 as amended; the step 5 report's evidence for the weight stands as history. |
| ADR-012 | Note | Decision 5 answered: ECR-003, conjugate gradients (option B), measured against multigrid and two library methods. Section H's cost table is retaken in the report, section 9; the Annex 20 room at G's 216x72 in section 12.1. |

## 7. Affected Artifacts

Found by searching the tree at origin/main (023b8f6) for `pressure_tol`, `max_pressure_iter`,
`JACOBI_WEIGHT`, `sweep`, `sweeps`, `pressure_sweeps`, `staggered-jacobi`, `SOLVER_PARAMETERS`,
`RULE_VERSION` and every saved-solve reuse (`.exists()`) in `scripts/`. Every file below is checked in
step 1, and every one given an edit changes there, so no commit leaves a consumer reading the old
solver's names or results.

### 7.1 Source code and scripts

| Artifact | Impact |
|----------|--------|
| `src/pressure.py` | `correct` solves by Jacobi-preconditioned CG with the stop of section 4; `JACOBI_WEIGHT` and the weighted `sweep` leave (ADR-013 decision 4); `PRESSURE_SOLVER_VERSION = 2` and `RESIDUAL_FLOOR` added; the constructor reads `pressure_rtol` and checks a closed domain is one component; `PressureCorrection.sweeps` renamed `iterations`, and `reached_cap` added. |
| `src/config.py` | `pressure_rtol` in [1e-10, 1), validated (below about 1e-12 the true residual of the product's first correction cannot be reached, so every such correction would run to the cap; test 36b); `pressure_tol` refused with a message naming `pressure_rtol`, and dropped from `_SOLVER_KEYS`; `max_pressure_iter` stays a required positive integer, now the CG cap (it has no default today and gets none). |
| `src/solver_staggered.py` | `last_pressure_sweeps` renamed `last_pressure_iterations`; `pressure_cap_hits` counted per solve, reset with the timers; under velocity_step an outer iteration whose correction reached the cap does not stop the solve; a warning at the first capped correction. |
| `src/stopping.py` | `IterationState.pressure_sweeps` renamed `pressure_iterations`. |
| `configs/*.yaml` (three), `validation/transport_cases.py` | `pressure_rtol: 1.0e-8` in place of `pressure_tol`; `max_pressure_iter: 5000` in place of the Jacobi caps (200, 500, 2,000), which mean nothing for CG. |
| `scripts/benchmark.py` | `SOLVER_PARAMETERS` lists `pressure_rtol` in place of `pressure_tol`: `solver_parameters` reads each name with `getattr`, so the old name raises at the first row. `STAGGERED_METHOD` becomes a new label, `staggered-cg`; `staggered-jacobi` is retired as `collocated-jacobi` was, kept known so its 20 stored rows still summarize (rows group by method, so old and new rows do not mix) and refused by `run_case`. `CELL_UPDATE_DEFINITIONS` gains the new label's work definition: one stencil evaluation per cell with an equation per CG iteration, plus vector operations. The work counter reads `pressure_iterations`. Each row records `pressure_cap_hits`. |
| `scripts/val001_order.py` | `reuse_key` builds on `solver_parameters`, so the key carries `pressure_rtol` after the change and a Jacobi-era saved Poiseuille solve no longer matches: it is re-solved, not reused. No edit of its own. |
| `scripts/self_convergence.py` | `METHODS` follows the new label, and so do the saved file names built from it (`{method}_{n}.npz`) and the six other lines that spell `staggered-jacobi` out in a file name, a docstring or a metric key; `solve_and_save` and `solve_tight` reuse any file that exists, so the new names are what keeps them from serving Jacobi-era cavity fields. `tight_field` falls back to `results/tester21b/staggered_{n}_tight.npz`, Jacobi-era fields with no solver identity; the fallback is removed, and the 60x60 and 100x100 solves it served are retaken under the new label when the Marchi comparison is next run. |
| `scripts/stopping_probe.py` | Reads no configuration pressure key (the draft of this table said it did). Its pressure dependencies: `instrument` appends `result.sweeps` (renamed), and four saved-solve reuses carry no solver identity (the fourth, `control`, found by test 36b). `solve_truth` returns any existing truth file; the tight truth is re-solved only when its stored tolerance differs; `verify_rule` reuses `{case}_rule.npz` when `rule_parameters` match, and those are `[scale, flux, *tols, RATE_WINDOW, RULE_VERSION]`. `control` reuses `{name}_control.npz` whenever it exists, and compares it bitwise with the truth's snapshot (two such files exist in the main tree today). `rule_parameters` appends `PRESSURE_SOLVER_VERSION`, and the two truth files and the control file store it and are re-solved when it differs or is missing. |
| `scripts/view_field.py` | `STAGGERED_METHOD` follows the new label. Saved viewer files carry their method, and `render` reads it, so old files still render under `staggered-jacobi`. |

### 7.2 Tests

| Artifact | Impact |
|----------|--------|
| `tests/test_pressure.py` | CG against `numpy.linalg.solve` on small open and closed systems; the relative residual reached, the floor, the true-residual check at exit, the cap and `reached_cap`; the identity between the residual and the corrected faces' imbalance; the projection on a closed domain; the refusal of a closed domain in two components. The Jacobi-specific tests (the -1 eigenvalue, the weight, `sweep`'s argument checks) are removed with the sweep (ADR-013 decision 4); its `_config` helper takes `pressure_rtol`. |
| `tests/test_config.py` | The key, its range and refusals; `pressure_tol` refused; its base fixture and the five full solver mappings of the relaxation-factor tests move to `pressure_rtol`. |
| `tests/test_boundary_concentration.py`, `tests/test_boundary_registry.py`, `tests/test_boundary_staggered.py`, `tests/test_mesh.py`, `tests/test_momentum.py`, `tests/test_particles.py`, `tests/test_solver_transport.py`, `tests/test_staggered.py`, `tests/test_benchmark_work.py` | Each builds a configuration mapping whose solver block sets `"pressure_tol"`, which step 1 refuses at load, so every test on those fixtures fails until the key becomes `pressure_rtol`. `test_benchmark_work` also builds `IterationState` with the renamed count and checks the work definition, which the new label changes. |
| `tests/test_validation.py` | Reads `pressure_tol` and the Jacobi caps from both validation case files; reads `pressure_rtol` and 5,000. |
| `tests/test_solver_staggered.py`, `tests/test_solver_selection.py`, `tests/test_stopping.py` | The renames; `test_solver_selection` also names the method label twice; `test_solver_staggered` adds the cap count and the velocity_step refusal on a capped correction. |
| `tests/test_benchmark.py` | The four `run_case` calls with `"staggered-jacobi"` take the new label; the summary test of stored rows keeps the old label, as it keeps `collocated-jacobi`; `run_case` refuses the retired label. |
| `tests/test_self_convergence.py`, `tests/test_view_field.py` | The label in the saved file names they build or read; `test_self_convergence` loses the `TESTER_DIR` fallback case. |
| `tests/test_stopping_probe.py` | `rule_parameters` changes with `PRESSURE_SOLVER_VERSION`, as `test_rule_parameters_change_with_the_rule_version` shows for `RULE_VERSION`; the truth files and the control file are re-solved when their stored version differs. |
| `tests/test_val001_order.py` | No edit: its key tests read `reuse_key`, which follows `SOLVER_PARAMETERS`. |
| `tests/test_conservation.py`, `tests/test_constancy.py` | Read VAL-001's face velocities from a solve, which change beyond rounding; the transport criteria are re-checked, not re-set. |

### 7.3 Documentation and records

| Artifact | Impact |
|----------|--------|
| `docs/SYSTEM.md` | REQ-S08 amended and REQ-S04 clarified (section 5); the `pressure.py`, `solver_staggered.py` and `stopping.py` contracts (ADR-013 section D); cascade rows; generated regions regenerate. |
| `docs/PROJECT_PLAN.md`, `docs/STATUS.md` | The request and its steps; ECR-002 step 5's dependency met on closure; Phase 6's deliverable `csolver/pressure_solve.cu` (PROJECT_PLAN line 308, "CUDA kernel for Jacobi pressure correction") becomes a CG kernel. |
| `docs/ADR/ADR-013-pressure-solver.md` | Create (this pass, Proposed). |
| `docs/reports/pressure_solver_ecr003.md` | Create (this pass): the evidence. |
| `benchmarks/results.jsonl` | Unchanged: its 42 rows, 20 of them `staggered-jacobi`, keep their labels and their `pressure_tol`; step 2's rows carry the new label and `pressure_rtol`. |
| `results/builder36/common36.py` (untracked) | Criterion 3 runs through it: `set_solver` writes `block["pressure_tol"] = p_tol`, which step 1 refuses. Step 1 changes that line to `pressure_rtol` and the cap to 5,000, and its report reproduces the edited probe in an appendix, as this report reproduces the original, with `outlet33b.py` and `frozen34.py` unchanged byte copies. |

### 7.4 Cascade impact

From SYSTEM.md section 3: `config.py -> pressure`: the new key and the refused one. `pressure.py ->
solver_staggered`: `PressureCorrection` keeps its shapes; its count field is renamed and a flag
added. `solver_staggered.py -> the harness, the viewer, scripts/stopping_probe.py,
tests/test_conservation.py, tests/test_constancy.py`: `last_pressure_sweeps` renamed, the cap count
added, the velocity_step stop refused on a capped correction; every face field changes beyond
rounding. `stopping.py -> solver_staggered, scripts/benchmark.py`: `IterationState`'s field renamed
(`scripts/self_convergence.py` and `scripts/stopping_probe.py` use the class as an annotation only,
and `scripts/val001_order.py` does not name it).
`scripts/benchmark.py -> scripts/val001_order.py`: the parameter list, and with it the reuse key.
The method label `staggered-jacobi -> staggered-cg` runs through `benchmark.py`, `view_field.py`,
`self_convergence.py` and their tests. Saved solves keyed without a solver identity
(`self_convergence.py`, `stopping_probe.py`) are invalidated by the label in their file names or by
`PRESSURE_SOLVER_VERSION` in their keys. Cross-cutting: array layout, units and coordinates
unchanged. The CUDA port (Phase 6) targets the CG loop.

## 8. Implementation Plan

Each step is one pull request through `/cfd-review` and `/cfd-test`.

| Step | Deliverable | Depends on |
|------|-------------|------------|
| 1 | `src/pressure.py`: the CG solve, its stop, floor, true-residual check, cap flag and component check; `src/config.py`: `pressure_rtol` and the refused `pressure_tol`; the count field renamed through `solver_staggered.py` and `stopping.py`, with the cap count and the velocity_step refusal; every configuration file and the transport stand-ins moved to the new key and cap; every consumer of section 7.1 and every test of section 7.2, the method label and the saved-solve keys included, so no harness row, saved field or truth file written after this step can carry the old solver's label or be reused as the new solver's. The 40x15 room at real air and the 80x30 room at Re 90 run with the built code and compared with the report's `pcg_1e-8` runs (criterion 3). | ADR-013 decisions 1, 3, 4 |
| 2 | The laminar baseline retaken (ECR-002 criterion 1): `val001_80x40`, `val001_80x40_stretched` and `val002_80x80` rerun under the new label, their rows the new baseline; VAL-001 and VAL-002 against their criteria; the transport gate tests re-checked. | step 1 |
| 3 | Records: ADR-013 accepted with planned against built, SYSTEM.md, PROJECT_PLAN.md, STATUS.md, this request closed. | steps 1, 2 |

ECR-002 step 5, the first that solves the 200x75 room to tolerance, waits for step 2.

## 9. Acceptance Criteria

1. **The solve.** `PressureCorrector.correct` solves by Jacobi-preconditioned CG; against a dense
   solve on small open and closed systems the result agrees to the configured tolerance; the stop,
   the floor, the true-residual check and the cap behave as REQ-S08 states, and a capped correction
   reports `reached_cap`; the corrected faces' imbalance equals the residual to rounding; under
   velocity_step a capped correction does not stop the solve, and the count is recorded. Method:
   test.
2. **The laminar baseline.** `val001_80x40`, `val001_80x40_stretched` and `val002_80x80` meet REQ-S02
   and REQ-S03 under the change and stop by `error_estimate_and_continuity`; their harness rows under
   the new method label are ECR-002 criterion 1's baseline from then on (ECR-002 section 9, criterion
   1, "if ECR-003 lands first"). The transport gate tests pass. Method: test and harness rows.
3. **The measured cases reproduce.** With the built code at the default: the 40x15 room at real air
   under T3 with ten momentum sweeps meets the velocity-step stop at 1,209 and the error-estimate stop
   at 2,822 outer iterations, within 1%; the 80x30 room at a thousand times air's viscosity at 233 and
   588. Both through the report's probe-only subclass for T3 and the sweeps
   (`results/builder36/common36.py` with `outlet33b.py` and `frozen34.py`), edited as section 7.3
   states and reproduced in the step's report. Method: report.
4. **The cost.** One correction on each of the report's three captured 200x75 systems at the default
   in under 0.5 s on the report's machine, or the harness's equivalent on another. Method: report.
5. **Records.** SYSTEM.md, PROJECT_PLAN.md, STATUS.md, ADR-013 (accepted, with planned against built)
   and this request updated and committed. Method: inspection.
6. **Review.** Each step's pull request carries its `/cfd-review` and `/cfd-test` reports. Method:
   inspection.

## 10. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| CG's iterations grow with the cells per side, so a correction costs about 7.5 times as much on a mesh twice as fine each way (1.1 s on 400x150; the report, section 12.1) | Certain | Medium on a finer mesh | MG-PCG is the measured successor (ADR-013 decision 1): the same CG loop, a multigrid preconditioner, its setup rewritten |
| Changing the solver changes laminar results beyond rounding, and the coarse 40x15 room has at least three steady solutions up to 0.041 m/s apart at one cell, which a change of path can choose between | Measured | Low | The baseline is retaken (criterion 2); the default reproduces the exact correction's solution; no criterion holds a field to the spread of the solutions |
| The outlets hold a standing imbalance under the committed treatment (the pressure rising each outer iteration), which a loose correction can never clear | Measured on 80x30 at Re 90, under the committed outlets and T3 | Medium | The default 1e-8; the outlet treatment itself is ECR-002 step 3's |
| The standing `||b||` on the product mesh is larger than `mass_imbalance_tol / 1e-8`, so the default does not hold the per-cell condition there | Unknown (not measured) | Medium: a run to the outer cap under error_estimate | ECR-002 step 5 measures it; ADR-013 decision 3 option (2), the absolute guard, is the remedy if it does |
| Phase 6's CUDA loop stops a borderline correction one iteration apart from NumPy's, and one CG iteration at 1e-8 moves the corrected faces by 4e-10 to 6e-9 m/s, above REQ-N03's 1e-10 | Possible in Phase 6 | Low | The equivalence test runs both loops for the same iteration count per correction and compares every output array at 1e-10; agreement of the stop tests is counted separately. A different summation order alone moved the faces by at most 1.6e-12 m/s (the report, section 12.2) |
| `max_pressure_iter` reached, as the committed cap of 200 is today in the probe room, freezing it while velocity_step reports convergence | Low at 5,000 (1,643 needed on 400x150) | High (the frozen state of the report's section 8.2) | `reached_cap` on the correction, the count per solve in every harness row and a warning; under velocity_step a capped correction cannot stop the solve; under error_estimate the continuity conditions refuse that state. The product configuration runs velocity_step until ADR-011 decision 6 and ECR-002's VAL-018 move it |
| The product's outer count is unknown (no laminar solve converged on 80x30 or 200x75) | Certain | Medium | ECR-002 step 5 measures it; the cost per 1,000 outer iterations is known (the report, section 9) |

## 11. Approval

By signing below, approvers confirm that the problem statement is accurate, the selected option
is appropriate given the alternatives, the requirement and ADR changes are consistent with the
architecture, and the acceptance criteria are sufficient to close the change.

| Role | Name | Approval | Date |
|------|------|----------|------|
| Author | Alex Moroz-Smietana | Approved, with ADR-013's six decisions each as ranked first | 2026-10-06 |
| Reviewer | Claude | Premise review 36, `/cfd-test 36` and `/cfd-test 36b` done; fix pass 36b and the orchestrator's text pass applied | 2026-10-06 |

---

## Document History

| Date | Change | Author |
|------|--------|--------|
| 2026-10-06 | Proposed, with ADR-013 and the evidence report `docs/reports/pressure_solver_ecr003.md`. ADR-013's decisions open. Premise review and `/cfd-test 36` to follow. | Alex Moroz-Smietana |
| 2026-10-06 | Fix pass 36b on premise review 36 and test 36: section 7 rebuilt from a search of the tree (the eight test fixtures, the harness parameter list, the method label and its readers, the saved solves without a solver identity, the probe criterion 3 runs through); REQ-N03 kept as written, with the narrower risk and Phase 6's test stated; the cap reported and refused as a velocity_step stop; the default's reasons restated to the evidence, with the guard and tight-start options; the rounding floor, the true-residual check and the one-component check in the design; the growth per doubling measured. | Alex Moroz-Smietana |
| 2026-10-06 | `/cfd-test 36b`'s text findings applied by the orchestrator (the fourth saved-solve reuse in `stopping_probe.py`, `pressure_rtol`'s lower bound, the Annex 20 cost, wording). ADR-013's six decisions taken by Alex, each as ranked first. | Alex Moroz-Smietana |
| 2026-10-06 | Accepted by Alex. Step 1 opened (prompt 37): REQ-S08's amended text and REQ-S04's clarification entered in `docs/SYSTEM.md` in the step's first commit, as ECR-002 step 0 did for that request; the solve, the keys, the consumers of section 7 and criteria 1, 3 and 4 follow in the same pull request. | Alex Moroz-Smietana |
