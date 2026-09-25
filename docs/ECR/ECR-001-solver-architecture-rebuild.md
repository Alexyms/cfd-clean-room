# ECR-001: Solver Architecture Rebuild (Staggered Grid, Non-Uniform Mesh, QUICK Advection)

**Project:** CFD Clean Room Simulation
**Change Request ID:** ECR-001
**Status:** Approved (last amended 2026-09-24, see Document History)
**Author:** Alex Moroz-Smietana
**Approver(s):** Alex Moroz-Smietana, Claude (pair)
**Date Raised:** 2026-04-16
**Phase Affected:** Phase 2 (Navier-Stokes Solver), in progress

---

## 1. Problem Statement

*Erratum 2026-09-22: every v figure in this section was measured against a v reference that is not Ghia's Table II. See section 12.*

VAL-002 (lid-driven cavity, Re=100) fails the acceptance criterion in REQ-S03 (centerline profiles within 2% of Ghia et al. 1982). The u-velocity profile along x=0.5 agrees with Ghia to a maximum absolute error of 1.6% and an L2 error of 0.77%, well within criterion. The v-velocity profile along y=0.5 disagrees with Ghia by a maximum absolute error of 20.1% and an L2 error of 9.3%, concentrated in the interior region x∈[0.28, 0.62]. The error does not decrease under grid refinement (it is present at 80×80 and remains at the same magnitude at 128×128), ruling out discretization truncation as the cause.

Related observations from diagnostic analysis:

- The computed vortex center from the stream function sits at (0.619, 0.744), agreeing with Ghia's (0.617, 0.734) to within three grid cells.
- Max |v| in the solver reaches 0.49 near the right wall, roughly twice the magnitude of Ghia's max centerline v.
- The discrete velocity divergence using central differences is spatially uniform at 2.5×10⁻³ across the interior, with a 1.6 spike at the top-right lid-wall corner.

The solver is converged to its configured tolerance (final residual 9.99×10⁻⁷, exited cleanly before max iterations) and the vortex macro-topology is correct. The 20% v-error represents a systematic defect in the discrete equations, not an implementation bug or an under-converged solution.

**Refinement behaviour, measured 2026-09-19 (amendment).** The benchmark harness (`scripts/benchmark.py`, records in `benchmarks/results.jsonl` at commits 4063813 and dcddb39) measured the maximum normalized centerline errors at 20x20, 40x40 and 80x80 with the validation settings. The u-error falls from 0.106 to 0.041 to 0.013, an observed order of 1.4 then 1.6. The v-error rises from 0.1765 to 0.1893 to 0.2072. The scheme converges in u and moves the wrong way in v on the same grids. A scheme that improves in one component and degrades in the other under refinement is converging to a different answer, which is a structural defect rather than a resolution shortfall. This sharpens the statement above that the error does not decrease: it increases.

**Independent confirmation by mass conservation, measured 2026-09-19 (amendment).** A second observation reaches the same wall treatment without reference to Ghia. The pressure solver probe (`docs/reports/pressure_solver_probe.md`, Tables B and E) measured the net mass flux through boundaries that should pass none. In the closed cavity the mass imbalance summed over all fluid cells settles at 2.90e-2, 9.23e-3 and 2.35e-3 (mass-flux units with rho = U = L = 1) at 20, 40 and 80 cells per side, and at convergence it is uniform over the domain: 9.02e-5, 6.46e-6 and 3.88e-7 per cell. Divided by the cell area, the 80x80 figure is the uniform divergence of 2.5e-3 already listed among the observations above, which the probe explains. The mechanism is the ghost-cell rule itself: the face flux at a wall is `0.5 (v_interior + v_ghost)` with `v_ghost = v_interior / 3`, which is `(2/3) v_interior` rather than zero, so mass crosses every wall. In a closed domain the pressure-correction system is then all-Neumann with a right-hand side that does not sum to zero, which has no solution; Jacobi drifts by a constant, the velocity correction reads only gradients, and the correction is a no-op at convergence. The wall treatment is therefore implicated by two independent routes: the velocity comparison against Ghia and the continuity measurement.

## 2. Root Cause Summary

The current solver combines three numerical choices that interact poorly and cannot be independently corrected without introducing new defects:

**Collocated variable arrangement with Rhie-Chow interpolation (REQ-S07).** Storing u, v, and p at identical cell-center locations requires Rhie-Chow pressure-weighted face velocity interpolation to prevent checkerboard decoupling. Rhie-Chow enforces continuity on corrected face fluxes, but the cell-center velocity correction uses a different (central-difference) pressure gradient than the face correction (compact gradient) uses. These gradients agree only in the limit of smooth pressure fields; in the Re=100 cavity where pressure curvature is strong, the inconsistency produces a systematic bias that the pressure-correction loop cannot resolve.

**Hybrid differencing advection scheme (REQ-S09).** The Spalding 1972 hybrid scheme switches between central differencing and upwinding based on cell Péclet number, providing first-order accuracy in advection-dominated regions. At Re=100 the cavity sits in the transitional regime where the scheme is CDS-dominated but introduces dispersive error in high-gradient zones, contributing to the observed over-intense descending jet.

**Ghost cell boundary treatment (ADR-008).** Wall and inlet BCs are imposed through the collocated ghost cell formula `u_bnd = (2·u_prescribed + u_int) / 3`, which places the physical BC at the domain face via linear interpolation. This scheme is O(h) accurate at walls (one order lower than the interior scheme) and was already documented as a known limitation requiring REQ-S02 to be relaxed from 1% to 2% for VAL-001 acceptance.

These three decisions are coupled by the collocated-grid choice. Fixing the Rhie-Chow cell-center inconsistency in isolation would require replacing the velocity correction stencil, which in turn changes the pressure-correction equation derivation, which in turn changes how BCs interact with the coupled system. Fixing the ghost cell O(h) accuracy in isolation has been attempted (see handoff: "wall momentum treatment" experiment) and created secondary defects that forced a revert. The defects are structural to the architecture, not to specific lines of code.

## 3. Options Considered

### 3.1 Option A: Relax REQ-S03 and document the defect

Revise REQ-S03 to accept the observed error pattern, with an ADR documenting the limitation of the current scheme. Similar to the precedent set by ADR-008 for REQ-S02.

**Rejected because:** VAL-002 is the primary validation test for 2D nonlinear advection and recirculation: the exact physics the cleanroom simulation depends on for particle transport. A 20% systematic error in the centerline velocity directly corresponds to 20% errors in particle advection trajectories. The error is not a bounded approximation but a bias in the wrong direction across a large fraction of the domain. Accepting this error would render the cleanroom simulation results quantitatively unreliable and qualitatively misleading. Portfolio-wise, accepting a 20% error on a textbook validation against published data signals a willingness to ship broken tools, which is inconsistent with the project's demonstration purpose.

### 3.2 Option B: Targeted fix within the current architecture

Attempt to reformulate the Rhie-Chow velocity correction to use compact gradients consistently, replace the ghost cell scheme with a wall-matched second-order BC, and retune the hybrid scheme parameters.

**Rejected because:** The three defects are coupled. Fixing one in isolation creates secondary defects elsewhere, as previously demonstrated. A full reformulation touching all three is equivalent in engineering effort to Option C but produces a less numerically robust result. The collocated + Rhie-Chow architecture is known in the CFD literature to require substantial maintenance effort for correctness at moderate Re; the cleanroom application's target Re range (50 to 2000) sits squarely in the difficult zone for this approach.

### 3.3 Option C: Rebuild solver on staggered grid with non-uniform mesh and QUICK advection

Replace the collocated layout with a staggered (MAC) arrangement (Harlow and Welch 1965). Extend the mesh to support per-axis geometric stretching. Replace the hybrid scheme with the QUICK scheme (Leonard 1979). Replace ghost cells with direct imposition of velocity BCs on physical faces.

**Selected.** Rationale in Section 4.

## 4. Recommended Change

Rebuild the Navier-Stokes solver on a staggered (MAC) variable arrangement with a non-uniform Cartesian mesh and second-order QUICK advection. The change is executed as one coordinated rebuild because the four components are tightly coupled: staggered grid enables direct BC imposition, direct BC imposition enables second-order wall accuracy, second-order wall accuracy enables QUICK to deliver its design convergence order, and the non-uniform mesh leverages the new architecture's accuracy to place resolution efficiently.

**Staggered MAC grid.** Pressure stored at cell centers (`p` shape [ny, nx]). u-velocity stored at east-west faces (`u` shape [ny, nx+1]). v-velocity stored at north-south faces (`v` shape [ny+1, nx]).

These are the internal storage shapes used by the solver. The public `solve_steady()` return contract is preserved: u, v, and p are interpolated to cell centers before return, so downstream consumers continue to receive `[ny, nx]` shaped arrays consistent with the existing cross-cutting array convention (SYSTEM.md Section 3.3). Interpolation is performed via simple face-to-center averaging, adding O(h²) interpolation error to the returned fields which is consistent with the staggered scheme's interior accuracy.

Continuity at cell (j, i) becomes `(u[j, i+1] - u[j, i]) / dx_cell[i] + (v[j+1, i] - v[j, i]) / dy_cell[j] = 0` exactly, with no Rhie-Chow interpolation required. The pressure gradient driving u-momentum naturally lives at u-face locations, giving automatic pressure-velocity coupling and eliminating checkerboard modes by construction.

**Non-uniform Cartesian mesh with geometric stretching.** Mesh stores explicit coordinate arrays `x_face[0..nx]`, `x_center[0..nx-1]`, `dx_cell[0..nx-1]`, and `dx_face[0..nx]` (the latter being the distance between adjacent cell centers, used for diffusive stencils across faces). Stretching is specified by geometric ratio and minimum spacing per wall, allowing independent refinement in x and y. This preserves the structured-grid i-j indexing so CUDA portability and vectorized NumPy operations are unaffected.

**QUICK advection scheme.** Face velocity for advective flux is reconstructed from a quadratic interpolant through three upstream-weighted cells, giving second-order accuracy globally on smooth flows. Boundary cells use special stencils documented in Ferziger & Peric chapter 4. QUICK is chosen over hybrid because cleanroom flows are smooth and moderate-Péclet throughout; over TVD schemes because the flow contains no shocks or sharp discontinuities that would motivate flux limiters. If later application evidence warrants boundedness guarantees (e.g., for sharp concentration fronts in Phase 3 transport), QUICK can be extended with a limiter as an incremental enhancement.

**Direct BC imposition.** Dirichlet velocity BCs are written directly onto the u or v face value at the physical wall location, with no ghost cell, no linear interpolation, and no 1:3 distance ratio formula. Wall-adjacent diffusive stencils handle the half-cell distance from the first interior velocity-point to the wall via the explicit `dx_face` array. This yields O(h²) wall accuracy, matching interior accuracy.

## 5. Requirement Changes

### 5.1 Modified requirements

| ID | Current text | Proposed text | Reason |
|----|-------------|---------------|--------|
| REQ-S02 | The NS solver shall reproduce the Poiseuille flow parabolic velocity profile with L2 error < 2.5%. (Relaxed from originally-specified <1% per ADR-008 due to O(h) wall accuracy of collocated ghost-cell scheme.) | No change to requirement text. Test criterion tightens from 2.5% (current, ADR-008 relaxation) to < 1% (original specification) once the staggered-grid rebuild lands and is validated. Tightening is part of the rebuild PR, not this ECR. | Staggered grid with direct BC imposition provides O(h²) wall accuracy, restoring the original <1% target. Applied post-rebuild validation. |
| REQ-S07 | The solver shall use a collocated variable arrangement with Rhie-Chow interpolation for face velocities. | The solver shall use a staggered (MAC) variable arrangement with pressure at cell centers, u-velocity at east-west cell faces, and v-velocity at north-south cell faces. | Staggered arrangement provides natural pressure-velocity coupling without interpolation artifacts. Resolves systemic bias causing VAL-002 failure. |
| REQ-S09 | The advection term shall be discretized using the hybrid differencing scheme (Spalding 1972). | The advection term shall be discretized using the QUICK scheme (Leonard 1979) with specialized stencils at boundary-adjacent cells. | Second-order accuracy globally on smooth flows; appropriate for moderate-Péclet cleanroom regime. Hybrid's first-order-upwind fallback is unnecessary for the target application. |

### 5.2 Unchanged requirements

| ID | Status | Note |
|----|--------|------|
| REQ-S01 | No change | Convergence requirement remains applicable. |
| REQ-S03 | No change | Validation criterion (2% Ghia centerline) remains. Expected to pass under new architecture. |
| REQ-S04 | No change | Incompressibility enforcement is strengthened (exact on staggered grid by construction). |
| REQ-S05 | No change | SIMPLE algorithm retained. Pressure correction equation derivation differs but algorithm structure is identical. |
| REQ-S06 | No change | NumPy reference + CUDA C++ architecture unaffected. CUDA port will target the new solver. |
| REQ-S08 | No change | Jacobi iteration for pressure correction retained. |
| REQ-S10 | No change | Under-relaxation factors remain configurable with same defaults. |

### 5.3 New requirements

| ID | Text | Rationale | Verified By |
|----|------|-----------|-------------|
| REQ-S11 | The mesh shall support independent geometric stretching in x and y directions, specified by minimum face spacing and geometric expansion ratio per wall. | Enables resolution clustering near walls, HEPA diffusers, and other high-gradient regions without uniform refinement of the entire domain. | Unit test |
| REQ-S12 | Dirichlet velocity boundary conditions shall be imposed directly on the staggered velocity components at the physical wall location, without ghost cell interpolation. | Eliminates O(h) wall accuracy limitation. Enables REQ-S02 criterion to return to < 1%. | Unit test, VAL-001 |

## 6. ADR Changes

| ADR | Action | Details |
|-----|--------|---------|
| ADR-003 | Amend | Structured grid retained. Layout changes from collocated to staggered. Mesh extended to support per-axis non-uniform spacing. Staircase representation of internal obstacles preserved. |
| ADR-008 | Supersede | Collocated ghost cell wall treatment no longer applicable. REQ-S02 criterion returns to < 1% (original wording). ADR-008 marked as superseded by ADR-010 in the registry. |
| ADR-009 | Discard | Drafted but uncommitted ADR for "VAL-002 structural validation criteria" (alternative acceptance scheme) is no longer needed. Root cause identified; point-wise Ghia comparison retained as criterion. |
| ADR-010 | New | "Solver Architecture V2: Staggered Grid, Non-Uniform Mesh, QUICK Advection." Documents the rebuild decision in detail. Pairs with this ECR. ADR-010 authoring is deferred to the rebuild PR (phase2/staggered-rebuild) because its technical content depends on implementation decisions made during the rebuild. This ECR establishes the intent; ADR-010 will record the executed design. |

## 7. Affected Artifacts

### 7.1 Source code

| Artifact | Impact | Scope |
|----------|--------|-------|
| `src/solver_ns.py` | Rewrite | Full solver replacement. ~800 lines → estimated ~600-900 lines after simplification. **Interface obligation (amendment 2026-09-20):** `solve_steady(on_iteration=None)` must keep accepting a callback that receives an `IterationState` (iteration, residual, pressure_sweeps, u, v, p) after every outer iteration, and the solver must keep exposing `last_pressure_sweeps` and per-stage wall time in `stage_seconds`. `scripts/benchmark.py` depends on all three to record error-versus-work trajectories. If the rebuilt solver drops them, the comparison between the old and new solver breaks at the moment it is needed. |
| `src/boundary.py` | Rewrite | Ghost cell logic removed. Direct BC imposition on staggered faces. |
| `src/mesh.py` | Extend | Add non-uniform coordinate arrays (`x_face`, `x_center`, `dx_cell`, `dx_face`, and y equivalents). Existing uniform-mesh API preserved as a special case. |
| `src/config.py` | Extend | Add mesh stretching parameters to YAML schema (min_spacing_{top,bottom,left,right}, stretch_ratio). |
| `configs/clean_room_default.yaml` | Update | Add mesh stretching section. Non-uniform spacing illustrated for cleanroom geometry. |

### 7.2 Tests

| Artifact | Impact |
|----------|--------|
| `tests/test_mesh.py` | Extend: add non-uniform mesh tests. |
| `tests/test_boundary.py` | Rewrite: staggered BC imposition tests. |
| `tests/test_solver_ns.py` | Rewrite: staggered coefficients, momentum sweep, pressure correction. |
| `tests/test_poiseuille.py` | Update: criterion tightens to 1% L2. |
| `tests/test_lid_cavity.py` | No change: same criterion, expected to pass under rebuilt solver. |

### 7.3 Documentation

| Artifact | Impact |
|----------|--------|
| `docs/SYSTEM.md` | Update: REQ-S02 test criterion note, REQ-S07 text, REQ-S09 text, REQ-S11 and REQ-S12 added, ADR summary table updated. |
| `docs/PROJECT_PLAN.md` | Update: Phase 2 deliverables marked as REBUILDING for affected modules. Phase 2 risks table revised. Phase 2 status remains IN PROGRESS. |
| `docs/ADR/ADR-008-collocated-ghost-cell-walls.md` | Update: add "Superseded by ADR-010" header. |
| `docs/ADR/ADR-010-staggered-grid-architecture.md` | Create |
| `docs/ECR/ECR-001-solver-architecture-rebuild.md` | Create (this document) |

### 7.4 Cascade impact

No impact on Phase 1 modules (`config.py`, `mesh.py`, `particles.py`) beyond the mesh extension noted above. No impact on Phase 3+ deliverables: transport solver and downstream phases consume velocity and pressure fields through unchanged interfaces (`NavierStokesSolver.solve_steady()` returns `(u, v, p)` in the same shapes; the staggered layout is internal to the solver).

## 8. Implementation Plan

Estimated effort: 10-15 working days, executed as a feature branch off `phase2/validation-tests`.

| Step | Deliverable | Estimated effort |
|------|-------------|------------------|
| 1 | Mesh module extension with non-uniform coordinate arrays and stretching functions. Unit tests for mesh geometry. | 2 days |
| 2 | Staggered field data structures, allocation, and I/O. | 1 day |
| 3 | Boundary module rewrite for direct staggered BC imposition. | 2 days |
| 4 | Momentum predictor with QUICK advection on staggered grid. | 3 days |
| 5 | Pressure correction equation on staggered grid, integrated with existing Jacobi solver. | 2 days |
| 6 | End-to-end integration, debugging, convergence tuning. | 2-3 days |
| 7 | VAL-001 revalidation (expected PASS at < 1% criterion). | 0.5 day |
| 8 | VAL-002 revalidation (expected PASS at < 2% criterion). | 0.5 day |
| 9 | ADR-010 write-up, SYSTEM.md and PROJECT_PLAN.md updates, ADR-008 supersession note. | 1 day |

*Note 2026-09-24, beside step 7:* the stopping rule is `error_estimate` (`src/stopping.py`), built ahead of step 7: the solve stops when the estimated iteration error over the largest prescribed boundary velocity, the worst per-cell mass imbalance (criterion 6 as written), and the summed imbalance over the through-flow are all below their tolerances, and reaching the iteration cap is reported as not converged. The summed condition bounds the flux drift of an open domain on any grid, which the per-cell bound alone does more loosely on every finer grid. It is opt-in by the solver key `stopping_rule`, and the velocity-step rule stays the default, bitwise. The validation cases switch to it in step 7. Evidence and the check that it stops where it should: `docs/reports/stopping_rule_evidence.md`, sections 4 to 6 and 9.

## 9. Acceptance Criteria

The change is accepted when all of the following are demonstrated:

1. VAL-001 (Poiseuille flow) passes at L2 error < 1% on 80×40 uniform mesh.
2. VAL-001 passes at L2 error < 1% on a non-uniform mesh with wall clustering (geometric ratio 1.05, min spacing 0.1·L/ny). *Note (amendment 2026-09-20): at a fixed cell count the ratio and the wall spacing are not independent; a symmetric geometric distribution that closes on the domain length has one free parameter. At ny = 40 a ratio of 1.05 gives a wall spacing of 0.605 L/ny, and a wall spacing of 0.1 L/ny requires a ratio of about 1.20. Step 1 of the implementation supports specifying either quantity with the other derived. Which one this criterion fixes is an open question recorded in `docs/STATUS.md` and must be settled before step 7.*

   *Amendment 2026-09-24, criterion 2: the mesh quantity.* Alex fixed the wall spacing at 0.1 L/ny, with the geometric ratio derived from it (about 1.20 at ny = 40), not the ratio of 1.05. It is the stricter test of the stretched stencils, and the wall-adjacent spacing is the quantity Phase 3 deposition depends on. No threshold changes.
3. VAL-002 (lid-driven cavity) passes at max centerline error < 2% for both u and v profiles on 80×80 uniform mesh.
   3a. *(Erratum 2026-09-22: the v baseline in this criterion is against the corrupted table; section 12 gives the corrected series.)* *(Amendment 2026-09-20.)* The VAL-002 maximum normalized centerline errors for u and for v each decrease monotonically across 20x20, 40x40 and 80x80 uniform meshes, with the observed order of convergence reported for both. Baseline to improve on, from `benchmarks/results.jsonl` (collocated solver, commits 4063813 and dcddb39): u 0.106, 0.041, 0.013; v 0.1765, 0.1893, 0.2072. The current scheme converges in u and moves the wrong way in v. A single-grid threshold can be met by luck while a directional defect remains; this criterion cannot.

   *Amendment 2026-09-24, criteria 3 and 3a: the reference.* Alex decided on 2026-09-24 that VAL-002 (criterion 3) and criterion 3a score the cavity against `marchi_2009_re100`, Marchi, Suero and Araki (2009), and that `ghia_1982_re100_r2` is reported beside them with no threshold. The reason is in `docs/reports/cavity_reference_marchi.md`, section 5: measured against Ghia, a correctly converging scheme stops improving at the level of Ghia's own error in the jet by the right wall, and a monotone criterion then fails the right answer. No threshold changes. The metric and the VAL-002 test change in step 8; until then both read `ghia_1982_re100_r2`.
4. Grid convergence study shows observed order of accuracy ≥ 1.8 (target 2.0) under uniform mesh refinement on VAL-001.
5. All existing Phase 1 validation tests (VAL-005, VAL-006, VAL-010, VAL-011) continue to pass.
6. Discrete continuity constraint (REQ-S04) satisfied to < 10⁻¹⁰ per cell, and the mass imbalance summed over the domain below the same bound. *(Amended 2026-09-20.)* Baseline to improve on, from `docs/reports/pressure_solver_probe.md` Tables B and E (collocated solver, VAL-002 at convergence): per-cell imbalance 9.02e-5, 6.46e-6 and 3.88e-7 at 20, 40 and 80 cells per side, uniform over the domain; net wall leak 2.90e-2, 9.23e-3 and 2.35e-3 (mass-flux units, rho = U = L = 1). The staggered grid satisfies this by construction, and with the baseline underneath it the criterion is a quantified improvement rather than a sanity check.
7. Code review passes per existing CI/CD pipeline.
8. SYSTEM.md, PROJECT_PLAN.md, ADR-010, and this ECR are updated and committed.

## 10. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Staggered grid implementation bugs produce new validation failures. | Medium | High | Reference implementations in Ferziger & Peric chapter 7 and Versteeg & Malalasekera chapter 6 serve as cross-check. Incremental development with unit tests on individual stencils before integration. |
| QUICK scheme instability at boundaries. | Medium | Medium | Use standard QUICK boundary stencils from literature (Leonard 1979, appendix). First validation on Poiseuille (smooth, analytical truth) before cavity. |
| Non-uniform mesh introduces coefficient scaling bugs. | Medium | Medium | Develop and validate on uniform mesh first (regression against current VAL-001 result). Enable stretching only after uniform case passes. |
| CUDA port (Phase 6) more complex on staggered grid than collocated. | Low | Low | Staggered grids are well-supported in CUDA CFD literature. Regular memory access patterns preserved. |
| Effort estimate underestimates debugging time. | Medium | Low | Job search is the binding timeline, not this phase. Phase 3 completion is the minimum viable portfolio artifact per PROJECT_PLAN; slipping Phase 2 completion by a week does not threaten portfolio deliverable. |
| Overshoots from QUICK near HEPA jets in Phase 4 scenarios. | Low | Low | If observed, add flux limiter as incremental enhancement (documented as future work in ADR-010). |

## 11. Approval

By signing below, approvers confirm:

- The problem statement and root cause summary are accurate.
- The selected option is appropriate given the alternatives considered.
- The requirement and ADR changes proposed are consistent with system architecture principles.
- The estimated effort is reasonable.
- The acceptance criteria are sufficient to close out the change.

| Role | Name | Approval | Date |
|------|------|----------|------|
| Author | Alex Moroz-Smietana | Approved | 2026-04-16 |
| Reviewer | Claude | Approved | 2026-04-16 |

## 12. Erratum, 2026-09-22: the Ghia v reference

**What the table was.** Every VAL-002 v figure in this document was measured against a v
reference that is not Ghia, Ghia and Shin (1982), Table II. It entered the repository with
the VAL-002 test on 2026-04-16 (commit d589b9f) and moved unchanged to
`validation/metrics.py` on 2026-09-19. Its sixteen stations were Table I's y stations
without 0.9766. Ten of its values are Table II values, but only five sit at their Table II
station, and six appear nowhere in Table II: it gave v(0.5) = -0.11477 where Table II gives
+0.05454. It fails a check that needs no source. The net vertical flux through y = 0.5 of a
closed cavity is zero, and the table integrates to -0.095 there, where Table II gives
-0.0004. It was found in the step 6 integration
(`docs/reports/staggered_integration_step6.md`, section 4) and replaced by Table II under a
new reference name, `ghia_1982_re100_r2`; stored rows scored against the old table keep the
old name, `ghia_1982_re100`.

**What it did to section 1.** The v figures there are a 20.1% maximum and 9.3% L2 error,
"concentrated in the interior region x in [0.28, 0.62]" and unchanged at 80x80 and 128x128.
The maximum is what the VAL-002 test that introduced the table in d589b9f scores at 80x80
against it. The L2 and 128x128 figures came from diagnostics that name no reference and are
presumed to use the same table. The interval is where four of the table's stations,
0.2813, 0.4531, 0.5000 and 0.6172, carry values that are not Table II's. The conclusion that
the 20% v error "represents a systematic defect in the discrete equations", and section
3.1's use of the same 20% as a bias in particle trajectories, rested on these figures and
are withdrawn as evidence. Section 1's u figures, the vortex-center location and the
uniform-divergence observation did not use the v table.

**What it did to the refinement argument.** The amendment's series in section 1, v 0.1765,
0.1893, 0.2072 at 20x20, 40x40 and 80x80, rising, was measured against the same table, and
so was criterion 3a's v baseline. Against Table II the collocated solver's v errors at the
same grids are 0.146, 0.0511, 0.00981: they fall under refinement, at an observed order of
1.5 then 2.4. The statement that the scheme "converges in u and moves the wrong way in v",
and that this is "a structural defect rather than a resolution shortfall", does not hold.
The u series, 0.106, 0.0408, 0.0134, is unchanged to every digit.

**Corrected criterion 3a baseline.** Collocated solver, reference `ghia_1982_re100_r2`, the
rows appended to `benchmarks/results.jsonl` by `fix/ghia-v-reference`: u 0.106, 0.0408,
0.0134; v 0.146, 0.0511, 0.00981. The staggered solver's v errors on the same grids are
0.0294, 0.0224, 0.0154.

**What it means for criterion 3.** Against Table II the collocated solver's 80x80
errors, u 0.0134 and v 0.00981, are both below criterion 3's 2%. VAL-002 in CI runs
at 40x40, where the collocated solver fails in both components (u 0.0408, v
0.0511), and it stays marked xfail on that basis.

**What the correction does not invalidate.** The measured wall leak, the criterion 6
baseline and section 1's second amendment, which reads the fields and no reference data;
the collocated inlet flux shortfall (`docs/reports/inlet_flux_comparison.md`); the O(h)
wall accuracy on VAL-001 recorded in ADR-008; and the staggered solver's VAL-001 and u
results, which never touched the v table.

---

## Document History

| Date | Change | Author |
|------|--------|--------|
| 2026-04-16 | Initial version. | Alex Moroz-Smietana |
| 2026-09-20 | Amendments before implementation: refinement series and continuity measurement added to the problem statement as a second independent confirmation of the wall-treatment defect; acceptance criterion 3a (monotone refinement) added and criterion 6 given its measured baseline; note on the over-determined mesh specification in criterion 2; solve_steady callback, sweep count and stage timing recorded as an interface obligation in 7.1. REQ-S08 untouched. | Alex Moroz-Smietana |
| 2026-09-22 | Erratum (section 12): the Ghia v reference used for every VAL-002 v figure was not Table II. Pointers added at the top of section 1 and beside criterion 3a; corrected v series given against reference ghia_1982_re100_r2. No figure in sections 1 to 11 edited. | Alex Moroz-Smietana |
| 2026-09-24 | Amendment under criteria 3 and 3a: VAL-002 and criterion 3a score the cavity against marchi_2009_re100, with ghia_1982_re100_r2 reported beside them and no threshold (docs/reports/cavity_reference_marchi.md). No threshold changed; the metric and the test change in step 8. | Alex Moroz-Smietana |
| 2026-09-24 | Amendment under criterion 2: the wall spacing 0.1 L/ny is fixed and the ratio derived (about 1.20 at ny = 40). Note beside step 7 in section 8: the stopping rule is error_estimate (src/stopping.py), opt-in with the velocity-step rule as the default, and the validation cases switch to it in step 7 (docs/reports/stopping_rule_evidence.md, section 9). No threshold changed. | Alex Moroz-Smietana |
