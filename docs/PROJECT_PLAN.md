# Project Plan

**Project:** CFD Clean Room Simulation
**Last Updated:** 2026-09-30
**Current Phase:** Phase 2 (Navier-Stokes Solver)

This document tracks development progress by phase. Code review reads this document to determine the current phase and verify that PRs are in scope; the policy is `docs/REVIEW_POLICY.md`. Update this document as work progresses.

---

## Phase Status Summary

| Phase | Name | Status | Gate Verdict | Report |
|-------|------|--------|-------------|--------|
| 0 | Infrastructure | COMPLETE | PASS | -- |
| 1 | Foundation | COMPLETE | PASS | phase1_foundation_report.md |
| 2 | Navier-Stokes Solver | GATE REVIEW | -- | -- |
| 3 | Transport Solver | NOT STARTED | -- | -- |
| 4 | Scenarios & Time Integration | NOT STARTED | -- | -- |
| 5 | Alert Monitoring System | NOT STARTED | -- | -- |
| 6 | CUDA Acceleration | NOT STARTED | -- | -- |
| 7 | Visualization & Portfolio | NOT STARTED | -- | -- |

Status values: NOT STARTED, IN PROGRESS, GATE REVIEW, COMPLETE

---

## Phase 0: Infrastructure

**Goal:** Repository setup, CI/CD pipeline, coding standards, project documentation.

### Deliverables

| Deliverable | Status |
|-------------|--------|
| GitHub repository (public) | DONE |
| CLAUDE.md (coding standards) | DONE |
| docs/SYSTEM.md (architecture) | DONE |
| docs/PROJECT_PLAN.md (this file) | DONE |
| pyproject.toml (ruff config, pytest markers) | DONE |
| requirements.txt | DONE |
| requirements-dev.txt | DONE |
| .gitignore | DONE |
| Pre-merge review and test commands (.claude/commands/cfd-review.md, cfd-test.md; replaced the review action on 2026-09-23) | DONE |
| .github/workflows/ci.yml (lint + test action) | DONE |
| Review policy (docs/REVIEW_POLICY.md, replaced the system prompt and script on 2026-09-21) | DONE |
| README.md | DONE |

### Gate Criteria

- All CI/CD workflows run successfully on a test PR
- Ruff formatting and linting pass on an empty project
- Code review bot posts a review comment on a test PR
- All project documentation committed to main

---

## Phase 1: Foundation

**Goal:** Build the infrastructure modules that all downstream code depends on. Validate particle physics computations.

**Branch prefix:** `phase1/`

### Deliverables

| Deliverable | Status | Notes |
|-------------|--------|-------|
| src/config.py | DONE | YAML loader with validation |
| src/mesh.py | DONE | Structured grid generation, cell classification |
| src/particles.py | DONE | Settling velocity, diffusion coeff, Cunningham correction, deposition velocity, HEPA efficiency |
| configs/clean_room_default.yaml | DONE | Updated with 8m x 3m room layout |
| tests/test_config.py | DONE | Unit tests: validation, rejection of bad input |
| tests/test_mesh.py | DONE | Unit + integration: geometry, classification, neighbors |
| tests/test_particles.py | DONE | Unit + validation: VAL-005, VAL-006 |

### Validation Gate

| Test ID | Description | Criterion | Status |
|---------|-------------|-----------|--------|
| VAL-005 | Stokes settling velocity | < 0.1% error vs analytical for all 5 size classes | PASS |
| VAL-006 | Brownian diffusion coefficient | < 0.1% error vs analytical for all 5 size classes | PASS |
| VAL-010 | Deposition velocity | < 0.1% error vs analytical (D/delta for ceiling/wall, D/delta + v_s for floor) for all 5 size classes | PASS |
| VAL-011 | HEPA interpolation | < 0.1% error vs algebraic log-space linear interpolation at intermediate diameter | PASS |

### Scope Additions

- `deposition_velocity` and `hepa_efficiency` added to `ParticlePhysics` in Phase 1. Rationale: pure particle physics with no external dependencies, avoids revisiting the module in Phase 3.

### Phase-Specific Risks

| Risk | Mitigation |
|------|------------|
| YAML schema design locks in a structure that needs rework later | Review config schema against all downstream module interfaces before implementation |

---

## Phase 2: Navier-Stokes Solver

**Goal:** Implement the SIMPLE algorithm for pressure-velocity coupling. Validate against analytical and benchmark solutions.

**Branch prefix:** `phase2/`

**Depends on:** Phase 1 complete (config, mesh)

### Deliverables

ECR-001 rebuilt the solver on a staggered grid in nine steps (2026-09-20 to 2026-10-01). ADR-010 records what was built; the staggered solver is the solver of record.

| Deliverable | Status | Notes |
|-------------|--------|-------|
| src/solver_staggered.py | DONE (ECR-001 steps 6 to 8) | Steady SIMPLE on the staggered MAC grid with the collocated solver's public shape. Stops by the velocity-step rule or, as both validation cases configure it, by the error_estimate rule. The solver of record (ADR-010). |
| src/staggered.py | DONE (ECR-001 step 2) | MAC field layout and face-to-center averaging (REQ-S07). |
| src/boundary_staggered.py, src/boundary_registry.py | DONE (ECR-001 step 3) | Direct Dirichlet imposition on the staggered faces (REQ-S12) over one shared reading of the configured boundaries (REQ-S12.1). |
| src/momentum.py | DONE (ECR-001 step 4) | Momentum predictor, QUICK by deferred correction over an upwind implicit matrix (REQ-S09). |
| src/pressure.py | DONE (ECR-001 step 5) | Pressure correction by weighted Jacobi, w = 2/3 (REQ-S08 as clarified 2026-09-22). |
| src/stopping.py | DONE (ahead of ECR-001 step 7) | The error_estimate stopping rule (REQ-S01, REQ-S04 as clarified 2026-09-24). |
| src/mesh.py | DONE (ECR-001 step 1) | Per-axis geometric wall clustering, mirrored about the midpoint (REQ-S11 as amended 2026-09-30). |
| src/solver_ns.py | RETAINED (collocated baseline) | SIMPLE on the collocated grid with Rhie-Chow and ghost-cell walls (ADR-008). Kept as the harness's before-and-after baseline; whether to retire it is Alex's open decision (docs/reports/phase2_navier_stokes_report.md, Deferred). |
| src/boundary.py (velocity/pressure BCs) | RETAINED (collocated baseline) | Collocated ghost-cell layer, now built on src/boundary_registry.py. Kept and decided with src/solver_ns.py. |
| tests/test_poiseuille.py | DONE | VAL-001: staggered at < 1% on 80x40 uniform and wall-clustered; collocated at 2.5%. |
| tests/test_lid_cavity.py | DONE | VAL-002: staggered at < 2% against marchi_2009_re100 on the case file's 40x40 in CI; collocated kept xfail against ghia_1982_re100_r2. |
| tests for the staggered modules | DONE | test_staggered.py, test_boundary_registry.py, test_boundary_staggered.py, test_momentum.py, test_pressure.py, test_solver_staggered.py, test_solver_selection.py, test_stopping.py. |
| tests/test_solver_ns.py | DONE (collocated, kept) | Unit and integration tests of the collocated solver. Not rewritten: the collocated solver was kept rather than replaced. |
| docs/ADR/ADR-010-staggered-grid-architecture.md | DONE (ECR-001 step 9) | The architecture as built, with planned against built. |
| scripts/view_field.py | DONE (development instrument) | Streamlines, pressure and cavity centerline profiles, for reading the field during the rebuild. Not the Phase 7 visualization deliverable; see Scope Changes. |
| Measurement instruments | DONE (development instruments) | scripts/benchmark.py and benchmarks/results.jsonl, validation/ (cases, references, metrics), scripts/self_convergence.py, scripts/stopping_probe.py, scripts/val001_order.py, scripts/gen_system_map.py. See Scope Changes. |

### Validation Gate

| Test ID | Description | Criterion | Status |
|---------|-------------|-----------|--------|
| VAL-001 | Poiseuille flow | L2 error < 1% on 80x40 (REQ-S02 as amended 2026-09-30), uniform and wall-clustered to 0.1 H / ny (ECR-001 criteria 1 and 2); observed order >= 1.8 under uniform refinement (criterion 4) | PASS on the staggered solver: 4.107e-4 uniform, 3.024e-3 clustered, order 1.992 under stopping rule version 3 (docs/reports/val001_revalidation_step7.md, addendum). The collocated solver measures 2.036e-2 against its own 2.5% (ADR-008). |
| VAL-002 | Lid-driven cavity | Maximum centerline error < 2% of the lid speed against marchi_2009_re100 at 80x80 (REQ-S03 as amended 2026-09-30, ECR-001 criterion 3); u and v errors each falling across 20x20, 40x40 and 80x80 (criterion 3a); ghia_1982_re100_r2 reported, unscored | PASS on the staggered solver: u 1.057e-3, v 7.356e-4 at 80x80; orders 2.24 and 2.11 (u), 2.12 and 2.07 (v) (docs/reports/val002_revalidation_step8.md). CI runs the 40x40 case file. The collocated test stays xfail at 40x40 against Ghia, u 0.0441 and v 0.0606. |

### Scope Changes

- C solver deliverables (csolver/pressure_solve.c, csolver.h, Makefile, test_c_parity.py) moved to Phase 6 (CUDA Acceleration). REQ-N03 is now validated against CUDA C++ rather than plain C.
- ECR-001 (approved 2026-04-16, closed 2026-10-01) replaced the collocated solver as the solver of record with a staggered one built alongside it. The collocated solver was kept as the harness baseline rather than rewritten in place; its retirement is deferred to Alex's decision. REQ-S11 was amended to the per-axis, mirrored stretching that was built.
- Measurement instruments added during the rebuild, without plan entries: the benchmark harness and its results file (PR #13), the validation package shared by tests and harness (PR #14), the system map generator (PR #12), and the self-convergence, stopping-rule and VAL-001 order scripts. Each exists to measure the solver; none is a deliverable of a later phase.
- scripts/view_field.py added as a Phase 2 development instrument (2026-09-19, PR #14). It exists so the velocity and pressure fields can be read against the Ghia reference while the solver is rebuilt under ECR-001. It is distinct from the Phase 7 visualization deliverable: no animation, no scenario output, no presentation quality, and it is not reused by scripts/visualize.py. Phase 7 remains NOT STARTED. The addition was made without a plan entry, which is a scope violation of the task prompt that introduced it, not of the build; recorded here on 2026-09-20 from the PR #14 review finding.

### Phase-Specific Risks

| Risk | Mitigation |
|------|------------|
| SIMPLE convergence failure on clean room geometry | Start validation with simple geometries (empty channel for Poiseuille, square cavity for lid-driven). Add obstacles incrementally. Under-relaxation defaults 0.7/0.3. |
| Staggered grid implementation bugs introduce new validation failures | Incremental development against reference implementations in Ferziger & Peric chapter 7. Uniform mesh validated before enabling stretching. |

---

## Phase 3: Transport Solver

**Goal:** Implement advection-diffusion equation for particle concentration transport. Validate diffusion, advection, and conservation independently.

**Branch prefix:** `phase3/`

**Depends on:** Phase 2 complete (NS solver produces velocity field)

### Deliverables

| Deliverable | Status | Notes |
|-------------|--------|-------|
| src/solver_transport.py | NOT STARTED | Advection-diffusion solver with v_ext interface, pure NumPy |
| src/boundary.py (concentration BCs) | NOT STARTED | Extension to existing boundary module |
| tests/test_diffusion.py | NOT STARTED | VAL-003 |
| tests/test_advection.py | NOT STARTED | VAL-004 |
| tests/test_conservation.py | NOT STARTED | VAL-007 |
| tests/test_solver_transport.py | NOT STARTED | Unit tests: source terms, settling integration |

### Validation Gate

| Test ID | Description | Criterion | Status |
|---------|-------------|-----------|--------|
| VAL-003 | Pure diffusion | L2 error < 1% vs analytical Gaussian | NOT RUN |
| VAL-004 | Pulse advection | Peak location error < 1 cell, shape preserved | NOT RUN |
| VAL-007 | Mass conservation | Imbalance < 0.01% of total mass | NOT RUN |

### Phase-Specific Risks

| Risk | Mitigation |
|------|------------|
| Numerical diffusion smears concentration fronts | Use higher-order advection scheme (QUICK or TVD limiter) if first-order upwind is too dissipative. |
| CFL restriction forces impractically small timestep | Implicit time integration for diffusion term. Explicit for advection only. |

---

## Phase 4: Scenarios & Time Integration

**Goal:** Build the scenario engine for contamination events and the time integration loop coordinating the NS and transport solvers. Implement the three reference scenarios.

**Branch prefix:** `phase4/`

**Depends on:** Phase 3 complete (transport solver validated)

### Deliverables

| Deliverable | Status | Notes |
|-------------|--------|-------|
| src/scenarios.py | NOT STARTED | Event management, source terms, BC modifications |
| src/time_integration.py | NOT STARTED | Timestep loop, CFL enforcement, solver coordination |
| src/io_manager.py | NOT STARTED | Output writing, checkpointing |
| configs/scenario_door_leak.yaml | NOT STARTED | Door seal failure scenario |
| configs/scenario_filter_breach.yaml | NOT STARTED | HEPA filter breach scenario |
| configs/scenario_equipment_dust.yaml | NOT STARTED | Equipment dust release scenario |
| tests/test_scenarios.py | NOT STARTED | Unit tests: event timing, activation, expiration |
| tests/test_time_integration.py | NOT STARTED | Integration tests: full timestep loop |
| tests/test_grid_convergence.py | NOT STARTED | VAL-008 |

### Validation Gate

| Test ID | Description | Criterion | Status |
|---------|-------------|-----------|--------|
| VAL-008 | Grid convergence | Observed order within 0.2 of theoretical | NOT RUN |

### Additional Gate Criteria

- All three scenarios run to completion without error
- Output files produced and loadable
- All previous validation tests still pass

### Phase-Specific Risks

| Risk | Mitigation |
|------|------------|
| Scenario BC modifications create solver instability | Ramp event onset over multiple timesteps rather than step-change. |

---

## Phase 5: Alert Monitoring System

**Goal:** Implement sensor probes, threshold monitoring, detection latency analysis, and sensor placement comparison.

**Branch prefix:** `phase5/`

**Depends on:** Phase 4 complete (scenarios produce time-evolving concentration fields)

### Deliverables

| Deliverable | Status | Notes |
|-------------|--------|-------|
| src/monitor.py | NOT STARTED | Sensor probes, threshold comparison, latency measurement |
| tests/test_monitor.py | NOT STARTED | Unit tests: threshold detection, false positive/negative |
| tests/test_alert_latency.py | NOT STARTED | VAL-009 |
| tests/test_sensor_evaluation.py | NOT STARTED | Integration: multi-scenario sensor comparison |
| Detection latency analysis output | NOT STARTED | Results for all scenarios with 2+ sensor configs |

### Validation Gate

| Test ID | Description | Criterion | Status |
|---------|-------------|-----------|--------|
| VAL-009 | Alert latency | Detection delay = 0 or 1 timestep | NOT RUN |

### Additional Gate Criteria

- Detection latency report generated for all scenarios with at least two sensor configurations
- Alert monitor does not modify concentration fields (separation of concerns verified)

---

## Phase 6: CUDA Acceleration

**Goal:** Implement CUDA C++ kernels for the performance-critical pressure correction and advection-diffusion inner loops. Validate against NumPy reference implementation.

**Branch prefix:** `phase6/`

**Depends on:** Phase 3 complete (both NS and transport solvers validated in NumPy)

### Deliverables

| Deliverable | Status | Notes |
|-------------|--------|-------|
| csolver/pressure_solve.cu | NOT STARTED | CUDA kernel for Jacobi pressure correction |
| csolver/advection_diffusion.cu | NOT STARTED | CUDA kernel for transport stencil |
| csolver/bindings.cpp | NOT STARTED | pybind11 Python interface |
| csolver/CMakeLists.txt | NOT STARTED | Build system for nvcc + pybind11 |
| tests/test_cuda_parity.py | NOT STARTED | REQ-N03: CUDA output matches NumPy reference |
| Benchmark results | NOT STARTED | Timing comparison: NumPy vs CUDA on default grid |

### Validation Gate

| Test ID | Description | Criterion | Status |
|---------|-------------|-----------|--------|
| REQ-N03 | CUDA parity | Max absolute difference < 1e-10 vs NumPy for VAL-001 through VAL-008 | NOT RUN |

### Phase-Specific Risks

| Risk | Mitigation |
|------|------------|
| GPU not available in CI | CUDA parity tests marked with @pytest.mark.cuda, skipped in GitHub Actions. Run locally before merge. CI validates physics via NumPy tests. |
| pybind11 + CUDA build complexity | CMake handles nvcc/pybind11 integration. Document build prerequisites in README. |

---

## Phase 7: Visualization & Portfolio

**Goal:** Produce animated visualizations, build the portfolio website page, and finalize all documentation.

**Branch prefix:** `phase7/`

**Depends on:** Phase 5 complete (alert data available for overlay)

### Deliverables

| Deliverable | Status | Notes |
|-------------|--------|-------|
| scripts/visualize.py | NOT STARTED | Matplotlib animations, streamlines, heatmaps. Distinct from scripts/view_field.py, the Phase 2 development instrument. |
| Animated output per scenario | NOT STARTED | MP4/WebM for each scenario, each size class |
| Portfolio website page | NOT STARTED | Hosted on alexyms.github.io |
| README.md (final version) | NOT STARTED | Project overview, setup instructions, results summary |
| docs/reports/ (all phase reports) | NOT STARTED | Completion reports for phases 1-6 |
| Interactive web visualization | STRETCH | JavaScript renderer with time scrubbing |

### Gate Criteria

- Visual outputs reviewed for physical plausibility
- Portfolio page published to alexyms.github.io
- All phase completion reports committed
- All validation tests pass
- README complete with setup instructions and results

---

## Timeline Notes

No fixed calendar dates. Phases are sequenced by dependency, not by schedule. The job search timeline creates external pressure, but shipping a correct Phase 3 (validated NS + transport solver) is more valuable than rushing to Phase 6 with unvalidated physics.

Phase 3 completion is the minimum viable portfolio artifact. A working, validated CFD solver with multi-class particle transport is demonstrable even without the alert system and visualization polish.

---

## Change Log

| Date | Change |
|------|--------|
| 2026-04-14 | Initial plan created. All phases at NOT STARTED except Phase 0 (IN PROGRESS). |
| 2026-04-14 | Phase 0 complete. All infrastructure deliverables DONE. CI and review bot validated on test PR #1. Phase 1 now IN PROGRESS. |
| 2026-04-15 | Phase 1 complete. Phase 2 architecture updates: replaced C/ctypes with NumPy reference + CUDA C++/pybind11 strategy. Added REQ-S07 through REQ-S10. C deliverables moved to new Phase 6 (CUDA Acceleration). Visualization renumbered to Phase 7. |
| 2026-04-16 | ECR-001 approved. Phase 2 solver rebuild on staggered MAC grid with non-uniform mesh and QUICK advection. Related source modules marked REBUILDING. VAL-001 criterion will tighten from the current 2.5% (ADR-008 relaxation) back to the originally-specified < 1% as part of the rebuild PR. This docs PR does not modify the live criterion; the change takes effect when the staggered-grid solver passes validation. |
| 2026-09-20 | scripts/view_field.py recorded as a Phase 2 development instrument, distinct from the Phase 7 visualization deliverable, which remains NOT STARTED. It was introduced in PR #14 without a plan entry; the PR #14 review caught the omission. |
| 2026-09-21 | Hand-rolled review pipeline (review.py, review_diff.py, their tests, the system prompt under .github/prompts/) removed. review.yml now runs the Claude Code code-review plugin once per pull request on the subscription; the policy moved to docs/REVIEW_POLICY.md, imported by CLAUDE.md, and governs both that run and review in VS Code. |
| 2026-09-21 | ECR-001 step 3 delivered: direct boundary imposition on the staggered grid (src/boundary_staggered.py, REQ-S12) and the shared boundary registry (src/boundary_registry.py, REQ-S12.1, derived). Collocated path unchanged and the harness rows reproduced. Inlet flux on VAL-001 compared between the two layers; see docs/reports/inlet_flux_comparison.md. |
| 2026-09-22 | ECR-001 step 4 delivered: staggered momentum predictor with QUICK advection by deferred correction (src/momentum.py, REQ-S07, REQ-S09). Returns u*, v* and the un-relaxed a_P diagonals as the step 5 contract. Not integrated into solve_steady; harness rows reproduced. |
| 2026-09-22 | ECR-001 step 5 delivered: staggered pressure correction (src/pressure.py, REQ-S04, REQ-S08). Closed-domain right-hand side sums to zero to rounding on 20, 40 and 80 cells per side, against the collocated 2.90e-2, 9.23e-3 and 2.35e-3. Undamped Jacobi found non-convergent on the closed system; see docs/reports/pressure_correction_step5.md. Not integrated; harness rows reproduced. |
| 2026-09-22 | REQ-S08 clarified, not amended: the pressure Jacobi sweep in src/pressure.py is weighted by 2/3, which maps the closed-domain -1 eigenvalue to -1/3. The closed-cavity correction now converges; measurements in docs/reports/pressure_correction_step5.md, section 5. A prerequisite for step 6, kept separate from it so the weighting is attributable on its own. Harness rows reproduced. |
| 2026-09-22 | ECR-001 step 6 delivered: staggered SIMPLE solver (src/solver_staggered.py, REQ-S04, REQ-S07) alongside the collocated one, selected by the harness --method label. Converges on the three default cases; staggered rows appended to benchmarks/results.jsonl, collocated rows reproduced. Per-cell imbalance at convergence above criterion 6's bound, and the Ghia v reference found not to match the published table; see docs/reports/staggered_integration_step6.md. Validation tests stay on the collocated solver until steps 7 and 8. |
| 2026-09-22 | Ghia v reference corrected to Table II as ghia_1982_re100_r2 (validation/metrics.py), guarded by a mass-conservation test with the corrupted table as its planted control. VAL-002 status corrected; ECR-001 erratum added (section 12); r2 rows appended for both solvers at 20x20, 40x40 and 80x80. Stored rows unchanged. |
| 2026-09-23 | VAL-002 metric changed to max_normalized_centerline_error_r2 (validation/metrics.py, cavity_true_centerline_errors), which samples the centerline profiles on x = 0.5 and y = 0.5 instead of half a cell off them. The harness, the viewer and tests/test_lid_cavity.py use it. The old metric and the stored rows are unchanged, and the harness summary keeps the two metrics apart. VAL-002 stays XFAIL. See docs/reports/cavity_self_convergence.md, section 9. |
| 2026-09-19 | VAL-001 recorded error corrected from 1.54% to 2.04% L2 on 80x40. The 1.54% figure was never reproducible: the test at commit d589b9f, whose message claims it, measures 2.036% and fails its own 2% assertion, and 2.036% is also what CI measured, so there is no platform difference between the two. The threshold moved from 2% to 2.5% in the same PR without a recorded reason. The reason is that the collocated ghost-cell scheme genuinely sits at 2.04% on this grid, above the 2% set on 2026-04-15. REQ-S02 keeps its 2.5% value. The ECR-001 plan to tighten it to < 1% after the rebuild is unaffected. |
| 2026-09-23 | Independent cavity reference marchi_2009_re100 (Marchi, Suero and Araki, 2009, Re = 100) added to validation/metrics.py, read from the paper's text layer by two routes and guarded in tests/test_validation.py by conservation and mass-flow checks with planted controls. scripts/self_convergence.py --marchi compares the extrapolated staggered centerlines with it; the remaining gap to Ghia is Ghia's (INFERRED). No metric, criterion or stored row changed, and VAL-002 stays on ghia_1982_re100_r2. See docs/reports/cavity_reference_marchi.md. |
| 2026-09-23 | Review action (.github/workflows/review.yml) removed. On each of its last three pull requests it ran for about a minute and never reached its review stage (no Opus sub-agents in its model usage, no comment), through two fix attempts. Review and test now run before the pull request, in fresh local Claude Code sessions, as /cfd-review and /cfd-test (.claude/commands/, now tracked); their first use found a Critical. Their reports are posted on the pull request as its record. ci.yml is unchanged. The repository secret and the GitHub App are left for removal after merge. |
| 2026-09-24 | Cavity reference decided by Alex: VAL-002 and ECR-001 criterion 3a score against marchi_2009_re100, with ghia_1982_re100_r2 reported beside them and no threshold (amendment under criteria 3 and 3a, ECR-001 section 9). The metric and the VAL-002 test change in step 8; until then they read ghia_1982_re100_r2. Stopping-rule evidence for step 7 added (scripts/stopping_probe.py): at the committed tolerance iteration error dominates VAL-001 80x40's stored metric, an estimate from the residual's own rate tracks the cavity's iteration error, and on the channel the error left once the pressure solve drops to a sweep or two is a flux drift that only the mass imbalance shows. No threshold, metric or stored row changed. See docs/reports/stopping_rule_evidence.md. |
| 2026-09-24 | Stopping rule for the staggered solver added ahead of ECR-001 step 7 (src/stopping.py): error_estimate stops when the estimated iteration error over the largest prescribed boundary velocity, the worst per-cell mass imbalance and the summed imbalance over the through-flow are all below their tolerances, and reports the iteration cap as not converged. Opt-in by the solver key stopping_rule; velocity_step, bitwise the old rule, stays the default, the solver block rejects unknown keys, and the collocated solver refuses error_estimate. The harness records the rule and takes converged and stop_reason from the staggered solver. REQ-S01 and REQ-S04 clarified, not amended. ECR-001 criterion 2 amended to fix the wall spacing with the ratio derived. No validation case, threshold, metric or stored row changed; the cases switch in step 7. See docs/reports/stopping_rule_evidence.md, section 9. |
| 2026-09-25 | ECR-001 step 7 delivered: VAL-001 revalidated on the staggered solver. configs/validation_poiseuille.yaml names stopping_rule error_estimate; tests/test_poiseuille.py adds the staggered test at < 1% on 80x40 uniform (criterion 1) and on the preset val001_80x40_stretched, y clustered to a wall cell of 0.1 H / ny (criterion 2), each asserting the stop by the rule; scripts/val001_order.py finds the reference-free order under uniform refinement 1.99 (criterion 4, judged without a reference as the ECR-001 note records). The collocated solver keeps velocity_step through validation.cases.with_velocity_step, its fields and rows unchanged, and the collocated test keeps 2.5%. The harness keys its summary on the stopping rule and gives a velocity-step stop one label. The first error_estimate rows are in benchmarks/results.jsonl. See docs/reports/val001_revalidation_step7.md. REQ-S02 text unchanged until step 9. |
| 2026-09-25 | ECR-001 step 8 delivered: VAL-002 revalidated on the staggered solver. configs/validation_cavity.yaml names stopping_rule error_estimate with max_simple_iter 20000; validation.metrics.cavity_marchi_centerline_errors (metric max_normalized_centerline_error_cubic) scores the true centerlines against marchi_2009_re100 at its stations by the cubic moved there from scripts/self_convergence.py, and the harness uses it; tests/test_lid_cavity.py adds the staggered test at < 2% on the case file's 40x40, asserting the stop by the rule. Criteria 3 and 3a pass from three rows at 20x20, 40x40 and 80x80, with Ghia reported beside them. The collocated test keeps its xfail and its Ghia metric through validation.cases.with_velocity_step, the self-convergence fields and collocated rows are unchanged, validation.cases.load_preset loads every named preset, and new harness rows record each axis's clustering. See docs/reports/val002_revalidation_step8.md. REQ-S03 text and the VAL-002 gate row unchanged until step 9. |
| 2026-09-30 | Stopping rule condition (d) added (src/stopping.py): error_estimate also requires the absolute signed domain sum of the per-cell imbalance below mass_imbalance_tol, ECR-001 criterion 6's domain-sum clause, which nothing checked before (review 27 B1). No configuration key. Cavity stops unchanged; channel stops later and criteria 1, 2 and 4 still pass, with rows for val001_80x40, val001_80x40_stretched and val002_80x80 appended. REQ-S04 clarified again, not amended; note under ECR-001 criterion 6. See docs/reports/stopping_rule_evidence.md, section 10. |
| 2026-09-30 | ECR-001 step 9: ADR-010 written, ADR-008 marked superseded, ECR-001 closed. REQ-S02 amended to < 1% and REQ-S03 to marchi_2009_re100 in docs/SYSTEM.md, REQ-S11 to the stretching built. Phase 2 deliverables and validation gate brought current: the staggered solver and its modules DONE, the collocated solver and boundary.py RETAINED as the harness baseline pending Alex's decision on retirement, VAL-001 and VAL-002 PASS at their amended criteria. Phase 2 at GATE REVIEW. A unit test now pins the cubic stencil of validation.metrics.lagrange (test 26b T1). |
