# Project Plan

**Project:** CFD Clean Room Simulation
**Last Updated:** 2026-10-08
**Current Phase:** Phase 3 (Transport Solver), in progress: design accepted (ADR-011, decisions of 2026-10-03); the product case blocked on ECR-002 (accepted 2026-10-04); Phase 2 complete

This document tracks development progress by phase. Code review reads this document to determine the current phase and verify that PRs are in scope; the policy is `docs/REVIEW_POLICY.md`. Update this document as work progresses.

---

## Phase Status Summary

| Phase | Name | Status | Gate Verdict | Report |
|-------|------|--------|-------------|--------|
| 0 | Infrastructure | COMPLETE | PASS | -- |
| 1 | Foundation | COMPLETE | PASS | phase1_foundation_report.md |
| 2 | Navier-Stokes Solver | COMPLETE | PASS | phase2_navier_stokes_report.md |
| 3 | Transport Solver | IN PROGRESS | -- | -- |
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
| src/solver_staggered.py | DONE (ECR-001 steps 6 to 8) | Steady SIMPLE on the staggered MAC grid. Stops by the velocity-step rule or, as both validation cases configure it, by the error_estimate rule. The solver of record (ADR-010). |
| src/staggered.py | DONE (ECR-001 step 2) | MAC field layout and face-to-center averaging (REQ-S07). |
| src/boundary_staggered.py, src/boundary_registry.py | DONE (ECR-001 step 3) | Direct Dirichlet imposition on the staggered faces (REQ-S12) over one shared reading of the configured boundaries (REQ-S12.1). |
| src/momentum.py | DONE (ECR-001 step 4) | Momentum predictor, QUICK by deferred correction over an upwind implicit matrix (REQ-S09). |
| src/pressure.py | DONE (ECR-001 step 5; the solve replaced in ECR-003 step 1, 2026-10-06) | Pressure correction by Jacobi-preconditioned conjugate gradients to a relative residual pressure_rtol, a rounding floor or a reported cap (REQ-S08 as amended 2026-10-06, ADR-013). The weighted Jacobi sweep of 2026-09-22 is history (docs/reports/pressure_correction_step5.md). |
| src/stopping.py | DONE (ahead of ECR-001 step 7) | The error_estimate stopping rule (REQ-S01, REQ-S04 as clarified 2026-09-24). |
| src/mesh.py | DONE (ECR-001 step 1) | Per-axis geometric wall clustering, mirrored about the midpoint (REQ-S11 as amended 2026-09-30). |
| src/solver_ns.py | RETIRED (2026-10-02, PR 29, tag collocated-final) | SIMPLE on the collocated grid with Rhie-Chow and ghost-cell walls (ADR-008). ECR-001's before-and-after baseline, deleted once the ECR closed; its 22 harness rows stay in benchmarks/results.jsonl. |
| src/boundary.py (velocity/pressure BCs) | RETIRED (2026-10-02, PR 29, tag collocated-final) | Collocated ghost-cell layer over src/boundary_registry.py, which stays. Retired with src/solver_ns.py. |
| tests/test_poiseuille.py | DONE | VAL-001: staggered at < 1% on 80x40 uniform and wall-clustered. |
| tests/test_lid_cavity.py | DONE | VAL-002: staggered at < 2% against marchi_2009_re100 on the case file's 40x40 in CI. |
| tests for the staggered modules | DONE | test_staggered.py, test_boundary_registry.py, test_boundary_staggered.py, test_momentum.py, test_pressure.py, test_solver_staggered.py, test_solver_selection.py, test_stopping.py. |
| tests/test_solver_ns.py, tests/test_boundary.py | RETIRED (2026-10-02, PR 29, tag collocated-final) | Unit and integration tests of the collocated solver and its boundary layer, deleted with them. |
| docs/ADR/ADR-010-staggered-grid-architecture.md | DONE (ECR-001 step 9) | The architecture as built, with planned against built. |
| scripts/view_field.py | DONE (development instrument) | Streamlines, pressure and cavity centerline profiles, for reading the field during the rebuild. Not the Phase 7 visualization deliverable; see Scope Changes. |
| Measurement instruments | DONE (development instruments) | scripts/benchmark.py and benchmarks/results.jsonl, validation/ (cases, references, metrics), scripts/self_convergence.py, scripts/stopping_probe.py, scripts/val001_order.py, scripts/gen_system_map.py. See Scope Changes. |

### Validation Gate

| Test ID | Description | Criterion | Status |
|---------|-------------|-----------|--------|
| VAL-001 | Poiseuille flow | L2 error < 1% on 80x40 (REQ-S02 as amended 2026-09-30), uniform and wall-clustered to 0.1 H / ny (ECR-001 criteria 1 and 2); observed order >= 1.8 under uniform refinement (criterion 4) | PASS on the staggered solver under staggered-cg (2026-10-07, ECR-003 step 2): 4.108e-4 uniform, 3.024e-3 clustered (docs/reports/ecr003_step2_baseline.md, section 4). Order 1.992, measured under the weighted Jacobi correction and not retaken; the CG-against-Jacobi field differences can move it by at most 0.003, and the realized change under CG is at most 3e-4 (section 10 there). History: under the weighted Jacobi correction, 4.107e-4 uniform, 3.024e-3 clustered, order 1.992 under stopping rule version 3 (docs/reports/val001_revalidation_step7.md, addendum). |
| VAL-002 | Lid-driven cavity | Maximum centerline error < 2% of the lid speed against marchi_2009_re100 at 80x80 (REQ-S03 as amended 2026-09-30, ECR-001 criterion 3); u and v errors each falling across 20x20, 40x40 and 80x80 (criterion 3a); ghia_1982_re100_r2 reported, unscored | PASS on the staggered solver under staggered-cg (2026-10-07, ECR-003 step 2): u 1.057e-3, v 7.356e-4 at 80x80 (docs/reports/ecr003_step2_baseline.md, section 4). Orders 2.24 and 2.11 (u), 2.12 and 2.07 (v), measured under the weighted Jacobi correction and not retaken; the CG-against-Jacobi field differences can move each by at most 6e-4, and the realized change under CG is at most 3e-4 (section 10 there). CI runs the 40x40 case file. History: under the weighted Jacobi correction, u 1.057e-3, v 7.356e-4 at 80x80 (docs/reports/val002_revalidation_step8.md). |

### Scope Changes

- C solver deliverables (csolver/pressure_solve.c, csolver.h, Makefile, test_c_parity.py) moved to Phase 6 (CUDA Acceleration). REQ-N03 is now validated against CUDA C++ rather than plain C.
- ECR-001 (approved 2026-04-16, closed 2026-10-01) replaced the collocated solver as the solver of record with a staggered one built alongside it. The collocated solver was kept as the harness baseline rather than rewritten in place; its retirement was deferred to Alex's decision, made on 2026-10-02 (retired in PR 29, tag collocated-final). REQ-S11 was amended to the per-axis, mirrored stretching that was built.
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
| docs/ADR/ADR-011-transport-solver-architecture.md | DONE | The Phase 3 design, Accepted at merge (PR 30); every module below is built from it. Its three open items and seven further points were decided by Alex on 2026-10-03 and are listed at its top |
| src/solver_transport.py | DONE (PR 32) | Built to ADR-011 B, C, D and F: the limited QUICK face value under forward Euler at cfl_number, implicit diffusion and deposition by Jacobi, settling on interior faces, sources, one MassBudget per class, FieldHistory. Contract in SYSTEM.md section 4 (built). The validation cases hand it stand-ins from validation/transport_cases.py |
| src/boundary_concentration.py (concentration BCs) | DONE (PR 31) | Beside the staggered boundary layer, reading boundary_registry.py's coverage_along, the one face-coverage derivation both layers share (decided 2026-10-02; ADR-011 E). Hands the transport solver read-only face data: carried inlet concentration, deposition velocity and surface code, settling mask |
| tests/test_boundary_concentration.py | DONE (PR 31) | Unit: surfaces at obstacle and edge faces, the settling mask and its count, the carried concentration with and without the HEPA factor. Integration: the two boundary layers agree on which faces are inlets on every committed configuration (REQ-S12.1) |
| configs/clean_room_default.yaml under error_estimate | BLOCKED on ECR-002 (2026-10-04) | Decision 6 of 2026-10-03: stopping_rule error_estimate with mass_imbalance_tol <= 1e-4 rho V_min / t_end and a cap above 500, set when the product case is solved and its stop measured; a precondition of REQ-T11 on the product case (ADR-011 G). The product room does not solve laminar at its Reynolds number (docs/reports/product_case_reynolds.md); it is solved in ECR-002 step 8 with the k-epsilon model, after ECR-002's outlet step and convergence measurement and ECR-003's pressure solve (ADR-012, decisions 1 and 5) |
| tests/test_constancy.py | DONE (PR 32; split 2026-10-07, ECR-003 step 1) | VAL-012, REQ-T11: a uniform field stays uniform to the per-cell bound the stopping rule's imbalance sets (ADR-011 G). Under weighted Jacobi: departure 2.758e-8 against a bound of 1.908e-7 (ratio 0.145) over 40.0 s on VAL-001 40x20. ECR-003's conjugate gradient correction balances those faces to rounding (worst |b_P| 6.2e-16 kg/s), so the departure is 4.0e-12 against a bound of 4.0e-11, ratio 0.100 (0.099 on Linux), and only the requirement's clause stays on them; the mechanism's clause runs on a planted field, uniform flow with one face column raised by 1e-4 m/s, ratio predicted 0.651 and measured 0.6513; the perturbed face drifts 6.7e-6 |
| tests/test_diffusion.py | DONE (PR 32) | VAL-003. Runs diffusion numbers 0.25 and 0.125, fits the first-order error split (time 66% at 0.25, 49% at 0.125), and runs the gate at 0.1 where the time share is 44% (ADR-011 H as amended). Measured relative L2 2.084e-3 at the gate |
| tests/test_advection.py | DONE (PR 32) | VAL-004, both rows; records the rotating puff through FieldHistory under pytest's tmp_path to check the format, and results/builder32/puff_visual.py runs the case itself to write the frames, the GIF and the PNG for Phase 7 (decision 3 of prompt 32b). Row 1 peak retained 0.8219, L2 0.1129, minimum 0, centroid error 0.024 cells; 684 steps; row 2 peak retained 0.7531, L2 0.1553, minimum 2e-52 of the peak, centroid error 0.055 cells; 3959 steps, 99 frames; the solver reproduces the prototype to four figures (results/builder32/item0.md) |
| tests/test_conservation.py | DONE (PR 32) | VAL-007 on a random face field and on the VAL-001 40x20 faces, every term on. Measured: relative residual -1.5e-13 on the random 64x36 field and -7.4e-15 on the VAL-001 40x20 faces after 500 steps with every term on, against the 1e-4 criterion; one inlet face dropped from the booking leaves 1.7e-2 and 3.2e-2 |
| tests/test_smith_hutton.py | DONE (PR 32) | VAL-013, REQ-T12: the field stays within the inlet's bounds at every step (ADR-011 H). Measured: minimum 4.122307e-9 at the bound, maximum 1.9999999923 under the bound 1.9999999959, over 8918 steps to the 20 s cap (last change 3.7e-10 of the inlet maximum); outlet profile against the inlet's mirror: largest difference 0.085 of a range of 2, relative L2 1.85%, unscored |
| tests/test_sealed_box.py | DONE (PR 32) | VAL-014: a sealed room loses mass at settling velocity over height, exact; guards the settling and deposition composition (ADR-011 D, H as amended). Measured: floor deposit exact to 4e-16 over 20 steps, budget residual 0, bounds held; the control plants the settling increment on the floor face and misses the line by 48% |
| tests/test_solver_transport.py | DONE (PR 32) | Unit and integration tests: the limiter, argument checks including NaN and bool inputs, stable_dt as the sum form, advection with the inlet value as the far node, the far node inside an obstacle, the domain face behind an edge obstacle, v_ext in both components on interior faces only, SOLID cells, implicit diffusion against a dense solve on a stretched mesh with the sweep-cap warning, settling and deposition composition, the implicit sink, sources, the budget's booking by surface, FieldHistory. Each planted defect of prompt 32 and each behaviour removal of review 32 B1 is killed by its test (results/builder32/mutation_log.md, results/builder32b/mutation_log.md) |

### Validation Gate

| Test ID | Description | Criterion | Status |
|---------|-------------|-----------|--------|
| VAL-003 | Pure diffusion | L2 error < 1% vs analytical Gaussian, at a step where the backward Euler error is below the spatial one (ADR-011 H) | PASS: relative L2 2.084e-3 at a diffusion number of 0.1 (375 steps), where the fit from the runs at 0.25 (3.451e-3) and 0.125 (2.311e-3) puts the time share at 44% and the spatial at 56%; budget residual 2e-15 (tests/test_diffusion.py). |
| VAL-004 | Pulse advection, oblique channel pulse (ADR-011 H, row 1) | Centroid error < 1 cell; peak retained > 73% and L2 error vs the exact translate < 17% (1.5 times the measured 82.2% and 11.3% of the chosen scheme at Courant number 0.1, results/builder30b/pulse_2d.json); no cell below zero | PASS: peak retained 0.8219, L2 0.1129, minimum 0, centroid error 0.024 cells; 684 steps (tests/test_advection.py). |
| VAL-004 | Pulse advection, rotating puff (row 2) | After one revolution: centroid error < 1 cell; peak retained > 62% and L2 error vs the initial field < 24% (1.5 times the measured 75.3% and 15.5%, rounded outward); no cell below zero | PASS: peak retained 0.7531, L2 0.1553, minimum 2e-52 of the peak, centroid error 0.055 cells; 3959 steps, 99 frames (tests/test_advection.py). |
| VAL-007 | Mass conservation | Imbalance < 0.01% of total mass | PASS: relative residual -1.5e-13 on the random 64x36 field and -7.4e-15 on the VAL-001 40x20 faces after 500 steps with every term on, against the 1e-4 criterion; one inlet face dropped from the booking leaves 1.7e-2 and 3.2e-2 (tests/test_conservation.py). Re-checked 2026-10-07 under ECR-003's CG correction: -1.065e-14 on the VAL-001 faces, the random field's -1.5e-13 unchanged (`docs/reports/ecr003_step2_baseline.md`, section 8). |
| VAL-012 | Constancy (REQ-T11) | On the solver's VAL-001 40x20 faces, the largest per-cell relative departure of a uniform field at most max_P |b_P| T / (rho V_P), and a perturbed interior face exceeding it; on a planted face field with a known imbalance, the departure at most the bound and at least a tenth of it (split 2026-10-07: the solver's faces are balanced to rounding since ECR-003 step 1, so the lower clause moved to the planted field) | PASS: solver's faces, departure 3.997e-12 against a bound of 3.996e-11 over 40.0 s (ratio 0.100; 0.099 on Linux), the perturbed face drifts 6.7e-6; planted field, departure 6.507e-4 against a bound of 9.990e-4, ratio 0.6513 against the predicted 0.651 (tests/test_constancy.py). Under weighted Jacobi the solver's faces gave 2.758e-8 against 1.908e-7, ratio 0.145. |
| VAL-013 | Smith-Hutton (REQ-T12) | Every cell within [1 - tanh(10), 1 + tanh(10)] of the reference concentration at every step, exact to rounding; outlet profile reported unscored | PASS: minimum 4.122307e-9 at the bound, maximum 1.9999999923 under the bound 1.9999999959, over 8918 steps to the 20 s cap (last change 3.7e-10 of the inlet maximum); outlet profile against the inlet's mirror: largest difference 0.085 of a range of 2, relative L2 1.85%, unscored (tests/test_smith_hutton.py). |
| VAL-014 | Sealed-box decay (ADR-011 D) | Floor deposit equals v_s C_0 W t to 1e-10 relative before the front reaches the floor; budget closes to rounding; no cell above C_0 or below zero (the doubled-floor clause dropped 2026-10-03, ADR-011 H as amended) | PASS: floor deposit 6.4000000000e6 against v_s C_0 W T 6.4000000000e6, relative difference 4e-16, over 20 steps of 51.4 s (T = 1028 s; the front at 8 of 30 cells, the leading edge 10 rows above the floor); budget residual 0; field within [0, C_0]; the settling increment planted on the floor face misses the line by 48% (tests/test_sealed_box.py). |

### ECR-002: the turbulence model (accepted 2026-10-04)

The product case waits for ECR-002 (`docs/ECR/ECR-002-turbulence-model.md`), whose design is
ADR-012. Its steps' deliverables and validation identifiers are tracked here as they land.

| Deliverable | Status | Notes |
|-------------|--------|-------|
| Step 0, the frozen-viscosity probe | DONE | `docs/reports/ecr002_step0_frozen_viscosity.md`: outcome B, then section 7's outcome A; ten momentum sweeps per outer iteration converge the coarse room across k-epsilon's range |
| src/scalar_scheme.py | DONE (step 1, prompt 35) | The transport solver's face value, advective flux and implicit solve moved into module functions both solvers call; the transport gate's fields bitwise main's at every step; the solve gains per-face conductances, a cell sink, a per-cell step and a held-cell mask |
| SimConfig.turbulence | DONE (step 1) | The optional section of ADR-012 I, validated per REQ-C02; absent means off; no solver reads it before step 6 |
| src/turbulence.py | DONE (step 1) | k and eps on a prescribed face field, standard and RNG (constants sourced, RNG's C_mu 0.0845 by Alex's decision of 2026-10-05); boundary values from a conditions object built each step; raises unless k and eps are positive and finite (REQ-S15) |
| tests/test_turbulence.py, tests/test_decaying_turbulence.py, tests/test_scalar_scheme.py | DONE (step 1) | Positivity on the Smith-Hutton and a random divergence-free field with production on, every implicit solve converged, against the planted explicit decay, and a long run whose solves are truncated; constancy on the VAL-001 faces; strain, the step's rate and the inflow on fields moving in y and in both directions, at walls and at horizontal and vertical obstacle faces; each sigma pinned; RNG's R split at S != 1; VAL-015. 21 planted defects, then 23 more from test 35 (results/builder35/mutation35.md, mutation35b.md), each fail a test on an assertion about the quantity |
| Step 2, transport coupling | DONE (prompt 40) | `solve_timestep` takes a keyword-only `eddy_viscosity` field; the face diffusivity is the Brownian coefficient plus nu_t / Sc_t with the harmonic-mean face value (REQ-T13); `transport.turbulent_schmidt` is an optional key with no default in code; None is the laminar path, bitwise. VAL-007, VAL-012 and VAL-013 rerun with a field (ECR-002 criterion 5); the tests are named in REQ-T13's verification column in `docs/SYSTEM.md` |
| Step 3, the outlets | DONE (prompt 42, 2026-10-08) | `fixed_flow_outlet` boundary type: an outward normal velocity and zero tangential velocity, stated or the equal share of the discrete inflow the stated outlets leave; the product configuration's four returns and hood use it, so the room has no pressure outlet and the corrector takes its closed-domain path (REQ-S18, ADR-012 decision 1 amended 2026-10-08). VAL-001 and VAL-002 hash identically to the base; the built room reproduces the probe's arm D0 bit for bit (`docs/reports/probe42/fixed42.py`). Review and test come before the pull request |
| Steps 4 to 8 | NOT STARTED | The momentum coupling, the convergence measurement, the coupled solve, the validation cases, the product (ECR-002 section 8). Step 5, the first to solve the 200x75 room to tolerance, waited for ECR-003 step 2 (ADR-012 decision 5; ECR-003 section 8, tracked below); the dependency is met, ECR-003 closed 2026-10-07. Step 5 retakes step 0's sweep result, measured under the weighted Jacobi correction, with the pressure solved by CG (ECR-002 section 8, step 5's note) |

| Test ID | Description | Criterion | Status |
|---------|-------------|-----------|--------|
| VAL-015 | Decaying turbulence (ECR-002 criterion 2) | In a closed box at rest, k and eps follow the exact decay with an error falling at first order in dt, observed order 0.9 to 1.1 over three steps in a ratio of two; field uniform to rounding | PASS for both variants: observed order of k 0.968 and 0.985 (standard), 0.946 and 0.974 (RNG); of eps 1.088 and 1.046 (standard), 1.046 and 1.023 (RNG), at dt of 1/40, 1/80 and 1/160 of 2 k0 / eps0; spread at most 1.0e-13 (tests/test_decaying_turbulence.py). The step sizes were set after a first run from 1/20, where the coarse pair fell outside the band twice, RNG's k at 0.887 and the standard eps at 1.163, the dt^2 term still visible (test 35 reproduced both) |
| VAL-016 | Plane Couette flow (criterion 4) | ECR-002 section 9 | NOT RUN (step 6) |
| VAL-017 | The Annex 20 room | OPEN by decision until the first coupled results | NOT RUN (step 7) |
| VAL-018 | The product room converges | Conditional on step 5 | NOT RUN (step 8) |
| VAL-019 | The backward-facing step | OPEN by decision until the first coupled results | NOT RUN (step 7) |

### ECR-003: the pressure solve (accepted 2026-10-06, closed 2026-10-07)

ECR-002 step 5 waited for ECR-003 (`docs/ECR/ECR-003-pressure-solver.md`), whose design is
ADR-013. Its steps are tracked here as they landed; each went through `/cfd-review` and
`/cfd-test`, steps 2 and 3 in one pull request (ECR-003 section 8, note). Closed 2026-10-07: ECR-002
step 5's dependency is met.

| Deliverable | Status | Notes |
|-------------|--------|-------|
| Step 1, the conjugate gradient solve | DONE (PR 62, merged 2026-10-07; prompts 37, 37b and 37c) | `src/pressure.py` solves the correction by Jacobi-preconditioned CG with ADR-013's stop, floor, true-residual check, reported cap and component checks; `pressure_rtol` replaces `pressure_tol`; the method label `staggered-cg`, looked up by `PRESSURE_SOLVER_VERSION`; every saved solve keyed on the solver's identity; criteria 1, 3 and 4 (`docs/reports/ecr003_step1_cg.md`). Review 37 and test 37's findings answered in the fix pass of prompt 37b; VAL-012 split by prompt 37c; test 37b's two missing tests deferred to issue 63 |
| Step 2, the laminar baseline | DONE (prompt 38, 2026-10-07) | Six `staggered-cg` rows at 311034e, two each of `val001_80x40`, `val001_80x40_stretched` and `val002_80x80`, each stopping by `error_estimate_and_continuity` with no capped correction; VAL-001 and VAL-002 pass (the gate rows above); the six transport gate files pass, 13 tests. The channel outer counts fell 61% and 50% from the Jacobi rows, past the prompt's 5%, and the step stopped: the Jacobi correction had left most of the imbalance in the faces. Alex accepted the rows as ECR-002 criterion 1's baseline on 2026-10-07 (`docs/reports/ecr003_step2_baseline.md`) |
| Step 3, the records | DONE (prompt 38, 2026-10-07) | ADR-013 Accepted (built) with planned against built; SYSTEM.md, this plan, STATUS.md and ECR-002 criterion 1 updated; the request closed. Dated notes: the net-outflow oscillation was the Jacobi correction's (ADR-010, `docs/reports/stopping_rule_evidence.md` section 10, SYSTEM.md); the Jacobi-era orders bounded, not retaken (the gate rows above); the cavity's cost (ADR-013 and the efficiency pass below); machine-specific face hashes (ECR-002 criterion 1); ECR-002 step 5 retakes step 0's sweep result under CG |

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

Note (issue 53 D6, 2026-10-07): on a stretched mesh the transport limiter's clamp is applied to the quadratic `quick_face_values` returns, with the other three limiter terms in their uniform-mesh form (ADR-011 B). The stretched case is tested for diffusion against a dense solve and not for advection accuracy, which VAL-008 measures here.

### Validation and Showcase Case: the Square Cylinder (Alex, 2026-10-04)

Laminar flow past a square cylinder in a channel is Phase 4's validation and showcase case, since it needs the time-accurate solve this phase builds. Reference: Sohankar, Norberg and Davidson (1998), "Low-Reynolds-number flow around a square cylinder at incidence: study of blockage, onset of vortex shedding and outlet boundary condition", Int. J. Numer. Meth. Fluids 26, 39-56 (https://www.cfd-sweden.se/lada/postscript_files/Sohankar_num-fluids.pdf). At 5% solid blockage and zero incidence the onset of vortex shedding is at Re_cr = 51.2 +/- 1.0 on the cylinder's side (page 51 and Table V; Norberg's near-zero-blockage experiments give 47 +/- 2, the +/- 2 Alex's note carried), and the Strouhal number at Re 100 is 0.146 (Table IV, case 1; 0.147 in Table II's case 1). The showcase runs Re 40 (steady, below onset), 60 and 100 (shedding). The criteria are set when Phase 4 is designed; ECR-002's backward-facing step (VAL-019) is the turbulent separation case, this one the laminar unsteady one.

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
| csolver/pressure_solve.cu | NOT STARTED | CUDA kernel for the conjugate gradient pressure correction: the five-point product and the diagonal scaling one thread per cell, three reductions per iteration, run for the NumPy loop's iteration count per correction in the REQ-N03 test (ADR-013 C; ECR-003, accepted 2026-10-06) |
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

Note (issue 53 D5, 2026-10-07): `FieldHistory.record` keeps frames at `step % every == 0`, so the last frame of the rotating puff is step 3920 of 3959. An animation that wants the final frame records it explicitly or chooses `every` to divide the step count.

### Gate Criteria

- Visual outputs reviewed for physical plausibility
- Portfolio page published to alexyms.github.io
- All phase completion reports committed
- All validation tests pass
- README complete with setup instructions and results

---

## Deferred: the Efficiency and Clean-up Pass

Work deferred to a later pass rather than to a phase. Each line is a candidate, not a decision,
with where it was recorded.

| Candidate | Recorded | Notes |
|-----------|----------|-------|
| An outer iteration that adapts rather than overshoots: an outflow correction, or under-relaxation that adapts to the damping the solver observes | 2026-10-02 (`docs/STATUS.md`, open questions) | Note 2026-10-07: the net-outflow oscillation that raised it belonged to the weighted Jacobi correction; under ECR-003's conjugate gradients it is gone on the validation channels (`docs/reports/ecr003_step2_baseline.md`, section 6). Whether the outer iteration overshoots on the product case is not measured |
| A pressure tolerance loose early and tight near the stop | 2026-10-07 (ECR-003 step 3) | On the 80x80 cavity CG at the default `pressure_rtol` of 1e-8 takes between about 2% and 18% more wall time than the two weighted Jacobi rows of identical arithmetic, inside the 7% to 8% spread between identical runs, the outer counts within 0.27% (`docs/reports/ecr003_step2_baseline.md`, sections 4 and 5; ADR-013, planned against built). Not measured. ADR-013 B measured a loose level from rest diverging in the 40x15 and 80x30 rooms at real air (1e-1), and its decision 3 ranks the reverse order, a tight start before a looser level; either needs the outer loop measured first |

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
| 2026-09-30 | ECR-001 step 9: ADR-010 written, ADR-008 marked superseded. REQ-S02 amended to < 1% and REQ-S03 to marchi_2009_re100 in docs/SYSTEM.md, REQ-S11 to the stretching built. Phase 2 deliverables and validation gate brought current: the staggered solver and its modules DONE, the collocated solver and boundary.py RETAINED as the harness baseline pending Alex's decision on retirement, VAL-001 and VAL-002 PASS at their amended criteria. Phase 2 at GATE REVIEW. A unit test now pins the cubic stencil of validation.metrics.lagrange (test 26b T1). |
| 2026-10-01 | ECR-001 closed at step 9, criterion 6 decided. Phase 2 complete, gate PASS: all six gate criteria hold (docs/reports/phase2_navier_stokes_report.md). Line coverage 98.2% against Phase 1's 95%. ECR-001 criterion 6 decided: condition (d) accepted as built, met on the channel at a zero crossing of the net outflow (ECR-001 section 9). The SYSTEM.md cascade rows and contract headings brought in line with the import graph (review 27 B2). The collocated solver's retirement and an outer iteration that does not overshoot are the first questions of Phase 3. |
| 2026-10-02 | Collocated solver retired (PR 29), the first Phase 3 pull request and Alex's decision of 2026-10-02: src/solver_ns.py, src/boundary.py, tests/test_solver_ns.py and tests/test_boundary.py deleted; annotated tag collocated-final on 98f8b1f, the last commit holding them; IterationState moved to src/stopping.py with its fields unchanged. The harness and the viewer default to staggered-jacobi, and the 22 stored collocated rows still summarize. The Phase 3 concentration boundary deliverable renamed to src/boundary_concentration.py, beside the staggered layer over boundary_registry.py. The adaptive outer iteration is deferred to a later efficiency and clean-up pass, not Phase 3. Line coverage reads 97.7% at the retirement commit (1200 statements, 28 uncovered) against 98.2% at the Phase 2 gate: no line lost coverage; 480 fully covered statements left the denominator. Alex decided on 2026-10-02 that Phase 3's gate criterion 4 compares against 97.7%, the figure at the retirement commit (review 29, B1). |
| 2026-10-03 | ADR-011, the Phase 3 transport design, proposed (PR 30): concentration at cell centres advected by the staggered face velocities, which the NS solver will expose (REQ-S13, proposed); explicit advection, implicit diffusion and deposition; settling as an interior face increment with the floor taking the deposition velocity whole; one mass budget; REQ-T11 (constancy) proposed with its bound and tests/test_constancy.py as a gate row; the product mass_imbalance_tol derived as 1e-4 rho V_min / t_end. Three items OPEN for Alex: the face scheme and positivity (forward Euler with unlimited QUICK is unstable at every Courant number), VAL-004's shape criterion, the supply concentration. Phase 3 IN PROGRESS. No module changed. |
| 2026-10-03 | ADR-011's three OPEN items closed and the document Accepted at merge (PR 30, builder-fix pass on premise review 30 and test 30; Alex's decisions of 2026-10-03). The face scheme: QUICK bounded by the UMIST limiter, forward Euler, Courant number at most 1/2; the validation cases run at 0.1 because the shape error grows with the Courant number (1D L2 8.6% at 0.1, 31% at 0.4, results/builder30b). VAL-004 split into two rows, an oblique channel pulse and a rotating puff, with thresholds at 1.5 times the measured error. Clean supply by default, no recirculation; particles enter as sources through a new sources argument on solve_timestep. REQ-T12 (positivity) added with VAL-013 Smith-Hutton; VAL-014 the sealed box guards the settling composition after test 30 found settling counted twice on obstacle tops; VAL-012 names the constancy test, replacing the REQ-T11 gate row. The product configuration moves to error_estimate when the product case is measured. A FieldHistory writer for Phase 7. No module changed; the register pin widened for REQ-T12. |
| 2026-10-03 | The first Phase 3 build (PR 31, branch phase3/face-velocities-and-concentration-boundary), three commits: StaggeredSolver exposes face_velocities as a read-only FaceVelocities at the end of every solve (REQ-S13), re-measured on VAL-001 40x20 from the faces alone with the worst cell and the signed sum below 1e-10; SimConfig gains the optional transport section (cfl_number, advection_scheme, max_diffusion_iter, diffusion_tol) and the segment keys concentration, hepa_filtered and deposition_surface, and the product supply is hepa_filtered with no concentration, a clean supply; src/boundary_concentration.py built over boundary_registry.coverage_along, the one face-coverage derivation the velocity layer now reads too, with a test that the two layers agree on every committed configuration. No transport solver, no scalar computation. Next: src/solver_transport.py and the six Phase 3 tests. |
| 2026-10-03 | PR 31's builder-fix pass on review 31 (one Bug, three instances: the SOLID-edge rule, edge_cells and the deposition_surface overrides had no test that could fail) and test 31 (confirmed by mutation). Alex's decisions: unknown segment keys, overlapping segments on one edge, and a concentration on a zero-normal velocity_inlet fail the load; a zero-normal inlet is a wall to the scalar layer (ADR-011 E amended), so the two layers' inlet sets are equal on every configuration, the cavity included; size clauses are counted in code lines. One derivation of the coverage inputs (staggered.edge_cell_inputs) for both layers; get_inlet_flux reads the shared coverage. Staggered outputs bitwise unchanged on test 31's random configurations that still load. |
| 2026-10-03 | The transport solver built (PR 32, branch phase3/transport-solver), five commits each green on its own: the shared validation material and the scalar core; settling, deposition, sources and the budget; VAL-003, VAL-004 (two rows), VAL-012 and VAL-013; VAL-007 and VAL-014; the records and the first visual (the rotating puff as a GIF under results/builder32/). Item 0 first: the real solver reproduces the prototype's VAL-004 figures to four figures on both rows, UMIST and upwind (results/builder32/item0.md). Six gate rows PASS with the measured value beside each criterion. One finding for decision: ADR-011 H's doubled-floor control for VAL-014 cannot hold under the implicit deposition sink the same ADR chooses (the floor row relaxes toward C_0 / 2 and the deposit rises by about 6%, not 100%); the composition it was meant to guard is guarded by planting the increment on the floor face instead, which fails the exact line by half. The product configuration's move to error_estimate is the remaining Phase 3 deliverable. |
| 2026-10-03 | PR 32's builder-fix pass on review 32 (two Bugs: five solver behaviours no test could fail on; relative() raising TypeError before the first step) and test 32 (both confirmed by surviving mutants; S9 refuted). Alex's decisions: VAL-014's doubled-floor clause dropped from ADR-011 H, the plan and the tests, the floor-face control named as the composition's guard; VAL-003's gate step chosen by the measured error split (diffusion number 0.1, time share 44%); tests write no deliverable under results/, the puff files come from results/builder32/puff_visual.py running the case. Five B1 tests each killed by the behaviour's removal; NaN and bool inputs refused before any arithmetic; protocols for what the solver reads; the integration tier used for the tests that run the real boundary layer. Item 0 reproduces bitwise. |
| 2026-10-04 | The product room found unsolvable by the laminar flow solver: Reynolds number 89,500 on its height and 1,190 per cell on the product mesh, against 5 and 100 for the validation cases; as committed the solve diverges by outer iteration 58, and on a 40x15 copy with the pressure solved further it diverges at the real viscosity and at ten times it and converges at a thousand times; heavier under-relaxation keeps it bounded without converging in 3,000 iterations (docs/reports/product_case_reynolds.md). One pressure correction on the product mesh needs 27,408 sweeps at the committed tolerance against a cap of 200. Alex decided to add a k-epsilon model (decision 1). ECR-002 proposed (docs/ECR/ECR-002-turbulence-model.md): ADR-004 superseded, turbulence modelling moved into scope by SYSTEM.md section 5's process, REQ-S14 to S17 and REQ-T13 proposed, seven steps. ADR-012 proposed (docs/ADR/ADR-012-turbulence-model.md) with seven decisions for Alex. Item 0: the Annex 20 specification and profiles obtained from Aalborg; the measured symmetry plane carries 1.02 to 1.32 of the inlet flux at x/H 1.0 and 0.60 to 0.64 at x/H 2.0, the plane z/W 0.4 1.11 to 1.13 there, so the model room was three-dimensional at x/H 2.0 and its scoring is decision 5. Phase 3's product-case deliverable BLOCKED on ECR-002; the seven gate rows stand. SYSTEM.md unchanged: requirement and scope text change when the ECR is accepted. No module changed. |
| 2026-10-04 | Prompt 33b, the fix pass on premise review 33 and test 33. The outlet measurement (docs/reports/product_case_reynolds.md section 8): air enters the room through the pressure outlets in every run that diverges and, at ten times the viscosity, the divergence sits at the hood exhaust; holding inward-turning outlet faces shut, fixing the hood's flow at 0.5 m/s, or both, moves and delays the divergence and converges nothing above a hundred times the viscosity, and on the product mesh the growth starts inside the room before any outlet face reverses. ECR-002 gains an outlet step (REQ-S18) and a convergence measurement before its first coupled solve, VAL-016 becomes plane Couette flow, the backward-facing step joins as VAL-019 (Alex, decision 3 of 2026-10-04) and VAL-018 is conditional; nine steps. ADR-012's decisions rewritten, eight with the outlets first. Rong and Nielsen (2008) fetched: one published standard k-epsilon prediction on the Annex 20 lines and no spread; no sourced standard k-epsilon range for the step either, so both thresholds stay OPEN. The square-cylinder case recorded as Phase 4's validation and showcase (Alex, decision 4 of 2026-10-04). No module changed. |
| 2026-10-04 | ADR-012's decisions taken by Alex: the outlets as T3, the hood's tangential velocity held at zero; both k-epsilon variants built, RNG for the product; scalable wall functions; k and eps by the transport scheme in pseudo-time; ECR-003 for the pressure solve, before ECR-002 step 5; the Annex 20 and backward-facing step thresholds set from the first coupled results; turbulent deposition deferred; Sc_t 0.7 configurable and the supply's turbulence per inlet; and a ninth, ECR-002 step 0, a risk-retirement probe at a frozen zero-equation eddy viscosity before anything is built. `/cfd-test 33b`'s findings, all text, applied by the orchestrator; its continued runs recorded in the report's section 8.6. No module changed. |
| 2026-10-04 | ECR-002 and ADR-012 accepted by Alex. ADR-004 superseded. SYSTEM.md's requirement and scope text (REQ-S14 to S18, REQ-T13, the amended REQ-S01, S10 and T01, section 5's scope) follow in ECR-002 step 0's pull request, as that edit moves the requirement register the tests pin. No module changed. |
| 2026-10-04 | ECR-002 step 0 (prompt 34). SYSTEM.md carries ECR-002 section 5's accepted text: REQ-S01 clarified, REQ-S10 and REQ-T01 amended, REQ-S14 to S18 and REQ-T13 added, none built yet; turbulence modelling in scope as k-epsilon; ADR-004 superseded by ADR-012; the register pin in tests/test_system_map.py widened. The frozen-viscosity probe (docs/reports/ecr002_step0_frozen_viscosity.md): on the 40x15 room under T3, the indoor zero-equation eddy viscosity at its published size, a core median of 8.2e-3 m^2/s, converges by the solver's own stop; scaled to core medians of 1.5e-3, 5e-4 and 1.5e-4 m^2/s it does not with one momentum sweep per outer iteration (the top of the range oscillates without departing, the middle and bottom grow), and with ten sweeps the top of the range converges. Upwind stalls by the report's rule; the stress source and its form do not decide the outcome; the pressure cap is not the cause. Outcome B: the result goes to Alex with step 5's aids ranked, more momentum sweeps first, before step 1 opens. ECR-002's step 0 row and ADR-012 decision 9 amended to what ran. No module changed. |
| 2026-10-05 | ECR-002 step 0, prompt 34b (Alex's decision of 2026-10-04): ten momentum sweeps per outer iteration at the middle and the bottom of k-epsilon's core range, core medians of 5e-4 and 1.5e-4 m^2/s, on the 40x15 room under T3 with the frozen zero-equation field. Both converge by the solver's own stop, at 1,209 and 1,205 outer iterations beside the top's 1,240, where one sweep grows to 44 and 68 m/s; the fifty-sweep reruns were not needed. Outcome A of the report's section 7: the sweep aid holds across the range on this grid, and by that outcome's terms step 1 opens and a configurable momentum sweep count, default 1, is built with step 4's viscosity field and used from step 5 on. ECR-002's step 0 row, the report's aid 1 and STATUS.md's step 0 paragraph extended by a sentence each. No module changed. |
| 2026-10-05 | ECR-002 step 1 (prompt 35, continued after item 0's stop): the transport scheme moved into src/scalar_scheme.py, bitwise on the transport gate; the turbulence configuration section; src/turbulence.py, k and eps on a prescribed face field, standard and RNG. Item 0 checked every constant against its source; RNG's C_mu stays 0.0845 against the preprint's printed ~0.085 (Alex, 2026-10-05). The orchestrator's build choices: eps held exactly by a mask, the conditions object passed every step, the edges' tangential velocities in it, ADR-012 F's harmonic face rule. VAL-015 PASS for both variants. A held eps that does not follow k runs away beside a shear, so step 6 rebuilds the wall cells' conditions every outer iteration. No module coupled; momentum.py, pressure.py, solver_staggered.py and every YAML unchanged. |
| 2026-10-05 | ECR-002 step 1 fix pass (prompt 35b) on review 35 (3 Bugs, 7 Suggestions) and test 35 (5 Bugs, 3 Suggestions): tests for the y-direction half of the strain, the step's rate and the inflow, for the state's and the faces' refusals and eps's positivity branch, for a true-time step uniform over every cell, for each sigma, for RNG's R at S != 1 and for the stable-step bound within 2%; the state's nu_t checked and the face differences masked beside SOLID cells; the random-field positivity run split into a converged run and a truncated long run; VAL-015's row records the 1/20 run. The 23 mutants of test 35 each fail a named test; the transport hashes still equal main's. The branch renamed feature/ecr002-k-epsilon-scalar. No module beyond src/turbulence.py changed. |
| 2026-10-06 | ECR-003 accepted by Alex with ADR-013's six decisions, each as ranked first (docs/ECR/ECR-003-pressure-solver.md, docs/ADR/ADR-013-pressure-solver.md, docs/reports/pressure_solver_ecr003.md): Jacobi-preconditioned conjugate gradients in NumPy, NumPy only, the relative residual pressure_rtol at 1e-8 with a rounding floor and the cap reported, pressure_tol refused, max_pressure_iter the CG cap at 5,000, C deferred to Phase 6, the outlet and finer-grid findings to ECR-002 steps 3 and 5. Step 1 built the same day (prompt 37, branch feature/ecr003-pressure-cg): src/pressure.py solves by CG with the stop, floor, true-residual check, cap flag and one-component check of ADR-013; the key, the renamed counts, pressure_cap_hits and the velocity_step refusal; every consumer of ECR-003 section 7 moved, the method label staggered-cg, saved solves keyed on PRESSURE_SOLVER_VERSION or the label, the one solver-key list (issue 38). Item 0: the built CG reproduces the probe's on the three recaptured 200x75 systems bit for bit. Criterion 3: 1,209 and 2,822 on 40x15, 233 and 588 on 80x30 at Re 90, the report's counts exactly. Criterion 4: about 0.15 s per correction on 200x75 (docs/reports/ecr003_step1_cg.md). REQ-S08 amended and REQ-S04 clarified in SYSTEM.md. Phase 6's pressure_solve.cu becomes a CG kernel. VAL-001 and VAL-002 are retaken in step 2; the laminar baseline rows of step 2 replace the staggered-jacobi rows as ECR-002 criterion 1's. |
| 2026-10-06 | ECR-003 step 1 fix pass (prompt 37b) on review 37 (1 Bug, 9 Suggestions) and test 37 (2 FAILs, 2 Suggestions): stopping_probe's verify_rule and analyse read the truth through solve_truth, so a weighted-Jacobi truth is solved again rather than served; the control, the truth readers and self_convergence's file names tested so only their own check stands between a call and a reused file; val001_order's key carries PRESSURE_SOLVER_VERSION; the method label defined once in src/pressure.py and looked up by version; an open domain with a component no outlet reaches refused at construction (every committed configuration and case still accepted); conjugate_gradient's arguments checked; the work counted in operator products, the exit checks included, under harness schema 2; the floor tested at its real value on the VAL-002 cavity's last system. Criteria 3 and 4 rerun and unchanged. ECR-003 tracked in its own section. |
| 2026-10-07 | ECR-003 steps 2 and 3 (prompt 38), the request closed; step 1 merged as PR 62 the same day. Step 2: six `staggered-cg` rows at 311034e; VAL-001 4.108e-4 and 3.024e-3, VAL-002 u 1.057e-3 and v 7.356e-4, every stop by `error_estimate_and_continuity` with no capped correction; the transport gate passes (VAL-007's VAL-001-faces residual -1.065e-14). The channel outer counts fell 61% and 50% and the step stopped; the Jacobi correction had left most of the imbalance in the faces, and Alex accepted the rows as ECR-002 criterion 1's baseline (docs/reports/ecr003_step2_baseline.md). Step 3: ADR-013 built, ECR-003 closed, the VAL-001 and VAL-002 gate rows carry the new values with the old as history and the bound on each Jacobi-era order (0.003, 6e-4); the efficiency pass gains its own section with a second candidate, a pressure tolerance loose early and tight near the stop; ECR-002 step 5's dependency met. Found on the way: above about 10,000 cells OpenBLAS's default threads make a 200x75 correction take about 1.05 s against 0.14 s on one thread, so ECR-003 criterion 4 holds on one thread only; left to Alex. | Alex Moroz-Smietana |
| 2026-10-07 | The cleanup pull request (prompt 39): GitHub issues 33, 36, 40, 42, 45, 47, 51 and 63 closed, issue 53's small items taken (its D5 and D6 are the notes under Phase 7 and Phase 4); issues 53 (D3, D4), 58 and 61 stay open with the Phase 3 gate report, ECR-002 step 6 and step 3. The BLAS thread limit decided on 2026-10-07 is built (`docs/reports/blas_threads.md`). ECR-001 step 9 is dated 2026-09-30 for its documents and 2026-10-01 for the closure in the 2026-09-30 and 2026-10-01 rows above. |
| 2026-10-07 | ECR-002 step 2 built (prompt 40, branch feature/ecr002-transport-coupling): the eddy viscosity field in the transport solver and `transport.turbulent_schmidt`. Review and test 40 come before the pull request. Details are in `docs/SYSTEM.md` (the `solver_transport.py` contract and the 2026-10-07 change log row). |
| 2026-10-08 | ECR-002 step 3 built (prompt 42, branch feature/ecr002-fixed-flow-outlets): the fixed-flow outlet, ADR-012 decision 1 amended by Alex (option 2). The product room's returns and hood are fixed-flow outlets and #61 closes with the pull request. Review and test 42 come before it is opened. Details are in `docs/SYSTEM.md` (the boundary contracts, REQ-S18 and the 2026-10-08 change log row). |
