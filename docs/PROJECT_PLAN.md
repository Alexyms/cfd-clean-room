# Project Plan

**Project:** CFD Clean Room Simulation
**Last Updated:** 2026-04-15
**Current Phase:** Phase 2 (Navier-Stokes Solver)

This document tracks development progress by phase. The automated code review system reads this document to determine the current phase and verify that PRs are in scope. Update this document as work progresses.

---

## Phase Status Summary

| Phase | Name | Status | Gate Verdict | Report |
|-------|------|--------|-------------|--------|
| 0 | Infrastructure | COMPLETE | PASS | -- |
| 1 | Foundation | COMPLETE | PASS | phase1_foundation_report.md |
| 2 | Navier-Stokes Solver | IN PROGRESS | -- | -- |
| 3 | Transport Solver | NOT STARTED | -- | -- |
| 4 | Scenarios & Time Integration | NOT STARTED | -- | -- |
| 5 | Alert Monitoring System | NOT STARTED | -- | -- |
| 6 | Visualization & Portfolio | NOT STARTED | -- | -- |

Status values: NOT STARTED, IN PROGRESS, GATE REVIEW, COMPLETE

---

## Phase 0: Infrastructure

**Goal:** Repository setup, CI/CD pipeline, coding standards, project documentation.

### Deliverables

| Deliverable | Status |
|-------------|--------|
| GitHub repository (public) | DONE |
| claude.md (coding standards) | DONE |
| docs/SYSTEM.md (architecture) | DONE |
| docs/PROJECT_PLAN.md (this file) | DONE |
| pyproject.toml (ruff config, pytest markers) | DONE |
| requirements.txt | DONE |
| requirements-dev.txt | DONE |
| .gitignore | DONE |
| .github/workflows/review.yml (code review action) | DONE |
| .github/workflows/ci.yml (lint + test action) | DONE |
| Review system prompt and script | DONE |
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

| Deliverable | Status | Notes |
|-------------|--------|-------|
| src/solver_ns.py | DONE | SIMPLE algorithm, pure NumPy, collocated grid with Rhie-Chow |
| src/boundary.py (velocity/pressure BCs) | DONE | Inlet, outlet, no-slip wall, ghost cell interpolation |
| tests/test_poiseuille.py | NOT STARTED | VAL-001 |
| tests/test_lid_cavity.py | NOT STARTED | VAL-002 |
| tests/test_solver_ns.py | DONE | Unit + integration tests: coefficients, sweeps, convergence |

### Validation Gate

| Test ID | Description | Criterion | Status |
|---------|-------------|-----------|--------|
| VAL-001 | Poiseuille flow | L2 error < 1% vs analytical parabolic profile | NOT RUN |
| VAL-002 | Lid-driven cavity | Centerline profiles within 2% of Ghia et al. | NOT RUN |

### Phase-Specific Risks

| Risk | Mitigation |
|------|------------|
| SIMPLE convergence failure on clean room geometry | Start with empty channel, add obstacles incrementally. Implement under-relaxation. |
| Python/C interface memory bugs | Validate C output against pure NumPy reference for all validation cases before switching to C path. |

---

## Phase 3: Transport Solver

**Goal:** Implement advection-diffusion equation for particle concentration transport. Validate diffusion, advection, and conservation independently.

**Branch prefix:** `phase3/`

**Depends on:** Phase 2 complete (NS solver produces velocity field)

### Deliverables

| Deliverable | Status | Notes |
|-------------|--------|-------|
| src/solver_transport.py | NOT STARTED | Advection-diffusion solver with v_ext interface |
| src/boundary.py (concentration BCs) | NOT STARTED | Extension to existing boundary module |
| csolver/advection_diffusion.c | NOT STARTED | C inner loop for transport stencil |
| tests/test_diffusion.py | NOT STARTED | VAL-003 |
| tests/test_advection.py | NOT STARTED | VAL-004 |
| tests/test_conservation.py | NOT STARTED | VAL-007 |
| tests/test_solver_transport.py | NOT STARTED | Unit tests: source terms, settling integration |
| tests/test_c_transport_parity.py | NOT STARTED | Integration: C output matches NumPy reference |

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

## Phase 6: Visualization & Portfolio

**Goal:** Produce animated visualizations, build the portfolio website page, and finalize all documentation.

**Branch prefix:** `phase6/`

**Depends on:** Phase 5 complete (alert data available for overlay)

### Deliverables

| Deliverable | Status | Notes |
|-------------|--------|-------|
| scripts/visualize.py | NOT STARTED | Matplotlib animations, streamlines, heatmaps |
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
