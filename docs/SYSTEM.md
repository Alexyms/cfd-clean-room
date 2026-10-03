# System Architecture Document

**Project:** CFD Clean Room Simulation
**Status:** ECR-001 closed; the staggered Navier-Stokes solver is built and validated and is the only solver (ADR-010; the collocated solver was retired on 2026-10-02, tag `collocated-final`). Phase status is in `docs/PROJECT_PLAN.md`.
**Last Updated:** 2026-10-03

This document is the single reference for system architecture, requirements, module interfaces, and dependency relationships. Review and test, run before each pull request as `/cfd-review` and `/cfd-test` in fresh Claude Code sessions (`.claude/commands/`), check branches against this document under the policy in `docs/REVIEW_POLICY.md`. Keep it current.

---

## 1. System Description

A from-scratch Computational Fluid Dynamics engine simulating clean room airflow and contamination transport. The system solves incompressible Navier-Stokes equations for a 2D velocity field using the Finite Volume method, then solves advection-diffusion equations for particle concentration across five size classes on top of that velocity field. An alert monitoring layer tracks contamination against ISO 14644 thresholds for reactive detection and proactive sensor placement analysis.

The simulation domain is a vertical cross-section of a semiconductor clean room with HEPA supply vents, return vents, an entry door, process equipment, and a laminar flow hood.

---

## 2. Requirements

Requirements are organized by subsystem. Each requirement has a unique ID, a rationale, and a traceability link to the validation test or architectural rule that verifies it.

### 2.1 Solver Requirements

Since ECR-001 step 9 (2026-09-30), "the NS solver" in this table is the staggered solver, `src/solver_staggered.py` (ADR-010), and since 2026-10-02 it is the only solver: the collocated solver it replaced was retired once the ECR had closed with it as the before-and-after baseline (section 4, Retired modules).

| ID | Requirement | Rationale | Verified By |
|----|-------------|-----------|-------------|
| REQ-S01 | The NS solver shall converge to a steady-state velocity field with residuals below a configurable tolerance. | Velocity field accuracy depends on convergence. Divergent or under-converged solutions produce meaningless transport results. Clarified 2026-09-24, not amended: under the staggered solver's `error_estimate` stopping rule (`src/stopping.py`) the configurable tolerance, `iteration_error_tol`, applies to the estimated iteration error, the velocity step times rho_hat / (1 - rho_hat) over the largest prescribed boundary velocity, with rho_hat the step's fitted geometric rate. A small step is not a small error: the error left is about the step times rho / (1 - rho), and rho approaches 1 under refinement. The velocity-step residual against `convergence_tol` stays the default rule and keeps its meaning. See `docs/reports/stopping_rule_evidence.md`, sections 4 and 9. | VAL-001, VAL-002 |
| REQ-S02 | The NS solver shall reproduce the Poiseuille flow parabolic velocity profile with L2 error < 1% on an 80x40 grid. | Validates basic FV discretization and pressure-velocity coupling against an exact analytical solution. Amended 2026-09-30 (ECR-001 step 9): the criterion returns from 2.5% to the original 1%, as ECR-001 section 6 records for the supersession of ADR-008, now that wall conditions are imposed directly on the staggered components (REQ-S12). The staggered solver measures 4.107e-4 on the uniform 80x40 grid and 3.024e-3 on the same grid clustered to a wall cell of 0.1 H / ny, and converges at order 1.99 under uniform refinement, judged without a reference (ECR-001 criteria 1, 2 and 4; `docs/reports/val001_revalidation_step7.md`, addendum, under stopping rule version 3). The 2.5% and its O(h) rationale belonged to the collocated ghost-cell walls (ADR-008), which measured 2.036e-2 against ADR-008's 2.5%; retired, tag `collocated-final`. | VAL-001 |
| REQ-S03 | The NS solver shall reproduce the lid-driven cavity centerline velocity profiles at Re = 100 with a maximum error below 2% of the lid speed against Marchi, Suero and Araki (2009), reference `marchi_2009_re100`. Ghia et al. (1982), reference `ghia_1982_re100_r2`, is reported beside it and is not scored. | Validates nonlinear advection, 2D pressure gradients, and recirculation handling. Amended 2026-09-30 (ECR-001 step 9), per the amendment of 2026-09-24 under ECR-001 criteria 3 and 3a: scored against Ghia, a correctly converging scheme meets a floor of about 0.005 in u on the vertical centerline near y = 0.85 and 0.009 in v at the jet stations by the right wall, which is Ghia's own error, and a series that must fall under refinement then fails the right answer (`docs/reports/cavity_reference_marchi.md`, section 5). The staggered solver measures u 1.057e-3 and v 7.356e-4 of the lid speed at 80x80 against Marchi, falling at second order from 20x20 (`docs/reports/val002_revalidation_step8.md`). Until 2026-09-22 the Ghia v table in use was not Ghia's (ECR-001 erratum, section 12). | VAL-002 |
| REQ-S04 | The velocity field shall satisfy the incompressibility constraint (divergence-free) to within configurable tolerance at every cell. | Mass conservation is fundamental. FV enforces this by construction, but numerical errors can accumulate. Clarified 2026-09-24, not amended: under the `error_estimate` stopping rule the per-cell tolerance, `mass_imbalance_tol`, is enforced at stopping. A solve is not converged until the worst absolute per-cell mass imbalance of its field is below it, which the velocity-step rule never checked. On an open domain the imbalance sets a drift of the through-flow that the velocity step does not see, and the per-cell bound alone lets that drift grow with the number of cells upstream, more on every finer grid. So the rule also requires the summed absolute imbalance, over rho times the inflow (on a closed domain rho times the velocity scale times the longer side), to be below `iteration_error_tol`. The flux through any cross-section differs from the inflow by at most the summed imbalance on one side of it, so this bounds the drift relative to the through-flow on any grid. Clarified again 2026-09-30: the rule also requires the absolute value of the signed imbalance summed over the domain, the net mass flux out of it, to be below `mass_imbalance_tol`. That is ECR-001 criterion 6's domain-sum clause at the bound the criterion names, beside the per-cell clause above. See `docs/reports/stopping_rule_evidence.md`, sections 4, 5, 9 and 10. Verification corrected 2026-09-30 (ECR-001 step 9): VAL-007 is the transport solver's conservation test (REQ-T05) and does not exist before Phase 3. The staggered VAL-001 and VAL-002 tests assert the stop `error_estimate_and_continuity`, which this tolerance gates. | Unit test, VAL-001, VAL-002; VAL-007 from Phase 3 |
| REQ-S05 | The solver shall use the SIMPLE algorithm for pressure-velocity coupling. | Industry-standard approach. Well-documented, stable, compatible with structured grids. | Architecture review |
| REQ-S06 | The pressure correction inner loop shall have a pure NumPy reference implementation for validation and a CUDA C++ accelerated implementation for production runs, called from Python via pybind11. The NumPy reference shall remain in the codebase permanently as the ground truth for equivalence testing (see REQ-N03). | NumPy reference validates physics independently of GPU code. CUDA acceleration is a separate deliverable after solver physics are validated. | Integration test |
| REQ-S07 | The solver shall use a staggered (MAC) variable arrangement with pressure at cell centers, u-velocity at east-west cell faces, and v-velocity at north-south cell faces. | Staggered arrangement provides natural pressure-velocity coupling without Rhie-Chow interpolation artifacts. Eliminates checkerboard modes by construction and enables exact discrete continuity enforcement. Required by ECR-001, whose motivating VAL-002 v-velocity error was later found to be measured against a corrupted Ghia table (ECR-001 erratum, 2026-09-22); the wall mass leak it also cites stands. | Architecture review, VAL-001, VAL-002 |
| REQ-S08 | The pressure correction equation shall be solved using Jacobi iteration. | Jacobi iteration updates all cells independently from previous-iteration neighbors, enabling full vectorization in NumPy and direct parallelization in CUDA (one thread per cell). Converges slower per iteration than Gauss-Seidel but each iteration is a single array operation. Clarified 2026-09-22, not amended: weighted Jacobi, `p_new = (1 - w) p_old + w (Jacobi update)` with w = 2/3, satisfies this requirement, because every cell still reads only previous-iteration neighbors and a scalar weight does not change that. On a closed domain the weight is necessary: every row has a_P equal to its neighbor sum and the grid is bipartite, so the checkerboard is an exact eigenvector of the plain update with eigenvalue -1 and never decays. The weight maps it to 1 - 2w = -1/3. See `docs/reports/pressure_correction_step5.md`, sections 3 and 5. | Unit test |
| REQ-S09 | The advection term shall be discretized using the QUICK scheme (Leonard 1979) with specialized stencils at boundary-adjacent cells. | QUICK provides second-order accuracy globally on smooth flows, appropriate for the moderate-Peclet regime of cleanroom flows. Does not require first-order upwind fallback of hybrid schemes. Required by ECR-001. | VAL-001, VAL-002 |
| REQ-S10 | Under-relaxation factors for velocity (default 0.7) and pressure (default 0.3) shall be configurable via the YAML configuration. | SIMPLE requires under-relaxation for stability. Factors control convergence rate vs. stability tradeoff. Configurable per REQ-C01. | Unit test |
| REQ-S11 | The mesh shall support independent geometric stretching in x and y directions, clustered toward both walls of an axis and mirrored about its midpoint, specified per axis by either the wall-adjacent cell width or the geometric expansion ratio, the other derived from the cell count. | Enables resolution clustering near walls without uniform refinement of the entire domain. Required by ECR-001. Amended 2026-09-30 (ECR-001 step 9) to what step 1 built. The earlier text asked for the spacing and the ratio per wall, but at a fixed cell count a symmetric geometric distribution has one free parameter (ECR-001 criterion 2 note), and the two walls of an axis share it. Clustering toward one wall of an axis only, or toward an interior region, is not built. | Unit test |
| REQ-S12 | Dirichlet velocity boundary conditions shall be imposed directly on the staggered velocity components at the physical wall location, without ghost cell interpolation. | Eliminates the O(h) wall accuracy limitation previously documented in ADR-008. Required by ECR-001. | Unit test, VAL-001 |
| REQ-S12.1 | The interpretation of configured boundary segments (which segment covers a point on a domain edge, its type, and the velocity it prescribes there) shall be implemented once and shared by every boundary imposition layer. | Derived from REQ-S12, for modularity rather than physics: the staggered velocity layer reads one configuration interpretation today, and the Phase 3 concentration layer (`src/boundary_concentration.py`, decided 2026-10-02) will be its second reader; a second copy could drift between them. | Unit test |
| REQ-S13 | The NS solver shall expose the face velocities of its last solve on their staggered storage locations: u on vertical faces [ny, nx+1], v on horizontal faces [ny+1, nx], float64, contiguous, read-only copies, whose two-face averages are the returned cell means and whose per-cell mass imbalance is `last_mass_imbalance`; when the solve stopped by `error_estimate`, that imbalance satisfies conditions (b) and (d) of REQ-S04. | Proposed by ADR-011 (PR 30), section A, status Proposed. Continuity is enforced on the faces and the returned cell means are their averages, which do not carry it (ADR-010, Consequences); the transport solver advects with the face fluxes, the constancy test (REQ-T11) predicts its drift from their imbalance, and reconstructing faces from the means would give an O(h^2) imbalance the stopping rule never bounded. Adding the attribute changes no existing signature. | Unit test (the three promises above, in tests/test_solver_staggered.py); tests/test_constancy.py |

### 2.2 Transport Requirements

| ID | Requirement | Rationale | Verified By |
|----|-------------|-----------|-------------|
| REQ-T01 | The transport solver shall solve the advection-diffusion equation for particle concentration on the velocity field produced by the NS solver. | Core physics coupling. Particles are carried by airflow and spread by diffusion. | VAL-003, VAL-004 |
| REQ-T02 | The transport solver shall support five discrete particle size classes: 0.1, 0.3, 0.5, 1.0, and 5.0 um. | Spans the relevant physics regimes from diffusion-dominated to settling-dominated and maps to ISO 14644 classification sizes. | Unit test |
| REQ-T03 | Each size class shall have independent settling velocity computed via Stokes drag with Cunningham slip correction. | Settling velocity varies by 1000x across the size range. Cunningham correction is significant below 1 um. | VAL-005 |
| REQ-T04 | Each size class shall have independent Brownian diffusion coefficient computed via Stokes-Einstein relation with Cunningham correction. | Diffusion dominates transport for sub-micron particles. | VAL-006 |
| REQ-T05 | The transport solver shall conserve total particle mass to within 0.01% across all timesteps (mass in domain + mass out = mass in + mass from sources). | FV formulation guarantees conservation by construction. This test catches numerical bugs. | VAL-007 |
| REQ-T06 | The transport solver shall accept an external force field parameter (v_ext) for future extension to electrostatic precipitation modeling. The parameter shall be structurally present but set to zero in v1. | Architectural extensibility per ADR-007. Avoids future refactoring of the solver interface. | Architecture review |
| REQ-T07 | Pure diffusion from a point source shall produce a Gaussian concentration profile with L2 error < 1% vs the analytical solution. | Validates the diffusion discretization independently of advection. | VAL-003 |
| REQ-T08 | Advection of a concentration pulse in a uniform flow shall preserve peak location to within 1 cell width and maintain pulse shape. | Validates advection discretization. Excessive numerical diffusion indicates the scheme is too dissipative. | VAL-004 |
| REQ-T09 | The particles module shall compute gravitational and diffusional deposition velocity for each size class, parameterized by boundary layer thickness and surface orientation (floor, ceiling, wall). | Deposition velocity is a boundary condition input for the transport solver. Floor deposition includes gravitational settling; ceiling and wall deposition are diffusion-only. | VAL-010 |
| REQ-T10 | The particles module shall estimate HEPA filter collection efficiency for each size class via interpolation of reference efficiency data. | HEPA efficiency determines the particle removal rate at supply vent boundaries. Efficiency varies by particle size with a minimum at the most-penetrating particle size (~0.3 um). | VAL-011 |
| REQ-T11 | A spatially uniform concentration field advected by a face velocity field that meets REQ-S04 at its stop, with diffusion, settling, deposition and sources off, shall remain uniform: after a simulated time T the largest relative departure of any FLUID cell from its initial value shall not exceed b_max T / (rho V_min), where b_max is the worst per-cell mass imbalance of that field (condition (b)) and V_min the smallest FLUID cell volume; and for the product configuration the bound evaluated at `mass_imbalance_tol` over the scenario's `t_end` shall be below REQ-T05's 0.01%. | Proposed by ADR-011 (PR 30), section G, status Proposed. A per-cell velocity imbalance is a source or sink of particle mass (ADR-010, Consequences, For Phase 3), and a conservative face-flux scheme that is exact on a uniform field turns the stopping rule's bound into a drift rate, b_P / (rho V_P) per second in cell P. VAL-007's budget closes by telescoping on any face field and cannot see this; the two instruments are kept apart. Measured on the saved VAL-001 80x40 histories at the stop: 7.05e-9 per second in the worst cell, 0.0025% per hour (results/builder30/constancy_drift.json; the derivation is ADR-011 G). The second clause ties `mass_imbalance_tol` to the scenario duration: `mass_imbalance_tol <= 1e-4 rho V_min / t_end`. | tests/test_constancy.py |

### 2.3 Configuration Requirements

| ID | Requirement | Rationale | Verified By |
|----|-------------|-----------|-------------|
| REQ-C01 | All simulation parameters shall be defined in a single YAML configuration file. No simulation parameter shall be hardcoded in source modules. | Lesson from Stock Transformer: scattered parameters cause synchronization failures. | Architecture review |
| REQ-C02 | The configuration loader shall validate all parameters at load time and raise clear errors for missing keys, out-of-range values, and type mismatches. | Fail fast. Do not let invalid config propagate to a solver crash 10 minutes into a run. | Unit test |
| REQ-C03 | Scenario configuration files shall inherit from a base configuration and override specific parameters only. | Avoids duplicating the full config for each scenario. Changes to base config automatically propagate. | Unit test |
| REQ-C04 | Physical constants (Boltzmann constant, gravity, air properties at standard conditions) shall be defined in exactly one location. | No duplication of constants across modules. | Architecture review |

### 2.4 Alert System Requirements

| ID | Requirement | Rationale | Verified By |
|----|-------------|-----------|-------------|
| REQ-A01 | The alert monitor shall detect threshold exceedance within one timestep of occurrence. | Detection latency of zero or one timestep. Delayed detection defeats the purpose of monitoring. | VAL-009 |
| REQ-A02 | Sensor locations shall be configurable via the YAML configuration. | Enables comparison of sensor placement strategies without code changes. | Unit test |
| REQ-A03 | Contamination thresholds shall be configurable per particle size class, following ISO 14644-1 classification limits. | Different size classes have different regulatory limits. Class 5 allows zero 5.0 um particles but 3.52 million 0.5 um particles per cubic meter. | Unit test |
| REQ-A04 | The monitor shall report detection latency per sensor per scenario: elapsed time from event onset to first threshold exceedance at each sensor. | Core metric for reactive monitoring. Answers "how fast do we know about a contamination event?" | Integration test |
| REQ-A05 | The monitor shall support evaluation of multiple sensor configurations across multiple scenarios to identify optimal placement. | Core value of proactive design mode. Answers "where should sensors go?" | Integration test |
| REQ-A06 | The alert monitor shall read concentration fields but never modify them. | Separation of concerns. Monitoring is observation, not intervention. | Architecture review |

### 2.5 Visualization Requirements

| ID | Requirement | Rationale | Verified By |
|----|-------------|-----------|-------------|
| REQ-V01 | The system shall produce animated visualizations of contamination evolution over time for each scenario. | Primary portfolio deliverable. Visual demonstration of physics and system behavior. | Manual review |
| REQ-V02 | Visualizations shall support per-size-class display showing how the same event produces different spatial patterns for different particle sizes. | Demonstrates the physics of multi-class transport, not just pretty pictures. | Manual review |
| REQ-V03 | Visualizations shall overlay alert sensor locations with status indicators that change state when thresholds are exceeded. | Connects the physics simulation to the monitoring system visually. | Manual review |

### 2.6 Numerical Requirements

| ID | Requirement | Rationale | Verified By |
|----|-------------|-----------|-------------|
| REQ-N01 | The simulation shall enforce the CFL condition for numerical stability at every timestep. | Explicit advection schemes are conditionally stable. Violating CFL produces divergent solutions. | Unit test |
| REQ-N02 | The solution shall demonstrate grid convergence at the expected order of accuracy when the grid is refined. | Confirms numerical accuracy. Second-order scheme should reduce error by 4x when grid spacing is halved. | VAL-008 |
| REQ-N03 | The CUDA C++ solver inner loop shall produce results identical to the pure NumPy reference implementation for all validation cases. Equivalence is defined as maximum absolute element-wise difference below 1e-10 for all output arrays. | Verifies the Python-CUDA interface is not introducing bugs through memory layout, dtype, or indexing errors. | Integration test |

---

## 3. Module Dependency Map

This section defines which modules depend on which. When a PR modifies a module, the code reviewer checks that downstream modules are still compatible.

### 3.1 Dependency Graph

Generated from the import statements in `src/` by `scripts/gen_system_map.py`. Regenerate with `python scripts/gen_system_map.py`; CI runs it with `--check` and fails a PR that changes `src/` without regenerating.

<!-- BEGIN GENERATED: dsm -->
```
                     boundary_registry  boundary_staggered  config  constants  mesh  momentum  particles  pressure  solver_staggered  staggered  stopping
boundary_registry            .                  .             X         .       .       .          .         .             .              .         .
boundary_staggered           X                  .             X         .       X       .          .         .             .              X         .
config                       .                  .             .         .       .       .          .         .             .              .         .
constants                    .                  .             .         .       .       .          .         .             .              .         .
mesh                         .                  .             X         .       .       .          .         .             .              .         .
momentum                     .                  X             X         .       X       .          .         .             .              X         .
particles                    .                  .             X         X       .       .          .         .             .              .         .
pressure                     .                  X             X         .       X       X          .         .             .              X         .
solver_staggered             .                  X             X         .       X       X          .         X             .              X         X
staggered                    .                  .             .         .       X       .          .         .             .              .         .
stopping                     .                  .             .         .       .       .          .         .             .              .         .
```

Rows import columns. Edges, 25 total:

```
boundary_registry  -> config
boundary_staggered -> boundary_registry, config, mesh, staggered
mesh               -> config
momentum           -> boundary_staggered, config, mesh, staggered
particles          -> config, constants
pressure           -> boundary_staggered, config, mesh, momentum, staggered
solver_staggered   -> boundary_staggered, config, mesh, momentum, pressure, staggered, stopping
staggered          -> mesh
```

Cycles of any length: **none**. Checked by depth-first search over the whole graph, not by looking for mutual pairs. A three-module cycle is the one that actually happens and a pair check answers 'none' in its presence.

This matrix is generated from the import statements in `src/` and describes the modules that exist today. Section 3.2 (cascade rules) is hand-authored and stays that way: the generator owns what the structure is, and the cascade rules are a human judgment about what follows from it.
<!-- END GENERATED: dsm -->

### 3.2 Cascade Rules

When a PR modifies a module, the reviewer verifies impact on downstream modules. Read this table as: "If you change X, check Y."

| Modified Module | Check These Downstream Modules | What to Check |
|-----------------|-------------------------------|---------------|
| config.py | boundary_registry, boundary_staggered, mesh, momentum, particles, pressure, solver_staggered; solver_transport, monitor, scenarios, time_integration (planned) | New/changed/removed config fields are handled by all consumers. No module accesses a field that no longer exists. No module ignores a new field it should use. |
| constants.py | particles | Constant names and SI values unchanged; no module defines its own copy (REQ-C04). |
| mesh.py | boundary_staggered, momentum, pressure, solver_staggered, staggered; solver_transport, monitor (planned) | Grid dimensions, cell arrays, and coordinate arrays are consumed correctly. Shape assumptions still hold. Centers stay face midpoints; staggered averaging depends on it. |
| staggered.py | solver_staggered, boundary_staggered, momentum, pressure | Face array shapes and the face-to-center averaging contract unchanged. |
| boundary_registry.py | boundary_staggered; boundary_concentration (planned, Phase 3) | Coverage rule (same edge, inclusive range, first match in configuration order, wall by default) and the prescribed-velocity decomposition unchanged. Every imposition layer reads them, so a change here moves each. |
| boundary_staggered.py | momentum, pressure, solver_staggered | Normal imposition writes domain faces only. Tangential data shape [n+1], outlet data shape [n], wall_distance semantics and the inward flux sign unchanged. |
| boundary_concentration.py (planned, Phase 3, ADR-011 E) | solver_transport | ConcentrationFaces array names and shapes (face-shaped, read-only), the surface codes the budget books deposition to, the settling mask, and the derivation of each condition from the registry's velocity type with the segment keys concentration, hepa_filtered and deposition_surface unchanged. |
| momentum.py | pressure, solver_staggered | MomentumPrediction shapes and the meaning of a_p_u and a_p_v (un-relaxed diagonal, positive exactly at the unknown faces) unchanged; boundary entries of u and v read as given and never written. |
| pressure.py | solver_staggered | PressureCorrection shapes, the right-hand side formed directly from face velocities with no compatibility correction, outlet faces corrected against p' = 0 with the nearest interior diagonal, closed-domain pin at the first FLUID cell, and the sweep count reported. |
| solver_staggered.py | solver_transport, time_integration (planned, Phases 3 and 4: the NS solver of section 2.1); scripts/benchmark.py, scripts/view_field.py, scripts/stopping_probe.py, scripts/val001_order.py, scripts/self_convergence.py | The public shape: cell-centered [ny, nx] float64 contiguous returns, the IterationState callback once per outer iteration with cell-centered fields and the corrector's sweep count, last_pressure_sweeps and stage_seconds reset per solve, reference_velocity F_ref / (rho h). Under the default velocity_step rule the stop is the residual below convergence_tol, the definition every stored velocity-step row was taken under, so outer iteration counts compare with them; residual_history keeps that definition under both rules. converged and stop_reason are set by every solve and reset at its start; the harness records them, with velocity_step_below_tol stored as residual_below_tol, the label every stored velocity-step row carries. |
| stopping.py | solver_staggered, scripts/stopping_probe.py, scripts/val001_order.py, scripts/benchmark.py, scripts/self_convergence.py | IterationState's fields, which the solver hands its on_iteration callback and the harness reads, unchanged. update(step, imbalance) answers converged only when the estimate, the worst imbalance, the absolute sum over flux_scale and the absolute signed sum are all below their tolerances; the imbalance callable, which returns an ImbalanceSummary read by name from one evaluation, is not called until the estimate is met; no estimate (inf) while the window is short, a step in it is zero or not finite, or rho_hat is outside (0, 1). RATE_WINDOW stays a module constant. A change of condition raises RULE_VERSION, which stopping_probe and val001_order store with their saved solves, so that they solve again, and the harness records in every error_estimate row, so that its summary keeps the versions apart. |
| solver_transport.py (planned, Phase 3, ADR-011 C and F) | time_integration, monitor (planned, Phases 4 and 5); tests/test_constancy.py | solve_timestep takes FaceVelocities, not cell means, and returns a new [ny, nx] float64 contiguous field with SOLID cells zero; stable_dt's definition; v_ext a FaceVelocities-shaped per-class drift, None read as zero (REQ-T06); MassBudget field names and that only the solver writes them. |
| particles.py | solver_transport, boundary_concentration (planned, Phase 3) | Settling velocity, diffusion coefficient and deposition velocity interface unchanged. Return types, units and signs unchanged (settling positive downward, deposition non-negative; the floor value includes settling, which the solver must not add again, ADR-011 D). |
| scenarios.py | time_integration; the boundary layers (planned, Phase 4) | Source term and BC modification interfaces unchanged. Event timing semantics unchanged. |
| monitor.py | time_integration | Update and query interfaces unchanged. Alert output format unchanged. |
| csolver/ | solver_staggered (REQ-N03's NumPy reference, ADR-010), solver_transport; both planned, Phase 6 | CUDA kernel signatures match pybind11 declarations. Memory layout assumptions (row-major, contiguous, double precision) unchanged. |
| time_integration.py | io_manager | Timestep sequence and output trigger logic unchanged. |

### 3.3 Cross-Cutting Concerns

These concerns span multiple modules. Changes to any of them require checking all modules that participate.

**Array conventions:** All 2D field arrays are shape `[ny, nx]`, row-major, contiguous, dtype `float64`. Every module that creates, passes, or receives a field array must follow this convention. A change to array layout or dtype cascades to every module.

**Unit system:** SI throughout. Meters, seconds, kilograms, Pascals. No CGS, no imperial, no implicit unit conversions. All values in config are SI.

**Coordinate system:** Origin at bottom-left of the domain. x increases left-to-right, y increases bottom-to-top. Gravity acts in the -y direction. Consistent across mesh, boundary, solver, and visualization.

### 3.4 Components

Generated. The responsibility and serves columns are editorial and come from `docs/system_map_annotations.toml`; the generator refuses to run when a module has no entry.

<!-- BEGIN GENERATED: components -->
| Module | Lines | Responsibility | Declares it serves |
|---|---|---|---|
| `src/boundary_registry.py` | 195 | Interprets the configured boundary segments once, answering which condition and prescribed velocity hold at a point on a domain edge, for every boundary imposition layer: the staggered velocity layer today, the Phase 3 concentration layer next. | S12.1 |
| `src/boundary_staggered.py` | 423 | Writes Dirichlet normal velocities exactly into the staggered domain-face entries and exposes the tangential wall values, wall distances and pressure outlets as data for the momentum and pressure steps. | S12 |
| `src/config.py` | 721 | Loads the YAML configuration into typed dataclasses and rejects missing keys, wrong types and out-of-range values at load time. | A02, A03, C01, C02, S10 |
| `src/constants.py` | 8 | Holds the physical constants shared by every module so that none of them defines its own copy. | C04 |
| `src/mesh.py` | 411 | Builds the structured grid, uniform or geometrically clustered at the walls, with the face, center, width and center-to-center arrays a face-based stencil needs, and classifies each cell as FLUID, SOLID or BOUNDARY. | S11 |
| `src/momentum.py` | 522 | Predicts u* and v* on the staggered grid with QUICK advection by deferred correction over an upwind implicit matrix, one under-relaxed Jacobi sweep per call, and returns the diagonal coefficients the pressure correction needs. | S07, S09 |
| `src/particles.py` | 255 | Computes per-size-class transport properties: Cunningham correction, settling velocity, Brownian diffusion, deposition velocity and HEPA efficiency. | T03, T04, T09, T10 |
| `src/pressure.py` | 441 | Assembles the staggered pressure correction equation from the momentum diagonals with the discrete divergence of u* as its right-hand side, solves it by weighted Jacobi iteration, corrects the face velocities and updates the pressure. | S04, S08 |
| `src/solver_staggered.py` | 292 | Runs steady SIMPLE on the staggered grid as one outer loop over the momentum predictor and the pressure correction, returning cell-centered fields through the harness's callback shape; stops by the velocity-step rule or, when configured, by the error-estimate rule. | S01, S02, S03, S04, S05, S07 |
| `src/staggered.py` | 152 | Defines the staggered (MAC) field layout: shapes and allocation of face-centered u and v and cell-centered p, and the face-to-center averaging the solver applies before returning. | S07 |
| `src/stopping.py` | 218 | Decides when the steady outer iteration has converged, on four conditions: (a) the iteration error estimated from the step and its fitted geometric rate, over a physical velocity scale; (b) the worst per-cell mass imbalance against its own tolerance; (c) the summed imbalance over the through-flow, which shares the tolerance of (a); and (d) the signed imbalance summed over the domain, which shares the tolerance of (b). Also defines IterationState, the snapshot a solver hands its callback once per outer iteration. | S01, S04 |

Total 12 Python files, 3638 lines. 1 empty `__init__.py` carry no row: a package marker with no code has no responsibility to record.

`Declares it serves` is an EDITORIAL CLAIM read from `docs/system_map_annotations.toml`. It says which requirements a module is meant to satisfy, not that it does. Whether a requirement is met is answered by the tests named in the register's `Verified By` column.
<!-- END GENERATED: components -->

### 3.5 Runtime Edges

Generated. Static import analysis cannot see a function bound into a registry by a decorator, so this table lists every non-inert decorator application in `src/`.

<!-- BEGIN GENERATED: runtime-edges -->
| Registry | Bound in | Applications | Functions |
|---|---|---|---|

**No runtime-bound edges.** Every decorator in `src/` is one of the inert set (`dataclass`, `staticmethod`, `property` and their kin), which bind nothing into a dispatch registry. The dependency matrix above is therefore the complete dispatch story for this system: nothing is bound at runtime that the import graph cannot see. This table exists so that the day a registry appears, it shows up here as a row rather than as silence.

**Source-derived. Nothing here is evidence about a running process.** These are decorator applications counted in the source text. `Applications` exceeds `Functions` where one function carries several non-inert decorators.
<!-- END GENERATED: runtime-edges -->

### 3.6 Source Fingerprint

<!-- BEGIN GENERATED: source-fingerprint -->
| Property | Value |
|---|---|
| Scope | `src/**/*.py` |
| Files hashed | 12 |
| Digest | `sha256:96c415bb19f0a4bbc92e37897e95ca4a59ee8d4afea8fe863b1105d0969cdbe9` |

This is what lets the document answer whether it is current, which is the one question a stale table cannot be asked. `python scripts/gen_system_map.py --check` recomputes the whole set of generated regions, this digest included, and exits non-zero on any disagreement.

A disagreement means the source tree moved and this document did not. It does not mean any hand-authored section is wrong, and in particular it says nothing about section 2, which no generator touches.
<!-- END GENERATED: source-fingerprint -->

---

## 4. Interface Contracts (Summary)

Detailed interface contracts are in the development plan. This section provides a quick reference for the reviewer.

### config.py --> all modules

```
SimConfig:
    SimConfig(yaml_path)            # load and validate a file
    SimConfig.from_dict(raw: dict)  # same validation on a parsed mapping
    room_width, room_height: float (meters)
    nx, ny: int
    stretch_x, stretch_y: StretchSpec (ratio >= 1, or min_spacing; one axis each)
    rho, mu: float (SI)
    temperature: float (K)
    particle_sizes: list[float] (meters)
    particle_density: float (kg/m^3)
    mean_free_path: float (meters)
    boundary_layer_thickness: float (meters)
    hepa_reference: HepaReference (diameters + efficiencies)
    dt, t_end: float (seconds)
    output_interval: int
    convergence_tol: float
    max_simple_iter: int
    alpha_velocity: float (0, 1]
    alpha_pressure: float (0, 1]
    max_pressure_iter: int
    pressure_tol: float
    stopping_rule: str  # optional: "velocity_step" (default) or "error_estimate"
    iteration_error_tol: float  # optional, default 1e-6; error_estimate only
    mass_imbalance_tol: float  # optional, default 1e-10; error_estimate only
    # any other key in the solver block raises ValueError at load
    boundaries: dict[str, BoundarySpec]
    obstacles: list[ObstacleSpec]
    # scenarios: deferred to Phase 4, loaded via separate scenario YAML files
    sensors: list[SensorSpec]
    thresholds: dict[str, float]
```

```
BoundarySpec:
    type: str  # "wall", "velocity_inlet", "pressure_outlet"
    location: str  # "top", "bottom", "left", "right"
    x_start, x_end: float | None  # for top/bottom segments
    y_start, y_end: float | None  # for left/right segments
    velocity: float | None  # magnitude, decomposed normal to the boundary
    u_velocity: float | None  # explicit x-component at the face
    v_velocity: float | None  # explicit y-component at the face
```

`u_velocity` and `v_velocity` are optional and only meaningful for
`type == "velocity_inlet"`. When either is present, both components are
read directly from the spec (missing component defaults to 0) and the
`velocity` magnitude field is ignored for decomposition. This supports
tangential inflow such as the moving lid in a lid-driven cavity. When
both are None, the `velocity` magnitude is decomposed normal to the
edge (positive inward).

### mesh.py --> staggered, boundary_staggered, momentum, pressure, solver_staggered (solver_transport, monitor planned)

```
Mesh:
    x, y: ndarray (face coordinates, x[0] = 0 and x[nx] = width exactly)
    xc, yc: ndarray (cell centers, midpoints of their two faces)
    dx, dy: float (uniform spacing L / n; mean spacing on a stretched mesh)
    dx_cell, dy_cell: ndarray (cell widths, shapes [nx] and [ny])
    dx_face, dy_face: ndarray (center-to-center distance at each face, shapes
        [nx+1] and [ny+1]; wall-to-first-center at the two boundary faces)
    stretch_ratio_x, stretch_ratio_y: float (effective ratio, 1.0 uniform)
    min_spacing_x, min_spacing_y: float (effective wall-adjacent width)
    is_uniform: bool
    cell_type: ndarray[ny, nx] (FLUID=0, SOLID=1, BOUNDARY=2)
    is_fluid(i: int, j: int) -> bool
    get_neighbors(i: int, j: int) -> list[tuple[int, int]]
```

### staggered.py --> boundary_staggered, momentum, pressure, solver_staggered

Internal to the solver (REQ-S07). Layout: u on vertical faces [ny, nx+1],
v on horizontal faces [ny+1, nx], p at cell centers [ny, nx].

```
u_shape(mesh), v_shape(mesh), p_shape(mesh) -> tuple[int, int]
allocate_fields(mesh) -> (u, v, p) zeroed, float64, contiguous
u_face_coordinates(mesh), v_face_coordinates(mesh),
cell_center_coordinates(mesh) -> (X, Y) 2D coordinate arrays
to_cell_centers(u, v) -> (u_c, v_c) each [ny, nx], plain two-face average
FaceVelocities: u [ny, nx+1], v [ny+1, nx]   # frozen; planned, ADR-011 A (REQ-S13)
```

### boundary_registry.py --> boundary_staggered (boundary_concentration planned, Phase 3)

Configuration interpretation implemented once for every boundary imposition
layer (REQ-S12.1): the staggered velocity layer today, the Phase 3
concentration layer next.
Reads the config only; no mesh and no field.

```
EDGES = ("bottom", "top", "left", "right")
EdgeCondition: bc_type, u_prescribed, v_prescribed   # frozen
covers(spec, edge, coordinate) -> bool       # same edge, inclusive range
condition_of(spec, edge) -> EdgeCondition    # magnitude decomposed inward, or explicit u/v
BoundaryRegistry:
    __init__(config: SimConfig)
    boundaries -> dict[str, BoundarySpec]
    spec(name) -> BoundarySpec               # KeyError if absent
    condition_at(edge, coordinate) -> EdgeCondition
        first covering segment in configuration order; no-slip wall when none
```

### boundary_staggered.py --> momentum, pressure, solver_staggered

Direct imposition on the staggered layout (REQ-S12). Two deliverables kept
apart: an imposer for the normal components, which are storage locations on
the domain faces, and data for the tangential and pressure conditions, which
have no storage location on the wall. Nothing is written outside the domain
and nothing is written to p.

```
StaggeredBoundary:
    __init__(mesh: Mesh, config: SimConfig)
    apply_normal_velocity(u, v) -> None
        writes u[:, 0], u[:, nx], v[0, :], v[ny, :] where the condition is
        Dirichlet (wall 0, inlet prescribed), exactly; outlet faces and every
        other entry untouched; ValueError on non-staggered shapes
    tangential_conditions() -> dict[edge, TangentialCondition]
        component ("u" on bottom/top, "v" on left/right), is_dirichlet [n+1],
        value [n+1], wall_distance = dy_face[0], dy_face[ny], dx_face[0] or
        dx_face[nx]; arrays read-only
    pressure_outlets() -> dict[edge, PressureOutletCondition]
        is_outlet [n], pressure (gauge datum, 0.0); arrays read-only
    has_pressure_outlet() -> bool
    get_inlet_flux(name) -> float            # sum of inward normal velocity times face width
    get_total_inlet_flux() -> float
    get_max_boundary_velocity() -> float
    A SOLID cell on an edge is a wall on every query.
```

### momentum.py --> pressure, solver_staggered

Momentum predictor on the staggered layout (REQ-S07, REQ-S09): first-order
upwind implicit matrix with the QUICK minus upwind advective flux carried as
an explicit deferred-correction source, one under-relaxed Jacobi sweep per
call with the collocated convention (diagonal divided by alpha_velocity,
``(1 - alpha) / alpha * a_P * phi`` added to the source).

```
quick_face_values(phi, nodes, left, faces, positive) -> face values
    quadratic through C, D and the next node upstream, Lagrange weights
    from the node positions; Leonard boundary form at the ends
MomentumPredictor:
    __init__(mesh: Mesh, config: SimConfig, boundary: StaggeredBoundary)
    momentum_coefficients(u, v) -> (MomentumCoefficients, MomentumCoefficients)
        a_p, a_s_plus, a_s_minus, a_t_plus, a_t_minus, b_boundary, b_deferred,
        each in the component's own shape; s is the component's axis
    predict(u, v, p) -> MomentumPrediction
        u_star [ny, nx+1], v_star [ny+1, nx]: boundary columns and rows are
            the input values, faces of SOLID cells are zero
        a_p_u [ny, nx+1], a_p_v [ny+1, nx]: un-relaxed diagonal, positive
            exactly at the unknown faces and zero elsewhere, so a_p > 0 is
            the mask of correctable faces and d = A_face / a_p is defined
            precisely there (the step 5 contract)
    Reads tangential_conditions for the wall values and wall distances;
    boundary entries of u and v (Dirichlet from StaggeredBoundary, outlet
    extrapolation by the caller) are read as given and never written.
```

### pressure.py --> solver_staggered

Pressure correction on the staggered layout (REQ-S04, REQ-S08). The
right-hand side is the discrete divergence of u* from the stored face
velocities, with no interpolation and no compatibility correction; walls
contribute no coefficient (homogeneous Neumann by absence); a pressure
outlet is ``p' = 0`` at its face; a closed domain is pinned at the first
FLUID cell after the solve and after the pressure update, as the
collocated solver does. The solve is weighted Jacobi, REQ-S08 as clarified
on 2026-09-22, with the weight a module constant rather than a
configuration key.

```
JACOBI_WEIGHT = 2/3                      # module constant
PressureCorrector:
    __init__(mesh: Mesh, config: SimConfig, boundary: StaggeredBoundary)
    needs_pin: bool, pin_cell: (j, i)
    mass_imbalance(u, v) -> [ny, nx]     # rho [(u_e - u_w) dy + (v_n - v_s) dx], 0 at SOLID
    coefficients(a_p_u, a_p_v) -> PressureCoefficients
        a_p, a_e, a_w, a_n, a_s [ny, nx]; a_nb = rho d_face A_face with
        d = A_face / a_P where a_P > 0, zero across walls, inlets and SOLID
        faces; an outlet face borrows the nearest interior diagonal and
        sits in a_p with no neighbour
    sweep(p_prime, coefficients, b, weight: float) -> [ny, nx]
        (1 - w) p' + w (sum(a_nb p'_nb) - b) / a_P where a_P > 0, zero
        elsewhere; weight in (0, 1], 1 is plain Jacobi; input not modified
    correct(prediction: MomentumPrediction, p) -> PressureCorrection
        u [ny, nx+1], v [ny+1, nx]: u* - d (p'_(s+) - p'_(s-)) at correctable
            faces and outlet faces; walls, inlets and SOLID faces untouched
        p [ny, nx]: p + alpha_pressure p', pinned in a closed domain
        p_prime [ny, nx], sweeps: int (sweeps with JACOBI_WEIGHT from p' = 0)
```

### solver_staggered.py --> solver_transport, time_integration (planned); scripts/benchmark.py, scripts/view_field.py, scripts/stopping_probe.py, scripts/val001_order.py, scripts/self_convergence.py

Steady SIMPLE on the staggered layout, the solver of record and since
2026-10-02 the only solver (ADR-010; the collocated solver it was built
beside is at tag collocated-final, Retired modules below). One outer
iteration is MomentumPredictor.predict then PressureCorrector.correct,
keeping the corrected fields. The public shape below is what the harness
drives through one callback; its ``--method`` label names this solver and
refuses the retired one by name. It does not define compute_residual() or
solve_timestep(); the second is Phase 4's.

```
StaggeredSolver:
    __init__(mesh: Mesh, config: SimConfig, boundary: StaggeredBoundary)
    solve_steady(on_iteration=None) -> tuple[ndarray, ndarray, ndarray]  # u, v, p
        cell-centered, each [ny, nx], float64, contiguous
    on_iteration: Callable[[IterationState], None] | None
        once per outer iteration with to_cell_centers of the corrected
        faces, p, the residual and the corrector's sweep count
    residual_history: list[float]
    reference_velocity: float   # F_ref / (rho h), h = max(x[nx]/nx, y[ny]/ny)
    last_pressure_sweeps: int   # reset at the start of each solve
    stage_seconds: dict[str, float]  # "momentum", "pressure", "correct"; no flux stage
    last_mass_imbalance: ndarray [ny, nx]   # of the returned faces; observability only
    flux_scale: float | None    # read-only; the error_estimate rule's flux scale,
                                # kg/s per unit depth; None under velocity_step
    converged: bool             # met its stopping rule, not the cap; reset per solve
    stop_reason: str | None     # "velocity_step_below_tol",
                                # "error_estimate_and_continuity" or "max_simple_iter"
    face_velocities: FaceVelocities | None   # planned, ADR-011 A (REQ-S13): None until a
                                # solve completes; read-only copies of the final faces;
                                # to_cell_centers of them is the return, their
                                # mass_imbalance is last_mass_imbalance
    Walls and inlets are written once by apply_normal_velocity and never
    again; outlet faces are extrapolated (zero gradient) before every
    prediction and corrected by the corrector. Residual: largest change of
    cell-centered u, v over FLUID cells divided by reference_velocity, the
    definition every stored velocity-step row was taken under; F_ref uses
    the staggered layer's exact inlet flux.
    Stop under velocity_step (default): residual < convergence_tol. Under
    error_estimate: ErrorEstimateRule fed the largest change in m/s, scaled
    by get_max_boundary_velocity(), with flux_scale rho times
    get_total_inlet_flux() (closed: rho times that velocity times the longer
    side), never reference_velocity; a zero velocity scale raises at
    construction, naming stopping_rule.
```

### stopping.py --> solver_staggered; scripts/benchmark.py, scripts/self_convergence.py, scripts/stopping_probe.py, scripts/val001_order.py

The error_estimate stopping rule, with no solver dependency so it can be
tested on synthetic histories. REQ-S01 as clarified 2026-09-24, REQ-S04 as
clarified 2026-09-24 and 2026-09-30. Also IterationState, the snapshot the
solver hands its on_iteration callback once per outer iteration, moved here
on 2026-10-02 from the retired collocated solver's module (Retired modules
below): the module that defines when an outer iteration ends also defines
what each iteration reports.

```
IterationState:  # frozen, eq=False: identity only, since the fields are arrays
    iteration: int        # zero-based outer iteration index
    residual: float       # scaled velocity-change residual of this iteration
    pressure_sweeps: int  # Jacobi sweeps of this iteration's pressure correction
    u, v, p: ndarray      # the solver's working fields, each [ny, nx], not
                          # copies; a callback reads them and never writes
ImbalanceSummary:  # frozen, keyword-only, read by name; kg/s per unit depth
    worst: float         # largest absolute per-cell imbalance
    absolute_sum: float  # sum of the absolute per-cell imbalances
    signed_sum: float    # sum of the signed ones: net mass flux out of the domain
ErrorEstimateRule:
    __init__(velocity_scale, flux_scale, iteration_error_tol, mass_imbalance_tol)
        each positive and finite and not bool, else ValueError
        flux_scale: kg/s per unit depth
    update(step: float, imbalance: Callable[[], ImbalanceSummary]) -> bool
        step: largest velocity change of one outer iteration, m/s
        imbalance(): the current field's ImbalanceSummary, from one
            evaluation; called only when (a) holds
        (a) step rho_hat / (1 - rho_hat) / velocity_scale < iteration_error_tol,
            rho_hat = exp of the least-squares slope of log(step) over the
            last RATE_WINDOW steps
        (b) worst < mass_imbalance_tol  (ECR-001 criterion 6, per cell)
        (c) absolute_sum / flux_scale < iteration_error_tol  (bounds the flux drift)
        (d) |signed_sum| < mass_imbalance_tol  (criterion 6, domain sum)
        True when all four hold
    estimate_history: list[float]  # (a)'s left side per step; inf when none
RATE_WINDOW = 100  # module constant, not configuration
RULE_VERSION = 3   # which conditions update applies; stored with saved solves
                   # and in the params of every error_estimate harness row
```

### particles.py --> solver_transport

```
ParticlePhysics:
    __init__(config: SimConfig)
    settling_velocity(size_class: int) -> float
    diffusion_coeff(size_class: int) -> float
    cunningham_correction(size_class: int) -> float
    deposition_velocity(size_class: int, surface: str) -> float
    hepa_efficiency(size_class: int) -> float
```

### boundary_concentration.py --> solver_transport (planned, Phase 3, ADR-011 E)

The registry's second reader (REQ-S12.1). Derives the scalar condition at every
face from the velocity type, with optional segment keys: `concentration`
(one value per class, velocity_inlet), `hepa_filtered` (bool, velocity_inlet),
`deposition_surface` (floor, ceiling, wall or none, wall; the edge decides by
default). Reads no concentration field and writes none.

```
ConcentrationBoundary:
    __init__(mesh: Mesh, config: SimConfig, physics: ParticlePhysics, registry: BoundaryRegistry)
    faces_for(size_class) -> ConcentrationFaces   # frozen, read-only arrays
        inflow_u [ny, nx+1], inflow_v [ny+1, nx]: concentration an inward flux carries
        deposition_u, deposition_v: deposition velocity at wall faces, domain and SOLID
        surface_u, surface_v: which surface each depositing face is booked to
        settling_u, settling_v: faces that carry the class's settling increment
```

### solver_transport.py --> time_integration, monitor (planned, Phase 3, ADR-011 B, C, D, F)

Draft, marked planned; OPEN 1 of ADR-011 decides the face scheme. Finite volume
on the cell-centred field, advected by the face velocities of REQ-S13; explicit
advection, implicit diffusion and deposition sink by Jacobi.

```
TransportSolver:
    __init__(mesh, config, physics: ParticlePhysics, boundary: ConcentrationBoundary)
    stable_dt(faces: FaceVelocities, size_class, v_ext=None) -> float
        cfl_number / max over FLUID cells of (max(|u_w|, |u_e|) / dx_cell
        + max(|v_s|, |v_n|) / dy_cell), the vertical faces carrying the class's
        settling increment and v_ext
    solve_timestep(C_k, faces, size_class, dt, v_ext=None) -> ndarray
        C_k [ny, nx] not modified; v_ext FaceVelocities-shaped per-class drift or
        None (zero; REQ-T06, ADR-007); ValueError if dt > stable_dt or shapes differ
        returns the new field, [ny, nx], float64, contiguous, SOLID cells zero;
        updates budget[size_class] from the fluxes applied
    budget: list[MassBudget]    # per class: initial, inflow, outflow, deposited by
                                # surface, source; in_domain(C_k, mesh); residual()
```

### monitor.py --> time_integration

```
AlertMonitor:
    update(C_fields: dict[int, ndarray], t: float) -> None
    get_alerts() -> list[Alert]
    get_detection_latency(event: str) -> dict[str, float]
```

### scenarios.py --> time_integration; the boundary layers (planned, Phase 4)

```
ScenarioManager:
    get_active_sources(t: float) -> list[SourceTerm]
    get_bc_modifications(t: float) -> dict
```

### Retired modules

The collocated SIMPLE solver `src/solver_ns.py` (ADR-008) and its ghost-cell
boundary layer `src/boundary.py`, with `tests/test_solver_ns.py` and
`tests/test_boundary.py`, were retired on 2026-10-02 (PR 29) once ECR-001 had
closed with them as its before-and-after baseline. They are not carried as
plumbing: a reader who needs them checks out the annotated tag
`collocated-final`, placed on 98f8b1f, the last commit that holds them. What
they measured is in ECR-001, ADR-008, ADR-010 (planned against built),
`docs/reports/inlet_flux_comparison.md` and
`docs/reports/pressure_solver_probe.md`, and their 22 `collocated-jacobi` rows
stay in `benchmarks/results.jsonl`, which `scripts/benchmark.py --summary`
still prints; `run_case` refuses the label by name. `IterationState`, which
both solvers handed their callback, is defined in `src/stopping.py`.

---

## 5. Scope Boundaries

### In Scope (v1)

- 2D vertical cross-section domain
- Incompressible laminar Navier-Stokes (SIMPLE algorithm)
- Finite Volume on structured rectangular grid
- Staircase representation of internal obstacles
- Five-class Eulerian particle transport with settling and diffusion
- Cunningham slip correction for sub-micron particles
- Three contamination scenarios: door seal leak, HEPA filter breach, equipment dust release
- Alert monitoring with configurable sensors and ISO 14644 thresholds
- Sensor placement comparison across scenarios
- Animated visualizations per size class
- Hybrid Python/CUDA C++ implementation (NumPy reference solver for validation, CUDA for production)
- Centralized YAML configuration
- Phase-gated development with validation tests before code

### Out of Scope (v1)

- 3D simulation (architecture supports extension, not implemented)
- Turbulence modeling (laminar assumption is physically justified per ADR-004)
- Thermal buoyancy / hot equipment convection
- Ionizer / electrostatic precipitation (extension point preserved per ADR-007)
- Unstructured meshes
- Multi-room or multi-bay simulation
- Real-time sensor hardware integration
- GPU acceleration beyond CUDA C++ pressure/transport kernels (e.g., multi-GPU, tensor core exploitation)
- Particle-particle interactions, coagulation, electrostatic effects
- Thermophoresis

### Scope Change Process

Any change to scope boundaries during development must be:
1. Documented in the PR description with rationale
2. Reflected in this document
3. Reflected in docs/PROJECT_PLAN.md
4. Reviewed for cascade impact using the dependency map above

---

## 6. Architecture Decision Records

ADR-008 and ADR-010 are files in `docs/ADR/`; the others are in the development plan document. Summary reference:

| ADR | Decision | Key Rationale |
|-----|----------|---------------|
| ADR-001 | Finite Volume discretization | Industry standard (Fluent, OpenFOAM). Conservation built into the method. |
| ADR-002 | 2D vertical cross-section | Captures gravity, HVAC flow, stratification. 3D adds complexity without proportional insight. |
| ADR-003 | Structured grid with staggered layout, non-uniform spacing, staircase boundaries | Clean room geometry is rectangular. Staggered arrangement gives natural pressure-velocity coupling. Non-uniform spacing enables efficient near-wall resolution. |
| ADR-004 | Laminar flow assumption | Clean rooms are engineered for laminar flow. Re well below transition. Physically correct, not a shortcut. |
| ADR-005 | Hybrid Python/CUDA C++ | Python orchestration with pure NumPy reference solver for validation, CUDA C++ kernels via pybind11 for accelerated production runs. NumPy solver is the primary implementation through Phase 3 validation. CUDA acceleration is a separate deliverable after solver physics are validated. |
| ADR-006 | Five particle size classes | Spans diffusion-dominated to settling-dominated regimes. Maps to ISO 14644. |
| ADR-007 | Ionizer modeling deferred | Scope risk. Extension point (v_ext) preserved in transport solver interface. |
| ADR-008 | Collocated ghost cell wall treatment (Superseded by ADR-010) | O(h) wall accuracy accepted for simplicity. Industry-standard approach. VAL-001 relaxed from 1% to 2.5%. |
| ADR-010 | Solver Architecture V2: Staggered Grid, Non-Uniform Mesh, QUICK Advection | Staggered MAC grid removes the collocated ghost-cell wall leak; the closed-domain sum of the mass imbalance is exact to rounding, and the stopping rule holds the per-cell imbalance and the signed domain sum below 1e-10 at every validation stop (ECR-001 acceptance criterion 6, as ADR-010's planned-against-built table records it). The VAL-002 v error once cited here was measured against a corrupted Ghia table (ECR-001 erratum). Non-uniform mesh clusters cells toward the walls; on the VAL-001 stencil it costs accuracy at a fixed cell count. QUICK by deferred correction; weighted Jacobi; the error_estimate stopping rule. Records what ECR-001 built, with planned against built. |
| ADR-011 | Transport Solver Architecture: Face-Advected Cell-Centred Concentration (Proposed) | Concentration at cell centres, advected by the staggered face velocities the NS solver exposes (REQ-S13), so a uniform haze drifts only at the rate the stopping rule bounds (REQ-T11). Explicit advection, implicit diffusion and deposition. Settling as a face increment on interior faces, the floor and ceiling taking the deposition velocity whole. One mass budget for every conservation claim. Three items OPEN for Alex: the face scheme and positivity, VAL-004's criterion, the supply concentration. Written before the build; gains planned against built at the Phase 3 gate. |

---

## Document History

| Date | Change | Author |
|------|--------|--------|
| 2026-04-14 | Initial version. Architecture defined pre-development. | Alex Moroz-Smietana |
| 2026-04-15 | Phase 2 architecture updates: collocated grid with Rhie-Chow (REQ-S07), Jacobi pressure solver (REQ-S08), hybrid advection scheme (REQ-S09), configurable under-relaxation (REQ-S10). ADR-005 amended from C/ctypes to CUDA C++/pybind11 with NumPy reference solver. REQ-S06 and REQ-N03 updated accordingly. | Alex Moroz-Smietana |
| 2026-04-16 | ECR-001 approved: solver architecture rebuild. REQ-S07 and REQ-S09 replaced for staggered grid and QUICK advection. REQ-S11 and REQ-S12 added for non-uniform mesh and direct BC imposition. ADR-003 amended, ADR-008 superseded, ADR-010 added. | Alex Moroz-Smietana |
| 2026-09-19 | Section 3.1 dependency graph replaced by a generated dependency matrix; 3.4 components, 3.5 runtime edges and 3.6 source fingerprint added as generated regions (scripts/gen_system_map.py). Section 2 untouched. | Alex Moroz-Smietana |
| 2026-09-20 | ECR-001 steps 1 and 2: mesh contract extended with per-cell widths, center-to-center face distances and per-axis stretching (REQ-S11); staggered.py added with the MAC layout and face-to-center averaging (REQ-S07); SimConfig gains stretch_x and stretch_y from an optional mesh section. Solver logic unchanged. | Alex Moroz-Smietana |
| 2026-09-21 | ECR-001 step 3: boundary_registry.py extracted from boundary.py as the configuration interpretation both boundary layers share (REQ-S12.1, derived from REQ-S12); boundary_staggered.py added with exact normal-component imposition and the tangential and pressure conditions exposed as data (REQ-S12). Collocated interface and output unchanged. Cascade rules and contracts added for the three boundary modules. | Alex Moroz-Smietana |
| 2026-09-22 | ECR-001 step 4: momentum.py added, the staggered momentum predictor with QUICK advection by deferred correction over an upwind implicit matrix (REQ-S07, REQ-S09). Its MomentumPrediction return is the coefficient contract for the step 5 pressure correction. Not integrated into solve_steady; collocated solver and harness rows unchanged. | Alex Moroz-Smietana |
| 2026-09-22 | ECR-001 step 5: pressure.py added, the staggered pressure correction (REQ-S04, REQ-S08 as written). The closed-domain right-hand side sums to zero to rounding, measured directly (acceptance criterion 6). Undamped Jacobi found not to converge on the closed system (exact -1 eigenvalue); recorded in docs/reports/pressure_correction_step5.md, REQ-S08 not amended. Not integrated into solve_steady; collocated solver and harness rows unchanged. | Alex Moroz-Smietana |
| 2026-09-22 | REQ-S08 clarified, not amended: weighted Jacobi with w = 2/3 satisfies it, since each cell still reads only previous-iteration neighbors. Rationale recorded in the requirement: the plain update has an exact -1 eigenvalue on the closed-domain system, which the weight maps to -1/3. pressure.py gains the JACOBI_WEIGHT constant and a public sweep(); correct() uses the weighted sweep, and the closed-cavity correction now converges. Not integrated into solve_steady; collocated solver and harness rows unchanged. | Alex Moroz-Smietana |
| 2026-09-22 | ECR-001 step 6: solver_staggered.py added, the staggered SIMPLE loop over momentum.py and pressure.py, alongside the collocated solver, which is unchanged. Contract added; cascade rows and section 4 headings for the staggered modules now name solver_staggered as their consumer; the collocated retirement moves to a later step. Stopping rule identical in definition to the collocated one, on the staggered layer's exact inlet flux. Measurements in docs/reports/staggered_integration_step6.md. | Alex Moroz-Smietana |
| 2026-09-22 | REQ-S07 rationale and the ADR-010 summary no longer claim a VAL-002 v defect: that error was measured against a corrupted Ghia v table, replaced by Table II as reference ghia_1982_re100_r2 (ECR-001 erratum). Requirement text unchanged. | Alex Moroz-Smietana |
| 2026-09-19 | REQ-S02 rationale corrected: the measured VAL-001 error on 80x40 is 2.04%, identical on CI and locally, which is why the criterion is 2.5% rather than 2%. The 1.54% previously recorded in PROJECT_PLAN.md was not reproducible at the commit that claimed it. Requirement value unchanged; the ECR-001 tightening to < 1% after the rebuild is unaffected. | Alex Moroz-Smietana |
| 2026-09-19 | solve_steady gains an optional on_iteration callback plus last_pressure_sweeps and stage_seconds attributes for the benchmark harness (scripts/benchmark.py). Observability only; solver logic unchanged. | Alex Moroz-Smietana |
| 2026-09-23 | The review Action (.github/workflows/review.yml) removed; the opening paragraph now names the local review and test commands that check branches against this document. No requirement, contract or module changed. | Alex Moroz-Smietana |
| 2026-09-24 | stopping.py added, the error_estimate stopping rule: the estimated iteration error over the largest prescribed boundary velocity, the worst per-cell mass imbalance, and the summed imbalance over the through-flow, each below its tolerance, with no pass at the iteration cap. REQ-S01 and REQ-S04 clarified, not amended: under error_estimate REQ-S01's tolerance applies to the estimated iteration error, and REQ-S04's per-cell tolerance is enforced at stopping together with the summed bound on the flux drift. SimConfig gains three optional solver keys (stopping_rule, default velocity_step; iteration_error_tol; mass_imbalance_tol) and rejects any unknown solver key. StaggeredSolver gains converged and stop_reason and is bitwise unchanged under the default rule; NavierStokesSolver refuses error_estimate. Contract, cascade rows and the system map updated. See docs/reports/stopping_rule_evidence.md, section 9. | Alex Moroz-Smietana |
| 2026-09-25 | StaggeredSolver exposes flux_scale, read-only: the flux scale its error_estimate rule was built with, None under velocity_step, so scripts can record the value the rule ran on. No behaviour change. The stopping.py component row names all three conditions. | Alex Moroz-Smietana |
| 2026-09-25 | Cascade row for solver_staggered.py lists scripts/val001_order.py (review 25 S1) and scripts/self_convergence.py (ECR-001 step 8 report, F8), the scripts that read the solver's public shape, and says the harness stores velocity_step_below_tol as residual_below_tol. No requirement, contract or module changed. | Alex Moroz-Smietana |
| 2026-09-30 | REQ-S04 clarified again, not amended: error_estimate gains condition (d), the absolute signed domain sum of the per-cell imbalance below mass_imbalance_tol, ECR-001 criterion 6's domain-sum clause, which nothing checked before (review 27 B1). No configuration key. The imbalance callable returns an ImbalanceSummary (worst, absolute_sum, signed_sum) in place of a pair; stopping.py gains RULE_VERSION, which scripts/stopping_probe.py and scripts/val001_order.py (through its new reuse_key) store with saved solves. Contract and the stopping.py cascade row updated. Cavity stops bitwise unchanged; channel stops later. See docs/reports/stopping_rule_evidence.md, section 10. | Alex Moroz-Smietana |
| 2026-10-01 | The harness records RULE_VERSION in the params of every error_estimate row and keys its summary on it; an error_estimate row without it reads as version 2, the three-condition rule (review 28 B1). The stopping.py cascade row names scripts/benchmark.py. The three rows taken at 8aac137, never on main, were replaced by rows that carry the version. Condition (d) accepted as built: it is met at a zero crossing of the net outflow's decaying oscillation (docs/reports/stopping_rule_evidence.md, section 10). No requirement, contract or rule behaviour changed. | Alex Moroz-Smietana |
| 2026-10-01 | The ADR-010 summary row states continuity as ADR-010's planned-against-built table does (review 27 S2); REQ-S03's floor against Ghia placed where its source puts it, u near y = 0.85 on the vertical centerline and v at the jet stations (S3); REQ-S02's measured value is the row taken under stopping rule version 3. No requirement value changed. | Alex Moroz-Smietana |
| 2026-10-01 | Section 3.2 rows for config.py, mesh.py, solver_ns.py, solver_staggered.py and csolver/ and the section 4 headings for mesh.py, solver_ns.py and solver_staggered.py brought in line with the generated import graph and with section 2.1 (review 27 B2): existing importers listed from the graph, planned consumers marked, the transport and time-integration consumers of the NS solver routed to solver_staggered, csolver/ to the solver of record. A constants.py row added. The solver_staggered contract says it lacks compute_residual and solve_timestep. No requirement, module or generated region changed. | Alex Moroz-Smietana |
| 2026-09-30 | ECR-001 step 9. REQ-S02 amended to < 1% on 80x40 (ECR-001 section 6) and REQ-S03 to score against marchi_2009_re100 with ghia_1982_re100_r2 reported unscored (amendment of 2026-09-24), each with its measured value. REQ-S11 amended to the per-axis, mirrored stretching step 1 built; REQ-S04's Verified By corrected (VAL-007 is Phase 3's). REQ-S01, S07, S08, S09 and S12 checked against the build and unchanged. Section 2.1 names the staggered solver as the NS solver; the collocated solver and boundary.py are described as the kept harness baseline, no longer as retired in a later step, and the NavierStokesSolver contract no longer claims staggered storage. ADR-010 added as a file and its summary row updated; ADR-008's row corrected to 2.5%. No module, interface or generated region changed. | Alex Moroz-Smietana |
| 2026-10-02 | The collocated solver retired (PR 29, Alex's decision of 2026-10-02): src/solver_ns.py, src/boundary.py and their tests deleted; annotated tag collocated-final on 98f8b1f, the last commit holding them; IterationState moved to stopping.py with its fields unchanged. Section 2.1 names the staggered solver as the only solver; REQ-S02's rationale keeps the collocated measurement as history, and REQ-S12.1's names the staggered layer and the Phase 3 concentration layer as the registry's readers. Section 3.2 rows and section 4 contracts for the two modules removed; the solver_staggered row and contract state the public shape in its own terms; a Retired modules note added to section 4. The harness and the viewer default to staggered-jacobi, and the 22 stored collocated rows still summarize. The adaptive outer iteration is deferred to a later efficiency pass, not Phase 3. No staggered computation changed: val002_20x20 reproduces bitwise. REQ-S02 and REQ-S03 move from the deleted solver_ns.py annotation to solver_staggered.py's, which changes the generated traceability table; REQ-S08 stays with pressure.py alone, which does that work. The scenarios.py cascade row and section 4 heading name the boundary layers (planned, Phase 4) in place of the deleted boundary module. | Alex Moroz-Smietana |
| 2026-10-03 | ADR-011, the Phase 3 transport design, added as a file with status Proposed (PR 30). REQ-S13 (the NS solver exposes its face velocities) and REQ-T11 (a uniform field stays uniform to the bound the stopping rule's per-cell condition sets) added as proposed, each with rationale and verification; no existing requirement's text changed. The solver_transport.py contract stub replaced by ADR-011's draft, marked planned, and a boundary_concentration.py contract added, marked planned; the staggered.py and solver_staggered.py contracts gain FaceVelocities and face_velocities, marked planned. Cascade rows for both planned modules; the particles.py row names its second consumer and the sign conventions. No module or generated region changed. | Alex Moroz-Smietana |
