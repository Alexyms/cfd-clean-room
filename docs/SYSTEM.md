# System Architecture Document

**Project:** CFD Clean Room Simulation
**Status:** ECR-001 closed; the staggered Navier-Stokes solver is built and validated and is the only solver (ADR-010; the collocated solver was retired on 2026-10-02, tag `collocated-final`). ECR-002, the k-epsilon turbulence model (ADR-012), was accepted on 2026-10-04 and closed on 2026-10-15 at step 9 (prompt 58): the model on a face field (`src/scalar_scheme.py`, `src/turbulence.py`), its coupling into the transport solver (`src/solver_transport.py`) and the flow solver (`src/momentum.py`, `src/solver_staggered.py`, `src/stopping.py`), the fixed-flow outlets, the wall functions and stopping condition (e) are built, the product configuration is `configs/clean_room_default.yaml`, and ADR-012's planned-against-built table records the build; section 2 marks each requirement's verification. ECR-003, the pressure correction solver (ADR-013), was accepted on 2026-10-06 and closed on 2026-10-07: its step 1 replaced the weighted Jacobi sweep by Jacobi-preconditioned conjugate gradients (REQ-S08 amended), and its step 2 retook the laminar baseline (REQ-S02, REQ-S03). Phase status is in `docs/PROJECT_PLAN.md`.
**Last Updated:** 2026-10-17

This document is the single reference for system architecture, requirements, module interfaces, and dependency relationships. Review and test, run before each pull request as `/cfd-review` and `/cfd-test` in fresh Claude Code sessions (`.claude/commands/`), check branches against this document under the policy in `docs/REVIEW_POLICY.md`. Keep it current.

---

## 1. System Description

A from-scratch Computational Fluid Dynamics engine simulating clean room airflow and contamination transport. The system solves incompressible Navier-Stokes equations for a 2D velocity field using the Finite Volume method, then solves advection-diffusion equations for particle concentration across five size classes on top of that velocity field. An alert monitoring layer tracks contamination against ISO 14644 thresholds for reactive detection and proactive sensor placement analysis.

The simulation domain is a vertical cross-section of a semiconductor clean room with HEPA supply vents, return vents, an entry door, process equipment, and a laminar flow hood.

The product's purpose is comparative (a stakeholder need, recorded by Alex on 2026-10-09, prompt 45): the tool says where particles accumulate in the room and how a change of layout moves that. It does not defend absolute particle counts. Its product validation (VAL-018, ECR-002 step 8) asks for the deposition hotspots and the ranking of layouts to be stable under grid refinement, under the two k-epsilon variants, and across the turbulent Schmidt number's literature range (0.2 to 1.3). Amended 2026-10-15 (Alex, prompt 56): the variant clause is closed, the product running the standard variant with VAL-020's caveat (turbulence over the equipment tops about 30% high, inflating turbulent mixing and absolute deposition there by a similar factor in every layout), so VAL-018 asks for the hotspots and the ranking of layouts to be stable under grid refinement (160x60 against 320x120) and across the turbulent Schmidt number's range (0.2 to 1.3).

Amended 2026-10-15 (Alex, prompt 58, before the Phase 3 gate): the comparative claim is stated per metric, not by one combined score. Step 8's single score (process-top deposition) ranked the deliberately poor layout best because it sweeps the litho-top emission across the room past the hood entry; the tool's job is a robust ranking on whatever the designer cares about, and a combined score defined now would be fitted to the result. The Phase 3 gate report (`docs/reports/phase3_transport_report.md`) states the claim per metric, ties included. Stated limit of the same date: VAL-021 is closed as a documented gap, since no published measurement fits a 2D slice (`docs/reports/ecr002_val021_feasibility.md`), so the transport's turbulent spreading is unvalidated against measurement until a 3D model (Phase 6) can be compared with one.

---

## 2. Requirements

Requirements are organized by subsystem. Each requirement has a unique ID, a rationale, and a traceability link to the validation test or architectural rule that verifies it.

### 2.1 Solver Requirements

Since ECR-001 step 9 (2026-09-30), "the NS solver" in this table is the staggered solver, `src/solver_staggered.py` (ADR-010), and since 2026-10-02 it is the only solver: the collocated solver it replaced was retired once the ECR had closed with it as the before-and-after baseline (section 4, Retired modules).

| ID | Requirement | Rationale | Verified By |
|----|-------------|-----------|-------------|
| REQ-S01 | The NS solver shall converge to a steady-state velocity field with residuals below a configurable tolerance. | Velocity field accuracy depends on convergence. Divergent or under-converged solutions produce meaningless transport results. Clarified 2026-09-24, not amended: under the staggered solver's `error_estimate` stopping rule (`src/stopping.py`) the configurable tolerance, `iteration_error_tol`, applies to the estimated iteration error, the velocity step times rho_hat / (1 - rho_hat) over the largest prescribed boundary velocity, with rho_hat the step's fitted geometric rate. A small step is not a small error: the error left is about the step times rho / (1 - rho), and rho approaches 1 under refinement. The velocity-step residual against `convergence_tol` stays the default rule and keeps its meaning. See `docs/reports/stopping_rule_evidence.md`, sections 4 and 9. Clarified again 2026-10-04 (ECR-002, accepted), not amended: with the turbulence model on, the `error_estimate` rule also requires the estimated iteration error of the eddy viscosity, over a scale from boundary data (the molecular viscosity plus the largest inlet eddy viscosity), below `iteration_error_tol` (condition (e)), and the rule version a solve records is 3 without (e) and 4 with it. Momentum and transport read nu_t, so a solve must not stop with it still moving (ADR-012 E). Built in ECR-002 step 6 (prompt 45, 2026-10-09): `ErrorEstimateRule(..., nu_scale=...)`, its version a property of the rule, which the solver reports (`rule_version`); every VAL-016 run stops under version 4 with (e) the last condition to hold. Clarified again 2026-10-11 (Alex, prompt 50; ADR-012 E, amended), not amended: (e)'s scale is the molecular viscosity plus the median eddy viscosity of the FLUID cells of the iterate just computed, formed every outer iteration (`stopping.viscosity_scale`) and passed with the step (`ErrorEstimateRule(..., viscosity_condition=True)`, `update(..., nu_scale=...)`), and the version with (e) on is 5. The inlet value was hundreds of times smaller than what a room with weak inlet turbulence makes, and no Annex 20 row could stop under it (`docs/reports/ecr002_val017_annex20.md`, section 5.2; `docs/reports/ecr002_rule_e_scale.md`). | VAL-001, VAL-002; with the turbulence model, tests/test_stopping.py (the (e) tests, the scale's median and its per-step use since prompt 50), tests/test_coupled_solve.py::TestWhatTheSolverExposes (the step and the scale the solver hands (e) each outer iteration, prompts 45b and 50), VAL-016 (tests/test_turbulent_channel.py), VAL-017 (the harness row `val017_216x72`) and VAL-018 (the harness row `val018_320x120`, which stops by the rule at 4,389 outer iterations on the product configuration; docs/reports/ecr002_step8_product.md, prompt 56) |
| REQ-S02 | The NS solver shall reproduce the Poiseuille flow parabolic velocity profile with L2 error < 1% on an 80x40 grid. | Validates basic FV discretization and pressure-velocity coupling against an exact analytical solution. Amended 2026-09-30 (ECR-001 step 9): the criterion returns from 2.5% to the original 1%, as ECR-001 section 6 records for the supersession of ADR-008, now that wall conditions are imposed directly on the staggered components (REQ-S12). The staggered solver measures 4.107e-4 on the uniform 80x40 grid and 3.024e-3 on the same grid clustered to a wall cell of 0.1 H / ny, and converges at order 1.99 under uniform refinement, judged without a reference (ECR-001 criteria 1, 2 and 4; `docs/reports/val001_revalidation_step7.md`, addendum, under stopping rule version 3). The 2.5% and its O(h) rationale belonged to the collocated ghost-cell walls (ADR-008), which measured 2.036e-2 against ADR-008's 2.5%; retired, tag `collocated-final`. Since ECR-003 (closed 2026-10-07) the pressure correction is solved by conjugate gradients, and the measured values are those of the `staggered-cg` rows at 311034e: 4.108e-4 uniform and 3.024e-3 clustered (`docs/reports/ecr003_step2_baseline.md`, section 4). The order 1.99 was measured under the weighted Jacobi correction and is not retaken; the measured CG-against-Jacobi field differences can move it by at most 0.003, and the realized change under CG is at most 3e-4 (section 10 there). | VAL-001 |
| REQ-S03 | The NS solver shall reproduce the lid-driven cavity centerline velocity profiles at Re = 100 with a maximum error below 2% of the lid speed against Marchi, Suero and Araki (2009), reference `marchi_2009_re100`. Ghia et al. (1982), reference `ghia_1982_re100_r2`, is reported beside it and is not scored. | Validates nonlinear advection, 2D pressure gradients, and recirculation handling. Amended 2026-09-30 (ECR-001 step 9), per the amendment of 2026-09-24 under ECR-001 criteria 3 and 3a: scored against Ghia, a correctly converging scheme meets a floor of about 0.005 in u on the vertical centerline near y = 0.85 and 0.009 in v at the jet stations by the right wall, which is Ghia's own error, and a series that must fall under refinement then fails the right answer (`docs/reports/cavity_reference_marchi.md`, section 5). The staggered solver measures u 1.057e-3 and v 7.356e-4 of the lid speed at 80x80 against Marchi, falling at second order from 20x20 (`docs/reports/val002_revalidation_step8.md`). Until 2026-09-22 the Ghia v table in use was not Ghia's (ECR-001 erratum, section 12). Since ECR-003 (closed 2026-10-07), under `staggered-cg` at 311034e: u 1.057e-3 and v 7.356e-4 of the lid speed at 80x80 (`docs/reports/ecr003_step2_baseline.md`, section 4). The second-order fall (2.24 and 2.11 in u, 2.12 and 2.07 in v) was measured under the weighted Jacobi correction and is not retaken; the measured field differences can move each order by at most 6e-4 (section 10 there). | VAL-002 |
| REQ-S04 | The velocity field shall satisfy the incompressibility constraint (divergence-free) to within configurable tolerance at every cell. | Mass conservation is fundamental. FV enforces this by construction, but numerical errors can accumulate. Clarified 2026-09-24, not amended: under the `error_estimate` stopping rule the per-cell tolerance, `mass_imbalance_tol`, is enforced at stopping. A solve is not converged until the worst absolute per-cell mass imbalance of its field is below it, which the velocity-step rule never checked. On an open domain the imbalance sets a drift of the through-flow that the velocity step does not see, and the per-cell bound alone lets that drift grow with the number of cells upstream, more on every finer grid. So the rule also requires the summed absolute imbalance, over rho times the inflow (on a closed domain rho times the velocity scale times the longer side), to be below `iteration_error_tol`. The flux through any cross-section differs from the inflow by at most the summed imbalance on one side of it, so this bounds the drift relative to the through-flow on any grid. Clarified again 2026-09-30: the rule also requires the absolute value of the signed imbalance summed over the domain, the net mass flux out of it, to be below `mass_imbalance_tol`. That is ECR-001 criterion 6's domain-sum clause at the bound the criterion names, beside the per-cell clause above. See `docs/reports/stopping_rule_evidence.md`, sections 4, 5, 9 and 10. Verification corrected 2026-09-30 (ECR-001 step 9): VAL-007 is the transport solver's conservation test (REQ-T05) and does not exist before Phase 3. The staggered VAL-001 and VAL-002 tests assert the stop `error_estimate_and_continuity`, which this tolerance gates. Clarified again 2026-10-06 (ECR-003, accepted), not amended: the corrected faces' per-cell imbalance equals the residual of the p' equation, so `pressure_rtol` bounds it relative to u*'s, and the stopping rule's per-cell condition can hold only where `pressure_rtol` times the 2-norm of b is below `mass_imbalance_tol`. Under ADR-011 G that tolerance shrinks with the smallest cell volume (3.2e-9 kg/s per metre on 200x75, 8e-10 on 400x150), and the standing norm of b at the product's converged state is not measured; the default 1e-8 met the condition on every case measured (`docs/reports/pressure_solver_ecr003.md`, sections 8 and 12). | Unit test, VAL-001, VAL-002; VAL-007 from Phase 3; tests/test_pressure.py (the identity between the residual and the corrected faces' imbalance) |
| REQ-S05 | The solver shall use the SIMPLE algorithm for pressure-velocity coupling. | Industry-standard approach. Well-documented, stable, compatible with structured grids. | Architecture review |
| REQ-S06 | The pressure correction inner loop shall have a pure NumPy reference implementation for validation and a CUDA C++ accelerated implementation for production runs, called from Python via pybind11. The NumPy reference shall remain in the codebase permanently as the ground truth for equivalence testing (see REQ-N03). | NumPy reference validates physics independently of GPU code. CUDA acceleration is a separate deliverable after solver physics are validated. | Integration test |
| REQ-S07 | The solver shall use a staggered (MAC) variable arrangement with pressure at cell centers, u-velocity at east-west cell faces, and v-velocity at north-south cell faces. | Staggered arrangement provides natural pressure-velocity coupling without Rhie-Chow interpolation artifacts. Eliminates checkerboard modes by construction and enables exact discrete continuity enforcement. Required by ECR-001, whose motivating VAL-002 v-velocity error was later found to be measured against a corrupted Ghia table (ECR-001 erratum, 2026-09-22); the wall mass leak it also cites stands. | Architecture review, VAL-001, VAL-002 |
| REQ-S08 | The pressure correction equation shall be solved by the conjugate gradient method preconditioned by one V-cycle of an additive-correction multigrid, or by its diagonal on request, from p' = 0, until the residual r = b + A p', where b is the discrete divergence of u* and r is the corrected faces' mass imbalance, has a 2-norm at most `pressure_rtol` times the 2-norm of b, or at most a rounding floor of 1e-13 times F, the stopping rule's flux scale, confirmed on the true residual at exit, or until the configured iteration cap, which the correction reports. On a closed domain b is projected onto the range of the operator before the solve and the stop reads the projected residual. | Amended 2026-10-06 (ECR-003, accepted; ADR-013) and again 2026-10-14 (prompt 54), built in ECR-003 step 1 (`src/pressure.py`). The requirement's rationale was per-cell data parallelism for the GPU: each Jacobi sweep updates every cell from the previous iterate, one thread per cell. A CG iteration keeps that for its two per-cell operations, the five-point product and the diagonal preconditioner, and adds three global reductions, which GPU libraries provide; the single array operation per iteration is traded for them. Weighted Jacobi needs 28,000 to 378,000 sweeps per correction on the product mesh to reach a relative residual of 1e-3 to 3e-3; CG reaches 1e-8 in about 860 iterations, 0.16 s (`docs/reports/pressure_solver_ecr003.md`, sections 7 and 9). The floor is in the text because on a closed domain it, not the relative level, can end a correction (section 12.2 there). History: from 2026-04-15 the text read "solved using Jacobi iteration"; clarified 2026-09-22, not amended, to weighted Jacobi with w = 2/3, because the plain update has an exact -1 eigenvalue on the closed-domain system (`docs/reports/pressure_correction_step5.md`, sections 3 and 5), a clarification ADR-013 supersedes. Amended 2026-10-14 (prompt 54, Alex's decision on `docs/reports/pressure_multigrid.md`, section 8): ADR-013 decision 1's named successor, built as an option in prompt 53 (`src/multigrid.py`), is the default preconditioner: one additive-correction multigrid V-cycle per CG iteration, data-parallel per cell on every level, with the stop, the floor and the cap unchanged. The solver key `pressure_preconditioner: jacobi` selects the diagonal, which every result stored before prompt 54 used, for parity work and the Phase 6 port. | tests/test_pressure.py: CG against a dense solve on open and closed systems, the stop, the floor, the true-residual check, the cap and its flag; VAL-001 and VAL-002 under the change (ECR-003 step 2); tests/test_multigrid.py: the multigrid preconditioner's coarse operators against the dense Galerkin product, its symmetry and positivity, MG-PCG against Jacobi-PCG and a dense solve, the `jacobi` path bitwise the base loop and the absent key bitwise the multigrid correction |
| REQ-S09 | The advection term shall be discretized using the QUICK scheme (Leonard 1979) with specialized stencils at boundary-adjacent cells. | QUICK provides second-order accuracy globally on smooth flows, appropriate for the moderate-Peclet regime of cleanroom flows. Does not require first-order upwind fallback of hybrid schemes. Required by ECR-001. | VAL-001, VAL-002 |
| REQ-S10 | Under-relaxation factors for velocity (default 0.7), pressure (default 0.3) and, with the turbulence model on, the eddy viscosity (`alpha_turbulence`) shall be configurable via the YAML configuration. | SIMPLE requires under-relaxation for stability. Factors control convergence rate vs. stability tradeoff. Configurable per REQ-C01. Amended 2026-10-04 (ECR-002, accepted): the coupled iteration feeds mu_t back into momentum (ADR-012 D), so its under-relaxation is configured with the other two. `alpha_turbulence` is validated at load since ECR-002 step 1 and read from step 6. | Unit test |
| REQ-S11 | The mesh shall support independent geometric stretching in x and y directions, clustered toward both walls of an axis and mirrored about its midpoint, specified per axis by either the wall-adjacent cell width or the geometric expansion ratio, the other derived from the cell count. | Enables resolution clustering near walls without uniform refinement of the entire domain. Required by ECR-001. Amended 2026-09-30 (ECR-001 step 9) to what step 1 built. The earlier text asked for the spacing and the ratio per wall, but at a fixed cell count a symmetric geometric distribution has one free parameter (ECR-001 criterion 2 note), and the two walls of an axis share it. Clustering toward one wall of an axis only, or toward an interior region, is not built. | Unit test |
| REQ-S12 | Dirichlet velocity boundary conditions shall be imposed directly on the staggered velocity components at the physical wall location, without ghost cell interpolation. | Eliminates the O(h) wall accuracy limitation previously documented in ADR-008. Required by ECR-001. | Unit test, VAL-001 |
| REQ-S12.1 | The interpretation of configured boundary segments (which segment covers a point on a domain edge, its type, and the velocity it prescribes there) shall be implemented once and shared by every boundary imposition layer. | Derived from REQ-S12, for modularity rather than physics: the staggered velocity layer reads one configuration interpretation today, and the Phase 3 concentration layer (`src/boundary_concentration.py`, decided 2026-10-02) will be its second reader; a second copy could drift between them. | Unit test |
| REQ-S13 | The NS solver shall expose the face velocities of its last solve on their staggered storage locations: u on vertical faces [ny, nx+1], v on horizontal faces [ny+1, nx], float64, contiguous, read-only copies, whose two-face averages are the returned cell means and whose per-cell mass imbalance is `last_mass_imbalance`; when the solve stopped by `error_estimate`, that imbalance satisfies REQ-S04's per-cell and domain-sum clauses. | Proposed by ADR-011 (PR 30), section A, status Proposed. Continuity is enforced on the faces and the returned cell means are their averages, which do not carry it (ADR-010, Consequences); the transport solver advects with the face fluxes, the constancy test (REQ-T11) predicts its drift from their imbalance, and reconstructing faces from the means would give an O(h^2) imbalance the stopping rule never bounded. Adding the attribute changes no existing signature. | tests/test_solver_staggered.py::TestFaceVelocities (the first two promises, bitwise) and tests/test_solver_staggered.py::test_val001_40x20_continuity_remeasured_from_the_exposed_faces (the continuity clause, validation marker); VAL-012 (tests/test_constancy.py, planned) |
| REQ-S14 | When configured, the NS solver shall model turbulence with the k-epsilon model in the variant ADR-012 A names, adding the eddy viscosity `rho C_mu k^2 / eps` to the molecular viscosity in the momentum equation's diffusive and stress terms. | The product room runs at Re 89,500 on its height, and the laminar solver, validated at 5 and 100, does not converge on it (`docs/reports/product_case_reynolds.md`); decision 1 of 2026-10-04 chose k-epsilon. Added 2026-10-04 (ECR-002, accepted). The model on a prescribed face field built in ECR-002 step 1 (`src/turbulence.py`, both variants); its coupling into momentum is steps 4 and 6. Step 4 (prompt 43, 2026-10-08) built momentum's side: `MomentumPredictor.predict` takes a cell viscosity field with ADR-012 D's face rule and form b's stress source, and `StaggeredSolver.solve_steady` a prescribed `eddy_viscosity`, handed on as `mu + rho nu_t`; the model's field reaches it in step 6. Step 6 (prompt 45, 2026-10-09) built the coupled solve: with the turbulence section, each outer iteration predicts with the model's under-relaxed nu_t, corrects, and steps k and eps on the corrected faces (ADR-012 D's note of that date); the product room's run is prompt 46's measurement. | VAL-015 (tests/test_decaying_turbulence.py, both variants), VAL-016 (tests/test_turbulent_channel.py: the developed 2D Couette profile equals the same-grid 1D solve to 1e-4, and the model's core k is within 1% of u_tau^2 / sqrt(C_mu); the full matrix in docs/reports/probe45/), tests/test_coupled_solve.py (the first outer iteration rebuilt by hand, bitwise), VAL-018 (the harness row `val018_320x120` and docs/reports/ecr002_step8_product.md: the product room and two layouts on the exact grid pair, the ranking stable under refinement and across the Schmidt number's range; prompt 56); VAL-017 (tests/test_annex20.py: the Annex 20 case on 216x72 pinned to its record and compared with the measurements unscored, its convergence clause a strict xfail until condition (e)'s scale is decided; docs/reports/ecr002_val017_annex20.md, prompt 49) and VAL-019 (tests/test_backward_step.py: the step case's loader, data and readers; the harness row `val019_656x60`, reported and not scored until its criterion is set; docs/reports/ecr002_val019_backstep.md, prompt 52) and VAL-020 (tests/test_impinging_jet.py: the impinging jet's loader, digitised data and readers against a second computation; the harness row `val020_272x90`, reported and not scored until its criterion is set; docs/reports/ecr002_val020_impinging_jet.md, prompt 55); the momentum coupling (step 4): tests/test_viscosity_channel.py (developed flow under a viscosity varying across the channel, against its integral, second order) and tests/test_momentum.py::TestViscosityFaceRule, TestStressSource and TestFieldPathAgainstTheProbe |
| REQ-S15 | The turbulent kinetic energy and its dissipation rate shall be positive, k > 0 and eps > 0, at every non-SOLID cell after every outer iteration. | The eddy viscosity is defined only then; ADR-012 C shows the scheme guarantees it, without clipping. Added 2026-10-04 (ECR-002, accepted). Built for the step on a prescribed field (ECR-002 step 1): KEpsilonModel.step raises PositivityError unless k and eps are positive and finite at every non-SOLID cell; the coupled solve calls it every outer iteration from step 6 (built, prompt 45), and stops on it naming the outer iteration. The argument assumes corrected faces that close every cell: a pressure correction stopped at its cap leaves them open, and the 48-row Couette run at 5,000 iterations raised it at outer iteration 6 (ADR-012 C's note of 2026-10-09). | tests/test_coupled_solve.py::TestRefusals::test_a_positivity_error_names_the_outer_iteration; every VAL-016 solve; tests/test_turbulence.py::TestPositivity (the Smith-Hutton field and a random divergence-free field with production on, both variants and both advection schemes, with the planted explicit decay as the control); the assertion in KEpsilonModel.step; VAL-015 (tests/test_decaying_turbulence.py) |
| REQ-S16 | With the turbulence model off, the NS solver shall reproduce VAL-001 and VAL-002, and the transport solver its gate rows, bitwise. | The Phase 2 gate and the Phase 3 gate rows stand on these results. The obstacle wall stencil (ADR-012 B) changes laminar results in rooms with obstacles, none of which has a validated result. Added 2026-10-04 (ECR-002, accepted); checked at each ECR-002 step that changes a solver. Step 1 changed the transport solver (its scheme moved to scalar_scheme.py): every field the transport tests' solver returned, 20,649 of them over 39 tests, the gate rows among them, hashed equal to main's, and a reordered diagonal sum changed three hashes while every test passed (results/builder35/bitwise.md, rerun after prompt 35b). Step 2 changed it again (the eddy_viscosity keyword): with no field the 43 step-taking tests of the seven transport files returned fields and budgets hashed equal to main's at 4ece52f over 20,662 steps, and a field of zeros is bitwise the None path (tests/test_solver_transport.py::TestEddyViscosityNoneAndZero; results/builder40/). Step 4 changed the flow solver (prompt 43, 2026-10-08): with no field, `val001_80x40`, `val001_80x40_stretched` and `val002_80x80` hash equal to the base (c7d88d91..., 9838d632..., a8a61e28..., this machine) at commits A (69f5323) and B2 (8068a04), the commit before the obstacle stencil and the last commit to change code (c673d2b after it changed only docstrings); B1 (44719ab) was not measured, and the cases have no SOLID cell for its stencil to act on (`TestObstacleWallStencil::test_a_room_without_obstacles_has_no_obstacle_face`); a zero eddy viscosity is bitwise the path without one, and the obstacle wall stencil changes only rooms with SOLID cells. Retaken 2026-10-14 (prompt 54): multigrid became the default pressure preconditioner, which changes every iterate, so the hashes of record under the default are now f2e5b888cb269112, 4c79ffb9def55fbb and 95efc0123aaabc55 (commit 759b3ad, this machine, `docs/reports/multigrid_default.md`); the three above are the diagonal's, kept, and the solver key `pressure_preconditioner: jacobi` reproduces them bitwise. ADR-014 step 0b (prompts 61 and 61b, 2026-10-16) changed both solvers for the equipment vents: the eleven flow hashes of record reproduce at e3ae27c (`docs/reports/adr014_step0b.md`, section 4), and the transport tests run under results/builder35/'s hash plugin at 776b36a return fields and budgets bitwise equal to origin/main's (3a06c7d) in all 60 step-taking tests main has, over 45,004 steps; the branch's six new vent tests have no counterpart (`results/builder61b/gate_cmp_776b36a.txt`). | Hashes of the VAL-001 and VAL-002 faces against main (from step 4; step 4's in its pull request); tests/test_solver_staggered.py::TestEddyViscosity and tests/test_momentum.py::TestFieldPathLimits (the None and zero paths); the transport solver's returned fields hashed against main (step 1, results/builder35/; step 2, results/builder40/); the transport gate tests unchanged |
| REQ-S17 | Walls and obstacle faces shall be treated by the wall treatment ADR-012 B names (ADR-012 decision 3; ranked first: scalable wall functions, the law of the wall evaluated no closer than the floor where it meets the linear law, y* = 11.53 for kappa 0.41 and E 9.793). | y+ on the product mesh is about 7 to 70 (ADR-012 B). Added 2026-10-04 (ECR-002, accepted). Step 4 (prompt 43, 2026-10-08) built two of its three parts: obstacle faces take the domain edge's wall stencil (half-cell distance, wall value zero, Leonard's boundary form in QUICK), and `MomentumPredictor.predict` takes the wall viscosity per wall face, domain edges and obstacle faces, through `wall_mu`; the wall function that fills it is step 6's. Step 6 (prompt 45, 2026-10-09) built the third: `TurbulenceBoundary` in `src/turbulence.py`, the wall viscosity on every wall and obstacle face, eps held and the production given in the wall cells, y*_0 computed from kappa and E. Known limitation (ADR-012 C's note of 2026-10-09): the strain in the second cell from a wall is overstated on every grid (du/dy 33% to 34% high, S^2 76% to 81%), inflating k there and, by diffusion, in the core. | VAL-016 (tests/test_turbulent_channel.py); tests/test_wall_functions.py (the wall viscosity against its formula and equal to mu at y*_0, the floor from the constants, the faces and cells that take a wall function, the wall cells' values, the corner rule under reflection, the inflow values and the start; TestTurbulenceBoundaryValues, prompt 45b: k_P the two cells' mean, y_P the half cell on a stretched mesh, the floor in the wall viscosity and the production, each edge's own wall speed); the obstacle stencil and the hook (step 4): tests/test_momentum.py::TestObstacleWallStencil, TestObstacleStencilOnEverySide, test_solid_walls_solve_as_the_domain_edges, test_an_obstacle_corner_face_adds_nothing_to_the_deferred_source and TestWallViscosity |
| REQ-S18 | A fan-set outlet shall be a fixed-flow outlet segment (`fixed_flow_outlet`): an outward normal velocity and zero tangential velocity in the flow solver, an outflow in the concentration layer, not counted as inflow. A segment states its velocity or states none; those that state none share what the stated ones leave of the discrete inflow at one face velocity, so outflow equals inflow to rounding on any mesh. With no pressure outlet in the configuration at least one fixed-flow outlet states none, and a remainder that is not positive on the mesh is refused; with a pressure outlet, every fixed-flow outlet states its velocity. A configuration with a fixed-flow outlet and no velocity inlet is refused at load (Alex, 2026-10-08, prompt 42b): its make-up air would enter through a pressure outlet. A pressure outlet keeps the zero-gradient copy, valid for developed outflow normal to the face; a configuration whose outlet cells take sideways flow drifts under it (GitHub issue 61), which is why the product's returns and hood are fixed-flow outlets. Applies with the turbulence model on or off. | Amended 2026-10-08 (ECR-002 step 3 built, prompt 42; ADR-012 decision 1 amended by Alex). As accepted on 2026-10-04 the requirement held a reversed pressure-outlet face at zero normal velocity for the iteration; that rule is not built. The outlet probe (`docs/reports/ecr002_step3_outlet_probe.md`, sections 7.3 and 7.7) measured the copy rule at every open return failing the 40x15 ladder at Re 895 and 8,950 whether the reversed faces are held shut or left open, and the pressure rising without end where outlet cells take sideways flow (GitHub issue 61); fixed-flow outlets converge at both rungs and leave the pressure still. The fan-set flows are also what the hardware does. Added 2026-10-04 (ECR-002, accepted). | VAL-001 and VAL-002 hash identically to the base (the hashes are in the pull request for prompt 42 and reproduce with `docs/reports/probe42/` and the baseline report's section 2.3 recipe); `tests/test_fixed_flow_outlet.py` (the configuration rules, the resolved velocities on a channel, the staggered faces, the concentration faces) and `tests/test_fixed_flow_product.py` (the balance to 1e-14 on four meshes, the hood's exact velocity, the shared return velocity, the pin, the concentration faces); the demonstrations of ECR-002 criterion 6 against probe arm D0 (`docs/reports/probe42/fixed42.py`) |

### 2.2 Transport Requirements

| ID | Requirement | Rationale | Verified By |
|----|-------------|-----------|-------------|
| REQ-T01 | The transport solver shall solve the advection-diffusion equation for particle concentration on the velocity field produced by the NS solver, with each class's diffusivity its Brownian coefficient plus, when the NS solver models turbulence, the turbulent particle diffusivity of REQ-T13. | Core physics coupling. Particles are carried by airflow and spread by diffusion. Amended 2026-10-04 (ECR-002, accepted): the diffusivity becomes a per-face field (ADR-012 F); with no turbulence model it is the Brownian coefficient, as before. The turbulent term is built (ECR-002 step 2, prompt 40): the solver takes an eddy viscosity field and forms the face diffusivity from it. | VAL-003 (tests/test_diffusion.py), VAL-004 (tests/test_advection.py), unchanged; tests/test_solver_transport.py and tests/test_scalar_scheme.py for the per-face diffusivity |
| REQ-T02 | The transport solver shall support five discrete particle size classes: 0.1, 0.3, 0.5, 1.0, and 5.0 um. | Spans the relevant physics regimes from diffusion-dominated to settling-dominated and maps to ISO 14644 classification sizes. | Unit test |
| REQ-T03 | Each size class shall have independent settling velocity computed via Stokes drag with Cunningham slip correction. | Settling velocity varies by 1000x across the size range. Cunningham correction is significant below 1 um. | VAL-005; the increment's placement by tests/test_solver_transport.py and VAL-014 (tests/test_sealed_box.py) |
| REQ-T04 | Each size class shall have independent Brownian diffusion coefficient computed via Stokes-Einstein relation with Cunningham correction. | Diffusion dominates transport for sub-micron particles. | VAL-006; the operator by VAL-003 (tests/test_diffusion.py) |
| REQ-T05 | The transport solver shall conserve total particle mass to within 0.01% across all timesteps (mass in domain + mass out = mass in + mass from sources). | FV formulation guarantees conservation by construction. This test catches numerical bugs. | VAL-007 (tests/test_conservation.py) |
| REQ-T06 | The transport solver shall accept an external force field parameter (v_ext) for future extension to electrostatic precipitation modeling. The parameter shall be structurally present but set to zero in v1. | Architectural extensibility per ADR-007. Avoids future refactoring of the solver interface. Note added 2026-10-03 (ADR-011 C; premise review 30 S9): the planned transport contract types v_ext as a per-class drift velocity on the staggered faces, FaceVelocities-shaped. In the Stokes regime a body force on a particle is a drift velocity through the class's mobility, which is how settling_velocity treats gravity, so a force field is converted once by its owner with ParticlePhysics and the solver adds faces. The text above is not amended. | Architecture review; tests/test_solver_transport.py (v_ext accepted, applied and bounded in stable_dt) |
| REQ-T07 | Pure diffusion from a point source shall produce a Gaussian concentration profile with L2 error < 1% vs the analytical solution. | Validates the diffusion discretization independently of advection. | VAL-003 (tests/test_diffusion.py) |
| REQ-T08 | Advection of a concentration pulse in a uniform flow shall preserve peak location to within 1 cell width and maintain pulse shape. | Validates advection discretization. Excessive numerical diffusion indicates the scheme is too dissipative. | VAL-004 (tests/test_advection.py, both rows) |
| REQ-T09 | The particles module shall compute gravitational and diffusional deposition velocity for each size class, parameterized by boundary layer thickness and surface orientation (floor, ceiling, wall). | Deposition velocity is a boundary condition input for the transport solver. Floor deposition includes gravitational settling; ceiling and wall deposition are diffusion-only. | VAL-010 |
| REQ-T10 | The particles module shall estimate HEPA filter collection efficiency for each size class via interpolation of reference efficiency data. | HEPA efficiency determines the particle removal rate at supply vent boundaries. Efficiency varies by particle size with a minimum at the most-penetrating particle size (~0.3 um). | VAL-011 |
| REQ-T11 | A spatially uniform concentration field advected by a face velocity field that meets REQ-S04's per-cell and domain-sum clauses at its stop, with every inlet carrying the same concentration and with diffusion, settling, deposition and sources switched off, shall stay uniform: after a simulated time T the largest over non-SOLID cells of a cell's relative departure from its initial value shall not exceed the largest over cells of |b_P| T / (rho V_P), where b_P is the cell's mass imbalance in that field and V_P its volume. | Proposed by ADR-011 (PR 30), section G; reworded 2026-10-03 on premise review 30 B3, B4 and S13 (the inlet clause, the per-cell form, and the product configuration moved to a deliverable). A per-cell velocity imbalance is a source or sink of particle mass (ADR-010, Consequences, For Phase 3), and a conservative face-flux scheme that is exact on a uniform field turns the stopping rule's bound into a drift rate, b_P / (rho V_P) per second in cell P. Advection alone does not preserve a uniform field at an inlet carrying a different value, so every inlet carries the uniform value. VAL-007's budget closes by telescoping on any face field and cannot see this; the two instruments are kept apart. Measured on the saved VAL-001 80x40 histories at the stop: 7.05e-9 per second in the worst cell, 0.0025% per hour (results/builder30/constancy_drift.json; the derivation is ADR-011 G). The bound is stated for face fields produced under `error_estimate`; the product configuration runs under `velocity_step` today and moves to `error_estimate` with `mass_imbalance_tol <= 1e-4 rho V_min / t_end` when the product case is measured (Alex, 2026-10-03; ADR-011 decision 6). | VAL-012 (tests/test_constancy.py) |
| REQ-T12 | Concentration shall be non-negative at every cell and every time step, and under pure advection shall never exceed the largest value present in the field and at its inlets. | Added 2026-10-03 (Alex; ADR-011 decision 4, premise review 30 S1). A concentration below zero is physically meaningless, a negative value at a sensor is a reading the monitor cannot interpret, and Phase 5 scores these fields against ISO limits. No linear scheme above first order is monotone (Godunov), so the transport solver bounds QUICK's face value with a TVD limiter under forward Euler at Courant number at most 1/2, where each update is a convex combination of neighbouring values (ADR-011 B); clipping after the step was rejected because it creates mass. The bounds are exact up to rounding, and the tests allow 1e-14 of the peak. | VAL-013 (tests/test_smith_hutton.py); VAL-004's no-negative clause (tests/test_advection.py); VAL-014 on a step front (tests/test_sealed_box.py); the implicit sink by tests/test_solver_transport.py |
| REQ-T13 | With the turbulence model on, each class's diffusivity on every interior face shall be its Brownian coefficient plus `nu_t / Sc_t`, Sc_t configurable. | The classes are tracers to the turbulence (ADR-012 F); Sc_t is configurable and 0.7 in the default configuration (ADR-012 decision 8). Added 2026-10-04 (ECR-002, accepted); built in ECR-002 step 2 (prompt 40, 2026-10-07): the face value of nu_t is the distance-weighted harmonic mean of the two cell values, zero when either is; Sc_t is the optional `transport.turbulent_schmidt`, with no default in code; a call without the field is bitwise the laminar path (REQ-S16). | tests/test_scalar_scheme.py::TestPiecewiseDiffusivity (a dense solve with a piecewise diffusivity, and its mean-diffusivity control); tests/test_solver_transport.py::TestEddyViscosityFaceRule, TestEddyViscosityRefusals, TestEddyViscosityPositivity, TestEddyViscosityNoneAndZero and TestEddyViscosityIsADiffusivity; VAL-007, VAL-012 and VAL-013 rerun with an eddy-viscosity field (the `with_an_eddy_viscosity_field` tests of tests/test_conservation.py, tests/test_constancy.py and tests/test_smith_hutton.py) |

### 2.3 Configuration Requirements

| ID | Requirement | Rationale | Verified By |
|----|-------------|-----------|-------------|
| REQ-C01 | All simulation parameters shall be defined in a single YAML configuration file. No simulation parameter shall be hardcoded in source modules. | Lesson from Stock Transformer: scattered parameters cause synchronization failures. | Architecture review |
| REQ-C02 | The configuration loader shall validate all parameters at load time and raise clear errors for missing keys, out-of-range values, and type mismatches. | Fail fast. Do not let invalid config propagate to a solver crash 10 minutes into a run. Since 2026-10-03 (PR 31, Alex's decisions 1 to 3 on review 31) this covers a boundary segment carrying a key it does not accept, two segments overlapping on one edge, a concentration key on an inlet whose normal velocity is zero, and a NaN or infinite value wherever a number is expected: each would change the physics with no message if it loaded. | Unit test |
| REQ-C03 | Scenario configuration files shall inherit from a base configuration and override specific parameters only. | Avoids duplicating the full config for each scenario. Changes to base config automatically propagate. | Unit test (tests/test_config.py, TestScenarioInheritance and test_the_reference_scenarios_extend_the_default_room; ADR-014 step 1) |
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
| REQ-N01 | The simulation shall enforce the CFL condition for numerical stability at every timestep. | Explicit advection schemes are conditionally stable. Violating CFL produces divergent solutions. | Unit test (tests/test_solver_transport.py: stable_dt, and a step above it refused) |
| REQ-N02 | The solution shall demonstrate grid convergence at the expected order of accuracy when the grid is refined. | Confirms numerical accuracy. Second-order scheme should reduce error by 4x when grid spacing is halved. | VAL-008 |
| REQ-N03 | The CUDA C++ solver inner loop shall produce results identical to the pure NumPy reference implementation for all validation cases. Equivalence is defined as maximum absolute element-wise difference below 1e-10 for all output arrays. | Verifies the Python-CUDA interface is not introducing bugs through memory layout, dtype, or indexing errors. | Integration test |

---

## 3. Module Dependency Map

This section defines which modules depend on which. When a PR modifies a module, the code reviewer checks that downstream modules are still compatible.

### 3.1 Dependency Graph

Generated from the import statements in `src/` by `scripts/gen_system_map.py`. Regenerate with `python scripts/gen_system_map.py`; CI runs it with `--check` and fails a PR that changes `src/` without regenerating.

<!-- BEGIN GENERATED: dsm -->
```
                         boundary_concentration  boundary_registry  boundary_staggered  boundary_vents  config  constants  io_manager  mesh  momentum  multigrid  particles  pressure  scalar_scheme  scenarios  solver_staggered  solver_transport  staggered  stopping  turbulence
boundary_concentration             .                     X                  .                 X           X         .          .        X       .          .          X         .            .            .             .                 .              X         .          .
boundary_registry                  .                     .                  .                 .           X         .          .        .       .          .          .         .            .            .             .                 .              .         .          .
boundary_staggered                 .                     X                  .                 X           X         .          .        X       .          .          .         .            .            .             .                 .              X         .          .
boundary_vents                     .                     X                  .                 .           X         .          .        X       .          .          .         .            .            .             .                 .              X         .          .
config                             .                     .                  .                 .           .         .          .        .       .          .          .         .            .            .             .                 .              .         .          .
constants                          .                     .                  .                 .           .         .          .        .       .          .          .         .            .            .             .                 .              .         .          .
io_manager                         .                     .                  .                 .           .         .          .        .       .          .          .         .            .            .             X                 .              .         .          .
mesh                               .                     .                  .                 .           X         .          .        .       .          .          .         .            .            .             .                 .              .         .          .
momentum                           .                     .                  X                 .           X         .          .        X       .          .          .         .            .            .             .                 .              X         .          .
multigrid                          .                     .                  .                 .           .         .          .        .       .          .          .         .            .            .             .                 .              .         .          .
particles                          .                     .                  .                 .           X         X          .        .       .          .          .         .            .            .             .                 .              .         .          .
pressure                           .                     .                  X                 .           X         .          .        X       X          X          .         .            .            .             .                 .              X         .          .
scalar_scheme                      .                     .                  .                 .           .         .          .        X       X          .          .         .            .            .             .                 .              .         .          .
scenarios                          X                     X                  .                 X           X         .          .        X       .          .          X         .            .            .             .                 .              .         .          .
solver_staggered                   .                     .                  X                 .           X         .          .        X       X          .          .         X            .            .             .                 .              X         X          X
solver_transport                   X                     .                  .                 .           X         .          .        X       .          .          X         .            X            .             .                 .              X         .          .
staggered                          .                     .                  .                 .           X         .          .        X       .          .          .         .            .            .             .                 .              .         .          .
stopping                           .                     .                  .                 .           .         .          .        .       .          .          .         .            .            .             .                 .              .         .          .
turbulence                         .                     X                  X                 X           X         .          .        X       .          .          .         .            X            .             .                 .              X         .          .
```

Rows import columns. Edges, 61 total:

```
boundary_concentration -> boundary_registry, boundary_vents, config, mesh, particles, staggered
boundary_registry      -> config
boundary_staggered     -> boundary_registry, boundary_vents, config, mesh, staggered
boundary_vents         -> boundary_registry, config, mesh, staggered
io_manager             -> solver_staggered
mesh                   -> config
momentum               -> boundary_staggered, config, mesh, staggered
particles              -> config, constants
pressure               -> boundary_staggered, config, mesh, momentum, multigrid, staggered
scalar_scheme          -> mesh, momentum
scenarios              -> boundary_concentration, boundary_registry, boundary_vents, config, mesh, particles
solver_staggered       -> boundary_staggered, config, mesh, momentum, pressure, staggered, stopping, turbulence
solver_transport       -> boundary_concentration, config, mesh, particles, scalar_scheme, staggered
staggered              -> config, mesh
turbulence             -> boundary_registry, boundary_staggered, boundary_vents, config, mesh, scalar_scheme, staggered
```

Cycles of any length: **none**. Checked by depth-first search over the whole graph, not by looking for mutual pairs. A three-module cycle is the one that actually happens and a pair check answers 'none' in its presence.

This matrix is generated from the import statements in `src/` and describes the modules that exist today. Section 3.2 (cascade rules) is hand-authored and stays that way: the generator owns what the structure is, and the cascade rules are a human judgment about what follows from it.
<!-- END GENERATED: dsm -->

### 3.2 Cascade Rules

When a PR modifies a module, the reviewer verifies impact on downstream modules. Read this table as: "If you change X, check Y."

| Modified Module | Check These Downstream Modules | What to Check |
|-----------------|-------------------------------|---------------|
| config.py | boundary_concentration, boundary_registry, boundary_staggered, boundary_vents, mesh, momentum, particles, pressure, scenarios, solver_staggered, solver_transport, staggered, turbulence; monitor, time_integration (planned) | New/changed/removed config fields are handled by all consumers. No module accesses a field that no longer exists. No module ignores a new field it should use. The optional transport block (SimConfig.transport, a TransportSpec of cfl_number, advection_scheme, max_diffusion_iter, diffusion_tol and the optional turbulent_schmidt, None when absent) is read by solver_transport, which refuses a configuration without it, and refuses an eddy viscosity field when turbulent_schmidt is absent; the optional turbulence block (SimConfig.turbulence, a TurbulenceSpec, None when absent) by turbulence, which refuses a configuration without it and advects k and eps with its advection_scheme (upwind when absent, prompt 48; the transport block's key of the same name and its umist default are separate), and by solver_staggered, which runs the coupled solve with it (ECR-002 step 6); the segment keys (concentration, hepa_filtered, deposition_surface, each allowed on one segment type) by boundary_concentration (ADR-011 E and I); output_interval becomes FieldHistory's interval. pressure_rtol, max_pressure_iter and the optional pressure_preconditioner (multigrid when absent since prompt 54, jacobi on request) are read by pressure (ECR-003 step 1); momentum_sweeps by momentum (ECR-002 step 4); pressure_tol is refused at load, so no module can read it. SOLVER_KEYS is the one ordered list of solver keys and scripts/benchmark.py records every row's params from it, so a key added here is in every row (GitHub issue 38). CFL_NUMBER_BOUND, PRESSURE_RTOL_BOUNDS, PRESSURE_PRECONDITIONERS (PRESSURE_JACOBI, MULTIGRID), the scheme names and the deposition surface names are this module's constants and consumers import them. The boundary type `fixed_flow_outlet` (ECR-002 step 3, ADR-012 D as amended 2026-10-08) is read by boundary_registry (`condition_of`, `fixed_flow_condition`) and boundary_staggered (the resolved velocities); a configuration with no pressure_outlet needs at least one fixed-flow outlet that states no velocity or stated flows that balance the inflow (ADR-014 J's closed box), and one with a pressure_outlet needs every fixed-flow outlet to state its velocity, both refused at load, as is a fixed-flow outlet in a file with no velocity_inlet, a `velocity` on a pressure_outlet and a blank `velocity` on a fixed_flow_outlet. The scenario section (ADR-014 B, step 1): SimConfig.scenario, a ScenarioSpec or None, read by scenarios; its times(), active(t) and mapping(events) are the one derivation of an interval's configuration, so the loader's validation of every interval and the manager's intervals cannot disagree; an empty replacement removes a segment; merge_mappings is decision 7's rule (mappings merge, lists replace whole). |
| constants.py | particles | Constant names and SI values unchanged; no module defines its own copy (REQ-C04). |
| mesh.py | boundary_staggered, momentum, pressure, scalar_scheme, scenarios, solver_staggered, staggered, solver_transport, turbulence; monitor (planned) | Grid dimensions, cell arrays, and coordinate arrays are consumed correctly. Shape assumptions still hold. Centers stay face midpoints; staggered averaging depends on it. solver_transport and turbulence read dx_face and dy_face for the diffusive flux on a stretched mesh and cell_type for which cells hold a scalar; scalar_scheme reads the node and face coordinates for its axes; turbulence reads the face coordinates for the corner derivatives. obstacle_mask is the one rule that puts an obstacle on the mesh: cell_type reads it, and the vents' faces (staggered.obstacle_face_inputs) are the faces of the cells it returns, so a change to the rule moves both together (ADR-014 J). |
| staggered.py | solver_staggered, boundary_staggered, boundary_concentration, boundary_vents, momentum, pressure, solver_transport, turbulence; tests/test_constancy.py, tests/test_conservation.py | Face array shapes and the face-to-center averaging contract unchanged. FaceVelocities (ADR-011 A, REQ-S13) is the transport solver's and the k-epsilon model's input type, so its field names, shapes and read-only owning float64 copies are part of that contract and the layout module is no longer internal to the velocity solver alone. edge_cells and edge_cell_inputs are where both boundary layers read which cells sit behind an edge's faces and which are SOLID; a change there moves both layers together, which the agreement test cannot see, so tests/test_staggered.py::TestEdgeCells checks them directly. obstacle_face_inputs is the same for the equipment vents (ADR-014 J): every vent layer reads a vent's faces through it, by way of boundary_vents.vent_runs, so its index order and its corner rule move the velocity, turbulence and concentration layers together. |
| boundary_registry.py | boundary_staggered, boundary_concentration, boundary_vents, scenarios, turbulence | Coverage rule (same edge, inclusive range, first match in configuration order, wall by default, SOLID cells walls) and the prescribed-velocity decomposition unchanged. A fixed_flow_outlet carries its stated velocity outward, or zero when it states none; the share of a segment that states none is the velocity layer's to resolve (fixed_flow_condition), so a layer reading `condition_of` alone sees zero for it. coverage_along is the one derivation of which faces a segment covers; both layers call it on staggered.edge_cell_inputs, and tests/test_boundary_concentration.py checks they agree on every committed configuration and on two with an obstacle on an edge under an inlet. A change here moves both layers. coverage_on is the vents' counterpart (ADR-014 J), read through boundary_vents.vent_runs by the velocity, turbulence and concentration layers; a vent's condition is condition_of on ROOM_EDGE_OF_FACE, this module's table since ADR-014 step 2 (staggered.py's before), so the sign of a vent's velocity follows that table. |
| boundary_staggered.py | momentum, pressure, solver_staggered, turbulence | Normal imposition writes domain faces only. Tangential data shape [n+1], outlet data shape [n], wall_distance semantics and the inward flux sign unchanged. A fixed-flow outlet face is Dirichlet in both components (its resolved outward velocity, tangential zero), is not a pressure outlet and not inlet flux, and does not enter get_max_boundary_velocity; a domain whose only outlets are fixed-flow reports has_pressure_outlet False, which is what engages the corrector's pin (ECR-002 step 3). wall_faces (ECR-002 step 6) marks a wall where the normal velocity is zero and the face is not an outlet, per edge cell and per corner; turbulence's wall functions read it. The vents (ADR-014 J): apply_normal_velocity also writes interior entries, every vent face, which momentum must keep rather than predict as zero; the inlet flux sums, the remainder rule and get_max_boundary_velocity count the vents, so the reference velocity, the corrector's flux_scale and the stopping rule's velocity scale move when a configuration gains one; vent_runs() and vent_faces() are the vent data momentum and turbulence read, built from boundary_vents. A configuration without vents is bitwise unchanged. |
| boundary_vents.py | boundary_staggered, boundary_concentration, scenarios, turbulence | ADR-014 J's vent faces, moved here from boundary_staggered at ADR-014 step 2 (review 61 S2) so the concentration layer reads them without importing the velocity layer. vent_runs is the one derivation of a vent's faces (staggered.obstacle_face_inputs for the faces, the registry's coverage_on for which vent covers each) and VentRun's fields, order and read-only arrays are what the three layers share, so a change here moves the velocity, turbulence and concentration layers together; vent_face_masks' four masks keep their shapes. The refusals (a vent that covers no face; a recirculating pair whose flows differ on the faces by more than MESH_BALANCE_TOLERANCE) stay here. A configuration without vents gets no runs and all-False masks. |
| boundary_concentration.py | solver_transport; scenarios (conditions_for builds one per interval) | ConcentrationFaces array names, shapes and dtypes (face-shaped, read-only; float64 inflow and deposition, int32 surface codes, bool settling_v), the surface codes SURFACE_NONE 0, SURFACE_FLOOR 1, SURFACE_CEILING 2, SURFACE_WALL 3 the budget books deposition to, the settling_v mask (there is no settling_u; settling acts in -y), and the derivation of each condition from the registry's velocity type with the segment keys concentration, hepa_filtered and deposition_surface unchanged (ADR-011 E). |
| momentum.py | pressure, solver_staggered, scalar_scheme | MomentumPrediction shapes and the meaning of a_p_u and a_p_v (un-relaxed diagonal, positive exactly at the unknown faces) unchanged; boundary entries of u and v read as given and never written. predict's two optional arguments (ECR-002 step 4): with mu_eff and wall_mu None the coefficients are bitwise the scalar path's, and solver_staggered passes mu_eff only when given an eddy_viscosity; the obstacle wall stencil changes coefficients only beside SOLID cells, which no validated case has. scalar_scheme reads quick_face_values only, for the transport solver: its signature, the `left` index of the low-side node of each face, the `positive` flow mask, the far node one past the upstream node (or one past the downstream node where that does not exist) and the caller placing the boundary value at its physical location as the end node; a change to any of these changes the transport face value. |
| turbulence.py | solver_staggered (the coupled solve, ECR-002 step 6); tests/test_turbulence.py, tests/test_decaying_turbulence.py, tests/test_wall_functions.py, tests/test_coupled_solve.py, tests/test_turbulent_channel.py | TurbulenceState's fields (k, eps and the kinematic nu_t, [ny, nx], read-only, SOLID zero) and the positivity promise: step raises PositivityError unless k and eps are positive and finite at every non-SOLID cell. TurbulenceConditions' fields, shapes and dtypes, built by the caller every step (step 6 builds them from the wall functions, the inlet keys and the staggered boundary layer's tangential values); a wall cell needs its eps held and its production given together, its eps following k, or k runs away beside a shear (prompt 35). step's two kinds of dt: None the per-cell pseudo-time step, capped, an error on a field at rest; a float one true-time step, refused above the stable step. VARIANTS' values are the sourced ones (results/builder35/constants.md); a change to one passes every test but tests/test_turbulence.py::TestConstants. TurbulenceBoundary (step 6): wall_viscosity returns wall_mu's layout, mu_w on wall_faces and the caller's values elsewhere; conditions builds a TurbulenceConditions from a state and corrected faces; initial_values is the solve's start. KAPPA, E_WALL and Y_STAR_FLOOR are module constants that tests/couette_reference.py copies and tests/test_turbulent_channel.py checks. |
| scalar_scheme.py | solver_transport, turbulence; tests/test_scalar_scheme.py and, through solver_transport, the transport gate tests (tests/test_advection.py and tests/test_smith_hutton.py plant their unlimited-face control on scalar_scheme.limited_face_values) | The transport gate's bits: implicit_step sums the diagonal west, east, south, north, then the sink, and the caller forms each face conductance before the call; a change of either order changes the gate's results without failing a test (prompt 35's hash control, results/builder35/). limited_face_values' clamp and its c_c guard on a zero downstream difference; advective_flux's boundary nodes (the inflow value where the flux enters, the adjacent cell otherwise) and its reading of a far node in a SOLID cell as the upstream value; implicit_step's non-negative result for non-negative input at any step, scalar or per cell, and its held mask (None the unheld path, bitwise; a held cell exactly its C*), which turbulence uses for the wall cells' eps. The method each caller names (prompt 51): the transport solver JACOBI, whose iterates keep a concentration's sign down to zero; turbulence CONJUGATE_GRADIENT, whose fallback to Jacobi from C* on the cap or a non-positive value keeps the result's sign; ImplicitResult's field names, which the model's last_iterations and last_fallbacks and the probes' traces read. |
| pressure.py | solver_staggered; scripts/benchmark.py and tests/test_constancy.py (coefficients, mass_imbalance); tests/test_turbulence.py (mass_imbalance); scripts/stopping_probe.py and scripts/val001_order.py (PRESSURE_SOLVER_VERSION); scripts/benchmark.py, scripts/view_field.py and scripts/self_convergence.py (STAGGERED_METHOD) | PressureCorrection shapes, the right-hand side formed directly from face velocities with no compatibility correction, outlet faces corrected against p' = 0 with the nearest interior diagonal, closed-domain projection before the solve and pin at the first FLUID cell after it, the iteration count and reached_cap reported, flux_scale formed here and read by the solver. The stop: pressure_rtol on the relative residual, RESIDUAL_FLOOR times flux_scale, the true residual confirmed at exit, max_pressure_iter the cap. A change to the solve that moves saved fields beyond rounding raises PRESSURE_SOLVER_VERSION, which scripts/stopping_probe.py stores with every saved solve (each reader of a saved truth goes through solve_truth, which checks it) and scripts/val001_order.py puts in its reuse key, and which selects STAGGERED_METHOD from STAGGERED_METHODS, the method label (staggered-mgcg at version 3) the harness, the viewer and scripts/self_convergence.py import and carry in rows and file names, the harness also running a case under the diagonal as staggered-cg, which sets the preconditioner the label names; a version with no label of its own fails at import, so no saved field or row of one solver is reused as another's. |
| multigrid.py | pressure (multigrid_preconditioner); tests/test_multigrid.py; docs/reports/probe53/ | Multigrid(coefficients, apply) and cycle(r) unchanged in shape; the coarse operators stay the exact Galerkin product of piecewise-constant 2x2 aggregation, so a closed domain's constant null space and SOLID rows reach every level; the cycle stays symmetric (the same weighted Jacobi sweep and count before and after) and positive on the range, as CG requires, and zero at cells with no equation. SMOOTHING_SWEEPS, SMOOTHER_FACTOR and COARSEST_CELLS change the iteration counts and times of every multigrid correction; a change to them repeats prompt 53's per-correction measurement and moves every laminar hash. Under the default key this runs in every pressure correction, so every field, the laminar hashes included, depends on it; under `jacobi` nothing here runs. |
| solver_staggered.py | face_velocities is the transport solver's input, read by tests/test_constancy.py and tests/test_conservation.py (no import: solver_transport takes a FaceVelocities); time_integration (planned, Phase 4: the NS solver of section 2.1); scripts/benchmark.py, scripts/view_field.py, scripts/stopping_probe.py, scripts/val001_order.py, scripts/self_convergence.py | The public shape: cell-centered [ny, nx] float64 contiguous returns, the IterationState callback once per outer iteration with cell-centered fields and the corrector's iteration count, last_pressure_iterations, pressure_cap_hits and stage_seconds reset per solve, reference_velocity F_ref / (rho h). Under the default velocity_step rule the stop is the residual below convergence_tol, the definition every stored velocity-step row was taken under, so outer iteration counts compare with them, except that since 2026-10-06 an outer iteration whose correction reached max_pressure_iter cannot stop the solve (ADR-013 B); residual_history keeps that definition under both rules. The harness records pressure_cap_hits in every row. converged and stop_reason are set by every solve and reset at its start; the harness records them, with velocity_step_below_tol stored as residual_below_tol, the label every stored velocity-step row carries. face_velocities is None before the first solve and set at the end of every solve, converged or not (REQ-S13). Since ADR-014 step 0 (prompt 60): flow_state and steady_checkpoint are set beside it; initial=None and resume=None are the solve from rest, bitwise (the hashes of record, REQ-S16), a warm start rebuilds nu_t from k and eps and restarts the rule, and a resume continues the checkpoint's solve bitwise, so time_integration (planned) and io_manager depend on both meanings staying apart. |
| stopping.py | solver_staggered, scripts/stopping_probe.py, scripts/val001_order.py, scripts/benchmark.py, scripts/self_convergence.py | IterationState's fields, which the solver hands its on_iteration callback and the harness reads, unchanged in position; the count field is pressure_iterations since 2026-10-06 (pressure_sweeps before), and pressure_products, added last the same day, both read by name in scripts/benchmark.py's WorkCounter, which counts the work in products. update(step, imbalance) answers converged only when the estimate, the worst imbalance, the absolute sum over flux_scale and the absolute signed sum are all below their tolerances; the imbalance callable, which returns an ImbalanceSummary read by name from one evaluation, is not called until the estimate is met; no estimate (inf) while the window is short, a step in it is zero or not finite, or rho_hat is outside (0, 1). RATE_WINDOW stays a module constant. Since ECR-002 step 6 the version is the rule's (ErrorEstimateRule.version: RULE_VERSION_WITHOUT_E 3, RULE_VERSION_WITH_E 5, with viscosity_condition on), read through the solver (StaggeredSolver.rule_version, solver_staggered.rule_version(config)) by stopping_probe and val001_order, which store it with their saved solves, so that they solve again, and by the harness, which records it in every error_estimate row, so that its summary keeps the versions apart. update's viscosity_step is given exactly when the rule has a nu_scale. window() and restore() (ADR-014 step 0) carry the rate fits' steps, oldest first, so a restored rule's estimates and stops are its source's; the steady checkpoint of solver_staggered and io_manager stores them. |
| solver_transport.py | time_integration, monitor (planned, Phases 4 and 5); the Phase 7 animation (FieldHistory); tests/test_solver_transport.py, tests/test_diffusion.py, tests/test_advection.py, tests/test_constancy.py, tests/test_smith_hutton.py, tests/test_conservation.py, tests/test_sealed_box.py; validation/transport_cases.py hands it stand-ins | solve_timestep takes FaceVelocities, not cell means, a sources rate array and the keyword-only eddy_viscosity field (nu_t per cell, m^2/s, None the laminar path bitwise; the face value is the distance-weighted harmonic mean, so a caller that hands another face rule changes the contract), and returns a new [ny, nx] float64 contiguous field with SOLID cells zero; stable_dt's definition, the sum of the two directional rates, infinite at rest; v_ext a FaceVelocities-shaped per-class drift on the interior faces, None read as zero (REQ-T06); MassBudget field names (initial, inflow, outflow, source, deposited by floor, ceiling, wall, obstacle, current) and that only the solver writes them; FieldHistory's record signature and npz keys (steps, times, C_<k>). The solver reads settling_velocity and diffusion_coeff of physics and faces_for of boundary and nothing else of either, so the validation cases hand it ScalarPhysics and FixedConditions (validation/transport_cases.py); a change to what it reads changes that contract. The vents (ADR-014 J, step 0b): ConcentrationFaces' vent_u and vent_v open faces of SOLID cells to the advection and the budget, and a recirculating discharge carries the flux-weighted mean of its intake's room cells in the field the step starts from, times its passed fraction; a configuration without vents runs the earlier lines bit for bit. |
| particles.py | boundary_concentration, scenarios, solver_transport | Settling velocity, diffusion coefficient and deposition velocity interface unchanged. Return types, units and signs unchanged (settling positive downward, deposition non-negative; the floor value includes settling, which the solver must not add again, ADR-011 D). |
| scenarios.py | time_integration (planned, Phase 4); tests/test_scenarios.py | Built at ADR-014 step 2. intervals(), flow_states(), sources_at(t), conditions_for(interval), initial_fields() and n_flow_states unchanged in signature and meaning; an event's start and end are interval boundaries, active from its start and expired at its end. The fingerprint reads FLOW_KEYS only, over the segments in configuration order with contiguous pieces of one segment joined, so a transport-only event (TRANSPORT_KEYS) never adds a flow state; every BoundarySpec field is in exactly one of the two tuples, which tests/test_scenarios.py checks, so a new segment key must be classified before it loads. A state joined from pieces is the split state to the rounding of the inflow sums, not bitwise. sources_at returns solve_timestep's sources, one [ny, nx] rate per class, zero in SOLID cells. |
| monitor.py | time_integration | Update and query interfaces unchanged. Alert output format unchanged. |
| csolver/ | solver_staggered (REQ-N03's NumPy reference, ADR-010), solver_transport; both planned, Phase 6 | CUDA kernel signatures match pybind11 declarations. Memory layout assumptions (row-major, contiguous, double precision) unchanged. |
| time_integration.py | nothing in src/ (planned, Phase 4; ADR-014 G: nothing imports it) | Timestep sequence and output trigger logic unchanged: one step for every class, the interval to the next switch divided into equal steps, the concentration fields and the budget carried across a switch unchanged (ADR-014 A). |
| io_manager.py | time_integration (planned, Phase 4); tests/test_io_manager.py | The run file's groups and dataset names (ADR-014 D) unchanged, so a run file written before a change still restarts after it; /steady_checkpoint/ (step 0) holds every field of SteadyCheckpoint as "<f8" so a resume from it is the uninterrupted solve bitwise. |

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
| `src/boundary_concentration.py` | 429 | Derives the per-face concentration conditions of each particle class from the registry's shared coverage and ParticlePhysics: the concentration an inlet carries, the deposition velocity and surface code at every wall and obstacle face, and the mask of faces that carry the settling increment, as read-only face-shaped data for the transport solver. | S12.1, T09, T10 |
| `src/boundary_registry.py` | 469 | Interprets the configured boundary segments once, answering which segment covers each point along a domain edge and which condition and prescribed velocity hold there, SOLID cells read as walls, for both boundary imposition layers: the staggered velocity layer and the concentration layer. | S12.1 |
| `src/boundary_staggered.py` | 693 | Writes Dirichlet normal velocities exactly into the staggered domain-face entries and exposes the tangential wall values, wall distances, pressure outlets and wall faces as data for the momentum and pressure steps and the wall functions. | S12 |
| `src/boundary_vents.py` | 250 | Derives the mesh faces each equipment vent covers on its obstacle face, from the staggered layout's obstacle-face inputs and the registry's first-match coverage, with the face masks the velocity, turbulence and concentration layers read, and refuses a vent that covers no face or a recirculating pair whose flows differ on the faces. | S12.1 |
| `src/config.py` | 2402 | Loads the YAML configuration into typed dataclasses and rejects missing keys, wrong types and out-of-range values at load time. | A02, A03, C01, C02, C03, S10 |
| `src/constants.py` | 8 | Holds the physical constants shared by every module so that none of them defines its own copy. | C04 |
| `src/io_manager.py` | 124 | Writes a steady solve's checkpoint into the run file's steady_checkpoint group and reads it back bitwise, so a solve stopped at its cap continues in another process; the rest of the run file is ADR-014 step 3's. | none |
| `src/mesh.py` | 442 | Builds the structured grid, uniform or geometrically clustered at the walls, with the face, center, width and center-to-center arrays a face-based stencil needs, and classifies each cell as FLUID, SOLID or BOUNDARY. | S11 |
| `src/momentum.py` | 1060 | Predicts u* and v* on the staggered grid with QUICK advection by deferred correction over an upwind implicit matrix, a configured number of under-relaxed Jacobi sweeps per call, an optional per-cell viscosity field with its face rule and stress source, and the domain edge's wall stencil at obstacle faces, and returns the diagonal coefficients the pressure correction needs. | S07, S09, S14 |
| `src/multigrid.py` | 384 | Builds an additive-correction multigrid hierarchy from a five-point pressure system by 2x2 aggregation, its coarse operators the exact Galerkin sums with no probing, and applies one symmetric V-cycle with weighted Jacobi smoothing and a dense pseudo-inverse on the coarsest level as the preconditioner of the pressure correction's conjugate gradients. | S08 |
| `src/particles.py` | 263 | Computes per-size-class transport properties: Cunningham correction, settling velocity, Brownian diffusion, deposition velocity and HEPA efficiency. | T03, T04, T09, T10 |
| `src/pressure.py` | 859 | Assembles the staggered pressure correction equation from the momentum diagonals with the discrete divergence of u* as its right-hand side, solves it by conjugate gradients preconditioned with its diagonal or, under the pressure_preconditioner key, with one multigrid V-cycle per iteration, to a relative residual, a rounding floor or a reported iteration cap, corrects the face velocities and updates the pressure. | S04, S08 |
| `src/scalar_scheme.py` | 568 | Holds the cell-centred scalar scheme the transport solver and the k-epsilon model share: QUICK's face value bounded by the UMIST limiter, the advective flux along one axis with the inflow value or the upwind cell at a domain face, and the backward Euler solve of diffusion with a non-negative cell sink by Jacobi or, for k and eps, by diagonal-preconditioned conjugate gradients that fall back to Jacobi on the cap or a non-positive value, on per-face conductances with one step or a step per cell, with an optional mask of cells held at their value. | S15, T12 |
| `src/scenarios.py` | 404 | Turns a loaded scenario into its intervals between event boundaries, keys each interval's flow state by a fingerprint of the segment keys the velocity layer reads so transport-only events add no state, and gives the source rate of every class at a time, an interval's concentration boundary and the initial fields on the mesh; it solves and marches nothing. | none |
| `src/solver_staggered.py` | 893 | Runs steady SIMPLE on the staggered grid as one outer loop over the momentum predictor and the pressure correction, with the turbulence section one k and eps step and the relaxed eddy viscosity inside it, returning cell-centered fields through the harness's callback shape and exposing the final faces as FaceVelocities; stops by the velocity-step rule or, when configured, by the error-estimate rule. | S01, S02, S03, S04, S05, S07, S13, S14, S15 |
| `src/solver_transport.py` | 1031 | Advances one particle class one explicit step on the staggered face velocities: QUICK's face value bounded by the UMIST limiter under forward Euler at a Courant number the configuration sets, implicit diffusion and deposition by Jacobi on per-face conductances, the Brownian coefficient plus nu_t / Sc_t on interior faces when handed an eddy viscosity field, the settling increment on interior faces, sources added and booked, SOLID cells zero; keeps one MassBudget per class and defines FieldHistory, the output contract for the animation. | N01, T01, T03, T04, T05, T06, T07, T08, T11, T12, T13 |
| `src/staggered.py` | 443 | Defines the staggered (MAC) field layout: shapes and allocation of face-centered u and v and cell-centered p, the face-to-center averaging the solver applies before returning, and FaceVelocities, the read-only face pair the solver exposes and the transport solver advects with. | S07, S13 |
| `src/stopping.py` | 408 | Decides when the steady outer iteration has converged, on four conditions: (a) the iteration error estimated from the step and its fitted geometric rate, over a physical velocity scale; (b) the worst per-cell mass imbalance against its own tolerance; (c) the summed imbalance over the through-flow, which shares the tolerance of (a); and (d) the signed imbalance summed over the domain, which shares the tolerance of (b); and with the turbulence model a fifth, (e), the same estimate on the eddy viscosity over the molecular viscosity plus the median eddy viscosity of the FLUID cells, which the solver passes each outer iteration. Also defines IterationState, the snapshot a solver hands its callback once per outer iteration. | S01, S04 |
| `src/turbulence.py` | 1622 | Advances the k-epsilon model's k and eps one step on a prescribed face velocity field, standard or RNG with each variant's constants as module data: advection, explicit growth from the strain the faces give and implicit decay and diffusion through the shared scalar scheme, boundary values from a conditions object the caller builds each step, the kinematic eddy viscosity, and an assertion that k and eps are positive and finite after every step; and the scalable wall functions, which build those conditions, the wall viscosity of the momentum wall faces and the inlet values the coupled solve starts from. | S14, S15, S17 |

Total 20 Python files, 12752 lines. 1 empty `__init__.py` carry no row: a package marker with no code has no responsibility to record.

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
| Files hashed | 20 |
| Digest | `sha256:15aaf96967860c4f030ae2a1ec79b33a23f94aaf39e4902e46fb06c232dc61e7` |

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
    SimConfig.from_dict(raw: dict, base_dir=None)   # same validation on a parsed mapping;
                                    # base_dir resolves a scenario's relative base (ADR-014 step 1)
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
    max_pressure_iter: int      # the CG iteration cap; required, no default; 5000 committed
    pressure_rtol: float        # relative residual of the pressure correction, in
                                # PRESSURE_RTOL_BOUNDS = [1e-10, 1); 1e-8 committed (ADR-013 D)
    momentum_sweeps: int        # optional, positive, default DEFAULT_MOMENTUM_SWEEPS = 1:
                                # momentum Jacobi sweeps per outer iteration (ECR-002 step 4)
    stopping_rule: str  # optional: "velocity_step" (default) or "error_estimate"
    iteration_error_tol: float  # optional, default 1e-6; error_estimate only
    mass_imbalance_tol: float  # optional, default 1e-10; error_estimate only
    pressure_preconditioner: str  # optional: "multigrid" (default, MULTIGRID, since
                                # prompt 54) or "jacobi" (PRESSURE_JACOBI);
                                # PRESSURE_PRECONDITIONERS lists both and
                                # DEFAULT_PRESSURE_PRECONDITIONER names the default;
                                # anything else raises
    # SOLVER_KEYS is the one ordered tuple of the fourteen keys above; the harness
    # records every row's params from it. solver.pressure_tol, the retired Jacobi
    # stop (RETIRED_PRESSURE_TOL_KEY), raises ValueError naming pressure_rtol;
    # any other key in the solver block raises ValueError at load
    transport: TransportSpec | None  # None when the section is absent (the validation cases)
        cfl_number: float (0, CFL_NUMBER_BOUND]  # the bound is 1/2, a module constant; 0.1 in use (ADR-011 I)
        advection_scheme: str       # "umist" (default) or "upwind"
        max_diffusion_iter: int     # positive
        diffusion_tol: float        # positive
        turbulent_schmidt: float | None  # optional, positive and finite, no default in code;
                                    # None when absent, and solve_timestep then refuses an
                                    # eddy_viscosity field (REQ-T13, ADR-012 decision 8)
    # any other key in the transport block raises ValueError at load
    turbulence: TurbulenceSpec | None  # None when the section is absent: the model is off (ADR-012 I)
        model: str                  # "k_epsilon"
        variant: str                # "standard" (default) or "rng" (ADR-012 A)
        wall_treatment: str         # "scalable_wall_functions" (ADR-012 B)
        cfl_number: float (0, CFL_NUMBER_BOUND]  # the k and eps pseudo-time Courant number (ADR-012 C)
        advection_scheme: str       # "upwind" (default) or "umist": the face value k and eps
                                    # are advected with (ADR-012 C and I, notes of 2026-10-10);
                                    # independent of the transport block's key and its default
        alpha_turbulence: float (0, 1]  # eddy-viscosity under-relaxation of the coupled solve
        max_iter: int               # positive; CG iteration cap of each k and eps solve
                                    # (Jacobi sweeps until prompt 51; the fallback's too)
        tol: float                  # positive; relative, as diffusion_tol
    # every key required but variant and advection_scheme; any other key raises ValueError at load, the model
    # constants included (they are src/turbulence.py's). With it present the flow solver runs
    # the coupled solve (ECR-002 step 6), stopping_rule must be error_estimate (refused at
    # load otherwise; the rule with condition (e), version 5 since 2026-10-11, scores the
    # turbulent cases), and every velocity_inlet
    # that admits air states turbulence_intensity and dissipation_length.
    boundaries: dict[str, BoundarySpec]
    obstacles: list[ObstacleSpec]
    scenario: ScenarioSpec | None   # ADR-014 B and G, built at step 1; None without the section
    # A file with a scenario section extends scenario.base (a path relative to the file):
    # the file's own sections merge over the base's (merge_mappings: mappings merge,
    # lists replace whole) and every attribute above is the merged base's. The file
    # states solver.t_end, solver.output_interval and transport.cfl_number itself.
    ScenarioSpec: base_path, base (the merged base, read-only), events: tuple[EventSpec, ...], t_end,
        path (the scenario file, None from from_dict), time_tolerance (EVENT_TIME_RTOL t_end)
        times() -> tuple[float, ...]          # 0, every event's start and end, t_end
        active(t) -> tuple[EventSpec, ...]    # start <= t < start + duration; never an initial
        is_active(event, t) -> bool           # the same, start and end read to time_tolerance
        mapping(events) -> dict               # the merged base with their replaces and sets applied
        # checkpoint_interval, listed by ADR-014 G, arrives with the run file (step 3)
    EventSpec: name, start, duration (None for an initial), change; end, segment
        Replace(segment, segments)            # segments: ((name, keys), ...), empty removes
        Set(segment, keys)
        Source(x_start, x_end, y_start, y_end, rate | emission)   # per class
        Initial(values, threshold_factor)     # values per class, resolved at load
    class_thresholds() -> tuple[float, ...]   # each class's threshold by diameter
    # refused at load (ValueError, TypeError for a wrong type): every rule of ADR-014 B
    # (tests/test_config.py, TestScenario*), and every interval's merged mapping must load
    # through from_dict; a recirculating pair of the base whose intake draws while no
    # discharge returns it is refused too. What needs the mesh (a source box with no
    # non-SOLID cell centre) is refused by ScenarioManager.
    # ADR-014 J, the equipment vents (step 0b, built 2026-10-16): BoundarySpec
    #   takes location "obstacle" with keys obstacle (an obstacle's name) and
    #   face (top, bottom, left, right); velocity_inlet and fixed_flow_outlet only
    sensors: list[SensorSpec]
    thresholds: dict[str, float]
```

```
BoundarySpec:
    type: str  # "wall", "velocity_inlet", "pressure_outlet", "fixed_flow_outlet"
    location: str  # "top", "bottom", "left", "right"; "obstacle" for a vent (ADR-014 J)
    x_start, x_end: float | None  # for top/bottom segments and vents on a top or bottom face
    y_start, y_end: float | None  # for left/right segments and vents on a side face
    velocity: float | None  # magnitude, decomposed normal to the boundary; on a fixed_flow_outlet, the outward normal velocity it holds, None to share the remaining inflow
    u_velocity: float | None  # explicit x-component at the face
    v_velocity: float | None  # explicit y-component at the face
    # ADR-011 E; each validated at load, allowed on one segment type only, keyword-only
    concentration: tuple[float, ...] | None  # one non-negative value per class, velocity_inlet only; None is a clean supply
    hepa_filtered: bool                # velocity_inlet only; default False
    deposition_surface: str | None     # wall only: floor, ceiling, wall or none; None lets the edge decide
    # ADR-012 C and I (ECR-002 step 6); keyword-only
    turbulence_intensity: float | None  # (0, 1); velocity_inlet that admits air, turbulence section present
    dissipation_length: float | None    # m, positive; l_e in eps = k^(3/2) / l_e; likewise
    # ADR-014 J (step 0b) and prompt 61's amendment; keyword-only, None on a domain-edge segment
    obstacle: str | None          # the obstacle a vent sits on
    face: str | None              # its face: top, bottom, left, right
    recirculate_from: str | None  # velocity_inlet vent only: the fixed-flow vent on the same obstacle it returns
```

A vent (`location: obstacle`, ADR-014 J) is a `velocity_inlet` (a cooling
discharge) or a `fixed_flow_outlet` (a process exhaust) on an obstacle face,
normal to it, with a required positive `velocity`; `u_velocity` and
`v_velocity` are refused on it (prompt 61: the obstacle layers read a face's
tangential velocity as zero). Refused at load, each naming the segment: an
`obstacle` no obstacle has or two share; a `face` not one of the four; a
missing `obstacle`, `face` or `velocity`; an extent outside the face's span or
empty, or given on the other axis; a face on the domain edge or with another
obstacle standing against it along the extent; two vents overlapping on one
face; a type other than the two; any vent key on a domain-edge segment; and a
`recirculate_from` that is not on a velocity_inlet, names no segment, names a
segment that is not a fixed-flow vent on the same obstacle, comes with a
`concentration`, or names an intake whose flow differs from the discharge's
(prompt 61b; the mesh checks the same on the covered faces, in
`boundary_vents.vent_runs`). An inlet vent states its turbulence keys with the section
present, as every inlet that admits air. That the extent covers a face of the
mesh is checked when the boundary is built.

`turbulence_intensity` and `dissipation_length` are required on every
velocity_inlet with a nonzero normal velocity when the turbulence section is
present, and refused on any other segment type, on a velocity_inlet whose
normal velocity is zero (a moving wall admits no air) and with the section
absent, each naming the segment; a bool, a string, None, NaN or a value out
of range raises.

A `fixed_flow_outlet` (ECR-002 step 3, ADR-012 D as amended 2026-10-08)
holds an outward normal velocity and zero tangential velocity. It takes an
optional positive finite `velocity` and refuses `u_velocity`, `v_velocity`,
`concentration`, `hepa_filtered` and `deposition_surface`, each naming the
segment. With no pressure_outlet in the file, at least one fixed-flow outlet
must state no velocity, and those share what the stated ones leave of the
discrete inflow at one face velocity; with a pressure_outlet, every fixed-flow
outlet must state its velocity. Both rules are checked at load; a remainder
that is zero or negative on the mesh is refused when the boundary is built.
Since step 0b (ADR-014 J) a file with no pressure_outlet whose fixed-flow
outlets all state their velocities loads when the stated outflow equals the
inflow on the configured extents to within 1e-9 relative (the closed box of a
discharge and an exhaust), and the boundary checks the same balance on the
mesh's faces.
Three further refusals at load (Alex, 2026-10-08, prompt 42b): a fixed-flow
outlet in a file with no velocity_inlet (the room's velocity scale comes from
what drives it in, and a room drawing make-up air through a pressure outlet is
the regime of GitHub issue 61, not supported); a `velocity` key on a
pressure_outlet, which would be ignored; and a blank `velocity` on a
fixed_flow_outlet, since "states none" is written by leaving the key out.

A `concentration` or `hepa_filtered` key on a segment that is not a
velocity_inlet, or on a velocity_inlet whose normal velocity is zero (a
tangential lid, a wall to the scalar layer), or a `deposition_surface` on one
that is not a wall, raises ValueError naming the segment; a `concentration`
list of the wrong length, a negative, NaN or infinite value, or a bool where
a number or a string is expected raises too. A key no segment accepts raises
naming the segment and the key. Two segments on one edge whose ranges overlap
in more than a point raise naming both; ranges meeting at one coordinate are
allowed, and the first in configuration order decides there (ADR-011 E,
amended 2026-10-03).

`u_velocity` and `v_velocity` are optional and only meaningful for
`type == "velocity_inlet"`. When either is present, both components are
read directly from the spec (missing component defaults to 0) and the
`velocity` magnitude field is ignored for decomposition. This supports
tangential inflow such as the moving lid in a lid-driven cavity. When
both are None, the `velocity` magnitude is decomposed normal to the
edge (positive inward).

### mesh.py --> staggered, boundary_staggered, momentum, pressure, scalar_scheme, scenarios, solver_staggered, solver_transport, turbulence (monitor planned)

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
        # a cell is SOLID when its center lies in an obstacle's bounds, to within
        # OBSTACLE_EDGE_TOLERANCE (1e-9) of the cell's width on each edge, so an edge
        # drawn through a center does not depend on the center's last ulp
    obstacle_mask(obstacle: ObstacleSpec) -> ndarray[ny, nx] bool
        # the cells one obstacle makes SOLID, by that rule; cell_type and the vents'
        # faces (staggered.obstacle_face_inputs, ADR-014 J) both read it
    is_fluid(i: int, j: int) -> bool
    get_neighbors(i: int, j: int) -> list[tuple[int, int]]
```

### staggered.py --> boundary_concentration, boundary_staggered, boundary_vents, momentum, pressure, solver_staggered, solver_transport, turbulence; tests/test_constancy.py, tests/test_conservation.py

Internal to the solver (REQ-S07), except FaceVelocities, which is the transport solver's
input type (ADR-011 A, REQ-S13), and edge_cells and edge_cell_inputs, which both boundary
layers read.
Layout: u on vertical faces [ny, nx+1], v on horizontal faces [ny+1, nx],
p at cell centers [ny, nx].

```
u_shape(mesh), v_shape(mesh), p_shape(mesh) -> tuple[int, int]
allocate_fields(mesh) -> (u, v, p) zeroed, float64, contiguous
u_face_coordinates(mesh), v_face_coordinates(mesh),
cell_center_coordinates(mesh) -> (X, Y) 2D coordinate arrays
to_cell_centers(u, v) -> (u_c, v_c) each [ny, nx], plain two-face average
check_staggered_pair(u, v) -> None       # ValueError unless the two shapes describe one mesh
edge_cells(cell_type, edge) -> ndarray   # view of the cells behind an edge's faces, in face order
edge_cell_inputs(mesh, edge) -> (coordinates, solid)   # the two inputs coverage_along takes, derived
                                         # once for both boundary layers (REQ-S12.1)
obstacle_face_inputs(mesh, obstacle, face) -> ObstacleFace   # ADR-014 J, step 0b
    ObstacleFace: component ("v" top/bottom, "u" sides), rows, cols (int64, in order along
        the face), coordinates (xc or yc), widths, corner_rows, corner_cols (the tangential
        wall-stencil faces between two of the run's faces, wall_mu layout), corner_coordinates
        # the faces between the obstacle's cells (Mesh.obstacle_mask) and a non-SOLID cell on
        # that side, never a domain-edge face; every vent layer reads its faces from here
FaceVelocities: u [ny, nx+1], v [ny+1, nx]   # frozen, eq=False; float64, C-contiguous, read-only,
                                             # owning its data (ADR-011 A, REQ-S13)
    FaceVelocities.copy_of(u, v)             # owning read-only copies; the constructor refuses
                                             # anything else, a read-only view included
```

### boundary_registry.py --> boundary_staggered, boundary_concentration, boundary_vents, scenarios, turbulence

Configuration interpretation implemented once for every boundary imposition
layer (REQ-S12.1): the staggered velocity layer and the concentration layer.
Reads the config only; no mesh and no field. The layers hand coverage_along
the coordinates along an edge and the SOLID cells behind them, both from
staggered.edge_cell_inputs, so neither layer derives them on its own.

```
EDGES = ("bottom", "top", "left", "right")
ROOM_EDGE_OF_FACE: {top: bottom, bottom: top, left: right, right: left}
                                             # the domain edge an obstacle face plays to the room
                                             # (staggered.py's until ADR-014 step 2)
EdgeCondition: bc_type, u_prescribed, v_prescribed   # frozen
EdgeCoverage: name, spec, condition, solid           # frozen; UNCOVERED and BEHIND_SOLID are the segment-less values
covers(spec, edge, coordinate) -> bool       # same edge, inclusive range
condition_of(spec, edge) -> EdgeCondition    # magnitude decomposed inward, or explicit u/v;
                                             # fixed_flow_outlet: its stated velocity outward, zero if none
fixed_flow_condition(edge, speed) -> EdgeCondition
                                             # type FIXED_FLOW_OUTLET, normal component signed along the axis
                                             # for an outward speed, tangential zero
BoundaryRegistry:
    __init__(config: SimConfig)
    boundaries -> dict[str, BoundarySpec]
    spec(name) -> BoundarySpec               # KeyError if absent
    segment_at(edge, coordinate) -> (name, BoundarySpec) | None
        first covering segment in configuration order
    condition_at(edge, coordinate) -> EdgeCondition
        first covering segment in configuration order; no-slip wall when none
    coverage_along(edge, coordinates, solid) -> list[EdgeCoverage]
        one per point; a SOLID point is a wall with no segment; the one
        derivation of face coverage both boundary layers read (REQ-S12.1);
        ValueError if the two sequences differ in length
    vents() -> dict[str, BoundarySpec]       # ADR-014 J: the segments at location "obstacle"
    coverage_on(obstacle, face, coordinates) -> list[EdgeCoverage]
        the vents' coverage_along on staggered.obstacle_face_inputs' coordinates:
        first covering vent in configuration order, UNCOVERED (a wall) where none,
        the condition the vent prescribes on ROOM_EDGE_OF_FACE[face]
covers_face(spec, obstacle, face, coordinate) -> bool   # a vent on that face, inclusive range
OBSTACLE = "obstacle"                        # config.OBSTACLE_LOCATION
```

### boundary_staggered.py --> momentum, pressure, solver_staggered, turbulence

Direct imposition on the staggered layout (REQ-S12). Two deliverables kept
apart: an imposer for the normal components, which are storage locations on
the domain faces, and data for the tangential and pressure conditions, which
have no storage location on the wall. Nothing is written outside the domain
and nothing is written to p. Which segment covers each edge cell and corner
is read from the registry's coverage_along on staggered.edge_cell_inputs, not
decided here, and get_inlet_flux attributes each face by the segment name that
coverage gives it, so a face is counted for one segment at most.

```
StaggeredBoundary:
    __init__(mesh: Mesh, config: SimConfig)
    apply_normal_velocity(u, v) -> None
        writes u[:, 0], u[:, nx], v[0, :], v[ny, :] where the condition is
        Dirichlet (wall 0, inlet prescribed, fixed-flow outlet its resolved
        outward velocity), exactly; pressure outlet faces and every other
        entry untouched; ValueError on non-staggered shapes
    tangential_conditions() -> dict[edge, TangentialCondition]
        component ("u" on bottom/top, "v" on left/right), is_dirichlet [n+1],
        value [n+1], wall_distance = dy_face[0], dy_face[ny], dx_face[0] or
        dx_face[nx]; arrays read-only
        a fixed-flow outlet's tangential locations are Dirichlet zero
    pressure_outlets() -> dict[edge, PressureOutletCondition]
        is_outlet [n], pressure (gauge datum, 0.0); arrays read-only; a
        fixed-flow outlet is not a pressure outlet
    has_pressure_outlet() -> bool            # False when the only outlets are fixed-flow
    fixed_flow_velocities() -> dict[str, float]
        outward normal velocity per fixed_flow_outlet segment, configuration
        order, m/s: the stated value, or the equal share of the discrete
        inflow less the stated outflow over the faces the shared coverage
        gives the segments that state none (the face widths of the mesh, so
        outflow equals inflow to rounding on any mesh); a copy. __init__
        raises ValueError when those segments cover no face or the remainder
        is zero or negative, naming inflow, stated outflow and remainder
    get_inlet_flux(name) -> float            # sum of inward normal velocity times face width
    get_total_inlet_flux() -> float          # velocity inlets only; a fixed-flow outlet is not counted
    get_max_boundary_velocity() -> float     # the largest velocity the room is driven by: inlets and moving walls; an outlet is not counted
    wall_faces() -> dict[edge, EdgeWalls]    # ECR-002 step 6, the wall functions' faces
        EdgeWalls: cells [n] (the edge cell's face is a wall and the cell is not
        SOLID), corners [n+1] (at the tangential storage locations, from the
        same coverage, a point bounding a SOLID edge cell included); a wall is
        a face with zero normal velocity that is not an outlet, at rest or
        moving; read-only
    A SOLID cell on an edge is a wall on every query.
    # ADR-014 J, the equipment vents (step 0b)
    vent_runs() -> list[VentRun]             # one per vent, configuration order; a copy
        (boundary_vents.vent_runs as built once here)
    vent_faces() -> VentFaces                # boundary_vents.vent_face_masks of those runs
    apply_normal_velocity also writes every vent face's normal velocity; the inlet
    flux sums count a discharge (velocity times its faces' widths); the remainder
    rule counts an exhaust as a stated outlet; get_max_boundary_velocity counts a
    discharge; __init__ raises when a vent covers no face of the mesh, and when every
    outlet states its velocity, none is a pressure outlet and the stated outflow
    differs from the inflow on the faces by more than config.MESH_BALANCE_TOLERANCE
    (1e-12, beside config.BALANCE_TOLERANCE, 1e-9, the same rule on the configured extents)
```

### boundary_vents.py --> boundary_staggered, boundary_concentration, scenarios, turbulence

ADR-014 J's vent faces (step 0b), in their own module since step 2 (review 61
S2): the velocity, turbulence and concentration layers import it, and the
concentration layer no longer imports the velocity layer for them. Reads the
mesh and the registry; no field.

```
vent_runs(mesh, config, registry) -> list[VentRun]   # the one derivation of the vents' faces,
                                                     # read by the velocity, turbulence and
                                                     # concentration layers (REQ-S12.1)
    VentRun: name, spec, edge (boundary_registry.ROOM_EDGE_OF_FACE), component, rows, cols,
        widths, condition, corner_rows, corner_cols; normal (the stored normal velocity), is_inlet
    raises ValueError when a vent covers no face of the mesh, or a recirculating pair's flows
    differ on the faces by more than config.MESH_BALANCE_TOLERANCE
vent_face_masks(mesh, runs) -> VentFaces             # read-only bool masks, all False without vents
    VentFaces: normal_u [ny, nx+1], normal_v [ny+1, nx], tangential_u and
    tangential_v [ny+1, nx+1] in wall_mu's layout; any
```

### momentum.py --> pressure, solver_staggered, scalar_scheme (quick_face_values)

Momentum predictor on the staggered layout (REQ-S07, REQ-S09, REQ-S14): first-order
upwind implicit matrix with the QUICK minus upwind advective flux carried as
an explicit deferred-correction source, ``solver.momentum_sweeps``
under-relaxed Jacobi sweeps per call (default one) with the collocated
convention (diagonal divided by alpha_velocity, ``(1 - alpha) / alpha * a_P *
phi`` added to the source, held at the outer iterate over the sweeps). Since
ECR-002 step 4 (2026-10-08) it takes an optional cell viscosity field with
ADR-012 D's face rule and form b's stress source, the wall viscosity hook,
and the domain edge's wall stencil at obstacle faces. Since step 0b of
ADR-014 (2026-10-16) a vent face (StaggeredBoundary.vent_faces) is kept at
the value apply_normal_velocity wrote, where every other face of a SOLID cell
is predicted as zero; it stays not an unknown (a_p zero), so the pressure
correction puts no coefficient across it and leaves it as predicted, and its
flux enters the corrector's right-hand side and mass_imbalance as stored.
Bitwise the earlier path without vents.

```
quick_face_values(phi, nodes, left, faces, positive) -> face values
    quadratic through C, D and the next node upstream, Lagrange weights
    from the node positions; Leonard boundary form at the ends
MomentumPredictor:
    __init__(mesh: Mesh, config: SimConfig, boundary: StaggeredBoundary)
    momentum_coefficients(u, v) -> (MomentumCoefficients, MomentumCoefficients)
        a_p, a_s_plus, a_s_minus, a_t_plus, a_t_minus, b_boundary, b_deferred,
        each in the component's own shape; s is the component's axis
    predict(u, v, p, mu_eff=None, wall_mu=None) -> MomentumPrediction
        Planned (ADR-014 decision 3 and G, step 6): transient=None, either
            (dt, u_old, v_old) for implicit Euler or (dt, (u_n, u_nm1),
            (v_n, v_nm1)) for BDF2; None bitwise the steady path, as mu_eff
        mu_eff: dynamic effective viscosity per cell, Pa s, [ny, nx] float64,
            finite and positive in every non-SOLID cell, SOLID cells not read;
            None is the scalar path with air's mu, bitwise, and a uniform field
            equal to air's gives the same bytes. Transverse faces: harmonic
            across the row boundary (distance-weighted), then width-weighted
            across the two columns; a SOLID cell takes its non-SOLID
            neighbour's value; form b's stress source rides with b_deferred
        wall_mu: {"u": ndarray, "v": ndarray} or None, each float64 [ny+1, nx+1];
            entry [j, i] is the face through the corner (x[i], y[j]) that the
            component's wall stencil crosses, horizontal for u and vertical for
            v; read only at domain-edge faces whose tangential condition is
            Dirichlet and at obstacle faces, each beside an unknown, where it
            must be finite and positive. The coupled solve (ECR-002 step 6) passes
            the wall functions' viscosity over stencil_viscosity
        TypeError or ValueError on a malformed mu_eff or wall_mu, before any
            arithmetic
        u_star [ny, nx+1], v_star [ny+1, nx]: boundary columns and rows are
            the input values, faces of SOLID cells are zero
        a_p_u [ny, nx+1], a_p_v [ny+1, nx]: un-relaxed diagonal, positive
            exactly at the unknown faces and zero elsewhere, so a_p > 0 is
            the mask of correctable faces and d = A_face / a_p is defined
            precisely there (the step 5 contract)
    stencil_viscosity(mu_eff) -> {"u": ndarray, "v": ndarray}   # ECR-002 step 6
        wall_mu's layout: the viscosity predict(u, v, p, mu_eff) puts on each
        face the wall stencil reads, air's elsewhere; given back as wall_mu
        it changes no bit
    Reads tangential_conditions for the wall values and wall distances;
    boundary entries of u and v (Dirichlet from StaggeredBoundary, outlet
    extrapolation by the caller) are read as given and never written.
    Obstacle faces: a tangential unknown whose transverse neighbour bounds a
    SOLID cell takes the domain edge's wall stencil, the wall at the face
    half a cell away with value zero, and Leonard's boundary form in the
    QUICK correction; a neighbour face bounding SOLID on one side only (an
    obstacle corner) is a wall over its whole span.
    p is the solver's pressure. With the turbulence model on (step 6) it is
    the modified pressure p + (2/3) rho k of ADR-012 D. No datum correction
    is built for pressure outlets (Alex, 2026-10-08): the product room has
    none since ECR-002 step 3, and VAL-001, which has one, runs laminar.
```

### pressure.py --> solver_staggered; scripts/benchmark.py and tests/test_constancy.py (coefficients, mass_imbalance), tests/test_turbulence.py (mass_imbalance), scripts/stopping_probe.py and scripts/val001_order.py (PRESSURE_SOLVER_VERSION), scripts/benchmark.py, scripts/view_field.py and scripts/self_convergence.py (STAGGERED_METHOD)

Pressure correction on the staggered layout (REQ-S04; REQ-S08 as amended
2026-10-06, ADR-013). The right-hand side is the discrete divergence of u*
from the stored face velocities, with no interpolation and no compatibility
correction; walls contribute no coefficient (homogeneous Neumann by
absence); a pressure outlet is ``p' = 0`` at its face; a closed domain's
right-hand side is projected onto the range before the solve and p' is
pinned at the first FLUID cell after it, and p after the update, as the
collocated solver did. The solve is conjugate gradients, from p' = 0, to a
relative residual, a rounding floor or a reported iteration cap (ECR-003
step 1); since prompt 53 the preconditioner is a callable, since prompt 54
one V-cycle of multigrid.py's hierarchy built each correction by default or,
under pressure_preconditioner ``jacobi``, the diagonal. Built 2026-10-06; the departures
from ADR-013 D's draft are the two module-level functions and the result
type below, which the tests and the probes call directly, ZERO_SCALE,
flux_scale on the corrector, which the solver reads, and the refusal of an
open domain with a component no outlet reaches (review 37 S7), which D's
one-component check covered for closed domains only.

```
PRESSURE_SOLVER_VERSION = 3              # 1 was the weighted sweep, 2 the diagonal-
                                         # preconditioned CG, 3 the multigrid default (prompt
                                         # 54); scripts store it with saved solves and solve
                                         # again when it differs
STAGGERED_METHODS = {1: "staggered-jacobi", 2: "staggered-cg", 3: "staggered-mgcg"}
                                         # one label per version
STAGGERED_METHOD = STAGGERED_METHODS[PRESSURE_SOLVER_VERSION]   # the scripts import it;
                                         # a version without a label fails at import
RESIDUAL_FLOOR = 1e-13                   # times the flux scale F; the face arithmetic's rounding
ZERO_SCALE = 1e-30                       # guard on a zero inflow, shared with solver_staggered
PRESSURE_BLAS_THREADS = 1                # BLAS threads the CG loop may use, applied by
                                         # threadpoolctl (user_api="blas"); a constant, not a
                                         # key: docs/reports/blas_threads.md. The one
                                         # dependency ECR-003 added: threadpoolctl>=3.2
apply_operator(coefficients, x) -> [ny, nx]
    (A x)_P = a_P x_P - sum(a_nb x_nb); zero at a cell with no equation; the five
    terms in the order the probe of docs/reports/pressure_solver_ecr003.md used
jacobi_preconditioner(coefficients) -> Callable   # r -> r / a_P where a_P > 0, else 0;
                                         # the product the loop formed before prompt 53,
                                         # so the jacobi path is the same to the bit
multigrid_preconditioner(coefficients, project: bool) -> Callable
    builds multigrid.Multigrid(coefficients, apply_operator) once and returns its
    cycle; project (the closed domain) removes the result's mean over the cells
    with an equation, so the search directions stay in the range
conjugate_gradient(apply, precondition, f, rtol, floor, max_iter)
    -> ConjugateGradientResult
    preconditioned CG from zero; precondition is r -> M^-1 r, symmetric and
    positive definite on the range; stops when ||r||_2 <= max(rtol ||f||_2, floor) on the
    recursive residual, confirmed on the true residual f - A x formed once more; a
    failed confirmation restarts from the true residual and goes on; max_iter caps
    it; a zero f returns at once with no iteration; raises ValueError unless
    rtol is a number in [0, 1), floor a finite number of at least 0 and max_iter
    a positive int, a bool refused for each, or when the preconditioner's first
    result is not f's shape (until prompt 53 the second argument was the inverse
    diagonal itself, checked against f's shape); the loop
    runs inside a limit of PRESSURE_BLAS_THREADS BLAS threads and the process's
    setting is restored on leaving, by return or by exception
ConjugateGradientResult: x, iterations: int, reached_cap: bool,
    residual_norm: float,                # the true residual's 2-norm at exit
    products: int                        # every product with the operator: one per
                                         # iteration and one per true-residual check
PressureCorrector:
    __init__(mesh: Mesh, config: SimConfig, boundary: StaggeredBoundary)
        reads rho, alpha_pressure, max_pressure_iter, pressure_rtol and
        pressure_preconditioner; closed domain: raises ValueError if the cells with an equation (non-SOLID
        cells with a non-SOLID 4-neighbour) are not one connected component;
        open domain: raises ValueError if a component holds no outlet cell
        (a cell whose outlet face borrows a diagonal, p' = 0 in its row)
    needs_pin: bool, pin_cell: (j, i)    # unchanged
    flux_scale: float                    # F: rho times the inflow, or closed, rho times the
                                         # largest prescribed boundary velocity times the
                                         # longer side; the stopping rule's definition
    mass_imbalance(u, v) -> [ny, nx]     # rho [(u_e - u_w) dy + (v_n - v_s) dx], 0 at SOLID
    coefficients(a_p_u, a_p_v) -> PressureCoefficients
        a_p, a_e, a_w, a_n, a_s [ny, nx]; a_nb = rho d_face A_face with
        d = A_face / a_P where a_P > 0, zero across walls, inlets and SOLID
        faces; an outlet face borrows the nearest interior diagonal and
        sits in a_p with no neighbour
    correct(prediction: MomentumPrediction, p) -> PressureCorrection
        p' by CG from zero, preconditioned by one multigrid V-cycle per
        iteration or, under pressure_preconditioner jacobi, the diagonal, to
        ||r||_2 <= pressure_rtol ||b||_2 or ||r||_2 <= RESIDUAL_FLOOR flux_scale,
        r = b + A p' formed once more at exit and checked, or to
        max_pressure_iter iterations; closed domain: b projected onto the
        range (its mean over the cells with an equation removed), the stop
        read on the projected residual, p' pinned after; under multigrid the
        preconditioned vector's mean is removed the same way
        u [ny, nx+1], v [ny+1, nx]: u* - d (p'_(s+) - p'_(s-)) at correctable
            faces and outlet faces; walls, inlets and SOLID faces untouched
        p [ny, nx]: p + alpha_pressure p', pinned in a closed domain
        p_prime [ny, nx]
        iterations: int                  # CG iterations; renamed from sweeps
        reached_cap: bool                # stopped at max_pressure_iter
        products: int                    # the solve's operator products; the harness
                                         # counts its work in them
    The residual b + A p' is the corrected faces' mass imbalance cell by
    cell, to the face arithmetic's rounding (REQ-S04 as clarified).
```

### multigrid.py --> pressure

Additive-correction multigrid (Hutchinson and Raithby 1986) as the
preconditioner of the pressure correction's CG, ADR-013 decision 1's named
successor, built 2026-10-14 (prompt 53) as an option the solver key
pressure_preconditioner selects and made its default by prompt 54. NumPy only (decision 2); every operation is
a whole-array expression, one thread per cell on the GPU path, and the
coarsest level is one small dense product. It knows the five-point system
and nothing else of the solver: the caller hands it the coefficients and the
operator, pressure.apply_operator, which it applies on every level.

```
COARSEST_CELLS = 64        # coarsening stops at or below this many cells, or at a side of one
SMOOTHING_SWEEPS = 1       # nu of the V(nu, nu) cycle
SMOOTHER_FACTOR = 4/3      # omega = SMOOTHER_FACTOR / 2 = 2/3 on every smoothed level:
                           # every level is weakly diagonally dominant, so by Gershgorin
                           # lambda_max(D^-1 A) <= 2 and omega lambda_max <= 4/3 < 2
DOMINANCE_TOLERANCE = 1e-12  # rounding allowed in the finest system's dominance check
FivePointCoefficients      # Protocol: a_p, a_e, a_w, a_n, a_s [ny, nx]
Level: coefficients, active [ny, nx] bool, smoother = omega / a_P [ny, nx],
    weight: float                        # zero on the coarsest, which is not smoothed
Multigrid(coefficients, apply)
    levels: list[Level]                  # finest first; each coarser one groups 2x2 cells
                                         # (a lone cell at an odd edge alone)
    coarsest_inverse: [n, n]             # dense pseudo-inverse of the coarsest operator
    cycle(r) -> z [ny, nx]               # one V(nu, nu) from zero; linear, symmetric and
                                         # positive semi-definite; zero at cells with no
                                         # equation; r not modified
    raises ValueError unless the five arrays are two-dimensional and of one shape,
        and every row of the finest system has a diagonal at least the sum of its
        four couplings (weak diagonal dominance, the premise of the fixed weight)
coarsen(c, active) -> (coarse c of c's type, coarse active)
    the Galerkin product P^T A P of piecewise-constant aggregation: a coarse face
    coefficient sums the fine ones across that face; a coarse diagonal is the
    children's diagonals minus twice the couplings inside the group; a coarse cell
    has an equation when a child has one
restrict(r) -> P^T r       # the sum over each group
add_prolonged(x, z)        # x += P z in place
```

### solver_staggered.py --> tests/test_constancy.py, tests/test_conservation.py (face_velocities, the transport solver's input), time_integration (planned); scripts/benchmark.py, scripts/view_field.py, scripts/stopping_probe.py, scripts/val001_order.py, scripts/self_convergence.py

Steady SIMPLE on the staggered layout, the solver of record and since
2026-10-02 the only solver (ADR-010; the collocated solver it was built
beside is at tag collocated-final, Retired modules below). One outer
iteration is MomentumPredictor.predict then PressureCorrector.correct,
keeping the corrected fields. The public shape below is what the harness
drives through one callback; its ``--method`` label names this solver and
refuses the retired one by name. It does not define compute_residual() or
solve_timestep(); the second is Phase 4's.

```
FlowState(u, v, p, k=None, eps=None, nu_t=None, stop_reason=None, outer=0)
    # frozen; read-only C-contiguous float64 copies, finite; u [ny, nx+1],
    # v [ny+1, nx], p and k, eps, nu_t [ny, nx]; k, eps, nu_t all or none
    # (turbulent); TypeError or ValueError at construction otherwise
SteadyCheckpoint(state, residual_history, velocity_steps, viscosity_steps,
                 rule_version, pressure_cap_hits)
    # frozen; len(residual_history) == state.outer
CAP_STOP = "max_simple_iter"   # the stop_reason of a solve that reached the cap
rule_version(config) -> int | None   # StaggeredSolver.rule_version without building a solver

StaggeredSolver:
    __init__(mesh: Mesh, config: SimConfig, boundary: StaggeredBoundary)
    solve_steady(on_iteration=None, *, eddy_viscosity=None) -> tuple[ndarray, ndarray, ndarray]
        u, v, p cell-centered, each [ny, nx], float64, contiguous
        eddy_viscosity: kinematic nu_t, m^2/s, [ny, nx] float64, finite and
            non-negative in every non-SOLID cell (the transport solver's
            convention), held for the solve; the predictor gets mu_eff = mu +
            rho nu_t (SOLID cells mu). None calls predict(u, v, p) as before,
            bitwise; a field of zeros is bitwise the same. TypeError or
            ValueError before the first iteration on a malformed field
            (ECR-002 step 4); refused (ValueError) with the turbulence section,
            whose model is the solve's one source of nu_t
        With the turbulence section (ECR-002 step 6, ADR-012 D's note of
            2026-10-09) the solve is coupled: each outer iteration predicts
            with mu + rho nu_t and the wall functions' wall_mu, corrects,
            steps k and eps on the corrected faces (pseudo-time), and relaxes
            nu_t by alpha_turbulence; it starts from the inlets'
            inflow-weighted k and eps; PositivityError stops it, naming the
            outer iteration. __init__ raises ValueError when no velocity inlet
            admits air. Without the section every call is the laminar one
    turbulence_state: TurbulenceState | None   # k, eps, relaxed nu_t of the last
                                # coupled solve, read-only; None otherwise
    Built in ADR-014 step 0 (prompt 60):
    solve_steady(..., initial: FlowState | None = None,
                 resume: SteadyCheckpoint | None = None)
        initial: the warm start (ADR-014 A). The faces and p copied from the
            state, then apply_normal_velocity writes this boundary's normal
            faces over them; with the turbulence section k and eps through
            KEpsilonModel.state, so nu_t is C_mu k^2 / eps and the state's
            nu_t is not read (the start docs/reports/probe59/warm59.py made by
            patching the two allocations, bitwise); the rule starts fresh.
            None is the start from rest, bitwise (a FlowState of rest and the
            inlets' uniform k and eps gives the same bits)
        resume: continues a solve (ADR-014 D): the faces, p, k, eps and the
            relaxed nu_t as they stood, residual_history, the rule's window
            (ErrorEstimateRule.restore) and pressure_cap_hits carried on, the
            outer count continued so max_simple_iter is the total; the
            uninterrupted solve with that cap, bitwise. Refused with initial,
            for a checkpoint whose solve converged, and under another
            rule_version
        Both: TypeError if not a FlowState / SteadyCheckpoint; ValueError
            before any iteration for another grid's shapes, k and eps given to
            a laminar solve or absent from a coupled one, k or eps not positive
            in a non-SOLID cell, or (resume) nu_t negative in one
        A solve whose fields went non-finite does not raise: it returns them
            as every solve did before this step and leaves flow_state and
            steady_checkpoint None, since a diverged state is neither a warm
            start nor a resume point (a warning names it)
    flow_state: FlowState | None      # the last solve's faces (face_velocities'
        # arrays), p, k, eps, the relaxed nu_t, stop_reason and outer count;
        # None before a solve completes and after one whose fields are not
        # finite, never an earlier solve's
    steady_checkpoint: SteadyCheckpoint | None   # the last solve's flow_state with
        # residual_history, the rule's window, rule_version and pressure_cap_hits;
        # None as flow_state is
    rule_version: int | None    # read-only: 3, 5 with the turbulence section
                                # (condition (e); 4 until 2026-10-11), None under
                                # velocity_step
    on_iteration: Callable[[IterationState], None] | None
        once per outer iteration with to_cell_centers of the corrected
        faces, p, the residual, and the corrector's iteration and product counts
    residual_history: list[float]
    reference_velocity: float   # F_ref / (rho h), h = max(x[nx]/nx, y[ny]/ny)
    last_pressure_iterations: int   # CG iterations of the last correction; reset at the
                                    # start of each solve (last_pressure_sweeps until 2026-10-06)
    pressure_cap_hits: int      # corrections of the last solve that stopped at
                                # max_pressure_iter; reset per solve; a warning at the first;
                                # the harness records it in every row
    stage_seconds: dict[str, float]  # "momentum", "pressure", "correct"; no flux stage;
                                     # "turbulence" too with the section
    last_mass_imbalance: ndarray [ny, nx]   # of the returned faces; observability only
    flux_scale: float | None    # read-only; the error_estimate rule's flux scale,
                                # kg/s per unit depth, the corrector's flux_scale;
                                # None under velocity_step
    converged: bool             # met its stopping rule, not the cap; reset per solve
    stop_reason: str | None     # "velocity_step_below_tol",
                                # "error_estimate_and_continuity" or "max_simple_iter"
    face_velocities: FaceVelocities | None   # ADR-011 A (REQ-S13): None until a solve
                                # completes, then set at the end of every solve, converged
                                # or not; read-only copies of the final faces;
                                # to_cell_centers of them is the return bitwise, their
                                # mass_imbalance is last_mass_imbalance bitwise; under
                                # error_estimate that imbalance meets REQ-S04's clauses
    Planned (ADR-014 G, step 6):
        solve_transient(dt, t_end, *, initial, step_tol, step_cap, on_step=None)
            -> FlowState            # tier 2
    Walls and inlets are written once by apply_normal_velocity and never
    again; outlet faces are extrapolated (zero gradient) before every
    prediction and corrected by the corrector. Residual: largest change of
    cell-centered u, v over FLUID cells divided by reference_velocity, the
    definition every stored velocity-step row was taken under; F_ref uses
    the staggered layer's exact inlet flux.
    Stop under velocity_step (default): residual < convergence_tol, and not
    on an outer iteration whose correction reached max_pressure_iter: a
    truncated correction leaves the faces unbalanced while the step is
    small, so such a solve runs to max_simple_iter, unconverged, with
    pressure_cap_hits saying why (ADR-013 B). Under error_estimate:
    ErrorEstimateRule fed the largest change in m/s, scaled by
    get_max_boundary_velocity(), with flux_scale the corrector's, rho times
    get_total_inlet_flux() (closed: rho times that velocity times the longer
    side), never reference_velocity; a zero velocity scale raises at
    construction, naming stopping_rule; the continuity conditions refuse
    what a capped correction leaves on their own. With the turbulence
    section the rule is built with viscosity_condition=True and each update
    also takes the largest nu_t change over non-SOLID cells and
    viscosity_scale(mu / rho, nu_t, FLUID cells) of the iterate just
    computed (ADR-012 E, amended 2026-10-11).
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
    pressure_iterations: int  # CG iterations of this iteration's pressure correction
                              # (pressure_sweeps until 2026-10-06; the position is kept)
    u, v, p: ndarray      # the solver's working fields, each [ny, nx], not
                          # copies; a callback reads them and never writes
    pressure_products: int  # that correction's products with the operator, its
                            # iterations and true-residual checks; added last on
                            # 2026-10-06, so no other field moved
ImbalanceSummary:  # frozen, keyword-only, read by name; kg/s per unit depth
    worst: float         # largest absolute per-cell imbalance
    absolute_sum: float  # sum of the absolute per-cell imbalances
    signed_sum: float    # sum of the signed ones: net mass flux out of the domain
viscosity_scale(nu: float, nu_t: ndarray, fluid: ndarray) -> float
    # nu + median(nu_t[fluid]), m^2/s: (e)'s scale, which the solver forms from
    # each outer iteration's relaxed nu_t over its FLUID cells (2026-10-11)
ErrorEstimateRule:
    __init__(velocity_scale, flux_scale, iteration_error_tol, mass_imbalance_tol,
             *, viscosity_condition=False)
        the four each positive and finite and not bool, else ValueError
        flux_scale: kg/s per unit depth
        viscosity_condition: bool, else TypeError; True applies (e). Until
            2026-10-11 the keyword was nu_scale, the molecular nu plus the
            largest inlet C_mu k^2 / eps, fixed per solve (version 4)
    version -> int   # 3 without (e), 5 with it; the solver reports it
    update(step: float, imbalance: Callable[[], ImbalanceSummary],
           viscosity_step=None, nu_scale=None) -> bool
        step: largest velocity change of one outer iteration, m/s
        imbalance(): the current field's ImbalanceSummary, from one
            evaluation; called only when (a) holds
        (a) step rho_hat / (1 - rho_hat) / velocity_scale < iteration_error_tol,
            rho_hat = exp of the least-squares slope of log(step) over the
            last RATE_WINDOW steps
        (b) worst < mass_imbalance_tol  (ECR-001 criterion 6, per cell)
        (c) absolute_sum / flux_scale < iteration_error_tol  (bounds the flux drift)
        (d) |signed_sum| < mass_imbalance_tol  (criterion 6, domain sum)
        (e) with viscosity_condition: viscosity_step rho_hat_nu / (1 - rho_hat_nu)
            / nu_scale < iteration_error_tol, the same fit on the largest nu_t
            change of each outer iteration, over the scale passed with that
            step; viscosity_step and nu_scale given exactly when (e) is on,
            else ValueError; nu_scale positive and finite, else ValueError
            before anything is recorded
        True when all four hold, and (e) when it is on; imbalance() is
            called only when (a) and (e) hold
    estimate_history: list[float]  # (a)'s left side per step; inf when none
    viscosity_estimate_history: list[float]  # (e)'s, likewise; empty with (e) off
    viscosity_scale_history: list[float]  # the nu_scale of each step; empty with (e) off
    window() -> (tuple[float, ...], tuple[float, ...])   # ADR-014 step 0: the last
        # RATE_WINDOW velocity and viscosity steps, oldest first; the second
        # empty with (e) off. The steady checkpoint carries them
    restore(steps, viscosity_steps=()) -> None   # seeds a rule that has taken no
        # step, so its stops continue another's bitwise; ValueError on a rule
        # that has stepped, a window longer than RATE_WINDOW, a step not a
        # finite non-negative float (bool refused), or viscosity steps that do
        # not match (e) and the velocity steps
RATE_WINDOW = 100  # module constant, not configuration
RULE_VERSION_WITHOUT_E = 3, RULE_VERSION_WITH_E = 5   # a rule's version (4 was (e) on
    # the inlet scale); RULE_VERSION was retired in ECR-002 step 6: scripts read the
    # version through the solver
```

### particles.py --> boundary_concentration, scenarios, solver_transport

```
ParticlePhysics:
    __init__(config: SimConfig)
    settling_velocity(size_class: int) -> float
    diffusion_coeff(size_class: int) -> float
    cunningham_correction(size_class: int) -> float
    deposition_velocity(size_class: int, surface: str) -> float
    hepa_efficiency(size_class: int) -> float
    # every method: TypeError for a bool or np.bool_ size_class, IndexError outside
    # the configured classes
```

### boundary_concentration.py --> scenarios, solver_transport

The registry's second reader (REQ-S12.1; ADR-011 E, amended 2026-10-03).
Derives the scalar condition at every face from the velocity type the shared
coverage gives, on the coordinates and SOLID mask staggered.edge_cell_inputs
derives for both layers, with the optional segment keys: `concentration` (one value per class,
velocity_inlet), `hepa_filtered` (bool, velocity_inlet),
`deposition_surface` (floor, ceiling, wall or none, wall; the edge decides by
default). Faces between a non-SOLID and a SOLID cell are walls of the
orientation the normal gives (ADR-011 D). Reads no concentration field and
writes none; imposes nothing, hands the solver data.

```
SURFACE_NONE = 0, SURFACE_FLOOR = 1, SURFACE_CEILING = 2, SURFACE_WALL = 3
SURFACE_CODES: dict[str, int]
EDGE_SURFACES: bottom floor, top ceiling, left and right wall   # unless deposition_surface overrides
ConcentrationFaces:                           # frozen, eq=False; every array read-only, _u [ny, nx+1], _v [ny+1, nx]
    inflow_u, inflow_v: float64              # concentration an inward flux carries; zero except at inlets
    deposition_u, deposition_v: float64      # deposition velocity at wall faces, domain and SOLID-adjacent;
                                             # zero elsewhere and on a "none" surface
    surface_u, surface_v: int32              # the surface code each depositing face is booked to
    settling_v: bool                         # horizontal faces between two non-SOLID cells; no settling_u
    # ADR-014 J (step 0b), keyword-only, defaults meaning no vent
    vent_u, vent_v: bool | None              # the vent faces (faces of a SOLID cell that carry a flux)
    recirculation: tuple[Recirculation, ...] # the discharges that return their intake's air
Recirculation:                               # frozen, eq=False; prompt 61's amendment to ADR-014 J
    intake_u, intake_v, discharge_u, discharge_v: bool   # face masks of the two vents
    passed: float                            # 1 - hepa_efficiency(k) when the discharge is hepa_filtered, else 1
ConcentrationBoundary:
    __init__(mesh: Mesh, config: SimConfig, physics: ParticlePhysics, registry: BoundaryRegistry)
        ValueError if physics and config disagree on the class count
    faces_for(size_class) -> ConcentrationFaces   # IndexError outside the configured classes;
                                                  # TypeError for a bool or non-int
        inlet (nonzero normal velocity): concentration[k], times 1 - hepa_efficiency(k)
            when hepa_filtered; 0 with no key
        zero-normal velocity_inlet (a lid): a wall with the edge's surface, no inflow
        outlet: nothing carried in (reversed flow brings clean air)
        wall: deposition_velocity(k, surface); SOLID edge cell: nothing
        obstacle faces: floor above, ceiling below, wall beside
        vents (from boundary_vents.vent_runs): a discharge's faces are inlet faces
            (concentration[k], HEPA-reduced, 0 with no key or with recirculate_from), an
            exhaust's outlet faces; neither deposits (SURFACE_NONE on every vent face)
```

### scalar_scheme.py --> solver_transport, turbulence

Built (prompt 35, ECR-002 step 1) by moving the transport solver's face
value, advective flux and implicit solve out of TransportSolver into module
functions, so that the k-epsilon model takes the same scheme by import without
importing the particle solver (a departure from ADR-012 I, which named
solver_transport). The transport gate's fields are bitwise those of main
(results/builder35/). The implicit solve gained what turbulence needs:
per-face conductances, a cell sink in the diagonal, a step per cell, and an
optional mask of cells held at their C*, the k-epsilon wall cells' eps; with
the mask None the transport gate's hashes are unchanged. ECR-002 step 7
(prompt 51) gave it a method: Jacobi, the default and the transport solver's,
or diagonal-preconditioned conjugate gradients, the k-epsilon model's, with a
fallback to Jacobi from C* when a CG solve reaches its cap or leaves a value at
or below zero in a cell it solves for (Alex, 2026-10-12). The Jacobi path's
bits are unchanged (a 240-system battery and the laminar hashes against main,
results/builder51/; docs/reports/turbulence_cg_solve.md).

```
Axis:                     # frozen; one advection direction, that axis last
    nodes, faces, left, solid_ext
mesh_axes(mesh) -> (Axis, Axis)
    the x axis for [ny, nx] fields and the y axis for transposed [nx, ny] fields
limited_face_values(c_up, c_c, c_d, quick) -> ndarray
    c_c + psi(r) (c_d - c_c) / 2, psi = max(0, min(2r, (1 + 3r) / 4, psi_quick, 2)),
    r = (c_c - c_up) / (c_d - c_c), psi_quick = 2 (quick - c_c) / (c_d - c_c); c_c where
    c_d == c_c
advective_flux(c, flux, inflow, axis, upwind, vents=None) -> ndarray
    c [nt, ns], flux and inflow [nt, ns+1]; flux times the face value: the inflow
    value at a domain face whose flux enters, the adjacent cell at one it leaves,
    the limited QUICK value (C_C with upwind) inside; a far node in a SOLID cell
    reads as the upstream value. The caller zeroes the flux of every face that
    carries none. vents (ADR-014 J), bool [nt, ns+1]: interior faces of a SOLID
    cell that carry a flux; the inflow value where it enters the room from the
    SOLID side, the room cell's value where it leaves, no stencil. None is the
    arithmetic without the argument, bit for bit.
JACOBI = "jacobi", CONJUGATE_GRADIENT = "cg", IMPLICIT_METHODS = (JACOBI, CONJUGATE_GRADIENT)
SCALAR_BLAS_THREADS = 1    # the BLAS pool the CG loop runs in, as pressure.py's
ImplicitResult:           # frozen
    field: ndarray, method: str, iterations: int, converged: bool,
    fallbacks: int, fallback_sweeps: int
    # iterations: Jacobi sweeps, or CG iterations (true-residual checks not
    # counted); fallbacks 1 when a CG solve fell back to Jacobi, its sweeps in
    # fallback_sweeps; converged is the returned field's (the fallback's if any)
implicit_step(c_star, volume, dt, g_u, g_v, diagonal, solid, tol, max_iter,
              held=None, method=JACOBI) -> ImplicitResult
    (V / dt + sum_f G_f + D_P) C_P - sum_f G_f C_N = (V / dt) C*, until the
    largest residual over the non-SOLID, unheld cells is at most tol times the
    largest |b| over all cells, or max_iter; dt a float or [ny, nx]; g_u
    [ny, nx+1] and g_v [ny+1, nx] the face conductances, zero where nothing
    diffuses; diagonal [ny, nx] the non-negative sink; SOLID cells held at zero;
    held [ny, nx] bool or None, cells kept at their C* exactly (a Dirichlet row,
    the limit of a dominant diagonal) whose neighbours read it. ValueError on a
    method not in IMPLICIT_METHODS. With no conductance and no sink it returns C*
    itself, zero iterations, under either method.
    jacobi: sweeps from C*; non-negative C* gives a non-negative result at any
    step and at the cap.
    cg: from C*, SOLID and held cells not unknowns; the stop tested on the
    recursive residual and confirmed on the true one, restarting from it if the
    confirmation fails (pressure.py's loop is not shared: its start and stop
    would have to become parameters). A solve that reaches max_iter, or ends
    with a value at or below zero in a cell it solves for, is discarded and
    solved by Jacobi from C* with the same tol and max_iter: the result is then
    the jacobi path's, bit for bit, so non-negative C* again gives a
    non-negative result. For fields positive in every solved cell (k and eps).
```

### turbulence.py --> solver_staggered; tests/test_turbulence.py, tests/test_decaying_turbulence.py, tests/test_wall_functions.py, tests/test_coupled_solve.py, tests/test_turbulent_channel.py

Built in ECR-002 step 1 (prompt 35) from ADR-012 A and C: k and eps advanced on
a prescribed face field, no coupling to momentum. Per unit density, so nu_t is
kinematic (REQ-S14's mu_t is rho nu_t). One step: advection by the shared
scheme with the configured face value, forward Euler (plain upwind by default,
the UMIST-limited QUICK value under `advection_scheme: umist`; ADR-012 C's note
of 2026-10-10, prompt 48); explicit growth from
the previous iterate (P = nu_t S^2, C_1 (eps / k) P, and RNG's R where it is
negative); one implicit solve per quantity with diffusion and the decay (eps / k;
C_2 eps / k and RNG's positive R over eps) in the diagonal, from the previous
iterate. Nothing is clipped. Strain: du/dx and dv/dy face differences, du/dy and
dv/dx at the four corners averaged to the centre, the edge's tangential velocity
at a domain edge and zero at an obstacle face read at the corner, so a no-slip
wall gives the full shear; a face beside a SOLID cell carries no air and reads as
zero in the face differences too, as in the rate and the advection (prompt 35b). The face diffusivity is the distance-weighted harmonic
mean of nu + nu_t / sigma across the face, ADR-012 F's rule for the transport
diffusivity, so k, eps and the particles use one rule. Constants sourced in
results/builder35/constants.md; RNG's C_mu 0.0845 by Alex's decision of
2026-10-05, the preprint printing "C_mu ~ 0.085".

Departures from ADR-012 I's draft:
1. The shared scheme is `scalar_scheme.py`, not functions imported from
   `solver_transport.py`: a flow-side model importing the particle solver would
   invert the layers.
2. The boundary values are a TurbulenceConditions the caller builds and passes
   to `step` every step, not a StaggeredBoundary given at construction: step 6's
   wall-function values change every outer iteration. So the constructor takes
   `(mesh, config)` and `initial` takes the uniform values it starts from.
3. The conditions carry the edges' tangential velocities, which a moving wall
   needs for its corner shear; step 6 fills them from the staggered boundary
   layer, which owns them.
4. The face diffusivity is ADR-012 F's harmonic mean, which the draft did not
   state. The eps of a wall cell is held exactly, by a mask in implicit_step,
   the limit of ADR-012 C's dominant diagonal.
5. `wall_viscosity` is not a KEpsilonModel method. Step 6 (prompt 45) built the
   wall functions as `TurbulenceBoundary`, constructed from the mesh, the
   configuration and the staggered boundary layer, which returns the wall
   viscosity in `wall_mu`'s layout and builds each step's conditions (ADR-012
   I's note of 2026-10-09).

```
VariantConstants:          # frozen: c_mu, c_1, c_2, sigma_k, sigma_e, eta_0, beta
VARIANTS = {"standard": (0.09, 1.44, 1.92, 1.0, 1.3), "rng": (0.0845, 1.42, 1.68, 0.7194,
            0.7194, eta_0 4.38, beta 0.012)}   # module constants, not configuration
PositivityError(RuntimeError):   # .minimum, the least value over non-SOLID cells
TurbulenceState:           # frozen, eq=False; [ny, nx] float64, read-only, SOLID zero
    k, eps: positive and finite in every non-SOLID cell
    nu_t: kinematic, m^2/s, non-negative and finite in every non-SOLID cell;
        state() builds C_mu k^2 / eps, a caller may build a state with another
        such field (step 6's under-relaxed nu_t)
TurbulenceConditions:      # frozen, eq=False; built by the caller for each step
    inflow_k_u, inflow_eps_u [ny, nx+1]; inflow_k_v, inflow_eps_v [ny+1, nx]   # float64,
        read on the domain faces whose flux enters; positive there
    tangential_bottom, tangential_top [nx+1]; tangential_left, tangential_right [ny+1]
        # the edge's own tangential velocity at the corners along it, m/s
    eps_held, production_given: [ny, nx] bool, non-SOLID cells only
    eps_wall: [ny, nx], positive where held; production: [ny, nx] m^2/s^3, >= 0 where given
    uniform(mesh, k, eps) -> TurbulenceConditions   # edges at rest, nothing held or given
StepTerms:                 # frozen, eq=False; per cell, per unit density
    production, rng_r, growth_k, growth_eps, decay_k, decay_eps
KEpsilonModel:
    __init__(mesh, config)   # ValueError without config.turbulence; reads its
                             # advection_scheme for step 1's face value
    constants: VariantConstants
    initial(k: float, eps: float) -> TurbulenceState   # uniform; TypeError on a bool,
                                                       # ValueError unless positive, finite
    state(k, eps) -> TurbulenceState   # checked copies with nu_t; ValueError on a shape or a
                                       # value not positive and finite in a non-SOLID cell
    eddy_viscosity(k, eps) -> ndarray  # C_mu k^2 / eps, kinematic, SOLID zero
    strain_squared(faces, conditions) -> ndarray       # 2 S_ij S_ij, 1/s^2
    terms(state, faces, conditions) -> StepTerms       # what a step adds and decays
    pseudo_time_step(faces) -> ndarray
        cfl / (max(|u_w|, |u_e|) / dx + max(|v_s|, |v_n|) / dy) per cell over the faces with
        a flux; a cell at rest and every SOLID cell take the largest finite value; ValueError
        when no non-SOLID cell moves (the step has no value there; pass a true-time dt)
    step(state, faces: FaceVelocities, conditions: TurbulenceConditions, dt=None)
            -> TurbulenceState
        dt None: the pseudo-time step; a float: one true-time step for every cell, ValueError
        above cfl / the largest cell rate, TypeError on a bool. Every check before any
        arithmetic: the state's shapes, k and eps positive and finite, nu_t non-negative
        and finite in non-SOLID cells and all three zero in SOLID ones (ValueError);
        the faces' shapes and finiteness; the conditions. Raises PositivityError unless k and eps are positive and finite at
        every non-SOLID cell after the step (REQ-S15).
    last_iterations: tuple[int, int]  # CG iterations of the last k and eps solves (prompt 51)
    last_fallbacks: tuple[int, int]   # 1 where that solve fell back to Jacobi (a warning)
    solves_converged: bool         # both met tol within max_iter, the fallback's where there
                                   # was one (a warning otherwise)
# ECR-002 step 6 (ADR-012 B and C): the wall functions and the coupled solve's boundary data
KAPPA = 0.41, E_WALL = 9.793           # module constants (Launder and Spalding 1974)
log_law_floor(kappa, e) -> float       # the root of kappa y = ln(E y) above 1 / kappa
Y_STAR_FLOOR = log_law_floor(KAPPA, E_WALL)   # 11.528, computed, never typed
wall_viscosity(k_p, y_p, c_mu, rho, nu) -> ndarray
    rho C_mu^(1/4) k_p^(1/2) kappa y_p / ln(E max(y*, y*_0)), Pa s; mu exactly at
    y* = y*_0, mu y* / y*_0 below it
inflow_values(intensity, normal_speed, dissipation_length) -> (k, eps)
    k = 1.5 (I |u_n|)^2, eps = k^(3/2) / l_e (no C_mu^(3/4))
TurbulenceBoundary:
    __init__(mesh, config, boundary: StaggeredBoundary)   # ValueError without the section
    # ADR-014 J (step 0b): a vent is not a wall. wall_faces drops the stencil faces over a
    # vent's span (StaggeredBoundary.vent_faces' tangential masks), wall_cells drops a vent
    # face as a cell's wall side, a discharge's faces carry inflow_values in the conditions'
    # inflow arrays and count as inlet faces for initial_values, and an exhaust's carry the
    # room cell's values (nothing enters). KEpsilonModel reads the vents from
    # boundary_vents.vent_runs: their faces carry flux in the advection, the rate and
    # the strain; no diffusion crosses them.
    wall_faces: dict["u" | "v", ndarray]   # bool [ny+1, nx+1], read-only: the faces
        that take a wall function, every wall (zero normal velocity, not an
        outlet) beside an unknown and every obstacle face
    wall_cells: ndarray                    # bool [ny, nx], read-only
    wall_viscosity(k, elsewhere) -> dict   # wall_mu: mu_w on wall_faces, y_P the
        unknown's half cell, k_P the mean of its two cells; elsewhere's values
        on every other face
    conditions(state, faces) -> TurbulenceConditions
        inflow k and eps per inlet face, the adjacent cell's elsewhere; the
        edges' tangential velocities (the adjacent face's at a pressure
        outlet); eps held at C_mu^(3/4) k^(3/2) / (kappa y_P) and the
        production (tau_w / rho) u_k / (kappa y_P) in every wall cell, a
        cell with several wall faces taking their mean
    initial_values() -> (k, eps)           # inflow-weighted means over the inlet faces
        raises ValueError when no velocity inlet admits air
        (largest_inlet_eddy_viscosity, condition (e)'s inlet scale until
        2026-10-11, removed 2026-10-16 with its tests, ADR-014 I)
```

### solver_transport.py --> time_integration, monitor (planned, Phases 4 and 5); the Phase 3 tests

Built (PR 32) from ADR-011 B, C, D and F. Finite volume on the cell-centred
field, advected by the face velocities of REQ-S13: the flux through a face
is the stored face velocity times the face length times a face
concentration, QUICK's quadratic bounded by the UMIST limiter, with C_C where
the downstream difference is zero. The face value, the advective flux and the
implicit solve are scalar_scheme.py's (prompt 35); this module hands them the
class's fluxes, conductances and deposition sink. Forward
Euler advection at cfl_number; backward Euler diffusion and deposition by
Jacobi to diffusion_tol within max_diffusion_iter. Non-negative and bounded
(REQ-T12). An inflow face carries the inlet value as Leonard's boundary node
at the face; an outflow face the upwind cell; a face between a non-SOLID and
a SOLID cell, and a domain face behind a SOLID edge cell, carries no
advective flux whatever velocity it holds; a far-upstream node in a SOLID
cell reads as the upstream value. Settling is subtracted on the faces
settling_v marks and nowhere else; v_ext is added on the interior faces
between two non-SOLID cells, both components. Given an eddy viscosity field,
the diffusivity of an interior face between two non-SOLID cells is the
Brownian coefficient plus nu_t / Sc_t, the face value of nu_t the
distance-weighted harmonic mean of the two cell values, zero when either is
(REQ-T13); no other face and no other term changes. Measured in
tests/test_diffusion.py (VAL-003), tests/test_advection.py (VAL-004),
tests/test_constancy.py (VAL-012), tests/test_smith_hutton.py (VAL-013),
tests/test_conservation.py (VAL-007) and tests/test_sealed_box.py (VAL-014).

```
OBSTACLE = "obstacle"
DEPOSIT_SURFACES = ("floor", "ceiling", "wall", "obstacle")   # MassBudget.deposited's keys

ParticleProperties(Protocol):   # what the solver reads of a particle model
    settling_velocity(size_class) -> float
    diffusion_coeff(size_class) -> float
ScalarConditions(Protocol):     # what the solver reads of a boundary layer
    faces_for(size_class) -> ConcentrationFaces

TransportSolver:
    __init__(mesh, config, physics: ParticlePhysics | ParticleProperties,
             boundary: ConcentrationBoundary | ScalarConditions)
        ValueError without config.transport, or when a class's ConcentrationFaces
        is not shaped for the mesh, not float64 / int32 / bool as the
        boundary_concentration contract names, or holds a value that is not
        finite or a negative inflow or deposition velocity. Reads
        physics.settling_velocity(k), physics.diffusion_coeff(k) and
        boundary.faces_for(k) for every class at construction, and nothing else
        of either; ParticlePhysics and ConcentrationBoundary are the production
        types, and validation/transport_cases.py hands stand-ins.
    stable_dt(faces: FaceVelocities, size_class, v_ext=None) -> float
        cfl_number / max over non-SOLID cells of (max(|u_w|, |u_e|) / dx_cell
        + max(|v_s|, |v_n|) / dy_cell), the face velocities masked on faces that
        carry no flux and carrying the class's settling increment and v_ext;
        inf when nothing moves. TypeError for a bool or non-int size_class,
        IndexError outside the configured classes, ValueError on a shape or a
        face value that is not finite.
    solve_timestep(C_k, faces, size_class, dt, v_ext=None, sources=None, *,
                   eddy_viscosity=None) -> ndarray
        C_k [ny, nx] not modified; v_ext FaceVelocities-shaped per-class drift or
        None (zero; REQ-T06, ADR-007); sources [ny, nx] non-negative rate of the
        class or None (zero), added as sources * dt after the advection update
        and before the implicit solve, sum(sources * V) * dt booked;
        eddy_viscosity (keyword-only) [ny, nx] float64 nu_t in m^2/s, kinematic,
        or None (the laminar path, bitwise; REQ-S16). With it, every interior
        face between two non-SOLID cells diffuses with D_B + nu_f / Sc_t, where
        nu_f = (d_P + d_E) / (d_P / nu_P + d_E / nu_E), d_P and d_E the
        distances from the face to the two cell centres, and zero when either
        cell's value is zero; Sc_t is config.transport.turbulent_schmidt. The
        conductance is (D_B + nu_f / Sc_t) A_f / d_f, formed once per call. Every
        other face (domain faces, faces next to a SOLID cell) keeps conductance
        zero, so a wall face's flux and deposition keep the Brownian rule
        (ADR-012 decision 7); the advection, settling, deposition, sources,
        budget and stable_dt do not read it. Values in SOLID cells are not read;
        TypeError for a bool or non-int size_class or a bool dt (a bool, an
        np.bool_ or a 0-d bool array), or an eddy_viscosity that is not a
        float64 ndarray; IndexError
        outside the configured classes; ValueError if dt is not finite and
        positive or exceeds stable_dt, a shape differs, C_k, faces, v_ext or
        sources holds a value that is not finite, a source is negative or
        sits in a SOLID cell, an eddy viscosity is negative or not finite in a
        non-SOLID cell, or one is given with transport.turbulent_schmidt absent.
        Every check runs before any arithmetic, so a refused call leaves the
        budget untouched.
        returns the new field, [ny, nx], float64, contiguous, SOLID cells zero;
        updates budget[size_class] from the fluxes and sources applied
    budget: list[MassBudget]        # one per class, written here only
    last_diffusion_sweeps: int      # Jacobi sweeps of the last implicit solve
    diffusion_converged: bool       # whether it met diffusion_tol within the cap
                                    # (a warning naming the cap is logged when it
                                    # did not). diffusion_tol is relative: the
                                    # solve stops when the largest cell residual
                                    # of the implicit system is below it times
                                    # the largest right-hand side (V / dt) C*.

MassBudget:                         # dataclass; particles per metre of depth
    initial: float | None           # in_domain of the first field stepped
    inflow, outflow, source: float  # cumulative; inflow and outflow count the vent faces
                                    # (ADR-014 J): a discharge in, an exhaust or intake out
    deposited: dict[str, float]     # by floor, ceiling, wall, obstacle
    current: float | None           # in_domain of the field the last step returned
    in_domain(C_k, mesh) -> float   # static; sum(C V) over non-SOLID cells
    residual() -> float             # initial + inflow + source - outflow - deposited - current
    relative() -> float             # residual over initial + inflow + source;
                                    # both ValueError before the first step, and
                                    # relative() ValueError when nothing was
                                    # supplied

FieldHistory:                       # output contract for Phase 7's animation
    __init__(every: int)            # every = config.output_interval; TypeError on a
                                    # non-int, ValueError when not positive
    every: int
    record(step: int, t: float, fields: dict[int, ndarray]) -> None
                                    # called by time_integration once per step; copies
                                    # every class field when step % every == 0
    frames: list[tuple[int, float, dict[int, ndarray]]]
    save(path) -> None              # npz: steps [n] int64, times [n] float64, C_<k>
                                    # [n, ny, nx] per class; ValueError with no frames
                                    # or when frames carry different classes
```

### monitor.py --> time_integration

```
AlertMonitor:
    update(C_fields: dict[int, ndarray], t: float) -> None
    get_alerts() -> list[Alert]
    get_detection_latency(event: str) -> dict[str, float]
```

### scenarios.py --> time_integration (planned, Phase 4); tests/test_scenarios.py

Built at ADR-014 step 2 (prompt 62), from ADR-014 A, B and G. The contract
replaces the earlier draft's get_active_sources(t) and get_bc_modifications(t):
a boundary modification is an interval's flow state, not a per-call mapping,
and the sources at a time are one rate array per class. Nothing here solves,
marches or writes.

```
ScenarioManager(config, mesh):              # ValueError without a scenario, on another grid's
                                            # mesh, for a source box with no non-SOLID centre,
                                            # or for an interval's vent that boundary_vents.
                                            # vent_runs refuses on the mesh (review 62 S6);
                                            # logs the interval and flow state counts (INFO)
    intervals() -> list[Interval]          # [t_start, t_end), covering [0, t_end); new objects
        Interval: t_start, t_end, fingerprint, mapping, config (SimConfig), events (names)
    flow_states() -> dict[str, dict]       # fingerprint -> the first interval's mapping with it
    n_flow_states -> int                   # the run's flow cost in solves, before any is run
    sources_at(t) -> dict[int, ndarray]    # class -> [ny, nx] rate, summed over active sources;
                                           # rate over the box's non-SOLID cells, an emission
                                           # divided by their area; zero elsewhere
    conditions_for(interval) -> ConcentrationBoundary   # from the interval's configuration
    initial_fields() -> dict[int, ndarray] # the initial event's value per class on non-SOLID
                                           # cells, zero in SOLID cells; zero without one
FLOW_KEYS, TRANSPORT_KEYS                  # the segment keys the fingerprint reads, and not
flow_segments(config) -> list[tuple]       # FLOW_KEYS per segment, contiguous pieces joined
flow_fingerprint(config) -> str            # SHA-256 of flow_segments
```

### time_integration.py --> nothing in src/ (planned, Phase 4)

Planned (ADR-014 A, D and G; built at steps 4 and 6). Orchestrates both tiers
and discretises nothing.

```
run(config, *, restart=None, on_step=None) -> RunRecord
    tier 1: the intervals, the flow states (cached by fingerprint, warm-started
            with solve_steady(initial=...)), the transport march at one step for
            every class, stable_dt recomputed at each switch
    tier 2: StaggeredSolver.solve_transient with the recorders of ADR-014 D
```

### io_manager.py --> time_integration (planned, Phase 4); tests/test_io_manager.py

ADR-014 D. Step 0 (prompt 60) built the steady checkpoint's group alone;
step 3 inherits the group's layout and the two functions and builds the rest
of the run file around them (planned, below).

```
write_steady_checkpoint(path, checkpoint: SteadyCheckpoint) -> None
    # h5py, mode "a": the file's other groups kept, /steady_checkpoint/ replaced;
    # datasets u, v, p (and k, eps, nu_t when turbulent) and residual_history,
    # velocity_steps, viscosity_steps, all "<f8"; attributes outer and
    # pressure_cap_hits, and stop_reason and rule_version when not None;
    # TypeError if not a SteadyCheckpoint
read_steady_checkpoint(path) -> SteadyCheckpoint
    # what was written, bitwise; KeyError naming the group when absent;
    # ValueError when it holds some but not all of k, eps and nu_t
STEADY_CHECKPOINT = "steady_checkpoint"   # the group's name
Planned (ADR-014 G, step 3): RunFile(path).write_config / write_flow_state /
    write_frames / checkpoint / read_checkpoint; /config, /flow/<fingerprint>/,
    /transport/, /checkpoint/ beside /steady_checkpoint/; the configuration hash
    a resume must match
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
- Incompressible Navier-Stokes (SIMPLE algorithm), laminar or with the k-epsilon turbulence model (ADR-012)
- Finite Volume on structured rectangular grid
- Staircase representation of internal obstacles
- Five-class Eulerian particle transport with settling and diffusion
- Cunningham slip correction for sub-micron particles
- Four contamination scenarios: door seal leak, HEPA filter breach, equipment dust release, and the clean-out (recovery), added 2026-10-16 (Alex's decision 3 of that date, ADR-014; recorded under the scope change process with PROJECT_PLAN's Phase 4 rows)
- Alert monitoring with configurable sensors and ISO 14644 thresholds
- Sensor placement comparison across scenarios
- Animated visualizations per size class
- Hybrid Python/CUDA C++ implementation (NumPy reference solver for validation, CUDA for production)
- Centralized YAML configuration
- Phase-gated development with validation tests before code

### Out of Scope (v1)

- 3D simulation (architecture supports extension, not implemented)
- LES, DNS, and RANS models other than k-epsilon
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

ADR-008, ADR-010, ADR-011, ADR-012, ADR-013 and ADR-014 are files in `docs/ADR/`; the others are in the development plan document. Summary reference:

| ADR | Decision | Key Rationale |
|-----|----------|---------------|
| ADR-001 | Finite Volume discretization | Industry standard (Fluent, OpenFOAM). Conservation built into the method. |
| ADR-002 | 2D vertical cross-section | Captures gravity, HVAC flow, stratification. 3D adds complexity without proportional insight. |
| ADR-003 | Structured grid with staggered layout, non-uniform spacing, staircase boundaries | Clean room geometry is rectangular. Staggered arrangement gives natural pressure-velocity coupling. Non-uniform spacing enables efficient near-wall resolution. |
| ADR-004 | Laminar flow assumption (Superseded by ADR-012) | Clean-room 'laminar flow' means unidirectional supply; at Re 89,500 the room's flow is turbulent (ECR-002). |
| ADR-005 | Hybrid Python/CUDA C++ | Python orchestration with pure NumPy reference solver for validation, CUDA C++ kernels via pybind11 for accelerated production runs. NumPy solver is the primary implementation through Phase 3 validation. CUDA acceleration is a separate deliverable after solver physics are validated. |
| ADR-006 | Five particle size classes | Spans diffusion-dominated to settling-dominated regimes. Maps to ISO 14644. |
| ADR-007 | Ionizer modeling deferred | Scope risk. Extension point (v_ext) preserved in transport solver interface. |
| ADR-008 | Collocated ghost cell wall treatment (Superseded by ADR-010) | O(h) wall accuracy accepted for simplicity. Industry-standard approach. VAL-001 relaxed from 1% to 2.5%. |
| ADR-010 | Solver Architecture V2: Staggered Grid, Non-Uniform Mesh, QUICK Advection | Staggered MAC grid removes the collocated ghost-cell wall leak; the closed-domain sum of the mass imbalance is exact to rounding, and the stopping rule holds the per-cell imbalance and the signed domain sum below 1e-10 at every validation stop (ECR-001 acceptance criterion 6, as ADR-010's planned-against-built table records it). On the open channel condition (d) was met at a zero crossing of a decaying oscillation of the net outflow, not where it had settled, which Alex accepted on 2026-10-01; the oscillation was the weighted Jacobi correction's, and under conjugate gradients (d) holds from the first outer iteration (ECR-003 step 2, `docs/reports/ecr003_step2_baseline.md`, section 6). The VAL-002 v error once cited here was measured against a corrupted Ghia table (ECR-001 erratum). Non-uniform mesh clusters cells toward the walls; on the VAL-001 stencil it costs accuracy at a fixed cell count. QUICK by deferred correction; weighted Jacobi (superseded by ADR-013); the error_estimate stopping rule. Records what ECR-001 built, with planned against built. |
| ADR-011 | Transport Solver Architecture: Face-Advected Cell-Centred Concentration (Accepted at merge, PR 30) | Concentration at cell centres, advected by the staggered face velocities the NS solver exposes (REQ-S13), so a uniform haze drifts only at the rate the stopping rule bounds (REQ-T11, VAL-012). QUICK's face value bounded by the UMIST limiter under forward Euler at Courant number at most 1/2, non-negative and bounded (REQ-T12, VAL-013); implicit diffusion and deposition. Settling as a face increment between fluid cells only, every solid-facing or domain horizontal face taking the deposition velocity whole (VAL-014). One mass budget, sources included, for every conservation claim. VAL-004 in two rows with measured thresholds. Clean supply by default, no recirculation. Ten decisions of 2026-10-03 listed at its top; Accepted (built) with planned against built since the Phase 3 gate (2026-10-15, prompt 58). |
| ADR-012 | Turbulence Model: k-epsilon on the Staggered Solver (Accepted 2026-10-04, with ECR-002; Accepted (built) 2026-10-15, ECR-002 closed, with planned against built) | The product room runs at Re 89,500 on its height and the laminar solver does not converge on it (`docs/reports/product_case_reynolds.md`). k-epsilon, both variants built, RNG for the product and standard for the published comparisons; scalable wall functions on walls and obstacle faces; k and eps by the transport scheme in pseudo-time, positive by construction; the eddy viscosity in the momentum conductances by Patankar's face rule, with the stress terms constant viscosity let drop as an explicit source; a fifth stopping condition on the eddy viscosity; particle diffusivity Brownian plus nu_t / Sc_t. The outlets gain a condition for entering air and the hood a fixed-flow exhaust segment (REQ-S18). The pressure solve moves to ECR-003. Nine decisions of 2026-10-04 listed at its top, the ninth a probe of convergence before the build (ECR-002 step 0); gains planned against built when ECR-002 closes. |
| ADR-013 | Pressure Correction Solver: Jacobi-Preconditioned Conjugate Gradients (Accepted 2026-10-06, with ECR-003; built, with planned against built, ECR-003 closed 2026-10-07) | Weighted Jacobi needs 28,000 to 378,000 sweeps per correction on the product mesh to reach a relative residual of 1e-3 to 3e-3, and a steady product solve would take 10 to 42 hours; Jacobi-preconditioned CG reaches 1e-8 in about 860 iterations, 0.16 s, keeps the per-cell update the GPU path wants with three reductions per iteration, needs no new dependency and converges on the closed cavity with b projected onto the range (`docs/reports/pressure_solver_ecr003.md`). The stop is the relative residual `pressure_rtol`, default 1e-8, with a rounding floor and the iteration cap reported. Supersedes ADR-010's weighted sweep. Its named successor, multigrid preconditioning of the same loop, built 2026-10-14 as an option, the default unchanged (prompt 53, `docs/reports/pressure_multigrid.md`). |
| ADR-014 | Scenarios and Time Integration: Piecewise-Steady Flow, Time-Accurate Transport, and a Transient Solve for the Cylinder (Proposed 2026-10-16, prompt 59, before the Phase 4 build; Accepted 2026-10-16 by Alex, with the equipment vents of its section J as an amendment) | Two tiers (Alex, 2026-10-16): the room's flow is a sequence of steady states, one per distinct set of velocity boundaries, each solved once per run and warm-started, with the particles marched through time on it; the square cylinder (VAL-022, proposed) runs a true transient, an implicit step with SIMPLE outer iterations inside it. Short events as their steady extreme; a fourth reference scenario, the clean-out. The transport march, not the flow, sets a scenario's cost, and a warm start saves little of a re-solve because the stop is set by the slow mode's geometric decay (`docs/reports/probe59/tables59.md`). Twelve decisions put to Alex at its top: the re-solve trigger and cache, the scenarios' Courant number, BDF2 with SIMPLE per step and a velocity-step criterion, the door as a two-way exchange, the scenario schema with `base:` inheritance (REQ-C03), within spec as every sensor below threshold with the ISO 14644-3 100:1 time reported, one HDF5 run file with checkpoint and restart, VAL-008 on a smooth monotone front with the Courant number halved per refinement, the cylinder's clustered grid and OPEN thresholds, the plan's three modules with the solver changes in the modules that own the arithmetic. |

---

## Document History

| Date | Change | Author |
|------|--------|--------|
| 2026-04-14 | Initial version. Architecture defined pre-development. | Alex Moroz-Smietana |
| 2026-04-15 | Phase 2 architecture updates: collocated grid with Rhie-Chow (REQ-S07), Jacobi pressure solver (REQ-S08), hybrid advection scheme (REQ-S09), configurable under-relaxation (REQ-S10). ADR-005 amended from C/ctypes to CUDA C++/pybind11 with NumPy reference solver. REQ-S06 and REQ-N03 updated accordingly. | Alex Moroz-Smietana |
| 2026-04-16 | ECR-001 approved: solver architecture rebuild. REQ-S07 and REQ-S09 replaced for staggered grid and QUICK advection. REQ-S11 and REQ-S12 added for non-uniform mesh and direct BC imposition. ADR-003 amended, ADR-008 superseded, ADR-010 added. | Alex Moroz-Smietana |
| 2026-09-19 | Section 3.1 dependency graph replaced by a generated dependency matrix; 3.4 components, 3.5 runtime edges and 3.6 source fingerprint added as generated regions (scripts/gen_system_map.py). Section 2 untouched. | Alex Moroz-Smietana |
| 2026-09-19 | REQ-S02 rationale corrected: the measured VAL-001 error on 80x40 is 2.04%, identical on CI and locally, which is why the criterion is 2.5% rather than 2%. The 1.54% previously recorded in PROJECT_PLAN.md was not reproducible at the commit that claimed it. Requirement value unchanged; the ECR-001 tightening to < 1% after the rebuild is unaffected. | Alex Moroz-Smietana |
| 2026-09-19 | solve_steady gains an optional on_iteration callback plus last_pressure_sweeps and stage_seconds attributes for the benchmark harness (scripts/benchmark.py). Observability only; solver logic unchanged. | Alex Moroz-Smietana |
| 2026-09-20 | ECR-001 steps 1 and 2: mesh contract extended with per-cell widths, center-to-center face distances and per-axis stretching (REQ-S11); staggered.py added with the MAC layout and face-to-center averaging (REQ-S07); SimConfig gains stretch_x and stretch_y from an optional mesh section. Solver logic unchanged. | Alex Moroz-Smietana |
| 2026-09-21 | ECR-001 step 3: boundary_registry.py extracted from boundary.py as the configuration interpretation both boundary layers share (REQ-S12.1, derived from REQ-S12); boundary_staggered.py added with exact normal-component imposition and the tangential and pressure conditions exposed as data (REQ-S12). Collocated interface and output unchanged. Cascade rules and contracts added for the three boundary modules. | Alex Moroz-Smietana |
| 2026-09-22 | ECR-001 step 4: momentum.py added, the staggered momentum predictor with QUICK advection by deferred correction over an upwind implicit matrix (REQ-S07, REQ-S09). Its MomentumPrediction return is the coefficient contract for the step 5 pressure correction. Not integrated into solve_steady; collocated solver and harness rows unchanged. | Alex Moroz-Smietana |
| 2026-09-22 | ECR-001 step 5: pressure.py added, the staggered pressure correction (REQ-S04, REQ-S08 as written). The closed-domain right-hand side sums to zero to rounding, measured directly (acceptance criterion 6). Undamped Jacobi found not to converge on the closed system (exact -1 eigenvalue); recorded in docs/reports/pressure_correction_step5.md, REQ-S08 not amended. Not integrated into solve_steady; collocated solver and harness rows unchanged. | Alex Moroz-Smietana |
| 2026-09-22 | REQ-S08 clarified, not amended: weighted Jacobi with w = 2/3 satisfies it, since each cell still reads only previous-iteration neighbors. Rationale recorded in the requirement: the plain update has an exact -1 eigenvalue on the closed-domain system, which the weight maps to -1/3. pressure.py gains the JACOBI_WEIGHT constant and a public sweep(); correct() uses the weighted sweep, and the closed-cavity correction now converges. Not integrated into solve_steady; collocated solver and harness rows unchanged. | Alex Moroz-Smietana |
| 2026-09-22 | ECR-001 step 6: solver_staggered.py added, the staggered SIMPLE loop over momentum.py and pressure.py, alongside the collocated solver, which is unchanged. Contract added; cascade rows and section 4 headings for the staggered modules now name solver_staggered as their consumer; the collocated retirement moves to a later step. Stopping rule identical in definition to the collocated one, on the staggered layer's exact inlet flux. Measurements in docs/reports/staggered_integration_step6.md. | Alex Moroz-Smietana |
| 2026-09-22 | REQ-S07 rationale and the ADR-010 summary no longer claim a VAL-002 v defect: that error was measured against a corrupted Ghia v table, replaced by Table II as reference ghia_1982_re100_r2 (ECR-001 erratum). Requirement text unchanged. | Alex Moroz-Smietana |
| 2026-09-23 | The review Action (.github/workflows/review.yml) removed; the opening paragraph now names the local review and test commands that check branches against this document. No requirement, contract or module changed. | Alex Moroz-Smietana |
| 2026-09-24 | stopping.py added, the error_estimate stopping rule: the estimated iteration error over the largest prescribed boundary velocity, the worst per-cell mass imbalance, and the summed imbalance over the through-flow, each below its tolerance, with no pass at the iteration cap. REQ-S01 and REQ-S04 clarified, not amended: under error_estimate REQ-S01's tolerance applies to the estimated iteration error, and REQ-S04's per-cell tolerance is enforced at stopping together with the summed bound on the flux drift. SimConfig gains three optional solver keys (stopping_rule, default velocity_step; iteration_error_tol; mass_imbalance_tol) and rejects any unknown solver key. StaggeredSolver gains converged and stop_reason and is bitwise unchanged under the default rule; NavierStokesSolver refuses error_estimate. Contract, cascade rows and the system map updated. See docs/reports/stopping_rule_evidence.md, section 9. | Alex Moroz-Smietana |
| 2026-09-25 | StaggeredSolver exposes flux_scale, read-only: the flux scale its error_estimate rule was built with, None under velocity_step, so scripts can record the value the rule ran on. No behaviour change. The stopping.py component row names all three conditions. | Alex Moroz-Smietana |
| 2026-09-25 | Cascade row for solver_staggered.py lists scripts/val001_order.py (review 25 S1) and scripts/self_convergence.py (ECR-001 step 8 report, F8), the scripts that read the solver's public shape, and says the harness stores velocity_step_below_tol as residual_below_tol. No requirement, contract or module changed. | Alex Moroz-Smietana |
| 2026-09-30 | REQ-S04 clarified again, not amended: error_estimate gains condition (d), the absolute signed domain sum of the per-cell imbalance below mass_imbalance_tol, ECR-001 criterion 6's domain-sum clause, which nothing checked before (review 27 B1). No configuration key. The imbalance callable returns an ImbalanceSummary (worst, absolute_sum, signed_sum) in place of a pair; stopping.py gains RULE_VERSION, which scripts/stopping_probe.py and scripts/val001_order.py (through its new reuse_key) store with saved solves. Contract and the stopping.py cascade row updated. Cavity stops bitwise unchanged; channel stops later. See docs/reports/stopping_rule_evidence.md, section 10. | Alex Moroz-Smietana |
| 2026-09-30 | ECR-001 step 9. REQ-S02 amended to < 1% on 80x40 (ECR-001 section 6) and REQ-S03 to score against marchi_2009_re100 with ghia_1982_re100_r2 reported unscored (amendment of 2026-09-24), each with its measured value. REQ-S11 amended to the per-axis, mirrored stretching step 1 built; REQ-S04's Verified By corrected (VAL-007 is Phase 3's). REQ-S01, S07, S08, S09 and S12 checked against the build and unchanged. Section 2.1 names the staggered solver as the NS solver; the collocated solver and boundary.py are described as the kept harness baseline, no longer as retired in a later step, and the NavierStokesSolver contract no longer claims staggered storage. ADR-010 added as a file and its summary row updated; ADR-008's row corrected to 2.5%. No module, interface or generated region changed. | Alex Moroz-Smietana |
| 2026-10-01 | The harness records RULE_VERSION in the params of every error_estimate row and keys its summary on it; an error_estimate row without it reads as version 2, the three-condition rule (review 28 B1). The stopping.py cascade row names scripts/benchmark.py. The three rows taken at 8aac137, never on main, were replaced by rows that carry the version. Condition (d) accepted as built: it is met at a zero crossing of the net outflow's decaying oscillation (docs/reports/stopping_rule_evidence.md, section 10). No requirement, contract or rule behaviour changed. | Alex Moroz-Smietana |
| 2026-10-01 | The ADR-010 summary row states continuity as ADR-010's planned-against-built table does (review 27 S2); REQ-S03's floor against Ghia placed where its source puts it, u near y = 0.85 on the vertical centerline and v at the jet stations (S3); REQ-S02's measured value is the row taken under stopping rule version 3. No requirement value changed. | Alex Moroz-Smietana |
| 2026-10-01 | Section 3.2 rows for config.py, mesh.py, solver_ns.py, solver_staggered.py and csolver/ and the section 4 headings for mesh.py, solver_ns.py and solver_staggered.py brought in line with the generated import graph and with section 2.1 (review 27 B2): existing importers listed from the graph, planned consumers marked, the transport and time-integration consumers of the NS solver routed to solver_staggered, csolver/ to the solver of record. A constants.py row added. The solver_staggered contract says it lacks compute_residual and solve_timestep. No requirement, module or generated region changed. | Alex Moroz-Smietana |
| 2026-10-02 | The collocated solver retired (PR 29, Alex's decision of 2026-10-02): src/solver_ns.py, src/boundary.py and their tests deleted; annotated tag collocated-final on 98f8b1f, the last commit holding them; IterationState moved to stopping.py with its fields unchanged. Section 2.1 names the staggered solver as the only solver; REQ-S02's rationale keeps the collocated measurement as history, and REQ-S12.1's names the staggered layer and the Phase 3 concentration layer as the registry's readers. Section 3.2 rows and section 4 contracts for the two modules removed; the solver_staggered row and contract state the public shape in its own terms; a Retired modules note added to section 4. The harness and the viewer default to staggered-jacobi, and the 22 stored collocated rows still summarize. The adaptive outer iteration is deferred to a later efficiency pass, not Phase 3. No staggered computation changed: val002_20x20 reproduces bitwise. REQ-S02 and REQ-S03 move from the deleted solver_ns.py annotation to solver_staggered.py's, which changes the generated traceability table; REQ-S08 stays with pressure.py alone, which does that work. The scenarios.py cascade row and section 4 heading name the boundary layers (planned, Phase 4) in place of the deleted boundary module. | Alex Moroz-Smietana |
| 2026-10-03 | ADR-011, the Phase 3 transport design, added as a file with status Proposed (PR 30). REQ-S13 (the NS solver exposes its face velocities) and REQ-T11 (a uniform field stays uniform to the bound the stopping rule's per-cell condition sets) added as proposed, each with rationale and verification; no existing requirement's text changed. The solver_transport.py contract stub replaced by ADR-011's draft, marked planned, and a boundary_concentration.py contract added, marked planned; the staggered.py and solver_staggered.py contracts gain FaceVelocities and face_velocities, marked planned. Cascade rows for both planned modules; the particles.py row names its second consumer and the sign conventions. No module or generated region changed. | Alex Moroz-Smietana |
| 2026-10-03 | ADR-011 revised to Accepted at merge on Alex's decisions of 2026-10-03 (PR 30, builder-fix pass after premise review 30 and test 30). REQ-T12 (positivity and boundedness) added. REQ-T11 reworded: REQ-S04's clauses named rather than lettered, every inlet carries the uniform value, the bound in its per-cell form, the product-configuration clause moved to a deliverable (decision 6); REQ-S13 names the clauses likewise. REQ-T06's rationale gains a note on v_ext as a drift velocity; its text is unchanged. The solver_transport.py draft contract gains the sources argument and the FieldHistory writer and counts over non-SOLID cells; boundary_concentration.py's drops settling_u; the SimConfig and BoundarySpec contracts carry the planned transport keys; the config.py and staggered.py cascade rows name the planned readers. Verified By names VAL-012, VAL-013 and VAL-014. No existing requirement's text changed; no module or generated region changed. | Alex Moroz-Smietana |
| 2026-10-03 | The first Phase 3 build (PR 31): staggered.py gains FaceVelocities, check_staggered_pair and edge_cells, and StaggeredSolver sets face_velocities at the end of every solve (REQ-S13 built; its Verified By names the tests). SimConfig gains the optional transport section as a TransportSpec and BoundarySpec the three segment keys, each allowed on one segment type; the product supply is hepa_filtered. boundary_registry.py gains segment_at and coverage_along, the one derivation of which faces a segment covers with SOLID cells read as walls, and boundary_staggered.py reads its cell and corner conditions from it with its public contract unchanged. boundary_concentration.py built to its contract, which moves from planned to built with the surface codes; cascade rows for config, staggered, boundary_registry, boundary_concentration, particles and solver_staggered updated. No requirement's text changed. | Alex Moroz-Smietana |
| 2026-10-03 | PR 31's fix pass on review 31 and test 31 (Alex's decisions 1 to 3 of 2026-10-03): a boundary segment with an unknown key, two overlapping segments on one edge, and a concentration key on a zero-normal velocity_inlet fail the load, and NaN or infinity is rejected wherever a number is expected; REQ-C02's rationale names them. A zero-normal velocity_inlet is a wall to the scalar layer (ADR-011 E amended). staggered.edge_cell_inputs derives the coordinates and SOLID mask both boundary layers hand coverage_along, and get_inlet_flux attributes flux by the name coverage gives each face. BoundarySpec.concentration is a tuple and the three keys are keyword-only; FaceVelocities owns its data and, with ConcentrationFaces, has eq=False; SURFACE_NAMES removed. Contracts for config.py, staggered.py, boundary_registry.py, boundary_staggered.py and boundary_concentration.py updated; cascade rows for staggered.py and boundary_registry.py. No requirement's text changed. | Alex Moroz-Smietana |
| 2026-10-03 | The transport solver built (PR 32, branch phase3/transport-solver): src/solver_transport.py to ADR-011 B, C, D and F, its contract from planned to built with the built signatures (limited_face_values, stable_dt infinite at rest, sources validated, last_diffusion_sweeps and diffusion_converged, MassBudget.current, FieldHistory.every and the npz keys). Cascade rows for config, mesh, staggered, boundary_concentration, solver_staggered, particles and solver_transport name the built consumer and the seven test files; the section 4 headings likewise. REQ-T01, T03 to T08, T11, T12 and N01 Verified By name the test files. VAL-003, VAL-004 (two rows), VAL-007, VAL-012, VAL-013 and VAL-014 measured (docs/PROJECT_PLAN.md). validation/metrics.py and validation/transport_cases.py are the tests' shared instrument. No requirement's text changed. | Alex Moroz-Smietana |
| 2026-10-03 | PR 32's fix pass on review 32 and test 32 (Alex's decisions 1 to 3 of 2026-10-03): ADR-011 H amended twice, VAL-014's doubled-floor control dropped with the floor-face control named as the composition's guard, and VAL-003's gate step moved to a diffusion number of 0.1 by the measured error split. The transport solver's contract gains the ParticleProperties and ScalarConditions protocols, finiteness and dtype checks on every input before any arithmetic, a bool dt refused, IndexError named, diffusion_tol stated as relative, and relative() ValueError before the first step (review B2). The momentum.py cascade row and heading name solver_transport and the quick_face_values conventions it depends on. Five behaviours of the solver that no test could fail on (review B1) each gain a test that its removal fails. No requirement's text changed. | Alex Moroz-Smietana |
| 2026-10-04 | ECR-002 accepted by Alex on 2026-10-04; its section 5 requirement and scope text entered here in step 0's pull request. REQ-S01 clarified, not amended (condition (e) on the eddy viscosity and rule version 4 with the turbulence model on); REQ-S10 amended (`alpha_turbulence`) and REQ-T01 amended (the diffusivity Brownian plus the turbulent term of REQ-T13); REQ-S14 to S18 and REQ-T13 added. None of the new or changed clauses is built yet, and each row names the ECR-002 step that builds it. Section 5: in scope, incompressible Navier-Stokes laminar or with the k-epsilon model; out of scope, ADR-004's exclusion of turbulence modelling removed and LES, DNS and other RANS models added. Section 6: ADR-004 superseded by ADR-012, an ADR-012 row added, and the list of ADRs held as files corrected. The register pin in tests/test_system_map.py widened to REQ-S18 and REQ-T13. No module, contract or generated region changed. | Alex Moroz-Smietana |
| 2026-10-05 | ECR-002 step 1, first commit (prompt 35): src/scalar_scheme.py holds the face value, the advective flux and the implicit solve, moved out of solver_transport.py with per-face conductances, a cell sink and a per-cell step; the transport gate tests return bitwise main's fields at every step (results/builder35/). Its contract and cascade row added; limited_face_values leaves the solver_transport contract; momentum.py's quick_face_values is read by scalar_scheme, not solver_transport. A departure from ADR-012 I, which named import from the transport solver: a flow-side model importing the particle solver would invert the layers. No requirement's text changed. | Alex Moroz-Smietana |
| 2026-10-05 | ECR-002 step 1, second commit (prompt 35): SimConfig gains the optional turbulence section as a TurbulenceSpec (model, variant defaulting to standard, wall_treatment, cfl_number in (0, 1/2], alpha_turbulence in (0, 1], max_iter, tol), validated per REQ-C02 with unknown keys refused; absent means the model is off. The config.py contract and cascade row name it; REQ-S10's rationale notes alpha_turbulence is validated now and read from step 6. No solver reads the section yet. No requirement's text changed. | Alex Moroz-Smietana |
| 2026-10-05 | ECR-002 step 1, records (prompt 35): src/turbulence.py built, k and eps on a prescribed face field, both variants; its contract added with the departures from ADR-012 I's draft (the shared scheme module, a conditions object passed every step, the edges' tangential velocities in it, ADR-012 F's harmonic face rule, eps held exactly; wall_viscosity not built), its cascade row, and scalar_scheme's held mask. REQ-S14's rationale says what is built; REQ-S15's rationale and Verified By name the step's assertion and the tests; VAL-015 passes for both variants. No requirement's text changed. | Alex Moroz-Smietana |
| 2026-10-05 | ECR-002 step 1, fix pass on review 35 and test 35 (prompt 35b): KEpsilonModel.step checks the state's nu_t (non-negative, finite) and that k, eps and nu_t are zero in SOLID cells before any arithmetic, and the strain's face differences read a face beside a SOLID cell as zero; the turbulence.py contract says so. Line 4 and REQ-S16's rationale and Verified By record step 1 built and the transport solver's fields hashed against main. The config.py, mesh.py and staggered.py cascade rows and the mesh.py and staggered.py headings name scalar_scheme and turbulence where they import them. No requirement's text changed. | Alex Moroz-Smietana |
| 2026-10-06 | ECR-003 accepted by Alex on 2026-10-06 with ADR-013's six decisions; its section 5 text entered here in step 1's first commit (prompt 37). REQ-S08 amended: the pressure correction is solved by Jacobi-preconditioned conjugate gradients from p' = 0 to a relative residual `pressure_rtol` or a rounding floor of 1e-13 times the flux scale, confirmed on the true residual at exit, or to a reported iteration cap, with b projected onto the range on a closed domain; the Jacobi text and its 2026-09-22 clarification are history. REQ-S04 clarified again, not amended: the per-cell imbalance is the residual of the p' equation, so `pressure_rtol` bounds it relative to u*'s. Section 6 gains ADR-013 and marks ADR-010's weighted sweep superseded. Not built in this commit; the contracts, cascade rows and generated regions follow in the same pull request. | Alex Moroz-Smietana |
| 2026-10-06 | ECR-003 step 1 built (prompt 37, branch feature/ecr003-pressure-cg): src/pressure.py solves the correction by Jacobi-preconditioned conjugate gradients (apply_operator, conjugate_gradient, ConjugateGradientResult, PRESSURE_SOLVER_VERSION, RESIDUAL_FLOOR, ZERO_SCALE; PressureCorrection carries iterations and reached_cap; the corrector forms flux_scale and refuses a closed domain in two components); SimConfig reads pressure_rtol in [1e-10, 1), refuses pressure_tol by name and exposes SOLVER_KEYS, the one solver-key list the harness records from (issue 38); StaggeredSolver renames last_pressure_sweeps to last_pressure_iterations, counts pressure_cap_hits and does not stop under velocity_step on a capped correction; IterationState.pressure_sweeps becomes pressure_iterations. The harness label is staggered-cg, staggered-jacobi retired beside collocated-jacobi; saved solves carry PRESSURE_SOLVER_VERSION or the label. Contracts for config.py, pressure.py, solver_staggered.py and stopping.py and their cascade rows updated; the pressure.py contract records the departures from ADR-013 D's draft. REQ-S08's Verified By tests exist; VAL-001 and VAL-002 are retaken in step 2. Every laminar field changes beyond rounding. | Alex Moroz-Smietana |
| 2026-10-06 | ECR-003 step 1 fix pass (prompt 37b) on review 37 and test 37. pressure.py: STAGGERED_METHODS and STAGGERED_METHOD, the method label looked up by PRESSURE_SOLVER_VERSION, which scripts/benchmark.py, scripts/view_field.py and scripts/self_convergence.py import; the constructor refuses an open domain with a component of the cells with an equation that holds no outlet cell, as it refuses a closed domain in two components; conjugate_gradient checks its arguments; ConjugateGradientResult and PressureCorrection carry products, every product with the operator, and IterationState gains pressure_products as its last field, which the harness counts its work in (row schema 2, work.inner_products). scripts/stopping_probe.py reads every truth through solve_truth; scripts/val001_order.py's reuse key carries PRESSURE_SOLVER_VERSION. Contracts and the pressure.py and stopping.py cascade rows updated; tests/test_turbulence.py named among mass_imbalance's consumers. No committed configuration or validation case is refused; criteria 3 and 4 unchanged. | Alex Moroz-Smietana |
| 2026-10-07 | Note to the 2026-10-01 entry that begins 'The harness records RULE_VERSION', on condition (d) (ECR-003 step 2): the decaying net-outflow oscillation at whose zero crossing (d) was met belonged to the weighted Jacobi correction, which left a median 99.9% of u*'s imbalance in the corrected faces on the two VAL-001 80x40 channels. Under ECR-003's conjugate gradients the faces balance to rounding at every outer iteration, and on those channels conditions (b) to (d) hold from the first and (a) alone sets the stop (`docs/reports/ecr003_step2_baseline.md`, section 6). The entry above stands as history; no requirement, contract or rule behaviour changed. | Alex Moroz-Smietana |
| 2026-10-07 | ECR-003 steps 2 and 3 (prompt 38); the request closed. REQ-S02's and REQ-S03's rationales name the values of the `staggered-cg` rows at 311034e (4.108e-4 and 3.024e-3; u 1.057e-3 and v 7.356e-4 of the lid speed) and say that the orders of convergence were measured under the weighted Jacobi correction and are not retaken, with the largest change the measured field differences can cause beside each (0.003 on 1.99, 6e-4 on the cavity's); the requirement texts are unchanged. A dated note beside the 2026-10-01 entry on condition (d): the net-outflow oscillation was the Jacobi correction's. Section 6's ADR-013 row: built, with planned against built. Line 4 records the closure. No module, contract, cascade row or generated region changed. | Alex Moroz-Smietana |
| 2026-10-07 | The cleanup pull request (prompt 39, branch fix/deferred-findings-cleanup), which closes GitHub issues 33, 36, 40, 42, 45, 47, 51 and 63. src/pressure.py: PRESSURE_BLAS_THREADS = 1, applied by conjugate_gradient around the CG loop with threadpoolctl (user_api blas, one ThreadpoolController built at import); threadpoolctl>=3.2 is added to requirements.txt; ECR-003 criterion 4 is met under the default environment and the face hash of val001_80x40 is unchanged, so PRESSURE_SOLVER_VERSION stays 2 (docs/reports/blas_threads.md). The unreachable-region refusal's message no longer says every correction would reach the cap. src/mesh.py: OBSTACLE_EDGE_TOLERANCE (1e-9 of the local cell width) on all four obstacle edges, which moves 50 cells of clean_room_default (column 57, the server rack's x_end edge) and no cell of any validation configuration. src/particles.py: TypeError for a bool or np.bool_ size_class. src/boundary_staggered.py: TangentialCondition has eq=False. src/solver_transport.py: a 0-d bool array is refused as dt. validation/metrics.py: lid_velocity and inlet_velocity are public and poiseuille_reference holds the analytic profile. validation/cases.py: load_wall_clustered validates before it divides. The contracts of mesh.py, particles.py, pressure.py and solver_transport.py are updated. No requirement changed. | Alex Moroz-Smietana |
| 2026-10-07 | ECR-002 step 2 built (prompt 40, branch feature/ecr002-transport-coupling): solve_timestep gains the keyword-only `eddy_viscosity` (REQ-T13, ADR-012 F). The diffusivity of an interior face between two non-SOLID cells is the Brownian coefficient plus nu_t / Sc_t, the face value of nu_t the distance-weighted harmonic mean of the two cell values (zero when either is), formed once per call and passed to the implicit solve as per-face conductances; every other face, the advection, settling, deposition, sources, budget and stable_dt are unchanged, and None is the laminar path with every field the 43 step-taking tests of the seven transport files return hashed equal to origin/main's at 4ece52f (20,662 steps; results/builder40/). `transport.turbulent_schmidt` is a new optional key (TransportSpec.turbulent_schmidt, None when absent, no default in code); the default configuration carries 0.7. The field is refused before any arithmetic when it is not a float64 ndarray, has the wrong shape, is not finite or is negative in a non-SOLID cell, or when Sc_t is not configured; values in SOLID cells are not read. REQ-T01 and REQ-T13 move from planned to built with their tests named. The new gate rows beside VAL-007, VAL-012 and VAL-013 run a prescribed non-uniform field (validation.transport_cases.prescribed_eddy_viscosity); VAL-012's at the committed implicit tolerance never iterates, so it also runs at 1e-15, where the departure falls from 4.0e-12 to 1.9e-12: the smoothing prompt 40's revised prediction (c) allows (its erratum). No requirement text changed. | Alex Moroz-Smietana |
| 2026-10-08 | ECR-002 step 3 built (prompt 42, branch feature/ecr002-fixed-flow-outlets): the boundary type `fixed_flow_outlet`. config.py accepts it with an optional positive `velocity`, refuses the inlet's component and concentration keys on it, and checks the two decision 4 rules that need no mesh at load; boundary_registry.py gains FIXED_FLOW_OUTLET and fixed_flow_condition; boundary_staggered.py resolves each segment's outward velocity once, from the shared coverage and the mesh's face widths, exposes it as fixed_flow_velocities(), writes it as a Dirichlet normal velocity with zero tangential velocity, leaves the segment out of the inlet flux, the pressure outlets and get_max_boundary_velocity, and refuses a remainder that is not positive; boundary_concentration.py needs no code (a fixed-flow outlet is an outflow face with no deposition), and turbulence.py a note for step 6. The product configuration moves its four returns (no velocity) and the hood (0.5 m/s) to the new type, so it has no pressure outlet and the corrector takes its closed-domain path. REQ-S18 and the other records follow in the same pull request. | Alex Moroz-Smietana |
| 2026-10-08 | ECR-002 step 4 built (prompt 43, branch feature/ecr002-viscosity-field): momentum with a cell viscosity field. The momentum.py contract gains predict's mu_eff (ADR-012 D's face rule, form b's stress source) and wall_mu (the wall-function hook, its corner layout and read faces), the sweep count and the obstacle wall stencil, diffusion and QUICK, with its corner rule; it says which pressure the solver returns and that no outlet datum is built (Alex, 2026-10-08). The solver_staggered.py contract gains solve_steady's eddy_viscosity keyword; the config.py contract the optional solver key momentum_sweeps (SOLVER_KEYS thirteen); the config.py and momentum.py cascade rows say who reads what. REQ-S14's rationale and Verified By name the coupling built and its tests; REQ-S16's record step 4's hashes. The momentum.py annotation's responsibility updated and both momentum.py and solver_staggered.py declare REQ-S14. No requirement's text changed. | Alex Moroz-Smietana |
| 2026-10-08 | ECR-002 step 3, fixes from review 42 and test 42 (prompt 42b). config.py refuses, at load, a fixed_flow_outlet in a file with no velocity_inlet, a `velocity` on a pressure_outlet and a blank `velocity` on a fixed_flow_outlet (Alex's decisions of 2026-10-08); `get_max_boundary_velocity` behaves as before and its docstring and contract line say what it measures (the largest velocity the room is driven by). REQ-S18's verification column no longer cites the gitignored results directory. REQ-S18 gains the velocity-inlet rule (review 42b S2). | Alex Moroz-Smietana |
| 2026-10-09 | ECR-002 step 6 built (prompt 45, branch feature/ecr002-coupled-solve), the coupled solve. config.py: velocity inlets that admit air state turbulence_intensity and dissipation_length with the turbulence section, refused otherwise; the section is refused under velocity_step. boundary_staggered.py: wall_faces. turbulence.py: the wall functions (KAPPA, E_WALL, Y_STAR_FLOOR, wall_viscosity, inflow_values, TurbulenceBoundary). momentum.py: stencil_viscosity. solver_staggered.py: the coupled outer iteration, turbulence_state, rule_version and rule_version(config), eddy_viscosity refused with the section. stopping.py: condition (e), the version the rule's, RULE_VERSION retired. Contracts, cascade rows and REQ-S01, S14, S15 and S17 updated; turbulence.py now imports boundary_registry and boundary_staggered. VAL-016 split by Alex (2026-10-09): the implementation against the same-grid 1D solve, the model's core k against the refined one (tests/test_turbulent_channel.py). VAL-001, its stretched twin, VAL-002 and the 40x15 ladder bitwise the base at every code commit. | Claude (builder), Alex Moroz-Smietana |
| 2026-10-10 | ECR-002 step 6, k and eps advected by upwind (prompt 48, branch feature/ecr002-turbulence-advection; Alex's decision of 2026-10-10, ADR-012 C and I). config.py: the turbulence block gains the optional `advection_scheme`, `upwind` or `umist`, absent meaning upwind, validated as `variant` is; the transport block's key and its umist default are unchanged. turbulence.py: step 1 takes the configured face value. The config.py and turbulence.py contracts, the config.py cascade row and REQ-S15's Verified By (the positivity tests under both schemes) say so. VAL-001, its stretched twin and VAL-002 hash to the base; no requirement's text changed. | Alex Moroz-Smietana |
| 2026-10-11 | ECR-002 step 7, condition (e)'s scale (prompt 50, branch feature/ecr002-rule-e-scale; Alex's decision of 2026-10-11, ADR-012 E amended). stopping.py: `viscosity_scale`, the molecular nu plus the median nu_t of the FLUID cells; `ErrorEstimateRule` takes `viscosity_condition` at construction in place of `nu_scale`, and `update` takes `nu_scale` with each step; `viscosity_scale_history`; RULE_VERSION_WITH_E is 5. solver_staggered.py forms the scale from each outer iteration's relaxed nu_t and passes it; `rule_version` reports 5 with the turbulence section. REQ-S01 clarified again, not amended. No other module changes; `TurbulenceBoundary.largest_inlet_eddy_viscosity` is no longer called by the solver. |
| 2026-10-12 | ECR-002 step 7, k and eps by conjugate gradients (prompt 51, branch feature/turbulence-cg-solve; Alex's fallback decision of 2026-10-12, ADR-012 C's note of that date). scalar_scheme.py: implicit_step takes `method`, JACOBI (the default) or CONJUGATE_GRADIENT, with the same max-norm stop and the cap counted in the method's iterations; a CG solve that reaches its cap or ends with a value at or below zero in a solved cell falls back to Jacobi from C*; ImplicitResult gains method, iterations (was sweeps), fallbacks and fallback_sweeps; the CG loop runs in its own one-thread BLAS limit. turbulence.py: both solves name CONJUGATE_GRADIENT; last_sweeps becomes last_iterations, with last_fallbacks. solver_transport.py names JACOBI. The scalar_scheme.py and turbulence.py contracts, the config.py contract's max_iter, the scalar_scheme.py cascade row and its component annotation updated. VAL-001, its stretched twin and VAL-002 hash to main, and a 240-system Jacobi battery is bitwise main's (REQ-S16). No requirement's text changed. See docs/reports/turbulence_cg_solve.md. | Alex Moroz-Smietana |
| 2026-10-14 | ADR-013's named successor built as an option (prompt 53, branch feature/pressure-multigrid; Alex's decision of 2026-10-14, ADR-013 decisions 1 and 2 noted). src/multigrid.py, new: additive-correction multigrid by 2x2 aggregation, its coarse operators the exact Galerkin sums, one symmetric V(1,1) cycle with weighted Jacobi smoothing and a dense pseudo-inverse on the coarsest level. src/pressure.py: conjugate_gradient takes its preconditioner as a callable (jacobi_preconditioner, multigrid_preconditioner), the diagonal one the same product as before; PressureCorrector reads pressure_preconditioner and, on a closed domain under multigrid, removes the preconditioned vector's mean. src/config.py: the optional solver key pressure_preconditioner, jacobi or multigrid, absent meaning jacobi, in SOLVER_KEYS, so the harness records it in every row. The default is unchanged: VAL-001, its stretched twin and VAL-002 reproduce the base's stop, count and face hash. REQ-S08 noted, not amended; the default is Alex's decision on the report's measurements. The contracts of config.py and pressure.py are updated, multigrid.py's added, with its cascade row. | Alex Moroz-Smietana |
| 2026-10-14 | Multigrid becomes the default pressure preconditioner (prompt 54, branch feature/multigrid-default; Alex's decision of 2026-10-14, ADR-013 decision 1). src/multigrid.py: every level's smoother weight is the fixed SMOOTHER_FACTOR / 2 (2/3), the power iterations, POWER_ITERATIONS and Level.eigenvalue removed, because every level is weakly diagonally dominant and lambda_max(D^-1 A) is at most 2 by Gershgorin; Multigrid raises ValueError for a finest system that is not weakly diagonally dominant. src/config.py: pressure_preconditioner absent means multigrid (DEFAULT_PRESSURE_PRECONDITIONER); jacobi stays selectable and is bitwise the pre-prompt-53 loop. src/pressure.py: PRESSURE_SOLVER_VERSION 3 with the label staggered-mgcg, because the default changes every iterate and so every saved history; scripts that store the version solve again. validation/cases.py: with_pressure_preconditioner. scripts/benchmark.py: the method label selects the preconditioner (staggered-mgcg multigrid, staggered-cg the diagonal, stored rows keep staggered-cg), METHODS and CELL_UPDATE_DEFINITIONS follow. REQ-S08's text names the multigrid preconditioner and the diagonal on request. Every laminar face hash changes and is retaken in the same change (docs/reports/multigrid_default.md). | Alex Moroz-Smietana |
| 2026-10-14 | The hashes of record under the multigrid default (prompt 54, commit E): REQ-S16 holds the diagonal's laminar face hashes beside the default's, with the date and commit; docs/reports/multigrid_default.md has the table. No interface changed. | Alex Moroz-Smietana |
| 2026-10-15 | ECR-002 step 8 (prompt 56, branch feature/ecr002-step8-product). No module changed. configs/clean_room_default.yaml is the product configuration (320x120, the k-epsilon section, error_estimate, the step 8 solver keys; Alex's decision 3 of 2026-10-15); configs/clean_room_l1_return3_closed.yaml and clean_room_l2_left_supply.yaml are the two layouts; validation/cases.py and scripts/benchmark.py gain the `product` family and the preset `val018_320x120`, whose row is in benchmarks/results.jsonl. Section 1's stakeholder need carries the closed variant clause. REQ-S01 and REQ-S14 name the VAL-018 row. docs/reports/ecr002_step8_product.md. | Claude (builder) |
| 2026-10-15 | Alex's decisions before the Phase 3 gate (prompt 58, branch phase3/gate-report). Section 1: the comparative claim stated per metric, and VAL-021 closed as a documented gap (the transport's turbulent spreading unvalidated against measurement). No module, interface or requirement text changed; `configs/clean_room_default.yaml` and the two layout files carry `max_simple_iter` 15,000 (ECR-002 step 8's note). | Alex Moroz-Smietana |
| 2026-10-15 | ECR-002 closed at step 9 (prompt 58, branch phase3/gate-report). The status line states the build; ADR-012's row reads Accepted (built) with planned against built. No module, interface, requirement text or generated region changed. | Alex Moroz-Smietana |
| 2026-10-15 | ADR-011's planned-against-built table added at the Phase 3 gate (prompt 58); its row reads Accepted (built). No module, interface, requirement text or generated region changed. | Alex Moroz-Smietana |
| 2026-10-16 | ADR-014, the Phase 4 design, proposed (prompt 59, branch phase4/design): two tiers of time integration, short events as their steady extreme and the clean-out scenario recorded as Alex's decisions of 2026-10-16; twelve decisions put to Alex with probes under `docs/reports/probe59/` (the warm-started re-solve of the product room against rest, the cylinder's implicit step at six settings and a long march, the transport scheme's observed order for VAL-008, the room's scales and the transport march's cost). Section 6 gains the row. The planned contracts of section 4 (scenarios.py, monitor.py) are unchanged until built; ADR-014 G carries the draft contracts for scenarios.py, time_integration.py, io_manager.py and the solver changes (a warm start and a transient loop in solver_staggered.py, a transient term in momentum.py, interior clustering in mesh.py, the scenario schema in config.py). Section 5 gains the clean-out as the fourth scenario (prompt 59b, under the scope change process). No module, generated region or requirement text changed. | Claude (builder) |
| 2026-10-16 | ADR-014 accepted by Alex (prompt 60, commit 0): the recommendation of each of its twelve decisions as written, decision 11's thresholds left OPEN as its recommendation states, and one amendment, the equipment vents as airflow only (ADR-014 J, step 0b; the limit in H). Section 4 gains ADR-014 G's contracts, marked planned: the scenario section of SimConfig and the obstacle-face segment, the transient argument of predict, FlowState, the warm start, flow_state and solve_transient on StaggeredSolver, and the scenarios.py contract replaced (intervals, flow_states, sources_at, conditions_for, initial_fields); headings for time_integration.py and io_manager.py. Section 3.2's scenarios.py and time_integration.py rows rewritten to the planned contracts and an io_manager.py row added. Section 6's ADR-014 row reads Accepted. No module, requirement text or generated region changed. | Claude (builder) |
| 2026-10-16 | ADR-014 step 0 built (prompt 60, branch phase4/step0-warm-start). solver_staggered.py: `FlowState`, `SteadyCheckpoint`, `CAP_STOP`; `solve_steady` gains the keyword-only `initial` (the warm start of ADR-014 A, the start prompt 59's probe made by patching, bitwise) and `resume` (the continuation of ADR-014 D, bitwise the uninterrupted solve); `flow_state` and `steady_checkpoint` set by every solve. stopping.py: `ErrorEstimateRule.window` and `restore`. New module io_manager.py: the run file's `/steady_checkpoint/` group, written and read back bitwise (the rest of the file is step 3's). turbulence.py: `TurbulenceBoundary.largest_inlet_eddy_viscosity` removed (no caller in src/ since 2026-10-11; ADR-014 I). Section 4's solver_staggered, stopping, turbulence and io_manager contracts and section 3.2's rows for them updated; the generated regions regenerated with io_manager. Every hash of record (REQ-S16's three and the rest of `docs/reports/multigrid_default.md`'s table, ECR-002 step 8's L0, prompt 59's T1 rest arm) reproduces with neither keyword (`docs/reports/adr014_step0.md`, section 3). No requirement text changed. | Claude (builder) |
| 2026-10-16 | ADR-014 step 0b built (prompt 61, branch phase4/step0b-vents): the equipment vents. config.py: `location: obstacle` with `obstacle`, `face` and `recirculate_from` on BoundarySpec, every load refusal of ADR-014 J, and a file of stated outlets that balance the inflow accepted (the closed box). mesh.py: `Mesh.obstacle_mask`, the one placement rule. staggered.py: `ROOM_EDGE_OF_FACE`, `ObstacleFace`, `obstacle_face_inputs`. boundary_registry.py: `vents`, `coverage_on`, `covers_face`, `OBSTACLE`; it now imports staggered. boundary_staggered.py: `VentRun`, `VentFaces`, `vent_runs`, `vent_face_masks`, `vent_runs()`, `vent_faces()`, `MESH_BALANCE_TOLERANCE`; the vents in apply_normal_velocity, the inflow sums, the remainder rule and the velocity scale. momentum.py keeps a vent face's value; pressure.py unchanged but its notes. scalar_scheme.advective_flux gains `vents=None`. turbulence.py drops the vent span from the wall functions and reads the vent faces. boundary_concentration.py: `Recirculation`, ConcentrationFaces' `vent_u`, `vent_v` and `recirculation`; it now imports boundary_staggered. solver_transport.py advects through and books the vents and fills a recirculating discharge each step. Every configuration without vents bitwise unchanged (`docs/reports/adr014_step0b.md`). No requirement's text changed. | Claude (builder) |
| 2026-10-16 | Prompt 61b, review 61's fix pass on phase4/step0b-vents. config.py: `MESH_BALANCE_TOLERANCE` moves here from boundary_staggered.py, beside `BALANCE_TOLERANCE`; a `recirculate_from` pair whose flows differ is refused at load, and `boundary_staggered.vent_runs` refuses one whose flows differ on the covered faces. `VentRun`'s arrays are read-only; `BoundaryRegistry.vents` documents its return. REQ-S16's row gains step 0b's transport check. No requirement's text changed. | Claude (builder) |
| 2026-10-17 | ADR-014 steps 1 and 2 built (prompt 62, branch phase4/step1-scenarios). config.py: the `scenario` section (ScenarioSpec, EventSpec, Replace, Set, Source, Initial, merge_mappings, class_thresholds; from_dict gains base_dir), every rule of ADR-014 B at load and every interval's merged mapping validated through from_dict; an empty replacement removes a segment (ADR-014 B, amendment of 2026-10-17). scenarios.py built: ScenarioManager with section G's contract, FLOW_KEYS and TRANSPORT_KEYS (ADR-014 A, amendment of the same date), flow_fingerprint. boundary_vents.py: VentRun, VentFaces, vent_runs and vent_face_masks moved from boundary_staggered.py, unchanged; ROOM_EDGE_OF_FACE moved from staggered.py to boundary_registry.py, which drops the registry's import of staggered; boundary_concentration no longer imports boundary_staggered. Section 3.2 gains the boundary_vents.py row and the scenarios.py row as built; section 4's config.py and scenarios.py contracts are built, boundary_vents.py's added. REQ-C03's verification names its tests; config.py's annotation serves it. Four reference scenario files under configs/. Every hash of record unchanged. | Claude (builder) |
| 2026-10-17 | Prompt 62b, the fix pass of review and test 62 on phase4/step1-scenarios. config.py: `EVENT_TIME_RTOL` (1e-9 of t_end), one instant for the overlap check, `times()`, the new `ScenarioSpec.is_active` and the end-by-t_end check (review 62 S5); `ScenarioSpec.path`, the scenario file (S4). scenarios.py: the manager runs `boundary_vents.vent_runs` on every interval (S6), so it imports boundary_vents; `sources_at` reads `is_active`. The join rule and the fan rule ratified (ADR-014 A and B, S8). Section 3.2's config.py row names boundary_vents and staggered (S2); three references to `boundary_staggered.vent_runs` now name `boundary_vents.vent_runs` (S1). | Claude (builder) |
