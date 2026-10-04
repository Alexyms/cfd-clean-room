# ECR-002: Turbulence Model (k-epsilon) for the Product Room

**Project:** CFD Clean Room Simulation
**Change Request ID:** ECR-002
**Status:** Proposed 2026-10-04. Decision 1 (k-epsilon) taken by Alex on 2026-10-04; the seven design decisions of ADR-012 are open. Requirement and scope text change in `docs/SYSTEM.md` when this request is accepted, not before.
**Author:** Alex Moroz-Smietana (drafted in the builder session of prompt 33)
**Approver(s):** Alex Moroz-Smietana, Claude (pair)
**Date Raised:** 2026-10-04
**Phase Affected:** Phase 2 (the Navier-Stokes solver), and Phase 3 (the transport solver's diffusivity; the product-case item waits for this change). The seven Phase 3 gate rows already passed are not reopened.

---

## 1. Problem Statement

The product room has never been solved. Its Reynolds number on the room height, from the
committed configuration, is `rho U H / mu = 1.2 x 0.45 x 3.0 / 1.81e-5 = 89,500`, and its cell
Reynolds number on the 0.04 m mesh 1,190. Every case the flow solver was validated on ran at 5
(VAL-001) or 100 (VAL-002), cell Reynolds numbers 0.13 and 2.5
(`docs/reports/product_case_reynolds.md`, section 2).

Measured on the committed configuration, rerun for this request (report, section 4):

- As committed (200x75, pressure capped at 200 sweeps) the solve diverges: the largest speed
  passes 9 m/s at outer iteration 58 against a supply of 0.45 m/s, the net mass outflow is -1.1
  kg/s of a 3.8 kg/s through-flow, and by iteration 499 the speed is 2.4e11 m/s.
- With the pressure solved toward 1e-8 (cap 40,000 sweeps) on a 40x15 copy of the room, the real
  viscosity diverges by outer 120 (26.1 m/s), its correction at the cap in 30 of those 121
  iterations. Ten times the viscosity (Re 8,950) diverges from about outer 200 while
  every pressure correction before outer 220 reached its tolerance; a hundred times (Re 895)
  stalls with its residual between 1e-3 and 5e-3; a thousand times (Re 90) converges, its
  residual falling from 5.3e-2 to 2.6e-5 over 1,000 iterations. The room with its four
  obstacles removed diverges as the full room does.
- With heavier under-relaxation (alpha_velocity 0.2) neither the 40x15 nor the 200x75 run
  diverges, and neither converges: on 40x15 the residual falls to 4.8e-6 by outer 2,809 and then
  rises again, never reaching the 1e-6 tolerance in 3,000 iterations; on 200x75 it plateaus near
  6e-4 over 300 (report, section 4, rows 7 and 8).

Separately, one pressure correction on the product mesh needs 27,408 sweeps to the committed
tolerance and 112,519 to 1e-8, against a cap of 200, and the count grows as the square of the
cells per side (report, section 5).

ADR-004 ("Laminar flow assumption") calls the room laminar: "Clean rooms are engineered for
laminar flow. Re well below transition." In clean-room usage "laminar flow" means
unidirectional, low-intensity supply air. In the Navier-Stokes sense a room at Re 89,500 with
shear layers at the supply edges, wakes behind four obstacles and a jet striking their tops is
turbulent, and SYSTEM.md section 5 holds turbulence modelling out of scope on that ADR's
authority.

## 2. Root Cause Summary

**The flow the configuration describes is turbulent.** The probes do not prove that no steady
laminar solution exists. They show that the solver does not find one at the room's Reynolds
number while it does at Re 90, on the same geometry, boundaries, outlets and initial field, and
that the divergence does not need an unconverged pressure solve (the Re 8,950 run) or the
obstacles (the empty room). A laminar field at a cell Reynolds number of 1,190 would not resolve
the shear layers it contains if one were found, and it would carry no turbulent mixing: particles
would spread only by Brownian diffusion, 1e-12 to 1e-9 m^2/s, where turbulent diffusion in a room
like this is near 1e-3 m^2/s (ADR-012 F). The concentration fields Phase 5 scores would be streaks
the real room does not show.

**ADR-004 conflated the two meanings of laminar.** Its rationale is about the supply's design
intent, not the Reynolds number of the flow in the room.

**The pressure solve's cost is a second, independent defect.** It makes the laminar product case
unsolvable at the committed settings as well, and it is the dominant cost of any converged solve
on the product mesh (ADR-012 H).

## 3. Options Considered

The four options the orchestrator put to Alex on 2026-10-04.

### 3.1 Option A: An algebraic indoor model

The zero-equation model of Chen and Xu (1998), `nu_t = 0.03874 V L` with V the local speed and L
the distance to the nearest wall, Airpak's default for rooms. Cheap: no new transport equations
and no positivity question.

**Not taken.** Its constant is fitted to indoor cases; the 2007 survey finds such models
"appropriate ... for the cases with similar flow characteristics as those used to develop the
models" (Zhai et al. 2007, page 14, remark 3). A reader cannot judge a result from a model of our
own choosing against a published record, which is decision 1's reason (section 3.3).

### 3.2 Option B: Viscosity raised to a laminar-solvable value, labelled as a demonstration

At a thousand times air's viscosity (Re 90) the room converges. That is a constant eddy viscosity
of 1.5e-2 m^2/s, two to fifty times the eddy viscosity that intensities of 2% to 10% give at
length scales of 5 to 30 cm (ADR-012, source 9), chosen because it converges.

**Not taken.** The value has no physical basis, a Re 90 flow does not separate at the equipment
edges as the room does, and the particle dispersion Phase 5 scores would be set by the chosen
number. It stays useful as a development scaffold: the Re 90 probe is the control in the report.

### 3.3 Option C: k-epsilon

The two-equation model of the published record, its eddy viscosity added to the momentum
equation and its turbulent diffusivity to the particle transport.

**Selected.** Decision 1 (Alex, 2026-10-04): a turbulence model is added and it is k-epsilon, an
industry standard rather than an algebraic model of our own choosing, because a reader will judge
the result against the published record of the model. The 2007 survey: standard k-epsilon with
wall functions "provides acceptable results (especially for global flow and temperature
patterns) with good computational economy" (Zhai et al. 2007, page 14, remark 1).

### 3.4 Option D: Laminar with heavier damping

At alpha_velocity 0.2 the runs neither diverge nor converge: on 40x15 the residual comes within
a factor of five of the stopping tolerance and turns upward (section 1).

**Not taken.** Under-relaxation changes the path to a fixed point, not the fixed point. If the
damped iteration converges, it converges to the laminar discrete equations at a cell Reynolds
number of 1,190 on the product mesh: not resolved, dependent on the mesh, and with no turbulent
mixing for the transport (section 2). The rejection rests on what the field would mean, not on
whether it can be reached; the report says how far the damped runs got.

## 4. Recommended Change

Add the k-epsilon model to the staggered solver as ADR-012 designs it, and the turbulent particle
diffusivity to the transport solver:

- k and eps at cell centres, advected by the corrected faces with the transport solver's bounded
  scheme in pseudo-time, growth explicit and decay implicit, so both stay positive (ADR-012 C).
- The eddy viscosity `rho C_mu k^2 / eps` added to the molecular viscosity in the momentum
  equation's face conductances, with the stress terms constant viscosity let it drop (ADR-012 D).
- Walls, obstacle faces included, through the law of the wall on the existing wall stencil
  (ADR-012 B).
- A fifth stopping condition on the eddy viscosity (ADR-012 E).
- Each class's diffusivity the Brownian coefficient plus `nu_t / Sc_t` (ADR-012 F).

With the model off, every validated laminar result is unchanged to the bit. What changes for
transport: the diffusivity becomes Brownian plus turbulent, per face. What does not: the face
scheme, the settling composition, deposition (decision 6), the budget, and the seven gate
results, which were judged on prescribed or laminar face fields (Alex, decision 2 of
2026-10-04).

The pressure solve is a dependency, not part of the model: ADR-012 decision 4 recommends a
separate ECR-003 for multigrid with weighted Jacobi as its smoother, done before step 6.

## 5. Requirement Changes

Proposed text. `docs/SYSTEM.md` changes when Alex accepts this request.

### 5.1 Modified requirements

| ID | Current text | Proposed text | Reason | Verified by |
|----|-------------|---------------|--------|-------------|
| REQ-S01 | The NS solver shall converge to a steady-state velocity field with residuals below a configurable tolerance. | No change to the text. Clarified: with the turbulence model on, the `error_estimate` rule also requires the estimated iteration error of the eddy viscosity over the largest effective viscosity below `iteration_error_tol` (condition (e)), and the rule version a solve records is 3 without (e) and 4 with it. | Momentum and transport read nu_t; a solve must not stop with it still moving (ADR-012 E). | tests/test_stopping.py; VAL-016, VAL-018 |
| REQ-S10 | Under-relaxation factors for velocity (default 0.7) and pressure (default 0.3) shall be configurable via the YAML configuration. | Under-relaxation factors for velocity (default 0.7), pressure (default 0.3) and, with the turbulence model on, the eddy viscosity (`alpha_turbulence`) shall be configurable via the YAML configuration. | The coupled iteration feeds mu_t back into momentum (ADR-012 D). | Unit test |
| REQ-T01 | The transport solver shall solve the advection-diffusion equation for particle concentration on the velocity field produced by the NS solver. | ... on the velocity field produced by the NS solver, with each class's diffusivity its Brownian coefficient plus, when the NS solver models turbulence, the turbulent particle diffusivity of REQ-T13. | The diffusivity becomes a per-face field (ADR-012 F). | VAL-003, VAL-004 (unchanged); tests/test_solver_transport.py |

### 5.2 Unchanged requirements

| ID | Note |
|----|------|
| REQ-S02, S03 | Laminar validation; run with the model off and reproduced to the bit (REQ-S16). |
| REQ-S04 | The per-cell and domain-sum clauses stand; the product tolerance follows ADR-011 G's formula. |
| REQ-S05, S06, S07, S11, S12, S12.1, S13 | Unchanged. The k-epsilon step and the eddy viscosity are new data, not new layouts. |
| REQ-S08 | Not changed by this request. ADR-012 decision 4 recommends ECR-003 for it. |
| REQ-S09 | Momentum's advection stays QUICK by deferred correction. k and eps use the bounded scalar scheme (REQ-S15). |
| REQ-T02 to T12 | T04 stays the Brownian coefficient; T09 unchanged unless decision 6 takes option 2; T11 is stated with diffusion off and holds; T12's argument carries over (ADR-012 F). |
| REQ-C01 to C04, N01 to N03 | C02 covers the new keys; the model constants are module constants (ADR-012 I). |

### 5.3 New requirements

| ID | Text | Rationale | Verified by |
|----|------|-----------|-------------|
| REQ-S14 | When configured, the NS solver shall model turbulence with the k-epsilon model in the variant ADR-012 A names, adding the eddy viscosity `rho C_mu k^2 / eps` to the molecular viscosity in the momentum equation's diffusive and stress terms. | Re 89,500; decision 1. | VAL-015, VAL-016, VAL-018; VAL-017 when its criterion is set |
| REQ-S15 | The turbulent kinetic energy shall be non-negative and its dissipation rate positive at every non-SOLID cell after every outer iteration. | The eddy viscosity is defined only then; ADR-012 C shows the scheme guarantees it, without clipping. | Unit tests on a Smith-Hutton field with production; an assertion in every solve; VAL-015 |
| REQ-S16 | With the turbulence model off, the NS solver shall reproduce VAL-001 and VAL-002, and the transport solver its gate rows, bitwise. | The Phase 2 gate and the Phase 3 gate rows stand on these results. The obstacle wall stencil (ADR-012 B) changes laminar results in rooms with obstacles, none of which has a validated result. | Hashes of the VAL-001 and VAL-002 faces against main; the transport gate tests unchanged |
| REQ-S17 | Walls and obstacle faces shall be treated by the wall treatment ADR-012 B names (decision 2; ranked first: scalable wall functions, the law of the wall evaluated no closer than y* = 11.06). | y+ on the product mesh is 7 to 40 (ADR-012 B). | VAL-016; a unit test of the wall viscosity against its formula |
| REQ-T13 | With the turbulence model on, each class's diffusivity on every interior face shall be its Brownian coefficient plus `nu_t / Sc_t`, Sc_t configurable. | The classes are tracers to the turbulence (ADR-012 F). | A dense-solve unit test with a piecewise diffusivity; VAL-007 and VAL-012 rerun with an eddy-viscosity field |

Proposed validation identifiers: VAL-015 (decaying turbulence), VAL-016 (channel log layer),
VAL-017 (Annex 20 room, criterion OPEN), VAL-018 (the product room converges). The register pin
in `tests/test_system_map.py` widens for REQ-S14 to S17 and REQ-T13 when they are added.

### 5.4 Scope, ADR-004 and the scope change process

SYSTEM.md section 5, on acceptance:

- In Scope: "Incompressible laminar Navier-Stokes (SIMPLE algorithm)" becomes "Incompressible
  Navier-Stokes (SIMPLE algorithm), laminar or with the k-epsilon turbulence model (ADR-012)".
- Out of Scope: "Turbulence modeling (laminar assumption is physically justified per ADR-004)"
  is removed, and "LES, DNS, and RANS models other than k-epsilon" is added.
- Section 6: ADR-004's row becomes "Laminar flow assumption (Superseded by ADR-012)", its
  rationale replaced by: "Clean-room 'laminar flow' means unidirectional supply; at Re 89,500 the
  room's flow is turbulent (ECR-002)."

The process, SYSTEM.md section 5: (1) the rationale is in this request and the pull request that
carries it; (2) SYSTEM.md changes on acceptance, in the fix pass after review; (3) PROJECT_PLAN.md
records the request now and the scope change on acceptance; (4) the cascade is section 7.4.

## 6. ADR Changes

| ADR | Action | Details |
|-----|--------|---------|
| ADR-004 | Supersede | Laminar flow assumption superseded by ADR-012. The ADR lives in the development plan, not a file; its SYSTEM.md row is marked. |
| ADR-010 | Amend (note) | The momentum predictor's viscosity becomes a field when the model is on; obstacle faces gain the wall stencil; the stopping rule records its version per solve. |
| ADR-011 | Amend (note) | The diffusivity is per face; ADR-012 F shows each of ADR-011's claims carries over. No decision of ADR-011 changes. |
| ADR-012 | New | "Turbulence Model: k-epsilon on the Staggered Solver." The design, proposed with seven decisions for Alex. |

## 7. Affected Artifacts

### 7.1 Source code, by what runs

The viscosity reaches computation in one place: `StaggeredSolver.solve_steady` calls
`MomentumPredictor.predict`, whose `_assemble` reads `self._mu` in four lines (`src/momentum.py`,
386 to 394: the streamwise conductance, the interior transverse conductance and the two wall
rows). `PressureCorrector.coefficients` sees it only through the diagonals the prediction
returns. `ParticlePhysics` reads `config.mu` for settling and Brownian diffusion and must keep
reading the molecular value. The particle diffusivity reaches computation in one place:
`TransportSolver.__init__` reads `physics.diffusion_coeff(k)` once per class (line 475),
`solve_timestep` passes it to `_implicit_step` (line 679), which multiplies the per-unit
conductances by it (line 824); `stable_dt` does not read it.

| Artifact | Impact | Step |
|----------|--------|------|
| `src/turbulence.py` | New: the k and eps step, wall-function values, the eddy viscosity. | 1, 4 |
| `src/solver_transport.py` | The explicit advection and the implicit solve move to module functions `turbulence.py` also calls (bitwise); the `eddy_viscosity` keyword and per-face diffusivity. | 1, 2 |
| `src/momentum.py` | Optional mu_e field and wall viscosity; the face rule; the stress source; the obstacle wall stencil. With None, the present lines unchanged. | 3 |
| `src/solver_staggered.py` | The k and eps step in the outer loop; `eddy_viscosity`; the modified pressure. | 4 |
| `src/stopping.py` | Condition (e); the version a property of the rule. | 4 |
| `src/config.py` | The `turbulence` section, `turbulent_schmidt`, the inlet keys. | 1, 2, 4 |
| `configs/clean_room_default.yaml` | The turbulence section; `error_estimate` (ADR-011 decision 6). | 6 |
| `scripts/benchmark.py`, `scripts/stopping_probe.py`, `scripts/val001_order.py` | Read the rule version from the solver; the harness records the turbulence keys. | 4 |
| `validation/cases.py`, `validation/metrics.py` | The decaying box, the channel and the Annex 20 cases; the log-layer slope and the Annex 20 profile metrics. | 1, 4, 5 |

### 7.2 Tests

| Artifact | Impact |
|----------|--------|
| `tests/test_turbulence.py` | New: sources, positivity, boundary values, VAL-015, constancy of a uniform k on the VAL-001 faces. |
| `tests/test_turbulent_channel.py` | New: VAL-016. |
| `tests/test_annex20.py` | New: VAL-017, once its criterion is set. |
| `tests/test_momentum.py`, `tests/test_solver_staggered.py` | The field path, the face rule, the stress source, the wall viscosity, the obstacle stencil; the scalar path bitwise. |
| `tests/test_stopping.py` | Condition (e) and the version per solve. |
| `tests/test_solver_transport.py` | The keyword; a piecewise diffusivity against a dense solve; positivity and uniform exactness with a field. |
| `tests/test_config.py`, `tests/test_system_map.py` | The keys; the register pin. |

VAL-018 is judged from a harness row and the step 6 report, not in CI, as the 80x80 cavity is.

### 7.3 Documentation

| Artifact | Impact |
|----------|--------|
| `docs/SYSTEM.md` | Section 2 (section 5 above), section 4 contracts (ADR-012 I), section 3.2 cascade rows, section 5 scope, section 6 ADR rows; generated regions regenerate. |
| `docs/PROJECT_PLAN.md` | Phase 3's product-case item blocked on this request; the steps as deliverables on acceptance. |
| `docs/STATUS.md` | Where the project stands. |
| `docs/ADR/ADR-012-turbulence-model.md` | Create (this pass, Proposed). |
| `docs/reports/product_case_reynolds.md` | Create (this pass): the evidence. |

### 7.4 Cascade impact

From the dependency map in SYSTEM.md section 3: `momentum.py` is read by `pressure.py`,
`solver_staggered.py` and `solver_transport.py` (`quick_face_values` only, unchanged). With the
optional arguments None, each reads exactly what it reads today. `solver_staggered.py` gains a
read-only attribute; its consumers are the harness, the viewer and the planned
`time_integration.py`, which passes `eddy_viscosity` to `solve_timestep`. `stopping.py`'s
version moves from a module constant to the rule, which the three scripts that store it must
follow. `solver_transport.py` gains a keyword; the validation stand-ins
(`validation/transport_cases.py`) are unaffected, since the solver still reads only
`settling_velocity` and `diffusion_coeff` of what it is handed. `config.py` gains a section read
by `turbulence.py`, `solver_staggered.py` and `solver_transport.py`, and segment keys read by the
two boundary layers through the registry. Cross-cutting: nu_t, k and eps are `[ny, nx]` float64
contiguous fields in SI (m^2/s, m^2/s^2, m^2/s^3); the coordinate system is unchanged. The CUDA
port (Phase 6) targets the new loops (ADR-012 I).

## 8. Implementation Plan

Each step is one pull request through `/cfd-review` and `/cfd-test`.

| Step | Deliverable | Depends on |
|------|-------------|------------|
| 1 | `src/turbulence.py`: k and eps on a prescribed velocity field, a scalar solve reusing the transport scheme; sources, inlet, outlet and wall values as data. VAL-015; positivity; a uniform k stays uniform on the VAL-001 faces. Transport's step moved to shared functions, its gate bitwise. No coupling. | decisions 1, 3 |
| 2 | Transport coupling: the `eddy_viscosity` keyword, the per-face diffusivity, `turbulent_schmidt`. The gate rows unchanged with None; a piecewise-D dense-solve test; VAL-007 and VAL-012 with a field. | step 1 (a field), decision 7 |
| 3 | Momentum with a cell viscosity field: the face rule, the stress source, the modified pressure, the wall viscosity hook, the obstacle wall stencil. VAL-001 and VAL-002 bitwise with None; a channel with a prescribed viscosity varying in y against its integral solution. | ADR-012 D |
| 4 | The coupled solve: the k and eps step in the outer loop, mu_t under-relaxation, wall functions, stopping condition (e). VAL-016. | steps 1, 3; decision 2 |
| 5 | The Annex 20 room, VAL-017: both variants, the published spread measured and reported, the criterion set by Alex. | step 4; decision 5 |
| 6 | The product room, VAL-018: the configuration moves to the model and to `error_estimate`; the supply-turbulence sensitivity pair; the y+ map. Phase 3's product-case item. | step 4; ECR-003 (decision 4) |
| 7 | Records: ADR-012 accepted with planned against built, SYSTEM.md, the plan, this request closed. | steps 1 to 6 |

## 9. Acceptance Criteria

The change is accepted when all of the following are demonstrated:

1. **Laminar limit.** With the model off, `val001_80x40`, `val001_80x40_stretched` and
   `val002_80x80` reproduce their stored rows' outer counts and stops, with face hashes equal to
   main's, and the transport gate tests pass unchanged. Method: test and harness rows.
2. **Decaying turbulence (VAL-015).** In a closed box at rest, k and eps follow the exact solution
   of `dk/dt = -eps`, `deps/dt = -C_2 eps^2 / k` with an error that falls at first order in dt,
   observed order 0.9 to 1.1. Method: test.
3. **Positivity (REQ-S15).** k >= 0 and eps > 0 in every non-SOLID cell after every outer
   iteration of every test and harness solve. Method: an assertion in the step, and unit tests.
4. **The channel (VAL-016).** The slope of u+ against ln y+ over the log-layer nodes within 3% of
   the model's 1 / kappa_m (kappa_m 0.433 for the standard constants), and k there within 3% of
   `u_tau^2 / sqrt(C_mu)`, on the finer of two grids with the gap falling under refinement; skin
   friction against Dean (1978) reported. Method: test.
5. **Transport with turbulence.** With an eddy-viscosity field, VAL-007's relative residual below
   1e-4, VAL-012's departure below its bound, and REQ-T12's bounds on Smith-Hutton. Method: test.
6. **The Annex 20 room (VAL-017).** OPEN: the criterion is set by Alex from the measured spread of
   published two-dimensional k-epsilon results (ADR-012 G (iii), decision 5). Method: test and
   report.
7. **The product room (VAL-018).** The product configuration stops by
   `error_estimate_and_continuity` under rule version 4 within its cap, with `mass_imbalance_tol`
   from ADR-011 G's formula. Method: harness row and the step 6 report.
8. **Records.** SYSTEM.md, PROJECT_PLAN.md, STATUS.md, ADR-012 (accepted, with planned against
   built) and this request updated and committed. Method: inspection.
9. **Review.** Each step's pull request carries its `/cfd-review` and `/cfd-test` reports.
   Method: inspection.

## 10. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| The room has no steady RANS solution: vortex shedding from the equipment edges. | Medium | High | Step 6 measures it; an unsteady solve would be a scope change, reported rather than tuned away (ADR-012 J). |
| Wall functions used below their range in slow corners and at stagnation points. | High | Medium | The scalable form bounds the error; step 6 maps y+ over every wall node. |
| k overproduced where the supply strikes the equipment tops. | High | Medium | RNG measured beside the standard model on Annex 20 (decision 1); the Kato-Launder limiter as the follow-on. |
| The Annex 20 data are three-dimensional at x/H = 2.0. | Certain | Medium | Decision 5; the flux check is in the report. |
| The pressure solve's cost makes the product solve impractical. | Certain | High | ECR-003 before step 6 (decision 4). |
| The coupled iteration oscillates through the mu_t feedback. | Medium | Medium | `alpha_turbulence` and the pseudo-time step; condition (e) refuses to stop on it. |
| Transport results sensitive to Sc_t (0.2 to 1.3 in the literature). | High | Medium | Decision 7; the product step reports the sensitivity. |
| Effort underestimated. | Medium | Low | Phase 3's minimum viable portfolio artifact does not need the product case; the steps are separable. |

## 11. Approval

By signing below, approvers confirm that the problem statement is accurate, the selected option
is appropriate given the alternatives, the requirement and ADR changes are consistent with the
architecture, and the acceptance criteria are sufficient to close the change.

| Role | Name | Approval | Date |
|------|------|----------|------|
| Author | Alex Moroz-Smietana | Pending | |
| Reviewer | Claude | Pending | |

---

## Document History

| Date | Change | Author |
|------|--------|--------|
| 2026-10-04 | Proposed, with ADR-012 and the evidence report `docs/reports/product_case_reynolds.md`. Decision 1 taken by Alex; ADR-012's seven decisions open. Premise review and `/cfd-test 33` to follow. | Alex Moroz-Smietana |
