# ECR-002: Turbulence Model (k-epsilon) for the Product Room

**Project:** CFD Clean Room Simulation
**Change Request ID:** ECR-002
**Status:** Accepted by Alex on 2026-10-04. Proposed the same day and revised after premise review 33, test 33, the outlet measurement of prompt 33b and test 33b. Decision 1 (k-epsilon) and the nine decisions of ADR-012 taken by Alex on 2026-10-04. The requirement and scope text of section 5 enter `docs/SYSTEM.md` in step 0's pull request, because that edit moves the requirement register `tests/test_system_map.py` pins.
**Author:** Alex Moroz-Smietana (drafted in the builder session of prompt 33)
**Approver(s):** Alex Moroz-Smietana, Claude (pair)
**Date Raised:** 2026-10-04
**Phase Affected:** Phase 2 (the Navier-Stokes solver, its outlets included), and Phase 3 (the transport solver's diffusivity; the product-case item waits for this change). The seven Phase 3 gate rows already passed are not reopened.

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
  stalls over 1,000 iterations with its residual between 1e-3 and 5e-3, and continued, diverges
  at 1,633; at a thousand times (Re 90) the residual falls from 5.3e-2 to 2.6e-5 over 1,000
  iterations and to 5.0e-6 over 3,000, still above the 1e-6 tolerance. The residual is the
  largest velocity change over the reference velocity, 16.2 m/s on 40x15, so 2.6e-5 is a change
  of 4.3e-4 m/s per iteration. The room with its four obstacles removed diverges as the full
  room does.
- With heavier under-relaxation (alpha_velocity 0.2) neither the 40x15 nor the 200x75 run
  diverges, and neither converges: on 40x15 the residual falls to 4.8e-6 by outer 2,809 and then
  rises again, never reaching the 1e-6 tolerance in 3,000 iterations; on 200x75 it plateaus near
  6e-4 over 300, 0.048 m/s per iteration (report, section 4, rows 7 and 8).
- The pressure outlets let air into the room with no condition of its own: each takes its
  interior neighbour's velocity whatever its sign. Air enters through outlet faces in every run
  that diverges, and at Re 8,950 the divergence sits at the hood exhaust while it draws air in.
  Holding inward-turning faces shut, fixing the hood's flow at 0.5 m/s, or both, moves and delays
  the divergence and converges nothing above Re 895; on the product mesh at the real viscosity
  the growth starts inside the room before any outlet face reverses (report, section 8).

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
number while its residual falls at Re 90, on the same geometry, boundaries, outlets and initial
field, and that the divergence does not need an unconverged pressure solve (the Re 8,950 run) or
the obstacles (the empty room). A laminar field at a cell Reynolds number of 1,190 would not
resolve the shear layers it contains if one were found, and it would carry no turbulent mixing:
particles would spread only by Brownian diffusion, 1e-12 to 1e-9 m^2/s, where the turbulent
diffusivity in the room's core is 7e-5 to 2e-3 m^2/s for supply intensities of 2% to 10%
(ADR-012 F). The concentration fields Phase 5 scores would be streaks the real room does not
show.

**The outlets are part of how the solve diverges, not why it does not converge.** The first
version of this section read the probes as the Reynolds number's alone. The premise review found
air entering the room through the pressure outlets where the divergence grew (premise review
B1), and the outlet measurement (report, section 8) located it: at Re 8,950 the divergence sits
at the hood exhaust while it draws air in, and a fixed-flow hood removes it there. But no outlet
treatment converges the room above Re 895, the floor returns diverge instead when the hood is
fixed, and on the product mesh at the real viscosity the growth starts inside the room before
any outlet face reverses. So the outlets need a condition for entering air (ADR-012 decision 1,
step 3). Beyond them the probes first pointed to the Reynolds number: the same room, its outlets
treated or not, has a falling residual at Re 90 and does not converge from Re 895 up. That holds
for the solver's one momentum sweep per outer iteration only. With ten, the laminar 40x15 room
under T3 converges at Re 8,950 and at real air, and the one-sweep iteration repels those
solutions (step 0's report, section 7.6, from test 34b); at Re 895 it stalls with ten or fifty,
cause open. So on that grid the non-convergence was the iteration, not the Reynolds number, and
the first paragraph's reasons for the model stand unchanged. Whether the solve converges on the
product mesh with the model is a hypothesis this request measures (steps 0 and 5; ADR-012 D),
not a claim it makes.

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

**Not taken.** The survey's remark on such models, whole: "The zero- and one-equation models
with specially tuned coefficients are appropriate (sometimes even better than significantly more
detailed models) for the cases with similar flow characteristics as those used to develop the
models" (Zhai et al. 2007, Part 1, page 14, remark 3). Chen and Xu's model was developed on
indoor airflow, which the product is, and it has a published record: Part 2 of the same study
evaluates "the indoor zero-equation model" against measurements in four enclosed flows and finds
it "less accurate" than the advanced models while it "is simple and always has good convergence
speed. Its results can be used as good initial fields for more advanced models to achieve
converged results" (Zhang et al. 2007, Part 2, page 16). The first version of this section quoted
remark 3 with its favourable clause removed and said the option had no published record; both
were wrong (premise review B5). Decision 1 rests on k-epsilon being an industry standard, which
stands, and Part 2's accuracy finding agrees with it. The zero-equation field keeps a role the
plan gives it: a frozen field for the convergence measurement and a candidate starting field
for the coupled solve (section 8, step 5).

### 3.2 Option B: Viscosity raised to a laminar-solvable value, labelled as a demonstration

At a thousand times air's viscosity (Re 90) the room's residual falls steadily. That is a
constant eddy viscosity of 1.5e-2 m^2/s, ten to three hundred times the eddy viscosity that
intensities of 2% to 10% give at dissipation lengths of 5 to 30 cm in the design's convention
(ADR-012, source 14), chosen because it converges. The first version said two to fifty times,
in the other convention (premise review B2).

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
a factor of five of the stopping tolerance and turns upward (section 1). It kept the 40x15 room
closer to a fixed point than any outlet treatment measured (report, section 8).

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
- The outlets: a condition for air that turns inward through a pressure outlet, and the hood as a
  fixed-flow exhaust, ranked in ADR-012 decision 1 from the outlet measurement (ADR-012 D). It
  applies with the model off too.

With the model off, every validated laminar result is unchanged to the bit; the outlet
condition changes the laminar product solve, which has none. What changes for transport: the
diffusivity becomes Brownian plus turbulent, per face. What does not: the face scheme, the
settling composition, deposition (ADR-012 decision 7), the budget, and the seven gate
results, which were judged on prescribed or laminar face fields (Alex, decision 2 of
2026-10-04).

The pressure solve is a dependency, not part of the model: ADR-012 decision 5 recommends a
separate ECR-003, choosing between Jacobi-preconditioned conjugate gradients and multigrid by
measurement on the product mesh; as taken, it lands before step 5.

## 5. Requirement Changes

Accepted text (2026-10-04). It enters `docs/SYSTEM.md` in step 0's pull request.

### 5.1 Modified requirements

| ID | Current text | Proposed text | Reason | Verified by |
|----|-------------|---------------|--------|-------------|
| REQ-S01 | The NS solver shall converge to a steady-state velocity field with residuals below a configurable tolerance. | No change to the text. Clarified: with the turbulence model on, the `error_estimate` rule also requires the estimated iteration error of the eddy viscosity, over a scale from boundary data (the molecular viscosity plus the largest inlet eddy viscosity), below `iteration_error_tol` (condition (e)), and the rule version a solve records is 3 without (e) and 4 with it. | Momentum and transport read nu_t; a solve must not stop with it still moving (ADR-012 E). | tests/test_stopping.py; VAL-016, VAL-018 |
| REQ-S10 | Under-relaxation factors for velocity (default 0.7) and pressure (default 0.3) shall be configurable via the YAML configuration. | Under-relaxation factors for velocity (default 0.7), pressure (default 0.3) and, with the turbulence model on, the eddy viscosity (`alpha_turbulence`) shall be configurable via the YAML configuration. | The coupled iteration feeds mu_t back into momentum (ADR-012 D). | Unit test |
| REQ-T01 | The transport solver shall solve the advection-diffusion equation for particle concentration on the velocity field produced by the NS solver. | ... on the velocity field produced by the NS solver, with each class's diffusivity its Brownian coefficient plus, when the NS solver models turbulence, the turbulent particle diffusivity of REQ-T13. | The diffusivity becomes a per-face field (ADR-012 F). | VAL-003, VAL-004 (unchanged); tests/test_solver_transport.py |

### 5.2 Unchanged requirements

| ID | Note |
|----|------|
| REQ-S02, S03 | Laminar validation; run with the model off and reproduced to the bit (REQ-S16). |
| REQ-S04 | The per-cell and domain-sum clauses stand; the product tolerance follows ADR-011 G's formula. |
| REQ-S05, S06, S07, S11, S12, S12.1, S13 | Unchanged. The k-epsilon step and the eddy viscosity are new data, not new layouts. |
| REQ-S08 | Not changed by this request. ADR-012 decision 5 recommends ECR-003 for it. |
| REQ-S09 | Momentum's advection stays QUICK by deferred correction. k and eps use the bounded scalar scheme (REQ-S15). |
| REQ-T02 to T12 | T04 stays the Brownian coefficient; T09 unchanged unless ADR-012 decision 7 takes option 2; T11 is stated with diffusion off and holds; T12's argument carries over (ADR-012 F). |
| REQ-C01 to C04, N01, N03 | C02 covers the new keys; the model constants are module constants (ADR-012 I). |
| REQ-N02 | Its VAL-008 applies to the laminar solver and the transport scheme. Under wall functions refinement moves the first node's y+, so turbulent runs report a grid sensitivity on two meshes with every first node in the wall-function range, not an observed order (ADR-012 G (v)). |

### 5.3 New requirements

| ID | Text | Rationale | Verified by |
|----|------|-----------|-------------|
| REQ-S14 | When configured, the NS solver shall model turbulence with the k-epsilon model in the variant ADR-012 A names, adding the eddy viscosity `rho C_mu k^2 / eps` to the molecular viscosity in the momentum equation's diffusive and stress terms. | Re 89,500; decision 1 of 2026-10-04. | VAL-015, VAL-016, VAL-018; VAL-017 and VAL-019 when their criteria are set |
| REQ-S15 | The turbulent kinetic energy and its dissipation rate shall be positive, k > 0 and eps > 0, at every non-SOLID cell after every outer iteration. | The eddy viscosity is defined only then; ADR-012 C shows the scheme guarantees it, without clipping. | Unit tests on a Smith-Hutton field with production; an assertion in every solve; VAL-015 |
| REQ-S16 | With the turbulence model off, the NS solver shall reproduce VAL-001 and VAL-002, and the transport solver its gate rows, bitwise. | The Phase 2 gate and the Phase 3 gate rows stand on these results. The obstacle wall stencil (ADR-012 B) changes laminar results in rooms with obstacles, none of which has a validated result. | Hashes of the VAL-001 and VAL-002 faces against main; the transport gate tests unchanged |
| REQ-S17 | Walls and obstacle faces shall be treated by the wall treatment ADR-012 B names (ADR-012 decision 3; ranked first: scalable wall functions, the law of the wall evaluated no closer than the floor where it meets the linear law, y* = 11.53 for kappa 0.41 and E 9.793). | y+ on the product mesh is about 7 to 70 (ADR-012 B). | VAL-016; a unit test of the wall viscosity against its formula |
| REQ-S18 | A pressure-outlet face whose velocity turns into the room shall carry the condition ADR-012 D names for entering air (ADR-012 decision 1, taken 2026-10-04: held at zero normal velocity for that outer iteration, its tangential condition the pressure outlet's zero gradient), and an exhaust whose flow a fan sets shall be a fixed-flow outlet segment: an outward normal velocity and zero tangential velocity in the flow solver, an outflow in the concentration layer, not counted as inflow. Applies with the turbulence model on or off. | Today a reversed outlet face has no condition. Air entering through the outlets accompanies every divergence measured, and at Re 8,950 it is where the divergence sits (report, section 8). | VAL-001 and VAL-002 bitwise; unit tests of the reversed-face condition and the fixed-flow segment; the report's outlet ladder rerun with the treatment as built |
| REQ-S18 (amended 2026-10-08) | A fan-set outlet shall be a fixed-flow outlet segment (`fixed_flow_outlet`): an outward normal velocity and zero tangential velocity in the flow solver, an outflow in the concentration layer, not counted as inflow. A segment states its velocity or states none; those that state none share what the stated ones leave of the discrete inflow at one face velocity, so outflow equals inflow to rounding on any mesh. With no pressure outlet in the configuration at least one fixed-flow outlet states none, and a remainder that is not positive on the mesh is refused; with a pressure outlet, every fixed-flow outlet states its velocity. A configuration with a fixed-flow outlet and no velocity inlet is refused at load (Alex, 2026-10-08, prompt 42b): its make-up air would enter through a pressure outlet. A pressure outlet keeps the zero-gradient copy, valid for developed outflow normal to the face; a configuration whose outlet cells take sideways flow drifts under it (GitHub issue 61), which is why the product's returns and hood are fixed-flow outlets. Applies with the turbulence model on or off. | Amended in step 3 (prompt 42; ADR-012 decision 1 amended by Alex). The text above it, a reversed pressure-outlet face held shut for the iteration, is not built. The outlet probe (`docs/reports/ecr002_step3_outlet_probe.md`, sections 7.3 and 7.7) found that the copy at every open return fails the 40x15 ladder at Re 895 and 8,950 with the reversed faces held shut or left open, and that fixed-flow outlets converge at both and remove the pressure's rise. | VAL-001 and VAL-002 hash identically to the base; the unit and integration tests of the balance, the faces, the pin and the concentration faces; the demonstrations of criterion 6 |
| REQ-T13 | With the turbulence model on, each class's diffusivity on every interior face shall be its Brownian coefficient plus `nu_t / Sc_t`, Sc_t configurable. | The classes are tracers to the turbulence (ADR-012 F). | A dense-solve unit test with a piecewise diffusivity; VAL-007 and VAL-012 rerun with an eddy-viscosity field |

Proposed validation identifiers: VAL-015 (decaying turbulence), VAL-016 (plane Couette flow),
VAL-017 (Annex 20 room, criterion OPEN), VAL-018 (the product room converges, conditional on
step 5), VAL-019 (the backward-facing step, criterion OPEN). The register pin in
`tests/test_system_map.py` widens for REQ-S14 to S18 and REQ-T13 when they are added.

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
| ADR-010 | Amend (note) | The momentum predictor's viscosity becomes a field when the model is on; obstacle faces gain the wall stencil; the stopping rule records its version per solve; pressure outlets gain a condition for entering air and a fixed-flow exhaust segment (REQ-S18). |
| ADR-011 | Amend (note) | The diffusivity is per face; ADR-012 F shows each of ADR-011's claims carries over. No decision of ADR-011 changes. |
| ADR-012 | New | "Turbulence Model: k-epsilon on the Staggered Solver." The design, proposed; its eight decisions taken by Alex on 2026-10-04 as ranked first, and a ninth added (step 0). |

## 7. Affected Artifacts

### 7.1 Source code, by what runs

The viscosity reaches computation in one place: `StaggeredSolver.solve_steady` calls
`MomentumPredictor.predict`, whose `_assemble` reads `self._mu` once (`src/momentum.py`, line 370,
`rho, mu = self._rho, self._mu`) and uses the local `mu` in four conductances (lines 386 to 394:
the streamwise conductance, the interior transverse conductance and the two wall rows).
`PressureCorrector.coefficients` sees it only through the diagonals the prediction returns.
`ParticlePhysics` reads `config.mu` for settling and Brownian diffusion and must keep reading the
molecular value. The particle diffusivity reaches computation in one place:
`TransportSolver.__init__` reads `physics.diffusion_coeff(k)` once per class (line 475),
`solve_timestep` passes it to `_implicit_step` (line 679), which multiplies the per-unit
conductances by it (line 824); `stable_dt` does not read it.

| Artifact | Impact | Step |
|----------|--------|------|
| `src/turbulence.py` | New: the k and eps step, wall-function values, the eddy viscosity. | 1, 6 |
| `src/solver_transport.py` | The explicit advection and the implicit solve move to module functions `turbulence.py` also calls (bitwise); the `eddy_viscosity` keyword and per-face diffusivity. | 1, 2 |
| `src/boundary_registry.py`, `src/boundary_staggered.py`, `src/boundary_concentration.py` | The fixed-flow exhaust segment: an outward prescribed velocity in the staggered layer, an outflow in the concentration layer, not counted in the inlet flux. Built 2026-10-08 as `fixed_flow_outlet`, with the remainder rule of ADR-012 D's note. | 3 |
| `src/pressure.py` | The open outlet faces an argument of the correction; with None, every outlet face, bitwise. Amended 2026-10-08 (step 3): not needed. The product room has no pressure outlet, so the corrector's existing closed-domain path (projection and pin) serves, and no argument is added. | 3 |
| `src/momentum.py` | Optional mu_e field and wall viscosity; the face rule; the stress source; the obstacle wall stencil. With None, the present lines unchanged. | 4 |
| `src/solver_staggered.py` | The outlet condition per outer iteration; the k and eps step in the outer loop; `eddy_viscosity`; the modified pressure. Amended 2026-10-08: step 3 changes nothing here; the pressure outlet's copy is as it was. | 3, 4, 6 |
| `src/stopping.py` | Condition (e); the version a property of the rule. | 6 |
| `src/config.py` | The `turbulence` section, `turbulent_schmidt`, the inlet keys, the exhaust segment type. | 1, 2, 3, 6 |
| `configs/clean_room_default.yaml` | The hood as a fixed-flow exhaust (step 3; amended 2026-10-08: the four floor returns too); the turbulence section and `error_estimate` (ADR-011 decision 6; step 8). Built 2026-10-15 (prompt 56): the product configuration of prompts 46 to 48 on 320x120, each key sourced in the file (step 8's note). | 3, 8 |
| `scripts/benchmark.py`, `scripts/stopping_probe.py`, `scripts/val001_order.py` | Read the rule version from the solver; the harness records the turbulence keys. | 6 |
| `validation/cases.py`, `validation/metrics.py` | The decaying box, plane Couette flow, the backward-facing step and the Annex 20 cases; their profile metrics. | 1, 6, 7 |

### 7.2 Tests

| Artifact | Impact |
|----------|--------|
| `tests/test_turbulence.py` | New: sources, positivity, boundary values, VAL-015, constancy of a uniform k on the VAL-001 faces. |
| `tests/test_turbulent_channel.py` | New: VAL-016, plane Couette flow; the plane channel reported. |
| `tests/test_annex20.py`, `tests/test_backward_step.py` | New: VAL-017 and VAL-019, once their criteria are set. |
| `tests/test_boundary_staggered.py`, `tests/test_boundary_concentration.py`, `tests/test_pressure.py` | The fixed-flow exhaust, the open outlet faces and the reversed-face condition (step 3). Built 2026-10-08 as `tests/test_fixed_flow_outlet.py` and `tests/test_fixed_flow_product.py`; no open outlet faces or reversed-face condition exists to test. |
| `tests/test_momentum.py`, `tests/test_solver_staggered.py` | The field path, the face rule, the stress source, the wall viscosity, the obstacle stencil; the scalar path bitwise. |
| `tests/test_stopping.py` | Condition (e) and the version per solve. |
| `tests/test_benchmark.py`, `tests/test_stopping_probe.py` | Both read `RULE_VERSION` (an import and an assertion; a monkeypatch), so they follow the version into the rule. |
| `tests/test_solver_transport.py` | The keyword; a piecewise diffusivity against a dense solve; positivity and uniform exactness with a field. |
| `tests/test_config.py`, `tests/test_system_map.py` | The keys; the register pin. |

VAL-018 is judged from a harness row and the step 8 report, not in CI, as the 80x80 cavity is.

### 7.3 Documentation

| Artifact | Impact |
|----------|--------|
| `docs/SYSTEM.md` | Section 2 (section 5 above), section 4 contracts (ADR-012 I), section 3.2 cascade rows, section 5 scope, section 6 ADR rows; generated regions regenerate. |
| `docs/PROJECT_PLAN.md` | Phase 3's product-case item blocked on this request; the steps as deliverables on acceptance. |
| `docs/STATUS.md` | Where the project stands. |
| `docs/ADR/ADR-012-turbulence-model.md` | Create (this pass, Proposed). |
| `docs/reports/product_case_reynolds.md` | Create (this pass): the evidence, section 8 the outlet measurement. |

### 7.4 Cascade impact

From the dependency map in SYSTEM.md section 3: `momentum.py` is read by `pressure.py`,
`solver_staggered.py` and `solver_transport.py` (`quick_face_values` only, unchanged). With the
optional arguments None, each reads exactly what it reads today. `solver_staggered.py` gains a
read-only attribute; its consumers are the harness, the viewer and the planned
`time_integration.py`, which passes `eddy_viscosity` to `solve_timestep`. `stopping.py`'s version
moves from a module constant to the rule, which the three scripts that store it must follow.
`solver_transport.py` gains a keyword; the validation stand-ins (`validation/transport_cases.py`)
are unaffected, since the solver still reads only `settling_velocity` and `diffusion_coeff` of what
it is handed. `config.py` gains a section read by `turbulence.py`, `solver_staggered.py` and
`solver_transport.py`, and segment keys and the exhaust segment type read by the two boundary
layers through the registry. `pressure.py`'s correction takes the open outlet faces, which only
`solver_staggered.py` passes. Cross-cutting: nu_t, k and eps are `[ny, nx]` float64 contiguous
fields in SI (m^2/s, m^2/s^2, m^2/s^3); the coordinate system is unchanged. The CUDA port (Phase 6)
targets the new loops (ADR-012 I).

## 8. Implementation Plan

Each step is one pull request through `/cfd-review` and `/cfd-test`. Decision numbers are
ADR-012's. Step 0 is a probe, not a build; it is numbered 0 so that the step numbers the reviews
cite stay valid. A step that measures commits its predictions and the meaning of each outcome
before its runs, in a commit of its own, so their order can be checked (test 33b, check 35).

| Step | Deliverable | Depends on |
|------|-------------|------------|
| 0 | Risk retirement before anything is built (ADR-012 decision 9, Alex, 2026-10-04): does the room's iteration converge at a realistic effective viscosity? The committed laminar solver through a probe-only subclass, as prompt 33b's outlets were (`src/` unchanged), on the 40x15 room with the T3 outlets and a frozen non-uniform eddy viscosity added to the molecular viscosity in the momentum conductances by ADR-012 D's face rule, with its stress source. The field has the shape of the indoor zero-equation model's (section 3.1), computed once from a Re 90 field. As run (amended 2026-10-04): that field as published has a core median of 8.2e-3 m^2/s, about 545 times air's and above k-epsilon's core range of 6.5e-5 to 1.5e-3 m^2/s, so it probes a stronger mixing than k-epsilon will produce; the probe kept its shape and scaled it to core medians of 1.5e-3, 5e-4 and 1.5e-4 m^2/s, with the unscaled field as one rung, controls at 1.5e-3 on the scheme, the momentum sweeps and the stress source, and the uniform runs of the outlet ladder as references. A report, `docs/reports/ecr002_step0_frozen_viscosity.md`, no product code. If the room converges, steps 1 to 8 proceed as planned; if not, the result goes to Alex with step 5's convergence aids ranked, before step 1 opens. Outcome (2026-10-04): the room does not converge in k-epsilon's range with one momentum sweep per outer iteration, where at the top of the range the steady solution is a fixed point one sweep repels (test 34); the unscaled field converges, the scaled ones do not with one momentum sweep per outer iteration, and ten sweeps converge the top of the range (the report, sections 6.5 and 6.6). Ten sweeps at the middle and the bottom of the range (prompt 34b, 2026-10-05): both converge from rest, at 1,209 and 1,205 outer iterations beside the top's 1,240, where one sweep grows, so the sweep aid holds across k-epsilon's range on this grid (the report, section 7). | none |
| 1 | `src/turbulence.py`: k and eps on a prescribed velocity field, a scalar solve reusing the transport scheme; sources, inlet, outlet and wall values as data. VAL-015; positivity; a uniform k stays uniform on the VAL-001 faces. Transport's step moved to shared functions, its gate bitwise. No coupling. | decisions 2, 4 |
| 2 | Transport coupling: the `eddy_viscosity` keyword, the per-face diffusivity, `turbulent_schmidt`. The gate rows unchanged with None; a piecewise-D dense-solve test; VAL-007 and VAL-012 with a field. Note 2026-10-07 (prompt 40): built; the tests are named in REQ-T13's verification column in `docs/SYSTEM.md`. | step 1 (a field), decision 8 |
| 3 | The outlets, REQ-S18 (premise review B1; report, section 8): the treatment Alex chose in decision 1, the hood as a fixed-flow exhaust and the floor returns' inward-turning faces held shut each outer iteration (ADR-012 D). The fixed-flow exhaust as a segment type of its own (an outward prescribed normal velocity and zero tangential velocity in the staggered layer, an outflow in the concentration layer, not counted as inflow); the open outlet faces passed to the pressure correction. VAL-001 and VAL-002 bitwise, their outlets checked to carry outflow only; the report's outlet ladder rerun with the treatment as built; the reversed-face count per segment and the largest speed's cell reported per outer iteration. Amended 2026-10-08 (prompt 42, built; ADR-012 decision 1 amended by Alex): option 2 of decision 1 instead. Every return and the hood is a `fixed_flow_outlet`, the hood stating 0.5 m/s and the returns sharing the remainder of the discrete inflow at one face velocity; no pressure outlet is left in the product room, so the pressure correction takes its closed-domain path with no change to its arithmetic, and the held-shut rule and the open-outlet-faces argument are not built. The pressure outlet's copy is unchanged. | decision 1 |
| 4 | Momentum with a cell viscosity field: the face rule, the stress source, the modified pressure and the outlets' datum, the wall viscosity hook, the obstacle wall stencil. VAL-001 and VAL-002 bitwise with None; a channel with a prescribed viscosity varying in y against its integral solution. Note 2026-10-08 (prompt 43): built. `MomentumPredictor.predict(u, v, p, mu_eff=None, wall_mu=None)` carries ADR-012 D's face rule and form b's stress source, bitwise the step 0 probe's on the product room; `wall_mu` is the hook, None until step 6; `solver.momentum_sweeps` (default 1) and `solve_steady(eddy_viscosity=...)` drive prescribed fields for step 5. The outlets' datum is dropped (Alex, 2026-10-08; ADR-012 D's note). The obstacle stencil takes the domain edge's wall stencil in diffusion and, by Alex's extension of 2026-10-08, in the QUICK correction; a SOLID-floor channel then solves as the domain-floor channel to rounding. VAL-001, its stretched twin and VAL-002 hash to the base at commits A and B2 (B1 not measured; the cases have no SOLID cell); the varying-viscosity channel converges at observed orders 2.02 and 2.00 (`tests/test_viscosity_channel.py`). | ADR-012 D |
| 5 | The convergence measurement (premise review B2), before any coupled solve: with step 3's outlets and step 4's viscosity field, the solver on 200x75 at prescribed uniform effective viscosities across the core's range, 6.5e-5 to 1.5e-3 m^2/s (ADR-012 D), and at a frozen non-uniform field from the indoor zero-equation model on the Re 90 field, each with the pressure solved to tolerance; and the lid-driven cavity at Re 1,000, a steady answer at a cell Reynolds number of tens. The report's 40x15 ladder (section 8) is the coarse-grid precedent: at uniform viscosities of 1.5e-4 and 1.5e-3 m^2/s nothing converged under any outlet treatment measured; step 0 is the first measurement at a non-uniform field. The outcome decides whether step 6 needs a convergence aid (continuation in viscosity from the zero-equation field, pseudo-transient continuation, more momentum sweeps per outer iteration) and whether VAL-018 stands as written. Note 2026-10-07 (ECR-003 closed, its step 3): the dependency on ECR-003 is met. Step 0's sweep result (on 40x15, one momentum sweep repels the steady state and ten converge) was measured under the weighted Jacobi correction, which left most of the imbalance in the faces (`docs/reports/ecr003_step2_baseline.md`, section 6); step 5 retakes it with the pressure solved by CG. Step 4 builds the configurable sweep count either way. The one CG measurement of that room so far is ECR-003 criterion 3: laminar at real air under T3 with ten sweeps, 1,209 and 2,822 outer iterations. ADR-013 decision 6's finding, that the laminar room converging on 40x15 converged on neither 80x30 nor 200x75 with ten sweeps and exact corrections, is step 5's to answer as well. On 200x75 a CG correction takes about 0.14 s with one BLAS thread and about 1.05 s under OpenBLAS's default threads (`docs/reports/ecr003_step2_baseline.md`, section 11), so step 5's runs state their thread setting. Note 2026-10-09 (prompt 44, measured): `docs/reports/ecr002_step5_convergence.md`. With one momentum sweep nothing converges on 80x30 or 200x75, not even the top of the range (uniform 1.5e-3 m^2/s diverges on 200x75 at outer 824); on 40x15 the top of the range and the three zero-equation fields converge at one sweep, the uniform 1.5e-4 and 6.5e-5 m^2/s rows past the prompt's cap (5,012 and 10,254 outer iterations) and laminar air grows. With ten sweeps the top of the range converges on 80x30 (uniform at 745 outer iterations, the zero-equation field Z2 at 630) and, as the uniform field only, on 200x75 (2,329; Z2 bounded to the 10,000 cap at a steady residual of 6e-4); Z3 (core median 5e-4) converges at ten on 80x30 and not on 200x75; the middle and bottom of the range (uniform 1.5e-4 and 6.5e-5 m^2/s, Z4) and laminar air converge on neither finer grid at ten sweeps, nor the uniform rows and laminar air on 80x30 at fifty. The step 4 corner rule's QUICK half decides the 1.5e-4 row on 80x30 (bounded with it, converged at 2,293 without it, nothing below it either way) and costs 4% to 5% of the top row's count on the finer grids while moving the converged field by 0.11 to 0.13 m/s at obstacle corners. `pressure_rtol` 1e-4 reproduces 1e-8's count to the iteration with the faces within 4.4e-10 m/s; 1e-2 raises the 200x75 count by 71%. One and ten sweeps give the same field within the stopping tolerance. 80x30 against 200x75 differs by up to 0.19 m/s at a sensor and 0.44 m/s at an obstacle face; each grid rounds the openings to whole faces, so 40x15 is a different room at the returns (1.054 against 1.21 m/s). The bounded rows neither fall nor grow and are not cleanly periodic (their residuals wander, recurring most strongly at 43 to 85 outer iterations on 80x30 and 300 to 350 on 200x75), located on 80x30 in the column between the door wall and the server rack (rounds 2 and 3, prompt 44b and the orchestrator's corrections). ADR-013 decision 6's finding stands. The cavity at Re 1,000 converges at one sweep on 40x40 and 80x80. The outcome, the questions for step 6 and VAL-018, and what the measurement does not settle are the report's sections 7 and 8; no decision is taken there. Round 2 (prompt 44b, 2026-10-09) corrected the grid-convergence tables and the prose; the changed values are the report's section 9. | steps 0, 3, 4; ECR-003 (decision 5) |
| 6 | The coupled solve: the k and eps step in the outer loop, mu_t under-relaxation, wall functions, stopping condition (e), and step 5's convergence aid if it needs one. VAL-016 (plane Couette flow against a one-dimensional reference); the plane channel against Dean, reported. Note 2026-10-09 (Alex, before step 6; ADR-012 D's note of that date): the coupled solve is built first with no convergence aid; step 4's corner rule stays, revisited only if a later step shows it deciding convergence. Step 6 verifies the coupled solve on cases with known answers and does not run the product room: that is prompt 46, a measurement, so that a convergence failure there cannot hold up review of code that is correct. If the coupled product room does not converge, the step stops there and the aid is chosen then, from step 0's ranking (section 6.6 of its report). Note 2026-10-09 (prompt 45): built. The inlet keys and the error_estimate requirement at load; the scalable wall functions (`TurbulenceBoundary` in `src/turbulence.py`); the coupled outer iteration in `StaggeredSolver.solve_steady`; condition (e) and the rule version a property of the rule. VAL-001, its stretched twin, VAL-002 and the 40x15 ladder hash to the base at every code commit. VAL-016 split by Alex (criterion 4's note) and passing. Two findings for the product step: `cfl_number` 0.5 locks the coupled iteration into a limit cycle where 0.25 converges, and the strain in the second cell from a wall is overstated on every grid (ADR-012 C's notes of 2026-10-09). The product room under the model is prompt 46's measurement, still to come. Note 2026-10-09 (prompt 46, measured): `docs/reports/ecr002_step6_product_coupled.md`. The coupled room converges on 200x75 with the standard variant (3,421 outer iterations, the 1e-8 check row stopping at the same count with the faces within 1.1e-11 m/s) and with RNG on 40x15 and 80x30; the standard variant on 40x15 and 80x30 reaches the 10,000 cap bounded in small cycles the rule's conditions (a) and (e) refuse to stop on (the velocity step steady at 1e-5 to 2e-4 of the scale, continuity met), located at the shear layer off the server rack's top corner and at return 2; RNG on 200x75 reaches the cap in a real oscillation of about 4 cm/s beside the litho tool's west face, with the per-cell imbalance about thirty times its tolerance. A stronger inlet turbulence (intensity 0.10, length 0.3 m) converges the standard variant on 80x30 where the baseline and the weaker setting do not. The core's nu_t / nu is the inlet's (median 16 to 17 at the baseline). Measurement 4 (where particles go) is skipped by the prompt's condition, since neither grid has both variants converged; the supplementary marches on the converged rows show the operator source in the gap captured entirely by return 2, the sensors reading nothing above 1e-12 per m^3, and the hotspots on the gap's floor and the litho tool's west face. The questions for Alex (the stop on a small cycle, the aid, the variant, the source and sensor placement for VAL-018) are the report's section 7; no decision is taken. Note 2026-10-10 (prompt 47, measured): `docs/reports/ecr002_step6_comparative.md`. The 80x30 cycle lives in the k and eps advection: with k and eps advected by upwind the row converges (1,046 outer iterations), with momentum's corner rule removed or `alpha_turbulence` halved it stays bounded, and a limiter diagnostic shows the UMIST clamp switching branch on a few tens of faces every tail iteration, beside the rack's east face and the etch chamber's west face in the gaps, with no flux changing sign. On the exact grids (160x60 and 320x120 round no stated position; 200x75 rounds fourteen) the standard variant converges on 160x60 (2,732) and stalls on 320x120 in the same kind of cycle at a residual near 1e-7, located at the litho tool's east top corner; RNG is bounded on both in a 19-iteration cycle in the gap above return 2. As a labelled supplement the upwind arm converges on both exact grids (2,003 and 4,472). All three sources the prompt placed pass the discrimination check as specified, one through a sensor (`near_door`) and two through surfaces only, and every plume ends in a return within a metre. On the upwind pair the largest deposition location is the same segment for every source and class, the five hotspots are shared five of five in five rows of six and four of five in the sixth, and the sensor orders agree; between the two schemes on 160x60 the five hotspots are identical. Predictions (a) holds, (b) fails on 320x120, (c) fails, (d) fails for S2, (e) and (f) are not scorable as written. The questions for Alex (the k and eps advection scheme or a stop on the cycle, VAL-018's grid, the sources and sensors, RNG's standing) are the report's section 7; no decision is taken. Note 2026-10-10 (Alex, prompt 48): three decisions on prompt 47's outcome. (1) k and eps are advected by plain upwind, through a turbulence key `advection_scheme` (`upwind` or `umist`, absent meaning `upwind`; ADR-012 C's and I's notes of that date), because the limiter's branch switching accompanies the standard variant's non-convergence and upwind converges 80x30, 160x60 and 320x120 while moving none of the five deposition hotspots on 160x60 for any source or class; step 7's validation cases run both schemes. (2) The product grid is 320x120, with 160x60 its coarser check (step 8's note). (3) RNG is rerun under upwind on the exact pair in prompt 48's build, with the comparison between variants that prompt 47 could not score. Note 2026-10-10 (prompt 48, built and measured): `docs/reports/ecr002_step6_variants.md`. The key is built; with it absent the standard variant reproduces prompt 47's upwind arm bit for bit on all three grids, and VAL-001, its stretched twin and VAL-002 hash to the base. RNG under upwind is bounded on 160x60 and on 320x120 at the cap, its cycle in the same gap above return 2 with the 19-iteration recurrence gone and a larger velocity step, so its non-convergence is not the limiter's. No grid has both variants converged, so the comparison between variants has no pair; the standard variant's marches reproduce prompt 47's bit for bit. Predictions (a) holds, (b) fails, (c) and (d) are not scorable; by the outcome written with them, the product runs the standard variant alone until VAL-020 decides between the variants. The questions for Alex (ADR-012 decision 2, VAL-018's variant clause, RNG's cycle) are the report's section 7. | steps 1, 4, 5; decision 3 |
| 7 | Validation against measurement, both variants: VAL-019, the backward-facing step; VAL-017, the Annex 20 room, with [Rong and Nielsen 2008]'s standard k-epsilon prediction digitized and reported beside it. Thresholds set by Alex from the first coupled results, each with its rationale, before anything is scored. Note 2026-10-10 (Alex, prompt 48): each case runs with k and eps advected by both schemes, `upwind` and `umist` (step 6's note), to measure whether upwind's numerical diffusion matters anywhere. Note 2026-10-11 (Alex, prompt 50): before VAL-019 and VAL-020, whose inlet turbulence is weak or absent, condition (e)'s scale becomes the molecular nu plus the median of nu_t over the FLUID cells, per outer iteration (ADR-012 E, rule version 5); the VAL-017 run moves from the suite to a harness row, scored by criterion 9's allowance. Note 2026-10-12 (Alex, prompt 52, before VAL-019): the step's inflow boundary layer is grown in a long upstream channel, as the reference grids do, its length set by a precursor run of the channel alone (the distance at which the layer reaches 1.5 H and Re_theta 5,000, plus 4 H, rounded up to a whole H), not prescribed as a profile; and the step runs on the current pressure solve, ADR-013's multigrid successor decided after this case's and the product's timings are in hand (ADR-012 G (ii), note of that date). | step 6; decision 6 |
| 8 | The product room, VAL-018: the configuration moves to the model, the step 3 outlets and `error_estimate`; both variants on the equipment tops and at the sensors; the supply-turbulence sensitivity pair; the y+ map. Phase 3's product-case item. Note 2026-10-09 (Alex; ADR-012 D's note and ADR-013 decision 3 as amended that day): the configuration takes `momentum_sweeps` 10, `pressure_rtol` 1e-4 with one row at 1e-8 as a check, and `max_simple_iter` 10,000. VAL-018's criterion is comparative (criterion 10's note). Note 2026-10-10 (Alex, prompt 48): the product grid is 320x120, which represents the room exactly (every stated position is a multiple of 0.05 m, where 200x75 rounds fourteen of them; `docs/reports/ecr002_step6_comparative.md`, section 5.1), and 160x60 is its coarser check; the finer exact grid, 640x240, waits for Phase 6. VAL-018 runs on 320x120, with k and eps advected by upwind (the turbulence section's default, step 6's note). Note 2026-10-11 (Alex, prompt 49): the product runs the standard variant until VAL-020 (the impinging jet) says which model is right at the equipment tops, because RNG converges on neither exact grid under either scheme (prompts 47 and 48). VAL-018's variant clause (both variants on the equipment tops and at the sensors) waits for VAL-020, and RNG's cycle is probed only if VAL-020 favours RNG; ADR-012 decision 2 carries the dated amendment. Note 2026-10-15 (Alex, prompt 56, before step 8): three decisions. (1) The product runs the standard variant and VAL-018's variant clause is closed (ADR-012 decision 2 as amended: RNG matched the measured turbulence at VAL-020's impingement point where the standard model reads 26 to 35% high, but its free jet was worse and it has not converged on the product's grids); the product carries the caveat that turbulence over the equipment tops reads about 30% high, inflating turbulent mixing and absolute deposition there in every layout alike. (2) VAL-019 and VAL-020 are report-only rows, status REPORTED (criteria 8 and 13). (3) The product's configuration moves into `configs/clean_room_default.yaml` (ADR-013 decision 3 as amended): 320x120, the turbulence section (standard, `upwind`, `cfl_number` 0.25, the inlet at 5% and 0.1 m), `stopping_rule: error_estimate`, `momentum_sweeps` 10, `pressure_rtol` 1e-4, `max_simple_iter` 10,000, `alpha_velocity` 0.5, the multigrid default, every value as prompts 46 to 48 ran it. The step's criterion set, layouts, score and predictions are criterion 10's note and `docs/reports/ecr002_step8_product.md`. Note 2026-10-15 (prompt 56, measured): built and measured; the product configuration is `configs/clean_room_default.yaml` with its harness family `product` and preset `val018_320x120`, the two layouts beside it. L0 (the room as it is) and L1 (return 3 closed) stop by the rule on both exact grids (L0 at 4,389 outer iterations on 320x120, the harness row `val018_320x120` at the same count, pass true); L2 (the supply over the left half of the ceiling) stops on neither within the file's cap, a cycle beside the door wall on 160x60 and on 320x120 a still field that only condition (e) refuses, which stops at 10,794 with the cap raised (a labelled supplement). The score (deposition on the litho tool's and the etch chamber's tops, summed over three sources, per class) ranks L2 < L1 < L0 in every ordering, on both grids and at Sc_t 0.2, 0.7 and 1.3, every margin several times the grid's and the Schmidt number's change to the scores; the largest deposition location is the same 0.2 m segment across grids and Schmidt numbers in all 18 rows and at least three of five hotspots are shared in 17. The stability clause of criterion 10 is met; the predicted order (L2 last, L0 and L1 near-equal) fails, because the score rewards the layout that sweeps the litho-top emission across the room to the hood, past the hood entry where a sensor reads it (criterion 10's note of that date). The y+ map and both variants on the tops are not part of the step as decided (decision 1 of 2026-10-15 closed the variant clause; y+ is the converged table of every room). | step 6, step 5's outcome; ECR-003 (decision 5) Note 2026-10-15 (Alex, prompt 58): the comparative claim is stated per metric, not by one combined score (criterion 10's note of that date); the product's `max_simple_iter` is 15,000 in `configs/clean_room_default.yaml`, so L2 on 320x120 (the rule's stop at 10,794) is inside the cap, with the layout files carrying the same key; L2's 160x60 cycle stays a recorded non-convergence. |
| 9 | Records: ADR-012 accepted with planned against built, SYSTEM.md, the plan, this request closed. | steps 1 to 8 |

## 9. Acceptance Criteria

The change is accepted when all of the following are demonstrated:

1. **Laminar limit.** With the model off, `val001_80x40`, `val001_80x40_stretched` and
   `val002_80x80` reproduce their rows' outer counts and stops, with face hashes equal to the
   main branch at the step's base, and the transport gate tests pass unchanged. If ECR-003 lands
   first it changes every laminar bit; the rows are retaken under it and become the baseline
   (premise review S11). Method: test and harness rows.

   *Note 2026-10-07: the baseline (ECR-003 step 2).* ECR-003 landed first. The baseline is the six
   `staggered-cg` rows taken at 311034e, main after ECR-003 step 1: ae24b120 and 63d655ba
   (`val001_80x40`, 1,559 outer iterations), a88453c2 and a8ac6dd8 (`val001_80x40_stretched`,
   1,124) and 284d7844 and b00e334c (`val002_80x80`, 12,814), each stopping by
   `error_estimate_and_continuity`, accepted by Alex on 2026-10-07. The face hashes, their
   definition and the saved fields are in `docs/reports/ecr003_step2_baseline.md`, sections 2.3
   and 4. The hashes are machine-specific: CG's three reductions go through OpenBLAS, which picks
   its kernel by CPU, and another kernel sums in another order and changes the bits (section 7
   there). The base's and the branch's hashes must come from the same machine, and the saved
   hashes belong to the machine that took them, the AMD Ryzen AI 9 HX 370 the rows record. On
   another machine the base is solved again there, at the step's base commit.

   *Note 2026-10-14: the baseline again (prompt 54).* Multigrid became the default pressure
   preconditioner, which changes every laminar bit once more. The rows are retaken under it
   (`benchmarks/results.jsonl`, method `staggered-mgcg`, commit 759b3ad) and the face hashes
   of `val001_80x40`, `val001_80x40_stretched` and `val002_80x80` under the default are in
   `docs/SYSTEM.md` (REQ-S16) and `docs/reports/multigrid_default.md`; the `staggered-cg`
   rows and hashes above are the diagonal's and are kept, and reproduce bitwise under the
   solver key `pressure_preconditioner: jacobi`. Later steps reproduce the new rows.
2. **Decaying turbulence (VAL-015).** In a closed box at rest, k and eps follow the exact solution
   of `dk/dt = -eps`, `deps/dt = -C_2 eps^2 / k` with an error that falls at first order in dt,
   observed order 0.9 to 1.1. Method: test.
3. **Positivity (REQ-S15).** k > 0 and eps > 0 in every non-SOLID cell after every outer
   iteration of every test and harness solve. Method: an assertion in the step, and unit tests.

   *Note 2026-10-09 (prompt 45, step 6):* the coupled solve calls the step's assertion every outer
   iteration and stops on it naming the iteration (`tests/test_coupled_solve.py`). It held in every
   VAL-016 solve that ran with its pressure corrections converged. The argument assumes corrected
   faces that close every cell; on 48 rows with the corrections capped at 5,000 CG iterations it
   failed at outer iteration 6 in the standard run, which ADR-012 C's note of that date records
   (`results/builder45/couette/positivity_5000.json`); an RNG run reported in the pull request
   stopped at 5, but its log was overwritten, so no record shows it. Corrected 2026-10-09 (review 45 B2):
   "outer iterations 5 and 6" stood here, which no record shows.
4. **Plane Couette flow (VAL-016).** In the developed section, k within 1% of `u_tau^2 /
   sqrt(C_mu)` across the core at Re_tau about 3,000, with u_tau from the constant stress; u /
   U_w and k / u_tau^2 against a one-dimensional solve of the same model and wall treatment
   within an allowance measured on the coarser of two grids and stated before the finer runs, the
   gap falling under refinement. The plane channel's skin friction against Dean (1978) reported.
   The first version's channel criterion, 3% on the log-layer slope and on k, failed a correct
   model (ADR-012 G (ii)). Method: test.

   *Note 2026-10-09 (Alex, prompt 45):* split, because under wall functions refinement moves the
   first node's y+ (ADR-012 G (v)) and the 2D core k carries a discretisation excess the model
   does not have. (a) The implementation: the developed 2D profile equals the one-dimensional
   solve on the same grid in u / U_w and k / u_tau^2 to within 1e-4, both grids and both variants;
   a wall-cell production scaled by 1.01 fails it. (b) The model: the refined one-dimensional
   solve's core k within 1% of `u_tau^2 / sqrt(C_mu)`, both variants, each against its own C_mu.
   (c) Reported, unscored: the 2D core-k excess and the first node's y+ on every grid. Met: (a) on
   12 and 24 rows, both variants, at most 8.5e-7 in u and 7.6e-5 in k; (b) within 0.3%;
   (c) 6% to 15% on 12 rows and 2% to 4% on 24 (and 0.4% to 1.4% on 48 from the same-grid
   one-dimensional solve; the 2D 48-row runs were too slow to finish), from the second-cell strain
   overshoot (ADR-012 C's note). The plane channel's skin friction is 0.95 to 0.98 of Dean's. `tests/test_turbulent_channel.py`
   runs (a) on one grid and one variant and (b) for both; the matrix is
   `docs/reports/probe45/`. Method: test and demonstration.
5. **Transport with turbulence.** With an eddy-viscosity field, VAL-007's relative residual below
   1e-4, VAL-012's departure below its bound, and REQ-T12's bounds on Smith-Hutton. Method: test.
6. **The outlets (step 3, REQ-S18).** VAL-001 and VAL-002 bitwise. The outlet ladder of the
   report's section 8 rerun with T3 as built reproduces, to rounding over the iterations the
   probe ran, the residual history of the probe rerun with the hood's tangential velocity held
   at zero as built (`outlet33b.py` with test 33b's `tangD` change). The probe as run kept that
   velocity at zero gradient, which differs by about 1e-6 relative by outer 25 (test 33b, B2). The Re 90 rung does not diverge over 3,000
   iterations. Method: test and report.
   Amended 2026-10-08 (prompt 42): T3 is not built, so the comparison above has nothing to compare. The criterion
   is: VAL-001 and VAL-002 hash identically to the base (the copy rule they run under is unchanged); and the
   built product configuration reproduces probe arm D0 (`docs/reports/ecr002_step3_outlet_probe.md`, section 7.7)
   with the same stops and the face hash equal. On 40x15 at alpha_velocity 0.5, one momentum sweep and
   `velocity_step`, Re 895 stops at 391 and Re 8,950 at 1,632. On the 80x30 drift case (a thousand times
   air's viscosity, ten sweeps) the stop is at 177 at `pressure_rtol` 1e-8, 1e-4 and 1e-2, and with the stop
   disabled the mean pressure changes by less than 1e-12 Pa per outer iteration over 3,000. Method:
   demonstration (`docs/reports/probe42/fixed42.py`) and test.
7. **The convergence measurement (step 5).** Reported, with its outcome and the decision it
   implies for step 6 and VAL-018, before step 6 opens. Method: report.

   *Note 2026-10-09 (prompt 44):* reported, `docs/reports/ecr002_step5_convergence.md`, section 7
   ("What this implies"). The outcome falls between the ECR's second and third: ten sweeps
   converge the uniform top of k-epsilon's range on 200x75 and nothing else there, and nothing
   below about 5e-4 m^2/s on 80x30 at any sweep count measured (one, ten and fifty), so step 6
   needs decisions from Alex on the sweep count, the corner rule and a stronger aid before it
   opens. The report decides none of them.
8. **The backward-facing step (VAL-019).** OPEN: the reattachment length and the profiles at
   x/H = 1, 4, 6 and 10 against Driver and Seegmiller, the threshold set by Alex from a sourced
   range of standard k-epsilon results on this step (ADR-012 G (ii), decision 6). Method: test
   and report.
   *Note 2026-10-13 (prompt 52, measured; status stays OPEN until Alex sets the allowance).* The
   first coupled results are `docs/reports/ecr002_val019_backstep.md`: the case
   (`configs/validation_backstep.yaml`, presets `val019_410x40`, `val019_656x60` and
   `val019_984x90`, the upstream channel 114 H long from the precursor run), the Resource's data
   with its provenance (`validation/backstep.py`, `validation/data/driver_seegmiller_1985/`), the
   runs (standard with `upwind` on three grids, `umist` on the medium, RNG on the medium), the
   comparison tables and figures, the harness row `val019_656x60` (reported, not scored) and
   `tests/test_backward_step.py`. The grids are what the committed mesh can express (x uniform, y
   clustered toward the walls with a face on the step height), not the stretched grids the prompt
   asked for (the report, section 2.4). The coarse grid converges by the rule; the medium grid is
   bounded at the cap in a cycle at the reattachment cell; RNG diverges at the pressure outlet.
   The standard model with wall functions reattaches at x_r / H 4.86 on the medium grid, 22%
   short of the measured 6.26 and shorter than the NPARC archive's low-Reynolds Chien k-epsilon
   (5.3); above the step's top the profiles are within 0.05 U_ref of the data, the reversed flow
   weaker than measured; the two schemes agree to 0.04 H and a tenth of the model's error. The
   report's section 7 puts the allowance, the grid, the stop on a bounded cycle, RNG's outlet
   and the cost to Alex.
   *Note 2026-10-15 (Alex, prompt 56): status REPORTED.* A report-only validation row: neither
   a readable published standard k-epsilon reference on this case nor a clean stop exists (the
   medium grid's cycle at the reattachment cell), so an allowance set now would be fitted to
   the result. The row records its known departures (the bubble about 20% short on the medium
   grid, half of it the grid) and the harness row `val019_656x60` pins them against regression.
   ADR-012 G (ii) carries the note.
9. **The Annex 20 room (VAL-017).** OPEN: the threshold set by Alex from the departure of the
   published standard k-epsilon prediction (Rong and Nielsen 2008) at the lines decision 6
   scores, or the spread across that report's models (ADR-012 G (iii)). Method: test and report.
   *Note 2026-10-11 (prompt 49, measured; status stays OPEN until Alex sets the allowance).* The
   first coupled results are `docs/reports/ecr002_val017_annex20.md`: the case (`configs/validation_annex20.yaml`, presets `val017_216x72` and
   `val017_375x125`), the data and Rong and Nielsen's digitized line with their provenance
   (`validation/annex20.py`), the four runs (standard, `upwind` and `umist`, both grids) and
   RNG on 216x72, the comparison tables and figures, and `tests/test_annex20.py`. No standard
   row stops by the rule. On 216x72 each reaches the cap with the residual falling, conditions
   (b) to (d) met and (a) at its threshold, because condition (e)'s scale, the inlet's eddy
   viscosity, is about twice nu on this case while the room makes several hundred nu (the
   report, section 5.2; a decision for Alex on ADR-012 E's scale); on 375x125 each stalls at the
   cap with the k and eps solves at their sweep cap (issue 58's case). The lines are compared
   at the cap, labelled by class: on the scored lines the
   solver's departure is below the published prediction's, the two schemes differ by a fraction
   of either's distance from the data (prediction (d) holds), and the floor line's departure is
   the model's, shared with the published run. The test pins the 216x72 run's record and expects
   its convergence clause to fail (strict xfail) until (e) is decided.
   *Note 2026-10-11 (Alex, prompt 50): the allowance is set.* On each scored line, x = H and the
   ceiling jet (y = 2.916 m), the RMS of u / u0 against the measurements, over the stations the
   line holds, is no larger than Rong and Nielsen's (2008) published standard k-epsilon RMS on
   that line: 0.070 at x = H and 0.136 on the jet line (the report above, section 5.3). x = 2H
   and the floor stay unscored, reported. The run is a harness row, `val017_216x72` in
   `scripts/benchmark.py`, scored by this allowance; the suite keeps the loader, data, sampler
   and data-hash tests. Method: harness row and report. Condition (e)'s scale changes first
   (ADR-012 E, amended the same day), so that the row can stop by the rule.
   *Note 2026-10-11 (prompt 50, built and measured): PASS.* Under rule version 5 the row stops by
   the rule on the committed case file, and both scored lines are within the allowance
   (`docs/reports/ecr002_rule_e_scale.md`, sections 5.1 and 5.2, and the row in
   `benchmarks/results.jsonl`). It stops at the file's k and eps tolerance, where 1e-13 stops at
   the same outer iteration, so the tolerance stays; the stop is above the old 10,000 cap, so the
   case file's `max_simple_iter` is 14,000 (the report's section 7 asks Alex to confirm it).
   Report 49's floor of the eddy-viscosity step was a pause at its cap, not the tolerance.
10. **The product room (VAL-018).** Conditional on step 5: the product configuration stops by
   `error_estimate_and_continuity` under rule version 5 (amended 2026-10-11, prompt 50) within its
   cap, with `mass_imbalance_tol`
   from ADR-011 G's formula. Method: harness row and the step 8 report.

   *Note 2026-10-09 (Alex, prompt 45):* the product's purpose is comparative, a stakeholder need
   stated in `docs/SYSTEM.md` section 1: the tool says where particles accumulate and how a layout
   change moves that, and does not defend absolute counts. The criterion for VAL-018, set for
   step 8: the deposition hotspots and the ranking of layouts stable under grid refinement, under
   the two k-epsilon variants, and across the turbulent Schmidt number's literature range (0.2 to
   1.3). The settings it runs under are step 8's note in section 8.

   *Note 2026-10-10 (Alex, prompt 48):* VAL-018 runs on 320x120, and "stable under grid
   refinement" compares it with 160x60, the pair that represents the same room exactly (section
   8, step 8's note of that date).

   *Note 2026-10-15 (Alex, prompt 56):* the variant clause is closed. The product runs the
   standard variant (ADR-012 decision 2 as amended that day, from VAL-020), with the caveat that
   turbulence over the equipment tops reads about 30% high, which inflates turbulent mixing and absolute deposition there by a similar factor in every layout, so layout comparisons stand while absolute tool-top deposition reads high. The criterion for step 8: the deposition hotspots and the ranking of layouts
   stable under grid refinement (160x60 against 320x120) and across the turbulent Schmidt
   number's range (0.2 to 1.3), on three layouts (the room as it is; return 3 closed with its
   flow shared by returns 2 and 4; the supply over the left half of the ceiling only), with the
   score, the layouts and the predictions committed before any run in
   `docs/reports/ecr002_step8_product.md`. The product's configuration moves into
   `configs/clean_room_default.yaml` in the same step (section 8, step 8's note).

   *Note 2026-10-15 (prompt 56, measured):* `docs/reports/ecr002_step8_product.md`. L0 (the room as it is) and L1 (return 3 closed) stop by the rule on both exact grids (L0 at 4,389 outer iterations on 320x120, the harness row `val018_320x120` at the same count, pass true); L2 (the supply over the left half of the ceiling) stops on neither within the file's cap, a cycle beside the door wall on 160x60 and on 320x120 a still field that only condition (e) refuses, which stops at 10,794 with the cap raised (a labelled supplement). The score (deposition on the litho tool's and the etch chamber's tops, summed over three sources, per class) ranks L2 < L1 < L0 in every ordering, on both grids and at Sc_t 0.2, 0.7 and 1.3, every margin several times the grid's and the Schmidt number's change to the scores; the largest deposition location is the same 0.2 m segment across grids and Schmidt numbers in all 18 rows and at least three of five hotspots are shared in 17. The stability clause of criterion 10 is met; the predicted order (L2 last, L0 and L1 near-equal) fails, because the score rewards the layout that sweeps the litho-top emission across the room to the hood, past the hood entry where a sensor reads it. The convergence clause
   holds on the product configuration; the comparative clause is measured and stable; the
   report's section 7 puts to Alex whether the score is the product's (one that counts the
   sensors beside the process surfaces), what L2's non-convergence within the cap means for
   the ranking, and at what resolution the claim is stated in the Phase 3 gate report.

   *Note 2026-10-15 (Alex, prompt 58; decisions 2 and 3 before the Phase 3 gate):* the
   comparative claim is stated per metric, not by one combined score. Step 8's single score
   (process-top deposition) ranked the deliberately poor layout best because it sweeps the
   litho-top emission across the room past the hood entry. The tool's job is a robust
   ranking on whatever the designer cares about; a combined score defined now would be
   fitted to the result. The Phase 3 gate report (`docs/reports/phase3_transport_report.md`)
   states the claim on three metrics defined before they are computed
   (`docs/reports/probe58/`), from step 8's records. The product's `max_simple_iter` becomes
   15,000 in `configs/clean_room_default.yaml` (the two layout files carry the same key):
   L2 on 320x120 stops by the rule at 10,794 outer iterations, inside the new cap, and runs
   that stop earlier are unaffected. L2's 160x60 cycle stays a recorded non-convergence. No
   run is repeated; the harness row `val018_320x120` keeps the cap it was taken under.
11. **Records.** SYSTEM.md, PROJECT_PLAN.md, STATUS.md, ADR-012 (accepted, with planned against
   built) and this request updated and committed. Method: inspection.
12. **Review.** Each step's pull request carries its `/cfd-review` and `/cfd-test` reports.
   Method: inspection.
13. **A planar impinging slot jet (VAL-020).** Added 2026-10-09 (Alex, prompt 45). OPEN: the
   threshold set by Alex from first results. Tells the standard model from RNG where a jet strikes
   a surface, which ADR-012 A found no planned case could. Data: Khayrullina, van Hooff, Blocken
   and van Heijst (2017), Exp Fluids 58(4):31, DOI 10.1007/s00348-017-2315-0, with the same
   authors' steady RANS comparison (2019), Eur J Mech B/Fluids 75:228-243, DOI
   10.1016/j.euromechflu.2018.10.003; alternatives closer to the product's regime in ADR-012 G
   (vi). Method: test and report.

   *Note 2026-10-15 (prompt 55, measured; status stays OPEN until Alex sets the allowance).* The
   case is built and measured on case I of Khayrullina et al. (2017), Re 8,000, w_jet 16 mm, h_jet /
   w_jet 22.5, the data digitised from the paper's figures (`validation/impinging_jet.py`; the
   lateral profiles at five of the six heights rest on the paper's Gaussian fit, the open-marker
   curves not being readable). The domain is 68 w_jet wide with fixed-flow outlets over the whole of
   both sides, a choice checked with a 100 w_jet run that moves the centreline by 0.03 V0 toward the
   data. Standard with `upwind` on 272x90 and 408x135, with `umist` on 408x135, and RNG with `upwind`
   on 408x135: no run stops by the rule; all five are bounded at the 20,000 cap in a cycle beside the
   outlets, 3 to 4 w_jet from a side wall, the jet still; RNG's residual is the one still falling at
   the cap (2.7e-6). At the cap the standard model's centreline is within 0.10 V0 (coarse) and 0.13
   V0 (fine) of the data, its potential core 6.5 to 7.6 w_jet against 4.8, its spreading rate within
   0.005 of the data's 0.099, and its k at the stagnation point 1.26 to 1.35 times the PIV estimate
   0.75 I_uv^2 V0^2 (1.9 to 2.0 times the in-plane half-sum), while RNG's is within 1% of it with a
   centreline 0.21 V0 off at worst. The harness row `val020_272x90` reports the same comparison;
   `tests/test_impinging_jet.py` pins the loader, the data and the readers. By the outcome the prompt
   wrote, RNG not converging within the cap means the product stays standard; the report's section
   7 asks Alex whether that stands given RNG's falling residual, and about the allowance, the stop
   on the outlet cycle and the domain width (`docs/reports/ecr002_val020_impinging_jet.md`).
   *Note 2026-10-15 (Alex, prompt 56): status REPORTED.* A report-only validation row, as
   VAL-019: no readable published standard k-epsilon reference on this case, and no run settles
   cleanly (the outlet cycle), so an allowance set now would be fitted to the result. The row
   records its known departures (the core long, the impingement turbulence high) and the harness
   row `val020_272x90` pins them against regression. The variant question it was built to answer
   is decided: the product runs the standard variant (ADR-012 decision 2 as amended; criterion
   10's note).
14. **A trend-level concentration check (VAL-021).** Added 2026-10-09 (Alex, prompt 45). OPEN:
   the threshold set by Alex from first results. The two-dimensional model's high- and
   low-concentration regions against a measured three-dimensional room, compared for trend, not
   point by point: Zhang and Chen (2006), Atmos Environ 40(18):3396-3408, DOI
   10.1016/j.atmosenv.2006.01.014, or Murakami, Kato, Nagano and Tanaka (1992), ASHRAE Trans
   98(1):82-97. Method: test and report.
   *Note 2026-10-15 (Alex, prompt 58; decision 1): CLOSED as a documented gap, threshold
   N/A.* No published measurement fits a 2D slice (`docs/reports/ecr002_val021_feasibility.md`).
   The transport's turbulent spreading stays unvalidated against measurement, a stated limit
   revisited with 3D in Phase 6.

## 10. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| The coupled iteration does not converge at the core's effective viscosity: on 40x15 the laminar solver with one momentum sweep converges under no outlet treatment measured at a uniform viscosity of 1.5e-4 m^2/s (two runs diverge, two grow or oscillate over 550 to 930 iterations), and at 1.5e-3 it stalls, then diverges, under today's outlets (report, sections 4 and 8). With ten sweeps the 40x15 room converges across k-epsilon's range and laminar at real air (step 0's report, section 7), which lowers the risk on that grid and leaves the product mesh to step 5. | Medium to high | High | Step 0 probes it before anything is built and step 5 measures it before the coupled solve; convergence aids ranked there; VAL-018 changes rather than its tolerance (premise review B2). |
| The room has no steady RANS solution: vortex shedding from the equipment edges. | Medium | High | Step 8 measures it; an unsteady solve would be a scope change, reported rather than tuned away (ADR-012 J). |
| Air drawn in through the outlets: reversed faces accompany every divergence measured, and at Re 8,950 the divergence sits at the hood; the fixed-flow hood moves it to the floor returns. | Measured | High | Step 3's treatment, decision 1; it moves and delays the divergence and is not counted on for convergence. |
| Wall functions used below their range in slow corners and at stagnation points. | High | Medium | The scalable form bounds the error; step 8 maps y+ over every wall node. |
| k overproduced where the supply strikes the equipment tops. | High | Medium | No planned case has impingement with measurements (ADR-012 A); step 8 runs both variants on the product and reports their difference over the tops and at the sensors; the Kato-Launder limiter as the follow-on. |
| The Annex 20 data are three-dimensional at x/H = 2.0. | Certain | Medium | Decision 6; the flux check is in the report. |
| The validation thresholds have no sourced spread yet (Annex 20, the step). | Certain | Medium | Decision 6 leaves both OPEN with the route to a source named. |
| The pressure solve's cost makes the product solve impractical. | Certain | High | ECR-003 before step 8 (decision 5). |
| The coupled iteration oscillates through the mu_t feedback. | Medium | Medium | `alpha_turbulence` and the pseudo-time step; condition (e) refuses to stop on it. |
| Transport results sensitive to Sc_t (0.2 to 1.3 in the literature). | High | Medium | Decision 8; the product step reports the sensitivity. |
| Effort underestimated. | Medium | Low | Phase 3's minimum viable portfolio artifact does not need the product case; the steps are separable. |

## 11. Approval

By signing below, approvers confirm that the problem statement is accurate, the selected option
is appropriate given the alternatives, the requirement and ADR changes are consistent with the
architecture, and the acceptance criteria are sufficient to close the change.

| Role | Name | Approval | Date |
|------|------|----------|------|
| Author | Alex Moroz-Smietana | Approved | 2026-10-04 |
| Reviewer | Claude | Approved: premise review 33, `/cfd-test 33` and `/cfd-test 33b` | 2026-10-04 |

---

## Document History

| Date | Change | Author |
|------|--------|--------|
| 2026-10-04 | Proposed, with ADR-012 and the evidence report `docs/reports/product_case_reynolds.md`. Decision 1 taken by Alex; ADR-012's seven decisions open. Premise review and `/cfd-test 33` to follow. | Alex Moroz-Smietana |
| 2026-10-04 | Revised after premise review 33, test 33 and the outlet measurement (report, section 8): section 2 restated, the outlet step (step 3, REQ-S18) and the convergence measurement (step 5) added before the coupled solve, VAL-016 moved to plane Couette flow, VAL-019 the backward-facing step added (Alex, decision 3 of 2026-10-04), VAL-018 made conditional; ADR-012's decisions eight, the outlets first. `/cfd-test 33b` to follow. | Alex Moroz-Smietana |
| 2026-10-04 | After `/cfd-test 33b`: ADR-012's eight decisions taken by Alex as ranked first, and a ninth added, the risk-retirement probe of step 0. REQ-S18 and step 3 name the fixed-flow hood's zero tangential velocity and a closed face's zero-gradient one; criterion 6 compares against the probe rerun with the built condition (test 33b, B2). ECR-003 lands before step 5. Measuring steps commit their predictions before their runs. Text applied by the orchestrator. | Alex Moroz-Smietana |
| 2026-10-04 | Accepted by Alex. SYSTEM.md's requirement and scope text follow in step 0's pull request. | Alex Moroz-Smietana |
| 2026-10-08 | Step 3 built (prompt 42) as fixed-flow outlets for every return and the hood, ADR-012 decision 1 amended by Alex the same day. REQ-S18 (section 5.3), the section 7 rows for `pressure.py` and its neighbours, section 8 step 3 and criterion 6 carry dated amendments; the original text stays beside each as history. | Alex Moroz-Smietana |
| 2026-10-07 | ECR-003 closed (its step 3, prompt 38). Criterion 1 names its baseline: the six `staggered-cg` rows at 311034e and `docs/reports/ecr003_step2_baseline.md`, with the base's and the branch's face hashes to come from one machine. Step 5's row: the dependency on ECR-003 met, and step 0's sweep result, measured under the weighted Jacobi correction, retaken there under CG. No criterion's threshold changed. | Alex Moroz-Smietana |
| 2026-10-09 | Step 5 measured (prompt 44): `docs/reports/ecr002_step5_convergence.md`; notes in section 8 step 5 and criterion 7. No decision taken. Round 2 (prompt 44b): the grid-convergence tables corrected, the bounded rows characterised; section 9 of the report. | Claude (builder), Alex Moroz-Smietana |
| 2026-10-09 | Alex's decisions before step 6 (prompt 45): the product solver settings (`momentum_sweeps` 10, `pressure_rtol` 1e-4 with a 1e-8 check row, `max_simple_iter` 10,000) for step 8; the corner rule kept; the coupled solve first with no aid built in advance. Notes in section 8 steps 6 and 8 and criterion 10 (VAL-018 comparative); criteria 13 and 14 added (VAL-020, VAL-021), thresholds OPEN. ADR-012 D and G and ADR-013 decision 3 carry the dated amendments. | Alex Moroz-Smietana |
| 2026-10-09 | Step 6 built (prompt 45): the coupled solve, the wall functions, condition (e). VAL-016 split by Alex (criterion 4's note); criterion 3's note on positivity under a capped correction. Two findings for the product step recorded in ADR-012 C: the `cfl_number` 0.5 limit cycle and the second-cell strain overshoot. Prompt 46 measures the product room. | Claude (builder), Alex Moroz-Smietana |
| 2026-10-09 | Step 6 measured on the product room (prompt 46): `docs/reports/ecr002_step6_product_coupled.md`; note in section 8 step 6. No decision taken. | Claude (builder) |
| 2026-10-10 | Step 6 measured on the exact grid pair with the comparative check (prompt 47): `docs/reports/ecr002_step6_comparative.md`; note in section 8 step 6. No decision taken. | Claude (builder) |
| 2026-10-10 | Alex's decisions on prompt 47 (prompt 48): k and eps advected by upwind through a turbulence key `advection_scheme`, absent meaning upwind; the product grid 320x120 with 160x60 its coarser check; RNG rerun under upwind on the exact pair. Notes in section 8 steps 6, 7 and 8 and criterion 10; ADR-012 decision 4 and sections C, I and J carry the dated amendments. | Alex Moroz-Smietana |
| 2026-10-10 | Step 6, the key built and the variants measured (prompt 48): `turbulence.advection_scheme`, upwind by default; parity with prompt 47's upwind arm bit for bit; RNG bounded on both exact grids under upwind; `docs/reports/ecr002_step6_variants.md`; note in section 8 step 6. No decision taken. | Claude (builder) |
| 2026-10-11 | Alex's decision before step 7 (prompt 49): the product runs the standard variant until VAL-020 says which model is right at the equipment tops; VAL-018's variant clause waits for VAL-020; RNG's cycle is probed only if VAL-020 favours RNG. Note in section 8 step 8; ADR-012 decision 2 carries the dated amendment. | Alex Moroz-Smietana |
| 2026-10-11 | Step 7, first case, VAL-017 measured (prompt 49, branch feature/ecr002-val017-annex20): the Annex 20 case, its loader and its data built; four standard runs and one RNG run; the comparison against the measurements and the published standard k-epsilon, reported, unscored; condition (e) found unattainable on this case (its inlet scale). Note under criterion 9; `docs/reports/ecr002_val017_annex20.md`. No threshold set. | Claude (builder) |
| 2026-10-11 | Alex's decisions on prompt 49 (prompt 50): the VAL-017 allowance (criterion 9's note), the VAL-017 run moved to a harness row, and condition (e)'s scale the molecular nu plus the FLUID-cell median of nu_t per outer iteration, rule version 5. Notes in section 8 step 7 and criterion 9; ADR-012 decision 6 and section E carry the dated amendments. | Alex Moroz-Smietana |
| 2026-10-13 | Step 7, second case, VAL-019 measured (prompt 52, branch feature/ecr002-val019-backstep): the backward-facing step's case, loader, data and comparison built; the precursor sets the upstream length at 114 H; three standard grids, one umist run and one RNG run; the standard model reattaches 22% short on the medium grid, the schemes agreeing, RNG diverging at the outlet; reported, unscored. Note under criterion 8; `docs/reports/ecr002_val019_backstep.md`. No threshold set. | Claude (builder) |
| 2026-10-11 | Step 7, (e)'s scale built and VAL-017 run as a harness row (prompt 50, branch feature/ecr002-rule-e-scale): rule version 5, laminar stops bitwise unchanged, the Annex 20 row stops at the file's tolerance and passes the allowance, the product rows stop 2.4% sooner with fields unchanged to 7e-9; CI runs the suite once. Criterion 9 PASS; `docs/reports/ecr002_rule_e_scale.md`. | Claude (builder) |
| 2026-10-15 | Step 7, third case, VAL-020 measured (prompt 55, branch feature/ecr002-val020-impinging-jet): the plane impinging jet's case, loader, digitised data and comparison built; standard on two grids and two schemes, RNG on the finer; every run bounded at the cap in a cycle beside the outlets, RNG's residual the one still falling; the standard centreline within 0.10 to 0.13 V0, the spreading rate within 0.005, the stagnation k 1.3 times the PIV estimate where RNG's is within 1%; reported, unscored. Note under criterion 13; `docs/reports/ecr002_val020_impinging_jet.md`. No threshold set. | Claude (builder) |
| 2026-10-15 | Alex's decisions before step 8 (prompt 56): the product runs the standard variant and VAL-018's variant clause is closed (ADR-012 decision 2 and section A, from VAL-020); VAL-019 and VAL-020 are report-only rows, status REPORTED (criteria 8 and 13, ADR-012 G); the product's configuration moves into `configs/clean_room_default.yaml` (ADR-013 decision 3 as amended). Notes in section 8 step 8 and criterion 10; the step 8 criterion set for VAL-018. | Alex Moroz-Smietana |
| 2026-10-15 | Step 8 built and measured (prompt 56, branch feature/ecr002-step8-product): the product configuration into `configs/clean_room_default.yaml` with the harness family and the VAL-018 row; the two layouts; three layouts on two grids, 39 marches at three Schmidt numbers, the score, the ranking, the margins and the hotspots. The ranking L2 < L1 < L0 is stable under grid refinement and across the Schmidt number's range with wide margins; the predicted order fails. Notes under criterion 10 and step 8; `docs/reports/ecr002_step8_product.md`. | Claude (builder) |
| 2026-10-15 | Alex's decisions before the Phase 3 gate (prompt 58): VAL-021 CLOSED as a documented gap, threshold N/A (criterion 14); the comparative claim stated per metric, not by one combined score (criterion 10, step 8); the product's `max_simple_iter` 15,000 (step 8; `configs/clean_room_default.yaml` and the two layout files, with the tests that read the key). ADR-012 G, SYSTEM.md section 1 and the PROJECT_PLAN rows carry the same notes. | Alex Moroz-Smietana |
