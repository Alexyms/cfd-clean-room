# ADR-011: Transport Solver Architecture: Face-Advected Cell-Centred Concentration

## Status
Accepted at merge (PR 30). Written 2026-10-03, before the Phase 3 build, from the staggered solver
as ADR-010 records it, and proposed with three questions open. Alex decided them on 2026-10-03,
with seven further points its premise review and test raised (review 30 and test 30, posted on
PR 30); the decisions are listed first, dated and attributed, and the sections carry them. This
document is what `solver_transport.py`, `boundary_concentration.py`, the Phase 3 validation tests,
Phase 4's `time_integration.py`, Phase 5's monitor, Phase 6's CUDA port and Phase 7's animation
will be built from, and it gains a planned-against-built table at the Phase 3 gate, as ADR-010's
was added at ECR-001 step 9. Bracketed numbers point at the sources listed at the end.

## Decisions taken 2026-10-03

Alex took these on 2026-10-03, on the proposed version of this document, its premise review and
its test; each names the section that carries it.

1. **The face scheme (was OPEN 1).** QUICK's face value bounded by a TVD limiter, forward Euler
   on advection, Courant number at most 1/2. The limiter is UMIST; section B says why. Rejected:
   unlimited QUICK under SSP-RK3 with an undershoot instrument (not positive: VAL-004's tails
   reach -2.5e-3 of the peak [8]) and first-order upwind (positive, and it leaves 34% of VAL-004's
   peak [8]). Rationale: a concentration below zero is physically meaningless and Phase 5 scores
   these fields against ISO limits; the product case runs at Courant number 0.11; the scheme is
   the momentum assembly with one clamp. Sections B and C.
2. **VAL-004's criterion (was OPEN 2).** The structure of the first alternative with measured
   thresholds: L2 error against the exactly translated pulse, peak retained, no cell below zero
   (exact, since the scheme is bounded), peak location within one cell. The two numbers are set
   from the chosen scheme's measurement at the test's own Courant number and travel [7, 8], at
   1.5 times the measured error, and section H records each measurement beside its threshold. A
   rotating puff joins VAL-004 as its second row. Rejected: a fitted numerical diffusion and the
   plan row as it stood ("shape preserved" is not a criterion). Section H.
3. **The supply concentration (was OPEN 3).** The supply is clean by default; particles enter as
   sources. An inlet carries a configured upstream concentration per class, zero by default, times
   one minus the class's HEPA efficiency when the inlet is marked filtered. No recirculation in
   v1. Rejected: recirculation (a feedback across the boundary that the plan's one-way coupling
   does not contain; a scope change under SYSTEM.md section 5) and a clean supply with the
   efficiency unused (contained in the decision). Section E.
4. **REQ-T12, positivity, a new requirement.** Concentration is non-negative at every cell and
   step, and under pure advection never exceeds the largest value present in the field and at its
   inlets. Verified by VAL-013 (Smith-Hutton) and VAL-004's no-negative clause (premise review
   S1). Sections B and H.
5. **`solve_timestep` gains a `sources` argument** (premise review B2): a cell-centred rate array
   for the class, `[ny, nx]` or `None`, which the solver adds over the step and books in the
   budget's source slot. Phase 4's `get_active_sources` has its consumer. Sections C and F, and
   the draft contract in SYSTEM.md section 4.
6. **The product configuration adopts `error_estimate` with the derived tolerance** when the
   product case is measured, already a Phase 3 item (premise review B4). REQ-T11's bound is
   stated for face fields produced under that rule, and section G names the configuration change
   as a precondition and a deliverable.
7. **VAL-014, the sealed box.** A closed room with no flow and a uniform initial concentration
   loses mass at the rate settling velocity over room height; exact, one line. It guards section
   D: a doubled removal doubles the rate. Section H designs it; it joins the gate.
8. **A field-history hook.** The output contract gains a writer that records the concentration
   fields every N steps, for Phase 7's animation; the rotating puff is what animates. Section C.
9. **Status Accepted at merge**, with the planned-against-built table added at the Phase 3 gate
   as ADR-010's was; the former Open items section is this list.
10. **Identifiers.** REQ-T12; VAL-012 constancy (REQ-T11), VAL-013 Smith-Hutton (REQ-T12),
    VAL-014 sealed-box decay. The rotating puff is VAL-004's second row, not a new ID. The
    register pin in `tests/test_system_map.py` widens for REQ-T12.

Accepted as they stand, no action: PR 30's register-pin edit (test 30 S3), the 102-line
`constancy_drift.py` (test 30 S6), section B's stop condition handled as OPEN 1 (test 30,
observations), and this document's length.

## Context
Phase 2 delivered a steady velocity field on a staggered grid whose continuity holds on the
faces: at every validation stop the worst per-cell mass imbalance and the signed domain sum are
below 1e-10 ([5], decision 5; REQ-S04's per-cell and domain-sum clauses). `solve_steady` returns
the cell means of those faces, and the means do not carry continuity ([5], Consequences, For
Phase 3). Phase 3 solves five advection-diffusion equations on that field (REQ-T01, T02) with
settling (REQ-T03), Brownian diffusion (REQ-T04), wall deposition (REQ-T09) and conservation to
0.01% (REQ-T05), explicit in advection under a CFL condition (REQ-N01), with `v_ext` kept
(REQ-T06, ADR-007). The product case is an 8 m by 3 m room on 200x75 cells of 0.04 m, air at
1.2 kg/m^3, a 0.45 m/s HEPA supply across the ceiling, four floor returns, a hood exhaust and
four obstacles [6]. Every validation case so far had unit density and square cells; section H
does not repeat that.

Six decisions were taken before this document (Alex, 2026-10-02 and 2026-10-03) and it builds
on them. **Record of prior decisions** (premise review S11). Two are recorded in
PROJECT_PLAN.md's status row of 2026-10-02: concentration boundary conditions live in
`src/boundary_concentration.py` over `src/boundary_registry.py`, and the adaptive outer iteration
is not a Phase 3 prerequisite. The other four are recorded in this document and in
PROJECT_PLAN.md's status rows of 2026-10-03, and nowhere earlier: the deliverable of PR 30 is
the design, not code; a constancy test joins the gate as REQ-T11 (VAL-012, section G); the
product `mass_imbalance_tol` is derived, not restated (section G); concentration sits at cell
centres and is advected by the face velocities, which `StaggeredSolver` must expose (REQ-S13,
section A). These six and the ten above are decisions; everything else here is the design's.

## Decision summary
1. `StaggeredSolver.face_velocities`, a frozen `FaceVelocities(u, v)` set at the end of every
   solve; `solve_steady`'s return unchanged (A, REQ-S13).
2. Finite volume on the cell-centred field; advective flux per face is the stored face velocity
   times the face concentration times the face length, with no interpolation of the velocity;
   diffusive flux by central differences over the face's centre-to-centre distance (B).
3. Face concentration: QUICK's value from `quick_face_values`, bounded by the UMIST limiter;
   exact on a uniform field; non-negative and bounded under forward Euler at Courant number at
   most 1/2 (REQ-T12); second order in space on smooth monotone data, first order at extrema
   (B; decisions 1 and 4).
4. Forward Euler on advection at Courant number at most 1/2, backward Euler on diffusion and on
   the deposition sink, Jacobi on the diffusion system; `stable_dt(faces, size_class)` from that
   bound; the validation cases run at Courant number 0.1 (C, REQ-N01).
5. `solve_timestep(C_k, faces, size_class, dt, v_ext=None, sources=None) -> ndarray`, the faces
   in place of the planned cell means; `v_ext` a per-class face drift velocity, zero in v1;
   `sources` a cell-centred rate the solver integrates and books; a `FieldHistory` writer for
   Phase 7 (C; REQ-T06; decisions 5 and 8).
6. Settling as a downward increment of the class's vertical face velocity on horizontal faces
   between two non-SOLID cells only; every face that meets a SOLID cell, the domain floor or the
   domain ceiling takes `deposition_velocity` for its orientation as its single removal (D, E;
   REQ-T09).
7. Concentration conditions derived from the registry's velocity type with explicit overrides; a
   clean supply by default, the inlet value times one minus the HEPA efficiency when filtered, no
   recirculation; obstacle faces deposit as oriented walls (E; decision 3).
8. A `MassBudget` the solver updates from the fluxes and sources it applies, summed over
   non-SOLID cells, which the tests read (F).
9. REQ-T11's text, per-cell bound and test (VAL-012); `mass_imbalance_tol <= 1e-4 rho V_min / T`,
   under which the product configuration moves to `error_estimate` (G; decision 6).
10. VAL-003 scaled; VAL-004 in two rows, the oblique channel pulse and the rotating puff, with
    thresholds set from [7] and [8]; VAL-007 on an arbitrary face field; VAL-013 Smith-Hutton for
    REQ-T12; VAL-014 the sealed box for section D (H).

## A. The face-field interface (REQ-S13)
`solve_steady` allocates `u`, `v` as locals, rebinds them to the corrector's arrays each outer
iteration, and returns `to_cell_centers(u, v)`; the final faces survive only in
`last_mass_imbalance`, which is computed from them (`src/solver_staggered.py`, lines 228 to
292). The consumers: the transport solver (the fluxes continuity was enforced on), the constancy
test (the same fluxes, to predict the drift), the harness (the face imbalance beside the stop),
the viewer (the flow drawn from faces rather than means) and Phase 6's CUDA port.

Three forms were considered. A fourth element or a dataclass returned by `solve_steady` changes
the signature every caller in SYSTEM.md section 4 reads and is a decision; a second method that
re-solves or caches is either expensive or the attribute under another name. The design is an
attribute: `face_velocities: FaceVelocities | None`, `None` until a solve completes, set beside
`last_mass_imbalance` at the end of every solve, converged or not. `FaceVelocities` is a frozen
dataclass in `src/staggered.py`, the layout module, with `u` [ny, nx+1] and `v` [ny+1, nx],
float64, C-contiguous, copies owned by the reader with the writeable flag cleared. The contract
promises: `to_cell_centers(faces.u, faces.v)` equals the returned cell means bitwise;
`PressureCorrector.mass_imbalance(faces.u, faces.v)` equals `last_mass_imbalance`; and when
`stop_reason` is `error_estimate_and_continuity`, the worst absolute entry of that imbalance and
the absolute value of its sum are below `mass_imbalance_tol`, REQ-S04's per-cell and domain-sum
clauses. Under `velocity_step` the faces are given with no continuity promise. The IterationState
callback is unchanged. Rejected: reconstructing faces from the cell means in the transport
solver, which gives a field whose per-cell imbalance is O(h^2) and not the one the rule bounded.

## B. The scalar discretization
The concentration of class k, `C_k` [ny, nx], number of particles per cubic metre (the
thresholds are per cubic metre [6]), sits at the pressure locations. The vertical face at
`u[j, i]` is exactly the face between cells (j, i-1) and (j, i), so the volume flux through it is
`u[j, i] dy_cell[j]` with no interpolation; likewise `v[j, i] dx_cell[i]`. The advective flux is
that volume flux times a face concentration `C_f`; the diffusive flux is
`D_k A_f (C_N - C_P) / d_face`, central over the centre-to-centre distance `dx_face`, `dy_face`,
which the mesh supplies for the stretched case (REQ-S11). Cell update: the sum of the four face
fluxes over the cell volume `dx_cell[i] dy_cell[j]`. Every quantity is SI and the depth is one
metre, as it is in the pressure correction. "Non-SOLID cells" below means every cell that holds
concentration: `mesh.cell_type` marks the outermost ring BOUNDARY, and that ring holds
concentration and carries every inlet and outlet face (premise review B5).

**The face concentration.** Three candidates were ranked and Alex chose the third (decision 1).
First-order upwind takes the upstream cell value: bounded, positive, and with an effective
diffusion of about `U dx (1 - c) / 2` that section H shows destroys VAL-004's pulse. QUICK takes
the quadratic through the upstream, downstream and next-upstream nodes, which `quick_face_values`
already evaluates with Lagrange weights from the node positions along either axis
(`src/momentum.py`, line 166), Leonard's boundary form included: second order on the stretched
mesh and the momentum equation's scheme (REQ-S09). It is linear, so by Godunov's theorem it is
not monotone: near a front it produces values below the smaller neighbour, and where that
neighbour is zero, a negative concentration. Nothing in SYSTEM.md said a concentration cannot be
negative; it cannot, a negative value at a sensor is a reading the monitor cannot interpret, and
REQ-T12 now says so (decision 4). The chosen scheme bounds the QUICK value: with `r` the ratio of
the upstream to the downstream difference at the face, `r = (C_C - C_U) / (C_D - C_C)`, the face
value is `C_C + psi(r)(C_D - C_C) / 2` with `psi` inside Sweby's TVD region, `0 <= psi <= min(2r,
2)`. In this notation QUICK's own line is `psi = (3 + r) / 4`.

The limiter is UMIST (Lien and Leschziner 1994), `psi(r) = max(0, min(2r, (1 + 3r) / 4, (3 + r)
/ 4, 2))`: the face value is QUICK's for `r` between 1 and 5 and the limiter lies inside Sweby's
region everywhere, which is what the positivity lemma below needs. SMART (Gaskell and Lau 1988),
`psi(r) = max(0, min(2r, (3 + r) / 4, 4))`, follows QUICK over a wider range, `r` from 3/7 to
13, but its upper limit of 4 is outside the region: under forward Euler the lemma's update
coefficient can be negative, and SMART's boundedness belongs to the steady discrete equation it
was built for, not to an explicit step. On a stretched mesh the quadratic is what
`quick_face_values` returns and the clamp is applied to it against the two neighbouring cell
values. Where the downstream difference `C_D - C_C` is zero the ratio is undefined and the face
value is `C_C`: that guard makes the scheme exact on a uniform field (below), and the build must
have it (test 30, observations). The momentum equation keeps the unlimited QUICK because a
velocity may be negative and its deferred correction is implicit; the scalar scheme is that
assembly with one clamp.

**Positivity and boundedness (REQ-T12).** With `0 <= psi <= min(2r, 2)` the explicit
one-dimensional update is `C_i - c [1 + psi_i / (2 r_i) - psi_(i-1) / 2] (C_i - C_(i-1))` with
the bracket in [0, 2], so at `c <= 1/2` each new value is a convex combination of `C_i` and
`C_(i-1)` (Harten 1983; Sweby 1984) and the field stays within its bounds. In two dimensions the
unsplit update is a convex combination when the sum of the two directional Courant numbers in the
cell is at most 1/2, which is the quantity `stable_dt` bounds (section C). The diffusion step is
an M-matrix solve, the deposition sink is implicit (section C) and a source is non-negative, so a
non-negative field stays non-negative, and under pure advection no value exceeds the largest
present in the field and at its inlets. Positivity cannot be shown for unlimited QUICK on the
VAL-004 pulse, and it fails there (section H). Clipping negative values after the step was
rejected: it creates mass, which VAL-007 counts. VAL-013 tests the bounds (section H).

**Order of accuracy (REQ-N02; premise review S2).** In space the scheme is second order on
smooth monotone data, where the face value is QUICK's, and first order at extrema, where the
limiter returns to upwind; the limited scheme is nonlinear. In time forward Euler is first order,
and its error is anti-diffusive, `-(c / 2) U dx` in the modified equation, which is why unlimited
QUICK is unstable under it (section C) and why the limited scheme steepens a pulse as the Courant
number rises: on the 1D Gaussian the L2 shape error is 8.6% at c = 0.1, 22% at 0.25 and 31% at
0.4 over the same travel [7]. Under refinement at a fixed Courant number the time error is first
order in `dx` and dominates. VAL-008 (Phase 4) therefore measures the transport solver two ways
and says which: at a fixed Courant number, where the expected order is one, and with the time step
proportional to `dx^2`, where the spatial order shows: two on VAL-003's Gaussian (backward Euler
at a fixed diffusion number is second order in `dx` for the same reason) and on VAL-004 away from
the peak, one at the peak, stated as such.

**Exactness on a uniform field.** Every candidate's face value is an affine combination of node
values with weights summing to one (Lagrange weights do; the clamp returns a value between
`C_C` and `C_D`, and the guard returns `C_C` where they are equal), so on a uniform field
`C_f = C` at every face, and the net advective flux out of cell P is `C` times its net volume
flux, `C b_P / rho` with `b_P` the per-cell mass imbalance of `pressure.mass_imbalance`. On a
divergence-free face field that is zero to rounding; on the rule's field it is the drift section
G bounds. The diffusive flux of a uniform field is zero exactly, and a uniform field satisfies
the implicit diffusion system with zero residual, so a partially converged Jacobi solve does not
disturb it either. That is what REQ-T11 tests.

## C. Time integration and the `solve_timestep` contract
Phase 4's `time_integration.py` owns the loop and the step; this solver advances one class by
one step and reports the step it can take. The plan asks for explicit advection and implicit
diffusion (Phase 3 risks). Three facts set the form.

First, forward Euler with the unlimited QUICK face value is unstable at every Courant number. For
one-dimensional advection the amplification factor is `g = 1 + z` with `z = -(c / 8)(3 e^{i th}
+ 3 - 7 e^{-i th} + e^{-2 i th})`; its real part is `-(c / 4)(1 - cos th)^2`, fourth order in
`th`, its imaginary part first order, so `|g|^2 = 1 + c^2 th^2 + O(th^4)` and the long waves
grow whatever `c` is: max |g| is 1.001 at c = 0.1 and 1.093 at c = 0.5, against 1.000 for upwind
[3]. QUICKEST (Leonard 1979) exists because of this. Third-order SSP Runge-Kutta with the same
QUICK operator is stable to c = 1 [3], and a TVD-limited face value under forward Euler is
stable at c <= 1/2 by the lemma that gives positivity (section B). Decision 1 fixes the
integrator: forward Euler with the limited face value at a Courant number of at most 1/2, which
`stable_dt` enforces as the sum of the directional Courant numbers in the worst cell; SSP-RK3 and
QUICKEST were the unlimited scheme's integrators and go with it. The bound is a stability bound,
not an accuracy one: the forward Euler error steepens a pulse in proportion to `c` (section B,
order of accuracy), so the validation cases run at Courant number 0.1, the product case's 0.11
(section H), and `cfl_number` is an accuracy choice below the bound. The momentum predictor's
deferred correction does not apply, since there is no implicit advection matrix to defer against;
`quick_face_values` is reused for the face value, not the mechanism.

**Recorded alternative, not chosen (orchestrator and Alex, 2026-10-03).** The steepening is forward
Euler's, not the limiter's. Heun's method, the two-stage second-order member of the SSP
Runge-Kutta family Shu and Osher (1988) describe, is a convex combination of two forward Euler
steps, so a face value that is TVD under forward Euler at `c <= 1/2` stays TVD and positive under
it at the same bound, and its time error is second order: the pulse no longer squares as `c`
rises, and the scheme is second order under refinement at a fixed Courant number, so VAL-008
would not need to refine the time step faster than the mesh. The cost is two advection
evaluations per step. It is not chosen here because the product case runs at `c = 0.11`, where
forward Euler's error is small (section H), and because the choice of step cost belongs to Phase
6, where the step is paid for on the GPU. If a larger step is wanted there, this is the change to
make, and it touches `solve_timestep` only.

Second, Brownian diffusion is unresolved everywhere in the bulk of the product case. The cell
Peclet number `U dx / D` is 2.6e7 for the 0.1 um class and 3.7e9 for 5 um, and the diffusion
number `D dt / dx^2` at the configured step is 4.3e-9 and 3.1e-11 [2]. Explicit diffusion would
be stable with a step of 5.8e5 s [2]; implicit diffusion costs one Jacobi sweep there, each sweep
contracting the error by about `4 d / (1 + 4 d)` with `d` the diffusion number. The design keeps
backward Euler on diffusion, solved by Jacobi because Phase 6's kernel form is a per-cell sweep
(ADR-005; section I's three loops); a banded direct solve or conjugate gradient would converge in
fewer sweeps on VAL-003's resolved coefficient and has no data-parallel form the port could keep
(premise review S7). VAL-003 runs a scaled coefficient at a diffusion number of order one
(section H), a clustered product mesh (ADR-010, Consequences, For Phase 3) could put
sub-millimetre cells at the walls, and the deposition sink `v_d C_P A_f / V_P` belongs in the
same implicit diagonal so that it can never overshoot zero. Diffusion matters through `D / delta`
at the walls (section D) and nowhere else.

Third, settling is a boundary-layer and residence-time effect: the 5 um settling velocity is
7.78e-4 m/s, 0.17% of the supply velocity, and a particle falls the room height in 3860 s
against a residence time of 6.7 s [2]. The step is set by the air.

**Contract (draft; SYSTEM.md section 4 carries it marked planned).**

```
TransportSolver:
    __init__(mesh, config, physics: ParticlePhysics, boundary: ConcentrationBoundary)
    stable_dt(faces: FaceVelocities, size_class: int, v_ext=None) -> float
        the largest dt for which the explicit advection step of this class is
        stable: cfl_number / max over non-SOLID cells of
        (max(|u_w|, |u_e|) / dx_cell + max(|v_s|, |v_n|) / dy_cell), the face
        velocities carrying the class's settling increment and v_ext;
        cfl_number from the configuration, at most 1/2 (section B)
    solve_timestep(C_k, faces, size_class, dt, v_ext=None, sources=None) -> ndarray
        C_k [ny, nx] float64 contiguous, not modified; faces: FaceVelocities;
        v_ext: FaceVelocities-shaped per-class drift increment or None (zero);
        sources: [ny, nx] rate of class k, particles per cubic metre per second,
        non-negative, or None (zero): added as sources * dt over the step and
        sum(sources * V) * dt booked in the budget's source slot
        ValueError if dt > stable_dt, on a shape mismatch, or on a source in a
        SOLID cell
        returns the new C_k, [ny, nx], float64, contiguous; SOLID cells zero;
        updates self.budget[size_class] from the fluxes and sources applied (F)
    budget: list[MassBudget]   # one per class, read by tests and Phase 4

FieldHistory:                  # the output contract for Phase 7 (decision 8)
    __init__(every: int)       # every = config.output_interval
    record(step: int, t: float, fields: dict[int, ndarray]) -> None
        keeps a copy of every class field when step % every == 0
    frames: list[tuple[int, float, dict[int, ndarray]]]
    save(path) -> None         # one npz: steps, times, and per class C as
                               # [n_frames, ny, nx]
```

The planned signature took `u, v` cell means; the faces replace them because the means do not
carry continuity (section A) and the vertical face velocity is where the settling increment is
added. `v_ext` is kept as REQ-T06 and ADR-007 require. It is a drift velocity, not a force: in
the Stokes regime a body force on a particle becomes a drift velocity through the class's
mobility, which is how `settling_velocity` already turns gravity into a velocity, so a force
field is converted once by its owner with `ParticlePhysics` and the solver receives faces it can
add. It is `None`, read as zero, in v1; a non-uniform `v_ext` has divergence, which is the
physics an electrostatic precipitator would add.

Sources enter through the `sources` argument (decision 5): `time_integration.py` turns Phase
4's `SourceTerm` objects into one `[ny, nx]` rate per class and passes it. The solver adds
`sources dt` to each non-SOLID cell after the advection update and before the implicit diffusion
and deposition solve, so a source in a floor cell deposits in the same step, and books
`sum(sources V) dt` as the step's source mass. The earlier draft had the caller add the rate
around the step, which left the budget's source slot with no writer (premise review B2).

The field history (decision 8) is a `FieldHistory` writer in `solver_transport.py`, which owns
the output contract of concentration fields. The solver never calls it, since it does not own the
loop: `time_integration.py` (Phase 4) calls `record` once per time step after the five classes
advance, with `every` from the existing `output_interval` key, and Phase 7's animation reads
`frames` or the saved file. `tests/test_advection.py` records the rotating puff of VAL-004 row 2
through it, so the format has a Phase 3 consumer and Phase 7 its first animation input.

## D. Settling
Per class the Stokes settling velocity `v_s` from `ParticlePhysics.settling_velocity` is a
downward velocity (REQ-T03). The solver forms the class's advecting vertical face velocity
`v_k = v - v_s + v_ext_v` on every horizontal face between two non-SOLID cells, and nowhere
else: not on the domain floor or ceiling, and not on any face that meets a SOLID cell. On the
faces that carry it a uniform increment has zero discrete divergence, the north and south terms
cancelling cell by cell, so constancy survives it. Every face the increment leaves out takes
`deposition_velocity` for its orientation as its single removal (section E): floor-type where
fluid sits above a solid or above the domain floor, which includes settling (REQ-T09);
ceiling-type where fluid sits below a solid or below the domain ceiling; wall-type at vertical
faces. So the top of an obstacle removes `(v_s + D / delta) C_P` through one flux, the floor
rule, and nothing else. The earlier draft added the settling flux there as well and removed
settling twice on every obstacle top (test 30 B1, premise review B1), which VAL-007 could not see
because both removals were booked as deposition; VAL-014 (section H) now guards the composition,
since a doubled removal doubles its rate.

The trap is REQ-T09: the floor deposition velocity `deposition_velocity(k, "floor")` is
`v_s + D / delta`, "includes gravitational settling", so a settling increment at a floor face
beside the floor deposition flux would remove each unit of mass twice. Two compositions were
considered. (i) Increment everywhere, and the floor condition hands the solver
`deposition_velocity - settling_velocity`, the diffusive part alone. (ii) Increment on faces
between two non-SOLID cells only; every other horizontal face takes the wall flux `J = v_d C_P`
with `v_d` the public `deposition_velocity` for its surface, whole. The design is (ii), at the
domain floor, the domain ceiling and every obstacle face alike: it reads REQ-T09 as written, uses
the public method unchanged, and keeps one rule for where mass leaves the fluid: at a boundary
face, by the boundary condition, and nowhere between two fluid cells. At the ceiling
`deposition_velocity(k, "ceiling")` is `D / delta`, nothing enters by settling since there is
nothing above, and the increment is zero there. At a HEPA inlet on the ceiling the inflow is the
face flux times the inlet concentration, and at the four floor returns of the product case,
`pressure_outlet` segments on the bottom edge, the outflow is the face flux times the upwind
value; neither carries the increment, so both stay the flux the registry prescribes, and what is
dropped is the settling contribution, 0.17% of the supply velocity for the 5 um class and less
for the others [2] (premise review S12). The 5 um column of [2] shows what this composes: at the
floor `v_d` is 7.78e-4 m/s
and `D / delta` is 4.9e-9 m/s, settling to five figures; for 0.1 um the parts are 8.8e-7 and
6.9e-7, comparable. Traces to REQ-T03 (the velocity per class), REQ-T05 (each floor flux is
booked once, as deposition) and REQ-T09 (the floor value includes settling and the solver does not
add it again).

## E. Boundary conditions and `boundary_concentration.py`
The registry stays the one interpretation of which segment covers a point and what type it is
(REQ-S12.1, now with two readers). `boundary_concentration.py` reads it and derives the scalar
condition from the velocity type, with these optional segment keys, validated at load (REQ-C02):
`concentration`, one non-negative value per class, particles per cubic metre, on a
`velocity_inlet` only, default all zero; `hepa_filtered`, a bool on a `velocity_inlet` only,
default false, under which the carried value is `concentration` times one minus
`hepa_efficiency(k)`; `deposition_surface`, one of `floor`, `ceiling`, `wall`, `none`, on a
`wall` only, overriding the edge default (bottom is floor, top is ceiling, left and right are
walls). A key on a segment of the wrong type is a load error, as an unknown solver key is.

The supply is clean by default; particles enter as sources (decision 3). An inlet carries a
configured upstream concentration per class, zero by default, times one minus the class's HEPA
efficiency when the inlet is marked filtered. No recirculation in v1. Sources (a dirty object on
a table, a person, a leak) are Phase 4 scenario terms and reach the solver through the `sources`
argument (decision 5). The efficiency sits in the data path REQ-T10 names, and Phase 4's filter
breach scenario changes the upstream value or the efficiency. The product supply is
`hepa_filtered: true` on `hepa_supply` with `concentration` absent, which is a clean supply.

The derived conditions. **Inlet:** the face flux is known and inward; the face concentration is
the carried value, so the inflow of class k is `|u_f| A_f C_in` and nothing diffuses across.
**Outlet:** the face value is the upwind cell's, no diffusive flux; where the corrected outlet
face velocity points inward the entering air is clean, and a count of reversed faces is exposed
for the harness. **Wall:** no advective flux (the face velocity is zero exactly, REQ-S12) and a
deposition flux `J = v_d C_P A_f` with `v_d` from `deposition_velocity(k, surface)`, treated
implicitly (section C), booked per surface. **SOLID cell faces:** every face between a non-SOLID
and a SOLID cell is a wall of the orientation its normal gives, top of an obstacle a floor,
underside a ceiling, sides walls, with `deposition_velocity` for that surface as its single
removal (section D), booked as obstacle deposition. Zero flux at obstacle faces was rejected: the
settling flux from the cell above an obstacle would have nowhere to go and mass would pile in
that cell. ADR-010 says obstacle accuracy was not a Phase 2 target; this is the domain-edge rule
applied to interior faces, not a refinement.

The module's contract: `ConcentrationBoundary(mesh, config, physics, registry)` with
`faces_for(size_class) -> ConcentrationFaces`, a frozen dataclass of face-shaped read-only
arrays: `inflow_u` [ny, nx+1], `inflow_v` [ny+1, nx], the concentration an inward flux carries
(zero except at inlets); `deposition_u`, `deposition_v`, the deposition velocity at wall faces,
domain and SOLID, zero elsewhere; `surface_u`, `surface_v`, small integers naming the surface
each depositing face is booked to; `settling_v` [ny+1, nx], the mask of horizontal faces that
carry the settling increment (section D). There is no `settling_u`: settling acts in -y (premise
review S6). Nothing here reads a concentration field or writes one.

## F. The mass budget instrument
VAL-007 and the deposition-as-tracked-sink test need one accounting, so the solver keeps it and
the tests read it. `MassBudget`, one per class, in `solver_transport.py`: `initial` (set when a
field is first stepped), cumulative `inflow`, `outflow`, `deposited` by surface name (`floor`,
`ceiling`, `wall`, `obstacle`), `source` (written by the solver from the `sources` argument,
decision 5), and `in_domain(C_k, mesh)` which sums `C V` over non-SOLID cells; the BOUNDARY ring
holds concentration and carries every inlet and outlet, so a sum over `FLUID` alone would omit
its mass (premise review B5). `residual()` is `initial + inflow + source - outflow - deposited -
in_domain`, and `relative()` divides by `initial + inflow + source`. The solver increments the
cumulative terms from the same face flux arrays it subtracts from the cells and the same source
array it adds, in the same step, so the test and the solver cannot disagree about what was
counted; nothing else writes them. The alternative, each test computing its own budget from the
fields and the boundary data, was rejected: two accountings can disagree about what was counted,
and a test could then pass on its own bookkeeping (test 30 S4).

The telescoping argument of decision 3, as the budget reads it: every interior face flux is
subtracted from one cell and added to its neighbour, so the sum over cells of the flux
divergence is the sum of the boundary face fluxes exactly, for any face velocities whatever,
divergence-free or not. The residual is therefore rounding, about the number of steps times the
machine epsilon times the mass, and VAL-007's 0.01% tests the flux assembly: that each face
appears in exactly two cells with opposite signs and that every boundary flux booked is the one
applied. It does not test the velocity field. That field's imbalance moves mass between cells
of a uniform haze at the rate `b_P / (rho V_P)`, which is constancy, REQ-T11, section G. The two
instruments are kept apart on purpose.

## G. REQ-T11 and the tolerance derivation
**Text.** REQ-T11: A spatially uniform concentration field advected by a face velocity field
that meets REQ-S04's per-cell and domain-sum clauses at its stop, with every inlet carrying the
same concentration and with diffusion, settling, deposition and sources switched off, shall stay
uniform: after a simulated time T the largest over non-SOLID cells of a cell's relative departure
from its initial value shall not exceed the largest over cells of `|b_P| T / (rho V_P)`, where
`b_P` is the cell's mass imbalance in that field and `V_P` its volume. Rationale: the cell means
do not carry continuity; the faces do, to the bound the stopping rule enforces, and a scheme that
is exact on a uniform field (section B) turns that bound into a drift rate. Verified by VAL-012,
`tests/test_constancy.py`. The inlet clause is there because advection alone does not preserve a
uniform field at an inlet: with the default carried value of zero the inlet cells fall toward
zero at the rate of the inflow, a departure unrelated to the imbalance the requirement is about
(premise review B3). The earlier text's second clause, on the product configuration, is now
decision 6 and the deliverable below rather than a requirement clause, since the bound is stated
for face fields produced under `error_estimate` and the product configuration does not run under
it yet (premise review B4).

**The bound.** With `C` uniform, cell P's content changes at `-C b_P / rho` per second (section
B), so its relative departure after T is `|b_P| T / (rho V_P)` to first order in T, and the
requirement bounds the largest of those cell by cell. The coarser `b_max T / (rho V_min)` pairs
the worst imbalance with the smallest cell, which need not be the same cell: equal on a uniform
mesh and an overstatement on a clustered one (premise review S13), and the test measures the
per-cell form anyway. Measured on the rule's own stops from the saved VAL-001 histories [1]: at
the 80x40 stop (outer 3988) the worst cell is 1.10e-12, giving 7.05e-9 per second, 0.0025% per
simulated hour; the signed domain sum is 5.63e-11, a net rate of 1.13e-10 per second over the
domain; at the 40x20 stop (outer 1389) the figures are 4.77e-9 and 1.63e-10 per second. After
the 80x40 stop the net outflow peaks at 8.71e-9 at outer 4359, 1.74e-8 per second [1]; it never
applies, because transport consumes the field at the stop. These reproduce the figures decision
3 was taken on to two figures.

**The test, VAL-012.** `tests/test_constancy.py` solves VAL-001 at 40x20 under `error_estimate`
(23.8 s at the version 3 stop [4]), takes `face_velocities`, sets `C` to one everywhere and the
inlet's carried concentration to one, and steps the advection alone with diffusion, settling,
deposition and sources off for N steps at `stable_dt`, T = N dt of the order of 40 s, so that the
predicted departure (about 2e-7) is six orders above the rounding floor and the second-order term
`(b T / rho V)^2` six orders below it. It asserts the measured largest departure is below the
bound and above a tenth of it, bounding the drift and confirming the mechanism. Its planted
control: one interior face perturbed, whose drift must exceed the bound. VAL-002's cavity, a
closed domain with no inlet where the domain-sum clause holds to rounding and the drift is
per-cell only, is the second field the test may use (test 30 S4).

**The product tolerance.** `mass_imbalance_tol` is absolute, in kg/s per unit depth, and 1e-10
means different things on the two cases: on the product mesh (cells of 1.6e-3 m^2, air at 1.2)
it gives a worst-cell drift of 5.21e-8 per second, 0.019% per hour, and it is 2.65e-11 of the
room's through-flow of 3.78 kg/s against 2.00e-9 of VAL-001 80x40's, 75 times stricter
relatively [1]. REQ-T05's fraction is the budget the haze may drift in the worst cell over a
scenario, so

    mass_imbalance_tol <= 1e-4 rho V_min / T

with T the scenario's `t_end` from the configuration: 3.20e-9 at the configured 60 s and
5.33e-11 for a one-hour scenario on the product mesh [1]. The default 1e-10
(`DEFAULT_MASS_IMBALANCE_TOL` in `src/config.py`, test 30 S2) satisfies the formula at 60 s and
fails it at an hour. The product YAML sets neither `mass_imbalance_tol` nor `stopping_rule`, so
today it runs under `velocity_step` with `max_simple_iter` 500, where the faces carry no
continuity promise (section A) and the bound bounds nothing. The configuration change is
therefore a precondition of REQ-T11 on the product case and a Phase 3 deliverable (decision 6):
`configs/clean_room_default.yaml` moves to `stopping_rule: error_estimate` with
`mass_imbalance_tol` from the formula at its `t_end` and a cap above 500, since VAL-001 80x40
needed 3988 outer iterations [4], when the product case is solved and its stop measured.

**The per-class table**, from `ParticlePhysics` on the product configuration with the mesh
spacing and the supply velocity read from it (instrument: `results/builder30/particle_table.py`
[2]):

| k | d_p (um) | C_c | v_s (m/s) | D (m^2/s) | Pe = U dx / D | v_s / U | D / delta (m/s) | v_d floor (m/s) |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 2.920 | 8.79e-7 | 6.92e-10 | 2.6e7 | 2.0e-6 | 6.9e-7 | 1.57e-6 |
| 1 | 0.3 | 1.577 | 4.27e-6 | 1.25e-10 | 1.4e8 | 9.5e-6 | 1.2e-7 | 4.40e-6 |
| 2 | 0.5 | 1.339 | 1.01e-5 | 6.35e-11 | 2.8e8 | 2.2e-5 | 6.3e-8 | 1.01e-5 |
| 3 | 1.0 | 1.168 | 3.52e-5 | 2.77e-11 | 6.5e8 | 7.8e-5 | 2.8e-8 | 3.52e-5 |
| 4 | 5.0 | 1.034 | 7.78e-4 | 4.90e-12 | 3.7e9 | 1.7e-3 | 4.9e-9 | 7.78e-4 |

U = 0.45 m/s, dx = 0.04 m, delta = 1e-3 m; the Courant number at the configured dt of 0.01 s is
0.1125 [2]. Diffusion is resolved nowhere in the bulk: `sqrt(4 D t)` over an hour is about 3 mm
for the smallest class, a tenth of a cell. So VAL-003 cannot use a physical coefficient on a
room-scale mesh and is scaled (section H); the bulk physics of the product case is advection
with settling, and Brownian motion acts through `D / delta` at the walls only.

## H. The validation tests as designs
Every case has non-unit density, a non-square domain, inputs off grid nodes, and signed
quantities where the sign matters. Numbers here are arithmetic on the stated inputs, shown so
they can be checked, or measurements from the two prototypes [7, 8], stand-ins for the scheme on
uniform grids and not the solver.

**VAL-003, pure diffusion (REQ-T07).** Closed domain 2.0 m by 1.2 m, 200x120 cells of 0.01 m,
rho 1.2, zero face field, deposition off, one class with a synthetic coefficient D = 1e-3 m^2/s
(the operator does not depend on the value; the physical values would need 1e5 s to move a
cell). Initial field: the two-dimensional Gaussian of standard deviation 0.05 m (five cells)
centred at (0.93, 0.61), off every node, given as exact cell averages (error-function integrals)
so the discrete initial mass is the analytical one. Analytical solution: the heat kernel,
sigma^2(t) = sigma_0^2 + 2 D t, amplitude scaled by sigma_0^2 / sigma^2 (Carslaw and Jaeger
1959, section 10.2; Crank 1975, chapter 2). Run to sigma = 2 sigma_0, t = 3 sigma_0^2 / (2 D)
= 3.75 s; the boundary is 5.9 final sigma from the centre, so the wall value is below 1e-7 of
the peak. Metric: L2 norm of the cell error over the L2 norm of the analytical cell values on
non-SOLID cells. Criterion: below 1% (the plan row). The time step is the build's to choose so
that the backward Euler error is below the spatial one; a diffusion number of 0.25 (dt = 0.025
s, 150 steps) is the design's starting point, and the budget must close to rounding over the run.

**VAL-004, pulse advection (REQ-T08; REQ-T12's no-negative clause), two rows.** Both rows run
at `cfl_number` 0.1, the product case's Courant number, because the limited scheme's shape error
grows with the Courant number under forward Euler (section B). Measured over VAL-004's travel
[7, 8], peak retained and L2 error against the exact pulse:

| Courant number | 1D aligned pulse [7] | Oblique channel pulse, row 1 [8] |
|---|---|---|
| 0.1 | 91.5%, 8.6% | 82.2%, 11.3% |
| 0.25 | 98.3%, 22.0% | 90.0%, 19.3% |
| 0.4 | 99.5%, 31.3% | 94.6%, 37.7% |

The peak is kept better at the higher Courant numbers because the time error's anti-diffusion
offsets the limiter's clipping there, and the shape is kept worse because the pulse squares; the
proposed version's 0.4 is withdrawn on that measurement. The unsplit oblique case costs more than
the aligned one, as expected. The thresholds below are 1.5 times the measured error of the chosen
scheme on the test's own geometry at its own Courant number, rounded outward to two figures.

*Row 1, the oblique channel pulse.* Domain 2.0 m by 1.2 m, 100x60 cells of 0.02 m, rho 1.2, a
uniform face field at the product supply speed 0.45 m/s directed 30 degrees above the x axis
(u = 0.390, v = 0.225, so the pulse crosses cells obliquely and no axis is special), D = 0,
settling off, one class. Initial Gaussian of sigma 0.08 m (four cells) centred at (0.31, 0.35),
off every node, as cell averages; the start is 4.4 sigma from the floor and the end (1.18, 0.85)
is 4.4 sigma from the ceiling, so the tail the boundary cuts is below 1e-4 of the peak (the
proposed version's start at 0.23 m was 2.9 sigma from the floor and cut 0.2% of the mass [8]).
Travel 1.0 m along the flow, 2.22 s, 50 cells; `cfl_number` 0.1 is 684 steps. Analytical
solution: the initial pulse translated by U t, as cell averages. Metrics and criteria, with the
measurement of the chosen scheme (UMIST, forward Euler) and the two controls beside each [8]:

| Metric | Measured | Threshold | Controls: unlimited QUICK under SSP-RK3; upwind |
|---|---|---|---|
| Peak retained | 82.2% | above 73% | 95.7%; 33.8% |
| L2 error over the exact pulse's norm | 11.3% | below 17% | 5.4%; 57.6% |
| Minimum | 0 | no cell below zero | -2.5e-3 of the peak in 2780 cells; 0 |
| Centroid error | 0.02 cells | within one cell (REQ-T08) | 0.001; 0.07 |

The centroid of the field is the peak location, since the centroid of a translated Gaussian is
exact and insensitive to the cell quantization an argmax has. The no-negative bound is exact for
a bounded scheme; the test allows rounding, 1e-14 of the peak, so that a subtraction of equal
fluxes cannot fail it. Upwind's effective diffusion `U dx (1 - c) / 2` is 4.1e-3 m^2/s at
c = 0.1, which over 2.22 s grows sigma^2 from 6.4e-3 to 2.4e-2 and lowers the peak to about a
third, which is why REQ-T08's rationale names it.

*Row 2, the rotating puff.* Domain 1.28 m square, 64x64 cells of 0.02 m, rho 1.2, solid-body
rotation about the centre at omega = 1 rad/s set directly on the faces, `u = -omega (y - y_c)`
and `v = omega (x - x_c)`, divergence-free to rounding cell by cell because u varies only in y
and v only in x; D = 0, settling off, one class. Initial Gaussian of sigma 0.08 m (four cells)
centred 12 cells from the axis, 20 cells (5 sigma) from the nearest boundary, as cell averages.
One revolution, 6.28 s, a path of 75 cells; `cfl_number` 0.1 in the fastest cell, at the domain
corners, is 3959 steps, and the puff's own Courant number is about 0.03. Every boundary face is
open (inflow clean, outflow upwind), so the test hands the solver a `ConcentrationFaces` of
zeros. Analytical solution: the initial field. Metrics as row 1, the centroid error a distance [8]:

| Metric | Measured | Threshold | Controls: unlimited QUICK under SSP-RK3; upwind |
|---|---|---|---|
| Peak retained | 75.3% | above 62% | 95.0%; 27.0% |
| L2 error over the initial field's norm | 15.5% | below 24% | 7.4%; 66.7% |
| Minimum | 0 | no cell below zero | -3.9e-3 of the peak in 1851 cells; 0 |
| Centroid error | 0.06 cells | within one cell | 0.00; 0.40 |

The puff is recorded through `FieldHistory` (section C) every `output_interval` steps; the frames
are Phase 7's first animation input, and the puff sees the scheme through every angle.

**VAL-007, mass conservation (REQ-T05).** Two runs on the budget of section F. (i) The flux
assembly on an arbitrary field: domain 1.6 m by 0.9 m, 64x36 cells, rho 1.2, a random face field
that is not divergence-free (seeded), one inlet segment on the left carrying a concentration,
one outlet on the right, deposition on all four walls for the 5 um class with settling, a source
in one interior cell; 500 steps. Criterion: `relative()` below 1e-4 (the plan row's 0.01%),
expected at rounding, about 1e-13, and the test prints both so the gap is visible. (ii) The same
budget on the VAL-001 40x20 converged faces under `error_estimate`, with the same conditions.
The planted control: the test drops one boundary face from the booking and must fail. REQ-T05's
"mass" is particle count here, since `C` is a number concentration; density enters only through
`b / rho`.

**VAL-012, constancy (REQ-T11).** Section G, "The test".

**VAL-013, Smith-Hutton (REQ-T12).** The bounded-advection case of Smith and Hutton (1982): a
field that turns a steep inlet profile through a half circle to an outlet on the same edge, where
an unbounded scheme overshoots at the front and undershoots behind it. Domain 2.0 m by 1.0 m,
100x50 cells of 0.02 m, rho 1.2, with `x' = x - 1.0` in [-1, 1] and `y' = y` in [0, 1]; the face
field is `u = U 2 y' (1 - x'^2)` on the u faces and `v = -U 2 x' (1 - y'^2)` on the v faces with
U = 0.45 m/s, whose discrete divergence is zero exactly on uniform cells (the x difference of u
and the y difference of v are `-4 x_c y_c dx dy` and its negative). The normal velocity is zero
exactly on the left, right and top edges, which are walls with `deposition_surface: none`; the
bottom edge is a `velocity_inlet` for `x'` in [-1, 0] carrying `C_in(x') = 1 + tanh(alpha (2 x'
+ 1))` per unit of a reference concentration, alpha = 10, and a `pressure_outlet` for `x'` in
(0, 1]. The inlet profile runs from `1 - tanh(alpha)`, 4e-9, to `1 + tanh(alpha)`, about 2; the
initial field is `1 - tanh(alpha)` everywhere; D = 0, settling off, no sources; `cfl_number` 0.1.
Run until the largest change per step is below 1e-10 of the inlet maximum or to 20 s, whichever
comes first. Criterion, asserted at every step: no cell below `1 - tanh(alpha)` and none above
`1 + tanh(alpha)`, the smallest and largest values present in the field and at its inlet, exact
up to the rounding allowance of VAL-004. These bounds need no prototype to set. Reported
unscored: the steady outlet profile against the inlet's mirror image, `C_in(-x')`, which pure
advection would reproduce; it measures the limiter's smearing of a front and goes into the
planned-against-built table. Planted control: an unlimited QUICK face value, computed in the test
from the same fields, must break the bounds, so the test can fail.

**VAL-014, the sealed box (section D).** A closed room: domain 8.0 m by 3.0 m, 40x30 cells
(0.2 m by 0.1 m, non-square on purpose), rho 1.2, four walls, no inlet or outlet, a zero face
field, the 5 um class (`v_s` 7.78e-4 m/s [2]), diffusion off, no sources, a uniform initial
concentration C_0 of 1e6 per cubic metre. With nothing entering through the ceiling, settling
carries a front down from it at `v_s` while the floor row still holds C_0, so the floor removes
`v_s C_0 W` per second and the exact line is `M(t) = M_0 (1 - v_s t / H)` until the front
reaches the floor at `H / v_s`, 3860 s. The faces come from the test, floor `deposition_v` equal
to `v_s` and zero elsewhere, so that the line is exact; through `ConcentrationBoundary` the floor
would carry `v_s + D / delta`, 6e-6 above `v_s` for this class [2], and the floor row would no
longer be stationary. Run 30 steps at `stable_dt` with `cfl_number` 0.4 (dt = 51 s, T = 1540 s;
the front has moved 12 of the 30 cells). Criteria: `deposited["floor"]` equals `v_s C_0 W T` to
1e-10 relative; `in_domain + deposited` equals `initial` to rounding; no cell above C_0 or below
zero (REQ-T12 on a step front). Planted control: the floor `deposition_v` doubled must double the
deposited mass to the same tolerance, the double count of test 30 B1 made visible; a build that
adds the settling increment at the floor face fails the primary run by a factor of two.

## I. Configuration and modules
**Configuration (REQ-C01 to C04).** A new `transport` section: `cfl_number`, a float in (0, 1/2],
the limited scheme's bound, with 0.1 the value the validation cases and the product case use;
`advection_scheme`, one of `umist` (default) and `upwind`, so the VAL-004 comparison can run both
from the same code, with unlimited QUICK not offered because forward Euler is unstable with it
(section C); `max_diffusion_iter`, a positive int; `diffusion_tol`, a positive float. The
scheme's stability bound is a module constant, as `JACOBI_WEIGHT` is; the fraction of it a run
uses is configuration. The existing `output_interval` key is `FieldHistory`'s `every`. The
boundary segment keys of section E. Every key is validated for type, range and the segment type
it is allowed on, with bools rejected where numbers are expected (`config.py`, REQ-C02); the
physical constants stay in `constants.py` (REQ-C04). SYSTEM.md section 4 carries the new keys
in the `SimConfig` and `BoundarySpec` contracts, marked planned (premise review S5).

**Modules.** `src/staggered.py`: gains `FaceVelocities` (A). `src/solver_staggered.py`: gains
`face_velocities` (A). `src/config.py`: the keys above. `src/boundary_concentration.py` (new):
derives the per-face concentration conditions from the registry and `ParticlePhysics` and hands
them to the solver as data (E). `src/solver_transport.py` (new): advances one class one step on
the face velocities with the scheme, settling, sources, implicit diffusion and deposition, keeps
the budget, and defines `FieldHistory` (B, C, D, F). `src/particles.py`,
`src/boundary_registry.py`, `src/mesh.py`: consumed, unchanged in contract.
`configs/clean_room_default.yaml`: moves to `error_estimate` when the product case is measured
(G, decision 6). Tests: `test_constancy.py` (VAL-012), `test_diffusion.py` (VAL-003),
`test_advection.py` (VAL-004, both rows), `test_conservation.py` (VAL-007),
`test_smith_hutton.py` (VAL-013), `test_sealed_box.py` (VAL-014), `test_solver_transport.py` and
`test_boundary_concentration.py` (unit), all under `tests/`.

**Cascade.** `solver_staggered.py -> solver_transport, tests/test_constancy.py`: the
`face_velocities` contract of A. `staggered.py -> solver_transport, tests/test_constancy.py`:
`FaceVelocities` is the transport solver's input type, so the layout module is no longer internal
to the velocity solver alone (premise review S5). `config.py -> boundary_concentration,
solver_transport`: the transport section and the segment keys. `boundary_registry.py` and
`particles.py -> boundary_concentration, solver_transport`: the coverage rule, the new optional
keys, the five methods' units and signs. `solver_transport.py -> time_integration, monitor,
Phase 7`: `solve_timestep`'s shapes and `sources`, `stable_dt`, the budget's names,
`FieldHistory`'s record and file. `boundary_concentration.py -> solver_transport`:
`ConcentrationFaces`.

**Phase 6.** The CUDA port targets three loops, each data-parallel over faces or cells with the
class as a further independent dimension: the face value and advective flux per face (the
limiter is a per-face clamp), the explicit divergence update per cell with the source added, and
the Jacobi sweep per cell. The budget's reductions are sums over faces. REQ-N03's reference is
the NumPy solver this document describes.

## J. What this design does not decide
The product mesh: ADR-010 asks for the clustering cost to be measured on the product case before
a mesh is chosen, and deposition's dependence on the wall spacing is why; nothing here fixes it.
The adaptive outer iteration: deferred on 2026-10-02 to an efficiency pass, not Phase 3; this
design consumes the field at the stop and does not depend on what follows it. Sources, events
and the time loop: Phase 4 (`scenarios.py`, `time_integration.py`), which receive `stable_dt`,
the `sources` argument, `FieldHistory.record` and the segment keys as their interface. The
monitor: Phase 5, which reads `C_k` and never writes it (REQ-A06). The CUDA kernels' form: Phase
6, which receives the three loops of section I.

Boundary modifications (premise review S14). A scenario that adds or changes a boundary
segment (a door leak, a breach that changes a segment's type) re-derives the concentration faces,
by a new `ConcentrationBoundary` over a new registry, and requires the velocity field re-solved,
since the staggered layer reads the same registry and the faces the transport solver advects with
must be the ones continuity was enforced on; one that changes only a segment's carried
concentration or efficiency re-derives the concentration faces alone. `get_bc_modifications` is
Phase 4's, and this is its constraint.

## Consequences
**Positive.** The transport solver advects with the fluxes continuity was enforced on, so a
uniform haze drifts at a rate the stopping rule already bounds and the constancy test measures,
not at the O(h^2) rate a reconstruction from cell means would give. The field is non-negative
and bounded by construction, so Phase 5 scores physical values. One budget serves every
conservation claim, sources included. The face scheme reuses the momentum stencil's evaluator
on the stretched mesh. The boundary module and the velocity layer read one registry. Phase 7 has
its animation input from the first Phase 3 test that produces one.

**Negative.** Five classes mean five explicit steps per time step at a Courant number well below
the stability bound, since the forward Euler error grows with it; implicit diffusion costs a sweep
or two per class on the product mesh for no accuracy there. The limited scheme is nonlinear,
first order at extrema, clips a Gaussian's peak by about a fifth over 50 cells of oblique travel
at c = 0.1 [8], and is first order under refinement at a fixed Courant number, so VAL-008 must
refine the time step faster than the mesh to see the spatial order. `face_velocities` doubles the
solver's retained field memory. REQ-T11 ties `mass_imbalance_tol` to the scenario duration, so a
longer scenario tightens the velocity solve, and the product configuration must move to
`error_estimate` for the bound to apply to it.

## Planned against built
To be filled at the Phase 3 gate, as ADR-010's table was at ECR-001 step 9: one row per decision
above, with what was built and where it was measured, and the VAL-013 outlet profile beside it.

## Alternatives considered
Each section names its own. Across the document: fully implicit advection (against the plan's
explicit-advection decision and REQ-N01, which presumes a CFL condition) and a Lagrangian
particle method (the system is Eulerian by SYSTEM.md section 5 and ADR-006). Not ranked in OPEN
1 (premise review S4): Leonard's ULTIMATE limiter, a MUSCL or van Leer TVD scheme and
flux-corrected transport, bounded schemes of the chosen kind that do not reuse `quick_face_values`.

## Sources
Each figure above is from one of these; section H's case numbers are arithmetic shown inline or
measurements from [7] and [8].

1. `results/builder30/constancy_drift.py`, `.json` and `.log`: the drift rates at the VAL-001
   stops and after them, from `results/builder28/signed_envelope_poiseuille_{40x20,80x40}.npz`,
   and the product-mesh evaluation of `mass_imbalance_tol` and the tolerance formula.
2. `results/builder30/particle_table.py`, `.json` and `.log`: the per-class table on
   `configs/clean_room_default.yaml`.
3. `results/builder30/fe_quick_stability.py`, `.json` and `.log`: the von Neumann amplification
   of forward Euler and SSP-RK3 with QUICK and upwind face values.
4. `docs/reports/stopping_rule_evidence.md`, section 10: the stops at outer 1389 and 3988, their
   wall times, and the post-stop oscillation.
5. `docs/ADR/ADR-010-staggered-grid-architecture.md`: decisions 1, 2 and 5; Consequences, For
   Phase 3.
6. `configs/clean_room_default.yaml`: the product case.
7. `results/builder30b/pulse_1d.py`, `.json` and `.log`: the 1D Gaussian pulse (sigma four
   cells, travel 50 cells) under UMIST with forward Euler, unlimited QUICK with SSP-RK3 and upwind
   with forward Euler at Courant numbers 0.1, 0.25 and 0.4; a prototype of the scheme, not the
   solver.
8. `results/builder30b/pulse_2d.py`, `.json` and `.log`: the same three schemes on VAL-004's
   oblique channel pulse and on the rotating puff, unsplit on the staggered faces, at the same
   three Courant numbers; a prototype, not the solver.

Gaskell and Lau (1988), SMART; Lien and Leschziner (1994), UMIST; Sweby (1984), the TVD region;
Harten (1983), the lemma; Leonard (1979), QUICK and QUICKEST; Shu and Osher (1988), SSP
Runge-Kutta; Smith and Hutton (1982), the bounded-advection test case; Carslaw and Jaeger (1959)
and Crank (1975), the heat kernel.
