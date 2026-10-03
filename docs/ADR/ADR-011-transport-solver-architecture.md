# ADR-011: Transport Solver Architecture: Face-Advected Cell-Centred Concentration

## Status
Proposed. Written 2026-10-03, before the Phase 3 build, from the staggered solver as ADR-010
records it. ADR-010 recorded what was built; this document is what `solver_transport.py`,
`boundary_concentration.py`, the constancy test, Phase 4's `time_integration.py`, Phase 5's
monitor and Phase 6's CUDA port will be built from, and it gains a planned-against-built table
at the Phase 3 gate. Bracketed numbers point at the sources listed at the end. Three questions
are open and listed first, so they can be decided in one reading; the design does not choose them.

## Open items for Alex

**OPEN 1 (section B, with C). The face concentration scheme, and with it positivity.** No
linear scheme above first order keeps a concentration non-negative, and forward Euler with the
unlimited QUICK face value is unstable at every Courant number [3]. Ranked:

1. QUICK's face value bounded by a TVD limiter (the SMART and UMIST family): the quadratic from
   `quick_face_values`, clamped to the interval the upstream gradient ratio allows. Equal to
   QUICK on smooth monotone data, non-negative and stable under forward Euler at Courant
   number at most 1/2 (Harten's lemma). Same assembly as the momentum scheme, one clamp added.
2. Unlimited QUICK under third-order SSP Runge-Kutta (stable to Courant number 1 [3]) with an
   undershoot instrument that counts and sizes negative values each step and never clips. Not
   positive; VAL-004's tails go negative at the truncation error.
3. First-order upwind: positive and stable under forward Euler, and too dissipative for VAL-004
   (section H shows the peak falling to about a third over the test's travel).

**OPEN 2 (section H). VAL-004's criterion.** "Shape preserved" is not a number. Ranked:

1. L2 error against the exactly translated pulse below 5% of the pulse's norm, peak retained
   above 90% of its initial value, and no cell below minus 1e-3 of the initial peak (exactly
   none below zero if OPEN 1 chooses the limited scheme), with the peak location within one
   cell as REQ-T08 already says.
2. The scheme's effective numerical diffusion, fitted from the growth of the pulse's second
   moment, below 1% of the upwind value U dx / 2 at the same Courant number.
3. Peak location only, as the plan row reads today, with shape reported unscored.

Each amends the Phase 3 gate row, which is why it is not chosen here.

**OPEN 3 (section E). The supply concentration and recirculation.** REQ-T10 reads HEPA
efficiency as the removal rate "at supply vent boundaries", which presumes air returning to the
filter. Ranked:

1. No return path in v1. An inlet carries a configured upstream concentration per class, zero by
   default, times one minus the class's HEPA efficiency when the inlet is marked filtered. The
   efficiency sits in the data path REQ-T10 names, the default supply is clean, and Phase 4's
   filter breach scenario changes the upstream value or the efficiency.
2. Recirculation: the supply concentration is the outlet-weighted mean outflow concentration of
   the previous step times one minus the efficiency. A feedback across the boundary that the
   plan's one-way coupling does not contain; a scope change under SYSTEM.md section 5.
3. A clean supply always, with `hepa_efficiency` unused until Phase 4. Contained in 1.

## Context
Phase 2 delivered a steady velocity field on a staggered grid whose continuity holds on the
faces: at every validation stop the worst per-cell mass imbalance and the signed domain sum are
below 1e-10 ([5], decision 5, conditions (b) and (d)). `solve_steady` returns the cell means of
those faces, and the means do not carry continuity ([5], Consequences, For Phase 3). Phase 3
solves five advection-diffusion equations on that field (REQ-T01, T02) with settling (REQ-T03),
Brownian diffusion (REQ-T04), wall deposition (REQ-T09) and conservation to 0.01% (REQ-T05),
explicit in advection under a CFL condition (REQ-N01), with `v_ext` kept (REQ-T06, ADR-007).
The product case is an 8 m by 3 m room on 200x75 cells of 0.04 m, air at 1.2 kg/m^3, a 0.45 m/s
HEPA supply across the ceiling, four floor returns, a hood exhaust and four obstacles [6]. Every
validation case so far had unit density and square cells; section H does not repeat that.

Six decisions were taken before this document (Alex, 2026-10-02 and 2026-10-03) and it builds
on them: the deliverable is the design; concentration boundary conditions live in
`src/boundary_concentration.py` over `src/boundary_registry.py`; the adaptive outer iteration is
not a Phase 3 prerequisite; a constancy test joins the gate as REQ-T11; the product
`mass_imbalance_tol` is derived, not restated; concentration sits at cell centres and is advected
by the face velocities, which `StaggeredSolver` must expose (REQ-S13).

## Decision summary
1. `StaggeredSolver.face_velocities`, a frozen `FaceVelocities(u, v)` set at the end of every
   solve; `solve_steady`'s return unchanged (A, REQ-S13).
2. Finite volume on the cell-centred field; advective flux per face is the stored face velocity
   times the face concentration times the face length, with no interpolation of the velocity;
   diffusive flux by central differences over the face's centre-to-centre distance (B).
3. Face concentration by the scheme OPEN 1 decides; every candidate is exact on a uniform field
   (B).
4. Forward Euler on advection, backward Euler on diffusion and on the deposition sink, Jacobi
   on the diffusion system; `stable_dt(faces, size_class)` from the CFL bound of the chosen
   scheme (C, REQ-N01).
5. `solve_timestep(C_k, faces, size_class, dt, v_ext=None) -> ndarray`, the faces in place of
   the planned cell means; `v_ext` is a per-class face drift velocity, zero in v1 (C, REQ-T06).
6. Settling as a downward increment of the class's vertical face velocity on interior
   horizontal faces only; the domain floor and ceiling take `deposition_velocity` whole (D).
7. Concentration conditions derived from the registry's velocity type with explicit overrides;
   obstacle faces deposit as oriented walls (E).
8. A `MassBudget` the solver updates from the fluxes it applies, which the tests read (F).
9. REQ-T11's text, bound and test; `mass_imbalance_tol <= 1e-4 rho V_min / T` (G).
10. VAL-003 scaled, VAL-004 oblique, VAL-007 on an arbitrary face field (H).

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
the absolute value of its sum are below `mass_imbalance_tol`, conditions (b) and (d). Under
`velocity_step` the faces are given with no continuity promise. The IterationState callback is
unchanged. Rejected: reconstructing faces from the cell means in the transport solver, which
gives a field whose per-cell imbalance is O(h^2) and not the one the rule bounded.

## B. The scalar discretization
The concentration of class k, `C_k` [ny, nx], number of particles per cubic metre (the
thresholds are per cubic metre [6]), sits at the pressure locations. The vertical face at
`u[j, i]` is exactly the face between cells (j, i-1) and (j, i), so the volume flux through it is
`u[j, i] dy_cell[j]` with no interpolation; likewise `v[j, i] dx_cell[i]`. The advective flux is
that volume flux times a face concentration `C_f`; the diffusive flux is
`D_k A_f (C_N - C_P) / d_face`, central over the centre-to-centre distance `dx_face`, `dy_face`,
which the mesh supplies for the stretched case (REQ-S11). Cell update: the sum of the four face
fluxes over the cell volume `dx_cell[i] dy_cell[j]`. Every quantity is SI and the depth is one
metre, as it is in the pressure correction.

**The face concentration.** Three candidates. First-order upwind takes the upstream cell value:
bounded, positive, and with an effective diffusion of about `U dx (1 - c) / 2` that section H
shows destroys VAL-004's pulse. QUICK takes the quadratic through the upstream, downstream and
next-upstream nodes, which `quick_face_values` already evaluates with Lagrange weights from the
node positions along either axis (`src/momentum.py`, line 166), Leonard's boundary form
included: second order on the stretched mesh and the momentum equation's scheme (REQ-S09). It is
linear, so by Godunov's theorem it is not monotone: near a front it produces values below the
smaller neighbour, and where that neighbour is zero, a negative concentration. Nothing in
SYSTEM.md says a concentration cannot be negative; it cannot, and a negative value at a sensor is
a reading the monitor cannot interpret. The third candidate bounds the QUICK value: with `r` the
ratio of the upstream to the downstream gradient at the face, the face value is `C_C +
psi(r)(C_D - C_C) / 2` with `psi` clamped to Sweby's TVD region, `0 <= psi <= min(2r, 2)`; SMART
(Gaskell and Lau 1988) and UMIST (Lien and Leschziner 1994) are QUICK inside that region and the
clamp outside it, so on smooth monotone data the face value is QUICK's own.

Positivity cannot be shown for QUICK on the VAL-004 pulse: the Gaussian's tails approach zero
and the undershoot there, of the truncation error's size, is negative. For the bounded form it
can: under forward Euler each cell's update is a convex combination of its own and its upstream
values when the Courant number is at most 1/2 (Harten's lemma; Sweby 1984), the diffusion step
is an M-matrix solve, and the deposition sink is implicit (section C), so a non-negative field
stays non-negative. Whether the clamp changes the scheme's kind is the judgment OPEN 1 asks for:
the face value is QUICK's wherever QUICK is bounded, the assembly is the same, and the momentum
equation has no clamp because a velocity may be negative. Clipping negative values after the
step was rejected: it creates mass, which VAL-007 counts.

**Exactness on a uniform field.** Every candidate's face value is an affine combination of node
values with weights summing to one (Lagrange weights do; the clamp returns a value between
`C_C` and `C_D`), so on a uniform field `C_f = C` at every face, and the net advective flux out
of cell P is `C` times its net volume flux, `C b_P / rho` with `b_P` the per-cell mass imbalance
of `pressure.mass_imbalance`. On a divergence-free face field that is zero to rounding; on the
rule's field it is the drift section G bounds. The diffusive flux of a uniform field is zero
exactly, and a uniform field satisfies the implicit diffusion system with zero residual, so a
partially converged Jacobi solve does not disturb it either. That is what REQ-T11 tests.

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
stable at c <= 1/2 by the lemma that gives positivity (section B). So the integrator follows
OPEN 1: forward Euler for the limited scheme, SSP-RK3 or QUICKEST for the unlimited one. The
momentum predictor's deferred correction does not apply, since there is no implicit advection
matrix to defer against; `quick_face_values` is reused for the face value, not the mechanism.

Second, Brownian diffusion is unresolved everywhere in the bulk of the product case. The cell
Peclet number `U dx / D` is 2.6e7 for the 0.1 um class and 3.7e9 for 5 um, and the diffusion
number `D dt / dx^2` at the configured step is 4.3e-9 and 3.1e-11 [2]. Explicit diffusion would
be stable with a step of 5.8e5 s [2]; implicit diffusion costs one Jacobi sweep there, each sweep
contracting the error by about `4 d / (1 + 4 d)` with `d` the diffusion number. The design keeps
backward Euler on diffusion, solved by Jacobi (data-parallel, REQ-S08's reason if not its
letter): VAL-003 runs a scaled coefficient at a diffusion number of order one (section H), a
clustered product mesh (ADR-010, Consequences, For Phase 3) could put sub-millimetre cells at the
walls, and the deposition sink `v_d C_P A_f / V_P` belongs in the same implicit diagonal so that
it can never overshoot zero. Diffusion matters through `D / delta` at the walls (section D) and
nowhere else.

Third, settling is a boundary-layer and residence-time effect: the 5 um settling velocity is
7.78e-4 m/s, 0.17% of the supply velocity, and a particle falls the room height in 3860 s
against a residence time of 6.7 s [2]. The step is set by the air.

**Contract (draft; SYSTEM.md section 4 carries it marked planned).**

```
TransportSolver:
    __init__(mesh, config, physics: ParticlePhysics, boundary: ConcentrationBoundary)
    stable_dt(faces: FaceVelocities, size_class: int, v_ext=None) -> float
        the largest dt for which the explicit advection step of this class is
        stable: cfl_number / max over FLUID cells of
        (max(|u_w|, |u_e|) / dx_cell + max(|v_s|, |v_n|) / dy_cell), the
        vertical faces carrying the class's settling increment and v_ext;
        cfl_number from the configuration, at most the scheme's bound
    solve_timestep(C_k, faces, size_class, dt, v_ext=None) -> ndarray
        C_k [ny, nx] float64 contiguous, not modified; faces: FaceVelocities;
        v_ext: FaceVelocities-shaped per-class drift increment or None (zero);
        ValueError if dt > stable_dt or on a shape mismatch
        returns the new C_k, [ny, nx], float64, contiguous; SOLID cells zero;
        updates self.budget[size_class] from the fluxes applied (section F)
    budget: list[MassBudget]   # one per class, read by tests and Phase 4
```

The planned signature took `u, v` cell means; the faces replace them because the means do not
carry continuity (section A) and the vertical face velocity is where the settling increment is
added. `v_ext` is kept as REQ-T06 and ADR-007 require. It is a drift velocity, not a force: in
the Stokes regime a body force on a particle becomes a drift velocity through the class's
mobility, which is how `settling_velocity` already turns gravity into a velocity, so a force
field is converted once by its owner with `ParticlePhysics` and the solver receives faces it can
add. It is `None`, read as zero, in v1; a non-uniform `v_ext` has divergence, which is the
physics an electrostatic precipitator would add. Sources (Phase 4) enter as a cell-centred rate
the caller adds to `C_k` around the step; the budget has a slot for them (section F).

## D. Settling
Per class the Stokes settling velocity `v_s` from `ParticlePhysics.settling_velocity` is a
downward velocity (REQ-T03). The solver forms the class's advecting vertical face velocity
`v_k = v - v_s + v_ext_v` on every interior horizontal face and on every face between a FLUID
cell above and a SOLID cell below. On interior faces a uniform increment has zero discrete
divergence, the north and south terms cancelling cell by cell, so constancy survives it. On a
FLUID-over-SOLID face the air velocity is zero and the class flux is `-v_s dx_cell C_f` with the
fluid cell upstream: deposition on the obstacle's top by settling (section E books it). On a
SOLID-over-FLUID face nothing settles out of a solid; the increment is zero there.

The trap is REQ-T09: the floor deposition velocity `deposition_velocity(k, "floor")` is
`v_s + D / delta`, "includes gravitational settling". Adding the settling increment at the domain
floor faces and the floor deposition flux would remove each unit of mass twice. Two compositions
were considered. (i) Increment everywhere, and the floor condition hands the solver
`deposition_velocity - settling_velocity`, the diffusive part alone. (ii) Increment on interior
faces only; the domain's horizontal boundary faces take the wall flux `J = v_d C_P` with `v_d`
the public `deposition_velocity` for the edge's surface, whole. The design is (ii): it reads
REQ-T09 as written, uses the public method unchanged, and keeps one rule for where mass leaves
the domain (section E): at a domain face, by the boundary condition, and nowhere in the interior.
At the ceiling `deposition_velocity(k, "ceiling")` is `D / delta`, nothing enters by settling
since there is nothing above, and the increment is zero there. At a HEPA inlet on the ceiling the
inflow is the face flux times the inlet concentration; the 0.17% the increment would add for the
largest class [2] is not added, so the inlet flux stays the one `get_inlet_flux` reports. The 5
um column of [2] shows what this composes: at the floor `v_d` is 7.78e-4 m/s and `D / delta` is
4.9e-9 m/s, settling to five figures; for 0.1 um the parts are 8.8e-7 and 6.9e-7, comparable.
Traces to REQ-T03 (the velocity per class), REQ-T05 (the floor flux is booked once, as
deposition) and REQ-T09 (the floor value includes settling and the solver does not add it again).

## E. Boundary conditions and `boundary_concentration.py`
The registry stays the one interpretation of which segment covers a point and what type it is
(REQ-S12.1, now with two readers). `boundary_concentration.py` reads it and derives the scalar
condition from the velocity type, with these optional segment keys, validated at load (REQ-C02):
`concentration`, one non-negative value per class, particles per cubic metre, on a
`velocity_inlet` only, default all zero; `hepa_filtered`, a bool on a `velocity_inlet` only,
default false, under which the carried value is `concentration` times one minus
`hepa_efficiency(k)` (OPEN 3); `deposition_surface`, one of `floor`, `ceiling`, `wall`, `none`,
on a `wall` only, overriding the edge default (bottom is floor, top is ceiling, left and right
are walls). A key on a segment of the wrong type is a load error, as an unknown solver key is.

The derived conditions. **Inlet:** the face flux is known and inward; the face concentration is
the carried value, so the inflow of class k is `|u_f| A_f C_in` and nothing diffuses across.
**Outlet:** the face value is the upwind cell's, no diffusive flux; where the corrected outlet
face velocity points inward the entering air is clean, and a count of reversed faces is exposed
for the harness. **Wall:** no advective flux (the face velocity is zero exactly, REQ-S12) and a
deposition flux `J = v_d C_P A_f` with `v_d` from `deposition_velocity(k, surface)`, treated
implicitly (section C), booked per surface. **SOLID cell faces:** every face between a FLUID
and a SOLID cell is a wall of the orientation its normal gives, top of an obstacle a floor,
underside a ceiling, sides walls, with `deposition_velocity` for that surface and the settling
flux of section D on the top, booked as obstacle deposition. Zero flux at obstacle faces was
rejected: the settling flux from the cell above an obstacle would have nowhere to go and mass
would pile in that cell. ADR-010 says obstacle accuracy was not a Phase 2 target; this is the
domain-edge rule applied to interior faces, not a refinement.

The module's contract: `ConcentrationBoundary(mesh, config, physics, registry)` with
`faces_for(size_class) -> ConcentrationFaces`, a frozen dataclass of face-shaped read-only
arrays: `inflow_u` [ny, nx+1], `inflow_v` [ny+1, nx], the concentration an inward flux carries
(zero except at inlets); `deposition_u`, `deposition_v`, the deposition velocity at wall faces,
domain and SOLID, zero elsewhere; `surface_u`, `surface_v`, small integers naming the surface
each depositing face is booked to; `settling_u`, `settling_v`, the mask of faces that carry the
settling increment (section D). Nothing here reads a concentration field or writes one. The
HEPA supply under OPEN 3's first alternative is `hepa_filtered: true` on `hepa_supply` with
`concentration` absent, which is a clean supply, and the breach scenario sets the upstream value.

## F. The mass budget instrument
VAL-007 and the deposition-as-tracked-sink test need one accounting, so the solver keeps it and
the tests read it. `MassBudget`, one per class, in `solver_transport.py`: `initial` (set when a
field is first stepped), cumulative `inflow`, `outflow`, `deposited` by surface name (`floor`,
`ceiling`, `wall`, `obstacle`), `source` (Phase 4 adds to it), and `in_domain(C_k, mesh)` which
sums `C V` over FLUID cells. `residual()` is `initial + inflow + source - outflow - deposited -
in_domain`, and `relative()` divides by `initial + inflow + source`. The solver increments the
cumulative terms from the same face flux arrays it subtracts from the cells, in the same step,
so the test and the solver cannot disagree about what was counted; nothing else writes them.

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
**Proposed text.** REQ-T11: A spatially uniform concentration field advected by a face velocity
field that meets REQ-S04 at its stop, with diffusion, settling, deposition and sources off, shall
remain uniform: after a simulated time T the largest relative departure of any FLUID cell from
its initial value shall not exceed `b_max T / (rho V_min)`, where `b_max` is the worst per-cell
mass imbalance of that field (condition (b)) and `V_min` the smallest FLUID cell volume; and for
the product configuration the bound evaluated at `mass_imbalance_tol` over the scenario's
`t_end` shall be below REQ-T05's 0.01%. Rationale: the cell means do not carry continuity; the
faces do, to the bound the stopping rule enforces, and a scheme that is exact on a uniform field
(section B) turns that bound into a drift rate. Verified by `tests/test_constancy.py`.

**The bound.** With `C` uniform, cell P's content changes at `-C b_P / rho` per second (section
B), so its relative departure after T is `|b_P| T / (rho V_P)` to first order in T, and the
largest over cells is at most `b_max T / (rho V_min)`. Measured on the rule's own stops from the
saved VAL-001 histories [1]: at the 80x40 stop (outer 3988) the worst cell is 1.10e-12, giving
7.05e-9 per second, 0.0025% per simulated hour; the signed domain sum is 5.63e-11, a net rate
of 1.13e-10 per second over the domain; at the 40x20 stop (outer 1389) the figures are 4.77e-9
and 1.63e-10 per second. After the 80x40 stop the net outflow peaks at 8.71e-9 at outer 4359,
1.74e-8 per second [1]; it never applies, because transport consumes the field at the stop.
These reproduce the figures decision 3 was taken on to two figures.

**The test.** `tests/test_constancy.py` solves VAL-001 at 40x20 under `error_estimate` (about
20 s), takes `face_velocities`, sets `C` to one everywhere, and steps the advection alone with
diffusion, settling, deposition and sources off for N steps at `stable_dt`, T = N dt of the
order of 40 s, so that the predicted departure (about 2e-7) is six orders above the rounding
floor and the second-order term `(b T / rho V)^2` six orders below it. It asserts the measured
largest departure is below the bound and above a tenth of it, bounding the drift and confirming
the mechanism, and asserts the bound at `mass_imbalance_tol` over the product `t_end` is below
1e-4. Its planted control: one interior face perturbed, whose drift must exceed the bound.

**The product tolerance.** `mass_imbalance_tol` is absolute, in kg/s per unit depth, and 1e-10
means different things on the two cases: on the product mesh (cells of 1.6e-3 m^2, air at 1.2)
it gives a worst-cell drift of 5.21e-8 per second, 0.019% per hour, and it is 2.65e-11 of the
room's through-flow of 3.78 kg/s against 2.00e-9 of VAL-001 80x40's, 75 times stricter
relatively [1]. REQ-T05's fraction is the budget the haze may drift in the worst cell over a
scenario, so

    mass_imbalance_tol <= 1e-4 rho V_min / T

with T the scenario's `t_end` from the configuration: 3.20e-9 at the configured 60 s and
5.33e-11 for a one-hour scenario on the product mesh [1]. The number is set when the product
case is solved and its stop measured, not here; the configured 1e-10 satisfies the formula at
60 s and fails it at an hour, which is what REQ-T11's second clause will say.

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
they can be checked.

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
FLUID cells. Criterion: below 1% (the plan row). The time step is the build's to choose so that
the backward Euler error is below the spatial one; a diffusion number of 0.25 (dt = 0.025 s,
150 steps) is the design's starting point, and the budget must close to rounding over the run.

**VAL-004, pulse advection (REQ-T08, OPEN 2).** Domain 2.0 m by 1.2 m, 100x60 cells of 0.02 m,
rho 1.2, a uniform face field at the product supply speed 0.45 m/s directed 30 degrees above the
x axis (u = 0.390, v = 0.225, so the pulse crosses cells obliquely and no axis is special),
D = 0, settling off, one class. Initial Gaussian of sigma 0.08 m (four cells) centred at (0.31,
0.23), off every node, as cell averages. Travel 1.0 m along the flow, 2.22 s, 50 cells; Courant
number 0.4 per the configured `cfl_number`. Analytical solution: the initial pulse translated by
U t. Metrics: peak location, by the centroid of the field (the centroid of a translated Gaussian
is exact, and insensitive to the cell quantization an argmax has), within one cell of the exact
centre; and the shape number of OPEN 2. What upwind would do, for scale: its effective
diffusion `U dx (1 - c) / 2` is 2.7e-3 m^2/s, which over 2.22 s grows sigma^2 from 6.4e-3 to
1.84e-2 and lowers the peak to 35% of its start, which is why REQ-T08's rationale names it.

**VAL-007, mass conservation (REQ-T05).** Two runs on the budget of section F. (i) The flux
assembly on an arbitrary field: domain 1.6 m by 0.9 m, 64x36 cells, rho 1.2, a random face field
that is not divergence-free (seeded), one inlet segment on the left carrying a concentration,
one outlet on the right, deposition on all four walls for the 5 um class with settling, no
sources; 500 steps. Criterion: `relative()` below 1e-4 (the plan row's 0.01%), expected at
rounding, about 1e-13, and the test prints both so the gap is visible. (ii) The same budget on
the VAL-001 40x20 converged faces under `error_estimate`, with the same conditions. The planted
control: the test drops one boundary face from the booking and must fail. REQ-T05's "mass" is
particle count here, since `C` is a number concentration; density enters only through `b / rho`.

## I. Configuration and modules
**Configuration (REQ-C01 to C04).** A new `transport` section: `cfl_number`, a float in (0, 1]
and at most the chosen scheme's bound (1/2 for the limited scheme); `advection_scheme`, one of
the names OPEN 1 ranks, so the VAL-004 comparison can run upwind against the chosen scheme from
the same code; `max_diffusion_iter`, a positive int; `diffusion_tol`, a positive float. The
scheme's stability bound is a module constant, as `JACOBI_WEIGHT` is; the fraction of it a run
uses is configuration. The boundary segment keys of section E. Every key is validated for type,
range and the segment type it is allowed on, with bools rejected where numbers are expected
(`config.py`, REQ-C02); the physical constants stay in `constants.py` (REQ-C04).

**Modules.** `src/staggered.py`: gains `FaceVelocities` (A). `src/solver_staggered.py`: gains
`face_velocities` (A). `src/config.py`: the keys above. `src/boundary_concentration.py` (new):
derives the per-face concentration conditions from the registry and `ParticlePhysics` and hands
them to the solver as data (E). `src/solver_transport.py` (new): advances one class one step on
the face velocities with the scheme, settling, implicit diffusion and deposition, and keeps the
budget (B, C, D, F). `src/particles.py`, `src/boundary_registry.py`, `src/mesh.py`: consumed,
unchanged in contract. Tests: `tests/test_constancy.py` (REQ-T11), the plan's three validation
files (VAL-003, 004, 007), `tests/test_solver_transport.py` and
`tests/test_boundary_concentration.py` (unit).

**Cascade.** `solver_staggered.py -> solver_transport, tests/test_constancy.py`: the
`face_velocities` contract of A. `boundary_registry.py` and `particles.py -> boundary_concentration,
solver_transport`: the coverage rule, the new optional keys, the five methods' units and signs.
`solver_transport.py -> time_integration, monitor`: `solve_timestep`'s shapes, `stable_dt`, the
budget's names. `boundary_concentration.py -> solver_transport`: `ConcentrationFaces`.

**Phase 6.** The CUDA port targets three loops, each data-parallel over faces or cells with the
class as a further independent dimension: the face value and advective flux per face (the
limiter is a per-face clamp), the explicit divergence update per cell, and the Jacobi sweep of
the diffusion system per cell. The budget's reductions are sums over faces. REQ-N03's reference
is the NumPy solver this document describes.

## J. What this design does not decide
The product mesh: ADR-010 asks for the clustering cost to be measured on the product case before
a mesh is chosen, and deposition's dependence on the wall spacing is why; nothing here fixes it.
The adaptive outer iteration: deferred on 2026-10-02 to an efficiency pass, not Phase 3; this
design consumes the field at the stop and does not depend on what follows it. Sources, events
and the time loop: Phase 4 (`scenarios.py`, `time_integration.py`), which receive `stable_dt`,
the budget's `source` slot and the segment keys as their interface. The monitor: Phase 5, which
reads `C_k` and never writes it (REQ-A06). The CUDA kernels' form: Phase 6, which receives the
three loops of section I.

## Consequences
**Positive.** The transport solver advects with the fluxes continuity was enforced on, so a
uniform haze drifts at a rate the stopping rule already bounds and the constancy test measures,
not at the O(h^2) rate a reconstruction from cell means would give. One budget serves every
conservation claim. The face scheme reuses the momentum stencil's evaluator on the stretched
mesh. The boundary module and the velocity layer read one registry.

**Negative.** Five classes mean five explicit steps per time step at the air's Courant bound;
implicit diffusion costs a sweep or two per class on the product mesh for no accuracy there. The
limited scheme, if chosen, is nonlinear and first order at extrema; the unlimited one needs a
three-stage integrator and admits negative values. `face_velocities` doubles the solver's
retained field memory. REQ-T11 ties `mass_imbalance_tol` to the scenario duration, so a longer
scenario tightens the velocity solve.

## Planned against built
To be filled at the Phase 3 gate, as ADR-010's table was at ECR-001 step 9: one row per decision
above, with what was built and where it was measured.

## Alternatives considered
Each section names its own. Across the document: fully implicit advection (against the plan's
explicit-advection decision and REQ-N01, which presumes a CFL condition) and a Lagrangian
particle method (the system is Eulerian by SYSTEM.md section 5 and ADR-006).

## Sources
Each figure above is from one of these; section H's case numbers are arithmetic shown inline.

1. `results/builder30/constancy_drift.py`, `.json` and `.log`: the drift rates at the VAL-001
   stops and after them, from `results/builder28/signed_envelope_poiseuille_{40x20,80x40}.npz`,
   and the product-mesh evaluation of `mass_imbalance_tol` and the tolerance formula.
2. `results/builder30/particle_table.py`, `.json` and `.log`: the per-class table on
   `configs/clean_room_default.yaml`.
3. `results/builder30/fe_quick_stability.py`, `.json` and `.log`: the von Neumann amplification
   of forward Euler and SSP-RK3 with QUICK and upwind face values.
4. `docs/reports/stopping_rule_evidence.md`, section 10: the stops at outer 1389 and 3988 and
   the post-stop oscillation.
5. `docs/ADR/ADR-010-staggered-grid-architecture.md`: decisions 1, 2 and 5; Consequences, For
   Phase 3.
6. `configs/clean_room_default.yaml`: the product case.

Gaskell and Lau (1988), SMART; Lien and Leschziner (1994), UMIST; Sweby (1984), the TVD region;
Harten (1983), the lemma; Leonard (1979), QUICK and QUICKEST; Shu and Osher (1988), SSP
Runge-Kutta; Carslaw and Jaeger (1959) and Crank (1975), the heat kernel.
