# ADR-012: Turbulence Model: k-epsilon on the Staggered Solver

## Status
Proposed. Written 2026-10-04, before the build, as the design for ECR-002
(`docs/ECR/ECR-002-turbulence-model.md`), from the evidence in
`docs/reports/product_case_reynolds.md`. Seven questions are put to Alex and listed first; the
sections carry them with their ranking and do not choose. When accepted this ADR supersedes
ADR-004 (laminar flow assumption) and gains a planned-against-built table when ECR-002 closes,
as ADR-010's was added at ECR-001 step 9. Bracketed numbers point at the sources at the end.

## Decisions for Alex
Each decision is a picture first, then the options ranked with their consequence, then the
detail and the section that carries it. Decision 1 of 2026-10-04 (a k-epsilon model, not an
algebraic one) is taken; these are what it leaves open.

**1. Which k-epsilon (section A).**
*Picture.* The model keeps two extra numbers in every cell: how much churning energy the air
carries there, and how fast that churning dies away. From the two it works out how strongly the
cell mixes momentum and particles with its neighbours. The variants differ in the rule for how
fast the churning dies, and they differ most where the downflow hits the top of a piece of
equipment and stops.
*Options, ranked.*
(1) Standard k-epsilon. The most published model in rooms like this one, the Annex 20 room
included, so a reader can set our result beside many others. Known to overstate the churning
where flow strikes a surface head on: the equipment tops will look more mixed than they are.
(2) RNG k-epsilon. The same equations with different constants and one extra term that lowers
the mixing where the flow is strained hard, which is the equipment tops; the survey calls it
similar or slightly better [2]. Fewer published results in this exact room.
(3) The low-Reynolds variants. They need millimetre cells at every wall and gave no or marginal
improvement with case-dependent stability problems [2]. Last.
*Recommendation.* Build (1) with (2) as a configuration variant: they share every line but the
constants and one source term, and the Annex 20 comparison (decision 5) then measures the
difference in this room rather than taking it from the literature. The product runs use (1)
unless that comparison favours (2).

**2. The wall treatment (section B).**
*Picture.* At every wall the air slows to rest in a layer a few millimetres thick, far thinner
than one of our 4 cm cells. Either we cut the space next to every wall into millimetre cells and
compute that layer, or we use the shape such layers are measured to take, the law of the wall,
to tell the first cell how hard the wall drags on it.
*Options, ranked.*
(1) Wall functions with a floor on the first node's distance in wall units (the scalable form).
Works on the product mesh as it stands. Where the air by a wall is slow, the first node sits
inside the layer's innermost part, and the floor treats it as sitting at that part's edge, which
misstates the drag there by a bounded amount instead of failing.
(2) Wall functions without the floor. Invalid wherever the first node is closer than the log
layer, which on the product mesh is the slow corners and the middle of each equipment top.
(3) Resolve the layer to the wall (a two-layer or low-Reynolds treatment). Needs wall cells of 1
to 3 mm, 13 to 40 times finer than today, at every wall and every obstacle face. The mesh can
cluster only toward the domain walls, not toward an obstacle's top, and clustering costs
accuracy and pressure sweeps on this stencil (ADR-010 decision 6).
*Recommendation.* (1).

**3. How k and epsilon are computed and kept positive (section C).**
*Picture.* Neither new number may go below zero. A negative churning energy means nothing, and
it would make the mixing strength negative, which makes the solve blow up. The transport solver
already moves particle concentrations in a way that cannot go below zero; the question is
whether to move these two numbers the same way.
*Options, ranked.*
(1) Move them with the transport scheme: the bounded face value of ADR-011, one explicit
pseudo-time step per outer iteration at a local Courant number of at most 1/2, the growth terms
added explicitly and the decay terms taken implicitly. Positive at every step by ADR-011's
argument, which section C shows carries over with these sources; reuses built and tested code.
(2) Solve them the way the momentum equation is solved, steady and implicit with a deferred
QUICK correction. Reaches the same answer, but an intermediate iterate can go negative and needs
clipping.
(3) First-order upwind, steady. Positive, but it smears k and epsilon across the shear layers
that make them.
*Recommendation.* (1).

**4. The pressure solve (section H).**
*Picture.* Every outer iteration solves for a pressure correction by passing information one
cell per sweep. On the 200x75 room one correction took 27,000 sweeps to the committed tolerance
and 112,000 to the probes' tighter one, and a converged solve needs thousands of corrections.
Multigrid passes the same information on coarser copies of the grid as well, and typically needs
some tens of sweeps' worth of work per correction whatever the grid (typical, not measured here).
*Options, ranked.*
(1) Multigrid with the weighted Jacobi sweep as its smoother, as a separate change (ECR-003,
REQ-S08 amended), done before ECR-002's product step. The laminar solver needs it too, and
ECR-002 stays about the model.
(2) The same multigrid as a step inside ECR-002.
(3) Keep plain weighted Jacobi. Hours per steady product solve in NumPy (section H), and every
scenario that changes a boundary needs a new solve.
*Recommendation.* (1).

**5. What the Annex 20 room is scored on (section G (iii)).**
*Picture.* The Annex 20 room is a published test room with measured air speeds along two
vertical lines. Before building on it we checked that the measured speeds carry as much air
through each line as enters the room, which they must in a flat room. At the first line they
carry about the right amount, between all of it and a third more depending on how the readings
are extended to the walls. At the second line the middle plane of the measured room carries six
tenths of it while a plane near its side wall carries a tenth more than all of it: the model
room was not flat there. No flat model can match the middle plane at the second line.
*Options, ranked.*
(1) Score the first line and the line along the jet under the ceiling; report the second line
unscored, with the flux finding beside it. The threshold is set from the spread of published
two-dimensional standard k-epsilon results against the same data, measured in the build.
(2) Score both lines against the mean of the two measured planes.
(3) Score the middle plane at both lines with an allowance equal to the measured shortfall.
(4) No turbulent room comparison; the channel check (G (ii)) and the product's convergence only.
*Recommendation.* (1), its number OPEN until the build has measured the published spread.

**6. Turbulent deposition (section F).**
*Picture.* Particles reach a wall by drifting across the thin still layer next to it. Today that
layer is a fixed 1 mm thick everywhere. With turbulence, how thin the layer is depends on how
hard the air scrubs that wall, which the wall treatment (decision 2) computes.
*Options, ranked.*
(1) Defer to a follow-on change after ECR-002. Deposition keeps its present rule; the bulk of the
room gains turbulent mixing and the walls do not, which understates deposition of the smallest
classes by an amount this change does not measure.
(2) Include it as ECR-002's last step: a turbulent deposition model, REQ-T09 changed, its own
validation data.
*Recommendation.* (1), because the model needs the wall treatment's friction velocity, which
exists only once decision 2 is built.

**7. Two model inputs (sections F and I).**
*Picture.* (a) How strongly the churning spreads particles compared with how strongly it spreads
momentum. (b) How much churning the filtered supply air already carries when it enters through
the ceiling.
*Options, ranked.* (a) A turbulent Schmidt number of 0.7, the conventional value, inside the 0.2
to 1.3 the literature spans [7]; or 1.0; or a variable form (outside this change). (b) No value
in code; every inlet states its intensity and dissipation length in the configuration, the
product's set from a sensitivity pair in the product step (section J); or Annex 20's convention
of 4% copied to the product supply now.
*Recommendation.* (a) 0.7, configurable. (b) The first.

## Context
The product room has never been solved. On the committed configuration its Reynolds number on
the room height is 89,500 and its cell Reynolds number on the 0.04 m mesh 1,190; every case the
flow solver was validated on ran at 5 (VAL-001) or 100 (VAL-002) [1, section 2]. As committed,
the solve diverges by outer 58; with the pressure solved toward 1e-8 on a coarse copy of the
room it diverges at the real viscosity and at ten times it, stalls at a hundred times and
converges at a thousand times [1, section 4].
ADR-004 calls the room laminar; in the clean-room sense, unidirectional and low in turbulence
intensity, it is, but not in the Navier-Stokes sense [1, section 3]. Alex decided on 2026-10-04 to
add k-epsilon (ECR-002 decision 1). This document is the design ECR-002's steps are built from.

What it plugs into. The staggered solver (ADR-010): p at cell centres, u and v on faces, QUICK by
deferred correction over an upwind implicit matrix with one under-relaxed Jacobi sweep per outer
iteration, a weighted Jacobi pressure solve, and the `error_estimate` stopping rule. Viscosity
enters it in one place, the four diffusion conductances of `MomentumPredictor._assemble`
(`src/momentum.py`, lines 386 to 394), as the scalar `config.mu`. The transport solver (ADR-011):
concentration at cell centres advected by the faces continuity was enforced on, a bounded QUICK
face value under forward Euler, and implicit diffusion and deposition by Jacobi; its diffusivity
is one scalar per class (`src/solver_transport.py`, lines 475 and 824).

## A. The variant (REQ-S14, proposed; decision 1)
Both variants carry, at cell centres, the turbulent kinetic energy k and its dissipation rate
eps, and form the eddy viscosity `mu_t = rho C_mu k^2 / eps`. Standard k-epsilon (Launder and
Spalding 1974):

    div(rho u k)   = div((mu + mu_t / sigma_k) grad k)   + P_k - rho eps
    div(rho u eps) = div((mu + mu_t / sigma_e) grad eps) + C_1 (eps / k) P_k - C_2 rho eps^2 / k
    P_k = mu_t S^2,  S^2 = 2 S_ij S_ij
    C_mu 0.09, C_1 1.44, C_2 1.92, sigma_k 1.0, sigma_e 1.3

RNG (Yakhot and Orszag 1986; Yakhot et al. 1992) keeps the form with C_mu 0.0845, C_1 1.42,
C_2 1.68, sigma_k = sigma_e = 0.7194, and subtracts from the eps equation
`R = C_mu rho eta^3 (1 - eta / eta_0) / (1 + beta eta^3) eps^2 / k`, `eta = S k / eps`,
eta_0 4.38, beta 0.012. These are the values as usually tabulated; the build checks them
against the primary papers before step 1 merges.

**What each misjudges here.** The supply strikes four equipment tops and stops: a stagnation
region, where the strain is normal rather than shear. P_k = mu_t S^2 grows with any strain, so
the standard model produces k there that a real stagnation flow does not have, the stagnation
point anomaly of the k-epsilon family; the equipment tops will read more turbulent and more
mixed than they are. RNG's R term changes sign where eta exceeds eta_0, which strong strain
produces, and raises eps there, so it lowers mu_t where the standard model overshoots; Chen
(1995), as the survey reports it, found RNG best overall on an impinging case with the standard
model competitive [2]. The zone above the equipment is the supply flowing down at a few percent
intensity: weakly turbulent, and both high-Reynolds forms only let k and eps decay there, which
is the right direction and is not transition modelling. The low-Reynolds variants add damping
functions for the near-wall region and need a first node near y+ = 1; the survey found "no or
marginal improvements on prediction accuracy" with "strong case-dependent stability problems"
[2], which ranks them last. A production limiter (Kato and Launder 1993) addresses the
stagnation overshoot in either variant; it is not ranked, and is the change to make if the
equipment tops show it.

**Cost.** Two scalar equations per outer iteration on the cell field, each the transport step of
section C; RNG adds one source term. Against the pressure solve both are small (section H).

**Record.** The survey's summary: standard k-epsilon with wall functions "provides acceptable
results (especially for global flow and temperature patterns) with good computational economy";
RNG "provides similar (or slightly better) results" [2, page 14, remarks 1, 2 and 4]. On the
Annex 20 room Susin et al. (2009), in three dimensions, found the standard model best against
RNG and k-omega [3, page 8]. This record is the reason for decision 1 and for ranking the
standard model first.

## B. Wall treatment (REQ-S17, proposed; decision 2)
**y+ on the product mesh.** The tangential velocity's first node sits half a cell from the wall,
0.02 m on the 0.04 m mesh (`wall_distance`, ADR-010 decision 2). With air's nu = mu / rho =
1.508e-5 m^2/s, `y+ = u_tau 0.02 / nu = 1326 u_tau`. The friction velocity is estimated from a
flat plate, because no field exists: outer speeds 0.1 to 0.45 m/s (the supply, and the slower
flow turning along floors and equipment tops), run lengths 0.3 to 3 m (an equipment top is 0.6
to 1.3 m), Cf from the turbulent correlation `0.0592 Re_x^-0.2` and the laminar `0.664
Re_x^-0.5`. Both give u_tau from 0.005 to 0.031 m/s, the run Reynolds numbers being 2,000 to
90,000 [9]. Hence:

| u_tau (m/s) | where | y+ at 0.02 m |
|---|---|---|
| 0.005 | slow corners, 0.1 m/s over 3 m | 6.6 |
| 0.01 | 0.1 to 0.2 m/s flows | 13 |
| 0.02 | 0.2 to 0.45 m/s over 1 to 3 m | 27 |
| 0.03 | 0.45 m/s over 0.3 to 0.6 m | 40 |

and toward zero at each stagnation point. Standard wall functions want the first node in the
log layer, y+ above about 30; below about 11 the node sits in the viscous sublayer, where the
log law is not the profile. The product mesh is between: in the tens over most surfaces, below
11 in the slow corners and near the middle of each equipment top.

**On the Annex 20 grid** (section G): along the ceiling jet the outer speed is 0.82 u0 at x/H 1
and 0.64 u0 at x/H 2 (the measured y = h/2 line [1, section 6]), u_tau about 0.02 m/s by the
same correlation at 0.37 m/s over 3 m. A wall cell of h/4 = 0.042 m puts the first node at
y+ 27; h/8 = 0.021 m at 14 [9]. The proposed grid (section G) uses 0.042 m.

**What each treatment asks of the mesh.** Wall functions (Launder and Spalding 1974) bridge the
layer: the wall shear on the first node is `tau_w = rho C_mu^(1/4) k_P^(1/2) kappa u_P /
ln(E y*)`, `y* = rho C_mu^(1/4) k_P^(1/2) y_P / mu`, kappa 0.41, E 9.793, with k's wall flux zero,
its production in the wall cell taken from tau_w, and eps in the wall cell set to `C_mu^(3/4)
k_P^(3/2) / (kappa y_P)`. Using k rather than u_tau as the velocity scale keeps the shear finite
at a stagnation point, where u_P falls to zero. The scalable form (Grotjans and Menter 1998)
evaluates the log law at `max(y*, 11.06)`, so a node inside the sublayer is treated as at its
edge: the error is bounded where (2) would be undefined. The mesh keeps its 0.04 m cells. In
code the wall function is a per-face wall viscosity, `mu_w = rho C_mu^(1/4) k_P^(1/2) kappa
y_P / ln(E max(y*, 11.06))`, put where `mu` now stands in the two wall rows of `_assemble`
(`src/momentum.py`, lines 389 to 394), so the existing wall stencil `(phi_P - phi_wall) /
wall_distance` carries it unchanged.

Resolving the layer needs the first node near y+ = 1: a wall cell of `2 nu / u_tau`, 3.0 mm at
u_tau 0.01 and 1.0 mm at 0.03, against 40 mm now [9]. Clustered at ratio 1.2 from 1.5 mm to 40
mm takes about 18 cells per wall per axis. REQ-S11 clusters each axis toward both domain walls,
mirrored, and toward nothing else, so an equipment top at y = 2.0 m cannot be reached at all.
ADR-010 decision 6 measured clustering on this stencil at ratio 1.2: 7.4 times the uniform
error at the same cell count, 96% of it the clustered stencil's own. And the pressure sweeps
grow with the cells per side (section H).

**Obstacle walls.** `momentum.py` treats a face of a SOLID cell as a fixed zero at its storage
location with no half-cell wall distance; obstacle accuracy was not a Phase 2 target (ADR-010,
Consequences). With wall functions the equipment tops are where the supply strikes, so the
obstacle faces need the wall stencil the domain edges have, with the half-cell distance, and
the wall function on it. That is part of step 3 of ECR-002, and it changes the laminar solver
only in rooms with obstacles, which no validation case has.

## C. Discretization of k and eps (REQ-S15, proposed; decision 3)
**Where they live.** At cell centres, beside p and the concentrations, so that the corrected
faces advect them with the fluxes continuity was enforced on (ADR-011 A), the scheme is exact on
a uniform field (ADR-011 B), and the production term reads the velocity gradients where the
staggered layout makes them exact: du/dx and dv/dy at a cell centre are face differences;
du/dy and dv/dx live at the cell corners and are averaged from the four corners to the centre.

**The step.** Recommended option (1): each outer iteration advances k and eps by one
pseudo-time step of the transport solver's scheme. Per cell P with local step `dt_P = cfl_t /
(max(|u_w|, |u_e|) / dx + max(|v_s|, |v_n|) / dy)`, `cfl_t <= 1/2`, capped at the largest finite
dt_P in the field so that a cell at rest takes a finite step (steps 2 and 3 are implicit in the
decay and would stay positive without the cap):

1. advect with the UMIST-limited QUICK face value, forward Euler (ADR-011 B), giving k*, eps*;
2. add the explicit growth: `k* + dt_P P_k / rho`, `eps* + dt_P C_1 (eps / k) P_k / rho`, with
   P_k, eps / k and mu_t from the previous iterate;
3. one implicit solve per quantity for diffusion and decay together: the decay `rho eps / k`
   (for k) and `C_2 rho eps / k` (for eps), from the previous iterate, sit in the diagonal with
   the diffusion, as the deposition sink does in ADR-011 C, solved by Jacobi.

**Positivity carries over.** ADR-011 B's lemma makes step 1 a convex combination of
neighbouring values at a cell Courant number of at most 1/2, so k* and eps* are non-negative.
The lemma is per cell, so a local step keeps it: each cell's update uses its own dt_P, which
makes pseudo-time non-conservative, and the steady fixed point does not depend on dt. Step 2
adds a non-negative quantity, since P_k = mu_t S^2 >= 0 and eps / k > 0. Step 3's matrix has
the diagonal `rho V / dt_P + sum G_f + decay V`, positive, and off-diagonals `-G_f <= 0` with
`G_f = (mu + mu_t / sigma) A_f / d_f`: strictly diagonally dominant with non-positive
off-diagonals, an M-matrix, and its right-hand side `rho V / dt_P` times step 2's field is
non-negative, so the solution is. On a connected domain with a positive inflow somewhere the
matrix is irreducible and the solution is strictly positive, so eps / k and mu_t stay defined
without a floor. Treating the decay explicitly instead would allow `k - dt eps < 0`.
Linearising the destruction this way is Patankar's rule for a source that must not drive a
variable negative (Patankar 1980, on source-term linearisation). RNG's R term enters the
diagonal where it is positive and the explicit growth where it is negative, which keeps both
steps' signs.

**Alternatives.** (2) the momentum assembly with the limited value as the deferred correction:
its converged equations are the same, but the explicit correction can make a cell's right-hand
side negative mid-iteration, so k needs clipping, the remedy ADR-011 B rejected for
concentration; for k a clip creates no mass that a test counts, but it is still a change to the
equations that no test fails. (3) first-order upwind, steady: an M-matrix with a non-negative
right-hand side, so positive, with the numerical diffusion ADR-011 H measured on VAL-004 (a peak
reduced to a third over 50 cells); k and eps are made in thin shear layers.

**Boundary values.** Inlet: `k = 1.5 (I |u_n|)^2` and `eps = k^(3/2) / l_e` from the segment's
intensity I and dissipation length l_e, configured per inlet (section I); the Annex 20 test sets
I = 0.04 and l_e = h / 10, its specification's equations (5) to (7) as written [4, page 2].
Outlet: the upwind cell's value on outflow; on a reversed face the adjacent cell's value
(zero gradient), where ADR-011 E brings clean air for a concentration, since k has no clean
state. Walls and obstacle faces: no advective flux (the normal velocity is zero, REQ-S12), zero
diffusive flux of k, eps in the wall cell held at the wall function's value (section B) by a
dominant diagonal, P_k in the wall cell from the wall shear.

## D. Coupling into the flow solver (REQ-S14, S16 and S01, proposed)
**The viscosity field.** `mu_e = mu + mu_t` per cell. `MomentumPredictor` gains an optional
cell field, None by default; with None the assembly runs the present scalar lines unchanged, so
the validated laminar results are the same bits (G (i), REQ-S16), not equal to rounding. The
obstacle wall stencil (section B) is the one change the laminar path sees, and only in rooms with
obstacles, none of which has a validated result.

**Its faces.** In `_assemble`, a component's streamwise faces lie at cell centres (`d_s`, line
386), so they take the cell's mu_e with no averaging. Its transverse faces lie at cell corners
(`d_t`, lines 387 to 394) and each spans two half cells on either side of a row boundary: the
flux crosses a row boundary in series and the two half-faces carry it in parallel. The face
value is the distance-weighted harmonic mean across the row boundary in each column, then the
width-weighted arithmetic mean of the two columns: Patankar's rule for a coefficient that jumps
between control volumes (Patankar 1980, on the interface conductivity), applied in each
direction as the geometry composes it. Rejected: the arithmetic mean of the four cells, which
overstates the flux where mu_t drops sharply, as it does across a jet's edge; the four-cell
harmonic mean, which puts the parallel paths in series. On a smooth field the three agree to
second order; they differ where mu_t jumps.

**The terms constant viscosity let it drop.** With mu_e varying, the viscous force on u is
`div(mu_e grad u) + d/dx(mu_e du/dx) + d/dy(mu_e dv/dx)`; the last two sum to zero only for
constant mu_e on a divergence-free field. They are added as an explicit source on the current
field, beside the deferred correction, from the same staggered differences (du/dx at centres,
dv/dx at corners). The Boussinesq stress also carries `-(2/3) rho k` on the diagonal, which is a
gradient and goes into the pressure: the solver's p becomes `p + (2/3) rho k`. Nothing reads p
but the views; the contract says which p is returned.

**Diagonal dominance and Jacobi.** The upwind matrix's coefficients are `a_nb = D_f + max(+-F_f,
0)` with `D_f = mu_e A_f / d_f >= 0`, and `a_P = sum a_nb + net F`. A larger mu_e raises D_f in
a_nb and a_P alike, so dominance holds for any non-negative mu_t, and the one Jacobi sweep per
outer iteration is unchanged in kind. What changes is the cell Peclet number: with nu_t of 1e-3
m^2/s, `U dx / nu_e` is 18 instead of 1,190 [9], so the implicit diffusion carries the face
coupling that the explicit deferred correction carries now. That is the expected reason the
outer iteration converges where it diverges laminar, a hypothesis that steps 3 and 6 measure.

**The outer iteration.** Momentum prediction, pressure correction (unchanged), then the k and
eps step of section C on the corrected faces, then `mu_t = (1 - a_t) mu_t_old + a_t rho C_mu
k^2 / eps` with `alpha_turbulence` = a_t. The k and eps step reads the corrected faces for the
reason the transport solver does. `alpha_velocity` keeps its meaning; a value for the turbulent
cases is a measurement of step 3, not a guess here. The pressure correction's d = A / a_P reads
the larger a_P and needs no change.

## E. The stopping rule (REQ-S01, clarified; rule version 4)
ADR-010's conditions (a) to (d) bound the velocity's iteration error, the per-cell imbalance, the
summed imbalance over the through-flow and the signed domain sum. Under k-epsilon the iterate
is (u, v, p, k, eps). Condition (a) estimates the remaining error from the velocity step and its
fitted geometric rate; that is valid for the joint iteration when its slowest mode shows in the
velocity. A mode living mostly in eps in a quiescent corner moves mu_t, and so the velocity,
slowly, and the velocity's rate can understate it.

Condition (e): the same estimate on the eddy viscosity, `step_nu * rho_hat / (1 - rho_hat) /
nu_scale < iteration_error_tol`, with step_nu the largest change of nu_t over one outer iteration,
rho_hat fitted over RATE_WINDOW steps as in (a), and nu_scale the largest nu_e in the field. It
shares (a)'s tolerance because it bounds the same kind of relative error. The rule needs all five,
with (e) not evaluated when the model is off, so laminar stops are unchanged. The version a solve
records becomes a property of the solve rather than the module constant `RULE_VERSION`: 3
when (e) is off, 4 when it applies. A laminar solve applies version 3's conditions, and
recording 4 for it would split the harness summary's stored laminar rows from new ones that are
bitwise the same;
`scripts/stopping_probe.py`, `scripts/val001_order.py` and `scripts/benchmark.py`, which store the
version today, read it from the solver. Rejected: a bound on the k and eps steps themselves, since
momentum and transport read only nu_t; and no fifth condition, which would let a solve stop with
the viscosity still moving. On the product case `mass_imbalance_tol` follows ADR-011 G's formula
`1e-4 rho V_min / t_end`, unchanged.

## F. Coupling into transport (REQ-T13, proposed; decisions 6 and 7)
**The diffusivity.** `D_k = D_B,k + nu_t / Sc_t` per face: the class's Brownian coefficient plus
the turbulent particle diffusivity, the same for every class. The classes are tracers to the
turbulence: the 5 um class has a relaxation time of 7.9e-5 s against a Kolmogorov time of about
0.4 s at eps 1e-4 m^2/s^3 [9], and its settling velocity is a few percent of a velocity
fluctuation of a few centimetres per second, so neither inertia nor crossing trajectories
separates it from the air. `Sc_t` is one configured value, 0.7 recommended, from a literature
range of 0.2 to 1.3 [7]. nu_t is given per cell; a transport face lies between two cell
centres, so the face value is the distance-weighted harmonic mean of the two (Patankar 1980, the
interface conductivity), with no corner case.

**The contract.** `solve_timestep` gains a keyword-only `eddy_viscosity: ndarray | None`, a
`[ny, nx]` field read from the flow solver; None is the present path, bitwise. The conductance
of `_implicit_step` becomes `(D_B + D_t,f) A_f / d_f` per face in place of `D A_f / d_f`.

**ADR-011's claims.** Exactness on a uniform field (VAL-012): the diffusive flux of a uniform
field is zero for any face conductance, and the advective part is unchanged; REQ-T11 is stated
with diffusion off in any case. Positivity (REQ-T12): the explicit advection is unchanged, and
the implicit matrix keeps its form, diagonal `V / dt + sum G_f + deposition` against
off-diagonals `-G_f`, an M-matrix for any `G_f >= 0`, which holds because nu_t >= 0 (section C).
Conservation (VAL-007): every face flux still leaves one cell and enters the next, whatever its
conductance. `stable_dt` bounds the explicit advection only and is unchanged. Jacobi on the
implicit system: with D_t near 1e-3 m^2/s, dt 0.01 s and dx 0.04 m the diffusion number is
6e-3 and a sweep contracts the error by about 0.02, so a few sweeps suffice. What changes in
kind: Brownian diffusion was unresolved everywhere, cell Peclet numbers 2.6e7 to 3.7e9 (ADR-011
G); with D_t near 1e-3 the cell Peclet number is about 18, and diffusion now shapes the bulk
field. VAL-003 already tests the operator at D = 1e-3 m^2/s and a diffusion number of 0.1;
a per-face coefficient needs the dense-solve unit test extended to a piecewise D.

**Deposition (decision 6).** ADR-011 deposits through `D / delta` with delta 1 mm fixed
(REQ-T09), plus settling on floors. In turbulent flow the near-wall concentration layer thins as
the wall shear grows, and indoor deposition models make the deposition velocity a function of
the friction velocity, Lai and Nazaroff (2000) the usual one [8]. That needs u_tau at every wall
face, which only the wall treatment of decision 2 provides once built, and it changes REQ-T09's
parameterization and needs its own validation data. Recommended: defer to a follow-on change;
D_t is zero at a wall face, so the wall flux keeps today's rule and the change stays separable.

## G. Validation (REQ-S14 to S17, REQ-T13; decision 5)
Every case has non-unit density, a non-square domain, and inputs off grid nodes, the Phase 2
lessons. The VAL identifiers are proposed and fixed when Alex accepts ECR-002.

**(i) The laminar limit.** With the model off, `val001_80x40`, `val001_80x40_stretched` and
`val002_80x80` reproduce their stored rows bitwise: the same outer count and stop, and a hash of
the returned faces equal to one taken at main before the change. The transport gate tests pass
unchanged with `eddy_viscosity` None. Criterion: equal bits (REQ-S16). Verified in every
step that touches momentum.py, solver_staggered.py or solver_transport.py.

**(ii) The model's implementation, with known answers.** Two checks, each with an exact answer.
*VAL-015, decaying turbulence.* A closed 2.4 m by 1.5 m box, rho 1.2, zero velocity, uniform
initial k0 and eps0: the k-epsilon equations reduce to `dk/dt = -eps`, `deps/dt = -C_2 eps^2 /
k`, whose solution is `k = k0 (1 + (C_2 - 1) eps0 t / k0)^(-1 / (C_2 - 1))` with eps from it.
Stepped in true time with a fixed dt, the computed k and eps must follow it with an error that
falls at first order in dt (observed order between 0.9 and 1.1 over three steps), and stay
uniform to rounding. It tests the decay terms and their implicit treatment against an exact
answer; at rest the production is zero.

*VAL-016, the channel.* A plane channel H = 0.3 m, L = 18 m (60 H, since a turbulent channel
develops over tens of heights), rho 1.2, air, bulk velocity chosen off the grid's rounding
(U = 1.37 m/s, Re on H about 27,000), wall functions, compared near the outlet where the profile
has stopped changing along x to within the comparison's allowance, which the step measures. The
k-epsilon log layer is an exact solution of the model: with constant shear stress `k = u_tau^2 /
sqrt(C_mu)`, `eps = u_tau^3 / (kappa_m y)` and `u+ = (1 / kappa_m) ln y+ + B`, u_tau from the
pressure gradient, where the model's own `kappa_m^2 = (C_2 - C_1) sigma_e sqrt(C_mu)` gives 0.433
for the standard constants [9]. Compared: the slope of u+ against ln y+ over the nodes between
y+ 30 and 0.2 of the half-height in wall units, against 1 / kappa_m, and k against `u_tau^2 /
sqrt(C_mu)` over the same nodes. Criterion: within 3% on the finer of two grids, the gap
falling under refinement. The answer is the model's, so the allowance is for discretization; 3%
is about half the gap between 1 / 0.433 and the wall function's 1 / 0.41, so the check tells a
log layer the model made from one copied off the wall function. Reported beside it, unscored:
the skin friction against Dean's (1978) correlation `Cf = 0.073 Re_m^-0.25` for two-dimensional
channels, which the build fetches and checks before quoting. The log-layer check tests the model
as built; Dean tests the model against physics.

**(iii) The Annex 20 room, conditional on item 0.** The specification [4, pages 2 and 3] and
the measurements [5] are in hand and their check is in [1, section 6]: at x/H = 1.0 the measured
symmetry-plane profile carries 1.02 to 1.32 of the inlet flux u0 h, depending on how the strips
between the outermost readings and the walls are closed (1.17 with no slip); at x/H = 2.0 it
carries 0.60 to 0.64 under any closure, while the plane z/W = 0.4, digitized by the builder from
the specification's figure 6, carries 1.11 to 1.13. The two planes differ by half the inlet flux
at the same section. The shortfall is the model room's three-dimensionality, the specification's
"slightly three-dimensional" flow [4, page 5], and a two-dimensional solution, which carries u0
h through every section, cannot match the symmetry plane at x/H = 2.0 closer than an integrated
0.021 u0 H, about 0.04 u0 if it sits in the return flow below mid-height. It is consistent with
the text under [3] figure 5, where a two-dimensional low-Reynolds k-epsilon prediction leaves the
counter flow "slightly underestimated" [6].

Case: L 9.0 m, H 3.0 m, slot h 0.168 m at the top of the left wall, outlet t 0.48 m at the foot
of the right wall, u0 0.455 m/s, nu 15.3e-6 m^2/s (Re 5,000 on h, 89,200 on H, the product
room's within 0.4%), inlet k and eps from I = 0.04 and l_e = h / 10. Grid 216 x 72, uniform
0.0417 m, so the slot spans 4.03 cells and its edge falls off a face; the inlet velocity is
scaled so the covered faces carry u0 h exactly. Compared: u / u0 along x/H 1.0 and 2.0 and along
y = h/2 and y = H - h/2, the four lines the specification names [4, page 3]. Criterion OPEN
(decision 5): its number is set from the spread of published two-dimensional standard k-epsilon
predictions against the same data, which the build measures by digitizing the predictions in
[6] figures 4 and 5 and [10] with the method of [1, section 6], not a round number.

**(iv) The product room converges.** VAL-018: `configs/clean_room_default.yaml` with the model
on, `stopping_rule: error_estimate`, `mass_imbalance_tol` from ADR-011 G's formula at its t_end,
stops by `error_estimate_and_continuity` under rule version 4 within its cap, on the product
mesh. Reported: the outer count, wall time, the stop's five readings, nu_t / nu over the room,
and y+ at every wall node, which section B estimated without a field.

## H. Cost (REQ-S08; decision 4)
Measured: one first-outer-iteration pressure correction to tolerance, the sweep cap lifted
[1, section 5]:

| Grid | Tolerance (Pa) | Sweeps | Seconds |
|---|---|---|---|
| product 40 x 15 | 1e-8 | 2,553 | 0.13 |
| product 100 x 38 | 1e-8 | 28,456 | 3.0 |
| product 200 x 75 | 1e-8 | 112,519 | 32 |
| product 200 x 75 | 1e-6 (committed) | 27,408 | 7.7 |
| Annex 20, 90 x 30 | 1e-8 | 64,113 | 6.4 |
| Annex 20, 180 x 60 | 1e-8 | 211,592 | 45 |

Doubling the cells per side multiplies the count by 3.3 to 4.0, ADR-010's N^2. The committed
cap of 200 sweeps is 137 times short of one correction at the committed tolerance. On the Re 90
probe the correction's count fell from 9,748 to 4,048 over 1,000 outer iterations, a factor of
2.4 [1, section 4]. A converged product solve needs thousands of outer iterations: VAL-001 80x40
needed 3,988 (ADR-010). At a third of the first correction's count and 0.28 ms a sweep, 4,000
iterations at 1e-8 are 4,000 x 37,000 x 0.28 ms, about 11.5 hours per steady solve; at 1e-6,
about 3 hours; the 216 x 72 Annex 20 grid is of the same order. k and eps add two scalar steps
of a few array passes and a few sweeps each per outer iteration, small beside that. Weighted
Jacobi is workable at 40 x 15 and not at the product mesh: decision 4 brings the deferred
multigrid question forward. A V-cycle whose smoother is the present weighted Jacobi sweep keeps
the per-cell, data-parallel update REQ-S08 asks for on every level and removes the N^2 factor;
REQ-S08's text names the algorithm, so it is amended, and the recommended route is a separate
ECR-003 that ECR-002's product step depends on. The laminar product solve has the same need: in
the 200 x 75 probe with a 5,000-sweep cap every correction stopped at its cap.

## I. Configuration and modules (REQ-C01, C02)
**Keys.** A new optional `turbulence` section, absent meaning the model is off and the
validation cases unchanged: `model`, `k_epsilon`; `variant`, `standard` (default) or `rng`;
`wall_treatment`, `scalable_wall_functions` (decision 2); `cfl_number`, the pseudo-time Courant
number of section C in (0, 1/2]; `alpha_turbulence` in (0, 1]; `max_iter` and `tol` for the
implicit k and eps solves. In `transport`: `turbulent_schmidt`, a positive float. On a
`velocity_inlet`: `turbulence_intensity` (a fraction in (0, 1)) and `dissipation_length` (m,
positive), each required on every velocity inlet with a nonzero normal velocity when the model is
on and refused otherwise, as `concentration` is. Every key validated for type, range, NaN and
bool (REQ-C02); unknown keys refused. The model constants of section A are module constants of
`src/turbulence.py`, one table per variant, as `JACOBI_WEIGHT` is in `pressure.py`: they define
the published model, and a configured C_mu would be a different model under the same name.

**Modules.** `src/turbulence.py` (new): the k and eps step of section C on a face field, the wall
function values of section B as data, the eddy viscosity; it reuses `limited_face_values` and
the implicit Jacobi of `solver_transport.py` by import rather than copy, which moves the
explicit advection and `_implicit_step` out of `TransportSolver` into module functions both
call, a refactor the transport gate tests must pass bitwise. `src/momentum.py`: the optional
mu_e field, the face rule of D, the stress source, the wall viscosity and the obstacle wall
stencil. `src/solver_staggered.py`: the k and eps step in the outer loop, and a read-only
`eddy_viscosity` beside `face_velocities`. `src/stopping.py`: condition (e), and the recorded
version moved into the rule (section E). `src/solver_transport.py`: the `eddy_viscosity`
argument. `src/config.py`: the keys. `src/boundary_staggered.py`: unchanged; wall distances and
tangential conditions already reach the stencil as data.

**Draft contracts** (SYSTEM.md section 4 gains them when ECR-002 is accepted).

```
turbulence.py:
    VARIANTS = {"standard": ..., "rng": ...}   # C_mu, C_1, C_2, sigma_k, sigma_e (+ R terms)
    TurbulenceState: k, eps, nu_t   # frozen; [ny, nx] float64, read-only; SOLID cells zero
    KEpsilonModel:
        __init__(mesh, config, boundary: StaggeredBoundary)
        initial() -> TurbulenceState        # uniform inlet values, positive
        step(state, faces: FaceVelocities, dt=None) -> TurbulenceState
            one step (section C): dt None is the local pseudo-time step, a float one uniform
            true-time step (VAL-015); k > 0 and eps > 0 at every non-SOLID cell
        wall_viscosity(state, faces) -> dict[edge, ndarray]   # mu_w per wall face (B)
momentum.py:
    MomentumPredictor.predict(u, v, p, mu_eff=None, wall_mu=None) -> MomentumPrediction
        mu_eff [ny, nx] or None (the scalar path, bitwise); wall_mu per wall face, domain
        edges and obstacle faces, or None
solver_staggered.py:
    StaggeredSolver.eddy_viscosity: ndarray | None   # nu_t of the last solve, read-only
stopping.py:
    ErrorEstimateRule.update(step, imbalance, viscosity_step=None)   # (e) when given
    ErrorEstimateRule.version -> int   # 3 without (e), 4 with it; the solver exposes it
solver_transport.py:
    solve_timestep(C_k, faces, size_class, dt, v_ext=None, sources=None, *, eddy_viscosity=None)
```

**Cascade rows.** `turbulence.py -> solver_staggered, tests`: TurbulenceState's fields and the
positivity promise. `momentum.py -> pressure, solver_staggered, solver_transport`: predict's two
optional arguments; with None the coefficients are bitwise today's. `solver_staggered.py ->
solver_transport (through time_integration), the harness, the viewer`: `eddy_viscosity` and the
modified pressure. `stopping.py -> solver_staggered, scripts/benchmark.py,
scripts/stopping_probe.py, scripts/val001_order.py`: (e), and the version read from the rule rather
than the module constant. `config.py -> turbulence, momentum, solver_staggered, solver_transport,
boundary layers`: the section and the segment keys. `solver_transport.py -> time_integration`: the
new keyword.

**Phase 6.** The k and eps step is the transport solver's three loops (the face value per face,
the explicit update per cell with the growth added, the Jacobi sweep per cell with the decay in
the diagonal), run twice, plus a per-cell eddy viscosity and a per-face average for the momentum
conductances. All are data-parallel; the multigrid of decision 4 adds restriction and
prolongation, also per cell.

## J. What this design does not decide
The product mesh: ADR-010 and ADR-011 left it to a measurement on the product case, and the wall
treatment now adds a y+ to that measurement; the 200 x 75 mesh is the default, not a decision.
Thermal effects: buoyancy from hot equipment stays out of scope (SYSTEM.md section 5); a k
equation is where a buoyancy production term would go. Three dimensions: out of scope; item 0
shows what a two-dimensional comparison cannot see. The supply's inlet turbulence: no value is
assumed (decision 7); step 6 runs the product case at two intensities and two dissipation
lengths and records how much the field moves. A time-accurate (unsteady RANS) solve: the
design is steady; if step 6 finds no steady RANS solution, that is a finding to report, not
something to tune away.

## Consequences
**Positive.** The product room gains a flow model a reader can judge against the published
record. The laminar validation stays bitwise. k and eps are positive by the same argument that
makes concentration positive, with one scalar step serving both. The transport solver's
guarantees all carry over; particle mixing becomes turbulent in the bulk, the dominant physics
of the product case. Wall functions keep the product mesh.

**Negative.** The equipment tops are a stagnation flow the model overstates. Wall functions are
used below their range in the slow corners. The model's validation has no clean room-scale
benchmark: the Annex 20 data are three-dimensional where a flat model is compared. Deposition
stays laminar at the walls until the follow-on change. A fifth stopping condition lengthens
every turbulent solve. The pressure solve must change first, and that is a second requirement
change.

## Alternatives considered
At the level of the ECR: an algebraic indoor model, the viscosity raised to a laminar-solvable
value, and laminar with heavier damping (ECR-002 section 3, with the reasons). Within the k-epsilon
family: k-omega and SST, which the survey rates well [2, page 14, remark 6] and decision 1 of
2026-10-04 excludes; a Reynolds-stress model ("marginal improvements ... not well justified by
the severe penalty on computing time", [2]). Each section names its own.

## Sources
1. `docs/reports/product_case_reynolds.md`: sections 2 to 6, the probes, the pressure cost and
   item 0.
2. Zhai, Zhang, Zhang and Chen (2007), "Evaluation of various turbulence models in predicting
   airflow and turbulence in enclosed environments by CFD: Part 1", HVAC&R Research 13(6),
   https://engineering.purdue.edu/~yanchen/paper/2007-8.pdf; the summary remarks on page 14 of
   that file, the RNG comparison on page 8.
3. Nielsen, Rong and Olmedo (2010), "The IEA Annex 20 Two-Dimensional Benchmark Test for CFD
   Predictions", Clima 2010; page 8 of the Aalborg file cites Susin et al. (2009).
4. Nielsen (1990), "Specification of a Two-Dimensional Test Case", Aalborg University, R9040,
   https://vbn.aau.dk/ws/files/197503356/Specification_of_a_TwoDimensional_Test_Case.pdf; page
   numbers are the report's own.
5. The Aalborg measurement workbook, "Laser-doppler measurements of the isothermal two
   dimensional test case", digitized in 2005 from [4] figure 5; location and checksum in [1,
   section 6].
6. [3], figures 4 and 5 and the text under figure 5.
7. Tominaga and Stathopoulos (2007), "Turbulent Schmidt numbers for CFD analysis with various
   types of flowfield", Atmospheric Environment 41; the 0.2 to 1.3 range is from the abstract as
   search results quote it, the paper not fetched (HTTP 403).
8. Lai and Nazaroff (2000), "Modeling indoor particle deposition from turbulent flow onto smooth
   surfaces", Journal of Aerosol Science 31; cited for its role, not checked here.
9. `results/builder33/calc33.py` and `calc33.json` (untracked): the Reynolds numbers, the inlet
   values, the friction-velocity and y+ estimates, the implied kappa, the eddy-viscosity scale and
   the particle relaxation time.
10. Rong and Nielsen (2008), DCE Technical Report 46, Aalborg University, as [3] cites it; not
    fetched.

Launder and Spalding (1974), the standard model and wall functions; Yakhot and Orszag (1986) and
Yakhot et al. (1992), RNG; Kato and Launder (1993), the production limiter; Grotjans and Menter
(1998), scalable wall functions; Patankar (1980), harmonic face coefficients and source
linearisation; Dean (1978), channel skin friction. Cited from their standard use; the build
checks each value it codes against the paper.
