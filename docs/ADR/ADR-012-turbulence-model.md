# ADR-012: Turbulence Model: k-epsilon on the Staggered Solver

## Status
Accepted by Alex on 2026-10-04, with ECR-002. Written 2026-10-04, before the build, as the design for ECR-002
(`docs/ECR/ECR-002-turbulence-model.md`), from the evidence in
`docs/reports/product_case_reynolds.md`; revised the same day after premise review 33, test 33 and
the outlet measurement of prompt 33b (that report's section 8), and again after `/cfd-test 33b`.
Alex took the decisions listed first on 2026-10-04: the eight put to him, each as ranked first,
and a ninth, a risk-retirement probe before anything is built (ECR-002 step 0). The sections carry
the options with their ranking; the record of what was taken is at the head of the decisions. As accepted it
supersedes ADR-004 (laminar flow assumption) and gains a planned-against-built table when ECR-002
closes, as ADR-010's was added at ECR-001 step 9. Bracketed numbers point at the sources at the
end.

## Decisions for Alex
Rewritten on 2026-10-04 after the outlet measurement of prompt 33b ([1], section 8), the premise
review and the test. Each decision is a picture first, then the options ranked with their
consequence, then the section that carries the detail. Decision 1 of 2026-10-04 (a k-epsilon
model, not an algebraic one) is taken; these are what it leaves open. The measurement settled
none of the earlier seven and added one, the outlets, which comes first because ECR-002's
outlet step comes before any coupled solve.

**Taken by Alex, 2026-10-04.** Every recommendation below, and a ninth decision. The options stay
as they were put.

1. The outlets: option (1), the report's T3. The hood exhaust's tangential velocity is held at
   zero, as the segment layer holds it on every segment that is not a pressure outlet
   (`src/boundary_staggered.py`, `_build_tangential`). A floor-return face held shut has only its
   normal velocity held; its tangential condition stays the pressure outlet's zero gradient.
   ECR-002 criterion 6 compares the built treatment against the probe rerun with the hood's
   tangential velocity held at zero, not against the probe as run, which kept it at zero gradient
   (test 33b, B2).
   Amended 2026-10-08 (Alex, prompt 42, ECR-002 step 3): option (2), every opening at a set
   flow, is taken in place of option (1). The four floor returns and the hood are fixed-flow
   outlets, ducted to fans as in a real clean room. A segment states its outward velocity (the
   hood's fan, 0.5 m/s) or states none; those that state none share what the stated ones leave
   of the discrete inflow, at one face velocity. With no pressure outlet left, the pressure
   system is the closed cavity's, which the corrector solves with its projection and pin. The
   objection put to option (2) below, a singular system compatible only when the shares sum to
   the supply exactly, is answered twice: ECR-003's closed-domain path solves that system as it
   stands, and the remainder rule forms the shares from the face widths of the mesh, so they
   sum exactly on any grid. The evidence is the outlet probe
   (`docs/reports/ecr002_step3_outlet_probe.md`, sections 7.3 and 7.7). Option (1) as taken on
   2026-10-04, with the held-shut rule and the open-outlet-faces argument it needed, is not
   built. The zero-gradient copy stays the pressure outlet's rule, correct for developed
   outflow normal to the face, as in VAL-001.
2. The variant: both built; RNG for the product, standard for the published comparisons, both
   run on the product and their difference reported.
3. The wall treatment: scalable wall functions.
4. k and eps: the transport scheme in pseudo-time, growth explicit and decay implicit.
5. The pressure solve: ECR-003, conjugate gradients against multigrid measured on the product
   mesh, landing before ECR-002 step 5, the first step that solves the 200x75 room to tolerance.
   Steps 0 to 4 do not wait for it.
6. The thresholds: Annex 20 option (1), with (4) reported. Both pass marks, the room's and the
   backward-facing step's, stay OPEN until the first coupled results exist, and are then set,
   each with its rationale, against the measurements and the published predictions (Rong and
   Nielsen 2008 for the room), before step 7 scores anything. What is not scored is reported.
7. Turbulent deposition: deferred to a follow-on change.
8. The inputs: Sc_t 0.7, configurable; each inlet states its intensity and dissipation length,
   no value in code.
9. Risk retirement before the build (ECR-002 step 0). The committed laminar solver, through a
   probe-only subclass as prompt 33b's outlets were, with the T3 outlets and a frozen non-uniform
   eddy viscosity from the indoor zero-equation model added to the molecular viscosity: does the
   room's iteration converge at a realistic effective viscosity? Asked before eight steps of model
   are built on the assumption that it does. The outcome decides whether a convergence aid comes
   before step 1. As run (2026-10-04, `docs/reports/ecr002_step0_frozen_viscosity.md`): the
   zero-equation field as published has a core median of 8.2e-3 m^2/s, about 545 times air's
   and above k-epsilon's core range, so the probe kept its shape and scaled it across that range,
   to core medians of 1.5e-3, 5e-4 and 1.5e-4 m^2/s, with the unscaled field as one rung.

**1. How the air leaves the room (section D; ECR-002 step 3).**
*Picture.* Air leaves through four grilles in the floor and the hood's opening in the right
wall. Today each opening is told only that the room's pressure there is zero, so air may flow out
or in, and where it flows in nothing says how. You set the hood as fan-driven at a set flow
(decision 2 of 2026-10-04); the grilles keep the room's pressure and take the rest of the supply.
Measured on the laminar solver ([1], section 8): air comes in through the openings in every run
that diverges, and at ten times the room's viscosity the divergence sits at the hood while it
draws air in. Fixing the hood's flow removes that, and shutting a grille face the air turns back
through slows the growth, but nothing tried makes the room converge above a hundred times the
viscosity, and at the real viscosity the solve goes wrong inside the room before any opening
draws air in. The openings need a rule for entering air whatever else is done, and that rule
will not by itself make the room solvable.
*Options, ranked.*
(1) The hood at its set flow; a grille face the air turns inward through held shut for that
iteration (the report's T3). Measured: the hood never draws air in; on the product mesh the solve
diverges latest, at outer iteration 427 against 267 today; on the coarse room, at ten times and
at the real viscosity, it grows to four or five times the converging runs' speed and holds there,
at 10 to 20 m/s, without diverging over the iterations run (to 1,000 in test 33b); at Re 90 its
residual falls faster than today's (6.0e-6 at outer 1,701 against 1.27e-5 at 1,675), not
converged. Needs a segment type for the hood and the pressure step told each
iteration which grille faces are open.
(2) Every opening at a set flow: the grilles' shares given in the configuration and the pressure
pinned at one cell, as the closed cavity's is. Nothing can draw air in, but the grilles' split,
which the room sets today, becomes an input that must add up to the supply exactly. Not
measured.
(3) The hood at its set flow, the grilles as today (T2). Measured: the hood's divergence goes and
the grilles diverge instead, at ten times and at the real viscosity.
(4) Today's openings. At ten times the viscosity the hood draws air in and the divergence sits
there.
*Recommendation.* (1): it keeps the split you set, it did best of those measured on the product
mesh, and it is the condition CFD codes ship for openings air can turn back through (OpenFOAM's
`inletOutlet`). It is recommended for what the openings do, not for convergence, which ECR-002
step 5 measures.
*Note, 2026-10-08.* Option (2) is the one taken (the amendment under decision 1 at the head of
the decisions). It was "not measured" when this was written; the outlet probe measured it.

**2. Which k-epsilon (section A).**
*Picture.* The model keeps two extra numbers in every cell: how much churning energy the air
carries there, and how fast that churning dies away. From the two it works out how strongly the
cell mixes momentum and particles with its neighbours. The variants differ in the rule for how
fast the churning dies, most where the downflow hits the top of a piece of equipment and stops.
*Options, ranked for the product.*
(1) RNG k-epsilon. The evaluations of indoor flows rank it among the best overall, and "very
well" in low-turbulence forced convection, which the room's core is; its extra term lowers the
mixing where the flow is strained hard, the equipment tops.
(2) Standard k-epsilon. The larger published record, the Annex 20 room and the backward-facing step
included, so it is the variant to compare with those published predictions; known to overstate the
churning where flow strikes a surface head on.
(3) The low-Reynolds variants. Millimetre cells at every wall, "no or marginal improvements" and
"strong case-dependent stability problems" in the survey. Last.
*Recommendation.* Build both (they share every line but the constants and one term): (1) for the
product, (2) for the published comparisons, and run both on the product to report how much they
differ over the equipment tops. No planned case can say which is right where the supply strikes:
none has impingement with measurements (section A).

**3. The wall treatment (section B).**
*Picture.* At every wall the air slows to rest in a layer a few millimetres thick, far thinner
than one of our 4 cm cells. Either we cut the space next to every wall into millimetre cells and
compute that layer, or we use the shape such layers are measured to take, the law of the wall,
to tell the first cell how hard the wall drags on it.
*Options, ranked.*
(1) Wall functions with a floor on the first node's distance in wall units (the scalable form).
Works on the product mesh as it stands, where the first node sits at y+ of about 7 to 70; where
the air is slow, the floor treats a node inside the layer's innermost part as at its edge, which
misstates the drag by a bounded amount instead of failing.
(2) Wall functions without the floor. Invalid in the slow corners and the middle of each
equipment top.
(3) Resolve the layer. Wall cells of 1 to 3 mm at every wall and obstacle face, which the mesh
cannot place on an obstacle's top, at a measured cost in accuracy and pressure sweeps.
*Recommendation.* (1).

**4. How k and epsilon are computed and kept positive (section C).**
*Picture.* Neither new number may go below zero: a negative churning energy means nothing and
makes the mixing strength negative, which blows the solve up. The transport solver already moves
particle concentrations in a way that cannot go below zero; the question is whether to move these
two numbers the same way.
*Options, ranked.*
(1) The transport scheme in pseudo-time, growth explicit and decay implicit: positive at every
step by ADR-011's argument, reused code.
(2) The momentum equation's way, steady with a deferred correction: an intermediate iterate can go
negative and needs clipping.
(3) First-order upwind, steady: positive, but it smears k and eps across the shear layers.
*Recommendation.* (1).

**5. The pressure solve (section H).**
*Picture.* Every outer iteration solves for a pressure correction by passing information one
cell per sweep. On the 200x75 room one correction took 27,000 sweeps to the committed tolerance
and 112,000 to the probes' tighter one, and a converged solve needs thousands of corrections.
Two methods pass the information much faster and keep the per-cell update the GPU port wants.
*Options, ranked.*
(1) A separate change, ECR-003, done before ECR-002's product step, choosing between
Jacobi-preconditioned conjugate gradients (iterations growing with the cells per side, nothing
to build over the obstacles) and multigrid with the present sweep as its smoother (fastest on a
smooth problem, but this one's coefficients vary by orders of magnitude and the obstacles cut the
grid) by measuring both on the product mesh. REQ-S08 is amended either way. The laminar solver
needs it too.
(2) The same inside ECR-002.
(3) Keep plain weighted Jacobi: at least hours per steady product solve, a lower bound.
*Recommendation.* (1). *Answered 2026-10-06:* ECR-003, accepted, chose Jacobi-preconditioned
conjugate gradients (its option B), measured against geometric multigrid, pyamg's algebraic
multigrid and SuperLU (ADR-013). This section's cost table is retaken in
`docs/reports/pressure_solver_ecr003.md`, section 9, and the Annex 20 room at G's 216x72 in its
section 12.1.

**6. The validation thresholds: the Annex 20 room and the backward-facing step (section G).**
*Picture.* Two published rooms or channels with measured air speeds are the model's tests against
reality. For the Annex 20 room, the measured speeds were checked against the air the room takes
in: at the first measuring line they carry about the right amount, at the second the middle
plane carries six tenths of it while a plane near the side wall carries a tenth more than all
of it, so the measured room was not flat there and a flat model cannot match its middle plane.
For both cases the pass mark has to come from how well published runs of the same model did,
and only one such run is in hand.
*Options, ranked, for the Annex 20 room.*
(1) Score the first line and the line along the ceiling jet; report the second unscored. The
pass mark: no worse than the one published standard k-epsilon run on the same lines (Rong and
Nielsen 2008) by an allowance you set, or the spread across that report's four models.
(2) Score both lines against the mean of the two measured planes.
(3) Score the middle plane at both lines with an allowance equal to the measured shortfall.
(4) Add the hot-wire profile from the widest model (W/H 4.7) at the second line, the measurement
nearest a flat room, as a reported reference.
(5) No room comparison.
*For the backward-facing step.* The reattachment length is measured at 6.26 +/- 0.10 step heights;
standard k-epsilon is known to reattach short, by 12 to 15% on a different step in the one
primary source found, and no primary source for this step's standard k-epsilon value was found.
The step reports its result against the measurement until a sourced range exists.
*Recommendation.* Annex 20 (1), with (4) reported; both pass marks OPEN.

**7. Turbulent deposition (section F).**
*Picture.* Particles reach a wall by drifting across the thin still layer next to it. Today that
layer is a fixed 1 mm thick everywhere. With turbulence, how thin it is depends on how hard the
air scrubs that wall, which the wall treatment computes.
*Options, ranked.*
(1) Defer to a follow-on change. Deposition keeps its present rule; the smallest classes'
deposition is understated by an amount this change does not measure.
(2) Include it as ECR-002's last step: a turbulent deposition model, REQ-T09 changed, its own
validation data.
*Recommendation.* (1): the model needs the wall treatment's friction velocity, which exists only
once decision 3 is built.

**8. Two model inputs (sections D, F and I).**
*Picture.* (a) How strongly the churning spreads particles compared with how strongly it spreads
momentum. (b) How much churning the filtered supply air carries when it enters through the
ceiling. (b) matters more than it looks: below the ceiling nothing makes new churning until the
air reaches the equipment, so the supply's own churning sets the room's mixing strength there,
and that is the value the solve's convergence depends on (section D).
*Options, ranked.* (a) A turbulent Schmidt number of 0.7, the conventional value, inside the 0.2
to 1.3 the literature spans; or 1.0. (b) No value in code; each inlet states its intensity and
its dissipation length (in the specification's convention, eps = k^(3/2) / l_e), the product's
set from a sensitivity pair in the product step; or Annex 20's 4% copied to the product now.
*Recommendation.* (a) 0.7, configurable. (b) The first.

## Context
The product room has never been solved. On the committed configuration its Reynolds number on
the room height is 89,500 and its cell Reynolds number on the 0.04 m mesh 1,190; every case the
flow solver was validated on ran at 5 (VAL-001) or 100 (VAL-002) [1, section 2]. As committed,
the solve diverges by outer 58; with the pressure solved toward 1e-8 on a coarse copy of the
room it diverges at the real viscosity and at ten times it, stalls at a hundred times and
diverges there after 1,633 iterations, and at a thousand times its residual falls, still above
the tolerance after 3,000 [1, sections 4 and 8]. Air enters through the pressure outlets in every
run that diverges; giving it a condition moves and delays the divergence and converges nothing
above a hundred times the viscosity [1, section 8].
ADR-004 calls the room laminar; in the clean-room sense, unidirectional and low in turbulence
intensity, it is, but not in the Navier-Stokes sense [1, section 3]. Alex decided on 2026-10-04 to
add k-epsilon (decision 1 of 2026-10-04, ECR-002 section 3.3). This document is the design
ECR-002's steps are built from.

What it plugs into. The staggered solver (ADR-010): p at cell centres, u and v on faces, QUICK by
deferred correction over an upwind implicit matrix with one under-relaxed Jacobi sweep per outer
iteration, a weighted Jacobi pressure solve, and the `error_estimate` stopping rule. Viscosity
enters it in one place, the four diffusion conductances of `MomentumPredictor._assemble`
(`src/momentum.py`, lines 386 to 394), as the scalar `config.mu`. The transport solver (ADR-011):
concentration at cell centres advected by the faces continuity was enforced on, a bounded QUICK
face value under forward Euler, and implicit diffusion and deposition by Jacobi; its diffusivity
is one scalar per class (`src/solver_transport.py`, lines 475 and 824).

## A. The variant (REQ-S14, proposed; decision 2)
Both variants carry, at cell centres, the turbulent kinetic energy k and its dissipation rate
eps, and form the eddy viscosity `mu_t = rho C_mu k^2 / eps`. Standard k-epsilon (Launder and
Spalding 1974):

    div(rho u k)   = div((mu + mu_t / sigma_k) grad k)   + P_k - rho eps
    div(rho u eps) = div((mu + mu_t / sigma_e) grad eps) + C_1 (eps / k) P_k - C_2 rho eps^2 / k
    P_k = mu_t S^2,  S^2 = 2 S_ij S_ij
    C_mu 0.09, C_1 1.44, C_2 1.92, sigma_k 1.0, sigma_e 1.3

RNG (Yakhot et al. 1992) keeps the form with C_mu 0.0845, C_1 1.42, C_2 1.68, sigma_k =
sigma_e = 0.7194, and subtracts from the eps equation `R = C_mu rho eta^3 (1 - eta / eta_0) /
(1 + beta eta^3) eps^2 / k`, `eta = S k / eps`, eta_0 4.38, beta 0.012. These are the values as
usually tabulated; the build checks them against the primary papers before step 1 merges. The
1992 form is not Yakhot and Orszag's 1986 one: Thangam and Speziale (1991) found the 1986 form,
with C_1 = 1.063, reattaching at X/H of about 4 on a backward-facing step measured at 7.1, "overly
dissipative" because C_1 is too close to 1 [12, page 14 and conclusion 3]. The design uses the
1992 constants.

**What each misjudges here.** The supply strikes four equipment tops and stops: a stagnation
region, where the strain is normal rather than shear. P_k = mu_t S^2 grows with any strain, so
the standard model produces k there that a real stagnation flow does not have, the stagnation
point anomaly of the k-epsilon family; the equipment tops will read more turbulent and more
mixed than they are. RNG's R term changes sign where eta exceeds eta_0, which strong strain
produces, and raises eps there, so it lowers mu_t where the standard model overshoots. The zone
above the equipment is the supply flowing down at a few percent intensity: weakly turbulent,
and both high-Reynolds forms only let k and eps decay there, which is the right direction and is
not transition modelling. The low-Reynolds variants add damping functions for the near-wall
region and need a first node near y+ = 1; the survey found that "Most LRN k-eps models and
nonlinear RANS models provide no or marginal improvements on prediction accuracy but suffer from
strong case-dependent stability problems and has long computing time" [2, page 14, remark 4],
which ranks them last. A
production limiter (Kato and Launder 1993) addresses the stagnation overshoot in either variant;
it is not ranked, and is the change to make if the equipment tops show it.

**The record, read whole.** For RNG: the survey reports that "the majority of comparison studies
indicated that the RNG k-eps model is slightly better than the standard k-eps model in terms of
the overall simulation performance", and that Chen (1995) "compared five k-eps based turbulence
models in predicting various convective airflows and an impinging flow. The results showed that
the RNG k-eps model had the best overall performance in terms of accuracy, numerical stability,
and computing time, while the standard k-eps model had competitive performance" [2, page 8]:
a ranking over that set of cases, not on the impinging case alone. Part 2 of the same study,
which tested eight models against measurements in four enclosed flows (a tall cavity, a room
with partitions, a square cavity in mixed convection, a fire room) and not the standard model,
concludes that "the v2f-dav and RNG k-eps models have the best overall performance compared to
the other models in terms of accuracy, computing efficiency, and robustness" and that "In the
forced convection flow with low turbulence levels, the RNG k-eps, the LRN-LS, the v2f-dav, and
the LES all performed very well" [11, pages 15 and 16]; the product's core is forced convection
at low turbulence. For the standard model: the survey's remark 1, that with wall functions it
"provides acceptable results (especially for global flow and temperature patterns) with good
computational economy" [2, page 14]; on the Annex 20 room, Susin et al. (2009), in three
dimensions, found it best against RNG and k-omega [3, page 8], and Rong and Nielsen (2008), in
two dimensions, found it the best of four models (with k-omega, BSL and SST) along both
horizontal lines except in the upper right corner [10, section 2.3].

**Which case can tell the variants apart on impingement.** None in the plan. The Annex 20 room is
a ceiling wall jet that turns down the far wall; nothing strikes a surface head on. The channel
and plane Couette flow have no stagnation point, and the backward-facing step (decision 3 of
2026-10-04, G (ii)) separates and reattaches but has no jet striking a surface. Part 2's four
cases have no downflow onto a surface either. The product room has the impingement and no
measurement. So the plan can measure how much the variants differ on the equipment tops (both on
the product, step 8), and the backward-facing step can rank them on separation and
reattachment against data, but no planned case says which is right where the supply strikes. An
impinging-jet case with published measurements would; none is sourced in this pass.

**Ranking, re-derived (premise B4).** For the product: (1) RNG, on the evidence over indoor
flows, the low-turbulence forced convection of the product's core, and its correction where the
strain is strong; (2) the standard model, whose record in this kind of room is the largest. Both
are built (they share every line but the constants and the R term). The standard model is kept
for the comparisons whose published predictions are standard k-epsilon ones (Annex 20 [10], the
backward-facing step [12]); step 8 runs both on the product and reports k, nu_t and the
concentration over the equipment tops and at the two sensors half a metre above them
(`above_gap_1`, `above_gap_2`, y = 2.5 m).

**Cost.** Two scalar equations per outer iteration on the cell field, each the transport step of
section C; RNG adds one source term. Against the pressure solve both are small (section H).

## B. Wall treatment (REQ-S17, proposed; decision 3)
**y+ on the product mesh.** The tangential velocity's first node sits half a cell from the wall,
0.02 m on the 0.04 m mesh (`wall_distance`, ADR-010 decision 2). With air's nu = mu / rho =
1.508e-5 m^2/s, `y+ = u_tau 0.02 / nu = 1326 u_tau`. The friction velocity is estimated from a
flat plate, because no field exists: outer speeds 0.1 to 0.45 m/s (the supply, and the slower
flow turning along floors and equipment tops), run lengths 0.3 to 3 m (an equipment top is 0.6
to 1.3 m), Cf from the turbulent correlation `0.0592 Re_x^-0.2` and the laminar `0.664
Re_x^-0.5`. Together they give u_tau from 0.005 to 0.031 m/s, the run Reynolds numbers being
2,000 to 90,000, below the turbulent correlation's range, which an estimate made without a field
cannot avoid [9]. The floor returns and the hood carry the supply's 3.8 kg/s through 3.1 m of
openings, about 1 m/s, and at 1 m/s over 2 m the same correlation gives u_tau 0.053 m/s
(premise review S9; [14]). Hence:

| u_tau (m/s) | where | y+ at 0.02 m |
|---|---|---|
| 0.005 | slow corners, 0.1 m/s over 3 m | 6.6 |
| 0.01 | 0.1 to 0.2 m/s flows | 13 |
| 0.02 | 0.2 to 0.45 m/s over 1 to 3 m | 27 |
| 0.03 | 0.45 m/s over 0.3 to 0.6 m | 40 |
| 0.053 | the outlet flows, 1 m/s over 2 m | 70 |

and toward zero at each stagnation point. Standard wall functions want the first node in the
log layer, y+ above about 30; below about 11 the node sits in the viscous sublayer, where the
log law is not the profile. The product mesh spans that range: about 7 to 70, below 11 in the
slow corners and near the middle of each equipment top.

**On the Annex 20 grid** (section G): along the ceiling jet the outer speed is 0.82 u0 at x/H 1
and 0.64 u0 at x/H 2 (the workbook's y = h/2 line [5]), u_tau about 0.02 m/s by the same
correlation at 0.37 m/s over 3 m. A wall cell of h/4 = 0.042 m puts the first node at y+ 27;
h/8 = 0.021 m at 14 [9]. The proposed grid (section G) uses 0.042 m.

**What each treatment asks of the mesh.** Wall functions (Launder and Spalding 1974) bridge the
layer: the wall shear on the first node is `tau_w = rho C_mu^(1/4) k_P^(1/2) kappa u_P /
ln(E y*)`, `y* = rho C_mu^(1/4) k_P^(1/2) y_P / mu`, kappa 0.41, E 9.793, with k's wall flux zero,
its production in the wall cell taken from tau_w, and eps in the wall cell set to `C_mu^(3/4)
k_P^(3/2) / (kappa y_P)`. Using k rather than u_tau as the velocity scale keeps the shear finite
at a stagnation point, where u_P falls to zero. The scalable form (Grotjans and Menter 1998)
evaluates the log law at `max(y*, y*_0)`, so a node inside the sublayer is treated as at its
edge: the error is bounded where (2) would be undefined. The floor y*_0 is where the linear law
`u+ = y+` meets the log law `u+ = ln(E y+) / kappa`; for kappa 0.41 and E 9.793 that is 11.53
[14]. The 11.06 first written here belongs to kappa 0.41 with B = 5.2 (E = 8.43), and at 11.06
the stated log law gives u+ = 11.43, 3% off the sublayer edge it stands for (premise review S1);
the build takes the floor from the constants it codes, one consistent set. The mesh keeps its
0.04 m cells. In code the wall function is a per-face wall viscosity, `mu_w = rho C_mu^(1/4)
k_P^(1/2) kappa y_P / ln(E max(y*, y*_0))`, put where `mu` now stands in the two wall rows of
`_assemble` (`src/momentum.py`, lines 389 to 394), so the existing wall stencil `(phi_P -
phi_wall) / wall_distance` carries it unchanged.

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
the wall function on it. That is part of ECR-002's momentum step, and it changes the laminar
solver only in rooms with obstacles, none of which has a validated result.

*Note, 2026-10-08 (step 4 built).* The paragraph above describes `momentum.py` before step 4.
Step 4 built the obstacle wall stencil: the wall at the obstacle face, half the unknown's cell
away, with the wall viscosity from the `wall_mu` hook when given, and, by Alex's extension of
2026-10-08, Leonard's boundary form in the QUICK correction there; a face bounding SOLID on one
side only is a wall over its whole span. Section D's step 4 note gives the measurements. The
lines of `_assemble` cited in this section have moved; the wall rows now read `wall_mu` when it
is given. The wall function itself is step 6's.

*Note, 2026-10-09 (step 6 built; prompt 45).* The wall functions are `src/turbulence.py`'s
`TurbulenceBoundary`, built per solver from the staggered boundary layer. kappa 0.41 and E 9.793
are module constants; the floor y*_0 is computed from them (`log_law_floor`, 11.528), never typed.
A wall is a domain-edge face with zero normal velocity that is not an outlet, at rest or moving
(`StaggeredBoundary.wall_faces`, read at the same coverage as the tangential condition), and every
obstacle face, the step 4 corner rule included; an inlet that admits air and either kind of outlet
take no wall function and keep the stencil's own viscosity. `wall_viscosity` puts `mu_w = rho
C_mu^(1/4) k_P^(1/2) kappa y_P / ln(E max(y*, y*_0))` on each such face in `wall_mu`'s layout, y_P
the unknown's half cell and k_P the mean of the two cells the unknown lies between. At y* = y*_0 it
is mu exactly; below the floor it is `mu y* / y*_0`, less than mu. Three rules the design left to
the build, stated in the class's docstring. (1) The wall cell's production per unit density is
`(tau_w / rho) u_k / (kappa y_P)`, `u_k = C_mu^(1/4) k_P^(1/2)`, tau_w the wall function's shear on
the cell-centre slip: the standard form, `P_k = tau_w (dU/dy)_P` with the log law's gradient
(Launder and Spalding 1974; Versteeg and Malalasekera 2007, chapter 9); in the log layer it equals
the held eps. (2) A cell with more than one wall face takes the arithmetic mean of its walls' eps
and productions: the single-wall rule on one wall, symmetric under reflection, and OpenFOAM's
corner weighting. (3) The inflow-weighted means of the inlet faces' k and eps are the uniform
state the coupled solve starts from. The inflow values are `k = 1.5 (I |u_n|)^2`, `eps =
k^(3/2) / l_e`; every other domain face carries the adjacent cell's value.

## C. Discretization of k and eps (REQ-S15, proposed; decision 4)
**Where they live.** At cell centres, beside p and the concentrations, so that the corrected
faces advect them with the fluxes continuity was enforced on (ADR-011 A), the scheme is exact on
a uniform field (ADR-011 B), and the production term reads the velocity gradients where the
staggered layout makes them exact: du/dx and dv/dy at a cell centre are face differences;
du/dy and dv/dx live at the cell corners and are averaged from the four corners to the centre.

**The step.** Recommended option (1): each outer iteration advances k and eps by one
pseudo-time step of the transport solver's scheme. Per cell P with local step `dt_P = cfl_t /
(max(|u_w|, |u_e|) / dx + max(|v_s|, |v_n|) / dy)`, `cfl_t <= 1/2`, capped at the largest finite
dt_P in the field so that a cell at rest takes a finite step. Step 3 is implicit in the decay and
would stay positive without the cap; step 2's explicit growth would not stay bounded where the
corner gradients give P_k > 0, so the cap serves steps 1 and 2 (test 33 S4):

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
I = 0.04 and l_e = h / 10, its specification's equations (5) to (7) as written [4, page 2]. The
convention is named because the other one is common: a source writing `eps = C_mu^(3/4)
k^(3/2) / l`, as commercial-code documentation often does, maps to `l_e = l / C_mu^(3/4) =
6.09 l`, and read into this key unconverted it would set eps 6.1 times too high and the inlet
eddy viscosity 6.1 times too low (premise review S3).
Outlet: the upwind cell's value on outflow, which every face of a fixed-flow exhaust carries. A
pressure-outlet face the air turns inward through takes what decision 1 gives it: held shut (option
1), it carries no advective flux and k and eps see a face with zero diffusive flux; left open
(options 3 and 4), it takes the adjacent cell's value (zero gradient), where ADR-011 E brings clean
air for a concentration, since k has no clean state. Walls and obstacle faces: no advective flux
(the normal velocity is zero, REQ-S12), zero diffusive flux of k, eps in the wall cell held at the
wall function's value (section B) by a dominant diagonal, P_k in the wall cell from the wall shear.

*Note, 2026-10-09 (step 6; prompt 45): `cfl_number` 0.5 and the coupled iteration.* At the bound,
`cfl_number` 0.5, the coupled plane Couette solve of G (ii) locks into a limit cycle and never
stops: a period of about 10.3 outer iterations in u, k, nu_t and v alike, the wall cell's k
swinging 0.6% (a 6 m channel, 60 by 12 cells, standard model). The period and the amplitude do
not change with the momentum sweeps (5, 10, 20), `alpha_velocity` 0.3 or `alpha_turbulence` 1.0,
so it lives in the k and eps step, not the flow's iteration. At 0.25 the same case stops by its
rule at outer 642, and every VAL-016 run uses 0.25. ADR-011 B's bound is the limited scheme's
positivity bound under forward Euler; at the bound itself the steady pseudo-time iteration need
not converge. Product runs (prompt 46) need a value below 0.5.

*Note, 2026-10-09 (step 6; prompt 45): positivity needs a converged correction.* The step's
positivity (above) holds for corrected faces that close every cell. On the 48-row Couette case (a
120 m channel in 0.25 m by 6.25 mm cells) every early correction stopped at the committed cap of
5,000 CG iterations, the first leaving a cell imbalance of 23% of the flux scale; the solve then
diverged as a whole, the imbalance reaching 1e24 times the flux scale and k 1e22, the implicit k
and eps solves stopping at their own cap (GitHub issue 58's case), and k went negative at outer
iteration 6, where `PositivityError` stopped it. With the cap at 100,000 the same twelve outer
iterations converge every correction (about 15,000 iterations each), keep the imbalance near 1e-11
of the flux scale and keep k and eps smooth and positive. The assertion did its job; the cause
was the capped correction, which the solver already reports (`pressure_cap_hits`). `python
docs/reports/probe45/positivity45.py 5000` and `... 100000` reproduce it (records
`results/builder45/couette/positivity_*.json`).

*Note, 2026-10-09 (step 6; prompt 45): a known limitation, the strain beside a wall.* The step's
strain takes du/dy at the four corners of a cell from the face differences and averages them to
the centre (above). In the second cell from a wall, where the profile is the wall function's
log law, that overstates the strain on every grid, so the production nu_t S^2 there, and with it
k, is too large; the excess diffuses into the core. Measured on plane Couette flow (G (ii)) against
the model's own refined answer (the 1D solve with the same wall cells, 161 sub-cells per 2D cell),
the second cell's du/dy is 34.4% high and S^2 81% high for the standard model, 32.8% to 33.0%
and 76% to 77% for RNG, the same on 12 and 24 rows; its k is 29% to 41% high (1.29 to 1.41). The
figures first reported, 33% and 77% (standard) and 31% and 71% (RNG), k 27% to 39%, were against
a reference refined to 21 sub-cells; refining it to 41, 81 and 161 raises them by about one
point. The orchestrator's estimate, 12.5% in du/dy and 27% in S^2, took the exact face gradients
of a log law; the stencil's own difference quotients give 20.7% and 46% on an exact log law. The
measured ratio R = S_st / g(1.5 h), S_st the stencil's value at the second cell's centre and
g(1.5 h) the refined solve's true gradient there, is the product of three factors:

- F1 = 1.125, averaging the exact face gradients of a log law (1 / h and 1 / (2 h)) against its
  gradient at the centre (1 / (1.5 h));
- F2 = 1.0730, the difference quotients between cell centres in place of those exact gradients,
  on the same log law ((ln 3 + ln(5 / 3)) / 2 / (1 / 1.5) = 1.2071 = F1 F2);
- F3 = R / (F1 F2), measured: the solved profile against the log law. It is the stencil on the
  solved profile over the stencil on the log law with the same u_tau (1.110 standard, 1.161 RNG,
  12 rows), times the log law's centre gradient over the true one (1.003 and 0.948). Most of it
  is the step from the wall cell to the second cell, 15% to 19% larger than the log law's.

Standard, 12 rows: 1.125 x 1.0730 x 1.1130 = 1.3435; RNG, 12 rows: 1.125 x 1.0730 x 1.0999 =
1.3277. `python docs/reports/probe45/mech45.py` reproduces the table (about a minute; record
`results/builder45/couette/mech45.json`). The limitation applies beside every wall and obstacle
face, the equipment tops included, wherever a wall cell's neighbour sits in the log layer. In the
core of plane Couette flow it raises k above `u_tau^2 / sqrt(C_mu)` by 6% to 15% on 12 rows and 2%
to 4% on 24 (G (ii)'s note). A fix (a gradient consistent with the wall function in the second
cell, or production limited there) is a candidate for a later step, not this one.

## D. Coupling into the flow solver (REQ-S14, S16, S18 and S01, proposed; decision 1)
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
but the views; the contract says which p is returned. The pressure outlets then hold the
modified pressure at their datum, so the static pressure at each differs from it by `-(2/3) rho
k` of the cell behind it: about 0.012 Pa at 10% intensity on a 1 m/s outlet, 2% of its dynamic
pressure, and different at each outlet (premise review S2). It concerns only the outlets held
at a pressure: a fixed-flow exhaust (decision 1) holds a velocity. ECR-002 step 4 decides whether
to correct the datum per face; VAL-018 records the value at every pressure outlet either way.

**Diagonal dominance and Jacobi.** The upwind matrix's coefficients are `a_nb = D_f + max(+-F_f,
0)` with `D_f = mu_e A_f / d_f >= 0`, and `a_P = sum a_nb + net F`. A larger mu_e raises D_f in
a_nb and a_P alike, so dominance holds for any non-negative mu_t, and the one Jacobi sweep per
outer iteration is unchanged in kind. What changes is the cell Peclet number. In the room's core,
below the ceiling and above the equipment, little produces k, so nu_t there is the supply's: 5.0e-5
to 1.5e-3 m^2/s for intensities of 2% to 10% and dissipation lengths of 5 to 30 cm in the design's
convention (section C), and `U dx / nu_e` at 0.45 m/s on 0.04 m is 12 to 280 instead of 1,190 (12
to 363 on nu_t alone) [14]. The first version took nu_t as 1e-3 m^2/s and the cell Peclet number as
18; that 1e-3 was the other convention's, 6.1 times larger for the same inputs (premise review B2).
The hypothesis: with the implicit diffusion carrying more of the face coupling than the explicit
deferred correction, the outer iteration converges where it diverges laminar. The laminar
measurement leans against it: on the 40x15 room at uniform viscosities of 1.5e-4 and 1.5e-3 m^2/s,
inside that range, nothing converged under any outlet treatment measured. At 1.5e-4 every run
diverged, grew or oscillated; at 1.5e-3 today's outlets stalled from about outer 600 and diverged
at 1,633, and the fixed-flow hood stayed bounded and unconverged to its cut at 1,001 and, continued in
test 33b, departed near outer 2,000, past 5 m/s at 2,165 against 1,277 [1, sections 4 and 8]. Those runs are on a grid five times coarser, so their cell Peclet numbers are five times
the product mesh's at the same viscosity, and a field large in the shear layers is not a uniform
one. ECR-002 step 0 probes the hypothesis before anything is built, step 5 measures it with the
built code before any coupled solve, and VAL-018 is conditional on it (G (iv)).

**The outlets (decision 1; premise review B1).** Today each pressure outlet face takes its interior
neighbour's velocity whatever its sign (`StaggeredSolver._extrapolate_outlets`) and is corrected
against p' = 0 (`PressureCorrector._face_d` borrows the interior d), so air entering through an
outlet has no condition of its own. The design assumed outflow there; the measurement [1, section
8] found air entering in every diverging run, at the hood at Re 8,950 where the divergence sits
under today's outlets, and found that no condition tried converges the room above Re 895. The
outlets need a condition for entering air before any coupled solve: the turbulent iteration would
inherit today's, and k and eps need a defined value on every face air crosses (section C). Option
(1) of decision 1, as it would be built: the hood exhaust becomes a segment type of its own, a
fixed-flow exhaust whose faces hold a configured outward velocity in the staggered layer, as an
inlet's hold an inward one, and whose tangential velocity is held at zero, as on every segment
that is not a pressure outlet (`_build_tangential`; decision 1 as taken); in the concentration layer it is an outflow, the upwind cell's value
carried out; `get_total_inlet_flux` does not count it, and the configuration refuses exhausts whose
total reaches the supply, so the floor returns at room pressure carry the rest as outflow. At the
floor returns, each outer iteration `_extrapolate_outlets` holds at zero normal velocity every face
whose extrapolated velocity points into the room, leaving its tangential condition the pressure
outlet's zero gradient, and gives the pressure corrector the faces left open; today the corrector fixes its outlet masks at construction (`src/pressure.py`, lines 161 to
164), so the open faces become an argument of the correction. A face the extrapolation leaves open
can still be turned inward by the correction (up to three per return at once on 40x15, counted
at every iteration in test 33b); it is reconsidered the next iteration. Option (2) adds the floor returns' shares as configured velocities
and leaves no Dirichlet pressure row, so the corrector's system is singular and compatible only
when the shares sum to the supply exactly, the closed cavity's case. Not ranked: the backflow
condition at the hood as well, with the hood at room pressure (the report's T1), which contradicts
decision 2 of 2026-10-04 and departed earlier at Re 895; and a total-pressure inflow, where
entering air takes the room's pressure less its dynamic head, which was not measured. VAL-001's
outlet is expected to carry outflow at every face in every iteration and VAL-002 has no outlet, so
both would stay bitwise under (1); the step checks it.

*Note, 2026-10-08 (step 3 built; decision 1 amended).* Option (2) was built, not option (1). The
floor returns and the hood are `fixed_flow_outlet` segments. The optional `velocity` is outward.
With no pressure outlet in the configuration at least one fixed-flow outlet states none, and
those share the remainder of the discrete inflow at one face velocity
(`StaggeredBoundary.fixed_flow_velocities`), so outflow equals inflow to rounding on any mesh;
with a pressure outlet, every fixed-flow outlet states its velocity. A remainder that is zero or
negative on the mesh is refused when the boundary is built, naming inflow, stated outflow and
remainder, in place of "refuses exhausts whose total reaches the supply" above. Faces hold the
velocity as a Dirichlet value with zero tangential velocity; the concentration layer carries the
upwind cell's value out and deposits nothing; `get_total_inlet_flux` and
`get_max_boundary_velocity` do not count them. The corrector is unchanged:
`has_pressure_outlet` is False, so it takes its closed-domain path. `_extrapolate_outlets` is
unchanged and stays the pressure outlet's rule, a zero-gradient copy that is correct for
developed outflow normal to the face (VAL-001's outlet, which stays bitwise). Where outlet cells
take air sideways it cannot close the cell and the room's pressure climbs without end (GitHub
issue 61), which is why the product room has no pressure outlet. The probe's arm D0 is the
reference the built room reproduces bit for bit (ECR-002 criterion 6).

**The outer iteration.** Momentum prediction, pressure correction (unchanged but for the open
outlet faces), then the k and eps step of section C on the corrected faces, then `mu_t = (1 - a_t)
mu_t_old + a_t rho C_mu k^2 / eps` with `alpha_turbulence` = a_t. The k and eps step reads the
corrected faces for the reason the transport solver does. `alpha_velocity` keeps its meaning; a
value for the turbulent cases is a measurement of ECR-002 step 5, not a guess here. The pressure
correction's d = A / a_P reads the larger a_P and needs no change.

*Note, 2026-10-09 (Alex, before step 6; prompt 45).* Three decisions on step 5's outcome
(`docs/reports/ecr002_step5_convergence.md`, section 7). (1) The product solver settings, for when
the product room runs the model: `momentum_sweeps` 10, the only count that converged anything on
the finer grids; `pressure_rtol` 1e-4, which reproduced 1e-8's outer count to the iteration on the
fixed-flow room, with one row at 1e-8 kept as a check (ADR-013 decision 3, amended the same day);
`max_simple_iter` 10,000, since the error-estimate stop came 1.6 to 2.5 times after the
velocity-step stop in every converged row and 500 cuts every converged fine-grid row. They are
applied to `configs/clean_room_default.yaml` in step 8 and used by the product measurement of
prompt 46. (2) Step 4's corner rule stays; it is revisited only if a later step shows it deciding
convergence. (3) The coupled solve comes first, with no convergence aid built in advance. If the
coupled product room does not converge (prompt 46), the step stops there and the aid is chosen
then, from step 0's ranking (`docs/reports/ecr002_step0_frozen_viscosity.md`, section 6.6).

*Note, 2026-10-08 (step 4 built; the outlets' datum dropped).* Alex decided on 2026-10-08 that
step 4 builds nothing for the outlets' datum. The modified pressure `p + (2/3) rho k` disagrees
only with a boundary that holds the pressure; the product room has had no pressure outlet since
step 3 (decision 1 as amended), and VAL-001, which has one, runs laminar. "ECR-002 step 4 decides
whether to correct the datum per face" above is answered: no correction. The `momentum.py`
contract in `docs/SYSTEM.md` says which pressure the solver returns.

*Note, 2026-10-08 (step 4 built; the obstacle stencil and its corner rule).* The face rule, form
b's stress source and the momentum sweep count are the step 0 probe's arithmetic
(`frozen34.py`), and `MomentumPredictor.predict(u, v, p, mu_eff=...)` reproduces its
FrozenPredictor bit for bit on the 40x15 product room before the obstacle stencil. The obstacle
faces take the domain edge's wall stencil (section B, "Obstacle walls") in two parts. The
diffusion: the wall at the face, half the unknown's cell away, wall value zero, the viscosity
from `wall_mu` when given. The QUICK correction, added by Alex on 2026-10-08 beyond the prompt's
text: with only the diffusion, a channel whose floor is a row of SOLID cells still differed from
the domain-floor channel by 1.3e-4 m/s, 1.3e-3 of the 0.1 m/s inflow, where the flow develops,
because QUICK took its far-upstream node from the zero stored at the SOLID face's location; with Leonard's boundary
form there, as at a domain edge, the two channels agree to 6e-16 m/s. The corner rule is the
same for both parts: a neighbour face that bounds a SOLID cell on one side only, at an
obstacle's corner, is a wall over its whole span. The diffusion takes the half distance over
the whole span. In the QUICK correction the face takes the wall value under both schemes and
adds nothing, and the unknown beside it takes the wall as its far node. The viscosity field's
face rule treats such a SOLID cell as it treats a domain edge, with the value of the non-SOLID
cell across the row boundary. On the product room the stencil moves the 40x15 ladder's stops
from 391 and 1,632 to 728 and 2,115 outer iterations and the 80x30 drift case's from 177 to
176, with no divergence and no drift (prompt 43's pull request). The corner rule's QUICK half
carries the whole rise at Re 895: the diffusion alone stops at 375, and with the corner faces'
correction removed (the face back to QUICK, a probe in `docs/reports/probe43/`) the run stops at
372. At Re 8,950 the same probe never converges in 3,000 outer iterations, though it stays
bounded, while the committed rule converges at 2,115. The rule's correction is zero, so a corner
face is advected by upwind with its true mass flux: first order at the corner, conservative, and
the more dissipative choice. Whether it still slows the outer loop on finer grids, where corners
are a smaller share of the room, is a question for step 5 (prompt 43b's pull-request section).

*Note, 2026-10-09 (step 6 built; prompt 45): the outer iteration.* With the configuration's
turbulence section `StaggeredSolver.solve_steady` runs the outer iteration above in that order:
the prediction with `mu_eff = mu + rho nu_t` and `wall_mu` from the wall functions (section B's
note) over `MomentumPredictor.stencil_viscosity`, the stencil's own viscosity on every face the
wall functions do not set; the correction; one pseudo-time step of k and eps on the corrected
faces; then nu_t relaxed by `alpha_turbulence`. k and eps are not relaxed. A prescribed
`eddy_viscosity` is refused with the section present (one source of nu_t per solve), and a
configuration with no velocity inlet that admits air is refused when the solver is built, since
the start and condition (e)'s scale come from the inlets. A PositivityError from the step stops
the solve and names the outer iteration. Without the section every call is the laminar one, the
rule's and the predictor's alike, so a probe's or a test's substitute still fits and VAL-001, its
stretched twin, VAL-002 and the 40x15 ladder are bitwise the base at every commit of the step.
The solver exposes the last solve's `turbulence_state` (k, eps, nu_t, read-only) in place of the
draft's `eddy_viscosity` attribute.

## E. The stopping rule (REQ-S01, clarified; rule version 4)
ADR-010's conditions (a) to (d) bound the velocity's iteration error, the per-cell imbalance, the
summed imbalance over the through-flow and the signed domain sum. Under k-epsilon the iterate
is (u, v, p, k, eps). Condition (a) estimates the remaining error from the velocity step and its
fitted geometric rate; that is valid for the joint iteration when its slowest mode shows in the
velocity. A mode living mostly in eps in a quiescent corner moves mu_t, and so the velocity,
slowly, and the velocity's rate can understate it.

Condition (e): the same estimate on the eddy viscosity, `step_nu * rho_hat / (1 - rho_hat) /
nu_scale < iteration_error_tol`, with step_nu the largest change of nu_t over one outer iteration,
rho_hat fitted over RATE_WINDOW steps as in (a), and nu_scale from boundary data fixed per
solve: the molecular nu plus the largest inlet eddy viscosity `C_mu k_in^2 / eps_in`. A field
maximum would move with the iterate and the grid, and one cell's overshoot (the stagnation
anomaly of section A) would loosen (e) everywhere; the rule's existing scales are boundary data
for the reason `src/stopping.py` gives for the flux scale, "A flux through the domain, not one
through a cell: a scale that moved with the grid would move (c) with it" (premise review S4). It
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

*Note, 2026-10-09 (step 6 built; prompt 45).* Built as drafted: `ErrorEstimateRule(...,
nu_scale=None)`, `update(step, imbalance, viscosity_step=None)`, with nu_scale the molecular nu
plus the largest `C_mu k_in^2 / eps_in` over the inlet faces, and `version` 3 without (e), 4 with
it. The imbalance is asked for only when (a) and, with it on, (e) hold. `RULE_VERSION` is gone;
`RULE_VERSION_WITHOUT_E` and `RULE_VERSION_WITH_E` name the two values, and
`solver_staggered.rule_version(config)` gives a solver's version without building one, which
`scripts/val001_order.py` needs to decide reuse before it builds a solver; `scripts/benchmark.py`
and `scripts/stopping_probe.py` read the same. The two tests section I's cascade rows name
changed with it: `tests/test_benchmark.py` (the import and the version assertion) and
`tests/test_stopping_probe.py` (the version an argument of `rule_parameters`). A third, which
those rows did not foresee, changed too: `tests/test_val001_order.py`, which patched
`val001_order.RULE_VERSION` and now patches `rule_version` (review 45 S3). In every VAL-016 run,
on both variants and every grid, (e) is the last of the five conditions to hold.

## F. Coupling into transport (REQ-T13, proposed; decisions 7 and 8)
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
implicit system: in the room's core nu_t is 5.0e-5 to 1.5e-3 m^2/s for inlet intensities of 2%
to 10% and dissipation lengths of 5 to 30 cm in the design's convention (section D; [14]), so D_t
is 7e-5 to 2.1e-3 m^2/s at Sc_t 0.7; at the top of that range, with dt 0.01 s and dx 0.04 m, the
diffusion number is 1.3e-2 and a sweep contracts the error by about 0.05, so a few sweeps
suffice. What changes in kind: Brownian diffusion was unresolved everywhere, cell Peclet numbers
2.6e7 to 3.7e9 (ADR-011 G); with D_t in that range the core's cell Peclet number is about 8 to
250, and in the shear layers, where nu_t is larger, lower; diffusion now shapes the bulk field.
VAL-003 already tests the operator at D = 1e-3 m^2/s and a diffusion number of 0.1; a per-face
coefficient needs the dense-solve unit test extended to a piecewise D.

**Deposition (decision 7).** ADR-011 deposits through `D / delta` with delta 1 mm fixed
(REQ-T09), plus settling on floors. In turbulent flow the near-wall concentration layer thins as
the wall shear grows, and indoor deposition models make the deposition velocity a function of
the friction velocity, Lai and Nazaroff (2000) the usual one [8]. That needs u_tau at every wall
face, which only the wall treatment of decision 3 provides once built, and it changes REQ-T09's
parameterization and needs its own validation data. Recommended: defer to a follow-on change;
D_t is zero at a wall face, so the wall flux keeps today's rule and the change stays separable.

## G. Validation (REQ-S14 to S18, REQ-T13; decision 6, and decision 3 of 2026-10-04)
Cases (ii) to (v) have non-unit density, a non-square domain and inputs off grid nodes, the
Phase 2 lessons. Case (i) reproduces VAL-001 and VAL-002 as they stand, at unit density and, for
the cavity, on a square domain, so it cannot honour them (test 33 S2). The VAL identifiers are
proposed and fixed when Alex accepts ECR-002.

**(i) The laminar limit.** With the model off, `val001_80x40`, `val001_80x40_stretched` and
`val002_80x80` reproduce their stored rows bitwise: the same outer count and stop, and a hash of
the returned faces equal to one taken at the step's base. The transport gate tests pass
unchanged with `eddy_viscosity` None. Criterion: equal bits (REQ-S16). Verified in every step
that touches momentum.py, solver_staggered.py or solver_transport.py. The baseline is the main
branch at each step's base: ECR-003 (decision 5), if it goes first, changes every laminar bit
and every stored outer count, the rows are retaken under it, and those are what later steps
reproduce (premise review S11).

**(ii) The model's implementation, with known answers.**
*VAL-015, decaying turbulence.* A closed 2.4 m by 1.5 m box, rho 1.2, zero velocity, uniform
initial k0 and eps0: the k-epsilon equations reduce to `dk/dt = -eps`, `deps/dt = -C_2 eps^2 /
k`, whose solution is `k = k0 (1 + (C_2 - 1) eps0 t / k0)^(-1 / (C_2 - 1))` with eps from it.
Stepped in true time with a fixed dt, the computed k and eps must follow it with an error that
falls at first order in dt (observed order between 0.9 and 1.1 over three steps), and stay
uniform to rounding. It tests the decay terms and their implicit treatment against an exact
answer; at rest the production is zero.

*VAL-016, plane Couette flow (premise review B3).* The channel first proposed here assumed the
shear stress constant over its comparison window, and in a channel it falls as `tau_w (1 -
y/delta)`. A one-dimensional solve of the same model with the wall-function values at y+ = 30
[15] puts the slope of u+ against ln y+ over that window (y+ 30 to 0.2 delta in wall units, ADR
case: H 0.3 m, u_tau 0.073 m/s, Re_tau 726) at 1.10 of 1 / kappa_m, and k at 0.76 to 1.00 of
`u_tau^2 / sqrt(C_mu)`: a correct implementation fails a 3% criterion on both, on every grid.
The review reached the same conclusion under local equilibrium (slope 0.95, k 0.80 to 0.96). In
plane Couette flow the total stress is constant across the gap, and the model's answer has a
sharp, exact property: wherever k is uniform the k equation reduces to production equal to
dissipation, which with constant stress makes `k = u_tau^2 / sqrt(C_mu)` whatever eps does, and
the wall function's balance in the wall cell gives the same value. The same one-dimensional
solve gives k within 0.5% of it across the core at Re_tau 2,984 (1.7% at 726, the molecular
viscosity's share), and the log-layer slope within 2% of 1 / kappa_m [15]. Case: a 0.3 m gap,
the lower wall at rest and the upper wall moving at U_w, both walls with wall functions (the
moving wall is a zero-normal velocity inlet, as the cavity's lid is), rho 1.2, air, the channel
long enough for the profile to stop changing (about 40 gaps; the step measures it), inlet
uniform at U_w / 2 (an antisymmetric Couette profile carries exactly that, so the developed
section has no pressure gradient), U_w of order 15 m/s for Re_tau about 3,000 on this gap (a
log-law estimate), chosen off the grid's rounding. Compared in the developed section: u / U_w and
k / u_tau^2 against a one-dimensional solve of the same equations with the same wall treatment,
written for the test
as the reference, and the core k against `u_tau^2 / sqrt(C_mu)` with u_tau from the constant
stress `(nu + nu_t) du/dy` at mid-gap. Criteria: the core k within 1%; the profiles within an
allowance set from the measured difference on the coarser of two grids, stated in the step
before the finer one runs, the gap falling under refinement. The plane channel stays as a
reported physics case: its skin friction against Dean's (1978) correlation `Cf = 0.073
Re_m^-0.25`, which the build fetches and checks before quoting, unscored. Couette tests the model
as built; the channel and the step below test it against physics.

*Note, 2026-10-09 (step 6 built; Alex's decision of that day; prompt 45): VAL-016's criterion is
split.* Under wall functions refinement moves the first node's y+ (G (v)), so "the gap falling
under refinement" is not defined: the reference itself changes with the wall cell. And the
criterion as written mixed two questions, whether the code solves its equations and whether the
model has the property, which the measurement separates. As built
(`tests/test_turbulent_channel.py`, the reference in `tests/couette_reference.py`, which imports
nothing from `src/`):
(a) the implementation: in the developed section the 2D solve equals the one-dimensional solve
on the same grid (the 2D stencil's own x-invariant limit: the same wall cells, the same face rule,
the same corner-averaged strain) in u / U_w and k / u_tau^2 to within 1e-4, each k over its own
u_tau. A wall-cell production scaled by 1.01 fails it.
(b) the model: the refined one-dimensional solve (41 sub-cells per 2D cell, the same wall cells)
keeps the core k (0.2 H to 0.8 H) within 1% of `u_tau^2 / sqrt(C_mu)`, each variant against its
own C_mu.
(c) reported, unscored: the 2D core k against `u_tau^2 / sqrt(C_mu)` and the first node's y+ on
every grid run. It is above the model's by the second-cell strain overshoot of section C's note.
The suite runs (a) on one grid and one variant (the standard model on 12 rows, about 46 s) and
(b) for both variants; the matrix is `docs/reports/probe45/couette45.py` and `tables45.py`.
The case: U_w 15.7 m/s (Re_tau 2,990 to 3,070), the inlet at U_w / 2 with an intensity of 6% and
a dissipation length of 0.1 m, a 120 m channel (400 gaps), 1 m cells along x on 12 rows and 0.5 m
on 24 (aspect ratios near 40), `cfl_number` 0.25 (section C's note), ten momentum sweeps. Every run stops by `error_estimate_and_continuity` under rule version 4, condition (e)
the last of the five to hold, and no k or eps solve reaches its cap.

| Variant | Rows | first-node y+ | (a) max du / U_w | (a) max d(k / u_tau^2) | (b) core k ratio | (c) 2D core k ratio |
|---|---|---|---|---|---|---|
| standard | 12 | 256 | 5.5e-8 | 1.3e-5 | 0.9995 to 1.0015 | 1.058 to 1.155 |
| standard | 24 | 127 | 1.1e-7 | 1.8e-5 | 0.9985 to 0.9986 | 1.016 to 1.042 |
| RNG | 12 | 252 | 2.7e-7 | 1.4e-5 | 0.9994 to 1.0011 | 1.059 to 1.134 |
| RNG | 24 | 125 | 8.5e-7 | 7.6e-5 | 0.9984 to 0.9985 | 1.018 to 1.042 |
| standard | 48 | 63 | not run to its stop | | 0.9977 to 0.9984 | 1.004 to 1.012 (1D, same grid) |
| RNG | 48 | 62 | not run to its stop | | 0.9974 to 0.9981 | 1.005 to 1.014 (1D, same grid) |

The 48-row 2D runs were stopped unfinished: on those cells (0.25 m by 6.25 mm) a converged
correction takes about 15,000 CG iterations, so an outer iteration takes about 9 s and a run some
hours, and at the committed cap of 5,000 the solve diverges (section C's note on positivity). Their
(c) values are the same-grid one-dimensional solve's, which (a) shows the 2D one equals on 12 and
24 rows.

The development length is about 200 gaps, not the 40 written above. On a 90 m channel in 0.3 m
cells, against the profile at 270 gaps, the change is 1.5e-2 of U_w in u and 5e-2 in k (relative to
its maximum) at 60 gaps, 6e-4 and 2e-2 at 120, 1.8e-5 and 1.7e-3 at 180 and 1.7e-5 and 2.2e-4 at
210; on the 120 m channels the profile changes by about 1e-6 of U_w or less and 1e-5 in k between 240
and 270 gaps. The streamwise spacing changes the developed answer by less than 1e-5 of U_w (0.1
against 0.3 m). The core is
slow because its k decays before the shear from the walls reaches it. The plane channel, reported
and unscored: its skin friction from the developed pressure gradient (and, equally to 1e-8, from
the wall function's shear) is 0.977 and 0.981 of Dean's correlation for the standard model on 12
and 24 rows and 0.953 and 0.950 for RNG, with `Cf = tau_w / (rho U_m^2 / 2)` and `Re_m = U_m 2h /
nu` = 156,000, 2h the full gap. Dean (1978), J. Fluids Eng. 100:215-223, was checked from its
abstract only; the full-height convention is the one channel DNS matches.

*VAL-019, the backward-facing step (decision 3 of 2026-10-04).* The NASA Turbulence Modeling
Resource's case of Driver and Seegmiller (1985) [13]: Reynolds number about 36,000 on the step
height H, the inflow boundary layer about 1.5 H thick before the step, Mach number about 0.128
(incompressible here), reattachment at x/H = 6.26 +/- 0.10, velocity and turbulence profiles at
x/H = 1, 4, 6 and 10, with data files for the profiles, the pressure and the skin friction.
Standard k-epsilon reattaches short; what is sourced: on the Kim, Kline and Johnston step
(measured X/H about 7.1), Thangam and Speziale (1991) found the standard model with wall
functions at X/H about 6.0, "approximately a 15% underprediction", and 6.25 with three-layer
wall functions, within 12%; the 1986 RNG at about 4; and earlier reports of 20 to 25% short
traced to under-resolution and to outflow conditions closer than X/H 25 to the step [12, file
pages 11, 14 and 15]. On Driver and Seegmiller's step itself the Resource lists computed results
for one k-epsilon variant (k-e-Rt) and none for the standard model, and this pass found no
primary source for a standard k-epsilon reattachment length on it (figures of 4.5 to 5 appear
in search results from tutorial pages and are not used). The criterion is therefore OPEN: it is
set against a sourced range of standard k-epsilon results on this step, the route being the
Resource's files and a primary standard k-epsilon result; until then the step reports the
reattachment length and the four profiles, with [12]'s 12 to 15% as context. From [12] the case
takes an outflow boundary at least 30 H downstream and a resolution study before any number is
read. It ranks the variants on separation and reattachment (section A), not on impingement.

**(iii) The Annex 20 room, conditional on item 0.** The specification [4, pages 2 and 3] and
the measurements [5] are in hand and their check is in [1, section 6]: at x/H = 1.0 the measured
symmetry-plane profile carries 1.02 to 1.32 of the inlet flux u0 h, depending on how the strips
between the outermost readings and the walls are closed (1.17 with no slip); at x/H = 2.0 it
carries 0.60 to 0.64 under any closure, while the plane z/W = 0.4, digitized by the builder from
the specification's figure 6, carries 1.11 to 1.13. The two planes differ by half the inlet flux
at the same section. That the shortfall is air crossing the width is inferred from the two
planes and consistent with the specification's figure 10, the hot-wire profiles at x/H = 2.0
for W/H = 4.7, 1.0 and 0.5 [4, page 13; 1, section 6]. A two-dimensional solution carries u0 h
through every section, so it cannot match the symmetry plane at x/H = 2.0 closer than an
integrated 0.021 u0 H, about 0.04 u0 if it sits in the return flow below mid-height; the text
introducing [3]'s figure 5 says a two-dimensional low-Reynolds k-epsilon prediction leaves the
counter flow "slightly underestimated" [6].

Case: L 9.0 m, H 3.0 m, slot h 0.168 m at the top of the left wall, outlet t 0.48 m at the foot
of the right wall, u0 0.455 m/s, nu 15.3e-6 m^2/s (Re 5,000 on h; 89,200 on H, within 0.4% of
the product room's, a coincidence of numbers and not a similarity, since Nielsen bases the
Reynolds number on the slot "because the flow in the ceiling region and in the rest of the room
is strongly influenced by the inlet conditions" [4, page 2]), inlet k and eps from I = 0.04 and
l_e = h / 10. Grid 216 x 72, uniform 0.0417 m, so the slot spans 4.03 cells and its edge falls
off a face; the inlet velocity is scaled so the covered faces carry u0 h exactly. Compared: u /
u0 along x/H 1.0 and 2.0 and along y = h/2 and y = H - h/2, the four lines the specification
names [4, page 3].

The threshold's source, checked (test 33 B1). [3]'s figure 4 is x/H = 2.0 only (standard and
stream-function k-epsilon and a one-equation model) and its figure 5 one low-Reynolds model at
x/H 1.0 and 2.0; neither has the y = h/2 line. Rong and Nielsen (2008) [10], fetched for this
pass, holds one standard k-epsilon prediction (Ansys CFX 11.0, wall functions with y+ above 11,
4,736 cells after a three-grid independence study) beside k-omega, BSL and SST, on all four lines
(x = 3 and 6 m, and y = 0.084 and 2.916 m measured up from the floor, the last being the
specification's ceiling-jet line y = h/2). So one published standard k-epsilon prediction exists
at the lines option (1) scores, and no spread of standard k-epsilon results does. The route
changes: the published prediction's own departure from the data, line by line, digitized from
[10]'s figures 5 and 9 with the method of [1, section 6], is the reference a standard k-epsilon
run here is held to (no worse than it by an allowance Alex sets), or the spread across [10]'s
four models is. Voigt (2000), which [10] cites for the same room, was not fetched. The number
stays OPEN (decision 6). A further reference the decision did not list: the W/H = 4.7 hot-wire
profile at x/H = 2.0 and Re 7,100 ([4] figure 10), the measurement nearest two dimensions at that
section, jet and floor return only (premise review S5).

**(iv) The product room converges.** VAL-018: `configs/clean_room_default.yaml` with the model
on, `stopping_rule: error_estimate`, `mass_imbalance_tol` from ADR-011 G's formula at its t_end,
stops by `error_estimate_and_continuity` under rule version 4 within its cap, on the product
mesh. Reported: the outer count, wall time, the stop's five readings, nu_t / nu over the room,
both variants' k, nu_t and concentration over the equipment tops and at the two sensors above
them (section A), and y+ at every wall node, which section B estimated without a field. VAL-018
is conditional on the convergence hypothesis of section D: ECR-002 step 5 measures convergence
at the core's effective viscosity before the coupled solve, and if that measurement fails, VAL-018
changes rather than the tolerance.

*Note, 2026-10-09 (Alex; prompt 45).* The product's purpose is comparative, a stakeholder need now
stated in `docs/SYSTEM.md` section 1: the tool says where particles accumulate and how a layout
change moves that; it does not defend absolute counts. VAL-018's criterion follows from it, set
for step 8: the deposition hotspots and the ranking of layouts stable under grid refinement, under
the two k-epsilon variants, and across the turbulent Schmidt number's literature range (0.2 to
1.3, section F). Convergence within the cap, above, is what makes the comparison possible; the
stability of the ranking is what the product is judged on.

*(vi) Two cases added (Alex, 2026-10-09; prompt 45). Thresholds OPEN until Alex sets them from
first results.*
*VAL-020, a planar impinging slot jet.* Section A found that no planned case tells the standard
model from RNG where a jet strikes a surface; this one does. Data: Khayrullina, van Hooff, Blocken
and van Heijst (2017), Experiments in Fluids 58(4):31, DOI 10.1007/s00348-017-2315-0 (open
access), with the same authors' steady RANS comparison (2019), European Journal of Mechanics
B/Fluids 75:228-243, DOI 10.1016/j.euromechflu.2018.10.003. Alternatives closer to the product's
regime: Ashforth-Frost, Jambunathan and Whitney (1997), Experimental Thermal and Fluid Science 14;
Zhe and Modi (2001), Journal of Fluids Engineering 123(1):112-120.
*VAL-021, a trend-level concentration check.* The two-dimensional model's high- and
low-concentration regions against a measured three-dimensional room, compared for trend, not
point by point. Data: Zhang and Chen (2006), Atmospheric Environment 40(18):3396-3408, DOI
10.1016/j.atmosenv.2006.01.014, or Murakami, Kato, Nagano and Tanaka (1992), ASHRAE Transactions
98(1):82-97.

**(v) Grid convergence under wall functions (premise review S10).** VAL-008 (Phase 4) asks for an
observed order within 0.2 of the theoretical one. Under wall functions refinement moves the
first node's y+, and below the scalable floor the model itself changes, so an observed order is
not defined in the usual way. VAL-008 applies to the laminar solver and the transport scheme;
turbulent runs report a grid sensitivity on two meshes with every first node inside the
wall-function range, not an observed order.

## H. Cost (REQ-S08; decision 5)
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
2.4 [1, section 4]. A converged product solve needs at least thousands of outer iterations: the
laminar VAL-001 channel at Re 5 needed 3,988 on 80x40 and the Re 100 cavity 12,849 on 80x80
(ADR-010), and the product case is coupled, on 15,000 cells, at a higher effective Reynolds
number. At a third of the first correction's count and 0.28 ms a sweep, 4,000 iterations at 1e-8
are 4,000 x 37,000 x 0.28 ms, about 11.5 hours per steady solve, and at 1e-6 about 3 hours: a
lower bound, three times larger at the cavity's count (premise review S8). The 216 x 72 Annex 20
grid is of the same order. k and eps add two scalar steps of a few array passes and a few sweeps
each per outer iteration, small beside that. Weighted Jacobi is workable at 40 x 15 and not at
the product mesh: decision 5 brings the deferred pressure-solver question forward, and REQ-S08,
whose text names Jacobi, is amended whichever replacement is chosen. The laminar product solve
has the same need: in the 200 x 75 probes with a 5,000-sweep cap every correction stopped at its
cap.

**The candidates (premise review S7).** Two keep the per-cell, data-parallel update REQ-S08 was
written for and map to one thread per cell on Phase 6's GPU:
- *Jacobi-preconditioned conjugate gradients.* The p' system is symmetric (each face coefficient
  `rho d_face A_face` is shared by its two cells) and positive definite with the outlets'
  Dirichlet rows; on the closed validation cavity it is singular and compatible, on which
  conjugate gradients converges, the pin applied after the solve as it is now. Each iteration
  is a matrix-vector product and two global reductions; the iteration count grows as the cells
  per side rather than its square; nothing has to be built over the staircase obstacles.
- *Multigrid with the weighted Jacobi sweep as its smoother.* A V-cycle removes the N^2 factor
  for a constant-coefficient Poisson problem in a few tens of sweeps' worth of work per
  correction. Here d = A / a_P varies by orders of magnitude between the outlet jets and the dead
  corners and SOLID cells cut the grid, so a geometric multigrid needs operator-dependent coarse
  grids to come near that figure; the figure is typical of the textbook problem, not measured
  here.
The ranking between them is ECR-003's, measured on the product mesh; this design needs one of
them before ECR-002's product step.

## I. Configuration and modules (REQ-C01, C02)
**Keys.** A new optional `turbulence` section, absent meaning the model is off and the
validation cases unchanged: `model`, `k_epsilon`; `variant`, `standard` (default) or `rng`;
`wall_treatment`, `scalable_wall_functions` (decision 3); `cfl_number`, the pseudo-time Courant
number of section C in (0, 1/2]; `alpha_turbulence` in (0, 1]; `max_iter` and `tol` for the
implicit k and eps solves. In `transport`: `turbulent_schmidt`, a positive float. On a
`velocity_inlet`: `turbulence_intensity` (a fraction in (0, 1)) and `dissipation_length` (m,
positive; l_e in `eps = k^(3/2) / l_e`, with no C_mu^(3/4), section C), each required on every
velocity inlet with a nonzero normal velocity when the model is on and refused otherwise, as
`concentration` is. Under decision 1's options (1) to (3), a segment type for a fixed-flow exhaust
with its outward face velocity (m/s, positive), refused when the exhausts' total reaches the
supply's. Every key validated for type, range, NaN and bool (REQ-C02); unknown keys refused. The
model constants of section A are module constants of `src/turbulence.py`, one table per variant, as
`JACOBI_WEIGHT` was in `pressure.py` until ECR-003 retired it with the weighted sweep on
2026-10-06: they define the published model, and a configured C_mu would be a different model
under the same name.

**Modules.** `src/turbulence.py` (new): the k and eps step of section C on a face field, the wall
function values of section B as data, the eddy viscosity; it reuses `limited_face_values` and the
implicit Jacobi of `solver_transport.py` by import rather than copy, which moves the explicit
advection and `_implicit_step` out of `TransportSolver` into module functions both call, a refactor
the transport gate tests must pass bitwise. `src/momentum.py`: the optional mu_e field, the face
rule of D, the stress source, the wall viscosity and the obstacle wall stencil.
`src/solver_staggered.py`: the outlet condition per outer iteration (D), the k and eps step in the
outer loop, and a read-only `eddy_viscosity` beside `face_velocities`. `src/pressure.py`: the open
outlet faces as an argument of the correction. `src/stopping.py`: condition (e), and the recorded
version moved into the rule (section E). `src/solver_transport.py`: the `eddy_viscosity` argument.
`src/config.py`: the keys and the exhaust segment type. `src/boundary_registry.py`,
`src/boundary_staggered.py`, `src/boundary_concentration.py`: the fixed-flow exhaust (D); wall
distances and tangential conditions already reach the stencil as data.

*Note, 2026-10-08.* `src/pressure.py` gains no open-outlet-faces argument
(`correct(prediction, p, open_outlets=None)` below is not built), and `src/solver_staggered.py`
needs no outlet condition per outer iteration beyond the copy it has. The exhaust segment type is
`fixed_flow_outlet`, with the remainder rule of section D's note in place of the refusal when the
exhausts' total reaches the supply's.

*Note, 2026-10-08 (step 4 built).* The momentum contract below was built as drafted for
`predict`; `wall_mu` was given a layout of its own, the one `docs/SYSTEM.md`'s `momentum.py`
contract states: a dict with keys "u" and "v", each [ny+1, nx+1] indexed by corner, read at the
domain-edge wall faces and the obstacle faces. `KEpsilonModel.wall_viscosity`'s draft return,
`dict[edge, ndarray]` per domain edge, is superseded by that layout, which is what step 6 must
produce. `StaggeredSolver.eddy_viscosity` below is not built; step 4 built a `solve_steady`
keyword, `eddy_viscosity`, that holds a prescribed field for the solve (`docs/SYSTEM.md`, the
`solver_staggered.py` contract).

*Note, 2026-10-09 (step 6 built; prompt 45).* Built against the draft below. The inlet keys as
drafted: required on every velocity inlet that admits air when the section is present, refused
on any other segment, on an inlet with zero normal velocity and with the section absent; and the
section is refused at load under `velocity_step`. `KEpsilonModel.wall_viscosity` is not built: its
final form is `TurbulenceBoundary(mesh, config, boundary).wall_viscosity(k, elsewhere)`, which
returns `wall_mu` in the step 4 layout, mu_w on the wall-function faces and `elsewhere`'s value,
`MomentumPredictor.stencil_viscosity(mu_eff)` in the solver, on every other face the stencil
reads. The same class builds each step's `TurbulenceConditions` (`conditions(state, faces)`),
the start (`initial_values`) and condition (e)'s inlet scale (`largest_inlet_eddy_viscosity`).
`StaggeredBoundary` gains `wall_faces()` and `MomentumPredictor` `stencil_viscosity(mu_eff)`, two
additions the draft did not list; `turbulence.py` now imports `boundary_registry` (the inlet
coverage, as the concentration layer reads it) and `boundary_staggered`. The stopping rule as
drafted (section E's note); `StaggeredSolver.eddy_viscosity` is `turbulence_state`.

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
pressure.py:
    PressureCorrector.correct(prediction, p, open_outlets=None) -> PressureCorrection
        open_outlets: per edge, the pressure-outlet faces open this outer iteration
        (decision 1); None is every outlet face, today's path bitwise
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
scripts/stopping_probe.py, scripts/val001_order.py, tests/test_benchmark.py (which imports the
constant and asserts on it), tests/test_stopping_probe.py (which monkeypatches it)`: (e), and the
version read from the rule rather than the module constant (test 33 S3). `config.py ->
turbulence, momentum, solver_staggered, solver_transport, boundary layers`: the section, the
segment keys and the exhaust type. `pressure.py -> solver_staggered`: the open outlet faces.
`boundary_staggered.py, boundary_concentration.py -> solver_staggered, solver_transport`: the
exhaust's faces. `solver_transport.py -> time_integration`: the new keyword.

**Phase 6.** The k and eps step is the transport solver's three loops (the face value per face,
the explicit update per cell with the growth added, the Jacobi sweep per cell with the decay in
the diagonal), run twice, plus a per-cell eddy viscosity and a per-face average for the momentum
conductances. All are data-parallel; the pressure solver of decision 5 adds restriction and
prolongation (multigrid) or global reductions (conjugate gradients).

## J. What this design does not decide
The product mesh: ADR-010 and ADR-011 left it to a measurement on the product case, and the wall
treatment now adds a y+ to that measurement; the 200 x 75 mesh is the default, not a decision.
Thermal effects: buoyancy from hot equipment stays out of scope (SYSTEM.md section 5); a k
equation is where a buoyancy production term would go. Three dimensions: out of scope; item 0
shows what a two-dimensional comparison cannot see. The supply's inlet turbulence: no value is
assumed (decision 8); step 8 runs the product case at two intensities and two dissipation
lengths and records how much the field moves. A time-accurate (unsteady RANS) solve: the
design is steady; if step 8 finds no steady RANS solution, that is a finding to report, not
something to tune away. Convergence: whether the coupled iteration converges at the core's
effective viscosity is measured by ECR-002 step 5, not decided here (section D).

## Consequences
**Positive.** The product room gains a flow model a reader can judge against the published
record. The laminar validation stays bitwise. k and eps are positive by the same argument that
makes concentration positive, with one scalar step serving both. The transport solver's
guarantees all carry over; particle mixing becomes turbulent in the bulk, the dominant physics
of the product case. Wall functions keep the product mesh. Air turning back through an outlet
gets a condition of its own, which the laminar solver gains too.

**Negative.** The equipment tops are a stagnation flow the model overstates. Wall functions are
used below their range in the slow corners. The model's validation has no clean room-scale
benchmark: the Annex 20 data are three-dimensional where a flat model is compared. Deposition
stays laminar at the walls until the follow-on change. A fifth stopping condition lengthens
every turbulent solve. The pressure solve must change first, and that is a second requirement
change. The outlet condition changes the laminar product solve, which has no validated result,
and does not by itself make it converge.

## Alternatives considered
At the level of the ECR: an algebraic indoor model, the viscosity raised to a laminar-solvable
value, and laminar with heavier damping (ECR-002 section 3, with the reasons). Within the k-epsilon
family: k-omega and SST, which decision 1 of 2026-10-04 excludes. The survey's remark 6 is
stronger than "rates well": "Most existing studies indicate that the SST k-omega model (Menter,
1994) has a better overall performance than the standard k-eps and the RNG k-eps models, but a
systematic evaluation (especially for modeling indoor airflows) is needed before a solid
conclusion can be reached" [2, page 14]. The reasons it is not put to Alex are in the sources this
design holds: two-dimensional SST on the Annex 20 benchmark shows "a large recirculating flow in
the occupied zone below the supply slot" that "does not corresponds to the Laser-Doppler
measurements" [3, page 6], and is furthest from the measurements along the floor line in [10]
(section 2.3); Part 2 finds that "the SST k-omega model has exhibited problems for low turbulence
flows" [11, page 16], which the product's core is (premise review S6). A Reynolds-stress model
("marginal improvements ... not well justified by the severe penalty on computing time", [2]).
Each section names its own.

## Sources
1. `docs/reports/product_case_reynolds.md`: sections 2 to 6, the probes, the pressure cost and
   item 0; section 8, the outlet measurement of prompt 33b.
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
6. [3], figures 4 and 5 and the paragraph introducing figure 5 (file page 4, above it).
7. Tominaga and Stathopoulos (2007), "Turbulent Schmidt numbers for CFD analysis with various
   types of flowfield", Atmospheric Environment 41; the 0.2 to 1.3 range is from the abstract as
   search results quote it, the paper not fetched (HTTP 403).
8. Lai and Nazaroff (2000), "Modeling indoor particle deposition from turbulent flow onto smooth
   surfaces", Journal of Aerosol Science 31; cited for its role, not checked here.
9. `results/builder33/calc33.py` and `calc33.json` (untracked): the Reynolds numbers, the inlet
   values, the friction-velocity and y+ estimates, the implied kappa, the eddy-viscosity scale and
   the particle relaxation time. Its eddy-viscosity scale is in the C_mu^(3/4) convention, 6.1
   times the design's for the same inputs; [14] restates it in the design's (premise review B2).
10. Rong and Nielsen (2008), "Simulation with different turbulence models in an annex 20 room
    benchmark test using Ansys CFX 11.0", DCE Technical Report 46, Aalborg University,
    https://homes.civil.aau.dk/pvn/cfd-benchmarks/two_d_literature/2005_2010/Rong_l_and_P_V_Nielsen_2008.pdf
    (the address the benchmark page's literature list links), SHA-256
    07d2ceba2fdefffce5fd02ba97a1da479c885cac3785de17fbddead3274722d4; fetched for prompt 33b.
11. Zhang, Zhang, Zhai and Chen (2007), "Evaluation of various turbulence models in predicting
    airflow and turbulence in enclosed environments by CFD: Part 2", HVAC&R Research 13(6),
    https://engineering.purdue.edu/~yanchen/paper/2007-9.pdf, SHA-256
    2417c037420a657c0efa17cafc1ea6b5250ee52f9bdaefc17e74a29548a0f89c; page numbers are the
    file's.
12. Thangam and Speziale (1991), "Turbulent separated flow past a backward-facing step: a
    critical evaluation of two-equation turbulence models", ICASE Report 91-23 (NASA CR-187532),
    https://ntrs.nasa.gov/api/citations/19910012138/downloads/19910012138.pdf, SHA-256
    15e23b8043b373dea9e00d91768287cc35773b5d75fc278873a9203a9ed7d0a5; the journal version is
    AIAA Journal 30(5), 1992. Page numbers are the file's.
13. NASA Turbulence Modeling Resource, "2D Backward Facing Step",
    https://tmbwg.github.io/turbmodels/backstep_val.html, read 2026-10-04: Driver and Seegmiller
    (1985), the flow conditions, the reattachment location and the profile stations quoted in
    G (ii).
14. `results/builder33b/calc33b.py` and `calc33b.json` (untracked; the script is in [1]'s
    appendix): the core eddy viscosity in the design's convention, the scalable floor for the
    stated constants, y+ at the outlet flows.
15. `results/builder33b/oned_keps.py` and `oned_keps.json` (untracked; in [1]'s appendix): the
    standard model across a plane channel and plane Couette flow in one dimension, with the
    wall-function values at y+ = 30.

Launder and Spalding (1974), the standard model and wall functions; Yakhot and Orszag (1986) and
Yakhot et al. (1992), RNG; Kato and Launder (1993), the production limiter; Grotjans and Menter
(1998), scalable wall functions; Patankar (1980), harmonic face coefficients and source
linearisation; Dean (1978), channel skin friction. Cited from their standard use; the build
checks each value it codes against the paper.
