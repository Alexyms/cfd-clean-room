# ECR-002 Step 3: The Outlet Probe

**Date:** 2026-10-08
**Tree:** branch `docs/ecr002-outlet-probe` from main at 745b498. No change under `src/`,
`validation/`, `configs/` or `tests/`: every treatment runs through a probe subclass, as prompt
33b's did. No harness row.
**Instruments:** `docs/reports/probe41/` (committed with this report): the arms as subclasses of
the committed solver and corrector, the rooms, the runners and the table script. Raw output under
`results/builder41/` (untracked).
**Order:** this header and sections 1 to 5 are committed before any script is written or any run
is made (the commit that adds this file). Section 5.2's predictions are the builder's, written at
the same time. Sections 6 onward are written after the runs and say so.

## 1. The question

Each outlet cell is a bathtub with a drain, the outlet face. Before every outer iteration
`StaggeredSolver._extrapolate_outlets` resets each open outlet face to its interior neighbour's
velocity, a zero-gradient copy. Where the steady flow carries air sideways into an outlet cell, the
copied drain is too small, the cell overflows by the same amount every iteration, and the pressure
correction answers by raising the room's pressure enough to push the excess through the outlet
faces. The next copy throws that answer away. The velocity is steady and the mean pressure climbs
by a fixed step every iteration, indefinitely (issue #61; `docs/reports/pressure_solver_ecr003.md`
sections 8.3 and 12.4: +0.0882 Pa per outer iteration under T3, +0.0587 under the committed
outlets, on 80x30 at a thousand times air's viscosity).

The committed treatment mixes two textbook conditions. It copies the interior velocity as an
outflow condition does, but never corrects that copy for continuity; and it holds the pressure as
a pressure outlet does, but discards the velocity the correction produced. ADR-012 D's step 3
design (T3: the hood as a fixed-flow exhaust, reversed floor-return faces held shut) keeps the copy
at the open returns, so it is not expected to remove the drift. This probe measures what does,
before step 3 is built and before ADR-012 decision 1 is revisited.

Read first, as prompt 41 lists them: `CLAUDE.md` and `docs/REVIEW_POLICY.md`; ADR-012 section D
and decision 1's options; ECR-002 sections 5.3 (REQ-S18), 8 step 3 and 9 criterion 6; the ECR-003
report's sections 8.3 and 12.4 and appendix N (`stall36.py`); `docs/reports/product_case_reynolds.md`
section 8 and appendix G (`outlet33b.py`); `src/solver_staggered.py`, `src/pressure.py`,
`src/boundary_staggered.py`, `configs/clean_room_default.yaml`; issue #61.

## 2. The arms

All on T3's footing unless stated: the hood a fixed-flow exhaust at 0.5 m/s outward with zero
tangential velocity (as ADR-012 D builds it, not 33b's zero gradient), and on the pressure-outlet
returns any face whose velocity points into the room held at zero normal velocity for that outer
iteration and removed from the correction's open faces.

- **A. T3 as designed.** The returns' open faces copied from the interior before every prediction.
  The reference: it must reproduce #61.
- **B. The correction sticks.** On the open return faces the normal velocity the pressure
  correction produced is kept into the next outer iteration; the interior copy is used only to
  start (outer iteration 1) and for a face that reopens after being held shut. The pressure
  outlet's textbook treatment.
- **C. Local continuity.** Before each prediction every open return face takes the velocity that
  closes its own cell given the cell's other faces (the outflow condition's correction, per cell).
- **D. Every return fixed-flow.** Each floor return holds a configured outward velocity, the
  remaining supply after the hood split over the returns at equal face velocity (proportional to
  open area), so inflow and outflow balance exactly. No pressure outlet remains: the pressure
  system is the closed-domain one, singular and compatible, solved with ECR-003's projection and
  pin. If the committed corrector refuses or mishandles this layout through a subclass alone, that
  arm stops and the report says exactly what `src/` would need.
- **E. One return at pressure.** As D, but the largest-area return stays a pressure outlet under
  arm B's rule (or C's, whichever removes the drift on its own; B if both do), and the other three
  carry fixed flows summing to the supply less the hood less an equal-velocity share for that
  return. **E0, its control:** the same layout with that return under A's copy rule.

## 3. The measurements

1. **The drift case.** 80x30 at a thousand times air's viscosity (Re 90), as ECR-003 sections 8.3
   and 12.4 ran it, all arms with `pressure_rtol` 1e-8 (committed), and arms A, B, C and D also at
   1e-4 and 1e-2. To 20,000 outer iterations or the error-estimate stop, whichever first. Per run:
   the mean pressure change per outer iteration over the last 100; `||b||` of the last correction
   and where it sits (the share on cells beside open outlet faces); the worst-cell and domain-sum
   imbalance at the end; the outer iterations at the velocity-step and error-estimate stops;
   reversed and held-shut faces per segment, most at once.
2. **The ladder.** 40x15 at Re 895 and Re 8,950 (prompt 33b's rungs, its alpha_velocity 0.5), all
   arms at 1e-8, to 3,000 outer iterations or divergence (speed above 100 m/s). Per run: the end
   state, the least residual and the residual at the end, the largest speed and its cell, reversed
   faces.
3. **VAL-001 80x40** under arms B and C (A is the committed path there, since VAL-001's single
   outlet sees no reversal): the outer count and stop against the `staggered-cg` baseline rows,
   the REQ-S02 metric against the baseline's 4.107e-4 to three figures, the face hash against the
   baseline's (section 2.3 of `docs/reports/ecr003_step2_baseline.md`), and the mean pressure
   change per outer iteration at the stop under the committed path (whether VAL-001 drifts at
   all).
4. **Arm D's flow against arm B's** on the drift case at their stops: the largest velocity
   difference and where, so the report says how much fixing the returns' flows changes the room.

## 4. Method (planned; section 6 records what was built and every adaptation)

The drift case is ECR-003 section 8.3's room as `common36.py`'s `product_t3` built it and
`outer37.py` reran it under the built CG solve: `configs/clean_room_default.yaml` regridded to
80x30, viscosity times 1,000, `alpha_velocity` 0.5, `alpha_pressure` 0.3 (the file's), ten
momentum Jacobi sweeps per outer iteration (`frozen34.py`'s `FrozenPredictor` with a zero
eddy-viscosity field, which step 0 showed is the committed assembly to the bit), from rest,
`error_estimate` with `iteration_error_tol` 1e-6 and `mass_imbalance_tol` 1e-4 rho V_min / t_end
= 2e-8 kg/s per metre (ADR-011 G), `max_pressure_iter` 5,000. Under those settings the CG solve at
1e-8 stopped at 588 outer iterations with the velocity-step stop at 233 (`results/builder37b/
c3_80x30_mu1000.json`, 2026-10-06), the counts the drift rows of sections 8.3 and 12.4 carry.

The ladder is prompt 33b's: 40x15, `alpha_velocity` 0.5, one momentum sweep (the committed
predictor, as `outlet33b.py` ran it), from rest, `velocity_step` with `convergence_tol` 1e-6, to
3,000 outer iterations or 100 m/s. Re 895 is the viscosity times 100 and Re 8,950 times 10 (the
report's `L100` and `L10` rungs; Re = rho 0.45 H / mu with H = 3 m).

The adaptations the prompt anticipates: the tolerance key is `pressure_rtol` (ADR-013 decision
4), the sweep cap is a CG iteration cap of 5,000, the correction reports `iterations`, `products`
and `reached_cap` instead of `sweeps`, and `IterationState` carries `pressure_iterations`. The
hood's tangential velocity is held at zero by test 33b's `tangD` wrapper on
`StaggeredBoundary.tangential_conditions` (every right-edge tangential location that is not
already Dirichlet, which on this configuration is the hood's, becomes Dirichlet zero), applied
before the predictor is built so both the wall shear and the QUICK boundary node see it. The
arms are an override of `StaggeredSolver._extrapolate_outlets` that writes the return and hood
faces and sets the corrector's `_out_bottom` and `_out_right` masks for that iteration, the
mechanism `outlet33b.py` used; arm D also sets the corrector's `needs_pin` and `pin_cell` so the
committed `correct` runs its closed-domain projection and pin. The scripts import nothing from
`results/`: the ten-sweep predictor is a subclass of `MomentumPredictor` carrying `frozen34.py`'s
`_sweep_n` on the committed `_assemble`, and a control compares it with `FrozenPredictor` from
`results/builder34/` bitwise over 50 outer iterations before the arms run.

Per correction the probe corrector records `||b||` of the right-hand side the committed
`mass_imbalance` forms from u* and v*, and the share of `||b||^2` on the cells beside the open
outlet faces of that iteration. The mean pressure is the mean of p over non-SOLID cells after each
outer iteration; its change per outer iteration is the mean of the differences over the last 100.
The velocity-step stop is the first outer iteration whose residual is below `convergence_tol`
(1e-6), recorded on the way under the error-estimate rule.

VAL-001 runs the harness's own preset (`validation.cases.load_preset("val001_80x40")`), its
configuration unchanged (`alpha_velocity` 0.7, `error_estimate`, `mass_imbalance_tol` 1e-10,
`pressure_rtol` 1e-8), through the committed `StaggeredSolver` for the committed path and through
the arm subclasses for B and C, with the single right-edge outlet as the "return". The metric is
`validation.metrics.poiseuille_l2_error`, the harness's. The face hash is section 2.3's: SHA-256
over the bytes of `face_velocities.u` then `.v`, C-ordered little-endian float64.

## 5. Predictions

### 5.1 The orchestrator's (prompt 41, 2026-10-08; written before any run)

- (a) Arm A reproduces #61 within 2%: +0.088 Pa per outer iteration, `||b||` about 0.079, all of
  it beside the open return faces.
- (b) Arms B and C remove the drift: the mean pressure change per outer iteration below 1e-6 Pa,
  `||b||` at the end below 1e-6.
- (c) With the drift gone, CG at 1e-4 meets the error-estimate stop under B, C and D (it never
  does under A: `mass_imbalance_tol / ||b||` is about 2.5e-7 there). At 1e-2: no prediction.
- (d) Arm D has no drift and no reversed return face by construction, and `||b||` at rounding.
- (e) E0 drifts and E does not: the drift is the copy rule's, wherever it remains.
- (f) VAL-001 under B and C: the REQ-S02 metric equals the baseline's to three figures; the face
  hashes differ; the outer count within 10% of 1,559. The committed path shows no drift on
  VAL-001 (below 1e-6 Pa per outer iteration).
- (g) Ladder: no arm diverges at Re 895 within 3,000 outer iterations (T3 stayed bounded there);
  every arm diverges at Re 8,950 or ends growing, since the Reynolds number, not the outlets, is
  that rung's problem (step 5's question).

**What each outcome means.** If (b) holds, the drift is the copy rule's and step 3 can fix it
without changing the room's flow model. If only D removes it, the pressure-outlet rule needs more
than these candidates. If (c) holds, ADR-013 decision 3's tight default can be revisited. If (f)
fails on the metric, the candidate changes the converged answer, not only the path, and is wrong
for VAL-001.

**Stop and report.** Arm A not reproducing #61's drift within 10% at 1e-8 means the reference is
not what the evidence says, and the other arms cannot be read against it. An arm that cannot be
built through a subclass without a change under `src/` stops, with what would be needed, and the
rest run. A pressure solve that fails in a way the arm does not predict (CG at its cap on every
correction, or a refusal) is a finding for that arm.

### 5.2 The builder's (written before any script, beside the orchestrator's)

- (a) The hood's tangential condition is the one difference from the 8.3 and 12.4 rows (those kept
  zero gradient; this probe holds zero). Test 33b measured that change at 1e-6 relative on the
  residual by outer 25 at Re 8,950; at Re 90 and on 80x30 I expect the drift rate and `||b||` to
  move by less than 1%, inside the orchestrator's 2%. The run with the hood at zero gradient is
  kept as a control (A0) so any gap is attributed, not guessed.
- (b) B and C both remove the drift, and at 1e-8 they are nearly the same iteration: C's
  closing velocity is the corrected face of the previous iteration plus that cell's residual
  imbalance divided by rho dx, which at 1e-8 is 1e-8 of the flux, so the two histories agree to
  about 1e-8 relative and stop within a few outer iterations of each other. They separate at 1e-4
  and 1e-2, where C re-closes each outlet cell exactly before the prediction and B carries the
  correction's leftover. Outer counts under B and C at 1e-8 within 20% of A's 588: the drift is a
  property of the pressure level, which the velocity does not read, so removing it moves the
  velocity's convergence little.
- (c) Holds. Under B, C and D the right-hand side falls with the iteration instead of standing at
  0.079, so `pressure_rtol` times `||b||` falls below `mass_imbalance_tol` on its own. At 1e-2 the
  same argument applies and I expect the stop to be met under B, C and D as well, later than at
  1e-8, provided the loose correction does not blow up the start: ADR-013 B found that a start from
  rest at one part in ten blows up only at real air and ten times its viscosity, not at a thousand
  times, which is this case.
- (d) Holds. The closed-domain path's projection removes the mean of f, which is the rounding
  residue of the exact flux balance; `||b||` at the end is at the floor, below 1e-12 times the
  supply.
- (e) Holds. E0's drift rate is below A's because fewer open faces carry the copy's deficit: the
  largest return's share of A's standing b, not all of it.
- (f) The metric agrees to three figures and the hashes differ, as predicted. The committed path
  on VAL-001 shows a pressure change per outer iteration below 1e-6 Pa: every outlet face there
  carries outflow and the copy differs from the corrected face only by the correction's own
  move, which vanishes with the iteration error. Outer count within 10% of 1,559.
- (g) Re 895: no arm diverges within 3,000, and under B, C and D the residual at 3,000 is lower
  than A's (test 33b's T3 rose from outer 601 and left 3.5 m/s near 2,000, so A ends growing).
  Re 8,950: A, B, C, E and E0 end growing or diverge; D, which cannot reverse a return face, I
  also expect to end growing, because the growth sits in the gap over return 4 inside the room.

## 6. Onward

Sections 6 onward are written after the runs.
