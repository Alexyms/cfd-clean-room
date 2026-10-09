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
the same time. Sections 6 onward are written after the runs and say so. The fix pass of prompt
41b (2026-10-08) committed section 5.3 alone before its two arms ran (section 7.7), and corrected
sections 6 to 10 against the records as review 41 and test 41 found them.

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

### 5.3 The fix pass's two arms (prompt 41b, 2026-10-08; committed before any new run)

Review 41 B3 and S4 asked for two arms the first pass did not have. The ladder of section 7.3
separates the arms that fail (A, F) from those that converge (B, C, D, E, E0), but two mechanisms
travel together in the failing arms: the copy at every open return, and the hold-shut rule acting
in nearly every iteration. Arms D and E left the floor returns' tangential velocity at the pressure
outlet's zero gradient, while a fixed-flow outlet as REQ-S18 defines it holds it at zero.

- **A-open.** Arm A's copy at every open return with the hold-shut rule off: a face whose copied
  velocity points into the room stays open and is corrected like any other outlet face. The hood
  as in every arm. Both ladder rungs, to 3,000 outer iterations or divergence, recording reversed
  faces per segment per iteration (most at once and the number of iterations with any reversed).
  If A-open converges, the hold-shut rule is implicated; if it fails, the copy is.
- **D0.** Arm D with the floor returns' tangential velocity held at zero (Dirichlet), as the
  hood's already is. The drift case at 1e-8, 1e-4 and 1e-2 and both ladder rungs, with the same
  measurements as arm D and the face hash of every run; D0 against D as measurement 4 compares
  flows, over non-SOLID cells.

**The orchestrator's predictions.**

- (h) A-open: no prediction on convergence; that is the question. It records reversed faces in
  most iterations at both rungs.
- (i) D0 converges at both rungs and on the drift case at every level, with no drift (the
  mechanism needs a copied face, and D0 has none). Its counts differ from D's by under 10%; its
  flow differs from D's mostly in the cells beside the returns.

**The builder's (written with the orchestrator's, before any new run).**

- (h) A-open is prompt 33b's T2 (the hood fixed, the returns copied, nothing held shut) under
  the built hood condition and the CG solve. T2 was measured: at Re 895 it was bounded to the
  cut at 1,001 and, continued in test 33b, passed 5 m/s at outer 2,165; at Re 8,950 it diverged
  at outer 615. I expect A-open to repeat that, diverging at Re 8,950 within about 700 outer
  iterations and ending growing at Re 895, with faces reversed in most iterations once the growth
  starts. That reading implicates the copy, not the hold-shut rule, which under B, C, D, E and E0
  never acted.
- (i) As the orchestrator's. The returns' tangential condition changes the wall shear on the
  floor's u faces at 22 of 80 columns on 80x30 and 13 of 40 on 40x15, so I expect D0's counts
  within 5% of D's (177; 391; 1,632) and its flow within 0.05 m/s of D's, largest beside the
  returns.

## 6. Method as built (written after the runs)

Everything ran as section 4 planned it, with these additions and adaptations.

**The scripts.** `docs/reports/probe41/outlet41.py` holds the arms, the rooms, the runner and the
comparisons; `run41.sh` is every run in the order it was launched; `summary41.py` prints the
tables below from the records. Nothing under `src/`, `validation/`, `configs/` or `tests/`
changed. The scripts import nothing from `results/`, except the control, which imports
`frozen34.py` from `results/builder34/` to compare against.

**Adaptations of the earlier probes.**
- `outlet33b.py` set the sweep tolerance with `pressure_tol` and a 40,000-sweep cap; here the
  solver block gets `pressure_rtol` (1e-8, 1e-4 or 1e-2) and `max_pressure_iter` 5,000, the
  committed cap. `PressureCorrection` carries `iterations`, `reached_cap` and `products` instead of
  `sweeps`, and the callback's state carries `pressure_iterations`.
- `stall36.py` and `outer37.py` reached the drift room through `common36.py`'s `product_t3`, which
  is `frozen34.py`'s `FrozenSolver` (ten momentum sweeps on the zero-field path) over
  `outlet33b.py`'s T3. Here the ten sweeps are `SweepPredictor`, a subclass of the committed
  `MomentumPredictor` whose `predict` calls the committed `_assemble` and `frozen34.py`'s `_sweep_n`
  verbatim. The control (`outlet41.py control`) ran arm A for 50 outer iterations through each
  predictor: the residual histories are equal bitwise and the face hashes are equal
  (828112bc...), so the drift case here is the ECR-003 room to the bit at the start.
- The hood's tangential velocity is held at zero by `hold_hood_tangential`, test 33b's `tangD`:
  the ten right-edge tangential locations that are not already Dirichlet (the hood's nine faces
  have ten `v` storage locations along the edge) become Dirichlet zero before the predictors are
  built. Arm A was also run with 33b's zero gradient (A0) as a control; section 7.1 reads it.
- `outlet33b.py` applied its treatment after calling the committed extrapolation; `ArmSolver`
  replaces `_extrapolate_outlets` outright, so a segment's rule is the only thing that writes its
  faces, and checks at construction that the segments cover every configured outlet face.
- The corrector's open masks are set per iteration as in `outlet33b.py`. Arm D (and F, section
  8) also set `needs_pin` and `pin_cell` on the committed corrector, which then runs its own
  closed-domain projection and pin in `correct`; nothing else was needed. The corrector's
  `_check_components` ran once at construction on the configured (open) layout and was not run
  again on the closed one; the drift room is one component either way.
- The mean pressure is the mean of `state.p` over non-SOLID cells after each outer iteration; the
  "last 100" figure is the mean of its 100 last differences. Because that window spans the end of
  a geometric convergence, the tables also give the mean of the last 10, and section 7.2 runs
  every arm to 3,000 outer iterations with the stop disabled (`--long`: `iteration_error_tol`
  1e-14, `mass_imbalance_tol` 1e-16) to read the pressure's movement far past the stop.
- The velocity-step iteration is 1-based, as `outer37.py` recorded it (233 on the drift case).
- The floor returns' tangential velocity. Under every arm but D0 it stays the pressure outlet's
  zero gradient, as the committed boundary layer gives it: `hold_hood_tangential` switches the
  right edge only. So arms D and E hold the returns' normal velocity and leave their tangential
  velocity free, which is not the fixed-flow segment REQ-S18 defines (an outward normal velocity
  and zero tangential velocity). D0 (section 7.7) holds both, through the same wrapper on the
  bottom edge (24 locations on 80x30, 10 on 40x15): the fixed-flow segment as REQ-S18 defines it.
- Measurement 4 and the other comparisons take the difference, its location and the scale over
  non-SOLID cells only, and the scale is the first-named run's (review 41 B1: the pressure is 0
  in SOLID cells, so a difference there was the difference of the two means). The nine
  comparisons were rerun from the saved records in the fix pass; no solve was rerun.

**The runs.** 49 solves, the control's two, and 9 comparisons in the first round, and in the fix
pass 8 solves (D0 at three levels and to 3,000, D0 and A-open on both rungs) and 3 comparisons,
launched in parallel with one BLAS thread each (`run41.sh`), 2026-10-08, Python 3.13.3, NumPy
2.4.4, on the machine the baseline rows record (AMD Ryzen AI 9 HX 370). The first round's runs were
made twice: a first pass with the script before the velocity-step count was made 1-based and
before the 160x80 pair was added (kept under `results/builder41/pass1/`; arm F was already in
it), and the pass the tables report; section 7.6 compares them. A-open's two runs were made twice
as well: the runner's end-of-run code assumed the solver had kept its faces, which a diverged run
never reaches, so the first attempt wrote no record; the runner now hashes the last correction's
faces for a diverged run, and the records are the second attempt's. The wall times in the tables were taken under that parallel
load and are not used. No correction in any run reached the cap, no corrector
refused a layout, and no arm had to stop for want of a `src/` change. Arm E ran with B's rule at
the largest return (floor_return_1, seven faces), since B removed the drift on its own (section
7.1), as the arm's definition prescribes.

*Note, 2026-10-08 (ECR-002 step 3, prompt 42b).* `outlet41.py` reads
`configs/clean_room_default.yaml` (`product_raw`). That file changed in step 3: its
five outlets are now `fixed_flow_outlet`, so after step 3 the script builds a room with no
pressure outlet, and the arms that leave a return at the copy rule (A, A-open, E and E0 at
least) no longer measure what this report records. The script reproduces the records of this
report at commit b961f15 and not after step 3. The built room is checked against arm D0
by `docs/reports/probe42/fixed42.py`, which wants the shipped file.

## 7. Results (written after the runs)

### 7.1 Measurement 1: the drift case

| Arm | rtol | Stop | velocity_step at | Outer | Mean p change per outer, last 100 (Pa) | last 10 | `\|\|b\|\|` end | Share beside open faces | Worst cell at end (kg/s per m) | Signed sum | Reversed, most at once | Held shut, most at once | CG per correction, median [max] | Cap hits | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 1e-08 | error_estimate_and_continuity | 233 | 588 | +8.820e-02 | +8.8e-02 | 7.93e-02 | 1.000 | 5.0e-11 | -3.4e-16 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | 303 [303] | 0 | 44 |
| A | 0.0001 | cap, velocity step met | 233 | 20,000 | +8.820e-02 | +8.8e-02 | 7.93e-02 | 1.000 | 6.9e-07 | +1.2e-16 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | 229 [229] | 0 | 552 |
| A | 0.01 | cap, velocity step met | 234 | 20,000 | +8.819e-02 | +8.8e-02 | 7.93e-02 | 1.000 | 7.2e-05 | +3.3e-10 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | 152 [179] | 0 | 413 |
| B | 1e-08 | error_estimate_and_continuity | 115 | 206 | -4.929e-07 | -1.8e-08 | 1.86e-08 | 0.438 | 3.7e-14 | -2.5e-13 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | 298 [307] | 0 | 23 |
| B | 0.0001 | error_estimate_and_continuity | 115 | 206 | -4.929e-07 | -1.8e-08 | 1.86e-08 | 0.438 | 1.6e-13 | -1.0e-12 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | 223 [229] | 0 | 20 |
| B | 0.01 | error_estimate_and_continuity | 115 | 205 | -5.156e-07 | -1.9e-08 | 2.00e-08 | 0.444 | 1.7e-11 | +4.0e-11 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | 121 [179] | 0 | 14 |
| C | 1e-08 | error_estimate_and_continuity | 115 | 206 | -4.929e-07 | -1.8e-08 | 1.86e-08 | 0.438 | 3.7e-14 | -2.5e-13 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | 298 [307] | 0 | 7 |
| C | 0.0001 | error_estimate_and_continuity | 115 | 206 | -4.929e-07 | -1.8e-08 | 1.86e-08 | 0.438 | 1.6e-13 | -1.0e-12 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | 223 [229] | 0 | 21 |
| C | 0.01 | error_estimate_and_continuity | 115 | 206 | -4.817e-07 | -1.8e-08 | 1.85e-08 | 0.434 | 1.9e-11 | +3.9e-11 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | 121 [179] | 0 | 4 |
| D | 1e-08 | error_estimate_and_continuity | 103 | 177 | +8.271e-06 | +2.2e-08 | 1.88e-08 | 0.000 | 4.0e-14 | +2.0e-15 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | 313 [320] | 0 | 23 |
| D | 0.0001 | error_estimate_and_continuity | 103 | 177 | +8.270e-06 | +2.2e-08 | 1.88e-08 | 0.000 | 1.6e-13 | +2.0e-15 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | 238 [254] | 0 | 5 |
| D | 0.01 | error_estimate_and_continuity | 103 | 177 | +7.587e-06 | +1.8e-08 | 1.82e-08 | 0.000 | 2.8e-11 | +2.0e-15 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | 58 [195] | 0 | 9 |
| E | 1e-08 | error_estimate_and_continuity | 103 | 177 | +7.616e-06 | -1.1e-08 | 1.80e-08 | 0.030 | 3.3e-14 | +1.2e-13 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | 340 [345] | 0 | 23 |
| E0 | 1e-08 | error_estimate_and_continuity | 106 | 177 | +1.181e-01 | +1.2e-01 | 5.69e-02 | 1.000 | 5.7e-11 | -6.4e-16 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | 339 [340] | 0 | 24 |
| A (A0, hood tangential zero gradient) | 1e-08 | error_estimate_and_continuity | 233 | 588 | +8.820e-02 | +8.8e-02 | 7.93e-02 | 1.000 | 5.1e-11 | -6.0e-16 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | 303 [303] | 0 | 21 |
| F | 1e-08 | error_estimate_and_continuity | 512 | 969 | -1.991e-08 | -7.7e-09 | 5.50e-09 | 0.000 | 3.5e-14 | +2.0e-15 | [0, 0, 0, 0, 0] | [1, 0, 1, 0, 0] | 323 [330] | 0 | 39 |
| F | 0.0001 | error_estimate_and_continuity | 512 | 969 | -1.991e-08 | -7.7e-09 | 5.50e-09 | 0.000 | 5.5e-14 | +2.1e-15 | [0, 0, 0, 0, 0] | [1, 0, 1, 0, 0] | 266 [268] | 0 | 54 |
| F | 0.01 | error_estimate_and_continuity | 513 | 970 | -2.000e-08 | -7.8e-09 | 5.53e-09 | 0.000 | 6.8e-12 | +2.5e-15 | [0, 0, 0, 0, 0] | [1, 0, 3, 0, 0] | 172 [195] | 0 | 40 |

**Arm A reproduces #61.** +0.0882 Pa per outer iteration, `||b||` 0.0793 with 100.0% of its
square on the cells beside the open return faces, the error-estimate stop at 588 and the
velocity-step stop at 233, as sections 8.3 and 12.4 recorded (+0.0882, 0.0793, 588, 233). The
hood's tangential condition makes no difference the table can show: A0 (zero gradient, as the
ECR-003 rows had it) gives the same numbers; the two residual histories differ by at most 2.1e-3
relative (at outer 55) and their drift rates by 7e-6 relative. At 1e-4 and 1e-2 arm A never stops
in 20,000 outer iterations, as section 12.4 found under the committed outlets: the velocity-step
stop comes at 233 and 234 and the worst cell then holds 6.9e-7 (1e-4) and 7.2e-5 (1e-2) kg/s per metre
against the 2e-8 the rule asks, 35 and 3,600 times over, with the pressure climbing at the same
+0.0882 throughout.

**B and C remove the drift, and are the same iteration.** The pressure's change per outer
iteration is -4.9e-7 over the last 100 and -1.8e-8 over the last 10, falling; `||b||` at the
end is 1.9e-8, the prediction's imbalance while the velocity is still moving, with 0.44 of its
square beside the open faces (the corrected faces' worst cell is 3.7e-14); it reaches 3e-15 only
in the 3,000-iteration run of section 7.2. B and C agree to 3e-10 m/s at every cell (section 7.4), as
predicted: at 1e-8 the velocity that closes a cell is the corrected face of the previous
iteration to the correction's own residual. Both stop at 206 outer iterations, with the
velocity-step stop at 115, against A's 588 and 233: the standing imbalance was costing the
velocity's convergence as well as the pressure's. No return face ever pointed into the room
under B or C, so the hold-shut rule never acted.

**D has no drift and no reversed face; `||b||` is 1.9e-8 at the stop, with the velocity still moving.** 177 outer iterations, the fewest
of any arm, the velocity-step stop at 103; `||b||` 1.9e-8 at the stop with no cell beside an
open face (there is none); the worst cell 4.0e-14. The committed corrector solved the closed
system through the subclass alone: none of the 177 corrections reached the cap (313 CG
iterations median, 320 at most, against A's 303), and the signed domain sum of the corrected
faces was 2e-15. Which of the two stops ended each solve the records do not show. The stop is the
larger of `pressure_rtol` times `||b||` and the floor `RESIDUAL_FLOOR` times the flux scale,
3.8e-13 here; from correction 99 on `||b||` is below 3.8e-5, so 79 of the 177 corrections had
their relative level under the floor and stopped on the floor, as section 12.2 of the ECR-003
report describes for a converging closed domain. The projection removed only rounding: the
fixed split is formed from the face widths, so the supply, the hood and the returns balance to
the arithmetic.

**E does not drift and E0 does.** E, with B's rule at floor_return_1 and fixed flows at the other
three, matches D's counts (177, 103) and ends at `||b||` 1.8e-8 with 3% of it beside the one open
return; its flow differs from D's by at most 0.010 m/s, at the return-1 cells, where its seven
faces take a pressure outlet's distribution (1.215 to 1.242 m/s) in place of D's uniform 1.227
(section 7.4), and agrees with D's at the return-4 cells to the digits shown. E0, the same layout with A's copy
at that return, climbs at +0.118 Pa per outer iteration, faster than A's +0.088, with `||b||`
0.0569 entirely beside its seven open faces.

**At 1e-4 and 1e-2, B, C and D stop where 1e-8 does.** 206 (B; 205 at 1e-2), 206 (C) and 177 (D)
outer iterations at every level, the worst cell at the stop 1.6e-13 (1e-4) and 1.7e-11 to 2.8e-11 (1e-2), below the 2e-8
bound; the CG work per correction falls from 298 to 313 iterations at 1e-8 to 223 to 238 at 1e-4
and 58 to 121 at 1e-2. Under A no level below 1e-8 ever stops.

**The flow under A is not B's.** Section 7.4: A's cell-centred velocity differs from B's by 0.42
m/s in u at (5.65, 2.05), the cell beside the etch chamber's top, and by 0.51 m/s in v at
(6.05, 0.05), a return-4 cell; the larger is 28% of A's largest speed, 1.78 m/s, where B's is
1.29. A's return faces carry the interior copy plus the standing
correction `d p'`, which the next copy discards, so the faces the momentum equations see and the
faces continuity holds are not the same faces; the velocity A settles on solves neither problem.
The drift is not a cosmetic offset of the pressure.

### 7.2 Measurement 1, continued: past the stop

| Arm | p change per outer at 200 (mean of 10) | p change per outer at 588 (mean of 10) | p change per outer at 1,000 (mean of 10) | p change per outer at 2,000 (mean of 10) | p change per outer at 3,000 (mean of 10) | `\|\|b\|\|` at 3,000 | Residual at 3,000 | Worst cell at 3,000 |
|---|---|---|---|---|---|---|---|---|
| A | +8.8e-02 | +8.8e-02 | +8.8e-02 | +8.8e-02 | +8.8e-02 | 7.93e-02 | 2.56e-17 | 5.0e-11 |
| B | -2.5e-08 | -9.9e-16 | -5.7e-16 | -1.5e-16 | -4.9e-17 | 3.15e-15 | 1.41e-17 | 3.9e-16 |
| C | -2.5e-08 | +2.7e-16 | +3.9e-16 | +2.5e-16 | +1.5e-16 | 1.18e-14 | 2.82e-17 | 3.0e-15 |
| D | +2.2e-09 | -5.5e-15 | +1.6e-16 | -3.7e-17 | -6.2e-18 | 8.44e-16 | 1.41e-17 | 9.2e-17 |
| E | +6.6e-11 | -1.4e-15 | -1.6e-15 | -1.0e-15 | -7.9e-16 | 3.65e-14 | 1.95e-16 | 4.9e-15 |
| E0 | +1.2e-01 | +1.2e-01 | +1.2e-01 | +1.2e-01 | +1.2e-01 | 5.69e-02 | 3.30e-17 | 5.7e-11 |
| F | -7.4e-03 | -7.5e-06 | -4.4e-09 | -1.4e-15 | -2.5e-15 | 4.29e-14 | 4.43e-16 | 9.2e-15 |

With the stop disabled, B, C, D and E move the pressure by at most 5.5e-15 Pa per outer iteration
at 588 and at most 1.6e-15 from 1,000 on, with `||b||` between 8e-16 and 4e-14 and the worst cell at or below 5e-15 at 3,000.
A holds +0.0882 and E0 +0.118 for all 3,000 iterations with `||b||` unchanged, while their
velocity residuals fall to 3e-17: the velocity is steady and the pressure climbs without end, as
#61 says.

### 7.3 Measurement 2: the ladder

| Re | Arm | Outer | End | Residual: least, at end | Largest speed at end (m/s), cell | Reversed faces, most at once (returns 1 to 4, hood) | Held shut, most at once | Mean p change per outer, last 100 (Pa) | `\|\|b\|\|` end | s |
|---|---|---|---|---|---|---|---|---|---|---|
| 895 | A | 3,000 | cap, growing | 2.36e-03, 2.18e-01 | 16.9, (6.3, 1.5) | [3, 3, 2, 1, 0] | [4, 3, 2, 2, 0] | -1.03e+00 | 4.75e+00 | 65 |
| 895 | B | 435 | velocity_step_below_tol | 9.90e-07, 9.90e-07 | 1.26, (2.7, 0.1) | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | -4.95e-09 | 5.47e-06 | 10 |
| 895 | C | 435 | velocity_step_below_tol | 9.90e-07, 9.90e-07 | 1.26, (2.7, 0.1) | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | -4.95e-09 | 5.47e-06 | 21 |
| 895 | D | 391 | velocity_step_below_tol | 9.89e-07, 9.89e-07 | 1.41, (6.1, 0.9) | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | -1.88e-07 | 9.32e-06 | 19 |
| 895 | E | 402 | velocity_step_below_tol | 9.51e-07, 9.51e-07 | 1.41, (6.1, 0.9) | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | +2.58e-07 | 9.82e-06 | 20 |
| 895 | E0 | 400 | velocity_step_below_tol | 8.71e-07, 8.71e-07 | 1.41, (6.1, 0.9) | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | +4.96e-02 | 6.19e-02 | 20 |
| 895 | F | 3,000 | cap, falling | 7.79e-03, 9.47e-02 | 6.14, (6.1, 0.9) | [0, 0, 0, 0, 0] | [1, 2, 1, 1, 0] | +5.99e-03 | 2.23e+00 | 52 |
| 8,950 | A | 3,000 | cap, growing | 5.03e-03, 2.71e-01 | 17.2, (6.3, 1.7) | [4, 3, 2, 2, 0] | [4, 3, 2, 3, 0] | -1.44e+00 | 5.12e+00 | 67 |
| 8,950 | B | 2,510 | velocity_step_below_tol | 9.92e-07, 9.92e-07 | 1.29, (4.9, 1.1) | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | -4.01e-09 | 4.49e-06 | 42 |
| 8,950 | C | 2,510 | velocity_step_below_tol | 9.92e-07, 9.92e-07 | 1.29, (4.9, 1.1) | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | -4.01e-09 | 4.49e-06 | 42 |
| 8,950 | D | 1,632 | velocity_step_below_tol | 9.85e-07, 9.85e-07 | 1.41, (6.1, 0.9) | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | +2.74e-10 | 4.74e-06 | 28 |
| 8,950 | E | 1,632 | velocity_step_below_tol | 9.91e-07, 9.91e-07 | 1.41, (6.1, 0.9) | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | +2.77e-11 | 4.75e-06 | 45 |
| 8,950 | E0 | 1,632 | velocity_step_below_tol | 9.85e-07, 9.85e-07 | 1.41, (6.1, 0.9) | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | +4.59e-02 | 5.49e-02 | 30 |
| 8,950 | F | 3,000 | cap, growing | 8.35e-03, 1.50e-01 | 10, (6.3, 1.3) | [0, 0, 0, 0, 0] | [4, 2, 1, 2, 0] | +1.07e-02 | 3.83e+00 | 66 |

**A and F fail at both rungs; B, C, D, E and E0 converge at both.** Under A the
residual's least value is 2.4e-3 (Re 895) and 5.0e-3 (Re 8,950), the end state 0.22 and 0.27 with
the largest speed 17 m/s over return 4 and the hood bench, `||b||` 4.7 and 5.1 of a supply of 3.9
kg/s per metre, and faces held shut in 2,830 and 2,843 of the 3,000 iterations, from outer 165
and 158 on, up to four at once on return 1. That is test 33b's picture of T3 at these rungs (10 to
20 m/s, residuals 0.1 to 0.35, held and reversed faces throughout) reproduced under the built
hood condition, and without the cap: no correction here stopped short.

Under B and C the room converges by the velocity-step rule at 435 outer iterations at Re 895 and
2,510 at Re 8,950, to a largest speed of 1.26 and 1.29 m/s, with no return face ever pointing into
the room and none ever held shut, so the two arms ran identical iterations (same counts, same
residuals to the digits shown). Under D, E and E0 it converges at 391 to 402 (Re 895) and 1,632
(Re 8,950); E0 drifts at +0.05 Pa per outer iteration while converging in velocity. On 40x15
returns 1 and 4 have four faces each (0.8 m); E's pressure return is return 1 by the last bit of
the widths' rounding (0.8000000000000002 against 0.7999999999999998), so its rows read "E with
return 1 at pressure" by a tie the drift case does not have (seven faces against six). At Re 8,950
under B every return face carries outflow at the end, 0.70 to 1.31 m/s, and the four returns take
0.85, 0.72, 0.43 and 0.74 m^2/s with the hood's 0.50, the supply's 3.24 on this grid to the
digits shown.

Prompt 33b's ladder (the Reynolds report, section 8.4) found no treatment that converged at
Re 895 or above, and test 33b continued T3 to 2,500 and 1,000 outer iterations without
convergence; ADR-012 D reads from that that "nothing tried makes the room converge above a
hundred times the viscosity". What this table separates is narrower than "the copy". Two
mechanisms travel together in the arms that fail: A copies the interior at every open return and
F scales that copy, and both hold return faces shut in nearly every iteration (2,830 and 2,843 of
3,000 under A, 2,963 and 2,979 under F), while no converging arm ever held a face shut. E0 keeps
the copy at one return and converges; F is not a plain copy and fails. Either mechanism, or both,
could be what stops convergence. The arm that separates them is A-open, the copy at every open
return with the hold-shut rule off (section 7.7): it diverges at both rungs, at outer 843 and 370,
with reversed faces in most iterations and none held shut. The copy at every open return fails
on this grid with or without the hold-shut rule; the hold-shut rule's own effect on a converging
iteration is unmeasured here, since it never acted in any arm that converged. Prediction (g) was
wrong on its second half: at one momentum sweep on 40x15 the laminar room converges at both rungs
under B, C, D, E and E0. What holds on 80x30 and 200x75 at real air, where ADR-013 decision 6
found the laminar room converging on neither with ten sweeps, is step 5's measurement, now with a
different outlet rule to make it under.

### 7.4 Measurement 4: the flows against each other

| Pair | max abs du (m/s), cell | max abs dv (m/s), cell | max abs dp after removing each mean (Pa), cell | Floor faces, max abs dv (m/s) | Scale: max abs u, v, p of the first-named run, fluid cells |
|---|---|---|---|---|---|
| D vs B | 6.431e-02, (6.15, 0.05) | 8.419e-02, (6.25, 0.05) | 9.584e-02, (2.75, 0.05) | 1.261e-01 | 0.906, 1.277, 1.082 |
| C vs B | 2.833e-10, (5.85, 0.05) | 3.489e-10, (5.95, 0.05) | 5.702e-10, (5.95, 0.05) | 5.215e-10 | 0.914, 1.288, 1.029 |
| E vs B | 6.431e-02, (6.15, 0.05) | 8.419e-02, (6.25, 0.05) | 9.584e-02, (2.75, 0.05) | 1.261e-01 | 0.906, 1.277, 1.094 |
| E vs D | 9.232e-03, (0.75, 0.05) | 1.015e-02, (0.55, 0.05) | 1.182e-02, (0.55, 0.05) | 1.474e-02 | 0.906, 1.277, 1.094 |
| A vs B | 4.150e-01, (5.65, 2.05) | 5.067e-01, (6.05, 0.05) | 1.175e+00, (6.35, 0.85) | 5.694e-01 | 1.197, 1.783, 2.128 |
| F vs B | 3.905e+00, (5.75, 2.05) | 4.536e+00, (6.05, 0.05) | 2.438e+01, (6.35, 0.85) | 4.692e+00 | 4.626, 5.813, 25.338 |
| F vs A | 3.516e+00, (5.75, 2.05) | 4.029e+00, (6.05, 0.05) | 2.321e+01, (6.35, 0.85) | 4.123e+00 | 4.626, 5.813, 25.338 |

On the drift case D's flow differs from B's by 0.064 m/s in u and 0.084 m/s in v (7% and 7% of
D's largest u and v), both at the return-4 cells, and by 0.126 m/s on the floor faces: fixing the
returns' flows at the equal-velocity split changes the room where the returns sit and little
elsewhere. The demeaned pressure differs by 0.096 Pa at (2.75, 0.05), a return-2 cell, of a
field whose largest value is 1.08 Pa. (The first round's table took the pressure difference over
every cell, SOLID ones included, where p is 0 in both runs and the difference is the difference
of the two means; four rows landed inside the server rack and the prose read one as return 1.
Review 41 B1 found it; the table and this paragraph are the fluid-only values.) E's flow is D's to
0.010 m/s. On the 40x15 ladder the gap is larger: 0.290 m/s in u at (4.9, 2.1), the cell that
touches the etch chamber's top corner at (5.0, 2.0), and 0.238 m/s in v over return 4 at Re
8,950, 32% and 18% of B's largest components; 0.275 and 0.222 m/s at Re 895.

### 7.5 Measurement 3: VAL-001

| Path | Outer | Stop | REQ-S02 metric | Equal to the baseline's 4.1077e-4 to three figures | Face hash (u then v) | Equals the baseline's | Mean p change per outer, last 100 (Pa) | last 10 | `\|\|b\|\|` end | Worst cell at end | Reversed faces, most | Held shut, most |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| committed | 1,559 (+0.0%) | error_estimate_and_continuity | 4.10770e-04 | True | c7d88d919e1141a5... | True | -2.38e-10 | -1.5e-10 | 7.38e-13 | 3.8e-16 | [0] | [0] |
| B | 1,533 (-1.7%) | error_estimate_and_continuity | 4.49223e-04 | False | 968f06183cb7edfb... | False | -3.01e-10 | -1.9e-10 | 2.36e-12 | 4.3e-16 | [0] | [0] |
| C | 1,533 (-1.7%) | error_estimate_and_continuity | 4.49223e-04 | False | 4d42e2b1e3dd7127... | False | -3.01e-10 | -1.9e-10 | 2.36e-12 | 4.3e-16 | [0] | [0] |
| F | 1,559 (+0.0%) | error_estimate_and_continuity | 4.10770e-04 | True | 9b4f42a94dcefdbc... | False | +7.27e-11 | +4.5e-11 | 7.15e-13 | 4.4e-16 | [0] | [0] |

| Grid | Path | Outer | REQ-S02 metric | Metric minus the committed path's | max abs du at cell centres vs committed (m/s), column | max abs v in the last column (m/s) | max abs (outlet face minus interior face) (m/s) |
|---|---|---|---|---|---|---|---|
| 40x20 | committed | 408 | 1.99904e-03 | +0.00e+00 | 0.00e+00, column 0 of 40 | 8.59e-09 | 3.36e-09 |
| 40x20 | B | 399 | 2.04699e-03 | +4.79e-05 | 3.42e-02, column 39 of 40 | 2.96e-02 | 1.72e-02 |
| 40x20 | F | 408 | 1.99904e-03 | +5.97e-12 | 2.01e-10, column 39 of 40 | 8.50e-09 | 3.31e-09 |
| 80x40 | committed | 1,559 | 4.10770e-04 | +0.00e+00 | 0.00e+00, column 0 of 80 | 4.38e-09 | 8.72e-10 |
| 80x40 | B | 1,533 | 4.49223e-04 | +3.85e-05 | 4.71e-02, column 79 of 80 | 3.68e-02 | 1.66e-02 |
| 80x40 | F | 1,559 | 4.10770e-04 | +7.06e-13 | 1.31e-11, column 79 of 80 | 4.38e-09 | 8.71e-10 |
| 160x80 | committed | 6,103 | 1.56823e-04 | +0.00e+00 | 0.00e+00, column 0 of 160 | 2.21e-09 | 2.22e-10 |
| 160x80 | B | 6,019 | 1.53702e-04 | -3.12e-06 | 5.81e-02, column 159 of 160 | 4.30e-02 | 1.61e-02 |
| 160x80 | F | 6,103 | 1.56823e-04 | -9.02e-14 | 5.35e-12, column 159 of 160 | 2.22e-09 | 2.22e-10 |

**The committed path reproduces the baseline.** 1,559 outer iterations, the metric 4.10770e-4 and
the face hash c7d88d91... equal to the `staggered-cg` row's (ae24b120, 63d655ba), and no drift:
-2.4e-10 Pa per outer iteration over the last 100, -1.5e-10 over the last 10, falling, with
`||b||` 7.4e-13 at the stop. VAL-001's single outlet carries outflow at every face in every
iteration, so A's copy and the corrected face agree there to the correction's own move.

**B and C change VAL-001's converged answer.** Both stop at 1,533 (-1.7%), inside the 10%
predicted, but the metric is 4.49223e-4 against 4.10770e-4: equal to one figure, not three, 9%
larger. The difference sits in the exit column. Under the committed path the transverse velocity
in the last column is 4e-9 m/s and each outlet face equals its interior neighbour to 9e-10; under
B the last column carries v up to 0.037 m/s (0.37 of the mean velocity) and the outlet faces differ
from their neighbours by up to 0.017 m/s, with the cell-centred u differing from the committed
path's by 0.047 m/s in that column and by 1.5e-5 m/s at mid-channel. The same structure appears on
40x20 (v 0.030 m/s) and 160x80 (v 0.043 m/s) and does not fall with the grid; the metric's
difference is +4.8e-5, +3.8e-5 and -3.1e-6 on the three grids (the 160x80 pair stops at 6,103 and
6,019 outer iterations), so at mid-channel the two answers approach each other under refinement
while the exit column does not. The committed copy gives each outlet face the zero-gradient value, the fully developed outflow condition, and continuity then
forces v_n = v_s in each exit cell; B and C give the face only the value that closes its cell,
which is also what a column of nonzero v with a compensating outlet face satisfies. The discrete
problem B and C solve has a weaker outlet condition and a different solution. The metric is
still 40 times inside REQ-S02's 1e-2, so the candidate is not inaccurate; it is a different
discretisation of the exit, and the prediction's reading stands: the candidate changes the
converged answer, not only the path, and is wrong for VAL-001 as VAL-001's bitwise clause reads.

### 7.6 Pass 1 against pass 2

Every run of the tables was also made in the first pass, before the velocity-step count was
made 1-based and before the 160x80 pair was added (`results/builder41/pass1/`). The 46 solves
the two passes share, and the control's two records, have the same face hash and the same outer
count in both; the three 160x80 runs exist in the second pass only. The control (`control.json`) holds in both passes:
the residual histories of the two predictors are equal bitwise over 50 outer iterations and
their face hashes are 828112bc...

### 7.7 The fix pass: A-open and D0 (written after the fix pass's runs)

The two arms of section 5.3, run after that section was committed (7973068). Every row carries its
face hash, so step 3 can compare against it on this machine.

| Run | Outer | End | velocity_step at | Residual: least, at end | Largest speed at end (m/s), cell | Mean p change per outer, last 100 (Pa) | last 10 | `\|\|b\|\|` end | Worst cell at end | Reversed faces, most at once | Iterations with any reversed | Held shut, most | Iterations with any held shut | CG per correction, median [max] | Face hash (u then v) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| drift_D0_1e-8 | 177 | error_estimate_and_continuity | 103 | 1.33e-09, 1.33e-09 | 1.28, (6.05, 0.15) | +6.65e-06 | +2.2e-08 | 1.81e-08 | 4.4e-14 | [0, 0, 0, 0, 0] | 0 of 177 | [0, 0, 0, 0, 0] | 0 of 177 | 313 [320] | b0cf5110f07052d7b8dfcc3d4f90a7698907bffc1e6105d9a08237e9d8170bd4 |
| drift_D0_1e-4 | 177 | error_estimate_and_continuity | 103 | 1.33e-09, 1.33e-09 | 1.28, (6.05, 0.15) | +6.64e-06 | +2.2e-08 | 1.81e-08 | 1.6e-13 | [0, 0, 0, 0, 0] | 0 of 177 | [0, 0, 0, 0, 0] | 0 of 177 | 237 [253] | b24691cc97f4346833da3c53cced58a7a8e677d3e4a20b850667251de09057a0 |
| drift_D0_1e-2 | 177 | error_estimate_and_continuity | 103 | 1.30e-09, 1.30e-09 | 1.28, (6.05, 0.15) | +6.21e-06 | +1.9e-08 | 1.74e-08 | 2.2e-11 | [0, 0, 0, 0, 0] | 0 of 177 | [0, 0, 0, 0, 0] | 0 of 177 | 61 [195] | ed47f42bde0402886aa899508ca11c395ed76c55f787846036ff64273ce1fb27 |
| drift_D0_1e-8_long | 3,000 | cap, velocity step met | 103 | 7.05e-18, 1.41e-17 | 1.28, (6.05, 0.15) | -8.33e-18 | -5.6e-18 | 9.07e-16 | 1.1e-16 | [0, 0, 0, 0, 0] | 0 of 3,000 | [0, 0, 0, 0, 0] | 0 of 3,000 | 1 [320] | c2092a91062b5471c6796359ace8504e5a208fc7990768ae5f672a01beb0d737 |
| ladder_Aopen_895 | 843 | diverged | None | 2.36e-03, 8.00e-01 | 104, (0.7, 0.1) | +1.87e+01 | +1.1e+02 | 1.04e+04 | 3.4e-05 | [4, 3, 0, 3, 0] | 668 of 843 | [0, 0, 0, 0, 0] | 0 of 843 | 166 [184] | 33d65ae44271d8d6accbb5dc3fc9b5c07ddb142b1f8c4d43b9a1e05fdb8f6d5e |
| ladder_Aopen_8950 | 370 | diverged | None | 5.03e-03, 7.39e-01 | 100, (2.5, 0.3) | +1.75e+01 | +9.4e+01 | 3.42e+03 | 1.8e-05 | [3, 3, 0, 1, 0] | 207 of 370 | [0, 0, 0, 0, 0] | 0 of 370 | 166 [188] | ce97d37a0b80461ed42d65efc818847caa053ec7389ba2d91b718de713e685d7 |
| ladder_D0_895 | 391 | velocity_step_below_tol | 391 | 9.84e-07, 9.84e-07 | 1.41, (6.1, 0.9) | -1.90e-07 | +2.0e-07 | 9.24e-06 | 6.0e-14 | [0, 0, 0, 0, 0] | 0 of 391 | [0, 0, 0, 0, 0] | 0 of 391 | 160 [162] | d971ecc51572b17871cd51102c8a88557dd20dc6fe32c44dd62569ddb0097540 |
| ladder_D0_8950 | 1,632 | velocity_step_below_tol | 1632 | 9.82e-07, 9.82e-07 | 1.41, (6.1, 0.9) | +2.73e-10 | +2.9e-09 | 4.73e-06 | 9.3e-14 | [0, 0, 0, 0, 0] | 0 of 1,632 | [0, 0, 0, 0, 0] | 0 of 1,632 | 165 [171] | 94263bb1d9f0415cbba93ad723f01fa73c385539b4a49bb239d86e367836233b |

| Pair | max abs du (m/s), cell | max abs dv (m/s), cell | max abs dp after removing each mean (Pa), cell | Floor faces, max abs dv (m/s) | Scale: max abs u, v, p of the first-named run, fluid cells |
|---|---|---|---|---|---|
| drift_D0_1e-8 vs drift_D_1e-8 | 9.695e-02, (0.55, 0.05) | 4.196e-02, (0.45, 0.15) | 6.021e-02, (0.65, 0.05) | 0.000e+00 | 0.906, 1.277, 1.129 |
| ladder_D0_895 vs ladder_D_895 | 3.614e-03, (0.9, 0.1) | 1.789e-03, (1.1, 0.3) | 1.859e-03, (0.5, 0.1) | 0.000e+00 | 1.013, 1.396, 1.254 |
| ladder_D0_8950 vs ladder_D_8950 | 5.147e-04, (0.9, 0.1) | 2.039e-04, (1.1, 0.3) | 2.338e-04, (0.5, 0.1) | 0.000e+00 | 1.021, 1.404, 1.250 |

**A-open diverges at both rungs.** The copy at every open return with nothing held shut passes
100 m/s at outer 843 (Re 895) and 370 (Re 8,950). The first reversed face appears at outer 176 and
164, about where arm A first held one shut (165 and 158), and from then on some corrected return
face points into the room in 668 of the 843 and 207 of the 370 iterations, up to four at once on
return 1; no face is ever held shut, by construction. The largest speed is 2.3 m/s at outer 100 at
both rungs, 4.8 and 7.9 m/s at 200, 8.9 and 12.2 at 300, and the run ends with the pressure
climbing by 17 to 19 Pa per outer iteration. That is prompt 33b's T2 (the hood fixed, the returns
copied, nothing held shut), which diverged at outer 615 at Re 8,950 and passed 5 m/s at 2,165 at Re
895 under the Jacobi solve; under CG at 1e-8 and the built hood condition it goes sooner.

**D0 converges everywhere D does, with D's counts, and does not drift.** 177 outer iterations on
the drift case at 1e-8, 1e-4 and 1e-2 (velocity-step at 103), 391 at Re 895 and 1,632 at Re 8,950:
the same counts as D to the iteration, with no face reversed or held shut, no correction at the
cap, and the pressure moving by 8e-18 Pa per outer iteration at 3,000. Holding the returns'
tangential velocity at zero moves the flow most beside the returns: against D, u differs by at
most 0.097 m/s, at (0.55, 0.05), and v by 0.042 m/s, at (0.45, 0.15), the return-1 cells, on the
drift case (11% and 3% of D0's largest components), and by 0.0036 and 0.0018 m/s on the ladder at
Re 895 and 0.0005 and 0.0002 at Re 8,950, also at return 1; the floor faces are identical, since
both arms fix them. The face hashes differ from D's in every run.

## 8. Arm F, found on the way (written after the runs)

Section 7.5's finding says what B and C lack: an outlet condition on the face's momentum. The
textbook outflow treatment (Patankar 1980, the outflow boundary; Versteeg and Malalasekera 2007,
chapter 9, outlet boundary conditions)
copies the interior as A does, scales the copies by one factor so the total outflow equals the
inflow, and then holds the outlet faces as known in the pressure correction, which leaves the
pressure to a pin. It keeps the zero-gradient condition and cannot drift, since the faces are not
corrected. It was run as arm F through the same subclass: each pressure return copies its interior
neighbour, inward faces are held shut as under A, the open copies are scaled so that the returns
and the hood together carry the supply (the equal-velocity split is used instead whenever the
copies carry no outflow, which happened once per run, at rest), and all return faces leave the
masks, so the corrector pins. F ran the drift case at the three levels and to 3,000 iterations,
both rungs of the ladder, and VAL-001 on 40x20, 80x40 and 160x80.

- **VAL-001.** F reproduces the committed path: 1,559 outer iterations, the metric 4.10770e-4 to
  the twelfth figure (+7e-13), the last column's v 4.4e-9 and the outlet faces equal to their
  neighbours to 9e-10, no drift (+4.5e-11 Pa per outer iteration over the last 10). The face hash
  differs: the pressure level is the pin's, not the outlet's, and the path differs by the scaling,
  which ends at 1 - 3.7e-14 on 80x40 (1 - 3e-15 on 40x20, 1 + 5e-14 on 160x80).
- **The drift case.** No drift (-2.0e-8 over the last 100 at the stop, 1e-15 at 3,000), `||b||`
  5.5e-9 at the stop, the same counts at 1e-8, 1e-4 and 1e-2 (969 to 970 outer iterations). But
  the flow is wrong: the whole supply leaves through return 4 at 2.5 to 6.0 m/s while returns 1 to
  3 carry 0.002 m/s or less, with one face of return 3 held shut from outer 175 and one of return
  1 from outer 325 on, and
  the largest speed 5.8 m/s against B's 1.29. One scale factor applied to every return cannot set
  the split between returns; whatever split the copies happen to carry is what the scaling
  preserves, and the iteration found a split with three returns dead. A single-outlet case cannot
  show this.
- **The ladder.** F does not converge at either rung (cap at 3,000, residual 0.095 and 0.15 at the
  end, `||b||` 2.2 and 3.8, up to four faces held shut at once). The hold-shut rule under a
  scaled copy flaps as it does under A.

Measured, then: on VAL-001 F reproduces the committed path to the twelfth figure, and on the room
with four returns it settles on a split with three returns dead. Whether a scaled copy with a
configured split per return would differ from D at all is a question this probe did not run.

## 9. Predictions against measurement (written after the runs)

| Prediction | Outcome | Measured |
|---|---|---|
| (a) A reproduces #61 within 2%: +0.088 Pa, `\|\|b\|\|` about 0.079, all beside the open returns | Holds | +0.0882 Pa per outer iteration, `\|\|b\|\|` 0.0793, share 1.000; stops at 588 and 233 as recorded |
| (b) B and C remove the drift: below 1e-6 Pa per outer, `\|\|b\|\|` below 1e-6 | Holds | -4.9e-7 (last 100), -1.8e-8 (last 10), 1e-15 at 3,000; `\|\|b\|\|` 1.9e-8 at the stop, 3e-15 at 3,000 |
| (c) CG at 1e-4 meets the error-estimate stop under B, C and D, never under A | Holds | B, C, D stop at 206, 206, 177 at 1e-4 and 205, 206, 177 at 1e-2; A never in 20,000 at either |
| (d) D has no drift, no reversed face, `\|\|b\|\|` at rounding | Holds, on the 3,000-iteration run | 1e-17 Pa per outer at 3,000; at the stop the last-100 figure is +8.3e-6, falling (+2.2e-8 over the last 10), above the 1e-6 bar (b) uses; no face reversed or shut; `\|\|b\|\|` 8e-16 at 3,000 (1.9e-8 at the stop, where the velocity was still moving) |
| (e) E0 drifts and E does not | Holds, on the 3,000-iteration run | E0 +0.118 Pa per outer throughout; E 1e-15 at 3,000, and at the stop +7.6e-6 over the last 100, falling (-1.1e-8 over the last 10) |
| (f) VAL-001 under B and C: metric equal to three figures, hashes differ, count within 10%, committed path no drift | Fails on the metric | 4.492e-4 against 4.108e-4 (one figure); hashes differ; 1,533 (-1.7%); committed drift -2e-10 |
| (g) Ladder: no arm diverges at Re 895; every arm diverges or ends growing at Re 8,950 | Fails on the second half | Nothing diverged in the first round. B, C, D, E, E0 converge at both rungs (391 to 435 and 1,632 to 2,510); A ends at the cap at both, growing, at 17 m/s; F ends at the cap at both |
| (h) A-open: no prediction on convergence; reversed faces in most iterations at both rungs | The second half holds | Diverges at outer 843 (Re 895) and 370 (Re 8,950); a reversed face in 668 of 843 and 207 of 370 iterations, from outer 176 and 164 |
| (i) D0 converges at both rungs and on the drift case at every level, no drift, counts within 10% of D's, flow differing mostly beside the returns | Holds | 177, 177, 177, 391, 1,632: D's counts exactly; 8e-18 Pa per outer at 3,000; the largest difference from D is 0.097 m/s, at the return-1 cells |

The builder's (section 5.2): (a) held, and A0 showed the tangential condition moved nothing the
table can see; (b) held, with B and C agreeing to 3e-10 m/s, but the counts were not "within 20%
of A's 588": they were 206, 65% fewer; (c) held at 1e-2 as well; (d) held; (e) wrong in its
detail, E0's rate is above A's, not below; (f) wrong on the metric, for the reason section 7.5
gives; (g) wrong, as the orchestrator's. The builder's of section 5.3: (h) held in kind and was wrong in
detail: A-open diverged at both rungs (843 and 370), where the builder expected a run ending
growing at Re 895; T2, its Jacobi-era counterpart, passed 5 m/s at 2,165 at Re 895 without
reaching the divergence threshold and diverged at 615 at Re 8,950; (i) held
on the counts, which are D's exactly, and was wrong on the amount, 0.097 m/s against the 0.05
predicted.

**Stop-and-report items.** None fired: A reproduced #61 to the digits recorded; every arm was built
through the subclass; no pressure solve failed or was refused in any run.

## 10. What this implies (written after the runs; questions, not decisions)

**Which arm removes the drift, and at what cost.** Every rule but the copy removes it: B and C
(the corrected face kept; the cell closed before prediction), D (fixed flows), E (fixed flows at
three returns, B at one) and F (the scaled copy held). The costs measured:

- B and C: the same iteration to 3e-10 m/s. On the room they converge in a third of A's outer
  iterations, with no face ever reversing, and at Re 895 and Re 8,950 on 40x15 where A does not
  converge at all. Their cost is the outlet condition they leave out: on VAL-001 the exit column
  carries a transverse velocity of 0.037 m/s and the metric moves from 4.108e-4 to 4.492e-4 on
  80x40, with the exit column's structure on every grid tried. Whether that same freedom changes
  the room's answer at the returns in a way that matters is not something VAL-001 can say; the return faces under B vary across one return
  (1.20 to 1.30 m/s at return 2), as a pressure outlet's would, and no reference exists for the
  room. The question for step 3: is a face value fixed by cell continuity alone an acceptable
  pressure outlet, or does the rebuilt outlet need a momentum condition on the face as well (a
  half-cell momentum equation at the outlet face with the outlet pressure in its source, which
  neither the probes nor `src/` has)?
- D: no outlet condition is needed because no face is free. The cost is the split: the returns'
  flows become configured inputs, and the room's flow differs from B's by 0.08 m/s at the return
  cells on 80x30 and 0.29 m/s at the cell touching the etch chamber's top corner on 40x15. The
  ECR-002 design (ADR-012 decision 1 as taken) treats the split as something the room sets; D
  takes it from the user. D0, the same with the returns' tangential velocity held at zero,
  converges with D's counts and differs from D by 0.1 m/s beside the returns (section 7.7).
- E: D's answer with one return left to the pressure, which then carries exactly the share the
  split would have given it. It changes nothing D does not, and removes the need for one input.
- F: measured, it reproduces VAL-001's single outlet to the twelfth figure and settles on a split
  with three of the room's four returns dead. Whether a scaled copy with a configured split per
  return differs from D at all was not run.

**Whether ADR-012 decision 1's rejection of option 2 still holds.** Option 2 was set aside as
"compatible only when the shares sum to the supply exactly, the closed cavity's case". Measured:
the committed corrector solves that case as it is, through the subclass alone, in 177 outer
iterations with the signed domain sum at 2e-15 and no correction at its cap (the records keep
`||b||` per correction but not which criterion stopped each solve; section 7.1); the compatibility is exact when the shares are formed from the face widths, as D forms
them, and the projection the ECR-003 solve already runs removes the rounding residue. What the
rejection said about the corrector is no longer the case. What it said about the input remains:
the split must sum to the supply less the hood, so a configured split is a derived quantity (a
share per return, the velocities computed) or a validated one (refused when it does not sum).
The questions: does Alex want the returns' split to be an input of the room's configuration at
all, given that it moves the room's flow by the amounts above; and if the hood is a fixed flow
(decision 2 of 2026-10-04) and the returns are pressure outlets, which of B, C or a face momentum
condition is the pressure outlet the rebuild adopts?

**What REQ-S16 and REQ-S18's "VAL-001 bitwise" would have to say.** No deliberate outlet fix
keeps VAL-001's faces bitwise. B and C change the metric (4.108e-4 to 4.492e-4, both inside
REQ-S02's 1e-2) and the exit column's structure; F keeps the metric to 7e-13 and changes the
hash through the pin and the path; D does not apply to a single outlet without becoming F. If
the pressure outlet is rebuilt, the clause would read as ECR-003's did for its change: VAL-001 and
VAL-002 within REQ-S02 and REQ-S03, and a baseline retaken at the step's base, with the hash
machine-specific as section 2.3 of the baseline report records. Under B or C the retaken VAL-001
baseline would carry the exit-column structure section 7.5 describes; under F or a face momentum
condition it would not. Which of those is acceptable as "VAL-001 passes" is the question the
requirement's wording has to answer before step 3 builds anything; today it answers "bitwise",
which none of them can meet. VAL-002 was not run; it has no outlet, and the arms write only outlet
faces, so it holds by construction under every arm.

**Whether ADR-013 decision 3's `pressure_rtol` default could be relaxed.** Issue #61's reason
for a tight default was this room: under the copy, only a correction below about 2.5e-7 of the
standing imbalance lets the error-estimate rule stop. With the drift gone that reason is gone:
under B, C and D the rule stops at the same outer iteration at 1e-8, 1e-4 and 1e-2, and the work
per correction falls by a quarter at 1e-4 and by half to four fifths at 1e-2. The other reasons
ADR-013 B gives were not measured here: the start from rest at real air (one part in ten blew up
within three outer iterations on 40x15 and 80x30 at real air) and reproducibility (two CG
implementations at one part in ten reached states 9.9e-5 m/s apart). The question is therefore
whether, after step 3 adopts a rule without the standing imbalance, the guard and tight-start
options of decision 3 are revisited against those two remaining reasons, measured on the rooms
step 5 runs, rather than against this one, which no longer needs the tight level under any rule
but the copy.

**Found on the way, for step 5.** On 40x15 at one momentum sweep the laminar room converges at
Re 895 and Re 8,950 under B, C, D, E and E0, and fails under A, F and A-open. The arm that fails
with nothing held shut (A-open) and the arm that converges with the copy at one return (E0) place
the failure on the copy at every open return, not on the hold-shut rule by itself; what the
hold-shut rule does to an iteration that would otherwise converge was not measured, since it
never acted in a converging arm. Every earlier measurement of this ladder (the Reynolds report,
section 8; test 33b; ADR-012 D's reading that nothing converges above Re 895) was made with the
copy at every open return. Step 5's convergence measurement on 80x30 and 200x75 at real air would
be made under step 3's outlets; whether ADR-013 decision 6's finding (no convergence on either at
real air with ten sweeps) was the outlets' or the grid's is open until then.

## Appendix: the scripts

`docs/reports/probe41/`, committed with this report:

- `outlet41.py`: the arms (`ArmSolver`, `Segment`, the rules), `SweepPredictor`,
  `RecordingCorrector`, the rooms, the runner, the control and the comparisons.
- `run41.sh`: every run in the order launched.
- `summary41.py`: the tables of section 7 from the records under `results/builder41/`.

SHA-256 of the committed bytes, which have LF line endings (`git show HEAD:<path> | sha256sum`
gives them; a Windows working copy under `autocrlf` is CRLF and hashes differently, which test
41 S2 found for the first round's table):

| File | SHA-256 of the committed bytes |
|---|---|
| `outlet41.py` | 9d91d299630129ef96ae002e5ba0e86e07601ed7195671e2df9a5acb9232badc |
| `run41.sh` | 2e6cfb8e0a8ab46faccafa6d01f2a7541ad2860d488d7c320381d90665b9f60c |
| `summary41.py` | 6566aaba4d3116cbbc4510296be6cbe5eac93f065d12702ddfa1887f7c2b43ea |

