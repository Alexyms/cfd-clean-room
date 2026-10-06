# ECR-003: The Pressure Solver, Measured

**Date:** 2026-10-06.
**Branch:** `docs/ecr-003-pressure-solver`, on main at ed83807, in the worktree
`cfd_clean_room_ecr003`. The momentum, pressure, solver and boundary modules are the same at
ed83807 and at main's 023b8f6, so what is measured here holds on both.
**Probes:** `results/builder36/` in the worktree (untracked), reproduced in the appendices.
**Order:** sections 1 to 5 (the question, the method, the controls on the harness, what each
outcome means, and both sets of predictions) are the first commit, made before item 0 and before
measurements 1 to 3 ran. The sections after them were written after the runs.
**36b:** section 12 and appendices Q to S were added in the fix pass of prompt 36b, after premise
review 36 and test 36, from runs of that pass in `results/builder36b/` (untracked). Sections 1 to 5
are unchanged; notes marked "36b" in sections 8, 10 and 11 point at section 12.

## 1. The question

One pressure correction on the 200x75 product mesh needs 27,408 weighted Jacobi sweeps to the
committed tolerance and 112,519 to 1e-8 Pa, against a committed cap of 200, and the count grows as
the square of the cells per side (`docs/reports/product_case_reynolds.md`, section 5). A steady
product solve needs thousands of corrections, so it would take hours (ADR-012 H). Two questions
follow, and both matter:

1. What should solve the correction?
2. How tightly does each correction need solving?

A solver a hundred times faster per correction is worth less if the outer loop needs no more
than a few hundred sweeps' accuracy anyway. Two earlier measurements hint that it may not. On the
collocated cavity the outer count did not depend on the sweep cap (`docs/reports/pressure_solver_probe.md`),
but that system was singular and inconsistent, so nothing could converge it. On the 40x15 room
under T3, raising the cap tenfold left the outer history unchanged (step 0's report, section 6.4),
but both caps there solved the correction far past any loose level.

The prompt sets the order: item 0, the gate, checks that the systems the candidates will solve are
symmetric and positive definite; measurement 1 times each candidate on one correction at one
accuracy measure; measurement 2 finds how tightly the outer loop needs it; measurement 3 projects a
steady product solve.

## 2. Method

### 2.1 Environment

AMD Ryzen AI 9 HX 370 (24 logical processors), 63 GB of memory, Windows 11, Python 3.13.3. A probe
environment, `results/builder36/venv36`, holds NumPy 2.4.4 (the project environment's version),
PyYAML 6.0.3, matplotlib 3.10.8, SciPy 1.18.1 and pyamg 5.3.0. SciPy and pyamg are installed there
only; `requirements.txt` is not touched. Every timed run sets `OPENBLAS_NUM_THREADS`,
`OMP_NUM_THREADS` and `MKL_NUM_THREADS` to 1, so the seconds compare one core against one core; the
dense Cholesky factorizations of item 0 are untimed and use every thread.

`results/builder33b/outlet33b.py` and `results/builder34/frozen34.py` are byte copies of the main
tree's (SHA-256 2ad2b9f2... and 3a85849b...), placed in the worktree so they import its `src/`.
Nothing in `src/` is edited, patched or imported into.

### 2.2 The rooms and the captured systems

`capture36.py` (appendix D) runs a room and saves the p' system the corrector is handed at outer
iterations 1, 100 and 1,000 (counted from zero, as every report here counts them): the five
coefficient arrays, b, the face d values, u*, v*, the outlet masks, the SOLID mask and the mesh
spacing. `ProbeCorrector` (appendix A) is a subclass of `PressureCorrector` that replaces only the
p' solve; the coefficients, the right-hand side, the pin, the velocity correction and the pressure
update are the committed lines, and with no solve given it calls the committed `correct` itself.

| Room | What it is | The correction that drives it | Stop |
|---|---|---|---|
| product200 | `configs/clean_room_default.yaml` on 200x75 under T3 (`outlet33b.py`: the hood at 0.5 m/s, floor-return faces turning inward held shut each outer iteration), air's viscosity, no eddy viscosity, ten momentum sweeps per outer iteration (`frozen34.py`'s switch on its zero field), alpha_velocity 0.5, from rest | SuperLU, exact | error_estimate, to 20,000 |
| product40 | the same on 40x15 | SuperLU | error_estimate, to 20,000 |
| cavity80 | VAL-002 80x80 as its case file sets it, velocity_step | today's (cap 500, 1e-8 Pa) | 1,001 outer iterations |
| channel80 | VAL-001 80x40 likewise | today's (cap 2,000, 1e-8 Pa) | 1,001 |
| annex180 | the Annex 20 room as `cost33.py` builds it, on 180x60, laminar, its outlet under T1 (inward faces held shut), ten sweeps | SuperLU | 1,001 |

The product run is driven by an exact correction for two reasons. Step 0's settings (a 40,000-sweep
cap toward 1e-8 Pa) would hit the cap on every correction on 200x75, where 112,519 are needed at
outer 0, so the captured states would carry that cap's under-solve; and every accurate enough
solver approximates the exact path, so it is the path the candidates will face. It also gives
measurement 3 the product's outer count, if the room converges. The validation cases are driven by
their own committed corrections, the path they were validated on.

The error_estimate stop uses the defaults' `iteration_error_tol` 1e-6 and `mass_imbalance_tol`
from ADR-011 G's formula, `1e-4 rho V_min / T` at the configured `t_end` of 60 s: 3.2e-9 kg/s per
metre on 200x75, 2.0e-8 on 80x30 and 8.0e-8 on 40x15. The velocity scale is the supply's 0.45 m/s
and the flux scale the supply's mass flow, 3.78 kg/s per metre on 200x75 and 80x30 and 3.888 on
40x15, where the 0.2 m faces cover 7.2 m of the 7.0 m supply.

**Item 0's checks** (`item0_36.py`, appendix E), on the cells with an equation:

- symmetry to the bit, `a_e[j, i]` against `a_w[j, i+1]` and `a_n[j, i]` against `a_s[j+1, i]`,
  the two copies of each face coefficient, and as a matrix, `max |A - A^T| / max |A|`;
- the sign structure (positive diagonal, non-positive off-diagonals, the row surplus
  `a_P - sum(a_nb)` non-negative), the rows where the surplus is positive (open outlet faces,
  p' = 0) and the connected components of the cell graph;
- the three smallest eigenvalues of A by shift-invert and the largest by Lanczos, and the same for
  `D^-1 A`, whose smallest eigenvalue sets Jacobi's slow rate and whose ratio is conjugate
  gradients' condition number;
- a dense Cholesky factorization of the whole matrix, on the cavity with the pin cell's row and
  column removed. It succeeds exactly when the matrix is positive definite. With 63 GB the
  product's 10,910 unknowns fit (1 GB), so the 40x15 copy is a second check, not a substitute;
- on the cavity, `A 1 = 0` and the compatibility `|sum b| / sum |b|`.

### 2.3 The candidates

Each is fixed here, before any captured system is solved, and none is tuned on the captured
systems afterwards (`solvers36.py`, appendix B).

- **A. Weighted Jacobi**, today's: `PressureCorrector.sweep` with `JACOBI_WEIGHT`, stopped as
  `correct` stops it, on the largest weighted change of p' in one sweep below a tolerance in
  pascals.
- **B. Jacobi-preconditioned conjugate gradients**, NumPy: the five-point product as five shifted
  array operations, the diagonal as preconditioner, three reductions per iteration (CG's two and
  the residual norm the stop reads). On the cavity the right-hand side is projected onto the range
  (its mean over the cells with an equation removed) and the pin applied after, as now.
- **C. Geometric multigrid with Galerkin coarse operators**, NumPy. Prolongation P is cell-centred
  bilinear: each fine cell takes 9/16, 3/16, 3/16 and 1/16 of its parent coarse cell and the three
  coarse cells nearest it, the weights renormalized over the coarse cells that exist and have an
  equation, so P reproduces a constant and a closed domain's null space reaches every level.
  Restriction is P^T, and the coarse operator `R A P` is formed by probing with 25 coloured vectors:
  the coarse stencil is at most 5 x 5, so each colour meets one stencil entry per row. Obstacles
  and the varying coefficients reach the coarse grids through `R A P`; nothing is rediscretized. A
  level with an odd side is padded with an inactive row or column. Smoothing is weighted Jacobi,
  with the committed weight 2/3 on the finest level and `4 / (3 lambda_max(D^-1 A))` on the coarse
  ones, lambda_max from 30 power iterations, since a Galerkin operator need not be diagonally
  dominant. V(2,2) cycles, coarsened to at most 64 cells or a side of 4, the coarsest solved by a
  pseudo-inverse. Run standalone and as the preconditioner of CG (MG-PCG): a V(2,2) from zero with
  the same smoother before and after is symmetric.
- **D. Algebraic multigrid, pyamg**, in the probe environment: smoothed aggregation with its
  defaults (symmetric block Gauss-Seidel smoothing), standalone and as the preconditioner of CG;
  Ruge-Stuben with its defaults as the preconditioner of CG; and smoothed aggregation with weighted
  Jacobi smoothing (2/3, two sweeps) as the preconditioner of CG, the data-parallel variant. The
  near-null-space vector is the constant, and the coarse solve a pseudo-inverse so the singular
  cavity works.
- **E. SuperLU** (SciPy's sparse LU), the reference. It is not one of the prompt's candidates. It
  gives the exact solution every other result is measured against, and its cost is reported because
  at 11,000 unknowns a sparse factorization is a real option that any decision should see.

### 2.4 One accuracy measure for every candidate

Today's stop, the largest per-sweep change of p' in pascals, is not a measure of the error, and CG
and multigrid have no such quantity. Every candidate is stopped and judged instead on the relative
residual of the p' equation, `||f - A x||_2 / ||f||_2` over the cells with an equation, with
`f = -b`. pyamg's own stop is the same quantity.

That residual is also the corrected faces' continuity. The corrector moves each correctable face by
`-d (p'_(s+) - p'_(s-))`, so the corrected cell's imbalance is `b + A p'`: the residual of the p'
equation, cell by cell, with its sign changed. A relative residual of 1e-2 means the corrected faces
carry one per cent of the imbalance u* carried, in the 2-norm. `item0_36.py` checks the identity
on every captured system by correcting faces with an arbitrary p' through the committed
arithmetic. So every result is reported both ways: the relative residual, and the corrected faces'
worst-cell, absolute-summed and signed-summed imbalance, each over the supply's mass flow, which are
the quantities REQ-S04's clauses and the error_estimate rule's conditions (b) to (d) read. Each
result is also compared with SuperLU's: the largest error of p' relative to p', and the largest
error of the corrected face velocity in m/s, the quantity the next outer iteration sees.

### 2.5 Measurement 1

`m1_36.py` (appendix F), on each captured system:

- today's loop to 1e-6 Pa (the committed tolerance) and 1e-8 Pa (the validation cases' and step
  0's), uncapped to 5,000,000 sweeps, and at the committed cap of 200: where each lands on the
  relative-residual scale is the answer to "what today's 1e-6 Pa stop corresponds to";
- today's sweep run until each relative level 1e-1, 1e-2, 1e-4, 1e-6 and 1e-8 is first crossed,
  checked every 10 sweeps, up to 2,000,000 sweeps, untimed; its seconds are the sweeps times the
  timed loop's seconds per sweep;
- B, C (standalone and MG-PCG) and the four D variants to each of the five levels;
- E once.

Timings are the median of three runs for any solve under two seconds, setup (the multigrid
hierarchy, pyamg's setup with its CSR assembly, the LU factorization) reported apart from the
solve. Since the matrix changes every outer iteration, a fresh setup per correction is the honest
cost; MG-PCG and SA-CG are also run with the hierarchy built on the previous captured system of the
same run (a stale preconditioner; CG still iterates on the current matrix), which bounds what
reusing a hierarchy over several outer iterations would save.

### 2.6 Measurement 2

The product room under T3 on 40x15 and 80x30, laminar at air's viscosity, ten momentum sweeps per
outer iteration, alpha_velocity 0.5, from rest (`outer36.py`, appendix G).

**Why laminar real air, not step 0's Z2 field.** Test 34b converged exactly this room on 40x15
with ten sweeps (1,208 outer iterations, step 0's report section 7.6), and its record `A-sw` is a
bitwise control for the run with today's correction at step 0's settings. It needs no frozen field,
which on 80x30 would mean recomputing step 0's base field at Re 90 on that grid, a second probe
chain. And with ten sweeps the eddy-viscosity runs followed nearly the laminar residual path
(section 7.4 there), so the laminar room stands for them on the question asked here, which is about
the coupling, not the mixing.

**The stops.** The solver stops by error_estimate (the tolerances of section 2.2) or at 20,000
outer iterations. On the way, the outer iteration at which velocity_step (the residual below 1e-6,
step 0's stop) would have stopped is recorded with the field there; the stopping rule does not
change the path, so one run gives both stops.

**The per-correction accuracies**, each a run on each grid:

| Run | The correction |
|---|---|
| direct | SuperLU, exact: the tightest run |
| pcg:3e-1, pcg:1e-1, pcg:1e-2, pcg:1e-4, pcg:1e-8 | candidate B to that relative residual |
| jacobi200 | today's loop at the committed cap 200 and 1e-6 Pa |
| jacobi40k | today's loop at step 0's cap 40,000 and 1e-8 Pa (on 40x15, test 34b's A-sw) |

Every Krylov run also stops at an absolute residual of 1e-13 times the supply's mass flow, far below
any tolerance here, so a right-hand side at rounding level cannot run a correction to its cap. At
the loosest level that passes, and at the next looser one, the run is repeated with MG-PCG, GMG
standalone and SA-CG, to check that the answer belongs to the level and not to CG's error shape.

**What "unchanged" means**, fixed here: a level is sufficient when (i) at both stops its outer
count is within 5% of direct's, and (ii) at both stops the largest difference of its cell-centred
velocity from direct's over the non-SOLID cells is within the sum of the two runs' own iteration
errors. At the velocity_step stop that error is estimated as the error_estimate rule estimates it,
`step rho_hat / (1 - rho_hat)` with rho_hat from the last 100 steps; at the error_estimate stop it
is at most `1e-6 x 0.45 m/s` per run by the rule itself.

**Fallback, fixed here.** If the direct run on 80x30 does not reach the velocity_step stop within
20,000 outer iterations, or diverges past 100 m/s, the 80x30 set is rerun at ten times air's
viscosity (Re 8,950, which test 34b converged on 40x15 at 1,310), and the real-air failure is
reported beside it.

Runs of one grid's set go in parallel, so their seconds carry that load; the outer counts are
deterministic.

### 2.7 Measurement 3

The cost of one steady product solve on 200x75 for each candidate at the level measurement 2 finds
sufficient, and the same on the Annex 20 180x60 grid:

    seconds = outer iterations x (momentum seconds per outer + correction seconds at that level)

The momentum seconds per outer iteration come from the product200 run (ten sweeps, measured), the
correction seconds from measurement 1 on that grid's three captured systems (setup included), and
the outer count from the product200 run's own stops if it converges, otherwise from measurement 2's
40x15 and 80x30 counts and ADR-012 H's range, each assumption stated. Where the best candidates
allow, one product solve is run end to end with the candidate at its level to check the projection
against a measured total.

## 3. Controls on the harness

All run before this section was written (`selftest36.py`, appendix C; `selftest36.json`).

| Control | Result |
|---|---|
| ProbeCorrector with the committed loop copied into its replaceable solve, against the committed `correct`, 40x15 T3 ten sweeps, 30 outer iterations | residuals, final faces and sweep counts bitwise equal |
| ProbeCorrector with the committed call, against test 34b's `A-sw` record (main tree, `results/tester34b/A-sw.json`, read only) | first 30 residuals and sweep counts bitwise equal |
| `cost33.py`'s room (the product as committed, T0, one sweep, alpha 0.7): the outer-0 system captured, today's loop run on it | 27,408 sweeps to 1e-6 Pa and 112,519 to 1e-8 Pa, the report's figures exactly; 0.23 and 0.25 ms per sweep (the report: 0.28) |
| Every candidate on a synthetic five-point system (45 x 61, two SOLID strips, a Dirichlet patch, face coefficients random over four orders of magnitude), to 1e-10 | B, MG-PCG, SA-CG, SA-Jacobi-CG and RS-CG converge and agree with SuperLU to 1e-11 to 3e-11 of p'. GMG standalone and SA standalone do not reach 1e-10 in 500 cycles (1.9e-4 and 1.8e-5) |
| The same system closed (singular, compatible right-hand side) | B, MG-PCG and RS-CG converge and agree with SuperLU to 1e-11; **SA-CG and SA-Jacobi-CG abort** after 18 and 17 iterations at 1.1e-2 and 3.6e-2 with pyamg's "indefinite preconditioner" and "indefinite matrix" warnings; GMG and SA standalone as above |
| The probed `R A P` against an explicit sparse `P^T A P`, first two coarse levels, both synthetic systems | largest relative difference 3e-16 to 4e-16; symmetric to 3e-16 |
| GMG standalone on constant-coefficient rectangles with no holes (32x32, 64x64, 45x61, 75x200), closed, a Dirichlet patch, Dirichlet on all edges | factor per V(2,2) cycle 0.19 closed at every size, 0.28 to 0.32 Dirichlet on all edges, 0.29 to 0.41 with a patch; 13 to 25 cycles to 1e-10 |
| `item0_36.py` on the cost33 control system (smoke test, not one of item 0's systems) | ran: symmetric to the bit, Cholesky succeeded on 10,910 unknowns in 6.3 s, `lambda_min(D^-1 A)` 8.1e-5, the identity of section 2.4 to 4e-15 (exact p') and 1.3e-13 (random p') of the largest b |

The synthetic coefficients jump by up to four orders of magnitude between neighbouring faces, which
the captured systems do not; that is where the geometric and the standalone algebraic cycles slow
down, and the rectangles show the geometric cycle is right where the coefficients are smooth. The
slow cycles on the synthetic system come from coarse cells that straddle the three-cell SOLID
strips, which a geometric coarsening cannot avoid. The closed-system abort of SA-CG is a property of
pyamg's smoothed aggregation on a singular system under these defaults; it is reported as found and
not tuned (section 2.3). Smoke tests of `outer36.py` (every mode, four outer iterations on 40x15)
and `capture36.py` (three on the cavity) ran before this commit to check the scripts run; they are
too short to say anything measurement 2 asks, and their outputs were moved aside.

## 4. What each outcome means

Written before the runs.

**Item 0.**
- Symmetric and positive definite on the product systems: conjugate gradients and the multigrid
  candidates apply as planned. Positive definite on the channel and the Annex 20 room; on the
  cavity singular with the constants as null space and a compatible right-hand side, which CG
  handles with the projection.
- Not symmetric, or not positive definite, on a product system: the work stops there. The
  candidate list changes to nonsymmetric Krylov methods (BiCGStab, GMRES) and multigrid for
  nonsymmetric systems, and the measurements do not run.

**Measurement 2**, in the prompt's terms. "Today's" accuracy is what the committed 1e-6 Pa stop
delivers when the cap does not bind, which measurement 1 places on the relative-residual scale.
- **A. A loose level, 1e-2 or looser, is sufficient.** Most of today's per-correction accuracy is
  wasted. The stop becomes a relative residual at the loosest sufficient level with a margin, and
  at loose levels a solver's setup weighs as much as its rate.
- **B. Only a tight level, 1e-4 or tighter, is sufficient, but no tighter than today's stop
  delivers.** The stop still changes to a relative residual; the ranking is measurement 1's at
  that level, where the rate dominates.
- **C. The outer loop needs tighter than today's stop delivers.** The premise reverses: the work
  stops and reports, and the ECR's emphasis moves from wasted accuracy to speed at tight accuracy.

**Measurement 3.** If any of B, C and D brings a steady product solve under an hour, the ranking
turns on what each costs the project (a dependency, a GPU path, code to keep) more than on speed.
If none does, the C path (CLAUDE.md's language policy) is needed now, not deferred.

## 5. Predictions

### 5.1 The orchestrator's (prompt 36, written before handover)

"Item 0: symmetric and positive definite on the open-boundary systems. Measurement 1: CG takes a
few hundred iterations on 200x75 to a relative residual of 1e-6 (iterations growing as the cells
per side), about two orders faster than Jacobi in seconds; MG-PCG and pyamg take tens of
iterations and are fastest, pyamg by a factor of a few over hand-written NumPy multigrid.
Measurement 2: a relative residual of about 1e-2 per correction leaves the outer count and field
unchanged, as the collocated probe and step 0's cap diagnostic suggest, so most of today's cost is
wasted accuracy. Measurement 3: a steady product solve drops from hours to minutes with any of B, C
or D; the ranking then turns on dependencies and the GPU path more than on speed."

### 5.2 Mine (written before item 0 and the measurements ran)

I saw the controls of section 3 before writing these, among them the cost33 control system's
`lambda_min(D^-1 A)` of 8.1e-5 and the synthetic results.

**Item 0.**
- I1. Every captured matrix is symmetric to the bit: `coefficients` takes both copies of a face
  coefficient from one array, `coef_u[:, 1:]` and `coef_u[:, :-1]`.
- I2. Every one has non-positive off-diagonals and weakly dominant rows, one connected component,
  and on the open domains a strict row in it at an open outlet face: an irreducibly diagonally
  dominant symmetric M-matrix, so positive definite. Cholesky succeeds on every product, channel
  and Annex 20 system.
- I3. The cavity: `A 1 = 0` to 1e-15 relative, the right-hand side compatible to 1e-15 relative,
  one eigenvalue at zero to rounding; with the pin row and column removed, Cholesky succeeds.
- I4. On 200x75 under T3, `lambda_min(D^-1 A)` between 2e-5 and 1.5e-4: T3 holds the hood and shuts
  inward return faces, so fewer strict rows than the cost33 control's five open outlets.

**Measurement 1, on the 200x75 systems.**
- P1. Today's 1e-6 Pa stop takes 5,000 to 30,000 sweeps and lands at a relative residual between
  1e-4 and 1e-2. The slow mode changes by `(2/3) lambda_min` times its error per sweep, so the stop
  leaves an error near `1e-6 / 5e-5`, about 0.02 Pa, in that mode.
- P2. The committed cap of 200 leaves a relative residual above 0.3 at outer 1, and the corrected
  faces there carry a net outflow error of more than half the supply.
- P3. Today's sweep needs more than 150,000 sweeps (more than 35 s) to reach a relative residual of
  1e-6.
- P4. B reaches 1e-6 in 300 to 800 iterations, 0.1 to 0.4 s.
- P5. C standalone takes 15 to 50 V(2,2) cycles to 1e-6, at 0.4 to 0.7 per cycle, slower than on
  the rectangles because obstacles and outlets cut the coarse grids; MG-PCG takes 10 to 25
  iterations; either costs 0.05 to 0.2 s with its setup.
- P6. SA-CG and RS-CG take 8 to 20 iterations to 1e-6 and 20 to 80 ms with setup, the fastest
  iterative candidates, two to four times faster than MG-PCG in total.
- P7. SuperLU costs 30 to 100 ms per correction, level with pyamg at 1e-6.
- P8. At 1e-1, B is the fastest NumPy candidate per correction, because it has no setup, and within
  a factor of two of pyamg.

**Measurement 2.**
- Q1. jacobi40k on 40x15 reproduces `A-sw` bitwise and reaches the velocity_step stop at 1,208.
- Q2. direct on 40x15 reaches the velocity_step stop within 10% of 1,208, and the error_estimate
  stop at 1.5 to 3 times that count.
- Q3. On both grids **1e-1 is the loosest sufficient level**: counts within 5% of direct's at both
  stops, the fields within the iteration errors. 1e-2 and tighter are indistinguishable from
  direct. 3e-1 converges, with a count more than 5% from direct's.
- Q4. At the error_estimate stop every converged run, at any level, carries the same field to
  within twice the stop's iteration error: the fixed point does not depend on the inner tolerance,
  because at a fixed point p' is zero and a relative residual then forces b to zero.
- Q5. jacobi200 converges on 40x15 with a count within 25% of direct's, and does not converge on
  80x30 (diverges, or never meets error_estimate within 20,000), where 200 sweeps leave the
  first corrections far short.
- Q6. jacobi40k and jacobi200 reach the error_estimate stop later than the relative-residual runs,
  by more than 10% where both converge. Once b is small, the first sweep's change, `(2/3) |b| / a_P`,
  is below the absolute tolerance, so today's loop falls to one sweep, which removes the local part
  of the imbalance and leaves the smooth part; that part then decays only as fast as one sweep's
  smooth mode per outer iteration. A relative residual has no such floor. This is the flux drift
  `docs/STATUS.md` records on the channel.

**Measurement 3.**
- R1. The 200x75 laminar product does not converge with the exact correction within 20,000 outer
  iterations: it stalls or oscillates without passing 100 m/s. The grid is 25 times finer in cells
  than step 0's, and resolves shear layers that 40x15 smears. If it does converge, its count is
  measurement 3's outer count.
- R2. On 200x75 ten momentum sweeps cost 30 to 80 ms per outer iteration and B's correction at 1e-1
  10 to 50 ms, so 0.05 to 0.15 s per outer iteration; at 3,000 to 15,000 outer iterations a steady
  product solve takes 3 to 40 minutes with B, C or D, against 3 to 30 hours with today's loop at
  its 1e-6 Pa stop.
- R3. B, C and D come within a factor of three of each other per steady solve, so the ranking turns
  on dependencies and the GPU path.

## 6. Item 0: the systems the candidates solve (written after the runs)

**Order.** The first commit (67e2f63) is dated 2026-10-06 09:27:45 -0700. The capture runs started
at 09:27:57 (each log opens with its start time), item 0's checks ran from 09:29 to 09:35,
measurement 1's timed runs from 09:35 to 09:39 and its untimed Jacobi runs after them, and
measurement 2 from 09:39:33. The diagnostics and rungs added after measurement 2's first runs
followed from 09:42 (restart) to 09:58 (the outlet diagnostic), and measurement 3 from 10:00.

**Symmetric and positive definite on every open system; the gate passes.** On all fifteen captured
systems the two copies of every face coefficient are the same double, so `A - A^T` is zero exactly,
not to rounding. Every diagonal is positive, every neighbour coefficient non-negative and every
row's surplus `a_P - sum(a_nb)` non-negative. On the product, channel and Annex 20 systems the cell
graph is one component with strict rows in it at the open outlet faces, and the dense Cholesky
factorization of the whole matrix succeeds on each, the product's 10,910 unknowns included. The
cavity is singular as expected and compatible, and its matrix with the pin cell removed factors. So
conjugate gradients and the multigrid candidates apply as planned (`item0_*.json`; `table36.py
item0`):

| System | Unknowns | Symmetric to the bit | Strict rows | Components (with a strict row) | lambda_min(A) | lambda_min(D^-1 A) | lambda_max(D^-1 A) | Cholesky (unknowns, s) |
|---|---|---|---|---|---|---|---|---|
| product200 outer 1 | 10,910 | yes | 56 | 1 (1) | 8.7e-6 | 3.7e-6 | 2.0 | succeeded (10,910, 6.0) |
| product200 outer 100 | 10,910 | yes | 56 | 1 (1) | 7.4e-6 | 1.5e-5 | 2.0 | succeeded (10,910, 5.8) |
| product200 outer 1000 | 10,910 | yes | 56 | 1 (1) | 6.2e-6 | 1.4e-5 | 2.0 | succeeded (10,910, 5.9) |
| product40 outer 1 | 430 | yes | 13 | 1 (1) | 9.1e-4 | 2.2e-4 | 2.0 | succeeded |
| product40 outer 100 | 430 | yes | 13 | 1 (1) | 9.0e-4 | 9.6e-5 | 2.0 | succeeded |
| product40 outer 1000 | 430 | yes | 12 | 1 (1) | 6.5e-4 | 6.9e-5 | 2.0 | succeeded |
| channel80 outer 1, 100, 569 | 3,200 | yes | 40 | 1 (1) | 1.4e-6 to 1.5e-6 | 9.7e-5 to 9.8e-5 | 2.0 | succeeded |
| annex180 outer 1 | 10,800 | yes | 10 | 1 (1) | 3.4e-5 | 4.1e-6 | 2.0 | succeeded (10,800, 5.6) |
| annex180 outer 100 | 10,800 | yes | 10 | 1 (1) | 2.9e-5 | 6.0e-6 | 2.0 | succeeded |
| annex180 outer 1000 | 10,800 | yes | 10 | 1 (1) | 1.4e-5 | 1.0e-5 | 2.0 | succeeded |
| cavity80 outer 1, 100, 1000 | 6,400 | yes | 0 | 1 (0) | zero to rounding (-2e-19 to -7e-19), next 5.7e-6 to 6.0e-6 | zero to rounding | 2.0 | succeeded with the pin removed (6,399, 1.4) |

On the cavity `A 1` is zero to 1.7e-16 of the largest diagonal, and `|sum b| / sum |b|` is 1.4e-17
at outer 1, 1.7e-15 at outer 100 and 3.7e-14 at outer 1,000: the sum stays at rounding while the
imbalance it is divided by falls by four orders. The channel was captured at its last iteration,
569, instead of 1,000: under velocity_step VAL-001 80x40 stops at outer 570. The identity of section
2.4, the committed imbalance arithmetic on faces corrected with any p' against `b + A p'`, holds to
6e-14 of the largest corrected face flux on every system, at the exact p' and at a random one.

**What the eigenvalues say about today's loop.** Weighted Jacobi's slowest mode shrinks by `1 -
(2/3) lambda_min(D^-1 A)` per sweep. On the product's systems that is 1 - 2.5e-6 at outer 1 and 1 -
1e-5 later: hundreds of thousands of sweeps per factor e. The cost33 control, the same room as
committed (T0, one sweep, alpha 0.7) at outer 0, had 8.1e-5. The T3 systems hold the hood at its
flow and so have 56 strict rows against 79, and the outer-1 system follows a first step taken with
alpha 0.5 and ten sweeps; the two were not separated. `lambda_max(D^-1 A)` is 2.0 to four figures on
every system, the bipartite five-point bound, so the condition number CG faces is `2 / lambda_min`:
about 5e5 at the product's outer 1 and 1.4e5 after.

**Against the predictions.** I1 (symmetric to the bit), I2 (M-matrix, Cholesky succeeds) and the
orchestrator's item-0 prediction held. I3 held in substance; its compatibility figure (1e-15) held
at outers 1 and 100 and missed at outer 1,000 (3.7e-14), where the rounding of the sum is divided by
an imbalance four orders smaller. I4 missed: `lambda_min(D^-1 A)` on 200x75 under T3 is 3.7e-6 to
1.5e-5, below the 2e-5 to 1.5e-4 predicted from the control.

## 7. Measurement 1: one correction, each candidate (written after the runs)

The timed runs ran one at a time from 09:35 to 09:39, beside the product200 capture run only. The
untimed Jacobi runs ran after them, in parallel (`m1_*.json`, `m1hist_*.json`; `table36.py m1mean`,
`hist_table36.py`).

### 7.1 Today's 1e-6 Pa stop on the relative-residual scale

Today's loop from p' = 0, the committed sweep and stop, on each captured system. The seconds are the
sweeps times a timed loop's seconds per sweep.

| System | ms per sweep | 1e-6 Pa stop: sweeps, s, relative residual, net outflow / supply | 1e-8 Pa stop: sweeps, s, relative residual | Sweeps to 1e-2 | to 1e-4 | to 1e-6 | to 1e-8 | Cap 200: relative residual, net outflow / supply |
|---|---|---|---|---|---|---|---|---|
| product200 1 | 0.210 | 378,336, 79.3, 2.9e-03, -2.6e-03 | 2,256,870, 473.2, 2.9e-05 | 1,090 | 1,758,140 | 3,636,670 | > 5,000,000 | 0.03, -4.8e-02 |
| product200 100 | 0.203 | 57,257, 11.6, 2.2e-03, -1.2e-03 | 541,640, 110.0, 2.0e-05 | 2,200 | 381,360 | 842,390 | 1,303,420 | 0.05, -1.3e-02 |
| product200 1000 | 0.218 | 27,916, 6.1, 9.7e-04, -9.6e-04 | 499,758, 109.0, 1.1e-05 | 720 | 260,420 | 762,390 | 1,264,220 | 0.02, -6.9e-03 |
| annex180 1 | 0.157 | 1,816, 0.3, 7.9e-04, -6.5e-02 | 470,678, 73.9, 4.8e-05 | 160 | 200,530 | 1,883,890 | 3,567,250 | 0.01, -7.6e-02 |
| annex180 100 | 0.153 | 903, 0.1, 3.8e-03, -1.6e-02 | 72,453, 11.1, 8.2e-04 | 380 | 596,870 | 1,756,780 | 2,916,690 | 0.02, -2.0e-02 |
| annex180 1000 | 0.151 | 1,448, 0.2, 4.9e-03, -2.2e-02 | 246,067, 37.3, 1.1e-04 | 670 | 264,550 | 955,140 | 1,645,730 | 0.03, -4.1e-02 |
| product40 1 | 0.064 | 29,023, 1.8, 3.1e-05, 1.4e-04 | 60,140, 3.8, 3.1e-07 | 170 | 21,050 | 52,170 | 83,280 | 0.01, -1.7e-02 |
| product40 100 | 0.046 | 9,393, 0.4, 7.5e-03, -3.1e-04 | 81,258, 3.7, 7.5e-05 | 4,040 | 76,850 | 148,710 | 220,560 | 0.03, -6.7e-03 |
| product40 1000 | 0.045 | 48, 0.0, 7.7e-02, -4.3e-05 | 3,308, 0.1, 5.2e-03 | 1,680 | 91,570 | 191,370 | 291,170 | 0.08, -4.3e-05 |
| channel80 1 | 0.081 | 108,361, 8.8, 5.2e-04, -9.3e-04 | 178,956, 14.5, 5.2e-06 | 63,000 | 133,600 | 204,190 | 274,790 | 0.62, -9.9e-01 |
| channel80 100 | 0.076 | 6,804, 0.5, 3.3e-01, 9.1e-04 | 78,079, 6.0, 3.3e-03 | 61,050 | 132,340 | 203,640 | 274,930 | 0.51, 1.4e-03 |
| channel80 569 | 0.071 | 1, 0.0, 9.5e-01, 9.2e-06 | 505, 0.0, 8.9e-01 | 69,920 | 141,180 | 212,440 | 283,700 | 0.95, 9.2e-06 |
| cavity80 1 | 0.125 | 18,005, 2.2, 4.3e-04, -1.1e-17 | 35,495, 4.4, 4.4e-06 | 6,180 | 23,600 | 41,130 | 58,660 | 0.10, -1.0e-17 |
| cavity80 100 | 0.105 | 3,500, 0.4, 1.3e-01, -6.1e-19 | 19,617, 2.1, 1.7e-03 | 12,940 | 30,440 | 47,940 | 65,440 | 0.39, -3.5e-19 |
| cavity80 1000 | 0.105 | 3, 0.0, 9.5e-01, -5.0e-19 | 8,040, 0.8, 2.9e-02 | 12,110 | 29,550 | 46,990 | 64,440 | 0.95, -5.0e-19 |

**Today's stop corresponds to no fixed accuracy.** On the product's systems the 1e-6 Pa stop takes
28,000 to 378,000 sweeps, 6 to 79 s, and lands at a relative residual of 1e-3 to 3e-3; on the Annex
20 room 900 to 1,800 sweeps and 8e-4 to 5e-3; on the 40x15 room's first correction 3e-5. As a solve
converges its right-hand side shrinks, the first sweep's change falls below the tolerance, and the
stop fires after one to three sweeps: on the channel at outer 569 (one sweep) and the cavity at
outer 1,000 (three) it leaves a relative residual of 0.95, the right-hand side all but untouched.
The stop's quantity is the largest weighted change of p' in one sweep, `(2/3) lambda_min(D^-1 A)`
times the slow mode's error once that mode dominates, so with `lambda_min` at 4e-6 to 1.5e-5 on the
product a 1e-6 Pa change means an error of 0.1 to 0.4 Pa in that mode, while on a small right-hand
side the change is small from the first sweep. Reaching a relative residual of 1e-6 takes today's
sweep 760,000 to 3.6 million sweeps on the product's systems (166 to 764 s) and up to 1.9 million on
the Annex 20 room. The product report's 27,408 sweeps (section 5 there) are reproduced exactly on
its own system, the room as committed at outer 0 (section 3); on the systems the T3 room with ten
momentum sweeps builds, the same tolerance takes 1 to 14 times as many.

The committed cap of 200 sweeps leaves 0.01 to 0.08 on the open rooms, where the first sweeps take
the rough part of the residual, and 0.1 to 0.95 on the closed cavity and the channel; on the
channel's first correction it leaves 99% of the through-flow as net outflow error.

### 7.2 Each candidate on the 200x75 product

The mean over the three captured systems (outer 1, 100 and 1,000), with the range; the iterations on
each system; and, at the largest of the three, the relative residual reached, the corrected faces'
net outflow error over the supply's 3.78 kg/s per metre, the error of p' and the error of the
corrected face velocity against SuperLU.

**product200**, outer 1, 100 and 1000

| Candidate | Level | Total ms, mean [min, max] | Setup ms, mean | Iterations | Relative residual, largest | Net outflow / supply, largest abs | p' error, largest | Face error m/s, largest |
|---|---|---|---|---|---|---|---|---|
| E_direct | exact | 34.7 [34.2, 35.5] | 33.0 | 1 | - | - | 0 | 0 |
| A_jacobi_cap200 | cap 200 | 42.1 | 0 | 200 | 0.0471 | 0.0477 | 0.705 | 0.123 |
| B_pcg | 0.1 | 3.7 [3.1, 4.2] | 0.1 | 24, 26, 14 | 0.097 | 0.0351 | 0.688 | 0.103 |
| C_gmg | 0.1 | 127.7 [124.5, 133.1] | 122.8 | 2, 1, 1 | 0.0795 | 0.0152 | 0.446 | 0.0387 |
| C_mg_pcg | 0.1 | 135.7 [132.8, 137.6] | 129.1 | 2, 2, 1 | 0.0743 | 8.9e-03 | 0.28 | 0.0305 |
| D_sa | 0.1 | 39.4 [36.3, 40.9] | 35.7 | 1, 1, 1 | 0.0958 | 4.7e-03 | 0.159 | 0.133 |
| D_sa_cg | 0.1 | 44.5 [39.5, 49.9] | 37.8 | 2, 1, 1 | 0.0897 | 2.1e-03 | 0.129 | 0.0322 |
| D_sa_jacobi_cg | 0.1 | 77.1 [76.5, 77.9] | 65.1 | 3, 2, 2 | 0.0798 | 2.8e-03 | 0.1 | 0.0613 |
| D_rs_cg | 0.1 | 44.6 [42.2, 47.2] | 37.3 | 1, 1, 1 | 0.0649 | 4.9e-03 | 0.165 | 0.0389 |
| B_pcg | 0.01 | 43.7 [18.1, 62.1] | 0.1 | 313, 245, 87 | 1.0e-02 | 4.7e-03 | 0.542 | 0.0656 |
| C_gmg | 0.01 | 181.6 [176.2, 185.2] | 164.7 | 3, 4, 3 | 9.9e-03 | 7.4e-03 | 0.0733 | 0.017 |
| C_mg_pcg | 0.01 | 176.7 [172.1, 181.9] | 160.3 | 4, 3, 3 | 6.1e-03 | 9.5e-05 | 0.0118 | 1.6e-03 |
| D_sa | 0.01 | 54.0 [47.2, 62.5] | 44.5 | 3, 3, 3 | 4.3e-03 | 5.2e-04 | 0.0427 | 0.0109 |
| D_sa_cg | 0.01 | 52.2 [50.8, 54.4] | 41.7 | 3, 3, 2 | 8.0e-03 | 1.3e-04 | 0.0182 | 5.1e-03 |
| D_sa_jacobi_cg | 0.01 | 77.1 [70.1, 82.6] | 63.2 | 6, 5, 4 | 9.8e-03 | 8.4e-05 | 0.0113 | 0.0136 |
| D_rs_cg | 0.01 | 45.3 [43.5, 47.1] | 32.9 | 2, 3, 2 | 8.2e-03 | 2.8e-04 | 0.0352 | 7.8e-03 |
| B_pcg | 0.0001 | 134.1 [132.1, 138.0] | 0.1 | 703, 680, 672 | 9.8e-05 | 5.0e-06 | 5.6e-04 | 7.9e-05 |
| C_gmg | 0.0001 | 209.2 [198.4, 218.2] | 168.6 | 9, 10, 8 | 9.8e-05 | 1.1e-04 | 2.9e-03 | 3.9e-04 |
| C_mg_pcg | 0.0001 | 193.8 [186.6, 201.7] | 167.4 | 7, 6, 5 | 9.5e-05 | 3.0e-07 | 1.3e-04 | 3.4e-05 |
| D_sa | 0.0001 | 64.1 [55.6, 76.5] | 41.3 | 10, 8, 6 | 9.9e-05 | 1.0e-05 | 3.3e-03 | 3.5e-04 |
| D_sa_cg | 0.0001 | 64.8 [59.0, 72.6] | 42.8 | 7, 6, 5 | 7.3e-05 | 8.8e-07 | 1.2e-04 | 3.2e-05 |
| D_sa_jacobi_cg | 0.0001 | 105.2 [96.5, 111.2] | 74.3 | 12, 11, 9 | 9.7e-05 | 4.4e-07 | 1.3e-04 | 1.4e-04 |
| D_rs_cg | 0.0001 | 57.8 [53.7, 60.5] | 36.3 | 5, 5, 5 | 5.1e-05 | 1.2e-06 | 1.5e-04 | 5.1e-05 |
| B_pcg | 1e-06 | 154.5 [150.4, 160.6] | 0.1 | 791, 794, 792 | 1.0e-06 | 1.4e-08 | 2.2e-06 | 6.0e-07 |
| C_gmg | 1e-06 | 243.3 [228.6, 260.6] | 161.5 | 17, 18, 15 | 8.4e-07 | 5.2e-07 | 2.6e-05 | 6.7e-06 |
| C_mg_pcg | 1e-06 | 202.9 [190.2, 210.2] | 162.2 | 9, 8, 8 | 6.2e-07 | 2.6e-09 | 2.4e-06 | 2.6e-07 |
| D_sa | 1e-06 | 101.3 [90.8, 113.9] | 48.0 | 20, 18, 13 | 8.0e-07 | 7.4e-08 | 2.7e-05 | 2.8e-06 |
| D_sa_cg | 1e-06 | 78.4 [71.8, 85.0] | 48.4 | 10, 9, 8 | 6.5e-07 | 2.6e-09 | 9.3e-07 | 3.3e-07 |
| D_sa_jacobi_cg | 1e-06 | 116.3 [107.8, 120.8] | 66.1 | 17, 15, 14 | 8.0e-07 | 3.2e-09 | 1.3e-06 | 2.4e-07 |
| D_rs_cg | 1e-06 | 63.6 [57.9, 69.8] | 35.0 | 7, 8, 8 | 7.6e-07 | 5.9e-09 | 1.1e-06 | 6.0e-07 |
| B_pcg | 1e-08 | 161.2 [157.1, 163.4] | 0.1 | 852, 861, 857 | 9.9e-09 | 7.4e-11 | 2.3e-08 | 1.1e-08 |
| C_gmg | 1e-08 | 280.3 [267.9, 296.2] | 165.8 | 26, 25, 22 | 7.4e-09 | 2.4e-09 | 2.3e-07 | 6.2e-08 |
| C_mg_pcg | 1e-08 | 228.6 [222.1, 236.5] | 173.9 | 11, 11, 10 | 6.8e-09 | 5.5e-11 | 8.8e-09 | 3.6e-09 |
| D_sa | 1e-08 | 128.1 [115.2, 137.0] | 47.6 | 30, 28, 20 | 8.8e-09 | 6.3e-10 | 2.3e-07 | 2.3e-08 |
| D_sa_cg | 1e-08 | 91.2 [87.5, 94.1] | 48.3 | 13, 12, 11 | 4.4e-09 | 1.8e-11 | 1.3e-08 | 1.8e-09 |
| D_sa_jacobi_cg | 1e-08 | 131.2 [126.3, 137.7] | 69.0 | 22, 20, 19 | 5.5e-09 | 4.6e-11 | 4.1e-09 | 5.2e-09 |
| D_rs_cg | 1e-08 | 74.9 [71.3, 79.6] | 36.6 | 9, 10, 11 | 6.2e-09 | 3.5e-11 | 5.3e-09 | 5.2e-09 |

### 7.3 Across the rooms

Total milliseconds per correction (setup and solve), mean over each room's three systems, at three
levels:

| Room | Level | B | C | C MG-PCG | D SA-CG | D RS-CG | E SuperLU | Today's 1e-6 Pa stop |
|---|---|---|---|---|---|---|---|---|
| Product 200x75 | 0.1 | 3.7 | 127.7 | 135.7 | 44.5 | 44.6 | 34.7 | 11627 (median) |
|  | 0.01 | 43.7 | 181.6 | 176.7 | 52.2 | 45.3 | 34.7 |  |
|  | 1e-06 | 154.5 | 243.3 | 202.9 | 78.4 | 63.6 | 34.7 |  |
| Annex 20 180x60 | 0.1 | 1.9 | 102.8 | 100.1 | 38.3 | 33.1 | 42.1 | 219 (median) |
|  | 0.01 | 8.7 | 117.7 | 137.4 | 58.3 | 48.5 | 42.1 |  |
|  | 1e-06 | 96.4 | 154.4 | 142.0 | 71.0 | 67.9 | 42.1 |  |
| Product 40x15 | 0.1 | 0.8 | 29.9 | 31.7 | 8.9 | 6.0 | 1.6 | 432 (median) |
|  | 0.01 | 3.1 | 32.5 | 31.7 | 10.6 | 6.0 | 1.6 |  |
|  | 1e-06 | 8.0 | 37.9 | 33.7 | 12.1 | 8.2 | 1.6 |  |
| VAL-001 80x40 | 0.1 | 6.2 | 65.5 | 63.1 | 21.9 | 18.2 | 12.9 | 520 (median) |
|  | 0.01 | 8.0 | 60.7 | 61.2 | 19.8 | 14.6 | 12.9 |  |
|  | 1e-06 | 15.3 | 93.1 | 83.1 | 29.7 | 24.0 | 12.9 |  |
| VAL-002 80x80 | 0.1 | 11.8 | 82.7 | 84.6 | 28.6 | 24.3 | 26.5 | 366 (median) |
|  | 0.01 | 15.0 | 79.9 | 87.8 | 35.6 | 30.2 | 26.5 |  |
|  | 1e-06 | 34.2 | 115.8 | 117.4 | 48.1 (not reached once) | 40.6 | 26.5 |  |

The ranking moves with the level. At 1e-1 Jacobi-PCG is the fastest on every room, because it has no
setup: on the product and Annex 20 rooms it needs 12 to 26 iterations and is 9 to 17 times faster
than the next candidate, on the small rooms about twice as fast. At 1e-2 it is still the fastest or
level on every room but 40x15, where SuperLU's 1.6 ms leads. At 1e-6 SuperLU and pyamg's two CG
variants lead on the product and Annex 20 rooms, and Jacobi-PCG stays second or close on the small
ones (8 to 34 ms). SuperLU costs one factorization, 1.6 ms on 40x15, 13 on the channel, 27 on the
cavity, 35 on the product and 42 on the Annex 20 room, and returns the exact correction. My NumPy
multigrid is the slowest at every level on every room. Its solve is fast: three or four cycles at
1e-2, and at 1e-6 on the product 41 ms of solve for MG-PCG and 82 for GMG. But its setup, the
Galerkin product formed by probing in Python, costs 120 to 175 ms on the product, two and a half to
five times pyamg's setup. That is the implementation's cost, not the method's; the setup is 25
probes per level, each a full prolongation, product and restriction in NumPy.

Jacobi-PCG spends most of its iterations on the first four decades and few on the next four: on the
product 245 to 313 iterations to 1e-2 at the first two captures, 672 to 703 to 1e-4, 791 to 794 to
1e-6 and 852 to 861 to 1e-8. Going from 1e-4 to 1e-8 costs 1.2 times; every Krylov candidate has the
same shape (RS-CG 58 to 75 ms, SA-CG 65 to 91, MG-PCG 194 to 229).

### 7.4 One relative residual, different errors

The relative residual in the 2-norm is the measure every candidate stopped on (section 2.4), and it
does not mean the same error for each. At 1e-1 on the product, Jacobi-PCG leaves 69% of p' wrong and
a net outflow error of up to 3.5% of the supply; MG-PCG at a similar residual leaves 28% and 0.9%,
the pyamg variants 10% to 17% and 0.2% to 0.5%. Conjugate gradients with a diagonal preconditioner,
like Jacobi itself, removes the rough part of the residual first; the smooth part, which carries
little residual but most of p' and of the net outflow error, is what remains. Multigrid removes
both. Measurement 2 shows the outer loop sees that difference.

### 7.5 Failures

**pyamg's preconditioned CG on the singular cavity.** On the closed cavity, smoothed aggregation
with CG does not reach 1e-6 at outer 1,000 (it stops at 2.0e-4). At 1e-8, smoothed aggregation with
CG and its Jacobi-smoothed variant fail at outers 100 and 1,000, and Ruge-Stuben with CG at 1,000,
stopping between 6.6e-7 and 4.9e-5. The logs carry pyamg's "indefinite preconditioner" and
"indefinite matrix" warnings, 20 to 36 per cavity run, and the histories show the residual reaching
5e-10 and climbing back to 2e-7 (smoothed aggregation, outer 1). The synthetic closed system of
section 3 failed the same way. The right-hand side is projected onto the range as for the other
candidates. The likely cause, not tested: the V-cycle puts a constant into the preconditioned
direction, which the singular matrix does not see, and pyamg's CG aborts on the indefiniteness that
follows. Not tuned (section 2.3). Every candidate converges to every level on every open system, and
the standalone smoothed-aggregation and geometric cycles, Jacobi-PCG and MG-PCG converge to 1e-8 on
the cavity.

### 7.6 Reusing a hierarchy

Built on the previous captured system of the same run and used as the preconditioner of CG on the
current matrix, a hierarchy 99 or 900 outer iterations old needs 6 to 31 times the iterations of a
fresh one. With pyamg's setup at 30 to 50 ms that never pays, on the product or the Annex 20 room.
With my NumPy setup it pays on the product at 1e-1 and 1e-2 (57 to 119 ms stale against 132 to 182
fresh) and not at 1e-6 (434 to 467 against 190 to 208). A lag of a few outer iterations was not
measured.

### 7.7 Against the predictions

| Prediction | Measured |
|---|---|
| Orchestrator: CG takes a few hundred iterations on 200x75 to 1e-6 | Held: 791 to 794 |
| Orchestrator: iterations growing as the cells per side | Held in proportion: 140 to 154 on 40x15 against 791 to 794 on 200x75, five times the cells per side for 5.3 times the iterations |
| Orchestrator: about two orders faster than Jacobi in seconds | Held at equal accuracy: 155 ms against 166 to 764 s at 1e-6, three orders; against today's 1e-6 Pa stop (6 to 79 s, at 1e-3 to 3e-3) 40 to 500 times |
| Orchestrator: MG-PCG and pyamg take tens of iterations and are fastest | Iterations held (7 to 10 for MG-PCG, SA-CG and RS-CG at 1e-6); fastest held for pyamg, not for my NumPy MG-PCG, which is the slowest in total because of its setup; SuperLU is faster than both |
| Orchestrator: pyamg a factor of a few over hand-written NumPy multigrid | Held: 2.6 to 3.2 times at 1e-6 (RS-CG 64 ms, SA-CG 78 against MG-PCG 203) |
| P1: today's stop takes 5,000 to 30,000 sweeps and lands at 1e-4 to 1e-2 | Partly: 27,916, 57,257 and 378,336 sweeps, two above the range; 9.7e-4 to 2.9e-3, inside |
| P2: cap 200 leaves above 0.3 and a net outflow error above half the supply at outer 1 | Missed: 0.029 and 4.8% |
| P3: Jacobi needs more than 150,000 sweeps and 35 s to reach 1e-6 | Held: 762,390 to 3,636,670 sweeps, 166 to 764 s |
| P4: B reaches 1e-6 in 300 to 800 iterations, 0.1 to 0.4 s | Held: 791 to 794, 0.15 to 0.16 s |
| P5: C standalone 15 to 50 cycles, 0.4 to 0.7 per cycle; MG-PCG 10 to 25 iterations; 0.05 to 0.2 s | Partly: 15 to 18 cycles; MG-PCG 8 to 9, below the range; 0.19 to 0.26 s, MG-PCG inside and GMG above |
| P6: SA-CG and RS-CG 8 to 20 iterations, 20 to 80 ms, two to four times faster than MG-PCG | Held: 7 to 10 iterations (RS-CG's 7 just below), 58 to 85 ms (SA-CG's 85 just above), 2.2 to 3.6 times |
| P7: SuperLU 30 to 100 ms, level with pyamg at 1e-6 | Range held (34 to 36 ms); faster than pyamg by about two, not level |
| P8: at 1e-1 B is the fastest NumPy candidate, within a factor of two of pyamg | Fastest held; it is ten times faster than pyamg, not within two of it |

## 8. Measurement 2: how tightly the outer loop needs the correction (written after the runs)

The 40x15 and 80x30 sets started at 09:39:34, after measurement 1, each grid's runs in parallel
beside the untimed Jacobi runs, so their seconds carry that load (`m2_*.json`; `table36.py m2`).
Three things happened that section 2.6 did not plan for, each reported where it falls: the 80x30
room did not converge at real air, nor at ten times its viscosity, the fallback; a field difference
far above the stop's iteration error turned out to be the room's own; and a converged state held a
standing imbalance at the outlets that no relative residual can clear. The diagnostics added for
them are named as added.

### 8.1 The 40x15 room, real air

| Run | Stop | velocity_step at | error_estimate at | Seconds | ms per outer | Inner iterations, median [max] | Relative residual reached, median | Field vs direct at velocity_step (m/s) | its error + direct's | Field vs direct at error_estimate (m/s) | Worst cell at the stop / tol |
|---|---|---|---|---|---|---|---|---|---|---|---|
| direct | error_estimate_and_continuity | 1209 | 2822 | 25 | 9.0 | 1 [1] | 1.2e-14 | - | - | - | 1.2e-09 |
| pcg_3e-1 | diverged | - | - | 0 | 7.9 | 3 [4] | 0.265 | - | - | - | 6.6e+09 |
| pcg_1e-1 | diverged | - | - | 0 | 9.3 | 10 [16] | 0.088 | - | - | - | 3.1e+08 |
| mgpcg_1e-1 | error_estimate_and_continuity | 1210 | 2829 | 164 | 58.1 | 2 [2] | 0.0143 | 3.2e-03 | 4.6e-03 | 3.2e-03 | 1.3e-03 |
| gmg_1e-1 | error_estimate_and_continuity | 1232 | 2923 | 168 | 57.4 | 2 [2] | 0.0368 | 1.2e-03 | 5.0e-03 | 1.2e-03 | 6.1e-03 |
| sacg_1e-1 | diverged | - | - | 2 | 536.6 | 1 [1] | 0.041 | - | - | - | 5.4e+07 |
| pcg_1e-2 | error_estimate_and_continuity | 1205 | 2818 | 34 | 11.9 | 119 [143] | 9.1e-03 | 0.041 | 4.6e-03 | 0.041 | 5.2e-04 |
| mgpcg_1e-2 | error_estimate_and_continuity | 1208 | 2820 | 166 | 58.8 | 3 [4] | 2.3e-03 | 1.8e-03 | 4.6e-03 | 1.8e-03 | 1.2e-04 |
| gmg_1e-2 | error_estimate_and_continuity | 1212 | 2834 | 167 | 59.0 | 3 [6] | 7.9e-03 | 1.7e-03 | 4.6e-03 | 1.8e-03 | 4.8e-04 |
| sacg_1e-2 | error_estimate_and_continuity | 1206 | 2820 | 78 | 27.5 | 3 [3] | 2.7e-03 | 1.8e-03 | 4.6e-03 | 1.8e-03 | 2.2e-04 |
| pcg_1e-4 | error_estimate_and_continuity | 1209 | 2822 | 42 | 15.1 | 143 [154] | 8.3e-05 | 1.4e-04 | 4.6e-03 | 1.4e-04 | 7.1e-06 |
| mgpcg_1e-4 | error_estimate_and_continuity | 1209 | 2822 | 172 | 60.9 | 5 [7] | 5.8e-05 | 4.2e-04 | 4.6e-03 | 4.2e-04 | 1.9e-06 |
| gmg_1e-4 | error_estimate_and_continuity | 1209 | 2822 | 179 | 63.5 | 8 [15] | 4.9e-05 | 2.5e-04 | 4.6e-03 | 2.5e-04 | 2.0e-06 |
| sacg_1e-4 | error_estimate_and_continuity | 1209 | 2822 | 82 | 29.1 | 5 [6] | 1.6e-05 | 1.3e-06 | 4.6e-03 | 1.3e-06 | 1.5e-06 |
| pcg_1e-8 | error_estimate_and_continuity | 1209 | 2822 | 46 | 16.2 | 162 [173] | 2.2e-08 | 2.4e-08 | 4.6e-03 | 2.4e-08 | 1.8e-06 |
| jacobi200 | max_simple_iter | 1684 | - | 184 | 9.2 | 28 [200] | 0.0982 | 0.0124 | 4.4e-03 | - | 232 |
| jacobi40k | max_simple_iter | 1208 | - | 2810 | 140.5 | 29 [40000] | 0.0781 | 1.1e-04 | 4.6e-03 | - | 1.51 |

Every level and solver that converges does so within 5% of direct's outer count at both stops;
today's two loops (below) never meet the error-estimate stop. The multigrid and pyamg repeats ran at
1e-1, 1e-2 and 1e-4, wider than section 2.6's two levels, because by its field criterion only 1e-8
passed (below). The levels differ in three ways.

**1e-1 and 3e-1 diverge with Jacobi-PCG, and 1e-1 with smoothed aggregation.** Jacobi-PCG's first
correction at 1e-1 meets its relative residual in five iterations by removing the spike the supply
puts in the top row of an air field at rest, and leaves the corrected faces with a net outflow error
of -3.29 kg/s, 85% of the 3.888 kg/s supply; the next momentum step passes 400 m/s. At 1e-2 the same
first correction still leaves 55% of the supply unbalanced, and the loop recovers. The two geometric
multigrid runs leave 15% and 16% at the first correction and converge; smoothed aggregation at 1e-1,
which takes one cycle per correction, leaves 4% to 9% and still diverges at outer 2, so the net
outflow alone does not decide it.

**At 1e-2 Jacobi-PCG settles on a different steady state.** It meets both stops within four
iterations of direct, but its field differs from direct's by 0.041 m/s at one cell, (4.7, 1.9), the
jet into the gap beside the litho tool's top corner where step 0 found the room's fastest air; the
median cell differs by 6e-5 m/s. At the error-estimate stop each run's iteration error is below
4.5e-7 m/s, so the two states are 0.041 m/s apart beyond either run's error. The multigrid solvers
at 1e-2 land 1.7e-3 to 1.8e-3 from direct at the same cell, Jacobi-PCG at 1e-4 1.4e-4, at 1e-8
2.4e-8.

**The difference is the room's, not the correction's** (two diagnostics added after the runs):

- *Restart* (`restart36.py`, appendix K). Continued from direct's converged state for 3,000 more
  outer iterations with Jacobi-PCG at 1e-2, the field moves 3.8e-7 m/s; continued from Jacobi-PCG
  1e-2's converged state with direct, it moves 3.9e-7. Each state is a fixed point of the other
  iteration to within the stop's iteration error. So the exact iteration has two fixed points 0.041
  m/s apart at that cell, and the inner tolerance chose between them by changing the path.
- *The path alone* (`control36.py`, appendix L). With the exact correction, nine or eleven momentum
  sweeps, or alpha_velocity 0.45 or 0.55, change only the path to the same steady equations. Their
  converged fields differ from direct's at the same cell by 1.1e-5, 1.6e-6, 4.8e-4 and 1.1e-5 m/s
  (medians 1.5e-8 to 6.6e-7), every one above the stop's iteration error.

| Run | Stop | velocity_step at | error_estimate at | Seconds | ms per outer | Inner iterations, median [max] | Relative residual reached, median | Field vs direct at velocity_step (m/s) | its error + direct's | Field vs direct at error_estimate (m/s) | Worst cell at the stop / tol |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exact, sw9_a0.5 | error_estimate_and_continuity | 1210 | 2822 | 25 | 8.7 | 1 [1] | 1.2e-14 | 1.4e-05 | 4.6e-03 | 1.1e-05 | 1.2e-09 |
| exact, sw11_a0.5 | error_estimate_and_continuity | 1209 | 2822 | 26 | 9.1 | 1 [1] | 1.2e-14 | 2.5e-06 | 4.6e-03 | 1.6e-06 | 1.2e-09 |
| exact, sw10_a0.45 | error_estimate_and_continuity | 1213 | 2875 | 26 | 9.0 | 1 [1] | 1.2e-14 | 1.3e-03 | 4.8e-03 | 4.8e-04 | 1.3e-09 |
| exact, sw10_a0.55 | error_estimate_and_continuity | 1227 | 2712 | 24 | 8.8 | 1 [1] | 1.2e-14 | 5.9e-04 | 5.0e-03 | 1.1e-05 | 1.2e-09 |

So on this grid the steady equations have a nearly neutral direction at one cell, and any change of
path moves where a converged run lands along it: by up to 4.8e-4 m/s for the path changes tried with
the exact correction. That, not the stop's iteration error, is the floor a field comparison can be
held to here. Section 2.6's criterion (ii), the field within the sum of the two runs' own iteration
errors, is passed at the error-estimate stop by Jacobi-PCG at 1e-8 alone, and by none of the exact
correction's own path changes; the path floor replaces it on this grid, a change made after the
runs. Against it, every solver at 1e-4 is indistinguishable from a path change (1.3e-6 to 4.2e-4
m/s); the multigrid solvers at 1e-2 and 1e-1 are 2.5 to 7 times over it (1.2e-3 to 3.2e-3) and
Jacobi-PCG at 1e-2 85 times over it.

*36b.* Section 12.3 reads this differently. The states are separate steady solutions, at least
three, not positions along one nearly neutral direction: the alpha 0.45 state and the CG 1e-2 state,
continued 4,000 outer iterations with the exact correction at alpha 0.5, stay 4.8e-4 and 0.041 m/s
from direct's with the same faces shut. So the "path floor" is a lower bound on how far apart
solutions lie, not a floor on a run's error, and the CG 1e-2 state is a solution, not an error.
After 100 corrections at 1e-8, CG at 1e-1 converges (2,739 outer iterations) and CG at 1e-1 and 1e-2
land 2.7e-4 and 1.0e-4 m/s from direct: the divergence at 1e-1 and the 0.041 m/s state above come
from a loose correction from rest. 3e-1 diverges even after the tight start.

**Today's loop at the committed cap of 200** reaches the velocity-step stop at 1,684, 39% later than
direct, its field there 0.012 m/s from direct's, and never the error-estimate stop in 20,000 outer
iterations: its corrections reach a relative residual of 0.10 in the median and 0.8 late in the run,
and the per-cell continuity condition never holds. **Today's loop at step 0's settings** (cap
40,000, 1e-8 Pa) reproduces test 34b's `A-sw` record to the bit, all 1,208 residuals and sweep
counts, and meets the velocity-step stop at 1,208, its field there 1.1e-4 m/s from direct's, inside
the path floor. It never meets the error-estimate stop in 20,000 outer iterations. Once the
right-hand side is small its stop fires after 24 to 430 sweeps, leaving 5% to 91% of the imbalance
per correction (0.078 in the median over the run), and the worst cell stays between 1.2e-7 and
1.7e-6 kg/s per metre, above the 8e-8 the stop asks. It ran 2,810 s, against direct's 25.

### 8.2 The 80x30 room: no converged laminar solve at real air or at ten times its viscosity

With the exact correction the 80x30 room at real air reaches its least residual, 1.2e-4, at outer
393 and then wanders, 1.3e-3 to 3.5e-3 between the 5th and 95th percentiles of the last 10,000 outer
iterations, the largest speed near 6.5 m/s; at ten times air's viscosity (Re 8,950, the
pre-committed fallback) its least residual is 1.0e-4 at outer 634 and it wanders between 1.1e-3 and
3.2e-3. Neither meets the velocity-step stop. The 40x15 room's convergence at real air (test 34b and
here) does not carry to 80x30. What the 80x30 runs can still show is which levels keep direct's path
and which leave it:

| Run | Viscosity | Stop | Outer iterations run | Least residual (at outer) | Residual, 5th to 95th percentile of the last 1,000 run | Largest speed at the end (m/s) | First correction's net outflow error / supply |
|---|---|---|---|---|---|---|---|
| direct | air | cap | 20,000 | 1.2e-04 (393) | 1.3e-03 to 3.6e-03 | 6.52 | 0.00 |
| pcg_3e-1 | air | diverged | 2 | 2.8e-01 (0) | - | 355 | -0.86 |
| pcg_1e-1 | air | diverged | 3 | 2.0e-01 (0) | - | 8.31e+03 | -0.86 |
| mgpcg_1e-1 | air | cap | 3,000 | 2.0e-04 (1427) | 1.3e-03 to 4.4e-03 | 6.51 | -0.27 |
| gmg_1e-1 | air | diverged | 3 | 1.8e-01 (0) | - | 331 | -0.30 |
| sacg_1e-1 | air | diverged | 29 | 3.2e-02 (18) | - | 770 | -0.07 |
| pcg_1e-2 | air | cap | 20,000 | 1.2e-04 (409) | 1.2e-03 to 3.5e-03 | 6.49 | -0.86 |
| mgpcg_1e-2 | air | cap | 3,000 | 1.2e-04 (395) | 1.3e-03 to 3.5e-03 | 6.51 | -0.05 |
| gmg_1e-2 | air | cap | 3,000 | 1.1e-04 (564) | 1.3e-03 to 3.5e-03 | 6.5 | -0.03 |
| sacg_1e-2 | air | cap | 3,000 | 1.2e-04 (394) | 1.3e-03 to 3.5e-03 | 6.48 | -0.02 |
| pcg_1e-4 | air | cap | 20,000 | 1.2e-04 (393) | 1.2e-03 to 3.5e-03 | 6.49 | -0.00 |
| pcg_1e-8 | air | cap | 20,000 | 1.2e-04 (393) | 1.3e-03 to 3.5e-03 | 6.5 | 0.00 |
| jacobi200 | air | cap; velocity_step at 5232 | 20,000 | 5.6e-17 (17120) | 5.6e-17 to 1.1e-16 | 9.47 | -0.85 |
| direct | 10 x air | cap | 20,000 | 1.0e-04 (634) | 1.1e-03 to 3.2e-03 | 6.49 | -0.00 |
| pcg_3e-1 | 10 x air | diverged | 6 | 3.3e-02 (0) | - | 157 | -0.86 |
| pcg_1e-1 | 10 x air | cap | 20,000 | 1.8e-04 (1096) | 1.1e-03 to 3.7e-03 | 6.46 | -0.86 |
| pcg_1e-2 | 10 x air | cap | 20,000 | 1.0e-04 (512) | 1.1e-03 to 3.2e-03 | 6.48 | 0.01 |
| pcg_1e-4 | 10 x air | cap | 20,000 | 1.0e-04 (634) | 1.1e-03 to 3.2e-03 | 6.49 | -0.00 |
| pcg_1e-8 | 10 x air | cap | 20,000 | 1.0e-04 (634) | 1.1e-03 to 3.2e-03 | 6.48 | -0.00 |
| jacobi200 | 10 x air | cap; velocity_step at 5191 | 20,000 | 5.6e-17 (17251) | 5.6e-17 to 1.1e-16 | 9.31 | -0.86 |

At real air, 1e-1 diverges with Jacobi-PCG and standalone geometric multigrid by outer 3 and with
smoothed aggregation by outer 29; only MG-PCG at 1e-1 follows direct's course. At 1e-2 every solver
follows it: least residual 1.1e-4 to 1.2e-4 at outer 393 to 564, the same wandering after. At ten
times the viscosity Jacobi-PCG at 1e-1 survives but bottoms out later (1.8e-4 at 1,096) and 3e-1
diverges. The 200x75 product room with the exact correction does the same as 80x30 at real air
(least residual 3.5e-4 at outer 150, then 1.6e-3 to 4.9e-3 between the 5th and 95th percentiles of
the last 10,000 outer iterations; section 9).

**Today's loop at the committed cap of 200 freezes without continuity.** On 80x30 at real air it
meets the velocity-step stop at outer 5,232, and at ten times the viscosity at 5,191, with the
corrected faces carrying a net imbalance of 46.5% and 44.2% of the supply (36b: 44.2% corrects
44.5%, the second run's value at its end); the velocity then stays
fixed to rounding (its change 1e-16) for the rest of 20,000 outer iterations. Captured at outer
5,231 (`freeze36.py`, appendix M, a diagnostic added after the runs) the system is not singular: T3
has shut returns 1 to 3 and the hood is held, leaving return 4's six faces open; Cholesky succeeds;
`lambda_min(D^-1 A)` is 3.2e-6. The right-hand side is spread over the whole room almost exactly in
proportion to the diagonal: `b / a_P` lies within 1.3% of its median over 90% of the cells. Weighted
Jacobi turns a residual proportional to its own diagonal into a uniform p', which moves no face, so
the velocity stands still while the pressure drifts and 1.77 kg/s per metre stays unaccounted. The
velocity-step rule calls that converged. The error-estimate rule does not.

*36b.* This is the probe room: T3 outlets, ten momentum sweeps, alpha_velocity 0.5. The committed
configuration (T0, one sweep, alpha_velocity 0.7) does not freeze at the cap of 200: it goes
non-finite at outer 7,635 on 80x30 and never meets the velocity-step stop on 200x75, every
correction at the cap (section 12.5).

### 8.3 The 80x30 room at a thousand times air's viscosity (added)

With the fallback spent, one rung was added so that 80x30 gives a converged case: a thousand times
air's viscosity (Re 90), where the room's residual falls steadily. With the exact correction it
meets the velocity-step stop at 233 and the error-estimate stop at 588.

| Run | Stop | velocity_step at | error_estimate at | Seconds | ms per outer | Inner iterations, median [max] | Relative residual reached, median | Field vs direct at velocity_step (m/s) | its error + direct's | Field vs direct at error_estimate (m/s) | Worst cell at the stop / tol |
|---|---|---|---|---|---|---|---|---|---|---|---|
| direct_mu1000 | error_estimate_and_continuity | 233 | 588 | 11 | 19.4 | 1 [1] | 7.8e-15 | - | - | - | 3.1e-09 |
| pcg_3e-1_mu1000 | max_simple_iter | - | - | 186 | 9.3 | 12 [102] | 0.29 | - | - | - | 2.4e+05 |
| pcg_1e-1_mu1000 | max_simple_iter | - | - | 224 | 11.2 | 30 [141] | 0.0964 | - | - | - | 5.7e+04 |
| mgpcg_1e-1_mu1000 | max_simple_iter | 233 | - | 1742 | 87.1 | 2 [2] | 7.6e-03 | 2.4e-03 | 2.7e-03 | - | 6.5e+03 |
| gmg_1e-1_mu1000 | max_simple_iter | - | - | 1716 | 85.8 | 2 [3] | 0.0416 | - | - | - | 7.9e+04 |
| sacg_1e-1_mu1000 | max_simple_iter | 230 | - | 601 | 30.1 | 1 [2] | 0.0385 | 5.9e-03 | 2.6e-03 | - | 4.2e+04 |
| pcg_1e-2_mu1000 | max_simple_iter | 234 | - | 405 | 20.3 | 152 [179] | 9.2e-03 | 1.6e-03 | 2.7e-03 | - | 3.49e+03 |
| mgpcg_1e-2_mu1000 | max_simple_iter | 233 | - | 1736 | 86.8 | 2 [3] | 7.6e-03 | 2.4e-03 | 2.7e-03 | - | 6.5e+03 |
| gmg_1e-2_mu1000 | max_simple_iter | 233 | - | 1808 | 90.4 | 4 [5] | 3.6e-03 | 9.2e-04 | 2.7e-03 | - | 3.45e+03 |
| sacg_1e-2_mu1000 | max_simple_iter | 233 | - | 641 | 32.1 | 3 [4] | 1.4e-03 | 1.2e-04 | 2.7e-03 | - | 992 |
| pcg_1e-4_mu1000 | max_simple_iter | 233 | - | 515 | 25.7 | 228 [228] | 9.8e-05 | 8.8e-06 | 2.7e-03 | - | 36.3 |
| pcg_1e-8_mu1000 | error_estimate_and_continuity | 233 | 588 | 21 | 36.4 | 303 [303] | 7.5e-09 | 5.8e-10 | 2.7e-03 | 5.7e-10 | 2.5e-03 |
| jacobi200_mu1000 | max_simple_iter | 289 | - | 576 | 28.8 | 200 [200] | 0.312 | 0.386 | 3.1e-03 | - | 4.4e+04 |

**No relative level from 3e-1 to 1e-4 meets the error-estimate stop in 20,000 outer iterations; 1e-8
meets it with direct's counts, 233 and 588.** At 1e-2 and 1e-4 every solver meets the velocity-step
stop at 230 to 234, as direct does at 233, and then the velocity all but stops changing (between
1e-7 and 1e-17 per outer iteration) while the corrected faces keep a worst-cell imbalance of 2e-5 to
1.3e-4 kg/s per metre at 1e-2 and 7e-7 at 1e-4, 36 to 6,500 times the 2e-8 the stop asks. At 1e-1
Jacobi-PCG and standalone multigrid never meet even the velocity-step stop; MG-PCG and smoothed
aggregation meet it (at 233 and 230) and then hold an imbalance as at 1e-2. Today's committed cap of
200 meets the velocity-step stop at 289 and freezes with 20% of the supply unaccounted.

**The cause is a standing imbalance at the outlets**, in direct's run as much as in Jacobi-PCG's
(`stall36.py`, appendix N, a diagnostic added after the runs). At the end of both runs the
right-hand side sits entirely on the cells beside the open return faces (100% of its square) and is
exactly anti-proportional to those cells' outlet coefficients (correlation -1.0000), and the room's
mean pressure rises by 0.0882 Pa every outer iteration in both. The committed outlet treatment
explains it. Before each prediction every open outlet face takes its interior neighbour's velocity,
and if the steady flow carries air sideways into an outlet cell, that zero-gradient value cannot
close the cell. The correction closes it with a uniform p' of 0.29 Pa, which moves only the outlet
faces, by `d p'`. The next extrapolation discards that move, and the same b returns. The interior
velocity is steady and the pressure climbs at a steady rate, 20,000 outer iterations long. With an
exact correction the corrected faces close to rounding, so the error-estimate rule's continuity
conditions hold and it stops; with a relative residual r a fraction r of that standing b stays in
the corrected faces every iteration, so the per-cell condition can never hold unless `r ||b||` is
below `mass_imbalance_tol`. Here `||b||` is 0.079, so a level of about 2.5e-7 is enough (the worst
cell is at most the 2-norm). The 40x15 room at real air has no such state: at its stop b is 1.6e-8
and the pressure moves 1e-8 Pa per outer iteration. The drifting pressure is the outlet treatment's
property, not the correction's, since it appears with the exact correction, and it is outside this
change; what belongs to this change is that a relative stop alone cannot satisfy the stopping rule
there.

*36b.* Measured here under T3 only, where the hood holds its flow. Under T0, the committed outlets,
the same state forms: the exact correction and CG at 1e-8 stop by the error estimate at 1,197 with
the pressure rising 0.0587 Pa per outer iteration and `||b||` 0.069, and CG at 1e-2 and 1e-4 never
stop in 20,000 outer iterations (section 12.4).

### 8.4 What measurement 2 says

The loosest per-correction accuracy depends on what has to stay unchanged:

| What must not change | Loosest level measured that keeps it | Where |
|---|---|---|
| The outer count, every solver | 1e-2 | 40x15 (both stops), 80x30 (the path to its stall), Re 90 (the velocity-step stop) |
| The outer count, multigrid only | 1e-1 (MG-PCG on both grids; GMG on 40x15 only) | 40x15, 80x30 |
| The converged field, to the floor the path itself sets (4.8e-4 m/s) | 1e-4 for every solver from rest; 1e-2 for none from rest; 1e-1 and 1e-2 with CG after 100 corrections at 1e-8 (36b, section 12.3) | 40x15 |
| The stopping rule's continuity conditions where the outlets hold a standing imbalance | an absolute bound: each correction must leave less than `mass_imbalance_tol` per cell; 1e-8 here, 1e-4 does not | 80x30 at Re 90 |

The relative residual in the 2-norm is not a solver-independent measure of enough: at the same level
the multigrid solvers survive 1e-1 where Jacobi-PCG and smoothed aggregation do not, and land about
23 times closer to direct at 1e-2, because they also remove the smooth error.

**Against the premise.** The premise was that most of today's per-correction accuracy is wasted,
since the outer loop needs no more than a few hundred sweeps' worth. For the outer count it holds:
1e-2 keeps it, and today's 1e-6 Pa stop delivers 1e-3 to 3e-3 on the product's systems at 6 to 79 s.
For anything the stop or a reader checks it does not: a loose level moves the converged field beyond
the path floor, and where the outlets hold a standing imbalance the stopping rule needs an absolute
bound that no relative level and not today's stop delivers. **This contradicts the premise in the
sense the prompt's stop condition names: the outer loop needs tighter pressure than today's for the
converged field and for the stopping rule's continuity, not looser.** It changes the emphasis of
ECR-003 from loosening the correction to solving it fast to a tight, absolutely bounded tolerance,
which is cheap: from 1e-4 to 1e-8 Jacobi-PCG's cost rises 1.2 times on the product (section 7.3).

*36b.* The bold sentence holds for the stopping rule's continuity, and for the field only with CG
and multigrid at fixed relative levels from rest. Today's loop at step 0's settings (jacobi40k, cap
40,000, 1e-8 Pa) leaves 7.8% of the imbalance per correction in the median, yet its field is 1.1e-4
m/s from direct's at its velocity-step stop and 7.2e-5 m/s at the end of its 20,000 outer
iterations, inside the 4.8e-4 m/s spread; it fails on continuity, its worst cell ending at 1.2e-7
kg/s per metre against the 8e-8 asked. So for the field the relative level is not the quantity that
decides; section 12.3 shows the start from rest is: after a tight start CG at 1e-1 lands 2.7e-4 m/s
from direct.

### 8.5 Against the predictions

| Prediction | Measured |
|---|---|
| Orchestrator: a relative residual of about 1e-2 leaves the outer count and the field unchanged | The count held at 1e-2 with every solver (40x15 at both stops, the path on 80x30, the velocity-step stop at Re 90). The field missed: Jacobi-PCG at 1e-2 lands 0.041 m/s from the exact correction's state at one cell, 85 times the path floor, and the multigrid solvers and pyamg 1.8e-3, 3.7 times it; at Re 90 no level looser than 1e-8 meets the stopping rule |
| Orchestrator: so most of today's cost is wasted accuracy | Missed, in section 8.4's sense: today's 1e-6 Pa stop delivers 1e-3 to 3e-3 on the product's systems, looser than the field and the stopping rule need. Today's cost is Jacobi's rate, not accuracy beyond need |
| Q1: jacobi40k on 40x15 reproduces A-sw bitwise and meets the velocity-step stop at 1,208 | Held: all 1,208 residuals and sweep counts bitwise; the velocity-step stop at 1,208 |
| Q2: direct on 40x15 meets the velocity-step stop within 10% of 1,208, the error-estimate stop at 1.5 to 3 times that | Held: 1,209 and 2,822 (2.3 times) |
| Q3: 1e-1 is the loosest sufficient level on both grids; 1e-2 and tighter indistinguishable from direct; 3e-1 converges, its count more than 5% off | Missed: 1e-1 and 3e-1 diverge with Jacobi-PCG on both grids; 1e-2 settles on a different steady state on 40x15 |
| Q4: at the error-estimate stop every converged run carries one field within twice the stop's iteration error, since the fixed point does not depend on the inner tolerance | Missed twice. Converged runs at 1e-2 and 1e-4 land 0.041 and 1.4e-4 m/s from direct: the room has nearby steady states and the path picks one (restart test). And at Re 90 the relative levels never reach the stop, because b at the converged state is not zero but stands on the outlet cells |
| Q5: jacobi200 converges on 40x15 within 25% of direct's count, and not on 80x30 | 40x15 missed: velocity-step stop at 1,684 (39% later), error-estimate stop never. 80x30 held in substance: it freezes with 47% of the supply unaccounted, though the velocity-step rule reports convergence at 5,232 |
| Q6: today's loop meets the error-estimate stop later than the relative-residual runs, by more than 10% where both converge, because its stop falls to one sweep once b is small | jacobi200 never meets it on 40x15. The fall to one to three sweeps is measured on the channel and cavity systems (section 7.1). jacobi40k never meets it in 20,000 outer iterations either: once b is small its stop fires after 24 to 430 sweeps and leaves 5% to 91% of it. Held, with 24 sweeps in place of one |

## 9. Measurement 3: one steady product solve (written after the runs)

**The outer count is an assumption.** With the exact correction the laminar 200x75 room does not
converge in 20,000 outer iterations: its residual reaches 3.5e-4 at outer 150 and then wanders,
1.6e-3 to 4.9e-3 between the 5th and 95th percentiles of the last 10,000, the largest speed between
2.7 and 3.0 m/s, nothing diverging. Neither does 80x30 at real air or at ten times its viscosity
(section 8.2). The converged cases here are 40x15 at real air (1,209 outer iterations to the
velocity-step stop, 2,822 to the error-estimate stop) and 80x30 at Re 90 (233 and 588). The
product's steady solve will be the k-epsilon one, and its outer count is ECR-002 step 5's
measurement. So the projection takes ADR-012 H's range, 3,000 to 13,000 outer iterations: the 40x15
room's error-estimate count rounded up, and VAL-002 80x80's 12,849, the slowest validated case. It
gives 6,000 between them and the cost per 1,000 outer iterations, which needs no assumption.

**Per outer iteration on 200x75**: the momentum stage with ten sweeps, 13.1 ms, timed over 30 outer
iterations in one process; the corrector's other work (the coefficients, the right-hand side, the
face and pressure update), 2.9 ms; and the correction, the median over the three captured systems
(`m3_36.py`, appendix O). At the level measurement 2 asks for, 1e-8:

| Correction | Iterations (outer 1, 100, 1000) | Correction ms | ms per outer iteration | Per 1,000 outer | 3,000 outer | 6,000 | 13,000 |
|---|---|---|---|---|---|---|---|
| B, Jacobi-PCG | 852, 861, 857 | 163.2 | 179.2 | 3.0 min | 9.0 min | 17.9 min | 38.8 min |
| C, GMG V(2,2) | 26, 25, 22 | 276.7 | 292.8 | 4.9 min | 14.6 min | 29.3 min | 63.4 min |
| C, MG-PCG | 11, 11, 10 | 227.2 | 243.2 | 4.1 min | 12.2 min | 24.3 min | 52.7 min |
| D, SA | 30, 28, 20 | 132.1 | 148.2 | 2.5 min | 7.4 min | 14.8 min | 32.1 min |
| D, SA-CG | 13, 12, 11 | 92.0 | 108.1 | 1.8 min | 5.4 min | 10.8 min | 23.4 min |
| D, SA-Jacobi-CG | 22, 20, 19 | 129.5 | 145.5 | 2.4 min | 7.3 min | 14.6 min | 31.5 min |
| D, RS-CG | 9, 10, 11 | 73.8 | 89.8 | 1.5 min | 4.5 min | 9.0 min | 19.5 min |
| E, SuperLU (exact) | 1, 1, 1 | 34.3 | 50.4 | 0.8 min | 2.5 min | 5.0 min | 10.9 min |
| A, today's 1e-6 Pa stop | 378,336, 57,257, 27,916 | 11,627.1 | 11,643.2 | 3.2 h | 9.7 h | 19.4 h | 42.0 h |
| A, today's cap 200 (1e-8 not reached) | 200, 200, 200 | 40.3 | 56.3 | 0.9 min | 2.8 min | 5.6 min | 12.2 min |

Today's loop at its 1e-6 Pa stop is the median of its three corrections, 11.6 s (28,000 to 378,000
sweeps); it delivers 1e-3 to 3e-3, not 1e-8. The committed cap of 200 is cheap and is in the table
for the cost alone: section 8.2 shows what it delivers.

**Checked in a run.** The 200x75 room driven for 1,000 outer iterations with each correction (three
runs in parallel beside other probes, `outer36.py`): SuperLU 72 ms per outer iteration against the
projection's 50, Ruge-Stuben with CG at 1e-8 106 against 90, Jacobi-PCG at 1e-8 196 against 179. The
in-run figures carry the parallel load and the harness's own records, 9% to 44% over the projection.
SuperLU's run reproduces the capture run's residuals to the bit; the two 1e-8 runs follow them
within 5% in residual and end at the same largest speed.

**The Annex 20 room on 180x60**, the same way (momentum 10.9 ms, corrector 2.2 ms per outer
iteration; its own three captured systems):

| Correction | Iterations (outer 1, 100, 1000) | Correction ms | ms per outer iteration | Per 1,000 outer | 3,000 outer | 6,000 | 13,000 |
|---|---|---|---|---|---|---|---|
| B, Jacobi-PCG | 672, 713, 737 | 106.3 | 119.4 | 2.0 min | 6.0 min | 11.9 min | 25.9 min |
| C, GMG V(2,2) | 13, 16, 17 | 171.0 | 184.1 | 3.1 min | 9.2 min | 18.4 min | 39.9 min |
| C, MG-PCG | 8, 8, 8 | 158.2 | 171.3 | 2.9 min | 8.6 min | 17.1 min | 37.1 min |
| D, SA | 16, 20, 19 | 98.9 | 112.0 | 1.9 min | 5.6 min | 11.2 min | 24.3 min |
| D, SA-CG | 9, 11, 10 | 83.2 | 96.3 | 1.6 min | 4.8 min | 9.6 min | 20.9 min |
| D, SA-Jacobi-CG | 15, 19, 16 | 119.3 | 132.4 | 2.2 min | 6.6 min | 13.2 min | 28.7 min |
| D, RS-CG | 8, 11, 11 | 70.0 | 83.1 | 1.4 min | 4.2 min | 8.3 min | 18.0 min |
| E, SuperLU (exact) | 1, 1, 1 | 40.0 | 53.1 | 0.9 min | 2.7 min | 5.3 min | 11.5 min |
| A, today's 1e-6 Pa stop | 1,816, 903, 1,448 | 219.2 | 232.3 | 3.9 min | 11.6 min | 23.2 min | 50.3 min |
| A, today's cap 200 (1e-8 not reached) | 200, 200, 200 | 30.3 | 43.3 | 0.7 min | 2.2 min | 4.3 min | 9.4 min |

**Reading.** At the accuracy measurement 2 asks for, every candidate takes a steady product solve
from 10 to 42 hours to minutes: SuperLU 2.5 to 11 minutes, Ruge-Stuben with CG 4.5 to 20, smoothed
aggregation with CG 5.4 to 23, Jacobi-PCG 9 to 39, my NumPy multigrid 12 to 63 (its setup). The
correction is most of every outer iteration: 68% of it with SuperLU, 82% to 95% with the others. On
the Annex 20 room today's stop costs 0.2 s, not 11.6, because there it fires after 900 to 1,800
sweeps at a relative residual of 8e-4 to 5e-3 (section 7.1): cheap because loose. Loosening every
candidate to 1e-2 would save 2.7 times with Jacobi-PCG, 1.5 to 2.2 times with pyamg and nothing with
SuperLU (`m3_product200_0.01.json`), against the risks section 8 measured.

| Prediction | Measured |
|---|---|
| Orchestrator: a steady product solve drops from hours to minutes with any of B, C or D | Held, at 1e-8: 4.5 to 63 minutes for B, C and D over the outer range, 10 to 42 hours today |
| Orchestrator: the ranking then turns on dependencies and the GPU path more than on speed | Held in part: B, C and D span a factor of 3.3 per outer iteration (Ruge-Stuben with CG to my NumPy multigrid, which is setup-bound); SuperLU, not a candidate, is a further 1.8 times faster than the fastest of them |
| R1: the 200x75 laminar room does not converge with the exact correction in 20,000 outer iterations, without passing 100 m/s | Held |
| R2: momentum 30 to 80 ms per outer iteration; B at 1e-1 10 to 50 ms; 0.05 to 0.15 s per outer; 3 to 40 minutes with B, C or D; 3 to 30 hours today | Momentum missed (13.1 ms, below the range); B at 1e-1 missed (3.7 ms, and 1e-1 is not the level that holds); at 1e-8, 0.09 to 0.29 s per outer and 4.5 to 63 minutes, above both ranges at their top; today 10 to 42 hours, above the range |
| R3: B, C and D within a factor of three of each other per steady solve | Missed narrowly: 3.3 times between Ruge-Stuben with CG and GMG at 1e-8 on the product |

## 10. Stop-and-report items, and what contradicts the cited reports

The prompt names four conditions to stop and report on. Item 0 did not trigger one. Two others did,
and the work went on past them because neither changes what the remaining measurements could
measure; both are carried into ECR-003 and ADR-013 as they stand.

1. **A candidate does not converge on a captured system.** pyamg's preconditioned CG on the singular
   VAL-002 cavity: smoothed aggregation with CG misses 1e-6 at outer 1,000, and at 1e-8 smoothed
   aggregation with CG and with Jacobi smoothing miss at outers 100 and 1,000 and Ruge-Stuben with
   CG at 1,000, aborting on pyamg's indefiniteness checks (section 7.5). Every candidate converges
   on every open system. Not tuned.
2. **Measurement 2 finds the outer loop needs tighter pressure than today's.** For the outer count a
   relative residual of 1e-2 is enough with every solver. For the converged field, held to the floor
   the path itself sets, 1e-4 is needed. Where the outlets hold a standing imbalance, the stopping
   rule's continuity conditions need an absolute bound that only 1e-8 met. Today's 1e-6 Pa stop
   delivers 1e-3 to 3e-3 on the product's systems (sections 8.4, 7.1). The ECR's emphasis moves from
   loosening the correction to solving it fast and tight.
   *36b.* Which leg fails depends on the solver. CG and multigrid at 1e-2 from rest keep the count and
   miss the field; today's loop at step 0's settings lands its field inside the spread and misses
   continuity. The field leg is a start-from-rest effect: after 100 tight corrections CG at 1e-1 and
   1e-2 land within the spread, and the "floor" is a lower bound on the spread of several steady
   solutions (sections 8.4 and 12.3). The continuity leg, at Re 90 on 80x30, holds under the
   committed outlets too (section 12.4).

**Against the reports cited in the prompt:**

- `docs/reports/product_case_reynolds.md` section 5 and ADR-012 H. The 27,408 and 112,519 sweeps are
  reproduced to the sweep on their own system. On the systems the T3 room with ten momentum sweeps
  builds, the committed tolerance takes 28,000 to 378,000 sweeps, median 57,000. ADR-012 H took a
  third of the first correction's count per correction, about 9,100 sweeps, for "about 3 hours" over
  4,000 outer iterations at 1e-6 Pa; at the measured median correction (11.6 s) the same 4,000 take
  about 13 hours. ADR-012 H called its figure a lower bound, so this quantifies it rather than
  contradicting it.
- `docs/reports/pressure_solver_probe.md`: the collocated outer count did not depend on the sweep
  cap. On the staggered solver it does at the committed cap: with 200 sweeps the 40x15 room never
  meets the error-estimate stop and the 80x30 room freezes with 47% of the supply unaccounted
  (section 8.2). Above 1e-2 the count does not depend on the level. The collocated system was
  singular and inconsistent, which the staggered system is not, so this is a different regime, not
  an error in that report.
- Step 0's report section 6.4: raising the cap from 40,000 to 400,000 left the 40x15 history
  unchanged. Consistent: both caps there solve far below 1e-2.
- Step 0's report section 7.6 and test 34b: the laminar 40x15 room converges with ten momentum
  sweeps at real air. Reproduced: 1,209 outer iterations to the velocity-step stop with the exact
  correction, and today's loop at step 0's settings reproduces test 34b's `A-sw` record bitwise to
  its stop at 1,208. **It does not carry to finer grids**: with ten sweeps and the exact correction
  the laminar room converges neither on 80x30 (at real air or at ten times its viscosity) nor on
  200x75, over 20,000 outer iterations each. `docs/STATUS.md` and ECR-002 section 2 state the 40x15
  result as "on that grid"; this measurement does not contradict them, and it bounds them. It bears
  on ECR-002 step 5, whose question is the product mesh.

## 11. What this does not settle

- **The product's outer count.** No laminar solve converged on 80x30 or 200x75. The k-epsilon
  solve's count is ECR-002 step 5's; measurement 3 is a projection over an assumed range.
- **The near-neutral direction at (4.7, 1.9).** Measured on the laminar 40x15 room only, the one
  converged real-air case. Whether the product's turbulent solve has such directions, and how large
  the floor they set is, is not measured. *36b:* they are at least three separate steady solutions,
  not one direction (section 12.3); what makes the room have them at that cell is not found.
- **The drifting pressure at the outlets.** A property of the committed outlet treatment
  (zero-gradient extrapolation before each prediction, p' = 0 at the face in the correction), found
  on the 80x30 room at Re 90 and absent from the 40x15 room at real air. Outside this change;
  ECR-002 step 3 rebuilds the outlets. *36b:* found under T0, the committed outlets, as well as
  under T3 (section 12.4).
- **A combined stop**, a relative level with an absolute floor tied to `mass_imbalance_tol`, was not
  run; only fixed relative levels were. 1e-8 met every condition here. *36b:* a tight start before a
  looser level was run on 40x15 only (section 12.3); the combined stop is still not run, and the
  standing `||b||` at the product's converged state is not measured.
- **My multigrid's setup** is the implementation's cost (25 probes per level in Python); a stencil
  formula or C would change its ranking, not measured.
- **pyamg on singular systems** with a pinned cell instead of a projected right-hand side, not run.
- **Reusing a hierarchy** over a lag of a few outer iterations, not run.
- **The timings** are one core of one machine with NumPy 2.4.4, SciPy 1.18.1 and pyamg 5.3.0; no GPU
  was measured.

## 12. Added in 36b: the controls premise review 36 and test 36 found missing (written after the runs)

Premise review 36 and test 36 measured several things this report had not: CG's growth past
200x75, the Annex 20 room at the grid ADR-012 G plans, a tight start before a loose level, a third
steady state, the summation order of CG's sums, the standing imbalance under the committed outlets,
the committed configuration on the finer grids, and CG with a sparse-matrix product. ECR-003 and
ADR-013 now rest on those results, so the fix pass of prompt 36b ran each again with its own probe,
`results/builder36b/probe36b.py` (appendix Q), launched by `run36b.sh` (appendix R) and read by
`summary36b.py` (appendix S). It imports the builder-36 modules of appendices A, B, D and G
read-only, runs in the probe environment of section 2.1 with BLAS on one thread, and writes only to
`results/builder36b/`. The timed runs (12.1, 12.6) ran one at a time with nothing beside them; the
outer-loop runs ran in parallel afterwards and are untimed. Where the premise review or the test
measured the same thing, its figure is given beside this one.

**The harness is the report's.** Four reruns reproduce saved fields of section 8 bitwise (u, v and
p): the 40x15 room with the exact correction, with CG at 1e-2 and with CG at 1e-8, and the 80x30
room at Re 90 with the exact correction. Their stops are the report's: 1,209 and 2,822, 1,205 and
2,818, 1,209 and 2,822, 233 and 588.

### 12.1 Growth with the mesh, the Annex 20 room at 216x72, and the residual CG stops on

Each system was captured at outer 100 of a run driven with SuperLU (the T3 room of section 8 with
ten momentum sweeps; the Annex 20 room under T1 as in section 2.2) and solved with Jacobi-PCG,
`solvers36.jacobi_pcg`. The milliseconds are seven solves of each system to 1e-8, interleaved across
the four so that a drift in the machine's speed falls on all of them alike (`retime`); the scaling
runs' own medians of three, run one system after another, differ from them by up to 17%.

| Room | Grid | Unknowns | Iterations to 1e-4 | to 1e-6 | to 1e-8 | ms per correction at 1e-8, median [fastest] | True minus recursive relative residual at exit, largest of the three levels |
|---|---|---|---|---|---|---|---|
| Product, T3 | 200x75 | 10,910 | 680 | 794 | 861 | 147 [136] | 2.3e-14 |
| Product, T3 | 400x150 | 43,800 | 1,241 | 1,469 | 1,643 | 1,132 [974] | 1.8e-14 |
| Annex 20, T1 | 180x60 | 10,800 | 548 | 645 | 713 | 97 [85] | 3.1e-15 |
| Annex 20, T1 | 216x72 | 15,552 | 663 | 785 | 872 | 178 [145] | 8.1e-15 |

**Growth per doubling.** From 200x75 to 400x150 the iterations grow 1.91 times and the unknowns 4.01
times, so the work grows 7.7 times; the time grew 7.7 times by the medians and 7.2 by the fastest
solves. ECR-003 and ADR-013 now give one figure, about 7.5 times per correction for each doubling
of the cells per side, in place of the draft's "four times" (ECR) and "eight times, extrapolated"
(ADR). The premise review measured the same iteration counts and 135.5 to 1,016 ms, 7.5 times.

**The Annex 20 room at 216x72**, the grid ADR-012 G plans for VAL-016: 872 iterations to 1e-8, 1.22
times its 180x60 count and level with the product's 861, on 1.43 times the product's unknowns;
0.18 s per correction against the product's 0.15 s. The premise review measured 872 iterations and
151 ms. Section 9's 180x60 count at outer 100, 713, is reproduced.

**The residual CG stops on.** The probe's CG stops on the residual it updates by recursion. At every
level on all four systems the true relative residual at exit, formed from p', is within 2.3e-14 of
the recursive one. The premise review found 9.1e-9 to 9.9e-9 true at 1e-8 on the three captured
200x75 systems.

### 12.2 The order of CG's sums, one iteration more, and the floor on the cavity

One correction on five captured systems (section 2.2) with CG's three sums taken in three orders:
NumPy's `vdot`, the reversed arrays through `np.sum`, and blocks of 256 summed in reverse block
order, a stand-in for a GPU's tree reduction. Each CG stops at its relative level or at the floor of
1e-13 times the flux scale F (rho times the inflow, on the cavity rho times the lid velocity times
the side), as every Krylov run of section 8 did. "One iteration more" is the `vdot` loop run one
iteration past its stop: what a second implementation stopping one iteration later would return.

| System | Level | Iterations, vdot / reversed / blocks | Largest p' (Pa) | p' moved by the order, reversed / blocks (Pa) | Faces moved by the order, reversed / blocks (m/s) | Faces moved by one iteration more (m/s) |
|---|---|---|---|---|---|---|
| product200 outer 1 | 1e-8 | 852 / 852 / 852 | 3.86 | 5.6e-14 / 5.9e-14 | 1.2e-12 / 1.5e-12 | 5.9e-09 |
| product200 outer 100 | 1e-8 | 861 / 861 / 861 | 0.266 | 9.5e-15 / 3.2e-15 | 5.8e-14 / 3.4e-14 | 1.4e-09 |
| product200 outer 1000 | 1e-8 | 857 / 857 / 857 | 0.247 | 2.0e-15 / 2.7e-15 | 4.9e-14 / 4.1e-14 | 8.9e-10 |
| product40 outer 100 | 1e-8 | 164 / 164 / 164 | 0.145 | 7.7e-16 / 8.8e-16 | 4.7e-14 / 3.9e-15 | 4.2e-10 |
| cavity80 outer 100 | 1e-8 | 384 / 384 / 384 (the floor) | 0.0138 | 4.0e-17 / 1.9e-17 | 1.1e-16 / 1.1e-16 | 2.0e-13 |
| product200 outer 1 | 1e-10 | 903 / 903 / 903 | 3.86 | 5.7e-14 / 5.9e-14 | 1.2e-12 / 1.6e-12 | 7.4e-11 |
| product200 outer 100 | 1e-10 | 928 / 928 / 928 | 0.266 | 9.5e-15 / 3.2e-15 | 1.1e-13 / 2.8e-14 | 2.0e-11 |
| product200 outer 1000 | 1e-10 | 925 / 925 / 925 | 0.247 | 2.0e-15 / 2.8e-15 | 5.5e-14 / 3.7e-14 | 9.5e-12 |
| product40 outer 100 | 1e-10 | 174 / 174 / 174 | 0.145 | 7.8e-16 / 8.9e-16 | 4.7e-14 / 7.8e-15 | 8.5e-12 |

In the outer loop, the 40x15 room at real air with every sum of CG reversed (`pcgrev:1e-8`) meets the
velocity-step stop at 1,209 and the error-estimate stop at 2,822, as CG with `vdot` does, and its
final field is within 1.1e-14 m/s of that run's.

**Reading.** The order of the sums alone left the iteration counts unchanged and moved the corrected
faces by at most 1.6e-12 m/s, two orders inside REQ-N03's 1e-10; the converged room moved by 1.1e-14
m/s. Test 36 found 1.4e-12 and 3.9e-14 m/s on outers 1 and 1,000 with its own blocked order; the
premise review 1.2e-12 m/s and 1.1e-14 for the room. One iteration more at 1e-8 moves the faces by
4.2e-10 to 5.9e-9 m/s on the open systems, more than 1e-10; at 1e-10 it stays below (8.5e-12 to
7.4e-11). So the risk to REQ-N03 is not the order of the sums but a stop that fires one iteration
apart, which these runs did not produce and a test can rule out by fixing the count (ADR-013 C). This
was a reordering on one CPU, not a GPU run.

**The floor on the cavity.** At outer 100 of VAL-002 80x80 the floor, not the relative level, ended
the correction: CG stopped at 384 iterations with a relative residual of 1.4e-8, at both levels. On
a closed domain a converged solve's right-hand side shrinks toward rounding, so the floor is part of
the stop and belongs in REQ-S08's text (ECR-003 section 5.1).

### 12.3 A tight start, and how many steady states the 40x15 room has

Every run of section 8 started from rest. These start from rest with CG at 1e-8 for the first 100
(or 1,300) corrections and switch to a looser level (`sched:N:LOOSE`); the room, the stop and the
divergence check are section 8.1's. The field is compared with section 8.1's direct run.

| Correction | velocity_step at | error_estimate at | Inner iterations, median [max] | Field vs direct, largest (at) | Shut return faces |
|---|---|---|---|---|---|
| exact (rerun) | 1,209 | 2,822 | 1 [1] | - | bottom 12 |
| CG 1e-2 from rest (rerun) | 1,205 | 2,818 | 119 [143] | 0.041 (4.7, 1.9) | bottom 12 |
| 100 at 1e-8, then 1e-1 | 1,196 | 2,739 | 35 [173] | 2.7e-4 (4.7, 1.9) | bottom 12 |
| 1,300 at 1e-8, then 1e-1 | 1,209 | 2,799 | 88 [173] | 1.6e-7 (6.1, 0.9) | bottom 12 |
| 100 at 1e-8, then 1e-2 | 1,206 | 2,821 | 120 [173] | 1.0e-4 (4.7, 1.9) | bottom 12 |
| 100 at 1e-8, then 3e-1 | - | diverged at outer 125 | 163 [173] | - | bottom 13 and 14 when it diverged |

From rest, CG at 1e-1 and 3e-1 diverged by outer 2 (section 8.1). After 100 tight corrections 1e-1
converges: its first loose corrections leave a net outflow error of 0.2% to 1% of the supply, where
the first correction from rest left 85%. 3e-1 still diverges, 25 outer iterations after the
switch. The premise review measured the 1e-1 rows (2,739 and 2.7e-4; 2,799 and 1.6e-7) and could
not tell whether its 3e-1 run diverged; this one does. The 1e-2 row is new.

Then each of three converged states was continued 4,000 outer iterations with the exact correction
(`continue`, restart36's loop of appendix K):

| State | Reached by | Stop | Distance from direct's state at its stop | Continued with | Drift from its stop | Last step (m/s) | Distance from direct's state at the end | Shut return faces, stop and end |
|---|---|---|---|---|---|---|---|---|
| D | exact, alpha_velocity 0.5 | 2,822 | - | exact, alpha 0.45 | 3.8e-7 | 1.3e-15 | 3.8e-7 | bottom 12 |
| A | exact, alpha_velocity 0.45 | 2,875 | 4.8e-4 at (4.7, 1.9) | exact, alpha 0.5 | 3.5e-7 | 4.4e-16 | 4.8e-4 at (4.7, 1.9) | bottom 12 |
| P | CG 1e-2, alpha_velocity 0.5 | 2,818 | 0.041 at (4.7, 1.9) | exact, alpha 0.5 | 3.9e-7 | 8.9e-16 | 0.041 at (4.7, 1.9) | bottom 12 |

The open bottom faces are the same twelve at every stop and every end, `[2, 3, 4, 5, 13, 14, 23, 24,
28, 29, 30, 31]`, with none shut on the right wall.

**Reading.** Continued with the exact iteration at alpha 0.5, the states A and P stay where they are:
they drift 3.5e-7 and 3.9e-7 m/s, the stop's own iteration error (test 36 measured 3.79e-7 for the
exact state continued from itself), their step falls to 1e-15 m/s, which is rounding, and the same
faces stay shut. A fixed point of the exact iteration solves the discrete steady equations with those
faces shut: the pressure update stops only where p' is zero, so the corrected faces are the
predicted ones and their imbalance b is zero, and ten momentum sweeps return their starting field
only if it solves the momentum equations. So the 40x15 room has at least three steady solutions,
which differ at (4.7, 1.9) by 4.8e-4 and 0.041 m/s, not one solution with a slow direction.
Section 8.1's "path floor" of 4.8e-4 m/s is the spread of the solutions its path changes happened to
reach: a lower bound on how far apart solutions lie, not a floor on any run's error. CG at 1e-2 from
rest reached another solution as alpha 0.45 did; after a tight start CG at 1e-1 and 1e-2 land within
that spread of the exact correction's solution. So section 8.1's divergence of 1e-1 and its 0.041
m/s state are effects of a loose correction from rest; the divergence of 3e-1 is not. What makes
this room have several solutions at that cell, the jet into the gap beside the litho tool's top
corner, is not found (test 36, S8).

### 12.4 The standing imbalance under the committed outlets

Section 8.3 measured the standing imbalance under T3, where the hood is held at a fixed flow. T0 is
the committed treatment: every outlet face takes its interior neighbour's velocity. The 80x30 room at
a thousand times air's viscosity, otherwise as section 8.3; `mass_imbalance_tol` is 2e-8 kg/s per
metre.

| Outlets | Correction | Stop | velocity_step at | error_estimate at | Worst cell at the end, kg/s per m (over the tolerance) | Mean pressure change per outer iteration, last 100 (Pa) | `||b||` at the end |
|---|---|---|---|---|---|---|---|
| T3 (rerun) | exact | error_estimate | 233 | 588 | 6.3e-17 | +0.0882 | 0.0793 |
| T0 | exact | error_estimate | 489 | 1,197 | 4.4e-17 | +0.0587 | 0.0694 |
| T0 | CG 1e-8 | error_estimate | 489 | 1,197 | 6.2e-11 | +0.0587 | 0.0694 |
| T0 | CG 1e-4 | none in 20,000 | 489 | - | 6.7e-7 (33 times) | +0.0587 | 0.0694 |
| T0 | CG 1e-2 | none in 20,000 | 487 | - | 9.1e-5 (4,500 times) | +0.0587 | 0.0694 |

**Reading.** The committed outlets hold the same kind of state: the exact correction and CG at 1e-8
stop by the error estimate at 1,197 with the mean pressure still rising 0.0587 Pa per outer
iteration, and 1e-4 and 1e-2 never stop, though both meet the velocity-step stop by outer 489,
their worst cell ending 33 and 4,500 times the tolerance. The hood held at its flow under T3 is not
the cause. The level needed is `mass_imbalance_tol / ||b||`, 2.9e-7 under T0 and 2.5e-7 under T3.
The premise review measured the T0 rows (1,197, 0.0587, 0.0694, and worst cells of 9.1e-5 and
6.7e-7); they are reproduced.

### 12.5 The committed configuration on 80x30 and 200x75

`configs/clean_room_default.yaml` with only nx, ny and `max_simple_iter` changed: T0, one momentum
sweep, alpha_velocity 0.7, the committed Jacobi loop (cap 200, 1e-6 Pa), velocity_step.

| Grid | Outer iterations run | Outcome | Least residual (at outer) | Residual over the last 1,000, 5th and 95th percentiles | Corrections that took 200 sweeps | velocity_step met |
|---|---|---|---|---|---|---|
| 80x30 | 7,636 | non-finite at outer 7,635 | 0.042 (15) | 4.6e132, 4.1e150 | all 7,636 | never |
| 200x75 | 3,000 | the cap of 3,000 | 7.9e-3 (11) | 1.4e44, 2.2e63 | all 3,000 | never |

**Reading.** The committed room diverges on both grids. Every correction runs the full 200 sweeps, but
the room never reaches a state the cap could freeze. The freeze of section 8.2, where velocity_step
calls a frozen, unbalanced field converged, belongs to the probe room (T3, ten momentum sweeps,
alpha_velocity 0.5), as ECR-003 section 1 now says. The premise review found the same outcomes
(non-finite from 7,635; 7.9e-3 at outer 11; a 5th percentile of 1.4e44) and gives 98% of the 80x30
corrections at the cap; this count takes every correction that ran 200 sweeps, whether or not its
stop fired at the last one.

### 12.6 CG with a sparse-matrix product

On the three captured 200x75 systems of section 7.2, the probe's Jacobi-PCG, whose product is five
shifted NumPy arrays, against the same loop (`solvers36.pcg`) with SciPy's CSR product on the cells
with an equation; five timed solves of each to 1e-8, alternating, in one process.

| System | Iterations, both | Five shifted arrays, ms (per iteration) | CSR product, ms (per iteration) | Per iteration, shifted over CSR |
|---|---|---|---|---|
| product200 outer 1 | 852 | 134 (0.157) | 87 (0.102) | 1.53 |
| product200 outer 100 | 861 | 140 (0.163) | 96 (0.112) | 1.46 |
| product200 outer 1000 | 857 | 183 (0.213) | 123 (0.144) | 1.48 |

**Reading.** The same iterations take about 1.5 times less with a library sparse product; test 36
found 0.09 to 0.10 s with its own CSR loop, 1.7 times. The 0.16 s of section 7.2 is the probe's
product, not CG's floor. A CSR product needs SciPy (ADR-013 decision 2) or compiled code (decision
5); with it, SuperLU's lead per outer iteration falls from 3.6 times to about two.

### 12.7 What section 12 changes in sections 8 to 11

Sections 1 to 5 stand as committed. Notes marked "36b" in sections 8.1 to 8.4, 10 and 11 point
here, and one figure in section 8.2 is corrected: the 80x30 jacobi200 run at ten times air's
viscosity met the velocity-step stop with 44.2% of the supply unaccounted, and 44.5% is that run's
value at its end (test 36, S2).

## Appendix A: common36.py

```python
"""Builder probe, prompt 36 (ECR-003): rooms, the probe corrector and system capture.

Nothing in src/ is edited or patched. The rooms are built from the committed
configurations; the product room runs under the T3 outlets of
results/builder33b/outlet33b.py and, with ten momentum sweeps, through
FrozenPredictor of results/builder34/frozen34.py with a zero eddy-viscosity
field (step 0, section 2.4: with mu_t = 0 the field path is the committed
predictor to the bit at one sweep). Both files are byte copies of the main
tree's (SHA-256 recorded in the report).

ProbeCorrector subclasses PressureCorrector and replaces only the p' solve:
the coefficients, the right-hand side, the pin, the velocity correction and
the pressure update are the committed lines, copied. With ``solve=None`` it
calls the committed ``correct`` itself. It can save the p' system it is
handed at chosen outer iterations, and it records, per correction, the
inner iterations, the relative residual reached and the norm of b.

The p' equation in the corrector's convention is A p' = -b with
(A x)_P = a_P x_P - sum(a_nb x_nb). The mass imbalance the corrected faces
leave in each cell is b + A p' exactly (the report, section 3.3), so the
residual of the p' equation and the corrected faces' continuity are one
vector read two ways.
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results" / "builder34"))
sys.path.insert(0, str(ROOT / "results" / "builder33b"))

import frozen34  # noqa: E402
from outlet33b import OutletSolver, segment_names  # noqa: E402

from src.boundary_staggered import StaggeredBoundary  # noqa: E402
from src.config import SimConfig  # noqa: E402
from src.mesh import Mesh  # noqa: E402
from src.momentum import MomentumPrediction  # noqa: E402
from src.pressure import (  # noqa: E402
    JACOBI_WEIGHT,
    PressureCoefficients,
    PressureCorrection,
    PressureCorrector,
)
from src.solver_staggered import StaggeredSolver  # noqa: E402
from validation.cases import load_preset, with_velocity_step  # noqa: E402

SYSTEMS = HERE / "systems"
SYSTEMS.mkdir(exist_ok=True)

# A p' solve: (coefficients, b, active, needs_pin) -> (p_prime, inner iterations).
Solve = Callable[[PressureCoefficients, np.ndarray, np.ndarray, bool], tuple[np.ndarray, int]]


def threads_note() -> dict[str, str | None]:
    """The BLAS thread settings a run saw, for the records."""
    keys = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    return {k: os.environ.get(k) for k in keys}


# ---------------------------------------------------------------------------
# The operator, in the corrector's convention
# ---------------------------------------------------------------------------


def apply_a(c: PressureCoefficients, x: np.ndarray) -> np.ndarray:
    """(A x)_P = a_P x_P - sum(a_nb x_nb); the edge coefficients are zero by construction."""
    y = c.a_p * x
    y[:, :-1] -= c.a_e[:, :-1] * x[:, 1:]
    y[:, 1:] -= c.a_w[:, 1:] * x[:, :-1]
    y[:-1, :] -= c.a_n[:-1, :] * x[1:, :]
    y[1:, :] -= c.a_s[1:, :] * x[:-1, :]
    return y


def corrected_imbalance(c: PressureCoefficients, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The mass imbalance the corrected faces leave in each cell: b + A p', kg/s per m."""
    out = b + apply_a(c, x)
    out[c.a_p <= 0.0] = 0.0
    return out


# ---------------------------------------------------------------------------
# The probe corrector
# ---------------------------------------------------------------------------


@dataclass
class CorrectionRecord:
    """What one correction did."""

    inner: int
    rel_residual: float
    b_norm: float
    seconds: float


class ProbeCorrector(PressureCorrector):
    """PressureCorrector with a replaceable p' solve, system capture and per-call records."""

    def __init__(  # type: ignore[no-untyped-def]
        self,
        mesh,
        config,
        boundary,
        solve: Solve | None = None,
        capture_at: tuple[int, ...] = (),
        capture_name: str = "",
        measure_residual: bool = True,
    ) -> None:
        super().__init__(mesh, config, boundary)
        self.solve = solve
        self.capture_at = set(capture_at)
        self.capture_name = capture_name
        self.measure_residual = measure_residual
        self.calls = 0
        self.records: list[CorrectionRecord] = []
        self.meta: dict = {}

    def _capture(self, k: int, prediction: MomentumPrediction, c, b) -> None:  # type: ignore[no-untyped-def]
        d_u, d_v = self._face_d(prediction.a_p_u, prediction.a_p_v)
        np.savez(
            SYSTEMS / f"{self.capture_name}_it{k}.npz",
            a_p=c.a_p, a_e=c.a_e, a_w=c.a_w, a_n=c.a_n, a_s=c.a_s, b=b,
            d_u=d_u, d_v=d_v, u_star=prediction.u_star, v_star=prediction.v_star,
            a_p_u=prediction.a_p_u, a_p_v=prediction.a_p_v,
            out_left=self._out_left, out_right=self._out_right,
            out_bottom=self._out_bottom, out_top=self._out_top,
            needs_pin=np.array(self.needs_pin), pin_cell=np.array(self.pin_cell),
            rho=np.array(self._rho), dx_cell=self._mesh.dx_cell, dy_cell=self._mesh.dy_cell,
            solid=self._solid,
            meta=np.array(json.dumps({**self.meta, "outer": k})),
        )

    def correct(self, prediction: MomentumPrediction, p: np.ndarray) -> PressureCorrection:
        """The committed correct with the p' solve replaced; solve None is the committed call."""
        k = self.calls
        self.calls += 1
        need_system = k in self.capture_at or self.measure_residual
        if need_system:
            c = self.coefficients(prediction.a_p_u, prediction.a_p_v)
            b = self.mass_imbalance(prediction.u_star, prediction.v_star)
            if k in self.capture_at:
                self._capture(k, prediction, c, b)
        if self.solve is None:
            t0 = time.perf_counter()
            result = super().correct(prediction, p)
            seconds = time.perf_counter() - t0
            if self.measure_residual:
                self.records.append(self._record(c, b, result.p_prime, result.sweeps, seconds))
            return result

        # The committed correct, line for line, with the solve replaced.
        u_star, v_star = prediction.u_star, prediction.v_star
        self._check_shapes(u_star, v_star)
        if p.shape != self._p_shape:
            raise ValueError(f"expected p of shape {self._p_shape}, got {p.shape}")
        if not need_system:
            c = self.coefficients(prediction.a_p_u, prediction.a_p_v)
            b = self.mass_imbalance(u_star, v_star)
        active = c.a_p > 0.0
        pin_j, pin_i = self.pin_cell
        t0 = time.perf_counter()
        p_prime, inner = self.solve(c, b, active, self.needs_pin)
        seconds = time.perf_counter() - t0
        if self.needs_pin:
            p_prime[active] -= p_prime[pin_j, pin_i]

        d_u, d_v = self._face_d(prediction.a_p_u, prediction.a_p_v)
        padded = np.zeros((self._p_shape[0] + 2, self._p_shape[1] + 2), dtype=np.float64)
        padded[1:-1, 1:-1] = p_prime
        u = u_star - d_u * (padded[1:-1, 1:] - padded[1:-1, :-1])
        v = v_star - d_v * (padded[1:, 1:-1] - padded[:-1, 1:-1])

        p_next = p.copy()
        p_next[active] += self._alpha_p * p_prime[active]
        if self.needs_pin:
            p_next[active] -= p_next[pin_j, pin_i]
        if self.measure_residual:
            self.records.append(self._record(c, b, p_prime, inner, seconds))
        return PressureCorrection(
            u=np.ascontiguousarray(u),
            v=np.ascontiguousarray(v),
            p=p_next,
            p_prime=p_prime,
            sweeps=inner,
        )

    def _record(self, c, b, p_prime, inner, seconds) -> CorrectionRecord:  # type: ignore[no-untyped-def]
        f = -b.copy()
        active = c.a_p > 0.0
        if self.needs_pin:
            f[active] -= f[active].mean()
        r = f - apply_a(c, p_prime)
        r[~active] = 0.0
        fn = float(np.linalg.norm(f))
        rel = float(np.linalg.norm(r)) / fn if fn > 0.0 else 0.0
        return CorrectionRecord(inner=int(inner), rel_residual=rel, b_norm=float(np.linalg.norm(b)), seconds=seconds)


def committed_jacobi(corrector: PressureCorrector, tol: float, cap: int) -> Solve:
    """Today's solve as a Solve: the committed weighted sweep and stop, copied from correct."""

    def solve(c, b, active, needs_pin):  # type: ignore[no-untyped-def]
        p_prime = np.zeros(c.a_p.shape, dtype=np.float64)
        sweeps = 0
        for _ in range(cap):
            p_new = corrector.sweep(p_prime, c, b, JACOBI_WEIGHT)
            diff = float(np.max(np.abs(p_new[active] - p_prime[active]))) if active.any() else 0.0
            p_prime = p_new
            sweeps += 1
            if diff < tol:
                break
        return p_prime, sweeps

    return solve


# ---------------------------------------------------------------------------
# Rooms
# ---------------------------------------------------------------------------


def product_raw(nx: int, ny: int, mu_factor: float = 1.0) -> dict:
    """configs/clean_room_default.yaml regridded, viscosity scaled."""
    raw = yaml.safe_load((ROOT / "configs/clean_room_default.yaml").read_text())
    raw["domain"]["nx"], raw["domain"]["ny"] = nx, ny
    raw["fluid"]["viscosity"] *= mu_factor
    return raw


def annex20_raw(nx: int, ny: int) -> dict:
    """The Annex 20 room as cost33.py builds it (product_case_reynolds.md, appendix B)."""
    raw = product_raw(nx, ny)
    raw["domain"].update(width=9.0, height=3.0)
    raw["fluid"].update(density=1.2, viscosity=1.2 * 15.3e-6)
    raw["boundaries"] = {
        "slot": {
            "type": "velocity_inlet", "location": "left",
            "y_start": 3.0 - 0.168, "y_end": 3.0, "velocity": 0.455,
        },
        "outlet": {
            "type": "pressure_outlet", "location": "right",
            "y_start": 0.0, "y_end": 0.48,
        },
    }
    raw["obstacles"] = []
    return raw


def set_solver(
    raw: dict,
    n_outer: int,
    alpha_u: float = 0.5,
    p_cap: int = 40000,
    p_tol: float = 1e-8,
    stopping: dict | None = None,
) -> dict:
    """The ladder's solver settings (step 0, section 2.1) unless given; stopping keys added."""
    block = raw["solver"]
    block["max_simple_iter"] = n_outer
    block["alpha_velocity"] = alpha_u
    block["max_pressure_iter"] = p_cap
    block["pressure_tol"] = p_tol
    if stopping:
        block.update(stopping)
    return raw


@dataclass
class Room:
    """A built solver and what the probes need beside it."""

    name: str
    cfg: SimConfig
    mesh: Mesh
    boundary: StaggeredBoundary
    solver: StaggeredSolver
    supply: float  # rho times the inlet volumetric flux, kg/s per m (closed: rho U L)
    meta: dict = field(default_factory=dict)


def _supply(cfg: SimConfig, mesh: Mesh, boundary: StaggeredBoundary) -> float:
    flux = boundary.get_total_inlet_flux()
    if flux > 0.0:
        return cfg.rho * flux
    return cfg.rho * boundary.get_max_boundary_velocity() * max(float(mesh.x[-1]), float(mesh.y[-1]))


def product_t3(
    name: str, nx: int, ny: int, n_outer: int, sweeps: int = 10, mu_factor: float = 1.0,
    p_cap: int = 40000, p_tol: float = 1e-8, alpha_u: float = 0.5, stopping: dict | None = None,
) -> Room:
    """The product room under T3, laminar (zero field), N momentum sweeps per outer iteration."""
    raw = set_solver(product_raw(nx, ny, mu_factor), n_outer, alpha_u, p_cap, p_tol, stopping)
    cfg = SimConfig.from_dict(raw)
    mesh = Mesh(cfg)
    boundary = StaggeredBoundary(mesh, cfg)
    hood = segment_names(mesh, cfg, "right") == "hood_exhaust"
    solver = frozen34.FrozenSolver(
        mesh, cfg, boundary, hood, np.zeros(mesh.cell_type.shape), sweeps=sweeps
    )
    meta = {"case": "product_t3", "nx": nx, "ny": ny, "sweeps": sweeps, "mu_factor": mu_factor,
            "alpha_velocity": alpha_u, "p_cap": p_cap, "p_tol": p_tol, "stopping": stopping or {}}
    return Room(name, cfg, mesh, boundary, solver, _supply(cfg, mesh, boundary), meta)


def product_committed(name: str, nx: int = 200, ny: int = 75, n_outer: int = 1) -> Room:
    """cost33.py's room: the product as committed (T0, one sweep, alpha 0.7, cap 200, 1e-6)."""
    raw = product_raw(nx, ny)
    raw["solver"]["max_simple_iter"] = n_outer
    cfg = SimConfig.from_dict(raw)
    mesh = Mesh(cfg)
    boundary = StaggeredBoundary(mesh, cfg)
    solver = StaggeredSolver(mesh, cfg, boundary)
    meta = {"case": "product_committed", "nx": nx, "ny": ny}
    return Room(name, cfg, mesh, boundary, solver, _supply(cfg, mesh, boundary), meta)


class AnnexSolver(OutletSolver):
    """outlet33b's T1 (backflow held shut) on the Annex 20 outlet, with N momentum sweeps."""

    def __init__(self, mesh, cfg, boundary, sweeps: int) -> None:  # type: ignore[no-untyped-def]
        no_hood = np.zeros(mesh.yc.shape[0], dtype=bool)
        super().__init__(mesh, cfg, boundary, "T1", 0.0, no_hood)
        self._predictor = frozen34.FrozenPredictor(
            mesh, cfg, boundary, np.zeros(mesh.cell_type.shape), sweeps=sweeps
        )


def annex20(name: str, nx: int, ny: int, n_outer: int, sweeps: int = 10) -> Room:
    """The Annex 20 room, laminar, outlet T1, ten momentum sweeps, step 0's settings."""
    raw = set_solver(annex20_raw(nx, ny), n_outer)
    cfg = SimConfig.from_dict(raw)
    mesh = Mesh(cfg)
    boundary = StaggeredBoundary(mesh, cfg)
    solver = AnnexSolver(mesh, cfg, boundary, sweeps)
    meta = {"case": "annex20_t1", "nx": nx, "ny": ny, "sweeps": sweeps}
    return Room(name, cfg, mesh, boundary, solver, _supply(cfg, mesh, boundary), meta)


def validation_case(name: str, preset: str, n_outer: int) -> Room:
    """A VAL-001 or VAL-002 preset as its case file configures it, under velocity_step."""
    cfg = with_velocity_step(load_preset(preset))
    cfg.max_simple_iter = n_outer
    mesh = Mesh(cfg)
    boundary = StaggeredBoundary(mesh, cfg)
    solver = StaggeredSolver(mesh, cfg, boundary)
    meta = {"case": preset, "nx": cfg.nx, "ny": cfg.ny}
    return Room(name, cfg, mesh, boundary, solver, _supply(cfg, mesh, boundary), meta)


def install_corrector(room: Room, solve: Solve | None, **kwargs) -> ProbeCorrector:  # type: ignore[no-untyped-def]
    """Replace the solver's corrector with a ProbeCorrector built from the same inputs."""
    corrector = ProbeCorrector(room.mesh, room.cfg, room.boundary, solve, **kwargs)
    corrector.meta = {"room": room.name, **room.meta, "supply": room.supply}
    room.solver._corrector = corrector
    return corrector


# ---------------------------------------------------------------------------
# Captured systems
# ---------------------------------------------------------------------------


@dataclass
class System:
    """A captured p' system and what evaluating a solution needs."""

    name: str
    c: PressureCoefficients
    b: np.ndarray
    d_u: np.ndarray
    d_v: np.ndarray
    u_star: np.ndarray
    v_star: np.ndarray
    needs_pin: bool
    pin_cell: tuple[int, int]
    rho: float
    dx_cell: np.ndarray
    dy_cell: np.ndarray
    solid: np.ndarray
    meta: dict

    @property
    def active(self) -> np.ndarray:
        return self.c.a_p > 0.0

    @property
    def supply(self) -> float:
        return float(self.meta["supply"])

    def rhs(self) -> np.ndarray:
        """f = -b, projected onto the range on a closed domain (mean over active cells removed)."""
        f = -self.b.copy()
        if self.needs_pin:
            act = self.active
            f[act] -= f[act].mean()
        f[~self.active] = 0.0
        return f


def load_system(name: str) -> System:
    """Load systems/NAME.npz."""
    z = np.load(SYSTEMS / f"{name}.npz")
    c = PressureCoefficients(a_p=z["a_p"], a_e=z["a_e"], a_w=z["a_w"], a_n=z["a_n"], a_s=z["a_s"])
    return System(
        name=name, c=c, b=z["b"], d_u=z["d_u"], d_v=z["d_v"], u_star=z["u_star"],
        v_star=z["v_star"], needs_pin=bool(z["needs_pin"]),
        pin_cell=(int(z["pin_cell"][0]), int(z["pin_cell"][1])), rho=float(z["rho"]),
        dx_cell=z["dx_cell"], dy_cell=z["dy_cell"], solid=z["solid"],
        meta=json.loads(str(z["meta"])),
    )


def corrected_faces(s: System, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """u* - d grad p' with p' = x pinned as the corrector pins it."""
    x = x.copy()
    if s.needs_pin:
        x[s.active] -= x[s.pin_cell]
    ny, nx = x.shape
    padded = np.zeros((ny + 2, nx + 2))
    padded[1:-1, 1:-1] = x
    u = s.u_star - s.d_u * (padded[1:-1, 1:] - padded[1:-1, :-1])
    v = s.v_star - s.d_v * (padded[1:, 1:-1] - padded[:-1, 1:-1])
    return u, v


def face_imbalance(s: System, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """PressureCorrector.mass_imbalance's arithmetic on the captured mesh spacing."""
    out = s.rho * ((u[:, 1:] - u[:, :-1]) * s.dy_cell[:, None] + (v[1:, :] - v[:-1, :]) * s.dx_cell[None, :])
    out[s.solid] = 0.0  # as PressureCorrector.mass_imbalance zeroes them
    return out
```

## Appendix B: solvers36.py

```python
"""Builder probe, prompt 36 (ECR-003): the candidate p' solvers.

Every solver solves A x = f on the cells with an equation (a_P > 0), where
f = -b (projected onto the range on a closed domain, as common36.System.rhs
does), and stops on the same measure: the 2-norm of the residual f - A x
over the 2-norm of f, below rtol. pyamg's own stop is that measure too
(MultilevelSolver.solve: ``normr < tol * normb``).

A. Weighted Jacobi: PressureCorrector.sweep with JACOBI_WEIGHT, today's.
B. Jacobi-preconditioned conjugate gradients, NumPy: one matrix-vector
   product of the five-point stencil and three reductions per iteration (the
   two of CG and the residual norm the stop reads).
C. Geometric multigrid with Galerkin coarse operators, NumPy: cell-centred
   bilinear prolongation P (weights 9/16, 3/16, 3/16, 1/16 from the parent
   coarse cell and its three nearest neighbours, renormalized over the coarse
   cells that exist and have an equation, so P reproduces constants), R = P^T,
   A_c = R A P formed by probing with 25 coloured vectors (the coarse stencil
   is at most 5 x 5, so each colour hits one stencil entry per row). A level
   with an odd dimension is padded by one inactive row or column before it is
   coarsened. Weighted Jacobi smoothing: on the finest level the committed
   weight 2/3, on coarse levels 4 / (3 lambda_max(D^-1 A)) with lambda_max
   from 30 power iterations (a Galerkin operator need not be diagonally
   dominant). V(2,2) cycles; the coarsest level (at most 64 cells, or a side
   of at most 4) solved by a pseudo-inverse. As a preconditioner of CG the
   same V(2,2) from zero, which is symmetric.
D. pyamg (probe environment only): smoothed aggregation and Ruge-Stuben with
   their defaults (Gauss-Seidel smoothing), standalone and as CG
   preconditioners, plus smoothed aggregation with weighted Jacobi smoothing,
   the data-parallel variant. Coarse solve 'pinv' so the singular cavity works.
E. Sparse direct (scipy SuperLU), the reference: the exact solution every
   other is measured against. Not one of the prompt's candidates.
"""

import time
from dataclasses import dataclass

import numpy as np

from common36 import JACOBI_WEIGHT, PressureCoefficients, apply_a

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class SolveResult:
    """What one solve returned and cost."""

    x: np.ndarray
    iterations: int
    setup_s: float
    solve_s: float
    history: list[float]  # relative residual after each iteration, where kept

    @property
    def total_s(self) -> float:
        return self.setup_s + self.solve_s


def rel_residual(c: PressureCoefficients, f: np.ndarray, x: np.ndarray) -> float:
    """||f - A x||_2 / ||f||_2 over cells with an equation."""
    r = f - apply_a(c, x)
    r[c.a_p <= 0.0] = 0.0
    fn = float(np.linalg.norm(f))
    return float(np.linalg.norm(r)) / fn if fn > 0.0 else 0.0


# ---------------------------------------------------------------------------
# A. Weighted Jacobi
# ---------------------------------------------------------------------------


def jacobi_committed(corrector, c, b, tol: float, cap: int) -> SolveResult:  # type: ignore[no-untyped-def]
    """Today's loop: committed sweep, stop on the largest weighted change below tol (Pa)."""
    active = c.a_p > 0.0
    t0 = time.perf_counter()
    p_prime = np.zeros(c.a_p.shape, dtype=np.float64)
    sweeps = 0
    for _ in range(cap):
        p_new = corrector.sweep(p_prime, c, b, JACOBI_WEIGHT)
        diff = float(np.max(np.abs(p_new[active] - p_prime[active]))) if active.any() else 0.0
        p_prime = p_new
        sweeps += 1
        if diff < tol:
            break
    return SolveResult(p_prime, sweeps, 0.0, time.perf_counter() - t0, [])


def jacobi_history(  # type: ignore[no-untyped-def]
    corrector, c, b, f, levels: list[float], cap: int, every: int = 10
) -> dict:
    """The committed sweep run until each relative-residual level is crossed (untimed).

    Returns the sweep at which each level was first seen (checked every
    ``every`` sweeps), and the largest weighted change at those sweeps.
    """
    active = c.a_p > 0.0
    p_prime = np.zeros(c.a_p.shape, dtype=np.float64)
    pending = sorted(levels, reverse=True)
    found: dict[str, dict] = {}
    trace: list[tuple[int, float, float]] = []
    for sweep in range(1, cap + 1):
        p_new = corrector.sweep(p_prime, c, b, JACOBI_WEIGHT)
        if sweep % every == 0 or sweep == cap:
            diff = float(np.max(np.abs(p_new[active] - p_prime[active])))
            rel = rel_residual(c, f, p_new)
            trace.append((sweep, rel, diff))
            while pending and rel < pending[0]:
                found[f"{pending[0]:.0e}"] = {"sweeps": sweep, "diff": diff, "rel": rel}
                pending.pop(0)
        p_prime = p_new
        if not pending:
            break
    return {"found": found, "trace": trace[:: max(1, len(trace) // 400)], "last": trace[-1]}


# ---------------------------------------------------------------------------
# B. Jacobi-preconditioned conjugate gradients
# ---------------------------------------------------------------------------


def pcg(  # type: ignore[no-untyped-def]
    matvec, precond, f: np.ndarray, rtol: float, maxiter: int, atol: float = 0.0, keep: bool = False
) -> tuple[np.ndarray, int, list[float]]:
    """Preconditioned CG from zero; stop when ||r||_2 <= max(rtol ||f||_2, atol)."""
    x = np.zeros_like(f)
    r = f.copy()
    fn = float(np.sqrt(np.vdot(f, f)))
    stop = max(rtol * fn, atol)
    hist: list[float] = []
    if fn == 0.0:
        return x, 0, hist
    z = precond(r)
    p = z.copy()
    rz = float(np.vdot(r, z))
    k = 0
    while k < maxiter:
        q = matvec(p)
        alpha = rz / float(np.vdot(p, q))
        x += alpha * p
        r -= alpha * q
        k += 1
        rn = float(np.sqrt(np.vdot(r, r)))
        if keep:
            hist.append(rn / fn)
        if rn <= stop:
            break
        z = precond(r)
        rz_new = float(np.vdot(r, z))
        p *= rz_new / rz
        p += z
        rz = rz_new
    return x, k, hist


def jacobi_pcg(c: PressureCoefficients, f: np.ndarray, rtol: float, maxiter: int = 20000,
               atol: float = 0.0, keep: bool = False) -> SolveResult:
    """B: CG with the diagonal a_P as preconditioner."""
    t0 = time.perf_counter()
    inv = np.where(c.a_p > 0.0, 1.0 / np.where(c.a_p > 0.0, c.a_p, 1.0), 0.0)
    t1 = time.perf_counter()
    x, k, hist = pcg(lambda v: apply_a(c, v), lambda r: inv * r, f, rtol, maxiter, atol, keep)
    return SolveResult(x, k, t1 - t0, time.perf_counter() - t1, hist)


# ---------------------------------------------------------------------------
# C. Geometric multigrid with Galerkin coarse operators
# ---------------------------------------------------------------------------

OFFSETS = [(dj, di) for dj in range(-2, 3) for di in range(-2, 3)]


class Level:
    """One grid: a stencil {offset: [ny, nx]} in the A x = f convention, and its mask."""

    def __init__(self, stencil: dict[tuple[int, int], np.ndarray], active: np.ndarray) -> None:
        self.stencil = {k: v for k, v in stencil.items() if np.any(v != 0.0)}
        self.active = active
        self.shape = active.shape
        diag = self.stencil[(0, 0)]
        self.dinv = np.where(active, 1.0 / np.where(active, diag, 1.0), 0.0)
        self.omega = JACOBI_WEIGHT
        self.width = max(max(abs(dj), abs(di)) for dj, di in self.stencil)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        ny, nx = self.shape
        w = self.width
        xp = np.zeros((ny + 2 * w, nx + 2 * w))
        xp[w:-w, w:-w] = x
        y = np.zeros(self.shape)
        for (dj, di), s in self.stencil.items():
            y += s * xp[w + dj : w + dj + ny, w + di : w + di + nx]
        return y

    def padded_even(self) -> "Level":
        """This level with one inactive row and/or column added so both sides are even."""
        ny, nx = self.shape
        py, px = ny % 2, nx % 2
        if not (py or px):
            return self
        st = {k: np.pad(v, ((0, py), (0, px))) for k, v in self.stencil.items()}
        lev = Level(st, np.pad(self.active, ((0, py), (0, px))))
        lev.omega = self.omega
        return lev


class Transfer:
    """Cell-centred bilinear prolongation from a coarse grid to an even fine grid, and R = P^T."""

    def __init__(self, fine_active: np.ndarray, coarse_active: np.ndarray) -> None:
        nyf, nxf = fine_active.shape
        nyc, nxc = coarse_active.shape
        j, i = np.meshgrid(np.arange(nyf), np.arange(nxf), indexing="ij")
        pj, pi = j // 2, i // 2
        sj = np.where(j % 2 == 0, -1, 1)
        si = np.where(i % 2 == 0, -1, 1)
        cand = [(pj, pi, 9.0), (pj, pi + si, 3.0), (pj + sj, pi, 3.0), (pj + sj, pi + si, 1.0)]
        idx, wts = [], []
        for cj, ci, w in cand:
            ok = (cj >= 0) & (cj < nyc) & (ci >= 0) & (ci < nxc)
            cjc, cic = np.clip(cj, 0, nyc - 1), np.clip(ci, 0, nxc - 1)
            ok &= coarse_active[cjc, cic]
            idx.append((cjc * nxc + cic).ravel())
            wts.append(np.where(ok, w, 0.0).ravel())
        total = sum(wts)
        fa = fine_active.ravel()
        scale = np.where(fa & (total > 0.0), 1.0 / np.where(total > 0.0, total, 1.0), 0.0)
        self.idx = idx
        self.wts = [w * scale for w in wts]
        self.fine_shape, self.coarse_shape = (nyf, nxf), (nyc, nxc)
        self.nc = nyc * nxc
        self.cat_idx = np.concatenate(idx)

    def prolong(self, xc: np.ndarray) -> np.ndarray:
        flat = xc.ravel()
        out = self.wts[0] * flat[self.idx[0]]
        for k in range(1, 4):
            out += self.wts[k] * flat[self.idx[k]]
        return out.reshape(self.fine_shape)

    def restrict(self, rf: np.ndarray) -> np.ndarray:
        flat = rf.ravel()
        vals = np.concatenate([w * flat for w in self.wts])
        return np.bincount(self.cat_idx, weights=vals, minlength=self.nc).reshape(self.coarse_shape)


def galerkin(fine: Level, tr: Transfer, coarse_active: np.ndarray) -> Level:
    """A_c = R A P by probing with the 25 colours of a 5 x 5 tiling."""
    nyc, nxc = coarse_active.shape
    J, I = np.meshgrid(np.arange(nyc), np.arange(nxc), indexing="ij")
    probes = {}
    for a in range(5):
        for b in range(5):
            e = ((J % 5 == a) & (I % 5 == b) & coarse_active).astype(np.float64)
            probes[(a, b)] = tr.restrict(fine.matvec(tr.prolong(e)))
    stencil = {}
    for dj, di in OFFSETS:
        tj, ti = J + dj, I + di
        ok = (tj >= 0) & (tj < nyc) & (ti >= 0) & (ti < nxc)
        ok &= coarse_active[np.clip(tj, 0, nyc - 1), np.clip(ti, 0, nxc - 1)] & coarse_active
        s = np.zeros((nyc, nxc))
        for a in range(5):
            for b in range(5):
                sel = ok & ((tj % 5) == a) & ((ti % 5) == b)
                s[sel] = probes[(a, b)][sel]
        stencil[(dj, di)] = s
    return Level(stencil, coarse_active.copy())


def lambda_max(level: Level, iters: int = 30, seed: int = 36) -> float:
    """Largest eigenvalue of D^-1 A by power iteration (an estimate from below)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(level.shape) * level.active
    lam = 1.0
    for _ in range(iters):
        y = level.dinv * level.matvec(x)
        n = float(np.linalg.norm(y))
        if n == 0.0:
            return 1.0
        lam = n / float(np.linalg.norm(x))
        x = y / n
    return lam


class GMG:
    """Galerkin geometric multigrid on a five-point p' system; V(nu1, nu2) cycles."""

    def __init__(self, c: PressureCoefficients, singular: bool, nu1: int = 2, nu2: int = 2,
                 max_coarse: int = 64) -> None:
        t0 = time.perf_counter()
        fine = Level(
            {(0, 0): c.a_p.copy(), (0, 1): -c.a_e, (0, -1): -c.a_w, (1, 0): -c.a_n, (-1, 0): -c.a_s},
            c.a_p > 0.0,
        )
        self.shape0 = c.a_p.shape
        self.nu1, self.nu2 = nu1, nu2
        self.levels: list[Level] = []
        self.transfers: list[Transfer] = []
        lev = fine
        while True:
            ny, nx = lev.shape
            if ny * nx <= max_coarse or min(ny, nx) <= 4:
                self.levels.append(lev)
                break
            lev = lev.padded_even()
            self.levels.append(lev)
            ny, nx = lev.shape
            act = lev.active.reshape(ny // 2, 2, nx // 2, 2).any(axis=(1, 3))
            tr = Transfer(lev.active, act)
            self.transfers.append(tr)
            nxt = galerkin(lev, tr, act)
            nxt.omega = 4.0 / (3.0 * lambda_max(nxt))
            lev = nxt
        # Coarsest: pseudo-inverse on the cells with an equation.
        coarse = self.levels[-1]
        n = coarse.active.size
        dense = np.zeros((n, n))
        for k in range(n):
            e = np.zeros(n)
            e[k] = 1.0
            dense[:, k] = coarse.matvec(e.reshape(coarse.shape)).ravel()
        act = coarse.active.ravel()
        sub = np.linalg.pinv(dense[np.ix_(act, act)])
        self.coarse_inv = (act, sub)
        self.singular = singular
        self.setup_s = time.perf_counter() - t0

    def _smooth(self, lev: Level, x: np.ndarray, f: np.ndarray, n: int) -> np.ndarray:
        for _ in range(n):
            x = x + lev.omega * lev.dinv * (f - lev.matvec(x))
        return x

    def _cycle(self, k: int, f: np.ndarray) -> np.ndarray:
        lev = self.levels[k]
        if k == len(self.levels) - 1:
            act, sub = self.coarse_inv
            x = np.zeros(f.size)
            x[act] = sub @ f.ravel()[act]
            return x.reshape(f.shape)
        x = lev.omega * lev.dinv * f  # first pre-sweep from zero
        x = self._smooth(lev, x, f, self.nu1 - 1)
        r = f - lev.matvec(x)
        tr = self.transfers[k]
        nxt = self.levels[k + 1]
        rc = tr.restrict(r)
        if rc.shape != nxt.shape:  # the next level was padded after coarsening
            rc = np.pad(rc, ((0, nxt.shape[0] - rc.shape[0]), (0, nxt.shape[1] - rc.shape[1])))
        xc = self._cycle(k + 1, rc)
        x = x + tr.prolong(xc[: tr.coarse_shape[0], : tr.coarse_shape[1]])
        return self._smooth(lev, x, f, self.nu2)

    def precond(self, r: np.ndarray) -> np.ndarray:
        """One V-cycle from zero on the finest grid's (unpadded) shape."""
        ny, nx = self.shape0
        top = self.levels[0]
        rp = r if r.shape == top.shape else np.pad(r, ((0, top.shape[0] - ny), (0, top.shape[1] - nx)))
        return self._cycle(0, rp)[:ny, :nx]


def gmg_solve(c: PressureCoefficients, f: np.ndarray, rtol: float, singular: bool,
              maxiter: int = 500, keep: bool = False, mg: GMG | None = None) -> SolveResult:
    """C standalone: x <- x + V(f - A x) until the relative residual is below rtol."""
    if mg is None:
        mg = GMG(c, singular)
        setup = mg.setup_s
    else:
        setup = 0.0
    t1 = time.perf_counter()
    x = np.zeros_like(f)
    fn = float(np.linalg.norm(f))
    hist: list[float] = []
    k = 0
    r = f.copy()
    while k < maxiter and fn > 0.0:
        x += mg.precond(r)
        r = f - apply_a(c, x)
        r[c.a_p <= 0.0] = 0.0
        k += 1
        rel = float(np.linalg.norm(r)) / fn
        if keep:
            hist.append(rel)
        if rel <= rtol:
            break
    return SolveResult(x, k, setup, time.perf_counter() - t1, hist)


def mg_pcg(c: PressureCoefficients, f: np.ndarray, rtol: float, singular: bool,
           maxiter: int = 500, keep: bool = False, mg: GMG | None = None) -> SolveResult:
    """C as the preconditioner of CG."""
    if mg is None:
        mg = GMG(c, singular)
        setup = mg.setup_s
    else:
        setup = 0.0
    t1 = time.perf_counter()
    x, k, hist = pcg(lambda v: apply_a(c, v), mg.precond, f, rtol, maxiter, keep=keep)
    return SolveResult(x, k, setup, time.perf_counter() - t1, hist)


# ---------------------------------------------------------------------------
# D and E: through scipy (probe environment only)
# ---------------------------------------------------------------------------


def to_csr(c: PressureCoefficients):  # type: ignore[no-untyped-def]
    """A on the cells with an equation as a scipy CSR matrix, and the index map."""
    import scipy.sparse as sp

    active = c.a_p > 0.0
    ny, nx = active.shape
    index = -np.ones((ny, nx), dtype=np.int64)
    index[active] = np.arange(int(active.sum()))
    rows, cols, vals = [index[active]], [index[active]], [c.a_p[active]]
    for coef, dj, di in ((c.a_e, 0, 1), (c.a_w, 0, -1), (c.a_n, 1, 0), (c.a_s, -1, 0)):
        jj, ii = np.nonzero(active & (coef != 0.0))
        tj, ti = jj + dj, ii + di
        rows.append(index[jj, ii])
        cols.append(index[tj, ti])
        vals.append(-coef[jj, ii])
    n = int(active.sum())
    a = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    )
    return a, index


def pyamg_solve(c: PressureCoefficients, f: np.ndarray, rtol: float, kind: str, accel: str | None,
                maxiter: int = 500, keep: bool = False, ml=None) -> SolveResult:  # type: ignore[no-untyped-def]
    """D: pyamg's solver of the given kind; the setup includes assembling the CSR matrix."""
    import pyamg

    t0 = time.perf_counter()
    a, index = to_csr(c)
    if ml is None:
        if kind == "sa":
            ml = pyamg.smoothed_aggregation_solver(a, B=np.ones((a.shape[0], 1)), coarse_solver="pinv")
        elif kind == "sa_jacobi":
            jac = ("jacobi", {"omega": 2.0 / 3.0, "iterations": 2})
            ml = pyamg.smoothed_aggregation_solver(
                a, B=np.ones((a.shape[0], 1)), presmoother=jac, postsmoother=jac, coarse_solver="pinv"
            )
        elif kind == "rs":
            ml = pyamg.ruge_stuben_solver(a, coarse_solver="pinv")
        else:
            raise ValueError(kind)
        stale = False
    else:
        stale = True  # a hierarchy built on another matrix: CG on this one, it as M
    t1 = time.perf_counter()
    active = c.a_p > 0.0
    rhs = f[active]
    res: list[float] = []
    if stale:
        sol, _ = pyamg.krylov.cg(a, rhs, tol=rtol, maxiter=maxiter, M=ml.aspreconditioner(),
                                 residuals=res)
    else:
        sol = ml.solve(rhs, tol=rtol, maxiter=maxiter, accel=accel, residuals=res)
    t2 = time.perf_counter()
    x = np.zeros_like(f)
    x[active] = sol
    rn = float(np.linalg.norm(rhs))
    hist = [r / rn for r in res[1:]] if keep else []
    return SolveResult(x, max(len(res) - 1, 0), t1 - t0, t2 - t1, hist), ml  # type: ignore[return-value]


def direct_solve(c: PressureCoefficients, f: np.ndarray, singular: bool,
                 pin: tuple[int, int] | None = None) -> SolveResult:
    """E: SuperLU on A; on a closed domain the pin cell's row and column removed."""
    import scipy.sparse.linalg as spla

    t0 = time.perf_counter()
    a, index = to_csr(c)
    active = c.a_p > 0.0
    rhs = f[active]
    keep_rows = np.ones(a.shape[0], dtype=bool)
    if singular:
        keep_rows[index[pin]] = False
        a = a[keep_rows][:, keep_rows]
        rhs = rhs[keep_rows]
    lu = spla.splu(a.tocsc())
    t1 = time.perf_counter()
    sol = lu.solve(rhs)
    t2 = time.perf_counter()
    full = np.zeros(int(active.sum()))
    full[keep_rows] = sol
    x = np.zeros_like(f)
    x[active] = full
    return SolveResult(x, 1, t1 - t0, t2 - t1, [])
```

## Appendix C: selftest36.py

```python
"""Builder probe, prompt 36: controls on the harness, before any measurement.

Usage: python selftest36.py [synthetic|rectangle|corrector|cost33|all]

synthetic  Every candidate on two synthetic five-point systems with SOLID
           holes, random positive face coefficients and odd sides: one with
           Dirichlet rows (positive definite), one closed (singular, a
           compatible right-hand side). Each solved to 1e-10 and compared
           with SuperLU. The probed Galerkin product against an explicit
           sparse P^T A P.
rectangle  GMG alone on constant-coefficient rectangles with no holes (closed,
           a Dirichlet patch on one edge, Dirichlet on all four edges), at four
           sizes: the convergence factor per V(2,2) cycle, which for this
           design should be about 0.2 to 0.3 and independent of the size.
corrector  The 40x15 room under T3, ten sweeps, step 0's settings, 30 outer
           iterations: ProbeCorrector with the committed call, and with the
           committed loop copied into the replaceable solve, against each
           other and against test 34b's A-sw record (the main tree's
           results/tester34b/A-sw.json, read only).
cost33     The product as committed (cost33.py's room): the outer-0 system
           captured and today's loop run on it to 1e-6 and 1e-8 Pa, against
           the sweep counts of product_case_reynolds.md section 5 (27,408 and
           112,519).
Writes selftest36.json.
"""

import json
import sys
import time

import numpy as np

from common36 import (
    HERE,
    PressureCoefficients,
    apply_a,
    committed_jacobi,
    install_corrector,
    load_system,
    product_committed,
    product_t3,
)
from solvers36 import (
    GMG,
    Level,
    Transfer,
    direct_solve,
    galerkin,
    gmg_solve,
    jacobi_committed,
    jacobi_pcg,
    mg_pcg,
    pyamg_solve,
    rel_residual,
)

A_SW = HERE.parents[2] / "cfd_clean_room" / "results" / "tester34b" / "A-sw.json"


def synthetic(ny: int, nx: int, closed: bool, seed: int) -> PressureCoefficients:
    """A five-point SPD (or closed, singular) system with holes, built as the corrector builds one."""
    rng = np.random.default_rng(seed)
    solid = np.zeros((ny, nx), dtype=bool)
    solid[ny // 3 : ny // 2, nx // 4 : nx // 3] = True
    solid[0 : ny // 2, (2 * nx) // 3 : (2 * nx) // 3 + 3] = True
    # face coefficients, varying over four orders of magnitude
    cu = 10.0 ** rng.uniform(-2, 2, (ny, nx + 1))
    cv = 10.0 ** rng.uniform(-2, 2, (ny + 1, nx))
    cu[:, 1:-1][solid[:, :-1] | solid[:, 1:]] = 0.0
    cv[1:-1, :][solid[:-1, :] | solid[1:, :]] = 0.0
    cu[:, 0] = 0.0
    cu[:, -1] = 0.0
    cv[-1, :] = 0.0
    if closed:
        cv[0, :] = 0.0
    else:
        cv[0, : nx // 5] = cu[0, 1]  # a Dirichlet patch on the bottom edge
        cv[0, nx // 5 :] = 0.0
    a_e, a_w = cu[:, 1:].copy(), cu[:, :-1].copy()
    a_n, a_s = cv[1:, :].copy(), cv[:-1, :].copy()
    a_p = a_e + a_w + a_n + a_s
    a_w[:, 0] = 0.0
    a_e[:, -1] = 0.0
    a_s[0, :] = 0.0
    a_n[-1, :] = 0.0
    for arr in (a_p, a_e, a_w, a_n, a_s):
        arr[solid] = 0.0
    return PressureCoefficients(a_p=a_p, a_e=a_e, a_w=a_w, a_n=a_n, a_s=a_s)


def check_synthetic(report: dict) -> None:
    import scipy.sparse as sp

    for closed in (False, True):
        c = synthetic(45, 61, closed, 36 + int(closed))
        act = c.a_p > 0.0
        rng = np.random.default_rng(7)
        f = rng.standard_normal(c.a_p.shape) * act
        pin = tuple(int(v) for v in np.argwhere(act)[0])
        if closed:
            f[act] -= f[act].mean()
        exact = direct_solve(c, f, closed, pin).x
        if closed:
            exact[act] -= exact[pin]
        label = "closed" if closed else "open"
        runs = {
            "pcg": jacobi_pcg(c, f, 1e-10),
            "gmg": gmg_solve(c, f, 1e-10, closed),
            "mg_pcg": mg_pcg(c, f, 1e-10, closed),
            "sa": pyamg_solve(c, f, 1e-10, "sa", None)[0],
            "sa_cg": pyamg_solve(c, f, 1e-10, "sa", "cg")[0],
            "sa_jacobi_cg": pyamg_solve(c, f, 1e-10, "sa_jacobi", "cg")[0],
            "rs_cg": pyamg_solve(c, f, 1e-10, "rs", "cg")[0],
        }
        for name, res in runs.items():
            x = res.x.copy()
            if closed:
                x[act] -= x[pin]
            err = float(np.max(np.abs(x - exact)) / np.max(np.abs(exact)))
            report[f"synthetic {label}: {name} iterations, rel residual, max error vs direct"] = [
                res.iterations, rel_residual(c, f, res.x), err,
            ]
        report[f"synthetic {label}: direct rel residual"] = rel_residual(c, f, exact)

        # The probed Galerkin product against an explicit sparse one, on the first two levels.
        mg = GMG(c, closed)
        for k in range(min(2, len(mg.transfers))):
            fine, tr, coarse = mg.levels[k], mg.transfers[k], mg.levels[k + 1]
            nf, nc = fine.active.size, tr.nc
            rows = np.concatenate([np.arange(nf)] * 4)
            pmat = sp.csr_matrix((np.concatenate(tr.wts), (rows, tr.cat_idx)), shape=(nf, nc))
            amat = sp.lil_matrix((nf, nf))
            ny, nx = fine.shape
            for (dj, di), s in fine.stencil.items():
                jj, ii = np.nonzero(s)
                ok = (jj + dj >= 0) & (jj + dj < ny) & (ii + di >= 0) & (ii + di < nx)
                amat[(jj * nx + ii)[ok], ((jj + dj) * nx + ii + di)[ok]] = s[jj[ok], ii[ok]]
            rap = (pmat.T @ amat.tocsr() @ pmat).toarray()
            probed = np.zeros((nc, nc))
            cy, cx = tr.coarse_shape
            for (dj, di), s in coarse.stencil.items():
                s = s[:cy, :cx]
                jj, ii = np.nonzero(s)
                ok = (jj + dj >= 0) & (jj + dj < cy) & (ii + di >= 0) & (ii + di < cx)
                probed[(jj * cx + ii)[ok], ((jj + dj) * cx + ii + di)[ok]] = s[jj[ok], ii[ok]]
            report[f"synthetic {label}: Galerkin level {k + 1} probed vs explicit, max rel diff"] = float(
                np.max(np.abs(rap - probed)) / np.max(np.abs(rap))
            )
            report[f"synthetic {label}: Galerkin level {k + 1} symmetric, max rel asymmetry"] = float(
                np.max(np.abs(probed - probed.T)) / np.max(np.abs(probed))
            )
        report[f"synthetic {label}: GMG level shapes"] = [list(lv.shape) for lv in mg.levels]
        report[f"synthetic {label}: GMG coarse omegas"] = [round(lv.omega, 4) for lv in mg.levels]


def rectangle(ny: int, nx: int, dirichlet: str) -> PressureCoefficients:
    """Unit face coefficients on a rectangle, with p' = 0 rows as asked."""
    a_e, a_w, a_n, a_s = (np.ones((ny, nx)) for _ in range(4))
    a_w[:, 0] = 0.0
    a_e[:, -1] = 0.0
    a_s[0, :] = 0.0
    a_n[-1, :] = 0.0
    a_p = a_e + a_w + a_n + a_s
    if dirichlet == "patch":
        a_p[0, : nx // 5] += 2.0
    elif dirichlet == "all":
        a_p[0, :] += 2.0
        a_p[-1, :] += 2.0
        a_p[:, 0] += 2.0
        a_p[:, -1] += 2.0
    return PressureCoefficients(a_p=a_p, a_e=a_e, a_w=a_w, a_n=a_n, a_s=a_s)


def check_rectangle(report: dict) -> None:
    for ny, nx in ((32, 32), (64, 64), (45, 61), (75, 200)):
        for kind in ("closed", "patch", "all"):
            c = rectangle(ny, nx, kind)
            f = np.random.default_rng(1).standard_normal((ny, nx))
            closed = kind == "closed"
            if closed:
                f -= f.mean()
            res = gmg_solve(c, f, 1e-10, closed, keep=True, maxiter=200)
            h = res.history
            report[f"rectangle {ny}x{nx} {kind}: V(2,2) cycles to 1e-10, last factor"] = [
                res.iterations, round(h[-1] / h[-2], 3),
            ]


def check_corrector(report: dict) -> None:
    n = 30
    runs = {}
    for mode in ("committed", "copy"):
        room = product_t3(f"self_{mode}", 40, 15, n)
        corr = install_corrector(room, None)
        if mode == "copy":
            corr.solve = committed_jacobi(corr, room.cfg.pressure_tol, room.cfg.max_pressure_iter)
        res: list[float] = []
        room.solver.solve_steady(on_iteration=lambda s: res.append(s.residual))
        fv = room.solver.face_velocities
        runs[mode] = (res, fv.u.copy(), fv.v.copy(), [r.inner for r in corr.records])
    a, b = runs["committed"], runs["copy"]
    report["corrector: copied loop vs committed call, residuals bitwise (1 = yes)"] = float(a[0] == b[0])
    report["corrector: copied loop vs committed call, final faces bitwise (1 = yes)"] = float(
        np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
    )
    report["corrector: sweeps equal (1 = yes)"] = float(a[3] == b[3])
    rec = json.loads(A_SW.read_text())
    report["corrector: committed call vs test 34b A-sw, first 30 residuals bitwise (1 = yes)"] = float(
        a[0] == rec["residual"][:n]
    )
    report["corrector: committed call vs test 34b A-sw, first 30 sweep counts equal (1 = yes)"] = float(
        a[3] == rec["sweeps"][:n]
    )


def check_cost33(report: dict) -> None:
    room = product_committed("cost33_control")
    corr = install_corrector(room, None, capture_at=(0,), capture_name="cost33", measure_residual=False)
    room.solver.solve_steady()
    s = load_system("cost33_it0")
    for tol, expected in ((1e-6, 27408), (1e-8, 112519)):
        t0 = time.perf_counter()
        res = jacobi_committed(corr, s.c, s.b, tol, 5_000_000)
        report[f"cost33: today's loop to {tol:.0e} Pa, sweeps (report: {expected})"] = res.iterations
        report[f"cost33: seconds at {tol:.0e} Pa, ms per sweep"] = [
            time.perf_counter() - t0, 1e3 * (time.perf_counter() - t0) / res.iterations,
        ]


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    out = HERE / "selftest36.json"
    report = json.loads(out.read_text()) if out.exists() else {}
    if which in ("synthetic", "all"):
        check_synthetic(report)
    if which in ("rectangle", "all"):
        check_rectangle(report)
    if which in ("corrector", "all"):
        check_corrector(report)
    if which in ("cost33", "all"):
        check_cost33(report)
    for k, v in report.items():
        print(f"{k}: {v}")
    out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
```

## Appendix D: capture36.py

```python
"""Builder probe, prompt 36, item 0: run a room and keep its p' systems.

Usage: python capture36.py ROOM [N_OUTER] [AT]

AT, comma-separated outer iterations, replaces the default 1,100,1000 (the
VAL-001 channel stops at 570 under velocity_step, so its third capture is its
last iteration, 569).

ROOM is one of
    product200   200x75 product under T3, laminar, ten momentum sweeps, the
                 correction solved by SuperLU (exact); error_estimate stop
                 (iteration_error_tol 1e-6, mass_imbalance_tol from ADR-011
                 G's formula at the configured t_end); captures outer 1, 100,
                 1000; runs on to its stop or N_OUTER (default 20,000), so it
                 also gives measurement 3 the product's outer count
    product40    the same on 40x15, captures 1, 100, 1000 (for the dense
                 Cholesky check beside the large one)
    cavity80     VAL-002 80x80 as its case file sets it, today's correction
                 (cap 500, 1e-8 Pa), velocity_step; captures 1, 100, 1000
    channel80    VAL-001 80x40 likewise (cap 2,000, 1e-8 Pa)
    annex180     the Annex 20 room on 180x60, laminar, outlet T1, ten sweeps,
                 SuperLU; captures 1, 100, 1000; stops at N_OUTER (1,001)

Per outer iteration it records the solver's residual, the step in m/s, the
inner iterations, the relative residual the correction reached, the norm of
b, the worst, absolute-summed and signed imbalance of the corrected faces,
the largest cell-centred speed, the faces T3 holds shut, and the seconds of
the momentum stage and of the p' solve. A run stops at the solver's stop, at
a non-finite field or past 100 m/s (diverged). Writes ROOM.json and the
final fields to ROOM.npz; the systems go to systems/ROOM_itK.npz.
"""

import json
import math
import sys
import time
from datetime import datetime

import numpy as np

from common36 import (
    HERE,
    Room,
    annex20,
    install_corrector,
    product_t3,
    threads_note,
    validation_case,
)
from solvers36 import direct_solve

DIVERGED_SPEED = 100.0


class Diverged(Exception):
    """Raised from the callback to end a run past 100 m/s or non-finite."""

CAPTURE = (1, 100, 1000)


def stopping_for(nx: int, ny: int, width: float, height: float, rho: float, t_end: float) -> dict:
    """error_estimate with the defaults' error tolerance and ADR-011 G's per-cell bound."""
    v_min = (width / nx) * (height / ny)
    return {
        "stopping_rule": "error_estimate",
        "iteration_error_tol": 1e-6,
        "mass_imbalance_tol": 1e-4 * rho * v_min / t_end,
    }


def exact_solve(c, b, active, needs_pin):  # type: ignore[no-untyped-def]
    f = -b.copy()
    if needs_pin:
        f[active] -= f[active].mean()
    f[~active] = 0.0
    pin = tuple(int(v) for v in np.argwhere(active)[0])
    res = direct_solve(c, f, needs_pin, pin)
    return res.x, 1


def build(room_name: str, n_outer: int) -> tuple[Room, object]:
    if room_name in ("product200", "product40"):
        nx, ny = (200, 75) if room_name == "product200" else (40, 15)
        stop = stopping_for(nx, ny, 8.0, 3.0, 1.2, 60.0)
        room = product_t3(room_name, nx, ny, n_outer, stopping=stop)
        return room, exact_solve
    if room_name == "cavity80":
        return validation_case(room_name, "val002_80x80", n_outer), None
    if room_name == "channel80":
        return validation_case(room_name, "val001_80x40", n_outer), None
    if room_name == "annex180":
        return annex20(room_name, 180, 60, n_outer), exact_solve
    raise SystemExit(f"unknown room {room_name}")


def run_room(room: Room, solve, room_name: str, capture_at: tuple[int, ...] = (),  # type: ignore[no-untyped-def]
             print_every: int = 50) -> dict:
    """Run a room to its stop with a given p' solve, recording every outer iteration."""
    corr = install_corrector(room, solve, capture_at=capture_at, capture_name=room_name)
    solver = room.solver
    started = datetime.now().isoformat(timespec="seconds")
    print(f"{room_name} start {started} {room.meta} supply {room.supply:.6g} threads {threads_note()}", flush=True)

    latest: dict[str, np.ndarray] = {}
    original = corr.correct

    def keep(prediction, p):  # type: ignore[no-untyped-def]
        out = original(prediction, p)
        latest["u"], latest["v"], latest["p"] = out.u, out.v, out.p
        return out

    corr.correct = keep  # type: ignore[method-assign]
    ref = solver.reference_velocity
    rec: dict[str, list] = {k: [] for k in (
        "residual", "step", "inner", "rel", "b_norm", "worst", "abs_sum", "signed",
        "max_speed", "closed", "momentum_s", "pressure_s", "clock")}
    t0 = time.perf_counter()
    last_momentum = [0.0]
    vs_stop: dict = {}

    def callback(state) -> None:  # type: ignore[no-untyped-def]
        speed = np.hypot(state.u, state.v)
        top = float(np.max(speed))
        imb = corr.mass_imbalance(latest["u"], latest["v"])
        r = corr.records[-1]
        rec["residual"].append(float(state.residual))
        rec["step"].append(float(state.residual) * ref)
        rec["inner"].append(r.inner)
        rec["rel"].append(r.rel_residual)
        rec["b_norm"].append(r.b_norm)
        rec["worst"].append(float(np.max(np.abs(imb))))
        rec["abs_sum"].append(float(np.sum(np.abs(imb))))
        rec["signed"].append(float(np.sum(imb)))
        rec["max_speed"].append(top)
        closed = int(np.sum(getattr(solver, "closed_bottom", np.zeros(1)))) + int(
            np.sum(getattr(solver, "closed_right", np.zeros(1))))
        rec["closed"].append(closed)
        mom = solver.stage_seconds["momentum"]
        rec["momentum_s"].append(mom - last_momentum[0])
        last_momentum[0] = mom
        rec["pressure_s"].append(r.seconds)
        rec["clock"].append(time.perf_counter() - t0)
        if state.residual < 1e-6 and not vs_stop:
            vs_stop.update(outer=state.iteration + 1, u=state.u.copy(), v=state.v.copy())
        it = state.iteration
        if it % print_every == 0 or it in capture_at:
            print(
                f"{room_name} it {it:6d} res {state.residual:.4e} inner {r.inner:6d} rel {r.rel_residual:.2e} "
                f"max|U| {top:.4g} worst {rec['worst'][-1]:.2e} signed {rec['signed'][-1]:.2e} "
                f"closed {closed} t {rec['clock'][-1]:8.1f}s",
                flush=True,
            )
        if not math.isfinite(top) or top > DIVERGED_SPEED:
            raise Diverged

    try:
        u, v, p = solver.solve_steady(on_iteration=callback)
        stop = solver.stop_reason
    except Diverged:
        stop = "diverged"
        u = v = p = None
    out = {
        "room": room_name, "meta": room.meta, "started": started, "supply": room.supply,
        "reference_velocity": ref, "stop": stop, "outer": len(rec["residual"]),
        "velocity_step_outer": vs_stop.get("outer"), "seconds": time.perf_counter() - t0,
        "threads": threads_note(), "stage_seconds": dict(solver.stage_seconds), **rec,
    }
    (HERE / f"{room_name}.json").write_text(json.dumps(out))
    if u is not None:
        np.savez(HERE / f"{room_name}.npz", u=u, v=v, p=p,
                 u_vs=vs_stop.get("u", np.zeros(1)), v_vs=vs_stop.get("v", np.zeros(1)))
    print(f"{room_name} done {datetime.now().isoformat(timespec='seconds')} stop {stop} "
          f"after {out['outer']} (velocity_step at {out['velocity_step_outer']})", flush=True)
    return out


def main() -> None:
    room_name = sys.argv[1]
    default = {"product200": 20000, "product40": 20000}.get(room_name, 1001)
    n_outer = int(sys.argv[2]) if len(sys.argv) > 2 else default
    at = tuple(int(k) for k in sys.argv[3].split(",")) if len(sys.argv) > 3 else CAPTURE
    room, solve = build(room_name, n_outer)
    run_room(room, solve, room_name, at)


if __name__ == "__main__":
    main()
```

## Appendix E: item0_36.py

```python
"""Builder probe, prompt 36, item 0: are the captured p' matrices symmetric and positive definite?

Usage: python item0_36.py SYSTEM [SYSTEM ...]

For each captured system (systems/SYSTEM.npz), on the cells with an
equation (a_P > 0):

- symmetry, bitwise: a_e[j, i] against a_w[j, i+1] and a_n[j, i] against
  a_s[j+1, i], the two copies of each face coefficient; and as a matrix,
  max |A - A^T| / max |A|;
- the sign structure: a_P > 0, every neighbour coefficient >= 0 (so the
  off-diagonals of A are <= 0), the row surplus a_P - sum(a_nb) >= 0, the
  rows where it is positive (outlet faces, Dirichlet p' = 0) and the
  connected components of the cell graph, each with or without such a row;
- the smallest eigenvalues of A by shift-invert (scipy eigsh, sigma just
  below zero), the largest by Lanczos, and the same for D^-1 A (the
  generalized problem A x = lambda D x), whose extremes set Jacobi's rate
  and conjugate gradients' condition number;
- a dense Cholesky factorization of A (of A with the pin cell's row and
  column removed on a closed domain), which succeeds exactly when the
  matrix is positive definite;
- on a closed domain, A 1 = 0 and the compatibility |sum b| / sum |b|;
- the fair stop's identity: the imbalance PressureCorrector's arithmetic
  gives on faces corrected with any x equals b + A x, checked at the direct
  solution and at a random x.
Writes item0_SYSTEM.json.
"""

import json
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import scipy.sparse.linalg as spla

from common36 import HERE, corrected_faces, corrected_imbalance, face_imbalance, load_system
from solvers36 import direct_solve, to_csr


def check(name: str) -> dict:
    s = load_system(name)
    c = s.c
    act = s.active
    out: dict = {"system": name, "meta": s.meta, "cells": int(act.size), "active": int(act.sum()),
                 "solid": int(s.solid.sum()), "inactive_not_solid": int((~act & ~s.solid).sum())}
    out["sym_bitwise_ew"] = bool(np.array_equal(c.a_e[:, :-1], c.a_w[:, 1:]))
    out["sym_bitwise_ns"] = bool(np.array_equal(c.a_n[:-1, :], c.a_s[1:, :]))
    a, index = to_csr(c)
    asym = abs(a - a.T)
    out["sym_matrix_max_rel"] = float(asym.max() / abs(a).max()) if asym.nnz else 0.0
    nb = c.a_e + c.a_w + c.a_n + c.a_s
    out["diag_positive_all"] = bool(np.all(c.a_p[act] > 0.0))
    out["neighbours_nonnegative_all"] = bool(min(float(x[act].min()) for x in (c.a_e, c.a_w, c.a_n, c.a_s)) >= 0.0)
    surplus = c.a_p - nb
    out["surplus_min_over_diag"] = float(np.min(surplus[act] / c.a_p[act]))
    strict = act & (surplus > 1e-12 * c.a_p)
    out["strict_rows"] = int(strict.sum())
    ncomp, labels = csgraph.connected_components(a, directed=False)
    strict_idx = index[strict]
    comps_with = len(set(labels[strict_idx].tolist())) if strict_idx.size else 0
    out["components"] = int(ncomp)
    out["components_with_strict_row"] = int(comps_with)

    diag = sp.diags(a.diagonal())
    amax = float(a.diagonal().max())
    sigma = -1e-9 * amax
    t0 = time.perf_counter()
    lo = spla.eigsh(a, k=3, sigma=sigma, which="LM", return_eigenvectors=False)
    hi = spla.eigsh(a, k=1, which="LA", return_eigenvectors=False)
    glo = spla.eigsh(a, k=3, M=diag, sigma=sigma, which="LM", return_eigenvectors=False)
    ghi = spla.eigsh(a, k=1, M=diag, which="LA", return_eigenvectors=False)
    out["eig_seconds"] = time.perf_counter() - t0
    out["eig_A_smallest"] = sorted(float(x) for x in lo)
    out["eig_A_largest"] = float(hi[0])
    out["eig_DinvA_smallest"] = sorted(float(x) for x in glo)
    out["eig_DinvA_largest"] = float(ghi[0])

    dense = a.toarray()
    pin = s.pin_cell
    if s.needs_pin:
        keep = np.ones(a.shape[0], dtype=bool)
        keep[index[pin]] = False
        dense = dense[np.ix_(keep, keep)]
        out["ones_null_max_rel"] = float(np.max(np.abs(a @ np.ones(a.shape[0]))) / amax)
        out["compatibility"] = float(abs(s.b[act].sum()) / np.abs(s.b[act]).sum())
    t0 = time.perf_counter()
    try:
        chol = np.linalg.cholesky(dense)
        out["cholesky"] = "succeeded"
        out["cholesky_min_pivot_sq_over_max"] = float(np.min(np.diag(chol)) ** 2 / amax)
        del chol
    except np.linalg.LinAlgError as err:
        out["cholesky"] = f"failed: {err}"
    out["cholesky_n"] = int(dense.shape[0])
    out["cholesky_seconds"] = time.perf_counter() - t0
    del dense

    f = s.rhs()
    x = direct_solve(c, f, s.needs_pin, pin).x
    rng = np.random.default_rng(36)
    bmax = float(np.max(np.abs(s.b)))
    for label, xx in (("direct", x), ("random", rng.standard_normal(x.shape) * act)):
        u, v = corrected_faces(s, xx)
        diff = face_imbalance(s, u, v) - corrected_imbalance(c, s.b, xx)
        out[f"identity_{label}_max_over_max_b"] = float(np.max(np.abs(diff[act])) / bmax)
        # Against the largest single face flux of the corrected field, the scale of the rounding.
        flux = s.rho * max(float(np.max(np.abs(u))) * float(s.dy_cell.max()),
                           float(np.max(np.abs(v))) * float(s.dx_cell.max()))
        out[f"identity_{label}_max_over_face_flux"] = float(np.max(np.abs(diff[act])) / flux)
    out["b_norm"] = float(np.linalg.norm(s.b))
    out["b_sum"] = float(s.b[act].sum())
    out["supply"] = s.supply
    return out


def main() -> None:
    for name in sys.argv[1:]:
        res = check(name)
        (HERE / f"item0_{name}.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
```

## Appendix F: m1_36.py

```python
"""Builder probe, prompt 36, measurement 1: one correction, each candidate, one accuracy measure.

Usage: python m1_36.py SYSTEM [PREVIOUS_SYSTEM]
       python m1_36.py SYSTEM --jacobi-history

On systems/SYSTEM.npz, every candidate of solvers36.py to each relative
residual level of LEVELS, and today's loop to its own stops. Each result is
read the same way (evaluate): the relative residual ||f - A x|| / ||f|| in
the 2-norm and the max-norm; the corrected faces' mass imbalance (b + A x)
as its worst cell, its absolute sum and its signed sum, each over the
supply's mass flow; the largest error of p' and of the corrected face
velocity against the SuperLU solution.

Today's loop, in a run of its own (--jacobi-history, untimed, so the 15
systems can run in parallel while the timed runs go one at a time): the
committed sweep from p' = 0 with the committed stop's quantity, the largest
weighted change, computed every sweep, up to 5,000,000 sweeps. Recorded and
evaluated as above: the sweep at which that change first falls below 1e-6 Pa
(the committed tolerance) and below 1e-8 Pa (the validation cases' and step
0's), and the sweep at which each relative level is first seen (the residual
checked every 10 sweeps). Its seconds are the sweeps times the timed run's
seconds per sweep, from 2,000 sweeps of the committed loop. The timed run
also runs the loop at the committed cap of 200 sweeps. Writes
m1hist_SYSTEM.json.

Timings: the median of three runs for every solve under two seconds, one run
otherwise; setup (the multigrid hierarchy, pyamg's setup with the CSR
assembly, SuperLU's factorization) apart from the solve. With
PREVIOUS_SYSTEM, MG-PCG and SA-CG are also run with the hierarchy built on
that system (a stale preconditioner; CG still uses the current matrix).
Writes m1_SYSTEM.json.
"""

import json
import math
import sys
import time
import warnings

import numpy as np

from common36 import HERE, PressureCorrector, apply_a, corrected_faces, corrected_imbalance, load_system
from solvers36 import (
    GMG,
    direct_solve,
    gmg_solve,
    jacobi_committed,
    jacobi_pcg,
    mg_pcg,
    pyamg_solve,
)

LEVELS = [1e-1, 1e-2, 1e-4, 1e-6, 1e-8]
warnings.simplefilter("ignore")


class _Shim(PressureCorrector):
    """Just enough of a corrector for its public sweep: the shape it checks against."""

    def __init__(self, shape: tuple[int, int]) -> None:  # noqa: D107
        self._p_shape = shape


def evaluate(s, x: np.ndarray, exact: np.ndarray, f: np.ndarray) -> dict:  # type: ignore[no-untyped-def]
    act = s.active
    c = s.c
    imb = corrected_imbalance(c, s.b, x)
    r = f - apply_a(c, x)
    r[~act] = 0.0
    fn2, fninf = float(np.linalg.norm(f)), float(np.max(np.abs(f)))
    xp, ep = x.copy(), exact.copy()
    if s.needs_pin:
        xp[act] -= xp[s.pin_cell]
        ep[act] -= ep[s.pin_cell]
    u, v = corrected_faces(s, x)
    ue, ve = corrected_faces(s, exact)
    return {
        "rel2": float(np.linalg.norm(r)) / fn2,
        "relinf": float(np.max(np.abs(r))) / fninf,
        "worst_over_supply": float(np.max(np.abs(imb))) / s.supply,
        "worst_kg_s": float(np.max(np.abs(imb))),
        "abs_sum_over_supply": float(np.sum(np.abs(imb))) / s.supply,
        "signed_over_supply": float(np.sum(imb)) / s.supply,
        "p_err_rel": float(np.max(np.abs(xp - ep)) / np.max(np.abs(ep))),
        "face_err_m_s": float(max(np.max(np.abs(u - ue)), np.max(np.abs(v - ve)))),
    }


def jacobi_events(s, f, exact, levels, tols, cap: int) -> dict:  # type: ignore[no-untyped-def]
    """The committed sweep from zero; evaluate where each stop tolerance and each level is first met."""
    from common36 import JACOBI_WEIGHT

    c = s.c
    shim = _Shim(c.a_p.shape)
    active = s.active
    p_prime = np.zeros(c.a_p.shape)
    pending_lev = sorted(levels, reverse=True)
    pending_tol = sorted(tols, reverse=True)
    events: dict[str, dict] = {}
    trace: list[tuple[int, float, float]] = []
    sweep = 0
    diff = math.inf
    for sweep in range(1, cap + 1):
        p_new = shim.sweep(p_prime, c, s.b, JACOBI_WEIGHT)
        diff = float(np.max(np.abs(p_new[active] - p_prime[active])))
        p_prime = p_new
        while pending_tol and diff < pending_tol[0]:
            events[f"stop {pending_tol[0]:.0e} Pa"] = {"sweeps": sweep, "diff": diff, **evaluate(s, p_prime, exact, f)}
            pending_tol.pop(0)
        if sweep % 10 == 0:
            rel = float(np.linalg.norm((f - apply_a(c, p_prime))[active])) / float(np.linalg.norm(f))
            trace.append((sweep, rel, diff))
            while pending_lev and rel < pending_lev[0]:
                events[f"level {pending_lev[0]:.0e}"] = {"sweeps": sweep, "diff": diff, **evaluate(s, p_prime, exact, f)}
                pending_lev.pop(0)
        if not pending_lev and not pending_tol:
            break
    last = evaluate(s, p_prime, exact, f)
    return {"events": events, "trace": trace[:: max(1, len(trace) // 400)],
            "last": {"sweeps": sweep, "diff": diff, **last}}


def timed(fn, repeats: int = 3):  # type: ignore[no-untyped-def]
    """Run fn once; if under two seconds, twice more; return the run with the median total."""
    first = fn()
    res0 = first[0] if isinstance(first, tuple) else first
    if res0.total_s >= 2.0:
        return first
    runs = [first] + [fn() for _ in range(repeats - 1)]
    totals = [(r[0] if isinstance(r, tuple) else r).total_s for r in runs]
    return runs[int(np.argsort(totals)[len(totals) // 2])]


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    name = args[0]
    prev = args[1] if len(args) > 1 else None
    s = load_system(name)
    f = s.rhs()
    singular = s.needs_pin
    if "--jacobi-history" in sys.argv:
        exact = direct_solve(s.c, f, singular, s.pin_cell).x
        t0 = time.perf_counter()
        hist = jacobi_events(s, f, exact, LEVELS, (1e-6, 1e-8), 5_000_000)
        hist["seconds_untimed"] = time.perf_counter() - t0
        (HERE / f"m1hist_{name}.json").write_text(json.dumps(hist))
        print(f"{name} jacobi events {json.dumps(hist['events'])} last {hist['last']}", flush=True)
        return
    out: dict = {"system": name, "meta": s.meta, "active": int(s.active.sum()), "levels": LEVELS, "rows": []}

    def row(cand: str, level, res, extra=None) -> None:  # type: ignore[no-untyped-def]
        entry = {"candidate": cand, "level": level, "iterations": res.iterations,
                 "setup_s": res.setup_s, "solve_s": res.solve_s, "total_s": res.total_s,
                 **evaluate(s, res.x, exact, f), **(extra or {})}
        out["rows"].append(entry)
        print(f"{name} {cand:16s} {str(level):8s} it {res.iterations:8d} setup {res.setup_s:8.4f} "
              f"solve {res.solve_s:9.4f} rel2 {entry['rel2']:.2e} worst/supply {entry['worst_over_supply']:.2e} "
              f"face err {entry['face_err_m_s']:.2e}", flush=True)

    ex = timed(lambda: direct_solve(s.c, f, singular, s.pin_cell))
    exact = ex.x
    row("E_direct", "exact", ex)

    shim = _Shim(s.c.a_p.shape)
    res = jacobi_committed(shim, s.c, s.b, 0.0, 2000)
    out["jacobi_ms_per_sweep"] = 1e3 * res.solve_s / res.iterations
    res = timed(lambda: jacobi_committed(shim, s.c, s.b, 1e-6, 200))
    row("A_jacobi_cap200", "1e-06 Pa", res)

    atol = 0.0
    for level in LEVELS:
        row("B_pcg", level, timed(lambda: jacobi_pcg(s.c, f, level, atol=atol)))
        row("C_gmg", level, timed(lambda: gmg_solve(s.c, f, level, singular)))
        row("C_mg_pcg", level, timed(lambda: mg_pcg(s.c, f, level, singular)))
        for kind, accel, label in (("sa", None, "D_sa"), ("sa", "cg", "D_sa_cg"),
                                   ("sa_jacobi", "cg", "D_sa_jacobi_cg"), ("rs", "cg", "D_rs_cg")):
            res, _ = timed(lambda: pyamg_solve(s.c, f, level, kind, accel))
            row(label, level, res)

    # Convergence histories at the tightest level, for the report's rates.
    out["histories"] = {
        "B_pcg": jacobi_pcg(s.c, f, 1e-10, keep=True).history,
        "C_gmg": gmg_solve(s.c, f, 1e-10, singular, keep=True).history,
        "C_mg_pcg": mg_pcg(s.c, f, 1e-10, singular, keep=True).history,
        "D_sa_cg": pyamg_solve(s.c, f, 1e-10, "sa", "cg", keep=True)[0].history,
        "D_rs_cg": pyamg_solve(s.c, f, 1e-10, "rs", "cg", keep=True)[0].history,
    }
    mg = GMG(s.c, singular)
    out["gmg_levels"] = [list(lv.shape) for lv in mg.levels]
    out["gmg_omegas"] = [lv.omega for lv in mg.levels]

    if prev is not None:
        p = load_system(prev)
        mg_old = GMG(p.c, singular)
        _, ml_old = pyamg_solve(p.c, p.rhs(), 0.5, "sa", "cg")
        for level in (1e-1, 1e-2, 1e-6):
            row("C_mg_pcg_stale", level, timed(lambda: mg_pcg(s.c, f, level, singular, mg=mg_old)),
                {"hierarchy_from": prev})
            res, _ = timed(lambda: pyamg_solve(s.c, f, level, "sa", "cg", ml=ml_old))
            row("D_sa_cg_stale", level, res, {"hierarchy_from": prev})

    (HERE / f"m1_{name}.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
```

## Appendix G: outer36.py

```python
"""Builder probe, prompt 36, measurement 2: how tightly the outer loop needs the correction.

Usage: python outer36.py NX NY MODE [N_OUTER] [MU_FACTOR]

The product room on NX x NY under T3, laminar (zero eddy-viscosity field) at
air's viscosity unless MU_FACTOR is given, ten momentum sweeps per outer
iteration, alpha_velocity 0.5, from rest. The solver stops by error_estimate
(iteration_error_tol 1e-6, mass_imbalance_tol 1e-4 rho V_min / t_end, ADR-011
G) or at N_OUTER (default 20,000); the outer iteration at which the
velocity_step rule (residual below 1e-6) would have stopped is recorded on
the way, with the field there. One run gives both stops, because the rule
does not change the path.

MODE is one of
    direct           SuperLU, the exact correction (the tightest run)
    pcg:RTOL         Jacobi-preconditioned CG to the relative residual RTOL
    mgpcg:RTOL       MG-preconditioned CG (solvers36.GMG) to RTOL
    gmg:RTOL         Galerkin V(2,2) cycles to RTOL
    sacg:RTOL        pyamg smoothed aggregation with CG to RTOL
    rscg:RTOL        pyamg Ruge-Stuben with CG to RTOL (added for measurement 3's
                     in-run check)
    jacobi200        today's committed loop at the committed cap 200, 1e-6 Pa
    jacobi40k        today's loop at step 0's cap 40,000, 1e-8 Pa (test 34b's A-sw
                     on 40x15, reproduced bitwise to its velocity_step stop)
Every Krylov mode also stops at an absolute 2-norm of 1e-13 times the supply's
mass flow, a floor far below any tolerance here, so a right-hand side at
rounding level cannot run to the iteration cap. Writes m2_NXxNY_MODE.json and
.npz through capture36.run_room.
"""

import sys

from capture36 import run_room, stopping_for
from common36 import product_t3
from solvers36 import gmg_solve, jacobi_pcg, mg_pcg, pyamg_solve, direct_solve

import numpy as np


def make_solve(mode: str, supply: float):  # type: ignore[no-untyped-def]
    atol = 1e-13 * supply

    def rhs(b, active, needs_pin):  # type: ignore[no-untyped-def]
        f = -b.copy()
        if needs_pin:
            f[active] -= f[active].mean()
        f[~active] = 0.0
        return f

    if mode == "direct":
        def solve(c, b, active, needs_pin):  # type: ignore[no-untyped-def]
            pin = tuple(int(v) for v in np.argwhere(active)[0])
            return direct_solve(c, rhs(b, active, needs_pin), needs_pin, pin).x, 1
        return solve
    kind, _, tol = mode.partition(":")
    rtol = float(tol)
    if kind == "pcg":
        def solve(c, b, active, needs_pin):  # type: ignore[no-untyped-def]
            res = jacobi_pcg(c, rhs(b, active, needs_pin), rtol, atol=atol)
            return res.x, res.iterations
    elif kind == "mgpcg":
        def solve(c, b, active, needs_pin):  # type: ignore[no-untyped-def]
            res = mg_pcg(c, rhs(b, active, needs_pin), rtol, needs_pin)
            return res.x, res.iterations
    elif kind == "gmg":
        def solve(c, b, active, needs_pin):  # type: ignore[no-untyped-def]
            res = gmg_solve(c, rhs(b, active, needs_pin), rtol, needs_pin)
            return res.x, res.iterations
    elif kind in ("sacg", "rscg"):
        amg = "sa" if kind == "sacg" else "rs"

        def solve(c, b, active, needs_pin):  # type: ignore[no-untyped-def]
            res, _ = pyamg_solve(c, rhs(b, active, needs_pin), rtol, amg, "cg")
            return res.x, res.iterations
    else:
        raise SystemExit(f"unknown mode {mode}")
    return solve


def main() -> None:
    nx, ny, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    n_outer = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
    mu_factor = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    stop = stopping_for(nx, ny, 8.0, 3.0, 1.2, 60.0)
    cap, tol = {"jacobi200": (200, 1e-6), "jacobi40k": (40000, 1e-8)}.get(mode, (40000, 1e-8))
    tag = f"m2_{nx}x{ny}_{mode.replace(':', '_')}" + (f"_mu{mu_factor:g}" if mu_factor != 1.0 else "")
    room = product_t3(tag, nx, ny, n_outer, mu_factor=mu_factor, p_cap=cap, p_tol=tol, stopping=stop)
    solve = None if mode.startswith("jacobi") else make_solve(mode, room.supply)
    run_room(room, solve, tag, print_every=100)


if __name__ == "__main__":
    main()
```

## Appendix H: run36.sh

```sh
#!/bin/sh
# Prompt 36 (ECR-003): the launchers. Every run single-threaded in BLAS, so the
# seconds compare one core against one core. From results/builder36/:
#   sh run36.sh selftest        the harness controls (section 3)
#   sh run36.sh capture         item 0's runs; product200 runs on in the background
#   sh run36.sh item0           item 0's checks on the captured systems
#   sh run36.sh m1              measurement 1's timed runs, one at a time
#   sh run36.sh m1hist          its untimed Jacobi histories, in parallel
#   sh run36.sh m2 NX NY        measurement 2's set on one grid, in parallel
#   sh run36.sh m2b NX NY LEVEL the solver-independence runs at one level
# Added after the first measurement 2 runs (report, section 8), in the order run:
#   sh run36.sh restart         restart36.py: each converged 40x15 state continued
#                               with the other correction
#   sh run36.sh control         control36.py: the exact correction on other paths
#   sh run36.sh fallback        80x30 at ten times air's viscosity (section 2.6)
#   sh run36.sh re90            80x30 at a thousand times, with the repeats
#   sh run36.sh repeats80       the multigrid and pyamg repeats on 80x30, air
#   sh run36.sh diag            freeze36.py and stall36.py
#   sh run36.sh m3              m3_36.py and the 1,000-iteration in-run check
PY=venv36/Scripts/python.exe
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs
case "$1" in
selftest)
    $PY selftest36.py all > logs/selftest36.log 2>&1 ;;
capture)
    nohup $PY capture36.py product200 > logs/product200.log 2>&1 &
    for r in product40 cavity80 annex180; do
        $PY capture36.py $r > logs/$r.log 2>&1 &
    done
    $PY capture36.py channel80 1001 1,100,569 > logs/channel80.log 2>&1 &
    wait ;;
item0)
    unset OPENBLAS_NUM_THREADS OMP_NUM_THREADS MKL_NUM_THREADS
    for r in product200 product40 cavity80 annex180; do
        $PY item0_36.py ${r}_it1 ${r}_it100 ${r}_it1000 > logs/item0_$r.log 2>&1
    done
    $PY item0_36.py channel80_it1 channel80_it100 channel80_it569 > logs/item0_channel80.log 2>&1 ;;
m1)
    for r in product200 cavity80 channel80 annex180 product40; do
        last=1000; [ "$r" = channel80 ] && last=569
        $PY m1_36.py ${r}_it1 > logs/m1_${r}_it1.log 2>&1
        $PY m1_36.py ${r}_it100 ${r}_it1 > logs/m1_${r}_it100.log 2>&1
        $PY m1_36.py ${r}_it$last ${r}_it100 > logs/m1_${r}_it$last.log 2>&1
    done ;;
m1hist)
    for r in product200 cavity80 channel80 annex180 product40; do
        last=1000; [ "$r" = channel80 ] && last=569
        for k in 1 100 $last; do
            $PY m1_36.py ${r}_it$k --jacobi-history > logs/m1hist_${r}_it$k.log 2>&1 &
        done
    done
    wait ;;
m2)
    for m in direct pcg:3e-1 pcg:1e-1 pcg:1e-2 pcg:1e-4 pcg:1e-8 jacobi200 jacobi40k; do
        tag=$(echo "$m" | tr ':' '_')
        $PY outer36.py "$2" "$3" "$m" > "logs/m2_$2x$3_$tag.log" 2>&1 &
    done
    wait ;;
m2b)
    for k in mgpcg gmg sacg; do
        $PY outer36.py "$2" "$3" "$k:$4" > "logs/m2_$2x$3_${k}_$4.log" 2>&1 &
    done
    wait ;;
restart)
    $PY restart36.py 40 15 direct pcg:1e-2 3000 > logs/restart_a.log 2>&1 &
    $PY restart36.py 40 15 pcg:1e-2 direct 3000 > logs/restart_b.log 2>&1 &
    $PY restart36.py 40 15 direct pcg:1e-4 3000 > logs/restart_c.log 2>&1 &
    wait ;;
control)
    for a in "9 0.5" "11 0.5" "10 0.45" "10 0.55"; do
        $PY control36.py 40 15 $a > "logs/m2c_40x15_$(echo $a | tr ' ' '_').log" 2>&1 &
    done
    wait ;;
fallback)
    for m in direct pcg:3e-1 pcg:1e-1 pcg:1e-2 pcg:1e-4 pcg:1e-8 jacobi200 jacobi40k; do
        tag=$(echo "$m" | tr ':' '_')
        $PY outer36.py 80 30 "$m" 20000 10 > "logs/m2_80x30_${tag}_mu10.log" 2>&1 &
    done
    wait ;;
re90)
    for m in direct pcg:3e-1 pcg:1e-1 pcg:1e-2 pcg:1e-4 pcg:1e-8 jacobi200 mgpcg:1e-1 gmg:1e-1 sacg:1e-1 mgpcg:1e-2 gmg:1e-2 sacg:1e-2; do
        tag=$(echo "$m" | tr ':' '_')
        $PY outer36.py 80 30 "$m" 20000 1000 > "logs/m2_80x30_${tag}_mu1000.log" 2>&1 &
    done
    wait ;;
repeats80)
    for m in mgpcg:1e-1 gmg:1e-1 sacg:1e-1 mgpcg:1e-2 gmg:1e-2 sacg:1e-2; do
        tag=$(echo "$m" | tr ':' '_')
        $PY outer36.py 80 30 "$m" 3000 > "logs/m2_80x30_${tag}.log" 2>&1 &
    done
    wait ;;
diag)
    $PY freeze36.py > logs/freeze80.log 2>&1
    $PY item0_36.py freeze80_it5231 > logs/item0_freeze80.log 2>&1
    $PY stall36.py pcg:1e-4 1500 > logs/stall_pcg.log 2>&1 &
    $PY stall36.py direct 588 > logs/stall_direct.log 2>&1 &
    wait
    $PY stall36.py direct 2822 40 15 1 > logs/stall_direct_40.log 2>&1 ;;
m3)
    for r in product200 annex180; do
        for lev in 1e-8 1e-2; do $PY m3_36.py $r $lev 3000 13000 6000 > logs/m3_${r}_$lev.log 2>&1; done
    done
    for m in pcg:1e-8 rscg:1e-8 direct; do
        tag=$(echo "$m" | tr ':' '_')
        $PY outer36.py 200 75 "$m" 1000 > "logs/m3_200x75_$tag.log" 2>&1 &
    done
    wait ;;
esac
```

## Appendix I: appendix36.py

```python
"""Builder probe, prompt 36: write the report's appendices from the probe files, verbatim.

Usage: python appendix36.py
Replaces everything from the line '## Appendix A' to the end of
docs/reports/pressure_solver_ecr003.md with the files below, each in a fence.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE.parents[1] / "docs" / "reports" / "pressure_solver_ecr003.md"
FILES = [
    ("A", "common36.py", "python"),
    ("B", "solvers36.py", "python"),
    ("C", "selftest36.py", "python"),
    ("D", "capture36.py", "python"),
    ("E", "item0_36.py", "python"),
    ("F", "m1_36.py", "python"),
    ("G", "outer36.py", "python"),
    ("H", "run36.sh", "sh"),
    ("I", "appendix36.py", "python"),
    ("J", "table36.py", "python"),
    ("K", "restart36.py", "python"),
    ("L", "control36.py", "python"),
    ("M", "freeze36.py", "python"),
    ("N", "stall36.py", "python"),
    ("O", "m3_36.py", "python"),
    ("P", "hist_table36.py", "python"),
]


def main() -> None:
    text = REPORT.read_text(encoding="utf-8")
    cut = text.find("\n## Appendix A")
    body = text if cut < 0 else text[:cut]
    parts = [body.rstrip("\n") + "\n"]
    for letter, name, lang in FILES:
        path = HERE / name
        if not path.exists():
            continue
        src = path.read_text(encoding="utf-8").rstrip("\n")
        parts.append(f"\n## Appendix {letter}: {name}\n\n```{lang}\n{src}\n```\n")
    REPORT.write_text("".join(parts), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
```

## Appendix J: table36.py

```python
"""Builder probe, prompt 36: the report's tables, from the records the probes wrote.

Usage: python table36.py item0 | m1 SYSTEM | m1all | m2 NXxNY | runs NAME... | m3
Prints markdown. Reads only the JSON and NPZ files beside it.
"""

import json
import math
import sys

import numpy as np

from common36 import HERE, Mesh, SimConfig, product_raw
from src.mesh import SOLID

ROOMS = ["product200", "product40", "cavity80", "channel80", "annex180"]


def load(name: str) -> dict:
    return json.loads((HERE / f"{name}.json").read_text())


def g(x: float, n: int = 2) -> str:
    """A number in short scientific or plain form."""
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    if x == 0:
        return "0"
    if abs(x) >= 1e4 or abs(x) < 1e-2:
        return f"{x:.{n - 1}e}"
    return f"{x:.{n + 1}g}"


def item0() -> None:
    print("| System | Active cells | Symmetric to the bit | max abs(A - A^T) / max A | Strict rows | Components (with a strict row) "
          "| lambda_min(A) | lambda_min(D^-1 A) | lambda_max(D^-1 A) | Cholesky (n, s) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in ROOMS:
        for k in (1, 100, 569, 1000):
            path = HERE / f"item0_{r}_it{k}.json"
            if not path.exists():
                continue
            d = json.loads(path.read_text())
            sym = "yes" if d["sym_bitwise_ew"] and d["sym_bitwise_ns"] else "NO"
            chol = f"{d['cholesky']} ({d['cholesky_n']}, {d['cholesky_seconds']:.1f})"
            print(f"| {r} outer {k} | {d['active']} | {sym} | {g(d['sym_matrix_max_rel'])} | {d['strict_rows']} "
                  f"| {d['components']} ({d['components_with_strict_row']}) | {g(d['eig_A_smallest'][0])} "
                  f"| {g(d['eig_DinvA_smallest'][0])} | {g(d['eig_DinvA_largest'], 4)} | {chol} |")
    print()
    print("| System | A 1 = 0 (max abs / max a_P) | abs(sum b) / sum abs(b) | three smallest eigenvalues of A | identity, exact p' | identity, random p' |")
    print("|---|---|---|---|---|---|")
    for r in ROOMS:
        for k in (1, 100, 569, 1000):
            path = HERE / f"item0_{r}_it{k}.json"
            if not path.exists():
                continue
            d = json.loads(path.read_text())
            print(f"| {r} outer {k} | {g(d.get('ones_null_max_rel'))} | {g(d.get('compatibility'))} "
                  f"| {', '.join(g(x) for x in d['eig_A_smallest'])} | {g(d['identity_direct_max_over_max_b'])} "
                  f"| {g(d['identity_random_max_over_max_b'])} |")


def m1(system: str) -> None:
    d = load(f"m1_{system}")
    hist_path = HERE / f"m1hist_{system}.json"
    print(f"**{system}** ({d['active']} unknowns; GMG levels {d['gmg_levels']})\n")
    print("| Candidate | Level | Iterations | Setup ms | Solve ms | Total ms | Relative residual | Worst cell / supply "
          "| Net outflow / supply | p' error | Face error m/s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in d["rows"]:
        print(f"| {row['candidate']} | {row['level']} | {row['iterations']} | {1e3 * row['setup_s']:.1f} "
              f"| {1e3 * row['solve_s']:.1f} | {1e3 * row['total_s']:.1f} | {g(row['rel2'])} | {g(row['worst_over_supply'])} "
              f"| {g(row['signed_over_supply'])} | {g(row['p_err_rel'])} | {g(row['face_err_m_s'])} |")
    if hist_path.exists():
        h = json.loads(hist_path.read_text())
        ms = d["jacobi_ms_per_sweep"]
        print()
        print("Today's loop from zero (untimed run; seconds at the timed run's "
              f"{ms:.3f} ms per sweep):")
        print()
        print("| Event | Sweeps | Seconds | Largest weighted change (Pa) | Relative residual | Worst cell / supply "
              "| Net outflow / supply | p' error | Face error m/s |")
        print("|---|---|---|---|---|---|---|---|---|")
        order = ["stop 1e-06 Pa", "stop 1e-08 Pa", "level 1e-01", "level 1e-02", "level 1e-04", "level 1e-06", "level 1e-08"]
        for key in sorted(h["events"], key=lambda k: h["events"][k]["sweeps"]):
            e = h["events"][key]
            print(f"| {key} | {e['sweeps']} | {e['sweeps'] * ms / 1e3:.1f} | {g(e['diff'])} | {g(e['rel2'])} "
                  f"| {g(e['worst_over_supply'])} | {g(e['signed_over_supply'])} | {g(e['p_err_rel'])} | {g(e['face_err_m_s'])} |")
        missing = [k for k in order if k not in h["events"]]
        if missing:
            last = h["last"]
            print(f"| not met by sweep {last['sweeps']}: {', '.join(missing)} | | | {g(last['diff'])} | {g(last['rel2'])} "
                  f"| {g(last['worst_over_supply'])} | {g(last['signed_over_supply'])} | {g(last['p_err_rel'])} | {g(last['face_err_m_s'])} |")


def summary_m1() -> None:
    """One line per system: total ms per candidate at each level, and Jacobi's stops."""
    levels = [0.1, 0.01, 1e-4, 1e-6, 1e-8]
    cands = ["B_pcg", "C_gmg", "C_mg_pcg", "D_sa", "D_sa_cg", "D_sa_jacobi_cg", "D_rs_cg"]
    for lev in levels:
        print(f"\nLevel {lev:g}: total ms (iterations)\n")
        print("| System | " + " | ".join(cands) + " | E_direct |")
        print("|---" * (len(cands) + 2) + "|")
        for r in ROOMS:
            for k in (1, 100, 569, 1000):
                path = HERE / f"m1_{r}_it{k}.json"
                if not path.exists():
                    continue
                d = json.loads(path.read_text())
                cells = []
                for c in cands:
                    row = [x for x in d["rows"] if x["candidate"] == c and x["level"] == lev]
                    if row:
                        x = row[0]
                        ok = "" if x["rel2"] <= lev * 1.0001 else " NOT REACHED"
                        cells.append(f"{1e3 * x['total_s']:.1f} ({x['iterations']}){ok}")
                    else:
                        cells.append("-")
                e = [x for x in d["rows"] if x["candidate"] == "E_direct"][0]
                print(f"| {r} {k} | " + " | ".join(cells) + f" | {1e3 * e['total_s']:.1f} |")


def rate_estimate(steps: list[float], window: int = 100) -> tuple[float, float]:
    """The error_estimate rule's estimate (m/s) and rho_hat from the last `window` steps."""
    s = np.array(steps[-window:])
    if s.size < window or not np.all(s > 0):
        return math.inf, math.nan
    x = np.arange(s.size) - (s.size - 1) / 2.0
    rho = math.exp(float(x @ np.log(s)) / float(x @ x))
    if not 0.0 < rho < 1.0:
        return math.inf, rho
    return float(s[-1]) * rho / (1.0 - rho), rho


def m2(grid: str, names: list[str] | None = None, base: str | None = None) -> None:
    nx, ny = (int(v) for v in grid.split("x"))
    cfg = SimConfig.from_dict(product_raw(nx, ny))
    fluid = Mesh(cfg).cell_type != SOLID
    if names is None:
        names = sorted(p.stem for p in HERE.glob(f"m2_{grid}_*.json"))
    base = base or f"m2_{grid}_direct"
    ref = load(base) if (HERE / f"{base}.json").exists() else None
    ref_f = np.load(HERE / f"{base}.npz") if (HERE / f"{base}.npz").exists() else None
    print(f"| Run | Stop | velocity_step at | error_estimate at | Seconds | ms per outer | Inner iterations, median [max] "
          "| Relative residual reached, median | Field vs direct at velocity_step (m/s) | its error + direct's "
          "| Field vs direct at error_estimate (m/s) | Worst cell at the stop / tol |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    tol = 1e-4 * 1.2 * (8.0 / nx) * (3.0 / ny) / 60.0
    for name in names:
        d = load(name)
        n = d["outer"]
        ee = n if d["stop"] == "error_estimate_and_continuity" else None
        vs = d["velocity_step_outer"]
        inner = np.array(d["inner"])
        rel = np.array(d["rel"])
        row_vs = row_ee = err_sum = "-"
        if ref is not None and ref_f is not None and name != base:
            f = np.load(HERE / f"{name}.npz") if (HERE / f"{name}.npz").exists() else None
            if f is not None and vs and ref["velocity_step_outer"] and f["u_vs"].size > 1:
                diff = max(np.max(np.abs(f["u_vs"] - ref_f["u_vs"])[fluid]), np.max(np.abs(f["v_vs"] - ref_f["v_vs"])[fluid]))
                e1, _ = rate_estimate(d["step"][:vs])
                e2, _ = rate_estimate(ref["step"][: ref["velocity_step_outer"]])
                row_vs, err_sum = g(float(diff)), g(e1 + e2)
            if f is not None and ee and ref["stop"] == "error_estimate_and_continuity":
                diff = max(np.max(np.abs(f["u"] - ref_f["u"])[fluid]), np.max(np.abs(f["v"] - ref_f["v"])[fluid]))
                row_ee = g(float(diff))
        print(f"| {name.replace('m2_' + grid + '_', '').replace('m2c_' + grid + '_', 'exact, ')} | {d['stop']} | {vs or '-'} | {ee or '-'} | {d['seconds']:.0f} "
              f"| {1e3 * d['seconds'] / n:.1f} | {int(np.median(inner))} [{int(inner.max())}] | {g(float(np.median(rel)))} "
              f"| {row_vs} | {err_sum} | {row_ee} | {g(d['worst'][-1] / tol)} |")


def runs(names: list[str]) -> None:
    for name in names:
        d = load(name)
        n = d["outer"]
        e, rho = rate_estimate(d["step"])
        print(f"{name}: stop {d['stop']} outer {n} velocity_step at {d['velocity_step_outer']} seconds {d['seconds']:.1f} "
              f"momentum ms/outer median {1e3 * np.median(d['momentum_s']):.2f} pressure ms/outer median "
              f"{1e3 * np.median(d['pressure_s']):.2f} last residual {d['residual'][-1]:.3e} max speed end "
              f"{d['max_speed'][-1]:.3f} least residual {min(d['residual']):.3e} at {int(np.argmin(d['residual']))} "
              f"estimate at end {e:.2e} rho {rho:.5f} worst end {d['worst'][-1]:.2e}")


def main() -> None:
    cmd = sys.argv[1]
    if cmd == "item0":
        item0()
    elif cmd == "m1":
        m1(sys.argv[2])
    elif cmd == "m1all":
        summary_m1()
    elif cmd == "m2":
        args = sys.argv[3:]
        base = None
        if args and args[0].startswith("--base="):
            base, args = args[0].split("=", 1)[1], args[1:]
        m2(sys.argv[2], args or None, base)
    elif cmd == "runs":
        runs(sys.argv[2:])


if __name__ == "__main__":
    main()


def m1mean(room: str, last: int = 1000) -> None:
    """Per candidate and level, over the room's three systems: total ms mean [min, max], iterations, accuracy."""
    systems = [f"{room}_it{k}" for k in (1, 100, last)]
    data = [load(f"m1_{s}") for s in systems]
    cands = ["B_pcg", "C_gmg", "C_mg_pcg", "D_sa", "D_sa_cg", "D_sa_jacobi_cg", "D_rs_cg"]
    print(f"**{room}**, outer 1, 100 and {last}\n")
    print("| Candidate | Level | Total ms, mean [min, max] | Setup ms, mean | Iterations | Relative residual, largest "
          "| Net outflow / supply, largest abs | p' error, largest | Face error m/s, largest |")
    print("|---|---|---|---|---|---|---|---|---|")
    e = [[x for x in d["rows"] if x["candidate"] == "E_direct"][0] for d in data]
    print(f"| E_direct | exact | {1e3 * np.mean([x['total_s'] for x in e]):.1f} [{1e3 * min(x['total_s'] for x in e):.1f}, "
          f"{1e3 * max(x['total_s'] for x in e):.1f}] | {1e3 * np.mean([x['setup_s'] for x in e]):.1f} | 1 | - | - | 0 | 0 |")
    c2 = [[x for x in d["rows"] if x["candidate"] == "A_jacobi_cap200"][0] for d in data]
    print(f"| A_jacobi_cap200 | cap 200 | {1e3 * np.mean([x['total_s'] for x in c2]):.1f} | 0 | 200 "
          f"| {g(max(x['rel2'] for x in c2))} | {g(max(abs(x['signed_over_supply']) for x in c2))} "
          f"| {g(max(x['p_err_rel'] for x in c2))} | {g(max(x['face_err_m_s'] for x in c2))} |")
    for lev in [0.1, 0.01, 1e-4, 1e-6, 1e-8]:
        for c in cands:
            rows = [[x for x in d["rows"] if x["candidate"] == c and x["level"] == lev][0] for d in data]
            tot = [1e3 * x["total_s"] for x in rows]
            its = ", ".join(str(x["iterations"]) for x in rows)
            miss = [s for s, x in zip((1, 100, last), rows) if x["rel2"] > lev * 1.0001]
            note = f" (not reached at outer {', '.join(str(m) for m in miss)})" if miss else ""
            print(f"| {c}{note} | {lev:g} | {np.mean(tot):.1f} [{min(tot):.1f}, {max(tot):.1f}] "
                  f"| {np.mean([1e3 * x['setup_s'] for x in rows]):.1f} | {its} | {g(max(x['rel2'] for x in rows))} "
                  f"| {g(max(abs(x['signed_over_supply']) for x in rows))} | {g(max(x['p_err_rel'] for x in rows))} "
                  f"| {g(max(x['face_err_m_s'] for x in rows))} |")


if __name__ == "__main__" and sys.argv[1] == "m1mean":
    m1mean(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1000)
```

## Appendix K: restart36.py

```python
"""Builder probe, prompt 36, a diagnostic added after measurement 2's first runs.

Usage: python restart36.py NX NY MODE_A MODE_B N_MORE

Runs the measurement 2 room to its error_estimate stop with correction
MODE_A (outer36's modes), then continues from that state, its faces and its
pressure, for N_MORE outer iterations with correction MODE_B, through the
same calls solve_steady makes (the T3 extrapolation, predict, correct), and
records how far the cell-centred velocity moves from the MODE_A stop and the
residual on the way. If the MODE_A state is a fixed point of the MODE_B
iteration it stays within the stop's iteration error; if the two iterations
have different fixed points it moves to MODE_B's. Writes
restart_NXxNY_A_B.json and .npz.
"""

import json
import sys

import numpy as np

from capture36 import stopping_for
from common36 import HERE, install_corrector, product_t3
from outer36 import make_solve
from src.mesh import SOLID
from src.staggered import to_cell_centers


def main() -> None:
    nx, ny, mode_a, mode_b, n_more = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5])
    stop = stopping_for(nx, ny, 8.0, 3.0, 1.2, 60.0)
    room = product_t3("restart", nx, ny, 20000, stopping=stop)
    corr = install_corrector(room, make_solve(mode_a, room.supply), measure_residual=False)
    solver = room.solver
    u_c, v_c, p = solver.solve_steady()
    n_a = len(solver.residual_history)
    fv = solver.face_velocities
    u, v = fv.u.copy(), fv.v.copy()
    p = p.copy()
    u0, v0 = u_c.copy(), v_c.copy()
    fluid = room.mesh.cell_type != SOLID
    corr.solve = make_solve(mode_b, room.supply)
    ref = solver.reference_velocity
    drift, res = [], []
    prev_u, prev_v = u0, v0
    for _ in range(n_more):
        solver._extrapolate_outlets(u, v)
        pred = solver._predictor.predict(u, v, p)
        out = corr.correct(pred, p)
        u, v, p = out.u, out.v, out.p
        uc, vc = to_cell_centers(u, v)
        res.append(float(max(np.max(np.abs(uc - prev_u)[fluid]), np.max(np.abs(vc - prev_v)[fluid]))) / ref)
        drift.append(float(max(np.max(np.abs(uc - u0)[fluid]), np.max(np.abs(vc - v0)[fluid]))))
        prev_u, prev_v = uc, vc
    tag = f"restart_{nx}x{ny}_{mode_a.replace(':', '_')}_{mode_b.replace(':', '_')}"
    out_d = {"mode_a": mode_a, "mode_b": mode_b, "stop_a": solver.stop_reason, "outer_a": n_a,
             "drift": drift, "residual": res}
    (HERE / f"{tag}.json").write_text(json.dumps(out_d))
    np.savez(HERE / f"{tag}.npz", u_a=u0, v_a=v0, u_b=prev_u, v_b=prev_v)
    print(f"{tag}: A stopped {solver.stop_reason} at {n_a}; after {n_more} with B the drift is {drift[-1]:.3e} m/s "
          f"(largest {max(drift):.3e}), last residual {res[-1]:.3e}", flush=True)


if __name__ == "__main__":
    main()
```

## Appendix L: control36.py

```python
"""Builder probe, prompt 36, a control added after measurement 2's first runs.

Usage: python control36.py NX NY SWEEPS ALPHA_U

The measurement 2 room with the exact correction (SuperLU) and a different
path to the same steady equations: SWEEPS momentum sweeps per outer
iteration and alpha_velocity ALPHA_U in place of 10 and 0.5. Neither enters
the steady discrete equations, only the path to them, so the difference of
its converged field from m2_NXxNY_direct's measures how much the path alone
selects among steady solutions. Writes m2c_NXxNY_swSWEEPS_aALPHA_U.json and
.npz through capture36.run_room.
"""

import sys

from capture36 import run_room, stopping_for
from common36 import product_t3
from outer36 import make_solve


def main() -> None:
    nx, ny, sweeps, alpha = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
    stop = stopping_for(nx, ny, 8.0, 3.0, 1.2, 60.0)
    tag = f"m2c_{nx}x{ny}_sw{sweeps}_a{alpha:g}"
    room = product_t3(tag, nx, ny, 20000, sweeps=sweeps, alpha_u=alpha, stopping=stop)
    run_room(room, make_solve("direct", room.supply), tag, print_every=500)


if __name__ == "__main__":
    main()
```

## Appendix M: freeze36.py

```python
"""Builder probe, prompt 36, a diagnostic added after measurement 2's runs.

Usage: python freeze36.py

Reruns m2_80x30_jacobi200 (the committed loop, cap 200 and 1e-6 Pa, on the
80x30 room at air's viscosity) to outer 5,233 and saves the p' system at outer
5,231, the iteration at which velocity_step first held there, so item0_36.py
can say whether the system the committed loop froze on is singular. Writes
freeze80.json and systems/freeze80_it5231.npz through capture36.run_room.
"""

from capture36 import run_room, stopping_for
from common36 import product_t3

stop = stopping_for(80, 30, 8.0, 3.0, 1.2, 60.0)
room = product_t3("freeze80", 80, 30, 5233, p_cap=200, p_tol=1e-6, stopping=stop)
run_room(room, None, "freeze80", capture_at=(5231,), print_every=500)
```

## Appendix N: stall36.py

```python
"""Builder probe, prompt 36, a diagnostic added after measurement 2's runs.

Usage: python stall36.py MODE N [NX NY MU_FACTOR]

The measurement 2 room, by default 80x30 at a thousand times air's viscosity
(the Re 90 rung), with correction MODE for N outer iterations, saving the p' system at outer N - 1
and the pressure field's mean over the cells with an equation at every outer
iteration, so the stalled state's right-hand side can be located and a
uniform drift of the pressure seen. Writes stall_MODE.json and
systems/stall_MODE_it(N-1).npz.
"""

import json
import sys

import numpy as np

from capture36 import stopping_for
from common36 import HERE, install_corrector, product_t3
from outer36 import make_solve

mode, n = sys.argv[1], int(sys.argv[2])
nx, ny, mu = (int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])) if len(sys.argv) > 5 else (80, 30, 1000.0)
tag = f"stall_{mode.replace(':', '_')}" + ("" if (nx, ny, mu) == (80, 30, 1000.0) else f"_{nx}x{ny}_mu{mu:g}")
room = product_t3(tag, nx, ny, n, mu_factor=mu, stopping=stopping_for(nx, ny, 8.0, 3.0, 1.2, 60.0))
corr = install_corrector(room, make_solve(mode, room.supply), capture_at=(n - 1,), capture_name=tag)
means, res = [], []
fluid = room.mesh.cell_type != 1


def cb(state) -> None:  # type: ignore[no-untyped-def]
    means.append(float(np.mean(state.p[fluid])))
    res.append(float(state.residual))


room.solver.solve_steady(on_iteration=cb)
(HERE / f"{tag}.json").write_text(json.dumps({"mode": mode, "p_mean": means, "residual": res,
                                              "stop": room.solver.stop_reason}))
print(tag, room.solver.stop_reason, len(res), "p mean at", [round(means[k], 6) for k in (0, len(means) // 2, -2, -1)])
```

## Appendix O: m3_36.py

```python
"""Builder probe, prompt 36, measurement 3: the cost of one steady solve, projected.

Usage: python m3_36.py ROOM LEVEL OUTER_LO OUTER_HI [OUTER_MID]

ROOM is product200 or annex180. Per outer iteration:

    momentum seconds    the momentum stage (T3 extrapolation and ten sweeps) over
                        30 outer iterations of the room from rest with SuperLU,
                        timed here, one process, so the seconds are not the
                        capture run's, which ran beside other probes
    corrector overhead  the committed corrector's work besides the solve, timed
                        here on the room's captured systems: coefficients,
                        mass_imbalance, _face_d and the face and pressure update
                        (median of 20)
    correction seconds  measurement 1's total (setup and solve) at LEVEL, the
                        median over the room's three captured systems; SuperLU's
                        factorization and solve; for today's loop, its 1e-6 Pa
                        stop's sweeps (the untimed run) times the timed seconds per
                        sweep, and its cap of 200, each the median of the three

and a steady solve is that times OUTER_LO, OUTER_MID and OUTER_HI outer
iterations, the range the report states with its sources. Prints a markdown
table and writes m3_ROOM_LEVEL.json.
"""

import json
import sys
import time

import numpy as np

from common36 import HERE, SYSTEMS, load_system

CANDIDATES = ["B_pcg", "C_gmg", "C_mg_pcg", "D_sa", "D_sa_cg", "D_sa_jacobi_cg", "D_rs_cg"]


def overhead(name: str) -> float:
    """Median seconds of the corrector's non-solve work on a captured system."""
    s = load_system(name)
    z = np.load(SYSTEMS / f"{name}.npz")
    from common36 import PressureCorrector

    class Shim(PressureCorrector):
        def __init__(self) -> None:  # noqa: D107
            self._p_shape = s.c.a_p.shape
            self._u_shape = z["a_p_u"].shape
            self._v_shape = z["a_p_v"].shape
            self._rho = s.rho
            self._alpha_p = 0.3
            self._solid = s.solid
            self._out_left, self._out_right = z["out_left"], z["out_right"]
            self._out_bottom, self._out_top = z["out_bottom"], z["out_top"]

            class M:
                dx_cell = s.dx_cell
                dy_cell = s.dy_cell

            self._mesh = M()

    sh = Shim()
    p = np.zeros(s.c.a_p.shape)
    x = np.zeros(s.c.a_p.shape)
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        c = sh.coefficients(z["a_p_u"], z["a_p_v"])
        b = sh.mass_imbalance(z["u_star"], z["v_star"])
        active = c.a_p > 0.0
        d_u, d_v = sh._face_d(z["a_p_u"], z["a_p_v"])
        padded = np.zeros((p.shape[0] + 2, p.shape[1] + 2))
        padded[1:-1, 1:-1] = x
        u = z["u_star"] - d_u * (padded[1:-1, 1:] - padded[1:-1, :-1])
        v = z["v_star"] - d_v * (padded[1:, 1:-1] - padded[:-1, 1:-1])
        p_next = p.copy()
        p_next[active] += 0.3 * x[active]
        times.append(time.perf_counter() - t0)
        del b, u, v
    return float(np.median(times))


def main() -> None:
    room, level = sys.argv[1], float(sys.argv[2])
    lo, hi = int(sys.argv[3]), int(sys.argv[4])
    mid = int(sys.argv[5]) if len(sys.argv) > 5 else int(round(np.sqrt(lo * hi)))
    from common36 import annex20, install_corrector, product_t3
    from capture36 import exact_solve

    built = product_t3("m3", 200, 75, 30) if room == "product200" else annex20("m3", 180, 60, 30)
    install_corrector(built, exact_solve, measure_residual=False)
    built.solver.solve_steady()
    momentum = built.solver.stage_seconds["momentum"] / 30
    systems = [f"{room}_it{k}" for k in (1, 100, 1000)]
    over = float(np.median([overhead(n) for n in systems]))
    rows = []
    for cand in CANDIDATES:
        vals, its, ok = [], [], True
        for n in systems:
            d = json.loads((HERE / f"m1_{n}.json").read_text())
            r = [x for x in d["rows"] if x["candidate"] == cand and x["level"] == level][0]
            vals.append(r["total_s"])
            its.append(r["iterations"])
            ok &= r["rel2"] <= level * 1.0001
        rows.append((cand, float(np.median(vals)), its, ok))
    direct = [[x for x in json.loads((HERE / f"m1_{n}.json").read_text())["rows"] if x["candidate"] == "E_direct"][0]
              for n in systems]
    rows.append(("E_direct (exact)", float(np.median([x["total_s"] for x in direct])), [1, 1, 1], True))
    sweeps, ms = [], []
    for n in systems:
        h = json.loads((HERE / f"m1hist_{n}.json").read_text())
        d = json.loads((HERE / f"m1_{n}.json").read_text())
        ev = h["events"].get("stop 1e-06 Pa")
        sweeps.append(ev["sweeps"] if ev else h["last"]["sweeps"])
        ms.append(d["jacobi_ms_per_sweep"])
    jac = float(np.median([sw * m / 1e3 for sw, m in zip(sweeps, ms)]))
    cap = [json.loads((HERE / f"m1_{n}.json").read_text()) for n in systems]
    cap200 = float(np.median([[x for x in d["rows"] if x["candidate"] == "A_jacobi_cap200"][0]["total_s"] for d in cap]))
    rows.append(("A_jacobi 1e-6 Pa stop", jac, sweeps, True))
    rows.append(("A_jacobi cap 200", cap200, [200, 200, 200], False))
    print(f"{room}: momentum {1e3 * momentum:.1f} ms per outer, corrector overhead {1e3 * over:.2f} ms, level {level:g}, "
          f"outer iterations {lo}, {mid}, {hi}\n")
    print(f"| Correction | Iterations (outer 1, 100, 1000) | Correction ms | ms per outer | Steady solve at {lo} | at {mid} | at {hi} |")
    print("|---|---|---|---|---|---|---|")
    out = []
    for cand, sec, its, ok in rows:
        per = momentum + over + sec
        note = "" if ok else " (level not reached)"
        cells = []
        for n in (lo, mid, hi):
            t = n * per
            cells.append(f"{t / 60:.1f} min" if t < 7200 else f"{t / 3600:.1f} h")
        print(f"| {cand}{note} | {', '.join(str(i) for i in its)} | {1e3 * sec:.1f} | {1e3 * per:.1f} | " + " | ".join(cells) + " |")
        out.append({"candidate": cand, "correction_s": sec, "per_outer_s": per, "iterations": its, "reached": ok})
    (HERE / f"m3_{room}_{level:g}.json").write_text(json.dumps(
        {"room": room, "level": level, "momentum_s": momentum, "overhead_s": over, "outer": [lo, mid, hi], "rows": out}))


if __name__ == "__main__":
    main()
```

## Appendix P: hist_table36.py

```python
"""Builder probe, prompt 36: today's loop on the relative-residual scale, one row per system.

Usage: python hist_table36.py
Reads m1hist_*.json (the untimed runs) and m1_*.json (seconds per sweep and the
cap-200 row). Prints markdown.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOMS = [("product200", 1000), ("annex180", 1000), ("product40", 1000), ("channel80", 569), ("cavity80", 1000)]


def main() -> None:
    print("| System | ms per sweep | 1e-6 Pa stop: sweeps, s, relative residual, net outflow / supply "
          "| 1e-8 Pa stop: sweeps, s, relative residual | Sweeps to 1e-2 | to 1e-4 | to 1e-6 | to 1e-8 "
          "| Cap 200: relative residual, net outflow / supply |")
    print("|---|---|---|---|---|---|---|---|---|")
    for room, last in ROOMS:
        for k in (1, 100, last):
            s = f"{room}_it{k}"
            hp, mp = HERE / f"m1hist_{s}.json", HERE / f"m1_{s}.json"
            if not hp.exists():
                print(f"| {room} {k} | (running) | | | | | | | |")
                continue
            h, d = json.loads(hp.read_text()), json.loads(mp.read_text())
            ms, ev, last_e = d["jacobi_ms_per_sweep"], h["events"], h["last"]

            def stop(key: str, net: bool = True) -> str:
                e = ev.get(key)
                if not e:
                    return f"not met in {last_e['sweeps']:,} (relative {last_e['rel2']:.1e})"
                out = f"{e['sweeps']:,}, {e['sweeps'] * ms / 1e3:.1f}, {e['rel2']:.1e}"
                return out + (f", {e['signed_over_supply']:.1e}" if net else "")

            def sweeps(key: str) -> str:
                e = ev.get(key)
                return f"{e['sweeps']:,}" if e else f"> {last_e['sweeps']:,}"

            c2 = [x for x in d["rows"] if x["candidate"] == "A_jacobi_cap200"][0]
            print(f"| {room} {k} | {ms:.3f} | {stop('stop 1e-06 Pa')} | {stop('stop 1e-08 Pa', False)} "
                  f"| {sweeps('level 1e-02')} | {sweeps('level 1e-04')} | {sweeps('level 1e-06')} "
                  f"| {sweeps('level 1e-08')} | {c2['rel2']:.2f}, {c2['signed_over_supply']:.1e} |")


if __name__ == "__main__":
    main()
```

## Appendix Q: probe36b.py

```python
"""Builder probe, prompt 36b: the runs behind the claims the fix pass adds.

The premise review and test 36 measured several things the report did not:
CG's growth to 400x150, the Annex 20 room at ADR-012's 216x72, a tight start
before a loose level, a third steady state, the summation order of CG's
reductions, the standing imbalance under the committed outlets (T0), the
committed configuration on 80x30 and 200x75, and CG with a sparse product.
ECR-003 and ADR-013 now cite each, so each is run again here, by this
builder, with the builder-36 probe modules imported read-only. Writes only
to this directory.

Usage (from results/builder36b/, with results/builder36/venv36, BLAS on one
thread):

    python probe36b.py scaling FAMILY NX NY
        FAMILY t3 (the measurement 2 room) or annex (Annex 20 under T1).
        Drives the room with SuperLU, keeps the p' system at outer 100 and
        solves it with Jacobi-PCG to 1e-4, 1e-6 and 1e-8, three timed solves
        each; records the recursive and the true relative residual at exit.
    python probe36b.py retime NAME [NAME ...]
        Jacobi-PCG to 1e-8 on systems saved by scaling, seven timed solves
        each, interleaved so a drift in the machine's speed falls on every
        system alike.
    python probe36b.py csr
        On the three captured product200 systems: Jacobi-PCG with the probe's
        five shifted arrays against the same loop with a SciPy CSR product,
        to 1e-8, five timed solves each.
    python probe36b.py order
        One correction to 1e-8 and 1e-10 on five captured systems with CG's
        three sums taken in three orders: NumPy's vdot, the reversed array
        through np.sum, and blocks of 256 summed in reverse block order.
    python probe36b.py outer NX NY MODE [MU] [ALPHA] [TREATMENT] [N_OUTER]
        The measurement 2 room (capture36.run_room) with MODE one of
        outer36's, or pcgrev:RTOL (CG with reversed sums) or
        sched:N:LOOSE (N corrections at 1e-8, then LOOSE). TREATMENT T3 (the
        report's) or T0 (every outlet extrapolated, as committed). Records the
        shut faces at the end and the field's distance from the report's
        m2_40x15_direct.npz where the grid is 40x15.
    python probe36b.py continue NX NY MODE_A ALPHA_A MODE_B ALPHA_B N_MORE
        Runs MODE_A at ALPHA_A to its error_estimate stop, then continues the
        state N_MORE outer iterations with MODE_B at ALPHA_B (restart36's
        loop), recording the drift, the last step and the distance from the
        report's direct state.
    python probe36b.py committed NX NY N_OUTER
        configs/clean_room_default.yaml with only nx, ny and max_simple_iter
        changed: T0, one momentum sweep, alpha_velocity 0.7, the committed
        Jacobi loop (cap 200, 1e-6 Pa), velocity_step. Records the residual,
        the corrections that reach the cap, and the first non-finite outer.
"""

import json
import math
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent
B36 = OUT.parent / "builder36"
sys.path.insert(0, str(B36))

import numpy as np  # noqa: E402

import capture36  # noqa: E402
import common36 as c36  # noqa: E402
import outer36  # noqa: E402
import solvers36  # noqa: E402

from src.mesh import SOLID  # noqa: E402
from src.staggered import to_cell_centers  # noqa: E402

capture36.HERE = OUT  # run_room writes its records here, not into builder36
LEVELS = (1e-4, 1e-6, 1e-8)


def rhs(b: np.ndarray, active: np.ndarray, needs_pin: bool) -> np.ndarray:
    """f = -b, projected onto the range on a closed domain, zero off the equations."""
    f = -b.copy()
    if needs_pin:
        f[active] -= f[active].mean()
    f[~active] = 0.0
    return f


def write(name: str, out: dict) -> None:
    (OUT / f"{name}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, list) or len(v) < 40}), flush=True)


# ---------------------------------------------------------------------------
# CG with its sums in a chosen order
# ---------------------------------------------------------------------------


def make_dot(order: str):  # type: ignore[no-untyped-def]
    """A dot product over whole arrays with the summation order named."""
    if order == "vdot":
        return lambda a, b: float(np.vdot(a, b))
    if order == "reversed":
        return lambda a, b: float(np.sum(a.ravel()[::-1] * b.ravel()[::-1]))
    if order == "blocks256":
        def dot(a, b):  # type: ignore[no-untyped-def]
            prod = (a.ravel() * b.ravel())
            pad = (-prod.size) % 256
            parts = np.concatenate([prod, np.zeros(pad)]).reshape(-1, 256).sum(axis=1)
            return float(np.sum(parts[::-1]))
        return dot
    raise ValueError(order)


def pcg_ordered(c, f: np.ndarray, rtol: float, atol: float, order: str,  # type: ignore[no-untyped-def]
                maxiter: int = 20000) -> tuple[np.ndarray, int, float]:
    """solvers36.pcg with the diagonal preconditioner and every sum taken in ORDER."""
    dot = make_dot(order)
    inv = np.where(c.a_p > 0.0, 1.0 / np.where(c.a_p > 0.0, c.a_p, 1.0), 0.0)
    x = np.zeros_like(f)
    r = f.copy()
    fn = math.sqrt(dot(f, f))
    stop = max(rtol * fn, atol)
    if fn == 0.0:
        return x, 0, 0.0
    z = inv * r
    p = z.copy()
    rz = dot(r, z)
    k = 0
    rn = fn
    while k < maxiter:
        q = c36.apply_a(c, p)
        alpha = rz / dot(p, q)
        x += alpha * p
        r -= alpha * q
        k += 1
        rn = math.sqrt(dot(r, r))
        if rn <= stop:
            break
        z = inv * r
        rz_new = dot(r, z)
        p *= rz_new / rz
        p += z
        rz = rz_new
    return x, k, rn / fn


# ---------------------------------------------------------------------------
# scaling and csr
# ---------------------------------------------------------------------------


def scaling(family: str, nx: int, ny: int) -> None:
    capture = 100
    if family == "t3":
        stop = capture36.stopping_for(nx, ny, 8.0, 3.0, 1.2, 60.0)
        room = c36.product_t3(f"scaling_{family}", nx, ny, capture + 1, stopping=stop)
    elif family == "annex":
        room = c36.annex20(f"scaling_{family}", nx, ny, capture + 1)
    else:
        raise SystemExit(family)
    kept: dict = {}
    calls = [0]

    def solve(c, b, active, needs_pin):  # type: ignore[no-untyped-def]
        if calls[0] == capture:
            kept.update(c=c, b=b.copy(), active=active.copy(), needs_pin=needs_pin)
        calls[0] += 1
        pin = tuple(int(v) for v in np.argwhere(active)[0])
        return solvers36.direct_solve(c, rhs(b, active, needs_pin), needs_pin, pin).x, 1

    c36.install_corrector(room, solve, measure_residual=False)
    t0 = time.perf_counter()
    room.solver.solve_steady()
    drive_s = time.perf_counter() - t0
    c, active = kept["c"], kept["active"]
    f = rhs(kept["b"], active, kept["needs_pin"])
    np.savez(OUT / f"system_{family}_{nx}x{ny}_it{capture}.npz", a_p=c.a_p, a_e=c.a_e, a_w=c.a_w,
             a_n=c.a_n, a_s=c.a_s, f=f)
    out: dict = {"family": family, "nx": nx, "ny": ny, "capture": capture,
                 "unknowns": int(active.sum()), "drive_s": drive_s,
                 "threads": c36.threads_note(), "levels": {}}
    for rtol in LEVELS:
        times = []
        for _ in range(3):
            res = solvers36.jacobi_pcg(c, f, rtol, maxiter=200000, keep=True)
            times.append(res.total_s)
        r = f - c36.apply_a(c, res.x)
        r[~active] = 0.0
        out["levels"][f"{rtol:.0e}"] = {
            "iterations": res.iterations,
            "recursive_rel": res.history[-1],
            "true_rel": float(np.linalg.norm(r) / np.linalg.norm(f)),
            "ms": [1e3 * t for t in times],
            "ms_median": 1e3 * float(np.median(times)),
        }
    write(f"scaling_{family}_{nx}x{ny}", out)


def retime(names: list[str], repeats: int = 7) -> None:
    """Jacobi-PCG to 1e-8 on saved systems, the timed solves interleaved across them."""
    systems = {}
    for name in names:
        z = np.load(OUT / f"{name}.npz")
        c = c36.PressureCoefficients(a_p=z["a_p"], a_e=z["a_e"], a_w=z["a_w"], a_n=z["a_n"], a_s=z["a_s"])
        systems[name] = (c, z["f"])
    times: dict[str, list[float]] = {name: [] for name in names}
    iterations = {}
    for _ in range(repeats):
        for name, (c, f) in systems.items():
            res = solvers36.jacobi_pcg(c, f, 1e-8, maxiter=200000)
            times[name].append(res.total_s)
            iterations[name] = res.iterations
    out = {"threads": c36.threads_note(), "repeats": repeats,
           "systems": {n: {"iterations": iterations[n], "ms": [1e3 * t for t in times[n]],
                           "ms_median": 1e3 * float(np.median(times[n])),
                           "ms_min": 1e3 * float(np.min(times[n]))} for n in names}}
    write("retime_" + "_".join(n.replace("system_", "") for n in names), out)


def csr() -> None:
    out: dict = {"threads": c36.threads_note(), "systems": {}}
    for k in (1, 100, 1000):
        s = c36.load_system(f"product200_it{k}")
        f = s.rhs()
        active = s.active
        a, _ = solvers36.to_csr(s.c)
        fa = f[active]
        inv = 1.0 / s.c.a_p[active]
        shift, sparse = [], []
        for _ in range(5):
            res = solvers36.jacobi_pcg(s.c, f, 1e-8)
            shift.append(res.total_s)
            t0 = time.perf_counter()
            xa, it, _ = solvers36.pcg(lambda v: a @ v, lambda r: inv * r, fa, 1e-8, 20000)
            sparse.append(time.perf_counter() - t0)
        x = np.zeros_like(f)
        x[active] = xa
        out["systems"][f"it{k}"] = {
            "shift_iterations": res.iterations, "csr_iterations": it,
            "shift_ms_median": 1e3 * float(np.median(shift)),
            "csr_ms_median": 1e3 * float(np.median(sparse)),
            "shift_ms_per_iteration": 1e3 * float(np.median(shift)) / res.iterations,
            "csr_ms_per_iteration": 1e3 * float(np.median(sparse)) / it,
            "p_diff_rel": float(np.abs(x - res.x).max() / np.abs(res.x).max()),
        }
    write("csr_product200", out)


# ---------------------------------------------------------------------------
# order
# ---------------------------------------------------------------------------


def order() -> None:
    out: dict = {"threads": c36.threads_note(), "systems": {}}
    for name in ("product200_it1", "product200_it100", "product200_it1000",
                 "product40_it100", "cavity80_it100"):
        s = c36.load_system(name)
        f = s.rhs()
        atol = 1e-13 * s.supply
        row: dict = {}
        for rtol in (1e-8, 1e-10):
            base = None
            entry: dict = {}
            for kind in ("vdot", "reversed", "blocks256"):
                x, k, rel = pcg_ordered(s.c, f, rtol, atol, kind)
                if s.needs_pin:
                    x = x.copy()
                    x[s.active] -= x[s.pin_cell]
                u, v = c36.corrected_faces(s, x)
                entry[kind] = {"iterations": k, "recursive_rel": rel}
                if base is None:
                    base = (x, u, v)
                    entry["max_abs_p"] = float(np.abs(x).max())
                    # The same loop run one iteration past its stop: what a
                    # second implementation stopping one iteration later
                    # would return.
                    x1, k1, _ = pcg_ordered(s.c, f, 0.0, 0.0, kind, maxiter=k + 1)
                    if s.needs_pin:
                        x1 = x1.copy()
                        x1[s.active] -= x1[s.pin_cell]
                    u1, v1 = c36.corrected_faces(s, x1)
                    entry["one_more"] = {
                        "iterations": k1,
                        "p_diff_pa": float(np.abs(x1 - x).max()),
                        "face_diff_m_s": float(max(np.abs(u1 - u).max(), np.abs(v1 - v).max())),
                    }
                else:
                    entry[kind]["p_diff_pa"] = float(np.abs(x - base[0]).max())
                    entry[kind]["face_diff_m_s"] = float(max(np.abs(u - base[1]).max(),
                                                             np.abs(v - base[2]).max()))
            row[f"{rtol:.0e}"] = entry
        out["systems"][name] = row
    write("order_systems", out)


# ---------------------------------------------------------------------------
# outer and continue
# ---------------------------------------------------------------------------


def make_solve(mode: str, supply: float):  # type: ignore[no-untyped-def]
    """outer36's modes, plus pcgrev:RTOL and sched:N:LOOSE."""
    atol = 1e-13 * supply
    if mode.startswith("pcgrev:"):
        rtol = float(mode.split(":")[1])

        def solve(c, b, active, needs_pin):  # type: ignore[no-untyped-def]
            x, k, _ = pcg_ordered(c, rhs(b, active, needs_pin), rtol, atol, "reversed")
            return x, k
        return solve
    if mode.startswith("sched:"):
        _, n_tight, loose = mode.split(":")
        n_tight, loose_rtol = int(n_tight), float(loose)
        calls = [0]

        def solve(c, b, active, needs_pin):  # type: ignore[no-untyped-def]
            rtol = 1e-8 if calls[0] < n_tight else loose_rtol
            calls[0] += 1
            res = solvers36.jacobi_pcg(c, rhs(b, active, needs_pin), rtol, atol=atol)
            return res.x, res.iterations
        return solve
    return outer36.make_solve(mode, supply)


def field_distance(nx: int, ny: int, u: np.ndarray, v: np.ndarray, mesh) -> dict:  # type: ignore[no-untyped-def]
    """Largest and median distance from the report's direct state on 40x15, and where."""
    ref_path = B36 / "m2_40x15_direct.npz"
    if (nx, ny) != (40, 15) or not ref_path.exists():
        return {}
    ref = np.load(ref_path)
    fluid = mesh.cell_type != SOLID
    d = np.maximum(np.abs(u - ref["u"]), np.abs(v - ref["v"]))
    d[~fluid] = 0.0
    j, i = np.unravel_index(int(np.argmax(d)), d.shape)
    return {"vs_direct_max": float(d.max()), "at": [float(mesh.xc[i]), float(mesh.yc[j])],
            "vs_direct_median": float(np.median(d[fluid]))}


def shut_faces(solver) -> dict:  # type: ignore[no-untyped-def]
    return {
        "closed_bottom": np.flatnonzero(getattr(solver, "closed_bottom", np.zeros(1, bool))).tolist(),
        "closed_right": np.flatnonzero(getattr(solver, "closed_right", np.zeros(1, bool))).tolist(),
        "open_bottom": np.flatnonzero(solver._corrector._out_bottom).tolist(),
        "open_right": np.flatnonzero(solver._corrector._out_right).tolist(),
    }


def outer(nx: int, ny: int, mode: str, mu: float, alpha: float, treatment: str, n_outer: int) -> None:
    tag = f"outer_{nx}x{ny}_{mode.replace(':', '_')}_mu{mu:g}_a{alpha:g}_{treatment}"
    stop = capture36.stopping_for(nx, ny, 8.0, 3.0, 1.2, 60.0)
    room = c36.product_t3(tag, nx, ny, n_outer, mu_factor=mu, alpha_u=alpha, stopping=stop)
    room.solver.treatment = treatment
    fluid = room.mesh.cell_type != SOLID
    p_means: list[float] = []
    solve_steady = room.solver.solve_steady

    def with_pressure(on_iteration=None):  # type: ignore[no-untyped-def]
        def cb(state) -> None:  # type: ignore[no-untyped-def]
            p_means.append(float(state.p[fluid].mean()))
            if on_iteration is not None:
                on_iteration(state)
        return solve_steady(on_iteration=cb)

    room.solver.solve_steady = with_pressure
    rec = capture36.run_room(room, make_solve(mode, room.supply), tag, print_every=500)
    extra: dict = {"mode": mode, "mu_factor": mu, "alpha": alpha, "treatment": treatment,
                   "mass_imbalance_tol": stop["mass_imbalance_tol"], **shut_faces(room.solver)}
    if rec["stop"] != "diverged":
        z = np.load(OUT / f"{tag}.npz")
        extra.update(field_distance(nx, ny, z["u"], z["v"], room.mesh))
        extra["worst_end"] = rec["worst"][-1]
        extra["b_norm_end"] = rec["b_norm"][-1]
        if len(p_means) > 101:
            extra["p_mean_end"] = p_means[-1]
            extra["p_mean_change_per_outer_last100"] = (p_means[-1] - p_means[-101]) / 100.0
    rec.update(extra)
    (OUT / f"{tag}.json").write_text(json.dumps(rec))
    print(json.dumps({k: v for k, v in rec.items() if not isinstance(v, list) or len(v) < 40}), flush=True)


def cont(nx: int, ny: int, mode_a: str, alpha_a: float, mode_b: str, alpha_b: float, n_more: int) -> None:
    tag = f"continue_{nx}x{ny}_{mode_a.replace(':', '_')}_a{alpha_a:g}_{mode_b.replace(':', '_')}_a{alpha_b:g}"
    stop = capture36.stopping_for(nx, ny, 8.0, 3.0, 1.2, 60.0)
    room = c36.product_t3(tag, nx, ny, 20000, alpha_u=alpha_a, stopping=stop)
    corr = c36.install_corrector(room, make_solve(mode_a, room.supply), measure_residual=False)
    solver = room.solver
    u_c, v_c, p = solver.solve_steady()
    out: dict = {"mode_a": mode_a, "alpha_a": alpha_a, "mode_b": mode_b, "alpha_b": alpha_b,
                 "stop_a": solver.stop_reason, "outer_a": len(solver.residual_history),
                 "faces_a": shut_faces(solver), "a": field_distance(nx, ny, u_c, v_c, room.mesh)}
    fv = solver.face_velocities
    u, v, p = fv.u.copy(), fv.v.copy(), p.copy()
    u0, v0 = u_c.copy(), v_c.copy()
    fluid = room.mesh.cell_type != SOLID
    corr.solve = make_solve(mode_b, room.supply)
    solver._predictor._alpha = alpha_b
    ref = solver.reference_velocity
    drift, step = [], []
    prev_u, prev_v = u0, v0
    for _ in range(n_more):
        solver._extrapolate_outlets(u, v)
        pred = solver._predictor.predict(u, v, p)
        res = corr.correct(pred, p)
        u, v, p = res.u, res.v, res.p
        uc, vc = to_cell_centers(u, v)
        step.append(float(max(np.max(np.abs(uc - prev_u)[fluid]), np.max(np.abs(vc - prev_v)[fluid]))))
        drift.append(float(max(np.max(np.abs(uc - u0)[fluid]), np.max(np.abs(vc - v0)[fluid]))))
        prev_u, prev_v = uc, vc
    out.update(drift_end=drift[-1], drift_max=max(drift), step_end_m_s=step[-1],
               step_end_over_ref=step[-1] / ref, step_at_1000=step[min(999, len(step) - 1)],
               faces_b=shut_faces(solver), b=field_distance(nx, ny, prev_u, prev_v, room.mesh))
    write(tag, out)


# ---------------------------------------------------------------------------
# committed
# ---------------------------------------------------------------------------


class Stop(Exception):
    """Raised from the callback at a non-finite residual."""


def committed(nx: int, ny: int, n_outer: int) -> None:
    room = c36.product_committed(f"committed_{nx}x{ny}", nx, ny, n_outer)
    solver = room.solver
    cap, tol = room.cfg.max_pressure_iter, room.cfg.pressure_tol
    corrector, correct = solver._corrector, solver._corrector.correct
    sweeps: list[int] = []

    def recorded(prediction, p):  # type: ignore[no-untyped-def]
        result = correct(prediction, p)
        sweeps.append(result.sweeps)
        return result

    corrector.correct = recorded
    residual: list[float] = []
    first_bad = [None]

    def cb(state) -> None:  # type: ignore[no-untyped-def]
        residual.append(float(state.residual))
        if not math.isfinite(state.residual):
            first_bad[0] = state.iteration
            raise Stop

    t0 = time.perf_counter()
    try:
        solver.solve_steady(on_iteration=cb)
        stop = solver.stop_reason or "max_simple_iter"
    except Stop:
        stop = "non-finite"
    r = np.array(residual)
    fin = r[np.isfinite(r)]
    least = int(np.argmin(fin)) if fin.size else None
    tail = fin[-1000:] if fin.size >= 1000 else fin
    out = {
        "nx": nx, "ny": ny, "n_outer": n_outer, "treatment": "committed (T0)",
        "alpha_velocity": room.cfg.alpha_velocity, "max_pressure_iter": cap, "pressure_tol": tol,
        "stop": stop, "outer": len(residual), "first_non_finite": first_bad[0],
        "least_residual": float(fin[least]) if least is not None else None, "least_at": least,
        "tail_p5": float(np.percentile(tail, 5)) if tail.size else None,
        "tail_p95": float(np.percentile(tail, 95)) if tail.size else None,
        "velocity_step_met": bool(np.any(fin < room.cfg.convergence_tol)),
        "corrections_at_cap": int(np.sum(np.array(sweeps) >= cap)), "corrections": len(sweeps),
        "seconds": time.perf_counter() - t0, "threads": c36.threads_note(),
    }
    write(f"committed_{nx}x{ny}", out)


def main() -> None:
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "scaling":
        scaling(args[0], int(args[1]), int(args[2]))
    elif cmd == "retime":
        retime(args)
    elif cmd == "csr":
        csr()
    elif cmd == "order":
        order()
    elif cmd == "outer":
        nx, ny, mode = int(args[0]), int(args[1]), args[2]
        mu = float(args[3]) if len(args) > 3 else 1.0
        alpha = float(args[4]) if len(args) > 4 else 0.5
        treatment = args[5] if len(args) > 5 else "T3"
        n_outer = int(args[6]) if len(args) > 6 else 20000
        outer(nx, ny, mode, mu, alpha, treatment, n_outer)
    elif cmd == "continue":
        cont(int(args[0]), int(args[1]), args[2], float(args[3]), args[4], float(args[5]), int(args[6]))
    elif cmd == "committed":
        committed(int(args[0]), int(args[1]), int(args[2]))
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
```

## Appendix R: run36b.sh

```sh
#!/bin/sh
# Prompt 36b: the launchers, from results/builder36b/. BLAS on one thread in
# every run. The timed set runs one at a time with nothing beside it; the
# outer-loop set runs in parallel afterwards and is untimed.
#   sh run36b.sh timed     scaling on four grids, then the CSR comparison
#   sh run36b.sh retime    the four systems scaling saved, timed again
#                          interleaved (added after the first timed set)
#   sh run36b.sh order     the summation-order probe on five captured systems
#   sh run36b.sh outer     the outer-loop runs, in parallel
#   sh run36b.sh committed the committed configuration on 80x30 and 200x75
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=../builder36/venv36/Scripts/python.exe
mkdir -p logs
case "$1" in
timed)
    for g in "t3 200 75" "t3 400 150" "annex 180 60" "annex 216 72"; do
        set -- $g
        $PY probe36b.py scaling $1 $2 $3 > logs/scaling_$1_$2x$3.log 2>&1
    done
    $PY probe36b.py csr > logs/csr.log 2>&1
    ;;
retime)
    $PY probe36b.py retime system_t3_200x75_it100 system_t3_400x150_it100 \
        system_annex_180x60_it100 system_annex_216x72_it100 > logs/retime.log 2>&1
    ;;
order)
    $PY probe36b.py order > logs/order.log 2>&1
    ;;
outer)
    # 40x15 at real air under T3: the reproduction controls with the shut
    # faces recorded, the tight starts, the reversed sums, the third state.
    $PY probe36b.py outer 40 15 direct > logs/o_direct.log 2>&1 &
    $PY probe36b.py outer 40 15 pcg:1e-2 > logs/o_pcg1e-2.log 2>&1 &
    $PY probe36b.py outer 40 15 pcg:1e-8 > logs/o_pcg1e-8.log 2>&1 &
    $PY probe36b.py outer 40 15 pcgrev:1e-8 > logs/o_pcgrev1e-8.log 2>&1 &
    $PY probe36b.py outer 40 15 sched:100:1e-1 > logs/o_sched100_1e-1.log 2>&1 &
    $PY probe36b.py outer 40 15 sched:1300:1e-1 > logs/o_sched1300_1e-1.log 2>&1 &
    $PY probe36b.py outer 40 15 sched:100:3e-1 > logs/o_sched100_3e-1.log 2>&1 &
    $PY probe36b.py outer 40 15 sched:100:1e-2 > logs/o_sched100_1e-2.log 2>&1 &
    $PY probe36b.py continue 40 15 direct 0.45 direct 0.5 4000 > logs/c_a045_a05.log 2>&1 &
    $PY probe36b.py continue 40 15 direct 0.5 direct 0.45 4000 > logs/c_a05_a045.log 2>&1 &
    $PY probe36b.py continue 40 15 pcg:1e-2 0.5 direct 0.5 4000 > logs/c_pcg1e-2_direct.log 2>&1 &
    # 80x30 at Re 90: T3 as the report ran it, and T0, the committed outlets.
    $PY probe36b.py outer 80 30 direct 1000 0.5 T3 > logs/o_re90_direct_T3.log 2>&1 &
    $PY probe36b.py outer 80 30 direct 1000 0.5 T0 > logs/o_re90_direct_T0.log 2>&1 &
    $PY probe36b.py outer 80 30 pcg:1e-2 1000 0.5 T0 > logs/o_re90_pcg1e-2_T0.log 2>&1 &
    $PY probe36b.py outer 80 30 pcg:1e-4 1000 0.5 T0 > logs/o_re90_pcg1e-4_T0.log 2>&1 &
    $PY probe36b.py outer 80 30 pcg:1e-8 1000 0.5 T0 > logs/o_re90_pcg1e-8_T0.log 2>&1 &
    wait
    ;;
committed)
    $PY probe36b.py committed 80 30 8000 > logs/committed_80x30.log 2>&1 &
    $PY probe36b.py committed 200 75 3000 > logs/committed_200x75.log 2>&1 &
    wait
    ;;
esac
```

## Appendix S: summary36b.py

```python
"""Builder probe, prompt 36b: the run records summarised as the report's section 12 tables give them."""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
B36 = HERE.parent / "builder36"


def g(x: float | None, n: int = 2) -> str:
    if x is None:
        return "-"
    return f"{x:.{n - 1}e}"


def outer_rows() -> None:
    ref = None
    names = sorted(p.stem for p in HERE.glob("outer_*.json"))
    print("| Run | Stop | velocity_step at | Outer | Inner, median [max] | Rel. residual, median | Field vs report's direct, max (at) | median | Worst cell at end / tol | Shut faces (bottom; right) | dp/outer, last 100 | b norm at end |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in names:
        d = json.loads((HERE / f"{name}.json").read_text())
        inner = np.array(d["inner"])
        rel = np.array(d["rel"])
        worst = d.get("worst_end")
        tol = d["mass_imbalance_tol"]
        at = d.get("at")
        print(f"| {name.replace('outer_', '')} | {d['stop']} | {d['velocity_step_outer'] or '-'} | {d['outer']} "
              f"| {int(np.median(inner))} [{int(inner.max())}] | {g(float(np.median(rel)))} "
              f"| {g(d.get('vs_direct_max'))} ({'-' if at is None else f'{at[0]:.2f}, {at[1]:.2f}'}) | {g(d.get('vs_direct_median'))} "
              f"| {g(None if worst is None else worst / tol)} | {d['closed_bottom']}; {d['closed_right']} "
              f"| {g(d.get('p_mean_change_per_outer_last100'), 3)} | {g(d.get('b_norm_end'), 3)} |")


def field_pair(a: str, b: str) -> None:
    za, zb = np.load(HERE / f"{a}.npz"), np.load(HERE / f"{b}.npz")
    d = max(np.abs(za["u"] - zb["u"]).max(), np.abs(za["v"] - zb["v"]).max())
    print(f"{a} vs {b}: max field difference {d:.3e} m/s")


def bitwise(mine: str, theirs: str) -> None:
    za, zb = np.load(HERE / f"{mine}.npz"), np.load(B36 / f"{theirs}.npz")
    same = all(np.array_equal(za[k], zb[k]) for k in ("u", "v", "p"))
    print(f"{mine} against builder36 {theirs}: bitwise {'equal' if same else 'DIFFERENT'}")


def jacobi40k() -> None:
    """Today's loop at step 0's settings on 40x15 (the report's run), against direct, from its records."""
    ref, run = np.load(B36 / "m2_40x15_direct.npz"), np.load(B36 / "m2_40x15_jacobi40k.npz")
    rec = json.loads((B36 / "m2_40x15_jacobi40k.json").read_text())
    import yaml  # noqa: PLC0415

    from src.config import SimConfig  # noqa: PLC0415
    from src.mesh import SOLID, Mesh  # noqa: PLC0415

    raw = yaml.safe_load((HERE.parents[1] / "configs/clean_room_default.yaml").read_text())
    raw["domain"]["nx"], raw["domain"]["ny"] = 40, 15
    mesh = Mesh(SimConfig.from_dict(raw))
    fluid = mesh.cell_type != SOLID
    # The end of run against direct's error_estimate stop; each velocity_step stop against the other.
    for key_u, key_v, label in (("u", "v", "end of run"), ("u_vs", "v_vs", "velocity_step stop")):
        d = np.maximum(np.abs(run[key_u] - ref[key_u]), np.abs(run[key_v] - ref[key_v]))
        d[~fluid] = 0.0
        j, i = np.unravel_index(int(np.argmax(d)), d.shape)
        print(f"jacobi40k vs direct at the {label}: {d.max():.3e} m/s at ({mesh.xc[i]:.2f}, {mesh.yc[j]:.2f})")
    print(f"jacobi40k: stop {rec['stop']} after {rec['outer']}, velocity_step at {rec['velocity_step_outer']}, "
          f"median relative residual {np.median(rec['rel']):.3f}, worst cell at the end {rec['worst'][-1]:.3e}")


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "outer"):
        outer_rows()
    if what in ("all", "pairs"):
        for mine, theirs in (("outer_40x15_direct_mu1_a0.5_T3", "m2_40x15_direct"),
                             ("outer_40x15_pcg_1e-2_mu1_a0.5_T3", "m2_40x15_pcg_1e-2"),
                             ("outer_40x15_pcg_1e-8_mu1_a0.5_T3", "m2_40x15_pcg_1e-8"),
                             ("outer_80x30_direct_mu1000_a0.5_T3", "m2_80x30_direct_mu1000")):
            if (HERE / f"{mine}.npz").exists():
                bitwise(mine, theirs)
        if (HERE / "outer_40x15_pcgrev_1e-8_mu1_a0.5_T3.npz").exists():
            field_pair("outer_40x15_pcgrev_1e-8_mu1_a0.5_T3", "outer_40x15_pcg_1e-8_mu1_a0.5_T3")
    if what in ("all", "continue"):
        for p in sorted(HERE.glob("continue_*.json")):
            d = json.loads(p.read_text())
            print(f"{p.stem}: A {d['stop_a']} at {d['outer_a']}, A vs direct {g(d['a'].get('vs_direct_max'))}; "
                  f"after B: drift {g(d['drift_end'])} (max {g(d['drift_max'])}), step at 1000 {g(d['step_at_1000'])}, "
                  f"last step {g(d['step_end_m_s'])} m/s; B vs direct {g(d['b'].get('vs_direct_max'))} at {d['b'].get('at')}; "
                  f"faces A {d['faces_a']['closed_bottom']}/{d['faces_a']['closed_right']} "
                  f"B {d['faces_b']['closed_bottom']}/{d['faces_b']['closed_right']}")
    if what in ("all", "jacobi40k"):
        jacobi40k()
    if what in ("all", "committed"):
        for p in sorted(HERE.glob("committed_*.json")):
            print(p.stem, json.loads(p.read_text()))


if __name__ == "__main__":
    main()
```
