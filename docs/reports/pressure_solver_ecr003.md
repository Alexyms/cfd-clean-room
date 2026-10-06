# ECR-003: The Pressure Solver, Measured

**Date:** 2026-10-06.
**Branch:** `docs/ecr-003-pressure-solver`, on main at ed83807, in the worktree
`cfd_clean_room_ecr003`. The momentum, pressure, solver and boundary modules are the same at
ed83807 and at main's 023b8f6, so what is measured here holds on both.
**Probes:** `results/builder36/` in the worktree (untracked), reproduced in the appendices.
**Order:** sections 1 to 5 (the question, the method, the controls on the harness, what each
outcome means, and both sets of predictions) are the first commit, made before item 0 and before
measurements 1 to 3 ran. The sections after them were written after the runs.

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

Usage: python capture36.py ROOM [N_OUTER]

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
    room, solve = build(room_name, n_outer)
    run_room(room, solve, room_name, CAPTURE)


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

Today's loop: to 1e-6 Pa (the committed tolerance) and 1e-8 Pa (the
validation cases' and step 0's), uncapped up to 5,000,000 sweeps; at the
committed cap of 200 sweeps; and the sweep at which each relative level is
first seen (checked every 10 sweeps, untimed, up to 2,000,000 sweeps; its
seconds are the sweeps times the timed loop's seconds per sweep). The history
is a run of its own (--jacobi-history, m1hist_SYSTEM.json), so the untimed
runs can go in parallel while the timed ones run one at a time.

Timings: the median of three runs for every solve under two seconds, one run
otherwise; setup (the multigrid hierarchy, pyamg's setup with the CSR
assembly, SuperLU's factorization) apart from the solve. With
PREVIOUS_SYSTEM, MG-PCG and SA-CG are also run with the hierarchy built on
that system (a stale preconditioner; CG still uses the current matrix).
Writes m1_SYSTEM.json.
"""

import json
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
    jacobi_history,
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
        shim = _Shim(s.c.a_p.shape)
        t0 = time.perf_counter()
        hist = jacobi_history(shim, s.c, s.b, f, LEVELS, 2_000_000)
        hist["seconds_untimed"] = time.perf_counter() - t0
        (HERE / f"m1hist_{name}.json").write_text(json.dumps(hist))
        print(f"{name} jacobi history {hist['found']} last {hist['last']}", flush=True)
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
    for tol in (1e-6, 1e-8):
        res = jacobi_committed(shim, s.c, s.b, tol, 5_000_000)
        row("A_jacobi_stop", f"{tol:.0e} Pa", res, {"ms_per_sweep": 1e3 * res.solve_s / res.iterations})
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
    elif kind == "sacg":
        def solve(c, b, active, needs_pin):  # type: ignore[no-untyped-def]
            res, _ = pyamg_solve(c, rhs(b, active, needs_pin), rtol, "sa", "cg")
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
PY=venv36/Scripts/python.exe
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs
case "$1" in
selftest)
    $PY selftest36.py all > logs/selftest36.log 2>&1 ;;
capture)
    nohup $PY capture36.py product200 > logs/product200.log 2>&1 &
    for r in product40 cavity80 channel80 annex180; do
        $PY capture36.py $r > logs/$r.log 2>&1 &
    done
    wait ;;
item0)
    unset OPENBLAS_NUM_THREADS OMP_NUM_THREADS MKL_NUM_THREADS
    for r in product200 product40 cavity80 channel80 annex180; do
        $PY item0_36.py ${r}_it1 ${r}_it100 ${r}_it1000 > logs/item0_$r.log 2>&1
    done ;;
m1)
    for r in product200 cavity80 channel80 annex180 product40; do
        $PY m1_36.py ${r}_it1 > logs/m1_${r}_it1.log 2>&1
        $PY m1_36.py ${r}_it100 ${r}_it1 > logs/m1_${r}_it100.log 2>&1
        $PY m1_36.py ${r}_it1000 ${r}_it100 > logs/m1_${r}_it1000.log 2>&1
    done ;;
m1hist)
    for r in product200 cavity80 channel80 annex180 product40; do
        for k in 1 100 1000; do
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
