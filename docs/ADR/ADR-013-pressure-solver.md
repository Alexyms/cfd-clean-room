# ADR-013: Pressure Correction Solver: Jacobi-Preconditioned Conjugate Gradients

## Status
Proposed 2026-10-06, with ECR-003 (`docs/ECR/ECR-003-pressure-solver.md`), from the measurements in
`docs/reports/pressure_solver_ecr003.md` [1]. A premise review and `/cfd-test 36` follow, then Alex
takes the decisions below. As accepted it amends REQ-S08 and supersedes ADR-010's weighted Jacobi
sweep as the pressure solve. Bracketed numbers point at the sources at the end; section numbers in
the decisions point at this design's sections, "the report" at [1].

## Decisions for Alex
Each decision is a picture first, then the options ranked with their consequence, then the section
that carries the detail. The ranking is mine, from the measurements; none is taken.

**1. What solves the pressure correction (section A).**
*Picture.* Every outer iteration the solver works out a pressure correction that makes the air's
flow add up in every cell. Today it does so by a sweep that passes information one cell at a time.
On the product mesh that takes tens to hundreds of thousands of sweeps per correction, 6 to 79
seconds, and still leaves about a thousandth of the imbalance; a steady solve would take 10 to 42
hours. Four methods were measured that do the same work in a few hundredths to a few tenths of a
second.
*Options, ranked.*
(1) Conjugate gradients with the diagonal as preconditioner, in NumPy. 0.16 s per correction on
200x75 to a relative residual of 1e-8; 9 to 39 minutes per steady product solve over the assumed
outer range. Each iteration is a five-point product and a division per cell, both one thread per
cell on a GPU, plus three sums over the whole grid. No setup, so a matrix that changes every outer
iteration costs nothing extra. Converged on every system measured, the closed cavity included.
Its iterations grow with the cells per side, so on a mesh twice as fine each way a correction would
cost about eight times as much (twice the iterations on four times the cells; extrapolated).
(2) A sparse direct solve (SciPy's SuperLU). 0.034 s per correction on 200x75, exact, 2.5 to 11
minutes per steady solve, 1.8 times faster than the fastest iterative method; no tolerance to
choose, and no inner tolerance can change which steady state a run settles on. SciPy becomes a
runtime dependency, and a factorization is not data-parallel: REQ-S08's GPU rationale is given up
and Phase 6's CUDA path would be a different library, not a port.
(3) Algebraic multigrid from pyamg, Ruge-Stuben preconditioning CG. 0.07 s per correction, 4.5 to 20
minutes per steady solve. SciPy and pyamg become runtime dependencies; the setup runs every
correction and is sequential; under its defaults its CG fails on the closed cavity at tight
tolerances.
(4) Geometric multigrid preconditioning CG, in NumPy. 0.23 s per correction as built for the
probe, 12 to 53 minutes, because its setup in Python takes 120 to 175 ms; its solve takes 40 to 60 ms
and needs 10 to 11 iterations, and it is the method whose cost grows least as the mesh is refined.
Data-parallel and NumPy only; its setup would have to be rewritten.
*Recommendation.* (1) now, with (4) named as the successor if a finer mesh makes (1) too slow: the
same CG loop with a multigrid preconditioner. (1) keeps the language policy and the GPU path, and
it is enough: the correction then costs 0.16 s of a 0.18 s outer iteration and a steady product solve
takes minutes. If speed on the 200x75 mesh matters more to you than the GPU path and the dependency,
(2) is 3.6 times faster per outer iteration and exact.

**2. Whether to take SciPy and pyamg as runtime dependencies (section C).**
*Picture.* The project's code needs only NumPy, PyYAML, matplotlib and h5py to run, and its
compiled part is planned in C and CUDA. SciPy is a large, standard numerical library; pyamg is a
smaller one built on it. The probes used both, in a separate environment.
*Options, ranked.*
(1) NumPy only. Goes with decision 1's option (1) or (4). The CG loop maps one to one onto a CUDA
kernel set (the five-point product, vector updates, sums), as REQ-S06 and REQ-N03 plan for the
inner loop; CLAUDE.md's C policy covers it if a CPU kernel is ever needed (section C).
(2) SciPy at runtime. Needed for decision 1's option (2); also gives a library CG to test against.
Phase 6 would need a GPU sparse direct solver of its own (cuDSS), not a port, and REQ-N03's
equivalence would compare two libraries.
(3) SciPy and pyamg at runtime. Needed for decision 1's option (3). The GPU counterpart is a
separate algebraic multigrid library (AmgX) with its own build; pyamg's handling of the closed
cavity needs a pin in place of the projection (not measured).
*Recommendation.* (1).

**3. The stop and its default (section B).**
*Picture.* A correction is never exact; the question is when to stop improving it. What is left over
is mass that the corrected faces still do not account for, cell by cell. Today the solver stops when
one sweep changes the correction by less than a millionth of a pascal, which says nothing fixed
about what is left: measured, it left anything from 0.003% to 95% of the imbalance. Looser is
cheaper, but the measurements found three ways it goes wrong. With this solver, at one part in ten
the solve blows up within three outer iterations, and at one part in a hundred the room settles
0.041 m/s away from where an exact correction puts it, at one cell. And in a room where the outlets
keep returning the same imbalance (found at Re 90 on 80x30), the stopping rule's per-cell continuity
can never be met unless every correction leaves less than about 2.5e-7 of it.
*Options, ranked.*
(1) The relative residual `||b + A p'||_2 <= pressure_rtol ||b||_2`, default 1e-8, with a floor at
rounding (stop also when the residual's 2-norm is below 1e-13 of rho times the inflow) and
`max_pressure_iter` as the iteration cap. Met every condition measured: the outer counts, the
converged field to 2.4e-8 m/s of the exact correction's on 40x15, the stopping rule on 80x30 at Re
90. Per correction it costs 1.2 times 1e-4 and 3.2 times 1e-2 with option (1) of decision 1 (2.7
times 1e-2 per outer iteration).
(2) The same at 1e-4. Keeps the counts and the converged field to the floor the path itself sets on
40x15; never meets the stopping rule's continuity conditions on 80x30 at Re 90. 1.2 times cheaper
than (1).
(3) 1e-2 with an absolute floor tied to the stopping rule: each correction also brings its worst
cell below a fraction of `mass_imbalance_tol`. Cheapest early in a solve; the combination was not
run, and 1e-2 alone moved the 40x15 field 0.041 m/s with this solver.
(4) Today's per-sweep change in pascals. Not a measure of the error; with a cap it froze the 80x30
room with 47% of the supply unaccounted.
*Recommendation.* (1).

**4. The configuration keys and what happens to the weighted sweep (section D).**
*Picture.* Every configuration file today says `pressure_tol: 1.0e-6` (or 1e-8) meaning pascals per
sweep, and `max_pressure_iter` meaning sweeps. Under the new stop both numbers mean something else.
*Options, ranked.*
(1) A new key, `pressure_rtol`, dimensionless in (0, 1); `pressure_tol` refused at load with a
message that names the change; `max_pressure_iter` kept as the CG iteration cap with a new default
from the measurements (about 860 iterations to 1e-8 on 200x75; 5,000 leaves margin). The weighted
sweep and `JACOBI_WEIGHT` removed with the tests that pin them; the step 5 report keeps their
evidence. Every committed file changes in the same pull request.
(2) The same keys with the new meaning. No file changes, and every saved configuration silently
means something else: a `pressure_tol` of 1e-6 written as pascals is read as a relative residual.
(3) A `pressure_solver` switch keeping weighted Jacobi selectable beside CG. The laminar results
could be reproduced to the bit on request; two solvers to keep, test and document, and REQ-S08
names one.
*Recommendation.* (1).

**5. Whether the C path is needed now (section C).**
*Picture.* The project's rule is NumPy first, with performance-critical inner loops in compiled
code, and REQ-S06 plans a CUDA version of the pressure loop for Phase 6.
*Options, ranked.*
(1) Defer to Phase 6. With option (1) of decision 1 a steady product solve takes minutes in NumPy;
the compiled loop is Phase 6's deliverable under REQ-S06 and REQ-N03, now a CG loop.
(2) A C kernel for the CG iteration now, through ctypes. Saves an unmeasured factor on the product
mesh (NumPy's per-iteration overhead on 11,000 cells is a few array passes) and adds a second
implementation to keep equivalent before the solver's physics has settled.
(3) A C setup for geometric multigrid now. Only worth it if decision 1 takes option (4).
*Recommendation.* (1).

**6. Two findings outside this change (section E).**
*Picture.* The measurement found two things about the flow solver that the pressure solver does not
cause and cannot fix. In the 80x30 room at Re 90 the pressure rises by the same amount every outer
iteration, forever, because the outlets' rule for the air leaving the room and the correction undo
each other. And the laminar room that converges on 40x15 converges on neither 80x30 nor 200x75,
with ten momentum sweeps and exact corrections.
*Options, ranked.*
(1) Record both where the work that owns them will read them: the outlet finding as an issue for
ECR-002 step 3, which rebuilds the outlets, with the report's section 8.3 as its evidence; the
finer-grid finding as an input to ECR-002 step 5, whose question it is.
(2) Widen ECR-003 to change the outlet treatment. A change to the flow solver's boundary conditions
inside a change to its linear solver, ahead of the step designed for it.
*Recommendation.* (1).

## Context
REQ-S08 names Jacobi iteration for its per-cell data parallelism, with the GPU port in mind; it was
clarified on 2026-09-22 to weighted Jacobi with w = 2/3, because the plain sweep has an exact -1
eigenvalue on a closed domain [2]. ADR-012 H measured one correction on the product mesh at 27,408
sweeps to the committed 1e-6 Pa and 112,519 to 1e-8 Pa, and decision 5 of ADR-012 (Alex,
2026-10-04) brought the solver forward as this change, to land before ECR-002 step 5 [3]. The
report measured the systems the candidates solve (item 0), one correction with each (measurement
1), how tightly the outer loop needs it (measurement 2) and a steady solve (measurement 3) [1].

## A. The solver (REQ-S08; decision 1)

**The systems** (the report, section 6). On all fifteen captured systems (the product on 200x75 and
40x15 under T3 with ten momentum sweeps, the Annex 20 room on 180x60, VAL-001 80x40 and VAL-002
80x80) the matrix is symmetric to the bit, with positive diagonal and non-positive off-diagonals.
Every open system is positive definite (a dense Cholesky factorization succeeds); the cavity is
singular with the constants as null space and a compatible right-hand side. The smallest eigenvalue
of `D^-1 A` is 3.7e-6 to 1.5e-5 on the product mesh and 4.1e-6 to 1.0e-5 on the Annex 20 room; the
largest is 2.0. So CG faces a condition number of about 1.4e5 to 5.4e5, and Jacobi a slowest mode
that shrinks by 1 - 2.5e-6 to 1 - 1e-5 per sweep.

**One correction on 200x75** (the report, section 7; the median over three captured systems,
setup included, one core):

| Method | 1e-2 | 1e-4 | 1e-8 | Iterations at 1e-8 |
|---|---|---|---|---|
| Weighted Jacobi, today's sweep | 0.23 s | 77 s | 276 s | 1.3 million to over 5 million |
| Jacobi-preconditioned CG | 51 ms | 132 ms | 163 ms | 852 to 861 |
| Geometric multigrid, standalone V(2,2) | 183 ms | 211 ms | 277 ms | 22 to 26 |
| Geometric multigrid preconditioning CG | 176 ms | 193 ms | 227 ms | 10 to 11 |
| pyamg, smoothed aggregation with CG | 51 ms | 63 ms | 92 ms | 11 to 13 |
| pyamg, Ruge-Stuben with CG | 45 ms | 59 ms | 74 ms | 9 to 11 |
| SuperLU (exact) | 34 ms | 34 ms | 34 ms | one factorization |

Jacobi's row is the median over the three systems of the sweeps to each level times each system's
timed seconds per sweep, 0.20 to 0.22 ms (the report, section 7.1). Today's 1e-6 Pa stop takes 28,000 to 378,000 sweeps and reaches
1e-3 to 3e-3.

**Why CG first.** Decision 1's option (1) is the one method that is NumPy only, data-parallel per
cell, free of setup and convergent on every system measured, and it brings a steady product solve
from hours to minutes. Its weakness is growth: 140 to 154 iterations to 1e-6 on 40x15 and 791 to
794 on 200x75, 5.3 times for five times the cells per side. On a mesh twice as fine each way a correction would take about 1.3 s (eight times; extrapolated,
not measured), and multigrid preconditioning, which needed 5 to 9 iterations at 1e-6 on every grid
measured, is then the upgrade: the same CG loop with a different preconditioner. The probe's multigrid setup, 25 probing
passes per level in Python, is what makes it slow today; written as a stencil formula for the
Galerkin product it would cost a few array passes per level.

**What the exact solve shows.** SuperLU is not one of ADR-012 H's candidates; it was measured as
the reference. At 11,000 unknowns in two dimensions a sparse factorization costs 34 ms, so it is the
fastest option here and removes the stop altogether. Its costs are the dependency and the loss of
the per-cell structure REQ-S08 exists for (decision 2).

## B. The stop (REQ-S08, REQ-S04; decision 3)

**One measure.** The residual of the p' equation, `b + A p'`, is the corrected faces' mass imbalance
cell by cell (the report, section 2.4, checked to 6e-14 of the largest face flux on every system).
So the relative residual is the fraction of u*'s imbalance the correction leaves, the quantity
REQ-S04 and the stopping rule's continuity conditions read.

**Not solver-independent.** At the same relative residual in the 2-norm the methods leave different
errors (the report, section 7.4): CG with a diagonal preconditioner, like Jacobi, removes the rough
part of the residual first and leaves the smooth part, which carries most of p' and all of the net
outflow; multigrid removes both. In the outer loop that shows (the report, section 8):

| Relative residual | Outer count | Converged field, 40x15 | Stopping rule, 80x30 at Re 90 |
|---|---|---|---|
| 3e-1 | Diverges with CG | - | Never meets velocity_step with CG |
| 1e-1 | Diverges with CG, smoothed aggregation, and on 80x30 standalone multigrid; holds with MG-PCG | 1.2e-3 to 3.2e-3 m/s from exact with multigrid | Never meets error_estimate |
| 1e-2 | Holds with every solver | 0.041 m/s with CG; 1.7e-3 to 1.8e-3 with multigrid and pyamg | Never meets error_estimate |
| 1e-4 | Holds | 1.3e-6 to 4.2e-4 m/s, within the path floor | Never meets error_estimate |
| 1e-8 | Holds | 2.4e-8 m/s (CG) | Meets it at direct's 588 |
| exact | Holds | - | Meets it at 588 |

The path floor is how far converged runs with the exact correction land apart when only the path
changes (nine or eleven momentum sweeps, alpha_velocity 0.45 or 0.55): up to 4.8e-4 m/s, at the same
cell. Restarting each of two converged states with the other correction left both in place, so the
room has nearby steady states and the inner tolerance, like any change of path, picks among them.

**Why the stopping rule needs an absolute bound.** In the 80x30 room at Re 90 the converged state
has b standing on the outlet cells, the pressure rising 0.088 Pa every outer iteration: the
zero-gradient outlet value is restored before every prediction and the correction answers it with a
uniform p' (the report, section 8.3). A correction with relative residual r leaves `r b` in the
corrected faces every iteration, so the per-cell condition holds only if `r ||b||` is below
`mass_imbalance_tol`; there that means r below about 2.5e-7. 1e-8 met it; 1e-4 did not, in 20,000
outer iterations with the velocity frozen to 1e-17.

**The default.** 1e-8 met every condition measured and costs little with CG, whose iterations go mostly to the
first four decades: 852 to 861 to 1e-8 against 672 to 703 to 1e-4 on 200x75. The
rounding floor (1e-13 of rho times the inflow in the 2-norm) keeps a right-hand side at rounding from
running to the cap. Reaching `max_pressure_iter` is recorded, not silent (ECR-003 section 10).

## C. Dependencies, the C policy and the GPU path (decisions 2 and 5)

The probes ran SciPy 1.18.1 and pyamg 5.3.0 in a separate environment; `requirements.txt` is
unchanged. With decision 1's option (1) the solver needs NumPy only. Phase 6 then ports one loop:
per iteration a five-point product and a diagonal scaling (one thread per cell), three vector
updates (one thread per cell) and three sums (a library reduction each). REQ-N03's equivalence at
1e-10 meets one difference from today: a GPU sums in another order, so iterates differ in their last
bits, and the comparison has to be made on the corrected faces at the default tolerance rather than
iterate by iterate. CLAUDE.md's language policy puts performance-critical loops in C through
ctypes; no C is needed now (decision 5), and the CG kernel is the five-point product the Jacobi
sweep already was.

## D. Configuration and modules (REQ-C01, C02; decision 4)

**Keys.** `solver.pressure_rtol`, a float in (0, 1), default 1e-8 in every committed file,
validated for type, range, NaN and bool. `solver.pressure_tol` refused at load with a message naming
`pressure_rtol`. `solver.max_pressure_iter`, a positive integer, the CG iteration cap; 5,000 in the
committed files (about 860 needed at 1e-8 on 200x75, 303 on 80x30 at Re 90, 150 to 175 on 40x15).

**Draft contract** (SYSTEM.md section 4 gains it when ECR-003 is accepted).

```
pressure.py:
    PressureCorrector:
        __init__(mesh, config, boundary)        # reads rho, alpha_pressure,
                                                # max_pressure_iter, pressure_rtol
        needs_pin, pin_cell                     # unchanged
        mass_imbalance(u, v) -> [ny, nx]        # unchanged
        coefficients(a_p_u, a_p_v) -> PressureCoefficients   # unchanged
        correct(prediction, p) -> PressureCorrection
            p' by Jacobi-preconditioned CG from zero to
            ||b + A p'||_2 <= pressure_rtol ||b||_2, a rounding floor or the cap;
            closed domain: b projected onto the range, p' pinned after
    PressureCorrection: u, v, p, p_prime, iterations: int   # renamed from sweeps
solver_staggered.py:
    StaggeredSolver.last_pressure_iterations: int           # renamed
stopping.py:
    IterationState.pressure_iterations: int                 # renamed
```

**Cascade.** ECR-003 section 7.4.

## E. What this design does not decide
The outlet treatment: the standing imbalance at the outlets belongs to ECR-002 step 3 (decision 6).
The product's outer count: no laminar solve converged on 80x30 or 200x75; the k-epsilon count is
ECR-002 step 5's. The product mesh: ADR-010, ADR-011 and ADR-012 leave it to a measurement on the
product case. Whether the near-neutral direction found on 40x15 exists in the turbulent product
solve: not measured.

## Consequences
**Positive.** A steady product solve takes minutes, not hours or days. The stop measures what it
claims: the fraction of the imbalance a correction leaves. Its default lets the stopping rule's
continuity conditions hold wherever they were measured, including the standing-imbalance state.
No new dependency; the GPU path is one loop.

**Negative.** REQ-S08's single array operation per iteration becomes a product and three global
sums. Every laminar result changes beyond rounding and the laminar baseline is retaken. CG's cost
grows faster than the cells on a finer mesh. SuperLU, measured 3.6 times faster on the product mesh,
is not taken. The configuration files all change.

## Alternatives considered
Weighted Jacobi with a relative stop (millions of sweeps per correction at 1e-8). Gauss-Seidel and
SOR, not measured: sequential per sweep, against REQ-S08's rationale, and still a sweep that passes
information one cell at a time. A looser default (decision 3, options 2 and 3). Each option of each
decision carries its consequence above.

## Sources
1. `docs/reports/pressure_solver_ecr003.md`: item 0 (section 6), one correction with each candidate
   (7), the outer loop (8), a steady solve (9); the probes in its appendices.
2. `docs/reports/pressure_correction_step5.md`, sections 3 and 5: the -1 eigenvalue and the weight.
3. `docs/ADR/ADR-012-turbulence-model.md`, decision 5 and section H;
   `docs/reports/product_case_reynolds.md`, section 5.
4. `docs/reports/ecr002_step0_frozen_viscosity.md`, sections 6.4 and 7.6, and test 34b: the 40x15
   room converging with ten momentum sweeps.

Hestenes and Stiefel (1952), conjugate gradients; Saad (2003), preconditioned Krylov methods;
Briggs, Henson and McCormick (2000), multigrid and Galerkin coarse operators; Ruge and Stueben (1987)
and Vanek, Mandel and Brezina (1996), algebraic multigrid. Cited from their standard use.
