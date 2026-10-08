# ADR-013: Pressure Correction Solver: Jacobi-Preconditioned Conjugate Gradients

## Status
Accepted (built). Built in ECR-003 step 1 (prompts 37 to 37c, PR 62, merged 2026-10-07) and
measured on the laminar baseline in step 2 (prompt 38); the section "Planned against built" was
added at step 3 on 2026-10-07, when ECR-003 closed [5, 6].
Accepted 2026-10-06 by Alex, each of the six decisions below as ranked first. Proposed 2026-10-06, with ECR-003 (`docs/ECR/ECR-003-pressure-solver.md`), from the measurements in
`docs/reports/pressure_solver_ecr003.md` [1]. Premise review 36 and `/cfd-test 36` found no Critical
and four distinct Bugs between them; the fix pass of prompt 36b revised this design on their
findings, with the runs it cites added to the report as its section 12, and `/cfd-test 36b`'s
text findings were applied by the orchestrator. Alex took the six decisions below on 2026-10-06,
each as ranked first. As
accepted it amends REQ-S08 and supersedes ADR-010's weighted Jacobi sweep as the pressure solve.
Bracketed numbers point at the sources at the end; section numbers in the decisions point at this
design's sections, "the report" at [1].

## Decisions for Alex
Each decision is a picture first, then the options ranked with their consequence, then the section
that carries the detail. The ranking is the builder's, from the measurements.

**Taken by Alex, 2026-10-06.** Every recommendation below. The options stay as they were put.

1. The solver: Jacobi-preconditioned conjugate gradients in NumPy, with multigrid preconditioning
   of the same CG loop named as the successor if a finer mesh makes CG too slow. Chosen for the GPU
   path (one thread per cell, plus three reductions per iteration) as much as for speed.
2. Dependencies: NumPy only; no SciPy or pyamg at runtime.
3. The stop: relative residual `pressure_rtol`, default 1e-8, with the rounding floor and the cap
   reported. To be revisited after ECR-002 step 3 rebuilds the outlets, with the guard and
   tight-start options as the candidates.
4. The keys: `pressure_rtol` in [1e-10, 1); `pressure_tol` refused at load with a message;
   `max_pressure_iter` the CG cap, 5,000 in committed files; the weighted sweep and `JACOBI_WEIGHT`
   removed with their tests.
5. The C path: deferred to Phase 6, where the CUDA kernel becomes a CG kernel.
6. The two findings outside this change: the outlet drift to ECR-002 step 3 (an issue), the
   finer-grid non-convergence to ECR-002 step 5.

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
Its iterations grow with the cells per side: on a mesh twice as fine each way a correction costs
about 7.5 times as much (1.9 times the iterations on four times the cells; measured from 200x75 to
400x150, 0.15 s to 1.1 s, section A). The 0.16 s is the probe's loop, which forms the product from
five shifted NumPy arrays; the same loop with a sparse-matrix product took about 1.5 times less
(section A).
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
(2) is 3.6 times faster per outer iteration and exact; against (1) with a sparse-matrix product its
lead is about two times.

**2. Whether to take SciPy and pyamg as runtime dependencies (section C).**
*Picture.* The project's code needs only NumPy, PyYAML, matplotlib and h5py to run, and its
compiled part is planned in C and CUDA. SciPy is a large, standard numerical library; pyamg is a
smaller one built on it. The probes used both, in a separate environment.
*Options, ranked.*
(1) NumPy only. Goes with decision 1's option (1) or (4). The CG loop maps one to one onto a CUDA
kernel set (the five-point product, vector updates, sums), as REQ-S06 and REQ-N03 plan for the
inner loop; CLAUDE.md's C policy covers it if a CPU kernel is ever needed (section C).
(2) SciPy at runtime. Needed for decision 1's option (2); also gives a library CG to test against,
and a sparse-matrix product about 1.5 times faster per CG iteration than the probe's NumPy one.
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
about what is left: measured, it left anything from 0.003% to 95% of the imbalance. With conjugate
gradients the stop is the fraction left over. Looser is cheaper: one part in a hundred costs a third
of what one part in a hundred million costs per correction. The measurements say three things about
how loose it can be.
- *Starting from rest is the hard part.* At one part in ten, the first correction of an air field at
  rest leaves 85% of the supply unbalanced, and the 40x15 and 80x30 rooms at real air blow up within
  three outer iterations (at ten and a thousand times air's viscosity they do not). After 100 tight
  corrections the same one part in ten converges, in about 2,740 to 2,760 outer iterations against
  the exact correction's 2,822 (2,739 and 2,760 in two CG implementations, test 36b). One part in three blows up even after the tight start.
- *The coarse room has more than one steady solution.* On 40x15 at least three states solve the
  same discrete equations, with the same return faces shut: two of them lie 4.8e-4 and 0.041 m/s
  from the exact correction's at one cell, and each stays put when the exact iteration is continued
  from it. Which one a run reaches depends on its path. A different under-relaxation picks one; so
  did one part in a hundred from rest, which reached the 0.041 m/s one. None of them is wrong. After the tight start, one part
  in ten lands 1.7e-4 to 2.7e-4 m/s and one part in a hundred 1.0e-4 m/s from the exact correction's
  state. At one part in ten even the implementation's rounding moves where a run lands: two CG
  implementations with the same stop reached states 9.9e-5 m/s apart, where at one part in a hundred
  they agreed to 4e-6 m/s (test 36b). Section C's fixed-count protocol is what keeps REQ-N03's test
  meaningful at a loose level.
  What a tight correction buys here is reproducibility: the exact correction's solution, to 2.4e-8
  m/s at one part in a hundred million.
- *One room needs a tight correction at its end, not its start.* Under the committed outlets, where
  every outlet face copies its neighbour, the 80x30 room at a thousand times air's viscosity settles
  with the outlets returning the same imbalance every outer iteration while the pressure climbs. The
  stopping rule's per-cell continuity there holds only if every correction leaves less than about
  3e-7 of that imbalance. Looser levels never stop. This is a defect of the committed outlet
  treatment (decision 6), which ECR-002 step 3 rebuilds. It is the one reason for a tight default
  that does not come from the start, and it may not survive step 3.

*Options, ranked.*
(1) The relative residual `||b + A p'||_2 <= pressure_rtol ||b||_2`, default 1e-8, with a floor at
rounding and `max_pressure_iter` as the iteration cap, a capped correction reported (section B). Met
every condition measured: the start from rest, the outer counts, the exact correction's solution
on 40x15, and the stopping rule on 80x30 at Re 90 under both the report's outlets and the committed
ones. Needs nothing configured beside it and works under either stopping rule. Per correction it
costs 1.2 times 1e-4 and 3.2 times 1e-2 (2.7 times 1e-2 per outer iteration on 200x75). Its weak
point: it keeps the per-cell condition only while `1e-8 ||b||` is below `mass_imbalance_tol`, which
shrinks with the cell volume (3.2e-9 kg/s per metre on 200x75 at ADR-011 G's value, 8e-10 on
400x150), and how large `||b||` stands at the product's converged state is not measured.
(2) A looser relative level with an absolute guard: a correction ends only when the relative level
is met and its worst cell is below a fraction of `mass_imbalance_tol`. This removes (1)'s weak point,
since the guard holds the per-cell condition whatever `||b||` is. Where `||b||` is large, as from
rest, the guard sets the stop and asks for more than the relative level, which also makes the start
tight; where `||b||` has decayed, the relative level sets it. Not run in the outer loop; its level,
its fraction and its cost over a solve are not measured.
(3) A tight start, then a looser level: 1e-8 for a set number of corrections, then 1e-1 or 1e-2.
Measured on one grid: after 100 tight corrections 1e-1 and 1e-2 converge on 40x15 within 3% of the
exact correction's count and land inside the spread of its solutions; 3e-1 still blows up. After the
switch it saves 2.7 times per outer iteration on 200x75 at 1e-2 and about nine times at 1e-1 (the
report, sections 7.2 and 9). The switch point is a new parameter, and on its own the looser level
never meets the stopping rule where the outlets hold a standing imbalance, so it needs (2)'s guard
there.
(4) 1e-4 from the start. Measured: keeps the counts and lands 1.4e-4 m/s from the exact correction's
state on 40x15; never meets the stopping rule on 80x30 at Re 90 under either outlet treatment. 1.2
times cheaper than (1).
(5) Today's per-sweep change in pascals. Not a measure of the error; with a cap it froze the 80x30
probe room with 47% of the supply unaccounted.
*Recommendation.* (1). It is the only option measured to meet every condition; (2) is not measured,
and (3) and (4) alone fail the standing-imbalance room. Its price is real: after a tight start, (3)
at 1e-1 would make an outer iteration on 200x75 about nine times cheaper. Revisit after ECR-002 step
3: if the rebuilt outlets remove the standing imbalance, the case for (1) rests on the start and on
reproducibility alone, and (2) or (3), run in the outer loop on the product mesh first, are the
candidates to replace it.

**4. The configuration keys and what happens to the weighted sweep (section D).**
*Picture.* Every configuration file today says `pressure_tol: 1.0e-6` (or 1e-8) meaning pascals per
sweep, and `max_pressure_iter` meaning sweeps. Under the new stop both numbers mean something else.
*Options, ranked.*
(1) A new key, `pressure_rtol`, dimensionless in [1e-10, 1) (section D); `pressure_tol` refused at load with a
message that names the change; `max_pressure_iter` kept as the CG iteration cap, still a required
key, set to 5,000 in the committed files (861 iterations to 1e-8 on 200x75 and 1,643 on 400x150).
The weighted sweep and `JACOBI_WEIGHT` removed with the tests that pin them; the step 5 report keeps
their evidence. Every committed file that names the key, the label or the count changes in the same
pull request, test fixtures included (ECR-003 section 7).
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
(1) Defer to Phase 6. With option (1) of decision 1 a steady product solve takes minutes in NumPy,
and a sensitivity pair at twice the resolution each way about 1.1 s per correction; the compiled
loop is Phase 6's deliverable under REQ-S06 and REQ-N03, now a CG loop.
(2) A C kernel for the CG iteration now, through ctypes. Saves an unmeasured factor on the product
mesh (a library sparse product alone took about 1.5 times less per iteration), and adds a second
implementation to keep equivalent before the solver's physics has settled.
(3) A C setup for geometric multigrid now. Only worth it if decision 1 takes option (4).
*Recommendation.* (1).

**6. Two findings outside this change (section E).**
*Picture.* The measurement found two things about the flow solver that the pressure solver does not
cause and cannot fix. In the 80x30 room at Re 90 the pressure rises by the same amount every outer
iteration, forever, because the outlets' rule for the air leaving the room and the correction undo
each other. It does so under the committed outlets (0.059 Pa per outer iteration, stopping by the
error estimate at 1,197) as under the report's T3 (0.088 Pa, at 588). And the laminar room that
converges on 40x15 converges on neither 80x30 nor 200x75, with ten momentum sweeps and exact
corrections.
*Options, ranked.*
(1) Record both where the work that owns them will read them: the outlet finding as an issue for
ECR-002 step 3, which rebuilds the outlets, with the report's sections 8.3 and 12 as its evidence;
the finer-grid finding as an input to ECR-002 step 5, whose question it is.
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
Its section 12 adds the controls premise review 36 and test 36 found missing, rerun.

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
timed seconds per sweep, 0.20 to 0.22 ms (the report, section 7.1). Today's 1e-6 Pa stop takes
28,000 to 378,000 sweeps and reaches 1e-3 to 3e-3.

**What the CG time is made of.** CG's milliseconds are the probe's loop, whose five-point product
is five shifted NumPy arrays. Run in one process on the same three systems, the same loop with a
SciPy sparse (CSR) product took 87 to 123 ms to 1e-8 against 134 to 183 ms, about 1.5 times less
per iteration, with the same iteration counts (the report, section 12.6; test 36 found 1.7
times). So part of the gap to SuperLU is the product's implementation. A CSR product needs SciPy
(decision 2) or compiled code (decision 5); the comparison here is for weighing those, and the
0.16 s is what decision 1's option (1) delivers as specified.

**Why CG first.** Decision 1's option (1) is the one method that is NumPy only, data-parallel per
cell, free of setup and convergent on every system measured, and it brings a steady product solve
from hours to minutes. Its weakness is growth. On the systems captured at outer 100 under T3 it takes
164 iterations to 1e-8 on 40x15, 861 on 200x75 and 1,643 on 400x150: 1.9 times per doubling of the
cells per side. A correction on 400x150 took 1.1 s against 0.15 s on 200x75, timed interleaved, 7.7
times by the medians and 7.2 by the fastest solves; the work grows 7.7 times (the report, sections
12.1 and 12.2). The Annex 20 room on 216x72, the grid ADR-012 G plans for VAL-016, takes 872
iterations, 1.2 times its 180x60 count and level with the product's 200x75 count on 1.04 times the
product's cells (the loop forms its product on the full grid): about the same cost per correction,
0.91 to 1.21 times the product's across three timings (the report, section 12.1; the premise
review; test 36b).
Multigrid preconditioning, which needed 5 to 9 iterations at 1e-6 on every grid measured, is the
upgrade when a finer mesh makes CG too slow: the same CG loop with a different preconditioner. The
probe's multigrid setup, 25 probing passes per level in Python, is what makes it slow today;
written as a stencil formula for the Galerkin product it would cost a few array passes per level.

**What the exact solve shows.** SuperLU is not one of ADR-012 H's candidates; it was measured as
the reference. At 11,000 unknowns in two dimensions a sparse factorization costs 34 ms, so it is the
fastest option here and removes the stop altogether. Its costs are the dependency and the loss of
the per-cell structure REQ-S08 exists for (decision 2).

## B. The stop (REQ-S08, REQ-S04; decision 3)

**One measure.** The residual of the p' equation, `b + A p'`, is the corrected faces' mass imbalance
cell by cell (the report, section 2.4, checked to 6e-14 of the largest face flux on every system).
So the relative residual is the fraction of u*'s imbalance the correction leaves, the quantity
REQ-S04 and the stopping rule's continuity conditions read. CG stops on the residual it updates by
recursion, not the one it would form from p'. On the four systems of the report's section 12.1
(200x75 and 400x150, the Annex 20 room on 180x60 and 216x72) the two agreed at exit to within
2.3e-14 of `||b||` at 1e-4, 1e-6 and 1e-8. The solve still checks: when the recursive residual
meets the stop, one more product forms the true one, and the correction ends only if that meets it
too.

**Not solver-independent.** At the same relative residual in the 2-norm the methods leave different
errors (the report, section 7.4): CG with a diagonal preconditioner, like Jacobi, removes the rough
part of the residual first and leaves the smooth part, which carries most of p' and all of the net
outflow; multigrid removes both. In the outer loop from rest that shows (the report, section 8):

| Relative residual | Outer count | Converged field, 40x15 | Stopping rule, 80x30 at Re 90 |
|---|---|---|---|
| 3e-1 | Diverges with CG, from rest and after 100 corrections at 1e-8 | - | Never meets velocity_step with CG |
| 1e-1 | Diverges from rest with CG, smoothed aggregation, and on 80x30 standalone multigrid; holds with MG-PCG; holds with CG after 100 corrections at 1e-8 | 1.2e-3 to 3.2e-3 m/s from exact with multigrid; 2.7e-4 with CG after the tight start | Never meets error_estimate |
| 1e-2 | Holds with every solver | 0.041 m/s with CG from rest, 1.0e-4 after the tight start; 1.7e-3 to 1.8e-3 with multigrid and pyamg | Never meets error_estimate (T3 and T0) |
| 1e-4 | Holds | 1.3e-6 to 4.2e-4 m/s | Never meets error_estimate (T3 and T0) |
| 1e-8 | Holds | 2.4e-8 m/s (CG) | Meets it at direct's count (T3 588, T0 1,197) |
| exact | Holds | - | Meets it (T3 588, T0 1,197) |

**Several steady solutions.** Converged runs with the exact correction land apart when only the path
changes: nine or eleven momentum sweeps, or alpha_velocity 0.45 or 0.55, move the 40x15 field by up
to 4.8e-4 m/s at one cell. That is not drift. The alpha 0.45 state, continued 4,000 outer iterations
with the exact correction at alpha 0.5, stays 4.8e-4 m/s from direct's while its step falls to 4e-16
m/s; the CG 1e-2 state, continued the same way, stays 0.041 m/s from it; all three states have the
same single return face shut (the report, section 12.3; test 36 found the same two-state picture by
its own route). So the steady equations on this grid have at least three solutions near one cell,
and the 4.8e-4 m/s the report called the path floor is a lower bound on how far apart solutions lie,
not a floor on any run's error. A looser correction from rest is one more change of path and can
pick another solution; a tight one reproduces the exact correction's. Their mechanism is not found
(section E).

**Why the stopping rule needs an absolute bound where the outlets hold an imbalance.** In the 80x30
room at Re 90 the converged state has b standing on the outlet cells, the pressure rising every
outer iteration: the zero-gradient outlet value is restored before every prediction and the
correction answers it with a uniform p' (the report, section 8.3). The same holds under the
committed outlets, where every outlet face is extrapolated (0.059 Pa per outer iteration, `||b||`
0.069) as under T3, where the hood holds its flow (0.088 Pa, 0.079) (the report, section 12.4). A
correction with relative residual r leaves `r b` in the corrected faces every iteration, so the
per-cell condition holds only if `r ||b||` is below `mass_imbalance_tol`; there that means r below
about 3e-7. 1e-8 met it under both treatments; 1e-2 and 1e-4 never did in 20,000 outer iterations,
though each met the velocity-step stop. What the stop then accepts is a velocity that is steady
and a pressure that is not: the outlet treatment's defect, which ECR-002 step 3 owns (decision 6).
The tight correction lets the rule stop there; it does not make that state right.

**The default.** 1e-8, for the reasons decision 3 gives: it met every condition measured, from rest,
and costs little with CG, whose iterations go mostly to the first four decades (852 to 861 to 1e-8
against 672 to 703 to 1e-4 on 200x75). The rounding floor stops a correction whose residual's
2-norm is below `1e-13 F`, where F is the stopping rule's flux scale: rho times the inflow, or on a
closed domain rho times the largest prescribed boundary velocity times the longer side, the
definition `StaggeredSolver` already uses for error_estimate. The corrector computes F itself, so
the floor is the same under either rule. 1e-13 is a module constant in `pressure.py` with that
reason, not a configured tolerance: the identity between the residual and the corrected faces'
imbalance holds to 6e-14 of the largest face flux, so below about 1e-13 F the residual is the face
arithmetic's rounding. On the cavity the floor, not the relative level, ends a converged solve's
corrections: at outer 100 of VAL-002 80x80 it stopped CG at 1.4e-8 (the report, section 12.2).

**The cap.** `max_pressure_iter` caps CG's iterations. A correction that stops there, rather than at
its relative level or the floor, says so: `PressureCorrection.reached_cap` is true, and the solver
counts such corrections in each solve (`StaggeredSolver.pressure_cap_hits`), logs a warning at the
first and records the count, which the harness writes with every row. Under velocity_step an outer
iteration whose correction reached the cap cannot stop the solve: a truncated correction makes the
velocity step small without converging the field, which is how the 80x30 probe room froze with 47%
of the supply unaccounted while velocity_step reported convergence (the report, section 8.2). Such a
solve runs on to `max_simple_iter`, ends unconverged, and the count says why. Under error_estimate the
continuity conditions already refuse that state; the count is recorded the same way. The product
configuration runs velocity_step until ADR-011 decision 6 and ECR-002's VAL-018 move it to
error_estimate, so until then this refusal, not the continuity conditions, is the guard it has. The
committed configuration itself does not reach a freeze: it goes non-finite at outer 7,635 on 80x30
and never meets the velocity step on 200x75 (the report, section 12.5).

## C. Dependencies, the C policy and the GPU path (decisions 2 and 5)

The probes ran SciPy 1.18.1 and pyamg 5.3.0 in a separate environment; `requirements.txt` is
unchanged. With decision 1's option (1) the solver needs NumPy only. Phase 6 then ports one loop:
per iteration a five-point product and a diagonal scaling (one thread per cell), three vector
updates (one thread per cell) and three sums (a library reduction each).

**REQ-N03 stands as written.** A GPU sums in another order than NumPy. Measured on five captured
systems with every sum of CG reversed, or taken in blocks of 256 in reverse block order: the
iteration counts were the same at 1e-8 and at 1e-10, p' moved by at most 6e-14 Pa and the corrected
faces by at most 1.6e-12 m/s. The converged 40x15 room with reversed sums stopped at the same outer
counts, its field within 1.1e-14 m/s (the report, section 12.2; test 36 found 1.4e-12 m/s by its own
route). That is two orders inside REQ-N03's 1e-10. The narrower risk is a correction whose residual
sits on the stop: the two loops can then stop one iteration apart, and one more iteration at 1e-8
moved the corrected faces by 4e-10 to 6e-9 m/s on the open systems, more than 1e-10. So Phase 6's
equivalence test runs both loops for the same number of CG iterations in each correction (the
NumPy loop's count, handed to the CUDA loop) and compares every output array at 1e-10; whether the
two stop tests agree is checked apart from it, as a count of corrections where they differ.

CLAUDE.md's language policy puts performance-critical loops in C through ctypes; no C is needed now
(decision 5), and the CG kernel is the five-point product the Jacobi sweep already was.

## D. Configuration and modules (REQ-C01, C02; decision 4)

**Keys.** `solver.pressure_rtol`, a float in [1e-10, 1), 1e-8 in every committed file. The lower
bound: on the product's first correction from rest the true residual cannot go below about 1.3e-13
of the flux scale, so a configured level at or below about 1e-12 runs that correction to the cap and
the velocity-step rule then refuses to stop on it (test 36b, S4); 1e-10 keeps two orders of margin
above the attainable level and two below the default. It is validated for
type, range, NaN and bool. `solver.pressure_tol` refused at load with a message naming
`pressure_rtol`. `solver.max_pressure_iter`, a required positive integer as today, now the CG
iteration cap; 5,000 in the committed files: 861 iterations are needed at 1e-8 on 200x75, 1,643 on
400x150, 303 on 80x30 at Re 90 and 162 to 173 on 40x15 (the report's section 8.1 median and
largest), so 5,000 leaves three times the largest.

**Draft contract** (SYSTEM.md section 4 gains it when ECR-003 is accepted).

```
pressure.py:
    PRESSURE_SOLVER_VERSION = 2                 # 1 was the weighted sweep; saved
                                                # solves keyed on the solver read it
    RESIDUAL_FLOOR = 1e-13                      # times the flux scale F; rounding
    PressureCorrector:
        __init__(mesh, config, boundary)        # reads rho, alpha_pressure,
                                                # max_pressure_iter, pressure_rtol;
                                                # closed domain: raises if the cells
                                                # with an equation are not one
                                                # connected component
        needs_pin, pin_cell                     # unchanged
        mass_imbalance(u, v) -> [ny, nx]        # unchanged
        coefficients(a_p_u, a_p_v) -> PressureCoefficients   # unchanged
        correct(prediction, p) -> PressureCorrection
            p' by Jacobi-preconditioned CG from zero to
            ||r||_2 <= pressure_rtol ||b||_2 or ||r||_2 <= RESIDUAL_FLOOR F,
            r = b + A p' formed once more at exit and checked,
            or to max_pressure_iter iterations;
            closed domain: b projected onto the range, the stop read on the
            projected residual, p' pinned after
    PressureCorrection: u, v, p, p_prime,
        iterations: int                         # renamed from sweeps
        reached_cap: bool                       # stopped at max_pressure_iter
solver_staggered.py:
    StaggeredSolver.last_pressure_iterations: int           # renamed
    StaggeredSolver.pressure_cap_hits: int                  # capped corrections, last solve
    velocity_step: an outer iteration whose correction reached the cap does not stop the solve
stopping.py:
    IterationState.pressure_iterations: int                 # renamed
```

**One component.** The projection removes one mean and the pin fixes one cell, so a closed domain
must have its cells with an equation in one connected component. Every captured system had one
(the report, section 6), and today's Jacobi pins one cell too. A closed room split by an obstacle
would need a mean and a pin per component; the constructor refuses it rather than solving it
wrongly.

**Cascade.** ECR-003 section 7.4.

## E. What this design does not decide
The outlet treatment: the standing imbalance at the outlets belongs to ECR-002 step 3 (decision 6).
The product's outer count: no laminar solve converged on 80x30 or 200x75; the k-epsilon count is
ECR-002 step 5's. The product mesh: ADR-010, ADR-011 and ADR-012 leave it to a measurement on the
product case. Whether the turbulent product solve has several steady solutions as the laminar 40x15
room does, and what makes that room have them (a degenerate face or cell beside the litho tool's top
corner is the obvious place to look): not measured. No ECR-003 criterion holds a field to the
spread of those solutions.

## Planned against built

Written 2026-10-07 at ECR-003 step 3, after step 1 merged (PR 62) and step 2 measured the
laminar baseline. Each row is a decision or section of this design, or the line of ECR-003 that
planned it, against what was built and measured.

| ADR-013 planned | Built | Why, and where measured |
|---|---|---|
| Decision 1, section A: Jacobi-preconditioned CG in NumPy, the solve inside `PressureCorrector.correct` (D's draft) | `apply_operator`, `conjugate_gradient` and `ConjugateGradientResult` module-level in `src/pressure.py`; `correct` calls them | The tests plant a lying operator and count iterations, the probes call the loop on captured systems, and Phase 6 can run it for a fixed count (section C). The built loop reproduces the probe's on the three recaptured 200x75 systems bit for bit, nine solves [5, sections 3 and 10] |
| Decision 2, section C: NumPy only | As planned; `requirements.txt` unchanged | |
| Decision 3, section B: `pressure_rtol` 1e-8, the floor `1e-13 F` with F the stopping rule's flux scale, the true residual checked at exit | As planned. F is `PressureCorrector.flux_scale`, public, and the solver's error_estimate rule reads it, so the floor and the rule share one F; `ZERO_SCALE` moved from `solver_staggered.py` to `pressure.py`, one guard for both. A failed exit check restarts from the true residual | [5, section 10]. The floor ends every correction of the 20x20 cavity from outer 235 [5, section 13.2], and the late corrections of every step 2 case, where the relative figure rises to 6e-3 (channels) and 7e-2 (cavity) while the corrected faces' worst cell stays at most 8.2e-12 kg/s per metre [6, section 6] |
| Section B, the cap: `reached_cap`, `pressure_cap_hits`, a warning, no velocity_step stop on a capped correction | As planned; every harness row records `pressure_cap_hits` | Step 2's six rows record none. The Jacobi rows they replace had 410, 129 and 3,728 corrections at their caps, which no row could record [6, section 6] |
| Section D: a closed domain whose cells with an equation are not one component refused | Built, a cell having an equation when it is non-SOLID with a non-SOLID 4-neighbour (a sealed cell is not counted); and an open domain with a component no outlet cell reaches refused as well | Not in D's plan; such a domain caps every correction (review 37 S7). 13 of 13 committed configurations, presets and transport cases are accepted [5, section 13.2] |
| Decision 4, section D: `pressure_rtol` in [1e-10, 1), `pressure_tol` refused, `max_pressure_iter` 5,000, the weighted sweep and `JACOBI_WEIGHT` removed | As planned; the refusal's range is formatted from `PRESSURE_RTOL_BOUNDS`. The harness records the solver keys from `SOLVER_KEYS` in `config.py`, the one list (issue 38), not from a `SOLVER_PARAMETERS` of its own as ECR-003 7.1 named | [5, sections 10 and 13.2] |
| Section D's contract: `PressureCorrection.iterations` and `reached_cap`; `IterationState.pressure_iterations` | As planned, and `products`, every product with the operator (one per iteration and one per true-residual check), on `ConjugateGradientResult` and `PressureCorrection`, with `IterationState.pressure_products` as its last field | Review 37 S5: the harness counts its work in operator products, row schema 2 with `work.inner_iterations` and `work.inner_products`; stored rows keep `inner_sweeps`. ECR-003 7.1 described the work as "plus vector operations"; the built definition counts stencil evaluations only, as every method's does [5, section 13.2] |
| ECR-003 7.1: the label `staggered-cg` in the harness, `staggered-jacobi` retired | As planned, with `STAGGERED_METHODS` and `STAGGERED_METHOD` defined once in `pressure.py`, looked up by `PRESSURE_SOLVER_VERSION`, and imported by the harness, the viewer and `self_convergence.py` | Review 37 S4: a version without a label fails at import [5, section 13.2] |
| ECR-003 7.1: four saved-solve reuses in `stopping_probe.py` keyed on the version; `val001_order.py` "no edit of its own" | Every saved truth read through one identity check, six reads in all; `val001_order.py`'s reuse key carries `PRESSURE_SOLVER_VERSION` | Review 37 B1 found the fifth read, the fix pass the sixth; the key had carried the solver's identity only through `pressure_rtol`'s name [5, section 13.1]. Test 37b's two missing tests are issue 63 |
| ECR-003 7.2: the transport criteria re-checked, not re-set | VAL-012 split by Alex on 2026-10-07: its requirement clause stays on the solver's VAL-001 faces, its lower clause moved to a planted field. Every other gate row passes unchanged | CG balances those faces to rounding (worst cell 6.2e-16 kg/s), so departure and bound were both rounding, their ratio on the 0.1 line (0.100 on Windows, 0.099 on Linux); ADR-011 G records the split. Step 2 re-ran the six gate files: 13 passed [6, section 8] |
| Criterion 3: the report's two rooms reproduce within 1% | Exactly: 1,209 and 2,822 on 40x15, 233 and 588 on 80x30 | [5, section 5] |
| Criterion 4: under 0.5 s per correction on 200x75 | 131 to 162 ms, the medians over three systems, with one BLAS thread; 1.03 to 1.10 s under OpenBLAS's default threads, the same iteration counts | Every timing in [1] and [5] set one BLAS thread. Above about 10,000 cells OpenBLAS splits each of CG's three reductions across threads at about a third of a millisecond apiece; below it, the step 2 cases among them, time and bits are the same either way. Whether the criterion holds the default setting, and how the setting is fixed, was decided by Alex on 2026-10-07: the code sets one BLAS thread for the pressure solve (threadpoolctl, one named constant), built in the cleanup pull request with the dot-product timing extended to a million elements [6, section 11] |
| The BLAS thread count (not planned; criterion 4 held it as an environment setting) | `PRESSURE_BLAS_THREADS = 1` in `src/pressure.py`, applied by `conjugate_gradient` around the CG loop with `threadpoolctl` (`user_api="blas"`, one `ThreadpoolController` built at import); `threadpoolctl>=3.2` added to `requirements.txt`, the one dependency this design added. Not a configuration key | Alex, 2026-10-07. One thread is no slower up to 9,600 elements and faster from 12,800 to 1,000,000, and threads pay only past a crossover between 1,000,000 and 1,500,000; the product mesh's vectors are 15,000. Entering and leaving the limit is 0.05% of a `val001_80x40` correction. The three 200x75 corrections take 134 to 139 ms with the thread variables unset (criterion 4 met under the default environment), and `val001_80x40`'s face hash is the step 2 hash, so the method version stays 2 [7] |
| Criterion 2 and the negative consequence: every laminar result changes beyond rounding and the baseline is retaken | VAL-001 4.108e-4 and 3.024e-3, VAL-002 u 1.057e-3 and v 7.356e-4 of the lid speed, each the Jacobi value to three figures; the faces moved by at most 1.1e-7 m/s on the channels and 7.3e-8 on the cavity. The channel outer counts fell 61% and 50%, the cavity's 0.27%. Six rows at 311034e, accepted by Alex on 2026-10-07 as ECR-002 criterion 1's baseline | The prediction's premise, that each Jacobi correction delivered its tolerance, was wrong: the median correction left 99.9% of the imbalance in the faces, so on the channels the stopping rule's condition (d) held only at zero crossings of the net outflow. Under CG conditions (b) to (d) hold from the first outer iteration and (a) alone sets the stop [6, sections 4 to 6]. The orders of convergence recorded under Jacobi are not retaken; the measured field differences can move none by more than 0.003 (VAL-001's 1.992) or 6e-4 (VAL-002's four) [6, section 10] |
| ECR-003 section 4: a tight level "costs little against a sweep" | On the 80x80 cavity the two CG rows take between about 2% and 18% more wall time than the two Jacobi rows of identical arithmetic, inside the 7% to 8% spread between identical runs, with the outer counts within 0.27%: 265 CG iterations per correction at the median against the Jacobi row's 192 sweeps on average at its loose stop. Measured; no cost difference resolved; no action taken | The cost case rests on the product mesh, where a sweep-based correction needs 28,000 to 378,000 sweeps (section A), not on the validation cases [6, sections 4 and 5]. PROJECT_PLAN's efficiency pass lists a tolerance loose early and tight near the stop as a candidate, not decided |
| Section C: REQ-N03 as written; Phase 6's test runs both loops for the same iteration count | Not built (Phase 6). Step 2 found the face bits depend on the BLAS kernel: one thread or many gives the same hash on one machine, and OpenBLAS picks its `ddot` kernel by CPU | ECR-002 criterion 1 compares hashes taken on one machine [6, section 7] |
| Decision 5: the C path deferred | Deferred; PROJECT_PLAN's Phase 6 deliverable `csolver/pressure_solve.cu` is a CG kernel | |
| Decision 6: the outlet drift to ECR-002 step 3, the finer-grid non-convergence to ECR-002 step 5 | Issue 61 for step 3; step 5's row in ECR-002 section 8 names both the finer-grid finding and the sweep result it retakes under CG | Step 3 notes in ECR-002, 2026-10-07 |

## Consequences
**Positive.** A steady product solve takes minutes, not hours or days. The stop measures what it
claims: the fraction of the imbalance a correction leaves. A correction cut short by the cap is
reported, and under velocity_step it can no longer end a solve. The default reproduces the exact
correction's solution where the room has several. No new dependency; the GPU path is one loop, and
REQ-N03 stands as written.

**Negative.** REQ-S08's single array operation per iteration becomes a product and three global
sums. Every laminar result changes beyond rounding and the laminar baseline is retaken. CG's cost
grows about 7.5 times per doubling of the cells per side. SuperLU, measured 3.6 times faster on the
product mesh, is not taken. The configuration files and the test fixtures all change. The default
lets the stopping rule's continuity conditions hold in the standing-imbalance state at 80x30 and Re
90, but what it then accepts is a steady velocity under a pressure still rising 0.059 to 0.088 Pa
every outer iteration (mean 53.6 Pa at the T3 stop: builder 36's `stall_direct.json`, rerun by test 36b at 53.567 Pa): the outlet defect of decision 6, not cured here.
Whether 1e-8 keeps the per-cell condition on the product mesh depends on a `||b||` not yet measured.

## Alternatives considered
Weighted Jacobi with a relative stop (millions of sweeps per correction at 1e-8). Gauss-Seidel and
SOR with red-black ordering, not measured: each half-sweep is data-parallel, so REQ-S08's rationale
survives, but it is still a sweep that passes information one cell per half-sweep, and SOR's rate
depends on a relaxation factor tuned to the slowest mode, which here changes with every outer
iteration. Chebyshev-accelerated Jacobi, not measured: it needs no global sums, so it keeps
REQ-S08's structure and is reproducible to the bit across summation orders, but it needs bounds on
the eigenvalues of `D^-1 A` and its iteration count is CG's worst-case bound, about 7,000 to 1e-8 at
a condition number of 5.4e5, against the 852 to 861 CG took. CG preconditioned by an incomplete
Cholesky factorization, IC(0), not measured: fewer iterations than the diagonal, but its two
triangular solves per iteration are sequential, as SOR's natural ordering is, against REQ-S08's
rationale. A looser default (decision 3, options 2 to 4). Each option of each decision carries its
consequence above.

## Sources
1. `docs/reports/pressure_solver_ecr003.md`: item 0 (section 6), one correction with each candidate
   (7), the outer loop (8), a steady solve (9), the controls added in 36b (12); the probes in its
   appendices.
2. `docs/reports/pressure_correction_step5.md`, sections 3 and 5: the -1 eigenvalue and the weight.
3. `docs/ADR/ADR-012-turbulence-model.md`, decision 5 and section H;
   `docs/reports/product_case_reynolds.md`, section 5.
4. `docs/reports/ecr002_step0_frozen_viscosity.md`, sections 6.4 and 7.6, and test 34b: the 40x15
   room converging with ten momentum sweeps.
5. `docs/reports/ecr003_step1_cg.md`: item 0 (section 3), criteria 3 and 4 (sections 5 and 6), the
   contracts that differ from section D's draft (10), the fix pass (13).
6. `docs/reports/ecr003_step2_baseline.md`: the rows (section 4), the prediction (5), why the
   channel counts moved (6), the controls (7), the transport gate (8), the orders (10).
7. `docs/reports/blas_threads.md`: the dot-product timing to 10,000,000 elements, the thread
   limit's overhead, criterion 4 under the default environment, the face hash under the limit.

Hestenes and Stiefel (1952), conjugate gradients; Saad (2003), preconditioned Krylov methods;
Briggs, Henson and McCormick (2000), multigrid and Galerkin coarse operators; Ruge and Stueben (1987)
and Vanek, Mandel and Brezina (1996), algebraic multigrid; Golub and Van Loan (2013), Chebyshev
iteration and incomplete factorizations. Cited from their standard use.
