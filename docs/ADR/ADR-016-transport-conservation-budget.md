# ADR-016: The Conservation Budget of the Implicit Transport Solve

## Status
Proposed, 2026-11-12 (prompt 78, branch `feature/conservation-budget`). The decision is
Alex's, of 2026-11-12, on the mechanism report of Phase 4's condition 1
(`docs/reports/budget_residual_mechanism.md`, "Alex's decisions (2026-11-12)"). This record
states it, derives the bound as built, gives the alternatives the mechanism report lists and
why none was taken alone, and the consequences. The build and its measurements are
`docs/reports/conservation_budget.md`.

## Context

REQ-T05 asks the transport solver to conserve each class's particles to 0.01% over a run.
Its basis was stated on 2026-11-12: a particle counter at the ISO 14644-1 Class 5 limit
resolves a concentration to about 10%, so a bookkeeping error below 1% of that cannot change
a reading, and 0.01% is ten times inside that.

Prompt 77 found what the budget's closure residual is. The implicit diffusion and deposition
solve, backward Euler by Jacobi sweeps, stops when its largest cell residual is within
`diffusion_tol` of the largest right-hand side. Summing the system over the non-SOLID cells,
every interior face's conductance appears once with each sign, so

    sum_P r_P = sum_P (V/dt)(C*_P - C_P) - sum_P D_P C_P,

and the solve changes the field's total by the sink, which the budget books, and by
`dt sum_P r_P`, which it books nowhere. That leftover is not zero before the solve
converges: a Jacobi sweep carries the residual's sum forward weighted by
`w_P = (sum_f G_f)_P / a_P`, which varies with `nu_t` and is smaller in a cell with a wall
face. It was the residual at the end and at the worst frame of every run, class and grid
measured, of either sign, and it moved by about ten per decade of `diffusion_tol`. At 1e-6
the 80x30 dust release is outside REQ-T05. On the dust release it grows with the grid, since
the step count grows faster than the leftover per step falls.

The max-norm stop bounds no sum, so no value of `diffusion_tol` keeps the leftover inside
REQ-T05 on every grid and run length.

## Decision (Alex, 2026-11-12)

The error accumulated from the unfinished implicit solve shall stay below REQ-T05's limit by
construction, on any grid and run length. Three parts, built together.

1. **Book it.** `MassBudget` gains the solve's leftover as its own term, `leftover`: each step
   adds `ImplicitResult.leftover`, `dt` times the sum of the final residual over the
   non-SOLID cells. The residual is defined with it,
   `initial + inflow + source - outflow - deposited - leftover - current`, so with every term
   booked the budget closes to rounding, and closure is again the check on bugs in the
   flux assembly that VAL-007 was written for. `relative()` keeps its meaning, the residual
   over the particles supplied; `leftover_relative()` is the leftover over them. The run
   file's budget rows carry the term as a last column. A file written before has no such
   column: it still reads, with the leftover None (unknown) and the residual as it was, and
   a restart from it is refused, since the leftover before the restart is unknown.
2. **Budget it per step.** The transport solve stops only when the max-norm test holds and
   the step's leftover is within its share,

       abs(leftover_n) <= f * L * S_n * dt_n / T,

   `L` = `transport.conservation_limit` (1e-4, REQ-T05), `f` = `transport.leftover_share`
   (0.1), `S_n` the class's particles supplied (initial, inflow and source) with step n's
   inflow and source booked, and `T` = `solver.t_end`. `implicit_step` takes the bound as
   `leftover_bound`; absent, the stop is the max-norm one alone, bit for bit, which is what
   the k-epsilon solves of k and eps use (REQ-S16's pattern). The leftover is formed from
   the residual the last stop test read, only once that test's max-norm part holds, and
   returned in `ImplicitResult`.
3. **Refuse a breach.** After every step, a class whose accumulated leftover reaches `L`
   times its supplied particles raises `ConservationBreachError`, and
   `scripts/run_scenario.py` reports it and exits 1, as it does a flow state that did not
   converge, leaving the last checkpoint written. With part 2 holding it cannot fire. It is
   the backstop for a solve the sweep cap stops before its share holds.

Both keys are in the base file and its copies and validated at load: `L` in (0, 1), `f` in
(0, 1], each a finite number and not a bool.

## The bound, as built

**Within one run.** The supply never falls: `initial` is fixed at the first step, `inflow`
adds an inward volume flux times a non-negative face value, and `source` adds a non-negative
rate. So `S_n <= S(t)` for every step n ending at or before t, and the leftover accumulated to
t is

    |A(t)| <= sum_n |leftover_n| <= (f L / T) sum_n S_n dt_n <= f L S(t) t / T.

For `t <= T` that is at most `f L S(t)`, a tenth of REQ-T05's limit at the configured share.
The share is spread over the run, not the step, so the bound does not depend on the step
count or the grid.

**Across a restart that extends `t_end`.** A run to `T1` restarted at `t1 <= T1` with a
later `t_end` `T2` has accumulated at most `f L S(t1) t1 / T1 <= f L S(t)` before the restart
and adds at most `f L S(t) (t - t1) / T2 <= f L S(t)` after it, so the bound at most doubles,
to `2 f L S(t)`, still a fifth of the limit. Extended again and again, `T1 < T2 < ... < Tm`,
the sum of `(T_k - T_(k-1)) / T_k` is at most `1 + ln(Tm / T1)`: the bound is
`f L S (1 + ln(Tm / T1))`, and the refusal stops the run before it can pass `L S`.

**A caller that marches past `T`.** The validation marches (VAL-024's scalar march among
them) step their own count, not to `solver.t_end`. Past `T` the bound grows as `t / T`; the
refusal still holds at `L S`.

**The share against the rounding floor.** If a step's share were below the rounding floor of
the leftover the stop forms, it could never be met. The computed residual of a cell carries
an error of a few units in the last place of the largest of its terms, `b_P`,
`sum_f G_f C_N` and `a_P C_P`, which near convergence are each about `a_P C_P`. Times `dt` and
summed, the floor is about `c u sum_P dt a_P C_P = c u (1 + kappa) Q`, with `u = 2^-53`, `c`
of order ten for the residual's four operations and the pairwise sum, `Q` the class's
content, and `kappa` the diffusion part of the diagonal over `V / dt`, `dt sum_f G_f / V`.
The share over that floor is

    f L S dt / (T c u (1 + kappa) Q) >= f L / (n c u (1 + kappa)),

`n = T / dt` the steps of the run, since `Q <= S` (the content is what was supplied less what
left, deposited or was left over, and the leftover is bounded above). At `f L = 1e-5` and
`c = 10` this is `9e9 / (n (1 + kappa))`. The 80x30 runs take at most 13,620 steps and
`kappa` is below 0.1 there; a 640x240 run of 120 s takes about 55,000 steps at Courant 0.4,
with `kappa` about 0.6 at the room's largest `nu_t`. The margin is above 1e5 on both, and the
share cannot fall to the floor at the configured values below about 1e9 steps.

## Alternatives (the mechanism report, section 7)

1. **Tighten `diffusion_tol`.** Each decade costs under one sweep per solve on 80x30 and
   divides the leftover by about ten, but it bounds the largest cell residual, not the sum.
   No tolerance reaches zero, and none guarantees anything on a finer grid or a longer run:
   the leftover's growth with the step count is the finding. Not taken.
2. **Stop on the summed residual.** A second stop on `dt |sum_P r_P|` beside the max-norm one.
   This is part 2, with the share set against the run's supply and length so that it sums to
   a bound rather than against each step's content, which would not.
3. **A conservative last update.** `C = C_K + dt r_K / V` after the last sweep closes the
   budget exactly at no cost, but it gives up the non-negative iterate: `r_K` can be
   negative in a cell near zero, so a concentration spanning many decades down to zero could
   go below it by about `diffusion_tol` times the field's largest value. Positivity is
   REQ-T12's and the alert thresholds read the low decades. Not taken.
4. **Conjugate gradients for transport.** It reaches the same max-norm residual in about the
   square root of Jacobi's sweeps, but a truncated CG iterate is not conservative either and
   has no sign guarantee, and its fallback to Jacobi brings the mechanism back. It moves the
   leftover; with 2 beside it, it could remove it. Not taken alone, and not needed with 2.
5. **Book the leftover.** Part 1. Alone it reports the non-conservation instead of bounding
   it, so a run outside REQ-T05 would close its budget and still be outside.
6. **Leave it.** Inside REQ-T05 on 320x120 at 1e-8 by the Phase 4 gate's margin, but the
   margin shrinks on the dust release with every refinement and is smaller at the worst frame
   than at the end. Not taken.

## Consequences

- Every transport field changes on a step whose share binds; where the max-norm stop alone
  already leaves the leftover within its share, the step is unchanged bit for bit. The
  k-epsilon solves are unchanged bit for bit, and so is every flow state and every flow
  harness row whose code path this change does not reach.
- The Phase 4 and Phase 5 records on the product grid were made before the change. Prompt 79
  reruns the product scenarios and the placement on the changed solve; the report of this
  pass names the run files and results that rerun supersedes.
- The cost is one sum over the cells per sweep, formed only after the max-norm test holds,
  and the extra sweeps where the share binds.
- A restart from a run file written before this change is refused, and the run has to be made
  again from the start.
- **Phase 6.** REQ-N03 asks the CUDA port to reproduce the NumPy reference within 1e-10. The
  stop now depends on the sign and size of a sum, not only on a maximum, so a port that forms
  the leftover in another order can stop on another sweep and differ by a whole sweep's
  change. The port has to reproduce this stop exactly: the max-norm test, then the leftover
  formed from the same residual, scalar `dt` times the sum, and the share
  `f * L * S * dt / T` evaluated in that order.
