# The Conservation Budget of the Implicit Transport Solve (REQ-T05)

**Date:** 2026-11-12
**Prompt:** 78 (`docs/prompts/cc-prompt-78-conservation-budget.md`)
**Branch:** `feature/conservation-budget`
**Status:** built and measured on 80x30. ADR-016 is Proposed. The product grid's runs are
prompt 79's (section 7).

Prompt 77 found that the particle budget's closure residual is the implicit diffusion
solve's leftover, `dt sum_P r_P`, which the max-norm stop leaves in the field and the budget
booked nowhere (`docs/reports/budget_residual_mechanism.md`). Alex decided on 2026-11-12 that
the accumulated leftover shall stay below REQ-T05's limit by construction. This pass builds
that decision (ADR-016): the leftover is booked as its own budget term, each step's solve
stops only when its leftover is within its share of the limit over the run, and a run whose
accumulated leftover reaches the limit is refused.

## The finding

The decision is built as stated, and on 80x30 it costs little. With the leftover booked
every run closes within rounding of its supply at every stored frame and every step (table
A1). Every step's leftover is within its share, and the accumulated leftover stays a small
fraction of `f L S` on every run and class (table B1), so the refusal never comes near
firing. The share binds on the dust release and the door leak at most release steps and
adds a quarter of a sweep per solve at most (table K1); the march's wall time moves within
the noise of two identical runs (table W1). The k-epsilon solves, every flow state and every
flow harness row retaken reproduce bit for bit, and no detection latency moved (tables F1,
H1, L1).

Two things did not go as predicted. Prediction (b)'s second clause fails: where the share
never binds, on the clean-out, the run is main's bit for bit (table M1) and its leftover is
prompt 77's T5, equal rather than smaller; on the filter breach the share bound 30 solves,
the field moved, and the leftover the run then accumulated ends slightly larger than prompt
77's. The share bounds the leftover against the supply; it does not promise to shrink a
leftover that was already inside it. And the seal leak carries nothing on 80x30: its opening
is 0.025 m, a quarter of a cell, so no face carries the leak and every budget stays at zero
(section 8).

## The predictions

Set by the orchestrator in prompt 78 on 2026-11-12, before any run of this pass, and
committed alone in 6b34a13 before the first run.

- (a) With the leftover booked, every 80x30 run's budget closes within 1e-12 of its supplied
  particles at every stored frame, every class.
- (b) The accumulated leftover stays within `f * L * S(t)` (1e-5 of the supply) at every step of
  every run, and ends smaller in magnitude than prompt 77's T5 for the same run and class.
- (c) Mean sweeps per solve rise by less than one, and march wall time by less than 15%, on every
  80x30 run against prompt 77's.
- (d) Every flow harness row and every flow state reproduces bitwise.
- (e) Every detection latency on 80x30 is within one time step of the run before this change.

The prompt's stop conditions: a flow row or flow state changes; the share cannot be met at
the configured values; a planted defect does not fail its test; the estimate exceeds the
4-hour budget. A failed prediction other than (d) is a finding, not a stop.

<!-- tables78 predictions begin -->
### P1. Prompt 78's predictions against the measurements

| Prediction | As stated (abridged) | Measured | Outcome |
|---|---|---|---|
| (a) | every 80x30 run, stored frame and class: abs(residual) within 1e-12 of the supply | 5 of 5 runs within it; table A1 | holds |
| (b) | accumulated leftover within f L S(t) at every step; ends smaller than prompt 77's T5 | largest abs(accumulated) over f L S(t) 0.0276; smaller at the end in 10 of 20 run-classes; table B1 | first clause holds; second FAILS in: R2, R4 |
| (c) | mean sweeps per solve rise by less than one, march wall by less than 15%, against prompt 77 | sweeps under one in 4 of 4 runs, wall under 15% in 4 of 4; tables K1, W1 | holds |
| (d) | every flow harness row and flow state reproduces bitwise | 19 of 19 states and rows bitwise; tables F1, H1, H2 | holds |
| (e) | every 80x30 detection latency within one time step of the run before | 5 of 5 runs within one step; tables L1, L2 | holds |
<!-- tables78 predictions end -->

(a), (c), (d) and (e) hold. (b)'s first clause holds with a wide margin; its second fails
on the clean-out and the filter breach, a finding, not a stop (the prompt's rule). Read
against what the decision promises, the clause is not one of its guarantees: the share keeps
each step's leftover below `f L S_n dt_n / T`, and on a run whose leftover is already below
that at every step, as the clean-out's is, nothing changes. The filter breach's extra sweeps
on 30 solves move its field, and the leftover of the steps after them is the leftover of
another field; that it ends 0.4% larger is the same kind of difference as any other
perturbation of the march, inside the bound.

(c) is scored against prompt 77, as stated. Prompt 77's walls carried its recorder, which
wrapped every step, so table W1 also gives main's walls from runs made side by side with the
branch's in this pass; against main every run is inside 15% as well. The clean-out is the
same arithmetic on both sides (table M1), and its two walls differ by 8%: that is the
measurement's noise under five processes.

(d) holds on what was compared. The flow rows' accuracy records differ from their rows of
record where the scoring or the configuration changed after the record was taken
(`val018_320x120`'s cap, raised from 10,000 to 15,000 after its row; `val020_272x90`'s
scoring, which gained the ring stations in 75cee14). The computed parts are bitwise in every
row, and `val020_272x90` retaken on origin/main gives the branch's row key for key (table
H1's last column).

## 1. What was built, against the decision

**Part 1, book it.** `scalar_scheme.implicit_step` returns the leftover of the field it
returns in `ImplicitResult.leftover`: `dt` times the sum over the cells of the residual
`b + sum_f G_f C_N - a_P C`, zero in SOLID and held cells (with a per-cell `dt`, the sum of
`dt r`). It is formed from the residual the stop's last test read, once and only once.
`MassBudget` gains `leftover`, which `solve_timestep` adds to every step, and the residual is
`initial + inflow + source - outflow - deposited - leftover - current`. `relative()` keeps
its meaning, the residual over `supplied()` (`initial + inflow + source`, now a method of its
own), and `leftover_relative()` is the leftover over the supply. The run file's budget rows
gain `leftover` as their last column (`io_manager.BUDGET_TERMS`), and rows are read by the
terms the file names, so a file written before this pass reads with the leftover None: the
residual is then the one it always had, the leftover in it, and `leftover_relative()`
refuses. A restart from such a file is refused by `time_integration.run` before the
configuration's check and before anything is written, with the reason (the leftover before
the restart is unknown), and `TransportSolver.restore_budgets` refuses a budget whose leftover
is unknown. `scripts/run_scenario.py`'s summary prints each class's leftover and its ratio
beside the residual.

**Part 2, budget it per step.** `implicit_step` takes `leftover_bound`: the stop needs the
max-norm test and `abs(leftover) <= leftover_bound`. Under Jacobi the leftover is formed only
on a sweep whose max-norm test holds; under CG only on a true residual that holds, and a CG
iterate that fails the bound restarts from it as one that fails the true-residual test does.
`solve_timestep` hands it `leftover_share * conservation_limit * S_n * dt / t_end`, evaluated
in that order, `S_n` the supply with the step's inflow and source booked and `t_end`
`config.t_end`. The k-epsilon solves pass no bound, and without it the stop is the max-norm
one, bit for bit: the absent argument skips the clause, and the leftover of the returned field
is formed after the loop from the residual the last test read, which no field depends on.

**Part 3, refuse a breach.** After the step, `solve_timestep` raises `ConservationBreachError`
(a RuntimeError naming the class, the leftover, the supply and the limit) when the
accumulated leftover is not zero and its magnitude is at least `conservation_limit` times the
supply. The budget holds the step and the field is not returned. `scripts/run_scenario.py`
catches it with the flow-state errors and exits 1; the run file and checkpoint file hold what
the last writes left.

**Configuration.** `transport.conservation_limit` (1e-4, in (0, 1)) and
`transport.leftover_share` (0.1, in (0, 1]) are required keys of the transport section,
validated at load (a bool or a string refused as a type, NaN, infinity and the ends out of
range refused as values), in the base file and its four copies (`clean_room_l1_return3_closed`,
`clean_room_l2_left_supply`, `clean_room_vents`, `validation_line_source`). The validation
cases that build a transport section by hand read both from the base file
(`validation.transport_cases.conservation_keys`).

**REQ-T05's 1e-4 written as a literal.** Pointed at the key: VAL-007's criterion
(`tests/test_conservation.py`), `tests/test_time_integration.py`'s `REQ_T05`, and ADR-011 G's
rule `mass_imbalance_tol <= 1e-4 rho V_min / T` wherever a test forms it
(`tests/helpers.py`, `tests/test_alert_latency.py`, `tests/test_time_integration.py`,
`tests/test_benchmark.py`, `tests/test_config.py`); the YAML comments that state the rule
name the key. Not pointed: the YAML values of `solver.mass_imbalance_tol` themselves, which
are numbers the rule gave (YAML cannot refer to another key, and the tolerance is a flow input,
so deriving it at load would be a change to every flow state's inputs); ADR-011 G's own text
and the earlier probes' scripts (`docs/reports/probe77/configs77.py`), which are records of
what was run; and the many `1e-4` that are other quantities (`pressure_rtol`, the cylinder's
`step_tol`, harness allowances).

## 2. The bound, as built

ADR-016 has the derivation; it holds for the code as built because the share is formed from
the supply after the step's inflow and source are booked, and the supply never falls (the
inflow adds an inward flux times a non-negative face value and the source a non-negative
rate). Within one run the accumulated leftover at `t <= T` is at most `f L S(t) t / T`, below
`f L S(t)`. A restart that extends `t_end` from `T1` to `T2` at most doubles it, to
`2 f L S(t)`; extended again and again to `Tm`, it is at most `f L S(t) (1 + ln(Tm / T1))`,
and the refusal stops the run before it passes `L S`. A caller that marches past `t_end`,
as VAL-024's scalar march does (it stops at its steady state, its configuration's `t_end` is
the flow solve's), keeps the refusal and loses the `t / T` factor's guarantee beyond `T`.

**The share against the rounding floor.** The prompt asked for this to be shown from the
arithmetic or the pass to stop. The share over the leftover's rounding floor is at least
`f L / (n c u (1 + kappa))`, `n` the run's steps (ADR-016). Table R1 gives the measured
smallest share over `u Q` on every 80x30 run and the arithmetic's figure on 320x120 and,
extrapolated, on 640x240. The margin is many orders above one on every row: the share cannot
fall to the floor at the configured values, and the stop condition did not fire.

<!-- tables78 floor begin -->
### R1. The share against the leftover's rounding floor

The floor of a step's leftover is about c u (1 + kappa) Q (ADR-016), u the unit roundoff, Q the content and kappa the largest dt sum_f G_f / V. The measured column is the smallest share over u Q at any scored step of the run; dividing by c (1 + kappa), c about ten, gives the margin.

| Run or grid | Steps | Largest kappa | Smallest share over u Q, measured | Margin at c = 10 |
|---|---|---|---|---|
| R1 (dust release, 80x30) | 6,811 | 0.1133 | 1.322e+07 | 1.188e+06 |
| R2 (clean-out, 80x30) | 13,620 | 0.1133 | 6.634e+06 | 5.959e+05 |
| R3 (door leak, 80x30) | 3,416 | 0.1133 | 2.553e+07 | 2.293e+06 |
| R4 (filter breach, 80x30) | 6,811 | 0.1133 | 1.323e+07 | 1.188e+06 |
| SL (seal leak, 80x30) | 6,811 | 0.1133 | nothing supplied | nothing supplied |
| 320x120 dust release (results/builder69, prompt 77's R10 steps) | 33,119 | 0.4199 | 2.720e+06 (f L / (n u), Q = S) | 1.915e+05 |
| 640x240, extrapolated (twice the steps and twice kappa of 320x120) | 66,238 | 0.8397 | 1.360e+06 (f L / (n u), Q = S) | 7.391e+04 |
<!-- tables78 floor end -->

## 3. What was run

**The runs.** `docs/reports/probe78/configs78.py` defines five scenarios on 80x30: R1 to R4,
prompt 77's (the dust release, the clean-out, the door leak and the filter breach at the
files' inputs, each on prompt 64's flow state with the `mass_imbalance_tol` that run
recorded), and SL, the seal leak, whose closed-door state is the door leak's and whose
leaking state each side solves, warm from it. Each was made three times by `probe78.py`:
`main` on origin/main's code in a worktree, the run before this change; `branch` on this
branch's code; `probe` on this branch's code with the recorder, which wraps
`TransportSolver.solve_timestep` and `implicit_step` and records every step and every solve,
and solves each system a second time without the bound to count the sweeps the share added.
Each run's main and branch jobs ran side by side in the queue (`run78.py`), five processes at
most, one BLAS thread each.

**The calibration.** The recorder is pure: the probe run's frames, budgets, sensor readings
and arrival maps equal the branch run's bit for bit at every stored frame (table C1).

<!-- tables78 calibration begin -->
### C1. The recorder's calibration: the probe run against the branch run, every stored frame

| Run | Frames | Fields bitwise at every frame | Budgets bitwise at every frame | Sensor readings bitwise | Arrival maps bitwise |
|---|---|---|---|---|---|
| R1 (dust release) | 70 | yes | yes | yes | yes |
| R2 (clean-out) | 138 | yes | yes | yes | yes |
| R3 (door leak) | 36 | yes | yes | yes | yes |
| R4 (filter breach) | 70 | yes | yes | yes | yes |
| SL (seal leak) | 70 | yes | yes | yes | yes |
<!-- tables78 calibration end -->

Where the share never binds the branch's run is main's, bit for bit (table M1).

<!-- tables78 branch_main begin -->
### M1. The branch run against main's, every stored frame

The budgets compared without the leftover, which main's files do not hold.

| Run | Frames | Frames whose fields are bitwise main's | First frame that differs (step) | Budgets bitwise at every frame |
|---|---|---|---|---|
| R1 (dust release) | 70 | 6 | 600 | NO |
| R2 (clean-out) | 138 | 138 | none | yes |
| R3 (door leak) | 36 | 6 | 600 | NO |
| R4 (filter breach) | 70 | 7 | 700 | NO |
| SL (seal leak) | 70 | 70 | none | yes |
<!-- tables78 branch_main end -->

**The harness.** The rows were retaken at f09d595, whose code is 05c9aff's (`val020_272x90`
started before the commit on the same `src/`, `validation/` and `scripts/benchmark.py`
bytes, `results/builder78/src_hashes_at_val020_start.txt`): VAL-024's two rows, whose scalar
march runs the transport solver, into `benchmarks/results.jsonl`, and the flow rows, retaken
only to see whether they reproduce, into `results/builder78/harness_flow.jsonl`.

Retaken: every turbulent flow row (`val017_216x72`, `val018_320x120`, `val019_656x60`,
`val020_272x90`), whose k and eps solves go through the changed `implicit_step`, and the five
laminar rows of VAL-001 and VAL-002. Not retaken: the four VAL-022 cylinder rows, laminar
transient marches of two to three hours each, which run no line this pass changed (no
k-epsilon model and no transport solver).

**Wall clock.** The runs started at 11:50 with `val020_272x90`, the longest (8,328 s in its row
of record under six processes), and the estimate from it was about 2.5 hours of the 4-hour
budget. The last job, `val020_272x90` retaken on origin/main to read its accuracy
differences, ended at 13:11: 1 hour 21 minutes of wall clock in all, five processes at most.

## 4. The tests and the planted defects

New: `tests/test_conservation_budget.py` (20 tests) and
`tests/test_time_integration.py::TestConservationBudget` (5 tests, on the 80x30 kept flow
state the module already solves); strengthened: `tests/test_config.py` (the two keys' types,
ranges, presence and values of record), `tests/test_solver_transport.py`'s sweep-cap test (the
two-sweep cap now also trips the refusal). The prompt's list:

- *An 80x30 window spanning a release closes to rounding at every step, every class:*
  `TestConservationBudget::test_the_window_closes_to_rounding_at_every_step_and_class`, the
  dust release rescheduled to 1 to 3 s of a 4 s run at `diffusion_tol` 1e-5, every step
  recorded. Plant P1.
- *Every step's leftover is within its share, and the accumulated leftover within the bound:*
  `test_every_steps_leftover_is_within_its_share_and_the_bound_holds` on the same window, and
  `TestSolverBudget::test_each_step_s_leftover_is_within_its_share` on a closed box at two
  `t_end`. Plants P2, P12, P13. `test_the_share_bites_in_the_window` and
  `test_the_share_made_the_solve_take_more_sweeps` show the share binds in both cases, so the
  two can fail.
- *`implicit_step` without the new argument equals an infinite share, bit for bit, for
  transport and for k and epsilon:* `TestWithoutTheBound`, every system the transport solver
  hands the solve over ten steps, and every k and eps system of 24 k-epsilon steps from the
  patchy start, through the onset of the capped CG solves and their Jacobi fallbacks. Plant P4.
- *The refusal fires when a planted case's accumulated leftover reaches the limit, and the
  runner exits 1 with the checkpoint left:* a one-sweep cap,
  `TestSolverBudget::test_the_refusal_fires_when_the_accumulated_leftover_reaches_the_limit` and
  `TestConservationBudget::test_a_breach_exits_1_and_leaves_the_checkpoint`. Plant P3. A large
  share cannot plant it: `f` is at most 1, and at 1 the bound is `L S t / T`, below `L S`.
- *An older run file reads, and its restart is refused:*
  `test_an_older_run_file_reads_and_its_restart_is_refused`, the window's run and checkpoint
  files rewritten as a writer before this pass wrote them. Plants P5, P6; P7 for
  `restore_budgets`.
- *Configuration:* `TestTransportSection`'s bad values, missing keys and edges, and every
  configuration of record carrying 1e-4 and 0.1. Plants P16 to P18.

The other plants (P8 to P11, P14, P15) are of the leftover's arithmetic and the budget's
ratios. Each was applied to a copy of the tree by `results/builder78/plant78.py`, which runs
the tests that claim it there; the unchanged copy passes every one of them (table P2).
 P11 survived the first plant run: the test's tolerance never let the max-norm part
hold within its five sweeps, so the bound's clause was not what kept the solve going. The
test now uses a tolerance the max-norm part meets from the first iterate, checks that the
unbounded solve takes no sweep, and catches the plant. P2 is caught twice: the unit test
fails, and on the 80x30 window the max-norm stop alone at the window's tolerance takes the
accumulated leftover past the limit, so the window's run raises the breach and its fixture
errors.

<!-- tables78 plants begin -->
### P2. The planted defects (results/builder78/plant78.py)

Each applied to a copy of the tree at the code commit and the tests that claim it run there; `none` runs every one of those tests on the unchanged copy.

| Plant | File | Tests run | Outcome | Caught |
|---|---|---|---|---|
| none | - | 19 | 52 passed in 65.19s (0:01:05) | baseline |
| P1_booking_removed | src/solver_transport.py | 2 | 2 failed in 58.44s | yes |
| P2_share_ignored_in_the_stop | src/solver_transport.py | 2 | 1 failed, 1 error in 56.51s | yes |
| P3_refusal_check_removed | src/solver_transport.py | 2 | 2 failed in 59.04s | yes |
| P4_absent_argument_takes_a_bound | src/scalar_scheme.py | 2 | 2 failed in 4.71s | yes |
| P5_reader_zips_current_terms | src/io_manager.py | 1 | 1 failed in 58.71s | yes |
| P6_restart_check_removed | src/time_integration.py | 1 | 1 failed in 59.98s | yes |
| P7_restore_check_removed | src/solver_transport.py | 1 | 1 failed in 1.63s | yes |
| P8_leftover_sign_flipped | src/scalar_scheme.py | 2 | 2 failed in 2.29s | yes |
| P9_step_applied_twice | src/scalar_scheme.py | 1 | 2 failed in 1.70s | yes |
| P10_bound_ignored_by_the_clause | src/scalar_scheme.py | 1 | 2 failed in 1.70s | yes |
| P11_capped_reported_converged | src/scalar_scheme.py | 1 | 1 failed in 1.72s | yes |
| P12_horizon_fixed_at_one_second | src/solver_transport.py | 1 | 1 failed in 2.26s | yes |
| P13_share_from_the_content | src/solver_transport.py | 1 | 1 failed in 1.58s | yes |
| P14_leftover_added_in_the_residual | src/solver_transport.py | 1 | 1 failed in 1.72s | yes |
| P15_unknown_leftover_read_as_zero | src/solver_transport.py | 1 | 1 failed in 1.75s | yes |
| P16_limit_range_open_at_one | src/config.py | 1 | 1 failed, 27 passed in 2.67s | yes |
| P17_share_range_closed_at_one | src/config.py | 1 | 1 failed in 1.90s | yes |
| P18_share_key_optional_at_one | src/config.py | 1 | 1 failed, 4 passed in 1.62s | yes |
<!-- tables78 plants end -->

**The suite.** At f09d595, the code commit's code, all three tiers: 3,043 tests passed, none failed or skipped (2,642 unit, 321 integration, 80 validation; `results/builder78/suite_final.txt` and `suite_final.xml`). `ruff format --check` and `ruff check` are clean, and `scripts/gen_system_map.py --check` agrees at both commits.

## 5. The measurements

### Closure and the bound

<!-- tables78 closure begin -->
### A1. Closure with the leftover booked: the largest |residual| over the supply (prediction (a))

| Run | Stored frames scored | Largest at a stored frame, over the classes | Steps scored (probe) | Largest at any step | Within 1e-12 at every frame and step |
|---|---|---|---|---|---|
| R1 (dust release) | 320 | 1.400e-14 | 31,215 | 1.417e-14 | yes |
| R2 (clean-out) | 685 | 6.355e-15 | 68,100 | 6.762e-15 | yes |
| R3 (door leak) | 150 | 5.314e-15 | 14,240 | 5.401e-15 | yes |
| R4 (filter breach) | 320 | 5.261e-14 | 31,215 | 5.307e-14 | yes |
| SL (seal leak) | 0 | 0.000e+00 | 0 | 0.000e+00 | yes |
<!-- tables78 closure end -->

<!-- tables78 bound begin -->
### B1. The accumulated leftover against its bound, every step (prediction (b))

f L = 1e-05. The step's share is f L S_n dt_n / T and the bound f L S(t) t / T, below f L S(t) for t <= T.

| Run | Class (um) | Largest abs(step leftover) over its share | Largest abs(accumulated) over f L S(t) t / T | Largest abs(accumulated) over f L S(t) | Over L S(t) | Accumulated at the end | Before this change at the end | Read from | Smaller in magnitude |
|---|---|---|---|---|---|---|---|---|---|
| R1 | 0.1 | 0.9917 | 0.0541 | 0.0215 | 0.0022 | -6.4551e-02 | -5.7257e-01 | prompt 77's T5 | yes |
| R1 | 0.3 | 0.9917 | 0.0541 | 0.0215 | 0.0022 | -6.4548e-02 | -5.7254e-01 | prompt 77's T5 | yes |
| R1 | 0.5 | 0.9919 | 0.0541 | 0.0215 | 0.0022 | -6.4542e-02 | -5.7249e-01 | prompt 77's T5 | yes |
| R1 | 1 | 0.9924 | 0.0540 | 0.0215 | 0.0022 | -6.4516e-02 | -5.7232e-01 | prompt 77's T5 | yes |
| R1 | 5 | 0.9804 | 0.0529 | 0.0212 | 0.0021 | -6.3698e-02 | -5.6751e-01 | prompt 77's T5 | yes |
| R2 | 0.1 | 0.1748 | 0.0975 | 0.0276 | 0.0028 | -4.5772e+01 | -4.5772e+01 | prompt 77's T5 | equal |
| R2 | 0.3 | 0.1748 | 0.0973 | 0.0276 | 0.0028 | -4.6715e+00 | -4.6715e+00 | prompt 77's T5 | equal |
| R2 | 0.5 | 0.1748 | 0.0975 | 0.0276 | 0.0028 | -1.6129e+00 | -1.6129e+00 | prompt 77's T5 | equal |
| R2 | 1 | 0.1748 | 0.0985 | 0.0276 | 0.0028 | -3.8211e-01 | -3.8211e-01 | prompt 77's T5 | equal |
| R2 | 5 | 0.1750 | 0.0986 | 0.0274 | 0.0027 | -1.3594e-02 | -1.3594e-02 | prompt 77's T5 | equal |
| R3 | 0.1 | 0.9988 | 0.0879 | 0.0237 | 0.0024 | 3.2596e+00 | 4.7542e+00 | prompt 77's T5 | yes |
| R3 | 0.3 | 0.9987 | 0.0879 | 0.0237 | 0.0024 | 3.3245e-01 | 4.8490e-01 | prompt 77's T5 | yes |
| R3 | 0.5 | 0.9986 | 0.0879 | 0.0237 | 0.0024 | 1.1471e-01 | 1.6731e-01 | prompt 77's T5 | yes |
| R3 | 1 | 0.9980 | 0.0878 | 0.0237 | 0.0024 | 2.7094e-02 | 3.9521e-02 | prompt 77's T5 | yes |
| R3 | 5 | 0.9989 | 0.0884 | 0.0237 | 0.0024 | 9.5557e-04 | 1.3650e-03 | prompt 77's T5 | yes |
| R4 | 0.1 | 0.9584 | 0.0484 | 0.0091 | 0.0009 | -1.4688e+01 | -1.4626e+01 | prompt 77's T5 | NO |
| R4 | 0.3 | 0.9585 | 0.0484 | 0.0091 | 0.0009 | -1.4982e+00 | -1.4919e+00 | prompt 77's T5 | NO |
| R4 | 0.5 | 0.9586 | 0.0484 | 0.0091 | 0.0009 | -5.1703e-01 | -5.1485e-01 | prompt 77's T5 | NO |
| R4 | 1 | 0.9590 | 0.0484 | 0.0091 | 0.0009 | -1.2222e-01 | -1.2170e-01 | prompt 77's T5 | NO |
| R4 | 5 | 0.9711 | 0.0487 | 0.0091 | 0.0009 | -4.3138e-03 | -4.2954e-03 | prompt 77's T5 | NO |
| SL | 0.1 | nothing supplied | - | - | - | 0.0000e+00 | - | - | - |
| SL | 0.3 | nothing supplied | - | - | - | 0.0000e+00 | - | - | - |
| SL | 0.5 | nothing supplied | - | - | - | 0.0000e+00 | - | - | - |
| SL | 1 | nothing supplied | - | - | - | 0.0000e+00 | - | - | - |
| SL | 5 | nothing supplied | - | - | - | 0.0000e+00 | - | - | - |
<!-- tables78 bound end -->

The largest step leftover over its share is near one on the runs where the share binds:
the solve stops at the first sweep whose leftover is inside it. The accumulated leftover is
far below its bound, because the leftover changes sign along a run and its steps cancel, as
prompt 77 found. The clean-out's step leftovers never come within a fifth of the share. The
seal leak supplies nothing on 80x30 and is not scored.

### The cost

<!-- tables78 sweeps begin -->
### K1. Sweeps per solve: the branch, the max-norm stop alone on the same systems, and prompt 77 (prediction (c))

| Run | Steps | Solves | Mean sweeps, branch | Mean, the max-norm stop alone | Mean, prompt 77 | Rise over prompt 77 | Under one | Solves the share made sweep on | Steps with one such solve or more | First such step, t (s) | Last, t (s) | Largest extra sweeps | Solves at the cap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 (dust release) | 6,811 | 34,055 | 4.54 | 4.29 | 4.29 | +0.25 | yes | 8,543 | 1,711 | 10.11 | 40.69 | 1 | 0 |
| R2 (clean-out) | 13,620 | 68,100 | 2.65 | 2.65 | 2.65 | +0.00 | yes | 0 | 0 | - | - | 0 | 0 |
| R3 (door leak) | 3,416 | 17,080 | 2.46 | 2.42 | 2.42 | +0.04 | yes | 696 | 140 | 10.02 | 12.38 | 1 | 0 |
| R4 (filter breach) | 6,811 | 34,055 | 4.52 | 4.52 | 4.52 | +0.00 | yes | 30 | 6 | 11.97 | 12.94 | 1 | 0 |
| SL (seal leak) | 6,811 | 34,055 | 0.00 | 0.00 | - | - | - | 0 | 0 | - | - | 0 | 0 |
<!-- tables78 sweeps end -->

<!-- tables78 wall begin -->
### W1. March wall time: the branch, main, and prompt 77 (prediction (c))

The march's wall time is the run's less the flow solves it made itself (SL solves its leaking state on both sides). The branch and main runs of a scenario ran side by side in the same queue. Prompt 77's runs carried its recorder, which wrapped every step.

| Run | Branch (s) | Main (s) | Prompt 77 (s) | Branch over main | Branch over prompt 77 | Rise under 15% against prompt 77 | Against main |
|---|---|---|---|---|---|---|---|
| R1 (dust release) | 90 | 94 | 81 | 0.957 | 1.110 | yes | yes |
| R2 (clean-out) | 183 | 169 | 210 | 1.083 | 0.872 | yes | yes |
| R3 (door leak) | 46 | 46 | 52 | 1.000 | 0.883 | yes | yes |
| R4 (filter breach) | 97 | 95 | 109 | 1.026 | 0.892 | yes | yes |
| SL (seal leak) | 86 | 83 | - | 1.036 | - | - | yes |
<!-- tables78 wall end -->

The extra sweeps are confined to the events: every step with a binding solve lies inside
the event's interval or just after its end (K1's first and last columns; each event starts at
10 s, the dust release's lasts 30 s), and table M1 dates each run's first divergence from main
to its event's onset. No solve
reached the 200-sweep cap. The wall times are within the noise table M1's clean-out sets.

### The latencies

<!-- tables78 latency begin -->
### L1. Detection latencies on 80x30 against the run before this change (prediction (e))

The arrival map's latency at each sensor's cell, which the monitor's equals to the step (ADR-015 D); every event, class and sensor. One step is the run's largest step.

| Run | Before | Entries | Reached in both | Never in both | Equal | Differ | Largest difference (s) | In steps | Within one step |
|---|---|---|---|---|---|---|---|---|---|
| R1 (dust release) | results/builder77/R1.h5 | 20 | 0 | 20 | 20 | 0 | 0.000000 | 0.00 | yes |
| R2 (clean-out) | results/builder77/R2.h5 | 20 | 20 | 0 | 20 | 0 | 0.000000 | 0.00 | yes |
| R3 (door leak) | results/builder77/R3.h5 | 20 | 0 | 20 | 20 | 0 | 0.000000 | 0.00 | yes |
| R4 (filter breach) | results/builder77/R4.h5 | 20 | 0 | 20 | 20 | 0 | 0.000000 | 0.00 | yes |
| SL (seal leak) | results/builder78/main/SL.h5 | 20 | 0 | 20 | 20 | 0 | 0.000000 | 0.00 | yes |
<!-- tables78 latency end -->

<!-- tables78 latency_changes begin -->
### L2. The latencies that differ

| Run | Event | Class (um) | Sensor | Before (s) | After (s) | Difference (s) |
|---|---|---|---|---|---|---|
| none | | | | | | |
<!-- tables78 latency_changes end -->

Every latency is the run before's exactly. On 80x30 most entries are never reached by
either run: on this grid the plumes do not reach the mounted sensors at their thresholds
within the runs. The clean-out's are reached, and are equal.

### The flow

<!-- tables78 flow_states begin -->
### F1. The flow states (prediction (d))

| Run | State | How the branch run came by it | Against | u, v, p, k, eps, nu_t bitwise | Outer iterations |
|---|---|---|---|---|---|
| R1 | 22ac4cb0815d | kept | results/builder64/coarse/equipment_dust.h5 | yes | 1021 against 1021 |
| R2 | 22ac4cb0815d | kept | results/builder64/coarse/clean_out.h5 | yes | 1021 against 1021 |
| R3 | 22ac4cb0815d | kept | results/builder64/coarse/door_leak.h5 | yes | 1021 against 1021 |
| R3 | eddc5d9212f0 | kept | results/builder64/coarse/door_leak.h5 | yes | 928 against 928 |
| R4 | 22ac4cb0815d | kept | results/builder64/coarse/filter_breach.h5 | yes | 1021 against 1021 |
| SL | 22ac4cb0815d | kept | results/builder64/coarse/door_leak.h5 | yes | 1021 against 1021 |
| SL | 71675510613a | solved | main's solve | yes | 100 against 100 |
<!-- tables78 flow_states end -->

<!-- tables78 harness_flow begin -->
### H1. The flow harness rows retaken (prediction (d))

Each against the last row of record of the case under the method of record. Bitwise means the row's work and outcome, and its trajectory at every outer iteration both rows sampled (residuals, outer counts, cell updates; the wall clock and the accuracy copy in the last sample left out), are equal as recorded, every float to the last bit. The accuracy record's keys that differ are named with both values. A row whose accuracy differs from its record is also retaken on origin/main (results/builder78/harness_main.jsonl) and compared whole, accuracy included.

| Case | Row of record (commit) | Outer, record | Outer, retaken | Samples compared | Wall (s), record | Retaken | Bitwise | Accuracy keys that differ from the record | Main retaken, every key equal to the branch's |
|---|---|---|---|---|---|---|---|---|---|
| val018_320x120 | a02edad4 (b9e8191) | 4,389 | 4,389 | 440 | 2144 | 870 | yes | components.max_simple_iter 10000 to 15000; value 0.4389 to 0.2926 | - |
| val017_216x72 | b1c3a436 (759b3ad) | 11,443 | 11,443 | 116 | 942 | 1261 | yes | none | - |
| val020_272x90 | fab5e311 (6b8e89d) | 20,000 | 20,000 | 2001 | 8328 | 2598 | yes | components.stagnation_k_ratio_isotropic_ring None to 1.2556424352965188; report.k_axis.errors_in_plane.mean 0.0013336619499060736 to 0.001550245875702779; report.k_axis.errors_in_plane.rms 0.0046306441393370685 to 0.005028797821069408; 16 more | yes |
| val002_40x40 | 860d2bf7 (759b3ad) | 3,435 | 3,435 | 36 | 41 | 43 | yes | none | - |
| val001_80x40 | b15fc106 (759b3ad) | 1,559 | 1,559 | 17 | 23 | 24 | yes | none | - |
| val001_80x40_stretched | 719dc2f7 (759b3ad) | 1,124 | 1,124 | 13 | 18 | 19 | yes | none | - |
| val002_20x20 | efca4ee9 (759b3ad) | 970 | 970 | 11 | 7 | 7 | yes | none | - |
| val002_80x80 | 475c39e0 (759b3ad) | 12,814 | 12,814 | 130 | 281 | 339 | yes | none | - |
| val019_656x60 | f7040827 (759b3ad) | 20,000 | 20,000 | 201 | 2852 | 3527 | yes | none | - |
<!-- tables78 harness_flow end -->

Every flow state the runs used reproduces: the kept ones are copied bitwise from their
sources, and the seal leak's leaking state, solved on both sides from the same warm start,
is bitwise main's. Every flow row retaken is bitwise in its computed parts (prediction (d)).

## 6. The harness rows changed

<!-- tables78 harness_transport begin -->
### H2. The harness rows whose transport output changes: VAL-024 retaken

The flow part (work, outcome, trajectory) is compared bitwise; the scalar march and the scored widths are the rows' old and new values, to the last digit a double holds.

| Case | Old row (commit) | New row (commit) | Flow part bitwise | Largest scored departure, old | New | Decay exponent, old | New | Width ratio 20 M, old | New | 50 M, old | New | March steps, old | New | March wall (s), old | New |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val024_280x50 | 05484863 (47e1e53) | a5f2be1a (f09d595) | yes | 0.14725021605264843 | 0.14725021605264854 | 1.0776093212578151 | 1.0776093212578151 | 1.0326787294310857 | 1.0326787294310864 | 0.85274978394735157 | 0.85274978394735146 | 3,063 | 3,063 | 29.2 | 29.9 |
| val024_560x100 | 74c818f8 (3153104) | 9adb1ae5 (f09d595) | yes | 0.14828623494384063 | 0.14828623494384063 | 1.0767968156661505 | 1.0767968156661505 | 1.0299031045541231 | 1.0299031045541238 | 0.85171376505615937 | 0.85171376505615937 | 5,256 | 5,256 | 116.9 | 118.3 |
<!-- tables78 harness_transport end -->

VAL-024's flow part reproduces bit for bit. Its scalar march, which runs the transport
solver, moves the widths in their last digits on both grids (table H2); on 560x100 the
largest scored departure, read at 50 M, is unchanged as recorded. The march's horizon is its configuration's `t_end`, the flow
solve's 60 s, about forty times the march, so its share is small (section 2). Both rows still
pass their criterion (ADR-015 A). The new rows are appended to
`benchmarks/results.jsonl` at 05c9aff's code (row commit f09d595, clean).

## 7. What prompt 79 must re-run on the product grid

Every transport output made before this change is superseded wherever a step's share binds,
which on 80x30 is most release steps (table K1). The records prompt 79 replaces:

- The Phase 4 gate's four reference scenarios on 320x120, `results/builder69/equipment_dust.h5`,
  `clean_out.h5`, `door_leak.h5` and `filter_breach.h5` (with their checkpoint files and
  summaries), and the tables of `docs/reports/phase4_scenarios_report.md` read from them: B1 to
  B3 (the budgets, now with the leftover booked), G1 and the scenario sections' numbers.
- The Phase 5 gate's five scenarios on 320x120, `results/builder76/equipment_dust.h5`,
  `clean_out.h5`, `door_leak.h5`, `filter_breach.h5` and `seal_leak.h5`, and what
  `docs/reports/phase5_alert_system_report.md` and `docs/reports/phase5_alert_report.md` read
  from them: the arrival maps, the alert tables, the latencies and the all-clear readings.
- The placement's inputs and results: prompt 75's 80x30 runs
  (`results/builder75/runs/dust.h5`, `door.h5`, `breach.h5`) and the placement tables the
  Phase 5 report builds on them and on `results/builder76/`.
- The sensitivity sweep of prompt 72 (`results/builder72/`, its 80x30, 160x60 and 320x120 runs)
  and `docs/reports/phase5_sensitivity.md`, whose latencies and plume readings are transport
  outputs.
- VAL-024's harness rows are retaken here (section 6) and need no rerun.

The flow states in all of these reproduce (prediction (d)), so prompt 79 can reuse them with
`--flows-from` and rerun the marches only. A restart of any of those runs is refused (section
1): each is made again from the start.

## 8. Deviations and limits

- **The seal leak on 80x30 carries nothing.** Its leaking segment is 0.025 m tall on a
  grid of 0.1 m cells, so it covers no face; the leaking state solves in 100 outer
  iterations to a field with no inflow of particles, and every class's supply is zero. The
  run is reported for its flow-state comparison and its latencies (all never on both sides).
  The prompt named it; it measures nothing of the budget here.
- **No 80x30 seal-leak run existed before this pass.** Its "before" for prediction (e) is
  this pass's main-side run (`results/builder78/main/SL.h5`), made on origin/main's code.
- **The 80x30 latencies' "before" for R1 to R4 is prompt 77's runs** (`results/builder77/R1.h5`
  to `R4.h5`, 2026-11-10), the latest at the files' inputs; their budgets are prompt 64's
  bitwise (prompt 77, table M1). Prompt 75's placement runs (`results/builder75/runs/`) are
  earlier.
- **`val018_320x120`'s retaken flow row records `git_dirty`**: the report's draft was a
  modified tracked file while it ran. The code it ran is 05c9aff's (its row's commit is
  f09d595, which changed only ADR-016 and SYSTEM.md). The draft was moved out of the tree
  before the VAL-024 rows of record finished, and they record clean.
- **`val020_272x90` started before the code commit**, on the same `src/`, `validation/` and
  `scripts/benchmark.py` bytes (`results/builder78/src_hashes_at_val020_start.txt`, checked
  at the commit).
- **The flow rows were compared on their computed parts.** A row's trajectory is compared at
  the outer iterations both rows sampled (`val017_216x72`'s row of record sampled every 100,
  this pass every 10) without the wall clock and the accuracy copy in its last sample.
- **ADR-016's rounding-floor figures were estimates when it was committed** (f09d595): kappa
  "below 0.1" on 80x30 and "about 0.6" on 640x240, 55,000 steps there. Table R1 measures 0.11
  and 0.42 (320x120), and the 640x240 extrapolation is 0.84 at 66,000 steps; the ADR is
  corrected in this report's commit. The margin stays above 7e4.
- **The ADR-011 G rule's YAML values and the earlier probes keep their literal 1e-4**
  (section 1).
- **Two of the prompt's tests are met by tests in two places**: the window tests need the
  kept 80x30 flow state the time-integration module solves once, so they sit there
  (`TestConservationBudget`); the rest are in `tests/test_conservation_budget.py`.

## Sources

1. `docs/reports/probe78/tables78.md`, generated by `tables78.py` from `results/builder78/`
   (untracked, this machine): per run and side `RUN.h5`, `RUN.txt` and `RUN.meta.json`, the
   probe's `RUN.steps.npz`, `harness_flow.jsonl`, `plants.json` and `runs78.json`; and
   `benchmarks/results.jsonl`. `--check` regenerates and compares, the report's copies too.
2. `configs78.py`, `probe78.py`, `run78.py` beside it; `results/builder78/plant78.py`.
3. Prompt 77's records, `results/builder77/` (R1 to R4: `RUN.steps.npz`, `RUN.probe.json`,
   `RUN.h5`), and the source runs `results/builder64/coarse/`.
4. ADR-016, `docs/reports/budget_residual_mechanism.md`, `docs/SYSTEM.md` (REQ-T05, section 4).
