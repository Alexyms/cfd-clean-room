# Phase 6 Step 5b: The Stopping Rule and the Outer Iteration (prompt 95)

**Date:** 2026-11-25; round 1 (prompt 95b) 2026-11-26
**Prompt:** 95 (`docs/prompts/cc-prompt-95-outer-iteration.md`), round 1 `cc-prompt-95b-round1-fixes.md`
**Branch:** `phase6/flow`
**Status:** built. All five predictions hold (T0), and the step's gate is met: every whole solve
stops for the reference's reason at the reference's outer count, not one apart, every flow field
inside its measured spread (T3). The first run of the planted defects left five alive, each a rule no
test read; a test commit catches them (section 4). Round 1 (section 6) answers review 95 and test 95:
the stop is pinned where continuity binds it, the outlets inside a solve, the failure paths and the
callbacks are tested, and every record of the tables was taken again at one commit, which
`tables95.md` names at its head. The tables are `docs/reports/probe95/tables95.md`, generated from
the records by `tables95.py` (`--check`).

This is the second half of Phase 6 step 5 (`docs/PROJECT_PLAN.md`, Phase 6, the steps table;
ADR-011 G and ADR-012 D and E as built and amended; ADR-014 A and D; ADR-017 B, C, F, G and H):
`ErrorEstimateRule`, `viscosity_scale` and the imbalance summary in the core's `stopping` module,
and `StaggeredSolver.solve_steady`'s outer iteration in its `flow` module, with the fields resident
and the reference's set-up, refusals, checkpoints and log lines driving it from Python
(`tests/core_driver.py`). Nothing under `src/` or `configs/` changes.

## The predictions

Set by the orchestrator in prompt 95, and committed alone before any run.

- (a) Every block of the first kind equals the reference bit for bit on every planted and captured
  input, the median and the imbalance sums included.
- (b) Every captured estimate is within 1e-11 of the reference's, relative, and no captured window
  has the two estimates on opposite sides of `iteration_error_tol`.
- (c) Every whole solve stops for the reference's reason at the reference's outer count, not one
  apart, and every flow field is inside the measured spread (ADR-017 H's first tier).
- (d) The D = 3 instantiation builds and its tests pass.
- (e) The 160x60 product solve from rest is at least four times faster than the reference's, same
  session.

## How they are scored (the builder, before any run)

The core is built under the project's flags (`CLAUDE.md`, C++ Coding Standards) and called
through its binding under WSL2 with the Linux environment, one BLAS thread, NumPy 2.4.4 with its
AVX-512 loops off (`NPY_DISABLE_CPU_FEATURES=X86_V4`) in every parity run and every whole solve of
the reference (Alex, prompt 95). "The reference" is `src/stopping.py` and
`StaggeredSolver.solve_steady` at `origin/main` (e320545), unchanged.

- **(a).** The blocks of the first kind are: in `stopping`, the rule's verdict and the readings it
  records given the same steps, scales and imbalance (each condition, the window, `restore`, the
  version), `x @ x` of the rate fit, the logarithms of the window and the exponential of the
  slope, `viscosity_scale` (the median of nu_t over the FLUID cells, plus nu) and the imbalance
  summary (the worst absolute imbalance and the two sums, in NumPy's order); in `flow`, every
  elementwise write of an outer iteration (the outlets' extrapolation, the effective viscosity,
  the eddy viscosity's relaxation, the cell-centred velocities) and the max-norms (the velocity
  step, the viscosity step, the residual). A captured input is one the probe kept from the
  reference's own outer iterations (section 3 of the prompt); a planted one is the parity
  module's. Holds when each output equals the reference's in every bit (a float compared by its
  64-bit pattern) on every one. One differing bit is a failure, whatever its cause.
- **(b).** For each captured outer iteration whose window is full, the core's estimate formed from
  the reference's window (velocity, and with condition (e) on the viscosity's) against the
  reference's estimate: the relative difference `|core - reference| / |reference|`, two infinities
  equal. Holds when every one is at most 1e-11, and when no captured iteration has one estimate
  below `iteration_error_tol` and the other at or above it.
- **(c).** The whole solves are the four the prompt names: the 80x30 and 160x60 product room from
  rest, the 40x40 cavity from rest, and the door leak's open state warm-started from its closed
  one on 80x30, each side from its own closed state, as a run on that side would. For each: the core's stop reason and outer count equal the reference's, and each
  flow field's largest difference over its largest magnitude (u, v, p, and k, eps, nu_t where the
  solve is coupled) is at most that field's measured spread, the largest P2 measured for it
  (`docs/reports/probe85/tables85.md`, P2: u 1.2e-10, v 6.1e-11, p 9.5e-11, k 7.8e-10, eps
  7.5e-10, nu_t 6.7e-10). An outer count one apart, or a field between the spread and its limit,
  fails (c) as stated, and is then read against ADR-017 H's three tiers and item 3 for the gate.
- **(d).** The GoogleTest program of three directions (`core/tests/test_three_directions.cpp`)
  compiles with `stopping` and `flow`'s per-direction parts instantiated at D = 3, and every test
  of it passes.
- **(e).** The 160x60 product room's solve from rest, the reference's `solve_steady` and the
  core's driven solve, in one process, one after the other, the wall time of each whole solve.
  Three sessions, each recording the machine's state as prompt 90's probe records it. Session 1
  is scored; 2 and 3 are reported beside it, with the time in each stage. Holds when the
  reference's time over the core's is at least 4.0 in session 1.

A failed prediction is a finding, not a stop.

## The result

### T0. The predictions

The five predictions are scored in `docs/reports/probe95/tables95.md`, T0, by the rule committed with them
above. All five hold. The step's gate holds with them: every whole solve of the gate stops for the
reference's reason at the reference's outer count, not one apart (ADR-017 H, item 3), every flow field
inside its measured spread, the first of ADR-017 H's three tiers (T3). No stop condition of the prompt was
met: no block of the first kind needed a change under `src/`, no captured window has its two estimates on
opposite sides of the tolerance (T2), no positivity failure fell on one side only, and no step needed
`sudo`.

## 1. The port and the chunks' exchange

**`stopping`** (`core/include/cfdcore/stopping.hpp`, `core/src/stopping.cpp`). `rule` is
`ErrorEstimateRule`: its window held oldest first, `record` (the update up to its call of the imbalance:
the velocity step and, with condition (e), the viscosity step and its scale, then each estimate), `accept`
(conditions (b), (c) and (d) on a summary), `restore`, `reset` and `version`. The window's length
(`RATE_WINDOW`) and the two version numbers (`RULE_VERSION_WITHOUT_E`, `RULE_VERSION_WITH_E`) are handed in
from Python with the other keys, so the core holds no copy of any of them. `estimate_of` is
`_estimate_of`: infinite below a full window, at a step that is zero or not finite, and at a fitted rate
not strictly between 0 and 1. Its abscissa `k - (n - 1) / 2` is exact, and so is `x @ x` (a sum of whole
quarters, every partial sum below 2^51 at the window's length, checked with fractions by the parity module);
the logarithms and the exponential are the C library's (`elementary::log`, and `elementary::exp`, added for
Python's `math.exp`). `x @ log(steps)` is the second kind: NumPy hands it to OpenBLAS's `ddot`, whose order
depends on the kernel it picks for the processor, and the core forms it with `reduce::blocked_dot`, one block
of the window. `summarise` is the solver's `_imbalance_summary` (the largest magnitude, NaN where one is NaN,
and the two sums in NumPy's pairwise order over the whole contiguous field), and `viscosity_scale` the
median of nu_t over the FLUID cells: NumPy 2.4.4's `_median` partitions a copy (a selection, introselect)
and takes `mean` of the middle value or the two middle values, `mean` being `add.reduce` over them divided
by the count (`numpy/lib/_function_base_impl.py`, `_median`; `numpy/_core/_methods.py`, `_mean`), and a NaN
anywhere makes it NaN (`_median_nancheck`). The core selects with `std::nth_element`, which gives the value
of a sorted position whatever its algorithm, takes the largest of the part below for the lower middle
value, and sums the one or two with `reduce::pairwise_sum` before dividing.

**`flow`** (`core/include/cfdcore/flow.hpp`, `core/src/flow.cpp`). `solver<D>` owns u, v, p and, when
coupled, k, eps and the relaxed nu_t, with every scratch array allocated at construction. `run(first,
records)` takes at most `records.size()` outer iterations in the reference's order of stages: the outlets'
extrapolation (`walls::outlets`, section 0), the prediction (laminar, under the fixed `mu + rho nu_t` the
reference forms once, or coupled: `effective_viscosity` with its two refusals, the predictor's check of it,
`stencil_viscosity`, the wall functions' `wall_viscosity` and their check, then `predict`), the correction
and its counts and cap, the k-epsilon step (the faces' and the formed conditions' checks, `model::step`,
its positivity verdict, `turbulence::relax`, the viscosity step and `viscosity_scale`), the cell centres,
the velocity step `max(du, dv)` as Python's `max` folds it and the residual, and the rule: under
`velocity_step` the residual below its bound on a correction that did not reach its cap; under
`error_estimate` `record`, then, where (a) and (e) hold, the corrector's imbalance and `accept`. A failure
ends the chunk with what the reference's raise names (the `failure` enumeration in `flow.hpp` lists them).
`cell_centres`, `largest_change` and `effective_viscosity` are the blocks the module adds; each is the
reference's elementwise arithmetic, written per direction. Section 0's two writes are `turbulence::relax`
and `walls::outlets<D>`.

**The binding and the exchange.** ADR-017 B's table, as built:

| Call | Python hands in | The core returns |
|---|---|---|
| `Flow2(grid, predictor, corrector, outlets, model, walls, ...)` | the modules on one grid (it keeps them alive), FLUID's code, mu, rho, nu, the relaxation, the residual's scale, the rule (`('velocity_step', tol)` or `('error_estimate', window, version_without_e, version_with_e, velocity_scale, flux_scale, iteration_error_tol, mass_imbalance_tol)`), a fixed `mu_eff`, read once here and refused where it is not finite or not positive in a non-SOLID cell | a handle |
| `flow_state(u, v, p, k, eps, nu_t)` | the start's fields, their normal velocities already written (ADR-017 E) | nothing; the fields are resident, the rule fresh |
| `restore(steps, viscosity_steps)` | a checkpoint's window | nothing; the rule is seeded |
| `solve_steady(max_outer, first)` | the chunk's outer count and its first iteration's number | the stop (`stopped`, `outer`), per outer iteration the residual, the velocity and viscosity steps, both estimates and the scale, the correction's iterations and products and whether it reached its cap, the k-epsilon step's fall-backs to Jacobi (k, eps) and whether both solves converged, the failure (its type's name and the reference's message, with a positivity failure's cells and the iteration), the failing iteration's record (its correction's count, products and cap, the step's fall-backs and convergence, and with a refused scale its residual), the stages' seconds |
| `flow_fields()` | | u, v, p and, when coupled, k, eps, nu_t, copies |
| `window()`, `rule_keys`, `rule_version` | | the rule's window, its keys and its version |

The first capped correction's iteration and count, which the reference's warning names, are read from the
records by the driver, not returned apart. `StoppingRule2` is the rule alone for the synthetic histories, its
refusals the reference's, with its histories; `imbalance_summary`, `viscosity_scale`, `cell_centres`,
`largest_change`, `python_max`, `effective_viscosity`, `relax_eddy_viscosity`, `OutletFaces2` and
`KEpsilon2.check_positive` are the blocks for the parity tests.

**The driver** (`tests/core_driver.CoreSolver`). `StaggeredSolver` with `solve_steady` stepped through a
`Flow2`. The methods it calls are the reference's own: `_effective_viscosity`, `_warm`, `_resumed`,
`_new_rule` (the reference's rule refuses a checkpoint's window before the core is seeded),
`apply_normal_velocity`, `_checkpoint` and `_finite`. The two set-up refusals of `solve_steady` (an eddy
viscosity beside the turbulence section, `initial` beside `resume`) are copied lines, tested against the
reference's (section 2). The loop's lines around the chunks are the reference's: the cap hits and the
first one's warning, the k-epsilon step's two warnings, the log lines at every fiftieth iteration and at
the stop, `IterationState` from the core's fields, the checkpoint where `(iteration + 1) % interval == 0`
falls, built from the core's fields and window. A chunk runs to the next checkpoint when `on_checkpoint` is
given, one outer iteration when `on_iteration` is, and to the cap otherwise. A failure is raised as the
reference raises it (`core_mapping.flow_error`): a positivity failure as `PositivityError` with the
reference's outer-iteration message, after the failing iteration's correction is counted as the reference
counts it, its step's warnings logged, and with a refused scale of condition (e) its residual appended and
`on_iteration` called, as the reference has done before its rule raises. A fixed viscosity the core refuses
when it is built is raised in the reference's words (`predict`'s own), since the reference reads it at its
first prediction. `steady_solves_in_the_core` has `time_integration` build `CoreSolver`; it is kept
because step 6 starts from it (it moves into the `solver.backend` key), and a scenario run tests it
(section 2).

**Where a part does not fit the Kernels rules, and why.**

- `walls::outlets<D>::extrapolate` is a copy along an edge's faces in place, so its output overlaps its
  input; it is not a block, and copies each listed face once per call.
- `viscosity_scale`'s selection is `std::nth_element`, an algorithm with branches on the data; a median is
  not an elementwise pass, and its copy of the FLUID cells into scratch is its one copy per call.
- The rule is a scalar computation on a window of `RATE_WINDOW` steps, shifted by one copy per outer
  iteration; nothing in it is a field.
- `python_max` folds the D components' changes with a branch, as Python's `max` does, so that a NaN first
  is kept and a NaN later is not.
- The outer iteration's stages branch on their statuses between blocks, never inside one.
- The binding allocates a chunk's records before it calls `run`; `run` allocates nothing.

Every new block vectorises under the project's flags (T5). The cell centres' block did not at first: GCC
reloaded its two row pointers from the operand struct inside the loop ("no vectype"), and reading them
once before the loop, with the block marked `CFDCORE_INLINE` as the lanes rules ask, vectorised it; the
values are the same.

## 2. The tests

`tests/test_core_flow.py` (CI's core job, `-m parity`; T7 gives the module's time):

- **The rule on synthetic histories**, `StoppingRule2` against `ErrorEstimateRule`: geometric histories at
  six rates (the window filling, the estimate of each, a rate at 1 and beyond inf), a constant history, a
  step that is zero, negative, NaN or infinite in the window, each continuity condition failing alone with
  the imbalance read only once (a) holds, (a) failing alone, (e) on with each of (a) and (e) failing alone,
  (e) off and the version, a restored rule continuing the one it was taken from with (e) on and off, the
  refusals of construction, `update` and `restore` with the reference's types and messages, `x @ x` exact,
  and 400 planted windows over nine decades. Verdicts, windows and scale histories bit for bit; estimates by
  level one on the fitted rate (Findings, F1).
- **The median** on 1, 2, 3, 4, 7, 8, 1,001 and 1,002 cells, spread, tied, mostly zero, with an infinity and
  with a NaN; **the imbalance summary** of planted faces on prompt 87's meshes (160x60 and 320x120 hold more
  cells than NumPy's buffer), with a NaN; **the cell centres and the max-norms**, a NaN read and one not
  read; Python's `max` fold of the velocity step on NaN and signed zeros; **the effective viscosity** on the
  meshes, a NaN in a SOLID cell not read, and its two refusals; **the
  rule's keys** against the solver's scales. Section 0's tests are in `tests/test_core_turbulence.py`: the
  relaxation against `_turbulence_step` on three rooms, the outlets against `_extrapolate_outlets` on the
  small rooms with an outlet on every edge, and the least value of a failure.
- **Outer iterations from planted fields**: three outer iterations warm-started from a planted state on
  prompt 87's five meshes, the cavity and (round 1) a small room with a pressure outlet on every edge, laminar,
  under a fixed eddy viscosity and coupled (the 320x120 mesh coupled only; the cavity not coupled, its walls
  admitting no air; the small room in the bit-for-bit comparison alone, since its flow grows without bound
  and a part in 1e12 at the first iteration is no comparison later), compared bit for bit in every field,
  the residuals, the corrections' counts and cap hits, the final imbalance and the rule's inputs (the
  velocity step, the viscosity step and its scale). The reference's conjugate gradients take the core's dot
  product (`_cfdcore.row_dot`, the trick of prompts 88 and 93) and both sides the diagonal preconditioner, so
  that each block of the chain is handed what the other side's was: the chain judged block by block (ADR-017
  G). The same iterations as the reference runs them, its dots OpenBLAS's and the multigrid its own, are
  within 1e-12 of each field's largest magnitude with the counts equal. After a solve, `face_velocities`
  and `turbulence_state` are the reference's bit for bit too (round 1).
- **The core's own**: a solve interrupted at a checkpoint and resumed by the core is the uninterrupted one
  bit for bit (the coupled 40x15 room, 451 outer iterations, and the laminar 16x16 cavity, 653), resumed
  from the second checkpoint and from the last before the stop; a solve with
  `on_iteration` is the solve without it, one core call per outer iteration; a laminar run planted to
  diverge ends at its cap of six with the reference's stop reason, its cap hits and no flow state; a
  positivity failure of the coupled solve at a Courant number of 40 raises the reference's
  `PositivityError`, its message and minimum the reference's to the bit, every correction at a cap of two
  and the failing one's counted; the conditions a discharge vent of zero intensity forms are refused as the
  reference's step refuses them; a capped correction does not stop the velocity step and an uncapped one
  does, its log lines the reference's with one warning in all; the checkpoints and the log lines fall where
  the reference's fall.
- **Where the stop binds** (round 1): four whole solves in which (a) and (e) hold a hundred outer
  iterations before the stop, so that only continuity, conditions (b) to (d), puts it off: on the 16x16
  cavity and the 40x15 room, each once with a tolerance the imbalance falls below later (both sides stop
  at the same outer iteration, `error_estimate_and_continuity`) and once with one it never reaches (both
  run to the cap), the log lines compared down to the last two at a stop. One `velocity_step` solve whose
  stop falls after the residual's lowest point so far and not before twice its bound, against the
  reference's. These stops are read by no other solve, whose estimates first hold at the iteration they
  stop at (F2).
- **The outlets inside the solve** (round 1): the small room's outer iterations in all three modes, each
  edge's outlet faces read, and VAL-001's channel at 40x20 solved whole at level 2, its pressure outlet
  read in every outer iteration (T3 has its row).
- **The failures a planted state reaches** (round 1), each against the reference's type, message,
  residual history, cap hits, last correction, callbacks and log lines, once with checkpoints every three
  outer iterations and once with `on_iteration` too: a refused scale of condition (e) (the reference has
  recorded the iteration and called back before its rule raises), an eddy viscosity and faces that are not
  finite, an effective viscosity that overflows, a wall viscosity that is not finite on each component, k
  not positive after a checkpoint and off the diagonal, eps not positive off the diagonal, and a fixed
  viscosity that overflows. The kinds no state reaches are listed with the reason in the module (section 6).
  The driver's two set-up refusals are tested against the reference's.
- **Callbacks, records and warnings** (round 1): every field of each `IterationState` and of the
  checkpoints written beside it, the cap hits a resume carries (every correction at a cap of two), the
  k-epsilon step's two warnings, also for a step that then fails its signs, and `steady_solves_in_the_core`
  on a scenario run (the product room at 80x30 solved by a `CoreSolver`, written to the run file, the patch
  undone also by an exception).
- **Across the two sides**: a reference checkpoint resumed by the core and a core checkpoint resumed by the
  reference, from the last checkpoint before the stop so that the stop reads the window carried across,
  each to its stop, against the reference's uninterrupted solve at level 2; a converged state
  warm-started under its own boundaries stops when the window fills on both sides.
- **The binding's refusals**: `Flow2`'s modules on another grid, a rule of no name, a bound that is not
  finite and positive, no FLUID cell; `flow_state`'s shapes and its turbulence fields for a laminar solve;
  a chunk of no iteration; a window the rule does not restore, and one handed to a solve with no rule (its
  own status, `no_rule`); a fixed viscosity that is not finite or not positive in a non-SOLID cell, a NaN
  in a SOLID cell not read; the rule's versions, which the core takes from the keys (7 and 11 are given
  back) and refuses below one; and each failure's mapping to the reference's exception.

CTest gains `test_stopping.cpp`, `test_flow.cpp` (a lid-driven box by hand: an outer iteration is the
prediction and the correction in turn, chunks of any size make the same solve, the velocity step's stop),
the relaxation's and the outlets' tests, and three tests of the three-direction program: an extruded room's
cell centres are the 2D room's on every plane bit for bit, the outlets copy along each direction, and
`solver<3>` runs three outer iterations on extruded closed rooms with the corrections leaving no imbalance
and its rule reading the steps as the rule alone reads them. Round 1 adds three: the versions the keys
carry (7 and 11 are given back, below one is refused), a fixed viscosity's two refusals, and a window handed
to a solve with no rule. The counts are T7.

## 3. Parity by kind

**The first kind.** Every block of the first kind equals the reference bit for bit on every planted case
(the parity module) and on every call of the reference's own solves of the gate's four cases (T1):
captured in a probe process, each call replayed through the core on the reference's inputs as it ran,
over every outer iteration, the first 50 and the 300 before the stop among them. The imbalance summary is
called once per solve, at the stop: the rule asks for it only where (a) and (e) hold, which in each of these
solves first happens at the outer iteration it stops at.

**The second kind.** Each captured estimate, formed by the core from the reference's steps, is within
1.5e-13 of the reference's, relative (T2), and no iteration has the two on opposite sides of the tolerance.
At the stop the binding condition sits close to its bound, (e) at 0.9948 to 0.9991 of the tolerance in the
coupled solves (the smallest is the door leak's open state, 0.994804) and (a) at 0.9985 in the cavity, and
the two sides agree there to the six digits shown.

**Whole solves.** T3: every state stops for the reference's reason at the reference's outer count, every
flow field within a few units in the fifteenth digit of the reference's, inside its measured spread by four
decades or more, and no outer iteration's correction takes another CG count than the reference's. The door
leak's open state is warm-started on each side from that side's closed state, and its closed state is the
80x30 product room's solve again (the same configuration, solved twice), so the five rows of the four cases
are four distinct solves. Round 1 adds the channel at 40x20 (VAL-001's room): the one whole solve with a
pressure outlet, which the stage that extrapolates the outlets' faces is read by. It is outside the
predictions, and T3 gives its row.

## 4. Planted defects

T6 lists every plant at the head, each built in a scratch tree with CTest, `tests/test_core_flow.py` and
section 0's tests run against it, and a control: the prompt's eleven, prompt 94's carried survivor, the
builder's fifteen, and round 1's, which are test 95's survivors and one on each piece round 1 added
(section 6). The first run, at the port's commit, caught 21 of the 26 that built; one did not build (the
abscissa plant left its centre unused, which `-Werror` refuses; mended by keeping the variable read). The
five survivors each named a rule no test read:

- `resume_restarts_window`: the resume tests resumed from the checkpoint at 300 outer iterations, more
  than a window before the stop, so a window dropped at the resume had refilled with the same steps by
  then and the stop fell where it falls. They now resume also from the last checkpoint before the stop,
  inside the first window (F6).
- `warning_at_every_cap`: no test compared the log lines of a solve whose corrections reach their cap. The
  capped velocity-step test now compares them with the reference's, one warning in all.
- `failing_cap_dropped`: the positivity test's corrections never reached their cap. They now stop at a cap
  of two, and the failing iteration's capped correction is counted on both sides.
- `conditions_unchecked`: no solve formed conditions the step refuses. A vented room whose discharge vents
  carry in k and eps of zero now raises the reference's `ValueError` on both sides.
- `python_max_keeps_a_later_nan`: a velocity step with one component's change NaN and the other's finite
  needs a correction that leaves one component non-finite, and the correction's p' reaches both: four
  planted divergences took both NaN in the same outer iteration. The fold is now a public block of `flow`
  with its own parity test against Python's `max`.

The test commit (aa92c9d) adds those tests, and the five with the mended plant were caught after it: 27 of
27 then. One plant of the first run was caught for the wrong reason: `chunks_with_on_iteration` removed
the `if` of the driver's chunk choice and left its `elif` without one, so the module failed to import. It
was mended to keep the branch and do nothing in it, and is caught after the test commit by the
`on_iteration` test.

One plant is reached by a check and not by a step: `least_over_solid_too` reads the least value of a
failure over the SOLID cells as well, and a step's own k and eps are zero in a SOLID cell, so no solve
reaches it. Section 0's test plants one infinity in a field and no NaN and says so in its docstring; it is
caught there alone.

## 5. The timing

T4: the 160x60 product room's solve from rest, the reference then the core in one process, three sessions
on mains power with no model server or game running (Windows' busiest processes recorded with each). The
core's stages are the modules of steps 2 to 5a; the pressure correction and the turbulence step are most of
its time, as they were in prompts 93 and 94, and the rule's share is negligible.

## 6. Round 1 (prompt 95b)

Review 95 approved the branch with seven suggestions; test 95 found two missing-test Bugs and five
suggestions, with the tester's own plants and fuzz in `results/tester95/`. Round 1 answers each. Its
numbers are the tables'; every record in them was taken again at one commit, which `tables95.md` names at
its head (T0 to T7 are of that commit, the plants' included).

- **The stop where it binds (test B1).** A stop that read only (a) and (e), one that read the imbalance of
  the prediction's faces in place of the correction's, and a velocity-step bound doubled in the core's
  set-up each passed every test, since no whole solve had (a) and (e) holding while continuity did not
  (F2). Section 2 lists the solves that do. The tester's three plants are in T6, caught.
- **The outlets inside the solve (test B2).** No room of the module had a pressure outlet, so the stage
  that gives each outlet face its interior neighbour's value before a prediction was read by section 0's
  block test alone, and none of the four one-edge plants nor the stage's removal was caught by a solve. The
  small room has an outlet on each edge and is a case of the block-by-block outer iterations in all three
  modes, and the channel at 40x20 is solved whole (T3 has its row, outside the five predictions). The
  tester's five plants are in T6, caught.
- **The failure paths (review S1, test S1).** Section 2 lists the failures a planted state reaches, each
  compared with the reference's. Three of them need a state built for the kind: an effective viscosity that
  overflows (a resumed nu_t of 1.7e308, which `rho nu_t` takes past the largest double), a wall viscosity
  that is not finite on each component (k of 1.7e308 in two neighbouring wall cells of a resumed state,
  whose mean overflows) and a fixed viscosity that overflows in the same way. The positivity failures fall
  after the first checkpoint (k) and off the diagonal (k and eps), which a failure at outer iteration 0
  and cell (0, 0) cannot tell. The failure kinds of `flow.hpp` that no state reaches:
  - `eddy_viscosity_negative`: the iterate is the step's `C_mu k^2 / eps` under the relaxation, a mean of
    two values that are not negative; a warm start rebuilds it from k and eps, and a resume refuses a
    negative one.
  - `mu_eff_not_positive`: `mu + rho nu_t` is at least `mu`, which is positive.
  - `wall_mu_not_positive`: the wall function is a product of positive numbers over a logarithm of at
    least `ln(E y*_0)`, above one, and the smallest positive k leaves its square root positive.
  - `nothing_moves`: the faces after a correction are all zero only where no boundary prescribes a
    velocity, and the rule the turbulent configuration needs refuses a velocity scale of zero when the
    solver is built.
  - `pressure`: `not_dominant` and `division_by_zero` need a system built by hand
    (`tests/test_core_pressure.py` reaches them); from a planted state they come only under the multigrid
    preconditioner and an extreme field, where the reference's dot product and the core's part, so that the
    two sides do not fail together (a probe of u spikes of 1e30, not kept as a record) and no test can
    compare them. `not_converged` is the dense solve's own failure, reached only from a NaN coefficient,
    where the reference's answer is whatever NumPy's `pinv` returns and no rule of the reference's own, so
    no test pins it.
  The tester's eight plants of S1 are in T6, caught.
- **A refused scale (test S3).** The core's chunk now returns the failing iteration's record: its
  correction's count, products and cap, the k-epsilon step's fall-backs and convergence and, where the
  rule refuses condition (e)'s scale, the residual. The driver appends the residual and calls
  `on_iteration` before it raises, as the reference does.
- **The step's warnings (test S4).** The chunk returns, per outer iteration, the fall-backs of k's and
  eps's conjugate gradients to Jacobi and whether both solves converged, and the driver logs the
  reference's two warnings from them, in its words and order, also for a step that then fails its signs.
- **A fixed viscosity (review S2).** `solver::check` reads it once with the predictor's own check, so a NaN
  or a non-positive entry in a non-SOLID cell is refused when the solver is built. The reference reads it
  at its first prediction, in `_check_mu_eff`'s words, which the mapping hands on.
- **The rule's versions (review S3).** `rule_keys` carries `version_without_e` and `version_with_e` beside
  the window, the binding takes them and the mapping passes the reference's `RULE_VERSION_WITHOUT_E` and
  `RULE_VERSION_WITH_E`. The core holds no copy of either; a rule given 7 and 11 reports them.
- **A window under velocity_step (review S4).** `restore` returns its own status, `no_rule`, which the
  binding names as what it is; it no longer reports a window as too long.
- **Callbacks, records and attributes (test S2).** Section 2 lists the tests. The tester's plants of S2 are
  in T6, caught, but for `t_stage_seconds_dropped`, which times what nothing can compare and is left out.
  The tester's one equivalent mutant, `t_rate_at_one_allowed`, is a row of T6 with its reason.
- **`steady_solves_in_the_core` (test S5).** Kept, and tested by one scenario run: step 6 builds its
  `solver.backend` key from it, and the probe and the parity tests share it until then.
- **Text and records (review S5 and S6, test S6).** The statements of the report that its records did not
  carry are corrected (sections 1 to 3 and the deviations); the timing, captures, whole solves,
  vectorisation, plants and every JUnit record were taken again at the head; `tables95.py` refuses an
  absent record and records of two commits; and `flow.xml` is a record.

## Findings

- **F1. The rate fit's estimate is ill-conditioned near a rate of one.** The estimate is `step rho / (1 -
  rho) / scale`, so a relative change of the fitted slope moves it by `1 / (1 - rho)` times as much: on a
  geometric history at a rate of 0.9999 the core's estimate parts from the reference's by 1.1e-12 with a
  dot product correct to its last few bits. The parity module bounds each estimate by 1e-12 times `1 + estimate
  scale / step`, which is `1 / (1 - rho)` read from the reference's own estimate. Prediction (b)'s fixed
  1e-11 held on the captured windows, whose rates the solves keep far enough from one (T2).
- **F2. The imbalance summary is read once per solve.** In each captured solve (a) and (e) first hold at the
  outer iteration the solve stops at, so the summary is computed there alone (T1). Its parity rests on the
  module's planted fields, the product meshes among them.
- **F3. The binding conditions at the stop are within a fraction of a per cent of their bounds** (T2), and
  equal on the two sides there. A port whose estimate were a few parts in a thousand off would stop the
  160x60 solve elsewhere; this one is a few parts in 1e13 off.
- **F4. Condition (e)'s median and the viscosity step reach no field within a few iterations.** The
  planted outer iterations stop long before a window fills, so a defect in the median, the scale or the
  viscosity step changes no field there; the parity module compares the rule's inputs bit for bit for that
  reason.
- **F5. The reference's divergence path.** Once a laminar solve's fields go non-finite the reference's
  correction finds no cell with an equation (its diagonal is NaN, and `a_p > 0` holds nowhere), takes no
  iteration and runs on to the cap; the core does the same, and the stop reason, the cap hits and the absence
  of a flow state are the reference's.
- **F6. A resume more than a window before the stop proves nothing of the window.** Every step the
  stop reads is taken after such a resume, so a port that dropped the window there would stop where the
  uninterrupted solve stops. The tests resume from the last checkpoint before the stop, whose window the
  stop reads (section 4).

## Deviations and limits

- The formed conditions' checks of `KEpsilonModel.step` (`_check_conditions`) moved from the binding into
  `turbulence::model` (`check_ranges`, `check_formed`), so that the core's outer iteration makes them; the
  binding maps their statuses to the reference's messages, and prompt 94's refusal tests pass unchanged.
- The coupled cavity is not a case: the cavity's walls admit no air, from which the coupled solve starts.
- The door leak's open state is warm-started on each side from that side's closed state, as a run on that
  side would; the two closed states are inside the spread of each other (T3).
- The binding returns the first cap's iteration and count inside the records, not apart.
- The built exchange departs from ADR-017 B's table in three ways. `flow_state(u, v, p, k, eps, nu_t)`
  takes the fields with their normal velocities already written, not `(boundary, initial)`: the boundary's
  layers stay in Python (ADR-017 E). `solve_steady(max_outer, first)` takes the number of the chunk's first
  outer iteration, which a failure's message and its records name, since a solve runs in chunks (to a
  checkpoint, or one iteration for `on_iteration`). It returns no reason: the driver maps the stop to the
  reference's name. ADR-017 B's rows are not edited here; step 6 records them with prompt 91's departures.
- The log lines of a chunk are written when it returns, so a solve with no checkpoint and no callback logs
  nothing until it ends, and its lines carry the end's time, not the iteration's. Read from the code, not
  measured; step 6 decides whether a progress line matters before the end.
- Whole solves are judged here on WSL2 with NumPy's AVX-512 loops off, as Alex set for this step; the
  reference with them on is not a row.

## Files

- `core/include/cfdcore/stopping.hpp`, `core/src/stopping.cpp`, `core/include/cfdcore/flow.hpp`,
  `core/src/flow.cpp`; `elementary`'s `exp`; `turbulence`'s `relax`, `check_ranges`, `check_formed` and the
  public `find_bad`; `walls`' `outlets<D>`; `core/python/bindings.cpp`; `core/CMakeLists.txt`.
- `core/tests/test_stopping.cpp`, `core/tests/test_flow.cpp`, additions to `test_turbulence.cpp`,
  `test_walls.cpp` and `test_three_directions.cpp`.
- `tests/test_core_flow.py`, additions to `tests/test_core_turbulence.py`, `tests/core_mapping.py` and
  `tests/core_driver.py`; `.github/workflows/ci.yml` runs the new module in the core job.
- `docs/reports/probe95/probe95.py` (capture, solve with the channel as round 1's case, timing, vecinfo,
  plants), `variants95.py`, `tables95.py` and `tables95.md` (`python docs/reports/probe95/tables95.py
  --check`, which refuses an absent record and records of two commits).
- Records in `results/builder95/` (not committed), each of the commit `tables95.md` names: `captures/`,
  `solves/`, `timing_1.json` to `timing_3.json`, `vecinfo.json`, `plants_s0.json` to `plants_s2.json`, and
  the JUnit records `ctest.xml`, `parity.xml`, `flow.xml` (`tests/test_core_flow.py` alone), `linux.xml` and
  `windows.xml`, each with a `.commit` file beside it. The records of the first round are in
  `results/builder95_round0/`.
