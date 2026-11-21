# Phase 6 Step 3a: One Class Step in the Core (prompt 90)

**Date:** 2026-11-20
**Prompt:** 90 (`docs/prompts/cc-prompt-90-class-step.md`)
**Branch:** `phase6/class-step`
**Status:** built. Predictions (a), (b) and (c) hold (T0). The prompt's stop condition, a sum
or an expression that cannot be made bitwise without changing `src/`, was not met: every sum is
the reference's bit for bit, one of them (a strided slice) by a NumPy rule found on the way
(F1). The vents room's flow state needs a raised iteration cap at 80x30 (F2). Review and test
90 come before the pull request.

This is the first half of Phase 6 step 3 (`docs/PROJECT_PLAN.md`, Phase 6, the steps table;
ADR-017 B, C, E, F and G): one particle class advanced one time step in the core's `transport`
module, as `TransportSolver.solve_timestep` advances it, with the budget booked as
`MassBudget` books it and the breach tested as ADR-016 tests it. Python keeps the
configuration, the interval plan and the events, and reads the budgets. Nothing under `src/`
or `configs/` changes.

## The predictions

Set by the orchestrator in prompt 90 on 2026-11-20, and committed alone before any run.

- (a) Every captured and planted class step equals the reference's bit for bit: the field, every
  budget term, the sweeps and the leftover, vents and recirculation included.
- (b) The interval set-up's quantities equal the reference's per-step values bit for bit on every
  mesh.
- (c) A class step on the developed 320x120 frame is at least five times faster than the
  reference's, same session.

## How they are scored (the builder, before any run)

The core is built under the project's flags (`CLAUDE.md`, C++ Coding Standards) and called
through its binding under WSL2 with the Linux environment, one BLAS thread. "The reference" is
`TransportSolver.solve_timestep` at `origin/main` (173496d), unchanged.

- **(a).** A class step's outputs are: the returned field; the budget's ten terms after the step
  (`initial`, `inflow`, `outflow`, `source`, the four `deposited` slots, `current`,
  `leftover`); the step's records (`last_diffusion_sweeps`, `diffusion_converged`,
  `last_leftover`, `last_leftover_bound`); and, where the reference raises
  `ConservationBreachError`, the error's four arguments with the budget it leaves. The
  prediction holds when, for every planted case of the parity module and every captured call of
  section 3, the core's outputs equal the reference's in every bit (a float compared by its
  64-bit pattern, so that 0.0 and -0.0 differ), and the captured and planted cases include at
  least one step with vents and a recirculating discharge. One differing bit is a failure.
- **(b).** The interval set-up's quantities are the ones the core computes once per interval
  where the reference computes them every class step: per class, the advecting volume fluxes
  (the reference's `u_adv * area`), `stable_dt`, the conductances, the deposition per face times
  `dt`, `dt` times the source rate and the source's total, the recirculating intakes' drawn flux,
  and the masks the step reads (the domain faces' inward and outward ones, the vents' sides);
  and, for the interval, the face eddy viscosity. Each is read from the core and compared with
  the value the reference forms inside `solve_timestep` on the same inputs, on every parity mesh
  (the five of prompt 87 and its three vent meshes) and at every captured call. The prediction
  holds when no bit differs on any of them.
- **(c).** On the stored 320x120 dust release's last frame (prompt 87's developed fields), each
  class stepped once under its last interval's conditions: the median of repeated
  `solve_timestep` calls over the median of repeated core steps, both in one process, after
  unrecorded warm-up calls. The core's step is the binding's `step` call with the interval set
  up beforehand; the reference's is the whole call, its per-step checks included, since that is
  what the march calls. Three sessions, each recording the machine's state as prompt 89's probe
  records it (Windows' busiest processes, the power state, the processor's performance and a
  NumPy calibration). Run 1 is scored; runs 2 and 3 are reported beside it. Holds when the ratio
  is at least 5.0 for every class in run 1.

A failed prediction is a finding, not a stop.

<!-- tables90 predictions begin -->
### T0. The predictions, scored by the rule committed with them

| Prediction | Measured | Verdict |
|---|---|---|
| (a) every captured and planted class step equals the reference's bit for bit | 200 of 200 captured calls with no differing bit in the field, the ten budget terms or the records, 40 of them with vents and a discharge recirculating (T2); 114 of 114 planted class-step parity cases pass (T3) | holds |
| (b) the interval set-up's quantities equal the reference's per-step values on every mesh | 200 of 200 captured calls with every quantity equal; 16 of 16 parity cases (eight meshes, two variants) pass (T2, T3) | holds |
| (c) a class step at least five times faster on the developed 320x120 frame | 6.81 to 8.68 times over the five classes in run 1, median 7.48 (run 2: 7.50 to 8.36; run 3: 7.77 to 8.77) (T5) | holds |
<!-- tables90 predictions end -->

The tables are generated by `docs/reports/probe90/tables90.py` from the records in
`results/builder90/` (untracked, this machine) and copied here between markers;
`tables90.py --check` regenerates them and compares, and fails if a marker pair is missing. The
full set is `docs/reports/probe90/tables90.md`.

## 1. The port

`core/include/cfdcore/transport.hpp` and `core/src/transport.cpp` hold `transport::solver<D>`,
built at D = 1 and 2 as the other modules are, direction D - 1 the vertical one that settling
acts along. It is constructed once on a scheme's grid with the mesh's spacings
(`dx_cell`, `dy_cell`, `dx_face`, `dy_face`), the transport keys with `t_end`, each class's
settling velocity and Brownian diffusion coefficient, and the three surface codes booked to the
floor, ceiling and wall slots, so the core holds no copy of the boundary layer's codes. Its
budgets are `MassBudget`'s ten terms per class. The binding exposes it as
`_cfdcore.TransportSolver2`.

**What the interval fixes, and what the step does.** The reference builds a `TransportSolver`
per interval (`time_integration._Run.transport`) and, in every class step, checks its inputs
and recomputes from them what the interval holds fixed. The core splits that work in three
calls:

| Call | When | What it computes | The reference's per-step counterpart |
|---|---|---|---|
| the constructor | once per run | the volumes `np.outer(dy_cell, dx_cell)`, each face's area, which faces carry a flux and which are interior between two non-SOLID cells, which side of an interior face is SOLID, the conductance per unit diffusivity `A_f / d_f`, the reaches of the eddy viscosity's harmonic mean, the runs of non-SOLID cells per row | `TransportSolver.__init__`'s masks, `_area_u`, `_conductance_u`, `_reach_x` |
| `set_interval` | once per interval | the conditions' checks (`_check_conditions`, the vents against class 0's); the checks of the faces, `v_ext`, the sources and the eddy viscosity; per class the advecting velocities and their volume fluxes, `stable_dt`, the deposition per face and per cell, the conductances, the domain faces' inward and outward masks, the vent masks by side and sign, the domain faces booked to each surface slot, each recirculating intake's drawn flux, the source's total `np.sum(rate * V)`; the face eddy viscosity | `solve_timestep`'s checks, `stable_dt`, `_advecting_velocities`, `flux_u = u_adv * area`, `_face_eddy_viscosity`, the conductances, `_deposit_u`, `_deposit_cell`, the `np.where` masks of the booking, `_recirculated`'s `drawn` |
| `set_step` | once per interval's step | `dt`'s checks, `deposit * dt` per face, `dt * rate` per cell | `solve_timestep`'s `dt` checks, `_book_deposition`'s `self._deposit_u[k] * dt`, `dt * rate` |
| `step` | every class step | the field's check and the SOLID cells zeroed, `dt` against the class's `stable_dt`, the initial content once, the recirculated inflow values, the flux in both directions (step 2's `scalar::advective_flux`), the divergence update and the source, the inflow and outflow at the domain faces and the vents, the bound, the implicit solve (step 2's `scalar::implicit_step`, Jacobi), the leftover, the deposition by surface, the content, the breach test | the rest of `solve_timestep` |

Each quantity of the second and third rows is shown equal, bit for bit, to the reference's
per-step value by the parity module and the replay (prediction (b), T0 and T2): the core
returns each through `interval_quantity`, `masks`, `drawn`, `source_total` and `stable_dt`, and
the tests compare them with the values the reference forms inside the step, read from it by
wrapping its helpers for the call.

**The sums.** Every sum the step forms is one the budget, the bound or the breach test reads,
so each is formed in the reference's order on the array the reference sums (ADR-017 G):

| Sum | The reference's array | The core |
|---|---|---|
| the content, `in_domain` | `(C * V)[cell_type != SOLID]`, a fresh contiguous array of the non-SOLID cells | the products written run by run into a buffer, `reduce::pairwise_sum` |
| the source's total | `rate * V`, the whole field | `pairwise_sum` over the field, at `set_interval` |
| each domain boundary's inflow and outflow | `np.where(flux_u[:, 0] > 0, adv_u[:, 0], 0)`, one per line of the direction | the line's boundary faces gathered with their masks, `pairwise_sum`; the four combined left to right as the reference writes them |
| the vents' flows | `np.where(vu & solid_low_u & (flux_u > 0), adv_u, 0)` over the whole face field | the same over the face field, `pairwise_sum` |
| a recirculation's drawn and carried flux | `np.abs(flux_u[intake_u])` and its product with the room cells, compacted | the intake's faces in face-field order, `pairwise_sum` |
| the obstacle deposition, x faces | `flux_u[:, 1:-1].sum()`, a strided slice | the interior x faces row by row into a buffer, `reduce::buffered_sum` (below) |
| the obstacle deposition, y faces | `flux_v[1:-1, :].sum()`, whole rows, contiguous | `pairwise_sum` |
| the domain deposition by surface | `flux_u[:, 0][surface_u[:, 0] == code]`, compacted | the faces of that code along the line, in order, `pairwise_sum` |
| the leftover | `dt * np.sum(residual)` | step 2's `acceptance` |

Python's own `+` is copied left to right in the order written: the boundary terms, the
deposition's four parts per surface, `supplied = initial + inflow + source`, and
`bound = share * limit * supplied * dt / t_end`.

**NumPy's order for a strided slice (finding F1).** `np.sum` of `flux_u[:, 1:-1]`, the
interior x faces, is neither the pairwise sum of its terms nor a sum of its rows' pairwise
sums. NumPy reduces a slice that is not contiguous through its iterator's buffer, 8,192
elements by default, filled with whole rows: each fill is summed pairwise and the fills are
added to 0.0 in order. A slice one element wide is read as one strided line and summed
pairwise without the buffer. The rule was found on NumPy 2.4.4 by comparing `np.sum` with
candidate orders on random slices of the budget's shapes and is ported as
`reduce::buffered_sum(values, row)`; `test_buffered_sum_equals_np_sum_on_strided_slices` pins
it on slices of six row counts by twelve widths (one element to 8,193, either side of a whole
fill), those larger than any slice the budget sums skipped. On the product grids the slice holds 2,370 terms at 80x30, one fill, and
38,280 at 320x120, five fills, so the rule decides the obstacle deposition's last bits on the
finer grids only (`obstacle_slice_flat` in T4).

**The binding.** `TransportSolver2(scheme, dx_cell, dy_cell, dx_face, dy_face, cfl, upwind,
max_sweeps, tol, limit, share, horizon, schmidt, settling, diffusion, surface_codes)`;
`set_interval(conditions, u, v, v_ext=None, eddy_viscosity=None, sources=None)` takes each
class's `ConcentrationFaces` as the boundary layer builds it; `set_step(dt)`;
`step(size_class, c, out)` returns the step's records (`TransportStep2`: sweeps, converged,
leftover, bound) and may write `out` over `c`. A refusal raises the reference's exception with
its message: `ValueError` for a value, `TypeError` for a bool `dt` or `size_class`,
`IndexError` for a class out of range, and the reference's own `ConservationBreachError`
(imported from `src.solver_transport`) with its four arguments, after the step, the budget
holding it as the reference's does. A wrong shape, dtype or layout is a `TypeError` at the
call, as ADR-017 B says of every array. `budgets()` returns new `MassBudget` objects;
`restore_budgets(budgets)` sets them whole with `restore_budgets`'s refusals and messages, so a
checkpoint's budgets cross both ways.

**Carried from prompt 89b.** `set_masks` (`core/src/scalar.cpp`) now refuses a `held` of
another length than the cells with a `std::invalid_argument`, as `masks_where_not` refuses one
for `solid`; `ImplicitStep.AHeldArrayOfAnotherLengthThrows` covers a shorter and a longer
array under both methods.

## 2. CLAUDE.md's kernel rules: where the port follows them and where it departs

**Follows.** The step's passes over a field are block functions on `lanes::cells<N>` walked by
`lanes::for_each_block`, their inputs in a struct of pointers, their outputs `CFDCORE_RESTRICT`
parameters: the copy of the field with the SOLID cells zeroed and the finiteness flag
(`copy_cells`), the content's terms (`product_cells`), the divergence update with and without
a source (`update_cells`), the deposition through the faces between two rows (`deposition_cells`)
and the vents' `np.where` (`where_cells`). GCC vectorises all five (T6). They walk the field by
rows; direction 1 enters as pointers to the rows above and below. A SOLID cell is computed and
masked by `lanes::keep`; a condition is a 64-bit mask, held at set-up (the SOLID cells, the
faces' kinds, the vents' sides and signs). Nothing is allocated in a step: every buffer is sized
by the constructor or by `set_interval`. A refusal a run can meet is a status the binding
raises as the reference's exception; a programming error is a `std::invalid_argument`.

**Departs, and why.**

- **The interval's passes are plain loops** with each face's coordinates worked out by division.
  They run once per interval and the reference's own recomputation of them every step is what
  is removed; a block form would not change a value and would cost a second version of each.
- **The booking's gathers read one element per line or per listed face**: the x boundary faces
  (one per row, a stride of a row between them), the domain faces of each surface code, a
  recirculation's intake cells. Each sum is over the compacted array the reference builds by
  `np.where` on a column or a boolean index, and its order is that array's; the gather is what
  forms it, and it touches at most one face per line.
- **The vents' sums pass over the whole face field**, eight passes per step with vents, because
  the reference sums `np.where` over the whole field and a sum's tree depends on where the zeros
  sit. A run without vents does not make them. An exact sparse form exists (a block of the tree
  whose terms are all +0.0 sums to +0.0) and was not built.
- **The implicit solve still forms its masks from `bool` arrays at every call**
  (`implicit_input`'s `solid`), the pass prompt 89 noted the march could save. Saving it changes
  step 2's interface; the timing (T5) shows what the step costs with it.

## 3. The captured calls

The reference's own `solve_timestep` calls, captured by `docs/reports/probe90/probe90.py
capture` in a probe process that wraps the method and four of its helpers on the class: the
three 80x30 placement scenarios and the 320x120 dust release from their stored flow states,
and a dust release on `configs/clean_room_vents.yaml` at 80x30 (`docs/reports/probe90/
scenario_vents80.yaml`) whose flow state is solved from rest. Every class in two windows,
across the onset and at the run's end. A kept call holds the step's inputs, the budget before
and after, the field returned or the breach raised, the step's records, and the values the
reference formed inside the step that the core fixes per interval; each interval's solver is
kept once, with its conditions. `probe90.py replay` hands each call to the core with the
captured budget restored, compares every output bit for bit, compares the interval's
quantities, calls the reference again on the captured inputs (it must reproduce the capture),
and checks that the interval inputs kept for the call's solver are the call's own by their
digests.

<!-- tables90 captures begin -->
### T1. The captured class steps

Each run from its stored flow state (prompt 81's runs; the vents run solved from rest), every class's steps in two windows: across the onset (two steps before it, three after) and the run's last three. A solver is one interval's `TransportSolver`.

| Scenario | Grid | Flow states from | Steps run | Calls kept | Solvers | Vents | Wall time, s |
|---|---|---|---|---|---|---|---|
| 80x30 dust release | 80x30 | `results/builder81/placement/dust.h5` | 6,811 | 40 | 3 | no | 65 |
| 80x30 door leak | 80x30 | `results/builder81/placement/door.h5` | 3,416 | 40 | 3 | no | 31 |
| 80x30 filter breach | 80x30 | `results/builder81/placement/breach.h5` | 6,811 | 40 | 3 | no | 65 |
| 80x30 vents dust release | 80x30 | `solved from rest` | 183 | 40 | 3 | yes | 399 |
| 320x120 dust release | 320x120 | `results/builder81/product/equipment_dust.h5` | 33,119 | 40 | 3 | no | 2139 |

The vents run's flow state, solved from rest with `max_simple_iter` raised to 60,000: stopped by `error_estimate_and_continuity` after 24,767 outer iterations in 403 s (`results/builder90/vents80_flow.json`); the base file's cap is 15,000.
<!-- tables90 captures end -->

<!-- tables90 replay begin -->
### T2. Every captured call replayed through the core

Per scenario, window and class: the calls, the cells per call, the Jacobi sweeps the reference took, and what differs from the reference in any bit: the field, the ten budget terms, the step's records (sweeps, converged, leftover, bound), the interval's quantities against the reference's per-step values (prediction (b)), and whether the reference called again on the captured inputs reproduces the capture and the kept interval inputs are the call's own.

| Scenario | Window | Class | Calls | Cells | Sweeps | Vents, recirculated | Source | Field bits differing | Budget terms differing | Records differing | Interval quantities differing | Reference reproduces, inputs the call's |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80x30 dust release | onset | 0 | 5 | 2,400 | 0-3 | no | yes | 0 | 0 | 0 | 0 | yes |
| 80x30 dust release | onset | 1 | 5 | 2,400 | 0-3 | no | yes | 0 | 0 | 0 | 0 | yes |
| 80x30 dust release | onset | 2 | 5 | 2,400 | 0-3 | no | yes | 0 | 0 | 0 | 0 | yes |
| 80x30 dust release | onset | 3 | 5 | 2,400 | 0-3 | no | yes | 0 | 0 | 0 | 0 | yes |
| 80x30 dust release | onset | 4 | 5 | 2,400 | 0-3 | no | yes | 0 | 0 | 0 | 0 | yes |
| 80x30 dust release | late | 0 | 3 | 2,400 | 5 | no | no | 0 | 0 | 0 | 0 | yes |
| 80x30 dust release | late | 1 | 3 | 2,400 | 5 | no | no | 0 | 0 | 0 | 0 | yes |
| 80x30 dust release | late | 2 | 3 | 2,400 | 5 | no | no | 0 | 0 | 0 | 0 | yes |
| 80x30 dust release | late | 3 | 3 | 2,400 | 5 | no | no | 0 | 0 | 0 | 0 | yes |
| 80x30 dust release | late | 4 | 3 | 2,400 | 5 | no | no | 0 | 0 | 0 | 0 | yes |
| 80x30 door leak | onset | 0 | 5 | 2,400 | 0-3 | no | no | 0 | 0 | 0 | 0 | yes |
| 80x30 door leak | onset | 1 | 5 | 2,400 | 0-3 | no | no | 0 | 0 | 0 | 0 | yes |
| 80x30 door leak | onset | 2 | 5 | 2,400 | 0-3 | no | no | 0 | 0 | 0 | 0 | yes |
| 80x30 door leak | onset | 3 | 5 | 2,400 | 0-3 | no | no | 0 | 0 | 0 | 0 | yes |
| 80x30 door leak | onset | 4 | 5 | 2,400 | 0-3 | no | no | 0 | 0 | 0 | 0 | yes |
| 80x30 door leak | late | 0 | 3 | 2,400 | 3 | no | no | 0 | 0 | 0 | 0 | yes |
| 80x30 door leak | late | 1 | 3 | 2,400 | 3 | no | no | 0 | 0 | 0 | 0 | yes |
| 80x30 door leak | late | 2 | 3 | 2,400 | 3 | no | no | 0 | 0 | 0 | 0 | yes |
| 80x30 door leak | late | 3 | 3 | 2,400 | 3 | no | no | 0 | 0 | 0 | 0 | yes |
| 80x30 door leak | late | 4 | 3 | 2,400 | 3 | no | no | 0 | 0 | 0 | 0 | yes |
| 80x30 filter breach | onset | 0 | 5 | 2,400 | 0-2 | no | no | 0 | 0 | 0 | 0 | yes |
| 80x30 filter breach | onset | 1 | 5 | 2,400 | 0-2 | no | no | 0 | 0 | 0 | 0 | yes |
| 80x30 filter breach | onset | 2 | 5 | 2,400 | 0-2 | no | no | 0 | 0 | 0 | 0 | yes |
| 80x30 filter breach | onset | 3 | 5 | 2,400 | 0-2 | no | no | 0 | 0 | 0 | 0 | yes |
| 80x30 filter breach | onset | 4 | 5 | 2,400 | 0-2 | no | no | 0 | 0 | 0 | 0 | yes |
| 80x30 filter breach | late | 0 | 3 | 2,400 | 5 | no | no | 0 | 0 | 0 | 0 | yes |
| 80x30 filter breach | late | 1 | 3 | 2,400 | 5 | no | no | 0 | 0 | 0 | 0 | yes |
| 80x30 filter breach | late | 2 | 3 | 2,400 | 5 | no | no | 0 | 0 | 0 | 0 | yes |
| 80x30 filter breach | late | 3 | 3 | 2,400 | 5 | no | no | 0 | 0 | 0 | 0 | yes |
| 80x30 filter breach | late | 4 | 3 | 2,400 | 5 | no | no | 0 | 0 | 0 | 0 | yes |
| 80x30 vents dust release | onset | 0 | 5 | 2,400 | 0-2 | yes, yes | yes | 0 | 0 | 0 | 0 | yes |
| 80x30 vents dust release | onset | 1 | 5 | 2,400 | 0-2 | yes, yes | yes | 0 | 0 | 0 | 0 | yes |
| 80x30 vents dust release | onset | 2 | 5 | 2,400 | 0-2 | yes, yes | yes | 0 | 0 | 0 | 0 | yes |
| 80x30 vents dust release | onset | 3 | 5 | 2,400 | 0-2 | yes, yes | yes | 0 | 0 | 0 | 0 | yes |
| 80x30 vents dust release | onset | 4 | 5 | 2,400 | 0-2 | yes, yes | yes | 0 | 0 | 0 | 0 | yes |
| 80x30 vents dust release | late | 0 | 3 | 2,400 | 2 | yes, yes | no | 0 | 0 | 0 | 0 | yes |
| 80x30 vents dust release | late | 1 | 3 | 2,400 | 2 | yes, yes | no | 0 | 0 | 0 | 0 | yes |
| 80x30 vents dust release | late | 2 | 3 | 2,400 | 2 | yes, yes | no | 0 | 0 | 0 | 0 | yes |
| 80x30 vents dust release | late | 3 | 3 | 2,400 | 2 | yes, yes | no | 0 | 0 | 0 | 0 | yes |
| 80x30 vents dust release | late | 4 | 3 | 2,400 | 2 | yes, yes | no | 0 | 0 | 0 | 0 | yes |
| 320x120 dust release | onset | 0 | 5 | 38,400 | 0-4 | no | yes | 0 | 0 | 0 | 0 | yes |
| 320x120 dust release | onset | 1 | 5 | 38,400 | 0-4 | no | yes | 0 | 0 | 0 | 0 | yes |
| 320x120 dust release | onset | 2 | 5 | 38,400 | 0-4 | no | yes | 0 | 0 | 0 | 0 | yes |
| 320x120 dust release | onset | 3 | 5 | 38,400 | 0-4 | no | yes | 0 | 0 | 0 | 0 | yes |
| 320x120 dust release | onset | 4 | 5 | 38,400 | 0-4 | no | yes | 0 | 0 | 0 | 0 | yes |
| 320x120 dust release | late | 0 | 3 | 38,400 | 8 | no | no | 0 | 0 | 0 | 0 | yes |
| 320x120 dust release | late | 1 | 3 | 38,400 | 8 | no | no | 0 | 0 | 0 | 0 | yes |
| 320x120 dust release | late | 2 | 3 | 38,400 | 8 | no | no | 0 | 0 | 0 | 0 | yes |
| 320x120 dust release | late | 3 | 3 | 38,400 | 8 | no | no | 0 | 0 | 0 | 0 | yes |
| 320x120 dust release | late | 4 | 3 | 38,400 | 8 | no | no | 0 | 0 | 0 | 0 | yes |

200 calls, 1,920,000 cell values compared, 0 with any difference; 0 raised a breach.
<!-- tables90 replay end -->

**The vents flow state (finding F2).** At 80x30 the vents room's flow state does not stop within
the base file's `max_simple_iter` of 15,000 from rest: it does stop by the error estimate and
continuity, at the outer count T1 gives, under a raised cap. The first two captures failed on
the cap (with the placement runs' `mass_imbalance_tol`, set for a 120 s run, and with the base
file's rule for this 3 s run); the scenario file raises the cap and says why. The state at
320x120 that prompt 61 solved stopped in 6,866 (`results/builder61/vents_320x120.json`).

## 4. The tests

<!-- tables90 tests begin -->
### T3. The tests

At 969dca9, under the project's flags. The parity module's new cases by test function, then the totals.

| Test | Cases | Passed | Failed | Skipped |
|---|---|---|---|---|
| `test_class_step_equals_the_reference_bit_for_bit` (planted class steps: field, budget, records, raise) | 64 | 64 | 0 | 0 |
| `test_the_planted_class_steps_reach_every_rule` (the planted inputs reach every rule) | 1 | 1 | 0 | 0 |
| `test_class_step_continues_a_restored_budget_bit_for_bit` (a budget restored mid-run) | 3 | 3 | 0 | 0 |
| `test_class_step_reads_and_sets_budgets_whole` (budgets read and set whole, restore's refusals) | 1 | 1 | 0 | 0 |
| `test_class_step_raises_the_references_breach` (the breach raised) | 4 | 4 | 0 | 0 |
| `test_class_step_breaches_at_equality_with_the_limit` (the breach at equality with the limit) | 1 | 1 | 0 | 0 |
| `test_class_step_interval_quantities_equal_the_per_step_values` (interval quantities against the per-step values) | 16 | 16 | 0 | 0 |
| `test_class_step_refuses_what_the_reference_refuses` (the refusals) | 24 | 24 | 0 | 0 |
| `test_buffered_sum_equals_np_sum_on_strided_slices` (NumPy's buffered order on strided slices) | 72 | 55 | 0 | 17 |
| the whole parity module | 3287 | 3270 | 0 | 17 |
| CTest | 91 | 91 | 0 | 0 |

New: 186 parity cases; 9 GoogleTest cases of `TransportStep` (AUniformFieldInAClosedBoxWithNoSettlingChangesByNothing, SettlingInTheSameBoxMovesTheUniformField, AUniformLineWithDiffusionChangesByNothing, AClosedBoxsBudgetClosesToRounding, OutMayBeTheFieldItself, AFieldThatIsNotFiniteIsRefusedBeforeAnythingIsBooked, AStepAboveTheStableStepIsRefusedBeforeAnythingIsBooked, ABreachIsRefusedAfterTheStepWithTheBudgetHoldingIt, ProgrammingErrorsThrow) and 1 for `held`'s length (ImplicitStep.AHeldArrayOfAnotherLengthThrows).
- The whole suite under Linux: 6371 passed, 17 skipped in 1281.06s (0:21:21).
- The whole suite under Windows: 3101 passed, 1 skipped in 1587.82s (0:26:27).
<!-- tables90 tests end -->

The Windows suite was run at 30b69f5: the one file changed since, `tests/test_core_parity.py`,
is skipped on Windows as a module, where `_cfdcore` is not built.

The parity cases (`tests/test_core_parity.py`, the section "One class step"): every mesh of
prompt 87 (the three product grids, the stretched 160x60, the obstacle-free 40x15) and its three
vent meshes, under QUICK with the UMIST limiter and under upwind, in four variants: the
configuration's own conditions on a positive field (`plain`); every input at once (planted
conditions with inflow on a third of the faces, deposition on every face kind and a surface code
drawn on every domain face, settling marked at random, sources, `v_ext` on two classes, an eddy
viscosity with zeros and a value in the SOLID cells the step must not read, and the bound
deciding every solve; `everything`); a field of both signs, whose supply can fall below zero,
where both solvers refuse the bound (`signed`); and the configuration's conditions with sources
and an eddy viscosity, the bound deciding (`sources-bound`). Three classes, three steps each,
every output compared after every step, the budgets of every class included.
`test_the_planted_class_steps_reach_every_rule` shows the planted inputs reach each rule the
port could get wrong. A budget restored mid-run, the budgets read and set whole with
`restore_budgets`' refusals, the breach on four meshes, the breach at exact equality with the
limit (the `>=`, with a limit found so that it times the supply equals the leftover), the
interval quantities on every mesh, and 24 refusals one at a time, each compared by exception
type and message with nothing booked.

The GoogleTests (`core/tests/test_transport.cpp`) need no reference: a uniform field in a closed
box with a divergence-free circulation, diffusion and no settling, on a grid whose values are
powers of two apart, comes back bit for bit after five steps under QUICK and upwind, its budget's
residual exactly zero; the same box with settling changes (the control); a uniform line at
D = 1 likewise; a closed box with circulation, settling, diffusion, deposition on every surface
kind and a source, stepped forty times, books nothing at its domain faces and closes its budget
within 1e-12 of the supply while its leftover stands far above that; `out` may be the field;
a field that is not finite (a SOLID cell's included) and a step above `stable_dt` are refused
with nothing booked; a breach is reported after the step with the budget holding it; the
programming errors throw.

## 5. The planted defects

<!-- tables90 plants begin -->
### T4. The planted defects: which tests catch each

Each built alone from the checkout's `core/` with one edit (`docs/reports/probe90/variants90.py`), then CTest, the parity module's class-step and buffered-sum cases, and the replay of every capture.

| Plant | What it changes | CTest | Parity cases failing (by test) | Replay: calls differing |
|---|---|---|---|---|
| `control` | no edit: every test passes | 100% passed, 0 tests failed out of 91 | none | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `boundary_sum_order` | a domain boundary's column summed left to right, not in NumPy's pairwise order | 100% passed, 0 tests failed out of 91 | `test_class_step_continues_a_restored_budget_bit_for_bit` 3, `test_class_step_equals_the_reference_bit_for_bit` 63, `test_class_step_raises_the_references_breach` 3 | dust80 0/40, door80 4/40, breach80 0/40, dust320 0/40, vents80 4/40 |
| `vents_into_out_swapped` | the vents' flow into the room booked as outflow and their flow out as inflow | 100% passed, 0 tests failed out of 91 | `test_class_step_continues_a_restored_budget_bit_for_bit` 1, `test_class_step_equals_the_reference_bit_for_bit` 24, `test_class_step_raises_the_references_breach` 1 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 15/40 |
| `recirculated_from_new_field` | a discharge's value from the field the step returns, not the one it starts from | 100% passed, 0 tests failed out of 91 | `test_class_step_continues_a_restored_budget_bit_for_bit` 1, `test_class_step_equals_the_reference_bit_for_bit` 23 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 5/40 |
| `deposition_wrong_surface` | each domain surface's deposition booked to the next slot (floor to ceiling, and so on) | 100% passed, 0 tests failed out of 91 | `test_class_step_breaches_at_equality_with_the_limit` 1, `test_class_step_continues_a_restored_budget_bit_for_bit` 3, `test_class_step_equals_the_reference_bit_for_bit` 60, `test_class_step_raises_the_references_breach` 4, `test_the_planted_class_steps_reach_every_rule` 1 | dust80 15/40, door80 30/40, breach80 15/40, dust320 15/40, vents80 15/40 |
| `content_with_solid` | the content summed over every cell, SOLID ones included | 100% passed, 0 tests failed out of 91 | `test_class_step_continues_a_restored_budget_bit_for_bit` 3, `test_class_step_equals_the_reference_bit_for_bit` 56, `test_class_step_raises_the_references_breach` 3, `test_the_planted_class_steps_reach_every_rule` 1 | dust80 20/40, door80 16/40, breach80 12/40, dust320 14/40, vents80 16/40 |
| `breach_strict` | the breach test's comparison strict: the leftover must exceed the limit times the supply | 100% passed, 0 tests failed out of 91 | `test_class_step_breaches_at_equality_with_the_limit` 1 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `bound_before_source` | the step's bound from the supply before the step's source is booked | 99% passed, 1 tests failed out of 91: `TransportStep.ABreachIsRefusedAfterTheStepWithTheBudgetHoldingIt` | `test_class_step_continues_a_restored_budget_bit_for_bit` 3, `test_class_step_equals_the_reference_bit_for_bit` 32, `test_the_planted_class_steps_reach_every_rule` 1 | dust80 15/40, door80 0/40, breach80 0/40, dust320 15/40, vents80 15/40 |
| `eddy_viscosity_unmasked` | the face eddy viscosity from the field as handed in, SOLID cells not zeroed | 100% passed, 0 tests failed out of 91 | `test_class_step_interval_quantities_equal_the_per_step_values` 7 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `deposition_without_area` | the deposition per face from the deposition velocity alone, without the face's area | 100% passed, 0 tests failed out of 91 | `test_class_step_breaches_at_equality_with_the_limit` 1, `test_class_step_continues_a_restored_budget_bit_for_bit` 3, `test_class_step_equals_the_reference_bit_for_bit` 60, `test_class_step_interval_quantities_equal_the_per_step_values` 16, `test_class_step_raises_the_references_breach` 4, `test_the_planted_class_steps_reach_every_rule` 1 | dust80 40/40, door80 40/40, breach80 40/40, dust320 40/40, vents80 40/40 |
| `stable_dt_from_the_air` | stable_dt from the air's face velocities, without the settling and the drift | 100% passed, 0 tests failed out of 91 | `test_class_step_equals_the_reference_bit_for_bit` 1, `test_class_step_interval_quantities_equal_the_per_step_values` 16, `test_class_step_refuses_what_the_reference_refuses` 1 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 40/40 |
| `divergence_low_first` | the divergence's further directions taken low face first, then high | 100% passed, 0 tests failed out of 91 | `test_class_step_breaches_at_equality_with_the_limit` 1, `test_class_step_continues_a_restored_budget_bit_for_bit` 3, `test_class_step_equals_the_reference_bit_for_bit` 60, `test_class_step_raises_the_references_breach` 4, `test_the_planted_class_steps_reach_every_rule` 1 | dust80 17/40, door80 24/40, breach80 24/40, dust320 18/40, vents80 16/40 |
| `obstacle_slice_flat` | the interior x faces' deposition summed pairwise over the slice, not through NumPy's buffer | 100% passed, 0 tests failed out of 91 | `test_class_step_breaches_at_equality_with_the_limit` 1, `test_class_step_continues_a_restored_budget_bit_for_bit` 1, `test_class_step_equals_the_reference_bit_for_bit` 25 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `leftover_not_booked` | the step's leftover not added to the budget | 98% passed, 2 tests failed out of 91: `TransportStep.AClosedBoxsBudgetClosesToRounding`, `TransportStep.ABreachIsRefusedAfterTheStepWithTheBudgetHoldingIt` | `test_class_step_breaches_at_equality_with_the_limit` 1, `test_class_step_continues_a_restored_budget_bit_for_bit` 3, `test_class_step_equals_the_reference_bit_for_bit` 60, `test_class_step_raises_the_references_breach` 4, `test_the_planted_class_steps_reach_every_rule` 1 | dust80 30/40, door80 30/40, breach80 30/40, dust320 30/40, vents80 30/40 |
| `drift_on_every_face` | v_ext added on every face, not only the interior faces between two non-SOLID cells | 100% passed, 0 tests failed out of 91 | `test_class_step_continues_a_restored_budget_bit_for_bit` 3, `test_class_step_equals_the_reference_bit_for_bit` 16, `test_class_step_interval_quantities_equal_the_per_step_values` 8, `test_the_planted_class_steps_reach_every_rule` 1 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `held_length_unchecked` | set_masks reads held without checking its length | 99% passed, 1 tests failed out of 91: `ImplicitStep.AHeldArrayOfAnotherLengthThrows` | none | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |

15 of 15 plants fail at least one test.
<!-- tables90 plants end -->

The prompt's eight are the first eight after the control; `eddy_viscosity_unmasked` is its
"interval quantity recomputed from the wrong input". What T4 shows beyond the counts:

- **The captured calls alone would not catch four of the plants.** `breach_strict` needs a
  leftover equal to the limit times the supply, which no run meets by chance; the parity module's
  equality case builds one. `obstacle_slice_flat` changes the obstacle deposition's last bits only
  where the interior x faces fill NumPy's buffer more than once and enough of them deposit:
  the product's few obstacle faces among the 320x120 grid's 38,280 summed alike in every
  captured call, while the parity cases, with deposition planted on a tenth of the faces, part. `drift_on_every_face`
  needs `v_ext`, which no production run passes. `eddy_viscosity_unmasked` changes no output of a
  step (below).
- **`eddy_viscosity_unmasked` is caught by prediction (b)'s comparison and by nothing else.**
  The face eddy viscosity beside a SOLID cell enters only times a conductance that is zero there,
  so the step's outputs cannot see whether the SOLID cells were masked. The first two plant runs
  (`results/builder90/plants_first.json`, `plants_second.json`) planted -1.0 and then a NaN in the
  SOLID cells, and the plant survived both: the harmonic mean is zero wherever its denominator is
  not above zero, so either value gives the masked field's face value. The cases now plant a large
  positive value (commit 969dca9), and the interval-quantity cases fail on the plant.
- **`held_length_unchecked`** fails CTest's `AHeldArrayOfAnotherLengthThrows` alone, as it should:
  no Python caller can hand the core a `held` of another length.

## 6. The timing

`probe90.py timing --run N`, three sessions one after another, each its own process, with the
machine on mains power and no other solver or test process running; each session's state is in
the second table. The processor ran without boost throughout (its performance counter at 85 to
94), Windows Defender's scanner took about a third of one processor, and the NumPy calibration
read the same in all three sessions, so the three are comparable with one another; as prompt 89
found (its F5), they are not with another day's numbers.

<!-- tables90 timing begin -->
### T5. A class step on the developed 320x120 frame: the core against `solve_timestep`

The stored 320x120 dust release's last frame (`results/builder81/product/equipment_dust.h5`) of each class, stepped under its last interval's captured conditions, faces and eddy viscosity. Medians of repeated calls in one process after warm-up calls: the reference's whole `solve_timestep`, and the core's `step` with the interval set up beforehand. The two steps were checked equal bit for bit before each was timed.

| Run | Class | Sweeps | Reference, ms | Core, ms | Reference over core |
|---|---|---|---|---|---|
| 1 | 0 | 10 | 11.979 | 1.758 | 6.81 |
| 1 | 1 | 10 | 12.912 | 1.742 | 7.41 |
| 1 | 2 | 10 | 12.726 | 1.701 | 7.48 |
| 1 | 3 | 10 | 13.545 | 1.561 | 8.68 |
| 1 | 4 | 10 | 12.639 | 1.616 | 7.82 |
| 2 | 0 | 10 | 12.177 | 1.503 | 8.10 |
| 2 | 1 | 10 | 12.372 | 1.636 | 7.56 |
| 2 | 2 | 10 | 12.603 | 1.507 | 8.36 |
| 2 | 3 | 10 | 11.739 | 1.511 | 7.77 |
| 2 | 4 | 10 | 11.651 | 1.554 | 7.50 |
| 3 | 0 | 10 | 12.392 | 1.596 | 7.77 |
| 3 | 1 | 10 | 12.410 | 1.561 | 7.95 |
| 3 | 2 | 10 | 12.544 | 1.610 | 7.79 |
| 3 | 3 | 10 | 13.278 | 1.513 | 8.77 |
| 3 | 4 | 10 | 12.885 | 1.560 | 8.26 |

| Run | Started | Calibration before and after, us | The core's interval set-up, ms | Windows before | Windows after |
|---|---|---|---|---|---|
| 1 | 2026-11-20 18:05:42 | 8.23, 8.34 | 5.12 | msedge 51; msmpeng 36; msedge 29; on mains True; processor performance 94 | msedge 61; msmpeng 32; msedge 26; on mains True; processor performance 86 |
| 2 | 2026-11-20 18:05:59 | 8.34, 8.04 | 5.01 | msmpeng 35; msedge 28; msedge 24; on mains True; processor performance 87 | msmpeng 34; msedge 23; code 21; on mains True; processor performance 85 |
| 3 | 2026-11-20 18:06:16 | 8.29, 8.34 | 4.54 | msmpeng 37; msedge 19; msedge 18; on mains True; processor performance 85 | msmpeng 37; msedge 21; msedge 18; on mains True; processor performance 85 |
<!-- tables90 timing end -->

## 7. The vectorisation report

<!-- tables90 loops begin -->
### T6. The loops GCC vectorises

`g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`, `-std=c++20 -O2 -ffp-contract=off -march=x86-64-v3 -DNDEBUG`, `-fopt-info-vec-optimized`: loops vectorised, each counted once by its place in the source (the report's lines repeat a loop for each inlined copy).

| Source | Loops vectorised | Report lines |
|---|---|---|
| `src/transport.cpp` | 5 | 19 |
| `src/scalar.cpp` | 11 | 32 |
| `src/reduce.cpp` | 2 | 6 |

The block functions of `transport.cpp` whose loop is vectorised: `copy_cells`, `deposition_cells`, `product_cells`, `update_cells`, `where_cells`.
<!-- tables90 loops end -->

`scalar.cpp`'s count is prompt 89b's (T15 of `docs/reports/probe89/tables89b.md`): the
length check added to `set_masks` is outside every loop.

## Findings

- **F1, NumPy sums a strided slice through its buffer, in whole rows.** Section 1. A port that
  sums `flux_u[:, 1:-1]` pairwise over its terms is the same computation as NumPy's where the
  slice fits one fill (80x30, 40x15) and not where it takes more (160x60, 320x120): the plant
  `obstacle_slice_flat` fails only cases on those grids, since on the others it changes nothing.
  `reduce::buffered_sum` copies the rule, and its parity test pins it. The rule depends on
  NumPy's default buffer size, so a run that changes `np.setbufsize` would part from the core on
  these grids; nothing in the project does.
- **F2, the vents room's flow state needs more than the base file's cap from rest at 80x30.**
  Section 3 and T1. It stops by the error estimate and continuity well beyond 15,000 outer
  iterations. Its residual history, read every thousand iterations, falls by about a decade per
  three thousand after the first thousand, in a staircase of two near-flat readings and a drop
  (`results/builder90/vents80_flow.json`). Any 80x30 vents run from rest needs the cap raised;
  the 320x120 state of prompt 61 did not.
- **F3, the march's set-up computes nothing per step that it could compute per interval, except
  the solve's masks.** The reference's per-step checks, `stable_dt`, velocities, fluxes,
  conductances, face eddy viscosity and deposition are formed once (section 1), each equal to the
  per-step value. The one per-step pass left that an interval fixes is the implicit solve's
  expansion of its `bool` masks (section 2).
- **F4, a class step on the developed 320x120 frame is between six and nine times faster in the
  core (T5)**, in every class and every session, each step taking the reference's ten Jacobi
  sweeps. Prediction (c) holds on run 1 (T0). How the core's time divides between the flux, the
  solve and the booking was not measured; the solve still expands its masks at every call
  (section 2), and the interval's set-up, once per interval, costs about three of the core's
  class steps.

## Deviations and limits

- **The refusals are made where the core computes their quantity.** The conditions, faces,
  `v_ext`, sources and eddy viscosity are refused by `set_interval` and `dt`'s value by
  `set_step`, not by the step that reads them. One bad input at a time, the exception and its
  message are the reference's (24 cases, section 4); with several bad at once the core can report
  another first. A wrong shape, dtype or layout is a `TypeError` where the reference raises a
  `ValueError`, as ADR-017 B says of every array the binding takes; `sources` of another dtype are
  converted to float64, as the reference's `np.asarray` converts them, and a field `c` is not.
- **The binding imports `src.solver_transport`** for `ConservationBreachError`, `MassBudget` and
  `DEPOSIT_SURFACES`, so that the exception and the budgets it returns are the reference's own
  types. The core library itself knows nothing of it. The reference logs a warning when a solve
  stops at its cap; the binding returns `converged` and logs nothing, which is step 6's adapter's
  to do.
- **`transport` is a template on D, built at 1 and 2**, the vertical direction D - 1. The
  strided-slice rule (F1) is measured on two-dimensional slices only; a three-dimensional
  reference does not exist yet (Phase 8).
- **The first vents capture merged two solvers.** The recorder told the intervals' solvers apart
  by `id()`, and a solver freed after its interval let the next one take its id. The recorder
  now holds every solver, the replay checks each call's interval inputs against their digests
  (T2's last column), and the vents run was captured again (the first capture is kept as
  `results/builder90/captures/vents80_merged.*` and not read). The 320x120 capture, taken before
  the fix, kept three solvers, and its calls pass the same check.
- **The captures' runs write frames only at the end** (`output_interval` raised), which changes
  no arithmetic of the march.
- **The vents scenario's keys.** `docs/reports/probe90/scenario_vents80.yaml` is a dust release in
  the vents file's room at 80x30, its `mass_imbalance_tol` the base file's rule for its 3 s and
  its `max_simple_iter` raised (F2).
- **No `src/` or `configs/` change.** The reference is unchanged; the prompt's stop condition, a
  sum or an expression that cannot be made bitwise without changing `src/`, was not met.

## Files

- `core/include/cfdcore/transport.hpp`, `core/src/transport.cpp`: the module.
- `core/include/cfdcore/reduce.hpp`, `core/src/reduce.cpp`: `buffered_sum`.
- `core/src/scalar.cpp`: `set_masks`' length check on `held`.
- `core/python/bindings.cpp`: `TransportSolver2`, `TransportStep2`, `buffered_sum`.
- `core/tests/test_transport.cpp` (new), `core/tests/test_scalar.cpp` (one case),
  `core/CMakeLists.txt`.
- `tests/test_core_parity.py`: the section "One class step".
- `docs/reports/probe90/`: `probe90.py` (capture, vents-flow, replay, timing, plants, vecinfo),
  `variants90.py` (the planted defects), `scenario_vents80.yaml`, `tables90.py` and `tables90.md`.
- Records in `results/builder90/` (untracked): the captures, `replay.json`, `plants.json` and
  `plants/`, `timing_1.json` to `timing_3.json`, `vecinfo.json`, `vents80_flow.json`,
  `parity.xml`, `ctest.xml`, the two suites' logs, and the run files in `runs/`.
