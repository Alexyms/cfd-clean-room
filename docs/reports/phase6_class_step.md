# Phase 6 Step 3a: One Class Step in the Core (prompt 90)

**Date:** 2026-11-20; round 1, 2026-11-21
**Prompt:** 90 (`docs/prompts/cc-prompt-90-class-step.md`); round 1, 90b (`docs/prompts/cc-prompt-90b-round1-fixes.md`)
**Branch:** `phase6/class-step`
**Status:** built. Predictions (a), (b) and (c) hold (T0). The prompt's stop condition, a sum
or an expression that cannot be made bitwise without changing `src/`, was not met: every sum is
the reference's bit for bit, one of them (a strided slice) by a NumPy rule found on the way
(F1). The vents room's flow state needs a raised iteration cap at 80x30 (F2). Review and test
90 found no defect in the port and two tests missing; round 1 (section 8) added them, made the
binding return plain data and answered the rest of their findings.

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
| (a) every captured and planted class step equals the reference's bit for bit | 200 of 200 captured calls with no differing bit in the field, the ten budget terms or the records, 40 of them with vents and a discharge recirculating (T2); 202 of 202 planted class-step parity cases pass (T3) | holds |
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
fill), those larger than any slice the budget sums skipped. Whether the rule decides a result
depends on the grid: the slice fits one fill on the 80x30 and 40x15 grids and takes more on
160x60 and 320x120, so it decides the obstacle deposition's last bits on those two only
(`obstacle_slice_flat` in T4 changes a result on them alone). The rule is NumPy's at its default
buffer size, `np.getbufsize()`, 8,192 elements, which `reduce::buffer_size` holds; a program that
changes the size changes the reference's order, and the core's no longer equals it (F1).

**The binding.** `TransportSolver2(scheme, dx_cell, dy_cell, dx_face, dy_face, cfl, upwind,
max_sweeps, tol, limit, share, horizon, schmidt, settling, diffusion, surface_codes)`;
`set_interval(conditions, u, v, v_ext=None, eddy_viscosity=None, sources=None)` takes each
class's `ConcentrationFaces` as the boundary layer builds it; `set_step(dt)`;
`step(size_class, c, out)` returns the step's records (`TransportStep2`: sweeps, converged,
leftover, bound) and may write `out` over `c`. A refusal raises the reference's exception with
its message: `ValueError` for a value, `TypeError` for a bool `dt` or `size_class` (`dt` read as
`math.isfinite` reads it, so a 0-d float array is a number; `size_class` read the same way by
every entry point that takes a class), `IndexError` for a class out of range. A breach raises
the core's own `_cfdcore.ConservationBreach`, after the step, the budget holding it as the
reference's does; it carries `size_class`, `leftover`, `supplied` and `limit`, the four
arguments of the reference's `ConservationBreachError`. A wrong shape, dtype or layout is a
`TypeError` at the call, as ADR-017 B says of every array.

**Plain data (ADR-017 B, round 1).** The binding is handed data and returns data, and imports
nothing from `src`. `budgets()` returns each class's budget as a new dict of its ten terms by
name: `initial`, `inflow`, `outflow`, `source`, `floor`, `ceiling`, `wall`, `obstacle`,
`current` and `leftover`. The four deposition names are the core's (`transport::slot_names`),
and the parity module checks them against `DEPOSIT_SURFACES`, in its order. `restore_budgets`
takes the same dicts back and refuses what it cannot read as data: a count other than the
classes', a budget that is not a dict, a missing or unknown term, a term that is not a number
(a Python bool is not one). The reference's types are made of the data on the Python side,
`tests/core_mapping.py`: `MassBudget` from the ten terms, `ConservationBreachError` from the
breach's attributes, and `restore_budgets`' refusals of a `MassBudget` (not a `MassBudget`,
other deposition keys, an unknown leftover), which concern the reference's type and are
compared with the reference's own. Step 6's `src/core_backend.py` will hold the mapping, and
the parity module and the probe share it until then.

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

The vents run's flow state, solved from rest at 80x30 with the iteration cap raised (the base file's `max_simple_iter` is 15,000), from `results/builder90/vents80_flow.json`:

| Cap raised to | Stopped by | Outer iterations | Wall time, s |
|---|---|---|---|
| 60,000 | `error_estimate_and_continuity` | 24,767 | 403 |
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

At cd2aec1, under the project's flags. The parity module's new cases by test function, then the totals.

| Test | Cases | Passed | Failed | Skipped |
|---|---|---|---|---|
| `test_class_step_equals_the_reference_bit_for_bit` (planted class steps: field, budget, records, raise) | 64 | 64 | 0 | 0 |
| `test_the_planted_class_steps_reach_every_rule` (the planted inputs reach every rule) | 1 | 1 | 0 | 0 |
| `test_class_step_continues_a_restored_budget_bit_for_bit` (a budget restored mid-run) | 3 | 3 | 0 | 0 |
| `test_class_step_hands_out_its_budgets_as_plain_data` (budgets as a dict of ten terms by name) | 1 | 1 | 0 | 0 |
| `test_class_step_restores_budgets_as_the_reference_does` (restore_budgets' refusals, through the mapping) | 1 | 1 | 0 | 0 |
| `test_class_step_refuses_budgets_it_cannot_read` (the binding's own refusals of a budget) | 1 | 1 | 0 | 0 |
| `test_class_step_raises_the_references_breach` (the breach raised) | 4 | 4 | 0 | 0 |
| `test_class_step_raises_a_breach_of_its_own` (the core's breach and its four attributes) | 1 | 1 | 0 | 0 |
| `test_class_step_breaches_at_equality_with_the_limit` (the breach at equality with the limit) | 1 | 1 | 0 | 0 |
| `test_class_step_an_empty_room_does_not_breach` (the breach test's zero guard) | 2 | 2 | 0 | 0 |
| `test_class_step_interval_quantities_equal_the_per_step_values` (interval quantities against the per-step values) | 16 | 16 | 0 | 0 |
| `test_class_step_one_solver_through_intervals_equals_the_reference` (one solver through thirteen intervals) | 4 | 4 | 0 | 0 |
| `test_class_step_recirculates_on_every_kind_of_face` (recirculating pairs on every kind of face) | 18 | 18 | 0 | 0 |
| `test_class_step_refuses_what_the_reference_refuses` (the refusals) | 24 | 24 | 0 | 0 |
| `test_class_step_dt_in_every_form_is_read_as_the_reference_reads_it` (dt in six forms) | 6 | 6 | 0 | 0 |
| `test_class_step_every_entry_point_reads_size_class_as_the_reference_does` (size_class at six entry points) | 54 | 54 | 0 | 0 |
| `test_class_step_binding_imports_nothing_from_src` (every src module unimportable) | 1 | 1 | 0 | 0 |
| `test_buffered_sum_equals_np_sum_on_strided_slices` (NumPy's buffered order on strided slices) | 72 | 55 | 0 | 17 |
| `test_buffered_sum_refuses_what_it_cannot_sum` (buffered_sum's refusals) | 1 | 1 | 0 | 0 |
| the whole parity module | 3376 | 3359 | 0 | 17 |
| CTest | 99 | 99 | 0 | 0 |

New: 275 parity cases; 11 GoogleTest cases of `TransportStep` (AUniformFieldInAClosedBoxWithNoSettlingChangesByNothing, SettlingInTheSameBoxMovesTheUniformField, AUniformLineWithDiffusionChangesByNothing, AClosedBoxsBudgetClosesToRounding, OutMayBeTheFieldItself, AFieldThatIsNotFiniteIsRefusedBeforeAnythingIsBooked, AStepAboveTheStableStepIsRefusedBeforeAnythingIsBooked, ABreachIsRefusedAfterTheStepWithTheBudgetHoldingIt, ASecondIntervalKeepsTheBudgetsAndEqualsAFreshSolverGivenThem, ASecondStepSizeInOneIntervalEqualsAFreshSolverSetToIt, ProgrammingErrorsThrow), 6 of `BufferedSum` (NoTermsSumToPositiveZeroWhateverTheRow, ARowOfZeroWithTermsIsRefused, TermsThatAreNotWholeRowsAreRefused, ARowOfOneIsThePairwiseSumOfAllTheTerms, EachFillHoldsWholeRowsAndTheFillsAreAddedInOrder, ARowLongerThanTheBufferIsAFillOfItsOwn) and 1 for `held`'s length (ImplicitStep.AHeldArrayOfAnotherLengthThrows).
- The whole suite under Linux: 6460 passed, 17 skipped in 1371.93s (0:22:51).
- The whole suite under Windows: 3101 passed, 1 skipped in 1486.82s (0:24:46).
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
| `control` | no edit: every test passes | 100% passed, 0 tests failed out of 99 | none | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `boundary_sum_order` | a domain boundary's column summed left to right, not in NumPy's pairwise order | 100% passed, 0 tests failed out of 99 | `test_class_step_continues_a_restored_budget_bit_for_bit` 3, `test_class_step_equals_the_reference_bit_for_bit` 63, `test_class_step_one_solver_through_intervals_equals_the_reference` 4, `test_class_step_raises_the_references_breach` 3, `test_class_step_recirculates_on_every_kind_of_face` 17 | dust80 0/40, door80 4/40, breach80 0/40, dust320 0/40, vents80 4/40 |
| `vents_into_out_swapped` | the vents' flow into the room booked as outflow and their flow out as inflow | 100% passed, 0 tests failed out of 99 | `test_class_step_continues_a_restored_budget_bit_for_bit` 1, `test_class_step_equals_the_reference_bit_for_bit` 24, `test_class_step_one_solver_through_intervals_equals_the_reference` 4, `test_class_step_raises_the_references_breach` 1, `test_class_step_recirculates_on_every_kind_of_face` 18 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 15/40 |
| `recirculated_from_new_field` | a discharge's value from the field the step returns, not the one it starts from | 100% passed, 0 tests failed out of 99 | `test_class_step_continues_a_restored_budget_bit_for_bit` 1, `test_class_step_equals_the_reference_bit_for_bit` 23, `test_class_step_one_solver_through_intervals_equals_the_reference` 4, `test_class_step_recirculates_on_every_kind_of_face` 18 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 5/40 |
| `deposition_wrong_surface` | each domain surface's deposition booked to the next slot (floor to ceiling, and so on) | 100% passed, 0 tests failed out of 99 | `test_class_step_breaches_at_equality_with_the_limit` 1, `test_class_step_continues_a_restored_budget_bit_for_bit` 3, `test_class_step_dt_in_every_form_is_read_as_the_reference_reads_it` 3, `test_class_step_equals_the_reference_bit_for_bit` 60, `test_class_step_hands_out_its_budgets_as_plain_data` 1, `test_class_step_one_solver_through_intervals_equals_the_reference` 4, `test_class_step_raises_the_references_breach` 4, `test_class_step_recirculates_on_every_kind_of_face` 18, `test_class_step_refuses_budgets_it_cannot_read` 1, `test_class_step_restores_budgets_as_the_reference_does` 1, `test_the_planted_class_steps_reach_every_rule` 1 | dust80 15/40, door80 30/40, breach80 15/40, dust320 15/40, vents80 15/40 |
| `content_with_solid` | the content summed over every cell, SOLID ones included | 100% passed, 0 tests failed out of 99 | `test_class_step_continues_a_restored_budget_bit_for_bit` 3, `test_class_step_equals_the_reference_bit_for_bit` 56, `test_class_step_one_solver_through_intervals_equals_the_reference` 4, `test_class_step_raises_a_breach_of_its_own` 1, `test_class_step_raises_the_references_breach` 3, `test_class_step_recirculates_on_every_kind_of_face` 18, `test_the_planted_class_steps_reach_every_rule` 1 | dust80 20/40, door80 16/40, breach80 12/40, dust320 14/40, vents80 16/40 |
| `breach_strict` | the breach test's comparison strict: the leftover must exceed the limit times the supply | 100% passed, 0 tests failed out of 99 | `test_class_step_breaches_at_equality_with_the_limit` 1 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `bound_before_source` | the step's bound from the supply before the step's source is booked | 99% passed, 1 tests failed out of 99: `TransportStep.ABreachIsRefusedAfterTheStepWithTheBudgetHoldingIt` | `test_class_step_continues_a_restored_budget_bit_for_bit` 3, `test_class_step_equals_the_reference_bit_for_bit` 32, `test_class_step_one_solver_through_intervals_equals_the_reference` 4, `test_the_planted_class_steps_reach_every_rule` 1 | dust80 15/40, door80 0/40, breach80 0/40, dust320 15/40, vents80 15/40 |
| `eddy_viscosity_unmasked` | the face eddy viscosity from the field as handed in, SOLID cells not zeroed | 100% passed, 0 tests failed out of 99 | `test_class_step_interval_quantities_equal_the_per_step_values` 7, `test_class_step_one_solver_through_intervals_equals_the_reference` 4 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `deposition_without_area` | the deposition per face from the deposition velocity alone, without the face's area | 100% passed, 0 tests failed out of 99 | `test_class_step_breaches_at_equality_with_the_limit` 1, `test_class_step_continues_a_restored_budget_bit_for_bit` 3, `test_class_step_dt_in_every_form_is_read_as_the_reference_reads_it` 3, `test_class_step_equals_the_reference_bit_for_bit` 60, `test_class_step_hands_out_its_budgets_as_plain_data` 1, `test_class_step_interval_quantities_equal_the_per_step_values` 16, `test_class_step_one_solver_through_intervals_equals_the_reference` 4, `test_class_step_raises_a_breach_of_its_own` 1, `test_class_step_raises_the_references_breach` 4, `test_class_step_recirculates_on_every_kind_of_face` 18, `test_class_step_refuses_budgets_it_cannot_read` 1, `test_class_step_restores_budgets_as_the_reference_does` 1, `test_the_planted_class_steps_reach_every_rule` 1 | dust80 40/40, door80 40/40, breach80 40/40, dust320 40/40, vents80 40/40 |
| `stable_dt_from_the_air` | stable_dt from the air's face velocities, without the settling and the drift | 100% passed, 0 tests failed out of 99 | `test_class_step_equals_the_reference_bit_for_bit` 1, `test_class_step_interval_quantities_equal_the_per_step_values` 16, `test_class_step_one_solver_through_intervals_equals_the_reference` 4, `test_class_step_recirculates_on_every_kind_of_face` 1, `test_class_step_refuses_what_the_reference_refuses` 1 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 40/40 |
| `divergence_low_first` | the divergence's further directions taken low face first, then high | 100% passed, 0 tests failed out of 99 | `test_class_step_breaches_at_equality_with_the_limit` 1, `test_class_step_continues_a_restored_budget_bit_for_bit` 3, `test_class_step_dt_in_every_form_is_read_as_the_reference_reads_it` 3, `test_class_step_equals_the_reference_bit_for_bit` 60, `test_class_step_hands_out_its_budgets_as_plain_data` 1, `test_class_step_one_solver_through_intervals_equals_the_reference` 4, `test_class_step_raises_a_breach_of_its_own` 1, `test_class_step_raises_the_references_breach` 4, `test_class_step_recirculates_on_every_kind_of_face` 18, `test_class_step_refuses_budgets_it_cannot_read` 1, `test_class_step_restores_budgets_as_the_reference_does` 1, `test_the_planted_class_steps_reach_every_rule` 1 | dust80 17/40, door80 24/40, breach80 24/40, dust320 18/40, vents80 16/40 |
| `obstacle_slice_flat` | the interior x faces' deposition summed pairwise over the slice, not through NumPy's buffer | 100% passed, 0 tests failed out of 99 | `test_class_step_breaches_at_equality_with_the_limit` 1, `test_class_step_continues_a_restored_budget_bit_for_bit` 1, `test_class_step_equals_the_reference_bit_for_bit` 25, `test_class_step_one_solver_through_intervals_equals_the_reference` 1 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `leftover_not_booked` | the step's leftover not added to the budget | 98% passed, 2 tests failed out of 99: `TransportStep.AClosedBoxsBudgetClosesToRounding`, `TransportStep.ABreachIsRefusedAfterTheStepWithTheBudgetHoldingIt` | `test_class_step_binding_imports_nothing_from_src` 1, `test_class_step_breaches_at_equality_with_the_limit` 1, `test_class_step_continues_a_restored_budget_bit_for_bit` 3, `test_class_step_dt_in_every_form_is_read_as_the_reference_reads_it` 3, `test_class_step_equals_the_reference_bit_for_bit` 60, `test_class_step_hands_out_its_budgets_as_plain_data` 1, `test_class_step_one_solver_through_intervals_equals_the_reference` 4, `test_class_step_raises_a_breach_of_its_own` 1, `test_class_step_raises_the_references_breach` 4, `test_class_step_recirculates_on_every_kind_of_face` 18, `test_class_step_refuses_budgets_it_cannot_read` 1, `test_class_step_restores_budgets_as_the_reference_does` 1, `test_the_planted_class_steps_reach_every_rule` 1 | dust80 30/40, door80 30/40, breach80 30/40, dust320 30/40, vents80 30/40 |
| `drift_on_every_face` | v_ext added on every face, not only the interior faces between two non-SOLID cells | 100% passed, 0 tests failed out of 99 | `test_class_step_continues_a_restored_budget_bit_for_bit` 3, `test_class_step_equals_the_reference_bit_for_bit` 16, `test_class_step_interval_quantities_equal_the_per_step_values` 8, `test_class_step_one_solver_through_intervals_equals_the_reference` 4, `test_the_planted_class_steps_reach_every_rule` 1 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `held_length_unchecked` | set_masks reads held without checking its length | 99% passed, 1 tests failed out of 99: `ImplicitStep.AHeldArrayOfAnotherLengthThrows` | none | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `deposit_step_stale` | a second set_step keeps the first step's deposition per face | 99% passed, 1 tests failed out of 99: `TransportStep.ASecondStepSizeInOneIntervalEqualsAFreshSolverSetToIt` | `test_class_step_one_solver_through_intervals_equals_the_reference` 4 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `source_kept_across_intervals` | a class's source of an earlier interval kept when the next has none | 99% passed, 1 tests failed out of 99: `TransportStep.ASecondIntervalKeepsTheBudgetsAndEqualsAFreshSolverGivenThem` | `test_class_step_one_solver_through_intervals_equals_the_reference` 4 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `budgets_zeroed_by_interval` | set_interval zeroes the budgets | 99% passed, 1 tests failed out of 99: `TransportStep.ASecondIntervalKeepsTheBudgetsAndEqualsAFreshSolverGivenThem` | `test_class_step_one_solver_through_intervals_equals_the_reference` 4 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `interval_leaves_stepped_set` | set_interval leaves the last interval's step in force, so a step runs on zeroed per-step arrays | 99% passed, 1 tests failed out of 99: `TransportStep.ASecondIntervalKeepsTheBudgetsAndEqualsAFreshSolverGivenThem` | `test_class_step_one_solver_through_intervals_equals_the_reference` 4 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `interval_keeps_old_vents` | a second interval without vents keeps the first interval's vent faces | 100% passed, 0 tests failed out of 99 | `test_class_step_one_solver_through_intervals_equals_the_reference` 4 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `inflow_kept_across_intervals` | the first interval's inflow values kept by the next | 100% passed, 0 tests failed out of 99 | `test_class_step_one_solver_through_intervals_equals_the_reference` 4 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `eddy_kept_across_intervals` | the face eddy viscosity of an earlier interval kept on a laminar one | 100% passed, 0 tests failed out of 99 | `test_class_step_one_solver_through_intervals_equals_the_reference` 4 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `recirculated_low_cell_only` | the room value of an intake face read from its low side only | 100% passed, 0 tests failed out of 99 | `test_class_step_one_solver_through_intervals_equals_the_reference` 4, `test_class_step_recirculates_on_every_kind_of_face` 8 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `recirculated_high_cell_only` | the room value of an intake face read from its high side only | 100% passed, 0 tests failed out of 99 | `test_class_step_continues_a_restored_budget_bit_for_bit` 1, `test_class_step_equals_the_reference_bit_for_bit` 23, `test_class_step_one_solver_through_intervals_equals_the_reference` 4, `test_class_step_recirculates_on_every_kind_of_face` 8 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 15/40 |
| `recirculated_passes_all` | the fraction a discharge passes dropped: it carries the whole drawn value | 100% passed, 0 tests failed out of 99 | `test_class_step_one_solver_through_intervals_equals_the_reference` 4, `test_class_step_recirculates_on_every_kind_of_face` 6 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `drawn_zero_divides` | the guard for an intake that draws nothing broken: zero is divided by zero | 100% passed, 0 tests failed out of 99 | `test_class_step_recirculates_on_every_kind_of_face` 4 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `recirculated_y_dropped` | the intake's y faces left out of the weighted sum, though its drawn flux counts them | 100% passed, 0 tests failed out of 99 | `test_class_step_recirculates_on_every_kind_of_face` 12 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `drawn_ignores_y` | the intake's drawn flux without its y faces | 100% passed, 0 tests failed out of 99 | `test_class_step_recirculates_on_every_kind_of_face` 12 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `breach_without_zero_guard` | the breach test without its leftover != 0 clause: an empty room breaches at its first step | 100% passed, 0 tests failed out of 99 | `test_class_step_an_empty_room_does_not_breach` 2 | dust80 10/40, door80 10/40, breach80 10/40, dust320 10/40, vents80 10/40 |
| `binding_imports_the_reference` | budgets() imports the reference's module, as the first binding did to make a MassBudget | 100% passed, 0 tests failed out of 99 | `test_class_step_binding_imports_nothing_from_src` 1 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `budgets_slots_reversed` | the four deposition terms handed out in the reverse order of the slots | 100% passed, 0 tests failed out of 99 | `test_class_step_breaches_at_equality_with_the_limit` 1, `test_class_step_continues_a_restored_budget_bit_for_bit` 3, `test_class_step_dt_in_every_form_is_read_as_the_reference_reads_it` 3, `test_class_step_equals_the_reference_bit_for_bit` 60, `test_class_step_hands_out_its_budgets_as_plain_data` 1, `test_class_step_one_solver_through_intervals_equals_the_reference` 4, `test_class_step_raises_the_references_breach` 4, `test_class_step_recirculates_on_every_kind_of_face` 18, `test_class_step_refuses_budgets_it_cannot_read` 1, `test_class_step_restores_budgets_as_the_reference_does` 1, `test_the_planted_class_steps_reach_every_rule` 1 | dust80 30/40, door80 30/40, breach80 15/40, dust320 30/40, vents80 30/40 |
| `restore_drops_current` | restore_budgets does not set the current content | 100% passed, 0 tests failed out of 99 | `test_class_step_continues_a_restored_budget_bit_for_bit` 3, `test_class_step_refuses_budgets_it_cannot_read` 1 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `restore_drops_outflow` | restore_budgets does not set the outflow | 100% passed, 0 tests failed out of 99 | `test_class_step_continues_a_restored_budget_bit_for_bit` 3 | dust80 15/40, door80 20/40, breach80 15/40, dust320 15/40, vents80 15/40 |
| `restore_accepts_bool_term` | restore_budgets reads a Python bool as a number | 100% passed, 0 tests failed out of 99 | `test_class_step_refuses_budgets_it_cannot_read` 1 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `restore_accepts_unknown_term` | restore_budgets takes a budget with a term it does not know, beside the ten | 100% passed, 0 tests failed out of 99 | `test_class_step_refuses_budgets_it_cannot_read` 1 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `breach_leftover_is_supplied` | the breach's leftover attribute carries the particles supplied | 100% passed, 0 tests failed out of 99 | `test_class_step_breaches_at_equality_with_the_limit` 1, `test_class_step_raises_a_breach_of_its_own` 1, `test_class_step_raises_the_references_breach` 4 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `record_bound_is_leftover` | the step's record gives the leftover as its bound | 100% passed, 0 tests failed out of 99 | `test_class_step_continues_a_restored_budget_bit_for_bit` 3, `test_class_step_equals_the_reference_bit_for_bit` 60, `test_class_step_hands_out_its_budgets_as_plain_data` 1, `test_class_step_one_solver_through_intervals_equals_the_reference` 4, `test_class_step_recirculates_on_every_kind_of_face` 18, `test_class_step_refuses_budgets_it_cannot_read` 1, `test_class_step_restores_budgets_as_the_reference_does` 1, `test_the_planted_class_steps_reach_every_rule` 1 | dust80 30/40, door80 30/40, breach80 30/40, dust320 30/40, vents80 30/40 |
| `dt_python_bool_unchecked` | a Python bool dt read as a number, 1.0 | 100% passed, 0 tests failed out of 99 | `test_class_step_dt_in_every_form_is_read_as_the_reference_reads_it` 1 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `dt_numpy_bool_unchecked` | a NumPy bool dt read as a number, 1.0 | 100% passed, 0 tests failed out of 99 | `test_class_step_dt_in_every_form_is_read_as_the_reference_reads_it` 1, `test_class_step_refuses_what_the_reference_refuses` 1 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `dt_bool_array_unchecked` | a 0-d bool array dt read as a number, 1.0 | 100% passed, 0 tests failed out of 99 | `test_class_step_dt_in_every_form_is_read_as_the_reference_reads_it` 1 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `dt_any_array_refused_as_bool` | every ndarray dt refused as a bool, a 0-d float array included (test 90 S4) | 100% passed, 0 tests failed out of 99 | `test_class_step_dt_in_every_form_is_read_as_the_reference_reads_it` 1 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `stable_dt_class_unchecked` | stable_dt takes any class that converts to an index, a bool included (review 90 S4) | 100% passed, 0 tests failed out of 99 | `test_class_step_every_entry_point_reads_size_class_as_the_reference_does` 9 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `buffered_sum_partial_rows_allowed` | buffered_sum takes terms that are not a whole number of rows | 99% passed, 1 tests failed out of 99: `BufferedSum.TermsThatAreNotWholeRowsAreRefused` | `test_buffered_sum_refuses_what_it_cannot_sum` 1 | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |
| `buffered_sum_zero_row_unchecked` | buffered_sum divides by a row of zero | 99% passed, 1 tests failed out of 99: `BufferedSum.ARowOfZeroWithTermsIsRefused` | none | dust80 0/40, door80 0/40, breach80 0/40, dust320 0/40, vents80 0/40 |

44 of 44 plants fail at least one test.
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
machine on mains power and no other solver or test process of the project running; each
session's state is in the second table. The load seen at run 1 was Edge and Windows Defender's
scanner, both busy (the table's "Windows before" and "after"), where the prompt asked for
nothing heavy running. A ratio is taken within one session, so a load falls on both of its
sides, and the calibration's ratios in the table show the three sessions comparable with one
another. As prompt 89 found (its F5), they are not with another day's numbers, so the report
gives ratios only and the times themselves stay in the records
(`results/builder90/timing_N.json`).

<!-- tables90 timing begin -->
### T5. A class step on the developed 320x120 frame: the core against `solve_timestep`

The stored 320x120 dust release's last frame (`results/builder81/product/equipment_dust.h5`) of each class, stepped under its last interval's captured conditions, faces and eddy viscosity. Medians of repeated calls in one process after warm-up calls: the reference's whole `solve_timestep`, and the core's `step` with the interval set up beforehand. The two steps were checked equal bit for bit before each was timed. Only ratios taken within one session are given: the times themselves move with the machine's state from day to day (prompt 89, F5) and stay in `results/builder90/timing_N.json`.

| Run | Class | Sweeps | Reference over core |
|---|---|---|---|
| 1 | 0 | 10 | 6.81 |
| 1 | 1 | 10 | 7.41 |
| 1 | 2 | 10 | 7.48 |
| 1 | 3 | 10 | 8.68 |
| 1 | 4 | 10 | 7.82 |
| 2 | 0 | 10 | 8.10 |
| 2 | 1 | 10 | 7.56 |
| 2 | 2 | 10 | 8.36 |
| 2 | 3 | 10 | 7.77 |
| 2 | 4 | 10 | 7.50 |
| 3 | 0 | 10 | 7.77 |
| 3 | 1 | 10 | 7.95 |
| 3 | 2 | 10 | 7.79 |
| 3 | 3 | 10 | 8.77 |
| 3 | 4 | 10 | 8.26 |

The session's state: the interval set-up (`set_interval`) in the core's class steps of the same session, the NumPy calibration's median over the first session's and, within the session, after over before, and what Windows was running before and after.

| Run | Started | Interval set-up, in class steps | Calibration over run 1's | Calibration, after over before | Windows before | Windows after |
|---|---|---|---|---|---|---|
| 1 | 2026-11-20 18:05:42 | 3.0 | 1.00 | 1.01 | msedge 51; msmpeng 36; msedge 29; on mains True; processor performance 94 | msedge 61; msmpeng 32; msedge 26; on mains True; processor performance 86 |
| 2 | 2026-11-20 18:05:59 | 3.3 | 0.99 | 0.96 | msmpeng 35; msedge 28; msedge 24; on mains True; processor performance 87 | msmpeng 34; msedge 23; code 21; on mains True; processor performance 85 |
| 3 | 2026-11-20 18:06:16 | 2.9 | 1.00 | 1.01 | msmpeng 37; msedge 19; msedge 18; on mains True; processor performance 85 | msmpeng 37; msedge 21; msedge 18; on mains True; processor performance 85 |
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

## 8. Round 1: review 90 and test 90

Review 90 approved the branch with eight Suggestions. Test 90 found no Critical and no defect in
the port, which it stepped through its own differential fuzz without a differing bit, and two
Bugs, both tests missing for rules the code has right (B1, B2), with seven Suggestions. Prompt
90b is the fix pass. T3 and T4 list every case and every plant by name, taken at the round's last
commit that changes code or tests.

Every new case passed on the code as reviewed, before any change to it. The two changes to the
binding that fix a behaviour (`dt` and `size_class`) were made after cases that failed on them.

| Finding | What was done |
|---|---|
| Test B1, review S1: nothing took one solver through a second interval or a second step size | `test_class_step_one_solver_through_intervals_equals_the_reference` takes one core solver through thirteen intervals, on two vents meshes and both schemes, the reference rebuilt for each with the budgets carried: a source that starts and stops, a flow switch, an eddy viscosity that starts, changes and stops, other steps inside an interval, a drift, other conditions, the vents taken away, other vents opened and the first vents returned. Fields, budgets and records are compared after every step and the interval's quantities after every step size, and a step before the interval's own `set_step` must be refused. Two GoogleTests check a second interval and a second step size against a fresh solver handed the budgets. The tester's five plants and review S1's three are in T4, each caught. |
| Test B2: the recirculation pinned in one orientation, with one fraction, on an intake that always draws | `test_class_step_recirculates_on_every_kind_of_face` plants nine cases on the vents room with a shelf, a SOLID block clear of the floor, so that a y face has the SOLID cell on its high side as well as its low: an intake on x faces and on y faces with the SOLID cell on either side, both directions in one intake, fractions other than one for each class, an intake that draws nothing, two pairs at once, and one of two drawing nothing. The cases assert that the planted inputs reach what they are planted for. The three plants the prompt names and three beside them (the high side alone, the y faces dropped from the weighted sum and from the drawn flux) are in T4, each caught. |
| Review S2: `buffered_sum`'s refusals | Six GoogleTests (no terms, a row of 0, terms that are not whole rows, a row of 1, fills of whole rows, a row longer than the buffer) and a parity case. The binding no longer repeats the core's refusals, so the parity case tests them through nanobind's `ValueError`. |
| Review S3, test S4: the forms of `dt` | One parity case per form, compared with the reference's outcome and message. The binding refuses an array as a bool by its dtype kind, so a 0-d float array steps. Plants: each of the three bool clauses removed, and every array refused. |
| Review S4: `size_class` guarded on `step` only | Every entry point reads the class through `read_class`; a parity case per entry point and bad class, compared with the reference's. Plant: `stable_dt` unchecked. |
| Test S1: the breach test's zero guard pinned by the replay only | `test_class_step_an_empty_room_does_not_breach`, a zero field with nothing supplied, and its plant. |
| Section 3 of the prompt: the binding knows nothing of the reference | Plain data and `_cfdcore.ConservationBreach` (section 1), `tests/core_mapping.py`, a case that runs a step, a restore and a breach with every `src` module unimportable, and eight plants. Every comparison of the earlier cases is unchanged and bit for bit. |
| Review S5 | The cascade and stability cells of `docs/SYSTEM.md` name what the binding reads (`ConcentrationFaces`, `Recirculation`, a drift's `u` and `v`) and what makes the reference's types of its data. |
| Review S6 | `_refusal_inputs` and `raise_refusal`'s unused `limit` removed. |
| Review S7, S8 | T5 gives ratios taken within a session, and section 6 states the load seen at run 1. Prose that restated a number points to a table, and T1's note on the vents flow state is a table. |
| F1's limit, F2's record | F1 and section 1 state the default buffer size; the deferred table of `docs/PROJECT_PLAN.md` lists the vents room's flow state beside the two already there. |

What the plants showed: `recirculated_high_cell_only` was caught by cases the branch already had,
since the vents file's intake has the SOLID cell on its high side and so its room cell on the
low, and `recirculated_low_cell_only` by none of them. The earlier cases could tell one direction
of the mistake and not the other, which is what the new cases are for.

## Findings

- **F1, NumPy sums a strided slice through its buffer, in whole rows.** Section 1. A port that
  sums `flux_u[:, 1:-1]` pairwise over its terms is the same computation as NumPy's where the
  slice fits one fill (80x30, 40x15) and not where it takes more (160x60, 320x120): the plant
  `obstacle_slice_flat` fails only cases on those grids, since on the others it changes nothing.
  `reduce::buffered_sum` copies the rule, and its parity test pins it. The limit: it copies
  NumPy's order at the default buffer size, `np.getbufsize()` (8,192). A program that changes
  the size changes the reference's order, and the core's no longer equals it on these grids;
  nothing in the project does.
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
- **F4, a class step on the developed 320x120 frame is faster in the core in every class and
  every session (T5)**, each step taking the reference's ten Jacobi sweeps. Prediction (c) holds
  on run 1 (T0). How the core's time divides between the flux, the solve and the booking was not
  measured; the solve still expands its masks at every call (section 2), and the interval's
  set-up, once per interval, is given in class steps in T5.

## Deviations and limits

- **The refusals are made where the core computes their quantity.** The conditions, faces,
  `v_ext`, sources and eddy viscosity are refused by `set_interval` and `dt`'s value by
  `set_step`, not by the step that reads them. One bad input at a time, the exception and its
  message are the reference's (24 cases, section 4); with several bad at once the core can report
  another first. A wrong shape, dtype or layout is a `TypeError` where the reference raises a
  `ValueError`, as ADR-017 B says of every array the binding takes; `sources` of another dtype are
  converted to float64, as the reference's `np.asarray` converts them, and a field `c` is not.
- **The binding imports nothing from `src`** (round 1; the first build imported
  `src.solver_transport` for the exception and the budget type). It returns data and raises its
  own exception, and `tests/core_mapping.py` makes the reference's types of them (section 1). The
  refusals of `restore_budgets` that concern `MassBudget` are the mapping's now, not the
  binding's, and the parity module compares them with the reference's through the mapping. The
  reference logs a warning when a solve stops at its cap; the binding returns `converged` and
  logs nothing, which is step 6's adapter's to do.
- **What round 1 left as it was.** Test 90's S2, S3, S5 (the `dt` of an int zero is reported as
  a float in the binding's message), S6 and S7 are Suggestions the prompt did not take up. A
  class too large for a long long is an `IndexError` as the reference's, since `read_class` is
  read for every entry point.
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
- `core/python/bindings.cpp`: `TransportSolver2`, `TransportStep2`, `buffered_sum`, and
  `ConservationBreach`.
- `core/tests/test_transport.cpp` (new), `core/tests/test_scalar.cpp` (one case),
  `core/CMakeLists.txt`.
- `tests/test_core_parity.py`: the section "One class step"; `tests/core_mapping.py`: the core's
  plain data made into the reference's types.
- `docs/reports/probe90/`: `probe90.py` (capture, vents-flow, replay, timing, plants, vecinfo),
  `variants90.py` (the planted defects, round 1's included), `scenario_vents80.yaml`,
  `tables90.py` and `tables90.md`.
- Records in `results/builder90/` (untracked): the captures, `replay.json`, `plants.json` and
  `plants/`, `timing_1.json` to `timing_3.json`, `vecinfo.json`, `vents80_flow.json`,
  `parity.xml`, `ctest.xml`, the two suites' logs, and the run files in `runs/`; the records
  that round 1 replaced are in `results/builder90/round0/`, and round 1's scripts and logs in
  `results/builder90b/`.
