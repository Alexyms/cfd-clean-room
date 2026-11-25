# Phase 6 Step 5a: The Turbulence Model and the Wall Functions (prompt 94)

**Date:** 2026-11-25
**Prompt:** 94 (`docs/prompts/cc-prompt-94-turbulence.md`)
**Branch:** `phase6/turbulence`
**Status:** built. Predictions (b) and (c) hold; (a) fails as stated, on blocks whose logarithm or cube
NumPy forms with its own AVX-512 loops on this machine, and holds where NumPy calls the C library; (d)
fails (T0). A stop condition was met before the port and Alex decided it on 2026-11-25 ("The result"
below).

This is the first half of Phase 6 step 5 (`docs/PROJECT_PLAN.md`, Phase 6, the steps table;
ADR-012 A to D as built and amended; ADR-014 J; ADR-017 C, F and G): `KEpsilonModel.step` on the
steady path and `TurbulenceBoundary`, in the core's `turbulence` and `walls` modules. Nothing under
`src/` or `configs/` changes. The true-time step of VAL-015 (a float `dt`) is tier 2's and stays on
the reference (ADR-017 F); the core refuses it.

## The predictions

Set by the orchestrator in prompt 94 on 2026-11-25, and committed alone before any run.

- (a) Every block of the first kind equals the reference bit for bit on every planted and captured
  input, the UMIST advection of k and eps included.
- (b) Every captured CG solve of k and eps stops at the reference's count, its field within 1e-12
  of its largest magnitude, and no captured step falls back where the reference does not.
- (c) The D = 3 instantiation builds and its tests pass.
- (d) A turbulence step with its conditions on the 320x120 state is at least five times faster than
  the reference's, same session.

## How they are scored (the builder, before any run)

The core is built under the project's flags (`CLAUDE.md`, C++ Coding Standards) and called
through its binding under WSL2 with the Linux environment, one BLAS thread. "The reference" is
`KEpsilonModel` and `TurbulenceBoundary` at `origin/main` (1bca2d8), unchanged, as this
environment's NumPy 2.4.4 runs them.

- **(a).** The blocks of the first kind are everything elementwise: in `turbulence` the eddy
  viscosity, the strain squared and the corner gradients it averages, the production, RNG's R, the
  two growths and the two decays, the pseudo-time step, each quantity's advected field and its
  explicit field, and the two pairs of conductances; in `walls` the wall viscosity on both
  components' wall stencils, the held eps and the given production, the four inflow arrays and the
  four edges' tangential velocities; and the set-up's fixed arrays and scalars (the wall faces and
  cells, the inlet values, the initial values, `log_law_floor`). Holds when each equals the
  reference's in every bit (a float compared by its 64-bit pattern, so that 0.0 and -0.0 differ) on
  every planted case of the parity module and every captured call. One differing bit is a failure,
  whatever its cause.
- **(b).** For each captured CG solve, of k and of eps, the core's solve of the captured system stops
  at the reference's count, and the largest difference of the field over the reference's largest
  magnitude is at most 1e-12; and no solve of the core's falls back where the reference's did not. A
  count that differs is reported apart from a field that differs. The whole step's own differences
  (k, eps and nu_t, each over its largest magnitude) are reported beside it and not scored
  (REQ-N03 as amended 2026-11-25).
- **(c).** The GoogleTest program of three directions (`core/tests/test_three_directions.cpp`)
  compiles with `turbulence` and `walls` instantiated at D = 3 and every test of it passes.
- **(d).** On the stored 320x120 product state (`results/builder81/product/equipment_dust.h5`), the
  arguments of the first outer iteration of a solve warm-started from it: the median of repeated
  `conditions` then `step` calls of the reference over the median of repeated core calls through the
  binding doing the same, both in one process, after unrecorded warm-up calls. Three sessions, each
  recording the machine's state as prompt 90's probe records it. Session 1 is scored; 2 and 3 are
  reported beside it. The CG solves are timed apart and reported. Holds when the ratio is at least
  5.0 in session 1.

A failed prediction is a finding, not a stop.

## The result

<!-- tables94 predictions begin -->
### T0. The predictions, scored by the rule committed with them

| Prediction | Measured | Verdict |
|---|---|---|
| (a) every block of the first kind equals the reference bit for bit on every planted and captured input | as NumPy runs by default on this machine: 2 blocks of 16 differ on the captured steps (the held cells and their eps (cube), the given production (log); 1,225 values, T3b), and the planted cases of a block that calls one of the two functions are skipped (T1; what it does to the functions is T6). Where NumPy calls the C library's log and pow (Alex, 2026-11-25): 0 of 800 block comparisons of the captured steps differ (T3), and 415 of 415 planted cases pass (T1) | fails as stated; holds where NumPy calls the C library |
| (b) every captured CG solve at the reference's count, its field within 1e-12, no fallback on one side | 100 solves: 0 counts differ, 0 fields beyond 1e-12, 0 fallbacks on one side only (T4) | holds |
| (c) the D = 3 instantiation builds and its tests pass | the program of three directions builds; 14 of 14 of its tests pass, 7 of 7 of them this step's (T1b) | holds |
| (d) a step with its conditions on the 320x120 state at least five times faster | 4.17 times in session 1 (session 2: 3.93; session 3: 4.03); the core's two calls in one: 4.50, 4.12, 4.38; with NumPy's AVX-512 loops off: 4.34, 4.33, 4.24, in one call 4.64, 4.55, 4.43 (T8) | fails |
<!-- tables94 predictions end -->

The tables are generated by `docs/reports/probe94/tables94.py` from the records in
`results/builder94/` (untracked, this machine) and copied here between markers;
`tables94.py --check` regenerates them and compares, and fails if a marker pair is missing. The
full set is `docs/reports/probe94/tables94.md`.

**A stop condition was met before the port, and Alex decided it (2026-11-25).** The prompt says to stop
at "a block of the first kind that cannot be made bitwise without changing `src/`". Three blocks are
such as the reference runs on this machine: the wall viscosity (`np.log`), the wall cells' held eps and
production (a cube and `np.log`) and RNG's R (a cube). On a processor with AVX-512, NumPy 2.4.4 forms
both functions with loops of its own, the cube's an assembly routine of Intel's SVML, and they part from
the C library's in the last bit on some values (T6). Where those loops are off
(`NPY_DISABLE_CPU_FEATURES=X86_V4`) NumPy calls the C library, as it does on a processor without
AVX-512. The work stopped there with the measurement, and Alex chose between calling the C library and
comparing at that level, judging the three blocks by 1e-12, porting NumPy's loops, and ending the pass.
His answer: the core calls the C library's `log` and `pow`; the parity tests and the captures of these
blocks run the reference with NumPy's AVX-512 loops off, the x86-64-v3 level the core itself builds for,
and require equality in every bit; and what the loops change is measured and reported. Prediction (a) is
scored by the rule committed before any run, as the reference runs by default, beside its score at that
level (T0). The decision has no place in ADR-017 yet: section G speaks of sums, not of elementary
functions (F1).

## 1. The port

`core/include/cfdcore/turbulence.hpp` and `walls.hpp`, with a source each; `elementary.hpp`, the C
library's `log` and `pow` behind two calls; and `positions.hpp`, the positions in a layout the two
modules share.

**Per direction.** Component c of the velocity lies on the faces normal to direction c. A
`turbulence::model<D>` holds its strain as a term for each direction and a shear for each pair of
directions; what makes a shear, the gradient at the pair's corners of component c along direction t, is
`gradient_rows<C, T>`, one form for each ordered pair, with the direction along the rows or across them.
A `walls::boundary<D>` holds a wall stencil for each ordered pair and a wall cell's walls by direction
and side. The slip along a wall is the magnitude of the D - 1 components along it, which for D = 2 is the
reference's absolute value; that is the one place where the count of directions is asked. A face's area
and a cell's volume are products over the other, and over all, directions, so a third direction's width
is in them already. The GoogleTest program of three directions instantiates both modules at 3
(`core/tests/test_three_directions.cpp`): 2D rooms of random widths with SOLID cells, walls, vents and
edges, extruded along each direction, have the 2D rooms' conditions, wall viscosity, terms, explicit
fields and conductances bit for bit on every plane and their step to the solves' tolerance; and a box of
unequal widths has a linear flow's strain, its faces' areas, its cells' volumes and a slip of two
components by hand.

**Set-up and step.** Computed once at construction and held bit for bit to the reference's values: the
faces that carry air and those diffusion crosses, the vents, the areas, the volumes, the far cell's share
of each harmonic mean, and for each ordered pair the distance a corner gradient divides by and where it
is defined, which the reference forms in every strain; in `walls` the wall-function faces with their
cells and distances, the wall cells with their walls in the reference's order (south, north, west,
east), each wall's own velocity, each inlet face's k and eps, and the entries of the inflow arrays that
no call changes. A step forms the carried velocities, the pseudo-time step and the volume fluxes, the
strain and the terms, the two explicit fields, and each quantity's conductances, sink and solve
(`scalar::implicit_step` by conjugate gradients with its fallback), then checks positivity and forms the
eddy viscosity. The core allocates nothing in a step.

**What stays in Python.** Which segment covers a face (ADR-017 E): the mapping reads the shared coverage
in `TurbulenceBoundary`'s order and hands the core faces with their numbers, and the refusal of an inlet
without its turbulence keys, which names the segment, is the mapping's (`tests/core_mapping.py`). The
variants' constants, kappa and E are handed in from `src/turbulence.py`, their one home. The state's
relaxation (`_turbulence_step`'s nu_t) and the per-iteration writes of `boundary_staggered.py` are the
outer iteration's and go with step 5's second half, as does the resident state: here the binding returns
new arrays.

**The refusals.** `KEpsilon2` reads a state, faces and conditions by attribute and makes the reference's
checks in its order with its types and messages; a true-time step is a `NotImplementedError` (ADR-017 F);
a quantity that is not positive and finite raises `_cfdcore.PositivityFailure` with the numbers, of which
the mapping makes `PositivityError`. `TurbulenceBoundary2` makes no check of a call's values, as the
reference makes none. An array that is not C-contiguous float64 is a `TypeError`.

**Where the Kernels rules are not followed, and why.** Every pass over a field is a block function walked
by rows, or over the whole field where no neighbour is read. Four loops are not blocks: the walk over the
wall faces and the one over the wall cells, whose elements are few and scattered and whose arithmetic is
a logarithm or a power, a call of the C library a cell at a time; RNG's eta and its cube, for the call and
because a square root that may set `errno` is a branch under the project's flags, so its array of cubes
is formed first and the rest of R is a block; and the list of a failed step's bad cells, formed only on a
failure. `flag_cells`, which turns the conditions' bools into masks, is a block GCC 13 leaves scalar (T9).

## 2. The tests

<!-- tables94 parity begin -->
### T1. The parity cases by function

| Test | Cases | Passed | Failed | Skipped | Skipped where NumPy runs its AVX-512 loops |
|---|---|---|---|---|---|
| `test_the_blocks_equal_the_reference_bit_for_bit` | 88 | 88 | 0 | 0 | 0 |
| `test_the_standard_terms_equal_the_reference_bit_for_bit` | 44 | 44 | 0 | 0 | 0 |
| `test_the_rng_terms_equal_the_reference_bit_for_bit` | 44 | 44 | 0 | 0 | 44 |
| `test_a_production_given_as_a_negative_zero_keeps_its_sign` | 1 | 1 | 0 | 0 | 0 |
| `test_the_rng_terms_are_within_level_one_in_any_environment` | 1 | 1 | 0 | 0 | 0 |
| `test_the_corner_gradients_equal_the_reference_bit_for_bit` | 44 | 44 | 0 | 0 | 0 |
| `test_small_rooms_equal_the_reference_bit_for_bit` | 72 | 72 | 0 | 0 | 72 |
| `test_the_vents_faces_carry_a_flux_diffusion_does_not_cross` | 1 | 1 | 0 | 0 | 0 |
| `test_the_fixed_arrays_equal_the_reference` | 11 | 11 | 0 | 0 | 0 |
| `test_the_conditions_equal_the_reference_bit_for_bit` | 88 | 88 | 0 | 0 | 88 |
| `test_the_wall_viscosity_equals_the_reference_bit_for_bit` | 22 | 22 | 0 | 0 | 22 |
| `test_the_walls_are_within_level_one_in_any_environment` | 1 | 1 | 0 | 0 | 0 |
| `test_a_boundary_used_again_holds_nothing_of_the_call_before` | 1 | 1 | 0 | 0 | 0 |
| `test_a_solid_edge_cell_takes_the_fields_largest_value` | 1 | 1 | 0 | 0 | 0 |
| `test_the_largest_value_is_read_over_the_non_solid_cells` | 1 | 1 | 0 | 0 | 0 |
| `test_the_initial_values_equal_the_reference` | 10 | 10 | 0 | 0 | 0 |
| `test_a_room_without_an_inlet_has_no_initial_values` | 1 | 1 | 0 | 0 | 0 |
| `test_the_initial_values_are_summed_as_python_sums_floats` | 1 | 1 | 0 | 0 | 0 |
| `test_log_law_floor_equals_the_reference` | 1 | 1 | 0 | 0 | 0 |
| `test_log_law_floor_refuses_laws_that_do_not_meet` | 3 | 3 | 0 | 0 | 0 |
| `test_inflow_values_equal_the_reference` | 1 | 1 | 0 | 0 | 0 |
| `test_wall_viscosity_equals_the_reference_bit_for_bit` | 2 | 2 | 0 | 0 | 2 |
| `test_a_step_equals_the_reference_by_its_blocks` | 36 | 36 | 0 | 0 | 36 |
| `test_a_step_on_the_other_rooms` | 3 | 3 | 0 | 0 | 3 |
| `test_advance_is_the_step_with_the_boundarys_conditions` | 2 | 2 | 0 | 0 | 0 |
| `test_a_solve_at_its_cap_falls_back_as_the_reference` | 3 | 3 | 0 | 0 | 0 |
| `test_a_cg_iterate_that_is_not_positive_falls_back_as_the_reference` | 1 | 1 | 0 | 0 | 0 |
| `test_a_positivity_failure_is_the_references` | 2 | 2 | 0 | 0 | 0 |
| `test_a_positivity_failure_by_infinity_is_the_references` | 1 | 1 | 0 | 0 | 0 |
| `test_a_step_refuses_what_the_reference_refuses` | 74 | 74 | 0 | 0 | 0 |
| `test_the_refusals_of_two_arguments_come_in_the_references_order` | 1 | 1 | 0 | 0 | 0 |
| `test_a_true_time_step_is_refused` | 5 | 5 | 0 | 0 | 0 |
| `test_an_array_the_core_cannot_read_is_a_type_error` | 12 | 12 | 0 | 0 | 0 |
| `test_the_state_is_the_references` | 1 | 1 | 0 | 0 | 0 |
| `test_the_models_construction_refuses` | 16 | 16 | 0 | 0 | 0 |
| `test_the_boundarys_construction_refuses` | 21 | 21 | 0 | 0 | 0 |
| `test_an_inlet_without_turbulence_is_refused_as_the_reference_refuses_it` | 1 | 1 | 0 | 0 | 0 |
| `test_the_references_vents_are_the_boundarys` | 1 | 1 | 0 | 0 | 0 |
| `tests/test_core_turbulence.py` | 619 | 619 | 0 | 0 | 267 |
<!-- tables94 parity end -->

<!-- tables94 tests begin -->
### T1b. CTest, the parity modules and the suites

| Tests | Cases | Passed | Failed | Skipped |
|---|---|---|---|---|
| CTest, suite `Turbulence` | 22 | 22 | 0 | 0 |
| CTest, suite `Elementary` | 1 | 1 | 0 | 0 |
| CTest, suite `Walls` | 15 | 15 | 0 | 0 |
| CTest, suite `MomentumThreeDirections` | 3 | 3 | 0 | 0 |
| CTest, suite `PressureThreeDirections` | 4 | 4 | 0 | 0 |
| CTest, suite `TurbulenceThreeDirections` | 7 | 7 | 0 | 0 |
| CTest, all suites | 233 | 233 | 0 | 0 |
| the five parity modules, WSL2, `CFD_CORE_REQUIRED=1`, `NPY_DISABLE_CPU_FEATURES=X86_V4` | 5,814 | 5,797 | 0 | 17 |
| the whole suite, WSL2, the core built | 8,924 | 8,640 | 0 | 284 |
| the whole suite, Windows, the core not built | 3,115 | 3,110 | 0 | 5 |
<!-- tables94 tests end -->

`tests/test_core_turbulence.py` (in CI's core job, which sets `NPY_DISABLE_CPU_FEATURES=X86_V4`) runs
prompt 87's meshes, the vents' and the shelf's, the Annex 20 mesh, a cavity whose four walls move, and small
rooms of every row length with random widths, planted SOLID cells, inlets on three edges (one oblique),
a moving lid, pressure outlets and planted vents, discharges and exhausts. The planted states span
decades with an eddy viscosity of their own, zero in places; the flows hold both signs, zeros of both
signs, values on the faces of SOLID cells and rooms mostly at rest. The tests of a block that calls an
elementary function ask for a NumPy that calls the C library, and skip with the reason elsewhere (an
error with `CFD_CORE_REQUIRED=1`); two more hold the same blocks to REQ-N03's 1e-12 in any environment.

The step is held in two parts. The fields the core's two solves start from, and each solve's operands,
are the reference's bit for bit: the core's step must equal, in every bit, the core's own `implicit_step`
on the systems the reference's step handed its solver. The solves are then asked prompt 88's rule on
inputs like a run's. On inputs planted over decades the rule is recorded and not asserted (T5): their
systems amplify the dot products' rounding, as prompt 88 found of systems near rounding.

<!-- tables94 planted begin -->
### T5. The CG rule on the parity module's planted steps

| Steps | Solves | Counts that differ | by at most | Fallbacks on one side only | Fields beyond 1e-12 at equal counts | The largest there | The largest at counts that differ |
|---|---|---|---|---|---|---|---|
| planted over decades (the module asserts the wiring only) | 72 | 6 | 2 | 0 | 13 | 1.6e-09 | 3.7e-06 |
| gentle (the module asserts the rule) | 72 | 0 | 0 | 0 | 0 | 2.9e-16 | 0 |
<!-- tables94 planted end -->

## 3. The captured steps

The reference's own calls, captured in a probe process from the first outer iterations of warm-started
solves with NumPy's AVX-512 loops off, and replayed through the core. No configuration of record advects
k and eps by UMIST, so the last case is the 80x30 product room with that one key changed
(`results/builder94/product80_umist.yaml`).

<!-- tables94 captures begin -->
### T2. The captured steps

| Case | The state the solve starts from | Steps | Of them different | Variant, advection | CG iterations of k | of eps | Fallbacks | Largest change of k over the steps, m^2/s^2 |
|---|---|---|---|---|---|---|---|---|
| 80x30 product room | results/builder81/placement/dust.h5 | 10 | 10 | standard, upwind | 12 to 12 | 10 to 10 | 0 | 9.7e-10 |
| 80x30 vents room | results/builder90/vents80_flow.npz | 10 | 10 | standard, upwind | 10 to 10 | 9 to 9 | 0 | 2.4e-11 |
| 216x72 Annex 20 room | its own solve from rest, to the cap of 400 outer iterations | 10 | 10 | standard, upwind | 11 to 12 | 9 to 10 | 0 | 1.3e-03 |
| 656x60 backward-facing step | its own solve from rest, to the cap of 400 outer iterations | 10 | 10 | standard, upwind | 22 to 24 | 19 to 21 | 0 | 1.4e+01 |
| 80x30 product room, k and eps by UMIST | results/builder81/placement/dust.h5 | 10 | 10 | standard, umist | 12 to 12 | 10 to 11 | 0 | 1.4e-02 |
<!-- tables94 captures end -->

<!-- tables94 first begin -->
### T3. The first kind on the captured steps: steps whose block differs in a bit, NumPy's AVX-512 loops off

| Block | 80x30 product room | 80x30 vents room | 216x72 Annex 20 room | 656x60 backward-facing step | 80x30 product room, k and eps by UMIST |
|---|---|---|---|---|---|
| the wall viscosity on both wall stencils (log) | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 |
| the four inflow arrays | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 |
| the four edges' tangential velocities | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 |
| the held cells and their eps (cube) | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 |
| the given production (log) | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 |
| the pseudo-time step | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 |
| the corner gradients du/dy and dv/dx | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 |
| the strain squared | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 |
| production, R, the two growths and the two decays | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 |
| k advected | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 |
| eps advected | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 |
| k's conductances | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 |
| eps's conductances | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 |
| the eddy viscosity | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 |
| k's explicit field | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 |
| eps's explicit field | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 | 0 of 10 |

The reference called again on a kept step's arguments gave other bits than the kept ones on 0 of 50 steps.
<!-- tables94 first end -->

<!-- tables94 solves begin -->
### T4. The captured CG solves, and the steps' own differences, each over the reference's largest magnitude

| Case | Quantity | Solves | Counts that differ | Fallbacks on one side only | Largest difference of a solve's field | of the step's | of the step's nu_t | of the step's with the core's own conditions |
|---|---|---|---|---|---|---|---|---|
| 80x30 product room | k | 10 | 0 | 0 | 1.9e-16 | 1.9e-16 | 6.9e-16 | 1.9e-16 |
| 80x30 product room | eps | 10 | 0 | 0 | 1.2e-16 | 1.2e-16 | 6.9e-16 | 1.2e-16 |
| 80x30 vents room | k | 10 | 0 | 0 | 1.5e-16 | 1.5e-16 | 5.5e-16 | 1.5e-16 |
| 80x30 vents room | eps | 10 | 0 | 0 | 1.4e-16 | 1.4e-16 | 5.5e-16 | 1.4e-16 |
| 216x72 Annex 20 room | k | 10 | 0 | 0 | 3.2e-16 | 3.2e-16 | 5.9e-16 | 3.2e-16 |
| 216x72 Annex 20 room | eps | 10 | 0 | 0 | 1.5e-16 | 1.5e-16 | 5.9e-16 | 1.5e-16 |
| 656x60 backward-facing step | k | 10 | 0 | 0 | 5.9e-16 | 5.9e-16 | 1.1e-15 | 5.9e-16 |
| 656x60 backward-facing step | eps | 10 | 0 | 0 | 2.3e-16 | 2.3e-16 | 1.1e-15 | 2.3e-16 |
| 80x30 product room, k and eps by UMIST | k | 10 | 0 | 0 | 1.8e-16 | 1.8e-16 | 6.9e-16 | 1.8e-16 |
| 80x30 product room, k and eps by UMIST | eps | 10 | 0 | 0 | 2.2e-16 | 2.2e-16 | 6.9e-16 | 2.2e-16 |
<!-- tables94 solves end -->

The last three columns of T4 are the chain's own differences, reported and judged at level 2
(REQ-N03 as amended 2026-11-25): the whole step of the core against the whole step of the reference, and
the same with the core's own conditions (`advance`).

<!-- tables94 rest begin -->
### T4b. The first outer iterations from rest, beside the captures and not scored, NumPy's AVX-512 loops off

| Case | Steps | With a block that differs in a bit | Counts that differ | Fallbacks on one side only | Largest difference of a solve's field |
|---|---|---|---|---|---|
| 80x30 product room | 10 | 0 | 0 | 0 | 1.3e-16 |
| 80x30 vents room | 10 | 0 | 0 | 0 | 1.4e-16 |
| 216x72 Annex 20 room | 10 | 0 | 0 | 0 | 1.3e-16 |
| 656x60 backward-facing step | 10 | 0 | 0 | 0 | 2.0e-16 |
| 80x30 product room, k and eps by UMIST | 10 | 0 | 0 | 0 | 1.8e-16 |
<!-- tables94 rest end -->

T4b is an addition to the prompt's captures: a solve from rest starts from uniform k and eps in a room
mostly at rest, which no warm start meets and where the pressure correction's fields once missed 1e-12
(`phase6_pressure.md`, T8). Here the rule holds.

## 4. The elementary functions

<!-- tables94 elementary begin -->
### T6. NumPy's logarithm and cube against the C library's

| NumPy's AVX-512 loops | Function | NumPy's loop | Values | Differ from the C library's | Largest relative difference | in units of the last place |
|---|---|---|---|---|---|---|
| X86_V4 off | `log(E y*)` | X86_V3 | 200,000 | 0 | 0 | 0 |
| X86_V4 off | `u_k**3` | baseline(X86_V2) | 200,000 | 0 | 0 | 0 |
| X86_V4 off | `eta**3` | baseline(X86_V2) | 200,000 | 0 | 0 | 0 |
| as it runs by default on this machine | `log(E y*)` | X86_V4 | 200,000 | 20 | 2.0e-16 | 1 |
| as it runs by default on this machine | `u_k**3` | X86_V4 | 200,000 | 10,618 | 2.2e-16 | 1 |
| as it runs by default on this machine | `eta**3` | X86_V4 | 200,000 | 10,599 | 2.2e-16 | 1 |
<!-- tables94 elementary end -->

<!-- tables94 default begin -->
### T3b. The blocks that differ where NumPy runs its AVX-512 loops, the same steps

| Block | Steps on which it differs | Values that differ | Largest difference over the reference's largest magnitude |
|---|---|---|---|
| the held cells and their eps (cube) | 50 of 50 | 1,219 | 2.7e-16 |
| the given production (log) | 5 of 50 | 6 | 9.8e-17 |

The other 14 blocks differ in no bit on any step.
<!-- tables94 default end -->

The captured steps hold no R (every case is the standard variant), and no wall viscosity of theirs meets
a logarithm the loops round differently; the planted RNG cases do meet the cube.

Two things follow from T6 that are not the port's. The reference's own coupled fields depend, in their
last bits, on whether the processor that runs NumPy has AVX-512, so two machines of the same NumPy can
give different ones (F2). And a run of the core is the same on both.

## 5. Planted defects

<!-- tables94 plants begin -->
### T7. The planted defects: which nets fail on each

| Plant | What it changes | Fails |
|---|---|---|
| `control` | nothing | nothing, as it must |
| `r_not_split` | the prompt's: RNG's R not split by sign (all of it in the growth) | CTest (1); parity (99 cases of 4 tests) |
| `decay_explicit` | the prompt's: the decay taken explicitly | CTest (2); parity (48 cases of 7 tests) |
| `buried_not_at_the_corner` | the prompt's: a corner gradient read from the stored zero beyond an obstacle | CTest (3); parity (258 cases of 8 tests); replay (product80, vents80, backstep, product80_umist) |
| `edge_tangential_dropped` | the prompt's: the edge's tangential velocity dropped (the adjacent face's in its place) | CTest (6); parity (334 cases of 10 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `step_uncapped` | the prompt's: the pseudo-time step uncapped at a cell at rest | CTest (1); parity (160 cases of 3 tests); replay (product80, vents80, backstep, product80_umist) |
| `production_given_ignored` | the prompt's: the production given ignored in a wall cell | CTest (2); parity (206 cases of 10 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `eps_not_held` | the prompt's: eps not held | CTest (2); parity (43 cases of 4 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `vent_as_wall` | the prompt's: a vent face treated as a wall | CTest (2); parity (81 cases of 5 tests); replay (vents80) |
| `floor_wrong_constants` | the prompt's: the scalable floor from the wrong constants | CTest (2); parity (1 cases of 1 tests) |
| `fallback_from_cg` | the prompt's: the fallback from the CG iterate | CTest (2); parity (9 cases of 5 tests) |
| `carried_unmasked` | a face beside a SOLID cell read as it is stored | CTest (3); parity (258 cases of 8 tests) |
| `gradient_everywhere` | a gradient between two buried faces, or an edge and one, not zero | CTest (2); parity (257 cases of 7 tests) |
| `gradient_sign` | a gradient's difference the other way round | CTest (4); parity (44 cases of 1 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `shear_of_one_component` | the second component's gradient written over the first's | CTest (3); parity (292 cases of 9 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `shear_mean_half` | the mean of the four corners a half of their sum | CTest (2); parity (292 cases of 9 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `shear_three_corners` | a corner left out of the mean and another counted twice | CTest (2); parity (292 cases of 9 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `diagonal_along_once` | the stretching along the rows counted once | CTest (5); parity (292 cases of 9 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `diagonal_across_once` | the stretching across the rows counted once | CTest (4); parity (292 cases of 9 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `strain_order` | the shear added before the stretching across the rows | CTest (1); parity (291 cases of 8 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `strain_in_solid` | the strain of SOLID cells kept | parity (136 cases of 2 tests); replay (product80, vents80, backstep, product80_umist) |
| `strain_width` | the rows' width across them the first row's | CTest (4); parity (291 cases of 8 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `edge_low_along_rows` | the low edge's value along the rows the adjacent face's | CTest (3); parity (334 cases of 10 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `edge_high_along_rows` | the high edge's value along the rows the low edge's | CTest (3); parity (333 cases of 9 tests); replay (backstep) |
| `edge_high_across_rows` | the high edge's value across the rows dropped | CTest (4); parity (333 cases of 9 tests) |
| `edge_at_the_centre` | the edges' positions taken at the first and last corner's neighbour | CTest (5); parity (335 cases of 10 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `span_zero_defined` | a gradient over no distance taken over one | CTest (2); parity (257 cases of 7 tests) |
| `pair_low_swapped` | the first component's edges swapped, low for high | CTest (2); parity (289 cases of 8 tests) |
| `pair_high_swapped` | the second component's edges swapped, low for high | CTest (1); parity (289 cases of 8 tests); replay (backstep) |
| `production_without_viscosity` | the production without the eddy viscosity | CTest (1); parity (204 cases of 8 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `ratio_in_solid` | eps / k formed in SOLID cells | CTest (1); parity (137 cases of 4 tests); replay (product80, vents80, backstep, product80_umist) |
| `ratio_inverted` | k / eps in the place of eps / k | CTest (2); parity (208 cases of 11 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `growth_c_2` | the growth of eps with C_2 | CTest (2); parity (105 cases of 6 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `growth_association` | the growth's products taken in another order | CTest (1); parity (105 cases of 6 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `decay_c_1` | the decay of eps with C_1 | CTest (2); parity (107 cases of 8 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `rng_eta_0` | R without its zero at eta_0 | CTest (1); parity (99 cases of 4 tests) |
| `rng_beta` | R's denominator without beta | CTest (1); parity (99 cases of 4 tests) |
| `rng_c_mu` | R without C_mu | CTest (1); parity (99 cases of 4 tests) |
| `rng_association` | R's last products taken in another order | parity (98 cases of 3 tests) |
| `rng_parts_swapped` | the positive part in the growth and the negative in the decay | CTest (1); parity (99 cases of 4 tests) |
| `rng_growth_signed` | the growth with R's magnitude whatever its sign | CTest (1); parity (99 cases of 4 tests) |
| `rng_decay_without_eps` | the positive part in the decay not over eps | CTest (1); parity (99 cases of 4 tests) |
| `rng_decay_in_solid` | the decay formed in SOLID cells | parity (69 cases of 3 tests) |
| `rng_r_in_solid` | R formed in SOLID cells | nothing: eta and eps are zero in a SOLID cell, so R is zero there unmasked |
| `eta_order` | eta as sqrt(S^2) times k / eps | parity (98 cases of 3 tests) |
| `eta_cube_products` | the cube of eta by products | parity (98 cases of 3 tests) |
| `eta_in_solid` | eta formed in SOLID cells | nothing: in a SOLID cell it makes eta, its cube and the coefficient NaN, and R, the one value formed from them, is masked |
| `r_for_every_variant` | R formed for the standard variant | CTest (7); parity (110 cases of 9 tests); replay (raises) |
| `has_r_by_one` | R where only one of its constants is given (the binding) | parity (457 cases of 31 tests); replay (raises) |
| `rate_of_one_face` | the rate from a cell's low faces alone | CTest (3); parity (203 cases of 6 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `rate_across_of_one_face` | the rate across the rows from a cell's high faces alone | CTest (3); parity (203 cases of 6 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `rate_larger_of_two` | the larger of the directions' rates in the place of their sum | CTest (10); parity (207 cases of 9 tests); replay (product80, vents80) |
| `rate_signed` | the rate from a signed velocity | CTest (2); parity (206 cases of 8 tests); replay (product80, vents80) |
| `solid_moves` | a SOLID cell counted among those that move | parity (69 cases of 3 tests); replay (vents80) |
| `cap_smallest` | a cell at rest takes the smallest step | CTest (1); parity (160 cases of 3 tests); replay (product80, vents80, backstep, product80_umist) |
| `cfl_squared` | the Courant number squared | CTest (4); parity (206 cases of 8 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `flux_without_area` | the volume flux without the face's area | CTest (1); parity (207 cases of 8 tests); replay (raises) |
| `nothing_moves_goes_on` | a step where nothing moves not refused | CTest (1); parity (1 cases of 1 tests) |
| `advection_scheme_swapped` | upwind where UMIST is asked and the reverse | parity (205 cases of 7 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `vents_not_advected` | the vents' rule left out of the advection | parity (80 cases of 3 tests); replay (vents80) |
| `vents_closed` | a vent's face carries nothing | CTest (1); parity (137 cases of 9 tests); replay (vents80) |
| `vents_diffuse` | diffusion crosses a vent's face | CTest (1); parity (89 cases of 6 tests); replay (vents80) |
| `edges_diffuse` | diffusion's mask open on the domain's faces | CTest (1); parity (12 cases of 2 tests) |
| `solid_low_open` | a face open whose low cell is SOLID | CTest (3); parity (269 cases of 11 tests); replay (product80, vents80, backstep, product80_umist) |
| `solid_high_open` | a face open whose high cell is SOLID | CTest (4); parity (267 cases of 10 tests); replay (product80, vents80, backstep, product80_umist) |
| `divergence_sign` | the divergence's sign along the rows | CTest (4); parity (207 cases of 8 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `divergence_order` | the divergence across the rows taken as one difference | parity (203 cases of 6 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `advection_order` | the step over the volume taken first | parity (203 cases of 6 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `growth_without_step` | the growth not times the step | CTest (1); parity (43 cases of 4 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `growth_of_k_for_eps` | eps grown by k's production | CTest (1); parity (42 cases of 3 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `inflow_of_k_for_eps` | eps carried in with k's values | CTest (1); parity (43 cases of 5 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `k_advected_as_eps` | k's explicit field from eps | parity (45 cases of 5 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `explicit_in_solid` | the explicit field kept in SOLID cells | parity (12 cases of 1 tests); replay (vents80) |
| `held_explicit_only` | eps held in the explicit field and not in the solve | CTest (1); parity (43 cases of 4 tests) |
| `held_solve_only` | eps held in the solve at its advected value | CTest (2); parity (43 cases of 4 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `held_by_given` | the held cells those whose production is given | CTest (3); parity (41 cases of 4 tests) |
| `k_held_too` | k held where eps is | parity (44 cases of 5 tests) |
| `sigma_swapped` | k diffused with eps's sigma | CTest (1); parity (149 cases of 6 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `eps_solved_with_ks_conductances` | eps's solve on k's conductances | parity (25 cases of 4 tests) |
| `diffusivity_without_air` | the diffusivity without air's viscosity | CTest (6); parity (208 cases of 9 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `harmonic_share` | the near cell's share in the place of the far one's | CTest (1); parity (203 cases of 6 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `harmonic_arithmetic` | the arithmetic mean | CTest (1); parity (203 cases of 6 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `conductance_area` | the conductance without the face's area | CTest (2); parity (205 cases of 7 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `conductance_distance` | the conductance's quotient taken first | parity (203 cases of 6 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `conductance_unmasked` | the conductance beside a SOLID cell kept | CTest (1); parity (161 cases of 4 tests); replay (product80, vents80, backstep, product80_umist) |
| `sink_without_volume` | the sink without the volume | CTest (1); parity (46 cases of 6 tests) |
| `eps_decays_as_k` | eps's decay k's | CTest (1); parity (44 cases of 5 tests) |
| `k_decays_as_eps` | k's decay eps's | CTest (1); parity (45 cases of 6 tests) |
| `solve_by_jacobi` | the solves by Jacobi | CTest (1); parity (45 cases of 5 tests) |
| `solve_tolerance` | the solves' tolerance ten times the configuration's | parity (41 cases of 5 tests) |
| `solve_cap` | the solves' cap one above the configuration's | CTest (1); parity (4 cases of 2 tests) |
| `solve_one_step` | the solves at one step for every cell | parity (44 cases of 5 tests) |
| `volume_of_one_width` | a cell's volume one of its widths | CTest (4); parity (217 cases of 9 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `area_of_its_own_width` | a face's area with the width along its own direction | CTest (5); parity (205 cases of 7 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `nu_t_linear` | the eddy viscosity with k in the place of k^2 | CTest (2); parity (161 cases of 3 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `nu_t_association` | the eddy viscosity's products taken in another order | CTest (1); parity (161 cases of 3 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `nu_t_in_solid` | the eddy viscosity formed in SOLID cells | CTest (3); parity (137 cases of 3 tests); replay (product80, vents80, backstep, product80_umist) |
| `k_positivity` | k not checked after the solves | CTest (1); parity (2 cases of 2 tests) |
| `eps_positivity` | eps not checked after the solves | CTest (1); parity (1 cases of 1 tests) |
| `eps_checked_first` | eps checked before k | parity (1 cases of 1 tests) |
| `bad_infinity_passes` | an infinite value passes | CTest (1); parity (4 cases of 2 tests) |
| `bad_last` | the last bad cell named, not the first | parity (3 cases of 2 tests) |
| `bad_count` | the bad cells counted one too many | CTest (1); parity (3 cases of 2 tests) |
| `bad_least_of_bad` | the least value read off the first cell alone | CTest (1); parity (2 cases of 1 tests) |
| `records_not_kept` | the step's counts not kept (the binding) | parity (45 cases of 5 tests) |
| `records_after_failure` | the counts kept only where the step succeeds (the binding) | parity (2 cases of 1 tests) |
| `converged_either` | converged where either solve is (the binding) | parity (1 cases of 1 tests) |
| `wall_floor_dropped` | the wall viscosity's log law not floored | CTest (1); parity (57 cases of 4 tests); replay (product80, vents80, annex216, product80_umist) |
| `wall_density` | the wall viscosity without the density | CTest (2); parity (95 cases of 4 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `wall_association` | the wall viscosity's products taken in another order | parity (94 cases of 3 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `wall_k_of_one_cell` | k at a wall face's node from one cell | CTest (1); parity (95 cases of 3 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `wall_elsewhere_lost` | the other faces' viscosity not kept | parity (95 cases of 3 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `stencil_edges_included` | the domain-edge faces counted among the unknowns | CTest (4); parity (106 cases of 4 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `stencil_solid_included` | the faces of SOLID cells counted among the unknowns | parity (97 cases of 4 tests); replay (product80, vents80, backstep, product80_umist) |
| `stencil_one_side` | a face bounds a SOLID cell on its high side only | CTest (2); parity (97 cases of 4 tests); replay (product80, vents80, backstep, product80_umist) |
| `stencil_corners_swapped` | the low edge's wall corners for the high one's | CTest (1); parity (103 cases of 4 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `stencil_vents` | a vent's span counted among the wall stencil's faces | CTest (1); parity (49 cases of 4 tests); replay (vents80) |
| `stencil_distance_swapped` | the distance to the other face | parity (99 cases of 3 tests); replay (product80, vents80, backstep, product80_umist) |
| `cells_vents` | the cell beside a vent a wall cell | CTest (1); parity (73 cases of 4 tests); replay (vents80) |
| `cells_edges_swapped` | the low edge's wall cells for the high one's | CTest (1); parity (163 cases of 4 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `cells_order` | a corner cell's walls summed in another order | parity (60 cases of 1 tests) |
| `cells_solid_included` | SOLID cells counted among the wall cells | parity (143 cases of 4 tests); replay (product80, vents80, backstep, product80_umist) |
| `cells_mean_dropped` | a corner cell's walls summed, not averaged | CTest (1); parity (161 cases of 3 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `production_mean_dropped` | a corner cell's productions summed, not averaged | CTest (1); parity (161 cases of 3 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `speed_of_one_corner` | the edge's own velocity at a cell one corner's | parity (72 cases of 1 tests) |
| `speed_dropped` | a moving wall at rest | CTest (1); parity (80 cases of 2 tests) |
| `speed_of_the_other_edge` | a wall's velocity the opposite edge's | CTest (1); parity (80 cases of 2 tests) |
| `speed_on_obstacles` | an obstacle's wall moves with the edge | parity (67 cases of 1 tests) |
| `held_cube_products` | the cube of u_k by products | parity (160 cases of 2 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `held_kappa` | the held eps without kappa | CTest (2); parity (161 cases of 3 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `slip_signed` | the slip with its sign | CTest (4); parity (163 cases of 4 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `slip_of_one_component` | in three directions the slip of one component | CTest (4) |
| `slip_unmasked` | the slip from a face of a SOLID cell as it is stored | parity (137 cases of 3 tests) |
| `slip_vents_masked` | a vent's face read as zero in the slip | parity (36 cases of 1 tests) |
| `slip_of_one_face` | the velocity along the wall one face's | parity (161 cases of 3 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `production_floor` | the production's log law not floored | parity (133 cases of 3 tests); replay (product80, vents80, annex216, product80_umist) |
| `production_association` | the production's quotient taken in another order | parity (160 cases of 2 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `root_c_mu` | C_mu^(1/4) by two square roots | CTest (5); parity (183 cases of 4 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `tangential_always_known` | an open edge's tangential velocity its condition's zero | CTest (3); parity (72 cases of 1 tests); replay (backstep) |
| `tangential_never_known` | a wall's tangential velocity the adjacent face's | CTest (1); parity (160 cases of 2 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `tangential_far_row` | the adjacent faces those of the other edge | CTest (1); parity (72 cases of 1 tests); replay (backstep) |
| `inflow_stand_in_least` | beside a SOLID edge cell a half of the largest value | CTest (1); parity (138 cases of 4 tests); replay (product80, vents80, backstep, product80_umist) |
| `inflow_zero` | beside a SOLID edge cell the cell's zero | CTest (1); parity (140 cases of 5 tests); replay (product80, vents80, backstep, product80_umist) |
| `inflow_stand_in_in_solid` | the largest value read over every cell | parity (1 cases of 1 tests) |
| `inflow_inlets_overwritten` | an inlet's value the adjacent cell's | CTest (1); parity (153 cases of 3 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `inflow_of_k_twice` | eps carried in with k's inlet value | CTest (1); parity (153 cases of 3 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `inflow_high_cell` | a high edge's value from one cell further in | CTest (1); parity (161 cases of 3 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `exhaust_high_cell` | an exhaust's room cell always the one on its high side | parity (68 cases of 3 tests); replay (vents80) |
| `exhaust_low_cell` | an exhaust's room cell always the one on its low side | CTest (1); parity (69 cases of 3 tests); replay (vents80) |
| `exhausts_dropped` | an exhaust's faces left without a value | CTest (1); parity (70 cases of 4 tests); replay (vents80) |
| `vent_inlets_dropped` | the discharge vents left out | CTest (2); parity (78 cases of 6 tests); replay (vents80) |
| `initial_unweighted` | the initial values' means not weighted | CTest (1); parity (9 cases of 2 tests) |
| `initial_signed` | the initial values weighted by the signed speed | CTest (1); parity (5 cases of 2 tests) |
| `initial_plain_sum` | the initial values summed without the compensation | parity (10 cases of 2 tests) |
| `initial_kahan` | the compensation of the smaller sum only (Kahan's) | parity (1 cases of 1 tests) |
| `inflow_values_signed` | the inlet's k from a signed speed's square's exponent | parity (36 cases of 3 tests) |
| `inflow_values_factor` | the inlet's k without its factor | CTest (1); parity (175 cases of 7 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `inflow_values_c_mu` | the inlet's eps with C_mu^(3/4) | CTest (1); parity (175 cases of 7 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `floor_loose` | Newton's method stopped a step early | CTest (1); parity (1 cases of 1 tests) |
| `floor_meeting_dropped` | laws that do not meet not refused | CTest (1); parity (3 cases of 1 tests) |
| `floor_meeting_at_one` | laws that touch not refused | parity (1 cases of 1 tests) |
| `square_by_product` | Python's square of a float by a product | parity (2 cases of 2 tests) |
| `cube_by_product` | every cube by products | CTest (1); parity (222 cases of 4 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `log_single` | the logarithm one bit off | CTest (1); parity (186 cases of 6 tests); replay (product80, vents80, annex216, backstep, product80_umist) |
| `three_pairs` | in three directions the last pair's shear left out | CTest (3) |
| `three_corner_stride` | the corners of a pair that crosses the rows read along the other direction | CTest (3) |
| `three_area` | a face's area the last other width alone | CTest (3) |
| `three_volume` | a cell's volume without its last width in three directions | CTest (3) |
| `three_divergence` | in three directions the last direction's divergence left out | CTest (2) |
| `check_state_shape` | the refusal removed: a state's shape | parity (6 cases of 1 tests) |
| `check_state_positive` | the refusal removed: a state's k and eps in range | parity (8 cases of 1 tests) |
| `check_state_nu_t` | the refusal removed: a state's nu_t in range | parity (4 cases of 2 tests) |
| `check_state_nu_t_zero` | the refusal removed: a state's nu_t of zero refused too | parity (269 cases of 13 tests) |
| `check_state_solid` | the refusal removed: a state's zero in SOLID cells | parity (3 cases of 1 tests) |
| `check_faces_finite` | the refusal removed: the faces finite | parity (2 cases of 2 tests) |
| `check_conditions_finite` | the refusal removed: a condition finite | parity (11 cases of 2 tests) |
| `check_conditions_solid` | the refusal removed: a condition naming a SOLID cell | parity (2 cases of 1 tests) |
| `check_inflow_edges` | the refusal removed: an inflow value on the domain's faces positive | parity (8 cases of 1 tests) |
| `check_inflow_vents` | the refusal removed: an inflow value on the vents' faces positive | parity (4 cases of 1 tests) |
| `check_eps_wall` | the refusal removed: the held eps positive | parity (1 cases of 1 tests) |
| `check_production` | the refusal removed: the given production not negative | parity (1 cases of 1 tests) |
| `check_true_time` | the refusal removed: a true-time step | parity (5 cases of 1 tests) |
| `check_other_grid` | the refusal removed: a boundary on another grid | parity (2 cases of 1 tests) |
| `check_state_call` | the refusal removed: the values of `state` | parity (1 cases of 1 tests) |
| `check_initial_values` | the refusal removed: a room without an inlet | CTest (1); parity (1 cases of 1 tests) |
| `check_exhaust_edge` | the refusal removed: an exhaust on a domain edge | CTest (1); parity (2 cases of 1 tests) |
| `check_inlet_numbers` | the refusal removed: an inlet's numbers | CTest (1); parity (2 cases of 1 tests) |
| `check_walls_constants` | the refusal removed: the boundary's constants | CTest (1); parity (2 cases of 1 tests) |
| `check_model_constants` | the refusal removed: the model's constants | CTest (1); parity (3 cases of 1 tests) |
| `check_model_rng` | the refusal removed: the constants of R | CTest (1); parity (1 cases of 1 tests) |
| `check_model_widths` | the refusal removed: a width | CTest (1); parity (1 cases of 1 tests) |
| `check_model_vents` | the refusal removed: vents in some directions | CTest (1); parity (exit -11) |
| `check_walls_conditions_length` | the refusal removed: the lengths of the conditions' arguments | CTest (1) |
| `check_model_tangential_length` | the refusal removed: the lengths of the tangential velocities | CTest (1) |

191 of 193 plants caught; not caught and not equivalent: none.
<!-- tables94 plants end -->

The plants were built and run in three streams, each beginning with the unchanged source; the
replay runs with NumPy's AVX-512 loops off. The first run left one plant that is not equivalent
uncaught: the largest value that stands in beside a SOLID edge cell read over every cell
(`inflow_stand_in_in_solid`), which a run's state, zero in SOLID cells, cannot show. The
reference's boundary makes no check of the state, so a planted one that is not zero there tells the
two apart, and `test_the_largest_value_is_read_over_the_non_solid_cells` holds it; T7's row is that
plant's run after the test was added. A plant in the last bit alone (`square_by_product`, for
Python's square of a float) is caught by the draws of `test_inflow_values_equal_the_reference`,
which are many for that reason. `floor_wrong_constants` fails the floor's own tests and no other: the
floor the boundary reads is handed in from the reference's constants.

## 6. The timing

A step with its conditions on the first outer iteration's arguments of a solve warm-started from the
stored 320x120 product state, in one process, through the binding: the reference's `conditions` then
`step`, the core's two calls of the same names, and the core's one call `advance`, which forms the
conditions in place and hands the step its own arrays.

<!-- tables94 timing begin -->
### T8. A step with its conditions on the 320x120 product state, one thread

| Session | Started | NumPy's AVX-512 loops | CG iterations of k and eps, both sides | Reference, ms (quartiles) | Core, two calls, ms (quartiles) | Core, one call, ms (quartiles) | Ratio | Ratio, one call | Calibration before and after, us | Windows before | Windows after |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-11-25 13:00:13 | on | 29 and 24 | 45.999 (43.33 to 49.00) | 11.019 (10.55 to 11.81) | 10.221 (9.80 to 10.91) | 4.17 | 4.50 | 8.50, 8.45 | msmpeng 39, msedge 38, system 19, on mains True, processor performance 86 | msmpeng 35, msedge 34, system 21, on mains True, processor performance 92 |
| 2 | 2026-11-25 13:00:35 | on | 29 and 24 | 43.126 (41.24 to 46.16) | 10.970 (10.57 to 11.61) | 10.465 (10.04 to 11.06) | 3.93 | 4.12 | 8.51, 8.48 | msmpeng 39, code 19, system 17, on mains True, processor performance 83 | msmpeng 61, code 14, system 13, on mains True, processor performance 86 |
| 3 | 2026-11-25 13:00:57 | on | 29 and 24 | 44.431 (42.02 to 47.50) | 11.027 (10.48 to 11.76) | 10.138 (9.77 to 10.86) | 4.03 | 4.38 | 8.78, 8.44 | msmpeng 48, svchost 23, code 14, on mains True, processor performance 87 | msmpeng 36, code 19, system 12, on mains True, processor performance 89 |
| 4 | 2026-11-25 12:58:56 | off | 29 and 24 | 48.166 (45.94 to 52.38) | 11.105 (10.70 to 11.96) | 10.377 (10.02 to 10.93) | 4.34 | 4.64 | 8.83, 8.51 | msmpeng 36, system 13, msedge 7, on mains True, processor performance 95 | msmpeng 36, msedge 11, system 9, on mains True, processor performance 85 |
| 5 | 2026-11-25 12:59:18 | off | 29 and 24 | 46.412 (42.98 to 51.78) | 10.719 (10.26 to 11.80) | 10.192 (9.73 to 11.09) | 4.33 | 4.55 | 8.92, 8.63 | msmpeng 33, system 17, msedge 13, on mains True, processor performance 88 | msmpeng 38, msedge 15, system 12, on mains True, processor performance 93 |
| 6 | 2026-11-25 12:59:41 | off | 29 and 24 | 47.963 (44.96 to 52.81) | 11.317 (10.89 to 12.11) | 10.823 (10.35 to 11.64) | 4.24 | 4.43 | 8.49, 8.45 | msmpeng 41, dwm 11, code 8, on mains True, processor performance 89 | msedge 63, msmpeng 37, code 28, on mains True, processor performance 89 |
<!-- tables94 timing end -->

<!-- tables94 parts begin -->
### T8b. The parts, the same sessions

| Session | Part | Reference, ms | Core, ms | Ratio |
|---|---|---|---|---|
| 1 | two CG solves | 33.036 | 8.902 | 3.71 |
| 1 | conditions | 6.443 | 0.884 | 7.29 |
| 1 | wall viscosity | 0.084 | 0.044 | 1.90 |
| 1 | terms | 4.523 | 1.443 | 3.14 |
| 1 | pseudo-time step | 0.855 | 0.178 | 4.80 |
| 1 | the step with its conditions less the two solves | 12.963 | 2.117 | 6.12 |
| 1 | the same, the core's two calls in one (`advance`) | 12.963 | 1.319 | 9.83 |
| 2 | two CG solves | 30.586 | 8.670 | 3.53 |
| 2 | conditions | 6.782 | 0.828 | 8.19 |
| 2 | wall viscosity | 0.104 | 0.044 | 2.37 |
| 2 | terms | 4.581 | 1.404 | 3.26 |
| 2 | pseudo-time step | 0.839 | 0.176 | 4.77 |
| 2 | the step with its conditions less the two solves | 12.540 | 2.300 | 5.45 |
| 2 | the same, the core's two calls in one (`advance`) | 12.540 | 1.795 | 6.99 |
| 3 | two CG solves | 30.142 | 8.558 | 3.52 |
| 3 | conditions | 6.418 | 0.811 | 7.91 |
| 3 | wall viscosity | 0.084 | 0.044 | 1.90 |
| 3 | terms | 4.409 | 1.372 | 3.21 |
| 3 | pseudo-time step | 0.817 | 0.178 | 4.59 |
| 3 | the step with its conditions less the two solves | 14.289 | 2.469 | 5.79 |
| 3 | the same, the core's two calls in one (`advance`) | 14.289 | 1.580 | 9.05 |
| 4 | two CG solves | 30.165 | 8.603 | 3.51 |
| 4 | conditions | 9.001 | 0.816 | 11.03 |
| 4 | wall viscosity | 0.087 | 0.044 | 1.98 |
| 4 | terms | 4.218 | 1.523 | 2.77 |
| 4 | pseudo-time step | 0.806 | 0.184 | 4.37 |
| 4 | the step with its conditions less the two solves | 18.001 | 2.502 | 7.19 |
| 4 | the same, the core's two calls in one (`advance`) | 18.001 | 1.774 | 10.14 |
| 5 | two CG solves | 30.171 | 8.288 | 3.64 |
| 5 | conditions | 8.159 | 0.834 | 9.78 |
| 5 | wall viscosity | 0.088 | 0.043 | 2.03 |
| 5 | terms | 4.542 | 1.713 | 2.65 |
| 5 | pseudo-time step | 0.859 | 0.180 | 4.79 |
| 5 | the step with its conditions less the two solves | 16.241 | 2.431 | 6.68 |
| 5 | the same, the core's two calls in one (`advance`) | 16.241 | 1.905 | 8.53 |
| 6 | two CG solves | 31.798 | 8.607 | 3.69 |
| 6 | conditions | 7.682 | 0.858 | 8.96 |
| 6 | wall viscosity | 0.107 | 0.043 | 2.48 |
| 6 | terms | 4.300 | 1.393 | 3.09 |
| 6 | pseudo-time step | 0.907 | 0.193 | 4.71 |
| 6 | the step with its conditions less the two solves | 16.165 | 2.710 | 5.97 |
| 6 | the same, the core's two calls in one (`advance`) | 16.165 | 2.216 | 7.29 |
<!-- tables94 parts end -->

Prediction (d) fails in every session, scored and beside (T8). T8b says where the time is: the two
solves are about four fifths of the core's step and are prompt 88's and 89's kernels, unchanged here,
and run between three and four times as fast as the reference's; everything else in the step is more
than five times as fast in every session, in the core's two calls and more so in its one, which neither
copies the conditions' twelve arrays out nor reads and checks them again. Sessions 1 to 3 are the
reference as NumPy runs by default, the state the scoring rule names; sessions 4 to 6, timed a minute
before them, have the AVX-512 loops off, and the reference's conditions are slower in them.

<!-- tables94 loops begin -->
### T9. The blocks GCC vectorises

| File | Block | Vectorised |
|---|---|---|
| `turbulence.cpp` | `carried_cells` | yes |
| `turbulence.cpp` | `product_cells` | yes |
| `turbulence.cpp` | `gradient_cells` | yes |
| `turbulence.cpp` | `strain_cells` | yes |
| `turbulence.cpp` | `production_cells` | yes |
| `turbulence.cpp` | `standard_cells` | yes |
| `turbulence.cpp` | `rng_cells` | yes |
| `turbulence.cpp` | `step_cells` | yes |
| `turbulence.cpp` | `cap_cells` | yes |
| `turbulence.cpp` | `explicit_cells` | yes |
| `turbulence.cpp` | `gamma_cells` | yes |
| `turbulence.cpp` | `conductance_cells` | yes |
| `turbulence.cpp` | `viscosity_cells` | yes |
| `turbulence.cpp` | `clear_solid_cells` | yes |
| `turbulence.cpp` | `not_finite_cells` | yes |
| `turbulence.cpp` | `out_of_range_cells` | yes |
| `turbulence.cpp` | `not_zero_cells` | yes |
| `turbulence.cpp` | `flag_cells` | no |
| `walls.cpp` | `largest_cells` | yes |
| `walls.cpp` | `tangential_cells` | yes |

g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0; -std=c++20 -O2 -ffp-contract=off -march=x86-64-v3 -DNDEBUG. 19 of 20 blocks vectorised, with vectors of 32 bytes.
<!-- tables94 loops end -->

## Findings

- **F1, the reference's logarithm and cube are NumPy's own on a processor with AVX-512.** No earlier
  module called one: the flux, the solves, the predictor and the correction are arithmetic, which is the
  same everywhere. ADR-017 G fixes the order of sums and says nothing of elementary functions; Alex's
  decision (above) is recorded here and in `docs/SYSTEM.md`, and its text in ADR-017 is the
  orchestrator's.
- **F2, the reference's runs of record depend on the machine's vector level.** A coupled solve's eps in
  the wall cells and, under RNG, its R carry NumPy's rounding, so the same NumPy gives other last bits
  without AVX-512. The size is T6's, at rounding; level 2's band is what judges it. NumPy under
  WSL2 on this machine runs the AVX-512 loops unless told otherwise, so the Linux records of the earlier
  steps hold them.
- **F3, the initial values of the coupled solve are a compensated sum.** `initial_values` uses Python's
  `sum` over floats, which since CPython 3.12 carries Neumaier's correction; the port copies it, and the
  planted faces show a plain sum giving other bits.
- **F4, the CG rule holds on every captured step and from rest, and not on states planted over decades**
  (T4, T4b, T5). Where a field is beyond 1e-12 there the two sides' systems are the same bits, so the
  difference is the dot products' order alone, amplified by the system; no fallback was taken on one side
  only.
- **F5, (d)'s limit is the conjugate gradient iteration**, as it was for the pressure correction
  (`phase6_pressure.md`, section 6). A faster step 5 would come from that kernel.

## Deviations and limits

- **RNG is compared on planted inputs only.** The captured cases the prompt names are the standard
  variant's, and no product run uses RNG.
- **On planted systems over decades the CG rule does not hold everywhere** (T5). A count that differs
  there is recorded; every captured solve keeps it.
- **A step's counts and fallbacks are the binding's attributes** (`last_iterations`, `last_fallbacks`,
  `solves_converged`), as the reference's, and are left as they were by a call that is refused.
- **A least value of zero in a positivity failure's message may print with the other sign** where a field
  holds both zeros: `np.min`'s tie across its vector lanes is not copied.
- **The core's model and boundary refuse a width, a constant or an inlet's length or width that is not
  finite and positive**, which the reference leaves to `SimConfig`; a limiter's or a solve's number is
  read as the binding reads numbers (a bool is a `TypeError`).
- **`walls` does not hold the outlets' extrapolation**, which ADR-017 C's table lists under it: it is
  `boundary_staggered.py`'s and the outer iteration's, step 5's second half.
- **The timings are of this machine in the state T8 records**, with the local model server's process open
  and idle, not closed by the builder; no game. NumPy's AVX-512 loops are off in them.
- **The binding's calls allocate their outputs**, which the run object of step 5's second half will not;
  the core allocates nothing in a call.
- **Compiled with GCC 13.3 only.** CI's runner is the check of its own GCC, and says nothing of which
  blocks it vectorises.
- **Nothing under `src/` or `configs/` changed**, and no step needed `sudo`.

## Files

- `core/include/cfdcore/turbulence.hpp`, `walls.hpp`, `elementary.hpp`, `positions.hpp` and their
  sources; `core/tests/test_turbulence.cpp`, `test_walls.cpp`, `test_three_directions.cpp`;
  `core/python/bindings.cpp` (`KEpsilon2`, `TurbulenceBoundary2`, `PositivityFailure`, `log_law_floor`,
  `wall_viscosity`, `inflow_values`); `core/CMakeLists.txt`.
- `tests/test_core_turbulence.py`, `tests/core_mapping.py` (the model's and the boundary's construction,
  the inlet faces, the results and the positivity failure), `.github/workflows/ci.yml`.
- `docs/reports/probe94/`: `probe94.py`, `variants94.py`, `tables94.py`, `tables94.md`.
- `results/builder94/` (untracked): the captures, `replay.json`, `replay_default.json`,
  `elementary*.json`, `planted.json`, `timing_1.json` to `timing_3.json`, `plants_s*.json`,
  `vecinfo.json`, `product80_umist.yaml`, the JUnit files, the scripts and logs.
