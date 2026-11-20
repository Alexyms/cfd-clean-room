# Prompt 89b: the tables

### T12. The sessions: the machine's state and the order of the builds

Each session times the Python reference and each build in its own process, one after another. Before each process the probe samples Windows for two seconds (the busiest processes in percent of one logical processor, the processors' performance percentage, whether the machine is on mains) and records the Windows power scheme and the power mode. The calibration is the median time of `np.sqrt` over 4,096 doubles in that process, the same code whatever build is timed.

| Run | Started | Order of the builds | On mains in every process | Power scheme and mode | Busiest process in any process of the session (percent of one logical processor) | Processor performance, percent | Calibration, us (lowest to highest) |
|---|---|---|---|---|---|---|---|
| 1 | 2026-11-20T13:28:07 | `reference`, `base`, `head_o2`, `head_v3`, `v3` | yes (5 of 5) | Performance; Best performance | officeclicktorun 99, msmpeng 71, msedge 50, vmmemwsl 47 | 100 to 101 | 8.35 to 9.45 |
| 2 | 2026-11-20T13:29:16 | `reference`, `v3`, `head_v3`, `head_o2`, `base` | yes (5 of 5) | Performance; Best performance | officeclicktorun 99, msedge 71, msmpeng 51, code 24 | 100 to 101 | 8.99 to 9.33 |
| 3 | 2026-11-20T13:30:26 | `reference`, `head_v3`, `base`, `v3`, `head_o2` | yes (5 of 5) | Performance; Best performance | battle.net 132, system 103, officeclicktorun 100, agent 99 | 100 to 101 | 8.62 to 9.01 |
| 4 | 2026-11-20T13:32:43 | `head_o2`, `head_v3`, `v3` | yes (3 of 3) | Performance; Best performance | system 72, msedge 55, msmpeng 36, wmiprvse 22 | 85 to 101 | 8.67 to 8.95 |
| 5 | 2026-11-20T13:33:25 | `v3`, `head_o2`, `head_v3` | yes (3 of 3) | Performance; Best performance | system 85, msedge 63, msmpeng 42, code 26 | 86 to 100 | 8.60 to 9.00 |
| 6 | 2026-11-20T13:34:07 | `head_v3`, `v3`, `head_o2` | yes (3 of 3) | Performance; Best performance | msmpeng 55, wmiprvse 27, code 22, system 19 | 86 to 95 | 9.01 to 9.02 |
| 7 | 2026-11-20T13:34:48 | `v3`, `head_v3`, `head_o2` | yes (3 of 3) | Performance; Best performance | system 86, msedge 84, wmiprvse 56, msmpeng 48 | 88 to 100 | 8.87 to 9.45 |
| 8 | 2026-11-20T13:35:28 | `head_o2`, `v3`, `head_v3` | yes (3 of 3) | Performance; Best performance | system 97, msedge 46, msmpeng 42, wmiprvse 21 | 85 to 100 | 9.01 to 9.11 |
| 9 | 2026-11-20T13:36:10 | `head_v3`, `head_o2`, `v3` | yes (3 of 3) | Performance; Best performance | memory compression 108, system 107, vmmemwsl 48, msedge 39 | 91 to 100 | 8.33 to 9.01 |

### T13. Each kernel under the new flags against round 0's head, on the developed 320x120 frames

Medians of repeated calls through the binding, in microseconds, as T4 gives them, in sessions 1 to 3 of T12. The inputs are prompt 88's developed frame of class 0 for the implicit kernels and prompt 87's ten developed rows for the flux, whose median over the five classes is given. A sweep is the full solve less the set-up with one residual, over the solve's sweeps; a CG iteration likewise. The two ratios are the time of round 0's head at `-O2` (the build the pass's numbers were taken on) and of the core before the pass over the time of the head now.

| Kernel | Run | Reference | Before the pass | Round 0's head, `-O2` | Round 0's head, v3 | Now, v3 | `-O2` head / now | Before the pass / now |
|---|---|---|---|---|---|---|---|---|
| a Jacobi sweep | 1 | 443 | 281 | 126 | 78.9 | 112 | 1.12 | 2.50 |
| a Jacobi sweep | 2 | 410 | 267 | 145 | 114 | 111 | 1.30 | 2.40 |
| a Jacobi sweep | 3 | 506 | 286 | 156 | 94.4 | 108 | 1.45 | 2.65 |
| the set-up with one residual | 1 | 726 | 553 | 261 | 249 | 216 | 1.21 | 2.56 |
| the set-up with one residual | 2 | 887 | 527 | 300 | 213 | 220 | 1.37 | 2.40 |
| the set-up with one residual | 3 | 662 | 633 | 286 | 195 | 216 | 1.32 | 2.92 |
| the Jacobi solve | 1 | 4274 | 2798 | 1270 | 880 | 1114 | 1.14 | 2.51 |
| the Jacobi solve | 2 | 4165 | 2664 | 1457 | 1123 | 1110 | 1.31 | 2.40 |
| the Jacobi solve | 3 | 4710 | 2921 | 1537 | 950 | 1079 | 1.42 | 2.71 |
| nothing to solve | 1 | 102 | 77.8 | 122 | 69.7 | 78.7 | 1.55 | 0.99 |
| nothing to solve | 2 | 101 | 101 | 108 | 99.2 | 76.9 | 1.40 | 1.32 |
| nothing to solve | 3 | 104 | 70.1 | 136 | 69.9 | 90.3 | 1.51 | 0.78 |
| a CG iteration | 1 | 558 | 459 | 232 | 165 | 190 | 1.22 | 2.42 |
| a CG iteration | 2 | 677 | 463 | 267 | 179 | 178 | 1.50 | 2.61 |
| a CG iteration | 3 | 569 | 454 | 271 | 162 | 183 | 1.48 | 2.48 |
| the CG solve | 1 | 4422 | 3381 | 1722 | 1256 | 1439 | 1.20 | 2.35 |
| the CG solve | 2 | 5067 | 3357 | 1987 | 1403 | 1366 | 1.45 | 2.46 |
| the CG solve | 3 | 4496 | 3370 | 2021 | 1233 | 1411 | 1.43 | 2.39 |
| the flux along x | 1 | 2610 | 500 | 234 | 270 | 188 | 1.25 | 2.66 |
| the flux along x | 2 | 2474 | 516 | 250 | 264 | 182 | 1.37 | 2.83 |
| the flux along x | 3 | 2209 | 454 | 247 | 126 | 183 | 1.35 | 2.48 |
| the flux along y | 1 | 2174 | 477 | 199 | 224 | 157 | 1.27 | 3.03 |
| the flux along y | 2 | 2140 | 476 | 219 | 240 | 142 | 1.55 | 3.36 |
| the flux along y | 3 | 2112 | 461 | 238 | 115 | 151 | 1.58 | 3.06 |

The calibration of each process (T12), in microseconds, in the order reference, before the pass, round 0's head at `-O2`, round 0's head at v3, now:

- run 1: 9.28, 9.45, 9.14, 8.66, 8.35
- run 2: 9.01, 9.00, 9.33, 8.99, 8.99
- run 3: 8.95, 8.62, 8.99, 8.75, 9.01

Every timed Jacobi solve's outputs equal the reference's bit for bit in every build and session (the number of differing outputs over the five classes: base 0, head_o2 0, head_v3 0, v3 0).

The flux per class and direction, microseconds, run 1, and the range of the ratio over the ten rows in each run:

| Class | Flux x, `-O2` head | now | ratio | Flux y, `-O2` head | now | ratio |
|---|---|---|---|---|---|---|
| 0 | 238 | 183 | 1.30 | 199 | 151 | 1.32 |
| 1 | 311 | 188 | 1.65 | 203 | 160 | 1.27 |
| 2 | 226 | 189 | 1.19 | 212 | 157 | 1.35 |
| 3 | 221 | 188 | 1.17 | 195 | 163 | 1.20 |
| 4 | 234 | 182 | 1.29 | 197 | 151 | 1.31 |

Over the ten rows: run 1: 1.17 to 1.65; run 2: 1.27 to 1.69; run 3: 0.95 to 1.64.

The flux's loop (round 0's head under v3) over its block form (now), median over the ten rows in each run: 1.42, 1.57, 0.69.

### T13b. The ratios in every session that timed the three builds

Each cell is the time of round 0's head at `-O2` over the time of the head now (`-O2` head / now), and for the flux also the time of round 0's head under v3, the flux a loop of one face at a time, over the time of the head now (loop / block): above 1 the head now is the faster. Sessions 1 to 3 are T13's; 4 to 9 were taken afterwards, with the three builds only and their order rotated, because in session 3 the loop's process ran at twice its speed in every other session (the cause was not found).

| Run | sweep | set-up | CG iteration | flux x | flux y | flux x, loop / block | flux y, loop / block |
|---|---|---|---|---|---|---|---|
| 1 | 1.12 | 1.21 | 1.22 | 1.25 | 1.27 | 1.43 | 1.43 |
| 2 | 1.30 | 1.37 | 1.50 | 1.37 | 1.55 | 1.45 | 1.69 |
| 3 | 1.45 | 1.32 | 1.48 | 1.35 | 1.58 | 0.69 | 0.76 |
| 4 | 1.07 | 1.62 | 1.52 | 1.28 | 1.31 | 1.46 | 1.42 |
| 5 | 1.46 | 1.34 | 1.55 | 1.31 | 1.31 | 1.39 | 1.38 |
| 6 | 1.40 | 1.23 | 1.49 | 1.34 | 1.50 | 1.36 | 1.59 |
| 7 | 1.33 | 1.34 | 1.57 | 1.20 | 1.30 | 1.50 | 1.61 |
| 8 | 1.72 | 1.25 | 1.51 | 1.47 | 1.58 | 1.52 | 1.46 |
| 9 | 1.59 | 1.22 | 1.49 | 1.29 | 1.41 | 1.50 | 1.62 |
| median | 1.40 | 1.32 | 1.50 | 1.31 | 1.41 | 1.45 | 1.46 |
| range | 1.07 to 1.72 | 1.21 to 1.62 | 1.22 to 1.57 | 1.20 to 1.47 | 1.27 to 1.58 | 0.69 to 1.52 | 0.76 to 1.69 |

### T14. The tests under the new flags

| Tests | Cases | Passed |
|---|---|---|
| GoogleTest cases of round 0's head (eacc71c) | 79 | 79 of those present |
| GoogleTest cases added by this round | 2 | 2 |
| Parity cases of round 0's head | 3,085 | 3,085 of those present |
| Parity cases added by this round | 16 | 16 |
| Captured flux calls of prompt 87 replayed, bit for bit | 120 | 120 |
| Captured Jacobi solves of prompt 88 replayed, every output bit for bit | 100 | 100 |
| Captured CG solves of prompt 88 at the reference's count | 40 | 40 |
| the same against the reference with the core's dot products, bit for bit | 40 | 40 |

Totals: 81 GoogleTest cases and 3,101 parity cases, of which 81 and 3,101 passed. The captured CG solves' largest difference from the reference, over its largest magnitude: 1.86e-16. The build had no warning (`-Werror`: 0 warnings). The build's flags: `-Wall -Wextra -Wpedantic -O2 -ffp-contract=off -march=x86-64-v3`.

The cases added:

| Added | Cases |
|---|---|
| GoogleTest `Grid.SolidFlagsAreMasksOfTheSolidCells` | 1 (passes) |
| GoogleTest `Lanes.MasksWhereNotRefuseFlagsOfAnotherLength` | 1 (passes) |
| `test_advective_flux_puts_nans_at_the_references_faces` | 12 |
| `test_implicit_a_nan_input_keeps_the_solve_running_to_its_cap` | 4 |

Round 0's head under v3 (the flux still a loop), the checks of this round: all pass.

### T15. The loops GCC vectorises, round 1

`g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`, `-fopt-info-vec-optimized`, per source file: the number of loops vectorised whole and of straight-line parts, each counted once for its place in the source, however many vector widths GCC vectorised it at.

| Source | Build | Loops vectorised | Vector width, bytes | Straight-line parts |
|---|---|---|---|---|
| `src/scalar.cpp` | before the pass (3c8d1ec), `-O2` | 0 | none | 3 |
| `src/scalar.cpp` | round 0's head (eacc71c), `-O2` | 10 | 16 | 6 |
| `src/scalar.cpp` | round 0's head under `-march=x86-64-v3` | 10 | 32 | 5 |
| `src/scalar.cpp` | the head now (`-march=x86-64-v3`, the flux in blocks) | 11 | 32 | 5 |
| `src/reduce.cpp` | before the pass (3c8d1ec), `-O2` | 2 | 16 | 0 |
| `src/reduce.cpp` | round 0's head (eacc71c), `-O2` | 2 | 16 | 0 |
| `src/reduce.cpp` | round 0's head under `-march=x86-64-v3` | 2 | 32 | 0 |
| `src/reduce.cpp` | the head now (`-march=x86-64-v3`, the flux in blocks) | 2 | 32 | 0 |

The block functions of `scalar.cpp` whose loop GCC vectorised under the new flags: `boundary_cells`, `direction_cells`, `product_cells`, `quick_cells`, `residual_cells`, `restart_cells`, `setup_cells`, `start_cells`, `sweep_cells`, `update_cells`, `upwind_cells`. Every block function of the file is among them.

Round 0's head under the same flags: `boundary_cells`, `direction_cells`, `product_cells`, `residual_cells`, `restart_cells`, `setup_cells`, `start_cells`, `sweep_cells`, `update_cells`, `upwind_cells`; not vectorised: none.

### T16. The planted defects, round 1: which tests catch each

39 defects, each built in a copy of the head under the new flags: 39 fail a GoogleTest case or a parity case or crash a run; 0 pass both. The controls, before and after, pass.

| Defect | What it changes | GoogleTest (CTest) | Parity (pytest) |
|---|---|---|---|
| `control` | the unchanged source | 100% tests passed, 0 tests failed out of 81 | 3101 passed in 40.09s |
| `walk_one_block_too_many` | a walk takes a whole block where fewer cells than a block are left | 99% tests passed, 1 tests failed out of 81 [failed: Lanes.AWalkVisitsEveryCellOnceInOrder] | the run ended without a summary, exit code -6: a crash |
| `walk_skips_the_cells_left` | a walk stops after its whole blocks | 74% tests passed, 21 tests failed out of 81 [failed: Lanes.AWalkVisitsEveryCellOnceInOrder, AdvectiveFlux.AUniformFieldGivesTheFluxTimesItsValue, AdvectiveFlux.AMonotoneFieldGivesNoNewExtremum, AdvectiveFlux.ALineOfCellsIsEachRowOfAFieldConstantAcrossIt, AdvectiveFlux.AFarNodeInASolidCellIsReadAsTheUpstreamValue, AdvectiveFlux.TheFarSolidRuleReadsQuickAsTheUpstreamValueToo, ImplicitStep.TheHandSystemsLeftoverIsTheChangeTheSinkDoesNotAccountFor, ImplicitStep.AUniformFieldWithNoSinkIsUnchanged, ImplicitStep.ANonNegativeStarGivesNonNegativeJacobiIterates, ImplicitStep.HeldAndSolidCellsKeepTheirValues, ImplicitStep.ConjugateGradientsReachTheDenseSolution, ImplicitSystem.TheDiagonalIsWestEastSouthNorthThenTheSink, ImplicitSystem.ANanIsSomethingToSolveAndMakesTheStopNan, Jacobi.OneSweepFromCStarIsTheHandSystemsFirstIterate, Jacobi.ConvergesToTheHandSystemsSolution, Jacobi.TheStopHoldsWithEqualityAndNotBelowIt, Jacobi.ABoundSweepsOnUntilTheLeftoverIsInsideIt, ConjugateGradient.OneIterationIsTheHandSystemsClosedForm, ConjugateGradient.ConvergesToTheHandSystemsSolutionInTwoOrThreeIterations, ConjugateGradient.TheStopHoldsWithEqualityAtTheStart, ConjugateGradient.ABoundRestartsFromTheTrueResidualUntilTheLeftoverIsInsideIt] | 2362 failed, 739 passed in 61.71s (0:01:01) [failed: test_advective_flux_accepts_an_out_that_touches_no_input 2, test_advective_flux_accepts_consistent_arguments 1, test_advective_flux_equals_the_reference_bit_for_bit 156, test_advective_flux_is_bit_for_bit_at_every_row_length 623, test_advective_flux_puts_nans_at_the_references_faces 12, test_implicit_a_nan_input_keeps_the_solve_running_to_its_cap 4, test_implicit_cg_fallback_is_the_jacobi_solve_bit_for_bit 10, test_implicit_cg_iterations_under_a_fallback_are_the_cgs_own 5, test_implicit_cg_meets_prompt_88s_rule 66, test_implicit_cg_positivity_is_read_in_the_cells_it_solves_for_only 5, test_implicit_cg_with_a_leftover_bound_and_the_cores_dot_products_is_bit_for_bit 45, test_implicit_cg_with_a_leftover_bound_meets_prompt_88s_rule 31, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit 90, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 335, test_implicit_jacobi_equals_the_reference_bit_for_bit 320, test_implicit_jacobi_is_bit_for_bit_at_every_row_length 335, test_implicit_jacobi_with_a_bound_is_bit_for_bit_at_every_row_length 168, test_implicit_stop_reads_b_in_every_cell 2, test_implicit_system_is_the_references_set_up_at_every_row_length 152] |
| `stencil_end_as_interior` | a row's last cell goes through the blocks, reading the cell beyond the row | 100% tests passed, 0 tests failed out of 81 | 640 failed, 2461 passed in 45.89s [failed: test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 218, test_implicit_jacobi_equals_the_reference_bit_for_bit 80, test_implicit_jacobi_is_bit_for_bit_at_every_row_length 230, test_implicit_jacobi_with_a_bound_is_bit_for_bit_at_every_row_length 112] |
| `end_cell_as_a_cell_between` | a row's last cell reads its east neighbour as the next element in memory, not the zero | 100% tests passed, 0 tests failed out of 81 | 640 failed, 2461 passed in 46.39s [failed: test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 222, test_implicit_jacobi_equals_the_reference_bit_for_bit 80, test_implicit_jacobi_is_bit_for_bit_at_every_row_length 223, test_implicit_jacobi_with_a_bound_is_bit_for_bit_at_every_row_length 115] |
| `padding_east_as_cell` | the zero read beyond a row's east end taken as the cell's own value | 100% tests passed, 0 tests failed out of 81 | 847 failed, 2254 passed in 47.99s [failed: test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 306, test_implicit_jacobi_equals_the_reference_bit_for_bit 80, test_implicit_jacobi_is_bit_for_bit_at_every_row_length 304, test_implicit_jacobi_with_a_bound_is_bit_for_bit_at_every_row_length 157] |
| `padding_west_as_cell` | the zero read beyond a row's west end taken as the cell's own value | 100% tests passed, 0 tests failed out of 81 | 854 failed, 2247 passed in 50.06s [failed: test_implicit_a_nan_input_keeps_the_solve_running_to_its_cap 2, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 306, test_implicit_jacobi_equals_the_reference_bit_for_bit 80, test_implicit_jacobi_is_bit_for_bit_at_every_row_length 303, test_implicit_jacobi_with_a_bound_is_bit_for_bit_at_every_row_length 163] |
| `padding_north_as_cell` | the zeros read above the last row taken as the row's own values | 100% tests passed, 0 tests failed out of 81 | 912 failed, 2189 passed in 48.84s [failed: test_implicit_a_nan_input_keeps_the_solve_running_to_its_cap 4, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 334, test_implicit_jacobi_equals_the_reference_bit_for_bit 80, test_implicit_jacobi_is_bit_for_bit_at_every_row_length 328, test_implicit_jacobi_with_a_bound_is_bit_for_bit_at_every_row_length 166] |
| `padding_south_as_cell` | the zeros read below the first row taken as the row's own values | 100% tests passed, 0 tests failed out of 81 | 908 failed, 2193 passed in 48.60s [failed: test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 332, test_implicit_jacobi_equals_the_reference_bit_for_bit 80, test_implicit_jacobi_is_bit_for_bit_at_every_row_length 329, test_implicit_jacobi_with_a_bound_is_bit_for_bit_at_every_row_length 167] |
| `sweep_update_masked_as_residual` | the Jacobi update zeroed in held cells as well as SOLID ones | 99% tests passed, 1 tests failed out of 81 [failed: ImplicitStep.HeldAndSolidCellsKeepTheirValues] | 328 failed, 2773 passed in 43.71s [failed: test_implicit_cg_fallback_is_the_jacobi_solve_bit_for_bit 10, test_implicit_cg_iterations_under_a_fallback_are_the_cgs_own 5, test_implicit_cg_meets_prompt_88s_rule 3, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit 3, test_implicit_jacobi_equals_the_reference_bit_for_bit 160, test_implicit_jacobi_is_bit_for_bit_at_every_row_length 147] |
| `residual_unmasked` | the residual not zeroed in the cells that are not unknowns | 98% tests passed, 2 tests failed out of 81 [failed: ImplicitStep.TheNeighbourSumIsEastWestNorthSouth, ImplicitStep.HeldAndSolidCellsKeepTheirValues] | 1039 failed, 2062 passed in 52.64s [failed: test_implicit_cg_fallback_is_the_jacobi_solve_bit_for_bit 10, test_implicit_cg_iterations_under_a_fallback_are_the_cgs_own 5, test_implicit_cg_meets_prompt_88s_rule 4, test_implicit_cg_positivity_is_read_in_the_cells_it_solves_for_only 5, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit 45, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 310, test_implicit_jacobi_equals_the_reference_bit_for_bit 192, test_implicit_jacobi_is_bit_for_bit_at_every_row_length 311, test_implicit_jacobi_with_a_bound_is_bit_for_bit_at_every_row_length 155, test_implicit_stop_reads_b_in_every_cell 2] |
| `product_unmasked` | the CG's product not zeroed in the cells that are not unknowns | 100% tests passed, 0 tests failed out of 81 | 364 failed, 2737 passed in 46.22s [failed: test_implicit_cg_iterations_under_a_fallback_are_the_cgs_own 5, test_implicit_cg_positivity_is_read_in_the_cells_it_solves_for_only 5, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit 45, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 309] |
| `neighbour_order` | the neighbour sum's terms along y taken south before north | 99% tests passed, 1 tests failed out of 81 [failed: ImplicitStep.TheNeighbourSumIsEastWestNorthSouth] | 789 failed, 2312 passed in 54.01s [failed: test_implicit_cg_fallback_is_the_jacobi_solve_bit_for_bit 10, test_implicit_cg_iterations_under_a_fallback_are_the_cgs_own 5, test_implicit_cg_positivity_is_read_in_the_cells_it_solves_for_only 5, test_implicit_cg_with_a_leftover_bound_and_the_cores_dot_products_is_bit_for_bit 40, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit 90, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 124, test_implicit_jacobi_equals_the_reference_bit_for_bit 320, test_implicit_jacobi_is_bit_for_bit_at_every_row_length 131, test_implicit_jacobi_with_a_bound_is_bit_for_bit_at_every_row_length 62, test_implicit_stop_reads_b_in_every_cell 2] |
| `diagonal_order` | the diagonal's terms along y taken north before south | 99% tests passed, 1 tests failed out of 81 [failed: ImplicitSystem.TheDiagonalIsWestEastSouthNorthThenTheSink] | 1024 failed, 2077 passed in 57.31s [failed: test_implicit_a_nan_input_keeps_the_solve_running_to_its_cap 2, test_implicit_cg_fallback_is_the_jacobi_solve_bit_for_bit 6, test_implicit_cg_iterations_under_a_fallback_are_the_cgs_own 5, test_implicit_cg_positivity_is_read_in_the_cells_it_solves_for_only 3, test_implicit_cg_with_a_leftover_bound_and_the_cores_dot_products_is_bit_for_bit 33, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit 74, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 181, test_implicit_jacobi_equals_the_reference_bit_for_bit 320, test_implicit_jacobi_is_bit_for_bit_at_every_row_length 179, test_implicit_jacobi_with_a_bound_is_bit_for_bit_at_every_row_length 96, test_implicit_stop_reads_b_in_every_cell 2, test_implicit_system_equals_the_references_set_up 10, test_implicit_system_is_the_references_set_up_at_every_row_length 113] |
| `stop_from_one_lane` | the stop's largest |b| taken from the first of the eight running maxima | 99% tests passed, 1 tests failed out of 81 [failed: ImplicitStep.ConjugateGradientsReachTheDenseSolution] | 578 failed, 2523 passed in 52.21s [failed: test_implicit_cg_fallback_is_the_jacobi_solve_bit_for_bit 1, test_implicit_cg_iterations_under_a_fallback_are_the_cgs_own 4, test_implicit_cg_positivity_is_read_in_the_cells_it_solves_for_only 1, test_implicit_cg_with_a_leftover_bound_and_the_cores_dot_products_is_bit_for_bit 11, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit 28, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 93, test_implicit_jacobi_equals_the_reference_bit_for_bit 124, test_implicit_jacobi_is_bit_for_bit_at_every_row_length 147, test_implicit_jacobi_with_a_bound_is_bit_for_bit_at_every_row_length 76, test_implicit_system_equals_the_references_set_up 10, test_implicit_system_is_the_references_set_up_at_every_row_length 83] |
| `stop_nan_dropped` | a NaN |b| left out of the stop | 99% tests passed, 1 tests failed out of 81 [failed: ImplicitSystem.ANanIsSomethingToSolveAndMakesTheStopNan] | 3101 passed in 42.66s |
| `outside_stop_strict` | the max-norm stop met strictly below the stop, not at it | 94% tests passed, 5 tests failed out of 81 [failed: ImplicitStep.TheNeighbourSumIsEastWestNorthSouth, Jacobi.TheStopHoldsWithEqualityAndNotBelowIt, Jacobi.ABoundSweepsOnUntilTheLeftoverIsInsideIt, ConjugateGradient.TheStopHoldsWithEqualityAtTheStart, ConjugateGradient.ABoundRestartsFromTheTrueResidualUntilTheLeftoverIsInsideIt] | 23 failed, 3078 passed in 42.12s [failed: test_implicit_a_zero_field_meets_the_stop_and_a_zero_bound_with_equality 20, test_implicit_jacobi_is_bit_for_bit_at_every_row_length 3] |
| `jacobi_last_copy_dropped` | the Jacobi solve's field left where its last sweep wrote it | 96% tests passed, 3 tests failed out of 81 [failed: ImplicitStep.TheHandSystemsLeftoverIsTheChangeTheSinkDoesNotAccountFor, ImplicitStep.AUniformFieldWithNoSinkIsUnchanged, Jacobi.TheStopHoldsWithEqualityAndNotBelowIt] | 532 failed, 2569 passed in 45.40s [failed: test_implicit_cg_fallback_is_the_jacobi_solve_bit_for_bit 1, test_implicit_cg_iterations_under_a_fallback_are_the_cgs_own 3, test_implicit_cg_meets_prompt_88s_rule 5, test_implicit_cg_with_a_leftover_bound_and_the_cores_dot_products_is_bit_for_bit 16, test_implicit_cg_with_a_leftover_bound_meets_prompt_88s_rule 6, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit 6, test_implicit_jacobi_equals_the_reference_bit_for_bit 215, test_implicit_jacobi_is_bit_for_bit_at_every_row_length 174, test_implicit_jacobi_with_a_bound_is_bit_for_bit_at_every_row_length 106] |
| `cg_dot_of_the_wrong_row` | the CG's r.z taken as z.z | 98% tests passed, 2 tests failed out of 81 [failed: ImplicitStep.ConjugateGradientsReachTheDenseSolution, ConjugateGradient.ConvergesToTheHandSystemsSolutionInTwoOrThreeIterations] | 452 failed, 2649 passed in 50.62s [failed: test_implicit_cg_iterations_under_a_fallback_are_the_cgs_own 5, test_implicit_cg_positivity_is_read_in_the_cells_it_solves_for_only 5, test_implicit_cg_with_a_leftover_bound_and_the_cores_dot_products_is_bit_for_bit 25, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit 90, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 327] |
| `cg_z_from_the_old_residual` | the CG's z formed from the residual before its update | 98% tests passed, 2 tests failed out of 81 [failed: ImplicitStep.ConjugateGradientsReachTheDenseSolution, ConjugateGradient.ConvergesToTheHandSystemsSolutionInTwoOrThreeIterations] | 452 failed, 2649 passed in 51.18s [failed: test_implicit_cg_iterations_under_a_fallback_are_the_cgs_own 5, test_implicit_cg_positivity_is_read_in_the_cells_it_solves_for_only 5, test_implicit_cg_with_a_leftover_bound_and_the_cores_dot_products_is_bit_for_bit 25, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit 90, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 327] |
| `cg_direction_fused` | the direction updated as fma(p, beta, z), the contraction -ffp-contract=off forbids | 100% tests passed, 0 tests failed out of 81 | 208 failed, 2893 passed in 43.41s [failed: test_implicit_cg_positivity_is_read_in_the_cells_it_solves_for_only 3, test_implicit_cg_with_a_leftover_bound_and_the_cores_dot_products_is_bit_for_bit 16, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit 65, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 124] |
| `block_dot_left_to_right` | a row's dot product summed left to right, not in pairwise order | 99% tests passed, 1 tests failed out of 81 [failed: BlockDot.TheBlocksAddedInOrderAreBlockedDot] | 282 failed, 2819 passed in 44.52s [failed: test_implicit_cg_positivity_is_read_in_the_cells_it_solves_for_only 4, test_implicit_cg_with_a_leftover_bound_and_the_cores_dot_products_is_bit_for_bit 23, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit 86, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 169] |
| `flux_end_node_ignores_inflow` | a line's low boundary node read as the cell's value whatever the flux | 100% tests passed, 0 tests failed out of 81 | 198 failed, 2903 passed in 43.88s [failed: test_advective_flux_equals_the_reference_bit_for_bit 124, test_advective_flux_is_bit_for_bit_at_every_row_length 70, test_advective_flux_puts_nans_at_the_references_faces 4] |
| `flux_weights_of_the_first_face` | every face of a row along x given the row's first face's weights | 100% tests passed, 0 tests failed out of 81 | 95 failed, 3006 passed in 40.92s [failed: test_advective_flux_equals_the_reference_bit_for_bit 52, test_advective_flux_is_bit_for_bit_at_every_row_length 42, test_advective_flux_puts_nans_at_the_references_faces 1] |
| `flux_far_solid_dropped` | a far node inside a SOLID cell read as it is | 98% tests passed, 2 tests failed out of 81 [failed: AdvectiveFlux.AFarNodeInASolidCellIsReadAsTheUpstreamValue, AdvectiveFlux.TheFarSolidRuleReadsQuickAsTheUpstreamValueToo] | 244 failed, 2857 passed in 41.93s [failed: test_advective_flux_equals_the_reference_bit_for_bit 116, test_advective_flux_is_bit_for_bit_at_every_row_length 123, test_advective_flux_puts_nans_at_the_references_faces 5] |
| `flux_high_boundary_sign` | the high boundary node takes the inflow value where the flux leaves | 99% tests passed, 1 tests failed out of 81 [failed: AdvectiveFlux.BoundaryFacesTakeTheInflowOnlyWhereTheFluxEnters] | 980 failed, 2121 passed in 43.02s [failed: test_advective_flux_equals_the_reference_bit_for_bit 312, test_advective_flux_is_bit_for_bit_at_every_row_length 656, test_advective_flux_puts_nans_at_the_references_faces 12] |
| `flux_upwind_downstream` | the upwind value taken from the downstream cell | 100% tests passed, 0 tests failed out of 81 | 420 failed, 2681 passed in 40.14s [failed: test_advective_flux_equals_the_reference_bit_for_bit 130, test_advective_flux_is_bit_for_bit_at_every_row_length 284, test_advective_flux_puts_nans_at_the_references_faces 6] |
| `flux_vents_dropped` | the vents' second pass not made | 99% tests passed, 1 tests failed out of 81 [failed: AdvectiveFlux.AVentTakesTheInflowEnteringAndTheRoomCellLeaving] | 342 failed, 2759 passed in 39.69s [failed: test_advective_flux_equals_the_reference_bit_for_bit 119, test_advective_flux_is_bit_for_bit_at_every_row_length 223] |
| `flux_last_row_boundary_dropped` | the high boundary faces of the lines along y not written | 99% tests passed, 1 tests failed out of 81 [failed: AdvectiveFlux.AUniformFieldGivesTheFluxTimesItsValue] | 499 failed, 2602 passed in 41.59s [failed: test_advective_flux_accepts_consistent_arguments 1, test_advective_flux_equals_the_reference_bit_for_bit 156, test_advective_flux_is_bit_for_bit_at_every_row_length 336, test_advective_flux_puts_nans_at_the_references_faces 6] |
| `select_swapped` | lanes::select returns the other value | 90% tests passed, 8 tests failed out of 81 [failed: Lanes.SelectPassesTheChosenValuesBits, NumpyMinMax.TiesReturnTheSecondArgument, NumpyMinMax.ANanInEitherIsANan, LimitedFaceValue.NoDownstreamDifferenceIsTheUpstreamValue, LimitedFaceValue.HandValuesOfEachBranch, AdvectiveFlux.BoundaryFacesTakeTheInflowOnlyWhereTheFluxEnters, AdvectiveFlux.AFarNodeInASolidCellIsReadAsTheUpstreamValue, AdvectiveFlux.TheFarSolidRuleReadsQuickAsTheUpstreamValueToo] | 1000 failed, 2101 passed in 44.95s [failed: test_advective_flux_equals_the_reference_bit_for_bit 312, test_advective_flux_is_bit_for_bit_at_every_row_length 672, test_advective_flux_puts_nans_at_the_references_faces 12, test_limited_face_values_equal_the_reference_bit_for_bit 4] |
| `keep_by_multiplication` | lanes::keep multiplies by one or zero where it should pass or drop bits | 99% tests passed, 1 tests failed out of 81 [failed: Lanes.KeepGivesPositiveZeroWhereTheMaskIsNotSet] | 125 failed, 2976 passed in 44.22s [failed: test_implicit_a_nan_input_keeps_the_solve_running_to_its_cap 4, test_implicit_cg_meets_prompt_88s_rule 1, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit 24, test_implicit_jacobi_equals_the_reference_bit_for_bit 96] |
| `outside_stop_nan_passes` | the stop met where the residual is not above it, so a NaN residual meets it | 100% tests passed, 0 tests failed out of 81 | 4 failed, 3097 passed in 40.16s [failed: test_implicit_a_nan_input_keeps_the_solve_running_to_its_cap 4] |
| `masks_where_not_unchecked` | masks_where_not reads the flags past the masks' length without a refusal | 99% tests passed, 1 tests failed out of 81 [failed: Lanes.MasksWhereNotRefuseFlagsOfAnotherLength] | 3101 passed in 39.52s |
| `solid_flags_inverted` | the grid's SOLID masks set where a cell is not SOLID | 91% tests passed, 7 tests failed out of 81 [failed: Grid.SolidIsTheHandedCode, Grid.SolidFlagsAreMasksOfTheSolidCells, Grid.TheGridCopiesItsInputs, Grid.ALineOfCells, AdvectiveFlux.AFarNodeInASolidCellIsReadAsTheUpstreamValue, AdvectiveFlux.TheFarSolidRuleReadsQuickAsTheUpstreamValueToo, AdvectiveFlux.AVentTakesTheInflowEnteringAndTheRoomCellLeaving] | 573 failed, 2528 passed in 44.29s [failed: test_advective_flux_equals_the_reference_bit_for_bit 200, test_advective_flux_is_bit_for_bit_at_every_row_length 367, test_advective_flux_puts_nans_at_the_references_faces 6] |
| `limited_zero_difference_unmasked` | the clamp's value kept where the downstream difference is zero | 96% tests passed, 3 tests failed out of 81 [failed: LimitedFaceValue.NoDownstreamDifferenceIsTheUpstreamValue, LimitedFaceValue.TheFaceValueLiesBetweenTheUpstreamAndDownstreamCells, AdvectiveFlux.AUniformFieldGivesTheFluxTimesItsValue] | 253 failed, 2848 passed in 41.94s [failed: test_advective_flux_accepts_an_out_that_touches_no_input 2, test_advective_flux_accepts_consistent_arguments 1, test_advective_flux_equals_the_reference_bit_for_bit 128, test_advective_flux_is_bit_for_bit_at_every_row_length 119, test_limited_face_values_equal_the_reference_bit_for_bit 3] |
| `limited_nan_difference_is_zero` | a NaN downstream difference read as a zero one | 100% tests passed, 0 tests failed out of 81 | 4 failed, 3097 passed in 39.64s [failed: test_advective_flux_puts_nans_at_the_references_faces 4] |
| `np_minimum_takes_the_first_of_a_tie` | np_minimum returns the first of two equal values, where NumPy returns the second | 99% tests passed, 1 tests failed out of 81 [failed: NumpyMinMax.TiesReturnTheSecondArgument] | 3 failed, 3098 passed in 39.72s [failed: test_limited_face_values_equal_the_reference_bit_for_bit 3] |
| `flux_weights_sign_swapped` | the far node's QUICK weight taken from the other direction's weights | 100% tests passed, 0 tests failed out of 81 | 190 failed, 2911 passed in 41.33s [failed: test_advective_flux_equals_the_reference_bit_for_bit 112, test_advective_flux_is_bit_for_bit_at_every_row_length 75, test_advective_flux_puts_nans_at_the_references_faces 3] |
| `flux_far_node_from_the_other_side` | the far node read from the downstream side of the face | 99% tests passed, 1 tests failed out of 81 [failed: AdvectiveFlux.AFarNodeInASolidCellIsReadAsTheUpstreamValue] | 392 failed, 2709 passed in 42.84s [failed: test_advective_flux_equals_the_reference_bit_for_bit 130, test_advective_flux_is_bit_for_bit_at_every_row_length 256, test_advective_flux_puts_nans_at_the_references_faces 6] |
| `flux_far_solid_from_the_other_side` | the far cell's SOLID flag read from the downstream side of the face | 98% tests passed, 2 tests failed out of 81 [failed: AdvectiveFlux.AFarNodeInASolidCellIsReadAsTheUpstreamValue, AdvectiveFlux.TheFarSolidRuleReadsQuickAsTheUpstreamValueToo] | 276 failed, 2825 passed in 41.39s [failed: test_advective_flux_equals_the_reference_bit_for_bit 120, test_advective_flux_is_bit_for_bit_at_every_row_length 150, test_advective_flux_puts_nans_at_the_references_faces 6] |
| `control_end` | the unchanged source, after every plant | 100% tests passed, 0 tests failed out of 81 | 3101 passed in 40.00s |

### T17. The whole suite, round 1

| Platform | Tests | Passed | Failed | Errors | Skipped | Wall time, s |
|---|---|---|---|---|---|---|
| Windows, `.venv`, the core not built | 3,102 | 3,101 | 0 | 0 | 1 | 1,599 |
| WSL2, the core built, `CFD_CORE_REQUIRED=1` | 6,202 | 6,202 | 0 | 0 | 0 | 1,336 |
