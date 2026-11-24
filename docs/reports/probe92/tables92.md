# Prompt 92's tables

### T0. The predictions, scored by the rule committed with them

| Prediction | Measured | Verdict |
|---|---|---|
| (a) every planted and captured `predict` call equals the reference's bit for bit | 40 of 40 captured `predict` calls with no differing bit in the four outputs (T3, with 30 `stencil_viscosity` calls beside them); 756 of 756 planted `predict` cases pass, each holding one call or several (T1) | holds |
| (b) the set-up's fixed quantities equal the reference's per-call values on every mesh | 118 of 118 cases pass, nine meshes under two boundaries and the small grids of every row length (T1) | holds |
| (c) `predict` on the 320x120 state at least five times faster per call | 11.06 times in run 1 (run 2: 11.03; run 3: 10.41) (T5) | holds |

### T1. The parity cases by function

| Test | Cases | Passed | Failed | Skipped |
|---|---|---|---|---|
| `test_predict_equals_the_reference_bit_for_bit` | 576 | 576 | 0 | 0 |
| `test_predict_with_planted_edges_equals_the_reference_bit_for_bit` | 72 | 72 | 0 | 0 |
| `test_the_sweeps_keep_the_vents_values` | 6 | 6 | 0 | 0 |
| `test_predict_is_bit_for_bit_at_every_row_length` | 100 | 100 | 0 | 0 |
| `test_one_predictor_through_many_calls_is_each_calls_own` | 1 | 1 | 0 | 0 |
| `test_the_planted_inputs_reach_every_rule` | 1 | 1 | 0 | 0 |
| `test_a_nan_lands_where_the_references_does` | 1 | 1 | 0 | 0 |
| `test_a_face_without_a_flux_takes_the_references_stencil` | 32 | 32 | 0 | 0 |
| `test_the_fixed_quantities_equal_the_references_per_call_values` | 18 | 18 | 0 | 0 |
| `test_the_fixed_quantities_at_every_row_length` | 100 | 100 | 0 | 0 |
| `test_the_coefficients_equal_the_references` | 20 | 20 | 0 | 0 |
| `test_stencil_viscosity_equals_the_reference_bit_for_bit` | 18 | 18 | 0 | 0 |
| `test_stencil_viscosity_is_bit_for_bit_at_every_row_length` | 20 | 20 | 0 | 0 |
| `test_predict_refuses_what_the_reference_refuses` | 48 | 48 | 0 | 0 |
| `test_predict_accepts_the_arguments_the_refusals_spoil` | 1 | 1 | 0 | 0 |
| `test_stencil_viscosity_refuses_what_the_reference_refuses` | 14 | 14 | 0 | 0 |
| `test_predict_refuses_a_transient_argument` | 8 | 8 | 0 | 0 |
| `test_predict_refuses_a_field_it_cannot_read` | 12 | 12 | 0 | 0 |
| `test_predict_refuses_a_viscosity_it_cannot_read` | 9 | 9 | 0 | 0 |
| `test_the_constructor_refuses_wrong_arguments` | 129 | 129 | 0 | 0 |
| `test_the_constructor_accepts_the_arguments_the_refusals_spoil` | 1 | 1 | 0 | 0 |
| `test_the_constructor_refuses_fewer_than_two_cells_as_the_reference_does` | 3 | 3 | 0 | 0 |
| `test_fixed_and_coefficients_refuse_names_and_components_they_do_not_have` | 1 | 1 | 0 | 0 |
| `test_the_binding_needs_nothing_from_src` | 1 | 1 | 0 | 0 |
| `test_the_outputs_are_new_arrays_and_the_inputs_untouched` | 1 | 1 | 0 | 0 |
| `test_the_meshes_hold_what_the_cases_need` | 1 | 1 | 0 | 0 |
| `tests/test_core_momentum.py` | 1,194 | 1,194 | 0 | 0 |

### T1b. CTest, the parity modules and the suites

| Tests | Cases | Passed | Failed | Skipped |
|---|---|---|---|---|
| CTest, suite `MomentumPredict` | 5 | 5 | 0 | 0 |
| CTest, suite `MomentumEquations` | 5 | 5 | 0 | 0 |
| CTest, suite `MomentumViscosity` | 3 | 3 | 0 | 0 |
| CTest, suite `MomentumOneDirection` | 4 | 4 | 0 | 0 |
| CTest, suite `MomentumPairs` | 1 | 1 | 0 | 0 |
| CTest, suite `MomentumSetup` | 2 | 2 | 0 | 0 |
| CTest, all suites | 142 | 142 | 0 | 0 |
| the three parity modules, WSL2, `CFD_CORE_REQUIRED=1` | 4,694 | 4,677 | 0 | 17 |
| the whole suite, WSL2, the core built | 7,795 | 7,778 | 0 | 17 |
| the whole suite, Windows, the core not built | 3,104 | 3,101 | 0 | 3 |

### T2. The captured calls

| Case | Warm-started from | predict calls | Distinct | Sweeps | Viscosity field | Wall viscosity | Largest change of u over the calls, m/s |
|---|---|---|---|---|---|---|---|
| 80x30 product room | `results/builder81/placement/dust.h5` | 10 | 10 | 10 | yes | yes | 1.2e-09 |
| 80x30 vents room | `results/builder90/vents80_flow.npz` | 10 | 10 | 10 | yes | yes | 7.6e-12 |
| 40x40 cavity | its own solve from rest, to its stop (3,435 outer, error_estimate_and_continuity) | 10 | 10 | 1 | no | no | 3.2e-08 |
| 216x72 Annex 20 room | its own solve from rest, to the cap (400 outer, max_simple_iter) | 10 | 10 | 10 | yes | yes | 2.9e-01 |

### T3. The captured calls replayed through the core, by output

| Case | Call | Output | Calls | Calls with a differing bit | Entries that differ | Calls the reference does not reproduce | Moving vent faces |
|---|---|---|---|---|---|---|---|
| 80x30 product room | `predict` | u* | 10 | 0 | 0 | 0 | 0 |
| 80x30 product room | `predict` | v* | 10 | 0 | 0 | 0 | 0 |
| 80x30 product room | `predict` | a_P u | 10 | 0 | 0 | 0 | 0 |
| 80x30 product room | `predict` | a_P v | 10 | 0 | 0 | 0 | 0 |
| 80x30 product room | `stencil_viscosity` | u's faces | 10 | 0 | 0 | 0 | - |
| 80x30 product room | `stencil_viscosity` | v's faces | 10 | 0 | 0 | 0 | - |
| 80x30 vents room | `predict` | u* | 10 | 0 | 0 | 0 | 220 |
| 80x30 vents room | `predict` | v* | 10 | 0 | 0 | 0 | 220 |
| 80x30 vents room | `predict` | a_P u | 10 | 0 | 0 | 0 | 220 |
| 80x30 vents room | `predict` | a_P v | 10 | 0 | 0 | 0 | 220 |
| 80x30 vents room | `stencil_viscosity` | u's faces | 10 | 0 | 0 | 0 | - |
| 80x30 vents room | `stencil_viscosity` | v's faces | 10 | 0 | 0 | 0 | - |
| 40x40 cavity | `predict` | u* | 10 | 0 | 0 | 0 | 0 |
| 40x40 cavity | `predict` | v* | 10 | 0 | 0 | 0 | 0 |
| 40x40 cavity | `predict` | a_P u | 10 | 0 | 0 | 0 | 0 |
| 40x40 cavity | `predict` | a_P v | 10 | 0 | 0 | 0 | 0 |
| 216x72 Annex 20 room | `predict` | u* | 10 | 0 | 0 | 0 | 0 |
| 216x72 Annex 20 room | `predict` | v* | 10 | 0 | 0 | 0 | 0 |
| 216x72 Annex 20 room | `predict` | a_P u | 10 | 0 | 0 | 0 | 0 |
| 216x72 Annex 20 room | `predict` | a_P v | 10 | 0 | 0 | 0 | 0 |
| 216x72 Annex 20 room | `stencil_viscosity` | u's faces | 10 | 0 | 0 | 0 | - |
| 216x72 Annex 20 room | `stencil_viscosity` | v's faces | 10 | 0 | 0 | 0 | - |

### T4. The planted defects: which nets fail on each

| Plant | What it changes | Fails |
|---|---|---|
| `control` | no edit: every test passes | nothing, as it must |
| `deferred_sign` | the deferred correction added to the source, not subtracted | CTest (1); parity (807 cases of 9 tests); replay (product80, vents80, cavity40, annex216) |
| `corner_means_swapped` | a corner's viscosity by the arithmetic mean across the flux and the harmonic along the face | CTest (1); parity (497 cases of 11 tests); replay (product80, vents80, annex216) |
| `wall_missed_at_corner` | the wall stencil only where the neighbour face lies between two SOLID cells, not at a corner | CTest (3); parity (865 cases of 13 tests); replay (product80, vents80) |
| `vent_overwritten` | a vent's stored value overwritten by the sweep with zero | CTest (1); parity (391 cases of 6 tests); replay (vents80) |
| `upwind_from_downstream` | the streamwise upwind coefficient taken from the downstream side | CTest (2); parity (827 cases of 10 tests); replay (product80, vents80, cavity40, annex216) |
| `diagonal_without_mass_flux` | the diagonal's net mass flux dropped | CTest (2); parity (827 cases of 10 tests); replay (product80, vents80, cavity40, annex216) |
| `relaxation_on_new_iterate` | the relaxation term of every sweep on that sweep's iterate, not on the call's input | CTest (3); parity (604 cases of 8 tests); replay (product80, vents80, annex216) |
| `one_sweep_fewer` | one sweep fewer than asked, where more than one is | CTest (2); parity (628 cases of 8 tests); replay (product80, vents80, annex216) |
| `upwind_from_downstream_t` | the transverse upwind coefficient taken from the downstream side | CTest (1); parity (827 cases of 10 tests); replay (product80, vents80, cavity40, annex216) |
| `transverse_flux_by_mean` | the transverse mass flux from the mean of the two half cells, not width-weighted | parity (827 cases of 10 tests); replay (product80, vents80, cavity40, annex216) |
| `diagonal_order` | the diagonal's neighbour sum added in another order | parity (827 cases of 10 tests); replay (product80, vents80, cavity40, annex216) |
| `flux_order` | the diagonal's net flux added in another order | parity (827 cases of 10 tests); replay (product80, vents80, cavity40, annex216) |
| `open_high_edge_kept` | a zero-gradient high edge's coefficient left in the diagonal | parity (201 cases of 4 tests) |
| `open_low_edge_kept` | a zero-gradient low edge's coefficient left in the diagonal | parity (201 cases of 4 tests) |
| `high_edge_value_dropped` | a Dirichlet high edge's value left out of the source | CTest (2); parity (181 cases of 3 tests); replay (cavity40) |
| `low_edge_value_dropped` | a Dirichlet low edge's value left out of the source | CTest (1); parity (181 cases of 3 tests) |
| `edges_swapped` | the high edge's condition read at the low one | CTest (2); parity (201 cases of 4 tests); replay (cavity40) |
| `obstacle_neighbour_kept_high` | the coefficient toward an obstacle's face on the high side not zeroed | parity (164 cases of 3 tests) |
| `obstacle_neighbour_kept_low` | the coefficient toward an obstacle's face on the low side not zeroed | parity (164 cases of 3 tests) |
| `solid_faces_unmasked` | the diagonal of a SOLID cell's face left as formed | CTest (2); parity (732 cases of 10 tests); replay (product80, vents80) |
| `domain_edge_entries_kept` | the other component's entries left at a component's domain-edge faces | parity (20 cases of 1 tests) |
| `unknown_by_not_zero` | a face taken for an unknown where its diagonal is not zero, not where it is positive | parity (213 cases of 5 tests) |
| `half_rho_dropped` | the density's half in the mass flux dropped | CTest (3); parity (827 cases of 10 tests); replay (product80, vents80, cavity40, annex216) |
| `takes_wall_not_at_obstacles` | wall_mu not read at the obstacle faces | CTest (1); parity (526 cases of 11 tests); replay (product80, vents80) |
| `takes_wall_not_at_edges` | wall_mu not read at the domain edges | CTest (1); parity (577 cases of 11 tests); replay (product80, vents80, annex216) |
| `open_edge_conducts` | a zero-gradient edge conducting as a wall does, with a viscosity field | parity (310 cases of 6 tests) |
| `obstacle_wall_distance` | the obstacle wall taken a whole cell away, not half | CTest (2); parity (846 cases of 12 tests); replay (product80, vents80) |
| `edge_wall_distance` | the high edge's wall distance read at the low edge | parity (795 cases of 12 tests); replay (product80, vents80, cavity40, annex216) |
| `quick_order` | QUICK's three products added in another order | parity (800 cases of 9 tests); replay (product80, vents80, cavity40, annex216) |
| `streamwise_blocked_pos` | the boundary form not taken beyond an obstacle's wall node, flow toward increasing s | parity (716 cases of 9 tests); replay (product80, vents80) |
| `streamwise_blocked_neg` | the boundary form not taken beyond an obstacle's wall node, flow toward decreasing s | parity (716 cases of 9 tests); replay (product80, vents80) |
| `transverse_wall_pos` | the obstacle's wall not the far node, flow toward increasing t | parity (715 cases of 9 tests); replay (product80, vents80) |
| `transverse_wall_neg` | the obstacle's wall not the far node, flow toward decreasing t | parity (715 cases of 9 tests); replay (product80, vents80) |
| `obstacle_face_corrected` | a deferred flux through an obstacle face | parity (706 cases of 9 tests); replay (product80, vents80) |
| `edge_node_not_value` | the node beyond a Dirichlet edge of a u row the adjacent value, not the wall's | parity (807 cases of 9 tests); replay (product80, vents80, cavity40, annex216) |
| `edge_node_not_value_v` | the node beyond a Dirichlet high edge of a v row the adjacent value, not the wall's | parity (797 cases of 9 tests); replay (product80, vents80, cavity40, annex216) |
| `end_far_node` | the streamwise weights at the last face from the node before it as in the interior's other end | CTest (1); parity (807 cases of 9 tests); replay (product80, vents80, cavity40, annex216) |
| `row_blocked_flags` | no row of cells marked as holding an obstacle | parity (716 cases of 9 tests); replay (product80, vents80) |
| `row_wall_flags` | no row of corners marked as holding an obstacle | parity (717 cases of 9 tests); replay (product80, vents80) |
| `edge_face_corrected_v` | a deferred flux through a v row's domain-edge faces | CTest (1); parity (807 cases of 9 tests); replay (product80, vents80, annex216) |
| `harmonic_solid_low` | a SOLID cell on the low side of a harmonic mean not replaced by its neighbour | CTest (1); parity (445 cases of 11 tests); replay (product80, vents80) |
| `harmonic_solid_high` | a SOLID cell on the high side of a harmonic mean not replaced by its neighbour | CTest (1); parity (445 cases of 11 tests); replay (product80, vents80) |
| `harmonic_fraction_swapped` | the harmonic mean's two distances swapped | parity (615 cases of 13 tests); replay (product80, vents80, annex216) |
| `mean_fraction_swapped` | the arithmetic mean's two widths swapped | CTest (1); parity (615 cases of 13 tests); replay (product80, vents80, annex216) |
| `first_mean_row_skipped` | the first row of cells' harmonic means of v taken from an earlier call | CTest (1); parity (497 cases of 11 tests); replay (product80, vents80, annex216) |
| `stress_dropped` | form b's stress source left out | CTest (1); parity (467 cases of 9 tests); replay (product80, vents80, annex216) |
| `stress_sign_t` | the low transverse face's stress added, not subtracted | parity (467 cases of 9 tests); replay (product80, vents80, annex216) |
| `stress_width_cancelled` | the transverse stress flux without dividing and multiplying by the width | parity (179 cases of 6 tests); replay (product80, vents80, annex216) |
| `pressure_sign` | the pressure source's sign | CTest (2); parity (807 cases of 9 tests); replay (product80, vents80, cavity40, annex216) |
| `relaxation_order` | the relaxation term's product in another order | parity (124 cases of 2 tests) |
| `source_order` | the source's parts added in another order | parity (181 cases of 3 tests); replay (cavity40) |
| `sweep_neighbours_swapped` | a sweep's transverse neighbours taken from the other sides | CTest (3); parity (807 cases of 9 tests); replay (product80, vents80, cavity40, annex216) |
| `sweep_order` | a sweep's four products added in another order | CTest (1); parity (799 cases of 9 tests); replay (product80, vents80, cavity40, annex216) |
| `gauss_seidel` | each sweep reading the rows the same sweep has written | CTest (2); parity (807 cases of 9 tests); replay (product80, vents80, cavity40, annex216) |
| `sweep_targets` | the last sweep written to the scratch iterate, not to the output | CTest (4); parity (807 cases of 9 tests); replay (product80, vents80, cavity40, annex216) |
| `iterate_edges_not_kept` | the scratch iterate's domain-edge faces left as the last call's | CTest (6); parity (628 cases of 8 tests); replay (product80, vents80, annex216) |
| `star_edges_not_kept` | the output's domain-edge faces not set to the input | CTest (6); parity (807 cases of 9 tests); replay (product80, vents80, cavity40, annex216) |
| `alpha_ignored` | the sweeps dividing by the un-relaxed diagonal | CTest (6); parity (783 cases of 9 tests); replay (product80, vents80, cavity40, annex216) |
| `held_reads_iterate` | a vent's faces set to the sweep's iterate, whose rows the last sweep wrote | CTest (1); parity (391 cases of 6 tests); replay (vents80) |
| `kept_row_first` | every kept row of fluxes the same one, not the last two along the direction | CTest (5); parity (847 cases of 11 tests); replay (product80, vents80, cavity40, annex216) |
| `area_without_width` | a streamwise face's area without the transverse width | CTest (5); parity (827 cases of 10 tests); replay (product80, vents80, cavity40, annex216) |
| `edge_relaxed_kept` | the relaxed diagonal beside a domain edge the one from before the edge's part left it | parity (181 cases of 3 tests) |
| `edge_neighbour_kept` | the coefficient toward a domain edge not zeroed | parity (20 cases of 1 tests) |
| `edge_at_every_face` | an edge's condition read at the row's first face along an edge that crosses the rows | parity (201 cases of 4 tests) |
| `cross_pressure_below` | the pressure below a row of v taken from the row itself | CTest (1); parity (807 cases of 9 tests); replay (product80, vents80, cavity40, annex216) |
| `stencil_everywhere` | stencil_viscosity's face rule at every face, read or not | CTest (1); parity (38 cases of 2 tests); replay (product80, vents80, annex216) |
| `stencil_last_row_u` | stencil_viscosity's top row of u left at air's viscosity | CTest (1); parity (38 cases of 2 tests); replay (product80, vents80, annex216) |
| `zero_flux_positive_s` | a streamwise face of exactly zero flux takes the positive direction's stencil | parity (16 cases of 1 tests) |
| `zero_flux_positive_t` | a transverse face of exactly zero flux takes the positive direction's stencil | parity (16 cases of 1 tests) |
| `number_zero` | the core's check: a zero taken for positive | CTest (1); parity (10 cases of 1 tests) |
| `number_infinite` | the core's check: an infinity taken for finite | CTest (1); parity (10 cases of 1 tests) |
| `number_last_width` | the core's check: the last width of a direction not checked | CTest (1); parity (16 cases of 1 tests) |
| `number_cell_widths` | the core's check: the cell widths' values removed | CTest (1); parity (8 cases of 1 tests) |
| `number_face_widths` | the core's check: the face widths' values removed | CTest (1); parity (8 cases of 1 tests) |
| `number_wall_distance` | the core's check: the wall distances removed | CTest (1); parity (16 cases of 1 tests) |
| `number_wall_distance_high` | the core's check: the high edges' wall distances not checked | CTest (1); parity (8 cases of 1 tests) |
| `number_rho` | the core's check: rho removed | CTest (1); parity (4 cases of 1 tests) |
| `number_mu` | the core's check: mu removed | CTest (1); parity (4 cases of 1 tests) |
| `number_alpha` | the core's check: alpha's lower end removed | CTest (1); parity (2 cases of 1 tests) |
| `number_alpha_zero` | the core's check: an alpha of zero accepted | CTest (1); parity (1 cases of 1 tests) |
| `number_alpha_above` | the core's check: alpha's upper end removed | CTest (1); parity (1 cases of 1 tests) |
| `number_alpha_one` | the core's check: an alpha of one refused | CTest (6); parity (25 cases of 2 tests) |
| `length_longer` | the core's check removed: a longer array accepted | CTest (1) |
| `length_pressure` | the core's check removed: the pressure's length | CTest (1) |
| `length_mu_eff` | the core's check removed: mu_eff's length in predict and equations | CTest (1) |
| `length_velocity` | the core's check removed: the velocities' lengths | CTest (1) |
| `length_velocity_first` | the core's check removed: the velocities' lengths but the first | CTest (1) |
| `length_wall_mu` | the core's check removed: wall_mu's lengths in predict and equations | CTest (1) |
| `length_fields_of_predict` | the core's check removed: the inputs' lengths in predict | CTest (1) |
| `length_fields_of_equations` | the core's check removed: the inputs' lengths in equations | CTest (1) |
| `length_star` | the core's check removed: star's lengths | CTest (1) |
| `length_diagonal` | the core's check removed: the diagonals' lengths in predict | CTest (1) |
| `length_equations_diagonal` | the core's check removed: the diagonal's length in equations | CTest (1) |
| `length_equations_component` | the core's check removed: the component of equations | CTest (1) |
| `length_stencil_mu` | the core's check removed: mu_eff's length in stencil_viscosity | CTest (1) |
| `length_stencil_out` | the core's check removed: the outputs' lengths in stencil_viscosity | CTest (1) |
| `length_check_viscosity` | the core's check removed: mu_eff's length in check_viscosity | CTest (1) |
| `length_check_wall` | the core's check removed: wall_mu's length in check_wall_viscosity | CTest (1) |
| `length_check_wall_component` | the core's check removed: the component of check_wall_viscosity | CTest (1) |
| `length_check_wall_slot` | the core's check removed: the slot of check_wall_viscosity | CTest (1) |
| `check_not_finite` | a viscosity that is not finite accepted | CTest (1); parity (14 cases of 2 tests) |
| `check_not_positive` | a viscosity of zero accepted | CTest (1); parity (4 cases of 2 tests) |
| `check_infinity` | an infinite viscosity taken for a finite one | CTest (1); parity (6 cases of 2 tests) |
| `check_solid_too` | a SOLID cell's viscosity checked | CTest (2); parity (490 cases of 17 tests) |
| `check_wall_everywhere` | a wall viscosity checked where no unknown reads it | parity (580 cases of 18 tests); replay (product80, vents80) |
| `check_first_cell_only` | the checks stopping before the cells after the whole blocks | CTest (2); parity (18 cases of 2 tests) |
| `check_two_cells` | the core's check removed: two cells per axis | CTest (2); parity (3 cases of 1 tests) |
| `check_cell_widths` | the core's check removed: the cell widths' lengths | CTest (1); parity (4 cases of 1 tests) |
| `check_face_widths` | the core's check removed: the face widths' lengths | CTest (1); parity (4 cases of 1 tests) |
| `check_edge_lengths` | the core's check removed: the edges' lengths | CTest (1) |
| `check_held_length` | the core's check removed: the held masks' lengths | CTest (1) |
| `check_sweeps` | the core's check removed: at least one sweep | CTest (1); parity (1 cases of 1 tests) |
| `check_throws` | the core's check removed: the constructor's throw | CTest (2); parity (58 cases of 2 tests) |
| `binding_edges_swapped` | the binding: bottom and top handed over swapped | parity (803 cases of 14 tests); replay (product80, vents80, cavity40, annex216) |
| `binding_left_right` | the binding: left's condition read from the right's | parity (821 cases of 14 tests); replay (product80, vents80, cavity40, annex216) |
| `binding_held_v_dropped` | the binding: the vents of v not handed over | parity (392 cases of 7 tests); replay (vents80) |
| `binding_held_shape` | the binding: the vents' shape not checked | parity (2 cases of 1 tests) |
| `binding_edge_count` | the binding: the number of edges not checked | parity (1 cases of 1 tests) |
| `binding_edge_flags` | the binding: the edges' flags' length not checked | parity (4 cases of 1 tests) |
| `binding_edge_values` | the binding: the edges' values' length not checked | parity (4 cases of 1 tests) |
| `binding_edge_unaligned` | the binding: an unaligned edge array accepted | parity (4 cases of 1 tests) |
| `binding_width_unaligned` | the binding: an unaligned width array accepted | parity (4 cases of 1 tests) |
| `binding_u_v_shape` | the binding: u's and v's shapes not checked | parity (6 cases of 1 tests) |
| `binding_v_shape` | the binding: v's shape not checked | parity (2 cases of 1 tests) |
| `binding_p_shape` | the binding: p's shape not checked | parity (2 cases of 1 tests) |
| `binding_field_unaligned` | the binding: an unaligned field accepted | parity (3 cases of 1 tests) |
| `binding_mu_type` | the binding: a viscosity that is not an ndarray not named | parity (20 cases of 2 tests) |
| `binding_mu_dtype` | the binding: a viscosity's dtype not checked | parity (8 cases of 2 tests) |
| `binding_mu_layout` | the binding: a viscosity's layout converted silently | parity (6 cases of 1 tests) |
| `binding_mu_shape` | the binding: a viscosity's shape not checked | parity (10 cases of 2 tests) |
| `binding_mu_unaligned` | the binding: an unaligned viscosity accepted | parity (3 cases of 1 tests) |
| `binding_mu_check` | the binding: mu_eff's values not checked | parity (13 cases of 2 tests) |
| `binding_wall_dict` | the binding: a wall_mu that is not a dict not named | parity (2 cases of 1 tests) |
| `binding_wall_keys` | the binding: wall_mu's keys not checked | parity (4 cases of 1 tests) |
| `binding_wall_extra_key` | the binding: an extra key of wall_mu accepted | parity (1 cases of 1 tests) |
| `binding_wall_check` | the binding: wall_mu's values not checked | parity (9 cases of 1 tests) |
| `binding_wall_swapped` | the binding: v's wall viscosity handed to u | parity (503 cases of 15 tests); replay (product80, vents80, annex216) |
| `binding_transient` | the binding: a transient argument ignored | parity (8 cases of 1 tests) |
| `binding_transient_with_a_field` | the binding: a transient argument ignored with a viscosity field (test 92 S1) | parity (4 cases of 1 tests) |
| `binding_transient_with_a_wall` | the binding: a transient argument ignored with a wall viscosity (test 92 S1) | parity (4 cases of 1 tests) |
| `binding_rho_bool` | the binding: a bool rho accepted | parity (3 cases of 1 tests) |
| `binding_mu_bool` | the binding: a bool mu accepted | parity (3 cases of 1 tests) |
| `binding_alpha_bool` | the binding: a bool alpha accepted | parity (3 cases of 1 tests) |
| `binding_sweeps_bool` | the binding: a bool or a float sweeps accepted | parity (4 cases of 1 tests) |
| `binding_distance_bool` | the binding: a bool wall distance accepted | parity (16 cases of 1 tests) |
| `binding_edge_parts` | the binding: an edge of another number of parts accepted | parity (8 cases of 1 tests) |
| `binding_wall_ignored` | the binding: wall_mu not handed to predict | parity (449 cases of 8 tests); replay (product80, vents80, annex216) |
| `binding_fixed_name` | the binding: an unknown name's refusal | parity (1 cases of 1 tests) |

148 of 148 plants fail at least one net.

### T5. `predict` per call on the 320x120 product state, one thread

| Run | Started | Reference, ms (quartiles) | Core, ms (quartiles) | Ratio | Calibration before and after, us | Windows before | Windows after |
|---|---|---|---|---|---|---|---|
| 1 | 2026-11-23 14:36:22 | 29.44 (28.67 to 31.36) | 2.662 (2.474 to 3.042) | 11.06 | 8.23, 9.24 | overwolfbrowser 124, msmpeng 33, overwolfbrowser 25, on mains True, processor performance 89 | overwolfbrowser 95, msmpeng 40, overwolfbrowser 29, on mains True, processor performance 92 |
| 2 | 2026-11-23 14:36:38 | 29.44 (28.80 to 31.20) | 2.668 (2.555 to 2.991) | 11.03 | 8.36, 8.22 | overwolfbrowser 43, msmpeng 35, overwolfbrowser 33, on mains True, processor performance 91 | overwolfbrowser 49, overwolfbrowser 40, msmpeng 35, on mains True, processor performance 89 |
| 3 | 2026-11-23 14:36:55 | 29.51 (29.28 to 30.03) | 2.834 (2.663 to 3.209) | 10.41 | 8.02, 8.16 | overwolfbrowser 110, msmpeng 54, overwolfbrowser 38, on mains True, processor performance 92 | overwolfbrowser 84, msmpeng 32, wmiprvse 25, on mains True, processor performance 93 |

### T5b. The parts, the same sessions

| Run | Call | Reference, ms | Core, ms | Ratio |
|---|---|---|---|---|
| 1 | predict with one sweep | 21.796 | 1.665 | 13.09 |
| 1 | predict on the scalar path, ten sweeps | 27.639 | 2.097 | 13.18 |
| 1 | stencil_viscosity | 1.678 | 0.154 | 10.91 |
| 2 | predict with one sweep | 20.311 | 1.452 | 13.99 |
| 2 | predict on the scalar path, ten sweeps | 25.430 | 2.216 | 11.47 |
| 2 | stencil_viscosity | 1.185 | 0.140 | 8.48 |
| 3 | predict with one sweep | 20.110 | 1.398 | 14.39 |
| 3 | predict on the scalar path, ten sweeps | 27.081 | 2.465 | 10.98 |
| 3 | stencil_viscosity | 1.210 | 0.141 | 8.56 |

### T6. The kernels' forms on the same call (the ratio to the head's over the three rounds)

| Form | What | predict, ten sweeps, us | one sweep, us | scalar path, ten sweeps, us | scalar path, one sweep, us | stencil_viscosity, us | Bit for bit |
|---|---|---|---|---|---|---|---|
| `head` | the kernels as they are | 2,964 | 1,701 | 2,534 | 1,196 | 148 | yes |
| `quick_loops` | the QUICK face values of both directions as loops of one face with a branch per choice | 3,520 (1.11 to 1.22) | 2,048 (1.15 to 1.25) | 2,978 (1.10 to 1.24) | 1,518 (1.21 to 1.27) | 146 (0.92 to 1.00) | yes |
| `obstacles_every_row` | the obstacle forms formed on every row, not on the rows that hold one | 3,344 (1.11 to 1.13) | 1,887 (1.09 to 1.12) | 2,758 (0.97 to 1.15) | 1,344 (1.10 to 1.19) | 154 (0.99 to 1.16) | yes |
| `np_maximum` | the upwind coefficients' maximum by scalar::np_maximum, whose blocks stay scalar | 3,513 (1.09 to 1.25) | 2,127 (1.21 to 1.25) | 3,067 (1.01 to 1.28) | 1,617 (1.34 to 1.35) | 162 (1.06 to 1.12) | yes |
| `two_scratches` | each component's equations in arrays of its own, not the two in turn in one set | 3,182 (1.00 to 1.14) | 1,858 (1.04 to 1.14) | 2,616 (0.97 to 1.09) | 1,274 (1.04 to 1.08) | 142 (0.90 to 1.01) | yes |

Started 2026-11-23 14:37:05. Windows before: overwolfbrowser 68, overwolfbrowser 41, msmpeng 37, on mains True, processor performance 93; after: oled care screensaver.scr 168, msmpeng 39, code 26, on mains True, processor performance 90.

### T7. The blocks GCC vectorises

| Block | Vectorised |
|---|---|
| `cell_conductance` | yes |
| `cell_flux` | yes |
| `corner_flux` | yes |
| `corner_conductance` | yes |
| `cell_deferred` | yes |
| `corner_deferred` | yes |
| `edge_nodes` | yes |
| `harmonic_cells` | yes |
| `mean_cells` | yes |
| `read_cells` | yes |
| `cell_stress` | yes |
| `corner_stress` | yes |
| `stream_neighbours` | yes |
| `cross_neighbours` | yes |
| `diagonals` | yes |
| `edge_unknowns` | yes |
| `sources` | yes |
| `sweep_cells` | yes |
| `verdict_cells` | yes |

g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0; -std=c++20 -O2 -ffp-contract=off -march=x86-64-v3 -DNDEBUG. 19 of 19 blocks vectorised, every one with vectors of 32 bytes. With `np_maximum` (T6) the blocks left scalar: `cell_flux`, `corner_flux`.

### T8. Each kernel and pass, the build of 6ca02d7 against the per-direction one

| Kernel or pass | 6ca02d7, us | Per direction, us | Ratio by session | Ratio, all rounds |
|---|---|---|---|---|
| cell_conductance u | 8.5 | 8.7 | 1.060, 1.004, 1.041 | 1.024 |
| cell_flux u | 13.1 | 12.8 | 0.995, 0.966, 0.944 | 0.979 |
| cell_stress u | 8.5 | 8.7 | 1.060, 1.004, 1.041 | 1.024 |
| cell_deferred u | 19.6 | 20.0 | 1.067, 1.020, 1.046 | 1.020 |
| corner_deferred u | 12.9 | 12.6 | 0.938, 0.952, 1.030 | 0.973 |
| cell_deferred obstacles u | 38.5 | 39.9 | 1.036, 1.034, 1.027 | 1.034 |
| corner_deferred obstacles u | 22.6 | 23.4 | 1.052, 1.011, 1.042 | 1.036 |
| corner_flux u | 17.1 | 17.1 | 1.009, 0.979, 1.010 | 1.005 |
| corner_stress u | 8.4 | 8.7 | 1.045, 1.018, 1.059 | 1.029 |
| corner_conductance field u | 10.6 | 10.9 | 1.046, 1.015, 1.058 | 1.027 |
| corner_conductance wall u | 13.4 | 14.1 | 1.072, 0.997, 1.080 | 1.048 |
| corner_conductance field wall u | 15.9 | 16.3 | 1.032, 1.012, 1.094 | 1.024 |
| harmonic_cells u | 11.2 | 12.0 | 1.109, 1.009, 1.043 | 1.068 |
| mean_cells u | 6.9 | 7.1 | 1.058, 1.023, 1.051 | 1.029 |
| coefficients u | 49.8 | 49.0 | 0.996, 0.981, 1.003 | 0.984 |
| sources field u | 50.9 | 52.9 | 1.071, 1.019, 1.051 | 1.038 |
| sources air u | 22.0 | 22.5 | 1.069, 1.020, 1.038 | 1.023 |
| sweep_cells u | 34.5 | 34.4 | 0.990, 1.009, 0.953 | 0.997 |
| sweep_cells held u | 36.7 | 37.4 | 0.893, 1.023, 1.163 | 1.019 |
| cell_conductance v | 8.5 | 8.7 | 1.039, 1.012, 1.043 | 1.022 |
| cell_flux v | 15.2 | 14.9 | 0.963, 0.957, 0.978 | 0.978 |
| cell_stress v | 8.5 | 8.7 | 1.048, 1.018, 1.043 | 1.026 |
| cell_deferred v | 13.5 | 12.7 | 0.952, 0.937, 0.963 | 0.939 |
| corner_deferred v | 19.2 | 19.9 | 1.063, 1.038, 1.041 | 1.037 |
| cell_deferred obstacles v | 24.3 | 25.2 | 1.045, 1.013, 1.031 | 1.036 |
| corner_deferred obstacles v | 35.1 | 36.3 | 1.025, 1.024, 1.030 | 1.035 |
| corner_flux v | 13.3 | 13.8 | 1.039, 1.007, 1.060 | 1.036 |
| corner_stress v | 8.4 | 8.7 | 1.057, 1.017, 1.042 | 1.031 |
| corner_conductance field v | 10.4 | 10.6 | 1.050, 1.016, 1.051 | 1.028 |
| corner_conductance wall v | 12.4 | 12.8 | 1.073, 1.006, 1.031 | 1.031 |
| corner_conductance field wall v | 14.9 | 15.4 | 1.037, 1.033, 1.058 | 1.030 |
| harmonic_cells v | 13.5 | 15.8 | 1.226, 1.132, 1.146 | 1.169 |
| mean_cells v | 5.0 | 5.1 | 1.057, 1.007, 1.038 | 1.018 |
| coefficients v | 49.4 | 50.0 | 1.026, 1.002, 1.033 | 1.012 |
| sources field v | 50.3 | 51.7 | 1.064, 1.019, 1.038 | 1.029 |
| sources air v | 21.8 | 22.3 | 1.052, 1.022, 1.031 | 1.022 |
| sweep_cells v | 35.5 | 35.6 | 1.015, 1.004, 1.024 | 1.003 |
| sweep_cells held v | 37.0 | 36.2 | 1.017, 0.979, 0.826 | 0.979 |
| edge_nodes | 6.8 | 7.1 | 1.026, 1.032, 1.045 | 1.043 |
| read_cells | 6.2 | 6.3 | 1.060, 1.049, 1.050 | 1.028 |
| equations field wall u | 230.5 | 235.0 | 1.059, 1.025, 1.045 | 1.019 |
| equations air u | 141.3 | 147.0 | 1.052, 1.031, 1.052 | 1.040 |
| equations field wall v | 220.7 | 225.6 | 1.060, 1.008, 1.038 | 1.022 |
| equations air v | 142.0 | 144.1 | 1.049, 1.002, 1.030 | 1.015 |
| cell_row u field | 58.0 | 58.9 | 1.026, 1.000, 1.033 | 1.016 |
| cell_row v field | 42.8 | 42.9 | 1.016, 0.988, 1.008 | 1.001 |
| corner_row u field | 78.2 | 76.0 | 0.988, 0.969, 0.992 | 0.972 |
| corner_viscosity u | 16.1 | 17.2 | 1.087, 1.045, 1.111 | 1.067 |
| corner_viscosity v | 17.8 | 17.7 | 1.031, 0.944, 1.020 | 0.993 |
| corner_row v field | 87.5 | 89.0 | 1.032, 0.999, 1.042 | 1.017 |
| unknown_row u field | 91.3 | 96.0 | 1.052, 1.022, 1.062 | 1.051 |
| unknown_row v field | 88.9 | 92.1 | 1.026, 1.016, 1.065 | 1.036 |
| cell_row u air | 42.0 | 42.4 | 0.988, 0.988, 1.033 | 1.009 |
| cell_row v air | 26.2 | 25.7 | 0.960, 1.001, 1.015 | 0.981 |
| corner_row u air | 28.6 | 25.7 | 0.909, 0.889, 0.941 | 0.901 |
| corner_row v air | 40.8 | 42.1 | 1.032, 1.022, 1.068 | 1.032 |
| unknown_row u air | 65.8 | 66.2 | 0.989, 1.000, 1.033 | 1.006 |
| unknown_row v air | 67.1 | 65.6 | 0.955, 0.974, 1.011 | 0.977 |
| sweep u | 25.5 | 25.8 | 0.991, 1.009, 0.972 | 1.009 |
| sweep v | 32.6 | 32.1 | 0.999, 0.978, 0.968 | 0.986 |
| predict one sweep field wall | 536.3 | 554.9 | 1.019, 0.994, 1.081 | 1.035 |
| sweep held u | 28.3 | 28.8 | 0.993, 0.995, 1.033 | 1.015 |
| sweep held v | 31.7 | 32.2 | 1.029, 0.896, 1.018 | 1.018 |
| predict ten sweeps field wall | 1090.4 | 1097.6 | 0.991, 0.974, 1.030 | 1.007 |
| predict ten sweeps air | 904.7 | 887.7 | 0.995, 0.969, 1.006 | 0.981 |
| stencil_viscosity | 54.7 | 56.4 | 1.083, 0.992, 1.025 | 1.030 |
| verdict_cells | 7.2 | 7.4 | 1.056, 1.001, 0.989 | 1.024 |

Above 1.05 in every session: `harmonic_cells v`. session 1 started 2026-11-23 18:06:26 (Windows before: wow 175, msmpeng 30, system 22, on mains True, processor performance 154); session 2 started 2026-11-23 18:07:18 (Windows before: wow 172, memory compression 61, msmpeng 33, on mains True, processor performance 162); session 3 started 2026-11-23 18:08:12 (Windows before: wow 199, msmpeng 44, wmiprvse 20, on mains True, processor performance 150).
