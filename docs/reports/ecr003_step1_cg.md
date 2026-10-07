# ECR-003 Step 1: The Conjugate Gradient Pressure Correction

**Date:** 2026-10-06
**Tree:** branch `feature/ecr003-pressure-cg` from main at 867ef89. The five commits are listed in
the pull request body; `src/pressure.py`, `src/config.py`, `src/solver_staggered.py` and
`src/stopping.py` are the modules changed.
**Instruments:** `results/builder37/` (untracked): `extract37.py` (the appendices of
`docs/reports/pressure_solver_ecr003.md` extracted byte for byte), the byte copies `common36.py`,
`solvers36.py`, `capture36.py` and `outer36.py` it wrote, the byte copies `outlet33b.py` and
`frozen34.py`, a probe environment `venv37` (NumPy 2.4.4, PyYAML 6.0.3, SciPy 1.18.1, for the
SuperLU-driven recapture only), `item0_probe.py` and `item0_built.py` (item 0 and criterion 4),
`make_common37.py` and the `common37.py` it derives (appendix A, with the diff), `outer37.py`
(criterion 3, appendix B), `mutate37.py` (the mutation log). Every run's log and json is beside it.
**Order:** item 0 ran before the solve was wired into `PressureCorrector.correct`
(`item0.md`); criteria 3 and 4 ran after commit 2 of the branch.

## 1. The question

ECR-003 step 1 (`docs/ECR/ECR-003-pressure-solver.md`, section 8) builds what ADR-013 states:
the pressure correction solved by conjugate gradients preconditioned with its diagonal, from
p' = 0, to a relative residual `pressure_rtol`, a rounding floor or a reported iteration cap,
with the right-hand side projected onto the range on a closed domain; the key, the renamed
counts, the capped-correction refusal; and every consumer of ECR-003 section 7 moved so that no
saved field or harness row of the old solver can pass as the new one's. The step owns three of
the request's acceptance criteria: 1 (the solve, by test), 3 (the report's two measured rooms
reproduce) and 4 (one correction on 200x75 in under 0.5 s). Its gate, item 0 of the prompt, is
that the built loop agrees with the probe the report measured, since the report's numbers are
otherwise numbers about another program.

## 2. Method

### 2.1 Environment

AMD Ryzen AI 9 HX 370, Windows 11, Python 3.13.3, NumPy 2.4.4 in the project environment. The
recapture ran in `venv37` because `capture36.py` drives the room with SuperLU; nothing else used
it, and `requirements.txt` is unchanged. Every timed run set `OPENBLAS_NUM_THREADS`,
`OMP_NUM_THREADS` and `MKL_NUM_THREADS` to 1. The suite timings were taken on snapshots of the
tree with nothing else of mine running; the criterion 4 timing ran alone.

### 2.2 The probes recreated

`extract37.py` reads the report as bytes, finds "## Appendix X: name" and the python fence after
it, writes the fenced bytes, reads them back and compares. All four files are identical to their
appendices; the SHA-256 of each is in `item0.md`. `outlet33b.py` and `frozen34.py` are byte copies
of `results/builder33b/` and `results/builder34/`, with the hashes the report's section 2.1
records (2ad2b9f2..., 3a85849b...). The classes `common36.py` imports from them, `OutletSolver`
and `FrozenSolver`, read no renamed field, so the copies run unchanged on the built solver; the
scripts' own `main` functions, which read `pressure_tol` and `pressure_sweeps`, are not called.

### 2.3 Item 0

`capture36.py product200 1001` recaptured the T3 room's p' systems at outer 1, 100 and 1,000 on
the branch's first commit, where `src/` is origin/main's. `item0_probe.py`, in a worktree of
origin/main (where `common36` still imports `JACOBI_WEIGHT`), ran `solvers36.jacobi_pcg` on each
at 1e-4, 1e-6 and 1e-8 with the probe's floor, `1e-13` times the supply. `item0_built.py` ran
`src.pressure.conjugate_gradient` with `apply_operator` on the same systems and compared counts
and solutions. The gate: stop if a count differs or a solution differs by more than the
tolerance's worth.

### 2.4 Criterion 1

`tests/test_pressure.py` and `tests/test_solver_staggered.py`, section 4. The dense comparisons
read the error against the residual's 2-norm over the smallest eigenvalue CG sees, computed from
the dense matrix (the smallest nonzero one on the singular cavity), not against a chosen number.

### 2.5 Criterion 3

`outer37.py` builds the report's rooms exactly as `outer36.py` did (`product_t3`: T3 outlets, ten
momentum sweeps, alpha_velocity 0.5, from rest, error_estimate at 1e-6 and ADR-011 G's per-cell
bound), through `common37.py`, the appendix-A probe with the edits ECR-003 section 7.3 names and
the three the renames force (appendix A). The probe corrector is installed with `solve=None`, so
every correction is `PressureCorrector.correct`, the built one, and the probe records its
iterations and relative residual. The velocity-step crossing (residual below 1e-6) is recorded on
the way. The report's runs to compare with are `pcg_1e-8` (section 8.1) and `pcg_1e-8_mu1000`
(section 8.3).

### 2.6 Criterion 4

`item0_built.py --time`: seven solves of each captured system to 1e-8, interleaved across the three
as the report's `retime` did, one BLAS thread, nothing else running; the median and the fastest.

### 2.7 The mutation log, the grep and the suite

`mutate37.py` copies `src/`, `tests/`, `validation/`, `configs/`, `scripts/` and `pyproject.toml`
into a scratch tree, plants one defect at a time and runs the tests named for it there; the
working tree is never edited. The planted defects are the prompt's traps (the stop on the
recursive residual, the closed domain's projection, the hidden cap, old results as new, the floor)
and the guards beside them. The grep is `git grep` over the tracked tree for `pressure_tol`,
`JACOBI_WEIGHT`, `staggered-jacobi` and `pressure_sweeps`. The suite ran on a snapshot of each
commit's tree; main's figure is from a worktree of 867ef89.

## 3. Item 0 (written after the runs)

The recapture gave 10,910 unknowns per system, no pin, as the report's. On every system and
level the built loop returned the probe's iteration count and its solution bit for bit:

| System | 1e-4 | 1e-6 | 1e-8 | Solutions |
|---|---|---|---|---|
| outer 1 | 703 | 791 | 852 | identical |
| outer 100 | 680 | 794 | 861 | identical |
| outer 1000 | 672 | 792 | 857 | identical |

These are the report's counts (section 7.2). The built loop keeps the probe's order of
operations: `apply_operator` forms the five terms as `apply_a` did, the three reductions are
`vdot`, the updates are the same statements. The true-residual check it adds at exit passed at
the first test on every solve, so it changed no count. The gate passed; the prompt's stop did not
fire. Record: `item0.md`.

## 4. Criterion 1: what the tests pin (written after the runs)

`tests/test_pressure.py` (REQ-S08 as amended, REQ-S04 as clarified):

- `TestConjugateGradient::test_agrees_with_a_dense_solve_to_the_tolerances_worth`: the channel
  and the cavity, each with an obstacle, at `pressure_rtol` 1e-10; the true relative residual at
  exit meets the level and the 2-norm error against the dense solution is within the residual
  over the smallest eigenvalue CG sees.
- `test_val002_cavity_system_agrees_with_a_dense_solve[0, 300]`: the VAL-002 cavity's own system
  on 20x20 at outer 0 and at outer 300 of its committed solve, against a dense solve of the
  pinned system, to the same bound; neither correction reached the cap.
- `test_stops_at_the_first_iteration_meeting_the_relative_level`: at 1e-4, 1e-8 and 1e-12 the
  true residual meets the level, one iteration fewer leaves it above, and the counts rise.
- `test_the_floor_ends_a_correction_at_rounding`: `RESIDUAL_FLOOR` is 1e-13; with it raised to
  1e-2 the correction stops below 1e-2 F and above the relative level, in fewer iterations; the
  same loop with a floor of zero takes the full count.
- `test_flux_scale_is_the_stopping_rules`: F is rho times the inflow on the channel and rho times
  the lid speed times the longer side on the 2 by 1 cavity, as the stopping rule defines it; zero
  with no boundary velocity.
- `test_exit_check_reads_the_true_residual_not_the_recursion`: an operator that perturbs its
  product on its third call makes the recursive residual a lie; a loop that trusts it (the
  planted defect, written out in the test) returns a solution whose true residual misses the
  stop by more than a hundredfold; the built loop ends with the true residual under the stop and
  reports it.
- `test_cap_is_reported_with_the_true_residual`, `TestCorrection::test_iterations_are_capped_by_
  max_pressure_iter_and_the_cap_reported`: three and seven iterations on systems that need
  more: `reached_cap` true, the residual reported the true one; the default cap is not reached.
- `test_residual_is_the_corrected_faces_imbalance`: on four systems solved loosely (cap 5),
  `b + A p'` equals `mass_imbalance` of the corrected faces to 64 eps of the largest face flux.
- `test_closed_domain_right_hand_side_is_projected_onto_the_range`: a lid face admitting air
  makes b incompatible; the correction still stops at the relative level on the projected
  residual, leaving the mean and nothing else.
- `test_closed_domain_in_two_components_is_refused`: a wall the full height of the cavity raises
  at construction naming two components; one block does not; an open domain is not checked.
- `TestCoefficients::test_apply_operator_is_the_dense_matrix`: the shifted products equal the
  dense matrix's product on both domains with an obstacle, and the matrix is symmetric.
- The unchanged properties keep their tests: the closed-domain right-hand side sums to zero to
  rounding on 20, 40 and 80 (ECR-001 criterion 6), the coefficients, the two-cell and 3x3 exact
  solutions, the pin, the untouched boundary faces, the rejected shapes.

`tests/test_solver_staggered.py::TestCappedCorrections` (ADR-013 B):

- capped corrections are counted in `pressure_cap_hits`, reset per solve, and warned once;
- a correction truncated to nothing (the faces and the pressure returned as predicted, flagged
  capped) settles the tiny cavity into the frozen shape of the report's section 8.2, a velocity
  step below `convergence_tol` with an imbalance of the supply's order; velocity_step does not
  stop on it and the solve ends at `max_simple_iter`, and the real, converging solve with every
  correction merely flagged capped does not stop either;
- under error_estimate the same truncated correction is refused through continuity;
- and the other direction: with a cap of one CG iteration the tiny cavity's outer loop still
  converges, its last correction meets the floor rather than the cap, and the stop is allowed.

`tests/test_config.py::TestPressureKeys`: `pressure_rtol` read in [1e-10, 1) with the lower bound
accepted and 1.0 refused; zero, negative, NaN, bool and string refused by name; `pressure_tol`
refused with the message that names `pressure_rtol`, with or without the new key beside it;
`max_pressure_iter` still required; `SOLVER_KEYS` the one tuple of twelve; every committed
configuration and the transport stand-ins at 1e-8 and 5,000. `tests/test_benchmark.py`: the
retired label refused, every solver key in a row's params (issue 38), `pressure_cap_hits` in the
outcome, `inner_iterations` in the work. `tests/test_stopping_probe.py`: the rule key changes
with `PRESSURE_SOLVER_VERSION`; the truth, tight-truth and control files are re-solved when the
stored version is missing or old and reused when it is this solver's, tested without a solve.
`tests/test_self_convergence.py`: `tight_field` reads the current label's file and nothing else.

## 5. Criterion 3: the measured rooms reproduce (written after the runs)

| Room | velocity_step at | error_estimate at | Stop | Inner iterations, median [max] | Relative residual, median | Cap hits | Worst cell at the stop | Seconds | The report's (`pcg_1e-8`) |
|---|---|---|---|---|---|---|---|---|---|
| 40x15, real air | 1,209 | 2,822 | error_estimate_and_continuity | 162 [173] | 2.2e-08 | 0 | 1.4e-13 of 8.0e-08 | 33 | 1,209 / 2,822; inner 162 [173]; 46 s |
| 80x30, a thousand times air's viscosity | 233 | 588 | error_estimate_and_continuity | 303 [303] | 7.5e-09 | 0 | 5.1e-11 of 2.0e-08 | 16 | 233 / 588; inner 303 [303]; 21 s |

Both rooms stop at the report's outer counts exactly, under both rules, with the report's inner
counts; the criterion's 1% was not needed. No correction reached the cap of 5,000. The seconds
are shorter than the report's probe runs because those ran several at once (section 8 there).
Records: `c3_40x15.json`, `c3_80x30_mu1000.json` and their logs.

## 6. Criterion 4: the cost (written after the runs)

One correction on each captured 200x75 system at 1e-8, seven solves interleaved, one thread:

| System | Iterations | ms, median | ms, fastest | ms, slowest |
|---|---|---|---|---|
| outer 1 | 852 | 131 | 130 | 206 |
| outer 100 | 861 | 138 | 128 | 199 |
| outer 1000 | 857 | 162 | 124 | 193 |

Under the 0.5 s the criterion asks, and at the report's 0.13 to 0.16 s (section 7.2: 157 to 163 ms;
section 12.1: 136 to 147 ms). The orchestrator's prediction of about 0.15 s held. Record:
`criterion4_timing.json`.

## 7. The mutation log (written after the runs)

21 of 21 planted defects fail a named test (`mutation37.json`; the harness is `mutate37.py`).

| Id | Trap or guard | Planted in | Tests run | Result | First failing test |
|---|---|---|---|---|---|
| M1 | The stop reads the right residual: the exit check removed, the recursion trusted | `src/pressure.py` | test_pressure.py::TestConjugateGradient::test_exit_check_reads_the_true_residual_not_the_recursion | KILLED | `tests/test_pressure.py::TestConjugateGradient::test_exit_check_reads_the_true_residual_not_the_recursion` |
| M2 | The stop reads the right residual: a failed exit check returns capped instead of going on | `src/pressure.py` | test_pressure.py::TestConjugateGradient::test_exit_check_reads_the_true_residual_not_the_recursion | KILLED | `tests/test_pressure.py::TestConjugateGradient::test_exit_check_reads_the_true_residual_not_the_recursion` |
| M3 | The closed domain: b not projected onto the range | `src/pressure.py` | test_pressure.py::TestConjugateGradient::test_closed_domain_right_hand_side_is_projected_onto_the_range | KILLED | `tests/test_pressure.py::TestConjugateGradient::test_closed_domain_right_hand_side_is_projected_onto_the_range` |
| M4 | The closed domain: p' not pinned after the solve | `src/pressure.py` | test_pressure.py::TestCorrection::test_closed_domain_pins_the_reference_cell; test_pressure.py::TestCorrection::test_pin_removes_a_nonzero_constant_mode | KILLED | `tests/test_pressure.py::TestCorrection::test_closed_domain_pins_the_reference_cell` |
| M5 | The closed domain: cells with an equation in two components accepted | `src/pressure.py` | test_pressure.py::TestConjugateGradient::test_closed_domain_in_two_components_is_refused | KILLED | `tests/test_pressure.py::TestConjugateGradient::test_closed_domain_in_two_components_is_refused` |
| M6 | The closed domain: the right-hand side left nonzero at cells without an equation | `src/pressure.py` | test_pressure.py::TestConjugateGradient::test_a_cell_without_an_equation_is_left_out_of_the_solve | KILLED | `tests/test_pressure.py::TestConjugateGradient::test_a_cell_without_an_equation_is_left_out_of_the_solve` |
| M7 | The cap is reported: a solve ended by the cap says it was not | `src/pressure.py` | test_pressure.py::TestCorrection::test_iterations_are_capped_by_max_pressure_iter_and_the_cap_reported; test_pressure.py::TestConjugateGradient::test_cap_is_reported_with_the_true_residual | KILLED | `tests/test_pressure.py::TestCorrection::test_iterations_are_capped_by_max_pressure_iter_and_the_cap_reported` |
| M8 | The cap is reported: velocity_step stops on a capped correction | `src/solver_staggered.py` | test_solver_staggered.py::TestCappedCorrections::test_velocity_step_does_not_stop_on_a_capped_correction | KILLED | `tests/test_solver_staggered.py::TestCappedCorrections::test_velocity_step_does_not_stop_on_a_capped_correction` |
| M9 | The cap is reported: capped corrections not counted | `src/solver_staggered.py` | test_solver_staggered.py::TestCappedCorrections::test_capped_corrections_are_counted_and_warned_once | KILLED | `tests/test_solver_staggered.py::TestCappedCorrections::test_capped_corrections_are_counted_and_warned_once` |
| M10 | The cap is reported: the count not reset per solve | `src/solver_staggered.py` | test_solver_staggered.py::TestContract::test_iteration_count_cap_hits_and_stage_timers_reset_at_the_start_of_each_solve | KILLED | `tests/test_solver_staggered.py::TestContract::test_iteration_count_cap_hits_and_stage_timers_reset_at_the_start_of_each_solve` |
| M11 | The floor is a module constant: the floor removed from the stop | `src/pressure.py` | test_pressure.py::TestConjugateGradient::test_the_floor_ends_a_correction_at_rounding | KILLED | `tests/test_pressure.py::TestConjugateGradient::test_the_floor_ends_a_correction_at_rounding` |
| M12 | The floor's F: the shorter side taken on a closed domain | `src/pressure.py` | test_pressure.py::TestConjugateGradient::test_flux_scale_is_the_stopping_rules; test_solver_staggered.py::TestStoppingRule::test_error_estimate_scales_are_physical_and_the_step_is_in_m_per_s | KILLED | `tests/test_pressure.py::TestConjugateGradient::test_flux_scale_is_the_stopping_rules` |
| M13 | The operator: one neighbour term with the wrong sign | `src/pressure.py` | test_pressure.py::TestCoefficients::test_apply_operator_is_the_dense_matrix; test_pressure.py::TestConjugateGradient::test_residual_is_the_corrected_faces_imbalance | KILLED | `tests/test_pressure.py::TestCoefficients::test_apply_operator_is_the_dense_matrix` |
| M14 | The keys: pressure_tol accepted (falls through to the unknown-key refusal) | `src/config.py` | test_config.py::TestPressureKeys::test_the_retired_pressure_tol_is_refused_naming_the_new_key | KILLED | `tests/test_config.py::TestPressureKeys::test_the_retired_pressure_tol_is_refused_naming_the_new_key` |
| M15 | The keys: pressure_rtol's lower bound not enforced | `src/config.py` | test_config.py::TestPressureKeys::test_bad_pressure_rtol_is_rejected | KILLED | `tests/test_config.py::TestPressureKeys::test_bad_pressure_rtol_is_rejected[9.9e-11-ValueError-solver\\.pressure_rtol must be in]` |
| M16 | The keys: pressure_rtol's upper bound inclusive | `src/config.py` | test_config.py::TestPressureKeys::test_bad_pressure_rtol_is_rejected | KILLED | `tests/test_config.py::TestPressureKeys::test_bad_pressure_rtol_is_rejected[1.0-ValueError-solver\\.pressure_rtol must be in \\[1e-10, 1\\.0\\)]` |
| M17 | Issue 38: the harness reads its own key list, missing pressure_rtol | `scripts/benchmark.py` | test_benchmark.py::test_every_solver_key_is_recorded_from_the_one_list; test_benchmark.py::test_staggered_velocity_step_stop_has_the_collocated_label | KILLED | `tests/test_benchmark.py::test_every_solver_key_is_recorded_from_the_one_list` |
| M18 | Old results as new: the harness keeps the Jacobi-era label | `scripts/benchmark.py` | test_benchmark.py::test_run_case_refuses_the_retired_staggered_jacobi_label_by_name | KILLED | `tests/test_benchmark.py::test_run_case_refuses_the_retired_staggered_jacobi_label_by_name` |
| M19 | Old results as new: a saved truth reused whenever it exists | `scripts/stopping_probe.py` | test_stopping_probe.py::test_saved_solves_are_reused_only_when_this_solver_wrote_them | KILLED | `tests/test_stopping_probe.py::test_saved_solves_are_reused_only_when_this_solver_wrote_them[no-version]` |
| M20 | Old results as new: the rule solve's key without the solver's version | `scripts/stopping_probe.py` | test_stopping_probe.py::test_rule_parameters_change_with_the_pressure_solver_version | KILLED | `tests/test_stopping_probe.py::test_rule_parameters_change_with_the_pressure_solver_version` |
| M21 | Old results as new: tight_field reads the Jacobi-era file name | `scripts/self_convergence.py` | test_self_convergence.py::test_tight_field_reads_the_current_labels_1e_9_snapshot_and_nothing_else | KILLED | `tests/test_self_convergence.py::test_tight_field_reads_the_current_labels_1e_9_snapshot_and_nothing_else` |

## 8. The grep

`git grep` over the tracked tree for `pressure_tol`, `JACOBI_WEIGHT`, `staggered-jacobi` and
`pressure_sweeps`, at the records commit: the survivors are history and intended references.
`benchmarks/results.jsonl` (42 rows with `pressure_tol`, 20 with `staggered-jacobi`), unchanged as
the request requires; the reports under `docs/reports/`, which quote the code they measured; the
ECR, ADR-010, ADR-011, ADR-012 and ADR-013 texts, which name what was replaced; SYSTEM.md's
history rows; `src/config.py`'s `RETIRED_PRESSURE_TOL_KEY` and the refusal message;
`src/solver_staggered.py`'s and `src/stopping.py`'s notes of the rename; `scripts/benchmark.py`'s
`STAGGERED_JACOBI_METHOD`, kept known so the stored rows summarize and refused by `run_case`;
`scripts/self_convergence.py`'s and `scripts/view_field.py`'s comments on the retired label; and
the tests that pin each refusal. No source, configuration, fixture or script reads a retired
name as a live one.

## 9. The suite

| Tree | Collected | Result | Wall time |
|---|---|---|---|
| main at 867ef89 | 908 | 907 passed, 1 skipped | 447 s |
| commit 2 (the solve) | 933 | 932 passed, 1 skipped | 203 s |
| commit 3 (the consumers) | 939 | 938 passed, 1 skipped | 216 s |
| commit 4 (the sealed-cell test) | 940 | 939 passed, 1 skipped | 208 s |

The delta is +32 tests. The runtime fell by about half, as predicted: every test solve that swept
to its cap under Jacobi now stops in tens to hundreds of CG iterations. `gen_system_map.py
--check` and ruff pass at every commit.

## 10. Contracts that differ from ADR-013 D's draft

- `apply_operator(coefficients, x)`, `conjugate_gradient(apply, inverse_diagonal, f, rtol, floor,
  max_iter)` and `ConjugateGradientResult` are module-level in `src/pressure.py`. The draft kept
  the solve inside `correct`; the loop is exposed so the tests can plant a lying operator and
  count iterations, the probes can call it on a captured system, and Phase 6 can run it for a
  fixed count (ADR-013 C).
- `ZERO_SCALE` moves from `solver_staggered.py` to `pressure.py`, and the solver imports it, so
  the corrector's flux scale and the solver's reference velocity read one guard.
- `PressureCorrector.flux_scale` is public and the solver's `_new_rule` reads it, so the floor
  and the error_estimate rule share one F; the solver's `flux_scale` property keeps its contract
  (None under velocity_step).
- The harness work key is `inner_iterations`; stored rows keep `inner_sweeps`, which nothing
  reads back. `pressure_cap_hits` sits in each row's `outcome`. `WorkCounter`'s first parameter
  is `cells_per_iteration`.
- The one-component check counts a cell as having an equation when it is non-SOLID with a
  non-SOLID 4-neighbour; a sealed single cell is not counted.

## 11. Against the prediction (orchestrator, written before handover)

| Prediction | Measured |
|---|---|
| Item 0 agrees exactly in iteration counts if the built CG keeps the probe's order of operations | Held: every count and every solution identical, bit for bit |
| Criterion 3 reproduces the report's outer counts exactly (1,209 and 2,822; 233 and 588) | Held exactly |
| Criterion 4 about 0.15 s per correction on 200x75 | Held: 131 to 162 ms by the medians |
| The suite's runtime falls | Held: 447 s to 208 s with 32 more tests |
| About 120 lines in `pressure.py`, 40 across config and the solver | `pressure.py` grew from 441 to 624 lines (the diff stat: 455 changed lines), the growth mostly the docstrings of the two module-level functions and the result type, the component check and the flux scale; config, the solver and stopping: 152 changed lines by the diff stat (67, 76 and 9), the bounds and the refusal message, the capped-correction count, warning and refusal, and their docstrings. Both above the prediction; the consumer edits were many and small as predicted, 1,262 lines added and 419 removed across the scripts, the tests, the configurations and the stand-ins |

## 12. What this does not settle

Step 2: the laminar baseline retaken under `staggered-cg` (VAL-001 and VAL-002 against their
criteria; the transport gate tests re-checked), whose rows become ECR-002 criterion 1's. Step 3:
ADR-013 accepted with planned against built and the request closed. The default's cost on a finer
mesh and its revisit after ECR-002 step 3 (ADR-013 decisions 1 and 3) are the request's open
items, not this step's. Whether a capped correction can arise in a real room under the committed
cap of 5,000 is not measured here: the report's largest count is 1,643 on 400x150.

## Appendix A: common37.py, the edited probe

Derived from the byte copy of appendix A of `docs/reports/pressure_solver_ecr003.md` by
`make_common37.py`; the diff first, then the file. `outlet33b.py` and `frozen34.py` are unchanged
byte copies (SHA-256 2ad2b9f2a02684569c113d782c7f5c8dedf956bcaf75da906bbc1eadb0ebbd4d and
3a85849b63e587e5e9a31a1d7c2c1ff5ae333390ad357e228616cfe7f90868d9, the report's).

```diff
--- common36.py
+++ common37.py
@@ -1,4 +1,10 @@
 """Builder probe, prompt 36 (ECR-003): rooms, the probe corrector and system capture.
+
+Prompt 37 copy: common36.py (appendix A of docs/reports/pressure_solver_ecr003.md)
+edited for the built solver, as make_common37.py records: set_solver writes
+pressure_rtol and a cap of 5,000; JACOBI_WEIGHT and committed_jacobi are gone
+with the sweep; the count field is iterations and a probe correction carries
+reached_cap. Nothing else changed.
 
 Nothing in src/ is edited or patched. The rooms are built from the committed
 configurations; the product room runs under the T3 outlets of
@@ -47,7 +53,6 @@
 from src.mesh import Mesh  # noqa: E402
 from src.momentum import MomentumPrediction  # noqa: E402
 from src.pressure import (  # noqa: E402
-    JACOBI_WEIGHT,
     PressureCoefficients,
     PressureCorrection,
     PressureCorrector,
@@ -157,7 +162,7 @@
             result = super().correct(prediction, p)
             seconds = time.perf_counter() - t0
             if self.measure_residual:
-                self.records.append(self._record(c, b, result.p_prime, result.sweeps, seconds))
+                self.records.append(self._record(c, b, result.p_prime, result.iterations, seconds))
             return result
 
         # The committed correct, line for line, with the solve replaced.
@@ -193,7 +198,8 @@
             v=np.ascontiguousarray(v),
             p=p_next,
             p_prime=p_prime,
-            sweeps=inner,
+            iterations=inner,
+            reached_cap=False,
         )
 
     def _record(self, c, b, p_prime, inner, seconds) -> CorrectionRecord:  # type: ignore[no-untyped-def]
@@ -206,24 +212,6 @@
         fn = float(np.linalg.norm(f))
         rel = float(np.linalg.norm(r)) / fn if fn > 0.0 else 0.0
         return CorrectionRecord(inner=int(inner), rel_residual=rel, b_norm=float(np.linalg.norm(b)), seconds=seconds)
-
-
-def committed_jacobi(corrector: PressureCorrector, tol: float, cap: int) -> Solve:
-    """Today's solve as a Solve: the committed weighted sweep and stop, copied from correct."""
-
-    def solve(c, b, active, needs_pin):  # type: ignore[no-untyped-def]
-        p_prime = np.zeros(c.a_p.shape, dtype=np.float64)
-        sweeps = 0
-        for _ in range(cap):
-            p_new = corrector.sweep(p_prime, c, b, JACOBI_WEIGHT)
-            diff = float(np.max(np.abs(p_new[active] - p_prime[active]))) if active.any() else 0.0
-            p_prime = p_new
-            sweeps += 1
-            if diff < tol:
-                break
-        return p_prime, sweeps
-
-    return solve
 
 
 # ---------------------------------------------------------------------------
@@ -262,8 +250,8 @@
     raw: dict,
     n_outer: int,
     alpha_u: float = 0.5,
-    p_cap: int = 40000,
-    p_tol: float = 1e-8,
+    p_cap: int = 5000,
+    p_rtol: float = 1e-8,
     stopping: dict | None = None,
 ) -> dict:
     """The ladder's solver settings (step 0, section 2.1) unless given; stopping keys added."""
@@ -271,7 +259,7 @@
     block["max_simple_iter"] = n_outer
     block["alpha_velocity"] = alpha_u
     block["max_pressure_iter"] = p_cap
-    block["pressure_tol"] = p_tol
+    block["pressure_rtol"] = p_rtol
     if stopping:
         block.update(stopping)
     return raw
@@ -299,10 +287,10 @@
 
 def product_t3(
     name: str, nx: int, ny: int, n_outer: int, sweeps: int = 10, mu_factor: float = 1.0,
-    p_cap: int = 40000, p_tol: float = 1e-8, alpha_u: float = 0.5, stopping: dict | None = None,
+    p_cap: int = 5000, p_rtol: float = 1e-8, alpha_u: float = 0.5, stopping: dict | None = None,
 ) -> Room:
     """The product room under T3, laminar (zero field), N momentum sweeps per outer iteration."""
-    raw = set_solver(product_raw(nx, ny, mu_factor), n_outer, alpha_u, p_cap, p_tol, stopping)
+    raw = set_solver(product_raw(nx, ny, mu_factor), n_outer, alpha_u, p_cap, p_rtol, stopping)
     cfg = SimConfig.from_dict(raw)
     mesh = Mesh(cfg)
     boundary = StaggeredBoundary(mesh, cfg)
@@ -311,7 +299,7 @@
         mesh, cfg, boundary, hood, np.zeros(mesh.cell_type.shape), sweeps=sweeps
     )
     meta = {"case": "product_t3", "nx": nx, "ny": ny, "sweeps": sweeps, "mu_factor": mu_factor,
-            "alpha_velocity": alpha_u, "p_cap": p_cap, "p_tol": p_tol, "stopping": stopping or {}}
+            "alpha_velocity": alpha_u, "p_cap": p_cap, "p_rtol": p_rtol, "stopping": stopping or {}}
     return Room(name, cfg, mesh, boundary, solver, _supply(cfg, mesh, boundary), meta)
```

```python
"""Builder probe, prompt 36 (ECR-003): rooms, the probe corrector and system capture.

Prompt 37 copy: common36.py (appendix A of docs/reports/pressure_solver_ecr003.md)
edited for the built solver, as make_common37.py records: set_solver writes
pressure_rtol and a cap of 5,000; JACOBI_WEIGHT and committed_jacobi are gone
with the sweep; the count field is iterations and a probe correction carries
reached_cap. Nothing else changed.

Nothing in src/ is edited or patched. The rooms are built from the committed
configurations; the product room runs under the T3 outlets of
results/builder33b/outlet33b.py and, with ten momentum sweeps, through
FrozenPredictor of results/builder34/frozen34.py with a zero eddy-viscosity
field (step 0, section 2.4: with mu_t = 0 the field path is the committed
predictor to the bit at one sweep). Both files are byte copies of the main
tree's (SHA-256 recorded in the report).

ProbeCorrector subclasses PressureCorrector and replaces only the p' solve:
the coefficients, the right-hand side, the pin, the velocity correction and
the pressure update are the committed lines, copied. With ``solve=None`` it
calls the committed ``correct`` itself. It can save the p' system it is
handed at chosen outer iterations, and it records, per correction, the
inner iterations, the relative residual reached and the norm of b.

The p' equation in the corrector's convention is A p' = -b with
(A x)_P = a_P x_P - sum(a_nb x_nb). The mass imbalance the corrected faces
leave in each cell is b + A p' exactly (the report, section 3.3), so the
residual of the p' equation and the corrected faces' continuity are one
vector read two ways.
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results" / "builder34"))
sys.path.insert(0, str(ROOT / "results" / "builder33b"))

import frozen34  # noqa: E402
from outlet33b import OutletSolver, segment_names  # noqa: E402

from src.boundary_staggered import StaggeredBoundary  # noqa: E402
from src.config import SimConfig  # noqa: E402
from src.mesh import Mesh  # noqa: E402
from src.momentum import MomentumPrediction  # noqa: E402
from src.pressure import (  # noqa: E402
    PressureCoefficients,
    PressureCorrection,
    PressureCorrector,
)
from src.solver_staggered import StaggeredSolver  # noqa: E402
from validation.cases import load_preset, with_velocity_step  # noqa: E402

SYSTEMS = HERE / "systems"
SYSTEMS.mkdir(exist_ok=True)

# A p' solve: (coefficients, b, active, needs_pin) -> (p_prime, inner iterations).
Solve = Callable[[PressureCoefficients, np.ndarray, np.ndarray, bool], tuple[np.ndarray, int]]


def threads_note() -> dict[str, str | None]:
    """The BLAS thread settings a run saw, for the records."""
    keys = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    return {k: os.environ.get(k) for k in keys}


# ---------------------------------------------------------------------------
# The operator, in the corrector's convention
# ---------------------------------------------------------------------------


def apply_a(c: PressureCoefficients, x: np.ndarray) -> np.ndarray:
    """(A x)_P = a_P x_P - sum(a_nb x_nb); the edge coefficients are zero by construction."""
    y = c.a_p * x
    y[:, :-1] -= c.a_e[:, :-1] * x[:, 1:]
    y[:, 1:] -= c.a_w[:, 1:] * x[:, :-1]
    y[:-1, :] -= c.a_n[:-1, :] * x[1:, :]
    y[1:, :] -= c.a_s[1:, :] * x[:-1, :]
    return y


def corrected_imbalance(c: PressureCoefficients, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The mass imbalance the corrected faces leave in each cell: b + A p', kg/s per m."""
    out = b + apply_a(c, x)
    out[c.a_p <= 0.0] = 0.0
    return out


# ---------------------------------------------------------------------------
# The probe corrector
# ---------------------------------------------------------------------------


@dataclass
class CorrectionRecord:
    """What one correction did."""

    inner: int
    rel_residual: float
    b_norm: float
    seconds: float


class ProbeCorrector(PressureCorrector):
    """PressureCorrector with a replaceable p' solve, system capture and per-call records."""

    def __init__(  # type: ignore[no-untyped-def]
        self,
        mesh,
        config,
        boundary,
        solve: Solve | None = None,
        capture_at: tuple[int, ...] = (),
        capture_name: str = "",
        measure_residual: bool = True,
    ) -> None:
        super().__init__(mesh, config, boundary)
        self.solve = solve
        self.capture_at = set(capture_at)
        self.capture_name = capture_name
        self.measure_residual = measure_residual
        self.calls = 0
        self.records: list[CorrectionRecord] = []
        self.meta: dict = {}

    def _capture(self, k: int, prediction: MomentumPrediction, c, b) -> None:  # type: ignore[no-untyped-def]
        d_u, d_v = self._face_d(prediction.a_p_u, prediction.a_p_v)
        np.savez(
            SYSTEMS / f"{self.capture_name}_it{k}.npz",
            a_p=c.a_p, a_e=c.a_e, a_w=c.a_w, a_n=c.a_n, a_s=c.a_s, b=b,
            d_u=d_u, d_v=d_v, u_star=prediction.u_star, v_star=prediction.v_star,
            a_p_u=prediction.a_p_u, a_p_v=prediction.a_p_v,
            out_left=self._out_left, out_right=self._out_right,
            out_bottom=self._out_bottom, out_top=self._out_top,
            needs_pin=np.array(self.needs_pin), pin_cell=np.array(self.pin_cell),
            rho=np.array(self._rho), dx_cell=self._mesh.dx_cell, dy_cell=self._mesh.dy_cell,
            solid=self._solid,
            meta=np.array(json.dumps({**self.meta, "outer": k})),
        )

    def correct(self, prediction: MomentumPrediction, p: np.ndarray) -> PressureCorrection:
        """The committed correct with the p' solve replaced; solve None is the committed call."""
        k = self.calls
        self.calls += 1
        need_system = k in self.capture_at or self.measure_residual
        if need_system:
            c = self.coefficients(prediction.a_p_u, prediction.a_p_v)
            b = self.mass_imbalance(prediction.u_star, prediction.v_star)
            if k in self.capture_at:
                self._capture(k, prediction, c, b)
        if self.solve is None:
            t0 = time.perf_counter()
            result = super().correct(prediction, p)
            seconds = time.perf_counter() - t0
            if self.measure_residual:
                self.records.append(self._record(c, b, result.p_prime, result.iterations, seconds))
            return result

        # The committed correct, line for line, with the solve replaced.
        u_star, v_star = prediction.u_star, prediction.v_star
        self._check_shapes(u_star, v_star)
        if p.shape != self._p_shape:
            raise ValueError(f"expected p of shape {self._p_shape}, got {p.shape}")
        if not need_system:
            c = self.coefficients(prediction.a_p_u, prediction.a_p_v)
            b = self.mass_imbalance(u_star, v_star)
        active = c.a_p > 0.0
        pin_j, pin_i = self.pin_cell
        t0 = time.perf_counter()
        p_prime, inner = self.solve(c, b, active, self.needs_pin)
        seconds = time.perf_counter() - t0
        if self.needs_pin:
            p_prime[active] -= p_prime[pin_j, pin_i]

        d_u, d_v = self._face_d(prediction.a_p_u, prediction.a_p_v)
        padded = np.zeros((self._p_shape[0] + 2, self._p_shape[1] + 2), dtype=np.float64)
        padded[1:-1, 1:-1] = p_prime
        u = u_star - d_u * (padded[1:-1, 1:] - padded[1:-1, :-1])
        v = v_star - d_v * (padded[1:, 1:-1] - padded[:-1, 1:-1])

        p_next = p.copy()
        p_next[active] += self._alpha_p * p_prime[active]
        if self.needs_pin:
            p_next[active] -= p_next[pin_j, pin_i]
        if self.measure_residual:
            self.records.append(self._record(c, b, p_prime, inner, seconds))
        return PressureCorrection(
            u=np.ascontiguousarray(u),
            v=np.ascontiguousarray(v),
            p=p_next,
            p_prime=p_prime,
            iterations=inner,
            reached_cap=False,
        )

    def _record(self, c, b, p_prime, inner, seconds) -> CorrectionRecord:  # type: ignore[no-untyped-def]
        f = -b.copy()
        active = c.a_p > 0.0
        if self.needs_pin:
            f[active] -= f[active].mean()
        r = f - apply_a(c, p_prime)
        r[~active] = 0.0
        fn = float(np.linalg.norm(f))
        rel = float(np.linalg.norm(r)) / fn if fn > 0.0 else 0.0
        return CorrectionRecord(inner=int(inner), rel_residual=rel, b_norm=float(np.linalg.norm(b)), seconds=seconds)


# ---------------------------------------------------------------------------
# Rooms
# ---------------------------------------------------------------------------


def product_raw(nx: int, ny: int, mu_factor: float = 1.0) -> dict:
    """configs/clean_room_default.yaml regridded, viscosity scaled."""
    raw = yaml.safe_load((ROOT / "configs/clean_room_default.yaml").read_text())
    raw["domain"]["nx"], raw["domain"]["ny"] = nx, ny
    raw["fluid"]["viscosity"] *= mu_factor
    return raw


def annex20_raw(nx: int, ny: int) -> dict:
    """The Annex 20 room as cost33.py builds it (product_case_reynolds.md, appendix B)."""
    raw = product_raw(nx, ny)
    raw["domain"].update(width=9.0, height=3.0)
    raw["fluid"].update(density=1.2, viscosity=1.2 * 15.3e-6)
    raw["boundaries"] = {
        "slot": {
            "type": "velocity_inlet", "location": "left",
            "y_start": 3.0 - 0.168, "y_end": 3.0, "velocity": 0.455,
        },
        "outlet": {
            "type": "pressure_outlet", "location": "right",
            "y_start": 0.0, "y_end": 0.48,
        },
    }
    raw["obstacles"] = []
    return raw


def set_solver(
    raw: dict,
    n_outer: int,
    alpha_u: float = 0.5,
    p_cap: int = 5000,
    p_rtol: float = 1e-8,
    stopping: dict | None = None,
) -> dict:
    """The ladder's solver settings (step 0, section 2.1) unless given; stopping keys added."""
    block = raw["solver"]
    block["max_simple_iter"] = n_outer
    block["alpha_velocity"] = alpha_u
    block["max_pressure_iter"] = p_cap
    block["pressure_rtol"] = p_rtol
    if stopping:
        block.update(stopping)
    return raw


@dataclass
class Room:
    """A built solver and what the probes need beside it."""

    name: str
    cfg: SimConfig
    mesh: Mesh
    boundary: StaggeredBoundary
    solver: StaggeredSolver
    supply: float  # rho times the inlet volumetric flux, kg/s per m (closed: rho U L)
    meta: dict = field(default_factory=dict)


def _supply(cfg: SimConfig, mesh: Mesh, boundary: StaggeredBoundary) -> float:
    flux = boundary.get_total_inlet_flux()
    if flux > 0.0:
        return cfg.rho * flux
    return cfg.rho * boundary.get_max_boundary_velocity() * max(float(mesh.x[-1]), float(mesh.y[-1]))


def product_t3(
    name: str, nx: int, ny: int, n_outer: int, sweeps: int = 10, mu_factor: float = 1.0,
    p_cap: int = 5000, p_rtol: float = 1e-8, alpha_u: float = 0.5, stopping: dict | None = None,
) -> Room:
    """The product room under T3, laminar (zero field), N momentum sweeps per outer iteration."""
    raw = set_solver(product_raw(nx, ny, mu_factor), n_outer, alpha_u, p_cap, p_rtol, stopping)
    cfg = SimConfig.from_dict(raw)
    mesh = Mesh(cfg)
    boundary = StaggeredBoundary(mesh, cfg)
    hood = segment_names(mesh, cfg, "right") == "hood_exhaust"
    solver = frozen34.FrozenSolver(
        mesh, cfg, boundary, hood, np.zeros(mesh.cell_type.shape), sweeps=sweeps
    )
    meta = {"case": "product_t3", "nx": nx, "ny": ny, "sweeps": sweeps, "mu_factor": mu_factor,
            "alpha_velocity": alpha_u, "p_cap": p_cap, "p_rtol": p_rtol, "stopping": stopping or {}}
    return Room(name, cfg, mesh, boundary, solver, _supply(cfg, mesh, boundary), meta)


def product_committed(name: str, nx: int = 200, ny: int = 75, n_outer: int = 1) -> Room:
    """cost33.py's room: the product as committed (T0, one sweep, alpha 0.7, cap 200, 1e-6)."""
    raw = product_raw(nx, ny)
    raw["solver"]["max_simple_iter"] = n_outer
    cfg = SimConfig.from_dict(raw)
    mesh = Mesh(cfg)
    boundary = StaggeredBoundary(mesh, cfg)
    solver = StaggeredSolver(mesh, cfg, boundary)
    meta = {"case": "product_committed", "nx": nx, "ny": ny}
    return Room(name, cfg, mesh, boundary, solver, _supply(cfg, mesh, boundary), meta)


class AnnexSolver(OutletSolver):
    """outlet33b's T1 (backflow held shut) on the Annex 20 outlet, with N momentum sweeps."""

    def __init__(self, mesh, cfg, boundary, sweeps: int) -> None:  # type: ignore[no-untyped-def]
        no_hood = np.zeros(mesh.yc.shape[0], dtype=bool)
        super().__init__(mesh, cfg, boundary, "T1", 0.0, no_hood)
        self._predictor = frozen34.FrozenPredictor(
            mesh, cfg, boundary, np.zeros(mesh.cell_type.shape), sweeps=sweeps
        )


def annex20(name: str, nx: int, ny: int, n_outer: int, sweeps: int = 10) -> Room:
    """The Annex 20 room, laminar, outlet T1, ten momentum sweeps, step 0's settings."""
    raw = set_solver(annex20_raw(nx, ny), n_outer)
    cfg = SimConfig.from_dict(raw)
    mesh = Mesh(cfg)
    boundary = StaggeredBoundary(mesh, cfg)
    solver = AnnexSolver(mesh, cfg, boundary, sweeps)
    meta = {"case": "annex20_t1", "nx": nx, "ny": ny, "sweeps": sweeps}
    return Room(name, cfg, mesh, boundary, solver, _supply(cfg, mesh, boundary), meta)


def validation_case(name: str, preset: str, n_outer: int) -> Room:
    """A VAL-001 or VAL-002 preset as its case file configures it, under velocity_step."""
    cfg = with_velocity_step(load_preset(preset))
    cfg.max_simple_iter = n_outer
    mesh = Mesh(cfg)
    boundary = StaggeredBoundary(mesh, cfg)
    solver = StaggeredSolver(mesh, cfg, boundary)
    meta = {"case": preset, "nx": cfg.nx, "ny": cfg.ny}
    return Room(name, cfg, mesh, boundary, solver, _supply(cfg, mesh, boundary), meta)


def install_corrector(room: Room, solve: Solve | None, **kwargs) -> ProbeCorrector:  # type: ignore[no-untyped-def]
    """Replace the solver's corrector with a ProbeCorrector built from the same inputs."""
    corrector = ProbeCorrector(room.mesh, room.cfg, room.boundary, solve, **kwargs)
    corrector.meta = {"room": room.name, **room.meta, "supply": room.supply}
    room.solver._corrector = corrector
    return corrector


# ---------------------------------------------------------------------------
# Captured systems
# ---------------------------------------------------------------------------


@dataclass
class System:
    """A captured p' system and what evaluating a solution needs."""

    name: str
    c: PressureCoefficients
    b: np.ndarray
    d_u: np.ndarray
    d_v: np.ndarray
    u_star: np.ndarray
    v_star: np.ndarray
    needs_pin: bool
    pin_cell: tuple[int, int]
    rho: float
    dx_cell: np.ndarray
    dy_cell: np.ndarray
    solid: np.ndarray
    meta: dict

    @property
    def active(self) -> np.ndarray:
        return self.c.a_p > 0.0

    @property
    def supply(self) -> float:
        return float(self.meta["supply"])

    def rhs(self) -> np.ndarray:
        """f = -b, projected onto the range on a closed domain (mean over active cells removed)."""
        f = -self.b.copy()
        if self.needs_pin:
            act = self.active
            f[act] -= f[act].mean()
        f[~self.active] = 0.0
        return f


def load_system(name: str) -> System:
    """Load systems/NAME.npz."""
    z = np.load(SYSTEMS / f"{name}.npz")
    c = PressureCoefficients(a_p=z["a_p"], a_e=z["a_e"], a_w=z["a_w"], a_n=z["a_n"], a_s=z["a_s"])
    return System(
        name=name, c=c, b=z["b"], d_u=z["d_u"], d_v=z["d_v"], u_star=z["u_star"],
        v_star=z["v_star"], needs_pin=bool(z["needs_pin"]),
        pin_cell=(int(z["pin_cell"][0]), int(z["pin_cell"][1])), rho=float(z["rho"]),
        dx_cell=z["dx_cell"], dy_cell=z["dy_cell"], solid=z["solid"],
        meta=json.loads(str(z["meta"])),
    )


def corrected_faces(s: System, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """u* - d grad p' with p' = x pinned as the corrector pins it."""
    x = x.copy()
    if s.needs_pin:
        x[s.active] -= x[s.pin_cell]
    ny, nx = x.shape
    padded = np.zeros((ny + 2, nx + 2))
    padded[1:-1, 1:-1] = x
    u = s.u_star - s.d_u * (padded[1:-1, 1:] - padded[1:-1, :-1])
    v = s.v_star - s.d_v * (padded[1:, 1:-1] - padded[:-1, 1:-1])
    return u, v


def face_imbalance(s: System, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """PressureCorrector.mass_imbalance's arithmetic on the captured mesh spacing."""
    out = s.rho * ((u[:, 1:] - u[:, :-1]) * s.dy_cell[:, None] + (v[1:, :] - v[:-1, :]) * s.dx_cell[None, :])
    out[s.solid] = 0.0  # as PressureCorrector.mass_imbalance zeroes them
    return out
```

## Appendix B: outer37.py

```python
"""Prompt 37, criterion 3: the report's two measured rooms with the built pressure solve.

Usage: python outer37.py NX NY [N_OUTER] [MU_FACTOR]

The product room on NX x NY under T3, laminar, ten momentum sweeps per outer
iteration, alpha_velocity 0.5, from rest, exactly as outer36.py (appendix G)
built it, through common37.py (the edited appendix A) and the byte copies of
outlet33b.py and frozen34.py. The correction is the built one: the probe
corrector is installed with solve=None, so it calls PressureCorrector.correct
and records each correction's iterations and relative residual. The solver
stops by error_estimate (iteration_error_tol 1e-6, mass_imbalance_tol
1e-4 rho V_min / t_end, ADR-011 G) or at N_OUTER (default 20,000); the outer
iteration at which velocity_step (residual below 1e-6) would have stopped is
recorded on the way. Writes c3_NXxNY[_muF].json beside this file.

ECR-003 criterion 3: 40x15 at real air meets velocity_step at 1,209 and
error_estimate at 2,822; 80x30 at a thousand times air's viscosity at 233
and 588, each within 1%.
"""

import json
import math
import sys
import time
from datetime import datetime

import numpy as np

from common37 import install_corrector, product_t3, threads_note

DIVERGED_SPEED = 100.0


class Diverged(Exception):
    """Raised from the callback to end a run past 100 m/s or non-finite."""


def stopping_for(nx: int, ny: int, width: float, height: float, rho: float, t_end: float) -> dict:
    """error_estimate with the defaults' error tolerance and ADR-011 G's per-cell bound."""
    v_min = (width / nx) * (height / ny)
    return {
        "stopping_rule": "error_estimate",
        "iteration_error_tol": 1e-6,
        "mass_imbalance_tol": 1e-4 * rho * v_min / t_end,
    }


def main() -> None:
    nx, ny = int(sys.argv[1]), int(sys.argv[2])
    n_outer = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    mu_factor = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    tag = f"c3_{nx}x{ny}" + (f"_mu{mu_factor:g}" if mu_factor != 1.0 else "")
    stop = stopping_for(nx, ny, 8.0, 3.0, 1.2, 60.0)
    room = product_t3(tag, nx, ny, n_outer, mu_factor=mu_factor, stopping=stop)
    corr = install_corrector(room, None)
    solver = room.solver
    started = datetime.now().isoformat(timespec="seconds")
    print(f"{tag} start {started} {room.meta} supply {room.supply:.6g} threads {threads_note()}", flush=True)
    rec: dict[str, list] = {k: [] for k in ("residual", "inner", "rel", "b_norm", "max_speed", "clock")}
    vs_stop: dict = {}
    t0 = time.perf_counter()

    def callback(state) -> None:  # type: ignore[no-untyped-def]
        top = float(np.max(np.hypot(state.u, state.v)))
        r = corr.records[-1]
        rec["residual"].append(float(state.residual))
        rec["inner"].append(r.inner)
        rec["rel"].append(r.rel_residual)
        rec["b_norm"].append(r.b_norm)
        rec["max_speed"].append(top)
        rec["clock"].append(time.perf_counter() - t0)
        if state.residual < 1e-6 and not vs_stop:
            vs_stop["outer"] = state.iteration + 1
        if state.iteration % 100 == 0:
            print(
                f"{tag} it {state.iteration:6d} res {state.residual:.4e} inner {r.inner:5d} "
                f"rel {r.rel_residual:.2e} max|U| {top:.4g} t {rec['clock'][-1]:7.1f}s",
                flush=True,
            )
        if not math.isfinite(top) or top > DIVERGED_SPEED:
            raise Diverged

    try:
        solver.solve_steady(on_iteration=callback)
        stop_reason = solver.stop_reason
    except Diverged:
        stop_reason = "diverged"
    inner = np.array(rec["inner"])
    out = {
        "room": tag,
        "meta": room.meta,
        "started": started,
        "supply": room.supply,
        "stop": stop_reason,
        "outer": len(rec["residual"]),
        "velocity_step_outer": vs_stop.get("outer"),
        "seconds": time.perf_counter() - t0,
        "ms_per_outer": 1e3 * (time.perf_counter() - t0) / max(len(rec["residual"]), 1),
        "inner_median": float(np.median(inner)) if inner.size else None,
        "inner_max": int(inner.max()) if inner.size else None,
        "rel_median": float(np.median(rec["rel"])) if rec["rel"] else None,
        "pressure_cap_hits": int(solver.pressure_cap_hits),
        "last_pressure_iterations": int(solver.last_pressure_iterations),
        "worst_cell_at_stop": float(np.abs(solver.last_mass_imbalance).max()),
        "mass_imbalance_tol": stop["mass_imbalance_tol"],
        "threads": threads_note(),
        "stage_seconds": dict(solver.stage_seconds),
        **rec,
    }
    with open(f"{tag}.json", "w", encoding="utf-8") as handle:
        json.dump(out, handle)
    print(
        f"{tag} done {datetime.now().isoformat(timespec='seconds')} stop {stop_reason} after "
        f"{out['outer']} (velocity_step at {out['velocity_step_outer']}); inner median "
        f"{out['inner_median']} max {out['inner_max']}; cap hits {out['pressure_cap_hits']}; "
        f"{out['seconds']:.0f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
```

## Appendix C: item0_built.py

```python
"""Item 0, built side: src.pressure.conjugate_gradient on the three recaptured systems.

Runs in the branch's tree with the project environment. Compares with
item0_probe.npz (the appendix CG, run in the origin/main worktree) and
writes item0_built.npz. With --time it times seven solves per system at
1e-8, interleaved across the systems, as the report's retime did.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from src.pressure import (  # noqa: E402
    RESIDUAL_FLOOR,
    PressureCoefficients,
    apply_operator,
    conjugate_gradient,
)


def load(k: int) -> tuple[PressureCoefficients, np.ndarray, float]:
    z = np.load(HERE / "systems" / f"product200_it{k}.npz")
    c = PressureCoefficients(a_p=z["a_p"], a_e=z["a_e"], a_w=z["a_w"], a_n=z["a_n"], a_s=z["a_s"])
    active = c.a_p > 0.0
    f = -z["b"].copy()
    if bool(z["needs_pin"]):
        f[active] -= f[active].mean()
    f[~active] = 0.0
    supply = float(json.loads(str(z["meta"]))["supply"])
    return c, f, supply


def solve(c, f, supply, rtol):
    active = c.a_p > 0.0
    inv = np.where(active, 1.0 / np.where(active, c.a_p, 1.0), 0.0)
    return conjugate_gradient(lambda x: apply_operator(c, x), inv, f, rtol, RESIDUAL_FLOOR * supply, 5000)


def main() -> None:
    probe = np.load(HERE / "item0_probe.npz")
    out = {}
    agree = True
    for k in (1, 100, 1000):
        c, f, supply = load(k)
        for rtol in (1e-4, 1e-6, 1e-8):
            res = solve(c, f, supply, rtol)
            tag = f"it{k}_{rtol:.0e}"
            out[f"x_{tag}"] = res.x
            out[f"iters_{tag}"] = np.array(res.iterations)
            px, pit = probe[f"x_{tag}"], int(probe[f"iters_{tag}"])
            dx = float(np.abs(res.x - px).max())
            true_rel = float(np.linalg.norm(f - apply_operator(c, res.x)) / np.linalg.norm(f))
            same = res.iterations == pit and np.array_equal(res.x, px)
            agree &= same
            print(
                f"built  outer {k:5d} rtol {rtol:.0e}: {res.iterations} iterations (probe {pit}), "
                f"true relative residual {true_rel:.3e}, cap {res.reached_cap}, "
                f"max|x_built - x_probe| {dx:.3e} Pa of max|x| {np.abs(res.x).max():.3e}, bitwise {same}"
            )
    np.savez(HERE / "item0_built.npz", **out)
    print("every count and every solution identical to the probe's:", agree)
    if "--time" in sys.argv:
        systems = {k: load(k) for k in (1, 100, 1000)}
        times = {k: [] for k in systems}
        iters = {}
        for _ in range(7):
            for k, (c, f, supply) in systems.items():
                t0 = time.perf_counter()
                res = solve(c, f, supply, 1e-8)
                times[k].append(time.perf_counter() - t0)
                iters[k] = res.iterations
        rows = {k: {"iterations": iters[k], "ms_median": 1e3 * float(np.median(t)), "ms_min": 1e3 * min(t), "ms_max": 1e3 * max(t)} for k, t in times.items()}
        for k, r in rows.items():
            print(f"timing outer {k:5d}: {r['iterations']} iterations, median {r['ms_median']:.1f} ms, min {r['ms_min']:.1f}, max {r['ms_max']:.1f}")
        (HERE / "criterion4_timing.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
```

## Appendix D: mutate37.py

```python
"""Builder mutation harness, prompt 37: each planted defect must fail a named test.

Usage: python mutate37.py   (from anywhere)

Copies src/, tests/, validation/, configs/, scripts/ and pyproject.toml into
results/builder37/mut_tree, applies one textual mutation at a time to the
copy, runs the tests named for it there, and restores the file. The working
tree is never edited, so nothing measured elsewhere can see a mutant.
Writes mutation37.json beside this file; mutation37.md is written from it.
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TREE = HERE / "mut_tree"
PY = str(ROOT / ".venv" / "Scripts" / "python")

P = "tests/test_pressure.py"
S = "tests/test_solver_staggered.py"
C = "tests/test_config.py"
B = "tests/test_benchmark.py"
SP = "tests/test_stopping_probe.py"
SC = "tests/test_self_convergence.py"

MUTANTS = [
    {
        "id": "M1",
        "trap": "The stop reads the right residual: the exit check removed, the recursion trusted",
        "file": "src/pressure.py",
        "old": "            r = f - apply(x)\n            r_norm = float(np.sqrt(np.vdot(r, r)))\n            if r_norm <= stop:\n                return ConjugateGradientResult(",
        "new": "            if True:\n                return ConjugateGradientResult(",
        "tests": [f"{P}::TestConjugateGradient::test_exit_check_reads_the_true_residual_not_the_recursion"],
    },
    {
        "id": "M2",
        "trap": "The stop reads the right residual: a failed exit check returns capped instead of going on",
        "file": "src/pressure.py",
        "old": "            z = inverse_diagonal * r\n            p = z.copy()\n            rz = float(np.vdot(r, z))\n            continue\n",
        "new": "            return ConjugateGradientResult(\n                x=x, iterations=k, reached_cap=True, residual_norm=r_norm\n            )\n",
        "tests": [f"{P}::TestConjugateGradient::test_exit_check_reads_the_true_residual_not_the_recursion"],
    },
    {
        "id": "M3",
        "trap": "The closed domain: b not projected onto the range",
        "file": "src/pressure.py",
        "old": "        f = -b\n        if self.needs_pin:\n            f[active] -= f[active].mean()\n",
        "new": "        f = -b\n        if False and self.needs_pin:\n            f[active] -= f[active].mean()\n",
        "tests": [f"{P}::TestConjugateGradient::test_closed_domain_right_hand_side_is_projected_onto_the_range"],
    },
    {
        "id": "M4",
        "trap": "The closed domain: p' not pinned after the solve",
        "file": "src/pressure.py",
        "old": "        p_prime = solved.x\n        if self.needs_pin:\n            p_prime[active] -= p_prime[pin_j, pin_i]\n",
        "new": "        p_prime = solved.x\n",
        "tests": [f"{P}::TestCorrection::test_closed_domain_pins_the_reference_cell", f"{P}::TestCorrection::test_pin_removes_a_nonzero_constant_mode"],
    },
    {
        "id": "M5",
        "trap": "The closed domain: cells with an equation in two components accepted",
        "file": "src/pressure.py",
        "old": "                self.pin_cell = (int(fluid_idx[0, 0]), int(fluid_idx[0, 1]))\n            self._check_one_component()\n",
        "new": "                self.pin_cell = (int(fluid_idx[0, 0]), int(fluid_idx[0, 1]))\n",
        "tests": [f"{P}::TestConjugateGradient::test_closed_domain_in_two_components_is_refused"],
    },
    {
        "id": "M6",
        "trap": "The closed domain: the right-hand side left nonzero at cells without an equation",
        "file": "src/pressure.py",
        "old": "            f[active] -= f[active].mean()\n        f[~active] = 0.0\n",
        "new": "            f[active] -= f[active].mean()\n",
        "tests": [f"{P}::TestConjugateGradient::test_a_cell_without_an_equation_is_left_out_of_the_solve"],
    },
    {
        "id": "M7",
        "trap": "The cap is reported: a solve ended by the cap says it was not",
        "file": "src/pressure.py",
        "old": "    return ConjugateGradientResult(\n        x=x, iterations=k, reached_cap=True, residual_norm=float(np.sqrt(np.vdot(r, r)))\n    )",
        "new": "    return ConjugateGradientResult(\n        x=x, iterations=k, reached_cap=False, residual_norm=float(np.sqrt(np.vdot(r, r)))\n    )",
        "tests": [f"{P}::TestCorrection::test_iterations_are_capped_by_max_pressure_iter_and_the_cap_reported", f"{P}::TestConjugateGradient::test_cap_is_reported_with_the_true_residual"],
    },
    {
        "id": "M8",
        "trap": "The cap is reported: velocity_step stops on a capped correction",
        "file": "src/solver_staggered.py",
        "old": "                stop = residual < self._convergence_tol and not corrected.reached_cap\n",
        "new": "                stop = residual < self._convergence_tol\n",
        "tests": [f"{S}::TestCappedCorrections::test_velocity_step_does_not_stop_on_a_capped_correction"],
    },
    {
        "id": "M9",
        "trap": "The cap is reported: capped corrections not counted",
        "file": "src/solver_staggered.py",
        "old": "            if corrected.reached_cap:\n                self.pressure_cap_hits += 1\n",
        "new": "            if corrected.reached_cap:\n                self.pressure_cap_hits += 0\n",
        "tests": [f"{S}::TestCappedCorrections::test_capped_corrections_are_counted_and_warned_once"],
    },
    {
        "id": "M10",
        "trap": "The cap is reported: the count not reset per solve",
        "file": "src/solver_staggered.py",
        "old": "        self.last_pressure_iterations = 0\n        self.pressure_cap_hits = 0\n        self.converged, self.stop_reason = False, None\n",
        "new": "        self.last_pressure_iterations = 0\n        self.converged, self.stop_reason = False, None\n",
        "tests": [f"{S}::TestContract::test_iteration_count_cap_hits_and_stage_timers_reset_at_the_start_of_each_solve"],
    },
    {
        "id": "M11",
        "trap": "The floor is a module constant: the floor removed from the stop",
        "file": "src/pressure.py",
        "old": "            RESIDUAL_FLOOR * self.flux_scale,\n            self._max_iter,",
        "new": "            0.0,\n            self._max_iter,",
        "tests": [f"{P}::TestConjugateGradient::test_the_floor_ends_a_correction_at_rounding"],
    },
    {
        "id": "M12",
        "trap": "The floor's F: the shorter side taken on a closed domain",
        "file": "src/pressure.py",
        "old": "            longer = max(float(mesh.x[-1]), float(mesh.y[-1]))",
        "new": "            longer = min(float(mesh.x[-1]), float(mesh.y[-1]))",
        "tests": [f"{P}::TestConjugateGradient::test_flux_scale_is_the_stopping_rules", f"{S}::TestStoppingRule::test_error_estimate_scales_are_physical_and_the_step_is_in_m_per_s"],
    },
    {
        "id": "M13",
        "trap": "The operator: one neighbour term with the wrong sign",
        "file": "src/pressure.py",
        "old": "    y[:, :-1] -= c.a_e[:, :-1] * x[:, 1:]\n",
        "new": "    y[:, :-1] += c.a_e[:, :-1] * x[:, 1:]\n",
        "tests": [f"{P}::TestCoefficients::test_apply_operator_is_the_dense_matrix", f"{P}::TestConjugateGradient::test_residual_is_the_corrected_faces_imbalance"],
    },
    {
        "id": "M14",
        "trap": "The keys: pressure_tol accepted (falls through to the unknown-key refusal)",
        "file": "src/config.py",
        "old": "        if RETIRED_PRESSURE_TOL_KEY in solver:\n            raise ValueError(",
        "new": "        if False:\n            raise ValueError(",
        "tests": [f"{C}::TestPressureKeys::test_the_retired_pressure_tol_is_refused_naming_the_new_key"],
    },
    {
        "id": "M15",
        "trap": "The keys: pressure_rtol's lower bound not enforced",
        "file": "src/config.py",
        "old": "        if not lower <= self.pressure_rtol < upper:",
        "new": "        if not self.pressure_rtol < upper:",
        "tests": [f"{C}::TestPressureKeys::test_bad_pressure_rtol_is_rejected"],
    },
    {
        "id": "M16",
        "trap": "The keys: pressure_rtol's upper bound inclusive",
        "file": "src/config.py",
        "old": "        if not lower <= self.pressure_rtol < upper:",
        "new": "        if not lower <= self.pressure_rtol <= upper:",
        "tests": [f"{C}::TestPressureKeys::test_bad_pressure_rtol_is_rejected"],
    },
    {
        "id": "M17",
        "trap": "Issue 38: the harness reads its own key list, missing pressure_rtol",
        "file": "scripts/benchmark.py",
        "old": "    params = {name: getattr(config, name) for name in SOLVER_KEYS}",
        "new": "    params = {name: getattr(config, name) for name in SOLVER_KEYS[:8] + SOLVER_KEYS[9:]}",
        "tests": [f"{B}::test_every_solver_key_is_recorded_from_the_one_list", f"{B}::test_staggered_velocity_step_stop_has_the_collocated_label"],
    },
    {
        "id": "M18",
        "trap": "Old results as new: the harness keeps the Jacobi-era label",
        "file": "scripts/benchmark.py",
        "old": 'STAGGERED_METHOD = "staggered-cg"\n',
        "new": 'STAGGERED_METHOD = "staggered-jacobi"\n',
        "tests": [f"{B}::test_run_case_refuses_the_retired_staggered_jacobi_label_by_name"],
    },
    {
        "id": "M19",
        "trap": "Old results as new: a saved truth reused whenever it exists",
        "file": "scripts/stopping_probe.py",
        "old": "    if not path.exists():\n        return False\n    with np.load(path) as saved:\n        return \"pressure_solver_version\" in saved.files and int(\n            saved[\"pressure_solver_version\"]\n        ) == int(PRESSURE_SOLVER_VERSION)",
        "new": "    return path.exists()",
        "tests": [f"{SP}::test_saved_solves_are_reused_only_when_this_solver_wrote_them"],
    },
    {
        "id": "M20",
        "trap": "Old results as new: the rule solve's key without the solver's version",
        "file": "scripts/stopping_probe.py",
        "old": "        [scale, flux, *tols, RATE_WINDOW, RULE_VERSION, PRESSURE_SOLVER_VERSION]",
        "new": "        [scale, flux, *tols, RATE_WINDOW, RULE_VERSION]",
        "tests": [f"{SP}::test_rule_parameters_change_with_the_pressure_solver_version"],
    },
    {
        "id": "M21",
        "trap": "Old results as new: tight_field reads the Jacobi-era file name",
        "file": "scripts/self_convergence.py",
        "old": '    path = FIELD_DIR / f"{STAGGERED_METHOD}_{n}_tol1e-9.npz"\n    with np.load(path) as data:',
        "new": '    path = FIELD_DIR / f"staggered-jacobi_{n}_tol1e-9.npz"\n    with np.load(path) as data:',
        "tests": [f"{SC}::test_tight_field_reads_the_current_labels_1e_9_snapshot_and_nothing_else"],
    },
]


def copy_tree() -> None:
    if TREE.exists():
        shutil.rmtree(TREE)
    TREE.mkdir()
    for name in ("src", "tests", "validation", "configs", "scripts"):
        shutil.copytree(ROOT / name, TREE / name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(ROOT / "pyproject.toml", TREE / "pyproject.toml")
    (TREE / "docs").mkdir()
    shutil.copy(ROOT / "docs" / "SYSTEM.md", TREE / "docs" / "SYSTEM.md")
    shutil.copy(ROOT / "docs" / "system_map_annotations.toml", TREE / "docs" / "system_map_annotations.toml")


def run_tests(tests: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-x", *tests],
        cwd=TREE,
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def first_failure(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("FAILED "):
            return line[len("FAILED ") :].split(" - ")[0]
    return ""


def main() -> int:
    copy_tree()
    results = []
    for m in MUTANTS:
        path = TREE / m["file"]
        original = path.read_text(encoding="utf-8")
        text = original.replace("\r\n", "\n")
        if text.count(m["old"]) != 1:
            results.append({**m, "result": "NOT PLANTED", "matches": text.count(m["old"])})
            print(f"{m['id']}: NOT PLANTED ({text.count(m['old'])} matches)", flush=True)
            continue
        path.write_text(text.replace(m["old"], m["new"]), encoding="utf-8")
        t0 = time.perf_counter()
        code, out = run_tests(m["tests"])
        path.write_text(original, encoding="utf-8")
        killed = code != 0
        tail = out.strip().splitlines()[-1] if out.strip() else ""
        results.append(
            {
                **m,
                "result": "KILLED" if killed else "SURVIVED",
                "first_failing_test": first_failure(out),
                "seconds": round(time.perf_counter() - t0, 1),
                "tail": tail,
            }
        )
        print(f"{m['id']}: {'KILLED' if killed else 'SURVIVED'} ({tail})", flush=True)
    (HERE / "mutation37.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    survivors = [r["id"] for r in results if r["result"] != "KILLED"]
    print(f"{len(results) - len(survivors)} of {len(results)} killed; survivors: {survivors}")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
```
