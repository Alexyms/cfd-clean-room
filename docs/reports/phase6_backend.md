# Phase 6 Step 6a: The Backend Switch and the 80x30 Runs End to End (prompt 96)

**Date:** 2026-11-26
**Prompt:** 96 (`docs/prompts/cc-prompt-96-backend.md`)
**Branch:** `phase6/backend`
**Status:** built. All five predictions hold (T0). Under `reference` nothing changed: prompt
91's three 80x30 runs reproduce bit for bit through `scripts/run_scenario.py`, and no stored
hash test is edited. Under `core` the five product scenarios on 80x30 make every decision of
the reference's runs, every flow state stops at the reference's outer count, and every field
is inside its measured spread; the core's restarts and its two monitor updates are bit for bit.
Every planted defect is caught (two plants are equivalent mutants, recorded with their reasons).
The tables are `docs/reports/probe96/tables96.md`, generated from the records in
`results/builder96/` by `tables96.py` (`--check`); every record names one commit (section 8 gives
it: round 1 made the records again at the commit of its last code change).

**Round 1 (prompt 96b, section 8)** fixed the findings of review 96 and test 96: the comparator's
two wrong readings, every item pinned, the backend named in a `with_backend` copy, a core run's
file read without the core, and the import cycle between `core_backend` and `time_integration`.

This is the first half of Phase 6 step 6 (`docs/PROJECT_PLAN.md`, Phase 6, the steps table;
ADR-017 B, E, F, G and H): the key `solver.backend`, the module `src/core_backend.py` that the
test drivers of prompts 91 and 95 move into, `AlertMonitor`'s update from readings, the
comparator of REQ-N03 level 2, and the five product scenarios on 80x30 run end to end under
each backend and compared.

## The predictions

Set by the orchestrator in prompt 96, and committed alone before any run.

- (a) Under `reference`, prompt 91's three 80x30 runs reproduce bit for bit, and no stored hash
  test changes.
- (b) The update from readings gives the alerts and latencies of the update from fields bit for
  bit on every run compared.
- (c) On all five 80x30 scenarios under `core`, every decision equals the reference's, every flow
  state stops at the reference's outer count, not one apart, and every field is inside the
  measured spread (ADR-017 H's first tier).
- (d) A restart under `core` is the uninterrupted core run bit for bit.
- (e) The 80x30 dust release runs end to end at least four times faster under `core`, same
  session.

## How they are scored (the builder, before any run)

Every run is made under WSL2 with the Linux environment (`~/.venvs/cfd_clean_room`, NumPy
2.4.4), one BLAS thread, at most three solver processes at once, through
`scripts/run_scenario.py` (a run interrupted for a restart is the one exception, below). A run
file is written on the Linux file system and copied into `results/builder96/` when it ends
(ADR-017 A). Every run compared with the core, on either side, has
`NPY_DISABLE_CPU_FEATURES=X86_V4` set (the prompt). "Bit for bit" means equal shapes and bytes,
a float compared by its 64-bit pattern.

- **(a).** Prompt 91's three Linux reference runs of record are
  `results/builder91/runs/reference/{dust80,door80,breach80}.h5`: the 80x30 dust release, door
  leak and filter breach from the flow states of prompt 86's files, made on 2026-11-22 at e152bbd.
  Their summaries state no `NPY_DISABLE_CPU_FEATURES` (the switch dates from prompt 94), so they
  are rerun without it, from prompt 91's own scenario files (`results/builder91/configs/`) with
  the same `--flows-from` files, through `scripts/run_scenario.py` at this branch's head, the key
  absent. Holds when, for each of the three, the stored frames' steps, times and fields, every
  stored budget row, the sensor readings and channels, the alert table, the arrival maps and the
  flow states equal the record's bit for bit; and when no test that asserts a stored hash (the
  run file's, a harness row's, a solve's digest) is edited by this branch and each passes on
  Windows and under WSL2.
- **(b).** Each of the five scenarios run under `core` is run a second time under `core` with
  the monitor on the fields (`CoreRun.monitor_reads_fields`, the stand-in of prompt 91 replaced by
  `AlertMonitor.update` on the fields copied out after every step), and each pair compared.
  Holds when, on every pair, the alert table (every row, every column) and the monitor's
  latencies per event, channel and sensor are equal bit for bit, and the sensor table too.
- **(c).** The five scenarios are main's `configs/scenario_*.yaml` (the clean-out, the door leak,
  the dust release, the filter breach and the seal leak) on 80x30 with `mass_imbalance_tol` the
  base file's rule for that grid, as prompt 91's probe wrote its files, each solved from rest on
  its own side. The comparator (`scripts/compare_runs.py`) reads the two run files and gives, item
  by item: every flow state's stop reason and outer count; every arrival time of the maps, every
  latency of the file's sensors and every clear time, each moved by how many steps or frames;
  every class's budget verdict against REQ-T05; and each field's largest difference with its tier
  (the measured spreads of ADR-017 H, P8's and P2's). Holds when, on all five, every decision
  moves by zero, every flow state's stop reason and outer count are the reference's, and every
  field reads "inside the spread". A flow state that reaches the cap on the reference counts as a
  stop at the cap: it holds when the core's state reaches the cap too, with its fields inside the
  spread.
- **(d).** The 80x30 dust release under `core`, twice interrupted and restarted through
  `scripts/run_scenario.py --restart`: once in the march, a run with a checkpoint every 1,000
  steps ended by an exception from `on_step` after step 1,500 (made in-process, since the script
  takes no stop), and once in the flow solve, a run capped at 600 outer iterations, which leaves
  its steady checkpoint and exits, restarted under the file's cap. Each restarted run is
  compared with the uninterrupted core run of the same configuration but for the restart's keys.
  Holds when both are bit for bit in every item (a)'s list names.
- **(e).** The 80x30 dust release from rest, end to end through `scripts/run_scenario.py`'s
  `main` in one process per run, under each backend, in three sessions, each session timing the
  two backends in alternating order on a machine on mains power with the game and any local model
  server closed, its state recorded. The flow solve's seconds, the march's and the rest (the set
  up, the plans, the writes and the end) are reported apart. Holds when, in every session, the
  reference's wall time over the core's is at least 4.

A failed prediction is a finding, recorded and scored as such.

## 1. What was built

### The key

`solver.backend` is the last of `SOLVER_KEYS`: `reference`, the default, or `core`
(`src/config.py`). `SimConfig.check_backend` refuses `core` at load, with the reason, with a transient
section, since tier 2's true-time step stays on the reference (ADR-017 F, option 1). Round 1 moved
the other refusal: where `_cfdcore` cannot be imported, naming `scripts/build_core.sh`, which is
every Windows environment, `check_core_module` refuses it where a core solver or march is built
(`core_backend.require_core`, first in `CoreSolver` and `CoreRun`) and in
`validation.cases.with_backend`, not at load, so a core run's file is read, analysed and reported
on where the core is not built. `SimConfig.with_backend` is the copy that names the backend, in
its attributes, its mapping and a scenario's merged base (from which every interval's flow solve
is built); `validation.cases.with_backend` calls it and looks for the module.

The key is in the run file's configuration and in its restart hash: it is not among
`RESTART_EXCLUDED_KEYS`, so a run file of one backend refuses a restart on the other, naming
`solver.backend`. It is in `flow_inputs` too (not among `FLOW_FREE_SOLVER_KEYS`), so `flows_from`
does not copy a flow state solved on one backend into a run on the other. An explicit
`reference` differs from an absent key in both, as every optional key's default does
(`momentum_sweeps: 1` against none, say); the hashes of every file without the key are unchanged.

### The backend module

`tests/core_driver.py` and `tests/core_mapping.py` are moved whole into `src/core_backend.py`,
the mapping to the reference's exceptions with them. `_cfdcore` is imported only inside the
functions that call it, so the module loads where the core is not built. What changed in the move:

- `CoreRun.monitor_reads_fields` (the mixin `CoreMarch`'s class attribute) defaults to False: the monitor reads the readings and channels
  the core's march returns per step (`AlertMonitor.update_from_readings`), so without `on_step`
  the march goes to the core in batches, to the next frame, checkpoint or interval end. Prompt
  91's stand-in fields (the readings at the sensors' cells, zero elsewhere) are gone. True still
  puts the monitor on the fields, copied out after every step, with the core's readings checked
  against the monitor's bit for bit.
- `CoreSolver.solve_transient` raises: a `CoreSolver` would otherwise run the reference's own
  transient loop under the core's name.
- `steady_solver(mesh, config, boundary, *, reference)` is the one factory: a `CoreSolver` under
  `core`, else the class `reference` names. `time_integration._Run._solve` and
  `scripts/benchmark.py`'s `run_case` each hand it their own module's name for
  `StaggeredSolver`, so a test that substitutes that name (`test_a_diverged_flow_state_stops_the_run`,
  `test_harness_builds_a_wall_clustered_preset_on_its_clustered_mesh`) still builds its substitute.
- `time_integration._run_state` builds `CoreRun` under `core` and `_Run` otherwise; `run` calls
  it where it built `_Run`. Prompt 91's `run` (which patched `_Run` for a call) and prompt 95's
  `steady_solves_in_the_core` (which patched `StaggeredSolver`) are gone; the key replaces both.
- The harness records the backend in `params.backend` through `SOLVER_KEYS`, the schema raised
  to 3; `row_backend` reads a row of schema 2 or before as the reference's. `--backend` selects
  it. The summary keeps the backends apart, and a core row is marked at its end so that a
  reference row prints as every stored row printed before.

The first build of this step had `CoreRun` subclass `_Run` in `core_backend`, which
`time_integration` reached through an import inside the call, so that the generated dependency
map (SYSTEM.md section 3.1) counted a cycle. Round 1 broke it (section 8): `core_backend` imports
nothing of `time_integration`, and `time_integration` imports `core_backend` at its top.

### The monitor's update from readings

`AlertMonitor.update_from_readings(step, t, readings, channels)` is the update from fields with
the reading and the summing taken out: the step and time refusals are the same code
(`_checked_time`), the alerts' opening, closing and peaks the same code (`_advance`), and the
arrays are copied, checked to be [classes, sensors] and finite. It reads the channels it is
handed, the core's sums in `channel_sums`' order (ADR-017 F), and sums nothing itself. SYSTEM.md
section 4 records it, the one interface change outside the new module (ADR-017 E).

### The comparator

`scripts/compare_runs.py` reads two run files of one scenario and gives ADR-017 H's verdict item
by item: every flow state's outer count and stop reason; every arrival time of the maps, every
alert's opening and closing step, the file's sensors' latencies (`AlertMonitor.latencies` on a
monitor restored from the alert table, so the rule is the monitor's), every clear time, the
clean-out's recovery times, the run's end and every class's budget verdict against REQ-T05; each
field's largest difference in its tier; the budget's closure on every stored frame; and the
records' bitwise flags. `compare_placement` gives the decisions that read several scenarios: the
searched layouts' cells under each rule and pool, their order, every unit's latency. The measures
are prompt 85's (`tables85.py`) with three changes: a difference of kind (a value finite in one
run and not in the other, an infinity's sign) reads as infinite, where prompt 85's compared the
finite entries of both; a step is the one the sensor table's step times give, so no plan is
rebuilt; and a middle-tier flow field passes only when some flow state's outer count is one apart and
none more; a middle-tier class field is read as beyond the limit (round 1, section 8), since the
CG and Jacobi counts are not in a run file and an outer count one apart moved the class fields
by 2.7e-11 at most in the probe.

`tests/test_compare_runs.py` pins what test 85 S1 found unpinned: a difference of either sign
read the same, a threshold crossed in one run only read as a move of infinity, a move across an
interval's end read in the later time's step, a field in each of the three tiers, and a failing
verdict of each kind (a decision, a field, a stop, a budget verdict, the closure), on copies of a
real run file with one record planted.

### The tests

New: `tests/test_compare_runs.py` (the measures, and whole run files planted); the
`TestBackendKey` class of `tests/test_config.py`; `TestUpdateFromReadings` in
`tests/test_monitor.py`; `TestBackendKeyInTheRun` in `tests/test_time_integration.py` (the
restart and the kept flow state refused across backends, and the reference's run and solver
under the default); two harness tests in `tests/test_benchmark.py`; `tests/test_core_backend.py`,
a parity module, which runs a short dust release on 80x30 on both backends with a sensor at the
release's cell and compares the core's run with the reference's by the comparator, its restart
from a checkpoint in the march with the uninterrupted core run, its monitor on the readings with
its monitor on the fields, and refuses a restart on the other backend in both directions; and
`tests/test_ci_workflow.py`, which checks that CI's core job names every module marked `parity`.
CI's core job now names each parity module on its own line, `tests/test_core_backend.py` among
them.

Changed, under `reference`: three assertions, each of a fact this step changes, and nothing
else. `tests/test_config.py`'s two tests of `SOLVER_KEYS` count sixteen keys and end with
`backend`; `tests/test_benchmark.py`'s schema assertion reads 3. Under `core`: the parity modules
import from `src/core_backend.py`; `tests/test_core_flow.py`'s test of prompt 95's patch is now a
test of the key; and `tests/test_core_march.py`'s refusal of a stop reason the core's state does
not bear out puts the monitor on the fields, since it relies on a march one step at a time, which
was prompt 91's default. No test that asserts a stored hash is edited (T11).

## 2. The runs

T1 lists them. Each scenario is main's file on 80x30, from rest, through
`scripts/run_scenario.py` in a process of its own (the monitor-on-the-fields runs through its
`main` in the probe's process, since the script has no switch for it). T2 is every flow state's
stop, T3 every decision of the five, T4 every field's largest difference and its tier, T5 the
placement over the four scenarios with a contaminating event. T6 is prediction (a): prompt 91's
three Linux reference runs made again through the script from their own files and flow states,
without `NPY_DISABLE_CPU_FEATURES` as their records were made. T7 is prediction (b), T8 prediction
(d): the march's restart, stopped after step 1,500 and continued from the checkpoint at 1,000,
and the flow solve's, capped at 600 outer iterations and continued under the file's cap, each
against the uninterrupted core run; and the checkpointed run against the one that is not.

What the decisions are read against: at 80x30 only the clean-out's sensors open an alert, and
the seal leak's inlet, 0.025 m high, is narrower than a cell and admits no particle on this grid
(its supply is zero on both sides), so its maps reach no cell. The
monitor's two updates are therefore compared on the clean-out's alerts and on every run's sensor
table, and `tests/test_core_backend.py` adds a run whose sensor at the release opens and closes
alerts.

## 3. The timing

T9: the 80x30 dust release from rest on each backend, three sessions, the flow solve, the march
and the rest apart. Each run is one process, alone, the two backends in alternating order, on
mains power, with no local model server and no game running; Windows' busiest processes are
recorded beside each session. The ratio is a same-session one, both sides under WSL2's clock.

## 4. The planted defects

T10: the prompt's nine, each where it would sit (the key in `RESTART_EXCLUDED_KEYS`, the default
`core`, the readings a step late in `CoreRun`, the channels rolled by a class in
`update_from_readings`, the comparator's difference signed, its tier bounds swapped, a
straddling move read as zero, the other run's times read on the reference's steps (the first
build's was the move read in the earlier time's step, a reading round 1 replaced), the factory
building the reference under `core`), two of them also against the core's whole runs, and ten of
the builder's, one per rule the step added that the prompt's list does not reach. Round 1's are
section 8's: test 96's own plants of the comparator, one for each defect a finding names, the
backend's routes and the harness. Each is made in a
`git archive` copy of the records' commit and its tests run there; the unchanged copy passes the
same tests. One of the builder's survived its first run, `kind_change_as_equal` (a change of kind
read on the finite entries only): `relative` refuses a change of kind twice, by the finite masks
and by the non-finite entries, and every case the first check refuses the second refuses too, so
the plant changes no result. It is recorded as equivalent, and `kind_change_unchecked`, which
removes both checks, is caught. `capped_ignored` is the second equivalent plant (section 8).

## 5. The predictions scored

T0.

## 6. Deviations and limits

- **Four commits before the report, as the prompt lists them**, with no `test:` commit: the
  planted defects ran after the probe's commit and none found a rule untested (T10); test 96
  found most of the comparator's items untested, and round 1 added the tests (section 8).
- **The planted defects' runner changed after the probe's commit**: `variants96.py` records the
  commit of the tree it plants in, which the tables check; it names the equivalent plant with its
  reason; it gained `kind_change_unchecked` and an `--only` that runs named plants and merges
  their records. The code it plants in is the records' commit's, unchanged.
- **The comparator's redundant check stays**: `relative`'s first test of a change of kind is
  implied by its second (section 4). Removing it changes no result, and the code of record is not
  edited after its runs.
- **CTest ran on the build of this branch's `core/`**, which no commit of the branch changes; it
  is main's.
- **The harness's summary marks a core row at its end** rather than in a column, because the
  stored rows' tests read the columns by position and a row's last word as its reference.
- **The comparator reads a run file where its scenario's base resolves**, the machine the run was
  made on, since `SimConfig.from_dict` reads the base again. The probe compares under WSL2 and the
  tables read `compare.json` on either platform. Since round 1 a core run's file is read where
  `_cfdcore` is not built; only its base path need resolve.
- **Not taken up.** Step 6b's work: the band on 320x120, the five scenarios and the placement
  there, the validation rows on the core, the wall time on the three grids. The vents room and the
  QUICK k and eps call that step 3's note carries are not among these runs. The log lines of a
  steady solve on the core are still written when its chunk returns (ADR-017 B, the exchange as
  built).

## 7. Documents updated

ADR-017 B (the exchange as built) and G (the C library), section 0's commit; SYSTEM.md section
3.2 (the config row's key, a row for `core_backend.py`, the driver's mentions renamed), section
3.4 (generated), section 4 (`update_from_readings`) and the history; PROJECT_PLAN.md step 6's
part a; STATUS.md in a sentence. CLAUDE.md does not name `tests/core_driver.py`, so it is
unchanged. Round 1 (prompt 96b) also updated SYSTEM.md (the config and time_integration rows, the
core row, section 4's pointer to `src/march_records.py`, the generated map and the history),
PROJECT_PLAN.md (the step's row, the two files' rows and the history), STATUS.md, ADR-017 H (the
comparator as built) and the annotations of `src/march_records.py`.

## 8. Round 1 (prompt 96b)

Review 96 (REQUEST_CHANGES: B1 to B3, S1 to S7) and test 96 (B1, B2, S1 to S9) found the
comparator wrong in two readings and untested in most of its items, and the backend's routes
and a module cycle open. This section says what each finding got and what pins it. The tables
are regenerated (`tables96.md`) from records made again at 5b53c7a, the round's last code
commit, because the cycle's removal moved `CoreRun` and the code the runs go through: every
run of the five scenarios on both backends, the runs with the monitor on the fields, prompt
91's three, the restarts, the timing sessions, the planted defects, both suites and CTest.
The remade run files are bit for bit the files of 8686a10, all eighteen of them (a comparison of
the two sets with the comparator, `results/builder96b/same_as_before.py`), so the first build's
results stand and what changed is the instrument and the code around it.

### The findings

| Finding | What was done | Pinned by (and planted) |
|---|---|---|
| Review B1: a class field's middle tier explained by an outer count one apart | the four class-field items read it as beyond the limit and say why (`MIDDLE_TIER_NOTE`); flow fields keep the rule | `test_a_class_field_in_the_middle_tier_is_not_explained_by_the_outer_count` (the four items, the outer count equal and one apart), its control inside the spread; `class_*_explained` |
| Test B1: a clear time moved across the last frame read as equal | a time in steps and a clear time are compared by the index of the step or frame each falls on (`grid_index`, `index_moves`), and a move is any nonzero difference of indexes | `test_a_clear_time_that_moves_onto_the_last_frame_is_a_move_of_one` (test 96's `own_clear2.py`, on a run whose last gap is 0.02 of the spacing), the unit tests of the index functions; `clear_move_over_the_gap`, `moves_rounded_to_the_nearest`, `time_on_the_next_step` |
| Test B2, review B2: items pinned by no test | one planted run-file test for each item test 96 lists, on the dust release and on a clean-out made from its flow state | the tests of `tests/test_compare_runs.py`; every plant of T10 that edits `scripts/compare_runs.py`, test 96's among them, rewritten against the code as it stands |
| Review B2: `compare_placement` untested, the first event's onset used for every unit | the onset is the unit's own event's, each run read on its own steps; planted pairs: cells and order that differ, a latency a step later, other step times, a second event in steps twice as long | `test_the_placement_of_a_run_against_its_copy_is_equal_in_every_item` and the five tests after it; `placement_*` |
| Review B3: the signs untested | the leftover lowered by a thousandth of the supply (the verdict and the change), the content lowered by a billionth (the closure), each failing | the three tests of the signs; the three `abs` removals and their negations as plants |
| Review S1: a NaN residual read as closed | the closure returns infinity for it | `test_a_budget_that_is_not_a_number_does_not_close`; `closure_nan_skipped` |
| Test S3: `with_backend` wrote no backend into the run file | `SimConfig.with_backend` writes `solver.backend` into the mapping and a scenario's base | `test_a_copy_writes_its_backend_into_its_mapping`, `test_a_scenario_copy_names_its_backend_in_every_interval`, and on the core `test_a_with_backend_copy_runs_on_the_core_and_its_file_names_it`; `with_backend_*` |
| Test S4, review S5: a core run's file refused where the core is not built | the module is looked for where a core solver or march is built, not at load | `test_core_loads_where_its_module_does_not_import`, `test_the_core_is_refused_where_a_solver_or_a_march_is_built`, `test_a_core_run_file_is_read_where_the_core_is_not_built`; `module_looked_for_at_load*`, `core_*_built_without_the_module` |
| Review S3, test S8: the cycle, private names across modules | `src/march_records.py`; `core_backend.CoreMarch` as a mixin; `time_integration.CoreRun`; `STOP_REASONS`, `field_digest`, `copies_of` and `flush` public | the generated map (`gen_system_map.py --check`: no cycle), the parity modules on `time_integration.CoreRun` |
| Test S1, review S6: the harness on the core untested | the factory is asked for the core, `main` hands the option on, and a `run_case` row on the core has the reference's outer count and error | `test_the_harness_asks_the_factory_for_the_core_under_backend_core`, `test_the_harness_hands_its_backend_option_to_run_case`, `test_a_harness_row_on_the_core_has_the_references_verdict`; `run_case_ignores_backend*`, `main_ignores_backend` |
| Test S2: `monitor_reads_fields` unpinned | the monitor's two updates are counted per run | `test_the_monitor_is_updated_from_the_readings_or_from_the_fields_as_asked`; `fields_mode_reads_readings` |
| Test S6: the bitwise tables print "yes" for a record a pair lacks | T6 to T8 print "n/a" | `tables96.py` |
| Test S7: the restart test is not the batched march | a march stopped from the checkpoint's write, with no `on_step`, restarted | `test_a_restart_from_a_checkpoint_of_the_batched_march_is_the_whole_run`; `march_one_step_a_call` |
| Review S7: T3 does not show what is reached | a decision item says how many entries each run reached, T3 prints it | `test_every_decision_reports_how_many_each_run_reached`; `reached_ref_twice` |
| Review S2, S4, test S9: stale paths, annotations, a repeated import, docstrings | the paths, the three annotations, the import and the docstring named | `ruff`, the review |

### What the corrected comparator says

Every pair of T3 to T8 passes with every field inside its measured spread, so neither new
reading moves a verdict of this branch: the class fields' middle tier and the clear times'
frames were not reached. T3 now shows how many entries each run reaches, which is what "every
decision equal" covers.

### Decisions and limits of the round

- **A second event in a real pair was not obtainable at 80x30.** Three scenarios with a second
  contaminating event in another flow state were tried (the door open at 1.5 and at 0.8 m/s, and
  the supply turned down to 0.2 m/s, each flow state warm-started from the dust release's); each
  ended at the cap of 15,000 outer iterations. The pair of the placement test is the dust release's own
  file with a second event written into its arrival record and every step after that event's
  onset made twice as long, which is what the test needs: a unit whose steps are not the first
  event's.
- **`capped_ignored` is an equivalent mutant.** A state of a capped run is read from the steady
  checkpoint, whose stop reason is the cap's or None; a finished run's state holds one of the
  rule's own reasons. The reasons differ whenever the capped flags do, and the reason is compared.
- **The comparator's plants read run files built once** (`CFD_COMPARE_RUNS`, `build_runs`): no
  plant changes how a run is made, and the unplanted tree's tests, run the same way, pass.
- **`march_one_step_a_call` is caught as a fixture error**: the batched run of `runs` asserts
  that the march went to the core in batches, so under the plant the fixture, and with it every
  test that uses it, errors.
- **The harness row on the core** is the one `val002_40x40` row of `benchmarks/results.jsonl`
  that names `core` (`params.backend`, schema 3), made at 5b53c7a on a clean tree. Its outcome
  is the reference's row of the same case and method (made at 759b3ad, schema 2): 3,435 outer
  iterations, the same stop, `error_estimate_and_continuity`, and the same accuracy value to its
  last digit, 0.00454871200064999; it took 3.4 s where the reference's took 41.0 s.
