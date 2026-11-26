# Phase 6 Step 6a: The Backend Switch and the 80x30 Runs End to End (prompt 96)

**Date:** 2026-11-26
**Prompt:** 96 (`docs/prompts/cc-prompt-96-backend.md`)
**Branch:** `phase6/backend`
**Status:** built. All five predictions hold (T0). Under `reference` nothing changed: prompt
91's three 80x30 runs reproduce bit for bit through `scripts/run_scenario.py`, and no stored
hash test is edited. Under `core` the five product scenarios on 80x30 make every decision of
the reference's runs, every flow state stops at the reference's outer count, and every field
is inside its measured spread; the core's restarts and its two monitor updates are bit for bit.
Every planted defect is caught (one plant is an equivalent mutant, recorded with its reason).
The tables are `docs/reports/probe96/tables96.md`, generated from the records in
`results/builder96/` by `tables96.py` (`--check`); every record names one commit, 8686a10.

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
(`src/config.py`). `SimConfig.check_backend` refuses `core` at load, with the reason, in two
cases: with a transient section, since tier 2's true-time step stays on the reference (ADR-017 F,
option 1); and where `_cfdcore` cannot be imported, naming `scripts/build_core.sh`, which is every
Windows environment. The transient refusal comes first, so its reason is the same where the core
is built and where it is not. `validation.cases.with_backend` sets the key on a copy of a case and
calls the same check.

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

- `CoreRun.monitor_reads_fields` defaults to False: the monitor reads the readings and channels
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

Both `time_integration` and the harness reach `core_backend` through an import inside the call,
since `CoreRun` subclasses `_Run`: the generated dependency map (SYSTEM.md section 3.1) counts the
cycle core_backend to time_integration to core_backend, which is one of a call and not of a
module's load.

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
rebuilt; and a middle-tier field passes only when some flow state's outer count is one apart and
none more, since the CG and Jacobi counts are not in a run file.

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
straddling move read as zero, the move read in the earlier time's step, the factory building the
reference under `core`), two of them also against the core's whole runs, and ten of the
builder's, one per rule the step added that the prompt's list does not reach. Each is made in a
`git archive` copy of the records' commit and its tests run there; the unchanged copy passes the
same tests. One of the builder's survived its first run, `kind_change_as_equal` (a change of kind
read on the finite entries only): `relative` refuses a change of kind twice, by the finite masks
and by the non-finite entries, and every case the first check refuses the second refuses too, so
the plant changes no result. It is recorded as equivalent, and `kind_change_unchecked`, which
removes both checks, is caught.

## 5. The predictions scored

T0.

## 6. Deviations and limits

- **Four commits before the report, as the prompt lists them**, with no `test:` commit: the
  planted defects ran after the probe's commit and none found a rule untested (T10).
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
  tables read `compare.json` on either platform.
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
unchanged.
