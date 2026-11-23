# Phase 6 Step 3b: The March in the Core (prompt 91)

**Date:** 2026-11-22 (round 1, prompt 91b: 2026-11-23)
**Prompt:** 91 (`docs/prompts/cc-prompt-91-march.md`); round 1: 91b (`docs/prompts/cc-prompt-91b-round1-fixes.md`)
**Branch:** `phase6/march`
**Status:** built. Predictions (a), (b), (c) and (d) hold (T0). Six whole scenario runs stepped
through the core, the 320x120 dust release among them, equal the Linux reference bit for bit at
every output step and in every decision, and the restart continues the uninterrupted run bit for
bit; against the Windows records every decision is equal and every field inside the band's
first tier, which meets step 3's gate. The prompt's stop, a whole-march difference, was not met.
The first plant run found two rules no test could see; both are tested now (F2). Round 1
(prompt 91b, after review and test 91) adds the tests the port was missing for the march's
breach and refused step, the within-spec rule and the restore of the recovery state (F5), and
pins the binding's refusals and guards one by one (section 4).

This is the second half of Phase 6 step 3 (`docs/PROJECT_PLAN.md`, Phase 6, the steps table;
ADR-017 B, C, E and F): the step loop in the core. Every class's step in class order, then the
cumulative channels, the arrival maps and the clean-out's recovery, with the fields resident in
the core between steps. Python keeps the configuration, the interval plans, the events, the
alert monitor, the frames and checkpoints, and everything it writes, and calls the core between
output steps as ADR-017 B's exchange table says. Step 6 changes `src/time_integration.py` to call
the core; here a driver in the probe does it, and nothing under `src/` or `configs/` changes.

## The predictions

Set by the orchestrator in prompt 91 on 2026-11-22, and committed alone before any run.

- (a) On every scenario run, the core-driven march equals the Linux reference's bit for bit at
  every output step and in every decision.
- (b) Against the Windows records, every decision is equal and every field is inside the band's
  first tier.
- (c) The restart continues the uninterrupted run bit for bit.
- (d) The 320x120 march window, monitor reads removed, is at least five times faster than the
  reference's, same session.

## How they are scored (the builder, before any run)

The core is built under the project's flags (`CLAUDE.md`, C++ Coding Standards) and called
through its binding under WSL2 with the Linux environment, one BLAS thread. "The reference" is
`time_integration.run` at `origin/main` (01c4649), unchanged. "The core-driven run" is the same
call in a probe process where the run's march (`_Run.march`) is replaced by the probe's driver,
which hands each interval to the core and steps through it; the plans, events, monitor, frames,
checkpoints, restart and end records are the reference's own code. Both runs write their run
files on the Linux file system and copy them into `results/builder91/runs/`.

The scenario runs, each made twice under Linux (the reference and the core-driven run):

| Run | Configuration | Flow states from |
|---|---|---|
| 80x30 dust release, door leak, filter breach | prompt 86's Linux files (`results/builder86/configs/reference/`) | `results/builder86/runs/reference/` |
| 80x30 clean-out, to `t_end` | `configs/scenario_clean_out.yaml` on 80x30 with the 80x30 runs' `mass_imbalance_tol` | the dust release's run file of step 1, when its flow inputs agree; solved once otherwise, and the core-driven run reads the reference's |
| 80x30 clean-out with a hold | the same with `scenario.hold_time` one air change (1/N, 5.562 s) | as the clean-out |
| 320x120 dust release | `results/builder81/product/configs/scenario_equipment_dust.yaml` | `results/builder81/product/equipment_dust.h5` |

What is compared, at every output step (every frame the run file holds): every class field and
every budget term (the frames); the arrival maps and the recovery fields, recorded at each
frame's write by a probe hook in both processes (a digest of each array's bytes at every frame,
the arrays themselves at the end); the sensor readings and channels at every step (the run
file's sensor table). And at the end every decision the run reports: the alert table, the
monitor's latencies and the maps' latencies at the sensors, the recovery record (the clear times,
the 100:1 times, the ratio and the local field), whether the hold stopped the run and at which
step, the steps and the final time, and each class's verdict against REQ-T05 (the accumulated
leftover and the budget's closure over the particles supplied, each below the conservation
limit).

- **(a).** Holds when, on every run of the table, every quantity above equals the reference's in
  every bit (a float compared by its 64-bit pattern) and every decision is equal. One differing
  bit is a failure, and the prompt's stop applies: the first step and the first block that
  differ are found and recorded, and nothing else is changed.
- **(b).** The three 80x30 core-driven runs against prompt 85's Windows records
  (`results/builder85/runs/reference/`), by prompt 86's comparison (`tables86.py`, itself
  `tables85.compare_run` and `tables85.Placement`): the decisions (arrival times, the file's
  sensors' latencies, the layouts' latencies under both rules and every pool, the searched
  cells, the layouts' order, the clear times, every class's verdict against REQ-T05) and each
  field's largest relative difference read against ADR-017 H. Holds when every decision is equal
  and every field is inside the measured spread (the first tier), the budget closing to 1e-12 of
  the supply.
- **(c).** The 80x30 dust release with a checkpoint every 1,000 steps, run through the core
  without a stop, and again through the core stopped after step 1,500 (an exception raised by
  `on_step`, as a crash would end it) and restarted through the core from its checkpoint (step
  1,000, inside the release interval, its sources on). Holds when the restarted run's file equals
  the uninterrupted one's in every bit: every frame, the sensor table from the start, the alert
  table, the arrival maps and the recovery fields at every frame after the restart, and every
  decision.
- **(d).** The 320x120 dust release's march over a window of 200 steps across the release (from
  the frame at step 2,700, about 9.8 s, through the onset at 10 s), started in each case from
  that frame's fields and budgets of the Linux reference's run file with fresh trackers and
  monitor: the reference's `_Run.march` over the window, the core-driven march with the monitor
  reading the fields every step, and the core-driven march with the monitor reading the core's
  returned readings instead (its per-step field reads removed), each in the same process. Three
  sessions, each a process of its own, the machine's state recorded as prompt 90's probe records
  it. The ratio is the reference's wall time over the core-driven one's. Run 1 is scored; runs 2
  and 3 are reported beside it. Holds when the ratio for the march with the monitor's reads
  removed is at least 5.0 in run 1.

A failed prediction is a finding, not a stop.

<!-- tables91 predictions begin -->
### T0. The predictions, scored by the rule committed with them

| Prediction | Measured | Verdict |
|---|---|---|
| (a) the core-driven march equals the Linux reference's bit for bit at every output step and in every decision | six runs, every frame, sensor row and tracker digest without a differing bit: yes (T2); every decision equal: yes (T3) | holds |
| (b) against the Windows records every decision equal and every field inside the band's first tier | every decision equal: yes (T6); 30 of 30 field readings inside the measured spread, the closure at most 5.1e-14 (T5) | holds |
| (c) the restart continues the uninterrupted run bit for bit | every frame, sensor row, tracker digest and end record equal: yes (T4) | holds |
| (d) the 320x120 window, monitor reads removed, at least five times faster than the reference's | 8.37 in session 1 (sessions 2 and 3: 8.13, 8.81); with the monitor's field reads 7.24, 7.73, 7.65 (T9) | holds |
<!-- tables91 predictions end -->

The tables are generated by `docs/reports/probe91/tables91.py` from the records in
`results/builder91/` (untracked, this machine) and copied here between markers;
`tables91.py --check` regenerates them and compares, and fails if a marker pair is missing. The
full set is `docs/reports/probe91/tables91.md`.

## 1. The port

Four modules of ADR-017 C, each a header in `core/include/cfdcore/` and a source in `core/src/`,
templates on the direction count built at 1 and 2 as the others are:

| Module | Owns | Ports | What it computes |
|---|---|---|---|
| `channels` | nothing | `monitor.channel_sum`, `channel_sums`, the monitor's reading at each sensor's cell and `AlertMonitor._sums` | a channel's field, zero then each member's field added in turn, walked by rows; each class's reading at each sensor's cell; each channel's reading at each sensor, the same sum on the readings |
| `arrival` | the maps, [events, channels, cells], and their last update's time | `ArrivalTracker.update`, `state`, `restore` | at each update, for every event whose onset is at or before the update's time, the cells not SOLID, not yet set and at or above their channel's threshold take the update's time |
| `recovery` | the local recovery field, the sensors' and the means' 100:1 times, each channel's and the room's within-spec start, the last update's time | `RecoveryTracker.update`, `held`, `state`, `restore` | the reference's update in its order: the local field per channel, within spec per channel (every sensor below), the room's start, the 100:1 times where defined; `held` as the reference compares it, `>=` |
| `march` | the run: the transport solver of prompt 90, the class fields, the maps, the recovery tracker, the free area, the plan and the step count and time | `_Run.march`'s inner loop, `IntervalPlan.time` | `march(n)`: every class's step in class order, the step count and time, the readings and channels, the channel fields when the maps or the recovery read them, the maps, the means and the recovery, the hold |

The free area is formed as the reference forms it, the non-SOLID cells' volumes in the cell
field's order summed pairwise (`np.outer(dy, dx)[live].sum()`), and each mean is the step's
`MassBudget.current` over it, the reference's `b.current / self.free_area`. The time of step i is
`t_start + i dt`, and the interval's end exactly at its last step. Nothing in a step allocates:
the run allocates its fields and scratch when it is built, and the binding allocates a call's
rows before the core steps.

**The run's interface against ADR-017 B's exchange table.** `_cfdcore.March2` is the table's
run for the march; the flow solve's calls (`Run(setup)`'s flow half, `flow_state`,
`solve_steady`, `flow_fields`) are steps 4 and 5's.

| ADR-017 B's call | `March2` | What crosses |
|---|---|---|
| `Run(setup)` | `March2(scheme, dx_cell, dy_cell, dx_face, dy_face, the transport keys, settling, diffusion, surface_codes, channels, thresholds, sensors, onsets, recovery)` | the mesh's spacings and the scheme on its grid, the transport keys and `t_end`, each class's settling and diffusion, the three surface codes; the channels, the thresholds, the sensors' (j, i) cells, the contaminating events' onsets, and for a clean-out its 100:1 targets, which classes have one, and the hold |
| `interval(plan, conditions, sources)` | `interval(conditions, u, v, v_ext, eddy_viscosity, sources, *, dt, steps, t_start, t_end, first)` | `TransportSolver2.set_interval`'s data and the plan (`IntervalPlan`'s step, step count and ends) with the step the march continues from; the face velocities and eddy viscosity cross here, per interval, since the flow solve is not in the core yet |
| `march(n)` | `march(n)` | per step: the step count and time, every class's budget (ten terms), each class's reading and each channel's at each sensor, each class's free-area mean with a recovery, each class's sweeps and whether its solve converged; and why it stopped (`count`, `interval`, `held` or `breach`, the last with the breach's four values) |
| `class_fields()`, `arrival()`, `recovery()` | the same, and `budgets()`, `held()` | the arrays, for the frame, the checkpoint and the end; the recovery state as `RecoveryTracker.state` names and shapes it |
| `restore(...)` | `restore(fields, budgets, step, t, first_above, arrival_t, recovery)` | a checkpoint's (or a run's start's) fields, budgets as plain data, step count and time, maps and recovery state; the next step is then the uninterrupted run's |

Two differences from the table, both because the flow solve is still Python's: the faces and
the eddy viscosity cross with each interval, not once per flow state; and `restore` is called at
the start of every march, from the reference's own state at t = 0 or at a checkpoint, so a new
run and a restart enter the core the same way. A breach is returned in the dict, not raised: the
steps before it are returned with it, so the driver writes what the reference writes before its
`ConservationBreachError`, then raises that error, made by `tests/core_mapping.py` of the four
values. A refusal of a step's input (a field not finite, a step above `stable_dt`, which no
planned interval meets) raises the class step's `ValueError` with its message.

**The binding and the mapping.** As prompt 90b set it: the binding is handed data and returns
data, NumPy arrays, numbers, dicts and lists, and imports nothing from `src`.
`tests/core_mapping.py` builds a `March2` from a configuration and its mesh (`core_march`), forms
the recovery keys as `RecoveryTracker` forms its own (`recovery_keys`), sets the resident state
from the reference's fields, budgets and trackers (`restore_march`), hands an interval over from
the reference's plan and boundary (`set_interval`), reads a step's budget row as `MassBudget`
objects (`budgets_of_row`), makes the breach's `ConservationBreachError` (`breach_error`), and
sets the reference's trackers from the core's state through their own restores
(`sync_trackers`), so the reference's checks of a map (a time before its event's onset, in a
SOLID cell, after the last update) read the core's maps at every frame. The binding also exposes
the three blocks for the parity tests: `channel_fields`, `sensor_channels`, `ArrivalMaps2` and
`Recovery2`.

**The driver** (`tests/core_driver.py`). `CoreRun` is `time_integration._Run` with `march`
replaced, and `core_driver.run` is `time_integration.run` with `CoreRun` in place of `_Run` for
the call. The plans, the events, the flow states, the monitor, the frames, the checkpoints, the
restart and the end records are the reference's own code; only the march differs. Each interval
is handed to the core; the core is asked for the steps to the next frame, checkpoint or interval
end, or one step at a time while something reads the fields every step: `on_step`, and the
monitor, which reads the fields at every step as the reference hands them, until step 6 gives it
an update from readings. With the monitor on the fields the driver also checks, at every step,
that the core's readings and channels are the monitor's bit for bit. With it on the readings
(`monitor_reads_fields` false, the timing's second core-driven march), the monitor is handed
fields that hold the core's readings at the sensors' cells and zero elsewhere, which it reads
alone. At every frame and checkpoint the reference's trackers are set from the core's state, so
the checkpoint carries the core's maps and recovery fields and the frame hook records them.
Before each interval the driver refuses one whose transport keys, particles or `t_end` are not
those the core was built with: the reference builds a solver per interval from the interval's
configuration, the core one per run.

## 2. CLAUDE.md's kernel rules

**Follows.** The three passes over a field are block functions on `lanes::cells<N>` walked by
`lanes::for_each_block` along each row: a channel's running sum (`add_cells`), an arrival map's
update (`map_cells`) and the local recovery field's (`local_cells`). Their inputs are structs of
pointers, their outputs `CFDCORE_RESTRICT`; the output of each is read and written in place and
overlaps no input, which each header says. A condition is a 64-bit mask (the channel at or
above its threshold, the cell not SOLID, the map not yet set) and the write is `lanes::select`.
GCC vectorises all three (T10). Nothing is allocated in a step. A caller's programming error
(a channel set that is not one channel per class, lengths that do not agree, an interval before
a restore, a plan whose first step is outside it, rows shorter than the steps asked for) is a
`std::invalid_argument`; a step's refusal is the class step's status.

**Departs, and why.** The per-sensor and per-class arithmetic (the readings, the sensors'
channels, within spec, the 100:1 times, the means) are plain loops over a handful of values; a
block would not change a value and would not be faster. The channel fields are formed one
member at a time, a pass per member, as NumPy forms them (`total += values[j]`): a channel of
five classes reads its cells five times, where one pass reading five fields would be one; the
order of each cell's additions is the same either way, and the timing (T9) shows what the march
costs with it.

## 3. The whole-march comparison

The probe (`docs/reports/probe91/probe91.py`) runs each scenario twice under Linux, one BLAS
thread: by `time_integration.run` and by `core_driver.run`, each from the same stored flow
states. The run files are written on the Linux file system and copied into
`results/builder91/runs/`. A hook on `RunFile.write_frame` in both processes records, at every
frame, a digest of the arrival maps and of each recovery array as the run's trackers hold them
then; `probe91.py compare` reads the pairs and writes `results/builder91/compare.json`.

<!-- tables91 runs begin -->
### T1. The runs, each made twice under Linux: the reference and the core-driven march

| Run | Steps | Frames | Intervals | Flow states (core-driven run) | Held | Reference wall, s | Core-driven wall, s |
|---|---|---|---|---|---|---|---|
| 80x30 dust release | 6811 | 70 | 3 | kept | no | 87 | 9 |
| 80x30 door leak | 3416 | 36 | 3 | kept, kept | no | 42 | 4 |
| 80x30 filter breach | 6811 | 70 | 3 | kept | no | 89 | 8 |
| 80x30 clean-out, to t_end | 13620 | 138 | 1 | kept | no | 162 | 16 |
| 80x30 clean-out, with a hold | 1976 | 21 | 1 | kept | yes | 24 | 3 |
| 320x120 dust release | 33119 | 333 | 3 | kept | no | 1344 | 370 |

Each from its stored flow states (`flows_from`): the 80x30 runs from step 1's Linux reference (`results/builder86/runs/reference/`), the clean-out's from the dust release's (its flow inputs are the dust release's), the 320x120 run from `results/builder81/product/equipment_dust.h5`. `kept` is a state copied from that file. The wall times are the whole calls, flow states read and files written included, under Linux, with two 320x120 runs and one 80x30 run at a time; they are for information only. The core-driven march hands the monitor the fields at every step, as the reference does (section 1).

This pass's three 80x30 references against step 1's Linux runs of the same files: frames, budgets, sensor table and arrival maps equal bit for bit, yes (80x30 dust release 70 frames; 80x30 door leak 36 frames; 80x30 filter breach 70 frames).
<!-- tables91 runs end -->

<!-- tables91 march begin -->
### T2. The whole march: the core-driven run against the Linux reference, bit for bit

| Run | Frames | Frames whose class fields differ | Frames whose budget terms differ | Sensor rows | Rows whose readings or channels differ | Frames with tracker digests | Frames whose arrival maps differ | Frames whose recovery fields differ (all six arrays) | End records differing (alerts, arrival, recovery) | Bit for bit |
|---|---|---|---|---|---|---|---|---|---|---|
| 80x30 dust release | 70 | 0 | 0 | 6812 | 0 | 70 | 0 | n/a | none | yes |
| 80x30 door leak | 36 | 0 | 0 | 3417 | 0 | 36 | 0 | n/a | none | yes |
| 80x30 filter breach | 70 | 0 | 0 | 6812 | 0 | 70 | 0 | n/a | none | yes |
| 80x30 clean-out, to t_end | 138 | 0 | 0 | 13621 | 0 | 138 | 0 | 0 | none | yes |
| 80x30 clean-out, with a hold | 21 | 0 | 0 | 1977 | 0 | 21 | 0 | 0 | none | yes |
| 320x120 dust release | 333 | 0 | 0 | 33120 | 0 | 333 | 0 | n/a | none | yes |

`probe91.py compare`: at every frame (every output step and the last), every class field and the ten terms of every class's budget by their 64-bit patterns; every row of the sensor table (each class's reading and each channel's at each sensor, every step); the arrival maps and the six recovery arrays as the run's trackers hold them when each frame is written, by a digest of their bytes (in the core-driven run they are set from the core's state just before); and at the end the alert table, `/arrival/` and `/recovery/` field by field. A 0 is no differing bit.
<!-- tables91 march end -->

<!-- tables91 decisions begin -->
### T3. The decisions each run reports: the reference's against the core-driven run's

| Run | Alerts | Alert table equal | Latencies reached (finite of all) | Latencies and map latencies equal | Recovery record equal (clear times, 100:1 times, ratio, local field) | Held | Hold, steps and final time equal | Largest leftover / supplied; closure / supplied (core-driven) | Every class within REQ-T05, the same on both |
|---|---|---|---|---|---|---|---|---|---|
| 80x30 dust release | 0 | yes | 0 of 20 | yes | n/a | no, 6811 steps | yes | 2.2e-07; 1.5e-14 | yes |
| 80x30 door leak | 0 | yes | 0 of 20 | yes | n/a | no, 3416 steps | yes | 2.4e-07; 5.4e-15 | yes |
| 80x30 filter breach | 0 | yes | 0 of 20 | yes | n/a | no, 6811 steps | yes | 9.1e-08; 5.1e-14 | yes |
| 80x30 clean-out, to t_end | 20 | yes | 20 of 20 | yes | yes | no, 13620 steps | yes | 2.6e-07; 5.7e-15 | yes |
| 80x30 clean-out, with a hold | 20 | yes | 20 of 20 | yes | yes | yes, step 1976 | yes | 1.1e-07; 3.1e-15 | yes |
| 320x120 dust release | 0 | yes | 0 of 20 | yes | n/a | no, 33119 steps | yes | 1.4e-06; 1.8e-13 | yes |

The latencies are the monitor's per event, channel and sensor (`RunRecord.latencies`, inf where never reached) and the maps' at the sensors' cells (`map_latencies`); equal is equal as JSON, every float to its last digit. A class's verdict is its accumulated leftover and its budget's closure each under REQ-T05's 1e-4 of the particles supplied.
<!-- tables91 decisions end -->

**The restart.**

<!-- tables91 restart begin -->
### T4. The restart: the 80x30 dust release stopped after a step and continued through the core, against the run without a stop

| Compared | Count | Differing (fields, or any) | Differing (budgets) |
|---|---|---|---|
| frames, from the start | 70 | 0 | 0 |
| sensor rows, from the start | 6812 | 0 | - |
| frames after the restart with tracker digests | 59 | 0 | - |

Stopped by an exception from `on_step` after step 1500, restarted from its checkpoint at step 1000 (`RunFile.discard_after` keeps the frames and sensor rows up to it), continued to step 6811. End records differing: none; decisions equal: yes. Both runs write a checkpoint every 1,000 steps; the uninterrupted one against the core-driven dust release of T2, which writes none, bit for bit in every frame, sensor row, tracker digest, end record and decision: yes.
<!-- tables91 restart end -->

**The clean-out's restarts** (round 1, test 91 B3). The dust release has no recovery, so T4 does
not restart the one thing a clean-out carries that it does not: the room's within-spec start,
the classes' starts, the means' times and the sensors' 100:1 times. The held 80x30 clean-out,
checkpointed every 100 steps, is stopped after a step and restarted through the core twice:
from a checkpoint before the room's within-spec stretch begins, and from one inside it.

<!-- tables91 cleanrestart begin -->
### T4b. The clean-out's restarts: the 80x30 held clean-out stopped after a step and continued through the core, from before and from inside its within-spec stretch, against the run without a stop

| Restarted | Stopped after step | From step | Room within spec at the checkpoint | Classes' starts set | Mean times set | Frames; differing (fields and budgets) | Sensor rows; differing | Frames after the restart with tracker digests; differing | Record, recovery and map differing | Decisions equal |
|---|---|---|---|---|---|---|---|---|---|---|
| before the stretch | 1650 | 1600 | no | 4 of 5 | 0 of 5 | 21; 0 and 0 | 1977; 0 | 4; 0 | none | yes |
| inside the stretch | 1850 | 1800 | yes, since t = 29.251 s | 5 of 5 | 0 of 5 | 21; 0 and 0 | 1977; 0 | 2; 0 | none | yes |

The held clean-out writes a checkpoint every 100 steps and holds at step 1976 (t = 34.819 s); its within-spec stretch begins at step 1660 (t = 29.251 s), read from the checkpoint inside it. Each restart's file against the uninterrupted run's (`hold80r`), bit for bit in every frame, sensor row, tracker digest, end record and decision; the checkpointed run against the core-driven held clean-out of T2, which writes none: yes. The checkpoint's recovery state is read from the file before the restart. The free-area means do not reach their 100:1 targets within the held run, so the means' times are unset at both checkpoints: the restarts pin the room's start and the classes' starts, and the decay case of section 4 pins the means' times.
<!-- tables91 cleanrestart end -->

**Round 1's runs against round 0's.** The binding changed after the runs of record (the guards
on the plan, time and count arguments, and one reader of the solver's arguments for
`TransportSolver2` and `March2`), and no source under `core/src/` or `core/include/` changed but
comments. Each core-driven run of T2 is round 1's, made again at the head, and each equals round
0's bit for bit:

<!-- tables91 round0 begin -->
### T4c. Round 1's core-driven runs against round 0's: the binding changed under them, the arithmetic did not

| Run | Steps | Frames; differing (fields and budgets) | Sensor rows; differing | Frames with tracker digests; differing | Record, recovery and map differing | Decisions equal |
|---|---|---|---|---|---|---|
| 80x30 dust release | 6811 | 70; 0 and 0 | 6812; 0 | 70; 0 | none | yes |
| 80x30 door leak | 3416 | 36; 0 and 0 | 3417; 0 | 36; 0 | none | yes |
| 80x30 filter breach | 6811 | 70; 0 and 0 | 6812; 0 | 70; 0 | none | yes |
| 80x30 clean-out, to t_end | 13620 | 138; 0 and 0 | 13621; 0 | 138; 0 | none | yes |
| 80x30 clean-out, with a hold | 1976 | 21; 0 and 0 | 1977; 0 | 21; 0 | none | yes |
| 320x120 dust release | 33119 | 333; 0 and 0 | 33120; 0 | 333; 0 | none | yes |

Round 0's files were made at e152bbd, round 1's at the commit its summaries name, each from the same flow states and the same configuration; the six runs of T2 and T3 are round 1's.
<!-- tables91 round0 end -->

**Against the Windows records** (step 3's gate, `docs/PROJECT_PLAN.md`: the march alone on the
reference's stored flow states, every decision inside the band and the budget closing as
ADR-016 states).

<!-- tables91 windows begin -->
### T5. The core-driven runs against the Windows records: each field's largest relative difference, read against ADR-017 H

| scenario | field | difference | measured spread | limit | tier |
|---|---|---|---|---|---|
| 80x30 dust release | u | 6.2e-16 | 7.8e-10 | 1.0e-08 | inside the spread |
| 80x30 dust release | v | 6.4e-16 | 7.8e-10 | 1.0e-08 | inside the spread |
| 80x30 dust release | p | 7.5e-16 | 7.8e-10 | 1.0e-08 | inside the spread |
| 80x30 dust release | k | 1.1e-15 | 7.8e-10 | 1.0e-08 | inside the spread |
| 80x30 dust release | eps | 1.2e-15 | 7.8e-10 | 1.0e-08 | inside the spread |
| 80x30 dust release | nu_t | 1.8e-15 | 7.8e-10 | 1.0e-08 | inside the spread |
| 80x30 dust release | concentration | 1.0e-15 | 2.7e-08 | 1.0e-06 | inside the spread |
| 80x30 dust release | concentration, pointwise | 6.8e-14 | 5.8e-08 | 1.0e-06 | inside the spread |
| 80x30 dust release | sensor channels | 8.9e-15 | 4.7e-08 | 1.0e-06 | inside the spread |
| 80x30 dust release | leftover / supplied, change | 1.2e-16 | 3.6e-10 | 1.0e-08 | inside the spread |
| 80x30 door leak | u | 1.1e-15 | 7.8e-10 | 1.0e-08 | inside the spread |
| 80x30 door leak | v | 8.2e-16 | 7.8e-10 | 1.0e-08 | inside the spread |
| 80x30 door leak | p | 7.5e-16 | 7.8e-10 | 1.0e-08 | inside the spread |
| 80x30 door leak | k | 1.1e-15 | 7.8e-10 | 1.0e-08 | inside the spread |
| 80x30 door leak | eps | 1.9e-15 | 7.8e-10 | 1.0e-08 | inside the spread |
| 80x30 door leak | nu_t | 2.2e-15 | 7.8e-10 | 1.0e-08 | inside the spread |
| 80x30 door leak | concentration | 3.1e-15 | 2.7e-08 | 1.0e-06 | inside the spread |
| 80x30 door leak | concentration, pointwise | 1.6e-14 | 5.8e-08 | 1.0e-06 | inside the spread |
| 80x30 door leak | sensor channels | 1.1e-14 | 4.7e-08 | 1.0e-06 | inside the spread |
| 80x30 door leak | leftover / supplied, change | 1.2e-16 | 3.6e-10 | 1.0e-08 | inside the spread |
| 80x30 filter breach | u | 6.2e-16 | 7.8e-10 | 1.0e-08 | inside the spread |
| 80x30 filter breach | v | 6.4e-16 | 7.8e-10 | 1.0e-08 | inside the spread |
| 80x30 filter breach | p | 7.5e-16 | 7.8e-10 | 1.0e-08 | inside the spread |
| 80x30 filter breach | k | 1.1e-15 | 7.8e-10 | 1.0e-08 | inside the spread |
| 80x30 filter breach | eps | 1.2e-15 | 7.8e-10 | 1.0e-08 | inside the spread |
| 80x30 filter breach | nu_t | 1.8e-15 | 7.8e-10 | 1.0e-08 | inside the spread |
| 80x30 filter breach | concentration | 6.3e-14 | 2.7e-08 | 1.0e-06 | inside the spread |
| 80x30 filter breach | concentration, pointwise | 1.7e-13 | 5.8e-08 | 1.0e-06 | inside the spread |
| 80x30 filter breach | sensor channels | 2.1e-15 | 4.7e-08 | 1.0e-06 | inside the spread |
| 80x30 filter breach | leftover / supplied, change | 4.1e-16 | 3.6e-10 | 1.0e-08 | inside the spread |

Differences as prompt 85's P2 and P6 measure them (`tables85.compare_run`), the Windows run (`results/builder85/runs/reference/`) as the reference: a flow field's over its states, a class's over every stored frame, the pointwise reading where the Windows run reads at least a hundredth of the class's threshold, the sensor channels at every step, the change in the accumulated leftover over the supply at the end. The spreads are P8's (the sensor channels P2's), the limits ten times them rounded up to a decade.

The budget's closure, the reference's own check: the largest residual over the supply at the run's end, 80x30 dust release 1.5e-14; 80x30 door leak 5.4e-15; 80x30 filter breach 5.1e-14, against 1.0e-12. The flow states' outer counts, Windows and the core-driven run's (read from step 1's Linux files): 80x30 dust release [1021] and [1021]; 80x30 door leak [1021, 928] and [1021, 928]; 80x30 filter breach [1021] and [1021].

### T6. The decisions against the Windows records: latencies and clear times moved, the searches, the layouts' order, the budget's verdicts

| decision | compared | moved 1 / > 1 / one only | largest move |
|---|---|---|---|
| arrival times (cells x events x channels) | 26,280 | 0 / 0 / 0 | 0.00 |
| the file's sensors' latencies | 60 | 0 / 0 / 0 | 0.00 |
| the layouts' latencies, worst_case | 60 | 0 / 0 / 0 | 0.00 |
| the layouts' latencies, mean | 60 | 0 / 0 / 0 | 0.00 |
| cells' clear times (cells x units) | 26,280 | 0 / 0 / 0 | 0.00 |
| room clear times (units) | 15 | 0 / 0 / 0 | 0.00 |

| rule | searched layouts whose cells differ | order, Windows | order, core-driven |
|---|---|---|---|
| worst_case | none | free, product, surface, the file's four | free, product, surface, the file's four |
| mean | none | free, product, surface, the file's four | free, product, surface, the file's four |

| scenario | leftover / supplied, Windows (largest class) | core-driven | every class's verdict against REQ-T05 the same |
|---|---|---|---|
| 80x30 dust release | 2.2e-07 | 2.2e-07 | yes |
| 80x30 door leak | 2.4e-07 | 2.4e-07 | yes |
| 80x30 filter breach | 9.1e-08 | 9.1e-08 | yes |

Prompt 85's P3 to P6 measures (`tables85.Placement`, `compare_run`), the Windows run as the reference: a latency moves by its difference over the step of the Windows run's interval, a clear time by its difference over the unit's frame spacing; the searches are `placement.search` over the three runs' maps with main's placement spec, under both rules and every pool.
<!-- tables91 windows end -->

## 4. The tests

<!-- tables91 tests begin -->
### T7. The tests

| Test | Cases | Passed | Failed | Skipped |
|---|---|---|---|---|
| `test_channel_fields_equal_channel_sums` | 4 | 4 | 0 | 0 |
| `test_sensor_channels_equal_the_monitors_and_the_fields_at_the_sensors` | 1 | 1 | 0 | 0 |
| `test_arrival_maps_equal_the_trackers` | 2 | 2 | 0 | 0 |
| `test_recovery_equals_the_tracker` | 2 | 2 | 0 | 0 |
| `test_march_equals_run_march_after_every_step` | 6 | 6 | 0 | 0 |
| `test_march_in_batches_equals_run_march` | 6 | 6 | 0 | 0 |
| `test_march_restored_mid_interval_continues_bit_for_bit` | 2 | 2 | 0 | 0 |
| `test_march_takes_the_steps_asked_for_and_stops_at_the_hold` | 1 | 1 | 0 | 0 |
| `test_march_refuses_what_it_cannot_read` | 1 | 1 | 0 | 0 |
| `test_march_restored_recovery_continues_bit_for_bit` | 2 | 2 | 0 | 0 |
| `test_recovery_within_spec_is_every_sensor_below_its_channels_threshold` | 2 | 2 | 0 | 0 |
| `test_march_breach_stops_as_run_march_stops` | 6 | 6 | 0 | 0 |
| `test_march_stops_as_run_march_does_on_a_field_that_is_not_finite` | 1 | 1 | 0 | 0 |
| `test_march_stops_as_run_march_does_on_a_step_above_the_stable_step` | 1 | 1 | 0 | 0 |
| `test_driver_refuses_a_march_that_returns_no_step_without_holding` | 1 | 1 | 0 | 0 |
| `test_driver_refuses_a_stop_reason_the_cores_state_does_not_bear_out` | 4 | 4 | 0 | 0 |
| `test_driver_refuses_a_count_at_the_intervals_end` | 1 | 1 | 0 | 0 |
| `test_driver_refuses_an_interval_the_core_was_not_built_for` | 1 | 1 | 0 | 0 |
| `test_march_refuses_a_bool_a_float_count_and_a_time_that_is_not_finite` | 20 | 20 | 0 | 0 |
| `test_march_refuses_a_bool_or_a_float_for_the_steps_asked` | 1 | 1 | 0 | 0 |
| `test_march_reads_numpy_numbers_and_a_nan_arrival_time` | 1 | 1 | 0 | 0 |
| `test_march2_refuses_what_it_is_built_from_when_it_cannot_read_it` | 16 | 16 | 0 | 0 |
| `test_march_restore_refuses_a_recovery_state_it_has_no_recovery_for` | 1 | 1 | 0 | 0 |
| `test_march_restore_refuses_an_unaligned_array` | 2 | 2 | 0 | 0 |
| `test_march_restore_refuses_an_unaligned_or_short_recovery_array` | 4 | 4 | 0 | 0 |
| `test_blocks_refuse_what_they_cannot_read` | 14 | 14 | 0 | 0 |
| `test_march_restore_refuses_a_cube_of_another_shape` | 3 | 3 | 0 | 0 |
| `test_march_restore_refuses_maps_of_another_shape` | 4 | 4 | 0 | 0 |
| `test_blocks_refuse_a_cube_of_another_shape` | 9 | 9 | 0 | 0 |
| `test_arrival_maps_refuse_maps_of_another_shape` | 4 | 4 | 0 | 0 |
| `test_march_binding_imports_nothing_from_src` | 1 | 1 | 0 | 0 |
| the parity modules (`test_core_parity.py` and `test_core_march.py`) | 3500 | 3483 | 0 | 17 |
| CTest | 122 | 122 | 0 | 0 |

New GoogleTest cases (23): Channels.AChannelIsZeroThenEachMemberAddedInTurn, Channels.ASensorsChannelIsTheChannelFieldReadAtItsCell, Channels.ASetThatIsNotOneChannelPerClassThrows, Arrival.ACellExactlyAtItsThresholdIsReachedAndOneBelowIsNot, Arrival.AnEventIsMappedFromItsOnsetAndATimeOnceSetIsKept, Arrival.ARestoreCopiesTheMapsAndTheTimeAndRefusesAnotherLength, Recovery.ACellOrASensorExactlyAtItsThresholdHasNotRecovered, Recovery.TheHoldHoldsAtEqualityAndTheTargetsWhereDefined, Recovery.ProgrammingErrorsThrow, Recovery.WithinSpecIsEverySensorBelowItsChannelsThreshold, Recovery.ARestoreCopiesEveryStartAndTimeToItsOwnArray, March.AMarchTakesTheStepsAskedForAndStopsAtTheIntervalsEnd, March.TheIntervalsLastStepIsAtItsEndExactly, March.StepsInBatchesEqualStepsOneAtATime, March.ARunRestoredFromAnothersStateContinuesItBitForBit, March.TheHoldStopsTheMarchAtTheStepItHolds, March.TheMeansAreTheContentOverTheFreeArea, March.ProgrammingErrorsThrow, March.ABreachBeforeAnyStepReturnsNoRowsAndCountsNoStep, March.ABreachAfterStepsReturnsTheRowsBeforeItAndNotItsStep, March.ANonFiniteFieldIsRefusedAtItsClassWithTheClassesBeforeItAdvanced, March.AStepAboveTheStableStepIsRefusedAtTheFirstClass, March.ARunRestoredInsideItsWithinSpecStretchKeepsItsStartsAndTimes.
- The whole suite under Linux: 6584 passed, 0 failed, 17 skipped in 1549 s.
- The whole suite under Windows: 3101 passed, 0 failed, 2 skipped in 1631 s.
<!-- tables91 tests end -->

**The parity cases** (`tests/test_core_march.py`, marker `parity`, run by CI's core job beside
`tests/test_core_parity.py`). The blocks against the reference's functions on planted fields,
on the product room and the vents room, 80x30 and 160x60: the channel fields against
`channel_sums` under four channel sets (the configuration's, the same reversed, each class alone,
another mix), on values of both signs over twelve decades with a -0.0 cell, and showing that the
reversed order gives other bits; the sensors' channels against the monitor's and against the
channel fields read at the sensors' cells; the arrival maps against `ArrivalTracker` after each
of seven updates, with values around every threshold, a cell exactly at it and one ulp either
side, four events with their onsets (two sharing one, one after the first updates), SOLID cells
at 1e30 and a cell that is reached, falls below and rises again; the recovery against
`RecoveryTracker` after each of eight updates, `held()` included, with cells and sensors exactly
at a threshold, readings and means exactly at the 100:1 target, a class whose target is not
defined, a within-spec stretch whose length is the hold exactly and a sensor at its threshold
after it. Each asserts that its planted inputs reach the rule they are planted for. Round 1 adds
one for the within-spec rule itself: updates that tip each channel in turn at each sensor in turn
by its class alone, at its threshold exactly, and others that tip a channel by a class below its
own threshold whose channel is not, each followed by updates back below, on each room; the core's
state equals the tracker's after every one, a within-spec decision made from the readings alone
equals the tracker's, and the planted updates are shown to separate the rule from the three wrong
ones (the first sensor alone, each class on its own reading, the room within spec when any channel
is). A GoogleTest does the same on three sensors and two channels with the classes' own readings
below their thresholds throughout, so that only a channel's reading can leave spec. The recovery
case also drives class 3's mean under its own target, which is not defined, so that no time is set
for it.

The march against `_Run.march` on the 80x30 product mesh with planted flows (faces of both signs
about 0.2 m/s, a positive eddy viscosity), in six cases: across the dust release's onset; across
the door leak's two flow switches; the clean-out from fields around its thresholds; the clean-out
within spec from the start, held at its fifth step by a hold equal to that step's time; the door
leak's last interval to its end, whose last step is at `t_end` and not at `t_start + steps dt`
(added by 03030b9); and the clean-out from the third step of its first interval with fields that
are zero but at the sensors' cells, where the first three classes start above their thresholds,
so that the channels fall within spec at different steps (the room's at the eleventh) and the
hold ends the march at the fifteenth (round 1). Each is driven the same way from the same
start (`_Run` and `CoreRun` on the same configuration, plans cut to a few steps of each
interval, the trackers updated at the start as `run` does at t = 0),
and after every step the fields, the budget terms, the maps, the recovery fields, the readings,
the channels and the means are compared bit for bit; at the end the fields, budgets, step count,
time and the hold. Each case again with the march in batches and the monitor on the readings. A
second core run restored from the first's state mid-interval continues it bit for bit (the
clean-out, with its recovery fields, and the dust release); `march(3)` takes three steps; the
held clean-out stops at the reference's step and a later march takes none; the binding's own
refusals; and a restore, a march and every read with every `src` module unimportable.

**The stops a march ends in** (round 1, review 91 B1, test 91 B1). Six scenario cases each place
a breach of ADR-016's limit with a limit between two consecutive leftover ratios (under a solve
capped at two sweeps with no share of the limit, so that a step's leftover is the same whatever
the limit): in class 0, 1 and 4, before any step, after one and after three, three of them in
the first or second step of a second interval. The reference's `_Run.march` and the core's
march, every step and in batches, raise `ConservationBreachError` with the same class,
leftover, supplied and limit in every bit and the same message, after the same steps with the
same fields and budgets, leaving the same run file (the frames and the checkpoint written before
the breach, the readings not yet flushed dropped). A field with a NaN in class 2, and a plan
whose step is twice the stable step, are refused by both with the same `ValueError`. The
GoogleTests place a breach by the same means in class 0 before any step and in class 1 after
two, with the rows before it equal to those of a run no limit reaches and the rows after it
unwritten, the step and time those of the last counted step, the step half taken (the budgets
of the classes up to the failing one booked it, the ones after it did not), and a restore from
the clean run's state breaching again at the same place with the same numbers; and a refused
field and a refused step, with the classes before the failing one advanced.

**The restore of the recovery state** (round 1, test 91 B3). The decay case, restored from the
first core run's state at step 8 (before the room's stretch, which begins at step 11) and at
step 13 (inside it), reaches the hold at step 15 with every array of the state the first run
reaches, the room's start, the classes' starts, the means' times and the sensors' times, each a
different array; the state handed over is asserted to hold what a restore must not lose or mix
up. A GoogleTest restores a run inside a stretch whose hold it must reach at the same step, and
`expect_same_state` compares the means' times as well. `tracker::restore` has a case that gives
every array its own time. The whole clean-out's restarts are T4b.

**The driver and the binding** (round 1, test 91 S1 to S3, review 91 S4, S5, S8). The driver
reads the core's stop reason and refuses one its state does not bear out (the interval's end
with steps left, a count with none, a hold the run does not hold, a breach without its
record), and a march that returned no step and neither held nor breached is an error where it
was a loop; a stand-in core shows each. An interval whose transport keys, particles or `t_end`
are not the run's is refused. The plan, time and count a restore, an interval or a march reads
are refused when they are a bool, a float where a count is read, a time that is not finite or a
negative step count; numbers NumPy hands over and a NaN arrival time (before the maps' first
update) are not. Each
refusal of `March2`'s construction, `ArrivalMaps2`, `Recovery2`, `channel_fields`,
`sensor_channels` and `March2.restore` has a case that reaches it, and a plant removing it
(T8b).

**The GoogleTests** (`core/tests/test_march.cpp`) need no reference: a channel is zero and then
each member in turn (an order of 1, 1e16 and -1e16 that gives 0 one way and 1 another, and a
-0.0 that becomes +0.0); a sensor's channel is the channel field at its cell; a cell exactly at
its threshold is reached and one ulp below is not; an event is mapped from its onset and a time
once set is kept; SOLID cells stay NaN; a cell or a sensor at its threshold has not recovered;
the hold holds at equality; the 100:1 times only where defined; `march(n)` takes the steps asked
for and stops at the interval's end with its last step at `t_end`; batches equal single steps; a
run restored from another's state continues it; the hold stops the march at the step it holds;
the means are the content over the free area; the programming errors throw.

## 5. The planted defects

<!-- tables91 plants begin -->
### T8. The planted defects: which tests catch each

Each built alone from the checkout's `core/` with its edits (`docs/reports/probe91/variants91.py`), then CTest, the march's parity module (`tests/test_core_march.py`), and two whole 80x30 runs through the core (the dust release and the held clean-out) against this pass's Linux references.

| Plant | What it changes | CTest | Parity cases failing (by test) | Whole runs against the reference |
|---|---|---|---|---|
| `control` | no edit: every test passes | 100% tests passed, 0 tests failed out of 122 | none | dust80 equal, hold80 equal |
| `channel_order_reversed` | a channel's field summed over its members last to first | 98% tests passed, 2 tests failed out of 122: `Channels.AChannelIsZeroThenEachMemberAddedInTurn`, `Channels.ASensorsChannelIsTheChannelFieldReadAtItsCell` | `test_channel_fields_equal_channel_sums` 4, `test_sensor_channels_equal_the_monitors_and_the_fields_at_the_sensors` 1 | dust80 equal, hold80 equal |
| `threshold_cell_recovered` | a cell exactly at its threshold read as recovered: the local field set only above it | 99% tests passed, 1 tests failed out of 122: `Recovery.ACellOrASensorExactlyAtItsThresholdHasNotRecovered` | `test_recovery_equals_the_tracker` 2, `test_recovery_within_spec_is_every_sensor_below_its_channels_threshold` 2 | dust80 equal, hold80 equal |
| `arrival_before_onset` | every event's map updated at every step, before its onset too | 98% tests passed, 2 tests failed out of 122: `Arrival.AnEventIsMappedFromItsOnsetAndATimeOnceSetIsKept`, `March.StepsInBatchesEqualStepsOneAtATime` | `test_arrival_maps_equal_the_trackers` 2 | dust80 equal, hold80 equal |
| `arrival_overwritten` | a cell's arrival time overwritten by every later step at or above the threshold | 99% tests passed, 1 tests failed out of 122: `Arrival.AnEventIsMappedFromItsOnsetAndATimeOnceSetIsKept` | `test_arrival_maps_equal_the_trackers` 2, `test_march_equals_run_march_after_every_step` 5 | dust80 differs: 0/70 frames, arrival, hold80 differs: 0/21 frames, arrival |
| `hold_strict` | the hold compared strictly: the stretch must exceed the hold time | 98% tests passed, 3 tests failed out of 122: `Recovery.TheHoldHoldsAtEqualityAndTheTargetsWhereDefined`, `March.TheHoldStopsTheMarchAtTheStepItHolds`, `March.ARunRestoredInsideItsWithinSpecStretchKeepsItsStartsAndTimes` | `test_march_equals_run_march_after_every_step` 2, `test_march_in_batches_equals_run_march` 2, `test_march_restored_recovery_continues_bit_for_bit` 2, `test_march_takes_the_steps_asked_for_and_stops_at_the_hold` 1, `test_recovery_equals_the_tracker` 2, `test_recovery_within_spec_is_every_sensor_below_its_channels_threshold` 2 | dust80 equal, hold80 equal |
| `free_area_whole` | the free-area mean over the whole domain's area, SOLID cells included | 99% tests passed, 1 tests failed out of 122: `March.TheMeansAreTheContentOverTheFreeArea` | `test_march_equals_run_march_after_every_step` 6 | dust80 equal, hold80 equal |
| `restore_drops_recovery` | a restore that sets the fields, budgets and maps and drops the recovery fields | 98% tests passed, 2 tests failed out of 122: `March.ARunRestoredFromAnothersStateContinuesItBitForBit`, `March.ARunRestoredInsideItsWithinSpecStretchKeepsItsStartsAndTimes` | `test_march_equals_run_march_after_every_step` 3, `test_march_in_batches_equals_run_march` 1, `test_march_restored_mid_interval_continues_bit_for_bit` 1, `test_march_restored_recovery_continues_bit_for_bit` 2, `test_march_takes_the_steps_asked_for_and_stops_at_the_hold` 1 | dust80 equal, hold80 equal |
| `march_one_too_many` | march(n) steps n + 1 steps, its rows sized for them | 96% tests passed, 5 tests failed out of 122: `March.AMarchTakesTheStepsAskedForAndStopsAtTheIntervalsEnd`, `March.StepsInBatchesEqualStepsOneAtATime`, `March.ARunRestoredFromAnothersStateContinuesItBitForBit`, `March.ABreachAfterStepsReturnsTheRowsBeforeItAndNotItsStep`, `March.ARunRestoredInsideItsWithinSpecStretchKeepsItsStartsAndTimes` | `test_march_breach_stops_as_run_march_stops` 4, `test_march_equals_run_march_after_every_step` 6, `test_march_reads_numpy_numbers_and_a_nan_arrival_time` 1, `test_march_restored_mid_interval_continues_bit_for_bit` 2, `test_march_restored_recovery_continues_bit_for_bit` 2, `test_march_takes_the_steps_asked_for_and_stops_at_the_hold` 1 | dust80 raised AssertionError, hold80 raised AssertionError |
| `sensor_channel_order_reversed` | a channel's reading at a sensor summed over its members last to first | 99% tests passed, 1 tests failed out of 122: `Channels.ASensorsChannelIsTheChannelFieldReadAtItsCell` | `test_march_breach_stops_as_run_march_stops` 4, `test_march_equals_run_march_after_every_step` 6, `test_sensor_channels_equal_the_monitors_and_the_fields_at_the_sensors` 1 | dust80 raised AssertionError, hold80 raised AssertionError |
| `channel_not_from_zero` | a channel started from its first member's value, not from zero (-0.0 kept) | 99% tests passed, 1 tests failed out of 122: `Channels.AChannelIsZeroThenEachMemberAddedInTurn` | `test_channel_fields_equal_channel_sums` 4 | dust80 equal, hold80 equal |
| `threshold_sensor_within` | a sensor exactly at its threshold read as within spec | 98% tests passed, 3 tests failed out of 122: `Recovery.ACellOrASensorExactlyAtItsThresholdHasNotRecovered`, `Recovery.TheHoldHoldsAtEqualityAndTheTargetsWhereDefined`, `Recovery.WithinSpecIsEverySensorBelowItsChannelsThreshold` | `test_recovery_equals_the_tracker` 2, `test_recovery_within_spec_is_every_sensor_below_its_channels_threshold` 2 | dust80 equal, hold80 equal |
| `arrival_in_solid` | the arrival maps set in SOLID cells too | 99% tests passed, 1 tests failed out of 122: `Arrival.AnEventIsMappedFromItsOnsetAndATimeOnceSetIsKept` | `test_arrival_maps_equal_the_trackers` 2 | dust80 equal, hold80 equal |
| `room_since_kept` | the room's within-spec start kept when a sensor rises above its threshold again | 98% tests passed, 2 tests failed out of 122: `Recovery.TheHoldHoldsAtEqualityAndTheTargetsWhereDefined`, `Recovery.WithinSpecIsEverySensorBelowItsChannelsThreshold` | `test_recovery_equals_the_tracker` 2, `test_recovery_within_spec_is_every_sensor_below_its_channels_threshold` 2 | dust80 equal, hold80 equal |
| `sensor_time_overwritten` | a sensor's 100:1 time overwritten by every later reading at or below the target | 98% tests passed, 2 tests failed out of 122: `Recovery.TheHoldHoldsAtEqualityAndTheTargetsWhereDefined`, `Recovery.ARestoreCopiesEveryStartAndTimeToItsOwnArray` | `test_march_equals_run_march_after_every_step` 3, `test_recovery_equals_the_tracker` 2, `test_recovery_within_spec_is_every_sensor_below_its_channels_threshold` 2 | dust80 equal, hold80 differs: 0/21 frames, recovery |
| `held_not_checked_at_entry` | a march that starts on a held run steps on | 99% tests passed, 1 tests failed out of 122: `March.TheHoldStopsTheMarchAtTheStepItHolds` | `test_march_takes_the_steps_asked_for_and_stops_at_the_hold` 1 | dust80 equal, hold80 equal |
| `last_time_not_t_end` | the interval's last step at t_start + steps dt, not at t_end exactly | 99% tests passed, 1 tests failed out of 122: `March.TheIntervalsLastStepIsAtItsEndExactly` | `test_march_equals_run_march_after_every_step` 1, `test_march_in_batches_equals_run_march` 1 | dust80 differs: 0/70 frames, arrival, hold80 equal |

16 of 16 plants fail at least one test. In the last column a count of frames is of the frames whose class fields or budget terms differ, then what else differs (the sensor table, `/arrival/`, `/recovery/`); `raised` is a run the driver stopped, its check of the core's readings against the monitor's, or of a march's rows against the frames, failing. The whole-run column cannot see a breach, a refusal or a restore: neither run breaches, is refused or restarts, so a plant in those rules reads `equal` there whatever it changes; they are T8b's, which are not run on whole runs, and T4b's whole restarts.
<!-- tables91 plants end -->

**Round 1's plants** (`docs/reports/probe91/variants91b.py`). The tester's plants of test 91, run
again, the plants for each rule round 1's tests pin, and one plant per refusal and guard of the
binding, each removing it. They are not run on whole runs: the two whole runs of T8 breach
nowhere, refuse nothing and restart nowhere, so the column cannot see these rules whatever a
plant changes in them (test 91 S4). The whole restarts are T4b.

<!-- tables91 plantsround1 begin -->
### T8b. Round 1's planted defects: which tests catch each

Each built alone from the checkout's `core/` with its edits (`docs/reports/probe91/variants91b.py`), then CTest and the march's parity module (`tests/test_core_march.py`); no whole runs. The plants are the tester's of test 91 (`results/tester91/plant.py`, its anchors updated where the binding changed), the plants of round 1's new tests, one plant per refusal of the binding and one per guard of the plan, time and count inputs. Each stream builds the unplanted core first and last.

| Plant | CTest cases failing | Parity cases failing (by test) | Caught |
|---|---|---|---|
| `breach_ignored` | 2: `March.ABreachBeforeAnyStepReturnsNoRowsAndCountsNoStep`, `March.ABreachAfterStepsReturnsTheRowsBeforeItAndNotItsStep` | `test_march_breach_stops_as_run_march_stops` 6 | yes |
| `refusal_ignored` | 2: `March.ANonFiniteFieldIsRefusedAtItsClassWithTheClassesBeforeItAdvanced`, `March.AStepAboveTheStableStepIsRefusedAtTheFirstClass` | `test_march_stops_as_run_march_does_on_a_field_that_is_not_finite` 1, `test_march_stops_as_run_march_does_on_a_step_above_the_stable_step` 1 | yes |
| `breach_step_counted` | 2: `March.ABreachBeforeAnyStepReturnsNoRowsAndCountsNoStep`, `March.ABreachAfterStepsReturnsTheRowsBeforeItAndNotItsStep` | `test_march_breach_stops_as_run_march_stops` 6 | yes |
| `breach_step_advanced` | 2: `March.ABreachBeforeAnyStepReturnsNoRowsAndCountsNoStep`, `March.ABreachAfterStepsReturnsTheRowsBeforeItAndNotItsStep` | none | yes |
| `refused_step_advanced` | 2: `March.ANonFiniteFieldIsRefusedAtItsClassWithTheClassesBeforeItAdvanced`, `March.AStepAboveTheStableStepIsRefusedAtTheFirstClass` | none | yes |
| `breach_leftover_is_the_steps` | 0 | `test_march_breach_stops_as_run_march_stops` 4 | yes |
| `breach_leftover_supplied_swapped` | 0 | `test_march_breach_stops_as_run_march_stops` 6 | yes |
| `breach_supplied_and_limit_swapped` | 0 | `test_march_breach_stops_as_run_march_stops` 6 | yes |
| `breach_limit_wrong` | 0 | `test_march_breach_stops_as_run_march_stops` 6 | yes |
| `breach_class_zero` | 0 | `test_march_breach_stops_as_run_march_stops` 3 | yes |
| `breach_raised_as_refusal` | 0 | `test_march_breach_stops_as_run_march_stops` 6 | yes |
| `refusal_returned_not_raised` | 0 | `test_march_stops_as_run_march_does_on_a_field_that_is_not_finite` 1, `test_march_stops_as_run_march_does_on_a_step_above_the_stable_step` 1 | yes |
| `breach_without_record` | 0 | `test_march_breach_stops_as_run_march_stops` 6 | yes |
| `within_first_sensor_only` | 1: `Recovery.WithinSpecIsEverySensorBelowItsChannelsThreshold` | `test_recovery_equals_the_tracker` 2, `test_recovery_within_spec_is_every_sensor_below_its_channels_threshold` 2 | yes |
| `within_reads_the_class_not_the_channel` | 1: `Recovery.WithinSpecIsEverySensorBelowItsChannelsThreshold` | `test_march_equals_run_march_after_every_step` 1, `test_march_in_batches_equals_run_march` 1, `test_march_restored_recovery_continues_bit_for_bit` 2, `test_recovery_within_spec_is_every_sensor_below_its_channels_threshold` 2 | yes |
| `within_reads_the_first_class_channel` | 2: `Recovery.TheHoldHoldsAtEqualityAndTheTargetsWhereDefined`, `Recovery.WithinSpecIsEverySensorBelowItsChannelsThreshold` | `test_march_equals_run_march_after_every_step` 2, `test_march_in_batches_equals_run_march` 2, `test_march_restored_recovery_continues_bit_for_bit` 2, `test_march_takes_the_steps_asked_for_and_stops_at_the_hold` 1, `test_recovery_within_spec_is_every_sensor_below_its_channels_threshold` 2 | yes |
| `room_within_when_any_class_is` | 2: `Recovery.TheHoldHoldsAtEqualityAndTheTargetsWhereDefined`, `Recovery.WithinSpecIsEverySensorBelowItsChannelsThreshold` | `test_march_equals_run_march_after_every_step` 1, `test_march_in_batches_equals_run_march` 1, `test_march_restored_recovery_continues_bit_for_bit` 2, `test_recovery_within_spec_is_every_sensor_below_its_channels_threshold` 2 | yes |
| `mean_time_where_undefined` | 2: `Recovery.TheHoldHoldsAtEqualityAndTheTargetsWhereDefined`, `March.ARunRestoredInsideItsWithinSpecStretchKeepsItsStartsAndTimes` | `test_recovery_equals_the_tracker` 2, `test_recovery_within_spec_is_every_sensor_below_its_channels_threshold` 2 | yes |
| `sensor_time_where_undefined` | 2: `Recovery.TheHoldHoldsAtEqualityAndTheTargetsWhereDefined`, `March.ARunRestoredInsideItsWithinSpecStretchKeepsItsStartsAndTimes` | `test_recovery_equals_the_tracker` 2, `test_recovery_within_spec_is_every_sensor_below_its_channels_threshold` 2 | yes |
| `mean_time_overwritten` | 0 | `test_march_equals_run_march_after_every_step` 2, `test_recovery_equals_the_tracker` 2, `test_recovery_within_spec_is_every_sensor_below_its_channels_threshold` 2 | yes |
| `mean_at_target_not_reached` | 1: `Recovery.TheHoldHoldsAtEqualityAndTheTargetsWhereDefined` | `test_recovery_equals_the_tracker` 2 | yes |
| `reading_at_target_not_reached` | 1: `Recovery.TheHoldHoldsAtEqualityAndTheTargetsWhereDefined` | `test_recovery_equals_the_tracker` 2 | yes |
| `target_reads_the_channel_not_the_class` | 1: `March.ARunRestoredFromAnothersStateContinuesItBitForBit` | `test_march_equals_run_march_after_every_step` 1, `test_recovery_equals_the_tracker` 2 | yes |
| `class_since_kept` | 2: `Recovery.TheHoldHoldsAtEqualityAndTheTargetsWhereDefined`, `Recovery.WithinSpecIsEverySensorBelowItsChannelsThreshold` | `test_recovery_equals_the_tracker` 2, `test_recovery_within_spec_is_every_sensor_below_its_channels_threshold` 2 | yes |
| `class_since_restarted_every_step` | 3: `Recovery.TheHoldHoldsAtEqualityAndTheTargetsWhereDefined`, `Recovery.ARestoreCopiesEveryStartAndTimeToItsOwnArray`, `March.ARunRestoredInsideItsWithinSpecStretchKeepsItsStartsAndTimes` | `test_march_equals_run_march_after_every_step` 2, `test_recovery_equals_the_tracker` 2, `test_recovery_within_spec_is_every_sensor_below_its_channels_threshold` 2 | yes |
| `room_since_restarted_every_step` | 4: `Recovery.TheHoldHoldsAtEqualityAndTheTargetsWhereDefined`, `Recovery.ARestoreCopiesEveryStartAndTimeToItsOwnArray`, `March.TheHoldStopsTheMarchAtTheStepItHolds`, `March.ARunRestoredInsideItsWithinSpecStretchKeepsItsStartsAndTimes` | `test_march_equals_run_march_after_every_step` 2, `test_march_in_batches_equals_run_march` 2, `test_march_restored_recovery_continues_bit_for_bit` 2, `test_march_takes_the_steps_asked_for_and_stops_at_the_hold` 1, `test_recovery_equals_the_tracker` 2, `test_recovery_within_spec_is_every_sensor_below_its_channels_threshold` 2 | yes |
| `local_field_in_solid` | 0 | `test_recovery_equals_the_tracker` 2 | yes |
| `tracker_restore_swaps_mean_and_class` | 2: `Recovery.ARestoreCopiesEveryStartAndTimeToItsOwnArray`, `March.ARunRestoredInsideItsWithinSpecStretchKeepsItsStartsAndTimes` | `test_march_equals_run_march_after_every_step` 1, `test_march_restored_recovery_continues_bit_for_bit` 2, `test_recovery_equals_the_tracker` 2 | yes |
| `march_restore_room_since_zero` | 2: `March.TheHoldStopsTheMarchAtTheStepItHolds`, `March.ARunRestoredInsideItsWithinSpecStretchKeepsItsStartsAndTimes` | `test_march_restored_recovery_continues_bit_for_bit` 2 | yes |
| `march_restore_swaps_mean_and_class` | 1: `March.ARunRestoredInsideItsWithinSpecStretchKeepsItsStartsAndTimes` | `test_march_equals_run_march_after_every_step` 1, `test_march_restored_recovery_continues_bit_for_bit` 2 | yes |
| `binding_restore_room_since_zero` | 0 | `test_march_restored_recovery_continues_bit_for_bit` 2 | yes |
| `binding_restore_room_since_nan` | 0 | `test_march_equals_run_march_after_every_step` 1, `test_march_in_batches_equals_run_march` 1, `test_march_restored_recovery_continues_bit_for_bit` 1, `test_march_takes_the_steps_asked_for_and_stops_at_the_hold` 1 | yes |
| `binding_restore_class_since_is_mean_time` | 0 | `test_march_restored_recovery_continues_bit_for_bit` 2 | yes |
| `binding_restore_mean_time_is_class_since` | 0 | `test_march_equals_run_march_after_every_step` 1, `test_march_restored_recovery_continues_bit_for_bit` 2 | yes |
| `restore_drops_step_and_time` | 2: `March.ARunRestoredFromAnothersStateContinuesItBitForBit`, `March.ARunRestoredInsideItsWithinSpecStretchKeepsItsStartsAndTimes` | `test_march_breach_stops_as_run_march_stops` 3, `test_march_equals_run_march_after_every_step` 4, `test_march_in_batches_equals_run_march` 4, `test_march_reads_numpy_numbers_and_a_nan_arrival_time` 1, `test_march_restored_mid_interval_continues_bit_for_bit` 2, `test_march_restored_recovery_continues_bit_for_bit` 2 | yes |
| `restore_drops_maps` | 1: `March.ARunRestoredFromAnothersStateContinuesItBitForBit` | `test_march_equals_run_march_after_every_step` 3, `test_march_reads_numpy_numbers_and_a_nan_arrival_time` 1, `test_march_restored_mid_interval_continues_bit_for_bit` 2, `test_march_restored_recovery_continues_bit_for_bit` 2 | yes |
| `restore_drops_budgets` | 4: `March.ARunRestoredFromAnothersStateContinuesItBitForBit`, `March.ABreachBeforeAnyStepReturnsNoRowsAndCountsNoStep`, `March.ABreachAfterStepsReturnsTheRowsBeforeItAndNotItsStep`, `March.ARunRestoredInsideItsWithinSpecStretchKeepsItsStartsAndTimes` | `test_march_restored_mid_interval_continues_bit_for_bit` 2, `test_march_restored_recovery_continues_bit_for_bit` 2 | yes |
| `restore_drops_fields` | 4: `March.StepsInBatchesEqualStepsOneAtATime`, `March.ABreachBeforeAnyStepReturnsNoRowsAndCountsNoStep`, `March.ABreachAfterStepsReturnsTheRowsBeforeItAndNotItsStep`, `March.ANonFiniteFieldIsRefusedAtItsClassWithTheClassesBeforeItAdvanced` | `test_march_breach_stops_as_run_march_stops` 6, `test_march_equals_run_march_after_every_step` 6, `test_march_in_batches_equals_run_march` 6, `test_march_restored_recovery_continues_bit_for_bit` 2, `test_march_stops_as_run_march_does_on_a_field_that_is_not_finite` 1 | yes |
| `hold_not_checked_after_a_step` | 2: `March.TheHoldStopsTheMarchAtTheStepItHolds`, `March.ARunRestoredInsideItsWithinSpecStretchKeepsItsStartsAndTimes` | `test_march_in_batches_equals_run_march` 2, `test_march_restored_recovery_continues_bit_for_bit` 2, `test_march_takes_the_steps_asked_for_and_stops_at_the_hold` 1 | yes |
| `interval_end_reported_as_count` | 1: `March.AMarchTakesTheStepsAskedForAndStopsAtTheIntervalsEnd` | `test_driver_refuses_a_count_at_the_intervals_end` 1, `test_march_breach_stops_as_run_march_stops` 3, `test_march_equals_run_march_after_every_step` 4, `test_march_in_batches_equals_run_march` 4 | yes |
| `count_reported_as_interval` | 1: `March.AMarchTakesTheStepsAskedForAndStopsAtTheIntervalsEnd` | `test_march_binding_imports_nothing_from_src` 1, `test_march_breach_stops_as_run_march_stops` 4, `test_march_equals_run_march_after_every_step` 6, `test_march_takes_the_steps_asked_for_and_stops_at_the_hold` 1 | yes |
| `channels_only_with_recovery` | 0 | `test_march_equals_run_march_after_every_step` 3 | yes |
| `means_from_class_zero` | 1: `March.TheMeansAreTheContentOverTheFreeArea` | `test_march_equals_run_march_after_every_step` 3 | yes |
| `time_accumulated` | 2: `March.AMarchTakesTheStepsAskedForAndStopsAtTheIntervalsEnd`, `March.TheIntervalsLastStepIsAtItsEndExactly` | `test_march_equals_run_march_after_every_step` 4 | yes |
| `interval_first_ignored` | 2: `March.ARunRestoredFromAnothersStateContinuesItBitForBit`, `March.ARunRestoredInsideItsWithinSpecStretchKeepsItsStartsAndTimes` | `test_march_breach_stops_as_run_march_stops` 3, `test_march_equals_run_march_after_every_step` 4, `test_march_in_batches_equals_run_march` 4, `test_march_reads_numpy_numbers_and_a_nan_arrival_time` 1, `test_march_restored_mid_interval_continues_bit_for_bit` 2, `test_march_restored_recovery_continues_bit_for_bit` 2 | yes |
| `row_inflow_outflow_swapped` | 0 | `test_march_breach_stops_as_run_march_stops` 4, `test_march_equals_run_march_after_every_step` 6 | yes |
| `row_deposition_slots_reversed` | 0 | `test_march_breach_stops_as_run_march_stops` 4, `test_march_equals_run_march_after_every_step` 6 | yes |
| `row_current_is_initial` | 2: `March.ABreachBeforeAnyStepReturnsNoRowsAndCountsNoStep`, `March.ABreachAfterStepsReturnsTheRowsBeforeItAndNotItsStep` | `test_march_breach_stops_as_run_march_stops` 4, `test_march_equals_run_march_after_every_step` 6 | yes |
| `recovery_before_the_maps_mean_stale` | 1: `March.StepsInBatchesEqualStepsOneAtATime` | `test_march_breach_stops_as_run_march_stops` 3, `test_march_equals_run_march_after_every_step` 4 | yes |
| `onset_strict` | 1: `Arrival.AnEventIsMappedFromItsOnsetAndATimeOnceSetIsKept` | `test_arrival_maps_equal_the_trackers` 2, `test_march_equals_run_march_after_every_step` 2 | yes |
| `arrival_threshold_strict` | 1: `Arrival.ACellExactlyAtItsThresholdIsReachedAndOneBelowIsNot` | `test_arrival_maps_equal_the_trackers` 2 | yes |
| `every_event_writes_the_first_map` | 1: `Arrival.AnEventIsMappedFromItsOnsetAndATimeOnceSetIsKept` | `test_arrival_maps_equal_the_trackers` 2 | yes |
| `started_strict` | 1: `Arrival.AnEventIsMappedFromItsOnsetAndATimeOnceSetIsKept` | `test_march_equals_run_march_after_every_step` 2 | yes |
| `sensor_order_reversed` | 1: `Channels.ASensorsChannelIsTheChannelFieldReadAtItsCell` | `test_march_breach_stops_as_run_march_stops` 4, `test_march_equals_run_march_after_every_step` 6, `test_recovery_equals_the_tracker` 2 | yes |
| `binding_interval_first_zero` | 0 | `test_march_breach_stops_as_run_march_stops` 3, `test_march_equals_run_march_after_every_step` 4, `test_march_in_batches_equals_run_march` 4, `test_march_reads_numpy_numbers_and_a_nan_arrival_time` 1, `test_march_restored_mid_interval_continues_bit_for_bit` 2, `test_march_restored_recovery_continues_bit_for_bit` 2 | yes |
| `binding_channels_are_readings` | 0 | `test_march_breach_stops_as_run_march_stops` 4, `test_march_equals_run_march_after_every_step` 6 | yes |
| `refusal_spacing_shape` | 0 | `test_march2_refuses_what_it_is_built_from_when_it_cannot_read_it` 4 | yes |
| `refusal_spacing_dx_cell_length` | 0 | `test_march2_refuses_what_it_is_built_from_when_it_cannot_read_it` 1 | yes |
| `refusal_spacing_dy_cell_length` | 0 | `test_march2_refuses_what_it_is_built_from_when_it_cannot_read_it` 1 | yes |
| `refusal_spacing_dx_face_length` | 0 | `test_march2_refuses_what_it_is_built_from_when_it_cannot_read_it` 1 | yes |
| `refusal_spacing_dy_face_length` | 0 | `test_march2_refuses_what_it_is_built_from_when_it_cannot_read_it` 1 | yes |
| `refusal_spacing_aligned` | 0 | `test_march2_refuses_what_it_is_built_from_when_it_cannot_read_it` 4 | yes |
| `refusal_no_class` | 0 | `test_march2_refuses_what_it_is_built_from_when_it_cannot_read_it` 1 | yes |
| `refusal_physics_lengths_differ` | 0 | `test_march2_refuses_what_it_is_built_from_when_it_cannot_read_it` 2 | yes |
| `refusal_sensor_row_outside` | 0 | `test_march2_refuses_what_it_is_built_from_when_it_cannot_read_it` 1, `test_march_refuses_what_it_cannot_read` 1 | yes |
| `refusal_sensor_column_outside` | 0 | `test_march2_refuses_what_it_is_built_from_when_it_cannot_read_it` 1 | yes |
| `refusal_recovery_state_without_recovery` | 0 | `test_march_restore_refuses_a_recovery_state_it_has_no_recovery_for` 1 | yes |
| `refusal_recovery_state_missing` | 0 | the module crashed | yes |
| `refusal_arrival_thresholds` | 0 | `test_blocks_refuse_what_they_cannot_read` 1 | yes |
| `refusal_recovery_thresholds` | 0 | `test_blocks_refuse_what_they_cannot_read` 1 | yes |
| `refusal_cube_classes` | 0 | `test_blocks_refuse_a_cube_of_another_shape` 3, `test_blocks_refuse_what_they_cannot_read` 3, `test_march_restore_refuses_a_cube_of_another_shape` 1 | yes |
| `refusal_cube_rows` | 0 | `test_blocks_refuse_a_cube_of_another_shape` 3, `test_march_restore_refuses_a_cube_of_another_shape` 1 | yes |
| `refusal_cube_columns` | 0 | `test_blocks_refuse_a_cube_of_another_shape` 3, `test_march_refuses_what_it_cannot_read` 1, `test_march_restore_refuses_a_cube_of_another_shape` 1 | yes |
| `refusal_cube_aligned` | 0 | `test_blocks_refuse_what_they_cannot_read` 3, `test_march_restore_refuses_an_unaligned_array` 1 | yes |
| `refusal_channel_fields_cube` | 0 | `test_blocks_refuse_a_cube_of_another_shape` 3, `test_blocks_refuse_what_they_cannot_read` 2 | yes |
| `refusal_arrival_update_cube` | 0 | `test_blocks_refuse_a_cube_of_another_shape` 3, `test_blocks_refuse_what_they_cannot_read` 2 | yes |
| `refusal_recovery_update_cube` | 0 | `test_blocks_refuse_a_cube_of_another_shape` 3, `test_blocks_refuse_what_they_cannot_read` 2 | yes |
| `refusal_march_restore_cube` | 0 | `test_march_refuses_what_it_cannot_read` 1, `test_march_restore_refuses_a_cube_of_another_shape` 3, `test_march_restore_refuses_an_unaligned_array` 1 | yes |
| `refusal_recovery_update_means` | 0 | `test_blocks_refuse_what_they_cannot_read` 1 | yes |
| `refusal_sensor_channels_shape` | 0 | `test_blocks_refuse_what_they_cannot_read` 1 | yes |
| `refusal_sensor_channels_aligned` | 0 | `test_blocks_refuse_what_they_cannot_read` 1 | yes |
| `refusal_maps_shape` | 0 | `test_arrival_maps_refuse_maps_of_another_shape` 4, `test_blocks_refuse_what_they_cannot_read` 1, `test_march_refuses_what_it_cannot_read` 1, `test_march_restore_refuses_maps_of_another_shape` 4 | yes |
| `refusal_maps_aligned` | 0 | `test_blocks_refuse_what_they_cannot_read` 1, `test_march_restore_refuses_an_unaligned_array` 1 | yes |
| `refusal_arrival_restore_maps` | 0 | `test_arrival_maps_refuse_maps_of_another_shape` 4, `test_blocks_refuse_what_they_cannot_read` 2 | yes |
| `refusal_march_restore_maps` | 0 | `test_march_refuses_what_it_cannot_read` 1, `test_march_restore_refuses_an_unaligned_array` 1, `test_march_restore_refuses_maps_of_another_shape` 4 | yes |
| `refusal_recovery_array_length` | 0 | `test_march_refuses_what_it_cannot_read` 1, `test_march_restore_refuses_an_unaligned_or_short_recovery_array` 4 | yes |
| `refusal_recovery_array_aligned` | 0 | `test_march_restore_refuses_an_unaligned_or_short_recovery_array` 4 | yes |
| `refusal_maps_events` | 0 | `test_arrival_maps_refuse_maps_of_another_shape` 1, `test_march_restore_refuses_maps_of_another_shape` 1 | yes |
| `refusal_maps_channels` | 0 | `test_arrival_maps_refuse_maps_of_another_shape` 1, `test_blocks_refuse_what_they_cannot_read` 1, `test_march_restore_refuses_maps_of_another_shape` 1 | yes |
| `refusal_maps_rows` | 0 | `test_arrival_maps_refuse_maps_of_another_shape` 1, `test_march_refuses_what_it_cannot_read` 1, `test_march_restore_refuses_maps_of_another_shape` 1 | yes |
| `refusal_maps_columns` | 0 | `test_arrival_maps_refuse_maps_of_another_shape` 1, `test_march_restore_refuses_maps_of_another_shape` 1 | yes |
| `guard_bool_number` | 0 | `test_march_refuses_a_bool_a_float_count_and_a_time_that_is_not_finite` 4, `test_march_refuses_what_it_cannot_read` 1 | yes |
| `guard_bool_count` | 0 | `test_march_refuses_a_bool_a_float_count_and_a_time_that_is_not_finite` 3, `test_march_refuses_a_bool_or_a_float_for_the_steps_asked` 1 | yes |
| `guard_float_count` | 0 | `test_march_refuses_a_bool_a_float_count_and_a_time_that_is_not_finite` 3, `test_march_refuses_a_bool_or_a_float_for_the_steps_asked` 1 | yes |
| `guard_count_overflow` | 0 | `test_march_refuses_a_bool_a_float_count_and_a_time_that_is_not_finite` 1, `test_march_refuses_a_bool_or_a_float_for_the_steps_asked` 1 | yes |
| `guard_time_finite` | 0 | `test_march_refuses_a_bool_a_float_count_and_a_time_that_is_not_finite` 7 | yes |
| `guard_time_nan_always_ok` | 0 | `test_march_refuses_a_bool_a_float_count_and_a_time_that_is_not_finite` 3 | yes |
| `guard_arrival_time_nan_refused` | 0 | `test_march_binding_imports_nothing_from_src` 1, `test_march_reads_numpy_numbers_and_a_nan_arrival_time` 1, `test_march_refuses_a_bool_a_float_count_and_a_time_that_is_not_finite` 20, `test_march_refuses_a_bool_or_a_float_for_the_steps_asked` 1, `test_march_refuses_what_it_cannot_read` 1, `test_march_restore_refuses_a_recovery_state_it_has_no_recovery_for` 1, `test_march_restore_refuses_an_unaligned_or_short_recovery_array` 4 | yes |
| `guard_restore_step_negative` | 0 | `test_march_refuses_a_bool_a_float_count_and_a_time_that_is_not_finite` 1 | yes |
| `guard_interval_first_range` | 0 | `test_march_refuses_a_bool_a_float_count_and_a_time_that_is_not_finite` 1, `test_march_refuses_what_it_cannot_read` 1 | yes |
| `guard_interval_steps_positive` | 0 | none | no: an equivalent mutant |
| `guard_march_n_positive` | 0 | `test_march_refuses_what_it_cannot_read` 1 | yes |
| `march_recovery_channel_readings_are_readings` | 0 | `test_march_equals_run_march_after_every_step` 1, `test_march_in_batches_equals_run_march` 1, `test_march_restored_recovery_continues_bit_for_bit` 2 | yes |
| `march_recovery_readings_are_channel_readings` | 1: `March.ARunRestoredFromAnothersStateContinuesItBitForBit` | `test_march_equals_run_march_after_every_step` 1 | yes |
| `breach_drops_the_rows_before_it` | 1: `March.ABreachAfterStepsReturnsTheRowsBeforeItAndNotItsStep` | none | yes |

104 of 105 plants fail at least one test; `guard_interval_steps_positive` is an equivalent mutant, as the clause `steps < 1` is implied by `0 <= first < steps`, which the same check tests, so a plan it would refuse is refused by the others. A plant that makes the module crash instead of fail is counted as caught. The controls (14 runs, at the start and the end of each stream) pass every test: yes.
<!-- tables91 plantsround1 end -->

## 6. The timing

<!-- tables91 timing begin -->
### T9. The 320x120 dust release's march over 200 steps across the release: the reference's over the core-driven

| Session | Core-driven march | Steps | Calls to the core | Reference over core-driven | Ends equal the reference's, bit for bit |
|---|---|---|---|---|---|
| 1 | core, monitor on the fields | 200 | 200 | 7.24 | yes |
| 1 | core, monitor on the readings | 200 | 3 | 8.37 | yes |
| 2 | core, monitor on the fields | 200 | 200 | 7.73 | yes |
| 2 | core, monitor on the readings | 200 | 3 | 8.13 | yes |
| 3 | core, monitor on the fields | 200 | 200 | 7.65 | yes |
| 3 | core, monitor on the readings | 200 | 3 | 8.81 | yes |

Each march from the Linux reference's frame at step 2,700 (about 9.8 s) of the 320x120 dust release through the onset at 10 s, with fresh trackers and monitor, frames written as the run writes them: the reference's `_Run.march`, then the core-driven march with the monitor reading the fields every step, then with the monitor reading the core's readings, each once, in one process. Only ratios within a session are given; the times stay in `results/builder91/timing_N.json`.

| Session | Started | Calibration, after over before | Windows before | Windows after |
|---|---|---|---|---|
| 1 | 2026-11-22 18:27:34 | 1.00 | msmpeng 38; system 25; code 17; on mains True; processor performance 83 | msmpeng 43; svchost 24; system 22; on mains True; processor performance 90 |
| 2 | 2026-11-22 18:28:15 | 1.12 | system 44; msedge 40; msmpeng 30; on mains True; processor performance 90 | msmpeng 35; msedge 32; msedge 30; on mains True; processor performance 95 |
| 3 | 2026-11-22 18:28:49 | 1.04 | msmpeng 50; msedge 47; system 36; on mains True; processor performance 94 | msmpeng 38; code 22; wmiprvse 17; on mains True; processor performance 96 |
<!-- tables91 timing end -->

## 7. The vectorisation report

<!-- tables91 loops begin -->
### T10. The loops GCC vectorises in the new modules

`g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`, `-std=c++20 -O2 -ffp-contract=off -march=x86-64-v3 -DNDEBUG`, `-fopt-info-vec-optimized`: loops vectorised, each counted once by its place in the source.

| Source | Loops vectorised | Report lines | Where (line: source) |
|---|---|---|---|
| `src/channels.cpp` | 1 | 2 | `27: for (std::size_t k = 0; k < N; ++k) {` |
| `src/arrival.cpp` | 1 | 2 | `28: for (std::size_t k = 0; k < N; ++k) {` |
| `src/recovery.cpp` | 1 | 2 | `36: for (std::size_t k = 0; k < N; ++k) {` |
| `src/march.cpp` | 0 | 0 |  |
<!-- tables91 loops end -->

## Findings

- **F1, the whole march equals the reference bit for bit.** This is the first comparison of
  whole runs and not only of blocks. Over six scenario runs, 33,119 steps of the 320x120 dust
  release among them, no frame, budget term, sensor row, arrival map, recovery field or decision
  differs from the Linux reference's (T2, T3). So nothing the run carries from step to step
  differs either: the resident fields, the budgets, the maps, the recovery fields, the step count
  and the times. The prompt's stop, a whole-march difference, was not met.
- **F2, the first plant run found two rules no test could see.** `last_time_not_t_end` (an
  interval's last step at `t_start + steps dt` instead of `t_end`) passed CTest and every parity
  case. Every plan the tests marched ended where the two agree: the GoogleTests use powers of
  two, and the parity marches cut each plan to a few steps, so their ends are `t_start + k dt` by
  construction. Only the whole 80x30 dust release saw it, in its arrival maps. Of the real runs'
  intervals (their summaries), the two differ on the dust release's last (40 s to 120 s in 4,540
  steps) and on the clean-out's one, which the held clean-out does not reach; of the planted
  plans, only on the door leak's last (15 s to 60 s in 597 steps on the planted flows).
  `restore_drops_recovery` passed CTest because the GoogleTest's recovery state at
  the restore was, one step later, what a fresh state becomes: every cell above its threshold
  and no 100:1 time yet. Commit 03030b9 adds a march case on the door leak's last interval to its
  end, a GoogleTest on a plan whose end is not `t_start + steps dt`, and targets that some
  sensors reach early; the second plant run (T8) catches all sixteen. The first run is kept in
  `results/builder91/round0/`. What 03030b9 pinned of the restore was the sensors' 100:1 times
  and a restore that drops all of the recovery state; the room's within-spec start, the
  classes' starts and the means' times, restored one array at a time, were not (test 91 B3),
  and are now: by a clean-out restored before and inside its within-spec stretch, by a
  GoogleTest on a restore inside it, and by whole restarts (F5, T4b).
- **F3, the gate.** Against the Windows records the three 80x30 core-driven runs make every
  decision the Windows runs make and every field is inside the band's measured spread, the
  budget closing far inside 1e-12 of the supply (T5, T6). These are step 1's readings of the
  Linux reference, as they must be: this pass's Linux references reproduce step 1's runs bit for
  bit (T1), and the core-driven runs equal them.
- **F4, the timing.** Prediction (d) holds in every session, and so does the march with the
  monitor reading the fields at every step, at a lower ratio in each session (T9): the price of
  copying the five class fields out of the core and handing them to the monitor every step,
  which step 6's update from readings removes. Windows' busiest processes during the sessions
  were Defender's scanner and Edge, where the prompt asked for nothing heavy running (T9's second
  table); a ratio is taken within one session, so a load falls on both its sides.
- **F5, round 1: three rules of the port nothing tested.** Review 91 B1 and test 91 B1 to B3 found
  no defect in the port. They found that the march's stop on a breach and on a refused step, the
  within-spec rule (every sensor, on the channel's reading) and the restore of the room's start,
  the classes' starts and the means' times each passed every test with a plant in it. The first
  plant run's lesson (F2) applied again: a plant in a rule is caught by a test only if its inputs
  separate the rule from the plant, and the cases planted for the other rules left these three
  with equal answers: the six whole runs never breach and no whole run restarted a clean-out,
  the within-spec cases moved every sensor and every class together, and the one mid-march
  restore of a clean-out held a room that is never within spec, so its start, its classes' starts
  and its means' times were NaN on both sides of the restore. The tests of section 4 separate
  them, T8b shows each plant failing one, and T4b restarts the clean-out whole. Three rules were
  caught by GoogleTest alone and are now caught by a parity case too (test 91 S2): a mean's time
  set where its target is not defined, the room within spec when any channel is, and a restore
  handing a zero room start. Round 1's own first plant run (T8b) left two plants uncaught, both
  in the binding's refusals: a cube whose row count is another's, which no test handed any entry
  point, and the binding's check of an interval's first step, which the core's own check refuses
  in nearly the same words under another prefix. Each is pinned now, dimension by dimension and
  by the binding's own message. A plant that removes the check that stops `March2.restore`
  reading a missing recovery state crashes the module instead of raising, and T8b counts a
  crash as caught.

## Deviations and limits

- **Five commits, not four.** The predictions; the port with its tests (e152bbd); a `test:`
  commit after the first plant run (03030b9, F2), kept apart because the runs of record were
  made at e152bbd and the commit changes no code under `core/src/`, `core/include/` or the
  binding; the probe; the report and records.
- **Round 1 (prompt 91b) is four commits, not three.** A `test:` commit for the tests of the
  breach, the within-spec rule, the restore and the binding's refusals (506e800); a `fix:` commit
  for the binding's guards, the driver's loop and the header comments, with their tests
  (b028332); a `refactor:` commit for the shared solver setup, which changes no behaviour and so
  is not in the `fix:` commit (f20ae23); and the records. Each of the first three builds under
  `-Werror` and passes CTest and the march's parity module on its own, built apart from the
  working tree.
- **The interface departs from ADR-017 B's table in three places**, all because the flow solve
  is still Python's (section 1): the faces and the eddy viscosity cross with each interval; every
  march starts with a `restore` from the reference's state, at t = 0 or at a checkpoint; a breach
  is returned with the steps before it and raised by the Python side. Steps 4 and 5 bring the
  flow state into the run and change the first. SYSTEM.md's core/ row records the three and
  its history row of 2026-11-23 says so; ADR-017 B is not amended.
- **The driver patches `time_integration._Run` for the call.** `core_driver.run` replaces the
  module's `_Run` with `CoreRun` while `time_integration.run` runs, so the orchestration is the
  reference's own code and nothing under `src/` changes. It is test infrastructure, in `tests/`;
  step 6 puts the call behind `solver.backend` in `src/time_integration.py`.
- **The monitor still reads the fields at every step** in the runs of record (T1, T2), as the
  reference hands them, so the core-driven march goes to the core one step at a time and copies
  the fields out after each. The march with the monitor on the core's readings (T9's second
  core-driven march) hands the monitor fields that hold the readings at the sensors' cells and
  zero elsewhere: a stand-in, bit for bit in its alerts since the monitor reads nothing else,
  until step 6 gives `AlertMonitor` an update from readings.
- **The scenario runs.** The clean-out's flow state is the dust release's, read from step 1's
  file: its flow inputs are the dust release's (`flow_inputs`), so the fingerprint is the same
  and the state is copied, not solved. The hold is one air change, 5.562 s, the value
  `configs/scenario_clean_out.yaml` gives for 80x30; the held run stops at step 1,976 on both
  sides (T3). The restart's runs write a checkpoint every 1,000 steps, the dust release's file
  none; the two equal each other bit for bit (T4).
- **The probe was edited after the runs began**, in its `compare` command (`MassBudget.residual`
  called as the method it is, a pair skipped while either of its runs is missing, the pair of
  the checkpointed and uncheckpointed runs added) and its `timing` command (the 320x120 frame
  read once per session); its `configs`, `run` and `restart` commands are as they ran. Round 1
  edited it again: `plants` runs through `test_plant`, which `plants-round1` shares and which
  takes the scratch tree as a parameter so that streams can run side by side; `restart` takes a
  scenario, a stop step and a tag, `restart-clean` is new, `configs` also writes the clean-out's
  checkpointed configuration, and `compare` pairs the held clean-out's restarts and each remade
  core run with round 0's; `tables91.py` has T4b, T4c and T8b.
- **The runs of T2 and T3 are round 1's**, made again at the head because the binding changed
  under them, and equal to round 0's bit for bit (T4c); round 0's files are in
  `results/builder91/runs_round0/`. The reference runs are not remade: nothing under `src/`
  changed. Nothing the driver, a checkpoint or the tests hand the binding is refused by the
  guards (T2, T4, T4b).
- **The suites were run at round 1's head** (T7); `tests/test_core_march.py` is skipped on
  Windows as a module, where `_cfdcore` is not built.
- **Not taken up.** The vents room and the RNG and QUICK k and eps options are not among the
  whole runs (the prompt's scenarios do not use them); prompt 90's replay covers a vented class
  step, and step 6's end-to-end benchmark reruns the validation rows. The 160x60 runs of the band
  are not rerun here.
- **No `src/` or `configs/` change.**

## Files

- `core/include/cfdcore/channels.hpp`, `arrival.hpp`, `recovery.hpp`, `march.hpp` and their
  sources in `core/src/`: the modules. `core/include/cfdcore/transport.hpp`: `volumes()`.
- `core/python/bindings.cpp`: `March2`, `ArrivalMaps2`, `Recovery2`, `channel_fields`,
  `sensor_channels`, `budget_terms`; `set_interval`'s and `restore_budgets`' readers shared.
- `core/tests/test_march.cpp` (new), `core/CMakeLists.txt`.
- `tests/test_core_march.py` (new), `tests/core_mapping.py` (the march's mapping, and
  `march_arguments`, the arguments `March2` is built from by name),
  `tests/core_driver.py` (new), `.github/workflows/ci.yml` (the core job runs the new module).
- `docs/reports/probe91/`: `probe91.py` (configs, run, restart, compare, timing, plants,
  plant-runs, vecinfo), `variants91.py` (the planted defects) and `variants91b.py` (round 1's, with plants-round1 and
  restart-clean), `tables91.py` and `tables91.md`.
- Records in `results/builder91/` (untracked): the configurations, the run files with their
  summaries and frame digests in `runs/` (round 0's core runs in `runs_round0/`),
  `compare.json`, `plants.json` and `plants_round1_s*.json` (and the first plant run in
  `round0/`, round 0's comparison and plants in `round1_before/`), `timing_1.json` to
  `timing_3.json`, `vecinfo.json`, `parity.xml`,
  `ctest.xml`, the two suites' JUnit files and the logs in `logs/`; the wrappers that run them
  under WSL2 (`probe.sh`, `bg.sh`, `pytest.sh`, `install.sh`), and round 1's in
  `results/builder91b/` (the scripts that build, run the plant streams and make the records,
  and the exploratory scripts that placed the breaches and the decay case).
