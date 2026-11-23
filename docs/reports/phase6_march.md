# Phase 6 Step 3b: The March in the Core (prompt 91)

**Date:** 2026-11-22
**Prompt:** 91 (`docs/prompts/cc-prompt-91-march.md`)
**Branch:** `phase6/march`
**Status:** built. Predictions (a), (b), (c) and (d) hold (T0). Six whole scenario runs stepped
through the core, the 320x120 dust release among them, equal the Linux reference bit for bit at
every output step and in every decision, and the restart continues the uninterrupted run bit for
bit; against the Windows records every decision is equal and every field inside the band's
first tier, which meets step 3's gate. The prompt's stop, a whole-march difference, was not met.
The first plant run found two rules no test could see; both are tested now (F2).

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
| 80x30 dust release | 6811 | 70 | 3 | kept | no | 87 | 8 |
| 80x30 door leak | 3416 | 36 | 3 | kept, kept | no | 42 | 4 |
| 80x30 filter breach | 6811 | 70 | 3 | kept | no | 89 | 8 |
| 80x30 clean-out, to t_end | 13620 | 138 | 1 | kept | no | 162 | 12 |
| 80x30 clean-out, with a hold | 1976 | 21 | 1 | kept | yes | 24 | 2 |
| 320x120 dust release | 33119 | 333 | 3 | kept | no | 1344 | 355 |

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
| `test_march_equals_run_march_after_every_step` | 5 | 5 | 0 | 0 |
| `test_march_in_batches_equals_run_march` | 5 | 5 | 0 | 0 |
| `test_march_restored_mid_interval_continues_bit_for_bit` | 2 | 2 | 0 | 0 |
| `test_march_takes_the_steps_asked_for_and_stops_at_the_hold` | 1 | 1 | 0 | 0 |
| `test_march_refuses_what_it_cannot_read` | 1 | 1 | 0 | 0 |
| `test_march_binding_imports_nothing_from_src` | 1 | 1 | 0 | 0 |
| the parity modules (`test_core_parity.py` and `test_core_march.py`) | 3400 | 3383 | 0 | 17 |
| CTest | 115 | 115 | 0 | 0 |

New GoogleTest cases (16): Channels.AChannelIsZeroThenEachMemberAddedInTurn, Channels.ASensorsChannelIsTheChannelFieldReadAtItsCell, Channels.ASetThatIsNotOneChannelPerClassThrows, Arrival.ACellExactlyAtItsThresholdIsReachedAndOneBelowIsNot, Arrival.AnEventIsMappedFromItsOnsetAndATimeOnceSetIsKept, Arrival.ARestoreCopiesTheMapsAndTheTimeAndRefusesAnotherLength, Recovery.ACellOrASensorExactlyAtItsThresholdHasNotRecovered, Recovery.TheHoldHoldsAtEqualityAndTheTargetsWhereDefined, Recovery.ProgrammingErrorsThrow, March.AMarchTakesTheStepsAskedForAndStopsAtTheIntervalsEnd, March.TheIntervalsLastStepIsAtItsEndExactly, March.StepsInBatchesEqualStepsOneAtATime, March.ARunRestoredFromAnothersStateContinuesItBitForBit, March.TheHoldStopsTheMarchAtTheStepItHolds, March.TheMeansAreTheContentOverTheFreeArea, March.ProgrammingErrorsThrow.
- The whole suite under Linux: 6484 passed, 0 failed, 17 skipped in 1399 s.
- The whole suite under Windows: 3101 passed, 0 failed, 2 skipped in 1378 s.
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
after it. Each asserts that its planted inputs reach the rule they are planted for.

The march against `_Run.march` on the 80x30 product mesh with planted flows (faces of both signs
about 0.2 m/s, a positive eddy viscosity), in four cases: across the dust release's onset; across
the door leak's two flow switches; the clean-out from fields around its thresholds; the clean-out
within spec from the start, held at its fifth step by a hold equal to that step's time. Both are
driven the same way from the same start (`_Run` and `CoreRun` on the same configuration, plans
cut to a few steps of each interval, the trackers updated at the start as `run` does at t = 0),
and after every step the fields, the budget terms, the maps, the recovery fields, the readings,
the channels and the means are compared bit for bit; at the end the fields, budgets, step count,
time and the hold. Each case again with the march in batches and the monitor on the readings. A
second core run restored from the first's state mid-interval continues it bit for bit (the
clean-out, with its recovery fields, and the dust release); `march(3)` takes three steps; the
held clean-out stops at the reference's step and a later march takes none; the binding's own
refusals; and a restore, a march and every read with every `src` module unimportable.

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
| `control` | no edit: every test passes | 100% tests passed, 0 tests failed out of 115 | none | dust80 equal, hold80 equal |
| `channel_order_reversed` | a channel's field summed over its members last to first | 98% tests passed, 2 tests failed out of 115: `Channels.AChannelIsZeroThenEachMemberAddedInTurn`, `Channels.ASensorsChannelIsTheChannelFieldReadAtItsCell` | `test_channel_fields_equal_channel_sums` 4, `test_sensor_channels_equal_the_monitors_and_the_fields_at_the_sensors` 1 | dust80 equal, hold80 equal |
| `threshold_cell_recovered` | a cell exactly at its threshold read as recovered: the local field set only above it | 99% tests passed, 1 tests failed out of 115: `Recovery.ACellOrASensorExactlyAtItsThresholdHasNotRecovered` | `test_recovery_equals_the_tracker` 2 | dust80 equal, hold80 equal |
| `arrival_before_onset` | every event's map updated at every step, before its onset too | 98% tests passed, 2 tests failed out of 115: `Arrival.AnEventIsMappedFromItsOnsetAndATimeOnceSetIsKept`, `March.StepsInBatchesEqualStepsOneAtATime` | `test_arrival_maps_equal_the_trackers` 2 | dust80 equal, hold80 equal |
| `arrival_overwritten` | a cell's arrival time overwritten by every later step at or above the threshold | 99% tests passed, 1 tests failed out of 115: `Arrival.AnEventIsMappedFromItsOnsetAndATimeOnceSetIsKept` | `test_arrival_maps_equal_the_trackers` 2, `test_march_equals_run_march_after_every_step` 4 | dust80 differs: 0/70 frames, arrival, hold80 differs: 0/21 frames, arrival |
| `hold_strict` | the hold compared strictly: the stretch must exceed the hold time | 98% tests passed, 2 tests failed out of 115: `Recovery.TheHoldHoldsAtEqualityAndTheTargetsWhereDefined`, `March.TheHoldStopsTheMarchAtTheStepItHolds` | `test_march_equals_run_march_after_every_step` 1, `test_march_in_batches_equals_run_march` 1, `test_march_takes_the_steps_asked_for_and_stops_at_the_hold` 1, `test_recovery_equals_the_tracker` 2 | dust80 equal, hold80 equal |
| `free_area_whole` | the free-area mean over the whole domain's area, SOLID cells included | 99% tests passed, 1 tests failed out of 115: `March.TheMeansAreTheContentOverTheFreeArea` | `test_march_equals_run_march_after_every_step` 5 | dust80 equal, hold80 equal |
| `restore_drops_recovery` | a restore that sets the fields, budgets and maps and drops the recovery fields | 99% tests passed, 1 tests failed out of 115: `March.ARunRestoredFromAnothersStateContinuesItBitForBit` | `test_march_equals_run_march_after_every_step` 2, `test_march_in_batches_equals_run_march` 1, `test_march_restored_mid_interval_continues_bit_for_bit` 1, `test_march_takes_the_steps_asked_for_and_stops_at_the_hold` 1 | dust80 equal, hold80 equal |
| `march_one_too_many` | march(n) steps n + 1 steps, its rows sized for them | 97% tests passed, 3 tests failed out of 115: `March.AMarchTakesTheStepsAskedForAndStopsAtTheIntervalsEnd`, `March.StepsInBatchesEqualStepsOneAtATime`, `March.ARunRestoredFromAnothersStateContinuesItBitForBit` | `test_march_equals_run_march_after_every_step` 5, `test_march_restored_mid_interval_continues_bit_for_bit` 2, `test_march_takes_the_steps_asked_for_and_stops_at_the_hold` 1 | dust80 raised AssertionError, hold80 raised AssertionError |
| `sensor_channel_order_reversed` | a channel's reading at a sensor summed over its members last to first | 99% tests passed, 1 tests failed out of 115: `Channels.ASensorsChannelIsTheChannelFieldReadAtItsCell` | `test_march_equals_run_march_after_every_step` 5, `test_sensor_channels_equal_the_monitors_and_the_fields_at_the_sensors` 1 | dust80 raised AssertionError, hold80 raised AssertionError |
| `channel_not_from_zero` | a channel started from its first member's value, not from zero (-0.0 kept) | 99% tests passed, 1 tests failed out of 115: `Channels.AChannelIsZeroThenEachMemberAddedInTurn` | `test_channel_fields_equal_channel_sums` 4 | dust80 equal, hold80 equal |
| `threshold_sensor_within` | a sensor exactly at its threshold read as within spec | 98% tests passed, 2 tests failed out of 115: `Recovery.ACellOrASensorExactlyAtItsThresholdHasNotRecovered`, `Recovery.TheHoldHoldsAtEqualityAndTheTargetsWhereDefined` | `test_recovery_equals_the_tracker` 2 | dust80 equal, hold80 equal |
| `arrival_in_solid` | the arrival maps set in SOLID cells too | 99% tests passed, 1 tests failed out of 115: `Arrival.AnEventIsMappedFromItsOnsetAndATimeOnceSetIsKept` | `test_arrival_maps_equal_the_trackers` 2 | dust80 equal, hold80 equal |
| `room_since_kept` | the room's within-spec start kept when a sensor rises above its threshold again | 99% tests passed, 1 tests failed out of 115: `Recovery.TheHoldHoldsAtEqualityAndTheTargetsWhereDefined` | `test_recovery_equals_the_tracker` 2 | dust80 equal, hold80 equal |
| `sensor_time_overwritten` | a sensor's 100:1 time overwritten by every later reading at or below the target | 99% tests passed, 1 tests failed out of 115: `Recovery.TheHoldHoldsAtEqualityAndTheTargetsWhereDefined` | `test_march_equals_run_march_after_every_step` 2, `test_recovery_equals_the_tracker` 2 | dust80 equal, hold80 differs: 0/21 frames, recovery |
| `held_not_checked_at_entry` | a march that starts on a held run steps on | 99% tests passed, 1 tests failed out of 115: `March.TheHoldStopsTheMarchAtTheStepItHolds` | `test_march_takes_the_steps_asked_for_and_stops_at_the_hold` 1 | dust80 equal, hold80 equal |
| `last_time_not_t_end` | the interval's last step at t_start + steps dt, not at t_end exactly | 99% tests passed, 1 tests failed out of 115: `March.TheIntervalsLastStepIsAtItsEndExactly` | `test_march_equals_run_march_after_every_step` 1, `test_march_in_batches_equals_run_march` 1 | dust80 differs: 0/70 frames, arrival, hold80 equal |

16 of 16 plants fail at least one test. In the last column a count of frames is of the frames whose class fields or budget terms differ, then what else differs (the sensor table, `/arrival/`, `/recovery/`); `raised` is a run the driver stopped, its check of the core's readings against the monitor's, or of a march's rows against the frames, failing.
<!-- tables91 plants end -->

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
  `results/builder91/round0/`.
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

## Deviations and limits

- **Five commits, not four.** The predictions; the port with its tests (e152bbd); a `test:`
  commit after the first plant run (03030b9, F2), kept apart because the runs of record were
  made at e152bbd and the commit changes no code under `core/src/`, `core/include/` or the
  binding; the probe; the report and records.
- **The interface departs from ADR-017 B's table in three places**, all because the flow solve
  is still Python's (section 1): the faces and the eddy viscosity cross with each interval; every
  march starts with a `restore` from the reference's state, at t = 0 or at a checkpoint; a breach
  is returned with the steps before it and raised by the Python side. Steps 4 and 5 bring the
  flow state into the run and change the first.
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
  read once per session); its `configs`, `run` and `restart` commands are as they ran.
- **The Windows suite was run at e152bbd** (T7): the one file changed since that it would read,
  `tests/test_core_march.py`, is skipped on Windows as a module, where `_cfdcore` is not built.
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
- `tests/test_core_march.py` (new), `tests/core_mapping.py` (the march's mapping),
  `tests/core_driver.py` (new), `.github/workflows/ci.yml` (the core job runs the new module).
- `docs/reports/probe91/`: `probe91.py` (configs, run, restart, compare, timing, plants,
  plant-runs, vecinfo), `variants91.py` (the planted defects), `tables91.py` and `tables91.md`.
- Records in `results/builder91/` (untracked): the configurations, the run files with their
  summaries and frame digests in `runs/`, `compare.json`, `plants.json` (and the first plant
  run in `round0/`), `timing_1.json` to `timing_3.json`, `vecinfo.json`, `parity.xml`,
  `ctest.xml`, the two suites' JUnit files and the logs in `logs/`; the wrappers that run them
  under WSL2 (`probe.sh`, `bg.sh`, `pytest.sh`, `install.sh`).
