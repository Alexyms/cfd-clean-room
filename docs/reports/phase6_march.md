# Phase 6 Step 3b: The March in the Core (prompt 91)

**Date:** 2026-11-22
**Prompt:** 91 (`docs/prompts/cc-prompt-91-march.md`)
**Branch:** `phase6/march`
**Status:** predictions committed, before any run.

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
