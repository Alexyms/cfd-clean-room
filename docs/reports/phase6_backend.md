# Phase 6 Step 6a: The Backend Switch and the 80x30 Runs End to End (prompt 96)

**Date:** 2026-11-26
**Prompt:** 96 (`docs/prompts/cc-prompt-96-backend.md`)
**Branch:** `phase6/backend`
**Status:** predictions committed before any run; the build, the runs and the scoring follow.

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
