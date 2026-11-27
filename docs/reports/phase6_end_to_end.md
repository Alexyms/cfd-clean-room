# Phase 6 Step 6b: The Band on 320x120 and the Product End to End (prompt 97)

**Date:** 2026-11-26
**Prompt:** 97 (`docs/prompts/cc-prompt-97-end-to-end.md`)
**Branch:** `phase6/end-to-end`
**Status:** predictions committed; no run made yet.

This is the second half of Phase 6 step 6 (`docs/PROJECT_PLAN.md`, Phase 6, the steps table;
ADR-017 F and H): the noise band measured on the product grid, REQ-N03 level 2 on the five
product scenarios on 320x120 and the probe's two 160x60 runs with the placement, the validation
rows of record rerun on the core, and the wall time on the three grids.

## The predictions

Set by the orchestrator in prompt 97, and committed alone before any run.

- (a) Every 320x120 measured spread is inside its limit (flow 1e-8, class fields and sensors
  1e-6, the leftover 1e-8 of the supply), and the flow fields' spread is below the probe's
  7.8e-10.
- (b) On the five 320x120 scenarios and the two 160x60 runs under `core`, every decision equals
  the reference's, every flow state stops at the reference's outer count, not one apart, and
  every field is inside its measured spread (ADR-017 H's first tier).
- (c) The placement on 320x120 under `core` gives the reference's searched cells under every
  rule and pool, and the same order of the layouts.
- (d) Every steady validation row rerun on the core keeps its stored verdict.
- (e) The 320x120 dust release runs end to end at least eight times faster under `core`.

## How they are scored (the builder, before any run)

Every run is made under WSL2 with the Linux environment (`~/.venvs/cfd_clean_room`), one BLAS
thread, `NPY_DISABLE_CPU_FEATURES=X86_V4` set for every run, the reference's included, through
`scripts/run_scenario.py` (or `scripts/benchmark.py` for a harness row), at most three solver
processes at once and at most two of them on 320x120. A run file is written on the Linux file
system and copied into `results/builder97/` when it ends. Every comparison is of two runs made on
this machine by this branch's code. The comparator is `scripts/compare_runs.py` as merged with
PR 127; its measures are the probe's (`docs/reports/probe85/tables85.py`) with the changes
`docs/reports/phase6_backend.md` section 1 lists.

The 320x120 scenarios are main's `configs/scenario_*.yaml` as they are (the base file is the
320x120 product room, its `mass_imbalance_tol` the file's rule for that grid). The 160x60 runs
are the probe's: main's dust release and filter breach on 160x60 with `mass_imbalance_tol` the
base file's rule for that grid (`probe85.mass_imbalance_tol`). A run that takes its flow states
from another run's file (`--flows-from`) is named in the report's table of runs, with the file.

- **(a).** The reference on 320x120 is run on the dust release and the door leak as it is and
  under each of the probe's two stop variants (`probe85.VARIANTS`): `diffusion_tol` times 0.99,
  and `outer_one_later` (each flow solve's first stop deferred by one outer iteration, the
  probe's patch of `ErrorEstimateRule.update` in the run's own process). Each variant's run is
  compared with the reference's by the comparator. The spread of a field is the largest
  difference the comparator reports for it over the four variant runs (two scenarios, two
  variants), as P8 took the probe's: the flow fields as one spread, the largest over u, v, p, k,
  eps and nu_t and every flow state; the concentration over every stored frame; the
  concentration pointwise; the sensor channels; the change of the accumulated leftover over the
  supply. Holds when each spread is at or below its limit (flow 1e-8, the concentration, its
  pointwise reading and the sensor channels 1e-6, the leftover 1e-8) and the flow spread is
  below 7.8e-10. The variants' decisions are reported beside, as the probe reported them.
- **(b).** Each of the seven scenarios (on 320x120 the clean-out, the door leak, the dust
  release, the filter breach and the seal leak; on 160x60 the dust release and the filter
  breach) is run under `reference` and under `core`, and each pair compared by the comparator,
  the 320x120 pairs with each field's spread set to the 320x120 spread of (a) and the 160x60
  pairs with the comparator's own spreads (the probe's, P8 and P2), the limits unchanged in
  both. Holds when, on all seven pairs, every decision item moves by zero, every flow state's
  stop reason is the reference's and its outer count equal to the reference's, the budget
  closes to 1e-12 of the supply on every stored frame, and every field item reads "inside the
  spread". A flow state the reference stops at the cap counts as a stop at the cap: it holds
  when the core's stops there too, with its fields inside the spread.
- **(c).** `compare_runs.compare_placement` on the four contaminating scenarios on 320x120 (the
  door leak, the dust release, the filter breach and the seal leak), the reference's four run
  files against the core's. Holds when every item naming a layout's cells (the file's four and
  each searched pool, under each rule) and each rule's order of the layouts passes. The units'
  latencies are reported beside.
- **(d).** Every harness row of record whose solve is steady, that is the latest stored row of
  each case and method made on the reference (`benchmark.row_backend`) in
  `benchmarks/results.jsonl` that is not a tier-2 run (`val022_*`), run
  once through `scripts/benchmark.py --backend core` with the same case and method. Holds when,
  for every row, the core row's `passed` equals the stored row's and its stop reason equals the
  stored row's. A row whose `passed` is None (the laminar rows, scored in the suite, and the
  report-only rows) keeps its verdict when the core's is None too and the stop reason is the
  same; a row bounded at its cap (VAL-019's medium grid, VAL-020) keeps it when the core's row
  stops at the cap too. The outer count, the value and the wall time are reported beside.
- **(e).** The 320x120 dust release from rest, through `run_scenario.main` in one process per
  run, under each backend once, in one session with no other run of this step in progress, on
  mains power with the machine's state recorded (Windows' busiest processes and the power state,
  before and after). The flow solve's seconds, the march's and the rest are reported apart.
  Holds when the reference's wall time over the core's is at least 8.

A failed prediction is a finding, recorded and scored as such.
