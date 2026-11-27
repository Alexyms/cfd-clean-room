# Phase 6 Step 6b: The Band on 320x120 and the Product End to End (prompt 97)

**Date:** 2026-11-27
**Prompt:** 97 (`docs/prompts/cc-prompt-97-end-to-end.md`) and 97b
(`docs/prompts/cc-prompt-97b-sweep-counts.md`)
**Branch:** `phase6/end-to-end`
**Status:** done. (a) and (e) fail, (b), (c), (d), (f), (g) and (h) hold. The 320x120 limits
are proposed to Alex and not applied.

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

## Results

Every number is in `docs/reports/probe97/tables97.md` (T0 to T22), which
`docs/reports/probe97/tables97.py` generates from the records in `results/builder97/`; its
`--check` regenerates the tables and compares them, and both modes refuse an absent record. The
records were made at two commits: prompt 97's runner at 69744ab until its stop, and prompt 97b's
at 77e9b3d after it. The tables accept the two only because no file the records were computed by
(`src/`, `scripts/`, `configs/`, `core/`, `docs/reports/probe85/`, `probe97.py`) differs between
them, and they refuse sweep records of more than one commit. The head of the tables names both
commits and the files that do differ. Prompt 97 asked for a check that fails on records of two
commits; this reading of it is the builder's, and it is set out for Alex under "For Alex".

### The predictions scored (T0)

- **(a) FAILS.** Under `diffusion_tol` times 0.99 the 320x120 dust release moved the
  concentration pointwise and the leftover's change beyond their limits (T2). The flow spread is
  below the probe's. Prompt 97 stopped here, and prompt 97b measured the cause (below).
- **(b) HOLDS.** On all seven pairs every flow state stops at the reference's outer count for the
  reference's reason (T4), every decision is equal (T5), every field is inside its spread and
  every budget closes (T6).
- **(c) HOLDS.** The placement gives the reference's cells for every layout under both rules,
  the same order and no unit latency moved (T8).
- **(d) HOLDS.** All sixteen steady rows keep their verdict and stop, at the stored outer count
  (T9).
- **(e) FAILS.** The 320x120 dust release ran 6.58 times faster under `core`, below the 8 the
  prediction asks (T11). One logical processor of the machine was busy throughout (section 6).
- **(f), (g) and (h) HOLD** (T14, T15).

### 1. The runner and the runs (T1)

`docs/reports/probe97/probe97.py` makes every run, comparison, harness row and timing session
of this step. Its `queue` command is the runner: a Python process under WSL2 started through
`Win32_Process.Create` (`launch97.ps1`), so it has no console window to close and no session to
end with. Beside it runs a holder of `SetThreadExecutionState` (`keepawake97.ps1`), which keeps
the machine out of idle sleep while the runner lives. The runner starts no server and listens
on nothing. A job is done when its record exists, so the runner skips it when started again; an
interrupted run loses only itself, and its partial files are removed when it starts again. The
runner starts a job when the records it needs exist (a run that keeps another's flow states
waits for that run's record). It runs at most three solver processes at once and at most two on
320x120 (the 320x120 harness row among them), with one comparison beside them, and the timing
sessions last, each alone. After each job it reads the job's record against the prompt's stop
rules:
- a run that does not end with exit 0;
- a band pair with a field beyond its limit;
- a pair of the backends with any failing item;
- a placement decision that differs;
- a harness row whose verdict or stop changes.

On a stop it writes `results/builder97/HALT`, starts nothing more and lets the jobs in progress
finish.

T1 lists the runs. On each side the flow states are solved once and shared, as prompt 69 did.
The dust release and the door leak solve theirs from rest. The clean-out, the filter breach and
the seal leak take the dust release's closed room through `--flows-from`, and on 160x60 the
filter breach takes the dust release's. A flow state solved on one backend is never copied into
a run on the other (`solver.backend` is a flow input). The variant `diffusion_tol` times 0.99
takes the reference's flow states, since the flow solve reads nothing of the transport section
(`time_integration.FLOW_FREE_SECTIONS`; P2 found the probe's diffusion variants' flow states
bit for bit the reference's). `outer_one_later` solves its own, with the probe's patch
installed in its process.

The band pair `band_dust_320x120_diffusion_tol_0.99` stopped the runner at 10:20 on 2026-11-27.
That HALT is kept as `results/builder97/stop_record/HALT_97.txt`. Prompt 97b removed it and
started the runner again at 13:10 through `sweeps97.py queue` (`launch97.ps1 -Queue queue97b`).
That queue ran the five instrumented runs and their analysis (T13) under the same caps, then
the rest of prompt 97's queue: the seal leak's reference, the core's runs, the placement,
VAL-018's row and the timing last. The halted pair's record stands, and the stop rule on a
band spread beyond its limit, which prompt 97b answers, was not read again. No other stop
occurred.

### 2. The band on 320x120 (T2, T3)

T2 sets each variant's largest difference beside the probe's spread and the limit. The flow
fields' 320x120 spread comes from `outer_one_later` on both scenarios, and the door leak's
diffusion variant is bit for bit the reference. Two fields are beyond their limits, both under
the diffusion variant on the dust release: the concentration pointwise and the leftover's
change. Under every variant no decision moved and every budget closed (T3). The 320x120 spread
in T2 is what the comparator reads as the first tier on that grid (`band320.json`). The core's
320x120 pairs are read against it, with the limits unchanged.

### 3. REQ-N03 level 2 (T4 to T7)

On all seven pairs the core stops each flow state at the reference's outer count, for the
reference's reason and under the same cap (T4). Every decision the comparator reads is equal,
item by item, with the same number reached on each side (T5). Every field is inside its spread,
read against T2's 320x120 spread on 320x120 and against the probe's on 160x60. The budget closes
to the reference's own check on both sides (T6). The core's fields lie several decades below the
spread they are read against. T7 shows which records are bit for bit: the step times and the
arrival maps on every pair, the frames and the budgets on none, as rounding gives. All seven
pairs pass in ADR-017 H's first tier, and they pass under the proposed limits too (T12).

### 4. The placement (T8)

`compare_placement` over the four contaminating scenarios gives the reference's detection
units, the reference's cells for every layout under each rule, and the same order of the
layouts. No unit latency moved.

### 5. The validation rows (T9, T10)

Every steady row of record was run once on the core. Each keeps its stored verdict and stop at
the stored outer count, and its value agrees with the stored one, with one exception
(T9). VAL-018's value is its outer count over `max_simple_iter`. The base file's cap has changed
since the row of record (T9 gives both caps), so the value moved while the outer count did not.
That is a change of the configuration, not of the solve. The tier-2 rows stay on the reference
(ADR-017 F, option 1) and were not rerun (T10).

### 6. The wall time (T11)

The timing ran last, each session alone, three sessions on 80x30 and on 160x60 with the order of
the backends alternated, and the one session (e) asks for on 320x120. Every session was on mains
power at the processor's full performance, with no model server running.

The ratio falls as the grid is refined: about thirteen on 80x30, about nine on 160x60, and on
320x120 below the eight (e) asks, in the flow solve and in the march alike (T11). The sessions on
each grid agree with each other to within about a tenth. Why the core gains less on the finer
grids is not measured here: neither side was profiled on 320x120.

The machine was not quiet. In every session's state, before and after, the busiest Windows
process is `python3.13` at 93 to 98% of one logical processor (T11). It is a stray `python3 -`
process left by a shell command of an earlier builder session on 2026-11-26 at 19:50, reading a
standard input that never closes. It was found only when the timing's records were read, so it
ran through every run of this step. It moves no result that is compared bit for bit or against a
limit. The machine has 24 logical processors, and the process loads both sides of a session
alike, so it is unlikely to account for a ratio of 6.58 against 8. Whether it does is not
measured: this session was not permitted to stop the process, and the timing was not run again.
Stopping it and running the 320x120 session again would settle it (For Alex).

## Prompt 97b: the sweep counts behind the 320x120 spread

**Date:** 2026-11-27
**Prompt:** 97b (`docs/prompts/cc-prompt-97b-sweep-counts.md`)

Prompt 97 stopped at item 1. On the 320x120 dust release, `diffusion_tol` times 0.99 moved the
pointwise concentration by 4.8e-6 (limit 1e-6) and the leftover's change by 7.6e-8 of the supply
(limit 1e-8); on the probe's grids the same variant moved them by 5.8e-8 and 3.6e-10.

**Alex's decision (2026-11-27).** The 320x120 limits are not set until the mechanism behind the
spread is measured. The hypothesis (H1, the orchestrator's): a tolerance 1% tighter makes some
steps' implicit solves take one more Jacobi sweep; a finer grid's sweep contracts less, so more
steps land inside that 1% and more stops flip, and the spread grows with the number flipped. A
port's rounding flips a stop only within rounding of the tolerance, so the variant may overstate
what a port does; the measurement says by how much.

### Prompt 97b's predictions

Set by the orchestrator in prompt 97b, and committed alone before the probe runs.

- (f) Each instrumented run that has a run of record on this machine is that run bit for bit (the
  160x60 variant has none; its base does).
- (g) Every class field's first difference falls at or after a step whose sweep count differs, on
  both grids (H1; refuted if a field differs before any count does).
- (h) The share of class-steps whose count differs is larger on 320x120 than on 160x60.

If (f) or (g) fails, the probe records it and stops: the spread then has another cause, and it
goes to Alex.

### How they are scored (the builder, before the probe runs)

The probe is `docs/reports/probe97/sweeps97.py`. It runs the reference's dust release, as it is
and with `diffusion_tol` times 0.99, on 320x120 and on 160x60, and the door leak as it is on
320x120, each through `run_scenario.main` in one process with the run's flow states from its run
of record (`--flows-from`), so that only the march is computed. The instrument wraps the Jacobi
solve's acceptance object in the probe's process and records, for every step and class, the
sweeps, what stopped the solve (the max-norm test with the leftover within its bound at the first
sweep that met it, the leftover's bound at a later sweep, or the cap), the stopping residual over
`diffusion_tol` times the largest right-hand side, and the leftover over its bound. It reads the
arrays the solve forms and computes nothing the solve returns.

- **(f).** Each instrumented run's file against its run of record in `results/builder97/runs/`
  (the reference's dust release and door leak on 320x120, the variant's dust release on 320x120,
  the reference's dust release on 160x60): every stored frame of every class field, every sensor
  channel and every budget term equal bit for bit. Holds when all four are.
- **(g).** Per grid and class, the first step whose sweep count differs between base and
  variant, and the first stored frame at which the class field differs (the probe's runs, each
  frame against the other side's). Holds when, on both grids and for every class whose field
  differs, the first differing count's step is at or before the step of the first differing
  frame. A class whose field never differs is reported and does not count against it.
- **(h).** The class-steps whose sweep count differs over all class-steps of the run, per grid.
  Holds when the 320x120 share is larger than the 160x60 share.

### Results of prompt 97b

#### 7. The instrument changes nothing (T13, T14)

The five instrumented runs ran under the same caps as the rest of the queue (T13's last
column). Each of the four that has a run of record equals it bit for bit in every record the
comparator reads, the frames, the budgets and the sensors among them (T14). (f) holds. A
replayed step reproduces the run's own step bit for bit at every flipped class-step (T17a).

#### 8. The counts by grid and time (T15, T16)

On 320x120 the variant's count differs from the base's in a few per cent of the class-steps,
on 160x60 in five class-steps, all at one step (T15). In every one the variant took one sweep
more, never fewer and never more than one. (h) holds.

On both grids, every class's field first differs at the step whose count first differs. The
digest of the field each step returns first differs at that step, and the first stored frame
that differs is the next frame after it (T15). (g) holds. From that step on the two runs carry
different states, so later steps' fields differ with their counts equal (T15's twelfth
column); the base's flips are counted against the variant's counts at the same step.

No count differs before the release ends at 40 s, and none after 60 s (T16). On 320x120 the
flips begin at 41.8 s and nearly all fall between 50 and 60 s. On 160x60 they fall at one step
just after 40 s.

#### 9. The effect of one flip (T17a, T17b, T20)

From the base run's state at each flipped class-step, the probe solved the step again with one
sweep more and one fewer. T17a gives the field difference that one step makes against the step
as it ran. The effect is nearly the same at every flip, the median close to the largest, and
one sweep fewer moves the field several times as far as one more, since each sweep contracts
what is left (T18). One sweep more moves the concentration, its pointwise reading and the
leftover by a fiftieth of their limits or less (T19's last column).

The leftover's change adds over the flips. For each class, the variant's change of the
leftover at the end equals the sum, over that class's flipped steps, of one sweep more's
change, to the sixth digit (T20). The fields do not add in that way, since each flip's
difference is carried by the flow and diffused after it is made. The pointwise spread lies
between one flip's effect and the sum the class's flips would give if they added (T22, T15).

#### 10. Why the flips fall after the release (T18)

T18 reads every solve's stopping residual over its max-norm threshold. A stop flips under the
variant when that ratio lies above 0.99, since from the same state the variant's threshold is
1% lower.

- **During the release (to 40 s).** The leftover's bound stops most solves on both dust runs.
  The max-norm test then holds with room to spare: the residual sits well below its threshold,
  and no stop on any run lies above 0.99.
- **After the release.** The source is off, the flow is steady and the class fields decay with
  a shape that changes less and less. The first residual over the threshold and the contraction
  per sweep are then nearly the same at every step, so nearly every solve stops after the same
  number of sweeps at nearly the same fraction of its threshold. That fraction is where the grid puts
  it. On 320x120 the base's median is 0.9894, its 90th percentile 0.9895: a hair below the
  variant's 1%. Between 41.8 and 60 s, while the shape still settles, the fraction moves above
  0.99 in the class-steps T16 counts; after 60 s it stays just below, and no stop flips. On
  160x60 the fraction settles near a quarter of the threshold, and only the one step just after
  40 s lies above 0.99. The door leak's stops sit near a tenth of the threshold before and after
  40 s, so its variant moved nothing, as prompt 97 found.

So H1's grid dependence is not the one H1 gave. The finer grid's sweep does contract less
(T18's contraction per sweep), which widens the range one sweep spans. Alone, that would raise
the share of flipped stops by a fraction, not by orders of magnitude. What decides the share is
where the settled fraction falls against the variant's 1%. On 320x120 it falls in the margin,
on 160x60 and on the door leak far from it. A grid between them, or a tolerance changed by 2%
rather than 1%, could give either outcome.

#### 11. How far the variant overstates a port (T21, T6)

A port's residual differs from the reference's by rounding, so it fails, or passes, only a
stop that lies within rounding of the threshold. T21 counts the base runs' stops by their
distance below the threshold. Those within 1e-2 are the stops the variant flips. On every base
run no stop lies within 1e-6 of its threshold; the closest is listed. A port whose rounding
moves the residual by much less than that flips none of them. The core's runs agree. On the
320x120 dust release, whose flips T17a measures, the core's fields stay five decades or more
below one flip's effect (T6), so no Jacobi stop flipped under the core there. On the other four
320x120 scenarios the core's fields stay more than two decades below it. On this grid the
variant measures thousands of flipped stops; the core, on the dust release, none.

#### 12. H1's verdict

H1 holds in its mechanism and fails in its account of the grid:

- **Holds.** A tolerance 1% tighter makes some steps' implicit solves take one more Jacobi
  sweep. The fields first differ at the first such step on both grids. The leftover's spread
  is the sum of the flips' effects. The spread grows with the number flipped.
- **Fails.** The finer grid's slower contraction is not what puts more steps inside the 1%. The
  320x120 dust release's settled stopping fraction happens to lie just outside the variant's
  margin (section 10).
- **The variant overstates a port.** As H1 expected, it does so by far: the variant flips
  thousands of stops on 320x120, and a port's rounding could flip only stops within rounding of
  the threshold, of which the base runs have none (section 11).

#### 13. The 320x120 limits, proposed and not applied (T19, T22, T12)

ADR-017 H is not edited beyond a dated as-built note, and the comparator's limits are not
changed. Three readings are set out for Alex:

1. **ADR-017 H's rule on the 320x120 spread (T19).** The largest measured times ten, rounded up
   to a decade. The pointwise concentration's limit becomes 1e-4 and the leftover's 1e-6; the
   flow and the concentration keep theirs. On the sensor channels the rule gives less than the
   current limit, and the proposal keeps the larger. This makes the band as wide as the
   variant's thousands of flips. A port that flipped no stop but carried a defect of that size
   would pass.
2. **A limit per flipped stop (T22).** Ten times the largest effect of one flipped stop, rounded
   up to a decade. That gives the current limits exactly: 1e-6 for the concentration and its
   pointwise reading, 1e-8 for the leftover. Read this way, the current limits already allow
   one flipped stop per class, and the variant's 320x120 run exceeds them because it flipped
   up to 1,385 per class (T15). A limit that scales with the flips needs the comparator to count them.
   A run file does not record the Jacobi counts (ADR-017 H, the comparator as built), and
   recording them is a change under `src/` that this pass does not make. The leftover adds
   exactly over its flips (T20), so a per-flip limit would be exact for it. The fields carry
   each flip's difference away, so the product is a loose upper bound for them (T22's sixth
   column).
3. **The current limits kept, the 320x120 spread recorded.** A port flips a stop only within
   its rounding of the threshold, and the base runs have no stop within 1e-6 (T21). The variant
   then measures how far a stop flipped by a 1% change moves the fields. It does not measure
   what a port does, and on 320x120 the two differ by orders of magnitude.

The choice moves no verdict. Every pair of the backends passes under the current limits and
under the proposed ones, each field in the first tier, read against the same spread (T12). The
band pair that stopped prompt 97 is within the proposed limits and not within the current ones.

## For Alex

1. **The 320x120 limits** (section 13). Three readings are set out, none applied, and none moves
   a verdict of the seven pairs. Step 6's level 2 holds under each of them; the step closes
   when one is chosen.
2. **Records of two commits** (Results, the first paragraph). Prompt 97 asked for a check that
   fails on records of two commits. `tables97.py` accepts the records of 69744ab and 77e9b3d
   because no file that computed a record differs between them, names both commits and the
   files that do differ, and refuses sweep records of more than one commit. If the check is to
   refuse two commits outright, prompt 97's records have to be made again at one commit.
3. **The wall time** (section 6). (e) fails at 6.58 on a machine with one logical processor
   busy throughout. The stray process is the WindowsApps `python3.exe` reading standard input
   (`python3 -`), started 2026-11-26 at 19:50. With it stopped and the session's folder moved
   aside, `probe97.py timing 320x120 1` alone (about 50 minutes) would say whether the ratio
   moves. If it stays below 8, the fall of the ratio with the grid is the finding, and a
   profile of both sides on 320x120 would say where the core gains less.
4. **VAL-018's value** (section 5) moved only because the base file's `max_simple_iter`
   changed after the row of record. T9 gives both caps.
