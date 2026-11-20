# Phase 6 Step 2c: the Scalar Kernels Made Fast (prompt 89)

**Date:** 2026-11-19
**Prompt:** 89 (`docs/prompts/cc-prompt-89-scalar-perf.md`)
**Branch:** `phase6/scalar-perf`
**Status:** built. Predictions (a), (b), (c) and (d) hold; (e) fails, on `-march=native`
(T0). No change that moved a bit was found or kept. One of the prompt's stop conditions
applies and is reported, not stopped on: under the project's flags the flux is within twice the
floor of the loop it is, and not within twice the floor two faces a division would allow
(section 3.5 says what limits it).

This is a performance pass on the two kernels of Phase 6 step 2 (`docs/PROJECT_PLAN.md`, Phase 6,
the steps table; ADR-017 A, C, D and G), before the march copies their style: the Jacobi sweep
and its residual, the implicit set-up, the conjugate gradient's product and updates, and the
advective flux with its limiter. The boundary was the same arithmetic, faster: every value the
reference computes, and the order it computes it in, is unchanged per element. Compiler flags
were measured and not changed; changes to the arithmetic were studied and not built. Nothing
under `src/` or `configs/` changes.

## The predictions

Set by the orchestrator in prompt 89 on 2026-11-19, and committed alone before any change.

- (a) After the pass every parity test, GoogleTest case and captured replay of prompts 87 and 88
  passes unchanged.
- (b) Under the current flags, a Jacobi sweep on 88's developed 320x120 frame costs at most
  0.07 ms (at least four times faster than now).
- (c) The implicit set-up with one residual (88's "no sweep") costs at most 0.15 ms.
- (d) The flux on 87's developed frame is at least 1.5 times faster per call than the core's now.
- (e) Once restructured, `-O3` and `-march=native` change the kernels' time by less than 25%:
  the structure matters more than the flags.

## How they are scored (the builder, before any change)

Every timing is taken under WSL2 with the Linux environment, one thread
(`OPENBLAS_NUM_THREADS=1`), one timing process at a time, the core called through the binding with
its scratch and output allocated once, as prompts 87 and 88 timed it. A timing is the median of
repeated calls after unrecorded warm-up calls. Each is taken three times, in three sessions of the
probe; run 1 is the one scored, and runs 2 and 3 are reported beside it and do not change a score.
"Now" is the core at `origin/main` (3c8d1ec), built again under the project's flags and timed in
the same session as the restructured core, the two builds in alternating processes, so that a
comparison of the two is not a comparison of two days' machine states. The numbers of prompts 87
and 88 (their T6 and T6b) are quoted beside, not scored against.

- **(a).** "Unchanged" is a claim about the tests: no line of an existing case in
  `tests/test_core_parity.py`, `core/tests/` or the replays of `probe87.py` and `probe88.py` is
  edited or removed between `origin/main` and the last commit of this pass that changes `core/`
  (`git diff` shows additions only, and any addition is listed). The prediction holds when, at
  that commit, under the project's flags: CTest passes every case that `origin/main` has (68);
  the parity module passes every case it has (1,404); the 120 captured flux calls of prompt 87
  replay with no differing bit; the 100 captured Jacobi solves of prompt 88 replay with no
  differing bit in any output (field, sweeps, converged, leftover, leftover sums); and the 40
  captured CG solves stop at the reference's count with the field within 1e-12 of the
  reference's largest magnitude, and equal the reference's CG with the core's dot products bit
  for bit. One differing bit, one failed case or one edited case is a failure.
- **(b).** A sweep's cost as 88's T6b defines it: on class 0's developed 320x120 frame (the
  inputs of `probe88.developed_inputs()`), the median of the full solve less the median of the
  solve with the cap at 0, over the sweeps of the full solve. The prediction holds when that is
  at most 0.070 ms for the restructured core under the project's flags in run 1. The ratio to
  "now" in the same session is given beside it; the prediction's "four times" is its
  parenthesis, and the score is the 0.07 ms.
- **(c).** The median of the solve with the cap at 0 on the same inputs (the set-up, one residual
  and the leftover, the binding's checks included), restructured core, project's flags, run 1.
  Holds at or below 0.150 ms.
- **(d).** For each of the ten developed rows of 87's T6b (five classes' last frames of the
  stored 320x120 run, each under that class's last captured flux and inflow, directions x and
  y): the median of "now" over the median of the restructured core, both under the project's
  flags, same session. Holds when the ratio is at least 1.5 in every row in run 1.
- **(e).** The kernels' times are those of (b), (c) and (d) (the sweep, the set-up with one
  residual, and the flux's median over the five classes in x and in y), and the conjugate
  gradient solve per call on class 0's developed frame. Each is measured for the restructured
  core under four builds: the project's flags, `-O3` in place of `-O2`, `-march=native` added,
  and both. The change is a build's time over the project-flags time, less one. Holds when the
  magnitude of every change is below 0.25 in run 1. A build under which a parity test, a
  GoogleTest case or a replay fails is reported as such and its times are not scored.

A failed prediction is a finding, not a stop.

<!-- tables89 predictions begin -->
### T0. The predictions, scored by the rule committed with them

| Prediction | Measured | Verdict |
|---|---|---|
| (a) every parity test, GoogleTest case and captured replay of prompts 87 and 88 passes unchanged | 68 of 68 GoogleTest cases and 1,404 of 1,404 parity cases that 3c8d1ec has; 120 of 120 captured flux calls and 100 of 100 captured Jacobi solves bit for bit; 40 of 40 captured CG solves at the reference's count, largest difference 1.86e-16, and 40 of 40 bit for bit with the core's dot products (T9); no line of an existing case edited or removed (section 6) | holds |
| (b) a Jacobi sweep at most 0.07 ms under the current flags | 0.0526 ms in run 1 (0.0509 and 0.0503 in runs 2 and 3); 1.91 times faster than the core before the pass in the same session (2.03, 1.97), not the four times of the prediction's parenthesis (T4) | holds |
| (c) the set-up with one residual at most 0.15 ms | 0.1063 ms in run 1 (0.1070, 0.1056) (T4) | holds |
| (d) the flux at least 1.5 times faster per call than the core's before | 1.90 to 2.25 times over the ten rows in run 1 (1.99 to 2.24, 1.97 to 2.29) (T4) | holds |
| (e) `-O3` and `-march=native` change the kernels' time by less than 25% | the largest change in run 1 is 29%, a Jacobi sweep under `-O3 -march=native`; at or beyond 25%: a Jacobi sweep under `-O3 -march=native` (-29%); the flux along y under `-O2 -march=native` (+28%); the set-up with one residual under `-O2 -march=native` (-26%) (runs 2 and 3: 29%, 28%); the checks under each set: all pass (T7) | FAILS |
<!-- tables89 predictions end -->

The tables are generated by `docs/reports/probe89/tables89.py` from the records in
`results/builder89/` (untracked, this machine) and copied here between markers;
`tables89.py --check` regenerates them and compares, and fails if a marker pair is missing. The
full set is `docs/reports/probe89/tables89.md`. A number in the text that is not in a table is
counted from the code, and says so.

## 1. The floor

What a sweep and a flux call could cost here if only the arithmetic and the memory traffic were
paid for. The counts and `floor89.cpp` were made and first run before any change to `core/`; T2
is its run beside the final timing sessions, the first run's record having been overwritten by
it.

**Counted from the code** (the core's loops at 3c8d1ec, which are the reference's operations):

| Kernel | Floating-point operations | of them divisions | Bytes read | Bytes written | Arrays of a 320x120 field it touches |
|---|---|---|---|---|---|
| a Jacobi sweep, per cell | 14: the neighbour sum (4 products, 3 additions), the residual (an addition, a product, a subtraction), the stop's magnitude and comparison, the update (a division, an addition) | 1 | 41: `b`, `a_P`, the iterate, the two conductance fields (8 each), the SOLID flag (1) | 16: the residual and the next iterate | 7 of 0.29 MiB and the flags: 2.1 MiB |
| the set-up, per cell | 10: the diagonal's four additions (three among the conductances, one for the sink) and its test against zero, `V / dt`, `a_P`, `b`, the magnitude and its comparison | 1 | 40: the two conductance fields, the sink, the volume, `C*` | 16: `a_P` and `b` | 7: 2.1 MiB |
| the limited QUICK flux, per interior face | 25: the sign's comparison, QUICK's value (3 products, 2 additions), the clamp (3 subtractions, 2 divisions, 6 products, 2 additions, 3 minima, a maximum, a comparison), the flux's product | 2 | 17: the flux, the field (each cell is read by four faces but comes from memory once), the far cell's SOLID flag; the weights are one-dimensional | 8: the face flux | 3 and the flags: 0.92 MiB |

A written array that is not in the cache is read before it is written, so the traffic is 73
bytes a cell for a sweep, 72 for the set-up and 33 bytes a face for a flux call.

**The working set against the caches.** `docs/reports/profile84.md` T9 gives the field's size,
0.29 MiB on 320x120, against the GPU's 32 MiB L2. The processor's caches are in T1: 48 KiB at the
first level, 1 MiB at the second per core and 16 MiB shared at the third. A sweep's arrays total
about twice the second-level cache, so a sweep streams them from the third level; a flux call's
fit the second level with little to spare.

**What is assumed, with its source.**

- *The vector width under the project's flags is two doubles.* The flags name no `-march`, so GCC
  targets the x86-64 baseline, whose widest vector is SSE2's 128 bits; its vectorisation report
  says "16 byte vectors" for every loop it vectorises (T8). This machine also has AVX2 and
  AVX-512 (`lscpu`, `results/builder89/machine.txt`), which `-march=native` uses.
- *The rates are measured on this machine, not taken from a table.* `floor89.cpp` times kernels
  written with intrinsics at each width (T2): a division, the residual's arithmetic, and the
  whole sweep on flat arrays, once at a size that fits the first-level cache (the arithmetic
  alone) and once at 320x120 cells (the arithmetic with the memory traffic), and the rate at
  which nine arrays of that size move. The whole sweep at 320x120 is an ideal kernel: no row
  ends, no boundary rows, the masks already 64 bits wide.

<!-- tables89 machine begin -->
### T1. The machine, and what else ran

| Item | Value |
|---|---|
| CPU | AMD Ryzen AI 9 HX 370 w/ Radeon 890M, 24 logical processors |
| Cache level 1 (data) | 48K, shared by logical processors 0-1 |
| Cache level 1 (instruction) | 32K, shared by logical processors 0-1 |
| Cache level 2 (unified) | 1024K, shared by logical processors 0-1 |
| Cache level 3 (unified) | 16384K, shared by logical processors 0-23 |
| Compiler | g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 |
| `-march=native` resolves to | `-march=znver4` |
| Kernel | 6.6.114.1-microsoft-standard-WSL2 |

Each session of `probe89.py session` times the reference and every build, one process after another. Before each process the probe samples Windows for two seconds: the busiest processes, in percent of one logical processor, and the processors' performance percentage. The calibration is the median time of `np.sqrt` over 4,096 doubles in that process, the same compiled code whatever build is timed.

| Run | Started | Calibration, us (lowest to highest over the processes) | Busiest processes at the first process | Processor performance |
|---|---|---|---|---|
| 1 | 2026-11-19T19:51:29 | 3.42 to 3.97 | wow 124, searchindexer 93, code 83, msmpeng 31 | 151 |
| 2 | 2026-11-19T19:54:26 | 3.43 to 3.82 | wow 113, msmpeng 26, system 22, code 18 | 141 |
| 3 | 2026-11-19T19:57:21 | 3.42 to 3.96 | wow 118, msmpeng 32, battle.net 21, system 19 | 156 |
<!-- tables89 machine end -->

<!-- tables89 floor begin -->
### T2. The floor: this machine's rates, measured by `floor89.cpp`

Each rate is the best of repeated timings of a kernel written with intrinsics at a stated width: 2 doubles an instruction (SSE2, what the project's flags allow), 4 (AVX2) and 8 (AVX-512). A cycle is a quarter of a step of a dependent 64-bit multiply-add chain.

| Rate | 2 doubles | 4 doubles | 8 doubles |
|---|---|---|---|
| a cycle, ns (4.83 GHz) | 0.2071 | | |
| a division, ns per element, arrays in the first-level cache | 0.408 | 0.204 | 0.230 |
| the residual's nine operations, ns per cell, first-level cache | 0.704 | 0.371 | 0.396 |
| the whole sweep, ns per cell, first-level cache | 0.857 | 0.443 | 0.450 |
| the whole sweep, ns per cell, 320x120 cells | 0.974 | 0.663 | 0.693 |
| **the sweep's floor, us per sweep of 38,400 cells** | **37.4** | **25.5** | **26.6** |
| the sweep with memory free (first-level cache rate), us per sweep | 32.9 | 17.0 | 17.3 |
| **the flux's floor from its two divisions a face, us per call of 38,520 faces** | **31.4** | **15.7** | **17.8** |
| the set-up pass from its one division a cell, us per pass | 15.7 | 7.8 | 8.8 |

A scalar division takes 0.816 ns (3.9 cycles), so a flux loop that divides one face at a time cannot take less than 62.9 us a call. Nine arrays of 38,400 doubles, read and written as the sweep's are, move 128 GB/s: at that rate a sweep's 73 bytes a cell take 21.9 us, a flux call's 33 bytes a face 9.9 us and the set-up pass's 72 bytes a cell 21.6 us.
<!-- tables89 floor end -->

**The floors.**

- *A Jacobi sweep:* T2's first bold row, the ideal kernel at 320x120, measured. Under the
  project's flags it is the two-double column. The sweep is bound by its arithmetic there, not by
  memory: the same kernel with every array in the first-level cache is only a little faster (the
  row below it), and the sweep's bytes at the arrays' measured rate take less than its arithmetic
  does (T2's last sentence). At four doubles the memory's share grows.
- *A flux call:* estimated, not measured with an ideal kernel. Each face needs two divisions and
  a divider does one at a time, so a call cannot take less than its faces times two divisions at
  the width's rate: T2's second bold row, and under it the figure for a loop that divides one
  face at a time. The memory's time is below all of them. The other 23 operations are not in this
  floor; section 3.4 measures what they cost.
- *The set-up pass:* its one division a cell and its bytes at the arrays' rate (T2); the larger
  of the two bounds it.

## 2. The style, and what decided it

Prompt 88 found that GCC vectorised no loop of the solve under the project's flags. Before
changing the core, one sweep's arithmetic was written six ways in a file of its own
(`docs/reports/probe89/styles89.cpp`) and timed under each flag set.

<!-- tables89 styles begin -->
### T3. One sweep written six ways (`styles89.cpp`), us per sweep of 320x120 cells

The best of 2,000 sweeps of a synthetic field, the cells 1 to 318 of each row. `blocks` is the style kept. The checksums of the six agree under every flag set.

| Style | `-O2` (the project's) | `-O3` | `-O2 -march=native` | `-O3 -march=native` | `-O2 -march=x86-64-v3` |
|---|---|---|---|---|---|
| `plain` | 64.4 | 66.3 | 63.4 | 59.0 | 60.2 |
| `lambda` | 119.5 | 69.0 | 116.5 | 61.2 | 116.3 |
| `lane_arrays` | 53.4 | 45.0 | 35.4 | 29.3 | 37.8 |
| `simd` | 40.3 | 43.4 | 34.1 | 27.9 | 31.2 |
| `blocks` | 41.8 | 45.1 | 36.3 | 29.2 | 36.8 |
| `intrinsics` | 37.6 | 40.4 | 39.5 | 36.4 | 39.1 |

Distinct checksums over the 30 runs: 1.
<!-- tables89 styles end -->

What the six, and the smaller experiments behind them (`results/builder89/exp/`, not committed),
showed about GCC 13.3 at `-O2`:

- **A loop is vectorised only if its trip count is a constant.** At `-O2` GCC's vectoriser uses
  its "very cheap" cost model, which vectorises a loop only where the vector code replaces the
  scalar code entirely, with no scalar loop for the cells left over and no run-time check (GCC's
  manual, `-fvect-cost-model`; `-O3` uses the "dynamic" model). A loop to a row's length fails
  that; a loop over a block of eight cells passes. `plain` is the first, and it stays scalar
  under `-O3` as well, for the next reason.
- **Every store must be known not to alias a load.** Without that the vectoriser needs a run-time
  overlap check, which the cost model refuses. `restrict` on a function's parameters gives it,
  and only the outputs need it; on a local pointer or a struct member it does not carry, and a
  lambda that captures the pointers loses it (`lambda`, the slowest of the six: the compiler
  reloads every pointer after every store).
- **The inputs may come in a struct**, if the function that walks the row receives the struct by
  value (a copy nothing else can point to, which the compiler then holds in registers) and if a
  helper that takes the struct is inlined early. Left to its own judgement GCC inlined such a
  helper after it had decided what to hold in registers, and the block stayed scalar; with
  `[[gnu::always_inline]]` it vectorised. T6's `helpers_not_forced_inline` is that one attribute
  removed from the kept code.
- **A select on a `bool` array is never vectorised**, under any flag set tried: the mask's
  elements are one byte and the values' eight. A mask held as 64 bits per cell and applied with
  bit operations is.
- **A reduction is vectorised as an OR of masks or as one running value per lane**, not as a
  floating-point maximum, whose order GCC will not change without `-ffast-math`.
- **Local lane arrays vectorise but run slower** under `-O2` than a block function with direct
  stores (`lane_arrays`): each value is stored to the stack and copied out.
- **`std::experimental::simd` is level with the kept style** in this file under every flag set,
  and ahead of it under `-march=x86-64-v3` (`simd`). It states the vector in the source, so it
  does not depend on the conditions above, and it reads a `bool` mask as it is. It was not
  adopted, and the reasons are judgement, not speed: it is a library extension header
  (libstdc++'s `experimental/simd`), and a block of it is a template on a vector type where the
  kept style's block is scalar code a reader checks against the reference line by line. The
  choice was made during the pass on timings of an earlier arrangement of the same six, in which
  it trailed under `-O2` (`results/builder89/exp/`, not records); T3 is the record, and on it the
  two are equal. F9 puts the choice to Alex.

`blocks` under the project's flags is within a small factor of `intrinsics`, the hand-written
SSE2 loop on the same rows, which is the floor's kernel, and is at or near the best of the six
under every flag set. That is the style kept; section 8 writes it down.

## 3. The kernels

Each kernel was restructured in its own commit, in the prompt's order, and after each the parity
module, CTest and the captured replays were run and the kernels timed (`probe89.py check` and
`time`). Two small commits follow the flux's (section 3.1, the row's end cells). T4 is before and
after; T5 the kernels after each commit; T6 the forms that were tried against the kept ones.

<!-- tables89 kernels begin -->
### T4. Each kernel before and after, on the developed 320x120 frames

Medians of repeated calls through the binding under the project's flags, in microseconds: `before` is the core at 3c8d1ec, `after` the head, the reference `src/scalar_scheme.py`. The inputs are prompt 88's developed frame of class 0 for the implicit kernels and prompt 87's ten developed rows for the flux, whose median over the five classes is given. A sweep is the full solve less the set-up with one residual, over the solve's sweeps; a CG iteration likewise from the CG solve and the same solve with a stop its first residual meets. Run 1 is the one scored.

| Kernel | Run | Reference | Before | After | Before / after | Reference / after |
|---|---|---|---|---|---|---|
| a Jacobi sweep | 1 | 168 | 100 | 52.6 | 1.91 | 3.20 |
| a Jacobi sweep | 2 | 150 | 103 | 50.9 | 2.03 | 2.95 |
| a Jacobi sweep | 3 | 153 | 99.1 | 50.3 | 1.97 | 3.05 |
| the set-up with one residual | 1 | 497 | 181 | 106 | 1.70 | 4.67 |
| the set-up with one residual | 2 | 227 | 183 | 107 | 1.71 | 2.12 |
| the set-up with one residual | 3 | 229 | 179 | 106 | 1.69 | 2.17 |
| the Jacobi solve | 1 | 1844 | 984 | 527 | 1.87 | 3.50 |
| the Jacobi solve | 2 | 1426 | 1010 | 514 | 1.96 | 2.77 |
| the Jacobi solve | 3 | 1456 | 972 | 508 | 1.91 | 2.87 |
| nothing to solve | 1 | 45.0 | 30.3 | 40.8 | 0.74 | 1.10 |
| nothing to solve | 2 | 39.9 | 29.8 | 40.6 | 0.73 | 0.98 |
| nothing to solve | 3 | 40.9 | 30.0 | 41.8 | 0.72 | 0.98 |
| a CG iteration | 1 | 218 | 170 | 94.5 | 1.79 | 2.31 |
| a CG iteration | 2 | 197 | 172 | 93.8 | 1.83 | 2.10 |
| a CG iteration | 3 | 196 | 172 | 95.1 | 1.81 | 2.07 |
| the CG solve | 1 | 1640 | 1241 | 704 | 1.76 | 2.33 |
| the CG solve | 2 | 1480 | 1253 | 702 | 1.78 | 2.11 |
| the CG solve | 3 | 1461 | 1254 | 713 | 1.76 | 2.05 |
| the flux along x | 1 | 874 | 173 | 87.3 | 1.98 | 10.01 |
| the flux along x | 2 | 865 | 174 | 86.5 | 2.01 | 10.00 |
| the flux along x | 3 | 823 | 171 | 85.7 | 1.99 | 9.60 |
| the flux along y | 1 | 795 | 175 | 78.2 | 2.23 | 10.17 |
| the flux along y | 2 | 772 | 173 | 77.7 | 2.23 | 9.93 |
| the flux along y | 3 | 736 | 170 | 78.3 | 2.17 | 9.40 |

The calibration of each process (T1), in microseconds, reference, before and after: run 1: 3.79, 3.78, 3.43; run 2: 3.79, 3.43, 3.81; run 3: 3.81, 3.80, 3.81.

The Jacobi solve takes 8 sweeps and the CG solve 6 iterations with 0 restarts and 0 fallbacks. Every timed Jacobi solve's outputs equal the reference's bit for bit (0 differing over the five classes). On a 2x2 grid, where the arithmetic is nothing, a call of `implicit_step` takes 2.55 us and one of `advective_flux` 0.97 us: the binding's cost per call.

The Jacobi solve per class and the flux per class and direction, run 1:

| Class | Jacobi solve, before | after | ratio | Flux x, before | after | ratio | Flux y, before | after | ratio |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1032 | 580 | 1.78 | 171 | 89.7 | 1.90 | 174 | 77.6 | 2.24 |
| 1 | 989 | 551 | 1.80 | 175 | 88.1 | 1.99 | 174 | 78.2 | 2.22 |
| 2 | 991 | 542 | 1.83 | 174 | 87.3 | 1.99 | 175 | 79.0 | 2.21 |
| 3 | 989 | 518 | 1.91 | 173 | 86.4 | 2.01 | 177 | 78.9 | 2.25 |
| 4 | 986 | 527 | 1.87 | 170 | 86.6 | 1.97 | 175 | 77.9 | 2.24 |
<!-- tables89 kernels end -->

<!-- tables89 changes begin -->
### T5. Each change with its effect: the kernels after each commit, run 1

One build per commit, all under the project's flags and timed in one session, in microseconds. A kernel moves at its own commit, and at another's only by what the two share (the block walk, the masks) or by the session's noise: runs 2 and 3 are in the records.

| Kernel | before this pass (3c8d1ec) | after the Jacobi sweep (101b1ad) | after the set-up (34e11fe) | after the conjugate gradient (104a02d) | after the flux (1b98cf7) and the end cells' pointers: the head (6a13cfc) |
|---|---|---|---|---|---|
| a Jacobi sweep | 100 | 46.7 | 49.1 | 53.0 | 52.6 |
| the set-up with one residual | 181 | 140 | 104 | 108 | 106 |
| the Jacobi solve | 984 | 514 | 497 | 533 | 527 |
| nothing to solve | 30.3 | 29.3 | 40.7 | 40.9 | 40.8 |
| a CG iteration | 170 | 169 | 175 | 93.4 | 94.5 |
| the CG solve | 1241 | 1232 | 1241 | 698 | 704 |
| the flux along x | 173 | 173 | 171 | 167 | 87.3 |
| the flux along y | 175 | 171 | 171 | 168 | 78.2 |
| *the process's calibration* | 3.78 | 3.79 | 3.81 | 3.48 | 3.43 |

Run 2:

| Kernel | before this pass (3c8d1ec) | after the Jacobi sweep (101b1ad) | after the set-up (34e11fe) | after the conjugate gradient (104a02d) | after the flux (1b98cf7) and the end cells' pointers: the head (6a13cfc) |
|---|---|---|---|---|---|
| a Jacobi sweep | 103 | 47.6 | 48.2 | 53.0 | 50.9 |
| the set-up with one residual | 183 | 130 | 105 | 108 | 107 |
| the Jacobi solve | 1010 | 511 | 490 | 533 | 514 |
| nothing to solve | 29.8 | 29.2 | 41.3 | 52.5 | 40.6 |
| a CG iteration | 172 | 170 | 167 | 95.4 | 93.8 |
| the CG solve | 1253 | 1233 | 1198 | 715 | 702 |
| the flux along x | 174 | 166 | 176 | 169 | 86.5 |
| the flux along y | 173 | 171 | 173 | 170 | 77.7 |
| *the process's calibration* | 3.43 | 3.80 | 3.47 | 3.81 | 3.81 |

Run 3:

| Kernel | before this pass (3c8d1ec) | after the Jacobi sweep (101b1ad) | after the set-up (34e11fe) | after the conjugate gradient (104a02d) | after the flux (1b98cf7) and the end cells' pointers: the head (6a13cfc) |
|---|---|---|---|---|---|
| a Jacobi sweep | 99.1 | 46.1 | 47.3 | 52.1 | 50.3 |
| the set-up with one residual | 179 | 128 | 104 | 109 | 106 |
| the Jacobi solve | 972 | 497 | 482 | 526 | 508 |
| nothing to solve | 30.0 | 29.4 | 40.3 | 40.2 | 41.8 |
| a CG iteration | 172 | 171 | 172 | 92.5 | 95.1 |
| the CG solve | 1254 | 1238 | 1217 | 693 | 713 |
| the flux along x | 171 | 173 | 172 | 166 | 85.7 |
| the flux along y | 170 | 175 | 171 | 170 | 78.3 |
| *the process's calibration* | 3.80 | 3.80 | 3.82 | 3.81 | 3.81 |
<!-- tables89 changes end -->

<!-- tables89 candidates begin -->
### T6. The forms tried against the kept ones, run 1 (runs 2 and 3 in brackets)

Each candidate is the head with one form changed (`variants89.py`), built under the project's flags and timed in the same session as the head. The ratio is the candidate's time over the head's for the kernels the change reaches: above 1 the kept form is the faster. Every candidate computes the same values; the check column is CTest, the parity module and the replays on the candidate.

| Candidate | What it is | Kernel | Candidate, us | Head, us | Candidate / head | Calibration, candidate and head | Checks |
|---|---|---|---|---|---|---|---|
| `flux_blocks` | the limited QUICK flux as a block: masks and selects, no branch | the flux along x | 155 | 87.3 | 1.77 (1.75, 1.76) | 3.79, 3.43 | all pass |
| `flux_blocks` | the limited QUICK flux as a block: masks and selects, no branch | the flux along y | 131 | 78.2 | 1.67 (1.66, 1.66) | 3.79, 3.43 | all pass |
| `flux_blocks_by_sign` | the block flux, a block whose fluxes share a sign reading one side only | the flux along x | 133 | 87.3 | 1.53 (1.54, 1.57) | 3.47, 3.43 | all pass |
| `flux_blocks_by_sign` | the block flux, a block whose fluxes share a sign reading one side only | the flux along y | 117 | 78.2 | 1.49 (1.49, 1.51) | 3.47, 3.43 | all pass |
| `flux_paired_division` | the kept flux, the clamp's two quotients formed as one pair | the flux along x | 91.8 | 87.3 | 1.05 (1.00, 1.01) | 3.44, 3.43 | all pass |
| `flux_paired_division` | the kept flux, the clamp's two quotients formed as one pair | the flux along y | 80.6 | 78.2 | 1.03 (1.02, 1.01) | 3.44, 3.43 | all pass |
| `jacobi_copies` | the Jacobi iterate copied in and out, as prompt 88 held it | the Jacobi solve | 534 | 527 | 1.01 (1.06, 1.12) | 3.79, 3.43 | all pass |
| `jacobi_two_masks` | the Jacobi sweep reading two mask arrays where no cell is held | a Jacobi sweep | 52.2 | 52.6 | 0.99 (1.00, 1.03) | 3.97, 3.43 | all pass |
| `jacobi_two_masks` | the Jacobi sweep reading two mask arrays where no cell is held | the Jacobi solve | 532 | 527 | 1.01 (1.01, 1.04) | 3.97, 3.43 | all pass |
| `neighbours_by_pointer` | a stencil's row neighbours read through a pointer each, between the ends too | a Jacobi sweep | 58.3 | 52.6 | 1.11 (1.08, 1.10) | 3.79, 3.43 | all pass |
| `neighbours_by_pointer` | a stencil's row neighbours read through a pointer each, between the ends too | a CG iteration | 97.3 | 94.5 | 1.03 (1.03, 1.01) | 3.79, 3.43 | all pass |
| `helpers_not_forced_inline` | the helpers a block calls left to the compiler's own inlining | a Jacobi sweep | 70.6 | 52.6 | 1.34 (1.41, 1.42) | 3.79, 3.43 | all pass |
| `helpers_not_forced_inline` | the helpers a block calls left to the compiler's own inlining | a CG iteration | 116 | 94.5 | 1.23 (1.23, 1.21) | 3.79, 3.43 | all pass |
| `row_loops_not_blocks` | each walk one loop to the row's length, not blocks of a constant length | a Jacobi sweep | 71.1 | 52.6 | 1.35 (1.37, 1.40) | 3.79, 3.43 | FAILS: 1 GoogleTest and 0 parity cases |
| `row_loops_not_blocks` | each walk one loop to the row's length, not blocks of a constant length | the set-up with one residual | 133 | 106 | 1.25 (1.25, 1.28) | 3.79, 3.43 | FAILS: 1 GoogleTest and 0 parity cases |
| `row_loops_not_blocks` | each walk one loop to the row's length, not blocks of a constant length | a CG iteration | 132 | 94.5 | 1.39 (1.38, 1.38) | 3.79, 3.43 | FAILS: 1 GoogleTest and 0 parity cases |
<!-- tables89 candidates end -->

### 3.1 The Jacobi sweep and its residual (commit 101b1ad)

- **Blocks.** The sweep walks each row in blocks of eight cells through one block function
  (`sweep_cells`): the residual, its mask, the stop's test and the update. The SOLID and held
  cells are computed like any other and masked, where prompt 88's loop branched on two `bool`
  arrays per cell. T6's `row_loops_not_blocks` is the same code with every cell taken one at a
  time.
- **Row ends apart.** A row's first and last cell run the same block function with a count of
  one, on operands whose missing neighbour is a row of zeros in the scratch, the zero the
  reference pads with; a missing row above or below is the same row of zeros. The cells between
  need no test of where they are, and nothing is padded or copied.
- **One pointer for a cell and its two row neighbours.** Between a row's ends a cell's west and
  east neighbours are read either side of its own pointer. With a pointer for each (T6's
  `neighbours_by_pointer`) the block has more pointers than the processor has registers free,
  the compiler reloads two from the stack every iteration, and the sweep and the CG's product
  slow. The first timing sessions of this pass showed it: the head then read the neighbours by
  a signed offset (commit 62ad9ce), which has the same effect, and its sweep was a quarter
  slower than the same sweep three commits earlier. Commit 6a13cfc is the form kept; the
  sessions before it are in `results/builder89/at_62ad9ce/`.
- **What the sweep gave back.** T5 has the head's sweep about a tenth above the sweep right after
  its own commit, in all three sessions. The step is at the CG's commit, which moved the sweep
  onto the walker it shares with the CG's product and residual (section 3.3). Its cause was not
  isolated.
- **Two of the changes above are not shown to be faster.** Reading one mask array
  (`jacobi_two_masks`) and not copying the iterate (`jacobi_copies`) are within the sessions'
  spread of the head in T6. They are kept because each is less in the code and in memory (no
  second array to fill, no copy) and neither is slower; by the prompt's rule alone, measurably
  faster, they would not have been.
- **The stop as a mask.** The reference's `np.abs(r).max() <= stop` holds exactly when no cell
  has `|r| <= stop` false, so the sweep ORs one mask per cell instead of carrying a maximum. A
  NaN residual or a NaN stop sets the mask, as it fails the reference's test.
- **Masks of 64 bits.** A solve expands its `bool` masks once, at its start, into the scratch.
  With no held cell the two masks are one array, and the sweep reads one (`jacobi_two_masks`
  reads two).
- **No copy of the iterate.** The first sweep reads `C*` where the caller holds it and writes the
  next iterate into the caller's `out`; later sweeps alternate between `out` and the scratch, and
  the field is copied once at the end only if it is not already in `out`. Prompt 88's loop copied
  `C*` into the scratch and the result out of it (`jacobi_copies`).

### 3.2 The implicit set-up (commit 34e11fe)

One blocked pass forms the diagonal's extra terms, whether any is non-zero, `V / dt`, `a_P`, `b`
and the largest `|b|`, where two passes did. The maximum is kept as eight running values, one per
cell of a block, and combined after the pass (a maximum is the same in any order); a NaN is left
out of it and flagged, and makes the stop NaN as `np.abs(b).max()` does. When nothing diffuses or
sinks the pass has still written `a_P` and `b`, which no caller reads: that call, the early
return, is the one thing this pass made slower (T4, "nothing to solve"), since it now divides
before it knows there is nothing to do. No caller of the product takes that path.

The set-up with one residual (88's "no sweep") is this pass, the masks, one residual pass with
its update, the leftover's pairwise sum and a copy of `C*` to `out`. The binding's own cost is a
few microseconds of it (T4's 2x2 call): 88's 0.1 ms for a call with nothing to solve was the
diagonal's pass and the copy, on a machine in a slower state (F5), not the binding.

### 3.3 The conjugate gradient (commit 104a02d)

- **Blocks.** The product, the true residual and the elementwise steps are block functions; the
  product and the residual share the sweep's neighbour sum and its walker with the row ends
  apart (`for_each_stencil_block`), so the bounds of a stencil's walk are written once.
- **Three walks an iteration where there were five.** A row is a block of the dot products
  (`reduce::blocked_dot`, prompt 88), so each row's dot product is taken as soon as the row is
  formed, while it is in the first-level cache, and the rows are added in order as `blocked_dot`
  adds them: `reduce::block_dot` is one block's term, and the sum of the rows equals
  `blocked_dot`'s bit for bit (GoogleTest `BlockDot`). The product is formed with `p . q`; `c`
  and `r` are updated with the stop's test, and `z = inverse * r` and `r . z` are formed beside
  them, used only if the iteration goes on (as the Jacobi sweep forms its next iterate); the
  direction's update is the third walk.

### 3.4 The advective flux and its limiter (commit 1b98cf7)

- **Rows, whatever the direction.** Prompt 87's loop walked each line along the flux's direction
  with that direction's stride and read every node through a test of where the line ends. The
  flux is now formed row by row of the field for both directions: for the faces normal to y a row
  of faces lies between two rows of cells, and a face's four nodes are four rows at the same
  column. Every load is contiguous.
- **Line ends by instantiation.** The first and the last interior face of a line take the
  boundary node in place of a far cell. The loop is a template on which end it is at, so the
  faces between carry no test; for y the first and last rows of faces are the ends.
- **Boundary faces and upwind as blocks**, with the boundary node a select.
- **Vents as a second pass.** A vent face's value replaces the scheme's, so the main pass ignores
  the vents and a pass over their flags rewrites the few faces that have one; with no vents it is
  not made.
- **The limited QUICK value stays one face at a time, with branches.** This is the one kernel
  where the block form lost under the project's flags, and by a wide margin: T6's `flux_blocks`
  is the same row walk with the face value as a block function, every choice a select and the
  clamp without a branch. A face's value is mostly choices (the upstream side, the far node
  inside an obstacle, the clamp's zero difference, three minima and a maximum with NumPy's NaN
  and tie rules). SSE2 has no instruction that chooses between two vectors, so a select is three
  instructions, and the block form's loop is about a hundred instructions for two faces
  (`results/builder89/exp/interior_run_y.s`, counted). The kept loop does what the flux's sign
  asks and no more, and on a developed flow, where the sign and the far cell change seldom from
  one face to the next, the processor predicts its branches. Under `-march=native` and
  `-march=x86-64-v3` the order is the reverse (T7's second table): AVX chooses with one
  instruction and carries four or eight faces, and the kept loop gains nothing from it.
- **Tried and dropped** (T6): the block form; the block form with one branch a block on the
  fluxes' signs, a block whose fluxes all point one way reading one side's nodes and weights
  only, which is faster than the plain block form and still slower than the kept loop; and, in
  the kept loop, the clamp's two quotients formed as one pair, since they share a divisor, which
  is no faster.

### 3.5 Each kernel against its floor

T4's "after" against T2's floors, both of this machine:

- *A Jacobi sweep* is within twice its floor (T2's first bold row, two doubles). What is left
  above it is counted in the code, not measured apart: the kept sweep takes the six cells a row
  of 320 leaves after its blocks and its two end cells one at a time, and builds a row's operands
  120 times a sweep, where the ideal kernel walks the field as one row. `blocks` of T3, which has
  no row ends either, sits between the two.
- *The flux* is within twice the floor of a loop that divides one face at a time (the sentence
  under T2), which is what the kept loop is, and is not within twice the floor two faces a
  division would allow (T2's second bold row, two doubles). What limits it under the project's
  flags is the choices, as section 3.4 says: the form that divides two faces at a time pays three
  instructions for each choice and is the slower (T6). With AVX the block form is the faster of
  the two (T7's second table) and is still several times its width's floor in T2: its other
  operations, not its divisions, bound it.
- *The set-up with one residual* is the set-up's pass, a residual pass (cheaper than a sweep,
  having no division), the masks, the leftover's sum and a copy. T4 gives the call; its parts
  were not timed apart, and the pass's floor in T2 is a part of it only.

## 4. The flags, measured and not applied

<!-- tables89 flags begin -->
### T7. The flags measured: the head's kernels under each set, run 1

Microseconds, and in brackets the change against the project's flags in runs 1, 2 and 3. The checks are CTest, the parity module and the captured replays on that build. The last column is not one of prediction (e)'s sets: `x86-64-v3` is AVX2 by a name that means the same on every machine, where `native` means this processor's AVX-512.

| Kernel | `-O2` (the project's) | `-O3` | `-O2 -march=native` | `-O3 -march=native` | `-O2 -march=x86-64-v3` |
|---|---|---|---|---|---|
| a Jacobi sweep | 52.6 | 50.9 (-3%, -2%, +3%) | 44.0 (-16%, -26%, -16%) | 37.1 (-29%, -23%, -22%) | 38.3 (-27%, -24%, -20%) |
| the set-up with one residual | 106 | 96.5 (-9%, -9%, -10%) | 78.6 (-26%, -29%, -27%) | 80.4 (-24%, -29%, -28%) | 99.9 (-6%, -27%, -31%) |
| the Jacobi solve | 527 | 504 | 430 | 377 | 406 |
| nothing to solve | 40.8 | 38.2 | 28.2 | 26.8 | 28.7 |
| a CG iteration | 94.5 | 83.0 | 78.1 | 77.7 | 65.4 |
| the CG solve | 704 | 631 (-10%, -9%, -10%) | 579 (-18%, -28%, -20%) | 579 (-18%, -22%, -24%) | 499 (-29%, -32%, -31%) |
| the flux along x | 87.3 | 87.4 (+0%, +1%, +2%) | 103 (+19%, +10%, +15%) | 95.6 (+9%, +11%, +11%) | 90.7 (+4%, +4%, +8%) |
| the flux along y | 78.2 | 77.9 (-0%, +1%, -0%) | 99.7 (+28%, +17%, +17%) | 90.9 (+16%, +19%, +16%) | 83.6 (+7%, +8%, +8%) |
| *the process's calibration, runs 1, 2 and 3* | 3.43, 3.81, 3.81 | 3.79, 3.50, 3.80 | 3.79, 3.80, 3.47 | 3.79, 3.80, 3.79 | 3.79, 3.71, 3.84 |
| **checks** | all pass | all pass | all pass | all pass | all pass |

The limited QUICK flux in its block form (`flux_blocks` of T6) under each set, against the kept loop under the same set, run 1 (runs 2 and 3 in brackets):

| Flags | Flux x: kept, us | block form, us | block / kept | Flux y: kept, us | block form, us | block / kept | Checks on the block form |
|---|---|---|---|---|---|---|---|
| `-O2` (the project's) | 87.3 | 155 | 1.77 (1.75, 1.76) | 78.2 | 131 | 1.67 (1.66, 1.66) | all pass |
| `-O3` | 87.4 | 152 | 1.74 (1.71, 1.72) | 77.9 | 127 | 1.63 (1.62, 1.65) | all pass |
| `-O2 -march=native` | 103 | 62.7 | 0.61 (0.69, 0.64) | 99.7 | 47.8 | 0.48 (0.53, 0.52) | all pass |
| `-O3 -march=native` | 95.6 | 65.5 | 0.69 (0.67, 0.70) | 90.9 | 47.6 | 0.52 (0.52, 0.53) | all pass |
| `-O2 -march=x86-64-v3` | 90.7 | 69.8 | 0.77 (0.78, 0.75) | 83.6 | 57.9 | 0.69 (0.69, 0.70) | all pass |
<!-- tables89 flags end -->

## 5. The vectorisation report

<!-- tables89 vectorised begin -->
### T8. The loops GCC vectorises: its report before and after

`g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`, `-fopt-info-vec-optimized`, per source file: the number of loops vectorised whole, and of straight-line parts (which are not loops).

| Source | Build | Loops vectorised | Vector width, bytes | Straight-line parts |
|---|---|---|---|---|
| `src/scalar.cpp` | before (3c8d1ec), `-O2` | 0 | none | 3 |
| `src/scalar.cpp` | `-O2` (the project's) | 10 | 16 | 6 |
| `src/scalar.cpp` | `-O3` | 7 | 16 | 10 |
| `src/scalar.cpp` | `-O2 -march=native` | 10 | 64 | 5 |
| `src/scalar.cpp` | `-O3 -march=native` | 10 | 32, 64 | 11 |
| `src/scalar.cpp` | `-O2 -march=x86-64-v3` | 10 | 32 | 5 |
| `src/reduce.cpp` | before (3c8d1ec), `-O2` | 2 | 16 | 0 |
| `src/reduce.cpp` | `-O2` (the project's) | 2 | 16 | 0 |
| `src/reduce.cpp` | `-O3` | 4 | 16 | 1 |
| `src/reduce.cpp` | `-O2 -march=native` | 2 | 64 | 0 |
| `src/reduce.cpp` | `-O3 -march=native` | 5 | 32, 64 | 1 |
| `src/reduce.cpp` | `-O2 -march=x86-64-v3` | 2 | 32 | 0 |

The head's loops vectorised under the project's flags, by the block function each is in: `boundary_cells`, `upwind_cells`, `sweep_cells`, `residual_cells`, `product_cells`, `start_cells`, `restart_cells`, `update_cells`, `direction_cells`, `setup_cells`. Every block function of the file is among them; the limited QUICK flux is not a block (section 3.4).
<!-- tables89 vectorised end -->

## 6. The net: the tests, and what was added to them

**Prediction (a)'s claim about the tests.** `git diff 3c8d1ec..6a13cfc --numstat -- tests/test_core_parity.py core/tests/`
shows no line removed from `tests/test_core_parity.py`, `core/tests/test_scalar.cpp` or
`core/tests/test_grid.cpp`, and none from `core/tests/test_reduce.cpp` (three `#include` lines and
three tests added); `core/tests/test_lanes.cpp` is new. `docs/reports/probe87/probe87.py` and
`docs/reports/probe88/probe88.py`, whose replays the probe calls, are unchanged.

<!-- tables89 tests begin -->
### T9. The tests at 6a13cfc, under the project's flags

| Tests | Cases | Passed |
|---|---|---|
| GoogleTest cases that 3c8d1ec has | 68 | 68 |
| GoogleTest cases added by this pass | 11 | 11 |
| Parity cases that 3c8d1ec has | 1404 | 1404 |
| Parity cases added by this pass | 1681 | 1681 |
| Captured flux calls of prompt 87 replayed, bit for bit | 120 | 120 |
| Captured Jacobi solves of prompt 88 replayed, every output bit for bit | 100 | 100 |
| Captured CG solves of prompt 88 at the reference's count | 40 | 40 |
| the same against the reference with the core's dot products, bit for bit | 40 | 40 |

The captured CG solves' largest difference from the reference, over its largest magnitude: 1.86e-16.

The cases added:

| Added | Cases |
|---|---|
| GoogleTest `BlockDot` | 3 |
| GoogleTest `Lanes` | 8 |
| `test_the_row_widths_leave_every_number_of_cells_after_the_blocks` | 1 |
| `test_implicit_jacobi_is_bit_for_bit_at_every_row_length` | 336 |
| `test_implicit_jacobi_with_a_bound_is_bit_for_bit_at_every_row_length` | 168 |
| `test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length` | 336 |
| `test_implicit_system_is_the_references_set_up_at_every_row_length` | 168 |
| `test_advective_flux_is_bit_for_bit_at_every_row_length` | 672 |
<!-- tables89 tests end -->

**Why tests were added before the first kernel changed.** Every parity mesh of prompts 87 and 88
is 40, 80, 160 or 320 cells wide, all multiples of eight. A walk in blocks of eight would leave
the same six cells on every one of them, and no mesh has a row shorter than a block, so a block
that started or ended one cell out of place could pass every existing case. The second commit of
this pass (0ae13a3, before any kernel changed) adds the bit-for-bit comparisons on grids of every
width from 1 to 19 cells, 26 and 33, and of 1, 2, 3 and 5 rows, built from arrays, not from a
configuration: Jacobi with scalar and per-cell steps and held cells, Jacobi with a bound that
binds and with a zero bound, CG with the core's dot products, the set-up, and the flux in both
directions with QUICK and upwind and with a random vent mask. A test checks that the widths leave
every count of cells from zero to seven after the blocks for each of the three kinds of walk, and
reach two blocks. The cases pass on the core as it was (T9's count for 3c8d1ec is of the old
cases; `check_base.json` holds the new ones passing there).

**Planted defects.** `probe89.py plants` builds each defect of `variants89.py` in a copy of the
head and runs CTest and the parity module on it.

<!-- tables89 plants begin -->
### T10. The planted defects: which tests catch each

| Defect | What it changes | GoogleTest (CTest) | Parity (pytest) |
|---|---|---|---|
| `control` | the unchanged source | 100% tests passed, 0 tests failed out of 79 | 3085 passed in 21.11s |
| `walk_one_block_too_many` | a walk takes a whole block where fewer cells than a block are left | 99% tests passed, 1 tests failed out of 79 [failed: Lanes.AWalkVisitsEveryCellOnceInOrder] | the run ended without a summary, exit code -6: a crash |
| `walk_skips_the_cells_left` | a walk stops after its whole blocks | 77% tests passed, 18 tests failed out of 79 [failed: Lanes.AWalkVisitsEveryCellOnceInOrder, AdvectiveFlux.AUniformFieldGivesTheFluxTimesItsValue, AdvectiveFlux.ALineOfCellsIsEachRowOfAFieldConstantAcrossIt, ImplicitStep.TheHandSystemsLeftoverIsTheChangeTheSinkDoesNotAccountFor, ImplicitStep.AUniformFieldWithNoSinkIsUnchanged, ImplicitStep.ANonNegativeStarGivesNonNegativeJacobiIterates, ImplicitStep.HeldAndSolidCellsKeepTheirValues, ImplicitStep.ConjugateGradientsReachTheDenseSolution, ImplicitSystem.TheDiagonalIsWestEastSouthNorthThenTheSink, ImplicitSystem.ANanIsSomethingToSolveAndMakesTheStopNan, Jacobi.OneSweepFromCStarIsTheHandSystemsFirstIterate, Jacobi.ConvergesToTheHandSystemsSolution, Jacobi.TheStopHoldsWithEqualityAndNotBelowIt, Jacobi.ABoundSweepsOnUntilTheLeftoverIsInsideIt, ConjugateGradient.OneIterationIsTheHandSystemsClosedForm, ConjugateGradient.ConvergesToTheHandSystemsSolutionInTwoOrThreeIterations, ConjugateGradient.TheStopHoldsWithEqualityAtTheStart, ConjugateGradient.ABoundRestartsFromTheTrueResidualUntilTheLeftoverIsInsideIt] | 2105 failed, 980 passed in 31.15s [failed: test_advective_flux_equals_the_reference_bit_for_bit 78, test_advective_flux_is_bit_for_bit_at_every_row_length 463, test_implicit_cg_fallback_is_the_jacobi_solve_bit_for_bit 10, test_implicit_cg_iterations_under_a_fallback_are_the_cgs_own 5, test_implicit_cg_meets_prompt_88s_rule 66, test_implicit_cg_positivity_is_read_in_the_cells_it_solves_for_only 5, test_implicit_cg_with_a_leftover_bound_and_the_cores_dot_products_is_bit_for_bit 45, test_implicit_cg_with_a_leftover_bound_meets_prompt_88s_rule 31, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit 90, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 335, test_implicit_jacobi_equals_the_reference_bit_for_bit 320, test_implicit_jacobi_is_bit_for_bit_at_every_row_length 335, test_implicit_jacobi_with_a_bound_is_bit_for_bit_at_every_row_length 168, test_implicit_stop_reads_b_in_every_cell 2, test_implicit_system_is_the_references_set_up_at_every_row_length 152] |
| `stencil_end_as_interior` | a row's last cell goes through the blocks, reading the cell beyond the row | 100% tests passed, 0 tests failed out of 79 | 645 failed, 2440 passed in 22.53s [failed: test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 218, test_implicit_jacobi_equals_the_reference_bit_for_bit 80, test_implicit_jacobi_is_bit_for_bit_at_every_row_length 233, test_implicit_jacobi_with_a_bound_is_bit_for_bit_at_every_row_length 114] |
| `end_cell_as_a_cell_between` | a row's last cell reads its east neighbour as the next element in memory, not the zero | 100% tests passed, 0 tests failed out of 79 | 656 failed, 2429 passed in 22.50s [failed: test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 225, test_implicit_jacobi_equals_the_reference_bit_for_bit 80, test_implicit_jacobi_is_bit_for_bit_at_every_row_length 234, test_implicit_jacobi_with_a_bound_is_bit_for_bit_at_every_row_length 117] |
| `padding_east_as_cell` | the zero read beyond a row's east end taken as the cell's own value | 100% tests passed, 0 tests failed out of 79 | 847 failed, 2238 passed in 23.58s [failed: test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 306, test_implicit_jacobi_equals_the_reference_bit_for_bit 80, test_implicit_jacobi_is_bit_for_bit_at_every_row_length 304, test_implicit_jacobi_with_a_bound_is_bit_for_bit_at_every_row_length 157] |
| `padding_west_as_cell` | the zero read beyond a row's west end taken as the cell's own value | 100% tests passed, 0 tests failed out of 79 | 852 failed, 2233 passed in 23.14s [failed: test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 306, test_implicit_jacobi_equals_the_reference_bit_for_bit 80, test_implicit_jacobi_is_bit_for_bit_at_every_row_length 303, test_implicit_jacobi_with_a_bound_is_bit_for_bit_at_every_row_length 163] |
| `padding_north_as_cell` | the zeros read above the last row taken as the row's own values | 100% tests passed, 0 tests failed out of 79 | 908 failed, 2177 passed in 23.78s [failed: test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 334, test_implicit_jacobi_equals_the_reference_bit_for_bit 80, test_implicit_jacobi_is_bit_for_bit_at_every_row_length 328, test_implicit_jacobi_with_a_bound_is_bit_for_bit_at_every_row_length 166] |
| `padding_south_as_cell` | the zeros read below the first row taken as the row's own values | 100% tests passed, 0 tests failed out of 79 | 908 failed, 2177 passed in 23.50s [failed: test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 332, test_implicit_jacobi_equals_the_reference_bit_for_bit 80, test_implicit_jacobi_is_bit_for_bit_at_every_row_length 329, test_implicit_jacobi_with_a_bound_is_bit_for_bit_at_every_row_length 167] |
| `sweep_update_masked_as_residual` | the Jacobi update zeroed in held cells as well as SOLID ones | 99% tests passed, 1 tests failed out of 79 [failed: ImplicitStep.HeldAndSolidCellsKeepTheirValues] | 328 failed, 2757 passed in 20.43s [failed: test_implicit_cg_fallback_is_the_jacobi_solve_bit_for_bit 10, test_implicit_cg_iterations_under_a_fallback_are_the_cgs_own 5, test_implicit_cg_meets_prompt_88s_rule 3, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit 3, test_implicit_jacobi_equals_the_reference_bit_for_bit 160, test_implicit_jacobi_is_bit_for_bit_at_every_row_length 147] |
| `residual_unmasked` | the residual not zeroed in the cells that are not unknowns | 97% tests passed, 2 tests failed out of 79 [failed: ImplicitStep.TheNeighbourSumIsEastWestNorthSouth, ImplicitStep.HeldAndSolidCellsKeepTheirValues] | 1039 failed, 2046 passed in 25.94s [failed: test_implicit_cg_fallback_is_the_jacobi_solve_bit_for_bit 10, test_implicit_cg_iterations_under_a_fallback_are_the_cgs_own 5, test_implicit_cg_meets_prompt_88s_rule 4, test_implicit_cg_positivity_is_read_in_the_cells_it_solves_for_only 5, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit 45, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 310, test_implicit_jacobi_equals_the_reference_bit_for_bit 192, test_implicit_jacobi_is_bit_for_bit_at_every_row_length 311, test_implicit_jacobi_with_a_bound_is_bit_for_bit_at_every_row_length 155, test_implicit_stop_reads_b_in_every_cell 2] |
| `product_unmasked` | the CG's product not zeroed in the cells that are not unknowns | 100% tests passed, 0 tests failed out of 79 | 364 failed, 2721 passed in 22.33s [failed: test_implicit_cg_iterations_under_a_fallback_are_the_cgs_own 5, test_implicit_cg_positivity_is_read_in_the_cells_it_solves_for_only 5, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit 45, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 309] |
| `neighbour_order` | the neighbour sum's terms along y taken south before north | 99% tests passed, 1 tests failed out of 79 [failed: ImplicitStep.TheNeighbourSumIsEastWestNorthSouth] | 789 failed, 2296 passed in 24.02s [failed: test_implicit_cg_fallback_is_the_jacobi_solve_bit_for_bit 10, test_implicit_cg_iterations_under_a_fallback_are_the_cgs_own 5, test_implicit_cg_positivity_is_read_in_the_cells_it_solves_for_only 5, test_implicit_cg_with_a_leftover_bound_and_the_cores_dot_products_is_bit_for_bit 40, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit 90, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 124, test_implicit_jacobi_equals_the_reference_bit_for_bit 320, test_implicit_jacobi_is_bit_for_bit_at_every_row_length 131, test_implicit_jacobi_with_a_bound_is_bit_for_bit_at_every_row_length 62, test_implicit_stop_reads_b_in_every_cell 2] |
| `diagonal_order` | the diagonal's terms along y taken north before south | 99% tests passed, 1 tests failed out of 79 [failed: ImplicitSystem.TheDiagonalIsWestEastSouthNorthThenTheSink] | 1022 failed, 2063 passed in 25.00s [failed: test_implicit_cg_fallback_is_the_jacobi_solve_bit_for_bit 6, test_implicit_cg_iterations_under_a_fallback_are_the_cgs_own 5, test_implicit_cg_positivity_is_read_in_the_cells_it_solves_for_only 3, test_implicit_cg_with_a_leftover_bound_and_the_cores_dot_products_is_bit_for_bit 33, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit 74, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 181, test_implicit_jacobi_equals_the_reference_bit_for_bit 320, test_implicit_jacobi_is_bit_for_bit_at_every_row_length 179, test_implicit_jacobi_with_a_bound_is_bit_for_bit_at_every_row_length 96, test_implicit_stop_reads_b_in_every_cell 2, test_implicit_system_equals_the_references_set_up 10, test_implicit_system_is_the_references_set_up_at_every_row_length 113] |
| `stop_from_one_lane` | the stop's largest |b| taken from the first of the eight running maxima | 99% tests passed, 1 tests failed out of 79 [failed: ImplicitStep.ConjugateGradientsReachTheDenseSolution] | 578 failed, 2507 passed in 22.36s [failed: test_implicit_cg_fallback_is_the_jacobi_solve_bit_for_bit 1, test_implicit_cg_iterations_under_a_fallback_are_the_cgs_own 4, test_implicit_cg_positivity_is_read_in_the_cells_it_solves_for_only 1, test_implicit_cg_with_a_leftover_bound_and_the_cores_dot_products_is_bit_for_bit 11, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit 28, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 93, test_implicit_jacobi_equals_the_reference_bit_for_bit 124, test_implicit_jacobi_is_bit_for_bit_at_every_row_length 147, test_implicit_jacobi_with_a_bound_is_bit_for_bit_at_every_row_length 76, test_implicit_system_equals_the_references_set_up 10, test_implicit_system_is_the_references_set_up_at_every_row_length 83] |
| `stop_nan_dropped` | a NaN |b| left out of the stop | 99% tests passed, 1 tests failed out of 79 [failed: ImplicitSystem.ANanIsSomethingToSolveAndMakesTheStopNan] | 3085 passed in 18.93s |
| `outside_stop_strict` | the max-norm stop met strictly below the stop, not at it | 94% tests passed, 5 tests failed out of 79 [failed: ImplicitStep.TheNeighbourSumIsEastWestNorthSouth, Jacobi.TheStopHoldsWithEqualityAndNotBelowIt, Jacobi.ABoundSweepsOnUntilTheLeftoverIsInsideIt, ConjugateGradient.TheStopHoldsWithEqualityAtTheStart, ConjugateGradient.ABoundRestartsFromTheTrueResidualUntilTheLeftoverIsInsideIt] | 23 failed, 3062 passed in 19.37s [failed: test_implicit_a_zero_field_meets_the_stop_and_a_zero_bound_with_equality 20, test_implicit_jacobi_is_bit_for_bit_at_every_row_length 3] |
| `jacobi_last_copy_dropped` | the Jacobi solve's field left where its last sweep wrote it | 96% tests passed, 3 tests failed out of 79 [failed: ImplicitStep.TheHandSystemsLeftoverIsTheChangeTheSinkDoesNotAccountFor, ImplicitStep.AUniformFieldWithNoSinkIsUnchanged, Jacobi.TheStopHoldsWithEqualityAndNotBelowIt] | 532 failed, 2553 passed in 22.46s [failed: test_implicit_cg_fallback_is_the_jacobi_solve_bit_for_bit 1, test_implicit_cg_iterations_under_a_fallback_are_the_cgs_own 3, test_implicit_cg_meets_prompt_88s_rule 5, test_implicit_cg_with_a_leftover_bound_and_the_cores_dot_products_is_bit_for_bit 16, test_implicit_cg_with_a_leftover_bound_meets_prompt_88s_rule 6, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit 6, test_implicit_jacobi_equals_the_reference_bit_for_bit 215, test_implicit_jacobi_is_bit_for_bit_at_every_row_length 174, test_implicit_jacobi_with_a_bound_is_bit_for_bit_at_every_row_length 106] |
| `cg_dot_of_the_wrong_row` | the CG's r.z taken as z.z | 97% tests passed, 2 tests failed out of 79 [failed: ImplicitStep.ConjugateGradientsReachTheDenseSolution, ConjugateGradient.ConvergesToTheHandSystemsSolutionInTwoOrThreeIterations] | 452 failed, 2633 passed in 26.56s [failed: test_implicit_cg_iterations_under_a_fallback_are_the_cgs_own 5, test_implicit_cg_positivity_is_read_in_the_cells_it_solves_for_only 5, test_implicit_cg_with_a_leftover_bound_and_the_cores_dot_products_is_bit_for_bit 25, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit 90, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 327] |
| `cg_z_from_the_old_residual` | the CG's z formed from the residual before its update | 97% tests passed, 2 tests failed out of 79 [failed: ImplicitStep.ConjugateGradientsReachTheDenseSolution, ConjugateGradient.ConvergesToTheHandSystemsSolutionInTwoOrThreeIterations] | 452 failed, 2633 passed in 24.66s [failed: test_implicit_cg_iterations_under_a_fallback_are_the_cgs_own 5, test_implicit_cg_positivity_is_read_in_the_cells_it_solves_for_only 5, test_implicit_cg_with_a_leftover_bound_and_the_cores_dot_products_is_bit_for_bit 25, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit 90, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 327] |
| `cg_direction_fused` | the direction updated as fma(p, beta, z), the contraction -ffp-contract=off forbids | 100% tests passed, 0 tests failed out of 79 | 208 failed, 2877 passed in 20.52s [failed: test_implicit_cg_positivity_is_read_in_the_cells_it_solves_for_only 3, test_implicit_cg_with_a_leftover_bound_and_the_cores_dot_products_is_bit_for_bit 16, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit 65, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 124] |
| `block_dot_left_to_right` | a row's dot product summed left to right, not in pairwise order | 99% tests passed, 1 tests failed out of 79 [failed: BlockDot.TheBlocksAddedInOrderAreBlockedDot] | 282 failed, 2803 passed in 20.42s [failed: test_implicit_cg_positivity_is_read_in_the_cells_it_solves_for_only 4, test_implicit_cg_with_a_leftover_bound_and_the_cores_dot_products_is_bit_for_bit 23, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit 86, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit_at_every_row_length 169] |
| `flux_end_node_ignores_inflow` | a line's low boundary node read as the cell's value whatever the flux | 100% tests passed, 0 tests failed out of 79 | 194 failed, 2891 passed in 19.65s [failed: test_advective_flux_equals_the_reference_bit_for_bit 124, test_advective_flux_is_bit_for_bit_at_every_row_length 70] |
| `flux_weights_of_the_first_face` | every face of a row along x given the row's first face's weights | 100% tests passed, 0 tests failed out of 79 | 94 failed, 2991 passed in 20.90s [failed: test_advective_flux_equals_the_reference_bit_for_bit 52, test_advective_flux_is_bit_for_bit_at_every_row_length 42] |
| `flux_far_solid_dropped` | a far node inside a SOLID cell read as it is | 97% tests passed, 2 tests failed out of 79 [failed: AdvectiveFlux.AFarNodeInASolidCellIsReadAsTheUpstreamValue, AdvectiveFlux.TheFarSolidRuleReadsQuickAsTheUpstreamValueToo] | 239 failed, 2846 passed in 20.30s [failed: test_advective_flux_equals_the_reference_bit_for_bit 116, test_advective_flux_is_bit_for_bit_at_every_row_length 123] |
| `flux_high_boundary_sign` | the high boundary node takes the inflow value where the flux leaves | 99% tests passed, 1 tests failed out of 79 [failed: AdvectiveFlux.BoundaryFacesTakeTheInflowOnlyWhereTheFluxEnters] | 968 failed, 2117 passed in 24.07s [failed: test_advective_flux_equals_the_reference_bit_for_bit 312, test_advective_flux_is_bit_for_bit_at_every_row_length 656] |
| `flux_upwind_downstream` | the upwind value taken from the downstream cell | 100% tests passed, 0 tests failed out of 79 | 414 failed, 2671 passed in 21.58s [failed: test_advective_flux_equals_the_reference_bit_for_bit 130, test_advective_flux_is_bit_for_bit_at_every_row_length 284] |
| `flux_vents_dropped` | the vents' second pass not made | 99% tests passed, 1 tests failed out of 79 [failed: AdvectiveFlux.AVentTakesTheInflowEnteringAndTheRoomCellLeaving] | 342 failed, 2743 passed in 23.12s [failed: test_advective_flux_equals_the_reference_bit_for_bit 119, test_advective_flux_is_bit_for_bit_at_every_row_length 223] |
| `flux_last_row_boundary_dropped` | the high boundary faces of the lines along y not written | 99% tests passed, 1 tests failed out of 79 [failed: AdvectiveFlux.AUniformFieldGivesTheFluxTimesItsValue] | 493 failed, 2592 passed in 23.86s [failed: test_advective_flux_accepts_consistent_arguments 1, test_advective_flux_equals_the_reference_bit_for_bit 156, test_advective_flux_is_bit_for_bit_at_every_row_length 336] |
| `select_swapped` | lanes::select returns the other value | 97% tests passed, 2 tests failed out of 79 [failed: Lanes.SelectPassesTheChosenValuesBits, AdvectiveFlux.BoundaryFacesTakeTheInflowOnlyWhereTheFluxEnters] | 983 failed, 2102 passed in 24.33s [failed: test_advective_flux_equals_the_reference_bit_for_bit 312, test_advective_flux_is_bit_for_bit_at_every_row_length 671] |
| `keep_by_multiplication` | lanes::keep multiplies by one or zero where it should pass or drop bits | 99% tests passed, 1 tests failed out of 79 [failed: Lanes.KeepGivesPositiveZeroWhereTheMaskIsNotSet] | 121 failed, 2964 passed in 21.43s [failed: test_implicit_cg_meets_prompt_88s_rule 1, test_implicit_cg_with_the_cores_dot_products_is_bit_for_bit 24, test_implicit_jacobi_equals_the_reference_bit_for_bit 96] |
| `control_end` | the unchanged source, after every plant | 100% tests passed, 0 tests failed out of 79 | 3085 passed in 19.45s |
<!-- tables89 plants end -->

## 7. The arithmetic changes studied, not built

Each of these changes a result, so none was built: each would start as a variant of
`src/scalar_scheme.py`, be measured against the noise band (ADR-017 H) and go to Alex. The
numbers they are weighed with: the developed frame's solve takes eight sweeps (T4), the first
steps of a release three or four (prompt 88, T4 there); the diagonal's diffusion share
`kappa = dt sum_f G_f / V` is at most 0.42 on 320x120 (ADR-016), so a Jacobi sweep cuts the error
by at least `(1 + kappa) / kappa`, about 3.4; and a sweep, the set-up and a CG iteration cost what
T4 gives. The reason the transport uses Jacobi (`src/scalar_scheme.py`, module docstring): every
iterate is `(b + sum_f G_f C_N) / a_P`, a combination of `b` and the neighbours with
non-negative weights, so a non-negative `C*` gives a non-negative field at any sweep count, over
a concentration that spans many decades down to zero.

| Change | What it would gain | What it would cost | The non-negative guarantee |
|---|---|---|---|
| Gauss-Seidel in the field's order | The error's factor per sweep is Jacobi's squared for this matrix, so about half the sweeps: four or five where there are eight | A cell reads its west and south neighbours' new values, so the sweep is a chain along the row: no vector instruction, and one cell's division and products (about 20 cycles of latency, an estimate from the operations) before the next can start, where the block sweep takes about seven cycles a cell (T4 over T2's cycle). Half the sweeps at about three times the cost each is a loss on this processor, and on the GPU the chain is serial. The residual is no longer a by-product of the update | Kept: each update is still a non-negative combination |
| Red-black ordering | The same factor per sweep as Gauss-Seidel, each colour's half-sweep independent of itself, so it vectorises and runs on the GPU. At best the eight sweeps become four or five, which is the sweeps' share of the solve, not the set-up's | The cells of one colour are every second cell: either the loads stride by two (half a vector's worth per load) or the fields are stored by colour, which every other kernel would then pay for. The fields and the sweep counts change, and so does the leftover that ADR-016's stop reads | Kept, for the same reason |
| Over-relaxation (SOR) | With a Jacobi factor near 0.3 the best factor is close to 1 and the gain over Gauss-Seidel small: the system is already well conditioned | As Gauss-Seidel or red-black, and a parameter to choose | Lost: with a factor above 1 the update is `(1 - omega) C + omega (...)`, and the first weight is negative |
| CG for the transport solve | About the square root of Jacobi's count: perhaps four or five iterations for eight sweeps | A CG iteration costs about twice a sweep (T4), so there is nothing to gain at these counts; its two dot products are reductions the stop reads, which is where the counts of prompt 88's tight cases parted from the reference's | Lost: a truncated CG iterate has no sign guarantee, which is why the k-epsilon solve carries a fallback to Jacobi (prompt 51) |
| Multigrid for the transport solve | Nothing at these counts: a V-cycle costs several sweeps, and eight sweeps already converge | The hierarchy and its transfers | Lost: a coarse-grid correction can be negative |
| A fixed sweep count, or the stop tested every few sweeps | On this processor almost nothing: the stop's test is three of a sweep's operations. On the GPU it removes a result handed back to the host every sweep (`docs/reports/profile84.md`, T7's scalars) | The solve no longer adapts to the field: a count that is right for the developed frame is twice what the first steps need. ADR-016's bound still needs the leftover of the field returned | Kept: the iterates are Jacobi's |
| The leftover summed in blocks, a row at a time, in place of NumPy's pairwise order | The sum could then be taken while each row is formed, with no pass over the residual at a stop, and computed in parallel; on this processor an estimated few percent of a solve, one pass in about ten | ADR-017 G puts the leftover in the first kind, the sums a stop reads in the reference's order. In another order its last bits move, and the bound's test can flip where the leftover sits on the bound, which moves a sweep count | Not affected |
| One reciprocal of the clamp's divisor in place of its two divisions | The kept flux loop is bound by its divider (section 1): one division a face in place of two is the largest single saving left in the flux | `r` and `psi_quick` are each rounded twice where they were rounded once, so face values change in their last bits. Not a sweep method, and listed because the flux's floor points at it | Not affected: the clamp still bounds `psi` between 0 and 2 |

One change that would not alter a result was not tried either: two sweeps fused into one pass
over memory, each row of the second sweep formed as soon as the rows it reads are ready. It
halves the memory traffic of a pair of sweeps, and T2 shows the sweep is not bound by memory under
the project's flags, so there is little for it to save there.

## 8. The style, written down

How a kernel of the core is written, as this pass left `scalar`:

- **It walks a field by rows.** A row is a line along direction 0 and is contiguous in memory.
  Another direction enters as pointers to other rows: a stencil's neighbours above and below,
  or the two rows of cells a row of faces lies between. No inner loop strides, and nothing is
  transposed or copied to make a direction contiguous.
- **A row's arithmetic is a block function**, written once: a template on a cell count whose
  loop runs that many times, taking `lanes::cells<N>` first. `lanes::for_each_block` calls it on
  eight cells at a time and then on each cell left. Its inputs come in a struct of pointers at
  the row's first cell, received by value by the function that walks the row; its outputs are
  parameters qualified `CFDCORE_RESTRICT`; a helper it shares with another block is
  `CFDCORE_INLINE`.
- **Boundaries are not tested for in a block.** A row's end cells run the same block function on
  one cell, with operands that hold the boundary's value: a row of zeros for a neighbour beyond
  the row or the field (the reference's padding), a boundary node in place of a far cell. A
  kernel whose first and last rows differ gets operands or an instantiation for them.
- **A block's pointers are few.** A cell's neighbours along the row are read either side of the
  cell's own pointer, not through a pointer each: a block with more pointers than the processor
  has registers free reloads them from the stack every iteration (T6, `neighbours_by_pointer`).
- **Masks are data.** A condition on a cell is a `lanes::mask`, a choice `lanes::select` or
  `lanes::keep`: bit operations, which pass a value's bits or do not and round nothing. A SOLID
  or held cell is computed and masked. `bool` masks are expanded to 64 bits once per solve or per
  grid.
- **Reductions.** A flag a stop reads is the OR of the cells' masks. A maximum is one running
  value per cell of a block, combined after the walk. A sum a stop or a budget reads keeps the
  reference's order through `reduce` (ADR-017 G); a dot product is `reduce`'s blocks, a row a
  block, taken as each row is formed.
- **Scratch** is allocated once on a layout and holds no state a caller must set. A kernel reads
  its input where the caller holds it and writes its output where it goes.
- **Bit for bit.** Each cell's operations are the reference's in the reference's order, one
  rounding each. Passes may be fused where each element's operations keep their order: a value
  formed beside another and discarded if a stop holds, a row's reduction taken with the row.
- **A kernel that is mostly choices is the exception**, under the project's flags: the limited
  QUICK face value is one loop a face with a branch per choice (section 3.4). Both forms are
  measured before one is chosen, and the kernel's comment says which was kept and why.
- **Errors**, as round 1 of prompt 88 left them: an exception (`std::invalid_argument`) for a
  caller's programming error that no run can meet, such as a block of zero or two arrays of
  different lengths (`reduce::blocked_dot`, `reduce::block_dot`); a status for a condition a run
  can meet, which the binding turns into the exception the reference raises
  (`implicit_status`).
- **After a change**, the compiler's vectorisation report and the parity tests: a block the
  compiler left scalar is a defect of the change (`probe89.py vecinfo`), and a moved bit is too.

**Proposed wording for CLAUDE.md**, a subsection after "Style" under "C++ Coding Standards", for
Alex to apply:

```markdown
### Kernels

A kernel is the arithmetic of one pass over a field. Its form was fixed by measurement
(`docs/reports/phase6_scalar_perf.md`); `core/include/cfdcore/lanes.hpp` holds its parts.

- Walk a field by its rows, the lines along direction 0. Another direction enters as pointers
  to other rows, never as a stride in the inner loop or a transposed copy.
- Write a row's arithmetic once, as a block function: a template on a cell count N whose loop
  runs N times, taking `lanes::cells<N>` first, called through `lanes::for_each_block`. Inputs
  arrive in a struct of pointers that the row's walker receives by value; outputs are parameters
  qualified `CFDCORE_RESTRICT`; a helper a block calls is `CFDCORE_INLINE`.
- No test of position and no branch on a cell's data inside a block. A row's end cells and a
  field's edge rows run the same block function on operands that hold the boundary's value. A
  condition is a `lanes::mask` and a choice is `lanes::select` or `lanes::keep`; a SOLID or held
  cell is computed and masked.
- Keep a block's pointers few: read a cell's neighbours along the row either side of the cell's
  own pointer, not through a pointer each.
- A flag a stop reads is the OR of the cells' masks; a maximum is kept per cell of a block and
  combined after the walk; a sum a stop or a budget reads goes through `reduce` in the
  reference's order.
- A kernel that is mostly choices may be a loop with branches, one element at a time, if it
  measures faster under the project's flags. Measure both forms and say in the kernel's comment
  which was kept and why.
- Each element's operations and their order are the reference's. Passes are fused only where
  each element's operations keep their order.
- No copy of a field to pad it or to hand it on, and no allocation in a step: scratch is
  allocated once on a layout.
- An exception for a caller's programming error that no run can meet; a status for a condition
  a run can meet, which the binding turns into the reference's exception.
- After changing a kernel, read the compiler's vectorisation report and run the parity tests.
```

## Findings

- **F1, the loop style is fixed by what GCC's `-O2` will vectorise, and that is narrow.** A loop
  of constant trip count, outputs qualified `restrict` as parameters, inputs in a struct held by
  value, shared helpers forced inline, masks of 64 bits. Each of those is a measured condition:
  without any one of them the block stays scalar (section 2; T6's `helpers_not_forced_inline` and
  `row_loops_not_blocks` are two of them removed from the kept code). The style is therefore
  written to GCC 13.3's vectoriser. CI builds with GCC 15, which this pass did not have: the
  parity tests there say whether a bit moved, and nothing there says whether a block was left
  scalar (limits).
- **F2, under the project's flags the limited QUICK flux is fastest as a scalar loop with
  branches, not as a block.** The block form computes the same values and is the slower under
  `-O2` and the faster by a wide margin under `-march=native` and `-march=x86-64-v3` (T6, T7).
  The kept form is the one that is faster under the flags CLAUDE.md fixes, as the prompt's rule
  asks; the block form is in `variants89.py`, with its checks passing under every flag set, so a
  flag change can take it without new work. The cause is the select: three instructions under
  SSE2, one with AVX. Any kernel that is mostly choices will meet the same trade: the momentum
  predictor's QUICK face values (step 4) are the next.
- **F3, prediction (e) fails on `-march=native`, in both directions, and holds for `-O3`.**
  `-O3` moves no kernel by more than about a tenth (T7): it is not worth a change. Under
  `-march=native` the block kernels (the sweep, the set-up, the CG) are a sixth to three tenths
  faster, by the wider vector, and the kept flux loop is a tenth to three tenths slower: it is
  scalar code and gains nothing from a wider vector. The flux's block form is what gains there
  (T7's second table). This machine's AVX-512 is no faster than its AVX2 on these kernels (T2;
  T7's last column against its third).
- **F4, a recommendation on the flags, for Alex to decide.** Keep `-O2 -ffp-contract=off`, and
  consider adding `-march=x86-64-v3`. It is AVX2 by a name every machine reads the same, where
  `-march=native` makes the build depend on the processor it was built on; GitHub's runners and
  this machine both have it; `-ffp-contract=off` keeps its fused multiply-add out of every
  expression, and the parity module, CTest and the replays pass under it (T7). With it the flux
  should take its block form (T7's second table), which `variants89.py` holds ready; without
  that the flag makes the flux slightly slower. What it costs: a module built with it does not
  run on an x86-64 processor without AVX2 (older than about 2015), and the two-double floor of
  T2 stops being the one to read results against.
- **F5, the machine's state moves a timing by more than most changes here.** Prompts 87 and 88
  timed the core on this machine at between two and three times what this pass's "before" takes
  for the same code and inputs (88's T6b has the sweep at 0.26 to 0.28 ms and the set-up with one
  residual at 0.48 to 0.50 ms; T4's "before" has them at 0.10 and 0.18 ms). Within this pass the
  three sessions taken at the earlier head, while other work ran on the machine, timed the same
  `base` build's sweep at 113 to 144 us (`results/builder89/at_62ad9ce/`) where the final three
  time it at 99 to 103 (T4). So prediction (b)'s 0.07 ms and (c)'s 0.15 ms, which were set
  against 88's numbers, are scored here as written on a machine in a faster state than 88's,
  and that is why (b) holds at about twice the speed of the core before the pass and not at the
  four times of its parenthesis. A ratio taken within one session is the number to carry
  forward; T1 gives what else ran in each (a game ran through all three), and the probe records
  the load and a calibration with every timing.
- **F6, the existing parity meshes could not see a misplaced block.** All are multiples of eight
  cells wide (section 6). The row-length cases close that, and T10's walk and row-end plants fail
  them.
- **F7, the sweep is bound by arithmetic, not memory, under the project's flags** (T2): fusing
  passes over memory buys little there. It bought the CG two of its five walks, because a dot
  product's pass is nearly all memory traffic.
- **F8, a refactor that changed no value cost a quarter of the sweep, and only the per-commit
  timing showed it.** The builder replaced an index that wrapped below zero by a signed offset
  (62ad9ce) after timing the kernels for the last time by hand, and the first three sessions then
  showed the head's sweep slower than the same sweep in the builds of two earlier commits, in
  every session (`results/builder89/at_62ad9ce/`). The cause was one more pointer than the block
  had registers for (section 3.1). A kernel's speed here depends on details the source does not
  show, so a kernel is timed again after any change to it, however small, against a build of
  the commit before: T5's builds per commit are what made the comparison possible.
- **F9, `std::experimental::simd` is as fast as the block style and does not depend on the
  vectoriser.** T3 has the two level under every flag set, `simd` ahead under
  `-march=x86-64-v3`. F1's conditions do not apply to it, and it would have been immune to F8.
  The pass built the block style (section 2 gives why, and that the timings it chose on were not
  these). Whether step 3 copies the block style or the core moves to the library form is a
  decision for Alex: the kernels' bodies would move one to one, and the parity tests are the
  net either way.

## Deviations and limits

- **Tests were added, none edited** (section 6): the row-length parity cases, `test_lanes.cpp` and
  three `BlockDot` cases. Prediction (a)'s "unchanged" is scored on the cases 3c8d1ec has.
- **`implicit_scratch` gained three members** (`free`, `fluid`, `zeros`), `grid` gained
  `solid_flags()`, `reduce` gained `block_dot`, and `lanes.hpp` is a new header with no source
  file of its own: it is templates and inline functions. No signature a test or the binding calls
  changed, and `core/python/bindings.cpp` is untouched.
- **`jacobi` now writes iterates into `out` while it runs**, and `advective_flux`'s and the
  solve's kernels are compiled to assume their output overlaps no input. The binding already
  refuses an output that shares memory with an input; a C++ caller must not pass one. The header
  says so for both.
- **After a call with nothing to solve the scratch's `a_P` and `b` hold what the one pass wrote**,
  where prompt 88 left the diagonal's extra terms there. No caller reads them after that return.
- **The masks are expanded once per solve**, from the `bool` arrays the caller hands in, which is
  a pass over them per solve that the march (step 3) can avoid by holding its masks as 64 bits.
  Its cost was not timed apart.
- **The flux's floor is an estimate** from its divisions (section 1); no ideal flux kernel was
  written with intrinsics. The sweep's is measured.
- **Timings are of one machine in the states T1 records**, with a game running through all three
  sessions (F5). The scores are run 1's by the committed rule; runs 2 and 3 are beside them.
- **Compiled with GCC 13.3 only**, with warnings as errors (`scripts/build_core.sh`,
  `results/builder89/build_core.log`). CI's core job uses GCC 15; its result on the pull request
  is the check of that compiler.
- **`row_loops_not_blocks` of T6 fails one GoogleTest case by construction**: it removes the whole
  blocks from the walker, and the case it fails is the walker's own test that whole blocks come
  first. Its parity cases and replays pass.
- **The checks, the planted defects, the suites and the timing sessions were all run again at
  6a13cfc**, the last commit that changes the code. The commit after it, 0738777, changes one
  comment in `core/src/scalar.cpp` and nothing else under `core/`. The records of the two earlier
  heads are in `results/builder89/at_1b98cf7/` and `at_62ad9ce/`; no table reads them.
- **The planted defects are of this pass's structure** (T10). Prompt 88's plants no longer apply
  as text, the loops they edited being rewritten; the rules they planted against are planted
  again here where the new code has a line for them (the neighbour sum's and the diagonal's
  order, the masks, the padding, the stop).
- **Nothing under `src/` or `configs/` changed.**

<!-- tables89 suites begin -->
### T11. The whole suite at 6a13cfc

The two suites ran at the same time, one BLAS thread each, beside the planted defects, so the wall times are not those of a suite alone.

| Platform | Tests | Passed | Failed | Errors | Skipped | Wall time, s |
|---|---|---|---|---|---|---|
| Windows, `.venv`, the core not built | 3,102 | 3,101 | 0 | 0 | 1 | 1,215 |
| WSL2, the core built, `CFD_CORE_REQUIRED=1` | 6,186 | 6,186 | 0 | 0 | 0 | 537 |
<!-- tables89 suites end -->

## Files

- `core/include/cfdcore/lanes.hpp`: the loop style's parts.
- `core/include/cfdcore/scalar.hpp`, `core/src/scalar.cpp`: the four kernels.
- `core/include/cfdcore/reduce.hpp`, `core/src/reduce.cpp`: `block_dot`.
- `core/include/cfdcore/grid.hpp`: `solid_flags()`.
- `core/tests/test_lanes.cpp`, `core/tests/test_reduce.cpp`, `tests/test_core_parity.py`: the
  tests added.
- `docs/reports/probe89/`: `probe89.py` (builds, checks, timings, plants, the vectorisation
  report), `variants89.py` (the candidates and the planted defects), `floor89.cpp`,
  `styles89.cpp`, `tables89.py` and `tables89.md`.
- `docs/PROJECT_PLAN.md`, `docs/SYSTEM.md`, `docs/STATUS.md`: step 2c's status.
