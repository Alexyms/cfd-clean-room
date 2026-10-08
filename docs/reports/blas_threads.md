# BLAS Threads for the Pressure Solve

## Prediction

Prediction (orchestrator, 2026-10-07). Threaded `np.vdot` pays a fixed cost of about 0.33 ms per
call (the section 11 table) and saves per-element time only once the vector streams from memory
at more than one core's bandwidth. With one core reading about 10 GB/s, two float64 vectors of N
elements take about 1.6 N ns on one thread; break-even against 0.33 ms is a few hundred thousand
elements. Predicted: (a) at every length up to 100,000 one thread is faster; (b) the crossover, if
any, lies between 300,000 and 3,000,000 elements; (c) under the in-process limit, a 15,000-element
`vdot` costs within 20% of the `OPENBLAS_NUM_THREADS=1` figure (about 4.3 microseconds); (d) the
`val001_80x40` final faces under the limit hash identically to the step 2 baseline; (e) a 200x75
correction under the process's default environment, with the limit, has a median under 0.5 s and
within 20% of the 136 to 143 ms one-thread figures.

## 1. What was measured

**Date:** 2026-10-07. **Tree:** branch `fix/deferred-findings-cleanup` from main at 43d60ab. The
predictions above were committed before any probe ran (`docs: record the BLAS thread
predictions`).

**Machine.** AMD Ryzen AI 9 HX 370 (12 cores, 24 logical processors; L2 12 MB and L3 24 MB in
all, as Windows reports them), 64 GB, Windows 11. Python 3.13.3, NumPy 2.4.4. The BLAS NumPy
loads, as `threadpoolctl.threadpool_info()` reports it: `user_api` blas, `internal_api` openblas,
`prefix` libscipy_openblas, version 0.3.31.188.0, `threading_layer` pthreads, `architecture`
SkylakeX, `num_threads` 24. `threadpoolctl` 3.7.0 finds this library, so the in-process limit
reaches the BLAS that `np.vdot` calls.

**Probe.** `scripts/blas_thread_probe.py` times `np.vdot` on two float64 vectors of N elements,
the repeat count scaled from a 50-call pilot so each length runs about 3 s. Three conditions, each
in its own process: `default` (no thread variable set), `env1` (`OPENBLAS_NUM_THREADS=1`), and
`limit` (the default process under `threadpool_limits(limits=1, user_api="blas")`, the mechanism
`src/pressure.py` uses). The sequence ran twice with nothing else running. Raw output is
`results/builder39/probe_out.txt` (untracked); every figure below is the mean of the two runs,
with the spread between them, (max - min) / mean, in brackets.

## 2. The probe table

Microseconds per `np.vdot` call.

| Elements | Default threads | `OPENBLAS_NUM_THREADS=1` | In-process limit of 1 | Default / env1 | Limit / env1 |
|---:|---:|---:|---:|---:|---:|
| 3,200 (80x40) | 2.03 (5%) | 2.08 (5%) | 2.07 (6%) | 0.98 | 1.00 |
| 6,400 (80x80) | 3.31 (5%) | 3.26 (16%) | 3.04 (1%) | 1.02 | 0.93 |
| 9,600 | 3.79 (2%) | 3.77 (3%) | 3.95 (10%) | 1.00 | 1.05 |
| 12,800 (160x80) | 353.85 (11%) | 4.51 (1%) | 4.70 (6%) | 78.49 | 1.04 |
| 15,000 (200x75) | 362.69 (4%) | 5.22 (8%) | 5.29 (1%) | 69.47 | 1.01 |
| 30,000 | 369.65 (3%) | 7.25 (10%) | 8.88 (3%) | 51.00 | 1.23 |
| 100,000 | 362.97 (5%) | 36.66 (6%) | 34.96 (3%) | 9.90 | 0.95 |
| 300,000 | 395.54 (4%) | 105.87 (2%) | 106.66 (0%) | 3.74 | 1.01 |
| 1,000,000 | 501.24 (3%) | 367.08 (0%) | 372.27 (15%) | 1.37 | 1.01 |

The prompt's list stopped at 1,000,000, where one thread was still ahead. Two more runs of the same
script with `--lengths` locate the crossover:

| Elements | Default threads | `OPENBLAS_NUM_THREADS=1` | In-process limit of 1 | Default / env1 | Limit / env1 |
|---:|---:|---:|---:|---:|---:|
| 1,500,000 | 412.78 (1%) | 443.47 (1%) | 544.31 (29%) | 0.93 | 1.23 |
| 2,000,000 | 596.14 (3%) | 804.63 (3%) | 810.45 (6%) | 0.74 | 1.01 |
| 3,000,000 | 642.19 (0%) | 1087.19 (2%) | 1083.53 (6%) | 0.59 | 1.00 |
| 10,000,000 | 2702.88 (6%) | 4397.84 (0%) | 4497.96 (2%) | 0.61 | 1.02 |

Reading the tables:

- Up to 9,600 elements the three conditions agree within their run-to-run spread (OpenBLAS runs
  `ddot` on the calling thread below its threshold).
- From 12,800 elements the default threads cost about 0.35 to 0.37 ms per call, whatever the length
  up to 100,000, which is a fixed coordination cost: 350 us against 4.5 to 37 us of arithmetic.
  One thread is faster by a factor of 78 at 12,800, 69 at the product mesh's 15,000, and 10 at
  100,000.
- The cost per call under default threads grows slowly with length (353 to 501 us from 12,800 to
  1,000,000 elements) while one thread's grows linearly, so the lines cross between 1,000,000 and
  1,500,000 elements. Past that threads win by up to 1.7 times (0.59 at 3,000,000), which is what
  streaming the vectors from memory on several cores buys. The pressure solve's vectors are at most
  15,000 elements on the product mesh, 60 to 100 times short of the crossover.
- The in-process limit agrees with the environment variable within the spread at every length but
  two. At 30,000 elements it reads 1.23 times the variable's figure in both runs (7.0 and 7.6 us
  under the variable, 9.0 and 8.7 us under the limit). Why is not established; it is above the
  product mesh and below 100,000, where the two agree again, and at 15,000 they are within 1%. At
  1,500,000 the limit's own spread is 29%.

## 3. The setting

`PRESSURE_BLAS_THREADS = 1` in `src/pressure.py`, applied by `conjugate_gradient` around the CG
loop and nothing else, with `user_api="blas"`. The `ThreadpoolController` is built once at import,
since discovery is the expensive part: 940 us per `ThreadpoolController()` against 6.5 us (median;
mean 7.95) to enter and leave a limit from the existing controller, over 20,000 enters
(`results/builder39/overhead.txt`).

**The cost of the limit.** On `val001_80x40` (1,559 outer iterations, one correction each) the
pressure stage took 23.78 s and 25.05 s in the two repeats, a mean correction of 15.25 ms and
16.07 ms. Entering and leaving the limit, 7.95 us mean, is 5.2e-4 and 4.9e-4 of that: about 0.05%.
The prompt's stop condition was 5%.

**The test.** `TestConjugateGradient.test_blas_pool_is_limited_inside_the_solve_and_restored_after`
in `tests/test_pressure.py` raises the process's BLAS pool to two threads, records the pool's size
at every operator product, once through `conjugate_gradient` and once through
`PressureCorrector.correct`, and asserts that every record is `PRESSURE_BLAS_THREADS` and that the
pool is back at two afterwards. With the `with` block deleted from `conjugate_gradient` both
parametrizations fail (the pool reads 2 inside the solve). A companion test raises inside the
operator and asserts the pool is restored; with the block replaced by a bare `limit(...)` call that
is never entered, all three tests fail. A machine whose BLAS cannot run two threads skips them,
since nothing distinguishes the limit there. The thread count the process starts with does not
matter to the test.

## 4. Demonstrations

CI hardware is not this machine and criterion 4 is stated on this machine, so these are recorded
here and not run in CI.

**Criterion 4 under the default environment.** `results/builder38/criterion4_threads.py` loads the
three captured 200x75 systems (`results/builder37b/systems/`) and solves each to 1e-8 through
`src.pressure.conjugate_gradient` seven times, interleaved. It printed the three thread variables
as unset (`{'OPENBLAS_NUM_THREADS': None, 'OMP_NUM_THREADS': None, 'MKL_NUM_THREADS': None}`).
The code under test is this tree, with the limit built in.

| Outer iteration | CG iterations | Median (ms) | Min (ms) | Max (ms) |
|---:|---:|---:|---:|---:|
| 1 | 852 | 133.6 | 132.2 | 140.6 |
| 100 | 861 | 139.4 | 135.8 | 214.0 |
| 1000 | 857 | 134.3 | 129.9 | 201.0 |

Under the default threads the same systems took 1,033 to 1,102 ms (step 2 report, section 11), and
under `OPENBLAS_NUM_THREADS=1`, 136 to 143 ms. The iteration counts are the same. Criterion 4
(under 0.5 s) is met under the default environment, with the medians 3.6 to 3.7 times inside it.
The two maxima over 200 ms are one slow solve of seven each and do not touch the median.

**Bits.** `val001_80x40` under the case file's rule, with the limit, two repeats, run by
`results/builder38/baseline38.py --tree .` against a scratch results file (the harness's own
`run_case`; `benchmarks/results.jsonl` is untouched): 1,559 outer iterations,
`error_estimate_and_continuity`, no capped correction, accuracy 4.107699e-4. The final faces,
hashed by section 2.3 of the step 2 report (SHA-256 over `u` then `v`, C-order little-endian
float64):

| | Hash |
|---|---|
| Step 2 report, `val001_80x40` | c7d88d919e1141a59c6d8561875f370db6e548ba12d286a68d062b4e1cef0595 |
| With the limit, repeat 1 | c7d88d919e1141a59c6d8561875f370db6e548ba12d286a68d062b4e1cef0595 |
| With the limit, repeat 2 | c7d88d919e1141a59c6d8561875f370db6e548ba12d286a68d062b4e1cef0595 |

Identical. `PRESSURE_SOLVER_VERSION` stays 2. Above about 10,000 cells the limit moves the bits by
rounding only (step 2 report, section 11: 7.1e-14 m/s on 160x80); no stored row or saved truth is
on a mesh that large.

## 5. Prediction against measurement

| | Predicted | Measured | |
|---|---|---|---|
| (a) | One thread faster at every length up to 100,000 | Equal within spread up to 9,600; faster from 12,800 (78 times), 10 times at 100,000 | Holds; below the threshold "faster" is "the same" |
| (b) | Crossover, if any, between 300,000 and 3,000,000 | Between 1,000,000 (default 1.37 times slower) and 1,500,000 (0.93) | Holds |
| (c) | Limit within 20% of the variable at 15,000 elements, "about 4.3 microseconds" | 5.29 us against 5.22 us, 1.01 times. The variable's own figure is 5.2 us, 21% above the 4.3 the prediction took from the step 2 report | The relation holds; the absolute figure did not reproduce |
| (d) | Same face hash on `val001_80x40` | Identical, both repeats | Holds |
| (e) | 200x75 median under 0.5 s and within 20% of 136 to 143 ms | 133.6, 139.4 and 134.3 ms | Holds |

## 6. Where it is recorded

ADR-013 "Planned against built" has a row for the thread limit. ECR-003 criterion 4 carries a dated
note (2026-10-07) that it is met under the default environment. `docs/SYSTEM.md`'s `pressure.py`
contract names the limit and the dependency. `requirements.txt` gains `threadpoolctl>=3.2`.
