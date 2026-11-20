# Prompt 89: the tables

### T0. The predictions, scored by the rule committed with them

| Prediction | Measured | Verdict |
|---|---|---|
| (a) every parity test, GoogleTest case and captured replay of prompts 87 and 88 passes unchanged | 68 of 68 GoogleTest cases and 1,404 of 1,404 parity cases that 3c8d1ec has; 120 of 120 captured flux calls and 100 of 100 captured Jacobi solves bit for bit; 40 of 40 captured CG solves at the reference's count, largest difference 1.86e-16, and 40 of 40 bit for bit with the core's dot products (T9); no line of an existing case edited or removed (section 6) | holds |
| (b) a Jacobi sweep at most 0.07 ms under the current flags | 0.0526 ms in run 1 (0.0509 and 0.0503 in runs 2 and 3); 1.91 times faster than the core before the pass in the same session (2.03, 1.97), not the four times of the prediction's parenthesis (T4) | holds |
| (c) the set-up with one residual at most 0.15 ms | 0.1063 ms in run 1 (0.1070, 0.1056) (T4) | holds |
| (d) the flux at least 1.5 times faster per call than the core's before | 1.90 to 2.25 times over the ten rows in run 1 (1.99 to 2.24, 1.97 to 2.29) (T4) | holds |
| (e) `-O3` and `-march=native` change the kernels' time by less than 25% | the largest change in run 1 is 29%, a Jacobi sweep under `-O3 -march=native`; at or beyond 25%: a Jacobi sweep under `-O3 -march=native` (-29%); the flux along y under `-O2 -march=native` (+28%); the set-up with one residual under `-O2 -march=native` (-26%) (runs 2 and 3: 29%, 28%); the checks under each set: all pass (T7) | FAILS |

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

### T11. The whole suite at 6a13cfc

The two suites ran at the same time, one BLAS thread each, beside the planted defects, so the wall times are not those of a suite alone.

| Platform | Tests | Passed | Failed | Errors | Skipped | Wall time, s |
|---|---|---|---|---|---|---|
| Windows, `.venv`, the core not built | 3,102 | 3,101 | 0 | 0 | 1 | 1,215 |
| WSL2, the core built, `CFD_CORE_REQUIRED=1` | 6,186 | 6,186 | 0 | 0 | 0 | 537 |
