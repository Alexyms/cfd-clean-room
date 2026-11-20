# Phase 6 Step 2c: the Scalar Kernels Made Fast (prompt 89)

**Date:** 2026-11-19
**Prompt:** 89 (`docs/prompts/cc-prompt-89-scalar-perf.md`)
**Branch:** `phase6/scalar-perf`
**Status:** the predictions and their scoring, committed before any change to `core/`.

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
