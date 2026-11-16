# Phase 6 Step 1: the Toolchain (prompt 86)

**Date:** 2026-11-16
**Prompt:** 86 (`docs/prompts/cc-prompt-86-toolchain.md`)
**Branch:** `phase6/toolchain`
**Status:** predictions committed; nothing run yet.

## The predictions

Set by the orchestrator in prompt 86 on 2026-11-16, and committed alone before any run.

- (a) The Linux reference is not bitwise the Windows records on at least one of the three runs
  (another C library's `exp`, `log` and `pow`), but every decision is equal and every field is
  inside the measured spread (ADR-017 H's first tier).
- (b) `pairwise_sum` equals `np.sum` bit for bit on every contiguous 1-D array tested (lengths
  0 to 1,000 and the three grids' cell counts, random values of both signs) and on C-contiguous
  2-D arrays of the three grids' shapes summed whole.
- (c) The 80x30 dust run's wall time with its run file on `/mnt/c` is within 10% of its wall time
  on the Linux file system.
- (d) The Linux reference's 80x30 dust run is within 10% of the Windows run's wall time.

## How they are scored (the builder, before the runs)

The three runs are the probe's 80x30 references (`docs/reports/probe85/probe85.py`): the dust
release, the door leak and the filter breach, each `time_integration.run` from rest in its own
process on one BLAS thread, under `.venv-linux/`, from main's scenario files with the grid and
`mass_imbalance_tol` written as `probe85.variant_mapping` writes them for `reference`. The
Windows records are `results/builder85/runs/reference/`. The comparison is `tables85.py`'s,
unchanged: `compare_run` for the fields, the budget and the bitwise flags, and `Placement` for
the searches, the layouts' order and the clear times, with the Windows run as the reference and
the Linux run as the variant.

- **(a), bitwise.** A run is bitwise when `compare_run`'s flag is true: every flow state's u, v,
  p, k, eps and nu_t, every stored frame of every class, every stored budget row, the sensor
  readings and channels and the arrival maps equal byte for byte. The first clause holds when
  at least one of the three runs is not bitwise.
- **(a), the decisions.** Equal means: no arrival time, sensor latency or layout latency moved by
  a step or reached in one run only (P3's three counts all zero); the searched cells of every
  pool under both rules the same and the four layouts' order the same (P4); no cell's or room's
  clear time moved by a frame or reached in one only (P5); and each class's verdict against
  REQ-T05 (the residual and the leftover within 1e-4 of the supply) the same (P6).
- **(a), the fields.** Inside the measured spread means each field's relative difference, as
  `compare_run` measures it, is at or below the largest the probe measured for it: the flow
  fields (each of u, v, p, k, eps and nu_t) 7.8e-10 and the class fields 2.7e-8 (P8); the class
  fields pointwise 5.8e-8 (P8); the sensor channels 4.7e-8 (P2's largest, as ADR-017 H takes
  it); the accumulated leftover's change 3.6e-10 of the supply (P8). These are the values the
  report prints, to two figures. The budget's closure (the residual within 1e-12 of the supply)
  is the reference's own check and holds or fails beside them. (a) holds when all three clauses
  hold.
- **(b).** Each array's `pairwise_sum` and `np.sum` are compared as 64-bit patterns (so 0.0 and
  -0.0 differ). The arrays: every length from 0 to 1,000, and 2,400, 9,600 and 38,400 (the 80x30,
  160x60 and 320x120 cell counts), with values from a fixed seed, standard normal times ten to a
  power uniform in [-6, 6], so both signs and many magnitudes; and the 2-D arrays of shapes
  (30, 80), (60, 160) and (120, 320) drawn the same way, `np.sum` of the 2-D array against
  `pairwise_sum` of its 1-D C-order view. (b) holds when every pair is equal.
- **(c).** Two runs of the 80x30 dust release under `.venv-linux/`, one with its run file (and
  its checkpoint file) under `results/builder86/` on `/mnt/c`, one under `~/cfd-runs/`, each run
  alone with no other solver process running, one after the other. The wall time is the
  summary's `wall_seconds` (`time_integration.run` alone, as `probe85.run` times it). (c) holds
  when |W(/mnt/c) - W(Linux)| / W(Linux) <= 0.10.
- **(d).** The Linux reference's dust run, made as the Windows record was made (the three
  references at once, one BLAS thread each), against the Windows record's `wall_seconds`. (d)
  holds when |W(Linux) - W(Windows)| / W(Windows) <= 0.10.
