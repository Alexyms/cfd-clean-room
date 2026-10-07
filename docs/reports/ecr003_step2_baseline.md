# ECR-003 Step 2: The Laminar Baseline under Conjugate Gradients

**Date:** 2026-10-07
**Tree:** branch `feature/ecr003-baseline-close` from main at 311034e (ECR-003 step 1 merged as PR
62). No change under `src/`, `scripts/`, `validation/` or `configs/`: the rows are taken on
311034e's source.
**Instruments:** `results/builder38/` (untracked): `baseline38.py`, which runs the harness's own
`run_case` and hashes the final faces of the same solve; two detached worktrees,
`wt_311034e` (main, the solver under test) and `wt_867ef89` (main before step 1, the weighted
Jacobi solver); every run's log and manifest beside them. Added after the runs began:
`diagnose38.py` (section 6) and `controls38.sh` (sections 7 and 8).
**Order:** this section and sections 1 to 3 are committed before the runs (03e339b). One run came
before them: a smoke test of `baseline38.py` on `val002_20x20` into a scratch file, which section 3
reports. The six rows ran alone; the controls, the diagnosis and the transport gate ran after them,
side by side, and none of their timings is used.

## 1. The question

ECR-003 step 2 (`docs/ECR/ECR-003-pressure-solver.md`, section 8) retakes the laminar baseline
under the conjugate gradient correction step 1 built. Criterion 2 asks that `val001_80x40`,
`val001_80x40_stretched` and `val002_80x80` meet REQ-S02 (VAL-001 below 1% on both meshes) and
REQ-S03 (VAL-002 below 2% of the lid speed against Marchi) under the change, stop by
`error_estimate_and_continuity`, and that the transport gate tests pass. The rows taken here, under
the label `staggered-cg`, are ECR-002 criterion 1's laminar baseline from this step on ("if ECR-003
lands first", ECR-002 section 9), and that criterion compares face hashes, so the hash of each
case's final faces is recorded with its definition and the fields are saved. (After the runs: the
step stopped on the prompt's second condition, and section 9 leaves the baseline to Alex.)

## 2. Method

### 2.1 The rows

`baseline38.py --tree wt_311034e` imports that worktree's `scripts/benchmark.py` and calls its
`run_case(case, "staggered-cg", 10, 1)`, as `benchmark.py --cases ... --repeats 2` would, and
appends each record to this branch's `benchmarks/results.jsonl` with the harness's own
`json.dumps(record) + "\n"`. The one addition is a subclass of `StaggeredSolver` that keeps the
instance and each correction's iteration count; it changes no arithmetic. The rows therefore carry
`git_commit` 311034e with `git_dirty` false, a commit that stays on main after this branch is
rebased onto it. Each case runs twice, one process at a time with nothing else of mine running
(`concurrent_processes` 1), trajectory sampled every 10 outer iterations, the case file's own
`error_estimate` rule (version 3). The environment is the one the Jacobi rows record: the AMD Ryzen
AI 9 HX 370, Windows 11, Python 3.13.3, NumPy 2.4.4 with OpenBLAS 0.3.31; no BLAS thread setting.

### 2.2 The rows compared

Two `staggered-jacobi` error_estimate rows exist for each case, but they are not two repeats: one
was taken under stopping rule version 2 (three conditions, 2026-09-25 and 26) and one under version
3 (the fourth condition, 2026-10-01).

| Case | Rule version 3, the comparison | Rule version 2, for context |
|---|---|---|
| `val001_80x40` | f56fce25: 3,988 outer, 4.1068e-4, 111.5 s | b8a2f3df: 3,154 outer, 4.1042e-4, 100.9 s |
| `val001_80x40_stretched` | 43b02c72: 2,253 outer, 3.0238e-3, 40.5 s | d2a57fe1: 1,523 outer, 3.0239e-3, 33.9 s |
| `val002_80x80` | 4a18533f: 12,849 outer, u 1.0573e-3, v 7.356e-4, 394.5 s | 0d0d7efa: 12,849 outer, the same, 426.5 s |

`src/stopping.py` has `RULE_VERSION = 3`, so the CG rows compare with the version 3 rows. The
prediction's accuracy figures are theirs.

### 2.3 The face hash

For each row: SHA-256 over the bytes of `StaggeredSolver.face_velocities.u` (shape [ny, nx+1])
followed by the bytes of `.v` (shape [ny+1, nx]), each C-ordered little-endian float64, nothing
else hashed. These are the read-only faces the solver keeps at the end of every solve, the
corrected faces of the outer iteration that stopped it. In Python:

```python
digest = hashlib.sha256(
    np.ascontiguousarray(faces.u, dtype="<f8").tobytes()
    + np.ascontiguousarray(faces.v, dtype="<f8").tobytes()
).hexdigest()
```

Each row's faces are saved as `results/builder38/baseline/{case}_{run_id[:8]}.npz` with arrays `u`
and `v` and the hash, run id and commit beside them; the hashes of u and v alone are in the
manifest.

### 2.4 Controls

- **Repeat.** The two repeats of a case must give the same hash; the solve is deterministic.
- **One BLAS thread.** CG's three reductions are `np.vdot`, which OpenBLAS may split across threads
  on long vectors. One more solve of each case with `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and
  `MKL_NUM_THREADS` set to 1, written to a scratch file, must give the same hash.
- **The Jacobi faces.** The stored Jacobi rows carry no hash. Each case is solved once more on
  `wt_867ef89` by that tree's own `run_case(case, "staggered-jacobi", 10, 1)` into a scratch file.
  If its outer count and accuracy equal the version 3 row's to the bit, its faces stand for that
  row's, and the CG faces are compared with them.

### 2.5 The transport gate tests

`tests/test_diffusion.py`, `test_advection.py`, `test_conservation.py`, `test_constancy.py`,
`test_smith_hutton.py` and `test_sealed_box.py` (VAL-003, VAL-004 in two rows, VAL-007, VAL-012,
VAL-013, VAL-014) on this branch. Two of them read VAL-001's faces from a solve
(`test_conservation.py`, `test_constancy.py`, ECR-003 section 7.2), so they ran under CG in step 1's
pull request already; this is the re-check criterion 2 names.

## 3. Prediction (orchestrator, written before handover)

| Prediction | What it rests on |
|---|---|
| Outer counts within 1% of the Jacobi rows | ECR-003 measurement 2: the outer count does not move above a relative level of 1e-2, and Jacobi's stop delivered about 1e-3 |
| Accuracy unchanged to three figures: VAL-001 4.107e-4 and 3.024e-3; VAL-002 u 1.057e-3, v 7.356e-4 of the lid speed | The same discrete solution, converged to the same iteration error |
| Wall time three to ten times shorter, the cavity most | The pressure stage is most of a Jacobi solve's time (92% to 98% at ECR-001 step 6) |
| Face hashes differ from the Jacobi rows' | Every laminar bit changes (ECR-003's stated consequence) |

The prompt's stops, which end the step at the measurement and send it to Alex: a case fails its
criterion, stops otherwise than by `error_estimate_and_continuity`, or records a cap hit; an outer
count moves by more than 5% from its Jacobi row, which contradicts measurement 2; a transport gate
test fails. A count between 1% and 5% away misses the prediction without stopping the step.

**Before this commit.** The smoke test of `baseline38.py` on `val002_20x20` (two repeats into
`results/builder38/smoke/`, not the committed file) stopped by `error_estimate_and_continuity` at
outer 970 with no cap hit, u 2.1436e-2 and v 1.3370e-2 of the lid speed, both repeats with face hash
8580ae56fa7676a8.... The only `staggered-jacobi` error_estimate row of that case, 5129231b under
rule version 2, stopped at 1,370 with u 2.1436e-2. The 20x20 cavity is not one of step 2's cases,
and its Jacobi row is under the other rule version, but the 29% fewer outer iterations is a sign
the prediction's premise may not hold on the validation cases: ECR-003 section 1 records Jacobi's
stop leaving a relative residual of 0.95, not 1e-3, on these two cases' own systems (VAL-001 80x40
at outer 569, VAL-002 80x80 at outer 1,000; `docs/reports/pressure_solver_ecr003.md`, section
7.1), and under `error_estimate` the per-cell continuity condition reads that residual.

## 4. The rows (written after the runs)

Six rows appended to `benchmarks/results.jsonl`, all at `git_commit` 311034e with `git_dirty`
false, `pressure_rtol` 1e-8, `max_pressure_iter` 5,000, rule version 3, one process:

| Case | Run ids | Outer | Stop | Cap hits | CG iterations per correction, median [max] | Accuracy | Wall s |
|---|---|---|---|---|---|---|---|
| `val001_80x40` | ae24b120, 63d655ba | 1,559 | `error_estimate_and_continuity` | 0 | 194 [244] | 4.1077e-4 | 29.7, 29.5 |
| `val001_80x40_stretched` | a88453c2, a8ac6dd8 | 1,124 | `error_estimate_and_continuity` | 0 | 186 [251] | 3.0237e-3 | 20.1, 20.2 |
| `val002_80x80` | 284d7844, b00e334c | 12,814 | `error_estimate_and_continuity` | 0 | 265 [414] | u 1.0572e-3, v 7.3556e-4 | 434.1, 465.9 |

REQ-S02 holds on both meshes (4.108e-4 and 3.024e-3, against 1e-2) and REQ-S03 at 80x80 (1.057e-3
of the lid speed, against 2e-2). Every row stops by the rule with no capped correction. Beside each
case's Jacobi row under the same rule version:

| Case | Outer, Jacobi to CG | Change | Accuracy, Jacobi to CG | Wall s, Jacobi to CG | ms per outer iteration |
|---|---|---|---|---|---|
| `val001_80x40` | 3,988 to 1,559 | -60.9% | 4.10683e-4 to 4.10770e-4 | 111.5 to 29.6 (3.8 times shorter) | 28.0 to 19.0 |
| `val001_80x40_stretched` | 2,253 to 1,124 | -50.1% | 3.02376e-3 to 3.02366e-3 | 40.5 to 20.2 (2.0 times shorter) | 18.0 to 17.9 |
| `val002_80x80` | 12,849 to 12,814 | -0.27% | u 1.05725e-3 to 1.05721e-3; v 7.35582e-4 to 7.35562e-4 | 394.5 to 434.1 and 465.9 (9% to 15% longer) | 30.7 to 33.9 and 36.4 |

The work in cell updates follows the counts: 3.40e9 to 1.00e9 and 1.14e9 to 6.81e8 on the
channels, and 1.59e10 to 2.18e10 on the cavity, where CG's 265 iterations per correction to 1e-8 or
the floor outnumber the Jacobi row's 192 sweeps on average (median 50) at its 1e-8 Pa stop. A CG
iteration took less time than a sweep (80 against 95 microseconds on 80x40, 111 against 136 on
80x80).

The face hashes, by section 2.3's definition, identical in both repeats of each case:

| Case | u then v | u alone | v alone |
|---|---|---|---|
| `val001_80x40` | c7d88d919e1141a59c6d8561875f370db6e548ba12d286a68d062b4e1cef0595 | 82a46a223e99f198a339a78566fbe5180d3771fbb09f413b4834a170b420c366 | 73b2a0f2dd2a71c23b861a67e3278aa98a807f3fa3ad3b3305906afd707fcc45 |
| `val001_80x40_stretched` | 9838d632f181d90fdd948da9737de2a2e814711c6e40d3dfd78a9f52cda2349b | 6db983c04b9238366ec430379083ba6fcf80e83897f8a40fb2514cd8b6cfa2c8 | cab2d39a1e02313996c024dafb47bdacd94cb7e130fe63e6f252232bf9602bf2 |
| `val002_80x80` | a8a61e28120e3f44d8abd5f39d740cae7a611209d67fed7a2b5b16328d6ea495 | a1ce63721b2ad112c722e8a94f5bdb4faee0c7f37ac2ec2fab4ef10974c43cea | 4078a3f8b017a95303451995483a37c431b2cf2f058c9b1c6014d2ffc83e289e |

The fields are saved as `results/builder38/baseline/{case}_{run_id[:8]}.npz`, six files; each
file's arrays, read back, reproduce the hash stored in it.

## 5. Against the prediction (written after the runs)

| Prediction | Measured |
|---|---|
| Outer counts within 1% of the Jacobi rows | Held on the cavity (-0.27%). Not on the channels: -60.9% and -50.1%, past the 5% stop (section 9) |
| Accuracy unchanged to three figures | Held to three significant figures on every case. At the four the prediction quotes, the uniform channel reads 4.108e-4 against 4.107e-4 (a difference of 8.7e-8, 2e-4 of the value); the stretched channel and the cavity agree at four |
| Wall time three to ten times shorter, the cavity most | Not held. The channels are 3.8 and 2.0 times shorter, from the fewer outer iterations: per outer iteration 1.5 and 1.0 times. The cavity takes 9% to 15% longer |
| Face hashes differ from the Jacobi rows' | Held on every case (section 7). The faces differ by at most 1.1e-7 m/s on the channels and 7.3e-8 m/s on the cavity |

## 6. Why the channel counts moved (written after the runs)

`diagnose38.py` ran each case once more on each tree, reading the rule's four quantities on the
corrected faces at every outer iteration and each correction's delivered relative residual,
`||imbalance(corrected faces)|| / ||imbalance(u*)||` (ADR-013 B: the residual of the p' equation is
the corrected faces' imbalance). Outer iterations are counted from 1.

| Case, solver | Delivered relative residual, median [10th percentile, smallest] | (a) first held; last failed | (b) last failed | (c) last failed | (d) held at | Stop |
|---|---|---|---|---|---|---|
| `val001_80x40`, Jacobi | 0.9997 [0.56, 3.2e-3] | 2,286; 3,982 | 1,804 | 2,778 | two outer iterations, 3,001 and 3,988 | 3,988 |
| `val001_80x40`, CG | 1e-8 or less until the floor binds | 1,559; 1,558 | never | never | every outer iteration | 1,559 |
| `val001_80x40_stretched`, Jacobi | 0.9990 [0.38, 3.6e-3] | 1,133; 2,230 | 1,226 | 1,145 | two outer iterations, 1,453 and 2,253 | 2,253 |
| `val001_80x40_stretched`, CG | as above | 1,124; 1,123 | never | never | every outer iteration | 1,124 |
| `val002_80x80`, Jacobi | 0.94 [0.41, 0.042] | 12,849; 12,848 | 11,275 | 2,945 | every outer iteration | 12,849 |
| `val002_80x80`, CG | as above | 12,814; 12,813 | never | never | every outer iteration | 12,814 |

Under CG each correction ends at the relative level of 1e-8 or, once the right-hand side is small,
at the rounding floor `1e-13 F`; there the relative figure rises (to 6e-3 on the channels and 7e-2 on
the cavity at the stop) while the corrected faces' worst cell is at most 4.11e-12, 8.15e-12 and
1.82e-12 kg/s per metre over the whole solve.

The reading:

1. **The prediction's premise.** Jacobi's 1e-8 Pa stop did not deliver a relative residual of
   about 1e-3 on these cases. The median correction left 99.97%, 99.90% and 94% of u*'s imbalance
   in the faces; 69% and 67% of the channel corrections were a single sweep. The first 410, 129 and
   3,728 corrections ran to the cap (2,000, 2,000 and 500 sweeps), unreported, since the Jacobi rows
   predate `pressure_cap_hits`. ECR-003's measurement 2 compared CG at several levels with the exact
   correction in the probe rooms; it measured no Jacobi level on these cases, and ECR-003 section 1
   had already recorded 0.95 on their systems. Measurement 2 is not contradicted; the prediction's
   reading of it was.
2. **The channels under Jacobi waited on condition (d).** With the corrections leaving the imbalance
   in the faces, the net outflow decays as the oscillation ADR-010 and
   `docs/reports/stopping_rule_evidence.md` (section 10) describe, and (d) holds only at its zero
   crossings: twice in each channel solve. Both channel stops are the first crossing after (a)
   settles. (a) itself settled late: on the uniform channel it first held at 2,286 and failed again
   until 3,982, on the stretched one at 1,133 and again until 2,230, as the oscillation passes
   through the velocity step the estimate reads.
3. **Under CG the oscillation is gone.** Every correction leaves the faces balanced to rounding, so
   the signed domain sum, the net outflow, never exceeds 4.6e-12 and 1.8e-11 kg/s per metre on the
   channels, (b), (c) and (d) hold from the first outer iteration, and (a) alone sets the stop, at
   1,559 and 1,124. The underdamped net-outflow mode was a product of the Jacobi correction, not of
   the outer loop's under-relaxation alone.
4. **The cavity has no net-outflow mode.** On the closed domain (d) holds throughout under both
   solvers, (a) sets both stops, and the counts agree to 0.27%, as the prediction said.
5. **The fields agree.** The CG faces differ from the Jacobi faces by at most 1.06e-7 m/s (uniform
   channel), 9.9e-8 (stretched) and 7.3e-8 (cavity). Each solve's iteration error is bounded by
   1e-6 of its velocity scale (0.1 m/s on the channels, the lid's 1 m/s on the cavity), so two
   solves may differ by up to 2e-7 and 2e-6 m/s; the channel differences sit near half that, the
   cavity's well inside it.

## 7. The controls (written after the runs)

| Control | Result |
|---|---|
| Repeat | Both repeats of each case: the same hash, outer count, inner counts and accuracy |
| One BLAS thread | The same three hashes (`results/builder38/controls/`) |
| The instrumentation | `diagnose38.py`'s CG runs give the rows' three hashes: the wrappers change no bit |
| The Jacobi faces | The reruns on 867ef89 reproduce rows f56fce25, 43b02c72 and 4a18533f exactly: outer counts, sweeps, cell updates, accuracy and its components, to the bit. Their face hashes: bbc8295ddc7e444f326020656a0e475e62c8a4245e10de118f9959f517685c1c, a86dcb5edb5feb323bfb6d79ae35c83627416510f4ad9d2aa1c77c5101e7f1f9 and 5bda695e1a2128b481cf4ffa6368ab72187dea880ee1aed0912f2f59dc8a66cc, none equal to its CG counterpart |

**What the hashes are of.** NumPy's `vdot` calls OpenBLAS's `ddot`, and this OpenBLAS (0.3.31,
built with DYNAMIC_ARCH) chooses its kernel by CPU. Another kernel or library sums CG's three
reductions in another order, which moved the corrected faces by up to 1.6e-12 m/s in ADR-013 C's
measurement and changes every bit downstream. The hashes are this machine's, as every stored row's
seconds are. ECR-002 criterion 1's bitwise comparison holds when the base and the branch run on one
machine; on another machine the base's hashes are retaken there, and these rows' outer counts,
stops and accuracy are the part that carries across machines to the reductions' rounding.

## 8. The transport gate (written after the runs)

`pytest -rP` on the six files: 13 passed in 65 s. Every printed value equals the one
`docs/PROJECT_PLAN.md` records except VAL-007's second case: the budget on the VAL-001 40x20 faces
closes to -1.065e-14 against the plan's -7.4e-15, which was measured on the Jacobi faces in PR 32;
both are rounding against the 1e-4 criterion. VAL-012 reads the values the plan recorded at its
split on 2026-10-07 (departure 3.997e-12 against a bound of 3.996e-11; planted ratio 0.6513).

## 9. The stop

The prompt's second stop fired: two outer counts moved by more than 5% from their Jacobi rows
(-60.9% and -50.1%). None of the others did: every case meets its criterion and stops by
`error_estimate_and_continuity` with no capped correction, and the transport gate passes. The step
ends here, at the measurement. The six rows are appended and committed as what they are, the solver
at 311034e measured; step 3's records (ADR-013 built, ECR-003 closed, SYSTEM.md, PROJECT_PLAN.md,
STATUS.md and ECR-002 criterion 1's pointer) are not written.

What is Alex's to decide:

1. **Whether these rows are ECR-002 criterion 1's laminar baseline.** The builder's reading is that
   they are: section 6 traces the moved counts to the Jacobi rows' stops, which waited on an
   imbalance the old correction left in the faces (and on silent capped corrections), not to the CG
   level, and the CG fields lie within the two solves' iteration-error bounds of the Jacobi fields.
2. **If so, what step 3 records beyond the prompt's list.** Three findings bear on standing text.
   The net-outflow oscillation, which ADR-010's "For Phase 3" section and the stopping-rule report's
   section 10 treat as the outer loop's, is absent under CG, and condition (d) no longer binds on the
   channels. The VAL-001 and VAL-002 gate rows quote orders of convergence (1.992; 2.24, 2.11, 2.12,
   2.07) measured under Jacobi, which step 2 does not retake (`scripts/val001_order.py` and the 20x20
   and 40x40 cavity rows would). On the 80x80 cavity CG at the default costs 9% to 15% more wall
   time than the Jacobi row, so ADR-013's cost case rests on the product mesh, not on the validation
   cases.
