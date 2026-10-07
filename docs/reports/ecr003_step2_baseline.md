# ECR-003 Step 2: The Laminar Baseline under Conjugate Gradients

**Date:** 2026-10-07
**Tree:** branch `feature/ecr003-baseline-close` from main at 311034e (ECR-003 step 1 merged as PR
62). No change under `src/`, `scripts/`, `validation/` or `configs/`: the rows are taken on
311034e's source.
**Instruments:** `results/builder38/` (untracked): `baseline38.py`, which runs the harness's own
`run_case` and hashes the final faces of the same solve; two detached worktrees,
`wt_311034e` (main, the solver under test) and `wt_867ef89` (main before step 1, the weighted
Jacobi solver); every run's log and manifest beside them.
**Order:** this section and sections 1 to 3 are committed before the runs. One run came before
them: a smoke test of `baseline38.py` on `val002_20x20` into a scratch file, which section 3
reports.

## 1. The question

ECR-003 step 2 (`docs/ECR/ECR-003-pressure-solver.md`, section 8) retakes the laminar baseline
under the conjugate gradient correction step 1 built. Criterion 2 asks that `val001_80x40`,
`val001_80x40_stretched` and `val002_80x80` meet REQ-S02 (VAL-001 below 1% on both meshes) and
REQ-S03 (VAL-002 below 2% of the lid speed against Marchi) under the change, stop by
`error_estimate_and_continuity`, and that the transport gate tests pass. The rows taken here, under
the label `staggered-cg`, are ECR-002 criterion 1's laminar baseline from this step on ("if ECR-003
lands first", ECR-002 section 9), and that criterion compares face hashes, so the hash of each
case's final faces is recorded with its definition and the fields are saved.

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
