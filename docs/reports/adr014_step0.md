# ADR-014 Step 0: The Housekeeping, the Warm Start and the Steady Checkpoint

**Date:** 2026-10-16
**Tree:** branch `phase4/step0-warm-start` from main at 1c58700 (PR 87, ADR-014 merged as
Proposed). The commits are listed in section 1 and in the pull request body.
**Instruments:** `docs/reports/probe60/` (`hash60.py`, `warm60.py`, `run60.sh`, `tables60.py`,
whose `--check` regenerates the tables and compares); records in
`results/builder60/` (untracked).
**Order:** this report's predictions (section 2) were committed before the two product runs of
section 4 and the hash runs of section 3 were started.

## 1. What the step built

ADR-014 section G, step 0, as prompt 60 sets it out:

- Commit 0, the records: ADR-014 accepted by Alex on 2026-10-16, each decision's recommendation
  as written; the amendment of the same date, the equipment vents as airflow only, as step 0b
  of section G and the design text of a new section J, with its limit in section H; SYSTEM.md's
  planned contracts; STATUS; PROJECT_PLAN's Phase 4 rows.
- The housekeeping batch of section I: the constants of tests 55 and 56 pinned;
  `largest_inlet_eddy_viscosity` removed with its tests and contract line; the docstring
  summaries over 88 characters shortened in the files the batch and the step touch;
  `field43.py`'s git lookup moved out of import; `scripts/preflight.py` added; the pipes inside
  table cells of the four ECR-002 reports escaped, at the reports and at the generator; the 16
  local branches merged into main deleted (the remote ones section I counted were already gone).
- The warm start: `FlowState` and `solve_steady(initial=...)` (ADR-014 A).
- The steady checkpoint and its bitwise continuation: `SteadyCheckpoint`,
  `solve_steady(resume=...)`, `ErrorEstimateRule.window` and `restore`, and the
  `/steady_checkpoint/` group of the run file in a new `src/io_manager.py` (ADR-014 D).

## 2. Predictions

**Orchestrator (prompt 60, 2026-10-16), as given.**

- (a) Every hash of record is unchanged with `initial=None`.
- (b) The self-warm-started product solve stops within one rule window (100 outer iterations)
  of its start.
- (c) The open-from-closed re-solve through the argument matches T1's warm arm to the outer
  iteration and its faces bitwise.

**Builder (2026-10-16, before the runs, scored apart).**

- (a) Holds. Nothing in the loop changed for a solve with neither keyword; the start from rest
  and a warm start from a state of rest give the same bits on both test fixtures
  (`tests/test_warm_start.py`), which a change on the shared path would break.
- (b) Holds at exactly 100: the rule starts fresh, so it cannot stop before its window is
  full, and both fixtures, warm-started from their converged states, stop at the first full
  window. The rebuilt nu_t differs from the relaxed one by less than condition (e)'s
  tolerance at a converged stop, so the first full window meets (e) as well.
- (c) Holds bitwise: the argument copies the same arrays the probe's patch returned and passes
  k and eps through the same `KEpsilonModel.state`, which `tests/test_warm_start.py` pins on
  the 40x15 room with a changed supply.

## 3. Prediction (a): the hashes of record

Every case whose face hash is of record, solved from rest at the branch's code (`hash60.py`,
worktree `results/builder60/B`), with neither `initial` nor `resume`: the harness rows and the
two product rows of prompt 54 (`docs/reports/multigrid_default.md`, the hash table), ECR-002
step 8's door-closed room on the product grid (L0) and T1's rest arm of prompt 59. Nothing under
`src/` changed after the hashes of prompt 54 were taken except this branch, and no case file
the harness rows read; the product file moved into `configs/clean_room_default.yaml` after
prompt 54 (prompt 56), which `coupled47.product_raw` reads, so the two product rows also ran at
the base commit 1c58700 as a control. L0 and T1's arm are solved from the `raw` mapping their
records keep. The runs were launched together, 15 processes and the planted-defect runs beside
them (`run60.sh runs`); counts and hashes do not depend on the load.

<!-- BEGIN tables60: hashes -->
| Case | Hash of record (commit, source) | Its stop | From rest, initial=None (commit) | Stop | Equal | Base 1c58700 control |
|---|---|---|---|---|---|---|
| val001_80x40 | f2e5b888cb269112 (759b3ad, prompt 54) | error_estimate_and_continuity at 1,559 | f2e5b888cb269112 (30233b2) | error_estimate_and_continuity at 1,559 | yes | - |
| val001_80x40_stretched | 4c79ffb9def55fbb (759b3ad, prompt 54) | error_estimate_and_continuity at 1,124 | 4c79ffb9def55fbb (30233b2) | error_estimate_and_continuity at 1,124 | yes | - |
| val002_20x20 | 1ecbedb180303c8f (759b3ad, prompt 54) | error_estimate_and_continuity at 970 | 1ecbedb180303c8f (30233b2) | error_estimate_and_continuity at 970 | yes | - |
| val002_40x40 | 7a881aaa88b1946d (759b3ad, prompt 54) | error_estimate_and_continuity at 3,435 | 7a881aaa88b1946d (30233b2) | error_estimate_and_continuity at 3,435 | yes | - |
| val002_80x80 | 95efc0123aaabc55 (759b3ad, prompt 54) | error_estimate_and_continuity at 12,814 | 95efc0123aaabc55 (30233b2) | error_estimate_and_continuity at 12,814 | yes | - |
| val017_216x72 | d59c4a1a644f67aa (759b3ad, prompt 54) | error_estimate_and_continuity at 11,443 | d59c4a1a644f67aa (30233b2) | error_estimate_and_continuity at 11,443 | yes | - |
| val019_656x60 | 2c8a57b634e018ab (759b3ad, prompt 54) | max_simple_iter at 20,000 | 2c8a57b634e018ab (30233b2) | max_simple_iter at 20,000 | yes | - |
| product_80x30 | 6f27f27f3e89963a (759b3ad, prompt 54) | error_estimate_and_continuity at 1,021 | 6f27f27f3e89963a (30233b2) | error_estimate_and_continuity at 1,021 | yes | 6f27f27f3e89963a (1c58700) |
| product_160x60 | d295ccb1a4c8f9e1 (759b3ad, prompt 54) | error_estimate_and_continuity at 1,953 | d295ccb1a4c8f9e1 (30233b2) | error_estimate_and_continuity at 1,953 | yes | d295ccb1a4c8f9e1 (1c58700) |
| L0_320x120 | cd0701fd8aab916d (b9e8191, ECR-002 step 8) | error_estimate_and_continuity at 4,389 | cd0701fd8aab916d (30233b2) | error_estimate_and_continuity at 4,389 | yes | - |
| t1_rest | 22364c66d2777fa8 (809d1c6, prompt 59, T1 rest arm) | error_estimate_and_continuity at 4,579 | 22364c66d2777fa8 (30233b2) | error_estimate_and_continuity at 4,579 | yes | - |
<!-- END tables60: hashes -->

**Scored.** (a) holds: every row reads "yes", each case stopping at the record's outer
iteration with the record's faces, and the two product rows give the same faces at the base
commit. The builder's (a) holds.

## 4. Item 6: the warm start on the product room

Two solves of the 320x120 room (`warm60.py`), each from L0's kept state (its faces hash to its
record's before the solve starts), through `solve_steady(initial=...)`: `self`, the door-closed
room of the product file, prediction (b); `open`, the open-door state of T1
(`warm59.door_open(warm59.product_raw())`), prediction (c), against T1's warm arm, which reached
the same start by patching the two allocations. The bound beside the face change is
`iteration_error_tol` times the velocity scale, the iteration error the rule allowed L0's stop.

<!-- BEGIN tables60: warm -->
| Arm | Door | Stop | Against the prediction | Largest face change from L0, u, v (m/s) | Face hash | Wall (s) | Processes |
|---|---|---|---|---|---|---|---|
| self | closed | error_estimate_and_continuity at 100 | within one window: yes | 4.400e-09, 2.852e-09 (bound 4.50e-07) | e978aa5167326540 | 18 | 15 |
| open | open | error_estimate_and_continuity at 3,872 | T1 warm 3,872; outer equal, faces bitwise, history equal | 3.676e-01, 6.319e-01 | 92d8e45e5b4b54ce | 767 | 15 |
<!-- END tables60: warm -->

**Scored.** (b) holds: the door-closed room warm-started from its own converged state stops by
the rule at the first full window, outer 100, its faces moved by far less than the bound. The
builder's (b), exactly 100, holds. (c) holds: through the argument the open-door re-solve
stops at T1's outer iteration with T1's faces and residual history, bit for bit, so the probe's
patch and the argument are the same start. The builder's (c) holds. Neither stop condition of
the prompt is met.

## 5. The tests and the planted defects

`tests/test_warm_start.py` (FlowState, the warm start, the resume), `tests/test_io_manager.py`
(the group written and read back, a resume from what was read) and the window tests added to
`tests/test_stopping.py` run on the VAL-001 channel at 40x20 (laminar, rule version 3, 408
outer iterations from rest) and the product file at 40x15 (coupled, version 5, 353), the
smallest committed rooms that stop by the rule in seconds; the 12x6 coupled channel of
`tests/test_coupled_solve.py` does not converge in 20,000 outer iterations. The housekeeping
tests are in `tests/test_benchmark.py`, `tests/test_impinging_jet.py` and
`tests/test_preflight.py`.

Each defect below was planted in the committed file, the named test files run against it and
the file restored; the control is the same files on the unplanted tree, so a test that fails
anyway does not read as a catch. The first group is the housekeeping batch's
(`plants_a.json`), the second step 0's (`plants_b.json`).

<!-- BEGIN tables60: plants -->
| Planted defect | File | Tests | Control exit | Planted exit | Result |
|---|---|---|---|---|---|
| float(np.median(nu_t[core] / nu)) -> float(np.mean(nu_t[core] / nu)) | scripts/benchmark.py | test_benchmark.py | 0 | 1 | caught |
| tops = max(obstacle.y_end for obstacle in config.obstacles) -> tops = min(obstacle.y_end for obstacle in config.obstacles) | scripts/benchmark.py | test_benchmark.py | 0 | 1 | caught |
| FREE_JET_H = 0.70 -> FREE_JET_H = 0.75 | validation/impinging_jet.py | test_impinging_jet.py | 0 | 1 | caught |
| IMPINGEMENT_H = 0.88 -> IMPINGEMENT_H = 0.80 | validation/impinging_jet.py | test_impinging_jet.py | 0 | 1 | caught |
| NARROW_FREE_JET_H = 0.30 -> NARROW_FREE_JET_H = 0.25 | validation/impinging_jet.py | test_impinging_jet.py | 0 | 1 | caught |
| PLATE_MEDIAN_HALF_WIDTH_W = 10.0 -> PLATE_MEDIAN_HALF_WIDTH_W = 3.0 | validation/impinging_jet.py | test_impinging_jet.py | 0 | 1 | caught |
| warm reads the state's nu_t | src/solver_staggered.py | test_warm_start.py | 0 | 1 | caught |
| warm start ignored | src/solver_staggered.py | test_warm_start.py | 0 | 1 | caught |
| warm skips the boundary's faces | src/solver_staggered.py | test_warm_start.py | 0 | 1 | caught |
| shape check removed | src/solver_staggered.py | test_warm_start.py | 0 | 1 | caught |
| laminar state accepted by a coupled solve | src/solver_staggered.py | test_warm_start.py | 0 | 1 | caught |
| coupled state accepted by a laminar solve | src/solver_staggered.py | test_warm_start.py | 0 | 1 | caught |
| resume rebuilds nu_t | src/solver_staggered.py | test_warm_start.py, test_io_manager.py | 0 | 1 | caught |
| resume leaves the window empty | src/solver_staggered.py | test_warm_start.py, test_io_manager.py | 0 | 1 | caught |
| resume restarts the residual history | src/solver_staggered.py | test_warm_start.py, test_io_manager.py | 0 | 1 | caught |
| resume counts from zero | src/solver_staggered.py | test_warm_start.py, test_io_manager.py | 0 | 1 | caught |
| resume resets the cap hits | src/solver_staggered.py | test_warm_start.py, test_io_manager.py | 0 | 1 | caught |
| converged checkpoint accepted | src/solver_staggered.py | test_warm_start.py | 0 | 1 | caught |
| rule version not compared | src/solver_staggered.py | test_warm_start.py | 0 | 1 | caught |
| FlowState keeps the caller's array | src/solver_staggered.py | test_warm_start.py | 0 | 1 | caught |
| flow_state outer off by one | src/solver_staggered.py | test_warm_start.py, test_io_manager.py | 0 | 1 | caught |
| window newest first | src/stopping.py | test_stopping.py, test_warm_start.py | 0 | 1 | caught |
| restore drops the viscosity window | src/stopping.py | test_stopping.py, test_warm_start.py | 0 | 1 | caught |
| io writes float32 | src/io_manager.py | test_io_manager.py | 0 | 1 | caught |
| io drops the viscosity window | src/io_manager.py | test_io_manager.py | 0 | 1 | caught |
| io reads a missing stop as a string | src/io_manager.py | test_io_manager.py | 0 | 1 | caught |
| io truncates the file | src/io_manager.py | test_io_manager.py | 0 | 1 | caught |
<!-- END tables60: plants -->

The first run of the second group (`plants_b_round1.json`) had two survivors, both resume
defects: a resume that left the rule's window empty, and one that reset the cap hits. The rule
reads nothing older than its window, so an empty window refills before any stop more than 100
outer iterations after the cut, which every cut of the test then was; and neither fixture caps
a correction. The resume test gained a cut 30 outer iterations before the uninterrupted stop,
and a channel capped at every correction pins the cap hits (commit 1c34ca8); the table is the
rerun against those tests, every plant caught.

The full suite ran once at 1c34ca8, the last commit to change code or tests, after the plants
had restored every file: `results/builder60/suite_full.txt`, every test passed.

## 6. What step 3 inherits

- The `/steady_checkpoint/` group's layout and the two functions of `src/io_manager.py`, which
  step 3 keeps and builds the run file around (ADR-014 D: `/config`, `/flow/<fingerprint>/`,
  `/transport/`, `/checkpoint/`).
- **The configuration check of a resume.** Today a resume refuses another grid, turbulence
  present or absent against the solve, a converged checkpoint and another rule version; it
  cannot see a changed tolerance, boundary or under-relaxation, and a resume under those is
  not a continuation. The run file's `/config` hash is where that check belongs, and step 3
  adds it.
- **The periodic checkpoint.** Step 0 leaves the checkpoint of a solve's last iterate
  (`steady_checkpoint` after the solve), which is what continuing a capped solve needs; a solve
  that crashes or is killed mid-way loses its iterations. ADR-014 D's rewrite every
  `checkpoint_interval` outer iterations needs a hook inside `solve_steady` (a callback handed
  the checkpoint) and the new solver key, both step 3's.
- `stage_seconds` and `last_pressure_iterations` are per call: a resumed solve's stage times
  are its own part's, so step 3's record of a solve that ran in pieces sums them.
- `/flow/<fingerprint>/` stores a FlowState as the group stores the checkpoint's state, so
  `write_flow_state` can share the layout.

## 7. Deviations from the design text

- **The continuation is `resume=`, not `initial=`.** ADR-014 D said a capped solve continues
  with `initial=` from the checkpoint. The warm start rebuilds nu_t from k and eps, as the
  probe did and as prediction (c) needs, and the next outer iteration reads the under-relaxed
  nu_t, which is not that; and the warm start's rule begins with an empty window, so it cannot
  stop before 100 more outer iterations wherever the interrupted solve would have stopped. Both
  break the bitwise continuation section D's test asks for, so the build keeps `initial=` as
  section A's warm start and adds `resume=` with a `SteadyCheckpoint` carrying what a
  continuation needs (section D's dated note). The contract is in SYSTEM.md section 4.
- **The warm-started faces are held to the rule's tolerance, not to rounding.** A solve
  warm-started from its own converged state keeps converging: the faces move by the iteration
  error the first stop left (section 4's table, and on the test fixtures), which the rule holds
  below `iteration_error_tol` times the velocity scale. The test asserts that bound and that
  the faces did move (section G's dated note).
- **FlowState's stop.** ADR-014 G drafted `FlowState(u, v, p, k, eps, nu_t, stop)`; the built
  type carries `stop_reason` and `outer` in place of one stop record, which is what the solver
  has to give.
- **`src/io_manager.py` exists from step 0**, holding only the steady checkpoint's group, as
  prompt 60 allowed; ADR-014 G lists it for step 3.
- **`io_manager` depends on `solver_staggered`, not on the three modules section G lists.**
  Section G's dependency line reads `io_manager.py -> solver_transport (FieldHistory),
  staggered (FaceVelocities), turbulence (TurbulenceState)`. The built module reads and
  writes a `SteadyCheckpoint` and its `FlowState`, which `src/solver_staggered.py` defines, and
  imports only those two, so SYSTEM.md section 3.1 shows the single edge `io_manager ->
  solver_staggered`. Step 3's `write_flow_state` imports the solver module for the same reason,
  and the edge to `solver_transport` for `FieldHistory` is step 3's own. No cycle: nothing the
  module imports imports it.
- **Section I's items.** None was struck. `scripts/preflight.py` was added; the pipes were
  located by rendering the four reports' tables by GFM's rule (51 rows of 54 tables) and
  escaped; the 19 remote branches section I counted at 809d1c6 had been deleted on the
  remote before this step (after `git fetch --prune` the remote holds one branch besides
  `main`, `docs/ecr-003-pressure-solver`, not merged), so only the 16 merged
  local branches were deleted (`results/builder60/deleted_branches.txt`).
  The docstring reflow covered the docstring and comment lines of the files the batch and the
  step changed under `src/` and `tests/`, of `field43.py`, and of the step's own additions to
  the generators. Review 60 found ten lines it had missed (seven in `tests/test_stopping.py`,
  three in the generators), and prompt 60b reflowed them. Lines over 88 columns that remain
  are code that ruff does not split (an f-string inside a `raise`, a `# type: ignore`
  comment) and the older docstring and usage lines of `tables44.py` and `tables46.py`,
  archived generators whose only change in this branch is the cell escape.
