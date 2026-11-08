# Phase 5 Completion Report: Alert Monitoring System

**Date:** 2026-11-05, revised 2026-11-07 after review 76, test 76 and Alex's decisions of that
day (prompt 76b), and 2026-11-08 after Alex's decisions of that day (prompt 76c) and round 2 of
review and test 76 (prompt 76d; the machine's local date)
**Phase Gate Verdict:** PROPOSED: CONDITIONAL PASS, for Alex to decide (the section "The proposed
verdict" below)
**Branches Merged:** six pull requests, #98 to #103, and this one (`phase5/step5-gate`):
- The design: #98 (ADR-015, Proposed, accepted by Alex on 2026-10-30).
- ADR-015's build order, section H: #99 (step 0, the coverage debt and VAL-024), #100 (step 1,
  the sensitivity sweep and the clean-out's third grid), #101 (step 2, the alert monitor and
  VAL-009), #102 (step 3, the arrival maps and REQ-A04's latency), #103 (step 4, the placement
  search and REQ-A05).
- This pull request: step 5, the four reference scenarios with their arrival maps, the latency
  report and this report; prompt 76b, the stand, the source rule, the fifth scenario (the door
  seal leak) and the all-clear role; prompt 76c, the stand band from 0.1 m, the all-clear
  sensor's position carried open and the flow-switch marker; and prompt 76d, round 2's findings.

The scenario numbers of this phase live in the tables `docs/reports/probe76/tables76.py`
generates from the run files under `results/builder76/`, `results/builder75/runs/` and
`results/builder72/` (untracked, this machine), copied into
`docs/reports/phase5_alert_report.md` and here between markers; its `--check` regenerates and
compares. The sentences point at the tables and restate a number only where a verdict turns on
it. The test and coverage figures are from one run at c7e9a29.

## Scope

### Planned

`docs/PROJECT_PLAN.md` Phase 5 as first written: sensor probes, threshold monitoring, detection
latency analysis and sensor placement comparison, VAL-009, and the two additional gate criteria.
ADR-015 (prompt 70 and its rounds, accepted 2026-10-30) fixed the design before any code: Alex's
three decisions of 2026-10-29 (a design pass first; Phase 4's spreading condition checked both
ways, by a validation row and a sensitivity sweep, before any detection result; placement by
arrival maps kept during the march), eight decisions put to Alex, and a build order of six steps
with the coverage debt first.

### Delivered

ADR-015 built to its design, with the planned-against-built table below (also in the ADR):

| Step | Prompt, date | Built | Departures and amendments recorded |
|---|---|---|---|
| Design | 70, 70b to 70d, 2026-10-29 and 2026-10-30 (PR 98) | ADR-015; the probes under `docs/reports/probe70/` | VAL-024 moved from a plane jet to a line source in grid turbulence (Alex, review 70 B2); the sweep read on the plume's edge; the threshold Alex's, at 20 and 50 M, with 80 M a carried limit; the spill region stated as outside what the row tests |
| 0, the coverage debt and VAL-024 | 71, 2026-10-30 (PR 99) | A test for each reachable refusal and fallback of Phase 4's Cv2; `validation/line_source.py`, the case file, the harness rows, `tests/test_line_source.py`; REQ-T14 | A defect the debt's tests found and fixed (`write_transient_checkpoint` accepted a None state); one arm left and listed as unreachable; the coverage taken before the last test-only commits, the tester reran it |
| 1, the sweep and the third grid | 72, 2026-10-31 (PR 100) | `docs/reports/phase5_sensitivity.md` and its generator; the clean-out on 160x60 | A sixth arm `widest` added by the orchestrator; five of twenty-six runs have no result, from two flow states stopped at the cap, reported and not fixed (carried below) |
| 2, the alert monitor | 73, 2026-10-31 (PR 101) | `src/monitor.py`, the march hook, `/transport/alerts/`, the runner's summary, the iso_class form; VAL-009 | The state carries every alert, not only the open ones, since the table is replaced whole; it rides in the checkpoint's existing named arrays; no sensors, no monitor |
| 3, the arrival maps | 74, 74b, 2026-11-01 (PR 102) | `ArrivalTracker`, `/arrival/`, restart, `latencies(onsets)`, REQ-A04; `arrival_from_frames`, `arrival_from_run` | `latencies` moved to section D's rule so the monitor and the map agree; a `set` giving a segment a concentration counts as contaminating; the frames map misses between-frame cells, so the criterion "within one output interval at every cell" was changed to what holds; the input checks moved to `io_manager`; `monitor` imports `io_manager` |
| 4, placement | 75, 75b, 2026-11-02 and 2026-11-03 (PR 103) | `latency`, the units, `score`, `candidates`, `search`, `rank`; the `placement` section; REQ-A05 | The unit is (scenario, event, class); `score` has no `rule`; the sweep is bisected and the programme runs on the maximal masks; the fallback reads as no layout of the count covering every coverable unit, and no coverable unit at all raises; `_maximal` chunked after a 4.7 GB peak on a synthetic map; the section out of the restart hash (orchestrator); frame-resolved units flagged; REQ-A05's exactness restated as built |
| 5, the runs, the report, the gate | 76, 2026-11-05 (this pull request) | The four scenarios on 320x120 with their maps; `docs/reports/phase5_alert_report.md` and `probe76/`; this report | A cell within the standoff of an opening is not a candidate (item 0), found before the runs by the prompt; the planned `scripts/latency_report.py` is `tables76.py`, as section E names it; the copies of the scenario files add a checkpoint every 1000 steps and nothing else |
| 5, revised: two roles | 76b, 2026-11-07 (this pull request) | Alex's three decisions of 2026-11-07: the `stand` mount and `stand_height`; a source box and its standoff never a candidate; the all-clear role (`placement.all_clear`, `clear_units`, `clear_score`, `search_roles`). The candidates over every scored scenario (review 76 S5). The seal leak, the fifth scenario (orchestrator), run on 320x120; the other four re-scored from step 5's files | Not planned in ADR-015: the role, the stand and the seal leak are new decisions recorded in its section D. The all-clear search is sequential, not joint. A run file's recorded mapping merges over the base file as it is now, so a stand band given without a stand is kept rather than refused (else every earlier run file failed to reload) |
| 5, revised: round 2's findings | 76d, 2026-11-08 (this pull request) | `candidates` refuses configurations of another room and ignores a stand band without a stand; tests that separate the classes of the all-clear computation, pin the stand rule, the sequential search, the refusals of `add_all_clear`, the marker at an onset and the seal leak's file | No run, no new module. `candidates`' contract in SYSTEM.md changes (the room check, the band) |
| 5, revised: the band and the open all-clear | 76c, 2026-11-08 (this pull request) | Alex's two decisions of 2026-11-08: `stand_height` 0.1 to 1.2 m and `all_clear` 0 in the product file and its three copies, the all-clear code and tests kept; `flow_switches` and `after_flow_switch`, the latency tables' marker "after the flow switch"; the five runs re-scored, the 0.3 m band's search and the three-plus-one kept as one row each | No run. `placement` imports `scenarios.flow_fingerprint` for the marker. Prediction (q) committed before the re-score and holds |

Also added without plan entries, each to measure the build: the probe directories
`docs/reports/probe70/`, `probe72/` and `probe76/`.

### Deferred

- **The counter's sample time** stays OPEN (ADR-015 C): an additive instrument delay until a
  counter is specified.
- **The seal leak on the coarser grids.** The 0.025 m slot covers no face of 80x30 and one 0.05 m
  face of 160x60, which then carries twice the flux, so the scenario is a 320x120 one and has no
  grid pair or sweep arm.
- **The ranking across the `widest` inputs on the product grid**: that state does not converge
  (carried below).
- **A layout searched for detection and clearing together.** The search is sequential (the
  detection sensors first, then the all-clear one); a joint search is not built.
- **Where an all-clear sensor goes** (Alex, 2026-11-08): carried open, the product file stating
  none. It needs a cost for clearing late, and a clean-out that runs until the room clears or a
  stated room-clean criterion.

## Test Results

One run: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pytest tests/ -v --cov=src --cov-branch
--cov-report=term-missing --cov-report=json:...` at c7e9a29 on 2026-11-08, the only run of the
build on the machine (`results/builder76/full_suite76c.txt`, untracked). The per-tier counts are
`pytest tests/ --collect-only -q -m <tier>` at the same commit; every collected test carries one of
the three markers. The branch changes nothing under `src/` or `tests/` after that commit. The runs
at a8549ab (step 5, 2,908 tests) and aefff78 (prompt 76b, 2,967) are superseded.

| Tier | Collected | Passed | Failed | Skipped or xfailed | Phase 4 (5aabe1d) |
|---|---|---|---|---|---|
| Unit | 2590 | 2590 | 0 | 0 | 1984 |
| Integration | 311 | 311 | 0 | 0 | 273 |
| Validation | 80 | 80 | 0 | 0 | 61 |
| All | 2981 | 2981 | 0 | 0 | 2318 |

The suite took 4045 s (1328 s for prompt 76b's run at aefff78; the machine's other load was not
recorded, so the time is not a measure of the code).

### Validation Tests

| ID | Description | Criterion | Status | Where |
|---|---|---|---|---|
| VAL-009 | Alert latency | The alert's step equals the first step at or above threshold, exact, on planted series and an 80x30 release; the march calls the monitor every step; the fields unwritten; a restart across an open alert bitwise | PASS | `tests/test_alert_latency.py`; six planted defects caught (prompt 73) |
| VAL-024 | A line source in decaying grid turbulence (Warhaft's source at 52 M) | The decay exponent within 0.1 of the model's 1.087; the plume's width over the measured within 20% at 20 and 50 M (Alex, 2026-10-30) | PASS: exponent 1.078 on 280x50 and 1.077 on 560x100; 1.033 and 0.853 at 20 and 50 M on 280x50, 1.030 and 0.852 on 560x100. Reported, not scored: 1.30 at 10 M, 0.784 at 80 M, the carried limit | `tests/test_line_source.py`; the harness rows `val024_280x50` and `val024_560x100` |

The earlier phases' rows: every test-carried row of Phases 1 to 4 is in the suite above. The six
harness rows the Phase 4 gate reproduced solve to the same face hashes at this source (table H1),
and table X1 finds this gate's four step 5 runs equal to the Phase 4 gate's, frames, budgets
and sensor readings, bitwise; the seal leak is new and its closed state equals the gate's rest
state to the bit (F2 of the latency report). Phase 5 changed no solver module: its source changes are
`config`, `io_manager`, `monitor`, `placement` and `time_integration`, the last around the march
and not in its arithmetic. The turbulent harness rows stand on prompt 68's check of every hash of
record and on the closed room's face hash, equal to the record's in table F2 of the latency
report, not on a separate solve at this commit.

<!-- tables76 hashes begin -->
### H1. The hashes of record at the gate's source (hash62.py, against the Phase 4 gate's _69 records)

| Case | Commit | Stop | Outer | Hash (first 16) | Equal to _69 | Wall (s) | Processes |
|---|---|---|---|---|---|---|---|
| val001_80x40 | f50206d | error_estimate_and_continuity | 1,559 | f2e5b888cb269112 | yes | 23 | 10 |
| val001_80x40_stretched | f50206d | error_estimate_and_continuity | 1,124 | 4c79ffb9def55fbb | yes | 19 | 10 |
| val002_20x20 | f50206d | error_estimate_and_continuity | 970 | 1ecbedb180303c8f | yes | 6 | 10 |
| val002_40x40 | f50206d | error_estimate_and_continuity | 3,435 | 7a881aaa88b1946d | yes | 43 | 10 |
| val002_80x80 | f50206d | error_estimate_and_continuity | 12,814 | 95efc0123aaabc55 | yes | 310 | 10 |
| product_80x30 | f50206d | error_estimate_and_continuity | 1,021 | 6f27f27f3e89963a | yes | 24 | 10 |
<!-- tables76 hashes end -->

<!-- tables76 bitwise begin -->
### X1. The runs against the gate's (prompt 69), the same scenario files before Phase 5's modules: stored frames, budgets and sensor readings

| Run | Frames here, the gate's | Fields, every class, bitwise | Budgets, bitwise | Sensor readings, bitwise |
|---|---|---|---|---|
| door_leak | 167, 167 | equal | equal | equal |
| filter_breach | 333, 333 | equal | equal | equal |
| equipment_dust | 333, 333 | equal | equal | equal |
| seal_leak | no gate run (a scenario of prompt 76b) | | | |
| clean_out | 664, 664 | equal | equal | equal |
<!-- tables76 bitwise end -->

### Coverage

From `results/builder76/coverage.json` (the same run) and `results/builder71/after/coverage.json`,
the step 0 run that the plan names as Phase 4's after the debt was paid.

<!-- tables76 coverage begin -->
### Cv1. Coverage, statements and branches apart, the gate against Phase 4's after step 0

| Measure | The gate | After step 0 (prompt 71) | Covered count, change | Percentage, change (points) |
|---|---|---|---|---|
| Statements covered | 7,079 of 7,079 (100.00%) | 5,807 of 5,807 (100.00%) | +1,272 | +0.00 |
| Branches covered | 2,541 of 2,542 (99.96%) | 2,005 of 2,006 (99.95%) | +536 | +0.01 |

Files with an uncovered statement: none. Uncovered branch arms: `src/time_integration.py` 613->616.
<!-- tables76 coverage end -->

Statements are at 100% and branches a hair above step 0's percentage (Cv1). The one uncovered arm
is `time_integration.py` 613->616, step 0's (552->555 then; the lines above it have moved),
unreachable because two unequal mappings differ at some key. Earlier runs each found one line or
arm the pass had just written and no test reached (`placement.py` 1226 at f50206d, step 5's item 0;
`clear_units`' second run on one mesh at a334264, prompt 76b); a test was added each time and the
run repeated. Prompt 76c's code was covered on its first run.

## The latency report

`docs/reports/phase5_alert_report.md` holds the tables section E lists, and its predictions. In
short: the file's four sensors detect none of the twenty event units (four scenarios, five classes).
On the mounting surfaces, with the dust's source box excluded, a searched four detect the breach
and the dust and neither the door nor the seal leak. With a stand (an inlet 0.1 to 1.2 m above
the floor) the product search detects all twenty within the breach's worst case, the seal leak
within a second from a stand just above the band's floor; at the earlier 0.3 m floor the seal
leak was detected tens of seconds late (L1, U2). Every layout that detects declares the room clean
early (C2). In the clean-out the room does not clear by 240 s, so the all-clear rule's choice is
the run's end, and the position is carried open. The ranking holds on 80x30 and 320x120, except
that on 80x30 the middle of the order inverts by candidacy (K1: the layout searched on 320x120
beats 80x30's own, three of its four cells not being candidates there), and at every sweep arm that
ran (K1 and K2, with a free row per arm); the latencies are comparative and do not carry over
between grids. Prompt 76's three predictions hold as step 5 scored them; of prompt
76b's five, (n) fails; prompt 76c's (q) holds (P1 to P3).

## Phase Gate Criteria

1. **All planned validation tests pass.** Holds: VAL-009 and VAL-024, the rows above, and every
   test-carried row of the earlier phases in the suite.
2. **All unit and integration tests pass.** Holds: every collected test passed in the one run at
   c7e9a29, none failed or skipped.
3. **No ruff errors.** Holds: `ruff format --check .` and `ruff check .` pass at HEAD.
4. **Coverage does not decrease.** Holds: 100.00% of statements against step 0's 100.00%, 99.96% of branches
   against its 99.95% (table Cv1), the measure the coverage debt's payment set.
5. **SYSTEM.md is current.** Holds on the generated regions (`scripts/gen_system_map.py --check` at
   HEAD) and on the hand-authored ones: the contracts of `monitor`, `placement`, `io_manager` and
   `config` were brought current at each step's pull request, and the Document History carries this
   gate.
6. **The report is filled out and committed.** This report; the verdict line at its head is
   proposed, and Alex sets it, as for Phases 3 and 4.

The plan's additional gate criteria for Phase 5:

- **The latency report for all scenarios with at least two sensor configurations.** Holds: the
  four event scenarios, per event, class and sensor, for six configurations (the file's four, the
  searched four without stands, with stands at the earlier 0.3 m band and with the product's 0.1 m
  band, the three-plus-one and the free optimum for one to four sensors), with every
  configuration's all-clear reading over the five scenarios. Generated from
  the run files, with a check mode.
- **The monitor does not modify the concentration fields.** Holds, by test and by construction.
  The march hands the monitor and the arrival tracker fields already made read-only;
  `tests/test_alert_latency.py::test_the_fields_are_read_only_and_unchanged_across_every_update`
  records every update's flags and bytes in a coarse run, for both instruments, and
  `test_a_writeable_field_is_left_bitwise_as_it_was` and
  `tests/test_monitor.py::test_update_leaves_a_writeable_field_bitwise_as_it_was` hold the same
  for a writeable field. Table X1 reads it at the scale of the product grid: every class's frames
  bitwise equal to a run made before the monitor existed.

Requirements REQ-A01 to A06, each with its verification, are in SYSTEM.md section 2.4; REQ-A04's
monitor against the map on the product grid is table A1 of the latency report.

## The carried limits

Phase 4's conditions first, then what this phase carries.

- **Phase 4's condition 1, the budget residual's mechanism, is still open** and due before Phase
  6's finer grids. It was the one condition not due in Phase 5 (ADR-015 F). The residual is
  unchanged to the bit (X1), inside REQ-T05 by 16 times at worst over the stored frames (table B1
  of the latency report), and bears on no latency: its size is B1's, against thresholds at 1e-2
  of the inlet value. The other two conditions were met by step 1.
- **The spill region rests on VAL-020.** The breach's and the dust's latencies are set here: the
  searched layouts' cells for them stand on the litho tool's top, where VAL-024's row does not test
  the spreading and the basis is VAL-020's excess turbulence over the tops, an inference and not a
  measurement. This is the carried limit that bears most on the detection result.
- **Separation bubbles short on practical grids (VAL-019),** Phase 3's condition, carried as Phase
  4 carried it (test 76 S5). The door's and the seal leak's corridor air stays in the recirculation
  beside the doorway, and the room's clear time in both is set at the doorway's foot (C0); the
  breach's and the dust's last cells lie in the pocket beside the litho tool's left side; the
  clean-out's never-clearing pocket is beside the bench. A shortened bubble clears sooner than a
  longer one would, so the model's clear times there are early, not late, and the premature
  intervals are read as the model's.
- **L2's cycle on 160x60,** Phase 3's condition (test 76 S5). The scenarios run on 320x120 and
  the layouts L1 and L2 are not run in Phase 5; a layout comparison of L2 on the coarse grid would
  carry the cycle.
- **VAL-024's 80 M limit.** The model's plume is about a fifth narrower than measured far
  downstream, near the returns (0.784 at 80 M). No working cell of a searched mountable layout
  stands there.
- **The two unconverged flow states** (the sweep's section 1.2): the `widest` inputs on 320x120 and
  the door's open state at the control arm on 160x60, both stopped by the 15,000 cap in a bounded
  or repeating residual. So the ranking has no product-grid reading at the widest inputs and the
  door none at its control and `sc02` arms on 160x60. The product's own inputs converge, and every
  state of this gate's runs did (F1 of the latency report). OPEN in the plan's deferred table.
- **The 2D slice.** Every latency is comparative between layouts on one grid: a line source,
  exchanges per metre of depth. A sensor on a line in the slice stands for a line of sensors in
  the room.
- **The steady extreme, and the door's and the seal's exchange values** (ADR-014 A, B): the door
  open is its fully open state for the whole 5 s, and its 0.25 m/s, its split and its corridor
  concentration are assumptions; the seal leak's 10 mm undercut at 2 m/s for 30 s is one too, and
  its slot is one product-grid cell. The finding that no sensor on a surface detects either, and
  that a stand does, is a finding on those.
- **The clean-out does not clear by 240 s.** The pocket between the hood's bench and the right wall
  is above threshold at the run's end in every class (Phase 4 found it not recovered by 240 s). The
  all-clear role's ranking treats that as unbounded for every layout that clears, so the run's end
  would decide an all-clear sensor's place; the position is carried open (the latency report's
  section 5).
- **The sweep's width limit and the frames' resolution.** The ranking's reading across inputs (K2)
  is frame-resolved, and its ties are ties to about a frame spacing.
- **The clean-out's mean stays comparative** (the sweep's section 2), as does the local recovery
  field the layouts' recovery is read from: a series that has not converged over three grids.
- **The candidates are a stated assumption:** the mounting list, the 0.05 m standoff, the stand's
  0.1 to 1.2 m band and the exclusions. The seal leak's detection rests on the band's floor (L1's
  0.3 m row). The source box and its standoff are now excluded, but the
  dust is one release position: a sensor just beyond the ring still reads it within a frame,
  so the dust's latency is still a statement about where the box is. The exact search is at its
  20-unit cap (decide item 2), so more release positions cannot be scored exactly as built.

## Implementation Decisions

Each is recorded where it was made; ADR-015 holds the formal record and carries the
planned-against-built table.

1. **Alex's decisions:** the three of 2026-10-29; VAL-024 as a line source; the threshold of
   2026-10-30; B to H as recommended.
2. **The orchestrator's during the build:** the `widest` arm (2026-10-30); the placement section out
   of the restart hash (2026-11-03); the opening rule (prompt 76 item 0); the seal leak as a fifth
   scenario (2026-11-07).
3. **Alex's of 2026-11-07** (prompt 76b): a sensor is a counter's inlet and a stand is a mounting
   option; a source box is not a candidate; detection and all-clear are separate roles, one of the
   four sensors in the second until 2026-11-08.
4. **Alex's of 2026-11-08** (prompt 76c): the stand band is 0.1 to 1.2 m; the all-clear sensor's
   position is carried open, the product file's `all_clear` 0, the code and its tests kept.
5. **The ranking of a "not clear"** (prompt 76b, ADR-015 D): a layout that does not clear never
   declares the room clean and ranks as zero; one that clears where the room does not ranks above
   every number. The definitions in the prompt left the two cases to the build.
6. **The unit is (scenario, event, class)** (prompt 75): every event of a scenario with several is
   a unit to detect.
7. **The fallback** (prompts 75 and 75b): when no layout of the count detects every coverable unit,
   the worst case is infinite for all and the layout is the mean rule's, and it says so.
8. **The flow-switch marker** (prompt 76c): a latency is marked when its arrival falls strictly
   after an interval boundary at which an event has ended and the flow fingerprint changed, at or
   after the unit's onset. An arrival on the boundary itself was reached on the step taken on the
   state before it, so it is not marked.
9. **Round 2 of review and test 76** (prompt 76d, the orchestrator, 2026-11-08): the round found
   as many findings as round 1 and every one was in code 76b or 76c added, so one fix pass with
   no round 3. The Bug, a missing guard on the all-clear computation's per-class thresholds, is
   fixed and the plants that survived are pinned.
10. **Measure before building, predictions before runs.** Each step committed its predictions before
   its runs or its re-score. (e), (f), (g), (i), (j), (k), (l), (m), (o), (p) and (q) hold; (h)
   failed on the mountable set on 80x30, and that failure is what led to reading the door on the
   product grid; (n) failed: the seal leak's air rises above the 0.3 m band's lower limit.

## Deviations from Architecture

- **Modules.** Two new, `monitor` and `placement`, as ADR-015 H planned. `monitor` imports
  `io_manager` and `placement` imports `boundary_registry` and `staggered` (the candidates' surfaces
  and openings), beyond H's list; section 3 of SYSTEM.md is generated from the source.
- **Interfaces.** `score` without `rule`; `search` taking `Units`; `RunArrival`, `Units`, `Score` and
  `Layout` as SYSTEM.md section 4 states them; `latencies` on section D's rule, not section C's
  first contract line. Prompt 76b: `candidates` takes a sequence of configurations and a
  `stand_height`; the all-clear functions and `ClearUnits`, `ClearScore` and `Roles` are new.
  Prompt 76c: `flow_switches` and `after_flow_switch`, for which `placement` imports
  `scenarios.flow_fingerprint`. Prompt 76d: `candidates` refuses a configuration of another room
  (grid, domain, obstacles) and ignores a `stand_height` given without `stand`.
- **Configuration.** The optional `placement` section, in the product file and its three layout
  copies, outside the flow inputs and the restart hash, with `stand`, `stand_height` and
  `all_clear` since prompt 76b (the band 0.1 to 1.2 m and `all_clear` 0 since prompt 76c);
  `thresholds: {iso_class: N}`; `configs/scenario_seal_leak.yaml`.
- **Requirements.** REQ-A01, A03 and A05 reworded at acceptance; REQ-A04 given its onset; REQ-A05's
  exactness restated as built; REQ-T14 added.
- **The extension point** `v_ext` is preserved and passed nothing (ADR-007).

Each change was recorded in SYSTEM.md's Document History when it was made.

## Lessons Learned

- **A rule that holds on one grid can be wrong on the next.** "Beside the face" and "within the
  standoff" are the same on 80x30 and differ on 320x120. The prompt caught it by thinking about the
  doorway before the runs, not a test; a candidate rule needs a test on the grid it will be read on.
- **Coverage found a line this pass wrote.** The side vent's face was added for item 0 and reached by
  no test until the suite's coverage said so.
- **A count of working cells says more than a layout.** Step 5's searched four were a searched two
  with two spares, and the spares set the layout's recovery.
- **What detects is not what clears.** The working cells stand where the plumes are and the supply
  scours, so they clear early.
- **Rerunning at the gate paid for itself.** Table X1's bitwise equality is a stronger statement about
  Phase 5's arithmetic than any review of the diff could give.
- **A run file is not self-contained.** Its recorded mapping merges over the base file on disk at
  reload, so a key added to the product file reaches every earlier run file. A dry run of the
  tables before the seal leak finished found the new refusal breaking every one of them; no test
  reloaded an old record.
- **A role's rule can be decided by a run's length.** The all-clear search put its sensor where the
  clean-out had not cleared by its end, not where the event scenarios clear last; a score that
  counts only one side of an error rewards never answering.
- **A fixture can hide what a test claims.** The all-clear tests' frames put every value on the
  same side of both classes' thresholds, so an assertion that class 1 was read against its own
  threshold could not fail. A test of a per-class rule needs a value between the thresholds.
- **A band's floor is a detection decision.** Lowering the stand's floor by 0.2 m moved the seal
  leak's detection from tens of seconds, one class after the event had ended, to under one.

## The proposed verdict

The builder's proposal is CONDITIONAL PASS, carrying one condition of Phase 4's. The case, for Alex
to decide:

**Why not FAIL.** Both planned validation rows pass, the full suite passes, lint and the system map
pass, SYSTEM.md is current, the latency report exists for all scenarios with six configurations,
and the monitor does not modify the fields, by test and at the product grid's scale. Every flow state
of the runs converged, and nothing found here is a wrong computed result against a stated contract.
Coverage is at step 0's figures or above (criterion 4).

**Why not an unconditional PASS.** Phase 4's condition 1, the budget residual's mechanism, is open
and was set to be found before Phase 6's finer grids. Phase 6 is next, so the condition belongs on
this gate's record and is the one thing that must happen before the next phase's grids. The two
unconverged states are not proposed as a condition: no Phase 5 reading at the product's inputs rests
on them, and the plan's deferred table carries them. Nor is the spill region (review 76 S4): it is a
limit of the turbulence model over the tool tops that no Phase 5 measurement can remove, VAL-020
carries it from Phase 3 and Phase 6's finer grids do not change it, so a condition on this gate
would have nothing to meet; it bears on where the breach's and the dust's detectors stand and on
their latencies, which are claimed only as a ranking.

**What Alex may want to decide beside the verdict.**

1. **The door and the seal leak.** On these assumptions no sensor fixed to a wall, the ceiling or a
   tool, with its inlet within about 0.7 m of the surface, detects a door opening in any class (at
   larger standoffs the openings' exclusion holds the plume, so the zero is by construction), and
   none detects the seal leak. A stand does: the door from 0.56 m, the seal leak from a stand just
   above the 0.1 m floor within a second (L1, U2). The band's floor decides the seal leak: at
   0.3 m it was read tens of seconds late. Whether a counter's inlet can stand that low by a
   doorway is a question about the room, not the model.
2. **The dust rests on one release position.** With the box and its standoff excluded, a sensor just
   beyond the ring still reads the release within a frame, so the dust's latency still says where
   the box is. The proposed remedy is to score over several release positions per tool top.
   That adds units, positions times five classes, and the exact worst-case search stops at 20
   units (`MAX_SET_COVER_UNITS`, 2^20 subsets, a megabyte of flags): the product search is at
   that cap now (four scenarios, five classes), so a second release position raises
   `ValueError`. The remedy needs one of two things first: the mean rule's greedy search with
   its one-sensor exchanges, which is not exact for the worst case, or a larger cap whose memory
   and time are measured first (the subsets double with each unit, 2^25 is 32 megabytes of
   flags, and the time has not been measured). Which one is Alex's choice beside the remedy.
3. **An all-clear requirement, proposed and not applied to SYSTEM.md.** "The placement module shall
   report, for a layout, each all-clear unit's layout clear time, the room's clear time with the cell
   that sets it, and the premature interval between them, frame-resolved with the spacing stated,
   and shall search a stated number of all-clear sensors added to the detection layout." Verified
   by `tests/test_placement.py`'s all-clear classes. The finding that stands under it is the
   premature interval: every detecting layout declares the room clean early, up to about a minute
   (C2).
4. **Where an all-clear sensor goes, carried open** (Alex, 2026-11-08). The score counts only
   clearing early, so a cell that never clears scores perfectly, and the clean-out's 240 s ends
   before its room clears; the three-plus-one's fourth cell is set by that run end, not by where
   clearing is read well. The open item needs a cost for clearing late, and a clean-out that runs
   until the room clears or a stated room-clean criterion.

Nothing above is reported as an absolute latency for the room. The claim is the ranking on the model,
stated per rule with ties, with the limits above beside it.

## Sources

1. `docs/ADR/ADR-015-alert-monitoring-and-placement.md`: the decisions, the dated notes per section,
   "Planned against built", the probes T1 to T12.
2. `docs/PROJECT_PLAN.md`, Phase 5: the deliverables, the validation gate, the additional criteria.
3. `docs/reports/phase5_alert_report.md` and `docs/reports/probe76/tables76.md`, generated by
   `tables76.py` from `results/builder76/`, `results/builder75/runs/`, `results/builder72/`,
   `results/builder69/` (X1 and F2) and `results/builder62/hash_*_76.json` (H1), all untracked;
   `run76.sh`, `configs76.py` and `hash76.sh`.
4. `docs/reports/phase5_sensitivity.md`: the sweep and the clean-out's third grid.
5. `docs/reports/phase4_scenarios_report.md`: the conditions and the runs compared.
6. `docs/SYSTEM.md` sections 2.4, 3 and 4 and the Document History.
