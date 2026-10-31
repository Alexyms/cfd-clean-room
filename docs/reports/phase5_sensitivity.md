# Phase 5 step 1: the sensitivity sweep and the clean-out's third grid

**Date:** 2026-10-30
**Branch:** `phase5/step1-sweep` (prompt 72, `docs/prompts/cc-prompt-72-step1-sweep.md`)
**What this is:** ADR-015 section B as accepted (the sweep of the three event scenarios over
the supply's turbulence intensity, its dissipation length and the turbulent Schmidt number),
which is condition 3 of the Phase 4 gate, and the clean-out on 160x60, which is its condition
2. Nothing under `src/` changes. The records are in `results/builder72/` (untracked, this
machine): the configurations written by `docs/reports/probe72/configs72.py`, the runs made by
`run72.py`, and every table below generated from the run files by `tables72.py`, whose
`--check` regenerates the tables here and in `docs/reports/probe72/tables72.md` and compares.
The tables in this report are the summaries; the per-class tables are in `tables72.md`.

## Predictions, committed before the runs

The orchestrator's (2026-10-30, in the prompt):

- (e) At the `widest` arm on 160x60, both gap sensors stay outside the breach plume's
  threshold contour.
- (f) The door's corridor air reaches no sensor at any arm (it hugs the floor to the first
  return).
- (g) The 160x60 clean-out's mean 100:1 ratio lies between 3.6 and 4.4.

The builder's (2026-10-30, before the runs, scored apart; the reasoning from the records,
`docs/reports/probe70/tables70.md` T4 and T6 and the gate report's G1 and L1):

- (e) Holds. On 160x60 the corner of largest supply nu_t (10%, 0.3 m, Sc_t 0.7) left the gap
  sensors 0.28 and 0.35 m outside the contour, and Sc_t 0.2 on the control's state moved them
  in by 0.03 to 0.06 m. The two together should leave 0.15 to 0.30 m, three to six cells,
  between each gap sensor's cell and the contour; neither cell is reached.
- (f) Holds at every arm. The corridor air enters over the lower half of the doorway at
  0.25 m/s, under the supply's downflow from x = 0.5 m, and the near_door sensor stands
  0.45 m above the inflow's top and 1 m into the room; on the product grid at the file's
  inputs its peak reading was 19 decades under threshold. At `widest` the contour comes
  nearest near_door, within about 0.5 m, and reaches no sensor's cell.
- (g) Holds, at the band's upper end: between 80x30's 3.03 and 320x120's 4.77, a first-order
  series puts 160x60 at about 4.2 and a second-order one at about 4.4, so 4.1 to 4.5.
- The dust release differs by class: its emission is 1e4 per second per metre in every class,
  so the source concentration in the downflow is about twice the 0.1 um threshold and over
  three hundred times the 5 um threshold. The 5 um contour is the widest on every arm and
  reaches no sensor's cell on 160x60 at any arm.
- The grid dependence: as in T6, each 160x60 contour lies about one coarse cell outside the
  320x120 contour at the same arm, so no distance on 320x120 is shorter than its 160x60
  reading by more than a cell, and no sensor the coarse grid leaves outside is reached on the
  fine grid.

## 1. The sweep

### 1.1 The arms and what is read

| Arm | Intensity | Dissipation length | Sc_t | Flow state | Grids |
|---|---|---|---|---|---|
| control | the file's 5% | 0.1 m | 0.7 | solved | 160x60 |
| i10 | 10% | 0.1 m | 0.7 | solved | 160x60 |
| max_nut | 10% | 0.3 m | 0.7 | solved | 160x60 |
| min_nut | 2% | 0.05 m | 0.7 | solved | 160x60, 320x120 |
| sc02 | 5% | 0.1 m | 0.2 | the control's | 160x60 |
| widest | 10% | 0.3 m | 0.2 | max_nut's on 160x60; solved on 320x120 | 160x60, 320x120 |

`widest` is an addition to the accepted design (orchestrator, 2026-10-30): the combination
that spreads a plume most, so the arm that decides whether "no sensor detects" survives the
inputs' range; ADR-015 section B records it. The two extreme arms, `widest` and `min_nut`,
are repeated on 320x120 for every scenario, as decision B (1) states. The arm's intensity and
length are set on the supply and on every event segment that carries the supply's keys: the
breach's three replacement segments (so the breach stays transport-only) and the door's
inflow segment, whose keys the scenario file defines as a copy of the supply's. The
per-cell mass imbalance bound is recomputed for each grid and the scenario's `t_end` as
ADR-011 G states it, as the probe did on 160x60.

Read per scenario, arm and sensor, with the class's threshold as the contour's level in every
scenario (the breach's panel and the door's corridor air carry a hundred times the thresholds,
so T4's "1% of the panel's air" is the threshold itself; the dust source has no inlet
concentration and the threshold is the sensor's):

- the distance from the sensor's cell to the nearest non-SOLID cell ever at or above the
  threshold, centre to centre, in metres and in cells (T6's reading), and whether the sensor's
  own cell is reached;
- the plume's width on each sensor's row at the threshold and at ten times it, at the stored
  frame nearest the event's end (70 s for the breach, 15 s for the door, 40 s for the dust) and
  the largest over the frames (T4's reading);
- the contour's grid dependence from the 320x120 arms: the same two readings on both grids;
- each sensor's peak reading over its threshold, stated as a number only when it is within a
  hundredth of the threshold; below that it is "below the resolved plume", because the tail's
  value is the grid's and not the inputs' (ADR-015 B, review 70 B3).

Any sensor the contour reaches on 160x60 at any arm is rerun on 320x120 at that arm before
anything is said about detection.

### 1.2 The runs

<!-- tables72 runs begin -->
(generated by tables72.py after the runs)
<!-- tables72 runs end -->

### 1.3 The filter breach

<!-- tables72 breach begin -->
(generated by tables72.py after the runs)
<!-- tables72 breach end -->

### 1.4 The door leak

<!-- tables72 door begin -->
(generated by tables72.py after the runs)
<!-- tables72 door end -->

### 1.5 The equipment dust release

<!-- tables72 dust begin -->
(generated by tables72.py after the runs)
<!-- tables72 dust end -->

### 1.6 The contour's grid dependence

<!-- tables72 grids begin -->
(generated by tables72.py after the runs)
<!-- tables72 grids end -->

## 2. The clean-out on 160x60

<!-- tables72 cleanout begin -->
(generated by tables72.py after the runs)
<!-- tables72 cleanout end -->

## 3. Predictions against measurement

(written after the runs)

## 4. The budget

(written after the runs: the estimate from the first minutes of each march against the
prompt's 8 hours of wall clock, and what the pass took)
