# Phase 5 step 1: the sensitivity sweep and the clean-out's third grid

**Date:** 2026-10-31 (the machine's local date; the prompt is dated 2026-10-30)
**Branch:** `phase5/step1-sweep` (prompt 72, `docs/prompts/cc-prompt-72-step1-sweep.md`)
**What this is:** ADR-015 section B as accepted (the sweep of the three event scenarios over
the supply's turbulence intensity, its dissipation length and the turbulent Schmidt number),
which is condition 3 of the Phase 4 gate, and the clean-out on 160x60, which is its condition
2. No arithmetic under `src/` changes; the branch's one `src/` hunk is a docstring that review 71
asked for. The records are in `results/builder72/` (untracked, this
machine): the configurations written by `docs/reports/probe72/configs72.py`, the runs made by
`run72.py`, and every table below generated from the run files by `tables72.py`, whose
`--check` regenerates the tables here and in `docs/reports/probe72/tables72.md` and compares.
The tables in this report are the summaries at 0.1 um unless a caption says otherwise; the
per-class tables (R1, D1, W1, P1, C1) are in `tables72.md`. The 320x120 runs at the file's
inputs are the gate's (prompt 69, `results/builder69/`), read by the same script and shown as
the control arm on the product grid.

**In one paragraph.** No sensor's cell is reached by any plume at any arm that ran, on either
grid (D1); twenty-one of the twenty-six runs have a result. The breach's contour comes
nearest: at the `widest` arm on 160x60 it stands about four cells from the two gap sensors,
whose readings are then the first of an event scenario inside the resolved range, a few
hundredths of threshold (P1). The door's corridor air stays on the floor at the arms that ran
and the dust's contour reaches no sensor in any class. Two flow states did not converge within
the solver's cap and are reported, not fixed (section 1.2): the `widest` inputs on 320x120
and the door's open state at the control arm on 160x60, so the product-grid repeat of
`widest` is missing for every scenario, and the door has no reading at its control arm on
160x60 and none at its `sc02` arm on either grid. The 160x60 clean-out's mean 100:1 ratio
falls inside prediction (g)'s band, but the three grids' mean times form an arithmetic series
(section 2): the mean's 100:1 time is not converging over these grids and stays comparative,
as the gate report reads it.

## Predictions, committed before the runs

Committed in 73cd0ac at 07:16:21 on 2026-10-31; the first run started at 07:16:49
(`runs72.json`).

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
ADR-011 G states it, as the probe did on 160x60 (on 320x120 with `t_end` 120 s it is half
the product file's, which was written for 60 s). The five 160x60 breach arms the probe ran
(prompt 70) were run again here and their sensor series are bitwise the probe's
(`results/builder70/`), so the breach's 160x60 rows below are the probe's T6 and T4 with the
`widest` arm added.

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
- each sensor's peak reading over its threshold, stated as a number only when it is at or
  above a hundredth of the threshold; below that it is "below the resolved plume", because
  the tail's value is the grid's and not the inputs' (ADR-015 B, review 70 B3);
- the field's largest value over its threshold per class (F1), which says how far above its
  threshold each plume's contour is drawn, and so how far the grid can move it.

Any sensor the contour reaches on 160x60 at any arm is rerun on 320x120 at that arm before
anything is said about detection. None was reached (D1), so no rerun was triggered.

### 1.2 The runs, and the two flow states that did not converge

<!-- tables72 runs begin -->
| Run | Grid | Flow states: origin, outer, stop, wall (s) | March steps | Run wall (s) |
|---|---|---|---|---|
| filter_breach_control_160x60 | 160x60 | rest, 1,953, error_estimate_and_continuity, 118 | 15,168 | 685 |
| filter_breach_i10_160x60 | 160x60 | rest, 2,024, error_estimate_and_continuity, 122 | 15,125 | 713 |
| filter_breach_max_nut_160x60 | 160x60 | rest, 3,067, error_estimate_and_continuity, 179 | 14,904 | 750 |
| filter_breach_min_nut_160x60 | 160x60 | rest, 2,282, error_estimate_and_continuity, 136 | 15,233 | 693 |
| filter_breach_min_nut_320x120 | 320x120 | rest, 4,477, error_estimate_and_continuity, 938 | 33,346 | 4381 |
| filter_breach_sc02_160x60 | 160x60 | kept, 1,953, error_estimate_and_continuity, 118 | 15,168 | 660 |
| filter_breach_widest_160x60 | 160x60 | kept, 3,067, error_estimate_and_continuity, 179 | 14,904 | 626 |
| filter_breach_widest_320x120 | not run: flow state 13962217d6cf stopped by max_simple_iter after 15000 outer iterations without converging; the march needs a converged state | | | |
| door_leak_control_160x60 | not run: flow state eddc5d9212f0 stopped by max_simple_iter after 15000 outer iterations without converging; the march needs a converged state | | | |
| door_leak_i10_160x60 | 160x60 | rest, 2,024, error_estimate_and_continuity, 120; warm from d9de8992f336, 2,015, error_estimate_and_continuity, 116 | 7,580 | 483 |
| door_leak_max_nut_160x60 | 160x60 | rest, 3,067, error_estimate_and_continuity, 165; warm from 13962217d6cf, 1,699, error_estimate_and_continuity, 98 | 7,470 | 509 |
| door_leak_min_nut_160x60 | 160x60 | rest, 2,282, error_estimate_and_continuity, 131; warm from 5eda1df1e928, 1,914, error_estimate_and_continuity, 107 | 7,635 | 484 |
| door_leak_min_nut_320x120 | 320x120 | rest, 4,477, error_estimate_and_continuity, 952; warm from 5eda1df1e928, 3,784, error_estimate_and_continuity, 790 | 16,714 | 3315 |
| door_leak_sc02_160x60 | not run: its flow state is control's, which did not converge | | | |
| door_leak_widest_160x60 | 160x60 | kept, 3,067, error_estimate_and_continuity, 165; kept, 1,699, error_estimate_and_continuity, 98 | 7,470 | 287 |
| door_leak_widest_320x120 | not run: flow state 13962217d6cf stopped by max_simple_iter after 15000 outer iterations without converging; the march needs a converged state | | | |
| equipment_dust_control_160x60 | 160x60 | rest, 1,953, error_estimate_and_continuity, 110 | 15,167 | 645 |
| equipment_dust_i10_160x60 | 160x60 | rest, 2,024, error_estimate_and_continuity, 118 | 15,125 | 658 |
| equipment_dust_max_nut_160x60 | 160x60 | rest, 3,067, error_estimate_and_continuity, 164 | 14,904 | 660 |
| equipment_dust_min_nut_160x60 | 160x60 | rest, 2,282, error_estimate_and_continuity, 119 | 15,233 | 608 |
| equipment_dust_min_nut_320x120 | 320x120 | rest, 4,477, error_estimate_and_continuity, 940 | 33,346 | 4523 |
| equipment_dust_sc02_160x60 | 160x60 | kept, 1,953, error_estimate_and_continuity, 110 | 15,167 | 590 |
| equipment_dust_widest_160x60 | 160x60 | kept, 3,067, error_estimate_and_continuity, 164 | 14,904 | 556 |
| equipment_dust_widest_320x120 | not run: flow state 13962217d6cf stopped by max_simple_iter after 15000 outer iterations without converging; the march needs a converged state | | | |

The queue ran from 2026-10-31T07:16:49 to 2026-10-31T09:14:50 (machine local time), 1.97 hours of wall clock, at most five runs at a time, each on one BLAS thread (`runs72.json`). Runs that exited without a result, with the minutes each took: filter_breach_widest_320x120 (54.0), door_leak_widest_320x120 (54.0), equipment_dust_widest_320x120 (49.0), door_leak_control_160x60 (14.2).
<!-- tables72 runs end -->

Five runs of the twenty-six have no result, from two flow states that stopped at the solver's
cap of 15,000 outer iterations without meeting the stopping rule (`logs/`, the lines the table
quotes). Both are defects found in a run and are reported here, not fixed; neither is a
property of the arm generator, since the same inputs converge on the other grid.

1. **The `widest` inputs on 320x120** (10% intensity, 0.3 m dissipation length, the room's
   base state, fingerprint 13962217d6cf): the momentum residual cycles between about 1e-5 and
   6e-5 from the first thousand iterations to the cap, with no trend, in all three scenario
   runs that solve it. On 160x60 the same inputs stop by the rule at 3,067 iterations (R1,
   the max_nut and widest rows). The product-grid repeat of `widest` is therefore missing for
   every scenario, and the fine-grid readings of this pass are at `min_nut` only, with the
   gate's file-input runs as the control. The cycle has the shape ECR-002 recorded beside the
   fixed-flow outlets on VAL-020 (ADR-012 D; the gate report's carried limits) and is the
   first seen on the room itself.
2. **The door's open state at the control arm on 160x60** (5%, 0.1 m; the open-door state,
   fingerprint eddc5d9212f0, warm-started from the closed state): from its first thousand
   iterations the residual, printed every fifty, repeats the same three values to seven
   digits (1.42e-7, 6.82e-8, 1.61e-7) to the cap, so it sits in a cycle whose period divides
   150 iterations (the log does not fix it closer). The closed state converges, the open state at the other four arms converges on
   160x60 (R1), and the same open state converges on 320x120 at the gate (3,872 iterations,
   `results/builder69/`). The door's control and `sc02` arms on 160x60 are therefore missing;
   the door's coarse-grid readings are at i10, max_nut, min_nut and widest, and its
   file-input readings are the gate's on 320x120.

Raising the cap would not finish either solve: neither residual history decays, the one
repeating and the other bounded with no trend. What would is a question for a prompt of its
own (the stopping rule's view of a limit cycle, and whether the cycle is the outlets' as in
VAL-020).

### 1.3 The filter breach

<!-- tables72 breach begin -->
Distance from each sensor's cell to the threshold contour, 0.1 um, metres (cells); a sensor whose own cell the contour reaches is marked REACHED:

| Arm | Grid | near_door | above_gap_1 | above_gap_2 | hood_entry |
|---|---|---|---|---|---|
| control | 160x60 | 1.59 (31.8) | 0.35 (7.1) | 0.45 (8.9) | 1.40 (28.0) |
| i10 | 160x60 | 1.56 (31.3) | 0.35 (7.1) | 0.43 (8.6) | 1.40 (28.0) |
| max_nut | 160x60 | 1.50 (30.0) | 0.28 (5.7) | 0.35 (7.1) | 1.31 (26.2) |
| min_nut | 160x60 | 1.60 (32.0) | 0.35 (7.1) | 0.45 (8.9) | 1.40 (28.0) |
| min_nut | 320x120 | 1.62 (64.8) | 0.39 (15.6) | 0.49 (19.7) | 1.45 (58.0) |
| sc02 | 160x60 | 1.55 (31.0) | 0.32 (6.4) | 0.39 (7.8) | 1.35 (27.0) |
| widest | 160x60 | 1.35 (27.0) | 0.18 (3.6) | 0.21 (4.2) | 1.20 (24.0) |
| control (gate, prompt 69) | 320x120 | 1.60 (64.1) | 0.39 (15.6) | 0.47 (18.8) | 1.43 (57.0) |

The same over every class, the shortest to the longest distance (m); REACHED names the classes whose contour reaches the sensor's cell:

| Arm | Grid | near_door | above_gap_1 | above_gap_2 | hood_entry |
|---|---|---|---|---|---|
| control | 160x60 | 1.59 to 1.59 | 0.35 to 0.35 | 0.45 to 0.45 | 1.40 to 1.40 |
| i10 | 160x60 | 1.56 to 1.56 | 0.35 to 0.35 | 0.43 to 0.43 | 1.40 to 1.40 |
| max_nut | 160x60 | 1.50 to 1.50 | 0.28 to 0.28 | 0.35 to 0.35 | 1.31 to 1.31 |
| min_nut | 160x60 | 1.60 to 1.60 | 0.35 to 0.35 | 0.45 to 0.45 | 1.40 to 1.40 |
| min_nut | 320x120 | 1.62 to 1.62 | 0.39 to 0.39 | 0.49 to 0.49 | 1.45 to 1.45 |
| sc02 | 160x60 | 1.55 to 1.55 | 0.32 to 0.32 | 0.39 to 0.39 | 1.35 to 1.35 |
| widest | 160x60 | 1.35 to 1.35 | 0.18 to 0.18 | 0.21 to 0.21 | 1.20 to 1.20 |
| control (gate, prompt 69) | 320x120 | 1.60 to 1.60 | 0.39 to 0.39 | 0.47 to 0.47 | 1.43 to 1.43 |

The plume's width on the sensor rows, 0.1 um: the largest over the frames of the cells at or above the threshold, times dx, and at ten times it, in metres (threshold / 10x):

| Arm | Grid | y = 2.50 m | y = 2.05 m | y = 1.50 m | y = 1.20 m |
|---|---|---|---|---|---|
| control | 160x60 | 1.05 / 0.90 | 1.90 / 1.80 | 0.95 / 0.85 | 0.95 / 0.85 |
| i10 | 160x60 | 1.10 / 0.90 | 1.90 / 1.80 | 0.95 / 0.85 | 1.00 / 0.85 |
| max_nut | 160x60 | 1.20 / 1.00 | 2.10 / 1.90 | 1.05 / 0.85 | 1.10 / 0.90 |
| min_nut | 160x60 | 0.95 / 0.90 | 1.90 / 1.80 | 0.95 / 0.85 | 0.95 / 0.85 |
| min_nut | 320x120 | 0.90 / 0.85 | 1.78 / 1.70 | 0.83 / 0.78 | 0.85 / 0.80 |
| sc02 | 160x60 | 1.10 / 1.00 | 2.00 / 1.80 | 1.05 / 0.85 | 1.05 / 0.90 |
| widest | 160x60 | 1.60 / 1.20 | 2.35 / 2.00 | 1.40 / 1.00 | 1.40 / 1.05 |
| control (gate, prompt 69) | 320x120 | 0.98 / 0.88 | 1.83 / 1.70 | 0.88 / 0.78 | 0.90 / 0.83 |

The field's largest value over the frames and the non-SOLID cells, over the class's threshold (F1 of `tables72.md`): how far above its threshold the contour is drawn:

| Arm | Grid | 0.1 um | 5 um |
|---|---|---|---|
| control | 160x60 | 100 | 99.9 |
| i10 | 160x60 | 100 | 99.8 |
| max_nut | 160x60 | 100 | 99.9 |
| min_nut | 160x60 | 100 | 99.8 |
| min_nut | 320x120 | 100 | 99.9 |
| sc02 | 160x60 | 100 | 99.9 |
| widest | 160x60 | 100 | 99.9 |
| control (gate, prompt 69) | 320x120 | 100 | 99.8 |

Sensor peaks (P1 of `tables72.md`): at or above 0.01 of the threshold, the resolved range, at: above_gap_1 at widest on 160x60, 0.1 um, 0.030 of threshold; above_gap_2 at widest on 160x60, 0.1 um, 0.016 of threshold; above_gap_1 at widest on 160x60, 5 um, 0.029 of threshold; above_gap_2 at widest on 160x60, 5 um, 0.015 of threshold; every other sensor, class and arm is below the resolved plume.
<!-- tables72 breach end -->

The breach is the scenario whose contour comes nearest a sensor. Across the five arms the
probe ran, the gap sensors stand five to nine cells outside the contour (the control to
max_nut rows); the added `widest` arm, the largest supply nu_t over the lowest Schmidt number,
brings it to about four cells and widens the plume on the gap sensors' row by
about half (the width table, y = 2.50 m). At that arm the gap sensors' peak readings are the
first of an event scenario inside the resolved range, at a few hundredths of threshold (P1;
the gate's S1 and the probe's T3 have none), and no cell of theirs is reached. The classes do not differ: the panel carries a hundred times each
class's threshold, so every class's contour is the same set of cells to the cell (the
per-class table). On the product grid the `min_nut` contour stands farther from every sensor
than on 160x60, by up to about one coarse cell, as the gate's control did (section 1.6).

### 1.4 The door leak

<!-- tables72 door begin -->
Distance from each sensor's cell to the threshold contour, 0.1 um, metres (cells); a sensor whose own cell the contour reaches is marked REACHED:

| Arm | Grid | near_door | above_gap_1 | above_gap_2 | hood_entry |
|---|---|---|---|---|---|
| i10 | 160x60 | 0.90 (18.0) | 2.87 (57.4) | 4.62 (92.4) | 5.46 (109.2) |
| max_nut | 160x60 | 0.85 (17.0) | 2.81 (56.2) | 4.55 (90.9) | 5.37 (107.5) |
| min_nut | 160x60 | 0.93 (18.6) | 2.90 (58.0) | 4.65 (93.0) | 5.47 (109.4) |
| min_nut | 320x120 | 0.96 (38.4) | 2.93 (117.0) | 4.70 (188.0) | 5.53 (221.1) |
| widest | 160x60 | 0.74 (14.9) | 2.69 (53.8) | 4.38 (87.7) | 5.18 (103.6) |
| control (gate, prompt 69) | 320x120 | 0.94 (37.4) | 2.90 (115.9) | 4.67 (186.8) | 5.50 (220.1) |

The same over every class, the shortest to the longest distance (m); REACHED names the classes whose contour reaches the sensor's cell:

| Arm | Grid | near_door | above_gap_1 | above_gap_2 | hood_entry |
|---|---|---|---|---|---|
| i10 | 160x60 | 0.90 to 0.90 | 2.87 to 2.87 | 4.62 to 4.62 | 5.46 to 5.46 |
| max_nut | 160x60 | 0.85 to 0.85 | 2.81 to 2.81 | 4.55 to 4.55 | 5.37 to 5.37 |
| min_nut | 160x60 | 0.93 to 0.94 | 2.90 to 2.90 | 4.65 to 4.65 | 5.47 to 5.47 |
| min_nut | 320x120 | 0.96 to 0.96 | 2.93 to 2.93 | 4.70 to 4.70 | 5.53 to 5.53 |
| widest | 160x60 | 0.74 to 0.74 | 2.69 to 2.69 | 4.38 to 4.38 | 5.18 to 5.18 |
| control (gate, prompt 69) | 320x120 | 0.94 to 0.94 | 2.90 to 2.90 | 4.67 to 4.67 | 5.50 to 5.50 |

The plume's width on the sensor rows, 0.1 um: the largest over the frames of the cells at or above the threshold, times dx, and at ten times it, in metres (threshold / 10x):

| Arm | Grid | y = 2.50 m | y = 2.05 m | y = 1.50 m | y = 1.20 m |
|---|---|---|---|---|---|
| i10 | 160x60 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| max_nut | 160x60 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.10 / 0.00 |
| min_nut | 160x60 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| min_nut | 320x120 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| widest | 160x60 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.25 / 0.00 |
| control (gate, prompt 69) | 320x120 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |

The field's largest value over the frames and the non-SOLID cells, over the class's threshold (F1 of `tables72.md`): how far above its threshold the contour is drawn:

| Arm | Grid | 0.1 um | 5 um |
|---|---|---|---|
| i10 | 160x60 | 100 | 100 |
| max_nut | 160x60 | 100 | 100 |
| min_nut | 160x60 | 100 | 100 |
| min_nut | 320x120 | 100 | 100 |
| widest | 160x60 | 100 | 100 |
| control (gate, prompt 69) | 320x120 | 100 | 100 |

Sensor peaks (P1 of `tables72.md`): every sensor of every class at every arm is below 0.01 of its threshold, below the resolved plume.
<!-- tables72 door end -->

The corridor air enters over the lower half of the doorway and falls to the first floor
return: the contour's nearest cell to near_door sits below the sensor and towards the door,
and no sensor row above the hood's holds a threshold cell at any arm (the width table's
zeros; the hood's row, at 1.2 m, holds a cell or a few at the two largest-nu_t arms only). Every sensor is below the resolved plume at every arm. The door's inflow segment
carries the arm's turbulence keys, so `widest` is the widest the door's own plume is made as
well as the supply's. The missing control and `sc02` rows are the capped state of section
1.2; the gate's 320x120 run stands as the control, and lies between the min_nut rows of the
two grids.

### 1.5 The equipment dust release

<!-- tables72 dust begin -->
Distance from each sensor's cell to the threshold contour, 0.1 um, metres (cells); a sensor whose own cell the contour reaches is marked REACHED:

| Arm | Grid | near_door | above_gap_1 | above_gap_2 | hood_entry |
|---|---|---|---|---|---|
| control | 160x60 | 2.14 (42.8) | 0.67 (13.5) | 0.98 (19.7) | 2.45 (49.0) |
| i10 | 160x60 | 2.15 (43.0) | 0.67 (13.5) | 0.98 (19.7) | 2.45 (49.0) |
| max_nut | 160x60 | 2.21 (44.1) | 0.67 (13.5) | 0.98 (19.7) | 2.45 (49.0) |
| min_nut | 160x60 | 2.14 (42.8) | 0.67 (13.5) | 0.98 (19.7) | 2.45 (49.0) |
| min_nut | 320x120 | 1.88 (75.0) | 0.65 (25.8) | 0.98 (39.4) | 2.44 (97.7) |
| sc02 | 160x60 | 2.21 (44.1) | 0.67 (13.5) | 0.98 (19.7) | 2.45 (49.0) |
| widest | 160x60 | 2.36 (47.3) | 0.75 (15.0) | 0.96 (19.2) | 2.41 (48.1) |
| control (gate, prompt 69) | 320x120 | 1.90 (76.0) | 0.65 (26.2) | 0.98 (39.4) | 2.44 (97.7) |

The same over every class, the shortest to the longest distance (m); REACHED names the classes whose contour reaches the sensor's cell:

| Arm | Grid | near_door | above_gap_1 | above_gap_2 | hood_entry |
|---|---|---|---|---|---|
| control | 160x60 | 1.65 to 2.14 | 0.49 to 0.67 | 0.71 to 0.98 | 2.13 to 2.45 |
| i10 | 160x60 | 1.62 to 2.15 | 0.47 to 0.67 | 0.54 to 0.98 | 1.90 to 2.45 |
| max_nut | 160x60 | 1.52 to 2.21 | 0.39 to 0.67 | 0.43 to 0.98 | 1.35 to 2.45 |
| min_nut | 160x60 | 1.66 to 2.14 | 0.50 to 0.67 | 0.89 to 0.98 | 2.36 to 2.45 |
| min_nut | 320x120 | 1.72 to 1.88 | 0.57 to 0.65 | 0.94 to 0.98 | 2.38 to 2.44 |
| sc02 | 160x60 | 1.57 to 2.21 | 0.43 to 0.67 | 0.47 to 0.98 | 1.40 to 2.45 |
| widest | 160x60 | 1.35 to 2.36 | 0.21 to 0.75 | 0.22 to 0.96 | 1.20 to 2.41 |
| control (gate, prompt 69) | 320x120 | 1.70 to 1.90 | 0.54 to 0.65 | 0.77 to 0.98 | 2.16 to 2.44 |

The plume's width on the sensor rows, 0.1 um: the largest over the frames of the cells at or above the threshold, times dx, and at ten times it, in metres (threshold / 10x):

| Arm | Grid | y = 2.50 m | y = 2.05 m | y = 1.50 m | y = 1.20 m |
|---|---|---|---|---|---|
| control | 160x60 | 0.00 / 0.00 | 0.75 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| i10 | 160x60 | 0.00 / 0.00 | 0.75 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| max_nut | 160x60 | 0.00 / 0.00 | 0.75 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| min_nut | 160x60 | 0.00 / 0.00 | 0.75 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| min_nut | 320x120 | 0.00 / 0.00 | 0.60 / 0.00 | 0.10 / 0.00 | 0.00 / 0.00 |
| sc02 | 160x60 | 0.00 / 0.00 | 0.75 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| widest | 160x60 | 0.00 / 0.00 | 0.70 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| control (gate, prompt 69) | 320x120 | 0.00 / 0.00 | 0.65 / 0.00 | 0.05 / 0.00 | 0.00 / 0.00 |

The plume's width on the sensor rows, 5 um: the largest over the frames of the cells at or above the threshold, times dx, and at ten times it, in metres (threshold / 10x):

| Arm | Grid | y = 2.50 m | y = 2.05 m | y = 1.50 m | y = 1.20 m |
|---|---|---|---|---|---|
| control | 160x60 | 0.00 / 0.00 | 1.35 / 1.10 | 0.55 / 0.55 | 0.60 / 0.60 |
| i10 | 160x60 | 0.00 / 0.00 | 1.60 / 1.20 | 0.60 / 0.55 | 0.65 / 0.60 |
| max_nut | 160x60 | 0.00 / 0.00 | 1.95 / 1.75 | 0.95 / 0.80 | 1.10 / 0.80 |
| min_nut | 160x60 | 0.00 / 0.00 | 1.10 / 1.00 | 0.55 / 0.55 | 0.60 / 0.55 |
| min_nut | 320x120 | 0.00 / 0.00 | 0.95 / 0.90 | 0.48 / 0.45 | 0.53 / 0.50 |
| sc02 | 160x60 | 0.00 / 0.00 | 1.85 / 1.70 | 0.95 / 0.60 | 1.00 / 0.65 |
| widest | 160x60 | 0.80 / 0.00 | 2.45 / 2.20 | 1.40 / 1.30 | 1.40 / 1.35 |
| control (gate, prompt 69) | 320x120 | 0.00 / 0.00 | 1.12 / 0.98 | 0.50 / 0.48 | 0.55 / 0.53 |

The field's largest value over the frames and the non-SOLID cells, over the class's threshold (F1 of `tables72.md`): how far above its threshold the contour is drawn:

| Arm | Grid | 0.1 um | 5 um |
|---|---|---|---|
| control | 160x60 | 6.86 | 2.33e+04 |
| i10 | 160x60 | 6.81 | 2.32e+04 |
| max_nut | 160x60 | 6.51 | 2.22e+04 |
| min_nut | 160x60 | 6.94 | 2.36e+04 |
| min_nut | 320x120 | 7.2 | 2.46e+04 |
| sc02 | 160x60 | 6.35 | 2.16e+04 |
| widest | 160x60 | 5 | 1.7e+04 |
| control (gate, prompt 69) | 320x120 | 7.29 | 2.49e+04 |

Sensor peaks (P1 of `tables72.md`): every sensor of every class at every arm is below 0.01 of its threshold, below the resolved plume.
<!-- tables72 dust end -->

The dust is the one scenario whose contour depends on the class, because its source emits the
same number per class while the thresholds span three and a half decades: the 0.1 um field
peaks at a few times its threshold (the F1 table above) and its contour is a patch over the
tool top (W1: no threshold cell on the gap sensors' row at any arm), while the 5 um field
peaks at four decades above its threshold and its contour reaches down the gaps beside the
tool (the 5 um width table). The widest 5 um contour, at `widest`, stands about four cells
from each gap sensor, the distance the breach's does there; no sensor's cell is reached in any
class at any arm.

### 1.6 The contour's grid dependence

<!-- tables72 grids begin -->
The two extreme arms on both grids, 0.1 um: the distance from each sensor's cell to the contour (m), 160x60 against 320x120, and the plume's largest width at the threshold on the gap sensors' row (y = 2.50 m) and the first row over the tool tops (y = 2.05 m). A positive difference means the fine grid's contour is farther from the sensor.

| Scenario | Arm | Sensor | 160x60 (m) | 320x120 (m) | Fine minus coarse (m) | Width at y 2.50, 160x60 / 320x120 (m) | Width at y 2.05, 160x60 / 320x120 (m) |
|---|---|---|---|---|---|---|---|
| filter_breach | widest | 320x120 not run: flow state 13962217d6cf stopped by max_simple_iter after 15000 outer iterations without converging; the march needs a converged state | | | | | |
| filter_breach | min_nut | near_door | 1.600 | 1.619 | +0.019 | 0.95 / 0.90 | 1.90 / 1.78 |
| filter_breach | min_nut | above_gap_1 | 0.354 | 0.389 | +0.035 | 0.95 / 0.90 | 1.90 / 1.78 |
| filter_breach | min_nut | above_gap_2 | 0.447 | 0.493 | +0.046 | 0.95 / 0.90 | 1.90 / 1.78 |
| filter_breach | min_nut | hood_entry | 1.400 | 1.450 | +0.050 | 0.95 / 0.90 | 1.90 / 1.78 |
| door_leak | widest | 320x120 not run: flow state 13962217d6cf stopped by max_simple_iter after 15000 outer iterations without converging; the march needs a converged state | | | | | |
| door_leak | min_nut | near_door | 0.930 | 0.960 | +0.030 | 0.00 / 0.00 | 0.00 / 0.00 |
| door_leak | min_nut | above_gap_1 | 2.899 | 2.926 | +0.027 | 0.00 / 0.00 | 0.00 / 0.00 |
| door_leak | min_nut | above_gap_2 | 4.648 | 4.699 | +0.051 | 0.00 / 0.00 | 0.00 / 0.00 |
| door_leak | min_nut | hood_entry | 5.472 | 5.526 | +0.054 | 0.00 / 0.00 | 0.00 / 0.00 |
| equipment_dust | widest | 320x120 not run: flow state 13962217d6cf stopped by max_simple_iter after 15000 outer iterations without converging; the march needs a converged state | | | | | |
| equipment_dust | min_nut | near_door | 2.138 | 1.875 | -0.263 | 0.00 / 0.00 | 0.75 / 0.60 |
| equipment_dust | min_nut | above_gap_1 | 0.673 | 0.645 | -0.028 | 0.00 / 0.00 | 0.75 / 0.60 |
| equipment_dust | min_nut | above_gap_2 | 0.985 | 0.985 | +0.000 | 0.00 / 0.00 | 0.75 / 0.60 |
| equipment_dust | min_nut | hood_entry | 2.452 | 2.443 | -0.009 | 0.00 / 0.00 | 0.75 / 0.60 |
<!-- tables72 grids end -->

For the breach and the door, the fine grid's contour stands farther from every sensor than
the coarse grid's, by up to about one coarse cell, and the breach's plume on the gap sensors' row is
narrower by one coarse cell (the door's rows hold no threshold cell on either grid): the
coarse distance is a one-sided bound, as ADR-015 B's picture states and as the gate's control
showed. The dust's 0.1 um contour is the exception: on 320x120 its nearest cell to near_door
is a quarter of a metre nearer than on 160x60, a cell at the sensor's own height in the gap
beside the server rack that the coarse grid does not raise above threshold (the gate's control
reaches a cell at the same height, one column over). A field that peaks at a few times its
threshold (F1) has a contour the grid moves by more than a cell, so the dust's 0.1 um
distances are grid-dependent at that size and the bound does not hold for them; the dust's
5 um contour, four decades above its threshold, moves by about one coarse cell between the
grids and in the same direction as the breach's (D1, min_nut rows: the fine grid's farther at
every sensor). The product-grid repeat of `widest` is missing (section 1.2); the coarse
`widest` rows are therefore bounds on the fine grid for the breach and the door only in the
direction the other arms show, not measured.

## 2. The clean-out on 160x60

<!-- tables72 cleanout begin -->
| Run | Grid | Run to t (s) | Well-mixed 100:1 time ln(100)/N (s) | Room recovery time (s) | Free-area mean 100:1 (s) | Ratio to the well-mixed | Free cells | Not recovered at the end, 0.1 um (cells, share) | 5 um |
|---|---|---|---|---|---|---|---|---|---|
| 80x30, step 4 (prompt 64; G1's row, run to 120 s) | 80x30 | 120.0 | 25.6135 | 25.6211 | 77.6564 | 3.0319 | 1,752 | 37 (2.11%) | 37 (2.11%) |
| 80x30, this pass (to 240 s) | 80x30 | 240.0 | 25.6135 | 25.6211 | 77.6564 | 3.0319 | 1,752 | 36 (2.05%) | 36 (2.05%) |
| 160x60, this pass | 160x60 | 240.0 | 25.6135 | 29.2117 | 100.0336 | 3.9055 | 7,008 | 145 (2.07%) | 145 (2.07%) |
| 320x120, the gate (prompt 69; G1's row) | 320x120 | 240.0 | 25.6135 | 27.9433 | 122.2410 | 4.7725 | 28,032 | 581 (2.07%) | 579 (2.07%) |

Observed order of convergence over 80x30 (to 240 s), 160x60 and 320x120 (refinement ratio 2; p = log2 of the ratio of successive differences, defined when the three values change monotonically). The unrecovered cells are counts: they are read in cells of the finest grid, and a difference of one cell of the finer grid of its pair (four fine cells between 80x30 and 160x60, one between 160x60 and 320x120) is the count's resolution, not a convergence; the share in brackets is the count over the grid's free cells.

| Quantity | 80x30 | 160x60 | 320x120 | Observed order |
|---|---|---|---|---|
| Room recovery time (s) | 25.6211 | 29.2117 | 27.9433 | none: the series is not monotone (differences -3.591 then +1.268) |
| Free-area mean 100:1 (s) | 77.6564 | 100.0336 | 122.2410 | 0.01 (differences -22.38 then -22.21) |
| Ratio to the well-mixed | 3.0319 | 3.9055 | 4.7725 | 0.01 (differences -0.8736 then -0.867) |
| Not recovered at 240 s, 0.1 um (cells, share) | 36 (2.055%) | 145 (2.069%) | 581 (2.073%) | none: a difference is at the count's resolution, one cell of the finer grid of its pair (differences -4 then -1 fine cells) |
| Not recovered at 240 s, 5 um (cells, share) | 36 (2.055%) | 145 (2.069%) | 579 (2.065%) | none: a difference is at the count's resolution, one cell of the finer grid of its pair (differences -4 then +1 fine cells) |
<!-- tables72 cleanout end -->

The 160x60 run is the reference clean-out at the file's inputs to 240 s; the gate's 80x30 row
(prompt 64) ran to 120 s, so an 80x30 run to 240 s was added for a like-for-like count of the
cells not recovered at the end (its recovery times are the gate's, as they must be, since both
were reached before 120 s). What the three grids show:

- **The free-area mean's 100:1 time does not converge over these grids.** Its three values
  rise by the same amount at each halving of the cell (the differences in the order table are
  equal to within one per cent), which is an observed order of zero: the series is arithmetic
  in the refinement level, not geometric, so no extrapolation is possible and the value on a
  finer grid is not predicted by these three. The ratio to the well-mixed time is the same
  series divided by a constant. The mean's tail is set by the regions that clear last (the
  gate report's L1 and L2: the pocket beside the bench and the upper corners), and the
  recovery of a cell near a wall or an obstacle corner depends on how the grid resolves the
  near-wall layer there; the three grids resolve it differently in a way that does not yet
  diminish with refinement.
- **The room recovery time is not monotone** (25.6, 29.2, 27.9 s): the 160x60 value is above
  both others. It is set by the near_door sensor, whose cell is a different fraction of the
  way across the flow at each grid (the sensor rule reads the cell below and left of a face),
  and the sensor's 100:1 crossing on a decaying curve moves with that cell. The three values
  lie within 15% of each other; no order is defined.
- **The cells not recovered at 240 s are the same region on the three grids, to within one
  cell of the finer grid at each refinement.** The counts are cell counts. In cells of the
  finest grid they are 576, 580 and 581 at 0.1 um and 576, 580 and 579 at 5 um (the order
  table's last two rows, which print the differences): each refinement's count is four times
  the coarser grid's, plus or minus one cell of the finer grid, which is a count's resolution
  and not a convergence, so no order is observed. What the counts show is the pocket between
  the bench and the right wall, a fixed region of the room, at the same share of the free
  area on every grid to within that resolution. (The first version of this report read a
  second order into the 0.1 um counts; review 72 B1 showed the second difference to be one
  320x120 cell, and its round 2 that the first is one 160x60 cell.)

The condition's purpose was to decide whether the mean's 100:1 ratio and the local recovery
field can be quoted as numbers of the room. The ratio cannot: the three grids give a series
that is still moving by a fifth of its value per refinement with no sign of slowing. The
unrecovered region's share of the free area, about two per cent, is the same on the three
grids to within one cell of the finer grid at each refinement (the three shares in C1 differ
by under two hundredths of a percentage point), so it can be quoted as a share of the room at
that precision; its time to recover cannot, since no grid recovers it by 240 s. The room recovery time
stays the sensor-based number the gate report quotes, with the 160x60 value beside the other
two.

## 3. Predictions against measurement

| Prediction | Measured | Verdict |
|---|---|---|
| (e) Both gap sensors outside the breach contour at `widest` on 160x60 | D1: above_gap_1 and above_gap_2 at 0.180 and 0.212 m (3.6 and 4.2 cells), neither cell reached | Holds |
| (f) The door's corridor air reaches no sensor at any arm | D1: no door sensor cell reached at the four 160x60 arms run, the min_nut 320x120 arm or the gate's control; the nearest contour cell to near_door lies below the inflow's top and towards the door at every arm | Holds for every arm that ran; the control and sc02 arms on 160x60 did not run (section 1.2) |
| (g) The 160x60 clean-out's mean 100:1 ratio between 3.6 and 4.4 | C1: 3.9055 | Holds |
| Builder (e): 0.15 to 0.30 m, three to six cells | 0.180 and 0.212 m, 3.6 and 4.2 cells | Holds |
| Builder (f): at `widest` the contour within about 0.5 m of near_door | 0.743 m | Fails: the plume stays lower than reasoned; the downflow from x = 0.5 m holds it on the floor |
| Builder (g): 4.1 to 4.5 | 3.9055 | Fails: the series is arithmetic (order zero), not first or second order, so the midpoint between the outer grids was the right guess and the convergence reasoning was wrong |
| Builder, the dust by class: the 5 um contour the widest, no sensor cell reached | W1 and D1: the 5 um contour is the widest in every arm; no cell reached | Holds |
| Builder, the grid dependence: no 320x120 distance shorter than its 160x60 reading by more than a cell | Section 1.6: holds for the breach and the door (fine farther by up to a coarse cell) and for the dust at 5 um; fails for the dust at 0.1 um, where near_door's distance is 0.263 m shorter on the fine grid | Fails in one class of one scenario, the contour of a field a few times above threshold |

## 4. The budget

The prompt's budget was 8 hours of wall clock, to be estimated from the first minutes of each
march. The queue's span is the generated line that closes the run table in section 1.2 (under
two hours, five runs at a time). The estimate taken at the first checkpoints of the 160x60
marches (`run72.py --status`) was 21 to 25 steps per second, so a 120 s coarse march of
about 15,000 steps takes about 11 minutes with its flow solve, and the 240 s clean-out about
20 minutes; the product grid's marches ran at the gate's rate (the run table: about 33,000
steps in the hour left after their flow solves). The four runs that exited without a result
(the three `widest` solves on 320x120 and the door's control on 160x60) cost the minutes the
same generated line gives, their 15,000 iterations each, and returned nothing. With all six
fine runs completing the pass would have taken about 2.5 hours; no run was dropped for the
budget.

## 5. What this says for detection

Both halves of the Phase 4 gate's condition 3 are now in: VAL-024 (step 0) and this sweep. On
the room's sensor layout no scenario reaches a sensor's cell at any arm that ran: the breach
and the dust at all six arms on 160x60 and at `min_nut` on 320x120, the door at four of the
six arms on 160x60 (i10, max_nut, min_nut and widest) and at `min_nut` on 320x120, and all
three at the file's inputs on 320x120 (the gate's runs). The contour comes within a few cells
of a sensor in two places, both at the `widest` arm on 160x60 and both at the gap sensors:
the breach's contour, where the sensors' readings are inside the resolved range, and the
dust's 5 um contour, as near but with the sensors' readings below the floor (D1 and P1,
`widest` rows). Three readings were not made (section 1.2): the `widest` arm on 320x120 for
every scenario, so neither of those two corners has a product-grid reading; and the door at
its control and `sc02` arms on 160x60, so the door's `sc02` arm (Sc_t 0.2 on the file's flow
state) is unmeasured on either grid. Where the report reasons past a missing reading it says
so here, and each is an inference from the arms that ran, not a measurement:

- the door's `sc02` arm is taken to lie inside `widest`'s reading, because at the door's
  four coarse arms the distance from near_door to the contour falls as the supply's nu_t
  rises and is shortest at `widest` (the door's distance table), and `sc02` has the control's
  nu_t, a sixth of `widest`'s, at the same Schmidt number, so its plume's diffusivity lies
  between `i10`'s and `max_nut`'s, both measured;
- the door's and the dust's `widest` corners on 320x120 are taken to lie outside their
  coarse readings in the direction the measured pair shows: at `min_nut` the fine grid's
  contour is farther from every sensor than the coarse grid's for the door at 0.1 um and for
  the dust at 5 um (section 1.6 and D1's min_nut rows), the classes whose contours come
  nearest; the dust's 0.1 um contour is excluded from this, since section 1.6 shows the bound
  does not hold for it, and it stands farther from every sensor than the 5 um one at every
  arm (D1).

With those qualifications the "never" of the gate report's table S1 is geometric for the door
across the range, for the dust across the range short of its widest corner, and for the
breach across the range short of its widest corner, the two corners where a gap sensor
stands about four coarse cells from the contour and the product-grid reading is the
measurement the pass could not make. The arrival maps of ADR-015 D carry the placement
question from here: the distances in D1 say which cells a sensor would have to stand in.
