# Phase 5 step 5: the latency report

**Date:** 2026-11-05 (the machine's local date)
**Branch:** `phase5/step5-gate` (prompt 76, `docs/prompts/cc-prompt-76-step5-gate.md`)
**What this is:** ADR-015 section E: the four reference scenarios on 320x120 with their arrival
maps and alert tables, the layouts scored and searched on them, and what the ranking can claim.
The runs are made by `docs/reports/probe76/run76.sh`, the records stay in `results/builder76/`
(untracked, this machine), and every table is generated from the run files by
`docs/reports/probe76/tables76.py`, whose `--check` regenerates and compares. The tables here
give the 0.1 um and 5 um classes where a caption says so; every class is in
`docs/reports/probe76/tables76.md`.

**In one paragraph.** The file's four sensors detect nothing in any of the three event
scenarios on the product grid. A searched layout on the mountable surfaces detects the breach
and the dust release in every class, and no mountable position at any standoff detects the door:
the corridor air stays low beside the doorway, where nothing can be mounted, and leaves by the
first floor return. Two cells of the searched four do all the work, one over the litho tool's
left corner and one inside the dust source's box, so both rest on the spill region's carried
limit. The ranking (free optimum, then searched, then the file's four) holds on both grids
pairs read, under both rules and at every sweep arm that ran; the latencies themselves do not
carry over between grids and are not claimed. The searched layout also reads the room clean
sooner than the file's four on its working cells, so a layout chosen to detect is not a layout
chosen to clear.

## Predictions, committed before the runs

Committed in a68d737 at 19:34:05 on 2026-11-05; the runs started at 19:34:10 (table R1).

The orchestrator's (2026-11-05, prompt 76), copied verbatim:

- (i) On 320x120 the file's four sensors detect none of the event scenarios' units.
- (j) With the doorway rule of item 0, no mountable candidate detects a door unit on 320x120
  either: the corridor air is taken by the first return before it reaches a wall a sensor can
  be mounted on.
- (k) The searched mountable four detect every breach and dust unit on 320x120, with a worst
  case under 3 s.

## 1. Before the runs: a sensor cannot hang in a doorway

The rule as first built excluded only the cells beside an opening's face and counted a domain
edge whole as a wall. On 80x30 the 0.05 m standoff holds one cell, so nothing showed. On
320x120 it holds two, and with the door open the built rule made 84 cells of column 1 in the
doorway's rows wall-mounted candidates, and likewise 280 cells of the second row under the
supply and 36 of the second column of the hood's exhaust. The rule is now that a mounting
surface is a face whose boundary is a wall in every interval of the scenario, and a cell within
the standoff of an opening is not a candidate, the free set included (ADR-015 D, "Openings";
`test_a_sensor_cannot_hang_in_the_open_doorway_on_the_product_grid`, whose plant is the
exclusion read as beside the face alone). The candidates here are the door scenario's: its
intervals open every opening the three event scenarios have.

## 2. The runs

The four files under `configs/` as they are on `main`, each loaded from a copy that adds a
checkpoint every 1000 steps (`docs/reports/probe76/configs76.py`), run concurrently, each from
rest, through `scripts/run_scenario.py`, one BLAS thread each. No stop condition fired: every
flow state converged, every budget is inside REQ-T05 at every stored frame, and the pass took
1 h 41 min against the 8 h budget (the estimate at the first checkpoints, about 10 steps per
second, gave about two hours).

<!-- tables76 reload begin -->
### R1. The run files reloaded (RunFile, every group read back)

| Run file | Commit | Started | Concurrent | Config SHA-256 (12) | Flow states | Frames (n, ny, nx) | Sensor readings (steps, classes, sensors) | Alerts | /arrival/ (events, classes, ny, nx), t_final (s) |
|---|---|---|---|---|---|---|---|---|---|
| door_leak.h5 | a68d737 | 2026-11-05T19:34:10 | 3 | 58b754a3ca73 | 2 | (167, 120, 320) | (16600, 5, 4) | 0 | step: (1, 5, 120, 320), 60.0000 |
| filter_breach.h5 | a68d737 | 2026-11-05T19:34:10 | 3 | 6b40bc1b6003 | 1 | (333, 120, 320) | (33121, 5, 4) | 0 | step: (1, 5, 120, 320), 120.0000 |
| equipment_dust.h5 | a68d737 | 2026-11-05T19:34:10 | 3 | 7f8b37cabf9d | 1 | (333, 120, 320) | (33120, 5, 4) | 0 | step: (1, 5, 120, 320), 120.0000 |
| clean_out.h5 | a68d737 | 2026-11-05T19:34:10 | 3 | 08ffca336251 | 1 | (664, 120, 320) | (66238, 5, 4) | 20 | step: (1, 5, 120, 320), 240.0000 |
<!-- tables76 reload end -->

<!-- tables76 flows begin -->
### F1. The flow states (320x120, from rest or warm-started)

| Run | State | Origin | Outer | Stop | Wall (s) | s per outer |
|---|---|---|---|---|---|---|
| door_leak | 22ac4cb0815d | rest | 4,389 | error_estimate_and_continuity | 842 | 0.192 |
| door_leak | eddc5d9212f0 | warm from 22ac4cb0815d | 3,872 | error_estimate_and_continuity | 731 | 0.189 |
| filter_breach | 22ac4cb0815d | rest | 4,389 | error_estimate_and_continuity | 842 | 0.192 |
| equipment_dust | 22ac4cb0815d | rest | 4,389 | error_estimate_and_continuity | 747 | 0.170 |
| clean_out | 22ac4cb0815d | rest | 4,389 | error_estimate_and_continuity | 819 | 0.187 |
<!-- tables76 flows end -->

<!-- tables76 states begin -->
### F2. The flow states against the gate's (prompt 69, the same files before the placement section)

| Run | State | Face hash (first 16) | The gate's | Largest face difference (m/s) |
|---|---|---|---|---|
| door_leak | 22ac4cb0815d | cd0701fd8aab916d | cd0701fd8aab916d | 0.00e+00 |
| door_leak | eddc5d9212f0 | 92d8e45e5b4b54ce | 92d8e45e5b4b54ce | 0.00e+00 |
| filter_breach | 22ac4cb0815d | cd0701fd8aab916d | cd0701fd8aab916d | 0.00e+00 |
| equipment_dust | 22ac4cb0815d | cd0701fd8aab916d | cd0701fd8aab916d | 0.00e+00 |
| clean_out | 22ac4cb0815d | cd0701fd8aab916d | cd0701fd8aab916d | 0.00e+00 |
<!-- tables76 states end -->

<!-- tables76 march begin -->
### M1. The marches (Courant 0.4, five classes at one step)

| Run | Intervals (as planned) | Steps | t at the end (s) | Run wall (s) | Flow solves' wall (s) | March wall (s) | March wall s per simulated s | Frames |
|---|---|---|---|---|---|---|---|---|
| door_leak | [0, 10) s: state 22ac4cb0815d, 2760 steps of 0.00362319 s, events none; [10, 15) s: state eddc5d9212f0, 1419 steps of 0.00352361 s, events door_open; [15, 60) s: state 22ac4cb0815d, 12420 steps of 0.00362319 s, events none | 16,599 | 60.0000 | 2995 | 1573 | 1422 | 23.70 | 167 |
| filter_breach | [0, 10) s: state 22ac4cb0815d, 2760 steps of 0.00362319 s, events none; [10, 70) s: state 22ac4cb0815d, 16560 steps of 0.00362319 s, events panel_breach; [70, 120) s: state 22ac4cb0815d, 13800 steps of 0.00362319 s, events none | 33,120 | 120.0000 | 4101 | 842 | 3258 | 27.15 | 333 |
| equipment_dust | [0, 10) s: state 22ac4cb0815d, 2760 steps of 0.00362319 s, events none; [10, 40) s: state 22ac4cb0815d, 8280 steps of 0.00362319 s, events litho_dust; [40, 120) s: state 22ac4cb0815d, 22079 steps of 0.00362335 s, events none | 33,119 | 120.0000 | 3675 | 747 | 2928 | 24.40 | 333 |
| clean_out | [0, 240) s: state 22ac4cb0815d, 66237 steps of 0.00362335 s, events none | 66,237 | 240.0000 | 6058 | 819 | 5239 | 21.83 | 664 |
<!-- tables76 march end -->

<!-- tables76 budgets begin -->
### B1. Budget closure (residual over the particles supplied), at the end and at the worst stored frame

| Run | Class (um) | Relative at the end | Worst relative over the stored frames | at t (s) | Inside REQ-T05's 1e-4 at every frame |
|---|---|---|---|---|---|
| door_leak | 0.1 | 7.003e-08 | 7.003e-08 | 60.00 | yes |
| door_leak | 0.3 | 7.003e-08 | 7.003e-08 | 60.00 | yes |
| door_leak | 0.5 | 7.002e-08 | 7.002e-08 | 60.00 | yes |
| door_leak | 1 | 7.000e-08 | 7.000e-08 | 60.00 | yes |
| door_leak | 5 | 6.930e-08 | 6.930e-08 | 60.00 | yes |
| filter_breach | 0.1 | -8.241e-08 | 3.927e-07 | 27.90 | yes |
| filter_breach | 0.3 | -8.240e-08 | 3.927e-07 | 27.90 | yes |
| filter_breach | 0.5 | -8.244e-08 | 3.927e-07 | 27.90 | yes |
| filter_breach | 1 | -8.249e-08 | 3.927e-07 | 27.90 | yes |
| filter_breach | 5 | -8.464e-08 | 3.921e-07 | 27.90 | yes |
| equipment_dust | 0.1 | -4.360e-06 | 6.201e-06 | 14.49 | yes |
| equipment_dust | 0.3 | -4.360e-06 | 6.201e-06 | 14.49 | yes |
| equipment_dust | 0.5 | -4.359e-06 | 6.200e-06 | 14.49 | yes |
| equipment_dust | 1 | -4.356e-06 | 6.195e-06 | 14.49 | yes |
| equipment_dust | 5 | -4.245e-06 | 6.204e-06 | 14.49 | yes |
| clean_out | 0.1 | -1.500e-06 | -1.504e-06 | 222.11 | yes |
| clean_out | 0.3 | -1.498e-06 | -1.502e-06 | 222.11 | yes |
| clean_out | 0.5 | -1.498e-06 | -1.501e-06 | 222.11 | yes |
| clean_out | 1 | -1.497e-06 | -1.500e-06 | 222.11 | yes |
| clean_out | 5 | -1.489e-06 | -1.491e-06 | 226.46 | yes |

The largest magnitude over every class of every run at every stored frame is 6.204e-06, inside REQ-T05's 1e-4 by 16 times.
<!-- tables76 budgets end -->

<!-- tables76 alerts begin -->
### A1. The alert tables, and the monitor's latency against the map's at the file's sensors

| Run | Alerts | Still open at the end | Sensor, class and event triples | The two agree on |
|---|---|---|---|---|
| door_leak | 0 | 0 | 20 | 20 |
| filter_breach | 0 | 0 | 20 | 20 |
| equipment_dust | 0 | 0 | 20 | 20 |
| clean_out | 20 | 0 | 20 | 20 |
<!-- tables76 alerts end -->

The flow states are the gate's to the last bit (F2), and the gate report's table X1 finds the
frames, budgets and sensor readings equal to the gate's as well, so the arrival maps and the
alert monitor changed nothing the march computes. The event scenarios open no alert; the
clean-out's twenty alerts open at t = 0 and all close. The monitor and the arrival map agree at
every sensor, class and event (A1), REQ-A04's two instruments on the product grid.

## 3. The layouts

<!-- tables76 layouts begin -->
### L1. The layouts on 320x120: mount ['walls', 'ceiling', 'obstacle_tops', 'obstacle_sides'] at 0.05 m, 1,737 mountable and 27,017 free candidates; the means are capped at each unit's remaining run time

| Layout | Cell centres (x, y), m | Rule | Method | Exact | Units scored | Uncovered | Worst case (s) | Mean (s) | Undetected | Coverage within 10 s | Distinct / maximal masks | Detected, of all fifteen | Cells setting a latency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| the file's four, all fifteen units | (0.988, 1.488), (2.688, 2.488), (4.788, 2.488), (6.188, 1.188) | - | given | no | 15 | 0 | inf | 90.000 | 15 | 0.000 | - | door 0 of 5, breach 0 of 5, dust 0 of 5 | 0 of 4 |
| the file's four, the mountable set's units | (0.988, 1.488), (2.688, 2.488), (4.788, 2.488), (6.188, 1.188) | - | given | no | 10 | 0 | inf | 110.000 | 10 | 0.000 | - | door 0 of 5, breach 0 of 5, dust 0 of 5 | 0 of 4 |
| searched, mountable, worst-case rule | (1.463, 0.013), (1.488, 0.013), (3.813, 2.013), (3.188, 2.038) | worst_case | set_cover | yes | 10 | 5 | 2.565 | 1.295 | 0 | 1.000 | 7 / 2 | door 0 of 5, breach 5 of 5, dust 5 of 5 | 2 of 4 |
| searched, mountable, mean rule | (1.463, 0.013), (3.813, 2.013), (3.188, 2.038), (3.863, 2.038) | mean | greedy_exchange | no | 10 | 5 | 2.565 | 1.295 | 0 | 1.000 | - | door 0 of 5, breach 5 of 5, dust 5 of 5 | 3 of 4 |
| free optimum, worst-case rule | (0.062, 0.013), (0.062, 0.562), (3.813, 2.013), (3.513, 2.938) | worst_case | set_cover | yes | 15 | 0 | 0.101 | 0.060 | 0 | 1.000 | 8 / 3 | door 5 of 5, breach 5 of 5, dust 5 of 5 | 3 of 4 |
| free optimum, mean rule | (0.062, 0.562), (3.813, 2.013), (3.838, 2.088), (3.513, 2.938) | mean | greedy_exchange | no | 15 | 0 | 0.101 | 0.060 | 0 | 1.000 | - | door 5 of 5, breach 5 of 5, dust 5 of 5 | 4 of 4 |
| free optimum, worst-case rule, the mountable set's units | (0.062, 0.013), (0.088, 0.013), (3.813, 2.013), (3.513, 2.938) | worst_case | set_cover | yes | 10 | 0 | 0.101 | 0.041 | 0 | 1.000 | 7 / 2 | door 5 of 5, breach 5 of 5, dust 5 of 5 | 3 of 4 |

Uncovered by the mountable set: door_leak door_open class 0, door_leak door_open class 1, door_leak door_open class 2, door_leak door_open class 3, door_leak door_open class 4.
Mountable candidates that detect a door unit: 0 of 1,737.
The door's plume reaches 854 cells, x up to 0.825 m and y up to 1.125 m; the first floor return spans x 0.5 to 1.25 m.
Mountable candidates that detect a door unit at other standoffs, the rule of item 0 held: 0.05 m: 0 of 1,737; 0.1 m: 0 of 3,414; 0.2 m: 0 of 6,585; 0.5 m: 0 of 11,566; 1 m: 0 of 5,291.

Over the 10 units the mountable set covers, the order per rule, best first:

- worst case: free optimum (worst case) < searched, mountable (worst case) = searched, mountable (mean) < the file's four;
- mean: free optimum (worst case) < searched, mountable (worst case) = searched, mountable (mean) < the file's four;
- coverage within 10 s: searched, mountable (worst case) = searched, mountable (mean) = free optimum (worst case) < the file's four;
- section D's ranking, the worst case with ties by the mean: free optimum (worst case), searched, mountable (worst case), searched, mountable (mean), the file's four.
<!-- tables76 layouts end -->

<!-- tables76 curve begin -->
### L2. The free optimum for one to four sensors, the worst-case rule (the bound)

| Layout | Cell centres (x, y), m | Rule | Method | Exact | Units scored | Uncovered | Worst case (s) | Mean (s) | Undetected | Coverage within 10 s | Distinct / maximal masks | Detected, of all fifteen | Cells setting a latency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 sensors | (3.838, 2.088) | mean | argmin | yes | 15 | 0 | inf | 17.567 | 5 | 0.667 | - | door 0 of 5, breach 5 of 5, dust 5 of 5 | 1 of 1 |
| 2 sensors | (0.062, 0.562), (3.138, 2.013) | worst_case | exhaustive | yes | 15 | 0 | 2.551 | 1.385 | 0 | 1.000 | - | door 5 of 5, breach 5 of 5, dust 5 of 5 | 2 of 2 |
| 3 sensors | (0.062, 0.562), (3.813, 2.013), (3.513, 2.938) | worst_case | set_cover | yes | 15 | 0 | 0.101 | 0.060 | 0 | 1.000 | 8 / 3 | door 5 of 5, breach 5 of 5, dust 5 of 5 | 3 of 3 |
| 4 sensors | (0.062, 0.013), (0.062, 0.562), (3.813, 2.013), (3.513, 2.938) | worst_case | set_cover | yes | 15 | 0 | 0.101 | 0.060 | 0 | 1.000 | 8 / 3 | door 5 of 5, breach 5 of 5, dust 5 of 5 | 3 of 4 |
<!-- tables76 curve end -->

What L1 says, in the order section E asks:

- **The file's four** detect none of the fifteen units, so their worst case is infinite and
  their mean is the mean of the caps. These three events do not reach them; the records do not
  say what the positions were chosen for.
- **The searched four on the mountable candidates** detect all ten units a mountable cell can
  reach. Two of the four set every latency (the last column): (3.188, 2.038), over the litho
  tool's left corner, where the breach's spill passes, and (3.813, 2.013), on the tool's top
  inside the dust source's box (x 3.80 to 3.90, y 2.00 to 2.10). The set cover needs two masks
  at the optimum, and the search fills the other two sensors by the mean; here they add nothing
  to any unit, so "the searched four" is a searched two with two spares. A sensor inside the
  source's box reads the release at its first steps, which is why the dust's latencies are at
  the step's scale in U2: that is a statement about where the box is, not a detection result of
  any weight. The 0.1 um dust field peaks at a few times its threshold (the sweep's F1), so that
  class's detection rests on standing in the patch.
- **The door has no mountable detector at any standoff.** The plume's reach and the counts at
  other standoffs are the lines under L1. The plume stays below the top of the doorway, whose
  wall cells are excluded by the opening, and ends over the first floor return. The exclusion
  widens with the standoff, as item 0's rule has it, so this is read at each standoff with the
  rule held. The sensor that would detect it is one the mounting list does not offer: on the
  floor, or free-standing in the doorway's lower half.
- **The free optimum** is the bound: three cells cover all fifteen units, the fourth is a spare
  (L2), and one sensor cannot cover the door and the rest (rule `mean`, the fallback). Its
  working cells stand at the plumes' sources: 0.06 m inside the doorway and under the breached
  panel, each just outside its opening's exclusion, and inside the dust's box. It says how much
  of the latency is position; it is not a place a sensor could be fixed.

## 4. The latency per unit

The counter's sample time is OPEN (ADR-015 C): a counter's sample adds to every latency below
as an instrument delay, about half the sample time and the same amount for every layout, and
is not modelled in the field. T7's time error is the only measurement of the time error, at the
file's sensors, on 160x60, in one direction; at a searched cell the column gives T7's largest as
the scale. The width column says how far the sweep's inputs move the plume on the cell's row.

<!-- tables76 headline begin -->
### U2. Per unit, the layout's latency, the least over its cells (s; never with the run's end)

| Scenario | Event | Class (um) | the file's four | searched, mountable | free optimum |
|---|---|---|---|---|---|
| door_leak | door_open | 0.1 | never (to 60) | never (to 60) | 0.099 |
| door_leak | door_open | 0.3 | never (to 60) | never (to 60) | 0.099 |
| door_leak | door_open | 0.5 | never (to 60) | never (to 60) | 0.099 |
| door_leak | door_open | 1 | never (to 60) | never (to 60) | 0.099 |
| door_leak | door_open | 5 | never (to 60) | never (to 60) | 0.099 |
| filter_breach | panel_breach | 0.1 | never (to 120) | 2.565 | 0.058 |
| filter_breach | panel_breach | 0.3 | never (to 120) | 2.565 | 0.058 |
| filter_breach | panel_breach | 0.5 | never (to 120) | 2.565 | 0.058 |
| filter_breach | panel_breach | 1 | never (to 120) | 2.565 | 0.058 |
| filter_breach | panel_breach | 5 | never (to 120) | 2.562 | 0.058 |
| equipment_dust | litho_dust | 0.1 | never (to 120) | 0.101 | 0.101 |
| equipment_dust | litho_dust | 0.3 | never (to 120) | 0.011 | 0.011 |
| equipment_dust | litho_dust | 0.5 | never (to 120) | 0.004 | 0.004 |
| equipment_dust | litho_dust | 1 | never (to 120) | 0.004 | 0.004 |
| equipment_dust | litho_dust | 5 | never (to 120) | 0.004 | 0.004 |
<!-- tables76 headline end -->

<!-- tables76 units begin -->
### U1. Per scenario, event, class and sensor, for each configuration (0.1 um and 5 um; every class in tables76.md). A file sensor's peak and time above are from its every-step readings, a searched cell's from the stored frames (every 100 steps, about 0.36 s). The time error is T7's (160x60, one direction), measured at the file's sensors only. The width is the plume's largest at the threshold on the cell's row over step 1's 160x60 arms, read at frame resolution.

| Scenario | Event | Class (um) | Configuration | Sensor | Latency (s) | Peak over threshold | Time above (s) | Read from | T7 time error | Sweep's width on the row (m) |
|---|---|---|---|---|---|---|---|---|---|---|
| door_leak | door_open | 0.1 | the file's four | near_door (0.988, 1.488) | never (to 60) | below the resolved plume (< 0.01) | 0.00 | steps | -0.03% | 0.00 to 0.00 |
| door_leak | door_open | 0.1 | the file's four | above_gap_1 (2.688, 2.488) | never (to 60) | below the resolved plume (< 0.01) | 0.00 | steps | -3.63% | 0.00 to 0.00 |
| door_leak | door_open | 0.1 | the file's four | above_gap_2 (4.788, 2.488) | never (to 60) | below the resolved plume (< 0.01) | 0.00 | steps | -3.49% | 0.00 to 0.00 |
| door_leak | door_open | 0.1 | the file's four | hood_entry (6.188, 1.188) | never (to 60) | below the resolved plume (< 0.01) | 0.00 | steps | -3.00% | 0.00 to 0.25 |
| door_leak | door_open | 0.1 | searched, mountable | S1 (1.463, 0.013) | never (to 60) | below the resolved plume (< 0.01) | 0.00 | frames | up to 3.63% (T7's largest) | 0.85 to 1.15 |
| door_leak | door_open | 0.1 | searched, mountable | S2 (1.488, 0.013) | never (to 60) | below the resolved plume (< 0.01) | 0.00 | frames | up to 3.63% (T7's largest) | 0.85 to 1.15 |
| door_leak | door_open | 0.1 | searched, mountable | S3 (3.813, 2.013) | never (to 60) | below the resolved plume (< 0.01) | 0.00 | frames | up to 3.63% (T7's largest) | 0.00 to 0.00 |
| door_leak | door_open | 0.1 | searched, mountable | S4 (3.188, 2.038) | never (to 60) | below the resolved plume (< 0.01) | 0.00 | frames | up to 3.63% (T7's largest) | 0.00 to 0.00 |
| door_leak | door_open | 0.1 | free optimum | F1 (0.062, 0.013) | 0.102 | 100 | 41.16 | frames | up to 3.63% (T7's largest) | 0.85 to 1.15 |
| door_leak | door_open | 0.1 | free optimum | F2 (0.062, 0.562) | 0.099 | 100 | 6.74 | frames | up to 3.63% (T7's largest) | 0.45 to 0.75 |
| door_leak | door_open | 0.1 | free optimum | F3 (3.813, 2.013) | never (to 60) | below the resolved plume (< 0.01) | 0.00 | frames | up to 3.63% (T7's largest) | 0.00 to 0.00 |
| door_leak | door_open | 0.1 | free optimum | F4 (3.513, 2.938) | never (to 60) | below the resolved plume (< 0.01) | 0.00 | frames | up to 3.63% (T7's largest) | 0.00 to 0.00 |
| door_leak | door_open | 5 | the file's four | near_door (0.988, 1.488) | never (to 60) | below the resolved plume (< 0.01) | 0.00 | steps | -0.03% | 0.00 to 0.00 |
| door_leak | door_open | 5 | the file's four | above_gap_1 (2.688, 2.488) | never (to 60) | below the resolved plume (< 0.01) | 0.00 | steps | -3.63% | 0.00 to 0.00 |
| door_leak | door_open | 5 | the file's four | above_gap_2 (4.788, 2.488) | never (to 60) | below the resolved plume (< 0.01) | 0.00 | steps | -3.49% | 0.00 to 0.00 |
| door_leak | door_open | 5 | the file's four | hood_entry (6.188, 1.188) | never (to 60) | below the resolved plume (< 0.01) | 0.00 | steps | -3.00% | 0.00 to 0.25 |
| door_leak | door_open | 5 | searched, mountable | S1 (1.463, 0.013) | never (to 60) | below the resolved plume (< 0.01) | 0.00 | frames | up to 3.63% (T7's largest) | 0.85 to 1.15 |
| door_leak | door_open | 5 | searched, mountable | S2 (1.488, 0.013) | never (to 60) | below the resolved plume (< 0.01) | 0.00 | frames | up to 3.63% (T7's largest) | 0.85 to 1.15 |
| door_leak | door_open | 5 | searched, mountable | S3 (3.813, 2.013) | never (to 60) | below the resolved plume (< 0.01) | 0.00 | frames | up to 3.63% (T7's largest) | 0.00 to 0.00 |
| door_leak | door_open | 5 | searched, mountable | S4 (3.188, 2.038) | never (to 60) | below the resolved plume (< 0.01) | 0.00 | frames | up to 3.63% (T7's largest) | 0.00 to 0.00 |
| door_leak | door_open | 5 | free optimum | F1 (0.062, 0.013) | 0.102 | 100 | 40.80 | frames | up to 3.63% (T7's largest) | 0.85 to 1.15 |
| door_leak | door_open | 5 | free optimum | F2 (0.062, 0.562) | 0.099 | 100 | 6.74 | frames | up to 3.63% (T7's largest) | 0.45 to 0.75 |
| door_leak | door_open | 5 | free optimum | F3 (3.813, 2.013) | never (to 60) | below the resolved plume (< 0.01) | 0.00 | frames | up to 3.63% (T7's largest) | 0.00 to 0.00 |
| door_leak | door_open | 5 | free optimum | F4 (3.513, 2.938) | never (to 60) | below the resolved plume (< 0.01) | 0.00 | frames | up to 3.63% (T7's largest) | 0.00 to 0.00 |
| filter_breach | panel_breach | 0.1 | the file's four | near_door (0.988, 1.488) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | steps | -0.03% | 0.95 to 1.40 |
| filter_breach | panel_breach | 0.1 | the file's four | above_gap_1 (2.688, 2.488) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | steps | -3.63% | 0.95 to 1.60 |
| filter_breach | panel_breach | 0.1 | the file's four | above_gap_2 (4.788, 2.488) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | steps | -3.49% | 0.95 to 1.60 |
| filter_breach | panel_breach | 0.1 | the file's four | hood_entry (6.188, 1.188) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | steps | -3.00% | 0.95 to 1.40 |
| filter_breach | panel_breach | 0.1 | searched, mountable | S1 (1.463, 0.013) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | frames | up to 3.63% (T7's largest) | 0.85 to 1.40 |
| filter_breach | panel_breach | 0.1 | searched, mountable | S2 (1.488, 0.013) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | frames | up to 3.63% (T7's largest) | 0.85 to 1.40 |
| filter_breach | panel_breach | 0.1 | searched, mountable | S3 (3.813, 2.013) | 3.446 | 96.8 | 66.30 | frames | up to 3.63% (T7's largest) | 1.90 to 2.35 |
| filter_breach | panel_breach | 0.1 | searched, mountable | S4 (3.188, 2.038) | 2.565 | 99.8 | 64.86 | frames | up to 3.63% (T7's largest) | 1.90 to 2.35 |
| filter_breach | panel_breach | 0.1 | free optimum | F1 (0.062, 0.013) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | frames | up to 3.63% (T7's largest) | 0.85 to 1.40 |
| filter_breach | panel_breach | 0.1 | free optimum | F2 (0.062, 0.562) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | frames | up to 3.63% (T7's largest) | 0.95 to 1.40 |
| filter_breach | panel_breach | 0.1 | free optimum | F3 (3.813, 2.013) | 3.446 | 96.8 | 66.30 | frames | up to 3.63% (T7's largest) | 1.90 to 2.35 |
| filter_breach | panel_breach | 0.1 | free optimum | F4 (3.513, 2.938) | 0.058 | 93.8 | 60.51 | frames | up to 3.63% (T7's largest) | 0.65 to 0.90 |
| filter_breach | panel_breach | 5 | the file's four | near_door (0.988, 1.488) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | steps | -0.03% | 0.95 to 1.40 |
| filter_breach | panel_breach | 5 | the file's four | above_gap_1 (2.688, 2.488) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | steps | -3.63% | 0.95 to 1.60 |
| filter_breach | panel_breach | 5 | the file's four | above_gap_2 (4.788, 2.488) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | steps | -3.49% | 0.95 to 1.60 |
| filter_breach | panel_breach | 5 | the file's four | hood_entry (6.188, 1.188) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | steps | -3.00% | 0.95 to 1.40 |
| filter_breach | panel_breach | 5 | searched, mountable | S1 (1.463, 0.013) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | frames | up to 3.63% (T7's largest) | 0.85 to 1.40 |
| filter_breach | panel_breach | 5 | searched, mountable | S2 (1.488, 0.013) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | frames | up to 3.63% (T7's largest) | 0.85 to 1.40 |
| filter_breach | panel_breach | 5 | searched, mountable | S3 (3.813, 2.013) | 3.431 | 96.7 | 66.30 | frames | up to 3.63% (T7's largest) | 1.90 to 2.35 |
| filter_breach | panel_breach | 5 | searched, mountable | S4 (3.188, 2.038) | 2.562 | 99.6 | 64.49 | frames | up to 3.63% (T7's largest) | 1.90 to 2.35 |
| filter_breach | panel_breach | 5 | free optimum | F1 (0.062, 0.013) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | frames | up to 3.63% (T7's largest) | 0.85 to 1.40 |
| filter_breach | panel_breach | 5 | free optimum | F2 (0.062, 0.562) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | frames | up to 3.63% (T7's largest) | 0.95 to 1.40 |
| filter_breach | panel_breach | 5 | free optimum | F3 (3.813, 2.013) | 3.431 | 96.7 | 66.30 | frames | up to 3.63% (T7's largest) | 1.90 to 2.35 |
| filter_breach | panel_breach | 5 | free optimum | F4 (3.513, 2.938) | 0.058 | 93.7 | 60.51 | frames | up to 3.63% (T7's largest) | 0.65 to 0.90 |
| equipment_dust | litho_dust | 0.1 | the file's four | near_door (0.988, 1.488) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | steps | -0.03% | 0.00 to 0.00 |
| equipment_dust | litho_dust | 0.1 | the file's four | above_gap_1 (2.688, 2.488) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | steps | -3.63% | 0.00 to 0.00 |
| equipment_dust | litho_dust | 0.1 | the file's four | above_gap_2 (4.788, 2.488) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | steps | -3.49% | 0.00 to 0.00 |
| equipment_dust | litho_dust | 0.1 | the file's four | hood_entry (6.188, 1.188) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | steps | -3.00% | 0.00 to 0.00 |
| equipment_dust | litho_dust | 0.1 | searched, mountable | S1 (1.463, 0.013) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | frames | up to 3.63% (T7's largest) | 0.00 to 0.00 |
| equipment_dust | litho_dust | 0.1 | searched, mountable | S2 (1.488, 0.013) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | frames | up to 3.63% (T7's largest) | 0.00 to 0.00 |
| equipment_dust | litho_dust | 0.1 | searched, mountable | S3 (3.813, 2.013) | 0.101 | 7.29 | 30.80 | frames | up to 3.63% (T7's largest) | 0.70 to 0.75 |
| equipment_dust | litho_dust | 0.1 | searched, mountable | S4 (3.188, 2.038) | never (to 120) | 0.551 | 0.00 | frames | up to 3.63% (T7's largest) | 0.70 to 0.75 |
| equipment_dust | litho_dust | 0.1 | free optimum | F1 (0.062, 0.013) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | frames | up to 3.63% (T7's largest) | 0.00 to 0.00 |
| equipment_dust | litho_dust | 0.1 | free optimum | F2 (0.062, 0.562) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | frames | up to 3.63% (T7's largest) | 0.00 to 0.00 |
| equipment_dust | litho_dust | 0.1 | free optimum | F3 (3.813, 2.013) | 0.101 | 7.29 | 30.80 | frames | up to 3.63% (T7's largest) | 0.70 to 0.75 |
| equipment_dust | litho_dust | 0.1 | free optimum | F4 (3.513, 2.938) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | frames | up to 3.63% (T7's largest) | 0.00 to 0.00 |
| equipment_dust | litho_dust | 5 | the file's four | near_door (0.988, 1.488) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | steps | -0.03% | 0.55 to 1.40 |
| equipment_dust | litho_dust | 5 | the file's four | above_gap_1 (2.688, 2.488) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | steps | -3.63% | 0.00 to 0.80 |
| equipment_dust | litho_dust | 5 | the file's four | above_gap_2 (4.788, 2.488) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | steps | -3.49% | 0.00 to 0.80 |
| equipment_dust | litho_dust | 5 | the file's four | hood_entry (6.188, 1.188) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | steps | -3.00% | 0.60 to 1.40 |
| equipment_dust | litho_dust | 5 | searched, mountable | S1 (1.463, 0.013) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | frames | up to 3.63% (T7's largest) | 0.60 to 1.40 |
| equipment_dust | litho_dust | 5 | searched, mountable | S2 (1.488, 0.013) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | frames | up to 3.63% (T7's largest) | 0.60 to 1.40 |
| equipment_dust | litho_dust | 5 | searched, mountable | S3 (3.813, 2.013) | 0.004 | 2.49e+04 | 34.06 | frames | up to 3.63% (T7's largest) | 1.10 to 2.45 |
| equipment_dust | litho_dust | 5 | searched, mountable | S4 (3.188, 2.038) | 1.181 | 1.81e+03 | 32.61 | frames | up to 3.63% (T7's largest) | 1.10 to 2.45 |
| equipment_dust | litho_dust | 5 | free optimum | F1 (0.062, 0.013) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | frames | up to 3.63% (T7's largest) | 0.60 to 1.40 |
| equipment_dust | litho_dust | 5 | free optimum | F2 (0.062, 0.562) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | frames | up to 3.63% (T7's largest) | 0.60 to 1.40 |
| equipment_dust | litho_dust | 5 | free optimum | F3 (3.813, 2.013) | 0.004 | 2.49e+04 | 34.06 | frames | up to 3.63% (T7's largest) | 1.10 to 2.45 |
| equipment_dust | litho_dust | 5 | free optimum | F4 (3.513, 2.938) | never (to 120) | below the resolved plume (< 0.01) | 0.00 | frames | up to 3.63% (T7's largest) | 0.00 to 0.00 |
<!-- tables76 units end -->

## 5. Each layout's clean-out recovery

<!-- tables76 recovery begin -->
### C1. The clean-out: each layout's recovery, the last time each cell was at or above its threshold (the run went to 240 s; inf is still above at the end). The well-mixed 100:1 time is 25.6135 s, the file's room recovery time, the first step below and staying below, 27.9433 s. T7's time error at the file's sensors is up to 3.63%, near_door 0.03%.

| Layout | Class (um) | Cells' recovery (s) | The layout's recovery, its last cell (s) |
|---|---|---|---|
| the file's four | 0.1 | 27.940, 1.391, 1.453, 3.290 | 27.940 |
| the file's four | 5 | 27.730, 1.391, 1.449, 3.283 | 27.730 |
| searched, mountable | 0.1 | 55.100, 55.560, 10.033, 7.268 | 55.560 |
| searched, mountable | 5 | 55.028, 55.488, 9.870, 7.167 | 55.488 |
| free optimum, 1 | 0.1 | 6.091 | 6.091 |
| free optimum, 1 | 5 | 6.022 | 6.022 |
| free optimum, 2 | 0.1 | 98.870, 8.522 | 98.870 |
| free optimum, 2 | 5 | 98.142, 8.392 | 98.142 |
| free optimum, 3 | 0.1 | 98.870, 10.033, 0.301 | 98.870 |
| free optimum, 3 | 5 | 98.142, 9.870, 0.297 | 98.142 |
| free optimum, 4 | 0.1 | 110.099, 98.870, 10.033, 0.301 | 110.099 |
| free optimum, 4 | 5 | 109.284, 98.142, 9.870, 0.297 | 109.284 |
<!-- tables76 recovery end -->

C1 reads the clean-out's local recovery field at each layout's cells, so a layout's recovery is
when it would call the room clean. The local field's value is the last step at or above, one
step before the room recovery time's first step below. The two cells that do the searched
layout's detecting clear in about 10 s and 7 s, under the supply, where the file's slowest
sensor (near_door) clears at the room recovery time. The searched layout's larger figure comes
from its two spares in the pocket beside the rack, which detect nothing. So what detects an
event and what says the room has cleared are different questions: a layout of the working cells
alone would call the room clean about three times sooner than the file's four, with more of the
free area still above threshold (the gate report's table L1 of Phase 4 puts a fifth of it above
100:1 at the file's room recovery time).

## 6. The maps

Per event scenario, the 0.1 um and 5 um classes, with the three layouts drawn. The clean-out's
map is t = 0 at every non-SOLID cell (ADR-015 D: the clearing companion says nothing new), so
it is not drawn.

![The door's arrival maps](phase5_arrival_door_leak.png)

![The breach's arrival maps](phase5_arrival_filter_breach.png)

![The dust's arrival maps](phase5_arrival_equipment_dust.png)

## 7. What the ranking can claim

Section D's claim is the ranking of layouts, per rule with ties, not an absolute latency. Two
readings of its stability, where the records allow:

<!-- tables76 stability begin -->
### K1. The searches on 80x30 and 320x120, the same rule and mount, each layout scored on both grids over the units the grid's mountable search covers; a layout from the other grid is read at the cells holding its centres

| Grid | Layout | Cell centres (x, y), m | Mountable on this grid | Worst case (s) | Mean (s) | Undetected | Coverage within 10 s |
|---|---|---|---|---|---|---|---|
| 80x30 | the file's four | (0.950, 1.450), (2.650, 2.450), (4.750, 2.450), (6.150, 1.150) | 0 of 4 | inf | 110.000 | 10 of 10 | 0.000 |
| 80x30 | searched here | (1.450, 0.050), (2.350, 0.050), (3.250, 2.050), (3.850, 2.050) | 4 of 4 | 1.974 | 1.006 | 0 of 10 | 1.000 |
| 80x30 | searched on 320x120 | (1.450, 0.050), (3.850, 2.050), (3.150, 2.050) | 2 of 3 | 1.727 | 0.883 | 0 of 10 | 1.000 |
| 80x30 | free optimum here | (0.150, 0.050), (0.250, 0.050), (3.850, 2.050), (3.550, 2.850) | 1 of 4 | 0.123 | 0.072 | 0 of 10 | 1.000 |
| 320x120 | the file's four | (0.988, 1.488), (2.688, 2.488), (4.788, 2.488), (6.188, 1.188) | 0 of 4 | inf | 110.000 | 10 of 10 | 0.000 |
| 320x120 | searched here | (1.463, 0.013), (1.488, 0.013), (3.813, 2.013), (3.188, 2.038) | 4 of 4 | 2.565 | 1.295 | 0 of 10 | 1.000 |
| 320x120 | searched on 80x30 | (1.438, 0.038), (2.338, 0.038), (3.238, 2.038), (3.838, 2.038) | 3 of 4 | 2.728 | 1.376 | 0 of 10 | 1.000 |
| 320x120 | free optimum here | (0.062, 0.013), (0.088, 0.013), (3.813, 2.013), (3.513, 2.938) | 1 of 4 | 0.101 | 0.041 | 0 of 10 | 1.000 |

Uncovered by the mountable set on 80x30: 5 units (door_leak); on 320x120: 5 (door_leak).

The order per rule, best first:

- 80x30: worst case free optimum here < searched on 320x120 < searched here < the file's four; mean free optimum here < searched on 320x120 < searched here < the file's four; coverage within 10 s searched here = searched on 320x120 = free optimum here < the file's four.
- 320x120: worst case free optimum here < searched here < searched on 80x30 < the file's four; mean free optimum here < searched here < searched on 80x30 < the file's four; coverage within 10 s searched here = searched on 80x30 = free optimum here < the file's four.
<!-- tables76 stability end -->

<!-- tables76 arms begin -->
### K2. The searched 320x120 layout on step 1's arm runs (arrival_from_run: frame-resolved, flagged), read at the cells holding its centres, beside the file's four and the arm's own mountable search, over the units the arm's own search covers

| Arm | Grid | Scenarios run | Frame-resolved | Layout | Mountable on this grid | Worst case (s) | Mean (s) | Undetected | Detected, of the arm's units | Own search's rule, method |
|---|---|---|---|---|---|---|---|---|---|---|
| control | 160x60 | breach, dust | yes | searched on 320x120 | 3 of 3 | 2.658 | 1.472 | 0 of 10 | breach 5 of 5, dust 5 of 5 | worst_case, set_cover |
| control | 160x60 | breach, dust | yes | the file's four | 0 of 4 | inf | 110.000 | 10 of 10 | breach 0 of 5, dust 0 of 5 | worst_case, set_cover |
| control | 160x60 | breach, dust | yes | the arm's own | 4 of 4 | 2.658 | 1.472 | 0 of 10 | breach 5 of 5, dust 5 of 5 | worst_case, set_cover |
| i10 | 160x60 | door, breach, dust | yes | searched on 320x120 | 3 of 3 | 2.690 | 1.500 | 0 of 10 | door 0 of 5, breach 5 of 5, dust 5 of 5 | worst_case, set_cover |
| i10 | 160x60 | door, breach, dust | yes | the file's four | 0 of 4 | inf | 110.000 | 10 of 10 | door 0 of 5, breach 0 of 5, dust 0 of 5 | worst_case, set_cover |
| i10 | 160x60 | door, breach, dust | yes | the arm's own | 4 of 4 | 2.690 | 1.500 | 0 of 10 | door 0 of 5, breach 5 of 5, dust 5 of 5 | worst_case, set_cover |
| max_nut | 160x60 | door, breach, dust | yes | searched on 320x120 | 3 of 3 | 2.077 | 1.272 | 0 of 10 | door 0 of 5, breach 5 of 5, dust 5 of 5 | worst_case, set_cover |
| max_nut | 160x60 | door, breach, dust | yes | the file's four | 0 of 4 | inf | 110.000 | 10 of 10 | door 0 of 5, breach 0 of 5, dust 0 of 5 | worst_case, set_cover |
| max_nut | 160x60 | door, breach, dust | yes | the arm's own | 4 of 4 | 2.077 | 1.272 | 0 of 10 | door 0 of 5, breach 5 of 5, dust 5 of 5 | worst_case, set_cover |
| min_nut | 160x60 | door, breach, dust | yes | searched on 320x120 | 3 of 3 | 2.600 | 1.418 | 0 of 10 | door 0 of 5, breach 5 of 5, dust 5 of 5 | worst_case, set_cover |
| min_nut | 160x60 | door, breach, dust | yes | the file's four | 0 of 4 | inf | 110.000 | 10 of 10 | door 0 of 5, breach 0 of 5, dust 0 of 5 | worst_case, set_cover |
| min_nut | 160x60 | door, breach, dust | yes | the arm's own | 4 of 4 | 2.600 | 1.418 | 0 of 10 | door 0 of 5, breach 5 of 5, dust 5 of 5 | worst_case, set_cover |
| sc02 | 160x60 | breach, dust | yes | searched on 320x120 | 3 of 3 | 2.658 | 1.472 | 0 of 10 | breach 5 of 5, dust 5 of 5 | worst_case, set_cover |
| sc02 | 160x60 | breach, dust | yes | the file's four | 0 of 4 | inf | 110.000 | 10 of 10 | breach 0 of 5, dust 0 of 5 | worst_case, set_cover |
| sc02 | 160x60 | breach, dust | yes | the arm's own | 4 of 4 | 2.658 | 1.472 | 0 of 10 | breach 5 of 5, dust 5 of 5 | worst_case, set_cover |
| widest | 160x60 | door, breach, dust | yes | searched on 320x120 | 3 of 3 | 2.077 | 1.272 | 0 of 10 | door 0 of 5, breach 5 of 5, dust 5 of 5 | worst_case, set_cover |
| widest | 160x60 | door, breach, dust | yes | the file's four | 0 of 4 | inf | 110.000 | 10 of 10 | door 0 of 5, breach 0 of 5, dust 0 of 5 | worst_case, set_cover |
| widest | 160x60 | door, breach, dust | yes | the arm's own | 4 of 4 | 2.077 | 1.272 | 0 of 10 | door 0 of 5, breach 5 of 5, dust 5 of 5 | worst_case, set_cover |
| min_nut | 320x120 | door, breach, dust | yes | searched on 320x120 | 4 of 4 | 2.954 | 1.551 | 0 of 10 | door 0 of 5, breach 5 of 5, dust 5 of 5 | worst_case, set_cover |
| min_nut | 320x120 | door, breach, dust | yes | the file's four | 0 of 4 | inf | 110.000 | 10 of 10 | door 0 of 5, breach 0 of 5, dust 0 of 5 | worst_case, set_cover |
| min_nut | 320x120 | door, breach, dust | yes | the arm's own | 4 of 4 | 2.954 | 1.551 | 0 of 10 | door 0 of 5, breach 5 of 5, dust 5 of 5 | worst_case, set_cover |

The order per rule, best first:

- control (160x60): worst case searched on 320x120 = the arm's own < the file's four; mean searched on 320x120 = the arm's own < the file's four.
- i10 (160x60): worst case searched on 320x120 = the arm's own < the file's four; mean searched on 320x120 = the arm's own < the file's four.
- max_nut (160x60): worst case searched on 320x120 = the arm's own < the file's four; mean searched on 320x120 = the arm's own < the file's four.
- min_nut (160x60): worst case searched on 320x120 = the arm's own < the file's four; mean searched on 320x120 = the arm's own < the file's four.
- sc02 (160x60): worst case searched on 320x120 = the arm's own < the file's four; mean searched on 320x120 = the arm's own < the file's four.
- widest (160x60): worst case searched on 320x120 = the arm's own < the file's four; mean searched on 320x120 = the arm's own < the file's four.
- min_nut (320x120): worst case searched on 320x120 = the arm's own < the file's four; mean searched on 320x120 = the arm's own < the file's four.
<!-- tables76 arms end -->

**Where it holds.**

- **Between 80x30 and 320x120** (K1): the same units are uncovered, the door's five, and under
  the worst case and the mean the free optimum comes first and the file's four last on both
  grids. The searched layouts' working cells are the same places: the tool's left corner and
  the source's box.
- **Across the sweep's inputs** (K2): at each arm that ran, the searched layout read on the arm's
  maps detects every breach and dust unit, the file's four detect none, and the searched layout
  ties the arm's own search under both rules. That is seven readings, six on 160x60 and `min_nut`
  on 320x120. They are frame-resolved and flagged: each latency is an upper bound on the arrival,
  a cell at or above only between two frames is missed, and the coarse frame spacing makes ties
  easier to reach, so the tie says the layout is as good as the arm's search to about 0.8 s on
  160x60 and not better than that.

**Where it does not.**

- **The latency's value.** The searched worst case moves by about a third between the two
  grids (K1), and the same cell reads differently on each. The 320x120 figures are not a
  convergence claim, and prediction (k)'s margin is a margin on this grid.
- **On 80x30 the 320x120 layout beats that grid's own search.** One of its cells, read at the
  coarse cell holding its centre, is not mountable there (K1's column): it is 0.071 m from the
  tool's corner on that grid, outside the 0.05 m standoff. So the exact search is not beaten over
  its candidates, and "mountable" is itself a statement on a grid.
- **The spares.** The two idle cells differ between grids and mean nothing; they are the first
  candidates the fill reaches. Only the working cells carry over.
- **The door.** Nothing mountable detects it on any grid or arm that ran, so the mountable
  layouts tie at it and the ranking is over the other ten units. Among free layouts it is
  detected; the door's control and `sc02` arms on 160x60 did not run (the sweep's section 1.2).
- **What was not read.** `widest` on 320x120 did not converge, so no arm of the widest inputs is
  on the product grid. The stability reading across inputs is coarse-grid at that corner.

ADR-014 H's limits stand: the 2D slice, the steady extreme of the door, and the spreading limit
as bounded by VAL-024 and the sweep. One bears directly on this result. Both working cells stand
on the litho tool's top, in the spill region, where VAL-024's row does not test the spreading
and the basis is VAL-020's excess turbulence over the tops (ADR-015 A), so the corner cell's
position is the model's reading of a region it is known to overstate there.

## 8. Predictions against measurement

<!-- tables76 predictions begin -->
### P1. Prompt 76's predictions against the measurements

| Prediction | Measured | Verdict |
|---|---|---|
| (i) the file's four detect none of the event scenarios' units on 320x120 | 0 of 15 units detected | holds |
| (j) no mountable candidate detects a door unit on 320x120 | 0 of 1,737 mountable candidates detect one; door units uncovered: 5 of 5 | holds |
| (k) the searched mountable four detect every breach and dust unit, worst case under 3 s | door 0 of 5, breach 5 of 5, dust 5 of 5; worst case over the units scored 2.565 s, 10 units scored | holds |
<!-- tables76 predictions end -->

All three hold. (j) holds for the reason it gives: the plume ends over the first return. (k)'s
worst case is the breach's spill reaching the tool's corner; it is under 3 s on this grid, and
over 3 s at no arm of K2 either, which is a coarser reading of the same place.
