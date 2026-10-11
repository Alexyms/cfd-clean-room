# ECR-002 Step 6: k and eps by Upwind, and the Variant Check

**Date:** 2026-10-10
**Tree:** branch `feature/ecr002-turbulence-advection` from main at 87283de. Commit A adds the
turbulence key `advection_scheme` (`upwind` or `umist`, absent meaning `upwind`) and advects k and
eps with it; nothing else under `src/` changes. Every flow run goes through the committed
`StaggeredSolver` with the key absent, so the default holds, and every particle march through the
committed `TransportSolver` with the converged flow handed to it.
**Instruments:** `docs/reports/probe48/` (committed after this section, before any run):
`coupled48.py` (prompt 47's runner, its `control` arm, nothing substituted), `transport48.py`
(prompt 47's march), `tables48.py` (the tables, prompt 47's where they apply), `figures48.py`
(prompt 47's figures), `bounded48.py` and the launcher `run48.sh`. Commit A's checks are
`check48.py` (the laminar hashes) and `plant48.py` (the planted defects). Raw output under
`results/builder48/` (untracked).
**Order:** sections 1 to 4 were committed before any run of item 4, this file alone. Section 5
onward was written after the runs.
**Policy:** `docs/REVIEW_POLICY.md`, "What blocks". The numbers live in the tables `tables48.py`
generates from the records; the sentences point at the tables.

## 1. The question

Alex's decisions of 2026-10-10 (ECR-002 section 8, step 6's note; ADR-012 C's note): k and eps
are advected by plain upwind by default, the product grid is 320x120 with 160x60 its coarser
check, and RNG is rerun under upwind on that pair. Two questions follow. Does the coupled model,
with k and eps advected by upwind, converge the product room on the exact grid pair for both
variants? And where both converge, does the variant change where particles collect? The answers
settle step 8's configuration and tell step 7 which schemes to run.

## 2. Method (fixed before the runs)

**The room and the settings** are prompt 47's (`docs/reports/ecr002_step6_comparative.md`,
section 2.1): `configs/clean_room_default.yaml` regridded, the turbulence section of prompt 46
(`cfl_number` 0.25, `alpha_turbulence` 0.7, `max_iter` 500, `tol` 1e-10, the baseline inlet
turbulence) with no `advection_scheme` key, the solver keys of step 5's measurement
(`error_estimate`, `momentum_sweeps` 10, `pressure_rtol` 1e-4, `max_simple_iter` 10,000), from
rest, one BLAS thread, the cell of the largest change recorded every outer iteration. The
classification is step 5's rule as prompt 47 applied it.

**Item 4, parity.** The standard variant on 80x30, 160x60 and 320x120 through the key. Prompt 47's
`upwind` arm passed `upwind=True` to the same `advective_flux` from a subclass of the model; the
key does the same from the configuration. Each row is compared with prompt 47's record of the arm
(`results/builder47/standard_GRID_upwind`): the stop, the outer count (1,046, 2,003 and 4,472),
the face hash (`28f437c84e3d5057` on 80x30) and, beyond what the prompt asks, every saved field
(u, v, k, eps, nu_t, p) and the residual history, bit for bit. A row that does not reproduce stops
the build.

**Item 5, the variants.** Standard and RNG on 160x60 and 320x120 under the default. The standard
rows are item 4's. Each converged run gets the three-panel figure (streamlines coloured by speed,
nu_t / nu on a log scale, k, the room's boundaries marked) and a row of the converged table; a
bounded run gets bounded44's characterisation.

**Item 6, the variant check.** On each exact grid where both variants converge, prompt 47's three
sources (S1 beside the door wall, S2 above the litho tool, S3 at the hood bench, 0.1 m squares) and
two classes (0.5 and 5 micrometres), marched as prompt 47 marched them (its section 2.5: transport
Courant number 0.4, the converged faces and eddy viscosity frozen, from a zero field, the steady
rule, 600 s cap). For each source and class: the largest deposition location under each variant,
the hotspots in common among the five largest 0.2 m segments, and the order of the sensors above
the floor (1e-6 of the source-cell concentration). The same for RNG between the two grids. The
standard variant's marches are rerun on this build's rooms, so the standard rows of every
comparison come from this build's records; they are compared with prompt 47's marches on the same
rooms. Counts and orders, not percentages.

## 3. Predictions

### 3.1 Orchestrator (2026-10-10; committed before item 4's runs)

- (a) The key reproduces probe 47's three upwind rows bit for bit.
- (b) RNG converges on both exact grids under upwind. Its 19-iteration cycle was the limiter's.
- (c) Between variants on each grid, for every source and class: the largest deposition location
  is the same segment, and at least four of the five hotspots are shared.
- (d) RNG agrees between grids as standard did: same largest location, at least four of five
  shared.

What each outcome means:

- (b) fails: RNG's cycle is not the limiter's, and the product runs the standard variant alone
  until VAL-020 decides between them.
- (c) fails: the variant decides the answer, and VAL-020 (the impinging jet) becomes the deciding
  validation before step 8.

### 3.2 Builder (2026-10-10, before item 4's runs; scored apart)

- (a) Holds, every saved field and the residual history included: the key and the arm make the
  same calls with the same arguments.
- (b) RNG converges on 160x60 and stays bounded on 320x120. Its cycle in prompt 47 had a velocity
  step about a twelfth of the scale and a per-cell imbalance tens of times its tolerance, larger
  than the standard variant's limiter cycle by orders, and on 200x75 (prompt 46) it was a real
  oscillation of about 4 cm/s beside the litho tool's west face; I read that as a flow-level
  oscillation that RNG's lower eddy viscosity in strained regions permits, which the face value
  of k and eps damps on the coarser grid and not on the finer one.
- (c) On every grid where both converge, the largest location is the same segment for every source
  and class, and at least four of five hotspots are shared for S1 and S2. For S3, whose last three
  hotspots are hood-bench segments at rates five orders below its first, I expect four of five
  for one class at least and do not predict the other.
- (d) Not scorable if (b)'s second half holds.

## 4. Stop conditions (from the prompt)

- Item 4's parity fails.
- A laminar hash moves (VAL-001, its stretched twin, VAL-002 against the base), or a transport
  gate test changes.
- A positivity error, or a capped pressure correction, in any run.

## 5. Results (written after the runs)

### 5.1 Order and cost

The five flow rows of items 4 and 5 were launched together at 16:08 (`run48.sh flows`), beside
commit A's three laminar rows of item 3 and the base's VAL-002 row, nine processes on twelve
cores; the marches of item 6 followed as the standard rows converged. The wall times in the
tables were taken beside other runs.

### 5.2 Item 3: what the default leaves alone

<!-- tables48 laminar begin -->
| Case | Stop: base, commit A | Outer: base, commit A | Face hash: base, commit A | Equal |
|---|---|---|---|---|
| val001_80x40 | error_estimate_and_continuity, error_estimate_and_continuity | 1,559, 1,559 | c7d88d919e1141a5, c7d88d919e1141a5 | yes |
| val001_80x40_stretched | error_estimate_and_continuity, error_estimate_and_continuity | 1,124, 1,124 | 9838d632f181d90f, 9838d632f181d90f | yes |
| val002_80x80 | error_estimate_and_continuity, error_estimate_and_continuity | 12,814, 12,814 | a8a61e28120e3f44, a8a61e28120e3f44 | yes |
<!-- tables48 laminar end -->

The three laminar cases stop at the same iteration with the same face hash at the base (main at
87283de, measured in a worktree) and at commit A; the full SHA-256 values are equal in the
records. The full suite passes at commit A, the transport gate tests among them. No transport
test constructs `KEpsilonModel`: a pytest plugin that wraps `KEpsilonModel._advect` and computes
both schemes on every call (kept with the records, `results/builder48/moves_commitA.json`) lists
every test that advects k or eps, and no transport test is on it.

The tests whose computed values moved with the default, found by the same plugin on the suite
before any test was changed (`moves_upwind.json`), all pass under it:

- run under both schemes now, each with its `umist` case kept: `TestPositivity`'s positivity
  cases, the long run with truncated solves and the planted explicit decay that controls them;
  `TestBoundaries::test_nothing_crosses_a_wall_or_an_obstacle_face`; and
  `test_uniform_k_and_eps_stay_uniform_on_the_val001_faces` (`tests/test_turbulence.py`);
- left on the default, since none exists to exercise a face value:
  `TestPositivity::test_the_assertion_names_eps_when_only_eps_goes_negative` and
  `TestTwoKindsOfStep::test_the_cap_keeps_the_growth_of_a_cell_at_rest_finite`
  (`tests/test_turbulence.py`); `TestRefusals::test_a_positivity_error_names_the_outer_iteration`,
  `TestWhatTheSolverExposes::test_condition_e_receives_the_largest_change_of_nu_t` and
  `TestWhatTheSolverExposes::test_the_state_is_read_only_and_the_stage_is_timed`
  (`tests/test_coupled_solve.py`); and VAL-016's Couette solve, which the module fixture of
  `tests/test_turbulent_channel.py` shares between `test_couette_converges_under_rule_version_4_val016`
  and `test_couette_equals_its_same_grid_reference_val016`. VAL-016's split criteria pass under the
  default: (a), the two-dimensional solve against its own one-dimensional limit, and (b), which
  runs the reference alone.

One test failed on the new field, not on a value:
`TestTurbulenceSection::test_present_section_is_read_with_standard_the_default`
(`tests/test_config.py`) compares a `TurbulenceSpec`, which now carries `advection_scheme`; it
names the field. The planted defects
of commit A (`plant48.py`, record `results/builder48/plants.json`): the limited value used under
`upwind`, upwind used under `umist`, a `umist` default and any string accepted as the key, each
failing at least one of the new tests, with no failure in the control.

### 5.3 Item 4: parity

<!-- tables48 parity begin -->
| Row (key absent) | Scheme parsed | Stop: this build, prompt 47 | Outer: this build, prompt 47 (expected) | Face hash: this build, prompt 47 | u, v, k, eps, nu_t, p and the residual history bitwise equal | Reproduces |
|---|---|---|---|---|---|---|
| standard_80x30_upwind | upwind | error_estimate_and_continuity, error_estimate_and_continuity | 1,046, 1,046 (1,046) | 28f437c84e3d5057, 28f437c84e3d5057 | yes | yes |
| standard_160x60_upwind | upwind | error_estimate_and_continuity, error_estimate_and_continuity | 2,003, 2,003 (2,003) | 73f5eaded654a1a7, 73f5eaded654a1a7 | yes | yes |
| standard_320x120_upwind | upwind | error_estimate_and_continuity, error_estimate_and_continuity | 4,472, 4,472 (4,472) | a055b8e30d9b00ca, a055b8e30d9b00ca | yes | yes |
<!-- tables48 parity end -->

Every row reproduces prompt 47's upwind arm: the stop, the count, the face hash, every saved
field and the residual history of every outer iteration.

### 5.4 Item 5: the variants

<!-- tables48 matrix begin -->
| Run | Class | Stop | Outer | Residual: least (at), end | Readings at stop (a), (b), (c), (d), (e) | Holds from (a), (b), (c), (d), (e) | Largest speed at end (m/s), cell | Peak speed | CG per correction: mean, largest | Cap hits | k, eps sweeps (mean) | Wall (s), per outer | Face hash |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| standard_80x30_upwind | converged | error_estimate_and_continuity | 1,046 | 1.68e-12 (1,046), 1.68e-12 | 5.9e-09, 6.8e-14, 2.2e-12, 2e-15, 9.8e-07 | 788, 192, 181, 1, 1,046 | 1.41, (2.75, 1.35) | 1.63 | 253, 287 | 0 | 19.5, 15.8 | 35, 0.033 | 28f437c84e3d5057 |
| standard_160x60_upwind | converged | error_estimate_and_continuity | 2,003 | 5.49e-13 (2,002), 5.49e-13 | 9.4e-09, 5.8e-14, 3.4e-12, 5.5e-15, 1e-06 | 1,432, 398, 374, 1, 2,003 | 1.54, (0.525, 0.025) | 1.75 | 399, 574 | 0 | 37.4, 28.2 | 186, 0.093 | 73f5eaded654a1a7 |
| rng_160x60_upwind | bounded (stalled) | max_simple_iter | 10,000 | 6.53e-04 (244), 8.45e-04 | -, 2.1e-07, 2.8e-05, 5.5e-15, 1.1e+05 | -, -, -, 1, - | 1.57, (0.525, 0.025) | 1.91 | 282, 575 | 0 | 32.7, 28.5 | 822, 0.082 | e5d3c22cd77d0ff0 |
| standard_320x120_upwind | converged | error_estimate_and_continuity | 4,472 | 1.94e-13 (4,472), 1.94e-13 | 1.2e-08, 1.4e-13, 3.7e-12, 8.6e-15, 1e-06 | 3,530, 1,068, 762, 1, 4,472 | 1.69, (0.5125, 0.0125) | 2.13 | 713, 1118 | 0 | 87.8, 63.7 | 2466, 0.552 | a055b8e30d9b00ca |
| rng_320x120_upwind | bounded (stalled) | max_simple_iter | 10,000 | 4.41e-04 (1,304), 5.25e-04 | -, 1.6e-07, 4.4e-05, 8.6e-15, - | -, -, -, 1, - | 1.75, (0.5125, 0.0125) | 2.42 | 414, 1121 | 0 | 65.5, 54.6 | 3663, 0.366 | b34ff4fa130e1294 |
<!-- tables48 matrix end -->

<!-- tables48 converged begin -->
| Run | Outer, stop | Readings at stop (a), (b), (c), (d), (e) | Holds from (a), (b), (c), (d), (e) | Wall (s) | Inlet nu_t / nu | Core nu_t / nu: median, 95th | y+ nodes | y+: median, least, largest | Share below 11.53: all; domain, tops, sides | Figure |
|---|---|---|---|---|---|---|---|---|---|---|
| standard_80x30_upwind | 1,046, error_estimate_and_continuity | 5.9e-09, 6.8e-14, 2.2e-12, 2e-15, 9.8e-07 | 788, 192, 181, 1, 1,046 | 35 | 16.4 | 17.4, 56.5 | 257 | 105, 8.19, 536 | 0.023; 0.062, 0, 0.0072 | ecr002_step6_standard_80x30_upwind.png |
| standard_160x60_upwind | 2,003, error_estimate_and_continuity | 9.4e-09, 5.8e-14, 3.4e-12, 5.5e-15, 1e-06 | 1,432, 398, 374, 1, 2,003 | 186 | 16.4 | 17.1, 58 | 514 | 61.8, 4.08, 296 | 0.035; 0.094, 0, 0.011 | ecr002_step6_standard_160x60_upwind.png |
| standard_320x120_upwind | 4,472, error_estimate_and_continuity | 1.2e-08, 1.4e-13, 3.7e-12, 8.6e-15, 1e-06 | 3,530, 1,068, 762, 1, 4,472 | 2466 | 16.4 | 17, 56.2 | 1,028 | 34.3, 2.11, 167 | 0.096; 0.24, 0, 0.038 | ecr002_step6_standard_320x120_upwind.png |
<!-- tables48 converged end -->

The bounded rows' tails, with the regions of largest change:

<!-- tables48 bounded begin -->
| Run | Outer, tail | Residual: least, median, largest | Amplitude p95/p5 | Drift (log10 per 1,000) | Period (height); strongest (height) | Largest speed: least, largest; cells | Located: share by region (largest of u, v); cells | Largest nu_t change: share by region; cells; size (m^2/s) |
|---|---|---|---|---|---|---|---|---|
| rng_160x60_upwind | 10,000, 2,000 | 7.08e-04, 8.69e-04, 9.58e-04 | 1.27 | 0.000142 | 10 (0.77); 59 (0.96) | 1.57, 1.57; (0.525, 0.025) | gap between equipment 1.00; (2.925, 1.475), (2.875, 1.275), (2.875, 1.325) | gap between equipment 1.00; server_rack face 0.00; (1.175, 1.575), (1.175, 1.525), (1.125, 0.675); 0.00011 |
| rng_320x120_upwind | 10,000, 2,000 | 4.73e-04, 5.32e-04, 5.90e-04 | 1.16 | -2.13e-05 | 7 (0.26); 183 (0.84) | 1.75, 1.75; (0.5125, 0.0125) | gap between equipment 0.80; hood_bench face 0.20; (2.9125, 1.5375), (2.8875, 1.4375), (2.8875, 1.4625) | gap between equipment 1.00; (2.9625, 0.4125), (2.9625, 0.7625), (2.9625, 0.7375); 8.1e-05 |
<!-- tables48 bounded end -->

RNG's two rows under upwind beside prompt 47's two under the limited value:

<!-- tables48 rng begin -->
| Grid, k and eps face value | Stop, outer | Velocity step: tail median, at the stop (m/s); scale | Worst cell imbalance at the stop; tolerance | Residual over the tail: median | Period (height); strongest (height) |
|---|---|---|---|---|---|
| 160x60, upwind (this build) | max_simple_iter, 10,000 | 0.0548, 0.0532; 0.45 | 2.1e-07; 5e-09 | 0.000869 | 10 (0.77); 59 (0.96) |
| 160x60, umist (prompt 47) | max_simple_iter, 10,000 | 0.0349, 0.036; 0.45 | 1.7e-07; 5e-09 | 0.000553 | 10 (0.24); 19 (0.99) |
| 320x120, upwind (this build) | max_simple_iter, 10,000 | 0.067, 0.0662; 0.45 | 1.6e-07; 1.3e-09 | 0.000532 | 7 (0.26); 183 (0.84) |
| 320x120, umist (prompt 47) | max_simple_iter, 10,000 | 0.0365, 0.0353; 0.45 | 6.4e-08; 1.3e-09 | 0.000289 | 9 (0.31); 19 (0.73) |
<!-- tables48 rng end -->

The standard variant converges on both exact grids; its rows are item 4's. RNG converges on
neither: both rows reach the 10,000 cap, bounded and stalled by step 5's rule. No run raised a
positivity error or hit the pressure correction's cap (the matrix's cap-hits column).

RNG's cycle under upwind is not the one prompt 47 found under the limited value, and it is not
smaller. Its largest velocity change sits where that one did, in the gap above return 2 between
about 1.3 and 1.5 m up (the bounded table's located column; prompt 47's report, section 5.3). The
19-iteration recurrence that was clean on both grids under the limited value is gone; the
strongest recurrence is now a longer one on each grid (the RNG table's last column). The velocity
step is larger under upwind on both grids, an eighth to a seventh of the velocity scale against
about a thirteenth, and the tail's residual is higher; the per-cell imbalance at the stop stays
tens of times its tolerance on 160x60 and over a hundred on 320x120 (the RNG table).

### 5.5 Item 6: the variant check

No grid has both variants converged, so the variant check has no pair on either grid, and RNG
has no comparison between grids. What ran is the standard variant's three sources on both exact
grids, as section 2 fixed. They reproduce prompt 47's marches of the upwind arm on the same rooms
bit for bit (the first table below), so every march's numbers are those of prompt 47's report,
section 5.4, and its concentration figures stand unchanged: `figures48.py` redrew all nine
figures of the standard rows from this build's records and each is byte for byte the committed
file (`results/builder48/figures.json`). RNG's rows are bounded and have no figure.

The second table is the one comparison the records allow, the standard variant between the two
grids, the reference prediction (d) names. It is prompt 47's supplementary comparison on the
same marches (its section 6.1), with the counts per pair under it.

<!-- tables48 marches begin -->
| March, class (um) | Steady at (s): this build, prompt 47 | Hotspots and their rates equal | Sensors equal | Per-surface deposition equal | Saved fields bitwise equal |
|---|---|---|---|---|---|
| standard_160x60_upwind_S1, 0.5 | 84, 84 | yes | yes | yes | yes |
| standard_160x60_upwind_S1, 5 | 84, 84 | yes | yes | yes | yes |
| standard_160x60_upwind_S2, 0.5 | 51, 51 | yes | yes | yes | yes |
| standard_160x60_upwind_S2, 5 | 51, 51 | yes | yes | yes | yes |
| standard_160x60_upwind_S3, 0.5 | 10, 10 | yes | yes | yes | yes |
| standard_160x60_upwind_S3, 5 | 10, 10 | yes | yes | yes | yes |
| standard_320x120_upwind_S1, 0.5 | 91, 91 | yes | yes | yes | yes |
| standard_320x120_upwind_S1, 5 | 91, 91 | yes | yes | yes | yes |
| standard_320x120_upwind_S2, 0.5 | 55, 55 | yes | yes | yes | yes |
| standard_320x120_upwind_S2, 5 | 55, 55 | yes | yes | yes | yes |
| standard_320x120_upwind_S3, 0.5 | 4, 4 | yes | yes | yes | yes |
| standard_320x120_upwind_S3, 5 | 4, 4 | yes | yes | yes | yes |
<!-- tables48 marches end -->

<!-- tables48 compare begin -->
| Pair, source, class (um) | Hotspots in common (of 5) | In common | Largest: first | Largest: second | Same surface | Sensor order above the floor: first | Second | Orders equal |
|---|---|---|---|---|---|---|---|---|
| grids, standard, S1, 0.5 | 5 | floor | x 1.2; floor | x 1.4; server_rack west | y 0.0; server_rack west | y 0.2; server_rack west | y 0.4 | floor | x 1.2 | floor | x 1.2 | yes | near_door | near_door | yes |
| grids, standard, S1, 5 | 5 | floor | x 1.2; floor | x 1.4; server_rack west | y 0.0; server_rack west | y 0.2; server_rack west | y 0.4 | floor | x 1.2 | floor | x 1.2 | yes | near_door | near_door | yes |
| grids, standard, S2, 0.5 | 5 | litho_tool top | x 3.6; litho_tool top | x 3.4; litho_tool top | x 3.8; litho_tool top | x 3.2; floor | x 3.0 | litho_tool top | x 3.6 | litho_tool top | x 3.6 | yes | none | none | yes |
| grids, standard, S2, 5 | 5 | litho_tool top | x 3.6; litho_tool top | x 3.4; litho_tool top | x 3.8; litho_tool top | x 3.2; floor | x 3.0 | litho_tool top | x 3.6 | litho_tool top | x 3.6 | yes | none | none | yes |
| grids, standard, S3, 0.5 | 4 | floor | x 6.2; hood_bench west | y 0.0; hood_bench west | y 0.2; hood_bench west | y 0.4 | floor | x 6.2 | floor | x 6.2 | yes | none | none | yes |
| grids, standard, S3, 5 | 5 | floor | x 6.2; floor | x 5.6; hood_bench west | y 0.0; hood_bench west | y 0.2; hood_bench west | y 0.4 | floor | x 6.2 | floor | x 6.2 | yes | none | none | yes |

Per pair, over its sources and classes:
| Pair | Rows (source, class) | Largest the same segment | At least four of five shared | Fewest shared | Sensor orders equal |
|---|---|---|---|---|---|
| grids, standard | 6 | 6 | 6 | 4 | 6 |
<!-- tables48 compare end -->

## 6. Predictions against the measurement

| Prediction | Measured |
|---|---|
| (a) The key reproduces probe 47's three upwind rows bit for bit | **Holds.** All three rows, every saved field and the residual history of every outer iteration included (section 5.3) |
| (b) RNG converges on both exact grids under upwind. Its 19-iteration cycle was the limiter's | **Fails.** Bounded at the cap on both grids (section 5.4). The 19-iteration recurrence went with the limited value, but the non-convergence did not: the iteration still cycles in the gap above return 2, with a larger velocity step than before |
| (c) Between variants on each grid, for every source and class: the largest deposition location is the same segment, and at least four of the five hotspots are shared | **Not scorable.** No grid has both variants converged |
| (d) RNG agrees between grids as standard did: same largest location, at least four of five shared | **Not scorable.** RNG converges on neither grid. The standard variant's comparison between grids, the reference (d) names, holds as prompt 47 found it (section 5.5) |

The outcome as section 3.1 wrote it: (b) fails, so RNG's cycle is not the limiter's, and the
product runs the standard variant alone until VAL-020 decides between the variants.

### 6.1 Builder's predictions (section 3.2), scored

(a) held as written. (b) was wrong on 160x60 (bounded, where I expected it to converge) and right
on 320x120; the reading behind it, that the face value of k and eps damps a flow-level
oscillation on the coarser grid, is contradicted on both grids, since the velocity step grew under
upwind. (c) and (d) are not scorable.

## 7. What this implies

Stated as questions; the decisions are Alex's.

- **The variant for step 8.** By section 3.1's reading of (b), the product runs the standard
  variant alone until VAL-020 decides between the two. ADR-012 decision 2 named RNG for the
  product and the standard model for the published comparisons; the measured standing is the
  other way round on every exact grid. Does decision 2 take a dated amendment now, or wait for
  VAL-020?
- **VAL-018's criterion.** Criterion 10's note asks for the hotspots and the ranking to be stable
  under grid refinement, under the two variants and across the turbulent Schmidt number's range.
  The variant clause cannot be scored while RNG converges on no exact grid. Does VAL-018 run
  without it until VAL-020 decides, or does step 8 wait for RNG?
- **RNG's non-convergence.** Under both face values it cycles in the same gap, at a velocity step
  of several centimetres per second, with the per-cell imbalance far above its tolerance; the
  standard variant on the same rooms converges. Whether that is a steady RANS solution that this
  iteration does not reach (a smaller `cfl_number`, `alpha_turbulence` or `alpha_velocity` might
  settle it) or no steady solution at all (ADR-012 J: a finding to report, not something to tune
  away) is not measured here.
- **Step 8's configuration**, as far as this build reaches: the standard variant, k and eps
  advected by upwind (the default), on 320x120 with 160x60 its check, both of which converge
  (section 5.4).

## 8. What this does not settle

- RNG's cycle: located, timed and set beside prompt 47's; not probed. No relaxation factor,
  Courant number or sweep count was changed.
- Whether upwind's smearing of k and eps matters anywhere: step 7's validation cases run both
  schemes (ECR-002 section 8, step 7's note of 2026-10-10).
- The sources are prompt 47's, and all three sit in capture zones (its section 7); the sensor
  orders compare one sensor or none.
