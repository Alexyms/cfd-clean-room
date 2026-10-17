# ADR-014 Step 0b: The Equipment Vents

**Date:** 2026-10-16 (the machine's local date)
**Prompt:** `docs/prompts/cc-prompt-61-step0b-vents.md`
**Branch:** `phase4/step0b-vents`, from `origin/main` at 3a06c7d (PR #88)
**Design:** ADR-014 section J, with prompt 61's amendment (the recirculated discharge)

## 1. Predictions (orchestrator, 2026-10-17; committed before the runs)

Recorded as given, before any product run started.

- (a) Every hash of record is unchanged for configurations without vents.
- (b) The vented room converges under the stopping rule in no more than 1.3 times the default's
  outer iterations.
- (c) The closed-box test (a discharge and an exhaust of equal flow as the only openings) holds the
  signed domain sum of the imbalance at rounding from the first outer iteration.
- (d) The recirculating discharge on the rack raises the near-door and above-gap-1 sensors' steady
  concentration against the default, because it lifts air from near the floor into the supply's
  path.

## 2. The vented room

`configs/clean_room_vents.yaml` is the default room with three vents; every key outside the
boundaries section is the default file's, and the default file is unchanged
(`tests/test_config.py::test_the_vents_file_is_the_default_room_with_three_vents`). The values
are assumptions, stated in the file's header with what each flow per metre of depth means:

| Vent | Type | Face | Extent (m) | Velocity (m/s) | Flow (m^2/s per metre) |
|---|---|---|---|---|---|
| `rack_intake` | fixed_flow_outlet | server_rack left, x = 1.5 | y 0.2 to 1.4 | 0.25 into the rack | 0.30 |
| `rack_discharge` | velocity_inlet, `recirculate_from: rack_intake` | server_rack top, y = 2.0 | x 1.6 to 2.2 | 0.5 up | 0.30 |
| `etch_exhaust` | fixed_flow_outlet | etch_chamber right, x = 5.6 | y 1.0 to 1.4 | 0.25 into the tool | 0.10 |

The returns that state no velocity then share 2.60 m^2/s per metre, against 2.70 in the default
room; the rule's velocity scale is the discharge's 0.5 m/s, against the supply's 0.45.

## 3. What was built

One commit per layer of section J's "What each layer reads", in the prompt's order, each with
its tests; the recirculated discharge is in layer 5's commit with its amendment to section J.

| Commit | Layer | What it closes |
|---|---|---|
| 01aa611 | Configuration (`config.py`) | `location: obstacle`, `obstacle`, `face`, the extent keys, the two allowed types and every load refusal of section J, one test each (`tests/test_config.py::TestVents`); `recirculate_from` parsed and checked; the closed box of stated flows accepted when balanced (departure 3) |
| 942a487 | Velocity (`mesh.py`, `staggered.py`, `boundary_registry.py`, `boundary_staggered.py`) | the vents' faces through the coverage rule on the obstacle's face coordinates (`Mesh.obstacle_mask`, `obstacle_face_inputs`, `coverage_on`, `vent_runs`); the normal velocity written on exactly the covered faces; the inflow sums, the corrector's `flux_scale`, the remainder rule with the exhausts as stated outflow; the mesh refusals |
| fc50190 | Momentum and pressure (`momentum.py`, `pressure.py` notes) | the vent face kept at its value, not an unknown; the pressure correction unchanged (no coefficient crosses the face); the closed box balanced at every outer iteration |
| ba1b9c6 | Wall treatment (`turbulence.py`, `scalar_scheme.py`) | `wall_faces` and the wall cells without the vent span; a discharge's inflow k and eps; the exhaust's room-cell outflow; `advective_flux(..., vents=)` |
| 5b4421c | Concentration and budget (`boundary_concentration.py`, `solver_transport.py`) | inlet and outlet faces, `SURFACE_NONE` on vents, vent fluxes in the budget; the recirculated discharge and its amendment to section J |
| c0fe534 | Product room | the predictions, then `configs/clean_room_vents.yaml` |
| e3ae27c | Instruments | `hash61.py`, `flow61.py` |
| 89b02c0 | Test | no k or eps diffuses across a vent face (the plant of section 8 that survived round 1) |

Every configuration without vents runs the earlier lines: each new branch is guarded by "is
there a vent", and where an argument was added its default is the old path (`advective_flux`'s
`vents=None`, the momentum sweep's `held=None`).

## 4. Prediction (a): the hashes of record

Every case whose face hash is of record (step 0's list, `docs/reports/adr014_step0.md` section
3), solved from rest at the branch's code by `hash61.py` in the worktree `results/builder61/B`
(e3ae27c, the five layers built and nothing under `src/` changed after), against the record
each hash comes from.

<!-- BEGIN tables61: hashes -->
<!-- END tables61: hashes -->

## 5. Prediction (b): the vented room converges

Both rooms on 320x120 from rest under the stopping rule with the cap 15,000 (`flow61.py room`),
launched together with the hash runs, so their wall times share one load. The worst signed sum is
the domain sum of the corrected faces' imbalance over every outer iteration.

<!-- BEGIN tables61: flows -->
<!-- END tables61: flows -->

## 6. Prediction (c): the closed box

`tests/vent_cases.py`'s box with a discharge on its top and an exhaust of equal flow on its side
as the only openings, as `tests/test_solver_staggered.py::TestVents` solves it (air 55 times as
viscous, so the 16 by 8 box stops by the rule), recorded by `flow61.py box`.

<!-- BEGIN tables61: box -->
<!-- END tables61: box -->

## 7. The metrics against the default

The three Phase 3 metrics (`docs/reports/probe58/metrics58.md`) on the case both rooms share,
320x120 at Sc_t 0.7, each class on its own: prompt 47's three sources marched to the steady rule
or the 600 s cap on each room's frozen faces and eddy viscosity (`transport61.py`, step 8's
march). M3 reads the removal net of what the vents bring in (`tables61.py`'s docstring): the
default room's supply is clean, so there it is step 8's figure; in the vented room the gross
outflow books the room air the rack's intake draws, which the discharge returns.

<!-- BEGIN tables61: metrics -->
<!-- END tables61: metrics -->

Prediction (d), the two sensors it names, per source and class:

<!-- BEGIN tables61: sensors -->
<!-- END tables61: sensors -->

The removal paths behind M3:

<!-- BEGIN tables61: removal -->
<!-- END tables61: removal -->

![The steady velocity near the rack and the etch chamber, both rooms](adr014_step0b_vents.png)

`docs/reports/adr014_step0b_vents.png` (`figure61.py`): the speed at the cell centres on one scale
for the four panels, arrows on every fourth cell, the vents drawn on their faces.

## 8. Planted defects

Each guard step 0b added, removed in the worktree `results/builder61/P` by one exact replacement
(`plants61.py`), with the tests named run against it after a control run of the same tests on the
unplanted file passed. Round 1 ran at e3ae27c: 28 of 29 caught, and the plant that opened the
k-epsilon diffusion across a vent face survived every test (`results/builder61/plants_round1.json`).
Commit 89b02c0 adds the test that pins it; round 2, below, ran every plant again at 89b02c0.

<!-- BEGIN tables61: plants -->
<!-- END tables61: plants -->

## 9. Departures from section J

Recorded in ADR-014 at the end of section J, "As built (step 0b)", each with its reason: no
angled jet on a vent; the tangential viscosity on a vent span is the domain inlet's as built;
a closed box of stated flows loads when balanced; two refusals the section did not list; the
discharge in the velocity scale and the initial state; where the vents are found. The
recirculated discharge is the prompt's amendment, written into section J with its reason, and
section H's limit on the discharge's concentration is amended to match.
