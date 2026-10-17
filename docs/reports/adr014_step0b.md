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

The measurements follow in later sections, generated from the records.
