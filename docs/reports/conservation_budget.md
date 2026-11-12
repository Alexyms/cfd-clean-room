# The Conservation Budget of the Implicit Transport Solve (REQ-T05)

**Date:** 2026-11-12
**Prompt:** 78 (`docs/prompts/cc-prompt-78-conservation-budget.md`)
**Branch:** `feature/conservation-budget`
**Status:** predictions committed before any run. The rest of the report follows the runs.

## The predictions

Set by the orchestrator in prompt 78 on 2026-11-12, before any run of this pass, and
committed alone before the first run.

- (a) With the leftover booked, every 80x30 run's budget closes within 1e-12 of its supplied
  particles at every stored frame, every class.
- (b) The accumulated leftover stays within `f * L * S(t)` (1e-5 of the supply) at every step of
  every run, and ends smaller in magnitude than prompt 77's T5 for the same run and class.
- (c) Mean sweeps per solve rise by less than one, and march wall time by less than 15%, on every
  80x30 run against prompt 77's.
- (d) Every flow harness row and every flow state reproduces bitwise.
- (e) Every detection latency on 80x30 is within one time step of the run before this change.

The prompt's stop conditions: a flow row or flow state changes; the share cannot be met at
the configured values; a planted defect does not fail its test; the estimate exceeds the
4-hour budget. A failed prediction other than (d) is a finding, not a stop.
