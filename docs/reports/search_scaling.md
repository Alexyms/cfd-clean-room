# How Far the Exact Placement Search Scales (prompt 82)

**Date:** 2026-11-14
**Prompt:** 82 (`docs/prompts/cc-prompt-82-search-scaling.md`)
**Branch:** `phase5/search-scaling`
**Status:** predictions committed; measurements not yet made.

The Phase 5 follow-up scores the dust release from several positions on the tool tops. Each
position adds five units (one per channel) to the four scenarios' twenty. The worst-case rule's
exact search (`placement._worst_case_cover`) keeps one entry per subset of the units, so its
cost doubles per unit, and it refuses above `MAX_SET_COVER_UNITS = 20`. This pass measures how
far that search, an exact enumeration over the maximal coverage masks, and the mean rule's
greedy search carry on this laptop. Nothing under `src/` changes; the probe is
`docs/reports/probe82/`, and no scenario is run.

## The predictions

Set by the orchestrator in prompt 82 on 2026-11-14, and committed alone before any measurement
of this pass.

- (a) The built set cover is limited by time before memory: on this laptop it finishes within
  30 minutes up to no more than 26 units at four sensors.
- (b) The maximal masks at the optimum's threshold stay below 2,000 up to 60 units.
- (c) The enumeration of combinations of at most four maximal masks returns the built cover's
  worst case wherever both run, and finishes within 5 minutes at 60 units.
- (d) Where the exact worst-case optimum is known, the greedy mean search's worst case is within
  10% of it.

The prompt's stop condition: the enumeration and the built cover disagree on a worst case
anywhere, since one of them is then wrong. A failed prediction other than that is a finding.
