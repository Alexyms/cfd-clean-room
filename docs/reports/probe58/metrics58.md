# Prompt 58: the three metrics, defined before anything is computed

**Date:** 2026-10-15 (the machine's local date). Committed alone, before `tables58.py` exists and
before any number is read from a record. Alex's decision 2 of 2026-10-15 (ECR-002 criterion 10's
note): the comparative claim is stated per metric, not by one combined score, because step 8's
single score ranked the deliberately poor layout best and a combined score defined now would be
fitted to the result.

## The records

Step 8's transport records, `results/builder56/transport_*.json` (untracked; read where they
are). Each is one march of one source on one converged flow at one turbulent Schmidt number, both
classes inside it (`per_class`), as `docs/reports/ecr002_step8_product.md` section 2.5 defines
the march. The fields read: `source.emission` (the source's emission rate Q), and per class
`by_surface` (the deposition rate per named surface), `sensors` (the steady concentration at each
sensor) and `outflow_rate_over_source` (the share of the emission leaving through the fixed-flow
outlets), with `stop` (steady or cap) carried beside every value.

The layouts enter as step 8 did: L0 and L1 from their own converged runs on both exact grids, and
L2 as step 8's labelled supplement, marked L2* in every table (`tables56.py`'s `SUPPLEMENT`): on
320x120 the rerun with the cap raised, which stops by the rule; on 160x60 the field at the cap, a
cycle the rule does not stop on. A starred row is not a converged product-configuration run, and
the claim is read with that in view. The one march that ends at its own cap rather than by the
steady rule (step 8 section 8) enters as recorded and is flagged in the metric table.

## The metrics

For a layout on a grid at a Schmidt number, per class:

- **M1, process-top deposition.** Step 8's score as section 2.6 there defines it, unchanged: the
  sum over the three sources of the deposition rate on `litho_tool top` plus `etch_chamber top`,
  particles per second per metre of depth at the march's steady state. Lower is better.
- **M2, operator exposure.** The steady concentration at the two operator-height sensors,
  `near_door` and `hood_entry`, each divided by its own source's emission rate Q, summed over the
  two sensors and the three sources. Lower is better. `hood_entry` lies inside S3's square (step 8
  section 2.5), so S3's `hood_entry` term reads the source cell itself; the definition includes it
  and the per-source terms are tabled beside the sum so a reader can see what carries the metric.
- **M3, removal.** The share of the emitted particles leaving through the returns and the hood,
  `outflow_rate_over_source`, averaged over the three sources. Higher is better.

## The ranking and the margins

The cases are step 8's: both grids at Sc_t 0.7, and 320x120 at Sc_t 0.2, 0.7 and 1.3; each
class on its own. On each case and metric the layouts are ordered best first (ascending for M1
and M2, descending for M3).

Each neighbouring difference in that order is set against the change the grid makes to each of
the two values (the absolute difference between the layout's 320x120 and 160x60 values at Sc_t
0.7) and, on 320x120, the change the Schmidt number makes to each (the larger of the absolute
differences between the layout's value at 0.2 and at 1.3 and its value at 0.7), as step 8
section 5.4 did. An order between two neighbours is **robust** when the difference exceeds every
applicable change; otherwise it is a **tie**, reported as a tie and not as an order. The robust
order of a case groups tied neighbours, written `A ~ B`. Each difference is also given relative
to the larger of the two values.

## The outputs

`tables58.py` generates, from the records: the metric table (every value, per source and
combined, with the stop flag), the ranking table (the order per case and metric, ties grouped),
the margin table (every neighbouring difference against its changes) and a summary (per metric
and class, whether one robust order holds across every case). It writes `tables58.md` and
`tables58.json` beside itself and the figure `docs/reports/phase3_metrics.png`: the three layouts
against the three metrics, one panel per metric and class, the cases as marker shapes and the
layouts in a fixed colour order, the robust order written in each panel with ties marked.
`tables58.py --check` regenerates the tables and compares them with the committed files, exiting
non-zero on any difference.
