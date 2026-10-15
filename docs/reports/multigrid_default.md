# Multigrid as the default pressure preconditioner (prompt 54)

**Date:** 2026-10-14
**Tree:** branch `feature/multigrid-default` from main at 14166aa (prompt 53 merged as PR 80).
Commit 0 (`f7a5de9`) records Alex's decision of 2026-10-14 under ADR-013 decision 1 and in
`docs/reports/pressure_multigrid.md`, section 8. Commit A (`11fb6ad`) fixes the smoother weight.
Commit B (`759b3ad`) makes the default.
**Policy:** `docs/REVIEW_POLICY.md`, "What blocks". The numbers live in generated tables; the
sentences point at them.

## 1. Predictions

Stated by the orchestrator on 2026-10-14 in prompt 54. Committed alone, before item 6's
measurements. The twelve captured systems are prompt 53's: six cases (VAL-001, the closed cavity,
the product room on 160x60 and 320x120, the Annex 20 room on 216x72 and the backward-facing step's
medium grid 656x60), each early (outer 50) and late (outer 2,000, or the last correction of a run
that stops before it).

- (a) The fixed weight changes no captured system's iteration count by more than one, and cuts
  setup by at least half where it was over 10% of the correction.
- (b) Every case in item 7 reaches the same kind of stop as its Jacobi record, within 5% of the
  outer count, and meets its criterion: VAL-001's and VAL-002's as before, VAL-017's allowance,
  VAL-016's split criteria, the product's faces within 1e-3 of the velocity scale, the step's x_r
  within 0.02 H.
- (c) Every laminar face hash changes (the preconditioner changes every iterate) and no laminar
  criterion moves by more than its rounding.

What each outcome means (prompt 54):

- (a) fails: the fixed weight is wrong for some level, and the power iterations stay.
- (b) fails on a case: the default does not change for that case's file until the difference is
  measured; the others proceed.

Stop and report (prompt 54): a case that fails its criterion under the new default; a positivity
error, a capped pressure correction, or a CG solve that reaches its cap.
