# A multigrid preconditioner for the pressure correction (ADR-013's successor, prompt 53)

**Date:** 2026-10-14
**Tree:** branch `feature/pressure-multigrid` from main at 13b0b9a (prompt 52 merged as PR 79).
Commit 0 records Alex's decision of 2026-10-14 under ADR-013 decisions 1 and 2. Commit A builds
the preconditioner (`src/multigrid.py`, `src/pressure.py`, `src/config.py`). Commit B is section 1
alone. Commit C is the probe; commit D this report's results and the records.
**Policy:** `docs/REVIEW_POLICY.md`, "What blocks". The numbers live in generated tables; the
sentences point at them.

## 1. Predictions

Stated by the orchestrator on 2026-10-14 in prompt 53. Committed alone, before item 6's
measurements. A captured system is one of the ten p' systems of item 6: VAL-001, the closed
cavity, the product room on 160x60 and 320x120, the Annex 20 room on 216x72 and the backward-facing
step's medium grid 656x60, each early (outer 50) and late (outer 2,000, or the last correction of
a run that stops before it).

- (a) Under the default every laminar hash and the pressure loop's results are the base's.
- (b) MG-PCG takes 10 to 40 iterations per correction on every captured system, and its count
  grows by less than a factor of two from the product's 160x60 to 320x120 (Jacobi-PCG's grows by
  about 1.9 per doubling, ADR-013 section A).
- (c) Setup is under 10% of a multigrid correction's time on every captured system.
- (d) A correction's total time falls at least fivefold on the step's medium grid and at least
  threefold on Annex 20 and the product's 320x120.
- (e) The whole runs reach the same stop as their `jacobi` records, within 5% of the outer count,
  with answers within the same tolerances prompt 51 used (Annex 20 lines within 1e-4 u0, product
  faces within 1e-3 of the velocity scale, the step's x_r within 0.02 H).
- (f) The step's medium grid runs at least three times faster per outer iteration alone.

What each outcome means (prompt 53):

- (d) and (f) hold: Alex decides on making `multigrid` the default, with VAL-020 and step 8 then
  run on it, and the laminar hashes rebaselined once in that change.
- (b) fails on the step only: the anisotropy defeats the point smoother; line smoothing or
  semi-coarsening is the next question, and the default changes for the uniform-grid cases first.
- (e) fails: a different stopped field at the same `pressure_rtol` means the two preconditioners
  leave different errors at the same residual (ADR-013 section 7.4); the report measures by how
  much before anything is decided.

Stop and report (prompt 53): a laminar hash that moves under the default; MG-PCG missing
`pressure_rtol` within the cap on a captured system, or taking more than 100 iterations on one;
a positivity error or a capped pressure correction in a whole run.
