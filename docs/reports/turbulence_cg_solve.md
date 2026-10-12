# k and eps by conjugate gradients (ECR-002 step 7, prompt 51)

**Status:** predictions committed before the runs of item 5; the measurements follow in later commits.

## 1. Predictions

Stated by the orchestrator on 2026-10-12 in prompt 51, with (f) added by Alex
on 2026-10-12 with the fallback decision. Committed alone, before item 5's
runs. A row is the Annex 20 harness row `val017_216x72` or one of prompt
48's product rows on 80x30 and 160x60.

- (a) The laminar hashes, the pressure loop and the transport gate tests are unchanged bit for bit.
- (b) The mean CG iterations per k or eps solve over the Annex 20 run are under 60, against 272 and 182 Jacobi sweeps.
- (c) Seconds per outer iteration over the whole Annex 20 run fall by at least 40%.
- (d) The Annex 20 row stops within 2% of main's outer count, both scored lines pass, and its line values differ from main's by under 1e-4 u0.
- (e) No positivity error in any run.
- (f) Zero fallbacks on the Annex 20 and product rows.

What each outcome means (prompt 51):

- (c) holds: the remaining step 7 cases and step 8 run on this solve, and
  the pressure correction becomes the largest cost; whether to take
  ADR-013's named successor then goes to Alex with this report's numbers.
- (c) fails while (b) holds: the CG loop's own cost per iteration eats the
  gain, and the next pass looks at the loop, not the method.
- (e) fails: the step stops and reports the cell, the iteration and the
  field. No clipping is added.

Item 4's checks of (a) were run at commit A before this commit and are
reported with the rest.
