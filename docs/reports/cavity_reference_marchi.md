# Lid-driven cavity: an independent reference for the gap to Ghia

**Date:** 2026-09-23
**Context:** section 9 of `docs/reports/cavity_self_convergence.md` left open whether the
staggered solution's converged distance from Ghia et al. in the jet by the right wall, about
0.005 in u and 0.008 to 0.009 in v, is Ghia's error or this scheme's.
**Instrument:** `python scripts/self_convergence.py --marchi`, which reads saved fields, solves
nothing, and writes `results/self_convergence/marchi_comparison.json`. The reference is
`marchi_2009_re100` in `validation/metrics.py`, and `tests/test_validation.py`,
`TestMarchiTable`, guards it. The extraction scratch and its outputs are under
`results/reference/` (gitignored). Provenance labels are as in the self-convergence report:
MEASURED, INFERRED, HYPOTHESISED, UNKNOWN.

**Answer, INFERRED: the gap is Ghia's.** Extrapolated from 80x80 and 100x100, the staggered
solution lies within 1.9e-5 of Marchi at all 30 of Marchi's points. Ghia's table lies about
0.005 below Marchi in u near y = 0.85, and 0.007 to 0.009 above it in v at x = 0.80 to 0.91,
which is the gap section 9 found. Nothing here switches VAL-002, the harness or any criterion.
They stay on `ghia_1982_re100_r2`.

## 1. The reference

Marchi, C. H., Suero, R. and Araki, L. K. (2009). "The lid-driven square cavity flow:
numerical solution with a 1024 x 1024 grid." J. Braz. Soc. Mech. Sci. & Eng. 31(3). DOI
10.1590/S1678-58782009000300004.

The method, as the paper describes it:

- Finite volumes, second order. Advection uses central differences through a deferred
  correction on upwind.
- Co-located variables, SIMPLEC coupling, and uniform grids from 2x2 to 1024x1024.
- Each grid iterated to machine round-off.
- Up to nine Richardson extrapolations per variable; the profiles get six. The error estimate
  U, from the paper's Eq. (13), is the difference between the two finest extrapolated values.

Each profile value is the mean of the face velocities of the two volumes either side of the
station, on x = 1/2 or y = 1/2. So it sits on the true centerline, as the r2 metric does. The
scheme differs from ours in both variable arrangement and advection (staggered QUICK here),
so a shared limit is not a shared discretization error.

For Re = 100, Table 6 gives u on x = 1/2 at y = k/16 and v on y = 1/2 at x = k/16, for k = 1
to 15. It also gives M, the integral of v(x, 1/2) over x from 0 to 1/2 (Eq. 4). Table 7 gives U
for each of these.

**A premise corrected:** the prompt called the stations unevenly spaced. They are sixteenths.
That makes the quadrature checks in section 3 sharper than Ghia's, and it gives them a blind
spot, also described there.

## 2. Reading the table

The PDF was read by three routes. All three results are MEASURED:

- A: `pdftotext -layout` (xpdf 4.00).
- B: `pypdf` 6.19.0 `extract_text`. It was installed outside the project environment and is
  not a dependency.
- C: `pypdf` glyph positions. Each label fragment goes to the value row with the nearest
  baseline.

- **Text layer.** Tables 6 and 7 are text, not images.
- **Digits.** All three routes return 41 rows of 5 values from each table. Compared as
  strings, they are identical digit for digit in all five Reynolds columns. pypdf splits some
  numbers inside ("1.0662 8389e-1"). The tokens are rejoined until each forms one number, and
  the digit comparison with route A would expose a bad join.
- **Column.** All three read the header as Re = 0.01, 10, 100, 400, 1000. Every row has five
  values, so the third is Re = 100. No Re = 100 value in Table 6 equals the value at the same
  row in another column. In Table 7, equal values occur only in the five coordinate rows (the
  positions of the extrema). There each column's U is 4.9e-4 or 9.8e-4, one or two half-cells
  of the 1024 grid. None occurs in the 31 rows stored.
- **Labels.** Here the routes did not agree. Route A prints every Table 7 label one line below
  its own values: all 41 rows, with the last label standing alone. It also drops the psi glyph
  in both tables. Route B keeps each label on its values' line. Route C settles which is right.
  Each label's baseline sits 0.8 to 1.4 pt below its own row's baseline (1.3 to 1.9 pt in
  Table 6), and the rows are 9.7 to 10.4 pt apart. Two facts in the table agree with the
  positions:
  - The five coordinate rows receive 4.9e-4 or 9.8e-4. The error of a position found on the
    grid is a multiple of the grid spacing.
  - v(0.5; 0.5) at Re = 0.01, printed to 1e-13, receives U = 3.7e-12.

  With route A's labels every stored U would be one station off. For example, u at y = 0.5
  would carry 2.8e-8 instead of 8.6e-9. The stored table uses the positional pairing.
- **Stored table.** The rows in `validation/metrics.py` were written by a script from the
  extraction output, not typed. They equal it exactly: 15 u rows, 15 v rows, M and its U,
  compared as floats parsed from the extracted strings.

## 3. The table checked against itself

The trapezoid rule's error on these stations was estimated first, from the table alone, as
(T(h) - T(2h)) / 3. MEASURED: 9.2e-4 for v over [0, 1/2], 3.2e-4 for u over [0, 1/2], 4.1e-5
for v over [0, 1], and 2.6e-3 for u over [0, 1], the last from u's rise to the lid. The
tolerance was then fixed at 0.005, about twice the largest. The Ghia check used 0.02 on
uneven stations.

Two published facts constrain the profiles besides zero net flux through each centerline:

- v over [0, 1/2] is M.
- u over y in [0, 1/2] on x = 1/2 is -M. The two half-lines bound the lower left quadrant, so
  what enters it leftward below the center leaves it upward left of the center.

M is its own row of Table 6, so the second fact ties the u column to it as well.

MEASURED:

| Check | Trapezoid | Simpson |
|---|---|---|
| v over [0, 1/2], minus M | -9.2e-4 | -1.4e-6 |
| u over [0, 1/2], plus M | +3.2e-4 | +4.3e-6 |
| v over [0, 1] | +2.4e-5 | -1.7e-5 |
| u over [0, 1] | +2.4e-3 | -1.4e-4 |

Each trapezoid miss is the size its estimate predicted. Simpson's rule, which is the
trapezoid corrected by that estimate, brings both half-line integrals to within 4.3e-6 of the
published M. That is two independent confirmations, one per column.

The planted controls, MEASURED:

| Control | v over [0, 1/2], minus M | v over [0, 1] |
|---|---|---|
| the Re = 400 v column | +3.8e-2 | -3.0e-4 |
| v at x = 0.25 and 0.75 swapped | -2.6e-2 | +2.4e-5 |
| each v read from the row above (route A's offset, applied to Table 6) | +3.1e-2 | +4.5e-2 |

**What the checks cannot see.** On uniform stations the full-line trapezoid weights every
interior station alike, so it cannot see any permutation of them. Another Re column is a
real profile and conserves mass too. Only the M check catches those two, and it cannot see a
swap within one half of the line. The tests carry the swap and the offset. The Re = 400
control is recorded here only, because testing it would mean storing its 15 values as well.

The table passes both checks and every control fails, so no stop condition was reached.

## 4. The comparison

**Fields.** The staggered solver at 20x20, 40x40 and 80x80 from `--solve-tight`, and at
100x100 from test 21b (`results/tester21b/staggered_100_tight.npz`). Each is the snapshot where
the residual first fell below 1e-9. Recovering the faces from the far wall leaves a residual
below 1.3e-15 on all four. The 60x60 field exists but the two pairs asked for do not need it.

**Profiles.** As section 9 takes them, with `face_profiles`: the exact staggered faces on
x = 1/2 and y = 1/2, walls appended, over the lid speed from the case.

**Interpolation.** To each Marchi station by the cubic through the four nearest nodes. On
80x80 the stations are face lines midway between two nodes; on 100x100 they are not. The
quintic through six nodes is the check. The largest cubic-to-quintic change is 9.8e-6 in u and
1.4e-6 in v at 80x80, and 3.0e-6 and 2.6e-7 at 100x100. In R(80, 100) it moves no value by
more than 9.1e-6. As section 9 notes, this estimates the cubic's error; it is not a strict
bound.

**Extrapolation.** R(40, 80) = f80 + (f80 - f40) / 3. R(80, 100) = f100 + (f100 - f80) /
(1.25^2 - 1). The order from 20, 40 and 80 at each station runs from 1.93 to 2.52 in u, and
from 1.95 to 3.07 in v. The 3.07 is v at x = 0.5, where the steps are below 1e-5.

**Ghia column, indicative.** Ghia's stations lie on his 1/128 grid and Marchi's on
sixteenths. A Marchi station is paired with the nearest interior Ghia station within 3/128.
Ghia's value is carried to Marchi's station along R(80, 100), Ghia(s_g) + R(s_m) - R(s_g),
and Marchi is subtracted. At the shared stations 0.0625 and 0.5 the carry is zero and the
entry uses no solver field. Elsewhere the carry reaches 0.094 in u, near the lid, and the
entry is as good as R's shape between the two points.

MEASURED, u on x = 1/2:

| y | Marchi u | Marchi U | order, 20/40/80 | f100 - Marchi | R(40, 80) - Marchi | R(80, 100) - Marchi | Ghia station | Ghia - Marchi, indicative |
|---|---|---|---|---|---|---|---|---|
| 0.0625 | -0.041975 | 4.5e-08 | 2.52 | +2.7e-06 | -2.3e-05 | -5.5e-06 | 0.0625 | +0.0001 |
| 0.1250 | -0.077125 | 7.2e-08 | 2.43 | +4.5e-05 | -3.9e-05 | -9.2e-06 | 0.1016 | +0.0001 |
| 0.1875 | -0.109816 | 8.6e-08 | 2.29 | +9.8e-05 | -5.0e-05 | -1.2e-05 | 0.1719 | +0.0002 |
| 0.2500 | -0.141930 | 8.6e-08 | 2.21 | +1.6e-04 | -5.6e-05 | -1.3e-05 | - | - |
| 0.3125 | -0.172712 | 7.3e-08 | 2.14 | +2.4e-04 | -5.8e-05 | -1.3e-05 | - | - |
| 0.3750 | -0.198471 | 5.0e-08 | 2.10 | +3.1e-04 | -5.3e-05 | -1.2e-05 | - | - |
| 0.4375 | -0.212962 | 2.0e-08 | 2.07 | +3.4e-04 | -4.1e-05 | -9.4e-06 | 0.4531 | +0.0031 |
| 0.5000 | -0.209149 | 8.6e-09 | 2.04 | +3.2e-04 | -2.4e-05 | -5.3e-06 | 0.5000 | +0.0033 |
| 0.5625 | -0.182081 | 2.8e-08 | 2.01 | +2.3e-04 | -4.4e-06 | -1.1e-06 | - | - |
| 0.6250 | -0.131256 | 3.5e-08 | 1.97 | +1.1e-04 | +1.2e-05 | +2.6e-06 | 0.6172 | +0.0024 |
| 0.6875 | -0.060246 | 3.7e-08 | 2.23 | -9.9e-06 | +2.4e-05 | +6.2e-06 | - | - |
| 0.7500 | +0.027874 | 4.6e-08 | 2.06 | -1.2e-04 | +4.4e-05 | +1.1e-05 | 0.7344 | -0.0009 |
| 0.8125 | +0.140425 | 7.1e-08 | 2.03 | -2.3e-04 | +5.7e-05 | +1.7e-05 | - | - |
| 0.8750 | +0.310557 | 1.1e-07 | 1.97 | -4.2e-04 | +7.0e-05 | +1.7e-05 | 0.8516 | -0.0050 |
| 0.9375 | +0.597467 | 9.5e-08 | 1.93 | -5.8e-04 | +1.1e-04 | +1.2e-05 | 0.9531 | -0.0039 |

MEASURED, v on y = 1/2:

| x | Marchi v | Marchi U | order, 20/40/80 | f100 - Marchi | R(40, 80) - Marchi | R(80, 100) - Marchi | Ghia station | Ghia - Marchi, indicative |
|---|---|---|---|---|---|---|---|---|
| 0.0625 | +0.094808 | 7.2e-08 | 1.95 | -9.1e-05 | +4.9e-05 | +1.3e-05 | 0.0625 | -0.0025 |
| 0.1250 | +0.149243 | 1.0e-07 | 2.12 | -2.4e-04 | +6.6e-05 | +1.7e-05 | - | - |
| 0.1875 | +0.174343 | 9.7e-08 | 2.10 | -3.0e-04 | +6.6e-05 | +1.6e-05 | - | - |
| 0.2500 | +0.179243 | 7.9e-08 | 2.11 | -3.0e-04 | +5.6e-05 | +1.3e-05 | 0.2344 | -0.0043 |
| 0.3125 | +0.169132 | 5.5e-08 | 2.12 | -2.6e-04 | +4.4e-05 | +9.3e-06 | - | - |
| 0.3750 | +0.145730 | 2.9e-08 | 2.15 | -2.0e-04 | +3.0e-05 | +5.7e-06 | - | - |
| 0.4375 | +0.108776 | 3.7e-09 | 2.19 | -1.2e-04 | +1.5e-05 | +2.3e-06 | - | - |
| 0.5000 | +0.057537 | 2.0e-08 | 3.07 | -8.0e-06 | +3.7e-07 | -8.4e-07 | 0.5000 | -0.0030 |
| 0.5625 | -0.007749 | 4.8e-08 | 2.02 | +1.3e-04 | -1.5e-05 | -3.7e-06 | - | - |
| 0.6250 | -0.084067 | 5.3e-08 | 2.09 | +2.7e-04 | -2.9e-05 | -6.4e-06 | - | - |
| 0.6875 | -0.163010 | 5.0e-08 | 2.10 | +3.8e-04 | -4.9e-05 | -1.0e-05 | - | - |
| 0.7500 | -0.227827 | 5.2e-08 | 2.16 | +3.8e-04 | -7.0e-05 | -1.5e-05 | - | - |
| 0.8125 | -0.253769 | 7.3e-08 | 2.28 | +2.4e-04 | -7.8e-05 | -1.9e-05 | 0.8047 | +0.0082 |
| 0.8750 | -0.218691 | 8.7e-08 | 2.21 | +5.2e-05 | -6.6e-05 | -1.7e-05 | 0.8594 | +0.0092 |
| 0.9375 | -0.123318 | 5.8e-08 | 2.49 | +6.1e-06 | -3.1e-05 | -8.7e-06 | 0.9453 | +0.0054 |

In brief, MEASURED:

- R(80, 100) lies within 1.7e-5 of Marchi in u and 1.9e-5 in v. R(40, 80) lies within 1.1e-4
  and 7.8e-5, and the unextrapolated 100x100 profile within 5.8e-4 and 3.8e-4.
- At 29 of the 30 stations, R(80, 100) is nearer Marchi than R(40, 80) and on the same side,
  by a median factor of 4.3. The exception is v at x = 0.5, where both are below 1e-6.
- Measured in Marchi's own U, R(80, 100) is 40 to 622 times U away in u and 42 to 633 in v,
  with a median of 168 over the 30 stations. Four stations are under 100: u at y = 0.5625 and
  0.625, and v at x = 0.5 and 0.5625.
- About 2.5e-6 of the 1.9e-5 is iteration error left in the 1e-9 fields. Test 22 (C9) formed
  R(80, 100) from the 1e-7, 1e-8 and 1e-9 snapshots. The change shrinks tenfold per decade of
  tolerance, from 2.1e-4 to 2.2e-5 in u and from 2.2e-4 to 2.3e-5 in v, which puts what
  remains beyond 1e-9 at 2.4e-6 in u and 2.5e-6 in v. Interpolation (at most 9.1e-6, above) and iteration are the two sources this
  comparison can put a number on. What they leave is truncation error that one order-2
  extrapolation does not remove (INFERRED).

**Ghia against Marchi without our solution.** This is a one-off, not committed:
`results/reference/ghia_vs_marchi_interpolated.txt`. Marchi's table, walls appended, is
interpolated to Ghia's own stations by the cubic and the quintic, and nothing from the solver
enters. At the four floor stations of section 9, MEASURED:

| Ghia station | Ghia - Marchi, cubic | quintic | section 9, order-2 value - Ghia |
|---|---|---|---|
| u, y = 0.8516 | -0.0049 | -0.0044 | +0.0051 |
| v, x = 0.9063 | +0.0073 | +0.0082 | -0.0080 |
| v, x = 0.8594 | +0.0087 | +0.0091 | -0.0093 |
| v, x = 0.8047 | +0.0081 | +0.0082 | -0.0083 |

Near the lid (u at y >= 0.95) and at the right wall (v at x >= 0.95), Marchi's sixteenths are
too coarse for this: the cubic and the quintic differ there by up to 5.1e-3 and 1.5e-3. Away
from the jet, Ghia differs from Marchi by up to about 0.004. Examples are v at x = 0.16 to
0.23 and u at y = 0.45 to 0.5. At the shared stations, where nothing is interpolated at all,
the differences are +0.0033 (u, y = 0.5), -0.0030 (v, x = 0.5), -0.0025 (v, x = 0.0625) and
+0.0001 (u, y = 0.0625).

## 5. Reading the answer

The readings were written into the prompt before any number was taken:

1. Ours within a few times Marchi's error estimate of Marchi, and Ghia off by about the 0.005
   to 0.009 gap at the jet points: the gap is Ghia's.
2. Ours off Marchi by about the same gap, and Ghia close to Marchi: the gap is ours.
3. Anything else, reported as found.

**The first reading matches, with its first clause measured on a different scale.**

- "Within a few times Marchi's error estimate" cannot be met, and is not met as written.
  Marchi's U at these points is at most 1.1e-7. Our extrapolation's own uncertainty, the
  change from R(40, 80) to R(80, 100), is up to 9.6e-5, nearly a thousand times larger. Our
  R(80, 100) differs from Marchi by at most 1.7e-5 in u and 1.9e-5 in v. That is inside our
  uncertainty, but 40 to 633 times Marchi's. The comparison that decides between the
  readings is with the gap: those largest differences are about 1/290 of the 0.005 gap in u
  and 1/480 of the 0.009 gap in v.
- "Ghia off by about the gap at the jet points" is met with our solution used to carry Ghia's
  values (section 4, first tables), and also without it (the last table).
- The second reading is excluded at every jet point.

**INFERRED.** The distance between the converged staggered solution and Ghia's table at the
jet is Ghia's error. Two schemes that share no discretization converge to the same values
within 2e-5 at 30 points. One is co-located central differences on grids up to 1024x1024 with
six Richardson extrapolations. The other is staggered QUICK on 80x80 and 100x100 with one.
Ghia's table differs from both by up to 0.009. Ghia's error is not confined to the jet: it
is 0.002 to 0.004 at the centerline crossing and in the upward flow on the left.

**What follows, for a decision not taken here, INFERRED.** Scored against Ghia, a correctly
converging scheme meets a floor of about 0.005 in u and 0.009 in v at the jet.

- Criterion 3a asks the error against Ghia to fall monotonically. Once a scheme's own error
  is below Ghia's, that error can rise under refinement. That is what section 9.1 recorded for
  the staggered series.
- VAL-002's 2% threshold is above the floor, so VAL-002 itself is not affected.

Which reference criterion 3a uses is Alex's decision. Nothing in this change alters a
reference, a metric or a criterion.

## 6. Stop conditions

- **No text layer:** not reached.
- **Two extractions disagree anywhere:** never in a digit. The routes did disagree on which
  row each Table 7 label belongs to. That was settled by the glyph positions (route C), a
  machine reading, and by the two facts in section 2. This is the one place the rule was
  read rather than followed to the letter. The pairing decides only the stored U values,
  which this report uses as a scale, and not the solution values.
- **A Re = 100 value equal to another column's:** not reached in any stored row.
- **The reference fails its own checks beyond the quadrature error:** not reached.
- **Ours and Marchi more than about 0.005 apart away from the jet:** not reached. The largest
  difference anywhere is 1.9e-5 for R(80, 100), and 5.8e-4 for the plain 100x100 profile.

## 7. How each number was taken

- **Extraction, digit and column checks:** `results/reference/extract_marchi.py` (scratch,
  gitignored; needs pypdf). It writes `marchi_route_*.json` (one per route),
  `marchi_extraction_check.json`, and the two text dumps, all in `results/reference/`, beside
  the PDF.
- **Stored table against the extraction:** one-off, `results/reference/stored_equals_extraction.txt`.
- **Quadrature estimate, table checks and controls:** one-off,
  `results/reference/quadrature_estimate.txt` and `table_checks.txt`. The checks and two of
  the controls are also `tests/test_validation.py`, `TestMarchiTable`.
- **Comparison tables and summary:** `python scripts/self_convergence.py --marchi`, then
  `results/self_convergence/marchi_comparison.json`.
- **Ghia against Marchi without a solver field:** one-off,
  `results/reference/ghia_vs_marchi_interpolated.txt`.
- **Field sources and face recovery residuals:** one-off, `results/reference/field_checks.txt`.
- **Iteration error in R(80, 100):** test 22, C9 (`docs/prompts/test-22.md`, not tracked), from
  the snapshots in the saved field files; its output is `results/tester22/iteration_error.txt`.
- **The comparison code's own guard:** `tests/test_self_convergence.py`, the three tests on
  `tight_field` and `marchi_comparison`, which run on synthetic fields of known order and limit.
