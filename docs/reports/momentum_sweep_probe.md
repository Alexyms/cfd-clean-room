# Momentum Sweep Probe: Is the Momentum Inner Solve the Rate Limiter?

*Erratum 2026-09-22: the v-error column was measured against the Ghia v reference in use until 2026-09-22 was not Ghia's Table II and failed mass conservation along the centerline; see the ECR-001 erratum, section 12, and reference `ghia_1982_re100_r2`. The tables below are unchanged. The column compares one solver's fields across sweep counts, so what it shows here, that k does not change the converged field, does not depend on the reference.*

**Date:** 2026-09-19
**Branch:** `scratch/momentum-sweeps` (experiment only, not for merge)
**Baseline:** `feature/validation-consolidation` at `dcddb39`
**Records:** `benchmarks/results.jsonl` on this branch, twelve rows at commit `cf8b3b1`

## Verdict

H2 is contradicted. Extra momentum sweeps cut the outer SIMPLE count by a
factor of 1.8 and then stop helping. P6 predicted roughly 160 outer
iterations at k = 10; the measurement is 874, and k = 30 is also 874. The
momentum inner solve accounts for less than half of the outer iterations.
The remaining 874 live in the pressure-velocity coupling or in the
iterate-change convergence criterion, which the pressure probe already
flagged.

The converged solution does not change with k. That is the good version of
the result: the extra sweeps change the path, not the destination.

## Hypothesis and prediction, as stated before measuring

**H2.** The momentum inner solve sets the outer iteration count.

**P6.** Raising `momentum_sweeps` to k reduces the outer count roughly as
1/k until the coupling becomes the limit. At 40x40, k = 10 should give
about 160 outer iterations rather than 1571. If the count barely moves, H2
is dead and the limit lives in the coupling or the convergence criterion.

**Verdict on P6:** contradicted. The count moves, but by 1.8x rather than
10x, and it saturates by k = 10.

## Method

`momentum_sweeps` was added to the solver configuration (optional, default
1, so every existing configuration is unchanged). Step 4 of `solve_steady`
repeats the u and v Jacobi sweeps k times with the coefficients and the
under-relaxation source held fixed; k = 1 reproduces the original
algorithm exactly (asserted by a unit test on this branch). Case files
`configs/validation_cavity_k{3,10,30}.yaml` are the VAL-002 file plus the
one key. The benchmark harness ran each case three times; the field
viewer then solved each case once more and saved the fields for
comparison. Runs shared the machine with the 80x80 cavity run on another
checkout (concurrency 2 in every record).

## Measurements

40x40 lid-driven cavity, validation settings (alpha_velocity 0.5, cap
500, tolerance 1e-8), three repeats each.

| k | Outer iterations | Wall s (min / med / max) | Cell updates | Momentum s | Pressure s | v-error |
|---|---|---|---|---|---|---|
| 1 | 1571 | 101.1 / 102.2 / 106.2 | 1.139e9 | 1.0 | 98.7 | 0.18931 |
| 3 | 977 | 64.4 / 65.2 / 67.1 | 7.139e8 | 1.2 | 62.3 | 0.18932 |
| 10 | 874 | 57.8 / 58.6 / 59.2 | 6.563e8 | 2.7 | 54.3 | 0.18932 |
| 30 | 874 | 52.5 / 53.4 / 53.9 | 7.068e8 | 6.2 | 45.6 | 0.18932 |

Outer counts and errors were identical across the three repeats of every
k. Stage times are from the first repeat.

Converged fields against k = 1, max-norm over the whole field (reference
magnitudes: `max |u| = 0.922`, `max |v| = 0.402`, `max |p| = 0.572`):

| k | u | v | p |
|---|---|---|---|
| 3 | 1.00e-4 | 9.39e-5 | 3.77e-5 |
| 10 | 1.16e-4 | 1.09e-4 | 4.38e-5 |
| 30 | 1.17e-4 | 1.10e-4 | 4.41e-5 |

These are at the level the outer tolerance (1e-6 on the scaled velocity
change) leaves undetermined, and the Ghia centerline error agrees to four
digits. The solution is unchanged.

## Reading

The 1571 to 977 to 874 sequence has the shape of two rate limiters in
series. The momentum sweep is one, and it is exhausted by k = 10. Whatever
is left is unaffected by how well the momentum system is solved, and the
pressure probe established that the pressure correction is a no-op at
convergence for this discretization. What remains is the SIMPLE coupling
itself under alpha_velocity = 0.5 and alpha_pressure = 0.3, and the
convergence test, which stops on the size of the last velocity change
rather than on any equation residual. Both are properties of the outer
algorithm, not of either inner solve.

A parenthetical consistency check on the prompt's arithmetic: the
observed alpha sensitivity (1571 to 1178 for 0.5 to 0.7) is consistent
with an under-relaxed outer iteration whether or not the momentum sweep is
the limit, so it did not discriminate, and P6 was the right test.

Wall time is worth a note even though it is not what the experiment was
about. k = 30 was the fastest configuration because 874 iterations of 500
pressure sweeps cost less than 1571 of them, and 60 momentum sweeps per
iteration are cheap next to 500 pressure sweeps. That is a fact about how
wasteful the pressure cap is, not about momentum.

## Recommendation for the rebuild

Do not carry a `momentum_sweeps` knob into the staggered solver on the
strength of this. It buys 1.8x on the collocated solver and nothing after
k = 10, and the collocated solver is being deleted. The result to carry is
the negative one: the outer count is set by the coupling and the
convergence criterion, so those are what the rebuild should instrument
first. A residual-based outer criterion (mass imbalance and momentum
residual, not iterate change) is the cheapest next probe and would say
whether the 874 is real convergence or the criterion stopping early.

## What this probe did not establish

- Only 40x40 was run. Whether the saturation point moves with N was not
  measured.
- The under-relaxation factors were held at the validation settings.
- Runs were concurrent with another solve; the timing column is
  indicative, the iteration and error columns are exact.
