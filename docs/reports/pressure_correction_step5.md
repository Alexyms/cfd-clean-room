# Pressure correction on the staggered grid: compatibility and Jacobi cost

**Date:** 2026-09-22
**Context:** ECR-001 step 5, `src/pressure.py`, before integration (step 6)
**Instrument:** a scratch script that runs five momentum predictor sweeps from rest on
each seeded case, then one pressure correction; numbers reproduced by
`tests/test_pressure.py` where stated

## 1. The right-hand side sums to zero on the closed domain

Acceptance criterion 6 measured directly. The mass imbalance of u* summed over the
cavity, rho = U = L = 1, five predictor sweeps from rest. The collocated figures are
the net wall leak from `docs/reports/pressure_solver_probe.md`, table E, at
convergence.

| Grid | Staggered `sum b` | Total absolute face flux | Collocated `sum b` (probe, table E) |
|---|---|---|---|
| 20x20 | -8.7e-19 | 6.1e-01 | 2.90e-02 |
| 40x40 | 2.2e-18 | 6.7e-01 | 9.23e-03 |
| 80x80 | -3.3e-19 | 7.0e-01 | 2.35e-03 |

The staggered sum is the rounding residue of a telescoping sum whose terms total
about 0.6 in magnitude; machine epsilon on that scale is 1.3e-16. The test bound is
eight epsilon times the total absolute face flux and the measured values sit two
orders of magnitude inside it. There is no compatibility correction anywhere in the
assembly: the sum vanishes because every wall face holds zero exactly.

## 2. Jacobi cost, REQ-S08 as written

Same fields, the case file's `pressure_tol` of 1e-8, first the case file's sweep cap
and then no cap (bound 200,000).

| Case | Cap | Sweeps at cap | max abs imbalance after (cap) | Sweeps to tol, uncapped | Seconds | max abs imbalance after (uncapped) |
|---|---|---|---|---|---|---|
| cavity 20x20 | 500 | 500 | 1.19e-04 | never (200,000) | 6.8 | 9.84e-05 |
| cavity 40x40 | 500 | 500 | 6.57e-05 | never (200,000) | 8.8 | 1.11e-05 |
| cavity 80x80 | 500 | 500 | 3.33e-05 | never (200,000) | 16.8 | 1.33e-06 |
| Poiseuille 80x40 | 2000 | 2000 | 3.19e-05 | 126,277 | 7.8 | 1.56e-10 |

Before correction the maximum imbalance is 2.8e-2, 1.3e-2, 6.3e-3 and 8.8e-4
respectively.

On the open channel Jacobi converges and is merely expensive: 126,277 sweeps for one
correction at the tolerance the case file asks for, against the 18,000 the probe
measured for the collocated 80x80 cavity system. That is the cost the prompt for this
step expected to see, and it is real.

## 3. On the closed domain undamped Jacobi does not converge at all

The cavity rows are not slow convergence. The residual after correction is, to
rounding, a pure checkerboard: `r / a_P = (-1)^(i+j) c` with a constant `c`, whose
sign flips with the parity of the sweep count (measured spread 1.6e-16 around the
constant at 4,000 sweeps on an 8x6 cavity; `tests/test_pressure.py`,
`test_closed_domain_plain_jacobi_stalls_on_the_checkerboard_mode`).

The mechanism is exact. In a closed domain every row has `a_P = sum(a_nb)`, and the
grid is bipartite, so the sign vector `s = (-1)^(i+j)` satisfies `sum(a_nb s_nb) =
-a_P s_P` in every cell, boundary cells included. It is therefore an eigenvector of
the Jacobi iteration matrix with eigenvalue exactly -1, alongside the constant vector
at +1. The component of the initial error along `s` never decays; it changes sign
each sweep. The collocated solver has the same eigenvalue but never met it, because
its right-hand side was incompatible and the iteration drifted instead. A pressure
outlet breaks the symmetry (its row has `a_P > sum(a_nb)`), which is why the channel
converges.

The smallest case shows the consequence: two cells in a row, one interior face
carrying an imbalance q. After an odd number of sweeps the correction overshoots by a
factor of two; after an even number it is zero. Which one the solver returns depends
on the cap.

This means a step 6 solver built on undamped Jacobi cannot satisfy acceptance
criterion 6 (per-cell continuity below 1e-10) on VAL-002, whatever the cap.

## 4. What was not done, and why

No damping factor, pin-as-Dirichlet or other change was made to the iteration. The
prompt for this step fixed REQ-S08 as written so that the layout change and any
solver change stay attributable separately, and this report is the measurement it
asked for. The observation for the amendment discussion: weighted Jacobi,
`p_new = (1 - w) p + w D^-1 (N p - b)` with w = 2/3, maps the -1 eigenvalue to -1/3
and leaves the update data-parallel per cell, which is the architectural content of
REQ-S08. The eigenvalue argument, not the sweep count, is the reason an amendment is
needed before step 6 can pass VAL-002.
