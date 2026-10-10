"""The second-cell strain overshoot of prompt 45 (ECR-002 step 6), measured and decomposed.

Usage:
    python mech45.py

Prints the table the ADR-012 note records and writes it to
results/builder45/couette/mech45.json (about a minute). It replaces the
session scratch scripts mech.py and mech2.py, whose reference had 21
sub-cells per 2D cell; this one refines it to 161 and shows the sequence.

Two profiles across the Couette gap, each from tests/couette_reference.py's 1D solve
(nothing from src/): the 2D stencil's own x-invariant limit (REFINE 1), which
the developed 2D solve equals to 1e-5 (couette45.py's records), and the
refined solve for the same wall cells, the model's answer. h is the 2D cell
height; the second cell's centre is at y = 1.5 h; u_tau is the refined
solve's, from its mid-gap stress.

- The stencil's du/dy there, S_st = (q_s + q_n) / 2, the mean of the
  difference quotients q_s = (u_1 - u_0) / h and q_n = (u_2 - u_1) / h of the
  REFINE 1 profile at the cell's two faces (y = h and 2 h): in 2D, du/dy at
  the four corners averaged to the centre, with no x variation.
- The true du/dy there, g(1.5 h), from the refined profile (its difference
  quotients between neighbouring reference cells, interpolated linearly).
- R = S_st / g(1.5 h), the overshoot in du/dy; R^2 the overshoot in S^2.

R is the product of three factors, each defined so the product is exact:

- F1 = (1 / h + 1 / (2 h)) / 2 / (1 / (1.5 h)) = 1.125. Averaging the exact
  face gradients of a log law, u = (u_tau / kappa) ln(E y u_tau / nu): the
  orchestrator's estimate, 12.5% in du/dy and 27% in S^2.
- F2 = [(ln 3 + ln(5 / 3)) / 2 / (1 / 1.5)] / F1 = 1.2071 / 1.125 = 1.0730.
  The stencil's difference quotients between the cell centres in place of
  the exact face gradients, on the same log law: together with F1, 20.7% in
  du/dy and 46% in S^2 on an exact log law.
- F3 = R / (F1 F2), measured: the solved profile against the log law. It is
  itself the product of S_st / S_log, the stencil on the REFINE 1 profile
  over the stencil on the log law with the refined solve's u_tau (the 2D
  profile's step from the wall cell to the second cell is larger than the
  log law's), and g_log(1.5 h) / g(1.5 h), the log law's gradient at the
  centre over the true one.
"""

import json
import math

import couette45 as c45
import numpy as np

from tests import couette_reference as ref

REFINE = 161
F1 = 0.5 * (1.0 + 0.5) * 1.5
LOG_QUOTIENTS = 0.5 * (math.log(3.0) + math.log(5.0 / 3.0)) * 1.5
F2 = LOG_QUOTIENTS / F1


def measure(variant: str, ny: int, refine: int = REFINE) -> dict:
    """R, its three factors and the second cell's k ratio for one variant and grid."""
    h = ref.GAP / ny
    same = ref.reference(variant, ny, 1)
    refined = ref.reference(variant, ny, refine, full=True)
    fine = refined["profile"]
    yc, uf = np.array(fine["yc"]), np.array(fine["u"])
    g_centre = float(
        np.interp(1.5 * h, 0.5 * (yc[1:] + yc[:-1]), np.diff(uf) / np.diff(yc))
    )
    u = same["u"]
    q_s, q_n = (u[1] - u[0]) / h, (u[2] - u[1]) / h
    s_st = 0.5 * (q_s + q_n)
    r = s_st / g_centre
    log_scale = refined["u_tau_mid"] / ref.KAPPA
    s_log = 0.5 * (math.log(3.0) + math.log(5.0 / 3.0)) * log_scale / h
    g_log = log_scale / (1.5 * h)
    f3 = r / (F1 * F2)
    assert math.isclose(f3, (s_st / s_log) * (g_log / g_centre), rel_tol=1e-12)
    k_fine = float(np.interp(1.5 * h, yc, np.array(fine["k"])))
    return {
        "variant": variant,
        "ny": ny,
        "refine": refine,
        "R_dudy": r,
        "R_S2": r * r,
        "F1": F1,
        "F2": F2,
        "F3": f3,
        "product": F1 * F2 * f3,
        "stencil_over_log_law_stencil": s_st / s_log,
        "log_law_over_true_gradient_at_centre": g_log / g_centre,
        "q_s_over_log_law": q_s / (math.log(3.0) * log_scale / h),
        "q_n_over_log_law": q_n / (math.log(5.0 / 3.0) * log_scale / h),
        "k_second_cell_ratio": same["k"][1] / k_fine,
    }


def main() -> None:
    sequence = [
        measure(variant, 12, refine)
        for variant in ("standard", "rng")
        for refine in (21, 41, 81)
    ]
    rows = [measure(v, ny) for v in ("standard", "rng") for ny in (12, 24)]
    record = {
        "exact_log_law": {
            "F1": F1,
            "F1_F2": LOG_QUOTIENTS,
            "F1_F2_S2": LOG_QUOTIENTS**2,
        },
        "refinement_sequence_12_rows": sequence,
        "rows": rows,
    }
    c45.OUT.mkdir(parents=True, exist_ok=True)
    (c45.OUT / "mech45.json").write_text(json.dumps(record, indent=1))
    print(
        f"exact log law: F1 {F1:.4f}, F1 F2 {LOG_QUOTIENTS:.4f} (S2 {LOG_QUOTIENTS**2:.4f})"
    )
    for row in sequence + rows:
        print(
            f"{row['variant']:8s} {row['ny']:2d} rows, refine {row['refine']:3d}: "
            f"du/dy {row['R_dudy']:.4f} S2 {row['R_S2']:.4f} = {row['F1']:.4f} x "
            f"{row['F2']:.4f} x {row['F3']:.4f} (F3 = "
            f"{row['stencil_over_log_law_stencil']:.4f} x "
            f"{row['log_law_over_true_gradient_at_centre']:.4f}; q_s/log "
            f"{row['q_s_over_log_law']:.4f}, q_n/log {row['q_n_over_log_law']:.4f}); "
            f"k second cell {row['k_second_cell_ratio']:.4f}"
        )


if __name__ == "__main__":
    main()
