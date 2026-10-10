"""Tables of prompt 45's VAL-016 demonstration, from the records couette45.py writes.

Usage:
    python tables45.py

VAL-016 as split by Alex on 2026-10-09 (ADR-012 G's note of that date), for
every Couette run on record:

- (a) the developed 2D profile at 0.8 of the length against the 1D solve on
  the same grid (REFINE 1): max |u - u_1D| / U_W and max |k / u_tau^2 -
  k_1D / u_tau_1D^2| over the cell centres, each k normalised by its own
  u_tau, against 1e-4;
- (b) the refined 1D solve (REFINE 41, the same wall cells): its core k
  (0.2 H to 0.8 H) over u_tau^2 / sqrt(C_mu), against 1%;
- (c) reported: the 2D core k over u_tau^2 / sqrt(C_mu) and the first node's
  y+, with u_tau from the 2D mid-gap stress.

Each row also gives the stop, the outer count, the outer iteration from which
each of the five conditions holds, the wall time and the profile's change
from 0.8 to 0.9 of the length. For each channel run: Cf from the pressure
gradient and from the wall function's shear against Dean's correlation,
Cf = 0.073 Re_m^(-1/4), with Cf = tau_w / (rho U_m^2 / 2) and Re_m = U_m 2h /
nu, 2h the full gap (Dean 1978, J. Fluids Eng. 100:215-223, from its
abstract).
"""

import json
import math

import couette45 as c45
import numpy as np

from tests import couette_reference as ref

OUT = c45.OUT
REFINE = 41
TOL_A, TOL_B = 1.0e-4, 0.01


def _load(name: str) -> dict | None:
    path = OUT / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _core(yc: list, k: list, u_tau: float, c_mu: float) -> list[float]:
    y, kk = np.array(yc), np.array(k)
    core = (y >= 0.2 * ref.GAP) & (y <= 0.8 * ref.GAP)
    ratio = kk[core] / (u_tau**2 / math.sqrt(c_mu))
    return [float(ratio.min()), float(ratio.max())]


def couette() -> list[dict]:
    rows = []
    for variant in ("standard", "rng"):
        c_mu = ref.VARIANTS[variant]["c_mu"]
        for ny in (12, 24, 48):
            run = _load(f"couette_{variant}_{ny}")
            if run is None:
                continue
            ut = run["u_tau_mid"]
            same = ref.reference(variant, ny, 1)
            refined = ref.reference(variant, ny, REFINE)
            du = np.abs(np.array(run["u"]) - np.array(same["u"])).max() / ref.U_W
            dk = np.abs(
                np.array(run["k"]) / ut**2
                - np.array(same["k"]) / same["u_tau_mid"] ** 2
            ).max()
            model = _core(refined["yc"], refined["k"], refined["u_tau_mid"], c_mu)
            rows.append(
                {
                    "variant": variant,
                    "ny": ny,
                    "dx": run["dx"],
                    "length": run["length"],
                    "stop": run["stop"],
                    "outer": run["outer"],
                    "seconds": run["seconds"],
                    "from_when": run["from_when"],
                    "profile_change_0.8_to_0.9": run["profile_change_against_station"][
                        "0.9"
                    ],
                    "a_du_over_U_W": float(du),
                    "a_dk_over_u_tau2": float(dk),
                    "a_u_tau_ratio": ut / same["u_tau_mid"],
                    "a_pass": bool(du < TOL_A and dk < TOL_A),
                    "b_core_k_ratio": model,
                    "b_pass": bool(max(abs(v - 1.0) for v in model) < TOL_B),
                    "c_core_k_ratio_2d": _core(run["yc"], run["k"], ut, c_mu),
                    "c_y_plus_first": ut * ref.GAP / ny / 2.0 / ref.NU,
                    "u_tau": ut,
                    "re_tau": ut * ref.GAP / 2.0 / ref.NU,
                    "k_min": run["k_min"],
                    "eps_min": run["eps_min"],
                }
            )
    return rows


def channel() -> list[dict]:
    rows = []
    re_m = ref.U_M * ref.GAP / ref.NU
    dean = 0.073 * re_m**-0.25
    dynamic = 0.5 * ref.RHO * ref.U_M**2
    for variant in ("standard", "rng"):
        for ny in (12, 24, 48):
            run = _load(f"channel_{variant}_{ny}")
            if run is None:
                continue
            from_pressure = -run["dpdx"] * ref.GAP / 2.0 / dynamic
            from_wall = 0.5 * sum(run["tau_wall"]) / dynamic
            rows.append(
                {
                    "variant": variant,
                    "ny": ny,
                    "dx": run["dx"],
                    "stop": run["stop"],
                    "outer": run["outer"],
                    "seconds": run["seconds"],
                    "from_when": run["from_when"],
                    "re_m": re_m,
                    "cf_dean": dean,
                    "cf_pressure": from_pressure,
                    "cf_wall": from_wall,
                    "pressure_over_dean": from_pressure / dean,
                    "profile_change_0.8_to_0.9": run["profile_change_against_station"][
                        "0.9"
                    ],
                }
            )
    return rows


def main() -> None:
    tables = {"couette": couette(), "channel": channel()}
    (OUT / "tables45.json").write_text(json.dumps(tables, indent=1))
    for r in tables["couette"]:
        print(
            f"couette {r['variant']:8s} {r['ny']:2d} rows dx {r['dx']}: {r['stop']} "
            f"{r['outer']} its {r['seconds']:.0f} s, from {r['from_when']}; "
            f"(a) du {r['a_du_over_U_W']:.1e} dk {r['a_dk_over_u_tau2']:.1e} "
            f"{'pass' if r['a_pass'] else 'FAIL'}; (b) {r['b_core_k_ratio'][0]:.4f}.."
            f"{r['b_core_k_ratio'][1]:.4f} {'pass' if r['b_pass'] else 'FAIL'}; "
            f"(c) 2D core {r['c_core_k_ratio_2d'][0]:.4f}..{r['c_core_k_ratio_2d'][1]:.4f}"
            f" y+ {r['c_y_plus_first']:.0f}, Re_tau {r['re_tau']:.0f}"
        )
    for r in tables["channel"]:
        print(
            f"channel {r['variant']:8s} {r['ny']:2d} rows dx {r['dx']}: {r['stop']} "
            f"{r['outer']} its {r['seconds']:.0f} s, from {r['from_when']}; Cf "
            f"{r['cf_pressure']:.4e} (wall {r['cf_wall']:.4e}), Dean "
            f"{r['cf_dean']:.4e}, ratio {r['pressure_over_dean']:.4f}"
        )


if __name__ == "__main__":
    main()
