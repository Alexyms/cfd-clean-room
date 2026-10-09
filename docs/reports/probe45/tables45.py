"""Tables of prompt 45's VAL-016 measurement, from the records couette45.py writes.

Usage:
    python tables45.py

For each Couette run: the stop, the outer count, the iteration from which each
of the five conditions holds, u_tau from the mid-gap stress, Re_tau, y+ of the
first node, the core k (0.2 H to 0.8 H) over u_tau^2 / sqrt(C_mu), and the
profile differences against the 1D reference on the same grid (REFINE 1, the
2D stencil's own x-invariant limit) and refined (REFINE 21, the model's
answer for the same wall cells): max |u - u_ref| / U_W over the cell centres
and max |k / u_tau^2 - k_ref / u_tau_ref^2|, each normalised by its own u_tau.
For each channel run: Cf from the pressure gradient and from the wall
function's shear against Dean's correlation, Cf = 0.073 Re_m^(-1/4), with
Cf = tau_w / (rho U_m^2 / 2) and Re_m = U_m 2h / nu, 2h the full gap.
"""

import json
import math
from pathlib import Path

import couette45 as c45

OUT = c45.OUT


def _load(name: str) -> dict | None:
    path = OUT / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def couette() -> list[dict]:
    rows = []
    for variant in ("standard", "rng"):
        for ny in (12, 24):
            run = _load(f"couette_{variant}_{ny}")
            if run is None:
                continue
            c_mu = c45.VARIANTS[variant]["c_mu"]
            ut = run["u_tau_mid"]
            k_eq = ut**2 / math.sqrt(c_mu)
            core = [
                k / k_eq
                for y, k in zip(run["yc"], run["k"], strict=True)
                if 0.2 * c45.GAP <= y <= 0.8 * c45.GAP
            ]
            row = {
                "variant": variant,
                "ny": ny,
                "stop": run["stop"],
                "outer": run["outer"],
                "from_when": run["from_when"],
                "u_tau": ut,
                "re_tau": ut * c45.GAP / 2.0 / c45.NU,
                "y_plus_first": ut * c45.GAP / ny / 2.0 / c45.NU,
                "core_k_ratio": [min(core), max(core)],
                "profile_change_0.9L": run["profile_change_against_station"]["0.9"],
                "k_min": run["k_min"],
                "eps_min": run["eps_min"],
            }
            for refine in (1, 21):
                ref = _load(f"ref_couette_{variant}_{ny}_r{refine}")
                if ref is None:
                    continue
                ut_r = ref["u_tau_mid"]
                du = max(abs(a - b) for a, b in zip(run["u"], ref["u"], strict=True))
                dk = max(
                    abs(a / ut**2 - b / ut_r**2)
                    for a, b in zip(run["k"], ref["k"], strict=True)
                )
                row[f"ref{refine}"] = {
                    "du_over_U_W": du / c45.U_W,
                    "dk_over_u_tau2": dk,
                    "u_tau_ratio": ut / ut_r,
                }
            rows.append(row)
    return rows


def channel() -> list[dict]:
    rows = []
    re_m = c45.U_M * c45.GAP / c45.NU
    dean = 0.073 * re_m**-0.25
    dynamic = 0.5 * c45.RHO * c45.U_M**2
    for variant in ("standard", "rng"):
        for ny in (12, 24):
            run = _load(f"channel_{variant}_{ny}")
            if run is None:
                continue
            from_pressure = -run["dpdx"] * c45.GAP / 2.0 / dynamic
            from_wall = 0.5 * sum(run["tau_wall"]) / dynamic
            rows.append(
                {
                    "variant": variant,
                    "ny": ny,
                    "stop": run["stop"],
                    "outer": run["outer"],
                    "from_when": run["from_when"],
                    "re_m": re_m,
                    "cf_dean": dean,
                    "cf_pressure": from_pressure,
                    "cf_wall": from_wall,
                    "pressure_over_dean": from_pressure / dean,
                    "wall_over_dean": from_wall / dean,
                    "profile_change_0.9L": run["profile_change_against_station"]["0.9"],
                }
            )
    return rows


def main() -> None:
    tables = {"couette": couette(), "channel": channel()}
    text = json.dumps(tables, indent=1)
    (Path(OUT) / "tables45.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
