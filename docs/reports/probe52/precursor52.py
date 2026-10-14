"""Prompt 52, item 2: read the precursor channel's boundary layer along x.

Usage:
    python precursor52.py [NAME] [--every H]

Reads results/builder52/NAME.json and NAME.npz (backstep52.py precursor;
NAME defaults to "precursor") and prints, every --every step heights along
the channel (default 10), the centreline speed over the inlet speed, the
bottom wall's delta_99 / H (the height above the wall at which u first
reaches 99% of the local centreline speed) and Re_theta (the momentum
thickness times the local centreline speed over nu), read at each u-face
column from the cell rows above the block as validation.backstep.
layer_thicknesses reads them. Then the first x at which each quantity
reaches the data's value (1.5 H; 5,000), the upstream length the prompt's
rule gives (that distance plus 4 H, rounded up to a whole H) for each, and
the inlet speed that would make the centreline U_ref at x/H = -4 of the
chosen length. The numbers are kept in results/builder52/precursor_NAME.json.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from src.config import SimConfig  # noqa: E402
from src.mesh import Mesh  # noqa: E402
from validation import backstep  # noqa: E402

OUT = ROOT / "results" / "builder52"


def layer_along_x(
    mesh: Mesh, config: SimConfig, u_faces: np.ndarray
) -> dict[str, np.ndarray]:
    """The centreline speed, delta_99 and Re_theta at every interior u-face column."""
    h = backstep.STEP_HEIGHT
    j0 = backstep.step_row(mesh, config)
    top = h + 0.5 * backstep.UPSTREAM_HEIGHTS * h
    rows = np.nonzero(mesh.yc[j0:] <= top)[0] + j0
    y = np.asarray(mesh.yc[rows]) - h
    nu = config.mu / config.rho
    cols = np.arange(1, mesh.x.size - 1)
    x_h = mesh.x[cols] / h
    u_centre = backstep.sample_u(mesh, u_faces, mesh.x[cols], np.full(cols.size, top))
    delta = np.full(cols.size, np.nan)
    re_theta = np.full(cols.size, np.nan)
    for n, i in enumerate(cols):
        try:
            d, theta = backstep.layer_thicknesses(y, u_faces[rows, i], u_centre[n])
        except ValueError:
            continue
        delta[n] = d / h
        re_theta[n] = u_centre[n] * theta / nu
    return {"x_h": x_h, "u_centre": u_centre, "delta_99_h": delta, "re_theta": re_theta}


def first_reaching(x: np.ndarray, q: np.ndarray, target: float) -> float | None:
    """The first x at which q reaches the target, linear between columns."""
    ok = np.isfinite(q)
    x, q = x[ok], q[ok]
    above = np.nonzero(q >= target)[0]
    if above.size == 0:
        return None
    j = int(above[0])
    if j == 0:
        return float(x[0])
    return float(x[j - 1] + (x[j] - x[j - 1]) * (target - q[j - 1]) / (q[j] - q[j - 1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="?", default="precursor")
    parser.add_argument("--every", type=float, default=10.0)
    args = parser.parse_args()
    rec = json.loads((OUT / f"{args.name}.json").read_text())
    arrays = np.load(OUT / f"{args.name}.npz")
    config = SimConfig.from_dict(rec["raw"])
    mesh = Mesh(config)
    u_in = rec["raw"]["boundaries"]["inlet"]["velocity"]
    along = layer_along_x(mesh, config, arrays["u_faces"])
    x_h = along["x_h"]
    print(
        f"{args.name}: stop {rec['stop']} after {rec['outer']} outer iterations, "
        f"{rec['seconds']:.0f} s; inlet {u_in} m/s, I {rec['turbulence_intensity']}, "
        f"l_e {rec['dissipation_length']} m; cap hits {rec['cap_hits']}"
    )
    print()
    print("| x/H | U_centre / U_in | delta_99 / H | Re_theta |")
    print("|---|---|---|---|")
    rows = []
    for target in np.arange(args.every, x_h[-1] + 1e-9, args.every):
        n = int(np.argmin(np.abs(x_h - target)))
        rows.append(
            {
                "x_h": float(x_h[n]),
                "u_centre_over_u_in": float(along["u_centre"][n] / u_in),
                "delta_99_h": float(along["delta_99_h"][n]),
                "re_theta": float(along["re_theta"][n]),
            }
        )
        print(
            f"| {x_h[n]:.1f} | {along['u_centre'][n] / u_in:.4f} | "
            f"{along['delta_99_h'][n]:.3f} | {along['re_theta'][n]:,.0f} |"
        )
    x_delta = first_reaching(x_h, along["delta_99_h"], backstep.DELTA_99_MEASURED)
    x_theta = first_reaching(x_h, along["re_theta"], backstep.RE_THETA_MEASURED)
    out = {
        "name": args.name,
        "stop": rec["stop"],
        "outer": rec["outer"],
        "seconds": rec["seconds"],
        "u_in": u_in,
        "rows": rows,
        "x_delta_99_reaches_1_5h": x_delta,
        "x_re_theta_reaches_5000": x_theta,
    }
    print()
    for label, x_cross in (
        ("delta_99 = 1.5 H", x_delta),
        ("Re_theta = 5,000", x_theta),
    ):
        if x_cross is None:
            print(f"{label}: not reached within {x_h[-1]:.0f} H")
            continue
        l_up = math.ceil(x_cross + 4.0 - 1e-9)
        n = int(np.argmin(np.abs(x_h - (l_up - 4.0))))
        ratio = float(along["u_centre"][n] / u_in)
        inlet = backstep.U_REF / ratio
        key = "delta" if label.startswith("delta") else "theta"
        out[f"l_up_h_{key}"] = l_up
        out[f"u_in_for_{key}"] = inlet
        out[f"centre_ratio_at_minus_4_{key}"] = ratio
        print(
            f"{label}: reached at x/H = {x_cross:.2f}; L_up = {l_up} H; at x/H = -4 of "
            f"that length the centreline is {ratio:.4f} U_in, so U_in = {inlet:.3f} m/s "
            f"makes it U_ref"
        )
    # Decision 1's purpose is the measured layer at x/H = -4, so each candidate
    # length is also scored by the RMS of u / U_centre against the data's 17
    # heights at that station, the layer read 4 H before the candidate's end.
    profile = backstep.PROFILES[backstep.REFERENCE_STATION]
    h = backstep.STEP_HEIGHT
    print()
    print(
        "| L_up / H | x/H read | RMS u / U_centre against the x/H = -4 data | largest |"
    )
    print("|---|---|---|---|")
    out["candidates"] = []
    for l_up in sorted({out.get("l_up_h_delta"), out.get("l_up_h_theta")} - {None}):
        x = (l_up - 4.0) * h
        n = int(np.argmin(np.abs(mesh.x - x)))
        ys = np.array(profile.y_over_h) * h
        u = backstep.sample_u(mesh, arrays["u_faces"], np.full(ys.size, mesh.x[n]), ys)
        u_c = float(along["u_centre"][n - 1])
        err = backstep.errors(u / u_c, np.array(profile.u_over_uref))
        out["candidates"].append(
            {"l_up_h": l_up, "x_h_read": float(mesh.x[n] / h), **err.__dict__}
        )
        print(f"| {l_up} | {mesh.x[n] / h:.1f} | {err.rms:.4f} | {err.max_abs:.4f} |")
    (OUT / f"precursor_{args.name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
