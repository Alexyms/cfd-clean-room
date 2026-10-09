"""Round 2 (prompt 44b item 4): the bounded rows characterised from their records.

Usage:
    python bounded44.py history NAME [NAME ...]    periodicity, amplitude and drift of the residual
    python bounded44.py where NAME                 where the largest change sits, by region (a
                                                   record run with conv44.py --locate)

`history` reads NAME.json under results/builder44/ and, over the tail of the
run (the last TAIL iterations, or the whole run past outer 100 when shorter),
reports: the residual's least, median and largest; the amplitude as the
ratio of the 95th to the 5th percentile; the drift as the slope of the
log10 residual per thousand iterations, least squares over the tail; and the
period as the lag of the first autocorrelation peak of the log10 residual
after the autocorrelation first goes negative, with the peak's height, as
compare34b.py took it (step 0's report, section 7.1), so a weak peak is not
read as a period. The same is done for the largest speed. `where` reads the
per-iteration cell of the largest change of u and of v, assigns each to a
region of the room from the configuration (a return, the hood, the supply
row under the ceiling, an obstacle face or top, the gap between two
obstacles, or elsewhere), and prints the share of the tail's iterations in
each region with the most frequent cells. Output goes to stdout and to
results/builder44/bounded_NAME.json.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "builder44"
TAIL = 2000
NEAR = 0.2


def load(name: str) -> dict:
    """A record by name."""
    return json.loads((OUT / f"{name}.json").read_text())


def period_of(series: np.ndarray) -> dict:
    """compare34b's period, and separately the strongest recurrence.

    ``period`` is step 0's measure (``compare34b.py``, step 0's report,
    appendix H): the first lag after the autocorrelation first goes negative
    at which it is a local maximum and positive, with its height, so a weak
    peak is reported as weak. ``strongest`` is the lag of the highest
    autocorrelation from that first negative lag to half the series, with its
    height; it is what round 2 first reported as the period (test 44b, B2).
    """
    a = np.asarray(series, dtype=float)
    a = a - a.mean()
    none = {"period": None, "height": None, "strongest": None, "strongest_height": None}
    if a.size < 50 or not np.any(a):
        return none
    full = np.correlate(a, a, mode="full")[a.size - 1 :]
    corr = full / full[0]
    negative = np.nonzero(corr < 0.0)[0]
    if negative.size == 0:
        return none
    start = int(negative[0])
    out = dict(none)
    for k in range(start + 1, corr.size - 1):
        if corr[k] >= corr[k - 1] and corr[k] >= corr[k + 1] and corr[k] > 0.0:
            out["period"], out["height"] = k, float(corr[k])
            break
    half = a.size // 2
    k = int(np.argmax(corr[start:half])) + start
    out["strongest"], out["strongest_height"] = k, float(corr[k])
    return out


def history(args: argparse.Namespace) -> None:
    """Periodicity, amplitude and drift of the residual and the largest speed over the tail."""
    for name in args.names:
        rec = load(name)
        res = np.array(rec["residual"])
        speed = np.array(rec["max_speed"])
        n = res.size
        tail = min(TAIL, max(n - 100, 0))
        r, s = res[-tail:], speed[-tail:]
        log_r = np.log10(r)
        x = np.arange(tail) / 1000.0
        slope = float(np.polyfit(x, log_r, 1)[0])
        out = {
            "name": name,
            "stop": rec["stop"],
            "outer": n,
            "tail_iterations": tail,
            "residual": {
                "least": float(r.min()),
                "median": float(np.median(r)),
                "largest": float(r.max()),
                "amplitude_p95_over_p5": float(
                    np.percentile(r, 95) / np.percentile(r, 5)
                ),
                "drift_log10_per_thousand": slope,
                **{f"acf_{k}": v for k, v in period_of(log_r).items()},
            },
            "max_speed": {
                "least": float(s.min()),
                "median": float(np.median(s)),
                "largest": float(s.max()),
                "spread": float(s.max() - s.min()),
                "cells": collections.Counter(map(tuple, rec["at"][-tail:])).most_common(
                    3
                ),
                **{f"acf_{k}": v for k, v in period_of(s).items()},
            },
            "medians_per_thousand": [
                float(np.median(res[a : a + 1000])) for a in range(0, n, 1000)
            ],
        }
        print(json.dumps(out, indent=1))
        (OUT / f"bounded_{name}.json").write_text(json.dumps(out))


def regions(raw: dict) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Named boxes, in priority order, each (x0, x1, y0, y1) grown by NEAR."""
    width, height = raw["domain"]["width"], raw["domain"]["height"]
    boxes: list[tuple[str, tuple[float, float, float, float]]] = []
    for name, seg in raw["boundaries"].items():
        if seg["type"] == "fixed_flow_outlet" and seg["location"] == "bottom":
            boxes.append(
                (f"return {name[-1]}", (seg["x_start"], seg["x_end"], 0.0, 0.0))
            )
        elif seg["type"] == "fixed_flow_outlet":
            boxes.append(("hood", (width, width, seg["y_start"], seg["y_end"])))
    for o in raw["obstacles"]:
        boxes.append(
            (f"{o['name']} top", (o["x_start"], o["x_end"], o["y_end"], o["y_end"]))
        )
        boxes.append(
            (f"{o['name']} face", (o["x_start"], o["x_end"], o["y_start"], o["y_end"]))
        )
    supply = raw["boundaries"]["hepa_supply"]
    boxes.append(("supply row", (supply["x_start"], supply["x_end"], height, height)))
    return [
        (n, (x0 - NEAR, x1 + NEAR, y0 - NEAR, y1 + NEAR))
        for n, (x0, x1, y0, y1) in boxes
    ]


def region_of(x: float, y: float, boxes: list, tops: float) -> str:
    """The first box a point falls in, else a gap between obstacles or elsewhere."""
    for name, (x0, x1, y0, y1) in boxes:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return name
    return "gap between equipment" if y < tops else "core above equipment"


def where(args: argparse.Namespace) -> None:
    """Share of the tail's iterations whose largest change sits in each region."""
    rec = load(args.name)
    if "du_at" not in rec:
        raise ValueError(f"{args.name} was not run with --locate")
    raw = yaml.safe_load((ROOT / "configs/clean_room_default.yaml").read_text())
    boxes = regions(raw)
    tops = max(o["y_end"] for o in raw["obstacles"])
    n = len(rec["residual"])
    tail = min(TAIL, max(n - 100, 0))
    out: dict = {"name": args.name, "outer": n, "tail_iterations": tail, "near": NEAR}
    for comp in ("u", "v"):
        cells = [tuple(c) for c in rec[f"d{comp}_at"][-tail:]]
        sizes = np.array(rec[f"d{comp}"][-tail:])
        by_region = collections.Counter(region_of(x, y, boxes, tops) for x, y in cells)
        out[comp] = {
            "share_by_region": {k: v / tail for k, v in by_region.most_common()},
            "cells": collections.Counter(cells).most_common(5),
            "change_m_per_s": {
                "least": float(sizes.min()),
                "median": float(np.median(sizes)),
                "largest": float(sizes.max()),
            },
        }
    both = [
        rec["du_at"][i] if rec["du"][i] >= rec["dv"][i] else rec["dv_at"][i]
        for i in range(n - tail, n)
    ]
    out["largest_of_both"] = {
        "share_by_region": {
            k: v / tail
            for k, v in collections.Counter(
                region_of(x, y, boxes, tops) for x, y in both
            ).most_common()
        },
        "cells": collections.Counter(map(tuple, both)).most_common(5),
    }
    print(json.dumps(out, indent=1))
    (OUT / f"bounded_where_{args.name}.json").write_text(json.dumps(out))


def main() -> None:
    """Dispatch one of the modes the module docstring lists."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("history")
    p.add_argument("names", nargs="+")
    p.set_defaults(func=history)
    p = sub.add_parser("where")
    p.add_argument("name")
    p.set_defaults(func=where)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
