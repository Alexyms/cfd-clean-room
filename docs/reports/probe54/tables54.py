"""The tables of docs/reports/multigrid_default.md, generated from prompt 54's records.

Usage:
    python tables54.py

Reads results/builder54/ of the main checkout (the records of weight54.py,
plant54.py, row54.py and inject54.py), the Jacobi records of earlier prompts
(results/builder51/ for the Annex 20 room, the product and VAL-016; prompt
53's reruns in results/builder53/; prompt 52's driver run for the step) and
benchmarks/results.jsonl (the step's clean Jacobi row), and replaces each
region of the report between ``<!-- BEGIN tables54: NAME -->`` and
``<!-- END tables54: NAME -->`` with the table NAME, so no number in the tables
is typed by hand. A region whose records are missing is left as it was and
named on stdout.
"""

import json
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[3]
MAIN = Path(
    subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=HERE,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
).parent
OUT = MAIN / "results" / "builder54"
OUT53 = MAIN / "results" / "builder53"
OUT52 = MAIN / "results" / "builder52"
OUT51 = MAIN / "results" / "builder51"
REPORT = HERE / "docs" / "reports" / "multigrid_default.md"
STEP2_BASELINE = HERE / "docs" / "reports" / "ecr003_step2_baseline.md"
RESULTS = HERE / "benchmarks" / "results.jsonl"
# The step's clean Jacobi row of record, alone on the machine (prompt 52, report
# section 5.9 there).
STEP_JACOBI_RUN_ID = "702969926e4a45dfa1c0961fc5f36a87"
CASE_NAMES = {
    "val001": "VAL-001 80x40",
    "cavity": "VAL-002 cavity 80x80 (closed)",
    "product160": "Product 160x60",
    "product320": "Product 320x120",
    "annex": "Annex 20 216x72",
    "step": "Step medium 656x60",
}
LAMINAR = (
    "val001_80x40",
    "val001_80x40_stretched",
    "val002_20x20",
    "val002_40x40",
    "val002_80x80",
)
CAVITIES = ("val002_20x20", "val002_40x40", "val002_80x80")
GRIDS = ("80x30", "160x60")
# The committed criteria: tests/test_poiseuille.py, tests/test_lid_cavity.py and
# the step 8 report for the cavity; prompt 54's own tolerances for the rest.
POISEUILLE_BOUND = 0.01
CAVITY_BOUND = 0.02
FACE_TOL = 1e-3
XR_TOL = 0.02
OUTER_TOL = 0.05
SETUP_SHARE = 0.10
SETUP_CUT = 0.5


def load(path: Path) -> dict | list:
    """One JSON record."""
    return json.loads(path.read_text())


def row(*cells: object) -> str:
    """One markdown table row, a pipe inside a cell escaped."""
    return "| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |"


def table(header: list[str], rows: list[list[object]]) -> str:
    """A markdown table."""
    lines = [row(*header), row(*["---"] * len(header))]
    lines += [row(*r) for r in rows]
    return "\n".join(lines)


def ms(seconds: float) -> str:
    """Milliseconds to two decimals."""
    return f"{1e3 * seconds:.2f}"


def short(digest: str) -> str:
    """The first sixteen hex digits of a face hash, as the earlier reports quote it."""
    return f"`{digest[:16]}`"


# ---------------------------------------------------------------------------
# Item 6: the weight alone
# ---------------------------------------------------------------------------


def weights_record() -> dict:
    """Item 6's record."""
    return load(OUT / "weight54.json")


def system_name(r: dict) -> str:
    """A captured system's label."""
    return f"{CASE_NAMES[r['case']]}, outer {r['outer']:,}"


def weight_systems() -> str:
    """Each system under the old and the fixed weight: iterations, setup, solve."""
    rows = []
    for r in weights_record()["rows"]:
        rows.append(
            [
                system_name(r),
                f"{r['old']['iterations']} / {r['new']['iterations']}",
                f"{r['solution_difference']:.1e}",
                f"{ms(r['old']['setup_seconds'])} / {ms(r['new']['setup_seconds'])}",
                f"{r['setup_ratio']:.2f}",
                f"{100 * r['old']['setup_share']:.1f}%",
                f"{ms(r['old']['solve_seconds'])} / {ms(r['new']['solve_seconds'])}",
                f"{ms(r['old']['total_seconds'])} / {ms(r['new']['total_seconds'])}",
                f"{r['total_ratio']:.2f}",
            ]
        )
    return table(
        [
            "System",
            "CG its, old / new",
            "max abs p' difference / max abs p', old",
            "Setup ms, old / new",
            "Setup new / old",
            "Old setup share of the correction",
            "Solve ms, old / new",
            "Total ms, old / new",
            "Total new / old",
        ],
        rows,
    )


def weight_levels() -> str:
    """The weight each level had and has."""
    rows = []
    for r in weights_record()["rows"]:
        smoothed = r["levels"][:-1]
        old = [lv["old_weight"] for lv in smoothed]
        est = [lv["old_eigenvalue"] for lv in smoothed]
        new = {lv["new_weight"] for lv in smoothed}
        rows.append(
            [
                system_name(r),
                len(smoothed),
                f"{min(est):.4f} to {max(est):.4f}",
                f"{min(old):.4f} to {max(old):.4f}",
                ", ".join(f"{w:.4f}" for w in sorted(new)),
            ]
        )
    return table(
        [
            "System",
            "Smoothed levels",
            "Old lambda_max estimate",
            "Old weight",
            "New weight",
        ],
        rows,
    )


def setup_parts() -> str:
    """Where the fixed-weight setup goes, by part (``setup54.py``)."""
    rows = []
    for r in load(OUT / "setup54.json")["rows"]:
        whole = r["whole"]
        rows.append(
            [
                system_name(r),
                ms(whole),
                ms(r["dominance"]),
                ms(r["coarsening"]),
                ms(r["smoothers"]),
                ms(r["pseudo_inverse"]),
                f"{100 * r['pseudo_inverse'] / whole:.0f}%",
            ]
        )
    return table(
        [
            "System",
            "Setup ms",
            "Dominance check",
            "Coarsening, all levels",
            "Smoother arrays",
            "Coarsest dense pseudo-inverse",
            "Pseudo-inverse share",
        ],
        rows,
    )


# ---------------------------------------------------------------------------
# The planted defects
# ---------------------------------------------------------------------------


def plants() -> str:
    """Commit A's and commit B's planted defects."""
    rows = []
    for group in ("A", "B"):
        for rec in load(OUT / f"plants_{group}.json"):
            failing = [name.split("::", 1)[1] for name in rec["failures"]]
            rows.append(
                [
                    f"`{rec['plant']}`",
                    f"{group} (`{rec['commit']}`)",
                    rec["meaning"],
                    "caught" if rec["caught"] else "SURVIVED",
                    len(failing),
                    "; ".join(failing[:3]) + (" ..." if len(failing) > 3 else ""),
                ]
            )
    return table(
        [
            "Plant",
            "Commit",
            "Defect",
            "Result",
            "Failing tests",
            "Failing (first three)",
        ],
        rows,
    )


# ---------------------------------------------------------------------------
# Item 7: the baseline
# ---------------------------------------------------------------------------


def laminar_pair(case: str) -> tuple[dict, dict]:
    """(default, diagonal) records of a laminar preset."""
    return load(OUT / f"row_{case}_mg.json"), load(OUT / f"row_{case}_jacobi.json")


def prompt53_hash(case: str) -> str | None:
    """The face hash prompt 53 recorded for a preset under the diagonal, if it did."""
    path = OUT53 / f"val_{case}_A.json"
    return load(path)["face_hash"] if path.exists() else None


def step_jacobi_row() -> dict:
    """The step's clean Jacobi row of record from benchmarks/results.jsonl."""
    for line in RESULTS.read_text().splitlines():
        rec = json.loads(line)
        if rec["run_id"] == STEP_JACOBI_RUN_ID:
            return rec
    raise KeyError(STEP_JACOBI_RUN_ID)


def value_text(accuracy: dict) -> str:
    """A harness accuracy as the tables quote it: the components of a cavity, else the value."""
    comps = accuracy.get("components")
    if comps and set(comps) == {"u", "v"}:
        return f"u {comps['u']:.4e}, v {comps['v']:.4e}"
    return f"{accuracy['value']:.4e}"


def baseline_rows() -> list[dict]:
    """One dict per item 7 case: the Jacobi record, the default's record, the hashes."""
    rows: list[dict] = []
    for case in LAMINAR:
        mg, jac = laminar_pair(case)
        mg_row, jac_row = mg["row"], jac["row"]
        rows.append(
            {
                "case": case,
                "kind": "laminar",
                "jacobi_source": "rerun under staggered-cg, this session",
                "jacobi_stop": jac_row["outcome"]["stop_reason"],
                "stop": mg_row["outcome"]["stop_reason"],
                "jacobi_outer": jac_row["work"]["outer_iterations"],
                "outer": mg_row["work"]["outer_iterations"],
                "jacobi_hash": jac["face_hash"],
                "hash": mg["face_hash"],
                "commit": mg["commit"],
                "jacobi_accuracy": jac_row["accuracy"],
                "accuracy": mg_row["accuracy"],
                "caps": mg_row["outcome"]["pressure_cap_hits"],
            }
        )
    annex = load(OUT / "row_val017_216x72_mg.json")
    record = load(OUT51 / "annex_cg.json")
    rows.append(
        {
            "case": "val017_216x72",
            "kind": "annex",
            "jacobi_source": "prompt 51 record",
            "jacobi_stop": record["row"]["outcome"]["stop_reason"],
            "stop": annex["row"]["outcome"]["stop_reason"],
            "jacobi_outer": record["row"]["work"]["outer_iterations"],
            "outer": annex["row"]["work"]["outer_iterations"],
            "jacobi_hash": record["face_hash"],
            "hash": annex["face_hash"],
            "commit": annex["commit"],
            "jacobi_accuracy": record["row"]["accuracy"],
            "accuracy": annex["row"]["accuracy"],
            "caps": annex["row"]["outcome"]["pressure_cap_hits"],
        }
    )
    step = load(OUT / "row_val019_656x60_mg.json")
    jac_step = step_jacobi_row()
    driver = load(OUT52 / "standard_656x60_upwind.json")
    rows.append(
        {
            "case": "val019_656x60",
            "kind": "step",
            "jacobi_source": "prompt 52 row of record",
            "jacobi_stop": jac_step["outcome"]["stop_reason"],
            "stop": step["row"]["outcome"]["stop_reason"],
            "jacobi_outer": jac_step["work"]["outer_iterations"],
            "outer": step["row"]["work"]["outer_iterations"],
            "jacobi_hash": None,
            "jacobi_driver_hash": driver["face_hash"],
            "hash": step["face_hash"],
            "commit": step["commit"],
            "jacobi_accuracy": jac_step["accuracy"],
            "accuracy": step["row"]["accuracy"],
            "caps": step["row"]["outcome"]["pressure_cap_hits"],
        }
    )
    for grid in GRIDS:
        mg = load(OUT / f"product_{grid}_mg.json")
        record = load(OUT51 / f"product_{grid}_cg.json")
        rows.append(
            {
                "case": f"product_{grid}",
                "kind": "product",
                "jacobi_source": "prompt 51 record",
                "jacobi_stop": record["stop"],
                "stop": mg["stop"],
                "jacobi_outer": record["outer"],
                "outer": mg["outer"],
                "jacobi_hash": record["face_hash"],
                "hash": mg["face_hash"],
                "commit": mg["commit"],
                "caps": mg["pressure_cap_hits"],
                "grid": grid,
                "scale": mg["velocity_scale"],
            }
        )
    v16 = load(OUT / "val016_mg.json")
    j16 = load(OUT51 / "val016_cg.json")
    rows.append(
        {
            "case": "val016",
            "kind": "val016",
            "jacobi_source": "prompt 51 record",
            "jacobi_stop": j16["stop"],
            "stop": v16["stop"],
            "jacobi_outer": j16["outer"],
            "outer": v16["outer"],
            "jacobi_hash": None,
            "hash": None,
            "commit": v16["commit"],
            "caps": v16["pressure_cap_hits"],
            "v16": v16,
            "j16": j16,
        }
    )
    return rows


def faces_difference(a: Path, b: Path) -> float:
    """Largest face difference between two saved face files."""
    fa, fb = np.load(a), np.load(b)
    return max(
        float(np.abs(fa["u_faces"] - fb["u_faces"]).max()),
        float(np.abs(fa["v_faces"] - fb["v_faces"]).max()),
    )


def answers(r: dict) -> tuple[str, str, str]:
    """(Jacobi answer, default answer, criterion) of one baseline row, as text."""
    kind = r["kind"]
    if kind == "laminar":
        case = r["case"]
        criterion = {
            "val001_80x40": f"L2 < {POISEUILLE_BOUND}",
            "val001_80x40_stretched": f"L2 < {POISEUILLE_BOUND}",
            "val002_80x80": f"u, v < {CAVITY_BOUND} (criterion 3)",
        }.get(case, "criterion 3a (table below)")
        return value_text(r["jacobi_accuracy"]), value_text(r["accuracy"]), criterion
    if kind == "annex":
        return (
            f"{r['jacobi_accuracy']['value']:.5f}",
            f"{r['accuracy']['value']:.5f}",
            "worst scored RMS / allowance < 1",
        )
    if kind == "step":
        a, b = r["jacobi_accuracy"]["components"], r["accuracy"]["components"]
        return (
            f"x_r/H {a['x_r_over_h']:.4f}",
            f"x_r/H {b['x_r_over_h']:.4f}",
            f"within {XR_TOL} H of the Jacobi row",
        )
    if kind == "product":
        diff = faces_difference(
            OUT51 / f"product_{r['grid']}_cg.npz", OUT / f"product_{r['grid']}_mg.npz"
        )
        return "-", f"{diff / r['scale']:.2e}", f"faces within {FACE_TOL} of the scale"
    v, j = r["v16"], r["j16"]
    return (
        f"du/U_w {j['a_du_over_uw']:.3e}, dk/u_tau^2 {j['a_dk_over_utau2']:.3e}",
        f"du/U_w {v['a_du_over_uw']:.3e}, dk/u_tau^2 {v['a_dk_over_utau2']:.3e}",
        f"(a) both < {v['a_tol']:g}; (b) core k within {v['b_tol']:g} of 1",
    )


def baseline() -> str:
    """Stop, outer count, answer against the Jacobi record and criterion, per case."""
    rows = []
    for r in baseline_rows():
        jacobi, new, criterion = answers(r)
        change = (r["outer"] - r["jacobi_outer"]) / r["jacobi_outer"]
        rows.append(
            [
                r["case"],
                r["jacobi_source"],
                f"{r['jacobi_stop']} at {r['jacobi_outer']:,}",
                f"{r['stop']} at {r['outer']:,}",
                f"{change:+.2%}",
                jacobi,
                new,
                criterion,
                r["caps"],
            ]
        )
    return table(
        [
            "Case",
            "Jacobi record",
            "Jacobi stop",
            "Default stop",
            "Outer change",
            "Jacobi answer",
            "Default answer (products: largest face difference over the scale)",
            "Criterion",
            "Capped corrections",
        ],
        rows,
    )


def hashes() -> str:
    """The old and the new face hash of every case that has one, with the commit."""
    rows = []
    for r in baseline_rows():
        if r["hash"] is None:
            continue
        old = r["jacobi_hash"]
        old_text = short(old) if old else "none stored"
        extra = ""
        if r["case"] == "val019_656x60":
            extra = f"prompt 52's driver run: {short(r['jacobi_driver_hash'])}"
        elif r["kind"] == "laminar" and old:
            sources = []
            held = prompt53_hash(r["case"])
            if held == old:
                sources.append("prompt 53's record")
            if old[:16] in STEP2_BASELINE.read_text():
                sources.append("`ecr003_step2_baseline.md`")
            extra = (
                "equals " + " and ".join(sources) if sources else "no earlier record"
            )
        rows.append(
            [
                r["case"],
                old_text,
                short(r["hash"]),
                "yes" if old and old != r["hash"] else ("-" if not old else "NO"),
                f"`{r['commit']}`",
                extra,
            ]
        )
    return table(
        [
            "Case",
            "Hash under the diagonal (kept)",
            "Hash under the default (of record)",
            "Changed",
            "Commit",
            "Note",
        ],
        rows,
    )


def cavity_orders() -> str:
    """Criterion 3a: u and v of the three cavity grids, the orders, default against diagonal."""
    rows = []
    for axis in ("u", "v"):
        for label, key in (("Jacobi", "jacobi_accuracy"), ("default", "accuracy")):
            by_grid = {
                r["case"]: r[key]["components"][axis]
                for r in baseline_rows()
                if r["case"] in CAVITIES
            }
            e = [by_grid[c] for c in CAVITIES]
            orders = [np.log2(e[i] / e[i + 1]) for i in range(2)]
            rows.append(
                [
                    axis,
                    label,
                    " / ".join(f"{x:.4e}" for x in e),
                    " / ".join(f"{o:.2f}" for o in orders),
                    "yes" if e[0] > e[1] > e[2] else "NO",
                ]
            )
    return table(
        [
            "Component",
            "Preconditioner",
            "Error at 20x20 / 40x40 / 80x80",
            "Order, 20 to 40 / 40 to 80",
            "Falls at each refinement",
        ],
        rows,
    )


def line_difference(a: dict, b: dict) -> dict[str, float]:
    """Largest difference on each Annex 20 line, u / u0."""
    return {key: float(np.abs(np.array(a[key]) - np.array(b[key])).max()) for key in a}


def annex() -> str:
    """The Annex 20 answers: each line's RMS, allowance and pass, and the line differences."""
    mg = load(OUT / "row_val017_216x72_mg.json")
    record = load(OUT51 / "annex_cg.json")
    diff = line_difference(mg["lines"], record["lines"])
    rows = []
    for key in mg["lines"]:
        a = mg["row"]["accuracy"]["lines"][key]
        b = record["row"]["accuracy"]["lines"][key]
        rows.append(
            [
                key,
                "yes" if a["scored"] else "no",
                f"{b['rms']:.6f}",
                f"{a['rms']:.6f}",
                f"{a['allowance']:.6f}",
                {True: "pass", False: "FAIL", None: "-"}[a["pass"]],
                f"{diff[key]:.2e}",
            ]
        )
    return table(
        [
            "Line",
            "Scored",
            "RMS / u0, Jacobi record",
            "RMS / u0, default",
            "Allowance",
            "Default",
            "Largest difference from the Jacobi record, u / u0",
        ],
        rows,
    )


def step() -> str:
    """The step's report-only figures, the Jacobi row of record against the default."""
    jacobi = step_jacobi_row()["accuracy"]["components"]
    new = load(OUT / "row_val019_656x60_mg.json")["row"]["accuracy"]["components"]
    rows = []
    for key in jacobi:
        rows.append(
            [
                key,
                f"{jacobi[key]:.6g}",
                f"{new[key]:.6g}",
                f"{new[key] - jacobi[key]:+.3g}",
            ]
        )
    return table(["Figure", "Jacobi row of record", "Default", "Difference"], rows)


def val016() -> str:
    """VAL-016's split criteria, the Jacobi record against the default."""
    rows = []
    for label, rec in (
        ("jacobi, prompt 51 record", load(OUT51 / "val016_cg.json")),
        ("default", load(OUT / "val016_mg.json")),
    ):
        rows.append(
            [
                label,
                f"`{rec['commit']}`",
                rec["stop"],
                rec["outer"],
                f"{rec['a_du_over_uw']:.3e}",
                f"{rec['a_dk_over_utau2']:.3e}",
                f"{rec['a_u_tau_ratio']:.7f}",
                "{:.5f} to {:.5f}".format(*rec["b_core_k_ratio"]["standard"]),
                "{:.5f} to {:.5f}".format(*rec["b_core_k_ratio"]["rng"]),
            ]
        )
    return table(
        [
            "Preconditioner",
            "Commit",
            "Stop",
            "Outer",
            "(a) max du / U_w (< 1e-4)",
            "(a) max dk / u_tau^2 (< 1e-4)",
            "(a) u_tau / 1D",
            "(b) standard core k ratio",
            "(b) RNG core k ratio",
        ],
        rows,
    )


# ---------------------------------------------------------------------------
# The predictions
# ---------------------------------------------------------------------------


def prediction_a() -> tuple[str, str]:
    """(measured, verdict) of prediction (a)."""
    rows = weights_record()["rows"]
    changes = [abs(r["new"]["iterations"] - r["old"]["iterations"]) for r in rows]
    big = [r for r in rows if r["old"]["setup_share"] > SETUP_SHARE]
    cut = [r for r in big if r["setup_ratio"] <= SETUP_CUT]
    ratios = [r["setup_ratio"] for r in big]
    measured = (
        f"iteration counts equal on {changes.count(0)} of {len(rows)} systems, the largest "
        f"change {max(changes)}; setup new / old {min(ratios):.2f} to {max(ratios):.2f} on "
        f"the {len(big)} systems whose old setup share was over {SETUP_SHARE:.0%}, at most "
        f"half on {len(cut)} of them"
    )
    first = max(changes) <= 1
    second = len(cut) == len(big)
    if first and second:
        verdict = "holds"
    elif first:
        verdict = "first clause holds; second clause fails"
    else:
        verdict = "fails"
    return measured, verdict


def prediction_b() -> tuple[str, str]:
    """(measured, verdict) of prediction (b)."""
    parts: list[str] = []
    failures: list[str] = []
    for r in baseline_rows():
        same_stop = r["stop"] == r["jacobi_stop"]
        change = (r["outer"] - r["jacobi_outer"]) / r["jacobi_outer"]
        ok = same_stop and abs(change) <= OUTER_TOL and r["caps"] == 0
        kind = r["kind"]
        if kind == "laminar":
            acc = r["accuracy"]
            if r["case"].startswith("val001"):
                ok = ok and acc["value"] < POISEUILLE_BOUND
            elif r["case"] == "val002_80x80":
                ok = ok and max(acc["components"].values()) < CAVITY_BOUND
        elif kind == "annex":
            ok = ok and bool(r["accuracy"]["passed"])
        elif kind == "step":
            a = r["jacobi_accuracy"]["components"]["x_r_over_h"]
            b = r["accuracy"]["components"]["x_r_over_h"]
            ok = ok and abs(a - b) <= XR_TOL
        elif kind == "product":
            diff = faces_difference(
                OUT51 / f"product_{r['grid']}_cg.npz",
                OUT / f"product_{r['grid']}_mg.npz",
            )
            ok = ok and diff <= FACE_TOL * r["scale"]
        elif kind == "val016":
            v = r["v16"]
            ok = (
                ok
                and v["a_du_over_uw"] < v["a_tol"]
                and v["a_dk_over_utau2"] < v["a_tol"]
                and all(
                    abs(x - 1.0) < v["b_tol"]
                    for key in ("standard", "rng")
                    for x in v["b_core_k_ratio"][key]
                )
            )
        if not ok:
            failures.append(r["case"])
        parts.append(f"{r['case']} {change:+.2%}")
    by_grid = {r["case"]: r for r in baseline_rows() if r["case"] in CAVITIES}
    for axis in ("u", "v"):
        e = [by_grid[c]["accuracy"]["components"][axis] for c in CAVITIES]
        if not e[0] > e[1] > e[2]:
            failures.append(f"criterion 3a ({axis})")
    measured = "outer change: " + ", ".join(parts)
    if failures:
        return measured + "; failing: " + ", ".join(failures), "fails"
    return measured, "holds"


def prediction_c() -> tuple[str, str]:
    """(measured, verdict) of prediction (c)."""
    changed, moved = [], []
    for r in baseline_rows():
        if r["kind"] != "laminar":
            continue
        changed.append(r["hash"] != r["jacobi_hash"])
        a, b = r["jacobi_accuracy"], r["accuracy"]
        if a.get("components"):
            pairs = [(a["components"][k], b["components"][k]) for k in a["components"]]
        else:
            pairs = [(a["value"], b["value"])]
        for x, y in pairs:
            if f"{x:.4e}" != f"{y:.4e}":
                moved.append(f"{r['case']} {x:.5e} to {y:.5e}")
    measured = (
        f"hash changed on {sum(changed)} of {len(changed)} laminar cases; criterion "
        f"values equal at the five significant figures the tables print on "
        f"{len(changed) - len(moved)} of {len(changed)}"
        + (": " + "; ".join(moved) if moved else "")
    )
    verdict = "holds" if all(changed) and not moved else "fails"
    return measured, verdict


def predictions() -> str:
    """The three predictions against the measurements."""
    rows = []
    for label, text, (measured, verdict) in (
        (
            "(a)",
            "The fixed weight changes no captured system's iteration count by more than "
            "one, and cuts setup by at least half where it was over 10% of the correction",
            prediction_a(),
        ),
        (
            "(b)",
            "Every case in item 7 reaches the same kind of stop as its Jacobi record, within "
            "5% of the outer count, and meets its criterion",
            prediction_b(),
        ),
        (
            "(c)",
            "Every laminar face hash changes and no laminar criterion moves by more than "
            "its rounding",
            prediction_c(),
        ),
    ):
        rows.append([label, text, measured, verdict])
    return table(["", "Prediction", "Measured", "Verdict"], rows)


TABLES: dict[str, Callable[[], str]] = {
    "weight_systems": weight_systems,
    "weight_levels": weight_levels,
    "setup_parts": setup_parts,
    "plants": plants,
    "baseline": baseline,
    "hashes": hashes,
    "cavity_orders": cavity_orders,
    "annex": annex,
    "step": step,
    "val016": val016,
    "predictions": predictions,
}


def main() -> None:
    """Replace every region whose records exist."""
    raw = REPORT.read_bytes().decode()
    newline = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    for name, build in TABLES.items():
        pattern = re.compile(
            rf"(<!-- BEGIN tables54: {name} -->\n).*?(<!-- END tables54: {name} -->)",
            re.DOTALL,
        )
        if not pattern.search(text):
            continue
        try:
            body = build()
        except (FileNotFoundError, KeyError) as missing:
            print(f"{name}: records missing ({missing})")
            continue
        text = pattern.sub(
            lambda m, body=body: m.group(1) + body + "\n" + m.group(2), text
        )
        print(f"{name}: written")
    REPORT.write_bytes(text.replace("\n", newline).encode())


if __name__ == "__main__":
    main()
