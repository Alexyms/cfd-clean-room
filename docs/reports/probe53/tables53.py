"""The tables of docs/reports/pressure_multigrid.md, generated from prompt 53's records.

Usage:
    python tables53.py

Reads results/builder53/ of the main checkout (the records of dev53.py,
plant53.py, check53.py, measure53.py, row53.py and inject53.py), prompt
51's records in results/builder51/ (the Jacobi records of the Annex 20,
product and VAL-016 runs) and benchmarks/results.jsonl (the step's clean
Jacobi row), and replaces each region of the report between
``<!-- BEGIN tables53: NAME -->`` and ``<!-- END tables53: NAME -->`` with
the table NAME, so no number in the tables is typed by hand. A region whose
records are missing is left as it was and named on stdout.
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
OUT = MAIN / "results" / "builder53"
OUT51 = MAIN / "results" / "builder51"
REPORT = HERE / "docs" / "reports" / "pressure_multigrid.md"
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
LAMINAR = ("val001_80x40", "val001_80x40_stretched", "val002_80x80")
GRIDS = ("80x30", "160x60")
# Prompt 51's tolerances for the answers, prediction (e).
LINE_TOL = 1e-4
FACE_TOL = 1e-3
XR_TOL = 0.02
OUTER_TOL = 0.05


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
    """Milliseconds to one decimal."""
    return f"{1e3 * seconds:.1f}"


def measure() -> dict:
    """Item 6's record."""
    return load(OUT / "measure53.json")


def stopped_by(r: dict) -> str:
    """What ends the correction: the relative level or the rounding floor."""
    return "floor" if r["floor_over_rtol_f"] > 1.0 else "pressure_rtol"


def system_name(r: dict) -> str:
    """A captured system's label."""
    return f"{CASE_NAMES[r['case']]}, outer {r['outer']:,}"


# ---------------------------------------------------------------------------
# Section 2: the development systems
# ---------------------------------------------------------------------------


def dev() -> str:
    """nu and the power-iteration count on the development systems."""
    rows = []
    for rec in load(OUT / "dev53.json"):
        jacobi = rec["jacobi"]
        rows.append(
            [
                rec["system"],
                "Jacobi-PCG",
                jacobi["iterations"],
                ms(jacobi["setup_seconds"]),
                ms(jacobi["solve_seconds"]),
                ms(jacobi["total_seconds"]),
            ]
        )
        for key, value in rec.items():
            if not key.startswith("nu"):
                continue
            nu, power = key[2:].split("_power")
            rows.append(
                [
                    "",
                    f"V({nu},{nu}), {power} power its",
                    value["iterations"],
                    ms(value["setup_seconds"]),
                    ms(value["solve_seconds"]),
                    ms(value["total_seconds"]),
                ]
            )
    return table(
        ["System", "Preconditioner", "CG its", "Setup ms", "Solve ms", "Total ms"], rows
    )


# ---------------------------------------------------------------------------
# Section 4: unchanged by construction, and the plants
# ---------------------------------------------------------------------------


def laminar() -> str:
    """Item 5: the laminar cases under the default, base against commit A."""
    rows = []
    for case in LAMINAR:
        a = load(OUT / f"val_{case}_base.json")
        b = load(OUT / f"val_{case}_A.json")
        rows.append(
            [
                case,
                f"`{a['commit']}`",
                f"`{b['commit']}`",
                f"{a['stop']} at {a['outer']:,}",
                f"{b['stop']} at {b['outer']:,}",
                f"`{a['face_hash'][:16]}`",
                f"`{b['face_hash'][:16]}`",
                "yes"
                if a["face_hash"] == b["face_hash"] and a["outer"] == b["outer"]
                else "NO",
            ]
        )
    return table(
        [
            "Case",
            "Base",
            "Commit A",
            "Base stop",
            "A stop",
            "Base face hash",
            "A face hash",
            "Equal",
        ],
        rows,
    )


def plants() -> str:
    """Commit A's planted defects."""
    rows = []
    for rec in load(OUT / "plants.json"):
        failing = sorted(
            {
                f.split("::")[1] + "::" + f.split("::")[2].split("[")[0]
                for f in rec["failures"]
            }
        )
        rows.append(
            [
                f"`{rec['plant']}`",
                rec["meaning"],
                "caught" if rec["caught"] else "SURVIVED",
                len(rec["failures"]),
                "; ".join(failing[:3]) + (" ..." if len(failing) > 3 else ""),
            ]
        )
    return table(
        ["Plant", "Defect", "Result", "Failing tests", "Failing (first three)"], rows
    )


# ---------------------------------------------------------------------------
# Section 5: one correction
# ---------------------------------------------------------------------------


def per_correction() -> str:
    """Item 6: iterations and time per correction, both preconditioners."""
    rows = []
    for r in measure()["rows"]:
        j, m = r["jacobi"], r["multigrid"]
        rows.append(
            [
                system_name(r),
                stopped_by(r),
                j["iterations"],
                ms(j["total_seconds"]),
                m["iterations"],
                ms(m["setup_seconds"]),
                ms(m["solve_seconds"]),
                ms(m["total_seconds"]),
                f"{100 * r['setup_share']:.1f}%",
                f"{r['speedup']:.1f}",
            ]
        )
    return table(
        [
            "System",
            "Stop set by",
            "Jacobi its",
            "Jacobi ms",
            "MG its",
            "MG setup ms",
            "MG solve ms",
            "MG total ms",
            "Setup share",
            "Jacobi / MG time",
        ],
        rows,
    )


def per_iteration() -> str:
    """Item 6: the cost of one iteration under each, and the two solutions' distance."""
    rows = []
    for r in measure()["rows"]:
        j, m = r["jacobi"], r["multigrid"]
        jit = j["solve_seconds"] / j["iterations"]
        mit = m["solve_seconds"] / m["iterations"]
        rows.append(
            [
                system_name(r),
                f"{1e3 * jit:.3f}",
                f"{1e3 * mit:.3f}",
                f"{mit / jit:.1f}",
                f"{j['iterations'] / m['iterations']:.1f}",
                f"{j['relative_residual']:.2e}",
                f"{m['relative_residual']:.2e}",
                f"{r['solution_difference']:.1e}",
            ]
        )
    return table(
        [
            "System",
            "Jacobi ms per it",
            "MG ms per it",
            "MG / Jacobi per it",
            "Jacobi / MG its",
            "Jacobi residual at exit",
            "MG residual at exit",
            "max |p'_MG - p'_J| / max |p'_J|",
        ],
        rows,
    )


def hierarchy() -> str:
    """Item 6: each system's levels, eigenvalue estimates and weights."""
    rows = []
    for r in measure()["rows"]:
        smoothed = r["levels"][:-1]
        eig = [lv["eigenvalue"] for lv in smoothed]
        weight = [lv["weight"] for lv in smoothed]
        rows.append(
            [
                system_name(r),
                " > ".join(
                    f"{s[1]}x{s[0]}" for s in (lv["shape"] for lv in r["levels"])
                ),
                f"{min(eig):.4f} to {max(eig):.4f}",
                f"{min(weight):.4f} to {max(weight):.4f}",
            ]
        )
    return table(
        [
            "System",
            "Levels (nx x ny), coarsest last",
            "lambda_max estimate",
            "Weight omega",
        ],
        rows,
    )


def passes() -> str:
    """Item 6 twice: each system's times in the two passes, an hour and a half apart."""
    first = {(r["case"], r["tag"]): r for r in measure()["rows"]}
    second = load(OUT / "measure53_pass2.json")["rows"]
    rows = []
    for r2 in second:
        r1 = first[r2["case"], r2["tag"]]
        rows.append(
            [
                system_name(r2),
                f"{ms(r1['jacobi']['total_seconds'])} / {ms(r2['jacobi']['total_seconds'])}",
                f"{ms(r1['multigrid']['total_seconds'])} / {ms(r2['multigrid']['total_seconds'])}",
                f"{100 * r1['setup_share']:.1f}% / {100 * r2['setup_share']:.1f}%",
                f"{r1['speedup']:.1f} / {r2['speedup']:.1f}",
            ]
        )
    return table(
        [
            "System",
            "Jacobi ms, pass 1 / 2",
            "MG ms, pass 1 / 2",
            "Setup share, pass 1 / 2",
            "Jacobi / MG time, pass 1 / 2",
        ],
        rows,
    )


def anatomy() -> str:
    """One cycle's time by level, beside one product and one Jacobi-PCG iteration."""
    rows = []
    for rec in load(OUT / "anatomy53.json"):
        name = {
            "annex_late": "Annex 20 216x72, outer 2,000",
            "step_late": "Step medium 656x60, outer 2,000",
        }[rec["system"]]
        rows.append(
            [name, "one product, finest", f"{1e3 * rec['product_seconds']:.3f}", ""]
        )
        rows.append(
            [
                "",
                "one Jacobi-PCG iteration",
                f"{1e3 * rec['jacobi_iteration_seconds']:.3f}",
                "",
            ]
        )
        rows.append(["", "one V-cycle", f"{1e3 * rec['cycle_seconds']:.3f}", ""])
        for level in rec["levels"]:
            ny, nx = level["shape"]
            rows.append(
                [
                    "",
                    f"of which level {nx}x{ny}",
                    f"{1e3 * level['own_seconds']:.3f}",
                    level["products"],
                ]
            )
    return table(["System", "Part", "ms", "Products in one cycle"], rows)


# ---------------------------------------------------------------------------
# Section 6: the whole runs
# ---------------------------------------------------------------------------


def step_jacobi_row() -> dict:
    """The step's clean Jacobi row of record from benchmarks/results.jsonl."""
    for line in RESULTS.read_text().splitlines():
        rec = json.loads(line)
        if rec["run_id"] == STEP_JACOBI_RUN_ID:
            return rec
    raise KeyError(STEP_JACOBI_RUN_ID)


def harness_summary(r: dict) -> dict:
    """Stop, count, time and pressure share of a harness row."""
    outer = r["work"]["outer_iterations"]
    stages = r["time"]["stages"]
    return {
        "stop": r["outcome"]["stop_reason"],
        "outer": outer,
        "wall": r["time"]["wall_seconds"],
        "per_outer": r["time"]["wall_seconds"] / outer,
        "pressure_per_outer": stages["pressure"] / outer,
        "pressure_share": stages["pressure"] / r["time"]["wall_seconds"],
        "its": r["work"]["inner_iterations"] / outer,
        "caps": r["outcome"]["pressure_cap_hits"],
        "concurrent": r["environment"]["concurrent_processes"],
        "commit": r["git_commit"][:7],
    }


def whole_rows() -> list[tuple[str, str, str, dict]]:
    """(run, preconditioner, record, summary) for every whole run present."""
    out = []
    step_j = step_jacobi_row()
    out.append(
        (
            "Step medium 656x60",
            "jacobi",
            "prompt 52 row of record, alone",
            harness_summary(step_j),
        )
    )
    step_mg = OUT / "row_val019_656x60_mg.json"
    if step_mg.exists():
        out.append(
            (
                "Step medium 656x60",
                "multigrid",
                "this branch, alone",
                harness_summary(load(step_mg)["row"]),
            )
        )
    annex51 = load(OUT51 / "annex_cg.json")
    out.append(
        (
            "Annex 20 216x72",
            "jacobi",
            "prompt 51 record, beside one other run",
            harness_summary(annex51["row"]),
        )
    )
    for label, pre in (("jacobi", "jacobi"), ("mg", "multigrid")):
        path = OUT / f"row_val017_216x72_{label}.json"
        if path.exists():
            out.append(
                (
                    "Annex 20 216x72",
                    pre,
                    "this branch, alone",
                    harness_summary(load(path)["row"]),
                )
            )
    for grid in GRIDS:
        rec = load(OUT51 / f"product_{grid}_cg.json")
        out.append(
            (
                f"Product {grid}",
                "jacobi",
                "prompt 51 record, four runs together",
                {
                    "stop": rec["stop"],
                    "outer": rec["outer"],
                    "wall": rec["wall_seconds"],
                    "per_outer": rec["seconds_per_outer"],
                    "pressure_per_outer": None,
                    "pressure_share": None,
                    "its": None,
                    "caps": None,
                },
            )
        )
        for label, pre in (("jacobi", "jacobi"), ("mg", "multigrid")):
            path = OUT / f"product_{grid}_{label}.json"
            if not path.exists():
                continue
            rec = load(path)
            stages = rec["stage_seconds"][0]
            out.append(
                (
                    f"Product {grid}",
                    pre,
                    "this branch, alone",
                    {
                        "stop": rec["stop"],
                        "outer": rec["outer"],
                        "wall": rec["wall_seconds"],
                        "per_outer": rec["seconds_per_outer"],
                        "pressure_per_outer": stages["pressure"] / rec["outer"],
                        "pressure_share": stages["pressure"] / rec["wall_seconds"],
                        "its": None,
                        "caps": rec["pressure_cap_hits"],
                    },
                )
            )
    for case in LAMINAR:
        for label, pre, where in (
            ("A", "jacobi", "commit A, beside five other runs"),
            ("mg", "multigrid", "this branch, alone"),
        ):
            path = OUT / f"val_{case}_{label}.json"
            if not path.exists():
                continue
            rec = load(path)
            stages = rec["stage_seconds"]
            out.append(
                (
                    case,
                    pre,
                    where,
                    {
                        "stop": rec["stop"],
                        "outer": rec["outer"],
                        "wall": rec["seconds"],
                        "per_outer": rec["seconds"] / rec["outer"],
                        "pressure_per_outer": stages["pressure"] / rec["outer"],
                        "pressure_share": stages["pressure"] / rec["seconds"],
                        "its": rec["pressure_iterations"] / rec["outer"],
                        "caps": rec["pressure_cap_hits"],
                    },
                )
            )
    return out


def whole() -> str:
    """Item 7: stop, count and time of every whole run."""
    rows = []
    for run, pre, where, s in whole_rows():
        rows.append(
            [
                run,
                pre,
                where,
                s["stop"],
                f"{s['outer']:,}",
                f"{s['wall']:,.0f}",
                f"{s['per_outer']:.4f}",
                ""
                if s["pressure_per_outer"] is None
                else f"{s['pressure_per_outer']:.4f}",
                ""
                if s["pressure_share"] is None
                else f"{100 * s['pressure_share']:.0f}%",
                "" if s["its"] is None else f"{s['its']:.1f}",
                "" if s["caps"] is None else s["caps"],
            ]
        )
    return table(
        [
            "Run",
            "Preconditioner",
            "Record",
            "Stop",
            "Outer",
            "Wall s",
            "s per outer",
            "Pressure s per outer",
            "Pressure share",
            "CG its per correction",
            "Capped corrections",
        ],
        rows,
    )


def gains() -> str:
    """Jacobi over multigrid, per outer iteration, in the pressure stage and per CG iteration."""
    pairs = {}
    for run, pre, where, summary in whole_rows():
        if where.startswith("prompt 51"):
            continue
        pairs.setdefault(run, {})[pre] = (where, summary)
    item6 = {
        "Step medium 656x60": "step",
        "Annex 20 216x72": "annex",
        "Product 160x60": "product160",
        "val001_80x40": "val001",
        "val002_80x80": "cavity",
    }
    first = {(r["case"], r["tag"]): r for r in measure()["rows"]}
    second = {
        (r["case"], r["tag"]): r for r in load(OUT / "measure53_pass2.json")["rows"]
    }
    rows = []
    for run, pair in pairs.items():
        if set(pair) != {"jacobi", "multigrid"}:
            continue
        (jw, j), (_mw, m) = pair["jacobi"], pair["multigrid"]

        def per_it(x: dict) -> str:
            if x["its"] is None:
                return ""
            return f"{1e3 * x['pressure_per_outer'] / x['its']:.3f}"

        case = item6.get(run)
        rows.append(
            [
                run,
                jw,
                f"{j['per_outer'] / m['per_outer']:.2f}",
                f"{j['pressure_per_outer'] / m['pressure_per_outer']:.2f}",
                per_it(j),
                per_it(m),
                ""
                if case is None
                else f"{first[case, 'late']['speedup']:.1f} / {second[case, 'late']['speedup']:.1f}",
            ]
        )
    return table(
        [
            "Run",
            "Jacobi record",
            "s per outer, Jacobi / MG",
            "Pressure stage, Jacobi / MG",
            "Jacobi ms per CG iteration in the run",
            "MG ms per CG iteration in the run",
            "Item 6, late system, Jacobi / MG time, pass 1 / 2",
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
    rerun = load(OUT / "row_val017_216x72_jacobi.json")
    diff_record = line_difference(mg["lines"], record["lines"])
    diff_rerun = line_difference(mg["lines"], rerun["lines"])
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
                f"{diff_record[key]:.2e}",
                f"{diff_rerun[key]:.2e}",
            ]
        )
    return table(
        [
            "Line",
            "Scored",
            "RMS / u0, Jacobi record",
            "RMS / u0, multigrid",
            "Allowance",
            "Multigrid",
            "Largest difference from the Jacobi record, u / u0",
            "From the Jacobi rerun",
        ],
        rows,
    )


def products() -> str:
    """The product rows: stop and faces against the Jacobi record and the rerun."""
    rows = []
    for grid in GRIDS:
        record = load(OUT51 / f"product_{grid}_cg.json")
        faces_record = np.load(OUT51 / f"product_{grid}_cg.npz")
        mg = load(OUT / f"product_{grid}_mg.json")
        faces_mg = np.load(OUT / f"product_{grid}_mg.npz")
        rerun = load(OUT / f"product_{grid}_jacobi.json")
        faces_rerun = np.load(OUT / f"product_{grid}_jacobi.npz")

        def largest(a: object, b: object) -> float:
            return max(
                float(np.abs(a["u_faces"] - b["u_faces"]).max()),
                float(np.abs(a["v_faces"] - b["v_faces"]).max()),
            )

        scale = mg["velocity_scale"]
        rows.append(
            [
                grid,
                f"{record['stop']} at {record['outer']:,}",
                f"{mg['stop']} at {mg['outer']:,}",
                f"{(mg['outer'] - record['outer']) / record['outer']:+.2%}",
                scale,
                f"{largest(faces_mg, faces_record) / scale:.2e}",
                "yes" if rerun["face_hash"] == record["face_hash"] else "no",
                f"{largest(faces_mg, faces_rerun) / scale:.2e}",
            ]
        )
    return table(
        [
            "Grid",
            "Jacobi record",
            "Multigrid",
            "Outer change",
            "Velocity scale, m/s",
            "Largest face difference over the scale, MG vs record",
            "Jacobi rerun bitwise the record",
            "MG vs rerun",
        ],
        rows,
    )


def val016() -> str:
    """VAL-016's split criteria under each preconditioner."""
    rows = []
    for label, rec in (
        ("jacobi, prompt 51 record", load(OUT51 / "val016_cg.json")),
        ("multigrid", load(OUT / "val016_mg.json")),
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


def step() -> str:
    """The step's answers: x_r and the other report-only figures, each preconditioner."""
    jacobi = step_jacobi_row()["accuracy"]["components"]
    mg = load(OUT / "row_val019_656x60_mg.json")["row"]["accuracy"]["components"]
    rows = []
    for key in jacobi:
        rows.append(
            [
                key,
                f"{jacobi[key]:.6g}",
                f"{mg[key]:.6g}",
                f"{mg[key] - jacobi[key]:+.3g}",
            ]
        )
    return table(["Figure", "Jacobi row of record", "Multigrid", "Difference"], rows)


def laminar_mg() -> str:
    """VAL-001, its twin and VAL-002 under multigrid against their criteria."""
    rows = []
    for case in LAMINAR:
        j = load(OUT / f"val_{case}_A.json")
        m = load(OUT / f"val_{case}_mg.json")
        comps = m["accuracy"].get("components")
        value = (
            ", ".join(f"{k} {v:.4e}" for k, v in comps.items())
            if comps
            else f"{m['accuracy']['value']:.4e}"
        )
        jcomps = j["accuracy"].get("components")
        jvalue = (
            ", ".join(f"{k} {v:.4e}" for k, v in jcomps.items())
            if jcomps
            else f"{j['accuracy']['value']:.4e}"
        )
        rows.append(
            [
                case,
                f"{j['stop']} at {j['outer']:,}",
                f"{m['stop']} at {m['outer']:,}",
                jvalue,
                value,
                f"< {m['criterion']}",
                "pass" if m["passes"] else "FAIL",
            ]
        )
    return table(
        [
            "Case",
            "Jacobi (commit A)",
            "Multigrid",
            "Jacobi value",
            "Multigrid value",
            "Criterion",
            "Multigrid",
        ],
        rows,
    )


# ---------------------------------------------------------------------------
# Section 7: the predictions
# ---------------------------------------------------------------------------


def predictions() -> str:
    """Each prediction against the records."""
    rows = []
    lam = [
        (load(OUT / f"val_{c}_base.json"), load(OUT / f"val_{c}_A.json"))
        for c in LAMINAR
    ]
    equal = sum(
        a["face_hash"] == b["face_hash"] and a["outer"] == b["outer"] for a, b in lam
    )
    rows.append(
        [
            "(a)",
            "Under the default every laminar hash and the pressure loop's results are the base's",
            f"{equal} of {len(lam)} laminar cases stop at the base's count with its face hash; "
            "the default correction equals the base loop to the bit on six test systems "
            "(`TestDefaultPath`); tests/test_pressure.py passes with the callable in "
            "place of the array",
            "holds" if equal == len(lam) else "FAILS",
        ]
    )
    m = measure()["rows"]
    its = [r["multigrid"]["iterations"] for r in m]
    rtol_its = [
        r["multigrid"]["iterations"] for r in m if stopped_by(r) == "pressure_rtol"
    ]
    by = {(r["case"], r["tag"]): r for r in m}
    growth_early = (
        by["product320", "early"]["multigrid"]["iterations"]
        / by["product160", "early"]["multigrid"]["iterations"]
    )
    growth_late = (
        by["product320", "late"]["multigrid"]["iterations"]
        / by["product160", "late"]["multigrid"]["iterations"]
    )
    in_range = all(10 <= i <= 40 for i in its)
    rows.append(
        [
            "(b)",
            "MG-PCG 10 to 40 iterations per correction on every captured system; under 2x growth from 160x60 to 320x120",
            f"{min(its)} to {max(its)} on all ten ({min(rtol_its)} to {max(rtol_its)} where pressure_rtol "
            f"sets the stop; the counts below 10 are floor-stopped converged states); growth "
            f"{growth_early:.2f}x early, {growth_late:.2f}x late (160x60's late system is "
            "floor-stopped, 320x120's is not)",
            (
                "holds"
                if in_range
                else "fails as worded, below 10 on two floor-stopped systems"
            )
            + (
                "; growth holds on the like-for-like early pair"
                if growth_early < 2
                else "; growth fails on the early pair"
            ),
        ]
    )
    both = [m, load(OUT / "measure53_pass2.json")["rows"]]
    shares = [r["setup_share"] for rows_ in both for r in rows_]
    over = sorted(
        {system_name(r) for rows_ in both for r in rows_ if r["setup_share"] >= 0.10}
    )
    rows.append(
        [
            "(c)",
            "Setup under 10% of a multigrid correction's time on every captured system",
            f"{100 * min(shares):.1f}% to {100 * max(shares):.1f}% over both passes; at or "
            f"above 10% in a pass on {len(over)} of {len(m)}: " + ", ".join(over),
            "holds" if not over else "fails",
        ]
    )
    speeds = [{(r["case"], r["tag"]): r["speedup"] for r in rows_} for rows_ in both]
    speed = {
        k: (
            min(s[k, "early"] for s in speeds),
            min(s[k, "late"] for s in speeds),
        )
        for k in ("step", "annex", "product320")
    }
    d_hold = (
        min(speed["step"]) >= 5
        and min(speed["annex"]) >= 3
        and min(speed["product320"]) >= 3
    )
    rows.append(
        [
            "(d)",
            "Correction time falls at least 5x on the step's medium grid, 3x on Annex 20 and the product's 320x120",
            "The smaller of the two passes: "
            + "; ".join(
                f"{name} {a:.1f}x early, {b:.1f}x late"
                for name, (a, b) in (
                    ("step", speed["step"]),
                    ("Annex 20", speed["annex"]),
                    ("320x120", speed["product320"]),
                )
            ),
            "holds"
            if d_hold
            else (
                "holds on the step and 320x120, fails on Annex 20"
                if min(speed["step"]) >= 5 and min(speed["product320"]) >= 3
                else "fails"
            ),
        ]
    )
    e_text, e_verdict = prediction_e()
    rows.append(
        [
            "(e)",
            "Whole runs reach the Jacobi records' stop within 5% of the outer count, answers within prompt 51's tolerances",
            e_text,
            e_verdict,
        ]
    )
    step_j = harness_summary(step_jacobi_row())
    step_path = OUT / "row_val019_656x60_mg.json"
    if step_path.exists():
        step_m = harness_summary(load(step_path)["row"])
        ratio = step_j["per_outer"] / step_m["per_outer"]
        rows.append(
            [
                "(f)",
                "The step's medium grid at least 3x faster per outer iteration alone",
                f"{step_j['per_outer']:.3f} s against {step_m['per_outer']:.3f} s, {ratio:.2f}x",
                "holds" if ratio >= 3 else "fails",
            ]
        )
    return table(["", "Prediction", "Measured", "Verdict"], rows)


def prediction_e() -> tuple[str, str]:
    """(e) from every whole run's record."""
    parts, ok = [], True
    mg = load(OUT / "row_val017_216x72_mg.json")
    rec = load(OUT51 / "annex_cg.json")
    a, b = mg["row"], rec["row"]
    change = (a["work"]["outer_iterations"] - b["work"]["outer_iterations"]) / b[
        "work"
    ]["outer_iterations"]
    lines = max(line_difference(mg["lines"], rec["lines"]).values())
    same = a["outcome"]["stop_reason"] == b["outcome"]["stop_reason"]
    ok &= same and abs(change) <= OUTER_TOL and lines <= LINE_TOL
    parts.append(f"Annex 20 outer {change:+.2%}, lines within {lines:.1e} u0")
    for grid in GRIDS:
        r51 = load(OUT51 / f"product_{grid}_cg.json")
        rmg = load(OUT / f"product_{grid}_mg.json")
        fa, fb = (
            np.load(OUT / f"product_{grid}_mg.npz"),
            np.load(OUT51 / f"product_{grid}_cg.npz"),
        )
        face = (
            max(float(np.abs(fa[k] - fb[k]).max()) for k in ("u_faces", "v_faces"))
            / rmg["velocity_scale"]
        )
        ch = (rmg["outer"] - r51["outer"]) / r51["outer"]
        ok &= rmg["stop"] == r51["stop"] and abs(ch) <= OUTER_TOL and face <= FACE_TOL
        parts.append(
            f"product {grid} outer {ch:+.2%}, faces within {face:.1e} of the scale"
        )
    sj = step_jacobi_row()
    smg = load(OUT / "row_val019_656x60_mg.json")["row"]
    dx = (
        smg["accuracy"]["components"]["x_r_over_h"]
        - sj["accuracy"]["components"]["x_r_over_h"]
    )
    ch = (smg["work"]["outer_iterations"] - sj["work"]["outer_iterations"]) / sj[
        "work"
    ]["outer_iterations"]
    ok &= (
        smg["outcome"]["stop_reason"] == sj["outcome"]["stop_reason"]
        and abs(ch) <= OUTER_TOL
        and abs(dx) <= XR_TOL
    )
    parts.append(f"step outer {ch:+.2%} (both at the cap), x_r {dx:+.3f} H")
    v51, vmg = load(OUT51 / "val016_cg.json"), load(OUT / "val016_mg.json")
    ok &= v51["stop"] == vmg["stop"] and v51["outer"] == vmg["outer"]
    parts.append(f"VAL-016 outer {vmg['outer']} against {v51['outer']}")
    return "; ".join(parts), "holds" if ok else "fails"


TABLES: dict[str, Callable[[], str]] = {
    "dev": dev,
    "laminar": laminar,
    "plants": plants,
    "per_correction": per_correction,
    "per_iteration": per_iteration,
    "hierarchy": hierarchy,
    "passes": passes,
    "anatomy": anatomy,
    "whole": whole,
    "gains": gains,
    "annex": annex,
    "products": products,
    "val016": val016,
    "step": step,
    "laminar_mg": laminar_mg,
    "predictions": predictions,
}


def main() -> None:
    """Replace every region whose records exist."""
    raw = REPORT.read_bytes().decode()
    newline = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    for name, build in TABLES.items():
        pattern = re.compile(
            rf"(<!-- BEGIN tables53: {name} -->\n).*?(<!-- END tables53: {name} -->)",
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
