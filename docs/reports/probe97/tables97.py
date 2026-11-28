"""The tables of prompt 97's report, from the records in results/builder97/.

    python docs/reports/probe97/tables97.py           # write tables97.md
    python docs/reports/probe97/tables97.py --check   # regenerate and compare, write nothing

Every number in ``docs/reports/phase6_end_to_end.md``'s results comes from here.
The records are the JSON files ``probe97.py`` writes: each run's record beside
its file under ``runs/<side>/``, each comparison and the band under
``compare/``, each core harness row and its record under ``rows/``, and each
timing session under ``timing/``; prompt 97b's sweep counts under ``sweeps/``
(each instrumented run's record, its step records ``<tag>.npz``, its flips
``<tag>.flips.json`` and ``analysis.json``). The tables read no run file, so they
are made on either platform. Both modes refuse an absent record, records that
name more than one commit unless no file the records were computed by
(``COMPUTED_BY``) differs between those commits, and sweep records of more than
one commit. Prompt 97b resumed prompt 97's runner at a later commit that changed
only the report and the runner's launch and added the sweep probe; the head of
the tables names every commit and the files that differ.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import probe97  # noqa: E402 -- follows sys.path.insert

RESULTS = probe97.RESULTS
SWEEPS = RESULTS / "sweeps"
SWEEP_TAGS = (
    "dust_320x120.base",
    "dust_320x120.var",
    "door_320x120.base",
    "dust_160x60.base",
    "dust_160x60.var",
)
TABLES = HERE / "tables97.md"
LABELS = {
    "clean": "clean-out",
    "door": "door leak",
    "dust": "dust release",
    "breach": "filter breach",
    "seal": "seal leak",
}
INSIDE = "inside the spread"
SPEEDUP = 8.0
PROBE_FLOW_SPREAD = 7.8e-10
# The paths a record's numbers depend on. Records of two commits are one set only
# when none of these differs between the commits. The sweep records are computed
# by sweeps97.py as well, which the later commit added; they must all name one
# commit, so that file is not compared across commits.
COMPUTED_BY = (
    "src/",
    "scripts/",
    "configs/",
    "core/",
    "docs/reports/probe85/",
    "docs/reports/probe97/probe97.py",
)
# A timing session made again, its first record kept beside it in T11 and not scored:
# a stray process held one logical processor through the first (Alex, 2026-11-27).
SUPERSEDED = probe97.RESULTS / "timing_superseded"
SUPERSEDED_SESSIONS = ((probe97.PRODUCT, 1),)
RELEASE_END = 40.0
FIELD_COLUMNS = (
    ("flow fields", None),
    ("concentration", "concentration, every"),
    ("concentration, pointwise", "concentration, pointwise"),
    ("sensor channels", "sensor channels"),
    ("leftover / supplied, change", "leftover"),
)


def load(path: Path) -> dict:
    """A JSON record."""
    return json.loads(path.read_text(encoding="utf-8"))


def e(value: object) -> str:
    """A difference: 0, inf, or two significant figures."""
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if value == 0.0:
        return "0"
    if math.isinf(value):
        return "inf"
    return f"{value:.1e}"


def number(value: object) -> float:
    """A record's number, its infinities written as text."""
    return float(value)


def yes(flag: bool) -> str:
    """yes or no."""
    return "yes" if flag else "no"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    """A markdown table."""
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out


def label_of(name: str) -> str:
    """A run's name as the report writes it."""
    scenario, grid = name.split("_", 1)
    return f"{LABELS[scenario]}, {grid}"


# ---------------------------------------------------------------------------
# The records


class Records:
    """Every record the tables read, checked to be present and to name one commit."""

    def __init__(self) -> None:
        self.missing: list[str] = []
        self.commits: dict[str, list[str]] = {}
        self.runs = {
            key: self._load(probe97.run_record(*key), "commit")
            for key in probe97.plan()
        }
        self.pairs = {
            p.label: self._load(probe97.compare_path(p.label), "commit")
            for p in probe97.pairs()
        }
        self.band = self._load(probe97.BAND_RECORD, "commit")
        self.placement = self._load(probe97.PLACEMENT_RECORD, "commit")
        self.rows = {}
        for case, method, stored in probe97.rows_of_record():
            path = probe97.row_path(case, method)
            record = self._load(path.with_suffix(".json"), "commit")
            if record is not None:
                self._note(record["core"]["commit"], f"{path.name} (the row)")
            self.rows[(case, method)] = (stored, record)
        self.timing = {
            key: self._session(probe97.timing_path(*key))
            for key in probe97.TIMING_SESSIONS
        }
        self.superseded = {
            key: self._session(SUPERSEDED / probe97.timing_path(*key).name)
            for key in SUPERSEDED_SESSIONS
        }
        self.sweeps = {
            tag: self._load(SWEEPS / f"{tag}.json", "commit") for tag in SWEEP_TAGS
        }
        self.analysis = self._load(SWEEPS / "analysis.json", "commit")
        if self.missing:
            raise SystemExit("absent records:\n  " + "\n  ".join(self.missing))
        swept = {rec["commit"] for rec in [*self.sweeps.values(), self.analysis]}
        if len(swept) != 1:
            raise SystemExit(f"the sweep records name more than one commit: {swept}")
        self.differing = differing_files(sorted(self.commits))
        if self.differing:
            lines = [f"{c[:12]}: {', '.join(n[:3])}" for c, n in self.commits.items()]
            raise SystemExit(
                "records of more than one commit, the code they were computed by "
                f"differing in {self.differing}:\n  " + "\n  ".join(lines)
            )

    def _session(self, folder: Path) -> tuple[dict | None, dict[str, dict | None]]:
        """A timing session's machine state and its two sides' records."""
        state = self._load(folder / "state.json", None)
        sides = {
            side: self._load(folder / f"{side}.json", "commit")
            for side in (probe97.REF, probe97.CORE)
        }
        return state, sides

    def _note(self, commit: str, name: str) -> None:
        self.commits.setdefault(commit, []).append(name)

    def _load(self, path: Path, key: str | None) -> dict | None:
        if not path.exists():
            self.missing.append(str(path.relative_to(ROOT)))
            return None
        record = load(path)
        if key is not None:
            self._note(record[key], str(path.relative_to(ROOT)))
        return record


def git_diff(first: str, other: str, paths: tuple[str, ...]) -> list[str]:
    """The files that differ between two commits, under PATHS (every file when empty)."""
    return subprocess.run(
        ["git", "diff", "--name-only", first, other, "--", *paths],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()


def differing_files(
    commits: list[str], paths: tuple[str, ...] = COMPUTED_BY
) -> list[str]:
    """The files under PATHS that differ between the first commit and any other."""
    out: list[str] = []
    for other in commits[1:]:
        out += [n for n in git_diff(commits[0], other, paths) if n not in out]
    return out


def items(pair: dict, criterion: str) -> list[dict]:
    """A comparison's items of one criterion."""
    return [i for i in pair["items"] if i["criterion"] == criterion]


def largest(pair: dict, prefix: str | None) -> tuple[float, list[str]]:
    """The largest difference of the field items PREFIX names (None: the flow fields), and their tiers."""
    values, tiers = [], []
    for i in items(pair, "field"):
        name = i["name"]
        flow = name.split(",")[0] in probe97.compare_runs.FLOW_FIELDS
        if (prefix is None and flow) or (
            prefix is not None and name.startswith(prefix)
        ):
            values.append(number(i["detail"]["difference"]))
            tiers.append(i["detail"]["tier"])
    return (max(values) if values else 0.0), tiers


# ---------------------------------------------------------------------------
# The tables


def runs_table(r: Records) -> list[str]:
    """T1: every run, where its flow states came from, its exit and its wall time."""
    rows = []
    for (side, name), source in probe97.plan().items():
        record = r.runs[(side, name)]
        later = record.get("outer_later")
        changed = "; ".join(
            f"{k} {old:g} to {new:g}" for k, old, new in record["changed"]
        )
        rows.append(
            [
                side,
                label_of(name),
                "solved" if source is None else f"kept from {source[0]}/{source[1]}",
                changed or "-",
                "-" if later is None else str(later["deferred"]),
                str(record["exit"]),
                f"{record['wall_seconds']:,.0f}",
                str(record["concurrent"]),
                record["finished"],
            ]
        )
    return [
        "### T1. The runs: each from rest through `run_scenario.main`, its flow states solved or kept from another run's file",
        "",
        *table(
            [
                "side",
                "run",
                "flow states",
                "changed",
                "flow stops deferred",
                "exit",
                "wall (s)",
                "solver processes beside",
                "finished",
            ],
            rows,
        ),
    ]


def band_table(r: Records) -> tuple[list[str], bool, bool]:
    """T2: the 320x120 spread of each field beside the probe's, and the limit."""
    band, limits, probe = r.band["band"], r.band["limits"], r.band["probe"]
    per = r.band["per_pair"]
    keys = ["u", *[k for k in band if k not in probe97.compare_runs.FLOW_FIELDS]]
    labels = {"u": "flow fields (u, v, p, k, eps, nu_t)"}
    rows, inside, flow_below = [], True, band["u"] < PROBE_FLOW_SPREAD
    pair_labels = [p.label for p in probe97.pairs() if p.kind == "band"]
    for key in keys:
        names = probe97.compare_runs.FLOW_FIELDS if key == "u" else (key,)
        cells = [e(max(per[label][n] for n in names)) for label in pair_labels]
        ok = band[key] <= limits[key]
        inside = inside and ok
        rows.append(
            [
                labels.get(key, key),
                *cells,
                e(band[key]),
                e(probe[key]),
                e(limits[key]),
                yes(ok),
            ]
        )
    heads = [
        label.replace("band_", "").replace("_320x120_", ", ") for label in pair_labels
    ]
    return (
        [
            "### T2. The band on 320x120: each field's largest difference from the reference under each stop variant, the 320x120 spread (the largest of the four), the probe's (P8; P2's for the sensor channels) and the limit",
            "",
            *table(
                [
                    "field",
                    *heads,
                    "320x120 spread",
                    "probe's spread",
                    "limit",
                    "within the limit",
                ],
                rows,
            ),
        ],
        inside,
        flow_below,
    )


def band_decisions_table(r: Records) -> list[str]:
    """T3: what the variants did to the decisions and the stops."""
    rows = []
    for pair in probe97.pairs():
        if pair.kind != "band":
            continue
        record = r.pairs[pair.label]
        moved = [i["name"] for i in items(record, "decision") if not i["passed"]]
        stops = [
            f"{i['detail']['ref outer']:,} to {i['detail']['other outer']:,}"
            for i in items(record, "stop")
        ]
        rows.append(
            [
                pair.other[0],
                label_of(pair.ref[1]),
                "; ".join(stops),
                ", ".join(moved) if moved else "none",
                yes(items(record, "closure")[0]["passed"]),
            ]
        )
    return [
        "### T3. The variants' flow stops (outer count, reference to variant) and the decisions they moved",
        "",
        *table(
            [
                "variant",
                "run",
                "outer counts per flow state",
                "decisions moved",
                "budget closes",
            ],
            rows,
        ),
    ]


def backend_pairs(r: Records) -> list[tuple[probe97.Pair, dict]]:
    """The seven pairs of the backends."""
    return [(p, r.pairs[p.label]) for p in probe97.pairs() if p.kind != "band"]


def stops_table(r: Records) -> tuple[list[str], bool]:
    """T4: every flow state's stop on each backend."""
    rows, equal = [], True
    for pair, record in backend_pairs(r):
        for i in items(record, "stop"):
            d = i["detail"]
            same = d["apart"] == 0 and d["ref reason"] == d["other reason"]
            same = same and d["ref capped"] == d["other capped"]
            equal = equal and same
            rows.append(
                [
                    label_of(pair.ref[1]),
                    i["name"].replace("flow state ", ""),
                    f"{d['ref outer']:,}",
                    f"{d['other outer']:,}",
                    str(d["apart"]),
                    d["ref reason"],
                    yes(d["ref reason"] == d["other reason"]),
                    yes(d["ref capped"] == d["other capped"]),
                ]
            )
    return (
        [
            "### T4. Every flow state's stop: the reference's outer count against the core's",
            "",
            *table(
                [
                    "run",
                    "state",
                    "reference outer",
                    "core outer",
                    "apart",
                    "reference's reason",
                    "same reason",
                    "same cap",
                ],
                rows,
            ),
        ],
        equal,
    )


def decisions_table(r: Records) -> tuple[list[str], bool]:
    """T5: every decision item of the seven pairs."""
    rows, equal = [], True
    for pair, record in backend_pairs(r):
        for i in items(record, "decision"):
            d = i["detail"]
            equal = equal and i["passed"]
            if "compared" in d:
                reached = (
                    f"{d['reached, ref']:,} / {d['reached, other']:,}"
                    if "reached, ref" in d
                    else "-"
                )
                moves = f"{d['moved one']} / {d['moved more']} / {d['one only']}"
                compared = f"{d['compared']:,}"
            else:
                reached, compared = "-", "-"
                moves = (
                    "equal"
                    if i["passed"]
                    else f"{d.get('ref')} against {d.get('other')}"
                )
            rows.append(
                [
                    label_of(pair.ref[1]),
                    i["name"],
                    compared,
                    reached,
                    moves,
                    yes(i["passed"]),
                ]
            )
    return (
        [
            "### T5. Every decision of the seven pairs: how many compared, how many each run reached, and the moves (by one step or frame / by more / reached in one run only)",
            "",
            *table(
                [
                    "run",
                    "decision",
                    "compared",
                    "reached, reference / core",
                    "moves",
                    "equal",
                ],
                rows,
            ),
        ],
        equal,
    )


def fields_table(r: Records) -> tuple[list[str], bool, bool]:
    """T6: each pair's largest difference per field, its tier, and the closure."""
    rows, inside, passed = [], True, True
    for pair, record in backend_pairs(r):
        cells = []
        for _, prefix in FIELD_COLUMNS:
            value, tiers = largest(record, prefix)
            worst = (
                INSIDE
                if all(t == INSIDE for t in tiers)
                else ", ".join(sorted({t for t in tiers if t != INSIDE}))
            )
            inside = inside and worst == INSIDE
            cells.append(e(value) if worst == INSIDE else f"{e(value)} ({worst})")
        closure = items(record, "closure")[0]
        passed = passed and record["verdict"] == "PASS"
        rows.append(
            [
                label_of(pair.ref[1]),
                "320x120's (T2)"
                if pair.kind.endswith(probe97.PRODUCT)
                else "the probe's",
                *cells,
                f"{e(number(closure['detail']['ref']))} / {e(number(closure['detail']['other']))}",
                record["verdict"],
            ]
        )
    return (
        [
            "### T6. Every field's largest difference from the reference (relative to the reference's largest magnitude), read against the spread named; a tier other than the first is written beside",
            "",
            *table(
                [
                    "run",
                    "spread",
                    *[c for c, _ in FIELD_COLUMNS],
                    "closure, reference / core",
                    "verdict",
                ],
                rows,
            ),
        ],
        inside,
        passed,
    )


def bitwise_table(r: Records) -> list[str]:
    """T7: which records of each pair are bit for bit."""
    keys: list[str] = []
    for _, record in backend_pairs(r):
        keys += [k for k in record["bitwise"] if k not in keys]
    rows = [
        [label_of(pair.ref[1]), *[yes(record["bitwise"].get(k, False)) for k in keys]]
        for pair, record in backend_pairs(r)
    ]
    return [
        "### T7. The records of each pair equal bit for bit",
        "",
        *table(["run", *keys], rows),
    ]


def placement_table(r: Records) -> tuple[list[str], bool]:
    """T8: the placement over the four contaminating scenarios on 320x120."""
    rows, equal = [], True
    for i in r.placement["items"]:
        d = i["detail"]
        decisive = "cells" in i["name"] or "order" in i["name"]
        if decisive:
            equal = equal and i["passed"]
        if "compared" in d:
            shown = f"{d['compared']} compared; {d['moved one']} / {d['moved more']} / {d['one only']} moved"
        elif "cells" in i["name"]:
            shown = (
                f"{len(d['ref'])} cells"
                if i["passed"]
                else f"{d['ref']} against {d['other']}"
            )
        elif "units" in i["name"]:
            shown = f"{len(d['ref'])} units"
        else:
            shown = (
                ", ".join(d["ref"])
                if i["passed"]
                else f"{d['ref']} against {d['other']}"
            )
        rows.append([i["name"], shown, yes(i["passed"])])
    return (
        [
            "### T8. The placement over the door leak, the dust release, the filter breach and the seal leak on 320x120, the reference's runs against the core's",
            "",
            *table(["item", "reference's", "equal"], rows),
            "",
            f"`compare_placement` took {r.placement['seconds']:,.0f} s.",
        ],
        equal,
    )


def rows_table(r: Records) -> tuple[list[str], bool]:
    """T9: every steady harness row of record against its run on the core."""
    rows, kept = [], True
    for (case, method), (stored, record) in sorted(r.rows.items()):
        s, c = record["stored"], record["core"]
        same = s["passed"] == c["passed"] and s["stop_reason"] == c["stop_reason"]
        kept = kept and same
        lines = probe97.row_path(case, method).read_text(encoding="utf-8").splitlines()
        cap = (
            stored["params"]["max_simple_iter"],
            json.loads(lines[-1])["params"]["max_simple_iter"],
        )
        rows.append(
            [
                case,
                method,
                f"{s['passed']!s} / {s['stop_reason']}",
                f"{c['passed']!s} / {c['stop_reason']}",
                yes(same),
                f"{s['outer']:,} / {c['outer']:,}",
                f"{cap[0]:,} / {cap[1]:,}",
                f"{s['value']:.10g} / {c['value']:.10g}",
                e(
                    abs(c["value"] - s["value"]) / abs(s["value"])
                    if s["value"]
                    else 0.0
                ),
                f"{s['wall_seconds']:,.0f} ({s['concurrent']}) / {c['wall_seconds']:,.0f} ({c['concurrent']})",
                s["commit"][:7],
            ]
        )
    return (
        [
            "### T9. The steady harness rows of record (the latest reference row per case and method) against the same row on the core",
            "",
            *table(
                [
                    "case",
                    "method",
                    "stored: passed / stop",
                    "core: passed / stop",
                    "verdict kept",
                    "outer, stored / core",
                    "max_simple_iter, stored / core",
                    "value, stored / core",
                    "value's relative change",
                    "wall s (processes), stored / core",
                    "stored at",
                ],
                rows,
            ),
            "",
            "The stored rows were made on Windows (`environment.os`), the core's under WSL2 on the same machine. A row whose `passed` is None is scored in the suite (VAL-001, VAL-002) or reported only (VAL-019, VAL-020). VAL-018's value is its outer count over `max_simple_iter`.",
        ],
        kept,
    )


def transient_table() -> list[str]:
    """T10: the tier-2 rows, which stay on the reference."""
    latest: dict[tuple[str, str], dict] = {}
    for line in probe97.HARNESS.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["case"].startswith(probe97.TRANSIENT_PREFIX):
            latest[(row["case"], row["method"])] = row
    rows = [
        [
            case,
            method,
            str(row["accuracy"].get("passed")),
            row["outcome"]["stop_reason"],
            row["git_commit"][:7],
        ]
        for (case, method), row in sorted(latest.items())
    ]
    return [
        "### T10. The tier-2 rows of record, which stay on the reference (ADR-017 F, option 1) and were not rerun",
        "",
        *table(["case", "method", "passed", "stop", "stored at"], rows),
    ]


def timing_table(r: Records) -> tuple[list[str], dict[str, list[float]]]:
    """T11: the dust release timed on each grid, its parts apart."""
    rows, ratios = [], {}
    sessions = [(key, value, False) for key, value in r.timing.items()]
    for key, value in r.superseded.items():
        at = next(i for i, (k, _, _) in enumerate(sessions) if k == key)
        sessions.insert(at, (key, value, True))
    for (grid, session), (state, sides), superseded in sessions:
        ref, core = sides[probe97.REF], sides[probe97.CORE]
        ratio = ref["total"] / core["total"]
        if not superseded:
            ratios.setdefault(grid, []).append(ratio)
        rows.append(
            [
                grid,
                f"{session}, superseded, not scored" if superseded else str(session),
                " then ".join(state["order"]),
                *[
                    f"{ref[k]:,.1f} / {core[k]:,.1f}"
                    for k in ("total", "flow", "march", "rest")
                ],
                f"{ratio:.2f}",
                f"{ref['flow'] / core['flow']:.2f}",
                f"{ref['march'] / core['march']:.2f}",
                "; ".join(state["before"]),
                "; ".join(state["after"]),
            ]
        )
    return (
        [
            "### T11. The dust release from rest on each grid, reference / core, seconds: the whole run, the flow solve, the march and the rest; the machine's state before and after each session. The superseded session ran with a stray process holding one logical processor; it was made again alone after that process was stopped, and is kept here, not scored",
            "",
            *table(
                [
                    "grid",
                    "session",
                    "order",
                    "total",
                    "flow",
                    "march",
                    "rest",
                    "reference / core",
                    "flow ratio",
                    "march ratio",
                    "Windows before (process % of one core; power; performance)",
                    "Windows after",
                ],
                rows,
            ),
        ],
        ratios,
    )


# ---------------------------------------------------------------------------
# Prompt 97b: the sweep counts, the proposed limits and the pairs scored under both


def decade_up(value: float) -> float:
    """The smallest power of ten at or above VALUE."""
    return 10.0 ** math.ceil(math.log10(value)) if value > 0.0 else 0.0


def band_rows(r: Records) -> list[tuple[str, str]]:
    """(label, key of the band record): the flow fields as one row, then each class field."""
    keys = [k for k in r.band["band"] if k not in probe97.compare_runs.FLOW_FIELDS]
    return [("flow fields (u, v, p, k, eps, nu_t)", "u"), *((k, k) for k in keys)]


def proposed_limits(r: Records) -> dict[str, tuple[float, float, float]]:
    """Per band key: ADR-017 H's rule on the 320x120 spread, the current limit, the proposal.

    The rule is the largest measured times ten, rounded up to a decade. The
    proposal is the larger of the rule and the current limit: on the fields where
    the 320x120 spread is smaller than the probe's, the probe's grids still set it.
    """
    out = {}
    band, limits = r.band["band"], r.band["limits"]
    for key in band:
        rule = decade_up(10.0 * number(band[key]))
        out[key] = (rule, number(limits[key]), max(rule, number(limits[key])))
    return out


def rescore(record: dict, limits: dict[str, float]) -> tuple[bool, list[str]]:
    """A pair's verdict with its field items read against LIMITS, the spreads its own.

    As ``compare_runs._field`` reads a field: inside the spread passes; between
    the spread and the limit passes only for a flow field whose stops explain it
    (some flow state's outer count one apart, none more); beyond fails. The
    decisions, stops and closure keep their record's verdict. Returns the verdict
    and the field items whose tier differs from the record's.
    """
    aparts = [i["detail"].get("apart", math.inf) for i in items(record, "stop")]
    explained = bool(aparts) and max(aparts) == 1
    passed, moved = True, []
    for i in record["items"]:
        if i["criterion"] != "field":
            passed = passed and i["passed"]
            continue
        d = i["detail"]
        key = probe97.band_key(i["name"])
        value, spread, limit = number(d["difference"]), number(d["spread"]), limits[key]
        if value <= spread:
            level = probe97.compare_runs.INSIDE
        elif value <= limit:
            level = probe97.compare_runs.BETWEEN
        else:
            level = probe97.compare_runs.BEYOND
        flow = key in probe97.compare_runs.FLOW_FIELDS
        ok = level == probe97.compare_runs.INSIDE or (
            level == probe97.compare_runs.BETWEEN and explained and flow
        )
        passed = passed and ok
        if level != d["tier"]:
            moved.append(f"{i['name']}: {d['tier']} to {level}")
    return passed, moved


def within(record: dict, limits: dict[str, float]) -> bool:
    """Whether every field item of a pair is at or below its limit in LIMITS."""
    return all(
        number(i["detail"]["difference"]) <= limits[probe97.band_key(i["name"])]
        for i in record["items"]
        if i["criterion"] == "field"
    )


def two_limits_table(r: Records) -> tuple[list[str], bool]:
    """T12: every pair's verdict under the current limits and under the proposed ones."""
    proposal = {k: v[2] for k, v in proposed_limits(r).items()}
    current_limits = {k: number(v) for k, v in r.band["limits"].items()}
    rows, same = [], True
    for pair in probe97.pairs():
        record = r.pairs[pair.label]
        ok, moved = rescore(record, proposal)
        if pair.kind == "band":
            # A band pair measures the spread: the question is only whether each
            # field is within its limit, which (a) reads under the current limits.
            current_text = f"within: {yes(within(record, current_limits))}"
            proposed_text = f"within: {yes(within(record, proposal))}"
        else:
            current_text = record["verdict"]
            proposed_text = "PASS" if ok else "FAIL"
            same = same and ok == (record["verdict"] == "PASS")
        rows.append(
            [
                pair.label,
                current_text,
                proposed_text,
                "; ".join(moved) if moved else "none",
            ]
        )
    return (
        [
            "### T12. Every pair under the current limits (the record's, kept by Alex on 2026-11-27) and under the rule's limits (T19, put to Alex and not adopted), each field read against the same spread: a pair of the backends by its verdict, a band pair by whether every field is within its limit",
            "",
            *table(
                [
                    "pair",
                    "current limits",
                    "the rule's limits (not adopted)",
                    "tiers that change",
                ],
                rows,
            ),
        ],
        same,
    )


def sweep_runs_table(r: Records) -> list[str]:
    """T13: the instrumented runs."""
    rows = []
    for tag in SWEEP_TAGS:
        rec = r.sweeps[tag]
        rows.append(
            [
                tag,
                f"{rec['diffusion_tol']:g}",
                rec["flows_from"].replace(str(Path.home()), "~"),
                rec["of_record"] or "none",
                f"{rec['steps']:,}",
                f"{sum(rec['class_steps'].values()):,}",
                f"{rec['flips']:,}" if rec["flips_from"] else "-",
                str(rec["exit"]),
                f"{rec['wall_seconds']:,.0f}",
                str(rec["concurrent"]),
            ]
        )
    return [
        "### T13. The instrumented runs (`sweeps97.py`): the reference's march with its flow states read from a run file, the flips measured on a base run against its variant's counts",
        "",
        *table(
            [
                "run",
                "diffusion_tol",
                "flow states from",
                "run of record",
                "steps",
                "class-steps",
                "flips measured",
                "exit",
                "wall (s)",
                "solver processes beside",
            ],
            rows,
        ),
    ]


def sweep_bitwise_table(r: Records) -> tuple[list[str], bool]:
    """T14: (f), each instrumented run against its run of record."""
    a = r.analysis
    keys: list[str] = []
    for v in a["bitwise"].values():
        keys += [k for k in v["flags"] if k not in keys]
    rows = [
        [
            tag,
            v["of_record"],
            *[yes(v["flags"].get(k, False)) for k in keys],
            v["verdict"],
        ]
        for tag, v in a["bitwise"].items()
    ]
    return (
        [
            "### T14. Each instrumented run against its run of record, bit for bit per record (prediction (f) reads the frames, the budgets and the sensors)",
            "",
            *table(["run", "run of record", *keys, "comparator"], rows),
        ],
        bool(a["f_holds"]),
    )


def t_or_dash(value: object) -> str:
    """A time in seconds, or a dash."""
    return "-" if value is None else f"{number(value):.3f}"


def sweep_counts_table(r: Records) -> list[str]:
    """T15: the class-steps whose count differs, by grid and class, and (g)."""
    rows = []
    for grid, g in r.analysis["grids"].items():
        for k, c in g["classes"].items():
            rows.append(
                [
                    grid,
                    k,
                    f"{c['class_steps']:,}",
                    f"{c['differing']:,}",
                    f"{c['share']:.3g}",
                    f"{c['variant_more']:,} / {c['variant_fewer']:,}",
                    str(c["largest_difference"]),
                    t_or_dash(c["first_flip_t"]),
                    t_or_dash(c["first_field_t"]),
                    "-"
                    if c["first_frame_step"] is None
                    else f"{c['first_frame_step']:,}",
                    "-"
                    if c["first_flip_step"] is None
                    else f"{c['first_flip_step']:,}",
                    f"{c['fields_differ_without_a_flip']:,}",
                    yes(c["g_holds"]),
                ]
            )
        rows.append(
            [
                grid,
                "all",
                f"{g['class_steps']:,}",
                f"{g['differing']:,}",
                f"{g['share']:.3g}",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )
    return [
        "### T15. The class-steps whose Jacobi count differs between the dust release as it is and with `diffusion_tol` times 0.99, by grid and class; where the count and the field first differ",
        "",
        *table(
            [
                "grid",
                "class",
                "class-steps",
                "count differs",
                "share",
                "variant more / fewer",
                "largest difference",
                "first count differs, t (s)",
                "first field differs, t (s)",
                "first stored frame differs, step",
                "first count differs, step",
                "steps whose field differs with the count equal",
                "(g)",
            ],
            rows,
        ),
        "",
        "A step's field is compared by a digest of the field the step returns; the stored frames by their bytes.",
    ]


def sweep_time_table(r: Records) -> list[str]:
    """T16: the flips by time, per grid."""
    grids = list(r.analysis["grids"])
    bins = r.analysis["grids"][grids[0]]["classes"]["0"]["by_time"]
    rows = []
    for n, b in enumerate(bins):
        cells = [f"{b['from']:g} to {b['to']:g}"]
        for grid in grids:
            g = r.analysis["grids"][grid]
            total = sum(c["by_time"][n]["class_steps"] for c in g["classes"].values())
            marked = sum(c["by_time"][n]["marked"] for c in g["classes"].values())
            cells += [
                f"{total:,}",
                f"{marked:,}",
                f"{marked / total:.3g}" if total else "-",
            ]
        rows.append(cells)
    head = ["t (s)"]
    for grid in grids:
        head += [f"{grid} class-steps", f"{grid} count differs", f"{grid} share"]
    return [
        f"### T16. The class-steps whose count differs, by time (the release ends at {RELEASE_END:g} s)",
        "",
        *table(head, rows),
    ]


def flip_effect_table(r: Records) -> list[str]:
    """T17a: the field difference one sweep more or fewer makes at a flipped step."""
    rows = []
    for grid, g in r.analysis["grids"].items():
        fe = g["flip_effect"]
        for label in ("more", "fewer"):
            m = fe[label]
            rows.append(
                [
                    grid,
                    f"one {label}",
                    f"{m['relative'].get('n', 0):,}",
                    yes(fe["replica_equal"]),
                    *[
                        " / ".join(
                            e(m[measure].get(q)) for q in ("median", "q90", "max")
                        )
                        for measure in (
                            "relative",
                            "pointwise",
                            "leftover_change_over_supply",
                        )
                    ],
                ]
            )
    return [
        "### T17a. One flip's effect: at every flipped class-step of the base run, the same step from the same state with one sweep more and one fewer, against the step as it ran (median / 90th percentile / largest)",
        "",
        *table(
            [
                "grid",
                "sweeps",
                "flips",
                "the replayed step equals the run's",
                "field, relative to its largest",
                "pointwise (cells at a hundredth of the threshold or more)",
                "leftover's change over the supply",
            ],
            rows,
        ),
    ]


def flip_sample_table(r: Records) -> list[str]:
    """T17b: a sample of the 320x120 flips, spread over the run."""
    g = r.analysis["grids"][probe97.PRODUCT]
    rows = []
    for f in g["flip_effect"]["sample"]:
        cells = [
            f"{f['t']:.3f}",
            str(f["k"]),
            f"{f['sweeps']} / {f['variant_sweeps']}",
            f["cause"],
            f"{f['ratio']:.5f}",
        ]
        for label in ("more", "fewer"):
            m = f[label]
            cells.append(
                "-"
                if m is None
                else f"{e(m['relative'])} / {e(m['pointwise'])} / {e(m['leftover_change_over_supply'])}"
            )
        rows.append(cells)
    return [
        "### T17b. A sample of the 320x120 flips, evenly spaced over the run's flips: one sweep more and one fewer (field relative / pointwise / leftover's change over the supply)",
        "",
        *table(
            [
                "t (s)",
                "class",
                "sweeps, base / variant",
                "base stopped by",
                "stopping residual / threshold",
                "one more",
                "one fewer",
            ],
            rows,
        ),
    ]


def distribution_table(r: Records) -> list[str]:
    """T18: the stopping residual's distribution before and after the release ends."""
    rows = []
    for tag in SWEEP_TAGS:
        for period, d in r.analysis["distributions"][tag].items():
            c = d["causes"]
            ratio, sweeps, rho = d["ratio"], d["sweeps"], d["contraction"]
            rows.append(
                [
                    tag,
                    period,
                    f"{d['class_steps']:,}",
                    f"{c['max-norm']:,} / {c['leftover bound']:,} / {c['cap']:,}",
                    " / ".join(
                        f"{ratio.get(q, 0.0):.4f}"
                        for q in ("q10", "median", "q90", "max")
                    ),
                    f"{d['above_flip_ratio_by_max_norm']:,} / {d['above_flip_ratio']:,}",
                    f"{d['above_flip_ratio'] / d['read']:.4f}" if d["read"] else "-",
                    " / ".join(f"{sweeps.get(q, 0.0):g}" for q in ("median", "max")),
                    f"{rho.get('median', 0.0):.4f}",
                    e(d["leftover_ratio"].get("median")),
                ]
            )
    return [
        f"### T18. The stopping residual over the max-norm threshold (`diffusion_tol` times the largest right-hand side), per run, to {RELEASE_END:g} s and after; the stops a 1% tighter tolerance fails from the same state are those above {probe97.DIFFUSION_FACTOR}",
        "",
        *table(
            [
                "run",
                "period",
                "class-steps",
                "stopped by max-norm / leftover's bound / cap",
                "residual / threshold, 10% / median / 90% / largest",
                f"above {probe97.DIFFUSION_FACTOR}: stopped by max-norm / all",
                f"share above {probe97.DIFFUSION_FACTOR}",
                "sweeps, median / largest",
                "contraction per sweep, median",
                "leftover / its bound, median",
            ],
            rows,
        ),
        "",
        "The contraction per sweep is (stopping residual / first residual) to the power one over the sweeps. Class-steps before the release, where the field is zero and so is the threshold, are counted but not read.",
    ]


def limits_table(r: Records) -> list[str]:
    """T19: the 320x120 limits ADR-017 H's rule gives, proposed and not applied."""
    proposal = proposed_limits(r)
    g = r.analysis["grids"][probe97.PRODUCT]
    one = g["flip_effect"]["more"]
    per_flip = {
        probe97.compare_runs.CONCENTRATION: one["relative"].get("max"),
        probe97.compare_runs.POINTWISE: one["pointwise"].get("max"),
        probe97.compare_runs.LEFTOVER: one["leftover_change_over_supply"].get("max"),
    }
    rows = []
    for label, key in band_rows(r):
        rule, current, proposed = proposal[key]
        rows.append(
            [
                label,
                e(number(r.band["probe"][key])),
                e(number(r.band["band"][key])),
                e(current),
                e(rule),
                e(proposed),
                e(per_flip.get(key)),
            ]
        )
    return [
        "### T19. The 320x120 limits by ADR-017 H's rule (the largest measured times ten, rounded up to a decade), put to Alex and not adopted; the largest effect of one sweep more at a flipped 320x120 step beside",
        "",
        *table(
            [
                "field",
                "probe's spread",
                "320x120 spread",
                "current limit",
                "the rule on the 320x120 spread",
                "the rule's reading (the larger of the two)",
                "one flip, largest (T17a)",
            ],
            rows,
        ),
        "",
        f"On 320x120 the dust release's variant flipped {g['differing']:,} class-steps (T15).",
    ]


BAND_DUST = f"band_dust_{probe97.PRODUCT}_diffusion_tol_{probe97.DIFFUSION_FACTOR}"
BUDGET_ITEM = "every class's budget verdict against REQ-T05"
NEAR = (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12)
# A stop's cause as sweeps97.py stores it: the index into this tuple.
CAUSES = ("max-norm", "leftover bound", "cap", "no solve")


def leftover_sum_table(r: Records) -> list[str]:
    """T20: the variant's change of each class's leftover against the sum of its flips."""
    pair = r.pairs[BAND_DUST]
    detail = next(i["detail"] for i in pair["items"] if i["name"] == BUDGET_ITEM)
    ref, other = detail["ref leftover / supplied"], detail["other leftover / supplied"]
    flips = load(SWEEPS / f"dust_{probe97.PRODUCT}.base.flips.json")
    rows = []
    for k, (a, b) in enumerate(zip(ref, other, strict=True)):
        mine = [f for f in flips if f["k"] == k]
        more = math.fsum(f["more"]["leftover_change_over_supply"] for f in mine)
        fewer = math.fsum(f["fewer"]["leftover_change_over_supply"] for f in mine)
        change = number(b) - number(a)
        rows.append(
            [
                str(k),
                f"{len(mine):,}",
                e(change),
                e(more),
                f"{more / change:.6f}" if change else "-",
                e(fewer),
            ]
        )
    return [
        f"### T20. The dust release on {probe97.PRODUCT}: each class's leftover over the supply at the end, the variant's change against the sum over the class's flipped steps of one sweep more's change (and one fewer's)",
        "",
        *table(
            [
                "class",
                "flips",
                "variant's change at the end (the band pair)",
                "sum of one more's change",
                "sum / change",
                "sum of one fewer's change",
            ],
            rows,
        ),
        "",
        "One sweep more is computed from the base run's state at each flipped step; the variant's state differs from the base's after its first flip.",
    ]


def near_threshold_table() -> list[str]:
    """T21: the base runs' stops by their distance below the max-norm threshold."""
    rows = []
    for tag in ("dust_320x120.base", "door_320x120.base", "dust_160x60.base"):
        z = np.load(SWEEPS / f"{tag}.npz")
        classes = sorted({name.split("_")[-1] for name in z.files if "_" in name})
        gaps = []
        for k in classes:
            ratio, cause = z[f"ratio_{k}"], z[f"cause_{k}"]
            keep = np.isfinite(ratio) & (cause != CAUSES.index("cap"))
            gaps.append(1.0 - ratio[keep])
        gap = np.concatenate(gaps)
        rows.append(
            [
                tag,
                f"{gap.size:,}",
                *[f"{int((gap <= eps).sum()):,}" for eps in NEAR],
                e(float(gap.min())),
            ]
        )
    return [
        "### T21. The base runs' stops (every class-step whose threshold is not zero and that did not stop at the cap) by how far the stopping residual lies below the max-norm threshold: the count within each relative distance, and the closest",
        "",
        *table(
            [
                "run",
                "stops read",
                *[f"within {eps:.0e}" for eps in NEAR],
                "closest",
            ],
            rows,
        ),
        "",
        f"A threshold times {probe97.DIFFUSION_FACTOR} fails the stops within 1e-02 from the same state. A port whose residual differs from the reference's by rounding fails, or passes, only stops within that rounding of the threshold.",
    ]


def options_table(r: Records) -> list[str]:
    """T22: readings of the 320x120 band, for Alex: by the rule and per flipped stop."""
    proposal = proposed_limits(r)
    g = r.analysis["grids"][probe97.PRODUCT]
    fe = g["flip_effect"]
    measures = {
        probe97.compare_runs.CONCENTRATION: "relative",
        probe97.compare_runs.POINTWISE: "pointwise",
        probe97.compare_runs.LEFTOVER: "leftover_change_over_supply",
    }
    most = max(c["differing"] for c in g["classes"].values())
    rows = []
    for label, key in band_rows(r):
        if key not in measures:
            continue
        rule, current, _ = proposal[key]
        one = max(number(fe[s][measures[key]]["max"]) for s in ("more", "fewer"))
        per_flip = decade_up(10.0 * one)
        rows.append(
            [
                label,
                e(current),
                e(rule),
                e(one),
                e(per_flip),
                e(per_flip * most),
                e(number(r.band["band"][key])),
            ]
        )
    return [
        f"### T22. Readings of the {probe97.PRODUCT} band put to Alex: ADR-017 H's rule on the variant's spread (T19, not adopted), and a limit per flipped stop, Alex's decision of 2026-11-27, which keeps the current limits (ten times the largest effect of one sweep more or fewer, rounded up to a decade), times the variant's flips in its most-flipped class ({most:,})",
        "",
        *table(
            [
                "field",
                "current limit",
                "the rule on the 320x120 spread",
                "one flip, largest of more and fewer (T17a)",
                "per flipped stop",
                f"per flipped stop times {most:,}",
                "320x120 spread",
            ],
            rows,
        ),
    ]


def predictions_table(scores: dict[str, tuple[str, str]]) -> list[str]:
    """T0: the predictions scored by the rule committed with them."""
    return [
        "### T0. The predictions, scored by the rule committed with them",
        "",
        *table(
            ["prediction", "measured", "verdict"],
            [[k, measured, verdict] for k, (measured, verdict) in scores.items()],
        ),
    ]


def build() -> str:
    """Every table, from the records."""
    r = Records()
    t1 = runs_table(r)
    t2, band_inside, flow_below = band_table(r)
    t3 = band_decisions_table(r)
    t4, stops_equal = stops_table(r)
    t5, decisions_equal = decisions_table(r)
    t6, fields_inside, verdicts_pass = fields_table(r)
    t7 = bitwise_table(r)
    t8, placement_equal = placement_table(r)
    t9, rows_kept = rows_table(r)
    t10 = transient_table()
    t11, ratios = timing_table(r)
    product = ratios[probe97.PRODUCT]
    closes = all(items(p, "closure")[0]["passed"] for _, p in backend_pairs(r))
    b_holds = stops_equal and decisions_equal and fields_inside and closes
    scores = {
        "a": (
            f"every spread within its limit: {yes(band_inside)}; flow spread {e(r.band['band']['u'])} below 7.8e-10: {yes(flow_below)} (T2)",
            "HOLDS" if band_inside and flow_below else "FAILS",
        ),
        "b": (
            f"outer counts equal: {yes(stops_equal)} (T4); decisions equal: {yes(decisions_equal)} (T5); every field inside its spread: {yes(fields_inside)}, closure: {yes(closes)} (T6)",
            "HOLDS" if b_holds else "FAILS",
        ),
        "c": (
            f"searched cells and order equal under both rules: {yes(placement_equal)} (T8)",
            "HOLDS" if placement_equal else "FAILS",
        ),
        "d": (
            f"{sum(1 for _ in r.rows)} rows, every verdict kept: {yes(rows_kept)} (T9)",
            "HOLDS" if rows_kept else "FAILS",
        ),
        "e": (
            "320x120 reference / core "
            + ", ".join(f"{x:.2f}" for x in product)
            + " (T11)",
            "HOLDS" if product and min(product) >= SPEEDUP else "FAILS",
        ),
    }
    t12, proposal_same = two_limits_table(r)
    t13 = sweep_runs_table(r)
    t14, f_holds = sweep_bitwise_table(r)
    t15 = sweep_counts_table(r)
    t16 = sweep_time_table(r)
    t17a = flip_effect_table(r)
    t17b = flip_sample_table(r)
    t18 = distribution_table(r)
    t19 = limits_table(r)
    t20 = leftover_sum_table(r)
    t21 = near_threshold_table()
    t22 = options_table(r)
    grids = r.analysis["grids"]
    shares = ", ".join(f"{grid} {g['share']:.3g}" for grid, g in grids.items())
    late = [
        f"{grid} class {k} {t_or_dash(c['first_flip_t'])} s / {t_or_dash(c['first_field_t'])} s"
        for grid, g in grids.items()
        for k, c in g["classes"].items()
        if not c["g_holds"]
    ]
    scores["f"] = (
        "every instrumented run with a run of record equals it bit for bit in its frames, budgets and sensors: "
        f"{yes(f_holds)} (T14)",
        "HOLDS" if f_holds else "FAILS",
    )
    scores["g"] = (
        "every class's first differing count at or before its first differing stored frame, on both grids: "
        f"{yes(r.analysis['g_holds'])}"
        + (f"; not: {', '.join(late)}" if late else "")
        + " (T15)",
        "HOLDS" if r.analysis["g_holds"] else "FAILS",
    )
    scores["h"] = (
        f"share of class-steps whose count differs: {shares} (T15)",
        "HOLDS" if r.analysis["h_holds"] else "FAILS",
    )
    commits = sorted(r.commits, key=lambda c: -len(r.commits[c]))
    others = differing_files(commits, ()) if len(commits) > 1 else []
    parts = [f"{c[:12]} ({len(r.commits[c])} records)" for c in commits]
    made = ", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else parts[0]
    head = [
        "# Prompt 97's tables (generated by `tables97.py` from `results/builder97/`; do not edit)",
        "",
        f"The records were made at {made}."
        + (
            f" Between the commits these files differ, none of them a file the records were computed by: {', '.join(others)}."
            if others
            else ""
        )
        + f" Every pair of the backends passes the comparator: {yes(verdicts_pass)}."
        + f" The rule's limits (T19, not adopted) would change no pair's verdict: {yes(proposal_same)} (T12).",
        "",
    ]
    blocks = [
        head,
        predictions_table(scores),
        t1,
        t2,
        t3,
        t4,
        t5,
        t6,
        t7,
        t8,
        t9,
        t10,
        t11,
        t12,
        t13,
        t14,
        t15,
        t16,
        t17a,
        t17b,
        t18,
        t19,
        t20,
        t21,
        t22,
    ]
    return "\n".join(line for block in blocks for line in [*block, ""]).rstrip() + "\n"


def main() -> int:
    """Write tables97.md, or with --check compare it with the records' tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = build()
    if args.check:
        same = TABLES.read_text(encoding="utf-8").replace("\r\n", "\n") == text
        print(
            "tables97.md matches the records"
            if same
            else "tables97.md DIFFERS from the records"
        )
        return 0 if same else 1
    TABLES.write_text(text, encoding="utf-8")
    print("wrote", TABLES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
