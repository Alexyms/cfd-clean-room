"""Prompt 93b's measurements: the corrected field's scale, and the pseudo-inverses' criterion.

    python docs/reports/probe93/probe93b.py rest CASE CAP
    python docs/reports/probe93/probe93b.py captured
    python docs/reports/probe93/probe93b.py inverses
    python docs/reports/probe93/probe93b.py rotations

Run under WSL2 with the Linux environment from the checkout's root, one BLAS
thread, through ``results/builder93b/probe.sh``. The rules are in
``docs/reports/phase6_pressure.md`` (sections 3 and 4), committed before any run.

- ``rest``: one solve from rest of test 93's nineteen, run by the tester's own
  ``results/tester93/own.py`` (untracked, this machine), every correction given to
  the core beside the reference, each corrected component's largest difference
  read on both scales.
- ``captured``: the 50 captured corrections of ``probe93.py capture``.
- ``inverses``: every coarsest system of the criterion, the core's and NumPy's
  pseudo-inverses against the referee's (``tests/decimal_inverse.py``).
- ``rotations``: the referee against the tester's 50-digit Jacobi rotations on the
  systems of T6b.

Everything is kept in ``results/builder93b/`` (not committed).
"""

import argparse
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import probe93  # noqa: E402 -- after sys.path

from src import core_backend as core_mapping  # noqa: E402 -- after sys.path
from src import multigrid, pressure  # noqa: E402 -- after sys.path
from src.pressure import (  # noqa: E402 -- after sys.path
    PressureCorrection,
    PressureCorrector,
)
from tests.decimal_inverse import referee, residuals  # noqa: E402 -- after sys.path

RESULTS = ROOT / "results" / "builder93b"
TESTER = ROOT / "results" / "tester93"
COMPONENTS = ("u", "v")


# ---------------------------------------------------------------------------
# The corrected field


def corrections(
    corrector: PressureCorrector,
    a_p_u: np.ndarray,
    a_p_v: np.ndarray,
    p_prime: np.ndarray,
) -> dict[str, np.ndarray]:
    """What the reference subtracts from u* and v*: d times the difference of its p'."""
    d_u, d_v = corrector._face_d(a_p_u, a_p_v)
    padded = np.zeros((p_prime.shape[0] + 2, p_prime.shape[1] + 2))
    padded[1:-1, 1:-1] = p_prime
    return {
        "u": d_u * (padded[1:-1, 1:] - padded[1:-1, :-1]),
        "v": d_v * (padded[1:, 1:-1] - padded[:-1, 1:-1]),
    }


def measured(
    corrector: PressureCorrector,
    arguments: dict[str, np.ndarray],
    expected: PressureCorrection,
    got: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Each corrected component's difference, the three magnitudes and both measures."""
    prime = corrections(
        corrector, arguments["a_p_u"], arguments["a_p_v"], expected.p_prime
    )
    out = {}
    for name in COMPONENTS:
        want = getattr(expected, name)
        difference = float(np.abs(got[name] - want).max())
        star = float(np.abs(arguments[f"{name}_star"]).max())
        correction = float(np.abs(prime[name]).max())
        corrected = float(np.abs(want).max())
        out[name] = {
            "difference": difference,
            "star": star,
            "correction": correction,
            "corrected": corrected,
            "by_terms": difference / max(star, correction) if difference else 0.0,
            "by_field": difference / corrected if difference else 0.0,
        }
    for name in ("p", "p_prime"):
        out[name] = {"by_field": probe93.largest(got[name], getattr(expected, name))}
    return out


def rest(name: str, cap: int) -> None:
    """One of test 93's solves from rest, with both measures on every correction."""
    os.environ["TESTER_NO_RECORD"] = "1"
    sys.path.insert(0, str(TESTER))
    import own

    rows: list[dict[str, object]] = []

    class Measured(own.Beside):
        """The tester's wrapper, with the measure taken on each call's own arguments."""

        def install(self) -> None:
            super().install()
            checked = PressureCorrector.correct
            beside = self

            def correct(
                self: PressureCorrector, prediction: object, p: np.ndarray
            ) -> PressureCorrection:
                expected = checked(self, prediction, p)
                arguments = {
                    key: own.c64(getattr(prediction, key)) for key in probe93.ARGUMENTS
                }
                got = beside.cores[id(self)].correct(*arguments.values(), own.c64(p))
                rows.append(
                    {
                        "call": len(rows) + 1,
                        "iterations": [int(expected.iterations), got["iterations"]],
                        **measured(self, arguments, expected, got),
                    }
                )
                return expected

            PressureCorrector.correct = correct

    own.Beside = Measured
    # It keeps the first coarsest system where it keeps its records.
    own.OUT = RESULTS / "own"
    own.OUT.mkdir(parents=True, exist_ok=True)
    own.solve(name, cap)
    keep("rest", f"{name}_{cap}", rows)


def captured() -> None:
    """The 50 captured corrections, with both measures."""
    for name in probe93.CASES:
        pair = probe93.correctors(name)
        rows = []
        for index, call in enumerate(probe93.loaded(name)):
            expected = pair.reference.correct(probe93.prediction_of(call), call["p"])
            arguments = {key: call[key] for key in probe93.ARGUMENTS}
            got = pair.core.correct(*arguments.values(), call["p"])
            rows.append(
                {
                    "call": index + 1,
                    "iterations": [int(expected.iterations), got["iterations"]],
                    **measured(pair.reference, arguments, expected, got),
                }
            )
        keep("captured", name, rows)


def keep(kind: str, name: str, rows: list[dict[str, object]]) -> None:
    """Write one case's rows and print its worst by each measure."""
    folder = RESULTS / "fields"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{kind}_{name}.json").write_text(
        json.dumps({"kind": kind, "case": name, "rows": rows}, indent=1) + "\n",
        encoding="utf-8",
    )
    worst = {
        measure: max(row[c][measure] for row in rows for c in COMPONENTS)
        for measure in ("by_terms", "by_field")
    }
    beyond = {
        m: sum(max(r[c][m] for c in COMPONENTS) > 1e-12 for r in rows) for m in worst
    }
    print(
        f"MEASURED {kind} {name}: {len(rows)} corrections, worst {worst}, "
        f"beyond {beyond}"
    )


# ---------------------------------------------------------------------------
# The pseudo-inverses


def systems() -> Iterator[tuple[str, str, np.ndarray]]:
    """Every system of the criterion: its kind, its name and its matrix."""
    import _cfdcore

    levels = core_mapping.MULTIGRID_LEVELS
    for name, five in probe93.planted_systems():
        kind = "worse" if name.startswith("worse") else "planted"
        yield kind, name, _cfdcore.Multigrid2(*five, levels).coarsest_matrix()
    for name in probe93.CASES:
        pair = probe93.correctors(name)
        for index, call in enumerate(probe93.loaded(name)):
            c = pair.reference.coefficients(call["a_p_u"], call["a_p_v"])
            hierarchy = multigrid.Multigrid(c, pressure.apply_operator)
            yield "captured", f"{name} {index}", probe93.dense_matrix(hierarchy)
    for path in sorted((TESTER / "matrices").glob("*.npy")):
        yield "rest", path.stem, np.load(path)


def inverses() -> None:
    """The core's and NumPy's pseudo-inverse of every system against the referee's."""
    import _cfdcore

    rcond = core_mapping.PINV_RCOND
    rows = []
    for kind, name, matrix in systems():
        judged = referee(matrix, rcond)
        core = _cfdcore.pseudo_inverse(matrix, rcond)
        theirs = np.linalg.pinv(matrix, hermitian=True)
        apart = float(np.abs(core - theirs).max())
        rows.append(
            {
                "kind": kind,
                "name": name,
                "cells": len(matrix),
                "dropped": judged.dropped,
                "condition": judged.condition,
                "largest_dropped": judged.largest_dropped,
                "largest": float(np.abs(judged.inverse).max()),
                "bound": judged.bound(4.0),
                "core_error": float(np.abs(core - judged.inverse).max()),
                "numpy_error": float(np.abs(theirs - judged.inverse).max()),
                "between": apart,
                "between_bound": judged.bound(8.0),
                "between_over_numpy": apart / float(np.abs(theirs).max()),
                "residuals": residuals(matrix, judged, rcond),
            }
        )
    (RESULTS / "inverses.json").write_text(
        json.dumps(rows, indent=1) + "\n", encoding="utf-8"
    )
    for kind in ("planted", "worse", "captured", "rest"):
        mine = [r for r in rows if r["kind"] == kind]
        print(
            kind,
            len(mine),
            "core beyond",
            sum(r["core_error"] > r["bound"] for r in mine),
            "numpy beyond",
            sum(r["numpy_error"] > r["bound"] for r in mine),
            "between beyond",
            sum(r["between"] > r["between_bound"] for r in mine),
            "largest share of the bound: core, numpy, between",
            *(
                f"{max(r[e] / r[b] for r in mine):.2g}"
                for e, b in (
                    ("core_error", "bound"),
                    ("numpy_error", "bound"),
                    ("between", "between_bound"),
                )
            ),
            "beyond 1e-12 of NumPy's",
            sum(r["between_over_numpy"] > 1e-12 for r in mine),
            "worst residual",
            f"{max(max(r['residuals']) for r in mine):.1e}",
        )


def rotations() -> None:
    """The referee against the tester's Jacobi rotations on T6b's systems."""
    os.environ["TESTER_NO_RECORD"] = "1"
    sys.path.insert(0, str(TESTER))
    import own

    before = json.loads(
        (probe93.FIRST / "conditioning.json").read_text(encoding="utf-8")
    )
    names = {r["name"] for r in before if r["between"] > 1e-12}
    rcond = core_mapping.PINV_RCOND
    rows = []
    for _kind, name, matrix in systems():
        if name not in names:
            continue
        judged = referee(matrix, rcond)
        theirs, _values = own.decimal_inverse(matrix, rcond)
        rows.append({"name": name, "apart": probe93.largest(judged.inverse, theirs)})
        print(rows[-1], flush=True)
    (RESULTS / "rotations.json").write_text(
        json.dumps(rows, indent=1) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------


def main() -> None:
    """Run one command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("rest", "captured", "inverses", "rotations")
    )
    parser.add_argument("case", nargs="?")
    parser.add_argument("cap", nargs="?", type=int)
    args = parser.parse_args()
    commands = {
        "rest": lambda: rest(args.case, args.cap),
        "captured": captured,
        "inverses": inverses,
        "rotations": rotations,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
