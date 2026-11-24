"""Prompt 93's probe: the reference's own pressure corrections, captured and replayed.

    python docs/reports/probe93/probe93.py capture product80
    python docs/reports/probe93/probe93.py capture vents80
    python docs/reports/probe93/probe93.py capture cavity40
    python docs/reports/probe93/probe93.py capture annex216
    python docs/reports/probe93/probe93.py capture backstep
    python docs/reports/probe93/probe93.py replay
    python docs/reports/probe93/probe93.py timing --run 1
    python docs/reports/probe93/probe93.py conditioning
    python docs/reports/probe93/probe93.py vecinfo
    python docs/reports/probe93/probe93.py plants

Run under WSL2 with the Linux environment (``~/.venvs/cfd_clean_room``) from
the checkout's root, one BLAS thread, through ``results/builder93/probe.sh``.
Nothing in ``src/`` changes: the probe wraps ``PressureCorrector.correct`` on
the class, and the wrapper calls the reference and keeps a copy of what it was
handed and what it returned.

- ``capture``: a steady solve warm-started (``solve_steady(initial=...)``) and
  stopped after its first ``CALLS`` outer iterations, every correction kept.
  The 80x30 product room starts from prompt 81's stored state, the vents room
  from prompt 90's, the cavity from the state its solve from rest reaches
  after ``CAVITY_OUTER`` outer iterations (at its stop the right-hand side is
  rounding and a correction takes no iteration), the Annex 20 room and the
  backward-facing step from theirs after ``ROOM_OUTER``, all made here.
- ``replay``: every kept correction through the core. The blocks of the first
  kind (the coefficients, the imbalance, the right-hand side, the operator, the
  levels, a sweep, the group sums, the projection, the coarsest matrix, the
  correction given the reference's p') bit for bit; the pseudo-inverse and a
  V-cycle on the scale of 1e-12; the correction's CG count and its fields. The
  reference called again on the captured arguments must give what was captured.
- ``timing``: a correction on the first outer iteration's arguments of a solve
  warm-started from the stored 320x120 product state, the core against the
  reference in one process, the hierarchy's build and the CG's iterations apart,
  with the machine's state.
- ``conditioning``: the pseudo-inverses of planted coarsest systems of growing
  spread, the core's and NumPy's, against one refined in extended precision.
- ``vecinfo``: GCC's vectorisation report on the two modules' blocks.
- ``plants``: each planted defect of ``variants93.py`` built in turn, with
  CTest, the parity module and the replay run against it.

Everything is kept in ``results/builder93/`` (not committed).
"""

import argparse
import contextlib
import dataclasses
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "probe90"))
sys.path.insert(0, str(HERE.parent / "probe92"))
sys.path.insert(0, str(HERE))

# Prompt 90's probe: its timer and its record of the machine's state; prompt
# 92's: its cases and the states they start from. Their commands are not used.
import probe90  # noqa: E402 -- after sys.path
import probe92  # noqa: E402 -- after sys.path

from src import multigrid, pressure  # noqa: E402 -- after sys.path
from src.boundary_staggered import StaggeredBoundary  # noqa: E402 -- after sys.path
from src.config import SimConfig  # noqa: E402 -- after sys.path
from src.mesh import Mesh  # noqa: E402 -- after sys.path
from src.momentum import MomentumPrediction  # noqa: E402 -- after sys.path
from src.pressure import PressureCorrector  # noqa: E402 -- after sys.path
from src.solver_staggered import FlowState  # noqa: E402 -- after sys.path
from tests import core_mapping  # noqa: E402 -- after sys.path
from validation.cases import load_backstep  # noqa: E402 -- after sys.path

RESULTS = ROOT / "results" / "builder93"
CAPTURES = RESULTS / "captures"
CASES = ("product80", "cavity40", "vents80", "annex216", "backstep")
# The outer iterations captured after the warm start, and those of the solves
# from rest that make the cavity's state and the rooms'.
CALLS = 10
CAVITY_OUTER = 200
ROOM_OUTER = 400
ARGUMENTS = ("u_star", "v_star", "a_p_u", "a_p_v")
FIELDS = ("u", "v", "p", "p_prime")
COUNTS = ("iterations", "reached_cap", "products")
FIVE = ("a_p", "a_e", "a_w", "a_n", "a_s")


# ---------------------------------------------------------------------------
# The cases


def case_config(name: str) -> SimConfig:
    """The configuration a case's corrector is built from."""
    if name == "backstep":
        return load_backstep()
    return probe92.case_config(name)


def start_of(name: str) -> tuple[FlowState, dict[str, object]]:
    """The state a case's solve is warm-started from, and where it came from."""
    if name in ("cavity40", "backstep", "annex216"):
        outer = CAVITY_OUTER if name == "cavity40" else ROOM_OUTER
        state, record = probe92.solved(case_config(name), outer)
        return state, {"from": "its own solve from rest, to the cap", **record}
    return probe92.start_of(name)


class Recorder:
    """Wraps ``PressureCorrector.correct`` on the class and keeps each call."""

    def __init__(self, calls: int) -> None:
        self.calls = calls
        self.kept: list[dict[str, np.ndarray]] = []

    def install(self) -> None:
        """Wrap the method; ``uninstall`` puts the reference's back."""
        self.correct = PressureCorrector.correct
        recorder = self

        def correct(
            self: PressureCorrector, prediction: MomentumPrediction, p: np.ndarray
        ) -> pressure.PressureCorrection:
            call = {key: np.array(getattr(prediction, key)) for key in ARGUMENTS}
            call["p"] = np.array(p)
            out = recorder.correct(self, prediction, p)
            for key in FIELDS:
                call["out_" + key] = np.array(getattr(out, key))
            for key in COUNTS:
                call["out_" + key] = np.array(getattr(out, key))
            recorder.kept.append(call)
            if len(recorder.kept) == recorder.calls:
                raise probe92.CapturedError
            return out

        PressureCorrector.correct = correct

    def uninstall(self) -> None:
        """Put the reference's method back."""
        PressureCorrector.correct = self.correct


def captured(name: str, calls: int) -> tuple[Recorder, dict[str, object]]:
    """The first ``calls`` corrections of a case's warm-started solve."""
    start, source = start_of(name)
    _mesh, _boundary, solver = probe92.solver_of(case_config(name))
    recorder = Recorder(calls)
    recorder.install()
    try:
        with contextlib.suppress(probe92.CapturedError):
            solver.solve_steady(initial=start)
    finally:
        recorder.uninstall()
    return recorder, source


def capture(name: str) -> None:
    """Keep a case's corrections in ``results/builder93/captures/``."""
    recorder, source = captured(name, CALLS)
    CAPTURES.mkdir(parents=True, exist_ok=True)
    np.savez(
        CAPTURES / f"{name}.npz",
        **{
            f"call{i}/{key}": value
            for i, call in enumerate(recorder.kept)
            for key, value in call.items()
        },
    )
    first, last = recorder.kept[0], recorder.kept[-1]
    config = case_config(name)
    record = {
        "case": name,
        "start": source,
        "calls": len(recorder.kept),
        "shape": list(first["p"].shape),
        "preconditioner": config.pressure_preconditioner,
        "rtol": config.pressure_rtol,
        "iterations": [int(c["out_iterations"]) for c in recorder.kept],
        "largest_step_u": float(np.abs(last["u_star"] - first["u_star"]).max()),
        "distinct_calls": len({c["u_star"].tobytes() for c in recorder.kept}),
    }
    (CAPTURES / f"{name}.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps(record))


def loaded(name: str) -> list[dict[str, np.ndarray]]:
    """A case's kept corrections, in order."""
    calls: dict[str, dict[str, np.ndarray]] = {}
    with np.load(CAPTURES / f"{name}.npz") as z:
        for key in z.files:
            call, part = key.split("/")
            calls.setdefault(call, {})[part] = z[key]
    return list(calls.values())


# ---------------------------------------------------------------------------
# Replay


@dataclasses.dataclass
class Pair:
    """A case's two correctors."""

    config: SimConfig
    reference: PressureCorrector
    core: object


def correctors(name: str) -> Pair:
    """A case's reference corrector and the core's, built from the same arguments."""
    config = case_config(name)
    mesh = Mesh(config)
    boundary = StaggeredBoundary(mesh, config)
    reference = PressureCorrector(mesh, config, boundary)
    return Pair(
        config,
        reference,
        core_mapping.pressure_corrector(mesh, config, boundary, reference),
    )


def differing(a: np.ndarray, b: np.ndarray) -> int:
    """How many entries of two arrays differ in their 64 bits; -1 for shapes or dtypes."""
    return probe92.differing(a, b)


def largest(a: np.ndarray, reference: np.ndarray) -> float:
    """The largest difference over the reference's largest magnitude."""
    scale = float(np.abs(reference).max())
    difference = float(np.abs(a - reference).max())
    return difference / scale if scale > 0.0 else difference


def prediction_of(call: dict[str, np.ndarray]) -> MomentumPrediction:
    """A kept correction's first argument."""
    return MomentumPrediction(**{key: call[key] for key in ARGUMENTS})


class _HandedError(Exception):
    """Carries what ``correct`` handed the conjugate gradients."""


def solve_arguments(
    reference: PressureCorrector, call: dict[str, np.ndarray]
) -> np.ndarray:
    """The right-hand side ``correct`` hands the conjugate gradients on a kept call."""
    kept = {}

    def stand_in(
        apply: object,
        precondition: object,
        f: np.ndarray,
        rtol: float,
        floor: float,
        max_iter: int,
    ) -> None:
        kept["f"] = f.copy()
        raise _HandedError

    original = pressure.conjugate_gradient
    pressure.conjugate_gradient = stand_in
    try:
        with contextlib.suppress(_HandedError):
            reference.correct(prediction_of(call), call["p"])
    finally:
        pressure.conjugate_gradient = original
    return kept["f"]


def corrected_by(
    reference: PressureCorrector, call: dict[str, np.ndarray], solved: np.ndarray
) -> pressure.PressureCorrection:
    """The reference's correction by a given solution of the p' equation."""

    def stand_in(*_args: object) -> pressure.ConjugateGradientResult:
        return pressure.ConjugateGradientResult(
            x=solved.copy(),
            iterations=0,
            reached_cap=False,
            residual_norm=0.0,
            products=0,
        )

    original = pressure.conjugate_gradient
    pressure.conjugate_gradient = stand_in
    try:
        return reference.correct(prediction_of(call), call["p"])
    finally:
        pressure.conjugate_gradient = original


def first_kind(pair: Pair, call: dict[str, np.ndarray]) -> dict[str, int]:
    """Every block of the first kind on one kept correction: entries that differ, by block."""
    import _cfdcore

    reference, core = pair.reference, pair.core
    diagonals = call["a_p_u"], call["a_p_v"]
    c = reference.coefficients(*diagonals)
    five = [getattr(c, name) for name in FIVE]
    out: dict[str, int] = {}
    got = core.coefficients(*diagonals)
    out["coefficients"] = sum(differing(getattr(c, name), got[name]) for name in FIVE)
    out["face_d"] = sum(
        differing(a, b)
        for a, b in zip(
            reference._face_d(*diagonals), core.face_d(*diagonals), strict=True
        )
    )
    stars = call["u_star"], call["v_star"]
    out["mass_imbalance"] = differing(
        reference.mass_imbalance(*stars), core.mass_imbalance(*stars)
    ) + differing(
        reference.mass_imbalance(call["out_u"], call["out_v"]),
        core.mass_imbalance(call["out_u"], call["out_v"]),
    )
    f = solve_arguments(reference, call)
    out["right_hand_side"] = differing(f, core.right_hand_side(*stars, *diagonals))
    solved = call["out_p_prime"]
    for name, x in (("apply_operator", solved), ("apply_operator_f", f)):
        out[name] = differing(
            pressure.apply_operator(c, x), _cfdcore.apply_operator(*five, x)
        )
    jacobi = pressure.jacobi_preconditioner(c)
    out["jacobi_preconditioner"] = differing(
        jacobi(f),
        _cfdcore.pressure_precondition(
            *five, f, "jacobi", reference.needs_pin, core_mapping.MULTIGRID_LEVELS
        ),
    )
    hierarchy = multigrid.Multigrid(c, pressure.apply_operator)
    mine = _cfdcore.Multigrid2(*five, core_mapping.MULTIGRID_LEVELS)
    out["levels"] = 0
    out["smoothers"] = 0
    out["sweep"] = 0
    for level, own in zip(hierarchy.levels, mine.levels(), strict=True):
        arrays = [getattr(level.coefficients, name) for name in FIVE]
        out["levels"] += sum(
            differing(a, own[name]) for name, a in zip(FIVE, arrays, strict=True)
        )
        if own["smoother"] is not None:
            out["smoothers"] += differing(level.smoother, own["smoother"])
    level = hierarchy.levels[0]
    x = level.smoother * f
    residual = f - pressure.apply_operator(c, x)
    if len(hierarchy.levels) > 1:
        out["sweep"] = differing(
            x + level.smoother * residual,
            _cfdcore.smooth(*five, level.smoother, x, f),
        )
    out["restrict"] = differing(
        multigrid.restrict(residual), _cfdcore.restrict(residual)
    )
    coarse = multigrid.restrict(residual)
    expected, prolonged = x.copy(), x.copy()
    multigrid.add_prolonged(expected, coarse)
    _cfdcore.add_prolonged(prolonged, coarse)
    out["add_prolonged"] = differing(expected, prolonged)
    active = c.a_p > 0.0
    projected = x.copy()
    projected[active] -= projected[active].mean()
    out["projection"] = differing(projected, _cfdcore.project_mean(x, active))
    out["coarsest_matrix"] = differing(dense_matrix(hierarchy), mine.coarsest_matrix())
    # The solution before its pin is not kept; a field the pin must move is
    # the pinned one with a constant added to every cell with an equation.
    unpinned = np.where(active, solved + 0.125, solved)
    expected_fields = corrected_by(reference, call, unpinned)
    got_fields = core.apply(*stars, *diagonals, call["p"], unpinned)
    out["correction"] = sum(
        differing(getattr(expected_fields, name), got_fields[name]) for name in FIELDS
    )
    return out


def dense_matrix(hierarchy: multigrid.Multigrid) -> np.ndarray:
    """The matrix the reference hands ``np.linalg.pinv`` for a hierarchy's coarsest level."""
    kept = {}
    original = np.linalg.pinv

    def stand_in(matrix: np.ndarray, hermitian: bool = False) -> np.ndarray:
        kept["matrix"] = matrix.copy()
        return original(matrix, hermitian=hermitian)

    np.linalg.pinv = stand_in
    try:
        multigrid._dense_pseudo_inverse(hierarchy.levels[-1].coefficients)
    finally:
        np.linalg.pinv = original
    return kept["matrix"]


def second_kind(pair: Pair, call: dict[str, np.ndarray]) -> dict[str, object]:
    """The pseudo-inverse and a V-cycle on one kept correction, on the scale of 1e-12."""
    import _cfdcore

    c = pair.reference.coefficients(call["a_p_u"], call["a_p_v"])
    five = [getattr(c, name) for name in FIVE]
    hierarchy = multigrid.Multigrid(c, pressure.apply_operator)
    mine = _cfdcore.Multigrid2(*five, core_mapping.MULTIGRID_LEVELS)
    matrix = dense_matrix(hierarchy)
    values = np.sort(np.abs(_cfdcore.symmetric_eigenvalues(matrix)))
    lapack = np.sort(np.abs(np.linalg.eigh(matrix)[0]))
    f = solve_arguments(pair.reference, call)
    project = pair.reference.needs_pin
    z = pressure.multigrid_preconditioner(c, project)(f)
    return {
        "cells": int(matrix.shape[0]),
        "pseudo_inverse": largest(mine.coarsest_inverse(), hierarchy.coarsest_inverse),
        "cycle": largest(mine.cycle(f), hierarchy.cycle(f)),
        "precondition": largest(
            _cfdcore.pressure_precondition(
                *five, f, "multigrid", project, core_mapping.MULTIGRID_LEVELS
            ),
            z,
        ),
        # Every magnitude over the largest, smallest first: what the cutoff reads.
        "smallest_core": float(values[0] / values[-1]),
        "smallest_lapack": float(lapack[0] / lapack[-1]),
        "next_lapack": float(lapack[1] / lapack[-1]),
    }


def replay(out_name: str = "replay") -> None:
    """Every kept correction through the core and the reference."""
    rows = []
    for name in CASES:
        if not (CAPTURES / f"{name}.npz").exists():
            print(name, "not captured")
            continue
        pair = correctors(name)
        for index, call in enumerate(loaded(name)):
            again = pair.reference.correct(prediction_of(call), call["p"])
            got = pair.core.correct(*(call[key] for key in ARGUMENTS), call["p"])
            row = {
                "case": name,
                "call": index,
                "needs_pin": bool(pair.reference.needs_pin),
                "blocks": first_kind(pair, call),
                "second": second_kind(pair, call),
                "reference_differs": sum(
                    differing(call["out_" + key], getattr(again, key)) for key in FIELDS
                ),
                "iterations": [int(call["out_iterations"]), int(got["iterations"])],
                "products": [int(call["out_products"]), int(got["products"])],
                "reached_cap": [bool(call["out_reached_cap"]), got["reached_cap"]],
                "fields": {
                    key: largest(got[key], call["out_" + key]) for key in FIELDS
                },
            }
            rows.append(row)
        mine = [r for r in rows if r["case"] == name]
        print(
            name,
            len(mine),
            "corrections,",
            sum(1 for r in mine if any(r["blocks"].values())),
            "with a block that differs,",
            sum(1 for r in mine if r["iterations"][0] != r["iterations"][1]),
            "counts that differ, worst field",
            f"{max(max(r['fields'][k] for k in ('u', 'v', 'p')) for r in mine):.1e},",
            "worst pinv",
            f"{max(r['second']['pseudo_inverse'] for r in mine):.1e},",
            "worst cycle",
            f"{max(r['second']['cycle'] for r in mine):.1e},",
            sum(1 for r in mine if r["reference_differs"]),
            "not reproduced",
        )
    target = RESULTS / f"{out_name}.json"
    if out_name != "replay":
        target = RESULTS / "plants" / f"{out_name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Timing

REFERENCE_REPEATS = 40
CORE_REPEATS = 200
# The two caps whose corrections' times give an iteration's.
CAPS = (1, 4)


def capped_correctors(name: str, cap: int) -> Pair:
    """A case's two correctors with the iteration cap replaced."""
    config = case_config(name)
    config.max_pressure_iter = cap
    mesh = Mesh(config)
    boundary = StaggeredBoundary(mesh, config)
    reference = PressureCorrector(mesh, config, boundary)
    core = core_mapping.pressure_corrector(mesh, config, boundary, reference)
    return Pair(config, reference, core)


def timing(run_number: int) -> None:
    """A correction on the 320x120 product state, core against reference, in one process."""
    import _cfdcore

    before = probe90.windows_load()
    calibration = probe90.calibration()
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    recorder, _source = captured("product320", 1)
    call = recorder.kept[0]
    pair = correctors("product320")
    prediction, p = prediction_of(call), call["p"]
    arguments = [call[key] for key in ARGUMENTS] + [p]
    got = pair.core.correct(*arguments)
    again = pair.reference.correct(prediction, p)
    ref = probe90.timed(
        lambda: pair.reference.correct(prediction, p), REFERENCE_REPEATS
    )
    own = probe90.timed(lambda: pair.core.correct(*arguments), CORE_REPEATS)
    # The parts. The build is each solver's own, the core's through a hierarchy
    # kept for it, so that neither the binding's copies nor an allocation is in
    # it. The iterations are read off corrections capped at two counts: the
    # difference of their times over the difference of the counts, each with
    # the one product the cap's true residual takes.
    c = pair.reference.coefficients(call["a_p_u"], call["a_p_v"])
    five = [getattr(c, name) for name in FIVE]
    f = solve_arguments(pair.reference, call)
    levels = core_mapping.MULTIGRID_LEVELS
    hierarchy = multigrid.Multigrid(c, pressure.apply_operator)
    mine = _cfdcore.Multigrid2(*five, levels)
    matrix = dense_matrix(hierarchy)
    parts = {
        "assembly": {
            "reference": probe90.timed(
                lambda: pair.reference.coefficients(call["a_p_u"], call["a_p_v"]),
                REFERENCE_REPEATS,
            ),
            "core": probe90.timed(
                lambda: pair.core.face_d(call["a_p_u"], call["a_p_v"]), CORE_REPEATS
            ),
        },
        "build": {
            "reference": probe90.timed(
                lambda: multigrid.Multigrid(c, pressure.apply_operator),
                REFERENCE_REPEATS,
            ),
            "core": probe90.timed(mine.rebuild, CORE_REPEATS),
        },
        "pseudo_inverse": {
            "reference": probe90.timed(
                lambda: np.linalg.pinv(matrix, hermitian=True), REFERENCE_REPEATS
            ),
            "core": probe90.timed(
                lambda: _cfdcore.pseudo_inverse(matrix, levels[-1]), CORE_REPEATS
            ),
        },
        "cycle": {
            "reference": probe90.timed(lambda: hierarchy.cycle(f), REFERENCE_REPEATS),
            "core": probe90.timed(lambda: mine.cycle(f), CORE_REPEATS),
        },
    }
    capped = {cap: capped_correctors("product320", cap) for cap in CAPS}
    for cap, short in capped.items():
        parts[f"capped_{cap}"] = {
            "reference": probe90.timed(
                lambda short=short: short.reference.correct(prediction, p),
                REFERENCE_REPEATS,
            ),
            "core": probe90.timed(
                lambda short=short: short.core.correct(*arguments), CORE_REPEATS
            ),
        }
    record = {
        "run": run_number,
        "started": started,
        "iterations": [int(again.iterations), int(got["iterations"])],
        "fields": {key: largest(got[key], getattr(again, key)) for key in FIELDS},
        "reference": ref,
        "core": own,
        "ratio": ref["median_s"] / own["median_s"],
        "parts": parts,
        "calibration_before": calibration,
        "calibration_after": probe90.calibration(),
        "windows_before": before,
        "windows_after": probe90.windows_load(),
    }
    (RESULTS / f"timing_{run_number}.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )
    print(
        f"run {run_number}: {ref['median_s'] * 1e3:.2f} ms against "
        f"{own['median_s'] * 1e3:.3f} ms, ratio {record['ratio']:.2f}, iterations "
        f"{record['iterations']}"
    )
    for part, times in parts.items():
        print(
            f"  {part}: {times['reference']['median_s'] * 1e3:.3f} against "
            f"{times['core']['median_s'] * 1e3:.3f} ms"
        )


# ---------------------------------------------------------------------------
# The pseudo-inverse against the exact one

# Rotations below this share of the largest diagonal are skipped, in an
# arithmetic whose rounding is 5e-20.
EXACT_SKIP = np.longdouble(2) ** -90


def exact_inverse(matrix: np.ndarray, rcond: float) -> tuple[np.ndarray, np.ndarray]:
    """The pseudo-inverse of a symmetric matrix in extended precision, and its eigenvalues.

    Two-sided Jacobi rotations in ``np.longdouble`` (x87's 64-bit mantissa), so
    its error is three decades below either double one's.
    """
    n = matrix.shape[0]
    lower = np.tril(matrix).astype(np.longdouble)
    work = lower + np.tril(lower, -1).T
    vectors = np.eye(n, dtype=np.longdouble)
    for _sweep in range(40):
        skip = EXACT_SKIP * np.abs(np.diag(work)).max()
        rotated = False
        for p in range(n - 1):
            for q in range(p + 1, n):
                entry = work[p, q]
                if abs(entry) <= skip:
                    continue
                rotated = True
                theta = (work[q, q] - work[p, p]) / (2 * entry)
                t = (1 if theta >= 0 else -1) / (
                    abs(theta) + np.sqrt(1 + theta * theta)
                )
                c = 1 / np.sqrt(1 + t * t)
                s = t * c
                row = work[p].copy()
                work[p] = c * row - s * work[q]
                work[q] = s * row + c * work[q]
                column = work[:, p].copy()
                work[:, p] = c * column - s * work[:, q]
                work[:, q] = s * column + c * work[:, q]
                work[p, q] = work[q, p] = 0
                vector = vectors[p].copy()
                vectors[p] = c * vector - s * vectors[q]
                vectors[q] = s * vector + c * vectors[q]
        if not rotated:
            break
    else:
        raise SystemExit("the extended rotations did not converge")
    values = np.diag(work)
    kept = np.abs(values) > np.longdouble(rcond) * np.abs(values).max()
    inverse = (vectors[kept].T / values[kept]) @ vectors[kept]
    return inverse, np.sort(np.abs(values))


def conditioning_row(matrix: np.ndarray, rcond: float) -> dict[str, object]:
    """One coarsest system: its condition, the dropped values and both inverses' errors."""
    import _cfdcore

    exact, values = exact_inverse(matrix, rcond)
    kept = values[values > np.longdouble(rcond) * values[-1]]
    dropped = values[: len(values) - len(kept)]
    numpy_inverse = np.linalg.pinv(matrix, hermitian=True)
    core_inverse = _cfdcore.pseudo_inverse(matrix, rcond)
    lapack = np.sort(np.abs(np.linalg.eigh(matrix)[0]))
    own = np.sort(np.abs(_cfdcore.symmetric_eigenvalues(matrix)))
    scale = np.abs(exact).max()
    return {
        "cells": int(matrix.shape[0]),
        "null": len(dropped),
        "condition": float(values[-1] / kept[0]),
        # The largest dropped magnitude over the largest, where the cutoff is
        # rcond: the exact one, LAPACK's and the core's.
        "dropped_exact": float(dropped[-1] / values[-1]) if len(dropped) else 0.0,
        "dropped_lapack": float(lapack[len(dropped) - 1] / lapack[-1])
        if len(dropped)
        else 0.0,
        "dropped_core": float(own[len(dropped) - 1] / own[-1]) if len(dropped) else 0.0,
        "numpy_error": float(np.abs(numpy_inverse - exact).max() / scale),
        "core_error": float(np.abs(core_inverse - exact).max() / scale),
        "between": largest(core_inverse, numpy_inverse),
    }


def planted_systems() -> list[tuple[str, list[np.ndarray]]]:
    """The parity module's planted systems, and worse conditioned ones besides."""
    from tests import test_core_pressure as module

    out = []
    for shape in module.COARSEST:
        for (solid, spread, outlet), five in module.planted_coarsest(shape):
            out.append((f"{shape[0]}x{shape[1]} {solid} {spread} {outlet}", five))
    rng = np.random.default_rng([module.SEED, 99])
    for shape in ((4, 10), (8, 8), (1, 40), (1, 64)):
        for spread in (2.0, 3.0, 4.0, 5.0):
            for solid in (0.0, 0.2):
                out.append(
                    (
                        f"worse {shape[0]}x{shape[1]} {solid} {spread}",
                        module._neumann(*shape, rng, spread, solid),
                    )
                )
    return out


def conditioning() -> None:
    """The planted and the captured coarsest systems' inverses against the exact ones."""
    import _cfdcore

    levels = core_mapping.MULTIGRID_LEVELS
    rows = []
    for name, five in planted_systems():
        matrix = _cfdcore.Multigrid2(*five, levels).coarsest_matrix()
        kind = "worse" if name.startswith("worse") else "planted"
        rows.append(
            {"kind": kind, "name": name, **conditioning_row(matrix, levels[-1])}
        )
    for name in CASES:
        pair = correctors(name)
        for index, call in enumerate(loaded(name)):
            got = pair.core.coefficients(call["a_p_u"], call["a_p_v"])
            matrix = _cfdcore.Multigrid2(
                *(got[key] for key in FIVE), levels
            ).coarsest_matrix()
            rows.append(
                {
                    "kind": "captured",
                    "name": f"{name} {index}",
                    **conditioning_row(matrix, levels[-1]),
                }
            )
        print(name, "done", flush=True)
    (RESULTS / "conditioning.json").write_text(
        json.dumps(rows, indent=1) + "\n", encoding="utf-8"
    )
    for kind in ("planted", "worse", "captured"):
        mine = [r for r in rows if r["kind"] == kind]
        print(
            kind,
            len(mine),
            "beyond 1e-12 of NumPy's:",
            sum(r["between"] > 1e-12 for r in mine),
            "worst between, numpy, core:",
            *(
                f"{max(r[k] for r in mine):.1e}"
                for k in ("between", "numpy_error", "core_error")
            ),
        )


# ---------------------------------------------------------------------------
# The vectorisation report

VEC_FLAGS = ["-std=c++20", "-O2", "-ffp-contract=off", "-march=x86-64-v3", "-DNDEBUG"]
VEC_FILES = ("pressure", "multigrid", "dense")


def vecinfo() -> None:
    """Which blocks of the three files GCC vectorises, its loops counted by place."""
    version = subprocess.run(["g++", "--version"], capture_output=True, text=True)
    record: dict[str, object] = {
        "flags": VEC_FLAGS,
        "compiler": version.stdout.splitlines()[0],
        "blocks": [],
    }
    for file in VEC_FILES:
        path = ROOT / "core" / "src" / f"{file}.cpp"
        done = subprocess.run(
            [
                "g++",
                *VEC_FLAGS,
                f"-I{ROOT / 'core' / 'include'}",
                "-c",
                str(path),
                "-o",
                "/dev/null",
                "-fopt-info-vec-optimized",
            ],
            capture_output=True,
            text=True,
        )
        found = re.findall(
            rf"{file}\.cpp:(\d+):\d+: optimized: +loop vectorized using (\d+) byte",
            done.stdout + done.stderr,
        )
        places = {int(line) for line, _width in found}
        name = ""
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = re.match(r"inline [\w:]+ (\w+)\(lanes::cells<N>", line)
            if match:
                name = match.group(1)
            if "for (std::size_t k = 0; k < N; ++k)" in line:
                record["blocks"].append(
                    {
                        "file": file,
                        "block": name,
                        "vectorised": number in places,
                        "widths": sorted({w for at, w in found if int(at) == number}),
                    }
                )
    (RESULTS / "vecinfo.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )
    for block in record["blocks"]:
        print(block["file"], block["block"], block["vectorised"], block["widths"])


# ---------------------------------------------------------------------------
# Planted defects

PLANT_WORK = Path.home() / "scratch93" / "plants"
PLANT_TIMEOUT = 900


def plants(stream: int, streams: int, only: list[str]) -> None:
    """Each planted defect built in turn, with the tests and the replay that fail on it."""
    import variants93

    work = PLANT_WORK / f"s{stream}"
    mine = [
        plant
        for i, plant in enumerate(variants93.PLANTS)
        if plant[0] in only
        or (not only and (i % streams == stream or plant[0] == "control"))
    ]
    records = []
    for name, what, edits in mine:
        record: dict[str, object] = {"name": name, "what": what, "stream": stream}
        failed = probe92.build(work, edits, tests=True)
        if failed:
            record["build"] = failed
        else:
            run_nets(name, work / "build", record)
        records.append(record)
        print(
            name,
            "|",
            record.get("build", "")[-300:],
            record.get("ctest"),
            "|",
            record.get("pytest"),
            "|",
            "; ".join(record.get("replay", [])),
            flush=True,
        )
        (RESULTS / f"plants_s{stream}.json").write_text(
            json.dumps(records, indent=1) + "\n", encoding="utf-8"
        )


def run_nets(name: str, tree: Path, record: dict[str, object]) -> None:
    """CTest, the parity module and the replay on one build, their failures into the record."""

    def run(command: list[str], **keys: object) -> tuple[int, str]:
        try:
            done = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=PLANT_TIMEOUT,
                **keys,
            )
        except subprocess.TimeoutExpired:
            # A plant that keeps a solve from ending is caught by that.
            return -1, "timed out"
        return done.returncode, done.stdout

    _code, out = run(["ctest", "--test-dir", str(tree), "--timeout", "60"])
    summary = [line for line in out.splitlines() if "tests passed" in line]
    record["ctest"] = summary[0].strip() if summary else out[-300:]
    # A test that crashes is listed as (SEGFAULT) or (Subprocess aborted), not (Failed).
    record["ctest_failed"] = re.findall(r"\d+ - (\S+) \([^)]+\)", out)
    env = {
        **os.environ,
        "PYTHONPATH": str(tree),
        "CFD_CORE_REQUIRED": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    module = "tests/test_core_pressure.py"
    code, out = run(
        [
            sys.executable,
            "-m",
            "pytest",
            module,
            "-q",
            "-rf",
            "--tb=no",
            "-p",
            "no:cacheprovider",
        ],
        cwd=ROOT,
        env=env,
    )
    lines = out.splitlines()
    record["pytest_returncode"] = code
    record["pytest"] = next(
        (x.strip("= ") for x in reversed(lines) if re.search(r" in [0-9.]+s", x)),
        lines[-1] if lines else "no output",
    )
    bad = [
        line.split("::", 1)[1].split(" ")[0]
        for line in lines
        if line.startswith(f"FAILED {module}::")
    ]
    record["parity_failed_by_test"] = dict(
        sorted(Counter(case.split("[")[0] for case in bad).items())
    )
    code, out = run(
        [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "replay",
            "--out",
            f"plant_{name}",
        ],
        cwd=ROOT,
        env=env,
    )
    record["replay_returncode"] = code
    record["replay"] = [x for x in out.splitlines() if " corrections," in x]


# ---------------------------------------------------------------------------


def main() -> None:
    """Run one command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("capture", "replay", "timing", "conditioning", "vecinfo", "plants"),
    )
    parser.add_argument("case", nargs="?")
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--out", default="replay")
    parser.add_argument("--stream", type=int, default=0)
    parser.add_argument("--streams", type=int, default=1)
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args()
    commands = {
        "capture": lambda: capture(args.case),
        "replay": lambda: replay(args.out),
        "timing": lambda: timing(args.run),
        "conditioning": conditioning,
        "vecinfo": vecinfo,
        "plants": lambda: plants(args.stream, args.streams, args.only),
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
