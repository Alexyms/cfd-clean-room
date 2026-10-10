"""What the stopping rule should measure: iteration error, its estimate, continuity.

Solves VAL-001 at 40x20 and 80x40 and the cavity at 20x20, 40x40 and 80x80 once
each with the committed settings except convergence_tol, set in memory to
TRUTH_TOL, and the iteration cap. The field at TRUTH_TOL is the truth for that
case and grid. The solver's corrector is wrapped on the instance to record each
outer iteration's worst per-cell imbalance and pressure iteration count; src/ is
not edited. Every saved solve stores PRESSURE_SOLVER_VERSION and is solved again
when the stored value differs or is missing, so a field written by the weighted
Jacobi sweep is never read as the conjugate gradient solve's. Every reader of a
truth, verify_rule's and analyse's included, takes it through solve_truth.
u and v are kept at each quarter decade of residual from 1e-5 to TRUTH_TOL and at
the case's own tolerance. The same-computation control runs first and stops the
script if an unwrapped solve at the committed tolerance differs from the wrapped
snapshot there. Fields go to results/stopping_probe/ (gitignored) and are solved
again only when their stored version is not this solver's; the analysis writes
summary.json there.

With --verify-rule each case is solved once more under the error_estimate
stopping rule (src/stopping.py) at its default tolerances, set in memory, and
the stop is read against the truth into verify_rule.json.

    python scripts/stopping_probe.py
    python scripts/stopping_probe.py --verify-rule
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# face_profiles is the one name read through self_convergence (prompt 23 asked
# for the cavity profiles to come from it); every other name is imported from
# the module that defines it, so pruning an import in self_convergence cannot
# break this script at run time.
import self_convergence as sc  # noqa: E402 -- path set above

from src.boundary_staggered import StaggeredBoundary  # noqa: E402 -- path set above
from src.config import SimConfig  # noqa: E402 -- path set above
from src.mesh import FLUID, Mesh  # noqa: E402 -- path set above
from src.pressure import PRESSURE_SOLVER_VERSION  # noqa: E402 -- path set above
from src.solver_staggered import StaggeredSolver  # noqa: E402 -- path set above
from src.stopping import (  # noqa: E402 -- path set above
    RATE_WINDOW,
    ErrorEstimateRule,
    ImbalanceSummary,
    IterationState,
)
from validation.cases import (  # noqa: E402 -- path set above
    case_path,
    with_velocity_step,
)
from validation.metrics import (  # noqa: E402 -- path set above
    MARCHI_U_ROWS,
    MARCHI_V_ROWS,
    inlet_velocity,
    lagrange,
    lid_velocity,
    poiseuille_l2_error,
    poiseuille_profiles,
)

if TYPE_CHECKING:
    from src.momentum import MomentumPrediction
    from src.pressure import PressureCorrection

OUT_DIR = REPO_ROOT / "results" / "stopping_probe"
TRUTH_TOL = 1.0e-11
LEVELS = tuple(10.0 ** (-k / 4) for k in range(20, 45))  # 1e-5 to 1e-11
# Control cases first, so a failed control stops the script before the rest.
CASES = (
    ("cavity", 20),
    ("poiseuille", 80),
    ("poiseuille", 40),
    ("cavity", 40),
    ("cavity", 80),
)
CONTROLS = ("cavity_20x20", "poiseuille_80x40")
# The channel's committed cap is 20000, the count prompt 23 stops at for 80x40.
MAX_OUTER = {"poiseuille": 20000, "cavity": 40000}
# Trailing window of the rho_hat fit, in outer iterations. It is below the outer count
# at every case's first snapshot (142 on VAL-001 40x20, the fastest case at about 71 per
# decade), so every snapshot has a rate. Halving and doubling it test the length.
WINDOW = 100
IMBALANCE_BOUND = 1.0e-10  # ECR-001 acceptance criterion 6
BUDGET_SECONDS = 3600.0
STORED_VAL001 = 0.002306261116980951  # benchmarks/results.jsonl, commit 0e7f5b0
# The 1e-11 channel truths carry their own flux drift, 9.0e-8 and 2.25e-7 of U
# by test 24, so --verify-rule reads the channel against a truth at 1e-13.
TIGHT_TRUTH_TOL = 1.0e-13


def case_name(case: str, n: int) -> str:
    """File stem of a case at n cells along x (n / 2 up the channel).

    Parameters
    ----------
    case : str
        "cavity" or "poiseuille".
    n : int
        Cells along x.

    Returns
    -------
    str
        ``<case>_<n>x<ny>``, ny equal to n for the cavity and n // 2 for the
        channel.
    """
    return f"{case}_{n}x{n if case == 'cavity' else n // 2}"


def case_config(
    case: str, n: int, tol: float | None = None, rule: str | None = None
) -> SimConfig:
    """The committed case at n cells along x; tolerance or rule, and cap, set if given.

    Parameters
    ----------
    case : str
        "cavity" or "poiseuille".
    n : int
        Cells along x.
    tol : float, optional
        ``convergence_tol`` to set in memory; the cap is set with it.
    rule : str, optional
        ``stopping_rule`` to set in memory; the cap is set with it.

    Returns
    -------
    SimConfig
        The case file's configuration with the grid, tolerance and rule
        replaced as given. The case file is not touched.

    Notes
    -----
    Without a rule it is velocity_step, the rule every truth and snapshot here
    was solved under, whatever the case file now names.
    """
    raw = yaml.safe_load(case_path(case).read_text(encoding="utf-8"))
    raw["domain"]["nx"], raw["domain"]["ny"] = n, (n if case == "cavity" else n // 2)
    if tol is not None:
        raw["solver"]["convergence_tol"] = tol
    if rule is not None:
        raw["solver"]["stopping_rule"] = rule
    if tol is not None or rule is not None:
        raw["solver"]["max_simple_iter"] = MAX_OUTER[case]
    config = SimConfig.from_dict(raw)
    return config if rule is not None else with_velocity_step(config)


def instrument(
    solver: StaggeredSolver,
) -> tuple[list[float], list[int], list[float], list[float]]:
    """Record each outer iteration's worst imbalance, iterations, absolute and signed sum.

    Parameters
    ----------
    solver : StaggeredSolver
        Not yet solved. Its corrector's ``correct`` is wrapped on the instance.

    Returns
    -------
    tuple[list[float], list[int], list[float], list[float]]
        Per outer iteration, filled as the solve runs: the worst per-cell mass
        imbalance of the corrected faces, the pressure iterations, the sum of
        absolute imbalances and the signed sum.

    Notes
    -----
    The wrapper returns the corrector's own result object, so the solve is
    unchanged; the same-computation control checks that bitwise.
    """
    corrector, correct = solver._corrector, solver._corrector.correct
    imbalance: list[float] = []
    iterations: list[int] = []
    total: list[float] = []
    signed: list[float] = []

    def recorded(prediction: MomentumPrediction, p: np.ndarray) -> PressureCorrection:
        result = correct(prediction, p)
        net = corrector.mass_imbalance(result.u, result.v)
        cells = np.abs(net)
        imbalance.append(float(cells.max()))
        total.append(float(cells.sum()))
        signed.append(float(net.sum()))
        iterations.append(result.iterations)
        return result

    corrector.correct = recorded
    return imbalance, iterations, total, signed


def written_by_this_solver(path: Path) -> bool:
    """Whether a saved solve holds the current PRESSURE_SOLVER_VERSION.

    Parameters
    ----------
    path : Path
        A saved ``.npz``; it need not exist.

    Returns
    -------
    bool
        True only when the file exists and stores the current version.

    Notes
    -----
    A file without the key was written before the key existed, by the
    weighted Jacobi sweep, and is not reused.
    """
    if not path.exists():
        return False
    with np.load(path) as saved:
        return "pressure_solver_version" in saved.files and int(
            saved["pressure_solver_version"]
        ) == int(PRESSURE_SOLVER_VERSION)


def solve_truth(case: str, n: int) -> Path:
    """Solve to TRUTH_TOL wrapped, keeping snapshots.

    Parameters
    ----------
    case : str
        "cavity" or "poiseuille".
    n : int
        Cells along x.

    Returns
    -------
    Path
        The ``.npz`` under OUT_DIR holding level, iteration, residual,
        imbalance, iterations, elapsed, reference_velocity, reached, the
        snapshots u_<k> and v_<k> and the pressure solver version.

    Notes
    -----
    An existing file is returned unsolved only when it was written by the
    current pressure solve (written_by_this_solver); otherwise it is solved
    again and overwritten.
    """
    path = OUT_DIR / f"{case_name(case, n)}.npz"
    if written_by_this_solver(path):
        return path
    config = case_config(case, n, TRUTH_TOL)
    mesh = Mesh(config)
    solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))
    imbalance, iterations, _total, _signed = instrument(solver)
    levels = sorted({*LEVELS, case_config(case, n).convergence_tol}, reverse=True)
    taken: list[tuple[float, int]] = []
    snaps: dict[str, np.ndarray] = {}
    elapsed: list[float] = []
    start = time.perf_counter()

    def observe(state: IterationState) -> None:
        elapsed.append(time.perf_counter() - start)
        while levels and state.residual < levels[0]:
            snaps[f"u_{len(taken)}"], snaps[f"v_{len(taken)}"] = (
                state.u.copy(),
                state.v.copy(),
            )
            taken.append((levels.pop(0), state.iteration))

    solver.solve_steady(on_iteration=observe)
    np.savez(
        path,
        level=np.array([t[0] for t in taken]),
        iteration=np.array([t[1] for t in taken], dtype=int),
        residual=np.array(solver.residual_history),
        imbalance=np.array(imbalance),
        iterations=np.array(iterations),
        elapsed=np.array(elapsed),
        reference_velocity=solver.reference_velocity,
        reached=not levels,
        pressure_solver_version=PRESSURE_SOLVER_VERSION,
        **snaps,
    )
    print(
        f"{path.stem}: {len(elapsed)} outer, {elapsed[-1]:.0f} s, reached {not levels}"
    )
    return path


def control(case: str, n: int) -> dict:
    """An unwrapped solve at the committed tolerance against the wrapped snapshot there.

    Parameters
    ----------
    case : str
        "cavity" or "poiseuille".
    n : int
        Cells along x.

    Returns
    -------
    dict
        outer, seconds and bitwise_equal; for the channel also l2, the
        metric of the unwrapped field, and l2_equals_stored.

    Raises
    ------
    SystemExit
        Unless u and v are bitwise equal and the outer counts agree.

    Notes
    -----
    The saved control is reused only when the current pressure solve wrote
    it (written_by_this_solver).
    """
    name, config = case_name(case, n), case_config(case, n)
    path, mesh = OUT_DIR / f"{name}_control.npz", Mesh(config)
    if not written_by_this_solver(path):
        solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))
        start = time.perf_counter()
        u, v, _p = solver.solve_steady()
        seconds = time.perf_counter() - start
        np.savez(
            path,
            u=u,
            v=v,
            outer=len(solver.residual_history),
            seconds=seconds,
            pressure_solver_version=PRESSURE_SOLVER_VERSION,
        )
    with np.load(path) as plain, np.load(solve_truth(case, n)) as wrapped:
        k = int(np.flatnonzero(wrapped["level"] == config.convergence_tol)[0])
        same = np.array_equal(plain["u"], wrapped[f"u_{k}"]) and np.array_equal(
            plain["v"], wrapped[f"v_{k}"]
        )
        outer = int(plain["outer"])
        out = {"outer": outer, "seconds": float(plain["seconds"])}
        out["bitwise_equal"] = bool(same and outer == int(wrapped["iteration"][k]) + 1)
        if case == "poiseuille":
            out["l2"] = poiseuille_l2_error(config, mesh, plain["u"]).value
            out["l2_equals_stored"] = out["l2"] == STORED_VAL001
    print(f"control {name}: {out}")
    if not out["bitwise_equal"]:
        raise SystemExit(f"same-computation control failed on {name}")
    return out


def rho_hat(history: np.ndarray, window: int) -> float:
    """Per-iteration convergence rate fitted to the tail of a residual history.

    Parameters
    ----------
    history : np.ndarray
        Residual per outer iteration, shape [outer].
    window : int
        Number of trailing entries to fit.

    Returns
    -------
    float
        exp of the least-squares slope of log(residual) over the last window
        entries; NaN if the history is shorter than the window.
    """
    if len(history) < window:
        return float("nan")
    return float(np.exp(np.polyfit(np.arange(window), np.log(history[-window:]), 1)[0]))


def estimate(step: float, rho: float) -> float:
    """Iteration error left by a geometric iteration after a step: step rho / (1 - rho).

    Parameters
    ----------
    step : float
        Largest velocity change of the last outer iteration, in m/s.
    rho : float
        Convergence rate per outer iteration, from rho_hat.

    Returns
    -------
    float
        The estimate in m/s. Infinite for a rate of 1 or more; NaN when there
        is no rate.
    """
    return float("inf") if rho >= 1.0 else step * rho / (1.0 - rho)


def true_error(
    u: np.ndarray,
    v: np.ndarray,
    truth: tuple[np.ndarray, np.ndarray],
    fluid: np.ndarray,
) -> float:
    """Largest abs difference from the truth in u or v over the FLUID cells.

    Parameters
    ----------
    u, v : np.ndarray
        Cell-centered fields, shape [ny, nx].
    truth : tuple[np.ndarray, np.ndarray]
        The reference u and v, same shape.
    fluid : np.ndarray
        Boolean mask of the FLUID cells, same shape.

    Returns
    -------
    float
        The largest absolute difference, in m/s.
    """
    return float(
        max(np.abs(u - truth[0])[fluid].max(), np.abs(v - truth[1])[fluid].max())
    )


def channel_readings(config: SimConfig, mesh: Mesh, u: np.ndarray) -> dict:
    """poiseuille_l2_error at nx/2, nx/4 and 3nx/4, and the range of u_num / u_ref at nx/2.

    Parameters
    ----------
    config : SimConfig
        Channel case configuration.
    mesh : Mesh
        Mesh of the solve.
    u : np.ndarray
        Cell-centered x-velocity, shape [ny, nx].

    Returns
    -------
    dict
        ratio_min and ratio_max of u_num / u_ref at nx/2, and the metric's
        value at the three columns as l2, l2_quarter and l2_three_quarter.

    Notes
    -----
    Each column is rolled to nx // 2, where the metric reads; every interior
    channel column has the same FLUID rows, so the metric's mask still applies.
    """
    _y, u_num, u_ref = poiseuille_profiles(config, mesh, u)
    out = {
        "ratio_min": float(np.min(u_num / u_ref)),
        "ratio_max": float(np.max(u_num / u_ref)),
    }
    for key, i in (
        ("l2", config.nx // 2),
        ("l2_quarter", config.nx // 4),
        ("l2_three_quarter", 3 * config.nx // 4),
    ):
        out[key] = poiseuille_l2_error(
            config, mesh, np.roll(u, config.nx // 2 - i, axis=1)
        ).value
    return out


def wall_stencil_profile(config: SimConfig, mesh: Mesh) -> np.ndarray:
    """Fully developed discrete u under the half-cell wall stencil, all ny rows (INFERRED).

    Parameters
    ----------
    config : SimConfig
        Channel case configuration.
    mesh : Mesh
        Mesh of the solve.

    Returns
    -------
    np.ndarray
        u per row, shape [ny], in m/s.

    Notes
    -----
    The interior three-point difference is exact for a parabola; the wall row's
    (u_0 - 0) / (dy / 2) is not, and shifts it by dy^2 / 4. Scaled so the
    midpoint sum carries the inflow.
    """
    y, dy, height = np.asarray(mesh.yc), np.asarray(mesh.dy_cell), config.room_height
    shape = y * (height - y) + dy**2 / 4.0
    return shape * inlet_velocity(config) * height / float(np.sum(shape * dy))


def marchi_stations(
    config: SimConfig, mesh: Mesh, u: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """u then v at Marchi's 30 stations, as marchi_comparison takes them, over the lid speed.

    Parameters
    ----------
    config : SimConfig
        Cavity case configuration; supplies the lid speed.
    mesh : Mesh
        Mesh of the solve.
    u, v : np.ndarray
        Cell-centered fields, shape [ny, nx].

    Returns
    -------
    np.ndarray
        The 30 normalized values, shape [30]: Marchi's u stations, then v.
    """
    lines = sc.face_profiles(mesh, u, v, lid_velocity(config))
    rows = (MARCHI_U_ROWS, MARCHI_V_ROWS)
    return np.concatenate(
        [
            lagrange(*ln, np.array([r[0] for r in rs]))
            for ln, rs in zip(lines, rows, strict=True)
        ]
    )


def analyse(case: str, n: int) -> dict:
    """Every snapshot of one case against its truth, and the truth against its reference.

    Parameters
    ----------
    case : str
        "cavity" or "poiseuille".
    n : int
        Cells along x.

    Returns
    -------
    dict
        The case summary: reference and physical velocity, outer count, seconds,
        final residual and when the imbalance fell below its bound; then, if
        TRUTH_TOL was reached, the truth's rate and remaining error, its
        discretization error against the reference and one row per snapshot.
        A case that did not reach TRUTH_TOL returns the first group and its
        whole residual history as ``residual_history``, and nothing that needs
        a truth.

    Notes
    -----
    The truth is read through solve_truth, so a file another solver wrote is
    solved again first, whatever ran before this call.
    """
    config = case_config(case, n)
    mesh = Mesh(config)
    fluid = mesh.cell_type == FLUID
    with np.load(solve_truth(case, n)) as saved:
        d = {k: saved[k] for k in saved.files}
    res, ref_vel, imb = d["residual"], float(d["reference_velocity"]), d["imbalance"]
    phys = lid_velocity(config) if case == "cavity" else inlet_velocity(config)
    below = np.flatnonzero(imb < IMBALANCE_BOUND)
    out: dict = {
        "reference_velocity": ref_vel,
        "physical_velocity": phys,
        "reached": bool(d["reached"]),
    }
    out |= {
        "outer": len(res),
        "seconds": float(d["elapsed"][-1]),
        "final_residual": float(res[-1]),
    }
    out["imbalance_below_bound_from"] = int(below[0]) + 1 if below.size else None
    out["residual_there"] = float(res[below[0]]) if below.size else None
    out["imbalance_stays_below"] = bool(
        below.size and below[0] + below.size == len(res)
    )
    if not out["reached"]:
        # Prompt 23: a case that cannot reach TRUTH_TOL is reported with its
        # residual history, so the stall can be read from the summary.
        out["residual_history"] = res.tolist()
        return out
    last = len(d["level"]) - 1
    truth = (d[f"u_{last}"], d[f"v_{last}"])
    rho_truth = rho_hat(res, WINDOW)
    out["truth"] = {
        "rho_hat": rho_truth,
        "remaining_error": estimate(res[-1] * ref_vel, rho_truth),
    }
    if case == "cavity":
        marchi = np.array([r[1] for r in (*MARCHI_U_ROWS, *MARCHI_V_ROWS)])
        at_truth = marchi_stations(config, mesh, *truth)
        out["discretization"] = float(np.abs(at_truth - marchi).max())
    else:
        dev = wall_stencil_profile(config, mesh)
        developed = np.repeat(dev[:, None], config.nx, axis=1)
        disc = channel_readings(config, mesh, truth[0])
        disc["prompt_prediction"] = 1.0 / (2.0 * config.ny**2)
        disc["wall_stencil_l2"] = poiseuille_l2_error(config, mesh, developed).value
        # At nx/2, 3nx/4 and the last FLUID column: development decays along x.
        gap = np.abs(truth[0] - dev[:, None]).max(axis=0)
        cols = (config.nx // 2, 3 * config.nx // 4, config.nx - 2)
        disc["wall_stencil_gap"] = [float(gap[i]) for i in cols]
        _y, u_col_truth, u_ref = poiseuille_profiles(config, mesh, truth[0])
        out["discretization"], out["truth_ratio_profile"] = (
            disc,
            (u_col_truth / u_ref).tolist(),
        )
    out["snapshots"] = []
    for k, it in enumerate(d["iteration"].tolist()):
        u, v = d[f"u_{k}"], d[f"v_{k}"]
        err, step = true_error(u, v, truth, fluid), float(res[it] * ref_vel)
        row = {
            "level": float(d["level"][k]),
            "outer": it + 1,
            "seconds": float(d["elapsed"][it]),
        }
        row |= {
            "residual": float(res[it]),
            "step": step,
            "true_error": err,
            "true_error_rel": err / phys,
        }
        row |= {"imbalance": float(imb[it]), "iterations": int(d["iterations"][it])}
        worst = np.maximum(np.abs(u - truth[0]), np.abs(v - truth[1])) * fluid
        row["worst_cell"] = [int(x) for x in np.unravel_index(worst.argmax(), u.shape)]
        for w in (WINDOW // 2, WINDOW, 2 * WINDOW):
            rho = rho_hat(res[: it + 1], w)
            est = estimate(step, rho)
            row[f"w{w}"] = {
                "rho_hat": rho,
                "estimate": est,
                "ratio": est / err if err else None,
            }
        if case == "cavity":
            at_snap = marchi_stations(config, mesh, u, v)
            row["metric_iteration_error"] = float(np.abs(at_snap - at_truth).max())
        else:
            row |= channel_readings(config, mesh, u)
            # Relative L2 of the difference on the metric's column, over its u_ref norm.
            _y, u_col, _u_ref = poiseuille_profiles(config, mesh, u)
            diff = np.sum((u_col - u_col_truth) ** 2) / np.sum(u_ref**2)
            row["metric_iteration_error"] = float(np.sqrt(diff))
        out["snapshots"].append(row)
    return out


def tight_truth(case: str, n: int) -> Path:
    """The case under the default rule to TIGHT_TRUTH_TOL, beside the probe's truth.

    Parameters
    ----------
    case : str
        "cavity" or "poiseuille".
    n : int
        Cells along x.

    Returns
    -------
    Path
        The ``.npz`` holding tol, u, v, outer, imbalance and the pressure
        solver version.

    Raises
    ------
    SystemExit
        If the cap stops the solve before TIGHT_TRUTH_TOL.

    Notes
    -----
    Solved once, and again if the file does not hold TIGHT_TRUTH_TOL as its
    tolerance or was not written by the current pressure solve.
    """
    path = OUT_DIR / f"{case_name(case, n)}_truth13.npz"
    stale = True
    if written_by_this_solver(path):
        with np.load(path) as saved:
            stale = "tol" not in saved.files or float(saved["tol"]) != TIGHT_TRUTH_TOL
    if stale:
        print(f"{path.stem}: solving to {TIGHT_TRUTH_TOL:.0e}", flush=True)
        config = case_config(case, n, TIGHT_TRUTH_TOL)
        mesh = Mesh(config)
        solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))
        u, v, _p = solver.solve_steady()
        if not solver.converged:
            raise SystemExit(f"{path.stem} did not reach {TIGHT_TRUTH_TOL:.0e}")
        worst = np.abs(solver.last_mass_imbalance).max()
        outer = len(solver.residual_history)
        np.savez(
            path,
            tol=TIGHT_TRUTH_TOL,
            u=u,
            v=v,
            outer=outer,
            imbalance=worst,
            pressure_solver_version=PRESSURE_SOLVER_VERSION,
        )
    return path


def rule_parameters(
    scale: float, flux: float, tols: tuple[float, float], version: int
) -> np.ndarray:
    """What a saved rule solve is reused under: the rule's inputs and both versions.

    Parameters
    ----------
    scale : float
        Largest prescribed boundary velocity, m/s.
    flux : float
        The solver's flux scale F, kg/s per unit depth.
    tols : tuple[float, float]
        ``iteration_error_tol`` and ``mass_imbalance_tol``.
    version : int
        The rule's version, read from the solver (``rule_version``).

    Returns
    -------
    np.ndarray
        scale, flux, both tolerances, RATE_WINDOW, the rule's version and
        PRESSURE_SOLVER_VERSION, shape [7].

    Notes
    -----
    A new condition changes no tolerance, so the rule's version is in the
    key; a new pressure solve changes every field beyond rounding, so
    PRESSURE_SOLVER_VERSION is too.
    """
    return np.array([scale, flux, *tols, RATE_WINDOW, version, PRESSURE_SOLVER_VERSION])


def verify_rule(case: str, n: int) -> dict:
    """One error_estimate solve at the default tolerances, its stop read against the truth.

    Parameters
    ----------
    case : str
        "cavity" or "poiseuille".
    n : int
        Cells along x.

    Returns
    -------
    dict
        The rule's outer count and seconds against the default rule's, the
        stop reason, the outer iteration from which each condition held, the
        estimate and summed imbalance at the stop, the true error relative to
        the velocity scale, and for the channel the metric at the stop and at
        the truth.

    Raises
    ------
    SystemExit
        If a fresh rule replaying the history does not stop where the solver
        did, or the recorded worst or signed sum at the stop is not the
        returned field's.

    Notes
    -----
    The corrector is wrapped as in the truth solve, so every iteration's
    imbalance is known and the wall time compares with the default rule's
    snapshot. A saved solve is reused only if its rule parameters, the
    rule's version and PRESSURE_SOLVER_VERSION match. The truth and the default
    rule's outer count and seconds are read through solve_truth, so a truth
    another solver wrote is solved again first (review 37 B1). Each condition
    is dated from the start of its final run. The channel is read against its
    TIGHT_TRUTH_TOL truth.
    """
    name, config = case_name(case, n), case_config(case, n, rule="error_estimate")
    mesh, path = Mesh(config), OUT_DIR / f"{name}_rule.npz"
    boundary = StaggeredBoundary(mesh, config)
    # The flux scale the solver built, so the stored parameters are the rule's own.
    solver = StaggeredSolver(mesh, config, boundary)
    scale, flux = boundary.get_max_boundary_velocity(), solver.flux_scale
    tols = (config.iteration_error_tol, config.mass_imbalance_tol)
    params = rule_parameters(scale, flux, tols, solver.rule_version)
    stale = True
    if path.exists():
        with np.load(path) as saved:
            stale = "params" not in saved.files or not np.array_equal(
                saved["params"], params
            )
    if stale:
        imbalance, _iterations, total, signed = instrument(solver)
        start = time.perf_counter()
        u, v, _p = solver.solve_steady()
        np.savez(
            path,
            params=params,
            u=u,
            v=v,
            seconds=time.perf_counter() - start,
            imbalance=np.array(imbalance),
            total=np.array(total),
            signed=np.array(signed),
            residual=np.array(solver.residual_history),
            reference_velocity=solver.reference_velocity,
            stop_reason=solver.stop_reason,
            returned=np.abs(solver.last_mass_imbalance).max(),
            returned_signed=solver.last_mass_imbalance.sum(),
        )
    print(f"{name}: {'solved' if stale else 'reused the saved solve'}", flush=True)
    with np.load(path) as saved:
        d = {k: saved[k] for k in saved.files}
    with np.load(solve_truth(case, n)) as saved:
        last = len(saved["level"]) - 1
        truth = (saved[f"u_{last}"], saved[f"v_{last}"])
        at_tol = saved["level"] == case_config(case, n).convergence_tol
        default_outer = int(saved["iteration"][np.flatnonzero(at_tol)[0]]) + 1
        default_seconds = float(saved["elapsed"][default_outer - 1])
    fluid = mesh.cell_type == FLUID
    out: dict = {}
    if case == "poiseuille":
        with np.load(tight_truth(case, n)) as saved:
            tight = (saved["u"], saved["v"])
        out["old_truth_gap"] = true_error(*truth, tight, fluid) / scale
        truth = tight
    res, imb, tot, net = d["residual"], d["imbalance"], d["total"], d["signed"]
    replay = ErrorEstimateRule(scale, flux, *tols)
    steps = res * float(d["reference_velocity"])
    # Each partial builds the recorded readings when the rule calls it.
    stops = [
        replay.update(
            s, partial(ImbalanceSummary, worst=w, absolute_sum=a, signed_sum=g)
        )
        for s, w, a, g in zip(steps, imb, tot, net, strict=True)
    ]
    controls = (imb[-1] == d["returned"], net[-1] == d["returned_signed"])
    if not (stops[-1] and not any(stops[:-1]) and all(controls)):
        raise SystemExit(f"{name}: a verify_rule control failed")
    # Outer iteration (1-based) from which each condition held to the stop.
    est = np.array(replay.estimate_history)
    held = (est < tols[0], imb < tols[1], tot / flux < tols[0], np.abs(net) < tols[1])
    since = [int(np.flatnonzero(~met)[-1]) + 2 if (~met).any() else 1 for met in held]
    out |= {"outer": len(res), "seconds": float(d["seconds"])}
    out |= {"default_outer": default_outer, "default_seconds": default_seconds}
    out |= {"stop_reason": str(d["stop_reason"]), "met_from": since}
    out |= {"estimate": float(est[-1]), "summed": float(tot[-1]) / flux}
    out["true_error_rel"] = true_error(d["u"], d["v"], truth, fluid) / scale
    out["imbalance"], out["signed_sum"] = float(imb[-1]), float(net[-1])
    if case == "poiseuille":
        out["metric"] = poiseuille_l2_error(config, mesh, d["u"]).value
        out["metric_truth"] = poiseuille_l2_error(config, mesh, truth[0]).value
    print(f"{name}: {out}", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the controls and the five solves, then analyse the saved fields into summary.json.

    Parameters
    ----------
    argv : list[str], optional
        Command-line arguments; ``sys.argv`` when omitted. With
        ``--verify-rule``, solve and read each case under error_estimate
        instead.

    Returns
    -------
    int
        Exit status, 0 when the run completes.

    Raises
    ------
    SystemExit
        If a control fails, the 80x40 channel does not reach TRUTH_TOL, or the
        solve time passes BUDGET_SECONDS.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify-rule", action="store_true")
    args = parser.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.verify_rule:
        verified = {case_name(c, n): verify_rule(c, n) for c, n in CASES}
        (OUT_DIR / "verify_rule.json").write_text(
            json.dumps(verified, indent=2), encoding="utf-8"
        )
        return 0
    summary: dict = {"truth_tol": TRUTH_TOL, "window": WINDOW, "control": {}}
    spent = 0.0
    for case, n in CASES:
        name = case_name(case, n)
        if name in CONTROLS:
            summary["control"][name] = control(case, n)
            spent += summary["control"][name]["seconds"]
        with np.load(solve_truth(case, n)) as saved:
            spent, reached = spent + float(saved["elapsed"][-1]), bool(saved["reached"])
        if name == "poiseuille_80x40" and not reached:
            raise SystemExit(
                f"{name} did not reach {TRUTH_TOL:.0e}; question 1 needs it"
            )
        if spent > BUDGET_SECONDS:
            raise SystemExit(f"solve time {spent:.0f} s passed the budget after {name}")
    summary["solve_seconds"] = spent
    summary["cases"] = {case_name(c, n): analyse(c, n) for c, n in CASES}
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"wrote summary.json; solve time {spent:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
