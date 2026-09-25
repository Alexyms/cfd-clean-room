"""What the stopping rule should measure: iteration error, its estimate, continuity.

Solves VAL-001 at 40x20 and 80x40 and the cavity at 20x20, 40x40 and 80x80 once
each with the committed settings except convergence_tol, set in memory to
TRUTH_TOL, and the iteration cap. The field at TRUTH_TOL is the truth for that
case and grid. The solver's corrector is wrapped on the instance to record each
outer iteration's worst per-cell imbalance and sweep count; src/ is not edited.
u and v are kept at each quarter decade of residual from 1e-5 to TRUTH_TOL and at
the case's own tolerance. The same-computation control runs first and stops the
script if an unwrapped solve at the committed tolerance differs from the wrapped
snapshot there. Fields go to results/stopping_probe/ (gitignored) and are never
solved again once saved; the analysis writes summary.json there.

    python scripts/stopping_probe.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# self_convergence already imports the solver, mesh, config and Marchi names
# this script uses, so they are read from it: sc.Mesh, sc.MARCHI_U_ROWS.
import self_convergence as sc  # noqa: E402 -- path set above

from validation.metrics import (  # noqa: E402 -- path set above
    _inlet_velocity,
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


def case_name(case: str, n: int) -> str:
    """File stem of a case at n cells along x (n / 2 up the channel)."""
    return f"{case}_{n}x{n if case == 'cavity' else n // 2}"


def case_config(case: str, n: int, tol: float | None = None) -> sc.SimConfig:
    """The committed case at n cells along x; tolerance and cap set in memory if tol is given."""
    raw = yaml.safe_load(sc.case_path(case).read_text(encoding="utf-8"))
    raw["domain"]["nx"], raw["domain"]["ny"] = n, (n if case == "cavity" else n // 2)
    if tol is not None:
        raw["solver"]["convergence_tol"] = tol
        raw["solver"]["max_simple_iter"] = MAX_OUTER[case]
    return sc.SimConfig.from_dict(raw)


def instrument(solver: sc.StaggeredSolver) -> tuple[list[float], list[int]]:
    """Record each outer iteration's worst per-cell imbalance and sweep count.

    The wrapper returns the corrector's own result object, so the solve is
    unchanged; the same-computation control checks that bitwise.
    """
    corrector, correct = solver._corrector, solver._corrector.correct
    imbalance: list[float] = []
    sweeps: list[int] = []

    def recorded(prediction: MomentumPrediction, p: np.ndarray) -> PressureCorrection:
        result = correct(prediction, p)
        imbalance.append(
            float(np.abs(corrector.mass_imbalance(result.u, result.v)).max())
        )
        sweeps.append(result.sweeps)
        return result

    corrector.correct = recorded
    return imbalance, sweeps


def solve_truth(case: str, n: int) -> Path:
    """Solve to TRUTH_TOL wrapped, keeping snapshots; an existing file is returned unsolved."""
    path = OUT_DIR / f"{case_name(case, n)}.npz"
    if path.exists():
        return path
    config = case_config(case, n, TRUTH_TOL)
    mesh = sc.Mesh(config)
    solver = sc.StaggeredSolver(mesh, config, sc.StaggeredBoundary(mesh, config))
    imbalance, sweeps = instrument(solver)
    levels = sorted({*LEVELS, case_config(case, n).convergence_tol}, reverse=True)
    taken: list[tuple[float, int]] = []
    snaps: dict[str, np.ndarray] = {}
    elapsed: list[float] = []
    start = time.perf_counter()

    def observe(state: sc.IterationState) -> None:
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
        sweeps=np.array(sweeps),
        elapsed=np.array(elapsed),
        reference_velocity=solver.reference_velocity,
        reached=not levels,
        **snaps,
    )
    print(
        f"{path.stem}: {len(elapsed)} outer, {elapsed[-1]:.0f} s, reached {not levels}"
    )
    return path


def control(case: str, n: int) -> dict:
    """An unwrapped solve at the committed tolerance against the wrapped snapshot there.

    Raises SystemExit unless u and v are bitwise equal and the outer counts agree.
    """
    name, config = case_name(case, n), case_config(case, n)
    path, mesh = OUT_DIR / f"{name}_control.npz", sc.Mesh(config)
    if not path.exists():
        solver = sc.StaggeredSolver(mesh, config, sc.StaggeredBoundary(mesh, config))
        start = time.perf_counter()
        u, v, _p = solver.solve_steady()
        seconds = time.perf_counter() - start
        np.savez(path, u=u, v=v, outer=len(solver.residual_history), seconds=seconds)
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
    """exp of the least-squares slope of log(residual) over the last window entries; NaN if short."""
    if len(history) < window:
        return float("nan")
    return float(np.exp(np.polyfit(np.arange(window), np.log(history[-window:]), 1)[0]))


def estimate(step: float, rho: float) -> float:
    """Iteration error left by a geometric iteration after a step: step rho / (1 - rho).

    Infinite for a rate of 1 or more; NaN when there is no rate.
    """
    return float("inf") if rho >= 1.0 else step * rho / (1.0 - rho)


def true_error(
    u: np.ndarray,
    v: np.ndarray,
    truth: tuple[np.ndarray, np.ndarray],
    fluid: np.ndarray,
) -> float:
    """Largest abs difference from the truth in u or v over the FLUID cells."""
    return float(
        max(np.abs(u - truth[0])[fluid].max(), np.abs(v - truth[1])[fluid].max())
    )


def channel_readings(config: sc.SimConfig, mesh: sc.Mesh, u: np.ndarray) -> dict:
    """poiseuille_l2_error at nx/2, nx/4 and 3nx/4, and the range of u_num / u_ref at nx/2.

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


def wall_stencil_profile(config: sc.SimConfig, mesh: sc.Mesh) -> np.ndarray:
    """Fully developed discrete u under the half-cell wall stencil, all ny rows (INFERRED).

    The interior three-point difference is exact for a parabola; the wall row's
    (u_0 - 0) / (dy / 2) is not, and shifts it by dy^2 / 4. Scaled so the
    midpoint sum carries the inflow.
    """
    y, dy, height = np.asarray(mesh.yc), np.asarray(mesh.dy_cell), config.room_height
    shape = y * (height - y) + dy**2 / 4.0
    return shape * _inlet_velocity(config) * height / float(np.sum(shape * dy))


def marchi_stations(
    config: sc.SimConfig, mesh: sc.Mesh, u: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """u then v at Marchi's 30 stations, as marchi_comparison takes them, over the lid speed."""
    lines = sc.face_profiles(mesh, u, v, sc._lid_velocity(config))
    rows = (sc.MARCHI_U_ROWS, sc.MARCHI_V_ROWS)
    return np.concatenate(
        [
            sc.lagrange(*ln, np.array([r[0] for r in rs]))
            for ln, rs in zip(lines, rows, strict=True)
        ]
    )


def analyse(case: str, n: int) -> dict:
    """Every snapshot of one case against its truth, and the truth against its reference."""
    config = case_config(case, n)
    mesh = sc.Mesh(config)
    fluid = mesh.cell_type == sc.FLUID
    with np.load(OUT_DIR / f"{case_name(case, n)}.npz") as saved:
        d = {k: saved[k] for k in saved.files}
    res, ref_vel, imb = d["residual"], float(d["reference_velocity"]), d["imbalance"]
    phys = sc._lid_velocity(config) if case == "cavity" else _inlet_velocity(config)
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
        return out
    last = len(d["level"]) - 1
    truth = (d[f"u_{last}"], d[f"v_{last}"])
    est_truth = estimate(res[-1] * ref_vel, rho_hat(res, WINDOW))
    out["truth"] = {"rho_hat": rho_hat(res, WINDOW), "remaining_error": est_truth}
    if case == "cavity":
        marchi = np.array([r[1] for r in (*sc.MARCHI_U_ROWS, *sc.MARCHI_V_ROWS)])
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
        row |= {"imbalance": float(imb[it]), "sweeps": int(d["sweeps"][it])}
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


def main() -> int:
    """Run the controls and the five solves, then analyse the saved fields into summary.json."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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
