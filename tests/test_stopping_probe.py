"""Unit tests for the estimator and the true-error pipeline of scripts/stopping_probe.py.

Each check is shown to catch the defect it exists for: the rate check rejects a
trailing window off by one iteration and a fit to the raw residual instead of
its log, and the true-error check tells a FLUID cell from a non-FLUID one.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stopping_probe  # noqa: E402 -- scripts/ is not a package; path set above

from src.config import SimConfig  # noqa: E402 -- follows sys.path.insert
from src.mesh import FLUID, Mesh  # noqa: E402 -- follows sys.path.insert
from validation.cases import load_case  # noqa: E402 -- follows sys.path.insert

WINDOW = 100


def rate_check(estimator: Callable[[np.ndarray, int], float]) -> bool:
    """True when an estimator recovers rho from two synthetic residual histories.

    A geometric history C rho^n has one rate, which any fit to its log returns.
    A history whose log is a n + b n^2 has a rate that drifts, and a straight-line
    fit to its log over indices m - (w - 1) / 2 to m + (w - 1) / 2 has slope
    exactly a + 2 b m, so a window shifted or resized by one index moves the
    answer by about b.
    """
    for rho in (0.5, 0.99, 0.999):
        history = 1e-3 * rho ** np.arange(400.0)
        if abs(estimator(history, WINDOW) - rho) > 1e-12:
            return False
    a, b, n = -0.01, 1e-5, np.arange(300.0)
    centre = n[-1] - (WINDOW - 1) / 2.0
    expected = np.exp(a + 2.0 * b * centre)
    return abs(estimator(np.exp(a * n + b * n**2), WINDOW) - expected) < 1e-12


def _fit(values: np.ndarray) -> float:
    return float(np.exp(np.polyfit(np.arange(len(values)), values, 1)[0]))


PLANTED = {
    "one entry too many": lambda h, w: _fit(np.log(h[-w - 1 :])),
    "one entry too few": lambda h, w: _fit(np.log(h[-w + 1 :])),
    "shifted back by one": lambda h, w: _fit(np.log(h[-w - 1 : -1])),
    "fit to raw values": lambda h, w: _fit(h[-w:]),
}


@pytest.mark.unit
def test_rho_hat_recovers_known_rates() -> None:
    """rho_hat returns the rate of a geometric history and of a drifting one."""
    assert rate_check(stopping_probe.rho_hat)


@pytest.mark.unit
@pytest.mark.parametrize("defect", sorted(PLANTED))
def test_rate_check_rejects_a_planted_defect(defect: str) -> None:
    """Each planted estimator defect fails the check rho_hat passes."""
    assert not rate_check(PLANTED[defect])


@pytest.mark.unit
def test_rho_hat_is_nan_on_a_history_shorter_than_the_window() -> None:
    """Too short a history gives no rate rather than a rate from fewer points."""
    assert np.isnan(stopping_probe.rho_hat(np.ones(WINDOW - 1), WINDOW))


@pytest.mark.unit
def test_rho_hat_reads_a_history_exactly_as_long_as_the_window() -> None:
    """A history of exactly window entries is enough for a rate."""
    history = 1e-3 * 0.9 ** np.arange(float(WINDOW))
    assert stopping_probe.rho_hat(history, WINDOW) == pytest.approx(0.9, abs=1e-12)


@pytest.mark.unit
def test_estimate_is_the_geometric_tail() -> None:
    """step rho / (1 - rho) is the sum of every later step of a geometric iteration."""
    step, rho = 2.0**-20, 0.75
    tail = sum(step * rho**k for k in range(1, 200))
    assert stopping_probe.estimate(step, rho) == pytest.approx(tail, rel=1e-14)
    assert stopping_probe.estimate(step, 1.0) == float("inf")
    assert np.isnan(stopping_probe.estimate(step, float("nan")))


@pytest.fixture
def cavity() -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """The 20x20 cavity's FLUID mask and a dyadic truth, so differences are exact."""
    mesh = Mesh(load_case("cavity", grid=(20, 20)))
    rng = np.random.default_rng(23)
    truth = rng.integers(-1024, 1024, size=(2, 20, 20)) / 1024.0
    return mesh.cell_type == FLUID, (truth[0], truth[1])


@pytest.mark.unit
@pytest.mark.parametrize("component", [0, 1])
def test_true_error_returns_a_planted_offset_in_a_fluid_cell(
    cavity: tuple[np.ndarray, tuple[np.ndarray, np.ndarray]], component: int
) -> None:
    """Truth plus 2^-20 in one FLUID cell of u or v reads as exactly 2^-20."""
    fluid, truth = cavity
    assert fluid[5, 7]
    planted = [truth[0].copy(), truth[1].copy()]
    planted[component][5, 7] += 2.0**-20
    assert stopping_probe.true_error(*planted, truth, fluid) == 2.0**-20


@pytest.mark.unit
def test_true_error_reads_a_negative_offset_as_its_size(
    cavity: tuple[np.ndarray, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Truth minus 2^-20 in one FLUID cell of u, then of v, reads as 2^-20."""
    fluid, truth = cavity
    for component in (0, 1):
        planted = [truth[0].copy(), truth[1].copy()]
        planted[component][5, 7] -= 2.0**-20
        assert stopping_probe.true_error(*planted, truth, fluid) == 2.0**-20


@pytest.mark.unit
@pytest.mark.parametrize("component", [0, 1])
def test_true_error_ignores_an_offset_outside_the_fluid(
    cavity: tuple[np.ndarray, tuple[np.ndarray, np.ndarray]], component: int
) -> None:
    """The same offset in the BOUNDARY ring reads as zero."""
    fluid, truth = cavity
    assert not fluid[0, 7]
    planted = [truth[0].copy(), truth[1].copy()]
    planted[component][0, 7] += 2.0**-20
    assert stopping_probe.true_error(*planted, truth, fluid) == 0.0


@pytest.mark.unit
def test_rule_parameters_change_with_the_rule_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A saved rule solve is solved again under another RULE_VERSION.

    Defect caught: the version dropped from the stored parameters (test 28 S1).
    """
    before = stopping_probe.rule_parameters(0.1, 0.05, (1e-6, 1e-10))
    monkeypatch.setattr(stopping_probe, "RULE_VERSION", stopping_probe.RULE_VERSION + 1)
    after = stopping_probe.rule_parameters(0.1, 0.05, (1e-6, 1e-10))
    assert not np.array_equal(before, after)


@pytest.mark.unit
def test_rule_parameters_change_with_the_pressure_solver_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A saved rule solve is solved again under another PRESSURE_SOLVER_VERSION.

    ECR-003 section 7.1: a field written by the weighted Jacobi sweep must
    not be served as the conjugate gradient solve's. Defect caught: the
    solver's version dropped from the stored parameters.
    """
    before = stopping_probe.rule_parameters(0.1, 0.05, (1e-6, 1e-10))
    monkeypatch.setattr(
        stopping_probe,
        "PRESSURE_SOLVER_VERSION",
        stopping_probe.PRESSURE_SOLVER_VERSION + 1,
    )
    after = stopping_probe.rule_parameters(0.1, 0.05, (1e-6, 1e-10))
    assert not np.array_equal(before, after)
    assert before[-1] == stopping_probe.PRESSURE_SOLVER_VERSION - 1


class _BuiltError(Exception):
    """Raised by the sentinel solver: a saved solve was not reused.

    Its one argument is the configuration of the solve that was asked for,
    so a test can tell the truth's solve from the rule's.
    """


STALE = [({}, "no-version"), ({"pressure_solver_version": 1}, "jacobi-era")]
CURRENT = {"pressure_solver_version": stopping_probe.PRESSURE_SOLVER_VERSION}


def _refuse_to_solve(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point OUT_DIR at tmp_path and make every solve raise _BuiltError.

    The solver is still built, since verify_rule reads its flux scale before
    deciding anything; only solve_steady refuses.
    """

    class Sentinel(stopping_probe.sc.StaggeredSolver):
        def __init__(self, mesh: Mesh, config: SimConfig, boundary: object) -> None:
            super().__init__(mesh, config, boundary)
            self.asked = config

        def solve_steady(self, *args: object, **kwargs: object) -> NoReturn:
            raise _BuiltError(self.asked)

    monkeypatch.setattr(stopping_probe, "OUT_DIR", tmp_path)
    monkeypatch.setattr(stopping_probe.sc, "StaggeredSolver", Sentinel)


def _save_truth(
    path: Path,
    saved: dict,
    u: np.ndarray,
    v: np.ndarray,
    case: str = "cavity",
    n: int = 20,
    elapsed: np.ndarray | None = None,
) -> None:
    """A truth file as solve_truth writes it, its second snapshot at the case tolerance.

    Complete, so a reader that skipped the version check would read it
    without error rather than fail on a missing key. ``elapsed`` defaults
    to a few seconds.
    """
    tol = stopping_probe.case_config(case, n).convergence_tol
    np.savez(
        path,
        level=np.array([10.0 * tol, tol]),
        iteration=np.array([3, 5]),
        residual=np.full(6, tol),
        imbalance=np.zeros(6),
        iterations=np.full(6, 9),
        elapsed=np.arange(6.0) if elapsed is None else elapsed,
        reference_velocity=1.0,
        reached=True,
        u_0=u,
        v_0=v,
        u_1=u,
        v_1=v,
        **saved,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("saved", "reused"),
    [*((s, False) for s, _ in STALE), (CURRENT, True)],
    ids=[*(i for _, i in STALE), "this-solver"],
)
def test_saved_solves_are_reused_only_when_this_solver_wrote_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, saved: dict, reused: bool
) -> None:
    """The truth and the tight truth each re-solve on a missing or old version.

    The decision is tested without a solve: solving raises, so a file that
    is reused returns, and one that is not reaches the sentinel. The control
    and the two readers of the truth have their own tests below, and the
    rule file is keyed by rule_parameters above.
    """
    _refuse_to_solve(monkeypatch, tmp_path)
    name = stopping_probe.case_name("cavity", 20)
    np.savez(tmp_path / f"{name}.npz", u=np.zeros(1), **saved)
    np.savez(
        tmp_path / f"{name}_truth13.npz",
        tol=stopping_probe.TIGHT_TRUTH_TOL,
        u=np.zeros(1),
        **saved,
    )
    for reuse in (
        lambda: stopping_probe.solve_truth("cavity", 20),
        lambda: stopping_probe.tight_truth("cavity", 20),
    ):
        if reused:
            assert reuse().exists()
        else:
            with pytest.raises(_BuiltError):
                reuse()
    assert not stopping_probe.written_by_this_solver(tmp_path / "absent.npz")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("saved", "reused"),
    [*((s, False) for s, _ in STALE), (CURRENT, True)],
    ids=[*(i for _, i in STALE), "this-solver"],
)
def test_control_is_reused_only_when_this_solver_wrote_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, saved: dict, reused: bool
) -> None:
    """The control re-solves on a missing or old version, beside a current truth.

    Test 37 T-B1 (a): with the truth stale too, solve_truth reached the
    sentinel whatever the control decided, so the control's own check was
    never tested. Here the truth is this solver's and agrees with the control
    bitwise, so only that check stands between the call and a reused field.
    Defect caught: the control reused whenever its file exists.
    """
    _refuse_to_solve(monkeypatch, tmp_path)
    name = stopping_probe.case_name("cavity", 20)
    u, v = np.full((20, 20), 0.25), np.full((20, 20), -0.5)
    _save_truth(tmp_path / f"{name}.npz", CURRENT, u, v)
    np.savez(tmp_path / f"{name}_control.npz", u=u, v=v, outer=6, seconds=1.0, **saved)
    if reused:
        out = stopping_probe.control("cavity", 20)
        assert out == {"outer": 6, "seconds": 1.0, "bitwise_equal": True}
    else:
        with pytest.raises(_BuiltError):
            stopping_probe.control("cavity", 20)


@pytest.mark.unit
@pytest.mark.parametrize("saved", [s for s, _ in STALE], ids=[i for _, i in STALE])
def test_readers_of_the_truth_re_solve_a_truth_this_solver_did_not_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, saved: dict
) -> None:
    """verify_rule and analyse each solve a stale truth again before reading it.

    Review 37 B1: verify_rule read the truth file directly, so under
    --verify-rule a weighted-Jacobi truth was served as this solver's. Here
    the rule file is current, so verify_rule reaches the truth without a
    solve of its own, and the sentinel must be reached by the truth's solve:
    the solve asked for is at TRUTH_TOL under velocity_step, not the rule's.
    Defect caught: either reader loading the truth file without solve_truth.
    """
    _refuse_to_solve(monkeypatch, tmp_path)
    name = stopping_probe.case_name("cavity", 20)
    zeros = np.zeros((20, 20))
    _save_truth(tmp_path / f"{name}.npz", saved, zeros, zeros)
    config = stopping_probe.case_config("cavity", 20, rule="error_estimate")
    mesh = stopping_probe.sc.Mesh(config)
    boundary = stopping_probe.sc.StaggeredBoundary(mesh, config)
    flux = stopping_probe.sc.StaggeredSolver(mesh, config, boundary).flux_scale
    tols = (config.iteration_error_tol, config.mass_imbalance_tol)
    params = stopping_probe.rule_parameters(
        boundary.get_max_boundary_velocity(), flux, tols
    )
    np.savez(tmp_path / f"{name}_rule.npz", params=params)
    for reader in (stopping_probe.verify_rule, stopping_probe.analyse):
        with pytest.raises(_BuiltError) as asked:
            reader("cavity", 20)
        assert asked.value.args[0].convergence_tol == stopping_probe.TRUTH_TOL
        assert asked.value.args[0].stopping_rule == "velocity_step"


def _poiseuille_rule_files(tmp_path: Path, case: str, n: int) -> str:
    """Write the rule file verify_rule reads, current for a case; return the case's name."""
    name = stopping_probe.case_name(case, n)
    config = stopping_probe.case_config(case, n, rule="error_estimate")
    mesh = stopping_probe.sc.Mesh(config)
    boundary = stopping_probe.sc.StaggeredBoundary(mesh, config)
    flux = stopping_probe.sc.StaggeredSolver(mesh, config, boundary).flux_scale
    tols = (config.iteration_error_tol, config.mass_imbalance_tol)
    params = stopping_probe.rule_parameters(
        boundary.get_max_boundary_velocity(), flux, tols
    )
    np.savez(tmp_path / f"{name}_rule.npz", params=params)
    return name


@pytest.mark.unit
@pytest.mark.parametrize("saved", [s for s, _ in STALE], ids=[i for _, i in STALE])
def test_verify_rule_reads_the_tight_truth_through_the_identity_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, saved: dict
) -> None:
    """A channel's tight truth from another solver is solved again before it is read.

    Test 37b T-B1 (TT1): the readers test above stops at the cavity, where
    verify_rule never reads a tight truth. Here a Poiseuille case has a
    current truth and rule file and a stale ``_truth13`` beside them, so the
    only read left is the tight truth's, and the sentinel must be reached at
    TIGHT_TRUTH_TOL. Defect caught: verify_rule loading the file directly,
    which scores the channel against a Jacobi-era truth without a sound
    (review 37 B1 for the channel).
    """
    _refuse_to_solve(monkeypatch, tmp_path)
    case, n = "poiseuille", 40
    name = _poiseuille_rule_files(tmp_path, case, n)
    zeros = np.zeros((20, 40))
    _save_truth(tmp_path / f"{name}.npz", CURRENT, zeros, zeros, case, n)
    np.savez(
        tmp_path / f"{name}_truth13.npz",
        tol=stopping_probe.TIGHT_TRUTH_TOL,
        u=zeros,
        v=zeros,
        **saved,
    )
    with pytest.raises(_BuiltError) as asked:
        stopping_probe.verify_rule(case, n)
    assert asked.value.args[0].convergence_tol == stopping_probe.TIGHT_TRUTH_TOL


@pytest.mark.unit
@pytest.mark.parametrize("saved", [s for s, _ in STALE], ids=[i for _, i in STALE])
def test_control_reads_its_truth_through_the_identity_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, saved: dict
) -> None:
    """The control re-solves a stale truth even beside a current control file.

    Test 37b T-B1 (CT1): the control's own test keeps the truth current by
    design, so the control's read of the truth was never tried against a
    stale one. Defect caught: the control loading the truth file directly,
    which compares a conjugate gradient solve with a stale snapshot.
    """
    _refuse_to_solve(monkeypatch, tmp_path)
    name = stopping_probe.case_name("cavity", 20)
    u, v = np.full((20, 20), 0.25), np.full((20, 20), -0.5)
    _save_truth(tmp_path / f"{name}.npz", saved, u, v)
    np.savez(
        tmp_path / f"{name}_control.npz", u=u, v=v, outer=6, seconds=1.0, **CURRENT
    )
    with pytest.raises(_BuiltError) as asked:
        stopping_probe.control("cavity", 20)
    assert asked.value.args[0].convergence_tol == stopping_probe.TRUTH_TOL


@pytest.mark.unit
@pytest.mark.parametrize("saved", [s for s, _ in STALE], ids=[i for _, i in STALE])
def test_main_reads_its_budget_through_the_identity_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, saved: dict
) -> None:
    """main's budget reads the truth through solve_truth, so a stale one is solved again.

    Test 37b T-B1 (MN1), the third read, which the tester left to a
    Suggestion. One non-control case is run, with a stale truth whose elapsed
    time is ten times the budget: read directly, main would stop on the
    budget (SystemExit); read through solve_truth it reaches the sentinel
    first. Defect caught: main loading ``{name}.npz`` for its budget.
    """
    _refuse_to_solve(monkeypatch, tmp_path)
    monkeypatch.setattr(stopping_probe, "CASES", (("poiseuille", 40),))
    name = stopping_probe.case_name("poiseuille", 40)
    assert name not in stopping_probe.CONTROLS
    zeros = np.zeros((20, 40))
    over_budget = np.full(6, 10.0 * stopping_probe.BUDGET_SECONDS)
    _save_truth(
        tmp_path / f"{name}.npz", saved, zeros, zeros, "poiseuille", 40, over_budget
    )
    with pytest.raises(_BuiltError) as asked:
        stopping_probe.main([])
    assert asked.value.args[0].convergence_tol == stopping_probe.TRUTH_TOL


@pytest.mark.unit
@pytest.mark.parametrize("tol", [None, 1e-11])
def test_case_config_without_a_rule_is_velocity_step(tol: float | None) -> None:
    """Truths and snapshots were solved under velocity_step, and the channel file now
    names error_estimate; a named rule still reaches the solver block."""
    assert stopping_probe.case_config("poiseuille", 40, tol).stopping_rule == (
        "velocity_step"
    )
    named = stopping_probe.case_config("poiseuille", 40, rule="error_estimate")
    assert named.stopping_rule == "error_estimate"
