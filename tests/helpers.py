"""Helpers shared by the test files (review 64 S9).

The fixtures and comparisons more than one test file uses: the warm-start
fixtures (tests/test_warm_start.py, read by tests/test_io_manager.py) and
the bitwise comparisons of solves, budgets and recovery records. A test
file imports them from here rather than from another test file.
"""

import numpy as np
import yaml

from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.io_manager import RecoveryRecord
from src.mesh import Mesh
from src.solver_staggered import StaggeredSolver
from src.solver_transport import DEPOSIT_SURFACES, MassBudget
from validation.cases import case_path

FLOW_FIELDS = ("u", "v", "p", "k", "eps", "nu_t")


def channel_raw(cap: int = 5000, interval: int | None = None) -> dict:
    """VAL-001's channel on 40x20 under its own error_estimate rule, as a mapping.

    INTERVAL, when given, is ``solver.checkpoint_interval``.
    """
    raw = yaml.safe_load(case_path("poiseuille").read_text(encoding="utf-8"))
    raw["domain"]["nx"], raw["domain"]["ny"] = 40, 20
    raw["solver"]["max_simple_iter"] = cap
    if interval is not None:
        raw["solver"]["checkpoint_interval"] = interval
    return raw


def channel_config(cap: int = 5000, interval: int | None = None) -> SimConfig:
    """``channel_raw`` loaded."""
    return SimConfig.from_dict(channel_raw(cap, interval))


def product_config(
    cap: int = 5000, inlet: float | None = None, interval: int | None = None
) -> SimConfig:
    """``product_raw`` loaded."""
    return SimConfig.from_dict(product_raw(cap, inlet, interval))


def product_raw(
    cap: int = 5000, inlet: float | None = None, interval: int | None = None
) -> dict:
    """The product file on 40x15, its imbalance tolerance by the file's formula."""
    raw = yaml.safe_load(case_path("product").read_text(encoding="utf-8"))
    raw["domain"]["nx"], raw["domain"]["ny"] = 40, 15
    cell = (raw["domain"]["width"] / 40) * (raw["domain"]["height"] / 15)
    raw["solver"]["mass_imbalance_tol"] = (
        1e-4 * raw["fluid"]["density"] * cell / raw["solver"]["t_end"]
    )
    raw["solver"]["max_simple_iter"] = cap
    if inlet is not None:
        raw["boundaries"]["hepa_supply"]["velocity"] = inlet
    if interval is not None:
        raw["solver"]["checkpoint_interval"] = interval
    return raw


SOLVE_FIXTURES = {"channel": channel_config, "product": product_config}
RAW_SOLVE_FIXTURES = {"channel": channel_raw, "product": product_raw}


def staggered_solver(config: SimConfig) -> StaggeredSolver:
    mesh = Mesh(config)
    return StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))


def solved_solver(config: SimConfig, **kwargs: object) -> StaggeredSolver:
    solver = staggered_solver(config)
    solver.solve_steady(**kwargs)
    return solver


def same_solves(a: StaggeredSolver, b: StaggeredSolver) -> None:
    """Every field, the history and the stop of two solves, bit for bit."""
    assert a.flow_state is not None and b.flow_state is not None
    for name in FLOW_FIELDS:
        x, y = getattr(a.flow_state, name), getattr(b.flow_state, name)
        assert (x is None) == (y is None), name
        if x is not None:
            assert x.tobytes() == y.tobytes(), name
    assert a.residual_history == b.residual_history
    assert (a.stop_reason, a.converged) == (b.stop_reason, b.converged)
    assert a.flow_state.outer == b.flow_state.outer == len(b.residual_history)
    assert a.pressure_cap_hits == b.pressure_cap_hits


def same_budgets(a: tuple[MassBudget, ...], b: tuple[MassBudget, ...]) -> None:
    """Two sequences of budgets, every term bit for bit, None as None."""
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        for name in ("initial", "inflow", "outflow", "source", "current"):
            p, q = getattr(x, name), getattr(y, name)
            assert (p is None) == (q is None), name
            if p is not None:
                assert np.float64(p).tobytes() == np.float64(q).tobytes(), name
        assert list(x.deposited) == list(y.deposited) == list(DEPOSIT_SURFACES)
        for surface in DEPOSIT_SURFACES:
            assert (
                np.float64(x.deposited[surface]).tobytes()
                == np.float64(y.deposited[surface]).tobytes()
            ), surface


def same_recovery(a: RecoveryRecord, b: RecoveryRecord) -> None:
    """Every field of two records, arrays bit for bit."""
    assert a.sensor_names == b.sensor_names
    assert (a.hold_time, a.held) == (b.hold_time, b.held)
    for name in (
        "t_final",
        "air_change_rate",
        "well_mixed_time",
        "room_recovery_time",
    ):
        assert (
            np.float64(getattr(a, name)).tobytes()
            == np.float64(getattr(b, name)).tobytes()
        ), name
    for name in (
        "thresholds",
        "initial",
        "recovery_time",
        "sensor_time",
        "mean_time",
        "ratio",
        "local_recovery",
        "local_recovery_max",
    ):
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype == np.float64, name
        assert x.shape == y.shape and x.tobytes() == y.tobytes(), name
    assert a.defined.dtype == b.defined.dtype == np.bool_
    assert a.defined.tolist() == b.defined.tolist()
