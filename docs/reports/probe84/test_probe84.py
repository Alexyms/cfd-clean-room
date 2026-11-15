"""Checks of prompt 84's stage timers on functions of known cost.

    python -m pytest docs/reports/probe84/test_probe84.py -q

The clock charges every nanosecond to exactly one label path; these tests
time busy waits of known length, nested calls and line regions, and check
that the solver's own markers are found once each.
"""

import sys
from pathlib import Path
from time import perf_counter_ns

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import probe84  # noqa: E402 -- follows sys.path.insert

MS = 1_000_000
# A busy wait overruns by the loop's last iteration and the clock's own reads;
# a millisecond is far above both on this machine.
SLACK = 1 * MS


def spin(ns: int) -> int:
    """Busy-wait NS nanoseconds; returns NS."""
    end = perf_counter_ns() + ns
    while perf_counter_ns() < end:
        pass
    return ns


class Toy:
    """A stage with a nested call and two inline regions."""

    def outer(self) -> int:
        spin(4 * MS)
        return self.inner() + 1

    def inner(self) -> int:
        spin(6 * MS)
        return 41

    def inline(self) -> int:
        a = spin(3 * MS)
        b = spin(5 * MS)
        c = self.inner()
        return a + b + c


def toy_specs() -> list[probe84.Wrap]:
    return [
        probe84.Wrap(Toy, "outer", "outer"),
        probe84.Wrap(Toy, "inner", "inner"),
        probe84.Wrap(Toy, "inline", "inline"),
    ]


TOY_REGIONS = {
    "inline": [("a = spin(3 * MS)", "first"), ("b = spin(5 * MS)", "second")],
}


def charges(clock: probe84.Clock) -> dict[tuple[str, ...], int]:
    return {tuple(p): ns for p, ns in clock.record()["charge"]}


def installed_toy() -> probe84.Clock:
    return probe84.install(
        toy_specs(), {"inline": Toy.inline}, TOY_REGIONS, {"inline": "start"}
    )


@pytest.fixture(autouse=True)
def clean() -> None:
    """Leave no wrapper installed, whatever a test does."""
    yield
    if probe84.CLOCK is not None:
        probe84.uninstall()


def test_charges_sum_to_the_total_and_nesting_is_exclusive() -> None:
    original = Toy.outer
    installed_toy()
    assert Toy().outer() == 42
    clock = probe84.uninstall()
    got = charges(clock)
    assert sum(got.values()) == clock.record()["total_ns"]
    assert 4 * MS <= got[("probe", "outer")] < 4 * MS + SLACK
    assert 6 * MS <= got[("probe", "outer", "inner")] < 6 * MS + SLACK
    assert Toy.outer is original


def test_regions_relabel_the_running_frame_and_calls_nest_under_them() -> None:
    installed_toy()
    assert Toy().inline() == 3 * MS + 5 * MS + 41
    got = charges(probe84.uninstall())
    assert 3 * MS <= got[("probe", "inline:first")] < 3 * MS + SLACK
    # The call on the third line runs while the second region is current.
    assert 5 * MS <= got[("probe", "inline:second")] < 5 * MS + SLACK
    assert 6 * MS <= got[("probe", "inline:second", "inner")] < 6 * MS + SLACK
    assert got.get(("probe", "inline:start"), 0) < SLACK


def test_unwrapped_results_and_calls_counted() -> None:
    before = Toy().inline()
    clock = installed_toy()
    Toy().inline()
    Toy().inline()
    probe84.uninstall()
    counted = {tuple(p): n for p, n in clock.record()["calls"]}
    assert counted[("probe", "inline:start")] == 2
    assert counted[("probe", "inline:second", "inner")] == 2
    assert Toy().inline() == before


def test_reinstall_after_uninstall_times_regions_again() -> None:
    installed_toy()
    Toy().inline()
    probe84.uninstall()
    installed_toy()
    Toy().inline()
    got = charges(probe84.uninstall())
    assert 3 * MS <= got[("probe", "inline:first")] < 3 * MS + SLACK


def test_marker_lines_refuses_an_absent_or_repeated_marker() -> None:
    with pytest.raises(RuntimeError, match="found 0 times"):
        probe84.marker_lines(Toy.inline, [("no such line", "x")])
    with pytest.raises(RuntimeError, match="found 5 times"):
        probe84.marker_lines(Toy.inline, [("", "every line")])


def test_the_solver_markers_are_found_once_each() -> None:
    from src.solver_staggered import StaggeredSolver
    from src.solver_transport import TransportSolver

    for name, function in (
        ("solve_steady", StaggeredSolver.solve_steady),
        ("solve_timestep", TransportSolver.solve_timestep),
    ):
        lines = probe84.marker_lines(function, probe84.REGIONS[name])
        assert len(lines) == len(probe84.REGIONS[name])


@pytest.mark.parametrize(
    ("path", "stage"),
    [
        (
            ("probe", "march", "transport3:implicit", "implicit", "implicit_solve"),
            "class.implicit",
        ),
        (("probe", "march", "transport0:checks", "stable_dt"), "class.remainder"),
        (("probe", "march", "transport4:booking", "in_domain"), "class.booking"),
        (("probe", "march", "observe", "monitor_update"), "step.monitor"),
        (("probe", "march", "flush"), "step.frames"),
        (("probe", "flow:momentum", "stencil_mu"), "flow.boundary"),
        (("probe", "flow:momentum", "predict", "assemble"), "flow.momentum"),
        (
            (
                "probe",
                "flow:stopping",
                "rule_update",
                "imbalance_summary",
                "mass_imbalance",
            ),
            "flow.stopping",
        ),
        (("probe", "flow:pressure", "correct", "mass_imbalance"), "flow.pressure"),
        (
            ("probe", "flow:turbulence", "turbulence_step", "turbulence_conditions"),
            "flow.boundary",
        ),
        (("probe", "run", "flow:loop", "steady_checkpoint"), "flow.io"),
        (("probe",), "outside"),
    ],
)
def test_stage_of_path(path: tuple[str, ...], stage: str) -> None:
    assert probe84.stage_of_path(path) == stage
