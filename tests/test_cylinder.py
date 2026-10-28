"""VAL-022, the square cylinder (ADR-014 F, step 6b): the suite's fast tests.

The three rows themselves are harness rows (``python scripts/benchmark.py
--cases val022_re40 val022_re60 val022_re100 val022_re100_uniform --repeats
1``), each a tier-2 run of hours, recorded in benchmarks/results.jsonl with
every step's lift and drag in docs/reports/val022/; their report is
docs/reports/val022/val022.md. What is pinned here:

- The case file and the presets: the clustered grid the file states (its
  counts, every breakpoint a face, the far field at most 0.3 d, the body's
  faces on breakpoints), each preset's Reynolds number, end and mesh.
- The start-up: the probe's perturbation of the lower boundary, refused
  where there is no such segment.
- The rows' measures on synthetic lifts with known answers: the per-period
  amplitudes, the decay (Re 40), the sustain (Re 60) with its margin, the
  quiet reading of a lift that no longer oscillates, the Strouhal band.
- The harness's reading of each row by its criterion.

The planted defects each test was shown to catch are named in its
docstring (docs/reports/val022/plant68.py).
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import benchmark

from src.forces import StrouhalReading
from src.mesh import SOLID, Mesh
from src.time_integration import TransientRunRecord
from validation import cylinder
from validation.cases import CYLINDER_PRESETS, CYLINDER_UNIFORM_SPACING, load_preset

PRESETS = ("val022_re40", "val022_re60", "val022_re100", "val022_re100_uniform")


@pytest.mark.unit
class TestCaseAndPresets:
    def test_the_clustered_grid_is_the_one_the_file_states(self) -> None:
        """280 by 150; every breakpoint a face; at most 0.3 d; 26 cells across the body."""
        config = load_preset("val022_re100")
        mesh = Mesh(config)
        assert (config.nx, config.ny) == (280, 150)
        for spec, faces in ((config.stretch_x, mesh.x), (config.stretch_y, mesh.y)):
            for point in spec.breakpoints:
                assert point.at in faces.tolist()
        assert mesh.dx_cell.max() <= 0.3 * (1.0 + 1e-12)
        assert mesh.dy_cell.max() <= 0.3 * (1.0 + 1e-12)
        assert mesh.min_spacing_x == pytest.approx(0.025, rel=1e-9)
        assert (mesh.cell_type == SOLID).sum() == 26 * 26
        assert mesh.stretch_ratio_x < 1.08 and mesh.stretch_ratio_y < 1.08

    @pytest.mark.parametrize("preset", PRESETS)
    def test_each_preset_is_its_reynolds_number_end_and_mesh(self, preset: str) -> None:
        reynolds, t_end, uniform = CYLINDER_PRESETS[preset]
        config = load_preset(preset)
        assert config.rho * 1.0 * 1.0 / config.mu == pytest.approx(reynolds, rel=1e-12)
        assert config.t_end == t_end and config.transient is not None
        assert config.transient.forces.obstacle == "cylinder"
        if uniform:
            assert (config.nx, config.ny) == (370, 200)
            assert config.nx * CYLINDER_UNIFORM_SPACING == pytest.approx(37.0)
            assert Mesh(config).is_uniform
        else:
            assert (config.nx, config.ny) == (280, 150)
        assert cylinder.ROW_CRITERIA[preset] in (
            cylinder.DECAY,
            cylinder.SUSTAIN,
            cylinder.BAND,
            cylinder.REPORT,
        )

    def test_the_rows_criteria_are_the_decision(self) -> None:
        assert cylinder.ROW_CRITERIA == {
            "val022_re40": cylinder.DECAY,
            "val022_re60": cylinder.SUSTAIN,
            "val022_re100": cylinder.BAND,
            "val022_re100_uniform": cylinder.REPORT,
        }
        assert cylinder.STROUHAL_BAND == (0.139, 0.153)
        assert pytest.approx(0.139, abs=1e-3) == 0.146 * 0.95
        assert pytest.approx(0.153, abs=1e-3) == 0.146 * 1.05

    def test_the_start_up_needs_a_lower_segment_to_slow(self) -> None:
        config = load_preset("val022_re100")
        raw = config.merged_mapping()
        del raw["boundaries"]["bottom"]
        from src.config import SimConfig

        with pytest.raises(ValueError, match="'bottom' segment"):
            cylinder.perturbed_startup(SimConfig.from_dict(raw))


def _lift(
    amplitude: object, periods: int = 12, period: float = 6.85, dt: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """A lift whose per-period amplitude is amplitude(t)."""
    t = np.arange(0.0, periods * period, dt)
    return t, amplitude(t) * np.sin(2.0 * math.pi * t / period + 0.2)  # type: ignore[operator]


@pytest.mark.unit
class TestMeasures:
    def test_the_amplitudes_of_a_steady_sine_are_its_amplitude(self) -> None:
        """Catches: the full range taken for half of it."""
        t, c_l = _lift(lambda t: 0.3 + 0.0 * t)
        found = cylinder.amplitudes(t, c_l)
        assert found.size == 11
        assert np.allclose(found, 0.3, rtol=2e-3)

    def test_a_decaying_lift_decays_and_does_not_sustain(self) -> None:
        """Catches: the decay's comparison reversed, the sustain's margin dropped."""
        t, c_l = _lift(lambda t: 0.3 * np.exp(-0.01 * t), periods=40)
        assert cylinder.decays(t, c_l)["passed"]
        assert not cylinder.sustains(t, c_l)["passed"]

    def test_a_growing_or_saturated_lift_sustains_and_does_not_decay(self) -> None:
        for shape in (lambda t: 0.01 * np.exp(0.01 * t), lambda t: 0.3 + 0.0 * t):
            t, c_l = _lift(shape, periods=40)
            assert cylinder.sustains(t, c_l)["passed"]
            assert not cylinder.decays(t, c_l)["passed"]

    def test_the_sustain_margin_admits_a_saturating_wobble_of_under_a_percent(
        self,
    ) -> None:
        """0.5% down one period passes; 2% down every period does not."""
        t, c_l = _lift(
            lambda t: 0.3 * (1.0 - 0.005 * np.sin(2 * math.pi * t / 40.0)), periods=40
        )
        assert cylinder.sustains(t, c_l)["passed"]
        t, c_l = _lift(lambda t: 0.3 * 0.98 ** (t / 6.85), periods=40)
        assert not cylinder.sustains(t, c_l)["passed"]

    def test_a_lift_that_stopped_oscillating_reads_quiet(self) -> None:
        """Fewer than six periods in the second half: the last tenth's magnitude."""
        t = np.arange(0.0, 200.0, 0.05)
        c_l = 1e-8 * np.exp(-t)
        verdict = cylinder.decays(t, c_l)
        assert verdict["measure"] == "quiet" and verdict["passed"]
        loud = cylinder.decays(t, np.full_like(t, 1e-3))
        assert loud["measure"] == "quiet" and not loud["passed"]

    def test_the_decay_reads_the_second_half_of_the_run(self) -> None:
        """Eight growing periods early, none in the second half: the quiet reading.

        The first third oscillates with a growing amplitude; the second half
        holds no upward crossing, so the measure takes the quiet reading and
        the lift passes. Catches (Y6): the window the whole run, which reads
        the early growth as the last periods and fails.
        """
        t = np.arange(0.0, 200.0, 0.05)
        early = 0.01 * np.exp(0.02 * t) * np.sin(2.0 * math.pi * t / 6.85)
        c_l = np.where(t < 60.0, early, 1e-8 * np.exp(-(t - 60.0)))
        verdict = cylinder.decays(t, c_l)
        assert verdict["measure"] == "quiet" and verdict["passed"]

    def test_the_quiet_reading_is_the_last_tenth_of_the_run(self) -> None:
        """Loud until t = 175 and quiet over the last tenth passes; loud in it does not.

        Catches (Y7): the reading's window the second half, which sees the
        loud stretch before t = 175 and fails the first lift.
        """
        t = np.arange(0.0, 200.0, 0.05)
        settled = np.where((t >= 100.0) & (t < 175.0), 1e-3, 0.0)
        verdict = cylinder.decays(t, settled)
        assert verdict["measure"] == "quiet" and verdict["passed"]
        assert verdict["largest_lift"] == 0.0
        loud_at_the_end = np.where(t >= 190.0, 1e-3, 0.0)
        assert not cylinder.decays(t, loud_at_the_end)["passed"]

    def test_the_decay_reads_the_last_six_amplitudes(self) -> None:
        """Rising to t = 190 then falling decays; falling then rising does not.

        Both have twenty periods in the second half. Catches (Y9): the first
        six amplitudes of the second half read in place of the last.
        """
        t, c_l = _lift(
            lambda t: np.where(
                t < 190.0, 0.1 + 0.001 * t, 0.29 * np.exp(-0.02 * (t - 190.0))
            ),
            periods=40,
        )
        assert cylinder.decays(t, c_l)["passed"]
        t, c_l = _lift(
            lambda t: np.where(
                t < 190.0,
                0.3 * np.exp(-0.01 * t),
                0.3 * math.exp(-1.9) * np.exp(0.02 * (t - 190.0)),
            ),
            periods=40,
        )
        assert not cylinder.decays(t, c_l)["passed"]

    def test_the_sustain_reads_the_last_five_periods(self) -> None:
        """A lift that fell and then held sustains; one that held and then fell does not.

        Catches (Y8): the oldest amplitudes read in place of the last.
        """
        t, c_l = _lift(
            lambda t: np.where(
                t < 150.0, 0.3 * np.exp(-0.02 * t), 0.3 * math.exp(-3.0)
            ),
            periods=40,
        )
        assert cylinder.sustains(t, c_l)["passed"]
        t, c_l = _lift(
            lambda t: np.where(t < 150.0, 0.3, 0.3 * np.exp(-0.03 * (t - 150.0))),
            periods=40,
        )
        assert not cylinder.sustains(t, c_l)["passed"]

    def test_fewer_than_six_periods_cannot_sustain(self) -> None:
        t, c_l = _lift(lambda t: 0.3 + 0.0 * t, periods=5)
        verdict = cylinder.sustains(t, c_l)
        assert not verdict["passed"] and verdict["periods"] < 6

    @pytest.mark.parametrize(
        ("st", "inside"),
        [
            (0.139, True),
            (0.146, True),
            (0.153, True),
            (0.1389, False),
            (0.1531, False),
            (None, False),
        ],
    )
    def test_the_band_is_inclusive(self, st: float | None, inside: bool) -> None:
        assert cylinder.strouhal_in_band(st) is inside


def _record(st: float | None, c_d: float) -> TransientRunRecord:
    # The spectrum's peak is 10% above the crossing reading, on the other side
    # of the band for the readings the tests use (0.145 inside, 0.131 outside),
    # so a harness that scores or reports it in place of the crossing reading
    # fails them.
    reading = (
        None
        if st is None
        else StrouhalReading(st, 1.0 / st, np.zeros(11), 1.1 * st, 0.001, False)
    )
    return TransientRunRecord(
        path=Path("run.h5"),
        steps=4000,
        t_final=200.0,
        wall_seconds=1.0,
        outer_mean=5.0,
        outer_max=128,
        strouhal=reading,
        strouhal_refusal=None if st is not None else "the lift holds 3 whole periods",
        drag_coefficient=c_d,
    )


@pytest.mark.unit
class TestHarnessReading:
    """benchmark.cylinder_accuracy reads each row by its criterion."""

    def test_re100_passes_inside_the_band_and_fails_outside(self) -> None:
        t, c_l = _lift(lambda t: 0.3 + 0.0 * t)
        inside = benchmark.cylinder_accuracy(
            "val022_re100", t, c_l, _record(0.145, 1.47)
        )
        outside = benchmark.cylinder_accuracy(
            "val022_re100", t, c_l, _record(0.131, 1.47)
        )
        assert inside["passed"] is True and inside["value"] == 0.145
        assert inside["strouhal"] == 0.145
        assert inside["spectral_strouhal"] == pytest.approx(1.1 * 0.145)
        assert inside["spectra_agree"] is False
        assert inside["drag_coefficient"] == 1.47
        assert outside["passed"] is False

    def test_the_rung_is_reported_not_scored(self) -> None:
        t, c_l = _lift(lambda t: 0.3 + 0.0 * t)
        row = benchmark.cylinder_accuracy(
            "val022_re100_uniform", t, c_l, _record(0.131, 1.4)
        )
        assert row["passed"] is None and row["value"] == 0.131

    def test_re40_and_re60_read_their_amplitudes(self) -> None:
        t, c_l = _lift(lambda t: 0.3 * np.exp(-0.01 * t), periods=40)
        steady = benchmark.cylinder_accuracy("val022_re40", t, c_l, _record(None, 1.6))
        shedding = benchmark.cylinder_accuracy(
            "val022_re60", t, c_l, _record(None, 1.5)
        )
        assert steady["passed"] is True and steady["value"] < 1.0
        assert steady["metric"] == "lift_amplitude_decay"
        assert (
            shedding["passed"] is False
            and shedding["metric"] == "lift_amplitude_sustain"
        )

    def test_a_refused_reading_gives_nan_and_fails_the_band(self) -> None:
        t, c_l = _lift(lambda t: 0.3 + 0.0 * t)
        row = benchmark.cylinder_accuracy("val022_re100", t, c_l, _record(None, 1.4))
        assert math.isnan(row["value"]) and row["passed"] is False
        assert row["strouhal_refusal"].startswith("the lift holds")


@pytest.mark.integration
def test_the_start_up_is_the_probes_on_a_small_channel() -> None:
    """Two hundred steady outer iterations with the lower boundary at 0.9 U.

    On the channel of tests/test_forces.py (freestream laterals, a block): the
    start-up equals, to the byte, a steady solve of the same mapping with the
    lower segment's velocity scaled by hand and the cap set to 200. Catches:
    the perturbation dropped, the cap not the probe's.
    """
    from src.boundary_staggered import StaggeredBoundary
    from src.config import SimConfig
    from src.solver_staggered import StaggeredSolver
    from tests.test_forces import _raw

    raw = _raw(60, 20, freestream=True)
    config = SimConfig.from_dict(raw)
    state, outer = cylinder.perturbed_startup(config)
    by_hand = _raw(60, 20, freestream=True)
    by_hand["boundaries"]["bottom"]["u_velocity"] = 0.9
    by_hand["solver"].update(max_simple_iter=200, convergence_tol=1e-12)
    hand = SimConfig.from_dict(by_hand)
    mesh = Mesh(hand)
    solver = StaggeredSolver(mesh, hand, StaggeredBoundary(mesh, hand))
    solver.solve_steady()
    assert outer == 200 == len(solver.residual_history)
    assert np.array_equal(state.u, solver.flow_state.u)
    assert np.array_equal(state.v, solver.flow_state.v)
    assert np.array_equal(state.p, solver.flow_state.p)
