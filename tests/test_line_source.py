"""VAL-024, a line source in decaying grid turbulence (ADR-015 A, step 0 of its build).

The case is configs/validation_line_source.yaml, loaded by
validation.cases.load_line_source, and compared with Warhaft (1984) by
validation.line_source, which stores the digitised widths of table T8 of
docs/reports/probe70/digitised70c.md. The criterion is Alex's decision of
2026-10-30 (ADR-015 A): the solver's decay exponent within 0.1 of the model's
own 1.087 (REQ-S14), and the plume's width within 20% of the measured at 20 and
50 mesh lengths downstream (REQ-T14). The 5, 10, 80 and 100 mesh length
stations are printed, not asserted: 5 and 10 are the free fall, 80 is a carried
limit and 100 has no measurement behind it.

What is pinned, in three groups. The loader and the stored data: the presets,
the inlet's turbulence against the experiment's constants, the grid-dependent
mass imbalance bound, the rows of T8 against the table in the repository, and
the reference at each station against the numbers T9 prints. The readers on
constructed fields whose answer follows from the field: the decay exponent on
an analytic decay, the width on Gaussian columns, the station lookup on a
width that is linear in x, and the criterion on planted widths either side of
its allowances. And the solved case on the suite's grid, 280x50 (the row
``val024_560x100`` is the harness's, ``python scripts/benchmark.py --cases
val024_560x100 --repeats 1``).

Planted defects. Each was shown to be caught, by the check named in its test:
the decay exponent read against ln x in place of ln(t_v + x / U) (the
criterion), the reference interpolated linearly in place of log-log (the
pinned reference), the width taken as the half-width, the 0.849 factor dropped
(the criterion on the solved field and the pinned reference), and the
stations shifted by one place against their data (the criterion and the
station lookup on a linear width).
"""

import re
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import benchmark

from src.boundary_staggered import StaggeredBoundary
from src.mesh import Mesh
from src.solver_staggered import StaggeredSolver
from validation import line_source as ls
from validation.cases import (
    CASE_GRIDS,
    LINE_SOURCE_GRIDS,
    load_line_source,
    load_preset,
)

ROOT = Path(__file__).resolve().parents[1]
DIGITISED = ROOT / "docs" / "reports" / "probe70" / "digitised70c.md"
GRID_280 = (280, 50)

# T9 of digitised70c.md: the measured width at the stations, mm, as printed
# (one decimal), and the Anand and Pope curve beside it.
T9_DATA_MM = {5.0: 2.7, 10.0: 4.8, 20.0: 8.5, 50.0: 16.0, 80.0: 21.7}
T9_CURVE_MM = {5.0: 2.5, 10.0: 4.7, 20.0: 8.3, 50.0: 16.1, 80.0: 21.6, 100.0: 24.6}
T9_PAST_THE_DATA_MM = (25.3, 22.5)
PIN_TOLERANCE_MM = 0.05
# The narrowest plume the 1 cm cells of the coarse grid can hold, m.
NARROWEST = 0.006


# ---------------------------------------------------------------------------
# The stored data and the reference
# ---------------------------------------------------------------------------


def t8_rows() -> list[tuple[int, str, float, float]]:
    """The rows of table T8 of digitised70c.md: (source, wire, x' / M, half-width / M)."""
    text = DIGITISED.read_text(encoding="utf-8")
    block = text.split("### T8.")[1].split("### T9.")[0]
    rows = []
    for line in block.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 10 and re.fullmatch(r"\d+", cells[0]):
            rows.append((int(cells[0]), cells[1], float(cells[3]), float(cells[4])))
    return rows


def check_the_reference_against_t9() -> None:
    """The reference at each station and past the data's end, against T9's numbers.

    Written as a function so a planted defect can be run through the same
    assertions and shown to fail them.
    """
    for station, want in T9_DATA_MM.items():
        got = ls.reference_sigma(station) * 1e3
        assert got == pytest.approx(want, abs=PIN_TOLERANCE_MM), station
    for station, want in T9_CURVE_MM.items():
        got = float(ls.reference_curve(station)) * 1e3
        assert got == pytest.approx(want, abs=PIN_TOLERANCE_MM), station
    extrapolated = ls.extrapolated_sigma(100.0) * 1e3
    collapse = ls.collapse_sigma(100.0) * 1e3
    assert (extrapolated, collapse) == pytest.approx(
        T9_PAST_THE_DATA_MM, abs=PIN_TOLERANCE_MM
    )


@pytest.mark.unit
class TestReferenceData:
    def test_the_stored_rows_are_the_rows_of_t8(self) -> None:
        """Every 52 M and 20 M row of T8, in the table's order, is stored as printed.

        Defect caught: a transcription error in a stored point, or a table
        regenerated without the module following it.
        """
        wires = {"0.025 mm": ls.FINE, "0.127 mm": ls.THICK}
        rows = t8_rows()
        at_52 = [(wires[w], x, hw) for s, w, x, hw in rows if s == 52]
        at_20 = [(x, hw) for s, _w, x, hw in rows if s == 20]
        assert len(at_52) == 18 and len(at_20) == 4
        assert tuple(at_52) == ls.WARHAFT_52M
        assert tuple(at_20) == ls.WARHAFT_20M

    def test_the_composite_has_the_stations_of_t9(self) -> None:
        """Seven stations, the near-source and repeated points averaged by bin."""
        xs, ys = ls.composite_52m()
        assert xs.round(2).tolist() == [0.55, 3.63, 5.44, 8.1, 30.03, 60.12, 80.24]
        assert np.all(np.diff(xs) > 0.0) and np.all(np.diff(ys) > 0.0)
        # sigma = 0.849 x half-width x M at the join: five readings averaged.
        join = np.mean([0.1703, 0.2098, 0.1955, 0.2064, 0.1749])
        assert ys[3] == pytest.approx(0.849 * join * 0.025, rel=1e-12)

    def test_the_reference_is_the_numbers_t9_prints(self) -> None:
        check_the_reference_against_t9()

    def test_the_data_lie_on_the_curve_within_five_percent_from_30_to_80_m(
        self,
    ) -> None:
        """ADR-015 A's check on the data: the curve is not the interpolant."""
        for station in (30.0, 60.0, 80.0):
            ratio = ls.reference_sigma(station) / float(ls.reference_curve(station))
            assert abs(ratio - 1.0) < 0.055, station

    def test_a_station_outside_the_data_is_refused(self) -> None:
        xs, _ys = ls.composite_52m()
        for station in (xs[0] - 0.01, xs[-1] + 0.01, 100.0):
            with pytest.raises(ValueError, match="outside the 52 M data"):
                ls.reference_sigma(float(station))

    def test_log_log_interpolation_is_exact_on_a_power_law(self) -> None:
        """A power law is a straight line in log-log: any station is exact, any other rule not.

        Defect caught: linear interpolation in x and sigma, which is 1% to 3%
        off on the real data between its stations.
        """
        xs = np.array([1.0, 4.0, 30.0, 60.0])
        ys = 2.5e-3 * xs**0.6
        for x in (1.5, 10.0, 45.0):
            assert ls.loglog_interp(x, xs, ys) == pytest.approx(
                2.5e-3 * x**0.6, rel=1e-13
            )
            assert float(np.interp(x, xs, ys)) != pytest.approx(
                2.5e-3 * x**0.6, rel=1e-3
            )

    def test_the_source_turbulence_is_the_experiments(self) -> None:
        """k and eps at 52 M from the velocity variances and the decay law.

        Warhaft's table 1 gives u' 0.153 and v' 0.142 m/s at the station and
        eps 0.265 m^2/s^3; the refit the reference uses is within 1% of them for
        the velocities, and its eps (m k_o U / x_o) is 0.243.
        """
        t = ls.source_turbulence()
        assert (t.sigma_u, t.sigma_v) == pytest.approx((0.1535, 0.1428), abs=1e-4)
        assert t.k == pytest.approx(0.032186, rel=1e-5)
        assert t.eps == pytest.approx(0.24263, rel=1e-4)
        assert t.lagrangian_time == pytest.approx(0.08422, rel=1e-4)
        assert t.virtual_time == pytest.approx(0.14419, rel=1e-4)
        assert t.k == 0.5 * (t.sigma_u**2 + 2.0 * t.sigma_v**2)
        assert t.eps == pytest.approx(1.4 * t.k * 7.0 / (52.0 * 0.025), rel=1e-14)
        assert ls.model_decay_exponent() == pytest.approx(1.0 / 0.92, rel=1e-15)

    def test_a_larger_mesh_raises_the_flight_time_and_lowers_eps(self) -> None:
        """eps = m k U / (x_o M): at 2.54 cm it is 1.6% lower than at 2.5 cm."""
        wide, narrow = ls.source_turbulence(0.0254), ls.source_turbulence(0.025)
        assert wide.k == narrow.k
        assert wide.eps / narrow.eps == pytest.approx(0.025 / 0.0254, rel=1e-14)

    def test_linear_interpolation_is_caught_by_the_pinned_reference(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Planted: the reference interpolated linearly. It moves 2.5% at 20 M."""

        def linear(x: float, xs: np.ndarray, ys: np.ndarray) -> float:
            return float(np.interp(x, xs, ys))

        monkeypatch.setattr(ls, "loglog_interp", linear)
        with pytest.raises(AssertionError):
            check_the_reference_against_t9()

    def test_the_half_width_is_caught_by_the_pinned_reference(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Planted: the reference taken as the half-width, the 0.849 dropped."""
        monkeypatch.setattr(ls, "HW_TO_SIGMA", 1.0)
        with pytest.raises(AssertionError):
            check_the_reference_against_t9()


# ---------------------------------------------------------------------------
# The loader
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoader:
    def test_the_presets_are_registered(self) -> None:
        assert {"val024_280x50", "val024_560x100"} == LINE_SOURCE_GRIDS
        for preset in LINE_SOURCE_GRIDS:
            kind, nx, ny = CASE_GRIDS[preset]
            assert kind == "line_source"
            assert preset.endswith(f"{nx}x{ny}")

    @pytest.mark.parametrize("preset", sorted(LINE_SOURCE_GRIDS))
    def test_the_domain_is_112_mesh_lengths_with_square_cells(
        self, preset: str
    ) -> None:
        config = load_preset(preset)
        mesh = Mesh(config)
        assert (config.room_width, config.room_height) == (2.8, 0.5)
        assert config.room_width / ls.MESH_LENGTH == pytest.approx(112.0, rel=1e-14)
        assert np.allclose(mesh.dx_cell, mesh.dy_cell[0], rtol=1e-12)
        assert mesh.is_uniform

    def test_the_inlet_carries_the_source_stations_turbulence(self) -> None:
        """The file's two numbers are the experiment's, derived again here.

        Defect caught: a constant of the experiment changed in the module or the
        file and not in the other, which would run a case that is not the one
        the reference describes.
        """
        config = load_preset("val024_560x100")
        inlet = config.boundaries["stream"]
        t = ls.source_turbulence()
        assert inlet.velocity == ls.SPEED
        assert inlet.turbulence_intensity == pytest.approx(t.intensity, rel=1e-14)
        assert inlet.dissipation_length == pytest.approx(
            t.dissipation_length, rel=1e-14
        )
        # k = 1.5 (I U)^2 and eps = k^(3/2) / l_e, the solver's convention.
        k = 1.5 * (inlet.turbulence_intensity * inlet.velocity) ** 2
        assert k == pytest.approx(t.k, rel=1e-13)
        assert k**1.5 / inlet.dissipation_length == pytest.approx(t.eps, rel=1e-13)

    def test_the_stream_is_uniform_with_a_fixed_flow_outlet_and_no_pressure_outlet(
        self,
    ) -> None:
        config = load_preset("val024_280x50")
        types = {spec.type for spec in config.boundaries.values()}
        assert types == {"velocity_inlet", "fixed_flow_outlet"}
        inlet = config.boundaries["stream"]
        assert (inlet.location, inlet.y_start, inlet.y_end) == ("left", 0.0, 0.5)
        outlet = config.boundaries["exit"]
        assert (outlet.location, outlet.y_start, outlet.y_end) == ("right", 0.0, 0.5)
        assert config.obstacles == []
        assert config.turbulence is not None and config.turbulence.variant == "standard"
        assert config.transport is not None
        assert config.transport.turbulent_schmidt == 0.7
        assert config.particle_sizes == [0.1e-6]

    @pytest.mark.parametrize(
        ("preset", "cell"), [("val024_280x50", 0.01), ("val024_560x100", 0.005)]
    )
    def test_the_mass_imbalance_bound_is_adr_011_gs_for_the_grids_cell(
        self, preset: str, cell: float
    ) -> None:
        """1e-4 rho V_cell / t_end, 2e-10 and 5e-11 for the two grids."""
        config = load_preset(preset)
        assert config.mass_imbalance_tol == pytest.approx(
            1.0e-4 * 1.2 * cell * cell / 60.0, rel=1e-12
        )

    def test_the_committed_files_own_bound_is_the_loaders(self) -> None:
        """The file carries the bound for its grid, so loading it unchanged agrees."""
        from src.config import SimConfig
        from validation.cases import case_path

        committed = SimConfig(case_path("line_source"))
        assert (committed.nx, committed.ny) == (560, 100)
        assert committed.mass_imbalance_tol == pytest.approx(
            load_line_source().mass_imbalance_tol, rel=1e-12
        )

    def test_cells_that_are_not_square_are_refused(self) -> None:
        with pytest.raises(ValueError, match="cells must be square"):
            load_line_source("line_source", grid=(280, 100))

    def test_a_case_without_the_named_inlet_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("validation.cases.LINE_SOURCE_INLET", "intake")
        with pytest.raises(ValueError, match="velocity inlet named 'intake'"):
            load_line_source()


# ---------------------------------------------------------------------------
# The readers on constructed fields
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def coarse_mesh() -> Mesh:
    return Mesh(load_preset("val024_280x50"))


def analytic_k(mesh: Mesh, n: float) -> np.ndarray:
    """k_o (1 + t / t_v)^(-n) along x at every row, the model's own decay."""
    t = ls.source_turbulence()
    x = np.asarray(mesh.xc)
    line = t.k * (1.0 + (x / ls.SPEED) / t.virtual_time) ** (-n)
    return np.tile(line, (mesh.cell_type.shape[0], 1))


def gaussian_field(
    mesh: Mesh, sigma_of_x: Callable[[np.ndarray], np.ndarray], centre: float
) -> np.ndarray:
    """A plume whose column at x is a Gaussian of width sigma_of_x(x) about ``centre``."""
    yc = np.asarray(mesh.yc)
    sigma = sigma_of_x(np.asarray(mesh.xc))
    return np.exp(-0.5 * ((yc[:, None] - centre) / sigma[None, :]) ** 2)


def planted_plume(
    mesh: Mesh,
    factor: float = 1.0,
    windows: tuple[tuple[float, float, float], ...] = (),
) -> np.ndarray:
    """A Gaussian plume whose width is ``factor`` times the measured one.

    The data to the last of the 52 M station, its extrapolation beyond. Each
    window (lo, hi, scale) multiplies the width again for x' / M in [lo, hi),
    so one station can be moved and no other. The mean sits at the source
    row's centre. A Gaussian much narrower than the 1 cm cell has a second
    moment of about zero on the grid (one cell holds it), so the width is
    floored at NARROWEST, 0.6 cell: the stations under about 14 M then read
    wide, and no test of the criterion reads them.
    """
    source_x = float(np.asarray(mesh.xc)[ls.SOURCE_COLUMN])
    xs, _ys = ls.composite_52m()

    def width(x: np.ndarray) -> np.ndarray:
        x_prime = np.maximum((x - source_x) / ls.MESH_LENGTH, xs[0])
        sigma = factor * np.array(
            [
                ls.reference_sigma(float(s))
                if s <= xs[-1]
                else ls.extrapolated_sigma(float(s))
                for s in x_prime
            ]
        )
        for lo, hi, scale in windows:
            sigma = np.where((x_prime >= lo) & (x_prime < hi), sigma * scale, sigma)
        return np.maximum(sigma, NARROWEST)

    centre = float(np.asarray(mesh.yc)[ls.source_row(mesh)])
    return gaussian_field(mesh, width, centre)


@pytest.mark.unit
class TestReaders:
    @pytest.mark.parametrize("n", [1.087, 1.4, 1.0])
    def test_the_decay_exponent_of_an_analytic_decay_is_recovered(
        self, coarse_mesh: Mesh, n: float
    ) -> None:
        """k = k_o (1 + t / t_v)^(-n) is a line of slope -n on the fit's axes."""
        k = analytic_k(coarse_mesh, n)
        got = ls.decay_exponent(
            coarse_mesh, k[ls.source_row(coarse_mesh)], ls.source_turbulence()
        )
        assert got == pytest.approx(n, rel=1e-10)

    def test_the_fit_uses_the_middle_80_percent_only(self, coarse_mesh: Mesh) -> None:
        """Values spoiled outside 10% to 90% of the length leave the exponent alone."""
        k = analytic_k(coarse_mesh, 1.087)[0].copy()
        xc = np.asarray(coarse_mesh.xc)
        k[xc < 0.1 * 2.8] *= 7.0
        k[xc > 0.9 * 2.8] *= 0.01
        got = ls.decay_exponent(coarse_mesh, k, ls.source_turbulence())
        assert got == pytest.approx(1.087, rel=1e-10)

    def test_the_decay_read_on_the_wrong_axis_is_far_from_the_models(
        self, coarse_mesh: Mesh
    ) -> None:
        """Planted: ln k against ln x, without the virtual origin t_v.

        The same analytic decay then reads about 0.5, so the criterion's 0.1
        either side of the model's 1.087 rejects it.
        """
        k = analytic_k(coarse_mesh, ls.model_decay_exponent())[0]
        xc = np.asarray(coarse_mesh.xc)
        chosen = (xc >= 0.1 * 2.8) & (xc <= 0.9 * 2.8)
        slope, _ = np.polyfit(np.log(xc[chosen]), np.log(k[chosen]), 1)
        assert abs(-slope - ls.model_decay_exponent()) > 5 * ls.DECAY_ALLOWANCE

    @pytest.mark.parametrize("centre", [0.255, 0.31])
    def test_the_width_of_a_gaussian_column_is_its_standard_deviation(
        self, coarse_mesh: Mesh, centre: float
    ) -> None:
        """Columns of widths 1.5 to 3.5 cm: the second moment is sigma, wherever the mean.

        The widest column's tails end more than five widths from the walls, and
        the cells are 1 cm, so neither the truncation nor the sampling shows.
        """
        field = gaussian_field(
            coarse_mesh, lambda x: 0.015 + 0.02 * x / 2.8 + 0 * x, centre
        )
        want = 0.015 + 0.02 * np.asarray(coarse_mesh.xc) / 2.8
        got = ls.plume_widths(coarse_mesh, field)
        assert got == pytest.approx(want, rel=1e-6)

    def test_a_column_without_concentration_has_no_width(
        self, coarse_mesh: Mesh
    ) -> None:
        field = gaussian_field(coarse_mesh, lambda x: 0.03 + 0 * x, 0.25)
        field[:, 10] = 0.0
        field[:, 11] = -1.0
        widths = ls.plume_widths(coarse_mesh, field)
        assert np.isnan(widths[10]) and np.isnan(widths[11])
        assert np.isfinite(np.delete(widths, [10, 11])).all()

    def test_a_station_is_read_where_it_is_not_one_cell_off(
        self, coarse_mesh: Mesh
    ) -> None:
        """A width linear in x' returns the line's value at each station exactly.

        Cells are 0.4 mesh lengths wide, so a station read one cell off is wrong
        by 0.4 x the slope. Columns within NEAR_SOURCE of the source hold a
        sentinel the reader must not use. Defect caught: stations measured from
        the first column, or from the cell next to the source.
        """
        xc = np.asarray(coarse_mesh.xc)
        source_x = float(xc[ls.SOURCE_COLUMN])
        slope = 1.0e-3  # m per mesh length
        x_prime = (xc - source_x) / ls.MESH_LENGTH
        widths = np.where(x_prime > ls.NEAR_SOURCE, 1.0e-3 + slope * x_prime, 9.9)
        for station in ls.STATIONS_M:
            want = 1.0e-3 + slope * station
            assert ls.width_at(coarse_mesh, widths, source_x, station) == pytest.approx(
                want, rel=1e-12
            )
            one_cell = 0.4
            shifted = ls.width_at(coarse_mesh, widths, source_x + 0.01, station)
            assert abs(shifted - want) == pytest.approx(slope * one_cell, rel=1e-9)

    def test_the_stations_shifted_by_one_are_caught_by_the_station_lookup(
        self, coarse_mesh: Mesh
    ) -> None:
        """Planted: the model read at the next station of the list.

        The lookup of the line above has a different value at every station,
        so the reading one station along differs from the right one by more
        than the interpolation's error by orders of magnitude.
        """
        xc = np.asarray(coarse_mesh.xc)
        source_x = float(xc[ls.SOURCE_COLUMN])
        x_prime = (xc - source_x) / ls.MESH_LENGTH
        widths = np.where(x_prime > ls.NEAR_SOURCE, 1.0e-3 + 1.0e-3 * x_prime, 9.9)
        right = [ls.width_at(coarse_mesh, widths, source_x, s) for s in ls.STATIONS_M]
        shifted = [
            ls.width_at(coarse_mesh, widths, source_x, s) for s in ls.STATIONS_M[1:]
        ]
        for a, b in zip(right[:-1], shifted, strict=True):
            assert abs(b - a) > 1.0e-3


# ---------------------------------------------------------------------------
# The criterion on planted fields
# ---------------------------------------------------------------------------


def verdict(
    mesh: Mesh,
    factor: float = 1.0,
    n: float | None = None,
    windows: tuple[tuple[float, float, float], ...] = (),
) -> dict:
    """``evaluate`` on a plume ``factor`` times the measured one and a decay of exponent n."""
    field = planted_plume(mesh, factor, windows)
    k = analytic_k(mesh, ls.model_decay_exponent() if n is None else n)
    return ls.evaluate(mesh, load_preset("val024_280x50"), k, field)


@pytest.mark.unit
class TestCriterion:
    def test_a_model_on_the_data_with_its_own_decay_passes(
        self, coarse_mesh: Mesh
    ) -> None:
        report = verdict(coarse_mesh)
        assert report["passed"] and report["decay"]["passed"]
        assert report["decay"]["n"] == pytest.approx(1.0 / 0.92, rel=1e-9)
        for row in report["stations"]:
            if row["scored"]:
                assert row["passed"] and row["departure"] < 5e-3
        assert report["value"] < 5e-3

    def test_the_decay_allowance_is_a_tenth_either_side_of_the_models(
        self, coarse_mesh: Mesh
    ) -> None:
        n = ls.model_decay_exponent()
        assert verdict(coarse_mesh, n=n + 0.09)["passed"]
        assert verdict(coarse_mesh, n=n - 0.09)["passed"]
        for bad in (n + 0.11, n - 0.11):
            report = verdict(coarse_mesh, n=bad)
            assert not report["decay"]["passed"] and not report["passed"]

    @pytest.mark.parametrize("station", ls.SCORED_STATIONS_M)
    def test_the_width_allowance_is_a_fifth_either_side_at_each_scored_station(
        self, coarse_mesh: Mesh, station: float
    ) -> None:
        """Widths 19% off pass and 21% off fail, high and low, at 20 and 50 M.

        The plume is scaled for the six mesh lengths around the station, so
        the other scored station is untouched and the verdict is this one's.
        """
        window = (station - 3.0, station + 3.0)
        for scale, passed in ((1.19, True), (0.81, True), (1.21, False), (0.79, False)):
            report = verdict(coarse_mesh, windows=((*window, scale),))
            row = {r["station"]: r for r in report["stations"]}[station]
            assert row["passed"] is passed, (station, scale)
            assert row["departure"] == pytest.approx(abs(scale - 1.0), abs=5e-3)
            assert report["passed"] is passed
            other = {r["station"]: r for r in report["stations"]}[
                50.0 if station == 20.0 else 20.0
            ]
            assert other["departure"] < 5e-3

    def test_the_stations_outside_the_criterion_do_not_move_it(
        self, coarse_mesh: Mesh
    ) -> None:
        """5 and 10 M (the free fall) and 80 and 100 M (the carried limit) are reported.

        A plume three times the measured at 80 and 100 M, and the floored one
        at 5 and 10 M, still passes, and the ratios are in the report.
        """
        report = verdict(coarse_mesh, windows=((70.0, 200.0, 3.0),))
        assert report["passed"]
        by_station = {row["station"]: row for row in report["stations"]}
        assert [by_station[s]["scored"] for s in ls.STATIONS_M] == [
            False,
            False,
            True,
            True,
            False,
            False,
        ]
        assert by_station[5.0]["ratio"][0] == pytest.approx(
            NARROWEST / by_station[5.0]["data_sigma_m"][0], rel=0.03
        )
        assert by_station[80.0]["ratio"][0] == pytest.approx(3.0, rel=0.01)
        row = by_station[100.0]
        assert row["ratio"] == pytest.approx(
            [row["model_sigma_m"] / d for d in row["data_sigma_m"]], rel=1e-14
        )
        assert row["ratio"][0] == pytest.approx(3.0, rel=0.01)
        assert row["ratio"][1] > row["ratio"][0]

    def test_the_hundred_mesh_length_row_carries_both_estimates(
        self, coarse_mesh: Mesh
    ) -> None:
        row = verdict(coarse_mesh)["stations"][-1]
        assert row["station"] == 100.0
        assert [d * 1e3 for d in row["data_sigma_m"]] == pytest.approx(
            T9_PAST_THE_DATA_MM, abs=PIN_TOLERANCE_MM
        )
        assert row["reference_curve_sigma_m"] * 1e3 == pytest.approx(24.6, abs=0.05)

    def test_a_half_width_reference_fails_a_model_a_little_under_the_data(
        self, coarse_mesh: Mesh, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Planted: the reference taken as the half-width (the 0.849 dropped).

        A model right to the data reads 0.849 of the defective reference,
        15% under, inside the allowance: the criterion alone cannot see the
        defect there, and the pinned reference does (above). A model 7% under
        the data, like the solved one at 50 M (0.853), reads 0.79 and fails.
        """
        field = planted_plume(coarse_mesh, 0.93)
        k = analytic_k(coarse_mesh, ls.model_decay_exponent())
        config = load_preset("val024_280x50")
        assert ls.evaluate(coarse_mesh, config, k, field)["passed"]
        monkeypatch.setattr(ls, "HW_TO_SIGMA", 1.0)
        report = ls.evaluate(coarse_mesh, config, k, field)
        assert not report["passed"]
        assert report["value"] == pytest.approx(1.0 - 0.93 * 0.849, abs=5e-3)
        exact = ls.evaluate(coarse_mesh, config, k, planted_plume(coarse_mesh, 1.0))
        assert exact["passed"] is False or exact["value"] < 0.2


# ---------------------------------------------------------------------------
# The solved case: VAL-024
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def solved() -> ls.Solution:
    """The suite's row: the flow solve and the scalar march on 280x50."""
    return ls.solve(load_preset("val024_280x50"))


@pytest.mark.validation
class TestVal024:
    """VAL-024: the decay exponent and the plume width against Warhaft (1984)."""

    def test_the_flow_stops_by_its_rule_and_the_plume_by_steadiness(
        self, solved: ls.Solution
    ) -> None:
        assert solved.flow["stop_reason"] == "error_estimate_and_continuity"
        assert solved.march["stopped_by"] == "steady"
        assert solved.march["last_relative_change"] <= ls.STEADY_RTOL
        assert np.isfinite(solved.field).all() and solved.field.min() >= 0.0

    def test_val024_the_decay_exponent_is_within_0_1_of_the_models_own(
        self, solved: ls.Solution, record_property: Callable[[str, object], None]
    ) -> None:
        """VAL-024, REQ-S14: n within 0.1 of 1 / (C_e2 - 1) = 1.087.

        The probe's value on this grid is 1.077 (T7 of tables70.md); the
        model's own exponent is below the measured grid-turbulence range of
        1.16 to 1.4, which the criterion does not test.
        """
        report = ls.evaluate(solved.mesh, solved.config, solved.k, solved.field)
        decay = report["decay"]
        record_property("decay_exponent", decay["n"])
        print(f"VAL-024 decay exponent {decay['n']:.4f} (model {decay['n_model']:.4f})")
        assert decay["departure"] <= ls.DECAY_ALLOWANCE

    def test_val024_the_width_is_within_20_percent_at_20_and_50_mesh_lengths(
        self, solved: ls.Solution, record_property: Callable[[str, object], None]
    ) -> None:
        """VAL-024, REQ-T14: the model over the measured width, 20 and 50 M.

        The other stations are printed, not asserted: 5 and 10 M are the free
        fall, which never reaches the sensors, 80 M is the carried limit
        (the model's plume about a fifth narrower than measured) and 100 M
        has no measurement behind it (ADR-015 A).
        """
        report = ls.evaluate(solved.mesh, solved.config, solved.k, solved.field)
        lines = ["VAL-024 plume width, model over measured (mm):"]
        for row in report["stations"]:
            data = ", ".join(f"{d * 1e3:.2f}" for d in row["data_sigma_m"])
            ratio = ", ".join(f"{r:.3f}" for r in row["ratio"])
            scored = "scored" if row["scored"] else "reported"
            lines.append(
                f"  {row['station']:5.0f} M  model {row['model_sigma_m'] * 1e3:6.2f}  "
                f"data {data}  ratio {ratio}  ({scored})"
            )
            record_property(f"ratio_{row['station']:g}M", row["ratio"][0])
        print("\n".join(lines))
        scored = {r["station"]: r for r in report["stations"] if r["scored"]}
        assert set(scored) == {20.0, 50.0}
        for station, row in scored.items():
            assert row["departure"] <= ls.WIDTH_ALLOWANCE, station
        assert report["passed"]

    # -- planted defects, through the solved fields --------------------------

    def test_the_decay_on_the_wrong_axis_fails_the_decay_clause(
        self, solved: ls.Solution, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Planted: the exponent fitted against ln x, without the virtual origin."""

        def against_x(
            mesh: Mesh, k_line: np.ndarray, source: ls.SourceTurbulence
        ) -> float:
            xc = np.asarray(mesh.xc)
            chosen = (xc >= 0.1 * 2.8) & (xc <= 0.9 * 2.8)
            slope, _ = np.polyfit(np.log(xc[chosen]), np.log(k_line[chosen]), 1)
            return -float(slope)

        monkeypatch.setattr(ls, "decay_exponent", against_x)
        report = ls.evaluate(solved.mesh, solved.config, solved.k, solved.field)
        assert not report["decay"]["passed"] and not report["passed"]

    def test_the_half_width_reference_fails_the_width_clause_at_50_m(
        self, solved: ls.Solution, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Planted: the 0.849 dropped. The model's 0.853 at 50 M reads 0.724."""
        monkeypatch.setattr(ls, "HW_TO_SIGMA", 1.0)
        report = ls.evaluate(solved.mesh, solved.config, solved.k, solved.field)
        by_station = {r["station"]: r for r in report["stations"]}
        assert by_station[50.0]["ratio"][0] == pytest.approx(0.724, abs=0.02)
        assert not by_station[50.0]["passed"] and not report["passed"]

    def test_the_stations_shifted_by_one_fail_the_width_clause(
        self, solved: ls.Solution, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Planted: the model read at the next station against this station's data.

        At 20 M the model's width at 50 M (13.6 mm) is set against 8.5 mm, a
        ratio of 1.6.
        """
        stations = list(ls.STATIONS_M)
        later = dict(zip(stations, stations[1:] + stations[-1:], strict=True))
        real = ls.width_at

        def shifted(
            mesh: Mesh, widths: np.ndarray, source_x: float, station: float
        ) -> float:
            return real(mesh, widths, source_x, later[station])

        monkeypatch.setattr(ls, "width_at", shifted)
        report = ls.evaluate(solved.mesh, solved.config, solved.k, solved.field)
        by_station = {r["station"]: r for r in report["stations"]}
        assert not by_station[20.0]["passed"] and not report["passed"]

    def test_linear_interpolation_of_the_reference_moves_the_ratios_it_is_pinned_by(
        self, solved: ls.Solution, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Planted: the reference interpolated linearly.

        The solved ratios move by 1% to 3%, inside the allowance, so the
        pinned reference (``check_the_reference_against_t9``) is what rejects
        the defect; here the move is shown on the solved field.
        """
        right = ls.evaluate(solved.mesh, solved.config, solved.k, solved.field)

        def linear(x: float, xs: np.ndarray, ys: np.ndarray) -> float:
            return float(np.interp(x, xs, ys))

        monkeypatch.setattr(ls, "loglog_interp", linear)
        wrong = ls.evaluate(solved.mesh, solved.config, solved.k, solved.field)
        moves = {
            a["station"]: b["ratio"][0] / a["ratio"][0] - 1.0
            for a, b in zip(right["stations"], wrong["stations"], strict=True)
        }
        assert moves[20.0] == pytest.approx(0.026, abs=0.005)
        assert moves[50.0] == pytest.approx(0.011, abs=0.005)


# ---------------------------------------------------------------------------
# The harness row
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestHarnessRow:
    """``val024_*`` in scripts/benchmark.py: the flow solve, then the march and the criterion."""

    def tiny(self, monkeypatch: pytest.MonkeyPatch) -> object:
        """The case on a 56x10 grid, three outer iterations and a march capped at 0.6 s."""
        config = load_line_source("line_source", grid=(56, 10))
        monkeypatch.setattr(config, "max_simple_iter", 3)
        monkeypatch.setattr(ls, "TIME_CAP", 0.6)
        monkeypatch.setattr(benchmark, "load_preset", lambda case_id: config)
        return config

    def test_the_row_carries_the_comparison_and_its_verdict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The row's report is ``evaluate`` on the same flow and a second march.

        Defects caught: a component read from the wrong station or estimate; the
        value taken from an unscored station; ``passed`` left None where a
        criterion is set; the march run on a stale state.
        """
        config = self.tiny(monkeypatch)
        row = benchmark.run_case("val024_280x50", benchmark.DEFAULT_METHOD, 1, 1)
        accuracy = row["accuracy"]
        assert (accuracy["metric"], accuracy["reference"]) == (
            benchmark.LINE_SOURCE_METRIC,
            benchmark.LINE_SOURCE_REFERENCE,
        )
        assert isinstance(accuracy["passed"], bool)

        mesh = Mesh(config)
        solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))
        solver.solve_steady()
        state, faces = solver.flow_state, solver.face_velocities
        assert state is not None and faces is not None
        field, march = ls.march_scalar(mesh, config, state.nu_t, faces)
        report = ls.evaluate(mesh, config, np.array(state.k), field)
        assert accuracy["report"] == report
        assert accuracy["value"] == report["value"]
        assert accuracy["passed"] == report["passed"]
        assert accuracy["march"]["steps"] == march["steps"]
        scored = [r["departure"] for r in report["stations"] if r["scored"]]
        assert accuracy["value"] == max(scored)
        ratios = {r["station"]: r["ratio"] for r in report["stations"]}
        components = accuracy["components"]
        assert components["decay_exponent"] == report["decay"]["n"]
        for station, ratio in ratios.items():
            assert components[f"width_ratio_{station:g}M"] == ratio[0]
        assert components["width_ratio_100M_collapse"] == ratios[100.0][1]
        assert len({round(v, 12) for v in components.values()}) == len(components)
        assert row["trajectory"][-1]["error"] == accuracy["value"]
        assert row["outcome"]["converged"] is False

    def test_the_family_is_scored_from_the_final_flow(self) -> None:
        assert benchmark.LINE_SOURCE_FAMILY in benchmark.FACE_SCORED_FAMILIES
        assert benchmark.CASES["val024_560x100"][0] == benchmark.LINE_SOURCE_FAMILY
