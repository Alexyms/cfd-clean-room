"""VAL-020, the plane impinging jet (ECR-002 step 7; ADR-012 G (vi)): the unit tests.

The case is configs/validation_impinging_jet.yaml, loaded by
validation.cases.load_impinging_jet with y clustered toward the plate and
the nozzle's plane, and compared with case I of Khayrullina et al. (2017)
by validation.impinging_jet, which stores the digitised data with its
provenance. No allowance is set (ECR-002 criterion 13, OPEN); the run is
the harness row ``val020_272x90`` (``python scripts/benchmark.py --cases
val020_272x90 --repeats 1``), reported and not scored, and the first
coupled results are docs/reports/ecr002_val020_impinging_jet.md.

These tests pin the loader (the nozzle on whole faces carrying V0 w on
both presets, the two outlets sharing the inflow, the wall spacing, the
file's settings and a grid that puts the nozzle's edge off a face
refused), the stored data (counts, order, the checks against the numbers
the paper states, a hash of every stored value), the readers on
constructed fields with hand-computed answers (the samplers, the axis fit,
the crossings, the rates, the half-width, the errors), and the whole
comparison against a second computation of every number it reports
(tests/impinging_jet_reference.py: the lesson of test 52).
"""

import hashlib
import json

import numpy as np
import pytest

from src.boundary_staggered import StaggeredBoundary
from src.mesh import Mesh
from src.turbulence import Y_STAR_FLOOR
from tests import impinging_jet_reference as reference
from validation import impinging_jet as ij
from validation.cases import (
    CASE_GRIDS,
    IMPINGING_JET_GRIDS,
    IMPINGING_JET_WALL_FRACTION,
    load_case,
    load_impinging_jet,
    load_preset,
)

V0_W = ij.V0 * ij.W_JET


@pytest.fixture(scope="module")
def coarse() -> tuple:
    config = load_preset("val020_272x90")
    return config, Mesh(config)


@pytest.mark.unit
class TestLoader:
    def test_the_presets_are_registered(self) -> None:
        assert {"val020_272x90", "val020_408x135"} == IMPINGING_JET_GRIDS
        for preset in IMPINGING_JET_GRIDS:
            kind, nx, ny = CASE_GRIDS[preset]
            assert kind == "impinging_jet"
            assert preset.endswith(f"{nx}x{ny}")

    @pytest.mark.parametrize(
        ("preset", "cells"), [("val020_272x90", 4), ("val020_408x135", 6)]
    )
    def test_the_nozzle_covers_whole_faces_and_carries_v0_w(
        self, preset: str, cells: int
    ) -> None:
        config = load_preset(preset)
        mesh = Mesh(config)
        boundary = StaggeredBoundary(mesh, config)
        spec = ij.nozzle(config)
        dx = config.room_width / config.nx
        assert spec.x_start / dx == pytest.approx(round(spec.x_start / dx), abs=1e-9)
        assert spec.x_end / dx == pytest.approx(round(spec.x_end / dx), abs=1e-9)
        assert (spec.x_end - spec.x_start) / dx == pytest.approx(cells)
        assert boundary.get_inlet_flux(ij.INLET) == pytest.approx(V0_W, rel=1e-12)
        assert ij.axis_x(config) == pytest.approx(0.5 * config.room_width)
        # The axis is a face: a centre column lies dx / 2 on each side of it.
        assert np.min(np.abs(mesh.xc - ij.axis_x(config))) == pytest.approx(0.5 * dx)

    @pytest.mark.parametrize("preset", sorted(IMPINGING_JET_GRIDS))
    def test_the_two_sides_share_the_inflow(self, preset: str) -> None:
        """Each fixed-flow outlet takes half of V0 w at one face velocity."""
        config = load_preset(preset)
        mesh = Mesh(config)
        boundary = StaggeredBoundary(mesh, config)
        speeds = boundary.fixed_flow_velocities()
        assert set(speeds) == {"outlet_left", "outlet_right"}
        for name, speed in speeds.items():
            spec = config.boundaries[name]
            assert (spec.y_start, spec.y_end) == (0.0, config.room_height)
            assert speed * config.room_height == pytest.approx(0.5 * V0_W, rel=1e-12)
        assert not boundary.has_pressure_outlet()

    @pytest.mark.parametrize("preset", sorted(IMPINGING_JET_GRIDS))
    def test_the_wall_cell_is_half_the_uniform_spacing(self, preset: str) -> None:
        config = load_preset(preset)
        mesh = Mesh(config)
        uniform = config.room_height / config.ny
        assert IMPINGING_JET_WALL_FRACTION == 0.5
        assert mesh.dy_cell[0] == pytest.approx(0.5 * uniform, rel=1e-9)
        assert mesh.dy_cell[-1] == pytest.approx(0.5 * uniform, rel=1e-9)
        assert mesh.stretch_ratio_x == 1.0
        assert mesh.stretch_ratio_y > 1.0

    def test_the_file_alone_loads_the_loaders_grid(self) -> None:
        file = load_case("impinging_jet")
        loaded = load_preset("val020_272x90")
        assert (file.nx, file.ny) == (272, 90)
        assert file.stretch_y.min_spacing == pytest.approx(loaded.stretch_y.min_spacing)
        assert np.allclose(Mesh(file).y, Mesh(loaded).y)

    def test_the_geometry_and_the_flow_are_the_papers(self) -> None:
        config = load_case("impinging_jet")
        spec = ij.nozzle(config)
        assert spec.x_end - spec.x_start == pytest.approx(ij.W_JET)
        assert config.room_height == pytest.approx(ij.H_JET)
        assert ij.H_JET / ij.W_JET == 22.5
        assert config.room_width == pytest.approx(68 * ij.W_JET)
        assert spec.velocity == pytest.approx(ij.V0)
        nu = config.mu / config.rho
        assert nu == pytest.approx(ij.NU)
        assert ij.V0 * ij.W_JET / nu == pytest.approx(ij.RE)
        assert config.obstacles == []

    def test_the_inlet_turbulence_is_the_papers_core_intensity(self) -> None:
        """k = 0.75 I_uv^2 V0^2 with I_uv 0.05; l_e = w / 10."""
        config = load_case("impinging_jet")
        spec = ij.nozzle(config)
        assert spec.turbulence_intensity == pytest.approx(0.05 / np.sqrt(2.0), rel=1e-3)
        k = 1.5 * (spec.turbulence_intensity * ij.V0) ** 2
        assert k == pytest.approx(ij.K_ISOTROPIC * 0.05**2 * ij.V0**2, rel=1e-3)
        assert spec.dissipation_length == pytest.approx(ij.W_JET / 10)

    def test_the_file_holds_the_model_and_the_solver_keys_as_val019_runs_them(
        self,
    ) -> None:
        config = load_case("impinging_jet")
        assert config.turbulence is not None
        assert config.turbulence.variant == "standard"
        assert config.turbulence.advection_scheme == "upwind"
        assert config.turbulence.cfl_number == 0.25
        assert config.turbulence.tol == 1e-10
        assert config.turbulence.max_iter == 500
        assert config.stopping_rule == "error_estimate"
        assert config.momentum_sweeps == 10
        assert config.pressure_rtol == 1e-4
        assert config.max_simple_iter == 20000
        assert config.pressure_preconditioner == "multigrid"

    def test_a_grid_that_puts_the_nozzle_edge_off_a_face_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must put it on a face"):
            load_impinging_jet("impinging_jet", grid=(270, 90))

    def test_a_case_without_the_nozzle_is_refused(self, coarse: tuple) -> None:
        config, _mesh = coarse
        other = load_case("annex20")
        with pytest.raises(ValueError, match="velocity inlet named 'nozzle'"):
            ij.nozzle(other)
        assert ij.nozzle(config).location == "top"


@pytest.mark.unit
class TestData:
    def test_the_counts_and_the_order(self) -> None:
        assert len(ij.CENTRELINE_ROWS) == 51
        assert len(ij.INTENSITY_ROWS) == 40
        assert len(ij.HALF_WIDTH_ROWS) == 11
        assert len(ij.LATERAL_V_48_ROWS) == 19
        assert len(ij.LATERAL_I_48_ROWS) == 23
        assert np.all(np.diff(ij.CENTRELINE_Y) > 0)
        assert np.all(np.diff(ij.INTENSITY_Y) >= 0)
        assert np.all(np.diff(ij.HALF_WIDTH_Y) > 0)
        assert np.all(np.diff([r[0] for r in ij.LATERAL_V_48_ROWS]) > 0)
        assert np.all(np.diff([r[0] for r in ij.LATERAL_I_48_ROWS]) > 0)
        assert ij.CENTRELINE_Y[0] > 0.06 and ij.CENTRELINE_Y[-1] < 1.0
        assert sum(r[2] for r in ij.INTENSITY_ROWS) == 30

    def test_the_ranges_are_the_papers(self) -> None:
        """The free-jet, impingement and plate-median ranges, by value (test 55 S1).

        The second computation (tests/impinging_jet_reference.py) reads the
        same constants, so only a test of the values catches a changed one;
        they set which stations enter prediction (c) and the y+ median.
        """
        assert ij.FREE_JET_H == 0.70
        assert ij.NARROW_FREE_JET_H == 0.30
        assert ij.IMPINGEMENT_H == 0.88
        assert ij.PLATE_MEDIAN_HALF_WIDTH_W == 10.0

    def test_the_digitised_curves_match_the_numbers_the_paper_states(self) -> None:
        """Section 4.1: the potential core, the half-velocity point, peak I_uv, K_y."""
        core = ij.potential_core_w(ij.CENTRELINE_Y, ij.CENTRELINE_V)
        assert core == pytest.approx(ij.STATED["potential_core_w"], abs=0.1)
        half = ij.first_crossing(ij.CENTRELINE_Y, ij.CENTRELINE_V, 0.5)
        assert half == pytest.approx(ij.STATED["half_velocity_h"], abs=0.04)
        rings = np.array([r[2] for r in ij.INTENSITY_ROWS])
        free = (ij.INTENSITY_Y < 0.8) & rings
        peak = int(np.argmax(np.where(free, ij.INTENSITY_I, -1.0)))
        assert ij.INTENSITY_I[peak] == pytest.approx(
            ij.STATED["i_uv_peak"][0], abs=0.01
        )
        assert ij.INTENSITY_Y[peak] == pytest.approx(
            ij.STATED["i_uv_peak"][1], abs=0.02
        )
        k_y = ij.spreading_rate(ij.HALF_WIDTH_Y, ij.HALF_WIDTH_DELTA)
        assert k_y == pytest.approx(ij.STATED["spreading_rate"], abs=0.005)
        assert ij.HALF_WIDTH_DELTA[0] == pytest.approx(0.5, abs=0.05)
        k_u = ij.decay_rate(ij.CENTRELINE_Y, ij.CENTRELINE_V)
        assert 0.13 < k_u < 0.21

    def test_the_lateral_profile_peaks_on_the_axis_and_the_gaussian_is_the_fit(
        self,
    ) -> None:
        x = np.array([r[0] for r in ij.LATERAL_V_48_ROWS])
        v = np.array([r[1] for r in ij.LATERAL_V_48_ROWS])
        assert v.max() > 0.97 and v.min() < 0.03
        assert abs(x[np.argmax(v)]) < 0.5
        assert ij.gaussian(np.array([0.0, 1.0]))[0] == 1.0
        assert ij.gaussian(np.array([1.0]))[0] == pytest.approx(0.5, abs=0.001)
        assert ij.GAUSSIAN_STATIONS[0] == -2.0 and ij.GAUSSIAN_STATIONS[-1] == 2.0

    def test_the_heights_convert_through_the_virtual_origin(self) -> None:
        assert ij.height_over_h(4.8) == pytest.approx(4.0 * ij.W_JET / ij.H_JET)
        assert ij.height_over_h(20.1) == pytest.approx(19.3 / 22.5)

    def test_the_stored_values_hash_as_printed(self) -> None:
        """Pins every stored value: a changed one fails here (VAL-017's device).

        The values were printed by docs/reports/probe55/data55.py from the
        digitiser's record; the hash is of their repr.
        """
        payload = repr(
            [
                ij.CENTRELINE_ROWS,
                ij.INTENSITY_ROWS,
                ij.HALF_WIDTH_ROWS,
                ij.LATERAL_V_48_ROWS,
                ij.LATERAL_I_48_ROWS,
            ]
        )
        assert hashlib.sha256(payload.encode()).hexdigest() == (
            "48ba10ff72f9c7d0ddaf77e7bfd57c3a5757075faced7af96840c27d7ee201ad"
        )


@pytest.mark.unit
class TestSampling:
    def test_a_linear_field_is_sampled_exactly_on_both_lattices(
        self, coarse: tuple
    ) -> None:
        _config, mesh = coarse
        xc, yf = np.meshgrid(mesh.xc, mesh.y)
        v = 0.3 * xc - 0.7 * yf + 0.1
        x = np.array([0.1, 0.544, 0.9, 1.0])
        y = np.array([0.0, 0.1, 0.2, 0.36])
        assert (
            np.abs(ij.sample_v(mesh, v, x, y) - (0.3 * x - 0.7 * y + 0.1)).max() < 1e-12
        )
        xc2, yc = np.meshgrid(mesh.xc, mesh.yc)
        k = 2.0 * xc2 + 5.0 * yc - 1.0
        inside = np.array([0.2, 0.5, 0.8])
        yi = np.array([0.05, 0.15, 0.3])
        assert (
            np.abs(
                ij.sample_cells(mesh, k, inside, yi) - (2.0 * inside + 5.0 * yi - 1.0)
            ).max()
            < 1e-12
        )

    def test_a_lattice_point_reads_its_own_value(self, coarse: tuple) -> None:
        _config, mesh = coarse
        rng = np.random.default_rng(55)
        v = rng.standard_normal((mesh.y.size, mesh.xc.size))
        assert (
            ij.sample_v(mesh, v, np.array([mesh.xc[7]]), np.array([mesh.y[3]]))[0]
            == v[3, 7]
        )
        k = rng.standard_normal((mesh.yc.size, mesh.xc.size))
        assert (
            ij.sample_cells(mesh, k, np.array([mesh.xc[9]]), np.array([mesh.yc[4]]))[0]
            == k[4, 9]
        )

    def test_the_frames_convert_both_ways(self) -> None:
        assert ij.solver_y(0.0) == pytest.approx(ij.H_JET)
        assert ij.solver_y(1.0) == 0.0
        assert ij.paper_y(ij.solver_y(np.array([0.25, 0.9]))) == pytest.approx(
            [0.25, 0.9]
        )


@pytest.mark.unit
class TestReaders:
    def test_the_axis_fit_is_exact_on_a_parabola(self) -> None:
        """a + b x^2 through four columns gives a, as (9 V_d - V_3d) / 8 does."""
        x = np.array([-0.006, -0.002, 0.002, 0.006])
        values = 0.8 - 3.0 * x**2
        assert ij.axis_fit(x, values) == pytest.approx(0.8, rel=1e-12)
        v_d, v_3d = 0.8 - 3.0 * 0.002**2, 0.8 - 3.0 * 0.006**2
        assert (9 * v_d - v_3d) / 8 == pytest.approx(0.8, rel=1e-12)
        stacked = np.stack([values, 2.0 * values])
        assert ij.axis_fit(x, stacked) == pytest.approx([0.8, 1.6], rel=1e-12)

    def test_the_axis_fit_reads_a_four_cell_nozzle_better_than_the_nearest_column(
        self,
    ) -> None:
        """A Gaussian of half-width 2 dx: the nearest column 0.958, the fit 0.993."""
        dx = 1.0
        x = np.array([-1.5, -0.5, 0.5, 1.5]) * dx
        values = np.exp(ij.GAUSS_A * (x / (2.0 * dx)) ** 2)
        assert values[1] == pytest.approx(0.9576, abs=1e-3)
        assert ij.axis_fit(x, values) == pytest.approx(0.9927, abs=1e-3)

    def test_the_axis_columns_are_the_two_nearest_on_each_side(
        self, coarse: tuple
    ) -> None:
        config, mesh = coarse
        cols = ij.axis_columns(mesh, config)
        x = mesh.xc[cols] - ij.axis_x(config)
        assert np.allclose(x, [-0.006, -0.002, 0.002, 0.006])

    def test_first_crossing_by_hand(self) -> None:
        y = np.array([0.1, 0.2, 0.3])
        assert ij.first_crossing(y, np.array([1.0, 0.96, 0.9]), 0.98) == pytest.approx(
            0.15
        )
        assert ij.first_crossing(y, np.array([0.9, 0.8, 0.7]), 0.98) == 0.1
        assert ij.potential_core_w(y, np.array([1.0, 0.96, 0.9])) == pytest.approx(
            0.15 * 22.5
        )
        with pytest.raises(ValueError, match="never falls below"):
            ij.first_crossing(y, np.array([1.0, 1.0, 1.0]), 0.98)
        with pytest.raises(ValueError, match="must increase"):
            ij.first_crossing(np.array([0.3, 0.2]), np.array([1.0, 0.5]), 0.98)

    def test_the_decay_rate_of_an_exact_line_and_its_range(self) -> None:
        """(V0 / V)^2 = 0.2 (y / w - y0 / w): K_u = 0.2 on the self-similar range."""
        y = np.array([0.2, 0.35, 0.5, 0.75, 0.9])
        distance = y * ij.H_JET / ij.W_JET - ij.VIRTUAL_ORIGIN_W
        v = 1.0 / np.sqrt(0.2 * distance)
        v[0], v[-1] = 10.0, 10.0  # outside the range, ignored
        assert ij.decay_rate(y, v) == pytest.approx(0.2, rel=1e-9)
        with pytest.raises(ValueError, match="fewer than two"):
            ij.decay_rate(np.array([0.1, 0.9]), np.array([1.0, 0.5]))
        with pytest.raises(ValueError, match="is zero"):
            ij.decay_rate(np.array([0.4, 0.5]), np.array([0.0, 0.5]))

    def test_the_spreading_rate_of_an_exact_line(self) -> None:
        y = np.array([2.0, 8.0, 12.0, 16.0, 20.0])
        delta = 0.5 + 0.11 * y
        delta[0], delta[-1] = 9.0, 9.0  # outside the range, ignored
        assert ij.spreading_rate(y, delta) == pytest.approx(0.11, rel=1e-9)
        with pytest.raises(ValueError, match="fewer than two"):
            ij.spreading_rate(np.array([2.0, 20.0]), np.array([0.5, 2.0]))

    def test_the_half_width_of_a_flat_topped_jet_by_hand(self, coarse: tuple) -> None:
        """A plateau over the axis columns, then a linear fall: the reader is exact.

        V = V_c for |x| <= 8 mm and V_c (1 - (|x| - 8 mm) / L) beyond, with L
        20 mm on the left and 40 mm on the right, so the speed halves at 18
        and 28 mm and the half-width is their mean, 23 mm = 1.4375 w. The
        plateau covers the four columns the axis fit reads, so the fit is
        V_c exactly, and the fall is linear across the crossing, so the
        interpolation between centre columns is exact.
        """
        config, mesh = coarse
        axis = ij.axis_x(config)
        xc, _yf = np.meshgrid(mesh.xc, mesh.y)
        plateau = 0.008
        fall = np.where(xc < axis, 0.020, 0.040)
        v = -0.4 * np.minimum(1.0, 1.0 - (np.abs(xc - axis) - plateau) / fall)
        got = ij.half_width_w(mesh, config, v, 0.5)
        assert got == pytest.approx(0.5 * (0.018 + 0.028) / ij.W_JET, rel=1e-9)
        xi = np.array([-1.0, 0.0, 1.0])
        profile, d = ij.self_similar(mesh, config, v, 0.5, xi)
        assert d == pytest.approx(got)
        # The profile is read with the mean half-width (23 mm), so the
        # narrow side reads 1 - 15 / 20 and the wide side 1 - 15 / 40.
        assert profile[1] == pytest.approx(1.0, rel=1e-9)
        assert profile[0] == pytest.approx(0.25, rel=1e-9)
        assert profile[2] == pytest.approx(0.625, rel=1e-9)

    def test_the_half_width_refuses_a_jet_that_never_halves(
        self, coarse: tuple
    ) -> None:
        config, mesh = coarse
        v = -0.3 * np.ones((mesh.y.size, mesh.xc.size))
        with pytest.raises(ValueError, match="never halves"):
            ij.half_width_w(mesh, config, v, 0.5)
        with pytest.raises(ValueError, match="not positive"):
            ij.half_width_w(mesh, config, np.zeros_like(v), 0.5)

    def test_the_centreline_of_a_field_linear_in_y(self, coarse: tuple) -> None:
        """v = -V0 (1 - 0.5 y / H), uniform in x: V_0,y / V0 = 0.5 + 0.5 s."""
        config, mesh = coarse
        _xc, yf = np.meshgrid(mesh.xc, mesh.y)
        v = -ij.V0 * (1.0 - 0.5 * yf / ij.H_JET)
        s = np.array([0.1, 0.5, 0.9])
        assert ij.centreline(mesh, config, v, s) == pytest.approx(
            0.5 + 0.5 * s, rel=1e-12
        )

    def test_errors_by_hand(self) -> None:
        e = ij.errors(np.array([1.0, 2.0, 4.0]), np.array([1.0, 1.0, 1.0]))
        assert (e.max_abs, e.mean, e.stations) == (3.0, 4.0 / 3.0, 3)
        assert e.rms == pytest.approx(np.sqrt(10.0 / 3.0))
        with pytest.raises(ValueError, match="against"):
            ij.errors(np.zeros(3), np.zeros(2))
        with pytest.raises(ValueError, match="no stations"):
            ij.errors(np.zeros(0), np.zeros(0))
        with pytest.raises(ValueError, match="not finite"):
            ij.errors(np.array([np.nan]), np.array([0.0]))

    def test_the_intensity_estimate_inverts_the_isotropic_k(self) -> None:
        i_uv = np.array([0.05, 0.2])
        assert ij.intensity_from_k(ij.K_ISOTROPIC * i_uv**2) == pytest.approx(i_uv)

    def test_the_stagnation_readings_and_y_star_on_a_hand_made_k(
        self, coarse: tuple
    ) -> None:
        config, mesh = coarse
        k = np.full((mesh.yc.size, mesh.xc.size), 0.01 * ij.V0**2)
        j = 3  # a row inside the impingement region, above the first cell
        assert ij.paper_y(mesh.yc[j]) > ij.IMPINGEMENT_H
        k[j, :] = 0.03 * ij.V0**2
        got = ij.stagnation(mesh, config, k)
        assert got["first_cell"] == pytest.approx(0.01, rel=1e-12)
        assert got["impingement_max"] == pytest.approx(0.03, rel=1e-12)
        assert got["impingement_max_y_over_h"] == pytest.approx(ij.paper_y(mesh.yc[j]))
        y_star = ij.plate_y_star(mesh, config, k)
        c_mu = 0.09
        nu = config.mu / config.rho
        expect = c_mu**0.25 * np.sqrt(0.01 * ij.V0**2) * (mesh.yc[0] - mesh.y[0]) / nu
        assert y_star["median"] == pytest.approx(expect, rel=1e-9)
        assert y_star["at_axis"] == pytest.approx(expect, rel=1e-9)
        assert y_star["largest"] == pytest.approx(expect, rel=1e-9)
        assert y_star["floor"] == Y_STAR_FLOOR


@pytest.mark.unit
class TestReport:
    def test_the_report_is_whole_and_serialisable(self, coarse: tuple) -> None:
        config, mesh = coarse
        v, k = reference.fields(mesh, config)
        report = ij.evaluate(mesh, config, v, k)
        json.dumps(report)
        assert set(report) == {
            "centreline",
            "potential_core_w",
            "half_velocity_h",
            "decay_rate",
            "half_width",
            "lateral",
            "lateral_intensity_48",
            "k_axis",
            "stagnation",
            "y_star",
        }
        assert report["centreline"]["all"]["stations"] == 51
        assert report["k_axis"]["errors_isotropic"]["stations"] == 30
        assert report["k_axis"]["errors_isotropic_all"]["stations"] == 40
        assert report["stagnation"]["data_station_ring"] is False
        assert report["stagnation"]["ring_station_y_over_h"] == 0.9731
        assert report["half_width"]["errors"]["stations"] == 11
        assert set(report["lateral"]) == {"4.8", "5.9", "7.9", "12.1", "14.9", "20.1"}
        assert report["lateral"]["4.8"]["source"] == "digitised"
        assert report["lateral"]["20.1"]["source"] == "gaussian"
        assert report["y_star"]["floor"] == Y_STAR_FLOOR

    def test_every_number_of_the_report_is_the_constructed_fields(
        self, coarse: tuple
    ) -> None:
        """Each value of ``evaluate`` against a second computation on one known field.

        The field (tests/impinging_jet_reference.py) has a centreline above
        V0 at the nozzle, a half-width that differs between the sides, a
        linear collapse in the impingement region and a k with its axis
        maximum inside the impingement region but not in the first cell.
        Defects caught: the axis read from the nearest column, the
        half-width from one side, the stagnation k from the first cell
        alone, the errors over the wrong range, y* over the whole plate.
        """
        config, mesh = coarse
        v, k = reference.fields(mesh, config)
        got = ij.evaluate(mesh, config, v, k)
        want = reference.expected(mesh, config, v, k)
        reference.assert_matches(got, want)
        # The field is built so that the readings differ from their wrong forms.
        core = got["potential_core_w"]["computed"]
        assert core == pytest.approx(
            (reference.CENTRE_TOP - 0.98) / reference.CENTRE_SLOPE * 22.5, rel=0.1
        )
        assert (
            got["stagnation"]["impingement_max"] > 1.2 * got["stagnation"]["first_cell"]
        )
        assert got["stagnation"]["impingement_max_y_over_h"] == pytest.approx(
            reference.K_PEAK_S, abs=0.02
        )
        assert got["y_star"]["largest"] > 1.05 * got["y_star"]["median"]
        d = got["lateral"]["4.8"]["delta_over_w"]
        assert d == pytest.approx(
            (reference.WIDTH_AT_NOZZLE + reference.WIDTH_SLOPE * 4.0)
            * 0.5
            * (1.0 + reference.WIDTH_SKEW),
            rel=0.05,
        )

    def test_a_jet_that_never_halves_reports_none_and_the_rest(
        self, coarse: tuple
    ) -> None:
        config, mesh = coarse
        _v, k = reference.fields(mesh, config)
        v = -0.3 * ij.V0 * np.ones((mesh.y.size, mesh.xc.size))
        report = ij.evaluate(mesh, config, v, k)
        json.dumps(report)
        assert report["half_width"]["computed"] == [None] * 11
        assert report["half_width"]["errors"] is None
        assert report["lateral"]["4.8"]["errors"] is None
        assert report["lateral_intensity_48"]["errors"] is None
        assert report["potential_core_w"]["computed"] == pytest.approx(
            ij.CENTRELINE_Y[0] * 22.5
        )
        assert report["half_velocity_h"]["computed"] == pytest.approx(
            ij.CENTRELINE_Y[0]
        )
        assert report["k_axis"]["errors_isotropic"]["stations"] == 30
        assert report["k_axis"]["errors_isotropic_all"]["stations"] == 40
        assert report["stagnation"]["data_station_ring"] is False
        assert report["stagnation"]["ring_station_y_over_h"] == 0.9731
        assert report["centreline"]["all"]["stations"] == 51
