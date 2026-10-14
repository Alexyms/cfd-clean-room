"""VAL-019, the backward-facing step (ECR-002 step 7; ADR-012 G (ii)): the unit tests.

The case is configs/validation_backstep.yaml, loaded by validation.cases.
load_backstep with y clustered so that a face falls on the step height on
any grid, and compared with Driver and Seegmiller's (1985) data by
validation.backstep, which stores the NASA Turbulence Modeling Resource's
three files with their provenance (validation/data/driver_seegmiller_1985/).
No allowance is set (ADR-012 G (ii), OPEN); the comparison reports.

The run itself is not here. It is the harness row ``val019_656x60``
(``python scripts/benchmark.py --cases val019_656x60 --repeats 1``),
reported and not scored, recorded in benchmarks/results.jsonl; the first
coupled results, with the resolution study, the comparison tables and the
figures, are docs/reports/ecr002_val019_backstep.md.

These tests pin the loader (the geometry, the corner on grid lines on every
preset, the inlet's coverage, the pressure outlet, the file's settings, the
refusals), the stored data (counts, stations, the files on disk against the
module and their checksums, the reattachment the measured Cf itself gives,
and a hash of every stored value), the sampler, the inflow reader and the
wall readers (the shear, the reattachment on constructed shear fields, the
coefficients, the pressure datum, y*) and the whole report's shape.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.boundary_staggered import StaggeredBoundary
from src.mesh import SOLID, Mesh
from src.turbulence import VARIANTS, Y_STAR_FLOOR, wall_viscosity
from tests import backstep_reference as reference
from validation import backstep
from validation.cases import (
    BACKSTEP_GRIDS,
    CASE_GRIDS,
    load_backstep,
    load_case,
    load_preset,
    step_corner_offsets,
    step_face_index,
    wall_spacing_for_face,
)

H = backstep.STEP_HEIGHT
L_UP_H = 114
TOTAL_H = L_UP_H + backstep.OUTLET_STATION
MEDIUM = "val019_656x60"
DATA_DIR = Path(__file__).resolve().parent.parent / "validation" / "data"
DATA_DIR = DATA_DIR / "driver_seegmiller_1985"
CHECKSUMS = {
    "profiles.exp.dat": "23e8b44db52bfc36c66af6b743a4080c34f0165eb213a3db29c0914e12cf598d",
    "cf.exp.dat": "cd9b6434b7956bc58e859de6af654cd7f891e25303babf1dd8633d6c58f74276",
    "cp.expnew.dat": "2b3ec9f8d8025b958d6ddcf6767daaf3c93083d45ef350a8cf9c4b46f93a882a",
}


@pytest.fixture(scope="module")
def medium() -> tuple:
    config = load_preset(MEDIUM)
    return config, Mesh(config)


def zones(path: Path) -> dict[str, list[list[float]]]:
    """The numeric rows of one of the Resource's files, keyed by zone title."""
    out: dict[str, list[list[float]]] = {}
    title = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text[0] == "#" or text.startswith("variables"):
            continue
        if text.startswith("zone"):
            title = text.split('"')[1]
            out.setdefault(title, [])
            continue
        out.setdefault(title, []).append([float(v) for v in text.split()])
    return out


@pytest.mark.unit
class TestLoader:
    def test_the_presets_are_registered(self) -> None:
        assert {"val019_410x40", MEDIUM, "val019_984x90"} == BACKSTEP_GRIDS
        for preset in BACKSTEP_GRIDS:
            kind, nx, ny = CASE_GRIDS[preset]
            assert kind == "backstep"
            assert preset.endswith(f"{nx}x{ny}")
            # 114 of 164 step heights: the step's face is on an x face only
            # when nx is a multiple of 82.
            assert nx % 82 == 0

    @pytest.mark.parametrize("preset", sorted(BACKSTEP_GRIDS))
    def test_the_corner_lies_on_grid_lines_on_every_preset(self, preset: str) -> None:
        """Both offsets at rounding; the y face is the one ny / 6 from the floor."""
        config = load_preset(preset)
        mesh = Mesh(config)
        dx_off, dy_off = step_corner_offsets(mesh, config)
        assert dx_off < 1e-12 and dy_off < 1e-12
        assert backstep.step_row(mesh, config) == step_face_index(config.ny)
        assert backstep.step_column(mesh, config) == config.nx * L_UP_H // TOTAL_H
        assert mesh.stretch_ratio_x == 1.0 and mesh.stretch_ratio_y > 1.0

    @pytest.mark.parametrize("preset", sorted(BACKSTEP_GRIDS))
    def test_the_step_is_the_solid_block_under_the_inlet(self, preset: str) -> None:
        config = load_preset(preset)
        mesh = Mesh(config)
        j, i = backstep.step_row(mesh, config), backstep.step_column(mesh, config)
        solid = mesh.cell_type == SOLID
        assert solid.sum() == j * i
        assert solid[:j, :i].all()
        assert not solid[j:, :].any() and not solid[:, i:].any()

    def test_the_file_alone_loads_the_loaders_grid(self) -> None:
        """The file states the derived spacing for its own grid, to the last digit."""
        from_file, from_loader = load_case("backstep"), load_preset(MEDIUM)
        assert (from_file.nx, from_file.ny) == (656, 60)
        assert from_file.stretch_y.min_spacing == from_loader.stretch_y.min_spacing
        assert np.array_equal(Mesh(from_file).y, Mesh(from_loader).y)

    def test_the_geometry_is_the_cases(self) -> None:
        """H 0.0127 m, 8 H upstream over the block, 9 H downstream, outlet at +50 H."""
        config = load_case("backstep")
        step = backstep.step(config)
        assert step.y_end == H and step.y_start == 0.0 and step.x_start == 0.0
        assert step.x_end == pytest.approx(L_UP_H * H)
        assert config.room_height == pytest.approx(backstep.DOWNSTREAM_HEIGHTS * H)
        assert config.room_width == pytest.approx(TOTAL_H * H)
        assert backstep.data_x(config, config.room_width) == pytest.approx(50.0)
        inlet = config.boundaries["inlet"]
        assert (inlet.y_start, inlet.y_end) == (H, config.room_height)
        outlet = config.boundaries["outlet"]
        assert outlet.type == "pressure_outlet" and outlet.location == "right"
        assert (outlet.y_start, outlet.y_end) == (0.0, config.room_height)

    def test_the_flow_is_the_cases(self) -> None:
        """Re_H 36,000 on U_ref; the inlet below U_ref, with the stated I and l_e."""
        config = load_case("backstep")
        nu = config.mu / config.rho
        assert nu == pytest.approx(backstep.NU)
        assert backstep.U_REF * H / nu == pytest.approx(36000.0)
        inlet = config.boundaries["inlet"]
        # The channel accelerates as its layers grow, so the inlet is slower
        # than the centreline it must make; the precursor gave 0.947 U_ref.
        # Pinned to the file's value: a looser bound let 43.5 m/s through,
        # a centreline 3% high, and every comparison is normalised by it.
        assert inlet.velocity == 41.862
        assert inlet.turbulence_intensity == 0.01
        assert inlet.dissipation_length == pytest.approx(H / 10)

    def test_the_file_holds_the_model_and_the_solver_keys_as_val017_runs_them(
        self,
    ) -> None:
        config = load_case("backstep")
        assert config.turbulence is not None
        assert config.turbulence.variant == "standard"
        assert config.turbulence.advection_scheme == "upwind"
        assert config.turbulence.cfl_number == 0.25
        assert config.turbulence.tol == 1.0e-10
        assert config.stopping_rule == "error_estimate"
        assert config.momentum_sweeps == 10
        assert config.pressure_rtol == 1.0e-4
        assert config.max_simple_iter == 20000
        assert config.alpha_velocity == 0.5

    @pytest.mark.parametrize("preset", sorted(BACKSTEP_GRIDS))
    def test_the_inlet_covers_the_channel_and_the_outlet_is_open(
        self, preset: str
    ) -> None:
        """The inlet carries U_in 8 H; the right edge is a pressure outlet, no fixed flow."""
        config = load_preset(preset)
        mesh = Mesh(config)
        boundary = StaggeredBoundary(mesh, config)
        u_in = config.boundaries["inlet"].velocity
        assert boundary.get_inlet_flux("inlet") == pytest.approx(
            u_in * backstep.UPSTREAM_HEIGHTS * H, rel=1e-12
        )
        assert boundary.has_pressure_outlet()
        assert boundary.fixed_flow_velocities() == {}
        assert boundary.pressure_outlets()["right"].is_outlet.all()
        assert not boundary.pressure_outlets()["left"].is_outlet.any()

    def test_a_grid_that_puts_the_step_face_off_a_grid_line_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must put it on a face"):
            load_backstep("backstep", grid=(655, 60))

    def test_the_face_index_rule(self) -> None:
        assert [step_face_index(n) for n in (40, 60, 90, 6, 7)] == [7, 10, 15, 1, 1]
        with pytest.raises(ValueError, match="at least 6 cells"):
            step_face_index(5)

    @pytest.mark.parametrize(
        ("n", "k", "target"), [(40, 7, 0.1), (61, 10, 0.1), (12, 2, 0.05)]
    )
    def test_the_wall_spacing_puts_the_face_on_the_target(
        self, n: int, k: int, target: float
    ) -> None:
        from src.mesh import geometric_faces, ratio_for_wall_spacing

        length = 1.0
        h = wall_spacing_for_face(length, n, k, target)
        faces = geometric_faces(length, n, ratio_for_wall_spacing(length, n, h))
        assert abs(faces[k] - target) < 1e-12
        assert 0.0 < h < length / n

    def test_the_wall_spacing_refuses_what_clustering_cannot_reach(self) -> None:
        assert wall_spacing_for_face(1.0, 10, 2, 0.2) == 0.1
        with pytest.raises(ValueError, match="can reach"):
            wall_spacing_for_face(1.0, 10, 2, 0.25)
        with pytest.raises(ValueError, match="face index"):
            wall_spacing_for_face(1.0, 10, 6, 0.1)


@pytest.mark.unit
class TestData:
    def test_the_profiles_hold_the_files_stations_and_heights(self) -> None:
        assert tuple(backstep.PROFILES) == (-4.0, 1.0, 4.0, 6.0, 10.0)
        assert backstep.PROFILE_STATIONS == (1.0, 4.0, 6.0, 10.0)
        counts = {-4.0: 17, 1.0: 28, 4.0: 28, 6.0: 28, 10.0: 28}
        for station, profile in backstep.PROFILES.items():
            assert profile.station == station
            assert len(profile.y_over_h) == len(profile.u_over_uref) == counts[station]
            assert len(profile.uv) == counts[station]
            assert np.all(np.diff(profile.y_over_h) > 0)
            # y is measured from the downstream floor: the upstream profile
            # starts above the step's top, the others above the floor.
            assert profile.y_over_h[0] == (1.1 if station == -4.0 else 0.1)
            assert profile.y_over_h[-1] == 8.2

    def test_the_wall_lines_hold_the_files_stations(self) -> None:
        assert len(backstep.CF_STATIONS) == 20
        assert np.all(np.diff(backstep.CF_STATIONS) > 0)
        assert backstep.CF_STATIONS[0] < 0 < backstep.CF_STATIONS[3]
        assert len(backstep.CP_BOTTOM_STATIONS) == 33
        assert len(backstep.CP_TOP_STATIONS) == 19
        assert np.all(np.diff(backstep.CP_BOTTOM_STATIONS) > 0)
        assert backstep.CP_BOTTOM_STATIONS[-1] == backstep.CP_ZERO_STATION
        assert backstep.CP_BOTTOM_MEASURED[-1] == 0.0

    def test_the_shifted_cp_is_the_original_less_one_constant(self) -> None:
        shifts = {round(r[1] - r[2], 4) for r in backstep.CP_BOTTOM_ROWS}
        shifts |= {round(r[1] - r[2], 4) for r in backstep.CP_TOP_ROWS}
        assert shifts == {-0.1141}

    def test_the_measured_cf_reattaches_inside_the_stated_bracket(self) -> None:
        """The reader on the data's own Cf: 6.28, within 6.26 +/- 0.10.

        The crossing lies between the stations at 5.882 and 7.090; the data
        page's value came from a finer interpolation of the same instrument.
        """
        x_r = backstep.reattachment(backstep.CF_STATIONS, backstep.CF_MEASURED)
        assert abs(x_r - backstep.X_R_MEASURED) < backstep.X_R_UNCERTAINTY
        negative = [
            x
            for x, cf in zip(backstep.CF_STATIONS, backstep.CF_MEASURED, strict=True)
            if cf < 0
        ]
        assert negative == pytest.approx([1.804, 2.804, 3.804, 4.804, 5.882])

    def test_the_published_lengths_say_which_model_they_are(self) -> None:
        values = backstep.PUBLISHED_X_R
        assert sorted(values.values()) == [5.30, 5.31, 5.55, 6.43]
        for name in values:
            if "k-epsilon" in name:
                assert "low-Re" in name
        assert backstep.X_R_MEASURED == 6.26 and backstep.X_R_UNCERTAINTY == 0.10

    def test_the_stored_values_are_the_files(self) -> None:
        """Every tuple equals the file on disk, parsed here, and the files are the fetched ones."""
        for name, digest in CHECKSUMS.items():
            assert hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest() == digest
        profiles = zones(DATA_DIR / "profiles.exp.dat")
        for title, rows in profiles.items():
            profile = backstep.PROFILES[float(title.split("x/H=")[1])]
            assert profile.y_over_h == tuple(r[1] for r in rows)
            assert profile.u_over_uref == tuple(r[2] for r in rows)
            assert profile.uv == tuple(r[6] for r in rows)
        cf = zones(DATA_DIR / "cf.exp.dat")[""]
        assert tuple(tuple(r) for r in cf) == backstep.CF_ROWS
        cp = zones(DATA_DIR / "cp.expnew.dat")
        assert tuple(tuple(r) for r in cp["bottom wall"]) == backstep.CP_BOTTOM_ROWS
        assert tuple(tuple(r) for r in cp["top wall"]) == backstep.CP_TOP_ROWS

    def test_the_stored_values_hash_as_printed(self) -> None:
        """Pins every stored value: a changed one fails here (VAL-017's device)."""
        payload = repr(
            [
                (s, p.y_over_h, p.u_over_uref, p.uv)
                for s, p in sorted(backstep.PROFILES.items())
            ]
            + [backstep.CF_ROWS, backstep.CP_BOTTOM_ROWS, backstep.CP_TOP_ROWS]
        )
        assert hashlib.sha256(payload.encode()).hexdigest() == (
            "40de679d622d9dc3a6d9ef36d19f7537a82a472ba90b36c36ad9c6298c5aac80"
        )


@pytest.mark.unit
class TestSampling:
    def test_the_frames_convert_both_ways(self, medium: tuple) -> None:
        config, _mesh = medium
        assert backstep.upstream_length(config) == pytest.approx(L_UP_H * H)
        x = backstep.case_x(config, np.array([-4.0, 0.0, 6.26]))
        assert x == pytest.approx([(L_UP_H - 4) * H, L_UP_H * H, (L_UP_H + 6.26) * H])
        assert backstep.data_x(config, x) == pytest.approx([-4.0, 0.0, 6.26])

    def test_a_linear_field_is_sampled_exactly(self, medium: tuple) -> None:
        config, mesh = medium
        x_faces, y_centres = np.meshgrid(mesh.x, mesh.yc)
        u = 0.3 * x_faces + 0.7 * y_centres + 0.1
        values = backstep.profile_values(mesh, config, u, 1.0)
        assert list(values) == list(backstep.PROFILES)
        for station, profile in backstep.PROFILES.items():
            y = np.array(profile.y_over_h) * H
            x = float(backstep.case_x(config, station))
            assert np.abs(values[station] - (0.3 * x + 0.7 * y + 0.1)).max() < 1e-12
        centre = backstep.centreline_speed(mesh, config, u)
        assert centre == pytest.approx(
            0.3 * float(backstep.case_x(config, -4.0)) + 0.7 * 5 * H + 0.1
        )

    def test_a_station_on_a_face_reads_that_face(self, medium: tuple) -> None:
        """x/H = 4 is 16 cells of a quarter step height past the step's face."""
        config, mesh = medium
        rng = np.random.default_rng(52)
        u = rng.standard_normal((mesh.yc.size, mesh.x.size))
        i = backstep.step_column(mesh, config) + 16
        assert mesh.x[i] == pytest.approx(float(backstep.case_x(config, 4.0)))
        j = 30
        got = backstep.sample_u(mesh, u, np.array([mesh.x[i]]), np.array([mesh.yc[j]]))
        assert got[0] == pytest.approx(u[j, i], rel=1e-12)

    def test_the_floor_and_the_ceiling_read_zero(self, medium: tuple) -> None:
        config, mesh = medium
        u = np.ones((mesh.yc.size, mesh.x.size))
        x = np.full(2, float(backstep.case_x(config, 10.0)))
        got = backstep.sample_u(mesh, u, x, np.array([0.0, config.room_height]))
        assert got == pytest.approx([0.0, 0.0])

    def test_errors_refuse_a_mismatch_an_empty_set_or_a_nan(self) -> None:
        e = backstep.errors(np.array([1.0, 2.0, 4.0]), np.array([0.0, 1.0, 1.0]))
        assert (e.max_abs, e.stations) == (3.0, 3)
        assert e.rms == pytest.approx(np.sqrt((1.0 + 1.0 + 9.0) / 3))
        with pytest.raises(ValueError, match="against"):
            backstep.errors(np.zeros(3), np.zeros(2))
        with pytest.raises(ValueError, match="no stations"):
            backstep.errors(np.zeros(0), np.zeros(0))
        with pytest.raises(ValueError, match="not finite"):
            backstep.errors(np.array([np.nan]), np.zeros(1))


def power_law_layer(y: np.ndarray, u_edge: float, delta: float) -> np.ndarray:
    """A 1/7-power turbulent layer: u = u_edge (y / delta)^(1/7) below delta."""
    return u_edge * np.minimum(y / delta, 1.0) ** (1.0 / 7.0)


@pytest.mark.unit
class TestInflow:
    def test_the_thicknesses_of_a_power_law_layer(self) -> None:
        """delta_99 = 0.99^7 delta and theta = (7/8 - 7/9) delta, to the trapezoid."""
        delta = 0.02
        y = np.linspace(1e-5, 0.05, 4000)
        d99, theta = backstep.layer_thicknesses(
            y, power_law_layer(y, 10.0, delta), 10.0
        )
        assert d99 == pytest.approx(0.99**7 * delta, rel=1e-3)
        assert theta == pytest.approx((7 / 8 - 7 / 9) * delta, rel=2e-3)

    def test_a_layer_that_never_reaches_the_edge_is_refused(self) -> None:
        y = np.linspace(0.001, 0.01, 10)
        with pytest.raises(ValueError, match="does not reach"):
            backstep.layer_thicknesses(y, np.full(10, 5.0), 10.0)

    def test_the_inflow_check_reads_the_layer_above_the_step(
        self, medium: tuple
    ) -> None:
        """A 1.5 H power-law layer on the step's top, the centreline U_ref: Re_theta near 5,000."""
        config, mesh = medium
        delta = 1.5 * H
        _x, yc = np.meshgrid(mesh.x, mesh.yc)
        u = power_law_layer(np.maximum(yc - H, 0.0), backstep.U_REF, delta)
        u[yc < H] = 0.0
        check = backstep.inflow_check(mesh, config, u)
        assert check.u_centre == pytest.approx(backstep.U_REF)
        assert check.u_centre_over_uref == pytest.approx(1.0)
        assert check.delta_99_over_h == pytest.approx(0.99**7 * 1.5, rel=0.02)
        # The trapezoid over the clustered rows (the first 0.06 H above the
        # wall) misses about 5% of a 1/7-power layer's theta; the reader's
        # integral is on the grid's rows by definition.
        theta = (7 / 8 - 7 / 9) * delta
        assert check.re_theta == pytest.approx(
            backstep.U_REF * theta / backstep.NU, rel=0.06
        )


@pytest.mark.unit
class TestWalls:
    def test_reattachment_interpolates_the_last_rising_crossing(self) -> None:
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        assert backstep.reattachment(x, np.array([1.0, -1.0, -2.0, -1.0, 1.0])) == 3.5
        # A secondary corner eddy puts an earlier crossing; the last one counts.
        assert backstep.reattachment(x, np.array([-1.0, 1.0, -1.0, -3.0, 1.0])) == 3.75
        # A reading exactly zero counts as attached.
        assert backstep.reattachment(x, np.array([-1.0, -1.0, 0.0, 1.0, 1.0])) == 2.0

    def test_reattachment_refuses_an_attached_flow_or_a_mismatch(self) -> None:
        x = np.array([0.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="never turns"):
            backstep.reattachment(x, np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="never turns"):
            backstep.reattachment(x, np.array([1.0, -1.0, -1.0]))
        with pytest.raises(ValueError, match="against"):
            backstep.reattachment(x, np.array([1.0, -1.0]))

    def test_the_shear_is_the_wall_functions_on_the_two_wall_rows(
        self, medium: tuple
    ) -> None:
        """Upstream the row above the block, downstream the floor row; the corner face left out."""
        config, mesh = medium
        rng = np.random.default_rng(7)
        u = rng.standard_normal((mesh.yc.size, mesh.x.size))
        k = 0.5 + rng.random((mesh.yc.size, mesh.xc.size))
        line = backstep.bottom_wall_shear(mesh, config, u, k)
        j_up, i_step = (
            backstep.step_row(mesh, config),
            backstep.step_column(mesh, config),
        )
        assert line.x_over_h.size == mesh.x.size - 3
        assert np.all(np.diff(line.x_over_h) > 0)
        assert not np.any(np.isclose(line.x_over_h, 0.0))
        c_mu = VARIANTS["standard"].c_mu
        nu = config.mu / config.rho
        for i in (1, i_step - 1, i_step + 1, mesh.x.size - 2):
            j = j_up if i < i_step else 0
            y_p = mesh.yc[j] - mesh.y[j]
            k_p = 0.5 * (k[j, i - 1] + k[j, i])
            mu_w = wall_viscosity(
                np.array([k_p]), np.array([y_p]), c_mu, config.rho, nu
            )[0]
            n = int(
                np.argmin(np.abs(line.x_over_h - backstep.data_x(config, mesh.x[i])))
            )
            assert line.values[n] == pytest.approx(mu_w * u[j, i] / y_p, rel=1e-12)

    def test_the_coefficients_scale_by_the_dynamic_pressure(self) -> None:
        line = backstep.WallLine(np.array([0.0, 1.0]), np.array([2.0, -4.0]))
        cf = backstep.friction_coefficient(line, 10.0, 2.0)
        assert cf.values == pytest.approx([0.02, -0.04])
        assert backstep.at_stations(cf, (0.5,)) == pytest.approx([-0.01])
        with pytest.raises(ValueError, match="outside"):
            backstep.at_stations(cf, (1.5,))

    def test_the_wall_pressure_follows_the_wall_rows_and_cp_is_zero_at_37_5(
        self, medium: tuple
    ) -> None:
        config, mesh = medium
        j_up, i_step = (
            backstep.step_row(mesh, config),
            backstep.step_column(mesh, config),
        )
        p = np.zeros((mesh.yc.size, mesh.xc.size))
        p[j_up, :] = 3.0
        p[0, :] = 5.0
        p[-1, :] = 7.0
        bottom = backstep.wall_pressure(mesh, config, p)
        assert bottom.values[:i_step] == pytest.approx(3.0)
        assert bottom.values[i_step:] == pytest.approx(5.0)
        assert backstep.wall_pressure(mesh, config, p, "top").values == pytest.approx(
            7.0
        )
        with pytest.raises(ValueError, match="bottom"):
            backstep.wall_pressure(mesh, config, p, "side")
        x = bottom.x_over_h
        sloped = backstep.WallLine(x, 100.0 * x)
        cp = backstep.pressure_coefficient(sloped, 10.0, 2.0)
        assert np.interp(37.5, cp.x_over_h, cp.values) == pytest.approx(0.0, abs=1e-12)
        assert np.interp(27.5, cp.x_over_h, cp.values) == pytest.approx(-1000.0 / 100.0)

    def test_y_star_of_a_uniform_k_on_the_three_walls(self, medium: tuple) -> None:
        config, mesh = medium
        k = np.full((mesh.yc.size, mesh.xc.size), 4.0)
        got = backstep.wall_y_star(mesh, config, k, 6.0)
        c_mu = VARIANTS["standard"].c_mu
        nu = config.mu / config.rho
        j_up = backstep.step_row(mesh, config)
        floor = c_mu**0.25 * 2.0 * (mesh.yc[0] - mesh.y[0]) / nu
        top = c_mu**0.25 * 2.0 * (mesh.yc[j_up] - mesh.y[j_up]) / nu
        ceiling = c_mu**0.25 * 2.0 * (mesh.y[-1] - mesh.yc[-1]) / nu
        assert got["floor_downstream"] == pytest.approx(floor)
        assert got["floor_at_station"] == pytest.approx(floor)
        assert got["step_top"] == pytest.approx(top)
        assert got["step_top_at_station"] == pytest.approx(top)
        assert got["ceiling_upstream"] == pytest.approx(ceiling)
        assert got["ceiling_at_station"] == pytest.approx(ceiling)
        assert got["floor"] == Y_STAR_FLOOR
        assert floor == pytest.approx(ceiling) and top > floor


@pytest.mark.unit
class TestReport:
    def _fields(self, mesh: Mesh, config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A plausible stand-in solution: a layer over every wall, a bubble on the floor."""
        _x, yc = np.meshgrid(mesh.x, mesh.yc)
        j_up, i_step = (
            backstep.step_row(mesh, config),
            backstep.step_column(mesh, config),
        )
        u = power_law_layer(yc, backstep.U_REF, 1.5 * H)
        u[:, :i_step] = power_law_layer(
            np.maximum(yc[:, :i_step] - H, 0.0), backstep.U_REF, 1.5 * H
        )
        u[:j_up, :i_step] = 0.0
        x_h = np.asarray(backstep.data_x(config, mesh.x))
        bubble = (x_h > 0.0) & (x_h < 5.5)
        u[0, bubble] = -1.0
        p = np.zeros((mesh.yc.size, mesh.xc.size))
        k = np.full((mesh.yc.size, mesh.xc.size), 2.0)
        return u, p, k

    def test_the_report_is_whole_and_serialisable(self, medium: tuple) -> None:
        config, mesh = medium
        u, p, k = self._fields(mesh, config)
        report = backstep.evaluate(mesh, config, u, p, k)
        json.dumps(report)
        assert set(report) == {
            "u_ref",
            "inflow",
            "x_r_over_h",
            "x_r_measured",
            "x_r_uncertainty",
            "profiles",
            "cf",
            "cp",
            "y_star",
        }
        assert report["u_ref"]["computed"] == pytest.approx(backstep.U_REF)
        assert set(report["profiles"]) == {"-4.0", "1.0", "4.0", "6.0", "10.0"}
        assert report["profiles"]["1.0"]["stations"] == 28
        assert report["cf"]["stations"] == 20 and report["cp"]["stations"] == 33
        # The reversed floor row ends between the faces at 5.5 H and 5.75 H.
        assert 5.5 < report["x_r_over_h"] < 5.75
        assert report["y_star"]["floor"] == Y_STAR_FLOOR

    def test_every_number_of_the_report_is_the_constructed_fields(
        self, medium: tuple
    ) -> None:
        """Each value of ``evaluate`` against a second computation on one known field.

        The field (tests/backstep_reference.py) has a centreline speed 10%
        off U_ref, a shear that changes sign inside a face pair, a pressure
        and a k that differ from row to row and along the wall, and a
        stress column and an error column that differ from the data they sit
        beside. Defects caught: a profile compared with the stresses, Cf
        with its error column, Cf or the profiles normalised by the
        constant U_ref, Cp by 2 rho, y* read over the floor from x/H = 0
        and not past reattachment.
        """
        config, mesh = medium
        u, p, k = reference.fields(mesh)
        want = reference.expected(mesh, config, u, p, k)
        got = backstep.evaluate(mesh, config, u, p, k)
        reference.assert_matches(got, want)
        # The field is built so that each wrong reading differs from the
        # right one by far more than the 1e-10 the comparison allows.
        assert abs(got["u_ref"]["computed"] / backstep.U_REF - 1.0) > 0.05
        assert 5.0 < got["x_r_over_h"] < 5.25
        x_cell = (mesh.xc - reference.STEP_FACE_H * H) / H
        k_floor = k[0, :]
        past = np.median(k_floor[x_cell > got["x_r_over_h"]])
        from_corner = np.median(k_floor[x_cell > 0.0])
        assert abs(past / from_corner - 1.0) > 0.05

    def test_the_constructed_layer_has_its_analytic_thicknesses(
        self, medium: tuple
    ) -> None:
        """delta_99 = 0.4 ln(100) H and theta = 0.2 H for u = U (1 - exp(-d / 0.4 H))."""
        config, mesh = medium
        u, p, k = reference.fields(mesh)
        inflow = backstep.evaluate(mesh, config, u, p, k)["inflow"]
        assert inflow["delta_99_over_h"] == pytest.approx(
            reference.LAYER * np.log(100.0), rel=0.03
        )
        nu = config.mu / config.rho
        theta = 0.5 * reference.LAYER * H
        assert inflow["re_theta"] == pytest.approx(
            reference.CENTRE * theta / nu, rel=0.05
        )

    def test_an_attached_floor_reports_no_reattachment(self, medium: tuple) -> None:
        config, mesh = medium
        u, p, k = self._fields(mesh, config)
        u[0, :] = np.abs(u[0, :])
        report = backstep.evaluate(mesh, config, u, p, k)
        assert report["x_r_over_h"] is None
        json.dumps(report)
