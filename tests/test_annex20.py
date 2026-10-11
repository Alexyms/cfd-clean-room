"""VAL-017, the Annex 20 room (ECR-002 step 7; ADR-012 G (iii)).

The case is configs/validation_annex20.yaml, loaded by validation.cases.
load_annex20 with the inlet velocity scaled so the faces the slot covers
carry u0 h on the grid at hand, and compared with Nielsen's (1990)
measurements by validation.annex20, which stores them with their provenance
and Rong and Nielsen's (2008) published standard k-epsilon prediction beside
them. No threshold is set: ADR-012 decision 6 leaves the allowance OPEN until
Alex sets it from the first coupled results, which are prompt 49's
(docs/reports/ecr002_val017_annex20.md).

The validation tests run the case once, on 216x72 under the default scheme,
as the suite loads it (about 18 minutes alone on this machine; the suite's
long case, as the 80x80 cavity is judged from a harness row). The solve
reaches its 10,000-iteration cap with the residual still falling: conditions
(a) to (d) of the stopping rule hold at the cap, and condition (e) cannot,
because its scale is the inlet's eddy viscosity (2.2 nu here) while the room
makes up to 900 nu, so the eddy-viscosity step's floor sits above what (e)
asks (the report, section 5.2). The first test pins the record of that run
(the readings and the four lines, to the recorded rounding) and prints the
comparison with the data without scoring it; the second asserts the stop by
the rule and is expected to fail, strictly, until Alex decides on (e)'s
scale, so the decision flips it rather than a quiet pass.

The unit tests pin the loader (the scaled flux on both presets, the outlet
taking the whole inflow, the file's settings), the stored data (counts,
order, and the net flux the two vertical profiles carry, which reproduces
the check of docs/reports/product_case_reynolds.md, section 6.3) and the
line sampler (exact on a linear field, a face's own value on a face).
"""

import numpy as np
import pytest

from src.boundary_staggered import StaggeredBoundary
from src.mesh import Mesh
from src.solver_staggered import StaggeredSolver
from validation import annex20
from validation.cases import (
    ANNEX20_GRIDS,
    CASE_GRIDS,
    load_annex20,
    load_case,
    load_preset,
)

U0_H = annex20.U0 * annex20.SLOT
NU_SPEC = 15.3e-6
RECORD_PRESET = "val017_216x72"
# The record of the committed case on 216x72 under the default scheme (prompt
# 49, results/builder49/standard_216x72_upwind, commit 6be9c2f): the stop,
# the residual and the worst per-cell imbalance at the cap, and u / u0 at the
# measurement stations of the four lines, rounded to five decimals.
RECORD_STOP = "max_simple_iter"
RECORD_OUTER = 10000
RECORD_RESIDUAL_BOUND = 1.0e-8  # the record's 6.9e-10, with room for another machine
RECORD_ROUNDING = 2.0e-5  # five decimals, and the iteration error at the cap (1e-6 u0)
RECORD_LINES: dict[str, tuple[float, ...]] = {
    "x3": (
        0.81869,
        0.72997,
        0.63237,
        0.56600,
        0.50661,
        0.41295,
        0.27224,
        0.16616,
        0.11402,
        0.05887,
        0.03237,
        0.01133,
        -0.01777,
        -0.04530,
        -0.07193,
        -0.09764,
        -0.12458,
        -0.13839,
        -0.14770,
        -0.15488,
        -0.16114,
        -0.16600,
        -0.16904,
        -0.16927,
        -0.15160,
    ),
    "x6": (
        0.58298,
        0.62671,
        0.61521,
        0.59073,
        0.56551,
        0.50730,
        0.45451,
        0.37402,
        0.29706,
        0.20497,
        0.13233,
        0.07701,
        0.01916,
        -0.03464,
        -0.08849,
        -0.13970,
        -0.19494,
        -0.22903,
        -0.25513,
        -0.27627,
        -0.29299,
        -0.30861,
        -0.32173,
        -0.33017,
        -0.33548,
    ),
    "y2916": (
        1.00781,
        1.01921,
        1.00547,
        0.95640,
        0.89901,
        0.84603,
        0.80017,
        0.76281,
        0.72887,
        0.69642,
        0.66788,
        0.64654,
        0.62183,
        0.59002,
        0.56273,
        0.52763,
        0.48048,
        0.34576,
        0.24082,
        0.18316,
        0.11168,
        0.05880,
        0.02398,
        0.00076,
        -0.00408,
        -0.02135,
        -0.03665,
        -0.04137,
    ),
    "y0084": (
        0.00039,
        -0.00605,
        -0.01405,
        -0.02171,
        -0.03079,
        -0.04099,
        -0.05246,
        -0.06823,
        -0.07671,
        -0.08677,
        -0.09736,
        -0.10955,
        -0.12820,
        -0.15038,
        -0.17117,
        -0.20319,
        -0.23212,
        -0.26216,
        -0.28921,
        -0.31762,
        -0.33658,
        -0.34525,
        -0.33687,
        -0.29931,
        -0.19608,
        -0.06849,
        0.13874,
        0.31025,
    ),
}


def _layer(config) -> tuple[Mesh, StaggeredBoundary]:
    mesh = Mesh(config)
    return mesh, StaggeredBoundary(mesh, config)


@pytest.mark.unit
class TestLoader:
    @pytest.mark.parametrize(
        ("preset", "faces"),
        [("val017_216x72", 4), ("val017_375x125", 7)],
    )
    def test_the_slot_carries_u0_h_on_every_preset(
        self, preset: str, faces: int
    ) -> None:
        """The scaled inlet carries u0 h through the faces the slot covers."""
        config = load_preset(preset)
        _mesh, boundary = _layer(config)
        dy = config.room_height / config.ny
        velocity = config.boundaries["inlet"].velocity
        assert velocity == pytest.approx(annex20.U0 * annex20.SLOT / (faces * dy))
        assert boundary.get_inlet_flux("inlet") == pytest.approx(U0_H, rel=1e-12)
        assert boundary.get_inlet_flux("inlet") / velocity / dy == pytest.approx(faces)

    def test_375x125_needs_no_scaling(self) -> None:
        """On the finer preset the slot falls on whole faces and u0 stands."""
        config = load_preset("val017_375x125")
        assert config.boundaries["inlet"].velocity == pytest.approx(annex20.U0)

    def test_the_unscaled_file_misses_u0_h_on_216x72(self) -> None:
        """Without the loader the ADR's grid carries 4 dy, not h, at u0."""
        config = load_case("annex20")
        assert (config.nx, config.ny) == (216, 72)
        _, boundary = _layer(config)
        assert boundary.get_inlet_flux("inlet") == pytest.approx(
            annex20.U0 * 4 * config.room_height / config.ny
        )
        assert boundary.get_inlet_flux("inlet") != pytest.approx(U0_H, rel=1e-6)

    @pytest.mark.parametrize("preset", sorted(ANNEX20_GRIDS))
    def test_the_outlet_takes_the_whole_inflow(self, preset: str) -> None:
        """The outlet states no velocity, so its faces carry u0 h out."""
        config = load_preset(preset)
        mesh, boundary = _layer(config)
        speed = boundary.fixed_flow_velocities()["outlet"]
        dy = config.room_height / config.ny
        covered = sum(
            1 for y in mesh.yc if 0.0 <= y <= config.boundaries["outlet"].y_end
        )
        assert speed * covered * dy == pytest.approx(U0_H, rel=1e-12)

    def test_the_presets_are_registered(self) -> None:
        assert {"val017_216x72", "val017_375x125"} == ANNEX20_GRIDS
        for preset in ANNEX20_GRIDS:
            kind, nx, ny = CASE_GRIDS[preset]
            assert kind == "annex20"
            assert preset.endswith(f"{nx}x{ny}")

    def test_the_file_holds_the_case_as_the_design_states_it(self) -> None:
        """Standard variant, the default scheme, the product's solver keys."""
        config = load_case("annex20")
        assert config.turbulence is not None
        assert config.turbulence.variant == "standard"
        assert config.turbulence.advection_scheme == "upwind"
        assert config.turbulence.cfl_number == 0.25
        assert config.stopping_rule == "error_estimate"
        assert config.momentum_sweeps == 10
        assert config.pressure_rtol == 1.0e-4
        assert config.max_simple_iter == 10000
        assert (config.room_width, config.room_height) == (9.0, 3.0)
        assert config.obstacles == []

    def test_the_inlet_is_the_specifications(self) -> None:
        """Re 5,000 on h, 4% intensity, l_e = h / 10 in the design's convention."""
        config = load_case("annex20")
        inlet = config.boundaries["inlet"]
        assert inlet.y_end - inlet.y_start == pytest.approx(annex20.SLOT)
        assert inlet.y_end == config.room_height
        assert inlet.turbulence_intensity == 0.04
        assert inlet.dissipation_length == pytest.approx(annex20.SLOT / 10)
        nu = config.mu / config.rho
        assert nu == pytest.approx(NU_SPEC)
        assert annex20.U0 * annex20.SLOT / nu == pytest.approx(5000.0, rel=1e-3)

    def test_a_grid_the_slot_misses_is_refused(self) -> None:
        """Half-metre cells put no centre inside the slot (the top one is at 2.75 m).

        The boundary layer refuses the outlet, which has no inflow to share,
        before the loader scales anything.
        """
        with pytest.raises(ValueError, match=r"inflow 0 m\^2/s"):
            load_annex20("annex20", grid=(9, 6))


@pytest.mark.unit
class TestData:
    def test_the_lines_hold_the_workbooks_stations_in_order(self) -> None:
        counts = {"x3": 25, "x6": 25, "y2916": 28, "y0084": 28}
        for key, line in annex20.LINES.items():
            assert len(line.stations) == counts[key]
            assert len(line.measured) == len(line.published) == counts[key]
            steps = np.diff(line.stations)
            # The workbook reads a vertical line down from the ceiling and
            # a horizontal one from the inlet wall.
            assert np.all(steps < 0) if line.vertical else np.all(steps > 0)
            assert min(line.stations) > 0.0
            assert max(line.stations) < (
                annex20.HEIGHT if line.vertical else annex20.LENGTH
            )

    def test_the_lines_sit_where_the_specification_puts_them(self) -> None:
        assert annex20.LINES["x3"].position == annex20.HEIGHT
        assert annex20.LINES["x6"].position == 2 * annex20.HEIGHT
        assert annex20.LINES["y2916"].position == pytest.approx(
            annex20.HEIGHT - annex20.SLOT / 2
        )
        assert annex20.LINES["y0084"].position == pytest.approx(annex20.SLOT / 2)

    def test_the_vertical_profiles_carry_the_reports_flux_ratios(self) -> None:
        """Section 6.3 of product_case_reynolds.md: 1.17 at x = H, 0.62 at 2H."""
        assert round(annex20.vertical_line_flux(annex20.LINES["x3"]), 2) == 1.17
        assert round(annex20.vertical_line_flux(annex20.LINES["x6"]), 2) == 0.62
        with pytest.raises(ValueError, match="not a vertical line"):
            annex20.vertical_line_flux(annex20.LINES["y2916"])

    def test_the_jet_and_the_return_have_their_signs(self) -> None:
        jet, floor = annex20.LINES["y2916"], annex20.LINES["y0084"]
        assert jet.measured[0] > 1.0
        assert all(u > 0.0 for u in jet.measured[:19])
        assert min(floor.measured) < -0.3
        assert floor.measured[-1] > 0.2

    def test_the_published_line_misses_three_stations(self) -> None:
        """The traced line does not reach the wall stations it starts beside."""
        missing = {
            key: [i for i, p in enumerate(line.published) if p is None]
            for key, line in annex20.LINES.items()
        }
        assert missing == {"x3": [], "x6": [0], "y2916": [0], "y0084": [0]}

    def test_the_stored_values_are_the_ones_checked_against_the_workbook(
        self,
    ) -> None:
        """Pins every stored station, measurement and published value.

        Test 49 (2026-10-11) matched all 106 stations and measurements to the
        workbook, which is not committed, so the values are pinned by a hash
        of their repr instead: a changed value fails here.
        """
        import hashlib

        payload = repr(
            [
                (key, line.stations, line.measured, line.published)
                for key, line in sorted(annex20.LINES.items())
            ]
        )
        assert hashlib.sha256(payload.encode()).hexdigest() == (
            "3e53d338c7414577e38dadd6f0c69bbc3ce7f272ec225aac3e078212f5cf4c00"
        )


@pytest.mark.unit
class TestSampling:
    def test_a_linear_field_is_sampled_exactly(self) -> None:
        mesh = Mesh(load_preset("val017_216x72"))
        x_faces, y_centres = np.meshgrid(mesh.x, mesh.yc)
        u = 0.3 * x_faces + 0.7 * y_centres + 0.1
        sampled = annex20.sample_lines(mesh, u)
        for key, line in annex20.LINES.items():
            stations = np.array(line.stations)
            x = np.full_like(stations, line.position) if line.vertical else stations
            y = stations if line.vertical else np.full_like(stations, line.position)
            exact = (0.3 * x + 0.7 * y + 0.1) / annex20.U0
            assert np.abs(sampled[key] - exact).max() < 1e-12

    def test_a_station_on_a_face_reads_that_face(self) -> None:
        """x = 3.0 m is face 72 of 216 over 9 m; the row's own value comes back."""
        mesh = Mesh(load_preset("val017_216x72"))
        rng = np.random.default_rng(49)
        u = rng.standard_normal((mesh.yc.size, mesh.x.size))
        j = 40
        got = annex20.sample_u(mesh, u, np.array([3.0]), np.array([mesh.yc[j]]))
        assert got[0] == pytest.approx(u[j, 72], rel=1e-12)

    def test_the_walls_read_zero(self) -> None:
        mesh = Mesh(load_preset("val017_216x72"))
        u = np.ones((mesh.yc.size, mesh.x.size))
        got = annex20.sample_u(mesh, u, np.array([4.5, 4.5]), np.array([0.0, 3.0]))
        assert got == pytest.approx([0.0, 0.0])

    def test_line_errors_skip_the_stations_the_reference_lacks(self) -> None:
        errors = annex20.line_errors(np.array([1.0, 2.0, 4.0]), (None, 1.0, 1.0))
        assert errors.stations == 2
        assert errors.max_abs == 3.0
        assert errors.rms == pytest.approx(np.sqrt((1.0 + 9.0) / 2))

    def test_line_errors_refuse_a_mismatch_or_an_empty_reference(self) -> None:
        with pytest.raises(ValueError, match="reference stations"):
            annex20.line_errors(np.zeros(3), (0.0, 0.0))
        with pytest.raises(ValueError, match="holds no station"):
            annex20.line_errors(np.zeros(2), (None, None))


@pytest.fixture(scope="module")
def annex20_216x72() -> dict:
    """The committed case on 216x72 under the default scheme, run to its stop or cap."""
    config = load_preset(RECORD_PRESET)
    mesh = Mesh(config)
    solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))
    solver.solve_steady()
    faces = solver.face_velocities
    assert faces is not None
    return {
        "config": config,
        "solver": solver,
        "lines": annex20.sample_lines(mesh, np.array(faces.u)),
    }


@pytest.mark.validation
def test_annex20_216x72_reproduces_its_record_val017(annex20_216x72: dict) -> None:
    """VAL-017: the run reaches its record's stop and reproduces its four lines.

    No threshold against the data is applied (ADR-012 decision 6, OPEN); the
    comparison is printed. The solve raises on a negative k or eps, so
    positivity is asserted by running at all; no pressure correction and no
    k or eps solve reaches its cap.
    """
    solver, config = annex20_216x72["solver"], annex20_216x72["config"]
    assert solver.stop_reason == RECORD_STOP
    assert len(solver.residual_history) == RECORD_OUTER
    assert solver.residual_history[-1] < RECORD_RESIDUAL_BOUND
    assert solver.pressure_cap_hits == 0
    assert solver.rule_version == 5
    assert float(np.abs(solver.last_mass_imbalance).max()) < config.mass_imbalance_tol
    state = solver.turbulence_state
    assert state is not None
    assert float(state.k[state.k > 0.0].min()) > 0.0
    for key, line in annex20.LINES.items():
        values = annex20_216x72["lines"][key]
        recorded = np.array(RECORD_LINES[key])
        against_data = annex20.line_errors(values, line.measured)
        against_record = np.abs(values - recorded).max()
        print(
            f"VAL-017 {line.label}: against the data max |d| {against_data.max_abs:.4f}, "
            f"rms {against_data.rms:.4f} (unscored); against the record "
            f"{against_record:.2e}"
        )
        assert against_record < RECORD_ROUNDING, key


@pytest.mark.validation
@pytest.mark.xfail(
    strict=True,
    reason="condition (e) cannot hold on this case with its inlet scale "
    "(docs/reports/ecr002_val017_annex20.md, section 5.2); for Alex",
)
def test_annex20_216x72_stops_by_the_rule_val017(annex20_216x72: dict) -> None:
    """VAL-017: the coupled solve stops by its rule, condition (e) included."""
    solver = annex20_216x72["solver"]
    assert solver.converged is True
    assert solver.stop_reason == "error_estimate_and_continuity"
