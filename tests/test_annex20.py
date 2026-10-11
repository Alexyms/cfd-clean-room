"""VAL-017, the Annex 20 room (ECR-002 step 7; ADR-012 G (iii)): the unit tests.

The case is configs/validation_annex20.yaml, loaded by validation.cases.
load_annex20 with the inlet velocity scaled so the faces the slot covers
carry u0 h on the grid at hand, and compared with Nielsen's (1990)
measurements by validation.annex20, which stores them with their provenance
and Rong and Nielsen's (2008) published standard k-epsilon prediction beside
them. The allowance (Alex, 2026-10-11; ADR-012 decision 6, amended): on x = H
and on the ceiling jet, the RMS of u / u0 against the measurements is no
larger than the published prediction's RMS on that line.

The run itself is not here. It is the harness row ``val017_216x72``
(``python scripts/benchmark.py --cases val017_216x72 --repeats 1``), scored by
the allowance and recorded in benchmarks/results.jsonl, as the 80x80 cavity
is: it costs about twenty minutes alone and ran twice per CI run while it
was a suite test (prompt 50). Its record and the runs behind the case
file's settings are docs/reports/ecr002_rule_e_scale.md; the first coupled
results, with the comparison tables and figures, are
docs/reports/ecr002_val017_annex20.md.

These tests pin the loader (the scaled flux on both presets, the outlet
taking the whole inflow, the file's settings), the stored data (counts,
order, the net flux the two vertical profiles carry, which reproduces the
check of docs/reports/product_case_reynolds.md, section 6.3, and a hash of
every stored value), the line sampler (exact on a linear field, a face's own
value on a face) and the allowance (its two values from the stored data,
the stations it is taken over, the boundary, the unscored lines).
"""

import numpy as np
import pytest

from src.boundary_staggered import StaggeredBoundary
from src.mesh import Mesh
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


@pytest.mark.unit
class TestAllowance:
    def _values(
        self, offset: dict[str, float | np.ndarray] | None = None
    ) -> dict[str, np.ndarray]:
        """The measurements themselves at every station, plus any offset per line."""
        offset = offset or {}
        return {
            key: np.array(line.measured) + offset.get(key, 0.0)
            for key, line in annex20.LINES.items()
        }

    def test_the_allowance_is_the_published_rms_the_decision_cites(self) -> None:
        """0.070 at x = H over 25 stations, 0.136 on the jet over the 27 it reaches."""
        assert annex20.SCORED_LINES == ("x3", "y2916")
        assert round(annex20.published_rms("x3"), 3) == 0.070
        assert round(annex20.published_rms("y2916"), 3) == 0.136
        x3, jet = annex20.LINES["x3"], annex20.LINES["y2916"]
        assert annex20.line_errors(np.array(x3.measured), x3.published).stations == 25
        assert annex20.line_errors(np.array(jet.measured), jet.published).stations == 27

    def test_the_measurements_pass_with_no_error(self) -> None:
        scores = annex20.score_lines(self._values())
        assert list(scores) == list(annex20.LINES)
        for key, score in scores.items():
            assert score.rms == 0.0
            assert score.allowance == annex20.published_rms(key)
            assert score.passed is (True if key in annex20.SCORED_LINES else None)

    @pytest.mark.parametrize("key", ["x3", "y2916"])
    def test_a_line_passes_at_its_allowance_and_fails_above_it(self, key: str) -> None:
        """A uniform offset is its own RMS. Defect caught: the comparison reversed."""
        allowance = annex20.published_rms(key)
        below = annex20.score_lines(self._values({key: allowance * (1.0 - 1e-9)}))
        above = annex20.score_lines(self._values({key: -allowance * (1.0 + 1e-9)}))
        assert below[key].passed is True
        assert above[key].passed is False
        assert above[key].rms == pytest.approx(allowance, rel=1e-8)

    def test_the_rms_is_over_every_station_of_the_line(self) -> None:
        """The jet's first station, which the published line misses, still counts.

        Defect caught: the RMS taken over the published line's stations only,
        under which this line would score 0 and pass.
        """
        jet = annex20.LINES["y2916"]
        offset = np.zeros(len(jet.stations))
        offset[0] = 1.0
        scores = annex20.score_lines(self._values({"y2916": offset}))
        assert scores["y2916"].stations == 28
        assert scores["y2916"].rms == pytest.approx(np.sqrt(1.0 / 28.0))
        assert scores["y2916"].passed is False

    def test_the_unscored_lines_never_fail(self) -> None:
        """x = 2H and the floor are reported, with no pass or fail."""
        scores = annex20.score_lines(self._values({"x6": 1.0, "y0084": 1.0}))
        for key in ("x6", "y0084"):
            assert scores[key].rms == pytest.approx(1.0)
            assert scores[key].scored is False
            assert scores[key].passed is None
        assert scores["x3"].passed and scores["y2916"].passed

    def test_a_missing_short_or_non_finite_line_is_refused(self) -> None:
        values = self._values()
        del values["x6"]
        with pytest.raises(ValueError, match="no values for the lines"):
            annex20.score_lines(values)
        values = self._values()
        values["x3"] = values["x3"][:-1]
        with pytest.raises(ValueError, match=r"x3: .* against 25 stations"):
            annex20.score_lines(values)
        values = self._values()
        values["y2916"][3] = np.nan
        with pytest.raises(ValueError, match="y2916: a sampled value is not finite"):
            annex20.score_lines(values)
