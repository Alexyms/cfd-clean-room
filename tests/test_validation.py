"""Tests for the validation package: case loading and error metrics."""

from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from src.boundary_concentration import SURFACE_FLOOR
from src.config import SimConfig
from src.mesh import FLUID, SOLID, Mesh
from validation.cases import (
    CASE_FILES,
    CASE_GRIDS,
    WALL_CLUSTERED_GRIDS,
    case_path,
    load_case,
    load_preset,
    load_wall_clustered,
    with_velocity_step,
)
from validation.metrics import (
    GHIA_U_VAL,
    GHIA_U_Y,
    GHIA_V_VAL,
    GHIA_V_X,
    MARCHI_M,
    MARCHI_REFERENCE,
    MARCHI_U_ROWS,
    MARCHI_V_ROWS,
    _bracket,
    cavity_centerline_errors,
    cavity_centerline_profiles,
    cavity_marchi_centerline_errors,
    cavity_true_centerline_errors,
    cavity_true_centerline_profiles,
    centroid,
    field_minimum,
    inlet_velocity,
    lagrange,
    lid_velocity,
    peak_retention,
    poiseuille_l2_error,
    poiseuille_profiles,
    relative_l2,
)
from validation.transport_cases import (
    FixedConditions,
    ScalarPhysics,
    conditions_with,
    diffusion_case,
    gaussian_cell_averages,
    gaussian_mass,
    oblique_pulse_case,
    rotating_puff_case,
    sealed_box_case,
    smith_hutton_case,
    transport_config,
    zero_conditions,
)


@pytest.mark.unit
class TestLoadCase:
    def test_every_case_file_exists_and_loads(self) -> None:
        for name in CASE_FILES:
            assert case_path(name).exists()
            config = load_case(name)
            assert config.nx > 0 and config.ny > 0

    def test_poiseuille_settings_are_the_validation_settings(self) -> None:
        config = load_case("poiseuille")
        assert (config.nx, config.ny) == (80, 40)
        assert config.max_pressure_iter == 5000
        assert config.pressure_rtol == 1.0e-8
        assert config.alpha_velocity == 0.7

    def test_cavity_settings_are_the_validation_settings(self) -> None:
        config = load_case("cavity")
        assert (config.nx, config.ny) == (40, 40)
        assert config.max_pressure_iter == 5000
        assert config.pressure_rtol == 1.0e-8
        assert config.alpha_velocity == 0.5

    def test_grid_override_changes_only_the_grid(self) -> None:
        base = load_case("cavity")
        overridden = load_case("cavity", grid=(20, 24))
        assert (overridden.nx, overridden.ny) == (20, 24)
        assert overridden.alpha_velocity == base.alpha_velocity
        assert overridden.max_pressure_iter == base.max_pressure_iter
        assert overridden.room_width == base.room_width

    def test_unknown_case_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown validation case"):
            load_case("channel")

    def test_grid_presets_name_known_cases(self) -> None:
        for case_id, (kind, nx, ny) in CASE_GRIDS.items():
            assert kind in CASE_FILES, case_id
            assert nx > 0 and ny > 0


def _without(config: SimConfig, name: str) -> dict:
    """Every attribute of a configuration but one."""
    return {k: v for k, v in vars(config).items() if k != name}


@pytest.mark.unit
class TestStepSevenCases:
    """ECR-001 step 7: VAL-001 names its rule, and criterion 2 has its own preset."""

    def test_poiseuille_case_carries_the_error_estimate_rule(self) -> None:
        """Decision 1: the case file names the rule, its tolerances at the defaults."""
        config = load_case("poiseuille")
        assert config.stopping_rule == "error_estimate"
        assert (config.iteration_error_tol, config.mass_imbalance_tol) == (1e-6, 1e-10)

    def test_with_velocity_step_changes_the_rule_and_nothing_else(self) -> None:
        """The collocated override returns velocity_step and leaves its input alone."""
        config = load_case("poiseuille")
        before = dict(vars(config))
        out = with_velocity_step(config)
        assert out.stopping_rule == "velocity_step"
        assert _without(out, "stopping_rule") == _without(config, "stopping_rule")
        assert vars(config) == before

    def test_stretched_preset_fixes_the_wall_spacing_and_derives_the_ratio(
        self,
    ) -> None:
        """Criterion 2 at 80x40: y wall cell 0.1 H / ny = 0.00125, ratio about 1.20."""
        kind, nx, ny = CASE_GRIDS["val001_80x40_stretched"]
        assert "val001_80x40_stretched" in WALL_CLUSTERED_GRIDS
        config = load_wall_clustered(kind, grid=(nx, ny))
        mesh = Mesh(config)
        for wall_cell in (mesh.dy_cell[0], mesh.dy_cell[-1]):
            assert wall_cell == pytest.approx(0.00125, rel=1e-12)
        assert mesh.stretch_ratio_y == pytest.approx(1.2057, abs=1e-4)
        assert mesh.stretch_ratio_x == 1.0
        base = load_case(kind, grid=(nx, ny))
        assert _without(config, "stretch_y") == _without(base, "stretch_y")

    def test_wall_spacing_follows_height_and_ny_not_width_and_nx(self) -> None:
        """H / ny equals L / nx at 80x40, so a grid with unequal sides tells them apart."""
        mesh = Mesh(load_wall_clustered("poiseuille", grid=(80, 20)))
        assert mesh.dy_cell[0] == pytest.approx(0.1 * 0.5 / 20, rel=1e-12)


@pytest.mark.unit
class TestStepEightCases:
    """ECR-001 step 8: the cavity names its rule, and a preset loads its own mesh."""

    def test_cavity_case_carries_the_error_estimate_rule(self) -> None:
        """Decision 1: the rule, its default tolerances, and a cap above 80x80's 12849."""
        config = load_case("cavity")
        assert config.stopping_rule == "error_estimate"
        assert (config.iteration_error_tol, config.mass_imbalance_tol) == (1e-6, 1e-10)
        assert config.max_simple_iter == 20000

    def test_load_preset_builds_each_preset_on_its_own_mesh(self) -> None:
        """Review 25 S5: the stretched preset and its twin share a tuple, not a mesh."""
        assert CASE_GRIDS["val001_80x40_stretched"] == CASE_GRIDS["val001_80x40"]
        stretched = Mesh(load_preset("val001_80x40_stretched"))
        assert stretched.dy_cell[0] == pytest.approx(0.00125, rel=1e-12)
        assert Mesh(load_preset("val001_80x40")).is_uniform
        for case_id, (_kind, nx, ny) in CASE_GRIDS.items():
            config = load_preset(case_id)
            assert (config.nx, config.ny) == (nx, ny)

    def test_load_preset_refuses_an_unknown_id(self) -> None:
        """An id outside CASE_GRIDS raises KeyError naming the known ones."""
        with pytest.raises(KeyError, match="unknown grid preset"):
            load_preset("val002_60x60")


@pytest.mark.unit
class TestPoiseuilleMetric:
    def test_exact_parabola_scores_zero(self) -> None:
        """The control: the analytical profile itself must have no error."""
        config = load_case("poiseuille", grid=(16, 12))
        mesh = Mesh(config)
        u = np.zeros((config.ny, config.nx))
        y, _u_num, u_ref = poiseuille_profiles(config, mesh, u)
        i_mid = config.nx // 2
        column = np.zeros(config.ny)
        column[np.isin(np.asarray(mesh.yc), y)] = u_ref
        u[:, i_mid] = column
        metric = poiseuille_l2_error(config, mesh, u)
        assert metric.value < 1e-12
        assert metric.metric == "l2_relative_error_u_midchannel"
        assert metric.reference == "analytical_poiseuille"
        assert metric.as_dict() == {
            "metric": metric.metric,
            "value": metric.value,
            "reference": metric.reference,
        }

    def test_a_flat_profile_scores_nonzero(self) -> None:
        """The other direction: a wrong profile must not score zero."""
        config = load_case("poiseuille", grid=(16, 12))
        mesh = Mesh(config)
        u = np.full((config.ny, config.nx), 0.1)
        assert poiseuille_l2_error(config, mesh, u).value > 0.1


@pytest.mark.unit
class TestLidVelocity:
    """The public readers the metrics and the scripts share."""

    def test_inlet_velocity_reads_the_channel_inlet(self) -> None:
        config = load_case("poiseuille", grid=(8, 8))
        assert inlet_velocity(config) == config.boundaries["inlet"].velocity

    def test_inlet_velocity_refuses_a_case_without_one_inlet_velocity(self) -> None:
        """The cavity's lid sets u_velocity and no magnitude."""
        with pytest.raises(ValueError, match="exactly one velocity inlet"):
            inlet_velocity(load_case("cavity", grid=(8, 8)))

    def test_reads_the_cavity_lids_speed(self) -> None:
        assert lid_velocity(load_case("cavity", grid=(8, 8))) == 1.0

    def test_refuses_a_case_without_one_lid_with_a_u_velocity(self) -> None:
        """The Poiseuille inlet sets a magnitude, not a tangential lid speed."""
        with pytest.raises(ValueError, match="exactly one lid"):
            lid_velocity(load_case("poiseuille", grid=(8, 8)))


@pytest.mark.unit
class TestCavityMetric:
    def _ghia_like_fields(self, config, mesh) -> tuple[np.ndarray, np.ndarray]:
        """Fields that follow the Ghia profiles along both centerlines."""
        u = np.zeros((config.ny, config.nx))
        v = np.zeros((config.ny, config.nx))
        yc = np.asarray(mesh.yc)
        xc = np.asarray(mesh.xc)
        u[:, config.nx // 2] = np.interp(yc, GHIA_U_Y[::-1], GHIA_U_VAL[::-1])
        v[config.ny // 2, :] = np.interp(xc, GHIA_V_X[::-1], GHIA_V_VAL[::-1])
        return u, v

    def test_value_is_the_larger_component_and_names_its_reference(self) -> None:
        config = load_case("cavity", grid=(16, 16))
        mesh = Mesh(config)
        u, v = self._ghia_like_fields(config, mesh)
        metric = cavity_centerline_errors(config, mesh, u, v)
        assert set(metric.components) == {"u", "v"}
        assert metric.value == max(metric.components.values())
        assert metric.reference == "ghia_1982_re100_r2"
        assert metric.as_dict()["components"] == metric.components

    def test_ghia_like_fields_score_far_lower_than_zero_fields(self) -> None:
        """Both directions in one place: the metric can be small and can be large."""
        config = load_case("cavity", grid=(32, 32))
        mesh = Mesh(config)
        u, v = self._ghia_like_fields(config, mesh)
        close = cavity_centerline_errors(config, mesh, u, v)
        zero = cavity_centerline_errors(
            config, mesh, np.zeros_like(u), np.zeros_like(v)
        )
        assert close.value < 0.05
        assert zero.value > 0.3


def _cavity(n: int, stretch: float = 1.0) -> tuple[SimConfig, Mesh]:
    """The cavity case on an n x n grid, with the same stretch ratio on both axes."""
    raw = yaml.safe_load(case_path("cavity").read_text(encoding="utf-8"))
    raw["domain"]["nx"] = raw["domain"]["ny"] = n
    raw["mesh"] = {"x": {"stretch_ratio": stretch}, "y": {"stretch_ratio": stretch}}
    config = SimConfig.from_dict(raw)
    return config, Mesh(config)


# u = A + B x and v = A + B y read exactly A + B / 2 on the centerlines.
A, B = 0.3, 0.8


def _linear_fields(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centered u = A + B x and v = A + B y, [ny, nx]."""
    x, y = np.meshgrid(np.asarray(mesh.xc), np.asarray(mesh.yc))
    return A + B * x, A + B * y


@pytest.mark.unit
class TestTrueCenterlineMetric:
    """cavity_true_centerline_errors samples on x = 0.5 and y = 0.5, not half a cell off."""

    def test_linear_field_is_read_on_the_centerline(self) -> None:
        """The new sampling recovers A + B/2 to rounding; the old one misses by B h/2.

        The old metric missing is the control: it shows the test can tell a
        sample on the centerline from one half a cell off it. Only the sampled
        values are compared; the wall values at each end are appended, not read.
        """
        n = 16
        config, mesh = _cavity(n)
        u, v = _linear_fields(mesh)
        _y, u_new, _x, v_new = cavity_true_centerline_profiles(config, mesh, u, v)
        _y, u_old, _x, v_old = cavity_centerline_profiles(config, mesh, u, v)
        for new, old in ((u_new, u_old), (v_new, v_old)):
            assert np.allclose(new[1:-1], A + B / 2, rtol=0.0, atol=1e-14)
            miss = np.asarray(old[1:-1]) - (A + B / 2)
            assert np.allclose(miss, B / (2 * n), rtol=0.0, atol=1e-14)

    def test_odd_grid_reads_the_middle_column_exactly(self) -> None:
        """x = 0.5 is a cell center on an odd grid, so both samplings read that column."""
        n = 21
        config, mesh = _cavity(n)
        u, v = np.random.default_rng(1).standard_normal((2, n, n))
        new = cavity_true_centerline_profiles(config, mesh, u, v)
        fluid = mesh.cell_type[:, n // 2] == FLUID
        assert np.array_equal(new[1][1:-1], u[fluid, n // 2])
        assert np.array_equal(new[3][1:-1], v[n // 2, fluid])
        assert new == cavity_centerline_profiles(config, mesh, u, v)

    @pytest.mark.parametrize("n", [16, 15])
    def test_stretched_mesh_is_read_on_the_centerline(self, n: int) -> None:
        """A linear field on a wall-clustered mesh, even and odd, is read at A + B/2."""
        config, mesh = _cavity(n, stretch=1.1)
        assert not mesh.is_uniform
        u, v = _linear_fields(mesh)
        _y, u_new, _x, v_new = cavity_true_centerline_profiles(config, mesh, u, v)
        assert np.allclose(u_new[1:-1], A + B / 2, rtol=0.0, atol=1e-14)
        assert np.allclose(v_new[1:-1], A + B / 2, rtol=0.0, atol=1e-14)

    def test_bracket_weights_an_unequal_pair_by_distance(self) -> None:
        """An unequal pair is weighted by distance, not by one half.

        The mesh stretches symmetrically, so its middle pair always weighs 1/2
        and cannot show this.
        """
        i, w = _bracket(np.array([0.1, 0.3, 0.45, 0.7, 0.9]), 0.5)
        assert (i, w) == (2, pytest.approx(0.2))

    def test_error_function_reads_the_centerline_not_the_offset_column(self) -> None:
        """A slope across the centerline leaves the new score unchanged.

        u = g(y) + B (x - 0.5) and v = k(x) + B (y - 0.5), with g and k Ghia's
        profiles, are g and k on the true centerlines at any B, so the new
        score must not move when B is added. The old metric reads the offset
        column, B h/2 away, and its score moves: that is the control.
        """
        config, mesh = _cavity(16)
        x, y = np.meshgrid(np.asarray(mesh.xc), np.asarray(mesh.yc))
        g = np.interp(y, GHIA_U_Y[::-1], GHIA_U_VAL[::-1])
        k = np.interp(x, GHIA_V_X[::-1], GHIA_V_VAL[::-1])
        fields = ((g, k), (g + B * (x - 0.5), k + B * (y - 0.5)))
        flat, sloped = (cavity_true_centerline_errors(config, mesh, *f) for f in fields)
        old_flat, old_sloped = (
            cavity_centerline_errors(config, mesh, *f) for f in fields
        )
        for c in ("u", "v"):
            assert sloped.components[c] == pytest.approx(flat.components[c], abs=1e-12)
            assert abs(old_sloped.components[c] - old_flat.components[c]) > 1e-3

    def test_new_metric_has_its_own_name_and_can_be_small_or_large(self) -> None:
        """The old name stays with the old sampling; the new one scores both ways."""
        config, mesh = _cavity(32)
        x, y = np.meshgrid(np.asarray(mesh.xc), np.asarray(mesh.yc))
        # Ghia's profiles, constant across each centerline, so both samplings see them.
        u = np.interp(y, GHIA_U_Y[::-1], GHIA_U_VAL[::-1])
        v = np.interp(x, GHIA_V_X[::-1], GHIA_V_VAL[::-1])
        close = cavity_true_centerline_errors(config, mesh, u, v)
        zero = cavity_true_centerline_errors(config, mesh, 0.0 * u, 0.0 * v)
        assert close.metric == "max_normalized_centerline_error_r2"
        assert cavity_centerline_errors(config, mesh, u, v).metric == (
            "max_normalized_centerline_error"
        )
        assert close.reference == "ghia_1982_re100_r2"
        assert close.value == max(close.components.values())
        assert close.value < 0.05
        assert zero.value > 0.3


# Ghia, Ghia and Shin (1982), Table II, Re = 100, written out independently of
# validation/metrics.py so the stored table is checked against a second copy.
TABLE_II_RE100: tuple[tuple[float, float], ...] = (
    (1.0000, 0.00000),
    (0.9688, -0.05906),
    (0.9609, -0.07391),
    (0.9531, -0.08864),
    (0.9453, -0.10313),
    (0.9063, -0.16914),
    (0.8594, -0.22445),
    (0.8047, -0.24533),
    (0.5000, 0.05454),
    (0.2344, 0.17527),
    (0.2266, 0.17507),
    (0.1563, 0.16077),
    (0.0938, 0.12317),
    (0.0781, 0.10890),
    (0.0703, 0.10091),
    (0.0625, 0.09233),
    (0.0000, 0.00000),
)

# The v table validation/metrics.py carried from d589b9f (2026-04-16) until
# revision r2: Table I's y stations paired with values that are mostly not
# Table II's. Kept as the planted control for the conservation check.
CORRUPTED_V_TABLE_D589B9F: tuple[tuple[float, float], ...] = (
    (1.0000, 0.00000),
    (0.9688, -0.05906),
    (0.9609, -0.07391),
    (0.9531, -0.08864),
    (0.8516, -0.24533),
    (0.7344, -0.22445),
    (0.6172, -0.16914),
    (0.5000, -0.11477),
    (0.4531, -0.10313),
    (0.2813, -0.04272),
    (0.1719, 0.02135),
    (0.1016, 0.07156),
    (0.0703, 0.09515),
    (0.0625, 0.10091),
    (0.0547, 0.10643),
    (0.0000, 0.00000),
)

# Lid velocity times cavity length; test_both_tables_conserve_mass_along_their_centerline
# gives the quadrature argument for the value.
CENTERLINE_FLUX_TOL = 0.02


def _centerline_flux(positions: tuple[float, ...], values: tuple[float, ...]) -> float:
    """Trapezoid integral of a tabulated centerline profile over the unit cavity."""
    order = np.argsort(positions)
    return float(np.trapezoid(np.asarray(values)[order], np.asarray(positions)[order]))


@pytest.mark.unit
class TestGhiaTables:
    """The reference tables themselves, checked before any solver is scored on them."""

    def test_v_table_is_table_ii_value_for_value(self) -> None:
        """Revision r2 holds exactly the seventeen Table II pairs."""
        assert tuple(zip(GHIA_V_X, GHIA_V_VAL, strict=True)) == TABLE_II_RE100

    def test_both_tables_conserve_mass_along_their_centerline(self) -> None:
        """Net flux through each centerline of a closed cavity is zero, from the tables alone.

        Along y = 0.5 the integral of v dx is the net vertical flux, and along
        x = 0.5 the integral of u dy the net horizontal flux; both vanish in a
        closed cavity. The trapezoid rule on these few unevenly spaced
        stations has its own error, and the u table, which matches Table I,
        shows it: 0.007. The tolerance of 0.02 sits above that; the corrupted
        v table integrates to -0.095 and misses it by a factor of about five.
        """
        assert abs(_centerline_flux(GHIA_V_X, GHIA_V_VAL)) < CENTERLINE_FLUX_TOL
        assert abs(_centerline_flux(GHIA_U_Y, GHIA_U_VAL)) < CENTERLINE_FLUX_TOL

    def test_conservation_check_fails_on_the_corrupted_table(self) -> None:
        """The planted control: the table used from d589b9f to r2 must fail the check.

        A guard shown only passing on good data has not been shown to guard
        anything.
        """
        x, v = zip(*CORRUPTED_V_TABLE_D589B9F, strict=True)
        flux = abs(_centerline_flux(x, v))
        assert not flux < CENTERLINE_FLUX_TOL
        # Not a marginal failure: the table misses by more than four tolerances.
        assert flux > 4.0 * CENTERLINE_FLUX_TOL

    def test_endpoints_are_the_wall_and_lid_conditions(self) -> None:
        """No slip at the floor and both side walls, the lid speed at the top, exactly."""
        u_at = dict(zip(GHIA_U_Y, GHIA_U_VAL, strict=True))
        v_at = dict(zip(GHIA_V_X, GHIA_V_VAL, strict=True))
        assert u_at[0.0] == 0.0
        assert u_at[1.0] == 1.0
        assert v_at[0.0] == 0.0
        assert v_at[1.0] == 0.0

    def test_positions_are_strictly_monotone(self) -> None:
        """Each table runs from the lid or right wall down to zero without repeats."""
        for positions in (GHIA_U_Y, GHIA_V_X):
            assert np.all(np.diff(positions) < 0.0)
            assert (positions[0], positions[-1]) == (1.0, 0.0)
            assert len(set(positions)) == len(positions)


# Marchi's stations are uniform at 1/16. The trapezoid error there, estimated from the
# table alone as (T(h) - T(2h)) / 3, is at most 9.2e-4 over [0, 1/2] and 2.6e-3 over
# [0, 1], the last from u's rise to the lid.
MARCHI_FLUX_TOL = 0.005
MARCHI_U = tuple(r[1] for r in MARCHI_U_ROWS)
MARCHI_V = tuple(r[1] for r in MARCHI_V_ROWS)


def _marchi_flux(
    values: tuple[float, ...] | list[float], top: float, upto: float = 1.0
) -> float:
    """Trapezoid integral of a profile at Marchi's stations, walls appended, to upto."""
    s = [0.0, *(k / 16 for k in range(1, 16)), 1.0]
    f = [0.0, *values, top]
    keep = [k for k, position in enumerate(s) if position <= upto]
    return _centerline_flux(tuple(s[k] for k in keep), tuple(f[k] for k in keep))


@pytest.mark.unit
class TestMarchiTable:
    """Marchi et al. (2009), Re = 100, checked against itself before anything uses it.

    The net flux through each centerline is zero. The flux up through y = 0.5
    left of x = 0.5 is the published M, and so is the flux leftward through
    x = 0.5 below y = 0.5, since the two half-lines close the lower left
    quadrant. On uniform stations the full-line trapezoid cannot see interior
    stations swapped, so the M checks carry that control.
    """

    def test_stations_are_the_sixteenths(self) -> None:
        """Both profiles hold the fifteen interior stations k / 16, in order."""
        for rows in (MARCHI_U_ROWS, MARCHI_V_ROWS):
            assert [r[0] for r in rows] == [k / 16 for k in range(1, 16)]

    def test_both_profiles_conserve_mass_and_carry_the_published_m(self) -> None:
        """Full lines give zero; v over [0, 1/2] gives M and u over [0, 1/2] gives -M.

        M is a separate row of Table 6, so the last two tie both columns to it.
        """
        assert abs(_marchi_flux(MARCHI_U, 1.0)) < MARCHI_FLUX_TOL
        assert abs(_marchi_flux(MARCHI_V, 0.0)) < MARCHI_FLUX_TOL
        assert abs(_marchi_flux(MARCHI_V, 0.0, upto=0.5) - MARCHI_M) < MARCHI_FLUX_TOL
        assert abs(_marchi_flux(MARCHI_U, 1.0, upto=0.5) + MARCHI_M) < MARCHI_FLUX_TOL

    def test_m_check_fails_with_two_stations_swapped(self) -> None:
        """v at x = 0.25 and 0.75 swapped misses M by 0.026 yet still conserves mass."""
        v = list(MARCHI_V)
        v[3], v[11] = v[11], v[3]
        assert abs(_marchi_flux(v, 0.0)) < MARCHI_FLUX_TOL
        assert abs(_marchi_flux(v, 0.0, upto=0.5) - MARCHI_M) > 4 * MARCHI_FLUX_TOL

    def test_conservation_check_fails_on_a_one_row_offset(self) -> None:
        """Each v value read from the row above it misses zero net flux by 0.045.

        pdftotext -layout prints every Table 7 label one line below its values.
        Read that way, Table 6 puts u(0.5; 0.9375), the row above v(0.0625; 0.5),
        at the first v station and drops the last v value.
        """
        v = [MARCHI_U[-1], *MARCHI_V[:-1]]
        assert abs(_marchi_flux(v, 0.0)) > 4 * MARCHI_FLUX_TOL


def _marchi_fields(
    u_lid: float,
) -> tuple[SimConfig, SimpleNamespace, np.ndarray, np.ndarray]:
    """Marchi's values times the lid speed, on a stand-in mesh centred on Marchi's stations.

    Fifteen centers at k / 16 put both midlines on the middle column and row,
    and every station on a node, where the cubic returns the node's value.
    """
    raw = yaml.safe_load(case_path("cavity").read_text(encoding="utf-8"))
    raw["boundaries"]["lid"]["u_velocity"] = u_lid
    s = np.array([k / 16 for k in range(1, 16)])
    mesh = SimpleNamespace(
        xc=s,
        yc=s,
        x=np.array([0.0, 1.0]),
        y=np.array([0.0, 1.0]),
        cell_type=np.full((15, 15), FLUID),
    )
    u = np.tile(u_lid * np.array([r[1] for r in MARCHI_U_ROWS])[:, None], (1, 15))
    v = np.tile(u_lid * np.array([r[1] for r in MARCHI_V_ROWS])[None, :], (15, 1))
    return SimConfig.from_dict(raw), mesh, u, v


@pytest.mark.unit
class TestMarchiMetric:
    """cavity_marchi_centerline_errors: VAL-002 and criterion 3a since ECR-001 step 8."""

    def test_marchi_own_values_at_the_stations_score_zero(self) -> None:
        """Read at Ghia's stations, or with u and v swapped, this is not zero."""
        metric = cavity_marchi_centerline_errors(*_marchi_fields(2.0))
        assert metric.components == {"u": 0.0, "v": 0.0}
        assert (metric.metric, metric.reference) == (
            "max_normalized_centerline_error_cubic",
            MARCHI_REFERENCE,
        )

    def test_an_offset_at_one_station_is_read_in_its_own_component(self) -> None:
        """v at x = 0.8125 raised by 0.008 under a lid of 2 reads 0.004; u stays 0."""
        config, mesh, u, v = _marchi_fields(2.0)
        v[:, 12] += 0.008
        metric = cavity_marchi_centerline_errors(config, mesh, u, v)
        assert metric.components["u"] == 0.0
        assert metric.components["v"] == pytest.approx(0.004, rel=1e-12)
        assert metric.value == metric.components["v"]

    def test_between_nodes_the_cubic_reads_a_cubic_profile_exactly(self) -> None:
        """Review 26 B1, test 26 T1: on a real 16x16 mesh no station is a node.

        u = U (y^3 - 3 y (1 - y)) and v = U x (1 - x)(x - 1/2), U the lid speed,
        are cubics that meet the wall values the profiles append, so the cubic
        through four nodes reads them exactly at every station and linear
        interpolation does not. u's worst error is negative and larger than
        v's, so a signed error, or v's value in place of the larger, reads
        differently. (U y^3 alone is y^3 over the lid, above Marchi everywhere.)
        """
        config = load_case("cavity", grid=(16, 16))
        mesh = Mesh(config)
        lid = config.boundaries["lid"].u_velocity
        s = np.array([r[0] for r in MARCHI_U_ROWS])
        assert not np.isin(s, mesh.yc).any() and not np.isin(s, mesh.xc).any()

        def p(y: np.ndarray) -> np.ndarray:
            return y**3 - 3.0 * y * (1.0 - y)

        def q(x: np.ndarray) -> np.ndarray:
            return x * (1.0 - x) * (x - 0.5)

        x, y = np.meshgrid(np.asarray(mesh.xc), np.asarray(mesh.yc))
        metric = cavity_marchi_centerline_errors(config, mesh, lid * p(y), lid * q(x))
        err_u = p(s) - np.array([r[1] for r in MARCHI_U_ROWS])
        err_v = q(s) - np.array([r[1] for r in MARCHI_V_ROWS])
        assert err_u[np.abs(err_u).argmax()] < -np.abs(err_v).max()
        assert metric.components["u"] == pytest.approx(np.abs(err_u).max(), rel=1e-12)
        assert metric.components["v"] == pytest.approx(np.abs(err_v).max(), rel=1e-12)
        assert metric.value == metric.components["u"]


@pytest.mark.unit
class TestLagrange:
    """lagrange: the cubic through the four nearest nodes, test 26b T1.

    A single 1 among zeros is not a cubic, so unlike the cubic profiles of
    TestMarchiMetric it reads differently through every choice of four nodes.
    What it reads is the weight lagrange gives that node.
    """

    NODES: np.ndarray = np.arange(8.0)

    def _weights(self, target: float) -> list[float]:
        """Each node's weight at target, read from a single 1 among zeros."""
        at = np.array([target])
        return [float(lagrange(self.NODES, e, at)[0]) for e in np.eye(len(self.NODES))]

    def test_at_a_midpoint_the_stencil_is_centered(self) -> None:
        """Between nodes 3 and 4 the weights are (-1, 9, 9, -1) / 16 on nodes 2 to 5.

        A stencil started one node up reads (5, 15, -5, 1) / 16 on nodes 3 to 6,
        and one node down (1, -5, 15, 5) / 16 on nodes 1 to 4.
        """
        expected = [0.0, 0.0, -1 / 16, 9 / 16, 9 / 16, -1 / 16, 0.0, 0.0]
        assert self._weights(3.5) == pytest.approx(expected, abs=1e-15)

    @pytest.mark.parametrize(
        ("target", "expected"),
        [
            (0.5, [5 / 16, 15 / 16, -5 / 16, 1 / 16, 0.0, 0.0, 0.0, 0.0]),
            (6.5, [0.0, 0.0, 0.0, 0.0, 1 / 16, -5 / 16, 15 / 16, 5 / 16]),
        ],
    )
    def test_at_each_end_the_stencil_shifts_inward(
        self, target: float, expected: list[float]
    ) -> None:
        """Between the first two nodes it is nodes 0 to 3, between the last two 4 to 7.

        Centered, these stencils would start at node -1 and end at node 8.
        """
        assert self._weights(target) == pytest.approx(expected, abs=1e-15)


# ---------------------------------------------------------------------------
# Transport metrics and cases (ADR-011 H; prompt 32 decision 2)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTransportMetrics:
    """The four field metrics on a mesh with an obstacle, so SOLID cells count for nothing."""

    def setup_method(self) -> None:
        self.config = transport_config(
            2.0,
            1.2,
            40,
            24,
            obstacles=[
                {
                    "name": "b",
                    "x_start": 0.1,
                    "x_end": 0.3,
                    "y_start": 0.1,
                    "y_end": 0.2,
                }
            ],
        )
        self.mesh = Mesh(self.config)
        assert (self.mesh.cell_type == SOLID).sum() == 8
        self.field = gaussian_cell_averages(self.mesh.x, self.mesh.y, (1.1, 0.7), 0.1)

    def test_a_field_against_itself_scores_zero(self) -> None:
        assert relative_l2(self.field, self.field, self.mesh) == 0.0

    def test_relative_l2_ignores_solid_cells_and_is_the_ratio_of_norms(self) -> None:
        other = self.field.copy()
        other[0, 0] += 0.3  # a BOUNDARY-ring cell, which counts
        solid = self.mesh.cell_type == SOLID
        other[solid] += 100.0  # must not count
        mask = ~solid
        expected = 0.3 / np.linalg.norm(self.field[mask])
        assert relative_l2(other, self.field, self.mesh) == pytest.approx(expected)

    def test_relative_l2_refuses_a_zero_reference(self) -> None:
        with pytest.raises(ValueError, match="nonzero exact field"):
            relative_l2(self.field, np.zeros_like(self.field), self.mesh)

    def test_the_centroid_of_a_translated_field_moves_by_the_translation(self) -> None:
        # sigma 0.06 keeps both pulses over seven sigma from every edge, so
        # the tail the domain cuts moves the centroid by less than 1e-12.
        a = gaussian_cell_averages(self.mesh.x, self.mesh.y, (0.9, 0.6), 0.06)
        b = gaussian_cell_averages(self.mesh.x, self.mesh.y, (1.3, 0.75), 0.06)
        xa, ya = centroid(a, self.mesh)
        xb, yb = centroid(b, self.mesh)
        assert xa == pytest.approx(0.9, abs=1e-9) and ya == pytest.approx(0.6, abs=1e-9)
        assert xb - xa == pytest.approx(0.4, abs=1e-9)
        assert yb - ya == pytest.approx(0.15, abs=1e-9)

    def test_centroid_refuses_an_empty_field(self) -> None:
        with pytest.raises(ValueError, match="nonzero content"):
            centroid(np.zeros_like(self.field), self.mesh)

    def test_peak_retention_and_minimum(self) -> None:
        clipped = 0.8 * self.field
        clipped[10, 10] = -1e-3
        clipped[2, 2] = -5.0  # a SOLID cell: not read
        assert self.mesh.cell_type[2, 2] == SOLID
        assert peak_retention(clipped, self.field, self.mesh) == pytest.approx(0.8)
        assert field_minimum(clipped, self.mesh) == -1e-3
        assert field_minimum(self.field, self.mesh) >= 0.0
        with pytest.raises(ValueError, match="positive peak"):
            peak_retention(self.field, np.zeros_like(self.field), self.mesh)


@pytest.mark.unit
class TestTransportCases:
    def test_gaussian_cell_averages_integrate_to_the_analytical_mass(self) -> None:
        config = transport_config(2.0, 1.2, 100, 60)
        mesh = Mesh(config)
        field = gaussian_cell_averages(mesh.x, mesh.y, (0.93, 0.61), 0.08, 2.5)
        content = (field * mesh.dx * mesh.dy).sum()
        assert content == pytest.approx(gaussian_mass(0.08, 2.5), rel=1e-12)
        # The peak cell average is below the continuous peak, which is off node.
        assert field.max() < 2.5

    def test_the_heat_kernel_keeps_its_mass_and_doubles_sigma_at_the_design_time(
        self,
    ) -> None:
        case = diffusion_case()
        mesh = case.mesh
        volume = mesh.dx * mesh.dy
        assert (case.initial * volume).sum() == pytest.approx(
            gaussian_mass(0.05), rel=1e-12
        )
        # At t_end the wall is 5.9 sigma from the centre, so the tail the
        # domain cuts is a few 1e-9 of the content (ADR-011 H).
        assert (case.exact * volume).sum() == pytest.approx(
            gaussian_mass(0.05), rel=1e-8
        )
        # The continuous amplitude is sigma_0^2 / sigma^2 = 1/4; the peak cell
        # average of a ten-cell sigma is within 0.4% of it.
        assert case.exact.max() == pytest.approx(0.25, rel=5e-3)
        assert case.t_end == pytest.approx(3.75)

    def test_the_rotation_and_smith_hutton_fields_are_divergence_free(self) -> None:
        for case in (rotating_puff_case(), smith_hutton_case()):
            mesh = case.mesh
            div = (case.faces.u[:, 1:] - case.faces.u[:, :-1]) * mesh.dy + (
                case.faces.v[1:, :] - case.faces.v[:-1, :]
            ) * mesh.dx
            scale = np.abs(case.faces.u).max() * mesh.dy
            assert np.abs(div).max() <= 1e-14 * scale, case.name

    def test_smith_hutton_walls_have_zero_normal_velocity_and_the_inlet_its_profile(
        self,
    ) -> None:
        case = smith_hutton_case()
        faces = case.faces
        assert np.all(faces.u[:, 0] == 0.0) and np.all(faces.u[:, -1] == 0.0)
        assert np.all(faces.v[-1, :] == 0.0)
        inflow = case.conditions.faces_for(0).inflow_v
        x_prime = case.mesh.xc - 1.0
        assert np.all(faces.v[0, x_prime < 0.0] > 0.0)
        assert np.all(faces.v[0, x_prime > 0.0] < 0.0)
        assert np.all(inflow[0, x_prime > 0.0] == 0.0)
        assert inflow[0, 0] == pytest.approx(1.0 - np.tanh(10.0 * 0.98), rel=1e-12)
        assert np.all(inflow[1:, :] == 0.0)

    def test_the_oblique_pulse_travels_one_metre_in_its_time(self) -> None:
        case = oblique_pulse_case()
        x0, y0 = centroid(case.initial, case.mesh)
        x1, y1 = centroid(case.exact, case.mesh)
        # The end is 4.4 sigma from the ceiling; the tail the domain cuts is
        # below 1e-4 of the peak (ADR-011 H) and moves the centroid by 2e-5.
        assert np.hypot(x1 - x0, y1 - y0) == pytest.approx(1.0, abs=1e-4)
        assert case.faces.u[0, 0] == pytest.approx(0.45 * np.cos(np.radians(30.0)))
        assert case.config.transport.cfl_number == 0.1

    def test_the_sealed_box_conditions_settle_inside_and_deposit_on_the_floor_only(
        self,
    ) -> None:
        case = sealed_box_case(settling=1e-3)
        faces = case.conditions.faces_for(0)
        assert np.all(faces.deposition_v[0, :] == 1e-3)
        assert np.all(faces.deposition_v[1:, :] == 0.0)
        assert np.all(faces.surface_v[0, :] == SURFACE_FLOOR)
        assert faces.settling_v[1:-1, :].all()
        assert not faces.settling_v[0, :].any() and not faces.settling_v[-1, :].any()
        assert case.t_end == pytest.approx(3.0 / 1e-3)

    def test_conditions_with_rejects_an_unknown_field_or_shape(self) -> None:
        mesh = Mesh(transport_config(1.0, 0.5, 4, 2))
        with pytest.raises(ValueError, match="not a ConcentrationFaces field"):
            conditions_with(mesh, settling_u=np.zeros((3, 4)))
        with pytest.raises(ValueError, match="must have shape"):
            conditions_with(mesh, inflow_u=np.zeros((2, 4)))

    def test_scalar_physics_checks_the_class_and_fixed_conditions_the_type(
        self,
    ) -> None:
        physics = ScalarPhysics(settling=1.0, diffusion=2.0)
        assert physics.settling_velocity(0) == 1.0 and physics.diffusion_coeff(0) == 2.0
        with pytest.raises(IndexError):
            physics.settling_velocity(1)
        with pytest.raises(TypeError):
            physics.diffusion_coeff(True)
        conditions = FixedConditions(
            zero_conditions(Mesh(transport_config(1.0, 0.5, 4, 2)))
        )
        with pytest.raises(TypeError):
            conditions.faces_for(True)
        with pytest.raises(IndexError):
            conditions.faces_for(-1)
