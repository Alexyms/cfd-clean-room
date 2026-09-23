"""Tests for the validation package: case loading and error metrics."""

import numpy as np
import pytest
import yaml

from src.config import SimConfig
from src.mesh import FLUID, Mesh
from validation.cases import CASE_FILES, CASE_GRIDS, case_path, load_case
from validation.metrics import (
    GHIA_U_VAL,
    GHIA_U_Y,
    GHIA_V_VAL,
    GHIA_V_X,
    _bracket,
    cavity_centerline_errors,
    cavity_centerline_profiles,
    cavity_true_centerline_errors,
    cavity_true_centerline_profiles,
    poiseuille_l2_error,
    poiseuille_profiles,
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
        assert config.max_pressure_iter == 2000
        assert config.pressure_tol == 1.0e-8
        assert config.alpha_velocity == 0.7

    def test_cavity_settings_are_the_validation_settings(self) -> None:
        config = load_case("cavity")
        assert (config.nx, config.ny) == (40, 40)
        assert config.max_pressure_iter == 500
        assert config.pressure_tol == 1.0e-8
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
