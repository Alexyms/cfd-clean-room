"""Tests for the validation package: case loading and error metrics."""

import numpy as np
import pytest

from src.mesh import Mesh
from validation.cases import CASE_FILES, CASE_GRIDS, case_path, load_case
from validation.metrics import (
    GHIA_U_VAL,
    GHIA_U_Y,
    GHIA_V_VAL,
    GHIA_V_X,
    cavity_centerline_errors,
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
        assert metric.reference == "ghia_1982_re100"
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
