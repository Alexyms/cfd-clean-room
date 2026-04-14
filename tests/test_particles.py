"""Tests for the particle physics module.

Unit tests verify correct types, edge cases, and monotonicity of
transport properties across size classes. Validation tests verify
computed values against hand-calculated analytical solutions.
"""

import pytest

from src.particles import ParticlePhysics

# Standard conditions matching the analytical reference calculations
PARTICLE_SIZES: list[float] = [0.1e-6, 0.3e-6, 0.5e-6, 1.0e-6, 5.0e-6]
PARTICLE_DENSITY: float = 1000.0  # kg/m^3
TEMPERATURE: float = 293.0  # K
MU: float = 1.81e-5  # Pa*s
MEAN_FREE_PATH: float = 67e-9  # m


@pytest.fixture()
def physics() -> ParticlePhysics:
    """Create a ParticlePhysics instance with standard conditions."""
    return ParticlePhysics(
        particle_sizes=PARTICLE_SIZES,
        particle_density=PARTICLE_DENSITY,
        temperature=TEMPERATURE,
        mu=MU,
        mean_free_path=MEAN_FREE_PATH,
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParticlePhysicsUnit:
    """Unit tests for ParticlePhysics methods."""

    def test_n_classes(self, physics: ParticlePhysics) -> None:
        """ParticlePhysics reports the correct number of size classes."""
        assert physics.n_classes == 5

    def test_particle_sizes_returns_copy(self, physics: ParticlePhysics) -> None:
        """particle_sizes property returns a copy, not the internal list."""
        sizes = physics.particle_sizes
        sizes[0] = 999.0
        assert physics.particle_sizes[0] != 999.0

    def test_cunningham_returns_float(self, physics: ParticlePhysics) -> None:
        """cunningham_correction returns a float for each size class."""
        for k in range(physics.n_classes):
            assert isinstance(physics.cunningham_correction(k), float)

    def test_settling_returns_float(self, physics: ParticlePhysics) -> None:
        """settling_velocity returns a float for each size class."""
        for k in range(physics.n_classes):
            assert isinstance(physics.settling_velocity(k), float)

    def test_diffusion_returns_float(self, physics: ParticlePhysics) -> None:
        """diffusion_coeff returns a float for each size class."""
        for k in range(physics.n_classes):
            assert isinstance(physics.diffusion_coeff(k), float)

    def test_all_cunningham_positive(self, physics: ParticlePhysics) -> None:
        """Cunningham correction is >= 1.0 for all size classes."""
        for k in range(physics.n_classes):
            assert physics.cunningham_correction(k) >= 1.0

    def test_all_settling_positive(self, physics: ParticlePhysics) -> None:
        """Settling velocity is positive for all size classes."""
        for k in range(physics.n_classes):
            assert physics.settling_velocity(k) > 0.0

    def test_all_diffusion_positive(self, physics: ParticlePhysics) -> None:
        """Diffusion coefficient is positive for all size classes."""
        for k in range(physics.n_classes):
            assert physics.diffusion_coeff(k) > 0.0

    def test_settling_increases_with_size(self, physics: ParticlePhysics) -> None:
        """Settling velocity increases monotonically with particle size."""
        velocities = [physics.settling_velocity(k) for k in range(physics.n_classes)]
        for i in range(len(velocities) - 1):
            assert velocities[i] < velocities[i + 1]

    def test_diffusion_decreases_with_size(self, physics: ParticlePhysics) -> None:
        """Diffusion coefficient decreases monotonically with particle size."""
        coefficients = [physics.diffusion_coeff(k) for k in range(physics.n_classes)]
        for i in range(len(coefficients) - 1):
            assert coefficients[i] > coefficients[i + 1]

    def test_cunningham_decreases_with_size(self, physics: ParticlePhysics) -> None:
        """Cunningham correction decreases toward 1.0 for larger particles."""
        corrections = [
            physics.cunningham_correction(k) for k in range(physics.n_classes)
        ]
        for i in range(len(corrections) - 1):
            assert corrections[i] > corrections[i + 1]

    def test_invalid_size_class_raises(self, physics: ParticlePhysics) -> None:
        """Out-of-range size class index raises IndexError."""
        with pytest.raises(IndexError):
            physics.settling_velocity(-1)
        with pytest.raises(IndexError):
            physics.settling_velocity(5)

    def test_deposition_floor_greater_than_wall(self, physics: ParticlePhysics) -> None:
        """Floor deposition velocity exceeds wall deposition velocity.

        Floor deposition includes both settling and diffusion, while
        wall deposition includes only diffusion.
        """
        for k in range(physics.n_classes):
            v_floor = physics.deposition_velocity(k, "floor")
            v_wall = physics.deposition_velocity(k, "wall")
            assert v_floor > v_wall

    def test_deposition_ceiling_equals_wall(self, physics: ParticlePhysics) -> None:
        """Ceiling and wall deposition velocities are equal.

        Both exclude gravitational settling.
        """
        for k in range(physics.n_classes):
            v_ceiling = physics.deposition_velocity(k, "ceiling")
            v_wall = physics.deposition_velocity(k, "wall")
            assert v_ceiling == v_wall

    def test_deposition_invalid_surface_raises(self, physics: ParticlePhysics) -> None:
        """Invalid surface type raises ValueError."""
        with pytest.raises(ValueError, match="surface must be one of"):
            physics.deposition_velocity(0, "diagonal")

    def test_deposition_all_positive(self, physics: ParticlePhysics) -> None:
        """Deposition velocity is positive for all classes and surfaces."""
        for k in range(physics.n_classes):
            for surface in ("floor", "ceiling", "wall"):
                assert physics.deposition_velocity(k, surface) > 0.0

    def test_hepa_efficiency_range(self, physics: ParticlePhysics) -> None:
        """HEPA efficiency is in [0, 1] for all size classes."""
        for k in range(physics.n_classes):
            eff = physics.hepa_efficiency(k)
            assert 0.0 <= eff <= 1.0

    def test_hepa_minimum_at_mpps(self, physics: ParticlePhysics) -> None:
        """HEPA efficiency is lowest at 0.3 um (most penetrating particle size)."""
        eff_mpps = physics.hepa_efficiency(1)  # 0.3 um
        for k in range(physics.n_classes):
            assert physics.hepa_efficiency(k) >= eff_mpps

    def test_hepa_interpolation_intermediate_diameter(self) -> None:
        """HEPA efficiency at 0.2 um falls between the 0.1 um and 0.3 um values."""
        pp = ParticlePhysics(
            particle_sizes=[0.2e-6],
            particle_density=1000.0,
            temperature=293.0,
            mu=1.81e-5,
            mean_free_path=67e-9,
        )
        eff = pp.hepa_efficiency(0)
        # 0.2 um sits between 0.1 um (0.99999) and 0.3 um (0.99970)
        assert 0.99970 < eff < 0.99999

    def test_hepa_clamp_below_smallest_reference(self) -> None:
        """HEPA efficiency at or below 0.1 um returns the 0.1 um reference value."""
        pp = ParticlePhysics(
            particle_sizes=[0.05e-6],
            particle_density=1000.0,
            temperature=293.0,
            mu=1.81e-5,
            mean_free_path=67e-9,
        )
        assert pp.hepa_efficiency(0) == 0.99999

    def test_hepa_clamp_above_largest_reference(self) -> None:
        """HEPA efficiency above 5.0 um returns the 5.0 um reference value."""
        pp = ParticlePhysics(
            particle_sizes=[10.0e-6],
            particle_density=1000.0,
            temperature=293.0,
            mu=1.81e-5,
            mean_free_path=67e-9,
        )
        assert pp.hepa_efficiency(0) == 0.99999


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

# Hand-calculated reference values at standard conditions:
# T=293K, rho_p=1000 kg/m^3, mu=1.81e-5 Pa*s, lambda=67e-9 m
# k_B=1.380649e-23 J/K, g=9.80665 m/s^2

REFERENCE_CUNNINGHAM: list[float] = [
    2.9202400543e00,  # 0.1 um
    1.5766834242e00,  # 0.3 um
    1.3386446537e00,  # 0.5 um
    1.1684525902e00,  # 1.0 um
    1.0336876000e00,  # 5.0 um
]

REFERENCE_SETTLING: list[float] = [
    8.7899853063e-07,  # 0.1 um
    4.2712658844e-06,  # 0.3 um
    1.0073372923e-05,  # 0.5 um
    3.5170674013e-05,  # 1.0 um
    7.7785547134e-04,  # 5.0 um
]

REFERENCE_DIFFUSION: list[float] = [
    6.9249996253e-10,  # 0.1 um
    1.2463053172e-10,  # 0.3 um
    6.3488710195e-11,  # 0.5 um
    2.7708454096e-11,  # 1.0 um
    4.9025327437e-12,  # 5.0 um
]


@pytest.mark.validation
class TestSettlingVelocityValidation:
    """VAL-005: Stokes settling velocity < 0.1% error vs analytical."""

    @pytest.mark.parametrize(
        ("size_class", "expected"),
        list(enumerate(REFERENCE_SETTLING)),
        ids=["0.1um", "0.3um", "0.5um", "1.0um", "5.0um"],
    )
    def test_settling_velocity(
        self,
        physics: ParticlePhysics,
        size_class: int,
        expected: float,
    ) -> None:
        """VAL-005: Settling velocity matches analytical value within 0.1%."""
        computed = physics.settling_velocity(size_class)
        relative_error = abs(computed - expected) / expected
        assert relative_error < 1e-3, (
            f"Size class {size_class}: relative error {relative_error:.2e} "
            f"exceeds 0.1% (computed={computed:.6e}, expected={expected:.6e})"
        )


@pytest.mark.validation
class TestDiffusionCoefficientValidation:
    """VAL-006: Brownian diffusion coefficient < 0.1% error vs analytical."""

    @pytest.mark.parametrize(
        ("size_class", "expected"),
        list(enumerate(REFERENCE_DIFFUSION)),
        ids=["0.1um", "0.3um", "0.5um", "1.0um", "5.0um"],
    )
    def test_diffusion_coeff(
        self,
        physics: ParticlePhysics,
        size_class: int,
        expected: float,
    ) -> None:
        """VAL-006: Diffusion coefficient matches analytical value within 0.1%."""
        computed = physics.diffusion_coeff(size_class)
        relative_error = abs(computed - expected) / expected
        assert relative_error < 1e-3, (
            f"Size class {size_class}: relative error {relative_error:.2e} "
            f"exceeds 0.1% (computed={computed:.6e}, expected={expected:.6e})"
        )


# Deposition velocity reference values derived from validated D and v_s:
# v_diff = D / delta (wall/ceiling), v_floor = v_s + D / delta
REFERENCE_DEPOSITION_WALL: list[float] = [
    6.9249996253e-07,  # 0.1 um
    1.2463053172e-07,  # 0.3 um
    6.3488710195e-08,  # 0.5 um
    2.7708454096e-08,  # 1.0 um
    4.9025327437e-09,  # 5.0 um
]

REFERENCE_DEPOSITION_FLOOR: list[float] = [
    1.5714984932e-06,  # 0.1 um
    4.3958964161e-06,  # 0.3 um
    1.0136861633e-05,  # 0.5 um
    3.5198382467e-05,  # 1.0 um
    7.7786037387e-04,  # 5.0 um
]


@pytest.mark.validation
class TestDepositionVelocityValidation:
    """Deposition velocity < 0.1% error vs analytical values.

    Expected values derived from the already-validated diffusion_coeff
    and settling_velocity: v_diff = D / delta, v_floor = v_s + v_diff.
    """

    @pytest.mark.parametrize(
        ("size_class", "expected"),
        list(enumerate(REFERENCE_DEPOSITION_WALL)),
        ids=["0.1um", "0.3um", "0.5um", "1.0um", "5.0um"],
    )
    def test_wall_deposition(
        self,
        physics: ParticlePhysics,
        size_class: int,
        expected: float,
    ) -> None:
        """Wall deposition velocity matches analytical value within 0.1%."""
        computed = physics.deposition_velocity(size_class, "wall")
        relative_error = abs(computed - expected) / expected
        assert relative_error < 1e-3, (
            f"Size class {size_class}: relative error {relative_error:.2e} "
            f"exceeds 0.1% (computed={computed:.6e}, expected={expected:.6e})"
        )

    @pytest.mark.parametrize(
        ("size_class", "expected"),
        list(enumerate(REFERENCE_DEPOSITION_FLOOR)),
        ids=["0.1um", "0.3um", "0.5um", "1.0um", "5.0um"],
    )
    def test_floor_deposition(
        self,
        physics: ParticlePhysics,
        size_class: int,
        expected: float,
    ) -> None:
        """Floor deposition velocity matches analytical value within 0.1%."""
        computed = physics.deposition_velocity(size_class, "floor")
        relative_error = abs(computed - expected) / expected
        assert relative_error < 1e-3, (
            f"Size class {size_class}: relative error {relative_error:.2e} "
            f"exceeds 0.1% (computed={computed:.6e}, expected={expected:.6e})"
        )


@pytest.mark.validation
class TestHepaEfficiencyValidation:
    """HEPA efficiency validation at an intermediate diameter.

    At 0.2 um, the expected efficiency is computed by log-space linear
    interpolation between the 0.1 um (0.99999) and 0.3 um (0.99970)
    reference points.
    """

    def test_hepa_interpolation_accuracy(self) -> None:
        """HEPA efficiency at 0.2 um matches hand-calculated value within 0.1%."""
        pp = ParticlePhysics(
            particle_sizes=[0.2e-6],
            particle_density=1000.0,
            temperature=293.0,
            mu=1.81e-5,
            mean_free_path=67e-9,
        )
        computed = pp.hepa_efficiency(0)
        expected = 9.9980703037e-01
        relative_error = abs(computed - expected) / expected
        assert relative_error < 1e-3, (
            f"Relative error {relative_error:.2e} exceeds 0.1% "
            f"(computed={computed:.10e}, expected={expected:.10e})"
        )


@pytest.mark.unit
class TestCunninghamCorrectionValidation:
    """Cunningham correction factor verification against analytical values."""

    @pytest.mark.parametrize(
        ("size_class", "expected"),
        list(enumerate(REFERENCE_CUNNINGHAM)),
        ids=["0.1um", "0.3um", "0.5um", "1.0um", "5.0um"],
    )
    def test_cunningham_correction(
        self,
        physics: ParticlePhysics,
        size_class: int,
        expected: float,
    ) -> None:
        """Cunningham correction matches analytical value within 0.1%."""
        computed = physics.cunningham_correction(size_class)
        relative_error = abs(computed - expected) / expected
        assert relative_error < 1e-3, (
            f"Size class {size_class}: relative error {relative_error:.2e} "
            f"exceeds 0.1% (computed={computed:.6e}, expected={expected:.6e})"
        )
