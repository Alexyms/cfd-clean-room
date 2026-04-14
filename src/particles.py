"""Size-dependent particle transport properties for clean room simulation.

Computes Cunningham slip correction, Stokes settling velocity, Brownian
diffusion coefficient, deposition velocity, and HEPA filter efficiency
for each of five particle size classes spanning the range from
diffusion-dominated (0.1 um) to settling-dominated (5.0 um).
"""

import math

# Physical constants (SI units)
BOLTZMANN_CONSTANT: float = 1.380649e-23  # J/K, exact by 2019 SI redefinition
GRAVITY: float = 9.80665  # m/s^2, standard gravity

# Cunningham correction empirical coefficients (Allen and Raabe, 1985)
_CUNNINGHAM_A1: float = 1.257
_CUNNINGHAM_A2: float = 0.4
_CUNNINGHAM_A3: float = 1.1


class ParticlePhysics:
    """Compute size-dependent particle transport properties.

    Provides settling velocity, Brownian diffusion coefficient,
    Cunningham slip correction, deposition velocity, and HEPA filter
    efficiency for each configured particle size class.

    Parameters
    ----------
    particle_sizes : list[float]
        Particle diameters in meters for each size class.
    particle_density : float
        Particle material density in kg/m^3.
    temperature : float
        Air temperature in K.
    mu : float
        Dynamic viscosity of air in Pa*s.
    mean_free_path : float
        Mean free path of air molecules in meters.
    """

    def __init__(
        self,
        particle_sizes: list[float],
        particle_density: float,
        temperature: float,
        mu: float,
        mean_free_path: float,
    ) -> None:
        self._particle_sizes = particle_sizes
        self._rho_p = particle_density
        self._temperature = temperature
        self._mu = mu
        self._lambda = mean_free_path
        self._n_classes = len(particle_sizes)

    @property
    def n_classes(self) -> int:
        """Number of particle size classes."""
        return self._n_classes

    @property
    def particle_sizes(self) -> list[float]:
        """Particle diameters in meters for each size class."""
        return list(self._particle_sizes)

    def _diameter(self, size_class: int) -> float:
        """Return particle diameter for the given size class index.

        Parameters
        ----------
        size_class : int
            Index into the particle sizes array.

        Returns
        -------
        float
            Particle diameter in meters.

        Raises
        ------
        IndexError
            If size_class is outside the valid range.
        """
        if size_class < 0 or size_class >= self._n_classes:
            raise IndexError(
                f"size_class {size_class} out of range [0, {self._n_classes - 1}]"
            )
        return self._particle_sizes[size_class]

    def cunningham_correction(self, size_class: int) -> float:
        """Compute the Cunningham slip correction factor.

        Accounts for non-continuum effects when particle diameter
        approaches the mean free path of air. The correction is
        significant below 1 um and negligible above ~5 um.

        Parameters
        ----------
        size_class : int
            Index into the configured particle sizes array.

        Returns
        -------
        float
            Dimensionless correction factor, always >= 1.0.

        Notes
        -----
        C_c = 1 + (2*lambda/d_p) * (A1 + A2 * exp(-A3 * d_p / (2*lambda)))

        where A1=1.257, A2=0.4, A3=1.1 (Allen and Raabe, 1985).
        """
        d_p = self._diameter(size_class)
        kn_factor = 2.0 * self._lambda / d_p
        return 1.0 + kn_factor * (
            _CUNNINGHAM_A1
            + _CUNNINGHAM_A2 * math.exp(-_CUNNINGHAM_A3 * d_p / (2.0 * self._lambda))
        )

    def settling_velocity(self, size_class: int) -> float:
        """Compute Stokes settling velocity with Cunningham slip correction.

        Uses the Stokes drag law corrected for slip at small particle sizes.
        Valid for Re_p << 1 (Stokes regime), which holds for all particle
        sizes in the clean room operating range (d_p <= 5 um).

        Parameters
        ----------
        size_class : int
            Index into the configured particle sizes array.

        Returns
        -------
        float
            Terminal settling velocity in m/s. Positive downward.

        Notes
        -----
        v_s = (rho_p * d_p^2 * g * C_c) / (18 * mu)
        """
        d_p = self._diameter(size_class)
        c_c = self.cunningham_correction(size_class)
        return (self._rho_p * d_p**2 * GRAVITY * c_c) / (18.0 * self._mu)

    def diffusion_coeff(self, size_class: int) -> float:
        """Compute Brownian diffusion coefficient via Stokes-Einstein relation.

        Diffusion dominates transport for sub-micron particles. The
        Cunningham correction is applied to account for slip effects.

        Parameters
        ----------
        size_class : int
            Index into the configured particle sizes array.

        Returns
        -------
        float
            Diffusion coefficient in m^2/s.

        Notes
        -----
        D = (k_B * T * C_c) / (3 * pi * mu * d_p)
        """
        d_p = self._diameter(size_class)
        c_c = self.cunningham_correction(size_class)
        return (BOLTZMANN_CONSTANT * self._temperature * c_c) / (
            3.0 * math.pi * self._mu * d_p
        )

    def deposition_velocity(self, size_class: int, surface: str) -> float:
        """Compute particle deposition velocity onto a surface.

        Combines gravitational settling and diffusive deposition. For
        floor surfaces, both mechanisms act together. For ceiling
        surfaces, only diffusive deposition contributes (gravity pulls
        particles away from the ceiling). For wall surfaces, only
        diffusive deposition contributes (gravity acts perpendicular
        to the wall).

        Parameters
        ----------
        size_class : int
            Index into the configured particle sizes array.
        surface : str
            Surface orientation: "floor", "ceiling", or "wall".

        Returns
        -------
        float
            Deposition velocity in m/s. Always non-negative.

        Raises
        ------
        ValueError
            If surface is not "floor", "ceiling", or "wall".

        Notes
        -----
        The diffusive deposition velocity is estimated as
        v_diff = D / delta, where delta is a boundary layer thickness.
        A typical value of delta = 1e-3 m is used, consistent with
        laminar boundary layers in clean room environments.
        """
        valid_surfaces = ("floor", "ceiling", "wall")
        if surface not in valid_surfaces:
            raise ValueError(
                f"surface must be one of {valid_surfaces}, got '{surface}'"
            )

        v_s = self.settling_velocity(size_class)
        d_coeff = self.diffusion_coeff(size_class)

        # Laminar boundary layer thickness for clean room conditions
        delta = 1e-3  # m
        v_diff = d_coeff / delta

        if surface == "floor":
            return v_s + v_diff
        # Ceiling and wall: gravity does not assist deposition
        return v_diff

    def hepa_efficiency(self, size_class: int) -> float:
        """Estimate HEPA filter single-pass collection efficiency.

        Uses the single-fiber efficiency model combining diffusion and
        interception mechanisms. Real HEPA filters achieve >= 99.97%
        efficiency at the most penetrating particle size (MPPS, ~0.3 um).
        This model captures the size-dependent trend for use in
        boundary condition calculations.

        Parameters
        ----------
        size_class : int
            Index into the configured particle sizes array.

        Returns
        -------
        float
            Collection efficiency as a fraction in [0, 1].

        Notes
        -----
        This is a simplified model for boundary condition purposes.
        The minimum efficiency occurs near 0.3 um where neither
        diffusion nor interception dominates. Efficiency values are
        based on typical HEPA filter performance data:

        - 0.1 um: 0.99999 (diffusion-dominated capture)
        - 0.3 um: 0.99970 (MPPS, minimum by definition)
        - 0.5 um: 0.99990 (interception begins to dominate)
        - 1.0 um: 0.99999 (interception-dominated)
        - 5.0 um: 0.99999 (inertial impaction dominant)

        For particle sizes not in the reference table, linear
        interpolation in log-diameter space is used.
        """
        d_p = self._diameter(size_class)
        d_um = d_p * 1e6  # convert to micrometers for lookup

        # Reference efficiency data points (diameter in um, efficiency)
        ref_diameters = [0.1, 0.3, 0.5, 1.0, 5.0]
        ref_efficiencies = [0.99999, 0.99970, 0.99990, 0.99999, 0.99999]

        # Clamp to reference range
        if d_um <= ref_diameters[0]:
            return ref_efficiencies[0]
        if d_um >= ref_diameters[-1]:
            return ref_efficiencies[-1]

        # Linear interpolation in log-diameter space
        log_d = math.log(d_um)
        for i in range(len(ref_diameters) - 1):
            log_d_lo = math.log(ref_diameters[i])
            log_d_hi = math.log(ref_diameters[i + 1])
            if log_d_lo <= log_d <= log_d_hi:
                t = (log_d - log_d_lo) / (log_d_hi - log_d_lo)
                return ref_efficiencies[i] + t * (
                    ref_efficiencies[i + 1] - ref_efficiencies[i]
                )

        # Should not reach here, but return MPPS value as fallback
        return 0.99970
