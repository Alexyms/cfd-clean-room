"""VAL-013: the Smith and Hutton (1982) bounded-advection case (REQ-T12, ADR-011 H).

A steep tanh profile enters the bottom edge for x' < 0, turns through a half
circle and leaves for x' > 0. Pure advection: no cell may fall below the
inlet's minimum, 1 - tanh(alpha), or rise above its maximum, 1 + tanh(alpha),
at any step, exact up to the rounding allowance of VAL-004. The steady outlet
profile against the inlet's mirror image is reported unscored, the limiter's
smearing of a front. The planted control removes the clamp in the solver by
monkeypatch: the unlimited quadratic breaks the bounds within a few hundred
steps.
"""

import math
from time import perf_counter

import numpy as np
import pytest

import src.scalar_scheme as scalar_scheme
from src.solver_transport import TransportSolver
from validation.transport_cases import (
    SMITH_HUTTON,
    SUPPLY_SPEED,
    TURBULENT_SCHMIDT,
    prescribed_eddy_viscosity,
    smith_hutton_case,
    smith_hutton_inlet,
)

ALPHA = SMITH_HUTTON["alpha"]
LOW = 1.0 - math.tanh(ALPHA)
HIGH = 1.0 + math.tanh(ALPHA)
ROUNDING = 1e-14 * HIGH
# Steady when the largest change per step is below this fraction of the inlet
# maximum, or at the 20 s cap, whichever comes first (ADR-011 H).
STEADY_FRACTION = 1e-10


@pytest.mark.validation
def test_smith_hutton_bounds_hold_at_every_step_val013() -> None:
    """VAL-013: every cell within [1 - tanh(10), 1 + tanh(10)] at every step."""
    case = smith_hutton_case()
    solver = TransportSolver(case.mesh, case.config, case.physics, case.conditions)
    dt = solver.stable_dt(case.faces, 0)
    cap = math.ceil(case.t_end / dt)
    c = case.initial.copy()
    worst_low, worst_high = float(c.min()), float(c.max())
    start = perf_counter()
    steps, change = 0, float("inf")
    for step in range(1, cap + 1):
        c_new = solver.solve_timestep(c, case.faces, 0, dt)
        low, high = float(c_new.min()), float(c_new.max())
        assert low >= LOW - ROUNDING, f"step {step}: {low} below {LOW}"
        assert high <= HIGH + ROUNDING, f"step {step}: {high} above {HIGH}"
        worst_low, worst_high = min(worst_low, low), max(worst_high, high)
        change = float(np.abs(c_new - c).max())
        c = c_new
        steps = step
        if change < STEADY_FRACTION * HIGH:
            break
    seconds = perf_counter() - start

    # Unscored: the outlet profile against the inlet's mirror image.
    x_prime = case.mesh.xc - 1.0
    outlet = x_prime > 0.0
    mirror = smith_hutton_inlet(-x_prime[outlet], ALPHA)
    profile = c[0, outlet]
    front_error = float(np.abs(profile - mirror).max())
    l2_error = float(np.linalg.norm(profile - mirror) / np.linalg.norm(mirror))
    print(
        f"VAL-013: {steps} steps of {dt:.3e} s ({steps * dt:.2f} s simulated, cap "
        f"{case.t_end} s) in {seconds:.1f} s; last change per step {change:.2e} of the "
        f"inlet maximum {HIGH:.1f} (steady below {STEADY_FRACTION}: {steps < cap}; the "
        f"cap is the other stop ADR-011 H allows)"
    )
    print(
        f"  minimum over the run {worst_low:.15f} (bound {LOW:.15f}); maximum "
        f"{worst_high:.15f} (bound {HIGH:.15f})"
    )
    print(
        f"  outlet profile against the inlet's mirror image, unscored: largest "
        f"difference {front_error:.4f} of a range of {HIGH - LOW:.4f}, relative L2 "
        f"{l2_error:.4f}"
    )
    assert worst_low >= LOW - ROUNDING
    assert worst_high <= HIGH + ROUNDING
    # The budget: inflow minus outflow is what the domain holds above its start.
    assert abs(solver.budget[0].relative()) < 1e-12


@pytest.mark.validation
def test_the_unlimited_quick_face_value_breaks_the_bounds_val013(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The planted control of ADR-011 H: with the clamp removed the front
    overshoots the inlet maximum or undershoots its minimum within a few
    hundred steps, so the bounds test can fail."""
    case = smith_hutton_case()
    monkeypatch.setattr(
        scalar_scheme,
        "limited_face_values",
        lambda c_up, c_c, c_d, quick: quick,
    )
    solver = TransportSolver(case.mesh, case.config, case.physics, case.conditions)
    dt = solver.stable_dt(case.faces, 0)
    c = case.initial.copy()
    broken_at = None
    for step in range(1, 501):
        c = solver.solve_timestep(c, case.faces, 0, dt)
        if c.min() < LOW - ROUNDING or c.max() > HIGH + ROUNDING:
            broken_at = step
            break
    print(
        f"VAL-013 control, unlimited QUICK: bounds broken at step {broken_at}; "
        f"minimum {c.min():.3e}, maximum {c.max():.6f}"
    )
    assert broken_at is not None


@pytest.mark.validation
def test_smith_hutton_bounds_hold_at_every_step_with_an_eddy_viscosity_field_val013() -> (
    None
):
    """VAL-013 with a field, ECR-002 criterion 5: every cell within the inlet's bounds.

    The test above with nu_t handed to every step: a non-uniform field from
    ``prescribed_eddy_viscosity`` (cell Peclet numbers 8 to 250 on
    ``nu_t / 0.7`` at the case's 0.45 m/s, a band of zero columns), so the
    front now diffuses as well as advects. The implicit matrix is an M-matrix
    for any non-negative conductance, so the bounds are predicted to hold at
    every step as they do without the field. Every implicit solve must
    converge; a capped one is a stop (prompt 40). Measured on 2026-10-07: the
    bounds hold to the same rounding allowance, and the budget closes to
    5.7e-12, the 1e-13 implicit tolerance summed over the run.
    """
    case = smith_hutton_case(TURBULENT_SCHMIDT)
    nu_t = prescribed_eddy_viscosity(case.mesh, SUPPLY_SPEED)
    solver = TransportSolver(case.mesh, case.config, case.physics, case.conditions)
    dt = solver.stable_dt(case.faces, 0)
    cap = math.ceil(case.t_end / dt)
    c = case.initial.copy()
    worst_low, worst_high = float(c.min()), float(c.max())
    sweeps: list[int] = []
    start = perf_counter()
    steps, change = 0, float("inf")
    for step in range(1, cap + 1):
        c_new = solver.solve_timestep(c, case.faces, 0, dt, eddy_viscosity=nu_t)
        assert solver.diffusion_converged, f"step {step}: implicit solve capped"
        sweeps.append(solver.last_diffusion_sweeps)
        low, high = float(c_new.min()), float(c_new.max())
        assert low >= LOW - ROUNDING, f"step {step}: {low} below {LOW}"
        assert high <= HIGH + ROUNDING, f"step {step}: {high} above {HIGH}"
        worst_low, worst_high = min(worst_low, low), max(worst_high, high)
        change = float(np.abs(c_new - c).max())
        c = c_new
        steps = step
        if change < STEADY_FRACTION * HIGH:
            break
    seconds = perf_counter() - start

    print(
        f"VAL-013 with a field: {steps} steps of {dt:.3e} s ({steps * dt:.2f} s "
        f"simulated, cap {case.t_end} s) in {seconds:.1f} s; last change per step "
        f"{change:.2e} (steady below {STEADY_FRACTION}: {steps < cap})"
    )
    print(
        f"  minimum over the run {worst_low:.15f} (bound {LOW:.15f}); maximum "
        f"{worst_high:.15f} (bound {HIGH:.15f})"
    )
    print(
        f"  implicit sweeps per step with the field: max {max(sweeps)}, mean "
        f"{sum(sweeps) / len(sweeps):.2f}, all {len(sweeps)} solves converged"
    )
    assert worst_low >= LOW - ROUNDING
    assert worst_high <= HIGH + ROUNDING
    assert max(sweeps) > 0
    # The budget closes to the implicit solve's tolerance, not to rounding:
    # the first test's 1e-12 holds because it never iterates. Measured on
    # 3,000 steps at tolerances 1e-11, 1e-13 and 1e-15: 6e-10, 4e-12 and
    # 2e-15 (results/builder40/probe013.py). This run is at 1e-13.
    assert abs(solver.budget[0].relative()) < 1e-10
