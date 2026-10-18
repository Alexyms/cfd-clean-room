"""VAL-022, the square cylinder (ADR-014 F, step 6b): the start-up and the rows' criteria.

Sohankar, Norberg and Davidson (1998) at 5% blockage: the Strouhal number at
Re 100 is 0.146 on every grid of their refinement study (their Table IV),
0.145 to 0.150 across their outlet treatments (Table II), the drag coefficient
1.45 to 1.49, and the onset of shedding Re_cr = 51.2 +/- 1 (Table V). Alex set
the thresholds on 2026-10-17 (ADR-014 F): at Re 100 the Strouhal number within
5% of 0.146 on the clustered grid, 0.139 to 0.153; the onset bracket as
stated, Re 40 steady and Re 60 shedding; the drag coefficient reported, not
scored. The uniform 0.1 d rung's Strouhal number is reported.

The start-up (``perturbed_startup``) is the probe's (docs/reports/probe59/
cylinder59.py, ``startup`` with ``--perturb``): STARTUP_OUTER steady outer
iterations from rest with the lower lateral boundary at STARTUP_BOTTOM_FACTOR
of U, an asymmetry for the shedding instability to grow from, the march
itself symmetric. Every row starts from it on its own grid and viscosity.

The measures, stated before the runs (prompt 68). Each is read from the lift
coefficient's per-period amplitudes (``amplitudes``): half its range between
two successive upward zero crossings, over the last AMPLITUDE_PERIODS whole
periods.

- Re 40, steady: the oscillation decays, each of the last five amplitudes
  below the one before. A lift that no longer completes six periods in the
  run's second half reads steady when its largest magnitude over the run's
  last tenth is below QUIET_LIFT, a level no shedding lift comes near.
- Re 60, shedding: a sustained periodic lift, each of the last five
  amplitudes at least SUSTAIN_RATIO of the one before (growing or saturated;
  the margin admits the last percent of a saturating signal's wobble).
- Re 100: the Strouhal number of the last ten periods (``forces.strouhal``)
  inside STROUHAL_BAND, the drag coefficient over the same window reported
  beside SOHANKAR_DRAG.
"""

import copy
from collections.abc import Callable
from itertools import pairwise

import numpy as np

from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.mesh import Mesh
from src.solver_staggered import FlowState, StaggeredSolver
from src.stopping import IterationState

# The probe's start-up (cylinder59.py --perturb): steady outer iterations from
# rest with the lower lateral boundary slowed.
STARTUP_OUTER = 200
STARTUP_BOTTOM_FACTOR = 0.9
STARTUP_BOUNDARY = "bottom"
# Sohankar et al. (1998): St 0.146 at Re 100, Table IV; C_D 1.45 to 1.49.
SOHANKAR_STROUHAL = 0.146
SOHANKAR_DRAG = (1.45, 1.49)
# Alex, 2026-10-17 (ADR-014 F): St within 5% of 0.146.
STROUHAL_BAND = (0.139, 0.153)
# Each VAL-022 preset's criterion: the decay (Re 40, below the onset), the
# sustain (Re 60, above it), the Strouhal band (Re 100 on the clustered grid)
# or a report (the uniform rung).
DECAY, SUSTAIN, BAND, REPORT = "decay", "sustain", "band", "report"
ROW_CRITERIA: dict[str, str] = {
    "val022_re40": DECAY,
    "val022_re60": SUSTAIN,
    "val022_re100": BAND,
    "val022_re100_uniform": REPORT,
}
# The periods the decay and sustain measures read.
AMPLITUDE_PERIODS = 5
# Each period's amplitude against the one before: a shedding lift's at least
# this, so growth and saturation pass and a decay of more than 1% does not.
SUSTAIN_RATIO = 0.99
# A lift that has stopped oscillating reads steady below this magnitude: the
# probe's shedding lift at Re 100 is of order 0.1 to 1, and rounding of order
# 1e-15.
QUIET_LIFT = 1e-6
# The fraction of the run at its end the quiet measure reads.
QUIET_FRACTION = 0.1


def perturbed_startup(
    config: SimConfig,
    on_iteration: Callable[[IterationState], None] | None = None,
) -> tuple[FlowState, int]:
    """The probe's perturbed start-up for a VAL-022 configuration.

    Parameters
    ----------
    config : SimConfig
        A row's configuration (its grid and viscosity are the start-up's).
    on_iteration : Callable[[IterationState], None], optional
        Handed to ``solve_steady``, for the harness's work count.

    Returns
    -------
    tuple[FlowState, int]
        The state after STARTUP_OUTER steady outer iterations from rest with
        the lower lateral boundary at STARTUP_BOTTOM_FACTOR of its velocity,
        and the outer iterations taken.

    Raises
    ------
    ValueError
        If the configuration has no lower boundary segment of that name.
    """
    raw = copy.deepcopy(config.merged_mapping())
    if STARTUP_BOUNDARY not in raw["boundaries"]:
        raise ValueError(
            f"the start-up slows the {STARTUP_BOUNDARY!r} segment and the "
            "configuration has none"
        )
    raw["boundaries"][STARTUP_BOUNDARY]["u_velocity"] *= STARTUP_BOTTOM_FACTOR
    raw["solver"]["max_simple_iter"] = STARTUP_OUTER
    raw["solver"]["convergence_tol"] = 1e-12
    raw["solver"]["stopping_rule"] = "velocity_step"
    raw.pop("transient", None)
    startup = SimConfig.from_dict(raw)
    mesh = Mesh(startup)
    solver = StaggeredSolver(mesh, startup, StaggeredBoundary(mesh, startup))
    solver.solve_steady(on_iteration)
    assert solver.flow_state is not None
    state = solver.flow_state
    return FlowState(u=state.u, v=state.v, p=state.p), len(solver.residual_history)


def amplitudes(t: np.ndarray, c_l: np.ndarray) -> np.ndarray:
    """Half the lift's range over each whole period between upward zero crossings.

    Parameters
    ----------
    t : np.ndarray
        Times, s, increasing.
    c_l : np.ndarray
        The lift coefficient there.

    Returns
    -------
    np.ndarray
        One amplitude per whole period, in time order; empty with fewer than
        two upward crossings.
    """
    c_l = np.asarray(c_l, dtype=np.float64)
    up = np.nonzero((c_l[:-1] < 0.0) & (c_l[1:] >= 0.0))[0]
    return np.array(
        [
            0.5 * (c_l[a + 1 : b + 1].max() - c_l[a + 1 : b + 1].min())
            for a, b in pairwise(up)
        ]
    )


def decays(t: np.ndarray, c_l: np.ndarray) -> dict[str, object]:
    """The Re 40 row: does the lift's oscillation decay? See the module's measures.

    Returns
    -------
    dict[str, object]
        ``passed``, the ``measure`` read ("amplitudes" or "quiet"), the last
        amplitudes or the quiet magnitude, and their ratios.
    """
    t = np.asarray(t, dtype=np.float64)
    late = t >= 0.5 * t[-1]
    recent = amplitudes(t[late], np.asarray(c_l)[late])
    if recent.size >= AMPLITUDE_PERIODS + 1:
        last = recent[-(AMPLITUDE_PERIODS + 1) :]
        ratios = last[1:] / last[:-1]
        return {
            "passed": bool(np.all(ratios < 1.0)),
            "measure": "amplitudes",
            "amplitudes": last.tolist(),
            "ratios": ratios.tolist(),
        }
    tail = t >= (1.0 - QUIET_FRACTION) * t[-1]
    quiet = float(np.max(np.abs(np.asarray(c_l)[tail])))
    return {"passed": quiet < QUIET_LIFT, "measure": "quiet", "largest_lift": quiet}


def sustains(t: np.ndarray, c_l: np.ndarray) -> dict[str, object]:
    """The Re 60 row: is the lift a sustained oscillation? See the module's measures.

    Returns
    -------
    dict[str, object]
        ``passed``, the last amplitudes and their ratios; ``passed`` False
        with fewer than six whole periods.
    """
    every = amplitudes(np.asarray(t), np.asarray(c_l))
    if every.size < AMPLITUDE_PERIODS + 1:
        return {"passed": False, "measure": "amplitudes", "periods": int(every.size)}
    last = every[-(AMPLITUDE_PERIODS + 1) :]
    ratios = last[1:] / last[:-1]
    return {
        "passed": bool(np.all(ratios >= SUSTAIN_RATIO)),
        "measure": "amplitudes",
        "amplitudes": last.tolist(),
        "ratios": ratios.tolist(),
    }


def strouhal_in_band(st: float | None) -> bool:
    """The Re 100 row's criterion: St inside STROUHAL_BAND, inclusive."""
    return st is not None and STROUHAL_BAND[0] <= st <= STROUHAL_BAND[1]
