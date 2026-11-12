"""Split T5, the implicit solve's leftover, into its diffusion and sink parts (prompt 77).

    python docs/reports/probe77/split77.py RUN

A diagnostic beside probe77.py, not one of the prompt's terms. The Jacobi
iteration of ``scalar_scheme._jacobi`` starts from ``C*`` and adds
``r_k / a_P`` each sweep, so its residual obeys ``r_{k+1} = N (r_k / a_P)``,
``N`` the neighbour sum ``sum_f G_f x_N``: after K sweeps ``r_K = M^K r_0``
with ``M = N diag(1 / a_P)``, linear in ``r_0``. The first residual is
``r_0 = (N C* - sum_f G_f C*) - D_P C*``, a diffusion part whose sum over the
cells is zero (each face's two terms cancel) and the sink at ``C*``. Carrying
the two parts apart through the solve's own K sweeps splits T5 = dt sum r_K
into T5_diffusion and T5_sink, whose sum is T5 up to rounding.

Each sweep moves the sum by ``sum_P r_P (sum_f G_f)_P / a_P - sum_P r_P``:
the sum of r after a sweep is the sum of r weighted by the local fraction
``w_P = (sum_f G_f)_P / a_P``. A zero-sum diffusion residual keeps a zero sum
only where w is uniform; w varies with nu_t and is smaller in a cell with a
wall face, so the diffusion part's leftover has either sign, while the sink
part is never positive (M has no negative entry and r_0's sink part none
either).

The run goes through scripts/run_scenario.py's ``main`` with configs77's
configuration and flow source, to results/builder77/split/RUN.h5, with one
name wrapped: ``implicit_step`` as ``src.solver_transport`` sees it, observing
only. Per step and class: the sweeps, T5 (from the returned field, as
probe77.py forms it), T5_diffusion, T5_sink, and ``dt sum_P D_P C*_P``, the
deposition the step's sink would take at ``C*``. Written to
results/builder77/split/RUN.split.npz.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import configs77  # noqa: E402 -- follows sys.path.insert
import probe77  # noqa: E402 -- follows sys.path.insert

import src.solver_transport as solver_transport  # noqa: E402 -- follows sys.path.insert
from src.scalar_scheme import _neighbour_sum  # noqa: E402 -- follows sys.path.insert

OUT = probe77.RESULTS / "split"
COLUMNS = ("sweeps", "T5", "T5_diffusion", "T5_sink", "sink_at_cstar")
ROWS: list[list[float]] = []
_ORIGINAL = solver_transport.implicit_step


def _propagate(
    r: np.ndarray,
    a_p: np.ndarray,
    g_u: np.ndarray,
    g_v: np.ndarray,
    solid: np.ndarray,
    sweeps: int,
) -> np.ndarray:
    """``M^sweeps r``, with M = N diag(1 / a_P) and SOLID cells held at zero."""
    for _ in range(sweeps):
        r = _neighbour_sum(np.pad(r / a_p, 1), g_u, g_v)
        r[solid] = 0.0
    return r


def implicit_step_split(*args: object, **kwargs: object) -> object:
    """``implicit_step``, with its leftover split into the diffusion and sink parts."""
    result = _ORIGINAL(*args, **kwargs)
    c_star, volume, dt, g_u, g_v, diagonal, solid = args[:7]
    over_dt = volume / dt
    sum_g = g_u[:, :-1] + g_u[:, 1:] + g_v[:-1, :] + g_v[1:, :]
    a_p = over_dt + (sum_g + diagonal)
    c_new = result.field
    residual = (
        over_dt * c_star + _neighbour_sum(np.pad(c_new, 1), g_u, g_v) - a_p * c_new
    )
    residual[solid] = 0.0
    diffusion = _neighbour_sum(np.pad(c_star, 1), g_u, g_v) - sum_g * c_star
    sink = -diagonal * c_star
    diffusion[solid] = 0.0
    sink[solid] = 0.0
    k = result.iterations
    live = ~solid
    ROWS.append(
        [
            float(k),
            dt * float(np.sum(residual[live])),
            dt * float(np.sum(_propagate(diffusion, a_p, g_u, g_v, solid, k)[live])),
            dt * float(np.sum(_propagate(sink, a_p, g_u, g_v, solid, k)[live])),
            dt * float(np.sum((diagonal * c_star)[live])),
        ]
    )
    return result


def main() -> int:
    """Run one configuration with the split observer."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", choices=sorted(configs77.BY_NAME))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    spec = configs77.BY_NAME[args.run]
    runner = probe77.load_runner()
    solver_transport.implicit_step = implicit_step_split
    started = time.perf_counter()
    try:
        status = runner.main(
            [
                str(configs77.OUT / f"scenario_{args.run}.yaml"),
                "--out",
                str(OUT / f"{args.run}.h5"),
                "--flows-from",
                str(spec.flows_from),
                "--quiet",
            ]
        )
    finally:
        solver_transport.implicit_step = _ORIGINAL
    rows = np.array(ROWS, dtype=np.float64)
    n_classes = 5
    np.savez(
        OUT / f"{args.run}.split.npz",
        columns=np.array(COLUMNS),
        **{f"class_{k}": rows[k::n_classes] for k in range(n_classes)},
    )
    record = {
        "run": args.run,
        "exit": status,
        "wall_seconds": time.perf_counter() - started,
    }
    (OUT / f"{args.run}.split.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(json.dumps(record))
    return status


if __name__ == "__main__":
    sys.exit(main())
