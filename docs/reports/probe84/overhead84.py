"""The timers' own cost per event, measured on functions that do nothing.

    python docs/reports/probe84/overhead84.py

The calibration's wall difference between wrapped and unwrapped runs is
within the spread of identical unwrapped runs on this machine, so it bounds
the overhead only loosely. This measures the two kinds of event the timers
add, a wrapped call (two clock readings, the path push and pop, the call
count) and a region marker (one clock reading and a relabel), each as the
median over repeated batches of the difference from the same work
unwrapped. ``tables84.py`` multiplies them by each run's event counts.
Writes ``results/builder84/overhead.json``.
"""

import functools
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter_ns

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import probe84  # noqa: E402 -- follows sys.path.insert

N = 200_000
BATCHES = 15


class Empty:
    """Two functions with nothing to do, one with region markers."""

    def nothing(self) -> None:
        return None

    def lines(self, n: int) -> int:
        total = 0
        for _ in range(n):
            total += 1
            total -= 1
        return total


SPECS = [probe84.Wrap(Empty, "nothing", "nothing")]
REGIONS = {"lines": [("total += 1", "a"), ("total -= 1", "b")]}


def batch_calls(obj: Empty) -> int:
    started = perf_counter_ns()
    for _ in range(N):
        obj.nothing()
    return perf_counter_ns() - started


def batch_lines(obj: Empty) -> int:
    started = perf_counter_ns()
    obj.lines(N)
    return perf_counter_ns() - started


def measure() -> dict:
    obj = Empty()
    call, line = [], []
    for _ in range(BATCHES):
        plain_c, plain_l = batch_calls(obj), batch_lines(obj)
        probe84.install(SPECS, {"lines": Empty.lines}, REGIONS, {})
        wrapped_c, wrapped_l = batch_calls(obj), batch_lines(obj)
        probe84.uninstall()
        call.append((wrapped_c - plain_c) / N)
        # Two marker lines run per loop pass.
        line.append((wrapped_l - plain_l) / (2 * N))
    return {
        "wrapped_call_ns": statistics.median(call),
        "region_event_ns": statistics.median(line),
        "call_batches_ns": call,
        "line_batches_ns": line,
        "n": N,
    }


INSTALL = probe84.install
# Each part: whether the march's regions are on, and the wrapped functions
# (None: probe84.wraps() under that setting; []: none).
PARTS = {
    "full": (True, None),
    "lean": (False, None),
    "regions": (True, []),
}


def ablate(grid: str, t_stop: float, parts: list[str]) -> dict:
    """A window run under each part of the timers, to find what costs.

    ``full``: every wrapper and the march's line regions (the first
    calibration's instrument); ``lean``: the instrument the measurements run,
    the march without its regions and detail wrappers; ``regions``: the line
    regions alone, no wrapper; ``none``: no timers.
    """
    flows = probe84.RESULTS / "runs" / f"equipment_dust_{grid}.h5"
    if grid == "320x120":
        flows = probe84.STORED
    out = probe84.RESULTS / "windows"
    out.mkdir(parents=True, exist_ok=True)
    walls = []
    for n, part in enumerate(parts):
        target = out / f"ablate_{grid}_{n}.h5"
        if part == "none":
            _f, record = probe84.window_march(grid, flows, 9.0, t_stop, target, False)
        else:
            regions, specs = PARTS[part]
            probe84.MARCH_REGIONS = regions
            probe84.install = functools.partial(INSTALL, specs)
            try:
                _f, record = probe84.window_march(
                    grid, flows, 9.0, t_stop, target, True
                )
            finally:
                probe84.install = INSTALL
                probe84.MARCH_REGIONS = False
        walls.append(record["wall_ns"])
        print(part, record["wall_ns"] / 1e9, flush=True)
    return {"grid": grid, "t_stop": t_stop, "parts": parts, "wall_ns": walls}


def main() -> int:
    probe84.RESULTS.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "ablate":
        parts = sys.argv[4].split(",")
        record = ablate(sys.argv[2], float(sys.argv[3]), parts)
        name = f"ablate_{sys.argv[2]}_{sys.argv[5]}.json"
        (probe84.RESULTS / name).write_text(json.dumps(record, indent=1))
        return 0
    record = measure()
    (probe84.RESULTS / "overhead.json").write_text(json.dumps(record, indent=1))
    print(record["wrapped_call_ns"], record["region_event_ns"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
