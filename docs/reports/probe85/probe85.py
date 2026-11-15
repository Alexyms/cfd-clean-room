"""Prompt 85's probe: how far the reference's own results move under a correct port's changes.

    python docs/reports/probe85/probe85.py configs
    python docs/reports/probe85/probe85.py run <variant> <scenario> <grid>
    python docs/reports/probe85/probe85.py list

The reference is main's code and main's scenario files, on 80x30 and 160x60.
A variant is one change a correct port could legitimately make, and nothing
else (``VARIANTS``):

- ``pressure_rtol`` or ``diffusion_tol`` times 0.99 or 1.01: a port whose
  solves stop at a slightly different point;
- every velocity inlet's velocity times 1 + 1e-15: a port that forms the
  boundary data in another order, a few units in the last place;
- ``leftover_sequential``: the implicit transport solve's leftover (ADR-016),
  the one sum its stop reads, taken left to right over the cells in row-major
  order, as a plain C++ loop takes it, in place of NumPy's pairwise sum. The
  stop is ADR-016's, the max-norm test then the leftover against its share;
  only the order of that sum's additions changes. The patch counts the sums
  whose two orders differ and the stop tests whose verdicts would differ;
- ``outer_one_later``, an addition to the prompt's list: each flow solve runs
  one more outer iteration after the first at which its stopping rule says
  stop, and then stops at the next at which it does. A port's rule reads sums
  over the cells (conditions (c) and (d)) and can stop one iteration apart; no
  other variant moves the outer stop. The patch wraps
  ``ErrorEstimateRule.update`` in this process only.

``reference`` is the files as they are, and ``repeat`` the same again in
another process, to show that the reference reproduces itself bit for bit.

Each run is ``time_integration.run`` from rest on one BLAS thread, written to
``results/builder85/runs/<variant>/<scenario>_<grid>.h5``, with a summary of
its record beside it (``.json``). Nothing under ``src/`` changes; the
reduction and the outer-stop variants replace methods in their own process only.

The grids' ``mass_imbalance_tol`` is the base file's rule (``conservation_limit``
times rho times the smallest cell's area over 60 s), as prompts 75, 79 and 81
wrote it for 80x30 (``docs/reports/probe79/configs79.py``, ``PLACEMENT_TOL``).
The door leak runs on 80x30 only: its open-door state does not converge on
160x60 at the file's inputs (prompt 72; prompt 79, F2).
"""

import dataclasses
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np  # noqa: E402 -- the thread variables are set first
import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from src import scalar_scheme, time_integration  # noqa: E402
from src.config import SimConfig  # noqa: E402
from src.stopping import ErrorEstimateRule  # noqa: E402
from src.time_integration import Provenance  # noqa: E402

RESULTS = ROOT / "results" / "builder85"
CONFIGS = RESULTS / "configs"
RUNS = RESULTS / "runs"

# Scenario short name to main's file.
SCENARIOS = {
    "dust": "scenario_equipment_dust.yaml",
    "door": "scenario_door_leak.yaml",
    "breach": "scenario_filter_breach.yaml",
}
# Cells per direction; the room is 8 m x 3 m (ROOM).
GRIDS = {"80x30": (80, 30), "160x60": (160, 60)}
# The scenarios each grid runs (the module docstring says why not the door on 160x60).
GRID_SCENARIOS = {"80x30": ("dust", "door", "breach"), "160x60": ("dust", "breach")}
# The base file's rule for mass_imbalance_tol, as run75.py wrote it for 80x30:
# conservation_limit x rho x V_min / 60 s.
RULE_LIMIT, RULE_RHO, RULE_SECONDS = 1.0e-4, 1.2, 60.0
ROOM = (8.0, 3.0)
INFLOW_FACTOR = 1.0 + 1.0e-15
# Variant name to (section, key, factor) for a scaled key, or a tag.
VARIANTS: dict[str, tuple[str, str, float] | str | None] = {
    "reference": None,
    "repeat": None,
    "pressure_rtol_0.99": ("solver", "pressure_rtol", 0.99),
    "pressure_rtol_1.01": ("solver", "pressure_rtol", 1.01),
    "diffusion_tol_0.99": ("transport", "diffusion_tol", 0.99),
    "diffusion_tol_1.01": ("transport", "diffusion_tol", 1.01),
    "inflow_1e-15": "inflow",
    "leftover_sequential": "sequential",
    "outer_one_later": "outer_later",
}


def mass_imbalance_tol(grid: str) -> float:
    """The base file's rule for GRID, in run75.py's order of operations."""
    nx, ny = GRIDS[grid]
    cell = (ROOM[0] / nx) * (ROOM[1] / ny)
    return RULE_LIMIT * RULE_RHO * cell / RULE_SECONDS


def config_path(variant: str, scenario: str, grid: str) -> Path:
    """The scenario file of one run."""
    return CONFIGS / variant / f"{scenario}_{grid}.yaml"


def run_path(variant: str, scenario: str, grid: str) -> Path:
    """The run file of one run."""
    return RUNS / variant / f"{scenario}_{grid}.h5"


def _scale_inlets(segments: dict | list, factor: float, scaled: list) -> None:
    """Scale every velocity inlet's velocity in SEGMENTS, in place, noting each."""
    items = segments.items() if isinstance(segments, dict) else enumerate(segments)
    for key, spec in items:
        if spec.get("type") == "velocity_inlet" and "velocity" in spec:
            old = float(spec["velocity"])
            spec["velocity"] = old * factor
            scaled.append((spec.get("name", key), old, spec["velocity"]))


def variant_mapping(variant: str, scenario: str, grid: str) -> tuple[dict, list]:
    """Main's scenario file with the grid, its tolerance and VARIANT's change.

    Returns the mapping and a list of what was changed, for the summary.
    """
    source = ROOT / "configs" / SCENARIOS[scenario]
    mapping = yaml.safe_load(source.read_text(encoding="utf-8"))
    base_path = (source.parent / mapping["scenario"]["base"]).resolve()
    mapping["scenario"]["base"] = str(base_path)
    nx, ny = GRIDS[grid]
    mapping["domain"] = {"nx": nx, "ny": ny}
    mapping.setdefault("solver", {})["mass_imbalance_tol"] = mass_imbalance_tol(grid)
    changed: list = []
    change = VARIANTS[variant]
    if isinstance(change, tuple):
        section, key, factor = change
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        old = float(mapping.get(section, {}).get(key, base[section][key]))
        mapping.setdefault(section, {})[key] = old * factor
        changed.append((f"{section}.{key}", old, old * factor))
    elif change == "inflow":
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        boundaries = json.loads(json.dumps(base["boundaries"]))
        _scale_inlets(boundaries, INFLOW_FACTOR, changed)
        mapping["boundaries"] = {
            name: {"velocity": spec["velocity"]}
            for name, spec in boundaries.items()
            if spec.get("type") == "velocity_inlet" and "velocity" in spec
        }
        for event in mapping["scenario"].get("events", []):
            if "replace" in event:
                _scale_inlets(event["replace"]["with"], INFLOW_FACTOR, changed)
    return mapping, changed


def write_configs() -> None:
    """Every run's scenario file, each checked by loading it."""
    for variant in VARIANTS:
        for grid, scenarios in GRID_SCENARIOS.items():
            for scenario in scenarios:
                mapping, changed = variant_mapping(variant, scenario, grid)
                path = config_path(variant, scenario, grid)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8"
                )
                path.with_suffix(".changed.json").write_text(json.dumps(changed))
                SimConfig(path)  # refuses an invalid file now


class SequentialLeftover:
    """The leftover's sum taken left to right in row-major order, and its counts.

    Replaces ``scalar_scheme._Acceptance._form`` and ``test`` in this process.
    Each formed leftover is also summed NumPy's way, so the record says how
    many sums the order changed and how many stop verdicts it would flip.
    """

    def __init__(self) -> None:
        self.formed = 0
        self.differ = 0
        self.largest_relative = 0.0
        self.tests = 0
        self.flipped = 0

    def install(self) -> None:
        """Replace the two methods."""
        probe = self

        def form(acceptance: object, residual: np.ndarray) -> float:
            if acceptance._scalar:
                pairwise = acceptance._dt * float(np.sum(residual))
                leftover = acceptance._dt * float(np.cumsum(residual)[-1])
            else:
                terms = acceptance._dt * residual
                pairwise = float(np.sum(terms))
                leftover = float(np.cumsum(terms)[-1])
            probe.formed += 1
            if leftover != pairwise:
                probe.differ += 1
                scale = max(abs(pairwise), abs(leftover))
                probe.largest_relative = max(
                    probe.largest_relative, abs(leftover - pairwise) / scale
                )
            acceptance._pairwise = pairwise
            acceptance._formed = residual
            acceptance.leftover = leftover
            return leftover

        def test(acceptance: object, residual: np.ndarray) -> bool:
            if acceptance._bound is None:
                return True
            leftover = form(acceptance, residual)
            probe.tests += 1
            verdict = abs(leftover) <= acceptance._bound
            if verdict != (abs(acceptance._pairwise) <= acceptance._bound):
                probe.flipped += 1
            return verdict

        scalar_scheme._Acceptance._form = form
        scalar_scheme._Acceptance.test = test

    def record(self) -> dict:
        """The counts, for the summary."""
        return {
            "formed": self.formed,
            "differ": self.differ,
            "largest_relative": self.largest_relative,
            "tests": self.tests,
            "flipped": self.flipped,
        }


class OuterOneLater:
    """Defer each stopping rule's first stop by one outer iteration, and count it.

    Replaces ``ErrorEstimateRule.update`` in this process: the first time a
    rule instance (one per flow solve) answers True the answer is turned to
    False once; every later answer is the rule's own. The rule records the
    iteration as it always does, so its window holds the extra step.
    """

    def __init__(self) -> None:
        self.deferred = 0
        self.original = ErrorEstimateRule.update

    def install(self) -> None:
        """Replace the method."""
        probe, original = self, self.original

        def update(rule: ErrorEstimateRule, *args: object, **kwargs: object) -> bool:
            stop = original(rule, *args, **kwargs)
            if stop and not getattr(rule, "_probe85_deferred", False):
                rule._probe85_deferred = True
                probe.deferred += 1
                return False
            return stop

        ErrorEstimateRule.update = update

    def record(self) -> dict:
        """The count, for the summary."""
        return {"deferred": self.deferred}


def commit() -> str:
    """HEAD's hash, or "unknown"."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _plain(value: object) -> object:
    """VALUE as JSON can hold it."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _walk(value: object) -> object:
    """Convert nested containers for JSON, inf and NaN as text."""
    if isinstance(value, dict):
        return {str(k): _walk(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_walk(v) for v in value]
    value = _plain(value)
    if isinstance(value, list):
        return [_walk(v) for v in value]
    return value


def run(variant: str, scenario: str, grid: str) -> None:
    """One run from rest, and its summary beside the run file."""
    from threadpoolctl import threadpool_info

    out = run_path(variant, scenario, grid)
    out.parent.mkdir(parents=True, exist_ok=True)
    for path in (out, out.with_name(out.stem + ".checkpoint.h5")):
        if path.exists():
            path.unlink()
    patch = later = None
    if VARIANTS[variant] == "sequential":
        patch = SequentialLeftover()
        patch.install()
    if VARIANTS[variant] == "outer_later":
        later = OuterOneLater()
        later.install()
    config = SimConfig(config_path(variant, scenario, grid))
    started = time.perf_counter()
    record = time_integration.run(config, out, provenance=Provenance.here(commit(), 0))
    wall = time.perf_counter() - started
    summary = {
        "variant": variant,
        "scenario": scenario,
        "grid": grid,
        "changed": json.loads(
            config_path(variant, scenario, grid)
            .with_suffix(".changed.json")
            .read_text()
        ),
        "wall_seconds": wall,
        "commit": commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "blas": [
            {k: d.get(k) for k in ("internal_api", "num_threads", "version")}
            for d in threadpool_info()
        ],
        "steps": record.steps,
        "t_final": record.t_final,
        "flows": [dataclasses.asdict(f) for f in record.flows],
        "intervals": [dataclasses.asdict(p) for p in record.intervals],
        "budgets": [dataclasses.asdict(b) for b in record.budgets],
        "latencies": record.latencies,
        "map_latencies": record.map_latencies,
        "alerts": [dataclasses.asdict(a) for a in record.alerts],
        "sequential": None if patch is None else patch.record(),
        "outer_later": None if later is None else later.record(),
    }
    out.with_suffix(".json").write_text(json.dumps(_walk(summary), indent=1))


def jobs() -> list[tuple[str, str, str]]:
    """Every (variant, scenario, grid), the 160x60 runs first."""
    out = []
    for grid in ("160x60", "80x30"):
        for scenario in GRID_SCENARIOS[grid]:
            for variant in VARIANTS:
                out.append((variant, scenario, grid))
    return out


def done(variant: str, scenario: str, grid: str) -> bool:
    """Whether the run's summary exists."""
    return run_path(variant, scenario, grid).with_suffix(".json").exists()


def main(argv: list[str] | None = None) -> int:
    """The command line."""
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print(__doc__)
        return 2
    if args[0] == "configs":
        write_configs()
    elif args[0] == "run":
        run(args[1], args[2], args[3])
    elif args[0] == "list":
        for job in jobs():
            print(*job, "done" if done(*job) else "")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
