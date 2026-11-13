"""The runs of prompt 78: five scenarios on 80x30, on main and on the branch.

    python docs/reports/probe78/configs78.py    # write the scenario files

R1 to R4 are prompt 77's: the dust release, the clean-out, the door leak and
the filter breach at the scenario files' inputs on 80x30, each on the flow
state of prompt 64's run of it (``results/builder64/coarse/``), with
``solver.mass_imbalance_tol`` the value that run recorded (Alex's decision of
2026-11-12 on prompt 77's departure). SL is the seal leak, which has no 80x30
run before this pass: its closed-door state is the door leak's kept state and
its leaking state is solved, warm from it, on each side, so the two sides'
solves of that state are compared bitwise.

Each run is made three times. ``main`` runs origin/main's code from a
worktree (``MAIN_TREE``) on main's own base file, the run before this change;
``branch`` runs the working tree's code, the change; ``probe`` runs the
working tree's code with ``probe78.py``'s recorder, every step and every
solve, and its run file is compared with ``branch``'s for the calibration.
Each side's base is its tree's configs/clean_room_default.yaml with the grid
set to 80x30, and nothing else changed: main's has no conservation keys, the
branch's has the two of ADR-016.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
MAIN_TREE = ROOT.parent / "cfd_main78"
RESULTS = ROOT / "results" / "builder78"
OUT = RESULTS / "configs"
GRID = (80, 30)
CHECKPOINT_INTERVAL = 1000
SIDES: tuple[str, ...] = ("main", "branch", "probe")
SOURCES = ROOT / "results" / "builder64" / "coarse"


@dataclass(frozen=True)
class Run:
    """One scenario of the pass.

    Parameters
    ----------
    name : str
        R1 to R4 (prompt 77's names) or SL.
    scenario : str
        The reference scenario's stem, ``scenario_<scenario>.yaml``.
    flows_from : Path
        The run file whose flow states the run reuses.
    before : Path or None
        The latest 80x30 run of the scenario before this change, whose
        latencies prediction (e) is read against; None where the pass makes
        it (SL, whose ``main`` run is that run).
    """

    name: str
    scenario: str
    flows_from: Path
    before: Path | None


RUNS: tuple[Run, ...] = (
    Run(
        "R1",
        "equipment_dust",
        SOURCES / "equipment_dust.h5",
        ROOT / "results/builder77/R1.h5",
    ),
    Run("R2", "clean_out", SOURCES / "clean_out.h5", ROOT / "results/builder77/R2.h5"),
    Run("R3", "door_leak", SOURCES / "door_leak.h5", ROOT / "results/builder77/R3.h5"),
    Run(
        "R4",
        "filter_breach",
        SOURCES / "filter_breach.h5",
        ROOT / "results/builder77/R4.h5",
    ),
    Run("SL", "seal_leak", SOURCES / "door_leak.h5", None),
)
BY_NAME: dict[str, Run] = {run.name: run for run in RUNS}


def tree(side: str) -> Path:
    """The tree whose code and base file a side runs."""
    return MAIN_TREE if side == "main" else ROOT


def run_path(name: str, side: str) -> Path:
    """The run file of one run on one side."""
    return RESULTS / side / f"{name}.h5"


def source_tolerance(path: Path) -> float:
    """The solver.mass_imbalance_tol the source run file was run with."""
    sys.path.insert(0, str(ROOT))
    from src.io_manager import RunFile

    return float(RunFile(path).read_config().mapping["solver"]["mass_imbalance_tol"])


def write_base(side: str) -> Path:
    """Write a side's base: its tree's base file on 80x30."""
    base = yaml.safe_load(
        (tree(side) / "configs" / "clean_room_default.yaml").read_text(encoding="utf-8")
    )
    base["domain"]["nx"], base["domain"]["ny"] = GRID
    path = OUT / f"base_{'main' if side == 'main' else 'branch'}.yaml"
    path.write_text(
        yaml.safe_dump(base, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return path


def scenario_path(name: str, side: str) -> Path:
    """The scenario file one run on one side reads."""
    return OUT / f"scenario_{name}_{'main' if side == 'main' else 'branch'}.yaml"


def write_scenario(run: Run, side: str, base: Path) -> Path:
    """Write one run's scenario file on a side's base."""
    scenario = yaml.safe_load(
        (tree(side) / "configs" / f"scenario_{run.scenario}.yaml").read_text(
            encoding="utf-8"
        )
    )
    scenario["scenario"]["base"] = str(base.resolve())
    scenario["solver"]["mass_imbalance_tol"] = source_tolerance(run.flows_from)
    scenario["solver"]["checkpoint_interval"] = CHECKPOINT_INTERVAL
    target = scenario_path(run.name, side)
    target.write_text(
        yaml.safe_dump(scenario, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return target


def main() -> None:
    """Write both sides' bases and scenario files."""
    OUT.mkdir(parents=True, exist_ok=True)
    for side in ("main", "branch"):
        base = write_base(side)
        for run in RUNS:
            print(write_scenario(run, side, base))


if __name__ == "__main__":
    main()
