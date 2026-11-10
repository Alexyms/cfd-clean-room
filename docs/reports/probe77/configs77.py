"""Write the residual probe's configurations (prompt 77).

Usage: python docs/reports/probe77/configs77.py  (from the repository root)

The files go to results/builder77/configs/. Each run of RUNS is one reference
scenario (configs/scenario_<name>.yaml as on main) on its base,
configs/clean_room_default.yaml, with these changes and nothing else, on the
pattern of docs/reports/probe72/configs72.py:

- the grid, on a base written for it;
- solver.mass_imbalance_tol, the value the source run (FLOWS) was run with,
  read from its run file; see below;
- solver.checkpoint_interval 1000 (a restart key, not a physics or grid key);
- transport.diffusion_tol, and for a rerun at the cap
  transport.max_diffusion_iter, where the run's arm sets them.

Every run reuses a flow state with --flows-from, so the runner compares the
flow inputs of this configuration with the source run's and refuses a
difference. ADR-011 G's rule, 1e-4 rho V_min / t_end at the scenario's own
t_end, is what configs72.py wrote; the source runs of prompts 64 and 69 used
the rule at the base's t_end of 60 s for every scenario (2e-8 on 80x30 and the
base's 1.25e-9 on 320x120), so the rule at the scenario's t_end would be
refused for every source but the door leak's on 80x30 and the 160x60 dust
release. ``refusals`` lists them; the generator takes the source's value,
which is a flow input only (the transport step does not read it), so the
flow state each run reuses is the one its source solved.

R1 to R4 are the four reference scenarios on 80x30 at the file's
diffusion_tol (1e-8): R1 the equipment dust, R2 the clean-out, R3 the door
leak, R4 the filter breach. R5 and R6 are the dust and the clean-out at 1e-6,
R7 and R8 at 1e-10; R9 the dust on 160x60, R10 on 320x120.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.io_manager import RunFile  # noqa: E402 -- follows sys.path.insert

OUT = ROOT / "results" / "builder77" / "configs"
CHECKPOINT_INTERVAL = 1000
COARSE, MEDIUM, FINE = (80, 30), (160, 60), (320, 120)
# The flow states each scenario and grid reuses (prompt 77, section 2).
FLOWS: dict[tuple[str, tuple[int, int]], Path] = {
    ("equipment_dust", COARSE): ROOT / "results/builder64/coarse/equipment_dust.h5",
    ("clean_out", COARSE): ROOT / "results/builder64/coarse/clean_out.h5",
    ("door_leak", COARSE): ROOT / "results/builder64/coarse/door_leak.h5",
    ("filter_breach", COARSE): ROOT / "results/builder64/coarse/filter_breach.h5",
    ("equipment_dust", MEDIUM): ROOT
    / "results/builder72/equipment_dust_control_160x60.h5",
    ("equipment_dust", FINE): ROOT / "results/builder69/equipment_dust.h5",
}
# The rerun's cap where a solve at 1e-10 reaches the file's 200 (prompt 77).
RERUN_SWEEPS = 2000


@dataclass(frozen=True)
class Run:
    """One run of the probe.

    Parameters
    ----------
    name : str
        R1 to R10, with a suffix for a rerun at RERUN_SWEEPS.
    scenario : str
        The reference scenario's stem, ``scenario_<scenario>.yaml``.
    grid : tuple[int, int]
        (nx, ny).
    diffusion_tol : float or None
        transport.diffusion_tol; None keeps the base's (1e-8).
    max_diffusion_iter : int or None
        transport.max_diffusion_iter; None keeps the base's (200).
    """

    name: str
    scenario: str
    grid: tuple[int, int]
    diffusion_tol: float | None = None
    max_diffusion_iter: int | None = None

    @property
    def flows_from(self) -> Path:
        """The run file whose flow states this run reuses."""
        return FLOWS[(self.scenario, self.grid)]

    @property
    def tolerance(self) -> float:
        """The diffusion_tol the run steps at."""
        return 1e-8 if self.diffusion_tol is None else self.diffusion_tol


RUNS: tuple[Run, ...] = (
    Run("R1", "equipment_dust", COARSE),
    Run("R2", "clean_out", COARSE),
    Run("R3", "door_leak", COARSE),
    Run("R4", "filter_breach", COARSE),
    Run("R5", "equipment_dust", COARSE, 1e-6),
    Run("R6", "clean_out", COARSE, 1e-6),
    Run("R7", "equipment_dust", COARSE, 1e-10),
    Run("R8", "clean_out", COARSE, 1e-10),
    Run("R9", "equipment_dust", MEDIUM),
    Run("R10", "equipment_dust", FINE),
)
# The reruns at the larger cap, written always and run only if R7 or R8
# reaches the file's cap.
RERUNS: tuple[Run, ...] = (
    Run("R7b", "equipment_dust", COARSE, 1e-10, RERUN_SWEEPS),
    Run("R8b", "clean_out", COARSE, 1e-10, RERUN_SWEEPS),
)
BY_NAME: dict[str, Run] = {run.name: run for run in RUNS + RERUNS}


def grid_tag(grid: tuple[int, int]) -> str:
    """The grid as file names spell it, "80x30"."""
    return f"{grid[0]}x{grid[1]}"


def rule_tolerance(base: dict, grid: tuple[int, int], t_end: float) -> float:
    """ADR-011 G's bound: 1e-4 rho V_min / t_end, V_min the cell area per metre."""
    v_min = (base["domain"]["width"] / grid[0]) * (base["domain"]["height"] / grid[1])
    return 1e-4 * base["fluid"]["density"] * v_min / t_end


def source_tolerance(path: Path) -> float:
    """The solver.mass_imbalance_tol the source run file was run with."""
    mapping = RunFile(path).read_config().mapping
    return float(mapping["solver"]["mass_imbalance_tol"])


def write_base(grid: tuple[int, int]) -> tuple[Path, dict]:
    """Write the room's base for a grid, the file's inputs otherwise."""
    base = yaml.safe_load(
        (ROOT / "configs" / "clean_room_default.yaml").read_text(encoding="utf-8")
    )
    base["domain"]["nx"], base["domain"]["ny"] = grid
    path = OUT / f"base_{grid_tag(grid)}.yaml"
    path.write_text(
        yaml.safe_dump(base, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return path, base


def write_scenario(run: Run, base_path: Path) -> Path:
    """Write one run's scenario file on its base."""
    scenario = yaml.safe_load(
        (ROOT / "configs" / f"scenario_{run.scenario}.yaml").read_text(encoding="utf-8")
    )
    scenario["scenario"]["base"] = str(base_path.resolve())
    scenario["solver"]["mass_imbalance_tol"] = source_tolerance(run.flows_from)
    scenario["solver"]["checkpoint_interval"] = CHECKPOINT_INTERVAL
    transport = scenario.setdefault("transport", {})
    if run.diffusion_tol is not None:
        transport["diffusion_tol"] = run.diffusion_tol
    if run.max_diffusion_iter is not None:
        transport["max_diffusion_iter"] = run.max_diffusion_iter
    target = OUT / f"scenario_{run.name}.yaml"
    target.write_text(
        yaml.safe_dump(scenario, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return target


def refusals() -> list[str]:
    """The runs whose source the rule's tolerance, at the scenario's t_end, differs from."""
    out = []
    base = yaml.safe_load(
        (ROOT / "configs" / "clean_room_default.yaml").read_text(encoding="utf-8")
    )
    for run in RUNS:
        scenario = yaml.safe_load(
            (ROOT / "configs" / f"scenario_{run.scenario}.yaml").read_text(
                encoding="utf-8"
            )
        )
        rule = rule_tolerance(base, run.grid, float(scenario["solver"]["t_end"]))
        source = source_tolerance(run.flows_from)
        if rule != source:
            out.append(
                f"{run.name}: the rule gives {rule:.6g}, the source "
                f"{run.flows_from.relative_to(ROOT)} was run at {source:.6g}"
            )
    return out


def main() -> int:
    """Write every base and scenario file of RUNS and RERUNS."""
    OUT.mkdir(parents=True, exist_ok=True)
    bases = {grid: write_base(grid)[0] for grid in (COARSE, MEDIUM, FINE)}
    for run in RUNS + RERUNS:
        print(f"wrote {write_scenario(run, bases[run.grid])}")
    for line in refusals():
        print(f"tolerance taken from the source: {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
