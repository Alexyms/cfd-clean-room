"""Write the sweep's configurations (ADR-015 B as accepted; prompt 72).

Usage: python docs/reports/probe72/configs72.py  (from the repository root)

The files go to results/builder72/configs/. Each run is one of the three
event scenarios (configs/scenario_filter_breach.yaml, scenario_door_leak.yaml,
scenario_equipment_dust.yaml) on its base, configs/clean_room_default.yaml,
at one arm of ARMS and one grid of GRIDS, with these changes and nothing else,
as configs70.py made the probe's breach arms:

- the grid, with solver.mass_imbalance_tol recomputed for it as ADR-011 G
  states the bound (1e-4 rho V_min / t_end, the scenario's t_end);
- the supply's turbulence_intensity and dissipation_length set to the arm's
  on the base's hepa_supply and on every event segment that carries the
  supply's keys: the breach's three replacement segments, so the breach stays
  transport-only (one flow state), and the door's inflow segment, whose keys
  the scenario file defines as a copy of the supply's;
- transport.turbulent_schmidt set to the arm's value;
- the base path made absolute, and solver.checkpoint_interval 1000 (a restart
  key, not a physics or grid key).

The clean-out (configs/scenario_clean_out.yaml) is written at the file's
inputs on 160x60, the third grid of Phase 4's condition 2, and on 80x30 to
the same 240 s (the gate's 80x30 run stopped at 120 s), with the same
tolerance rule and checkpoint key.

An arm whose flow state is another arm's (FLOWS_FROM) differs from it in
transport.turbulent_schmidt alone, so run72.py passes --flows-from and the
runner's flow-input check accepts it.
"""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "builder72" / "configs"
CHECKPOINT_INTERVAL = 1000
SCENARIOS = ("filter_breach", "door_leak", "equipment_dust")
# name: (intensity, dissipation length m, Sc_t)
ARMS = {
    "control": (0.05, 0.1, 0.7),  # the file's values
    "i10": (0.10, 0.1, 0.7),  # the top of the intensity range
    "max_nut": (0.10, 0.3, 0.7),  # the range's corner of largest supply nu_t
    "min_nut": (0.02, 0.05, 0.7),  # the corner of smallest supply nu_t
    "sc02": (0.05, 0.1, 0.2),  # the Schmidt range's low end, flows from control
    "widest": (0.10, 0.3, 0.2),  # largest nu_t over the lowest Sc_t, flows from max_nut
}
# The arms whose flow state is another arm's (the orchestrator's table): on
# 160x60 only, since max_nut is not run on 320x120 and widest solves there.
FLOWS_FROM = {"sc02": "control", "widest": "max_nut"}
COARSE, FINE = (160, 60), (320, 120)
# The arms repeated on the product grid: the two extremes (ADR-015 B (1)).
FINE_ARMS = ("widest", "min_nut")
GRIDS: dict[str, tuple[tuple[int, int], ...]] = {
    arm: (COARSE, FINE) if arm in FINE_ARMS else (COARSE,) for arm in ARMS
}
CLEAN_OUT_GRIDS = ((160, 60), (80, 30))
CLEAN_OUT_T_END = 240.0
# Event segments carrying the supply's turbulence keys, per scenario file.
SUPPLY_KEYS = ("turbulence_intensity", "dissipation_length")


def grid_tag(grid: tuple[int, int]) -> str:
    return f"{grid[0]}x{grid[1]}"


def mass_imbalance_tol(base: dict, grid: tuple[int, int], t_end: float) -> float:
    """ADR-011 G's bound: 1e-4 rho V_min / t_end, V_min the cell area per metre."""
    v_min = (base["domain"]["width"] / grid[0]) * (base["domain"]["height"] / grid[1])
    return 1e-4 * base["fluid"]["density"] * v_min / t_end


def write_base(
    name: str, grid: tuple[int, int], arm: tuple[float, float, float] | None
) -> tuple[Path, dict]:
    """Write the room's base for an arm and grid; None keeps the file's inputs."""
    base = yaml.safe_load(
        (ROOT / "configs" / "clean_room_default.yaml").read_text(encoding="utf-8")
    )
    base["domain"]["nx"], base["domain"]["ny"] = grid
    if arm is not None:
        intensity, length, schmidt = arm
        base["boundaries"]["hepa_supply"]["turbulence_intensity"] = intensity
        base["boundaries"]["hepa_supply"]["dissipation_length"] = length
        base["transport"]["turbulent_schmidt"] = schmidt
    path = OUT / f"base_{name}_{grid_tag(grid)}.yaml"
    path.write_text(
        yaml.safe_dump(base, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return path, base


def carry_supply_keys(scenario: dict, intensity: float, length: float) -> None:
    """Set the supply's keys on every event segment that carries them."""
    for event in scenario["scenario"]["events"]:
        replace = event.get("replace")
        if replace is None:
            continue
        for segment in replace["with"]:
            if all(key in segment for key in SUPPLY_KEYS):
                segment["turbulence_intensity"] = intensity
                segment["dissipation_length"] = length


def write_scenario(
    scenario_name: str,
    run_name: str,
    base_path: Path,
    base: dict,
    grid: tuple[int, int],
    arm: tuple[float, float, float] | None,
    t_end: float | None = None,
) -> Path:
    scenario = yaml.safe_load(
        (ROOT / "configs" / f"scenario_{scenario_name}.yaml").read_text(
            encoding="utf-8"
        )
    )
    scenario["scenario"]["base"] = str(base_path.resolve())
    if arm is not None:
        carry_supply_keys(scenario, arm[0], arm[1])
    if t_end is not None:
        scenario["solver"]["t_end"] = t_end
    scenario["solver"]["mass_imbalance_tol"] = mass_imbalance_tol(
        base, grid, scenario["solver"]["t_end"]
    )
    scenario["solver"]["checkpoint_interval"] = CHECKPOINT_INTERVAL
    target = OUT / f"scenario_{run_name}.yaml"
    target.write_text(
        yaml.safe_dump(scenario, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return target


def run_names() -> list[tuple[str, str, str, tuple[int, int]]]:
    """(run name, scenario, arm, grid) for every sweep run, then the clean-outs."""
    names = [
        (f"{scenario}_{arm}_{grid_tag(grid)}", scenario, arm, grid)
        for scenario in SCENARIOS
        for arm in ARMS
        for grid in GRIDS[arm]
    ]
    names += [
        (f"clean_out_{grid_tag(grid)}", "clean_out", "file", grid)
        for grid in CLEAN_OUT_GRIDS
    ]
    return names


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    bases: dict[tuple[str, tuple[int, int]], tuple[Path, dict]] = {}
    for run_name, scenario, arm, grid in run_names():
        key = (arm, grid)
        if key not in bases:
            bases[key] = write_base(arm, grid, ARMS.get(arm))
        base_path, base = bases[key]
        t_end = CLEAN_OUT_T_END if scenario == "clean_out" else None
        target = write_scenario(
            scenario, run_name, base_path, base, grid, ARMS.get(arm), t_end
        )
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
