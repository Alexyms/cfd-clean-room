"""Write the breach arms of prompt 70's probe B (results/builder70/configs/).

Usage: python docs/reports/probe70/configs70.py  (from the repository root)

Each arm is configs/scenario_filter_breach.yaml on its base,
configs/clean_room_default.yaml, with these changes and nothing else:
- the grid 160x60, with solver.mass_imbalance_tol recomputed for it as
  ADR-011 G states the bound (1e-4 rho V_min / t_end, the scenario's t_end),
  as prompt 59b's clean-out probe did on the same grid;
- the supply's turbulence_intensity and dissipation_length set to the arm's
  values on the base's hepa_supply and on the event's three replacement
  segments alike, so the breach stays transport-only (one flow state);
- transport.turbulent_schmidt set to the arm's value;
- the base path made absolute, and solver.checkpoint_interval 1000.
"""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "builder70" / "configs"
GRID = (160, 60)
CHECKPOINT_INTERVAL = 1000
# name: (intensity, dissipation length m, Sc_t)
ARMS = {
    "i05": (0.05, 0.1, 0.7),  # the file's values: the control
    "i10": (0.10, 0.1, 0.7),  # the top of the intensity range, prediction (b)
    "i10_l30": (0.10, 0.3, 0.7),  # the range's corner of largest supply nu_t
    "i02_l05": (0.02, 0.05, 0.7),  # the corner of smallest supply nu_t
    "sc02": (0.05, 0.1, 0.2),  # the Schmidt range's low end, flows from i05
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    base_source = ROOT / "configs" / "clean_room_default.yaml"
    scenario_source = ROOT / "configs" / "scenario_filter_breach.yaml"
    for name, (intensity, length, schmidt) in ARMS.items():
        base = yaml.safe_load(base_source.read_text(encoding="utf-8"))
        base["domain"]["nx"], base["domain"]["ny"] = GRID
        base["boundaries"]["hepa_supply"]["turbulence_intensity"] = intensity
        base["boundaries"]["hepa_supply"]["dissipation_length"] = length
        base["transport"]["turbulent_schmidt"] = schmidt
        base_path = OUT / f"base_{name}.yaml"
        base_path.write_text(
            yaml.safe_dump(base, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        scenario = yaml.safe_load(scenario_source.read_text(encoding="utf-8"))
        scenario["scenario"]["base"] = str(base_path.resolve())
        for segment in scenario["scenario"]["events"][0]["replace"]["with"]:
            segment["turbulence_intensity"] = intensity
            segment["dissipation_length"] = length
        v_min = (base["domain"]["width"] / GRID[0]) * (
            base["domain"]["height"] / GRID[1]
        )
        scenario["solver"]["mass_imbalance_tol"] = (
            1e-4 * base["fluid"]["density"] * v_min / scenario["solver"]["t_end"]
        )
        scenario["solver"]["checkpoint_interval"] = CHECKPOINT_INTERVAL
        target = OUT / f"scenario_filter_breach_{name}.yaml"
        target.write_text(
            yaml.safe_dump(scenario, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
