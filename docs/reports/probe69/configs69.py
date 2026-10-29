"""Write the step 7 copies of the four reference scenarios (prompt 69).

Usage: python docs/reports/probe69/configs69.py  (from the repository root)

Each copy is the file on main with two changes and nothing else: the base
path made absolute, so the copy loads from results/builder69/configs/, and
solver.checkpoint_interval 1000 (a restart key, io_manager.RESTART_EXCLUDED_KEYS,
not a physics or grid key), so a crash or a sleep loses at most 1000 steps,
about 3.6 simulated seconds at Courant 0.4. The grid, the events and the run
keys are main's. The copies go to results/builder69/configs/.
"""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
SCENARIOS = ("clean_out", "door_leak", "equipment_dust", "filter_breach")
CHECKPOINT_INTERVAL = 1000


def main() -> int:
    out = ROOT / "results" / "builder69" / "configs"
    out.mkdir(parents=True, exist_ok=True)
    for name in SCENARIOS:
        source = ROOT / "configs" / f"scenario_{name}.yaml"
        mapping = yaml.safe_load(source.read_text(encoding="utf-8"))
        base = (source.parent / mapping["scenario"]["base"]).resolve()
        mapping["scenario"]["base"] = str(base)
        mapping["solver"]["checkpoint_interval"] = CHECKPOINT_INTERVAL
        target = out / f"scenario_{name}.yaml"
        target.write_text(
            yaml.safe_dump(mapping, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
