"""Write prompt 79's configurations and name its jobs (the re-run on the budgeted solve).

Usage: python docs/reports/probe79/configs79.py  (from the repository root)

Three groups, each the configuration of the runs it replaces, written from
main's files by the generator those runs used, and nothing else changed:

- P, the product grid: the five reference scenarios on 320x120 as
  ``docs/reports/probe76/configs76.py`` writes them (main's file, the base path
  made absolute, solver.checkpoint_interval 1000). They replace
  ``results/builder76/`` and, for the four the Phase 4 gate ran,
  ``results/builder69/``, whose runs prompt 76 found bitwise equal.
- S, the sensitivity sweep: every run of ``docs/reports/probe72/configs72.py``,
  its module run with its output directory pointed here.
- L, the placement's small-grid runs: prompt 75's three 80x30 runs as
  ``results/builder75/run75.py``'s ``coarse()`` built them (the base path
  absolute, the domain section ``nx: 80, ny: 30``, solver.mass_imbalance_tol
  1e-4 rho V_min / 60 s as run75.py wrote it as a literal, for all three).

Every job names its predecessor, the run file it replaces, and passes it as
``--flows-from`` so its converged flow states are copied in and only the march
is new. The one job with no predecessor is ``door_leak_sc02_160x60``, which
prompt 72 did not run (its flow state is control's, whose open state did not
converge): it waits for this pass's control run and takes its flows from it,
as run72.py did.
"""

import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe72"))

import configs72  # noqa: E402 -- follows sys.path.insert

RESULTS = ROOT / "results" / "builder79"
CONFIGS = RESULTS / "configs"
GROUP_DIRS = {"P": "product", "S": "sweep", "L": "placement"}
CHECKPOINT_INTERVAL = 1000

PRODUCT_SCENARIOS = (
    "clean_out",
    "door_leak",
    "equipment_dust",
    "filter_breach",
    "seal_leak",
)
PRODUCT_BEFORE = ROOT / "results" / "builder76"
SWEEP_BEFORE = ROOT / "results" / "builder72"
PLACEMENT_BEFORE = ROOT / "results" / "builder75" / "runs"
# run75.py's names and the scenario file each was made from.
PLACEMENT_RUNS = {
    "door": "scenario_door_leak.yaml",
    "breach": "scenario_filter_breach.yaml",
    "dust": "scenario_equipment_dust.yaml",
}
# run75.py's literal: 1e-4 rho V_min / 60 s on 80x30, for all three runs.
PLACEMENT_TOL = 1.0e-4 * 1.2 * (0.1 * 0.1) / 60.0
FINE = "320x120"


def group_dir(group: str) -> Path:
    """The directory of a group's run files, summaries and logs."""
    return RESULTS / GROUP_DIRS[group]


def write_product() -> None:
    """Write the five product-grid copies, as configs76.py does."""
    out = CONFIGS / "product"
    out.mkdir(parents=True, exist_ok=True)
    for name in PRODUCT_SCENARIOS:
        source = ROOT / "configs" / f"scenario_{name}.yaml"
        mapping = yaml.safe_load(source.read_text(encoding="utf-8"))
        mapping["scenario"]["base"] = str(
            (source.parent / mapping["scenario"]["base"]).resolve()
        )
        mapping["solver"]["checkpoint_interval"] = CHECKPOINT_INTERVAL
        target = out / f"scenario_{name}.yaml"
        target.write_text(
            yaml.safe_dump(mapping, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        print(f"wrote {target}")


def write_sweep() -> None:
    """Write the sweep's bases and scenarios with configs72's own code."""
    configs72.OUT = CONFIGS / "sweep"
    configs72.main()


def write_placement() -> None:
    """Write prompt 75's three 80x30 scenarios as run75.py's coarse() built them."""
    out = CONFIGS / "placement"
    out.mkdir(parents=True, exist_ok=True)
    for name, file in PLACEMENT_RUNS.items():
        raw = yaml.safe_load((ROOT / "configs" / file).read_text(encoding="utf-8"))
        raw["scenario"]["base"] = str(ROOT / "configs" / "clean_room_default.yaml")
        raw["domain"] = {"nx": 80, "ny": 30}
        raw["solver"]["mass_imbalance_tol"] = PLACEMENT_TOL
        target = out / f"scenario_{name}.yaml"
        target.write_text(
            yaml.safe_dump(raw, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        print(f"wrote {target}")


def jobs() -> list[dict]:
    """Every job: name, group, grid, configuration, run file, flows-from, dependency.

    ``before`` is the run file the job replaces (None for the one job prompt 72
    did not run); ``flows_from`` is the file passed as ``--flows-from``;
    ``depends`` names a job of this pass that must finish first.
    """
    rows = []
    for name in PRODUCT_SCENARIOS:
        before = PRODUCT_BEFORE / f"{name}.h5"
        rows.append(
            {
                "name": name,
                "group": "P",
                "grid": FINE,
                "config": CONFIGS / "product" / f"scenario_{name}.yaml",
                "before": before,
                "flows_from": before,
                "depends": None,
            }
        )
    for name, _scenario, arm, grid in configs72.run_names():
        before = SWEEP_BEFORE / f"{name}.h5"
        depends = None
        flows_from = before
        if not before.exists():
            source = configs72.FLOWS_FROM[arm]
            depends = name.replace(f"_{arm}_", f"_{source}_")
            flows_from = group_dir("S") / f"{depends}.h5"
            before = None
        rows.append(
            {
                "name": name,
                "group": "S",
                "grid": configs72.grid_tag(grid),
                "config": CONFIGS / "sweep" / f"scenario_{name}.yaml",
                "before": before,
                "flows_from": flows_from,
                "depends": depends,
            }
        )
    for name in PLACEMENT_RUNS:
        before = PLACEMENT_BEFORE / f"{name}.h5"
        rows.append(
            {
                "name": name,
                "group": "L",
                "grid": "80x30",
                "config": CONFIGS / "placement" / f"scenario_{name}.yaml",
                "before": before,
                "flows_from": before,
                "depends": None,
            }
        )
    for row in rows:
        row["out"] = group_dir(row["group"]) / f"{row['name']}.h5"
    return rows


def main() -> int:
    """Write every configuration of jobs()."""
    write_product()
    write_sweep()
    write_placement()
    return 0


if __name__ == "__main__":
    sys.exit(main())
