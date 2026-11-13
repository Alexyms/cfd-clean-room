"""Which changed lines a VAL-022 run executes (prompt 78b, review 78 S3).

    python docs/reports/probe78/cover78b.py [--base REV] [--out FILE]

The branch edits ``time_integration.run`` (the restart branch), ``io_manager``
(the budget rows), ``solver_transport``, ``scalar_scheme`` and ``config``. The
four VAL-022 cylinder rows are tier-2 runs through ``time_integration.run``,
hours long each, and were not retaken in prompt 78 on the argument that a
tier-2 run from rest executes none of the changed statements. This script
shows it for one case: it loads the ``val022_re40`` preset, shortens it to a
few steps with a frame at every step and a checkpoint every second step (so
the run file and checkpoint writers run), makes the harness's perturbed
start-up and runs ``time_integration.run`` as ``scripts/benchmark.py`` does,
under ``coverage`` started after every module is imported. The statements the
diff against the merge base added or changed are intersected with the lines the
run executed. The record goes to results/builder78/val022_coverage.json, which
``tables78.py`` reads.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import coverage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from src.config import SimConfig  # noqa: E402 -- follows sys.path.insert
from src.time_integration import (  # noqa: E402 -- follows sys.path.insert
    Provenance,
    run,
)
from validation import cylinder  # noqa: E402 -- follows sys.path.insert
from validation.cases import load_preset  # noqa: E402 -- follows sys.path.insert

BASE = "8e61788a8c918b178b4a1fd16fd981e6c075a859"
OUT = ROOT / "results" / "builder78" / "val022_coverage.json"
CASE = "val022_re40"
STEPS = 4
# Code the diff touches, as the tables name it. Tests, docs and configs are not
# executed by the run; validation/ is read, not edited, for a tier-2 case.
TREES = ("src", "scripts", "validation")
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def changed_lines(base: str) -> dict[str, set[int]]:
    """The lines of each .py file under TREES that the working tree changes against BASE."""
    diff = subprocess.run(
        ["git", "diff", "--unified=0", base, "--", *TREES],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    changed: dict[str, set[int]] = {}
    current = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
            if current.endswith(".py"):
                changed.setdefault(current, set())
            else:
                current = None
        elif current is not None:
            match = HUNK.match(line)
            if match:
                start = int(match.group(1))
                count = 1 if match.group(2) is None else int(match.group(2))
                changed[current].update(range(start, start + count))
    return {name: lines for name, lines in changed.items() if lines}


def shortened(config: SimConfig) -> SimConfig:
    """The preset with a few steps, a frame every step and a checkpoint every second."""
    raw = config.merged_mapping()
    raw["solver"]["t_end"] = STEPS * raw["solver"]["dt"]
    raw["solver"]["output_interval"] = 1
    raw["solver"]["checkpoint_interval"] = 2
    return SimConfig.from_dict(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    changed = changed_lines(args.base)
    config = shortened(load_preset(CASE))
    initial, startup_outer = cylinder.perturbed_startup(config)
    folder = Path(
        tempfile.mkdtemp(prefix="cover78b_", dir=ROOT / "results" / "builder78")
    )
    cov = coverage.Coverage(
        data_file=None,
        include=[str(ROOT / name) for name in changed],
    )
    cov.start()
    record = run(
        config,
        folder / "run.h5",
        provenance=Provenance("cover78b", "2026-11-12T00:00:00", "this-machine", 1),
        initial=initial,
    )
    cov.stop()
    data = cov.get_data()
    rows = []
    for name, lines in sorted(changed.items()):
        path = str(ROOT / name)
        _, statements, _, _, _ = cov.analysis2(path)
        executable = sorted(lines & set(statements))
        executed = sorted(set(executable) & set(data.lines(path) or []))
        rows.append(
            {
                "file": name,
                "changed_lines": len(lines),
                "changed_statements": len(executable),
                "executed_in_file": len(data.lines(path) or []),
                "executed_statements": len(executed),
                "executed_lines": executed,
            }
        )
    result = {
        "case": CASE,
        "base": args.base,
        "steps": record.steps,
        "startup_outer": startup_outer,
        "tier": "tier 2 (transient)",
        "files": rows,
    }
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for row in rows:
        print(
            f"{row['file']}: {row['changed_statements']} changed statements, "
            f"{row['executed_statements']} executed {row['executed_lines']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
