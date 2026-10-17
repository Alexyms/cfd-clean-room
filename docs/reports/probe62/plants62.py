"""Prompt 62: planted defects in the scenario schema and the scenario manager.

Each plant removes or bends one guard of ADR-014 steps 1 and 2 by a text
replacement in a worktree of the branch (``--root``, never the checkout being
measured), runs the test files that should catch it, records which tests
fail, and restores the file from git before the next plant. A control run
with no plant must pass first, in the same tree, or nothing is recorded.

Usage:
    python docs/reports/probe62/plants62.py --root results/builder62/P

The record is results/builder62/plants.json of the main checkout: per plant,
its name, the file, whether the named tests failed, and the failing test ids.
Line endings: the worktree's files are CRLF on Windows, so each replacement is
made on the text with its newlines normalised and written back with the
file's own.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

CONFIG = "src/config.py"
SCENARIOS = "src/scenarios.py"
TESTS_CONFIG = "tests/test_config.py"
TESTS_SCENARIOS = "tests/test_scenarios.py"

# (name, file, old, new, test files). Each old string occurs once.
PLANTS: list[tuple[str, str, str, str, list[str]]] = [
    (
        "fingerprint: turbulence_intensity out of FLOW_KEYS",
        SCENARIOS,
        '    "face",\n    "turbulence_intensity",\n',
        '    "face",\n',
        [TESTS_SCENARIOS],
    ),
    (
        "fingerprint: velocity out of FLOW_KEYS",
        SCENARIOS,
        '    "y_end",\n    "velocity",\n',
        '    "y_end",\n',
        [TESTS_SCENARIOS],
    ),
    (
        "fingerprint: concentration read as a flow key",
        SCENARIOS,
        '    "dissipation_length",\n)\n',
        '    "dissipation_length",\n    "concentration",\n)\n',
        [TESTS_SCENARIOS],
    ),
    (
        "fingerprint: segment order ignored (sorted)",
        SCENARIOS,
        "    text = json.dumps(flow_segments(config), allow_nan=False)\n",
        "    text = json.dumps(sorted(flow_segments(config), key=repr), allow_nan=False)\n",
        [TESTS_SCENARIOS],
    ),
    (
        "joining: pieces joined across a gap",
        SCENARIOS,
        "    return same and previous[hi] == keys[lo] and previous[hi] is not None\n",
        "    return same and previous[hi] is not None\n",
        [TESTS_SCENARIOS],
    ),
    (
        "joining: no pieces joined",
        SCENARIOS,
        "        if out and _continues(out[-1], keys, spec):\n",
        "        if False:\n",
        [TESTS_SCENARIOS],
    ),
    (
        "overlap: check removed",
        CONFIG,
        "                if a.start < b.end and b.start < a.end:\n",
        "                if False:\n",
        [TESTS_CONFIG, TESTS_SCENARIOS],
    ),
    (
        "overlap: touching events refused",
        CONFIG,
        "                if a.start < b.end and b.start < a.end:\n",
        "                if a.start <= b.end and b.start <= a.end:\n",
        [TESTS_CONFIG],
    ),
    (
        "emission: not divided by the area",
        SCENARIOS,
        "            per_class = np.asarray(source.emission, dtype=np.float64) / area\n",
        "            per_class = np.asarray(source.emission, dtype=np.float64)\n",
        [TESTS_SCENARIOS],
    ),
    (
        "emission: divided by the whole box, SOLID cells included",
        SCENARIOS,
        "            area = float(self._volume[cells].sum())\n",
        "            area = float(self._volume[np.outer(in_y, in_x)].sum())\n",
        [TESTS_SCENARIOS],
    ),
    (
        "source: SOLID cells in the box get the rate",
        SCENARIOS,
        "        cells = np.outer(in_y, in_x) & self._open\n",
        "        cells = np.outer(in_y, in_x)\n",
        [TESTS_SCENARIOS],
    ),
    (
        "merged validation: intervals not loaded",
        CONFIG,
        "                SimConfig.from_dict(mapping)\n            except (TypeError, ValueError) as err:\n",
        "                pass\n            except (TypeError, ValueError) as err:\n",
        [TESTS_CONFIG],
    ),
    (
        "pair rule: a half-stopped fan allowed",
        CONFIG,
        "                if intake in boundaries and intake not in returned:\n",
        "                if False:\n",
        [TESTS_CONFIG, TESTS_SCENARIOS],
    ),
    (
        "activation: active after the start and at the end",
        CONFIG,
        "            if event.end is not None and event.start <= t < event.end\n",
        "            if event.end is not None and event.start < t <= event.end\n",
        [TESTS_CONFIG, TESTS_SCENARIOS],
    ),
    (
        "replace: replacements appended instead of in place",
        CONFIG,
        "                boundaries = rebuilt\n",
        "                boundaries = {\n"
        "                    k: rebuilt[k]\n"
        "                    for k in sorted(rebuilt, key=lambda n: n in dict(change.segments))\n"
        "                }\n",
        [TESTS_CONFIG],
    ),
    (
        "set: the segment's keys win over the event's",
        CONFIG,
        "                    **boundaries[change.segment],\n                    **_thaw(change.keys),\n",
        "                    **_thaw(change.keys),\n                    **boundaries[change.segment],\n",
        [TESTS_CONFIG],
    ),
    (
        "t_end: an event ending at t_end refused",
        CONFIG,
        "            if start + duration > self.t_end:\n",
        "            if start + duration >= self.t_end:\n",
        [TESTS_CONFIG],
    ),
    (
        "run keys: inherited from the base",
        CONFIG,
        "            if not isinstance(block, dict) or key not in block:\n",
        "            if False:\n",
        [TESTS_CONFIG],
    ),
    (
        "initial: the factor not applied",
        CONFIG,
        "            values=tuple(t * factor for t in self.class_thresholds(ctx)),\n",
        "            values=tuple(t for t in self.class_thresholds(ctx)),\n",
        [TESTS_CONFIG],
    ),
    (
        "initial: SOLID cells hold the initial value",
        SCENARIOS,
        "            k: np.where(self._open, self._initial[k], 0.0)\n",
        "            k: np.full(self._open.shape, self._initial[k])\n",
        [TESTS_SCENARIOS],
    ),
    (
        "initial: two initials allowed",
        CONFIG,
        "        if len(initials) > 1:\n",
        "        if False:\n",
        [TESTS_CONFIG],
    ),
]


# Prompt 62b (review and test 62 B1): each key of the join's equality clause
# dropped alone, the test report's E4 pair, and the review's velocity-only
# clause.
_SAME = "    same = all(previous[k] == keys[k] for k in FLOW_KEYS if k not in _EXTENT_KEYS)\n"
JOIN_CLAUSE_KEYS: tuple[str, ...] = (
    "type",
    "location",
    "velocity",
    "u_velocity",
    "v_velocity",
    "obstacle",
    "face",
    "turbulence_intensity",
    "dissipation_length",
)
for _key in JOIN_CLAUSE_KEYS:
    PLANTS.append(
        (
            f"join (62b): {_key} not compared",
            SCENARIOS,
            _SAME,
            _SAME.replace("_EXTENT_KEYS)", f'_EXTENT_KEYS and k != "{_key}")'),
            [TESTS_SCENARIOS],
        )
    )
PLANTS.append(
    (
        "join (62b): obstacle and face not compared",
        SCENARIOS,
        _SAME,
        _SAME.replace(
            "_EXTENT_KEYS)", '_EXTENT_KEYS and k not in ("obstacle", "face"))'
        ),
        [TESTS_SCENARIOS],
    )
)
PLANTS.append(
    (
        "join (62b): velocity alone compared",
        SCENARIOS,
        _SAME,
        '    same = previous["velocity"] == keys["velocity"]\n',
        [TESTS_SCENARIOS],
    )
)


# Prompt 62b (review 62 B2): the source's class index.
_RATES = "        rates[:, cells] = per_class[:, None]\n"
PLANTS.append(
    (
        "source (62b): every class given class 0's value",
        SCENARIOS,
        _RATES,
        "        rates[:, cells] = per_class[0]\n",
        [TESTS_SCENARIOS],
    )
)
PLANTS.append(
    (
        "source (62b): the classes reversed",
        SCENARIOS,
        _RATES,
        "        rates[:, cells] = per_class[::-1, None]\n",
        [TESTS_SCENARIOS],
    )
)


def replace_once(path: Path, old: str, new: str) -> None:
    """Replace ``old`` by ``new`` once, keeping the file's newlines."""
    text = path.read_text(encoding="utf-8", newline="")
    newline = "\r\n" if "\r\n" in text else "\n"
    flat = text.replace(newline, "\n")
    if flat.count(old) != 1:
        raise SystemExit(f"{path}: the plant's text occurs {flat.count(old)} times")
    path.write_text(
        flat.replace(old, new).replace("\n", newline), encoding="utf-8", newline=""
    )


def run_tests(root: Path, files: list[str]) -> tuple[bool, list[str]]:
    """Run the files in ``root``; whether any failed, and the failing ids."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rf", *files],
        cwd=root,
        capture_output=True,
        text=True,
    )
    failed = [
        line.split(" ", 2)[1]
        for line in result.stdout.splitlines()
        if line.startswith("FAILED ")
    ]
    return result.returncode != 0, failed


def main() -> None:
    """Control first, then each plant, restored after its run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    every_file = sorted({f for *_, files in PLANTS for f in files})
    failed, ids = run_tests(root, every_file)
    if failed:
        raise SystemExit(f"the control run fails in {root}: {ids}")
    records = []
    for name, file, old, new, files in PLANTS:
        target = root / file
        try:
            replace_once(target, old, new)
            caught, ids = run_tests(root, files)
        finally:
            subprocess.run(["git", "checkout", "--", file], cwd=root, check=True)
        records.append({"plant": name, "file": file, "caught": caught, "failing": ids})
        print(
            f"{'caught' if caught else 'SURVIVED'}: {name} ({len(ids)} failing)",
            flush=True,
        )
    clean, ids = run_tests(root, every_file)
    if clean:
        raise SystemExit(f"the tree did not restore: {ids}")
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    out = Path(common).parent / "results" / "builder62" / "plants.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"commit": commit, "plants": records}, indent=1))


if __name__ == "__main__":
    main()
