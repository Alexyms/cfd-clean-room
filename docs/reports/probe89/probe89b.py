"""Prompt 89b's probe: the core built for x86-64-v3 against round 0's head.

    python docs/reports/probe89/probe89b.py build v3
    python docs/reports/probe89/probe89b.py check v3
    python docs/reports/probe89/probe89b.py session --run 1 --builds reference base head_o2 head_v3 v3
    python docs/reports/probe89/probe89b.py plants
    python docs/reports/probe89/probe89b.py vecinfo

Run under WSL2 with the Linux environment, as ``probe89.py`` is. This file
configures ``probe89.py`` for round 1 and runs it, so the builds, the checks,
the timings, the planted defects and the vectorisation report are the same
code as round 0's; only the builds, the records' directory (``results/builder89b/``),
the flags of the planted defects' builds and the table of planted defects
differ. Each child process ``probe89.py`` starts (the replays, the timing of
each build) is this script again.

The builds (``BUILDS``):

- ``base``: ``core/`` at ``origin/main`` before prompt 89 (3c8d1ec), under the
  flags it was built with.
- ``head_o2``: ``core/`` at eacc71c, round 0's head, under those flags: the
  build the pass's numbers were taken on, and the one round 1 is measured against.
- ``head_v3``: the same sources with ``-march=x86-64-v3`` added, the flux still
  the loop of one face at a time: the form the decision to switch the flux
  was made against.
- ``v3``: the checkout's ``core/``, which builds under ``-march=x86-64-v3``
  with the flux in blocks. The sources of a commit come from
  ``results/builder89b/src_<commit>.tar`` (``git archive --format=tar -o
  results/builder89b/src_<commit>.tar <commit> core``, run from Windows).

Every timing process records, beside the load ``probe89.py`` samples, the
Windows power scheme and power mode.
"""

import sys
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import probe89  # noqa: E402 -- after sys.path
import variants89b  # noqa: E402

BASE_FLAGS = probe89.PROJECT_FLAGS
V3_FLAGS = BASE_FLAGS + " -march=x86-64-v3"
BUILDS: dict[str, dict[str, str]] = {
    "base": {"source": "3c8d1ec", "candidate": "", "flags": BASE_FLAGS},
    "head_o2": {"source": "eacc71c", "candidate": "", "flags": BASE_FLAGS},
    "head_v3": {"source": "eacc71c", "candidate": "", "flags": V3_FLAGS},
    "v3": {"source": "head", "candidate": "", "flags": V3_FLAGS},
}
# The overlays Windows 11 keeps for its power mode setting.
POWER_MODES = {
    "00000000-0000-0000-0000-000000000000": "Balanced",
    "961cc777-2547-4f9d-8174-7d86181b8a7a": "Best power efficiency",
    "3af9b8d9-7c97-431d-ad78-34a8bfea439f": "Better performance",
    "ded574b5-45a0-4f42-8737-46345c09c238": "Best performance",
}


def power_lines() -> list[str]:
    """The Windows power scheme and, on mains, the power mode, as text lines."""
    script = (
        "powercfg /getactivescheme;"
        " $k = Get-ItemProperty"
        " 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Power\\User\\PowerSchemes'"
        " -ErrorAction SilentlyContinue;"
        " 'power mode overlay on mains ' + $k.ActiveOverlayAcPowerScheme"
    )
    try:
        done = probe89.run_command(
            ["powershell.exe", "-NoProfile", "-Command", script], timeout=60
        )
    except (OSError, probe89.subprocess.TimeoutExpired):
        return []
    # powercfg ends its line without a newline, so the mode's line follows on it.
    lines = [
        line.strip()
        for line in done.stdout.replace(")power mode", ")\npower mode").splitlines()
        if line.strip()
    ]
    out = []
    for line in lines:
        if line.startswith("power mode overlay on mains "):
            guid = line.rsplit(" ", 1)[1]
            line += f" ({POWER_MODES.get(guid, 'not one of the known overlays')})"
        out.append(line)
    return out


def configure() -> None:
    """Point ``probe89`` at round 1's builds, records and planted defects."""
    probe89.RESULTS = probe89.ROOT / "results" / "builder89b"
    probe89.WORK = probe89.Path.home() / "scratch89b" / "builds"
    probe89.PROJECT_FLAGS = V3_FLAGS
    probe89.BUILDS.clear()
    probe89.BUILDS.update(BUILDS)
    probe89.variants89 = variants89b
    sampled: Callable[[], list[str]] = probe89.windows_load

    def load_and_power() -> list[str]:
        return [*sampled(), *power_lines()]

    probe89.windows_load = load_and_power


def main(argv: list[str] | None = None) -> int:
    """Configure ``probe89`` for round 1 and run one of its commands."""
    configure()
    return probe89.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
