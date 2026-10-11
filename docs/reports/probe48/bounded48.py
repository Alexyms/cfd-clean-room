"""Step 5's bounded-row characterisation, pointed at prompt 48's records.

Usage:
    python bounded48.py history NAME [NAME ...]    as bounded44.py history
    python bounded48.py where NAME                 as bounded44.py where

The functions are docs/reports/probe44/bounded44.py's, unchanged, as
prompts 46 and 47 used them; only the records directory is this build's
(results/builder48/). Every row of prompt 48 is run with the cell of the
largest change recorded (coupled48.py, through coupled47.py), so `where`
applies to any of them.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe44"))

import bounded44  # noqa: E402

bounded44.OUT = ROOT / "results" / "builder48"

if __name__ == "__main__":
    bounded44.main()
