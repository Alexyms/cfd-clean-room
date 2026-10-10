"""Step 5's bounded-row characterisation, pointed at this step's records.

Usage:
    python bounded47.py history NAME [NAME ...]    as bounded44.py history
    python bounded47.py where NAME                 as bounded44.py where

The functions are docs/reports/probe44/bounded44.py's, unchanged, as
prompt 46's bounded46.py used them; only the records directory is this
step's (results/builder47/). Every row of prompt 47 is run with the cell of
the largest change recorded, so `where` applies to any of them.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe44"))

import bounded44  # noqa: E402

bounded44.OUT = ROOT / "results" / "builder47"

if __name__ == "__main__":
    bounded44.main()
