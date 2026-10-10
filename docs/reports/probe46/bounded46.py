"""Step 5's bounded-row characterisation, pointed at this step's records.

Usage:
    python bounded46.py history NAME [NAME ...]    as bounded44.py history
    python bounded46.py where NAME                 as bounded44.py where (a --locate record)

The functions are docs/reports/probe44/bounded44.py's, unchanged; only the
records directory is this step's (results/builder46/), so the period
measure (compare34b's) and the strongest recurrence are the ones step 5
reported, and `where` reads the same regions from the configuration. The
JSON written beside the record is bounded44's (bounded_NAME.json,
bounded_where_NAME.json), under results/builder46/.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe44"))

import bounded44  # noqa: E402

bounded44.OUT = ROOT / "results" / "builder46"

if __name__ == "__main__":
    bounded44.main()
