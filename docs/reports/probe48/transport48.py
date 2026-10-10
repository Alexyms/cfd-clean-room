"""Item 6 of prompt 48: prompt 47's three sources marched on this build's converged rooms.

Usage:
    python transport48.py run NAME SOURCE [--cfl C] [--cap-seconds S] [--tag T]

NAME is a converged flow record of coupled48.py under results/builder48/;
SOURCE is S1, S2 or S3. The march is prompt 47's
(docs/reports/probe47/transport47.py ``run``) unchanged: the same squares
(no source moved in prompt 47, so its sources.json was never written), the
two classes 0.5 and 5 micrometres, the transport Courant number 0.4, the
record's frozen faces and eddy viscosity, from a zero field, to the steady
rule or 600 s. Only the record directory differs: results/builder48/, as
transport_NAME_SOURCE[_TAG].
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe47"))
sys.path.insert(0, str(ROOT / "docs" / "reports" / "probe46"))

import transport46  # noqa: E402
import transport47  # noqa: E402

OUT = ROOT / "results" / "builder48"
transport46.OUT = OUT
transport47.OUT = OUT


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "run":
        raise SystemExit("usage: transport48.py run NAME SOURCE [options]")
    transport47.main()
