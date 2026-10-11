"""Print the Annex 20 data tables of validation/annex20.py from the records.

Usage:
    python data49.py

Reads the digitizer's record (results/builder49/rong_nielsen_2008_digitized.json,
digitize49.py), which carries the workbook's four measured lines in the
room's coordinates (metres, y up from the floor) and the published standard
k-epsilon prediction resampled at the measurement stations, and prints the
Python tuples the validation module stores, so that nothing in that module
is typed by hand. The module is assembled from this output once; a rerun
must print the same text.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RECORD = ROOT / "results" / "builder49" / "rong_nielsen_2008_digitized.json"
ORDER = ("x3", "x6", "y2916", "y0084")


def main() -> None:
    """One tuple of (station, measured, published) rows per line."""
    rec = json.loads(RECORD.read_text())
    for key in ORDER:
        st = rec["panels"][key]["at_stations"]
        print(f"{key.upper()}_ROWS: tuple[tuple[float, float, float | None], ...] = (")
        for c, m, p in zip(
            st["coordinate"],
            st["measured_u_over_u0"],
            st["published_u_over_u0"],
            strict=True,
        ):
            pub = "None" if p is None else f"{p:.4f}"
            print(f"    ({c:.4f}, {m:.5f}, {pub}),")
        print(")")


if __name__ == "__main__":
    main()
