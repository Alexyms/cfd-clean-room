"""Print the Driver and Seegmiller (1985) data for validation/backstep.py.

Usage:
    python data52.py            print the module's data tuples
    python data52.py check      the files' checksums against the module's provenance

Reads the three files the NASA Turbulence Modeling Resource publishes for
its 2D backward-facing step case (ADR-012 source 13), stored with their
provenance under validation/data/driver_seegmiller_1985/, and prints the
Python tuples validation/backstep.py holds, so no value in the module is
typed. The profile file's y is y/H, u and v are over U_ref, uv is 1000
u'v' / U_ref^2; the Cf file's x is x/H; the Cp file holds the shifted and
the original coefficient on the bottom and the top wall.
"""

import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "validation" / "data" / "driver_seegmiller_1985"
FILES = ("profiles.exp.dat", "cf.exp.dat", "cp.expnew.dat")


def zones(path: Path) -> dict[str, list[list[float]]]:
    """The numeric rows of a Tecplot-style file, keyed by zone title ('' before any)."""
    out: dict[str, list[list[float]]] = {}
    title = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or text.startswith("variables"):
            continue
        if text.startswith("zone"):
            title = text.split('"')[1]
            out.setdefault(title, [])
            continue
        out.setdefault(title, []).append([float(v) for v in text.split()])
    return out


def station_of(title: str) -> float:
    """'exp, x/H=-4' -> -4.0."""
    return float(title.split("x/H=")[1])


def fmt(values: list[float]) -> str:
    return ", ".join(repr(v) for v in values)


def print_profiles() -> None:
    data = zones(DATA / "profiles.exp.dat")
    print("PROFILE_ROWS: dict[float, tuple[tuple[float, float, float], ...]] = {")
    for title, rows in data.items():
        x = station_of(title)
        print(f"    {x!r}: (")
        for row in rows:
            # columns: ID, y, u, v, uu, vv, uv, uuu, uvv, vuu, vvv
            print(f"        ({row[1]!r}, {row[2]!r}, {row[6]!r}),")
        print("    ),")
    print("}")


def print_cf() -> None:
    data = zones(DATA / "cf.exp.dat")
    print("CF_ROWS: tuple[tuple[float, float, float], ...] = (")
    for row in data[""]:
        print(f"    ({fmt(row)}),")
    print(")")


def print_cp() -> None:
    data = zones(DATA / "cp.expnew.dat")
    for title, name in (("bottom wall", "CP_BOTTOM_ROWS"), ("top wall", "CP_TOP_ROWS")):
        print(f"{name}: tuple[tuple[float, float, float], ...] = (")
        for row in data[title]:
            print(f"    ({fmt(row)}),")
        print(")")


def check() -> None:
    for name in FILES:
        digest = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        print(f"{digest}  {name}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check()
        return
    print_profiles()
    print()
    print_cf()
    print()
    print_cp()


if __name__ == "__main__":
    main()
