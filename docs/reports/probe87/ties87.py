"""NumPy's tie and NaN rules for the clamp's minimum and maximum (prompt 87b).

    python docs/reports/probe87/ties87.py

The core copies ``np.minimum`` and ``np.maximum`` on two doubles (``scalar.hpp``,
``np_minimum`` and ``np_maximum``) and takes ``np.minimum.reduce`` over the
clamp's four arrays as a fold from the first. This measures what NumPy does, on
the NumPy of the environment that runs it, at every array length from 1 to
1,000 and on three 2-D shapes:

- ``np.minimum`` and ``np.maximum`` return their second argument where the two
  compare equal, ``0.0`` against ``-0.0`` in both orders, and ``np.maximum(0.0, x)``
  with a scalar first argument, as the clamp calls it;
- a NaN in either argument gives a NaN;
- ``np.minimum.reduce`` over four arrays equals the fold
  ``minimum(minimum(minimum(a, b), c), d)`` bit for bit, where the arrays hold
  every one of the sixteen patterns of ``0.0`` and ``-0.0`` across the four
  rows, each pattern alone and all of them mixed along the array (shifted
  sixteen ways, so every position meets every pattern), and where one row
  holds NaNs.

Mixing the patterns along the array is what a SIMD reduction would show: lanes
that pair elements 0 and 2 against 1 and 3 resolve a tie between two zeros of
opposite sign differently from a fold from the first. The record is written to
``results/builder87/ties87.json`` for ``tables87.py``; the exit status is 1 if
any case departs from the rule.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RECORD = ROOT / "results" / "builder87" / "ties87.json"
LENGTHS = range(1, 1001)
SHAPES = ((30, 79), (119, 320), (120, 319))
# Four rows, two zeros each: sixteen patterns of signs.
PATTERNS = 16


def bits(a: np.ndarray) -> np.ndarray:
    """The 64-bit patterns of a float64 array, so that 0.0 and -0.0 differ."""
    return np.ascontiguousarray(a, dtype=np.float64).view(np.uint64)


def equal_bits(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two arrays have one shape and equal 64-bit patterns."""
    return a.shape == b.shape and bool(np.array_equal(bits(a), bits(b)))


def zeros_by_pattern(shape: tuple[int, ...], row: int, shift: int) -> np.ndarray:
    """An array of signed zeros whose element j has the sign of bit ``row`` of j + shift."""
    index = np.arange(int(np.prod(shape))).reshape(shape)
    negative = ((index + shift) % PATTERNS >> row) & 1
    return np.where(negative == 1, -0.0, 0.0)


def fold(rows: list[np.ndarray]) -> np.ndarray:
    """``np.minimum`` folded from the first row, the order the core takes."""
    result = rows[0]
    for row in rows[1:]:
        result = np.minimum(result, row)
    return result


def tie_rules(shape: tuple[int, ...]) -> list[bool]:
    """The second-of-a-tie rule for minimum and maximum on one shape, in four checks."""
    plus, minus = np.zeros(shape), np.full(shape, -0.0)
    mixed = zeros_by_pattern(shape, 0, 0)
    opposite = -mixed
    return [
        # np.minimum(a, b) with a == b elementwise returns b, whichever zero b is.
        equal_bits(np.minimum(plus, minus), minus)
        and equal_bits(np.minimum(minus, plus), plus)
        and equal_bits(np.minimum(mixed, opposite), opposite),
        equal_bits(np.maximum(plus, minus), minus)
        and equal_bits(np.maximum(minus, plus), plus)
        and equal_bits(np.maximum(mixed, opposite), opposite),
        # The clamp's own call, a scalar zero against an array.
        equal_bits(np.maximum(0.0, minus), minus)
        and equal_bits(np.maximum(0.0, plus), plus)
        and equal_bits(np.maximum(0.0, opposite), opposite),
        # A NaN in either argument gives a NaN, in both functions.
        all(
            bool(np.isnan(f(a, b)).all())
            for f in (np.minimum, np.maximum)
            for a, b in ((np.full(shape, np.nan), plus), (plus, np.full(shape, np.nan)))
        ),
    ]


def reduce_rules(shape: tuple[int, ...]) -> list[bool]:
    """``np.minimum.reduce`` against the fold on one shape, in three checks."""
    # Each of the sixteen patterns alone: row r is -0.0 where bit r is set.
    alone = True
    for pattern in range(PATTERNS):
        rows = [np.full(shape, -0.0 if (pattern >> r) & 1 else 0.0) for r in range(4)]
        alone = alone and equal_bits(np.minimum.reduce(rows), fold(rows))
    # All sixteen along the array, shifted every way.
    mixed = True
    for shift in range(PATTERNS):
        rows = [zeros_by_pattern(shape, r, shift) for r in range(4)]
        mixed = mixed and equal_bits(np.minimum.reduce(rows), fold(rows))
    # A NaN in one row, the others signed zeros: every position is a NaN in both.
    nan = True
    for position in range(4):
        rows = [zeros_by_pattern(shape, r, 0) for r in range(4)]
        rows[position] = np.full(shape, np.nan)
        nan = (
            nan
            and bool(np.isnan(np.minimum.reduce(rows)).all())
            and bool(np.isnan(fold(rows)).all())
        )
    return [alone, mixed, nan]


CHECKS = [
    "np.minimum returns the second of a tie",
    "np.maximum returns the second of a tie",
    "np.maximum(0.0, x) returns the second of a tie",
    "a NaN in either argument gives a NaN",
    "np.minimum.reduce equals the fold, each pattern alone",
    "np.minimum.reduce equals the fold, patterns mixed along the array",
    "np.minimum.reduce of four arrays with a NaN row gives NaN",
]


def measure() -> dict[str, object]:
    """Every check at every length and shape: the cases run and the departures."""
    shapes = [(n,) for n in LENGTHS] + list(SHAPES)
    cases = [0] * len(CHECKS)
    departures: list[list[str]] = [[] for _ in CHECKS]
    for shape in shapes:
        results = tie_rules(shape) + reduce_rules(shape)
        for i, ok in enumerate(results):
            cases[i] += 1
            if not ok:
                departures[i].append("x".join(map(str, shape)))
    return {
        "numpy": np.__version__,
        "lengths": [LENGTHS.start, LENGTHS.stop - 1],
        "shapes": [list(s) for s in SHAPES],
        "rows": [
            {"check": check, "cases": n, "departures": len(d), "first": d[:5]}
            for check, n, d in zip(CHECKS, cases, departures, strict=True)
        ],
    }


def main() -> int:
    """Measure, print one line per check, write the record."""
    record = measure()
    for row in record["rows"]:
        print(
            f"{row['cases']:5d} shapes, {row['departures']} departures: {row['check']}"
        )
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return 1 if any(row["departures"] for row in record["rows"]) else 0


if __name__ == "__main__":
    sys.exit(main())
