"""Checks of prompt 82's enumeration against brute force and the built set cover.

    python -m pytest docs/reports/probe82/test_probe82.py -q

Small random tables only: the measurement's own tables are in the queue.
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import probe82  # noqa: E402 -- follows sys.path.insert

from src import placement  # noqa: E402 -- probe82 put the repository on the path


def brute(masks: list[int], target: int, slots: int) -> bool:
    """Whether some combination of at most SLOTS masks contains TARGET."""
    for k in range(slots + 1):
        for combo in itertools.combinations(masks, k):
            union = 0
            for m in combo:
                union |= m
            if union & target == target:
                return True
    return False


def random_table(
    rng: np.random.Generator, units: int, n: int, plume: bool
) -> np.ndarray:
    """A [n, units] latency table: independent noise, or fronts from a point per unit."""
    if plume:
        where = rng.uniform(0, 1, n)
        source = rng.uniform(0, 1, units)
        speed = rng.uniform(0.05, 0.5, units)
        table = np.abs(where[:, None] - source[None, :]) / speed[None, :]
        table = np.round(table + rng.uniform(0, 0.3, (n, units)), 2)
        table[table > 4.0] = np.inf
    else:
        table = np.round(rng.uniform(0, 5, (n, units)), 1)
        table[rng.uniform(0, 1, (n, units)) < 0.4] = np.inf
    return np.ascontiguousarray(table)


@pytest.mark.parametrize("seed", range(40))
def test_feasibility_matches_brute_force(seed: int) -> None:
    """Feasibility.covers agrees with every combination tried, slots 0 to 4."""
    rng = np.random.default_rng(seed)
    units = int(rng.integers(4, 11))
    masks = np.unique(rng.integers(0, 1 << units, int(rng.integers(3, 25))))
    maximal = placement._maximal(masks)
    full = (1 << units) - 1
    targets = [full, *map(int, rng.integers(0, 1 << units, 6))]
    for slots in range(5):
        f = probe82.Feasibility(maximal)
        for target in targets:
            assert f.covers(target, slots) == brute(
                [int(m) for m in maximal], target, slots
            )


@pytest.mark.parametrize("seed", range(10))
def test_fewest_matches_the_subset_table(seed: int) -> None:
    """Fewest gives the built _fewest's value on every subset."""
    rng = np.random.default_rng(100 + seed)
    units = int(rng.integers(4, 10))
    count = int(rng.integers(1, 5))
    masks = np.unique(rng.integers(0, 1 << units, int(rng.integers(3, 20))))
    maximal = placement._maximal(masks)
    table = placement._fewest(maximal, units, count)
    fewest = probe82.Fewest(probe82.Feasibility(maximal), count)
    for subset in range(1 << units):
        assert fewest[subset] == table[subset]


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("plume", [True, False])
def test_enumeration_returns_the_built_layout(seed: int, plume: bool) -> None:
    """enumerate_cover returns the built set cover's layout, counts 1 to 4."""
    rng = np.random.default_rng(1000 + seed)
    units = int(rng.integers(4, 13))
    n = int(rng.integers(8, 40))
    table = random_table(rng, units, n, plume)
    reached = np.isfinite(table).any(axis=0)
    if not reached.any():
        return
    table = np.ascontiguousarray(table[:, reached])
    caps = np.full(table.shape[1], 10.0)
    one = probe82.enumerate_cover(table, caps, 1, {})
    argmin = placement._best_one(table, caps, placement.WORST_CASE)
    best = table[list(argmin)].max()
    assert (one.columns != ()) == np.isfinite(best)
    if one.columns:
        assert table[list(one.columns)].max() == best
    for count in range(2, 5):
        if count > n:
            continue
        built = placement._worst_case_cover(table, caps, count)
        mine = probe82.enumerate_cover(table, caps, count, {})
        assert mine.columns == built.columns
        assert (mine.distinct, mine.maximal) == (built.distinct, built.maximal)


def test_estimate_bounds_the_subset_arrays() -> None:
    """The set cover's estimate holds 2^units subsets at 45 bytes each at least."""
    assert probe82.estimate("cover", 20, 100) >= 45 * 2**20
    assert probe82.estimate("cover", 28, 2025) > probe82.MEMORY_LIMIT
