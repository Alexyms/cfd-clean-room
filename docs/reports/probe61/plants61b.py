"""Prompt 61b: review 61's three plants, then every plant of the 61b tests.

Usage:
    python plants61b.py review --tree TREE   # review 61 B1's plants, on the tests as they stand
    python plants61b.py all --tree TREE      # those and one per branch the 61b tests cover
    python plants61b.py pair --tree TREE     # item 2: the two pair-balance refusals

TREE is a worktree of the branch (results/builder61/P). The loop is plants61.py's
``run_plants``: a control run of the named tests on the unplanted file first, one
exact replacement, the tests, the file restored from git. Records go to
results/builder61b/plants_review.json, plants_all.json and plants_pair.json of
the main checkout.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

from plants61 import run_plants  # noqa: E402

Plant = tuple[str, str, str, str, list[str]]

TRANSPORT = "tests/test_solver_transport.py::TestVents"
WALLS = "tests/test_wall_functions.py::TestVents"
# The 61b tests: the shelf's orientations, the stretched pair, the registry.
NEW = [
    "tests/test_boundary_staggered.py::TestVentOrientations",
    "tests/test_momentum.py::test_the_shelf_vents_keep_their_velocity_through_the_prediction",
    "tests/test_wall_functions.py::TestVentOrientations",
    "tests/test_boundary_concentration.py::"
    "test_the_shelf_vents_scalar_conditions_in_every_orientation",
    "tests/test_solver_transport.py::test_the_budget_books_every_vent_orientation_in_and_out",
    "tests/test_solver_transport.py::TestStretchedRecirculation",
    "tests/test_boundary_registry.py::TestObstacleFaces",
]


def _term(sign: str, condition: str, adv: str) -> tuple[str, str]:
    """One term of _vent_fluxes as it stands, and the term zeroed."""
    term = f"np.sum(np.where({condition}, {adv}, 0.0))"
    return f"{sign}{term}", f"{sign}0.0"


# Review 61 B1's three plants, against the vent tests as they stood at 8c756d1.
REVIEW: list[Plant] = [
    (
        "review (1): solid_high_v and solid_low_v swapped in the v terms of out",
        "src/solver_transport.py",
        "            + np.sum(np.where(vv & self._solid_high_v & (flux_v > 0.0), adv_v, 0.0))\n"
        "            - np.sum(np.where(vv & self._solid_low_v & (flux_v < 0.0), adv_v, 0.0))\n"
        "        )\n"
        "        return float(into), float(out)",
        "            + np.sum(np.where(vv & self._solid_low_v & (flux_v > 0.0), adv_v, 0.0))\n"
        "            - np.sum(np.where(vv & self._solid_high_v & (flux_v < 0.0), adv_v, 0.0))\n"
        "        )\n"
        "        return float(into), float(out)",
        [TRANSPORT, WALLS],
    ),
    (
        "review (2): a vent below a cell left its north wall",
        "src/turbulence.py",
        "        north[:-1, :] = solid[1:, :] & ~vents.normal_v[1:-1, :]",
        "        north[:-1, :] = solid[1:, :]",
        [TRANSPORT, WALLS],
    ),
    (
        "review (3): the recirculated mean unweighted",
        "src/solver_transport.py",
        "            drawn = (\n"
        "                np.abs(flux_u[pair.intake_u]).sum()\n"
        "                + np.abs(flux_v[pair.intake_v]).sum()\n"
        "            )\n"
        "            carried = 0.0\n"
        "            if drawn > 0.0:\n"
        "                carried = (\n"
        "                    np.sum(np.abs(flux_u[pair.intake_u]) * room_u[pair.intake_u])\n"
        "                    + np.sum(np.abs(flux_v[pair.intake_v]) * room_v[pair.intake_v])\n"
        "                ) / drawn",
        "            drawn = float(pair.intake_u.sum() + pair.intake_v.sum())\n"
        "            carried = 0.0\n"
        "            if drawn > 0.0:\n"
        "                carried = (\n"
        "                    np.sum(room_u[pair.intake_u]) + np.sum(room_v[pair.intake_v])\n"
        "                ) / drawn",
        [TRANSPORT, WALLS],
    ),
]

_TERMS = [
    (
        "into: a discharge on a right face",
        "            ",
        "vu & self._solid_low_u & (flux_u > 0.0)",
        "adv_u",
    ),
    (
        "into: a discharge on a left face",
        "            - ",
        "vu & self._solid_high_u & (flux_u < 0.0)",
        "adv_u",
    ),
    (
        "into: a discharge on a bottom face",
        "            - ",
        "vv & self._solid_high_v & (flux_v < 0.0)",
        "adv_v",
    ),
    (
        "out: an exhaust on a bottom face",
        "            + ",
        "vv & self._solid_high_v & (flux_v > 0.0)",
        "adv_v",
    ),
    (
        "out: an exhaust on a top face",
        "            - ",
        "vv & self._solid_low_v & (flux_v < 0.0)",
        "adv_v",
    ),
]

# Each plant runs the review's three against the 61b tests too, so the record
# shows which new test catches it.
ALL: list[Plant] = [
    (f"{name.split(':')[0]} against the 61b tests", rel, old, new, NEW)
    for name, rel, old, new, _ in REVIEW
]
ALL += [
    (
        f"_vent_fluxes {label} not booked",
        "src/solver_transport.py",
        *_term(sign, condition, adv),
        NEW,
    )
    for label, sign, condition, adv in _TERMS
]
ALL += [
    (
        "turbulence: a vent on a right face left its cell's west wall",
        "src/turbulence.py",
        "        west[:, 1:] = solid[:, :-1] & ~vents.normal_u[:, 1:-1]",
        "        west[:, 1:] = solid[:, :-1]",
        NEW,
    ),
    (
        "turbulence: an exhaust on a v face reads the cell inside the obstacle",
        "src/turbulence.py",
        "cells = np.where(self._solid[rows, cols], rows - 1, rows)",
        "cells = np.where(self._solid[rows, cols], rows, rows)",
        NEW,
    ),
    (
        "staggered: a bottom face read one row too low",
        "src/staggered.py",
        "        rows, cols = np.nonzero(cells[1:, :] & open_cell[:-1, :])\n"
        "        rows = rows + 1",
        "        rows, cols = np.nonzero(cells[1:, :] & open_cell[:-1, :])",
        NEW,
    ),
    (
        "staggered: a bottom face plays the bottom edge",
        "src/staggered.py",
        '    "bottom": "top",',
        '    "bottom": "bottom",',
        NEW,
    ),
    (
        "registry: covers_face ignores the obstacle's name",
        "src/boundary_registry.py",
        "or spec.obstacle != obstacle ",
        "",
        NEW,
    ),
]


# Item 2 (review 61 S6): a recirculating pair of unequal flow refused.
PAIR: list[Plant] = [
    (
        "config: a pair of unequal flow accepted at load",
        "src/config.py",
        "        if abs(returned - drawn) > BALANCE_TOLERANCE * max(returned, drawn):",
        "        if False:",
        ["tests/test_config.py::test_a_recirculating_pair_of_unequal_flow_is_refused"],
    ),
    (
        "velocity: a pair of unequal flow on the mesh accepted",
        "src/boundary_staggered.py",
        "    _check_recirculated_flows(runs)\n",
        "",
        [
            "tests/test_boundary_staggered.py::TestVents::"
            "test_a_recirculating_pair_must_balance_on_the_mesh_faces"
        ],
    ),
]


def main() -> None:
    """Run the chosen list."""
    parser = argparse.ArgumentParser()
    parser.add_argument("which", choices=("review", "all", "pair"))
    parser.add_argument("--tree", required=True)
    args = parser.parse_args()
    plants = {"review": REVIEW, "all": REVIEW + ALL, "pair": PAIR}[args.which]
    out = ROOT / "results" / "builder61b" / f"plants_{args.which}.json"
    run_plants(plants, Path(args.tree).resolve(), out)


if __name__ == "__main__":
    main()
