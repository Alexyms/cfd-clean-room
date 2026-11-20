"""Prompt 89b's planted defects: round 0's, aimed at the flux in its block form, and new ones.

Read by ``probe89b.py plants``. Same form as ``variants89.py``: each plant is a
list of edits to a copy of the checkout's ``core/``, the exact text replaced
once or the plant refused. Round 0's plants are ``variants89.PLANTS`` except two
that edited the scalar loop the flux no longer is (``flux_end_node_ignores_inflow``
and ``flux_far_solid_dropped``), which are written again against the block.
The new ones are planted against what this round added: the stop's NaN clause,
the refusal of ``masks_where_not``, the grid's SOLID masks and the branch-free
clamp.
"""

import variants89
from variants89 import CPP, GRID_CPP, HPP, LANES, Edit

# Round 0's plants are the only ones that need candidates' machinery, and
# `probe89.plants()` looks the table up: there are no candidates this round.
CANDIDATES: dict[str, dict[str, object]] = {}

AGAINST_THE_BLOCK: dict[str, tuple[str, list[Edit]]] = {
    "flux_end_node_ignores_inflow": (
        "a line's low boundary node read as the cell's value whatever the flux",
        [
            (
                CPP,
                "            at_low ? low_boundary_node(in.end_flux_low[i], in.end_inflow_low[i], low)\n"
                "                   : in.far_low[i];",
                "            at_low ? low : in.far_low[i];",
            )
        ],
    ),
    "flux_far_solid_dropped": (
        "a far node inside a SOLID cell read as it is",
        [
            (
                CPP,
                "        const lanes::mask far_solid = (positive & solid_below) | (~positive & solid_above);",
                "        const lanes::mask far_solid = lanes::none;",
            )
        ],
    ),
}

ROUND_0 = [
    (name, *AGAINST_THE_BLOCK.get(name, (what, edits)))
    for name, what, edits in variants89.PLANTS
]

NEW: list[tuple[str, str, list[Edit]]] = [
    (
        "outside_stop_nan_passes",
        "the stop met where the residual is not above it, so a NaN residual meets it",
        [
            (
                CPP,
                "return lanes::mask_of(!(std::fabs(r) <= stop));",
                "return lanes::mask_of(std::fabs(r) > stop);",
            )
        ],
    ),
    (
        "masks_where_not_unchecked",
        "masks_where_not reads the flags past the masks' length without a refusal",
        [
            (
                LANES,
                "    if (flags.size() != out.size()) {\n"
                '        throw std::invalid_argument("masks_where_not: the flags and the masks differ in length");\n'
                "    }\n",
                "",
            )
        ],
    ),
    (
        "solid_flags_inverted",
        "the grid's SOLID masks set where a cell is not SOLID",
        [
            (
                GRID_CPP,
                "lanes::mask_of(cell_type_[cell] == input.solid)",
                "lanes::mask_of(cell_type_[cell] != input.solid)",
            )
        ],
    ),
    (
        "limited_zero_difference_unmasked",
        "the clamp's value kept where the downstream difference is zero",
        [
            (
                HPP,
                "    return lanes::select(defined, c_c + 0.5 * psi * d_down, c_c);",
                "    return c_c + 0.5 * psi * d_down;",
            )
        ],
    ),
    (
        "limited_nan_difference_is_zero",
        "a NaN downstream difference read as a zero one",
        [
            (
                HPP,
                "    const lanes::mask defined = lanes::mask_of(d_down != 0.0);",
                "    const lanes::mask defined = lanes::mask_of(d_down > 0.0) | lanes::mask_of(d_down < 0.0);",
            )
        ],
    ),
    (
        "np_minimum_takes_the_first_of_a_tie",
        "np_minimum returns the first of two equal values, where NumPy returns the second",
        [
            (
                HPP,
                "    return lanes::select(lanes::mask_of(a < b) | lanes::mask_of(a != a), a, b);",
                "    return lanes::select(lanes::mask_of(a <= b) | lanes::mask_of(a != a), a, b);",
            )
        ],
    ),
    (
        "flux_weights_sign_swapped",
        "the far node's QUICK weight taken from the other direction's weights",
        [
            (
                CPP,
                "lanes::select(positive, in.w_pos[0][w], in.w_neg[0][w]) * far +",
                "lanes::select(positive, in.w_neg[0][w], in.w_pos[0][w]) * far +",
            )
        ],
    ),
    (
        "flux_far_node_from_the_other_side",
        "the far node read from the downstream side of the face",
        [
            (
                CPP,
                "        const double far = lanes::select(positive, below, above);",
                "        const double far = lanes::select(positive, above, below);",
            )
        ],
    ),
    (
        "flux_far_solid_from_the_other_side",
        "the far cell's SOLID flag read from the downstream side of the face",
        [
            (
                CPP,
                "        const lanes::mask far_solid = (positive & solid_below) | (~positive & solid_above);",
                "        const lanes::mask far_solid = (positive & solid_above) | (~positive & solid_below);",
            )
        ],
    ),
]

# The control first and last, as in round 0.
PLANTS: list[tuple[str, str, list[Edit]]] = [
    ROUND_0[0],
    *ROUND_0[1:-1],
    *NEW,
    ROUND_0[-1],
]
