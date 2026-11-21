"""Prompt 90's planted defects: each a list of edits to a copy of the checkout's ``core/``.

Read by ``probe90.py plants``. An edit is ``(file, old, new)``: the exact text
``old`` replaced by ``new``, found exactly once, or the plant is refused. The
first eight are the prompt's; the rest are planted against the other rules the
port copies.
"""

TRANSPORT = "src/transport.cpp"
SCALAR = "src/scalar.cpp"

Edit = tuple[str, str, str]

PLANTS: list[tuple[str, str, list[Edit]]] = [
    ("control", "no edit: every test passes", []),
    (
        "boundary_sum_order",
        "a domain boundary's column summed left to right, not in NumPy's pairwise order",
        [
            (
                TRANSPORT,
                "            return reduce::pairwise_sum(std::span<const double>(terms_.data(), lines.size()));",
                "            double total = 0.0;\n"
                "            for (std::size_t i = 0; i < lines.size(); ++i) {\n"
                "                total += terms_[i];\n"
                "            }\n"
                "            return total;",
            )
        ],
    ),
    (
        "vents_into_out_swapped",
        "the vents' flow into the room booked as outflow and their flow out as inflow",
        [
            (
                TRANSPORT,
                "        inflow = inflow + into;\n        outflow = outflow + out;",
                "        inflow = inflow + out;\n        outflow = outflow + into;",
            )
        ],
    ),
    (
        "recirculated_from_new_field",
        "a discharge's value from the field the step returns, not the one it starts from",
        [
            (
                TRANSPORT,
                "    recirculate(k);\n    for (int d = 0; d < D; ++d) {\n",
                "    for (int d = 0; d < D; ++d) {\n",
            ),
            (
                TRANSPORT,
                "    b.leftover = b.leftover + result.leftover;\n",
                "    b.leftover = b.leftover + result.leftover;\n"
                "    std::copy(out.begin(), out.end(), c_.begin());\n"
                "    recirculate(k);\n",
            ),
        ],
    ),
    (
        "deposition_wrong_surface",
        "each domain surface's deposition booked to the next slot (floor to ceiling, and so on)",
        [
            (
                TRANSPORT,
                "        b.deposited[s] = b.deposited[s] + total;",
                "        b.deposited[(s + 1) % 3] = b.deposited[(s + 1) % 3] + total;",
            )
        ],
    ),
    (
        "content_with_solid",
        "the content summed over every cell, SOLID ones included",
        [
            (
                TRANSPORT,
                "    return reduce::pairwise_sum(std::span<const double>(terms_.data(), live_count_));",
                "    for (std::size_t i = 0; i < c.size(); ++i) {\n"
                "        terms_[i] = c[i] * volume_[i];\n"
                "    }\n"
                "    return reduce::pairwise_sum(std::span<const double>(terms_.data(), c.size()));",
            )
        ],
    ),
    (
        "breach_strict",
        "the breach test's comparison strict: the leftover must exceed the limit times the supply",
        [
            (
                TRANSPORT,
                "std::fabs(b.leftover) >= keys_.limit * supplied",
                "std::fabs(b.leftover) > keys_.limit * supplied",
            )
        ],
    ),
    (
        "bound_before_source",
        "the step's bound from the supply before the step's source is booked",
        [
            (
                TRANSPORT,
                "    if (q.has_source) {\n        b.source = b.source + q.source_total * dt_;\n    }\n    book_boundaries(k, b);\n",
                "    book_boundaries(k, b);\n",
            ),
            (
                TRANSPORT,
                "    const double bound = keys_.share * keys_.limit * supplied * dt_ / keys_.horizon;\n",
                "    const double bound = keys_.share * keys_.limit * supplied * dt_ / keys_.horizon;\n"
                "    if (q.has_source) {\n        b.source = b.source + q.source_total * dt_;\n    }\n",
            ),
        ],
    ),
    (
        "eddy_viscosity_unmasked",
        "the face eddy viscosity from the field as handed in, SOLID cells not zeroed",
        [
            (
                TRANSPORT,
                "            nu_cells_[i] = lanes::keep(fluid_[i], input.eddy_viscosity[i]);",
                "            nu_cells_[i] = input.eddy_viscosity[i];",
            )
        ],
    ),
    (
        "deposition_without_area",
        "the deposition per face from the deposition velocity alone, without the face's area",
        [
            (
                TRANSPORT,
                "                q.deposit[d][f] = c.deposition[d][f] * area_[d][f];",
                "                q.deposit[d][f] = c.deposition[d][f];",
            )
        ],
    ),
    (
        "stable_dt_from_the_air",
        "stable_dt from the air's face velocities, without the settling and the drift",
        [
            (
                TRANSPORT,
                "                double rate = scalar::np_maximum(std::fabs(q.velocity[0][w]),\n"
                "                                                 std::fabs(q.velocity[0][w + 1])) /",
                "                double rate = scalar::np_maximum(std::fabs(input.faces[0][w]),\n"
                "                                                 std::fabs(input.faces[0][w + 1])) /",
            ),
            (
                TRANSPORT,
                "                    rate = rate + scalar::np_maximum(std::fabs(q.velocity[d][low]),\n"
                "                                                     std::fabs(q.velocity[d][high])) /",
                "                    rate = rate + scalar::np_maximum(std::fabs(input.faces[d][low]),\n"
                "                                                     std::fabs(input.faces[d][high])) /",
            ),
        ],
    ),
    (
        "divergence_low_first",
        "the divergence's further directions taken low face first, then high",
        [
            (
                TRANSPORT,
                "            divergence = divergence + in.adv_high[d][i];\n"
                "            divergence = divergence - in.adv_low[d][i];",
                "            divergence = divergence - in.adv_low[d][i];\n"
                "            divergence = divergence + in.adv_high[d][i];",
            )
        ],
    ),
    (
        "obstacle_slice_flat",
        "the interior x faces' deposition summed pairwise over the slice, not through NumPy's buffer",
        [
            (
                TRANSPORT,
                "        const double sum =\n"
                "            d == D - 1 ? reduce::pairwise_sum(slice) : reduce::buffered_sum(slice, row);",
                "        const double sum = reduce::pairwise_sum(slice);",
            )
        ],
    ),
    (
        "leftover_not_booked",
        "the step's leftover not added to the budget",
        [
            (
                TRANSPORT,
                "    b.leftover = b.leftover + result.leftover;\n",
                "",
            )
        ],
    ),
    (
        "drift_on_every_face",
        "v_ext added on every face, not only the interior faces between two non-SOLID cells",
        [
            (
                TRANSPORT,
                "velocity = velocity + lanes::select(inner_[d][f], input.drift[k][d][f], 0.0);",
                "velocity = velocity + input.drift[k][d][f];",
            )
        ],
    ),
    (
        "held_length_unchecked",
        "set_masks reads held without checking its length",
        [
            (
                SCALAR,
                "    if (in.held.size() != scratch.free.size()) {\n"
                '        throw std::invalid_argument("implicit_step: the held flags and the cells differ in length");\n'
                "    }\n",
                "",
            )
        ],
    ),
]
