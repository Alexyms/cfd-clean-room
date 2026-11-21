"""Prompt 90's planted defects: each a list of edits to a copy of the checkout's ``core/``.

Read by ``probe90.py plants``. An edit is ``(file, old, new)``: the exact text
``old`` replaced by ``new``, found exactly once, or the plant is refused. The
first eight are the prompt's; the rest are planted against the other rules the
port copies. Prompt 90b's round 1 adds the plants test 90 found no test could
see (one solver through several intervals, the recirculation in each
orientation, the breach test's guard) and review 90's three for a second
``set_interval``; the later ones are planted against the refusals it asked for.
"""

TRANSPORT = "src/transport.cpp"
SCALAR = "src/scalar.cpp"
REDUCE = "src/reduce.cpp"
BINDING = "python/bindings.cpp"

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
    # --- one solver through several intervals and steps (test 90 B1, review 90 S1) ---
    (
        "deposit_step_stale",
        "a second set_step keeps the first step's deposition per face",
        [
            (
                TRANSPORT,
                "                q.deposit_dt[d][f] = q.deposit[d][f] * dt;",
                "                if (q.deposit_dt[d][f] == 0.0) {\n"
                "                    q.deposit_dt[d][f] = q.deposit[d][f] * dt;\n"
                "                }",
            )
        ],
    ),
    (
        "source_kept_across_intervals",
        "a class's source of an earlier interval kept when the next has none",
        [
            (
                TRANSPORT,
                "        q.has_source = !input.sources.empty() && !input.sources[k].empty();\n"
                "        q.rate.clear();\n"
                "        q.dt_rate.clear();\n"
                "        q.source_total = 0.0;\n"
                "        if (q.has_source) {",
                "        const bool given = !input.sources.empty() && !input.sources[k].empty();\n"
                "        if (given || !q.has_source) {\n"
                "            q.rate.clear();\n"
                "            q.dt_rate.clear();\n"
                "            q.source_total = 0.0;\n"
                "        }\n"
                "        q.has_source = q.has_source || given;\n"
                "        if (given) {",
            )
        ],
    ),
    (
        "budgets_zeroed_by_interval",
        "set_interval zeroes the budgets",
        [
            (
                TRANSPORT,
                "    interval_ = true;\n    stepped_ = false;\n",
                "    interval_ = true;\n    stepped_ = false;\n"
                "    budgets_.assign(budgets_.size(), budget{});\n",
            )
        ],
    ),
    (
        "interval_leaves_stepped_set",
        "set_interval leaves the last interval's step in force, so a step runs on zeroed per-step arrays",
        [
            (
                TRANSPORT,
                "    interval_ = true;\n    stepped_ = false;\n",
                "    interval_ = true;\n",
            )
        ],
    ),
    (
        "interval_keeps_old_vents",
        "a second interval without vents keeps the first interval's vent faces",
        [
            (
                TRANSPORT,
                "    vents_ = !each[0].vents[0].empty();\n",
                "    vents_ = vents_ || !each[0].vents[0].empty();\n",
            ),
            (
                TRANSPORT,
                "            vent_flags_[d][f] = vents_ && each[0].vents[d][f];\n",
                "            if (!each[0].vents[d].empty()) {\n"
                "                vent_flags_[d][f] = each[0].vents[d][f];\n"
                "            }\n",
            ),
        ],
    ),
    (
        "inflow_kept_across_intervals",
        "the first interval's inflow values kept by the next",
        [
            (
                TRANSPORT,
                "            q.inflow[d].assign(c.inflow[d].begin(), c.inflow[d].end());",
                "            if (q.inflow[d].empty()) {\n"
                "                q.inflow[d].assign(c.inflow[d].begin(), c.inflow[d].end());\n"
                "            }",
            )
        ],
    ),
    (
        "eddy_kept_across_intervals",
        "the face eddy viscosity of an earlier interval kept on a laminar one",
        [
            (
                TRANSPORT,
                "    nu_cells_.clear();\n"
                "    for (int d = 0; d < D; ++d) {\n"
                "        nu_faces_[d].clear();\n"
                "    }\n"
                "    if (turbulent) {",
                "    if (turbulent) {",
            ),
            (
                TRANSPORT,
                "                q.conductance[d][f] = turbulent ? (diffusion",
                "                q.conductance[d][f] = !nu_faces_[d].empty() ? (diffusion",
            ),
        ],
    ),
    # --- the recirculation in every orientation (test 90 B2) ---
    (
        "recirculated_low_cell_only",
        "the room value of an intake face read from its low side only",
        [
            (
                TRANSPORT,
                "(low == no_cell ? 0.0 : c_[low]) + (high == no_cell ? 0.0 : c_[high]);",
                "(low == no_cell ? 0.0 : c_[low]);",
            )
        ],
    ),
    (
        "recirculated_high_cell_only",
        "the room value of an intake face read from its high side only",
        [
            (
                TRANSPORT,
                "(low == no_cell ? 0.0 : c_[low]) + (high == no_cell ? 0.0 : c_[high]);",
                "(high == no_cell ? 0.0 : c_[high]);",
            )
        ],
    ),
    (
        "recirculated_passes_all",
        "the fraction a discharge passes dropped: it carries the whole drawn value",
        [
            (
                TRANSPORT,
                "        const double value = air.passed * carried;",
                "        const double value = carried;",
            )
        ],
    ),
    (
        "drawn_zero_divides",
        "the guard for an intake that draws nothing broken: zero is divided by zero",
        [
            (
                TRANSPORT,
                "        if (air.drawn > 0.0) {",
                "        if (air.drawn >= 0.0) {",
            )
        ],
    ),
    (
        "recirculated_y_dropped",
        "the intake's y faces left out of the weighted sum, though its drawn flux counts them",
        [
            (
                TRANSPORT,
                "                total = d == 0 ? sum : total + sum;",
                "                total = d == 0 ? sum : total;",
            )
        ],
    ),
    (
        "drawn_ignores_y",
        "the intake's drawn flux without its y faces",
        [
            (
                TRANSPORT,
                "                    drawn = drawn + reduce::pairwise_sum(air.weight[d]);",
                "                    drawn = drawn + 0.0;",
            )
        ],
    ),
    # --- the breach test's guard and the refusal of buffered_sum (review 90 S2, test 90 S1) ---
    (
        "breach_without_zero_guard",
        "the breach test without its leftover != 0 clause: an empty room breaches at its first step",
        [
            (
                TRANSPORT,
                "    if (b.leftover != 0.0 && std::fabs(b.leftover) >= keys_.limit * supplied) {",
                "    if (std::fabs(b.leftover) >= keys_.limit * supplied) {",
            )
        ],
    ),
    # --- dt and size_class read as the reference reads them (review 90 S3 and S4, test 90 S4) ---
    (
        "dt_python_bool_unchecked",
        "a Python bool dt read as a number, 1.0",
        [
            (
                BINDING,
                '        nb::isinstance<bool>(dt) || PyObject_IsInstance(dt.ptr(), numpy.attr("bool_").ptr()) == 1 ||\n',
                '        PyObject_IsInstance(dt.ptr(), numpy.attr("bool_").ptr()) == 1 ||\n',
            )
        ],
    ),
    (
        "dt_numpy_bool_unchecked",
        "a NumPy bool dt read as a number, 1.0",
        [
            (
                BINDING,
                '        nb::isinstance<bool>(dt) || PyObject_IsInstance(dt.ptr(), numpy.attr("bool_").ptr()) == 1 ||\n',
                "        nb::isinstance<bool>(dt) ||\n",
            )
        ],
    ),
    (
        "dt_bool_array_unchecked",
        "a 0-d bool array dt read as a number, 1.0",
        [
            (
                BINDING,
                '        (PyObject_IsInstance(dt.ptr(), numpy.attr("ndarray").ptr()) == 1 &&\n'
                '         nb::cast<std::string>(nb::getattr(nb::getattr(dt, "dtype"), "kind")) == "b");\n',
                "        false;\n",
            )
        ],
    ),
    (
        "dt_any_array_refused_as_bool",
        "every ndarray dt refused as a bool, a 0-d float array included (test 90 S4)",
        [
            (
                BINDING,
                'nb::cast<std::string>(nb::getattr(nb::getattr(dt, "dtype"), "kind")) == "b");',
                'nb::cast<std::string>(nb::getattr(nb::getattr(dt, "dtype"), "kind")) != "");',
            )
        ],
    ),
    (
        "stable_dt_class_unchecked",
        "stable_dt takes any class that converts to an index, a bool included (review 90 S4)",
        [
            (
                BINDING,
                "                return t.solver.stable_dt(read_class(size_class, t.solver.classes()));",
                "                return t.solver.stable_dt(nb::cast<std::size_t>(size_class));",
            )
        ],
    ),
    (
        "buffered_sum_partial_rows_allowed",
        "buffered_sum takes terms that are not a whole number of rows",
        [
            (
                REDUCE,
                "    if (row == 0 || values.size() % row != 0) {",
                "    if (row == 0) {",
            )
        ],
    ),
    (
        "buffered_sum_zero_row_unchecked",
        "buffered_sum divides by a row of zero",
        [
            (
                REDUCE,
                "    if (row == 0 || values.size() % row != 0) {",
                "    if (values.size() % row != 0) {",
            )
        ],
    ),
]
