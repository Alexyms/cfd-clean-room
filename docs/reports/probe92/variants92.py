"""Prompt 92's kernel forms and planted defects: edits to a copy of the checkout's ``core/``.

Read by ``probe92.py forms`` and ``probe92.py plants``. An edit is ``(file,
old, new)``: the exact text ``old`` replaced by ``new``, found exactly once, or
the variant is refused. The first eight plants are the prompt's; the rest are
against the other rules the port copies, and one for each refusal of the core
and the binding, removed.
"""

MOMENTUM = "src/momentum.cpp"
BINDING = "python/bindings.cpp"

Edit = tuple[str, str, str]


def momentum(old: str, new: str) -> Edit:
    """An edit of momentum.cpp."""
    return (MOMENTUM, old, new)


def binding(old: str, new: str) -> Edit:
    """An edit of bindings.cpp."""
    return (BINDING, old, new)


# ---------------------------------------------------------------------------
# The kernels' forms

_CELL_BLOCK = """        const double f = in.flux[i];
        const mask positive = lanes::mask_of(f > 0.0);
        const double q_pos = along<C>(in.std_pos[0], i) * below +
                             along<C>(in.std_pos[1], i) * lo + along<C>(in.std_pos[2], i) * hi;
        const double q_neg = along<C>(in.std_neg[0], i) * above +
                             along<C>(in.std_neg[1], i) * hi + along<C>(in.std_neg[2], i) * lo;
        double q = lanes::select(positive, q_pos, q_neg);
        if constexpr (Obstacles) {
            const double alt_pos = along<C>(in.alt_pos[0], i) * above +
                                   along<C>(in.alt_pos[1], i) * lo +
                                   along<C>(in.alt_pos[2], i) * hi;
            const double alt_neg = along<C>(in.alt_neg[0], i) * below +
                                   along<C>(in.alt_neg[1], i) * hi +
                                   along<C>(in.alt_neg[2], i) * lo;
            q = lanes::select(positive & in.blocked_pos[i], alt_pos, q);
            q = lanes::select(~positive & in.blocked_neg[i], alt_neg, q);
        }
        out[i] = (q - lanes::select(positive, lo, hi)) * f;
"""
_CELL_LOOP = """        const double f = in.flux[i];
        double q = 0.0;
        double up = 0.0;
        if (f > 0.0) {
            up = lo;
            if (Obstacles && in.blocked_pos[i] != lanes::none) {
                q = along<C>(in.alt_pos[0], i) * above + along<C>(in.alt_pos[1], i) * lo +
                    along<C>(in.alt_pos[2], i) * hi;
            } else {
                q = along<C>(in.std_pos[0], i) * below + along<C>(in.std_pos[1], i) * lo +
                    along<C>(in.std_pos[2], i) * hi;
            }
        } else {
            up = hi;
            if (Obstacles && in.blocked_neg[i] != lanes::none) {
                q = along<C>(in.alt_neg[0], i) * below + along<C>(in.alt_neg[1], i) * hi +
                    along<C>(in.alt_neg[2], i) * lo;
            } else {
                q = along<C>(in.std_neg[0], i) * above + along<C>(in.std_neg[1], i) * hi +
                    along<C>(in.std_neg[2], i) * lo;
            }
        }
        out[i] = (q - up) * f;
"""
_CORNER_BLOCK = """        const double f = in.flux[i];
        const mask positive = lanes::mask_of(f > 0.0);
        const double q_pos = along<T>(in.pos[0], i) * below + along<T>(in.pos[1], i) * lo +
                             along<T>(in.pos[2], i) * hi;
        const double q_neg = along<T>(in.neg[0], i) * above + along<T>(in.neg[1], i) * hi +
                             along<T>(in.neg[2], i) * lo;
        double q = lanes::select(positive, q_pos, q_neg);
        if constexpr (Obstacles) {
            const double wall_pos =
                along<T>(in.wall_pos[0], i) * lo + along<T>(in.wall_pos[1], i) * hi;
            const double wall_neg =
                along<T>(in.wall_neg[0], i) * hi + along<T>(in.wall_neg[1], i) * lo;
            q = lanes::select(positive & in.far_wall_low[i], wall_pos, q);
            q = lanes::select(~positive & in.far_wall_high[i], wall_neg, q);
        }
        double flux = (q - lanes::select(positive, lo, hi)) * f;
        if constexpr (Obstacles) {
            flux = lanes::keep(~in.wall_face[i], flux);
        }
        out[i] = flux;
"""
_CORNER_LOOP = """        const double f = in.flux[i];
        if (Obstacles && in.wall_face[i] != lanes::none) {
            out[i] = 0.0;
            continue;
        }
        double q = 0.0;
        double up = 0.0;
        if (f > 0.0) {
            up = lo;
            if (Obstacles && in.far_wall_low[i] != lanes::none) {
                q = along<T>(in.wall_pos[0], i) * lo + along<T>(in.wall_pos[1], i) * hi;
            } else {
                q = along<T>(in.pos[0], i) * below + along<T>(in.pos[1], i) * lo +
                    along<T>(in.pos[2], i) * hi;
            }
        } else {
            up = hi;
            if (Obstacles && in.far_wall_high[i] != lanes::none) {
                q = along<T>(in.wall_neg[0], i) * hi + along<T>(in.wall_neg[1], i) * lo;
            } else {
                q = along<T>(in.neg[0], i) * above + along<T>(in.neg[1], i) * hi +
                    along<T>(in.neg[2], i) * lo;
            }
        }
        out[i] = (q - up) * f;
"""

# The walkers as loops of one face, for the branching bodies.
_STREAM_WALK = """    if (obstacles) {
        lanes::for_each_block(first, last, [&](auto cells, std::size_t m) {
            cell_deferred<C, Inner, true>(cells, run, m, out);
        });
    } else {
        lanes::for_each_block(first, last, [&](auto cells, std::size_t m) {
            cell_deferred<C, Inner, false>(cells, run, m, out);
        });
    }
"""
_STREAM_LOOP = """    for (std::size_t m = first; m < last; ++m) {
        if (obstacles) {
            cell_deferred<C, Inner, true>(lanes::cells<1>{}, run, m, out);
        } else {
            cell_deferred<C, Inner, false>(lanes::cells<1>{}, run, m, out);
        }
    }
"""
_CROSS_WALK = _STREAM_WALK.replace("cell_deferred<C", "corner_deferred<T")
_CROSS_LOOP = _STREAM_LOOP.replace("cell_deferred<C", "corner_deferred<T")

FORMS: list[tuple[str, str, list[Edit]]] = [
    ("head", "the kernels as they are", []),
    (
        "quick_loops",
        "the QUICK face values of both directions as loops of one face with a branch per choice",
        [
            momentum(_CELL_BLOCK, _CELL_LOOP),
            momentum(_CORNER_BLOCK, _CORNER_LOOP),
            momentum(_STREAM_WALK, _STREAM_LOOP),
            momentum(_CROSS_WALK, _CROSS_LOOP),
        ],
    ),
    (
        "obstacles_every_row",
        "the obstacle forms formed on every row, not on the rows that hold one",
        [
            momentum(
                "    const bool obstacles = o.row_blocked[cell / nx] != 0;",
                "    const bool obstacles = true;",
            ),
            momentum(
                "    const bool obstacles = w.row_walls[corner / w.corners.cells(0)] != 0;",
                "    const bool obstacles = true;",
            ),
        ],
    ),
    (
        "np_maximum",
        "the upwind coefficients' maximum by scalar::np_maximum, whose blocks stay scalar",
        [
            momentum(
                "    return lanes::keep(lanes::mask_of(!(x <= 0.0)), x);",
                "    return scalar::np_maximum(x, 0.0);",
            )
        ],
    ),
    (
        "two_scratches",
        "each component's equations in arrays of its own, not the two in turn in one set",
        [
            (
                "include/cfdcore/momentum.hpp",
                "    std::array<std::vector<double>, 2 * D> coefficient_;\n"
                "    std::vector<double> source_;\n"
                "    std::vector<double> relaxed_;\n"
                "    std::vector<double> iterate_;\n",
                "    std::array<std::vector<double>, 2 * D> coefficient_;\n"
                "    std::vector<double> source_;\n"
                "    std::vector<double> relaxed_;\n"
                "    std::vector<double> iterate_;\n"
                "    std::array<std::vector<double>, 2 * D> coefficient2_;\n"
                "    std::vector<double> source2_;\n"
                "    std::vector<double> relaxed2_;\n"
                "    std::vector<double> iterate2_;\n"
                "    void swap_scratch() {\n"
                "        coefficient_.swap(coefficient2_);\n"
                "        source_.swap(source2_);\n"
                "        relaxed_.swap(relaxed2_);\n"
                "        iterate_.swap(iterate2_);\n"
                "    }\n",
            ),
            momentum(
                "    for (auto* a : {&source_, &relaxed_, &iterate_}) {\n",
                "    for (auto& a : coefficient2_) {\n        a.assign(faces, 0.0);\n    }\n"
                "    for (auto* a : {&source_, &relaxed_, &iterate_, &source2_, &relaxed2_, &iterate2_}) {\n",
            ),
            momentum(
                "        assemble<C>(velocity, p, mu_eff, wall_mu, diagonal[C].data());\n"
                "        sweep<C>(velocity[C].data(), diagonal[C].data(), star[C].data());\n",
                "        if (C == 1) {\n            swap_scratch();\n        }\n"
                "        assemble<C>(velocity, p, mu_eff, wall_mu, diagonal[C].data());\n"
                "        sweep<C>(velocity[C].data(), diagonal[C].data(), star[C].data());\n"
                "        if (C == 1) {\n            swap_scratch();\n        }\n",
            ),
        ],
    ),
]


# ---------------------------------------------------------------------------
# The planted defects

PLANTS: list[tuple[str, str, list[Edit]]] = [
    ("control", "no edit: every test passes", []),
    # The prompt's eight.
    (
        "deferred_sign",
        "the deferred correction added to the source, not subtracted",
        [
            momentum(
                "double deferred = -net;",
                "double deferred = net;",
            )
        ],
    ),
    (
        "corner_means_swapped",
        "a corner's viscosity by the arithmetic mean across the flux and the harmonic along the face",
        [
            momentum(
                "        out[i] = a + (b - a) * (a * frac_b) / (frac_a * b + frac_b * a);",
                "        out[i] = a + (b - a) * frac_b + 0.0 * frac_a;",
            ),
            momentum(
                "        out[i] = a + (b - a) * along<C>(frac_high, i);",
                "        const double frac_b = along<C>(frac_high, i);\n"
                "        const double frac_a = 1.0 - frac_b;\n"
                "        out[i] = a + (b - a) * (a * frac_b) / (frac_a * b + frac_b * a);",
            ),
        ],
    ),
    (
        "wall_missed_at_corner",
        "the wall stencil only where the neighbour face lies between two SOLID cells, not at a corner",
        [
            momentum(
                "if (r + 1 < nt && solid_face(g, c, after(at, t))) {",
                "if (r + 1 < nt && g.solid(index_of(g.fields(), after(before(at, c), t))) &&\n"
                "            g.solid(index_of(g.fields(), after(at, t)))) {",
            ),
            momentum(
                "if (r > 0 && solid_face(g, c, before(at, t))) {",
                "if (r > 0 && g.solid(index_of(g.fields(), before(before(at, c), t))) &&\n"
                "            g.solid(index_of(g.fields(), before(at, t)))) {",
            ),
        ],
    ),
    (
        "vent_overwritten",
        "a vent's stored value overwritten by the sweep with zero",
        [
            momentum(
                "out[i] = lanes::select(unknown, next, lanes::keep(in.held[i], in.own[i]));",
                "out[i] = lanes::select(unknown, next, lanes::keep(lanes::none, in.own[i]));",
            )
        ],
    ),
    (
        "upwind_from_downstream",
        "the streamwise upwind coefficient taken from the downstream side",
        [
            momentum(
                "            times_area<C>(across, in.half_rho * (in.low[i] + in.high[i]), in.width_t, i);\n"
                "        flux[i] = f;\n"
                "        plus[i] = in.conductance[i] + positive_part(-f);\n"
                "        minus[i] = in.conductance[i] + positive_part(f);\n",
                "            times_area<C>(across, in.half_rho * (in.low[i] + in.high[i]), in.width_t, i);\n"
                "        flux[i] = f;\n"
                "        plus[i] = in.conductance[i] + positive_part(f);\n"
                "        minus[i] = in.conductance[i] + positive_part(-f);\n",
            )
        ],
    ),
    (
        "diagonal_without_mass_flux",
        "the diagonal's net mass flux dropped",
        [
            momentum(
                "lanes::keep(in.unknown[i], sum + net);",
                "lanes::keep(in.unknown[i], sum + 0.0 * net);",
            )
        ],
    ),
    (
        "relaxation_on_new_iterate",
        "the relaxation term of every sweep on that sweep's iterate, not on the call's input",
        [
            momentum(
                "b[i] = b[i] + deferred + pressure + in.relaxation * in.a_p[i] * in.own[i];",
                "b[i] = b[i] + deferred + pressure + 0.0 * (in.relaxation * in.a_p[i] * in.own[i]);",
            ),
            momentum(
                "    const double* a_p;\n    const mask* held;\n    const double* own;\n};\n\n// The iterate",
                "    const double* a_p;\n    const mask* held;\n    const double* own;\n"
                "    double relaxation;\n};\n\n// The iterate",
            ),
            momentum(
                "numerator = numerator + in.b[i];",
                "numerator = numerator + (in.b[i] + in.relaxation * in.a_p[i] * in.x[i]);",
            ),
            momentum(
                "            run.a_p = diagonal + row;\n",
                "            run.a_p = diagonal + row;\n            run.relaxation = relaxation_;\n",
            ),
        ],
    ),
    (
        "one_sweep_fewer",
        "one sweep fewer than asked, where more than one is",
        [
            momentum(
                "sweeps_(in.sweeps)",
                "sweeps_(std::max<std::int64_t>(in.sweeps - 1, 1))",
            )
        ],
    ),
    # The mass fluxes and the coefficients.
    (
        "upwind_from_downstream_t",
        "the transverse upwind coefficient taken from the downstream side",
        [
            momentum(
                "                                        in.high[i] * along<C>(in.width_high, i));\n"
                "        flux[i] = f;\n"
                "        plus[i] = in.conductance[i] + positive_part(-f);\n"
                "        minus[i] = in.conductance[i] + positive_part(f);\n",
                "                                        in.high[i] * along<C>(in.width_high, i));\n"
                "        flux[i] = f;\n"
                "        plus[i] = in.conductance[i] + positive_part(f);\n"
                "        minus[i] = in.conductance[i] + positive_part(-f);\n",
            )
        ],
    ),
    (
        "transverse_flux_by_mean",
        "the transverse mass flux from the mean of the two half cells, not width-weighted",
        [
            momentum(
                "in.half_rho * (in.low[i] * along<C>(in.width_low, i) +\n"
                "                                        in.high[i] * along<C>(in.width_high, i));",
                "in.half_rho * ((in.low[i] + in.high[i]) * 0.5 * (along<C>(in.width_low, i) +\n"
                "                                        along<C>(in.width_high, i)));",
            )
        ],
    ),
    (
        "diagonal_order",
        "the diagonal's neighbour sum added in another order",
        [
            momentum(
                "high_side<T>(in.cross[slot_of(C, T)].plus, in.cross[slot_of(C, T)].plus_above, i) +\n"
                "                in.cross[slot_of(C, T)].minus[i]),",
                "(high_side<T>(in.cross[slot_of(C, T)].plus, in.cross[slot_of(C, T)].plus_above, i) +\n"
                "                in.cross[slot_of(C, T)].minus[i])),",
            )
        ],
    ),
    (
        "flux_order",
        "the diagonal's net flux added in another order",
        [
            momentum(
                "high_side<T>(in.cross[slot_of(C, T)].flux, in.cross[slot_of(C, T)].flux_above, i) -\n"
                "                in.cross[slot_of(C, T)].flux[i]),",
                "(high_side<T>(in.cross[slot_of(C, T)].flux, in.cross[slot_of(C, T)].flux_above, i) -\n"
                "                in.cross[slot_of(C, T)].flux[i])),",
            )
        ],
    ),
    (
        "open_high_edge_kept",
        "a zero-gradient high edge's coefficient left in the diagonal",
        [
            momentum(
                "        beside(corners.low->plus.data() + 1, w.high_known, w.high_value, toward_high, nx - 1, nx);",
                "        beside(corners.low->plus.data() + 1, std::vector<mask>(w.high_known.size(), lanes::all),\n"
                "               w.high_value, toward_high, nx - 1, nx);",
            ),
            momentum(
                "            beside(corners.high->plus.data(), w.high_known, w.high_value, toward_high, first,",
                "            beside(corners.high->plus.data(), std::vector<mask>(w.high_known.size(), lanes::all),\n"
                "                   w.high_value, toward_high, first,",
            ),
        ],
    ),
    (
        "open_low_edge_kept",
        "a zero-gradient low edge's coefficient left in the diagonal",
        [
            momentum(
                "        beside(corners.low->minus.data(), w.low_known, w.low_value, toward_low, 0, 1);",
                "        beside(corners.low->minus.data(), std::vector<mask>(w.low_known.size(), lanes::all),\n"
                "               w.low_value, toward_low, 0, 1);",
            ),
            momentum(
                "            beside(corners.low->minus.data(), w.low_known, w.low_value, toward_low, first, nx_);",
                "            beside(corners.low->minus.data(), std::vector<mask>(w.low_known.size(), lanes::all),\n"
                "                   w.low_value, toward_low, first, nx_);",
            ),
        ],
    ),
    (
        "high_edge_value_dropped",
        "a Dirichlet high edge's value left out of the source",
        [
            momentum(
                "        beside(corners.low->plus.data() + 1, w.high_known, w.high_value, toward_high, nx - 1, nx);",
                "        beside(corners.low->plus.data() + 1, w.high_known, zeros_, toward_high, nx - 1, nx);",
            ),
            momentum(
                "            beside(corners.high->plus.data(), w.high_known, w.high_value, toward_high, first,",
                "            beside(corners.high->plus.data(), w.high_known, zeros_, toward_high, first,",
            ),
            momentum(
                "    zeros_.assign(nx_ + 1, 0.0);",
                "    zeros_.assign(nx_ + g.fields().cell_count() + 1, 0.0);",
            ),
        ],
    ),
    (
        "low_edge_value_dropped",
        "a Dirichlet low edge's value left out of the source",
        [
            momentum(
                "        beside(corners.low->minus.data(), w.low_known, w.low_value, toward_low, 0, 1);",
                "        beside(corners.low->minus.data(), w.low_known, zeros_, toward_low, 0, 1);",
            ),
            momentum(
                "            beside(corners.low->minus.data(), w.low_known, w.low_value, toward_low, first, nx_);",
                "            beside(corners.low->minus.data(), w.low_known, zeros_, toward_low, first, nx_);",
            ),
            momentum(
                "    zeros_.assign(nx_ + 1, 0.0);",
                "    zeros_.assign(nx_ + g.fields().cell_count() + 1, 0.0);",
            ),
        ],
    ),
    (
        "edges_swapped",
        "the high edge's condition read at the low one",
        [
            momentum(
                "        beside(corners.low->minus.data(), w.low_known, w.low_value, toward_low, 0, 1);",
                "        beside(corners.low->minus.data(), w.high_known, w.high_value, toward_low, 0, 1);",
            ),
            momentum(
                "            beside(corners.low->minus.data(), w.low_known, w.low_value, toward_low, first, nx_);",
                "            beside(corners.low->minus.data(), w.high_known, w.high_value, toward_low, first, nx_);",
            ),
        ],
    ),
    (
        "obstacle_neighbour_kept_high",
        "the coefficient toward an obstacle's face on the high side not zeroed",
        [momentum("unknown[i] & couples_high[i], a_n", "unknown[i], a_n")],
    ),
    (
        "obstacle_neighbour_kept_low",
        "the coefficient toward an obstacle's face on the low side not zeroed",
        [momentum("unknown[i] & couples_low[i], a_s", "unknown[i], a_s")],
    ),
    (
        "solid_faces_unmasked",
        "the diagonal of a SOLID cell's face left as formed",
        [
            momentum(
                "lanes::keep(in.unknown[i], sum + net);",
                "lanes::keep(in.unknown[i] | lanes::all, sum + net);",
            )
        ],
    ),
    (
        "domain_edge_entries_kept",
        "the other component's entries left at a component's domain-edge faces",
        [
            momentum(
                "            a[row] = 0.0;\n            a[row + nx] = 0.0;\n",
                "            a[row] = a[row] + 0.0;\n",
            ),
            momentum(
                "                    std::fill(a.data() + row, a.data() + row + nx_, 0.0);\n",
                "                    std::fill(a.data() + row, a.data() + row, 0.0);\n",
            ),
        ],
    ),
    (
        "unknown_by_not_zero",
        "a face taken for an unknown where its diagonal is not zero, not where it is positive",
        [
            momentum(
                "lanes::mask_of(a_p > 0.0), a_p / alpha",
                "lanes::mask_of(a_p != 0.0), a_p / alpha",
            ),
            momentum(
                "lanes::mask_of(in.a_p[i] > 0.0)", "lanes::mask_of(in.a_p[i] != 0.0)"
            ),
        ],
    ),
    (
        "half_rho_dropped",
        "the density's half in the mass flux dropped",
        [momentum("half_rho_(in.rho * 0.5)", "half_rho_(in.rho)")],
    ),
    # The walls' conductances.
    (
        "takes_wall_not_at_obstacles",
        "wall_mu not read at the obstacle faces",
        [
            momentum(
                "            w.reach[i] = wall_distance[i];\n            w.takes_wall[i] = lanes::all;\n",
                "            w.reach[i] = wall_distance[i];\n",
            )
        ],
    ),
    (
        "takes_wall_not_at_edges",
        "wall_mu not read at the domain edges",
        [momentum("            w.takes_wall[i] = known;\n", "")],
    ),
    (
        "open_edge_conducts",
        "a zero-gradient edge conducting as a wall does, with a viscosity field",
        [
            momentum(
                "            w.conducts[i] = known;\n",
                "",
            )
        ],
    ),
    (
        "obstacle_wall_distance",
        "the obstacle wall taken a whole cell away, not half",
        [
            momentum(
                "            w.reach[i] = wall_distance[i];\n",
                "            w.reach[i] = 2.0 * wall_distance[i];\n",
            )
        ],
    ),
    (
        "edge_wall_distance",
        "the high edge's wall distance read at the low edge",
        [
            momentum(
                "w.reach[i] = r == 0 ? low.wall_distance : high.wall_distance;",
                "w.reach[i] = high.wall_distance;",
            )
        ],
    ),
    # The deferred correction.
    (
        "quick_order",
        "QUICK's three products added in another order",
        [
            momentum(
                "        const double q_pos = along<C>(in.std_pos[0], i) * below +\n"
                "                             along<C>(in.std_pos[1], i) * lo + along<C>(in.std_pos[2], i) * hi;",
                "        const double q_pos = along<C>(in.std_pos[2], i) * hi +\n"
                "                             along<C>(in.std_pos[1], i) * lo + along<C>(in.std_pos[0], i) * below;",
            )
        ],
    ),
    (
        "streamwise_blocked_pos",
        "the boundary form not taken beyond an obstacle's wall node, flow toward increasing s",
        [
            momentum(
                "            q = lanes::select(positive & in.blocked_pos[i], alt_pos, q);\n",
                "",
            )
        ],
    ),
    (
        "streamwise_blocked_neg",
        "the boundary form not taken beyond an obstacle's wall node, flow toward decreasing s",
        [
            momentum(
                "            q = lanes::select(~positive & in.blocked_neg[i], alt_neg, q);\n",
                "",
            )
        ],
    ),
    (
        "transverse_wall_pos",
        "the obstacle's wall not the far node, flow toward increasing t",
        [
            momentum(
                "            q = lanes::select(positive & in.far_wall_low[i], wall_pos, q);\n",
                "",
            )
        ],
    ),
    (
        "transverse_wall_neg",
        "the obstacle's wall not the far node, flow toward decreasing t",
        [
            momentum(
                "            q = lanes::select(~positive & in.far_wall_high[i], wall_neg, q);\n",
                "",
            )
        ],
    ),
    (
        "obstacle_face_corrected",
        "a deferred flux through an obstacle face",
        [momentum("            flux = lanes::keep(~in.wall_face[i], flux);\n", "")],
    ),
    (
        "edge_node_not_value",
        "the node beyond a Dirichlet edge of a u row the adjacent value, not the wall's",
        [
            momentum(
                "        out[i] = lanes::select(known[i], value[i], adjacent[i]);",
                "        out[i] = lanes::select(known[i] & lanes::none, value[i], adjacent[i]);",
            )
        ],
    ),
    (
        "edge_node_not_value_v",
        "the node beyond a Dirichlet high edge of a v row the adjacent value, not the wall's",
        [
            momentum(
                "lanes::select(w.high_known[edge_at], w.high_value[edge_at], row[nx - 1]);",
                "row[nx - 1];",
            )
        ],
    ),
    (
        "end_far_node",
        "the streamwise weights at the last face from the node before it as in the interior's other end",
        [
            momentum(
                "run.far_high = s + 1 < o.ns ? high + first + stride : low + first - stride;",
                "run.far_high = s + 1 < o.ns ? high + first + stride : low + first;",
            )
        ],
    ),
    (
        "row_blocked_flags",
        "no row of cells marked as holding an obstacle",
        [
            momentum(
                "            o.row_blocked[i / nx] = 1;",
                "            o.row_blocked[i / nx] = 0;",
            )
        ],
    ),
    (
        "row_wall_flags",
        "no row of corners marked as holding an obstacle",
        [
            momentum(
                "            w.row_walls[i / row] = 1;",
                "            w.row_walls[i / row] = 0;",
            )
        ],
    ),
    (
        "edge_face_corrected_v",
        "a deferred flux through a v row's domain-edge faces",
        [
            momentum(
                "        deferred[0] = 0.0;\n        deferred[nx] = 0.0;\n",
                "        deferred[0] = out.flux[0];\n        deferred[nx] = out.flux[nx];\n",
            )
        ],
    ),
    # The viscosity field.
    (
        "harmonic_solid_low",
        "a SOLID cell on the low side of a harmonic mean not replaced by its neighbour",
        [
            momentum(
                "lanes::select(solid_low[i] & ~solid_high[i], high[i], low[i]);",
                "lanes::select(lanes::none & solid_low[i], high[i], low[i]);",
            )
        ],
    ),
    (
        "harmonic_solid_high",
        "a SOLID cell on the high side of a harmonic mean not replaced by its neighbour",
        [
            momentum(
                "lanes::select(solid_high[i] & ~solid_low[i], low[i], high[i]);",
                "lanes::select(lanes::none & solid_high[i], low[i], high[i]);",
            )
        ],
    ),
    (
        "harmonic_fraction_swapped",
        "the harmonic mean's two distances swapped",
        [
            momentum(
                "w.harmonic_low[r] = d_lo / (d_lo + d_hi);",
                "w.harmonic_low[r] = d_hi / (d_lo + d_hi);",
            )
        ],
    ),
    (
        "mean_fraction_swapped",
        "the arithmetic mean's two widths swapped",
        [
            momentum(
                "o.mean_high[s] = w_r / (w_l + w_r);",
                "o.mean_high[s] = w_l / (w_l + w_r);",
            )
        ],
    ),
    (
        "first_mean_row_skipped",
        "the first row of cells' harmonic means of v taken from an earlier call",
        [
            momentum(
                "harmonic_row<C, T>(before(at, C), mu, s == 1);",
                "harmonic_row<C, T>(before(at, C), mu, false);",
            )
        ],
    ),
    (
        "stress_dropped",
        "form b's stress source left out",
        [
            momentum(
                "            deferred = deferred + stress;\n",
                "            deferred = deferred + 0.0 * stress;\n",
            )
        ],
    ),
    (
        "stress_sign_t",
        "the low transverse face's stress added, not subtracted",
        [
            momentum(
                "               high_side<T>(in.stress, in.stress_above, i) -\n",
                "               high_side<T>(in.stress, in.stress_above, i) +\n",
            )
        ],
    ),
    (
        "stress_width_cancelled",
        "the transverse stress flux without dividing and multiplying by the width",
        [
            momentum(
                "        out[i] = (high[i] - low[i]) / w * w;",
                "        out[i] = (high[i] - low[i]) * (w / w);",
            )
        ],
    ),
    # The sources and the sweeps.
    (
        "pressure_sign",
        "the pressure source's sign",
        [
            momentum(
                "across, -(in.p[i] - low_side<C>(in.p_below, in.p, i))",
                "across, (in.p[i] - low_side<C>(in.p_below, in.p, i))",
            )
        ],
    ),
    (
        "relaxation_order",
        "the relaxation term's product in another order",
        [
            momentum(
                "in.relaxation * in.a_p[i] * in.own[i];",
                "in.relaxation * (in.a_p[i] * in.own[i]);",
            )
        ],
    ),
    (
        "source_order",
        "the source's parts added in another order",
        [
            momentum(
                "b[i] = b[i] + deferred + pressure +",
                "b[i] = b[i] + (deferred + pressure) +",
            )
        ],
    ),
    (
        "sweep_neighbours_swapped",
        "a sweep's transverse neighbours taken from the other sides",
        [
            momentum(
                "    return sum + in.plus[Part][i] * high_neighbour<d, Inner>(in, i) +\n"
                "           in.minus[Part][i] * low_neighbour<d, Inner>(in, i);",
                "    return sum + in.plus[Part][i] * low_neighbour<d, Inner>(in, i) +\n"
                "           in.minus[Part][i] * high_neighbour<d, Inner>(in, i);",
            ),
        ],
    ),
    (
        "sweep_order",
        "a sweep's four products added in another order",
        [
            momentum(
                "    return sum + in.plus[Part][i] * high_neighbour<d, Inner>(in, i) +\n"
                "           in.minus[Part][i] * low_neighbour<d, Inner>(in, i);",
                "    return sum + (in.plus[Part][i] * high_neighbour<d, Inner>(in, i) +\n"
                "                  in.minus[Part][i] * low_neighbour<d, Inner>(in, i));",
            )
        ],
    ),
    (
        "gauss_seidel",
        "each sweep reading the rows the same sweep has written",
        [
            momentum(
                "run.below[d] = at[d] > 0 ? run.x - stride : zeros_.data();",
                "run.below[d] = at[d] > 0 ? next + row - stride : zeros_.data();",
            )
        ],
    ),
    (
        "sweep_targets",
        "the last sweep written to the scratch iterate, not to the output",
        [momentum("(sweeps_ - n) % 2 == 1 ? star", "(sweeps_ - n) % 2 == 0 ? star")],
    ),
    (
        "iterate_edges_not_kept",
        "the scratch iterate's domain-edge faces left as the last call's",
        [momentum("        edges(iterate_.data());\n", "")],
    ),
    (
        "star_edges_not_kept",
        "the output's domain-edge faces not set to the input",
        [momentum("    edges(star);\n", "")],
    ),
    (
        "alpha_ignored",
        "the sweeps dividing by the un-relaxed diagonal",
        [
            momentum(
                "lanes::mask_of(a_p > 0.0), a_p / alpha",
                "lanes::mask_of(a_p > 0.0), a_p + 0.0 * alpha",
            )
        ],
    ),
    (
        "held_reads_iterate",
        "a vent's faces set to the sweep's iterate, whose rows the last sweep wrote",
        [
            momentum(
                "lanes::keep(in.held[i], in.own[i])", "lanes::keep(in.held[i], in.b[i])"
            )
        ],
    ),
    # What the restructure adds (prompt 92b).
    (
        "kept_row_first",
        "every kept row of fluxes the same one, not the last two along the direction",
        [momentum("row + (at[d] % 2) * stride;", "row + (at[d] % 1) * stride;")],
    ),
    (
        "area_without_width",
        "a streamwise face's area without the transverse width",
        [
            momentum(
                "((x = x * along<T>(widths[slot_of(C, T)], i)), ...);",
                "((x = x * (0.0 * along<T>(widths[slot_of(C, T)], i) + 1.0)), ...);",
            )
        ],
    ),
    (
        "edge_relaxed_kept",
        "the relaxed diagonal beside a domain edge the one from before the edge's part left it",
        [
            momentum(
                "        relaxed[i] = relaxed_diagonal(p, in.alpha);\n        neighbour[i] = 0.0;\n",
                "        relaxed[i] = relaxed[i] + 0.0;\n        neighbour[i] = 0.0;\n",
            )
        ],
    ),
    (
        "edge_neighbour_kept",
        "the coefficient toward a domain edge not zeroed",
        [
            momentum(
                "        neighbour[i] = 0.0;\n",
                "        neighbour[i] = neighbour[i] + 0.0;\n",
            )
        ],
    ),
    (
        "edge_at_every_face",
        "an edge's condition read at the row's first face along an edge that crosses the rows",
        [
            momentum(
                "    const std::size_t edge_at = edge_index(o.faces, T, at);\n    double* toward_high",
                "    const std::size_t edge_at = 0 * edge_index(o.faces, T, at);\n    double* toward_high",
            )
        ],
    ),
    (
        "cross_pressure_below",
        "the pressure below a row of v taken from the row itself",
        [momentum("    rhs.p_below = p + cell_below;", "    rhs.p_below = p + cell;")],
    ),
    # stencil_viscosity.
    (
        "stencil_everywhere",
        "stencil_viscosity's face rule at every face, read or not",
        [
            momentum(
                "        out[i] = lanes::select(read[i], viscosity[i], air);",
                "        out[i] = lanes::select(read[i] | lanes::all, viscosity[i], air);",
            )
        ],
    ),
    (
        "stencil_last_row_u",
        "stencil_viscosity's top row of u left at air's viscosity",
        [
            momentum(
                "        last[C] = w.corners.cells(C) - 1;\n",
                "        last[C] = w.corners.cells(C) - 1;\n        last[T] = last[T] - (C == 0 ? 1 : 0);\n",
            )
        ],
    ),
    # The core's checks.
    (
        "check_not_finite",
        "a viscosity that is not finite accepted",
        [
            momentum(
                "    if (total.not_finite != lanes::none) {\n        return viscosity_status::not_finite;\n    }\n",
                "",
            )
        ],
    ),
    (
        "check_not_positive",
        "a viscosity of zero accepted",
        [
            momentum(
                "read & lanes::mask_of(mu[i] <= 0.0);",
                "read & lanes::mask_of(mu[i] < 0.0);",
            )
        ],
    ),
    (
        "check_infinity",
        "an infinite viscosity taken for a finite one",
        [
            momentum(
                "read & lanes::mask_of(!(std::fabs(mu[i]) <= std::numeric_limits<double>::max()));",
                "read & lanes::mask_of(mu[i] != mu[i]);",
            )
        ],
    ),
    (
        "check_solid_too",
        "a SOLID cell's viscosity checked",
        [
            momentum(
                "verdict_of(mu_eff, grid_->solid_flags(), lanes::all)",
                "verdict_of(mu_eff, grid_->solid_flags(), lanes::none)",
            )
        ],
    ),
    (
        "check_wall_everywhere",
        "a wall viscosity checked where no unknown reads it",
        [
            momentum(
                "                w.wall_read[i] = known & o.unknown[index_of(o.faces, face)];",
                "                w.wall_read[i] = known | (o.unknown[index_of(o.faces, face)] & lanes::none);",
            )
        ],
    ),
    (
        "check_first_cell_only",
        "the checks stopping before the cells after the whole blocks",
        [
            momentum(
                "    lanes::for_each_block(0, mu.size(), [&](auto cells, std::size_t m) {\n        const verdicts v",
                "    lanes::for_each_block(0, mu.size() / 8 * 8 - (mu.size() % 8 == 0 ? 8 : 0), [&](auto cells, std::size_t m) {\n        const verdicts v",
            )
        ],
    ),
    *[
        (
            f"check_{name}",
            f"the core's check removed: {what}",
            [momentum(old, new)],
        )
        for name, what, old, new in (
            (
                "two_cells",
                "two cells per axis",
                "        if (g.cells(d) < 2) {",
                "        if (g.cells(d) < 1) {",
            ),
            (
                "cell_widths",
                "the cell widths' lengths",
                "in.cell_widths[d].size() != g.cells(d)",
                "in.cell_widths[d].empty()",
            ),
            (
                "face_widths",
                "the face widths' lengths",
                "in.face_widths[d].size() != g.cells(d) + 1",
                "in.face_widths[d].empty()",
            ),
            (
                "edge_lengths",
                "the edges' lengths",
                "e->is_dirichlet.size() != faces || e->value.size() != faces",
                "e->is_dirichlet.empty() || faces == 0",
            ),
            (
                "held_length",
                "the held masks' lengths",
                "!in.held[d].empty() && in.held[d].size() != g.fields().face_count(d)",
                "false",
            ),
            (
                "sweeps",
                "at least one sweep",
                "    if (in.sweeps < 1) {",
                "    if (in.sweeps < 0) {",
            ),
            (
                "throws",
                "the constructor's throw",
                "        throw std::invalid_argument(std::string(reason));",
                "        (void)reason;",
            ),
        )
    ],
    # The binding.
    *[
        (f"binding_{name}", f"the binding: {what}", [binding(old, new)])
        for name, what, old, new in (
            (
                "edges_swapped",
                "bottom and top handed over swapped",
                'input.low[0][0] = read_edge(tangential[0], along_x, "bottom"',
                'input.low[0][0] = read_edge(tangential[1], along_x, "bottom"',
            ),
            (
                "left_right",
                "left's condition read from the right's",
                'read_edge(tangential[2], along_y, "left"',
                'read_edge(tangential[3], along_y, "left"',
            ),
            (
                "held_v_dropped",
                "the vents of v not handed over",
                "        if (*held[c]) {\n            require_shape(**held[c], face_shape(g, c), who, c == 0",
                "        if (*held[c] && c == 0) {\n            require_shape(**held[c], face_shape(g, c), who, c == 0",
            ),
            (
                "held_shape",
                "the vents' shape not checked",
                '            require_shape(**held[c], face_shape(g, c), who, c == 0 ? "held_u" : "held_v");\n',
                "",
            ),
            (
                "edge_count",
                "the number of edges not checked",
                "    if (nb::len(tangential) != 4) {",
                "    if (nb::len(tangential) < 3) {",
            ),
            (
                "edge_flags",
                "the edges' flags' length not checked",
                "!nb::try_cast<BoolVector>(edge[0], known, false) || known.shape(0) != n ||",
                "!nb::try_cast<BoolVector>(edge[0], known, false) ||",
            ),
            (
                "edge_values",
                "the edges' values' length not checked",
                "!nb::try_cast<Vector>(edge[1], value, false) || value.shape(0) != n) {",
                "!nb::try_cast<Vector>(edge[1], value, false)) {",
            ),
            (
                "edge_unaligned",
                "an unaligned edge array accepted",
                "    require_aligned(value.data(), alignof(double), who);\n    condition.wall_distance",
                "    condition.wall_distance",
            ),
            (
                "width_unaligned",
                "an unaligned width array accepted",
                "    for (const Vector* v : {&dx_cell, &dy_cell, &dx_face, &dy_face}) {\n        require_aligned(v->data(), alignof(double), who);\n    }\n    if (nb::len(tangential)",
                "    if (nb::len(tangential)",
            ),
            (
                "u_v_shape",
                "u's and v's shapes not checked",
                "    if (!has_shape(u, u_shape) || !has_shape(v, v_shape)) {",
                "    if (false) {",
            ),
            (
                "v_shape",
                "v's shape not checked",
                "!has_shape(u, u_shape) || !has_shape(v, v_shape)",
                "!has_shape(u, u_shape)",
            ),
            (
                "p_shape",
                "p's shape not checked",
                "    if (!has_shape(p, p_shape)) {",
                "    if (false) {",
            ),
            (
                "field_unaligned",
                "an unaligned field accepted",
                '        require_aligned(a->data(), alignof(double), "MomentumPredictor2");\n',
                "        (void)a;\n",
            ),
            (
                "mu_type",
                "a viscosity that is not an ndarray not named",
                '    if (!nb::isinstance(value, numpy.attr("ndarray"))) {',
                "    if (false) {",
            ),
            (
                "mu_dtype",
                "a viscosity's dtype not checked",
                '    if (!dtype.equal(numpy.attr("float64"))) {',
                "    if (false) {",
            ),
            (
                "mu_layout",
                "a viscosity's layout converted silently",
                "    if (!nb::try_cast<Array>(value, array, false)) {",
                "    if (!nb::try_cast<Array>(value, array, true)) {",
            ),
            (
                "mu_shape",
                "a viscosity's shape not checked",
                "    if (!has_shape(array, shape)) {",
                "    if (false) {",
            ),
            (
                "mu_unaligned",
                "an unaligned viscosity accepted",
                '    require_aligned(array.data(), alignof(double), "MomentumPredictor2");\n',
                "",
            ),
            (
                "mu_check",
                "mu_eff's values not checked",
                "    switch (m.check_viscosity(field)) {",
                "    switch (viscosity_status::ok) {",
            ),
            (
                "wall_dict",
                "a wall_mu that is not a dict not named",
                "    if (!nb::isinstance<nb::dict>(value)) {",
                "    if (false) {",
            ),
            (
                "wall_keys",
                "wall_mu's keys not checked",
                '    if (nb::len(fields) != 2 || !fields.contains("u") || !fields.contains("v")) {',
                "    if (false) {",
            ),
            (
                "wall_extra_key",
                "an extra key of wall_mu accepted",
                "    if (nb::len(fields) != 2 || !fields",
                "    if (!fields",
            ),
            (
                "wall_check",
                "wall_mu's values not checked",
                "        switch (m.check_wall_viscosity(c, 0, spans[c][0])) {",
                "        switch (viscosity_status::ok) {",
            ),
            (
                "wall_swapped",
                "v's wall viscosity handed to u",
                "        keep[c] = read_viscosity(fields[key], name,",
                '        keep[c] = read_viscosity(fields[c == 0 ? "v" : "u"], name,',
            ),
            (
                "transient",
                "a transient argument ignored",
                "    if (!transient.is_none()) {",
                "    if (false) {",
            ),
            (
                "wall_ignored",
                "wall_mu not handed to predict",
                "span_of_array(p), mu.field, mu.walls,\n              {out(u_star)",
                "span_of_array(p), mu.field, {},\n              {out(u_star)",
            ),
            (
                "fixed_name",
                "an unknown name's refusal",
                '    refuse_value("MomentumPredictor2.fixed: no quantity is named " + name);',
                "    return nb::none();",
            ),
        )
    ],
]
