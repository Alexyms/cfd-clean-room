"""Prompt 89's variants of ``core/``: candidates measured and dropped, and planted defects.

Read by ``probe89.py``. Each variant is a list of edits to a copy of the
checkout's ``core/`` on the Linux file system; the checkout is not changed.
The edits are written against ``core/`` as it was at eacc71c, round 0's head
(prompt 89b changed the flux and the flags: ``variants89b.py`` holds the plants
written against the core after it, and ``probe89.py`` builds a candidate of this
file on a commit's sources, ``results/builder89*/src_<commit>.tar``, as well).
An edit is ``(file, old, new)``, the exact text ``old`` replaced by ``new``,
or ``(file, start, end, new)``, the text from ``start`` up to ``end`` replaced
by ``new``. A text that is not found exactly once refuses the variant instead
of leaving it unapplied.

``CANDIDATES`` are forms of a kernel that compute the same values and were
measured against the kept form (the report's table of changes). ``PLANTS`` are
defects: each must fail a test, and ``probe89.py plants`` records which.
"""

HPP = "include/cfdcore/scalar.hpp"
CPP = "src/scalar.cpp"
LANES = "include/cfdcore/lanes.hpp"
GRID_HPP = "include/cfdcore/grid.hpp"
GRID_CPP = "src/grid.cpp"

Edit = tuple[str, str, str] | tuple[str, str, str, str]

# The limited QUICK flux as a block of lanes.hpp: every choice a mask and a
# select, the clamp without a branch. The form the kept loop is compared with.
BLOCK_QUICK = """// The limited QUICK flux through N interior faces from element `first`
// (`advective_flux` without `upwind`). The upstream, downstream and far nodes
// are chosen by the flux's sign; QUICK's value is three products added left to
// right, the far node's first (`quick_face_values`); a far node inside a
// SOLID cell is read as the upstream value, for the far node and for QUICK's
// value alike; the clamp is `limited_face_value`. `End` says which boundary
// nodes stand for far cells; `PerFace` reads the weights per face.
template <line_end End, bool PerFace, std::size_t N>
inline void quick_cells(lanes::cells<N>, const face_row& in, std::size_t first,
                        double* CFDCORE_RESTRICT out) {
    constexpr bool at_low = End == line_end::low || End == line_end::both;
    constexpr bool at_high = End == line_end::high || End == line_end::both;
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double f = in.flux[i];
        const lanes::mask positive = lanes::mask_of(f > 0.0);
        const double low = in.low[i];
        const double high = in.high[i];
        const double below =
            at_low ? low_boundary_node(in.end_flux_low[i], in.end_inflow_low[i], low)
                   : in.far_low[i];
        const double above =
            at_high ? high_boundary_node(in.end_flux_high[i], in.end_inflow_high[i], high)
                    : in.far_high[i];
        const lanes::mask solid_below = at_low ? lanes::none : in.solid_far_low[i];
        const lanes::mask solid_above = at_high ? lanes::none : in.solid_far_high[i];
        const double c_c = lanes::select(positive, low, high);
        const double c_d = lanes::select(positive, high, low);
        const double far = lanes::select(positive, below, above);
        const std::size_t w = PerFace ? i : 0;
        const double unlimited = lanes::select(positive, in.w_pos[0][w], in.w_neg[0][w]) * far +
                                 lanes::select(positive, in.w_pos[1][w], in.w_neg[1][w]) * c_c +
                                 lanes::select(positive, in.w_pos[2][w], in.w_neg[2][w]) * c_d;
        // A far node inside an obstacle is read as the upstream value, the
        // zero-gradient rule a domain wall gets.
        const lanes::mask far_solid = (positive & solid_below) | (~positive & solid_above);
        const double c_up = lanes::select(far_solid, c_c, far);
        const double quick = lanes::select(far_solid, c_c, unlimited);
        out[i] = f * limited_face_value(c_up, c_c, c_d, quick);
    }
}

// `quick_cells` over a run of `count` faces. The operands arrive by value: a
// copy nothing else can reach, so the compiler holds its pointers in registers.
template <line_end End, bool PerFace>
void quick_run(const face_row run, std::size_t count, double* out) {
    lanes::for_each_block(0, count, [&](auto cells, std::size_t m) {
        quick_cells<End, PerFace>(cells, run, m, out);
    });
}

"""

# The same with a branch a block on the fluxes' signs: a block whose fluxes
# all point one way reads one side's nodes and weights.
BLOCK_QUICK_BY_SIGN = """// The limited QUICK flux through N interior faces from element `first`
// (`advective_flux` without `upwind`). The upstream, downstream and far nodes
// are chosen by the flux's sign; QUICK's value is three products added left to
// right, the far node's first (`quick_face_values`); a far node inside a
// SOLID cell is read as the upstream value, for the far node and for QUICK's
// value alike; the clamp is `limited_face_value`. `End` says which boundary
// nodes stand for far cells; `PerFace` reads the weights per face. `Flow` is
// what the caller found the block's fluxes to be: under `positive` and
// `negative` the sign's mask is a constant and the block reads one side only.
enum class flow { mixed, positive, negative };

template <line_end End, bool PerFace, flow Flow, std::size_t N>
inline void quick_cells(lanes::cells<N>, const face_row& in, std::size_t first,
                        double* CFDCORE_RESTRICT out) {
    constexpr bool at_low = End == line_end::low || End == line_end::both;
    constexpr bool at_high = End == line_end::high || End == line_end::both;
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double f = in.flux[i];
        const lanes::mask positive = Flow == flow::mixed      ? lanes::mask_of(f > 0.0)
                                     : Flow == flow::positive ? lanes::all
                                                              : lanes::none;
        const double low = in.low[i];
        const double high = in.high[i];
        const double below =
            at_low ? low_boundary_node(in.end_flux_low[i], in.end_inflow_low[i], low)
                   : in.far_low[i];
        const double above =
            at_high ? high_boundary_node(in.end_flux_high[i], in.end_inflow_high[i], high)
                    : in.far_high[i];
        const lanes::mask solid_below = at_low ? lanes::none : in.solid_far_low[i];
        const lanes::mask solid_above = at_high ? lanes::none : in.solid_far_high[i];
        const double c_c = lanes::select(positive, low, high);
        const double c_d = lanes::select(positive, high, low);
        const double far = lanes::select(positive, below, above);
        const std::size_t w = PerFace ? i : 0;
        const double unlimited = lanes::select(positive, in.w_pos[0][w], in.w_neg[0][w]) * far +
                                 lanes::select(positive, in.w_pos[1][w], in.w_neg[1][w]) * c_c +
                                 lanes::select(positive, in.w_pos[2][w], in.w_neg[2][w]) * c_d;
        // A far node inside an obstacle is read as the upstream value, the
        // zero-gradient rule a domain wall gets.
        const lanes::mask far_solid = (positive & solid_below) | (~positive & solid_above);
        const double c_up = lanes::select(far_solid, c_c, far);
        const double quick = lanes::select(far_solid, c_c, unlimited);
        out[i] = f * limited_face_value(c_up, c_c, c_d, quick);
    }
}

// `quick_cells` over a run of `count` faces. The operands arrive by value: a
// copy nothing else can reach, so the compiler holds its pointers in registers.
template <std::size_t N>
CFDCORE_INLINE flow flow_of(const double* flux) {
    lanes::mask positive = lanes::none;
    lanes::mask other = lanes::none;
    for (std::size_t k = 0; k < N; ++k) {
        const lanes::mask toward = lanes::mask_of(flux[k] > 0.0);
        positive |= toward;
        other |= ~toward;
    }
    if (other == lanes::none) {
        return flow::positive;
    }
    return positive == lanes::none ? flow::negative : flow::mixed;
}

template <line_end End, bool PerFace>
void quick_run(const face_row run, std::size_t count, double* out) {
    lanes::for_each_block(0, count, [&]<std::size_t N>(lanes::cells<N> cells, std::size_t m) {
        if (N == 1) {
            quick_cells<End, PerFace, flow::mixed>(cells, run, m, out);
            return;
        }
        switch (flow_of<N>(run.flux + m)) {
        case flow::positive:
            quick_cells<End, PerFace, flow::positive>(cells, run, m, out);
            break;
        case flow::negative:
            quick_cells<End, PerFace, flow::negative>(cells, run, m, out);
            break;
        case flow::mixed:
            quick_cells<End, PerFace, flow::mixed>(cells, run, m, out);
            break;
        }
    });
}

"""

QUICK_START = "// The limited QUICK flux through a run of `count` interior faces\n"
QUICK_END = (
    "// The flux through a run of `count` interior faces, by the scheme asked for"
)


def _block_flux(quick: str) -> list[Edit]:
    """The edits that put a block form of the limited QUICK flux in place of the kept loop."""
    return [
        (
            HPP,
            '#include "cfdcore/grid.hpp"\n#include "cfdcore/layout.hpp"\n',
            '#include "cfdcore/grid.hpp"\n#include "cfdcore/lanes.hpp"\n'
            '#include "cfdcore/layout.hpp"\n',
        ),
        (
            HPP,
            "inline double np_minimum(double a, double b) "
            "{ return (a < b || std::isnan(a)) ? a : b; }\n"
            "inline double np_maximum(double a, double b) "
            "{ return (a > b || std::isnan(a)) ? a : b; }\n",
            "CFDCORE_INLINE double np_minimum(double a, double b) {\n"
            "    return lanes::select(lanes::mask_of(a < b) | lanes::mask_of(a != a), a, b);\n"
            "}\n"
            "CFDCORE_INLINE double np_maximum(double a, double b) {\n"
            "    return lanes::select(lanes::mask_of(a > b) | lanes::mask_of(a != a), a, b);\n"
            "}\n",
        ),
        (
            HPP,
            "inline double limited_face_value(double c_up, double c_c, double c_d, double quick) {\n"
            "    const double d_down = c_d - c_c;\n"
            "    // The reference's `d_down != 0.0`: a NaN difference takes the limited branch.\n"
            "    if (d_down == 0.0) {\n"
            "        return c_c;\n"
            "    }\n",
            "CFDCORE_INLINE double limited_face_value(double c_up, double c_c, double c_d,\n"
            "                                         double quick) {\n"
            "    const double d_down = c_d - c_c;\n"
            "    const lanes::mask defined = lanes::mask_of(d_down != 0.0);\n",
        ),
        (
            HPP,
            "    const double psi = np_maximum(0.0, low);\n"
            "    return c_c + 0.5 * psi * d_down;\n",
            "    const double psi = np_maximum(0.0, low);\n"
            "    return lanes::select(defined, c_c + 0.5 * psi * d_down, c_c);\n",
        ),
        (
            GRID_HPP,
            '#include "cfdcore/layout.hpp"\n',
            '#include "cfdcore/lanes.hpp"\n#include "cfdcore/layout.hpp"\n',
        ),
        (
            GRID_HPP,
            "    std::span<const std::uint8_t> solid_flags() const { return solid_; }\n",
            "    std::span<const std::uint8_t> solid_flags() const { return solid_; }\n"
            "    std::span<const lanes::mask> solid_masks() const { return solid_mask_; }\n",
        ),
        (
            GRID_HPP,
            "    std::vector<std::uint8_t> solid_;\n",
            "    std::vector<std::uint8_t> solid_;\n    std::vector<lanes::mask> solid_mask_;\n",
        ),
        (
            GRID_CPP,
            "      solid_(input.cell_type.size()) {",
            "      solid_(input.cell_type.size()),\n      solid_mask_(input.cell_type.size()) {",
        ),
        (
            GRID_CPP,
            "        solid_[cell] = cell_type_[cell] == input.solid ? 1 : 0;\n",
            "        solid_[cell] = cell_type_[cell] == input.solid ? 1 : 0;\n"
            "        solid_mask_[cell] = lanes::mask_of(cell_type_[cell] == input.solid);\n",
        ),
        (
            CPP,
            "    const std::uint8_t* solid_far_low = nullptr;\n"
            "    const std::uint8_t* solid_far_high = nullptr;\n",
            "    const lanes::mask* solid_far_low = nullptr;\n"
            "    const lanes::mask* solid_far_high = nullptr;\n",
        ),
        (CPP, QUICK_START, QUICK_END, quick),
        (
            CPP,
            "    const std::uint8_t* solid = g.solid_flags().data();\n",
            "    const lanes::mask* solid = g.solid_masks().data();\n",
        ),
    ]


CANDIDATES: dict[str, dict[str, object]] = {
    "flux_blocks": {
        "what": "the limited QUICK flux as a block: masks and selects, no branch",
        "edits": _block_flux(BLOCK_QUICK),
    },
    "flux_blocks_by_sign": {
        "what": "the block flux, a block whose fluxes share a sign reading one side only",
        "edits": _block_flux(BLOCK_QUICK_BY_SIGN),
    },
    "flux_paired_division": {
        "what": "the kept flux, the clamp's two quotients formed as one pair",
        "edits": [
            (
                HPP,
                "    const double r = (c_c - c_up) / d_down;\n"
                "    const double psi_quick = 2.0 * (quick - c_c) / d_down;\n",
                "    const double numerator[2] = {c_c - c_up, 2.0 * (quick - c_c)};\n"
                "    double quotient[2];\n"
                "    for (int j = 0; j < 2; ++j) {\n"
                "        quotient[j] = numerator[j] / d_down;\n"
                "    }\n"
                "    const double r = quotient[0];\n"
                "    const double psi_quick = quotient[1];\n",
            )
        ],
    },
    "jacobi_copies": {
        "what": "the Jacobi iterate copied in and out, as prompt 88 held it",
        "edits": [
            (
                CPP,
                "    const double* x = in.c_star.data();\n"
                "    double* next = out.data();\n"
                "    double* spare = scratch.next.data();\n",
                "    std::copy(in.c_star.begin(), in.c_star.end(), scratch.c.begin());\n"
                "    const double* x = scratch.c.data();\n"
                "    double* next = scratch.next.data();\n"
                "    double* spare = scratch.c.data();\n",
            ),
            (
                CPP,
                "    if (x != out.data()) {\n        std::copy(x, x + n, out.begin());\n    }\n",
                "    std::copy(x, x + n, out.begin());\n",
            ),
        ],
    },
    "jacobi_two_masks": {
        "what": "the Jacobi sweep reading two mask arrays where no cell is held",
        "edits": [
            (
                CPP,
                "    if (in.held.empty()) {\n        return scratch.fluid.data();\n    }\n",
                "    if (in.held.empty()) {\n"
                "        std::copy(scratch.fluid.begin(), scratch.fluid.end(), scratch.free.begin());\n"
                "        return scratch.free.data();\n"
                "    }\n",
            )
        ],
    },
    "neighbours_by_pointer": {
        "what": "a stencil's row neighbours read through a pointer each, between the ends too",
        "edits": [
            (
                CPP,
                "    lanes::for_each_block(1, n - 1, [&](auto count, std::size_t m) {\n"
                "        block(count, row_between{}, row, m, 0);\n"
                "    });\n",
                "    if (n > 2) {\n"
                "        const stencil_row<D> between = at_cell(row, 1, n);\n"
                "        lanes::for_each_block(0, n - 2, [&](auto count, std::size_t k) {\n"
                "            block(count, row_end{}, between, k, 1);\n"
                "        });\n"
                "    }\n",
            )
        ],
    },
    "helpers_not_forced_inline": {
        "what": "the helpers a block calls left to the compiler's own inlining",
        "edits": [
            (
                LANES,
                "#define CFDCORE_INLINE [[gnu::always_inline]] inline\n",
                "#define CFDCORE_INLINE inline\n",
            )
        ],
    },
    "row_loops_not_blocks": {
        "what": "each walk one loop to the row's length, not blocks of a constant length",
        "edits": [
            (
                LANES,
                "    for (; m + width <= last; m += width) {\n"
                "        block(cells<width>{}, m);\n"
                "    }\n",
                "",
            )
        ],
    },
}

PLANTS: list[tuple[str, str, list[Edit]]] = [
    ("control", "the unchanged source", []),
    (
        "walk_one_block_too_many",
        "a walk takes a whole block where fewer cells than a block are left",
        [
            (
                LANES,
                "    for (; m + width <= last; m += width) {",
                "    for (; m + width <= last + 1; m += width) {",
            )
        ],
    ),
    (
        "walk_skips_the_cells_left",
        "a walk stops after its whole blocks",
        [
            (
                LANES,
                "    for (; m < last; ++m) {\n        block(cells<1>{}, m);\n    }\n",
                "",
            )
        ],
    ),
    (
        "stencil_end_as_interior",
        "a row's last cell goes through the blocks, reading the cell beyond the row",
        [
            (
                CPP,
                "    lanes::for_each_block(1, n - 1, [&](auto count, std::size_t m) {\n"
                "        block(count, row_between{}, row, m, 0);\n"
                "    });\n"
                "    if (n > 1) {\n"
                "        block(lanes::cells<1>{}, row_end{}, at_cell(row, n - 1, n), 0, n - 1);\n"
                "    }\n",
                "    lanes::for_each_block(1, n, [&](auto count, std::size_t m) {\n"
                "        block(count, row_between{}, row, m, 0);\n"
                "    });\n",
            )
        ],
    ),
    (
        "end_cell_as_a_cell_between",
        "a row's last cell reads its east neighbour as the next element in memory, not the zero",
        [
            (
                CPP,
                "        block(lanes::cells<1>{}, row_end{}, at_cell(row, n - 1, n), 0, n - 1);\n",
                "        block(lanes::cells<1>{}, row_between{}, at_cell(row, n - 1, n), 0, n - 1);\n",
            )
        ],
    ),
    (
        "padding_east_as_cell",
        "the zero read beyond a row's east end taken as the cell's own value",
        [
            (
                CPP,
                "cell.east = m + 1 < n ? row.x + m + 1 : row.zeros;",
                "cell.east = m + 1 < n ? row.x + m + 1 : row.x + m;",
            )
        ],
    ),
    (
        "padding_west_as_cell",
        "the zero read beyond a row's west end taken as the cell's own value",
        [
            (
                CPP,
                "cell.west = m > 0 ? row.x + m - 1 : row.zeros;",
                "cell.west = m > 0 ? row.x + m - 1 : row.x + m;",
            )
        ],
    ),
    (
        "padding_north_as_cell",
        "the zeros read above the last row taken as the row's own values",
        [
            (
                CPP,
                "s.has_high[d] ? x + s.cell + s.stride[d] : row.zeros;",
                "s.has_high[d] ? x + s.cell + s.stride[d] : x + s.cell;",
            )
        ],
    ),
    (
        "padding_south_as_cell",
        "the zeros read below the first row taken as the row's own values",
        [
            (
                CPP,
                "s.has_low[d] ? x + s.cell - s.stride[d] : row.zeros;",
                "s.has_low[d] ? x + s.cell - s.stride[d] : x + s.cell;",
            )
        ],
    ),
    (
        "sweep_update_masked_as_residual",
        "the Jacobi update zeroed in held cells as well as SOLID ones",
        [
            (
                CPP,
                "next[i] = lanes::keep(in.fluid[i], in.x[i] + ri / in.a_p[i]);",
                "next[i] = lanes::keep(in.free[i], in.x[i] + ri / in.a_p[i]);",
            )
        ],
    ),
    (
        "residual_unmasked",
        "the residual not zeroed in the cells that are not unknowns",
        [
            (
                CPP,
                "    return lanes::keep(in.free[i],\n"
                "                       (in.b[i] + neighbour_sum(where, in, i)) - in.a_p[i] * in.x[i]);",
                "    return (in.b[i] + neighbour_sum(where, in, i)) - in.a_p[i] * in.x[i];",
            )
        ],
    ),
    (
        "product_unmasked",
        "the CG's product not zeroed in the cells that are not unknowns",
        [
            (
                CPP,
                "q[i] = lanes::keep(in.free[i], in.a_p[i] * in.x[i] - neighbour_sum(where, in, i));",
                "q[i] = in.a_p[i] * in.x[i] - neighbour_sum(where, in, i);",
            )
        ],
    ),
    (
        "neighbour_order",
        "the neighbour sum's terms along y taken south before north",
        [
            (
                CPP,
                "        sum = sum + in.g_high[d][i] * in.x_high[d][i];\n"
                "        sum = sum + in.g_low[d][i] * in.x_low[d][i];\n",
                "        sum = sum + in.g_low[d][i] * in.x_low[d][i];\n"
                "        sum = sum + in.g_high[d][i] * in.x_high[d][i];\n",
            )
        ],
    ),
    (
        "diagonal_order",
        "the diagonal's terms along y taken north before south",
        [
            (
                CPP,
                "            extra = extra + in.g_low[d][i];\n            extra = extra + in.g_high[d][i];\n",
                "            extra = extra + in.g_high[d][i];\n            extra = extra + in.g_low[d][i];\n",
            )
        ],
    ),
    (
        "stop_from_one_lane",
        "the stop's largest |b| taken from the first of the eight running maxima",
        [
            (
                CPP,
                "    for (const double lane : largest) {\n        value = lane > value ? lane : value;\n    }\n",
                "    value = largest[0];\n",
            )
        ],
    ),
    (
        "stop_nan_dropped",
        "a NaN |b| left out of the stop",
        [
            (
                CPP,
                "    if (flags.nan != lanes::none) {\n        value =",
                "    if (false) {\n        value =",
            )
        ],
    ),
    (
        "outside_stop_strict",
        "the max-norm stop met strictly below the stop, not at it",
        [
            (
                CPP,
                "return lanes::mask_of(!(std::fabs(r) <= stop));",
                "return lanes::mask_of(!(std::fabs(r) < stop));",
            )
        ],
    ),
    (
        "jacobi_last_copy_dropped",
        "the Jacobi solve's field left where its last sweep wrote it",
        [
            (
                CPP,
                "    if (x != out.data()) {\n        std::copy(x, x + n, out.begin());\n    }\n",
                "",
            )
        ],
    ),
    (
        "cg_dot_of_the_wrong_row",
        "the CG's r.z taken as z.z",
        [
            (
                CPP,
                "            rz_new += reduce::block_dot(row_of(r, line), row_of(z, line));",
                "            rz_new += reduce::block_dot(row_of(z, line), row_of(z, line));",
            )
        ],
    ),
    (
        "cg_z_from_the_old_residual",
        "the CG's z formed from the residual before its update",
        [
            (
                CPP,
                "        z[i] = inverse[i] * ri;\n",
                "        z[i] = inverse[i] * (ri + alpha * q[i]);\n",
            )
        ],
    ),
    (
        "cg_direction_fused",
        "the direction updated as fma(p, beta, z), the contraction -ffp-contract=off forbids",
        [
            (
                CPP,
                "        p[i] = p[i] * beta + z[i];",
                "        p[i] = std::fma(p[i], beta, z[i]);",
            )
        ],
    ),
    (
        "block_dot_left_to_right",
        "a row's dot product summed left to right, not in pairwise order",
        [
            (
                "src/reduce.cpp",
                "    return pairwise([pa, pb](std::size_t i) { return pa[i] * pb[i]; }, 0, a.size());",
                "    double sum = 0.0;\n"
                "    for (std::size_t i = 0; i < a.size(); ++i) {\n"
                "        sum += pa[i] * pb[i];\n"
                "    }\n"
                "    return sum;",
            )
        ],
    ),
    (
        "flux_end_node_ignores_inflow",
        "a line's low boundary node read as the cell's value whatever the flux",
        [
            (
                CPP,
                "            c_up = at_low ? low_boundary_node(run.end_flux_low[i], run.end_inflow_low[i], low)\n"
                "                          : run.far_low[i];",
                "            c_up = at_low ? low : run.far_low[i];",
            )
        ],
    ),
    (
        "flux_weights_of_the_first_face",
        "every face of a row along x given the row's first face's weights",
        [
            (
                CPP,
                "        const std::size_t w = PerFace ? i : 0;",
                "        const std::size_t w = 0;",
            )
        ],
    ),
    (
        "flux_far_solid_dropped",
        "a far node inside a SOLID cell read as it is",
        [
            (
                CPP,
                "        if (far_solid) {\n            c_up = c_c;",
                "        if (false) {\n            c_up = c_c;",
            )
        ],
    ),
    (
        "flux_high_boundary_sign",
        "the high boundary node takes the inflow value where the flux leaves",
        [
            (
                CPP,
                "    return lanes::select(lanes::mask_of(flux < 0.0), inflow, cell);",
                "    return lanes::select(lanes::mask_of(flux > 0.0), inflow, cell);",
            )
        ],
    ),
    (
        "flux_upwind_downstream",
        "the upwind value taken from the downstream cell",
        [
            (
                CPP,
                "        out[i] = f * lanes::select(lanes::mask_of(f > 0.0), in.low[i], in.high[i]);",
                "        out[i] = f * lanes::select(lanes::mask_of(f > 0.0), in.high[i], in.low[i]);",
            )
        ],
    ),
    (
        "flux_vents_dropped",
        "the vents' second pass not made",
        [(CPP, "    if (vents.empty()) {\n        return;\n    }\n", "    return;\n")],
    ),
    (
        "flux_last_row_boundary_dropped",
        "the high boundary faces of the lines along y not written",
        [(CPP, "        if (q + 1 == n) {", "        if (false) {")],
    ),
    (
        "select_swapped",
        "lanes::select returns the other value",
        [
            (
                LANES,
                "    return std::bit_cast<double>((std::bit_cast<mask>(if_set) & m) |\n"
                "                                 (std::bit_cast<mask>(otherwise) & ~m));",
                "    return std::bit_cast<double>((std::bit_cast<mask>(if_set) & ~m) |\n"
                "                                 (std::bit_cast<mask>(otherwise) & m));",
            )
        ],
    ),
    (
        "keep_by_multiplication",
        "lanes::keep multiplies by one or zero where it should pass or drop bits",
        [
            (
                LANES,
                "    return std::bit_cast<double>(std::bit_cast<mask>(value) & m);",
                "    return value * (m == all ? 1.0 : 0.0);",
            )
        ],
    ),
    ("control_end", "the unchanged source, after every plant", []),
]
