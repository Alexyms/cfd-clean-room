#include "cfdcore/multigrid.hpp"

#include "cfdcore/reduce.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace cfdcore::multigrid {

namespace {

void require_length(std::size_t got, std::size_t expected, const char* what) {
    if (got != expected) {
        throw std::invalid_argument(what);
    }
}

// ---------------------------------------------------------------------------
// The stencil

// What the stencil of a row's cells reads, each pointer at the row's first
// cell, so that cell m is element m of every array.
//
// x:             the field the operator is applied to
// centre, high, low: the system's diagonal and couplings
// x_high, x_low: for each direction d >= 1, the rows of x on the high and the
//                low side, a row of zeros where the field ends. Entry 0 is read
//                at a row's two end cells alone (`row_end`), whose operands are
//                at the cell and not at the row's first
// r, smoother:   read by the residual and the sweep alone
// zeros:         the row of zeros
template <int D>
struct stencil_row {
    const double* x = nullptr;
    const double* centre = nullptr;
    const double* high[D] = {};
    const double* low[D] = {};
    const double* x_high[D] = {};
    const double* x_low[D] = {};
    const double* r = nullptr;
    const double* smoother = nullptr;
    const double* zeros = nullptr;
};

// The operands of a row for the field x; `r` and `smoother` as needed.
template <int D>
stencil_row<D> operands(const system<D>& s, std::size_t cell, const double* zeros,
                        const double* x, const double* r, const double* smoother) {
    const auto& fields = *s.fields;
    stencil_row<D> row;
    row.x = x + cell;
    row.centre = s.centre.data() + cell;
    row.high[0] = s.high[0].data() + cell;
    row.low[0] = s.low[0].data() + cell;
    for (int d = 1; d < D; ++d) {
        const std::size_t stride = fields.cell_stride(d);
        const std::size_t position = (cell / stride) % fields.cells(d);
        row.high[d] = s.high[d].data() + cell;
        row.low[d] = s.low[d].data() + cell;
        row.x_high[d] = position + 1 < fields.cells(d) ? row.x + stride : zeros;
        row.x_low[d] = position > 0 ? row.x - stride : zeros;
    }
    row.r = r == nullptr ? nullptr : r + cell;
    row.smoother = smoother == nullptr ? nullptr : smoother + cell;
    row.zeros = zeros;
    return row;
}

// The operands at cell m of an n-cell row, for one end cell: the neighbour it
// lacks along direction 0 is read from the row of zeros.
template <int D>
stencil_row<D> at_cell(const stencil_row<D>& row, std::size_t m, std::size_t n) {
    stencil_row<D> cell = row;
    cell.x = row.x + m;
    cell.centre = row.centre + m;
    for (int d = 0; d < D; ++d) {
        cell.high[d] = row.high[d] + m;
        cell.low[d] = row.low[d] + m;
    }
    cell.x_low[0] = m > 0 ? row.x + m - 1 : row.zeros;
    cell.x_high[0] = m + 1 < n ? row.x + m + 1 : row.zeros;
    for (int d = 1; d < D; ++d) {
        cell.x_high[d] = row.x_high[d] + m;
        cell.x_low[d] = row.x_low[d] + m;
    }
    cell.r = row.r == nullptr ? nullptr : row.r + m;
    cell.smoother = row.smoother == nullptr ? nullptr : row.smoother + m;
    return cell;
}

// Where a cell's neighbours along direction 0 are read: between a row's ends
// either side of the cell's own pointer, at an end through a pointer each.
struct row_between {};
struct row_end {};

template <int D>
CFDCORE_INLINE double high_of(row_between, const stencil_row<D>& in, std::size_t i) {
    return in.x[i + 1];
}
template <int D>
CFDCORE_INLINE double low_of(row_between, const stencil_row<D>& in, std::size_t i) {
    return in.x[i - 1];
}
template <int D>
CFDCORE_INLINE double high_of(row_end, const stencil_row<D>& in, std::size_t i) {
    return in.x_high[0][i];
}
template <int D>
CFDCORE_INLINE double low_of(row_end, const stencil_row<D>& in, std::size_t i) {
    return in.x_low[0][i];
}

// (A x) at element i: the diagonal's term, less each direction's high then low
// term, one after another (`apply_operator`'s order).
template <typename Where, int D>
CFDCORE_INLINE double product_at(Where where, const stencil_row<D>& in, std::size_t i) {
    double y = in.centre[i] * in.x[i];
    y = y - in.high[0][i] * high_of(where, in, i);
    y = y - in.low[0][i] * low_of(where, in, i);
    for (int d = 1; d < D; ++d) {
        y = y - in.high[d][i] * in.x_high[d][i];
        y = y - in.low[d][i] * in.x_low[d][i];
    }
    return y;
}

template <std::size_t N, typename Where, int D>
inline void product_cells(lanes::cells<N>, Where where, const stencil_row<D>& in,
                          std::size_t first, double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        out[i] = product_at(where, in, i);
    }
}

template <std::size_t N, typename Where, int D>
inline void residual_cells(lanes::cells<N>, Where where, const stencil_row<D>& in,
                           std::size_t first, double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        out[i] = in.r[i] - product_at(where, in, i);
    }
}

// A weighted Jacobi sweep over N cells. Masked zeroes the cells without a
// positive diagonal, the finest level's last pass.
template <bool Masked, std::size_t N, typename Where, int D>
inline void smooth_cells(lanes::cells<N>, Where where, const stencil_row<D>& in,
                         std::size_t first, double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double next = in.x[i] + in.smoother[i] * (in.r[i] - product_at(where, in, i));
        out[i] = Masked ? lanes::keep(lanes::mask_of(in.centre[i] > 0.0), next) : next;
    }
}

// Walks a row of n cells as block(count, where, operands, first, at): the two
// end cells apart, the cells between in blocks (scalar.cpp's walk).
template <int D, typename Block>
CFDCORE_INLINE void for_each_stencil_block(const stencil_row<D>& row, std::size_t n,
                                           Block&& block) {
    block(lanes::cells<1>{}, row_end{}, at_cell(row, 0, n), 0, 0);
    lanes::for_each_block(1, n - 1, [&](auto count, std::size_t m) {
        block(count, row_between{}, row, m, 0);
    });
    if (n > 1) {
        block(lanes::cells<1>{}, row_end{}, at_cell(row, n - 1, n), 0, n - 1);
    }
}

template <int D>
void product_cells_of(const stencil_row<D> row, std::size_t n, double* out) {
    for_each_stencil_block(row, n,
                           [&](auto count, auto where, const stencil_row<D>& in,
                               std::size_t first, std::size_t at) {
                               product_cells(count, where, in, first, out + at);
                           });
}

template <int D>
void residual_cells_of(const stencil_row<D> row, std::size_t n, double* out) {
    for_each_stencil_block(row, n,
                           [&](auto count, auto where, const stencil_row<D>& in,
                               std::size_t first, std::size_t at) {
                               residual_cells(count, where, in, first, out + at);
                           });
}

template <bool Masked, int D>
void smooth_cells_of(const stencil_row<D> row, std::size_t n, double* out) {
    for_each_stencil_block(row, n,
                           [&](auto count, auto where, const stencil_row<D>& in,
                               std::size_t first, std::size_t at) {
                               smooth_cells<Masked>(count, where, in, first, out + at);
                           });
}

// A sweep over every row.
template <bool Masked, int D>
void smooth_field(const system<D>& s, const double* zeros, const double* smoother,
                  const double* x, const double* r, double* out) {
    const std::size_t n0 = s.fields->cells(0);
    for (const auto& line : s.fields->lines(0)) {
        smooth_cells_of<Masked>(operands(s, line.cell, zeros, x, r, smoother), n0,
                                out + line.cell);
    }
}

// The first sweep, from zero: x = smoother * r.
template <std::size_t N>
inline void first_cells(lanes::cells<N>, const double* smoother, const double* r,
                        std::size_t first, double* CFDCORE_RESTRICT x) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        x[i] = smoother[i] * r[i];
    }
}

// ---------------------------------------------------------------------------
// The groups

// The rows of fine cells one coarse row's groups hold: one for each choice of
// the low or the high cell along every direction but 0. Row c takes direction
// d's high cell where bit d - 1 of c is set.
template <int D>
inline constexpr std::size_t group_rows = std::size_t{1} << (D - 1);

// A row of each, at the rows' first cells.
template <int D>
using row_set = std::array<const double*, group_rows<D>>;

// Where the fine rows of the groups of a coarse row begin, in the fine cell
// field, and whether there is such a row (a side's last group may be one wide).
template <int D>
struct group {
    std::array<std::size_t, group_rows<D>> cell;
    std::array<bool, group_rows<D>> present;
};

template <int D>
group<D> group_of(const layout::layout<D>& fine, const layout::layout<D>& coarse,
                  std::size_t coarse_cell) {
    group<D> g{};
    g.present.fill(true);
    for (int d = 1; d < D; ++d) {
        const std::size_t at = (coarse_cell / coarse.cell_stride(d)) % coarse.cells(d);
        for (std::size_t c = 0; c < group_rows<D>; ++c) {
            const std::size_t position = 2 * at + ((c >> (d - 1)) & 1);
            g.cell[c] += position * fine.cell_stride(d);
            g.present[c] = g.present[c] && position < fine.cells(d);
        }
    }
    return g;
}

// The m-th of the rows whose bit d - 1 is `side`.
constexpr std::size_t row_with(std::size_t m, int d, std::size_t side) {
    const std::size_t below = (std::size_t{1} << (d - 1)) - 1;
    return (m & below) | (side << (d - 1)) | ((m & ~below) << 1);
}

// The sum over R rows by pairs, adjacent rows' innermost (a group's direction 1
// before its direction 2): of each row's two cells of group i, or of its cell
// on one side along direction 0. R rows from row First.
template <std::size_t R, std::size_t First = 0, std::size_t S>
CFDCORE_INLINE double sum_pairs(const std::array<const double*, S>& rows, std::size_t i) {
    if constexpr (R == 1) {
        return rows[First][2 * i] + rows[First][2 * i + 1];
    } else {
        return sum_pairs<R / 2, First>(rows, i) + sum_pairs<R / 2, First + R / 2>(rows, i);
    }
}
template <std::size_t Side, std::size_t R, std::size_t First = 0, std::size_t S>
CFDCORE_INLINE double sum_side(const std::array<const double*, S>& rows, std::size_t i) {
    if constexpr (R == 1) {
        return rows[First][2 * i + Side];
    } else {
        return sum_side<Side, R / 2, First>(rows, i) +
               sum_side<Side, R / 2, First + R / 2>(rows, i);
    }
}

// What a coarse row's diagonals read: the rows of the fine diagonal, of the
// couplings along direction 0, and of those along each further direction d, the
// groups' low side of it alone (entry 0 is not used).
template <int D>
struct centre_row {
    row_set<D> centre;
    row_set<D> along;
    std::array<std::array<const double*, group_rows<D> / 2>, D> across;
};

// The diagonals of N coarse cells: the group's, less twice the couplings inside
// it (its low cells' along each direction).
//
// This block and `along_cells` read one of each two cells along the row, a load
// with gaps, which GCC at -O2 does not vectorise (it would have to read past
// the row); they stay loops of one group at a time. All of a correction's
// coarsening, them included, is about a fiftieth of it at 320x120, so no form
// that reads the other cells to throw them away was worth its arithmetic.
template <std::size_t N, int D>
inline void centre_cells(lanes::cells<N>, const centre_row<D>& in, std::size_t first,
                         double* CFDCORE_RESTRICT out) {
    constexpr std::size_t R = group_rows<D>;
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        double inside = sum_side<0, R>(in.along, i);
        if constexpr (D > 1) {
            for (std::size_t d = 1; d < D; ++d) {
                inside = inside + sum_pairs<R / 2>(in.across[d], i);
            }
        }
        out[i] = sum_pairs<R>(in.centre, i) - 2.0 * inside;
    }
}

// The couplings of N coarse cells along direction 0, the sums of the fine ones
// of the groups' low (Side 0) or high (Side 1) cells.
template <std::size_t Side, std::size_t N, std::size_t R>
inline void along_cells(lanes::cells<N>, const std::array<const double*, R>& rows,
                        std::size_t first, double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        out[i] = sum_side<Side, R>(rows, i);
    }
}

// And along a direction the rows do not lie along, from the rows on its low or
// its high side.
template <std::size_t N, std::size_t R>
inline void across_cells(lanes::cells<N>, const std::array<const double*, R>& rows,
                         std::size_t first, double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        out[i] = sum_pairs<R>(rows, i);
    }
}

// The fine rows of a group as pointers: the row of zeros where there is none.
template <int D>
row_set<D> rows_of(const std::vector<double>& field, const group<D>& g, const double* zeros) {
    row_set<D> rows{};
    for (std::size_t c = 0; c < group_rows<D>; ++c) {
        rows[c] = g.present[c] ? field.data() + g.cell[c] : zeros;
    }
    return rows;
}

// The last cells of rows of an odd length, each followed by the zero cell its
// group is padded with, so that a block of one reads them at element 0.
template <std::size_t R>
struct padded_ends {
    padded_ends() = default;
    padded_ends(const std::array<const double*, R>& rows, std::size_t last) {
        for (std::size_t c = 0; c < R; ++c) {
            values[c] = {rows[c][last], 0.0};
        }
    }
    std::array<const double*, R> rows() const {
        std::array<const double*, R> out{};
        for (std::size_t c = 0; c < R; ++c) {
            out[c] = values[c].data();
        }
        return out;
    }
    std::array<std::array<double, 2>, R> values{};
};

// Walks a coarse row as block(count, rows, first, at): the groups with two
// cells along direction 0 in blocks, then the last one of a fine row of an odd
// length n, on padded operands. A block's outputs start `at` cells beyond the
// row's first.
template <std::size_t R, typename Block>
CFDCORE_INLINE void for_each_group_block(const std::array<const double*, R>& rows, std::size_t n,
                                         Block&& block) {
    const std::size_t whole = n / 2;
    lanes::for_each_block(0, whole,
                          [&](auto count, std::size_t m) { block(count, rows, m, 0); });
    if (n % 2 == 1) {
        const padded_ends<R> ends(rows, n - 1);
        block(lanes::cells<1>{}, ends.rows(), 0, whole);
    }
}

template <std::size_t Side, std::size_t R>
void along_row(const std::array<const double*, R> rows, std::size_t n, double* out) {
    for_each_group_block(rows, n,
                         [&](auto count, const std::array<const double*, R>& in,
                             std::size_t first,
                             std::size_t at) { along_cells<Side>(count, in, first, out + at); });
}

template <std::size_t R>
void across_row(const std::array<const double*, R> rows, std::size_t n, double* out) {
    for_each_group_block(rows, n,
                         [&](auto count, const std::array<const double*, R>& in,
                             std::size_t first,
                             std::size_t at) { across_cells(count, in, first, out + at); });
}

template <int D>
void centre_row_of(const centre_row<D> in, std::size_t n, double* out) {
    const std::size_t whole = n / 2;
    lanes::for_each_block(0, whole,
                          [&](auto count, std::size_t m) { centre_cells(count, in, m, out); });
    if (n % 2 == 1) {
        const padded_ends<group_rows<D>> centre(in.centre, n - 1);
        const padded_ends<group_rows<D>> along(in.along, n - 1);
        std::array<padded_ends<group_rows<D> / 2>, D> across;
        centre_row<D> end;
        end.centre = centre.rows();
        end.along = along.rows();
        for (std::size_t d = 1; d < D; ++d) {
            across[d] = padded_ends<group_rows<D> / 2>(in.across[d], n - 1);
            end.across[d] = across[d].rows();
        }
        centre_cells(lanes::cells<1>{}, end, 0, out + whole);
    }
}

// The group sums along P rows of fine cells, in the reference's order: element
// `at` of each row in turn, added to `sum`.
template <std::size_t P, std::size_t C = 0>
CFDCORE_INLINE double add_rows(double sum, const std::array<const double*, P>& rows,
                               std::size_t at) {
    if constexpr (C == P) {
        return sum;
    } else {
        return add_rows<P, C + 1>(sum + rows[C][at], rows, at);
    }
}

// The group sums of N coarse cells over P rows of fine cells: the groups' low
// cells along direction 0, row after row, then, with Both, their high cells. A
// cell a group does not have adds nothing, so no row or cell of zeros stands
// for it.
template <bool Both, std::size_t N, std::size_t P>
inline void restrict_cells(lanes::cells<N>, const std::array<const double*, P>& rows,
                           std::size_t first, double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        double sum = add_rows<P, 1>(rows[0][2 * i], rows, 2 * i);
        if constexpr (Both) {
            sum = add_rows<P>(sum, rows, 2 * i + 1);
        }
        out[i] = sum;
    }
}

template <std::size_t P>
void restrict_row(const std::array<const double*, P> rows, std::size_t n, double* out) {
    const std::size_t whole = n / 2;
    lanes::for_each_block(0, whole, [&](auto count, std::size_t m) {
        restrict_cells<true>(count, rows, m, out);
    });
    if (n % 2 == 1) {
        restrict_cells<false>(lanes::cells<1>{}, rows, whole, out);
    }
}

// The same with `present` of at most P rows, the first of `rows`: a power of
// two, since a side's last group is one wide or two.
template <std::size_t P, std::size_t R>
void restrict_present(const std::array<const double*, R>& rows, std::size_t present,
                      std::size_t n, double* out) {
    if (present == P) {
        std::array<const double*, P> first{};
        std::copy_n(rows.begin(), P, first.begin());
        restrict_row(first, n, out);
    } else if constexpr (P > 1) {
        restrict_present<P / 2>(rows, present, n, out);
    }
}

// Each fine cell of a row gains its group's value: `x += P z` for N groups.
template <std::size_t N>
inline void prolong_cells(lanes::cells<N>, const double* z, std::size_t first,
                          double* CFDCORE_RESTRICT x) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        x[2 * i] = x[2 * i] + z[i];
        x[2 * i + 1] = x[2 * i + 1] + z[i];
    }
}

void prolong_row(const double* z, std::size_t n, double* x) {
    const std::size_t whole = n / 2;
    lanes::for_each_block(0, whole, [&](auto count, std::size_t m) {
        prolong_cells(count, z, m, x);
    });
    if (n % 2 == 1) {
        x[n - 1] = x[n - 1] + z[whole];
    }
}

// The same for every fine row of the groups of the coarse row at `cell`.
template <int D>
void prolong_rows(const layout::layout<D>& fine, const layout::layout<D>& coarse,
                  std::size_t cell, const double* z, double* x) {
    const group<D> g = group_of(fine, coarse, cell);
    for (std::size_t c = 0; c < group_rows<D>; ++c) {
        if (g.present[c]) {
            prolong_row(z + cell, fine.cells(0), x + g.cell[c]);
        }
    }
}

// The layers of a field: its rows grouped by their position along the last
// direction, in order; one layer where the rows lie along it. A cell's
// neighbours are in its own layer or the two beside it.
template <int D>
std::size_t layer_count(const layout::layout<D>& fields) {
    return D > 1 ? fields.cells(D - 1) : 1;
}

template <int D>
std::span<const layout::line> layer(const layout::layout<D>& fields, std::size_t at) {
    const auto rows = fields.lines(0);
    const std::size_t per_layer = rows.size() / layer_count(fields);
    return rows.subspan(at * per_layer, per_layer);
}

// Whether any of N rows' diagonals is below the sum of its couplings, beyond
// the tolerance: the couplings of each direction summed, then the directions.
template <std::size_t N, int D>
inline lanes::mask dominance_cells(lanes::cells<N>, const stencil_row<D>& in, std::size_t first,
                                   double allowance) {
    lanes::mask below = lanes::none;
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        double couplings = in.high[0][i] + in.low[0][i];
        for (int d = 1; d < D; ++d) {
            couplings = couplings + (in.high[d][i] + in.low[d][i]);
        }
        below |= lanes::mask_of(in.centre[i] * allowance < couplings);
    }
    return below;
}

template <int D>
lanes::mask dominance_row(const system<D>& s, std::size_t cell, double allowance) {
    const auto row = operands<D>(s, cell, nullptr, s.centre.data(), nullptr, nullptr);
    lanes::mask below = lanes::none;
    lanes::for_each_block(0, s.fields->cells(0), [&](auto count, std::size_t m) {
        below |= dominance_cells(count, row, m, allowance);
    });
    return below;
}

// The smoother of N cells: the weight over the diagonal, zero where that is not
// positive.
template <std::size_t N>
inline void smoother_cells(lanes::cells<N>, const double* centre, double weight,
                           std::size_t first, double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        out[i] = weight * lanes::keep(lanes::mask_of(centre[i] > 0.0), 1.0 / centre[i]);
    }
}

// The rows on one side of direction d of a set of a group's rows.
template <int D>
std::array<const double*, group_rows<D> / 2> side_of(const row_set<D>& rows, int d,
                                                     std::size_t side) {
    std::array<const double*, group_rows<D> / 2> out{};
    for (std::size_t m = 0; m < out.size(); ++m) {
        out[m] = rows[row_with(m, d, side)];
    }
    return out;
}

// One row of the Galerkin operator, the coarse row at `cell` (`coarsen`).
template <int D>
void coarsen_row(const system<D>& fine, const double* zeros, std::size_t cell, system<D>& out) {
    const std::size_t n = fine.fields->cells(0);
    const group<D> g = group_of(*fine.fields, *out.fields, cell);
    centre_row<D> in{};
    in.centre = rows_of<D>(fine.centre, g, zeros);
    in.along = rows_of<D>(fine.high[0], g, zeros);
    along_row<1>(in.along, n, out.high[0].data() + cell);
    along_row<0>(rows_of<D>(fine.low[0], g, zeros), n, out.low[0].data() + cell);
    if constexpr (D > 1) {
        for (int d = 1; d < D; ++d) {
            const row_set<D> high = rows_of<D>(fine.high[d], g, zeros);
            const row_set<D> low = rows_of<D>(fine.low[d], g, zeros);
            in.across[d] = side_of<D>(high, d, 0);
            across_row(side_of<D>(high, d, 1), n, out.high[d].data() + cell);
            across_row(side_of<D>(low, d, 0), n, out.low[d].data() + cell);
        }
    }
    centre_row_of(in, n, out.centre.data() + cell);
}

template <int D>
std::size_t cell_count_of(const system<D>& s) {
    return s.fields->cell_count();
}

template <int D>
bool is_coarsest(const layout::layout<D>& fields, std::int64_t coarsest_cells) {
    std::size_t shortest = fields.cells(0);
    for (int d = 1; d < D; ++d) {
        shortest = std::min(shortest, fields.cells(d));
    }
    return fields.cell_count() <= static_cast<std::size_t>(coarsest_cells) || shortest == 1;
}

// The cells of the level a hierarchy on `fields` stops at.
template <int D>
std::size_t coarsest_size(const layout::layout<D>& fields, std::int64_t coarsest_cells) {
    if (is_coarsest(fields, coarsest_cells)) {
        return fields.cell_count();
    }
    return coarsest_size(coarser(fields), coarsest_cells);
}

// The keys, or the reason they are refused.
const keys& checked(const keys& k) {
    if (k.coarsest_cells < 1) {
        throw std::invalid_argument("multigrid: coarsest_cells is at least 1");
    }
    if (k.sweeps < 1) {
        throw std::invalid_argument("multigrid: sweeps is at least 1");
    }
    if (!(std::isfinite(k.weight) && k.weight > 0.0)) {
        throw std::invalid_argument("multigrid: weight is finite and positive");
    }
    if (!(std::isfinite(k.dominance_tolerance) && k.dominance_tolerance >= 0.0)) {
        throw std::invalid_argument("multigrid: dominance_tolerance is finite and at least 0");
    }
    if (!(std::isfinite(k.rcond) && k.rcond >= 0.0)) {
        throw std::invalid_argument("multigrid: rcond is finite and at least 0");
    }
    return k;
}

}  // namespace

// ---------------------------------------------------------------------------

template <int D>
system<D>::system(const layout::layout<D>& given)
    : fields(&given), centre(given.cell_count(), 0.0) {
    for (int d = 0; d < D; ++d) {
        high[d].assign(given.cell_count(), 0.0);
        low[d].assign(given.cell_count(), 0.0);
    }
}

template <int D>
void product_row(const system<D>& s, const layout::line& line, std::span<const double> zeros,
                 std::span<const double> x, double* out) {
    product_cells_of(operands(s, line.cell, zeros.data(), x.data(), nullptr, nullptr),
                     s.fields->cells(0), out);
}

template <int D>
void residual_row(const system<D>& s, const layout::line& line, std::span<const double> zeros,
                  std::span<const double> x, std::span<const double> r, double* out) {
    residual_cells_of(operands(s, line.cell, zeros.data(), x.data(), r.data(), nullptr),
                      s.fields->cells(0), out);
}

template <int D>
void apply_operator(const system<D>& s, std::span<const double> zeros, std::span<const double> x,
                    std::span<double> out) {
    require_length(zeros.size(), s.fields->cells(0), "apply_operator: the row of zeros");
    require_length(x.size(), cell_count_of(s), "apply_operator: x");
    require_length(out.size(), cell_count_of(s), "apply_operator: out");
    for (const auto& line : s.fields->lines(0)) {
        product_row(s, line, zeros, x, out.data() + line.cell);
    }
}

template <int D>
void smooth(const system<D>& s, std::span<const double> zeros, std::span<const double> smoother,
            std::span<const double> x, std::span<const double> r, std::span<double> out) {
    require_length(zeros.size(), s.fields->cells(0), "smooth: the row of zeros");
    require_length(smoother.size(), cell_count_of(s), "smooth: the smoother");
    require_length(x.size(), cell_count_of(s), "smooth: x");
    require_length(r.size(), cell_count_of(s), "smooth: r");
    require_length(out.size(), cell_count_of(s), "smooth: out");
    smooth_field<false>(s, zeros.data(), smoother.data(), x.data(), r.data(), out.data());
}

template <int D>
layout::layout<D> coarser(const layout::layout<D>& fine) {
    std::array<std::size_t, D> cells{};
    for (int d = 0; d < D; ++d) {
        cells[d] = (fine.cells(d) + 1) / 2;
    }
    return layout::layout<D>(cells);
}

template <int D>
void restrict_to(const layout::layout<D>& fine, const layout::layout<D>& coarse,
                 std::span<const double> r, std::span<double> out) {
    require_length(r.size(), fine.cell_count(), "restrict_to: r");
    require_length(out.size(), coarse.cell_count(), "restrict_to: out");
    for (const auto& line : coarse.lines(0)) {
        const group<D> g = group_of(fine, coarse, line.cell);
        row_set<D> rows{};
        std::size_t present = 0;
        for (std::size_t c = 0; c < group_rows<D>; ++c) {
            if (g.present[c]) {
                rows[present++] = r.data() + g.cell[c];
            }
        }
        restrict_present<group_rows<D>>(rows, present, fine.cells(0), out.data() + line.cell);
    }
}

template <int D>
void add_prolonged(const layout::layout<D>& fine, const layout::layout<D>& coarse,
                   std::span<const double> z, std::span<double> x) {
    require_length(z.size(), coarse.cell_count(), "add_prolonged: z");
    require_length(x.size(), fine.cell_count(), "add_prolonged: x");
    for (const auto& line : coarse.lines(0)) {
        prolong_rows(fine, coarse, line.cell, z.data(), x.data());
    }
}

template <int D>
void coarsen(const system<D>& fine, std::span<const double> zeros, system<D>& out) {
    const auto& from = *fine.fields;
    const auto& to = *out.fields;
    for (int d = 0; d < D; ++d) {
        require_length(to.cells(d), (from.cells(d) + 1) / 2, "coarsen: the coarse layout");
    }
    require_length(zeros.size(), from.cells(0), "coarsen: the row of zeros");
    for (const auto& line : to.lines(0)) {
        coarsen_row(fine, zeros.data(), line.cell, out);
    }
}

// ---------------------------------------------------------------------------

template <int D>
    requires(D >= 1)
hierarchy<D>::hierarchy(const layout::layout<D>& fields, const keys& k)
    : keys_(checked(k)),
      eigen_(coarsest_size(fields, k.coarsest_cells)),
      matrix_(eigen_.size() * eigen_.size(), 0.0),
      inverse_(matrix_.size(), 0.0),
      zeros_(fields.cells(0), 0.0) {
    levels_.emplace_back(fields);
    while (true) {
        level& l = levels_.back();
        const std::size_t n = l.fields->cell_count();
        const bool coarsest = is_coarsest(*l.fields, keys_.coarsest_cells);
        if (levels_.size() > 1) {
            l.own = std::make_unique<system<D>>(*l.fields);
            l.r.assign(n, 0.0);
            l.x.assign(n, 0.0);
        }
        if (coarsest) {
            break;
        }
        l.smoother.assign(n, 0.0);
        l.x.assign(n, 0.0);
        l.other.assign(n, 0.0);
        l.rows.assign(l.fields->cells(0) * group_rows<D>, 0.0);
        auto next = std::make_unique<layout::layout<D>>(coarser(*l.fields));
        levels_.emplace_back(*next);
        levels_.back().own_fields = std::move(next);
    }
}

template <int D>
    requires(D >= 1)
const system<D>& hierarchy<D>::level_system(std::size_t index) const {
    return index == 0 ? *finest_ : *levels_[index].own;
}

template <int D>
    requires(D >= 1)
build_status hierarchy<D>::build(const system<D>& s) {
    for (int d = 0; d < D; ++d) {
        require_length(s.fields->cells(d), levels_.front().fields->cells(d),
                       "multigrid: the system is on another layout");
    }
    finest_ = &s;
    // The finest level's dominance, each level's smoother and the next level's
    // operator, a coarse layer at a time, so that each pass finds the rows of
    // coefficients where the one before it left them. The coarse levels inherit
    // dominance from the finest through the sums.
    lanes::mask below = lanes::none;
    const double allowance = 1.0 + keys_.dominance_tolerance;
    if (levels_.size() == 1) {
        for (const auto& line : s.fields->lines(0)) {
            below |= dominance_row(s, line.cell, allowance);
        }
    }
    for (std::size_t l = 0; l + 1 < levels_.size(); ++l) {
        const system<D>& from = level_system(l);
        system<D>& to = *levels_[l + 1].own;
        const auto& fine = *from.fields;
        double* smoother = levels_[l].smoother.data();
        for (std::size_t at = 0; at < layer_count(*to.fields); ++at) {
            for (std::size_t m = 2 * at; m < std::min(2 * at + 2, layer_count(fine)); ++m) {
                for (const auto& line : layer(fine, m)) {
                    if (l == 0) {
                        below |= dominance_row(from, line.cell, allowance);
                    }
                    lanes::for_each_block(0, fine.cells(0), [&](auto count, std::size_t i) {
                        smoother_cells(count, from.centre.data() + line.cell, keys_.weight, i,
                                       smoother + line.cell);
                    });
                }
            }
            for (const auto& line : layer(*to.fields, at)) {
                coarsen_row(from, zeros_.data(), line.cell, to);
            }
        }
        if (below != lanes::none) {
            break;
        }
    }
    if (below != lanes::none) {
        finest_ = nullptr;
        return build_status::not_dominant;
    }
    form_matrix(level_system(levels_.size() - 1));
    if (eigen_.pseudo_inverse(matrix_, keys_.rcond, inverse_) != dense::eigen_status::ok) {
        finest_ = nullptr;
        return build_status::not_converged;
    }
    return build_status::ok;
}

// The coarsest operator as a dense matrix (`_dense_pseudo_inverse`): each entry
// zero less the coupling toward the neighbour.
template <int D>
    requires(D >= 1)
void hierarchy<D>::form_matrix(const system<D>& s) {
    const auto& fields = *s.fields;
    const std::size_t n = fields.cell_count();
    std::fill(matrix_.begin(), matrix_.end(), 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        matrix_[i * n + i] = s.centre[i];
        for (int d = 0; d < D; ++d) {
            const std::size_t stride = fields.cell_stride(d);
            const std::size_t position = (i / stride) % fields.cells(d);
            if (position + 1 < fields.cells(d)) {
                matrix_[i * n + i + stride] = 0.0 - s.high[d][i];
            }
            if (position > 0) {
                matrix_[i * n + i - stride] = 0.0 - s.low[d][i];
            }
        }
    }
}

// z = the pseudo-inverse times r: each value one block of `reduce`'s dot product.
template <int D>
    requires(D >= 1)
void hierarchy<D>::solve_coarsest(std::span<const double> r, std::span<double> z) const {
    const std::size_t n = eigen_.size();
    for (std::size_t i = 0; i < n; ++i) {
        z[i] = reduce::block_dot(std::span<const double>(inverse_).subspan(i * n, n), r);
    }
}

template <int D>
    requires(D >= 1)
void hierarchy<D>::cycle(std::span<const double> r, std::span<double> z) {
    if (finest_ == nullptr) {
        throw std::invalid_argument("multigrid: cycle before build");
    }
    const std::size_t n = levels_.front().fields->cell_count();
    require_length(r.size(), n, "multigrid: r");
    require_length(z.size(), n, "multigrid: z");
    const std::size_t last = levels_.size() - 1;
    if (last == 0) {
        solve_coarsest(r, z);
        for (std::size_t i = 0; i < n; ++i) {
            z[i] = lanes::keep(lanes::mask_of(finest_->centre[i] > 0.0), z[i]);
        }
        return;
    }
    for (std::size_t index = 0; index < last; ++index) {
        descend(index, index == 0 ? r.data() : levels_[index].r.data());
    }
    solve_coarsest(levels_[last].r, levels_[last].x);
    for (std::size_t index = last; index-- > 0;) {
        ascend(index, index == 0 ? r.data() : levels_[index].r.data(),
               index == 0 ? z.data() : nullptr);
    }
}

// Level `index` on the way down: the sweeps from zero on the residual r, then
// the group sums of what they leave, as the next level's residual.
//
// The passes walk the field together, a layer at a time, so that a layer is
// still in the cache when the next pass reads it: the group sums of a coarse
// layer read the iterate no further than the fine layer after its own.
template <int D>
    requires(D >= 1)
void hierarchy<D>::descend(std::size_t index, const double* r) {
    level& l = levels_[index];
    level& next = levels_[index + 1];
    const system<D>& s = level_system(index);
    const auto& fine = *l.fields;
    const auto& coarse = *next.fields;
    const std::size_t n0 = fine.cells(0);
    const std::size_t layers = layer_count(fine);
    std::size_t formed = 0;
    const auto form_to = [&](std::size_t end) {
        for (; formed < std::min(end, layers); ++formed) {
            for (const auto& line : layer(fine, formed)) {
                lanes::for_each_block(0, n0, [&](auto count, std::size_t m) {
                    first_cells(count, l.smoother.data() + line.cell, r + line.cell, m,
                                l.x.data() + line.cell);
                });
            }
        }
    };
    if (keys_.sweeps > 1) {
        form_to(layers);
    }
    for (std::int64_t sweep = 1; sweep < keys_.sweeps; ++sweep) {
        smooth_field<false>(s, zeros_.data(), l.smoother.data(), l.x.data(), r, l.other.data());
        std::swap(l.x, l.other);
    }
    for (std::size_t at = 0; at < layer_count(coarse); ++at) {
        form_to(2 * at + 3);
        for (const auto& line : layer(coarse, at)) {
            const group<D> g = group_of(fine, coarse, line.cell);
            row_set<D> rows{};
            std::size_t present = 0;
            for (std::size_t c = 0; c < group_rows<D>; ++c) {
                if (g.present[c]) {
                    double* row = l.rows.data() + present * n0;
                    residual_cells_of(
                        operands(s, g.cell[c], zeros_.data(), l.x.data(), r, nullptr), n0, row);
                    rows[present++] = row;
                }
            }
            restrict_present<group_rows<D>>(rows, present, n0, next.r.data() + line.cell);
        }
    }
}

// Level `index` on the way up: the next level's correction, then the sweeps
// again. The finest level's last sweep writes the cycle's result to z, zero
// where there is no equation; another level's result is left in its x. The
// correction and the first sweep walk together, as `descend`'s passes do.
template <int D>
    requires(D >= 1)
void hierarchy<D>::ascend(std::size_t index, const double* r, double* z) {
    level& l = levels_[index];
    const level& next = levels_[index + 1];
    const system<D>& s = level_system(index);
    const auto& fine = *l.fields;
    const auto& coarse = *next.fields;
    const std::size_t n0 = fine.cells(0);
    const std::size_t coarse_layers = layer_count(coarse);
    std::size_t corrected = 0;
    for (std::int64_t sweep = 0; sweep < keys_.sweeps; ++sweep) {
        const bool to_result = index == 0 && sweep + 1 == keys_.sweeps;
        double* out = to_result ? z : l.other.data();
        for (std::size_t at = 0; at < layer_count(fine); ++at) {
            // The layers the sweep of this one reads, up to the one after it.
            for (; corrected < std::min((at + 1) / 2 + 1, coarse_layers); ++corrected) {
                for (const auto& line : layer(coarse, corrected)) {
                    prolong_rows(fine, coarse, line.cell, next.x.data(), l.x.data());
                }
            }
            for (const auto& line : layer(fine, at)) {
                const auto row = operands(s, line.cell, zeros_.data(), l.x.data(), r,
                                          l.smoother.data());
                if (to_result) {
                    smooth_cells_of<true>(row, n0, out + line.cell);
                } else {
                    smooth_cells_of<false>(row, n0, out + line.cell);
                }
            }
        }
        if (!to_result) {
            std::swap(l.x, l.other);
        }
    }
}

#define CFDCORE_MULTIGRID_INSTANTIATE(D)                                                          \
    template struct system<D>;                                                                    \
    template class hierarchy<D>;                                                                  \
    template void product_row<D>(const system<D>&, const layout::line&, std::span<const double>,  \
                                 std::span<const double>, double*);                               \
    template void residual_row<D>(const system<D>&, const layout::line&, std::span<const double>, \
                                  std::span<const double>, std::span<const double>, double*);     \
    template void apply_operator<D>(const system<D>&, std::span<const double>,                    \
                                    std::span<const double>, std::span<double>);                  \
    template void smooth<D>(const system<D>&, std::span<const double>, std::span<const double>,   \
                            std::span<const double>, std::span<const double>, std::span<double>); \
    template layout::layout<D> coarser<D>(const layout::layout<D>&);                              \
    template void restrict_to<D>(const layout::layout<D>&, const layout::layout<D>&,              \
                                 std::span<const double>, std::span<double>);                     \
    template void add_prolonged<D>(const layout::layout<D>&, const layout::layout<D>&,            \
                                   std::span<const double>, std::span<double>);                   \
    template void coarsen<D>(const system<D>&, std::span<const double>, system<D>&);

CFDCORE_MULTIGRID_INSTANTIATE(1)
CFDCORE_MULTIGRID_INSTANTIATE(2)

#undef CFDCORE_MULTIGRID_INSTANTIATE

}  // namespace cfdcore::multigrid
