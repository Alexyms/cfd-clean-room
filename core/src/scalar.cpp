#include "cfdcore/scalar.hpp"

#include "cfdcore/lanes.hpp"
#include "cfdcore/reduce.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>

namespace cfdcore::scalar {

namespace {

// One line of cells along a direction, with its two boundary nodes: node m is
// the low boundary node for m = 0, cell m - 1 for m = 1..n, and the high
// boundary node for m = n + 1, as the reference's `ext` holds them.
template <int D>
class line_view {
public:
    line_view(const grid::grid<D>& g, std::span<const double> c, const layout::line& line,
              std::size_t stride, std::size_t n, double low, double high)
        : g_(g), c_(c), cell_(line.cell), stride_(stride), n_(n), low_(low), high_(high) {}

    double node(std::size_t m) const {
        if (m == 0) {
            return low_;
        }
        if (m == n_ + 1) {
            return high_;
        }
        return c_[cell_ + (m - 1) * stride_];
    }

    // mesh_axes pads the SOLID mask with one False on each side.
    bool solid_node(std::size_t m) const {
        if (m == 0 || m == n_ + 1) {
            return false;
        }
        return g_.solid(cell_ + (m - 1) * stride_);
    }

private:
    const grid::grid<D>& g_;
    std::span<const double> c_;
    std::size_t cell_;
    std::size_t stride_;
    std::size_t n_;
    double low_;
    double high_;
};

// QUICK's face value at interior face f (between cells f and f + 1, nodes
// f + 1 and f + 2), as quick_face_values forms f_pos and f_neg: three
// products added left to right.
template <int D>
double quick_at(const quick_weights& w, std::size_t f, bool positive, const line_view<D>& line) {
    if (positive) {
        return w.positive[0][f] * line.node(f) + w.positive[1][f] * line.node(f + 1) +
               w.positive[2][f] * line.node(f + 2);
    }
    return w.negative[0][f] * line.node(f + 3) + w.negative[1][f] * line.node(f + 2) +
           w.negative[2][f] * line.node(f + 1);
}

}  // namespace

void limited_face_values(std::span<const double> c_up, std::span<const double> c_c,
                         std::span<const double> c_d, std::span<const double> quick,
                         std::span<double> out) {
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = limited_face_value(c_up[i], c_c[i], c_d[i], quick[i]);
    }
}

template <int D>
    requires(D >= 1)
scheme<D>::scheme(const grid::grid<D>& g) : grid_(&g) {
    for (int d = 0; d < D; ++d) {
        const auto nodes = g.nodes(d);
        const auto faces = g.interior_faces(d);
        auto& w = weights_[d];
        for (int i = 0; i < 3; ++i) {
            w.positive[i].resize(faces.size());
            w.negative[i].resize(faces.size());
        }
        // quick_face_values with left k = f + 1: far_pos = k - 1 and
        // far_neg = k + 2, both inside the nodes, which run from the low
        // boundary face to the high one.
        for (std::size_t f = 0; f < faces.size(); ++f) {
            const auto pos = lagrange_weights(nodes[f], nodes[f + 1], nodes[f + 2], faces[f]);
            const auto neg = lagrange_weights(nodes[f + 3], nodes[f + 2], nodes[f + 1], faces[f]);
            for (int i = 0; i < 3; ++i) {
                w.positive[i][f] = pos[i];
                w.negative[i][f] = neg[i];
            }
        }
    }
}

template <int D>
void quick_face_values(const scheme<D>& s, int d, std::span<const double> c,
                       std::span<const double> c_low, std::span<const double> c_high,
                       std::span<const bool> positive, std::span<double> out) {
    const auto& g = s.mesh();
    const auto& fields = g.fields();
    const std::size_t n = fields.cells(d);
    const std::size_t stride = fields.cell_stride(d);
    const auto& w = s.weights(d);
    const auto lines = fields.lines(d);
    for (std::size_t l = 0; l < lines.size(); ++l) {
        const line_view<D> line(g, c, lines[l], stride, n, c_low[l], c_high[l]);
        for (std::size_t f = 0; f + 1 < n; ++f) {
            const std::size_t face = lines[l].face + (f + 1) * stride;
            out[face] = quick_at(w, f, positive[face], line);
        }
    }
}

template <int D>
void advective_flux(const scheme<D>& s, int d, std::span<const double> c,
                    std::span<const double> flux, std::span<const double> inflow, bool upwind,
                    std::span<const bool> vents, std::span<double> out) {
    const auto& g = s.mesh();
    const auto& fields = g.fields();
    const std::size_t n = fields.cells(d);
    const std::size_t stride = fields.cell_stride(d);
    const auto& w = s.weights(d);
    const bool with_vents = !vents.empty();
    for (const auto& ln : fields.lines(d)) {
        const std::size_t low_face = ln.face;
        const std::size_t high_face = ln.face + n * stride;
        const double c_low = flux[low_face] > 0.0 ? inflow[low_face] : c[ln.cell];
        const double c_high =
            flux[high_face] < 0.0 ? inflow[high_face] : c[ln.cell + (n - 1) * stride];
        const line_view<D> line(g, c, ln, stride, n, c_low, c_high);
        out[low_face] = flux[low_face] * c_low;
        // Interior face f lies between nodes k = f + 1 and k + 1, cells f and f + 1.
        for (std::size_t f = 0; f + 1 < n; ++f) {
            const std::size_t face = ln.face + (f + 1) * stride;
            const std::size_t k = f + 1;
            const double interior = flux[face];
            const bool positive = interior > 0.0;
            const double c_c = positive ? line.node(k) : line.node(k + 1);
            double value = c_c;
            if (!upwind) {
                const double c_d = positive ? line.node(k + 1) : line.node(k);
                double c_up = positive ? line.node(k - 1) : line.node(k + 2);
                double quick = quick_at(w, f, positive, line);
                // A far node inside an obstacle is read as the upstream value,
                // the zero-gradient rule a domain wall gets.
                const bool far_solid = positive ? line.solid_node(k - 1) : line.solid_node(k + 2);
                if (far_solid) {
                    c_up = c_c;
                    quick = c_c;
                }
                value = limited_face_value(c_up, c_c, c_d, quick);
            }
            if (with_vents && vents[face]) {
                const bool solid_low = line.solid_node(k);
                const bool solid_high = line.solid_node(k + 1);
                const bool entering =
                    (solid_low && interior > 0.0) || (solid_high && interior < 0.0);
                const double room = solid_low ? line.node(k + 1) : line.node(k);
                value = entering ? inflow[face] : room;
            }
            out[face] = interior * value;
        }
        out[high_face] = flux[high_face] * c_high;
    }
}

// ---------------------------------------------------------------------------
// The implicit solve.

namespace {

// One row of the cell field (a line along direction 0) and what the stencil
// of its cells needs along every direction: the row's first cell and low face
// normal to direction 0, and for each direction d >= 1 the stride, the offset
// from a cell to its low face normal to d, and whether the row has a
// neighbouring row below and above. Formed once per row, so no cell's
// arithmetic holds a division for an index.
template <int D>
struct row_stencil {
    std::size_t cell = 0;
    std::size_t face0 = 0;
    std::size_t n0 = 0;
    std::array<std::size_t, D> stride{};
    std::array<std::size_t, D> face_offset{};
    std::array<bool, D> has_low{};
    std::array<bool, D> has_high{};
};

template <int D>
row_stencil<D> stencil_of(const layout::layout<D>& fields, const layout::line& row) {
    row_stencil<D> s;
    s.cell = row.cell;
    s.face0 = row.face;
    s.n0 = fields.cells(0);
    for (int d = 1; d < D; ++d) {
        const std::size_t stride = fields.cell_stride(d);
        const std::size_t position = (row.cell / stride) % fields.cells(d);
        s.stride[d] = stride;
        // layout::low_face, the same for every cell of the row.
        s.face_offset[d] = (row.cell / (stride * fields.cells(d))) * stride;
        s.has_low[d] = position > 0;
        s.has_high[d] = position + 1 < fields.cells(d);
    }
    return s;
}

// Whether a cell is not an unknown of the CG solve: SOLID or held.
inline bool is_fixed(std::span<const bool> solid, std::span<const bool> held, std::size_t i) {
    return solid[i] || (!held.empty() && held[i]);
}

// What the stencil of one row's cells reads, each pointer at the row's first
// cell, so that cell m of the row is element m of every array.
//
// x:      the field whose neighbours are summed; x[m - 1] and x[m + 1] are
//         read, so only the cells that have both neighbours in the row are
//         reached through it (the two end cells go through `end_cell`)
// g:      direction 0's conductances, g[m] on cell m's low face and g[m + 1]
//         on its high one
// x_low, x_high, g_low, g_high: for each direction d >= 1, the rows of x on
//         the low and the high side (a row of zeros where the field ends, the
//         reference's padding) and the conductances on the cells' low and
//         high faces normal to d; entry 0 is not used
// b, a_p: the system's right-hand side and diagonal
// free:   set where the cell is an unknown (not SOLID, not held)
// fluid:  set where the cell is not SOLID
template <int D>
struct stencil_row {
    const double* x = nullptr;
    const double* g = nullptr;
    const double* x_low[D] = {};
    const double* x_high[D] = {};
    const double* g_low[D] = {};
    const double* g_high[D] = {};
    const double* b = nullptr;
    const double* a_p = nullptr;
    const lanes::mask* free = nullptr;
    const lanes::mask* fluid = nullptr;
};

// The operands of a row for the field `x`, on a system `implicit_system` set
// up, with the masks `set_masks` formed and the `free` array it returned.
template <int D>
stencil_row<D> operands(const row_stencil<D>& s,
                        const std::array<std::span<const double>, D>& conductance,
                        const double* x, const implicit_scratch<D>& scratch,
                        const lanes::mask* free) {
    stencil_row<D> row;
    row.x = x + s.cell;
    row.g = conductance[0].data() + s.face0;
    for (int d = 1; d < D; ++d) {
        const std::size_t low = s.cell + s.face_offset[d];
        row.g_low[d] = conductance[d].data() + low;
        row.g_high[d] = conductance[d].data() + low + s.stride[d];
        row.x_low[d] = s.has_low[d] ? x + s.cell - s.stride[d] : scratch.zeros.data();
        row.x_high[d] = s.has_high[d] ? x + s.cell + s.stride[d] : scratch.zeros.data();
    }
    row.b = scratch.b.data() + s.cell;
    row.a_p = scratch.a_p.data() + s.cell;
    row.free = free + s.cell;
    row.fluid = scratch.fluid.data() + s.cell;
    return row;
}

// The operands of cell m of an n-cell row alone, as element 0. Its neighbours
// along direction 0 are read from `window`, which is filled with the cell's
// value between its two neighbours', a zero standing for a neighbour beyond
// the row's end as the reference's padding does. The window must outlive the
// operands.
template <int D>
stencil_row<D> end_cell(const stencil_row<D>& row, std::size_t m, std::size_t n,
                        std::array<double, 3>& window) {
    window = {m > 0 ? row.x[m - 1] : 0.0, row.x[m], m + 1 < n ? row.x[m + 1] : 0.0};
    stencil_row<D> cell = row;
    cell.x = window.data() + 1;
    cell.g = row.g + m;
    for (int d = 1; d < D; ++d) {
        cell.x_low[d] = row.x_low[d] + m;
        cell.x_high[d] = row.x_high[d] + m;
        cell.g_low[d] = row.g_low[d] + m;
        cell.g_high[d] = row.g_high[d] + m;
    }
    cell.b = row.b + m;
    cell.a_p = row.a_p + m;
    cell.free = row.free + m;
    cell.fluid = row.fluid + m;
    return cell;
}

// `_neighbour_sum` at element i: sum_f G_f x_N, the terms added east, west
// (direction 0's high then low face), then each further direction's high then
// low face: north, south for D = 2.
template <int D>
CFDCORE_INLINE double neighbour_sum(const stencil_row<D>& in, std::size_t i) {
    double sum = in.g[i + 1] * in.x[i + 1] + in.g[i] * in.x[i - 1];
    for (int d = 1; d < D; ++d) {
        sum = sum + in.g_high[d][i] * in.x_high[d][i];
        sum = sum + in.g_low[d][i] * in.x_low[d][i];
    }
    return sum;
}

// The residual r = (b + sum_f G_f x_N) - a_P x at element i, zero where the
// cell is not an unknown.
template <int D>
CFDCORE_INLINE double residual_at(const stencil_row<D>& in, std::size_t i) {
    return lanes::keep(in.free[i], (in.b[i] + neighbour_sum(in, i)) - in.a_p[i] * in.x[i]);
}

// Whether a residual fails the stop: the reference's `np.abs(r).max() <= stop`
// holds exactly when no cell's mask is set (a NaN residual, or a NaN stop,
// sets it).
CFDCORE_INLINE lanes::mask outside_stop(double r, double stop) {
    return lanes::mask_of(!(std::fabs(r) <= stop));
}

// One Jacobi sweep over the N cells from element `first` (lanes.hpp): the
// residual into r, the next iterate x + r / a_P into next with SOLID cells at
// zero. Returns the cells' `outside_stop` masks, ORed. r and next overlap
// nothing the operands point to.
template <std::size_t N, int D>
inline lanes::mask sweep_cells(lanes::cells<N>, const stencil_row<D>& in, std::size_t first,
                               double stop, double* CFDCORE_RESTRICT r,
                               double* CFDCORE_RESTRICT next) {
    lanes::mask outside = lanes::none;
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double ri = residual_at(in, i);
        r[i] = ri;
        outside |= outside_stop(ri, stop);
        next[i] = lanes::keep(in.fluid[i], in.x[i] + ri / in.a_p[i]);
    }
    return outside;
}

// Walks the n cells of a row for a kernel that reads each cell's neighbours
// along direction 0: the two end cells apart, each on a window holding the
// zero beyond the row, and the cells between in blocks. `block` is called as
// block(count, operands, first, shift): `count` cells from element `first` of
// `operands`, whose outputs lie `shift` cells beyond the row's first.
template <int D, typename Block>
CFDCORE_INLINE void for_each_stencil_block(const stencil_row<D>& row, std::size_t n,
                                           Block&& block) {
    std::array<double, 3> window{};
    block(lanes::cells<1>{}, end_cell(row, 0, n, window), 0, 0);
    lanes::for_each_block(1, n - 1, [&](auto count, std::size_t m) { block(count, row, m, 0); });
    if (n > 1) {
        block(lanes::cells<1>{}, end_cell(row, n - 1, n, window), 0, n - 1);
    }
}

// One Jacobi sweep over a row of n cells; r and next are at the row's first
// cell. The operands arrive by value: a copy that nothing else can reach, so
// the compiler holds its pointers in registers through the blocks.
template <int D>
lanes::mask sweep_row(const stencil_row<D> row, std::size_t n, double stop, double* r,
                      double* next) {
    lanes::mask outside = lanes::none;
    for_each_stencil_block(row, n,
                           [&](auto count, const stencil_row<D>& in, std::size_t first,
                               std::size_t shift) {
                               outside |= sweep_cells(count, in, first, stop, r + shift,
                                                      next + shift);
                           });
    return outside;
}

// The residual alone over N cells from element `first`, as `sweep_cells`
// forms it: the conjugate gradient's true residual.
template <std::size_t N, int D>
inline lanes::mask residual_cells(lanes::cells<N>, const stencil_row<D>& in, std::size_t first,
                                  double stop, double* CFDCORE_RESTRICT r) {
    lanes::mask outside = lanes::none;
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double ri = residual_at(in, i);
        r[i] = ri;
        outside |= outside_stop(ri, stop);
    }
    return outside;
}

template <int D>
lanes::mask residual_row(const stencil_row<D> row, std::size_t n, double stop, double* r) {
    lanes::mask outside = lanes::none;
    for_each_stencil_block(row, n,
                           [&](auto count, const stencil_row<D>& in, std::size_t first,
                               std::size_t shift) {
                               outside |= residual_cells(count, in, first, stop, r + shift);
                           });
    return outside;
}

// The matrix product q = a_P x - sum_f G_f x_N over N cells from element
// `first`, zero where the cell is not an unknown.
template <std::size_t N, int D>
inline void product_cells(lanes::cells<N>, const stencil_row<D>& in, std::size_t first,
                          double* CFDCORE_RESTRICT q) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        q[i] = lanes::keep(in.free[i], in.a_p[i] * in.x[i] - neighbour_sum(in, i));
    }
}

template <int D>
void product_row(const stencil_row<D> row, std::size_t n, double* q) {
    for_each_stencil_block(row, n,
                           [&](auto count, const stencil_row<D>& in, std::size_t first,
                               std::size_t shift) {
                               product_cells(count, in, first, q + shift);
                           });
}

// The conjugate gradient's elementwise steps over N cells of a row from
// element `first`. Every output is restrict-qualified and overlaps no other
// array of its call; an array updated in place is read through its output.

// The start: c = C* with the SOLID cells at zero, and the preconditioner
// 1 / a_P, zero where the cell is not an unknown.
template <std::size_t N>
inline void start_cells(lanes::cells<N>, std::size_t first, const double* c_star,
                        const double* a_p, const lanes::mask* free, const lanes::mask* fluid,
                        double* CFDCORE_RESTRICT c, double* CFDCORE_RESTRICT inverse) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        c[i] = lanes::keep(fluid[i], c_star[i]);
        inverse[i] = lanes::keep(free[i], 1.0 / a_p[i]);
    }
}

// A restart: z = inverse * r and p = z.
template <std::size_t N>
inline void restart_cells(lanes::cells<N>, std::size_t first, const double* inverse,
                          const double* r, double* CFDCORE_RESTRICT z,
                          double* CFDCORE_RESTRICT p) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double zi = inverse[i] * r[i];
        z[i] = zi;
        p[i] = zi;
    }
}

// An iteration's update: c <- c + alpha p and r <- r - alpha q, and beside
// them z = inverse * r, which the next direction reads unless this residual
// ends the iteration. Returns the residuals' `outside_stop` masks, ORed.
template <std::size_t N>
inline lanes::mask update_cells(lanes::cells<N>, std::size_t first, double alpha, double stop,
                                const double* p, const double* q, const double* inverse,
                                double* CFDCORE_RESTRICT c, double* CFDCORE_RESTRICT r,
                                double* CFDCORE_RESTRICT z) {
    lanes::mask outside = lanes::none;
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        c[i] = c[i] + alpha * p[i];
        const double ri = r[i] - alpha * q[i];
        r[i] = ri;
        outside |= outside_stop(ri, stop);
        z[i] = inverse[i] * ri;
    }
    return outside;
}

// The next direction: p <- p * beta, then p <- p + z.
template <std::size_t N>
inline void direction_cells(lanes::cells<N>, std::size_t first, double beta, const double* z,
                            double* CFDCORE_RESTRICT p) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        p[i] = p[i] * beta + z[i];
    }
}

// What the set-up of one row's cells reads, each pointer at the row's first
// cell.
//
// g:              direction 0's conductances, g[m] on cell m's low face and
//                 g[m + 1] on its high one
// g_low, g_high:  for each direction d >= 1, the conductances on the cells'
//                 low and high faces normal to d; entry 0 is not used
// sink, volume, c_star: the solve's inputs of those names
// dt_cells:       the per-cell step, read only by the per-cell set-up
template <int D>
struct setup_row {
    const double* g = nullptr;
    const double* g_low[D] = {};
    const double* g_high[D] = {};
    const double* sink = nullptr;
    const double* volume = nullptr;
    const double* c_star = nullptr;
    const double* dt_cells = nullptr;
};

// What a set-up pass found: `any` set if a cell's diagonal has a non-zero
// extra term (the reference's `diag_extra.any()`: a NaN counts, a -0.0 does
// not), `nan` set if a |b| is NaN.
struct setup_flags {
    lanes::mask any = lanes::none;
    lanes::mask nan = lanes::none;
};

// The set-up of N cells from element `first` (lanes.hpp): the diagonal's extra
// terms west, east (direction 0's low then high face), each further
// direction's low then high face, then the sink; a_P = V / dt + that;
// b = (V / dt) C*. `largest` holds one running maximum of |b| per cell of a
// block, lanes::width of them: a maximum is the same in any order, and a NaN
// is left out of it and flagged. PerCell reads the step per cell, else `dt`.
// a_p, b and largest overlap nothing the operands point to.
template <bool PerCell, std::size_t N, int D>
inline setup_flags setup_cells(lanes::cells<N>, const setup_row<D>& in, std::size_t first,
                               double dt, double* CFDCORE_RESTRICT a_p,
                               double* CFDCORE_RESTRICT b, double* CFDCORE_RESTRICT largest) {
    lanes::mask any = lanes::none;
    lanes::mask nan = lanes::none;
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        double extra = in.g[i] + in.g[i + 1];
        for (int d = 1; d < D; ++d) {
            extra = extra + in.g_low[d][i];
            extra = extra + in.g_high[d][i];
        }
        extra = extra + in.sink[i];
        any |= lanes::mask_of(extra != 0.0);
        const double over_dt = PerCell ? in.volume[i] / in.dt_cells[i] : in.volume[i] / dt;
        a_p[i] = over_dt + extra;
        const double bi = over_dt * in.c_star[i];
        b[i] = bi;
        const double magnitude = std::fabs(bi);
        nan |= lanes::mask_of(magnitude != magnitude);
        largest[k] = magnitude > largest[k] ? magnitude : largest[k];
    }
    return {any, nan};
}

// The set-up of a row of n cells, in blocks and then the cells left. a_p and
// b are at the row's first cell.
template <bool PerCell, int D>
setup_flags setup_row_cells(const setup_row<D> row, std::size_t n, double dt, double* a_p,
                            double* b, double* largest) {
    setup_flags flags;
    lanes::for_each_block(0, n, [&](auto count, std::size_t m) {
        const setup_flags f = setup_cells<PerCell>(count, row, m, dt, a_p, b, largest);
        flags.any |= f.any;
        flags.nan |= f.nan;
    });
    return flags;
}

// Forms the masks of a solve's cells in the scratch, `fluid` where not SOLID
// and `free` where neither SOLID nor held, and returns the `free` array to
// read: with no held cell the two are one array, so a block reads one.
template <int D>
const lanes::mask* set_masks(const implicit_input<D>& in, implicit_scratch<D>& scratch) {
    lanes::masks_where_not(in.solid, scratch.fluid);
    if (in.held.empty()) {
        return scratch.fluid.data();
    }
    for (std::size_t i = 0; i < scratch.free.size(); ++i) {
        scratch.free[i] = scratch.fluid[i] & lanes::mask_of(!in.held[i]);
    }
    return scratch.free.data();
}

}  // namespace

template <int D>
    requires(D >= 1)
implicit_scratch<D>::implicit_scratch(const layout::layout<D>& f)
    : fields(&f),
      a_p(f.cell_count(), 0.0),
      b(f.cell_count(), 0.0),
      c(f.cell_count(), 0.0),
      next(f.cell_count(), 0.0),
      residual(f.cell_count(), 0.0),
      inverse(f.cell_count(), 0.0),
      z(f.cell_count(), 0.0),
      p(f.cell_count(), 0.0),
      q(f.cell_count(), 0.0),
      dt_masked(f.cell_count(), 0.0),
      terms(f.cell_count(), 0.0),
      free(f.cell_count(), lanes::all),
      fluid(f.cell_count(), lanes::all),
      zeros(f.cells(0), 0.0) {}

implicit_status check_leftover_bound(std::optional<double> bound) {
    // Written as "not >= 0" so that a NaN bound is refused as well.
    if (bound.has_value() && !(*bound >= 0.0)) {
        return implicit_status::bound_negative_or_nan;
    }
    return implicit_status::ok;
}

template <int D>
implicit_setup implicit_system(const implicit_input<D>& in, implicit_scratch<D>& scratch) {
    const auto& fields = *scratch.fields;
    const std::size_t n0 = fields.cells(0);
    const bool per_cell = !in.dt_cells.empty();
    double* a_p = scratch.a_p.data();
    double* b = scratch.b.data();
    // One pass: the diagonal, the right-hand side and what the stop reads.
    std::array<double, lanes::width> largest{};
    setup_flags flags;
    for (const auto& line : fields.lines(0)) {
        const auto s = stencil_of(fields, line);
        setup_row<D> row;
        row.g = in.conductance[0].data() + s.face0;
        for (int d = 1; d < D; ++d) {
            const std::size_t low = s.cell + s.face_offset[d];
            row.g_low[d] = in.conductance[d].data() + low;
            row.g_high[d] = in.conductance[d].data() + low + s.stride[d];
        }
        row.sink = in.sink.data() + s.cell;
        row.volume = in.volume.data() + s.cell;
        row.c_star = in.c_star.data() + s.cell;
        row.dt_cells = per_cell ? in.dt_cells.data() + s.cell : nullptr;
        const setup_flags f =
            per_cell ? setup_row_cells<true>(row, n0, in.dt, a_p + s.cell, b + s.cell,
                                             largest.data())
                     : setup_row_cells<false>(row, n0, in.dt, a_p + s.cell, b + s.cell,
                                              largest.data());
        flags.any |= f.any;
        flags.nan |= f.nan;
    }
    if (flags.any == lanes::none) {
        return {false, 0.0};
    }
    // `np.abs(b).max()`: NaN if any |b| is.
    double value = 0.0;
    for (const double lane : largest) {
        value = lane > value ? lane : value;
    }
    if (flags.nan != lanes::none) {
        value = std::numeric_limits<double>::quiet_NaN();
    }
    return {true, in.tol * value};
}

acceptance::acceptance(double dt, std::span<const double> dt_cells, std::span<const bool> solid,
                       std::optional<double> bound, std::span<double> dt_masked,
                       std::span<double> terms)
    : scalar_(dt_cells.empty()), dt_(dt), dt_masked_(dt_masked), terms_(terms), bound_(bound) {
    if (!scalar_) {
        // A per-cell step need not be finite in a SOLID cell, where the
        // residual is zero, and inf times zero would put NaN in the sum.
        for (std::size_t i = 0; i < dt_masked.size(); ++i) {
            dt_masked[i] = solid[i] ? 0.0 : dt_cells[i];
        }
    }
}

double acceptance::form(std::span<const double> residual, std::uint64_t residual_id) {
    double leftover = 0.0;
    if (scalar_) {
        leftover = dt_ * reduce::pairwise_sum(residual);
    } else {
        for (std::size_t i = 0; i < residual.size(); ++i) {
            terms_[i] = dt_masked_[i] * residual[i];
        }
        leftover = reduce::pairwise_sum(terms_);
    }
    formed_ = residual_id;
    leftover_ = leftover;
    ++sums_;
    return leftover;
}

bool acceptance::test(std::span<const double> residual, std::uint64_t residual_id) {
    if (!bound_.has_value()) {
        return true;
    }
    return std::fabs(form(residual, residual_id)) <= *bound_;
}

void acceptance::final(std::span<const double> residual, std::uint64_t residual_id) {
    if (formed_ != residual_id) {
        form(residual, residual_id);
    }
}

template <int D>
solve_outcome jacobi(const implicit_input<D>& in, double stop, acceptance& accept,
                     implicit_scratch<D>& scratch, std::span<double> out) {
    const auto& fields = *scratch.fields;
    const std::size_t n = fields.cell_count();
    const std::size_t n0 = fields.cells(0);
    double* r = scratch.residual.data();
    const std::span<const double> residual(r, n);
    const lanes::mask* free = set_masks(in, scratch);
    // The iterate is C* itself until the first update. The next iterate is
    // written to `out` and to the scratch in turn, so no sweep copies a field.
    const double* x = in.c_star.data();
    double* next = out.data();
    double* spare = scratch.next.data();
    std::int64_t sweeps = 0;
    std::uint64_t id = 0;
    bool converged = false;
    while (true) {
        // The residual of x, and x + r / a_P with the SOLID cells at zero,
        // used only if this residual does not stop the solve.
        lanes::mask outside = lanes::none;
        for (const auto& line : fields.lines(0)) {
            const auto row =
                operands<D>(stencil_of(fields, line), in.conductance, x, scratch, free);
            outside |= sweep_row(row, n0, stop, r + line.cell, next + line.cell);
        }
        ++id;
        if (outside == lanes::none && accept.test(residual, id)) {
            converged = true;
            break;
        }
        if (sweeps >= in.max_iter) {
            break;
        }
        x = next;
        std::swap(next, spare);
        ++sweeps;
    }
    accept.final(residual, id);
    if (x != out.data()) {
        std::copy(x, x + n, out.begin());
    }
    return {sweeps, converged, 0};
}

template <int D>
solve_outcome conjugate_gradient(const implicit_input<D>& in, double stop, acceptance& accept,
                                 implicit_scratch<D>& scratch) {
    const auto& fields = *scratch.fields;
    const std::size_t n = fields.cell_count();
    // A row of the cell field: a block of the walks and of the dot products.
    const std::size_t n0 = fields.cells(0);
    const auto rows = fields.lines(0);
    const double* a_p = scratch.a_p.data();
    double* c = scratch.c.data();
    double* r = scratch.residual.data();
    double* inverse = scratch.inverse.data();
    double* z = scratch.z.data();
    double* p = scratch.p.data();
    double* q = scratch.q.data();
    const std::span<const double> r_span(r, n);
    const lanes::mask* free = set_masks(in, scratch);
    const lanes::mask* fluid = scratch.fluid.data();
    const auto row_of = [n0](const double* field, const layout::line& line) {
        return std::span<const double>(field + line.cell, n0);
    };
    for (const auto& line : rows) {
        const std::size_t at = line.cell;
        lanes::for_each_block(0, n0, [&](auto count, std::size_t m) {
            start_cells(count, m, in.c_star.data() + at, a_p + at, free + at, fluid + at, c + at,
                        inverse + at);
        });
    }
    std::uint64_t id = 0;
    // r = b + sum_f G_f C_N - a_P C, zero at the fixed cells; whether it meets the stop.
    const auto true_residual = [&]() {
        lanes::mask outside = lanes::none;
        for (const auto& line : rows) {
            const auto row =
                operands<D>(stencil_of(fields, line), in.conductance, c, scratch, free);
            outside |= residual_row(row, n0, stop, r + line.cell);
        }
        ++id;
        return outside == lanes::none;
    };
    // q = a_P p - sum_f G_f p_N, zero at the fixed cells, and p . q: each
    // row's dot product is taken as soon as the row's product is formed.
    const auto product_dot = [&]() {
        double total = 0.0;
        for (const auto& line : rows) {
            const auto row =
                operands<D>(stencil_of(fields, line), in.conductance, p, scratch, free);
            product_row(row, n0, q + line.cell);
            total += reduce::block_dot(row_of(p, line), row_of(q, line));
        }
        return total;
    };
    // z = inverse * r and p = z, and r . z: the start, and each restart.
    const auto restart = [&]() {
        double total = 0.0;
        for (const auto& line : rows) {
            const std::size_t at = line.cell;
            lanes::for_each_block(0, n0, [&](auto count, std::size_t m) {
                restart_cells(count, m, inverse + at, r + at, z + at, p + at);
            });
            total += reduce::block_dot(row_of(r, line), row_of(z, line));
        }
        return total;
    };

    if (true_residual() && accept.test(r_span, id)) {
        accept.final(r_span, id);
        return {0, true, 0};
    }
    double rz = restart();
    std::int64_t iterations = 0;
    std::int64_t restarts = 0;
    while (iterations < in.max_iter) {
        const double alpha = rz / product_dot();
        // c, r, the stop's test on r, and with them z and r . z for the next
        // direction, which only an iteration that goes on reads.
        lanes::mask outside = lanes::none;
        double rz_new = 0.0;
        for (const auto& line : rows) {
            const std::size_t at = line.cell;
            lanes::for_each_block(0, n0, [&](auto count, std::size_t m) {
                outside |= update_cells(count, m, alpha, stop, p + at, q + at, inverse + at,
                                        c + at, r + at, z + at);
            });
            rz_new += reduce::block_dot(row_of(r, line), row_of(z, line));
        }
        ++iterations;
        if (outside == lanes::none || iterations == in.max_iter) {
            // The recursion drifts from the true residual: confirm on it.
            if (true_residual() && accept.test(r_span, id)) {
                accept.final(r_span, id);
                return {iterations, true, restarts};
            }
            if (iterations == in.max_iter) {
                break;
            }
            rz = restart();
            ++restarts;
            continue;
        }
        const double beta = rz_new / rz;
        for (const auto& line : rows) {
            const std::size_t at = line.cell;
            lanes::for_each_block(0, n0, [&](auto count, std::size_t m) {
                direction_cells(count, m, beta, z + at, p + at);
            });
        }
        rz = rz_new;
    }
    // The loop leaves only by the cap's break, where r is c's true residual.
    accept.final(r_span, id);
    return {iterations, false, restarts};
}

template <int D>
implicit_status implicit_step(const implicit_input<D>& in, implicit_scratch<D>& scratch,
                              std::span<double> out, implicit_result& result) {
    const implicit_status status = check_leftover_bound(in.leftover_bound);
    if (status != implicit_status::ok) {
        return status;
    }
    result = implicit_result{};
    result.method = in.method;
    const implicit_setup setup = implicit_system(in, scratch);
    if (!setup.diffuses) {
        std::copy(in.c_star.begin(), in.c_star.end(), out.begin());
        return implicit_status::ok;
    }
    acceptance accept(in.dt, in.dt_cells, in.solid, in.leftover_bound, scratch.dt_masked,
                      scratch.terms);
    if (in.method == implicit_method::jacobi) {
        const solve_outcome o = jacobi(in, setup.stop, accept, scratch, out);
        result.iterations = o.iterations;
        result.converged = o.converged;
        result.leftover = accept.leftover();
        result.leftover_sums = accept.sums();
        return implicit_status::ok;
    }
    const solve_outcome cg = conjugate_gradient(in, setup.stop, accept, scratch);
    result.iterations = cg.iterations;
    result.restarts = cg.restarts;
    bool positive = true;
    for (std::size_t i = 0; positive && i < in.c_star.size(); ++i) {
        // `c[~fixed] > 0.0`: a NaN is not positive.
        positive = is_fixed(in.solid, in.held, i) || scratch.c[i] > 0.0;
    }
    if (cg.converged && positive) {
        std::copy(scratch.c.begin(), scratch.c.end(), out.begin());
        result.converged = true;
        result.leftover = accept.leftover();
        result.leftover_sums = accept.sums();
        return implicit_status::ok;
    }
    // Not from the CG iterate: a non-negative start is what makes every
    // Jacobi iterate non-negative.
    acceptance fresh(in.dt, in.dt_cells, in.solid, in.leftover_bound, scratch.dt_masked,
                     scratch.terms);
    const solve_outcome j = jacobi(in, setup.stop, fresh, scratch, out);
    result.converged = j.converged;
    result.fallbacks = 1;
    result.fallback_sweeps = j.iterations;
    result.leftover = fresh.leftover();
    result.leftover_sums = accept.sums() + fresh.sums();
    return implicit_status::ok;
}

template class scheme<1>;
template class scheme<2>;

template struct implicit_scratch<1>;
template struct implicit_scratch<2>;

template implicit_setup implicit_system<1>(const implicit_input<1>&, implicit_scratch<1>&);
template implicit_setup implicit_system<2>(const implicit_input<2>&, implicit_scratch<2>&);
template solve_outcome jacobi<1>(const implicit_input<1>&, double, acceptance&,
                                 implicit_scratch<1>&, std::span<double>);
template solve_outcome jacobi<2>(const implicit_input<2>&, double, acceptance&,
                                 implicit_scratch<2>&, std::span<double>);
template solve_outcome conjugate_gradient<1>(const implicit_input<1>&, double, acceptance&,
                                             implicit_scratch<1>&);
template solve_outcome conjugate_gradient<2>(const implicit_input<2>&, double, acceptance&,
                                             implicit_scratch<2>&);
template implicit_status implicit_step<1>(const implicit_input<1>&, implicit_scratch<1>&,
                                          std::span<double>, implicit_result&);
template implicit_status implicit_step<2>(const implicit_input<2>&, implicit_scratch<2>&,
                                          std::span<double>, implicit_result&);

template void quick_face_values<1>(const scheme<1>&, int, std::span<const double>,
                                   std::span<const double>, std::span<const double>,
                                   std::span<const bool>, std::span<double>);
template void quick_face_values<2>(const scheme<2>&, int, std::span<const double>,
                                   std::span<const double>, std::span<const double>,
                                   std::span<const bool>, std::span<double>);
template void advective_flux<1>(const scheme<1>&, int, std::span<const double>,
                                std::span<const double>, std::span<const double>, bool,
                                std::span<const bool>, std::span<double>);
template void advective_flux<2>(const scheme<2>&, int, std::span<const double>,
                                std::span<const double>, std::span<const double>, bool,
                                std::span<const bool>, std::span<double>);

}  // namespace cfdcore::scalar
