#include "cfdcore/scalar.hpp"

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
// neighbouring row below and above. Formed once per row, so the per-cell loop
// does no division.
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

// `_neighbour_sum` at cell m of a row: sum_f G_f x_N, a missing neighbour read
// as zero, as the reference's zero padding reads it. The terms are added east,
// west (direction 0's high then low face), then each further direction's high
// then low face: north, south for D = 2.
template <int D, typename Conductances>
double neighbour_sum(const row_stencil<D>& s, std::size_t m,
                     const Conductances& g, const double* x) {
    const std::size_t i = s.cell + m;
    const double east = m + 1 < s.n0 ? x[i + 1] : 0.0;
    const double west = m > 0 ? x[i - 1] : 0.0;
    double sum = g[0][s.face0 + m + 1] * east + g[0][s.face0 + m] * west;
    for (int d = 1; d < D; ++d) {
        const std::size_t low = i + s.face_offset[d];
        const double high_value = s.has_high[d] ? x[i + s.stride[d]] : 0.0;
        const double low_value = s.has_low[d] ? x[i - s.stride[d]] : 0.0;
        sum = sum + g[d][low + s.stride[d]] * high_value;
        sum = sum + g[d][low] * low_value;
    }
    return sum;
}

// The diagonal's extra terms at cell m of a row: west, east (direction 0's
// low then high face), each further direction's low then high face (south,
// north for D = 2), then the sink. The transport gate's bits depend on this
// order.
template <int D, typename Conductances>
double diagonal_extra(const row_stencil<D>& s, std::size_t m,
                      const Conductances& g, const double* sink) {
    const std::size_t i = s.cell + m;
    double extra = g[0][s.face0 + m] + g[0][s.face0 + m + 1];
    for (int d = 1; d < D; ++d) {
        const std::size_t low = i + s.face_offset[d];
        extra = extra + g[d][low];
        extra = extra + g[d][low + s.stride[d]];
    }
    return extra + sink[i];
}

// The largest |x| seen so far, as np.abs(x).max() forms it: a NaN anywhere
// makes the maximum NaN, which no stop accepts.
class max_abs {
public:
    void add(double x) {
        const double a = std::fabs(x);
        nan_ = nan_ || std::isnan(a);
        largest_ = a > largest_ ? a : largest_;
    }
    // Whether the maximum is at most `stop`: false when it is NaN or `stop` is.
    bool within(double stop) const { return !nan_ && largest_ <= stop; }
    double value() const { return nan_ ? std::numeric_limits<double>::quiet_NaN() : largest_; }

private:
    double largest_ = 0.0;
    bool nan_ = false;
};

// Whether a cell is not an unknown of the CG solve: SOLID or held.
inline bool is_fixed(std::span<const bool> solid, std::span<const bool> held, std::size_t i) {
    return solid[i] || (!held.empty() && held[i]);
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
      terms(f.cell_count(), 0.0) {}

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
    double* a_p = scratch.a_p.data();
    bool any = false;
    for (const auto& row : fields.lines(0)) {
        const auto s = stencil_of(fields, row);
        for (std::size_t m = 0; m < s.n0; ++m) {
            const double extra = diagonal_extra(s, m, in.conductance, in.sink.data());
            a_p[s.cell + m] = extra;
            // `.any()`: a NaN is true, a -0.0 false.
            any = any || extra != 0.0;
        }
    }
    if (!any) {
        return {false, 0.0};
    }
    const bool scalar = in.dt_cells.empty();
    max_abs largest;
    for (std::size_t i = 0; i < fields.cell_count(); ++i) {
        const double over_dt = scalar ? in.volume[i] / in.dt : in.volume[i] / in.dt_cells[i];
        a_p[i] = over_dt + a_p[i];
        scratch.b[i] = over_dt * in.c_star[i];
        largest.add(scratch.b[i]);
    }
    return {true, in.tol * largest.value()};
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
    const double* a_p = scratch.a_p.data();
    const double* b = scratch.b.data();
    double* r = scratch.residual.data();
    // The iterate and the next one swap buffers each sweep.
    double* c = scratch.c.data();
    double* next = scratch.next.data();
    std::copy(in.c_star.begin(), in.c_star.end(), c);
    const std::span<const double> residual(r, n);
    const bool with_held = !in.held.empty();
    std::int64_t sweeps = 0;
    std::uint64_t id = 0;
    bool converged = false;
    while (true) {
        max_abs largest;
        for (const auto& row : fields.lines(0)) {
            const auto s = stencil_of(fields, row);
            for (std::size_t m = 0; m < s.n0; ++m) {
                const std::size_t i = s.cell + m;
                double ri = (b[i] + neighbour_sum(s, m, in.conductance, c)) - a_p[i] * c[i];
                if (in.solid[i] || (with_held && in.held[i])) {
                    ri = 0.0;
                }
                r[i] = ri;
                largest.add(ri);
                // c + r / a_P, then the SOLID cells set to zero; used only if
                // this sweep's residual does not stop the solve.
                next[i] = in.solid[i] ? 0.0 : c[i] + ri / a_p[i];
            }
        }
        ++id;
        if (largest.within(stop) && accept.test(residual, id)) {
            converged = true;
            break;
        }
        if (sweeps >= in.max_iter) {
            break;
        }
        std::swap(c, next);
        ++sweeps;
    }
    accept.final(residual, id);
    std::copy(c, c + n, out.begin());
    return {sweeps, converged, 0};
}

template <int D>
solve_outcome conjugate_gradient(const implicit_input<D>& in, double stop, acceptance& accept,
                                 implicit_scratch<D>& scratch) {
    const auto& fields = *scratch.fields;
    const std::size_t n = fields.cell_count();
    // The dot products' blocks: one row of the cell field each.
    const std::size_t block = fields.cells(0);
    const double* a_p = scratch.a_p.data();
    const double* b = scratch.b.data();
    double* c = scratch.c.data();
    double* r = scratch.residual.data();
    double* inverse = scratch.inverse.data();
    double* z = scratch.z.data();
    double* p = scratch.p.data();
    double* q = scratch.q.data();
    const std::span<const double> r_span(r, n);
    const std::span<const double> z_span(z, n);
    const std::span<const double> p_span(p, n);
    const std::span<const double> q_span(q, n);
    for (std::size_t i = 0; i < n; ++i) {
        c[i] = in.solid[i] ? 0.0 : in.c_star[i];
        inverse[i] = is_fixed(in.solid, in.held, i) ? 0.0 : 1.0 / a_p[i];
    }
    std::uint64_t id = 0;
    // r = b + sum_f G_f C_N - a_P C, zero at the fixed cells; whether it meets the stop.
    const auto true_residual = [&]() {
        max_abs largest;
        for (const auto& row : fields.lines(0)) {
            const auto s = stencil_of(fields, row);
            for (std::size_t m = 0; m < s.n0; ++m) {
                const std::size_t i = s.cell + m;
                double ri = (b[i] + neighbour_sum(s, m, in.conductance, c)) - a_p[i] * c[i];
                if (is_fixed(in.solid, in.held, i)) {
                    ri = 0.0;
                }
                r[i] = ri;
                largest.add(ri);
            }
        }
        ++id;
        return largest.within(stop);
    };
    // q = a_P p - sum_f G_f p_N, zero at the fixed cells.
    const auto product = [&]() {
        for (const auto& row : fields.lines(0)) {
            const auto s = stencil_of(fields, row);
            for (std::size_t m = 0; m < s.n0; ++m) {
                const std::size_t i = s.cell + m;
                const double qi = a_p[i] * p[i] - neighbour_sum(s, m, in.conductance, p);
                q[i] = is_fixed(in.solid, in.held, i) ? 0.0 : qi;
            }
        }
    };
    // z = inverse * r and p = z: the start, and each restart.
    const auto restart = [&]() {
        for (std::size_t i = 0; i < n; ++i) {
            z[i] = inverse[i] * r[i];
            p[i] = z[i];
        }
        return reduce::blocked_dot(r_span, z_span, block);
    };

    if (true_residual() && accept.test(r_span, id)) {
        accept.final(r_span, id);
        return {0, true, 0};
    }
    double rz = restart();
    std::int64_t iterations = 0;
    std::int64_t restarts = 0;
    while (iterations < in.max_iter) {
        product();
        const double alpha = rz / reduce::blocked_dot(p_span, q_span, block);
        max_abs largest;
        for (std::size_t i = 0; i < n; ++i) {
            c[i] = c[i] + alpha * p[i];
            r[i] = r[i] - alpha * q[i];
            largest.add(r[i]);
        }
        ++iterations;
        if (largest.within(stop) || iterations == in.max_iter) {
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
        for (std::size_t i = 0; i < n; ++i) {
            z[i] = inverse[i] * r[i];
        }
        const double rz_new = reduce::blocked_dot(r_span, z_span, block);
        const double beta = rz_new / rz;
        for (std::size_t i = 0; i < n; ++i) {
            p[i] = p[i] * beta;
            p[i] = p[i] + z[i];
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
