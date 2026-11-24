#include "cfdcore/pressure.hpp"

#include "cfdcore/reduce.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace cfdcore::pressure {

namespace {

using lanes::mask;

void require_length(std::size_t got, std::size_t expected, const char* what) {
    if (got != expected) {
        throw std::invalid_argument(what);
    }
}

// An operand of the faces normal to direction C that varies along every other
// direction, at element i of a row: one value for the row where C lies along
// the rows, per element otherwise.
template <int C>
CFDCORE_INLINE double beside(const double* p, std::size_t i) {
    return p[C == 0 ? 0 : i];
}

// The index of a row, or of its first element, in an array over every direction
// but d, from the index of the row's first cell.
template <int D>
std::size_t across_index(const layout::layout<D>& cells, int d, std::size_t cell) {
    const std::size_t stride = cells.cell_stride(d);
    return (cell / (stride * cells.cells(d))) * stride + cell % stride;
}

// Runs body<C>() for each direction in turn.
template <int D, typename Body>
CFDCORE_INLINE void for_each_direction(Body&& body) {
    [&]<int... C>(std::integer_sequence<int, C...>) {
        (body.template operator()<C>(), ...);
    }(std::make_integer_sequence<int, D>{});
}

// ---------------------------------------------------------------------------
// The p' equation

// d = A_face / a_P of N faces, zero where the face has no diagonal.
template <int C, std::size_t N>
inline void face_d_cells(lanes::cells<N>, const double* diagonal, const double* area,
                         std::size_t first, double* CFDCORE_RESTRICT d) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        d[i] = lanes::keep(lanes::mask_of(diagonal[i] > 0.0), beside<C>(area, i) / diagonal[i]);
    }
}

// The d of N faces of a domain edge the rows lie along: an outlet's, from the
// diagonal of the row of faces beside it, the face's own elsewhere.
template <std::size_t N>
inline void borrow_cells(lanes::cells<N>, const mask* outlet, const double* near,
                         const double* area, std::size_t first, double* CFDCORE_RESTRICT d) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        d[i] = lanes::select(outlet[i] & lanes::mask_of(near[i] > 0.0), area[i] / near[i], d[i]);
    }
}

// What one direction's couplings of a row of cells read: d on the cells' low
// and high faces (of direction 0, on face i and, at `low`, i + 1), the faces'
// area and the SOLID flags.
struct coupling_row {
    const double* low;
    const double* high;
    const double* area;
    const mask* solid;
};

// Direction C's couplings of N cells, rho d A, and their sum into the diagonal,
// the high face's before the low face's; zero in SOLID cells.
template <int C, std::size_t N>
inline void coupling_cells(lanes::cells<N>, const coupling_row& in, double rho, std::size_t first,
                           double* CFDCORE_RESTRICT high, double* CFDCORE_RESTRICT low,
                           double* CFDCORE_RESTRICT centre) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double area = beside<C>(in.area, i);
        const double d_high = C == 0 ? in.low[i + 1] : in.high[i];
        const double to_high = rho * d_high * area;
        const double to_low = rho * in.low[i] * area;
        const double sum = C == 0 ? to_high + to_low : (centre[i] + to_high) + to_low;
        high[i] = lanes::keep(~in.solid[i], to_high);
        low[i] = lanes::keep(~in.solid[i], to_low);
        centre[i] = lanes::keep(~in.solid[i], sum);
    }
}

template <int C>
void coupling_row_of(const coupling_row in, double rho, std::size_t n, double* high, double* low,
                     double* centre) {
    lanes::for_each_block(0, n, [&](auto cells, std::size_t m) {
        coupling_cells<C>(cells, in, rho, m, high, low, centre);
    });
}

// What the mass imbalance of a row of cells reads: for each direction the
// velocity on the low and the high faces (direction 0's high face is `low`'s
// next) and the faces' area; the SOLID flags; and the diagonal.
template <int D>
struct imbalance_row {
    const double* low[D] = {};
    const double* high[D] = {};
    const double* area[D] = {};
    const mask* solid = nullptr;
    const double* centre = nullptr;
};

// rho times the sum over the directions of (high - low) area, at element i.
template <int D>
CFDCORE_INLINE double imbalance_at(const imbalance_row<D>& in, double rho, std::size_t i) {
    double sum = (in.low[0][i + 1] - in.low[0][i]) * in.area[0][0];
    for (int d = 1; d < D; ++d) {
        sum = sum + (in.high[d][i] - in.low[d][i]) * in.area[d][i];
    }
    return rho * sum;
}

template <std::size_t N, int D>
inline void imbalance_cells(lanes::cells<N>, const imbalance_row<D>& in, double rho,
                            std::size_t first, double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        out[i] = lanes::keep(~in.solid[i], imbalance_at(in, rho, i));
    }
}

// The right-hand side of N cells, minus the imbalance, zero where the cell has
// no equation (a SOLID cell has none).
template <std::size_t N, int D>
inline void source_cells(lanes::cells<N>, const imbalance_row<D>& in, double rho,
                         std::size_t first, double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        out[i] = lanes::keep(lanes::mask_of(in.centre[i] > 0.0), -imbalance_at(in, rho, i));
    }
}

template <bool Source, int D>
void imbalance_row_of(const imbalance_row<D> in, double rho, std::size_t n, double* out) {
    lanes::for_each_block(0, n, [&](auto cells, std::size_t m) {
        if constexpr (Source) {
            source_cells(cells, in, rho, m, out);
        } else {
            imbalance_cells(cells, in, rho, m, out);
        }
    });
}

// x - shift of N cells that have an equation; the others keep their value.
template <std::size_t N>
inline void shift_cells(lanes::cells<N>, const double* centre, double shift, std::size_t first,
                        double* CFDCORE_RESTRICT x) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        x[i] = lanes::select(lanes::mask_of(centre[i] > 0.0), x[i] - shift, x[i]);
    }
}

void shift_row(const double* centre, double shift, std::size_t n, double* x) {
    lanes::for_each_block(
        0, n, [&](auto cells, std::size_t m) { shift_cells(cells, centre, shift, m, x); });
}

// ---------------------------------------------------------------------------
// The correction

// A row's corrected velocity over N faces: u* - d (p'_high - p'_low), the two
// cells' values each in an array of its own.
template <std::size_t N>
inline void velocity_cells(lanes::cells<N>, const double* star, const double* d,
                           const double* p_low, const double* p_high, std::size_t first,
                           double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        out[i] = star[i] - d[i] * (p_high[i] - p_low[i]);
    }
}

void velocity_row(const double* star, const double* d, const double* p_low, const double* p_high,
                  std::size_t n, double* out) {
    lanes::for_each_block(0, n, [&](auto cells, std::size_t m) {
        velocity_cells(cells, star, d, p_low, p_high, m, out);
    });
}

// The updated pressure of N cells: p + alpha p' where there is an equation,
// with Pin less the reference cell's; p elsewhere.
template <bool Pin, std::size_t N>
inline void pressure_cells(lanes::cells<N>, const double* centre, const double* p,
                           const double* p_prime, double alpha, double pinned, std::size_t first,
                           double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double relaxed = p[i] + alpha * p_prime[i];
        const double next = Pin ? relaxed - pinned : relaxed;
        out[i] = lanes::select(lanes::mask_of(centre[i] > 0.0), next, p[i]);
    }
}

// ---------------------------------------------------------------------------
// The conjugate gradients' elementwise steps

template <std::size_t N>
inline void inverse_cells(lanes::cells<N>, const double* centre, std::size_t first,
                          double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        out[i] = lanes::keep(lanes::mask_of(centre[i] > 0.0), 1.0 / centre[i]);
    }
}

template <std::size_t N>
inline void scale_cells(lanes::cells<N>, const double* inverse, const double* r,
                        std::size_t first, double* CFDCORE_RESTRICT z) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        z[i] = inverse[i] * r[i];
    }
}

// x <- x + alpha p and r <- r - alpha q.
template <std::size_t N>
inline void update_cells(lanes::cells<N>, double alpha, const double* p, const double* q,
                         std::size_t first, double* CFDCORE_RESTRICT x,
                         double* CFDCORE_RESTRICT r) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        x[i] = x[i] + alpha * p[i];
        r[i] = r[i] - alpha * q[i];
    }
}

// p <- p beta, then p <- p + z.
template <std::size_t N>
inline void direction_cells(lanes::cells<N>, double beta, const double* z, std::size_t first,
                            double* CFDCORE_RESTRICT p) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        p[i] = p[i] * beta + z[i];
    }
}

const char* keys_reason(const solve_keys& k) {
    if (!(k.rtol >= 0.0 && k.rtol < 1.0)) {
        return "pressure: rtol is in [0, 1)";
    }
    if (!(std::isfinite(k.floor) && k.floor >= 0.0)) {
        return "pressure: floor is finite and at least 0";
    }
    if (k.max_iter < 1) {
        return "pressure: max_iter is at least 1";
    }
    return nullptr;
}

bool finite_positive(double x) { return std::isfinite(x) && x > 0.0; }

}  // namespace

std::span<const std::size_t> cells_with_equation(std::span<const double> centre,
                                                 std::span<std::size_t> out) {
    require_length(out.size(), centre.size(), "pressure: the list of cells");
    std::size_t count = 0;
    for (std::size_t i = 0; i < centre.size(); ++i) {
        // Written each time and kept where it counts: no branch on the cell.
        out[count] = i;
        count += centre[i] > 0.0 ? 1 : 0;
    }
    return out.first(count);
}

double mean_over(std::span<const double> field, std::span<const std::size_t> cells) {
    const double sum = cells.size() == field.size() ? reduce::pairwise_sum(field)
                                                    : reduce::pairwise_sum_at(field, cells);
    return sum / static_cast<double>(cells.size());
}

// ---------------------------------------------------------------------------

template <int D>
    requires(D >= 1)
solver<D>::solver(const layout::layout<D>& fields, bool with_multigrid,
                  const multigrid::keys& keys)
    : fields_(&fields),
      inverse_(fields.cell_count(), 0.0),
      r_(fields.cell_count(), 0.0),
      z_(fields.cell_count(), 0.0),
      p_(fields.cell_count(), 0.0),
      q_(fields.cell_count(), 0.0),
      zeros_(fields.cells(0), 0.0) {
    if (with_multigrid) {
        hierarchy_ = std::make_unique<multigrid::hierarchy<D>>(fields, keys);
    }
}

template <int D>
    requires(D >= 1)
solve_status solver<D>::prepare(const multigrid::system<D>& s, const solve_keys& k) {
    for (int d = 0; d < D; ++d) {
        require_length(s.fields->cells(d), fields_->cells(d),
                       "pressure: the system is on another layout");
    }
    if (const char* reason = keys_reason(k)) {
        throw std::invalid_argument(reason);
    }
    centre_ = s.centre.data();
    method_ = k.method;
    project_ = k.project;
    const std::size_t n = fields_->cell_count();
    if (method_ == preconditioner::jacobi) {
        lanes::for_each_block(0, n, [&](auto cells, std::size_t m) {
            inverse_cells(cells, centre_, m, inverse_.data());
        });
        return solve_status::ok;
    }
    if (hierarchy_ == nullptr) {
        throw std::invalid_argument("pressure: the solver was built without multigrid");
    }
    switch (hierarchy_->build(s)) {
        case multigrid::build_status::not_dominant:
            return solve_status::not_dominant;
        case multigrid::build_status::not_converged:
            return solve_status::not_converged;
        case multigrid::build_status::ok:
            break;
    }
    return solve_status::ok;
}

// z = M^-1 r, and r . z by rows.
template <int D>
    requires(D >= 1)
double solver<D>::precondition_dot(std::span<const double> r, std::span<double> z) {
    const std::size_t n0 = fields_->cells(0);
    double mean = 0.0;
    if (method_ == preconditioner::multigrid) {
        hierarchy_->cycle(r, z);
        if (!project_.empty()) {
            mean = mean_over(z, project_);
        }
    }
    double total = 0.0;
    for (const auto& line : fields_->lines(0)) {
        const std::size_t at = line.cell;
        if (method_ == preconditioner::jacobi) {
            lanes::for_each_block(0, n0, [&](auto cells, std::size_t m) {
                scale_cells(cells, inverse_.data() + at, r.data() + at, m, z.data() + at);
            });
        } else if (!project_.empty()) {
            shift_row(centre_ + at, mean, n0, z.data() + at);
        }
        total += reduce::block_dot(r.subspan(at, n0), z.subspan(at, n0));
    }
    return total;
}

template <int D>
    requires(D >= 1)
void solver<D>::precondition(std::span<const double> r, std::span<double> z) {
    require_length(r.size(), fields_->cell_count(), "pressure: r");
    require_length(z.size(), fields_->cell_count(), "pressure: z");
    if (centre_ == nullptr) {
        throw std::invalid_argument("pressure: precondition before prepare");
    }
    precondition_dot(r, z);
}

// r = f - A x, and its squared norm by rows.
template <int D>
    requires(D >= 1)
double solver<D>::true_residual(const multigrid::system<D>& s, std::span<const double> f,
                                std::span<const double> x) {
    const std::size_t n0 = fields_->cells(0);
    double total = 0.0;
    for (const auto& line : fields_->lines(0)) {
        multigrid::residual_row<D>(s, line, zeros_, x, f, r_.data() + line.cell);
        const auto row = std::span<const double>(r_).subspan(line.cell, n0);
        total += reduce::block_dot(row, row);
    }
    return total;
}

template <int D>
    requires(D >= 1)
solve_result solver<D>::solve(const multigrid::system<D>& s, std::span<const double> f,
                              const solve_keys& k, std::span<double> x) {
    const std::size_t n = fields_->cell_count();
    const std::size_t n0 = fields_->cells(0);
    require_length(f.size(), n, "pressure: f");
    require_length(x.size(), n, "pressure: x");
    solve_result result;
    result.status = prepare(s, k);
    if (result.status != solve_status::ok) {
        return result;
    }
    const auto row_of = [n0](const std::vector<double>& field, const layout::line& line) {
        return std::span<const double>(field).subspan(line.cell, n0);
    };
    std::fill(x.begin(), x.end(), 0.0);
    const double f_norm = std::sqrt(reduce::blocked_dot(f, f, n0));
    const double level = k.rtol * f_norm;
    const double stop = k.floor > level ? k.floor : level;
    if (f_norm == 0.0) {
        return result;
    }
    std::copy(f.begin(), f.end(), r_.begin());
    // A start's or a restart's direction is z itself, which changes places with p.
    double rz = precondition_dot(r_, z_);
    std::swap(z_, p_);
    std::int64_t checks = 0;
    while (result.iterations < k.max_iter) {
        // q = A p and p . q, each row's dot product as soon as its product is formed.
        double pq = 0.0;
        for (const auto& line : fields_->lines(0)) {
            multigrid::product_row<D>(s, line, zeros_, p_, q_.data() + line.cell);
            pq += reduce::block_dot(row_of(p_, line), row_of(q_, line));
        }
        if (pq == 0.0) {
            result.status = solve_status::division_by_zero;
            return result;
        }
        const double alpha = rz / pq;
        double rr = 0.0;
        for (const auto& line : fields_->lines(0)) {
            const std::size_t at = line.cell;
            lanes::for_each_block(0, n0, [&](auto cells, std::size_t m) {
                update_cells(cells, alpha, p_.data() + at, q_.data() + at, m, x.data() + at,
                             r_.data() + at);
            });
            rr += reduce::block_dot(row_of(r_, line), row_of(r_, line));
        }
        ++result.iterations;
        if (std::sqrt(rr) <= stop) {
            // The recursion drifts from the true residual: confirm on it, and
            // restart from it where it misses the stop.
            const double norm = std::sqrt(true_residual(s, f, x));
            ++checks;
            if (norm <= stop) {
                result.residual_norm = norm;
                result.products = result.iterations + checks;
                return result;
            }
            rz = precondition_dot(r_, z_);
            std::swap(z_, p_);
            continue;
        }
        const double rz_new = precondition_dot(r_, z_);
        if (rz == 0.0) {
            result.status = solve_status::division_by_zero;
            return result;
        }
        const double beta = rz_new / rz;
        lanes::for_each_block(0, n, [&](auto cells, std::size_t m) {
            direction_cells(cells, beta, z_.data(), m, p_.data());
        });
        rz = rz_new;
    }
    result.reached_cap = true;
    result.residual_norm = std::sqrt(true_residual(s, f, x));
    result.products = result.iterations + checks + 1;
    return result;
}

// ---------------------------------------------------------------------------

template <int D>
    requires(D >= 1)
std::string_view corrector<D>::check(const grid::grid<D>& g, const corrector_input<D>& input) {
    for (int d = 0; d < D; ++d) {
        if (input.cell_widths[d].size() != g.cells(d)) {
            return "the cell widths are not one per cell along their direction";
        }
        for (const double width : input.cell_widths[d]) {
            if (!finite_positive(width)) {
                return "a cell width is not finite and positive";
            }
        }
        const std::size_t edge = g.fields().cell_count() / g.cells(d);
        if (input.outlet_low[d].size() != edge || input.outlet_high[d].size() != edge) {
            return "an edge's outlets are not one per face of the edge";
        }
    }
    if (!finite_positive(input.rho)) {
        return "rho is not finite and positive";
    }
    if (!finite_positive(input.alpha)) {
        return "alpha is not finite and positive";
    }
    return {};
}

template <int D>
    requires(D >= 1)
component_count corrector<D>::count_components(const grid::grid<D>& g,
                                               const corrector_input<D>& input) {
    const auto& fields = g.fields();
    const std::size_t n = fields.cell_count();
    // Each neighbour of a cell that is in the grid and not SOLID.
    const auto for_each_open = [&](std::size_t cell, auto&& body) {
        for (int d = 0; d < D; ++d) {
            const std::size_t stride = fields.cell_stride(d);
            const std::size_t at = (cell / stride) % fields.cells(d);
            if (at > 0 && !g.solid(cell - stride)) {
                body(cell - stride);
            }
            if (at + 1 < fields.cells(d) && !g.solid(cell + stride)) {
                body(cell + stride);
            }
        }
    };
    // Whether the cell's row holds an outlet face's coefficient: a face of a
    // domain edge whose d is borrowed from the face inward of the cell.
    const auto has_outlet = [&](std::size_t cell) {
        for (int d = 0; d < D; ++d) {
            const std::size_t stride = fields.cell_stride(d);
            const std::size_t at = (cell / stride) % fields.cells(d);
            const std::size_t edge = across_index(fields, d, cell);
            if (fields.cells(d) == 1) {
                continue;
            }
            if ((at == 0 && input.outlet_low[d][edge] && !g.solid(cell + stride)) ||
                (at + 1 == fields.cells(d) && input.outlet_high[d][edge] &&
                 !g.solid(cell - stride))) {
                return true;
            }
        }
        return false;
    };
    component_count count;
    std::vector<std::uint8_t> seen(n, 0);
    std::vector<std::size_t> grown;
    for (std::size_t start = 0; start < n; ++start) {
        bool open = false;
        if (!g.solid(start) && seen[start] == 0) {
            // A sealed single cell has no equation.
            for_each_open(start, [&](std::size_t) { open = true; });
        }
        if (!open) {
            continue;
        }
        ++count.components;
        bool reaches = false;
        seen[start] = 1;
        grown.assign(1, start);
        while (!grown.empty()) {
            const std::size_t cell = grown.back();
            grown.pop_back();
            reaches = reaches || has_outlet(cell);
            for_each_open(cell, [&](std::size_t next) {
                if (seen[next] == 0) {
                    seen[next] = 1;
                    grown.push_back(next);
                }
            });
        }
        count.stranded += reaches ? 0 : 1;
    }
    return count;
}

template <int D>
    requires(D >= 1)
corrector<D>::corrector(const grid::grid<D>& g, const corrector_input<D>& input)
    : grid_(&g),
      rho_(input.rho),
      alpha_(input.alpha),
      keys_{input.rtol, input.floor, input.max_iter, input.method, {}},
      system_(g.fields()),
      f_(g.fields().cell_count(), 0.0),
      cells_(g.fields().cell_count(), 0),
      zeros_(g.cells(0), 0.0),
      solver_(g.fields(), input.method == preconditioner::multigrid, input.levels) {
    const std::string_view reason = check(g, input);
    if (!reason.empty()) {
        throw std::invalid_argument(std::string(reason));
    }
    if (const char* bad = keys_reason(keys_)) {
        throw std::invalid_argument(bad);
    }
    const auto& cells = g.fields();
    needs_pin_ = true;
    for (int c = 0; c < D; ++c) {
        const std::size_t edge = cells.cell_count() / cells.cells(c);
        area_[c].assign(edge, 1.0);
        outlet_low_[c].resize(edge);
        outlet_high_[c].resize(edge);
        for (std::size_t i = 0; i < edge; ++i) {
            // The position of entry i along each of the other directions.
            std::size_t rest = i;
            for (int t = 0; t < D; ++t) {
                if (t != c) {
                    area_[c][i] = area_[c][i] * input.cell_widths[t][rest % cells.cells(t)];
                    rest /= cells.cells(t);
                }
            }
            outlet_low_[c][i] = lanes::mask_of(input.outlet_low[c][i]);
            outlet_high_[c][i] = lanes::mask_of(input.outlet_high[c][i]);
            if (input.outlet_low[c][i] || input.outlet_high[c][i]) {
                needs_pin_ = false;
            }
        }
        d_[c].assign(cells.face_count(c), 0.0);
        std::array<std::size_t, D> counts{};
        for (int d = 0; d < D; ++d) {
            counts[d] = cells.cells(d) + (d == c ? 1 : 0);
        }
        const layout::layout<D> faces(counts);
        const std::size_t stride = cells.cell_stride(c);
        for (const auto& line : faces.lines(0)) {
            const std::size_t block = line.cell / (stride * counts[c]);
            const std::size_t at = (line.cell / stride) % counts[c];
            face_row row;
            row.face = line.cell;
            row.across = across_index(faces, c, line.cell);
            row.has_low = c > 0 && at > 0;
            row.has_high = c > 0 && at + 1 < counts[c];
            row.high_cell = block * stride * cells.cells(c) + at * stride + line.cell % stride;
            row.low_cell = row.has_low ? row.high_cell - stride : 0;
            face_rows_[c].push_back(row);
        }
    }
    for (const auto& line : cells.lines(0)) {
        cell_row row;
        row.cell = line.cell;
        for (int d = 0; d < D; ++d) {
            row.face[d] = cells.low_face(d, line.cell);
            row.across[d] = across_index(cells, d, line.cell);
        }
        cell_rows_.push_back(row);
    }
    if (needs_pin_) {
        const auto types = g.cell_types();
        const auto first = std::find(types.begin(), types.end(), input.fluid);
        pin_ = first == types.end() ? 0 : static_cast<std::size_t>(first - types.begin());
    }
}

template <int D>
    requires(D >= 1)
void corrector<D>::require_faces(const std::array<std::span<const double>, D>& a,
                                 const char* what) const {
    for (int c = 0; c < D; ++c) {
        require_length(a[c].size(), d_[c].size(), what);
    }
}

template <int D>
    requires(D >= 1)
template <int C>
void corrector<D>::form_d(std::span<const double> diagonal) {
    const std::size_t n0 = grid_->cells(0);
    const std::size_t count = C == 0 ? n0 + 1 : n0;
    const std::size_t stride = grid_->fields().face_stride(C);
    for (const face_row& row : face_rows_[C]) {
        const double* a_p = diagonal.data() + row.face;
        const double* area = area_[C].data() + row.across;
        double* d = d_[C].data() + row.face;
        lanes::for_each_block(0, count, [&](auto cells, std::size_t m) {
            face_d_cells<C>(cells, a_p, area, m, d);
        });
        // An outlet face takes the diagonal of the nearest interior face.
        if constexpr (C == 0) {
            if (outlet_low_[0][row.across] != lanes::none && a_p[1] > 0.0) {
                d[0] = area[0] / a_p[1];
            }
            if (outlet_high_[0][row.across] != lanes::none && a_p[n0 - 1] > 0.0) {
                d[n0] = area[0] / a_p[n0 - 1];
            }
        } else {
            if (!row.has_low) {
                const mask* outlet = outlet_low_[C].data() + row.across;
                lanes::for_each_block(0, n0, [&](auto cells, std::size_t m) {
                    borrow_cells(cells, outlet, a_p + stride, area, m, d);
                });
            }
            if (!row.has_high) {
                const mask* outlet = outlet_high_[C].data() + row.across;
                lanes::for_each_block(0, n0, [&](auto cells, std::size_t m) {
                    borrow_cells(cells, outlet, a_p - stride, area, m, d);
                });
            }
        }
    }
}

template <int D>
    requires(D >= 1)
template <int C>
void corrector<D>::add_couplings(const cell_row& row) {
    const auto& cells = grid_->fields();
    const std::size_t n0 = cells.cells(0);
    const std::size_t stride = cells.cell_stride(C);
    const double* low = d_[C].data() + row.face[C];
    const coupling_row in{low, low + stride, area_[C].data() + row.across[C],
                          grid_->solid_flags().data() + row.cell};
    double* high_out = system_.high[C].data() + row.cell;
    double* low_out = system_.low[C].data() + row.cell;
    coupling_row_of<C>(in, rho_, n0, high_out, low_out, system_.centre.data() + row.cell);
    // A face of the domain's edge stays in the diagonal and has no neighbour:
    // p' = 0 at an outlet.
    if constexpr (C == 0) {
        low_out[0] = 0.0;
        high_out[n0 - 1] = 0.0;
    } else {
        const std::size_t at = (row.cell / stride) % cells.cells(C);
        if (at == 0) {
            std::fill(low_out, low_out + n0, 0.0);
        }
        if (at + 1 == cells.cells(C)) {
            std::fill(high_out, high_out + n0, 0.0);
        }
    }
}

template <int D>
    requires(D >= 1)
void corrector<D>::assemble(const std::array<std::span<const double>, D>& diagonal) {
    require_faces(diagonal, "pressure: a diagonal");
    for_each_direction<D>([&]<int C>() { form_d<C>(diagonal[C]); });
    for (const cell_row& row : cell_rows_) {
        for_each_direction<D>([&]<int C>() { add_couplings<C>(row); });
    }
    if (needs_pin_) {
        active_ = cells_with_equation(system_.centre, cells_);
        keys_.project = active_;
    }
}

template <int D>
    requires(D >= 1)
template <bool Source>
void corrector<D>::form_imbalance(const std::array<std::span<const double>, D>& velocity,
                                  double* out) const {
    const auto& cells = grid_->fields();
    for (const cell_row& row : cell_rows_) {
        imbalance_row<D> in;
        for (int d = 0; d < D; ++d) {
            in.low[d] = velocity[d].data() + row.face[d];
            in.high[d] = in.low[d] + cells.face_stride(d);
            in.area[d] = area_[d].data() + row.across[d];
        }
        in.solid = grid_->solid_flags().data() + row.cell;
        in.centre = system_.centre.data() + row.cell;
        imbalance_row_of<Source>(in, rho_, cells.cells(0), out + row.cell);
    }
}

template <int D>
    requires(D >= 1)
void corrector<D>::mass_imbalance(const std::array<std::span<const double>, D>& velocity,
                                  std::span<double> out) const {
    require_faces(velocity, "pressure: a velocity");
    require_length(out.size(), f_.size(), "pressure: the imbalance");
    form_imbalance<false>(velocity, out.data());
}

template <int D>
    requires(D >= 1)
void corrector<D>::right_hand_side(const std::array<std::span<const double>, D>& star,
                                   std::span<double> out) const {
    require_faces(star, "pressure: a velocity");
    require_length(out.size(), f_.size(), "pressure: the right-hand side");
    form_imbalance<true>(star, out.data());
    if (!active_.empty()) {
        const double mean = mean_over(out, active_);
        shift_row(system_.centre.data(), mean, out.size(), out.data());
    }
}

template <int D>
    requires(D >= 1)
template <int C>
void corrector<D>::correct_component(std::span<const double> star, const double* p_prime,
                                     double* out) const {
    const std::size_t n0 = grid_->cells(0);
    for (const face_row& row : face_rows_[C]) {
        const double* u = star.data() + row.face;
        const double* d = d_[C].data() + row.face;
        double* to = out + row.face;
        if constexpr (C == 0) {
            // Face i lies between cells i - 1 and i of the row: the faces
            // between two cells, then the two end ones, whose other cell is zero.
            const double* p = p_prime + row.high_cell;
            const double* zero = zeros_.data();
            velocity_row(u + 1, d + 1, p, p + 1, n0 - 1, to + 1);
            velocity_cells(lanes::cells<1>{}, u, d, zero, p, 0, to);
            velocity_cells(lanes::cells<1>{}, u + n0, d + n0, p + n0 - 1, zero, 0, to + n0);
        } else {
            velocity_row(u, d, row.has_low ? p_prime + row.low_cell : zeros_.data(),
                         row.has_high ? p_prime + row.high_cell : zeros_.data(), n0, to);
        }
    }
}

template <int D>
    requires(D >= 1)
void corrector<D>::apply(const std::array<std::span<const double>, D>& star,
                         std::span<const double> p, std::span<double> p_prime,
                         const std::array<std::span<double>, D>& velocity,
                         std::span<double> p_next) const {
    const std::size_t n = f_.size();
    require_faces(star, "pressure: a velocity");
    require_length(p.size(), n, "pressure: p");
    require_length(p_prime.size(), n, "pressure: p'");
    require_length(p_next.size(), n, "pressure: the updated p");
    for (int c = 0; c < D; ++c) {
        require_length(velocity[c].size(), d_[c].size(), "pressure: a corrected velocity");
    }
    const double* centre = system_.centre.data();
    if (needs_pin_) {
        shift_row(centre, p_prime[pin_], n, p_prime.data());
    }
    for_each_direction<D>([&]<int C>() {
        correct_component<C>(star[C], p_prime.data(), velocity[C].data());
    });
    if (needs_pin_) {
        // The reference cell's own update, which every cell with an equation loses.
        const double pinned =
            centre[pin_] > 0.0 ? p[pin_] + alpha_ * p_prime[pin_] : p[pin_];
        lanes::for_each_block(0, n, [&](auto cells, std::size_t m) {
            pressure_cells<true>(cells, centre, p.data(), p_prime.data(), alpha_, pinned, m,
                                 p_next.data());
        });
    } else {
        lanes::for_each_block(0, n, [&](auto cells, std::size_t m) {
            pressure_cells<false>(cells, centre, p.data(), p_prime.data(), alpha_, 0.0, m,
                                  p_next.data());
        });
    }
}

template <int D>
    requires(D >= 1)
solve_result corrector<D>::correct(const std::array<std::span<const double>, D>& star,
                                   const std::array<std::span<const double>, D>& diagonal,
                                   std::span<const double> p,
                                   const std::array<std::span<double>, D>& velocity,
                                   std::span<double> p_next, std::span<double> p_prime) {
    assemble(diagonal);
    right_hand_side(star, f_);
    const solve_result result = solver_.solve(system_, f_, keys_, p_prime);
    if (result.status == solve_status::ok) {
        apply(star, p, p_prime, velocity, p_next);
    }
    return result;
}

template class solver<1>;
template class solver<2>;
template class corrector<1>;
template class corrector<2>;

}  // namespace cfdcore::pressure
