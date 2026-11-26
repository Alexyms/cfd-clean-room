#include "cfdcore/walls.hpp"

#include "cfdcore/elementary.hpp"
#include "cfdcore/positions.hpp"
#include "cfdcore/scalar.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace cfdcore::walls {

namespace {

using lanes::mask;
using positions::position;

void require(bool condition, const char* what) {
    if (!condition) {
        throw std::invalid_argument(what);
    }
}

bool positive_and_finite(double x) { return std::isfinite(x) && x > 0.0; }

// The wall function at one node. Not a block: the logarithm is the C library's,
// one call at a time, and the wall faces are few and scattered.
double wall_viscosity_at(double k, double y, double root_c_mu, double rho, double nu,
                         const wall_law& law) {
    const double u_k = root_c_mu * std::sqrt(k);
    const double y_star = u_k * y / nu;
    return rho * u_k * law.kappa * y /
           elementary::log(law.e * scalar::np_maximum(y_star, law.floor));
}

// The running sum of Python's `sum` over floats: Neumaier's compensated sum
// (CPython 3.12 and later).
class python_sum {
public:
    void add(double x) {
        if (!started_) {
            // The start is the integer 0, and 0 + x is x.
            sum_ = 0.0 + x;
            started_ = true;
            return;
        }
        const double total = sum_ + x;
        if (std::fabs(sum_) >= std::fabs(x)) {
            lost_ += (sum_ - total) + x;
        } else {
            lost_ += (x - total) + sum_;
        }
        sum_ = total;
    }
    double value() const {
        return lost_ != 0.0 && std::isfinite(lost_) ? sum_ + lost_ : sum_;
    }

private:
    double sum_ = 0.0;
    double lost_ = 0.0;
    bool started_ = false;
};

// The largest of N non-SOLID cells' values, one per cell of a block into
// `largest`. Returns the NaNs' masks, ORed.
template <std::size_t N>
inline mask largest_cells(lanes::cells<N>, const double* q, const mask* fluid,
                          double lowest, std::size_t first,
                          double* CFDCORE_RESTRICT largest) {
    mask nan = lanes::none;
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double value = lanes::select(fluid[i], q[i], lowest);
        nan |= lanes::mask_of(value != value);
        // Not `lanes::select`, which the Kernels rule names for a choice: GCC 13.3 turns
        // this ternary into a maximum and vectorises the block, and the select form
        // into control flow, which it does not (38 against 90 microseconds over
        // 320 x 120 cells, docs/reports/probe94/bench_max.cpp). Time it again after a
        // change of flags.
        largest[k] = value > largest[k] ? value : largest[k];
    }
    return nan;
}

// N entries of an edge's tangential velocity along a row: the condition's value
// where there is one, the adjacent face's velocity elsewhere.
template <std::size_t N>
inline void tangential_cells(lanes::cells<N>, const mask* known, const double* value,
                             const double* adjacent, std::size_t first,
                             double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        out[i] = lanes::select(known[i], value[i], adjacent[i]);
    }
}

// The number of entries of an array over every direction of a layout but one.
template <int D>
std::size_t edge_count(const layout::layout<D>& fields, int without) {
    return fields.cell_count() / fields.cells(without);
}

}  // namespace

floor_result log_law_floor(double kappa, double e) {
    // The reference's constants of the iteration: 100 steps, 4 units in the last
    // place of the root.
    if (!(elementary::log(e / kappa) > 1.0)) {
        return {floor_status::no_meeting, 0.0};
    }
    double y = 2.0 / kappa;
    for (int i = 0; i < 100; ++i) {
        const double step = (kappa * y - elementary::log(e * y)) / (kappa - 1.0 / y);
        y -= step;
        const double ulp = std::nextafter(y, std::numeric_limits<double>::infinity()) - y;
        if (std::fabs(step) <= 4.0 * ulp) {
            return {floor_status::ok, y};
        }
    }
    return {floor_status::not_settled, 0.0};
}

void wall_viscosity(std::span<const double> k_p, std::span<const double> y_p, double c_mu,
                    double rho, double nu, const wall_law& law, std::span<double> out) {
    require(k_p.size() == out.size() && y_p.size() == out.size(),
            "walls: the arrays of a wall viscosity differ in length");
    const double root = elementary::power(c_mu, 0.25);
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = wall_viscosity_at(k_p[i], y_p[i], root, rho, nu, law);
    }
}

carried_in inflow_values(double intensity, double normal_speed, double dissipation_length) {
    const double k = 1.5 * elementary::power(intensity * std::fabs(normal_speed), 2.0);
    return {k, elementary::power(k, 1.5) / dissipation_length};
}

// ---------------------------------------------------------------------------
// Set-up

template <int D>
    requires(D >= 1)
std::string_view boundary<D>::check(const grid::grid<D>& g, const boundary_input<D>& input) {
    const auto& fields = g.fields();
    const bool vented = !input.vents[0].empty();
    for (int d = 0; d < D; ++d) {
        const std::size_t edge = edge_count(fields, d);
        if (input.wall_cells_low[d].size() != edge || input.wall_cells_high[d].size() != edge) {
            return "an edge's wall cells are not one per cell of the edge";
        }
        if (input.vents[d].size() != (vented ? fields.face_count(d) : 0)) {
            return "the vent faces normal to a direction are not one per face";
        }
        const layout::layout<D> faces = positions::faces_layout(fields, d, d);
        for (int q = 0; q < D - 1; ++q) {
            const int t = momentum::transverse(d, q);
            const std::size_t along = edge_count(faces, t);
            for (const auto& corners :
                 {input.wall_corners_low[d][q], input.wall_corners_high[d][q]}) {
                if (corners.size() != along) {
                    return "an edge's wall corners are not one per face of the edge";
                }
            }
            for (const auto& condition : {input.low[d][q], input.high[d][q]}) {
                if (condition.is_dirichlet.size() != along || condition.value.size() != along) {
                    return "an edge's tangential condition is not one per face of the edge";
                }
            }
            const std::size_t corners = positions::faces_layout(fields, d, t).cell_count();
            if (input.vent_corners[d][q].size() != (vented ? corners : 0)) {
                return "a stencil's vent faces are not one per corner";
            }
        }
    }
    for (const auto& inlets : {input.edge_inlets, input.vent_inlets}) {
        for (const inlet_face& inlet : inlets) {
            if (inlet.direction < 0 || inlet.direction >= D ||
                inlet.face >= fields.face_count(inlet.direction)) {
                return "an inlet's face is not on the grid";
            }
            if (!(std::isfinite(inlet.intensity) && std::isfinite(inlet.normal_speed) &&
                  positive_and_finite(inlet.dissipation_length) &&
                  positive_and_finite(inlet.width))) {
                return "an inlet's number is not finite, or its length or width not positive";
            }
        }
    }
    for (const exhaust_face& exhaust : input.exhausts) {
        if (exhaust.direction < 0 || exhaust.direction >= D ||
            exhaust.face >= fields.face_count(exhaust.direction)) {
            return "an exhaust's face is not on the grid";
        }
        const std::size_t at =
            (exhaust.face / fields.face_stride(exhaust.direction)) %
            (fields.cells(exhaust.direction) + 1);
        if (at == 0 || at == fields.cells(exhaust.direction)) {
            return "an exhaust's face is on a domain edge";
        }
    }
    for (const double x : {input.c_mu, input.rho, input.nu, input.law.kappa, input.law.e,
                           input.law.floor}) {
        if (!positive_and_finite(x)) {
            return "a constant is not finite and positive";
        }
    }
    return {};
}

template <int D>
    requires(D >= 1)
boundary<D>::boundary(const grid::grid<D>& g, const boundary_input<D>& input)
    : grid_(&g),
      faces_{[&]<int... d>(std::integer_sequence<int, d...>) {
          return std::array<layout::layout<D>, D>{positions::faces_layout(g.fields(), d, d)...};
      }(std::make_integer_sequence<int, D>{})},
      rho_(input.rho),
      nu_(input.nu),
      law_(input.law),
      root_c_mu_(elementary::power(input.c_mu, 0.25)) {
    const std::string_view reason = check(g, input);
    if (!reason.empty()) {
        throw std::invalid_argument("walls::boundary: " + std::string(reason));
    }
    const auto& fields = g.fields();
    for (std::size_t i = 0; i < fields.cell_count(); ++i) {
        fluid_.push_back(lanes::mask_of(!g.solid(i)));
    }
    // The faces that carry air: no SOLID cell beside them, or a vent.
    for (int d = 0; d < D; ++d) {
        live_[d].assign(faces_[d].cell_count(), lanes::none);
        for (std::size_t f = 0; f < live_[d].size(); ++f) {
            const position<D> at = positions::position_of(faces_[d], f);
            const bool open =
                (at[d] == 0 ||
                 !g.solid(positions::index_of(fields, positions::below(at, d)))) &&
                (at[d] == g.cells(d) || !g.solid(positions::index_of(fields, at)));
            live_[d][f] = lanes::mask_of(open || (!input.vents[d].empty() && input.vents[d][f]));
        }
    }
    build_stencils(input);
    build_wall_cells(input);
    build_inflow(input);
    for (int c = 0; c < D; ++c) {
        for (int q = 0; q < D - 1; ++q) {
            const auto copy = [](const momentum::edge_condition& condition,
                                 std::vector<mask>& known, std::vector<double>& value,
                                 std::vector<double>& tangential) {
                for (const bool is_known : condition.is_dirichlet) {
                    known.push_back(lanes::mask_of(is_known));
                }
                value.assign(condition.value.begin(), condition.value.end());
                tangential.assign(condition.value.size(), 0.0);
            };
            copy(input.low[c][q], known_low_[c][q], value_low_[c][q], tangential_low_[c][q]);
            copy(input.high[c][q], known_high_[c][q], value_high_[c][q], tangential_high_[c][q]);
        }
    }
}

// The wall-function faces (`_velocity_walls`): an unknown's transverse face is a
// wall where it lies on a wall corner of a domain edge, or where the face beyond
// it bounds a SOLID cell; not over a vent's span.
template <int D>
    requires(D >= 1)
void boundary<D>::build_stencils(const boundary_input<D>& input) {
    const auto& fields = grid_->fields();
    stencils_.resize(D);
    for (int c = 0; c < D; ++c) {
        const layout::layout<D>& faces = faces_[c];
        const std::size_t nc = grid_->cells(c);
        const auto interior = [&](const position<D>& at) { return at[c] > 0 && at[c] < nc; };
        const auto bounds_solid = [&](const position<D>& at) {
            return interior(at) &&
                   (grid_->solid(positions::index_of(fields, positions::below(at, c))) ||
                    grid_->solid(positions::index_of(fields, at)));
        };
        stencils_[c].resize(D - 1);
        for (int q = 0; q < D - 1; ++q) {
            const int t = momentum::transverse(c, q);
            const std::size_t nt = grid_->cells(t);
            const auto nodes = grid_->nodes(t);
            const auto between = grid_->interior_faces(t);
            const auto edge_at = [&](std::size_t j) {
                return j == 0 ? nodes[0] : (j == nt ? nodes[nt + 1] : between[j - 1]);
            };
            const layout::layout<D> corners = positions::faces_layout(fields, c, t);
            stencil_walls& walls = stencils_[c][q];
            walls.mask.assign(corners.cell_count(), lanes::none);
            for (const bool high : {false, true}) {
                for (std::size_t f = 0; f < faces.cell_count(); ++f) {
                    const position<D> at = positions::position_of(faces, f);
                    if (!interior(at) || bounds_solid(at)) {
                        continue;
                    }
                    const std::size_t edge = positions::index_without(faces, at, t);
                    const bool on_edge = at[t] == (high ? nt - 1 : 0);
                    bool is_wall = false;
                    if (on_edge) {
                        is_wall = (high ? input.wall_corners_high : input.wall_corners_low)[c][q]
                                                                                          [edge];
                    } else {
                        is_wall = bounds_solid(high ? positions::above(at, t)
                                                    : positions::below(at, t));
                    }
                    const position<D> corner_at = high ? positions::above(at, t) : at;
                    const std::size_t corner = positions::index_of(corners, corner_at);
                    const auto& vents = input.vent_corners[c][q];
                    if (!is_wall || (!vents.empty() && vents[corner])) {
                        continue;
                    }
                    walls.mask[corner] = lanes::all;
                    walls.corner.push_back(corner);
                    walls.cell_low.push_back(
                        positions::index_of(fields, positions::below(at, c)));
                    walls.cell_high.push_back(positions::index_of(fields, at));
                    const double centre = nodes[at[t] + 1];
                    walls.distance.push_back(high ? edge_at(at[t] + 1) - centre
                                                  : centre - edge_at(at[t]));
                }
            }
        }
    }
}

// The wall cells, each with its walls in the reference's order: the last
// direction's low and high side first (south, north, west, east for D = 2).
template <int D>
    requires(D >= 1)
void boundary<D>::build_wall_cells(const boundary_input<D>& input) {
    const auto& fields = grid_->fields();
    const std::size_t n = fields.cell_count();
    wall_cells_ = std::make_unique<bool[]>(n);
    for (std::size_t i = 0; i < n; ++i) {
        wall_cells_[i] = false;
        if (grid_->solid(i)) {
            continue;
        }
        const position<D> at = positions::position_of(fields, i);
        wall_cell cell{i, walls_.size(), walls_.size()};
        for (int d = D - 1; d >= 0; --d) {
            const std::size_t nd = grid_->cells(d);
            const auto nodes = grid_->nodes(d);
            const auto between = grid_->interior_faces(d);
            const std::size_t edge = positions::index_without(fields, at, d);
            for (const bool high : {false, true}) {
                const bool on_edge = at[d] == (high ? nd - 1 : 0);
                const position<D> face_at = high ? positions::above(at, d) : at;
                bool is_wall = false;
                if (on_edge) {
                    is_wall = (high ? input.wall_cells_high : input.wall_cells_low)[d][edge];
                } else {
                    const std::size_t face = positions::index_of(faces_[d], face_at);
                    is_wall = grid_->solid(positions::index_of(
                                  fields, high ? positions::above(at, d)
                                               : positions::below(at, d))) &&
                              (input.vents[d].empty() || !input.vents[d][face]);
                }
                if (!is_wall) {
                    continue;
                }
                const double centre = nodes[at[d] + 1];
                const std::size_t j = face_at[d];
                const double face_x =
                    j == 0 ? nodes[0] : (j == nd ? nodes[nd + 1] : between[j - 1]);
                walls_.push_back(
                    {d, high ? face_x - centre : centre - face_x, wall_speed_.size()});
                for (int q = 0; q < D - 1; ++q) {
                    // The edge's own velocity at the cell, the mean of its two
                    // corners' along the component; an obstacle is at rest.
                    const int c = momentum::transverse(d, q);
                    double speed = 0.0;
                    if (on_edge) {
                        const auto& condition =
                            (high ? input.high : input.low)[c][momentum::slot_of(c, d)];
                        const std::size_t low =
                            positions::index_without(faces_[c], at, d);
                        const std::size_t next =
                            positions::index_without(faces_[c], positions::above(at, c), d);
                        speed = 0.5 * (condition.value[low] + condition.value[next]);
                    }
                    wall_speed_.push_back(speed);
                }
            }
        }
        cell.last = walls_.size();
        if (cell.last > cell.first) {
            wall_cells_[i] = true;
            cells_.push_back(cell);
        }
    }
    eps_wall_.assign(n, 0.0);
    production_.assign(n, 0.0);
}

// The inflow arrays: the inlets' values, which no call changes, and the list of
// the entries that take a cell's value in every call.
template <int D>
    requires(D >= 1)
void boundary<D>::build_inflow(const boundary_input<D>& input) {
    const auto& fields = grid_->fields();
    for (int d = 0; d < D; ++d) {
        inflow_k_[d].assign(faces_[d].cell_count(), 0.0);
        inflow_eps_[d].assign(faces_[d].cell_count(), 0.0);
    }
    std::array<std::vector<bool>, D> has_inlet;
    for (int d = 0; d < D; ++d) {
        has_inlet[d].assign(faces_[d].cell_count(), false);
    }
    for (const auto& inlets : {input.edge_inlets, input.vent_inlets}) {
        for (const inlet_face& inlet : inlets) {
            const carried_in value = inflow_values(inlet.intensity, inlet.normal_speed,
                                                   inlet.dissipation_length);
            inlets_.push_back(inlet);
            inlet_values_.push_back(value);
            inflow_k_[inlet.direction][inlet.face] = value.k;
            inflow_eps_[inlet.direction][inlet.face] = value.eps;
            has_inlet[inlet.direction][inlet.face] = true;
        }
    }
    for (int d = 0; d < D; ++d) {
        for (std::size_t f = 0; f < faces_[d].cell_count(); ++f) {
            const position<D> at = positions::position_of(faces_[d], f);
            const bool low = at[d] == 0;
            if ((!low && at[d] != grid_->cells(d)) || has_inlet[d][f]) {
                continue;
            }
            outflow_.push_back(
                {d, f, positions::index_of(fields, low ? at : positions::below(at, d))});
        }
    }
    for (const exhaust_face& exhaust : input.exhausts) {
        const position<D> at = positions::position_of(faces_[exhaust.direction], exhaust.face);
        // The room's cell: the one its flux leaves, opposite the obstacle.
        std::size_t cell = positions::index_of(fields, at);
        if (grid_->solid(cell)) {
            cell = positions::index_of(fields, positions::below(at, exhaust.direction));
        }
        outflow_.push_back({exhaust.direction, exhaust.face, cell});
    }
}

// ---------------------------------------------------------------------------
// The calls

template <int D>
    requires(D >= 1)
void boundary<D>::wall_viscosity(
    std::span<const double> k, const momentum::per_pair<std::span<const double>, D>& elsewhere,
    const momentum::per_pair<std::span<double>, D>& out) {
    require(k.size() == grid_->fields().cell_count(), "walls: k is not one value per cell");
    for (int c = 0; c < D; ++c) {
        for (int q = 0; q < D - 1; ++q) {
            stencil_walls& walls = stencils_[c][q];
            require(elsewhere[c][q].size() == walls.mask.size() &&
                        out[c][q].size() == walls.mask.size(),
                    "walls: a wall viscosity is not one value per corner");
            std::copy(elsewhere[c][q].begin(), elsewhere[c][q].end(), out[c][q].begin());
            // A loop of one face at a time: a wall face's cells and its corner lie
            // anywhere, and the logarithm is a call.
            for (std::size_t i = 0; i < walls.corner.size(); ++i) {
                const double k_p = 0.5 * (k[walls.cell_low[i]] + k[walls.cell_high[i]]);
                out[c][q][walls.corner[i]] =
                    wall_viscosity_at(k_p, walls.distance[i], root_c_mu_, rho_, nu_, law_);
            }
        }
    }
}

template <int D>
    requires(D >= 1)
double boundary<D>::live_velocity(int d, std::size_t face, const double* velocity) const {
    return lanes::keep(live_[d][face], velocity[face]);
}

// The entries of a quantity's inflow arrays that take a cell's value (`_inflow`):
// the cell's where it is positive, the largest of the non-SOLID cells' beside a
// SOLID edge cell, where nothing enters.
template <int D>
    requires(D >= 1)
void boundary<D>::inflow(std::span<const double> q, std::array<std::vector<double>, D>& out) {
    const double lowest = -std::numeric_limits<double>::infinity();
    std::array<double, lanes::width> largest;
    largest.fill(lowest);
    mask nan = lanes::none;
    lanes::for_each_block(0, q.size(), [&](auto cells, std::size_t m) {
        nan |= largest_cells(cells, q.data(), fluid_.data(), lowest, m, largest.data());
    });
    double stand_in = lowest;
    for (const double lane : largest) {
        stand_in = lane > stand_in ? lane : stand_in;
    }
    if (nan != lanes::none) {
        stand_in = std::numeric_limits<double>::quiet_NaN();
    }
    // A loop of one entry at a time, not a block: the entries are the few faces
    // along the domain's edge and its exhausts, each at a place of its own in
    // two arrays, so there is no row to walk.
    for (const outflow_entry& entry : outflow_) {
        const double cell = q[entry.cell];
        out[entry.direction][entry.face] =
            lanes::select(lanes::mask_of(cell > 0.0), cell, stand_in);
    }
}

template <int D>
    requires(D >= 1)
turbulence::conditions<D> boundary<D>::conditions(
    const std::array<std::span<const double>, D>& velocity, std::span<const double> k,
    std::span<const double> eps) {
    const auto& fields = grid_->fields();
    require(k.size() == fields.cell_count() && eps.size() == k.size(),
            "walls: k or eps is not one value per cell");
    for (int d = 0; d < D; ++d) {
        require(velocity[d].size() == faces_[d].cell_count(),
                "walls: a velocity component is not one value per face");
    }
    // The wall cells, one at a time: they are few and scattered, and the cube and
    // the logarithms are calls.
    for (const wall_cell& cell : cells_) {
        const double u_k = root_c_mu_ * std::sqrt(k[cell.cell]);
        const double cube = elementary::power(u_k, 3.0);
        double eps_sum = 0.0;
        double production_sum = 0.0;
        for (std::size_t w = cell.first; w < cell.last; ++w) {
            const wall& side = walls_[w];
            double slip = 0.0;
            for (int q = 0; q < D - 1; ++q) {
                const int c = momentum::transverse(side.direction, q);
                const std::size_t low = fields.low_face(c, cell.cell);
                const double* u = velocity[c].data();
                const double along =
                    0.5 * (live_velocity(c, low, u) +
                           live_velocity(c, low + fields.face_stride(c), u));
                const double relative =
                    along - wall_speed_[side.speed + static_cast<std::size_t>(q)];
                if constexpr (D <= 2) {
                    slip = std::fabs(relative);
                } else {
                    slip += relative * relative;
                }
            }
            if constexpr (D > 2) {
                slip = std::sqrt(slip);
            }
            const double y_star = u_k * side.distance / nu_;
            const double log_term =
                elementary::log(law_.e * scalar::np_maximum(y_star, law_.floor));
            eps_sum += cube / (law_.kappa * side.distance);
            production_sum += u_k * u_k * slip / (side.distance * log_term);
        }
        const auto count = static_cast<double>(cell.last - cell.first);
        eps_wall_[cell.cell] = eps_sum / count;
        production_[cell.cell] = production_sum / count;
    }
    turbulence::conditions<D> out;
    for (int c = 0; c < D; ++c) {
        for (int q = 0; q < D - 1; ++q) {
            const int t = momentum::transverse(c, q);
            const layout::layout<D>& faces = faces_[c];
            const auto edge = [&](bool high, const std::vector<mask>& known,
                                  const std::vector<double>& value,
                                  std::vector<double>& tangential) {
                // The component's faces nearest the edge: rows of them where the
                // direction across the edge crosses the rows, one face a row
                // where it lies along them.
                const std::size_t near = (high ? grid_->cells(t) - 1 : 0) * faces.cell_stride(t);
                const double* u = velocity[c].data();
                for (const auto& line : faces.lines(0)) {
                    const position<D> at = positions::position_of(faces, line.cell);
                    if (t != 0 && at[t] != 0) {
                        continue;
                    }
                    const std::size_t first = positions::index_without(faces, at, t);
                    const std::size_t count = t == 0 ? 1 : faces.cells(0);
                    lanes::for_each_block(0, count, [&](auto cells, std::size_t m) {
                        tangential_cells(cells, known.data() + first, value.data() + first,
                                         u + line.cell + near, m, tangential.data() + first);
                    });
                }
            };
            edge(false, known_low_[c][q], value_low_[c][q], tangential_low_[c][q]);
            edge(true, known_high_[c][q], value_high_[c][q], tangential_high_[c][q]);
            out.low[c][q] = tangential_low_[c][q];
            out.high[c][q] = tangential_high_[c][q];
        }
    }
    inflow(k, inflow_k_);
    inflow(eps, inflow_eps_);
    for (int d = 0; d < D; ++d) {
        out.inflow_k[d] = inflow_k_[d];
        out.inflow_eps[d] = inflow_eps_[d];
    }
    out.eps_held = wall_cells();
    out.eps_wall = eps_wall_;
    out.production_given = wall_cells();
    out.production = production_;
    return out;
}

template <int D>
    requires(D >= 1)
std::optional<carried_in> boundary<D>::initial_values() const {
    if (inlets_.empty()) {
        return std::nullopt;
    }
    python_sum total;
    python_sum k;
    python_sum eps;
    for (std::size_t i = 0; i < inlets_.size(); ++i) {
        const double inflow = std::fabs(inlets_[i].normal_speed) * inlets_[i].width;
        total.add(inflow);
        k.add(inlet_values_[i].k * inflow);
        eps.add(inlet_values_[i].eps * inflow);
    }
    return carried_in{k.value() / total.value(), eps.value() / total.value()};
}

template class boundary<1>;
template class boundary<2>;

}  // namespace cfdcore::walls
