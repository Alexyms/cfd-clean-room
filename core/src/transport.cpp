#include "cfdcore/transport.hpp"

#include "cfdcore/reduce.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace cfdcore::transport {

namespace {

// No cell on that side of a face: the domain's edge.
constexpr std::size_t no_cell = std::numeric_limits<std::size_t>::max();

// One row of the cell field (a line along direction 0) and where its cells'
// faces and neighbours along every other direction are: the row's first cell
// and its low face normal to direction 0; for each direction d >= 1 the stride,
// the offset from a cell to its low face normal to d, the row's position along
// d, and whether a row lies below it along d. Formed once per row, so no cell's
// arithmetic holds a division for an index.
template <int D>
struct row_place {
    std::size_t cell = 0;
    std::size_t face0 = 0;
    std::array<std::size_t, D> stride{};
    std::array<std::size_t, D> face_offset{};
    std::array<std::size_t, D> position{};
    std::array<bool, D> has_low{};
};

template <int D>
row_place<D> place_of(const layout::layout<D>& fields, const layout::line& row) {
    row_place<D> p;
    p.cell = row.cell;
    p.face0 = row.face;
    for (int d = 1; d < D; ++d) {
        const std::size_t stride = fields.cell_stride(d);
        p.stride[d] = stride;
        // layout::low_face, the same for every cell of the row.
        p.face_offset[d] = (row.cell / (stride * fields.cells(d))) * stride;
        p.position[d] = (row.cell / stride) % fields.cells(d);
        p.has_low[d] = p.position[d] > 0;
    }
    return p;
}

// The coordinates of an element of a field whose extent along direction e is
// extent[e], direction 0 fastest. Used at set-up only.
template <int D>
std::array<std::size_t, D> coordinates(std::size_t index,
                                       const std::array<std::size_t, D>& extent) {
    std::array<std::size_t, D> at{};
    for (int e = 0; e < D; ++e) {
        at[e] = index % extent[e];
        index /= extent[e];
    }
    return at;
}

// The extents of faces(d): the cell counts with one more along d.
template <int D>
std::array<std::size_t, D> face_extent(const layout::layout<D>& fields, int d) {
    std::array<std::size_t, D> extent{};
    for (int e = 0; e < D; ++e) {
        extent[e] = fields.cells(e) + (e == d ? 1 : 0);
    }
    return extent;
}

// The cell field index of the cell at the given coordinates.
template <int D>
std::size_t cell_index(const layout::layout<D>& fields, const std::array<std::size_t, D>& at) {
    std::size_t index = 0;
    for (int e = D - 1; e >= 0; --e) {
        index = index * fields.cells(e) + at[e];
    }
    return index;
}

void require_length(std::size_t got, std::size_t expected, const char* what) {
    if (got != expected) {
        throw std::invalid_argument(std::string("transport: ") + what +
                                    " does not have the length the grid needs");
    }
}

// Whether a double is finite: x - x is zero for a finite x and NaN otherwise.
CFDCORE_INLINE bool is_finite(double x) { return x - x == 0.0; }

// --- The step's kernels. Each is a block function (lanes.hpp) over N cells or
// faces from element `first`; its output is restrict-qualified and overlaps no
// input. ----------------------------------------------------------------------

// The field a step starts from (`np.array(C_k, copy=True)`, then
// `c[self._solid] = 0.0`), and whether a value is not finite
// (`np.isfinite(c).all()`, which reads the field before the SOLID cells are
// zeroed). Returns the OR of the cells' not-finite masks.
struct copy_row {
    const double* c;
    const lanes::mask* fluid;
};

template <std::size_t N>
inline lanes::mask copy_cells(lanes::cells<N>, const copy_row& in, std::size_t first,
                              double* CFDCORE_RESTRICT out) {
    lanes::mask bad = lanes::none;
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double value = in.c[i];
        bad |= lanes::mask_of(!is_finite(value));
        out[i] = lanes::keep(in.fluid[i], value);
    }
    return bad;
}

// The terms of a content sum at a run of non-SOLID cells: `C_k * volumes`.
template <std::size_t N>
inline void product_cells(lanes::cells<N>, const double* c, const double* volume,
                          std::size_t first, double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        out[i] = c[i] * volume[i];
    }
}

// The explicit update of a row: `c - dt * divergence / volume`, then with a
// source `+ dt * rate`, then the SOLID cells at zero. The divergence is the
// reference's `adv_u[:, 1:] - adv_u[:, :-1] + adv_v[1:, :] - adv_v[:-1, :]`,
// left to right: direction 0's high face less its low, then each further
// direction's high face added and its low face taken away.
//
// adv0:              direction 0's flux at the row's first low face, so that
//                    cell k's two faces are adv0[k] and adv0[k + 1]
// adv_low, adv_high: for each d >= 1, the flux on the row's low and high faces
//                    normal to d; entry 0 is not used
template <int D>
struct update_row {
    const double* c = nullptr;
    const double* adv0 = nullptr;
    const double* adv_low[D] = {};
    const double* adv_high[D] = {};
    const double* volume = nullptr;
    const double* dt_rate = nullptr;
    const lanes::mask* fluid = nullptr;
};

template <bool Source, std::size_t N, int D>
inline void update_cells(lanes::cells<N>, const update_row<D>& in, std::size_t first, double dt,
                         double* CFDCORE_RESTRICT c_star) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        double divergence = in.adv0[i + 1] - in.adv0[i];
        for (int d = 1; d < D; ++d) {
            divergence = divergence + in.adv_high[d][i];
            divergence = divergence - in.adv_low[d][i];
        }
        double value = in.c[i] - dt * divergence / in.volume[i];
        if constexpr (Source) {
            value = value + in.dt_rate[i];
        }
        c_star[i] = lanes::keep(in.fluid[i], value);
    }
}

template <bool Source, int D>
void update_row_cells(const update_row<D> row, std::size_t n, double dt, double* c_star) {
    lanes::for_each_block(0, n, [&](auto cells, std::size_t m) {
        update_cells<Source>(cells, row, m, dt, c_star);
    });
}

// The deposition through a run of faces each between a cell of `low` and the
// cell of `high` beside it: `deposit * dt * (C_low + C_high)`
// (`_book_deposition`'s padded sums), `deposit` holding the deposition per face
// times dt.
struct face_pair_row {
    const double* deposit;
    const double* low;
    const double* high;
};

template <std::size_t N>
inline void deposition_cells(lanes::cells<N>, const face_pair_row& in, std::size_t first,
                             double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        out[i] = in.deposit[i] * (in.low[i] + in.high[i]);
    }
}

// `np.where(mask, a, 0.0)` over a run of faces.
template <std::size_t N>
inline void where_cells(lanes::cells<N>, const lanes::mask* m, const double* a,
                        std::size_t first, double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        out[i] = lanes::select(m[i], a[i], 0.0);
    }
}

// --- The interval's kernels: once per interval, so written as plain passes
// over the field (each element's operations still the reference's, in order).

// The distance-weighted harmonic mean of two cell values at the face between
// them (`_harmonic_mean`), evaluated as `(d_low + d_high) low high / (d_low high
// + d_high low)` and zero where the denominator is not positive.
CFDCORE_INLINE double harmonic_mean(double low, double high, double d_low, double d_high) {
    const double denominator = d_low * high + d_high * low;
    const lanes::mask positive = lanes::mask_of(denominator > 0.0);
    const double safe = lanes::select(positive, denominator, 1.0);
    return lanes::select(positive, (d_low + d_high) * low * high / safe, 0.0);
}

}  // namespace

// What an interval fixes for one class.
template <int D>
    requires(D >= 1)
struct solver<D>::per_class {
    // Per direction, one per face of faces(d): the inflow values (the
    // recirculating discharges' faces rewritten every step), the advecting
    // velocity, its volume flux, the deposition per face and per face times dt,
    // and the conductance.
    std::array<std::vector<double>, D> inflow;
    std::array<std::vector<double>, D> velocity;
    std::array<std::vector<double>, D> flux;
    std::array<std::vector<double>, D> deposit;
    std::array<std::vector<double>, D> deposit_dt;
    std::array<std::vector<double>, D> conductance;
    // Per direction, one per line of lines(d): the low face's inward and
    // outward flux, the high face's inward and outward.
    std::array<std::array<std::vector<lanes::mask>, 4>, D> domain;
    // Per direction, one per face of faces(d), with vents only: the vent faces
    // entering from the low side, from the high side, leaving to the high side
    // and to the low side.
    std::array<std::array<std::vector<lanes::mask>, 4>, D> vent;
    // Per direction, end (low, high) and slot (floor, ceiling, wall): the
    // positions along lines(d) of the domain faces booked to the slot.
    std::array<std::array<std::array<std::vector<std::size_t>, 3>, 2>, D> surface;
    std::vector<double> deposit_cell;
    double stable_dt = std::numeric_limits<double>::infinity();
    bool has_source = false;
    std::vector<double> rate;
    std::vector<double> dt_rate;
    double source_total = 0.0;
    // A recirculating discharge: per direction, |flux| at the intake's faces in
    // face-field order and the cell on each side of each (`no_cell` beyond the
    // domain), the discharge's faces; its fraction and its drawn flux.
    struct return_air {
        std::array<std::vector<double>, D> weight;
        std::array<std::vector<std::size_t>, D> low_cell;
        std::array<std::vector<std::size_t>, D> high_cell;
        std::array<std::vector<std::size_t>, D> discharge;
        double passed = 0.0;
        double drawn = 0.0;
    };
    std::vector<return_air> returns;
    std::vector<double> drawn;
};

template <int D>
    requires(D >= 1)
solver<D>::solver(const scalar::scheme<D>& scheme, const geometry<D>& spacing,
                  const settings& keys, std::vector<class_physics> physics)
    : scheme_(&scheme),
      grid_(&scheme.mesh()),
      keys_(keys),
      physics_(std::move(physics)),
      budgets_(physics_.size()),
      implicit_(scheme.mesh().fields()) {
    if (physics_.empty()) {
        throw std::invalid_argument("transport: a solver needs at least one class");
    }
    const auto& fields = grid_->fields();
    const std::size_t n = fields.cell_count();
    for (int d = 0; d < D; ++d) {
        require_length(spacing.widths[d].size(), fields.cells(d), "a widths array");
        require_length(spacing.face_distances[d].size(), fields.cells(d) + 1,
                       "a face_distances array");
        widths_[d].assign(spacing.widths[d].begin(), spacing.widths[d].end());
    }
    // The cell volumes, np.outer(dy_cell, dx_cell): the widths multiplied from
    // the highest direction down.
    std::array<std::size_t, D> cell_extent{};
    for (int e = 0; e < D; ++e) {
        cell_extent[e] = fields.cells(e);
    }
    volume_.assign(n, 0.0);
    fluid_.assign(n, lanes::all);
    solid_ = std::make_unique<bool[]>(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto at = coordinates<D>(i, cell_extent);
        double v = widths_[D - 1][at[D - 1]];
        for (int e = D - 2; e >= 0; --e) {
            v = v * widths_[e][at[e]];
        }
        volume_[i] = v;
        solid_[i] = grid_->solid(i);
        fluid_[i] = lanes::mask_of(!solid_[i]);
    }
    // Per direction, on faces(d): the face's area (the product of the other
    // directions' widths: dy_cell for an x face, dx_cell for a y face), whether
    // it carries a flux (`_live_u`) and whether it is interior between two
    // non-SOLID cells (`_inner_u`), which side of an interior face is SOLID
    // (`_solid_low_u`), and the conductance per unit diffusivity, A_f / d_f on
    // the interior faces (`_conductance_u`).
    for (int d = 0; d < D; ++d) {
        const auto extent = face_extent(fields, d);
        const std::size_t faces = fields.face_count(d);
        area_[d].assign(faces, 1.0);
        unit_conductance_[d].assign(faces, 0.0);
        live_[d].assign(faces, lanes::none);
        inner_[d].assign(faces, lanes::none);
        solid_low_[d].assign(faces, lanes::none);
        solid_high_[d].assign(faces, lanes::none);
        vent_flags_[d] = std::make_unique<bool[]>(faces);
        for (std::size_t f = 0; f < faces; ++f) {
            const auto at = coordinates<D>(f, extent);
            bool first = true;
            double area = 1.0;
            for (int e = D - 1; e >= 0; --e) {
                if (e != d) {
                    area = first ? widths_[e][at[e]] : area * widths_[e][at[e]];
                    first = false;
                }
            }
            area_[d][f] = area;
            const std::size_t along = at[d];
            bool live = false;
            bool inner = false;
            if (along == 0) {
                live = !solid_[cell_index<D>(fields, at)];
            } else if (along == fields.cells(d)) {
                auto low = at;
                low[d] = along - 1;
                live = !solid_[cell_index<D>(fields, low)];
            } else {
                auto low = at;
                low[d] = along - 1;
                const bool solid_low = solid_[cell_index<D>(fields, low)];
                const bool solid_high = solid_[cell_index<D>(fields, at)];
                live = !solid_low && !solid_high;
                inner = live;
                solid_low_[d][f] = lanes::mask_of(solid_low);
                solid_high_[d][f] = lanes::mask_of(solid_high);
            }
            live_[d][f] = lanes::mask_of(live);
            inner_[d][f] = lanes::mask_of(inner);
            unit_conductance_[d][f] =
                lanes::select(inner_[d][f], area / spacing.face_distances[d][along], 0.0);
        }
        // The distances from each interior face to the two cell centres it
        // joins (`_reach_x`: the face less the low centre, the high centre less
        // the face), from the grid's nodes and interior faces.
        const auto nodes = grid_->nodes(d);
        const auto inside = grid_->interior_faces(d);
        reach_low_[d].assign(inside.size(), 0.0);
        reach_high_[d].assign(inside.size(), 0.0);
        for (std::size_t f = 0; f < inside.size(); ++f) {
            reach_low_[d][f] = inside[f] - nodes[f + 1];
            reach_high_[d][f] = nodes[f + 2] - inside[f];
        }
    }
    // The runs of non-SOLID cells along each row, in the cell field's order: a
    // content's terms are theirs, one run after another, as the reference's
    // boolean index lays them out.
    for (const auto& line : fields.lines(0)) {
        std::size_t m = 0;
        while (m < fields.cells(0)) {
            if (solid_[line.cell + m]) {
                ++m;
                continue;
            }
            const std::size_t start = m;
            while (m < fields.cells(0) && !solid_[line.cell + m]) {
                ++m;
            }
            live_runs_.push_back({line.cell + start, m - start});
            live_count_ += m - start;
        }
    }
    c_.assign(n, 0.0);
    c_star_.assign(n, 0.0);
    std::size_t largest = n;
    for (int d = 0; d < D; ++d) {
        adv_[d].assign(fields.face_count(d), 0.0);
        largest = std::max(largest, fields.face_count(d));
    }
    terms_.assign(largest, 0.0);
    class_.resize(physics_.size());
}

template <int D>
    requires(D >= 1)
solver<D>::~solver() = default;

template <int D>
    requires(D >= 1)
void solver<D>::check_class(std::size_t k) const {
    if (k >= physics_.size()) {
        throw std::invalid_argument("transport: the class is out of range");
    }
}

template <int D>
    requires(D >= 1)
void solver<D>::check_lengths(const std::vector<conditions<D>>& each,
                              const interval_input<D>& input) const {
    const auto& fields = grid_->fields();
    const std::size_t n = fields.cell_count();
    const std::size_t count = physics_.size();
    if (each.size() != count) {
        throw std::invalid_argument("transport: the conditions are not one per class");
    }
    if (!input.drift.empty() && input.drift.size() != count) {
        throw std::invalid_argument("transport: the drifts are not one per class");
    }
    if (!input.sources.empty() && input.sources.size() != count) {
        throw std::invalid_argument("transport: the sources are not one per class");
    }
    for (const auto& c : each) {
        bool any = false;
        bool all = true;
        for (int d = 0; d < D; ++d) {
            require_length(c.inflow[d].size(), fields.face_count(d), "an inflow array");
            require_length(c.deposition[d].size(), fields.face_count(d), "a deposition array");
            require_length(c.surface[d].size(), fields.face_count(d), "a surface array");
            any = any || !c.vents[d].empty();
            all = all && !c.vents[d].empty();
            if (!c.vents[d].empty()) {
                require_length(c.vents[d].size(), fields.face_count(d), "a vents array");
            }
            for (const auto& r : c.recirculations) {
                require_length(r.intake[d].size(), fields.face_count(d), "an intake array");
                require_length(r.discharge[d].size(), fields.face_count(d), "a discharge array");
            }
        }
        if (any && !all) {
            throw std::invalid_argument("transport: vents are given in every direction or none");
        }
        require_length(c.settling.size(), fields.face_count(D - 1), "a settling array");
    }
    for (int d = 0; d < D; ++d) {
        require_length(input.faces[d].size(), fields.face_count(d), "a faces array");
    }
    for (const auto& drift : input.drift) {
        bool any = false;
        bool all = true;
        for (int d = 0; d < D; ++d) {
            any = any || !drift[d].empty();
            all = all && !drift[d].empty();
            if (!drift[d].empty()) {
                require_length(drift[d].size(), fields.face_count(d), "a drift array");
            }
        }
        if (any && !all) {
            throw std::invalid_argument("transport: a drift is given in every direction or none");
        }
    }
    for (const auto& s : input.sources) {
        if (!s.empty()) {
            require_length(s.size(), n, "a sources array");
        }
    }
    if (!input.eddy_viscosity.empty()) {
        require_length(input.eddy_viscosity.size(), n, "the eddy viscosity");
    }
}

template <int D>
    requires(D >= 1)
refusal solver<D>::check_values(const std::vector<conditions<D>>& each,
                                const interval_input<D>& input) const {
    const std::size_t n = grid_->fields().cell_count();
    const std::size_t count = physics_.size();
    // Each class's conditions (`_check_conditions`): the inflow and deposition
    // arrays finite and non-negative, in the contract's order; a recirculation
    // only with vent faces, passing a fraction in [0, 1].
    static constexpr const char* inflow_names[2] = {"inflow_u", "inflow_v"};
    static constexpr const char* deposition_names[2] = {"deposition_u", "deposition_v"};
    const auto values_ok = [](std::span<const double> a) {
        bool ok = true;
        for (const double x : a) {
            ok = ok && is_finite(x) && x >= 0.0;
        }
        return ok;
    };
    for (std::size_t k = 0; k < count; ++k) {
        const auto& c = each[k];
        const int ki = static_cast<int>(k);
        for (int d = 0; d < D; ++d) {
            if (!values_ok(c.inflow[d])) {
                return {status::condition_not_finite_or_negative, ki,
                        D == 2 ? inflow_names[d] : "inflow"};
            }
        }
        for (int d = 0; d < D; ++d) {
            if (!values_ok(c.deposition[d])) {
                return {status::condition_not_finite_or_negative, ki,
                        D == 2 ? deposition_names[d] : "deposition"};
            }
        }
        if (!c.recirculations.empty() && c.vents[0].empty()) {
            return {status::recirculation_without_vents, ki};
        }
        for (const auto& r : c.recirculations) {
            // Written as "not within" so that a NaN fraction is refused.
            if (!(0.0 <= r.passed && r.passed <= 1.0)) {
                return {status::passed_out_of_range, ki, nullptr, r.passed};
            }
        }
    }
    // Class 0's vents are the room's: every class names the same faces.
    if (!each[0].vents[0].empty()) {
        for (std::size_t k = 0; k < count; ++k) {
            bool same = !each[k].vents[0].empty();
            for (int d = 0; same && d < D; ++d) {
                same = std::equal(each[k].vents[d].begin(), each[k].vents[d].end(),
                                  each[0].vents[d].begin());
            }
            if (!same) {
                return {status::vents_differ, static_cast<int>(k)};
            }
        }
    }
    // Then what `solve_timestep` checks: the faces, each drift, each class's
    // sources and the eddy viscosity.
    for (int d = 0; d < D; ++d) {
        for (const double x : input.faces[d]) {
            if (!is_finite(x)) {
                return {status::faces_not_finite};
            }
        }
    }
    for (std::size_t k = 0; k < input.drift.size(); ++k) {
        for (int d = 0; d < D; ++d) {
            for (const double x : input.drift[k][d]) {
                if (!is_finite(x)) {
                    return {status::drift_not_finite, static_cast<int>(k)};
                }
            }
        }
    }
    for (std::size_t k = 0; k < input.sources.size(); ++k) {
        const auto rate = input.sources[k];
        if (rate.empty()) {
            continue;
        }
        bool finite = true;
        bool negative = false;
        bool in_solid = false;
        for (std::size_t i = 0; i < n; ++i) {
            finite = finite && is_finite(rate[i]);
            negative = negative || rate[i] < 0.0;
            in_solid = in_solid || (solid_[i] && rate[i] != 0.0);
        }
        if (!finite) {
            return {status::sources_not_finite, static_cast<int>(k)};
        }
        if (negative) {
            return {status::sources_negative, static_cast<int>(k)};
        }
        if (in_solid) {
            return {status::source_in_solid, static_cast<int>(k)};
        }
    }
    if (!input.eddy_viscosity.empty()) {
        if (!keys_.schmidt.has_value()) {
            return {status::eddy_viscosity_without_schmidt};
        }
        // Read in the non-SOLID cells only (`field[self._live]`).
        bool finite = true;
        bool negative = false;
        for (std::size_t i = 0; i < n; ++i) {
            const double x = input.eddy_viscosity[i];
            finite = finite && (solid_[i] || is_finite(x));
            negative = negative || (!solid_[i] && x < 0.0);
        }
        if (!finite) {
            return {status::eddy_viscosity_not_finite};
        }
        if (negative) {
            return {status::eddy_viscosity_negative};
        }
    }
    return {};
}

template <int D>
    requires(D >= 1)
refusal solver<D>::set_interval(const std::vector<conditions<D>>& each,
                                const interval_input<D>& input) {
    check_lengths(each, input);
    const refusal refused = check_values(each, input);
    if (!refused.ok()) {
        return refused;
    }
    // Nothing is refused. Every array an interval needs is sized here, so a
    // step allocates nothing.
    const auto& fields = grid_->fields();
    const std::size_t n = fields.cell_count();
    const std::size_t n0 = fields.cells(0);
    interval_ = true;
    stepped_ = false;
    vents_ = !each[0].vents[0].empty();
    // The faces that carry a flux: a vent face too (`live_u | self._vent_u`).
    for (int d = 0; d < D; ++d) {
        vent_live_[d] = live_[d];
        for (std::size_t f = 0; f < fields.face_count(d); ++f) {
            vent_flags_[d][f] = vents_ && each[0].vents[d][f];
            vent_live_[d][f] |= lanes::mask_of(vent_flags_[d][f]);
        }
    }
    // The face eddy viscosity (`_face_eddy_viscosity`) of nu_t with the SOLID
    // cells at zero (`_check_eddy_viscosity`), zero on every domain face.
    const bool turbulent = !input.eddy_viscosity.empty();
    nu_cells_.clear();
    for (int d = 0; d < D; ++d) {
        nu_faces_[d].clear();
    }
    if (turbulent) {
        nu_cells_.assign(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            nu_cells_[i] = lanes::keep(fluid_[i], input.eddy_viscosity[i]);
        }
        for (int d = 0; d < D; ++d) {
            nu_faces_[d].assign(fields.face_count(d), 0.0);
        }
        for (const auto& line : fields.lines(0)) {
            const auto p = place_of(fields, line);
            const double* nu = nu_cells_.data() + p.cell;
            // Direction 0: the row's interior faces m = 1 to n0 - 1.
            for (std::size_t m = 1; m < n0; ++m) {
                nu_faces_[0][p.face0 + m] =
                    harmonic_mean(nu[m - 1], nu[m], reach_low_[0][m - 1], reach_high_[0][m - 1]);
            }
            // Each further direction: the row's low faces, where a row lies below.
            for (int d = 1; d < D; ++d) {
                if (!p.has_low[d]) {
                    continue;
                }
                const std::size_t along = p.position[d] - 1;
                const double* below = nu - p.stride[d];
                double* out = nu_faces_[d].data() + p.cell + p.face_offset[d];
                for (std::size_t m = 0; m < n0; ++m) {
                    out[m] = harmonic_mean(below[m], nu[m], reach_low_[d][along],
                                           reach_high_[d][along]);
                }
            }
        }
    }
    const auto high_face = [&](int d, const layout::line& line) {
        return line.face + fields.cells(d) * fields.face_stride(d);
    };
    for (std::size_t k = 0; k < physics_.size(); ++k) {
        auto& q = class_[k];
        const auto& c = each[k];
        const double settling = physics_[k].settling;
        const double diffusion = physics_[k].diffusion;
        const bool drift = !input.drift.empty() && !input.drift[k][0].empty();
        for (int d = 0; d < D; ++d) {
            const std::size_t faces = fields.face_count(d);
            const bool vertical = d == D - 1;
            q.inflow[d].assign(c.inflow[d].begin(), c.inflow[d].end());
            q.velocity[d].assign(faces, 0.0);
            q.flux[d].assign(faces, 0.0);
            q.deposit[d].assign(faces, 0.0);
            q.deposit_dt[d].assign(faces, 0.0);
            q.conductance[d].assign(faces, 0.0);
            for (std::size_t f = 0; f < faces; ++f) {
                // `_advecting_velocities`: the air's where the face carries a
                // flux, less the class's settling velocity on the vertical
                // direction's marked faces, plus the drift on the interior faces
                // between two non-SOLID cells, in that order.
                double velocity = lanes::select(vent_live_[d][f], input.faces[d][f], 0.0);
                if (vertical) {
                    velocity = velocity -
                               lanes::select(lanes::mask_of(c.settling[f]), settling, 0.0);
                }
                if (drift) {
                    velocity = velocity + lanes::select(inner_[d][f], input.drift[k][d][f], 0.0);
                }
                q.velocity[d][f] = velocity;
                // `u_adv * self._area_u`, `deposition_u * self._area_u`, and the
                // conductance: `D_B G`, or `(D_B + nu_f / Sc_t) G`.
                q.flux[d][f] = velocity * area_[d][f];
                q.deposit[d][f] = c.deposition[d][f] * area_[d][f];
                q.conductance[d][f] = turbulent ? (diffusion + nu_faces_[d][f] / *keys_.schmidt) *
                                                      unit_conductance_[d][f]
                                                : diffusion * unit_conductance_[d][f];
            }
            // The domain faces' masks along lines(d): the low face's inward
            // (flux > 0) and outward (< 0), the high face's inward (< 0) and
            // outward (> 0).
            const auto lines = fields.lines(d);
            for (auto& m : q.domain[d]) {
                m.assign(lines.size(), lanes::none);
            }
            for (std::size_t m = 0; m < lines.size(); ++m) {
                const double low = q.flux[d][lines[m].face];
                const double high = q.flux[d][high_face(d, lines[m])];
                q.domain[d][0][m] = lanes::mask_of(low > 0.0);
                q.domain[d][1][m] = lanes::mask_of(low < 0.0);
                q.domain[d][2][m] = lanes::mask_of(high < 0.0);
                q.domain[d][3][m] = lanes::mask_of(high > 0.0);
            }
            // The vent faces by side and sign (`_vent_fluxes`).
            for (auto& m : q.vent[d]) {
                m.clear();
            }
            if (vents_) {
                for (auto& m : q.vent[d]) {
                    m.assign(faces, lanes::none);
                }
                for (std::size_t f = 0; f < faces; ++f) {
                    const lanes::mask vent = lanes::mask_of(vent_flags_[d][f]);
                    const lanes::mask positive = lanes::mask_of(q.flux[d][f] > 0.0);
                    const lanes::mask negative = lanes::mask_of(q.flux[d][f] < 0.0);
                    q.vent[d][0][f] = vent & solid_low_[d][f] & positive;
                    q.vent[d][1][f] = vent & solid_high_[d][f] & negative;
                    q.vent[d][2][f] = vent & solid_high_[d][f] & positive;
                    q.vent[d][3][f] = vent & solid_low_[d][f] & negative;
                }
            }
            // The domain faces booked to each slot, along lines(d)
            // (`flux_u[:, 0][conditions.surface_u[:, 0] == code]`).
            for (int end = 0; end < 2; ++end) {
                for (std::size_t s = 0; s < 3; ++s) {
                    auto& positions = q.surface[d][end][s];
                    positions.clear();
                    for (std::size_t m = 0; m < lines.size(); ++m) {
                        const std::size_t f = end == 0 ? lines[m].face : high_face(d, lines[m]);
                        if (c.surface[d][f] == keys_.surface_codes[s]) {
                            positions.push_back(m);
                        }
                    }
                }
            }
        }
        // The deposition per cell (`_deposit_cell`): the cell's faces summed low
        // then high along each direction in turn, zero in SOLID cells; and
        // stable_dt, cfl over the largest, across non-SOLID cells, of the sum
        // over directions of max(|u_low|, |u_high|) / width (`stable_dt`). A
        // SOLID cell's rate is kept as zero, which leaves the largest as it is:
        // every rate is at least zero.
        q.deposit_cell.assign(n, 0.0);
        double worst = 0.0;
        for (const auto& line : fields.lines(0)) {
            const auto p = place_of(fields, line);
            for (std::size_t m = 0; m < n0; ++m) {
                const std::size_t i = p.cell + m;
                const std::size_t w = p.face0 + m;
                double sum = q.deposit[0][w] + q.deposit[0][w + 1];
                double rate = scalar::np_maximum(std::fabs(q.velocity[0][w]),
                                                 std::fabs(q.velocity[0][w + 1])) /
                              widths_[0][m];
                for (int d = 1; d < D; ++d) {
                    const std::size_t low = i + p.face_offset[d];
                    const std::size_t high = low + p.stride[d];
                    sum = sum + q.deposit[d][low];
                    sum = sum + q.deposit[d][high];
                    rate = rate + scalar::np_maximum(std::fabs(q.velocity[d][low]),
                                                     std::fabs(q.velocity[d][high])) /
                                      widths_[d][p.position[d]];
                }
                q.deposit_cell[i] = lanes::keep(fluid_[i], sum);
                const double kept = lanes::keep(fluid_[i], rate);
                worst = kept > worst ? kept : worst;
            }
        }
        q.stable_dt = worst == 0.0 ? std::numeric_limits<double>::infinity() : keys_.cfl / worst;
        // The source: kept for set_step's dt * rate, with its total, `np.sum(rate
        // * self._volume)` over the whole field.
        q.has_source = !input.sources.empty() && !input.sources[k].empty();
        q.rate.clear();
        q.dt_rate.clear();
        q.source_total = 0.0;
        if (q.has_source) {
            q.rate.assign(input.sources[k].begin(), input.sources[k].end());
            q.dt_rate.assign(n, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                terms_[i] = q.rate[i] * volume_[i];
            }
            q.source_total = reduce::pairwise_sum(std::span<const double>(terms_.data(), n));
        }
        // Each recirculating discharge (`_recirculated`), with vents only.
        q.returns.clear();
        q.drawn.clear();
        if (vents_) {
            for (const auto& r : c.recirculations) {
                typename per_class::return_air air;
                air.passed = r.passed;
                for (int d = 0; d < D; ++d) {
                    const auto extent = face_extent(fields, d);
                    for (std::size_t f = 0; f < fields.face_count(d); ++f) {
                        if (r.intake[d][f]) {
                            air.weight[d].push_back(std::fabs(q.flux[d][f]));
                            auto at = coordinates<D>(f, extent);
                            const std::size_t along = at[d];
                            air.high_cell[d].push_back(along < fields.cells(d)
                                                           ? cell_index<D>(fields, at)
                                                           : no_cell);
                            at[d] = along > 0 ? along - 1 : 0;
                            air.low_cell[d].push_back(along > 0 ? cell_index<D>(fields, at)
                                                                : no_cell);
                        }
                        if (r.discharge[d][f]) {
                            air.discharge[d].push_back(f);
                        }
                    }
                }
                // `np.abs(flux_u[intake_u]).sum() + np.abs(flux_v[intake_v]).sum()`.
                double drawn = reduce::pairwise_sum(air.weight[0]);
                for (int d = 1; d < D; ++d) {
                    drawn = drawn + reduce::pairwise_sum(air.weight[d]);
                }
                air.drawn = drawn;
                q.drawn.push_back(drawn);
                q.returns.push_back(std::move(air));
            }
        }
    }
    return {};
}

template <int D>
    requires(D >= 1)
refusal solver<D>::set_step(double dt) {
    if (!interval_) {
        throw std::invalid_argument("transport: set_step before set_interval");
    }
    if (!is_finite(dt)) {
        return {status::dt_not_finite, -1, nullptr, dt};
    }
    if (!(dt > 0.0)) {
        return {status::dt_not_positive, -1, nullptr, dt};
    }
    dt_ = dt;
    stepped_ = true;
    // `self._deposit_u[k] * dt` and `dt * rate`.
    for (auto& q : class_) {
        for (int d = 0; d < D; ++d) {
            for (std::size_t f = 0; f < q.deposit[d].size(); ++f) {
                q.deposit_dt[d][f] = q.deposit[d][f] * dt;
            }
        }
        if (q.has_source) {
            for (std::size_t i = 0; i < q.rate.size(); ++i) {
                q.dt_rate[i] = dt * q.rate[i];
            }
        }
    }
    return {};
}

template <int D>
    requires(D >= 1)
double solver<D>::stable_dt(std::size_t k) const {
    check_class(k);
    if (!interval_) {
        throw std::invalid_argument("transport: stable_dt before set_interval");
    }
    return class_[k].stable_dt;
}

template <int D>
    requires(D >= 1)
double solver<D>::content(std::span<const double> c) {
    // `float(np.sum((C_k * volumes)[mesh.cell_type != SOLID]))`: the terms of
    // the non-SOLID cells in the cell field's order, then NumPy's pairwise sum.
    double* out = terms_.data();
    for (const auto& r : live_runs_) {
        lanes::for_each_block(0, r.length, [&](auto cells, std::size_t m) {
            product_cells(cells, c.data() + r.cell, volume_.data() + r.cell, m, out);
        });
        out += r.length;
    }
    return reduce::pairwise_sum(std::span<const double>(terms_.data(), live_count_));
}

template <int D>
    requires(D >= 1)
void solver<D>::recirculate(std::size_t k) {
    // `_recirculated`: each discharge carries the flux-weighted mean of the
    // cells its intake draws from, in the field the step starts from, times the
    // fraction it passes. A cell beyond the domain reads as the zero padding.
    auto& q = class_[k];
    for (const auto& air : q.returns) {
        double carried = 0.0;
        if (air.drawn > 0.0) {
            double total = 0.0;
            for (int d = 0; d < D; ++d) {
                const std::size_t count = air.weight[d].size();
                for (std::size_t m = 0; m < count; ++m) {
                    const std::size_t low = air.low_cell[d][m];
                    const std::size_t high = air.high_cell[d][m];
                    const double room =
                        (low == no_cell ? 0.0 : c_[low]) + (high == no_cell ? 0.0 : c_[high]);
                    terms_[m] = air.weight[d][m] * room;
                }
                const double sum =
                    reduce::pairwise_sum(std::span<const double>(terms_.data(), count));
                total = d == 0 ? sum : total + sum;
            }
            carried = total / air.drawn;
        }
        const double value = air.passed * carried;
        for (int d = 0; d < D; ++d) {
            for (const std::size_t f : air.discharge[d]) {
                q.inflow[d][f] = value;
            }
        }
    }
}

template <int D>
    requires(D >= 1)
void solver<D>::book_boundaries(std::size_t k, budget& b) {
    // The domain faces: a positive flux enters at the low face and leaves at
    // the high face. `np.where` over a column or a row of faces, each summed in
    // NumPy's order, combined left to right as the reference writes them.
    const auto& fields = grid_->fields();
    const auto& q = class_[k];
    double inflow = 0.0;
    double outflow = 0.0;
    for (int d = 0; d < D; ++d) {
        const auto lines = fields.lines(d);
        const std::size_t high = fields.cells(d) * fields.face_stride(d);
        const auto sum_where = [&](const std::vector<lanes::mask>& m, std::size_t offset) {
            for (std::size_t i = 0; i < lines.size(); ++i) {
                terms_[i] = lanes::select(m[i], adv_[d][lines[i].face + offset], 0.0);
            }
            return reduce::pairwise_sum(std::span<const double>(terms_.data(), lines.size()));
        };
        const double low_in = sum_where(q.domain[d][0], 0);
        const double high_in = sum_where(q.domain[d][2], high);
        const double low_out = sum_where(q.domain[d][1], 0);
        const double high_out = sum_where(q.domain[d][3], high);
        if (d == 0) {
            inflow = low_in - high_in;
            outflow = -low_out + high_out;
        } else {
            inflow = inflow + low_in;
            inflow = inflow - high_in;
            outflow = outflow - low_out;
            outflow = outflow + high_out;
        }
    }
    // The vents (`_vent_fluxes`): `np.where` over the whole face field each.
    if (vents_) {
        double into = 0.0;
        double out = 0.0;
        for (int d = 0; d < D; ++d) {
            const std::size_t faces = fields.face_count(d);
            const auto sum_where = [&](const std::vector<lanes::mask>& m) {
                double* to = terms_.data();
                lanes::for_each_block(0, faces, [&](auto cells, std::size_t f) {
                    where_cells(cells, m.data(), adv_[d].data(), f, to);
                });
                return reduce::pairwise_sum(std::span<const double>(terms_.data(), faces));
            };
            const double enter_low = sum_where(q.vent[d][0]);
            const double enter_high = sum_where(q.vent[d][1]);
            const double leave_high = sum_where(q.vent[d][2]);
            const double leave_low = sum_where(q.vent[d][3]);
            if (d == 0) {
                into = enter_low - enter_high;
                out = leave_high - leave_low;
            } else {
                into = into + enter_low;
                into = into - enter_high;
                out = out + leave_high;
                out = out - leave_low;
            }
        }
        inflow = inflow + into;
        outflow = outflow + out;
    }
    b.inflow = b.inflow + inflow * dt_;
    b.outflow = b.outflow + outflow * dt_;
}

template <int D>
    requires(D >= 1)
void solver<D>::book_deposition(std::size_t k, std::span<const double> c, budget& b) {
    // `_book_deposition`: `v_d A_f dt (C_low + C_high)` on every face, the
    // missing neighbour of a domain face read as the zero padding.
    const auto& fields = grid_->fields();
    const auto& q = class_[k];
    const std::size_t n0 = fields.cells(0);
    // The interior faces normal to each direction, into one buffer in the face
    // field's order, summed as NumPy sums the slice: `flux_u[:, 1:-1]` is not
    // contiguous (its rows are summed through NumPy's buffer), the slice normal
    // to the last direction, `flux_v[1:-1, :]`, is.
    double obstacle = 0.0;
    for (int d = 0; d < D; ++d) {
        std::size_t at = 0;
        std::size_t row = 0;
        for (const auto& line : fields.lines(0)) {
            const auto p = place_of(fields, line);
            if (d == 0) {
                if (n0 > 1) {
                    const face_pair_row in{q.deposit_dt[0].data() + p.face0 + 1, c.data() + p.cell,
                                           c.data() + p.cell + 1};
                    double* out = terms_.data() + at;
                    lanes::for_each_block(0, n0 - 1, [&](auto cells, std::size_t m) {
                        deposition_cells(cells, in, m, out);
                    });
                    at += n0 - 1;
                }
                row = n0 - 1;
            } else if (p.has_low[d]) {
                const face_pair_row in{q.deposit_dt[d].data() + p.cell + p.face_offset[d],
                                       c.data() + p.cell - p.stride[d], c.data() + p.cell};
                double* out = terms_.data() + at;
                lanes::for_each_block(0, n0, [&](auto cells, std::size_t m) {
                    deposition_cells(cells, in, m, out);
                });
                at += n0;
            }
        }
        if (d > 0) {
            // A row of the slice: the cells below d along every direction, with
            // one fewer along d.
            row = fields.cells(d) - 1;
            for (int e = 0; e < d; ++e) {
                row *= fields.cells(e);
            }
        }
        const std::span<const double> slice(terms_.data(), at);
        const double sum =
            d == D - 1 ? reduce::pairwise_sum(slice) : reduce::buffered_sum(slice, row);
        obstacle = d == 0 ? sum : obstacle + sum;
    }
    b.deposited[obstacle_slot] = b.deposited[obstacle_slot] + obstacle;
    // The domain faces, by the surface their code names: per slot, the low and
    // the high faces along each direction in turn, each a boolean index of the
    // faces with that code.
    for (std::size_t s = 0; s < 3; ++s) {
        double total = 0.0;
        for (int d = 0; d < D; ++d) {
            const auto lines = fields.lines(d);
            const std::size_t high_face = fields.cells(d) * fields.face_stride(d);
            const std::size_t high_cell = (fields.cells(d) - 1) * fields.cell_stride(d);
            for (int end = 0; end < 2; ++end) {
                const auto& positions = q.surface[d][end][s];
                for (std::size_t m = 0; m < positions.size(); ++m) {
                    const auto& line = lines[positions[m]];
                    terms_[m] = end == 0 ? q.deposit_dt[d][line.face] * (0.0 + c[line.cell])
                                         : q.deposit_dt[d][line.face + high_face] *
                                               (c[line.cell + high_cell] + 0.0);
                }
                const double sum =
                    reduce::pairwise_sum(std::span<const double>(terms_.data(), positions.size()));
                total = (d == 0 && end == 0) ? sum : total + sum;
            }
        }
        b.deposited[s] = b.deposited[s] + total;
    }
}

template <int D>
    requires(D >= 1)
refusal solver<D>::step(std::size_t k, std::span<const double> c, std::span<double> out,
                        step_record& record) {
    check_class(k);
    if (!stepped_) {
        throw std::invalid_argument("transport: step before set_step");
    }
    const auto& fields = grid_->fields();
    const std::size_t n = fields.cell_count();
    const std::size_t n0 = fields.cells(0);
    require_length(c.size(), n, "a field");
    require_length(out.size(), n, "an output field");
    auto& q = class_[k];
    budget& b = budgets_[k];
    const int ki = static_cast<int>(k);

    // The field, refused if not finite, with the SOLID cells at zero; then the
    // step against the class's stable step. Nothing is booked before both.
    lanes::mask bad = lanes::none;
    for (const auto& line : fields.lines(0)) {
        const copy_row in{c.data() + line.cell, fluid_.data() + line.cell};
        double* to = c_.data() + line.cell;
        lanes::for_each_block(0, n0, [&](auto cells, std::size_t m) {
            bad |= copy_cells(cells, in, m, to);
        });
    }
    if (bad != lanes::none) {
        return {status::field_not_finite, ki};
    }
    // Written as "not dt <= limit", as the reference writes it.
    if (!(dt_ <= q.stable_dt)) {
        return {status::dt_above_stable, ki, nullptr, dt_, q.stable_dt};
    }
    if (!b.initial.has_value()) {
        b.initial = content(c_);
    }

    // The advective fluxes, with the recirculated inflow values first.
    recirculate(k);
    for (int d = 0; d < D; ++d) {
        const std::span<const bool> vents =
            vents_ ? std::span<const bool>(vent_flags_[d].get(), fields.face_count(d))
                   : std::span<const bool>();
        scalar::advective_flux(*scheme_, d, c_, q.flux[d], q.inflow[d], keys_.upwind, vents,
                               adv_[d]);
    }
    // The explicit update, the source and its booking.
    for (const auto& line : fields.lines(0)) {
        const auto p = place_of(fields, line);
        update_row<D> row;
        row.c = c_.data() + p.cell;
        row.adv0 = adv_[0].data() + p.face0;
        for (int d = 1; d < D; ++d) {
            row.adv_low[d] = adv_[d].data() + p.cell + p.face_offset[d];
            row.adv_high[d] = row.adv_low[d] + p.stride[d];
        }
        row.volume = volume_.data() + p.cell;
        row.fluid = fluid_.data() + p.cell;
        if (q.has_source) {
            row.dt_rate = q.dt_rate.data() + p.cell;
            update_row_cells<true>(row, n0, dt_, c_star_.data() + p.cell);
        } else {
            update_row_cells<false>(row, n0, dt_, c_star_.data() + p.cell);
        }
    }
    if (q.has_source) {
        b.source = b.source + q.source_total * dt_;
    }
    book_boundaries(k, b);

    // The step's share of REQ-T05 (ADR-016), left to right, as the reference
    // writes it: S_n with this step's inflow and source booked.
    const double supplied = *b.initial + b.inflow + b.source;
    const double bound = keys_.share * keys_.limit * supplied * dt_ / keys_.horizon;
    scalar::implicit_input<D> in{};
    in.c_star = c_star_;
    in.volume = volume_;
    in.dt = dt_;
    for (int d = 0; d < D; ++d) {
        in.conductance[d] = q.conductance[d];
    }
    in.sink = q.deposit_cell;
    in.solid = std::span<const bool>(solid_.get(), n);
    in.tol = keys_.tol;
    in.max_iter = keys_.max_sweeps;
    in.method = scalar::implicit_method::jacobi;
    in.leftover_bound = bound;
    scalar::implicit_result result;
    if (scalar::implicit_step(in, implicit_, out, result) != scalar::implicit_status::ok) {
        // The reference's implicit_step refuses a bound that is negative or NaN,
        // after the step has booked its source, inflow and outflow.
        return {status::bound_negative_or_nan, ki, nullptr, bound};
    }
    b.leftover = b.leftover + result.leftover;
    book_deposition(k, out, b);
    b.current = content(out);
    record.sweeps = result.iterations;
    record.converged = result.converged;
    record.leftover = result.leftover;
    record.bound = bound;
    record.accumulated = b.leftover;
    record.supplied = supplied;
    if (b.leftover != 0.0 && std::fabs(b.leftover) >= keys_.limit * supplied) {
        return {status::breach, ki};
    }
    return {};
}

template <int D>
    requires(D >= 1)
const budget& solver<D>::budget_of(std::size_t k) const {
    check_class(k);
    return budgets_[k];
}

template <int D>
    requires(D >= 1)
void solver<D>::set_budgets(const std::vector<budget>& budgets) {
    if (budgets.size() != physics_.size()) {
        throw std::invalid_argument("transport: the budgets are not one per class");
    }
    budgets_ = budgets;
}

template <int D>
    requires(D >= 1)
const typename solver<D>::per_class& solver<D>::interval_class(std::size_t k) const {
    check_class(k);
    if (!interval_) {
        throw std::invalid_argument("transport: no interval is set up");
    }
    return class_[k];
}

template <int D>
    requires(D >= 1)
std::span<const double> solver<D>::advecting_velocity(std::size_t k, int d) const {
    return interval_class(k).velocity.at(d);
}
template <int D>
    requires(D >= 1)
std::span<const double> solver<D>::volume_flux(std::size_t k, int d) const {
    return interval_class(k).flux.at(d);
}
template <int D>
    requires(D >= 1)
std::span<const double> solver<D>::conductance(std::size_t k, int d) const {
    return interval_class(k).conductance.at(d);
}
template <int D>
    requires(D >= 1)
std::span<const double> solver<D>::deposition_step(std::size_t k, int d) const {
    return interval_class(k).deposit_dt.at(d);
}
template <int D>
    requires(D >= 1)
std::span<const double> solver<D>::deposition_cell(std::size_t k) const {
    return interval_class(k).deposit_cell;
}
template <int D>
    requires(D >= 1)
std::span<const double> solver<D>::source_step(std::size_t k) const {
    return interval_class(k).dt_rate;
}
template <int D>
    requires(D >= 1)
double solver<D>::source_total(std::size_t k) const {
    return interval_class(k).source_total;
}
template <int D>
    requires(D >= 1)
std::span<const double> solver<D>::drawn(std::size_t k) const {
    return interval_class(k).drawn;
}
template <int D>
    requires(D >= 1)
std::span<const double> solver<D>::face_eddy_viscosity(int d) const {
    return nu_faces_.at(d);
}
template <int D>
    requires(D >= 1)
std::span<const double> solver<D>::unit_conductance(int d) const {
    return unit_conductance_.at(d);
}
template <int D>
    requires(D >= 1)
std::span<const lanes::mask> solver<D>::domain_mask(std::size_t k, int d, int which) const {
    return interval_class(k).domain.at(d).at(which);
}
template <int D>
    requires(D >= 1)
std::span<const lanes::mask> solver<D>::vent_mask(std::size_t k, int d, int which) const {
    return interval_class(k).vent.at(d).at(which);
}

template class solver<1>;
template class solver<2>;

}  // namespace cfdcore::transport
