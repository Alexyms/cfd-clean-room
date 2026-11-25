// The momentum predictor on the staggered grid (ADR-017 C, module `momentum`;
// ADR-012 B and D, ADR-014 J).
//
// A port of `src/momentum.py`'s `MomentumPredictor.predict` on the steady path
// and of its `stencil_viscosity`, written to equal them bit for bit on finite
// inputs (REQ-N03 level 1, ADR-017 G): every value is elementwise over the
// faces, formed by the reference's operations in the reference's order, one
// rounding each (the build's -ffp-contract=off). There is no sum over the
// field and no iteration count to decide. The transient term (ADR-014
// decision 3) is tier 2's and is not here (ADR-017 F).
//
// Written per direction (ADR-017 decision 4; Alex, 2026-11-23). Component c is
// the velocity along direction c, its streamwise direction s; the D - 1 others
// are its transverse directions t. Its stencil is one streamwise part and one
// part for each ordered pair (c, t), and every quantity of a transverse face
// belongs to its pair. A pair's slot q is t's place among the directions other
// than c, in ascending order: for D = 2 each component has the one slot 0.
//
// The reference writes one component's arithmetic with the component's own
// axis last and transposes every array for v. The core transposes nothing:
// every field is walked by its rows, the lines along direction 0, which are
// contiguous in memory, and a direction either lies along the rows (0) or
// crosses them (the others). Three layouts hold what the reference's frames
// do, each row-major with direction 0 the fastest (layout.hpp):
//
//   cells    n_d along every d; the streamwise faces of the momentum control
//            volumes lie at the cell centres. [ny, nx] for D = 2
//   corners  of a pair (c, t), n_d + 1 along c and along t, where the pair's
//            transverse faces lie; also `wall_mu`'s layout. [ny + 1, nx + 1]
//            for D = 2, for both pairs
//   faces    of component c, n_c + 1 along c: the unknowns. [ny, nx + 1] for u
//            and [ny + 1, nx] for v
//
// Instantiated for D = 2, which the reference fixes, and for D = 1, whose one
// component has no transverse direction, in the core's own tests. D = 3 is
// instantiated by core/tests/test_three_directions.cpp alone, on rooms extruded
// from D = 2, and what the module would still need for it is listed in
// docs/reports/phase6_momentum.md.

#pragma once

#include "cfdcore/grid.hpp"
#include "cfdcore/lanes.hpp"
#include "cfdcore/layout.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace cfdcore::momentum {

// A value for each ordered pair (c, t) of a component and a direction
// transverse to it, read as [c][q] with q the pair's slot.
template <typename Value, int D>
using per_pair = std::array<std::array<Value, D - 1>, D>;

// The direction of component c's slot q, and the slot of its direction t.
constexpr int transverse(int c, int q) { return q < c ? q : q + 1; }
constexpr int slot_of(int c, int t) { return t < c ? t : t - 1; }

// The tangential condition of one component along one domain edge transverse
// to it (`TangentialCondition`), one entry per face position of the component
// on the edge: its face layout without the transverse direction, the two ends
// along the component's own axis included. [n_c + 1] for D = 2.
//
// is_dirichlet:  whether the wall value is prescribed; a zero-gradient edge
//                (a pressure outlet) otherwise
// value:         the prescribed tangential velocity, m/s; 0 elsewhere
// wall_distance: from the edge to the component's first row of nodes, m
struct edge_condition {
    std::span<const bool> is_dirichlet;
    std::span<const double> value;
    double wall_distance = 0.0;
};

// What a predictor is built from besides its grid. Component 0 is u and 1 is
// v; an array is per direction (x, then y), per component or per pair.
//
// cell_widths: per direction, the n_d cell widths, m (`Mesh.dx_cell`, `dy_cell`)
// face_widths: per direction, the n_d + 1 distances across the faces, m: the
//              widths of the control volumes the faces' unknowns sit in
//              (`Mesh.dx_face`, `dy_face`)
// low, high:   per pair, the condition on the low and the high domain edge
//              along t: bottom and top for u, left and right for v
// held:        per component, the vent faces the sweeps keep (ADR-014 J;
//              `VentFaces.normal_u`, `normal_v`), in the component's face
//              layout; empty without vents
// rho, mu:     air's density, kg/m^3, and viscosity, Pa s
// alpha:       the velocity under-relaxation (`solver.alpha_velocity`)
// sweeps:      the Jacobi sweeps per call (`solver.momentum_sweeps`), at least 1
template <int D>
struct predictor_input {
    std::array<std::span<const double>, D> cell_widths;
    std::array<std::span<const double>, D> face_widths;
    per_pair<edge_condition, D> low;
    per_pair<edge_condition, D> high;
    std::array<std::span<const bool>, D> held;
    double rho = 0.0;
    double mu = 0.0;
    double alpha = 0.0;
    std::int64_t sweeps = 0;
};

// Why a viscosity is refused, in the reference's order: any value that is not
// finite before any that is not positive, over the whole array.
enum class viscosity_status { ok, not_finite, not_positive };

// What one pair's equations read that no call changes: computed once at
// construction, each equal bit for bit to what the reference forms in every
// call (or holds from its own construction). A row of an array is a line along
// direction 0 whatever the pair.
template <int D>
struct pair_setup {
    // The pair's corners, as a layout with one cell per corner, and the cell
    // count along t.
    layout::layout<D> corners;
    std::size_t nt = 0;

    // The cell widths along t, [nt], m.
    std::vector<double> width_t{};

    // QUICK's weights at the transverse faces, [nt + 1] with entries 1 to
    // nt - 1 used; `wall_*` the weights of the two interior nodes where the
    // far node is an obstacle's wall, whose value is zero (`_obstacle_transverse`).
    std::array<std::vector<double>, 3> t_pos{};
    std::array<std::vector<double>, 3> t_neg{};
    std::array<std::vector<double>, 2> wall_pos{};
    std::array<std::vector<double>, 2> wall_neg{};

    // The two fractions of the viscosity's harmonic mean across a transverse
    // face, [nt + 1], the interior entries used.
    std::vector<double> harmonic_low{};
    std::vector<double> harmonic_high{};

    // The distance a transverse face's conductance divides by: the nodes'
    // between two unknowns, the wall's at a domain edge or an obstacle face.
    // Corners, m.
    std::vector<double> reach{};

    // The scalar path's conductances: corners, Pa s.
    std::vector<double> conductance_t{};

    // Corners: an obstacle face of the wall stencil (`_ObstacleWalls.face`); a
    // face that takes `wall_mu` (an obstacle face or a Dirichlet edge's); a
    // face that conducts (all but a zero-gradient edge's); the far node of a
    // flow toward increasing t, or against it, is an obstacle's wall; a face
    // whose wall viscosity an unknown reads (`_wall_read`).
    std::vector<lanes::mask> wall_face{};
    std::vector<lanes::mask> takes_wall{};
    std::vector<lanes::mask> conducts{};
    std::vector<lanes::mask> far_wall_low{};
    std::vector<lanes::mask> far_wall_high{};
    std::vector<lanes::mask> wall_read{};

    // The component's faces: the neighbour on the high or the low side along t
    // is not an obstacle's (true also beside a domain edge, which the assembly
    // treats).
    std::vector<lanes::mask> couples_high{};
    std::vector<lanes::mask> couples_low{};

    // The edges' conditions, one entry per face position as `edge_condition`.
    std::vector<lanes::mask> low_known{};
    std::vector<lanes::mask> high_known{};
    std::vector<double> low_value{};
    std::vector<double> high_value{};

    // Whether a row of corners holds an obstacle's, so that a row without one
    // runs the block without them.
    std::vector<std::uint8_t> row_walls{};
};

// What one component's equations read that no call changes, beside its pairs'.
template <int D>
struct component_setup {
    // The component's faces, as a layout with one cell per face, and the cell
    // count along s.
    layout::layout<D> faces;
    std::size_t ns = 0;

    // The widths along s: cells [ns] and faces [ns + 1], m.
    std::vector<double> width_s{};
    std::vector<double> face_width{};

    // QUICK's weights at the streamwise faces, [ns] each: for a flow toward
    // increasing s over the far-upstream, upstream and downstream nodes, and
    // against it; `std_*` as `quick_face_values` takes them, with the boundary
    // form at the domain's two ends, and `alt_*` the boundary form of
    // `_obstacle_streamwise`, whose far node is the next one downstream.
    std::array<std::vector<double>, 3> std_pos{};
    std::array<std::vector<double>, 3> std_neg{};
    std::array<std::vector<double>, 3> alt_pos{};
    std::array<std::vector<double>, 3> alt_neg{};

    // The fraction of the viscosity's arithmetic mean along s, [ns + 1], the
    // interior entries used.
    std::vector<double> mean_high{};

    // The scalar path's conductances: cells, Pa s.
    std::vector<double> conductance_s{};

    // Cells: the far-upstream node of a flow toward increasing s, or against
    // it, lies beyond an obstacle's wall node.
    std::vector<lanes::mask> blocked_pos{};
    std::vector<lanes::mask> blocked_neg{};

    // The component's faces: not a domain-edge face and not a face of a SOLID
    // cell; a vent's face the sweeps keep (empty without vents).
    std::vector<lanes::mask> unknown{};
    std::vector<lanes::mask> held{};

    // Whether a row of cells holds a blocked one, as a pair's `row_walls`.
    std::vector<std::uint8_t> row_blocked{};

    // The component's pairs by slot, D - 1 of them.
    std::vector<pair_setup<D>> pairs{};
};

// The predictor on one grid, which must outlive it. Every array is allocated
// at construction; a call allocates nothing.
template <int D>
    requires(D >= 1)
class predictor {
public:
    // Why `input` cannot make a predictor on `g`, or an empty view when it
    // can: fewer than two cells along a direction, an array whose length is
    // not the grid's, a width, a wall distance, rho or mu that is not finite
    // and positive, alpha outside (0, 1] (the configuration's range), or fewer
    // than one sweep.
    static std::string_view check(const grid::grid<D>& g, const predictor_input<D>& input);

    // Builds the predictor from an input `check` accepts, copying every array.
    // Throws std::invalid_argument for one it does not.
    //
    // Every call below that takes an array throws std::invalid_argument for one
    // whose length is not the grid's and for a component or slot there is not.
    // The accessors `neighbour` and `setup` check nothing: an index there is not
    // is the caller's error.
    predictor(const grid::grid<D>& g, const predictor_input<D>& input);

    // The checks of `predict`'s viscosities. `mu_eff`: cells, Pa s, finite and
    // positive in every non-SOLID cell. `wall_mu`: that of the pair of
    // component c and slot q, in corners, finite and positive at every face
    // `wall_read` marks; what it holds elsewhere is not read.
    viscosity_status check_viscosity(std::span<const double> mu_eff) const;
    viscosity_status check_wall_viscosity(int c, int q, std::span<const double> wall_mu) const;

    // Every component's momentum equations assembled on the current field, with
    // the QUICK minus upwind flux as an explicit source, and `sweeps`
    // under-relaxed Jacobi sweeps of each (`MomentumPredictor.predict`,
    // transient None).
    //
    // velocity: per component, its faces, m/s, the domain-edge faces and the
    //           vents' faces already set
    // p:        pressure, cells, Pa
    // mu_eff:   dynamic effective viscosity, cells, Pa s, one
    //           `check_viscosity` accepts; empty for air's, the scalar path.
    //           SOLID cells are not read
    // wall_mu:  per pair, the wall stencil's viscosity in corners, one
    //           `check_wall_viscosity` accepts; empty for air's or the field's
    //           own
    // star:     output, per component the predicted velocity (u* and v*), its
    //           faces. Domain-edge faces keep the input; faces of SOLID cells
    //           are zero, except a vent's
    // diagonal: output, per component the un-relaxed diagonal, its faces,
    //           kg/(m s) per unit depth for D = 2; positive exactly at the
    //           faces the pressure correction may correct and zero elsewhere
    //
    // An output overlaps no input and no other output.
    void predict(const std::array<std::span<const double>, D>& velocity,
                 std::span<const double> p, std::span<const double> mu_eff,
                 const per_pair<std::span<const double>, D>& wall_mu,
                 const std::array<std::span<double>, D>& star,
                 const std::array<std::span<double>, D>& diagonal);

    // The viscosity the wall stencil takes when `wall_mu` is not given
    // (`MomentumPredictor.stencil_viscosity`).
    //
    // mu_eff: as `predict` takes it, not empty
    // out:    output per pair, corners, Pa s: the field's face rule at every
    //         face `wall_read` marks, air's viscosity on the rest. Overlaps no
    //         input
    void stencil_viscosity(std::span<const double> mu_eff,
                           const per_pair<std::span<double>, D>& out);

    // One component's equations as `predict` assembles them, without the
    // sweeps: its diagonal into `diagonal` and the parts below, which stand
    // until the next call of either.
    //
    // c: the component; the other arguments as `predict` takes them. The
    // output overlaps no input
    void equations(int c, const std::array<std::span<const double>, D>& velocity,
                   std::span<const double> p, std::span<const double> mu_eff,
                   const per_pair<std::span<const double>, D>& wall_mu,
                   std::span<double> diagonal);

    // The parts of the component last assembled, in its face layout: the
    // neighbour coefficients, zero at faces that are not unknowns, toward s+
    // and s- (which = 0 and 1) and toward t+ and t- of slot q (2 + 2q and
    // 3 + 2q); and the whole source with the relaxation term, read at unknowns
    // only.
    std::span<const double> neighbour(int which) const { return coefficient_[which]; }
    std::span<const double> source() const { return source_; }

    // What is fixed at construction for component c.
    const component_setup<D>& setup(int c) const { return setup_[c]; }

    const grid::grid<D>& mesh() const { return *grid_; }

private:
    // Where a row lies in a layout: its position along each direction that
    // crosses the rows. Entry 0 is 0.
    using position = std::array<std::size_t, D>;

    // One row of the fluxes through a row of cells or of corners: the mass
    // flux, the coefficients it gives the unknowns on its low and its high side
    // (toward s+ or t+, and toward s- or t-), the QUICK minus upwind flux and
    // the stress flux of form b.
    struct flux_row {
        std::vector<double> conductance;
        std::vector<double> flux;
        std::vector<double> plus;
        std::vector<double> minus;
        std::vector<double> deferred;
        std::vector<double> stress;
        std::vector<double> viscosity;
    };

    // The rows of fluxes on the low and the high side of a row of unknowns
    // along one direction: one row twice where the direction lies along it.
    struct flux_sides {
        const flux_row* low = nullptr;
        const flux_row* high = nullptr;
    };

    // The length checks of the arguments `predict` and `equations` share.
    void require_fields(const std::array<std::span<const double>, D>& velocity,
                        std::span<const double> p, std::span<const double> mu_eff,
                        const per_pair<std::span<const double>, D>& wall_mu) const;

    template <int C>
    void assemble(const std::array<std::span<const double>, D>& velocity,
                  std::span<const double> pressure, std::span<const double> mu_eff,
                  const per_pair<std::span<const double>, D>& wall_mu, double* diagonal);
    template <int C>
    void cell_row(const position& at, const double* own, const double* mu, flux_row& out);
    template <int C, int T>
    void corner_row(const position& at, const double* own, const double* other, const double* mu,
                    const double* wall, flux_row& out);
    template <int C, int T>
    const double* harmonic_row(const position& at, const double* mu, bool form);
    template <int C, int T>
    void corner_viscosity(const position& at, const double* mu, double* out);
    template <int C>
    void unknown_row(const position& at, const double* own, const double* p, const double* mu,
                     const flux_sides& cells, const std::array<flux_sides, D>& corners,
                     double* diagonal);
    template <int C, int T>
    void edge_rows(const position& at, const flux_sides& corners, double* diagonal);
    template <int C>
    void sweep(const double* own, const double* diagonal, double* star);

    // Which kept row holds the one at `at`, where the rows are kept until the
    // walk has passed the next along direction d, which crosses them.
    std::size_t kept_row(int d, const position& at) const;

    const grid::grid<D>* grid_;
    std::vector<component_setup<D>> setup_;
    // The cells of a row.
    std::size_t nx_;
    double mu_;
    double alpha_;
    double half_rho_;
    double relaxation_;
    std::int64_t sweeps_;

    // One component's equations and the sweeps' other iterate. The components
    // take turns in them, so a call touches a share of the memory.
    std::array<std::vector<double>, 2 * D> coefficient_;
    std::vector<double> source_;
    std::vector<double> relaxed_;
    std::vector<double> iterate_;
    // The kept rows (`kept_row`) of the streamwise fluxes and, by slot, of the
    // transverse ones and of the harmonic means a corner's viscosity averages.
    std::vector<flux_row> cell_rows_;
    std::array<std::vector<flux_row>, D> corner_rows_;
    std::array<std::vector<std::vector<double>>, D> means_;
    // Rows of edge values standing for a row of nodes.
    std::vector<double> edge_;
    std::vector<double> zeros_;
};

extern template class predictor<1>;
extern template class predictor<2>;

}  // namespace cfdcore::momentum
