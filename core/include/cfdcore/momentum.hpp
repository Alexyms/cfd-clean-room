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
// The reference writes one component's arithmetic with the component's own
// axis last and transposes every array for v. The core transposes nothing: a
// component's own axis is called s and the transverse one t, and for u (s = x,
// t = y) and for v (s = y, t = x) alike every field is walked by its rows, the
// lines along x, which are contiguous in memory. Three layouts hold what the
// reference's frames do:
//
//   cells    [ny, nx]          the streamwise faces of the momentum control
//                              volumes lie at the cell centres
//   corners  [ny + 1, nx + 1]  their transverse faces at the cell corners, for
//                              both components; also `wall_mu`'s layout
//   faces    [ny, nx + 1] (u), [ny + 1, nx] (v)  the unknowns
//
// The stencil has one transverse direction, as the reference's has, so the
// module is built for D = 2 only, unlike the cell-centred modules: a third
// direction would add terms whose order no reference fixes.

#pragma once

#include "cfdcore/grid.hpp"
#include "cfdcore/lanes.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace cfdcore::momentum {

// The tangential condition of one component along one domain edge
// (`TangentialCondition`), one entry per face position along the component's
// own axis, the two ends included.
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

// What a predictor is built from besides its grid. Component 0 is u and 1 is v;
// an array is per direction (x, then y) or per component as it says.
//
// cell_widths: per direction, the n_d cell widths, m (`Mesh.dx_cell`, `dy_cell`)
// face_widths: per direction, the n_d + 1 distances across the faces, m: the
//              widths of the control volumes the faces' unknowns sit in
//              (`Mesh.dx_face`, `dy_face`)
// low, high:   per component, the condition on the low and the high transverse
//              edge: bottom and top for u, left and right for v
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
    std::array<edge_condition, D> low;
    std::array<edge_condition, D> high;
    std::array<std::span<const bool>, D> held;
    double rho = 0.0;
    double mu = 0.0;
    double alpha = 0.0;
    std::int64_t sweeps = 0;
};

// Why a viscosity is refused, in the reference's order: any value that is not
// finite before any that is not positive, over the whole array.
enum class viscosity_status { ok, not_finite, not_positive };

// What one component's equations read that no call changes: computed once at
// construction, each equal bit for bit to what the reference forms in every
// call (or holds from its own construction). nt and ns are the cell counts
// along t and s. A row of an array is a line along x whatever the component.
struct component_setup {
    std::size_t nt = 0;
    std::size_t ns = 0;

    // The widths: cells along s [ns] and t [nt], faces along s [ns + 1], m.
    std::vector<double> width_s;
    std::vector<double> width_t;
    std::vector<double> face_width;

    // QUICK's weights at the streamwise faces, [ns] each: for a flow toward
    // increasing s over the far-upstream, upstream and downstream nodes, and
    // against it; `std_*` as `quick_face_values` takes them, with the boundary
    // form at the domain's two ends, and `alt_*` the boundary form of
    // `_obstacle_streamwise`, whose far node is the next one downstream.
    std::array<std::vector<double>, 3> std_pos;
    std::array<std::vector<double>, 3> std_neg;
    std::array<std::vector<double>, 3> alt_pos;
    std::array<std::vector<double>, 3> alt_neg;

    // QUICK's weights at the transverse faces, [nt + 1] with entries 1 to
    // nt - 1 used; `wall_*` the weights of the two interior nodes where the
    // far node is an obstacle's wall, whose value is zero (`_obstacle_transverse`).
    std::array<std::vector<double>, 3> t_pos;
    std::array<std::vector<double>, 3> t_neg;
    std::array<std::vector<double>, 2> wall_pos;
    std::array<std::vector<double>, 2> wall_neg;

    // The viscosity means' fractions: the harmonic mean's two across a
    // transverse face [nt + 1], the arithmetic mean's along s [ns + 1]; the
    // interior entries used.
    std::vector<double> harmonic_low;
    std::vector<double> harmonic_high;
    std::vector<double> mean_high;

    // The distance a transverse face's conductance divides by: the nodes'
    // between two unknowns, the wall's at a domain edge or an obstacle face.
    // [ny + 1, nx + 1] in corners, m.
    std::vector<double> reach;

    // The scalar path's conductances, Pa s: cells and corners.
    std::vector<double> conductance_s;
    std::vector<double> conductance_t;

    // Cells: the far-upstream node of a flow toward increasing s, or against
    // it, lies beyond an obstacle's wall node.
    std::vector<lanes::mask> blocked_pos;
    std::vector<lanes::mask> blocked_neg;

    // Corners: an obstacle face of the wall stencil (`_ObstacleWalls.face`); a
    // face that takes `wall_mu` (an obstacle face or a Dirichlet edge's); a
    // face that conducts (all but a zero-gradient edge's); the far node of a
    // flow toward increasing t, or against it, is an obstacle's wall; a face
    // whose wall viscosity an unknown reads (`_wall_read`).
    std::vector<lanes::mask> wall_face;
    std::vector<lanes::mask> takes_wall;
    std::vector<lanes::mask> conducts;
    std::vector<lanes::mask> far_wall_low;
    std::vector<lanes::mask> far_wall_high;
    std::vector<lanes::mask> wall_read;

    // The component's faces: not a domain-edge face and not a face of a SOLID
    // cell; the neighbour on the high or the low transverse side is not an
    // obstacle's (true also beside a domain edge, which the assembly treats);
    // a vent's face the sweeps keep (empty without vents).
    std::vector<lanes::mask> unknown;
    std::vector<lanes::mask> couples_high;
    std::vector<lanes::mask> couples_low;
    std::vector<lanes::mask> held;

    // The edges' conditions along s, [ns + 1].
    std::vector<lanes::mask> low_known;
    std::vector<lanes::mask> high_known;
    std::vector<double> low_value;
    std::vector<double> high_value;

    // Whether a row of cells holds a blocked one, and a row of corners an
    // obstacle's, so that a row with neither runs the block without them.
    std::vector<std::uint8_t> row_blocked;
    std::vector<std::uint8_t> row_walls;
};

// The predictor on one grid, which must outlive it. Every array is allocated
// at construction; a call allocates nothing.
template <int D>
    requires(D == 2)
class predictor {
public:
    // Why `input` cannot make a predictor on `g`, or an empty view when it
    // can: fewer than two cells along a direction, an array whose length is
    // not the grid's, or fewer than one sweep.
    static std::string_view check(const grid::grid<D>& g, const predictor_input<D>& input);

    // Builds the predictor from an input `check` accepts, copying every array.
    // Throws std::invalid_argument for one it does not.
    predictor(const grid::grid<D>& g, const predictor_input<D>& input);

    // The checks of `predict`'s viscosities. `mu_eff`: [ny, nx], Pa s, finite
    // and positive in every non-SOLID cell. `wall_mu`: component c's, in
    // corners, finite and positive at every face `wall_read` marks; what it
    // holds elsewhere is not read.
    viscosity_status check_viscosity(std::span<const double> mu_eff) const;
    viscosity_status check_wall_viscosity(int c, std::span<const double> wall_mu) const;

    // Both momentum equations assembled on the current field, with the QUICK
    // minus upwind flux as an explicit source, and `sweeps` under-relaxed
    // Jacobi sweeps of each (`MomentumPredictor.predict`, transient None).
    //
    // velocity: u [ny, nx + 1] and v [ny + 1, nx], m/s, the domain-edge faces
    //           and the vents' faces already set
    // p:        pressure, [ny, nx], Pa
    // mu_eff:   dynamic effective viscosity per cell, [ny, nx], Pa s, one
    //           `check_viscosity` accepts; empty for air's, the scalar path.
    //           SOLID cells are not read
    // wall_mu:  per component, the wall stencil's viscosity in corners, one
    //           `check_wall_viscosity` accepts; both empty for air's or the
    //           field's own
    // star:     output, u* and v*, the velocity's shapes. Domain-edge faces
    //           keep the input; faces of SOLID cells are zero, except a vent's
    // diagonal: output, the un-relaxed diagonals, kg/(m s) per unit depth, the
    //           velocity's shapes; positive exactly at the faces the pressure
    //           correction may correct and zero elsewhere
    //
    // An output overlaps no input and no other output.
    void predict(const std::array<std::span<const double>, D>& velocity,
                 std::span<const double> p, std::span<const double> mu_eff,
                 const std::array<std::span<const double>, D>& wall_mu,
                 const std::array<std::span<double>, D>& star,
                 const std::array<std::span<double>, D>& diagonal);

    // The viscosity the wall stencil takes when `wall_mu` is not given
    // (`MomentumPredictor.stencil_viscosity`).
    //
    // mu_eff: as `predict` takes it, not empty
    // out:    output per component, corners, Pa s: the field's face rule at
    //         every face `wall_read` marks, air's viscosity on the rest.
    //         Overlaps no input
    void stencil_viscosity(std::span<const double> mu_eff,
                           const std::array<std::span<double>, D>& out);

    // One component's equations as `predict` assembles them, without the
    // sweeps: its diagonal into `diagonal` and the parts below, which stand
    // until the next call of either.
    //
    // c: the component; the other arguments as `predict` takes them. The
    // output overlaps no input
    void equations(int c, const std::array<std::span<const double>, D>& velocity,
                   std::span<const double> p, std::span<const double> mu_eff,
                   const std::array<std::span<const double>, D>& wall_mu,
                   std::span<double> diagonal);

    // The parts of the component last assembled, in its face layout: the
    // neighbour coefficients toward s+, s-, t+ and t- (which = 0 to 3), zero
    // at faces that are not unknowns, and the whole source with the
    // relaxation term, read at unknowns only.
    std::span<const double> neighbour(int which) const { return coefficient_[which]; }
    std::span<const double> source() const { return source_; }

    // What is fixed at construction for component c.
    const component_setup& setup(int c) const { return setup_[c]; }

    const grid::grid<D>& mesh() const { return *grid_; }

private:
    // One row of the fluxes through a row of cells or of corners: the mass
    // flux, the coefficients it gives the unknowns on its low and its high side
    // (toward s+ or t+, and toward s- or t-), the QUICK minus upwind flux and
    // the stress flux of form b. One of two, the row before kept for the
    // unknowns that lie between two rows.
    struct flux_row {
        std::vector<double> conductance;
        std::vector<double> flux;
        std::vector<double> plus;
        std::vector<double> minus;
        std::vector<double> deferred;
        std::vector<double> stress;
        std::vector<double> viscosity;
    };

    template <int C>
    void assemble(const std::array<std::span<const double>, D>& velocity,
                  std::span<const double> pressure, std::span<const double> mu_eff,
                  const std::array<std::span<const double>, D>& wall_mu, double* diagonal);
    template <int C>
    void cell_row(std::size_t j, const double* own, const double* mu, flux_row& out);
    template <int C>
    void corner_row(std::size_t j, const double* own, const double* other, const double* mu,
                    const double* wall, flux_row& out);
    template <int C>
    void corner_viscosity(std::size_t j, const double* mu, double* out);
    template <int C>
    void unknown_row(std::size_t j, const double* own, const double* p, const double* mu,
                     const flux_row* cell_minus, const flux_row* cell_plus,
                     const flux_row* corner_minus, const flux_row* corner_plus,
                     double* diagonal);
    template <int C>
    void sweep(const double* own, const double* diagonal, double* star);

    const grid::grid<D>* grid_;
    std::array<component_setup, D> setup_;
    std::size_t nx_;
    std::size_t ny_;
    double mu_;
    double alpha_;
    double half_rho_;
    double relaxation_;
    std::int64_t sweeps_;

    // One component's equations and the sweeps' other iterate. The two
    // components take turns in them, so a call touches half the memory.
    std::array<std::vector<double>, 4> coefficient_;
    std::vector<double> source_;
    std::vector<double> relaxed_;
    std::vector<double> iterate_;
    std::array<flux_row, 2> cell_rows_;
    std::array<flux_row, 2> corner_rows_;
    // Rows of the harmonic means a corner's viscosity averages, and of edge
    // values standing for a row of nodes.
    std::array<std::vector<double>, 2> means_;
    std::vector<double> edge_;
    std::vector<double> zeros_;
};

extern template class predictor<2>;

}  // namespace cfdcore::momentum
