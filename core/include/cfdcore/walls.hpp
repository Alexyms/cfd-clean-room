// The wall functions and the turbulence model's boundary (ADR-017 C, module
// `walls`; ADR-012 B and C, ADR-014 J).
//
// A port of `src/turbulence.py`'s `TurbulenceBoundary` and of the functions it
// is made of: the wall viscosity of the momentum wall stencil, and the
// conditions of each k and eps step (the inflow values, the edges' tangential
// velocities, eps held and the production given in the wall cells), each
// elementwise value by the reference's operations in the reference's order (REQ-N03
// level 1). The logarithm and the powers are the C library's (elementary.hpp).
//
// Which segment covers which face is Python's (ADR-017 E): the module is handed
// per-face data. What is fixed over a flow state, the wall faces and cells, the
// distances, the inlets' values, is formed once at construction.
//
// Written per direction (ADR-017 decision 4): a domain edge is one side, low or
// high, of a direction; a wall stencil belongs to an ordered pair (c, t), the
// component c and the direction t its wall faces are crossed along, read as
// [c][q] with q the pair's slot. The slip along a wall normal to t is the
// magnitude of the D - 1 components along it: for D = 2 the one component's.
//
// Instantiated for D = 2 and, in the core's own tests, 1 and 3.

#pragma once

#include "cfdcore/grid.hpp"
#include "cfdcore/lanes.hpp"
#include "cfdcore/layout.hpp"
#include "cfdcore/momentum.hpp"
#include "cfdcore/turbulence.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace cfdcore::walls {

// The floor y*_0 of the scalable form, where u+ = y+ meets u+ = ln(E y+) / kappa
// (`log_law_floor`): the root of kappa y = ln(E y) above 1 / kappa, by Newton's
// method from 2 / kappa. kappa, e: the law's constants, dimensionless, positive.
//
// `no_meeting` where the two laws do not meet and `not_settled` where the
// iteration does not settle, the reference's two refusals.
enum class floor_status { ok, no_meeting, not_settled };
struct floor_result {
    floor_status status = floor_status::ok;
    double y = 0.0;
};
floor_result log_law_floor(double kappa, double e);

// The constants of the law of the wall, dimensionless: kappa, E and the floor.
struct wall_law {
    double kappa = 0.0;
    double e = 0.0;
    double floor = 0.0;
};

// The wall viscosity elementwise over equal-length arrays (`wall_viscosity`):
// `rho C_mu^(1/4) k^(1/2) kappa y / ln(E max(y*, floor))`, Pa s.
//
// k_p: k at the first node, m^2/s^2; y_p: its distance to the wall, m; c_mu;
// rho: kg/m^3; nu: m^2/s; out: output, as long, overlapping no input
void wall_viscosity(std::span<const double> k_p, std::span<const double> y_p, double c_mu,
                    double rho, double nu, const wall_law& law, std::span<double> out);

// k (m^2/s^2) and eps (m^2/s^3) carried in by an inlet's face (`inflow_values`).
//
// intensity: a fraction; normal_speed: the face's normal velocity, m/s, either
// sign; dissipation_length: m
struct carried_in {
    double k = 0.0;
    double eps = 0.0;
};
carried_in inflow_values(double intensity, double normal_speed, double dissipation_length);

// One face that admits air, on a domain edge or a discharge vent's.
//
// direction: the direction the face is normal to
// face:      its index in that direction's face field
// width:     the face's width, m, which weighs its inflow in the initial values
struct inlet_face {
    int direction = 0;
    std::size_t face = 0;
    double intensity = 0.0;
    double normal_speed = 0.0;
    double dissipation_length = 0.0;
    double width = 0.0;
};

// One face of an exhaust vent.
struct exhaust_face {
    int direction = 0;
    std::size_t face = 0;
};

// What a boundary is built from besides its grid.
//
// wall_cells_low, _high: per direction, whether the face of each cell along the
//              low or the high domain edge is a wall (`EdgeWalls.cells`), one
//              entry per cell of the edge
// wall_corners_low, _high: per ordered pair (c, t), whether the domain edge along
//              t is a wall at each face position of component c on it
//              (`EdgeWalls.corners`), as `turbulence::conditions::low`
// low, high:   per ordered pair, the edges' tangential conditions
//              (`TangentialCondition`; the wall distance is not read)
// vents:       per direction, the vent faces normal to it, a face field, or empty
//              in every direction for none (`VentFaces.normal_u`, `normal_v`)
// vent_corners: per ordered pair, the faces of component c's wall stencil over a
//              vent's span, in the pair's corners, or empty for none
//              (`VentFaces.tangential_u`, `tangential_v`)
// edge_inlets: the domain faces that admit air, in the reference's order
// vent_inlets: the discharge vents' faces, after them
// exhausts:    the exhaust vents' faces
// c_mu, rho (kg/m^3), nu (m^2/s), law: the variant's and air's constants and the
//              law of the wall's
template <int D>
struct boundary_input {
    std::array<std::span<const bool>, D> wall_cells_low;
    std::array<std::span<const bool>, D> wall_cells_high;
    momentum::per_pair<std::span<const bool>, D> wall_corners_low;
    momentum::per_pair<std::span<const bool>, D> wall_corners_high;
    momentum::per_pair<momentum::edge_condition, D> low;
    momentum::per_pair<momentum::edge_condition, D> high;
    std::array<std::span<const bool>, D> vents;
    momentum::per_pair<std::span<const bool>, D> vent_corners;
    std::span<const inlet_face> edge_inlets;
    std::span<const inlet_face> vent_inlets;
    std::span<const exhaust_face> exhausts;
    double c_mu = 0.0;
    double rho = 0.0;
    double nu = 0.0;
    wall_law law;
};

// The wall-function faces of one pair's stencil (`_VelocityWalls`): where each
// lies in the pair's corners, the two cells the unknown beside it lies between,
// and the unknown's distance to it.
struct stencil_walls {
    std::vector<lanes::mask> mask;
    std::vector<std::size_t> corner;
    std::vector<std::size_t> cell_low;
    std::vector<std::size_t> cell_high;
    std::vector<double> distance;
};

// The boundary on one grid, which must outlive it. Every array is allocated at
// construction; a call allocates nothing.
template <int D>
    requires(D >= 1)
class boundary {
public:
    // Why `input` cannot make a boundary on `g`, or an empty view when it can: an
    // array whose length is not the grid's, a face that is not on the grid, an
    // exhaust's face that is on a domain edge, or a constant that is not finite
    // and positive.
    static std::string_view check(const grid::grid<D>& g, const boundary_input<D>& input);

    // Builds the boundary from an input `check` accepts, copying every array.
    // Throws std::invalid_argument for one it does not. Every call below throws
    // it for an array whose length is not the grid's.
    boundary(const grid::grid<D>& g, const boundary_input<D>& input);

    // `wall_mu` for the momentum prediction (`TurbulenceBoundary.wall_viscosity`).
    //
    // k:         cells, m^2/s^2, positive in every non-SOLID cell
    // elsewhere: per pair, the viscosity every other face of the wall stencil
    //            keeps, in corners, Pa s
    // out:       output per pair, corners; overlaps no input
    void wall_viscosity(std::span<const double> k,
                        const momentum::per_pair<std::span<const double>, D>& elsewhere,
                        const momentum::per_pair<std::span<double>, D>& out);

    // One step's conditions from the iterate and the corrected faces
    // (`TurbulenceBoundary.conditions`): views of the boundary's own arrays, which
    // stand until the next call.
    //
    // velocity: per component, its faces, m/s; k, eps: cells
    turbulence::conditions<D> conditions(
        const std::array<std::span<const double>, D>& velocity, std::span<const double> k,
        std::span<const double> eps);

    // The uniform k and eps the coupled solve starts from (`initial_values`):
    // their means over the inlet faces weighted by each face's inflow, summed as
    // Python's `sum` sums floats. None where no face admits air.
    std::optional<carried_in> initial_values() const;

    // What is fixed at construction: the wall-function faces of the pair of
    // component c and slot q, the cells with a wall face, how many walls each
    // has, and each inlet face's k and eps in the order handed in.
    const stencil_walls& stencil(int c, int q) const { return stencils_[c][q]; }
    std::span<const bool> wall_cells() const {
        return {wall_cells_.get(), grid_->fields().cell_count()};
    }
    std::span<const carried_in> inlet_values() const { return inlet_values_; }

    const grid::grid<D>& mesh() const { return *grid_; }

private:
    // One wall of a wall cell: the direction it is normal to, the distance to it, m,
    // and where its own tangential velocities start in `wall_speed_`, m/s, one for
    // each component along it, by slot.
    struct wall {
        int direction = 0;
        double distance = 0.0;
        std::size_t speed = 0;
    };
    // A wall cell, its walls in the reference's order (`first` to `last` of
    // `walls_`).
    struct wall_cell {
        std::size_t cell = 0;
        std::size_t first = 0;
        std::size_t last = 0;
    };
    // An entry of an inflow array that takes its cell's value: a domain face that
    // admits no air, or an exhaust's face.
    struct outflow_entry {
        int direction = 0;
        std::size_t face = 0;
        std::size_t cell = 0;
    };

    void build_stencils(const boundary_input<D>& input);
    void build_wall_cells(const boundary_input<D>& input);
    void build_inflow(const boundary_input<D>& input);
    double live_velocity(int d, std::size_t face, const double* velocity) const;
    void inflow(std::span<const double> q, std::array<std::vector<double>, D>& out);

    const grid::grid<D>* grid_;
    std::array<layout::layout<D>, D> faces_;
    double rho_;
    double nu_;
    wall_law law_;
    double root_c_mu_;  // C_mu^(1/4)

    std::vector<lanes::mask> fluid_;  // set where not SOLID
    std::array<std::vector<lanes::mask>, D> live_;  // the faces that carry air
    std::vector<std::vector<stencil_walls>> stencils_;  // [c][q]
    std::vector<wall> walls_;
    std::vector<wall_cell> cells_;
    std::vector<double> wall_speed_;
    std::unique_ptr<bool[]> wall_cells_;

    std::vector<inlet_face> inlets_;
    std::vector<carried_in> inlet_values_;
    std::vector<outflow_entry> outflow_;

    // The edges' tangential conditions and the conditions' own arrays.
    momentum::per_pair<std::vector<lanes::mask>, D> known_low_;
    momentum::per_pair<std::vector<lanes::mask>, D> known_high_;
    momentum::per_pair<std::vector<double>, D> value_low_;
    momentum::per_pair<std::vector<double>, D> value_high_;
    momentum::per_pair<std::vector<double>, D> tangential_low_;
    momentum::per_pair<std::vector<double>, D> tangential_high_;
    std::array<std::vector<double>, D> inflow_k_;
    std::array<std::vector<double>, D> inflow_eps_;
    std::vector<double> eps_wall_;
    std::vector<double> production_;
};

extern template class boundary<1>;
extern template class boundary<2>;

}  // namespace cfdcore::walls
