// The cell-centred scalar scheme's explicit face flux (ADR-017 C, module
// `scalar`; ADR-011 B and E, ADR-014 J).
//
// A port of `src/scalar_scheme.py`'s `limited_face_values` and
// `advective_flux` and of `src/momentum.py`'s `quick_face_values` with its
// Lagrange weights, written to equal them bit for bit (REQ-N03 level 1,
// ADR-017 G): every value is formed by the reference's operations in the
// reference's order, one rounding each (the build's -ffp-contract=off), and
// nothing is reassociated. The reference advects along its last axis and
// transposes for y; the core walks its own layout along the direction's
// stride instead (layout.hpp), and line m along y is row m of the
// reference's transposed call.
//
// The face flux is elementwise over the faces: no sum and no iteration.

#pragma once

#include "cfdcore/grid.hpp"

#include <array>
#include <cmath>
#include <span>
#include <vector>

namespace cfdcore::scalar {

// NumPy's np.minimum and np.maximum on two doubles: a NaN in either gives a
// NaN, and where the two compare equal (0.0 and -0.0 among them) the second
// is returned, as NumPy 2.4's loops return it on every length (prompt 87,
// results/builder87/zeros.py). The sign of a zero psi reaches the face value
// when the upstream value is -0.0, so the tie matters for bit equality.
inline double np_minimum(double a, double b) { return (a < b || std::isnan(a)) ? a : b; }
inline double np_maximum(double a, double b) { return (a > b || std::isnan(a)) ? a : b; }

// The UMIST clamp of a QUICK face value at one face (`limited_face_values`).
//
// c_up:  the node upstream of the upstream cell
// c_c:   the upstream cell's value
// c_d:   the downstream cell's value
// quick: the unlimited quadratic's face value
// (all in the scalar's unit, which the result shares)
//
// Returns c_c + 0.5 psi (c_d - c_c) with psi = max(0, min(2r, (1 + 3r) / 4,
// psi_quick, 2)), r = (c_c - c_up) / (c_d - c_c) and
// psi_quick = 2 (quick - c_c) / (c_d - c_c); c_c where c_d - c_c is zero.
inline double limited_face_value(double c_up, double c_c, double c_d, double quick) {
    const double d_down = c_d - c_c;
    // The reference's `d_down != 0.0`: a NaN difference takes the limited branch.
    if (d_down == 0.0) {
        return c_c;
    }
    const double r = (c_c - c_up) / d_down;
    const double psi_quick = 2.0 * (quick - c_c) / d_down;
    // np.minimum.reduce over [2r, (1 + 3r) / 4, psi_quick, 2] folds from the first.
    double low = np_minimum(2.0 * r, (1.0 + 3.0 * r) / 4.0);
    low = np_minimum(low, psi_quick);
    low = np_minimum(low, 2.0);
    const double psi = np_maximum(0.0, low);
    return c_c + 0.5 * psi * d_down;
}

// The UMIST clamp elementwise over equal-length arrays (`limited_face_values`).
//
// c_up, c_c, c_d, quick: as limited_face_value, one value per face, any length
// out:                   output, the limited face value per face, same length
void limited_face_values(std::span<const double> c_up, std::span<const double> c_c,
                         std::span<const double> c_d, std::span<const double> quick,
                         std::span<double> out);

// The Lagrange weights of the quadratic through three nodes at a point x_f,
// in `_lagrange_weights`' expressions: w_a = (x_f - x_b)(x_f - x_c) /
// ((x_a - x_b)(x_a - x_c)), and w_b and w_c by cycling the nodes. Metres in,
// dimensionless out.
inline std::array<double, 3> lagrange_weights(double x_a, double x_b, double x_c, double x_f) {
    return {(x_f - x_b) * (x_f - x_c) / ((x_a - x_b) * (x_a - x_c)),
            (x_f - x_a) * (x_f - x_c) / ((x_b - x_a) * (x_b - x_c)),
            (x_f - x_a) * (x_f - x_b) / ((x_c - x_a) * (x_c - x_b))};
}

// The QUICK weights of one direction's n - 1 interior faces, interior face f
// lying between cells f and f + 1. For a flow toward increasing coordinate
// the quadratic runs through the far-upstream, upstream and downstream nodes
// (cells f - 1, f and f + 1, a boundary node standing for a missing cell);
// against it through cells f + 2, f + 1 and f. positive[i][f] and
// negative[i][f] weigh those three nodes in that order.
struct quick_weights {
    std::array<std::vector<double>, 3> positive;
    std::array<std::vector<double>, 3> negative;
};

// The scheme's set-up for a grid: the QUICK weights of every direction,
// computed once, as `quick_face_values` computes them per call from the
// nodes, `left = 1..n-1` and the interior faces of `mesh_axes`. The grid
// must outlive the scheme.
template <int D>
    requires(D >= 1)
class scheme {
public:
    explicit scheme(const grid::grid<D>& g);

    const grid::grid<D>& mesh() const { return *grid_; }
    const quick_weights& weights(int d) const { return weights_[d]; }

private:
    const grid::grid<D>* grid_;
    std::array<quick_weights, D> weights_;
};

// QUICK's unlimited face values at the interior faces normal to direction d
// (`quick_face_values` as `advective_flux` calls it, on the field with its
// two boundary nodes).
//
// d:        the direction, 0 to D - 1
// c:        the cell field (layout.hpp), the scalar's unit
// c_low:    the low boundary node's value per line along d, in
//           `layout::lines(d)`'s order
// c_high:   the high boundary node's value per line along d
// positive: one per face normal to d (the face field), true where the flow
//           crosses toward increasing coordinate; read at interior faces only
// out:      output, one per face normal to d; the interior faces are written
//           and the two boundary faces of each line are left as they are
template <int D>
void quick_face_values(const scheme<D>& s, int d, std::span<const double> c,
                       std::span<const double> c_low, std::span<const double> c_high,
                       std::span<const bool> positive, std::span<double> out);

// The advective flux through every face normal to direction d
// (`advective_flux`): the volume flux times the face value.
//
// d:      the direction, 0 to D - 1
// c:      the cell field (layout.hpp), the scalar's unit (particles per m^3,
//         or k and eps)
// flux:   the volume flux per face normal to d, m^3/s per unit depth (m^2/s
//         in 2D), positive toward increasing coordinate, zero on every face
//         that carries nothing
// inflow: the value carried in, one per face normal to d; read at a line's
//         boundary face where the flux points inward and at a vent face
// upwind: first-order upwind instead of the UMIST-limited QUICK value
// vents:  one per face normal to d, true at an equipment vent (an interior
//         face between a SOLID and a room cell that carries a flux; ADR-014
//         J), read at interior faces only; empty for none, which runs the
//         arithmetic without vents bit for bit
// out:    output, one per face normal to d: flux times the face value, the
//         scalar's unit times m^2/s
//
// The face value: at a boundary face the inflow value where the flux enters
// and the adjacent cell's otherwise; at an interior face the upstream value
// with `upwind`, else the limited QUICK value, a far-upstream node inside a
// SOLID cell read as the upstream value; at a vent the inflow value where the
// flux enters the room from the SOLID side and the room cell's where it
// leaves.
template <int D>
void advective_flux(const scheme<D>& s, int d, std::span<const double> c,
                    std::span<const double> flux, std::span<const double> inflow, bool upwind,
                    std::span<const bool> vents, std::span<double> out);

extern template class scheme<1>;
extern template class scheme<2>;

}  // namespace cfdcore::scalar
