// The cell-centred scalar scheme: the explicit face flux and the implicit
// solve (ADR-017 C, module `scalar`; ADR-011 B, C and E, ADR-012 C, ADR-014 J,
// ADR-016).
//
// A port of `src/scalar_scheme.py`'s `limited_face_values` and
// `advective_flux` and of `src/momentum.py`'s `quick_face_values` with its
// Lagrange weights, written to equal them bit for bit on finite inputs
// (REQ-N03 level 1, ADR-017 G): every value is formed by the reference's
// operations in the reference's order, one rounding each (the build's
// -ffp-contract=off), and nothing is reassociated. With a NaN in an input both
// give NaNs at the same faces, but a NaN's sign or payload may differ; the
// transport solver refuses a non-finite field before the flux runs. The
// reference advects along its last axis and transposes for y; the core
// transposes nothing and walks every field by its rows, the lines along
// direction 0, which are contiguous in memory (layout.hpp): for the faces
// normal to another direction a row of faces lies between two rows of cells,
// and line m along y is row m of the reference's transposed call.
//
// The face flux is elementwise over the faces: no sum and no iteration. The
// implicit solve, `implicit_step` and its parts, is the second half of this
// header (prompt 88).

#pragma once

#include "cfdcore/grid.hpp"
#include "cfdcore/lanes.hpp"
#include "cfdcore/layout.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace cfdcore::scalar {

// NumPy's np.minimum and np.maximum on two doubles: a NaN in either gives a
// NaN, and where the two compare equal (0.0 and -0.0 among them) the second
// is returned, as NumPy 2.4's loops return it. The rule is pinned by
// tests/test_core_parity.py::test_numpy_min_and_max_return_the_second_of_a_tie
// and measured on every length from 1 to 1,000 by docs/reports/probe87/ties87.py.
// The sign of a zero psi reaches the face value when the upstream value is
// -0.0, so the tie matters for bit equality.
//
// Written with masks (lanes.hpp) and no branch, so that a kernel's block can
// hold them: `a != a` is the NaN test.
CFDCORE_INLINE double np_minimum(double a, double b) {
    return lanes::select(lanes::mask_of(a < b) | lanes::mask_of(a != a), a, b);
}
CFDCORE_INLINE double np_maximum(double a, double b) {
    return lanes::select(lanes::mask_of(a > b) | lanes::mask_of(a != a), a, b);
}

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
//
// The limited value is formed whatever the difference is and discarded where it
// is zero, so that the function has no branch: the quotients are then 0 / 0 or
// x / 0, which round to a NaN or an infinity and are never read.
CFDCORE_INLINE double limited_face_value(double c_up, double c_c, double c_d, double quick) {
    const double d_down = c_d - c_c;
    // The reference's `d_down != 0.0`: a NaN difference takes the limited branch.
    const lanes::mask defined = lanes::mask_of(d_down != 0.0);
    const double r = (c_c - c_up) / d_down;
    const double psi_quick = 2.0 * (quick - c_c) / d_down;
    // np.minimum.reduce over [2r, (1 + 3r) / 4, psi_quick, 2] is taken as a fold
    // from the first. The evidence is a measurement of the reduce against that
    // fold (docs/reports/probe87/ties87.py), not an argument from the tie rule.
    double low = np_minimum(2.0 * r, (1.0 + 3.0 * r) / 4.0);
    low = np_minimum(low, psi_quick);
    low = np_minimum(low, 2.0);
    const double psi = np_maximum(0.0, low);
    return lanes::select(defined, c_c + 0.5 * psi * d_down, c_c);
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

    // The grid the scheme was built on: the caller's own, not a copy, so it is
    // valid as long as that grid is.
    const grid::grid<D>& mesh() const { return *grid_; }

    // The QUICK weights of direction d, 0 to D - 1: each of the six arrays
    // holds one entry per interior face along d (n_d - 1 of them, dimensionless),
    // entry f for the face between cells f and f + 1.
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
//         scalar's unit times m^2/s; it must overlap no input, which the
//         kernels are compiled to assume (the binding refuses one that does)
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

// ---------------------------------------------------------------------------
// The implicit solve (`implicit_step`; ADR-011 C, ADR-012 C, ADR-016).
//
// Backward Euler for diffusion and a non-negative cell sink:
// a_P C_P - sum_f G_f C_N = (V / dt) C*, a_P = V / dt + sum_f G_f + D_P, by
// Jacobi or by diagonal-preconditioned conjugate gradients, with the CG's
// fallback to Jacobi from C*. Every elementwise value is formed by the
// reference's operations in the reference's order, one rounding each, and the
// leftover's sum in NumPy's pairwise order, so the Jacobi path equals the
// reference bit for bit (REQ-N03 level 1). The CG's two dot products per
// iteration are the one departure: the reference's go to OpenBLAS's ddot,
// whose order is the library's, and the core's go through
// `reduce::blocked_dot` with a block per row of the cell field (ADR-017 G,
// the second kind), so a CG solve is compared for the reference's iteration
// count with its field within 1e-12 (prompt 88).
//
// A cell field is one value per cell in layout.hpp's order ([ny, nx] for
// D = 2); a face field normal to d one value per face normal to d.

// The two methods (`JACOBI` and `CONJUGATE_GRADIENT`).
enum class implicit_method { jacobi, conjugate_gradient };

// Whether a solve's inputs can be solved: the one refusal the core makes, the
// reference's ValueError for a leftover bound that is negative or NaN. The
// method's name and a bool bound are the binding's to refuse.
enum class implicit_status { ok, bound_negative_or_nan };

// The inputs of one solve.
//
// c_star:          the explicit field C*, a cell field in the scalar's unit
//                  (particles per m^3, or k and eps); zero in SOLID cells by
//                  the callers' contract, though the solve reads what it is given
// volume:          cell volumes per metre of depth, m^2, a cell field
// dt:              the step, s, when dt_cells is empty
// dt_cells:        the step per cell, s, a cell field, or empty for `dt`
// conductance:     per direction d, G_f = Gamma_f A_f / d_f per face normal
//                  to d, m^2/s per metre of depth; zero where nothing diffuses
// sink:            the non-negative sink per cell in the diagonal (the
//                  reference's `diagonal`), in the units of V / dt
// solid:           the SOLID mask, a cell field; those cells are held at zero
// held:            non-SOLID cells held at their value of C*, a cell field, or
//                  empty for none
// tol:             the stop: the largest residual at most tol times the
//                  largest |b|, dimensionless
// max_iter:        the cap, Jacobi sweeps or CG iterations; zero or negative
//                  allows none
// method:          Jacobi, or CG with the fallback
// leftover_bound:  ADR-016's bound on |leftover|, in the units of V C, or none
template <int D>
struct implicit_input {
    std::span<const double> c_star;
    std::span<const double> volume;
    double dt;
    std::span<const double> dt_cells;
    std::array<std::span<const double>, D> conductance;
    std::span<const double> sink;
    std::span<const bool> solid;
    std::span<const bool> held;
    double tol;
    std::int64_t max_iter;
    implicit_method method;
    std::optional<double> leftover_bound;
};

// What one solve returns beside its field (`ImplicitResult`), and two counts
// the reference does not return.
//
// method:          the method asked for
// iterations:      Jacobi sweeps, or CG iterations (one matrix product each)
// converged:       whether the returned field met the stop within the cap,
//                  the fallback's when there was one
// fallbacks:       1 when a CG solve fell back to Jacobi from C*, else 0
// fallback_sweeps: the fallback's Jacobi sweeps, 0 without one
// leftover:        dt times the sum of the returned field's residual (with a
//                  per-cell dt, the sum of dt r), in the units of V C
// leftover_sums:   how many times a leftover was summed over the cells, by
//                  every acceptance of the solve: the stop's cost (ADR-016,
//                  "formed only after the max-norm test holds")
// restarts:        CG restarts from the true residual after the recursion met
//                  the stop and the true residual did not
struct implicit_result {
    implicit_method method = implicit_method::jacobi;
    std::int64_t iterations = 0;
    bool converged = true;
    int fallbacks = 0;
    std::int64_t fallback_sweeps = 0;
    double leftover = 0.0;
    std::int64_t leftover_sums = 0;
    std::int64_t restarts = 0;
};

// The scratch of the implicit solve on one layout, allocated once at set-up
// so that a solve allocates nothing. Every array is a cell field.
template <int D>
    requires(D >= 1)
struct implicit_scratch {
    explicit implicit_scratch(const layout::layout<D>& fields);

    const layout::layout<D>* fields;
    std::vector<double> a_p;        // the diagonal
    std::vector<double> b;          // the right-hand side (V / dt) C*
    std::vector<double> c;          // the iterate
    std::vector<double> next;       // Jacobi's next iterate
    std::vector<double> residual;   // Jacobi's residual, CG's r
    std::vector<double> inverse;    // CG's preconditioner, 1 / a_P off the fixed cells
    std::vector<double> z;          // CG's preconditioned residual
    std::vector<double> p;          // CG's direction
    std::vector<double> q;          // CG's product A p
    std::vector<double> dt_masked;  // a per-cell dt with SOLID cells at zero
    std::vector<double> terms;      // the leftover's terms dt r per cell
    // Set where a cell is an unknown (neither SOLID nor held) and where it is
    // not SOLID (lanes.hpp), formed from a solve's masks at its start; `free`
    // is formed only when a cell is held, `fluid` standing for it otherwise.
    std::vector<std::uint64_t> free;
    std::vector<std::uint64_t> fluid;
    // One row of zeros, cells(0) long: the neighbours of a row beyond the
    // field's end, as the reference's zero padding reads them.
    std::vector<double> zeros;
};

// The leftover bound's refusal: negative or NaN (an infinite bound accepts
// every finite leftover). None is accepted.
implicit_status check_leftover_bound(std::optional<double> bound);

// The system's set-up (`implicit_step` before the solve).
//
// Writes a_P and b into the scratch and returns whether anything diffuses or
// sinks (the reference's `diag_extra.any()`) and the stop. The diagonal's
// extra terms are summed west, east, south, north (low then high face of
// each direction), then the sink; a_P = V / dt + that; b = (V / dt) C*; the
// stop is tol times the largest |b| over every cell, NaN if any |b| is NaN.
// When nothing diffuses or sinks the stop is 0, and a_P and b hold what the
// one pass wrote, which no caller reads.
struct implicit_setup {
    bool diffuses;
    double stop;
};
template <int D>
implicit_setup implicit_system(const implicit_input<D>& in, implicit_scratch<D>& scratch);

// ADR-016's acceptance (`_Acceptance`): the stop's second clause, and the
// leftover of the field a solve returns.
//
// `test` is called with the residual of an iterate that met the max-norm stop:
// it forms the leftover from that residual and accepts the iterate when there
// is no bound or |leftover| <= bound; with no bound it forms nothing. `final`
// is called with the residual of the iterate the solve returns and forms its
// leftover, unless the last formation was from that same residual. A residual
// is named by an id its solver gives each new residual it forms, so a
// residual updated in place keeps its id, as the reference's array keeps its
// identity. The leftover is dt times the pairwise sum of the residual (scalar
// dt), or the pairwise sum of dt_masked times the residual (per-cell dt, with
// dt_masked zero in SOLID cells, where a per-cell dt need not be finite).
class acceptance {
public:
    // dt, dt_cells: the step as implicit_input holds it; solid: the SOLID mask;
    // bound: the leftover bound or none; dt_masked, terms: scratch cell fields,
    // unused with a scalar dt.
    acceptance(double dt, std::span<const double> dt_cells, std::span<const bool> solid,
               std::optional<double> bound, std::span<double> dt_masked,
               std::span<double> terms);

    bool test(std::span<const double> residual, std::uint64_t residual_id);
    void final(std::span<const double> residual, std::uint64_t residual_id);

    // The leftover last formed, 0.0 before any.
    double leftover() const { return leftover_; }
    // How many times a leftover was formed.
    std::int64_t sums() const { return sums_; }

private:
    double form(std::span<const double> residual, std::uint64_t residual_id);

    bool scalar_;
    double dt_;
    std::span<const double> dt_masked_;
    std::span<double> terms_;
    std::optional<double> bound_;
    std::uint64_t formed_ = 0;
    double leftover_ = 0.0;
    std::int64_t sums_ = 0;
};

// The outcome of one Jacobi or CG solve inside `implicit_step`.
struct solve_outcome {
    std::int64_t iterations;
    bool converged;
    std::int64_t restarts;
};

// Jacobi sweeps from C* (`_jacobi`), on a system `implicit_system` set up.
//
// Each sweep forms the residual r = b + sum_f G_f C_N - a_P C (the neighbour
// sum taken east, west, north, south, a missing neighbour read as zero), zero
// in SOLID and held cells; stops when the largest |r| is at most `stop` and
// `accept` takes the iterate, or at the cap; else C <- C + r / a_P with SOLID
// cells set to zero. The returned field is written to `out` (a cell field),
// which also holds iterates while the solve runs and must overlap no input;
// `accept.final` is called with the last residual. A non-negative C* gives
// non-negative iterates throughout.
template <int D>
solve_outcome jacobi(const implicit_input<D>& in, double stop, acceptance& accept,
                     implicit_scratch<D>& scratch, std::span<double> out);

// Diagonal-preconditioned conjugate gradients (`_conjugate_gradient`), on a
// system `implicit_system` set up, from C* with SOLID cells at zero. SOLID and
// held cells are not unknowns: their residual and product are zero and their
// values never change. The iteration is Hestenes and Stiefel's with the
// residual recursion; when the recursion meets the stop, or at the cap, the
// true residual is formed, and the solve ends converged only if it meets the
// stop and `accept` takes it, else restarts from it (or, at the cap, ends).
// The direction is updated as p <- p * beta, then p <- p + z. The dot
// products are `reduce::blocked_dot` with a block per row. The field is left
// in `scratch.c`; `accept.final` is called with the last residual.
template <int D>
solve_outcome conjugate_gradient(const implicit_input<D>& in, double stop, acceptance& accept,
                                 implicit_scratch<D>& scratch);

// The implicit solve (`implicit_step`): the bound's refusal, the set-up, the
// early return when nothing diffuses or sinks (the field is C*), then the
// method; under CG, a solve that reaches its cap or leaves a value at or
// below zero in a cell it solves for is discarded and Jacobi solves from C*
// with a fresh acceptance. The field is written to `out` (a cell field) and
// the rest to `result`; on a refusal neither is written.
template <int D>
implicit_status implicit_step(const implicit_input<D>& in, implicit_scratch<D>& scratch,
                              std::span<double> out, implicit_result& result);

extern template struct implicit_scratch<1>;
extern template struct implicit_scratch<2>;

}  // namespace cfdcore::scalar
