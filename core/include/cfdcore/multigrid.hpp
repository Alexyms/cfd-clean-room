// The additive-correction multigrid (ADR-017 C, module `multigrid`; ADR-013 as
// amended 2026-10-14).
//
// A port of `src/multigrid.py` and of the operator it is handed
// (`pressure.apply_operator`): the hierarchy built from one system's
// coefficients and one symmetric V-cycle of it, the preconditioner of the
// pressure correction's conjugate gradients. Every elementwise value equals the
// reference's bit for bit (ADR-017 G); the coarsest level's pseudo-inverse is
// compared to 1e-12 and its product is `reduce`'s fixed order, so a whole cycle
// is compared to 1e-12.
//
// Written per direction (ADR-017 decision 4): a system holds a diagonal and, for
// each direction, a coupling toward the high and one toward the low neighbour,
// and a group of the next level is 2^D cells, one wide where a side is odd.
// Every field is a cell field of its level's layout (layout.hpp), and every pass
// walks it by rows. Instantiated for D = 2 and, in the core's own tests, 1.

#pragma once

#include "cfdcore/dense.hpp"
#include "cfdcore/lanes.hpp"
#include "cfdcore/layout.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace cfdcore::multigrid {

// The two-point-per-direction system of one level (`PressureCoefficients`):
//
//   (A x)_P = centre_P x_P - sum over d of (high[d]_P x_high + low[d]_P x_low)
//
// with a missing neighbour read as zero. For D = 2, high[0] is a_e, low[0] a_w,
// high[1] a_n and low[1] a_s, each a cell field, kg/(m s) per unit depth in the
// p' equation. The contract is the reference's: the couplings across the
// domain's edges are zero, as is every coefficient of a cell with no equation.
template <int D>
struct system {
    explicit system(const layout::layout<D>& fields);

    // The level's layout, which must outlive the system.
    const layout::layout<D>* fields;
    std::vector<double> centre;
    std::array<std::vector<double>, D> high;
    std::array<std::vector<double>, D> low;
};

// One row of `A x`: the cells of a line along direction 0.
//
// s:     the system
// line:  one of its layout's lines along direction 0
// zeros: a row of zeros, at least as long as a row of the layout
// x:     cell field
// out:   output, the row's n_0 values; overlaps no input
template <int D>
void product_row(const system<D>& s, const layout::line& line, std::span<const double> zeros,
                 std::span<const double> x, double* out);

// One row of `r - A x`, as `product_row`; r is a cell field.
template <int D>
void residual_row(const system<D>& s, const layout::line& line, std::span<const double> zeros,
                  std::span<const double> x, std::span<const double> r, double* out);

// `A x` whole (`apply_operator`).
//
// zeros: a row of zeros, as long as a row of the layout
// x:     cell field
// out:   output, cell field; overlaps no input
//
// Throws std::invalid_argument for an array of another length.
template <int D>
void apply_operator(const system<D>& s, std::span<const double> zeros, std::span<const double> x,
                    std::span<double> out);

// One weighted Jacobi sweep: out = x + smoother * (r - A x).
//
// zeros:    a row of zeros, as long as a row of the layout
// smoother: cell field, the weight over the diagonal, zero where there is none
// x, r:     cell fields
// out:      output, cell field; overlaps no input
//
// Throws std::invalid_argument for an array of another length.
template <int D>
void smooth(const system<D>& s, std::span<const double> zeros, std::span<const double> smoother,
            std::span<const double> x, std::span<const double> r, std::span<double> out);

// The layout of the next coarser level: half the cells along each direction,
// rounded up.
template <int D>
layout::layout<D> coarser(const layout::layout<D>& fine);

// `P^T r`: the sum over each group's cells (`restrict`), direction 0's low
// cells before its high ones.
//
// fine, coarse: the two levels' layouts, the second `coarser` of the first
// r:            cell field of `fine`
// out:          output, cell field of `coarse`; overlaps r not at all
//
// Throws std::invalid_argument for a field of another length.
template <int D>
void restrict_to(const layout::layout<D>& fine, const layout::layout<D>& coarse,
                 std::span<const double> r, std::span<double> out);

// `x += P z`: each fine cell gains its group's value (`add_prolonged`).
//
// z: cell field of `coarse`
// x: cell field of `fine`, changed in place; overlaps z not at all
//
// Throws std::invalid_argument for a field of another length.
template <int D>
void add_prolonged(const layout::layout<D>& fine, const layout::layout<D>& coarse,
                   std::span<const double> z, std::span<double> x);

// The Galerkin operator `P^T A P` (`coarsen`): a coarse coupling sums the fine
// ones across its face; a coarse diagonal is the group's diagonals less twice
// the couplings inside the group. A missing cell of a group one wide adds zero.
//
// fine:  the finer system
// zeros: a row of zeros, as long as a row of the finer layout
// out:   output, the coarser one, on `coarser(*fine.fields)`
//
// Throws std::invalid_argument for a system on another layout or a row of
// another length. The reference's coarse mask of the cells with an equation is
// not formed: nothing reads it.
template <int D>
void coarsen(const system<D>& fine, std::span<const double> zeros, system<D>& out);

// The hierarchy's fixed numbers, handed in (`src/multigrid.py`'s constants and
// `np.linalg.pinv`'s default cutoff), so that each is defined once.
//
// coarsest_cells: coarsening stops at this many cells or fewer, or when a side
//                 has one (COARSEST_CELLS), at least 1
// sweeps:         weighted Jacobi sweeps before and after each coarse
//                 correction (SMOOTHING_SWEEPS), at least 1
// weight:         the sweeps' weight (SMOOTHER_FACTOR / 2), finite and positive
// dominance_tolerance: the rounding allowed when a row is tested for weak
//                 diagonal dominance (DOMINANCE_TOLERANCE), at least 0
// rcond:          the pseudo-inverse's cutoff, at least 0
struct keys {
    std::int64_t coarsest_cells = 0;
    std::int64_t sweeps = 0;
    double weight = 0.0;
    double dominance_tolerance = 0.0;
    double rcond = 0.0;
};

// Whether a hierarchy was built, in the reference's order: the finest system is
// not weakly diagonally dominant (its refusal), or the coarsest level's
// eigenvalues did not converge (dense.hpp).
enum class build_status { ok, not_dominant, not_converged };

// The hierarchy of a finest level's layout: every level's arrays allocated at
// construction, their values set by `build` for one system at a time.
template <int D>
    requires(D >= 1)
class hierarchy {
public:
    // fields: the finest level's layout, which must outlive the hierarchy.
    // Throws std::invalid_argument for keys outside their ranges.
    hierarchy(const layout::layout<D>& fields, const keys& k);
    // A level's system points at the level's layout.
    hierarchy(const hierarchy&) = delete;
    hierarchy& operator=(const hierarchy&) = delete;

    // Builds every level for the finest system `s` (`Multigrid.__init__`), which
    // must stay as it is while `cycle` is called. After a status other than ok
    // the hierarchy is not to be cycled. Throws std::invalid_argument for a
    // system on another layout.
    build_status build(const system<D>& s);

    // One V(sweeps, sweeps) cycle from zero on A z = r (`Multigrid.cycle`).
    //
    // r: cell field of the finest level
    // z: output, likewise, zero where the diagonal is not positive; overlaps no
    //    input
    //
    // Throws std::invalid_argument for a field of another length, or before a
    // `build`.
    void cycle(std::span<const double> r, std::span<double> z);

    // The levels, finest first, and each one's system and smoother. The
    // coarsest level is solved and not smoothed: its smoother is empty.
    std::size_t levels() const { return levels_.size(); }
    const system<D>& level_system(std::size_t level) const;
    std::span<const double> smoother(std::size_t level) const { return levels_[level].smoother; }

    // The coarsest level's operator as a dense matrix and its pseudo-inverse,
    // n * n values each for its n cells in the cell field's order.
    std::span<const double> coarsest_matrix() const { return matrix_; }
    std::span<const double> coarsest_inverse() const { return inverse_; }

private:
    struct level {
        explicit level(const layout::layout<D>& given) : fields(&given) {}
        // The level's layout and its system: the finest's are the caller's.
        const layout::layout<D>* fields;
        std::unique_ptr<layout::layout<D>> own_fields;
        std::unique_ptr<system<D>> own;
        std::vector<double> smoother;
        // The level's residual (the finest's is the caller's), its iterate, the
        // sweeps' other iterate, and one group's rows of the fine residual.
        std::vector<double> r;
        std::vector<double> x;
        std::vector<double> other;
        std::vector<double> rows;
    };

    void form_matrix(const system<D>& s);
    void solve_coarsest(std::span<const double> r, std::span<double> z) const;
    void descend(std::size_t index, const double* r);
    void ascend(std::size_t index, const double* r, double* z);

    keys keys_;
    std::vector<level> levels_;
    const system<D>* finest_ = nullptr;
    dense::semidefinite eigen_;
    std::vector<double> matrix_;
    std::vector<double> inverse_;
    // A row of zeros: the neighbours beyond the field's edges.
    std::vector<double> zeros_;
};

extern template struct system<1>;
extern template struct system<2>;
extern template class hierarchy<1>;
extern template class hierarchy<2>;

}  // namespace cfdcore::multigrid
