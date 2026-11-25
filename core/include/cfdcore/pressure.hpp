// The pressure correction on the staggered grid (ADR-017 C, module `pressure`;
// ADR-013 as amended 2026-10-14).
//
// A port of `src/pressure.py`: the p' equation's coefficients from the predicted
// velocities and the momentum diagonals, its solve by preconditioned conjugate
// gradients, and the correction of the velocities and the pressure. Every
// elementwise value and the closed domain's means equal the reference's bit for
// bit (ADR-017 G, the first kind). The conjugate gradients' dot products are
// `reduce`'s fixed blocks, one per row of cells, where the reference's are
// OpenBLAS's, and the preconditioner's coarsest solve is the core's own, so a
// correction is compared at the reference's iteration count (REQ-N03 level 1).
//
// Written per direction (ADR-017 decision 4): the component of the velocity
// along direction c lives on the faces normal to c, and what the reference
// writes for u and for v is one form of c. Every field is walked by its rows.
// Instantiated for D = 2 and, in the core's own tests, 1.

#pragma once

#include "cfdcore/grid.hpp"
#include "cfdcore/lanes.hpp"
#include "cfdcore/layout.hpp"
#include "cfdcore/multigrid.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace cfdcore::pressure {

// The preconditioner of the conjugate gradients (`solver.pressure_preconditioner`).
enum class preconditioner { jacobi, multigrid };

// The conjugate gradients' arguments, as `_check_solve_arguments` reads them.
//
// rtol:     relative level, in [0, 1): the stop is ||r|| <= rtol ||f||
// floor:    absolute level, finite and at least 0: the stop is also met at
//           ||r|| <= floor
// max_iter: the iteration cap, at least 1
// method:   the preconditioner
// project:  on a closed domain the cells with an equation, in ascending order: a
//           multigrid-preconditioned vector has its mean over them removed. Empty
//           on an open one, and not read under jacobi
struct solve_keys {
    double rtol = 0.0;
    double floor = 0.0;
    std::int64_t max_iter = 0;
    preconditioner method = preconditioner::multigrid;
    std::span<const std::size_t> project;
};

// The cells whose diagonal is positive, the ones with an equation, in ascending
// order.
//
// centre: the diagonal, a cell field
// out:    output, at least as long; the entries past those returned are not used
//
// Returns the part of `out` that holds them.
std::span<const std::size_t> cells_with_equation(std::span<const double> centre,
                                                 std::span<std::size_t> out);

// How a solve ended. `not_dominant` is the reference's refusal of a system that
// is not weakly diagonally dominant, `not_converged` a coarsest level whose
// eigenvalues do not converge (LAPACK's error there), and `division_by_zero` a
// step or a direction whose denominator is zero, where the reference's own
// division raises.
enum class solve_status { ok, not_dominant, not_converged, division_by_zero };

// What a solve returns (`ConjugateGradientResult` without its field).
//
// iterations:    iterations performed
// products:      products with the operator: one per iteration, one per true
//                residual
// residual_norm: 2-norm of the true residual at exit, the right-hand side's unit
// reached_cap:   the cap ended the solve
struct solve_result {
    solve_status status = solve_status::ok;
    std::int64_t iterations = 0;
    std::int64_t products = 0;
    double residual_norm = 0.0;
    bool reached_cap = false;
};

// The arithmetic mean of a field over some of its cells, the bits of the
// reference's `field[mask].mean()`.
//
// field: cell field
// cells: the indices of those cells in ascending order, at least one
double mean_over(std::span<const double> field, std::span<const std::size_t> cells);

// Preconditioned conjugate gradients on one layout (`conjugate_gradient`), every
// array allocated at construction; a solve allocates nothing.
template <int D>
    requires(D >= 1)
class solver {
public:
    // fields: the cells' layout, which must outlive the solver. The multigrid's
    // levels are allocated only where they are wanted (`with_multigrid`).
    // Throws std::invalid_argument for multigrid keys outside their ranges.
    solver(const layout::layout<D>& fields, bool with_multigrid, const multigrid::keys& keys);

    // Solves A x = f from zero.
    //
    // s:   the system, on the solver's layout, `multigrid::system`'s contract
    // f:   right-hand side, cell field; zero where there is no equation
    // k:   the solve's numbers; the list it views must stand while the solver is
    //      used with them
    // x:   output, cell field. Overlaps no input. Where the status is not ok it
    //      holds nothing of use
    //
    // The iteration is Hestenes and Stiefel's with the residual recursion; a
    // stop on it is confirmed on the true residual, which the solve restarts
    // from where it fails (ADR-013 B). The dot products are
    // `reduce::blocked_dot` with a block per row. Throws std::invalid_argument
    // for a field of another length, keys outside their ranges, or multigrid
    // asked of a solver built without it.
    solve_result solve(const multigrid::system<D>& s, std::span<const double> f,
                       const solve_keys& k, std::span<double> x);

    // One application of the preconditioner a solve under `k` makes,
    // z = M^-1 r, for the parity tests: `prepare` sets it up for a system.
    solve_status prepare(const multigrid::system<D>& s, const solve_keys& k);
    void precondition(std::span<const double> r, std::span<double> z);

    // The multigrid's levels; null without it.
    const multigrid::hierarchy<D>* hierarchy() const { return hierarchy_.get(); }

private:
    double precondition_dot(std::span<const double> r, std::span<double> z);
    double true_residual(const multigrid::system<D>& s, std::span<const double> f,
                         std::span<const double> x);

    const layout::layout<D>* fields_;
    std::unique_ptr<multigrid::hierarchy<D>> hierarchy_;
    preconditioner method_ = preconditioner::jacobi;
    std::span<const std::size_t> project_;
    const double* centre_ = nullptr;  // the diagonal of the system prepared
    std::vector<double> inverse_;     // the diagonal preconditioner
    std::vector<double> r_;
    std::vector<double> z_;
    std::vector<double> p_;
    std::vector<double> q_;
    std::vector<double> zeros_;
};

// What a corrector is built from besides its grid.
//
// cell_widths: per direction, the n_d cell widths, m (`Mesh.dx_cell`, `dy_cell`)
// outlet_low, outlet_high: per direction, whether a face of the low and of the
//              high domain edge normal to it is a pressure outlet, one per face
//              in the cell layout without that direction
//              (`StaggeredBoundary.pressure_outlets`): [ny] for left and right,
//              [nx] for bottom and top
// fluid:       the type of a FLUID cell (`mesh.FLUID`), for the pin
// rho, alpha:  air's density, kg/m^3, and the pressure's relaxation
//              (`solver.alpha_pressure`)
// rtol, floor, max_iter, method: `solve_keys`'; floor is the rounding level of
//              the stop, RESIDUAL_FLOOR times the flux scale, kg/s per unit depth
// levels:      the multigrid's keys, read under that method alone
template <int D>
struct corrector_input {
    std::array<std::span<const double>, D> cell_widths;
    std::array<std::span<const bool>, D> outlet_low;
    std::array<std::span<const bool>, D> outlet_high;
    std::int32_t fluid = 0;
    double rho = 0.0;
    double alpha = 0.0;
    double rtol = 0.0;
    double floor = 0.0;
    std::int64_t max_iter = 0;
    preconditioner method = preconditioner::multigrid;
    multigrid::keys levels;
};

// The connected components of the cells with a pressure equation, and how many
// of them reach no pressure outlet (`PressureCorrector._check_components`).
struct component_count {
    std::int64_t components = 0;
    std::int64_t stranded = 0;
};

// The pressure correction on one grid, which must outlive it. Every array is
// allocated at construction; a call allocates nothing.
template <int D>
    requires(D >= 1)
class corrector {
public:
    // Why `input` cannot make a corrector on `g`, or an empty view when it can:
    // an array whose length is not the grid's, a width, rho or alpha that is not
    // finite and positive. Whether the keys are in their ranges and the
    // components solvable the caller asks apart, since the reference refuses
    // those with messages of its own.
    static std::string_view check(const grid::grid<D>& g, const corrector_input<D>& input);

    // The components of the cells with an equation, for the refusal of a domain
    // the correction cannot solve: a closed domain of more than one, an open one
    // with any stranded. Reads an input `check` accepts.
    static component_count count_components(const grid::grid<D>& g,
                                            const corrector_input<D>& input);

    // Builds the corrector from an input `check` accepts, copying every array.
    // Throws std::invalid_argument for one it does not and for keys outside
    // their ranges.
    //
    // Every call below throws std::invalid_argument for an array whose length is
    // not the grid's.
    corrector(const grid::grid<D>& g, const corrector_input<D>& input);

    // Whether no edge face is a pressure outlet, so p' is pinned; and the cell
    // it is pinned at, the first typed FLUID (0 where there is none), as an
    // index into the cell field.
    bool needs_pin() const { return needs_pin_; }
    std::size_t pin_cell() const { return pin_; }

    // The mass imbalance of every cell from the face velocities
    // (`mass_imbalance`): rho times the sum over the directions of the two
    // faces' difference times their area, zero in SOLID cells.
    //
    // velocity: per component, its faces, m/s
    // out:      output, cells, kg/s per unit depth; overlaps no input
    void mass_imbalance(const std::array<std::span<const double>, D>& velocity,
                        std::span<double> out) const;

    // The p' equation from the momentum diagonals (`coefficients`) and, for the
    // correction, d = A_face / a_P on every face, zero where a face is not
    // correctable, an outlet's borrowed from the nearest interior face. They
    // stand until the next call.
    //
    // diagonal: per component, its faces, un-relaxed, kg/(m s) per unit depth
    void assemble(const std::array<std::span<const double>, D>& diagonal);
    const multigrid::system<D>& equation() const { return system_; }
    std::span<const double> face_d(int c) const { return d_[c]; }

    // The right-hand side of the equation last assembled: minus the mass
    // imbalance, on a closed domain less its mean over the cells with an
    // equation, zero in the others.
    //
    // star: per component the predicted velocity, its faces, m/s
    // out:  output, cells; overlaps no input
    void right_hand_side(const std::array<std::span<const double>, D>& star,
                         std::span<double> out) const;

    // The correction by a solved p' of the equation last assembled: on a closed
    // domain p' has the reference cell's value removed first.
    //
    // star:     as `right_hand_side`
    // p:        pressure, cells, Pa
    // p_prime:  in and out, cells, Pa: the solve's and then the pinned one
    // velocity: output, per component its corrected faces, m/s
    // p_next:   output, cells, Pa: p plus the relaxed p' in the cells with an
    //           equation, pinned on a closed domain
    //
    // An output overlaps no input and no other output.
    void apply(const std::array<std::span<const double>, D>& star, std::span<const double> p,
               std::span<double> p_prime, const std::array<std::span<double>, D>& velocity,
               std::span<double> p_next) const;

    // The whole correction (`PressureCorrector.correct`): the arguments of
    // `assemble`, `right_hand_side` and `apply`, the solve between them. Where
    // the result's status is not ok the outputs hold nothing of use.
    solve_result correct(const std::array<std::span<const double>, D>& star,
                         const std::array<std::span<const double>, D>& diagonal,
                         std::span<const double> p,
                         const std::array<std::span<double>, D>& velocity,
                         std::span<double> p_next, std::span<double> p_prime);

    const solver<D>& solution() const { return solver_; }
    const grid::grid<D>& mesh() const { return *grid_; }

private:
    // One row of cells: its first cell, and for each direction its low face in
    // that direction's faces and its index among the rows of an array over the
    // other directions.
    struct cell_row {
        std::size_t cell = 0;
        std::array<std::size_t, D> face{};
        std::array<std::size_t, D> across{};
    };
    // One row of the faces normal to a direction: its first face, the cells on
    // its low and its high side (`has_*`: whether there are any; along direction
    // 0 the row's own cells, and not read), and its index as above.
    struct face_row {
        std::size_t face = 0;
        std::size_t low_cell = 0;
        std::size_t high_cell = 0;
        bool has_low = false;
        bool has_high = false;
        std::size_t across = 0;
    };

    void require_faces(const std::array<std::span<const double>, D>& a, const char* what) const;
    template <int C>
    void form_d(std::span<const double> diagonal);
    template <int C>
    void add_couplings(const cell_row& row);
    template <bool Source>
    void form_imbalance(const std::array<std::span<const double>, D>& velocity,
                        double* out) const;
    template <int C>
    void correct_component(std::span<const double> star, const double* p_prime, double* out) const;

    const grid::grid<D>* grid_;
    double rho_;
    double alpha_;
    solve_keys keys_;
    bool needs_pin_ = false;
    std::size_t pin_ = 0;
    std::vector<cell_row> cell_rows_;
    std::array<std::vector<face_row>, D> face_rows_;
    // Per direction, the faces' area and the two edges' outlets, in the cell
    // layout without that direction.
    std::array<std::vector<double>, D> area_;
    std::array<std::vector<lanes::mask>, D> outlet_low_;
    std::array<std::vector<lanes::mask>, D> outlet_high_;
    multigrid::system<D> system_;
    std::array<std::vector<double>, D> d_;
    std::vector<double> f_;
    // The cells with an equation, on a closed domain, and the list's storage.
    std::vector<std::size_t> cells_;
    std::span<const std::size_t> active_;
    std::vector<double> zeros_;
    solver<D> solver_;
};

extern template class solver<1>;
extern template class solver<2>;
extern template class corrector<1>;
extern template class corrector<2>;

}  // namespace cfdcore::pressure
