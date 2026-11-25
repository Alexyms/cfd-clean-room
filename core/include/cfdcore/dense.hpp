// Small dense matrices (ADR-017 C, beside `multigrid`): the pseudo-inverse of a
// symmetric positive semi-definite matrix, which the coarsest level is solved
// with.
//
// The reference forms it with `np.linalg.pinv(matrix, hermitian=True)`, whose
// eigenvalues come from LAPACK's symmetric eigensolver. The core has no LAPACK
// and could not copy its order of operations, so it forms the eigenvalues by
// Jacobi's rotations and is compared with NumPy's matrix to 1e-12 of its
// largest magnitude (ADR-017 G, the second kind).
//
// The rotations are one-sided: the matrix is factored by Cholesky's method with
// pivoting, A = G^T G, and the rows of G are rotated in pairs until they are
// orthogonal. Each is then an eigenvector times the square root of its
// eigenvalue. Every pass is along rows, no eigenvector is accumulated apart, and
// the eigenvalue of a null vector is the square of what rounding leaves of a
// row, well below the cutoff's 1e-15 of the largest.

#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace cfdcore::dense {

// Whether the rotations made the rows orthogonal within the sweeps allowed. No
// finite semi-definite matrix has been found that they do not.
enum class eigen_status { ok, not_converged };

// The eigenproblem and the pseudo-inverse of n by n symmetric positive
// semi-definite matrices, with the scratch of both allocated at construction.
// A matrix semi-definite to rounding alone is taken as the semi-definite one
// beside it: what Cholesky's method leaves below the rounding of the largest
// diagonal is dropped.
class semidefinite {
public:
    explicit semidefinite(std::size_t n);

    std::size_t size() const { return n_; }

    // Factors a matrix and rotates the factor's rows orthogonal.
    //
    // matrix: n * n values, row-major; the lower triangle is read and the upper
    //         one taken as its mirror, as `np.linalg.eigh` reads it
    //
    // Throws std::invalid_argument for an array of another length.
    eigen_status decompose(std::span<const double> matrix);

    // The n eigenvalues of the last matrix, in no order; after `not_converged`
    // they mean nothing.
    std::span<const double> eigenvalues() const { return values_; }

    // The pseudo-inverse of a matrix, by NumPy's rule: an eigenvalue above
    // `rcond` times the largest is inverted, the rest are dropped.
    //
    // matrix: as `decompose` reads it
    // rcond:  the cutoff relative to the largest (`pinv`'s default is 1e-15). Below
    //         the machine's epsilon it is as epsilon, where the factor stops
    // out:    output, n * n values, row-major, symmetric bit for bit; may be
    //         `matrix` itself. Not written after `not_converged`
    //
    // A matrix with a value that is not finite has a pseudo-inverse of NaNs, so that
    // a field that has diverged stays one. This is the core's own rule and not
    // parity: NumPy's result there depends on the LAPACK it is built with (NaNs
    // under 2.4.4, zeros under 2.5.5, and in some cases it raises).
    //
    // Throws std::invalid_argument for an array of another length.
    eigen_status pseudo_inverse(std::span<const double> matrix, double rcond,
                                std::span<double> out);

private:
    void factorise(std::span<const double> matrix);

    std::size_t n_;
    std::size_t rank_ = 0;
    std::vector<double> work_;    // the matrix less the pivots taken from it
    std::vector<double> factor_;  // G, rank_ rows of n
    std::vector<double> values_;
    std::vector<std::size_t> order_;  // the pivots in their order, then the rest
};

}  // namespace cfdcore::dense
