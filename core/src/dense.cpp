#include "cfdcore/dense.hpp"

#include "cfdcore/lanes.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace cfdcore::dense {

namespace {

// The sweeps allowed. Convergence is quadratic: a 64 by 64 matrix takes about
// six, so a matrix that has not converged by here never will.
constexpr int max_sweeps = 60;

// Two rows are taken as orthogonal at this cosine: a few roundings of the dot
// product that measures it, which a further rotation would only redraw.
constexpr double orthogonal = 0x1.0p-50;

// a . b over n entries, in eight running sums.
double dot(const double* a, const double* b, std::size_t n) {
    std::array<double, lanes::width> sums{};
    std::size_t i = 0;
    for (; i + lanes::width <= n; i += lanes::width) {
        for (std::size_t k = 0; k < lanes::width; ++k) {
            sums[k] = sums[k] + a[i + k] * b[i + k];
        }
    }
    for (std::size_t k = 0; i + k < n; ++k) {
        sums[k] = sums[k] + a[i + k] * b[i + k];
    }
    return ((sums[0] + sums[1]) + (sums[2] + sums[3])) +
           ((sums[4] + sums[5]) + (sums[6] + sums[7]));
}

// The rotation of two rows over N entries: x <- x - s (y + tau x) and
// y <- y + s (x - tau y), both formed from the rows as they were.
template <std::size_t N>
inline void rotate_cells(lanes::cells<N>, double s, double tau, std::size_t first,
                         double* CFDCORE_RESTRICT x, double* CFDCORE_RESTRICT y) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double xi = x[i];
        const double yi = y[i];
        x[i] = xi - s * (yi + tau * xi);
        y[i] = yi + s * (xi - tau * yi);
    }
}

// out <- out + scale x over N entries.
template <std::size_t N>
inline void add_scaled_cells(lanes::cells<N>, double scale, const double* x, std::size_t first,
                             double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        out[i] = out[i] + scale * x[i];
    }
}

void add_scaled(double scale, const double* x, std::size_t n, double* out) {
    lanes::for_each_block(0, n, [&](auto cells, std::size_t m) {
        add_scaled_cells(cells, scale, x, m, out);
    });
}

void require_square(std::size_t got, std::size_t n) {
    if (got != n * n) {
        throw std::invalid_argument("semidefinite: the matrix is not n by n");
    }
}

}  // namespace

semidefinite::semidefinite(std::size_t n)
    : n_(n), work_(n * n, 0.0), factor_(n * n, 0.0), values_(n, 0.0), order_(n, 0) {}

// Cholesky with the largest remaining diagonal as each step's pivot:
// A = sum over k of factor_k^T factor_k, with factor_k the rows of `factor_`, for
// as long as the pivots stand clear of the rounding of the largest diagonal. With m
// rows left where it stops, the m smallest eigenvalues are at most what is left's,
// and so at most m epsilon times the largest diagonal: with up to four rows left
// none of them is above the pseudo-inverse's cutoff (1e-15 of the largest). With
// more, an eigenvalue of up to 64 epsilon could be dropped that NumPy keeps, though
// only where the rest of the m are smaller by as much; no system planted or captured
// has one (docs/reports/phase6_pressure.md, T6).
void semidefinite::factorise(std::span<const double> matrix) {
    const std::size_t n = n_;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            work_[i * n + j] = matrix[i * n + j];
            work_[j * n + i] = matrix[i * n + j];
        }
        order_[i] = i;
    }
    rank_ = 0;
    double smallest = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        // The pivot among the rows not taken yet; `order_` holds them last.
        std::size_t at = k;
        for (std::size_t m = k + 1; m < n; ++m) {
            if (work_[order_[m] * (n + 1)] > work_[order_[at] * (n + 1)]) {
                at = m;
            }
        }
        std::swap(order_[k], order_[at]);
        const std::size_t p = order_[k];
        const double pivot = work_[p * (n + 1)];
        if (k == 0) {
            smallest = std::numeric_limits<double>::epsilon() * pivot;
        }
        if (pivot <= smallest) {
            return;
        }
        const double root = std::sqrt(pivot);
        double* row = factor_.data() + k * n;
        for (std::size_t j = 0; j < n; ++j) {
            row[j] = work_[p * n + j] / root;
        }
        ++rank_;
        for (std::size_t m = k + 1; m < n; ++m) {
            const std::size_t i = order_[m];
            if (row[i] != 0.0) {
                add_scaled(-row[i], row, n, work_.data() + i * n);
            }
        }
        // The row and the column of the pivot are done: zero, so that the
        // factor's later rows are exactly zero there.
        for (std::size_t j = 0; j < n; ++j) {
            work_[j * n + p] = 0.0;
        }
    }
}

eigen_status semidefinite::decompose(std::span<const double> matrix) {
    require_square(matrix.size(), n_);
    const std::size_t n = n_;
    factorise(matrix);
    std::fill(values_.begin(), values_.end(), 0.0);
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        bool rotated = false;
        for (std::size_t k = 0; k < rank_; ++k) {
            const double* row = factor_.data() + k * n;
            values_[k] = dot(row, row, n);
        }
        for (std::size_t p = 0; p + 1 < rank_; ++p) {
            double* x = factor_.data() + p * n;
            for (std::size_t q = p + 1; q < rank_; ++q) {
                double* y = factor_.data() + q * n;
                const double entry = dot(x, y, n);
                // Not `>`: a NaN rotates, so `decompose` of one runs out of sweeps.
                if (entry * entry <= orthogonal * orthogonal * values_[p] * values_[q]) {
                    continue;
                }
                rotated = true;
                // t = sign(theta) / (|theta| + sqrt(1 + theta^2)) with theta the
                // difference over twice the entry, the fraction multiplied out.
                const double difference = values_[q] - values_[p];
                const double twice = 2.0 * entry;
                const double root =
                    std::fabs(difference) + std::sqrt(difference * difference + twice * twice);
                const double t = (difference < 0.0 ? -twice : twice) / root;
                const double c = 1.0 / std::sqrt(1.0 + t * t);
                const double s = t * c;
                const double tau = s / (1.0 + c);
                values_[p] -= t * entry;
                values_[q] += t * entry;
                lanes::for_each_block(0, n, [&](auto cells, std::size_t m) {
                    rotate_cells(cells, s, tau, m, x, y);
                });
            }
        }
        if (!rotated) {
            return eigen_status::ok;
        }
    }
    return eigen_status::not_converged;
}

eigen_status semidefinite::pseudo_inverse(std::span<const double> matrix, double rcond,
                                          std::span<double> out) {
    const std::size_t n = n_;
    require_square(matrix.size(), n);
    require_square(out.size(), n);
    bool finite = true;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            finite = finite && std::isfinite(matrix[i * n + j]);
        }
    }
    if (!finite) {
        std::fill(out.begin(), out.end(), std::numeric_limits<double>::quiet_NaN());
        return eigen_status::ok;
    }
    if (decompose(matrix) != eigen_status::ok) {
        return eigen_status::not_converged;
    }
    double largest = 0.0;
    for (const double value : values_) {
        largest = std::max(largest, value);
    }
    const double cutoff = rcond * largest;
    std::fill(out.begin(), out.end(), 0.0);
    for (std::size_t k = 0; k < rank_; ++k) {
        if (!(values_[k] > cutoff)) {
            continue;
        }
        // A row is the eigenvector times the square root of its eigenvalue.
        const double* row = factor_.data() + k * n;
        for (std::size_t i = 0; i < n; ++i) {
            const double scale = row[i] / values_[k] / values_[k];
            add_scaled(scale, row, i + 1, out.data() + i * n);
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            out[j * n + i] = out[i * n + j];
        }
    }
    return eigen_status::ok;
}

}  // namespace cfdcore::dense
