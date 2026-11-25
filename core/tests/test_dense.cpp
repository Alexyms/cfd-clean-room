// The dense module's own tests (ADR-017 D): pseudo-inverses known without NumPy.
// The comparison with NumPy's and with the 60-digit one is
// tests/test_core_pressure.py's.

#include "cfdcore/dense.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using cfdcore::dense::eigen_status;
using cfdcore::dense::semidefinite;

constexpr double rcond = 1e-15;
constexpr double nan_value = std::numeric_limits<double>::quiet_NaN();

std::uint64_t bits(double x) { return std::bit_cast<std::uint64_t>(x); }

// A deterministic value in [0, 1) for index i.
double hashed(std::size_t i) {
    std::uint64_t z = 0x9e3779b97f4a7c15ULL * (i + 1);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z ^= z >> 31;
    return static_cast<double>(z >> 11) * 0x1.0p-53;
}

// The dense matrix of a five-point system on nx by ny cells: random couplings
// across the faces between open cells, about one cell in eleven SOLID on a grid of
// more than twenty, and with `open` a term added to the first cells' diagonals.
std::vector<double> five_point(std::size_t nx, std::size_t ny, std::size_t seed, bool open) {
    const std::size_t n = nx * ny;
    std::vector<double> a(n * n, 0.0);
    const auto solid = [&](std::size_t i) { return n > 20 && hashed(seed + 7 * i) < 1.0 / 11.0; };
    const auto couple = [&](std::size_t i, std::size_t j, double c) {
        if (!solid(i) && !solid(j)) {
            a[i * n + j] -= c;
            a[j * n + i] -= c;
            a[i * n + i] += c;
            a[j * n + j] += c;
        }
    };
    for (std::size_t i = 0; i < n; ++i) {
        if (i % nx + 1 < nx) {
            couple(i, i + 1, 0.2 + hashed(seed + 1000 + i));
        }
        if (i + nx < n) {
            couple(i, i + nx, 0.2 + hashed(seed + 2000 + i));
        }
    }
    for (std::size_t i = 0; open && i < std::min<std::size_t>(3, n); ++i) {
        a[i * n + i] += a[i * n + i] > 0.0 ? 0.7 : 0.0;
    }
    return a;
}

void expect_matrix(const std::vector<double>& got, const std::vector<double>& expected,
                   double tolerance) {
    ASSERT_EQ(got.size(), expected.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], expected[i], tolerance) << i;
    }
}

TEST(PseudoInverse, OfAnInvertibleMatrixIsItsInverse) {
    semidefinite eigen(2);
    std::vector<double> out(4);
    ASSERT_EQ(eigen.pseudo_inverse(std::vector<double>{2.0, -1.0, -1.0, 2.0}, rcond, out),
              eigen_status::ok);
    expect_matrix(out, {2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0}, 1e-15);
    std::vector<double> values(eigen.eigenvalues().begin(), eigen.eigenvalues().end());
    std::sort(values.begin(), values.end());
    EXPECT_NEAR(values[0], 1.0, 1e-15);
    EXPECT_NEAR(values[1], 3.0, 1e-15);
}

TEST(PseudoInverse, OfASingularMatrixDropsItsNullVector) {
    // Two cells joined by one coupling, and three in a row: the minimum-norm
    // solutions, by hand.
    semidefinite two(2);
    std::vector<double> out(4);
    ASSERT_EQ(two.pseudo_inverse(std::vector<double>{1.0, -1.0, -1.0, 1.0}, rcond, out),
              eigen_status::ok);
    expect_matrix(out, {0.25, -0.25, -0.25, 0.25}, 1e-15);
    semidefinite three(3);
    out.assign(9, 0.0);
    const std::vector<double> path{1.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 1.0};
    ASSERT_EQ(three.pseudo_inverse(path, rcond, out), eigen_status::ok);
    expect_matrix(out,
                  {5.0 / 9.0, -1.0 / 9.0, -4.0 / 9.0, -1.0 / 9.0, 2.0 / 9.0, -1.0 / 9.0,
                   -4.0 / 9.0, -1.0 / 9.0, 5.0 / 9.0},
                  1e-15);
}

TEST(PseudoInverse, ZeroRowsStayZeroAndAZeroMatrixIsItsOwn) {
    semidefinite eigen(3);
    std::vector<double> out(9, 1.0);
    ASSERT_EQ(eigen.pseudo_inverse(
                  std::vector<double>{2.0, 0.0, -2.0, 0.0, 0.0, 0.0, -2.0, 0.0, 4.0}, rcond, out),
              eigen_status::ok);
    expect_matrix(out, {1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5}, 1e-15);
    for (const std::size_t zero : {1u, 3u, 4u, 5u, 7u}) {
        EXPECT_EQ(out[zero], 0.0);
    }
    out.assign(9, 1.0);
    ASSERT_EQ(eigen.pseudo_inverse(std::vector<double>(9, 0.0), rcond, out), eigen_status::ok);
    EXPECT_EQ(out, std::vector<double>(9, 0.0));
}

TEST(PseudoInverse, TheCutoffIsAgainstTheLargestAndExclusive) {
    // diag(8, small): the small one is kept above 1e-15 of 8 and dropped at it
    // and below; what is kept is inverted whatever its size.
    semidefinite eigen(2);
    std::vector<double> out(4);
    for (const double small : {1.0e-14, 8.1e-15}) {
        ASSERT_EQ(eigen.pseudo_inverse(std::vector<double>{8.0, 0.0, 0.0, small}, rcond, out),
                  eigen_status::ok);
        EXPECT_NEAR(out[3] * small, 1.0, 1e-14) << small;
        EXPECT_NEAR(out[0], 0.125, 1e-16);
    }
    for (const double small : {0x1.0p-37, 7.9e-15, 2.0e-15, 1.0e-18}) {
        // With a cutoff that is a power of two the smallest lands on it exactly.
        const double cut = small == 0x1.0p-37 ? 0x1.0p-40 : rcond;
        ASSERT_EQ(eigen.pseudo_inverse(std::vector<double>{8.0, 0.0, 0.0, small}, cut, out),
                  eigen_status::ok);
        EXPECT_EQ(out[3], 0.0) << small;
        EXPECT_NEAR(out[0], 0.125, 1e-16);
    }
}

// `parts` paths of m cells beside one another, unit couplings, and a term of `corner`
// added to each one's first diagonal: not diagonal, each path's smallest eigenvalue
// about corner / m with its vector spread over the path's cells, and the inverse of
// each block known, 1 / corner + min(i, j).
struct paths {
    std::vector<double> matrix;
    double corner;  // the term the diagonal holds after its rounding
    double largest;  // the largest eigenvalue, to a few parts in a thousand
};

paths corner_paths(std::size_t parts, std::size_t m, double corner) {
    const std::size_t n = parts * m;
    paths out{std::vector<double>(n * n, 0.0), 0.0, 0.0};
    for (std::size_t part = 0; part < parts; ++part) {
        const std::size_t at = part * m;
        for (std::size_t i = 0; i + 1 < m; ++i) {
            out.matrix[(at + i) * n + at + i] += 1.0;
            out.matrix[(at + i + 1) * n + at + i + 1] += 1.0;
            out.matrix[(at + i) * n + at + i + 1] = -1.0;
            out.matrix[(at + i + 1) * n + at + i] = -1.0;
        }
        out.matrix[at * n + at] += corner;
    }
    out.corner = out.matrix[0] - 1.0;
    const double angle = 3.14159265358979323846 * static_cast<double>(m - 1) /
                         static_cast<double>(m);
    out.largest = 2.0 - 2.0 * std::cos(angle);
    return out;
}

// The pivot the factor stops at, away from a diagonal matrix: an eigenvalue either
// side of the cutoff is kept and dropped though its vector is spread over every cell
// and no diagonal of the matrix is small. With one path the last pivot is
// 1 / (A^-1) at the last cell, never below the eigenvalue; with several each
// path's is its own.
TEST(PseudoInverse, TheFactorsStopKeepsWhatTheCutoffKeepsOffTheDiagonal) {
    for (const std::size_t parts : {1u, 2u, 4u}) {
        for (const std::size_t m : {4u, 8u, 16u}) {
            const std::size_t n = parts * m;
            semidefinite eigen(n);
            std::vector<double> x(n * n);
            const double unit = 4.0 * static_cast<double>(m);  // corner over unit, about the ratio
            for (const double ratio : {1.6e-15, 3.0e-15, 1.0e-14, 1.0e-13, 1.0e-10}) {
                const paths p = corner_paths(parts, m, ratio * unit);
                ASSERT_GT(p.corner / static_cast<double>(m), 1.2 * rcond * p.largest);
                ASSERT_EQ(eigen.pseudo_inverse(p.matrix, rcond, x), eigen_status::ok);
                for (std::size_t part = 0; part < parts; ++part) {
                    // The entry is 1 / corner to the share of a rounding the corner is.
                    const std::size_t last = part * m + m - 1;
                    EXPECT_NEAR(x[last * n + last] * p.corner, 1.0, 0.3) << m << " " << ratio;
                }
            }
            for (const double ratio : {0.0, 1.0e-17, 2.0e-16}) {
                const paths p = corner_paths(parts, m, ratio * unit);
                ASSERT_LT(p.corner / static_cast<double>(m), 0.8 * rcond * p.largest);
                ASSERT_EQ(eigen.pseudo_inverse(p.matrix, rcond, x), eigen_status::ok);
                for (const double value : x) {
                    EXPECT_LT(std::fabs(value), static_cast<double>(m)) << m << " " << ratio;
                }
            }
        }
    }
}

TEST(PseudoInverse, ReadsTheLowerTriangleAndIsSymmetricBitForBit) {
    const std::size_t n = 9;
    std::vector<double> a = five_point(3, 3, 3, true);
    semidefinite eigen(n);
    std::vector<double> out(n * n);
    ASSERT_EQ(eigen.pseudo_inverse(a, rcond, out), eigen_status::ok);
    std::vector<double> upper_spoilt = a;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            upper_spoilt[i * n + j] = nan_value;
            EXPECT_EQ(bits(out[i * n + j]), bits(out[j * n + i]));
        }
    }
    std::vector<double> again(n * n);
    ASSERT_EQ(eigen.pseudo_inverse(upper_spoilt, rcond, again), eigen_status::ok);
    for (std::size_t i = 0; i < n * n; ++i) {
        EXPECT_EQ(bits(again[i]), bits(out[i]));
    }
    // The output may be the input.
    ASSERT_EQ(eigen.pseudo_inverse(a, rcond, a), eigen_status::ok);
    EXPECT_EQ(a, out);
}

// The four conditions of Moore and Penrose, A X A = A among them, on systems
// open and closed and as large as the coarsest level.
TEST(PseudoInverse, MeetsMooreAndPenrosesConditions) {
    for (const bool open : {true, false}) {
        for (const std::size_t nx : {1u, 2u, 5u, 8u}) {
            const std::size_t n = nx * 8;
            const std::vector<double> a = five_point(nx, 8, 17 * nx, open);
            semidefinite eigen(n);
            std::vector<double> x(n * n);
            ASSERT_EQ(eigen.pseudo_inverse(a, rcond, x), eigen_status::ok);
            const auto times = [n](const std::vector<double>& p, const std::vector<double>& q) {
                std::vector<double> out(n * n, 0.0);
                for (std::size_t i = 0; i < n; ++i) {
                    for (std::size_t k = 0; k < n; ++k) {
                        for (std::size_t j = 0; j < n; ++j) {
                            out[i * n + j] += p[i * n + k] * q[k * n + j];
                        }
                    }
                }
                return out;
            };
            const std::vector<double> ax = times(a, x);
            const double largest = *std::max_element(x.begin(), x.end());
            expect_matrix(times(ax, a), a, 1e-12);
            expect_matrix(times(x, ax), x, 1e-12 * largest);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < i; ++j) {
                    EXPECT_NEAR(ax[i * n + j], ax[j * n + i], 1e-12);
                }
            }
        }
    }
}

TEST(PseudoInverse, AValueThatIsNotFiniteGivesNaNs) {
    semidefinite eigen(2);
    for (const double bad : {nan_value, std::numeric_limits<double>::infinity()}) {
        for (const std::size_t at : {0u, 2u, 3u}) {
            std::vector<double> a{2.0, -1.0, -1.0, 2.0};
            a[at] = bad;
            std::vector<double> out(4, 1.0);
            ASSERT_EQ(eigen.pseudo_inverse(a, rcond, out), eigen_status::ok);
            for (const double value : out) {
                EXPECT_TRUE(std::isnan(value));
            }
        }
    }
    EXPECT_EQ(semidefinite(2).decompose(std::vector<double>{nan_value, 1.0, 1.0, nan_value}),
              eigen_status::not_converged);
}

TEST(PseudoInverse, ProgrammingErrorsThrow) {
    semidefinite eigen(2);
    std::vector<double> four(4, 0.0);
    std::vector<double> three(3, 0.0);
    EXPECT_THROW((void)eigen.decompose(three), std::invalid_argument);
    EXPECT_THROW((void)eigen.pseudo_inverse(three, rcond, four), std::invalid_argument);
    EXPECT_THROW((void)eigen.pseudo_inverse(four, rcond, three), std::invalid_argument);
}

}  // namespace
