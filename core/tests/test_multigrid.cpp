// The multigrid module's own tests (ADR-017 D): results known without the
// reference. The comparison with multigrid.py is tests/test_core_pressure.py's.

#include "cfdcore/multigrid.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using cfdcore::layout::layout;
using cfdcore::multigrid::build_status;
using cfdcore::multigrid::hierarchy;
using cfdcore::multigrid::keys;
using cfdcore::multigrid::system;

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

keys default_keys() {
    keys k;
    k.coarsest_cells = 4;
    k.sweeps = 1;
    k.weight = 2.0 / 3.0;
    k.dominance_tolerance = 1e-12;
    k.rcond = rcond;
    return k;
}

// A symmetric system on a grid: random couplings across the faces between open
// cells, every eleventh cell SOLID, and with `open` a coupling of the corner's
// first cells to a face held at zero.
template <int D>
void fill(system<D>& s, std::size_t seed, bool open) {
    const auto& fields = *s.fields;
    const std::size_t n = fields.cell_count();
    const auto solid = [&](std::size_t i) { return n > 20 && hashed(seed + 7 * i) < 1.0 / 11.0; };
    for (int d = 0; d < D; ++d) {
        const std::size_t stride = fields.cell_stride(d);
        for (std::size_t i = 0; i < n; ++i) {
            if ((i / stride) % fields.cells(d) + 1 < fields.cells(d) && !solid(i) &&
                !solid(i + stride)) {
                const double c = 0.2 + hashed(seed + 1000 * static_cast<std::size_t>(d + 1) + i);
                s.high[d][i] = c;
                s.low[d][i + stride] = c;
            }
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int d = 0; d < D; ++d) {
            sum += s.high[d][i] + s.low[d][i];
        }
        s.centre[i] = sum;
    }
    if (open) {
        for (std::size_t i = 0; i < std::min<std::size_t>(3, n); ++i) {
            s.centre[i] += s.centre[i] > 0.0 ? 0.7 : 0.0;
        }
    }
}

// The system as a dense matrix.
template <int D>
std::vector<double> dense_of(const system<D>& s) {
    const auto& fields = *s.fields;
    const std::size_t n = fields.cell_count();
    std::vector<double> a(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        a[i * n + i] = s.centre[i];
        for (int d = 0; d < D; ++d) {
            const std::size_t stride = fields.cell_stride(d);
            const std::size_t at = (i / stride) % fields.cells(d);
            if (at + 1 < fields.cells(d)) {
                a[i * n + i + stride] -= s.high[d][i];
            }
            if (at > 0) {
                a[i * n + i - stride] -= s.low[d][i];
            }
        }
    }
    return a;
}

// The index of a fine cell's group in the coarse layout.
template <int D>
std::size_t group_index(const layout<D>& fine, const layout<D>& coarse, std::size_t cell) {
    std::size_t index = 0;
    for (int d = 0; d < D; ++d) {
        index += (cell / fine.cell_stride(d)) % fine.cells(d) / 2 * coarse.cell_stride(d);
    }
    return index;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

std::vector<double> field(std::size_t n, std::size_t seed) {
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = hashed(seed + i) - 0.5;
    }
    return out;
}

void expect_matrix(const std::vector<double>& got, const std::vector<double>& expected,
                   double tolerance) {
    ASSERT_EQ(got.size(), expected.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], expected[i], tolerance) << i;
    }
}

// ---------------------------------------------------------------------------
// The operator and the groups

TEST(MultigridOperator, OfAHandSystem) {
    // Two by two: x = (1, 2; 3, 4), couplings 1 and 2 along x, 3 and 4 along y.
    layout<2> fields({2, 2});
    system<2> s(fields);
    s.centre = {5.0, 6.0, 7.0, 8.0};
    s.high[0] = {1.0, 0.0, 2.0, 0.0};
    s.low[0] = {0.0, 1.0, 0.0, 2.0};
    s.high[1] = {3.0, 4.0, 0.0, 0.0};
    s.low[1] = {0.0, 0.0, 3.0, 4.0};
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> zeros(2, 0.0);
    std::vector<double> out(4);
    cfdcore::multigrid::apply_operator<2>(s, zeros, x, out);
    EXPECT_EQ(out, (std::vector<double>{5.0 - 2.0 - 9.0, 12.0 - 1.0 - 16.0, 21.0 - 8.0 - 3.0,
                                        32.0 - 6.0 - 8.0}));
    const std::vector<double> r{1.0, 1.0, 1.0, 1.0};
    const std::vector<double> half(4, 0.5);
    std::vector<double> next(4);
    cfdcore::multigrid::smooth<2>(s, zeros, half, x, r, next);
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(next[i], x[i] + 0.5 * (1.0 - out[i]));
    }
}

template <int D>
void expect_groups(const std::array<std::size_t, D>& cells) {
    const layout<D> fine(cells);
    const layout<D> coarse = cfdcore::multigrid::coarser(fine);
    for (int d = 0; d < D; ++d) {
        EXPECT_EQ(coarse.cells(d), (cells[d] + 1) / 2);
    }
    const std::vector<double> r = field(fine.cell_count(), 1);
    const std::vector<double> z = field(coarse.cell_count(), 2);
    std::vector<double> sums(coarse.cell_count(), nan_value);
    cfdcore::multigrid::restrict_to<D>(fine, coarse, r, sums);
    std::vector<double> expected(coarse.cell_count(), 0.0);
    for (std::size_t i = 0; i < r.size(); ++i) {
        expected[group_index(fine, coarse, i)] += r[i];
    }
    for (std::size_t i = 0; i < sums.size(); ++i) {
        EXPECT_NEAR(sums[i], expected[i], 1e-15);
    }
    std::vector<double> x = r;
    cfdcore::multigrid::add_prolonged<D>(fine, coarse, z, x);
    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_EQ(bits(x[i]), bits(r[i] + z[group_index(fine, coarse, i)]));
    }
    // P^T and P are each other's transposes.
    std::vector<double> pz(fine.cell_count(), 0.0);
    cfdcore::multigrid::add_prolonged<D>(fine, coarse, z, pz);
    EXPECT_NEAR(dot(sums, z), dot(r, pz), 1e-14);
}

TEST(MultigridGroups, RestrictionAndProlongationOverEveryShape) {
    for (std::size_t nx = 1; nx < 20; ++nx) {
        expect_groups<1>({nx});
        for (const std::size_t ny : {1u, 2u, 3u, 6u, 7u}) {
            expect_groups<2>({nx, ny});
        }
    }
}

// The Galerkin operator is P^T A P, formed densely.
template <int D>
void expect_galerkin(const std::array<std::size_t, D>& cells, bool open) {
    const layout<D> fields(cells);
    const layout<D> coarse = cfdcore::multigrid::coarser(fields);
    system<D> s(fields);
    fill(s, 31 + cells[0], open);
    system<D> out(coarse);
    const std::vector<double> zeros(cells[0], 0.0);
    cfdcore::multigrid::coarsen<D>(s, zeros, out);
    const std::size_t n = fields.cell_count();
    const std::size_t m = coarse.cell_count();
    const std::vector<double> a = dense_of(s);
    std::vector<double> expected(m * m, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            expected[group_index(fields, coarse, i) * m + group_index(fields, coarse, j)] +=
                a[i * n + j];
        }
    }
    expect_matrix(dense_of(out), expected, 1e-13);
    // The domain's edges have no coupling on any level.
    for (int d = 0; d < D; ++d) {
        for (std::size_t i = 0; i < m; ++i) {
            const std::size_t at = (i / coarse.cell_stride(d)) % coarse.cells(d);
            if (at == 0) {
                EXPECT_EQ(out.low[d][i], 0.0);
            }
            if (at + 1 == coarse.cells(d)) {
                EXPECT_EQ(out.high[d][i], 0.0);
            }
        }
    }
}

TEST(MultigridGalerkin, IsTheTripleProductFormedDensely) {
    for (const bool open : {true, false}) {
        for (const std::size_t nx : {1u, 2u, 3u, 5u, 8u, 13u, 17u, 18u, 35u}) {
            expect_galerkin<1>({nx}, open);
            for (const std::size_t ny : {1u, 2u, 3u, 4u, 7u}) {
                expect_galerkin<2>({nx, ny}, open);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The cycle

template <int D>
void expect_symmetric_cycle(const std::array<std::size_t, D>& cells, bool open,
                            std::int64_t sweeps) {
    const layout<D> fields(cells);
    const std::size_t n = fields.cell_count();
    system<D> s(fields);
    fill(s, 5 + cells[0], open);
    keys k = default_keys();
    k.sweeps = sweeps;
    hierarchy<D> h(fields, k);
    ASSERT_EQ(h.build(s), build_status::ok);
    std::vector<double> r1 = field(n, 11);
    std::vector<double> r2 = field(n, 12);
    for (std::size_t i = 0; i < n; ++i) {
        if (!(s.centre[i] > 0.0)) {
            r1[i] = r2[i] = 0.0;
        }
    }
    std::vector<double> z1(n, nan_value);
    std::vector<double> z2(n, nan_value);
    h.cycle(r1, z1);
    h.cycle(r2, z2);
    const double scale = std::sqrt(dot(r1, r1) * dot(z2, z2));
    EXPECT_NEAR(dot(z1, r2), dot(r1, z2), 1e-14 * scale + 1e-300);
    EXPECT_GE(dot(r1, z1), 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        if (!(s.centre[i] > 0.0)) {
            EXPECT_EQ(bits(z1[i]), bits(0.0));
        }
    }
    // The cycle keeps nothing of the one before it.
    std::vector<double> again(n, nan_value);
    h.cycle(r1, again);
    EXPECT_EQ(again, z1);
    ASSERT_EQ(h.build(s), build_status::ok);
    h.cycle(r1, again);
    EXPECT_EQ(again, z1);
}

TEST(MultigridCycle, IsASymmetricPositiveOperator) {
    for (const bool open : {true, false}) {
        for (const std::int64_t sweeps : {1, 2, 3}) {
            for (const std::size_t nx : {2u, 3u, 9u, 16u, 21u}) {
                expect_symmetric_cycle<1>({nx}, open, sweeps);
                for (const std::size_t ny : {1u, 2u, 5u, 8u}) {
                    expect_symmetric_cycle<2>({nx, ny}, open, sweeps);
                }
            }
        }
    }
}

// One level: the cycle is the pseudo-inverse.
TEST(MultigridCycle, OfASmallSystemIsItsExactSolution) {
    const layout<2> fields({2, 2});
    system<2> s(fields);
    fill(s, 9, true);
    hierarchy<2> h(fields, default_keys());
    ASSERT_EQ(h.levels(), 1u);
    ASSERT_EQ(h.build(s), build_status::ok);
    const std::vector<double> r{1.0, -2.0, 0.5, 3.0};
    std::vector<double> z(4);
    std::vector<double> back(4);
    h.cycle(r, z);
    cfdcore::multigrid::apply_operator<2>(s, std::vector<double>(2, 0.0), z, back);
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(back[i], r[i], 1e-13);
    }
}

// Two levels on a line of cells, every step by hand: the sweep from zero, the
// group sums of the residual, the two-cell solve, its correction and the sweep.
TEST(MultigridCycle, OfTwoLevelsIsItsStepsByHand) {
    const layout<1> fields({4});
    system<1> s(fields);
    s.centre = {3.0, 2.0, 2.0, 1.0};  // the first cell has an outlet's 2 besides
    s.high[0] = {1.0, 1.0, 1.0, 0.0};
    s.low[0] = {0.0, 1.0, 1.0, 1.0};
    keys k = default_keys();
    k.coarsest_cells = 2;
    k.weight = 0.5;
    hierarchy<1> h(fields, k);
    ASSERT_EQ(h.levels(), 2u);
    ASSERT_EQ(h.build(s), build_status::ok);
    // The groups: diagonals 3 + 2 - 2 = 3 and 2 + 1 - 2 = 1, coupling 1.
    EXPECT_EQ(h.level_system(1).centre, (std::vector<double>{3.0, 1.0}));
    EXPECT_EQ(h.level_system(1).high[0], (std::vector<double>{1.0, 0.0}));
    EXPECT_EQ(h.level_system(1).low[0], (std::vector<double>{0.0, 1.0}));
    const std::vector<double> r{6.0, 4.0, -4.0, 2.0};
    const auto apply = [](const std::vector<double>& x) {
        return std::vector<double>{3.0 * x[0] - x[1], 2.0 * x[1] - x[2] - x[0],
                                   2.0 * x[2] - x[3] - x[1], x[3] - x[2]};
    };
    std::vector<double> x{1.0, 1.0, -1.0, 1.0};  // half the residual over the diagonal
    std::vector<double> ax = apply(x);
    const double r0 = (r[0] - ax[0]) + (r[1] - ax[1]);
    const double r1 = (r[2] - ax[2]) + (r[3] - ax[3]);
    // (3 -1; -1 1) has the inverse (1 1; 1 3) / 2.
    const double z0 = (r0 + r1) / 2.0;
    const double z1 = (r0 + 3.0 * r1) / 2.0;
    x = {x[0] + z0, x[1] + z0, x[2] + z1, x[3] + z1};
    ax = apply(x);
    std::vector<double> got(4);
    h.cycle(r, got);
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(got[i], x[i] + 0.5 / s.centre[i] * (r[i] - ax[i]), 1e-14) << i;
    }
}

TEST(MultigridHierarchy, StopsAtTheCellsAskedOrAtASideOfOne) {
    keys k = default_keys();
    k.coarsest_cells = 64;
    const auto levels = [&](std::size_t nx, std::size_t ny) {
        const layout<2> fields({nx, ny});
        return hierarchy<2>(fields, k).levels();
    };
    EXPECT_EQ(levels(8, 8), 1u);
    EXPECT_EQ(levels(9, 8), 2u);
    EXPECT_EQ(levels(80, 30), 4u);    // 80x30, 40x15, 20x8, 10x4
    EXPECT_EQ(levels(320, 120), 6u);  // and 160x60 before them
    EXPECT_EQ(levels(1000, 1), 1u);   // a side of one, whatever the cells
    EXPECT_EQ(levels(1000, 2), 2u);
    EXPECT_EQ(levels(1000, 3), 3u);
}

TEST(MultigridHierarchy, RefusesASystemThatIsNotWeaklyDominant) {
    for (const std::size_t nx : {2u, 12u}) {  // one level, and several
        const layout<2> fields({nx, 5});
        system<2> s(fields);
        fill(s, 2, false);
        keys k = default_keys();
        k.coarsest_cells = 10;
        hierarchy<2> h(fields, k);
        EXPECT_EQ(h.levels() == 1, nx == 2);
        ASSERT_EQ(h.build(s), build_status::ok);
        for (const std::size_t at : {std::size_t{0}, fields.cell_count() - 1, nx + 1}) {
            if (!(s.centre[at] > 0.0)) {
                continue;
            }
            system<2> bad = s;
            bad.centre[at] *= 1.0 - 1e-9;
            EXPECT_EQ(h.build(bad), build_status::not_dominant) << at;
            std::vector<double> r(fields.cell_count(), 1.0);
            std::vector<double> z(fields.cell_count());
            EXPECT_THROW(h.cycle(r, z), std::invalid_argument);
            // Rounding's shortfall is allowed.
            bad.centre[at] = s.centre[at] * (1.0 - 1e-14);
            EXPECT_EQ(h.build(bad), build_status::ok) << at;
        }
    }
}

TEST(MultigridHierarchy, ProgrammingErrorsThrow) {
    const layout<2> fields({6, 5});
    const layout<2> other({5, 6});
    const auto spoilt = [](auto&& change) {
        keys k = default_keys();
        change(k);
        return k;
    };
    const double inf = std::numeric_limits<double>::infinity();
    for (const keys& k :
         {spoilt([](keys& x) { x.coarsest_cells = 0; }), spoilt([](keys& x) { x.sweeps = 0; }),
          spoilt([](keys& x) { x.weight = 0.0; }), spoilt([&](keys& x) { x.weight = inf; }),
          spoilt([](keys& x) { x.weight = nan_value; }),
          spoilt([](keys& x) { x.dominance_tolerance = -1e-12; }),
          spoilt([&](keys& x) { x.dominance_tolerance = inf; }),
          spoilt([](keys& x) { x.rcond = -1.0; }), spoilt([](keys& x) { x.rcond = nan_value; })}) {
        EXPECT_THROW(hierarchy<2>(fields, k), std::invalid_argument);
    }
    hierarchy<2> h(fields, default_keys());
    std::vector<double> r(30, 0.0);
    std::vector<double> z(30, 0.0);
    std::vector<double> wrong(29, 0.0);
    EXPECT_THROW(h.cycle(r, z), std::invalid_argument);  // before a build
    const system<2> elsewhere(other);
    EXPECT_THROW((void)h.build(elsewhere), std::invalid_argument);
    system<2> s(fields);
    fill(s, 1, true);
    ASSERT_EQ(h.build(s), build_status::ok);
    EXPECT_THROW(h.cycle(wrong, z), std::invalid_argument);
    EXPECT_THROW(h.cycle(r, wrong), std::invalid_argument);

    const std::vector<double> zeros(6, 0.0);
    const std::vector<double> short_zeros(5, 0.0);
    using namespace cfdcore::multigrid;
    EXPECT_THROW(apply_operator<2>(s, short_zeros, r, z), std::invalid_argument);
    EXPECT_THROW(apply_operator<2>(s, zeros, wrong, z), std::invalid_argument);
    EXPECT_THROW(apply_operator<2>(s, zeros, r, wrong), std::invalid_argument);
    EXPECT_THROW(smooth<2>(s, short_zeros, r, r, r, z), std::invalid_argument);
    EXPECT_THROW(smooth<2>(s, zeros, wrong, r, r, z), std::invalid_argument);
    EXPECT_THROW(smooth<2>(s, zeros, r, wrong, r, z), std::invalid_argument);
    EXPECT_THROW(smooth<2>(s, zeros, r, r, wrong, z), std::invalid_argument);
    EXPECT_THROW(smooth<2>(s, zeros, r, r, r, wrong), std::invalid_argument);
    const layout<2> coarse = coarser(fields);
    std::vector<double> c(9, 0.0);
    EXPECT_THROW(restrict_to<2>(fields, coarse, wrong, c), std::invalid_argument);
    EXPECT_THROW(restrict_to<2>(fields, coarse, r, wrong), std::invalid_argument);
    EXPECT_THROW(add_prolonged<2>(fields, coarse, wrong, z), std::invalid_argument);
    EXPECT_THROW(add_prolonged<2>(fields, coarse, c, wrong), std::invalid_argument);
    system<2> to(coarse);
    system<2> not_coarse(fields);
    EXPECT_THROW(coarsen<2>(s, short_zeros, to), std::invalid_argument);
    EXPECT_THROW(coarsen<2>(s, zeros, not_coarse), std::invalid_argument);
}

}  // namespace
