// The momentum module's own tests (ADR-017 D): results known without the
// reference. The comparison with momentum.py is tests/test_core_momentum.py's.

#include "cfdcore/grid.hpp"
#include "cfdcore/momentum.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using cfdcore::momentum::edge_condition;
using cfdcore::momentum::predictor;
using cfdcore::momentum::predictor_input;
using cfdcore::momentum::viscosity_status;

constexpr std::int32_t fluid_type = 0;
constexpr std::int32_t solid_type = 1;

std::uint64_t bits(double x) { return std::bit_cast<std::uint64_t>(x); }

// A deterministic value in [0, 1) for index i, so a planted field needs no
// generator.
double hashed(std::size_t i) {
    std::uint64_t z = 0x9e3779b97f4a7c15ULL * (i + 1);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z ^= z >> 31;
    return static_cast<double>(z >> 11) * 0x1.0p-53;
}

// A bool array that a span can view (std::vector<bool> holds bits), as large
// as the largest these tests need.
struct flags {
    explicit flags(std::size_t n, bool value = false) : size(n) {
        if (n > values.size()) {
            throw std::length_error("flags holds 256 at most");
        }
        values.fill(value);
    }
    std::span<const bool> span() const { return std::span<const bool>(values).first(size); }
    bool& operator[](std::size_t i) { return values[i]; }
    std::array<bool, 256> values;
    std::size_t size;
};

// One edge's condition, its arrays held.
struct edge {
    edge(std::size_t n, double distance) : known(n, true), value(n, 0.0), wall_distance(distance) {}
    edge_condition condition() const { return {known.span(), value, wall_distance}; }
    flags known;
    std::vector<double> value;
    double wall_distance;
};

// A box of nx by ny cells of width h with walls at rest, every array a
// predictor and its calls read. Faces are u [ny, nx + 1] and v [ny + 1, nx].
struct box {
    box(std::size_t nx_, std::size_t ny_, double h_)
        : nx(nx_),
          ny(ny_),
          h(h_),
          x(nx + 1),
          xc(nx),
          y(ny + 1),
          yc(ny),
          types(nx * ny, fluid_type),
          dx(nx, h),
          dy(ny, h),
          dx_face(nx + 1, h),
          dy_face(ny + 1, h),
          bottom(nx + 1, h / 2),
          top(nx + 1, h / 2),
          left(ny + 1, h / 2),
          right(ny + 1, h / 2),
          held_u(ny * (nx + 1)),
          held_v((ny + 1) * nx),
          u(ny * (nx + 1), 0.0),
          v((ny + 1) * nx, 0.0),
          p(nx * ny, 0.0) {
        for (std::size_t i = 0; i <= nx; ++i) {
            x[i] = h * static_cast<double>(i);
        }
        for (std::size_t j = 0; j <= ny; ++j) {
            y[j] = h * static_cast<double>(j);
        }
        for (std::size_t i = 0; i < nx; ++i) {
            xc[i] = h * (static_cast<double>(i) + 0.5);
        }
        for (std::size_t j = 0; j < ny; ++j) {
            yc[j] = h * (static_cast<double>(j) + 0.5);
        }
        dx_face.front() = dx_face.back() = h / 2;
        dy_face.front() = dy_face.back() = h / 2;
    }

    void build(double alpha, std::int64_t sweeps, bool with_held = false) {
        const cfdcore::grid::grid_input<2> g{
            .faces = {x, y}, .centres = {xc, yc}, .cell_type = types, .solid = solid_type};
        mesh = std::make_unique<cfdcore::grid::grid<2>>(g);
        input = predictor_input<2>{};
        input.cell_widths = {dx, dy};
        input.face_widths = {dx_face, dy_face};
        input.low[0][0] = bottom.condition();
        input.low[1][0] = left.condition();
        input.high[0][0] = top.condition();
        input.high[1][0] = right.condition();
        if (with_held) {
            input.held = {held_u.span(), held_v.span()};
        }
        input.rho = rho;
        input.mu = mu;
        input.alpha = alpha;
        input.sweeps = sweeps;
        made = std::make_unique<predictor<2>>(*mesh, input);
        for (int c = 0; c < 2; ++c) {
            star[c].assign(c == 0 ? u.size() : v.size(), std::nan(""));
            a_p[c].assign(star[c].size(), std::nan(""));
        }
    }

    void predict(std::span<const double> mu_eff = {},
                 cfdcore::momentum::per_pair<std::span<const double>, 2> wall = {}) {
        made->predict({u, v}, p, mu_eff, wall, {star[0], star[1]}, {a_p[0], a_p[1]});
    }

    std::size_t uf(std::size_t j, std::size_t i) const { return j * (nx + 1) + i; }
    std::size_t vf(std::size_t j, std::size_t i) const { return j * nx + i; }
    std::size_t corner(std::size_t j, std::size_t i) const { return j * (nx + 1) + i; }

    std::size_t nx, ny;
    double h;
    double rho = 1.2;
    double mu = 0.05;
    std::vector<double> x, xc, y, yc;
    std::vector<std::int32_t> types;
    std::vector<double> dx, dy, dx_face, dy_face;
    edge bottom, top, left, right;
    flags held_u, held_v;
    std::vector<double> u, v, p;
    predictor_input<2> input;
    std::unique_ptr<cfdcore::grid::grid<2>> mesh;
    std::unique_ptr<predictor<2>> made;
    std::array<std::vector<double>, 2> star;
    std::array<std::vector<double>, 2> a_p;
};

// Every condition of an edge, zero gradient.
void slip(edge& e) {
    for (std::size_t i = 0; i < e.known.size; ++i) {
        e.known[i] = false;
    }
}

TEST(MomentumPredict, AUniformFlowInAChannelWithNoPressureGradientIsUnchanged) {
    // Along x and along y, the channel's walls without shear or moving with
    // the flow; air's viscosity and one that varies by cell.
    for (const int along : {0, 1}) {
        for (const bool moving : {false, true}) {
            for (const bool field : {false, true}) {
                box b(11, 9, 0.125);
                const double speed = 0.7;
                auto& flow = along == 0 ? b.u : b.v;
                std::fill(flow.begin(), flow.end(), speed);
                for (edge* e : along == 0 ? std::array<edge*, 2>{&b.bottom, &b.top}
                                          : std::array<edge*, 2>{&b.left, &b.right}) {
                    if (moving) {
                        std::fill(e->value.begin(), e->value.end(), speed);
                    } else {
                        slip(*e);
                    }
                }
                b.build(0.5, 3);
                std::vector<double> mu(b.nx * b.ny);
                for (std::size_t i = 0; i < mu.size(); ++i) {
                    mu[i] = 0.01 + hashed(i);
                }
                b.predict(field ? std::span<const double>(mu) : std::span<const double>());
                for (const double value : b.star[along]) {
                    EXPECT_NEAR(value, speed, 1e-14);
                }
                // The wall's tangential component is at rest and stays so.
                for (const double value : b.star[1 - along]) {
                    EXPECT_EQ(value, 0.0);
                }
            }
        }
    }
}

TEST(MomentumPredict, ALinearShearBetweenAWallAtRestAndAMovingOneIsUnchanged) {
    // Couette flow along x and along y: no advection, and a viscous force that
    // is zero only if the wall distance is the half cell's.
    for (const int along : {0, 1}) {
        box b(8, 10, 0.25);
        const std::size_t across = along == 0 ? b.ny : b.nx;
        const double depth = b.h * static_cast<double>(across);
        edge& moving = along == 0 ? b.top : b.right;
        std::fill(moving.value.begin(), moving.value.end(), 1.0);
        for (std::size_t j = 0; j < b.ny; ++j) {
            for (std::size_t i = 0; i <= b.nx; ++i) {
                if (along == 0) {
                    b.u[b.uf(j, i)] = b.yc[j] / depth;
                }
            }
        }
        for (std::size_t j = 0; j <= b.ny; ++j) {
            for (std::size_t i = 0; i < b.nx; ++i) {
                if (along == 1) {
                    b.v[b.vf(j, i)] = b.xc[i] / depth;
                }
            }
        }
        b.build(0.7, 4);
        b.predict();
        const auto& before = along == 0 ? b.u : b.v;
        for (std::size_t i = 0; i < before.size(); ++i) {
            EXPECT_NEAR(b.star[along][i], before[i], 1e-14);
        }
    }
}

// A discretely divergence-free field from a stream function at the corners.
void divergence_free(box& b) {
    const auto psi = [&](std::size_t j, std::size_t i) { return hashed(1000 * j + i); };
    for (std::size_t j = 0; j < b.ny; ++j) {
        for (std::size_t i = 0; i <= b.nx; ++i) {
            b.u[b.uf(j, i)] = (psi(j + 1, i) - psi(j, i)) / b.h;
        }
    }
    for (std::size_t j = 0; j <= b.ny; ++j) {
        for (std::size_t i = 0; i < b.nx; ++i) {
            b.v[b.vf(j, i)] = -(psi(j, i + 1) - psi(j, i)) / b.h;
        }
    }
}

// The neighbour coefficients of component c at a face of the box's layout.
double neighbour_sum(const box& b, std::size_t face) {
    return b.made->neighbour(0)[face] + b.made->neighbour(1)[face] +
           b.made->neighbour(2)[face] + b.made->neighbour(3)[face];
}

TEST(MomentumEquations, ADivergenceFreeFieldsDiagonalIsItsNeighbourSum) {
    box b(9, 7, 0.1);
    divergence_free(b);
    b.build(1.0, 1);
    b.made->equations(0, {b.u, b.v}, b.p, {}, {}, b.a_p[0]);
    for (std::size_t j = 1; j + 1 < b.ny; ++j) {
        for (std::size_t i = 1; i < b.nx; ++i) {
            const double sum = neighbour_sum(b, b.uf(j, i));
            EXPECT_NEAR(b.a_p[0][b.uf(j, i)], sum, 1e-13 * sum);
        }
    }
    b.made->equations(1, {b.u, b.v}, b.p, {}, {}, b.a_p[1]);
    for (std::size_t j = 1; j < b.ny; ++j) {
        for (std::size_t i = 1; i + 1 < b.nx; ++i) {
            const double sum = neighbour_sum(b, b.vf(j, i));
            EXPECT_NEAR(b.a_p[1][b.vf(j, i)], sum, 1e-13 * sum);
        }
    }
}

TEST(MomentumEquations, ADivergingFieldsDiagonalExceedsItsNeighbourSumByTheNetFlux) {
    // u = x: each control volume loses rho h dy of mass more than it receives.
    box b(6, 5, 0.5);
    for (std::size_t j = 0; j < b.ny; ++j) {
        for (std::size_t i = 0; i <= b.nx; ++i) {
            b.u[b.uf(j, i)] = b.x[i];
        }
    }
    b.build(1.0, 1);
    b.made->equations(0, {b.u, b.v}, b.p, {}, {}, b.a_p[0]);
    const std::size_t face = b.uf(2, 3);
    EXPECT_NEAR(b.a_p[0][face] - neighbour_sum(b, face), b.rho * b.h * b.h, 1e-13);
}

TEST(MomentumEquations, TheCoefficientsAreUpwind) {
    // Flow toward +x and +y: the convective part is on the upstream side only.
    box b(6, 6, 0.5);
    std::fill(b.u.begin(), b.u.end(), 2.0);
    std::fill(b.v.begin(), b.v.end(), 3.0);
    b.build(1.0, 1);
    const double flux_x = b.rho * 2.0 * b.h;
    const double flux_y = b.rho * 3.0 * b.h;
    b.made->equations(0, {b.u, b.v}, b.p, {}, {}, b.a_p[0]);
    std::size_t face = b.uf(2, 3);
    EXPECT_NEAR(b.made->neighbour(0)[face], b.mu, 1e-14);
    EXPECT_NEAR(b.made->neighbour(1)[face], b.mu + flux_x, 1e-13);
    EXPECT_NEAR(b.made->neighbour(2)[face], b.mu, 1e-14);
    EXPECT_NEAR(b.made->neighbour(3)[face], b.mu + flux_y, 1e-13);
    b.made->equations(1, {b.u, b.v}, b.p, {}, {}, b.a_p[1]);
    face = b.vf(3, 2);
    EXPECT_NEAR(b.made->neighbour(0)[face], b.mu, 1e-14);
    EXPECT_NEAR(b.made->neighbour(1)[face], b.mu + flux_y, 1e-13);
    EXPECT_NEAR(b.made->neighbour(2)[face], b.mu, 1e-14);
    EXPECT_NEAR(b.made->neighbour(3)[face], b.mu + flux_x, 1e-13);
}

TEST(MomentumEquations, AtRestTheDiagonalIsTheConductances) {
    // Unit cells: mu toward each neighbour, 2 mu toward a wall half a cell
    // away, nothing toward a zero-gradient edge, whatever block is beside.
    box b(5, 5, 1.0);
    slip(b.top);
    b.types[2 * b.nx + 2] = solid_type;
    b.build(1.0, 1);
    b.predict();
    // u: an interior face, one over the wall, one under the open edge, and one
    // over the block, whose face is a wall.
    EXPECT_DOUBLE_EQ(b.a_p[0][b.uf(2, 1)], 4.0 * b.mu);
    EXPECT_DOUBLE_EQ(b.a_p[0][b.uf(0, 2)], 5.0 * b.mu);
    EXPECT_DOUBLE_EQ(b.a_p[0][b.uf(4, 2)], 3.0 * b.mu);
    EXPECT_DOUBLE_EQ(b.a_p[0][b.uf(3, 2)], 5.0 * b.mu);
    EXPECT_DOUBLE_EQ(b.a_p[0][b.uf(1, 3)], 5.0 * b.mu);
    // v: beside the left wall, and beside the block on either side.
    EXPECT_DOUBLE_EQ(b.a_p[1][b.vf(2, 0)], 5.0 * b.mu);
    EXPECT_DOUBLE_EQ(b.a_p[1][b.vf(2, 1)], 5.0 * b.mu);
    EXPECT_DOUBLE_EQ(b.a_p[1][b.vf(3, 3)], 5.0 * b.mu);
    // The block's own faces are not unknowns, nor the domain's.
    for (const std::size_t face : {b.uf(2, 2), b.uf(2, 3), b.uf(1, 0), b.uf(1, b.nx)}) {
        EXPECT_EQ(bits(b.a_p[0][face]), bits(0.0));
    }
    for (const std::size_t face : {b.vf(2, 2), b.vf(3, 2), b.vf(0, 1), b.vf(b.ny, 1)}) {
        EXPECT_EQ(bits(b.a_p[1][face]), bits(0.0));
    }
}

TEST(MomentumEquations, AWallViscosityReplacesAirsAtTheWallsOnly) {
    box b(5, 5, 1.0);
    b.types[2 * b.nx + 2] = solid_type;
    b.build(1.0, 1);
    std::vector<double> wall((b.nx + 1) * (b.ny + 1), 3.0 * b.mu);
    b.predict({}, {{{wall}, {wall}}});
    EXPECT_DOUBLE_EQ(b.a_p[0][b.uf(2, 1)], 4.0 * b.mu);
    EXPECT_DOUBLE_EQ(b.a_p[0][b.uf(0, 2)], 9.0 * b.mu);
    EXPECT_DOUBLE_EQ(b.a_p[0][b.uf(3, 2)], 9.0 * b.mu);
    EXPECT_DOUBLE_EQ(b.a_p[1][b.vf(3, 3)], 9.0 * b.mu);
}

TEST(MomentumPredict, FromRestAPressureGradientGivesTheForceOverTheRelaxedDiagonal) {
    box b(6, 6, 0.5);
    for (std::size_t j = 0; j < b.ny; ++j) {
        for (std::size_t i = 0; i < b.nx; ++i) {
            b.p[j * b.nx + i] = 2.0 * b.xc[i] + 3.0 * b.yc[j];
        }
    }
    b.build(0.5, 1);
    b.predict();
    const std::size_t uf = b.uf(3, 3);
    const std::size_t vf = b.vf(3, 3);
    EXPECT_DOUBLE_EQ(b.star[0][uf], -2.0 * b.h * b.h / (b.a_p[0][uf] / 0.5));
    EXPECT_DOUBLE_EQ(b.star[1][vf], -3.0 * b.h * b.h / (b.a_p[1][vf] / 0.5));
}

// The reference's Jacobi sweeps written out on the equations one component's
// assembly leaves: every unknown from the previous iterate, the rest kept.
std::vector<double> swept(const box& b, int c, const std::vector<double>& start,
                          const std::vector<double>& diagonal, double alpha,
                          std::int64_t sweeps) {
    const std::size_t rows = c == 0 ? b.ny : b.ny + 1;
    const std::size_t width = c == 0 ? b.nx + 1 : b.nx;
    std::vector<double> current = start;
    for (std::int64_t n = 0; n < sweeps; ++n) {
        std::vector<double> next = start;
        for (std::size_t j = 0; j < rows; ++j) {
            for (std::size_t i = 0; i < width; ++i) {
                const bool edge_face =
                    c == 0 ? i == 0 || i + 1 == width : j == 0 || j + 1 == rows;
                if (edge_face) {
                    continue;
                }
                const std::size_t at = j * width + i;
                const double east = i + 1 < width ? current[at + 1] : 0.0;
                const double west = i > 0 ? current[at - 1] : 0.0;
                const double north = j + 1 < rows ? current[at + width] : 0.0;
                const double south = j > 0 ? current[at - width] : 0.0;
                const double sum =
                    c == 0 ? b.made->neighbour(0)[at] * east + b.made->neighbour(1)[at] * west +
                                 b.made->neighbour(2)[at] * north +
                                 b.made->neighbour(3)[at] * south
                           : b.made->neighbour(0)[at] * north +
                                 b.made->neighbour(1)[at] * south +
                                 b.made->neighbour(2)[at] * east + b.made->neighbour(3)[at] * west;
                next[at] = diagonal[at] > 0.0
                               ? (sum + b.made->source()[at]) / (diagonal[at] / alpha)
                               : 0.0;
            }
        }
        current = next;
    }
    return current;
}

TEST(MomentumPredict, TheSweepsAreJacobiSweepsOfTheAssembledEquations) {
    // An odd and an even count, so that the last iterate lands in the output
    // either way, with the coefficients and the source held.
    for (const std::int64_t sweeps : {1, 2, 3, 4, 7}) {
        box b(13, 6, 0.25);
        divergence_free(b);
        for (std::size_t i = 0; i < b.p.size(); ++i) {
            b.p[i] = hashed(i + 77);
        }
        b.types[2 * b.nx + 4] = solid_type;
        b.build(0.6, sweeps);
        b.predict();
        for (int c = 0; c < 2; ++c) {
            const auto& own = c == 0 ? b.u : b.v;
            std::vector<double> diagonal(own.size());
            b.made->equations(c, {b.u, b.v}, b.p, {}, {}, diagonal);
            const auto expected = swept(b, c, own, diagonal, 0.6, sweeps);
            for (std::size_t i = 0; i < own.size(); ++i) {
                EXPECT_EQ(bits(b.star[c][i]), bits(expected[i])) << c << " " << i;
                EXPECT_EQ(bits(b.a_p[c][i]), bits(diagonal[i])) << c << " " << i;
            }
        }
    }
}

TEST(MomentumPredict, AHeldFaceKeepsItsValueAndAnObstaclesOtherFacesAreZero) {
    for (const std::int64_t sweeps : {1, 2, 5}) {
        box b(7, 7, 0.5);
        b.types[3 * b.nx + 3] = solid_type;
        std::fill(b.u.begin(), b.u.end(), 0.25);
        std::fill(b.v.begin(), b.v.end(), -0.75);
        b.held_u[b.uf(3, 4)] = true;
        b.held_v[b.vf(4, 3)] = true;
        b.build(0.5, sweeps, true);
        b.predict();
        EXPECT_EQ(b.star[0][b.uf(3, 4)], 0.25);
        EXPECT_EQ(bits(b.star[0][b.uf(3, 3)]), bits(0.0));
        EXPECT_EQ(b.star[1][b.vf(4, 3)], -0.75);
        EXPECT_EQ(bits(b.star[1][b.vf(3, 3)]), bits(0.0));
        // Not unknowns, so the pressure correction leaves them.
        EXPECT_EQ(b.a_p[0][b.uf(3, 4)], 0.0);
        EXPECT_EQ(b.a_p[1][b.vf(4, 3)], 0.0);
        // The domain's faces keep the input.
        EXPECT_EQ(b.star[0][b.uf(1, 0)], 0.25);
        EXPECT_EQ(b.star[0][b.uf(1, b.nx)], 0.25);
        EXPECT_EQ(b.star[1][b.vf(0, 1)], -0.75);
        EXPECT_EQ(b.star[1][b.vf(b.ny, 1)], -0.75);
    }
}

TEST(MomentumViscosity, AUniformFieldOfAirsViscosityIsTheScalarPathBitForBit) {
    box b(10, 8, 0.1);
    divergence_free(b);
    b.types[3 * b.nx + 4] = solid_type;
    b.build(0.5, 3);
    b.predict();
    const auto star = b.star;
    const auto a_p = b.a_p;
    std::vector<double> mu(b.nx * b.ny, b.mu);
    b.predict(mu);
    for (int c = 0; c < 2; ++c) {
        for (std::size_t i = 0; i < star[c].size(); ++i) {
            EXPECT_EQ(bits(b.star[c][i]), bits(star[c][i]));
            EXPECT_EQ(bits(b.a_p[c][i]), bits(a_p[c][i]));
        }
    }
}

TEST(MomentumViscosity, TheStencilTakesTheFaceRulesValueAtTheWalls) {
    // Rows of cells of viscosity 1, 2, 4, 8: at the bottom wall the wall cells'
    // mean, 1. The columns of the block's sides take the harmonic mean's
    // SOLID rule: the fluid cell's value, not read from the block.
    box b(4, 4, 1.0);
    b.types[1 * b.nx + 2] = solid_type;
    b.build(0.5, 1);
    std::vector<double> mu(b.nx * b.ny);
    for (std::size_t j = 0; j < b.ny; ++j) {
        for (std::size_t i = 0; i < b.nx; ++i) {
            mu[j * b.nx + i] = std::ldexp(1.0, static_cast<int>(j));
        }
    }
    mu[0] = 3.0;
    mu[1 * b.nx + 2] = std::nan("");
    std::array<std::vector<double>, 2> out;
    for (auto& a : out) {
        a.assign((b.nx + 1) * (b.ny + 1), 0.0);
    }
    b.made->stencil_viscosity(mu, {{{out[0]}, {out[1]}}});
    // u's bottom wall between cells of 3 and 1: their width-weighted mean.
    EXPECT_EQ(out[0][b.corner(0, 1)], 2.0);
    EXPECT_EQ(out[0][b.corner(0, 2)], 1.0);
    EXPECT_EQ(out[0][b.corner(4, 1)], 8.0);
    // Over the block, half the span of corner (2, 2) is the block's: the
    // harmonic mean of 2 and 4 in one column, 4 carried across in the other.
    const double harmonic = 2.0 + (4.0 - 2.0) * (2.0 * 0.5) / (0.5 * 4.0 + 0.5 * 2.0);
    EXPECT_EQ(out[0][b.corner(2, 2)], harmonic + (4.0 - harmonic) * 0.5);
    // Under it, 1 is carried up in the block's column.
    const double below = 1.0 + (2.0 - 1.0) * (1.0 * 0.5) / (0.5 * 2.0 + 0.5 * 1.0);
    EXPECT_EQ(out[0][b.corner(1, 2)], below + (1.0 - below) * 0.5);
    // v's left wall between rows of 2 and 4, and air's where no unknown reads.
    EXPECT_EQ(out[1][b.corner(2, 0)], 3.0);
    EXPECT_EQ(out[1][b.corner(2, 1)], b.mu);
    EXPECT_EQ(out[0][b.corner(0, 0)], b.mu);
}

TEST(MomentumViscosity, TheChecksReadWhatTheEquationsReadAndNothingElse) {
    box b(5, 4, 1.0);
    b.types[1 * b.nx + 2] = solid_type;
    b.build(0.5, 1);
    std::vector<double> mu(b.nx * b.ny, 1.0);
    mu[1 * b.nx + 2] = std::nan("");
    EXPECT_EQ(b.made->check_viscosity(mu), viscosity_status::ok);
    // The last cell, past the whole blocks, and one that is in them.
    for (const std::size_t cell : {mu.size() - 1, std::size_t{0}}) {
        for (const double bad : {std::nan(""), std::numeric_limits<double>::infinity(),
                                 -std::numeric_limits<double>::infinity()}) {
            auto spoiled = mu;
            spoiled[cell] = bad;
            EXPECT_EQ(b.made->check_viscosity(spoiled), viscosity_status::not_finite);
        }
        for (const double bad : {0.0, -0.0, -1.0}) {
            auto spoiled = mu;
            spoiled[cell] = bad;
            EXPECT_EQ(b.made->check_viscosity(spoiled), viscosity_status::not_positive);
        }
    }
    auto both = mu;
    both[0] = -1.0;
    both.back() = std::nan("");
    EXPECT_EQ(b.made->check_viscosity(both), viscosity_status::not_finite);

    std::vector<double> wall((b.nx + 1) * (b.ny + 1), std::nan(""));
    for (int c = 0; c < 2; ++c) {
        auto read = wall;
        const auto& marked = b.made->setup(c).pairs[0].wall_read;
        std::size_t last = 0;
        for (std::size_t i = 0; i < read.size(); ++i) {
            if (marked[i] != cfdcore::lanes::none) {
                read[i] = 1.0;
                last = i;
            }
        }
        EXPECT_EQ(b.made->check_wall_viscosity(c, 0, read), viscosity_status::ok);
        read[last] = std::nan("");
        EXPECT_EQ(b.made->check_wall_viscosity(c, 0, read), viscosity_status::not_finite);
        read[last] = 0.0;
        EXPECT_EQ(b.made->check_wall_viscosity(c, 0, read), viscosity_status::not_positive);
    }
}

// A straight duct of n cells along one direction, which has no transverse one:
// one component, no wall stencil and no pair.
struct duct {
    explicit duct(std::size_t n)
        : x(n + 1), xc(n), types(n, fluid_type), u(n + 1, 0.0), p(n, 0.0) {
        // Widths that differ, so a weight or a distance read from the wrong
        // cell shows.
        for (std::size_t i = 0; i < n; ++i) {
            x[i + 1] = x[i] + 0.5 + hashed(i);
        }
        for (std::size_t i = 0; i < n; ++i) {
            xc[i] = 0.5 * (x[i] + x[i + 1]);
            dx.push_back(x[i + 1] - x[i]);
        }
        dx_face.push_back(xc.front() - x.front());
        for (std::size_t i = 1; i < n; ++i) {
            dx_face.push_back(xc[i] - xc[i - 1]);
        }
        dx_face.push_back(x.back() - xc.back());
    }

    void build(double alpha, std::int64_t sweeps) {
        mesh = std::make_unique<cfdcore::grid::grid<1>>(cfdcore::grid::grid_input<1>{
            .faces = {x}, .centres = {xc}, .cell_type = types, .solid = solid_type});
        input = predictor_input<1>{};
        input.cell_widths = {dx};
        input.face_widths = {dx_face};
        input.rho = rho;
        input.mu = mu;
        input.alpha = alpha;
        input.sweeps = sweeps;
        made = std::make_unique<predictor<1>>(*mesh, input);
        star.assign(u.size(), std::nan(""));
        a_p.assign(u.size(), std::nan(""));
    }

    void predict(std::span<const double> mu_eff = {}) {
        made->predict({u}, p, mu_eff, {}, {star}, {a_p});
    }

    double rho = 1.2;
    double mu = 0.05;
    std::vector<double> x, xc, dx, dx_face;
    std::vector<std::int32_t> types;
    std::vector<double> u, p;
    predictor_input<1> input;
    std::unique_ptr<cfdcore::grid::grid<1>> mesh;
    std::unique_ptr<predictor<1>> made;
    std::vector<double> star, a_p;
};

// QUICK's value at point f: the quadratic through the far-upstream, upstream
// and downstream nodes, by Lagrange's basis.
double quick(double far_x, double up_x, double down_x, double far, double up, double down,
             double f) {
    return far * (f - up_x) * (f - down_x) / ((far_x - up_x) * (far_x - down_x)) +
           up * (f - far_x) * (f - down_x) / ((up_x - far_x) * (up_x - down_x)) +
           down * (f - far_x) * (f - up_x) / ((down_x - far_x) * (down_x - up_x));
}

TEST(MomentumOneDirection, AUniformFlowWithNoPressureGradientIsUnchanged) {
    for (const double speed : {0.7, -0.7}) {
        for (const bool field : {false, true}) {
            duct d(11);
            std::fill(d.u.begin(), d.u.end(), speed);
            d.build(0.5, 3);
            std::vector<double> mu(d.p.size());
            for (std::size_t i = 0; i < mu.size(); ++i) {
                mu[i] = 0.01 + hashed(i);
            }
            d.predict(field ? std::span<const double>(mu) : std::span<const double>());
            for (const double value : d.star) {
                EXPECT_NEAR(value, speed, 1e-14);
            }
        }
    }
}

TEST(MomentumOneDirection, TheEquationsAreTheStreamwisePartAlone) {
    // Written out by hand at every face, with and without a viscosity field,
    // for a flow of both signs: two neighbours, the net flux between them, the
    // QUICK correction, form b's stress and the pressure.
    for (const bool field : {false, true}) {
        const std::size_t n = 9;
        duct d(n);
        for (std::size_t i = 0; i <= n; ++i) {
            d.u[i] = hashed(100 + i) - 0.5;
        }
        for (std::size_t i = 0; i < n; ++i) {
            d.p[i] = hashed(200 + i);
        }
        std::vector<double> mu(n, d.mu);
        if (field) {
            for (std::size_t i = 0; i < n; ++i) {
                mu[i] = 0.02 + hashed(300 + i);
            }
        }
        d.build(0.6, 1);
        d.made->equations(0, {d.u}, d.p,
                          field ? std::span<const double>(mu) : std::span<const double>(), {},
                          d.a_p);
        // The flux through cell i and what it gives the faces on either side.
        std::vector<double> flux(n), correction(n), diffusion(n);
        for (std::size_t i = 0; i < n; ++i) {
            flux[i] = d.rho * 0.5 * (d.u[i] + d.u[i + 1]);
            diffusion[i] = mu[i] / d.dx[i];
            const bool positive = flux[i] > 0.0;
            const std::size_t up = positive ? i : i + 1;
            const std::size_t down = positive ? i + 1 : i;
            // The far-upstream node, or the next one downstream at an end.
            std::size_t far = 0;
            if (positive) {
                far = i >= 1 ? i - 1 : i + 2;
            } else {
                far = i + 2 <= n ? i + 2 : i - 1;
            }
            const double face = quick(d.x[far], d.x[up], d.x[down], d.u[far], d.u[up], d.u[down],
                                      d.xc[i]);
            correction[i] = (face - d.u[up]) * flux[i];
        }
        for (std::size_t i = 1; i < n; ++i) {
            const double east = diffusion[i] + std::max(-flux[i], 0.0);
            const double west = diffusion[i - 1] + std::max(flux[i - 1], 0.0);
            const double diagonal = east + west + flux[i] - flux[i - 1];
            EXPECT_NEAR(d.made->neighbour(0)[i], east, 1e-13);
            EXPECT_NEAR(d.made->neighbour(1)[i], west, 1e-13);
            EXPECT_NEAR(d.a_p[i], diagonal, 1e-13);
            double source = -(correction[i] - correction[i - 1]) - (d.p[i] - d.p[i - 1]) +
                            (1.0 - 0.6) / 0.6 * diagonal * d.u[i];
            if (field) {
                const double low = d.x[i] - d.xc[i - 1];
                const double high = d.xc[i] - d.x[i];
                const double mean = mu[i - 1] + (mu[i] - mu[i - 1]) * high / (low + high);
                source += (mu[i] - mean) * (d.u[i + 1] - d.u[i]) / d.dx[i] -
                          (mu[i - 1] - mean) * (d.u[i] - d.u[i - 1]) / d.dx[i - 1];
            }
            EXPECT_NEAR(d.made->source()[i], source, 1e-13);
        }
        for (const std::size_t end : {std::size_t{0}, n}) {
            EXPECT_EQ(bits(d.a_p[end]), bits(0.0));
            EXPECT_EQ(bits(d.made->neighbour(0)[end]), bits(0.0));
            EXPECT_EQ(bits(d.made->neighbour(1)[end]), bits(0.0));
        }
    }
}

TEST(MomentumOneDirection, TheSweepsAreJacobiSweepsAndTheEndsKeepTheInput) {
    for (const std::size_t n : {2, 3, 4, 9, 10, 21}) {
        for (const std::int64_t sweeps : {1, 2, 5}) {
            duct d(n);
            for (std::size_t i = 0; i <= n; ++i) {
                d.u[i] = hashed(i) - 0.25;
            }
            if (n >= 9) {
                d.types[4] = solid_type;
                d.u[4] = d.u[5] = 0.0;
            }
            d.build(0.6, sweeps);
            d.predict();
            std::vector<double> diagonal(n + 1);
            d.made->equations(0, {d.u}, d.p, {}, {}, diagonal);
            std::vector<double> current = d.u;
            for (std::int64_t k = 0; k < sweeps; ++k) {
                std::vector<double> next = d.u;
                for (std::size_t i = 1; i < n; ++i) {
                    const double sum = d.made->neighbour(0)[i] * current[i + 1] +
                                       d.made->neighbour(1)[i] * current[i - 1] +
                                       d.made->source()[i];
                    next[i] = diagonal[i] > 0.0 ? sum / (diagonal[i] / 0.6) : 0.0;
                }
                current = next;
            }
            for (std::size_t i = 0; i <= n; ++i) {
                EXPECT_EQ(bits(d.star[i]), bits(current[i])) << n << " " << i;
                EXPECT_EQ(bits(d.a_p[i]), bits(diagonal[i])) << n << " " << i;
            }
        }
    }
}

TEST(MomentumOneDirection, ItHasNoPairAndRefusesOneCell) {
    duct d(4);
    d.build(0.5, 1);
    EXPECT_TRUE(d.made->setup(0).pairs.empty());
    // No pair, so nothing to write and nothing to read.
    std::vector<double> mu(d.p.size(), 1.0);
    d.made->stencil_viscosity(mu, {});
    EXPECT_EQ(d.made->check_viscosity(mu), viscosity_status::ok);
    mu.back() = -1.0;
    EXPECT_EQ(d.made->check_viscosity(mu), viscosity_status::not_positive);
    duct thin(1);
    EXPECT_THROW(thin.build(0.5, 1), std::invalid_argument);
}

TEST(MomentumPairs, ASlotsDirectionAndADirectionsSlotAreInverse) {
    for (int c = 0; c < 4; ++c) {
        int last = -1;
        for (int q = 0; q < 3; ++q) {
            const int t = cfdcore::momentum::transverse(c, q);
            EXPECT_NE(t, c);
            EXPECT_GT(t, last);
            EXPECT_EQ(cfdcore::momentum::slot_of(c, t), q);
            last = t;
        }
    }
}

TEST(MomentumSetup, EveryCallRefusesAnArrayOfAnotherLength) {
    box b(5, 4, 1.0);
    b.types[1 * b.nx + 2] = solid_type;
    b.build(0.5, 2);
    // An array of each of the calls' lengths, and one a value longer or shorter.
    const std::size_t corners = (b.nx + 1) * (b.ny + 1);
    const std::vector<double> ones(corners + 1, 1.0);
    std::vector<double> room(corners + 1, 0.0);
    const auto given = [&](std::size_t n) { return std::span<const double>(ones).first(n); };
    const std::size_t cells = b.nx * b.ny;
    const std::array<std::size_t, 2> faces{b.u.size(), b.v.size()};
    struct call {
        std::array<std::size_t, 2> velocity, star, diagonal, wall, stencil;
        std::size_t p, mu;
    };
    const call good{faces, faces, faces, {corners, corners}, {corners, corners}, cells, cells};
    // Each output its own array, so that none overlaps another.
    std::array<std::array<std::vector<double>, 2>, 3> out;
    for (auto& pair : out) {
        for (auto& a : pair) {
            a.assign(corners + 1, 0.0);
        }
    }
    const auto taken = [&](int which, int c, std::size_t n) {
        return std::span<double>(out[which][c]).first(n);
    };
    const auto predict = [&](const call& k) {
        b.made->predict({given(k.velocity[0]), given(k.velocity[1])}, given(k.p), given(k.mu),
                        {{{given(k.wall[0])}, {given(k.wall[1])}}},
                        {taken(0, 0, k.star[0]), taken(0, 1, k.star[1])},
                        {taken(1, 0, k.diagonal[0]), taken(1, 1, k.diagonal[1])});
    };
    const auto equations = [&](int c, const call& k) {
        b.made->equations(c, {given(k.velocity[0]), given(k.velocity[1])}, given(k.p),
                          given(k.mu), {{{given(k.wall[0])}, {given(k.wall[1])}}},
                          taken(1, 0, k.diagonal[c == 1 ? 1 : 0]));
    };
    const auto stencil = [&](const call& k) {
        b.made->stencil_viscosity(
            given(k.mu), {{{taken(2, 0, k.stencil[0])}, {taken(2, 1, k.stencil[1])}}});
    };
    EXPECT_NO_THROW(predict(good));
    EXPECT_NO_THROW(equations(0, good));
    EXPECT_NO_THROW(equations(1, good));
    EXPECT_NO_THROW(stencil(good));
    for (const int by : {-1, 1}) {
        const auto off = [&](std::size_t n) { return by < 0 ? n - 1 : n + 1; };
        const auto changed = [&](auto change) {
            call k = good;
            change(k);
            return k;
        };
        for (int c = 0; c < 2; ++c) {
            const call velocity = changed([&](call& k) { k.velocity[c] = off(faces[c]); });
            const call wall = changed([&](call& k) { k.wall[c] = off(corners); });
            const call diagonal = changed([&](call& k) { k.diagonal[c] = off(faces[c]); });
            EXPECT_THROW(predict(velocity), std::invalid_argument);
            EXPECT_THROW(predict(wall), std::invalid_argument);
            EXPECT_THROW(predict(diagonal), std::invalid_argument);
            EXPECT_THROW(predict(changed([&](call& k) { k.star[c] = off(faces[c]); })),
                         std::invalid_argument);
            for (int of = 0; of < 2; ++of) {
                EXPECT_THROW(equations(of, velocity), std::invalid_argument);
                EXPECT_THROW(equations(of, wall), std::invalid_argument);
            }
            EXPECT_THROW(equations(c, diagonal), std::invalid_argument);
            EXPECT_THROW(stencil(changed([&](call& k) { k.stencil[c] = off(corners); })),
                         std::invalid_argument);
            EXPECT_THROW(b.made->check_wall_viscosity(c, 0, given(off(corners))),
                         std::invalid_argument);
        }
        const call p = changed([&](call& k) { k.p = off(cells); });
        const call mu = changed([&](call& k) { k.mu = off(cells); });
        EXPECT_THROW(predict(p), std::invalid_argument);
        EXPECT_THROW(predict(mu), std::invalid_argument);
        EXPECT_THROW(equations(0, p), std::invalid_argument);
        EXPECT_THROW(equations(0, mu), std::invalid_argument);
        EXPECT_THROW(stencil(mu), std::invalid_argument);
        EXPECT_THROW(b.made->check_viscosity(given(off(cells))), std::invalid_argument);
    }
    // An empty viscosity is air's for an equation and no field for the rest.
    EXPECT_THROW(b.made->check_viscosity({}), std::invalid_argument);
    EXPECT_THROW(b.made->check_wall_viscosity(0, 0, {}), std::invalid_argument);
    EXPECT_THROW(stencil([&] {
                     call k = good;
                     k.mu = 0;
                     return k;
                 }()),
                 std::invalid_argument);
    // A component or a slot there is not.
    for (const int c : {-1, 2}) {
        EXPECT_THROW(equations(c, good), std::invalid_argument);
        EXPECT_THROW(b.made->check_wall_viscosity(c, 0, given(corners)), std::invalid_argument);
    }
    for (const int q : {-1, 1}) {
        EXPECT_THROW(b.made->check_wall_viscosity(0, q, given(corners)), std::invalid_argument);
    }
}

TEST(MomentumSetup, ProgrammingErrorsThrow) {
    box b(4, 5, 1.0);
    b.build(0.5, 1);
    const std::vector<double> three(3, 1.0);
    const flags three_flags(3);
    std::vector<double> spoiled;
    const auto refused = [&](auto change) {
        predictor_input<2> input = b.input;
        change(input);
        EXPECT_FALSE(predictor<2>::check(*b.mesh, input).empty());
        EXPECT_THROW(predictor<2>(*b.mesh, input), std::invalid_argument);
    };
    EXPECT_TRUE(predictor<2>::check(*b.mesh, b.input).empty());
    for (int d = 0; d < 2; ++d) {
        refused([&](auto& in) { in.cell_widths[d] = three; });
        refused([&](auto& in) { in.face_widths[d] = three; });
        refused([&](auto& in) { in.low[d][0].is_dirichlet = three_flags.span(); });
        refused([&](auto& in) { in.low[d][0].value = three; });
        refused([&](auto& in) { in.high[d][0].is_dirichlet = three_flags.span(); });
        refused([&](auto& in) { in.high[d][0].value = three; });
        refused([&](auto& in) { in.held[d] = three_flags.span(); });
    }
    refused([](auto& in) { in.sweeps = 0; });
    // The numbers: each alone, by each way of not being finite and positive.
    const double nan = std::nan("");
    const double inf = std::numeric_limits<double>::infinity();
    for (const double bad : {0.0, -0.0, -1.0, nan, inf, -inf}) {
        refused([&](auto& in) { in.rho = bad; });
        refused([&](auto& in) { in.mu = bad; });
        refused([&](auto& in) { in.alpha = bad; });
        for (int d = 0; d < 2; ++d) {
            refused([&](auto& in) { in.low[d][0].wall_distance = bad; });
            refused([&](auto& in) { in.high[d][0].wall_distance = bad; });
            // The last entry, so that a check of the first alone shows.
            for (const bool faces : {false, true}) {
                refused([&](auto& in) {
                    auto& widths = faces ? in.face_widths[d] : in.cell_widths[d];
                    spoiled.assign(widths.begin(), widths.end());
                    spoiled.back() = bad;
                    widths = spoiled;
                });
            }
        }
    }
    refused([](auto& in) { in.alpha = std::nextafter(1.0, 2.0); });
    // The ends of the ranges are inside them.
    predictor_input<2> ends = b.input;
    ends.alpha = 1.0;
    ends.rho = ends.mu = std::numeric_limits<double>::denorm_min();
    EXPECT_TRUE(predictor<2>::check(*b.mesh, ends).empty());
    ends.rho = ends.mu = std::numeric_limits<double>::max();
    EXPECT_TRUE(predictor<2>::check(*b.mesh, ends).empty());
    for (const auto& [nx, ny] : {std::pair{1, 4}, std::pair{4, 1}}) {
        box thin(nx, ny, 1.0);
        EXPECT_THROW(thin.build(0.5, 1), std::invalid_argument);
    }
}

}  // namespace
