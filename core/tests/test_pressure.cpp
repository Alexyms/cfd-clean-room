// The pressure module's own tests (ADR-017 D): results known without the
// reference. The comparison with pressure.py is tests/test_core_pressure.py's.

#include "cfdcore/grid.hpp"
#include "cfdcore/pressure.hpp"
#include "cfdcore/reduce.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using cfdcore::layout::layout;
using cfdcore::multigrid::system;
using cfdcore::pressure::corrector;
using cfdcore::pressure::corrector_input;
using cfdcore::pressure::preconditioner;
using cfdcore::pressure::solve_keys;
using cfdcore::pressure::solve_result;
using cfdcore::pressure::solve_status;
using cfdcore::pressure::solver;

constexpr std::int32_t fluid_type = 0;
constexpr std::int32_t solid_type = 1;
constexpr std::int32_t edge_type = 2;  // neither, as the mesh's BOUNDARY cells
constexpr double nan_value = std::numeric_limits<double>::quiet_NaN();
constexpr double infinity = std::numeric_limits<double>::infinity();

std::uint64_t bits(double x) { return std::bit_cast<std::uint64_t>(x); }

double hashed(std::size_t i) {
    std::uint64_t z = 0x9e3779b97f4a7c15ULL * (i + 1);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z ^= z >> 31;
    return static_cast<double>(z >> 11) * 0x1.0p-53;
}

cfdcore::multigrid::keys level_keys() {
    cfdcore::multigrid::keys k;
    k.coarsest_cells = 4;
    k.sweeps = 1;
    k.weight = 2.0 / 3.0;
    k.dominance_tolerance = 1e-12;
    k.rcond = 1e-15;
    return k;
}

// A bool array a span can view.
struct flags {
    explicit flags(std::size_t n) : size(n), values(new bool[n]) {
        std::fill(values.get(), values.get() + n, false);
    }
    std::span<const bool> span() const { return {values.get(), size}; }
    std::size_t size;
    std::unique_ptr<bool[]> values;
};

// A room of nx by ny cells of the given widths, every array a corrector and its
// calls read. The edges' outlets are left, right, bottom, top.
struct room {
    room(std::vector<double> dx_, std::vector<double> dy_)
        : nx(dx_.size()),
          ny(dy_.size()),
          dx(std::move(dx_)),
          dy(std::move(dy_)),
          types(nx * ny, fluid_type) {
        outlets.emplace_back(ny);
        outlets.emplace_back(ny);
        outlets.emplace_back(nx);
        outlets.emplace_back(nx);
        u.assign(ny * (nx + 1), 0.0);
        v.assign((ny + 1) * nx, 0.0);
        a_u = u;
        a_v = v;
        p.assign(nx * ny, 0.0);
    }

    std::size_t cell(std::size_t j, std::size_t i) const { return j * nx + i; }
    std::size_t u_face(std::size_t j, std::size_t i) const { return j * (nx + 1) + i; }
    std::size_t v_face(std::size_t j, std::size_t i) const { return j * nx + i; }

    cfdcore::grid::grid<2> grid() const {
        const auto axis = [](const std::vector<double>& widths) {
            std::pair<std::vector<double>, std::vector<double>> out{{0.0}, {}};
            for (const double w : widths) {
                out.second.push_back(out.first.back() + 0.5 * w);
                out.first.push_back(out.first.back() + w);
            }
            return out;
        };
        const auto x = axis(dx);
        const auto y = axis(dy);
        return cfdcore::grid::grid<2>({.faces = {x.first, y.first},
                                       .centres = {x.second, y.second},
                                       .cell_type = types,
                                       .solid = solid_type});
    }

    corrector_input<2> input() const {
        corrector_input<2> in;
        in.cell_widths = {dx, dy};
        in.outlet_low = {outlets[0].span(), outlets[2].span()};
        in.outlet_high = {outlets[1].span(), outlets[3].span()};
        in.fluid = fluid_type;
        in.rho = 1.2;
        in.alpha = 0.3;
        in.rtol = 1e-12;
        in.floor = 0.0;
        in.max_iter = 500;
        in.method = method;
        in.levels = level_keys();
        return in;
    }

    // Diagonals at every face two open cells share, velocities everywhere but
    // on the faces of SOLID cells and, unless an outlet, the domain's edges.
    void plant() {
        const auto solid = [&](std::size_t j, std::size_t i) {
            return types[cell(j, i)] == solid_type;
        };
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 1; i < nx; ++i) {
                if (!solid(j, i) && !solid(j, i - 1)) {
                    u[u_face(j, i)] = hashed(u_face(j, i)) - 0.5;
                    a_u[u_face(j, i)] = 0.5 + hashed(5000 + u_face(j, i));
                }
            }
        }
        for (std::size_t j = 1; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                if (!solid(j, i) && !solid(j - 1, i)) {
                    v[v_face(j, i)] = hashed(9000 + v_face(j, i)) - 0.5;
                    a_v[v_face(j, i)] = 0.5 + hashed(13000 + v_face(j, i));
                }
            }
        }
        for (std::size_t i = 0; i < p.size(); ++i) {
            p[i] = hashed(17000 + i);
        }
    }

    std::size_t nx, ny;
    std::vector<double> dx, dy;
    std::vector<std::int32_t> types;
    std::vector<flags> outlets;
    std::vector<double> u, v, a_u, a_v, p;
    preconditioner method = preconditioner::multigrid;
};

struct corrected {
    explicit corrected(const room& r)
        : u(r.u.size(), nan_value),
          v(r.v.size(), nan_value),
          p(r.p.size(), nan_value),
          p_prime(r.p.size(), nan_value) {}
    solve_result run(corrector<2>& c, const room& r) {
        return c.correct({r.u, r.v}, {r.a_u, r.a_v}, r.p, {u, v}, p, p_prime);
    }
    std::vector<double> u, v, p, p_prime;
};

std::vector<double> widths(std::size_t n, std::size_t seed) {
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = 0.5 + hashed(seed + i);
    }
    return out;
}

// ---------------------------------------------------------------------------
// The equation

// Two cells side by side, an outlet on the right: every number by hand.
TEST(PressureEquation, OfTwoCellsAndAnOutlet) {
    room r({2.0, 4.0}, {0.5});
    r.outlets[1].values[0] = true;
    const auto g = r.grid();
    corrector<2> c(g, r.input());
    EXPECT_FALSE(c.needs_pin());
    r.a_u = {0.0, 2.0, 0.0};
    c.assemble({r.a_u, r.a_v});
    // d = dy / a_P at the inner face, and the outlet's from the same diagonal.
    EXPECT_EQ(std::vector<double>(c.face_d(0).begin(), c.face_d(0).end()),
              (std::vector<double>{0.0, 0.25, 0.25}));
    const double coupling = 1.2 * 0.25 * 0.5;
    const auto& s = c.equation();
    EXPECT_EQ(s.high[0], (std::vector<double>{coupling, 0.0}));
    EXPECT_EQ(s.low[0], (std::vector<double>{0.0, coupling}));
    EXPECT_EQ(s.high[1], std::vector<double>(2, 0.0));
    EXPECT_EQ(s.low[1], std::vector<double>(2, 0.0));
    EXPECT_EQ(s.centre, (std::vector<double>{coupling, coupling + coupling}));

    // The imbalance: inflow 1 at the left, 3 between, 2 out, and v of 0 to 1 up.
    r.u = {1.0, 3.0, 2.0};
    r.v = {0.0, 0.0, 1.0, -1.0};
    std::vector<double> b(2);
    c.mass_imbalance({r.u, r.v}, b);
    EXPECT_EQ(b, (std::vector<double>{1.2 * (2.0 * 0.5 + 1.0 * 2.0),
                                      1.2 * (-1.0 * 0.5 + -1.0 * 4.0)}));
    std::vector<double> f(2);
    c.right_hand_side({r.u, r.v}, f);
    EXPECT_EQ(f, (std::vector<double>{-b[0], -b[1]}));

    // The correction by p' = (1, 2): the outlet face against p' = 0.
    r.p = {10.0, 20.0};
    std::vector<double> p_prime{1.0, 2.0};
    corrected out(r);
    c.apply({r.u, r.v}, r.p, p_prime, {out.u, out.v}, out.p);
    EXPECT_EQ(out.u, (std::vector<double>{1.0, 3.0 - 0.25 * 1.0, 2.0 - 0.25 * -2.0}));
    EXPECT_EQ(out.v, r.v);
    EXPECT_EQ(out.p, (std::vector<double>{10.0 + 0.3 * 1.0, 20.0 + 0.3 * 2.0}));
    EXPECT_EQ(p_prime, (std::vector<double>{1.0, 2.0}));
}

// The same along the other direction and at the low edges, whose roles differ.
TEST(PressureEquation, AnOutletOnEveryEdge) {
    for (std::size_t edge = 0; edge < 4; ++edge) {
        SCOPED_TRACE(edge);
        room r(widths(3, 40), widths(4, 50));
        const std::size_t length = edge < 2 ? r.ny : r.nx;
        r.outlets[edge].values[length - 1] = true;
        r.plant();
        const auto g = r.grid();
        corrector<2> c(g, r.input());
        EXPECT_FALSE(c.needs_pin());
        c.assemble({r.a_u, r.a_v});
        const std::size_t last = length - 1;
        // The outlet's face, its inward neighbour's diagonal and its cell.
        const std::array<std::size_t, 4> face{r.u_face(last, 0), r.u_face(last, r.nx),
                                              r.v_face(0, last), r.v_face(r.ny, last)};
        const std::array<std::size_t, 4> near{r.u_face(last, 1), r.u_face(last, r.nx - 1),
                                              r.v_face(1, last), r.v_face(r.ny - 1, last)};
        const std::array<std::size_t, 4> cell{r.cell(last, 0), r.cell(last, r.nx - 1),
                                              r.cell(0, last), r.cell(r.ny - 1, last)};
        const int direction = edge < 2 ? 0 : 1;
        const double area = edge < 2 ? r.dy[last] : r.dx[last];
        const auto& a = edge < 2 ? r.a_u : r.a_v;
        const auto d = c.face_d(direction);
        EXPECT_EQ(bits(d[face[edge]]), bits(area / a[near[edge]]));
        std::size_t borrowed = 0;
        for (std::size_t f = 0; f < d.size(); ++f) {
            borrowed += d[f] > 0.0 && !(a[f] > 0.0);
        }
        EXPECT_EQ(borrowed, 1u);
        // In its cell's diagonal, with no coupling across it.
        const auto& s = c.equation();
        const std::size_t i = cell[edge];
        const double couplings = s.high[0][i] + s.low[0][i] + s.high[1][i] + s.low[1][i];
        EXPECT_NEAR(s.centre[i] - couplings, 1.2 * d[face[edge]] * area, 1e-15);
        EXPECT_EQ(edge % 2 == 0 ? s.low[direction][i] : s.high[direction][i], 0.0);
    }
}

// An outlet whose inward face has no diagonal borrows nothing.
TEST(PressureEquation, AnOutletBesideAFaceWithoutADiagonalHasNoCoefficient) {
    room r({1.0, 1.0}, {1.0, 1.0});
    r.outlets[0].values[0] = r.outlets[0].values[1] = true;
    r.outlets[2].values[0] = true;
    r.plant();
    r.a_u[r.u_face(0, 1)] = -1.0;
    r.a_v[r.v_face(1, 0)] = nan_value;
    const auto g = r.grid();
    corrector<2> c(g, r.input());
    c.assemble({r.a_u, r.a_v});
    EXPECT_EQ(c.face_d(0)[r.u_face(0, 0)], 0.0);
    EXPECT_EQ(c.face_d(0)[r.u_face(0, 1)], 0.0);
    EXPECT_GT(c.face_d(0)[r.u_face(1, 0)], 0.0);
    EXPECT_EQ(c.face_d(1)[r.v_face(0, 0)], 0.0);
    EXPECT_EQ(c.face_d(1)[r.v_face(1, 0)], 0.0);
}

TEST(PressureEquation, ASolidCellHasNoEquationAndNoImbalance) {
    room r(widths(9, 1), widths(6, 2));
    r.types[r.cell(2, 4)] = solid_type;
    r.plant();
    // Planted on its faces too, which a momentum prediction never does.
    r.a_u[r.u_face(2, 4)] = r.a_u[r.u_face(2, 5)] = 3.0;
    r.u[r.u_face(2, 5)] = 7.0;
    const auto g = r.grid();
    corrector<2> c(g, r.input());
    c.assemble({r.a_u, r.a_v});
    const auto& s = c.equation();
    const std::size_t at = r.cell(2, 4);
    EXPECT_EQ(s.centre[at], 0.0);
    for (int d = 0; d < 2; ++d) {
        EXPECT_EQ(s.high[d][at], 0.0);
        EXPECT_EQ(s.low[d][at], 0.0);
    }
    EXPECT_GT(s.high[0][r.cell(2, 3)], 0.0);
    std::vector<double> b(r.p.size(), nan_value);
    c.mass_imbalance({r.u, r.v}, b);
    EXPECT_EQ(b[at], 0.0);
    EXPECT_NE(b[r.cell(2, 5)], 0.0);
}

// ---------------------------------------------------------------------------
// The closed domain

TEST(PressureClosed, TheRightHandSideLosesItsMeanOverTheCellsWithAnEquation) {
    room r(widths(11, 3), widths(7, 4));
    r.types[r.cell(0, 0)] = solid_type;
    r.types[r.cell(3, 5)] = solid_type;
    r.plant();
    // A wall that leaks, so that the mean is not zero to begin with.
    r.u[r.u_face(4, 0)] = 0.3;
    const auto g = r.grid();
    corrector<2> c(g, r.input());
    ASSERT_TRUE(c.needs_pin());
    c.assemble({r.a_u, r.a_v});
    std::vector<double> b(r.p.size());
    std::vector<double> f(r.p.size(), nan_value);
    c.mass_imbalance({r.u, r.v}, b);
    c.right_hand_side({r.u, r.v}, f);
    double mean = 0.0;
    std::size_t count = 0;
    for (std::size_t i = 0; i < f.size(); ++i) {
        if (c.equation().centre[i] > 0.0) {
            mean -= b[i];
            ++count;
        } else {
            EXPECT_EQ(bits(f[i]), bits(0.0));
        }
    }
    EXPECT_EQ(count, f.size() - 2);
    mean /= static_cast<double>(count);
    EXPECT_GT(std::fabs(mean), 1e-4);
    for (std::size_t i = 0; i < f.size(); ++i) {
        if (c.equation().centre[i] > 0.0) {
            EXPECT_NEAR(f[i], -b[i] - mean, 1e-15);
        }
    }
    EXPECT_NEAR(std::accumulate(f.begin(), f.end(), 0.0), 0.0, 1e-14);
}

TEST(PressureClosed, ThePinIsTheFirstFluidCell) {
    room r(widths(6, 3), widths(5, 4));
    std::fill(r.types.begin(), r.types.begin() + 8, edge_type);
    r.types[8] = solid_type;
    r.types[9] = fluid_type;
    r.plant();
    const auto g = r.grid();
    corrector<2> c(g, r.input());
    EXPECT_EQ(c.pin_cell(), 9u);
    corrected out(r);
    ASSERT_EQ(out.run(c, r).status, solve_status::ok);
    EXPECT_EQ(out.p_prime[9], 0.0);
    EXPECT_EQ(out.p[9], 0.0);
    // Every cell with an equation moved with it; the SOLID cell kept its value.
    EXPECT_EQ(out.p[8], r.p[8]);
    EXPECT_NEAR(out.p[20] - (r.p[20] + 0.3 * out.p_prime[20]),
                -(r.p[9] + 0.3 * out.p_prime[9]), 1e-15);
    std::fill(r.types.begin(), r.types.end(), edge_type);
    const auto none = r.grid();
    EXPECT_EQ(corrector<2>(none, r.input()).pin_cell(), 0u);
}

// The corrected field leaves no imbalance, whatever the preconditioner, with
// SOLID cells and with outlets.
TEST(PressureCorrection, LeavesNoImbalance) {
    for (const bool open : {false, true}) {
        for (const auto method : {preconditioner::jacobi, preconditioner::multigrid}) {
            room r(widths(17, 3), widths(9, 4));
            r.types[r.cell(4, 4)] = r.types[r.cell(4, 5)] = solid_type;
            r.outlets[1].values[3] = open;
            r.method = method;
            r.plant();
            if (open) {
                r.u[r.u_face(2, 0)] = 0.5;  // an inlet
                r.u[r.u_face(3, r.nx)] = 0.1;
            }
            const auto g = r.grid();
            corrector<2> c(g, r.input());
            corrected out(r);
            const solve_result result = out.run(c, r);
            ASSERT_EQ(result.status, solve_status::ok);
            EXPECT_FALSE(result.reached_cap);
            EXPECT_GT(result.iterations, 2);
            std::vector<double> b(r.p.size());
            c.mass_imbalance({out.u, out.v}, b);
            for (const double value : b) {
                EXPECT_NEAR(value, 0.0, 1e-11);
            }
            // Walls, inlets and SOLID faces did not move.
            for (std::size_t j = 0; j < r.ny; ++j) {
                EXPECT_EQ(bits(out.u[r.u_face(j, 0)]), bits(r.u[r.u_face(j, 0)]));
            }
            EXPECT_EQ(out.u[r.u_face(4, 5)], 0.0);
            EXPECT_EQ(c.equation().centre[r.cell(4, 4)], 0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// The conjugate gradients

// A line of cells, coupled by 1, the first one to a face held at zero.
std::unique_ptr<system<1>> line_system(const layout<1>& fields) {
    auto s = std::make_unique<system<1>>(fields);
    const std::size_t n = fields.cell_count();
    for (std::size_t i = 0; i < n; ++i) {
        s->high[0][i] = i + 1 < n ? 1.0 : 0.0;
        s->low[0][i] = i > 0 ? 1.0 : 0.0;
        s->centre[i] = s->high[0][i] + s->low[0][i] + (i == 0 ? 1.0 : 0.0);
    }
    return s;
}

solve_keys cg_keys(preconditioner method) {
    solve_keys k;
    k.rtol = 1e-13;
    k.floor = 0.0;
    k.max_iter = 400;
    k.method = method;
    return k;
}

TEST(PressureConjugateGradient, SolvesALineExactly) {
    const std::size_t n = 37;
    const layout<1> fields({n});
    const auto s = line_system(fields);
    // x_i = i + 1 has A x = 0 but in the last cell, where it is 1.
    std::vector<double> f(n, 0.0);
    f[n - 1] = 1.0;
    for (const auto method : {preconditioner::jacobi, preconditioner::multigrid}) {
        solver<1> cg(fields, true, level_keys());
        std::vector<double> x(n, nan_value);
        const solve_result result = cg.solve(*s, f, cg_keys(method), x);
        EXPECT_EQ(result.status, solve_status::ok);
        EXPECT_FALSE(result.reached_cap);
        EXPECT_LE(result.iterations, 60);
        EXPECT_GE(result.products, result.iterations + 1);
        EXPECT_LE(result.residual_norm, 1e-13);
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(x[i], static_cast<double>(i + 1), 1e-9);
        }
    }
}

// The stop is read on the true residual: the recursion's is confirmed, and one
// more product is counted for each confirmation.
TEST(PressureConjugateGradient, StopsOnTheTrueResidual) {
    const std::size_t n = 64;
    const layout<1> fields({n});
    const auto s = line_system(fields);
    std::vector<double> f(n);
    for (std::size_t i = 0; i < n; ++i) {
        f[i] = (hashed(i) - 0.5) * std::pow(10.0, 8.0 * hashed(70 + i) - 4.0);
    }
    for (const double rtol : {1e-2, 1e-8, 1e-13, 2e-15}) {
        solve_keys k = cg_keys(preconditioner::jacobi);
        k.rtol = rtol;
        solver<1> cg(fields, false, level_keys());
        std::vector<double> x(n);
        const solve_result result = cg.solve(*s, f, k, x);
        double norm = 0.0;
        double f_norm = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double left = i > 0 ? x[i - 1] : 0.0;
            const double right = i + 1 < n ? x[i + 1] : 0.0;
            const double residual = f[i] - (s->centre[i] * x[i] - s->high[0][i] * right -
                                            s->low[0][i] * left);
            norm += residual * residual;
            f_norm += f[i] * f[i];
        }
        EXPECT_NEAR(result.residual_norm, std::sqrt(norm), 1e-9 * std::sqrt(norm));
        if (!result.reached_cap) {
            EXPECT_LE(result.residual_norm, rtol * std::sqrt(f_norm));
        }
        EXPECT_GE(result.products, result.iterations + 1);
    }
}

TEST(PressureConjugateGradient, TheCapTheFloorAndAZeroRightHandSide) {
    const std::size_t n = 50;
    const layout<1> fields({n});
    const auto s = line_system(fields);
    std::vector<double> f(n, 0.0);
    f[n - 1] = 1.0;
    solver<1> cg(fields, false, level_keys());
    std::vector<double> x(n, nan_value);
    solve_keys k = cg_keys(preconditioner::jacobi);
    k.max_iter = 3;
    solve_result result = cg.solve(*s, f, k, x);
    EXPECT_EQ(result.status, solve_status::ok);
    EXPECT_TRUE(result.reached_cap);
    EXPECT_EQ(result.iterations, 3);
    EXPECT_EQ(result.products, 4);
    EXPECT_GT(result.residual_norm, 0.0);
    // The floor alone ends a solve the relative level would not: the first
    // iteration of this one leaves a residual of 1 / sqrt(2) or so.
    k.max_iter = 400;
    k.rtol = 0.0;
    k.floor = 2.0;
    result = cg.solve(*s, f, k, x);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_EQ(result.products, 2);
    EXPECT_FALSE(result.reached_cap);
    // A zero right-hand side: nothing formed, and x is zero.
    std::fill(x.begin(), x.end(), nan_value);
    result = cg.solve(*s, std::vector<double>(n, 0.0), k, x);
    EXPECT_EQ(result.status, solve_status::ok);
    EXPECT_EQ(result.iterations, 0);
    EXPECT_EQ(result.products, 0);
    EXPECT_FALSE(result.reached_cap);
    EXPECT_EQ(result.residual_norm, 0.0);
    EXPECT_EQ(x, std::vector<double>(n, 0.0));
}

// Two cells and a right-hand side the operator cannot see: p . A p is zero.
TEST(PressureConjugateGradient, ADivisionByZeroIsAStatus) {
    const layout<1> fields({2});
    system<1> s(fields);
    s.centre = {1.0, 1.0};
    s.high[0] = {1.0, 0.0};
    s.low[0] = {0.0, 1.0};
    solver<1> cg(fields, false, level_keys());
    std::vector<double> x(2);
    const solve_result result =
        cg.solve(s, std::vector<double>{1.0, 1.0}, cg_keys(preconditioner::jacobi), x);
    EXPECT_EQ(result.status, solve_status::division_by_zero);
}

TEST(PressureConjugateGradient, ProjectsThePreconditionedVector) {
    // A closed line with a SOLID cell in its corner.
    const layout<2> fields({7, 4});
    system<2> s(fields);
    for (std::size_t j = 0; j < 4; ++j) {
        for (std::size_t i = 0; i < 7; ++i) {
            const std::size_t c = j * 7 + i;
            const bool solid = c == 0;
            if (i + 1 < 7 && !solid) {
                s.high[0][c] = 1.0;
                s.low[0][c + 1] = 1.0;
            }
            if (j + 1 < 4 && !solid) {
                s.high[1][c] = 1.0;
                s.low[1][c + 7] = 1.0;
            }
        }
    }
    for (std::size_t c = 0; c < 28; ++c) {
        s.centre[c] = s.high[0][c] + s.low[0][c] + s.high[1][c] + s.low[1][c];
    }
    std::vector<std::size_t> storage(28);
    const auto cells = cfdcore::pressure::cells_with_equation(s.centre, storage);
    ASSERT_EQ(cells.size(), 27u);
    EXPECT_EQ(cells.front(), 1u);
    solve_keys k = cg_keys(preconditioner::multigrid);
    solver<2> cg(fields, true, level_keys());
    std::vector<double> r(28);
    for (std::size_t c = 1; c < 28; ++c) {
        r[c] = hashed(c);
    }
    std::vector<double> plain(28);
    ASSERT_EQ(cg.prepare(s, k), solve_status::ok);
    cg.precondition(r, plain);
    k.project = cells;
    std::vector<double> projected(28);
    ASSERT_EQ(cg.prepare(s, k), solve_status::ok);
    cg.precondition(r, projected);
    const double mean = std::accumulate(plain.begin(), plain.end(), 0.0) / 27.0;
    EXPECT_GT(std::fabs(mean), 1e-3);
    EXPECT_EQ(projected[0], 0.0);
    for (std::size_t c = 1; c < 28; ++c) {
        EXPECT_NEAR(projected[c], plain[c] - mean, 1e-15);
    }
}

// ---------------------------------------------------------------------------
// The means

TEST(PressureMean, IsTheReferencesPairwiseMean) {
    for (const std::size_t n : {1u, 7u, 8u, 129u, 1000u, 4000u}) {
        std::vector<double> field(n);
        for (std::size_t i = 0; i < n; ++i) {
            field[i] = (hashed(i) - 0.5) * std::pow(10.0, 20.0 * hashed(3 * i) - 10.0);
        }
        std::vector<std::size_t> every(n);
        std::iota(every.begin(), every.end(), std::size_t{0});
        EXPECT_EQ(bits(cfdcore::pressure::mean_over(field, every)),
                  bits(cfdcore::reduce::pairwise_sum(field) / static_cast<double>(n)));
        std::vector<std::size_t> some;
        std::vector<double> taken;
        for (std::size_t i = 0; i < n; ++i) {
            if (hashed(91 * i) < 0.7 || i == 0) {
                some.push_back(i);
                taken.push_back(field[i]);
            }
        }
        EXPECT_EQ(bits(cfdcore::reduce::pairwise_sum_at(field, some)),
                  bits(cfdcore::reduce::pairwise_sum(taken)));
        EXPECT_EQ(bits(cfdcore::pressure::mean_over(field, some)),
                  bits(cfdcore::reduce::pairwise_sum(taken) / static_cast<double>(some.size())));
    }
}

// ---------------------------------------------------------------------------
// The components and the refusals

TEST(PressureComponents, AreCountedAndTheStrandedOnes) {
    room r(std::vector<double>(7, 1.0), std::vector<double>(5, 1.0));
    const auto count = [&] {
        const auto g = r.grid();
        return corrector<2>::count_components(g, r.input());
    };
    EXPECT_EQ(count().components, 1);
    EXPECT_EQ(count().stranded, 1);
    // A wall across the room.
    for (std::size_t j = 0; j < r.ny; ++j) {
        r.types[r.cell(j, 3)] = solid_type;
    }
    EXPECT_EQ(count().components, 2);
    r.outlets[1].values[2] = true;
    EXPECT_EQ(count().components, 2);
    EXPECT_EQ(count().stranded, 1);
    r.outlets[2].values[1] = true;
    EXPECT_EQ(count().stranded, 0);
    // A sealed single cell has no equation and is no component.
    r.types[r.cell(1, 4)] = r.types[r.cell(1, 5)] = r.types[r.cell(0, 6)] = solid_type;
    r.types[r.cell(0, 5)] = solid_type;
    r.types[r.cell(0, 4)] = fluid_type;
    EXPECT_EQ(count().components, 2);
    // Diagonal neighbours are not joined, and an outlet whose inward neighbour
    // is SOLID reaches nothing.
    room s(std::vector<double>(4, 1.0), std::vector<double>(4, 1.0));
    for (const auto& [j, i] : {std::pair<int, int>{0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 0}, {2, 1},
                               {3, 0}, {3, 1}}) {
        s.types[s.cell(static_cast<std::size_t>(j), static_cast<std::size_t>(i))] = solid_type;
    }
    s.outlets[0].values[0] = true;
    const auto g = s.grid();
    EXPECT_EQ(corrector<2>::count_components(g, s.input()).components, 2);
    EXPECT_EQ(corrector<2>::count_components(g, s.input()).stranded, 1);
    s.types[s.cell(0, 1)] = solid_type;
    const auto closer = s.grid();
    EXPECT_EQ(corrector<2>::count_components(closer, s.input()).stranded, 2);
}

TEST(PressureComponents, AnOutletOfAGridOneCellAcrossBorrowsNothing) {
    room r(std::vector<double>(5, 1.0), {1.0});
    for (std::size_t i = 0; i < 5; ++i) {
        r.outlets[2].values[i] = r.outlets[3].values[i] = true;
    }
    const auto g = r.grid();
    EXPECT_EQ(corrector<2>::count_components(g, r.input()).stranded, 1);
}

TEST(PressureCorrector, ProgrammingErrorsThrow) {
    room r(widths(5, 1), widths(4, 2));
    r.plant();
    const auto g = r.grid();
    const auto refused = [&](auto&& change) {
        corrector_input<2> in = r.input();
        change(in);
        EXPECT_FALSE(corrector<2>::check(g, in).empty());
        EXPECT_THROW(corrector<2>(g, in), std::invalid_argument);
    };
    const std::vector<double> four(4, 1.0);
    const std::vector<double> five(5, 1.0);
    refused([&](auto& in) { in.cell_widths[0] = four; });
    refused([&](auto& in) { in.cell_widths[1] = five; });
    for (const double bad : {0.0, -1.0, nan_value, infinity}) {
        std::vector<double> spoilt = r.dx;
        spoilt.back() = bad;
        refused([&](auto& in) { in.cell_widths[0] = spoilt; });
        refused([&](auto& in) { in.rho = bad; });
        refused([&](auto& in) { in.alpha = bad; });
    }
    const flags three(3);
    refused([&](auto& in) { in.outlet_low[0] = three.span(); });
    refused([&](auto& in) { in.outlet_low[1] = three.span(); });
    refused([&](auto& in) { in.outlet_high[0] = three.span(); });
    refused([&](auto& in) { in.outlet_high[1] = three.span(); });
    // The keys, whose messages the caller gives.
    const auto thrown = [&](auto&& change) {
        corrector_input<2> in = r.input();
        change(in);
        EXPECT_TRUE(corrector<2>::check(g, in).empty());
        EXPECT_THROW(corrector<2>(g, in), std::invalid_argument);
    };
    for (const double bad : {-1e-9, 1.0, nan_value}) {
        thrown([&](auto& in) { in.rtol = bad; });
    }
    for (const double bad : {-1e-30, infinity, nan_value}) {
        thrown([&](auto& in) { in.floor = bad; });
    }
    thrown([](auto& in) { in.max_iter = 0; });
    thrown([](auto& in) { in.levels.sweeps = 0; });

    corrector<2> c(g, r.input());
    std::vector<double> u = r.u, v = r.v, p = r.p, pp = r.p;
    std::vector<double> bad_u(u.size() + 1), bad_v(v.size() - 1), bad_p(p.size() + 1);
    EXPECT_THROW(c.assemble({bad_u, r.a_v}), std::invalid_argument);
    EXPECT_THROW(c.assemble({r.a_u, bad_v}), std::invalid_argument);
    c.assemble({r.a_u, r.a_v});
    EXPECT_THROW(c.mass_imbalance({bad_u, r.v}, p), std::invalid_argument);
    EXPECT_THROW(c.mass_imbalance({r.u, r.v}, bad_p), std::invalid_argument);
    EXPECT_THROW(c.right_hand_side({r.u, bad_v}, p), std::invalid_argument);
    EXPECT_THROW(c.right_hand_side({r.u, r.v}, bad_p), std::invalid_argument);
    EXPECT_THROW(c.apply({bad_u, r.v}, r.p, pp, {u, v}, p), std::invalid_argument);
    EXPECT_THROW(c.apply({r.u, r.v}, bad_p, pp, {u, v}, p), std::invalid_argument);
    EXPECT_THROW(c.apply({r.u, r.v}, r.p, bad_p, {u, v}, p), std::invalid_argument);
    EXPECT_THROW(c.apply({r.u, r.v}, r.p, pp, {bad_u, v}, p), std::invalid_argument);
    EXPECT_THROW(c.apply({r.u, r.v}, r.p, pp, {u, bad_v}, p), std::invalid_argument);
    EXPECT_THROW(c.apply({r.u, r.v}, r.p, pp, {u, v}, bad_p), std::invalid_argument);
}

TEST(PressureConjugateGradient, ProgrammingErrorsThrow) {
    const layout<1> fields({9});
    const layout<1> other({8});
    const auto s = line_system(fields);
    const auto elsewhere = line_system(other);
    solver<1> cg(fields, false, level_keys());
    std::vector<double> f(9, 1.0);
    std::vector<double> x(9);
    std::vector<double> wrong(8);
    EXPECT_THROW(cg.precondition(f, x), std::invalid_argument);  // before a prepare
    const solve_keys jacobi = cg_keys(preconditioner::jacobi);
    EXPECT_THROW((void)cg.solve(*s, wrong, jacobi, x), std::invalid_argument);
    EXPECT_THROW((void)cg.solve(*s, f, jacobi, wrong), std::invalid_argument);
    EXPECT_THROW((void)cg.solve(*elsewhere, f, jacobi, x), std::invalid_argument);
    EXPECT_THROW((void)cg.solve(*s, f, cg_keys(preconditioner::multigrid), x),
                 std::invalid_argument);
    ASSERT_EQ(cg.prepare(*s, jacobi), solve_status::ok);
    EXPECT_THROW(cg.precondition(wrong, x), std::invalid_argument);
    EXPECT_THROW(cg.precondition(f, wrong), std::invalid_argument);
    std::vector<std::size_t> few(8);
    EXPECT_THROW((void)cfdcore::pressure::cells_with_equation(s->centre, few),
                 std::invalid_argument);
    solve_keys bad = jacobi;
    bad.rtol = 1.0;
    EXPECT_THROW((void)cg.solve(*s, f, bad, x), std::invalid_argument);
    bad = jacobi;
    bad.floor = -1.0;
    EXPECT_THROW((void)cg.solve(*s, f, bad, x), std::invalid_argument);
    bad = jacobi;
    bad.max_iter = 0;
    EXPECT_THROW((void)cg.solve(*s, f, bad, x), std::invalid_argument);
}

// A system that is not weakly dominant is refused by the solve under multigrid.
TEST(PressureConjugateGradient, ASystemThatIsNotDominantIsAStatus) {
    const layout<1> fields({9});
    auto s = line_system(fields);
    s->centre[4] = 1.9;
    solver<1> cg(fields, true, level_keys());
    std::vector<double> f(9, 1.0);
    std::vector<double> x(9);
    EXPECT_EQ(cg.solve(*s, f, cg_keys(preconditioner::multigrid), x).status,
              solve_status::not_dominant);
    EXPECT_EQ(cg.solve(*s, f, cg_keys(preconditioner::jacobi), x).status, solve_status::ok);
}

// D = 1: a duct of cells between a wall and an outlet, the correction exact.
TEST(PressureOneDirection, ADuctsCorrection) {
    const std::size_t n = 6;
    std::vector<double> faces(n + 1), centres(n), width(n, 1.0);
    std::iota(faces.begin(), faces.end(), 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        centres[i] = static_cast<double>(i) + 0.5;
    }
    const std::vector<std::int32_t> types(n, fluid_type);
    const cfdcore::grid::grid<1> g(
        {.faces = {faces}, .centres = {centres}, .cell_type = types, .solid = solid_type});
    flags none(1), outlet(1);
    outlet.values[0] = true;
    corrector_input<1> in;
    in.cell_widths = {width};
    in.outlet_low = {none.span()};
    in.outlet_high = {outlet.span()};
    in.fluid = fluid_type;
    in.rho = 1.0;
    in.alpha = 0.5;
    in.rtol = 1e-13;
    in.max_iter = 100;
    in.method = preconditioner::multigrid;
    in.levels = level_keys();
    corrector<1> c(g, in);
    EXPECT_FALSE(c.needs_pin());
    std::vector<double> a(n + 1, 2.0);
    a[0] = a[n] = 0.0;
    std::vector<double> star{1.0, 0.2, -0.4, 3.0, 0.5, 0.0, 0.7};
    std::vector<double> u(n + 1), p(n, 4.0), next(n), p_prime(n);
    const solve_result result = c.correct({star}, {a}, p, {u}, next, p_prime);
    ASSERT_EQ(result.status, solve_status::ok);
    for (std::size_t i = 0; i < n; ++i) {
        // Every face carries the inlet's flow.
        EXPECT_NEAR(u[i + 1], 1.0, 1e-12);
        EXPECT_EQ(next[i], 4.0 + 0.5 * p_prime[i]);
    }
    EXPECT_EQ(u[0], 1.0);
}

}  // namespace
