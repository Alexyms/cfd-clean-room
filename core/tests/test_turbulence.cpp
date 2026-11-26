// The k-epsilon model's blocks on planted fields against their hand results: what
// needs no reference (ADR-017 D). Its parity is tests/test_core_turbulence.py's.

#include "cfdcore/turbulence.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace cfdcore::turbulence {
namespace {

constexpr double nan_value = std::numeric_limits<double>::quiet_NaN();

struct axis {
    explicit axis(const std::vector<double>& w) : widths(w) {
        for (const double x : w) {
            centres.push_back(faces.back() + 0.5 * x);
            faces.push_back(faces.back() + x);
        }
        between.push_back(centres.front() - faces.front());
        for (std::size_t i = 1; i < w.size(); ++i) {
            between.push_back(centres[i] - centres[i - 1]);
        }
        between.push_back(faces.back() - centres.back());
    }
    std::vector<double> faces{0.0};
    std::vector<double> centres;
    std::vector<double> widths;
    std::vector<double> between;
};

// A bool array a span can view.
struct flags {
    explicit flags(std::size_t n) : size(n), data(new bool[n]) {
        std::fill(data.get(), data.get() + n, false);
    }
    std::span<const bool> span() const { return {data.get(), size}; }
    bool& operator[](std::size_t i) { return data[i]; }
    std::size_t size;
    std::unique_ptr<bool[]> data;
};

constexpr constants standard{0.09, 1.44, 1.92, 1.0, 1.3, false, 0.0, 0.0};
constexpr constants rng{0.0845, 1.42, 1.68, 0.7194, 0.7194, true, 4.38, 0.012};

// A room of nx by ny cells, its model, its fields and its conditions: at rest,
// nothing held, one inflow everywhere.
struct room {
    room(const std::vector<double>& wx, const std::vector<double>& wy, constants variant = standard,
         const std::vector<std::size_t>& solid = {}, bool upwind = true,
         std::int64_t max_iter = 500, const std::vector<std::size_t>& vents_u = {})
        : x(wx), y(wy), nx(wx.size()), ny(wy.size()), types(nx * ny, 0), vent_u((nx + 1) * ny),
          vent_v(nx * (ny + 1)) {
        for (const std::size_t i : solid) {
            types[i] = 1;
        }
        for (const std::size_t f : vents_u) {
            vent_u[f] = true;
        }
        g = std::make_unique<grid::grid<2>>(
            grid::grid_input<2>{.faces = {x.faces, y.faces},
                                .centres = {x.centres, y.centres},
                                .cell_type = types,
                                .solid = 1});
        input.cell_widths = {x.widths, y.widths};
        input.face_widths = {x.between, y.between};
        if (!vents_u.empty()) {
            input.vents = {vent_u.span(), vent_v.span()};
        }
        input.variant = variant;
        input.cfl = 0.25;
        input.upwind = upwind;
        input.max_iter = max_iter;
        input.tol = 1e-12;
        input.nu = 0.01;
        m = std::make_unique<model<2>>(*g, input);
        const std::size_t n = nx * ny;
        u.assign((nx + 1) * ny, 0.0);
        v.assign(nx * (ny + 1), 0.0);
        k.assign(n, 1.0);
        eps.assign(n, 1.0);
        nu_t.assign(n, 0.0);
        for (const std::size_t i : solid) {
            k[i] = eps[i] = 0.0;
        }
        in_u.assign(u.size(), 1.0);
        in_v.assign(v.size(), 1.0);
        edge_x.assign(nx + 1, 0.0);
        edge_y.assign(ny + 1, 0.0);
        top = edge_x;
        eps_wall.assign(n, 0.0);
        production.assign(n, 0.0);
        held = std::make_unique<flags>(n);
        given = std::make_unique<flags>(n);
    }

    conditions<2> c() const {
        conditions<2> out;
        out.inflow_k = {in_u, in_v};
        out.inflow_eps = {in_u, in_v};
        out.low[0][0] = edge_x;
        out.high[0][0] = top;
        out.low[1][0] = edge_y;
        out.high[1][0] = edge_y;
        out.eps_held = held->span();
        out.eps_wall = eps_wall;
        out.production_given = given->span();
        out.production = production;
        return out;
    }
    state s() const { return {k, eps, nu_t}; }
    std::array<std::span<const double>, 2> velocity() const { return {u, v}; }

    axis x, y;
    std::size_t nx, ny;
    std::vector<std::int32_t> types;
    flags vent_u, vent_v;
    std::unique_ptr<grid::grid<2>> g;
    model_input<2> input;
    std::unique_ptr<model<2>> m;
    std::vector<double> u, v, k, eps, nu_t, in_u, in_v, edge_x, edge_y, top, eps_wall, production;
    std::unique_ptr<flags> held, given;
};

const std::vector<double> even4(4, 0.5);
const std::vector<double> even3(3, 0.25);

TEST(Turbulence, EddyViscosityIsZeroInSolidCells) {
    room r(even4, even3, standard, {5});
    r.k.assign(12, 2.0);
    r.eps.assign(12, 3.0);
    r.k[5] = r.eps[5] = 0.0;
    std::vector<double> out(12, nan_value);
    r.m->eddy_viscosity(r.k, r.eps, out);
    for (std::size_t i = 0; i < 12; ++i) {
        EXPECT_EQ(out[i], i == 5 ? 0.0 : 0.09 * 4.0 / 3.0);
    }
}

TEST(Turbulence, AShearFlowHasItsGradientAtEveryCorner) {
    // u = 3 y with the edges' own velocities: du/dy is 3 everywhere, S^2 is 9.
    room r({0.5, 0.7, 0.4, 0.9}, {0.2, 0.3, 0.25});
    for (std::size_t j = 0; j < 3; ++j) {
        for (std::size_t i = 0; i < 5; ++i) {
            r.u[j * 5 + i] = 3.0 * r.y.centres[j];
        }
    }
    r.top.assign(5, 3.0 * r.y.faces.back());
    std::vector<double> corners(4 * 5, nan_value);
    r.m->corner_gradient(0, 0, r.u, r.edge_x, r.top, corners);
    for (const double g : corners) {
        EXPECT_NEAR(g, 3.0, 1e-13);
    }
    std::vector<double> s2(12, nan_value);
    r.m->strain_squared(r.velocity(), r.c(), s2);
    for (const double x : s2) {
        EXPECT_NEAR(x, 9.0, 1e-12);
    }
}

TEST(Turbulence, AStretchingFlowHasTwiceItsDiagonalsSquared) {
    // u = 2 x, v = -2 y: S^2 = 2 (2^2 + 2^2) = 16, and no shear inside the room.
    room r({0.5, 0.7, 0.4, 0.9}, {0.2, 0.3, 0.25});
    for (std::size_t j = 0; j < 3; ++j) {
        for (std::size_t i = 0; i < 5; ++i) {
            r.u[j * 5 + i] = 2.0 * r.x.faces[i];
        }
    }
    for (std::size_t j = 0; j < 4; ++j) {
        for (std::size_t i = 0; i < 4; ++i) {
            r.v[j * 4 + i] = -2.0 * r.y.faces[j];
        }
    }
    r.edge_x.assign(5, 0.0);
    // The edges move with the flow along them.
    for (std::size_t i = 0; i < 5; ++i) {
        r.edge_x[i] = r.top[i] = 2.0 * r.x.faces[i];
    }
    for (std::size_t j = 0; j < 4; ++j) {
        r.edge_y[j] = -2.0 * r.y.faces[j];
    }
    std::vector<double> s2(12, nan_value);
    r.m->strain_squared(r.velocity(), r.c(), s2);
    for (const double x : s2) {
        EXPECT_NEAR(x, 16.0, 1e-12);
    }
}

TEST(Turbulence, AWallGivesTheFullShearOfTheHalfCellBesideIt) {
    // u = 1 in every cell between edges at rest: the corners on the edges see
    // 1 / (half a cell), the corners between none.
    room r(even4, even3);
    r.u.assign(r.u.size(), 1.0);
    std::vector<double> corners(4 * 5, nan_value);
    r.m->corner_gradient(0, 0, r.u, r.edge_x, r.edge_x, corners);
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(corners[i], 1.0 / 0.125);
        EXPECT_EQ(corners[5 + i], 0.0);
        EXPECT_EQ(corners[10 + i], 0.0);
        EXPECT_EQ(corners[15 + i], -1.0 / 0.125);
    }
}

TEST(Turbulence, AnObstaclesFaceIsZeroAtTheCornerItself) {
    // A SOLID cell in the middle of the bottom row: its two faces hold a planted
    // value, which is read as zero at the corner above them, so the gradient there
    // is that across the half cell over the obstacle. Below, where the domain's
    // edge and the obstacle's face coincide, it is zero.
    room r(even3, even3, standard, {1});
    r.u.assign(r.u.size(), 1.0);
    std::vector<double> corners(4 * 4, nan_value);
    r.m->corner_gradient(0, 0, r.u, r.edge_x, r.edge_x, corners);
    for (const std::size_t i : {std::size_t{1}, std::size_t{2}}) {
        EXPECT_EQ(corners[4 + i], 1.0 / 0.125);
        EXPECT_EQ(corners[i], 0.0);
    }
    EXPECT_EQ(corners[0], 1.0 / 0.125);
    EXPECT_EQ(corners[4], 0.0);
    // A tangential velocity on the edge beneath the obstacle moves nothing there.
    std::vector<double> moving(4, 5.0);
    r.m->corner_gradient(0, 0, r.u, moving, r.edge_x, corners);
    EXPECT_EQ(corners[1], 0.0);
    EXPECT_EQ(corners[0], (1.0 - 5.0) / 0.125);
}

TEST(Turbulence, AVentsFaceIsReadAsItIs) {
    // The face on the low side of the SOLID cell is a vent: it carries its
    // velocity, so the gradient over it is that of the two rows of faces.
    room r(even3, even3, standard, {4}, true, 500, {1 * 4 + 1});
    r.u.assign(r.u.size(), 1.0);
    r.u[1 * 4 + 1] = 4.0;
    std::vector<double> corners(4 * 4, nan_value);
    r.m->corner_gradient(0, 0, r.u, r.edge_x, r.edge_x, corners);
    EXPECT_EQ(corners[1 * 4 + 1], (4.0 - 1.0) / 0.25);
    // The other face of the SOLID cell is an obstacle's.
    EXPECT_EQ(corners[1 * 4 + 2], (0.0 - 1.0) / 0.125);
    EXPECT_EQ(r.m->live_faces(0)[5], lanes::all);
    EXPECT_EQ(r.m->inner_faces(0)[5], lanes::none);
    EXPECT_EQ(r.m->live_faces(0)[6], lanes::none);
    EXPECT_EQ(r.m->live_faces(0)[0], lanes::all);
    EXPECT_EQ(r.m->inner_faces(0)[0], lanes::none);
}

TEST(Turbulence, TheStandardTermsAreTheirFormulas) {
    room r(even4, even3, standard, {3});
    for (std::size_t i = 0; i < 12; ++i) {
        r.k[i] = 2.0 + static_cast<double>(i);
        r.eps[i] = 1.0 + static_cast<double>(i);
        r.nu_t[i] = 0.5;
    }
    r.k[3] = r.eps[3] = r.nu_t[3] = 0.0;
    r.u.assign(r.u.size(), 1.0);
    r.given->data[7] = true;
    r.production.assign(12, 11.0);
    std::vector<double> s2(12);
    r.m->strain_squared(r.velocity(), r.c(), s2);
    r.m->terms(r.s(), r.velocity(), r.c());
    for (std::size_t i = 0; i < 12; ++i) {
        const double p = i == 3 ? 0.0 : (i == 7 ? 11.0 : 0.5 * s2[i]);
        const double ratio = i == 3 ? 0.0 : r.eps[i] / r.k[i];
        EXPECT_EQ(r.m->production()[i], p);
        EXPECT_EQ(r.m->decay_k()[i], ratio);
        EXPECT_EQ(r.m->growth_eps()[i], 1.44 * ratio * p);
        EXPECT_EQ(r.m->decay_eps()[i], 1.92 * ratio);
        EXPECT_EQ(r.m->rng_r()[i], 0.0);
    }
}

TEST(Turbulence, RngsRSplitsByItsSign) {
    // eta below eta_0 makes R positive, above it negative: the two halves of a
    // strong shear over k / eps of two sizes.
    room r(even4, even3, rng);
    for (std::size_t j = 0; j < 3; ++j) {
        for (std::size_t i = 0; i < 5; ++i) {
            r.u[j * 5 + i] = 20.0 * r.y.centres[j];
        }
    }
    r.top.assign(5, 20.0 * r.y.faces.back());
    for (std::size_t i = 0; i < 12; ++i) {
        r.eps[i] = i % 2 == 0 ? 1.0 : 40.0;
    }
    r.m->terms(r.s(), r.velocity(), r.c());
    std::vector<double> s2(12);
    r.m->strain_squared(r.velocity(), r.c(), s2);
    for (std::size_t i = 0; i < 12; ++i) {
        const double eta = std::sqrt(s2[i]) / r.eps[i];
        const double cube = eta * eta * eta;
        const double expected = 0.0845 * cube * (1.0 - eta / 4.38) / (1.0 + 0.012 * cube) *
                                r.eps[i] * r.eps[i];
        const double value = r.m->rng_r()[i];
        EXPECT_NEAR(value, expected, 1e-13 * std::fabs(expected));
        EXPECT_EQ(value < 0.0, i % 2 == 0);
        const double growth = 1.42 * r.eps[i] * r.m->production()[i];
        const double decay = 1.68 * r.eps[i];
        EXPECT_EQ(r.m->growth_eps()[i], value < 0.0 ? growth - value : growth);
        EXPECT_EQ(r.m->decay_eps()[i], value > 0.0 ? decay + value / r.eps[i] : decay);
    }
}

TEST(Turbulence, TheStepIsTheCourantNumberOverTheRateAndCappedAtRest) {
    room r(even4, even3);
    // One face moves: its two cells have a rate, the others take their cap.
    r.u[1 * 5 + 2] = -3.0;
    r.v[1 * 4 + 1] = 0.5;
    ASSERT_TRUE(r.m->pseudo_time_step(r.velocity()));
    const auto dt = r.m->time_step();
    const double both = 0.25 / (3.0 / 0.5 + 0.5 / 0.25);
    const double x_only = 0.25 / (3.0 / 0.5);
    const double y_only = 0.25 / (0.5 / 0.25);
    const double cap = std::max({both, x_only, y_only});
    EXPECT_EQ(dt[1 * 4 + 1], both);
    EXPECT_EQ(dt[1 * 4 + 2], x_only);
    EXPECT_EQ(dt[0 * 4 + 1], y_only);
    EXPECT_EQ(dt[2 * 4 + 3], cap);
}

TEST(Turbulence, NothingThatMovesIsAStatus) {
    room r(even4, even3, standard, {5});
    // The faces of the SOLID cell carry nothing, whatever they hold.
    r.u[1 * 5 + 1] = 4.0;
    EXPECT_FALSE(r.m->pseudo_time_step(r.velocity()));
    std::vector<double> k(12), eps(12), nu_t(12);
    const auto result = r.m->step(r.s(), r.velocity(), r.c(), k, eps, nu_t);
    EXPECT_EQ(result.status, step_status::nothing_moves);
}

TEST(Turbulence, AUniformFieldInAUniformFlowIsAdvectedToItself) {
    for (const bool upwind : {true, false}) {
        room r(even4, even3, standard, {}, upwind);
        r.u.assign(r.u.size(), 0.7);
        ASSERT_TRUE(r.m->pseudo_time_step(r.velocity()));
        std::vector<double> out(12, nan_value);
        r.m->advected(r.k, {r.in_u, r.in_v}, out);
        EXPECT_EQ(out, r.k);
    }
}

TEST(Turbulence, AnInflowValueEntersWhereTheFluxDoes) {
    room r(even4, even3);
    r.u.assign(r.u.size(), 0.5);
    for (std::size_t j = 0; j < 3; ++j) {
        r.in_u[j * 5] = 3.0;
        r.in_u[j * 5 + 4] = 100.0;
    }
    ASSERT_TRUE(r.m->pseudo_time_step(r.velocity()));
    std::vector<double> out(12, nan_value);
    r.m->advected(r.k, {r.in_u, r.in_v}, out);
    for (std::size_t j = 0; j < 3; ++j) {
        EXPECT_EQ(out[j * 4], 1.0 + 0.25 * 2.0);
        EXPECT_EQ(out[j * 4 + 1], 1.0);
        EXPECT_EQ(out[j * 4 + 3], 1.0);
    }
}

TEST(Turbulence, TheExplicitFieldHoldsEpsWhereItIsHeld) {
    room r(even4, even3, standard, {2});
    r.u.assign(r.u.size(), 0.5);
    r.held->data[6] = true;
    r.eps_wall[6] = 42.0;
    r.production.assign(12, 2.0);
    for (std::size_t i = 0; i < 12; ++i) {
        r.given->data[i] = i != 2;
    }
    ASSERT_TRUE(r.m->pseudo_time_step(r.velocity()));
    r.m->terms(r.s(), r.velocity(), r.c());
    std::vector<double> k_star(12, nan_value), eps_star(12, nan_value);
    r.m->explicit_fields(r.s(), r.c(), k_star, eps_star);
    EXPECT_EQ(eps_star[6], 42.0);
    EXPECT_EQ(k_star[2], 0.0);
    EXPECT_EQ(eps_star[2], 0.0);
    // A cell away from the obstacle: one value carried through, and the growth.
    const double dt = r.m->time_step()[8];
    EXPECT_EQ(k_star[8], 1.0 + dt * 2.0);
    EXPECT_EQ(eps_star[8], 1.0 + dt * (1.44 * 2.0));
}

TEST(Turbulence, AConductanceIsTheHarmonicMeanOfTheTwoDiffusivities) {
    room r({0.5, 0.25, 1.0}, {0.2, 0.4}, standard, {2});
    r.nu_t = {0.1, 0.3, 0.0, 0.2, 0.4, 0.6};
    std::vector<double> g_u(8, nan_value), g_v(9, nan_value);
    r.m->conductances(quantity::eps, r.nu_t, {g_u, g_v});
    const auto gamma = [&](std::size_t i) { return 0.01 + r.nu_t[i] / 1.3; };
    const auto harmonic = [&](std::size_t a, double da, std::size_t b, double db, double area) {
        return area / (0.5 * da / gamma(a) + 0.5 * db / gamma(b));
    };
    EXPECT_NEAR(g_u[1], harmonic(0, 0.5, 1, 0.25, 0.2), 1e-15);
    EXPECT_NEAR(g_u[5], harmonic(3, 0.5, 4, 0.25, 0.4), 1e-15);
    EXPECT_NEAR(g_u[6], harmonic(4, 0.25, 5, 1.0, 0.4), 1e-15);
    EXPECT_NEAR(g_v[3], harmonic(0, 0.2, 3, 0.4, 0.5), 1e-15);
    // Domain faces and the SOLID cell's conduct nothing.
    for (const std::size_t f : {0, 2, 3, 4, 7}) {
        EXPECT_EQ(g_u[f], 0.0);
    }
    for (const std::size_t f : {0, 1, 2, 5, 6, 7, 8}) {
        EXPECT_EQ(g_v[f], 0.0);
    }
    // k's own sigma, and equal diffusivities return themselves.
    r.nu_t.assign(6, 0.25);
    r.m->conductances(quantity::k, r.nu_t, {g_u, g_v});
    EXPECT_EQ(g_u[5], 0.26 * 0.4 / 0.375);
}

TEST(Turbulence, AStepOfAUniformRoomIsTheDecayOfItsCells) {
    // Uniform k and eps in a uniform flow, no production: each cell's implicit
    // decay alone, since nothing is left to advect or diffuse.
    room r(even4, even3);
    r.u.assign(r.u.size(), 0.5);
    r.given->data[0] = false;
    for (std::size_t i = 0; i < 12; ++i) {
        r.given->data[i] = true;
    }
    std::vector<double> k(12, nan_value), eps(12, nan_value), nu_t(12, nan_value);
    const auto result = r.m->step(r.s(), r.velocity(), r.c(), k, eps, nu_t);
    ASSERT_EQ(result.status, step_status::ok);
    EXPECT_TRUE(result.solves[0].converged && result.solves[1].converged);
    const double dt = 0.25;
    for (std::size_t i = 0; i < 12; ++i) {
        EXPECT_NEAR(k[i], 1.0 / (1.0 + dt), 1e-11);
        EXPECT_NEAR(eps[i], 1.0 / (1.0 + dt * 1.92), 1e-11);
        EXPECT_EQ(nu_t[i], 0.09 * (k[i] * k[i]) / eps[i]);
    }
}

TEST(Turbulence, AStepHoldsEpsAndLeavesSolidCellsAtZero) {
    room r(even4, even3, rng, {5, 6});
    r.u.assign(r.u.size(), 0.5);
    r.held->data[0] = true;
    r.eps_wall[0] = 7.0;
    std::vector<double> k(12, nan_value), eps(12, nan_value), nu_t(12, nan_value);
    const auto result = r.m->step(r.s(), r.velocity(), r.c(), k, eps, nu_t);
    ASSERT_EQ(result.status, step_status::ok);
    EXPECT_EQ(eps[0], 7.0);
    for (const std::size_t i : {5, 6}) {
        EXPECT_EQ(k[i], 0.0);
        EXPECT_EQ(eps[i], 0.0);
        EXPECT_EQ(nu_t[i], 0.0);
    }
}

TEST(Turbulence, ASolveAtItsCapFallsBackAndStaysPositive) {
    room r(even4, even3, standard, {}, true, 1);
    r.u.assign(r.u.size(), 0.5);
    for (std::size_t i = 0; i < 12; ++i) {
        r.k[i] = std::pow(10.0, static_cast<double>(i % 5) - 4.0);
        r.nu_t[i] = 1.0;
    }
    std::vector<double> k(12), eps(12), nu_t(12);
    const auto result = r.m->step(r.s(), r.velocity(), r.c(), k, eps, nu_t);
    ASSERT_EQ(result.status, step_status::ok);
    EXPECT_EQ(result.solves[0].fallbacks, 1);
    EXPECT_EQ(result.solves[0].iterations, 1);
    EXPECT_FALSE(result.solves[0].converged);
    EXPECT_TRUE(std::all_of(k.begin(), k.end(), [](double q) { return q > 0.0; }));
}

TEST(Turbulence, ANegativeInflowLeavesABadCellItsStatusNames) {
    // The checks are the caller's: an inflow value below zero makes k negative in
    // the cells it enters, and the status names the first of them.
    for (const bool first : {true, false}) {
        room r(even4, even3);
        r.u.assign(r.u.size(), 0.5);
        auto c = r.c();
        std::vector<double> negative(r.in_u);
        negative[5] = -50.0;
        (first ? c.inflow_k : c.inflow_eps)[0] = negative;
        std::vector<double> k(12), eps(12), nu_t(12);
        const auto result = r.m->step(r.s(), r.velocity(), c, k, eps, nu_t);
        EXPECT_EQ(result.status,
                  first ? step_status::k_not_positive : step_status::eps_not_positive);
        EXPECT_EQ(result.bad.first, 4u);
        EXPECT_LT(result.bad.value, 0.0);
        EXPECT_EQ(result.bad.least, result.bad.value);
        const auto& bad = first ? k : eps;
        EXPECT_EQ(result.bad.count,
                  static_cast<std::size_t>(std::count_if(bad.begin(), bad.end(),
                                                         [](double q) { return !(q > 0.0); })));
    }
}

TEST(Turbulence, TheRangeChecksReadNonSolidCellsOnly) {
    room r(even4, even3, standard, {5});
    std::vector<double> q(12, 1.0);
    for (const double bad : {nan_value, -1.0, 0.0, std::numeric_limits<double>::infinity()}) {
        q.assign(12, 1.0);
        q[5] = bad;
        EXPECT_TRUE(r.m->in_range(q, true));
        q[5] = 1.0;
        q[11] = bad;
        EXPECT_FALSE(r.m->in_range(q, true));
        EXPECT_EQ(r.m->in_range(q, false), bad == 0.0);
    }
    q.assign(12, 1.0);
    EXPECT_FALSE(r.m->zero_in_solid(q));
    q[5] = -0.0;
    EXPECT_TRUE(r.m->zero_in_solid(q));
    q[5] = nan_value;
    EXPECT_FALSE(r.m->zero_in_solid(q));
    EXPECT_TRUE(all_finite(std::vector<double>(9, -1.0)));
    for (const std::size_t i : {0, 7, 8}) {
        for (const double bad : {nan_value, -std::numeric_limits<double>::infinity()}) {
            std::vector<double> x(9, 1.0);
            x[i] = bad;
            EXPECT_FALSE(all_finite(x));
        }
    }
}

TEST(Turbulence, AnInputItCannotBeBuiltFromIsRefused) {
    room r(even4, even3);
    const auto refused = [&](auto&& spoil) {
        model_input<2> input = r.input;
        spoil(input);
        EXPECT_FALSE(model<2>::check(*r.g, input).empty());
        EXPECT_THROW(model<2>(*r.g, input), std::invalid_argument);
    };
    const std::vector<double> three(3, 0.5);
    const std::vector<double> zero(4, 0.0);
    const flags vents(5 * 3);
    EXPECT_TRUE(model<2>::check(*r.g, r.input).empty());
    refused([&](auto& in) { in.cell_widths[0] = three; });
    refused([&](auto& in) { in.cell_widths[0] = zero; });
    refused([&](auto& in) { in.face_widths[1] = three; });
    refused([&](auto& in) { in.vents[0] = vents.span(); });
    refused([&](auto& in) { in.vents[1] = vents.span(); });
    refused([&](auto& in) { in.cfl = 0.0; });
    refused([&](auto& in) { in.nu = nan_value; });
    refused([&](auto& in) { in.tol = 0.0; });
    refused([&](auto& in) { in.tol = -1.0e-10; });
    refused([&](auto& in) { in.tol = nan_value; });
    refused([&](auto& in) { in.tol = std::numeric_limits<double>::infinity(); });
    refused([&](auto& in) { in.max_iter = 0; });
    refused([&](auto& in) { in.max_iter = -5; });
    refused([&](auto& in) { in.variant.sigma_e = -1.3; });
    refused([&](auto& in) { in.variant.has_r = true; });
}

TEST(Turbulence, AnArrayOfAnotherLengthIsRefused) {
    room r(even4, even3);
    std::vector<double> a(12), b(12), c(12), one(11);
    const auto velocity = r.velocity();
    EXPECT_THROW(r.m->eddy_viscosity(one, r.eps, a), std::invalid_argument);
    EXPECT_THROW(r.m->pseudo_time_step({one, r.v}), std::invalid_argument);
    EXPECT_THROW(r.m->strain_squared(velocity, r.c(), one), std::invalid_argument);
    EXPECT_THROW(r.m->corner_gradient(0, 0, r.u, one, r.top, a), std::invalid_argument);
    EXPECT_THROW(r.m->corner_gradient(2, 0, r.u, r.edge_x, r.top, a), std::invalid_argument);
    EXPECT_THROW(r.m->corner_gradient(0, 1, r.u, r.edge_x, r.top, a), std::invalid_argument);
    EXPECT_THROW(r.m->advected(one, {r.in_u, r.in_v}, a), std::invalid_argument);
    EXPECT_THROW(r.m->advected(r.k, {one, r.in_v}, a), std::invalid_argument);
    EXPECT_THROW(r.m->conductances(quantity::k, one, {a, b}), std::invalid_argument);
    EXPECT_THROW(r.m->conductances(quantity::k, r.nu_t, {a, b}), std::invalid_argument);
    EXPECT_THROW(r.m->in_range(one, true), std::invalid_argument);
    EXPECT_THROW(r.m->zero_in_solid(one), std::invalid_argument);
    EXPECT_THROW(r.m->explicit_fields(r.s(), r.c(), a, one), std::invalid_argument);
    EXPECT_THROW(r.m->step(r.s(), velocity, r.c(), a, b, one), std::invalid_argument);
    EXPECT_THROW(r.m->step({one, r.eps, r.nu_t}, velocity, r.c(), a, b, c),
                 std::invalid_argument);
    const auto spoiled = [&](auto&& spoil) {
        auto conditions = r.c();
        spoil(conditions);
        EXPECT_THROW(r.m->terms(r.s(), velocity, conditions), std::invalid_argument);
    };
    const flags short_flags(11);
    spoiled([&](auto& x) { x.inflow_k[0] = one; });
    spoiled([&](auto& x) { x.inflow_eps[1] = one; });
    spoiled([&](auto& x) { x.low[0][0] = one; });
    spoiled([&](auto& x) { x.high[1][0] = one; });
    spoiled([&](auto& x) { x.eps_wall = one; });
    spoiled([&](auto& x) { x.production = one; });
    spoiled([&](auto& x) { x.eps_held = short_flags.span(); });
    spoiled([&](auto& x) { x.production_given = short_flags.span(); });
}

TEST(Turbulence, OneDirectionHasNoShear) {
    const axis x({0.5, 0.25, 1.0});
    const std::vector<std::int32_t> types(3, 0);
    const grid::grid<1> g(grid::grid_input<1>{
        .faces = {x.faces}, .centres = {x.centres}, .cell_type = types, .solid = 1});
    model_input<1> input{};
    input.cell_widths = {x.widths};
    input.face_widths = {x.between};
    input.variant = standard;
    input.cfl = 0.5;
    input.max_iter = 100;
    input.tol = 1e-12;
    input.nu = 0.01;
    model<1> m(g, input);
    const std::vector<double> u{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> inflow(4, 1.0), ones(3, 1.0), zeros(3, 0.0);
    const flags none(3);
    conditions<1> c;
    c.inflow_k = {inflow};
    c.inflow_eps = {inflow};
    c.eps_held = none.span();
    c.production_given = none.span();
    c.eps_wall = zeros;
    c.production = zeros;
    std::vector<double> s2(3);
    m.strain_squared({u}, c, s2);
    EXPECT_EQ(s2[0], 2.0 * 4.0);
    EXPECT_EQ(s2[1], 2.0 * 16.0);
    EXPECT_EQ(s2[2], 2.0 * 1.0);
    std::vector<double> k(3), eps(3), nu_t(3);
    const auto result = m.step({ones, ones, zeros}, {u}, c, k, eps, nu_t);
    EXPECT_EQ(result.status, step_status::ok);
    EXPECT_EQ(m.time_step()[0], 0.5 / (1.0 / 0.5));
}

}  // namespace
}  // namespace cfdcore::turbulence
