// The wall functions and the turbulence boundary on planted fields against their
// hand results: what needs no reference (ADR-017 D). Their parity is
// tests/test_core_turbulence.py's.

#include "cfdcore/walls.hpp"

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

namespace cfdcore::walls {
namespace {

constexpr double kappa = 0.41;
constexpr double e_wall = 9.793;
constexpr double nan_value = std::numeric_limits<double>::quiet_NaN();

wall_law law() { return {kappa, e_wall, log_law_floor(kappa, e_wall).y}; }

// A bool array a span can view.
struct flags {
    explicit flags(std::size_t n, bool value = false) : size(n), data(new bool[n]) {
        std::fill(data.get(), data.get() + n, value);
    }
    std::span<const bool> span() const { return {data.get(), size}; }
    std::size_t size;
    std::unique_ptr<bool[]> data;
};

// A room of 4 by 3 unit cells whose four edges are walls at rest, one cell SOLID.
// The edges' order is bottom, top, left, right.
struct room {
    static constexpr std::size_t nx = 4;
    static constexpr std::size_t ny = 3;

    explicit room(const std::vector<std::size_t>& solid = {}) : types(nx * ny, 0) {
        for (const std::size_t i : solid) {
            types[i] = 1;
        }
        g = std::make_unique<grid::grid<2>>(grid::grid_input<2>{.faces = {faces_x, faces_y},
                                                                .centres = {centres_x, centres_y},
                                                                .cell_type = types,
                                                                .solid = 1});
        for (int e = 0; e < 4; ++e) {
            const std::size_t n = e < 2 ? nx : ny;
            cells.emplace_back(n, true);
            corners.emplace_back(n + 1, true);
            known.emplace_back(n + 1, true);
            value.emplace_back(n + 1, 0.0);
        }
        u.assign((nx + 1) * ny, 0.0);
        v.assign(nx * (ny + 1), 0.0);
        k.assign(nx * ny, 0.04);
        eps.assign(nx * ny, 0.5);
        for (const std::size_t i : solid) {
            k[i] = eps[i] = 0.0;
        }
    }

    boundary_input<2> input() {
        boundary_input<2> in;
        in.wall_cells_low = {cells[2].span(), cells[0].span()};
        in.wall_cells_high = {cells[3].span(), cells[1].span()};
        in.wall_corners_low[0][0] = corners[0].span();
        in.wall_corners_high[0][0] = corners[1].span();
        in.wall_corners_low[1][0] = corners[2].span();
        in.wall_corners_high[1][0] = corners[3].span();
        in.low[0][0] = {known[0].span(), value[0], 0.5};
        in.high[0][0] = {known[1].span(), value[1], 0.5};
        in.low[1][0] = {known[2].span(), value[2], 0.5};
        in.high[1][0] = {known[3].span(), value[3], 0.5};
        in.edge_inlets = inlets;
        in.vent_inlets = vent_inlets;
        in.exhausts = exhausts;
        if (vents) {
            in.vents = {vent_u.span(), vent_v.span()};
            in.vent_corners[0][0] = corners_u.span();
            in.vent_corners[1][0] = corners_v.span();
        }
        in.c_mu = 0.09;
        in.rho = 1.2;
        in.nu = 1.5e-5;
        in.law = law();
        return in;
    }
    boundary<2> build() { return boundary<2>(*g, input()); }
    std::array<std::span<const double>, 2> velocity() const { return {u, v}; }

    std::vector<double> faces_x{0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> faces_y{0.0, 1.0, 2.0, 3.0};
    std::vector<double> centres_x{0.5, 1.5, 2.5, 3.5};
    std::vector<double> centres_y{0.5, 1.5, 2.5};
    std::vector<std::int32_t> types;
    std::unique_ptr<grid::grid<2>> g;
    std::vector<flags> cells, corners, known;
    std::vector<std::vector<double>> value;
    std::vector<inlet_face> inlets, vent_inlets;
    std::vector<exhaust_face> exhausts;
    bool vents = false;
    flags vent_u{(nx + 1) * ny}, vent_v{nx * (ny + 1)};
    flags corners_u{(nx + 1) * (ny + 1)}, corners_v{(nx + 1) * (ny + 1)};
    std::vector<double> u, v, k, eps;
};

// The wall function's own numbers at a node.
double u_k_of(double k) { return std::pow(0.09, 0.25) * std::sqrt(k); }
double log_term(double k, double y) {
    return std::log(e_wall * std::max(u_k_of(k) * y / 1.5e-5, law().floor));
}

TEST(Walls, TheFloorIsWhereTheTwoLawsMeet) {
    const floor_result r = log_law_floor(kappa, e_wall);
    ASSERT_EQ(r.status, floor_status::ok);
    EXPECT_NEAR(r.y, std::log(e_wall * r.y) / kappa, 1e-14);
    EXPECT_NEAR(r.y, 11.53, 0.005);
    EXPECT_EQ(log_law_floor(1.0, 1.0).status, floor_status::no_meeting);
    EXPECT_EQ(log_law_floor(0.41, 0.41).status, floor_status::no_meeting);
}

TEST(Walls, TheWallViscosityIsAirsAtTheFloorAndAboveItBeyond) {
    const double nu = 1.5e-5;
    const double y = 0.01;
    const double at_floor = std::pow(law().floor * nu / y / std::pow(0.09, 0.25), 2.0);
    const std::vector<double> k{at_floor, 100.0 * at_floor, 0.01 * at_floor};
    const std::vector<double> ys(3, y);
    std::vector<double> mu(3, nan_value);
    wall_viscosity(k, ys, 0.09, 1.2, nu, law(), mu);
    EXPECT_NEAR(mu[0], 1.2 * nu, 1e-16);
    EXPECT_GT(mu[1], 2.0 * mu[0]);
    EXPECT_NEAR(mu[2], 0.1 * mu[0], 1e-16);
    std::vector<double> two(2);
    EXPECT_THROW(wall_viscosity(k, ys, 0.09, 1.2, nu, law(), two), std::invalid_argument);
    EXPECT_THROW(wall_viscosity(k, two, 0.09, 1.2, nu, law(), mu), std::invalid_argument);
}

TEST(Walls, TheInflowValuesTakeTheSpeedsMagnitude) {
    const carried_in value = inflow_values(0.05, -0.4, 0.1);
    EXPECT_NEAR(value.k, 1.5 * 0.02 * 0.02, 1e-18);
    EXPECT_NEAR(value.eps, std::pow(value.k, 1.5) / 0.1, 1e-16);
    EXPECT_EQ(inflow_values(0.05, 0.4, 0.1).k, value.k);
}

TEST(Walls, AnObstaclesAndAnEdgesFacesAreWallsWhereThereIsAnUnknown) {
    room r({5});  // the cell in the middle row, second column
    auto b = r.build();
    // u's stencil: the interior faces of the bottom and the top row have the
    // edges' corners, and the faces above and below those of the obstacle its.
    const auto& u = b.stencil(0, 0);
    const auto is_wall = [](const stencil_walls& s, std::size_t corner) {
        return s.mask[corner] == lanes::all;
    };
    for (std::size_t i = 1; i < 4; ++i) {
        EXPECT_TRUE(is_wall(u, i));
        EXPECT_TRUE(is_wall(u, 15 + i));
    }
    // Faces x = 1 and x = 2 of the middle row bound the obstacle.
    for (const std::size_t i : {1, 2}) {
        EXPECT_TRUE(is_wall(u, 5 + i));
        EXPECT_TRUE(is_wall(u, 10 + i));
    }
    EXPECT_FALSE(is_wall(u, 5 + 3));
    EXPECT_EQ(u.corner.size(), 10u);
    for (const double d : u.distance) {
        EXPECT_EQ(d, 0.5);
    }
    // v's: the interior faces of the first and last column, and those beside it.
    const auto& v = b.stencil(1, 0);
    for (const std::size_t j : {1, 2}) {
        EXPECT_TRUE(is_wall(v, 5 * j));
        EXPECT_TRUE(is_wall(v, 5 * j + 4));
        EXPECT_TRUE(is_wall(v, 5 * j + 1));
        EXPECT_TRUE(is_wall(v, 5 * j + 2));
        EXPECT_FALSE(is_wall(v, 5 * j + 3));
    }
    EXPECT_EQ(v.corner.size(), 8u);
}

TEST(Walls, AnInletsCornersAndAVentsSpanAreNotWalls) {
    room r({5});
    r.corners[0].data[2] = false;
    r.vents = true;
    r.vent_v.data[2 * 4 + 1] = true;
    r.corners_u.data[10 + 1] = true;
    auto b = r.build();
    const auto& u = b.stencil(0, 0);
    EXPECT_EQ(u.mask[2], lanes::none);
    EXPECT_EQ(u.mask[10 + 1], lanes::none);
    EXPECT_EQ(u.mask[10 + 2], lanes::all);
    EXPECT_EQ(u.corner.size(), 8u);
}

TEST(Walls, TheWallViscosityIsPlantedAtTheWallFacesOnly) {
    room r({5});
    r.k[0] = 0.01;
    r.k[1] = 0.03;
    auto b = r.build();
    const std::vector<double> u(20, 7.0), v(20, 8.0);
    std::vector<double> out_u(20, nan_value), out_v(20, nan_value);
    b.wall_viscosity(r.k, {{{u}, {v}}}, {{{out_u}, {out_v}}});
    // The face at x = 1 of the bottom row lies between cells 0 and 1.
    const double k_p = 0.5 * (0.01 + 0.03);
    EXPECT_NEAR(out_u[1], 1.2 * u_k_of(k_p) * kappa * 0.5 / log_term(k_p, 0.5), 1e-15);
    EXPECT_EQ(out_u[0], 7.0);
    EXPECT_EQ(out_u[5 + 3], 7.0);
    EXPECT_EQ(out_v[5 + 3], 8.0);
    EXPECT_NE(out_v[5], 8.0);
}

TEST(Walls, AWallCellTakesTheMeanOfItsWalls) {
    room r;
    // The bottom edge moves at 2, the left one is at rest.
    r.value[0].assign(5, 2.0);
    r.u.assign(r.u.size(), 3.0);
    r.v.assign(r.v.size(), -0.5);
    auto b = r.build();
    const auto c = b.conditions(r.velocity(), r.k, r.eps);
    const double u_k = u_k_of(0.04);
    const double held = u_k * u_k * u_k / (kappa * 0.5);
    const auto production = [&](double slip) {
        return u_k * u_k * slip / (0.5 * log_term(0.04, 0.5));
    };
    // The corner cell: the bottom's slip is 3 - 2, the left's 0.5.
    EXPECT_TRUE(c.eps_held[0] && c.production_given[0]);
    EXPECT_NEAR(c.eps_wall[0], held, 1e-15);
    EXPECT_NEAR(c.production[0], 0.5 * (production(1.0) + production(0.5)), 1e-16);
    // A cell of the bottom row has one wall, one of the top row its own.
    EXPECT_NEAR(c.production[1], production(1.0), 1e-16);
    EXPECT_NEAR(c.production[9], production(3.0), 1e-16);
    // The middle row's inner cells have none.
    EXPECT_FALSE(c.eps_held[5]);
    EXPECT_EQ(c.eps_wall[5], 0.0);
    EXPECT_EQ(c.production[5], 0.0);
    EXPECT_FALSE(b.wall_cells()[6]);
}

TEST(Walls, AFaceOfASolidCellHasNoSpeedAndAVentsHasItsOwn) {
    // The obstacle at cell 5: the cell right of it has its low u face read as
    // zero, unless that face is a vent, which is then no wall of the cell either.
    for (const bool vent : {false, true}) {
        room r({5});
        r.u.assign(r.u.size(), 2.0);
        r.cells[1].data[2] = false;
        r.vents = true;
        r.vent_u.data[1 * 5 + 2] = vent;
        r.vent_v.data[2 * 4 + 1] = true;
        auto b = r.build();
        const auto c = b.conditions(r.velocity(), r.k, r.eps);
        EXPECT_EQ(c.eps_held[6], !vent);
        // Cell 1, under the obstacle, has the bottom edge and the obstacle: its
        // own faces carry 2.
        EXPECT_TRUE(c.eps_held[1]);
        // Cell 9 over the vent of the obstacle's top has the top edge alone.
        const double u_k = u_k_of(0.04);
        EXPECT_TRUE(c.eps_held[9]);
        EXPECT_NEAR(c.production[9], u_k * u_k * 2.0 / (0.5 * log_term(0.04, 0.5)), 1e-15);
    }
}

TEST(Walls, TheTangentialVelocityIsTheConditionsOrTheAdjacentFaces) {
    room r;
    // Each edge's values lie beyond every face velocity below (u is 0 to 14, v is
    // 100 to 115), so an edge that read the wrong one of the two shows on every
    // edge: the top edge's own value once equalled the face beside it.
    const std::array<double, 4> edge_value{1000.0, 1001.0, 1002.0, 1003.0};
    for (int e = 0; e < 4; ++e) {
        const auto edge = static_cast<std::size_t>(e);
        r.value[edge].assign(e < 2 ? 5 : 4, edge_value[edge]);
        r.known[edge].data[1] = false;
    }
    for (std::size_t i = 0; i < r.u.size(); ++i) {
        r.u[i] = static_cast<double>(i);
    }
    for (std::size_t i = 0; i < r.v.size(); ++i) {
        r.v[i] = 100.0 + static_cast<double>(i);
    }
    auto b = r.build();
    const auto c = b.conditions(r.velocity(), r.k, r.eps);
    // The open face of each edge takes the face of the room beside it.
    EXPECT_EQ(c.low[0][0][1], r.u[1]);
    EXPECT_EQ(c.high[0][0][1], r.u[2 * 5 + 1]);
    EXPECT_EQ(c.low[1][0][1], r.v[1 * 4]);
    EXPECT_EQ(c.high[1][0][1], r.v[1 * 4 + 3]);
    // Every other face of an edge, the ends among them, takes the edge's value.
    for (const std::size_t i : {0u, 2u, 3u, 4u}) {
        EXPECT_EQ(c.low[0][0][i], edge_value[0]) << i;
        EXPECT_EQ(c.high[0][0][i], edge_value[1]) << i;
    }
    for (const std::size_t i : {0u, 2u, 3u}) {
        EXPECT_EQ(c.low[1][0][i], edge_value[2]) << i;
        EXPECT_EQ(c.high[1][0][i], edge_value[3]) << i;
    }
}

TEST(Walls, AnInflowIsTheInletsOrTheCellsOrTheLargest) {
    room r({0});
    r.inlets.push_back({1, 3 * 4 + 2, 0.05, -0.4, 0.1, 1.0});
    r.k[3] = 0.09;
    auto b = r.build();
    const auto c = b.conditions(r.velocity(), r.k, r.eps);
    const carried_in in = inflow_values(0.05, -0.4, 0.1);
    EXPECT_EQ(c.inflow_k[1][3 * 4 + 2], in.k);
    EXPECT_EQ(c.inflow_eps[1][3 * 4 + 2], in.eps);
    EXPECT_EQ(c.inflow_k[1][3], 0.09);
    EXPECT_EQ(c.inflow_k[0][4], 0.09);
    EXPECT_EQ(c.inflow_k[1][3 * 4 + 1], 0.04);
    // Beside the SOLID corner cell the largest value stands in.
    EXPECT_EQ(c.inflow_k[1][0], 0.09);
    EXPECT_EQ(c.inflow_k[0][0], 0.09);
    EXPECT_EQ(c.inflow_eps[1][0], 0.5);
    // No interior face has one.
    EXPECT_EQ(c.inflow_k[0][1], 0.0);
}

TEST(Walls, AVentsInflowIsItsOwnAndAnExhaustsTheRoomCells) {
    room r({5});
    r.vents = true;
    r.vent_u.data[1 * 5 + 1] = true;
    r.vent_u.data[1 * 5 + 2] = true;
    r.vent_v.data[1 * 4 + 1] = true;
    r.vent_inlets.push_back({0, 1 * 5 + 1, 0.2, -0.3, 0.02, 1.0});
    r.exhausts.push_back({0, 1 * 5 + 2});
    r.exhausts.push_back({1, 1 * 4 + 1});
    r.k[6] = 0.6;
    r.k[1] = 0.7;
    auto b = r.build();
    const auto c = b.conditions(r.velocity(), r.k, r.eps);
    EXPECT_EQ(c.inflow_k[0][1 * 5 + 1], inflow_values(0.2, -0.3, 0.02).k);
    EXPECT_EQ(c.inflow_k[0][1 * 5 + 2], 0.6);
    EXPECT_EQ(c.inflow_k[1][1 * 4 + 1], 0.7);
}

TEST(Walls, TheInitialValuesAreTheInletsWeightedMeans) {
    room r;
    EXPECT_FALSE(r.build().initial_values().has_value());
    r.inlets.push_back({1, 3 * 4, 0.05, -0.4, 0.1, 2.0});
    r.vent_inlets.push_back({0, 0, 0.1, 0.8, 0.2, 0.5});
    const auto values = r.build().initial_values();
    ASSERT_TRUE(values.has_value());
    const carried_in a = inflow_values(0.05, -0.4, 0.1);
    const carried_in b = inflow_values(0.1, 0.8, 0.2);
    EXPECT_NEAR(values->k, (a.k * 0.8 + b.k * 0.4) / 1.2, 1e-16);
    EXPECT_NEAR(values->eps, (a.eps * 0.8 + b.eps * 0.4) / 1.2, 1e-16);
}

TEST(Walls, AnInputItCannotBeBuiltFromIsRefused) {
    const auto refused = [](auto&& spoil) {
        room r;
        spoil(r);
        EXPECT_FALSE(boundary<2>::check(*r.g, r.input()).empty());
        EXPECT_THROW(r.build(), std::invalid_argument);
    };
    EXPECT_TRUE(boundary<2>::check(*room().g, room().input()).empty());
    refused([](room& r) { r.cells[0] = flags(3); });
    refused([](room& r) { r.cells[3] = flags(4); });
    refused([](room& r) { r.corners[1] = flags(4); });
    refused([](room& r) { r.value[2].pop_back(); });
    refused([](room& r) { r.known[2] = flags(3); });
    refused([](room& r) { r.inlets.push_back({1, 16, 0.05, -0.4, 0.1, 1.0}); });
    refused([](room& r) { r.inlets.push_back({2, 0, 0.05, -0.4, 0.1, 1.0}); });
    refused([](room& r) { r.inlets.push_back({1, 0, nan_value, -0.4, 0.1, 1.0}); });
    refused([](room& r) { r.vent_inlets.push_back({1, 0, 0.05, -0.4, 0.0, 1.0}); });
    refused([](room& r) { r.vent_inlets.push_back({1, 0, 0.05, -0.4, 0.1, -1.0}); });
    refused([](room& r) { r.exhausts.push_back({0, 0}); });
    refused([](room& r) { r.exhausts.push_back({0, 4}); });
    refused([](room& r) { r.exhausts.push_back({1, 16}); });
    refused([](room& r) {
        r.vents = true;
        r.vent_u = flags(3);
    });
    refused([](room& r) {
        r.vents = true;
        r.corners_v = flags(3);
    });
    const auto constant = [](auto&& spoil) {
        room r;
        auto input = r.input();
        spoil(input);
        EXPECT_THROW(boundary<2>(*r.g, input), std::invalid_argument);
    };
    constant([](auto& in) { in.c_mu = 0.0; });
    constant([](auto& in) { in.rho = nan_value; });
    constant([](auto& in) { in.nu = -1.0; });
    constant([](auto& in) { in.law.floor = 0.0; });
}

TEST(Walls, AnArrayOfAnotherLengthIsRefused) {
    room r;
    auto b = r.build();
    const std::vector<double> one(11), corners(20);
    std::vector<double> out(20), short_out(19);
    EXPECT_THROW(b.conditions({one, r.v}, r.k, r.eps), std::invalid_argument);
    EXPECT_THROW(b.conditions(r.velocity(), one, r.eps), std::invalid_argument);
    EXPECT_THROW(b.conditions(r.velocity(), r.k, one), std::invalid_argument);
    EXPECT_THROW(b.wall_viscosity(one, {{{corners}, {corners}}}, {{{out}, {out}}}),
                 std::invalid_argument);
    EXPECT_THROW(b.wall_viscosity(r.k, {{{one}, {corners}}}, {{{out}, {out}}}),
                 std::invalid_argument);
    EXPECT_THROW(b.wall_viscosity(r.k, {{{corners}, {corners}}}, {{{out}, {short_out}}}),
                 std::invalid_argument);
}

TEST(Walls, OneDirectionHasNoStencil) {
    const std::vector<double> faces{0.0, 1.0, 2.0};
    const std::vector<double> centres{0.5, 1.5};
    const std::vector<std::int32_t> types(2, 0);
    const grid::grid<1> g(grid::grid_input<1>{
        .faces = {faces}, .centres = {centres}, .cell_type = types, .solid = 1});
    const flags wall(1, true);
    boundary_input<1> input{};
    input.wall_cells_low = {wall.span()};
    input.wall_cells_high = {wall.span()};
    input.c_mu = 0.09;
    input.rho = 1.2;
    input.nu = 1.5e-5;
    input.law = law();
    boundary<1> b(g, input);
    const std::vector<double> u(3, 0.0), k(2, 0.04), eps(2, 0.5);
    const auto c = b.conditions({u}, k, eps);
    EXPECT_TRUE(c.eps_held[0] && c.eps_held[1]);
    const double u_k = u_k_of(0.04);
    EXPECT_NEAR(c.eps_wall[0], u_k * u_k * u_k / (kappa * 0.5), 1e-15);
    EXPECT_EQ(c.production[1], 0.0);
    EXPECT_EQ(c.inflow_k[0][0], 0.04);
}

}  // namespace
}  // namespace cfdcore::walls
