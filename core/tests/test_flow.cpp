// The outer iteration on a lid-driven box against its hand results and against
// itself: what needs no reference (ADR-017 D). Its parity is
// tests/test_core_flow.py's.

#include "cfdcore/flow.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace cfdcore::flow {
namespace {

constexpr double nan_value = std::numeric_limits<double>::quiet_NaN();

std::uint64_t bits(double x) { return std::bit_cast<std::uint64_t>(x); }

// A bool array a span can view.
struct flags {
    explicit flags(std::size_t n, bool value = false) : size(n), data(new bool[n]) {
        std::fill(data.get(), data.get() + n, value);
    }
    std::span<const bool> span() const { return {data.get(), size}; }
    std::size_t size;
    std::unique_ptr<bool[]> data;
};

// An edge's tangential condition with its arrays.
struct edge {
    edge(std::size_t n, double speed, double distance)
        : known(n, true), value(n, speed), wall_distance(distance) {}
    momentum::edge_condition condition() const {
        return {known.span(), value, wall_distance};
    }
    flags known;
    std::vector<double> value;
    double wall_distance;
};

// A box of nx by ny unit cells, closed, its top wall moving at one metre a second.
struct box {
    static constexpr std::size_t nx = 6;
    static constexpr std::size_t ny = 5;

    box()
        : types(nx * ny, 0),
          g(grid::grid_input<2>{.faces = {faces_x, faces_y},
                                .centres = {centres_x, centres_y},
                                .cell_type = types,
                                .solid = 1}),
          bottom(nx + 1, 0.0, 0.5),
          top(nx + 1, 1.0, 0.5),
          left(ny + 1, 0.0, 0.5),
          right(ny + 1, 0.0, 0.5),
          closed_x(ny),
          closed_y(nx),
          predictor(g, predictor_input()),
          corrector(g, corrector_input()),
          outlets(g, {closed_x.span(), closed_y.span()}, {closed_x.span(), closed_y.span()}) {}

    momentum::predictor_input<2> predictor_input() const {
        momentum::predictor_input<2> in{};
        in.cell_widths = {widths_x, widths_y};
        in.face_widths = {between_x, between_y};
        in.low[0][0] = bottom.condition();
        in.high[0][0] = top.condition();
        in.low[1][0] = left.condition();
        in.high[1][0] = right.condition();
        in.rho = 1.0;
        in.mu = 0.05;
        in.alpha = 0.6;
        in.sweeps = 2;
        return in;
    }

    pressure::corrector_input<2> corrector_input() const {
        pressure::corrector_input<2> in{};
        in.cell_widths = {widths_x, widths_y};
        in.outlet_low = {closed_x.span(), closed_y.span()};
        in.outlet_high = {closed_x.span(), closed_y.span()};
        in.fluid = 0;
        in.rho = 1.0;
        in.alpha = 0.4;
        in.rtol = 1e-12;
        in.floor = 0.0;
        in.max_iter = 400;
        in.method = pressure::preconditioner::jacobi;
        in.levels = {.coarsest_cells = 8,
                     .sweeps = 1,
                     .weight = 2.0 / 3.0,
                     .dominance_tolerance = 1e-12,
                     .rcond = 1e-15};
        return in;
    }

    flow_input<2> laminar(stop_rule rule = stop_rule::velocity_step) const {
        flow_input<2> in{};
        in.fluid = 0;
        in.mu = 0.05;
        in.rho = 1.0;
        in.nu = 0.05;
        in.alpha_turbulence = 1.0;
        in.residual_scale = 1.0;
        in.rule = rule;
        in.convergence_tol = 1e-30;
        in.keys = {.velocity_scale = 1.0,
                   .flux_scale = 6.0,
                   .iteration_error_tol = 1e-6,
                   .mass_imbalance_tol = 1e-9,
                   .viscosity_condition = false,
                   .window = 20,
                   .version_without_e = 3,
                   .version_with_e = 5};
        return in;
    }

    modules<2> parts() {
        return {.predictor = &predictor, .corrector = &corrector, .outlets = &outlets};
    }

    std::vector<double> faces_x{0, 1, 2, 3, 4, 5, 6};
    std::vector<double> centres_x{0.5, 1.5, 2.5, 3.5, 4.5, 5.5};
    std::vector<double> widths_x = std::vector<double>(nx, 1.0);
    std::vector<double> between_x{0.5, 1, 1, 1, 1, 1, 0.5};
    std::vector<double> faces_y{0, 1, 2, 3, 4, 5};
    std::vector<double> centres_y{0.5, 1.5, 2.5, 3.5, 4.5};
    std::vector<double> widths_y = std::vector<double>(ny, 1.0);
    std::vector<double> between_y{0.5, 1, 1, 1, 1, 0.5};
    std::vector<std::int32_t> types;
    grid::grid<2> g;
    edge bottom, top, left, right;
    flags closed_x, closed_y;
    momentum::predictor<2> predictor;
    pressure::corrector<2> corrector;
    walls::outlets<2> outlets;
};

TEST(CellCentres, AreTheMeanOfEachCellsTwoFaces) {
    const layout::layout<2> cells({3, 2});
    const std::vector<double> u{0, 1, 2, 3, 10, 20, 30, 40};
    const std::vector<double> v{1, 2, 3, 5, 6, 7, 9, 10, 11};
    std::vector<double> out(6);
    cell_centres<2>(cells, 0, u, out);
    EXPECT_EQ(out, (std::vector<double>{0.5, 1.5, 2.5, 15, 25, 35}));
    cell_centres<2>(cells, 1, v, out);
    EXPECT_EQ(out, (std::vector<double>{3, 4, 5, 7, 8, 9}));
    std::vector<double> short_out(5);
    EXPECT_THROW(cell_centres<2>(cells, 0, u, short_out), std::invalid_argument);
}

TEST(LargestChange, ReadsTheMaskedCellsAndKeepsANaN) {
    std::vector<double> now{1.0, -4.0, 2.0, 7.0, 0.0, 3.0, 1.0, 1.0, 1.0, 1.0};
    const std::vector<double> before(now.size(), 1.0);
    std::vector<lanes::mask> cells(now.size(), lanes::all);
    cells[3] = lanes::none;
    EXPECT_EQ(largest_change(now, before, cells), 5.0);
    now[3] = nan_value;
    EXPECT_EQ(largest_change(now, before, cells), 5.0);
    now[9] = nan_value;
    EXPECT_TRUE(std::isnan(largest_change(now, before, cells)));
}

TEST(Solver, RefusesModulesAndNumbersItCannotUse) {
    box b;
    EXPECT_TRUE(solver<2>::check(b.g, b.parts(), b.laminar()).empty());
    modules<2> missing = b.parts();
    missing.corrector = nullptr;
    EXPECT_FALSE(solver<2>::check(b.g, missing, b.laminar()).empty());
    flow_input<2> in = b.laminar();
    in.mode = viscosity_mode::coupled;
    EXPECT_FALSE(solver<2>::check(b.g, b.parts(), in).empty());
    in = b.laminar();
    in.mode = viscosity_mode::fixed;
    EXPECT_FALSE(solver<2>::check(b.g, b.parts(), in).empty());
    for (const double bad : {0.0, -1.0, nan_value}) {
        in = b.laminar();
        in.residual_scale = bad;
        EXPECT_FALSE(solver<2>::check(b.g, b.parts(), in).empty());
        in = b.laminar();
        in.convergence_tol = bad;
        EXPECT_FALSE(solver<2>::check(b.g, b.parts(), in).empty());
    }
    in = b.laminar(stop_rule::error_estimate);
    in.keys.window = 1;
    EXPECT_FALSE(solver<2>::check(b.g, b.parts(), in).empty());
    in = b.laminar(stop_rule::error_estimate);
    in.keys.version_with_e = 0;
    EXPECT_FALSE(solver<2>::check(b.g, b.parts(), in).empty());
    in = b.laminar();
    in.fluid = 7;
    EXPECT_FALSE(solver<2>::check(b.g, b.parts(), in).empty());
    EXPECT_THROW(solver<2>(b.g, missing, b.laminar()), std::invalid_argument);
}

// A fixed viscosity is read once, as the predictor reads it: the reason names
// which of the two refusals it is, a value that is not finite before one that is
// not positive.
TEST(Solver, ReadsAFixedViscosityOnceAndRefusesWhatThePredictorRefuses) {
    box b;
    std::vector<double> mu_eff(box::nx * box::ny, 0.05);
    flow_input<2> in = b.laminar();
    in.mode = viscosity_mode::fixed;
    in.mu_eff = mu_eff;
    EXPECT_TRUE(solver<2>::check(b.g, b.parts(), in).empty());
    EXPECT_NO_THROW(solver<2>(b.g, b.parts(), in));
    for (const double bad : {nan_value, std::numeric_limits<double>::infinity(),
                             -std::numeric_limits<double>::infinity()}) {
        mu_eff[7] = bad;
        EXPECT_NE(solver<2>::check(b.g, b.parts(), in).find("finite"), std::string_view::npos);
        EXPECT_THROW(solver<2>(b.g, b.parts(), in), std::invalid_argument);
    }
    for (const double bad : {0.0, -0.05}) {
        mu_eff[7] = bad;
        EXPECT_NE(solver<2>::check(b.g, b.parts(), in).find("positive"),
                  std::string_view::npos);
        EXPECT_THROW(solver<2>(b.g, b.parts(), in), std::invalid_argument);
    }
    // Both at once: the value that is not finite is named first, as the reference does.
    mu_eff[7] = nan_value;
    mu_eff[8] = -1.0;
    EXPECT_NE(solver<2>::check(b.g, b.parts(), in).find("finite"), std::string_view::npos);
}

// The box's faces at rest; the lid's velocity is its edge's tangential condition.
std::array<std::vector<double>, 2> rest(const box&) {
    return {std::vector<double>((box::nx + 1) * box::ny, 0.0),
            std::vector<double>(box::nx * (box::ny + 1), 0.0)};
}

TEST(Solver, AnOuterIterationIsThePredictionAndTheCorrectionInTurn) {
    box b;
    solver<2> s(b.g, b.parts(), b.laminar());
    const auto faces = rest(b);
    const std::vector<double> p(box::nx * box::ny, 0.0);
    s.start({faces[0], faces[1]}, p, {}, {}, {});
    std::vector<iteration_record> records(1);
    const chunk_result result = s.run(0, records);
    ASSERT_EQ(result.failure.kind, failure::none);
    ASSERT_EQ(result.outer, 1);
    // The same two calls made by hand.
    std::array<std::vector<double>, 2> star = faces, diagonal = faces, corrected = faces;
    b.predictor.predict({faces[0], faces[1]}, p, {}, {}, {star[0], star[1]},
                        {diagonal[0], diagonal[1]});
    std::vector<double> next(p.size()), p_prime(p.size());
    const auto solved = b.corrector.correct({star[0], star[1]}, {diagonal[0], diagonal[1]}, p,
                                            {corrected[0], corrected[1]}, next, p_prime);
    EXPECT_EQ(records[0].pressure_iterations, solved.iterations);
    for (int c = 0; c < 2; ++c) {
        for (std::size_t i = 0; i < corrected[c].size(); ++i) {
            EXPECT_EQ(bits(s.velocity(c)[i]), bits(corrected[c][i]));
        }
    }
    for (std::size_t i = 0; i < next.size(); ++i) {
        EXPECT_EQ(bits(s.pressure()[i]), bits(next[i]));
    }
    // The residual: the largest change of the cell centres, which start at zero.
    std::array<std::vector<double>, 2> centres{std::vector<double>(p.size()),
                                               std::vector<double>(p.size())};
    double step = 0.0;
    for (int c = 0; c < 2; ++c) {
        cell_centres<2>(b.g.fields(), c, corrected[c], centres[c]);
        for (const double x : centres[c]) {
            step = std::max(step, std::fabs(x));
        }
    }
    EXPECT_EQ(records[0].step, step);
    EXPECT_EQ(records[0].residual, step / 1.0);
    EXPECT_GT(step, 0.0);
    EXPECT_TRUE(std::isnan(records[0].viscosity_step));
}

TEST(Solver, ChunksOfAnySizeMakeTheSameSolve) {
    box b;
    const auto faces = rest(b);
    const std::vector<double> p(box::nx * box::ny, 0.0);
    std::vector<iteration_record> whole(60), first(25), second(35);
    solver<2> one(b.g, b.parts(), b.laminar(stop_rule::error_estimate));
    one.start({faces[0], faces[1]}, p, {}, {}, {});
    EXPECT_EQ(one.run(0, whole).outer, 60);
    solver<2> two(b.g, b.parts(), b.laminar(stop_rule::error_estimate));
    two.start({faces[0], faces[1]}, p, {}, {}, {});
    EXPECT_EQ(two.run(0, first).outer, 25);
    EXPECT_EQ(two.run(25, second).outer, 35);
    for (std::size_t n = 0; n < 60; ++n) {
        const iteration_record& r = n < 25 ? first[n] : second[n - 25];
        EXPECT_EQ(bits(r.residual), bits(whole[n].residual));
        EXPECT_EQ(bits(r.rule.estimate), bits(whole[n].rule.estimate));
    }
    for (int c = 0; c < 2; ++c) {
        for (std::size_t i = 0; i < one.velocity(c).size(); ++i) {
            EXPECT_EQ(bits(one.velocity(c)[i]), bits(two.velocity(c)[i]));
        }
    }
    // The window full, an estimate; a restart forgets it.
    EXPECT_TRUE(std::isfinite(whole[59].rule.estimate));
    ASSERT_NE(two.rule(), nullptr);
    EXPECT_EQ(two.rule()->steps().size(), 20u);
    two.start({faces[0], faces[1]}, p, {}, {}, {});
    EXPECT_TRUE(two.rule()->steps().empty());
    // A window seeded into a fresh rule.
    std::vector<double> kept(one.rule()->steps().begin(), one.rule()->steps().end());
    EXPECT_EQ(two.restore(kept, {}), stopping::restore_status::ok);
    EXPECT_EQ(two.restore(kept, {}), stopping::restore_status::already_stepped);
}

// A solve under the velocity step has no rule: an empty window restores nothing and
// a window with a step is refused as what it is, a window for no rule.
TEST(Solver, AWindowRestoredUnderTheVelocityStepIsRefusedAsHavingNoRule) {
    box b;
    solver<2> s(b.g, b.parts(), b.laminar());
    EXPECT_EQ(s.restore({}, {}), stopping::restore_status::ok);
    const std::vector<double> steps{1e-3};
    EXPECT_EQ(s.restore(steps, {}), stopping::restore_status::no_rule);
    EXPECT_EQ(s.restore({}, steps), stopping::restore_status::no_rule);
    EXPECT_EQ(s.rule(), nullptr);
}

TEST(Solver, StopsUnderTheVelocityStepOnlyBelowItsBound) {
    box b;
    flow_input<2> in = b.laminar();
    in.convergence_tol = 1e-3;
    solver<2> s(b.g, b.parts(), in);
    const auto faces = rest(b);
    const std::vector<double> p(box::nx * box::ny, 0.0);
    s.start({faces[0], faces[1]}, p, {}, {}, {});
    std::vector<iteration_record> records(2000);
    const chunk_result result = s.run(0, records);
    ASSERT_TRUE(result.stopped);
    const auto n = static_cast<std::size_t>(result.outer);
    EXPECT_LT(records[n - 1].residual, 1e-3);
    for (std::size_t k = 0; k + 1 < n; ++k) {
        EXPECT_TRUE(records[k].residual >= 1e-3 || records[k].reached_cap);
    }
}

}  // namespace
}  // namespace cfdcore::flow
