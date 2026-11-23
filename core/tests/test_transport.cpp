// The transport module's own tests (ADR-017 D): results known without the
// reference. The comparison with solver_transport.py is tests/test_core_parity.py's.

#include "cfdcore/grid.hpp"
#include "cfdcore/scalar.hpp"
#include "cfdcore/transport.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using cfdcore::transport::budget;
using cfdcore::transport::class_physics;
using cfdcore::transport::conditions;
using cfdcore::transport::geometry;
using cfdcore::transport::interval_input;
using cfdcore::transport::settings;
using cfdcore::transport::solver;
using cfdcore::transport::status;
using cfdcore::transport::step_record;

constexpr std::int32_t fluid_type = 0;
constexpr std::int32_t solid_type = 1;
constexpr std::int32_t floor_code = 1;
constexpr std::int32_t ceiling_code = 2;
constexpr std::int32_t wall_code = 3;

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

// A bool array that a span can view (std::vector<bool> holds bits).
struct flags {
    explicit flags(std::size_t n) : values(std::make_unique<bool[]>(n)), size(n) {}
    std::span<const bool> span() const { return {values.get(), size}; }
    bool& operator[](std::size_t i) { return values[i]; }
    std::unique_ptr<bool[]> values;
    std::size_t size;
};

// A box of nx by ny cells of width h, every array a solver and its interval
// read, held so the spans stay valid. Faces are the reference's u [ny, nx + 1]
// and v [ny + 1, nx]. The domain faces deposit at `wall` (floor at y = 0,
// ceiling at the top, walls at the sides), and so do the faces of a SOLID
// block when there is one.
struct box2 {
    box2(std::size_t nx_, std::size_t ny_, double h, bool block)
        : nx(nx_),
          ny(ny_),
          x(nx_ + 1),
          xc(nx_),
          y(ny_ + 1),
          yc(ny_),
          types(nx_ * ny_, fluid_type),
          dx(nx_, h),
          dy(ny_, h),
          dx_face(nx_ + 1, h),
          dy_face(ny_ + 1, h),
          settling(ny_ * nx_ + nx_),
          u((nx_ + 1) * ny_, 0.0),
          v(nx_ * (ny_ + 1), 0.0),
          inflow_u((nx_ + 1) * ny_, 0.0),
          inflow_v(nx_ * (ny_ + 1), 0.0),
          deposition_u((nx_ + 1) * ny_, 0.0),
          deposition_v(nx_ * (ny_ + 1), 0.0),
          surface_u((nx_ + 1) * ny_, 0),
          surface_v(nx_ * (ny_ + 1), 0) {
        for (std::size_t i = 0; i <= nx; ++i) {
            x[i] = h * static_cast<double>(i);
        }
        for (std::size_t j = 0; j <= ny; ++j) {
            y[j] = h * static_cast<double>(j);
        }
        for (std::size_t i = 0; i < nx; ++i) {
            xc[i] = 0.5 * (x[i] + x[i + 1]);
        }
        for (std::size_t j = 0; j < ny; ++j) {
            yc[j] = 0.5 * (y[j] + y[j + 1]);
        }
        dx_face.front() = dx_face.back() = 0.5 * h;
        dy_face.front() = dy_face.back() = 0.5 * h;
        if (block) {
            for (std::size_t j = 0; j < ny / 3; ++j) {
                for (std::size_t i = nx / 3; i < nx / 2; ++i) {
                    types[j * nx + i] = solid_type;
                }
            }
        }
    }

    bool is_solid(std::size_t i, std::size_t j) const {
        return types[j * nx + i] == solid_type;
    }

    // Deposition velocity `wall` on every domain face and every face between a
    // non-SOLID and a SOLID cell, booked to the floor, ceiling and walls.
    void deposit(double wall) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i <= nx; ++i) {
                const bool edge = i == 0 || i == nx;
                const bool beside =
                    !edge && is_solid(i - 1, j) != is_solid(i, j);
                if (edge || beside) {
                    deposition_u[j * (nx + 1) + i] = wall;
                    surface_u[j * (nx + 1) + i] = edge ? wall_code : 0;
                }
            }
        }
        for (std::size_t j = 0; j <= ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                const bool edge = j == 0 || j == ny;
                const bool beside = !edge && is_solid(i, j - 1) != is_solid(i, j);
                if (edge || beside) {
                    deposition_v[j * nx + i] = wall;
                    surface_v[j * nx + i] = j == 0 ? floor_code : (j == ny ? ceiling_code : 0);
                }
            }
        }
    }

    // A circulation from a stream function psi on the nodes, zero on the
    // domain's edge and in the block: u = (psi above - psi below) / h on an x
    // face, v = -(psi right - psi left) / h on a y face. Its discrete divergence
    // is zero, exactly when psi's values and h are powers of two apart.
    void circulate(double scale) {
        std::vector<double> psi((nx + 1) * (ny + 1), 0.0);
        for (std::size_t j = 1; j < ny; ++j) {
            for (std::size_t i = 1; i < nx; ++i) {
                bool by_block = false;
                for (std::size_t a = i - 1; a <= i; ++a) {
                    for (std::size_t b = j - 1; b <= j; ++b) {
                        by_block = by_block || is_solid(a, b);
                    }
                }
                psi[j * (nx + 1) + i] =
                    by_block ? 0.0 : scale * static_cast<double>((i * 7 + j * 3) % 5);
            }
        }
        const double h = dx[0];
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i <= nx; ++i) {
                u[j * (nx + 1) + i] = (psi[(j + 1) * (nx + 1) + i] - psi[j * (nx + 1) + i]) / h;
            }
        }
        for (std::size_t j = 0; j <= ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                v[j * nx + i] = -(psi[j * (nx + 1) + i + 1] - psi[j * (nx + 1) + i]) / h;
            }
        }
    }

    // Settling on every interior y face between two non-SOLID cells.
    void settle() {
        for (std::size_t j = 1; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                settling[j * nx + i] = !is_solid(i, j - 1) && !is_solid(i, j);
            }
        }
    }

    cfdcore::grid::grid_input<2> grid_input() const {
        return {.faces = {x, y}, .centres = {xc, yc}, .cell_type = types, .solid = solid_type};
    }

    geometry<2> spacing() const {
        return {.widths = {std::span<const double>(dx), std::span<const double>(dy)},
                .face_distances = {std::span<const double>(dx_face),
                                   std::span<const double>(dy_face)}};
    }

    conditions<2> faces() const {
        conditions<2> c;
        c.inflow = {std::span<const double>(inflow_u), std::span<const double>(inflow_v)};
        c.deposition = {std::span<const double>(deposition_u),
                        std::span<const double>(deposition_v)};
        c.surface = {std::span<const std::int32_t>(surface_u),
                     std::span<const std::int32_t>(surface_v)};
        c.settling = settling.span();
        return c;
    }

    interval_input<2> air() const {
        interval_input<2> in;
        in.faces = {std::span<const double>(u), std::span<const double>(v)};
        return in;
    }

    std::size_t nx;
    std::size_t ny;
    std::vector<double> x;
    std::vector<double> xc;
    std::vector<double> y;
    std::vector<double> yc;
    std::vector<std::int32_t> types;
    std::vector<double> dx;
    std::vector<double> dy;
    std::vector<double> dx_face;
    std::vector<double> dy_face;
    flags settling;
    std::vector<double> u;
    std::vector<double> v;
    std::vector<double> inflow_u;
    std::vector<double> inflow_v;
    std::vector<double> deposition_u;
    std::vector<double> deposition_v;
    std::vector<std::int32_t> surface_u;
    std::vector<std::int32_t> surface_v;
};

settings keys(double limit = 1e-4, double tol = 1e-13, std::int64_t sweeps = 500) {
    settings s;
    s.cfl = 0.5;
    s.upwind = false;
    s.max_sweeps = sweeps;
    s.tol = tol;
    s.limit = limit;
    s.share = 0.1;
    s.horizon = 120.0;
    s.surface_codes = {floor_code, ceiling_code, wall_code};
    return s;
}

// The largest power of two at or below x: a step whose V / dt is exact.
double power_of_two_below(double x) { return std::exp2(std::floor(std::log2(x))); }

// initial + inflow + source - outflow - deposited - leftover - current, as
// MassBudget.residual forms it.
double residual(const budget& b) {
    double r = *b.initial + b.inflow + b.source - b.outflow;
    r = r - (b.deposited[0] + b.deposited[1] + b.deposited[2] + b.deposited[3]);
    return r - b.leftover - *b.current;
}

TEST(TransportStep, AUniformFieldInAClosedBoxWithNoSettlingChangesByNothing) {
    for (const bool upwind : {false, true}) {
        box2 box(12, 9, 0.5, true);
        box.circulate(0.125);
        const cfdcore::grid::grid<2> g(box.grid_input());
        const cfdcore::scalar::scheme<2> s(g);
        settings k = keys();
        k.upwind = upwind;
        solver<2> transport(s, box.spacing(), k, {{0.0, 0.25}});
        ASSERT_TRUE(transport.set_interval({box.faces()}, box.air()).ok());
        const double dt = power_of_two_below(transport.stable_dt(0));
        ASSERT_TRUE(transport.set_step(dt).ok());
        std::vector<double> c(box.types.size(), 3.0);
        for (std::size_t i = 0; i < c.size(); ++i) {
            c[i] = box.types[i] == solid_type ? 0.0 : 3.0;
        }
        std::vector<double> out(c.size(), -1.0);
        for (int n = 0; n < 5; ++n) {
            step_record record;
            ASSERT_TRUE(transport.step(0, c, out, record).ok());
            EXPECT_EQ(record.sweeps, 0);
            EXPECT_EQ(record.leftover, 0.0);
            for (std::size_t i = 0; i < c.size(); ++i) {
                ASSERT_EQ(bits(out[i]), bits(c[i])) << "cell " << i << " step " << n;
            }
        }
        const budget& b = transport.budget_of(0);
        EXPECT_EQ(bits(*b.current), bits(*b.initial));
        EXPECT_EQ(b.inflow, 0.0);
        EXPECT_EQ(b.outflow, 0.0);
        EXPECT_EQ(residual(b), 0.0);
    }
}

TEST(TransportStep, SettlingInTheSameBoxMovesTheUniformField) {
    // The control of the test above: the box and the field are the same, and
    // settling alone is enough to change them, so "changes by nothing" is not
    // a step that cannot change anything.
    box2 box(12, 9, 0.5, true);
    box.circulate(0.125);
    box.settle();
    const cfdcore::grid::grid<2> g(box.grid_input());
    const cfdcore::scalar::scheme<2> s(g);
    solver<2> transport(s, box.spacing(), keys(), {{0.01, 0.25}});
    ASSERT_TRUE(transport.set_interval({box.faces()}, box.air()).ok());
    ASSERT_TRUE(transport.set_step(power_of_two_below(transport.stable_dt(0))).ok());
    std::vector<double> c(box.types.size());
    for (std::size_t i = 0; i < c.size(); ++i) {
        c[i] = box.types[i] == solid_type ? 0.0 : 3.0;
    }
    std::vector<double> out(c.size(), -1.0);
    step_record record;
    ASSERT_TRUE(transport.step(0, c, out, record).ok());
    std::size_t changed = 0;
    for (std::size_t i = 0; i < c.size(); ++i) {
        changed += bits(out[i]) != bits(c[i]);
    }
    EXPECT_GT(changed, 0U);
}

TEST(TransportStep, AUniformLineWithDiffusionChangesByNothing) {
    // The direction template at D = 1: a column of cells, nothing moves, the
    // diffusion's residual is exactly zero.
    const std::size_t n = 17;
    std::vector<double> faces(n + 1);
    std::vector<double> centres(n);
    for (std::size_t i = 0; i <= n; ++i) {
        faces[i] = 0.25 * static_cast<double>(i);
    }
    for (std::size_t i = 0; i < n; ++i) {
        centres[i] = 0.5 * (faces[i] + faces[i + 1]);
    }
    std::vector<std::int32_t> types(n, fluid_type);
    const cfdcore::grid::grid<1> g({.faces = {faces}, .centres = {centres}, .cell_type = types,
                                    .solid = solid_type});
    const cfdcore::scalar::scheme<1> s(g);
    std::vector<double> widths(n, 0.25);
    std::vector<double> distances(n + 1, 0.25);
    distances.front() = distances.back() = 0.125;
    solver<1> transport(s, {.widths = {widths}, .face_distances = {distances}}, keys(),
                        {{0.0, 0.5}});
    std::vector<double> zeros(n + 1, 0.0);
    std::vector<std::int32_t> codes(n + 1, 0);
    flags settling(n + 1);
    conditions<1> c;
    c.inflow = {zeros};
    c.deposition = {zeros};
    c.surface = {codes};
    c.settling = settling.span();
    interval_input<1> air;
    air.faces = {zeros};
    ASSERT_TRUE(transport.set_interval({c}, air).ok());
    EXPECT_TRUE(std::isinf(transport.stable_dt(0)));
    ASSERT_TRUE(transport.set_step(0.5).ok());
    std::vector<double> field(n, 1.5);
    std::vector<double> out(n, -1.0);
    step_record record;
    ASSERT_TRUE(transport.step(0, field, out, record).ok());
    EXPECT_EQ(record.sweeps, 0);
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(bits(out[i]), bits(1.5));
    }
    EXPECT_EQ(residual(transport.budget_of(0)), 0.0);
}

// A closed box stepped many times with everything the budget books but an
// inlet: a circulation, settling, diffusion, deposition on every surface kind
// and a source. Returns the solver after the steps.
struct stepped_box {
    explicit stepped_box(settings k = keys())
        : box(20, 13, 0.3, true),
          g(box.grid_input()),
          s(g),
          transport(s, box.spacing(), k, {{0.004, 1e-3}, {0.0, 2e-3}}) {
        box.circulate(0.07);
        box.settle();
        box.deposit(0.003);
        rate.assign(box.types.size(), 0.0);
        for (std::size_t i = 0; i < rate.size(); ++i) {
            rate[i] = (box.types[i] != solid_type && i % 11 == 0) ? 50.0 * hashed(i) : 0.0;
        }
        field.assign(box.types.size(), 0.0);
        for (std::size_t i = 0; i < field.size(); ++i) {
            field[i] = box.types[i] == solid_type ? 0.0 : 1e3 * hashed(i + 7);
        }
    }

    interval_input<2> air() const {
        interval_input<2> in = box.air();
        in.sources = {std::span<const double>(rate), std::span<const double>()};
        return in;
    }

    box2 box;
    cfdcore::grid::grid<2> g;
    cfdcore::scalar::scheme<2> s;
    solver<2> transport;
    std::vector<double> rate;
    std::vector<double> field;
};

TEST(TransportStep, AClosedBoxsBudgetClosesToRounding) {
    // A loose max-norm stop, so that ADR-016's bound decides each solve and the
    // leftover stands well above the sums' rounding.
    stepped_box run(keys(1e-4, 1e-6));
    auto& transport = run.transport;
    ASSERT_TRUE(transport.set_interval({run.box.faces(), run.box.faces()}, run.air()).ok());
    const double stable = std::min(transport.stable_dt(0), transport.stable_dt(1));
    ASSERT_TRUE(transport.set_step(0.9 * stable).ok());
    std::array<std::vector<double>, 2> c = {run.field, run.field};
    std::vector<double> out(run.field.size());
    for (int n = 0; n < 40; ++n) {
        for (std::size_t k = 0; k < 2; ++k) {
            step_record record;
            ASSERT_TRUE(transport.step(k, c[k], out, record).ok());
            ASSERT_TRUE(record.converged);
            c[k] = out;
        }
    }
    for (std::size_t k = 0; k < 2; ++k) {
        const budget& b = transport.budget_of(k);
        const double supplied = *b.initial + b.inflow + b.source;
        // Closed: nothing crosses a domain face.
        EXPECT_EQ(b.inflow, 0.0);
        EXPECT_EQ(b.outflow, 0.0);
        // Every slot books, and the class with the source books it.
        for (std::size_t s = 0; s < 4; ++s) {
            EXPECT_GT(b.deposited[s], 0.0) << "class " << k << " slot " << s;
        }
        EXPECT_EQ(b.source > 0.0, k == 0);
        EXPECT_LT(std::fabs(residual(b)), 1e-12 * supplied) << "class " << k;
        // The leftover is booked and is not what closes the budget: without
        // it the residual is the leftover, many decades above rounding.
        EXPECT_NE(b.leftover, 0.0);
        EXPECT_GT(std::fabs(residual(b) + b.leftover), 1e3 * std::fabs(residual(b)));
    }
}

TEST(TransportStep, OutMayBeTheFieldItself) {
    stepped_box a;
    stepped_box b;
    for (auto* run : {&a, &b}) {
        ASSERT_TRUE(
            run->transport.set_interval({run->box.faces(), run->box.faces()}, run->air()).ok());
        ASSERT_TRUE(run->transport.set_step(0.5 * run->transport.stable_dt(0)).ok());
    }
    std::vector<double> in_place = a.field;
    std::vector<double> out(a.field.size());
    step_record record;
    ASSERT_TRUE(a.transport.step(0, in_place, in_place, record).ok());
    ASSERT_TRUE(b.transport.step(0, b.field, out, record).ok());
    for (std::size_t i = 0; i < out.size(); ++i) {
        ASSERT_EQ(bits(in_place[i]), bits(out[i]));
    }
}

TEST(TransportStep, AFieldThatIsNotFiniteIsRefusedBeforeAnythingIsBooked) {
    stepped_box run;
    ASSERT_TRUE(run.transport.set_interval({run.box.faces(), run.box.faces()}, run.air()).ok());
    ASSERT_TRUE(run.transport.set_step(0.5 * run.transport.stable_dt(0)).ok());
    std::vector<double> out(run.field.size(), -1.0);
    // A NaN in a SOLID cell is refused too: the reference checks the field
    // before it zeroes the SOLID cells.
    constexpr double inf = std::numeric_limits<double>::infinity();
    for (const double bad : {std::nan(""), inf, -inf}) {
        for (const bool in_solid : {false, true}) {
            std::vector<double> c = run.field;
            std::size_t at = 0;
            while ((run.box.types[at] == solid_type) != in_solid) {
                ++at;
            }
            c[at] = bad;
            step_record record;
            const auto refused = run.transport.step(0, c, out, record);
            EXPECT_EQ(refused.code, status::field_not_finite);
            EXPECT_FALSE(run.transport.budget_of(0).initial.has_value());
        }
    }
    EXPECT_EQ(out, std::vector<double>(run.field.size(), -1.0));
}

TEST(TransportStep, AStepAboveTheStableStepIsRefusedBeforeAnythingIsBooked) {
    stepped_box run;
    ASSERT_TRUE(run.transport.set_interval({run.box.faces(), run.box.faces()}, run.air()).ok());
    const double stable = run.transport.stable_dt(0);
    const double above = std::nextafter(stable, std::numeric_limits<double>::infinity());
    ASSERT_TRUE(run.transport.set_step(above).ok());
    std::vector<double> out(run.field.size());
    step_record record;
    const auto refused = run.transport.step(0, run.field, out, record);
    EXPECT_EQ(refused.code, status::dt_above_stable);
    EXPECT_FALSE(run.transport.budget_of(0).initial.has_value());
    // At the stable step itself the step is taken.
    ASSERT_TRUE(run.transport.set_step(stable).ok());
    EXPECT_TRUE(run.transport.step(0, run.field, out, record).ok());
}

TEST(TransportStep, ABreachIsRefusedAfterTheStepWithTheBudgetHoldingIt) {
    // A limit so small that the first step's leftover reaches it, the solve
    // stopped by its cap long before its share could hold.
    stepped_box run(keys(1e-300, 1e-13, 2));
    ASSERT_TRUE(run.transport.set_interval({run.box.faces(), run.box.faces()}, run.air()).ok());
    ASSERT_TRUE(run.transport.set_step(0.5 * run.transport.stable_dt(0)).ok());
    std::vector<double> out(run.field.size(), -1.0);
    step_record record;
    const auto refused = run.transport.step(0, run.field, out, record);
    ASSERT_EQ(refused.code, status::breach);
    const budget& b = run.transport.budget_of(0);
    EXPECT_TRUE(b.current.has_value());
    EXPECT_EQ(bits(record.accumulated), bits(b.leftover));
    EXPECT_EQ(bits(record.supplied), bits(*b.initial + b.inflow + b.source));
    EXPECT_GE(std::fabs(b.leftover), 1e-300 * record.supplied);
}

// Whether two budgets hold the same bits in every term.
bool same_bits(const budget& a, const budget& b) {
    const auto same = [](const std::optional<double>& x, const std::optional<double>& y) {
        return x.has_value() == y.has_value() && (!x.has_value() || bits(*x) == bits(*y));
    };
    bool equal = same(a.initial, b.initial) && same(a.current, b.current) &&
                 bits(a.inflow) == bits(b.inflow) && bits(a.outflow) == bits(b.outflow) &&
                 bits(a.source) == bits(b.source) && bits(a.leftover) == bits(b.leftover);
    for (std::size_t s = 0; s < a.deposited.size(); ++s) {
        equal = equal && bits(a.deposited[s]) == bits(b.deposited[s]);
    }
    return equal;
}

TEST(TransportStep, ASecondIntervalKeepsTheBudgetsAndEqualsAFreshSolverGivenThem) {
    // The march sets up interval after interval on one solver. A second interval
    // keeps what the first booked, needs a step of its own, and is then the
    // solver a fresh one is on the second interval's inputs once handed those
    // budgets: nothing of the first interval (its source, its circulation) stays.
    stepped_box run(keys(1e-4, 1e-6));
    stepped_box fresh(keys(1e-4, 1e-6));
    auto& t = run.transport;
    ASSERT_TRUE(t.set_interval({run.box.faces(), run.box.faces()}, run.air()).ok());
    ASSERT_TRUE(t.set_step(0.9 * std::min(t.stable_dt(0), t.stable_dt(1))).ok());
    std::array<std::vector<double>, 2> c = {run.field, run.field};
    std::vector<double> out(run.field.size());
    step_record record;
    for (int n = 0; n < 4; ++n) {
        for (std::size_t k = 0; k < 2; ++k) {
            ASSERT_TRUE(t.step(k, c[k], out, record).ok());
            c[k] = out;
        }
    }
    const std::vector<budget> carried = {t.budget_of(0), t.budget_of(1)};
    ASSERT_GT(carried[0].source, 0.0);
    // The second interval: another circulation, and no source.
    run.box.circulate(0.04);
    fresh.box.circulate(0.04);
    ASSERT_TRUE(t.set_interval({run.box.faces(), run.box.faces()}, run.box.air()).ok());
    for (std::size_t k = 0; k < 2; ++k) {
        EXPECT_TRUE(same_bits(t.budget_of(k), carried[k])) << "class " << k;
    }
    EXPECT_THROW(t.step(0, c[0], out, record), std::invalid_argument);
    fresh.transport.set_budgets(carried);
    ASSERT_TRUE(
        fresh.transport.set_interval({fresh.box.faces(), fresh.box.faces()}, fresh.box.air())
            .ok());
    const double dt = 0.6 * std::min(t.stable_dt(0), t.stable_dt(1));
    ASSERT_TRUE(t.set_step(dt).ok());
    ASSERT_TRUE(fresh.transport.set_step(dt).ok());
    std::array<std::vector<double>, 2> other = c;
    std::vector<double> other_out(out.size());
    for (int n = 0; n < 3; ++n) {
        for (std::size_t k = 0; k < 2; ++k) {
            ASSERT_TRUE(t.step(k, c[k], out, record).ok());
            ASSERT_TRUE(fresh.transport.step(k, other[k], other_out, record).ok());
            for (std::size_t i = 0; i < out.size(); ++i) {
                ASSERT_EQ(bits(out[i]), bits(other_out[i])) << "cell " << i << " step " << n;
            }
            EXPECT_TRUE(same_bits(t.budget_of(k), fresh.transport.budget_of(k)))
                << "class " << k << " step " << n;
            c[k] = out;
            other[k] = other_out;
        }
    }
}

TEST(TransportStep, ASecondStepSizeInOneIntervalEqualsAFreshSolverSetToIt) {
    // set_step again replaces what the first fixed, the deposition per face and
    // the source rate, each times dt.
    stepped_box run(keys(1e-4, 1e-6));
    stepped_box fresh(keys(1e-4, 1e-6));
    for (auto* box : {&run, &fresh}) {
        ASSERT_TRUE(
            box->transport.set_interval({box->box.faces(), box->box.faces()}, box->air()).ok());
    }
    const double stable = std::min(run.transport.stable_dt(0), run.transport.stable_dt(1));
    ASSERT_TRUE(run.transport.set_step(0.4 * stable).ok());
    std::array<std::vector<double>, 2> c = {run.field, run.field};
    std::vector<double> out(run.field.size());
    step_record record;
    for (std::size_t k = 0; k < 2; ++k) {
        ASSERT_TRUE(run.transport.step(k, c[k], out, record).ok());
        c[k] = out;
    }
    fresh.transport.set_budgets({run.transport.budget_of(0), run.transport.budget_of(1)});
    ASSERT_TRUE(run.transport.set_step(0.8 * stable).ok());
    ASSERT_TRUE(fresh.transport.set_step(0.8 * stable).ok());
    std::vector<double> other_out(out.size());
    for (std::size_t k = 0; k < 2; ++k) {
        ASSERT_TRUE(run.transport.step(k, c[k], out, record).ok());
        ASSERT_TRUE(fresh.transport.step(k, c[k], other_out, record).ok());
        for (std::size_t i = 0; i < out.size(); ++i) {
            ASSERT_EQ(bits(out[i]), bits(other_out[i])) << "cell " << i << " class " << k;
        }
        EXPECT_TRUE(same_bits(run.transport.budget_of(k), fresh.transport.budget_of(k)))
            << "class " << k;
    }
}

TEST(TransportStep, ProgrammingErrorsThrow) {
    stepped_box run;
    std::vector<double> out(run.field.size());
    step_record record;
    EXPECT_THROW(run.transport.set_step(0.1), std::invalid_argument);
    EXPECT_THROW(run.transport.set_interval({run.box.faces()}, run.air()), std::invalid_argument);
    auto short_faces = run.box.faces();
    short_faces.inflow[0] = short_faces.inflow[0].first(3);
    EXPECT_THROW(run.transport.set_interval({short_faces, run.box.faces()}, run.air()),
                 std::invalid_argument);
    ASSERT_TRUE(run.transport.set_interval({run.box.faces(), run.box.faces()}, run.air()).ok());
    EXPECT_THROW(run.transport.step(0, run.field, out, record), std::invalid_argument);
    ASSERT_TRUE(run.transport.set_step(0.5 * run.transport.stable_dt(0)).ok());
    EXPECT_THROW(run.transport.step(2, run.field, out, record), std::invalid_argument);
    EXPECT_THROW(run.transport.step(0, std::span<const double>(run.field).first(5), out, record),
                 std::invalid_argument);
    EXPECT_THROW(run.transport.set_budgets({budget{}}), std::invalid_argument);
}

}  // namespace
