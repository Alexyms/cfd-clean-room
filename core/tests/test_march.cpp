// The step loop's own tests (ADR-017 D): channels, arrival, recovery and march,
// with results known without the reference. The comparisons with monitor.py and
// time_integration.py are tests/test_core_march.py's.

#include "cfdcore/arrival.hpp"
#include "cfdcore/channels.hpp"
#include "cfdcore/grid.hpp"
#include "cfdcore/march.hpp"
#include "cfdcore/recovery.hpp"
#include "cfdcore/scalar.hpp"
#include "cfdcore/transport.hpp"

#include <gtest/gtest.h>

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

using cfdcore::channels::set;
using cfdcore::march::plan;
using cfdcore::march::resident;
using cfdcore::march::rows;
using cfdcore::march::run;
using cfdcore::march::stop;

constexpr std::int32_t fluid_type = 0;
constexpr std::int32_t solid_type = 1;
constexpr double nan = std::numeric_limits<double>::quiet_NaN();

std::uint64_t bits(double x) { return std::bit_cast<std::uint64_t>(x); }

bool same_bits(std::span<const double> a, std::span<const double> b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (bits(a[i]) != bits(b[i])) {
            return false;
        }
    }
    return true;
}

// A deterministic value in [0, 1) for index i.
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
    std::unique_ptr<bool[]> values;
    std::size_t size;
};

// A closed box of nx by ny cells of width h with a SOLID block, a circulation
// from a stream function, settling on the interior y faces, deposition on every
// domain face and no inflow: every array a run and its interval read.
struct box {
    box(std::size_t nx_, std::size_t ny_, double h)
        : nx(nx_), ny(ny_), x(nx_ + 1), xc(nx_), y(ny_ + 1), yc(ny_),
          types(nx_ * ny_, fluid_type), dx(nx_, h), dy(ny_, h), dx_face(nx_ + 1, h),
          dy_face(ny_ + 1, h), settling(nx_ * (ny_ + 1)), u((nx_ + 1) * ny_, 0.0),
          v(nx_ * (ny_ + 1), 0.0), inflow_u((nx_ + 1) * ny_, 0.0), inflow_v(nx_ * (ny_ + 1), 0.0),
          deposition_u((nx_ + 1) * ny_, 0.0), deposition_v(nx_ * (ny_ + 1), 0.0),
          surface_u((nx_ + 1) * ny_, 0), surface_v(nx_ * (ny_ + 1), 0) {
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
        for (std::size_t j = 0; j < ny / 3; ++j) {
            for (std::size_t i = nx / 3; i < nx / 2; ++i) {
                types[j * nx + i] = solid_type;
            }
        }
        std::vector<double> psi((nx + 1) * (ny + 1), 0.0);
        for (std::size_t j = 1; j < ny; ++j) {
            for (std::size_t i = 1; i < nx; ++i) {
                bool by_block = false;
                for (std::size_t a = i - 1; a <= i; ++a) {
                    for (std::size_t b = j - 1; b <= j; ++b) {
                        by_block = by_block || solid(a, b);
                    }
                }
                psi[j * (nx + 1) + i] =
                    by_block ? 0.0 : 0.125 * static_cast<double>((i * 7 + j * 3) % 5);
            }
        }
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i <= nx; ++i) {
                u[j * (nx + 1) + i] = (psi[(j + 1) * (nx + 1) + i] - psi[j * (nx + 1) + i]) / h;
            }
        }
        for (std::size_t j = 0; j <= ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                v[j * nx + i] = -(psi[j * (nx + 1) + i + 1] - psi[j * (nx + 1) + i]) / h;
                if (j > 0 && j < ny) {
                    settling.values[j * nx + i] = !solid(i, j - 1) && !solid(i, j);
                }
                if (j == 0 || j == ny) {
                    deposition_v[j * nx + i] = 1e-3;
                    surface_v[j * nx + i] = j == 0 ? 1 : 2;
                }
            }
        }
    }

    bool solid(std::size_t i, std::size_t j) const { return types[j * nx + i] == solid_type; }
    std::size_t cells() const { return nx * ny; }

    cfdcore::grid::grid_input<2> grid_input() const {
        return {.faces = {x, y}, .centres = {xc, yc}, .cell_type = types, .solid = solid_type};
    }

    cfdcore::transport::geometry<2> spacing() const {
        return {.widths = {std::span<const double>(dx), std::span<const double>(dy)},
                .face_distances = {std::span<const double>(dx_face),
                                   std::span<const double>(dy_face)}};
    }

    cfdcore::transport::conditions<2> faces() const {
        cfdcore::transport::conditions<2> c;
        c.inflow = {std::span<const double>(inflow_u), std::span<const double>(inflow_v)};
        c.deposition = {std::span<const double>(deposition_u),
                        std::span<const double>(deposition_v)};
        c.surface = {std::span<const std::int32_t>(surface_u),
                     std::span<const std::int32_t>(surface_v)};
        c.settling = settling.span();
        return c;
    }

    cfdcore::transport::interval_input<2> air() const {
        cfdcore::transport::interval_input<2> in;
        in.faces = {std::span<const double>(u), std::span<const double>(v)};
        return in;
    }

    std::size_t nx;
    std::size_t ny;
    std::vector<double> x, xc, y, yc;
    std::vector<std::int32_t> types;
    std::vector<double> dx, dy, dx_face, dy_face;
    flags settling;
    std::vector<double> u, v, inflow_u, inflow_v, deposition_u, deposition_v;
    std::vector<std::int32_t> surface_u, surface_v;
};

cfdcore::transport::settings keys() {
    cfdcore::transport::settings s;
    s.cfl = 0.5;
    s.max_sweeps = 500;
    s.tol = 1e-13;
    s.limit = 1e-4;
    s.share = 0.1;
    s.horizon = 120.0;
    s.surface_codes = {1, 2, 3};
    return s;
}

// Three classes, the cumulative channels (class k and every larger one).
std::vector<std::vector<std::size_t>> cumulative() { return {{0, 1, 2}, {1, 2}, {2}}; }
std::vector<cfdcore::transport::class_physics> physics() {
    return {{1e-5, 1e-5}, {1e-4, 1e-6}, {1e-3, 1e-7}};
}

// The rows of n steps of a run, held so their spans stay valid.
struct row_store {
    row_store(std::size_t n, std::size_t classes, std::size_t sensors)
        : step(n), t(n), budgets(n * classes * cfdcore::march::budget_terms),
          readings(n * classes * sensors), channels(n * classes * sensors), means(n * classes),
          sweeps(n * classes), converged(n * classes) {}
    rows view() {
        return {step, t, budgets, readings, channels, means, sweeps, converged};
    }
    std::vector<std::int64_t> step;
    std::vector<double> t, budgets, readings, channels, means;
    std::vector<std::int64_t> sweeps;
    std::vector<std::uint8_t> converged;
};

// A positive field per class, zero in the SOLID cells.
std::vector<double> planted(const box& b, std::size_t classes, double scale) {
    std::vector<double> out(classes * b.cells(), 0.0);
    for (std::size_t k = 0; k < classes; ++k) {
        for (std::size_t i = 0; i < b.cells(); ++i) {
            const bool solid = b.types[i] == solid_type;
            out[k * b.cells() + i] = solid ? 0.0 : scale * (1.0 + hashed(i + 977 * k));
        }
    }
    return out;
}

// A run's resident state read whole, to hand to another run.
struct snapshot {
    std::vector<double> fields, first_above, last_above, sensor_time, mean_time, class_since;
    std::vector<cfdcore::transport::budget> budgets;
    std::int64_t step = 0;
    double t = 0.0, arrival_t = 0.0, room_since = 0.0, recovery_t = 0.0;

    template <int D>
    explicit snapshot(const run<D>& r)
        : fields(r.fields().begin(), r.fields().end()),
          first_above(r.maps().first_above().begin(), r.maps().first_above().end()),
          step(r.step()), t(r.time()), arrival_t(r.maps().time()) {
        for (std::size_t k = 0; k < r.classes(); ++k) {
            budgets.push_back(r.budget_of(k));
        }
        if (const auto* tr = r.tracker()) {
            last_above.assign(tr->last_above().begin(), tr->last_above().end());
            sensor_time.assign(tr->sensor_time().begin(), tr->sensor_time().end());
            mean_time.assign(tr->mean_time().begin(), tr->mean_time().end());
            class_since.assign(tr->class_since().begin(), tr->class_since().end());
            room_since = tr->room_since();
            recovery_t = tr->time();
        }
    }

    resident view() const {
        resident s;
        s.fields = fields;
        s.budgets = budgets;
        s.step = step;
        s.t = t;
        s.first_above = first_above;
        s.arrival_t = arrival_t;
        s.last_above = last_above;
        s.sensor_time = sensor_time;
        s.mean_time = mean_time;
        s.class_since = class_since;
        s.room_since = room_since;
        s.recovery_t = recovery_t;
        return s;
    }
};

// A fresh state at t = 0: the fields, zero budgets, the maps and recovery NaN.
resident fresh(const std::vector<double>& fields, std::size_t classes,
               const std::vector<double>& maps, const std::vector<double>* recovery_cells,
               const std::vector<double>* sensor_nan, const std::vector<double>* class_nan) {
    resident s;
    s.fields = fields;
    s.budgets.assign(classes, cfdcore::transport::budget{});
    s.first_above = maps;
    s.arrival_t = nan;
    if (recovery_cells != nullptr) {
        s.last_above = *recovery_cells;
        s.sensor_time = *sensor_nan;
        s.mean_time = *class_nan;
        s.class_since = *class_nan;
        s.room_since = nan;
        s.recovery_t = nan;
    }
    return s;
}

cfdcore::march::monitor_setup setup(const box& b, std::optional<cfdcore::recovery::keys> rec) {
    cfdcore::march::monitor_setup m;
    m.channels = cumulative();
    m.thresholds = {3.0, 2.0, 1.0};
    m.sensors = {b.nx + 1, b.cells() / 2 + 3, b.cells() - 2};
    m.onsets = {0.0, 0.25};
    m.recovery = std::move(rec);
    return m;
}

// --- channels ---------------------------------------------------------------

TEST(Channels, AChannelIsZeroThenEachMemberAddedInTurn) {
    const cfdcore::layout::layout<2> fields({2, 1});
    // Class 0 is 1, class 1 is 1e16, class 2 is -1e16 in cell 0: the order of
    // the additions decides the sum. Cell 1 holds -0.0 in every class.
    const std::vector<double> classes = {1.0, -0.0, 1e16, -0.0, -1e16, -0.0};
    std::vector<double> out(2);
    const std::vector<std::size_t> forward = {0, 1, 2};
    cfdcore::channels::channel_field(fields, classes, forward, out);
    EXPECT_EQ(bits(out[0]), bits(0.0));  // (1 + 1e16) rounds to 1e16, then less 1e16
    EXPECT_EQ(bits(out[1]), bits(0.0));  // 0.0 + -0.0 is +0.0, as np.zeros + -0.0
    const std::vector<std::size_t> other = {1, 2, 0};
    cfdcore::channels::channel_field(fields, classes, other, out);
    EXPECT_EQ(out[0], 1.0);
}

TEST(Channels, ASensorsChannelIsTheChannelFieldReadAtItsCell) {
    const box b(13, 7, 0.25);
    const cfdcore::grid::grid<2> g(b.grid_input());
    const auto classes = planted(b, 3, 1e3);
    const set channels(cumulative());
    std::vector<double> fields(3 * b.cells());
    cfdcore::channels::channel_fields(g.fields(), classes, channels, fields);
    const std::vector<std::size_t> sensors = {0, 5, 44, b.cells() - 1};
    std::vector<double> readings(3 * sensors.size());
    std::vector<double> sums(3 * sensors.size());
    cfdcore::channels::sensor_readings(classes, b.cells(), sensors, readings);
    cfdcore::channels::sensor_channels(readings, sensors.size(), channels, sums);
    for (std::size_t k = 0; k < 3; ++k) {
        for (std::size_t s = 0; s < sensors.size(); ++s) {
            EXPECT_EQ(bits(readings[k * sensors.size() + s]),
                      bits(classes[k * b.cells() + sensors[s]]));
            EXPECT_EQ(bits(sums[k * sensors.size() + s]),
                      bits(fields[k * b.cells() + sensors[s]]));
        }
    }
}

TEST(Channels, ASetThatIsNotOneChannelPerClassThrows) {
    EXPECT_THROW(set(std::vector<std::vector<std::size_t>>{}), std::invalid_argument);
    EXPECT_THROW(set({{0}, {0}}), std::invalid_argument);        // channel 1 leaves out 1
    EXPECT_THROW(set({{0, 1, 1}, {1}}), std::invalid_argument);  // class 1 twice
    EXPECT_THROW(set({{0, 2}, {1}}), std::invalid_argument);     // no class 2
    EXPECT_NO_THROW(set({{1, 0}, {1}}));
    const cfdcore::layout::layout<2> fields({2, 2});
    std::vector<double> out(4);
    const std::vector<double> classes(7, 1.0);
    const std::vector<std::size_t> members = {0};
    EXPECT_THROW(cfdcore::channels::channel_field(fields, classes, members, out),
                 std::invalid_argument);
    std::vector<double> readings(3);
    const std::vector<std::size_t> outside = {4};
    EXPECT_THROW(cfdcore::channels::sensor_readings(std::vector<double>(4, 0.0), 4, outside,
                                                    std::span<double>(readings).first(1)),
                 std::invalid_argument);
}

// --- arrival ----------------------------------------------------------------

TEST(Arrival, ACellExactlyAtItsThresholdIsReachedAndOneBelowIsNot) {
    const box b(8, 6, 0.25);
    const cfdcore::grid::grid<2> g(b.grid_input());
    cfdcore::arrival::maps<2> m(g, {2.0}, {0.0});
    // Three cells of the top row, clear of the SOLID block.
    const std::size_t at = b.cells() - 3;
    std::vector<double> sum(b.cells(), 0.0);
    sum[at] = 2.0;
    sum[at + 1] = std::nextafter(2.0, 0.0);
    sum[at + 2] = std::nextafter(2.0, 3.0);
    m.update(0.5, sum);
    EXPECT_EQ(m.first_above()[at], 0.5);
    EXPECT_TRUE(std::isnan(m.first_above()[at + 1]));
    EXPECT_EQ(m.first_above()[at + 2], 0.5);
    EXPECT_EQ(m.time(), 0.5);
}

TEST(Arrival, AnEventIsMappedFromItsOnsetAndATimeOnceSetIsKept) {
    const box b(8, 6, 0.25);
    const cfdcore::grid::grid<2> g(b.grid_input());
    cfdcore::arrival::maps<2> m(g, {1.0}, {1.0, 2.0});
    const std::vector<double> above(b.cells(), 5.0);
    m.update(0.5, above);  // before both onsets: the time only
    EXPECT_EQ(m.time(), 0.5);
    for (const double x : m.first_above()) {
        EXPECT_TRUE(std::isnan(x));
    }
    m.update(1.0, above);  // event 0 at its onset; event 1 not started
    m.update(2.5, above);  // event 1 from here; event 0 keeps 1.0
    const std::size_t n = b.cells();
    for (std::size_t i = 0; i < n; ++i) {
        if (b.types[i] == solid_type) {
            EXPECT_TRUE(std::isnan(m.first_above()[i]));
            EXPECT_TRUE(std::isnan(m.first_above()[n + i]));
        } else {
            EXPECT_EQ(m.first_above()[i], 1.0);
            EXPECT_EQ(m.first_above()[n + i], 2.5);
        }
    }
}

TEST(Arrival, ARestoreCopiesTheMapsAndTheTimeAndRefusesAnotherLength) {
    const box b(8, 6, 0.25);
    const cfdcore::grid::grid<2> g(b.grid_input());
    cfdcore::arrival::maps<2> m(g, {1.0, 1.0}, {0.0});
    std::vector<double> maps(2 * b.cells(), nan);
    maps[3] = 0.75;
    m.restore(maps, 0.75);
    EXPECT_TRUE(same_bits(m.first_above(), maps));
    EXPECT_EQ(m.time(), 0.75);
    EXPECT_THROW(m.restore(std::span<const double>(maps).first(b.cells()), 0.75),
                 std::invalid_argument);
    EXPECT_THROW(cfdcore::arrival::maps<2>(g, {1.0}, {}), std::invalid_argument);
    EXPECT_THROW(cfdcore::arrival::maps<2>(g, {1.0}, {2.0, 1.0}), std::invalid_argument);
    EXPECT_THROW(cfdcore::arrival::maps<2>(g, {-1.0}, {0.0}), std::invalid_argument);
}

// --- recovery ---------------------------------------------------------------

TEST(Recovery, ACellOrASensorExactlyAtItsThresholdHasNotRecovered) {
    const box b(8, 6, 0.25);
    const cfdcore::grid::grid<2> g(b.grid_input());
    cfdcore::recovery::tracker<2> r(g, {{2.0}, {0.5}, {true}, std::nullopt}, 1);
    std::vector<double> sum(b.cells(), 1.0);
    sum[4] = 2.0;
    const std::vector<double> at = {2.0};
    const std::vector<double> mean = {1.0};
    r.update(0.25, sum, at, at, mean);
    EXPECT_EQ(r.last_above()[4], 0.25);
    EXPECT_TRUE(std::isnan(r.last_above()[5]));
    EXPECT_TRUE(std::isnan(r.class_since()[0]));
    EXPECT_TRUE(std::isnan(r.room_since()));
    const std::vector<double> below = {std::nextafter(2.0, 0.0)};
    r.update(0.5, sum, below, below, mean);
    EXPECT_EQ(r.class_since()[0], 0.5);
    EXPECT_EQ(r.room_since(), 0.5);
    EXPECT_FALSE(r.held());  // no hold
}

TEST(Recovery, TheHoldHoldsAtEqualityAndTheTargetsWhereDefined) {
    const box b(8, 6, 0.25);
    const cfdcore::grid::grid<2> g(b.grid_input());
    cfdcore::recovery::tracker<2> r(g, {{2.0, 2.0}, {0.5, 0.5}, {true, false}, 0.5}, 1);
    const std::vector<double> sum(2 * b.cells(), 0.0);
    const std::vector<double> low = {0.5, 0.5};
    r.update(0.25, sum, low, low, low);
    EXPECT_FALSE(r.held());
    EXPECT_EQ(r.sensor_time()[0], 0.25);
    EXPECT_TRUE(std::isnan(r.sensor_time()[1]));  // class 1's target is not defined
    EXPECT_EQ(r.mean_time()[0], 0.25);
    EXPECT_TRUE(std::isnan(r.mean_time()[1]));
    r.update(0.75, sum, low, low, low);  // 0.75 - 0.25 is the hold exactly
    EXPECT_TRUE(r.held());
    EXPECT_EQ(r.sensor_time()[0], 0.25);  // the first time is kept
    const std::vector<double> above = {2.0, 0.0};
    r.update(1.0, sum, above, above, low);
    EXPECT_FALSE(r.held());
    EXPECT_TRUE(std::isnan(r.room_since()));
    EXPECT_TRUE(std::isnan(r.class_since()[0]));
    EXPECT_EQ(r.class_since()[1], 0.25);
}

TEST(Recovery, ProgrammingErrorsThrow) {
    const box b(8, 6, 0.25);
    const cfdcore::grid::grid<2> g(b.grid_input());
    EXPECT_THROW(cfdcore::recovery::tracker<2>(g, {{1.0}, {1.0}, {true}, 1.0}, 0),
                 std::invalid_argument);
    EXPECT_THROW(cfdcore::recovery::tracker<2>(g, {{1.0}, {1.0, 2.0}, {true}, 1.0}, 1),
                 std::invalid_argument);
    cfdcore::recovery::tracker<2> r(g, {{1.0}, {1.0}, {true}, 1.0}, 1);
    const std::vector<double> one = {1.0};
    EXPECT_THROW(r.update(1.0, one, one, one, one), std::invalid_argument);
    EXPECT_THROW(r.restore(one, one, one, one, 0.0, 0.0), std::invalid_argument);
}

// --- march ------------------------------------------------------------------

struct world {
    world()
        : b(16, 9, 0.25), g(b.grid_input()), s(g), fields(planted(b, 3, 1e3)),
          maps(2 * 3 * b.cells(), nan), cells(3 * b.cells(), nan), sensor_nan(9, nan),
          class_nan(3, nan) {}

    std::unique_ptr<run<2>> make(bool with_recovery, std::optional<double> hold = std::nullopt) {
        std::optional<cfdcore::recovery::keys> rec;
        if (with_recovery) {
            // Targets inside the planted fields' range, so some sensors reach theirs
            // at the first step and a restore that dropped the times would show.
            rec = cfdcore::recovery::keys{
                {3.0, 2.0, 1.0}, {1.5e3, 1.5e3, 1.5e3}, {true, true, false}, hold};
        }
        auto r = std::make_unique<run<2>>(s, b.spacing(), keys(), physics(), setup(b, rec));
        r->restore(fresh(fields, 3, maps, with_recovery ? &cells : nullptr, &sensor_nan,
                         &class_nan));
        return r;
    }

    // An interval of `steps` steps of dt from t_start.
    plan interval(std::int64_t steps, double t_start = 0.0, std::int64_t first = 0) {
        const double dt = 1.0 / 64.0;
        return {t_start, t_start + static_cast<double>(steps) * dt, dt, steps, first};
    }

    box b;
    cfdcore::grid::grid<2> g;
    cfdcore::scalar::scheme<2> s;
    std::vector<double> fields, maps, cells, sensor_nan, class_nan;
};

void expect_same_state(const run<2>& a, const run<2>& b) {
    EXPECT_TRUE(same_bits(a.fields(), b.fields()));
    EXPECT_TRUE(same_bits(a.maps().first_above(), b.maps().first_above()));
    EXPECT_EQ(a.step(), b.step());
    EXPECT_EQ(bits(a.time()), bits(b.time()));
    for (std::size_t k = 0; k < a.classes(); ++k) {
        EXPECT_EQ(bits(*a.budget_of(k).current), bits(*b.budget_of(k).current));
        EXPECT_EQ(bits(a.budget_of(k).leftover), bits(b.budget_of(k).leftover));
        EXPECT_EQ(bits(a.budget_of(k).deposited[0]), bits(b.budget_of(k).deposited[0]));
    }
    if (a.tracker() != nullptr) {
        ASSERT_NE(b.tracker(), nullptr);
        EXPECT_TRUE(same_bits(a.tracker()->last_above(), b.tracker()->last_above()));
        EXPECT_TRUE(same_bits(a.tracker()->class_since(), b.tracker()->class_since()));
        EXPECT_TRUE(same_bits(a.tracker()->sensor_time(), b.tracker()->sensor_time()));
        EXPECT_EQ(bits(a.tracker()->room_since()), bits(b.tracker()->room_since()));
    }
}

TEST(March, AMarchTakesTheStepsAskedForAndStopsAtTheIntervalsEnd) {
    world w;
    auto r = w.make(false);
    const plan p = w.interval(5, 0.5);
    ASSERT_TRUE(r->interval({w.b.faces(), w.b.faces(), w.b.faces()}, w.b.air(), p).ok());
    row_store out(5, 3, 3);
    auto first = r->march(2, out.view());
    EXPECT_EQ(first.steps, 2u);
    EXPECT_EQ(first.reason, stop::count);
    EXPECT_EQ(r->remaining(), 3u);
    EXPECT_EQ(out.step[0], 1);
    EXPECT_EQ(out.step[1], 2);
    EXPECT_EQ(bits(out.t[1]), bits(0.5 + 2.0 * p.dt));
    auto rest = r->march(10, out.view());
    EXPECT_EQ(rest.steps, 3u);
    EXPECT_EQ(rest.reason, stop::interval_end);
    EXPECT_EQ(bits(out.t[2]), bits(p.t_end));
    EXPECT_EQ(r->step(), 5);
    EXPECT_EQ(r->remaining(), 0u);
    EXPECT_EQ(r->march(1, out.view()).steps, 0u);
}

TEST(March, TheIntervalsLastStepIsAtItsEndExactly) {
    world w;
    auto r = w.make(false);
    // An interval whose end is not t_start + steps dt in floating point, its
    // step below the box's stable step.
    const double t_start = 0.1;
    double t_end = 0.0;
    std::int64_t steps = 0;
    for (std::int64_t n = 3; n <= 20 && steps == 0; ++n) {
        for (int k = 1; k < 200; ++k) {
            const double end = t_start + 0.001 * k;
            const double dt = (end - t_start) / static_cast<double>(n);
            if (dt <= 0.01 && t_start + static_cast<double>(n) * dt != end) {
                t_end = end;
                steps = n;
                break;
            }
        }
    }
    ASSERT_GT(steps, 0);
    const double dt = (t_end - t_start) / static_cast<double>(steps);
    const plan p{t_start, t_end, dt, steps, 0};
    ASSERT_TRUE(r->interval({w.b.faces(), w.b.faces(), w.b.faces()}, w.b.air(), p).ok());
    const auto n = static_cast<std::size_t>(steps);
    row_store out(n, 3, 3);
    ASSERT_EQ(r->march(n, out.view()).steps, n);
    EXPECT_EQ(bits(out.t[n - 2]), bits(t_start + static_cast<double>(steps - 1) * dt));
    EXPECT_EQ(bits(out.t[n - 1]), bits(t_end));
    EXPECT_NE(bits(out.t[n - 1]), bits(t_start + static_cast<double>(steps) * dt));
}

TEST(March, StepsInBatchesEqualStepsOneAtATime) {
    world w;
    auto a = w.make(true);
    auto b = w.make(true);
    const plan p = w.interval(7);
    ASSERT_TRUE(a->interval({w.b.faces(), w.b.faces(), w.b.faces()}, w.b.air(), p).ok());
    ASSERT_TRUE(b->interval({w.b.faces(), w.b.faces(), w.b.faces()}, w.b.air(), p).ok());
    row_store out(7, 3, 3);
    EXPECT_EQ(a->march(7, out.view()).steps, 7u);
    for (int i = 0; i < 7; ++i) {
        EXPECT_EQ(b->march(1, out.view()).steps, 1u);
    }
    expect_same_state(*a, *b);
    // The first event's maps were reached; the second's onset is after the
    // interval, so its maps are untouched.
    bool reached = false;
    bool second = false;
    const auto maps = a->maps().first_above();
    for (std::size_t i = 0; i < maps.size(); ++i) {
        const bool set = maps[i] == maps[i];
        (i < 3 * w.b.cells() ? reached : second) |= set;
    }
    EXPECT_TRUE(reached);
    EXPECT_FALSE(second);
}

TEST(March, ARunRestoredFromAnothersStateContinuesItBitForBit) {
    world w;
    auto a = w.make(true);
    const plan p = w.interval(9);
    const auto faces = std::vector{w.b.faces(), w.b.faces(), w.b.faces()};
    ASSERT_TRUE(a->interval(faces, w.b.air(), p).ok());
    row_store out(9, 3, 3);
    ASSERT_EQ(a->march(4, out.view()).steps, 4u);
    const snapshot mid(*a);
    ASSERT_EQ(a->march(5, out.view()).steps, 5u);

    auto b = w.make(true);
    b->restore(mid.view());
    ASSERT_TRUE(b->interval(faces, w.b.air(), w.interval(9, 0.0, 4)).ok());
    ASSERT_EQ(b->march(5, out.view()).steps, 5u);
    expect_same_state(*a, *b);
    // The state handed over held 100:1 times already, which the steps after
    // could not set again: a restore that dropped them would differ here.
    bool reached = false;
    for (const double x : mid.sensor_time) {
        reached = reached || x == x;
    }
    EXPECT_TRUE(reached);
}

TEST(March, TheHoldStopsTheMarchAtTheStepItHolds) {
    world w;
    // Every reading far below its threshold from the first update, so the room is
    // within spec from step 1 and holds at the step whose time less step 1's is
    // the hold exactly.
    for (double& x : w.fields) {
        x *= 1e-6;
    }
    const double dt = 1.0 / 64.0;
    auto r = w.make(true, 3.0 * dt);
    ASSERT_TRUE(r->interval({w.b.faces(), w.b.faces(), w.b.faces()}, w.b.air(),
                            w.interval(10))
                    .ok());
    row_store out(10, 3, 3);
    const auto done = r->march(10, out.view());
    EXPECT_EQ(done.reason, stop::held);
    EXPECT_EQ(done.steps, 4u);
    EXPECT_EQ(r->tracker()->room_since(), dt);
    EXPECT_TRUE(r->held());
    const auto again = r->march(10, out.view());
    EXPECT_EQ(again.steps, 0u);
    EXPECT_EQ(again.reason, stop::held);
}

TEST(March, TheMeansAreTheContentOverTheFreeArea) {
    world w;
    auto r = w.make(true);
    ASSERT_TRUE(
        r->interval({w.b.faces(), w.b.faces(), w.b.faces()}, w.b.air(), w.interval(1)).ok());
    row_store out(1, 3, 3);
    ASSERT_EQ(r->march(1, out.view()).steps, 1u);
    double free = 0.0;
    for (std::size_t i = 0; i < w.b.cells(); ++i) {
        free += w.b.types[i] == solid_type ? 0.0 : 0.0625;
    }
    EXPECT_EQ(r->free_area(), free);  // powers of two: every order gives one sum
    for (std::size_t k = 0; k < 3; ++k) {
        EXPECT_EQ(bits(out.means[k]), bits(*r->budget_of(k).current / r->free_area()));
    }
}

TEST(March, ProgrammingErrorsThrow) {
    world w;
    run<2> bare(w.s, w.b.spacing(), keys(), physics(), setup(w.b, std::nullopt));
    const auto faces = std::vector{w.b.faces(), w.b.faces(), w.b.faces()};
    EXPECT_THROW((void)bare.interval(faces, w.b.air(), w.interval(3)), std::invalid_argument);
    auto r = w.make(false);
    row_store out(3, 3, 3);
    EXPECT_THROW((void)r->march(1, out.view()), std::invalid_argument);  // no interval
    EXPECT_THROW((void)r->interval(faces, w.b.air(), w.interval(3, 0.0, 3)), std::invalid_argument);
    ASSERT_TRUE(r->interval(faces, w.b.air(), w.interval(3)).ok());
    row_store short_rows(1, 3, 3);
    EXPECT_THROW((void)r->march(2, short_rows.view()), std::invalid_argument);
    resident wrong = fresh(w.fields, 3, w.maps, nullptr, nullptr, nullptr);
    wrong.budgets.pop_back();
    EXPECT_THROW(r->restore(wrong), std::invalid_argument);
    const resident with_cells = fresh(w.fields, 3, w.maps, &w.cells, &w.sensor_nan, &w.class_nan);
    EXPECT_THROW(r->restore(with_cells), std::invalid_argument);  // no recovery here
    auto setup_bad = setup(w.b, std::nullopt);
    setup_bad.sensors.push_back(w.b.cells());
    EXPECT_THROW(run<2>(w.s, w.b.spacing(), keys(), physics(), setup_bad), std::invalid_argument);
}

}  // namespace
