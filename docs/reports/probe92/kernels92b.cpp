// Prompt 92b's kernel timing: each kernel of momentum.cpp, timed alone on the
// rows of a 320 by 120 field, and the passes that call them. Includes the
// module's source, so that the same file times the commit before the
// restructure (HEAD_FORM) and the one after. Built and run by
// `probe92.py kernels`; prints one JSON object of microseconds per pass.

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <vector>

// The passes are private members. Every header momentum.cpp reads is included
// above or here before the word is redefined, so only the module's own is changed.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>
#include "cfdcore/grid.hpp"
#include "cfdcore/scalar.hpp"
#define private public
#include "momentum.cpp"
#undef private

using namespace cfdcore;
using namespace cfdcore::momentum;

namespace {

constexpr std::size_t nx = 320;
constexpr std::size_t ny = 120;
constexpr std::size_t width = nx + 8;
constexpr int batches = 41;
constexpr int passes = 8;

double hashed(std::size_t i) {
    std::uint64_t z = 0x9e3779b97f4a7c15ULL * (i + 1);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z ^= z >> 31;
    return static_cast<double>(z >> 11) * 0x1.0p-53;
}

// The smallest batch's microseconds for one call of `pass`.
template <typename Pass>
double timed(Pass&& pass) {
    double best = 1e300;
    for (int b = 0; b < batches; ++b) {
        const auto start = std::chrono::steady_clock::now();
        for (int p = 0; p < passes; ++p) {
            pass();
            asm volatile("" : : : "memory");
        }
        const std::chrono::duration<double, std::micro> spent =
            std::chrono::steady_clock::now() - start;
        best = std::min(best, spent.count() / passes);
    }
    return best;
}

bool first_result = true;
template <typename Pass>
void report(const std::string& name, Pass&& pass) {
    std::printf("%s\"%s\": %.4f", first_result ? "{" : ", ", name.c_str(), timed(pass));
    first_result = false;
}

// A field of rows with room about it, so that a row's neighbours exist.
struct rows {
    rows(std::size_t seed, double low, double high) : values((ny + 8) * width) {
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] = low + (high - low) * hashed(seed + i);
        }
    }
    double* row(std::size_t j) { return values.data() + (j + 4) * width + 4; }
    std::vector<double> values;
};
struct mask_rows {
    mask_rows(std::size_t seed, double share) : values((ny + 8) * width) {
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] = lanes::mask_of(hashed(seed + i) < share);
        }
    }
    lanes::mask* row(std::size_t j) { return values.data() + (j + 4) * width + 4; }
    std::vector<lanes::mask> values;
};

// Read at run time, so that a division by it stays one.
volatile double alpha = 0.5;

rows a(1, -1.0, 1.0), b(1000003, -1.0, 1.0), e(5000011, 0.1, 2.0), f(7000003, 0.1, 2.0);
rows g(11000027, -1.0, 1.0), h(13000027, 0.1, 1.0), w(17000023, 0.5, 1.5), o1(19, 0.0, 1.0);
rows o2(23, 0.0, 1.0), o3(29, 0.0, 1.0), o4(31, 0.0, 1.0), o5(37, 0.5, 1.0), o6(41, 0.0, 1.0);
rows o7(43, 0.0, 1.0);
mask_rows m1(47, 0.1), m2(53, 0.9), m3(59, 0.95), m4(61, 0.02);

template <typename Body>
void each_row(Body&& body) {
    for (std::size_t j = 0; j < ny; ++j) {
        body(j);
    }
}

stream_run stream_of(std::size_t j, std::size_t stride) {
    stream_run r{};
    r.low = a.row(j);
    r.high = r.low + stride;
    r.far_low = r.low - stride;
    r.far_high = r.high + stride;
    for (int m = 0; m < 3; ++m) {
        r.std_pos[m] = w.row(0) + m;
        r.std_neg[m] = w.row(1) + m;
        r.alt_pos[m] = w.row(2) + m;
        r.alt_neg[m] = w.row(3) + m;
    }
    r.blocked_pos = m1.row(j);
    r.blocked_neg = m4.row(j);
    r.flux = g.row(j);
    return r;
}
cross_run cross_of(std::size_t j, std::size_t stride) {
    cross_run r{};
    r.low = a.row(j);
    r.high = r.low + stride;
    r.far_low = r.low - stride;
    r.far_high = r.high + stride;
    for (int m = 0; m < 3; ++m) {
        r.pos[m] = w.row(0) + m;
        r.neg[m] = w.row(1) + m;
    }
    for (int m = 0; m < 2; ++m) {
        r.wall_pos[m] = w.row(2) + m;
        r.wall_neg[m] = w.row(3) + m;
    }
    r.far_wall_low = m1.row(j);
    r.far_wall_high = m4.row(j);
    r.wall_face = m4.row(j);
    r.flux = g.row(j);
    return r;
}

// The kernels of the component along direction C, whose transverse direction is T.
template <int C>
void kernels() {
    constexpr int T = 1 - C;
    const std::string s = C == 0 ? " u" : " v";
    const std::size_t stride_s = C == 0 ? 1 : width;
    const std::size_t stride_t = C == 0 ? width : 1;
#ifdef HEAD_FORM
    constexpr int cross = C;
#else
    constexpr int cross = T;
    constexpr transverse_directions<2, C> across{};
#endif
    const double* wt = w.row(0);
    const double* ws = w.row(1);

    report("cell_conductance" + s, [&] {
        each_row([&](std::size_t j) {
#ifdef HEAD_FORM
            lanes::for_each_block(0, nx, [&](auto cells, std::size_t m) {
                cell_conductance<C>(cells, e.row(j), wt, ws, m, o1.row(0));
            });
#else
            const cell_conductance_run<1> r{e.row(j), {wt}, ws};
            lanes::for_each_block(0, nx, [&](auto cells, std::size_t m) {
                cell_conductance<C>(cells, across, r, m, o1.row(0));
            });
#endif
        });
    });
    report("cell_flux" + s, [&] {
        each_row([&](std::size_t j) {
            const double* low = a.row(j);
#ifdef HEAD_FORM
            cell_flux_row<C>({low, low + stride_s, wt, e.row(j), 0.6}, nx, o1.row(0), o2.row(0),
                             o3.row(0));
#else
            cell_flux_row<C>(across, cell_flux_run<1>{low, low + stride_s, {wt}, e.row(j), 0.6},
                             nx, o1.row(0), o2.row(0), o3.row(0));
#endif
        });
    });
    report("cell_stress" + s, [&] {
        each_row([&](std::size_t j) {
            const double* low = a.row(j);
#ifdef HEAD_FORM
            lanes::for_each_block(0, nx, [&](auto cells, std::size_t m) {
                cell_stress<C>(cells, low, low + stride_s, ws, wt, m, o1.row(0));
            });
#else
            const cell_stress_run<1> r{low, low + stride_s, ws, {wt}};
            lanes::for_each_block(0, nx, [&](auto cells, std::size_t m) {
                cell_stress<C>(cells, across, r, m, o1.row(0));
            });
#endif
        });
    });
    for (const bool obstacles : {false, true}) {
        report(std::string("cell_deferred") + (obstacles ? " obstacles" : "") + s, [&] {
            each_row([&](std::size_t j) {
                stream_row<C, C == 0>(stream_of(j, stride_s), obstacles, 0, nx, o1.row(0));
            });
        });
        report(std::string("corner_deferred") + (obstacles ? " obstacles" : "") + s, [&] {
            each_row([&](std::size_t j) {
                cross_row<cross, C == 1>(cross_of(j, stride_t), obstacles, 0, nx, o1.row(0));
            });
        });
    }
    report("corner_flux" + s, [&] {
        each_row([&](std::size_t j) {
            const double* low = b.row(j);
            corner_flux_row<C>({low, low + stride_s, wt, ws, e.row(j), 0.6}, nx, o1.row(0),
                               o2.row(0), o3.row(0));
        });
    });
    report("corner_stress" + s, [&] {
        each_row([&](std::size_t j) {
            const double* low = b.row(j);
            lanes::for_each_block(0, nx, [&](auto cells, std::size_t m) {
                corner_stress<C>(cells, low, low + stride_s, ws, m, o1.row(0));
            });
        });
    });
    const auto conductance_of = [&](std::size_t j) {
        return corner_conductance_run{e.row(j), f.row(j),  ws,       h.row(j),
                                      o5.row(j), m1.row(j), m2.row(j)};
    };
    report("corner_conductance field" + s, [&] {
        each_row([&](std::size_t j) {
            corner_conductance_row<C, true, false>(conductance_of(j), nx, o1.row(0));
        });
    });
    report("corner_conductance wall" + s, [&] {
        each_row([&](std::size_t j) {
            corner_conductance_row<C, false, true>(conductance_of(j), nx, o1.row(0));
        });
    });
    report("corner_conductance field wall" + s, [&] {
        each_row([&](std::size_t j) {
            corner_conductance_row<C, true, true>(conductance_of(j), nx, o1.row(0));
        });
    });
    report("harmonic_cells" + s, [&] {
        each_row([&](std::size_t j) {
            const double* low = e.row(j);
            lanes::for_each_block(0, nx, [&](auto cells, std::size_t m) {
                harmonic_cells<cross>(cells, low, low + stride_t, m4.row(j),
                                      m4.row(j) + stride_t, h.row(0), h.row(1), m, o1.row(0));
            });
        });
    });
    report("mean_cells" + s, [&] {
        each_row([&](std::size_t j) {
            lanes::for_each_block(1, nx, [&](auto cells, std::size_t m) {
                mean_cells<C>(cells, e.row(j), C == 0 ? e.row(j) : e.row(j + 1), h.row(0), m,
                              o1.row(0));
            });
        });
    });
    // The coefficients, the diagonal and the relaxed one, with what makes them.
    report("coefficients" + s, [&] {
        each_row([&](std::size_t j) {
#ifdef HEAD_FORM
            coefficient_run r{};
            r.cell_flux = a.row(1);
            r.cell_flux_below = a.row(2);
            r.cell_plus = e.row(1);
            r.cell_minus = f.row(1);
            r.cell_minus_below = f.row(2);
            r.corner_flux = g.row(1);
            r.corner_flux_above = g.row(2);
            r.corner_plus = h.row(1);
            r.corner_plus_above = h.row(2);
            r.corner_minus = w.row(1);
            r.unknown = m3.row(j);
            r.couples_high = m2.row(j);
            r.couples_low = m2.row(j + 1);
            r.edge_known = m1.row(0);
            r.edge_value = b.row(0);
            r.alpha = alpha;
            if constexpr (C == 0) {
                coefficient_row<C, beside::none>(r, 1, nx, o1.row(j), o2.row(j), o3.row(j),
                                                 o4.row(j), o5.row(j), o6.row(j), o7.row(j));
            } else {
                coefficient_row<C, beside::low>(r, 0, 1, o1.row(j), o2.row(j), o3.row(j),
                                                o4.row(j), o5.row(j), o6.row(j), o7.row(j));
                coefficient_row<C, beside::none>(r, 1, nx - 1, o1.row(j), o2.row(j), o3.row(j),
                                                 o4.row(j), o5.row(j), o6.row(j), o7.row(j));
                coefficient_row<C, beside::high>(r, nx - 1, nx, o1.row(j), o2.row(j), o3.row(j),
                                                 o4.row(j), o5.row(j), o6.row(j), o7.row(j));
            }
#else
            const std::size_t first = C == 0 ? 1 : 0;
            diagonal_run<1> r{};
            r.stream = {a.row(1), a.row(2), e.row(1), f.row(1), f.row(2)};
            r.cross[0] = {g.row(1), g.row(2), h.row(1), h.row(2), w.row(1)};
            r.unknown = m3.row(j);
            stream_neighbour_row<C>(r.stream, r.unknown, first, nx, o1.row(j), o2.row(j));
            cross_neighbour_row<T>(r.cross[0], r.unknown, m2.row(j), m2.row(j + 1), first, nx,
                                   o3.row(j), o4.row(j));
            r.alpha = alpha;
            diagonal_row<C>(across, r, first, nx, o5.row(j), o6.row(j), o7.row(j));
            if constexpr (C == 1) {
                edge_run low{w.row(1), m1.row(0), b.row(0), m3.row(j), alpha};
                edge_unknown_row<T>(low, 0, 1, o4.row(j), o5.row(j), o6.row(j), o7.row(j));
                edge_run high{h.row(1) + 1, m1.row(0), b.row(0), m3.row(j), alpha};
                edge_unknown_row<T>(high, nx - 1, nx, o3.row(j), o5.row(j), o6.row(j),
                                    o7.row(j));
            }
#endif
        });
    });
    const auto sources_of = [&]<bool Field>(std::size_t j) {
#ifdef HEAD_FORM
        source_run r{};
        r.corner_deferred = b.row(1);
        r.corner_deferred_above = b.row(2);
        r.corner_mu = f.row(1);
        r.corner_mu_above = f.row(2);
        r.corner_stress = g.row(3);
        r.corner_stress_above = g.row(4);
        r.width_t = wt;
#else
        source_run<1> r{};
        r.cross[0] = {b.row(1), b.row(2), f.row(1), f.row(2), g.row(3), g.row(4)};
        r.width_t[0] = wt;
#endif
        r.cell_deferred = a.row(1);
        r.cell_deferred_below = a.row(2);
        r.p = g.row(j);
        r.p_below = C == 0 ? g.row(j) : g.row(j + 1);
        r.own = a.row(j);
        r.a_p = o5.row(j);
        r.mu = e.row(j);
        r.mu_below = C == 0 ? e.row(j) : e.row(j + 1);
        r.mean_high = h.row(0);
        r.cell_stress = g.row(1);
        r.cell_stress_below = g.row(2);
        r.relaxation = 1.0;
        const std::size_t first = C == 0 ? 1 : 0;
        std::fill(o7.row(j), o7.row(j) + nx, 0.0);
#ifdef HEAD_FORM
        source_row<C, Field>(r, first, nx, o7.row(j));
#else
        source_row<C, Field>(across, r, first, nx, o7.row(j));
#endif
    };
    report("sources field" + s, [&] {
        each_row([&](std::size_t j) { sources_of.template operator()<true>(j); });
    });
    report("sources air" + s, [&] {
        each_row([&](std::size_t j) { sources_of.template operator()<false>(j); });
    });
    const auto sweeps_of = [&]<bool Held>(std::size_t j) {
#ifdef HEAD_FORM
        sweep_run r{};
        r.below = a.row(j) - width;
        r.above = a.row(j) + width;
        r.a_sp = e.row(j);
        r.a_sm = f.row(j);
        r.a_tp = h.row(j);
        r.a_tm = w.row(j);
#else
        sweep_run<2> r{};
        r.below[1] = a.row(j) - width;
        r.above[1] = a.row(j) + width;
        r.plus[0] = e.row(j);
        r.minus[0] = f.row(j);
        r.plus[1] = h.row(j);
        r.minus[1] = w.row(j);
#endif
        r.x = a.row(j);
        r.b = b.row(j);
        r.relaxed = o6.row(j);
        r.a_p = o5.row(j);
        r.held = m4.row(j);
        r.own = g.row(j);
        sweep_row<C, Held>(r, o3.row(0), C == 0 ? 1 : 0, nx, o7.row(j));
    };
    report("sweep_cells" + s, [&] {
        each_row([&](std::size_t j) { sweeps_of.template operator()<false>(j); });
    });
    report("sweep_cells held" + s, [&] {
        each_row([&](std::size_t j) { sweeps_of.template operator()<true>(j); });
    });
}

// The passes themselves, on a predictor of a room with obstacles.
void whole() {
    std::vector<double> x(nx + 1), xc(nx), y(ny + 1), yc(ny);
    std::vector<double> dx(nx), dy(ny), dxf(nx + 1), dyf(ny + 1);
    for (std::size_t i = 0; i <= nx; ++i) {
        x[i] = 0.025 * static_cast<double>(i);
        dxf[i] = i == 0 || i == nx ? 0.0125 : 0.025;
    }
    for (std::size_t j = 0; j <= ny; ++j) {
        y[j] = 0.025 * static_cast<double>(j);
        dyf[j] = j == 0 || j == ny ? 0.0125 : 0.025;
    }
    for (std::size_t i = 0; i < nx; ++i) {
        xc[i] = 0.025 * (static_cast<double>(i) + 0.5);
        dx[i] = 0.025;
    }
    for (std::size_t j = 0; j < ny; ++j) {
        yc[j] = 0.025 * (static_cast<double>(j) + 0.5);
        dy[j] = 0.025;
    }
    std::vector<std::int32_t> types(nx * ny, 0);
    for (std::size_t j = 0; j < 36; ++j) {
        for (std::size_t i = 80; i < 130; ++i) {
            types[j * nx + i] = types[j * nx + i + 110] = 1;
        }
    }
    const grid::grid<2> mesh(grid::grid_input<2>{
        .faces = {x, y}, .centres = {xc, yc}, .cell_type = types, .solid = 1});
    static std::array<bool, nx + 1> known_x;
    static std::array<bool, ny + 1> known_y;
    known_x.fill(true);
    known_y.fill(true);
    const std::vector<double> zero_x(nx + 1, 0.0), zero_y(ny + 1, 0.0);
    predictor_input<2> in{};
    in.cell_widths = {dx, dy};
    in.face_widths = {dxf, dyf};
    const edge_condition ex{known_x, zero_x, 0.0125};
    const edge_condition ey{known_y, zero_y, 0.0125};
#ifdef HEAD_FORM
    in.low = {ex, ey};
    in.high = {ex, ey};
#else
    in.low[0][0] = in.high[0][0] = ex;
    in.low[1][0] = in.high[1][0] = ey;
#endif
    in.rho = 1.2;
    in.mu = 1.8e-5;
    in.alpha = 0.5;
    in.sweeps = 1;
    predictor<2> made(mesh, in);
    std::vector<double> u(ny * (nx + 1)), v((ny + 1) * nx), p(nx * ny), mu(nx * ny);
    for (std::size_t i = 0; i < u.size(); ++i) {
        u[i] = hashed(i) - 0.5;
    }
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = hashed(i + 71) - 0.5;
    }
    for (std::size_t i = 0; i < p.size(); ++i) {
        p[i] = hashed(i + 73);
        mu[i] = 1e-4 + 1e-3 * hashed(i + 79);
    }
    std::array<std::vector<double>, 2> walls{std::vector<double>((nx + 1) * (ny + 1), 1e-4),
                                             std::vector<double>((nx + 1) * (ny + 1), 1e-4)};
    std::array<std::vector<double>, 2> star{u, v}, diagonal{u, v}, out{walls};
#ifdef HEAD_FORM
    const std::array<std::span<const double>, 2> given{walls[0], walls[1]};
    const std::array<std::span<double>, 2> stencil{out[0], out[1]};
    const std::array<std::span<const double>, 2> none{};
#else
    const per_pair<std::span<const double>, 2> given{{{walls[0]}, {walls[1]}}};
    const per_pair<std::span<double>, 2> stencil{{{out[0]}, {out[1]}}};
    const per_pair<std::span<const double>, 2> none{};
#endif
    for (int c = 0; c < 2; ++c) {
        const std::string s = c == 0 ? " u" : " v";
        report("equations field wall" + s,
               [&] { made.equations(c, {u, v}, p, mu, given, diagonal[c]); });
        report("equations air" + s, [&] { made.equations(c, {u, v}, p, {}, none, diagonal[c]); });
    }
    // The passes of a row, over every row, with a viscosity field and a wall
    // viscosity and on air's.
    for (const double* m : {static_cast<const double*>(mu.data()),
                            static_cast<const double*>(nullptr)}) {
        const std::string s = m == nullptr ? " air" : " field";
        const double* wall_u = m == nullptr ? nullptr : walls[0].data();
        const double* wall_v = m == nullptr ? nullptr : walls[1].data();
#ifdef HEAD_FORM
        auto& kept = made.cell_rows_;
        auto& corners = made.corner_rows_;
        report("cell_row u" + s, [&] {
            each_row([&](std::size_t j) { made.cell_row<0>(j, u.data(), m, kept[0]); });
        });
        report("cell_row v" + s, [&] {
            each_row([&](std::size_t j) { made.cell_row<1>(j, v.data(), m, kept[0]); });
        });
        report("corner_row u" + s, [&] {
            each_row([&](std::size_t j) {
                made.corner_row<0>(j, u.data(), v.data(), m, wall_u, corners[0]);
            });
        });
        if (m != nullptr) {
            double* to = corners[0].viscosity.data();
            report("corner_viscosity u", [&] {
                each_row([&](std::size_t j) { made.corner_viscosity<0>(j, m, to); });
            });
            report("corner_viscosity v", [&] {
                for (std::size_t j = 1; j < ny; ++j) {
                    made.corner_viscosity<1>(j, m, to);
                }
            });
        }
        report("corner_row v" + s, [&] {
            for (std::size_t j = 1; j < ny; ++j) {
                made.corner_row<1>(j, v.data(), u.data(), m, wall_v, corners[0]);
            }
        });
        report("unknown_row u" + s, [&] {
            each_row([&](std::size_t j) {
                made.unknown_row<0>(j, u.data(), p.data(), m, &kept[0], &kept[0], &corners[0],
                                    &corners[1], diagonal[0].data());
            });
        });
        report("unknown_row v" + s, [&] {
            for (std::size_t j = 1; j < ny; ++j) {
                made.unknown_row<1>(j, v.data(), p.data(), m, &kept[0], &kept[1], &corners[0],
                                    &corners[0], diagonal[1].data());
            }
        });
#else
        using place = std::array<std::size_t, 2>;
        auto& kept = made.cell_rows_;
        auto& corners = made.corner_rows_[0];
        report("cell_row u" + s, [&] {
            each_row([&](std::size_t j) { made.cell_row<0>(place{0, j}, u.data(), m, kept[0]); });
        });
        report("cell_row v" + s, [&] {
            each_row([&](std::size_t j) { made.cell_row<1>(place{0, j}, v.data(), m, kept[0]); });
        });
        report("corner_row u" + s, [&] {
            each_row([&](std::size_t j) {
                made.corner_row<0, 1>(place{0, j}, u.data(), v.data(), m, wall_u, corners[0]);
            });
        });
        if (m != nullptr) {
            double* to = corners[0].viscosity.data();
            report("corner_viscosity u", [&] {
                each_row([&](std::size_t j) { made.corner_viscosity<0, 1>(place{0, j}, m, to); });
            });
            report("corner_viscosity v", [&] {
                for (std::size_t j = 1; j < ny; ++j) {
                    made.corner_viscosity<1, 0>(place{0, j}, m, to);
                }
            });
        }
        report("corner_row v" + s, [&] {
            for (std::size_t j = 1; j < ny; ++j) {
                made.corner_row<1, 0>(place{0, j}, v.data(), u.data(), m, wall_v, corners[0]);
            }
        });
        report("unknown_row u" + s, [&] {
            each_row([&](std::size_t j) {
                made.unknown_row<0>(place{0, j}, u.data(), p.data(), m, {&kept[0], &kept[0]},
                                    {{{&corners[0], &corners[1]}}}, diagonal[0].data());
            });
        });
        report("unknown_row v" + s, [&] {
            for (std::size_t j = 1; j < ny; ++j) {
                made.unknown_row<1>(place{0, j}, v.data(), p.data(), m, {&kept[0], &kept[1]},
                                    {{{&corners[0], &corners[0]}}}, diagonal[1].data());
            }
        });
#endif
    }
    made.equations(0, {u, v}, p, mu, given, diagonal[0]);
    report("sweep u", [&] { made.sweep<0>(u.data(), diagonal[0].data(), star[0].data()); });
    made.equations(1, {u, v}, p, mu, given, diagonal[1]);
    report("sweep v", [&] { made.sweep<1>(v.data(), diagonal[1].data(), star[1].data()); });
    report("predict one sweep field wall", [&] {
        made.predict({u, v}, p, mu, given, {star[0], star[1]}, {diagonal[0], diagonal[1]});
    });
    static std::array<bool, ny * (nx + 1)> held_u;
    static std::array<bool, (ny + 1) * nx> held_v;
    for (std::size_t k = 0; k < held_u.size(); ++k) {
        held_u[k] = hashed(k) < 0.01;
    }
    for (std::size_t k = 0; k < held_v.size(); ++k) {
        held_v[k] = hashed(k + 83) < 0.01;
    }
    in.held = {held_u, held_v};
    predictor<2> vents(mesh, in);
    vents.equations(0, {u, v}, p, mu, given, diagonal[0]);
    report("sweep held u", [&] { vents.sweep<0>(u.data(), diagonal[0].data(), star[0].data()); });
    vents.equations(1, {u, v}, p, mu, given, diagonal[1]);
    report("sweep held v", [&] { vents.sweep<1>(v.data(), diagonal[1].data(), star[1].data()); });
    in.held = {};
    in.sweeps = 10;
    predictor<2> ten(mesh, in);
    report("predict ten sweeps field wall", [&] {
        ten.predict({u, v}, p, mu, given, {star[0], star[1]}, {diagonal[0], diagonal[1]});
    });
    report("predict ten sweeps air", [&] {
        ten.predict({u, v}, p, {}, none, {star[0], star[1]}, {diagonal[0], diagonal[1]});
    });
    report("stencil_viscosity", [&] { made.stencil_viscosity(mu, stencil); });
    report("verdict_cells", [&] {
        if (made.check_viscosity(mu) != viscosity_status::ok) {
            std::abort();
        }
    });
}

}  // namespace

// argv[1]: how many blocks to hold on the heap before the predictor is built, which
// moves its arrays: where they fall changes the passes' times by a few hundredths.
int main(int argc, char** argv) {
    std::vector<std::unique_ptr<char[]>> held;
    for (int k = 0; argc > 1 && k < std::atoi(argv[1]); ++k) {
        held.emplace_back(new char[2300 + 700 * static_cast<std::size_t>(k)]);
    }
    kernels<0>();
    kernels<1>();
    report("edge_nodes", [&] {
        each_row([&](std::size_t j) {
            lanes::for_each_block(0, nx, [&](auto cells, std::size_t m) {
                edge_nodes(cells, m1.row(j), a.row(j), b.row(j), m, o1.row(0));
            });
        });
    });
    report("read_cells", [&] {
        each_row([&](std::size_t j) {
            lanes::for_each_block(0, nx, [&](auto cells, std::size_t m) {
                read_cells(cells, m1.row(j), e.row(j), 1.8e-5, m, o1.row(j));
            });
        });
    });
    whole();
    std::printf("}\n");
    return 0;
}
