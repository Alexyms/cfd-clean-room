// The modules written per direction, instantiated at three (ADR-017 decision
// 4; review 92 S4). Nothing in the core is built for D = 3: this program
// compiles the modules' sources itself, so that a change which makes one of
// them two-directional again fails here, and it is a program of its own so
// that the instantiations reach no other.
//
// A 2D room extruded along each of the three directions in turn, its own two
// axes on the other two in both orders, must give the 2D room's equations bit
// for bit on every plane. The extruded direction's cells have unit width: a
// transverse face's area does not carry the width along a third direction yet
// (docs/reports/phase6_momentum.md, section 7).

#include "../src/grid.cpp"
#include "../src/layout.cpp"
#include "../src/momentum.cpp"
#include "../src/multigrid.cpp"
#include "../src/pressure.cpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <span>
#include <vector>

template class cfdcore::layout::layout<3>;
template class cfdcore::grid::grid<3>;
template class cfdcore::momentum::predictor<3>;
template struct cfdcore::multigrid::system<3>;
template class cfdcore::multigrid::hierarchy<3>;
template class cfdcore::pressure::corrector<3>;

// In the module's namespace, so the position helpers of momentum.cpp are in scope.
namespace cfdcore::momentum {
namespace {

using position3 = std::array<std::size_t, 3>;

constexpr std::size_t depth = 3;  // cells along the extruded direction
constexpr double alpha = 1.0;     // the source then holds no relaxation term
constexpr int rooms = 60;

std::uint64_t bits(double x) { return std::bit_cast<std::uint64_t>(x); }

// Values in [0, 1), the same sequence in every run.
class draws {
public:
    explicit draws(std::size_t seed) : state_(seed) {}
    double operator()() {
        std::uint64_t z = 0x9e3779b97f4a7c15ULL * (++state_);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z ^= z >> 31;
        return static_cast<double>(z >> 11) * 0x1.0p-53;
    }
    std::vector<double> field(std::size_t n, double low) {
        std::vector<double> out(n);
        for (double& x : out) {
            x = low + (*this)();
        }
        return out;
    }

private:
    std::size_t state_;
};

struct axis {
    std::vector<double> faces{0.0};
    std::vector<double> centres;
    std::vector<double> widths;
    std::vector<double> between;
};

axis make_axis(const std::vector<double>& widths) {
    axis a;
    a.widths = widths;
    for (const double w : widths) {
        a.centres.push_back(a.faces.back() + 0.5 * w);
        a.faces.push_back(a.faces.back() + w);
    }
    a.between.push_back(a.centres.front() - a.faces.front());
    for (std::size_t i = 1; i < widths.size(); ++i) {
        a.between.push_back(a.centres[i] - a.centres[i - 1]);
    }
    a.between.push_back(a.faces.back() - a.centres.back());
    return a;
}

// An edge's condition with its arrays (std::vector<bool> holds bits).
struct edge {
    explicit edge(std::size_t n) : known(new bool[n]), value(n, 0.0) {
        std::fill(known.get(), known.get() + n, false);
    }
    edge_condition condition() const {
        return {std::span<const bool>(known.get(), value.size()), value, distance};
    }
    std::unique_ptr<bool[]> known;
    std::vector<double> value;
    double distance = 0.5;
};

// A 2D room: axes of random widths, SOLID cells, edges part Dirichlet and part
// open, and the arrays a call reads.
struct room2 {
    room2(int trial, draws& draw)
        : n{3 + static_cast<std::size_t>(trial) % 11, 2 + static_cast<std::size_t>(trial) % 7},
          axes{make_axis(draw.field(n[0], 0.5)), make_axis(draw.field(n[1], 0.5))},
          types(n[0] * n[1], 0) {
        for (auto& type : types) {
            type = draw() < 0.2 ? 1 : 0;
        }
        for (int c = 0; c < 2; ++c) {
            for (int side = 0; side < 2; ++side) {
                edge e(n[c] + 1);
                for (std::size_t i = 0; i < n[c] + 1; ++i) {
                    e.known[i] = draw() < 0.6;
                    e.value[i] = e.known[i] ? draw() - 0.5 : 0.0;
                }
                e.distance = 0.3 + draw();
                edges.push_back(std::move(e));
            }
        }
        velocity = {draw.field(n[1] * (n[0] + 1), -0.5), draw.field((n[1] + 1) * n[0], -0.5)};
        pressure = draw.field(n[0] * n[1], 0.0);
        viscosity = draw.field(n[0] * n[1], 0.01);
        const std::size_t corners = (n[0] + 1) * (n[1] + 1);
        wall = {draw.field(corners, 0.01), draw.field(corners, 0.01)};
    }

    const edge& edge_of(int c, int side) const {
        return edges[static_cast<std::size_t>(2 * c + side)];
    }

    std::array<std::size_t, 2> n;
    std::array<axis, 2> axes;
    std::vector<std::int32_t> types;
    std::vector<edge> edges;
    std::array<std::vector<double>, 2> velocity;
    std::vector<double> pressure;
    std::vector<double> viscosity;
    std::array<std::vector<double>, 2> wall;
};

grid::grid<2> grid_of(const room2& r) {
    return grid::grid<2>(grid::grid_input<2>{.faces = {r.axes[0].faces, r.axes[1].faces},
                                             .centres = {r.axes[0].centres, r.axes[1].centres},
                                             .cell_type = r.types,
                                             .solid = 1});
}

predictor_input<2> input_of(const room2& r) {
    predictor_input<2> in{};
    in.cell_widths = {r.axes[0].widths, r.axes[1].widths};
    in.face_widths = {r.axes[0].between, r.axes[1].between};
    for (int c = 0; c < 2; ++c) {
        in.low[c][0] = r.edge_of(c, 0).condition();
        in.high[c][0] = r.edge_of(c, 1).condition();
    }
    in.rho = 1.2;
    in.mu = 0.02;
    in.alpha = alpha;
    in.sweeps = 1;
    return in;
}

// The room extruded: its axis 0 along direction `a`, its axis 1 along `b` and
// `depth` cells of unit width along `e`.
struct room3 {
    room3(const room2& r, int a_, int b_, int e_)
        : a(a_), b(b_), e(e_), from(&r), unit(make_axis(std::vector<double>(depth, 1.0))),
          n{}, cells(counts()) {
        types.resize(cells.cell_count());
        pressure.resize(cells.cell_count());
        viscosity.resize(cells.cell_count());
        for_each_cell(cells, [&](const position3& at) {
            const std::size_t i = index_of(cells, at);
            const std::size_t k = at[b] * r.n[0] + at[a];
            types[i] = r.types[k];
            pressure[i] = r.pressure[k];
            viscosity[i] = r.viscosity[k];
        });
        for (int c = 0; c < 3; ++c) {
            auto size = n;
            ++size[c];
            faces.emplace_back(size);
            velocity[c].assign(faces[c].cell_count(), 0.0);
            if (component(c) >= 0) {
                for_each_cell(faces[c], [&](const position3& at) {
                    velocity[c][index_of(faces[c], at)] =
                        r.velocity[component(c)][plane(component(c), at)];
                });
            }
        }
        for (int c = 0; c < 3; ++c) {
            for (int q = 0; q < 2; ++q) {
                add_pair(c, transverse(c, q), q);
            }
        }
    }

    // The 2D component a 3D one is, or -1 for the one along the extrusion.
    int component(int c) const { return c == a ? 0 : c == b ? 1 : -1; }

    // The index in the 2D component's faces of a face of the 3D one.
    std::size_t plane(int m, const position3& at) const {
        return at[b] * (from->n[0] + (m == 0 ? 1 : 0)) + at[a];
    }

    std::size_t corner(const position3& at) const { return at[b] * (from->n[0] + 1) + at[a]; }

    const axis& axis_of(int d) const {
        return d == a ? from->axes[0] : d == b ? from->axes[1] : unit;
    }

    std::array<std::size_t, 3> counts() {
        n[a] = from->n[0];
        n[b] = from->n[1];
        n[e] = depth;
        return n;
    }

    void add_pair(int c, int t, int q) {
        const int m = component(c);
        const std::size_t count = cells.cell_count() / n[c] / n[t] * (n[c] + 1);
        for (int side = 0; side < 2; ++side) {
            edge& made = edges.emplace_back(count);
            if (m >= 0 && t != e) {
                const edge& source = from->edge_of(m, side);
                made.distance = source.distance;
                for_each_cell(faces[c], [&](const position3& at) {
                    if (at[t] == 0) {
                        const std::size_t i = edge_index(faces[c], t, at);
                        made.known[i] = source.known[at[c]];
                        made.value[i] = source.value[at[c]];
                    }
                });
            }
            (side == 1 ? high : low)[c][q] = made.condition();
        }
        auto size = n;
        ++size[c];
        ++size[t];
        layout::layout<3>& layout = corners[c][q].emplace_back(size);
        wall[c][q].assign(layout.cell_count(), 0.5);
        if (m >= 0 && t != e) {
            for_each_cell(layout, [&](const position3& at) {
                wall[c][q][index_of(layout, at)] = from->wall[m][corner(at)];
            });
        }
    }

    grid::grid<3> make_grid() const {
        grid::grid_input<3> in{};
        for (int d = 0; d < 3; ++d) {
            in.faces[d] = axis_of(d).faces;
            in.centres[d] = axis_of(d).centres;
        }
        in.cell_type = types;
        in.solid = 1;
        return grid::grid<3>(in);
    }

    predictor_input<3> input() const {
        predictor_input<3> in{};
        for (int d = 0; d < 3; ++d) {
            in.cell_widths[d] = axis_of(d).widths;
            in.face_widths[d] = axis_of(d).between;
        }
        in.low = low;
        in.high = high;
        in.rho = 1.2;
        in.mu = 0.02;
        in.alpha = alpha;
        in.sweeps = 1;
        return in;
    }

    int a, b, e;
    const room2* from;
    axis unit;
    std::array<std::size_t, 3> n;
    layout::layout<3> cells;
    std::vector<std::int32_t> types;
    std::vector<double> pressure;
    std::vector<double> viscosity;
    std::deque<layout::layout<3>> faces;
    std::array<std::vector<double>, 3> velocity;
    std::deque<edge> edges;
    per_pair<edge_condition, 3> low, high;
    per_pair<std::deque<layout::layout<3>>, 3> corners;
    per_pair<std::vector<double>, 3> wall;
};

// One room, its 2D predictor and the extruded one, and the arguments of a call.
struct extruded {
    explicit extruded(int trial)
        : draw(static_cast<std::size_t>(trial) * 1000003),
          flat(trial, draw),
          solid(flat, orders[trial % 6][0], orders[trial % 6][1], orders[trial % 6][2]),
          grid2(grid_of(flat)),
          grid3(solid.make_grid()),
          predictor2(grid2, input_of(flat)),
          predictor3(grid3, solid.input()) {
        velocity2 = {flat.velocity[0], flat.velocity[1]};
        velocity3 = {solid.velocity[0], solid.velocity[1], solid.velocity[2]};
        if (trial % 2 == 1) {
            mu2 = flat.viscosity;
            mu3 = solid.viscosity;
        }
        if ((trial / 2) % 2 == 1) {
            wall2 = {{{flat.wall[0]}, {flat.wall[1]}}};
            for (int c = 0; c < 3; ++c) {
                for (int q = 0; q < 2; ++q) {
                    wall3[c][q] = solid.wall[c][q];
                }
            }
        }
    }
    extruded(const extruded&) = delete;

    static constexpr int orders[6][3] = {{0, 1, 2}, {1, 0, 2}, {0, 2, 1},
                                         {2, 0, 1}, {1, 2, 0}, {2, 1, 0}};
    draws draw;
    room2 flat;
    room3 solid;
    grid::grid<2> grid2;
    grid::grid<3> grid3;
    predictor<2> predictor2;
    predictor<3> predictor3;
    std::array<std::span<const double>, 2> velocity2;
    std::array<std::span<const double>, 3> velocity3;
    std::span<const double> mu2, mu3;
    per_pair<std::span<const double>, 2> wall2;
    per_pair<std::span<const double>, 3> wall3;
};

TEST(MomentumThreeDirections, AnExtrudedRoomHasThe2DRoomsEquations) {
    std::size_t compared = 0;
    for (int trial = 0; trial < rooms; ++trial) {
        SCOPED_TRACE(trial);
        extruded x(trial);
        for (int c = 0; c < 3; ++c) {
            SCOPED_TRACE(c);
            const auto& faces = x.solid.faces[c];
            std::vector<double> diagonal3(faces.cell_count(),
                                          std::numeric_limits<double>::quiet_NaN());
            x.predictor3.equations(c, x.velocity3, x.solid.pressure, x.mu3, x.wall3, diagonal3);
            const int m = x.solid.component(c);
            if (m < 0) {
                // The component along the extrusion: at rest, with no source.
                for_each_cell(faces, [&](const position3& at) {
                    const std::size_t i = index_of(faces, at);
                    if (diagonal3[i] > 0.0) {
                        EXPECT_NEAR(x.predictor3.source()[i], 0.0, 1e-15);
                    }
                });
                continue;
            }
            std::vector<double> diagonal2(x.flat.velocity[m].size());
            x.predictor2.equations(m, x.velocity2, x.flat.pressure, x.mu2, x.wall2, diagonal2);
            // The 3D slots of the 2D transverse direction and of the extrusion.
            const int q = slot_of(c, m == 0 ? x.solid.b : x.solid.a);
            const int e = slot_of(c, x.solid.e);
            const auto& p2 = x.predictor2;
            const auto& p3 = x.predictor3;
            for_each_cell(faces, [&](const position3& at) {
                const std::size_t i = index_of(faces, at);
                const std::size_t k = x.solid.plane(m, at);
                EXPECT_EQ(bits(p3.neighbour(0)[i]), bits(p2.neighbour(0)[k]));
                EXPECT_EQ(bits(p3.neighbour(1)[i]), bits(p2.neighbour(1)[k]));
                EXPECT_EQ(bits(p3.neighbour(2 + 2 * q)[i]), bits(p2.neighbour(2)[k]));
                EXPECT_EQ(bits(p3.neighbour(3 + 2 * q)[i]), bits(p2.neighbour(3)[k]));
                if (diagonal2[k] > 0.0) {
                    EXPECT_GT(diagonal3[i], 0.0);
                    EXPECT_EQ(bits(p3.source()[i]), bits(p2.source()[k]));
                }
                // The extruded direction's two coefficients are the rest.
                const double rest =
                    diagonal3[i] - p3.neighbour(2 + 2 * e)[i] - p3.neighbour(3 + 2 * e)[i];
                EXPECT_NEAR(rest, diagonal2[k], 1e-13 * std::max(1e-3, std::fabs(diagonal3[i])));
                ++compared;
            });
        }
    }
    EXPECT_GT(compared, 10000u);
}

TEST(MomentumThreeDirections, AnExtrudedRoomHasThe2DRoomsStencilViscosity) {
    std::size_t compared = 0;
    for (int trial = 1; trial < rooms; trial += 2) {
        SCOPED_TRACE(trial);
        extruded x(trial);
        std::array<std::vector<double>, 2> out2{x.flat.wall[0], x.flat.wall[1]};
        x.predictor2.stencil_viscosity(x.flat.viscosity, {{{out2[0]}, {out2[1]}}});
        per_pair<std::vector<double>, 3> out3 = x.solid.wall;
        per_pair<std::span<double>, 3> to{};
        for (int c = 0; c < 3; ++c) {
            for (int q = 0; q < 2; ++q) {
                to[c][q] = out3[c][q];
            }
        }
        x.predictor3.stencil_viscosity(x.solid.viscosity, to);
        for (int c = 0; c < 3; ++c) {
            const int m = x.solid.component(c);
            if (m < 0) {
                continue;
            }
            const int q = slot_of(c, m == 0 ? x.solid.b : x.solid.a);
            const auto& corners = x.solid.corners[c][q].front();
            for_each_cell(corners, [&](const position3& at) {
                EXPECT_EQ(bits(out3[c][q][index_of(corners, at)]),
                          bits(out2[m][x.solid.corner(at)]));
                ++compared;
            });
        }
    }
    EXPECT_GT(compared, 1000u);
}

TEST(MomentumThreeDirections, TheSweepsWriteEveryFaceAndEveryPlaneAlike) {
    for (int trial = 0; trial < rooms; ++trial) {
        SCOPED_TRACE(trial);
        extruded x(trial);
        std::array<std::vector<double>, 3> star, diagonal;
        for (int c = 0; c < 3; ++c) {
            star[c].assign(x.solid.velocity[c].size(), std::numeric_limits<double>::quiet_NaN());
            diagonal[c] = star[c];
        }
        x.predictor3.predict(x.velocity3, x.solid.pressure, x.mu3, x.wall3,
                             {star[0], star[1], star[2]},
                             {diagonal[0], diagonal[1], diagonal[2]});
        for (int c = 0; c < 3; ++c) {
            const auto& faces = x.solid.faces[c];
            for_each_cell(faces, [&](const position3& at) {
                auto mirrored = at;
                mirrored[x.solid.e] = faces.cells(x.solid.e) - 1 - at[x.solid.e];
                const double here = star[c][index_of(faces, at)];
                EXPECT_TRUE(std::isfinite(here));
                EXPECT_TRUE(std::isfinite(diagonal[c][index_of(faces, at)]));
                if (x.solid.component(c) >= 0) {
                    EXPECT_NEAR(here, star[c][index_of(faces, mirrored)], 1e-13);
                }
            });
        }
    }
}

}  // namespace
}  // namespace cfdcore::momentum

// The pressure correction at three directions: the Galerkin operator formed
// densely, the cycle's symmetry and a corrected field without imbalance.
namespace cfdcore::pressure {
namespace {

// A box of nx by ny by nz cells with SOLID ones, random widths, velocities and
// diagonals at the faces two open cells share, and with `open` one outlet.
struct box3 {
    box3(const std::array<std::size_t, 3>& n_, bool open_) : n(n_), open(open_), cells(n_) {
        momentum::draws draw(n[0] + 17 * n[1] + 289 * n[2] + (open ? 1 : 0));
        for (int d = 0; d < 3; ++d) {
            axes.push_back(momentum::make_axis(draw.field(n[d], 0.5)));
            const std::size_t edge = cells.cell_count() / n[d];
            for (int side = 0; side < 2; ++side) {
                outlets.emplace_back(edge);
            }
        }
        outlets[3].known[0] = open;  // the high edge along y
        types.assign(cells.cell_count(), 0);
        types[cells.cell_count() / 2 + 1] = 1;
        for (int c = 0; c < 3; ++c) {
            auto faces = n;
            ++faces[c];
            const layout::layout<3> at(faces);
            star[c].assign(at.cell_count(), 0.0);
            diagonal[c].assign(at.cell_count(), 0.0);
            const std::size_t stride = cells.cell_stride(c);
            for (std::size_t f = 0; f < at.cell_count(); ++f) {
                const std::size_t position = (f / stride) % faces[c];
                if (position == 0 || position == n[c]) {
                    continue;
                }
                const std::size_t block = f / (stride * faces[c]);
                const std::size_t high = block * stride * n[c] + position * stride + f % stride;
                if (types[high] == 0 && types[high - stride] == 0) {
                    star[c][f] = draw() - 0.5;
                    diagonal[c][f] = 0.5 + draw();
                }
            }
        }
        pressure = draw.field(cells.cell_count(), 0.0);
    }

    grid::grid<3> make_grid() const {
        grid::grid_input<3> in{};
        for (int d = 0; d < 3; ++d) {
            in.faces[d] = axes[static_cast<std::size_t>(d)].faces;
            in.centres[d] = axes[static_cast<std::size_t>(d)].centres;
        }
        in.cell_type = types;
        in.solid = 1;
        return grid::grid<3>(in);
    }

    corrector_input<3> input() const {
        corrector_input<3> in{};
        for (int d = 0; d < 3; ++d) {
            const std::size_t e = static_cast<std::size_t>(2 * d);
            in.cell_widths[d] = axes[static_cast<std::size_t>(d)].widths;
            in.outlet_low[d] = {outlets[e].known.get(), outlets[e].value.size()};
            in.outlet_high[d] = {outlets[e + 1].known.get(), outlets[e + 1].value.size()};
        }
        in.fluid = 0;
        in.rho = 1.2;
        in.alpha = 0.3;
        in.rtol = 1e-12;
        in.max_iter = 300;
        in.method = preconditioner::multigrid;
        in.levels = {.coarsest_cells = 8,
                     .sweeps = 1,
                     .weight = 2.0 / 3.0,
                     .dominance_tolerance = 1e-12,
                     .rcond = 1e-15};
        return in;
    }

    std::array<std::size_t, 3> n;
    bool open;
    layout::layout<3> cells;
    std::vector<momentum::axis> axes;
    std::deque<momentum::edge> outlets;
    std::vector<std::int32_t> types;
    std::array<std::vector<double>, 3> star, diagonal;
    std::vector<double> pressure;
};

constexpr std::array<std::array<std::size_t, 3>, 5> boxes{
    {{4, 4, 4}, {5, 3, 4}, {9, 5, 3}, {8, 7, 6}, {3, 9, 10}}};

template <typename Visit>
void for_each_coupling(const multigrid::system<3>& s, Visit&& visit) {
    const auto& fields = *s.fields;
    for (std::size_t i = 0; i < fields.cell_count(); ++i) {
        visit(i, i, s.centre[i]);
        for (int d = 0; d < 3; ++d) {
            const std::size_t stride = fields.cell_stride(d);
            const std::size_t at = (i / stride) % fields.cells(d);
            if (at + 1 < fields.cells(d)) {
                visit(i, i + stride, -s.high[d][i]);
            }
            if (at > 0) {
                visit(i, i - stride, -s.low[d][i]);
            }
        }
    }
}

TEST(PressureThreeDirections, TheGalerkinOperatorIsTheTripleProduct) {
    for (const auto& n : boxes) {
        const box3 b(n, true);
        const auto g = b.make_grid();
        corrector<3> c(g, b.input());
        c.assemble({b.diagonal[0], b.diagonal[1], b.diagonal[2]});
        const auto coarse = multigrid::coarser(b.cells);
        multigrid::system<3> to(coarse);
        multigrid::coarsen<3>(c.equation(), std::vector<double>(n[0], 0.0), to);
        const auto group = [&](std::size_t cell) {
            std::size_t index = 0;
            for (int d = 0; d < 3; ++d) {
                index +=
                    (cell / b.cells.cell_stride(d)) % n[d] / 2 * coarse.cell_stride(d);
            }
            return index;
        };
        const std::size_t m = coarse.cell_count();
        std::vector<double> expected(m * m, 0.0), got(m * m, 0.0);
        for_each_coupling(c.equation(), [&](std::size_t i, std::size_t j, double a) {
            expected[group(i) * m + group(j)] += a;
        });
        for_each_coupling(to, [&](std::size_t i, std::size_t j, double a) {
            got[i * m + j] += a;
        });
        for (std::size_t i = 0; i < m * m; ++i) {
            EXPECT_NEAR(got[i], expected[i], 1e-12);
        }
    }
}

TEST(PressureThreeDirections, TheCycleIsSymmetric) {
    for (const auto& n : boxes) {
        const box3 b(n, true);
        const auto g = b.make_grid();
        corrector<3> c(g, b.input());
        c.assemble({b.diagonal[0], b.diagonal[1], b.diagonal[2]});
        multigrid::hierarchy<3> h(b.cells, b.input().levels);
        ASSERT_EQ(h.build(c.equation()), multigrid::build_status::ok);
        momentum::draws draw(5);
        std::vector<double> r1 = draw.field(b.cells.cell_count(), -0.5);
        std::vector<double> r2 = draw.field(b.cells.cell_count(), -0.5);
        for (std::size_t i = 0; i < r1.size(); ++i) {
            if (!(c.equation().centre[i] > 0.0)) {
                r1[i] = r2[i] = 0.0;
            }
        }
        std::vector<double> z1(r1.size()), z2(r1.size());
        h.cycle(r1, z1);
        h.cycle(r2, z2);
        double a = 0.0, across = 0.0, scale = 0.0;
        for (std::size_t i = 0; i < r1.size(); ++i) {
            a += z1[i] * r2[i];
            across += r1[i] * z2[i];
            scale += std::fabs(z1[i] * r2[i]);
        }
        EXPECT_NEAR(a, across, 1e-12 * scale);
    }
}

TEST(PressureThreeDirections, ACorrectionLeavesNoImbalance) {
    for (const bool open : {true, false}) {
        for (const auto& n : boxes) {
            const box3 b(n, open);
            const auto g = b.make_grid();
            const auto count = corrector<3>::count_components(g, b.input());
            ASSERT_EQ(count.components, 1);
            EXPECT_EQ(count.stranded, open ? 0 : 1);
            corrector<3> c(g, b.input());
            std::array<std::vector<double>, 3> out = b.star;
            std::vector<double> next(b.pressure.size()), p_prime(b.pressure.size());
            const solve_result result = c.correct(
                {b.star[0], b.star[1], b.star[2]}, {b.diagonal[0], b.diagonal[1], b.diagonal[2]},
                b.pressure, {out[0], out[1], out[2]}, next, p_prime);
            ASSERT_EQ(result.status, solve_status::ok);
            EXPECT_FALSE(result.reached_cap);
            std::vector<double> imbalance(b.pressure.size());
            c.mass_imbalance({out[0], out[1], out[2]}, imbalance);
            for (const double value : imbalance) {
                EXPECT_NEAR(value, 0.0, 1e-10);
            }
        }
    }
}

}  // namespace
}  // namespace cfdcore::pressure
