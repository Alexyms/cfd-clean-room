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
#include "../src/scalar.cpp"
#include "../src/turbulence.cpp"
#include "../src/walls.cpp"

#include "cfdcore/positions.hpp"

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
#include <set>
#include <span>
#include <vector>

template class cfdcore::layout::layout<3>;
template class cfdcore::grid::grid<3>;
template class cfdcore::momentum::predictor<3>;
template struct cfdcore::multigrid::system<3>;
template class cfdcore::multigrid::hierarchy<3>;
template class cfdcore::pressure::corrector<3>;
template class cfdcore::turbulence::model<3>;
template class cfdcore::walls::boundary<3>;

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

// A face's area is the product of the widths along its two other directions.
TEST(PressureThreeDirections, AFacesAreaIsTheProductOfItsWidths) {
    const box3 b({5, 4, 3}, false);
    const auto g = b.make_grid();
    corrector<3> c(g, b.input());
    c.assemble({b.diagonal[0], b.diagonal[1], b.diagonal[2]});
    for (int d = 0; d < 3; ++d) {
        auto counts = b.n;
        ++counts[d];
        const layout::layout<3> faces(counts);
        std::size_t compared = 0;
        for (std::size_t f = 0; f < faces.cell_count(); ++f) {
            if (!(b.diagonal[d][f] > 0.0)) {
                EXPECT_EQ(c.face_d(d)[f], 0.0);
                continue;
            }
            double area = 1.0;
            for (int t = 0; t < 3; ++t) {
                if (t != d) {
                    const std::size_t at = (f / faces.cell_stride(t)) % faces.cells(t);
                    area *= b.axes[static_cast<std::size_t>(t)].widths[at];
                }
            }
            EXPECT_NEAR(c.face_d(d)[f], area / b.diagonal[d][f], 1e-14);
            ++compared;
        }
        EXPECT_GT(compared, 10u);
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

// The turbulence model and the wall functions at three directions: a 2D room
// extruded along each direction in turn, its own two axes on the other two in
// their order (a sum over the directions is taken in their order, so the room's
// own transpose is another room), must have the 2D room's conditions, terms and
// explicit fields bit for bit on every plane.
namespace cfdcore::turbulence {
namespace {

using momentum::bits;
using momentum::draws;
using momentum::edge_index;
using momentum::for_each_cell;
using momentum::index_of;
using momentum::position3;
using momentum::room2;
using momentum::room3;
using momentum::slot_of;
using momentum::transverse;

constexpr int turbulence_rooms = 48;

// A bool array a span can view.
struct flags {
    explicit flags(std::size_t n) : size(n), data(new bool[n]) {
        std::fill(data.get(), data.get() + n, false);
    }
    std::span<const bool> span() const { return {data.get(), size}; }
    std::size_t size;
    std::unique_ptr<bool[]> data;
};

constants variant_of(bool has_r) {
    return has_r ? constants{0.0845, 1.42, 1.68, 0.7194, 0.7194, true, 4.38, 0.012}
                 : constants{0.09, 1.44, 1.92, 1.0, 1.3, false, 0.0, 0.0};
}

// The 2D room's and the extruded room's models and boundaries, and the arguments
// of a step on each.
struct pair_of_rooms {
    explicit pair_of_rooms(int trial_)
        : trial(trial_),
          draw(static_cast<std::size_t>(trial_) * 7919 + 11),
          flat(trial_, draw),
          solid(flat, orders[trial_ % 3][0], orders[trial_ % 3][1], orders[trial_ % 3][2]),
          grid2(momentum::grid_of(flat)),
          grid3(solid.make_grid()) {
        const std::size_t cells2 = flat.types.size();
        k = draw.field(cells2, 0.1);
        eps = draw.field(cells2, 0.1);
        nu_t = draw.field(cells2, 0.0);
        for (std::size_t i = 0; i < cells2; ++i) {
            if (flat.types[i] == 1) {
                k[i] = eps[i] = nu_t[i] = 0.0;
            }
        }
        k3 = planes(k);
        eps3 = planes(eps);
        nu_t3 = planes(nu_t);
        build_vents();
        build_model();
        build_boundary();
    }
    pair_of_rooms(const pair_of_rooms&) = delete;

    // A 2D cell field on every plane.
    std::vector<double> planes(const std::vector<double>& field) const {
        std::vector<double> out(solid.cells.cell_count());
        for_each_cell(solid.cells, [&](const position3& at) {
            out[index_of(solid.cells, at)] = field[at[solid.b] * flat.n[0] + at[solid.a]];
        });
        return out;
    }

    // A share of the faces between a SOLID and an open cell are vents.
    void build_vents() {
        for (int c = 0; c < 2; ++c) {
            vents2.emplace_back(flat.velocity[c].size());
            const std::size_t row = flat.n[0] + (c == 0 ? 1 : 0);
            for (std::size_t f = 0; f < vents2.back().size; ++f) {
                const std::size_t i = f % row;
                const std::size_t j = f / row;
                const std::size_t at = c == 0 ? i : j;
                if (at == 0 || at == flat.n[c]) {
                    continue;
                }
                const std::size_t high = j * flat.n[0] + i;
                const std::size_t low = high - (c == 0 ? 1 : flat.n[0]);
                vents2.back().data[f] =
                    flat.types[low] != flat.types[high] && draw() < 0.4;
            }
        }
        for (int c = 0; c < 3; ++c) {
            vents3.emplace_back(solid.velocity[c].size());
            const int m = solid.component(c);
            if (m >= 0) {
                for_each_cell(solid.faces[c], [&](const position3& at) {
                    vents3.back().data[index_of(solid.faces[c], at)] =
                        vents2[m].data[solid.plane(m, at)];
                });
            }
        }
    }

    void build_model() {
        model_input<2> in2{};
        model_input<3> in3{};
        in2.cell_widths = {flat.axes[0].widths, flat.axes[1].widths};
        in2.face_widths = {flat.axes[0].between, flat.axes[1].between};
        in2.vents = {vents2[0].span(), vents2[1].span()};
        for (int d = 0; d < 3; ++d) {
            in3.cell_widths[d] = solid.axis_of(d).widths;
            in3.face_widths[d] = solid.axis_of(d).between;
            in3.vents[d] = vents3[d].span();
        }
        in2.variant = in3.variant = variant_of(trial % 2 == 1);
        in2.cfl = in3.cfl = 0.25;
        in2.upwind = in3.upwind = (trial / 2) % 2 == 0;
        in2.max_iter = in3.max_iter = 500;
        in2.tol = in3.tol = 1e-13;
        in2.nu = in3.nu = 1e-3;
        model2 = std::make_unique<model<2>>(grid2, in2);
        model3 = std::make_unique<model<3>>(grid3, in3);
    }

    void build_boundary() {
        walls::boundary_input<2> in2{};
        walls::boundary_input<3> in3{};
        // The 2D edges' wall cells per direction and side, and the wall corners
        // and the conditions of the component along the edge.
        for (int d = 0; d < 2; ++d) {
            const int c = 1 - d;
            for (int side = 0; side < 2; ++side) {
                auto& cells = flat_cells.emplace_back(flat.n[c]);
                auto& corners = flat_corners.emplace_back(flat.n[c] + 1);
                for (std::size_t i = 0; i < cells.size; ++i) {
                    cells.data[i] = draw() < 0.6;
                }
                for (std::size_t i = 0; i < corners.size; ++i) {
                    corners.data[i] = draw() < 0.6;
                }
            }
        }
        for (int d = 0; d < 2; ++d) {
            const int c = 1 - d;
            in2.wall_cells_low[d] = flat_cells[2 * d].span();
            in2.wall_cells_high[d] = flat_cells[2 * d + 1].span();
            in2.wall_corners_low[c][0] = flat_corners[2 * d].span();
            in2.wall_corners_high[c][0] = flat_corners[2 * d + 1].span();
            in2.low[c][0] = flat.edge_of(c, 0).condition();
            in2.high[c][0] = flat.edge_of(c, 1).condition();
        }
        for (int d = 0; d < 3; ++d) {
            const int m = solid.component(d);
            for (int side = 0; side < 2; ++side) {
                auto& cells = deep_cells.emplace_back(solid.cells.cell_count() / solid.n[d]);
                if (m >= 0) {
                    const int other = m == 0 ? solid.b : solid.a;
                    for_each_cell(solid.cells, [&](const position3& at) {
                        if (at[d] == 0) {
                            cells.data[edge_index(solid.cells, d, at)] =
                                flat_cells[2 * m + side].data[at[other]];
                        }
                    });
                }
                (side == 0 ? in3.wall_cells_low : in3.wall_cells_high)[d] = cells.span();
            }
        }
        for (int c = 0; c < 3; ++c) {
            const int m = solid.component(c);
            for (int q = 0; q < 2; ++q) {
                const int t = transverse(c, q);
                for (int side = 0; side < 2; ++side) {
                    auto& corners =
                        deep_corners.emplace_back(solid.faces[c].cell_count() / solid.n[t]);
                    if (m >= 0 && t != solid.e) {
                        for_each_cell(solid.faces[c], [&](const position3& at) {
                            if (at[t] == 0) {
                                corners.data[edge_index(solid.faces[c], t, at)] =
                                    flat_corners[2 * (1 - m) + side].data[at[c]];
                            }
                        });
                    }
                    (side == 0 ? in3.wall_corners_low : in3.wall_corners_high)[c][q] =
                        corners.span();
                }
                in3.low[c][q] = solid.low[c][q];
                in3.high[c][q] = solid.high[c][q];
                auto& over = deep_vents.emplace_back(solid.corners[c][q].front().cell_count());
                if (m >= 0 && t != solid.e) {
                    over_vents(m, t, solid.corners[c][q].front(), over);
                }
                in3.vent_corners[c][q] = over.span();
            }
        }
        for (int m = 0; m < 2; ++m) {
            in2.vent_corners[m][0] = flat_vents[m].span();
            in2.vents[m] = vents2[m].span();
        }
        for (int d = 0; d < 3; ++d) {
            in3.vents[d] = vents3[d].span();
        }
        in2.c_mu = in3.c_mu = 0.09;
        in2.rho = in3.rho = 1.2;
        in2.nu = in3.nu = 1e-3;
        in2.law = in3.law = {0.41, 9.793, walls::log_law_floor(0.41, 9.793).y};
        walls2 = std::make_unique<walls::boundary<2>>(grid2, in2);
        walls3 = std::make_unique<walls::boundary<3>>(grid3, in3);
    }

    // The stencil's corners over a vent's span: some of the 2D room's corners, on
    // every plane.
    void over_vents(int m, int, const layout::layout<3>& corners, flags& out) {
        if (flat_vents.empty()) {
            for (int c = 0; c < 2; ++c) {
                flat_vents.emplace_back((flat.n[0] + 1) * (flat.n[1] + 1));
                for (std::size_t i = 0; i < flat_vents.back().size; ++i) {
                    flat_vents.back().data[i] = draw() < 0.1;
                }
            }
        }
        for_each_cell(corners, [&](const position3& at) {
            out.data[index_of(corners, at)] = flat_vents[m].data[solid.corner(at)];
        });
    }

    std::array<std::span<const double>, 2> u2() const {
        return {flat.velocity[0], flat.velocity[1]};
    }
    std::array<std::span<const double>, 3> u3() const {
        return {solid.velocity[0], solid.velocity[1], solid.velocity[2]};
    }
    state s2() const { return {k, eps, nu_t}; }
    state s3() const { return {k3, eps3, nu_t3}; }

    // Whether every plane of a 3D cell field is the 2D one, bit for bit.
    void expect_planes(std::span<const double> deep, std::span<const double> plane) const {
        for_each_cell(solid.cells, [&](const position3& at) {
            EXPECT_EQ(bits(deep[index_of(solid.cells, at)]),
                      bits(plane[at[solid.b] * flat.n[0] + at[solid.a]]));
        });
    }
    // And of the face fields normal to the 2D room's component m.
    void expect_faces(int m, std::span<const double> deep, std::span<const double> plane) const {
        const int c = m == 0 ? solid.a : solid.b;
        for_each_cell(solid.faces[c], [&](const position3& at) {
            EXPECT_EQ(bits(deep[index_of(solid.faces[c], at)]), bits(plane[solid.plane(m, at)]));
        });
    }

    static constexpr int orders[3][3] = {{0, 1, 2}, {0, 2, 1}, {1, 2, 0}};
    int trial;
    draws draw;
    room2 flat;
    room3 solid;
    grid::grid<2> grid2;
    grid::grid<3> grid3;
    std::vector<double> k, eps, nu_t, k3, eps3, nu_t3;
    std::deque<flags> vents2, vents3, flat_cells, flat_corners, deep_cells, deep_corners;
    std::deque<flags> flat_vents, deep_vents;
    std::unique_ptr<model<2>> model2;
    std::unique_ptr<model<3>> model3;
    std::unique_ptr<walls::boundary<2>> walls2;
    std::unique_ptr<walls::boundary<3>> walls3;
};

TEST(TurbulenceThreeDirections, AnExtrudedRoomHasThe2DRoomsConditions) {
    std::size_t walls_found = 0;
    for (int trial = 0; trial < turbulence_rooms; ++trial) {
        SCOPED_TRACE(trial);
        pair_of_rooms x(trial);
        const auto c2 = x.walls2->conditions(x.u2(), x.k, x.eps);
        const auto c3 = x.walls3->conditions(x.u3(), x.k3, x.eps3);
        x.expect_planes(c3.eps_wall, c2.eps_wall);
        x.expect_planes(c3.production, c2.production);
        for (std::size_t i = 0; i < x.k3.size(); ++i) {
            walls_found += c3.eps_held[i];
        }
        for (int m = 0; m < 2; ++m) {
            const int c = m == 0 ? x.solid.a : x.solid.b;
            const int t = m == 0 ? x.solid.b : x.solid.a;
            x.expect_faces(m, c3.inflow_k[c], c2.inflow_k[m]);
            x.expect_faces(m, c3.inflow_eps[c], c2.inflow_eps[m]);
            const int q = slot_of(c, t);
            for_each_cell(x.solid.faces[c], [&](const position3& at) {
                if (at[t] == 0) {
                    const std::size_t i = edge_index(x.solid.faces[c], t, at);
                    EXPECT_EQ(bits(c3.low[c][q][i]), bits(c2.low[m][0][at[c]]));
                    EXPECT_EQ(bits(c3.high[c][q][i]), bits(c2.high[m][0][at[c]]));
                }
            });
        }
    }
    EXPECT_GT(walls_found, 1000u);
}

TEST(TurbulenceThreeDirections, AnExtrudedRoomHasThe2DRoomsWallViscosity) {
    std::size_t compared = 0;
    for (int trial = 0; trial < turbulence_rooms; ++trial) {
        SCOPED_TRACE(trial);
        pair_of_rooms x(trial);
        std::array<std::vector<double>, 2> out2{x.flat.wall[0], x.flat.wall[1]};
        x.walls2->wall_viscosity(x.k, {{{x.flat.wall[0]}, {x.flat.wall[1]}}},
                                 {{{out2[0]}, {out2[1]}}});
        momentum::per_pair<std::vector<double>, 3> out3 = x.solid.wall;
        momentum::per_pair<std::span<const double>, 3> from{};
        momentum::per_pair<std::span<double>, 3> to{};
        for (int c = 0; c < 3; ++c) {
            for (int q = 0; q < 2; ++q) {
                from[c][q] = x.solid.wall[c][q];
                to[c][q] = out3[c][q];
            }
        }
        x.walls3->wall_viscosity(x.k3, from, to);
        for (int m = 0; m < 2; ++m) {
            const int c = m == 0 ? x.solid.a : x.solid.b;
            const int q = slot_of(c, m == 0 ? x.solid.b : x.solid.a);
            const auto& corners = x.solid.corners[c][q].front();
            compared += x.walls2->stencil(m, 0).corner.size();
            for_each_cell(corners, [&](const position3& at) {
                EXPECT_EQ(bits(out3[c][q][index_of(corners, at)]),
                          bits(out2[m][x.solid.corner(at)]));
            });
        }
    }
    EXPECT_GT(compared, 500u);
}

TEST(TurbulenceThreeDirections, AnExtrudedRoomHasThe2DRoomsBlocks) {
    for (int trial = 0; trial < turbulence_rooms; ++trial) {
        SCOPED_TRACE(trial);
        pair_of_rooms x(trial);
        const auto c2 = x.walls2->conditions(x.u2(), x.k, x.eps);
        const auto c3 = x.walls3->conditions(x.u3(), x.k3, x.eps3);
        const std::size_t n2 = x.k.size();
        const std::size_t n3 = x.k3.size();
        std::vector<double> a2(n2), a3(n3), b2(n2), b3(n3);
        x.model2->strain_squared(x.u2(), c2, a2);
        x.model3->strain_squared(x.u3(), c3, a3);
        x.expect_planes(a3, a2);
        ASSERT_TRUE(x.model2->pseudo_time_step(x.u2()));
        ASSERT_TRUE(x.model3->pseudo_time_step(x.u3()));
        x.expect_planes(x.model3->time_step(), x.model2->time_step());
        x.model2->terms(x.s2(), x.u2(), c2);
        x.model3->terms(x.s3(), x.u3(), c3);
        x.expect_planes(x.model3->production(), x.model2->production());
        x.expect_planes(x.model3->rng_r(), x.model2->rng_r());
        x.expect_planes(x.model3->growth_eps(), x.model2->growth_eps());
        x.expect_planes(x.model3->decay_k(), x.model2->decay_k());
        x.expect_planes(x.model3->decay_eps(), x.model2->decay_eps());
        x.model2->explicit_fields(x.s2(), c2, a2, b2);
        x.model3->explicit_fields(x.s3(), c3, a3, b3);
        x.expect_planes(a3, a2);
        x.expect_planes(b3, b2);
        std::array<std::vector<double>, 2> g2{x.flat.velocity[0], x.flat.velocity[1]};
        std::array<std::vector<double>, 3> g3{x.solid.velocity[0], x.solid.velocity[1],
                                              x.solid.velocity[2]};
        for (const auto which : {quantity::k, quantity::eps}) {
            x.model2->conductances(which, x.nu_t, {g2[0], g2[1]});
            x.model3->conductances(which, x.nu_t3, {g3[0], g3[1], g3[2]});
            for (int m = 0; m < 2; ++m) {
                x.expect_faces(m, g3[m == 0 ? x.solid.a : x.solid.b], g2[m]);
            }
        }
    }
}

TEST(TurbulenceThreeDirections, AnExtrudedRoomsStepIsThe2DRoomsOnEveryPlane) {
    std::size_t steps = 0;
    for (int trial = 0; trial < turbulence_rooms; ++trial) {
        SCOPED_TRACE(trial);
        pair_of_rooms x(trial);
        const auto c2 = x.walls2->conditions(x.u2(), x.k, x.eps);
        const auto c3 = x.walls3->conditions(x.u3(), x.k3, x.eps3);
        const std::size_t n2 = x.k.size();
        const std::size_t n3 = x.k3.size();
        std::array<std::vector<double>, 3> out2, out3;
        out2.fill(std::vector<double>(n2));
        out3.fill(std::vector<double>(n3));
        const auto r2 = x.model2->step(x.s2(), x.u2(), c2, out2[0], out2[1], out2[2]);
        const auto r3 = x.model3->step(x.s3(), x.u3(), c3, out3[0], out3[1], out3[2]);
        ASSERT_EQ(r2.status, r3.status);
        if (r2.status != step_status::ok) {
            continue;
        }
        ++steps;
        // The extruded direction's diffusion crosses faces between equal values,
        // so the solution is the 2D one; the solves stop where they stop.
        for (std::size_t f = 0; f < 3; ++f) {
            for_each_cell(x.solid.cells, [&](const position3& at) {
                const double plane = out2[f][at[x.solid.b] * x.flat.n[0] + at[x.solid.a]];
                EXPECT_NEAR(out3[f][index_of(x.solid.cells, at)], plane,
                            1e-8 * std::fabs(plane));
            });
        }
    }
    EXPECT_GT(steps, 30u);
}

// A box of unequal widths along its three directions, every edge open.
struct box3 {
    box3() : grid(make()) {
        input.cell_widths = {axes[0].widths, axes[1].widths, axes[2].widths};
        input.face_widths = {axes[0].between, axes[1].between, axes[2].between};
        input.variant = variant_of(false);
        input.cfl = 0.25;
        input.max_iter = 100;
        input.tol = 1e-12;
        input.nu = 0.5;
        for (int c = 0; c < 3; ++c) {
            faces.push_back(positions::faces_layout(grid.fields(), c, c));
            velocity[c].assign(faces[c].cell_count(), 0.0);
        }
    }
    grid::grid<3> make() {
        types.assign(3 * 2 * 4, 0);
        return grid::grid<3>(grid::grid_input<3>{
            .faces = {axes[0].faces, axes[1].faces, axes[2].faces},
            .centres = {axes[0].centres, axes[1].centres, axes[2].centres},
            .cell_type = types,
            .solid = 1});
    }
    std::array<momentum::axis, 3> axes{momentum::make_axis({0.5, 0.25, 0.75}),
                                       momentum::make_axis({0.2, 0.4}),
                                       momentum::make_axis({1.0, 2.0, 0.5, 1.5})};
    std::vector<std::int32_t> types;
    grid::grid<3> grid;
    model_input<3> input{};
    std::deque<layout::layout<3>> faces;
    std::array<std::vector<double>, 3> velocity;
};

TEST(TurbulenceThreeDirections, ALinearFlowHasItsStrain) {
    const double a[3][3] = {{1.0, 2.0, -3.0}, {0.5, -4.0, 1.5}, {2.5, -1.0, 3.0}};
    box3 b;
    model<3> m(b.grid, b.input);
    // Component c at a point, and the edges' own velocities the same flow's.
    const auto flow = [&](int c, const std::array<double, 3>& p) {
        return a[c][0] * p[0] + a[c][1] * p[1] + a[c][2] * p[2];
    };
    conditions<3> c;
    momentum::per_pair<std::vector<double>, 3> low, high;
    for (int d = 0; d < 3; ++d) {
        const auto place = [&](const position3& at) {
            std::array<double, 3> p{};
            for (int e = 0; e < 3; ++e) {
                p[e] = e == d ? b.axes[e].faces[at[e]] : b.axes[e].centres[at[e]];
            }
            return p;
        };
        for_each_cell(b.faces[d], [&](const position3& at) {
            b.velocity[d][index_of(b.faces[d], at)] = flow(d, place(at));
        });
        for (int q = 0; q < 2; ++q) {
            const int t = transverse(d, q);
            low[d][q].assign(b.faces[d].cell_count() / b.faces[d].cells(t), 0.0);
            high[d][q] = low[d][q];
            for_each_cell(b.faces[d], [&](const position3& at) {
                if (at[t] == 0) {
                    auto p = place(at);
                    const std::size_t i = edge_index(b.faces[d], t, at);
                    p[t] = b.axes[t].faces.front();
                    low[d][q][i] = flow(d, p);
                    p[t] = b.axes[t].faces.back();
                    high[d][q][i] = flow(d, p);
                }
            });
            c.low[d][q] = low[d][q];
            c.high[d][q] = high[d][q];
        }
        c.inflow_k[d] = b.velocity[d];
        c.inflow_eps[d] = b.velocity[d];
    }
    const std::vector<double> zero(24, 0.0);
    const flags none(24);
    c.eps_wall = c.production = zero;
    c.eps_held = c.production_given = none.span();
    std::vector<double> s2(24);
    m.strain_squared({b.velocity[0], b.velocity[1], b.velocity[2]}, c, s2);
    double expected = 0.0;
    for (int d = 0; d < 3; ++d) {
        expected += 2.0 * a[d][d] * a[d][d];
        for (int e = d + 1; e < 3; ++e) {
            expected += (a[d][e] + a[e][d]) * (a[d][e] + a[e][d]);
        }
    }
    for (const double value : s2) {
        EXPECT_NEAR(value, expected, 1e-11);
    }
}

TEST(TurbulenceThreeDirections, AFacesAreaAndACellsVolumeAreTheProductsOfTheirWidths) {
    box3 b;
    model<3> m(b.grid, b.input);
    const std::vector<double> nu_t(24, 0.0);
    std::array<std::vector<double>, 3> g = b.velocity;
    m.conductances(quantity::k, nu_t, {g[0], g[1], g[2]});
    for (int d = 0; d < 3; ++d) {
        for_each_cell(b.faces[d], [&](const position3& at) {
            double expected = 0.5 / b.axes[d].between[at[d]];
            for (int e = 0; e < 3; ++e) {
                expected *= e == d ? 1.0 : b.axes[e].widths[at[e]];
            }
            const bool edge = at[d] == 0 || at[d] + 1 == b.faces[d].cells(d);
            EXPECT_NEAR(g[d][index_of(b.faces[d], at)], edge ? 0.0 : expected, 1e-14);
        });
    }
    for_each_cell(b.grid.fields(), [&](const position3& at) {
        EXPECT_NEAR(m.volume()[index_of(b.grid.fields(), at)],
                    b.axes[0].widths[at[0]] * b.axes[1].widths[at[1]] * b.axes[2].widths[at[2]],
                    1e-15);
    });
}

TEST(TurbulenceThreeDirections, TheSlipIsTheMagnitudeOfBothComponentsAlongTheWall) {
    box3 b;
    walls::boundary_input<3> in{};
    std::deque<flags> made;
    std::deque<std::vector<double>> values;
    for (int d = 0; d < 3; ++d) {
        const std::size_t edge = b.grid.fields().cell_count() / b.grid.cells(d);
        // The low edge along y is the one wall.
        auto& low = made.emplace_back(edge);
        std::fill_n(low.data.get(), edge, d == 1);
        in.wall_cells_low[d] = low.span();
        in.wall_cells_high[d] = made.emplace_back(edge).span();
        for (int q = 0; q < 2; ++q) {
            const std::size_t n =
                b.faces[d].cell_count() / b.faces[d].cells(transverse(d, q));
            in.wall_corners_low[d][q] = in.wall_corners_high[d][q] = made.emplace_back(n).span();
            in.low[d][q] = {made.emplace_back(n).span(), values.emplace_back(n, 0.0), 0.5};
            in.high[d][q] = in.low[d][q];
        }
    }
    in.c_mu = 0.09;
    in.rho = 1.2;
    in.nu = 1.5e-5;
    in.law = {0.41, 9.793, walls::log_law_floor(0.41, 9.793).y};
    walls::boundary<3> w(b.grid, in);
    b.velocity[0].assign(b.velocity[0].size(), 3.0);
    b.velocity[1].assign(b.velocity[1].size(), 100.0);
    b.velocity[2].assign(b.velocity[2].size(), -4.0);
    const std::vector<double> k(24, 0.04), eps(24, 0.5);
    const auto c = w.conditions({b.velocity[0], b.velocity[1], b.velocity[2]}, k, eps);
    const double u_k = std::pow(0.09, 0.25) * 0.2;
    const double y = 0.1;
    const double log_term = std::log(9.793 * std::max(u_k * y / 1.5e-5, in.law.floor));
    for_each_cell(b.grid.fields(), [&](const position3& at) {
        const std::size_t i = index_of(b.grid.fields(), at);
        EXPECT_EQ(c.eps_held[i], at[1] == 0);
        EXPECT_NEAR(c.production[i], at[1] == 0 ? u_k * u_k * 5.0 / (y * log_term) : 0.0, 1e-14);
    });
}

}  // namespace
}  // namespace cfdcore::turbulence

// The position helpers at three directions: the two-direction program
// (test_positions.cpp) has the rest of their tests.
namespace cfdcore::positions {
namespace {

TEST(PositionsThreeDirections, AnIndexAndItsPositionAreEachOthersInverse) {
    const layout::layout<3> cells({4, 3, 2});
    std::set<std::size_t> seen;
    for (std::size_t i = 0; i < cells.cell_count(); ++i) {
        const position<3> at = position_of(cells, i);
        EXPECT_LT(at[0], 4u);
        EXPECT_LT(at[1], 3u);
        EXPECT_LT(at[2], 2u);
        EXPECT_EQ(index_of(cells, at), i);
        seen.insert(index_of(cells, at));
    }
    EXPECT_EQ(seen.size(), cells.cell_count());
}

TEST(PositionsThreeDirections, AnEdgesEntryIsIndexedWithoutItsNormalDirection) {
    const layout::layout<3> cells({4, 3, 2});
    // The others in their order, the lower direction fastest.
    EXPECT_EQ(index_without(cells, position<3>{{3, 2, 1}}, 0), 2u + 1u * 3u);
    EXPECT_EQ(index_without(cells, position<3>{{3, 2, 1}}, 1), 3u + 1u * 4u);
    EXPECT_EQ(index_without(cells, position<3>{{3, 2, 1}}, 2), 3u + 2u * 4u);
    // Every entry of an edge is reached once.
    std::set<std::size_t> seen;
    for (std::size_t i = 0; i < cells.cell_count(); ++i) {
        const position<3> at = position_of(cells, i);
        if (at[1] == 0) {
            seen.insert(index_without(cells, at, 1));
        }
    }
    EXPECT_EQ(seen.size(), 4u * 2u);
    EXPECT_EQ(*seen.rbegin(), 4u * 2u - 1u);
}

TEST(PositionsThreeDirections, AFaceLayoutHasOneMoreAlongItsDirectionsOnly) {
    const layout::layout<3> room({4, 3, 2});
    const layout::layout<3> w_faces = faces_layout(room, 2, 2);
    EXPECT_EQ(w_faces.cells(0), 4u);
    EXPECT_EQ(w_faces.cells(1), 3u);
    EXPECT_EQ(w_faces.cells(2), 3u);
    const layout::layout<3> corners = faces_layout(room, 0, 2);
    EXPECT_EQ(corners.cells(0), 5u);
    EXPECT_EQ(corners.cells(1), 3u);
    EXPECT_EQ(corners.cells(2), 3u);
}

}  // namespace
}  // namespace cfdcore::positions
