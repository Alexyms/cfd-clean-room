// The scalar module's own tests (ADR-017 D): results known without the
// reference. The comparison with scalar_scheme.py is tests/test_core_parity.py's.

#include "cfdcore/grid.hpp"
#include "cfdcore/layout.hpp"
#include "cfdcore/scalar.hpp"

#include <gtest/gtest.h>

#include <algorithm>
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

using cfdcore::grid::grid;
using cfdcore::grid::grid_input;
using cfdcore::scalar::advective_flux;
using cfdcore::scalar::limited_face_value;
using cfdcore::scalar::np_maximum;
using cfdcore::scalar::np_minimum;
using cfdcore::scalar::scheme;

constexpr std::int32_t fluid = 0;
constexpr std::int32_t solid = 1;

std::uint64_t bits(double x) { return std::bit_cast<std::uint64_t>(x); }

// Faces of a stretched axis: widths growing by `ratio` from `first`.
std::vector<double> stretched_faces(std::size_t n, double first, double ratio) {
    std::vector<double> faces(n + 1, 0.0);
    double width = first;
    for (std::size_t i = 0; i < n; ++i) {
        faces[i + 1] = faces[i] + width;
        width *= ratio;
    }
    return faces;
}

std::vector<double> centres_of(const std::vector<double>& faces) {
    std::vector<double> centres(faces.size() - 1);
    for (std::size_t i = 0; i + 1 < faces.size(); ++i) {
        centres[i] = 0.5 * (faces[i] + faces[i + 1]);
    }
    return centres;
}

// A 2D grid's arrays, held so the grid's spans stay valid while it is built.
struct grid2_arrays {
    std::vector<double> x;
    std::vector<double> xc;
    std::vector<double> y;
    std::vector<double> yc;
    std::vector<std::int32_t> types;

    grid2_arrays(std::size_t nx, std::size_t ny, double ratio_x, double ratio_y)
        : x(stretched_faces(nx, 0.1, ratio_x)),
          xc(centres_of(x)),
          y(stretched_faces(ny, 0.07, ratio_y)),
          yc(centres_of(y)),
          types(nx * ny, fluid) {}

    grid_input<2> input() const {
        return {.faces = {x, y}, .centres = {xc, yc}, .cell_type = types, .solid = solid};
    }
};

// ---- The element functions -------------------------------------------------

TEST(NumpyMinMax, TiesReturnTheSecondArgument) {
    EXPECT_EQ(bits(np_minimum(0.0, -0.0)), bits(-0.0));
    EXPECT_EQ(bits(np_minimum(-0.0, 0.0)), bits(0.0));
    EXPECT_EQ(bits(np_maximum(0.0, -0.0)), bits(-0.0));
    EXPECT_EQ(bits(np_maximum(-0.0, 0.0)), bits(0.0));
    EXPECT_EQ(np_minimum(1.0, 2.0), 1.0);
    EXPECT_EQ(np_maximum(1.0, 2.0), 2.0);
}

TEST(NumpyMinMax, ANanInEitherIsANan) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isnan(np_minimum(nan, 1.0)));
    EXPECT_TRUE(std::isnan(np_minimum(1.0, nan)));
    EXPECT_TRUE(std::isnan(np_maximum(nan, 1.0)));
    EXPECT_TRUE(std::isnan(np_maximum(1.0, nan)));
}

TEST(LimitedFaceValue, NoDownstreamDifferenceIsTheUpstreamValue) {
    EXPECT_EQ(bits(limited_face_value(5.0, 2.0, 2.0, 9.0)), bits(2.0));
    EXPECT_EQ(bits(limited_face_value(5.0, -0.0, 0.0, 9.0)), bits(-0.0));
}

TEST(LimitedFaceValue, HandValuesOfEachBranch) {
    // c_c = 1, c_d = 2 (d = 1): psi is the least of 2r, (1 + 3r) / 4,
    // psi_quick = 2 (quick - 1) and 2, at least 0; the value is 1 + psi / 2.
    // r = 0.25 (c_up = 0.75), quick = 1.5: 2r = 0.5, (1 + 0.75) / 4 = 0.4375, 1.
    EXPECT_EQ(limited_face_value(0.75, 1.0, 2.0, 1.5), 1.0 + 0.5 * 0.4375);
    // r = 0.125 (c_up = 0.875): 2r = 0.25 is the least.
    EXPECT_EQ(limited_face_value(0.875, 1.0, 2.0, 1.5), 1.125);
    // r = 4 (c_up = -3), quick = 1.25: psi_quick = 0.5 is the least.
    EXPECT_EQ(limited_face_value(-3.0, 1.0, 2.0, 1.25), 1.25);
    // r = 4, quick = 3: 2r = 8, 3.25, 4 and the cap 2: psi = 2 gives c_d.
    EXPECT_EQ(limited_face_value(-3.0, 1.0, 2.0, 3.0), 2.0);
    // A local extremum (r < 0, c_up = 1.5): psi = 0 gives c_c.
    EXPECT_EQ(limited_face_value(1.5, 1.0, 2.0, 1.5), 1.0);
}

TEST(LimitedFaceValue, TheFaceValueLiesBetweenTheUpstreamAndDownstreamCells) {
    // psi in [0, 2] puts c_c + psi d / 2 between c_c and c_d whatever c_up and quick.
    // Dyadic values, so c_c + (c_d - c_c) is c_d exactly and rounding, being
    // monotone, cannot carry the face value past either cell.
    const double values[] = {-7.0, -1.0, -0.0, 0.0, 0.375, 1.0, 2.5, 40.0};
    for (double c_up : values) {
        for (double c_c : values) {
            for (double c_d : values) {
                for (double quick : values) {
                    const double f = limited_face_value(c_up, c_c, c_d, quick);
                    EXPECT_GE(f, std::min(c_c, c_d));
                    EXPECT_LE(f, std::max(c_c, c_d));
                }
            }
        }
    }
}

TEST(LagrangeWeights, UniformInteriorFaceIsQuicksEighths) {
    // Nodes 0.5, 1.5, 2.5 (far, upstream, downstream) and the face at 2.0:
    // -1/8, 6/8, 3/8, each exact in binary.
    const auto w = cfdcore::scalar::lagrange_weights(0.5, 1.5, 2.5, 2.0);
    EXPECT_EQ(w[0], -0.125);
    EXPECT_EQ(w[1], 0.75);
    EXPECT_EQ(w[2], 0.375);
}

// ---- The flux --------------------------------------------------------------

// A bool mask from 0 and 1 entries (std::vector<bool> holds no bool array).
std::unique_ptr<bool[]> mask_of(const std::vector<std::uint8_t>& entries) {
    auto mask = std::make_unique<bool[]>(entries.size());
    for (std::size_t i = 0; i < entries.size(); ++i) {
        mask[i] = entries[i] != 0;
    }
    return mask;
}

// The flux on every face normal to d of a field, with a planted flux and no vents.
std::vector<double> flux_of(const scheme<2>& s, int d, const std::vector<double>& c,
                            const std::vector<double>& flux, const std::vector<double>& inflow,
                            bool upwind) {
    std::vector<double> out(flux.size(), -1.0);
    advective_flux(s, d, c, flux, inflow, upwind, {}, out);
    return out;
}

TEST(AdvectiveFlux, AUniformFieldGivesTheFluxTimesItsValue) {
    // Stretched, with SOLID cells and a flux of both signs: every face value
    // is the field's value exactly, QUICK and upwind alike.
    grid2_arrays a(9, 7, 1.13, 0.91);
    for (std::size_t cell : {10u, 11u, 20u, 33u, 48u}) {
        a.types[cell] = solid;
    }
    grid<2> g(a.input());
    scheme<2> s(g);
    const double value = 0.7316;
    std::vector<double> c(g.fields().cell_count(), value);
    for (int d = 0; d < 2; ++d) {
        const std::size_t faces = g.fields().face_count(d);
        std::vector<double> flux(faces);
        for (std::size_t f = 0; f < faces; ++f) {
            flux[f] = (f % 3 == 0 ? -1.0 : 1.0) * (0.1 + 0.37 * static_cast<double>(f % 5));
        }
        flux[4] = 0.0;
        std::vector<double> inflow(faces, value);
        for (bool upwind : {false, true}) {
            const auto out = flux_of(s, d, c, flux, inflow, upwind);
            for (std::size_t f = 0; f < faces; ++f) {
                EXPECT_EQ(bits(out[f]), bits(flux[f] * value)) << d << " " << f << " " << upwind;
            }
        }
    }
}

TEST(AdvectiveFlux, AMonotoneFieldGivesNoNewExtremum) {
    // A field increasing along x (and along y) on a stretched grid, under a
    // flux of +1 or -1 on every face: each interior face value lies between
    // the two cells beside it, so no face carries a value outside the field's.
    grid2_arrays a(12, 9, 1.08, 1.2);
    grid<2> g(a.input());
    scheme<2> s(g);
    const auto& fields = g.fields();
    std::vector<double> c(fields.cell_count());
    for (std::size_t j = 0; j < 9; ++j) {
        for (std::size_t i = 0; i < 12; ++i) {
            const double x = a.xc[i];
            const double y = a.yc[j];
            c[j * 12 + i] = std::exp(1.7 * x) + 3.0 * y * y + (x > 0.8 ? 2.0 : 0.0);
        }
    }
    for (int d = 0; d < 2; ++d) {
        const std::size_t stride = fields.cell_stride(d);
        const std::size_t n = fields.cells(d);
        for (double sign : {1.0, -1.0}) {
            std::vector<double> flux(fields.face_count(d), sign);
            std::vector<double> inflow(fields.face_count(d), 0.0);
            const auto out = flux_of(s, d, c, flux, inflow, false);
            for (const auto& ln : fields.lines(d)) {
                for (std::size_t m = 1; m < n; ++m) {
                    const double face = out[ln.face + m * stride] / sign;
                    const double lo = c[ln.cell + (m - 1) * stride];
                    const double hi = c[ln.cell + m * stride];
                    EXPECT_GE(face, std::min(lo, hi)) << d << " " << m;
                    EXPECT_LE(face, std::max(lo, hi)) << d << " " << m;
                }
            }
        }
    }
}

TEST(AdvectiveFlux, ALineOfCellsIsEachRowOfAFieldConstantAcrossIt) {
    // D = 1 against D = 2: a field constant across y, its flux along x the
    // same on every row, and SOLID cells in whole columns, so each row of
    // the 2D flux along x is the 1D flux on the line, bit for bit.
    const std::size_t nx = 11;
    const std::size_t ny = 5;
    grid2_arrays a(nx, ny, 1.1, 1.0);
    for (std::size_t j = 0; j < ny; ++j) {
        a.types[j * nx + 6] = solid;
    }
    grid<2> g2(a.input());
    scheme<2> s2(g2);
    std::vector<std::int32_t> types1(nx, fluid);
    types1[6] = solid;
    grid<1> g1(grid_input<1>{
        .faces = {a.x}, .centres = {a.xc}, .cell_type = types1, .solid = solid});
    scheme<1> s1(g1);

    std::vector<double> line(nx);
    std::vector<double> line_flux(nx + 1);
    std::vector<double> line_inflow(nx + 1, 0.0);
    for (std::size_t i = 0; i < nx; ++i) {
        line[i] = std::sin(1.3 * static_cast<double>(i)) * 10.0 + 0.5;
    }
    for (std::size_t f = 0; f <= nx; ++f) {
        line_flux[f] = std::cos(0.9 * static_cast<double>(f));
    }
    line_flux[0] = 0.4;
    line_inflow[0] = 3.25;
    line_flux[nx] = -0.2;
    line_inflow[nx] = -1.5;
    std::vector<double> c2(nx * ny);
    std::vector<double> flux2((nx + 1) * ny);
    std::vector<double> inflow2((nx + 1) * ny);
    for (std::size_t j = 0; j < ny; ++j) {
        std::copy(line.begin(), line.end(), c2.begin() + static_cast<std::ptrdiff_t>(j * nx));
        std::copy(line_flux.begin(), line_flux.end(),
                  flux2.begin() + static_cast<std::ptrdiff_t>(j * (nx + 1)));
        std::copy(line_inflow.begin(), line_inflow.end(),
                  inflow2.begin() + static_cast<std::ptrdiff_t>(j * (nx + 1)));
    }
    for (bool upwind : {false, true}) {
        std::vector<double> out1(nx + 1);
        advective_flux(s1, 0, line, line_flux, line_inflow, upwind, {}, out1);
        const auto out2 = flux_of(s2, 0, c2, flux2, inflow2, upwind);
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t f = 0; f <= nx; ++f) {
                EXPECT_EQ(bits(out2[j * (nx + 1) + f]), bits(out1[f])) << j << " " << f;
            }
        }
    }
}

// A line of eight unit cells for the boundary, far-solid and vent rules.
struct line_case {
    std::vector<double> x{0, 1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<double> xc{0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5};
    std::vector<std::int32_t> types = std::vector<std::int32_t>(8, fluid);

    std::vector<double> run(const std::vector<double>& c, const std::vector<double>& flux,
                            const std::vector<double>& inflow, bool upwind,
                            const std::vector<std::uint8_t>& vents = {}) const {
        grid<1> g(grid_input<1>{.faces = {x}, .centres = {xc}, .cell_type = types, .solid = solid});
        scheme<1> s(g);
        std::vector<double> out(9);
        const auto mask = mask_of(vents);
        advective_flux(s, 0, c, flux, inflow, upwind,
                       std::span<const bool>(mask.get(), vents.size()), out);
        return out;
    }
};

TEST(AdvectiveFlux, BoundaryFacesTakeTheInflowOnlyWhereTheFluxEnters) {
    line_case lc;
    const std::vector<double> c{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<double> inflow(9, 0.0);
    inflow[0] = 100.0;
    inflow[8] = 200.0;
    std::vector<double> in_flux(9, 1.0);
    in_flux[8] = -1.0;
    auto out = lc.run(c, in_flux, inflow, false);
    EXPECT_EQ(out[0], 100.0);
    EXPECT_EQ(out[8], -200.0);
    std::vector<double> out_flux(9, 1.0);
    out_flux[0] = -1.0;
    out = lc.run(c, out_flux, inflow, false);
    EXPECT_EQ(out[0], -1.0);
    EXPECT_EQ(out[8], 8.0);
}

TEST(AdvectiveFlux, AFarNodeInASolidCellIsReadAsTheUpstreamValue) {
    // Flow toward +x through face 3 (cells 2 to 3): its far node is cell 1.
    // A spike there changes the face value unless cell 1 is SOLID; SOLID, the
    // face value is the limiter's with c_up = quick = c_c, which is c_c.
    line_case lc;
    std::vector<double> c{1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<double> flux(9, 1.0);
    const std::vector<double> inflow(9, 0.0);
    const auto smooth = lc.run(c, flux, inflow, false);
    c[1] = -1.0e6;
    const auto spiked = lc.run(c, flux, inflow, false);
    EXPECT_NE(spiked[3], smooth[3]);
    lc.types[1] = solid;
    const auto shielded = lc.run(c, flux, inflow, false);
    EXPECT_EQ(shielded[3], 3.0);
    // Against the flow the far node of face 3 is cell 4: the same rule.
    std::vector<double> c2{1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<double> back(9, -1.0);
    line_case lc2;
    c2[4] = 1.0e6;
    EXPECT_NE(lc2.run(c2, back, inflow, false)[3], -4.0);
    lc2.types[4] = solid;
    EXPECT_EQ(lc2.run(c2, back, inflow, false)[3], -4.0);
}

TEST(AdvectiveFlux, TheFarSolidRuleReadsQuickAsTheUpstreamValueToo) {
    // The rule sets c_up and quick to c_c. With only c_up set, r is zero but
    // psi_quick still holds the quadratic's own value, and it decides the sign
    // of a zero psi: here c_c = -0.0 with a negative downstream difference and
    // a positive quick - c_c, so psi_quick is negative and psi = max(0.0, that)
    // is +0.0 where the rule gives -0.0 (its psi_quick is -0.0, a tie that
    // keeps the second argument's sign). The face value is c_c + 0.5 psi d:
    // +0.0 with the rule, -0.0 without it. The SOLID cell holds -40, which
    // makes quick 4.625 from the weights -1/8, 3/4 and 3/8.
    //
    // Flow toward +x through face 3 (cells 2 to 3), far node cell 1.
    line_case with_flow;
    with_flow.types[1] = solid;
    const std::vector<double> c{0.0, -40.0, -0.0, -1.0, 5.0, 6.0, 7.0, 8.0};
    const std::vector<double> inflow(9, 0.0);
    EXPECT_EQ(bits(with_flow.run(c, std::vector<double>(9, 1.0), inflow, false)[3]), bits(0.0));
    // Against the flow, through the same face, the upstream cell is 3 and the
    // far node cell 4: the flux's sign flips the product's.
    line_case against;
    against.types[4] = solid;
    const std::vector<double> c_back{1.0, 2.0, -1.0, -0.0, -40.0, 6.0, 7.0, 8.0};
    EXPECT_EQ(bits(against.run(c_back, std::vector<double>(9, -1.0), inflow, false)[3]),
              bits(-0.0));
}

TEST(AdvectiveFlux, AVentTakesTheInflowEnteringAndTheRoomCellLeaving) {
    // Cells 0-2 SOLID; face 3 (cells 2 and 3) a vent.
    line_case lc;
    for (std::size_t i : {0u, 1u, 2u}) {
        lc.types[i] = solid;
    }
    const std::vector<double> c{9, 9, 9, 4, 5, 6, 7, 8};
    std::vector<double> inflow(9, 0.0);
    inflow[3] = 50.0;
    std::vector<std::uint8_t> vents(9, 0);
    vents[3] = 1;
    for (bool upwind : {false, true}) {
        const std::vector<double> entering(9, 2.0);
        EXPECT_EQ(lc.run(c, entering, inflow, upwind, vents)[3], 100.0);
        const std::vector<double> leaving(9, -2.0);
        EXPECT_EQ(lc.run(c, leaving, inflow, upwind, vents)[3], -8.0);
    }
    // With the SOLID side high: cells 5-7 SOLID, face 5 (cells 4 and 5) a vent.
    line_case high;
    for (std::size_t i : {5u, 6u, 7u}) {
        high.types[i] = solid;
    }
    std::vector<double> inflow_h(9, 0.0);
    inflow_h[5] = 30.0;
    std::vector<std::uint8_t> vents_h(9, 0);
    vents_h[5] = 1;
    const std::vector<double> c_h{1, 2, 3, 4, 5, 9, 9, 9};
    EXPECT_EQ(high.run(c_h, std::vector<double>(9, -1.0), inflow_h, false, vents_h)[5], -30.0);
    EXPECT_EQ(high.run(c_h, std::vector<double>(9, 1.0), inflow_h, false, vents_h)[5], 5.0);
}

TEST(AdvectiveFlux, NoVentsIsAnEmptyMaskBitForBit) {
    line_case lc;
    lc.types[2] = solid;
    const std::vector<double> c{1.1, -2.3, 3.7, 4.2, -5.9, 6.0, 7.5, 0.25};
    std::vector<double> flux{0.5, -1.25, 0.75, -0.3, 0.2, 0.0, -0.6, 0.9, -0.4};
    const std::vector<double> inflow(9, 0.125);
    const auto none = lc.run(c, flux, inflow, false);
    const auto zero = lc.run(c, flux, inflow, false, std::vector<std::uint8_t>(9, 0));
    for (std::size_t f = 0; f < 9; ++f) {
        EXPECT_EQ(bits(none[f]), bits(zero[f])) << f;
    }
}

TEST(Scheme, InteriorWeightsOfAUniformGridAreQuicksEighths) {
    // Faces at the integers: the interior faces whose three nodes are all
    // cell centres carry -1/8, 6/8, 3/8 in both directions; the faces beside
    // a boundary node, half a cell from its neighbour, do not.
    line_case lc;
    grid<1> g(grid_input<1>{
        .faces = {lc.x}, .centres = {lc.xc}, .cell_type = lc.types, .solid = solid});
    scheme<1> s(g);
    const auto& w = s.weights(0);
    ASSERT_EQ(w.positive[0].size(), 7u);
    for (std::size_t f = 1; f < 7; ++f) {
        EXPECT_EQ(w.positive[0][f], -0.125) << f;
        EXPECT_EQ(w.positive[1][f], 0.75) << f;
        EXPECT_EQ(w.positive[2][f], 0.375) << f;
    }
    for (std::size_t f = 0; f < 6; ++f) {
        EXPECT_EQ(w.negative[0][f], -0.125) << f;
        EXPECT_EQ(w.negative[1][f], 0.75) << f;
        EXPECT_EQ(w.negative[2][f], 0.375) << f;
    }
    EXPECT_NE(w.positive[0][0], -0.125);
    EXPECT_NE(w.negative[0][6], -0.125);
}

// ---- The implicit solve ----------------------------------------------------

using cfdcore::layout::layout;
using cfdcore::scalar::acceptance;
using cfdcore::scalar::check_leftover_bound;
using cfdcore::scalar::implicit_input;
using cfdcore::scalar::implicit_method;
using cfdcore::scalar::implicit_result;
using cfdcore::scalar::implicit_scratch;
using cfdcore::scalar::implicit_status;
using cfdcore::scalar::implicit_system;
using cfdcore::scalar::implicit_step;

// A bool cell field: std::vector<bool> has no contiguous storage to span.
class bool_field {
public:
    explicit bool_field(std::size_t n, bool value = false) : data_(new bool[n]), n_(n) {
        std::fill(data_.get(), data_.get() + n, value);
    }
    bool& operator[](std::size_t i) { return data_[i]; }
    std::span<const bool> span() const { return {data_.get(), n_}; }

private:
    std::unique_ptr<bool[]> data_;
    std::size_t n_;
};

// A two-dimensional system of [ny, nx] cells with every array it needs, held so
// the input's spans stay valid.
struct system2 {
    system2(std::size_t nx_, std::size_t ny_)
        : nx(nx_),
          ny(ny_),
          fields({nx_, ny_}),
          c_star(nx_ * ny_, 0.0),
          volume(nx_ * ny_, 1.0),
          dt_cells(),
          g_u(ny_ * (nx_ + 1), 0.0),
          g_v((ny_ + 1) * nx_, 0.0),
          sink(nx_ * ny_, 0.0),
          solid(nx_ * ny_),
          held(nx_ * ny_) {}

    // The u face east of cell (i, j) and the v face north of it.
    double& east(std::size_t i, std::size_t j) { return g_u[j * (nx + 1) + i + 1]; }
    double& north(std::size_t i, std::size_t j) { return g_v[(j + 1) * nx + i]; }

    // Interior conductances between two non-SOLID cells, `value(i, j)` each,
    // and zero on the domain faces, as the transport solver forms them.
    template <typename F>
    void conduct(F value) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                const bool here = solid[j * nx + i];
                if (i + 1 < nx && !here && !solid[j * nx + i + 1]) {
                    east(i, j) = value(i, j);
                }
                if (j + 1 < ny && !here && !solid[(j + 1) * nx + i]) {
                    north(i, j) = 0.5 * value(i, j);
                }
            }
        }
    }

    implicit_input<2> input(implicit_method method, double tol, std::int64_t max_iter,
                            bool with_held = false) const {
        implicit_input<2> in{};
        in.c_star = c_star;
        in.volume = volume;
        in.dt = dt;
        in.dt_cells = dt_cells;
        in.conductance = {std::span<const double>(g_u), std::span<const double>(g_v)};
        in.sink = sink;
        in.solid = solid.span();
        if (with_held) {
            in.held = held.span();
        }
        in.tol = tol;
        in.max_iter = max_iter;
        in.method = method;
        return in;
    }

    std::size_t nx;
    std::size_t ny;
    layout<2> fields;
    std::vector<double> c_star;
    std::vector<double> volume;
    double dt = 1.0;
    std::vector<double> dt_cells;
    std::vector<double> g_u;
    std::vector<double> g_v;
    std::vector<double> sink;
    bool_field solid;
    bool_field held;
};

// Solves a system and returns the field; the result goes to `result`.
std::vector<double> solve(const implicit_input<2>& in, const layout<2>& fields,
                          implicit_result& result) {
    implicit_scratch<2> scratch(fields);
    std::vector<double> out(fields.cell_count(), -1.0);
    EXPECT_EQ(implicit_step(in, scratch, out, result), implicit_status::ok);
    return out;
}

// A system of 9 by 7 cells with a SOLID block, conductances over three decades
// and a sink in half the cells; C* over many decades, zero in the SOLID cells.
system2 planted_system() {
    system2 s(9, 7);
    for (std::size_t j = 2; j < 4; ++j) {
        for (std::size_t i = 3; i < 5; ++i) {
            s.solid[j * s.nx + i] = true;
        }
    }
    s.conduct([](std::size_t i, std::size_t j) {
        return std::ldexp(1.0 + 0.1 * static_cast<double>(i), static_cast<int>(j % 4) - 3);
    });
    for (std::size_t k = 0; k < s.c_star.size(); ++k) {
        s.volume[k] = 0.01 * (1.0 + 0.05 * static_cast<double>(k % 5));
        s.sink[k] = (k % 2 == 0) ? 0.003 * static_cast<double>(k % 7) : 0.0;
        s.c_star[k] = s.solid[k] ? 0.0 : std::pow(10.0, static_cast<double>(k % 13) - 9.0);
    }
    s.dt = 0.25;
    return s;
}

TEST(ImplicitStep, TheHandSystemsLeftoverIsTheChangeTheSinkDoesNotAccountFor) {
    // Three cells in a line, V = 1 and dt = 1, so V / dt = 1; conductance 1
    // across both interior faces and none at the ends; a sink of 1 in the
    // middle cell: a_P = [2, 4, 2]. C* = [0, 2, 0], b = C*. With tol 0 the
    // stop never holds and the cap ends the solve.
    // One sweep: r = b + sum G C_N - a_P C = [2, -6, 2], C = [1, 0.5, 1].
    //   Its residual: [0 + 0.5 - 2, 2 + 2 - 2, 0 + 0.5 - 2] = [-1.5, 2, -1.5],
    //   leftover -1. The total changed by 2.5 - 2 = 0.5; the sink took
    //   dt D C = 0.5; -(0.5 + 0.5) = -1.
    // Two sweeps: C = [0.25, 1, 0.25], residual [0.5, -1.5, 0.5], leftover
    //   -0.5. The total changed by -0.5, the sink took 1: -(-0.5 + 1) = -0.5.
    layout<1> fields({3});
    const std::vector<double> c_star{0.0, 2.0, 0.0};
    const std::vector<double> volume{1.0, 1.0, 1.0};
    const std::vector<double> g{0.0, 1.0, 1.0, 0.0};
    const std::vector<double> sink{0.0, 1.0, 0.0};
    bool_field solid(3);
    const std::vector<std::vector<double>> fields_after{{1.0, 0.5, 1.0}, {0.25, 1.0, 0.25}};
    const std::vector<double> leftovers{-1.0, -0.5};
    for (std::int64_t sweeps = 1; sweeps <= 2; ++sweeps) {
        implicit_input<1> in{};
        in.c_star = c_star;
        in.volume = volume;
        in.dt = 1.0;
        in.conductance = {std::span<const double>(g)};
        in.sink = sink;
        in.solid = solid.span();
        in.tol = 0.0;
        in.max_iter = sweeps;
        in.method = implicit_method::jacobi;
        implicit_scratch<1> scratch(fields);
        std::vector<double> out(3, -1.0);
        implicit_result result;
        ASSERT_EQ(implicit_step(in, scratch, out, result), implicit_status::ok);
        EXPECT_EQ(out, fields_after[sweeps - 1]) << sweeps;
        EXPECT_EQ(result.iterations, sweeps);
        EXPECT_FALSE(result.converged);
        EXPECT_EQ(result.leftover, leftovers[sweeps - 1]) << sweeps;
        double change = 0.0;
        double sunk = 0.0;
        for (std::size_t i = 0; i < 3; ++i) {
            change += volume[i] * (out[i] - c_star[i]);
            sunk += in.dt * sink[i] * out[i];
        }
        EXPECT_EQ(result.leftover, -(change + sunk)) << sweeps;
    }
}

TEST(ImplicitStep, AUniformFieldWithNoSinkIsUnchanged) {
    // Conductances only between non-SOLID cells and none at the domain faces,
    // as the callers form them: on a uniform field every diffusion term
    // cancels, so the residual is rounding alone and the stop holds at once.
    for (const bool per_cell : {false, true}) {
        system2 s = planted_system();
        std::fill(s.sink.begin(), s.sink.end(), 0.0);
        for (std::size_t k = 0; k < s.c_star.size(); ++k) {
            s.c_star[k] = s.solid[k] ? 0.0 : 3.7e5;
        }
        if (per_cell) {
            s.dt_cells.resize(s.c_star.size());
            for (std::size_t k = 0; k < s.dt_cells.size(); ++k) {
                s.dt_cells[k] = 0.01 * (1.0 + static_cast<double>(k % 3));
            }
        }
        for (const auto method : {implicit_method::jacobi, implicit_method::conjugate_gradient}) {
            implicit_result result;
            const auto out = solve(s.input(method, 1e-12, 50), s.fields, result);
            for (std::size_t k = 0; k < out.size(); ++k) {
                EXPECT_EQ(bits(out[k]), bits(s.c_star[k])) << k;
            }
            EXPECT_EQ(result.iterations, 0);
            EXPECT_TRUE(result.converged);
            EXPECT_EQ(result.fallbacks, 0);
        }
    }
}

TEST(ImplicitStep, ANonNegativeStarGivesNonNegativeJacobiIterates) {
    // Strong diffusion (dt G / V from about 21 to 680 over the faces, with
    // dt = 4, V of 0.010 to 0.012 and G of 0.0625 to 1.7) over a field of many
    // decades with zeros beside large values: every iterate, read by capping
    // the solve at each sweep in turn with a stop that never holds, is
    // non-negative.
    system2 s = planted_system();
    s.dt = 4.0;
    for (std::size_t k = 0; k < s.c_star.size(); ++k) {
        if (k % 4 == 1 && !s.solid[k]) {
            s.c_star[k] = 0.0;
        }
        if (k % 11 == 0 && !s.solid[k]) {
            s.c_star[k] = 1e-300;
        }
    }
    for (std::int64_t cap = 0; cap <= 40; ++cap) {
        implicit_result result;
        const auto out = solve(s.input(implicit_method::jacobi, 0.0, cap), s.fields, result);
        EXPECT_EQ(result.iterations, cap);
        for (std::size_t k = 0; k < out.size(); ++k) {
            EXPECT_GE(out[k], 0.0) << cap << " " << k;
        }
    }
}

TEST(ImplicitStep, NothingToSolveReturnsCStar) {
    // No conductance and no sink, written as -0.0 too (`.any()` reads it as
    // zero): C* comes back as it was, SOLID values included, with no solve.
    for (const double zero : {0.0, -0.0}) {
        system2 s = planted_system();
        std::fill(s.g_u.begin(), s.g_u.end(), zero);
        std::fill(s.g_v.begin(), s.g_v.end(), zero);
        std::fill(s.sink.begin(), s.sink.end(), zero);
        s.c_star[3 * s.nx + 3] = 5.0;
        implicit_result result;
        const auto out = solve(s.input(implicit_method::jacobi, 1e-8, 10), s.fields, result);
        EXPECT_EQ(out, s.c_star);
        EXPECT_EQ(result.iterations, 0);
        EXPECT_TRUE(result.converged);
        EXPECT_EQ(bits(result.leftover), bits(0.0));
        EXPECT_EQ(result.leftover_sums, 0);
    }
}

TEST(ImplicitSystem, TheDiagonalIsWestEastSouthNorthThenTheSink) {
    // One cell: W = 2^53, E = 1, S = 1, N = -2^53, sink 1, V / dt = 1. In the
    // reference's order ((((W + E) + S) + N) + sink) = ((2^53 + 1) - 2^53) + 1
    // = 1 (2^53 + 1 is a tie that rounds to 2^53), so a_P = 2; north before
    // south would give 2^53 - 2^53 + 1 + 1 = 2 and a_P = 3.
    system2 s(1, 1);
    s.g_u = {0x1p53, 1.0};
    s.g_v = {1.0, -0x1p53};
    s.sink = {1.0};
    s.c_star = {3.0};
    implicit_scratch<2> scratch(s.fields);
    const auto setup = implicit_system(s.input(implicit_method::jacobi, 0.5, 1), scratch);
    EXPECT_TRUE(setup.diffuses);
    EXPECT_EQ(scratch.a_p[0], 2.0);
    EXPECT_EQ(scratch.b[0], 3.0);
    EXPECT_EQ(setup.stop, 1.5);
}

TEST(ImplicitSystem, ANanIsSomethingToSolveAndMakesTheStopNan) {
    system2 s(2, 1);
    s.sink = {0.0, std::numeric_limits<double>::quiet_NaN()};
    s.c_star = {1.0, 1.0};
    implicit_scratch<2> scratch(s.fields);
    const auto setup = implicit_system(s.input(implicit_method::jacobi, 1e-8, 1), scratch);
    EXPECT_TRUE(setup.diffuses);
    EXPECT_FALSE(std::isnan(setup.stop));
    s.c_star = {1.0, std::numeric_limits<double>::quiet_NaN()};
    const auto with_nan = implicit_system(s.input(implicit_method::jacobi, 1e-8, 1), scratch);
    EXPECT_TRUE(std::isnan(with_nan.stop));
}

TEST(ImplicitStep, TheNeighbourSumIsEastWestNorthSouth) {
    // The centre of 3 by 3 cells at C* = 0, its eight neighbours held at 1:
    // its residual is its neighbour sum, ((E + W) + N) + S with E = 2^53,
    // W = 1, N = 1, S = -2^53, which is (2^53 + 1) - 2^53 = 0, since
    // 2^53 + 1 + 1 rounds to 2^53 twice. Every other residual is held at zero,
    // so the stop holds at once with tol 0 and the leftover is 0. South before
    // north would give 1, and the cap of no sweeps would leave the leftover 1.
    system2 s(3, 3);
    std::fill(s.c_star.begin(), s.c_star.end(), 1.0);
    s.c_star[4] = 0.0;
    for (std::size_t k = 0; k < 9; ++k) {
        s.held[k] = k != 4;
    }
    s.east(1, 1) = 0x1p53;  // east of the centre
    s.east(0, 1) = 1.0;     // west of the centre
    s.north(1, 1) = 1.0;    // north of the centre
    s.north(1, 0) = -0x1p53;  // south of the centre
    implicit_result result;
    const auto out = solve(s.input(implicit_method::jacobi, 0.0, 0, true), s.fields, result);
    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.iterations, 0);
    EXPECT_EQ(bits(result.leftover), bits(0.0));
    EXPECT_EQ(out, s.c_star);
}

TEST(ImplicitStep, HeldAndSolidCellsKeepTheirValues) {
    // Held cells keep C* bit for bit and SOLID cells end at zero, under both
    // methods, from a C* that is not zero in its SOLID cells.
    system2 s = planted_system();
    for (std::size_t k = 0; k < s.c_star.size(); ++k) {
        s.held[k] = !s.solid[k] && k % 6 == 0;
        if (s.solid[k]) {
            s.c_star[k] = 7.0;
        }
    }
    for (const auto method : {implicit_method::jacobi, implicit_method::conjugate_gradient}) {
        implicit_result result;
        const auto out = solve(s.input(method, 1e-10, 500, true), s.fields, result);
        EXPECT_TRUE(result.converged);
        for (std::size_t k = 0; k < out.size(); ++k) {
            if (s.solid[k]) {
                EXPECT_EQ(bits(out[k]), bits(0.0)) << k;
            } else if (s.held[k]) {
                EXPECT_EQ(bits(out[k]), bits(s.c_star[k])) << k;
            }
        }
    }
}

TEST(ImplicitStep, ConjugateGradientsReachTheDenseSolution) {
    // The system solved densely by Gaussian elimination with partial pivoting,
    // against CG at a tight tolerance with a per-cell step.
    system2 s = planted_system();
    s.dt_cells.resize(s.c_star.size());
    for (std::size_t k = 0; k < s.dt_cells.size(); ++k) {
        s.dt_cells[k] = 0.1 * (1.0 + static_cast<double>(k % 4));
    }
    implicit_result result;
    const auto out =
        solve(s.input(implicit_method::conjugate_gradient, 1e-14, 500), s.fields, result);
    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.fallbacks, 0);
    EXPECT_GT(result.iterations, 0);
    const std::size_t n = s.c_star.size();
    std::vector<double> a(n * n, 0.0);
    std::vector<double> rhs(n, 0.0);
    for (std::size_t j = 0; j < s.ny; ++j) {
        for (std::size_t i = 0; i < s.nx; ++i) {
            const std::size_t k = j * s.nx + i;
            if (s.solid[k]) {
                a[k * n + k] = 1.0;
                continue;
            }
            const double w = s.g_u[j * (s.nx + 1) + i];
            const double e = s.g_u[j * (s.nx + 1) + i + 1];
            const double so = s.g_v[j * s.nx + i];
            const double no = s.g_v[(j + 1) * s.nx + i];
            const double over_dt = s.volume[k] / s.dt_cells[k];
            a[k * n + k] = over_dt + w + e + so + no + s.sink[k];
            if (i > 0) a[k * n + k - 1] = -w;
            if (i + 1 < s.nx) a[k * n + k + 1] = -e;
            if (j > 0) a[k * n + k - s.nx] = -so;
            if (j + 1 < s.ny) a[k * n + k + s.nx] = -no;
            rhs[k] = over_dt * s.c_star[k];
        }
    }
    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        for (std::size_t r = col + 1; r < n; ++r) {
            if (std::fabs(a[r * n + col]) > std::fabs(a[pivot * n + col])) pivot = r;
        }
        for (std::size_t c = 0; c < n; ++c) std::swap(a[col * n + c], a[pivot * n + c]);
        std::swap(rhs[col], rhs[pivot]);
        for (std::size_t r = col + 1; r < n; ++r) {
            const double f = a[r * n + col] / a[col * n + col];
            for (std::size_t c = col; c < n; ++c) a[r * n + c] -= f * a[col * n + c];
            rhs[r] -= f * rhs[col];
        }
    }
    std::vector<double> exact(n, 0.0);
    for (std::size_t r = n; r-- > 0;) {
        double sum = rhs[r];
        for (std::size_t c = r + 1; c < n; ++c) sum -= a[r * n + c] * exact[c];
        exact[r] = sum / a[r * n + r];
    }
    double largest = 0.0;
    for (double v : exact) largest = std::max(largest, std::fabs(v));
    for (std::size_t k = 0; k < n; ++k) {
        EXPECT_NEAR(out[k], exact[k], 1e-11 * largest) << k;
    }
}

TEST(ImplicitStep, AZeroLeftByConjugateGradientsFallsBackToJacobiFromCStar) {
    // A cell with no conductance, no sink and C* = 0 keeps the value 0 under
    // CG, which is not positive: the CG solve is discarded and Jacobi solves
    // from C*, so the field, the converged flag and the leftover are the
    // Jacobi solve's bit for bit.
    system2 s = planted_system();
    const std::size_t lone = 5 * s.nx + 7;
    s.c_star[lone] = 0.0;
    s.sink[lone] = 0.0;
    s.east(7, 5) = 0.0;
    s.east(6, 5) = 0.0;
    s.north(7, 5) = 0.0;
    s.north(7, 4) = 0.0;
    implicit_result cg;
    const auto out = solve(s.input(implicit_method::conjugate_gradient, 1e-10, 500), s.fields, cg);
    implicit_result jacobi;
    const auto alone = solve(s.input(implicit_method::jacobi, 1e-10, 500), s.fields, jacobi);
    EXPECT_EQ(cg.fallbacks, 1);
    EXPECT_GT(cg.iterations, 0);
    EXPECT_EQ(cg.fallback_sweeps, jacobi.iterations);
    EXPECT_EQ(cg.converged, jacobi.converged);
    EXPECT_EQ(bits(cg.leftover), bits(jacobi.leftover));
    for (std::size_t k = 0; k < out.size(); ++k) {
        EXPECT_EQ(bits(out[k]), bits(alone[k])) << k;
    }
}

TEST(ImplicitStep, ConjugateGradientsAtTheCapFallBackToJacobi) {
    system2 s = planted_system();
    implicit_result cg;
    const auto out = solve(s.input(implicit_method::conjugate_gradient, 1e-12, 2), s.fields, cg);
    implicit_result jacobi;
    const auto alone = solve(s.input(implicit_method::jacobi, 1e-12, 2), s.fields, jacobi);
    EXPECT_EQ(cg.iterations, 2);
    EXPECT_EQ(cg.fallbacks, 1);
    EXPECT_EQ(cg.fallback_sweeps, 2);
    EXPECT_FALSE(cg.converged);
    EXPECT_EQ(bits(cg.leftover), bits(jacobi.leftover));
    for (std::size_t k = 0; k < out.size(); ++k) {
        EXPECT_EQ(bits(out[k]), bits(alone[k])) << k;
    }
}

TEST(ImplicitStep, ANegativeOrNanBoundIsRefusedAndNothingIsWritten) {
    EXPECT_EQ(check_leftover_bound(std::nullopt), implicit_status::ok);
    EXPECT_EQ(check_leftover_bound(0.0), implicit_status::ok);
    EXPECT_EQ(check_leftover_bound(std::numeric_limits<double>::infinity()), implicit_status::ok);
    EXPECT_EQ(check_leftover_bound(-1e-300), implicit_status::bound_negative_or_nan);
    EXPECT_EQ(check_leftover_bound(std::numeric_limits<double>::quiet_NaN()),
              implicit_status::bound_negative_or_nan);
    system2 s = planted_system();
    auto in = s.input(implicit_method::jacobi, 1e-8, 10);
    in.leftover_bound = -1.0;
    implicit_scratch<2> scratch(s.fields);
    std::vector<double> out(s.c_star.size(), -1.0);
    implicit_result result;
    result.iterations = 99;
    EXPECT_EQ(implicit_step(in, scratch, out, result), implicit_status::bound_negative_or_nan);
    EXPECT_EQ(result.iterations, 99);
    EXPECT_EQ(out, std::vector<double>(s.c_star.size(), -1.0));
}

TEST(ImplicitStep, AHeldArrayOfAnotherLengthThrows) {
    // set_masks reads held[i] for every cell: a shorter array would be read
    // past its end, a longer one is not one flag per cell. Both methods form
    // the masks, so both refuse (review 89 S9's pattern, for `held`).
    system2 s = planted_system();
    for (const auto method : {implicit_method::jacobi, implicit_method::conjugate_gradient}) {
        for (const std::size_t n : {s.c_star.size() - 1, s.c_star.size() + 1}) {
            bool_field wrong(n);
            auto in = s.input(method, 1e-8, 10);
            in.held = wrong.span();
            implicit_scratch<2> scratch(s.fields);
            std::vector<double> out(s.c_star.size(), -1.0);
            implicit_result result;
            EXPECT_THROW(implicit_step(in, scratch, out, result), std::invalid_argument);
        }
        // One flag per cell is accepted.
        auto in = s.input(method, 1e-8, 10, true);
        implicit_scratch<2> scratch(s.fields);
        std::vector<double> out(s.c_star.size(), -1.0);
        implicit_result result;
        EXPECT_EQ(implicit_step(in, scratch, out, result), implicit_status::ok);
    }
}

// ---- The two solvers called directly ---------------------------------------
//
// The hand system of the leftover test above: three cells in a line, V = dt = 1,
// conductance 1 across both interior faces, a sink of 1 in the middle cell,
// a_P = [2, 4, 2], C* = [0, 2, 0] = b. Its residual at C* is [2, -6, 2]. The
// exact solution of A C = b is [1/3, 2/3, 1/3]: the sum of C changes by
// 4/3 - 2 = -2/3 and the sink takes 2/3, so the leftover of the solution is 0.
//
// Jacobi from C*, each iterate with its residual (the leftover is dt times the
// residual's sum):
//   C0 = [0, 2, 0]         r0 = [2, -6, 2]         leftover -2
//   C1 = [1, 0.5, 1]       r1 = [-1.5, 2, -1.5]    leftover -1
//   C2 = [0.25, 1, 0.25]   r2 = [0.5, -1.5, 0.5]   leftover -0.5
// CG from C*, with the diagonal preconditioner: z = [1, -1.5, 1], p = z,
// r.z = 13, A p = [3.5, -8, 3.5], p.A p = 19, so alpha = 13/19 and
//   C1 = [13/19, 37/38, 13/19]   r1 = [-15/38, -10/19, -15/38]   leftover -25/19.
// Restarted from r1 (the true residual), the second iteration gives
//   C2 = [0.31767.., 0.72932.., 0.31767..]   leftover -25/266.
// The stop is tol times max |b| = 2 tol, so tol 3 makes it 6, the largest |r0|:
// the stop met with equality at the start.

struct direct_solve {
    cfdcore::scalar::solve_outcome outcome;
    std::vector<double> field;
    double leftover;
    std::int64_t sums;
};

enum class solver { jacobi, conjugate_gradient };

// Sets the system up and calls the one solver, without `implicit_step`'s
// early return or fallback.
template <int D>
direct_solve solve_directly(const implicit_input<D>& in, const layout<D>& fields, solver which) {
    implicit_scratch<D> scratch(fields);
    const auto setup = implicit_system(in, scratch);
    EXPECT_TRUE(setup.diffuses);
    acceptance accept(in.dt, in.dt_cells, in.solid, in.leftover_bound, scratch.dt_masked,
                      scratch.terms);
    direct_solve result{};
    if (which == solver::jacobi) {
        result.field.assign(fields.cell_count(), -1.0);
        result.outcome = cfdcore::scalar::jacobi(in, setup.stop, accept, scratch, result.field);
    } else {
        result.outcome = cfdcore::scalar::conjugate_gradient(in, setup.stop, accept, scratch);
        result.field = scratch.c;
    }
    result.leftover = accept.leftover();
    result.sums = accept.sums();
    return result;
}

// The hand system as a 1D line, and as a line along each direction of a 2D
// layout (an x line of 3 by 1 cells and a y line of 1 by 3). `check` is called
// with each one's input and layout, and a label.
template <typename Check>
void for_each_hand_layout(double tol, std::int64_t max_iter, std::optional<double> bound,
                          Check check) {
    const std::vector<double> c_star{0.0, 2.0, 0.0};
    const std::vector<double> volume{1.0, 1.0, 1.0};
    const std::vector<double> sink{0.0, 1.0, 0.0};
    {
        const std::vector<double> g{0.0, 1.0, 1.0, 0.0};
        bool_field not_solid(3);
        layout<1> fields({3});
        implicit_input<1> in{};
        in.c_star = c_star;
        in.volume = volume;
        in.dt = 1.0;
        in.conductance = {std::span<const double>(g)};
        in.sink = sink;
        in.solid = not_solid.span();
        in.tol = tol;
        in.max_iter = max_iter;
        in.leftover_bound = bound;
        check(in, fields, "1D line");
    }
    {
        system2 along_x(3, 1);
        along_x.c_star = c_star;
        along_x.sink = sink;
        along_x.east(0, 0) = 1.0;
        along_x.east(1, 0) = 1.0;
        auto in = along_x.input(implicit_method::jacobi, tol, max_iter);
        in.leftover_bound = bound;
        check(in, along_x.fields, "2D line along x");
    }
    {
        system2 along_y(1, 3);
        along_y.c_star = c_star;
        along_y.sink = sink;
        along_y.north(0, 0) = 1.0;
        along_y.north(0, 1) = 1.0;
        auto in = along_y.input(implicit_method::jacobi, tol, max_iter);
        in.leftover_bound = bound;
        check(in, along_y.fields, "2D line along y");
    }
}

TEST(Jacobi, OneSweepFromCStarIsTheHandSystemsFirstIterate) {
    for_each_hand_layout(0.0, 1, std::nullopt, [](const auto& in, const auto& fields, auto label) {
        const direct_solve s = solve_directly(in, fields, solver::jacobi);
        EXPECT_EQ(s.field, (std::vector<double>{1.0, 0.5, 1.0})) << label;
        EXPECT_EQ(s.outcome.iterations, 1) << label;
        EXPECT_FALSE(s.outcome.converged) << label;
        EXPECT_EQ(s.leftover, -1.0) << label;
        EXPECT_EQ(s.sums, 1) << label;
    });
}

TEST(Jacobi, ConvergesToTheHandSystemsSolution) {
    for_each_hand_layout(1e-14, 500, std::nullopt, [](const auto& in, const auto& f, auto label) {
        const direct_solve s = solve_directly(in, f, solver::jacobi);
        EXPECT_TRUE(s.outcome.converged) << label;
        // The residual halves each sweep, from 6 to 2e-14: about 48 sweeps.
        EXPECT_GT(s.outcome.iterations, 40) << label;
        EXPECT_LT(s.outcome.iterations, 60) << label;
        EXPECT_NEAR(s.field[0], 1.0 / 3.0, 1e-13) << label;
        EXPECT_NEAR(s.field[1], 2.0 / 3.0, 1e-13) << label;
        EXPECT_NEAR(s.field[2], 1.0 / 3.0, 1e-13) << label;
        EXPECT_NEAR(s.leftover, 0.0, 1e-13) << label;
    });
}

TEST(Jacobi, TheStopHoldsWithEqualityAndNotBelowIt) {
    // Tol 3 makes the stop 6, the largest |r0|: met at the start, so no sweep.
    for_each_hand_layout(3.0, 10, std::nullopt, [](const auto& in, const auto& fields, auto label) {
        const direct_solve s = solve_directly(in, fields, solver::jacobi);
        EXPECT_EQ(s.outcome.iterations, 0) << label;
        EXPECT_TRUE(s.outcome.converged) << label;
        EXPECT_EQ(s.field, (std::vector<double>{0.0, 2.0, 0.0})) << label;
        EXPECT_EQ(s.leftover, -2.0) << label;
    });
    // One step below 3: the stop is just under 6, so C0 does not meet it, and
    // C1 (largest |r1| = 2) does.
    for_each_hand_layout(std::nextafter(3.0, 0.0), 10, std::nullopt,
                         [](const auto& in, const auto& fields, auto label) {
                             const direct_solve s = solve_directly(in, fields, solver::jacobi);
                             EXPECT_EQ(s.outcome.iterations, 1) << label;
                             EXPECT_TRUE(s.outcome.converged) << label;
                         });
}

TEST(Jacobi, ABoundSweepsOnUntilTheLeftoverIsInsideIt) {
    // The stop holds at every iterate (tol 3), so the bound decides. At C0 the
    // leftover is -2, at C1 -1, at C2 -0.5, each formed once by the test.
    struct row {
        double bound;
        std::int64_t sweeps;
        double leftover;
        std::int64_t sums;
    };
    for (const row& expected : {row{1.5, 1, -1.0, 2}, row{1.0, 1, -1.0, 2}, row{0.5, 2, -0.5, 3}}) {
        for_each_hand_layout(3.0, 10, expected.bound, [&](const auto& in, const auto& f, auto lab) {
            const char* label = lab;
            const direct_solve s = solve_directly(in, f, solver::jacobi);
            EXPECT_EQ(s.outcome.iterations, expected.sweeps) << label << " " << expected.bound;
            EXPECT_TRUE(s.outcome.converged) << label << " " << expected.bound;
            EXPECT_EQ(s.leftover, expected.leftover) << label << " " << expected.bound;
            EXPECT_EQ(s.sums, expected.sums) << label << " " << expected.bound;
        });
    }
    // A zero bound is never met by these leftovers: the cap ends the solve,
    // and its last iterate's leftover, formed by the test, is not formed again.
    for_each_hand_layout(3.0, 2, 0.0, [](const auto& in, const auto& fields, auto label) {
        const direct_solve s = solve_directly(in, fields, solver::jacobi);
        EXPECT_EQ(s.outcome.iterations, 2) << label;
        EXPECT_FALSE(s.outcome.converged) << label;
        EXPECT_EQ(s.leftover, -0.5) << label;
        EXPECT_EQ(s.sums, 3) << label;
    });
}

TEST(ConjugateGradient, OneIterationIsTheHandSystemsClosedForm) {
    for_each_hand_layout(0.0, 1, std::nullopt, [](const auto& in, const auto& fields, auto label) {
        const direct_solve s = solve_directly(in, fields, solver::conjugate_gradient);
        EXPECT_EQ(s.outcome.iterations, 1) << label;
        EXPECT_FALSE(s.outcome.converged) << label;
        EXPECT_EQ(s.outcome.restarts, 0) << label;
        EXPECT_NEAR(s.field[0], 13.0 / 19.0, 1e-15) << label;
        EXPECT_NEAR(s.field[1], 37.0 / 38.0, 1e-15) << label;
        EXPECT_NEAR(s.field[2], 13.0 / 19.0, 1e-15) << label;
        EXPECT_NEAR(s.leftover, -25.0 / 19.0, 1e-14) << label;
        EXPECT_EQ(s.sums, 1) << label;
    });
}

TEST(ConjugateGradient, ConvergesToTheHandSystemsSolutionInTwoOrThreeIterations) {
    // The Krylov space of a start symmetric about the middle cell has two
    // dimensions, so the second iterate is the solution up to rounding.
    for_each_hand_layout(1e-12, 50, std::nullopt, [](const auto& in, const auto& f, auto label) {
        const direct_solve s = solve_directly(in, f, solver::conjugate_gradient);
        EXPECT_TRUE(s.outcome.converged) << label;
        EXPECT_GE(s.outcome.iterations, 2) << label;
        EXPECT_LE(s.outcome.iterations, 3) << label;
        EXPECT_EQ(s.outcome.restarts, 0) << label;
        EXPECT_NEAR(s.field[0], 1.0 / 3.0, 1e-13) << label;
        EXPECT_NEAR(s.field[1], 2.0 / 3.0, 1e-13) << label;
        EXPECT_NEAR(s.field[2], 1.0 / 3.0, 1e-13) << label;
        EXPECT_NEAR(s.leftover, 0.0, 1e-13) << label;
    });
}

TEST(ConjugateGradient, TheStopHoldsWithEqualityAtTheStart) {
    // Tol 3: the stop is 6, the largest |r0|, so no iteration is made.
    for_each_hand_layout(3.0, 10, std::nullopt, [](const auto& in, const auto& fields, auto label) {
        const direct_solve s = solve_directly(in, fields, solver::conjugate_gradient);
        EXPECT_EQ(s.outcome.iterations, 0) << label;
        EXPECT_TRUE(s.outcome.converged) << label;
        EXPECT_EQ(s.field, (std::vector<double>{0.0, 2.0, 0.0})) << label;
        EXPECT_EQ(s.leftover, -2.0) << label;
        EXPECT_EQ(s.sums, 1) << label;
    });
}

TEST(ConjugateGradient, ABoundRestartsFromTheTrueResidualUntilTheLeftoverIsInsideIt) {
    // The max-norm stop holds throughout (tol 3), so the bound decides.
    // 1.5 takes the first iterate (leftover -25/19). 1.0 refuses it, the solve
    // restarts from its true residual and the second iterate (leftover
    // -25/266) is taken. A zero bound refuses both, and the cap ends the solve.
    for_each_hand_layout(3.0, 10, 1.5, [](const auto& in, const auto& fields, auto label) {
        const direct_solve s = solve_directly(in, fields, solver::conjugate_gradient);
        EXPECT_EQ(s.outcome.iterations, 1) << label;
        EXPECT_TRUE(s.outcome.converged) << label;
        EXPECT_EQ(s.outcome.restarts, 0) << label;
        EXPECT_NEAR(s.leftover, -25.0 / 19.0, 1e-14) << label;
        EXPECT_EQ(s.sums, 2) << label;
    });
    for_each_hand_layout(3.0, 10, 1.0, [](const auto& in, const auto& fields, auto label) {
        const direct_solve s = solve_directly(in, fields, solver::conjugate_gradient);
        EXPECT_EQ(s.outcome.iterations, 2) << label;
        EXPECT_TRUE(s.outcome.converged) << label;
        EXPECT_EQ(s.outcome.restarts, 1) << label;
        EXPECT_NEAR(s.field[0], 0.3176691729323308, 1e-14) << label;
        EXPECT_NEAR(s.field[1], 0.7293233082706767, 1e-14) << label;
        EXPECT_NEAR(s.leftover, -25.0 / 266.0, 1e-14) << label;
        EXPECT_EQ(s.sums, 3) << label;
    });
    for_each_hand_layout(3.0, 2, 0.0, [](const auto& in, const auto& fields, auto label) {
        const direct_solve s = solve_directly(in, fields, solver::conjugate_gradient);
        EXPECT_EQ(s.outcome.iterations, 2) << label;
        EXPECT_FALSE(s.outcome.converged) << label;
        EXPECT_EQ(s.outcome.restarts, 1) << label;
        EXPECT_NEAR(s.leftover, -25.0 / 266.0, 1e-14) << label;
        EXPECT_EQ(s.sums, 3) << label;
    });
}

TEST(Acceptance, WithoutABoundTheTestFormsNothingAndFinalFormsOnce) {
    bool_field solid(4);
    std::vector<double> masked(4);
    std::vector<double> terms(4);
    acceptance accept(0.5, {}, solid.span(), std::nullopt, masked, terms);
    const std::vector<double> residual{1.0, -0.25, 2.0, 0.25};
    EXPECT_TRUE(accept.test(residual, 1));
    EXPECT_EQ(accept.sums(), 0);
    accept.final(residual, 1);
    EXPECT_EQ(accept.sums(), 1);
    EXPECT_EQ(accept.leftover(), 0.5 * 3.0);
    accept.final(residual, 1);
    EXPECT_EQ(accept.sums(), 1);
}

TEST(Acceptance, WithABoundTheTestFormsTheLeftoverAndFinalDoesNotFormItAgain) {
    bool_field solid(4);
    std::vector<double> masked(4);
    std::vector<double> terms(4);
    const std::vector<double> residual{1.0, -0.25, 2.0, 0.25};
    // dt 0.5 times the sum 3: a leftover of 1.5, at the bound and inside it.
    acceptance at(0.5, {}, solid.span(), 1.5, masked, terms);
    EXPECT_TRUE(at.test(residual, 1));
    acceptance beyond(0.5, {}, solid.span(), 1.4999, masked, terms);
    EXPECT_FALSE(beyond.test(residual, 1));
    EXPECT_EQ(beyond.sums(), 1);
    beyond.final(residual, 1);
    EXPECT_EQ(beyond.sums(), 1);
    // A new residual is formed again.
    const std::vector<double> next{0.5, 0.0, 0.0, 0.0};
    beyond.final(next, 2);
    EXPECT_EQ(beyond.sums(), 2);
    EXPECT_EQ(beyond.leftover(), 0.25);
}

TEST(Acceptance, APerCellStepIsReadAsZeroInSolidCells) {
    // An infinite step in a SOLID cell, whose residual is zero: inf * 0 would
    // be NaN, and the masked step gives 1 * 1 + 0 * 0 + 2 * 1 = 3.
    bool_field solid(3);
    solid[1] = true;
    const std::vector<double> dt{1.0, std::numeric_limits<double>::infinity(), 2.0};
    std::vector<double> masked(3);
    std::vector<double> terms(3);
    acceptance accept(0.0, dt, solid.span(), std::nullopt, masked, terms);
    const std::vector<double> residual{1.0, 0.0, 1.0};
    accept.final(residual, 1);
    EXPECT_EQ(accept.leftover(), 3.0);
}

}  // namespace
