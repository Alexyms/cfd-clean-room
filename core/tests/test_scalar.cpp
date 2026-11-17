// The scalar module's own tests (ADR-017 D): results known without the
// reference. The comparison with scalar_scheme.py is tests/test_core_parity.py's.

#include "cfdcore/grid.hpp"
#include "cfdcore/scalar.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
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

}  // namespace
