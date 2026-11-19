// The grid and layout modules' own tests (ADR-017 D): the indexing, at D = 1
// and D = 2, against the multi-index it stands for. The comparison with the
// reference's `mesh_axes` is tests/test_core_parity.py's.

#include "cfdcore/grid.hpp"
#include "cfdcore/layout.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

namespace {

using cfdcore::grid::grid;
using cfdcore::grid::grid_input;
using cfdcore::layout::layout;

TEST(Layout, LineOfCells) {
    layout<1> fields({7});
    EXPECT_EQ(fields.cell_count(), 7u);
    EXPECT_EQ(fields.face_count(0), 8u);
    EXPECT_EQ(fields.cell_stride(0), 1u);
    ASSERT_EQ(fields.lines(0).size(), 1u);
    EXPECT_EQ(fields.lines(0)[0].cell, 0u);
    EXPECT_EQ(fields.lines(0)[0].face, 0u);
    for (std::size_t i = 0; i < 7; ++i) {
        EXPECT_EQ(fields.low_face(0, i), i);
    }
}

TEST(Layout, RowMajorStridesAndFaceFields) {
    // [ny, nx] = [3, 5]: the u faces [3, 6] and the v faces [4, 5].
    layout<2> fields({5, 3});
    EXPECT_EQ(fields.cell_count(), 15u);
    EXPECT_EQ(fields.cell_stride(0), 1u);
    EXPECT_EQ(fields.cell_stride(1), 5u);
    EXPECT_EQ(fields.face_count(0), 18u);
    EXPECT_EQ(fields.face_count(1), 20u);
    EXPECT_EQ(fields.face_stride(0), 1u);
    EXPECT_EQ(fields.face_stride(1), 5u);
    EXPECT_EQ(fields.cell_field().size(), 15u);
    EXPECT_EQ(fields.face_field(0).size(), 18u);
    EXPECT_EQ(fields.face_field(1).size(), 20u);
}

TEST(Layout, EveryCellsFacesAreItsMultiIndexsFaces) {
    // Cell (j, i) has u faces (j, i) and (j, i + 1) in [ny, nx + 1], and v
    // faces (j, i) and (j + 1, i) in [ny + 1, nx].
    const std::size_t nx = 6;
    const std::size_t ny = 4;
    layout<2> fields({nx, ny});
    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t i = 0; i < nx; ++i) {
            const std::size_t cell = j * nx + i;
            EXPECT_EQ(fields.low_face(0, cell), j * (nx + 1) + i);
            EXPECT_EQ(fields.low_face(0, cell) + fields.face_stride(0), j * (nx + 1) + i + 1);
            EXPECT_EQ(fields.low_face(1, cell), j * nx + i);
            EXPECT_EQ(fields.low_face(1, cell) + fields.face_stride(1), (j + 1) * nx + i);
        }
    }
}

TEST(Layout, LinesCoverEveryCellAndFaceOnce) {
    const std::size_t nx = 6;
    const std::size_t ny = 4;
    layout<2> fields({nx, ny});
    for (int d = 0; d < 2; ++d) {
        std::vector<int> cell_seen(fields.cell_count(), 0);
        std::vector<int> face_seen(fields.face_count(d), 0);
        for (const auto& line : fields.lines(d)) {
            for (std::size_t m = 0; m < fields.cells(d); ++m) {
                ++cell_seen[line.cell + m * fields.cell_stride(d)];
            }
            for (std::size_t m = 0; m <= fields.cells(d); ++m) {
                ++face_seen[line.face + m * fields.face_stride(d)];
            }
        }
        for (int seen : cell_seen) {
            EXPECT_EQ(seen, 1) << d;
        }
        for (int seen : face_seen) {
            EXPECT_EQ(seen, 1) << d;
        }
    }
}

TEST(Layout, LinesRunInTheTransposedCallsOrder) {
    // The x lines are the rows j = 0..ny-1; the y lines the columns
    // i = 0..nx-1, the rows of the reference's transposed field.
    const std::size_t nx = 6;
    const std::size_t ny = 4;
    layout<2> fields({nx, ny});
    ASSERT_EQ(fields.lines(0).size(), ny);
    ASSERT_EQ(fields.lines(1).size(), nx);
    for (std::size_t j = 0; j < ny; ++j) {
        EXPECT_EQ(fields.lines(0)[j].cell, j * nx);
        EXPECT_EQ(fields.lines(0)[j].face, j * (nx + 1));
    }
    for (std::size_t i = 0; i < nx; ++i) {
        EXPECT_EQ(fields.lines(1)[i].cell, i);
        EXPECT_EQ(fields.lines(1)[i].face, i);
    }
}

TEST(Layout, OneCellAlongADirection) {
    layout<2> fields({1, 3});
    EXPECT_EQ(fields.face_count(0), 6u);
    EXPECT_EQ(fields.lines(0).size(), 3u);
    EXPECT_EQ(fields.lines(1).size(), 1u);
    EXPECT_EQ(fields.low_face(0, 2), 4u);
}

struct grid_arrays {
    std::vector<double> x{0.0, 0.5, 1.5, 3.0};
    std::vector<double> xc{0.25, 1.0, 2.25};
    std::vector<double> y{0.0, 1.0, 3.0};
    std::vector<double> yc{0.5, 2.0};
    std::vector<std::int32_t> cell_type{2, 1, 2, 2, 0, 1};

    grid_input<2> input() const {
        return {.faces = {x, y}, .centres = {xc, yc}, .cell_type = cell_type, .solid = 1};
    }
};

TEST(Grid, NodesAreTheBoundaryFacesAroundTheCentres) {
    grid_arrays a;
    ASSERT_TRUE(grid<2>::check(a.input()).empty());
    grid<2> g(a.input());
    EXPECT_EQ(g.cells(0), 3u);
    EXPECT_EQ(g.cells(1), 2u);
    const std::vector<double> nodes_x{0.0, 0.25, 1.0, 2.25, 3.0};
    const std::vector<double> nodes_y{0.0, 0.5, 2.0, 3.0};
    EXPECT_EQ(std::vector<double>(g.nodes(0).begin(), g.nodes(0).end()), nodes_x);
    EXPECT_EQ(std::vector<double>(g.nodes(1).begin(), g.nodes(1).end()), nodes_y);
    const std::vector<double> faces_x{0.5, 1.5};
    const std::vector<double> faces_y{1.0};
    EXPECT_EQ(std::vector<double>(g.interior_faces(0).begin(), g.interior_faces(0).end()), faces_x);
    EXPECT_EQ(std::vector<double>(g.interior_faces(1).begin(), g.interior_faces(1).end()), faces_y);
}

TEST(Grid, SolidIsTheHandedCode) {
    grid_arrays a;
    grid<2> g(a.input());
    const std::array<bool, 6> solid{false, true, false, false, false, true};
    for (std::size_t cell = 0; cell < 6; ++cell) {
        EXPECT_EQ(g.solid(cell), solid[cell]) << cell;
        EXPECT_EQ(g.cell_types()[cell], a.cell_type[cell]);
    }
    auto other = a.input();
    other.solid = 2;
    grid<2> g2(other);
    EXPECT_TRUE(g2.solid(0));
    EXPECT_FALSE(g2.solid(1));
}

TEST(Grid, TheGridCopiesItsInputs) {
    grid_arrays a;
    grid<2> g(a.input());
    a.x[0] = -9.0;
    a.cell_type[1] = 0;
    EXPECT_EQ(g.nodes(0)[0], 0.0);
    EXPECT_TRUE(g.solid(1));
}

TEST(Grid, ALineOfCells) {
    std::vector<double> x{0.0, 1.0, 3.0};
    std::vector<double> xc{0.5, 2.0};
    std::vector<std::int32_t> types{0, 1};
    grid_input<1> input{.faces = {x}, .centres = {xc}, .cell_type = types, .solid = 1};
    ASSERT_TRUE(grid<1>::check(input).empty());
    grid<1> g(input);
    EXPECT_EQ(g.nodes(0).size(), 4u);
    EXPECT_EQ(g.interior_faces(0).size(), 1u);
    EXPECT_EQ(g.interior_faces(0)[0], 1.0);
    EXPECT_TRUE(g.solid(1));
}

TEST(Grid, CheckRefusesInconsistentArrays) {
    grid_arrays a;
    auto no_cells = a.input();
    std::vector<double> none;
    std::vector<double> one{0.0};
    no_cells.centres[1] = none;
    no_cells.faces[1] = one;
    EXPECT_EQ(grid<2>::check(no_cells), "a direction has no cell");

    auto short_faces = a.input();
    std::vector<double> three{0.0, 1.0, 2.0};
    short_faces.faces[0] = three;
    EXPECT_EQ(grid<2>::check(short_faces), "a direction's faces are not one more than its centres");

    auto few_types = a.input();
    std::vector<std::int32_t> five(5, 0);
    few_types.cell_type = five;
    EXPECT_EQ(grid<2>::check(few_types), "cell_type does not hold one type per cell");
}

TEST(Grid, ConsistentCoordinatesHaveNoBadDirection) {
    grid_arrays a;
    EXPECT_EQ(grid<2>::first_bad_direction(a.input()), -1);
    // One cell along a direction: the nodes are the two faces and the centre.
    std::vector<double> x{0.0, 2.0};
    std::vector<double> xc{1.0};
    std::vector<std::int32_t> types{0};
    grid_input<1> one{.faces = {x}, .centres = {xc}, .cell_type = types, .solid = 1};
    EXPECT_EQ(grid<1>::first_bad_direction(one), -1);
}

TEST(Grid, EachKindOfBadCoordinateNamesItsDirection) {
    using arrays = std::vector<double>;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    // Each change breaks one thing along a direction, given its faces and
    // centres: a face that repeats or decreases, a NaN face, an infinite end
    // face (which keeps the order, so only the finiteness test refuses it), a
    // centre that repeats, decreases or is NaN, a centre at or beyond an end face.
    struct bad_case {
        const char* what;
        std::function<void(arrays& faces, arrays& centres)> change;
    };
    const bad_case cases[] = {
        {"a face repeats", [](arrays& f, arrays&) { f[1] = f[0]; }},
        {"a face decreases", [](arrays& f, arrays&) { f[1] = f[0] - 1.0; }},
        {"a NaN face", [&](arrays& f, arrays&) { f[1] = nan; }},
        {"an infinite last face", [&](arrays& f, arrays&) { f.back() = inf; }},
        {"an infinite first face", [&](arrays& f, arrays&) { f.front() = -inf; }},
        {"a centre repeats", [](arrays&, arrays& c) { c[1] = c[0]; }},
        {"a centre decreases", [](arrays&, arrays& c) { c[1] = c[0] - 0.1; }},
        {"a NaN centre", [&](arrays&, arrays& c) { c[0] = nan; }},
        {"a centre at the low face", [](arrays& f, arrays& c) { c[0] = f[0]; }},
        {"a centre at the high face", [](arrays& f, arrays& c) { c.back() = f.back(); }},
        {"a centre below the low face", [](arrays& f, arrays& c) { c[0] = f[0] - 1.0; }},
        {"a centre past the high face", [](arrays& f, arrays& c) { c.back() = f.back() + 1.0; }},
    };
    for (int d = 0; d < 2; ++d) {
        for (const auto& bad : cases) {
            grid_arrays a;
            bad.change(d == 0 ? a.x : a.y, d == 0 ? a.xc : a.yc);
            EXPECT_TRUE(grid<2>::check(a.input()).empty()) << bad.what << " " << d;
            EXPECT_EQ(grid<2>::first_bad_direction(a.input()), d) << bad.what << " " << d;
        }
    }
}

TEST(Grid, TheFirstBadDirectionIsTheLowest) {
    grid_arrays a;
    a.x[1] = 5.0;
    a.y[1] = -1.0;
    EXPECT_EQ(grid<2>::first_bad_direction(a.input()), 0);
}

}  // namespace
