// Positions in a layout (positions.hpp): the index of a cell or a face and back,
// what the turbulence model and the wall functions read one layout's row at
// another layout's place by. The layout is the core's two- and one-direction
// ones; the three-direction program (test_three_directions.cpp) tests the same
// functions at D = 3.

#include "cfdcore/positions.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <set>

namespace cfdcore::positions {
namespace {

using layout::layout;

TEST(Positions, AnIndexAndItsPositionAreEachOthersInverse) {
    const layout<2> cells({5, 3});
    std::set<std::size_t> seen;
    for (std::size_t i = 0; i < cells.cell_count(); ++i) {
        const position<2> at = position_of(cells, i);
        EXPECT_LT(at[0], 5u);
        EXPECT_LT(at[1], 3u);
        EXPECT_EQ(index_of(cells, at), i);
        seen.insert(index_of(cells, at));
    }
    EXPECT_EQ(seen.size(), cells.cell_count());
}

TEST(Positions, DirectionZeroIsTheFastest) {
    const layout<2> cells({5, 3});
    const position<2> at = position_of(cells, 7);
    EXPECT_EQ(at[0], 2u);
    EXPECT_EQ(at[1], 1u);
    EXPECT_EQ(index_of(cells, above(at, 0)), 8u);
    EXPECT_EQ(index_of(cells, above(at, 1)), 12u);
    EXPECT_EQ(index_of(cells, below(at, 0)), 6u);
    EXPECT_EQ(index_of(cells, below(at, 1)), 2u);
}

TEST(Positions, AFaceLayoutHasOneMoreAlongItsDirectionsAndTheCornersOneMoreAlongTwo) {
    const layout<2> cells({5, 3});
    const layout<2> u_faces = faces_layout(cells, 0, 0);
    EXPECT_EQ(u_faces.cells(0), 6u);
    EXPECT_EQ(u_faces.cells(1), 3u);
    const layout<2> v_faces = faces_layout(cells, 1, 1);
    EXPECT_EQ(v_faces.cells(0), 5u);
    EXPECT_EQ(v_faces.cells(1), 4u);
    // A pair's corners, where the faces of the two directions cross.
    const layout<2> corners = faces_layout(cells, 0, 1);
    EXPECT_EQ(corners.cells(0), 6u);
    EXPECT_EQ(corners.cells(1), 4u);
    EXPECT_EQ(corners.cell_count(), 24u);
    // The pair is unordered.
    const layout<2> flipped = faces_layout(cells, 1, 0);
    EXPECT_EQ(flipped.cells(0), 6u);
    EXPECT_EQ(flipped.cells(1), 4u);
    // A face of the line's own direction in one direction.
    const layout<1> line({7});
    EXPECT_EQ(faces_layout(line, 0, 0).cells(0), 8u);
}

// The index into an array over every direction but one: an entry of a domain edge
// normal to that direction.
TEST(Positions, AnEdgesEntryIsIndexedWithoutItsNormalDirection) {
    const layout<2> cells({5, 3});
    // Normal to direction 0 the edge runs along direction 1, normal to 1 along 0.
    EXPECT_EQ(index_without(cells, position<2>{{4, 2}}, 0), 2u);
    EXPECT_EQ(index_without(cells, position<2>{{4, 2}}, 1), 4u);
    std::set<std::size_t> low_edge;
    for (std::size_t i = 0; i < cells.cell_count(); ++i) {
        const position<2> at = position_of(cells, i);
        if (at[1] == 0) {
            low_edge.insert(index_without(cells, at, 1));
        }
    }
    EXPECT_EQ(low_edge.size(), 5u);
    EXPECT_EQ(*low_edge.begin(), 0u);
    EXPECT_EQ(*low_edge.rbegin(), 4u);
}

TEST(Positions, ADirectionOfOneCellIsAllowed) {
    const layout<2> cells({1, 4});
    EXPECT_EQ(position_of(cells, 3)[0], 0u);
    EXPECT_EQ(position_of(cells, 3)[1], 3u);
    EXPECT_EQ(index_without(cells, position_of(cells, 3), 1), 0u);
    EXPECT_EQ(index_without(cells, position_of(cells, 3), 0), 3u);
}

}  // namespace
}  // namespace cfdcore::positions
