// The loop style's masks and selects (lanes.hpp): each passes or drops a
// value's bits and computes nothing, which is what keeps a masked kernel the
// reference's bit for bit.

#include "cfdcore/lanes.hpp"

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

using cfdcore::lanes::all;
using cfdcore::lanes::keep;
using cfdcore::lanes::mask;
using cfdcore::lanes::mask_of;
using cfdcore::lanes::masks_where_not;
using cfdcore::lanes::none;
using cfdcore::lanes::select;

std::uint64_t bits(double value) { return std::bit_cast<std::uint64_t>(value); }

// Values whose bits a multiplication by one or an addition of zero would
// change, or that compare equal with different bits.
std::vector<double> awkward_values() {
    return {0.0,
            -0.0,
            1.5,
            -2.25,
            std::numeric_limits<double>::denorm_min(),
            -std::numeric_limits<double>::denorm_min(),
            std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::quiet_NaN(),
            -std::numeric_limits<double>::quiet_NaN(),
            std::bit_cast<double>(std::uint64_t{0x7ff8000000001234})};
}

TEST(Lanes, AMaskIsAllOnesOrAllZeros) {
    EXPECT_EQ(mask_of(true), ~std::uint64_t{0});
    EXPECT_EQ(mask_of(false), std::uint64_t{0});
    EXPECT_EQ(all, mask_of(true));
    EXPECT_EQ(none, mask_of(false));
}

TEST(Lanes, KeepPassesEveryBitWhereTheMaskIsSet) {
    for (const double value : awkward_values()) {
        EXPECT_EQ(bits(keep(all, value)), bits(value));
    }
}

TEST(Lanes, KeepGivesPositiveZeroWhereTheMaskIsNotSet) {
    // The reference's `x[cells] = 0.0`: +0.0 whatever the value was, a
    // negative one, a NaN or an infinity included.
    for (const double value : awkward_values()) {
        EXPECT_EQ(bits(keep(none, value)), bits(0.0));
    }
}

TEST(Lanes, SelectPassesTheChosenValuesBits) {
    for (const double a : awkward_values()) {
        for (const double b : awkward_values()) {
            EXPECT_EQ(bits(select(all, a, b)), bits(a));
            EXPECT_EQ(bits(select(none, a, b)), bits(b));
        }
    }
}

TEST(Lanes, MasksWhereNotComplementTheFlags) {
    const bool flags[5] = {false, true, true, false, true};
    std::vector<mask> out(5, std::uint64_t{0x1234});
    masks_where_not(flags, out);
    EXPECT_EQ(out, (std::vector<mask>{all, none, none, all, none}));
}

TEST(Lanes, MasksWhereNotWithoutFlagsSetEveryCell) {
    std::vector<mask> out(4, none);
    masks_where_not({}, out);
    EXPECT_EQ(out, std::vector<mask>(4, all));
}

TEST(Lanes, ABlockIsAWholeNumberOfVectorsAtEveryWidth) {
    // 128, 256 and 512 bits hold two, four and eight doubles.
    EXPECT_EQ(cfdcore::lanes::width % 2, 0u);
    EXPECT_EQ(cfdcore::lanes::width % 4, 0u);
    EXPECT_EQ(cfdcore::lanes::width % 8, 0u);
}

}  // namespace
