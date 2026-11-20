// The reduce module's own tests (ADR-017 D): results known without the
// reference. The comparison with np.sum is tests/test_core_parity.py's.

#include "cfdcore/reduce.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using cfdcore::reduce::pairwise_sum;

// The unit in the last place of 1.0, and half of it.
constexpr double ulp_one = 0x1p-52;
constexpr double half_ulp_one = 0x1p-53;

std::uint64_t bits(double x) { return std::bit_cast<std::uint64_t>(x); }

// A plain left-to-right sum, the order the pairwise sum is set against.
double sequential(const std::vector<double>& values) {
    double sum = 0.0;
    for (double v : values) {
        sum += v;
    }
    return sum;
}

TEST(PairwiseSum, EmptyIsPositiveZero) {
    std::vector<double> none;
    EXPECT_EQ(bits(pairwise_sum(none)), bits(0.0));
}

TEST(PairwiseSum, OneElementIsItself) {
    for (double v : {1.5, -2.25e-300, 7.0e300, 0.0}) {
        std::vector<double> one{v};
        EXPECT_EQ(bits(pairwise_sum(one)), bits(v)) << v;
    }
}

TEST(PairwiseSum, NegativeZerosSumToPositiveZero) {
    // NumPy adds the sum to its identity 0.0, so -0.0 does not survive.
    for (std::size_t n : {1u, 7u, 8u, 129u}) {
        std::vector<double> zeros(n, -0.0);
        EXPECT_EQ(bits(pairwise_sum(zeros)), bits(0.0)) << n;
    }
}

TEST(PairwiseSum, IntegersSumExactlyAtEveryLength) {
    // Every partial sum is an integer below 2^53, so every order gives the
    // exact value; this checks that no term is dropped or counted twice.
    for (std::size_t n = 0; n <= 1000; ++n) {
        std::vector<double> values(n);
        for (std::size_t i = 0; i < n; ++i) {
            values[i] = static_cast<double>(i + 1) * ((i % 3 == 0) ? -1.0 : 1.0);
        }
        double exact = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            exact += values[i];
        }
        EXPECT_EQ(pairwise_sum(values), exact) << n;
    }
}

TEST(PairwiseSum, EightAccumulatorsKeepTermsBelowTheLastPlace) {
    // A one and fifteen terms of half its last place: left to right each
    // addition ties and rounds back to one; the eight accumulators pair the
    // small terms first, giving 1 + 7 ulp (np.sum's value).
    std::vector<double> values(16, half_ulp_one);
    values[0] = 1.0;
    EXPECT_EQ(sequential(values), 1.0);
    EXPECT_EQ(pairwise_sum(values), 1.0 + 7.0 * ulp_one);
}

TEST(PairwiseSum, EightTermsAreCombinedNotAddedInOrder) {
    // A one and seven terms of half its last place. Eight terms are the
    // smallest array the accumulators take, one term each; a plain sum is for
    // fewer than eight. Left to right every addition ties and rounds back to
    // one. The accumulators combine pairwise: r0 + r1 ties back to one, and
    // the other six terms make 3 ulp between them: 1 + 3 ulp (np.sum's value).
    std::vector<double> values(8, half_ulp_one);
    values[0] = 1.0;
    EXPECT_EQ(sequential(values), 1.0);
    EXPECT_EQ(pairwise_sum(values), 1.0 + 3.0 * ulp_one);
}

TEST(PairwiseSum, ABlockOfExactly128TermsIsNotSplit) {
    // A one and 127 terms of half its last place: 128 terms are one block of
    // eight accumulators, 16 terms each. The one's accumulator rounds every
    // tie back to one and the other seven hold 8 ulp each, giving 1 + 56 ulp
    // (np.sum's value). Split into 64 and 64, as blocks of 64 or a block that
    // ends below 128 terms would, the halves give 1 + 28 ulp and 32 ulp:
    // 1 + 60 ulp.
    std::vector<double> values(128, half_ulp_one);
    values[0] = 1.0;
    EXPECT_EQ(pairwise_sum(values), 1.0 + 56.0 * ulp_one);
    std::span<const double> all(values);
    double halves = pairwise_sum(all.first(64)) + pairwise_sum(all.subspan(64));
    EXPECT_EQ(halves, 1.0 + 60.0 * ulp_one);
}

TEST(PairwiseSum, BlocksOf128NotOf64SplitAboveAndBelowThem) {
    // A one and 191 terms of half its last place: 192 terms split into 96 and
    // 96, each a block of 128 or fewer. The first 96 give 1 + 42 ulp (12
    // terms to an accumulator: 6 ulp in each but the one's) and the last 96
    // give 48 ulp exactly: 1 + 90 ulp, np.sum's value. With blocks of 64 each
    // half splits again, the first 96 into two 48s (1 + 21 ulp and 24 ulp)
    // and the last into 24 ulp and 24 ulp: 1 + 93 ulp.
    std::vector<double> values(192, half_ulp_one);
    values[0] = 1.0;
    EXPECT_EQ(pairwise_sum(values), 1.0 + 90.0 * ulp_one);
}

TEST(PairwiseSum, TermsBeyondTheLastBlockAreAddedAfter) {
    // Ten terms: the eight accumulators give 1 + 3 ulp, and the two left over
    // are added one at a time, each a tie rounding to the even 1 + 4 ulp.
    std::vector<double> values(10, half_ulp_one);
    values[0] = 1.0;
    EXPECT_EQ(pairwise_sum(values), 1.0 + 4.0 * ulp_one);
}

TEST(PairwiseSum, AboveABlockSplitsAtHalfRoundedDownToEight) {
    // A one and 135 terms of half its last place: 136 terms split into 64 and
    // 72 (68 rounded down to a multiple of 8). The first 64 give 1 + 28 ulp
    // (the one's accumulator rounds back to one, the other seven hold 4 ulp
    // each), the last 72 give 36 ulp exactly: 1 + 64 ulp, np.sum's value. A
    // split at 68 would leave four terms after the last block of the first
    // half, each a tie rounding back, and give 1 + 62 ulp.
    std::vector<double> values(136, half_ulp_one);
    values[0] = 1.0;
    EXPECT_EQ(pairwise_sum(values), 1.0 + 64.0 * ulp_one);
    std::span<const double> all(values);
    double halves = pairwise_sum(all.first(64)) + pairwise_sum(all.subspan(64));
    EXPECT_EQ(bits(pairwise_sum(all)), bits(halves));
}

// ---- The blocked dot product (the second kind, ADR-017 G) -------------------

using cfdcore::reduce::blocked_dot;

TEST(BlockedDot, EmptyIsPositiveZero) {
    std::vector<double> none;
    EXPECT_EQ(bits(blocked_dot(none, none, 4)), bits(0.0));
}

TEST(BlockedDot, OneBlockIsThePairwiseSumOfTheProducts) {
    // A block as long as the arrays: the products summed in pairwise_sum's order.
    std::vector<double> a(300);
    std::vector<double> b(300);
    std::vector<double> products(300);
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = 1.0 + static_cast<double>(i) * 0x1p-20;
        b[i] = (i % 3 == 0 ? -1.0 : 1.0) * (0.5 + static_cast<double>(i) * 0x1p-11);
        products[i] = a[i] * b[i];
    }
    EXPECT_EQ(bits(blocked_dot(a, b, 300)), bits(pairwise_sum(products)));
    EXPECT_EQ(bits(blocked_dot(a, b, 1000)), bits(pairwise_sum(products)));
}

TEST(BlockedDot, BlocksAreSummedApartAndAddedInOrder) {
    // [2^53, 1, 1, 1, 1, 1, 1, 1] against ones. In blocks of 4 the first block
    // is 2^53 (each 2^53 + 1 is a tie that rounds back to the even 2^53) and
    // the second 4, added to 0.0 in order: 2^53 + 4. As one block of 8 the
    // eight accumulators hold a term each and give
    // ((2^53 + 1) + (1 + 1)) + ((1 + 1) + (1 + 1)) = (2^53 + 2) + 4 = 2^53 + 6.
    const double big = 0x1p53;
    std::vector<double> a{big, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<double> ones(a.size(), 1.0);
    EXPECT_EQ(blocked_dot(a, ones, 4), big + 4.0);
    EXPECT_EQ(blocked_dot(a, ones, 8), big + 6.0);
    EXPECT_EQ(bits(blocked_dot(a, ones, 8)), bits(pairwise_sum(a)));
}

TEST(BlockedDot, ALastShortBlockHoldsWhatRemains) {
    // Seven terms in blocks of 3: [0, 1, 2], [3, 4, 5], [6]; integers, so exact.
    std::vector<double> a{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    std::vector<double> b{7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    EXPECT_EQ(blocked_dot(a, b, 3), 84.0);
    EXPECT_EQ(blocked_dot(a, b, 1), 84.0);
}

// What blocked_dot refuses with, or an empty string when it returns.
std::string refusal(std::span<const double> a, std::span<const double> b, std::size_t block) {
    try {
        blocked_dot(a, b, block);
    } catch (const std::invalid_argument& error) {
        return error.what();
    }
    return "";
}

TEST(BlockedDot, ABlockOfZeroIsRefusedAtTheCall) {
    // With a zero block the loop never advances, so a non-empty array would
    // hang the caller; an empty one is refused too, since a block is at least 1.
    std::vector<double> a{1.0, 2.0, 3.0};
    std::vector<double> b{4.0, 5.0, 6.0};
    EXPECT_THROW(blocked_dot(a, b, 0), std::invalid_argument);
    std::vector<double> none;
    EXPECT_THROW(blocked_dot(none, none, 0), std::invalid_argument);
    EXPECT_NE(refusal(a, b, 0).find("block length"), std::string::npos);
    // A block of 1 is the least that is accepted.
    EXPECT_EQ(blocked_dot(a, b, 1), 32.0);
}

TEST(BlockedDot, ArraysOfDifferentLengthsAreRefusedWhicheverIsShorter) {
    std::vector<double> three{1.0, 2.0, 3.0};
    std::vector<double> two{4.0, 5.0};
    EXPECT_THROW(blocked_dot(three, two, 2), std::invalid_argument);
    EXPECT_THROW(blocked_dot(two, three, 2), std::invalid_argument);
    std::vector<double> none;
    EXPECT_THROW(blocked_dot(none, two, 2), std::invalid_argument);
    EXPECT_NE(refusal(three, two, 2).find("differ in length"), std::string::npos);
    // The block is checked first: a zero block names itself, not the lengths.
    EXPECT_NE(refusal(three, two, 0).find("block length"), std::string::npos);
    // Equal lengths that are not a whole number of blocks are not refused.
    EXPECT_EQ(refusal(three, three, 2), "");
}

using cfdcore::reduce::block_dot;

TEST(BlockDot, TheBlocksAddedInOrderAreBlockedDot) {
    // Lengths below, at and above a block of 128, and a block length that
    // leaves a short last block: each block's term added to 0.0 in block
    // order is blocked_dot's value bit for bit.
    for (const std::size_t n : {std::size_t{1}, std::size_t{7}, std::size_t{128},
                                std::size_t{300}, std::size_t{1000}}) {
        std::vector<double> a(n);
        std::vector<double> b(n);
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = 1.0 + 1e-7 * static_cast<double>((i * 37) % 101);
            b[i] = (i % 3 == 0 ? -1.0 : 1.0) / static_cast<double>(i + 1);
        }
        for (const std::size_t block : {std::size_t{1}, std::size_t{9}, std::size_t{128},
                                        std::size_t{320}}) {
            double total = 0.0;
            for (std::size_t first = 0; first < n; first += block) {
                const std::size_t count = std::min(block, n - first);
                total += block_dot(std::span<const double>(a).subspan(first, count),
                                   std::span<const double>(b).subspan(first, count));
            }
            EXPECT_EQ(total, blocked_dot(a, b, block)) << n << " terms, blocks of " << block;
        }
    }
}

TEST(BlockDot, EmptyIsPositiveZero) {
    std::vector<double> none;
    EXPECT_EQ(block_dot(none, none), 0.0);
    EXPECT_FALSE(std::signbit(block_dot(none, none)));
}

TEST(BlockDot, ArraysOfDifferentLengthsAreRefused) {
    std::vector<double> three{1.0, 2.0, 3.0};
    std::vector<double> two{4.0, 5.0};
    EXPECT_THROW(block_dot(three, two), std::invalid_argument);
    EXPECT_THROW(block_dot(two, three), std::invalid_argument);
}

}  // namespace
