// The reduce module's own tests (ADR-017 D): results known without the
// reference. The comparison with np.sum is tests/test_core_parity.py's.

#include "cfdcore/reduce.hpp"

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <span>
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

}  // namespace
