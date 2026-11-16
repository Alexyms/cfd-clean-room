#include "cfdcore/reduce.hpp"

#include <cstddef>

namespace cfdcore::reduce {

namespace {

// NumPy's block size for the pairwise sum (PW_BLOCKSIZE in its loops).
constexpr std::size_t block_size = 128;
// The accumulators NumPy unrolls a block into.
constexpr std::size_t unroll = 8;

// The pairwise sum of values[0, n), in NumPy's order (pairwise_sum in its
// loops_utils.h). The recursion's depth is log2(n / 128), so it cannot grow
// beyond about fifty for any array memory can hold.
double pairwise(const double* values, std::size_t n) {
    if (n < unroll) {
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            sum += values[i];
        }
        return sum;
    }
    if (n <= block_size) {
        double r[unroll];
        for (std::size_t j = 0; j < unroll; ++j) {
            r[j] = values[j];
        }
        std::size_t i = unroll;
        for (; i < n - (n % unroll); i += unroll) {
            for (std::size_t j = 0; j < unroll; ++j) {
                r[j] += values[i + j];
            }
        }
        double sum = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) {
            sum += values[i];
        }
        return sum;
    }
    std::size_t half = n / 2;
    half -= half % unroll;
    return pairwise(values, half) + pairwise(values + half, n - half);
}

}  // namespace

double pairwise_sum(std::span<const double> values) {
    // np.add.reduce starts from the identity, 0.0, and adds the pairwise sum
    // of every term to it in one call of its inner loop.
    return 0.0 + pairwise(values.data(), values.size());
}

}  // namespace cfdcore::reduce
