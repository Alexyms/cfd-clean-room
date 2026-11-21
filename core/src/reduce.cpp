#include "cfdcore/reduce.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace cfdcore::reduce {

namespace {

// NumPy's block size for the pairwise sum (PW_BLOCKSIZE in its loops).
constexpr std::size_t block_size = 128;
// The accumulators NumPy unrolls a block into.
constexpr std::size_t unroll = 8;

// The pairwise sum of term(0) .. term(n - 1), in NumPy's order (pairwise_sum
// in its loops_utils.h). `term` gives one term by its index from the start of
// this subrange plus `first`. The recursion's depth is log2(n / 128), so it
// cannot grow beyond about fifty for any array memory can hold.
template <typename Term>
double pairwise(const Term& term, std::size_t first, std::size_t n) {
    if (n < unroll) {
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            sum += term(first + i);
        }
        return sum;
    }
    if (n <= block_size) {
        double r[unroll];
        for (std::size_t j = 0; j < unroll; ++j) {
            r[j] = term(first + j);
        }
        std::size_t i = unroll;
        for (; i < n - (n % unroll); i += unroll) {
            for (std::size_t j = 0; j < unroll; ++j) {
                r[j] += term(first + i + j);
            }
        }
        double sum = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) {
            sum += term(first + i);
        }
        return sum;
    }
    std::size_t half = n / 2;
    half -= half % unroll;
    return pairwise(term, first, half) + pairwise(term, first + half, n - half);
}

}  // namespace

double pairwise_sum(std::span<const double> values) {
    // np.add.reduce starts from the identity, 0.0, and adds the pairwise sum
    // of every term to it in one call of its inner loop.
    const double* v = values.data();
    return 0.0 + pairwise([v](std::size_t i) { return v[i]; }, 0, values.size());
}

double buffered_sum(std::span<const double> values, std::size_t row) {
    if (values.empty()) {
        return 0.0;
    }
    if (row == 0 || values.size() % row != 0) {
        throw std::invalid_argument("buffered_sum: the terms are not a whole number of rows");
    }
    // A slice of one column is one strided line to NumPy: no buffer.
    if (row == 1) {
        return pairwise_sum(values);
    }
    const double* v = values.data();
    const auto term = [v](std::size_t i) { return v[i]; };
    // Each fill of the buffer holds as many whole rows as fit, and at least one.
    const std::size_t fill = std::max<std::size_t>(1, buffer_size / row) * row;
    double total = 0.0;
    for (std::size_t first = 0; first < values.size(); first += fill) {
        total += pairwise(term, first, std::min(fill, values.size() - first));
    }
    return total;
}

double blocked_dot(std::span<const double> a, std::span<const double> b, std::size_t block) {
    // A block of zero never advances the loop below, and a shorter b is read
    // past its end; neither is a result, so both are refused at the call.
    if (block == 0) {
        throw std::invalid_argument("blocked_dot: the block length is at least 1");
    }
    if (a.size() != b.size()) {
        throw std::invalid_argument("blocked_dot: the two arrays differ in length");
    }
    const double* pa = a.data();
    const double* pb = b.data();
    const auto product = [pa, pb](std::size_t i) { return pa[i] * pb[i]; };
    double total = 0.0;
    for (std::size_t first = 0; first < a.size(); first += block) {
        total += pairwise(product, first, std::min(block, a.size() - first));
    }
    return total;
}

double block_dot(std::span<const double> a, std::span<const double> b) {
    // A shorter b would be read past its end.
    if (a.size() != b.size()) {
        throw std::invalid_argument("block_dot: the two arrays differ in length");
    }
    const double* pa = a.data();
    const double* pb = b.data();
    return pairwise([pa, pb](std::size_t i) { return pa[i] * pb[i]; }, 0, a.size());
}

}  // namespace cfdcore::reduce
