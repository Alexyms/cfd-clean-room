// The largest value over the non-SOLID cells (walls.cpp, `largest_cells`) in its two
// spellings, the ternary the source has and `lanes::select`, on a 320 x 120 field, as
// the block runs it. The two are timed in alternation and the least of 40 rounds of
// 200 calls is printed. Built from the checkout's root with the project's flags:
//
//   g++ -Icore/include -std=c++20 -O2 -ffp-contract=off -march=x86-64-v3 -DNDEBUG
//       docs/reports/probe94/bench_max.cpp -o bench_max
//
// Prompt 94b: GCC 13.3 vectorises the ternary (a maximum) and not the select (control
// flow in the loop), and the select form takes about 2.4 times as long.
#include "cfdcore/lanes.hpp"

#include <chrono>
#include <cstdio>
#include <vector>

using cfdcore::lanes::mask;
namespace lanes = cfdcore::lanes;

template <std::size_t N>
CFDCORE_INLINE mask largest_ternary(lanes::cells<N>, const double* q, const mask* fluid,
                                    double lowest, std::size_t first,
                                    double* CFDCORE_RESTRICT largest) {
    mask nan = lanes::none;
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double value = lanes::select(fluid[i], q[i], lowest);
        nan |= lanes::mask_of(value != value);
        largest[k] = value > largest[k] ? value : largest[k];
    }
    return nan;
}

template <std::size_t N>
CFDCORE_INLINE mask largest_select(lanes::cells<N>, const double* q, const mask* fluid,
                                   double lowest, std::size_t first,
                                   double* CFDCORE_RESTRICT largest) {
    mask nan = lanes::none;
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double value = lanes::select(fluid[i], q[i], lowest);
        nan |= lanes::mask_of(value != value);
        largest[k] = lanes::select(lanes::mask_of(value > largest[k]), value, largest[k]);
    }
    return nan;
}

template <bool Select>
double largest_of(const std::vector<double>& q, const std::vector<mask>& fluid) {
    const double lowest = -1e300;
    double largest[lanes::width];
    for (double& x : largest) {
        x = lowest;
    }
    mask nan = lanes::none;
    lanes::for_each_block(0, q.size(), [&](auto cells, std::size_t m) {
        if constexpr (Select) {
            nan |= largest_select(cells, q.data(), fluid.data(), lowest, m, largest);
        } else {
            nan |= largest_ternary(cells, q.data(), fluid.data(), lowest, m, largest);
        }
    });
    double best = lowest;
    for (const double lane : largest) {
        best = lane > best ? lane : best;
    }
    return nan != lanes::none ? -1.0 : best;
}

int main() {
    const std::size_t n = 320 * 120;
    std::vector<double> q(n);
    std::vector<mask> fluid(n, lanes::all);
    for (std::size_t i = 0; i < n; ++i) {
        q[i] = static_cast<double>((i * 2654435761u) % 1000003u) * 1e-6;
        if (i % 17 == 0) {
            fluid[i] = lanes::none;
        }
    }
    double sink = 0.0;
    double best[2] = {1e9, 1e9};
    for (int round = 0; round < 40; ++round) {
        for (int form = 0; form < 2; ++form) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int rep = 0; rep < 200; ++rep) {
                q[rep] += 1e-9;
                sink += form == 0 ? largest_of<false>(q, fluid) : largest_of<true>(q, fluid);
            }
            const auto t1 = std::chrono::steady_clock::now();
            const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 200;
            if (us < best[form]) {
                best[form] = us;
            }
        }
    }
    std::printf("ternary %.2f us, select %.2f us per call on %zu cells (sink %g)\n", best[0],
                best[1], n, sink);
}
