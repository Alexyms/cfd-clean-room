// The machine's rates behind prompt 89's floor estimate: what a Jacobi sweep
// and a flux call could cost here if nothing but the arithmetic and the memory
// traffic were paid for.
//
//   g++ -std=c++20 -O2 -ffp-contract=off floor89.cpp -o floor89 && ./floor89
//
// Built by `probe89.py floor`, which keeps the JSON this prints. Every kernel
// is written with intrinsics at a stated vector width, so the rate is that
// width's and not what a compiler makes of a loop: 2 doubles (SSE2, what the
// project's flags allow), 4 (AVX2) and 8 (AVX-512), the wider two behind
// target attributes so the file builds without -march. Each measurement is
// the best and the median of repeated timings.
//
// - clock:    a dependent chain x = x * a + c in 64-bit integers, four cycles
//             a step on Zen 4 and Zen 5 (a multiply's latency of 3 and an
//             add's of 1), so the time per step over four is a cycle.
// - divide:   out[i] = a[i] / b[i] on arrays that fit the first-level cache.
// - arithmetic: the sweep's residual without its division,
//             r = (b + ((g1 x1 + g2 x2) + g3 x3) + g4 x4) - a x, nine
//             operations, on arrays that fit the first-level cache.
// - sweep:    the whole sweep per cell (the residual, its mask, the stop's
//             test, the update with its division and mask) on flat arrays,
//             once at a size that fits the first-level cache and once at
//             320x120 cells, where the arrays total more than the second-level
//             cache. No row ends, no boundaries: an ideal kernel.
// - stream:   a[i] = b[i] + c[i] over arrays of 320x120 cells, as many arrays
//             as the sweep reads and writes, the bytes moved per second.

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

struct timing {
    double best;
    double median;
};

// The best and median time of one call of `fn`, in seconds, over `repeats` calls.
template <typename F>
timing time_of(F&& fn, int repeats) {
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeats));
    for (int i = 0; i < 3; ++i) {
        fn();
    }
    for (int i = 0; i < repeats; ++i) {
        const auto start = clock_type::now();
        fn();
        const auto stop = clock_type::now();
        times.push_back(std::chrono::duration<double>(stop - start).count());
    }
    std::sort(times.begin(), times.end());
    return {times.front(), times[times.size() / 2]};
}

// Storage aligned to 64 bytes, as a cache line and the widest vector are.
template <typename T>
struct aligned_array {
    explicit aligned_array(std::size_t n, T value)
        : size(n), data(static_cast<T*>(std::aligned_alloc(64, ((n * sizeof(T) + 63) / 64) * 64))) {
        std::fill(data, data + n, value);
    }
    ~aligned_array() { std::free(data); }
    aligned_array(const aligned_array&) = delete;
    aligned_array& operator=(const aligned_array&) = delete;
    std::size_t size;
    T* data;
};

std::uint64_t sink = 0;

std::uint64_t chain(std::uint64_t steps) {
    std::uint64_t x = 1;
    for (std::uint64_t i = 0; i < steps; ++i) {
        x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    }
    return x;
}

// The arrays of one sweep over n cells with a row length of `stride`: the
// iterate carries `stride` cells of margin at each end for its neighbours.
struct sweep_arrays {
    sweep_arrays(std::size_t n, std::size_t stride_)
        : cells(n),
          stride(stride_),
          gu(n + 1, 0.125),
          gv(n + stride_, 0.25),
          x(n + 2 * stride_, 1.5),
          b(n, 3.0),
          ap(n, 2.0),
          keep(n, ~std::uint64_t{0}),
          r(n, 0.0),
          next(n, 0.0) {
        for (std::size_t i = 0; i < n; ++i) {
            x.data[i + stride_] = 1.0 + 1e-3 * static_cast<double>(i % 97);
            ap.data[i] = 2.0 + 1e-3 * static_cast<double>(i % 89);
            if (i % 11 == 0) {
                keep.data[i] = 0;
            }
        }
    }
    std::size_t cells;
    std::size_t stride;
    aligned_array<double> gu;
    aligned_array<double> gv;
    aligned_array<double> x;
    aligned_array<double> b;
    aligned_array<double> ap;
    aligned_array<std::uint64_t> keep;
    aligned_array<double> r;
    aligned_array<double> next;
};

// One sweep at each width. The operations and their order are the core's:
// east, west, north, south; (b + sum) - a_P x; the mask; |r| <= stop; the
// update x + r / a_P and its mask. Returns whether any cell fails the stop.
bool sweep_sse2(sweep_arrays& s, double stop_value) {
    const double* x = s.x.data + s.stride;
    const __m128d stop = _mm_set1_pd(stop_value);
    const __m128d magnitude = _mm_castsi128_pd(_mm_set1_epi64x(0x7fffffffffffffffLL));
    __m128d outside = _mm_setzero_pd();
    for (std::size_t i = 0; i < s.cells; i += 2) {
        const __m128d xi = _mm_loadu_pd(x + i);
        __m128d sum = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(s.gu.data + i + 1), _mm_loadu_pd(x + i + 1)),
                                 _mm_mul_pd(_mm_loadu_pd(s.gu.data + i), _mm_loadu_pd(x + i - 1)));
        sum = _mm_add_pd(sum, _mm_mul_pd(_mm_loadu_pd(s.gv.data + i + s.stride),
                                         _mm_loadu_pd(x + i + s.stride)));
        sum = _mm_add_pd(sum, _mm_mul_pd(_mm_loadu_pd(s.gv.data + i), _mm_loadu_pd(x + i - s.stride)));
        const __m128d ap = _mm_load_pd(s.ap.data + i);
        const __m128d keep = _mm_castsi128_pd(
            _mm_load_si128(reinterpret_cast<const __m128i*>(s.keep.data + i)));
        __m128d r = _mm_sub_pd(_mm_add_pd(_mm_load_pd(s.b.data + i), sum), _mm_mul_pd(ap, xi));
        r = _mm_and_pd(r, keep);
        outside = _mm_or_pd(outside, _mm_cmpnle_pd(_mm_and_pd(r, magnitude), stop));
        _mm_store_pd(s.r.data + i, r);
        _mm_store_pd(s.next.data + i, _mm_and_pd(_mm_add_pd(xi, _mm_div_pd(r, ap)), keep));
    }
    return _mm_movemask_pd(outside) != 0;
}

__attribute__((target("avx2"))) bool sweep_avx2(sweep_arrays& s, double stop_value) {
    const double* x = s.x.data + s.stride;
    const __m256d stop = _mm256_set1_pd(stop_value);
    const __m256d magnitude = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7fffffffffffffffLL));
    __m256d outside = _mm256_setzero_pd();
    for (std::size_t i = 0; i < s.cells; i += 4) {
        const __m256d xi = _mm256_loadu_pd(x + i);
        __m256d sum =
            _mm256_add_pd(_mm256_mul_pd(_mm256_loadu_pd(s.gu.data + i + 1), _mm256_loadu_pd(x + i + 1)),
                          _mm256_mul_pd(_mm256_loadu_pd(s.gu.data + i), _mm256_loadu_pd(x + i - 1)));
        sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_loadu_pd(s.gv.data + i + s.stride),
                                               _mm256_loadu_pd(x + i + s.stride)));
        sum = _mm256_add_pd(
            sum, _mm256_mul_pd(_mm256_loadu_pd(s.gv.data + i), _mm256_loadu_pd(x + i - s.stride)));
        const __m256d ap = _mm256_load_pd(s.ap.data + i);
        const __m256d keep = _mm256_castsi256_pd(
            _mm256_load_si256(reinterpret_cast<const __m256i*>(s.keep.data + i)));
        __m256d r =
            _mm256_sub_pd(_mm256_add_pd(_mm256_load_pd(s.b.data + i), sum), _mm256_mul_pd(ap, xi));
        r = _mm256_and_pd(r, keep);
        outside = _mm256_or_pd(outside, _mm256_cmp_pd(_mm256_and_pd(r, magnitude), stop, _CMP_NLE_UQ));
        _mm256_store_pd(s.r.data + i, r);
        _mm256_store_pd(s.next.data + i,
                        _mm256_and_pd(_mm256_add_pd(xi, _mm256_div_pd(r, ap)), keep));
    }
    return _mm256_movemask_pd(outside) != 0;
}

__attribute__((target("avx512f,avx512dq"))) bool sweep_avx512(sweep_arrays& s, double stop_value) {
    const double* x = s.x.data + s.stride;
    const __m512d stop = _mm512_set1_pd(stop_value);
    __mmask8 outside = 0;
    for (std::size_t i = 0; i < s.cells; i += 8) {
        const __m512d xi = _mm512_loadu_pd(x + i);
        __m512d sum =
            _mm512_add_pd(_mm512_mul_pd(_mm512_loadu_pd(s.gu.data + i + 1), _mm512_loadu_pd(x + i + 1)),
                          _mm512_mul_pd(_mm512_loadu_pd(s.gu.data + i), _mm512_loadu_pd(x + i - 1)));
        sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_loadu_pd(s.gv.data + i + s.stride),
                                               _mm512_loadu_pd(x + i + s.stride)));
        sum = _mm512_add_pd(
            sum, _mm512_mul_pd(_mm512_loadu_pd(s.gv.data + i), _mm512_loadu_pd(x + i - s.stride)));
        const __m512d ap = _mm512_load_pd(s.ap.data + i);
        const __m512d keep = _mm512_castsi512_pd(_mm512_load_si512(s.keep.data + i));
        __m512d r =
            _mm512_sub_pd(_mm512_add_pd(_mm512_load_pd(s.b.data + i), sum), _mm512_mul_pd(ap, xi));
        r = _mm512_and_pd(r, keep);
        outside = static_cast<__mmask8>(
            outside | _mm512_cmp_pd_mask(_mm512_abs_pd(r), stop, _CMP_NLE_UQ));
        _mm512_store_pd(s.r.data + i, r);
        _mm512_store_pd(s.next.data + i,
                        _mm512_and_pd(_mm512_add_pd(xi, _mm512_div_pd(r, ap)), keep));
    }
    return outside != 0;
}

// The residual without the division and the masks: nine operations a cell.
void arithmetic_sse2(sweep_arrays& s) {
    const double* x = s.x.data + s.stride;
    for (std::size_t i = 0; i < s.cells; i += 2) {
        const __m128d xi = _mm_loadu_pd(x + i);
        __m128d sum = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(s.gu.data + i + 1), _mm_loadu_pd(x + i + 1)),
                                 _mm_mul_pd(_mm_loadu_pd(s.gu.data + i), _mm_loadu_pd(x + i - 1)));
        sum = _mm_add_pd(sum, _mm_mul_pd(_mm_loadu_pd(s.gv.data + i + s.stride),
                                         _mm_loadu_pd(x + i + s.stride)));
        sum = _mm_add_pd(sum, _mm_mul_pd(_mm_loadu_pd(s.gv.data + i), _mm_loadu_pd(x + i - s.stride)));
        _mm_store_pd(s.r.data + i, _mm_sub_pd(_mm_add_pd(_mm_load_pd(s.b.data + i), sum),
                                              _mm_mul_pd(_mm_load_pd(s.ap.data + i), xi)));
    }
}

__attribute__((target("avx2"))) void arithmetic_avx2(sweep_arrays& s) {
    const double* x = s.x.data + s.stride;
    for (std::size_t i = 0; i < s.cells; i += 4) {
        const __m256d xi = _mm256_loadu_pd(x + i);
        __m256d sum =
            _mm256_add_pd(_mm256_mul_pd(_mm256_loadu_pd(s.gu.data + i + 1), _mm256_loadu_pd(x + i + 1)),
                          _mm256_mul_pd(_mm256_loadu_pd(s.gu.data + i), _mm256_loadu_pd(x + i - 1)));
        sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_loadu_pd(s.gv.data + i + s.stride),
                                               _mm256_loadu_pd(x + i + s.stride)));
        sum = _mm256_add_pd(
            sum, _mm256_mul_pd(_mm256_loadu_pd(s.gv.data + i), _mm256_loadu_pd(x + i - s.stride)));
        _mm256_store_pd(s.r.data + i,
                        _mm256_sub_pd(_mm256_add_pd(_mm256_load_pd(s.b.data + i), sum),
                                      _mm256_mul_pd(_mm256_load_pd(s.ap.data + i), xi)));
    }
}

__attribute__((target("avx512f"))) void arithmetic_avx512(sweep_arrays& s) {
    const double* x = s.x.data + s.stride;
    for (std::size_t i = 0; i < s.cells; i += 8) {
        const __m512d xi = _mm512_loadu_pd(x + i);
        __m512d sum =
            _mm512_add_pd(_mm512_mul_pd(_mm512_loadu_pd(s.gu.data + i + 1), _mm512_loadu_pd(x + i + 1)),
                          _mm512_mul_pd(_mm512_loadu_pd(s.gu.data + i), _mm512_loadu_pd(x + i - 1)));
        sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_loadu_pd(s.gv.data + i + s.stride),
                                               _mm512_loadu_pd(x + i + s.stride)));
        sum = _mm512_add_pd(
            sum, _mm512_mul_pd(_mm512_loadu_pd(s.gv.data + i), _mm512_loadu_pd(x + i - s.stride)));
        _mm512_store_pd(s.r.data + i,
                        _mm512_sub_pd(_mm512_add_pd(_mm512_load_pd(s.b.data + i), sum),
                                      _mm512_mul_pd(_mm512_load_pd(s.ap.data + i), xi)));
    }
}

void divide_scalar(const double* a, const double* b, double* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = _mm_cvtsd_f64(_mm_div_sd(_mm_load_sd(a + i), _mm_load_sd(b + i)));
    }
}

void divide_sse2(const double* a, const double* b, double* out, std::size_t n) {
    for (std::size_t i = 0; i < n; i += 2) {
        _mm_store_pd(out + i, _mm_div_pd(_mm_load_pd(a + i), _mm_load_pd(b + i)));
    }
}

__attribute__((target("avx2"))) void divide_avx2(const double* a, const double* b, double* out,
                                                 std::size_t n) {
    for (std::size_t i = 0; i < n; i += 4) {
        _mm256_store_pd(out + i, _mm256_div_pd(_mm256_load_pd(a + i), _mm256_load_pd(b + i)));
    }
}

__attribute__((target("avx512f"))) void divide_avx512(const double* a, const double* b, double* out,
                                                      std::size_t n) {
    for (std::size_t i = 0; i < n; i += 8) {
        _mm512_store_pd(out + i, _mm512_div_pd(_mm512_load_pd(a + i), _mm512_load_pd(b + i)));
    }
}

// a[i] = b[i] + c[i] over `streams` arrays in turn: each is read once and
// written once per pass, two doubles at a time.
void stream_pass(std::vector<std::unique_ptr<aligned_array<double>>>& arrays, std::size_t n) {
    const std::size_t count = arrays.size();
    for (std::size_t k = 0; k + 2 < count; k += 3) {
        double* a = arrays[k]->data;
        const double* b = arrays[k + 1]->data;
        const double* c = arrays[k + 2]->data;
        for (std::size_t i = 0; i < n; i += 2) {
            _mm_store_pd(a + i, _mm_add_pd(_mm_load_pd(b + i), _mm_load_pd(c + i)));
        }
    }
}

void print_timing(const char* name, timing t, double per, bool last = false) {
    std::printf("  \"%s\": {\"best_ns\": %.5f, \"median_ns\": %.5f}%s\n", name, t.best * 1e9 / per,
                t.median * 1e9 / per, last ? "" : ",");
}

}  // namespace

int main() {
    constexpr std::size_t grid_cells = 320 * 120;
    constexpr std::size_t grid_stride = 320;
    constexpr std::size_t small_cells = 256;
    constexpr std::size_t small_stride = 16;
    constexpr int small_passes = 400;
    const bool avx2 = __builtin_cpu_supports("avx2");
    const bool avx512 = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq");

    std::printf("{\n");
    std::printf("  \"grid_cells\": %zu,\n  \"small_cells\": %zu,\n", grid_cells, small_cells);
    std::printf("  \"avx2\": %s,\n  \"avx512\": %s,\n", avx2 ? "true" : "false",
                avx512 ? "true" : "false");

    constexpr std::uint64_t steps = 1ULL << 24;
    print_timing("chain_step", time_of([&] { sink += chain(steps); }, 21),
                 static_cast<double>(steps));

    {
        aligned_array<double> a(small_cells, 3.0);
        aligned_array<double> b(small_cells, 1.7);
        aligned_array<double> out(small_cells, 0.0);
        const double per = static_cast<double>(small_cells) * small_passes;
        const auto repeat = [&](auto&& kernel) {
            return time_of(
                [&] {
                    for (int p = 0; p < small_passes; ++p) {
                        kernel(a.data, b.data, out.data, small_cells);
                    }
                },
                201);
        };
        print_timing("divide_scalar", repeat(divide_scalar), per);
        print_timing("divide_sse2", repeat(divide_sse2), per);
        if (avx2) {
            print_timing("divide_avx2", repeat(divide_avx2), per);
        }
        if (avx512) {
            print_timing("divide_avx512", repeat(divide_avx512), per);
        }
    }

    {
        sweep_arrays small(small_cells, small_stride);
        const double per = static_cast<double>(small_cells) * small_passes;
        const auto repeat = [&](auto&& kernel) {
            return time_of(
                [&] {
                    for (int p = 0; p < small_passes; ++p) {
                        kernel(small);
                    }
                },
                201);
        };
        print_timing("arithmetic_sse2", repeat(arithmetic_sse2), per);
        if (avx2) {
            print_timing("arithmetic_avx2", repeat(arithmetic_avx2), per);
        }
        if (avx512) {
            print_timing("arithmetic_avx512", repeat(arithmetic_avx512), per);
        }
        bool any = false;
        const auto sweep = [&](auto&& kernel) {
            return time_of(
                [&] {
                    for (int p = 0; p < small_passes; ++p) {
                        any = kernel(small, 1e300) || any;
                    }
                },
                201);
        };
        print_timing("sweep_small_sse2", sweep(sweep_sse2), per);
        if (avx2) {
            print_timing("sweep_small_avx2", sweep(sweep_avx2), per);
        }
        if (avx512) {
            print_timing("sweep_small_avx512", sweep(sweep_avx512), per);
        }
        sink += any ? 1 : 0;
    }

    {
        sweep_arrays grid(grid_cells, grid_stride);
        const double per = static_cast<double>(grid_cells);
        bool any = false;
        print_timing("sweep_grid_sse2", time_of([&] { any = sweep_sse2(grid, 1e300) || any; }, 2001),
                     per);
        if (avx2) {
            print_timing("sweep_grid_avx2",
                         time_of([&] { any = sweep_avx2(grid, 1e300) || any; }, 2001), per);
        }
        if (avx512) {
            print_timing("sweep_grid_avx512",
                         time_of([&] { any = sweep_avx512(grid, 1e300) || any; }, 2001), per);
        }
        sink += any ? 1 : 0;
    }

    {
        // Nine arrays, as the sweep's eight and one more: three triads a pass.
        std::vector<std::unique_ptr<aligned_array<double>>> arrays;
        for (int k = 0; k < 9; ++k) {
            arrays.push_back(std::make_unique<aligned_array<double>>(grid_cells, 1.0 + k));
        }
        // Each triad reads two arrays and writes one: 24 bytes a cell, three triads.
        const double bytes = 3.0 * 24.0 * static_cast<double>(grid_cells);
        print_timing("stream_byte", time_of([&] { stream_pass(arrays, grid_cells); }, 2001), bytes,
                     true);
    }
    std::printf("}\n");
    return sink == 0 ? 1 : 0;
}
