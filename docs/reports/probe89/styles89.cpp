// Prompt 89's loop styles: one Jacobi sweep's arithmetic written six ways and
// timed on a 320x120 field, to choose how the core's kernels are written.
//
//   g++ -std=c++20 -O2 -ffp-contract=off styles89.cpp -o styles89 && ./styles89
//
// Built and run by `probe89.py styles` under each flag set; it keeps the JSON
// this prints. Every style forms each cell's values by the same operations in
// the same order (the sum east, west, north, south; the residual; its mask;
// the stop's test; the update and its mask), so the checksums agree, and only
// how the loop is written differs:
//
// - plain:       one loop to the row's length, each value stored as it is formed
// - lambda:      blocks of eight cells, the cell's arithmetic and its stores in
//                a lambda the block walker calls
// - lane_arrays: blocks of eight cells formed into local arrays, then stored
// - simd:        std::experimental::simd at the target's native width
// - blocks:      the style kept (core/include/cfdcore/lanes.hpp): a block
//                function templated on its cell count, the inputs in a struct,
//                the outputs restrict-qualified
// - intrinsics:  SSE2 intrinsics, two cells an instruction: what the project's
//                flags allow at best
//
// The field is synthetic and has no row ends or boundary rows: the cells 1 to
// nx - 2 of every row are swept, each row with its own pointers.

#include <emmintrin.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <experimental/simd>
#include <vector>

namespace {

constexpr std::size_t width = 8;
using mask = std::uint64_t;

[[gnu::always_inline]] inline double keep(mask m, double value) {
    return std::bit_cast<double>(std::bit_cast<mask>(value) & m);
}
[[gnu::always_inline]] inline mask mask_of(bool condition) {
    return condition ? ~mask{0} : mask{0};
}

// One row's operands, each pointer at the row's first cell.
struct row {
    std::size_t n;
    const double* g;
    const double* g_north;
    const double* g_south;
    const double* x;
    const double* x_north;
    const double* x_south;
    const double* b;
    const double* a_p;
    const mask* free;
    const mask* fluid;
    double* r;
    double* next;
};

struct cell_values {
    double r;
    double next;
    mask outside;
};

[[gnu::always_inline]] inline cell_values cell(const row& w, std::size_t m, double stop) {
    double sum = w.g[m + 1] * w.x[m + 1] + w.g[m] * w.x[m - 1];
    sum = sum + w.g_north[m] * w.x_north[m];
    sum = sum + w.g_south[m] * w.x_south[m];
    const double ri = keep(w.free[m], (w.b[m] + sum) - w.a_p[m] * w.x[m]);
    return {ri, keep(w.fluid[m], w.x[m] + ri / w.a_p[m]), mask_of(!(std::fabs(ri) <= stop))};
}

mask sweep_plain(const row w, double stop) {
    mask outside = 0;
    for (std::size_t m = 1; m + 1 < w.n; ++m) {
        const cell_values v = cell(w, m, stop);
        w.r[m] = v.r;
        w.next[m] = v.next;
        outside |= v.outside;
    }
    return outside;
}

template <typename Cell>
inline void for_each_cell(std::size_t first, std::size_t last, Cell&& each) {
    std::size_t m = first;
    for (; m + width <= last; m += width) {
        for (std::size_t k = 0; k < width; ++k) {
            each(m + k);
        }
    }
    for (; m < last; ++m) {
        each(m);
    }
}

mask sweep_lambda(const row w, double stop) {
    mask outside = 0;
    for_each_cell(1, w.n - 1, [&](std::size_t m) {
        const cell_values v = cell(w, m, stop);
        w.r[m] = v.r;
        w.next[m] = v.next;
        outside |= v.outside;
    });
    return outside;
}

mask sweep_lane_arrays(const row w, double stop) {
    mask outside = 0;
    std::size_t m = 1;
    const std::size_t last = w.n - 1;
    for (; m + width <= last; m += width) {
        double r[width];
        double next[width];
        for (std::size_t k = 0; k < width; ++k) {
            const cell_values v = cell(w, m + k, stop);
            r[k] = v.r;
            next[k] = v.next;
            outside |= v.outside;
        }
        for (std::size_t k = 0; k < width; ++k) {
            w.r[m + k] = r[k];
        }
        for (std::size_t k = 0; k < width; ++k) {
            w.next[m + k] = next[k];
        }
    }
    for (; m < last; ++m) {
        const cell_values v = cell(w, m, stop);
        w.r[m] = v.r;
        w.next[m] = v.next;
        outside |= v.outside;
    }
    return outside;
}

namespace stdx = std::experimental;

template <typename V>
inline bool cells_simd(const row& w, std::size_t m, double stop) {
    using M = typename V::mask_type;
    using I = stdx::rebind_simd_t<mask, V>;
    const auto load = [](const double* p) { return V(p, stdx::element_aligned); };
    const V x = load(w.x + m);
    V sum = load(w.g + m + 1) * load(w.x + m + 1) + load(w.g + m) * load(w.x + m - 1);
    sum = sum + load(w.g_north + m) * load(w.x_north + m);
    sum = sum + load(w.g_south + m) * load(w.x_south + m);
    const V a_p = load(w.a_p + m);
    V r = (load(w.b + m) + sum) - a_p * x;
    const M fixed = stdx::__proposed::static_simd_cast<M>(I(w.free + m, stdx::element_aligned) == 0);
    const M solid = stdx::__proposed::static_simd_cast<M>(I(w.fluid + m, stdx::element_aligned) == 0);
    stdx::where(fixed, r) = 0.0;
    r.copy_to(w.r + m, stdx::element_aligned);
    V next = x + r / a_p;
    stdx::where(solid, next) = 0.0;
    next.copy_to(w.next + m, stdx::element_aligned);
    return stdx::any_of(!(stdx::abs(r) <= stop));
}

mask sweep_simd(const row w, double stop) {
    using V = stdx::native_simd<double>;
    using S = stdx::simd<double, stdx::simd_abi::scalar>;
    bool outside = false;
    std::size_t m = 1;
    const std::size_t last = w.n - 1;
    for (; m + V::size() <= last; m += V::size()) {
        outside |= cells_simd<V>(w, m, stop);
    }
    for (; m < last; ++m) {
        outside |= cells_simd<S>(w, m, stop);
    }
    return outside ? 1 : 0;
}

template <std::size_t N>
inline mask cells_block(const row& w, std::size_t first, double stop, double* __restrict__ r,
                        double* __restrict__ next) {
    mask outside = 0;
    for (std::size_t k = 0; k < N; ++k) {
        const cell_values v = cell(w, first + k, stop);
        r[first + k] = v.r;
        next[first + k] = v.next;
        outside |= v.outside;
    }
    return outside;
}

mask sweep_blocks(const row w, double stop) {
    mask outside = 0;
    std::size_t m = 1;
    const std::size_t last = w.n - 1;
    for (; m + width <= last; m += width) {
        outside |= cells_block<width>(w, m, stop, w.r, w.next);
    }
    for (; m < last; ++m) {
        outside |= cells_block<1>(w, m, stop, w.r, w.next);
    }
    return outside;
}

mask sweep_intrinsics(const row w, double stop_value) {
    const __m128d stop = _mm_set1_pd(stop_value);
    const __m128d magnitude = _mm_castsi128_pd(_mm_set1_epi64x(0x7fffffffffffffffLL));
    __m128d outside = _mm_setzero_pd();
    const auto masks = [](const mask* p) {
        return _mm_castsi128_pd(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
    };
    std::size_t m = 1;
    const std::size_t last = w.n - 1;
    for (; m + 2 <= last; m += 2) {
        const __m128d x = _mm_loadu_pd(w.x + m);
        __m128d sum = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(w.g + m + 1), _mm_loadu_pd(w.x + m + 1)),
                                 _mm_mul_pd(_mm_loadu_pd(w.g + m), _mm_loadu_pd(w.x + m - 1)));
        sum = _mm_add_pd(sum, _mm_mul_pd(_mm_loadu_pd(w.g_north + m), _mm_loadu_pd(w.x_north + m)));
        sum = _mm_add_pd(sum, _mm_mul_pd(_mm_loadu_pd(w.g_south + m), _mm_loadu_pd(w.x_south + m)));
        const __m128d a_p = _mm_loadu_pd(w.a_p + m);
        __m128d r = _mm_sub_pd(_mm_add_pd(_mm_loadu_pd(w.b + m), sum), _mm_mul_pd(a_p, x));
        r = _mm_and_pd(r, masks(w.free + m));
        outside = _mm_or_pd(outside, _mm_cmpnle_pd(_mm_and_pd(r, magnitude), stop));
        _mm_storeu_pd(w.r + m, r);
        _mm_storeu_pd(w.next + m,
                      _mm_and_pd(_mm_add_pd(x, _mm_div_pd(r, a_p)), masks(w.fluid + m)));
    }
    mask any = _mm_movemask_pd(outside) != 0 ? ~mask{0} : 0;
    for (; m < last; ++m) {
        any |= cells_block<1>(w, m, stop_value, w.r, w.next);
    }
    return any;
}

}  // namespace

int main() {
    constexpr std::size_t nx = 320;
    constexpr std::size_t ny = 120;
    constexpr std::size_t n = nx * ny;
    constexpr int repeats = 2000;
    std::vector<double> g(n + ny + 8, 0.125);
    std::vector<double> g_v(n + nx + 8, 0.25);
    std::vector<double> x(n + 2 * nx + 8, 1.5);
    std::vector<double> b(n, 3.0);
    std::vector<double> a_p(n, 2.0);
    std::vector<double> r(n, 0.0);
    std::vector<double> next(n, 0.0);
    std::vector<mask> free(n, ~mask{0});
    std::vector<mask> fluid(n, ~mask{0});
    for (std::size_t i = 0; i < n; ++i) {
        x[i + nx] = 1.0 + 1e-3 * static_cast<double>(i % 97);
        a_p[i] = 2.0 + 1e-3 * static_cast<double>(i % 89);
        if (i % 11 == 0) {
            free[i] = 0;
            fluid[i] = 0;
        }
    }
    bool first = true;
    std::printf("{\n");
    const auto run = [&](const char* name, auto&& sweep) {
        std::vector<double> times;
        mask outside = 0;
        for (int repeat = 0; repeat < repeats + 3; ++repeat) {
            const auto start = std::chrono::steady_clock::now();
            for (std::size_t j = 0; j < ny; ++j) {
                const row w{nx,
                            g.data() + j * (nx + 1),
                            g_v.data() + (j + 1) * nx,
                            g_v.data() + j * nx,
                            x.data() + nx + j * nx,
                            x.data() + nx + (j + 1) * nx,
                            x.data() + j * nx,
                            b.data() + j * nx,
                            a_p.data() + j * nx,
                            free.data() + j * nx,
                            fluid.data() + j * nx,
                            r.data() + j * nx,
                            next.data() + j * nx};
                outside |= sweep(w, 1e300);
            }
            const auto stop = std::chrono::steady_clock::now();
            if (repeat >= 3) {
                times.push_back(std::chrono::duration<double>(stop - start).count());
            }
        }
        std::sort(times.begin(), times.end());
        double checksum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            checksum += r[i] + next[i];
        }
        std::printf("%s  \"%s\": {\"best_us\": %.2f, \"median_us\": %.2f, \"checksum\": %.17g, "
                    "\"outside\": %d}",
                    first ? "" : ",\n", name, times.front() * 1e6, times[times.size() / 2] * 1e6,
                    checksum, static_cast<int>(outside != 0));
        first = false;
    };
    run("plain", sweep_plain);
    run("lambda", sweep_lambda);
    run("lane_arrays", sweep_lane_arrays);
    run("simd", sweep_simd);
    run("blocks", sweep_blocks);
    run("intrinsics", sweep_intrinsics);
    std::printf("\n}\n");
    return 0;
}
