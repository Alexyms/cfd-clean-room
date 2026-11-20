// The loop style of the core's kernels (prompt 89; CLAUDE.md, C++ Coding
// Standards): what a kernel is written with so that the compiler turns a row
// of cells into vector instructions under the project's flags, and so that
// every value stays the reference's bit for bit.
//
// A kernel walks its field by rows, the lines along direction 0, which are
// contiguous in memory (layout.hpp). The arithmetic of a row is written once,
// as a block function: a template on a cell count N whose loop runs exactly N
// times. It is called with N = `width` over the row's whole blocks and with
// N = 1 over the cells left, so both run the same expressions. GCC at -O2
// vectorises a loop only when its trip count is a constant, which a block of
// `width` cells gives it and a loop to a row's length does not.
//
// A block function reads its inputs through a struct of pointers and writes
// through parameters qualified CFDCORE_RESTRICT: a promise that no input
// overlaps an output, without which the compiler must assume every store
// changes the next load and keeps the loop scalar. The caller owns that
// promise.
//
// A block function has no branch on a cell's data. A condition is a `mask`,
// all ones or all zeros per cell, and a choice between two values is `select`
// or `keep`: bit operations, which round nothing. A SOLID or held cell is
// computed like any other and masked. A reduction a stop reads (whether every
// cell is within a bound) is the OR of the cells' masks, exact in any order.
//
// An expression several block functions share (a stencil's sum, say) is a
// function of the operands and an index, marked CFDCORE_INLINE. The mark is
// not a hint: GCC inlines an unmarked helper after it has decided which
// structs to split into registers, the operands then stay in memory, and the
// block stays scalar.
//
// Each cell's operations are the reference's, in the reference's order, one
// rounding each (-ffp-contract=off): a block changes how many cells an
// instruction carries, never what a cell computes.

#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>

// `restrict` as the compiler spells it: the pointer is the only way the
// function reaches what it points to.
#if defined(__GNUC__) || defined(__clang__)
#define CFDCORE_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define CFDCORE_RESTRICT __restrict
#else
#define CFDCORE_RESTRICT
#endif

// A function inlined into its caller whatever the compiler would choose.
#if defined(__GNUC__) || defined(__clang__)
#define CFDCORE_INLINE [[gnu::always_inline]] inline
#elif defined(_MSC_VER)
#define CFDCORE_INLINE __forceinline
#else
#define CFDCORE_INLINE inline
#endif

namespace cfdcore::lanes {

// The cells of one block. Eight doubles are one 512-bit vector, two 256-bit
// or four 128-bit ones, so a block is a whole number of vectors at every
// width the compiler may be given.
inline constexpr std::size_t width = 8;

// A condition on one cell: every bit set where it holds, none where it does not.
using mask = std::uint64_t;
inline constexpr mask all = ~mask{0};
inline constexpr mask none = mask{0};

// The mask of a condition.
CFDCORE_INLINE mask mask_of(bool condition) { return condition ? all : none; }

// `value` where `m` is set and +0.0 where it is not: the reference's
// `x[cells] = 0.0`. No arithmetic: the bits of `value` pass or do not.
CFDCORE_INLINE double keep(mask m, double value) {
    return std::bit_cast<double>(std::bit_cast<mask>(value) & m);
}

// `if_set` where `m` is set, `otherwise` where it is not: the reference's
// `np.where`. No arithmetic: a NaN's payload and a zero's sign pass unchanged.
CFDCORE_INLINE double select(mask m, double if_set, double otherwise) {
    return std::bit_cast<double>((std::bit_cast<mask>(if_set) & m) |
                                 (std::bit_cast<mask>(otherwise) & ~m));
}

// The masks of a bool cell field's complement: `out[i]` is set where
// `flags[i]` is false. An empty `flags` sets every `out[i]`.
//
// flags: one bool per cell, or empty for none set
// out:   output, one mask per cell; its length is the cell count
inline void masks_where_not(std::span<const bool> flags, std::span<mask> out) {
    if (flags.empty()) {
        for (mask& m : out) {
            m = all;
        }
        return;
    }
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = mask_of(!flags[i]);
    }
}

}  // namespace cfdcore::lanes
