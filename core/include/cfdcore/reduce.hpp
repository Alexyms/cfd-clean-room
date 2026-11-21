// The core's reductions (ADR-017 C, module `reduce`; ADR-017 G).
//
// Two kinds (ADR-017 G). A sum that a stop, a refusal or a budget reads is
// formed in the reference's order, so that the core's value equals NumPy's
// bit for bit given the same array: `pairwise_sum`, NumPy's pairwise sum.
// Every other reduction is formed in fixed blocks combined in block order, so
// that it does not depend on how many threads compute the blocks:
// `blocked_dot`, the conjugate gradient's dot product (step 2b), whose
// reference is OpenBLAS's ddot, an order the core cannot copy.

#pragma once

#include <cstddef>
#include <span>

namespace cfdcore::reduce {

// The sum of a contiguous array of doubles in NumPy's pairwise order.
//
// values: the terms, any length including zero, contiguous, in any unit; the
//         sum is in the same unit.
//
// Returns the sum `np.sum` gives for the same C-contiguous float64 array: 0.0
// added to the pairwise sum of all the terms. The pairwise sum of n terms is a
// plain left-to-right sum below 8 terms; up to 128 terms it is eight
// accumulators over the terms in blocks of eight, combined as
// ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7)), with the terms beyond the
// last whole block added after; above 128 it is the sum of the first n2 terms
// plus the sum of the rest, n2 being n / 2 rounded down to a multiple of 8.
// The leading 0.0 turns a sum of negative zeros into 0.0, as NumPy's does.
double pairwise_sum(std::span<const double> values);

// The sum `np.sum` gives for a two-dimensional array that is not contiguous: a
// slice of whole rows' parts, such as the interior faces `a[:, 1:-1]` of a face
// field, its rows laid out one after another in `values`.
//
// values: the slice's terms row by row, contiguous, in any unit
// row:    the slice's row length; `values` holds a whole number of rows
//
// NumPy reduces such a slice through its iterator's buffer, `buffer_size`
// elements by default (np.getbufsize()), filled with whole rows: each fill is
// summed in `pairwise_sum`'s order and the fills' sums are added to 0.0 in
// order, so the result parts from the pairwise sum of all the terms once there
// are more than one fill. A row of one element is the exception: NumPy reads
// that slice as one strided line and sums it pairwise without the buffer. The
// rule was measured on NumPy 2.4.4 against `np.sum` on slices of every kind the
// transport budget sums (prompt 90), and is pinned by the parity module
// (`test_buffered_sum_equals_np_sum_on_strided_slices`). Throws
// std::invalid_argument for a row of 0 with terms, or a length that is not a
// whole number of rows.
inline constexpr std::size_t buffer_size = 8192;
double buffered_sum(std::span<const double> values, std::size_t row);

// The dot product of two arrays in fixed blocks (ADR-017 G, the second kind).
//
// a, b:  the two arrays, equal lengths, contiguous, in any units; the result
//        is in the product of the two
// block: the block length, at least 1; the arrays are cut into blocks of this
//        many terms from the first, the last block holding what remains, so a
//        length need not be a whole number of blocks
//
// Throws std::invalid_argument for a block of 0 (the partition is undefined)
// and for arrays of different lengths (the shorter would be read past its end),
// in every build, not only a debug one.
//
// Each term a[i] * b[i] is one rounded product. Each block's terms are summed
// in `pairwise_sum`'s order (without its leading 0.0), and the block sums are
// added to 0.0 in block order. The partition is fixed by the length and the
// block, never by a thread count, so a parallel build that computes the
// blocks apart and adds them in order gives the same bits. The implicit solve
// takes a block as one row of the cell field, a line along direction 0.
// Returns 0.0 for empty arrays.
double blocked_dot(std::span<const double> a, std::span<const double> b, std::size_t block);

// One block's term of `blocked_dot`: the products a[i] * b[i], one rounding
// each, summed in `pairwise_sum`'s order without its leading 0.0.
//
// a, b: the block's two arrays, equal lengths, contiguous, in any units; the
//       result is in the product of the two
//
// Adding the blocks' results to 0.0 in block order gives `blocked_dot`'s value
// bit for bit, so a caller that forms its arrays block by block can take each
// block's dot product while the block is in the cache. Throws
// std::invalid_argument for arrays of different lengths. Returns 0.0 for empty
// arrays.
double block_dot(std::span<const double> a, std::span<const double> b);

}  // namespace cfdcore::reduce
