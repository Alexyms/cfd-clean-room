// The core's reductions (ADR-017 C, module `reduce`; ADR-017 G).
//
// A sum that a stop, a refusal or a budget reads is formed in the reference's
// order, so that the core's value equals NumPy's bit for bit given the same
// array. Phase 6 step 1 builds the first of them, NumPy's pairwise sum.

#pragma once

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

}  // namespace cfdcore::reduce
