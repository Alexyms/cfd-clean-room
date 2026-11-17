// The nanobind module _cfdcore (ADR-017 B): the core's blocks as Python
// functions, and nothing else. Arrays cross as aligned, C-contiguous float64
// NumPy arrays of a stated dimension; `noconvert` makes a wrong dtype, layout
// or dimension a TypeError at the call instead of a silent copy, and the
// functions below refuse an unaligned array the same way.

// First, as Python.h requires: after a standard header its pyconfig.h redefines feature macros.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "cfdcore/reduce.hpp"

#include <cstdint>
#include <span>

namespace nb = nanobind;

namespace {

// A read-only, C-contiguous, one-dimensional float64 array on the CPU.
using Vector = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

double pairwise_sum(const Vector& values) {
    // np.sum adds an unaligned array through a buffer, in chunks, which changes its
    // pairwise order, so the bit-for-bit promise holds for aligned arrays only.
    if (reinterpret_cast<std::uintptr_t>(values.data()) % alignof(double) != 0) {
        throw nb::type_error(
            "pairwise_sum: the array is not aligned (data pointer not a multiple of eight)");
    }
    return cfdcore::reduce::pairwise_sum(
        std::span<const double>(values.data(), values.shape(0)));
}

}  // namespace

NB_MODULE(_cfdcore, m) {
    m.doc() = "The compiled core of the clean room simulation (ADR-017).";

    m.def("pairwise_sum", &pairwise_sum, nb::arg("values").noconvert(),
          "The sum of a C-contiguous 1-D float64 array in NumPy's pairwise order.\n\n"
          "Equals np.sum(values) bit for bit on an aligned array (ADR-017 G). A wrong\n"
          "dtype, a non-contiguous or unaligned array or another dimension raises\n"
          "TypeError.");
}
