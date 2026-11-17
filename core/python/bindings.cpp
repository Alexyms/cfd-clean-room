// The nanobind module _cfdcore (ADR-017 B): the core's blocks as Python
// functions, and nothing else. Arrays cross as aligned, C-contiguous NumPy
// arrays of a stated dtype and dimension; `noconvert` makes a wrong dtype,
// layout or dimension a TypeError at the call instead of a silent copy, and
// the functions below refuse an unaligned array or a wrong shape the same way.

// First, as Python.h requires: after a standard header its pyconfig.h redefines feature macros.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "cfdcore/grid.hpp"
#include "cfdcore/reduce.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace nb = nanobind;

namespace {

// A read-only, C-contiguous, one-dimensional float64 array on the CPU.
using Vector = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
// A read-only, C-contiguous, two-dimensional int32 array on the CPU.
using IntMatrix = nb::ndarray<const std::int32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
// A new one-dimensional float64 NumPy array, owned by Python.
using OutVector = nb::ndarray<nb::numpy, double, nb::ndim<1>>;

// Refuses an array whose data pointer is not a multiple of the element's alignment.
void require_aligned(const void* data, std::size_t alignment, std::string_view who) {
    if (reinterpret_cast<std::uintptr_t>(data) % alignment != 0) {
        throw nb::type_error((std::string(who) + ": an array is not aligned").c_str());
    }
}

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

std::span<const double> as_span(const Vector& v) {
    return std::span<const double>(v.data(), v.shape(0));
}

// A copy of a core array as a new NumPy array.
OutVector to_numpy(std::span<const double> values) {
    auto* copy = new double[values.size()];
    for (std::size_t i = 0; i < values.size(); ++i) {
        copy[i] = values[i];
    }
    nb::capsule owner(copy, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    return OutVector(copy, {values.size()}, owner);
}

using Grid2 = cfdcore::grid::grid<2>;

Grid2 make_grid(const Vector& x, const Vector& xc, const Vector& y, const Vector& yc,
                const IntMatrix& cell_type, std::int32_t solid) {
    for (const auto* v : {&x, &xc, &y, &yc}) {
        require_aligned(v->data(), alignof(double), "Grid2");
    }
    require_aligned(cell_type.data(), alignof(std::int32_t), "Grid2");
    if (cell_type.shape(0) != yc.shape(0) || cell_type.shape(1) != xc.shape(0)) {
        throw nb::type_error("Grid2: cell_type is not [len(yc), len(xc)]");
    }
    const cfdcore::grid::grid_input<2> input{
        .faces = {as_span(x), as_span(y)},
        .centres = {as_span(xc), as_span(yc)},
        .cell_type = std::span<const std::int32_t>(cell_type.data(), cell_type.size()),
        .solid = solid,
    };
    const std::string_view reason = Grid2::check(input);
    if (!reason.empty()) {
        throw nb::type_error((std::string("Grid2: ") + std::string(reason)).c_str());
    }
    return Grid2(input);
}

int direction(int d) {
    if (d != 0 && d != 1) {
        throw nb::value_error("the direction is 0 (x) or 1 (y)");
    }
    return d;
}

}  // namespace

NB_MODULE(_cfdcore, m) {
    m.doc() = "The compiled core of the clean room simulation (ADR-017).";

    m.def("pairwise_sum", &pairwise_sum, nb::arg("values").noconvert(),
          "The sum of a C-contiguous 1-D float64 array in NumPy's pairwise order.\n\n"
          "Equals np.sum(values) bit for bit on an aligned array (ADR-017 G). A wrong\n"
          "dtype, a non-contiguous or unaligned array or another dimension raises\n"
          "TypeError.");

    nb::class_<Grid2>(m, "Grid2",
                      "The core's two-dimensional grid, built once from a Mesh's arrays.\n\n"
                      "Grid2(x, xc, y, yc, cell_type, solid): the face and centre\n"
                      "coordinates per direction (float64, 1-D), the cell types (int32,\n"
                      "[ny, nx]) and the type of a SOLID cell. Inconsistent shapes, a\n"
                      "wrong dtype or an unaligned array raise TypeError.")
        .def("__init__",
             [](Grid2* self, const Vector& x, const Vector& xc, const Vector& y, const Vector& yc,
                const IntMatrix& cell_type, std::int32_t solid) {
                 new (self) Grid2(make_grid(x, xc, y, yc, cell_type, solid));
             },
             nb::arg("x").noconvert(), nb::arg("xc").noconvert(), nb::arg("y").noconvert(),
             nb::arg("yc").noconvert(), nb::arg("cell_type").noconvert(), nb::arg("solid"))
        .def_prop_ro(
            "shape",
            [](const Grid2& g) { return nb::make_tuple(g.cells(1), g.cells(0)); },
            "The cell field's shape, (ny, nx).")
        .def(
            "nodes", [](const Grid2& g, int d) { return to_numpy(g.nodes(direction(d))); },
            nb::arg("direction"),
            "The boundary faces around the cell centres along a direction (Axis.nodes).")
        .def(
            "interior_faces",
            [](const Grid2& g, int d) { return to_numpy(g.interior_faces(direction(d))); },
            nb::arg("direction"), "The interior face coordinates along a direction (Axis.faces).");
}
