// The nanobind module _cfdcore (ADR-017 B): the core's blocks as Python
// functions, and nothing else. Arrays cross as aligned, C-contiguous NumPy
// arrays of a stated dtype and dimension; `noconvert` makes a wrong dtype,
// layout or dimension a TypeError at the call instead of a silent copy, and
// the functions below refuse an unaligned array or a wrong shape the same way.

// First, as Python.h requires: after a standard header its pyconfig.h redefines feature macros.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

#include "cfdcore/grid.hpp"
#include "cfdcore/reduce.hpp"
#include "cfdcore/scalar.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
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
// Read-only, C-contiguous two-dimensional float64 and bool arrays on the CPU.
using Matrix = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using BoolMatrix = nb::ndarray<const bool, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
// A writable, C-contiguous two-dimensional float64 array on the CPU: an output.
using OutMatrix = nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
// Read-only and writable C-contiguous float64 arrays of any dimension.
using Array = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;
using OutArray = nb::ndarray<double, nb::c_contig, nb::device::cpu>;

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

using Scheme2 = cfdcore::scalar::scheme<2>;

// The shape of the face field normal to direction d, (rows, columns).
std::array<std::size_t, 2> face_shape(const Grid2& g, int d) {
    return d == 0 ? std::array<std::size_t, 2>{g.cells(1), g.cells(0) + 1}
                  : std::array<std::size_t, 2>{g.cells(1) + 1, g.cells(0)};
}

// Refuses a two-dimensional array of another shape, or an unaligned one.
template <typename A>
void require_shape(const A& a, std::array<std::size_t, 2> shape, std::string_view who,
                   std::string_view name) {
    if (a.shape(0) != shape[0] || a.shape(1) != shape[1]) {
        throw nb::type_error((std::string(who) + ": " + std::string(name) + " is not [" +
                              std::to_string(shape[0]) + ", " + std::to_string(shape[1]) + "]")
                                 .c_str());
    }
    require_aligned(a.data(), alignof(std::remove_cv_t<typename A::Scalar>), who);
}

// Refuses an output that shares memory with an input: the core reads every
// input while it writes the output.
void require_apart(const void* out, std::size_t out_bytes, const void* in, std::size_t in_bytes,
                   std::string_view who) {
    const auto o = reinterpret_cast<std::uintptr_t>(out);
    const auto i = reinterpret_cast<std::uintptr_t>(in);
    if (o < i + in_bytes && i < o + out_bytes) {
        throw nb::type_error((std::string(who) + ": out shares memory with an input").c_str());
    }
}

template <typename T>
std::span<const T> span_of(
    const nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>& a) {
    return std::span<const T>(a.data(), a.size());
}

// What advective_flux reads and writes, once its checks have passed.
struct flux_arguments {
    std::span<const double> c;
    std::span<const double> flux;
    std::span<const double> inflow;
    std::span<const bool> vents;
    std::span<double> out;
};

flux_arguments check_flux(const Scheme2& s, int d, const Matrix& c, const Matrix& flux,
                          const Matrix& inflow, OutMatrix& out,
                          const std::optional<BoolMatrix>& vents) {
    constexpr std::string_view who = "advective_flux";
    const Grid2& g = s.mesh();
    direction(d);
    const auto faces = face_shape(g, d);
    require_shape(c, {g.cells(1), g.cells(0)}, who, "c");
    require_shape(flux, faces, who, "flux");
    require_shape(inflow, faces, who, "inflow");
    require_shape(out, faces, who, "out");
    flux_arguments args{span_of(c), span_of(flux), span_of(inflow), {},
                        std::span<double>(out.data(), out.size())};
    const std::size_t out_bytes = out.size() * sizeof(double);
    for (const auto& in : {args.c, args.flux, args.inflow}) {
        require_apart(out.data(), out_bytes, in.data(), in.size_bytes(), who);
    }
    if (vents) {
        require_shape(*vents, faces, who, "vents");
        args.vents = span_of(*vents);
        require_apart(out.data(), out_bytes, args.vents.data(), args.vents.size_bytes(), who);
    }
    return args;
}

void advective_flux(const Scheme2& s, int d, const Matrix& c, const Matrix& flux,
                    const Matrix& inflow, bool upwind, OutMatrix& out,
                    const std::optional<BoolMatrix>& vents) {
    const auto a = check_flux(s, d, c, flux, inflow, out, vents);
    cfdcore::scalar::advective_flux(s, d, a.c, a.flux, a.inflow, upwind, a.vents, a.out);
}

// advective_flux's argument conversion and checks, without the flux.
void advective_flux_overhead(const Scheme2& s, int d, const Matrix& c, const Matrix& flux,
                             const Matrix& inflow, bool /*upwind*/, OutMatrix& out,
                             const std::optional<BoolMatrix>& vents) {
    check_flux(s, d, c, flux, inflow, out, vents);
}

void quick_face_values(const Scheme2& s, int d, const Matrix& c, const Vector& c_low,
                       const Vector& c_high, const BoolMatrix& positive, OutMatrix& out) {
    constexpr std::string_view who = "quick_face_values";
    const Grid2& g = s.mesh();
    direction(d);
    const auto faces = face_shape(g, d);
    const std::size_t lines = g.fields().lines(d).size();
    require_shape(c, {g.cells(1), g.cells(0)}, who, "c");
    require_shape(positive, faces, who, "positive");
    require_shape(out, faces, who, "out");
    for (const auto* v : {&c_low, &c_high}) {
        if (v->shape(0) != lines) {
            throw nb::type_error("quick_face_values: c_low and c_high hold one value per line");
        }
        require_aligned(v->data(), alignof(double), who);
    }
    const std::size_t out_bytes = out.size() * sizeof(double);
    require_apart(out.data(), out_bytes, c.data(), c.size() * sizeof(double), who);
    require_apart(out.data(), out_bytes, c_low.data(), lines * sizeof(double), who);
    require_apart(out.data(), out_bytes, c_high.data(), lines * sizeof(double), who);
    require_apart(out.data(), out_bytes, positive.data(), positive.size(), who);
    cfdcore::scalar::quick_face_values(s, d, span_of(c), as_span(c_low), as_span(c_high),
                                       span_of(positive),
                                       std::span<double>(out.data(), out.size()));
}

void limited_face_values(const Array& c_up, const Array& c_c, const Array& c_d,
                         const Array& quick, OutArray& out) {
    constexpr std::string_view who = "limited_face_values";
    require_aligned(out.data(), alignof(double), who);
    for (const auto* a : {&c_up, &c_c, &c_d, &quick}) {
        bool same = a->ndim() == out.ndim();
        for (std::size_t i = 0; same && i < out.ndim(); ++i) {
            same = a->shape(i) == out.shape(i);
        }
        if (!same) {
            throw nb::type_error("limited_face_values: the arrays' shapes differ");
        }
        require_aligned(a->data(), alignof(double), who);
        require_apart(out.data(), out.size() * sizeof(double), a->data(),
                      a->size() * sizeof(double), who);
    }
    const auto span = [](const Array& a) { return std::span<const double>(a.data(), a.size()); };
    cfdcore::scalar::limited_face_values(span(c_up), span(c_c), span(c_d), span(quick),
                                         std::span<double>(out.data(), out.size()));
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

    nb::class_<Scheme2>(m, "ScalarScheme2",
                        "The scalar scheme's set-up on a Grid2: the QUICK weights of both\n"
                        "directions, computed once. Keeps its grid alive.")
        .def(nb::init<const Grid2&>(), nb::arg("grid"), nb::keep_alive<1, 2>())
        .def(
            "weights",
            [](const Scheme2& s, int d) {
                const auto& w = s.weights(direction(d));
                return nb::make_tuple(to_numpy(w.positive[0]), to_numpy(w.positive[1]),
                                      to_numpy(w.positive[2]), to_numpy(w.negative[0]),
                                      to_numpy(w.negative[1]), to_numpy(w.negative[2]));
            },
            nb::arg("direction"),
            "The six weight arrays of a direction's interior faces: the positive\n"
            "direction's far, upstream and downstream nodes', then the negative's.");

    m.def("quick_face_values", &quick_face_values, nb::arg("scheme"), nb::arg("direction"),
          nb::arg("c").noconvert(), nb::arg("c_low").noconvert(), nb::arg("c_high").noconvert(),
          nb::arg("positive").noconvert(), nb::arg("out").noconvert(),
          "QUICK's unlimited values at the interior faces normal to a direction.\n\n"
          "c [ny, nx]; c_low and c_high the boundary nodes per line along the direction;\n"
          "positive (bool) and out on the face field normal to it. Writes out's interior\n"
          "faces only.");
    m.def("limited_face_values", &limited_face_values, nb::arg("c_up").noconvert(),
          nb::arg("c_c").noconvert(), nb::arg("c_d").noconvert(), nb::arg("quick").noconvert(),
          nb::arg("out").noconvert(),
          "The UMIST clamp of QUICK face values, elementwise, into out (any one shape).");
    m.def("advective_flux", &advective_flux, nb::arg("scheme"), nb::arg("direction"),
          nb::arg("c").noconvert(), nb::arg("flux").noconvert(), nb::arg("inflow").noconvert(),
          nb::arg("upwind"), nb::arg("out").noconvert(),
          nb::arg("vents").noconvert().none() = nb::none(),
          "The volume flux times the face value on every face normal to a direction.\n\n"
          "c [ny, nx]; flux, inflow, out and vents (bool, or None) on the face field\n"
          "normal to the direction: [ny, nx + 1] for 0, [ny + 1, nx] for 1. A wrong\n"
          "shape, dtype or layout, an unaligned array or an out sharing memory with an\n"
          "input raises TypeError.");
    m.def("advective_flux_overhead", &advective_flux_overhead, nb::arg("scheme"),
          nb::arg("direction"), nb::arg("c").noconvert(), nb::arg("flux").noconvert(),
          nb::arg("inflow").noconvert(), nb::arg("upwind"), nb::arg("out").noconvert(),
          nb::arg("vents").noconvert().none() = nb::none(),
          "advective_flux's argument conversion and checks without the flux: the\n"
          "binding's cost per call, for timing (prompt 87).");
}
