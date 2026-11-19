// The nanobind module _cfdcore (ADR-017 B): the core's blocks as Python
// functions, and nothing else. Arrays cross as aligned, C-contiguous NumPy
// arrays of a stated dtype and dimension; `noconvert` makes a wrong dtype,
// layout or dimension a TypeError at the call instead of a silent copy, and
// the functions below refuse an unaligned array or a wrong shape the same way.

// First, as Python.h requires: after a standard header its pyconfig.h redefines feature macros.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "cfdcore/grid.hpp"
#include "cfdcore/reduce.hpp"
#include "cfdcore/scalar.hpp"

#include <algorithm>
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
    // Shapes are right, so a bad coordinate is a bad value, not a bad array.
    const int bad = Grid2::first_bad_direction(input);
    if (bad >= 0) {
        throw nb::value_error((std::string("Grid2: the face or node coordinates along ") +
                               (bad == 0 ? "x" : "y") + " (direction " + std::to_string(bad) +
                               ") are not finite and strictly increasing")
                                  .c_str());
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

// The elements of a two-dimensional array of any of the aliases above, as one span.
template <typename A>
std::span<const typename A::Scalar> span_of(const A& a) {
    return std::span<const typename A::Scalar>(a.data(), a.size());
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

// --- The implicit solve -----------------------------------------------------

double row_dot(const Matrix& a, const Matrix& b) {
    constexpr std::string_view who = "row_dot";
    if (a.shape(0) != b.shape(0) || a.shape(1) != b.shape(1)) {
        throw nb::type_error("row_dot: a and b differ in shape");
    }
    require_aligned(a.data(), alignof(double), who);
    require_aligned(b.data(), alignof(double), who);
    // A block is at least 1; an array with no columns has no terms, so its
    // block is never read and the dot product is 0.0, as np.vdot's is.
    const std::size_t block = std::max<std::size_t>(a.shape(1), 1);
    return cfdcore::reduce::blocked_dot(span_of(a), span_of(b), block);
}

using Scratch2 = cfdcore::scalar::implicit_scratch<2>;
using Input2 = cfdcore::scalar::implicit_input<2>;
using cfdcore::scalar::implicit_method;
using cfdcore::scalar::implicit_result;

// The scratch with the grid it was sized on, which the binding checks shapes against.
struct ImplicitScratch2 {
    explicit ImplicitScratch2(const Grid2& g) : grid(&g), scratch(g.fields()) {}
    const Grid2* grid;
    Scratch2 scratch;
};

// The arrays of one call kept alive while the core reads them.
struct implicit_arrays {
    std::optional<Matrix> dt_cells;
};

// The step: a number (a Python float, a NumPy scalar or a 0-d array, as
// np.ndim(dt) == 0 reads it), or a float64 array of the cell field's shape.
void read_step(nb::handle dt, const Grid2& g, Input2& in, implicit_arrays& keep) {
    constexpr std::string_view who = "implicit_step";
    const nb::object ndim = nb::getattr(dt, "ndim", nb::none());
    if (ndim.is_none() || nb::cast<long>(ndim) == 0) {
        double value = 0.0;
        if (!nb::try_cast<double>(dt, value)) {
            throw nb::type_error("implicit_step: dt is a number or a float64 cell field");
        }
        in.dt = value;
        return;
    }
    Matrix cells;
    if (!nb::try_cast<Matrix>(dt, cells, false)) {
        throw nb::type_error(
            "implicit_step: dt is a number or a C-contiguous float64 cell field");
    }
    require_shape(cells, {g.cells(1), g.cells(0)}, who, "dt");
    keep.dt_cells = cells;
    in.dt_cells = span_of(*keep.dt_cells);
}

// The method by its reference name; ValueError otherwise, as the reference raises it.
implicit_method read_method(nb::handle method) {
    if (nb::isinstance<nb::str>(method)) {
        const std::string name = nb::cast<std::string>(method);
        if (name == "jacobi") {
            return implicit_method::jacobi;
        }
        if (name == "cg") {
            return implicit_method::conjugate_gradient;
        }
    }
    throw nb::value_error(("method must be one of ('jacobi', 'cg'), got " +
                           nb::cast<std::string>(nb::repr(method)))
                              .c_str());
}

// The bound: None, or a number; a bool (Python's or NumPy's) is a TypeError,
// as the reference refuses it, since it would read as 1.0 or 0.0.
std::optional<double> read_bound(nb::handle bound) {
    if (bound.is_none()) {
        return std::nullopt;
    }
    const nb::object numpy_bool = nb::module_::import_("numpy").attr("bool_");
    if (nb::isinstance<bool>(bound) || PyObject_IsInstance(bound.ptr(), numpy_bool.ptr()) == 1) {
        throw nb::type_error(("leftover_bound must be a number or None, not a bool, got " +
                              nb::cast<std::string>(nb::repr(bound)))
                                 .c_str());
    }
    double value = 0.0;
    if (!nb::try_cast<double>(bound, value)) {
        throw nb::type_error("leftover_bound must be a number or None");
    }
    return value;
}

// The checks and the input of implicit_step and implicit_system.
Input2 check_implicit(const ImplicitScratch2& s, const Matrix& c_star, const Matrix& volume,
                      nb::handle dt, const Matrix& g_u, const Matrix& g_v,
                      const Matrix& diagonal, implicit_arrays& keep) {
    constexpr std::string_view who = "implicit_step";
    const Grid2& g = *s.grid;
    const std::array<std::size_t, 2> cells{g.cells(1), g.cells(0)};
    require_shape(c_star, cells, who, "c_star");
    require_shape(volume, cells, who, "volume");
    require_shape(g_u, face_shape(g, 0), who, "g_u");
    require_shape(g_v, face_shape(g, 1), who, "g_v");
    require_shape(diagonal, cells, who, "diagonal");
    Input2 in{};
    in.c_star = span_of(c_star);
    in.volume = span_of(volume);
    read_step(dt, g, in, keep);
    in.conductance = {span_of(g_u), span_of(g_v)};
    in.sink = span_of(diagonal);
    return in;
}

nb::tuple implicit_system(ImplicitScratch2& s, const Matrix& c_star, const Matrix& volume,
                          nb::handle dt, const Matrix& g_u, const Matrix& g_v,
                          const Matrix& diagonal, double tol) {
    implicit_arrays keep;
    Input2 in = check_implicit(s, c_star, volume, dt, g_u, g_v, diagonal, keep);
    in.tol = tol;
    const auto setup = cfdcore::scalar::implicit_system(in, s.scratch);
    return nb::make_tuple(setup.diffuses, setup.stop);
}

implicit_result implicit_step(ImplicitScratch2& s, const Matrix& c_star, const Matrix& volume,
                              nb::handle dt, const Matrix& g_u, const Matrix& g_v,
                              const Matrix& diagonal, const BoolMatrix& solid, double tol,
                              std::int64_t max_iter, OutMatrix& out,
                              const std::optional<BoolMatrix>& held, nb::handle method,
                              nb::handle leftover_bound) {
    constexpr std::string_view who = "implicit_step";
    // The reference's order: the method, then the bound's type, then its value.
    const implicit_method chosen = read_method(method);
    const std::optional<double> bound = read_bound(leftover_bound);
    implicit_arrays keep;
    Input2 in = check_implicit(s, c_star, volume, dt, g_u, g_v, diagonal, keep);
    const Grid2& g = *s.grid;
    const std::array<std::size_t, 2> cells{g.cells(1), g.cells(0)};
    require_shape(solid, cells, who, "solid");
    require_shape(out, cells, who, "out");
    in.solid = span_of(solid);
    if (held) {
        require_shape(*held, cells, who, "held");
        in.held = span_of(*held);
    }
    const std::size_t out_bytes = out.size() * sizeof(double);
    for (const auto& a : {in.c_star, in.volume, in.dt_cells, in.conductance[0],
                          in.conductance[1], in.sink}) {
        require_apart(out.data(), out_bytes, a.data(), a.size_bytes(), who);
    }
    require_apart(out.data(), out_bytes, in.solid.data(), in.solid.size_bytes(), who);
    require_apart(out.data(), out_bytes, in.held.data(), in.held.size_bytes(), who);
    in.tol = tol;
    in.max_iter = max_iter;
    in.method = chosen;
    in.leftover_bound = bound;
    implicit_result result;
    const auto status = cfdcore::scalar::implicit_step(
        in, s.scratch, std::span<double>(out.data(), out.size()), result);
    if (status == cfdcore::scalar::implicit_status::bound_negative_or_nan) {
        throw nb::value_error(("leftover_bound must be non-negative or None, got " +
                               nb::cast<std::string>(nb::repr(leftover_bound)))
                                  .c_str());
    }
    return result;
}

const char* method_name(implicit_method m) {
    return m == implicit_method::jacobi ? "jacobi" : "cg";
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
                      "wrong dtype or an unaligned array raise TypeError; faces or nodes\n"
                      "that are not finite and strictly increasing along a direction raise\n"
                      "ValueError naming it.")
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

    m.def("row_dot", &row_dot, nb::arg("a").noconvert(), nb::arg("b").noconvert(),
          "The dot product of two float64 arrays of one 2-D shape, a block per row\n"
          "(reduce.blocked_dot): the conjugate gradient's dot product (ADR-017 G).");

    nb::class_<ImplicitScratch2>(m, "ImplicitScratch2",
                                 "The implicit solve's scratch on a Grid2, allocated once, so a\n"
                                 "solve allocates nothing. Keeps its grid alive.")
        .def(nb::init<const Grid2&>(), nb::arg("grid"), nb::keep_alive<1, 2>())
        .def(
            "diagonal", [](const ImplicitScratch2& s) { return to_numpy(s.scratch.a_p); },
            "a_P as implicit_system last formed it, flattened (a copy).")
        .def(
            "rhs", [](const ImplicitScratch2& s) { return to_numpy(s.scratch.b); },
            "b as implicit_system last formed it, flattened (a copy).");

    nb::class_<implicit_result>(m, "ImplicitResult2",
                                "What implicit_step returns beside the field it writes to out:\n"
                                "ImplicitResult's fields, and leftover_sums and restarts.")
        .def_prop_ro("method", [](const implicit_result& r) { return method_name(r.method); })
        .def_ro("iterations", &implicit_result::iterations)
        .def_ro("converged", &implicit_result::converged)
        .def_ro("fallbacks", &implicit_result::fallbacks)
        .def_ro("fallback_sweeps", &implicit_result::fallback_sweeps)
        .def_ro("leftover", &implicit_result::leftover)
        .def_ro("leftover_sums", &implicit_result::leftover_sums)
        .def_ro("restarts", &implicit_result::restarts);

    m.def("implicit_system", &implicit_system, nb::arg("scratch"), nb::arg("c_star").noconvert(),
          nb::arg("volume").noconvert(), nb::arg("dt"), nb::arg("g_u").noconvert(),
          nb::arg("g_v").noconvert(), nb::arg("diagonal").noconvert(), nb::arg("tol"),
          "implicit_step's set-up alone: returns (diffuses, stop) and leaves a_P and b in\n"
          "the scratch (ImplicitScratch2.diagonal and rhs).");
    m.def("implicit_step", &implicit_step, nb::arg("scratch"), nb::arg("c_star").noconvert(),
          nb::arg("volume").noconvert(), nb::arg("dt"), nb::arg("g_u").noconvert(),
          nb::arg("g_v").noconvert(), nb::arg("diagonal").noconvert(),
          nb::arg("solid").noconvert(), nb::arg("tol"), nb::arg("max_iter"),
          nb::arg("out").noconvert(), nb::arg("held").noconvert().none() = nb::none(),
          nb::arg("method").none() = "jacobi", nb::arg("leftover_bound").none() = nb::none(),
          "scalar_scheme.implicit_step: the field into out [ny, nx], the rest returned.\n\n"
          "c_star, volume, diagonal, solid and held (bool, or None) [ny, nx]; g_u\n"
          "[ny, nx + 1]; g_v [ny + 1, nx]; dt a number or a float64 [ny, nx]. A wrong\n"
          "shape, dtype or layout, an unaligned array, an out sharing memory with an\n"
          "input, or a bool leftover_bound raises TypeError; an unknown method, or a\n"
          "negative or NaN leftover_bound, ValueError, as the reference raises them.");
}

