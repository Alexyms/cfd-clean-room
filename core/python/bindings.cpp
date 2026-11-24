// The nanobind module _cfdcore (ADR-017 B): the core's blocks as Python
// functions, and nothing else. Arrays cross as aligned, C-contiguous NumPy
// arrays of a stated dtype and dimension; `noconvert` makes a wrong dtype,
// layout or dimension a TypeError at the call instead of a silent copy, and
// the functions below refuse an unaligned array or a wrong shape the same way.

// First, as Python.h requires: after a standard header its pyconfig.h redefines feature macros.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "cfdcore/arrival.hpp"
#include "cfdcore/channels.hpp"
#include "cfdcore/grid.hpp"
#include "cfdcore/march.hpp"
#include "cfdcore/momentum.hpp"
#include "cfdcore/recovery.hpp"
#include "cfdcore/reduce.hpp"
#include "cfdcore/scalar.hpp"
#include "cfdcore/transport.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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

// --- One class step (prompt 90) ---------------------------------------------

using Transport2 = cfdcore::transport::solver<2>;
using cfdcore::transport::refusal;
using cfdcore::transport::status;

// A Python float's str, as an f-string formats it.
std::string float_str(double x) { return nb::cast<std::string>(nb::str(nb::float_(x))); }
std::string float_repr(double x) { return nb::cast<std::string>(nb::repr(nb::float_(x))); }

// Raises the exception the reference raises for a refusal.
[[noreturn]] void raise_refusal(const refusal& r) {
    const std::string k = std::to_string(r.size_class);
    switch (r.code) {
        case status::condition_not_finite_or_negative:
            throw nb::value_error(
                ("class " + k + ": " + r.name + " must be finite and non-negative").c_str());
        case status::recirculation_without_vents:
            throw nb::value_error(
                ("class " + k + ": a recirculation needs the vent faces").c_str());
        case status::passed_out_of_range:
            throw nb::value_error(
                ("class " + k + ": a recirculation passes a fraction in [0, 1], got " +
                 float_str(r.value))
                    .c_str());
        case status::vents_differ:
            throw nb::value_error(("class " + k +
                                   " names other vent faces than class 0; the vents are the "
                                   "room's, the same for every class")
                                      .c_str());
        case status::faces_not_finite:
            throw nb::value_error("faces must be finite");
        case status::drift_not_finite:
            throw nb::value_error("v_ext must be finite");
        case status::sources_not_finite:
            throw nb::value_error("sources must be finite");
        case status::sources_negative:
            throw nb::value_error("sources must be non-negative (ADR-011 B)");
        case status::source_in_solid:
            throw nb::value_error("a source sits in a SOLID cell");
        case status::eddy_viscosity_without_schmidt:
            throw nb::value_error(
                "an eddy_viscosity field needs transport.turbulent_schmidt in the configuration "
                "(REQ-T13)");
        case status::eddy_viscosity_not_finite:
            throw nb::value_error("eddy_viscosity must be finite in every non-SOLID cell");
        case status::eddy_viscosity_negative:
            throw nb::value_error("eddy_viscosity must be non-negative (ADR-012 F)");
        case status::dt_not_finite:
            throw nb::value_error(("dt must be finite, got " + float_str(r.value)).c_str());
        case status::dt_not_positive:
            throw nb::value_error(("dt must be positive, got " + float_str(r.value)).c_str());
        case status::field_not_finite:
            throw nb::value_error("C_k must be finite");
        case status::dt_above_stable:
            throw nb::value_error(("dt " + float_str(r.value) + " exceeds stable_dt " +
                                   float_str(r.other) + " for class " + k +
                                   "; the explicit advection step is a convex combination "
                                   "only below it")
                                      .c_str());
        case status::bound_negative_or_nan:
            throw nb::value_error(
                ("leftover_bound must be non-negative or None, got " + float_repr(r.value))
                    .c_str());
        case status::breach:
        case status::ok:
            break;
    }
    throw std::logic_error("raise_refusal: not a refusal the binding raises");
}

// The solver with the scheme it was built on (kept alive by the binding).
struct TransportSolver2 {
    TransportSolver2(const Scheme2& s, const cfdcore::transport::geometry<2>& spacing,
                     const cfdcore::transport::settings& keys,
                     std::vector<cfdcore::transport::class_physics> physics)
        : scheme(&s), solver(s, spacing, keys, std::move(physics)), limit(keys.limit) {}
    const Scheme2* scheme;
    Transport2 solver;
    double limit;
};

// What a solver is built from, read and checked once for the two classes that
// build one (`TransportSolver2` and `March2`): the keys, each class's physics and
// the mesh's spacings, which are views of the caller's arrays.
struct solver_setup {
    cfdcore::transport::settings keys;
    std::vector<cfdcore::transport::class_physics> physics;
    cfdcore::transport::geometry<2> spacing;
};

solver_setup read_solver_setup(std::string_view who, const Scheme2& s, const Vector& dx_cell,
                               const Vector& dy_cell, const Vector& dx_face,
                               const Vector& dy_face, double cfl, bool upwind,
                               std::int64_t max_sweeps, double tol, double limit, double share,
                               double horizon, std::optional<double> schmidt,
                               const std::vector<double>& settling,
                               const std::vector<double>& diffusion,
                               const std::array<std::int32_t, 3>& surface_codes) {
    const Grid2& g = s.mesh();
    const std::array<std::pair<const Vector*, std::size_t>, 4> spacings{
        {{&dx_cell, g.cells(0)}, {&dy_cell, g.cells(1)}, {&dx_face, g.cells(0) + 1},
         {&dy_face, g.cells(1) + 1}}};
    for (const auto& [v, n] : spacings) {
        if (v->shape(0) != n) {
            throw nb::type_error((std::string(who) +
                                  ": dx_cell, dy_cell, dx_face and dy_face are nx, ny, nx + 1 "
                                  "and ny + 1 long")
                                     .c_str());
        }
        require_aligned(v->data(), alignof(double), who);
    }
    if (settling.empty() || settling.size() != diffusion.size()) {
        throw nb::value_error(
            (std::string(who) + ": settling and diffusion hold one value per class").c_str());
    }
    cfdcore::transport::settings keys;
    keys.cfl = cfl;
    keys.upwind = upwind;
    keys.max_sweeps = max_sweeps;
    keys.tol = tol;
    keys.limit = limit;
    keys.share = share;
    keys.horizon = horizon;
    keys.schmidt = schmidt;
    keys.surface_codes = surface_codes;
    std::vector<cfdcore::transport::class_physics> physics;
    for (std::size_t k = 0; k < settling.size(); ++k) {
        physics.push_back({settling[k], diffusion[k]});
    }
    cfdcore::transport::geometry<2> spacing{
        .widths = {as_span(dx_cell), as_span(dy_cell)},
        .face_distances = {as_span(dx_face), as_span(dy_face)}};
    return {keys, std::move(physics), spacing};
}

void transport_init(TransportSolver2* self, const Scheme2& s, const Vector& dx_cell,
                    const Vector& dy_cell, const Vector& dx_face, const Vector& dy_face,
                    double cfl, bool upwind, std::int64_t max_sweeps, double tol, double limit,
                    double share, double horizon, std::optional<double> schmidt,
                    const std::vector<double>& settling, const std::vector<double>& diffusion,
                    const std::array<std::int32_t, 3>& surface_codes) {
    solver_setup setup = read_solver_setup("TransportSolver2", s, dx_cell, dy_cell, dx_face,
                                           dy_face, cfl, upwind, max_sweeps, tol, limit, share,
                                           horizon, schmidt, settling, diffusion, surface_codes);
    new (self) TransportSolver2(s, setup.spacing, setup.keys, std::move(setup.physics));
}

// A face field's array of one of the conditions' attributes, refused unless
// it is a C-contiguous array of the dtype and shape the contract names.
template <typename A>
A face_array(nb::handle owner, const char* name, std::array<std::size_t, 2> shape, int k,
             const char* dtype) {
    A array;
    const nb::object value = nb::getattr(owner, name);
    if (!nb::try_cast<A>(value, array, false) || array.shape(0) != shape[0] ||
        array.shape(1) != shape[1]) {
        throw nb::type_error(("class " + std::to_string(k) + ": " + name +
                              " must be a C-contiguous " + dtype + " array of shape (" +
                              std::to_string(shape[0]) + ", " + std::to_string(shape[1]) + ")")
                                 .c_str());
    }
    require_aligned(array.data(), alignof(std::remove_cv_t<typename A::Scalar>), "set_interval");
    return array;
}

// The arrays of one call to set_interval, kept alive while the core reads them.
struct interval_arrays {
    std::vector<Matrix> doubles;
    std::vector<IntMatrix> ints;
    std::vector<BoolMatrix> bools;
};

std::span<const double> keep(interval_arrays& keep, Matrix a) {
    keep.doubles.push_back(a);
    return span_of(keep.doubles.back());
}
std::span<const std::int32_t> keep(interval_arrays& keep, IntMatrix a) {
    keep.ints.push_back(a);
    return span_of(keep.ints.back());
}
std::span<const bool> keep(interval_arrays& keep, BoolMatrix a) {
    keep.bools.push_back(a);
    return span_of(keep.bools.back());
}

cfdcore::transport::conditions<2> read_conditions(nb::handle faces, int k, const Grid2& g,
                                                  interval_arrays& arrays) {
    const auto u = face_shape(g, 0);
    const auto v = face_shape(g, 1);
    cfdcore::transport::conditions<2> c;
    c.inflow = {keep(arrays, face_array<Matrix>(faces, "inflow_u", u, k, "float64")),
                keep(arrays, face_array<Matrix>(faces, "inflow_v", v, k, "float64"))};
    c.deposition = {keep(arrays, face_array<Matrix>(faces, "deposition_u", u, k, "float64")),
                    keep(arrays, face_array<Matrix>(faces, "deposition_v", v, k, "float64"))};
    c.surface = {keep(arrays, face_array<IntMatrix>(faces, "surface_u", u, k, "int32")),
                 keep(arrays, face_array<IntMatrix>(faces, "surface_v", v, k, "int32"))};
    c.settling = keep(arrays, face_array<BoolMatrix>(faces, "settling_v", v, k, "bool"));
    const nb::object vent_u = nb::getattr(faces, "vent_u");
    const nb::object vent_v = nb::getattr(faces, "vent_v");
    if (vent_u.is_none() != vent_v.is_none()) {
        throw nb::value_error(
            ("class " + std::to_string(k) + ": vent_u and vent_v are given together or not")
                .c_str());
    }
    if (!vent_u.is_none()) {
        c.vents = {keep(arrays, face_array<BoolMatrix>(faces, "vent_u", u, k, "bool")),
                   keep(arrays, face_array<BoolMatrix>(faces, "vent_v", v, k, "bool"))};
    }
    for (nb::handle pair : nb::getattr(faces, "recirculation")) {
        cfdcore::transport::recirculation<2> r;
        r.intake = {keep(arrays, face_array<BoolMatrix>(pair, "intake_u", u, k, "bool")),
                    keep(arrays, face_array<BoolMatrix>(pair, "intake_v", v, k, "bool"))};
        r.discharge = {keep(arrays, face_array<BoolMatrix>(pair, "discharge_u", u, k, "bool")),
                       keep(arrays, face_array<BoolMatrix>(pair, "discharge_v", v, k, "bool"))};
        r.passed = nb::cast<double>(nb::getattr(pair, "passed"));
        c.recirculations.push_back(r);
    }
    return c;
}

// A two-dimensional float64 array of a stated shape, refused as a TypeError otherwise.
Matrix shaped(nb::handle value, std::array<std::size_t, 2> shape, const std::string& what,
              bool convert) {
    Matrix array;
    if (!nb::try_cast<Matrix>(value, array, convert) || array.shape(0) != shape[0] ||
        array.shape(1) != shape[1]) {
        throw nb::type_error((what + " must be a C-contiguous float64 array of shape (" +
                              std::to_string(shape[0]) + ", " + std::to_string(shape[1]) + ")")
                                 .c_str());
    }
    require_aligned(array.data(), alignof(double), "set_interval");
    return array;
}

// An interval's conditions and fixed inputs, read as set_interval takes them,
// their arrays kept alive in `arrays` while the core reads them.
struct interval_read {
    std::vector<cfdcore::transport::conditions<2>> each;
    cfdcore::transport::interval_input<2> input;
};

interval_read read_interval(const Grid2& g, std::size_t n, nb::handle conditions, nb::handle u,
                            nb::handle v, nb::handle v_ext, nb::handle eddy_viscosity,
                            nb::handle sources, interval_arrays& arrays) {
    const std::array<std::size_t, 2> cells{g.cells(1), g.cells(0)};
    std::vector<cfdcore::transport::conditions<2>> each;
    for (nb::handle faces : conditions) {
        each.push_back(read_conditions(faces, static_cast<int>(each.size()), g, arrays));
    }
    if (each.size() != n) {
        throw nb::value_error("set_interval: the conditions are not one per class");
    }
    cfdcore::transport::interval_input<2> input;
    input.faces = {keep(arrays, shaped(u, face_shape(g, 0), "faces.u", false)),
                   keep(arrays, shaped(v, face_shape(g, 1), "faces.v", false))};
    if (!v_ext.is_none()) {
        for (nb::handle drift : v_ext) {
            if (drift.is_none()) {
                input.drift.push_back({});
                continue;
            }
            input.drift.push_back(
                {keep(arrays,
                      shaped(nb::getattr(drift, "u"), face_shape(g, 0), "v_ext.u", false)),
                 keep(arrays,
                      shaped(nb::getattr(drift, "v"), face_shape(g, 1), "v_ext.v", false))});
        }
        if (input.drift.size() != n) {
            throw nb::value_error("set_interval: v_ext holds one drift (or None) per class");
        }
    }
    if (!eddy_viscosity.is_none()) {
        input.eddy_viscosity = keep(arrays, shaped(eddy_viscosity, cells, "eddy_viscosity", false));
    }
    if (!sources.is_none()) {
        for (nb::handle rate : sources) {
            if (rate.is_none()) {
                input.sources.push_back({});
                continue;
            }
            // `np.asarray(sources, dtype=np.float64)`: another dtype is converted.
            input.sources.push_back(keep(arrays, shaped(rate, cells, "sources", true)));
        }
        if (input.sources.size() != n) {
            throw nb::value_error("set_interval: sources holds one array (or None) per class");
        }
    }
    return {std::move(each), std::move(input)};
}

void transport_set_interval(TransportSolver2& t, nb::handle conditions, nb::handle u,
                            nb::handle v, nb::handle v_ext, nb::handle eddy_viscosity,
                            nb::handle sources) {
    interval_arrays arrays;
    const interval_read read = read_interval(t.scheme->mesh(), t.solver.classes(), conditions, u,
                                             v, v_ext, eddy_viscosity, sources, arrays);
    const refusal r = t.solver.set_interval(read.each, read.input);
    if (!r.ok()) {
        raise_refusal(r);
    }
}

// Whether a value is a bool of any kind: Python's, NumPy's or a 0-d bool array.
bool is_bool(nb::handle value) {
    const nb::module_ numpy = nb::module_::import_("numpy");
    // A 0-d bool array is neither a bool nor an np.bool_, and would step with
    // dt = 1.0, so the array's dtype kind decides.
    return nb::isinstance<bool>(value) ||
           PyObject_IsInstance(value.ptr(), numpy.attr("bool_").ptr()) == 1 ||
           (PyObject_IsInstance(value.ptr(), numpy.attr("ndarray").ptr()) == 1 &&
            nb::cast<std::string>(nb::getattr(nb::getattr(value, "dtype"), "kind")) == "b");
}

// A number as the reference reads `dt`: a bool of any kind is a TypeError,
// anything else a number converted as `math.isfinite` converts it (a 0-d float
// array and an np.float64 are numbers). `what` names the argument in the refusal.
double read_number(nb::handle value, const char* what) {
    if (is_bool(value)) {
        throw nb::type_error((std::string(what) + " must be a number, not a bool").c_str());
    }
    const double number = PyFloat_AsDouble(value.ptr());
    if (number == -1.0 && PyErr_Occurred() != nullptr) {
        PyErr_Clear();
        throw nb::type_error((std::string(what) + " must be a number").c_str());
    }
    return number;
}

double read_dt(nb::handle dt) { return read_number(dt, "dt"); }

// A time, s: a number that is finite, or NaN where `nan_ok` (a time not yet set).
double read_time(nb::handle value, const char* what, bool nan_ok = false) {
    const double t = read_number(value, what);
    if (!std::isfinite(t) && !(nan_ok && std::isnan(t))) {
        throw nb::value_error((std::string(what) + " must be finite, got " + float_str(t)).c_str());
    }
    return t;
}

// A count: an int (NumPy's integers too) that is not a bool and fits 64 bits.
std::int64_t read_count(nb::handle value, const char* what) {
    if (is_bool(value)) {
        throw nb::type_error((std::string(what) + " must be an int, not a bool").c_str());
    }
    const nb::object index = nb::steal(PyNumber_Index(value.ptr()));
    if (!index.is_valid()) {
        PyErr_Clear();
        throw nb::type_error((std::string(what) + " must be an int").c_str());
    }
    int overflow = 0;
    const long long count = PyLong_AsLongLongAndOverflow(index.ptr(), &overflow);
    if (overflow != 0) {
        throw nb::value_error((std::string(what) + " is out of range").c_str());
    }
    return static_cast<std::int64_t>(count);
}

void transport_set_step(TransportSolver2& t, nb::handle dt) {
    const refusal r = t.solver.set_step(read_dt(dt));
    if (!r.ok()) {
        raise_refusal(r);
    }
}

// The name of a value's type, as `type(value).__name__`.
std::string python_type_name(nb::handle value) {
    const nb::handle type(reinterpret_cast<PyObject*>(Py_TYPE(value.ptr())));
    return nb::cast<std::string>(nb::getattr(type, "__name__"));
}

// `size_class` as the reference reads it (`_check_class`): an int that is not a
// bool, in [0, n). Every entry point that takes a class reads it here.
std::size_t read_class(nb::handle size_class, std::size_t n) {
    if (nb::isinstance<bool>(size_class) || !PyLong_Check(size_class.ptr())) {
        throw nb::type_error(
            ("size_class must be an int, got " + python_type_name(size_class)).c_str());
    }
    // An int too large for a long long reads as -1, which is out of range, and is
    // named as Python prints it; the overflow flag is not needed to tell.
    int overflow = 0;
    const long long k = PyLong_AsLongLongAndOverflow(size_class.ptr(), &overflow);
    if (k < 0 || static_cast<std::size_t>(k) >= n) {
        throw nb::index_error(("size_class " + nb::cast<std::string>(nb::str(size_class)) +
                               " out of range [0, " + std::to_string(n - 1) + "]")
                                  .c_str());
    }
    return static_cast<std::size_t>(k);
}

// A class's accumulated leftover reached the limit (ADR-016). `step` throws it
// and the module raises it as `_cfdcore.ConservationBreach`, which carries the
// four values as attributes; the Python side makes the reference's error of it.
struct conservation_breach : std::exception {
    conservation_breach(std::size_t k, double accumulated, double particles, double bound)
        : size_class(k), leftover(accumulated), supplied(particles), limit(bound) {}
    const char* what() const noexcept override {
        return "the accumulated leftover reached the conservation limit";
    }
    std::size_t size_class;
    double leftover;
    double supplied;
    double limit;
};

// Raises a conservation_breach as the exception type the payload holds, with the
// class, the leftover, the particles supplied and the limit as its attributes.
void translate_breach(const std::exception_ptr& pending, void* payload) {
    try {
        std::rethrow_exception(pending);
    } catch (const conservation_breach& breach) {
        const nb::object type = nb::borrow(static_cast<PyObject*>(payload));
        char message[256];
        std::snprintf(message, sizeof message,
                      "class %zu: the accumulated leftover %.6g reached the conservation limit "
                      "(%g) of the %.6g particles supplied",
                      breach.size_class, breach.leftover, breach.limit, breach.supplied);
        nb::object error = type(message);
        error.attr("size_class") = breach.size_class;
        error.attr("leftover") = breach.leftover;
        error.attr("supplied") = breach.supplied;
        error.attr("limit") = breach.limit;
        PyErr_SetObject(type.ptr(), error.ptr());
    }
}

cfdcore::transport::step_record transport_step(TransportSolver2& t, nb::handle size_class,
                                               const Matrix& c, OutMatrix& out) {
    constexpr std::string_view who = "TransportSolver2.step";
    const Grid2& g = t.scheme->mesh();
    const std::size_t k = read_class(size_class, t.solver.classes());
    require_shape(c, {g.cells(1), g.cells(0)}, who, "c");
    require_shape(out, {g.cells(1), g.cells(0)}, who, "out");
    cfdcore::transport::step_record record;
    const refusal r =
        t.solver.step(k, span_of(c), std::span<double>(out.data(), out.size()), record);
    if (r.code == status::breach) {
        throw conservation_breach(k, record.accumulated, record.supplied, t.limit);
    }
    if (!r.ok()) {
        raise_refusal(r);
    }
    return record;
}

nb::object optional_float(const std::optional<double>& value) {
    return value.has_value() ? nb::object(nb::float_(*value)) : nb::none();
}

// A class's budget as plain data: a new dict of its ten terms by name,
// `initial` and `current` None before the first step, the deposition by the
// slots' names (floor, ceiling, wall, obstacle).
nb::dict budget_terms(const cfdcore::transport::budget& b) {
    nb::dict terms;
    terms["initial"] = optional_float(b.initial);
    terms["inflow"] = nb::float_(b.inflow);
    terms["outflow"] = nb::float_(b.outflow);
    terms["source"] = nb::float_(b.source);
    for (std::size_t slot = 0; slot < cfdcore::transport::slots; ++slot) {
        const std::string_view name = cfdcore::transport::slot_names[slot];
        terms[nb::str(name.data(), name.size())] = nb::float_(b.deposited[slot]);
    }
    terms["current"] = optional_float(b.current);
    terms["leftover"] = nb::float_(b.leftover);
    return terms;
}

// Every class's budget, in class order.
nb::list transport_budgets(const TransportSolver2& t) {
    nb::list out;
    for (std::size_t k = 0; k < t.solver.classes(); ++k) {
        out.append(budget_terms(t.solver.budget_of(k)));
    }
    return out;
}

// The names of a budget's ten terms, in the order `budgets` returns them.
std::vector<std::string> budget_term_names() {
    std::vector<std::string> names = {"initial", "inflow", "outflow", "source"};
    for (const std::string_view name : cfdcore::transport::slot_names) {
        names.emplace_back(name);
    }
    names.insert(names.end(), {"current", "leftover"});
    return names;
}

// Every class's budget read whole from its ten terms by name, as `budgets`
// returns them (`restore_budgets`). What a budget is made of in Python (the reference's
// MassBudget, a checkpoint's unknown leftover) is the caller's to check before
// it hands the terms over.
std::vector<cfdcore::transport::budget> read_budgets(nb::handle budgets, std::size_t n) {
    const std::size_t given = nb::len(budgets);
    if (given != n) {
        throw nb::value_error(("restore_budgets needs one budget per class (" + std::to_string(n) +
                               "), got " + std::to_string(given))
                                  .c_str());
    }
    const std::vector<std::string> names = budget_term_names();
    std::vector<cfdcore::transport::budget> out;
    std::size_t k = 0;
    for (nb::handle each : budgets) {
        const std::string index = std::to_string(k);
        if (!nb::isinstance<nb::dict>(each)) {
            throw nb::type_error(("budget " + index + " must be a dict of its ten terms, got " +
                                  python_type_name(each))
                                     .c_str());
        }
        const nb::dict terms = nb::borrow<nb::dict>(each);
        bool exact = terms.size() == names.size();
        for (const std::string& name : names) {
            exact = exact && terms.contains(name.c_str());
        }
        if (!exact) {
            throw nb::value_error(("budget " + index +
                                   " must hold exactly the terms initial, inflow, outflow, "
                                   "source, floor, ceiling, wall, obstacle, current and leftover")
                                      .c_str());
        }
        // A term is a number (a Python bool is not one); initial and current may be None.
        const auto term = [&](const std::string& name, bool may_be_none) -> std::optional<double> {
            const nb::object value = terms[name.c_str()];
            if (value.is_none() && may_be_none) {
                return std::nullopt;
            }
            double x = 0.0;
            if (value.is_none() || nb::isinstance<bool>(value) || !nb::try_cast<double>(value, x)) {
                throw nb::type_error(
                    ("budget " + index + ": " + name + " must be a number" +
                     (may_be_none ? " or None" : ""))
                        .c_str());
            }
            return x;
        };
        cfdcore::transport::budget c;
        c.initial = term("initial", true);
        c.inflow = *term("inflow", false);
        c.outflow = *term("outflow", false);
        c.source = *term("source", false);
        for (std::size_t slot = 0; slot < cfdcore::transport::slots; ++slot) {
            c.deposited[slot] = *term(std::string(cfdcore::transport::slot_names[slot]), false);
        }
        c.current = term("current", true);
        c.leftover = *term("leftover", false);
        out.push_back(c);
        ++k;
    }
    return out;
}

void transport_restore_budgets(TransportSolver2& t, nb::handle budgets) {
    t.solver.set_budgets(read_budgets(budgets, t.solver.classes()));
}

// A face field of the interval's, as a new array of the face field's shape.
nb::ndarray<nb::numpy, double, nb::ndim<2>> to_numpy2(std::span<const double> values,
                                                       std::array<std::size_t, 2> shape) {
    auto* copy = new double[values.size()];
    std::copy(values.begin(), values.end(), copy);
    nb::capsule owner(copy, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    return nb::ndarray<nb::numpy, double, nb::ndim<2>>(copy, {shape[0], shape[1]}, owner);
}

nb::ndarray<nb::numpy, bool, nb::ndim<1>> masks_to_numpy(std::span<const cfdcore::lanes::mask> m) {
    auto* copy = new bool[m.size()];
    for (std::size_t i = 0; i < m.size(); ++i) {
        copy[i] = m[i] != cfdcore::lanes::none;
    }
    nb::capsule owner(copy, [](void* p) noexcept { delete[] static_cast<bool*>(p); });
    return nb::ndarray<nb::numpy, bool, nb::ndim<1>>(copy, {m.size()}, owner);
}

// The interval's quantities by name (prediction (b)): a face field per class
// and direction, a cell field per class, or a face field per direction.
nb::object transport_quantity(const TransportSolver2& t, const std::string& name,
                              nb::handle size_class, int d) {
    const Grid2& g = t.scheme->mesh();
    const std::array<std::size_t, 2> cells{g.cells(1), g.cells(0)};
    const auto& s = t.solver;
    if (name == "deposition_cell" || name == "source_step") {
        const std::size_t k = read_class(size_class, s.classes());
        const auto values = name == "deposition_cell" ? s.deposition_cell(k) : s.source_step(k);
        if (values.empty()) {
            return nb::none();
        }
        return nb::cast(to_numpy2(values, cells));
    }
    direction(d);
    const auto faces = face_shape(g, d);
    if (name == "face_eddy_viscosity" || name == "unit_conductance") {
        const auto values =
            name == "face_eddy_viscosity" ? s.face_eddy_viscosity(d) : s.unit_conductance(d);
        if (values.empty()) {
            return nb::none();
        }
        return nb::cast(to_numpy2(values, faces));
    }
    const std::size_t k = read_class(size_class, s.classes());
    if (name == "advecting_velocity") {
        return nb::cast(to_numpy2(s.advecting_velocity(k, d), faces));
    }
    if (name == "volume_flux") {
        return nb::cast(to_numpy2(s.volume_flux(k, d), faces));
    }
    if (name == "conductance") {
        return nb::cast(to_numpy2(s.conductance(k, d), faces));
    }
    if (name == "deposition_step") {
        return nb::cast(to_numpy2(s.deposition_step(k, d), faces));
    }
    throw nb::value_error(("no interval quantity named " + name).c_str());
}

nb::tuple transport_masks(const TransportSolver2& t, const std::string& which,
                          nb::handle size_class, int d) {
    direction(d);
    const auto& s = t.solver;
    const std::size_t k = read_class(size_class, s.classes());
    if (which == "domain") {
        return nb::make_tuple(masks_to_numpy(s.domain_mask(k, d, 0)),
                              masks_to_numpy(s.domain_mask(k, d, 1)),
                              masks_to_numpy(s.domain_mask(k, d, 2)),
                              masks_to_numpy(s.domain_mask(k, d, 3)));
    }
    if (which == "vent") {
        return nb::make_tuple(masks_to_numpy(s.vent_mask(k, d, 0)),
                              masks_to_numpy(s.vent_mask(k, d, 1)),
                              masks_to_numpy(s.vent_mask(k, d, 2)),
                              masks_to_numpy(s.vent_mask(k, d, 3)));
    }
    throw nb::value_error("the masks are 'domain' or 'vent'");
}

// The core refuses a row of zero with terms, and terms that are not whole rows,
// with a std::invalid_argument, which nanobind raises as a ValueError.
double buffered_sum(const Vector& values, std::size_t row) {
    require_aligned(values.data(), alignof(double), "buffered_sum");
    return cfdcore::reduce::buffered_sum(as_span(values), row);
}

// --- The march (prompt 91) ---------------------------------------------------

// A read-only, C-contiguous three-dimensional float64 array on the CPU: one cell
// field per class, [classes, ny, nx].
using Cube = nb::ndarray<const double, nb::ndim<3>, nb::c_contig, nb::device::cpu>;
// A read-only, C-contiguous four-dimensional float64 array on the CPU: the
// arrival maps, [events, channels, ny, nx].
using Maps = nb::ndarray<const double, nb::ndim<4>, nb::c_contig, nb::device::cpu>;

// A copy of core values as a new NumPy array of the given shape.
template <typename T>
nb::ndarray<nb::numpy, T> shaped_copy(std::span<const T> values,
                                      const std::vector<std::size_t>& shape) {
    auto* copy = new T[std::max<std::size_t>(values.size(), 1)];
    std::copy(values.begin(), values.end(), copy);
    nb::capsule owner(copy, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    return nb::ndarray<nb::numpy, T>(copy, shape.size(), shape.data(), owner);
}

// Refuses a cube that is not [classes, ny, nx], or an unaligned one.
void require_cube(const Cube& c, std::size_t classes, const Grid2& g, std::string_view who,
                  std::string_view name) {
    if (c.shape(0) != classes || c.shape(1) != g.cells(1) || c.shape(2) != g.cells(0)) {
        throw nb::type_error((std::string(who) + ": " + std::string(name) + " is not [" +
                              std::to_string(classes) + ", " + std::to_string(g.cells(1)) + ", " +
                              std::to_string(g.cells(0)) + "]")
                                 .c_str());
    }
    require_aligned(c.data(), alignof(double), who);
}

std::span<const double> span_of3(const Cube& c) {
    return std::span<const double>(c.data(), c.size());
}

// The sensors' (j, i) cells as cell field indices, refused outside the grid.
std::vector<std::size_t> sensor_indices(const Grid2& g,
                                        const std::vector<std::pair<std::size_t, std::size_t>>& at,
                                        std::string_view who) {
    std::vector<std::size_t> out;
    for (const auto& [j, i] : at) {
        if (j >= g.cells(1) || i >= g.cells(0)) {
            throw nb::value_error((std::string(who) + ": a sensor's cell (" + std::to_string(j) +
                                   ", " + std::to_string(i) + ") is outside the grid")
                                      .c_str());
        }
        out.push_back(j * g.cells(0) + i);
    }
    return out;
}

nb::ndarray<nb::numpy, double> channel_fields(
    const Grid2& g, const Cube& fields, const std::vector<std::vector<std::size_t>>& channels) {
    const cfdcore::channels::set set(channels);
    require_cube(fields, set.size(), g, "channel_fields", "fields");
    std::vector<double> out(fields.size());
    cfdcore::channels::channel_fields(g.fields(), span_of3(fields), set, out);
    return shaped_copy<double>(out, {set.size(), g.cells(1), g.cells(0)});
}

nb::ndarray<nb::numpy, double> sensor_channels(
    const Matrix& readings, const std::vector<std::vector<std::size_t>>& channels) {
    const cfdcore::channels::set set(channels);
    if (readings.shape(0) != set.size()) {
        throw nb::type_error("sensor_channels: readings is not [classes, sensors]");
    }
    require_aligned(readings.data(), alignof(double), "sensor_channels");
    std::vector<double> out(readings.size());
    cfdcore::channels::sensor_channels(span_of(readings), readings.shape(1), set, out);
    return shaped_copy<double>(out, {readings.shape(0), readings.shape(1)});
}

// The arrival maps on a grid, with the channels they read (`ArrivalTracker`).
struct ArrivalMaps2 {
    ArrivalMaps2(const Grid2& g, std::vector<double> thresholds, std::vector<double> onsets,
                 const std::vector<std::vector<std::size_t>>& members)
        : grid(&g),
          channels(members),
          maps(g, std::move(thresholds), std::move(onsets)),
          sums(channels.size() * g.fields().cell_count(), 0.0) {
        if (maps.channels() != channels.size()) {
            throw nb::value_error("ArrivalMaps2: one threshold per channel");
        }
    }
    const Grid2* grid;
    cfdcore::channels::set channels;
    cfdcore::arrival::maps<2> maps;
    std::vector<double> sums;
};

void arrival_update(ArrivalMaps2& a, double t, const Cube& fields) {
    require_cube(fields, a.channels.size(), *a.grid, "ArrivalMaps2.update", "fields");
    if (a.maps.started(t)) {
        cfdcore::channels::channel_fields(a.grid->fields(), span_of3(fields), a.channels, a.sums);
    }
    a.maps.update(t, a.sums);
}

nb::ndarray<nb::numpy, double> arrival_maps(const cfdcore::arrival::maps<2>& m, const Grid2& g) {
    return shaped_copy<double>(m.first_above(), {m.events(), m.channels(), g.cells(1), g.cells(0)});
}

void require_maps(const Maps& first, const cfdcore::arrival::maps<2>& m, const Grid2& g,
                  std::string_view who) {
    if (first.shape(0) != m.events() || first.shape(1) != m.channels() ||
        first.shape(2) != g.cells(1) || first.shape(3) != g.cells(0)) {
        throw nb::type_error((std::string(who) + ": first_above is not [events, channels, ny, nx]")
                                 .c_str());
    }
    require_aligned(first.data(), alignof(double), who);
}

// The recovery's keys from Python: the thresholds, and a dict of the target, the
// defined flags and the hold (None for none).
cfdcore::recovery::keys recovery_keys(const std::vector<double>& thresholds, nb::handle spec) {
    cfdcore::recovery::keys k;
    k.thresholds = thresholds;
    k.target = nb::cast<std::vector<double>>(spec["target"]);
    k.defined = nb::cast<std::vector<bool>>(spec["defined"]);
    const nb::object hold = spec["hold"];
    if (!hold.is_none()) {
        k.hold = nb::cast<double>(hold);
    }
    return k;
}

// A recovery tracker's state as `RecoveryTracker.state` names and shapes it.
nb::dict recovery_state(const cfdcore::recovery::tracker<2>& r, const Grid2& g) {
    nb::dict out;
    const std::size_t c = r.classes();
    out["last_above"] = shaped_copy<double>(r.last_above(), {c, g.cells(1), g.cells(0)});
    out["sensor_time"] = shaped_copy<double>(r.sensor_time(), {c, r.sensors()});
    out["mean_time"] = shaped_copy<double>(r.mean_time(), {c});
    out["class_since"] = shaped_copy<double>(r.class_since(), {c});
    const double room = r.room_since();
    const double t = r.time();
    out["room_since"] = shaped_copy<double>(std::span<const double>(&room, 1), {});
    out["t"] = shaped_copy<double>(std::span<const double>(&t, 1), {});
    return out;
}

// One array of a recovery state dict, of the length the tracker holds.
Array state_array(nb::handle state, const char* name, std::size_t length, std::string_view who) {
    Array a;
    if (!nb::try_cast<Array>(state[name], a, false) || a.size() != length) {
        throw nb::type_error((std::string(who) + ": the recovery state's " + name +
                              " is not a C-contiguous float64 array of " +
                              std::to_string(length) + " values")
                                 .c_str());
    }
    require_aligned(a.data(), alignof(double), who);
    return a;
}

struct recovery_arrays {
    Array last_above, sensor_time, mean_time, class_since, room_since, t;
};

recovery_arrays read_recovery(nb::handle state, const cfdcore::recovery::tracker<2>& r,
                              const Grid2& g, std::string_view who) {
    const std::size_t c = r.classes();
    return {state_array(state, "last_above", c * g.fields().cell_count(), who),
            state_array(state, "sensor_time", c * r.sensors(), who),
            state_array(state, "mean_time", c, who),
            state_array(state, "class_since", c, who),
            state_array(state, "room_since", 1, who),
            state_array(state, "t", 1, who)};
}

std::span<const double> span_of_array(const Array& a) {
    return std::span<const double>(a.data(), a.size());
}

// The recovery tracker on a grid, with the channels and sensors it reads
// (`RecoveryTracker`).
struct Recovery2 {
    Recovery2(const Grid2& g, const std::vector<double>& thresholds, nb::handle spec,
              const std::vector<std::pair<std::size_t, std::size_t>>& cells,
              const std::vector<std::vector<std::size_t>>& members)
        : grid(&g),
          channels(members),
          sensors(sensor_indices(g, cells, "Recovery2")),
          tracker(g, recovery_keys(thresholds, spec), sensors.size()),
          sums(channels.size() * g.fields().cell_count(), 0.0),
          readings(channels.size() * sensors.size(), 0.0),
          channel_readings(channels.size() * sensors.size(), 0.0) {
        if (tracker.classes() != channels.size()) {
            throw nb::value_error("Recovery2: one threshold per channel");
        }
    }
    const Grid2* grid;
    cfdcore::channels::set channels;
    std::vector<std::size_t> sensors;
    cfdcore::recovery::tracker<2> tracker;
    std::vector<double> sums;
    std::vector<double> readings;
    std::vector<double> channel_readings;
};

void recovery_update(Recovery2& r, double t, const Cube& fields, const std::vector<double>& means) {
    require_cube(fields, r.channels.size(), *r.grid, "Recovery2.update", "fields");
    if (means.size() != r.channels.size()) {
        throw nb::value_error("Recovery2.update: means hold one value per class");
    }
    const auto all = span_of3(fields);
    cfdcore::channels::channel_fields(r.grid->fields(), all, r.channels, r.sums);
    cfdcore::channels::sensor_readings(all, r.grid->fields().cell_count(), r.sensors, r.readings);
    cfdcore::channels::sensor_channels(r.readings, r.sensors.size(), r.channels,
                                       r.channel_readings);
    r.tracker.update(t, r.sums, r.readings, r.channel_readings, means);
}

void recovery_restore(Recovery2& r, nb::handle state) {
    const recovery_arrays a = read_recovery(state, r.tracker, *r.grid, "Recovery2.restore");
    r.tracker.restore(span_of_array(a.last_above), span_of_array(a.sensor_time),
                      span_of_array(a.mean_time), span_of_array(a.class_since),
                      a.room_since.data()[0], a.t.data()[0]);
}

using March2Run = cfdcore::march::run<2>;

// The run with the scheme it was built on (kept alive by the binding).
struct March2 {
    const Scheme2* scheme;
    std::unique_ptr<March2Run> run;
    double limit;
};

void march_init(March2* self, const Scheme2& s, const Vector& dx_cell, const Vector& dy_cell,
                const Vector& dx_face, const Vector& dy_face, double cfl, bool upwind,
                std::int64_t max_sweeps, double tol, double limit, double share, double horizon,
                std::optional<double> schmidt, const std::vector<double>& settling,
                const std::vector<double>& diffusion,
                const std::array<std::int32_t, 3>& surface_codes,
                const std::vector<std::vector<std::size_t>>& channels,
                const std::vector<double>& thresholds,
                const std::vector<std::pair<std::size_t, std::size_t>>& sensors,
                const std::vector<double>& onsets, nb::handle recovery) {
    constexpr std::string_view who = "March2";
    const Grid2& g = s.mesh();
    solver_setup setup = read_solver_setup(who, s, dx_cell, dy_cell, dx_face, dy_face, cfl,
                                           upwind, max_sweeps, tol, limit, share, horizon,
                                           schmidt, settling, diffusion, surface_codes);
    cfdcore::march::monitor_setup monitor;
    monitor.channels = channels;
    monitor.thresholds = thresholds;
    monitor.sensors = sensor_indices(g, sensors, who);
    monitor.onsets = onsets;
    if (!recovery.is_none()) {
        monitor.recovery = recovery_keys(thresholds, recovery);
    }
    new (self) March2{&s,
                      std::make_unique<March2Run>(s, setup.spacing, setup.keys,
                                                  std::move(setup.physics), std::move(monitor)),
                      limit};
}

void march_restore(March2& m, const Cube& fields, nb::handle budgets, nb::handle step,
                   nb::handle t, const Maps& first_above, nb::handle arrival_t,
                   nb::handle recovery) {
    constexpr std::string_view who = "March2.restore";
    const Grid2& g = m.scheme->mesh();
    auto& r = *m.run;
    require_cube(fields, r.classes(), g, who, "fields");
    require_maps(first_above, r.maps(), g, who);
    cfdcore::march::resident state;
    state.fields = span_of3(fields);
    state.budgets = read_budgets(budgets, r.classes());
    state.step = read_count(step, "step");
    if (state.step < 0) {
        throw nb::value_error("March2.restore: step is negative");
    }
    state.t = read_time(t, "t");
    state.first_above = std::span<const double>(first_above.data(), first_above.size());
    // NaN before the maps' first update.
    state.arrival_t = read_time(arrival_t, "arrival_t", true);
    std::optional<recovery_arrays> kept;
    if (r.has_recovery()) {
        if (recovery.is_none()) {
            throw nb::value_error("March2.restore: a run with a recovery needs its state");
        }
        kept = read_recovery(recovery, *r.tracker(), g, who);
        state.last_above = span_of_array(kept->last_above);
        state.sensor_time = span_of_array(kept->sensor_time);
        state.mean_time = span_of_array(kept->mean_time);
        state.class_since = span_of_array(kept->class_since);
        state.room_since = kept->room_since.data()[0];
        state.recovery_t = kept->t.data()[0];
    } else if (!recovery.is_none()) {
        throw nb::value_error("March2.restore: recovery state given to a run without a recovery");
    }
    r.restore(state);
}

void march_interval(March2& m, nb::handle conditions, nb::handle u, nb::handle v,
                    nb::handle v_ext, nb::handle eddy_viscosity, nb::handle sources,
                    nb::handle dt, nb::handle steps, nb::handle t_start, nb::handle t_end,
                    nb::handle first) {
    const std::int64_t n_steps = read_count(steps, "steps");
    const std::int64_t first_step = read_count(first, "first");
    const double start = read_time(t_start, "t_start");
    const double end = read_time(t_end, "t_end");
    if (n_steps < 1 || first_step < 0 || first_step >= n_steps) {
        throw nb::value_error("March2.interval: the first step is not in [0, steps)");
    }
    interval_arrays arrays;
    const interval_read read = read_interval(m.scheme->mesh(), m.run->classes(), conditions, u,
                                             v, v_ext, eddy_viscosity, sources, arrays);
    const cfdcore::march::plan p{start, end, read_dt(dt), n_steps, first_step};
    const refusal r = m.run->interval(read.each, read.input, p);
    if (!r.ok()) {
        raise_refusal(r);
    }
}

const char* stop_name(cfdcore::march::stop s) {
    switch (s) {
        case cfdcore::march::stop::count:
            return "count";
        case cfdcore::march::stop::interval_end:
            return "interval";
        case cfdcore::march::stop::held:
            return "held";
        case cfdcore::march::stop::breach:
            return "breach";
        case cfdcore::march::stop::refused:
            return "refused";
    }
    return "count";
}

nb::dict march_steps(March2& m, nb::handle steps_asked) {
    const std::int64_t n = read_count(steps_asked, "n");
    if (n < 1) {
        throw nb::value_error("March2.march: n is at least 1");
    }
    auto& r = *m.run;
    const std::size_t rows_n = std::min(static_cast<std::size_t>(n), r.remaining());
    const std::size_t c = r.classes();
    const std::size_t s = r.sensors();
    std::vector<std::int64_t> step(rows_n);
    std::vector<double> t(rows_n);
    std::vector<double> budgets(rows_n * c * cfdcore::march::budget_terms);
    std::vector<double> readings(rows_n * c * s);
    std::vector<double> channels(rows_n * c * s);
    std::vector<double> means(r.has_recovery() ? rows_n * c : 0);
    std::vector<std::int64_t> sweeps(rows_n * c);
    std::vector<std::uint8_t> converged(rows_n * c);
    const cfdcore::march::rows out{step, t, budgets, readings, channels, means, sweeps, converged};
    const cfdcore::march::result done = r.march(static_cast<std::size_t>(n), out);
    if (done.reason == cfdcore::march::stop::refused) {
        raise_refusal(done.refusal);
    }
    const std::size_t m_rows = done.steps;
    nb::dict result;
    result["step"] = shaped_copy<std::int64_t>(std::span<const std::int64_t>(step).first(m_rows),
                                               {m_rows});
    result["t"] = shaped_copy<double>(std::span<const double>(t).first(m_rows), {m_rows});
    result["budgets"] = shaped_copy<double>(
        std::span<const double>(budgets).first(m_rows * c * cfdcore::march::budget_terms),
        {m_rows, c, cfdcore::march::budget_terms});
    result["readings"] = shaped_copy<double>(
        std::span<const double>(readings).first(m_rows * c * s), {m_rows, c, s});
    result["channels"] = shaped_copy<double>(
        std::span<const double>(channels).first(m_rows * c * s), {m_rows, c, s});
    result["means"] = r.has_recovery()
                          ? nb::cast(shaped_copy<double>(
                                std::span<const double>(means).first(m_rows * c), {m_rows, c}))
                          : nb::none();
    result["sweeps"] = shaped_copy<std::int64_t>(
        std::span<const std::int64_t>(sweeps).first(m_rows * c), {m_rows, c});
    std::vector<bool> flags(converged.begin(), converged.begin() + m_rows * c);
    auto* copy = new bool[std::max<std::size_t>(flags.size(), 1)];
    std::copy(flags.begin(), flags.end(), copy);
    nb::capsule owner(copy, [](void* p) noexcept { delete[] static_cast<bool*>(p); });
    const std::array<std::size_t, 2> shape{m_rows, c};
    result["converged"] = nb::ndarray<nb::numpy, bool>(copy, 2, shape.data(), owner);
    result["stopped"] = stop_name(done.reason);
    if (done.reason == cfdcore::march::stop::breach) {
        nb::dict breach;
        breach["size_class"] = done.refusal.size_class;
        breach["leftover"] = done.record.accumulated;
        breach["supplied"] = done.record.supplied;
        breach["limit"] = m.limit;
        result["breach"] = breach;
    } else {
        result["breach"] = nb::none();
    }
    return result;
}

nb::list march_budgets(const March2& m) {
    nb::list out;
    for (std::size_t k = 0; k < m.run->classes(); ++k) {
        out.append(budget_terms(m.run->budget_of(k)));
    }
    return out;
}

nb::dict march_arrival(const March2& m) {
    nb::dict out;
    out["first_above"] = arrival_maps(m.run->maps(), m.scheme->mesh());
    out["arrival_t"] = m.run->maps().time();
    return out;
}

nb::object march_recovery(const March2& m) {
    const auto* r = m.run->tracker();
    return r == nullptr ? nb::none() : nb::object(recovery_state(*r, m.scheme->mesh()));
}

// --- The momentum predictor --------------------------------------------------

using Predictor2 = cfdcore::momentum::predictor<2>;
using cfdcore::momentum::component_setup;
using cfdcore::momentum::pair_setup;
using cfdcore::momentum::viscosity_status;

// A read-only, C-contiguous, one-dimensional bool array on the CPU.
using BoolVector = nb::ndarray<const bool, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

[[noreturn]] void refuse_type(const std::string& message) { throw nb::type_error(message.c_str()); }
[[noreturn]] void refuse_value(const std::string& message) {
    throw nb::value_error(message.c_str());
}

// A shape as Python prints it: "(30, 81)".
std::string shape_text(std::array<std::size_t, 2> shape) {
    return "(" + std::to_string(shape[0]) + ", " + std::to_string(shape[1]) + ")";
}
std::string shape_text(const Array& a) {
    nb::list sizes;
    for (std::size_t i = 0; i < a.ndim(); ++i) {
        sizes.append(a.shape(i));
    }
    return nb::cast<std::string>(nb::str(nb::tuple(sizes)));
}

bool has_shape(const Array& a, std::array<std::size_t, 2> shape) {
    return a.ndim() == 2 && a.shape(0) == shape[0] && a.shape(1) == shape[1];
}

// A float64 array argument as the reference reads it (`_check_mu_eff`,
// `_check_wall_mu`): anything but a float64 ndarray is a TypeError naming what
// it is, another shape a ValueError naming both.
Array read_viscosity(nb::handle value, const std::string& name,
                     std::array<std::size_t, 2> shape) {
    const nb::module_ numpy = nb::module_::import_("numpy");
    if (!nb::isinstance(value, numpy.attr("ndarray"))) {
        refuse_type(name + " must be a float64 ndarray, got " + python_type_name(value));
    }
    const nb::object dtype = value.attr("dtype");
    if (!dtype.equal(numpy.attr("float64"))) {
        refuse_type(name + " must be a float64 ndarray, got " +
                    nb::cast<std::string>(nb::str(dtype)));
    }
    Array array;
    // A non-contiguous array, which the reference reads, the core does not.
    if (!nb::try_cast<Array>(value, array, false)) {
        refuse_type(name + " must be a C-contiguous float64 ndarray");
    }
    if (!has_shape(array, shape)) {
        refuse_value("expected " + name + " of shape " + shape_text(shape) + ", got " +
                     shape_text(array));
    }
    require_aligned(array.data(), alignof(double), "MomentumPredictor2");
    return array;
}

// One edge's tangential condition: (is_dirichlet, value, wall_distance).
cfdcore::momentum::edge_condition read_edge(nb::handle edge, std::size_t n, const char* name,
                                            std::vector<BoolVector>& flags,
                                            std::vector<Vector>& values) {
    const std::string who = std::string("MomentumPredictor2: ") + name;
    BoolVector known;
    Vector value;
    cfdcore::momentum::edge_condition condition;
    if (!nb::try_cast<BoolVector>(edge[0], known, false) || known.shape(0) != n ||
        !nb::try_cast<Vector>(edge[1], value, false) || value.shape(0) != n) {
        refuse_type(who + " is not (is_dirichlet, value, wall_distance) with bool and float64 "
                          "arrays of " +
                    std::to_string(n) + " entries");
    }
    require_aligned(value.data(), alignof(double), who);
    condition.wall_distance = nb::cast<double>(edge[2]);
    flags.push_back(known);
    values.push_back(value);
    condition.is_dirichlet = std::span<const bool>(flags.back().data(), n);
    condition.value = as_span(values.back());
    return condition;
}

void momentum_init(Predictor2* self, const Grid2& g, const Vector& dx_cell, const Vector& dy_cell,
                   const Vector& dx_face, const Vector& dy_face, double rho, double mu,
                   double alpha, std::int64_t sweeps, nb::handle tangential,
                   const std::optional<BoolMatrix>& held_u,
                   const std::optional<BoolMatrix>& held_v) {
    constexpr std::string_view who = "MomentumPredictor2";
    for (const Vector* v : {&dx_cell, &dy_cell, &dx_face, &dy_face}) {
        require_aligned(v->data(), alignof(double), who);
    }
    if (nb::len(tangential) != 4) {
        refuse_type("MomentumPredictor2: tangential holds the bottom, top, left and right edges");
    }
    std::vector<BoolVector> flags;
    std::vector<Vector> values;
    flags.reserve(4);
    values.reserve(4);
    cfdcore::momentum::predictor_input<2> input;
    input.cell_widths = {as_span(dx_cell), as_span(dy_cell)};
    input.face_widths = {as_span(dx_face), as_span(dy_face)};
    const std::size_t along_x = g.cells(0) + 1;
    const std::size_t along_y = g.cells(1) + 1;
    // Each component's one pair, with the other direction.
    input.low[0][0] = read_edge(tangential[0], along_x, "bottom", flags, values);
    input.low[1][0] = read_edge(tangential[2], along_y, "left", flags, values);
    input.high[0][0] = read_edge(tangential[1], along_x, "top", flags, values);
    input.high[1][0] = read_edge(tangential[3], along_y, "right", flags, values);
    const std::array<const std::optional<BoolMatrix>*, 2> held{&held_u, &held_v};
    for (int c = 0; c < 2; ++c) {
        if (*held[c]) {
            require_shape(**held[c], face_shape(g, c), who, c == 0 ? "held_u" : "held_v");
            input.held[c] = span_of(**held[c]);
        }
    }
    input.rho = rho;
    input.mu = mu;
    input.alpha = alpha;
    input.sweeps = sweeps;
    // The core's reason, the reference's message among them, is a ValueError.
    new (self) Predictor2(g, input);
}

// The field's refusals in the reference's order (`_check_mu_eff`), or its span.
std::span<const double> read_mu_eff(const Predictor2& m, nb::handle value, Array& keep) {
    const Grid2& g = m.mesh();
    keep = read_viscosity(value, "mu_eff", {g.cells(1), g.cells(0)});
    const std::span<const double> field = span_of_array(keep);
    switch (m.check_viscosity(field)) {
        case viscosity_status::not_finite:
            refuse_value("mu_eff must be finite in every non-SOLID cell");
        case viscosity_status::not_positive:
            refuse_value("mu_eff must be positive in every non-SOLID cell");
        case viscosity_status::ok:
            break;
    }
    return field;
}

// `wall_mu`'s refusals in the reference's order (`_check_wall_mu`), or its spans.
cfdcore::momentum::per_pair<std::span<const double>, 2> read_wall_mu(
    const Predictor2& m, nb::handle value, std::array<Array, 2>& keep) {
    if (!nb::isinstance<nb::dict>(value)) {
        refuse_type("wall_mu must be a dict with keys 'u' and 'v', got " +
                    python_type_name(value));
    }
    const nb::dict fields = nb::borrow<nb::dict>(value);
    if (nb::len(fields) != 2 || !fields.contains("u") || !fields.contains("v")) {
        const nb::object names = nb::module_::import_("builtins").attr("sorted")(fields);
        refuse_value("wall_mu must have keys 'u' and 'v', got " +
                     nb::cast<std::string>(nb::str(names)));
    }
    const Grid2& g = m.mesh();
    cfdcore::momentum::per_pair<std::span<const double>, 2> spans;
    for (int c = 0; c < 2; ++c) {
        const char* key = c == 0 ? "u" : "v";
        const std::string name = std::string("wall_mu['") + key + "']";
        keep[c] = read_viscosity(fields[key], name, {g.cells(1) + 1, g.cells(0) + 1});
        spans[c][0] = span_of_array(keep[c]);
        switch (m.check_wall_viscosity(c, 0, spans[c][0])) {
            case viscosity_status::not_finite:
                refuse_value(name + " must be finite at every wall face");
            case viscosity_status::not_positive:
                refuse_value(name + " must be positive at every wall face");
            case viscosity_status::ok:
                break;
        }
    }
    return spans;
}

// The viscosities of one call, empty where not given, and the arrays they
// view, kept alive while the core reads them.
struct viscosity_arrays {
    Array field;
    std::array<Array, 2> walls;
};
struct viscosities {
    std::span<const double> field;
    cfdcore::momentum::per_pair<std::span<const double>, 2> walls;
};

viscosities read_viscosities(const Predictor2& m, nb::handle mu_eff, nb::handle wall_mu,
                             viscosity_arrays& arrays) {
    viscosities mu;
    if (!mu_eff.is_none()) {
        mu.field = read_mu_eff(m, mu_eff, arrays.field);
    }
    if (!wall_mu.is_none()) {
        mu.walls = read_wall_mu(m, wall_mu, arrays.walls);
    }
    return mu;
}

// A new [rows, columns] float64 NumPy array for an output, its values not set.
nb::ndarray<nb::numpy, double> new_matrix(std::array<std::size_t, 2> shape) {
    auto* data = new double[shape[0] * shape[1]];
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    return nb::ndarray<nb::numpy, double>(data, 2, shape.data(), owner);
}

// The fields' shapes, refused as the reference refuses them (`_check_shapes`, `predict`).
void check_fields(const Predictor2& m, const Array& u, const Array& v, const Array& p) {
    const Grid2& g = m.mesh();
    const auto u_shape = face_shape(g, 0);
    const auto v_shape = face_shape(g, 1);
    const std::array<std::size_t, 2> p_shape{g.cells(1), g.cells(0)};
    if (!has_shape(u, u_shape) || !has_shape(v, v_shape)) {
        refuse_value("expected staggered shapes u " + shape_text(u_shape) + " and v " +
                     shape_text(v_shape) + ", got u " + shape_text(u) + " and v " +
                     shape_text(v));
    }
    if (!has_shape(p, p_shape)) {
        refuse_value("expected p of shape " + shape_text(p_shape) + ", got " + shape_text(p));
    }
    for (const Array* a : {&u, &v, &p}) {
        require_aligned(a->data(), alignof(double), "MomentumPredictor2");
    }
}

nb::tuple momentum_predict(Predictor2& m, const Array& u, const Array& v, const Array& p,
                           nb::handle mu_eff, nb::handle wall_mu, nb::handle transient) {
    check_fields(m, u, v, p);
    const auto u_shape = face_shape(m.mesh(), 0);
    const auto v_shape = face_shape(m.mesh(), 1);
    viscosity_arrays arrays;
    const viscosities mu = read_viscosities(m, mu_eff, wall_mu, arrays);
    if (!transient.is_none()) {
        PyErr_SetString(PyExc_NotImplementedError,
                        "the core's predictor is the steady path: a transient prediction stays "
                        "on the reference (ADR-017 F)");
        throw nb::python_error();
    }
    auto u_star = new_matrix(u_shape);
    auto v_star = new_matrix(v_shape);
    auto a_p_u = new_matrix(u_shape);
    auto a_p_v = new_matrix(v_shape);
    const auto out = [](nb::ndarray<nb::numpy, double>& a) {
        return std::span<double>(a.data(), a.size());
    };
    m.predict({span_of_array(u), span_of_array(v)}, span_of_array(p), mu.field, mu.walls,
              {out(u_star), out(v_star)}, {out(a_p_u), out(a_p_v)});
    return nb::make_tuple(u_star, v_star, a_p_u, a_p_v);
}

nb::dict momentum_stencil_viscosity(Predictor2& m, nb::handle mu_eff) {
    Array keep;
    const std::span<const double> field = read_mu_eff(m, mu_eff, keep);
    const Grid2& g = m.mesh();
    auto u = new_matrix({g.cells(1) + 1, g.cells(0) + 1});
    auto v = new_matrix({g.cells(1) + 1, g.cells(0) + 1});
    m.stencil_viscosity(field, {{{std::span<double>(u.data(), u.size())},
                                 {std::span<double>(v.data(), v.size())}}});
    nb::dict out;
    out["u"] = u;
    out["v"] = v;
    return out;
}

nb::dict momentum_coefficients(Predictor2& m, int component, const Array& u, const Array& v,
                               const Array& p, nb::handle mu_eff, nb::handle wall_mu) {
    const int c = direction(component);
    check_fields(m, u, v, p);
    viscosity_arrays arrays;
    const viscosities mu = read_viscosities(m, mu_eff, wall_mu, arrays);
    const auto shape = face_shape(m.mesh(), c);
    const std::vector<std::size_t> sizes{shape[0], shape[1]};
    auto a_p = new_matrix(shape);
    const std::span<double> diagonal(a_p.data(), a_p.size());
    m.equations(c, {span_of_array(u), span_of_array(v)}, span_of_array(p), mu.field, mu.walls,
                diagonal);
    nb::dict out;
    const std::array<const char*, 4> names{"a_s_plus", "a_s_minus", "a_t_plus", "a_t_minus"};
    for (int which = 0; which < 4; ++which) {
        out[names[which]] = shaped_copy<double>(m.neighbour(which).first(a_p.size()), sizes);
    }
    out["a_p"] = a_p;
    return out;
}

// A mask field as 0 and 1; a 2 is a mask that is neither all ones nor none,
// which no set-up should leave.
nb::object mask_field(const std::vector<cfdcore::lanes::mask>& masks,
                      const std::vector<std::size_t>& shape) {
    std::vector<std::uint8_t> bytes(masks.size());
    for (std::size_t i = 0; i < masks.size(); ++i) {
        bytes[i] = masks[i] == cfdcore::lanes::all ? 1 : 2;
        if (masks[i] == cfdcore::lanes::none) {
            bytes[i] = 0;
        }
    }
    return nb::cast(shaped_copy<std::uint8_t>(bytes, shape));
}

// The value a (name, pointer) table holds under a name, null for none.
template <typename Table>
auto found(const Table& table, const std::string& name) -> decltype(table[0].second) {
    for (const auto& [key, value] : table) {
        if (name == key) {
            return value;
        }
    }
    return nullptr;
}

// A quantity of one component's set-up by its name in `component_setup` or in its
// one pair's: a set of weights as a list of 1-D arrays, a field as a matrix.
nb::object momentum_fixed(const Predictor2& m, const std::string& name, int component) {
    const int c = direction(component);
    const component_setup<2>& o = m.setup(c);
    const pair_setup<2>& w = o.pairs[0];
    const Grid2& g = m.mesh();
    const std::vector<std::size_t> cells{g.cells(1), g.cells(0)};
    const std::vector<std::size_t> corners{g.cells(1) + 1, g.cells(0) + 1};
    const auto face = face_shape(g, c);
    const std::vector<std::size_t> faces{face[0], face[1]};
    const auto list_of = [](const auto& arrays) {
        nb::list out;
        for (const auto& a : arrays) {
            out.append(to_numpy(a));
        }
        return out;
    };
    using three = std::array<std::vector<double>, 3>;
    using two = std::array<std::vector<double>, 2>;
    using line = std::vector<double>;
    using masks = std::vector<cfdcore::lanes::mask>;
    const std::array<std::pair<const char*, const three*>, 6> threes{
        {{"std_pos", &o.std_pos},
         {"std_neg", &o.std_neg},
         {"alt_pos", &o.alt_pos},
         {"alt_neg", &o.alt_neg},
         {"t_pos", &w.t_pos},
         {"t_neg", &w.t_neg}}};
    const std::array<std::pair<const char*, const two*>, 2> twos{
        {{"wall_pos", &w.wall_pos}, {"wall_neg", &w.wall_neg}}};
    const std::array<std::pair<const char*, const line*>, 3> lines{
        {{"harmonic_low", &w.harmonic_low},
         {"harmonic_high", &w.harmonic_high},
         {"mean_high", &o.mean_high}}};
    const std::array<std::pair<const char*, const line*>, 1> cell_fields{
        {{"conductance_s", &o.conductance_s}}};
    const std::array<std::pair<const char*, const line*>, 2> corner_fields{
        {{"conductance_t", &w.conductance_t}, {"reach", &w.reach}}};
    const std::array<std::pair<const char*, const masks*>, 2> cell_masks{
        {{"blocked_pos", &o.blocked_pos}, {"blocked_neg", &o.blocked_neg}}};
    const std::array<std::pair<const char*, const masks*>, 6> corner_masks{
        {{"wall_face", &w.wall_face},
         {"takes_wall", &w.takes_wall},
         {"conducts", &w.conducts},
         {"far_wall_low", &w.far_wall_low},
         {"far_wall_high", &w.far_wall_high},
         {"wall_read", &w.wall_read}}};
    const std::array<std::pair<const char*, const masks*>, 3> face_masks{
        {{"unknown", &o.unknown},
         {"couples_high", &w.couples_high},
         {"couples_low", &w.couples_low}}};
    if (const auto* a = found(threes, name)) {
        return list_of(*a);
    }
    if (const auto* a = found(twos, name)) {
        return list_of(*a);
    }
    if (const auto* a = found(lines, name)) {
        return nb::cast(to_numpy(*a));
    }
    if (const auto* a = found(cell_fields, name)) {
        return nb::cast(shaped_copy<double>(*a, cells));
    }
    if (const auto* a = found(corner_fields, name)) {
        return nb::cast(shaped_copy<double>(*a, corners));
    }
    if (const auto* a = found(cell_masks, name)) {
        return mask_field(*a, cells);
    }
    if (const auto* a = found(corner_masks, name)) {
        return mask_field(*a, corners);
    }
    if (const auto* a = found(face_masks, name)) {
        return mask_field(*a, faces);
    }
    refuse_value("MomentumPredictor2.fixed: no quantity is named " + name);
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

    m.def("buffered_sum", &buffered_sum, nb::arg("values").noconvert(), nb::arg("row"),
          "np.sum of a strided 2-D slice whose rows of `row` terms lie one after another\n"
          "in `values`: NumPy's buffered order (reduce.buffered_sum, prompt 90). The order\n"
          "is NumPy's at its default buffer size, np.getbufsize() = 8,192; a program that\n"
          "changes the size changes np.sum's order and no longer equals this. A row of 0 with\n"
          "terms, or terms that are not a whole number of rows, raises ValueError.");

    // The breach a step raises (ADR-016): an exception of the core's own, carrying
    // the class, the leftover, the particles supplied and the limit as attributes.
    // The reference's ConservationBreachError is made of it on the Python side.
    PyObject* breach_type = PyErr_NewExceptionWithDoc(
        "_cfdcore.ConservationBreach",
        "A class's accumulated leftover reached the conservation limit (ADR-016).\n\n"
        "Attributes: size_class, leftover, supplied and limit.",
        PyExc_RuntimeError, nullptr);
    if (breach_type == nullptr) {
        throw nb::python_error();
    }
    m.attr("ConservationBreach") = nb::steal(breach_type);
    // The payload is the type, kept alive by the module's attribute.
    nb::register_exception_translator(translate_breach, breach_type);

    nb::class_<cfdcore::transport::step_record>(
        m, "TransportStep2",
        "What one class step records beside its field and budget: TransportSolver's\n"
        "last_diffusion_sweeps (sweeps), diffusion_converged (converged), last_leftover\n"
        "(leftover) and last_leftover_bound (bound).")
        .def_ro("sweeps", &cfdcore::transport::step_record::sweeps)
        .def_ro("converged", &cfdcore::transport::step_record::converged)
        .def_ro("leftover", &cfdcore::transport::step_record::leftover)
        .def_ro("bound", &cfdcore::transport::step_record::bound);

    nb::class_<TransportSolver2>(
        m, "TransportSolver2",
        "One particle class advanced one step, and its budget (prompt 90): the core's\n"
        "TransportSolver.solve_timestep, bit for bit.\n\n"
        "TransportSolver2(scheme, dx_cell, dy_cell, dx_face, dy_face, cfl, upwind,\n"
        "max_sweeps, tol, limit, share, horizon, schmidt, settling, diffusion,\n"
        "surface_codes): the mesh's spacings, the transport keys and t_end, each class's\n"
        "settling velocity and diffusion coefficient, and the floor, ceiling and wall\n"
        "surface codes. Keeps its scheme alive. set_interval and set_step fix what an\n"
        "interval and its step fix; step advances a class; budgets and restore_budgets\n"
        "read and set every class's budget whole, as plain data: a dict of its ten terms\n"
        "by name. The module imports nothing from the reference.")
        .def("__init__", &transport_init, nb::arg("scheme"), nb::arg("dx_cell").noconvert(),
             nb::arg("dy_cell").noconvert(), nb::arg("dx_face").noconvert(),
             nb::arg("dy_face").noconvert(), nb::arg("cfl"), nb::arg("upwind"),
             nb::arg("max_sweeps"), nb::arg("tol"), nb::arg("limit"), nb::arg("share"),
             nb::arg("horizon"), nb::arg("schmidt").none(), nb::arg("settling"),
             nb::arg("diffusion"), nb::arg("surface_codes"), nb::keep_alive<1, 2>())
        .def_prop_ro("classes", [](const TransportSolver2& t) { return t.solver.classes(); })
        .def("set_interval", &transport_set_interval, nb::arg("conditions"), nb::arg("u"),
             nb::arg("v"), nb::arg("v_ext").none() = nb::none(),
             nb::arg("eddy_viscosity").none() = nb::none(), nb::arg("sources").none() = nb::none(),
             "Fix an interval: each class's ConcentrationFaces (in class order), the air's\n"
             "face velocities u [ny, nx + 1] and v [ny + 1, nx], v_ext (None, or one\n"
             "FaceVelocities or None per class), the eddy viscosity [ny, nx] or None, and\n"
             "sources (None, or one rate [ny, nx] or None per class). Refuses what\n"
             "TransportSolver and solve_timestep refuse, with their ValueError; a wrong\n"
             "shape, dtype or layout is a TypeError.")
        .def("set_step", &transport_set_step, nb::arg("dt"),
             "Fix the interval's step, s: finite and positive (a bool is a TypeError).")
        .def(
            "stable_dt",
            [](const TransportSolver2& t, nb::handle size_class) {
                return t.solver.stable_dt(read_class(size_class, t.solver.classes()));
            },
            nb::arg("size_class"), "The class's stable_dt on the interval set up.")
        .def("step", &transport_step, nb::arg("size_class"), nb::arg("c").noconvert(),
             nb::arg("out").noconvert(),
             "Advance class size_class one step: c [ny, nx] into out [ny, nx] (out may be\n"
             "c), the budget booked. Returns the step's records. Raises what\n"
             "solve_timestep raises: ValueError for a field that is not finite or a step\n"
             "above stable_dt (nothing booked), and after a step that breaches (the budget\n"
             "holds it, out the field) _cfdcore.ConservationBreach, whose attributes are\n"
             "the class, the leftover, the particles supplied and the limit.")
        .def("budgets", &transport_budgets,
             "Every class's budget as a new dict of its ten terms by name: initial, inflow,\n"
             "outflow, source, floor, ceiling, wall, obstacle, current and leftover; initial\n"
             "and current are None before the first step.")
        .def("restore_budgets", &transport_restore_budgets, nb::arg("budgets"),
             "Set every class's budget whole from one dict of its ten terms per class, as\n"
             "budgets returns them (initial and current may be None). A count other than\n"
             "the classes' is a ValueError, a budget that is not a dict or holds a term that\n"
             "is not a number a TypeError, and one that lacks a term or holds another a\n"
             "ValueError.")
        .def("interval_quantity", &transport_quantity, nb::arg("name"), nb::arg("size_class") = 0,
             nb::arg("direction") = 0,
             "A copy of one of the interval's quantities: advecting_velocity, volume_flux,\n"
             "conductance or deposition_step (per class and direction), deposition_cell or\n"
             "source_step (per class; None without a source), face_eddy_viscosity (None on\n"
             "the laminar path) or unit_conductance (per direction).")
        .def(
            "source_total",
            [](const TransportSolver2& t, nb::handle size_class) {
                return t.solver.source_total(read_class(size_class, t.solver.classes()));
            },
            nb::arg("size_class"), "np.sum(rate * volume) of the class's source, 0.0 without one.")
        .def(
            "drawn",
            [](const TransportSolver2& t, nb::handle size_class) {
                const auto d = t.solver.drawn(read_class(size_class, t.solver.classes()));
                return std::vector<double>(d.begin(), d.end());
            },
            nb::arg("size_class"), "Each recirculating intake's drawn flux, in order.")
        .def("masks", &transport_masks, nb::arg("which"), nb::arg("size_class"),
             nb::arg("direction"),
             "The masks the booking reads: 'domain', four bool arrays along the lines of\n"
             "the direction (low face inward, outward, high face inward, outward); or\n"
             "'vent', four bool arrays on the faces (entering from the low side, from the\n"
             "high side, leaving to the high side, to the low side), empty without vents.");

    // --- The march (prompt 91) ----------------------------------------------

    m.def("channel_fields", &channel_fields, nb::arg("grid"), nb::arg("fields").noconvert(),
          nb::arg("channels"),
          "Every channel's field (monitor.channel_sums): fields [classes, ny, nx], channels\n"
          "one list of classes per class; each channel zero, then each member added in\n"
          "turn. Returns a new [classes, ny, nx] array.");
    m.def("sensor_channels", &sensor_channels, nb::arg("readings").noconvert(),
          nb::arg("channels"),
          "Each channel's reading at each sensor (AlertMonitor._sums): readings\n"
          "[classes, sensors] summed as channel_fields sums the fields.");

    nb::class_<ArrivalMaps2>(m, "ArrivalMaps2",
                             "The arrival maps (ArrivalTracker) on a Grid2: ArrivalMaps2(grid,\n"
                             "thresholds, onsets, channels). Keeps its grid alive.")
        .def(nb::init<const Grid2&, std::vector<double>, std::vector<double>,
                      const std::vector<std::vector<std::size_t>>&>(),
             nb::arg("grid"), nb::arg("thresholds"), nb::arg("onsets"), nb::arg("channels"),
             nb::keep_alive<1, 2>())
        .def("update", &arrival_update, nb::arg("t"), nb::arg("fields").noconvert(),
             "ArrivalTracker.update: fields [classes, ny, nx].")
        .def_prop_ro(
            "first_above",
            [](const ArrivalMaps2& a) { return nb::cast(arrival_maps(a.maps, *a.grid)); },
            "[events, channels, ny, nx], the first time at or above (a copy).")
        .def_prop_ro(
            "t", [](const ArrivalMaps2& a) { return a.maps.time(); },
            "The last update's time, NaN before any.")
        .def(
            "restore",
            [](ArrivalMaps2& a, const Maps& first, double t) {
                require_maps(first, a.maps, *a.grid, "ArrivalMaps2.restore");
                a.maps.restore(std::span<const double>(first.data(), first.size()), t);
            },
            nb::arg("first_above").noconvert(), nb::arg("t"),
            "Set the maps and the last update's time (copied).");

    nb::class_<Recovery2>(m, "Recovery2",
                          "The clean-out's recovery tracker (RecoveryTracker) on a Grid2:\n"
                          "Recovery2(grid, thresholds, keys, cells, channels), keys a dict of\n"
                          "target, defined and hold (None for none), cells the sensors' (j, i).\n"
                          "Keeps its grid alive.")
        .def(nb::init<const Grid2&, const std::vector<double>&, nb::handle,
                      const std::vector<std::pair<std::size_t, std::size_t>>&,
                      const std::vector<std::vector<std::size_t>>&>(),
             nb::arg("grid"), nb::arg("thresholds"), nb::arg("keys"), nb::arg("cells"),
             nb::arg("channels"), nb::keep_alive<1, 2>())
        .def("update", &recovery_update, nb::arg("t"), nb::arg("fields").noconvert(),
             nb::arg("means"), "RecoveryTracker.update: fields [classes, ny, nx], the means.")
        .def(
            "held", [](const Recovery2& r) { return r.tracker.held(); },
            "RecoveryTracker.held.")
        .def(
            "state", [](const Recovery2& r) { return recovery_state(r.tracker, *r.grid); },
            "The state as RecoveryTracker.state names and shapes it (new arrays).")
        .def("restore", &recovery_restore, nb::arg("state"),
             "Set the state from a dict of RecoveryTracker.state's arrays (copied).");

    nb::class_<March2>(
        m, "March2",
        "The march with the fields resident (prompt 91; ADR-017 B's run): every class's\n"
        "step in class order, then the readings and channels, the arrival maps and the\n"
        "recovery, as _Run.march steps them, bit for bit.\n\n"
        "March2(scheme, dx_cell, dy_cell, dx_face, dy_face, cfl, upwind, max_sweeps, tol,\n"
        "limit, share, horizon, schmidt, settling, diffusion, surface_codes, channels,\n"
        "thresholds, sensors, onsets, recovery): TransportSolver2's arguments, then the\n"
        "channels (one list of classes per class), the thresholds per class, the sensors'\n"
        "(j, i) cells, the contaminating events' onsets and the recovery's keys (a dict of\n"
        "target, defined and hold, or None). Keeps its scheme alive. restore sets the whole\n"
        "resident state, interval sets an interval and its plan, march steps it. Plain\n"
        "data only; the module imports nothing from the reference.")
        .def("__init__", &march_init, nb::arg("scheme"), nb::arg("dx_cell").noconvert(),
             nb::arg("dy_cell").noconvert(), nb::arg("dx_face").noconvert(),
             nb::arg("dy_face").noconvert(), nb::arg("cfl"), nb::arg("upwind"),
             nb::arg("max_sweeps"), nb::arg("tol"), nb::arg("limit"), nb::arg("share"),
             nb::arg("horizon"), nb::arg("schmidt").none(), nb::arg("settling"),
             nb::arg("diffusion"), nb::arg("surface_codes"), nb::arg("channels"),
             nb::arg("thresholds"), nb::arg("sensors"), nb::arg("onsets"),
             nb::arg("recovery").none(), nb::keep_alive<1, 2>())
        .def_prop_ro("classes", [](const March2& m) { return m.run->classes(); })
        .def_prop_ro("free_area", [](const March2& m) { return m.run->free_area(); },
                     "The non-SOLID cells' area, summed as the reference sums it.")
        .def_prop_ro("step", [](const March2& m) { return m.run->step(); })
        .def_prop_ro("t", [](const March2& m) { return m.run->time(); })
        .def_prop_ro("remaining", [](const March2& m) { return m.run->remaining(); },
                     "The steps left in the interval, 0 without one.")
        .def("restore", &march_restore, nb::arg("fields").noconvert(), nb::arg("budgets"),
             nb::arg("step"), nb::arg("t"), nb::arg("first_above").noconvert(),
             nb::arg("arrival_t"), nb::arg("recovery").none() = nb::none(),
             "Set the resident state: the class fields [classes, ny, nx], every class's\n"
             "budget (TransportSolver2.budgets' dicts), the run's step count and time, the\n"
             "arrival maps [events, channels, ny, nx] and their last update's time (NaN\n"
             "before the first), and with a recovery its state (Recovery2.state's dict).\n"
             "Clears the interval. A bool or a float where a count is read, a negative step\n"
             "or a time that is not finite is refused.")
        .def("interval", &march_interval, nb::arg("conditions"), nb::arg("u"), nb::arg("v"),
             nb::arg("v_ext").none() = nb::none(), nb::arg("eddy_viscosity").none() = nb::none(),
             nb::arg("sources").none() = nb::none(), nb::kw_only(), nb::arg("dt"),
             nb::arg("steps"), nb::arg("t_start"), nb::arg("t_end"), nb::arg("first") = 0,
             "Set an interval: TransportSolver2.set_interval's arguments, then its plan\n"
             "(IntervalPlan's dt, steps, t_start and t_end) and the step within it the\n"
             "march continues from. Refuses what set_interval and set_step refuse; a bool,\n"
             "a float where a count is read or a time that is not finite is refused (a\n"
             "TypeError and a ValueError), as is a first step outside [0, steps).")
        .def("march", &march_steps, nb::arg("n"),
             "At most n steps of the interval (fewer at its end, after the step that holds,\n"
             "or before a step that breaches). Returns a dict: step and t [m]; budgets\n"
             "[m, classes, 10] in budget_terms' order (NaN for a term not set); readings and\n"
             "channels [m, classes, sensors]; means [m, classes] (None without a recovery);\n"
             "sweeps and converged [m, classes]; stopped ('count', 'interval', 'held' or\n"
             "'breach'); breach (None, or a dict of size_class, leftover, supplied and limit).")
        .def(
            "class_fields",
            [](const March2& m) {
                const Grid2& g = m.scheme->mesh();
                return shaped_copy<double>(m.run->fields(),
                                           {m.run->classes(), g.cells(1), g.cells(0)});
            },
            "The class fields, [classes, ny, nx] (a copy).")
        .def("budgets", &march_budgets, "Every class's budget, as TransportSolver2.budgets.")
        .def("arrival", &march_arrival, "A dict of first_above (a copy) and arrival_t.")
        .def("recovery", &march_recovery, "The recovery state (Recovery2.state), or None.")
        .def(
            "held", [](const March2& m) { return m.run->held(); },
            "Whether the clean-out's hold holds.");
    nb::class_<Predictor2>(
        m, "MomentumPredictor2",
        "The staggered momentum predictor on a grid (MomentumPredictor's steady path).\n\n"
        "MomentumPredictor2(grid, dx_cell, dy_cell, dx_face, dy_face, rho, mu, alpha, sweeps,\n"
        "tangential, held_u=None, held_v=None): the mesh's widths, air's density and\n"
        "viscosity, solver.alpha_velocity and momentum_sweeps, the bottom, top, left and\n"
        "right edges' conditions as (is_dirichlet, value, wall_distance) tuples, and the vent\n"
        "faces the sweeps keep (VentFaces.normal_u, normal_v). Fewer than two cells along an\n"
        "axis, a width of another length or fewer than one sweep raise ValueError; an edge or a\n"
        "vent mask of another shape TypeError.")
        .def("__init__", &momentum_init, nb::arg("grid"), nb::arg("dx_cell").noconvert(),
             nb::arg("dy_cell").noconvert(), nb::arg("dx_face").noconvert(),
             nb::arg("dy_face").noconvert(), nb::arg("rho"), nb::arg("mu"), nb::arg("alpha"),
             nb::arg("sweeps"), nb::arg("tangential"), nb::arg("held_u").noconvert() = nb::none(),
             nb::arg("held_v").noconvert() = nb::none(), nb::keep_alive<1, 2>())
        .def("predict", &momentum_predict, nb::arg("u").noconvert(), nb::arg("v").noconvert(),
             nb::arg("p").noconvert(), nb::arg("mu_eff") = nb::none(),
             nb::arg("wall_mu") = nb::none(), nb::arg("transient") = nb::none(),
             "MomentumPredictor.predict on the steady path, bit for bit: (u_star, v_star, a_p_u,\n"
             "a_p_v), new arrays. The reference's refusals are raised with its types and\n"
             "messages; an array that is not C-contiguous float64 is a TypeError, and a\n"
             "transient argument a NotImplementedError (ADR-017 F).")
        .def("stencil_viscosity", &momentum_stencil_viscosity, nb::arg("mu_eff").none(),
             "MomentumPredictor.stencil_viscosity, bit for bit: the dict of \"u\" and \"v\".")
        .def("coefficients", &momentum_coefficients, nb::arg("component"),
             nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("p").noconvert(),
             nb::arg("mu_eff") = nb::none(), nb::arg("wall_mu") = nb::none(),
             "The neighbour coefficients and the diagonal of component 0 (u) or 1 (v), named as\n"
             "MomentumCoefficients names them, for the parity tests.")
        .def("fixed", &momentum_fixed, nb::arg("name"), nb::arg("component"),
             "A quantity computed once at construction, for the parity tests: the weights, the\n"
             "means' fractions, the scalar conductances, the distances and the masks (0 or 1).");
    m.attr("budget_terms") = [] {
        nb::list names;
        for (const std::string_view name : cfdcore::march::budget_term_names) {
            names.append(nb::str(name.data(), name.size()));
        }
        return names;
    }();
}
