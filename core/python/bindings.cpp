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
#include "cfdcore/dense.hpp"
#include "cfdcore/flow.hpp"
#include "cfdcore/grid.hpp"
#include "cfdcore/march.hpp"
#include "cfdcore/momentum.hpp"
#include "cfdcore/multigrid.hpp"
#include "cfdcore/pressure.hpp"
#include "cfdcore/recovery.hpp"
#include "cfdcore/reduce.hpp"
#include "cfdcore/scalar.hpp"
#include "cfdcore/stopping.hpp"
#include "cfdcore/transport.hpp"
#include "cfdcore/turbulence.hpp"
#include "cfdcore/walls.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
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
cfdcore::momentum::edge_condition read_edge(nb::handle edge, std::size_t n,
                                            const std::string& who,
                                            std::vector<BoolVector>& flags,
                                            std::vector<Vector>& values) {
    BoolVector known;
    Vector value;
    cfdcore::momentum::edge_condition condition;
    if (nb::len(edge) != 3 || !nb::try_cast<BoolVector>(edge[0], known, false) ||
        known.shape(0) != n || !nb::try_cast<Vector>(edge[1], value, false) ||
        value.shape(0) != n) {
        refuse_type(who + " is not (is_dirichlet, value, wall_distance) with bool and float64 "
                          "arrays of " +
                    std::to_string(n) + " entries");
    }
    require_aligned(value.data(), alignof(double), who);
    condition.wall_distance = read_number(edge[2], (who + " wall_distance").c_str());
    flags.push_back(known);
    values.push_back(value);
    condition.is_dirichlet = std::span<const bool>(flags.back().data(), n);
    condition.value = as_span(values.back());
    return condition;
}

void momentum_init(Predictor2* self, const Grid2& g, const Vector& dx_cell, const Vector& dy_cell,
                   const Vector& dx_face, const Vector& dy_face, nb::handle rho, nb::handle mu,
                   nb::handle alpha, nb::handle sweeps, nb::handle tangential,
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
    input.low[0][0] = read_edge(tangential[0], along_x, "MomentumPredictor2: bottom", flags, values);
    input.low[1][0] = read_edge(tangential[2], along_y, "MomentumPredictor2: left", flags, values);
    input.high[0][0] = read_edge(tangential[1], along_x, "MomentumPredictor2: top", flags, values);
    input.high[1][0] = read_edge(tangential[3], along_y, "MomentumPredictor2: right", flags, values);
    const std::array<const std::optional<BoolMatrix>*, 2> held{&held_u, &held_v};
    for (int c = 0; c < 2; ++c) {
        if (*held[c]) {
            require_shape(**held[c], face_shape(g, c), who, c == 0 ? "held_u" : "held_v");
            input.held[c] = span_of(**held[c]);
        }
    }
    input.rho = read_number(rho, "rho");
    input.mu = read_number(mu, "mu");
    input.alpha = read_number(alpha, "alpha");
    input.sweeps = read_count(sweeps, "sweeps");
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

// --- The pressure correction ------------------------------------------------

using Corrector2 = cfdcore::pressure::corrector<2>;
using System2 = cfdcore::multigrid::system<2>;
using Hierarchy2 = cfdcore::multigrid::hierarchy<2>;
using Layout2 = cfdcore::layout::layout<2>;
using cfdcore::pressure::preconditioner;
using cfdcore::pressure::solve_keys;
using cfdcore::pressure::solve_result;
using cfdcore::pressure::solve_status;
using NewArray = nb::ndarray<nb::numpy, double>;

// A pressure solve that ended where the reference's own arithmetic raises: a
// coarsest level whose eigenvalues do not converge (LAPACK's error) or a
// division by zero in the conjugate gradients. The module raises it as
// `_cfdcore.PressureBreakdown`, which the Python side maps.
struct pressure_breakdown : std::exception {
    pressure_breakdown(const char* why, std::int64_t done) : reason(why), iterations(done) {}
    const char* what() const noexcept override { return reason; }
    const char* reason;
    std::int64_t iterations;
};

void translate_breakdown(const std::exception_ptr& pending, void* payload) {
    try {
        std::rethrow_exception(pending);
    } catch (const pressure_breakdown& breakdown) {
        const nb::object type = nb::borrow(static_cast<PyObject*>(payload));
        nb::object error = type((std::string("the pressure solve ended: ") + breakdown.reason +
                                 " after " + std::to_string(breakdown.iterations) + " iterations")
                                    .c_str());
        error.attr("reason") = breakdown.reason;
        error.attr("iterations") = breakdown.iterations;
        PyErr_SetObject(type.ptr(), error.ptr());
    }
}

constexpr const char* not_dominant_message =
    "the finest system is not weakly diagonally dominant: a diagonal is below the sum of its "
    "couplings";

void require_solved(solve_status status, std::int64_t iterations) {
    switch (status) {
        case solve_status::ok:
            return;
        case solve_status::not_dominant:
            throw nb::value_error(not_dominant_message);
        case solve_status::not_converged:
            throw pressure_breakdown("not_converged", iterations);
        case solve_status::division_by_zero:
            throw pressure_breakdown("division_by_zero", iterations);
    }
}

preconditioner read_preconditioner(nb::handle value) {
    std::string name;
    if (nb::try_cast<std::string>(value, name)) {
        if (name == "jacobi") {
            return preconditioner::jacobi;
        }
        if (name == "multigrid") {
            return preconditioner::multigrid;
        }
    }
    refuse_value("preconditioner must be 'jacobi' or 'multigrid', got " +
                 nb::cast<std::string>(nb::repr(value)));
}

// The multigrid's keys: (coarsest_cells, sweeps, weight, dominance_tolerance, rcond).
cfdcore::multigrid::keys read_levels(nb::handle levels) {
    if (nb::len(levels) != 5) {
        refuse_type("levels is not (coarsest_cells, sweeps, weight, dominance_tolerance, rcond)");
    }
    cfdcore::multigrid::keys k;
    k.coarsest_cells = read_count(levels[0], "coarsest_cells");
    k.sweeps = read_count(levels[1], "sweeps");
    k.weight = read_number(levels[2], "weight");
    k.dominance_tolerance = read_number(levels[3], "dominance_tolerance");
    k.rcond = read_number(levels[4], "rcond");
    return k;
}

// The conjugate gradients' numbers, refused as `_check_solve_arguments` refuses
// them: a number is a Python int or float (np.float64 is one), a bool is not.
void read_solve_arguments(nb::handle rtol, nb::handle floor, nb::handle max_iter, solve_keys& k) {
    const auto number = [](nb::handle value) {
        return !PyBool_Check(value.ptr()) &&
               (PyLong_Check(value.ptr()) || PyFloat_Check(value.ptr()));
    };
    const auto text = [](nb::handle value) { return nb::cast<std::string>(nb::repr(value)); };
    // An int too large for a double is above rtol's range, as Python compares it,
    // and an OverflowError as floor, which the reference asks `math.isfinite` of.
    const auto as_double = [](nb::handle value, bool overflows) {
        const double x = PyFloat_AsDouble(value.ptr());
        if (x == -1.0 && PyErr_Occurred() != nullptr) {
            if (overflows) {
                throw nb::python_error();
            }
            PyErr_Clear();
            return std::numeric_limits<double>::infinity();
        }
        return x;
    };
    k.rtol = number(rtol) ? as_double(rtol, false) : -1.0;
    if (!(k.rtol >= 0.0 && k.rtol < 1.0)) {
        refuse_value("rtol must be a number in [0, 1), got " + text(rtol));
    }
    k.floor = number(floor) ? as_double(floor, true) : -1.0;
    if (!(std::isfinite(k.floor) && k.floor >= 0.0)) {
        refuse_value("floor must be a finite number of at least 0, got " + text(floor));
    }
    int overflow = 0;
    k.max_iter = PyLong_Check(max_iter.ptr()) && !PyBool_Check(max_iter.ptr())
                     ? PyLong_AsLongLongAndOverflow(max_iter.ptr(), &overflow)
                     : 0;
    if (overflow > 0) {
        // A cap beyond 64 bits is one no solve reaches.
        k.max_iter = std::numeric_limits<std::int64_t>::max();
    }
    if (k.max_iter < 1) {
        refuse_value("max_iter must be a positive integer, got " + text(max_iter));
    }
}

// The five coefficient arrays of a system, refused as `Multigrid` refuses them:
// the diagonal, then the couplings toward east, west, north and south.
struct FivePoint {
    explicit FivePoint(const std::array<const Array*, 5>& arrays)
        : fields(checked(arrays)), system(fields) {
        const std::array<std::vector<double>*, 5> to{&system.centre, &system.high[0],
                                                     &system.low[0], &system.high[1],
                                                     &system.low[1]};
        for (std::size_t a = 0; a < to.size(); ++a) {
            const auto from = span_of_array(*arrays[a]);
            std::copy(from.begin(), from.end(), to[a]->begin());
        }
        zeros.assign(fields.cells(0), 0.0);
    }
    // The system points at the layout beside it.
    FivePoint(const FivePoint&) = delete;
    FivePoint& operator=(const FivePoint&) = delete;
    static Layout2 checked(const std::array<const Array*, 5>& arrays) {
        bool same = arrays[0]->ndim() == 2;
        for (const Array* a : arrays) {
            same = same && has_shape(*a, {arrays[0]->shape(0), arrays[0]->shape(1)});
        }
        if (!same) {
            std::string shapes;
            for (const Array* a : arrays) {
                shapes += (shapes.empty() ? "" : ", ") + shape_text(*a);
            }
            refuse_value("expected five two-dimensional coefficient arrays of one shape, got [" +
                         shapes + "]");
        }
        for (const Array* a : arrays) {
            require_aligned(a->data(), alignof(double), "five-point system");
        }
        if (arrays[0]->size() == 0) {
            refuse_value("a system has at least one cell");
        }
        return Layout2({arrays[0]->shape(1), arrays[0]->shape(0)});
    }
    std::vector<std::size_t> shape() const { return {fields.cells(1), fields.cells(0)}; }
    Layout2 fields;
    System2 system;
    std::vector<double> zeros;
};

std::span<double> out_span(NewArray& a) { return std::span<double>(a.data(), a.size()); }

NewArray new_field(const Layout2& fields) { return new_matrix({fields.cells(1), fields.cells(0)}); }

// A field argument of a layout's shape.
std::span<const double> read_field(const Array& a, const Layout2& fields, const char* name) {
    if (!has_shape(a, {fields.cells(1), fields.cells(0)})) {
        refuse_value(std::string("expected ") + name + " of shape " +
                     shape_text({fields.cells(1), fields.cells(0)}) + ", got " + shape_text(a));
    }
    require_aligned(a.data(), alignof(double), name);
    return span_of_array(a);
}

nb::dict system_arrays(const System2& s) {
    const std::vector<std::size_t> shape{s.fields->cells(1), s.fields->cells(0)};
    nb::dict out;
    const auto field = [&](const std::vector<double>& v) {
        return shaped_copy<double>(std::span<const double>(v), shape);
    };
    out["a_p"] = field(s.centre);
    out["a_e"] = field(s.high[0]);
    out["a_w"] = field(s.low[0]);
    out["a_n"] = field(s.high[1]);
    out["a_s"] = field(s.low[1]);
    return out;
}

// A hierarchy with the finest system it was built from.
struct Multigrid2 {
    Multigrid2(const std::array<const Array*, 5>& arrays, const cfdcore::multigrid::keys& k)
        : finest(arrays), hierarchy(finest.fields, k) {
        switch (hierarchy.build(finest.system)) {
            case cfdcore::multigrid::build_status::not_dominant:
                throw nb::value_error(not_dominant_message);
            case cfdcore::multigrid::build_status::not_converged:
                throw pressure_breakdown("not_converged", 0);
            case cfdcore::multigrid::build_status::ok:
                break;
        }
    }
    FivePoint finest;
    Hierarchy2 hierarchy;
};

nb::list multigrid_levels(const Multigrid2& m) {
    nb::list out;
    for (std::size_t l = 0; l < m.hierarchy.levels(); ++l) {
        const System2& s = m.hierarchy.level_system(l);
        nb::dict level = system_arrays(s);
        // The coarsest level is solved: it has no smoother.
        level["smoother"] = m.hierarchy.smoother(l).empty()
                                ? nb::none()
                                : nb::cast(shaped_copy<double>(
                                      m.hierarchy.smoother(l),
                                      {s.fields->cells(1), s.fields->cells(0)}));
        out.append(level);
    }
    return out;
}

nb::object square(std::span<const double> values) {
    const auto n = static_cast<std::size_t>(std::llround(std::sqrt(values.size())));
    return nb::cast(shaped_copy<double>(values, {n, n}));
}

NewArray five_point_apply(const Array& a_p, const Array& a_e, const Array& a_w, const Array& a_n,
                          const Array& a_s, const Array& x) {
    const FivePoint s({&a_p, &a_e, &a_w, &a_n, &a_s});
    const auto field = read_field(x, s.fields, "x");
    NewArray out = new_field(s.fields);
    cfdcore::multigrid::apply_operator<2>(s.system, s.zeros, field, out_span(out));
    return out;
}

NewArray five_point_smooth(const Array& a_p, const Array& a_e, const Array& a_w, const Array& a_n,
                           const Array& a_s, const Array& smoother, const Array& x,
                           const Array& r) {
    const FivePoint s({&a_p, &a_e, &a_w, &a_n, &a_s});
    const auto by = read_field(smoother, s.fields, "smoother");
    const auto field = read_field(x, s.fields, "x");
    const auto residual = read_field(r, s.fields, "r");
    NewArray out = new_field(s.fields);
    cfdcore::multigrid::smooth<2>(s.system, s.zeros, by, field, residual, out_span(out));
    return out;
}

nb::dict five_point_coarsen(const Array& a_p, const Array& a_e, const Array& a_w, const Array& a_n,
                            const Array& a_s) {
    const FivePoint s({&a_p, &a_e, &a_w, &a_n, &a_s});
    const Layout2 fields = cfdcore::multigrid::coarser(s.fields);
    System2 out(fields);
    cfdcore::multigrid::coarsen<2>(s.system, s.zeros, out);
    return system_arrays(out);
}

NewArray restrict_field(const Matrix& r) {
    require_aligned(r.data(), alignof(double), "restrict");
    if (r.size() == 0) {
        refuse_value("restrict: r has at least one value");
    }
    const Layout2 fine({r.shape(1), r.shape(0)});
    const Layout2 coarse = cfdcore::multigrid::coarser(fine);
    NewArray out = new_field(coarse);
    cfdcore::multigrid::restrict_to<2>(fine, coarse, span_of(r), out_span(out));
    return out;
}

void add_prolonged(OutMatrix& x, const Matrix& z) {
    constexpr std::string_view who = "add_prolonged";
    require_aligned(x.data(), alignof(double), who);
    require_aligned(z.data(), alignof(double), who);
    if (x.size() == 0) {
        refuse_value("add_prolonged: x has at least one value");
    }
    const Layout2 fine({x.shape(1), x.shape(0)});
    const Layout2 coarse = cfdcore::multigrid::coarser(fine);
    require_shape(z, {coarse.cells(1), coarse.cells(0)}, who, "z");
    require_apart(x.data(), x.size() * sizeof(double), z.data(), z.size() * sizeof(double), who);
    cfdcore::multigrid::add_prolonged<2>(fine, coarse, span_of(z),
                                         std::span<double>(x.data(), x.size()));
}

// A square matrix argument's side.
std::size_t read_square(const Matrix& matrix, std::string_view who) {
    require_aligned(matrix.data(), alignof(double), who);
    if (matrix.shape(0) != matrix.shape(1)) {
        refuse_type(std::string(who) + ": the matrix is not square");
    }
    return matrix.shape(0);
}

NewArray pseudo_inverse(const Matrix& matrix, nb::handle rcond) {
    const std::size_t n = read_square(matrix, "pseudo_inverse");
    cfdcore::dense::semidefinite eigen(n);
    NewArray out = new_matrix({n, n});
    if (eigen.pseudo_inverse(span_of(matrix), read_number(rcond, "rcond"), out_span(out)) !=
        cfdcore::dense::eigen_status::ok) {
        throw pressure_breakdown("not_converged", 0);
    }
    return out;
}

OutVector symmetric_eigenvalues(const Matrix& matrix) {
    cfdcore::dense::semidefinite eigen(read_square(matrix, "symmetric_eigenvalues"));
    if (eigen.decompose(span_of(matrix)) != cfdcore::dense::eigen_status::ok) {
        throw pressure_breakdown("not_converged", 0);
    }
    return to_numpy(eigen.eigenvalues());
}

nb::dict solve_numbers(const solve_result& r) {
    nb::dict out;
    out["iterations"] = r.iterations;
    out["reached_cap"] = r.reached_cap;
    out["residual_norm"] = r.residual_norm;
    out["products"] = r.products;
    return out;
}

nb::dict five_point_cg(const Array& a_p, const Array& a_e, const Array& a_w, const Array& a_n,
                       const Array& a_s, const Array& f, nb::handle rtol, nb::handle floor,
                       nb::handle max_iter, nb::handle method, bool project, nb::handle levels) {
    const FivePoint s({&a_p, &a_e, &a_w, &a_n, &a_s});
    const auto right = read_field(f, s.fields, "f");
    solve_keys k;
    read_solve_arguments(rtol, floor, max_iter, k);
    k.method = read_preconditioner(method);
    std::vector<std::size_t> cells(project ? s.fields.cell_count() : 0);
    if (project) {
        k.project = cfdcore::pressure::cells_with_equation(s.system.centre, cells);
    }
    cfdcore::pressure::solver<2> solver(s.fields, k.method == preconditioner::multigrid,
                                        read_levels(levels));
    NewArray x = new_field(s.fields);
    const solve_result result = solver.solve(s.system, right, k, out_span(x));
    require_solved(result.status, result.iterations);
    nb::dict out = solve_numbers(result);
    out["x"] = x;
    return out;
}

NewArray five_point_precondition(const Array& a_p, const Array& a_e, const Array& a_w,
                                 const Array& a_n, const Array& a_s, const Array& r,
                                 nb::handle method, bool project, nb::handle levels) {
    const FivePoint s({&a_p, &a_e, &a_w, &a_n, &a_s});
    const auto residual = read_field(r, s.fields, "r");
    solve_keys k;
    k.max_iter = 1;
    k.method = read_preconditioner(method);
    std::vector<std::size_t> cells(project ? s.fields.cell_count() : 0);
    if (project) {
        k.project = cfdcore::pressure::cells_with_equation(s.system.centre, cells);
    }
    cfdcore::pressure::solver<2> solver(s.fields, k.method == preconditioner::multigrid,
                                        read_levels(levels));
    require_solved(solver.prepare(s.system, k), 0);
    NewArray z = new_field(s.fields);
    solver.precondition(residual, out_span(z));
    return z;
}

// The mean of the cells where `mask` holds removed from them: the closed
// domain's projection on its own, for the parity tests.
NewArray project_mean(const Matrix& x, const BoolMatrix& mask) {
    require_shape(mask, {x.shape(0), x.shape(1)}, "project_mean", "mask");
    require_aligned(x.data(), alignof(double), "project_mean");
    const auto values = span_of(x);
    std::vector<std::size_t> cells;
    for (std::size_t i = 0; i < mask.size(); ++i) {
        if (mask.data()[i]) {
            cells.push_back(i);
        }
    }
    NewArray out = new_matrix({x.shape(0), x.shape(1)});
    std::copy(values.begin(), values.end(), out.data());
    if (!cells.empty()) {
        const double mean = cfdcore::pressure::mean_over(values, cells);
        for (const std::size_t i : cells) {
            out.data()[i] -= mean;
        }
    }
    return out;
}

void corrector_init(Corrector2* self, const Grid2& g, const Vector& dx_cell, const Vector& dy_cell,
                    nb::handle outlets, nb::handle fluid, nb::handle rho, nb::handle alpha,
                    nb::handle rtol, nb::handle floor, nb::handle max_iter, nb::handle method,
                    nb::handle levels) {
    constexpr std::string_view who = "PressureCorrector2";
    for (const Vector* v : {&dx_cell, &dy_cell}) {
        require_aligned(v->data(), alignof(double), who);
    }
    if (nb::len(outlets) != 4) {
        refuse_type("PressureCorrector2: outlets holds the left, right, bottom and top edges");
    }
    std::array<BoolVector, 4> flags;
    const std::array<const char*, 4> names{"left", "right", "bottom", "top"};
    cfdcore::pressure::corrector_input<2> input;
    for (std::size_t e = 0; e < 4; ++e) {
        const std::size_t n = g.cells(e < 2 ? 1 : 0);
        if (!nb::try_cast<BoolVector>(outlets[e], flags[e], false) || flags[e].shape(0) != n) {
            refuse_type(std::string(who) + ": " + names[e] +
                        " outlets are not a bool array of " + std::to_string(n) + " entries");
        }
        const std::span<const bool> edge(flags[e].data(), n);
        (e % 2 == 0 ? input.outlet_low : input.outlet_high)[e / 2] = edge;
    }
    input.cell_widths = {as_span(dx_cell), as_span(dy_cell)};
    input.fluid = static_cast<std::int32_t>(read_count(fluid, "fluid"));
    input.rho = read_number(rho, "rho");
    input.alpha = read_number(alpha, "alpha");
    solve_keys k;
    read_solve_arguments(rtol, floor, max_iter, k);
    input.rtol = k.rtol;
    input.floor = k.floor;
    input.max_iter = k.max_iter;
    input.method = read_preconditioner(method);
    input.levels = read_levels(levels);
    const std::string_view reason = Corrector2::check(g, input);
    if (!reason.empty()) {
        refuse_value(std::string(who) + ": " + std::string(reason));
    }
    // The components no correction can solve, refused as `_check_components` does.
    const auto count = Corrector2::count_components(g, input);
    bool closed = true;
    for (const BoolVector& edge : flags) {
        closed = closed && std::none_of(edge.data(), edge.data() + edge.shape(0),
                                        [](bool outlet) { return outlet; });
    }
    if (closed && count.components > 1) {
        refuse_value("closed domain: the cells with a pressure equation form " +
                     std::to_string(count.components) +
                     " connected components; the pressure correction projects one mean and pins "
                     "one cell, so it solves one component only (ADR-013 D)");
    }
    if (!closed && count.stranded > 0) {
        refuse_value("open domain: " + std::to_string(count.stranded) + " of the " +
                     std::to_string(count.components) +
                     " connected components of the cells with a pressure equation reach no "
                     "pressure outlet; with no p' = 0 face their block of the pressure correction "
                     "is singular. A component an inlet feeds has no outflow to balance its "
                     "supply, so the correction cannot converge; one with no supply converges to a "
                     "level nothing fixes");
    }
    new (self) Corrector2(g, input);
}

// Refuses velocities of another shape, as `_check_shapes` does.
void check_velocities(const Corrector2& c, const Array& u, const Array& v) {
    const auto u_shape = face_shape(c.mesh(), 0);
    const auto v_shape = face_shape(c.mesh(), 1);
    if (!has_shape(u, u_shape) || !has_shape(v, v_shape)) {
        refuse_value("expected staggered shapes u " + shape_text(u_shape) + " and v " +
                     shape_text(v_shape) + ", got u " + shape_text(u) + " and v " +
                     shape_text(v));
    }
    require_aligned(u.data(), alignof(double), "PressureCorrector2");
    require_aligned(v.data(), alignof(double), "PressureCorrector2");
}

// And diagonals, as `coefficients` does.
void check_diagonals(const Corrector2& c, const Array& a_p_u, const Array& a_p_v) {
    const auto u_shape = face_shape(c.mesh(), 0);
    const auto v_shape = face_shape(c.mesh(), 1);
    if (!has_shape(a_p_u, u_shape) || !has_shape(a_p_v, v_shape)) {
        refuse_value("expected diagonals of shapes " + shape_text(u_shape) + " and " +
                     shape_text(v_shape) + ", got " + shape_text(a_p_u) + " and " +
                     shape_text(a_p_v));
    }
    require_aligned(a_p_u.data(), alignof(double), "PressureCorrector2");
    require_aligned(a_p_v.data(), alignof(double), "PressureCorrector2");
}

nb::dict corrector_coefficients(Corrector2& c, const Array& a_p_u, const Array& a_p_v) {
    check_diagonals(c, a_p_u, a_p_v);
    c.assemble({span_of_array(a_p_u), span_of_array(a_p_v)});
    return system_arrays(c.equation());
}

nb::tuple corrector_face_d(Corrector2& c, const Array& a_p_u, const Array& a_p_v) {
    check_diagonals(c, a_p_u, a_p_v);
    c.assemble({span_of_array(a_p_u), span_of_array(a_p_v)});
    const auto faces = [&](int d) {
        const auto shape = face_shape(c.mesh(), d);
        return shaped_copy<double>(c.face_d(d), {shape[0], shape[1]});
    };
    return nb::make_tuple(faces(0), faces(1));
}

NewArray corrector_imbalance(const Corrector2& c, const Array& u, const Array& v) {
    check_velocities(c, u, v);
    NewArray out = new_field(c.mesh().fields());
    c.mass_imbalance({span_of_array(u), span_of_array(v)}, out_span(out));
    return out;
}

NewArray corrector_right_hand_side(Corrector2& c, const Array& u_star, const Array& v_star,
                                   const Array& a_p_u, const Array& a_p_v) {
    check_velocities(c, u_star, v_star);
    check_diagonals(c, a_p_u, a_p_v);
    c.assemble({span_of_array(a_p_u), span_of_array(a_p_v)});
    NewArray out = new_field(c.mesh().fields());
    c.right_hand_side({span_of_array(u_star), span_of_array(v_star)}, out_span(out));
    return out;
}

// The arrays a correction returns, u, v, p and p_prime by name.
struct corrected {
    explicit corrected(const Grid2& g)
        : u(new_matrix(face_shape(g, 0))),
          v(new_matrix(face_shape(g, 1))),
          p(new_matrix({g.cells(1), g.cells(0)})),
          p_prime(new_matrix({g.cells(1), g.cells(0)})) {}
    nb::dict named() const {
        nb::dict out;
        out["u"] = u;
        out["v"] = v;
        out["p"] = p;
        out["p_prime"] = p_prime;
        return out;
    }
    NewArray u;
    NewArray v;
    NewArray p;
    NewArray p_prime;
};

nb::dict corrector_correct(Corrector2& c, const Array& u_star, const Array& v_star,
                           const Array& a_p_u, const Array& a_p_v, const Array& p) {
    check_velocities(c, u_star, v_star);
    const auto pressure = read_field(p, c.mesh().fields(), "p");
    check_diagonals(c, a_p_u, a_p_v);
    corrected out(c.mesh());
    const solve_result result =
        c.correct({span_of_array(u_star), span_of_array(v_star)},
                  {span_of_array(a_p_u), span_of_array(a_p_v)}, pressure,
                  {out_span(out.u), out_span(out.v)}, out_span(out.p), out_span(out.p_prime));
    require_solved(result.status, result.iterations);
    nb::dict named = out.named();
    named.update(solve_numbers(result));
    return named;
}

nb::dict corrector_apply(Corrector2& c, const Array& u_star, const Array& v_star,
                         const Array& a_p_u, const Array& a_p_v, const Array& p,
                         const Array& p_prime) {
    check_velocities(c, u_star, v_star);
    const auto pressure = read_field(p, c.mesh().fields(), "p");
    const auto solved = read_field(p_prime, c.mesh().fields(), "p_prime");
    check_diagonals(c, a_p_u, a_p_v);
    c.assemble({span_of_array(a_p_u), span_of_array(a_p_v)});
    corrected out(c.mesh());
    std::copy(solved.begin(), solved.end(), out.p_prime.data());
    c.apply({span_of_array(u_star), span_of_array(v_star)}, pressure, out_span(out.p_prime),
            {out_span(out.u), out_span(out.v)}, out_span(out.p));
    return out.named();
}

// ---------------------------------------------------------------------------
// The turbulence model and the wall functions (prompt 94)

using Model2 = cfdcore::turbulence::model<2>;
using Boundary2 = cfdcore::walls::boundary<2>;
using cfdcore::turbulence::step_status;

// The last step's records beside the model, as `KEpsilonModel` holds them.
struct KEpsilon2 {
    KEpsilon2(const Grid2& g, const cfdcore::turbulence::model_input<2>& input)
        : model(g, input) {}
    Model2 model;
    cfdcore::turbulence::step_result last{};
};

// A quantity is not positive and finite at a non-SOLID cell after a step
// (REQ-S15). The module raises it as `_cfdcore.PositivityFailure`, carrying the
// numbers the reference's error states; the Python side makes that error of it.
struct positivity_failure : std::exception {
    positivity_failure(const char* name, const cfdcore::turbulence::bad_cells& cells,
                       std::size_t columns)
        : quantity(name), bad(cells), nx(columns) {}
    const char* what() const noexcept override { return quantity; }
    const char* quantity;
    cfdcore::turbulence::bad_cells bad;
    std::size_t nx;
};

void translate_positivity(const std::exception_ptr& pending, void* payload) {
    try {
        std::rethrow_exception(pending);
    } catch (const positivity_failure& failure) {
        const nb::object type = nb::borrow(static_cast<PyObject*>(payload));
        nb::object error = type((std::string(failure.quantity) +
                                 " is not positive and finite at " +
                                 std::to_string(failure.bad.count) + " non-SOLID cell(s)")
                                    .c_str());
        error.attr("quantity") = failure.quantity;
        error.attr("count") = failure.bad.count;
        error.attr("row") = failure.bad.first / failure.nx;
        error.attr("column") = failure.bad.first % failure.nx;
        error.attr("value") = failure.bad.value;
        error.attr("minimum") = failure.bad.least;
        PyErr_SetObject(type.ptr(), error.ptr());
    }
}

// The str() of an attribute of an object: a shape as Python prints it, a dtype's name.
std::string text_of(nb::handle value, const char* attribute) {
    return nb::cast<std::string>(nb::str(nb::getattr(value, attribute)));
}

// An attribute of an object as a C-contiguous float64 array of the [rows, columns]
// the caller has checked, which must not be unaligned.
Array float_array(nb::handle value, const std::string& name) {
    Array array;
    if (!nb::try_cast<Array>(value, array, false)) {
        refuse_type(name + " must be a C-contiguous float64 ndarray");
    }
    require_aligned(array.data(), alignof(double), name);
    return array;
}

std::array<std::size_t, 2> cell_shape(const Grid2& g) { return {g.cells(1), g.cells(0)}; }

// The state's refusals (`_check_state`), or its arrays.
struct state_arrays {
    Array k;
    Array eps;
    Array nu_t;
    cfdcore::turbulence::state spans() const {
        return {span_of_array(k), span_of_array(eps), span_of_array(nu_t)};
    }
};

state_arrays read_state(const Model2& m, nb::handle state) {
    const auto shape = cell_shape(m.mesh());
    const std::array<const char*, 3> names{"k", "eps", "nu_t"};
    state_arrays arrays;
    const std::array<Array*, 3> fields{&arrays.k, &arrays.eps, &arrays.nu_t};
    for (std::size_t i = 0; i < 3; ++i) {
        const nb::object value = nb::getattr(state, names[i]);
        if (text_of(value, "shape") != shape_text(shape)) {
            refuse_value(std::string("state.") + names[i] + " must have shape " +
                         shape_text(shape) + ", got " + text_of(value, "shape"));
        }
        *fields[i] = float_array(value, std::string("state.") + names[i]);
    }
    for (std::size_t i = 0; i < 2; ++i) {
        if (!m.in_range(span_of_array(*fields[i]), true)) {
            refuse_value(std::string("state.") + names[i] +
                         " must be positive and finite in every non-SOLID cell");
        }
    }
    if (!m.in_range(span_of_array(arrays.nu_t), false)) {
        refuse_value("state.nu_t must be non-negative and finite in every non-SOLID cell");
    }
    for (std::size_t i = 0; i < 3; ++i) {
        if (!m.zero_in_solid(span_of_array(*fields[i]))) {
            refuse_value(std::string("state.") + names[i] + " must be zero in SOLID cells");
        }
    }
    return arrays;
}

// The faces' refusals (`_check_faces`), or their arrays.
struct face_arrays {
    Array u;
    Array v;
    std::array<std::span<const double>, 2> spans() const {
        return {span_of_array(u), span_of_array(v)};
    }
};

face_arrays read_faces(const Model2& m, nb::handle faces) {
    const auto u_shape = face_shape(m.mesh(), 0);
    const auto v_shape = face_shape(m.mesh(), 1);
    const nb::object u = nb::getattr(faces, "u");
    const nb::object v = nb::getattr(faces, "v");
    if (text_of(u, "shape") != shape_text(u_shape) || text_of(v, "shape") != shape_text(v_shape)) {
        refuse_value("faces must have shapes u " + shape_text(u_shape) + " and v " +
                     shape_text(v_shape) + ", got u " + text_of(u, "shape") + " and v " +
                     text_of(v, "shape"));
    }
    face_arrays arrays{float_array(u, "faces.u"), float_array(v, "faces.v")};
    if (!cfdcore::turbulence::all_finite(span_of_array(arrays.u)) ||
        !cfdcore::turbulence::all_finite(span_of_array(arrays.v))) {
        refuse_value("faces must be finite");
    }
    return arrays;
}

// The conditions' refusals (`_check_conditions`), in the reference's order, or
// their arrays.
struct condition_arrays {
    std::array<Array, 10> floats;
    std::array<BoolMatrix, 2> flags;
    cfdcore::turbulence::conditions<2> spans() const {
        cfdcore::turbulence::conditions<2> c;
        c.inflow_k = {span_of_array(floats[0]), span_of_array(floats[2])};
        c.inflow_eps = {span_of_array(floats[1]), span_of_array(floats[3])};
        c.low[0][0] = span_of_array(floats[4]);
        c.high[0][0] = span_of_array(floats[5]);
        c.low[1][0] = span_of_array(floats[6]);
        c.high[1][0] = span_of_array(floats[7]);
        c.eps_wall = span_of_array(floats[8]);
        c.production = span_of_array(floats[9]);
        c.eps_held = span_of(flags[0]);
        c.production_given = span_of(flags[1]);
        return c;
    }
};

// The conditions' float arrays by name, in the reference's order of checks
// (`_check_conditions`); `condition_arrays::floats` and the spans of
// `refuse_formed_conditions` are indexed alike.
constexpr std::array<const char*, 10> condition_names{
    "inflow_k_u",
    "inflow_eps_u",
    "inflow_k_v",
    "inflow_eps_v",
    "tangential_bottom",
    "tangential_top",
    "tangential_left",
    "tangential_right",
    "eps_wall",
    "production",
};

// A step's refusal of its conditions (`_check_conditions`), from the core's check,
// in the reference's words; empty where the check passed.
std::string condition_message(const cfdcore::turbulence::condition_check& check) {
    using cfdcore::turbulence::condition_status;
    switch (check.status) {
        case condition_status::ok:
            return {};
        case condition_status::not_finite:
            return std::string("conditions.") + condition_names[check.array] + " must be finite";
        case condition_status::inflow_on_domain:
            return "conditions' inflow k and eps must be positive on the domain faces";
        case condition_status::inflow_on_vents:
            return "conditions' inflow k and eps must be positive on the vent faces";
        case condition_status::eps_wall:
            return "conditions.eps_wall must be positive where eps is held";
        case condition_status::production:
            return "conditions.production must be non-negative where it is given";
    }
    return {};
}

void refuse_condition(const cfdcore::turbulence::condition_check& check) {
    const std::string message = condition_message(check);
    if (!message.empty()) {
        refuse_value(message);
    }
}

// The conditions' range checks, from the inflow's positivity down, in the
// reference's order: what a step refuses of conditions whose arrays are of the
// right shapes and finite.
void refuse_out_of_range(const KEpsilon2& self, const cfdcore::turbulence::conditions<2>& c) {
    refuse_condition(self.model.check_ranges(c));
}

// The refusals of `_check_conditions` that conditions the boundary formed can meet
// (`advance`): a value that is not finite, then the ranges. The arrays' shapes and
// dtypes are the boundary's own, and so are its flags, which name no SOLID cell: it
// builds its wall cells from the non-SOLID ones.
void refuse_formed_conditions(const KEpsilon2& self,
                              const cfdcore::turbulence::conditions<2>& c) {
    refuse_condition(self.model.check_formed(c));
}

condition_arrays read_conditions(const KEpsilon2& self, nb::handle conditions) {
    const Grid2& g = self.model.mesh();
    const std::string u_shape = shape_text(face_shape(g, 0));
    const std::string v_shape = shape_text(face_shape(g, 1));
    const std::string p_shape = shape_text(cell_shape(g));
    const std::string along_x = "(" + std::to_string(g.cells(0) + 1) + ",)";
    const std::string along_y = "(" + std::to_string(g.cells(1) + 1) + ",)";
    const std::array<std::string, 10> shapes{u_shape, u_shape, v_shape, v_shape, along_x,
                                             along_x, along_y, along_y, p_shape, p_shape};
    condition_arrays arrays;
    for (std::size_t i = 0; i < shapes.size(); ++i) {
        const char* const name = condition_names[i];
        const std::string& shape = shapes[i];
        const nb::object value = nb::getattr(conditions, name);
        if (text_of(value, "shape") != shape || text_of(value, "dtype") != "float64") {
            refuse_value(std::string("conditions.") + name + " must be float64 of shape " +
                         shape + ", got " + text_of(value, "dtype") + " " +
                         text_of(value, "shape"));
        }
        arrays.floats[i] = float_array(value, std::string("conditions.") + name);
        if (!cfdcore::turbulence::all_finite(span_of_array(arrays.floats[i]))) {
            refuse_value(std::string("conditions.") + name + " must be finite");
        }
    }
    const std::array<const char*, 2> flags{"eps_held", "production_given"};
    for (std::size_t i = 0; i < flags.size(); ++i) {
        const std::string name = std::string("conditions.") + flags[i];
        const nb::object value = nb::getattr(conditions, flags[i]);
        if (text_of(value, "shape") != p_shape || text_of(value, "dtype") != "bool") {
            refuse_value(name + " must be bool of shape " + p_shape);
        }
        if (!nb::try_cast<BoolMatrix>(value, arrays.flags[i], false)) {
            refuse_type(name + " must be a C-contiguous bool ndarray");
        }
        bool in_solid = false;
        for (std::size_t cell = 0; cell < arrays.flags[i].size(); ++cell) {
            in_solid |= arrays.flags[i].data()[cell] & g.solid(cell);
        }
        if (in_solid) {
            refuse_value(name + " names a SOLID cell");
        }
    }
    refuse_out_of_range(self, arrays.spans());
    return arrays;
}

constexpr const char* nothing_moves_message =
    "no cell moves, so the pseudo-time step has no value; pass a true-time dt";

void model_init(KEpsilon2* self, const Grid2& g, const Vector& dx_cell, const Vector& dy_cell,
                const Vector& dx_face, const Vector& dy_face, nb::handle variant, nb::handle cfl,
                bool upwind, nb::handle max_iter, nb::handle tol, nb::handle nu,
                const std::optional<BoolMatrix>& vent_u, const std::optional<BoolMatrix>& vent_v) {
    constexpr std::string_view who = "KEpsilon2";
    for (const Vector* v : {&dx_cell, &dy_cell, &dx_face, &dy_face}) {
        require_aligned(v->data(), alignof(double), who);
    }
    if (nb::len(variant) != 7) {
        refuse_type("KEpsilon2: variant is (c_mu, c_1, c_2, sigma_k, sigma_e, eta_0, beta)");
    }
    cfdcore::turbulence::model_input<2> input;
    input.cell_widths = {as_span(dx_cell), as_span(dy_cell)};
    input.face_widths = {as_span(dx_face), as_span(dy_face)};
    input.variant.c_mu = read_number(variant[0], "c_mu");
    input.variant.c_1 = read_number(variant[1], "c_1");
    input.variant.c_2 = read_number(variant[2], "c_2");
    input.variant.sigma_k = read_number(variant[3], "sigma_k");
    input.variant.sigma_e = read_number(variant[4], "sigma_e");
    // The reference has R where both constants are there.
    input.variant.has_r = !nb::handle(variant[5]).is_none() && !nb::handle(variant[6]).is_none();
    if (input.variant.has_r) {
        input.variant.eta_0 = read_number(variant[5], "eta_0");
        input.variant.beta = read_number(variant[6], "beta");
    }
    input.cfl = read_number(cfl, "cfl");
    input.upwind = upwind;
    input.max_iter = read_count(max_iter, "max_iter");
    input.tol = read_number(tol, "tol");
    input.nu = read_number(nu, "nu");
    const std::array<const std::optional<BoolMatrix>*, 2> vents{&vent_u, &vent_v};
    for (int d = 0; d < 2; ++d) {
        if (*vents[d]) {
            require_shape(**vents[d], face_shape(g, d), who, d == 0 ? "vent_u" : "vent_v");
            input.vents[d] = span_of(**vents[d]);
        }
    }
    const std::string_view reason = Model2::check(g, input);
    if (!reason.empty()) {
        refuse_value(std::string(who) + ": " + std::string(reason));
    }
    new (self) KEpsilon2(g, input);
}

NewArray copy_field(std::span<const double> values, std::array<std::size_t, 2> shape) {
    NewArray out = new_matrix(shape);
    std::copy(values.begin(), values.end(), out.data());
    return out;
}

NewArray model_eddy_viscosity(const KEpsilon2& self, const Matrix& k, const Matrix& eps) {
    const auto shape = cell_shape(self.model.mesh());
    require_shape(k, shape, "KEpsilon2", "k");
    require_shape(eps, shape, "KEpsilon2", "eps");
    NewArray out = new_matrix(shape);
    self.model.eddy_viscosity(span_of(k), span_of(eps), out_span(out));
    return out;
}

// `KEpsilonModel.state`: checked copies, zero in SOLID cells, and their viscosity.
nb::dict model_state(const KEpsilon2& self, nb::handle k, nb::handle eps) {
    const Model2& m = self.model;
    const auto shape = cell_shape(m.mesh());
    const std::array<const char*, 2> names{"k", "eps"};
    const std::array<nb::handle, 2> values{k, eps};
    std::array<NewArray, 2> copies;
    for (std::size_t i = 0; i < 2; ++i) {
        const Array given = float_array(values[i], names[i]);
        if (!has_shape(given, shape)) {
            refuse_value(std::string("expected ") + names[i] + " of shape " + shape_text(shape) +
                         ", got " + shape_text(given));
        }
        if (!m.in_range(span_of_array(given), true)) {
            refuse_value(std::string(names[i]) +
                         " must be positive and finite in every non-SOLID cell");
        }
        copies[i] = copy_field(span_of_array(given), shape);
        for (std::size_t cell = 0; cell < copies[i].size(); ++cell) {
            if (m.mesh().solid(cell)) {
                copies[i].data()[cell] = 0.0;
            }
        }
    }
    NewArray nu_t = new_matrix(shape);
    m.eddy_viscosity(out_span(copies[0]), out_span(copies[1]), out_span(nu_t));
    nb::dict out;
    out["k"] = copies[0];
    out["eps"] = copies[1];
    out["nu_t"] = nu_t;
    return out;
}

NewArray model_corner_gradient(KEpsilon2& self, int component, const Matrix& velocity,
                               const Vector& low, const Vector& high) {
    const int c = direction(component);
    const Grid2& g = self.model.mesh();
    require_shape(velocity, face_shape(g, c), "KEpsilon2", "velocity");
    for (const Vector* v : {&low, &high}) {
        if (v->shape(0) != g.cells(c) + 1) {
            refuse_type("KEpsilon2: an edge's tangential velocity is not one value per face");
        }
        require_aligned(v->data(), alignof(double), "KEpsilon2");
    }
    NewArray out = new_matrix({g.cells(1) + 1, g.cells(0) + 1});
    self.model.corner_gradient(c, 0, span_of(velocity), as_span(low), as_span(high),
                               out_span(out));
    return out;
}

NewArray model_strain(KEpsilon2& self, nb::handle faces, nb::handle conditions) {
    const face_arrays u = read_faces(self.model, faces);
    const condition_arrays c = read_conditions(self, conditions);
    NewArray out = new_matrix(cell_shape(self.model.mesh()));
    self.model.strain_squared(u.spans(), c.spans(), out_span(out));
    return out;
}

nb::dict model_terms(KEpsilon2& self, nb::handle state, nb::handle faces, nb::handle conditions) {
    Model2& m = self.model;
    const state_arrays s = read_state(m, state);
    const face_arrays u = read_faces(m, faces);
    const condition_arrays c = read_conditions(self, conditions);
    m.terms(s.spans(), u.spans(), c.spans());
    const auto shape = cell_shape(m.mesh());
    nb::dict out;
    out["production"] = copy_field(m.production(), shape);
    out["rng_r"] = copy_field(m.rng_r(), shape);
    out["growth_k"] = copy_field(m.production(), shape);
    out["growth_eps"] = copy_field(m.growth_eps(), shape);
    out["decay_k"] = copy_field(m.decay_k(), shape);
    out["decay_eps"] = copy_field(m.decay_eps(), shape);
    return out;
}

NewArray model_time_step(KEpsilon2& self, nb::handle faces) {
    const face_arrays u = read_faces(self.model, faces);
    if (!self.model.pseudo_time_step(u.spans())) {
        refuse_value(nothing_moves_message);
    }
    return copy_field(self.model.time_step(), cell_shape(self.model.mesh()));
}

NewArray model_advected(KEpsilon2& self, nb::handle faces, const Matrix& q, const Matrix& inflow_u,
                        const Matrix& inflow_v) {
    Model2& m = self.model;
    const face_arrays u = read_faces(m, faces);
    require_shape(q, cell_shape(m.mesh()), "KEpsilon2", "q");
    require_shape(inflow_u, face_shape(m.mesh(), 0), "KEpsilon2", "inflow_u");
    require_shape(inflow_v, face_shape(m.mesh(), 1), "KEpsilon2", "inflow_v");
    if (!m.pseudo_time_step(u.spans())) {
        refuse_value(nothing_moves_message);
    }
    NewArray out = new_matrix(cell_shape(m.mesh()));
    m.advected(span_of(q), {span_of(inflow_u), span_of(inflow_v)}, out_span(out));
    return out;
}

nb::tuple model_explicit_fields(KEpsilon2& self, nb::handle state, nb::handle faces,
                                nb::handle conditions) {
    Model2& m = self.model;
    const state_arrays s = read_state(m, state);
    const face_arrays u = read_faces(m, faces);
    const condition_arrays c = read_conditions(self, conditions);
    if (!m.pseudo_time_step(u.spans())) {
        refuse_value(nothing_moves_message);
    }
    m.terms(s.spans(), u.spans(), c.spans());
    NewArray k_star = new_matrix(cell_shape(m.mesh()));
    NewArray eps_star = new_matrix(cell_shape(m.mesh()));
    m.explicit_fields(s.spans(), c.spans(), out_span(k_star), out_span(eps_star));
    return nb::make_tuple(k_star, eps_star);
}

cfdcore::turbulence::quantity read_quantity(const std::string& name) {
    if (name == "k") {
        return cfdcore::turbulence::quantity::k;
    }
    if (name != "eps") {
        refuse_value("the quantity is 'k' or 'eps'");
    }
    return cfdcore::turbulence::quantity::eps;
}

nb::tuple model_conductances(KEpsilon2& self, const std::string& quantity, const Matrix& nu_t) {
    const Grid2& g = self.model.mesh();
    require_shape(nu_t, cell_shape(g), "KEpsilon2", "nu_t");
    NewArray g_u = new_matrix(face_shape(g, 0));
    NewArray g_v = new_matrix(face_shape(g, 1));
    self.model.conductances(read_quantity(quantity), span_of(nu_t),
                            {out_span(g_u), out_span(g_v)});
    return nb::make_tuple(g_u, g_v);
}

// A step on checked arrays: its fields by name, or the refusal its status names.
nb::dict run_step(KEpsilon2& self, const cfdcore::turbulence::state& s,
                  const std::array<std::span<const double>, 2>& velocity,
                  const cfdcore::turbulence::conditions<2>& c) {
    const Grid2& g = self.model.mesh();
    NewArray k = new_matrix(cell_shape(g));
    NewArray eps = new_matrix(cell_shape(g));
    NewArray nu_t = new_matrix(cell_shape(g));
    const auto result = self.model.step(s, velocity, c, out_span(k), out_span(eps),
                                        out_span(nu_t));
    if (result.status == step_status::nothing_moves) {
        refuse_value(nothing_moves_message);
    }
    self.last = result;
    if (result.status == step_status::k_not_positive) {
        throw positivity_failure("k", result.bad, g.cells(0));
    }
    if (result.status == step_status::eps_not_positive) {
        throw positivity_failure("eps", result.bad, g.cells(0));
    }
    nb::dict out;
    out["k"] = k;
    out["eps"] = eps;
    out["nu_t"] = nu_t;
    return out;
}

nb::dict model_step(KEpsilon2& self, nb::handle state, nb::handle faces, nb::handle conditions,
                    nb::handle dt) {
    const state_arrays s = read_state(self.model, state);
    const face_arrays u = read_faces(self.model, faces);
    const condition_arrays c = read_conditions(self, conditions);
    if (!dt.is_none()) {
        PyErr_SetString(PyExc_NotImplementedError,
                        "the core's k-epsilon step is the local pseudo-time step: a true-time dt "
                        "stays on the reference (ADR-017 F)");
        throw nb::python_error();
    }
    return run_step(self, s.spans(), u.spans(), c.spans());
}

// A step with the conditions the boundary forms of the same state and faces.
nb::dict model_advance(KEpsilon2& self, Boundary2& walls, nb::handle state, nb::handle faces) {
    if (&walls.mesh() != &self.model.mesh()) {
        refuse_value("KEpsilon2: the boundary is on another grid");
    }
    const state_arrays s = read_state(self.model, state);
    const face_arrays u = read_faces(self.model, faces);
    const auto c = walls.conditions(u.spans(), span_of_array(s.k), span_of_array(s.eps));
    // The step makes the check of its conditions whoever formed them, so a state
    // whose wall values underflow is refused as the reference refuses it.
    refuse_formed_conditions(self, c);
    return run_step(self, s.spans(), u.spans(), c);
}

// The positivity test of a step's field alone (`_check_positive`): None, or the
// PositivityFailure a step raises for it.
void model_check_positive(const KEpsilon2& self, const std::string& quantity, const Matrix& q) {
    require_shape(q, cell_shape(self.model.mesh()), "KEpsilon2", "q");
    const char* const name = read_quantity(quantity) == cfdcore::turbulence::quantity::k
                                 ? "k"
                                 : "eps";
    cfdcore::turbulence::bad_cells bad;
    if (self.model.find_bad(span_of(q), bad)) {
        throw positivity_failure(name, bad, self.model.mesh().cells(0));
    }
}

// The eddy viscosity's relaxation (`_turbulence_step`), a new array.
NewArray relax_eddy_viscosity(const Matrix& old, const Matrix& stepped, nb::handle alpha) {
    if (old.shape(0) != stepped.shape(0) || old.shape(1) != stepped.shape(1)) {
        refuse_type("relax_eddy_viscosity: the two fields differ in shape");
    }
    NewArray out = new_matrix({old.shape(0), old.shape(1)});
    cfdcore::turbulence::relax(span_of(old), span_of(stepped), read_number(alpha, "alpha"),
                               out_span(out));
    return out;
}

// The outlets' extrapolation (`_extrapolate_outlets`) on one grid.
using Outlets2 = cfdcore::walls::outlets<2>;

// The outlets beside the grid they were built on.
struct OutletFaces2 {
    OutletFaces2(const Grid2& g, const std::array<std::span<const bool>, 4>& edges)
        : grid(&g), outlets(g, {edges[0], edges[2]}, {edges[1], edges[3]}) {}
    const Grid2* grid;
    Outlets2 outlets;
};

// The four edges' outlet flags, left, right, bottom and top, as the corrector takes
// them, in the order of an outlets' input.
std::array<std::span<const bool>, 4> read_outlet_edges(const Grid2& g, nb::handle outlets,
                                                       std::array<BoolVector, 4>& flags,
                                                       std::string_view who) {
    if (nb::len(outlets) != 4) {
        refuse_type(std::string(who) + ": outlets holds the left, right, bottom and top edges");
    }
    const std::array<const char*, 4> names{"left", "right", "bottom", "top"};
    std::array<std::span<const bool>, 4> edges;
    for (std::size_t e = 0; e < 4; ++e) {
        const std::size_t n = g.cells(e < 2 ? 1 : 0);
        if (!nb::try_cast<BoolVector>(outlets[e], flags[e], false) || flags[e].shape(0) != n) {
            refuse_type(std::string(who) + ": " + names[e] +
                        " outlets are not a bool array of " + std::to_string(n) + " entries");
        }
        edges[e] = std::span<const bool>(flags[e].data(), n);
    }
    return edges;
}

void outlets_init(OutletFaces2* self, const Grid2& g, nb::handle outlets) {
    std::array<BoolVector, 4> flags;
    const auto edges = read_outlet_edges(g, outlets, flags, "OutletFaces2");
    if (g.cells(0) < 2 || g.cells(1) < 2) {
        refuse_value("OutletFaces2: the grid needs two cells along each direction");
    }
    new (self) OutletFaces2(g, edges);
}

// Copies of u and v with their outlet faces written.
nb::tuple outlets_extrapolate(const OutletFaces2& self, const Matrix& u, const Matrix& v) {
    const Grid2& g = *self.grid;
    require_shape(u, face_shape(g, 0), "OutletFaces2", "u");
    require_shape(v, face_shape(g, 1), "OutletFaces2", "v");
    NewArray u_out = new_matrix(face_shape(g, 0));
    NewArray v_out = new_matrix(face_shape(g, 1));
    std::copy(u.data(), u.data() + u.size(), u_out.data());
    std::copy(v.data(), v.data() + v.size(), v_out.data());
    self.outlets.extrapolate({out_span(u_out), out_span(v_out)});
    return nb::make_tuple(u_out, v_out);
}

nb::object masks_matrix(std::span<const cfdcore::lanes::mask> masks,
                        std::array<std::size_t, 2> shape) {
    std::vector<std::uint8_t> bytes(masks.size());
    for (std::size_t i = 0; i < masks.size(); ++i) {
        bytes[i] = masks[i] == cfdcore::lanes::all ? 1 : 2;
        if (masks[i] == cfdcore::lanes::none) {
            bytes[i] = 0;
        }
    }
    return nb::cast(shaped_copy<std::uint8_t>(bytes, {shape[0], shape[1]}));
}

// A (direction, row, column) triple's face in its direction's face field.
std::size_t read_face(const Grid2& g, nb::handle item, int& d, const char* what) {
    d = static_cast<int>(read_count(item[0], what));
    direction(d);
    const auto shape = face_shape(g, d);
    const std::int64_t row = read_count(item[1], what);
    const std::int64_t column = read_count(item[2], what);
    if (row < 0 || column < 0 || static_cast<std::size_t>(row) >= shape[0] ||
        static_cast<std::size_t>(column) >= shape[1]) {
        refuse_value(std::string("TurbulenceBoundary2: ") + what + " is not on the grid");
    }
    return static_cast<std::size_t>(row) * shape[1] + static_cast<std::size_t>(column);
}

std::vector<cfdcore::walls::inlet_face> read_inlets(const Grid2& g, nb::handle items) {
    std::vector<cfdcore::walls::inlet_face> out;
    for (nb::handle item : items) {
        if (nb::len(item) != 7) {
            refuse_type(
                "TurbulenceBoundary2: an inlet's face is (direction, row, column, intensity, "
                "normal speed, dissipation length, width)");
        }
        cfdcore::walls::inlet_face face;
        face.face = read_face(g, item, face.direction, "an inlet's face");
        face.intensity = read_number(item[3], "an inlet's intensity");
        face.normal_speed = read_number(item[4], "an inlet's normal speed");
        face.dissipation_length = read_number(item[5], "an inlet's dissipation length");
        face.width = read_number(item[6], "an inlet's width");
        out.push_back(face);
    }
    return out;
}

cfdcore::walls::wall_law read_law(nb::handle law) {
    if (nb::len(law) != 3) {
        refuse_type("the law of the wall is (kappa, E, floor)");
    }
    return {read_number(law[0], "kappa"), read_number(law[1], "E"), read_number(law[2], "floor")};
}

void boundary_init(Boundary2* self, const Grid2& g, nb::handle wall_cells, nb::handle wall_corners,
                   nb::handle tangential, nb::handle vents, nb::handle edge_inlets,
                   nb::handle vent_inlets, nb::handle exhausts, nb::handle c_mu, nb::handle rho,
                   nb::handle nu, nb::handle law) {
    const std::string who = "TurbulenceBoundary2";
    if (nb::len(wall_cells) != 4 || nb::len(wall_corners) != 4 || nb::len(tangential) != 4) {
        refuse_type(who + ": the edges are the bottom, top, left and right ones");
    }
    const std::array<const char*, 4> edges{"bottom", "top", "left", "right"};
    cfdcore::walls::boundary_input<2> input;
    std::array<BoolVector, 4> cells;
    std::array<BoolVector, 4> corners;
    std::vector<BoolVector> flags;
    std::vector<Vector> values;
    flags.reserve(4);
    values.reserve(4);
    for (std::size_t e = 0; e < 4; ++e) {
        // The bottom and top edges lie along x (component u), the others along y.
        const int c = e < 2 ? 0 : 1;
        const std::size_t n = g.cells(c);
        const std::string name = who + ": " + edges[e];
        if (!nb::try_cast<BoolVector>(wall_cells[e], cells[e], false) || cells[e].shape(0) != n) {
            refuse_type(name + " wall cells are not a bool array of " + std::to_string(n) +
                        " entries");
        }
        if (!nb::try_cast<BoolVector>(wall_corners[e], corners[e], false) ||
            corners[e].shape(0) != n + 1) {
            refuse_type(name + " wall corners are not a bool array of " + std::to_string(n + 1) +
                        " entries");
        }
        const bool high = e % 2 == 1;
        const std::span<const bool> edge_cells(cells[e].data(), n);
        const std::span<const bool> edge_corners(corners[e].data(), n + 1);
        (high ? input.wall_cells_high : input.wall_cells_low)[1 - c] = edge_cells;
        (high ? input.wall_corners_high : input.wall_corners_low)[c][0] = edge_corners;
        (high ? input.high : input.low)[c][0] = read_edge(tangential[e], n + 1, name, flags, values);
    }
    std::array<BoolMatrix, 4> masks;
    if (!vents.is_none()) {
        const std::array<std::size_t, 2> corner_shape{g.cells(1) + 1, g.cells(0) + 1};
        const std::array<std::array<std::size_t, 2>, 4> shapes{face_shape(g, 0), face_shape(g, 1),
                                                               corner_shape, corner_shape};
        if (nb::len(vents) != 4) {
            refuse_type(who + ": vents is (normal_u, normal_v, tangential_u, tangential_v)");
        }
        for (std::size_t i = 0; i < 4; ++i) {
            if (!nb::try_cast<BoolMatrix>(vents[i], masks[i], false)) {
                refuse_type(who + ": a vent mask is not a C-contiguous bool ndarray");
            }
            require_shape(masks[i], shapes[i], who, "a vent mask");
        }
        input.vents = {span_of(masks[0]), span_of(masks[1])};
        input.vent_corners[0][0] = span_of(masks[2]);
        input.vent_corners[1][0] = span_of(masks[3]);
    }
    const auto edge_faces = read_inlets(g, edge_inlets);
    const auto vent_faces = read_inlets(g, vent_inlets);
    std::vector<cfdcore::walls::exhaust_face> exhaust_faces;
    for (nb::handle item : exhausts) {
        if (nb::len(item) != 3) {
            refuse_type(who + ": an exhaust's face is (direction, row, column)");
        }
        cfdcore::walls::exhaust_face face;
        face.face = read_face(g, item, face.direction, "an exhaust's face");
        exhaust_faces.push_back(face);
    }
    input.edge_inlets = edge_faces;
    input.vent_inlets = vent_faces;
    input.exhausts = exhaust_faces;
    input.c_mu = read_number(c_mu, "c_mu");
    input.rho = read_number(rho, "rho");
    input.nu = read_number(nu, "nu");
    input.law = read_law(law);
    const std::string_view reason = Boundary2::check(g, input);
    if (!reason.empty()) {
        refuse_value(who + ": " + std::string(reason));
    }
    new (self) Boundary2(g, input);
}

nb::dict boundary_wall_viscosity(Boundary2& self, const Matrix& k, nb::handle elsewhere) {
    const Grid2& g = self.mesh();
    const std::array<std::size_t, 2> shape{g.cells(1) + 1, g.cells(0) + 1};
    require_shape(k, cell_shape(g), "TurbulenceBoundary2", "k");
    const std::array<const char*, 2> keys{"u", "v"};
    std::array<Matrix, 2> kept;
    nb::dict out;
    std::array<NewArray, 2> fields;
    for (std::size_t c = 0; c < 2; ++c) {
        const nb::object value = nb::borrow(elsewhere[keys[c]]);
        if (!nb::try_cast<Matrix>(value, kept[c], false)) {
            refuse_type(std::string("TurbulenceBoundary2: elsewhere['") + keys[c] +
                        "'] must be a C-contiguous float64 ndarray");
        }
        require_shape(kept[c], shape, "TurbulenceBoundary2", "elsewhere");
        fields[c] = new_matrix(shape);
        out[keys[c]] = fields[c];
    }
    self.wall_viscosity(span_of(k), {{{span_of(kept[0])}, {span_of(kept[1])}}},
                        {{{out_span(fields[0])}, {out_span(fields[1])}}});
    return out;
}

// Refuses faces of another shape; the reference makes no check of its own here.
face_arrays boundary_faces(const Boundary2& self, nb::handle faces) {
    face_arrays arrays{float_array(nb::getattr(faces, "u"), "faces.u"),
                       float_array(nb::getattr(faces, "v"), "faces.v")};
    if (!has_shape(arrays.u, face_shape(self.mesh(), 0)) ||
        !has_shape(arrays.v, face_shape(self.mesh(), 1))) {
        refuse_type("TurbulenceBoundary2: the faces are not the grid's");
    }
    return arrays;
}

nb::dict boundary_conditions(Boundary2& self, nb::handle state, nb::handle faces) {
    const Grid2& g = self.mesh();
    const Array k = float_array(nb::getattr(state, "k"), "state.k");
    const Array eps = float_array(nb::getattr(state, "eps"), "state.eps");
    if (!has_shape(k, cell_shape(g)) || !has_shape(eps, cell_shape(g))) {
        refuse_type("TurbulenceBoundary2: k or eps is not the grid's");
    }
    const face_arrays u = boundary_faces(self, faces);
    const auto c = self.conditions(u.spans(), span_of_array(k), span_of_array(eps));
    const auto u_shape = face_shape(g, 0);
    const auto v_shape = face_shape(g, 1);
    const auto shape = cell_shape(g);
    const auto flags = [&](std::span<const bool> values) {
        return nb::cast(shaped_copy<bool>(values, {shape[0], shape[1]}));
    };
    nb::dict out;
    out["inflow_k_u"] = copy_field(c.inflow_k[0], u_shape);
    out["inflow_k_v"] = copy_field(c.inflow_k[1], v_shape);
    out["inflow_eps_u"] = copy_field(c.inflow_eps[0], u_shape);
    out["inflow_eps_v"] = copy_field(c.inflow_eps[1], v_shape);
    out["tangential_bottom"] = to_numpy(c.low[0][0]);
    out["tangential_top"] = to_numpy(c.high[0][0]);
    out["tangential_left"] = to_numpy(c.low[1][0]);
    out["tangential_right"] = to_numpy(c.high[1][0]);
    out["eps_held"] = flags(c.eps_held);
    out["eps_wall"] = copy_field(c.eps_wall, shape);
    out["production_given"] = flags(c.production_given);
    out["production"] = copy_field(c.production, shape);
    return out;
}

nb::tuple boundary_initial_values(const Boundary2& self) {
    const auto values = self.initial_values();
    if (!values) {
        refuse_value(
            "no velocity inlet admits air, so nothing sets the supply's k and eps: the coupled "
            "solve takes its initial state and condition (e)'s scale from the inlets");
    }
    return nb::make_tuple(values->k, values->eps);
}

// What a stencil's wall faces are, for the parity tests: the mask, and per face
// its corner, its two cells and the distance, in the core's order.
nb::dict boundary_stencil(const Boundary2& self, int component) {
    const Grid2& g = self.mesh();
    const auto& walls = self.stencil(direction(component), 0);
    const auto indices = [](const std::vector<std::size_t>& values) {
        std::vector<std::int64_t> copy(values.begin(), values.end());
        return nb::cast(shaped_copy<std::int64_t>(copy, {copy.size()}));
    };
    nb::dict out;
    out["mask"] = masks_matrix(walls.mask, {g.cells(1) + 1, g.cells(0) + 1});
    out["corner"] = indices(walls.corner);
    out["cell_low"] = indices(walls.cell_low);
    out["cell_high"] = indices(walls.cell_high);
    out["distance"] = to_numpy(walls.distance);
    return out;
}

nb::list boundary_inlet_values(const Boundary2& self) {
    nb::list out;
    for (const auto& value : self.inlet_values()) {
        out.append(nb::make_tuple(value.k, value.eps));
    }
    return out;
}

double log_law_floor(nb::handle kappa, nb::handle e) {
    const double k = read_number(kappa, "kappa");
    const double big_e = read_number(e, "e");
    const auto result = cfdcore::walls::log_law_floor(k, big_e);
    switch (result.status) {
        case cfdcore::walls::floor_status::no_meeting:
            refuse_value("u+ = y+ and u+ = ln(" + float_str(big_e) + " y+) / " + float_str(k) +
                         " do not meet");
        case cfdcore::walls::floor_status::not_settled:
            refuse_value("Newton's method did not settle on the log-law floor");
        case cfdcore::walls::floor_status::ok:
            break;
    }
    return result.y;
}

OutVector wall_viscosity_values(const Vector& k_p, const Vector& y_p, nb::handle c_mu,
                                nb::handle rho, nb::handle nu, nb::handle law) {
    if (k_p.shape(0) != y_p.shape(0)) {
        refuse_type("wall_viscosity: k_p and y_p differ in length");
    }
    require_aligned(k_p.data(), alignof(double), "wall_viscosity");
    require_aligned(y_p.data(), alignof(double), "wall_viscosity");
    std::vector<double> out(k_p.shape(0));
    cfdcore::walls::wall_viscosity(as_span(k_p), as_span(y_p), read_number(c_mu, "c_mu"),
                                   read_number(rho, "rho"), read_number(nu, "nu"), read_law(law),
                                   out);
    return to_numpy(out);
}

nb::tuple inflow_values(nb::handle intensity, nb::handle normal_speed, nb::handle length) {
    const auto values = cfdcore::walls::inflow_values(
        read_number(intensity, "intensity"), read_number(normal_speed, "normal_speed"),
        read_number(length, "dissipation_length"));
    return nb::make_tuple(values.k, values.eps);
}

}  // namespace

// ---------------------------------------------------------------------------
// The stopping rule and the outer iteration (prompt 95)

using Rule = cfdcore::stopping::rule;

// `_check_positive(name, value)`: a bool, a value that is not finite or not
// positive is a ValueError naming it as Python prints it; a value `math.isfinite`
// cannot read is its TypeError.
double read_positive(nb::handle value, const char* name) {
    const std::string shown = nb::cast<std::string>(nb::str(value));
    if (nb::isinstance<bool>(value)) {
        refuse_value(std::string(name) + " must be positive and finite, got " + shown);
    }
    const double number = PyFloat_AsDouble(value.ptr());
    if (number == -1.0 && PyErr_Occurred() != nullptr) {
        PyErr_Clear();
        refuse_type("must be real number, not " + python_type_name(value));
    }
    if (!(std::isfinite(number) && number > 0.0)) {
        refuse_value(std::string(name) + " must be positive and finite, got " + shown);
    }
    return number;
}

// `float(value)`, as the reference converts a step.
double python_float(nb::handle value) { return nb::cast<double>(nb::float_(value)); }

// The rule's keys from the reference's arguments, refused as `ErrorEstimateRule`
// refuses them.
cfdcore::stopping::rule_keys read_rule_keys(nb::handle window, nb::handle velocity_scale,
                                            nb::handle flux_scale,
                                            nb::handle iteration_error_tol,
                                            nb::handle mass_imbalance_tol,
                                            nb::handle viscosity_condition) {
    cfdcore::stopping::rule_keys keys;
    keys.velocity_scale = read_positive(velocity_scale, "velocity_scale");
    keys.flux_scale = read_positive(flux_scale, "flux_scale");
    keys.iteration_error_tol = read_positive(iteration_error_tol, "iteration_error_tol");
    keys.mass_imbalance_tol = read_positive(mass_imbalance_tol, "mass_imbalance_tol");
    if (!nb::isinstance<bool>(viscosity_condition)) {
        refuse_type("viscosity_condition must be a bool, got " +
                    python_type_name(viscosity_condition));
    }
    keys.viscosity_condition = nb::cast<bool>(viscosity_condition);
    const std::int64_t length = read_count(window, "window");
    if (length < 2) {
        refuse_value("window must hold at least two steps, got " + std::to_string(length));
    }
    keys.window = static_cast<std::size_t>(length);
    return keys;
}

// The rule with the readings' histories, as `ErrorEstimateRule` keeps them.
struct StoppingRule2 {
    explicit StoppingRule2(const cfdcore::stopping::rule_keys& keys) : rule(keys) {}
    Rule rule;
    std::vector<double> estimates;
    std::vector<double> viscosity_estimates;
    std::vector<double> viscosity_scales;
};

void rule_init(StoppingRule2* self, nb::handle window, nb::handle velocity_scale,
               nb::handle flux_scale, nb::handle iteration_error_tol,
               nb::handle mass_imbalance_tol, nb::handle viscosity_condition) {
    new (self) StoppingRule2(read_rule_keys(window, velocity_scale, flux_scale,
                                            iteration_error_tol, mass_imbalance_tol,
                                            viscosity_condition));
}

nb::tuple window_of(const Rule& r) {
    nb::list steps;
    nb::list viscosity;
    for (const double s : r.steps()) {
        steps.append(nb::float_(s));
    }
    for (const double s : r.viscosity_steps()) {
        viscosity.append(nb::float_(s));
    }
    return nb::make_tuple(nb::tuple(steps), nb::tuple(viscosity));
}

// A window's steps as `restore` reads them, refused in its order: the length, then
// each value's type and range.
std::vector<double> read_window(nb::handle values, const char* name, std::size_t window) {
    const std::size_t n = nb::len(values);
    if (n > window) {
        refuse_value(std::string(name) + " holds " + std::to_string(n) +
                     " steps, more than RATE_WINDOW (" + std::to_string(window) + ")");
    }
    std::vector<double> out;
    out.reserve(n);
    for (nb::handle value : values) {
        if (nb::isinstance<bool>(value) || !PyFloat_Check(value.ptr())) {
            refuse_value(std::string(name) + " must hold floats, got " +
                         nb::cast<std::string>(nb::repr(value)));
        }
        const double x = PyFloat_AsDouble(value.ptr());
        if (!(std::isfinite(x) && x >= 0.0)) {
            refuse_value(std::string(name) + " must be finite and non-negative");
        }
        out.push_back(x);
    }
    return out;
}

// `restore`'s refusals in its order, then the core's seeding.
void restore_rule(Rule& r, bool histories, nb::handle steps, nb::handle viscosity_steps) {
    if (r.stepped() || histories) {
        refuse_value("restore seeds a new rule; this one has taken steps");
    }
    const std::size_t window = r.keys().window;
    const std::vector<double> velocity = read_window(steps, "steps", window);
    const std::vector<double> viscosity = read_window(viscosity_steps, "viscosity_steps", window);
    const bool on = r.keys().viscosity_condition;
    const std::size_t expected = on ? velocity.size() : 0;
    if (viscosity.size() != expected) {
        refuse_value("viscosity_steps must hold " + std::to_string(expected) +
                     " steps with condition (e) " + (on ? "on" : "off") + ", got " +
                     std::to_string(viscosity.size()));
    }
    if (r.restore(velocity, viscosity) != cfdcore::stopping::restore_status::ok) {
        refuse_value("restore: the core refused a window the checks accepted");
    }
}

void rule_restore(StoppingRule2& self, nb::handle steps, nb::handle viscosity_steps) {
    restore_rule(self.rule, !self.estimates.empty(), steps, viscosity_steps);
}

bool rule_update(StoppingRule2& self, nb::handle step, nb::handle imbalance,
                 nb::handle viscosity_step, nb::handle nu_scale) {
    const bool on = self.rule.keys().viscosity_condition;
    if (viscosity_step.is_none() == on || nu_scale.is_none() == on) {
        refuse_value(
            "viscosity_step and nu_scale are given exactly when the rule applies condition (e) "
            "(ADR-012 E)");
    }
    double scale = 0.0;
    if (!nu_scale.is_none()) {
        // Checked before anything is recorded, so a refused step leaves no trace.
        scale = read_positive(nu_scale, "nu_scale");
    }
    const double velocity = python_float(step);
    const double viscosity = on ? python_float(viscosity_step) : 0.0;
    const bool estimates = self.rule.record(velocity, viscosity, scale);
    const auto& last = self.rule.last();
    self.estimates.push_back(last.estimate);
    if (on) {
        self.viscosity_scales.push_back(last.nu_scale);
        self.viscosity_estimates.push_back(last.viscosity_estimate);
    }
    if (!estimates) {
        return false;
    }
    const nb::object found = imbalance();
    return self.rule.accept({python_float(found.attr("worst")),
                             python_float(found.attr("absolute_sum")),
                             python_float(found.attr("signed_sum"))});
}

double rule_estimate_of(StoppingRule2& self, nb::handle values, nb::handle scale) {
    std::vector<double> steps;
    for (nb::handle value : values) {
        steps.push_back(python_float(value));
    }
    return self.rule.estimate_of(steps, python_float(scale));
}

nb::list list_of(const std::vector<double>& values) {
    nb::list out;
    for (const double x : values) {
        out.append(nb::float_(x));
    }
    return out;
}

// The imbalance summary of a cell field (`_imbalance_summary`'s three readings).
nb::tuple imbalance_summary(const Matrix& imbalance) {
    std::vector<double> magnitude(imbalance.size());
    const auto found = cfdcore::stopping::summarise(span_of(imbalance), magnitude);
    return nb::make_tuple(found.worst, found.absolute_sum, found.signed_sum);
}

// The FLUID cells' indices of a bool mask, ascending.
std::vector<std::size_t> cells_of(const BoolMatrix& mask) {
    std::vector<std::size_t> cells;
    for (std::size_t i = 0; i < mask.size(); ++i) {
        if (mask.data()[i]) {
            cells.push_back(i);
        }
    }
    return cells;
}

double viscosity_scale_of(nb::handle nu, const Matrix& nu_t, const BoolMatrix& fluid) {
    if (nu_t.shape(0) != fluid.shape(0) || nu_t.shape(1) != fluid.shape(1)) {
        refuse_type("viscosity_scale: nu_t and fluid differ in shape");
    }
    const std::vector<std::size_t> cells = cells_of(fluid);
    if (cells.empty()) {
        refuse_value("viscosity_scale: no FLUID cell");
    }
    std::vector<double> scratch(cells.size());
    return cfdcore::stopping::viscosity_scale(read_number(nu, "nu"), span_of(nu_t), cells,
                                              scratch);
}

nb::tuple cell_centres_of(const Grid2& g, const Matrix& u, const Matrix& v) {
    require_shape(u, face_shape(g, 0), "cell_centres", "u");
    require_shape(v, face_shape(g, 1), "cell_centres", "v");
    NewArray u_c = new_matrix(cell_shape(g));
    NewArray v_c = new_matrix(cell_shape(g));
    cfdcore::flow::cell_centres<2>(g.fields(), 0, span_of(u), out_span(u_c));
    cfdcore::flow::cell_centres<2>(g.fields(), 1, span_of(v), out_span(v_c));
    return nb::make_tuple(u_c, v_c);
}

double largest_change_of(const Matrix& now, const Matrix& before, const BoolMatrix& cells) {
    if (now.shape(0) != before.shape(0) || now.shape(1) != before.shape(1) ||
        cells.shape(0) != now.shape(0) || cells.shape(1) != now.shape(1)) {
        refuse_type("largest_change: the arrays differ in shape");
    }
    std::vector<cfdcore::lanes::mask> masks(cells.size());
    for (std::size_t i = 0; i < masks.size(); ++i) {
        masks[i] = cfdcore::lanes::mask_of(cells.data()[i]);
    }
    return cfdcore::flow::largest_change(span_of(now), span_of(before), masks);
}

// Python's `max` of the components' changes, as the outer iteration folds them.
double python_max_of(const std::vector<double>& values) {
    if (values.empty()) {
        refuse_value("python_max: no value");
    }
    return cfdcore::flow::python_max(values);
}

// The coupled prediction's effective viscosity, refused in the reference's words.
NewArray effective_viscosity_of(const Matrix& nu_t, const BoolMatrix& live, nb::handle mu,
                                nb::handle rho) {
    if (nu_t.shape(0) != live.shape(0) || nu_t.shape(1) != live.shape(1)) {
        refuse_type("effective_viscosity: nu_t and live differ in shape");
    }
    std::vector<cfdcore::lanes::mask> masks(live.size());
    for (std::size_t i = 0; i < masks.size(); ++i) {
        masks[i] = cfdcore::lanes::mask_of(live.data()[i]);
    }
    NewArray out = new_matrix({nu_t.shape(0), nu_t.shape(1)});
    bool not_finite = false;
    if (!cfdcore::flow::effective_viscosity(span_of(nu_t), masks, read_number(mu, "mu"),
                                            read_number(rho, "rho"), out_span(out),
                                            not_finite)) {
        refuse_value(not_finite ? "eddy_viscosity must be finite in every non-SOLID cell"
                                : "eddy_viscosity must be non-negative (ADR-012 D)");
    }
    return out;
}

// The solve with its modules, which it keeps alive.
using FlowSolver2 = cfdcore::flow::solver<2>;

struct Flow2 {
    Flow2(const Grid2& g, const cfdcore::flow::modules<2>& m,
          const cfdcore::flow::flow_input<2>& input)
        : grid(&g), coupled(input.mode == cfdcore::flow::viscosity_mode::coupled),
          solver(g, m, input) {}
    const Grid2* grid;
    bool coupled;
    FlowSolver2 solver;
};

void flow_init(Flow2* self, const Grid2& g, Predictor2& predictor, Corrector2& corrector,
               const OutletFaces2& outlets, KEpsilon2* model, Boundary2* walls,
               nb::handle fluid, nb::handle mu, nb::handle rho, nb::handle nu,
               nb::handle alpha_turbulence, nb::handle residual_scale, nb::handle rule,
               const std::optional<Matrix>& mu_eff) {
    constexpr std::string_view who = "Flow2";
    if (outlets.grid != &g) {
        refuse_value("Flow2: the outlets are on another grid");
    }
    cfdcore::flow::modules<2> m;
    m.predictor = &predictor;
    m.corrector = &corrector;
    m.outlets = &outlets.outlets;
    m.model = model == nullptr ? nullptr : &model->model;
    m.walls = walls;
    cfdcore::flow::flow_input<2> input;
    input.fluid = static_cast<std::int32_t>(read_count(fluid, "fluid"));
    input.mu = read_number(mu, "mu");
    input.rho = read_number(rho, "rho");
    input.nu = read_number(nu, "nu");
    input.alpha_turbulence = read_number(alpha_turbulence, "alpha_turbulence");
    input.residual_scale = read_number(residual_scale, "residual_scale");
    input.mode = model != nullptr ? cfdcore::flow::viscosity_mode::coupled
                 : mu_eff        ? cfdcore::flow::viscosity_mode::fixed
                                 : cfdcore::flow::viscosity_mode::laminar;
    if (mu_eff) {
        require_shape(*mu_eff, cell_shape(g), who, "mu_eff");
        input.mu_eff = span_of(*mu_eff);
    }
    const std::string kind = nb::cast<std::string>(nb::str(rule[0]));
    if (kind == "velocity_step") {
        if (nb::len(rule) != 2) {
            refuse_type("Flow2: rule is ('velocity_step', convergence_tol)");
        }
        input.rule = cfdcore::flow::stop_rule::velocity_step;
        input.convergence_tol = read_number(rule[1], "convergence_tol");
    } else if (kind == "error_estimate") {
        if (nb::len(rule) != 6) {
            refuse_type(
                "Flow2: rule is ('error_estimate', window, velocity_scale, flux_scale, "
                "iteration_error_tol, mass_imbalance_tol)");
        }
        input.rule = cfdcore::flow::stop_rule::error_estimate;
        input.keys = read_rule_keys(rule[1], rule[2], rule[3], rule[4], rule[5],
                                    nb::bool_(model != nullptr));
    } else {
        refuse_value("Flow2: the rule is 'velocity_step' or 'error_estimate'");
    }
    const std::string_view reason = FlowSolver2::check(g, m, input);
    if (!reason.empty()) {
        refuse_value(std::string(who) + ": " + std::string(reason));
    }
    new (self) Flow2(g, m, input);
}

void flow_state(Flow2& self, const Matrix& u, const Matrix& v, const Matrix& p,
                const std::optional<Matrix>& k, const std::optional<Matrix>& eps,
                const std::optional<Matrix>& nu_t) {
    const Grid2& g = *self.grid;
    require_shape(u, face_shape(g, 0), "Flow2", "u");
    require_shape(v, face_shape(g, 1), "Flow2", "v");
    require_shape(p, cell_shape(g), "Flow2", "p");
    if (k.has_value() != self.coupled || eps.has_value() != self.coupled ||
        nu_t.has_value() != self.coupled) {
        refuse_value("Flow2: k, eps and nu_t are given exactly when the solve is coupled");
    }
    std::span<const double> turbulence[3];
    const std::array<const std::optional<Matrix>*, 3> fields{&k, &eps, &nu_t};
    for (std::size_t i = 0; i < 3; ++i) {
        if (*fields[i]) {
            require_shape(**fields[i], cell_shape(g), "Flow2", "a turbulence field");
            turbulence[i] = span_of(**fields[i]);
        }
    }
    self.solver.start({span_of(u), span_of(v)}, span_of(p), turbulence[0], turbulence[1],
                      turbulence[2]);
}

void flow_restore(Flow2& self, nb::handle steps, nb::handle viscosity_steps) {
    const Rule* r = self.solver.rule();
    if (r == nullptr) {
        if (nb::len(steps) != 0 || nb::len(viscosity_steps) != 0) {
            refuse_value("Flow2: a window restored under velocity_step");
        }
        return;
    }
    std::vector<double> velocity;
    std::vector<double> viscosity;
    for (nb::handle s : steps) {
        velocity.push_back(python_float(s));
    }
    for (nb::handle s : viscosity_steps) {
        viscosity.push_back(python_float(s));
    }
    if (self.solver.restore(velocity, viscosity) != cfdcore::stopping::restore_status::ok) {
        refuse_value("Flow2: the window is not one the rule restores");
    }
}

// A chunk's failure by name, with the message the reference's raise carries and
// the type it raises.
nb::object failure_of(const Flow2& self, const cfdcore::flow::chunk_failure& f) {
    using cfdcore::flow::failure;
    if (f.kind == failure::none) {
        return nb::none();
    }
    nb::dict out;
    out["iteration"] = f.iteration;
    std::string type = "ValueError";
    std::string message;
    const std::string component = f.component == 0 ? "u" : "v";
    switch (f.kind) {
        case failure::none:
            break;
        case failure::eddy_viscosity_not_finite:
            message = "eddy_viscosity must be finite in every non-SOLID cell";
            break;
        case failure::eddy_viscosity_negative:
            message = "eddy_viscosity must be non-negative (ADR-012 D)";
            break;
        case failure::mu_eff_not_finite:
            message = "mu_eff must be finite in every non-SOLID cell";
            break;
        case failure::mu_eff_not_positive:
            message = "mu_eff must be positive in every non-SOLID cell";
            break;
        case failure::wall_mu_not_finite:
            message = "wall_mu['" + component + "'] must be finite at every wall face";
            break;
        case failure::wall_mu_not_positive:
            message = "wall_mu['" + component + "'] must be positive at every wall face";
            break;
        case failure::pressure:
            switch (f.pressure) {
                case solve_status::not_dominant:
                    message = not_dominant_message;
                    break;
                case solve_status::division_by_zero:
                    type = "ZeroDivisionError";
                    message = "division_by_zero";
                    break;
                case solve_status::not_converged:
                case solve_status::ok:
                    type = "LinAlgError";
                    message = "not_converged";
                    break;
            }
            break;
        case failure::faces_not_finite:
            message = "faces must be finite";
            break;
        case failure::conditions:
            message = condition_message(f.conditions);
            break;
        case failure::nothing_moves:
            message = nothing_moves_message;
            break;
        case failure::k_not_positive:
        case failure::eps_not_positive: {
            type = "PositivityError";
            const bool is_k = f.kind == failure::k_not_positive;
            out["quantity"] = is_k ? "k" : "eps";
            out["count"] = f.bad.count;
            out["row"] = f.bad.first / self.grid->cells(0);
            out["column"] = f.bad.first % self.grid->cells(0);
            out["value"] = f.bad.value;
            out["minimum"] = f.bad.least;
            break;
        }
        case failure::nu_scale:
            message = "nu_scale must be positive and finite, got " + float_str(f.nu_scale);
            break;
    }
    out["type"] = type;
    out["message"] = message;
    return out;
}

template <typename T, typename Field>
nb::ndarray<nb::numpy, T> column(const std::vector<cfdcore::flow::iteration_record>& records,
                                 std::size_t n, Field field) {
    std::vector<T> values(n);
    for (std::size_t i = 0; i < n; ++i) {
        values[i] = field(records[i]);
    }
    auto* data = new T[n > 0 ? n : 1];
    std::copy(values.begin(), values.end(), data);
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    const std::size_t shape[1] = {n};
    return nb::ndarray<nb::numpy, T>(data, 1, shape, owner);
}

nb::dict flow_solve(Flow2& self, nb::handle max_outer, nb::handle first) {
    const std::int64_t count = read_count(max_outer, "max_outer");
    const std::int64_t start = read_count(first, "first");
    if (count < 1 || start < 0) {
        refuse_value("Flow2: a chunk takes at least one outer iteration, from one at or after 0");
    }
    std::vector<cfdcore::flow::iteration_record> records(static_cast<std::size_t>(count));
    const cfdcore::flow::chunk_result result = self.solver.run(start, records);
    const auto n = static_cast<std::size_t>(result.outer);
    using record = cfdcore::flow::iteration_record;
    nb::dict out;
    out["stopped"] = result.stopped;
    out["outer"] = result.outer;
    out["residual"] = column<double>(records, n, [](const record& r) { return r.residual; });
    out["step"] = column<double>(records, n, [](const record& r) { return r.step; });
    out["viscosity_step"] =
        column<double>(records, n, [](const record& r) { return r.viscosity_step; });
    out["estimate"] = column<double>(records, n, [](const record& r) { return r.rule.estimate; });
    out["viscosity_estimate"] =
        column<double>(records, n, [](const record& r) { return r.rule.viscosity_estimate; });
    out["nu_scale"] = column<double>(records, n, [](const record& r) { return r.rule.nu_scale; });
    out["pressure_iterations"] =
        column<std::int64_t>(records, n, [](const record& r) { return r.pressure_iterations; });
    out["pressure_products"] =
        column<std::int64_t>(records, n, [](const record& r) { return r.pressure_products; });
    out["reached_cap"] = column<bool>(records, n, [](const record& r) { return r.reached_cap; });
    out["failure"] = failure_of(self, result.failure);
    // A failing iteration whose correction finished: the reference counts its cap
    // before it raises.
    nb::object failing = nb::none();
    using cfdcore::flow::failure;
    const failure kind = result.failure.kind;
    const bool after_correction = kind == failure::faces_not_finite ||
                                  kind == failure::conditions || kind == failure::nothing_moves ||
                                  kind == failure::k_not_positive ||
                                  kind == failure::eps_not_positive || kind == failure::nu_scale;
    if (after_correction) {
        failing = nb::make_tuple(records[n].pressure_iterations, records[n].reached_cap);
    }
    out["failing_correction"] = failing;
    nb::dict seconds;
    seconds["momentum"] = result.seconds.momentum;
    seconds["pressure"] = result.seconds.pressure;
    seconds["turbulence"] = result.seconds.turbulence;
    seconds["correct"] = result.seconds.correct;
    seconds["stopping"] = result.seconds.stopping;
    out["stage_seconds"] = seconds;
    return out;
}

nb::dict flow_fields(const Flow2& self) {
    const Grid2& g = *self.grid;
    nb::dict out;
    out["u"] = copy_field(self.solver.velocity(0), face_shape(g, 0));
    out["v"] = copy_field(self.solver.velocity(1), face_shape(g, 1));
    out["p"] = copy_field(self.solver.pressure(), cell_shape(g));
    if (self.coupled) {
        out["k"] = copy_field(self.solver.k(), cell_shape(g));
        out["eps"] = copy_field(self.solver.eps(), cell_shape(g));
        out["nu_t"] = copy_field(self.solver.nu_t(), cell_shape(g));
    }
    return out;
}

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
        "axis, widths of another length, a width, a wall distance, rho or mu that is not\n"
        "finite and positive, alpha outside (0, 1] or fewer than one sweep raise ValueError; an\n"
        "edge or a vent mask of another shape, and a bool or another type where a number is\n"
        "read, TypeError.")
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
    // A pressure solve that cannot go on, where the reference's own arithmetic
    // raises; its ZeroDivisionError or LinAlgError is made of it on the Python side.
    PyObject* breakdown_type = PyErr_NewExceptionWithDoc(
        "_cfdcore.PressureBreakdown",
        "A pressure solve ended where the reference's arithmetic raises.\n\n"
        "Attributes: reason, \"division_by_zero\" or \"not_converged\" (the coarsest level's\n"
        "eigenvalues), and iterations.",
        PyExc_RuntimeError, nullptr);
    if (breakdown_type == nullptr) {
        throw nb::python_error();
    }
    m.attr("PressureBreakdown") = nb::steal(breakdown_type);
    nb::register_exception_translator(translate_breakdown, breakdown_type);

    nb::class_<Multigrid2>(
        m, "Multigrid2",
        "The additive-correction multigrid of a five-point system (Multigrid).\n\n"
        "Multigrid2(a_p, a_e, a_w, a_n, a_s, levels): the five [ny, nx] arrays and the keys\n"
        "(coarsest_cells, sweeps, weight, dominance_tolerance, rcond). The reference's\n"
        "refusals are raised with its types and messages; keys outside their ranges are a\n"
        "ValueError.")
        .def("__init__",
             [](Multigrid2* self, const Array& a_p, const Array& a_e, const Array& a_w,
                const Array& a_n, const Array& a_s, nb::handle levels) {
                 new (self) Multigrid2({&a_p, &a_e, &a_w, &a_n, &a_s}, read_levels(levels));
             },
             nb::arg("a_p").noconvert(), nb::arg("a_e").noconvert(), nb::arg("a_w").noconvert(),
             nb::arg("a_n").noconvert(), nb::arg("a_s").noconvert(), nb::arg("levels"))
        .def("cycle",
             [](Multigrid2& self, const Array& r) {
                 const auto residual = read_field(r, self.finest.fields, "r");
                 NewArray z = new_field(self.finest.fields);
                 self.hierarchy.cycle(residual, out_span(z));
                 return z;
             },
             nb::arg("r").noconvert(), "One V-cycle from zero (Multigrid.cycle), to 1e-12.")
        .def("rebuild",
             [](Multigrid2& self) { (void)self.hierarchy.build(self.finest.system); },
             "Builds the levels again from the same system, as a correction builds its own: for\n"
             "timing.")
        .def("levels", &multigrid_levels,
             "The levels, finest first: the five arrays by name and the smoother (None on the\n"
             "coarsest), bit for bit the reference's.")
        .def("coarsest_matrix",
             [](const Multigrid2& self) { return square(self.hierarchy.coarsest_matrix()); },
             "The coarsest operator as a dense matrix, bit for bit the reference's.")
        .def("coarsest_inverse",
             [](const Multigrid2& self) { return square(self.hierarchy.coarsest_inverse()); },
             "Its pseudo-inverse, to 1e-12 of NumPy's largest magnitude.");
    m.def("apply_operator", &five_point_apply, nb::arg("a_p").noconvert(),
          nb::arg("a_e").noconvert(), nb::arg("a_w").noconvert(), nb::arg("a_n").noconvert(),
          nb::arg("a_s").noconvert(), nb::arg("x").noconvert(),
          "The p' operator (pressure.apply_operator), bit for bit: a new array.");
    m.def("smooth", &five_point_smooth, nb::arg("a_p").noconvert(), nb::arg("a_e").noconvert(),
          nb::arg("a_w").noconvert(), nb::arg("a_n").noconvert(), nb::arg("a_s").noconvert(),
          nb::arg("smoother").noconvert(), nb::arg("x").noconvert(), nb::arg("r").noconvert(),
          "A weighted Jacobi sweep, x + smoother (r - A x), bit for bit: a new array.");
    m.def("coarsen", &five_point_coarsen, nb::arg("a_p").noconvert(), nb::arg("a_e").noconvert(),
          nb::arg("a_w").noconvert(), nb::arg("a_n").noconvert(), nb::arg("a_s").noconvert(),
          "The Galerkin operator of 2x2 aggregation (multigrid.coarsen), bit for bit: the five\n"
          "arrays by name.");
    m.def("restrict", &restrict_field, nb::arg("r").noconvert(),
          "The group sums (multigrid.restrict), bit for bit: a new array.");
    m.def("add_prolonged", &add_prolonged, nb::arg("x").noconvert(), nb::arg("z").noconvert(),
          "x += P z in place (multigrid.add_prolonged), bit for bit.");
    m.def("pseudo_inverse", &pseudo_inverse, nb::arg("matrix").noconvert(), nb::arg("rcond"),
          "The pseudo-inverse of a symmetric matrix by NumPy's cutoff rule, to 1e-12 of\n"
          "np.linalg.pinv(matrix, hermitian=True).");
    m.def("symmetric_eigenvalues", &symmetric_eigenvalues, nb::arg("matrix").noconvert(),
          "The eigenvalues pseudo_inverse reads, in no order.");
    m.def("project_mean", &project_mean, nb::arg("x").noconvert(), nb::arg("mask").noconvert(),
          "x with the mean of its masked cells removed from them, bit for bit: a new array.");
    m.def("pressure_cg", &five_point_cg, nb::arg("a_p").noconvert(), nb::arg("a_e").noconvert(),
          nb::arg("a_w").noconvert(), nb::arg("a_n").noconvert(), nb::arg("a_s").noconvert(),
          nb::arg("f").noconvert(), nb::arg("rtol").none(), nb::arg("floor").none(),
          nb::arg("max_iter").none(), nb::arg("preconditioner").none(), nb::arg("project"),
          nb::arg("levels"),
          "pressure.conjugate_gradient on a five-point system, with the reference's\n"
          "preconditioner of that name: x, iterations, reached_cap, residual_norm and products\n"
          "by name. Refuses its numbers as the reference does.");
    m.def("pressure_precondition", &five_point_precondition, nb::arg("a_p").noconvert(),
          nb::arg("a_e").noconvert(), nb::arg("a_w").noconvert(), nb::arg("a_n").noconvert(),
          nb::arg("a_s").noconvert(), nb::arg("r").noconvert(), nb::arg("preconditioner"),
          nb::arg("project"), nb::arg("levels"),
          "One application of the preconditioner pressure_cg uses: a new array.");
    nb::class_<Corrector2>(
        m, "PressureCorrector2",
        "The staggered pressure correction on a grid (PressureCorrector).\n\n"
        "PressureCorrector2(grid, dx_cell, dy_cell, outlets, fluid, rho, alpha, rtol, floor,\n"
        "max_iter, preconditioner, levels): the mesh's widths, the left, right, bottom and top\n"
        "edges' outlet faces (pressure_outlets()[edge].is_outlet), the type of a FLUID cell,\n"
        "air's density, solver.alpha_pressure, pressure_rtol, RESIDUAL_FLOOR times the flux\n"
        "scale, max_pressure_iter and pressure_preconditioner, and the multigrid's keys as\n"
        "Multigrid2 takes them. The reference's refusals are raised with its types and\n"
        "messages; a width, rho or alpha that is not finite and positive is a ValueError, an\n"
        "outlet array of another kind a TypeError.")
        .def("__init__", &corrector_init, nb::arg("grid"), nb::arg("dx_cell").noconvert(),
             nb::arg("dy_cell").noconvert(), nb::arg("outlets"), nb::arg("fluid"), nb::arg("rho"),
             nb::arg("alpha"), nb::arg("rtol").none(), nb::arg("floor").none(),
             nb::arg("max_iter").none(), nb::arg("preconditioner").none(), nb::arg("levels"),
             nb::keep_alive<1, 2>())
        .def_prop_ro("needs_pin", &Corrector2::needs_pin)
        .def_prop_ro("pin_cell",
                     [](const Corrector2& c) {
                         const std::size_t nx = c.mesh().cells(0);
                         return nb::make_tuple(c.pin_cell() / nx, c.pin_cell() % nx);
                     })
        .def("coefficients", &corrector_coefficients, nb::arg("a_p_u").noconvert(),
             nb::arg("a_p_v").noconvert(),
             "PressureCorrector.coefficients, bit for bit: the five arrays by name.")
        .def("face_d", &corrector_face_d, nb::arg("a_p_u").noconvert(),
             nb::arg("a_p_v").noconvert(),
             "d = A_face / a_P on u's and on v's faces, bit for bit.")
        .def("mass_imbalance", &corrector_imbalance, nb::arg("u").noconvert(),
             nb::arg("v").noconvert(),
             "PressureCorrector.mass_imbalance, bit for bit: a new array.")
        .def("right_hand_side", &corrector_right_hand_side, nb::arg("u_star").noconvert(),
             nb::arg("v_star").noconvert(), nb::arg("a_p_u").noconvert(),
             nb::arg("a_p_v").noconvert(),
             "The right-hand side correct solves for, projected on a closed domain, bit for bit.")
        .def("correct", &corrector_correct, nb::arg("u_star").noconvert(),
             nb::arg("v_star").noconvert(), nb::arg("a_p_u").noconvert(),
             nb::arg("a_p_v").noconvert(), nb::arg("p").noconvert(),
             "PressureCorrector.correct: u, v, p, p_prime, iterations, reached_cap, products and\n"
             "the residual's norm by name.")
        .def("apply", &corrector_apply, nb::arg("u_star").noconvert(),
             nb::arg("v_star").noconvert(), nb::arg("a_p_u").noconvert(),
             nb::arg("a_p_v").noconvert(), nb::arg("p").noconvert(),
             nb::arg("p_prime").noconvert(),
             "The correction by a given solution of the equation, bit for bit: u, v, p and the\n"
             "pinned p_prime by name.");

    // A step that left k or eps at a value that is not positive and finite, whose
    // PositivityError the Python side makes.
    PyObject* positivity_type = PyErr_NewExceptionWithDoc(
        "_cfdcore.PositivityFailure",
        "k or eps is not positive and finite at a non-SOLID cell after a step (REQ-S15).\n\n"
        "Attributes: quantity (\"k\" or \"eps\"), count, the first cell's row and column and its\n"
        "value, and minimum, the least value over the non-SOLID cells (NaN where one is).",
        PyExc_RuntimeError, nullptr);
    if (positivity_type == nullptr) {
        throw nb::python_error();
    }
    m.attr("PositivityFailure") = nb::steal(positivity_type);
    nb::register_exception_translator(translate_positivity, positivity_type);

    m.def("log_law_floor", &log_law_floor, nb::arg("kappa"), nb::arg("e"),
          "turbulence.log_law_floor, bit for bit, with its refusals.");
    m.def("wall_viscosity", &wall_viscosity_values, nb::arg("k_p").noconvert(),
          nb::arg("y_p").noconvert(), nb::arg("c_mu"), nb::arg("rho"), nb::arg("nu"),
          nb::arg("law"),
          "turbulence.wall_viscosity on 1-D arrays, with (kappa, E, floor): a new array. Its\n"
          "logarithm is the C library's, NumPy's own where NumPy calls it.");
    m.def("inflow_values", &inflow_values, nb::arg("intensity"), nb::arg("normal_speed"),
          nb::arg("dissipation_length"), "turbulence.inflow_values, bit for bit: (k, eps).");

    nb::class_<Boundary2>(
        m, "TurbulenceBoundary2",
        "The coupled solve's boundary data for k, eps and the walls (TurbulenceBoundary).\n\n"
        "TurbulenceBoundary2(grid, wall_cells, wall_corners, tangential, vents, edge_inlets,\n"
        "vent_inlets, exhausts, c_mu, rho, nu, law): per edge (bottom, top, left, right) the\n"
        "wall cells and corners (wall_faces()[edge].cells, .corners) and the tangential\n"
        "condition as (is_dirichlet, value, wall_distance); the vents' four masks as\n"
        "(normal_u, normal_v, tangential_u, tangential_v) or None; the domain faces that admit\n"
        "air, then the discharge vents', each (direction, row, column, intensity, normal speed,\n"
        "dissipation length, width); the exhausts' (direction, row, column); the variant's\n"
        "C_mu, air's density and kinematic viscosity and the law of the wall's (kappa, E,\n"
        "floor). An array of another shape or dtype, and a bool or another type where a number\n"
        "is read, is a TypeError; a face off the grid or a constant that is not finite and\n"
        "positive a ValueError.")
        .def("__init__", &boundary_init, nb::arg("grid"), nb::arg("wall_cells"),
             nb::arg("wall_corners"), nb::arg("tangential"), nb::arg("vents").none(),
             nb::arg("edge_inlets"), nb::arg("vent_inlets"), nb::arg("exhausts"), nb::arg("c_mu"),
             nb::arg("rho"), nb::arg("nu"), nb::arg("law"), nb::keep_alive<1, 2>())
        .def("wall_viscosity", &boundary_wall_viscosity, nb::arg("k").noconvert(),
             nb::arg("elsewhere"),
             "TurbulenceBoundary.wall_viscosity: new arrays by \"u\" and \"v\".")
        .def("conditions", &boundary_conditions, nb::arg("state"), nb::arg("faces"),
             "TurbulenceBoundary.conditions: TurbulenceConditions' arrays by name, new ones.")
        .def("initial_values", &boundary_initial_values,
             "TurbulenceBoundary.initial_values, bit for bit, with its refusal.")
        .def("stencil", &boundary_stencil, nb::arg("component"),
             "The wall-function faces of component 0 (u) or 1 (v), for the parity tests.")
        .def("inlet_values", &boundary_inlet_values,
             "Each inlet face's (k, eps), in the order handed in.")
        .def_prop_ro("wall_cells", [](const Boundary2& self) {
            const auto shape = cell_shape(self.mesh());
            return nb::cast(shaped_copy<bool>(self.wall_cells(), {shape[0], shape[1]}));
        });

    nb::class_<KEpsilon2>(
        m, "KEpsilon2",
        "k-epsilon on prescribed face velocities (KEpsilonModel's steady path).\n\n"
        "KEpsilon2(grid, dx_cell, dy_cell, dx_face, dy_face, variant, cfl, upwind, max_iter,\n"
        "tol, nu, vent_u=None, vent_v=None): the mesh's widths, the variant's (c_mu, c_1, c_2,\n"
        "sigma_k, sigma_e, eta_0, beta) with None for the last two where it has no R, the\n"
        "configuration's turbulence keys, air's kinematic viscosity and the vent faces\n"
        "(VentFaces.normal_u, normal_v). A state, faces and conditions are any objects with the\n"
        "attributes of TurbulenceState, FaceVelocities and TurbulenceConditions. The\n"
        "reference's refusals are raised with its types and messages; an array that is not\n"
        "C-contiguous float64, or a bool where a number is read, is a TypeError.")
        .def("__init__", &model_init, nb::arg("grid"), nb::arg("dx_cell").noconvert(),
             nb::arg("dy_cell").noconvert(), nb::arg("dx_face").noconvert(),
             nb::arg("dy_face").noconvert(), nb::arg("variant"), nb::arg("cfl"),
             nb::arg("upwind").noconvert(), nb::arg("max_iter"), nb::arg("tol"), nb::arg("nu"),
             nb::arg("vent_u").noconvert() = nb::none(),
             nb::arg("vent_v").noconvert() = nb::none(), nb::keep_alive<1, 2>())
        .def("eddy_viscosity", &model_eddy_viscosity, nb::arg("k").noconvert(),
             nb::arg("eps").noconvert(), "KEpsilonModel.eddy_viscosity, bit for bit.")
        .def("state", &model_state, nb::arg("k"), nb::arg("eps"),
             "KEpsilonModel.state, bit for bit: k, eps and nu_t by name.")
        .def("corner_gradient", &model_corner_gradient, nb::arg("component"),
             nb::arg("velocity").noconvert(), nb::arg("low").noconvert(),
             nb::arg("high").noconvert(),
             "du/dy (component 0) or dv/dx (1) at the corners [ny+1, nx+1], bit for bit.")
        .def("strain_squared", &model_strain, nb::arg("faces"), nb::arg("conditions"),
             "KEpsilonModel.strain_squared, bit for bit.")
        .def("terms", &model_terms, nb::arg("state"), nb::arg("faces"), nb::arg("conditions"),
             "KEpsilonModel.terms: StepTerms' arrays by name. RNG's cube is the C library's.")
        .def("pseudo_time_step", &model_time_step, nb::arg("faces"),
             "KEpsilonModel.pseudo_time_step, bit for bit.")
        .def("advected", &model_advected, nb::arg("faces"), nb::arg("q").noconvert(),
             nb::arg("inflow_u").noconvert(), nb::arg("inflow_v").noconvert(),
             "Step 1 of a quantity at the pseudo-time step of the faces, bit for bit.")
        .def("explicit_fields", &model_explicit_fields, nb::arg("state"), nb::arg("faces"),
             nb::arg("conditions"), "Steps 1 and 2: (k*, eps*).")
        .def("conductances", &model_conductances, nb::arg("quantity"),
             nb::arg("nu_t").noconvert(),
             "The face conductances of \"k\" or \"eps\" from nu_t, bit for bit: (g_u, g_v).")
        .def("step", &model_step, nb::arg("state"), nb::arg("faces"), nb::arg("conditions"),
             nb::arg("dt") = nb::none(),
             "KEpsilonModel.step: k, eps and nu_t by name, new arrays. The refusals are the\n"
             "reference's; a true-time dt is a NotImplementedError (ADR-017 F), and a quantity\n"
             "that is not positive and finite after the step raises PositivityFailure.")
        .def("advance", &model_advance, nb::arg("boundary"), nb::arg("state"), nb::arg("faces"),
             "step with the conditions the boundary, on the same grid, forms of the state and\n"
             "the faces. The state, the faces and the formed conditions are refused as step\n"
             "refuses them: a ValueError of the reference's message, before any arithmetic.")
        .def_prop_ro("last_iterations",
                     [](const KEpsilon2& self) {
                         return nb::make_tuple(self.last.solves[0].iterations,
                                               self.last.solves[1].iterations);
                     })
        .def_prop_ro("last_fallbacks",
                     [](const KEpsilon2& self) {
                         return nb::make_tuple(self.last.solves[0].fallbacks,
                                               self.last.solves[1].fallbacks);
                     })
        .def_prop_ro("solves_converged",
                     [](const KEpsilon2& self) {
                         return self.last.solves[0].converged && self.last.solves[1].converged;
                     })
        .def_prop_ro("volume",
                     [](const KEpsilon2& self) {
                         return nb::cast(
                             copy_field(self.model.volume(), cell_shape(self.model.mesh())));
                     })
        .def("check_positive", &model_check_positive, nb::arg("quantity"),
             nb::arg("q").noconvert(),
             "The positivity test a step makes of its field (`_check_positive`), given the\n"
             "field: None, or the PositivityFailure the step raises.")
        .def("live_faces", [](const KEpsilon2& self, int d) {
            return masks_matrix(self.model.live_faces(direction(d)),
                                face_shape(self.model.mesh(), d));
        }, nb::arg("direction"), "The faces that carry an advective flux, 0 or 1.")
        .def("inner_faces", [](const KEpsilon2& self, int d) {
            return masks_matrix(self.model.inner_faces(direction(d)),
                                face_shape(self.model.mesh(), d));
        }, nb::arg("direction"), "The faces diffusion crosses, 0 or 1.");

    m.def("relax_eddy_viscosity", &relax_eddy_viscosity, nb::arg("old").noconvert(),
          nb::arg("stepped").noconvert(), nb::arg("alpha"),
          "(1 - alpha) old + alpha stepped, the solver's relaxation of nu_t, bit for bit.");

    nb::class_<OutletFaces2>(m, "OutletFaces2",
                         "The pressure outlets' faces of a grid, each given its interior\n"
                         "neighbour's value before a prediction (`_extrapolate_outlets`).")
        .def("__init__", &outlets_init, nb::arg("grid"), nb::arg("outlets"),
             nb::keep_alive<1, 2>(),
             "outlets: the left, right, bottom and top edges' bool arrays, as\n"
             "PressureCorrector2 takes them.")
        .def("extrapolate", &outlets_extrapolate, nb::arg("u").noconvert(),
             nb::arg("v").noconvert(), "Copies of u and v with the outlet faces written.")
        .def_prop_ro(
            "count", [](const OutletFaces2& self) { return self.outlets.count(); },
            "The outlet faces held.");

    nb::class_<StoppingRule2>(
        m, "StoppingRule2",
        "The error_estimate rule (ErrorEstimateRule), its window and readings, with\n"
        "RATE_WINDOW handed in as the window.")
        .def("__init__", &rule_init, nb::arg("window"), nb::arg("velocity_scale"),
             nb::arg("flux_scale"), nb::arg("iteration_error_tol"),
             nb::arg("mass_imbalance_tol"), nb::arg("viscosity_condition") = false,
             "Refused as ErrorEstimateRule refuses its arguments.")
        .def_prop_ro("version", [](const StoppingRule2& self) { return self.rule.version(); })
        .def("window", [](const StoppingRule2& self) { return window_of(self.rule); },
             "The velocity and the viscosity steps, oldest first, as two tuples.")
        .def("restore", &rule_restore, nb::arg("steps"),
             nb::arg("viscosity_steps") = nb::tuple(),
             "ErrorEstimateRule.restore, with its refusals.")
        .def("update", &rule_update, nb::arg("step"), nb::arg("imbalance"),
             nb::arg("viscosity_step") = nb::none(), nb::arg("nu_scale") = nb::none(),
             "ErrorEstimateRule.update: imbalance a callable returning the three readings,\n"
             "called only when (a) and, with it on, (e) hold.")
        .def("estimate_of", &rule_estimate_of, nb::arg("steps"), nb::arg("scale"),
             "_estimate_of on the steps given, oldest first.")
        .def_prop_ro("estimate_history",
                     [](const StoppingRule2& self) { return list_of(self.estimates); })
        .def_prop_ro("viscosity_estimate_history",
                     [](const StoppingRule2& self) { return list_of(self.viscosity_estimates); })
        .def_prop_ro("viscosity_scale_history",
                     [](const StoppingRule2& self) { return list_of(self.viscosity_scales); });

    m.def("imbalance_summary", &imbalance_summary, nb::arg("imbalance").noconvert(),
          "_imbalance_summary's (worst, absolute_sum, signed_sum) of a cell field, bit for bit.");
    m.def("viscosity_scale", &viscosity_scale_of, nb::arg("nu"), nb::arg("nu_t").noconvert(),
          nb::arg("fluid").noconvert(), "nu plus the median of nu_t over the FLUID cells.");
    m.def("cell_centres", &cell_centres_of, nb::arg("grid"), nb::arg("u").noconvert(),
          nb::arg("v").noconvert(), "to_cell_centers' (u_c, v_c), bit for bit.");
    m.def("python_max", &python_max_of, nb::arg("values"),
          "max(du, dv) as the outer iteration folds the components' changes.");
    m.def("effective_viscosity", &effective_viscosity_of, nb::arg("nu_t").noconvert(),
          nb::arg("live").noconvert(), nb::arg("mu"), nb::arg("rho"),
          "_effective_viscosity's mu + rho nu_t over the live cells, bit for bit.");
    m.def("largest_change", &largest_change_of, nb::arg("now").noconvert(),
          nb::arg("before").noconvert(), nb::arg("cells").noconvert(),
          "np.max(np.abs(now[cells] - before[cells])), NaN propagated.");

    nb::class_<Flow2>(
        m, "Flow2",
        "The steady solve's outer iteration with the fields resident (ADR-017 B): the\n"
        "modules it calls, which it keeps alive, and the solver's numbers.")
        .def("__init__", &flow_init, nb::arg("grid"), nb::arg("predictor"),
             nb::arg("corrector"), nb::arg("outlets"), nb::arg("model").none(),
             nb::arg("walls").none(), nb::arg("fluid"), nb::arg("mu"), nb::arg("rho"),
             nb::arg("nu"), nb::arg("alpha_turbulence"), nb::arg("residual_scale"),
             nb::arg("rule"), nb::arg("mu_eff").none(), nb::keep_alive<1, 2>(),
             nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>(), nb::keep_alive<1, 5>(),
             nb::keep_alive<1, 6>(), nb::keep_alive<1, 7>(),
             "rule: ('velocity_step', convergence_tol) or ('error_estimate', window,\n"
             "velocity_scale, flux_scale, iteration_error_tol, mass_imbalance_tol).\n"
             "The model and its boundary make the solve coupled; mu_eff a fixed one.")
        .def("flow_state", &flow_state, nb::arg("u").noconvert(), nb::arg("v").noconvert(),
             nb::arg("p").noconvert(), nb::arg("k").noconvert().none() = nb::none(),
             nb::arg("eps").noconvert().none() = nb::none(),
             nb::arg("nu_t").noconvert().none() = nb::none(),
             "Begins a solve from these fields, the rule fresh.")
        .def("restore", &flow_restore, nb::arg("steps"), nb::arg("viscosity_steps"),
             "Seeds the fresh rule with a checkpoint's window.")
        .def("solve_steady", &flow_solve, nb::arg("max_outer"), nb::arg("first"),
             "At most max_outer outer iterations, the first numbered first: the stop's\n"
             "state, the readings and the records by name.")
        .def("flow_fields", &flow_fields, "u, v, p and, when coupled, k, eps, nu_t: copies.")
        .def("window",
             [](const Flow2& self) {
                 const Rule* r = self.solver.rule();
                 return r == nullptr ? nb::make_tuple(nb::tuple(), nb::tuple()) : window_of(*r);
             })
        .def_prop_ro(
            "rule_keys",
            [](const Flow2& self) -> nb::object {
                const Rule* r = self.solver.rule();
                if (r == nullptr) {
                    return nb::none();
                }
                const auto& k = r->keys();
                return nb::make_tuple(k.window, k.velocity_scale, k.flux_scale,
                                      k.iteration_error_tol, k.mass_imbalance_tol,
                                      k.viscosity_condition);
            },
            "(window, velocity_scale, flux_scale, iteration_error_tol, mass_imbalance_tol,\n"
            "viscosity_condition) of the rule; None under velocity_step.")
        .def_prop_ro("rule_version", [](const Flow2& self) -> nb::object {
            const Rule* r = self.solver.rule();
            if (r == nullptr) {
                return nb::none();
            }
            return nb::int_(r->version());
        });

    m.attr("budget_terms") = [] {
        nb::list names;
        for (const std::string_view name : cfdcore::march::budget_term_names) {
            names.append(nb::str(name.data(), name.size()));
        }
        return names;
    }();
}
