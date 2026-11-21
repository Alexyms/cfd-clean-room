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
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "cfdcore/grid.hpp"
#include "cfdcore/reduce.hpp"
#include "cfdcore/scalar.hpp"
#include "cfdcore/transport.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
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

// The reference's module, for the exception and the budget it defines.
nb::module_ reference_module() { return nb::module_::import_("src.solver_transport"); }

// A Python float's str, as an f-string formats it.
std::string float_str(double x) { return nb::cast<std::string>(nb::str(nb::float_(x))); }
std::string float_repr(double x) { return nb::cast<std::string>(nb::repr(nb::float_(x))); }

// Raises the exception the reference raises for a refusal.
[[noreturn]] void raise_refusal(const refusal& r, double limit) {
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
    (void)limit;
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

void transport_init(TransportSolver2* self, const Scheme2& s, const Vector& dx_cell,
                    const Vector& dy_cell, const Vector& dx_face, const Vector& dy_face,
                    double cfl, bool upwind, std::int64_t max_sweeps, double tol, double limit,
                    double share, double horizon, std::optional<double> schmidt,
                    const std::vector<double>& settling, const std::vector<double>& diffusion,
                    const std::array<std::int32_t, 3>& surface_codes) {
    constexpr std::string_view who = "TransportSolver2";
    const Grid2& g = s.mesh();
    const std::array<std::pair<const Vector*, std::size_t>, 4> spacings{
        {{&dx_cell, g.cells(0)}, {&dy_cell, g.cells(1)}, {&dx_face, g.cells(0) + 1},
         {&dy_face, g.cells(1) + 1}}};
    for (const auto& [v, n] : spacings) {
        if (v->shape(0) != n) {
            throw nb::type_error(
                "TransportSolver2: dx_cell, dy_cell, dx_face and dy_face are nx, ny, nx + 1 "
                "and ny + 1 long");
        }
        require_aligned(v->data(), alignof(double), who);
    }
    if (settling.empty() || settling.size() != diffusion.size()) {
        throw nb::value_error("TransportSolver2: settling and diffusion hold one value per class");
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
    const cfdcore::transport::geometry<2> spacing{
        .widths = {as_span(dx_cell), as_span(dy_cell)},
        .face_distances = {as_span(dx_face), as_span(dy_face)}};
    new (self) TransportSolver2(s, spacing, keys, std::move(physics));
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

void transport_set_interval(TransportSolver2& t, nb::handle conditions, nb::handle u,
                            nb::handle v, nb::handle v_ext, nb::handle eddy_viscosity,
                            nb::handle sources) {
    const Grid2& g = t.scheme->mesh();
    const std::size_t n = t.solver.classes();
    const std::array<std::size_t, 2> cells{g.cells(1), g.cells(0)};
    interval_arrays arrays;
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
    const refusal r = t.solver.set_interval(each, input);
    if (!r.ok()) {
        raise_refusal(r, t.limit);
    }
}

// `dt` as the reference reads it: a bool (Python's, NumPy's or a 0-d bool
// array) is a TypeError, anything else a number.
double read_dt(nb::handle dt) {
    const nb::module_ numpy = nb::module_::import_("numpy");
    const bool is_bool =
        nb::isinstance<bool>(dt) || PyObject_IsInstance(dt.ptr(), numpy.attr("bool_").ptr()) == 1 ||
        (PyObject_IsInstance(dt.ptr(), numpy.attr("ndarray").ptr()) == 1 &&
         nb::cast<bool>(nb::getattr(dt, "dtype").equal(numpy.attr("bool_"))));
    if (is_bool) {
        throw nb::type_error("dt must be a number, not a bool");
    }
    double value = 0.0;
    if (!nb::try_cast<double>(dt, value)) {
        throw nb::type_error("dt must be a number");
    }
    return value;
}

void transport_set_step(TransportSolver2& t, nb::handle dt) {
    const refusal r = t.solver.set_step(read_dt(dt));
    if (!r.ok()) {
        raise_refusal(r, t.limit);
    }
}

// `size_class` as the reference reads it (`_check_class`).
std::size_t read_class(nb::handle size_class, std::size_t n) {
    if (nb::isinstance<bool>(size_class) || !PyLong_Check(size_class.ptr())) {
        const nb::handle type(reinterpret_cast<PyObject*>(Py_TYPE(size_class.ptr())));
        throw nb::type_error(("size_class must be an int, got " +
                              nb::cast<std::string>(nb::getattr(type, "__name__")))
                                 .c_str());
    }
    const long long k = nb::cast<long long>(size_class);
    if (k < 0 || static_cast<std::size_t>(k) >= n) {
        throw nb::index_error(("size_class " + std::to_string(k) + " out of range [0, " +
                               std::to_string(n - 1) + "]")
                                  .c_str());
    }
    return static_cast<std::size_t>(k);
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
        // The reference's ConservationBreachError with its four arguments.
        const nb::object error = reference_module().attr("ConservationBreachError")(
            static_cast<long long>(k), record.accumulated, record.supplied, t.limit);
        PyErr_SetObject(reinterpret_cast<PyObject*>(Py_TYPE(error.ptr())), error.ptr());
        throw nb::python_error();
    }
    if (!r.ok()) {
        raise_refusal(r, t.limit);
    }
    return record;
}

nb::object optional_float(const std::optional<double>& value) {
    return value.has_value() ? nb::object(nb::float_(*value)) : nb::none();
}

// Every class's budget as the reference's MassBudget, a new object each.
nb::list transport_budgets(const TransportSolver2& t) {
    const nb::module_ ref = reference_module();
    const nb::object surfaces = ref.attr("DEPOSIT_SURFACES");
    const nb::object mass_budget = ref.attr("MassBudget");
    nb::list out;
    for (std::size_t k = 0; k < t.solver.classes(); ++k) {
        const auto& b = t.solver.budget_of(k);
        nb::dict deposited;
        std::size_t slot = 0;
        for (nb::handle name : surfaces) {
            deposited[name] = nb::float_(b.deposited[slot++]);
        }
        out.append(mass_budget(nb::arg("initial") = optional_float(b.initial),
                               nb::arg("inflow") = b.inflow, nb::arg("outflow") = b.outflow,
                               nb::arg("source") = b.source, nb::arg("deposited") = deposited,
                               nb::arg("current") = optional_float(b.current),
                               nb::arg("leftover") = b.leftover));
    }
    return out;
}

std::optional<double> read_optional(nb::handle value) {
    if (value.is_none()) {
        return std::nullopt;
    }
    return nb::cast<double>(value);
}

// `restore_budgets`: the reference's refusals and messages, then the budgets set whole.
void transport_restore_budgets(TransportSolver2& t, nb::handle budgets) {
    const nb::module_ ref = reference_module();
    const nb::object surfaces = ref.attr("DEPOSIT_SURFACES");
    const nb::object mass_budget = ref.attr("MassBudget");
    const std::size_t n = t.solver.classes();
    const std::size_t given = nb::len(budgets);
    if (given != n) {
        throw nb::value_error(("restore_budgets needs one budget per class (" + std::to_string(n) +
                               "), got " + std::to_string(given))
                                  .c_str());
    }
    const nb::object builtins = nb::module_::import_("builtins");
    std::vector<cfdcore::transport::budget> out;
    std::size_t k = 0;
    for (nb::handle b : budgets) {
        const std::string index = std::to_string(k);
        if (PyObject_IsInstance(b.ptr(), mass_budget.ptr()) != 1) {
            throw nb::type_error(
                ("budget " + index + " must be a MassBudget, got " +
                 nb::cast<std::string>(nb::getattr(builtins.attr("type")(b), "__name__")))
                    .c_str());
        }
        const nb::object deposited = nb::getattr(b, "deposited");
        const nb::object keys = builtins.attr("tuple")(deposited);
        if (!keys.equal(surfaces)) {
            const nb::object list = builtins.attr("list");
            throw nb::value_error(("budget " + index + " books deposition to " +
                                   nb::cast<std::string>(nb::repr(list(deposited))) +
                                   ", not " + nb::cast<std::string>(nb::repr(list(surfaces))))
                                      .c_str());
        }
        if (nb::getattr(b, "leftover").is_none()) {
            throw nb::value_error(
                ("budget " + index +
                 "'s leftover is unknown: it was written before the implicit solve's leftover "
                 "was booked (prompt 78), so the accumulated leftover before the restart is "
                 "unknown and REQ-T05's bound cannot be carried across it (ADR-016)")
                    .c_str());
        }
        cfdcore::transport::budget c;
        c.initial = read_optional(nb::getattr(b, "initial"));
        c.inflow = nb::cast<double>(nb::getattr(b, "inflow"));
        c.outflow = nb::cast<double>(nb::getattr(b, "outflow"));
        c.source = nb::cast<double>(nb::getattr(b, "source"));
        std::size_t slot = 0;
        for (nb::handle name : surfaces) {
            c.deposited[slot++] = nb::cast<double>(deposited[name]);
        }
        c.current = read_optional(nb::getattr(b, "current"));
        c.leftover = nb::cast<double>(nb::getattr(b, "leftover"));
        out.push_back(c);
        ++k;
    }
    t.solver.set_budgets(out);
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

std::size_t class_index(const TransportSolver2& t, std::size_t k) {
    if (k >= t.solver.classes()) {
        throw nb::index_error("the class is out of range");
    }
    return k;
}

// The interval's quantities by name (prediction (b)): a face field per class
// and direction, a cell field per class, or a face field per direction.
nb::object transport_quantity(const TransportSolver2& t, const std::string& name, std::size_t k,
                              int d) {
    const Grid2& g = t.scheme->mesh();
    const std::array<std::size_t, 2> cells{g.cells(1), g.cells(0)};
    const auto& s = t.solver;
    if (name == "deposition_cell" || name == "source_step") {
        const auto values = name == "deposition_cell" ? s.deposition_cell(class_index(t, k))
                                                      : s.source_step(class_index(t, k));
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
    class_index(t, k);
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

nb::tuple transport_masks(const TransportSolver2& t, const std::string& which, std::size_t k,
                          int d) {
    direction(d);
    class_index(t, k);
    const auto& s = t.solver;
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

double buffered_sum(const Vector& values, std::size_t row) {
    require_aligned(values.data(), alignof(double), "buffered_sum");
    if (row == 0 ? values.shape(0) != 0 : values.shape(0) % row != 0) {
        throw nb::value_error("buffered_sum: the terms are not a whole number of rows");
    }
    return cfdcore::reduce::buffered_sum(as_span(values), row);
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
          "in `values`: NumPy's buffered order (reduce.buffered_sum, prompt 90).");

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
        "read and set every class's MassBudget whole.")
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
            [](const TransportSolver2& t, std::size_t k) {
                return t.solver.stable_dt(class_index(t, k));
            },
            nb::arg("size_class"), "The class's stable_dt on the interval set up.")
        .def("step", &transport_step, nb::arg("size_class"), nb::arg("c").noconvert(),
             nb::arg("out").noconvert(),
             "Advance class size_class one step: c [ny, nx] into out [ny, nx] (out may be\n"
             "c), the budget booked. Returns the step's records. Raises what\n"
             "solve_timestep raises: ValueError for a field that is not finite or a step\n"
             "above stable_dt (nothing booked), ConservationBreachError after a step that\n"
             "breaches (the budget holds it, out the field).")
        .def("budgets", &transport_budgets, "Every class's budget, as new MassBudget objects.")
        .def("restore_budgets", &transport_restore_budgets, nb::arg("budgets"),
             "Set every class's budget whole from MassBudget objects, as restore_budgets\n"
             "does, with its refusals.")
        .def("interval_quantity", &transport_quantity, nb::arg("name"), nb::arg("size_class") = 0,
             nb::arg("direction") = 0,
             "A copy of one of the interval's quantities: advecting_velocity, volume_flux,\n"
             "conductance or deposition_step (per class and direction), deposition_cell or\n"
             "source_step (per class; None without a source), face_eddy_viscosity (None on\n"
             "the laminar path) or unit_conductance (per direction).")
        .def(
            "source_total",
            [](const TransportSolver2& t, std::size_t k) {
                return t.solver.source_total(class_index(t, k));
            },
            nb::arg("size_class"), "np.sum(rate * volume) of the class's source, 0.0 without one.")
        .def(
            "drawn",
            [](const TransportSolver2& t, std::size_t k) {
                const auto d = t.solver.drawn(class_index(t, k));
                return std::vector<double>(d.begin(), d.end());
            },
            nb::arg("size_class"), "Each recirculating intake's drawn flux, in order.")
        .def("masks", &transport_masks, nb::arg("which"), nb::arg("size_class"),
             nb::arg("direction"),
             "The masks the booking reads: 'domain', four bool arrays along the lines of\n"
             "the direction (low face inward, outward, high face inward, outward); or\n"
             "'vent', four bool arrays on the faces (entering from the low side, from the\n"
             "high side, leaving to the high side, to the low side), empty without vents.");
}
