#include "cfdcore/turbulence.hpp"

#include "cfdcore/elementary.hpp"
#include "cfdcore/positions.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace cfdcore::turbulence {

namespace {

using lanes::mask;
using positions::position;

// ---------------------------------------------------------------------------
// Directions

// The pairs of directions c < t, numbered (0, 1), (0, 2), (1, 2) and so on: the
// shear stresses of the strain, in the order it adds them.
template <int D>
inline constexpr int pair_count = D * (D - 1) / 2;

struct pair_of_directions {
    int c;
    int t;
};

template <int D>
constexpr pair_of_directions pair_at(int p) {
    for (int c = 0; c < D; ++c) {
        for (int t = c + 1; t < D; ++t) {
            if (p == 0) {
                return {c, t};
            }
            --p;
        }
    }
    return {0, 0};
}

// The number of the pair of two directions, in either order.
template <int D>
constexpr int pair_index(int a, int b) {
    for (int p = 0; p < pair_count<D>; ++p) {
        if (pair_at<D>(p).c == (a < b ? a : b) && pair_at<D>(p).t == (a < b ? b : a)) {
            return p;
        }
    }
    return 0;
}

// Runs body.operator()<P>() for each pair in turn.
template <int D, typename Body>
void for_each_pair(Body&& body) {
    [&]<int... P>(std::integer_sequence<int, P...>) {
        (body.template operator()<P>(), ...);
    }(std::make_integer_sequence<int, pair_count<D>>{});
}

// The entries an operand struct holds per pair; one unused where there is none.
template <int D>
inline constexpr int pair_slots = pair_count<D> == 0 ? 1 : pair_count<D>;

void require(bool condition, const char* what) {
    if (!condition) {
        throw std::invalid_argument(what);
    }
}

// ---------------------------------------------------------------------------
// Velocities and the strain

// The velocities of N faces where the face carries air, zero elsewhere: the
// reference's `np.where(self._live_u, u, 0.0)`.
template <std::size_t N>
inline void carried_cells(lanes::cells<N>, const double* velocity, const mask* live,
                          std::size_t first, double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        out[i] = lanes::keep(live[i], velocity[i]);
    }
}

// The product of two arrays over N elements: a flux, velocity times area, or a
// cell's sink, volume times decay.
template <std::size_t N>
inline void product_cells(lanes::cells<N>, const double* a, const double* b, std::size_t first,
                          double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        out[i] = a[i] * b[i];
    }
}

// What the corner gradient of a row reads: the values on the corners' low and
// high sides, the distance between them and where the gradient is defined. A
// row along the direction of the gradient reads both sides off one array,
// `high`, at i - 1 and i.
struct gradient_row {
    const double* low;
    const double* high;
    const double* span;
    const mask* defined;
};

// The gradient at N corners (`_corner_difference`), written, or with `Add` added
// to what the other component's pass wrote: the pair's shear.
template <bool Along, bool Add, std::size_t N>
inline void gradient_cells(lanes::cells<N>, const gradient_row& in, std::size_t first,
                           double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double low = Along ? in.high[i - 1] : in.low[i];
        const double g = lanes::keep(in.defined[i], (in.high[i] - low) / in.span[i]);
        out[i] = Add ? out[i] + g : g;
    }
}

// What the strain of a row of cells reads: for each direction the velocity on
// the cells' low and high faces (direction 0's high face is `low`'s next) and
// the width, per cell along direction 0 and the row's own along the others; for
// each pair of directions (c, t) the shear at the cells' corners, in rows at the
// low and the high side along t, and, where c crosses the rows, along c too.
template <int D>
struct strain_row {
    const double* low[D] = {};
    const double* high[D] = {};
    const double* width0 = nullptr;
    double width[D] = {};
    const double* shear[pair_slots<D>][4] = {};
    const mask* fluid = nullptr;
};

// `S^2` of N cells: twice each direction's squared face difference, then each
// pair's shear, the mean of its four corners, squared.
template <int D, std::size_t N>
inline void strain_cells(lanes::cells<N>, const strain_row<D>& in, std::size_t first,
                         double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double along = (in.low[0][i + 1] - in.low[0][i]) / in.width0[i];
        double s2 = 2.0 * (along * along);
        for (int d = 1; d < D; ++d) {
            const double across = (in.high[d][i] - in.low[d][i]) / in.width[d];
            s2 = s2 + 2.0 * (across * across);
        }
        for (int p = 0; p < pair_count<D>; ++p) {
            // The corners low and high along t, each low and high along c; a pair
            // of direction 0 reads both of a row's off one pointer.
            const bool with_rows = pair_at<D>(p).c == 0;
            const double sum =
                with_rows ? in.shear[p][0][i] + in.shear[p][0][i + 1] + in.shear[p][2][i] +
                                in.shear[p][2][i + 1]
                          : in.shear[p][0][i] + in.shear[p][1][i] + in.shear[p][2][i] +
                                in.shear[p][3][i];
            const double centre = 0.25 * sum;
            s2 = s2 + centre * centre;
        }
        out[i] = lanes::keep(in.fluid[i], s2);
    }
}

// ---------------------------------------------------------------------------
// The terms

// What the terms read, each pointer at a cell field's first cell.
struct terms_cells_in {
    const double* k;
    const double* eps;
    const double* nu_t;
    const double* strain;
    const double* given;
    const mask* is_given;
    const mask* fluid;
};

// The production of N cells, `nu_t S^2` or the given value, and eps / k, both
// zero in SOLID cells.
template <std::size_t N>
inline void production_cells(lanes::cells<N>, const terms_cells_in& in, std::size_t first,
                             double* CFDCORE_RESTRICT production,
                             double* CFDCORE_RESTRICT ratio) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        production[i] = lanes::keep(
            in.fluid[i], lanes::select(in.is_given[i], in.given[i], in.nu_t[i] * in.strain[i]));
        ratio[i] = lanes::keep(in.fluid[i], in.eps[i] / in.k[i]);
    }
}

// The growth and the decay of eps over N cells for a variant without R.
template <std::size_t N>
inline void standard_cells(lanes::cells<N>, double c_1, double c_2, const double* production,
                           const double* ratio, std::size_t first,
                           double* CFDCORE_RESTRICT growth, double* CFDCORE_RESTRICT decay) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        growth[i] = c_1 * ratio[i] * production[i];
        decay[i] = c_2 * ratio[i];
    }
}

// What R and its split read beside the fields above: eta and its cube.
struct rng_row {
    const double* eta;
    const double* cube;
    const double* eps;
    const double* ratio;
    const double* production;
    const mask* fluid;
};

// RNG's R of N cells and its split by sign (ADR-012 C): the negative part adds to
// eps's growth, the positive part, over eps, to its decay. `np.minimum(R, 0.0)`
// and `np.maximum(R, 0.0)` are written as a kept mask, as `momentum`'s positive
// part is: a comparison with a constant zero folds to a branch under GCC 13.
template <std::size_t N>
inline void rng_cells(lanes::cells<N>, const rng_row& in, const constants& v, std::size_t first,
                      double* CFDCORE_RESTRICT r, double* CFDCORE_RESTRICT growth,
                      double* CFDCORE_RESTRICT decay) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double coefficient =
            v.c_mu * in.cube[i] * (1.0 - in.eta[i] / v.eta_0) / (1.0 + v.beta * in.cube[i]);
        const double ri = lanes::keep(in.fluid[i], coefficient * in.eps[i] * in.ratio[i]);
        r[i] = ri;
        const double negative = lanes::keep(lanes::mask_of(!(ri >= 0.0)), ri);
        const double positive = lanes::keep(lanes::mask_of(!(ri <= 0.0)), ri);
        growth[i] = v.c_1 * in.ratio[i] * in.production[i] - negative;
        decay[i] = lanes::keep(in.fluid[i], v.c_2 * in.ratio[i] + positive / in.eps[i]);
    }
}

// ---------------------------------------------------------------------------
// The pseudo-time step

// What the rate of a row of cells reads: the carried velocity on the low and the
// high faces and the widths, as `strain_row`.
template <int D>
struct rate_row {
    const double* low[D] = {};
    const double* high[D] = {};
    const double* width0 = nullptr;
    double width[D] = {};
    const mask* fluid = nullptr;
};

// The larger of two magnitudes: `np.maximum` of two values that are not negative.
CFDCORE_INLINE double larger(double a, double b) {
    return lanes::select(lanes::mask_of(a > b), a, b);
}

// The step of N cells, `cfl / rate` where the cell moves and -1, which no step is,
// elsewhere; the largest step of each cell of a block into `largest`. Returns the
// moving cells' masks, ORed.
template <int D, std::size_t N>
inline mask step_cells(lanes::cells<N>, const rate_row<D>& in, double cfl, std::size_t first,
                       double* CFDCORE_RESTRICT dt, double* CFDCORE_RESTRICT largest) {
    mask any = lanes::none;
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        double rate =
            larger(std::fabs(in.low[0][i]), std::fabs(in.low[0][i + 1])) / in.width0[i];
        for (int d = 1; d < D; ++d) {
            rate = rate +
                   larger(std::fabs(in.low[d][i]), std::fabs(in.high[d][i])) / in.width[d];
        }
        const mask moving = in.fluid[i] & lanes::mask_of(rate > 0.0);
        any |= moving;
        const double step = lanes::select(moving, cfl / rate, -1.0);
        dt[i] = step;
        largest[k] = lanes::select(lanes::mask_of(step > largest[k]), step, largest[k]);
    }
    return any;
}

// The steps of N cells at rest, set to the largest finite one.
template <std::size_t N>
inline void cap_cells(lanes::cells<N>, double largest, std::size_t first,
                      double* CFDCORE_RESTRICT dt) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        dt[i] = lanes::select(lanes::mask_of(dt[i] < 0.0), largest, dt[i]);
    }
}

// ---------------------------------------------------------------------------
// The explicit part

// What the explicit part of a row of cells reads: the advective fluxes through
// the cells' low and high faces (direction 0's high face is `low`'s next), the
// quantity, its growth, the step, the volume, and where a cell is held and at
// what.
template <int D>
struct explicit_row {
    const double* low[D] = {};
    const double* high[D] = {};
    const double* q = nullptr;
    const double* dt = nullptr;
    const double* volume = nullptr;
    const double* growth = nullptr;
    const mask* fluid = nullptr;
    const mask* held = nullptr;
    const double* held_value = nullptr;
};

// What a pass of the explicit part forms: the advected field alone, the explicit
// field, or the explicit field with cells held.
enum class part { advected, grown, held };

// The explicit part of N cells: the divergence high less low, direction by
// direction, then `q - dt div / V`, then the growth, then the held value.
template <part Part, int D, std::size_t N>
inline void explicit_cells(lanes::cells<N>, const explicit_row<D>& in, std::size_t first,
                           double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        double divergence = in.low[0][i + 1] - in.low[0][i];
        for (int d = 1; d < D; ++d) {
            divergence = divergence + in.high[d][i];
            divergence = divergence - in.low[d][i];
        }
        double star = in.q[i] - in.dt[i] * divergence / in.volume[i];
        if constexpr (Part != part::advected) {
            star = lanes::keep(in.fluid[i], star + in.dt[i] * in.growth[i]);
        }
        if constexpr (Part == part::held) {
            star = lanes::select(in.held[i], in.held_value[i], star);
        }
        out[i] = star;
    }
}

template <part Part, int D>
void explicit_row_cells(const explicit_row<D> row, std::size_t n, double* out) {
    lanes::for_each_block(0, n, [&](auto cells, std::size_t m) {
        explicit_cells<Part>(cells, row, m, out);
    });
}

// ---------------------------------------------------------------------------
// The conductances

// The diffusivity of N cells, 1 in SOLID ones, which no conductance reads.
template <std::size_t N>
inline void gamma_cells(lanes::cells<N>, const double* nu_t, const mask* fluid, double nu,
                        double sigma, std::size_t first, double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        out[i] = lanes::select(fluid[i], nu + nu_t[i] / sigma, 1.0);
    }
}

// What the conductances of a row of interior faces read: the diffusivity of the
// cells on their low and high sides (off `high` at i - 1 and i for faces normal
// to the rows' direction), the far cell's share of the distance between them and
// that distance, per face along the rows or the row's one, the area and the mask.
struct conductance_row {
    const double* low;
    const double* high;
    const double* share;
    const double* distance;
    const double* area;
    const mask* inner;
};

// The conductance of N faces: the distance-weighted harmonic mean of the two
// cells' diffusivities, written so that equal ones return themselves, times
// the area over the distance.
template <bool Along, std::size_t N>
inline void conductance_cells(lanes::cells<N>, const conductance_row& in, std::size_t first,
                              double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double low = Along ? in.high[i - 1] : in.low[i];
        const double high = in.high[i];
        const std::size_t w = Along ? i : 0;
        const double face = low / (1.0 + in.share[w] * (low / high - 1.0));
        out[i] = lanes::keep(in.inner[i], face * in.area[i] / in.distance[w]);
    }
}

// ---------------------------------------------------------------------------
// The end of a step

// The eddy viscosity of N cells, `C_mu k^2 / eps`, zero in SOLID ones.
template <std::size_t N>
inline void viscosity_cells(lanes::cells<N>, double c_mu, const double* k, const double* eps,
                            const mask* fluid, std::size_t first, double* CFDCORE_RESTRICT out) {
    for (std::size_t m = 0; m < N; ++m) {
        const std::size_t i = first + m;
        out[i] = lanes::keep(fluid[i], c_mu * (k[i] * k[i]) / eps[i]);
    }
}

// N cells with the SOLID ones set to zero in place.
template <std::size_t N>
inline void clear_solid_cells(lanes::cells<N>, const mask* fluid, std::size_t first,
                              double* CFDCORE_RESTRICT q) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        q[i] = lanes::keep(fluid[i], q[i]);
    }
}

// Whether any of N values is not finite.
template <std::size_t N>
inline mask not_finite_cells(lanes::cells<N>, const double* q, std::size_t first) {
    mask bad = lanes::none;
    for (std::size_t k = 0; k < N; ++k) {
        bad |= lanes::mask_of(!(std::fabs(q[first + k]) <= std::numeric_limits<double>::max()));
    }
    return bad;
}

// Whether any of N non-SOLID cells is not finite, or is below zero, or with
// `Positive` is not above it.
template <bool Positive, std::size_t N>
inline mask out_of_range_cells(lanes::cells<N>, const double* q, const mask* fluid,
                               std::size_t first) {
    mask bad = lanes::none;
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const mask above = lanes::mask_of(Positive ? q[i] > 0.0 : q[i] >= 0.0);
        const mask finite = lanes::mask_of(q[i] <= std::numeric_limits<double>::max());
        bad |= fluid[i] & ~(above & finite);
    }
    return bad;
}

template <std::size_t N>
inline mask not_zero_cells(lanes::cells<N>, const double* q, const mask* fluid,
                           std::size_t first) {
    mask bad = lanes::none;
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        bad |= ~fluid[i] & lanes::mask_of(q[i] != 0.0);
    }
    return bad;
}

// The masks of a bool cell field.
template <std::size_t N>
inline void flag_cells(lanes::cells<N>, const bool* flags, std::size_t first,
                       mask* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        out[first + k] = lanes::mask_of(flags[first + k]);
    }
}

bool positive_and_finite(double x) { return std::isfinite(x) && x > 0.0; }

}  // namespace

bool all_finite(std::span<const double> values) {
    mask bad = lanes::none;
    lanes::for_each_block(0, values.size(), [&](auto cells, std::size_t m) {
        bad |= not_finite_cells(cells, values.data(), m);
    });
    return bad == lanes::none;
}

// ---------------------------------------------------------------------------
// Set-up

template <int D>
    requires(D >= 1)
std::string_view model<D>::check(const grid::grid<D>& g, const model_input<D>& input) {
    bool any = false;
    bool every = true;
    for (int d = 0; d < D; ++d) {
        const std::size_t n = g.cells(d);
        if (input.cell_widths[d].size() != n) {
            return "the cell widths along a direction are not one per cell";
        }
        if (input.face_widths[d].size() != n + 1) {
            return "the face widths along a direction are not one per face";
        }
        for (const auto& widths : {input.cell_widths[d], input.face_widths[d]}) {
            for (const double w : widths) {
                if (!positive_and_finite(w)) {
                    return "a width is not finite and positive";
                }
            }
        }
        const std::size_t vents = input.vents[d].size();
        if (vents != 0 && vents != g.fields().face_count(d)) {
            return "the vent faces normal to a direction are not one per face";
        }
        any = any || vents != 0;
        every = every && vents != 0;
    }
    if (any != every) {
        return "the vent faces are given for some directions and not for others";
    }
    const constants& v = input.variant;
    for (const double x : {v.c_mu, v.c_1, v.c_2, v.sigma_k, v.sigma_e, input.cfl, input.nu}) {
        if (!positive_and_finite(x)) {
            return "a constant of the variant, cfl or nu is not finite and positive";
        }
    }
    if (v.has_r && !(positive_and_finite(v.eta_0) && positive_and_finite(v.beta))) {
        return "eta_0 or beta is not finite and positive";
    }
    // The configuration's own range (`turbulence.tol` positive and finite,
    // `turbulence.max_iter` a positive integer): a cap below one would let a
    // solve return its start as converged, and a tolerance that is zero, negative
    // or NaN has no residual that meets it.
    if (!positive_and_finite(input.tol)) {
        return "tol is not finite and positive";
    }
    if (input.max_iter < 1) {
        return "max_iter is not positive";
    }
    return {};
}

template <int D>
    requires(D >= 1)
model<D>::model(const grid::grid<D>& g, const model_input<D>& input)
    : grid_(&g),
      scheme_(g),
      solver_(g.fields()),
      variant_(input.variant),
      cfl_(input.cfl),
      upwind_(input.upwind),
      max_iter_(input.max_iter),
      tol_(input.tol),
      nu_(input.nu),
      faces_{[&]<int... d>(std::integer_sequence<int, d...>) {
          return std::array<layout::layout<D>, D>{positions::faces_layout(g.fields(), d, d)...};
      }(std::make_integer_sequence<int, D>{})} {
    const std::string_view reason = check(g, input);
    if (!reason.empty()) {
        throw std::invalid_argument("turbulence::model: " + std::string(reason));
    }
    const auto& fields = g.fields();
    const std::size_t n = fields.cell_count();
    fluid_.resize(n);
    solid_ = std::make_unique<bool[]>(n);
    for (std::size_t i = 0; i < n; ++i) {
        solid_[i] = g.solid(i);
        fluid_[i] = lanes::mask_of(!g.solid(i));
    }
    volume_.assign(n, 1.0);
    for (std::size_t i = 0; i < n; ++i) {
        const position<D> at = positions::position_of(fields, i);
        // The last direction's width first: the reference's `np.outer(dy, dx)`.
        for (int d = D - 1; d >= 0; --d) {
            volume_[i] = volume_[i] * input.cell_widths[d][at[d]];
        }
    }
    vented_ = !input.vents[0].empty();
    for (int d = 0; d < D; ++d) {
        cell_widths_[d].assign(input.cell_widths[d].begin(), input.cell_widths[d].end());
        face_widths_[d].assign(input.face_widths[d].begin(), input.face_widths[d].end());
        const std::size_t cells = g.cells(d);
        const auto nodes = g.nodes(d);
        const auto between = g.interior_faces(d);
        far_share_[d].assign(cells + 1, 0.0);
        for (std::size_t f = 1; f < cells; ++f) {
            far_share_[d][f] = (nodes[f + 1] - between[f - 1]) / input.face_widths[d][f];
        }
        const layout::layout<D>& faces = faces_[d];
        const std::size_t count = faces.cell_count();
        area_[d].assign(count, 1.0);
        live_[d].assign(count, lanes::none);
        inner_[d].assign(count, lanes::none);
        if (vented_) {
            vents_[d] = std::make_unique<bool[]>(count);
        }
        for (std::size_t f = 0; f < count; ++f) {
            const position<D> at = positions::position_of(faces, f);
            for (int e = 0; e < D; ++e) {
                if (e != d) {
                    area_[d][f] = area_[d][f] * input.cell_widths[e][at[e]];
                }
            }
            // A face carries a flux where each cell beside it is not SOLID, and
            // diffusion crosses it where there are two.
            const bool has_low = at[d] > 0;
            const bool has_high = at[d] < cells;
            const bool low_open =
                !has_low || !g.solid(positions::index_of(fields, positions::below(at, d)));
            const bool high_open = !has_high || !g.solid(positions::index_of(fields, at));
            const bool open = low_open && high_open;
            inner_[d][f] = lanes::mask_of(open && has_low && has_high);
            const bool vent = vented_ && input.vents[d][f];
            live_[d][f] = lanes::mask_of(open || vent);
            if (vented_) {
                vents_[d][f] = vent;
            }
        }
        carried_[d].assign(count, 0.0);
        flux_[d].assign(count, 0.0);
        advective_[d].assign(count, 0.0);
        conductance_[d].assign(count, 0.0);
    }
    pairs_.resize(D);
    for (int c = 0; c < D; ++c) {
        pairs_[c].resize(D - 1);
        for (int q = 0; q < D - 1; ++q) {
            const int t = momentum::transverse(c, q);
            const layout::layout<D> corners = positions::faces_layout(fields, c, t);
            const std::size_t nt = g.cells(t);
            const auto nodes = g.nodes(t);
            const auto between = g.interior_faces(t);
            pair_setup& pair = pairs_[c][q];
            pair.span.resize(corners.cell_count());
            pair.defined.resize(corners.cell_count());
            for (std::size_t i = 0; i < corners.cell_count(); ++i) {
                const position<D> at = positions::position_of(corners, i);
                const std::size_t j = at[t];
                const auto buried = [&](const position<D>& face) {
                    return live_[c][positions::index_of(faces_[c], face)] == lanes::none;
                };
                const double corner =
                    j == 0 ? nodes[0] : (j == nt ? nodes[nt + 1] : between[j - 1]);
                // The domain edges are never buried.
                const double low =
                    j > 0 && buried(positions::below(at, t)) ? corner : nodes[j];
                const double high = j < nt && buried(at) ? corner : nodes[j + 1];
                const double span = high - low;
                pair.defined[i] = lanes::mask_of(span > 0.0);
                pair.span[i] = span > 0.0 ? span : 1.0;
            }
        }
    }
    for (int p = 0; p < pair_count<D>; ++p) {
        const pair_of_directions pair = pair_at<D>(p);
        corners_.push_back(positions::faces_layout(fields, pair.c, pair.t));
        shear_.push_back(corners_.back().cell_field());
    }
    for (auto* field : {&strain_, &production_, &ratio_, &eta_, &cube_, &rng_r_, &growth_eps_,
                        &decay_eps_, &dt_, &gamma_, &sink_, &k_star_, &eps_star_}) {
        field->assign(n, 0.0);
    }
    held_.assign(n, lanes::none);
    given_.assign(n, lanes::none);
}

// ---------------------------------------------------------------------------
// Checks

template <int D>
    requires(D >= 1)
void model<D>::require_velocity(const std::array<std::span<const double>, D>& velocity) const {
    for (int d = 0; d < D; ++d) {
        require(velocity[d].size() == faces_[d].cell_count(),
                "turbulence: a velocity component is not one value per face");
    }
}

template <int D>
    requires(D >= 1)
void model<D>::require_conditions(const conditions<D>& c) const {
    const std::size_t n = grid_->fields().cell_count();
    for (int d = 0; d < D; ++d) {
        require(c.inflow_k[d].size() == faces_[d].cell_count() &&
                    c.inflow_eps[d].size() == faces_[d].cell_count(),
                "turbulence: an inflow array is not one value per face");
        for (int q = 0; q < D - 1; ++q) {
            const std::size_t edge =
                faces_[d].cell_count() / faces_[d].cells(momentum::transverse(d, q));
            require(c.low[d][q].size() == edge && c.high[d][q].size() == edge,
                    "turbulence: a tangential velocity is not one value per face of the edge");
        }
    }
    require(c.eps_held.size() == n && c.eps_wall.size() == n && c.production_given.size() == n &&
                c.production.size() == n,
            "turbulence: a wall cells' array is not one value per cell");
}

template <int D>
    requires(D >= 1)
void model<D>::require_state(const state& s) const {
    const std::size_t n = grid_->fields().cell_count();
    require(s.k.size() == n && s.eps.size() == n && s.nu_t.size() == n,
            "turbulence: a field of the state is not one value per cell");
}

template <int D>
    requires(D >= 1)
bool model<D>::in_range(std::span<const double> field, bool positive) const {
    require(field.size() == fluid_.size(), "turbulence: a field is not one value per cell");
    mask bad = lanes::none;
    lanes::for_each_block(0, field.size(), [&](auto cells, std::size_t m) {
        bad |= positive ? out_of_range_cells<true>(cells, field.data(), fluid_.data(), m)
                        : out_of_range_cells<false>(cells, field.data(), fluid_.data(), m);
    });
    return bad == lanes::none;
}

template <int D>
    requires(D >= 1)
bool model<D>::zero_in_solid(std::span<const double> field) const {
    require(field.size() == fluid_.size(), "turbulence: a field is not one value per cell");
    mask bad = lanes::none;
    lanes::for_each_block(0, field.size(), [&](auto cells, std::size_t m) {
        bad |= not_zero_cells(cells, field.data(), fluid_.data(), m);
    });
    return bad == lanes::none;
}

// ---------------------------------------------------------------------------
// The parts

template <int D>
    requires(D >= 1)
void model<D>::eddy_viscosity(std::span<const double> k, std::span<const double> eps,
                              std::span<double> out) const {
    require(k.size() == fluid_.size() && eps.size() == k.size() && out.size() == k.size(),
            "turbulence: a field is not one value per cell");
    lanes::for_each_block(0, k.size(), [&](auto cells, std::size_t m) {
        viscosity_cells(cells, variant_.c_mu, k.data(), eps.data(), fluid_.data(), m, out.data());
    });
}

template <int D>
    requires(D >= 1)
void model<D>::carry(const std::array<std::span<const double>, D>& velocity) {
    for (int d = 0; d < D; ++d) {
        lanes::for_each_block(0, carried_[d].size(), [&](auto cells, std::size_t m) {
            carried_cells(cells, velocity[d].data(), live_[d].data(), m, carried_[d].data());
        });
    }
}

template <int D>
    requires(D >= 1)
template <int C, int T, bool Add>
void model<D>::gradient_rows(const double* faces, std::span<const double> low,
                             std::span<const double> high, double* out) {
    const layout::layout<D>& face_fields = faces_[C];
    const layout::layout<D>& corners = corners_[pair_index<D>(C, T)];
    const pair_setup& pair = pairs_[C][momentum::slot_of(C, T)];
    const std::size_t n = corners.cells(0);
    const std::size_t nt = grid_->cells(T);
    for (const auto& line : corners.lines(0)) {
        const position<D> at = positions::position_of(corners, line.cell);
        gradient_row row{nullptr, nullptr, pair.span.data() + line.cell,
                         pair.defined.data() + line.cell};
        double* corner = out + line.cell;
        if constexpr (T == 0) {
            // The gradient along the rows: the row's two end corners lie on the
            // domain's edges, whose values stand for the faces beyond them.
            const double* face = faces + positions::index_of(face_fields, at);
            const std::size_t edge = positions::index_without(face_fields, at, 0);
            const double first[2] = {low[edge], face[0]};
            const double last[2] = {face[nt - 1], high[edge]};
            const auto end = [&](const double* values, std::size_t i) {
                gradient_cells<false, Add>(
                    lanes::cells<1>{},
                    gradient_row{values, values + 1, row.span + i, row.defined + i}, 0,
                    corner + i);
            };
            end(first, 0);
            row.high = face;
            lanes::for_each_block(1, nt, [&](auto cells, std::size_t m) {
                gradient_cells<true, Add>(cells, row, m, corner);
            });
            end(last, nt);
        } else {
            // Across the rows: the faces' rows below and above the corners', an
            // edge's values standing for a missing one.
            const std::size_t edge = positions::index_without(face_fields, at, T);
            row.low = at[T] > 0
                          ? faces + positions::index_of(face_fields, positions::below(at, T))
                          : low.data() + edge;
            row.high = at[T] < nt ? faces + positions::index_of(face_fields, at)
                                  : high.data() + edge;
            lanes::for_each_block(0, n, [&](auto cells, std::size_t m) {
                gradient_cells<false, Add>(cells, row, m, corner);
            });
        }
    }
}

template <int D>
    requires(D >= 1)
template <bool Add>
void model<D>::gradient(int c, int q, const double* faces, std::span<const double> low,
                        std::span<const double> high, double* out) {
    const int t = momentum::transverse(c, q);
    [&]<int... P>(std::integer_sequence<int, P...>) {
        const auto one = [&]<int C, int T>() {
            if constexpr (C != T) {
                if (c == C && t == T) {
                    gradient_rows<C, T, Add>(faces, low, high, out);
                }
            }
        };
        (one.template operator()<P / D, P % D>(), ...);
    }(std::make_integer_sequence<int, D * D>{});
}

template <int D>
    requires(D >= 1)
void model<D>::corner_gradient(int c, int q, std::span<const double> velocity,
                               std::span<const double> low, std::span<const double> high,
                               std::span<double> out) {
    require(c >= 0 && c < D && q >= 0 && q < D - 1, "turbulence: there is no such pair");
    const int t = momentum::transverse(c, q);
    const std::size_t edge = faces_[c].cell_count() / faces_[c].cells(t);
    require(velocity.size() == faces_[c].cell_count() && low.size() == edge &&
                high.size() == edge && out.size() == pairs_[c][q].span.size(),
            "turbulence: an array of a corner gradient is not the pair's length");
    lanes::for_each_block(0, carried_[c].size(), [&](auto cells, std::size_t m) {
        carried_cells(cells, velocity.data(), live_[c].data(), m, carried_[c].data());
    });
    gradient<false>(c, q, carried_[c].data(), low, high, out.data());
}

// The strain of the carried velocities.
template <int D>
    requires(D >= 1)
void model<D>::strain(const conditions<D>& c, double* out) {
    const auto& fields = grid_->fields();
    const std::size_t n0 = fields.cells(0);
    for_each_pair<D>([&]<int P>() {
        constexpr int a = pair_at<D>(P).c;
        constexpr int b = pair_at<D>(P).t;
        constexpr int qa = momentum::slot_of(a, b);
        constexpr int qb = momentum::slot_of(b, a);
        double* shear = shear_[P].data();
        gradient<false>(a, qa, carried_[a].data(), c.low[a][qa], c.high[a][qa], shear);
        gradient<true>(b, qb, carried_[b].data(), c.low[b][qb], c.high[b][qb], shear);
    });
    for (const auto& line : fields.lines(0)) {
        const position<D> at = positions::position_of(fields, line.cell);
        strain_row<D> row;
        row.low[0] = carried_[0].data() + line.face;
        row.width0 = cell_widths_[0].data();
        for (int d = 1; d < D; ++d) {
            const std::size_t face = fields.low_face(d, line.cell);
            row.low[d] = carried_[d].data() + face;
            row.high[d] = row.low[d] + fields.face_stride(d);
            row.width[d] = cell_widths_[d][at[d]];
        }
        for (int p = 0; p < pair_count<D>; ++p) {
            const pair_of_directions pair = pair_at<D>(p);
            const layout::layout<D>& corners = corners_[static_cast<std::size_t>(p)];
            const double* shear = shear_[static_cast<std::size_t>(p)].data() +
                                  positions::index_of(corners, at);
            const std::size_t up_c = corners.cell_stride(pair.c);
            const std::size_t up_t = corners.cell_stride(pair.t);
            row.shear[p][0] = shear;
            row.shear[p][1] = shear + up_c;
            row.shear[p][2] = shear + up_t;
            row.shear[p][3] = shear + up_t + up_c;
        }
        row.fluid = fluid_.data() + line.cell;
        double* s2 = out + line.cell;
        const auto walk = [n0, s2](const strain_row<D> in) {
            lanes::for_each_block(0, n0, [&](auto cells, std::size_t m) {
                strain_cells<D>(cells, in, m, s2);
            });
        };
        walk(row);
    }
}

template <int D>
    requires(D >= 1)
void model<D>::strain_squared(const std::array<std::span<const double>, D>& velocity,
                              const conditions<D>& c, std::span<double> out) {
    require_velocity(velocity);
    require_conditions(c);
    require(out.size() == fluid_.size(), "turbulence: a field is not one value per cell");
    carry(velocity);
    strain(c, out.data());
}

template <int D>
    requires(D >= 1)
void model<D>::form_terms(const state& s, const conditions<D>& c) {
    const std::size_t n = fluid_.size();
    strain(c, strain_.data());
    lanes::for_each_block(0, n, [&](auto cells, std::size_t m) {
        flag_cells(cells, c.production_given.data(), m, given_.data());
        flag_cells(cells, c.eps_held.data(), m, held_.data());
    });
    const terms_cells_in in{s.k.data(),          s.eps.data(),  s.nu_t.data(), strain_.data(),
                            c.production.data(), given_.data(), fluid_.data()};
    lanes::for_each_block(0, n, [&](auto cells, std::size_t m) {
        production_cells(cells, in, m, production_.data(), ratio_.data());
    });
    if (!variant_.has_r) {
        lanes::for_each_block(0, n, [&](auto cells, std::size_t m) {
            standard_cells(cells, variant_.c_1, variant_.c_2, production_.data(), ratio_.data(),
                           m, growth_eps_.data(), decay_eps_.data());
        });
        return;
    }
    // eta and its cube call the C library, a cell at a time.
    for (std::size_t i = 0; i < n; ++i) {
        const double eta = solid_[i] ? 0.0 : std::sqrt(strain_[i]) * s.k[i] / s.eps[i];
        eta_[i] = eta;
        cube_[i] = elementary::power(eta, 3.0);
    }
    const rng_row rng{eta_.data(),   cube_.data(),       s.eps.data(),
                      ratio_.data(), production_.data(), fluid_.data()};
    const constants v = variant_;
    lanes::for_each_block(0, n, [&](auto cells, std::size_t m) {
        rng_cells(cells, rng, v, m, rng_r_.data(), growth_eps_.data(), decay_eps_.data());
    });
}

template <int D>
    requires(D >= 1)
void model<D>::terms(const state& s, const std::array<std::span<const double>, D>& velocity,
                     const conditions<D>& c) {
    require_state(s);
    require_velocity(velocity);
    require_conditions(c);
    carry(velocity);
    form_terms(s, c);
}

template <int D>
    requires(D >= 1)
bool model<D>::form_time_step() {
    const auto& fields = grid_->fields();
    const std::size_t n0 = fields.cells(0);
    std::array<double, lanes::width> largest{};
    mask any = lanes::none;
    for (const auto& line : fields.lines(0)) {
        const position<D> at = positions::position_of(fields, line.cell);
        rate_row<D> row;
        row.low[0] = carried_[0].data() + line.face;
        row.width0 = cell_widths_[0].data();
        for (int d = 1; d < D; ++d) {
            const std::size_t face = fields.low_face(d, line.cell);
            row.low[d] = carried_[d].data() + face;
            row.high[d] = row.low[d] + fields.face_stride(d);
            row.width[d] = cell_widths_[d][at[d]];
        }
        row.fluid = fluid_.data() + line.cell;
        double* dt = dt_.data() + line.cell;
        const auto walk = [&](const rate_row<D> in) {
            lanes::for_each_block(0, n0, [&](auto cells, std::size_t m) {
                any |= step_cells<D>(cells, in, cfl_, m, dt, largest.data());
            });
        };
        walk(row);
    }
    if (any == lanes::none) {
        return false;
    }
    double cap = 0.0;
    for (const double lane : largest) {
        cap = lane > cap ? lane : cap;
    }
    lanes::for_each_block(0, dt_.size(), [&](auto cells, std::size_t m) {
        cap_cells(cells, cap, m, dt_.data());
    });
    // The volume fluxes of the same velocities, which the advection reads.
    for (int d = 0; d < D; ++d) {
        lanes::for_each_block(0, flux_[d].size(), [&](auto cells, std::size_t m) {
            product_cells(cells, carried_[d].data(), area_[d].data(), m, flux_[d].data());
        });
    }
    return true;
}

template <int D>
    requires(D >= 1)
bool model<D>::pseudo_time_step(const std::array<std::span<const double>, D>& velocity) {
    require_velocity(velocity);
    carry(velocity);
    return form_time_step();
}

template <int D>
    requires(D >= 1)
void model<D>::advective_fluxes(std::span<const double> q,
                                const std::array<std::span<const double>, D>& inflow) {
    for (int d = 0; d < D; ++d) {
        const std::span<const bool> vents =
            vented_ ? std::span<const bool>(vents_[d].get(), flux_[d].size())
                    : std::span<const bool>();
        scalar::advective_flux<D>(scheme_, d, q, flux_[d], inflow[d], upwind_, vents,
                                  advective_[d]);
    }
}

// Steps 1 and 2 of one quantity into `out`: with a growth the explicit field,
// without one the advected field; `held` cells take `held_value`.
template <int D>
    requires(D >= 1)
void model<D>::explicit_field(std::span<const double> q,
                              const std::array<std::span<const double>, D>& inflow,
                              const double* growth, const mask* held, const double* held_value,
                              double* out) {
    advective_fluxes(q, inflow);
    const auto& fields = grid_->fields();
    const std::size_t n0 = fields.cells(0);
    for (const auto& line : fields.lines(0)) {
        explicit_row<D> row;
        row.low[0] = advective_[0].data() + line.face;
        for (int d = 1; d < D; ++d) {
            row.low[d] = advective_[d].data() + fields.low_face(d, line.cell);
            row.high[d] = row.low[d] + fields.face_stride(d);
        }
        row.q = q.data() + line.cell;
        row.dt = dt_.data() + line.cell;
        row.volume = volume_.data() + line.cell;
        row.fluid = fluid_.data() + line.cell;
        if (growth == nullptr) {
            explicit_row_cells<part::advected>(row, n0, out + line.cell);
            continue;
        }
        row.growth = growth + line.cell;
        if (held == nullptr) {
            explicit_row_cells<part::grown>(row, n0, out + line.cell);
            continue;
        }
        row.held = held + line.cell;
        row.held_value = held_value + line.cell;
        explicit_row_cells<part::held>(row, n0, out + line.cell);
    }
}

template <int D>
    requires(D >= 1)
void model<D>::advected(std::span<const double> q,
                        const std::array<std::span<const double>, D>& inflow,
                        std::span<double> out) {
    require(q.size() == fluid_.size() && out.size() == q.size(),
            "turbulence: a field is not one value per cell");
    for (int d = 0; d < D; ++d) {
        require(inflow[d].size() == faces_[d].cell_count(),
                "turbulence: an inflow array is not one value per face");
    }
    explicit_field(q, inflow, nullptr, nullptr, nullptr, out.data());
}

template <int D>
    requires(D >= 1)
void model<D>::explicit_fields(const state& s, const conditions<D>& c, std::span<double> k_star,
                               std::span<double> eps_star) {
    require_state(s);
    require_conditions(c);
    require(k_star.size() == fluid_.size() && eps_star.size() == k_star.size(),
            "turbulence: a field is not one value per cell");
    explicit_field(s.k, c.inflow_k, production_.data(), nullptr, nullptr, k_star.data());
    explicit_field(s.eps, c.inflow_eps, growth_eps_.data(), held_.data(), c.eps_wall.data(),
                   eps_star.data());
}

template <int D>
    requires(D >= 1)
void model<D>::conductances(quantity which, std::span<const double> nu_t,
                            const std::array<std::span<double>, D>& out) {
    require(nu_t.size() == fluid_.size(), "turbulence: a field is not one value per cell");
    const double sigma = which == quantity::k ? variant_.sigma_k : variant_.sigma_e;
    lanes::for_each_block(0, nu_t.size(), [&](auto cells, std::size_t m) {
        gamma_cells(cells, nu_t.data(), fluid_.data(), nu_, sigma, m, gamma_.data());
    });
    const auto& fields = grid_->fields();
    const std::size_t n0 = fields.cells(0);
    for (int d = 0; d < D; ++d) {
        require(out[d].size() == faces_[d].cell_count(),
                "turbulence: a conductance array is not one value per face");
        double* g = out[d].data();
        const std::size_t stride = fields.cell_stride(d);
        for (const auto& line : fields.lines(0)) {
            if (d == 0) {
                // The row's interior faces; its two domain faces conduct nothing.
                const conductance_row row{nullptr,
                                          gamma_.data() + line.cell,
                                          far_share_[0].data(),
                                          face_widths_[0].data(),
                                          area_[0].data() + line.face,
                                          inner_[0].data() + line.face};
                double* face = g + line.face;
                face[0] = 0.0;
                face[n0] = 0.0;
                lanes::for_each_block(1, n0, [&](auto cells, std::size_t m) {
                    conductance_cells<true>(cells, row, m, face);
                });
                continue;
            }
            const std::size_t at = (line.cell / stride) % fields.cells(d);
            const std::size_t low_face = fields.low_face(d, line.cell);
            if (at == 0) {
                std::fill_n(g + low_face, n0, 0.0);
            } else {
                const conductance_row row{gamma_.data() + line.cell - stride,
                                          gamma_.data() + line.cell,
                                          far_share_[d].data() + at,
                                          face_widths_[d].data() + at,
                                          area_[d].data() + low_face,
                                          inner_[d].data() + low_face};
                double* face = g + low_face;
                lanes::for_each_block(0, n0, [&](auto cells, std::size_t m) {
                    conductance_cells<false>(cells, row, m, face);
                });
            }
            if (at + 1 == fields.cells(d)) {
                std::fill_n(g + low_face + fields.face_stride(d), n0, 0.0);
            }
        }
    }
}

template <int D>
    requires(D >= 1)
scalar::implicit_result model<D>::solve(std::span<const double> star,
                                        std::span<const double> decay,
                                        std::span<const bool> held, std::span<double> out) {
    lanes::for_each_block(0, sink_.size(), [&](auto cells, std::size_t m) {
        product_cells(cells, volume_.data(), decay.data(), m, sink_.data());
    });
    scalar::implicit_input<D> in{};
    in.c_star = star;
    in.volume = volume_;
    in.dt_cells = dt_;
    for (int d = 0; d < D; ++d) {
        in.conductance[d] = conductance_[d];
    }
    in.sink = sink_;
    in.solid = std::span<const bool>(solid_.get(), fluid_.size());
    in.held = held;
    in.tol = tol_;
    in.max_iter = max_iter_;
    in.method = scalar::implicit_method::conjugate_gradient;
    scalar::implicit_result result;
    // The one refusal is a leftover bound's, and there is none.
    (void)scalar::implicit_step<D>(in, solver_, out, result);
    return result;
}

// Whether a quantity is not positive and finite in some non-SOLID cell, and
// those cells (`_check_positive`).
template <int D>
    requires(D >= 1)
bool model<D>::find_bad(std::span<const double> q, bad_cells& bad) const {
    if (in_range(q, true)) {
        return false;
    }
    bad = bad_cells{};
    bool started = false;
    for (std::size_t i = 0; i < q.size(); ++i) {
        if (solid_[i]) {
            continue;
        }
        if (!positive_and_finite(q[i])) {
            if (bad.count == 0) {
                bad.first = i;
                bad.value = q[i];
            }
            ++bad.count;
        }
        // `np.min`: a NaN stays.
        bad.least = started ? scalar::np_minimum(bad.least, q[i]) : q[i];
        started = true;
    }
    return true;
}

template <int D>
    requires(D >= 1)
step_result model<D>::step(const state& s,
                           const std::array<std::span<const double>, D>& velocity,
                           const conditions<D>& c, std::span<double> k, std::span<double> eps,
                           std::span<double> nu_t) {
    require_state(s);
    require_velocity(velocity);
    require_conditions(c);
    require(k.size() == fluid_.size() && eps.size() == k.size() && nu_t.size() == k.size(),
            "turbulence: an output is not one value per cell");
    step_result result;
    carry(velocity);
    if (!form_time_step()) {
        result.status = step_status::nothing_moves;
        return result;
    }
    form_terms(s, c);
    explicit_fields(s, c, k_star_, eps_star_);
    const auto conductance = [&] {
        std::array<std::span<double>, D> g;
        for (int d = 0; d < D; ++d) {
            g[d] = conductance_[d];
        }
        return g;
    }();
    conductances(quantity::k, s.nu_t, conductance);
    result.solves[0] = solve(k_star_, ratio_, {}, k);
    conductances(quantity::eps, s.nu_t, conductance);
    result.solves[1] = solve(eps_star_, decay_eps_, c.eps_held, eps);
    if (find_bad(k, result.bad)) {
        result.status = step_status::k_not_positive;
        return result;
    }
    if (find_bad(eps, result.bad)) {
        result.status = step_status::eps_not_positive;
        return result;
    }
    lanes::for_each_block(0, k.size(), [&](auto cells, std::size_t m) {
        clear_solid_cells(cells, fluid_.data(), m, k.data());
        clear_solid_cells(cells, fluid_.data(), m, eps.data());
    });
    eddy_viscosity(k, eps, nu_t);
    return result;
}

template class model<1>;
template class model<2>;

}  // namespace cfdcore::turbulence
