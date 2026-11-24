#include "cfdcore/momentum.hpp"

#include "cfdcore/scalar.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace cfdcore::momentum {

namespace {

using lanes::mask;

// ---------------------------------------------------------------------------
// Directions

// Whether direction d lies along the rows, or crosses them.
template <int d>
inline constexpr bool along_row = d == 0;

// The transverse directions of a component, as a type a block folds over.
template <int... T>
struct directions {};

template <int C, int... Q>
directions<transverse(C, Q)...> transverse_of(std::integer_sequence<int, Q...>);

template <int D, int C>
using transverse_directions = decltype(transverse_of<C>(std::make_integer_sequence<int, D - 1>{}));

// Calls body.operator()<T>() for each direction T transverse to component C, by slot.
template <int C, int... T, typename Body>
CFDCORE_INLINE void for_each_transverse(directions<T...>, [[maybe_unused]] Body&& body) {
    (body.template operator()<T>(), ...);
}

// The entries an operand struct holds per slot; one unused where there is none.
template <int... T>
inline constexpr std::size_t slots = sizeof...(T) == 0 ? 1 : sizeof...(T);

// An operand that varies along direction d, at element i of a row: per element
// where d lies along the rows, the row's one value otherwise.
template <int d, typename Value>
CFDCORE_INLINE Value along(const Value* p, std::size_t i) {
    return p[along_row<d> ? i : 0];
}

// An operand of a domain edge normal to direction d, which varies along every other.
template <int d, typename Value>
CFDCORE_INLINE Value on_edge(const Value* p, std::size_t i) {
    return p[along_row<d> ? 0 : i];
}

// The streamwise face on the low side of unknown i along its own direction d:
// cell i - 1 of the row, or cell i of the row below. The high one is cell i.
template <int d>
CFDCORE_INLINE double low_side(const double* below, const double* row, std::size_t i) {
    return along_row<d> ? row[i - 1] : below[i];
}

// The transverse face on the high side of unknown i along direction d: corner
// i + 1 of the row, or corner i of the row above. The low one is corner i.
template <int d>
CFDCORE_INLINE double high_side(const double* row, const double* above, std::size_t i) {
    return along_row<d> ? row[i + 1] : above[i];
}

// `x` times the area of a streamwise face of component C: the control volume's
// width along each transverse direction in turn.
template <int C, int... T>
CFDCORE_INLINE double times_area(directions<T...>, double x,
                                 [[maybe_unused]] const double* const* widths,
                                 [[maybe_unused]] std::size_t i) {
    ((x = x * along<T>(widths[slot_of(C, T)], i)), ...);
    return x;
}

// `np.maximum(x, 0.0)`: a NaN stays, and -0.0 gives 0.0. The comparison is one
// and its mask is kept, where `scalar::np_maximum` of a constant zero folds
// to a branch that GCC 13 leaves in the loop (the block stays scalar).
CFDCORE_INLINE double positive_part(double x) {
    return lanes::keep(lanes::mask_of(!(x <= 0.0)), x);
}

// ---------------------------------------------------------------------------
// Conductances and mass fluxes

// What the conductance of a row of streamwise faces reads.
template <std::size_t Slots>
struct cell_conductance_run {
    const double* mu;
    const double* width_t[Slots];
    const double* width_s;
};

// The conductance of N streamwise faces with a viscosity field: mu dt / ds.
template <int C, int... T, std::size_t N>
inline void cell_conductance(lanes::cells<N>, directions<T...> across,
                             const cell_conductance_run<slots<T...>>& in, std::size_t first,
                             double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        out[i] = times_area<C>(across, in.mu[i], in.width_t, i) / along<C>(in.width_s, i);
    }
}

// What the mass flux through a row of streamwise faces reads: the component's
// values at the nodes on either side.
template <std::size_t Slots>
struct cell_flux_run {
    const double* low;
    const double* high;
    const double* width_t[Slots];
    const double* conductance;
    double half_rho;
};

// The mass flux through N streamwise faces and the coefficients it gives the
// unknowns on either side: toward s+ for the one on its low side, toward s- for
// the one on its high side, first-order upwind.
template <int C, int... T, std::size_t N>
inline void cell_flux(lanes::cells<N>, directions<T...> across,
                      const cell_flux_run<slots<T...>>& in, std::size_t first,
                      double* CFDCORE_RESTRICT flux, double* CFDCORE_RESTRICT plus,
                      double* CFDCORE_RESTRICT minus) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double f =
            times_area<C>(across, in.half_rho * (in.low[i] + in.high[i]), in.width_t, i);
        flux[i] = f;
        plus[i] = in.conductance[i] + positive_part(-f);
        minus[i] = in.conductance[i] + positive_part(f);
    }
}

// What the mass flux through a row of transverse faces reads: the velocity
// along the transverse direction in the two half cells along s.
struct corner_flux_run {
    const double* low;
    const double* high;
    const double* width_low;
    const double* width_high;
    const double* conductance;
    double half_rho;
};

// The mass flux through N transverse faces and the coefficients it gives the
// unknowns on either side, toward t+ and toward t-, as `cell_flux`.
template <int C, std::size_t N>
inline void corner_flux(lanes::cells<N>, const corner_flux_run& in, std::size_t first,
                        double* CFDCORE_RESTRICT flux, double* CFDCORE_RESTRICT plus,
                        double* CFDCORE_RESTRICT minus) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double f = in.half_rho * (in.low[i] * along<C>(in.width_low, i) +
                                        in.high[i] * along<C>(in.width_high, i));
        flux[i] = f;
        plus[i] = in.conductance[i] + positive_part(-f);
        minus[i] = in.conductance[i] + positive_part(f);
    }
}

// What the conductance of a row of transverse faces reads.
struct corner_conductance_run {
    const double* viscosity;
    const double* wall;
    const double* face_width;
    const double* reach;
    const double* fixed;
    const mask* takes_wall;
    const mask* conducts;
};

// The conductance of N transverse faces where it is not the scalar path's
// fixed one: with a viscosity field (`Field`), the corner viscosity's; with a
// wall viscosity (`Wall`), that one at the faces of the wall stencil.
template <int C, bool Field, bool Wall, std::size_t N>
inline void corner_conductance(lanes::cells<N>, const corner_conductance_run& in,
                               std::size_t first, double* CFDCORE_RESTRICT out) {
    static_assert(Field || Wall);
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        if constexpr (Field) {
            double m = in.viscosity[i];
            if constexpr (Wall) {
                m = lanes::select(in.takes_wall[i], in.wall[i], m);
            }
            out[i] = lanes::keep(in.conducts[i], m * along<C>(in.face_width, i) / in.reach[i]);
        } else {
            out[i] = lanes::select(in.takes_wall[i],
                                   in.wall[i] * along<C>(in.face_width, i) / in.reach[i],
                                   in.fixed[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// The deferred correction: QUICK minus upwind

// What the QUICK face values of a run of streamwise faces read, each pointer at
// the run's first face: the component's values at the nodes k - 1, k, k + 1 and
// k + 2, and the weights, per face or the run's one set. At the domain's two
// ends the far node that does not exist is the other one: the weights of the
// boundary form are for it, and no choice between the two is read there.
struct stream_run {
    const double* far_low;
    const double* low;
    const double* high;
    const double* far_high;
    const double* std_pos[3];
    const double* std_neg[3];
    const double* alt_pos[3];
    const double* alt_neg[3];
    const mask* blocked_pos;
    const mask* blocked_neg;
    const double* flux;
};

// The (QUICK - upwind) flux through N streamwise faces. `Inner` reads the nodes
// of a row along s off one pointer. `Obstacles` takes the boundary form where
// the far node lies beyond an obstacle's wall node (`_obstacle_streamwise`).
//
// The value is mostly choices, so both forms were timed under the project's
// flags (docs/reports/phase6_momentum.md, section 3): the block with a select
// per choice, kept, and a loop of one face at a time with a branch per choice.
template <int C, bool Inner, bool Obstacles, std::size_t N>
inline void cell_deferred(lanes::cells<N>, const stream_run& in, std::size_t first,
                          double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double lo = in.low[i];
        const double hi = Inner ? in.low[i + 1] : in.high[i];
        const double below = Inner ? in.low[i - 1] : in.far_low[i];
        const double above = Inner ? in.low[i + 2] : in.far_high[i];
        const double f = in.flux[i];
        const mask positive = lanes::mask_of(f > 0.0);
        const double q_pos = along<C>(in.std_pos[0], i) * below +
                             along<C>(in.std_pos[1], i) * lo + along<C>(in.std_pos[2], i) * hi;
        const double q_neg = along<C>(in.std_neg[0], i) * above +
                             along<C>(in.std_neg[1], i) * hi + along<C>(in.std_neg[2], i) * lo;
        double q = lanes::select(positive, q_pos, q_neg);
        if constexpr (Obstacles) {
            const double alt_pos = along<C>(in.alt_pos[0], i) * above +
                                   along<C>(in.alt_pos[1], i) * lo +
                                   along<C>(in.alt_pos[2], i) * hi;
            const double alt_neg = along<C>(in.alt_neg[0], i) * below +
                                   along<C>(in.alt_neg[1], i) * hi +
                                   along<C>(in.alt_neg[2], i) * lo;
            q = lanes::select(positive & in.blocked_pos[i], alt_pos, q);
            q = lanes::select(~positive & in.blocked_neg[i], alt_neg, q);
        }
        out[i] = (q - lanes::select(positive, lo, hi)) * f;
    }
}

// What the QUICK face values of a run of transverse faces read, each pointer
// at the run's first face: the component's values in the rows of nodes r - 2,
// r - 1, r and r + 1 about face r, a domain edge's value standing for a missing
// row.
struct cross_run {
    const double* far_low;
    const double* low;
    const double* high;
    const double* far_high;
    const double* pos[3];
    const double* neg[3];
    const double* wall_pos[2];
    const double* wall_neg[2];
    const mask* far_wall_low;
    const mask* far_wall_high;
    const mask* wall_face;
    const double* flux;
};

// The (QUICK - upwind) flux through N interior faces transverse along direction
// T. `Inner` reads the nodes of a row along T off one pointer. `Obstacles`
// takes the wall as the far node beside an obstacle face and adds nothing at
// one (`_obstacle_transverse`).
template <int T, bool Inner, bool Obstacles, std::size_t N>
inline void corner_deferred(lanes::cells<N>, const cross_run& in, std::size_t first,
                            double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double lo = in.low[i];
        const double hi = Inner ? in.low[i + 1] : in.high[i];
        const double below = Inner ? in.low[i - 1] : in.far_low[i];
        const double above = Inner ? in.low[i + 2] : in.far_high[i];
        const double f = in.flux[i];
        const mask positive = lanes::mask_of(f > 0.0);
        const double q_pos = along<T>(in.pos[0], i) * below + along<T>(in.pos[1], i) * lo +
                             along<T>(in.pos[2], i) * hi;
        const double q_neg = along<T>(in.neg[0], i) * above + along<T>(in.neg[1], i) * hi +
                             along<T>(in.neg[2], i) * lo;
        double q = lanes::select(positive, q_pos, q_neg);
        if constexpr (Obstacles) {
            const double wall_pos =
                along<T>(in.wall_pos[0], i) * lo + along<T>(in.wall_pos[1], i) * hi;
            const double wall_neg =
                along<T>(in.wall_neg[0], i) * hi + along<T>(in.wall_neg[1], i) * lo;
            q = lanes::select(positive & in.far_wall_low[i], wall_pos, q);
            q = lanes::select(~positive & in.far_wall_high[i], wall_neg, q);
        }
        double flux = (q - lanes::select(positive, lo, hi)) * f;
        if constexpr (Obstacles) {
            flux = lanes::keep(~in.wall_face[i], flux);
        }
        out[i] = flux;
    }
}

// The node beyond a domain edge for N faces: the wall value, or across a
// zero-gradient edge the adjacent node's.
template <std::size_t N>
inline void edge_nodes(lanes::cells<N>, const mask* known, const double* value,
                       const double* adjacent, std::size_t first,
                       double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        out[i] = lanes::select(known[i], value[i], adjacent[i]);
    }
}

// ---------------------------------------------------------------------------
// The viscosity field

// `_harmonic` of N pairs of cells across a face transverse along direction T,
// the SOLID one of a pair taking the other's value. The mean of two SOLID cells
// is whatever they hold (a NaN, say): no unknown's equation reads it, so the
// field needs no copy with SOLID cells at air's viscosity, as `_check_mu_eff`
// makes.
template <int T, std::size_t N>
inline void harmonic_cells(lanes::cells<N>, const double* low, const double* high,
                           const mask* solid_low, const mask* solid_high,
                           const double* frac_low, const double* frac_high, std::size_t first,
                           double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double a = lanes::select(solid_low[i] & ~solid_high[i], high[i], low[i]);
        const double b = lanes::select(solid_high[i] & ~solid_low[i], low[i], high[i]);
        const double frac_a = along<T>(frac_low, i);
        const double frac_b = along<T>(frac_high, i);
        out[i] = a + (b - a) * (a * frac_b) / (frac_a * b + frac_b * a);
    }
}

// `_lerp` of N pairs of rows' values across an interior position along s: the
// values at i - 1 and i of one row along s, or at i of two rows it crosses.
template <int C, std::size_t N>
inline void mean_cells(lanes::cells<N>, const double* below, const double* row,
                       const double* frac_high, std::size_t first,
                       double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double a = low_side<C>(below, row, i);
        const double b = row[i];
        out[i] = a + (b - a) * along<C>(frac_high, i);
    }
}

// The corner viscosity of N faces where an unknown reads it, air's elsewhere.
template <std::size_t N>
inline void read_cells(lanes::cells<N>, const mask* read, const double* viscosity, double air,
                       std::size_t first, double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        out[i] = lanes::select(read[i], viscosity[i], air);
    }
}

// What the stress flux through a row of streamwise faces reads.
template <std::size_t Slots>
struct cell_stress_run {
    const double* low;
    const double* high;
    const double* width_s;
    const double* width_t[Slots];
};

// The stress flux through N streamwise faces: du/ds dt (`_stress_source`).
template <int C, int... T, std::size_t N>
inline void cell_stress(lanes::cells<N>, directions<T...> across,
                        const cell_stress_run<slots<T...>>& in, std::size_t first,
                        double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        out[i] = times_area<C>(across, (in.high[i] - in.low[i]) / along<C>(in.width_s, i),
                               in.width_t, i);
    }
}

// The stress flux through N transverse faces: the gradient along s of the
// velocity along the transverse direction, times the face's width. The quotient
// times the same width is the reference's, and rounds twice.
template <int C, std::size_t N>
inline void corner_stress(lanes::cells<N>, const double* low, const double* high,
                          const double* face_width, std::size_t first,
                          double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double w = along<C>(face_width, i);
        out[i] = (high[i] - low[i]) / w * w;
    }
}

// ---------------------------------------------------------------------------
// The equations of a row of unknowns

// What the streamwise faces about a row of unknowns hold of the mass flux:
// those on their high side, and on their low side in the row `_below`.
struct stream_faces {
    const double* flux;
    const double* flux_below;
    const double* plus;
    const double* minus;
    const double* minus_below;
};

// What the faces transverse along one direction about a row of unknowns hold
// of the mass flux: those on their low side, and on their high side in the row
// `_above`.
struct cross_faces {
    const double* flux;
    const double* flux_above;
    const double* plus;
    const double* plus_above;
    const double* minus;
};

// The neighbour coefficients of N unknowns along their own direction, zero at
// faces that are not unknowns.
template <int C, std::size_t N>
inline void stream_neighbours(lanes::cells<N>, const stream_faces& in, const mask* unknown,
                              std::size_t first, double* CFDCORE_RESTRICT plus,
                              double* CFDCORE_RESTRICT minus) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        plus[i] = lanes::keep(unknown[i], in.plus[i]);
        minus[i] = lanes::keep(unknown[i], low_side<C>(in.minus_below, in.minus, i));
    }
}

// The neighbour coefficients of N unknowns along transverse direction T, zero
// also toward an obstacle's face. A domain edge's are `edge_unknowns`'.
template <int T, std::size_t N>
inline void cross_neighbours(lanes::cells<N>, const cross_faces& in, const mask* unknown,
                             const mask* couples_high, const mask* couples_low,
                             std::size_t first, double* CFDCORE_RESTRICT plus,
                             double* CFDCORE_RESTRICT minus) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double a_n = high_side<T>(in.plus, in.plus_above, i);
        const double a_s = in.minus[i];
        plus[i] = lanes::keep(unknown[i] & couples_high[i], a_n);
        minus[i] = lanes::keep(unknown[i] & couples_low[i], a_s);
    }
}

// What the diagonal of a row of unknowns reads.
template <std::size_t Slots>
struct diagonal_run {
    stream_faces stream;
    cross_faces cross[Slots];
    const mask* unknown;
    double alpha;
};

// The diagonal the sweeps divide by: relaxed, and 1 where a_P is not positive.
CFDCORE_INLINE double relaxed_diagonal(double a_p, double alpha) {
    return lanes::select(lanes::mask_of(a_p > 0.0), a_p / alpha, 1.0);
}

// The un-relaxed diagonal of N unknowns, zero at faces that are not: the
// neighbour coefficients' sum and then the net mass flux, each direction by
// direction; and the relaxed one. `known`, the known neighbours' part of the
// source, starts at zero.
template <int C, int... T, std::size_t N>
inline void diagonals(lanes::cells<N>, directions<T...>, const diagonal_run<slots<T...>>& in,
                      std::size_t first, double* CFDCORE_RESTRICT a_p,
                      double* CFDCORE_RESTRICT relaxed, double* CFDCORE_RESTRICT known) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        double sum = in.stream.plus[i] + low_side<C>(in.stream.minus_below, in.stream.minus, i);
        ((sum = sum +
                high_side<T>(in.cross[slot_of(C, T)].plus, in.cross[slot_of(C, T)].plus_above, i) +
                in.cross[slot_of(C, T)].minus[i]),
         ...);
        double net = in.stream.flux[i] - low_side<C>(in.stream.flux_below, in.stream.flux, i);
        ((net = net +
                high_side<T>(in.cross[slot_of(C, T)].flux, in.cross[slot_of(C, T)].flux_above, i) -
                in.cross[slot_of(C, T)].flux[i]),
         ...);
        const double p = lanes::keep(in.unknown[i], sum + net);
        a_p[i] = p;
        relaxed[i] = relaxed_diagonal(p, in.alpha);
        known[i] = 0.0;
    }
}

// What the unknowns beside a domain edge read: the coefficient toward it
// (`coefficient`, the faces on the edge's side), and the edge's condition.
struct edge_run {
    const double* coefficient;
    const mask* known;
    const double* value;
    const mask* unknown;
    double alpha;
};

// The domain edge normal to direction T beside N unknowns. A Dirichlet edge's
// value is a known neighbour and moves to the source; a zero-gradient edge has
// the unknown itself as its neighbour, so that coefficient leaves the diagonal.
// Either way no neighbour is there.
template <int T, std::size_t N>
inline void edge_unknowns(lanes::cells<N>, const edge_run& in, std::size_t first,
                          double* CFDCORE_RESTRICT neighbour, double* CFDCORE_RESTRICT a_p,
                          double* CFDCORE_RESTRICT relaxed, double* CFDCORE_RESTRICT known) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const mask is_known = on_edge<T>(in.known, i);
        const double a = in.coefficient[i];
        known[i] = known[i] + lanes::keep(is_known, a * on_edge<T>(in.value, i));
        const double p = lanes::keep(in.unknown[i], a_p[i] - lanes::keep(~is_known, a));
        a_p[i] = p;
        relaxed[i] = relaxed_diagonal(p, in.alpha);
        neighbour[i] = 0.0;
    }
}

// What the sources of a row of unknowns read of the faces transverse along one
// direction, the last four with a viscosity field.
struct cross_source {
    const double* deferred;
    const double* deferred_above;
    const double* mu;
    const double* mu_above;
    const double* stress;
    const double* stress_above;
};

// What the sources of a row of unknowns read, beside their diagonal.
template <std::size_t Slots>
struct source_run {
    const double* cell_deferred;
    const double* cell_deferred_below;
    cross_source cross[Slots];
    const double* p;
    const double* p_below;
    const double* width_t[Slots];
    const double* own;
    const double* a_p;
    // With a viscosity field:
    const double* mu;
    const double* mu_below;
    const double* mean_high;
    const double* cell_stress;
    const double* cell_stress_below;
    double relaxation;
};

// One transverse direction's part of form b's stress source, added to `stress`.
template <int T>
CFDCORE_INLINE double with_cross_stress(const cross_source& in, double stress, double mu_f,
                                        std::size_t i) {
    return stress +
           (high_side<T>(in.mu, in.mu_above, i) - mu_f) *
               high_side<T>(in.stress, in.stress_above, i) -
           (in.mu[i] - mu_f) * in.stress[i];
}

// The source of N unknowns: the known neighbours' (already in `b`), the
// deferred correction with form b's stress source, the pressure's and the
// relaxation term's, added in the reference's order. Read at unknowns only.
template <int C, bool Field, int... T, std::size_t N>
inline void sources(lanes::cells<N>, directions<T...> across,
                    const source_run<slots<T...>>& in, std::size_t first,
                    double* CFDCORE_RESTRICT b) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        double net = in.cell_deferred[i] -
                     low_side<C>(in.cell_deferred_below, in.cell_deferred, i);
        ((net = net +
                high_side<T>(in.cross[slot_of(C, T)].deferred,
                             in.cross[slot_of(C, T)].deferred_above, i) -
                in.cross[slot_of(C, T)].deferred[i]),
         ...);
        double deferred = -net;
        if constexpr (Field) {
            const double mu_high = in.mu[i];
            const double mu_low = low_side<C>(in.mu_below, in.mu, i);
            const double mu_f = mu_low + (mu_high - mu_low) * along<C>(in.mean_high, i);
            double stress = (mu_high - mu_f) * in.cell_stress[i] -
                            (mu_low - mu_f) *
                                low_side<C>(in.cell_stress_below, in.cell_stress, i);
            ((stress = with_cross_stress<T>(in.cross[slot_of(C, T)], stress, mu_f, i)), ...);
            deferred = deferred + stress;
        }
        const double pressure = times_area<C>(
            across, -(in.p[i] - low_side<C>(in.p_below, in.p, i)), in.width_t, i);
        b[i] = b[i] + deferred + pressure + in.relaxation * in.a_p[i] * in.own[i];
    }
}

// ---------------------------------------------------------------------------
// The sweeps

// What a Jacobi sweep of a row of unknowns reads: the iterate in the row and,
// by direction, in the rows below and above it, a row of zeros standing for one
// beyond the domain; entry 0 for the end cells of a row alone, whose missing
// neighbour is a zero (the reference's padding). The coefficients are by part:
// the streamwise one's and then each slot's.
template <int D>
struct sweep_run {
    const double* x;
    const double* below[D];
    const double* above[D];
    const double* plus[D];
    const double* minus[D];
    const double* b;
    const double* relaxed;
    const double* a_p;
    const mask* held;
    const double* own;
};

// The iterate's value at the neighbour of unknown i on the low or the high side
// along direction d. `Inner` reads a neighbour along the row beside the unknown.
template <int d, bool Inner, int D>
CFDCORE_INLINE double low_neighbour(const sweep_run<D>& in, std::size_t i) {
    return along_row<d> && Inner ? in.x[i - 1] : in.below[d][i];
}
template <int d, bool Inner, int D>
CFDCORE_INLINE double high_neighbour(const sweep_run<D>& in, std::size_t i) {
    return along_row<d> && Inner ? in.x[i + 1] : in.above[d][i];
}

// One part's two products of an unknown's neighbour sum, added to `sum`.
template <int Part, int d, bool Inner, int D>
CFDCORE_INLINE double with_neighbours(const sweep_run<D>& in, double sum, std::size_t i) {
    return sum + in.plus[Part][i] * high_neighbour<d, Inner>(in, i) +
           in.minus[Part][i] * low_neighbour<d, Inner>(in, i);
}

// One Jacobi sweep of N unknowns of a row; a face that is not an unknown is
// zero, or with `Held` keeps a vent face's input value.
template <int C, bool Inner, bool Held, int D, int... T, std::size_t N>
inline void sweep_cells(lanes::cells<N>, directions<T...>, const sweep_run<D>& in,
                        std::size_t first, double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        double numerator = in.plus[0][i] * high_neighbour<C, Inner>(in, i) +
                           in.minus[0][i] * low_neighbour<C, Inner>(in, i);
        ((numerator = with_neighbours<1 + slot_of(C, T), T, Inner>(in, numerator, i)), ...);
        numerator = numerator + in.b[i];
        const double next = numerator / in.relaxed[i];
        const mask unknown = lanes::mask_of(in.a_p[i] > 0.0);
        if constexpr (Held) {
            out[i] = lanes::select(unknown, next, lanes::keep(in.held[i], in.own[i]));
        } else {
            out[i] = lanes::keep(unknown, next);
        }
    }
}

// The run's operands from element m of the row on, for an end cell.
template <int D>
sweep_run<D> sweep_at(const sweep_run<D>& row, std::size_t m) {
    sweep_run<D> at = row;
    at.x = row.x + m;
    for (int d = 0; d < D; ++d) {
        at.below[d] = d == 0 ? nullptr : row.below[d] + m;
        at.above[d] = d == 0 ? nullptr : row.above[d] + m;
        at.plus[d] = row.plus[d] + m;
        at.minus[d] = row.minus[d] + m;
    }
    at.b = row.b + m;
    at.relaxed = row.relaxed + m;
    at.a_p = row.a_p + m;
    at.held = row.held == nullptr ? nullptr : row.held + m;
    at.own = row.own + m;
    return at;
}

template <int C, bool Held, int D>
void sweep_row(const sweep_run<D> row, const double* zeros, std::size_t first, std::size_t last,
               double* out) {
    constexpr transverse_directions<D, C> across{};
    if constexpr (along_row<C>) {
        // The ends of a row along s are domain-edge faces, which no sweep
        // writes, so every unknown has both neighbours in its row.
        lanes::for_each_block(first, last, [&](auto cells, std::size_t m) {
            sweep_cells<C, true, Held>(cells, across, row, m, out);
        });
    } else {
        sweep_run<D> end = sweep_at(row, first);
        end.below[0] = zeros;
        end.above[0] = row.x + first + 1;
        sweep_cells<C, false, Held>(lanes::cells<1>{}, across, end, 0, out + first);
        lanes::for_each_block(first + 1, last - 1, [&](auto cells, std::size_t m) {
            sweep_cells<C, true, Held>(cells, across, row, m, out);
        });
        end = sweep_at(row, last - 1);
        end.below[0] = row.x + last - 2;
        end.above[0] = zeros;
        sweep_cells<C, false, Held>(lanes::cells<1>{}, across, end, 0, out + last - 1);
    }
}

// The row walkers, which take their operands by value: a copy nothing else can
// reach, so the compiler holds its pointers in registers.
template <int C, bool Inner>
void stream_row(const stream_run run, bool obstacles, std::size_t first, std::size_t last,
                double* out) {
    if (obstacles) {
        lanes::for_each_block(first, last, [&](auto cells, std::size_t m) {
            cell_deferred<C, Inner, true>(cells, run, m, out);
        });
    } else {
        lanes::for_each_block(first, last, [&](auto cells, std::size_t m) {
            cell_deferred<C, Inner, false>(cells, run, m, out);
        });
    }
}

template <int T, bool Inner>
void cross_row(const cross_run run, bool obstacles, std::size_t first, std::size_t last,
               double* out) {
    if (obstacles) {
        lanes::for_each_block(first, last, [&](auto cells, std::size_t m) {
            corner_deferred<T, Inner, true>(cells, run, m, out);
        });
    } else {
        lanes::for_each_block(first, last, [&](auto cells, std::size_t m) {
            corner_deferred<T, Inner, false>(cells, run, m, out);
        });
    }
}

template <int C, int... T>
void cell_flux_row(directions<T...> across, const cell_flux_run<slots<T...>> run,
                   std::size_t count, double* flux, double* plus, double* minus) {
    lanes::for_each_block(0, count, [&](auto cells, std::size_t m) {
        cell_flux<C>(cells, across, run, m, flux, plus, minus);
    });
}

template <int C>
void corner_flux_row(const corner_flux_run run, std::size_t count, double* flux, double* plus,
                     double* minus) {
    lanes::for_each_block(0, count, [&](auto cells, std::size_t m) {
        corner_flux<C>(cells, run, m, flux, plus, minus);
    });
}

template <int C>
void stream_neighbour_row(const stream_faces run, const mask* unknown, std::size_t first,
                          std::size_t last, double* plus, double* minus) {
    lanes::for_each_block(first, last, [&](auto cells, std::size_t m) {
        stream_neighbours<C>(cells, run, unknown, m, plus, minus);
    });
}

template <int T>
void cross_neighbour_row(const cross_faces run, const mask* unknown, const mask* couples_high,
                         const mask* couples_low, std::size_t first, std::size_t last,
                         double* plus, double* minus) {
    lanes::for_each_block(first, last, [&](auto cells, std::size_t m) {
        cross_neighbours<T>(cells, run, unknown, couples_high, couples_low, m, plus, minus);
    });
}

template <int C, int... T>
void diagonal_row(directions<T...> across, const diagonal_run<slots<T...>> run,
                  std::size_t first, std::size_t last, double* a_p, double* relaxed,
                  double* known) {
    lanes::for_each_block(first, last, [&](auto cells, std::size_t m) {
        diagonals<C>(cells, across, run, m, a_p, relaxed, known);
    });
}

template <int T>
void edge_unknown_row(const edge_run run, std::size_t first, std::size_t last, double* neighbour,
                      double* a_p, double* relaxed, double* known) {
    lanes::for_each_block(first, last, [&](auto cells, std::size_t m) {
        edge_unknowns<T>(cells, run, m, neighbour, a_p, relaxed, known);
    });
}

template <int C, bool Field, int... T>
void source_row(directions<T...> across, const source_run<slots<T...>> run, std::size_t first,
                std::size_t last, double* b) {
    lanes::for_each_block(first, last, [&](auto cells, std::size_t m) {
        sources<C, Field>(cells, across, run, m, b);
    });
}

template <int C, bool Field, bool Wall>
void corner_conductance_row(const corner_conductance_run run, std::size_t count, double* out) {
    lanes::for_each_block(0, count, [&](auto cells, std::size_t m) {
        corner_conductance<C, Field, Wall>(cells, run, m, out);
    });
}

// ---------------------------------------------------------------------------
// The checks of a viscosity

struct verdicts {
    mask not_finite = lanes::none;
    mask not_positive = lanes::none;
};

// The refusal masks of N viscosities, read where `where` is set or clear.
template <std::size_t N>
inline verdicts verdict_cells(lanes::cells<N>, const double* mu, const mask* where, mask flip,
                              std::size_t first) {
    verdicts v;
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const mask read = where[i] ^ flip;
        v.not_finite |=
            read & lanes::mask_of(!(std::fabs(mu[i]) <= std::numeric_limits<double>::max()));
        v.not_positive |= read & lanes::mask_of(mu[i] <= 0.0);
    }
    return v;
}

// flip: `lanes::none` to read where `where` is set, `lanes::all` where it is clear.
viscosity_status verdict_of(std::span<const double> mu, std::span<const mask> where, mask flip) {
    verdicts total;
    lanes::for_each_block(0, mu.size(), [&](auto cells, std::size_t m) {
        const verdicts v = verdict_cells(cells, mu.data(), where.data(), flip, m);
        total.not_finite |= v.not_finite;
        total.not_positive |= v.not_positive;
    });
    if (total.not_finite != lanes::none) {
        return viscosity_status::not_finite;
    }
    return total.not_positive != lanes::none ? viscosity_status::not_positive
                                             : viscosity_status::ok;
}

// ---------------------------------------------------------------------------
// Positions in a layout

// A position in a layout, one entry per direction. The cast keeps D from being
// deduced from it, where an array's size is not an int.
template <int D>
using position_in = std::array<std::size_t, static_cast<std::size_t>(D)>;

// Calls body(at) for every row whose position lies in [first, last) along each
// direction that crosses the rows, direction 1 the fastest. No range is empty.
template <std::size_t D, typename Body>
void for_each_row(const std::array<std::size_t, D>& first, const std::array<std::size_t, D>& last,
                  Body&& body) {
    std::array<std::size_t, D> at = first;
    at[0] = 0;
    for (;;) {
        body(at);
        std::size_t d = 1;
        while (d < D && ++at[d] == last[d]) {
            at[d] = first[d];
            ++d;
        }
        if (d == D) {
            return;
        }
    }
}

// Every cell of a layout, its position along each direction handed to body.
template <int D, typename Body>
void for_each_cell(const layout::layout<D>& l, Body&& body) {
    position_in<D> last{};
    for (int d = 0; d < D; ++d) {
        last[d] = l.cells(d);
    }
    for_each_row(position_in<D>{}, last, [&](position_in<D> at) {
        for (at[0] = 0; at[0] < last[0]; ++at[0]) {
            body(std::as_const(at));
        }
    });
}

// The index of a position in a layout.
template <int D>
std::size_t index_of(const layout::layout<D>& l, const position_in<D>& at) {
    std::size_t index = 0;
    for (int d = 0; d < D; ++d) {
        index += at[d] * l.cell_stride(d);
    }
    return index;
}

// The index of a position on the edge normal to direction t of a component
// whose faces are `faces`: the layout without t.
template <int D>
std::size_t edge_index(const layout::layout<D>& faces, int t, const position_in<D>& at) {
    std::size_t index = 0;
    std::size_t stride = 1;
    for (int d = 0; d < D; ++d) {
        if (d != t) {
            index += at[d] * stride;
            stride *= faces.cells(d);
        }
    }
    return index;
}

// `at` one further, or one back, along direction d.
template <std::size_t D>
std::array<std::size_t, D> after(std::array<std::size_t, D> at, int d) {
    ++at[d];
    return at;
}
template <std::size_t D>
std::array<std::size_t, D> before(std::array<std::size_t, D> at, int d) {
    --at[d];
    return at;
}

// A grid's cells, with one more along each of some directions.
template <int D>
layout::layout<D> extended(const grid::grid<D>& g, std::initializer_list<int> along) {
    std::array<std::size_t, D> cells{};
    for (int d = 0; d < D; ++d) {
        cells[d] = g.cells(d);
    }
    for (const int d : along) {
        ++cells[d];
    }
    return layout::layout<D>(cells);
}

// ---------------------------------------------------------------------------
// The set-up

std::vector<double> as_vector(std::span<const double> a) { return {a.begin(), a.end()}; }

void copy_weights(std::array<std::vector<double>, 3>& into, std::size_t at,
                  const std::array<double, 3>& weights) {
    for (std::size_t m = 0; m < 3; ++m) {
        into[m][at] = weights[m];
    }
}

// The face positions along a direction: 0 to n, the domain's ends among them.
template <int D>
std::vector<double> face_positions(const grid::grid<D>& g, int d) {
    const auto nodes = g.nodes(d);
    const auto interior = g.interior_faces(d);
    std::vector<double> all{nodes.front()};
    all.insert(all.end(), interior.begin(), interior.end());
    all.push_back(nodes.back());
    return all;
}

// Whether a face of component c, not on the domain's edge, is a SOLID cell's.
template <int D>
bool solid_face(const grid::grid<D>& g, int c, const position_in<D>& face) {
    return g.solid(index_of(g.fields(), before(face, c))) || g.solid(index_of(g.fields(), face));
}

// The quantities of the streamwise direction, whose pairs are not set yet.
template <int D>
component_setup<D> make_component(const grid::grid<D>& g, const predictor_input<D>& in, int c) {
    component_setup<D> o{.faces = extended(g, {c}), .ns = g.cells(c)};
    const std::size_t ns = o.ns;
    const std::size_t cells = g.fields().cell_count();
    const std::size_t faces = o.faces.cell_count();
    o.width_s = as_vector(in.cell_widths[c]);
    o.face_width = as_vector(in.face_widths[c]);
    const std::vector<double> s_faces = face_positions(g, c);
    const auto s_centers = g.nodes(c).subspan(1, ns);

    for (auto* w : {&o.std_pos, &o.std_neg, &o.alt_pos, &o.alt_neg}) {
        for (auto& v : *w) {
            v.assign(ns, 0.0);
        }
    }
    // quick_face_values on the nodes s_faces, left = 0 to ns - 1, and
    // _obstacle_streamwise's replacement nodes.
    const std::size_t last = ns;
    for (std::size_t k = 0; k < ns; ++k) {
        const std::size_t far_pos = k >= 1 ? k - 1 : k + 2;
        const std::size_t far_neg = k + 2 <= last ? k + 2 : k - 1;
        const std::size_t alt_pos = k + 2 <= last ? k + 2 : k - 1;
        const std::size_t alt_neg = k >= 1 ? k - 1 : k + 2;
        const double lo = s_faces[k];
        const double hi = s_faces[k + 1];
        const double f = s_centers[k];
        copy_weights(o.std_pos, k, scalar::lagrange_weights(s_faces[far_pos], lo, hi, f));
        copy_weights(o.std_neg, k, scalar::lagrange_weights(s_faces[far_neg], hi, lo, f));
        copy_weights(o.alt_pos, k, scalar::lagrange_weights(s_faces[alt_pos], lo, hi, f));
        copy_weights(o.alt_neg, k, scalar::lagrange_weights(s_faces[alt_neg], hi, lo, f));
    }
    o.mean_high.assign(ns + 1, 0.0);
    for (std::size_t s = 1; s < ns; ++s) {
        const double w_l = s_faces[s] - s_centers[s - 1];
        const double w_r = s_centers[s] - s_faces[s];
        o.mean_high[s] = w_r / (w_l + w_r);
    }

    o.unknown.assign(faces, lanes::none);
    for_each_cell(o.faces, [&](const auto& at) {
        if (at[c] > 0 && at[c] < ns && !solid_face(g, c, at)) {
            o.unknown[index_of(o.faces, at)] = lanes::all;
        }
    });
    o.blocked_pos.assign(cells, lanes::none);
    o.blocked_neg.assign(cells, lanes::none);
    o.conductance_s.assign(cells, 0.0);
    for_each_cell(g.fields(), [&](const auto& at) {
        const std::size_t i = index_of(g.fields(), at);
        const std::size_t s = at[c];
        o.blocked_pos[i] =
            lanes::mask_of(s > 0 && g.solid(index_of(g.fields(), before(at, c))));
        o.blocked_neg[i] =
            lanes::mask_of(s + 1 < ns && g.solid(index_of(g.fields(), after(at, c))));
        double conductance = in.mu;
        for (int q = 0; q < D - 1; ++q) {
            const int t = transverse(c, q);
            conductance = conductance * in.cell_widths[t][at[t]];
        }
        o.conductance_s[i] = conductance / o.width_s[s];
    });

    if (!in.held[c].empty()) {
        o.held.resize(faces);
        for (std::size_t i = 0; i < faces; ++i) {
            o.held[i] = lanes::mask_of(in.held[c][i]);
        }
    }
    const std::size_t nx = g.cells(0);
    o.row_blocked.assign(cells / nx, 0);
    for (std::size_t i = 0; i < cells; ++i) {
        if ((o.blocked_pos[i] | o.blocked_neg[i]) != lanes::none) {
            o.row_blocked[i / nx] = 1;
        }
    }
    return o;
}

// The quantities of the pair of component c, whose set-up is o, and direction t.
template <int D>
pair_setup<D> make_pair_setup(const grid::grid<D>& g, const predictor_input<D>& in, int c, int t,
                              const component_setup<D>& o) {
    pair_setup<D> w{.corners = extended(g, {c, t}), .nt = g.cells(t)};
    const std::size_t ns = o.ns;
    const std::size_t nt = w.nt;
    const std::size_t corners = w.corners.cell_count();
    const std::size_t faces = o.faces.cell_count();
    w.width_t = as_vector(in.cell_widths[t]);
    const std::vector<double> t_faces = face_positions(g, t);
    const auto t_centers = g.nodes(t).subspan(1, nt);
    const auto t_nodes = g.nodes(t);

    for (auto* a : {&w.t_pos, &w.t_neg}) {
        for (auto& v : *a) {
            v.assign(nt + 1, 0.0);
        }
    }
    for (auto* a : {&w.wall_pos, &w.wall_neg}) {
        for (auto& v : *a) {
            v.assign(nt + 1, 0.0);
        }
    }
    w.harmonic_low.assign(nt + 1, 0.0);
    w.harmonic_high.assign(nt + 1, 0.0);
    for (std::size_t r = 1; r < nt; ++r) {
        const double f = t_faces[r];
        copy_weights(w.t_pos, r,
                     scalar::lagrange_weights(t_nodes[r - 1], t_nodes[r], t_nodes[r + 1], f));
        copy_weights(w.t_neg, r,
                     scalar::lagrange_weights(t_nodes[r + 2], t_nodes[r + 1], t_nodes[r], f));
        const auto pos =
            scalar::lagrange_weights(t_faces[r - 1], t_centers[r - 1], t_centers[r], f);
        const auto neg =
            scalar::lagrange_weights(t_faces[r + 1], t_centers[r], t_centers[r - 1], f);
        for (std::size_t m = 0; m < 2; ++m) {
            w.wall_pos[m][r] = pos[m + 1];
            w.wall_neg[m][r] = neg[m + 1];
        }
        const double d_lo = f - t_centers[r - 1];
        const double d_hi = t_centers[r] - f;
        w.harmonic_low[r] = d_lo / (d_lo + d_hi);
        w.harmonic_high[r] = 1.0 - w.harmonic_low[r];
    }

    // The obstacle walls (`_obstacle_walls`). A corner has the position of the
    // face on its high side along t.
    w.couples_high.assign(faces, lanes::all);
    w.couples_low.assign(faces, lanes::all);
    w.wall_face.assign(corners, lanes::none);
    w.far_wall_low.assign(corners, lanes::none);
    w.far_wall_high.assign(corners, lanes::none);
    std::vector<double> wall_distance(corners, 1.0);
    for_each_cell(o.faces, [&](const auto& at) {
        const std::size_t face = index_of(o.faces, at);
        if (o.unknown[face] == lanes::none) {
            return;
        }
        const std::size_t r = at[t];
        const std::size_t low = index_of(w.corners, at);
        const std::size_t high = index_of(w.corners, after(at, t));
        if (r + 1 < nt && solid_face(g, c, after(at, t))) {
            // The unknown's neighbour on the high side is an obstacle's face:
            // it is the far node of a flow toward decreasing t through the
            // unknown's low face.
            w.couples_high[face] = lanes::none;
            w.wall_face[high] = lanes::all;
            wall_distance[high] = t_faces[r + 1] - t_centers[r];
            if (r > 0) {
                w.far_wall_high[low] = lanes::all;
            }
        }
        if (r > 0 && solid_face(g, c, before(at, t))) {
            w.couples_low[face] = lanes::none;
            w.wall_face[low] = lanes::all;
            wall_distance[low] = t_centers[r] - t_faces[r];
            if (r + 1 < nt) {
                w.far_wall_low[high] = lanes::all;
            }
        }
    });

    // The edges' conditions.
    const edge_condition& low = in.low[c][slot_of(c, t)];
    const edge_condition& high = in.high[c][slot_of(c, t)];
    const std::size_t edge = low.value.size();
    w.low_known.resize(edge);
    w.high_known.resize(edge);
    for (std::size_t i = 0; i < edge; ++i) {
        w.low_known[i] = lanes::mask_of(low.is_dirichlet[i]);
        w.high_known[i] = lanes::mask_of(high.is_dirichlet[i]);
    }
    w.low_value = as_vector(low.value);
    w.high_value = as_vector(high.value);

    // What each transverse face's conductance divides by, and whose
    // viscosity it takes; then the scalar path's conductances.
    w.reach.assign(corners, 1.0);
    w.conducts.assign(corners, lanes::all);
    w.takes_wall.assign(corners, lanes::none);
    w.wall_read.assign(corners, lanes::none);
    w.conductance_t.assign(corners, 0.0);
    for_each_cell(w.corners, [&](const auto& at) {
        const std::size_t i = index_of(w.corners, at);
        const std::size_t r = at[t];
        const std::size_t s = at[c];
        w.reach[i] = in.face_widths[t][r];
        if (r == 0 || r == nt) {
            const std::size_t e = edge_index(o.faces, t, at);
            const mask known = r == 0 ? w.low_known[e] : w.high_known[e];
            w.reach[i] = r == 0 ? low.wall_distance : high.wall_distance;
            w.conducts[i] = known;
            w.takes_wall[i] = known;
            if (s > 0 && s < ns) {
                const auto face = r == 0 ? at : before(at, t);
                w.wall_read[i] = known & o.unknown[index_of(o.faces, face)];
            }
        }
        if (w.wall_face[i] != lanes::none) {
            w.reach[i] = wall_distance[i];
            w.takes_wall[i] = lanes::all;
            w.wall_read[i] = lanes::all;
        }
        w.conductance_t[i] = lanes::keep(w.conducts[i], in.mu * o.face_width[s] / w.reach[i]);
    });

    const std::size_t row = w.corners.cells(0);
    w.row_walls.assign(corners / row, 0);
    for (std::size_t i = 0; i < corners; ++i) {
        if ((w.wall_face[i] | w.far_wall_low[i] | w.far_wall_high[i]) != lanes::none) {
            w.row_walls[i / row] = 1;
        }
    }
    return w;
}

}  // namespace

template <int D>
    requires(D >= 1)
std::string_view predictor<D>::check(const grid::grid<D>& g, const predictor_input<D>& in) {
    for (int d = 0; d < D; ++d) {
        if (g.cells(d) < 2) {
            return "the momentum predictor needs at least 2 cells per axis";
        }
    }
    for (int d = 0; d < D; ++d) {
        if (in.cell_widths[d].size() != g.cells(d)) {
            return "a direction's cell widths are not one per cell";
        }
        if (in.face_widths[d].size() != g.cells(d) + 1) {
            return "a direction's face widths are not one per face";
        }
        for (int q = 0; q < D - 1; ++q) {
            // The component's faces on an edge normal to the pair's t.
            const std::size_t t = g.cells(transverse(d, q));
            const std::size_t faces = g.fields().face_count(d) / t;
            for (const edge_condition* e : {&in.low[d][q], &in.high[d][q]}) {
                if (e->is_dirichlet.size() != faces || e->value.size() != faces) {
                    return "an edge condition does not hold one entry per face along the edge";
                }
            }
        }
        if (!in.held[d].empty() && in.held[d].size() != g.fields().face_count(d)) {
            return "the held faces are not one flag per face";
        }
    }
    if (in.sweeps < 1) {
        return "sweeps is below 1";
    }
    return {};
}

template <int D>
    requires(D >= 1)
predictor<D>::predictor(const grid::grid<D>& g, const predictor_input<D>& in)
    : grid_(&g),
      nx_(g.cells(0)),
      mu_(in.mu),
      alpha_(in.alpha),
      half_rho_(in.rho * 0.5),
      relaxation_((1.0 - in.alpha) / in.alpha),
      sweeps_(in.sweeps) {
    const std::string_view reason = check(g, in);
    if (!reason.empty()) {
        throw std::invalid_argument(std::string(reason));
    }
    std::size_t faces = 0;
    for (int c = 0; c < D; ++c) {
        setup_.push_back(make_component(g, in, c));
        for (int q = 0; q < D - 1; ++q) {
            setup_[c].pairs.push_back(make_pair_setup(g, in, c, transverse(c, q), setup_[c]));
        }
        faces = std::max(faces, g.fields().face_count(c));
    }
    for (auto& a : coefficient_) {
        a.assign(faces, 0.0);
    }
    for (auto* a : {&source_, &relaxed_, &iterate_}) {
        a->assign(faces, 0.0);
    }
    // Two rows along the last direction, for each position along the ones
    // between it and the rows.
    std::size_t kept = D == 1 ? 1 : 2;
    for (int d = 1; d < D - 1; ++d) {
        kept *= g.cells(d) + 1;
    }
    const auto keep_rows = [&](std::vector<flux_row>& rows) {
        rows.resize(kept);
        for (flux_row& row : rows) {
            for (auto* a : {&row.conductance, &row.flux, &row.plus, &row.minus, &row.deferred,
                            &row.stress, &row.viscosity}) {
                a->assign(nx_ + 1, 0.0);
            }
        }
    };
    keep_rows(cell_rows_);
    for (int q = 0; q < D - 1; ++q) {
        keep_rows(corner_rows_[q]);
        means_[q].assign(kept, std::vector<double>(nx_ + 1, 0.0));
    }
    edge_.assign(2 * (nx_ + 1), 0.0);
    zeros_.assign(nx_ + 1, 0.0);
}

template <int D>
    requires(D >= 1)
viscosity_status predictor<D>::check_viscosity(std::span<const double> mu_eff) const {
    return verdict_of(mu_eff, grid_->solid_flags(), lanes::all);
}

template <int D>
    requires(D >= 1)
viscosity_status predictor<D>::check_wall_viscosity(int c, int q,
                                                    std::span<const double> wall_mu) const {
    return verdict_of(wall_mu, setup_[c].pairs[q].wall_read, lanes::none);
}

template <int D>
    requires(D >= 1)
std::size_t predictor<D>::kept_row(int d, const position& at) const {
    std::size_t row = 0;
    std::size_t stride = 1;
    for (int e = 1; e < d; ++e) {
        row += at[e] * stride;
        stride *= grid_->cells(e) + 1;
    }
    return row + (at[d] % 2) * stride;
}

// The harmonic means across the pair's transverse faces of a row of cells at
// `at`, whose entry along T is the faces' and whose others are the cells'; at a
// domain edge the cells' own values. Formed, or with `form` false the row an
// earlier call formed.
template <int D>
    requires(D >= 1)
template <int C, int T>
const double* predictor<D>::harmonic_row(const position& at, const double* mu, bool form) {
    const pair_setup<D>& w = setup_[C].pairs[slot_of(C, T)];
    const mask* solid = grid_->solid_flags().data();
    const layout::layout<D>& cells_of = grid_->fields();
    const std::size_t nx = nx_;
    if constexpr (along_row<T>) {
        double* mean = means_[slot_of(C, T)][kept_row(C, at)].data();
        if (form) {
            const std::size_t start = index_of(cells_of, at);
            const double* cell = mu + start;
            const mask* flags = solid + start;
            mean[0] = cell[0];
            // Face i of the row lies between cells i - 1 and i.
            lanes::for_each_block(0, nx - 1, [&](auto cells, std::size_t m) {
                harmonic_cells<T>(cells, cell, cell + 1, flags, flags + 1,
                                  w.harmonic_low.data() + 1, w.harmonic_high.data() + 1, m,
                                  mean + 1);
            });
            mean[nx] = cell[nx - 1];
        }
        return mean;
    } else {
        const std::size_t r = at[T];
        const std::size_t high = index_of(cells_of, at);
        if (r == 0) {
            return mu + high;
        }
        const std::size_t low = high - cells_of.cell_stride(T);
        if (r == w.nt) {
            return mu + low;
        }
        double* mean = means_[slot_of(C, T)][along_row<C> ? 0 : kept_row(C, at)].data();
        if (form) {
            lanes::for_each_block(0, nx, [&](auto cells, std::size_t m) {
                harmonic_cells<T>(cells, mu + low, mu + high, solid + low, solid + high,
                                  w.harmonic_low.data() + r, w.harmonic_high.data() + r, m, mean);
            });
        }
        return mean;
    }
}

// The viscosity at the row of the pair's corners at `at` (`_corner_viscosity`),
// at the positions interior along s. Where s crosses the rows, they are asked
// for in ascending order along it from position 1: each call keeps the harmonic
// means of its row of cells for the next.
template <int D>
    requires(D >= 1)
template <int C, int T>
void predictor<D>::corner_viscosity(const position& at, const double* mu, double* out) {
    const component_setup<D>& o = setup_[C];
    if constexpr (along_row<C>) {
        const double* row = harmonic_row<C, T>(at, mu, true);
        lanes::for_each_block(1, nx_, [&](auto cells, std::size_t m) {
            mean_cells<C>(cells, row, row, o.mean_high.data(), m, out);
        });
    } else {
        const std::size_t s = at[C];
        const double* below = harmonic_row<C, T>(before(at, C), mu, s == 1);
        const double* row = harmonic_row<C, T>(at, mu, true);
        const std::size_t count = o.pairs[slot_of(C, T)].corners.cells(0);
        lanes::for_each_block(0, count, [&](auto cells, std::size_t m) {
            mean_cells<C>(cells, below, row, o.mean_high.data() + s, m, out);
        });
    }
}

// The fluxes through the row of streamwise faces at `at`, one per cell.
template <int D>
    requires(D >= 1)
template <int C>
void predictor<D>::cell_row(const position& at, const double* own, const double* mu,
                            flux_row& out) {
    constexpr transverse_directions<D, C> across{};
    const component_setup<D>& o = setup_[C];
    const std::size_t nx = nx_;
    const std::size_t stride = o.faces.cell_stride(C);
    const std::size_t cell = index_of(grid_->fields(), at);
    const double* low = own + index_of(o.faces, at);
    const double* high = low + stride;
    const double* width_s = o.width_s.data() + at[C];
    const auto with_widths = [&](auto run) {
        for_each_transverse<C>(across, [&]<int T> {
            constexpr int q = slot_of(C, T);
            run.width_t[q] = o.pairs[q].width_t.data() + at[T];
        });
        return run;
    };
    const double* conductance = o.conductance_s.data() + cell;
    if (mu != nullptr) {
        constexpr std::size_t count = D == 1 ? 1 : D - 1;
        cell_conductance_run<count> diffusion{};
        diffusion.mu = mu + cell;
        diffusion.width_s = width_s;
        diffusion = with_widths(diffusion);
        double* to = out.conductance.data();
        lanes::for_each_block(0, nx, [&](auto cells, std::size_t m) {
            cell_conductance<C>(cells, across, diffusion, m, to);
        });
        conductance = to;
        cell_stress_run<count> stress{};
        stress.low = low;
        stress.high = high;
        stress.width_s = width_s;
        stress = with_widths(stress);
        double* stresses = out.stress.data();
        lanes::for_each_block(0, nx, [&](auto cells, std::size_t m) {
            cell_stress<C>(cells, across, stress, m, stresses);
        });
    }
    cell_flux_run<D == 1 ? 1 : D - 1> flux{};
    flux.low = low;
    flux.high = high;
    flux.conductance = conductance;
    flux.half_rho = half_rho_;
    cell_flux_row<C>(across, with_widths(flux), nx, out.flux.data(), out.plus.data(),
                     out.minus.data());

    // The deferred correction: the run's operands from cell `first` on.
    const bool obstacles = o.row_blocked[cell / nx] != 0;
    const auto run_at = [&](std::size_t first) {
        stream_run run{};
        // The faces' position along s, whose ends have no node on one side.
        const std::size_t s = along_row<C> ? first : at[C];
        run.low = low + first;
        run.high = high + first;
        run.far_high = s + 1 < o.ns ? high + first + stride : low + first - stride;
        run.far_low = s > 0 ? low + first - stride : run.far_high;
        for (std::size_t m = 0; m < 3; ++m) {
            run.std_pos[m] = o.std_pos[m].data() + s;
            run.std_neg[m] = o.std_neg[m].data() + s;
            run.alt_pos[m] = o.alt_pos[m].data() + s;
            run.alt_neg[m] = o.alt_neg[m].data() + s;
        }
        run.blocked_pos = o.blocked_pos.data() + cell + first;
        run.blocked_neg = o.blocked_neg.data() + cell + first;
        run.flux = out.flux.data() + first;
        return run;
    };
    double* deferred = out.deferred.data();
    if constexpr (along_row<C>) {
        stream_row<C, false>(run_at(0), obstacles, 0, 1, deferred);
        stream_row<C, true>(run_at(1), obstacles, 0, nx - 2, deferred + 1);
        stream_row<C, false>(run_at(nx - 1), obstacles, 0, 1, deferred + nx - 1);
    } else {
        stream_row<C, false>(run_at(0), obstacles, 0, nx, deferred);
    }
}

// The fluxes through the row of the pair's transverse faces at `at`, at the
// positions interior along s.
template <int D>
    requires(D >= 1)
template <int C, int T>
void predictor<D>::corner_row(const position& at, const double* own, const double* other,
                              const double* mu, const double* wall, flux_row& out) {
    const component_setup<D>& o = setup_[C];
    const pair_setup<D>& w = o.pairs[slot_of(C, T)];
    const bool field = mu != nullptr;
    const std::size_t nx = nx_;
    const std::size_t corner = index_of(w.corners, at);
    // The first position of the row and the count of those written: 1 to
    // nx - 1 of a row along s, every one of a row s crosses.
    const std::size_t first = along_row<C> ? 1 : 0;
    const std::size_t count = along_row<C> ? nx - 1 : w.corners.cells(0);
    // The transverse velocity on either half of the faces, at the positions
    // s - 1 and s of its own face layout.
    const layout::layout<D>& other_faces = setup_[T].faces;
    const double* other_high = other + index_of(other_faces, at) + first;
    const double* other_low = other_high - other_faces.cell_stride(C);
    const std::size_t s = along_row<C> ? 1 : at[C];
    const double* face_width = o.face_width.data() + s;

    const double* conductance = w.conductance_t.data() + corner + first;
    if (field) {
        corner_viscosity<C, T>(at, mu, out.viscosity.data());
    }
    if (field || wall != nullptr) {
        const corner_conductance_run run{out.viscosity.data() + first,
                                         wall == nullptr ? nullptr : wall + corner + first,
                                         face_width,
                                         w.reach.data() + corner + first,
                                         w.conductance_t.data() + corner + first,
                                         w.takes_wall.data() + corner + first,
                                         w.conducts.data() + corner + first};
        double* to = out.conductance.data() + first;
        if (!field) {
            corner_conductance_row<C, false, true>(run, count, to);
        } else if (wall == nullptr) {
            corner_conductance_row<C, true, false>(run, count, to);
        } else {
            corner_conductance_row<C, true, true>(run, count, to);
        }
        conductance = to;
    }
    if (field) {
        lanes::for_each_block(0, count, [&](auto cells, std::size_t m) {
            corner_stress<C>(cells, other_low, other_high, face_width, m,
                             out.stress.data() + first);
        });
    }
    corner_flux_row<C>({other_low, other_high, o.width_s.data() + s - 1, o.width_s.data() + s,
                        conductance, half_rho_},
                       count, out.flux.data() + first, out.plus.data() + first,
                       out.minus.data() + first);

    // The deferred correction. A domain edge's face takes the edge's value
    // under both schemes and adds nothing.
    const bool obstacles = w.row_walls[corner / w.corners.cells(0)] != 0;
    double* deferred = out.deferred.data();
    const auto with_weights = [&](cross_run run, std::size_t r, std::size_t i) {
        for (std::size_t m = 0; m < 3; ++m) {
            run.pos[m] = w.t_pos[m].data() + r;
            run.neg[m] = w.t_neg[m].data() + r;
        }
        for (std::size_t m = 0; m < 2; ++m) {
            run.wall_pos[m] = w.wall_pos[m].data() + r;
            run.wall_neg[m] = w.wall_neg[m].data() + r;
        }
        run.far_wall_low = w.far_wall_low.data() + corner + i;
        run.far_wall_high = w.far_wall_high.data() + corner + i;
        run.wall_face = w.wall_face.data() + corner + i;
        run.flux = out.flux.data() + i;
        return run;
    };
    const std::size_t edge_at = edge_index(o.faces, T, at);
    if constexpr (along_row<T>) {
        const double* row = own + index_of(o.faces, at);
        const double below = lanes::select(w.low_known[edge_at], w.low_value[edge_at], row[0]);
        const double above =
            lanes::select(w.high_known[edge_at], w.high_value[edge_at], row[nx - 1]);
        // Face i lies between nodes i - 1 and i of the row.
        const auto run_at = [&](std::size_t i) {
            cross_run run{};
            run.low = row + i - 1;
            run.high = row + i;
            run.far_low = i >= 2 ? row + i - 2 : &below;
            run.far_high = i + 1 < nx ? row + i + 1 : &above;
            return with_weights(run, i, i);
        };
        deferred[0] = 0.0;
        deferred[nx] = 0.0;
        cross_row<T, false>(run_at(1), obstacles, 0, 1, deferred + 1);
        if (nx >= 4) {
            cross_row<T, true>(run_at(2), obstacles, 0, nx - 3, deferred + 2);
        }
        if (nx >= 3) {
            cross_row<T, false>(run_at(nx - 1), obstacles, 0, 1, deferred + nx - 1);
        }
    } else {
        const std::size_t r = at[T];
        if (r == 0 || r == w.nt) {
            std::fill(deferred, deferred + nx + 1, 0.0);
            return;
        }
        // Face r lies between the rows of nodes r - 1 and r.
        const std::size_t stride = o.faces.cell_stride(T);
        const std::size_t width = o.faces.cells(0);
        const double* high_row = own + index_of(o.faces, at);
        const double* low_row = high_row - stride;
        cross_run run{};
        run.low = low_row + first;
        run.high = high_row + first;
        run.far_low = run.low - stride;
        run.far_high = run.high + stride;
        if (r == 1) {
            double* edge = edge_.data();
            const mask* known = w.low_known.data() + edge_at;
            const double* value = w.low_value.data() + edge_at;
            lanes::for_each_block(0, width, [&](auto cells, std::size_t m) {
                edge_nodes(cells, known, value, low_row, m, edge);
            });
            run.far_low = edge + first;
        }
        if (r + 1 == w.nt) {
            double* edge = edge_.data() + width;
            const mask* known = w.high_known.data() + edge_at;
            const double* value = w.high_value.data() + edge_at;
            lanes::for_each_block(0, width, [&](auto cells, std::size_t m) {
                edge_nodes(cells, known, value, high_row, m, edge);
            });
            run.far_high = edge + first;
        }
        cross_row<T, false>(with_weights(run, r, first), obstacles, 0, count, deferred + first);
    }
}

// The domain edges normal to direction T beside the row of unknowns at `at`:
// the row itself where T crosses the rows, its end cells where T lies along them.
template <int D>
    requires(D >= 1)
template <int C, int T>
void predictor<D>::edge_rows(const position& at, const flux_sides& corners, double* diagonal) {
    const component_setup<D>& o = setup_[C];
    constexpr int q = slot_of(C, T);
    const pair_setup<D>& w = o.pairs[q];
    const std::size_t row = index_of(o.faces, at);
    const std::size_t edge_at = edge_index(o.faces, T, at);
    double* toward_high = coefficient_[2 + 2 * q].data() + row;
    double* toward_low = coefficient_[3 + 2 * q].data() + row;
    // The unknowns from `first` up to `last` of the row, whose faces on the
    // edge's side have the coefficients `faces` at the same elements.
    const auto beside = [&](const double* faces, const std::vector<mask>& known,
                            const std::vector<double>& value, double* neighbour,
                            std::size_t first, std::size_t last) {
        edge_run run{};
        run.coefficient = faces;
        run.known = known.data() + edge_at;
        run.value = value.data() + edge_at;
        run.unknown = o.unknown.data() + row;
        run.alpha = alpha_;
        edge_unknown_row<T>(run, first, last, neighbour, diagonal + row, relaxed_.data() + row,
                            source_.data() + row);
    };
    if constexpr (along_row<T>) {
        const std::size_t nx = nx_;
        beside(corners.low->minus.data(), w.low_known, w.low_value, toward_low, 0, 1);
        beside(corners.low->plus.data() + 1, w.high_known, w.high_value, toward_high, nx - 1, nx);
    } else {
        const std::size_t first = along_row<C> ? 1 : 0;
        if (at[T] == 0) {
            beside(corners.low->minus.data(), w.low_known, w.low_value, toward_low, first, nx_);
        }
        if (at[T] + 1 == w.nt) {
            beside(corners.high->plus.data(), w.high_known, w.high_value, toward_high, first,
                   nx_);
        }
    }
}

// The equations of the row of unknowns at `at`, from the rows of fluxes about
// it: `cells` along s, and by slot `corners` along each transverse direction.
template <int D>
    requires(D >= 1)
template <int C>
void predictor<D>::unknown_row(const position& at, const double* own, const double* p,
                               const double* mu, const flux_sides& cells,
                               const std::array<flux_sides, D>& corners, double* diagonal) {
    constexpr transverse_directions<D, C> across{};
    constexpr std::size_t count = D == 1 ? 1 : D - 1;
    const component_setup<D>& o = setup_[C];
    const std::size_t nx = nx_;
    const std::size_t row = index_of(o.faces, at);
    const std::size_t first = along_row<C> ? 1 : 0;
    const std::size_t last = nx;
    double* known = source_.data() + row;

    diagonal_run<count> run{};
    run.stream = {cells.high->flux.data(), cells.low->flux.data(), cells.high->plus.data(),
                  cells.high->minus.data(), cells.low->minus.data()};
    run.unknown = o.unknown.data() + row;
    run.alpha = alpha_;
    stream_neighbour_row<C>(run.stream, run.unknown, first, last, coefficient_[0].data() + row,
                            coefficient_[1].data() + row);
    for_each_transverse<C>(across, [&]<int T> {
        constexpr int q = slot_of(C, T);
        const pair_setup<D>& w = o.pairs[q];
        run.cross[q] = {corners[q].low->flux.data(), corners[q].high->flux.data(),
                        corners[q].low->plus.data(), corners[q].high->plus.data(),
                        corners[q].low->minus.data()};
        cross_neighbour_row<T>(run.cross[q], run.unknown, w.couples_high.data() + row,
                               w.couples_low.data() + row, first, last,
                               coefficient_[2 + 2 * q].data() + row,
                               coefficient_[3 + 2 * q].data() + row);
    });
    diagonal_row<C>(across, run, first, last, diagonal + row, relaxed_.data() + row, known);
    for_each_transverse<C>(across, [&]<int T> {
        edge_rows<C, T>(at, corners[slot_of(C, T)], diagonal);
    });
    if constexpr (along_row<C>) {
        // The domain-edge faces, where no unknown is and another component's
        // entries may stand.
        diagonal[row] = 0.0;
        diagonal[row + nx] = 0.0;
        for (auto& a : coefficient_) {
            a[row] = 0.0;
            a[row + nx] = 0.0;
        }
    }

    const std::size_t cell = index_of(grid_->fields(), at);
    const std::size_t cell_below = along_row<C> ? cell : cell - grid_->fields().cell_stride(C);
    source_run<count> rhs{};
    rhs.cell_deferred = cells.high->deferred.data();
    rhs.cell_deferred_below = cells.low->deferred.data();
    for_each_transverse<C>(across, [&]<int T> {
        constexpr int q = slot_of(C, T);
        rhs.cross[q] = {corners[q].low->deferred.data(),  corners[q].high->deferred.data(),
                        corners[q].low->viscosity.data(), corners[q].high->viscosity.data(),
                        corners[q].low->stress.data(),    corners[q].high->stress.data()};
        rhs.width_t[q] = o.pairs[q].width_t.data() + at[T];
    });
    rhs.p = p + cell;
    rhs.p_below = p + cell_below;
    rhs.own = own + row;
    rhs.a_p = diagonal + row;
    if (mu != nullptr) {
        rhs.mu = mu + cell;
        rhs.mu_below = mu + cell_below;
    }
    rhs.mean_high = o.mean_high.data() + at[C];
    rhs.cell_stress = cells.high->stress.data();
    rhs.cell_stress_below = cells.low->stress.data();
    rhs.relaxation = relaxation_;
    if (mu != nullptr) {
        source_row<C, true>(across, rhs, first, last, known);
    } else {
        source_row<C, false>(across, rhs, first, last, known);
    }
}

// Component C's equations. The rows of unknowns are walked once; each row of
// fluxes is formed as the walk reaches it and kept while two rows of unknowns
// read it.
template <int D>
    requires(D >= 1)
template <int C>
void predictor<D>::assemble(const std::array<std::span<const double>, D>& velocity,
                            std::span<const double> pressure, std::span<const double> mu_eff,
                            [[maybe_unused]] const per_pair<std::span<const double>, D>& wall_mu,
                            double* diagonal) {
    constexpr transverse_directions<D, C> across{};
    const double* mu = mu_eff.empty() ? nullptr : mu_eff.data();
    const double* own = velocity[C].data();
    const double* p = pressure.data();
    const component_setup<D>& o = setup_[C];
    position first{};
    position last{};
    for (int d = 0; d < D; ++d) {
        last[d] = grid_->cells(d);
    }
    first[C] = along_row<C> ? 0 : 1;
    for_each_row(first, last, [&](const position& at) {
        flux_sides cells;
        if constexpr (along_row<C>) {
            cell_row<C>(at, own, mu, cell_rows_[0]);
            cells = {&cell_rows_[0], &cell_rows_[0]};
        } else {
            const position below = before(at, C);
            flux_row& low = cell_rows_[kept_row(C, below)];
            flux_row& high = cell_rows_[kept_row(C, at)];
            if (at[C] == 1) {
                cell_row<C>(below, own, mu, low);
            }
            cell_row<C>(at, own, mu, high);
            cells = {&low, &high};
        }
        std::array<flux_sides, D> corners{};
        for_each_transverse<C>(across, [&]<int T> {
            constexpr int q = slot_of(C, T);
            const double* other = velocity[T].data();
            const double* wall = wall_mu[C][q].empty() ? nullptr : wall_mu[C][q].data();
            if constexpr (along_row<T>) {
                flux_row& row = corner_rows_[q][0];
                corner_row<C, T>(at, own, other, mu, wall, row);
                corners[q] = {&row, &row};
            } else {
                const position above = after(at, T);
                flux_row& low = corner_rows_[q][kept_row(T, at)];
                flux_row& high = corner_rows_[q][kept_row(T, above)];
                if (at[T] == 0) {
                    corner_row<C, T>(at, own, other, mu, wall, low);
                }
                corner_row<C, T>(above, own, other, mu, wall, high);
                corners[q] = {&low, &high};
            }
        });
        unknown_row<C>(at, own, p, mu, cells, corners, diagonal);
    });
    if constexpr (!along_row<C>) {
        // The rows of domain-edge faces, as a row's ends in `unknown_row`.
        for (const std::size_t plane : {std::size_t{0}, o.ns}) {
            first[C] = plane;
            last[C] = plane + 1;
            for_each_row(first, last, [&](const position& at) {
                const std::size_t row = index_of(o.faces, at);
                std::fill(diagonal + row, diagonal + row + nx_, 0.0);
                for (auto& a : coefficient_) {
                    std::fill(a.data() + row, a.data() + row + nx_, 0.0);
                }
            });
        }
    }
}

// The under-relaxed Jacobi sweeps of component C's equations from its input,
// `own`, into `star` (`_sweep_n`).
template <int D>
    requires(D >= 1)
template <int C>
void predictor<D>::sweep(const double* own, const double* diagonal, double* star) {
    const component_setup<D>& o = setup_[C];
    const std::size_t nx = nx_;
    position first_row{};
    position last_row{};
    for (int d = 0; d < D; ++d) {
        last_row[d] = grid_->cells(d);
    }
    first_row[C] = along_row<C> ? 0 : 1;
    const std::size_t first = along_row<C> ? 1 : 0;
    // The domain-edge faces, which keep the input in every iterate.
    const auto edges = [&](double* to) {
        if constexpr (along_row<C>) {
            for_each_row(first_row, last_row, [&](const position& at) {
                const std::size_t row = index_of(o.faces, at);
                to[row] = own[row];
                to[row + nx] = own[row + nx];
            });
        } else {
            for (const std::size_t plane : {std::size_t{0}, o.ns}) {
                position low = first_row;
                position high = last_row;
                low[C] = plane;
                high[C] = plane + 1;
                for_each_row(low, high, [&](const position& at) {
                    const std::size_t row = index_of(o.faces, at);
                    std::copy(own + row, own + row + nx, to + row);
                });
            }
        }
    };
    edges(star);
    if (sweeps_ > 1) {
        edges(iterate_.data());
    }
    const double* current = own;
    for (std::int64_t n = 0; n < sweeps_; ++n) {
        // The last sweep writes star, so the sweeps before it alternate.
        double* next = (sweeps_ - n) % 2 == 1 ? star : iterate_.data();
        for_each_row(first_row, last_row, [&](const position& at) {
            const std::size_t row = index_of(o.faces, at);
            sweep_run<D> run{};
            run.x = current + row;
            for (int d = 1; d < D; ++d) {
                const std::size_t stride = o.faces.cell_stride(d);
                run.below[d] = at[d] > 0 ? run.x - stride : zeros_.data();
                run.above[d] = at[d] + 1 < o.faces.cells(d) ? run.x + stride : zeros_.data();
            }
            for (int part = 0; part < D; ++part) {
                run.plus[part] = coefficient_[2 * part].data() + row;
                run.minus[part] = coefficient_[2 * part + 1].data() + row;
            }
            run.b = source_.data() + row;
            run.relaxed = relaxed_.data() + row;
            run.a_p = diagonal + row;
            run.own = own + row;
            if (o.held.empty()) {
                sweep_row<C, false>(run, zeros_.data(), first, nx, next + row);
            } else {
                run.held = o.held.data() + row;
                sweep_row<C, true>(run, zeros_.data(), first, nx, next + row);
            }
        });
        current = next;
    }
}

template <int D>
    requires(D >= 1)
void predictor<D>::predict(const std::array<std::span<const double>, D>& velocity,
                           std::span<const double> p, std::span<const double> mu_eff,
                           const per_pair<std::span<const double>, D>& wall_mu,
                           const std::array<std::span<double>, D>& star,
                           const std::array<std::span<double>, D>& diagonal) {
    const auto component = [&]<int C> {
        assemble<C>(velocity, p, mu_eff, wall_mu, diagonal[C].data());
        sweep<C>(velocity[C].data(), diagonal[C].data(), star[C].data());
    };
    [&]<int... C>(std::integer_sequence<int, C...>) {
        (component.template operator()<C>(), ...);
    }(std::make_integer_sequence<int, D>{});
}

template <int D>
    requires(D >= 1)
void predictor<D>::equations(int c, const std::array<std::span<const double>, D>& velocity,
                             std::span<const double> p, std::span<const double> mu_eff,
                             const per_pair<std::span<const double>, D>& wall_mu,
                             std::span<double> diagonal) {
    const auto component = [&]<int C> {
        if (c == C) {
            assemble<C>(velocity, p, mu_eff, wall_mu, diagonal.data());
        }
    };
    [&]<int... C>(std::integer_sequence<int, C...>) {
        (component.template operator()<C>(), ...);
    }(std::make_integer_sequence<int, D>{});
}

template <int D>
    requires(D >= 1)
void predictor<D>::stencil_viscosity(std::span<const double> mu_eff,
                                     [[maybe_unused]] const per_pair<std::span<double>, D>& out) {
    const auto pair = [&]<int C, int T> {
        const pair_setup<D>& w = setup_[C].pairs[slot_of(C, T)];
        const std::span<double> to = out[C][slot_of(C, T)];
        std::fill(to.begin(), to.end(), mu_);
        double* viscosity = corner_rows_[slot_of(C, T)][0].viscosity.data();
        const std::size_t from = along_row<C> ? 1 : 0;
        const std::size_t past = along_row<C> ? nx_ : w.corners.cells(0);
        position first{};
        position last{};
        for (int d = 0; d < D; ++d) {
            last[d] = w.corners.cells(d);
        }
        // The positions interior along s.
        first[C] = 1;
        last[C] = w.corners.cells(C) - 1;
        for_each_row(first, last, [&](const position& at) {
            corner_viscosity<C, T>(at, mu_eff.data(), viscosity);
            const std::size_t row = index_of(w.corners, at);
            const mask* read = w.wall_read.data() + row;
            lanes::for_each_block(from, past, [&](auto cells, std::size_t m) {
                read_cells(cells, read, viscosity, mu_, m, to.data() + row);
            });
        });
    };
    const auto component = [&]<int C> {
        for_each_transverse<C>(transverse_directions<D, C>{}, [&]<int T> {
            pair.template operator()<C, T>();
        });
    };
    [&]<int... C>(std::integer_sequence<int, C...>) {
        (component.template operator()<C>(), ...);
    }(std::make_integer_sequence<int, D>{});
}

template class predictor<1>;
template class predictor<2>;

}  // namespace cfdcore::momentum
