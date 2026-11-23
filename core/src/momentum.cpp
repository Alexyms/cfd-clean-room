#include "cfdcore/momentum.hpp"

#include "cfdcore/scalar.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace cfdcore::momentum {

namespace {

using lanes::mask;

// Whether a component's own axis s is the rows' axis x (u) or not (v).
template <int C>
inline constexpr bool own_along_row = C == 0;

// An operand that varies along s, or along t, at element i of a row: per
// element where that axis is the row's, the row's one value otherwise.
template <int C>
CFDCORE_INLINE double s_at(const double* p, std::size_t i) {
    return p[own_along_row<C> ? i : 0];
}
template <int C>
CFDCORE_INLINE double t_at(const double* p, std::size_t i) {
    return p[own_along_row<C> ? 0 : i];
}
template <int C>
CFDCORE_INLINE mask s_at(const mask* p, std::size_t i) {
    return p[own_along_row<C> ? i : 0];
}

// The streamwise faces on the low and the high side of unknown i: cells i - 1
// and i of the row for u, cell i of the row below and of the row for v.
template <int C>
CFDCORE_INLINE double cell_low(const double* below, const double* row, std::size_t i) {
    return own_along_row<C> ? row[i - 1] : below[i];
}

// The transverse face on the high side of unknown i: corner i of the corner
// row above for u, corner i + 1 of the row for v. The low one is corner i of
// the row for both.
template <int C>
CFDCORE_INLINE double corner_high(const double* row, const double* above, std::size_t i) {
    return own_along_row<C> ? above[i] : row[i + 1];
}

// `np.maximum(x, 0.0)`: a NaN stays, and -0.0 gives 0.0. The comparison is one
// and its mask is kept, where `scalar::np_maximum` of a constant zero folds
// to a branch that GCC 13 leaves in the loop (the block stays scalar).
CFDCORE_INLINE double positive_part(double x) {
    return lanes::keep(lanes::mask_of(!(x <= 0.0)), x);
}

// ---------------------------------------------------------------------------
// Conductances and mass fluxes

// The conductance of N streamwise faces with a viscosity field: mu dt / ds.
template <int C, std::size_t N>
inline void cell_conductance(lanes::cells<N>, const double* mu, const double* width_t,
                             const double* width_s, std::size_t first,
                             double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        out[i] = mu[i] * t_at<C>(width_t, i) / s_at<C>(width_s, i);
    }
}

// What the mass flux through a row of streamwise faces reads: the component's
// values at the nodes on either side.
struct cell_flux_run {
    const double* low;
    const double* high;
    const double* width_t;
    const double* conductance;
    double half_rho;
};

// The mass flux through N streamwise faces and the coefficients it gives the
// unknowns on either side: toward s+ for the one on its low side, toward s- for
// the one on its high side, first-order upwind.
template <int C, std::size_t N>
inline void cell_flux(lanes::cells<N>, const cell_flux_run& in, std::size_t first,
                      double* CFDCORE_RESTRICT flux, double* CFDCORE_RESTRICT plus,
                      double* CFDCORE_RESTRICT minus) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double f = in.half_rho * (in.low[i] + in.high[i]) * t_at<C>(in.width_t, i);
        flux[i] = f;
        plus[i] = in.conductance[i] + positive_part(-f);
        minus[i] = in.conductance[i] + positive_part(f);
    }
}

// What the mass flux through a row of transverse faces reads: the other
// component's values in the two half cells.
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
        const double f = in.half_rho * (in.low[i] * s_at<C>(in.width_low, i) +
                                        in.high[i] * s_at<C>(in.width_high, i));
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
            out[i] = lanes::keep(in.conducts[i], m * s_at<C>(in.face_width, i) / in.reach[i]);
        } else {
            out[i] = lanes::select(in.takes_wall[i],
                                   in.wall[i] * s_at<C>(in.face_width, i) / in.reach[i],
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

// The (QUICK - upwind) flux through N streamwise faces. `Inner` reads a u
// row's nodes off one pointer. `Obstacles` takes the boundary form where the
// far node lies beyond an obstacle's wall node (`_obstacle_streamwise`).
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
        const double q_pos = s_at<C>(in.std_pos[0], i) * below +
                             s_at<C>(in.std_pos[1], i) * lo + s_at<C>(in.std_pos[2], i) * hi;
        const double q_neg = s_at<C>(in.std_neg[0], i) * above +
                             s_at<C>(in.std_neg[1], i) * hi + s_at<C>(in.std_neg[2], i) * lo;
        double q = lanes::select(positive, q_pos, q_neg);
        if constexpr (Obstacles) {
            const double alt_pos = s_at<C>(in.alt_pos[0], i) * above +
                                   s_at<C>(in.alt_pos[1], i) * lo +
                                   s_at<C>(in.alt_pos[2], i) * hi;
            const double alt_neg = s_at<C>(in.alt_neg[0], i) * below +
                                   s_at<C>(in.alt_neg[1], i) * hi +
                                   s_at<C>(in.alt_neg[2], i) * lo;
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

// The (QUICK - upwind) flux through N interior transverse faces. `Inner` reads
// a v row's nodes off one pointer. `Obstacles` takes the wall as the far node
// beside an obstacle face and adds nothing at one (`_obstacle_transverse`).
template <int C, bool Inner, bool Obstacles, std::size_t N>
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
        const double q_pos = t_at<C>(in.pos[0], i) * below + t_at<C>(in.pos[1], i) * lo +
                             t_at<C>(in.pos[2], i) * hi;
        const double q_neg = t_at<C>(in.neg[0], i) * above + t_at<C>(in.neg[1], i) * hi +
                             t_at<C>(in.neg[2], i) * lo;
        double q = lanes::select(positive, q_pos, q_neg);
        if constexpr (Obstacles) {
            const double wall_pos =
                t_at<C>(in.wall_pos[0], i) * lo + t_at<C>(in.wall_pos[1], i) * hi;
            const double wall_neg =
                t_at<C>(in.wall_neg[0], i) * hi + t_at<C>(in.wall_neg[1], i) * lo;
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

// `_harmonic` of N pairs of cells across a transverse face, the SOLID one of a
// pair taking the other's value. The mean of two SOLID cells is whatever they
// hold (a NaN, say): no unknown's equation reads it, so the field needs no copy
// with SOLID cells at air's viscosity, as `_check_mu_eff` makes.
template <int C, std::size_t N>
inline void harmonic_cells(lanes::cells<N>, const double* low, const double* high,
                           const mask* solid_low, const mask* solid_high,
                           const double* frac_low, const double* frac_high, std::size_t first,
                           double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double a = lanes::select(solid_low[i] & ~solid_high[i], high[i], low[i]);
        const double b = lanes::select(solid_high[i] & ~solid_low[i], low[i], high[i]);
        const double frac_a = t_at<C>(frac_low, i);
        const double frac_b = t_at<C>(frac_high, i);
        out[i] = a + (b - a) * (a * frac_b) / (frac_a * b + frac_b * a);
    }
}

// `_lerp` of N pairs of rows' values across an interior position along s: the
// values at i - 1 and i of one row (u), or at i of two (v).
template <int C, std::size_t N>
inline void mean_cells(lanes::cells<N>, const double* below, const double* row,
                       const double* frac_high, std::size_t first,
                       double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double a = cell_low<C>(below, row, i);
        const double b = row[i];
        out[i] = a + (b - a) * s_at<C>(frac_high, i);
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

// The stress flux through N streamwise faces: du/ds dt (`_stress_source`).
template <int C, std::size_t N>
inline void cell_stress(lanes::cells<N>, const double* low, const double* high,
                        const double* width_s, const double* width_t, std::size_t first,
                        double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        out[i] = (high[i] - low[i]) / s_at<C>(width_s, i) * t_at<C>(width_t, i);
    }
}

// The stress flux through N transverse faces: the other component's gradient
// along s times the face's width. The quotient times the same width is the
// reference's, and rounds twice.
template <int C, std::size_t N>
inline void corner_stress(lanes::cells<N>, const double* low, const double* high,
                          const double* face_width, std::size_t first,
                          double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double w = s_at<C>(face_width, i);
        out[i] = (high[i] - low[i]) / w * w;
    }
}

// ---------------------------------------------------------------------------
// The equations of a row of unknowns

// The transverse domain edge a run of unknowns lies beside.
enum class beside { none, low, high };

// What the coefficients of a row of unknowns read: `cell_*` the streamwise
// faces' of the row and, for v, of the row below; `corner_*` the transverse
// faces' of the corner row and, for u, of the one above.
struct coefficient_run {
    const double* cell_flux;
    const double* cell_flux_below;
    const double* cell_plus;
    const double* cell_minus;
    const double* cell_minus_below;
    const double* corner_flux;
    const double* corner_flux_above;
    const double* corner_plus;
    const double* corner_plus_above;
    const double* corner_minus;
    const mask* unknown;
    const mask* couples_high;
    const mask* couples_low;
    const mask* edge_known;
    const double* edge_value;
    double alpha;
};

// The neighbour coefficients of N unknowns, the un-relaxed diagonal, the
// relaxed one the sweeps divide by (1 where a_P is not positive) and the known
// neighbour's contribution at a domain edge. Zero at faces that are not
// unknowns.
template <int C, beside Edge, std::size_t N>
inline void coefficients(lanes::cells<N>, const coefficient_run& in, std::size_t first,
                         double* CFDCORE_RESTRICT a_sp, double* CFDCORE_RESTRICT a_sm,
                         double* CFDCORE_RESTRICT a_tp, double* CFDCORE_RESTRICT a_tm,
                         double* CFDCORE_RESTRICT a_p, double* CFDCORE_RESTRICT relaxed,
                         double* CFDCORE_RESTRICT known) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double f_e = in.cell_flux[i];
        const double f_w = cell_low<C>(in.cell_flux_below, in.cell_flux, i);
        const double f_n = corner_high<C>(in.corner_flux, in.corner_flux_above, i);
        const double f_s = in.corner_flux[i];
        const double a_e = in.cell_plus[i];
        const double a_w = cell_low<C>(in.cell_minus_below, in.cell_minus, i);
        double a_n = corner_high<C>(in.corner_plus, in.corner_plus_above, i);
        double a_s = in.corner_minus[i];
        double p = a_e + a_w + a_n + a_s + (f_e - f_w + f_n - f_s);
        // A Dirichlet edge's value is a known neighbour and moves to the
        // source; a zero-gradient edge has the unknown itself as its
        // neighbour, so that coefficient leaves the diagonal.
        double b = 0.0;
        if constexpr (Edge == beside::high) {
            const mask is_known = s_at<C>(in.edge_known, i);
            b = b + lanes::keep(is_known, a_n * s_at<C>(in.edge_value, i));
            p = p - lanes::keep(~is_known, a_n);
            a_n = 0.0;
        }
        if constexpr (Edge == beside::low) {
            const mask is_known = s_at<C>(in.edge_known, i);
            b = b + lanes::keep(is_known, a_s * s_at<C>(in.edge_value, i));
            p = p - lanes::keep(~is_known, a_s);
            a_s = 0.0;
        }
        const mask here = in.unknown[i];
        p = lanes::keep(here, p);
        a_sp[i] = lanes::keep(here, a_e);
        a_sm[i] = lanes::keep(here, a_w);
        a_tp[i] = lanes::keep(here & in.couples_high[i], a_n);
        a_tm[i] = lanes::keep(here & in.couples_low[i], a_s);
        a_p[i] = p;
        relaxed[i] = lanes::select(lanes::mask_of(p > 0.0), p / in.alpha, 1.0);
        known[i] = b;
    }
}

// What the sources of a row of unknowns read, beside their diagonal.
struct source_run {
    const double* cell_deferred;
    const double* cell_deferred_below;
    const double* corner_deferred;
    const double* corner_deferred_above;
    const double* p;
    const double* p_below;
    const double* width_t;
    const double* own;
    const double* a_p;
    // With a viscosity field:
    const double* mu;
    const double* mu_below;
    const double* mean_high;
    const double* corner_mu;
    const double* corner_mu_above;
    const double* cell_stress;
    const double* cell_stress_below;
    const double* corner_stress;
    const double* corner_stress_above;
    double relaxation;
};

// The source of N unknowns: the known neighbours' (already in `b`), the
// deferred correction with form b's stress source, the pressure's and the
// relaxation term's, added in the reference's order. Read at unknowns only.
template <int C, bool Field, std::size_t N>
inline void sources(lanes::cells<N>, const source_run& in, std::size_t first,
                    double* CFDCORE_RESTRICT b) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        double deferred = -(in.cell_deferred[i] -
                            cell_low<C>(in.cell_deferred_below, in.cell_deferred, i) +
                            corner_high<C>(in.corner_deferred, in.corner_deferred_above, i) -
                            in.corner_deferred[i]);
        if constexpr (Field) {
            const double mu_high = in.mu[i];
            const double mu_low = cell_low<C>(in.mu_below, in.mu, i);
            const double mu_f = mu_low + (mu_high - mu_low) * s_at<C>(in.mean_high, i);
            const double stress =
                (mu_high - mu_f) * in.cell_stress[i] -
                (mu_low - mu_f) * cell_low<C>(in.cell_stress_below, in.cell_stress, i) +
                (corner_high<C>(in.corner_mu, in.corner_mu_above, i) - mu_f) *
                    corner_high<C>(in.corner_stress, in.corner_stress_above, i) -
                (in.corner_mu[i] - mu_f) * in.corner_stress[i];
            deferred = deferred + stress;
        }
        const double pressure =
            -(in.p[i] - cell_low<C>(in.p_below, in.p, i)) * t_at<C>(in.width_t, i);
        b[i] = b[i] + deferred + pressure + in.relaxation * in.a_p[i] * in.own[i];
    }
}

// ---------------------------------------------------------------------------
// The sweeps

// What a Jacobi sweep of a row of unknowns reads: the iterate in the row, and
// in the rows below and above, a row of zeros standing for one beyond the
// domain; `west` and `east` for a v row's end cells alone, whose missing
// neighbour is a zero (the reference's padding).
struct sweep_run {
    const double* x;
    const double* below;
    const double* above;
    const double* west;
    const double* east;
    const double* a_sp;
    const double* a_sm;
    const double* a_tp;
    const double* a_tm;
    const double* b;
    const double* relaxed;
    const double* a_p;
    const mask* held;
    const double* own;
};

// One Jacobi sweep of N unknowns of a row; a face that is not an unknown is
// zero, or with `Held` keeps a vent face's input value.
template <int C, bool Inner, bool Held, std::size_t N>
inline void sweep_cells(lanes::cells<N>, const sweep_run& in, std::size_t first,
                        double* CFDCORE_RESTRICT out) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double west = Inner ? in.x[i - 1] : in.west[i];
        const double east = Inner ? in.x[i + 1] : in.east[i];
        const double sp = own_along_row<C> ? east : in.above[i];
        const double sm = own_along_row<C> ? west : in.below[i];
        const double tp = own_along_row<C> ? in.above[i] : east;
        const double tm = own_along_row<C> ? in.below[i] : west;
        const double numerator = in.a_sp[i] * sp + in.a_sm[i] * sm + in.a_tp[i] * tp +
                                 in.a_tm[i] * tm + in.b[i];
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
sweep_run sweep_at(const sweep_run& row, std::size_t m) {
    sweep_run at = row;
    at.x = row.x + m;
    at.below = row.below + m;
    at.above = row.above + m;
    at.a_sp = row.a_sp + m;
    at.a_sm = row.a_sm + m;
    at.a_tp = row.a_tp + m;
    at.a_tm = row.a_tm + m;
    at.b = row.b + m;
    at.relaxed = row.relaxed + m;
    at.a_p = row.a_p + m;
    at.held = row.held == nullptr ? nullptr : row.held + m;
    at.own = row.own + m;
    return at;
}

template <int C, bool Held>
void sweep_row(const sweep_run row, const double* zeros, std::size_t first, std::size_t last,
               double* out) {
    if constexpr (own_along_row<C>) {
        // The ends of a u row are domain-edge faces, which no sweep writes, so
        // every unknown has both neighbours in its row.
        lanes::for_each_block(first, last, [&](auto cells, std::size_t m) {
            sweep_cells<C, true, Held>(cells, row, m, out);
        });
    } else {
        sweep_run end = sweep_at(row, first);
        end.west = zeros;
        end.east = row.x + first + 1;
        sweep_cells<C, false, Held>(lanes::cells<1>{}, end, 0, out + first);
        lanes::for_each_block(first + 1, last - 1, [&](auto cells, std::size_t m) {
            sweep_cells<C, true, Held>(cells, row, m, out);
        });
        end = sweep_at(row, last - 1);
        end.west = row.x + last - 2;
        end.east = zeros;
        sweep_cells<C, false, Held>(lanes::cells<1>{}, end, 0, out + last - 1);
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

template <int C, bool Inner>
void cross_row(const cross_run run, bool obstacles, std::size_t first, std::size_t last,
               double* out) {
    if (obstacles) {
        lanes::for_each_block(first, last, [&](auto cells, std::size_t m) {
            corner_deferred<C, Inner, true>(cells, run, m, out);
        });
    } else {
        lanes::for_each_block(first, last, [&](auto cells, std::size_t m) {
            corner_deferred<C, Inner, false>(cells, run, m, out);
        });
    }
}

template <int C>
void cell_flux_row(const cell_flux_run run, std::size_t count, double* flux, double* plus,
                   double* minus) {
    lanes::for_each_block(0, count, [&](auto cells, std::size_t m) {
        cell_flux<C>(cells, run, m, flux, plus, minus);
    });
}

template <int C>
void corner_flux_row(const corner_flux_run run, std::size_t count, double* flux, double* plus,
                     double* minus) {
    lanes::for_each_block(0, count, [&](auto cells, std::size_t m) {
        corner_flux<C>(cells, run, m, flux, plus, minus);
    });
}

template <int C, beside Edge>
void coefficient_row(const coefficient_run run, std::size_t first, std::size_t last, double* a_sp,
                     double* a_sm, double* a_tp, double* a_tm, double* a_p, double* relaxed,
                     double* known) {
    lanes::for_each_block(first, last, [&](auto cells, std::size_t m) {
        coefficients<C, Edge>(cells, run, m, a_sp, a_sm, a_tp, a_tm, a_p, relaxed, known);
    });
}

template <int C, bool Field>
void source_row(const source_run run, std::size_t first, std::size_t last, double* b) {
    lanes::for_each_block(first, last, [&](auto cells, std::size_t m) {
        sources<C, Field>(cells, run, m, b);
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
// The set-up, in the reference's frame: t the rows, s along them

struct frame {
    std::size_t nx;
    int component;
    std::size_t cell(std::size_t t, std::size_t s) const {
        return component == 0 ? t * nx + s : s * nx + t;
    }
    std::size_t corner(std::size_t t, std::size_t s) const {
        return component == 0 ? t * (nx + 1) + s : s * (nx + 1) + t;
    }
    std::size_t face(std::size_t t, std::size_t s) const {
        return component == 0 ? t * (nx + 1) + s : s * nx + t;
    }
};

void copy_weights(std::array<std::vector<double>, 3>& into, std::size_t at,
                  const std::array<double, 3>& weights) {
    for (std::size_t m = 0; m < 3; ++m) {
        into[m][at] = weights[m];
    }
}

template <int D>
component_setup make_setup(const grid::grid<D>& g, const predictor_input<D>& in, int c) {
    const int own = c;
    const int across = 1 - c;
    const std::size_t nx = g.cells(0);
    const std::size_t ny = g.cells(1);
    const frame at{nx, c};
    component_setup o;
    o.ns = g.cells(own);
    o.nt = g.cells(across);
    const std::size_t ns = o.ns;
    const std::size_t nt = o.nt;
    const std::size_t cells = nx * ny;
    const std::size_t corners = (nx + 1) * (ny + 1);
    const std::size_t faces = nt * (ns + 1);
    const auto as_vector = [](std::span<const double> a) {
        return std::vector<double>(a.begin(), a.end());
    };
    o.width_s = as_vector(in.cell_widths[own]);
    o.width_t = as_vector(in.cell_widths[across]);
    o.face_width = as_vector(in.face_widths[own]);
    const std::vector<double> across_width = as_vector(in.face_widths[across]);

    // The positions along each axis: faces 0 to n, the centres between them.
    const auto positions = [&](int d) {
        const auto nodes = g.nodes(d);
        const auto interior = g.interior_faces(d);
        std::vector<double> all{nodes.front()};
        all.insert(all.end(), interior.begin(), interior.end());
        all.push_back(nodes.back());
        return all;
    };
    const std::vector<double> s_faces = positions(own);
    const std::vector<double> t_faces = positions(across);
    const auto s_centers = g.nodes(own).subspan(1, ns);
    const auto t_centers = g.nodes(across).subspan(1, nt);
    const auto t_nodes = g.nodes(across);

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
    for (auto* w : {&o.t_pos, &o.t_neg}) {
        for (auto& v : *w) {
            v.assign(nt + 1, 0.0);
        }
    }
    for (auto* w : {&o.wall_pos, &o.wall_neg}) {
        for (auto& v : *w) {
            v.assign(nt + 1, 0.0);
        }
    }
    o.harmonic_low.assign(nt + 1, 0.0);
    o.harmonic_high.assign(nt + 1, 0.0);
    for (std::size_t r = 1; r < nt; ++r) {
        const double f = t_faces[r];
        copy_weights(o.t_pos, r,
                     scalar::lagrange_weights(t_nodes[r - 1], t_nodes[r], t_nodes[r + 1], f));
        copy_weights(o.t_neg, r,
                     scalar::lagrange_weights(t_nodes[r + 2], t_nodes[r + 1], t_nodes[r], f));
        const auto pos =
            scalar::lagrange_weights(t_faces[r - 1], t_centers[r - 1], t_centers[r], f);
        const auto neg =
            scalar::lagrange_weights(t_faces[r + 1], t_centers[r], t_centers[r - 1], f);
        for (std::size_t m = 0; m < 2; ++m) {
            o.wall_pos[m][r] = pos[m + 1];
            o.wall_neg[m][r] = neg[m + 1];
        }
        const double d_lo = f - t_centers[r - 1];
        const double d_hi = t_centers[r] - f;
        o.harmonic_low[r] = d_lo / (d_lo + d_hi);
        o.harmonic_high[r] = 1.0 - o.harmonic_low[r];
    }
    o.mean_high.assign(ns + 1, 0.0);
    for (std::size_t s = 1; s < ns; ++s) {
        const double w_l = s_faces[s] - s_centers[s - 1];
        const double w_r = s_centers[s] - s_faces[s];
        o.mean_high[s] = w_r / (w_l + w_r);
    }

    // The unknowns and the obstacle walls (`_obstacle_walls`).
    const auto solid = [&](std::size_t t, std::size_t s) { return g.solid(at.cell(t, s)); };
    const auto solid_face = [&](std::size_t t, std::size_t s) {
        return solid(t, s - 1) || solid(t, s);
    };
    o.unknown.assign(faces, lanes::none);
    o.couples_high.assign(faces, lanes::all);
    o.couples_low.assign(faces, lanes::all);
    o.wall_face.assign(corners, lanes::none);
    o.far_wall_low.assign(corners, lanes::none);
    o.far_wall_high.assign(corners, lanes::none);
    std::vector<double> wall_distance(corners, 1.0);
    for (std::size_t t = 0; t < nt; ++t) {
        for (std::size_t s = 1; s < ns; ++s) {
            if (solid_face(t, s)) {
                continue;
            }
            o.unknown[at.face(t, s)] = lanes::all;
            if (t + 1 < nt && solid_face(t + 1, s)) {
                // The unknown's north face is an obstacle's: it is the far
                // node of a flow toward decreasing t through its south face.
                o.couples_high[at.face(t, s)] = lanes::none;
                o.wall_face[at.corner(t + 1, s)] = lanes::all;
                wall_distance[at.corner(t + 1, s)] = t_faces[t + 1] - t_centers[t];
                if (t > 0) {
                    o.far_wall_high[at.corner(t, s)] = lanes::all;
                }
            }
            if (t > 0 && solid_face(t - 1, s)) {
                o.couples_low[at.face(t, s)] = lanes::none;
                o.wall_face[at.corner(t, s)] = lanes::all;
                wall_distance[at.corner(t, s)] = t_centers[t] - t_faces[t];
                if (t + 1 < nt) {
                    o.far_wall_low[at.corner(t + 1, s)] = lanes::all;
                }
            }
        }
    }
    o.blocked_pos.assign(cells, lanes::none);
    o.blocked_neg.assign(cells, lanes::none);
    for (std::size_t t = 0; t < nt; ++t) {
        for (std::size_t s = 0; s < ns; ++s) {
            o.blocked_pos[at.cell(t, s)] = lanes::mask_of(s > 0 && solid(t, s - 1));
            o.blocked_neg[at.cell(t, s)] = lanes::mask_of(s + 1 < ns && solid(t, s + 1));
        }
    }

    // The edges' conditions.
    const edge_condition& low = in.low[c];
    const edge_condition& high = in.high[c];
    o.low_known.resize(ns + 1);
    o.high_known.resize(ns + 1);
    for (std::size_t s = 0; s <= ns; ++s) {
        o.low_known[s] = lanes::mask_of(low.is_dirichlet[s]);
        o.high_known[s] = lanes::mask_of(high.is_dirichlet[s]);
    }
    o.low_value = as_vector(low.value);
    o.high_value = as_vector(high.value);

    // What each transverse face's conductance divides by, and whose
    // viscosity it takes; then the scalar path's conductances.
    o.reach.assign(corners, 1.0);
    o.conducts.assign(corners, lanes::all);
    o.takes_wall.assign(corners, lanes::none);
    o.wall_read.assign(corners, lanes::none);
    o.conductance_t.assign(corners, 0.0);
    o.conductance_s.assign(cells, 0.0);
    for (std::size_t t = 0; t <= nt; ++t) {
        for (std::size_t s = 0; s <= ns; ++s) {
            const std::size_t i = at.corner(t, s);
            o.reach[i] = across_width[t];
            if (t == 0 || t == nt) {
                const mask known = t == 0 ? o.low_known[s] : o.high_known[s];
                o.reach[i] = t == 0 ? low.wall_distance : high.wall_distance;
                o.conducts[i] = known;
                o.takes_wall[i] = known;
                if (s > 0 && s < ns) {
                    o.wall_read[i] = known & o.unknown[at.face(t == 0 ? 0 : nt - 1, s)];
                }
            }
            if (o.wall_face[i] != lanes::none) {
                o.reach[i] = wall_distance[i];
                o.takes_wall[i] = lanes::all;
                o.wall_read[i] = lanes::all;
            }
            o.conductance_t[i] = lanes::keep(o.conducts[i], in.mu * o.face_width[s] / o.reach[i]);
        }
    }
    for (std::size_t t = 0; t < nt; ++t) {
        for (std::size_t s = 0; s < ns; ++s) {
            o.conductance_s[at.cell(t, s)] = in.mu * o.width_t[t] / o.width_s[s];
        }
    }

    if (!in.held[c].empty()) {
        o.held.resize(faces);
        for (std::size_t i = 0; i < faces; ++i) {
            o.held[i] = lanes::mask_of(in.held[c][i]);
        }
    }
    o.row_blocked.assign(ny, 0);
    o.row_walls.assign(ny + 1, 0);
    for (std::size_t i = 0; i < cells; ++i) {
        if ((o.blocked_pos[i] | o.blocked_neg[i]) != lanes::none) {
            o.row_blocked[i / nx] = 1;
        }
    }
    for (std::size_t i = 0; i < corners; ++i) {
        if ((o.wall_face[i] | o.far_wall_low[i] | o.far_wall_high[i]) != lanes::none) {
            o.row_walls[i / (nx + 1)] = 1;
        }
    }
    return o;
}

}  // namespace

template <int D>
    requires(D == 2)
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
        for (const edge_condition* e : {&in.low[d], &in.high[d]}) {
            if (e->is_dirichlet.size() != g.cells(d) + 1 || e->value.size() != g.cells(d) + 1) {
                return "an edge condition does not hold one entry per face along the edge";
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
    requires(D == 2)
predictor<D>::predictor(const grid::grid<D>& g, const predictor_input<D>& in)
    : grid_(&g),
      nx_(g.cells(0)),
      ny_(g.cells(1)),
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
        setup_[c] = make_setup(g, in, c);
        faces = std::max(faces, g.fields().face_count(c));
    }
    for (auto* a : {&coefficient_[0], &coefficient_[1], &coefficient_[2], &coefficient_[3],
                    &source_, &relaxed_, &iterate_}) {
        a->assign(faces, 0.0);
    }
    for (auto* rows : {&cell_rows_, &corner_rows_}) {
        for (flux_row& row : *rows) {
            for (auto* a : {&row.conductance, &row.flux, &row.plus, &row.minus, &row.deferred,
                            &row.stress, &row.viscosity}) {
                a->assign(nx_ + 1, 0.0);
            }
        }
    }
    for (auto& a : means_) {
        a.assign(nx_ + 1, 0.0);
    }
    edge_.assign(2 * (nx_ + 1), 0.0);
    zeros_.assign(nx_ + 1, 0.0);
}

template <int D>
    requires(D == 2)
viscosity_status predictor<D>::check_viscosity(std::span<const double> mu_eff) const {
    return verdict_of(mu_eff, grid_->solid_flags(), lanes::all);
}

template <int D>
    requires(D == 2)
viscosity_status predictor<D>::check_wall_viscosity(int c, std::span<const double> wall_mu) const {
    return verdict_of(wall_mu, setup_[c].wall_read, lanes::none);
}

// The viscosity at corner row j of the component's transverse faces
// (`_corner_viscosity`), at the positions interior along s. For v the rows are
// asked for in order from j = 1: each call keeps the harmonic means of its row
// of cells for the next.
template <int D>
    requires(D == 2)
template <int C>
void predictor<D>::corner_viscosity(std::size_t j, const double* mu, double* out) {
    const component_setup& o = setup_[C];
    const mask* solid = grid_->solid_flags().data();
    const std::size_t nx = nx_;
    if constexpr (own_along_row<C>) {
        const double* row = nullptr;
        if (j == 0 || j == ny_) {
            row = mu + (j == 0 ? 0 : (ny_ - 1) * nx);
        } else {
            double* mean = means_[0].data();
            const std::size_t low = (j - 1) * nx;
            lanes::for_each_block(0, nx, [&](auto cells, std::size_t m) {
                harmonic_cells<C>(cells, mu + low, mu + low + nx, solid + low, solid + low + nx,
                                  o.harmonic_low.data() + j, o.harmonic_high.data() + j, m, mean);
            });
            row = mean;
        }
        lanes::for_each_block(1, nx, [&](auto cells, std::size_t m) {
            mean_cells<C>(cells, row, row, o.mean_high.data(), m, out);
        });
    } else {
        const auto mean_row = [&](std::size_t q) {
            double* mean = means_[q % 2].data();
            const double* cell = mu + q * nx;
            const mask* flags = solid + q * nx;
            mean[0] = cell[0];
            // Face i of the row lies between cells i - 1 and i.
            lanes::for_each_block(0, nx - 1, [&](auto cells, std::size_t m) {
                harmonic_cells<C>(cells, cell, cell + 1, flags, flags + 1,
                                  o.harmonic_low.data() + 1, o.harmonic_high.data() + 1, m,
                                  mean + 1);
            });
            mean[nx] = cell[nx - 1];
        };
        if (j == 1) {
            mean_row(0);
        }
        mean_row(j);
        const double* below = means_[(j - 1) % 2].data();
        const double* row = means_[j % 2].data();
        lanes::for_each_block(0, nx + 1, [&](auto cells, std::size_t m) {
            mean_cells<C>(cells, below, row, o.mean_high.data() + j, m, out);
        });
    }
}

// The fluxes through row j of the streamwise faces, one per cell.
template <int D>
    requires(D == 2)
template <int C>
void predictor<D>::cell_row(std::size_t j, const double* own, const double* mu, flux_row& out) {
    const component_setup& o = setup_[C];
    const std::size_t nx = nx_;
    const std::size_t stride = own_along_row<C> ? 1 : nx;
    const double* low = own + j * (own_along_row<C> ? nx + 1 : nx);
    const double* high = low + stride;
    const double* width_s = o.width_s.data() + (own_along_row<C> ? 0 : j);
    const double* width_t = o.width_t.data() + (own_along_row<C> ? j : 0);
    const double* conductance = o.conductance_s.data() + j * nx;
    if (mu != nullptr) {
        const double* cell_mu = mu + j * nx;
        lanes::for_each_block(0, nx, [&](auto cells, std::size_t m) {
            cell_conductance<C>(cells, cell_mu, width_t, width_s, m, out.conductance.data());
        });
        conductance = out.conductance.data();
        lanes::for_each_block(0, nx, [&](auto cells, std::size_t m) {
            cell_stress<C>(cells, low, high, width_s, width_t, m, out.stress.data());
        });
    }
    cell_flux_row<C>({low, high, width_t, conductance, half_rho_}, nx, out.flux.data(),
                     out.plus.data(), out.minus.data());

    // The deferred correction: the run's operands from cell `first` on.
    const bool obstacles = o.row_blocked[j] != 0;
    const std::size_t k = own_along_row<C> ? 0 : j;
    const auto run_at = [&](std::size_t first) {
        stream_run run{};
        const std::size_t w = k + (own_along_row<C> ? first : 0);
        // The faces' position along s, whose ends have no node on one side.
        const std::size_t position = own_along_row<C> ? first : j;
        run.low = low + first;
        run.high = high + first;
        run.far_high = position + 1 < o.ns ? high + first + stride : low + first - stride;
        run.far_low = position > 0 ? low + first - stride : run.far_high;
        for (std::size_t m = 0; m < 3; ++m) {
            run.std_pos[m] = o.std_pos[m].data() + w;
            run.std_neg[m] = o.std_neg[m].data() + w;
            run.alt_pos[m] = o.alt_pos[m].data() + w;
            run.alt_neg[m] = o.alt_neg[m].data() + w;
        }
        run.blocked_pos = o.blocked_pos.data() + j * nx + first;
        run.blocked_neg = o.blocked_neg.data() + j * nx + first;
        run.flux = out.flux.data() + first;
        return run;
    };
    double* deferred = out.deferred.data();
    if constexpr (own_along_row<C>) {
        stream_row<C, false>(run_at(0), obstacles, 0, 1, deferred);
        stream_row<C, true>(run_at(1), obstacles, 0, nx - 2, deferred + 1);
        stream_row<C, false>(run_at(nx - 1), obstacles, 0, 1, deferred + nx - 1);
    } else {
        stream_row<C, false>(run_at(0), obstacles, 0, nx, deferred);
    }
}

// The fluxes through corner row j of the transverse faces, at the positions
// interior along s.
template <int D>
    requires(D == 2)
template <int C>
void predictor<D>::corner_row(std::size_t j, const double* own, const double* other,
                              const double* mu, const double* wall, flux_row& out) {
    const component_setup& o = setup_[C];
    const bool field = mu != nullptr;
    const std::size_t nx = nx_;
    const std::size_t corner = j * (nx + 1);
    // The first position and the count of those written: 1 to nx - 1 of a u
    // row, every one of a v row.
    const std::size_t first = own_along_row<C> ? 1 : 0;
    const std::size_t count = own_along_row<C> ? nx - 1 : nx + 1;
    // The other component on either half of the faces: v's cells i - 1 and i of
    // row j for u, u's rows j - 1 and j for v.
    const double* other_high = own_along_row<C> ? other + j * nx + 1 : other + j * (nx + 1);
    const double* other_low = own_along_row<C> ? other_high - 1 : other_high - (nx + 1);
    const std::size_t s = own_along_row<C> ? 1 : j;
    const double* face_width = o.face_width.data() + s;

    const double* conductance = o.conductance_t.data() + corner + first;
    if (field) {
        corner_viscosity<C>(j, mu, out.viscosity.data());
    }
    if (field || wall != nullptr) {
        const corner_conductance_run run{out.viscosity.data() + first,
                                         wall == nullptr ? nullptr : wall + corner + first,
                                         face_width,
                                         o.reach.data() + corner + first,
                                         o.conductance_t.data() + corner + first,
                                         o.takes_wall.data() + corner + first,
                                         o.conducts.data() + corner + first};
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
    const bool obstacles = o.row_walls[j] != 0;
    double* deferred = out.deferred.data();
    const auto with_weights = [&](cross_run run, std::size_t r, std::size_t i) {
        for (std::size_t m = 0; m < 3; ++m) {
            run.pos[m] = o.t_pos[m].data() + r;
            run.neg[m] = o.t_neg[m].data() + r;
        }
        for (std::size_t m = 0; m < 2; ++m) {
            run.wall_pos[m] = o.wall_pos[m].data() + r;
            run.wall_neg[m] = o.wall_neg[m].data() + r;
        }
        run.far_wall_low = o.far_wall_low.data() + corner + i;
        run.far_wall_high = o.far_wall_high.data() + corner + i;
        run.wall_face = o.wall_face.data() + corner + i;
        run.flux = out.flux.data() + i;
        return run;
    };
    if constexpr (own_along_row<C>) {
        if (j == 0 || j == ny_) {
            std::fill(deferred, deferred + nx + 1, 0.0);
            return;
        }
        const std::size_t width = nx + 1;
        cross_run run{};
        run.low = own + (j - 1) * width + 1;
        run.high = run.low + width;
        run.far_low = run.low - width;
        run.far_high = run.high + width;
        if (j == 1) {
            double* edge = edge_.data();
            lanes::for_each_block(0, width, [&](auto cells, std::size_t m) {
                edge_nodes(cells, o.low_known.data(), o.low_value.data(), own, m, edge);
            });
            run.far_low = edge + 1;
        }
        if (j + 1 == ny_) {
            double* edge = edge_.data() + width;
            const double* adjacent = own + (ny_ - 1) * width;
            lanes::for_each_block(0, width, [&](auto cells, std::size_t m) {
                edge_nodes(cells, o.high_known.data(), o.high_value.data(), adjacent, m, edge);
            });
            run.far_high = edge + 1;
        }
        cross_row<C, false>(with_weights(run, j, 1), obstacles, 0, nx - 1, deferred + 1);
    } else {
        const double* row = own + j * nx;
        const double below = lanes::select(o.low_known[j], o.low_value[j], row[0]);
        const double above = lanes::select(o.high_known[j], o.high_value[j], row[nx - 1]);
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
        cross_row<C, false>(run_at(1), obstacles, 0, 1, deferred + 1);
        if (nx >= 4) {
            cross_row<C, true>(run_at(2), obstacles, 0, nx - 3, deferred + 2);
        }
        if (nx >= 3) {
            cross_row<C, false>(run_at(nx - 1), obstacles, 0, 1, deferred + nx - 1);
        }
    }
}

// The equations of row j of the unknowns, from the rows of fluxes about it.
template <int D>
    requires(D == 2)
template <int C>
void predictor<D>::unknown_row(std::size_t j, const double* own, const double* p, const double* mu,
                               const flux_row* cell_minus, const flux_row* cell_plus,
                               const flux_row* corner_minus, const flux_row* corner_plus,
                               double* diagonal) {
    const component_setup& o = setup_[C];
    const std::size_t nx = nx_;
    const std::size_t row = j * (own_along_row<C> ? nx + 1 : nx);
    const std::size_t first = own_along_row<C> ? 1 : 0;
    const std::size_t last = nx;
    coefficient_run run{};
    run.cell_flux = cell_plus->flux.data();
    run.cell_flux_below = cell_minus->flux.data();
    run.cell_plus = cell_plus->plus.data();
    run.cell_minus = cell_plus->minus.data();
    run.cell_minus_below = cell_minus->minus.data();
    run.corner_flux = corner_minus->flux.data();
    run.corner_flux_above = corner_plus->flux.data();
    run.corner_plus = corner_minus->plus.data();
    run.corner_plus_above = corner_plus->plus.data();
    run.corner_minus = corner_minus->minus.data();
    run.unknown = o.unknown.data() + row;
    run.couples_high = o.couples_high.data() + row;
    run.couples_low = o.couples_low.data() + row;
    run.alpha = alpha_;
    const auto coefficients_of = [&]<beside Edge>(std::size_t from, std::size_t to) {
        coefficient_row<C, Edge>(run, from, to, coefficient_[0].data() + row,
                                 coefficient_[1].data() + row, coefficient_[2].data() + row,
                                 coefficient_[3].data() + row, diagonal + row,
                                 relaxed_.data() + row, source_.data() + row);
    };
    const auto beside_low = [&] {
        run.edge_known = o.low_known.data() + (own_along_row<C> ? 0 : j);
        run.edge_value = o.low_value.data() + (own_along_row<C> ? 0 : j);
    };
    const auto beside_high = [&] {
        run.edge_known = o.high_known.data() + (own_along_row<C> ? 0 : j);
        run.edge_value = o.high_value.data() + (own_along_row<C> ? 0 : j);
    };
    if constexpr (own_along_row<C>) {
        if (j == 0) {
            beside_low();
            coefficients_of.template operator()<beside::low>(first, last);
        } else if (j + 1 == ny_) {
            beside_high();
            coefficients_of.template operator()<beside::high>(first, last);
        } else {
            coefficients_of.template operator()<beside::none>(first, last);
        }
        // The domain-edge faces, where no unknown is and the other
        // component's entries may stand.
        for (double* a : {diagonal, coefficient_[0].data(), coefficient_[1].data(),
                          coefficient_[2].data(), coefficient_[3].data()}) {
            a[row] = 0.0;
            a[row + nx] = 0.0;
        }
    } else {
        beside_low();
        coefficients_of.template operator()<beside::low>(0, 1);
        coefficients_of.template operator()<beside::none>(1, nx - 1);
        beside_high();
        coefficients_of.template operator()<beside::high>(nx - 1, nx);
    }

    source_run rhs{};
    rhs.cell_deferred = cell_plus->deferred.data();
    rhs.cell_deferred_below = cell_minus->deferred.data();
    rhs.corner_deferred = corner_minus->deferred.data();
    rhs.corner_deferred_above = corner_plus->deferred.data();
    rhs.p = p + j * nx;
    rhs.p_below = own_along_row<C> ? rhs.p : rhs.p - nx;
    rhs.width_t = o.width_t.data() + (own_along_row<C> ? j : 0);
    rhs.own = own + row;
    rhs.a_p = diagonal + row;
    if (mu != nullptr) {
        rhs.mu = mu + j * nx;
        rhs.mu_below = own_along_row<C> ? rhs.mu : rhs.mu - nx;
    }
    rhs.mean_high = o.mean_high.data() + (own_along_row<C> ? 0 : j);
    rhs.corner_mu = corner_minus->viscosity.data();
    rhs.corner_mu_above = corner_plus->viscosity.data();
    rhs.cell_stress = cell_plus->stress.data();
    rhs.cell_stress_below = cell_minus->stress.data();
    rhs.corner_stress = corner_minus->stress.data();
    rhs.corner_stress_above = corner_plus->stress.data();
    rhs.relaxation = relaxation_;
    if (mu != nullptr) {
        source_row<C, true>(rhs, first, last, source_.data() + row);
    } else {
        source_row<C, false>(rhs, first, last, source_.data() + row);
    }
}

// Component C's equations. The rows of unknowns are walked once; each row of
// fluxes is formed as the walk reaches it and kept while two rows of unknowns
// read it.
template <int D>
    requires(D == 2)
template <int C>
void predictor<D>::assemble(const std::array<std::span<const double>, D>& velocity,
                            std::span<const double> pressure, std::span<const double> mu_eff,
                            const std::array<std::span<const double>, D>& wall_mu,
                            double* diagonal) {
    const double* mu = mu_eff.empty() ? nullptr : mu_eff.data();
    const double* own = velocity[C].data();
    const double* other = velocity[1 - C].data();
    const double* p = pressure.data();
    const double* wall = wall_mu[C].empty() ? nullptr : wall_mu[C].data();
    if constexpr (own_along_row<C>) {
        corner_row<C>(0, own, other, mu, wall, corner_rows_[0]);
        for (std::size_t j = 0; j < ny_; ++j) {
            flux_row& low = corner_rows_[j % 2];
            flux_row& high = corner_rows_[(j + 1) % 2];
            cell_row<C>(j, own, mu, cell_rows_[0]);
            corner_row<C>(j + 1, own, other, mu, wall, high);
            unknown_row<C>(j, own, p, mu, &cell_rows_[0], &cell_rows_[0], &low, &high,
                           diagonal);
        }
    } else {
        cell_row<C>(0, own, mu, cell_rows_[0]);
        for (std::size_t j = 1; j < ny_; ++j) {
            flux_row& low = cell_rows_[(j - 1) % 2];
            flux_row& high = cell_rows_[j % 2];
            cell_row<C>(j, own, mu, high);
            corner_row<C>(j, own, other, mu, wall, corner_rows_[0]);
            unknown_row<C>(j, own, p, mu, &low, &high, &corner_rows_[0], &corner_rows_[0],
                           diagonal);
        }
        for (double* a : {diagonal, coefficient_[0].data(), coefficient_[1].data(),
                          coefficient_[2].data(), coefficient_[3].data()}) {
            std::fill(a, a + nx_, 0.0);
            std::fill(a + ny_ * nx_, a + (ny_ + 1) * nx_, 0.0);
        }
    }
}

// The under-relaxed Jacobi sweeps of component C's equations from its input,
// `own`, into `star` (`_sweep_n`).
template <int D>
    requires(D == 2)
template <int C>
void predictor<D>::sweep(const double* own, const double* diagonal, double* star) {
    const component_setup& o = setup_[C];
    const std::size_t nx = nx_;
    const std::size_t width = own_along_row<C> ? nx + 1 : nx;
    const std::size_t rows = own_along_row<C> ? ny_ : ny_ + 1;
    const std::size_t first_row = own_along_row<C> ? 0 : 1;
    const std::size_t first = own_along_row<C> ? 1 : 0;
    // The domain-edge faces, which keep the input in every iterate.
    const auto edges = [&](double* to) {
        if constexpr (own_along_row<C>) {
            for (std::size_t j = 0; j < rows; ++j) {
                to[j * width] = own[j * width];
                to[j * width + nx] = own[j * width + nx];
            }
        } else {
            std::copy(own, own + width, to);
            std::copy(own + (rows - 1) * width, own + rows * width, to + (rows - 1) * width);
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
        for (std::size_t j = first_row; j < ny_; ++j) {
            const std::size_t at = j * width;
            sweep_run run{};
            run.x = current + at;
            run.below = j > 0 ? run.x - width : zeros_.data();
            run.above = j + 1 < rows ? run.x + width : zeros_.data();
            run.a_sp = coefficient_[0].data() + at;
            run.a_sm = coefficient_[1].data() + at;
            run.a_tp = coefficient_[2].data() + at;
            run.a_tm = coefficient_[3].data() + at;
            run.b = source_.data() + at;
            run.relaxed = relaxed_.data() + at;
            run.a_p = diagonal + at;
            run.own = own + at;
            if (o.held.empty()) {
                sweep_row<C, false>(run, zeros_.data(), first, nx, next + at);
            } else {
                run.held = o.held.data() + at;
                sweep_row<C, true>(run, zeros_.data(), first, nx, next + at);
            }
        }
        current = next;
    }
}

template <int D>
    requires(D == 2)
void predictor<D>::predict(const std::array<std::span<const double>, D>& velocity,
                           std::span<const double> p, std::span<const double> mu_eff,
                           const std::array<std::span<const double>, D>& wall_mu,
                           const std::array<std::span<double>, D>& star,
                           const std::array<std::span<double>, D>& diagonal) {
    assemble<0>(velocity, p, mu_eff, wall_mu, diagonal[0].data());
    sweep<0>(velocity[0].data(), diagonal[0].data(), star[0].data());
    assemble<1>(velocity, p, mu_eff, wall_mu, diagonal[1].data());
    sweep<1>(velocity[1].data(), diagonal[1].data(), star[1].data());
}

template <int D>
    requires(D == 2)
void predictor<D>::equations(int c, const std::array<std::span<const double>, D>& velocity,
                             std::span<const double> p, std::span<const double> mu_eff,
                             const std::array<std::span<const double>, D>& wall_mu,
                             std::span<double> diagonal) {
    if (c == 0) {
        assemble<0>(velocity, p, mu_eff, wall_mu, diagonal.data());
    } else {
        assemble<1>(velocity, p, mu_eff, wall_mu, diagonal.data());
    }
}

template <int D>
    requires(D == 2)
void predictor<D>::stencil_viscosity(std::span<const double> mu_eff,
                                     const std::array<std::span<double>, D>& out) {
    const auto rows = [&]<int C>(std::size_t first, std::size_t last) {
        const component_setup& o = setup_[C];
        const std::size_t width = nx_ + 1;
        std::fill(out[C].begin(), out[C].end(), mu_);
        double* viscosity = corner_rows_[0].viscosity.data();
        const std::size_t from = own_along_row<C> ? 1 : 0;
        const std::size_t to = own_along_row<C> ? nx_ : nx_ + 1;
        for (std::size_t j = first; j < last; ++j) {
            corner_viscosity<C>(j, mu_eff.data(), viscosity);
            const mask* read = o.wall_read.data() + j * width;
            double* row = out[C].data() + j * width;
            lanes::for_each_block(from, to, [&](auto cells, std::size_t m) {
                read_cells(cells, read, viscosity, mu_, m, row);
            });
        }
    };
    rows.template operator()<0>(0, ny_ + 1);
    rows.template operator()<1>(1, ny_);
}

template class predictor<2>;

}  // namespace cfdcore::momentum
