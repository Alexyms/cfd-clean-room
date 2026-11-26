// The k-epsilon model on the staggered grid (ADR-017 C, module `turbulence`;
// ADR-012 A, C and D, ADR-014 J).
//
// A port of `src/turbulence.py`'s `KEpsilonModel.step` on the steady path, one
// local pseudo-time step of k and eps on prescribed face velocities, written to
// equal it block by block: every elementwise value by the reference's operations
// in the reference's order, one rounding each, and the two implicit solves by
// `scalar::implicit_step` with the conjugate gradients and their fallback
// (REQ-N03 level 1, ADR-017 G). RNG's cube of eta goes to the C library's pow
// (elementary.hpp). The true-time step of VAL-015 is tier 2's and is not here
// (ADR-017 F).
//
// Written per direction (ADR-017 decision 4). Component c of the velocity lies
// on the faces normal to direction c. The strain's diagonal is a term per
// direction and its shear a term per pair of directions, and what makes a
// shear, the gradient at the corners of component c along a direction t, is
// formed per ordered pair (c, t), read as [c][q] with q the pair's slot
// (`momentum::per_pair`). Corners of a pair are n + 1 along both its directions
// and n along the others. Every field is walked by its rows.
//
// Instantiated for D = 2 and, in the core's own tests, 1 and 3.

#pragma once

#include "cfdcore/grid.hpp"
#include "cfdcore/lanes.hpp"
#include "cfdcore/layout.hpp"
#include "cfdcore/momentum.hpp"
#include "cfdcore/scalar.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace cfdcore::turbulence {

// The constants of one variant (`VariantConstants`), dimensionless. eta_0 and
// beta are read only where `has_r` says the variant has RNG's R.
struct constants {
    double c_mu = 0.0;
    double c_1 = 0.0;
    double c_2 = 0.0;
    double sigma_k = 0.0;
    double sigma_e = 0.0;
    bool has_r = false;
    double eta_0 = 0.0;
    double beta = 0.0;
};

// What a model is built from besides its grid, per direction (x, then y).
//
// cell_widths: the n_d cell widths, m (`Mesh.dx_cell`, `dy_cell`)
// face_widths: the n_d + 1 distances across the faces, m (`Mesh.dx_face`, `dy_face`)
// vents:       the vent faces normal to the direction (ADR-014 J;
//              `VentFaces.normal_u`, `normal_v`), a face field; empty for none
//              in any direction, and then empty in every one
// variant:     the variant's constants
// cfl:         the turbulence Courant number (`turbulence.cfl_number`)
// upwind:      advect k and eps by upwind, else by the UMIST-limited QUICK value
// max_iter, tol: the implicit solves' cap and tolerance
// nu:          air's kinematic viscosity, m^2/s
template <int D>
struct model_input {
    std::array<std::span<const double>, D> cell_widths;
    std::array<std::span<const double>, D> face_widths;
    std::array<std::span<const bool>, D> vents;
    constants variant;
    double cfl = 0.0;
    bool upwind = true;
    std::int64_t max_iter = 0;
    double tol = 0.0;
    double nu = 0.0;
};

// The boundary values of one step (`TurbulenceConditions`).
//
// inflow_k, inflow_eps: per direction, k (m^2/s^2) and eps (m^2/s^3) carried in
//                  through the faces normal to it, face fields, read at the
//                  domain's faces and the vents'
// low, high:       per ordered pair (c, t), component c's tangential velocity on
//                  the low and the high domain edge along t, m/s, one entry per
//                  face position of the component on the edge (the edge's other
//                  directions, n_c + 1 along c)
// eps_held:        the cells whose eps is held; eps_wall its value there
// production_given: the cells whose production is given; production its value
//                  there, m^2/s^3. Cell fields, none of them in a SOLID cell
template <int D>
struct conditions {
    std::array<std::span<const double>, D> inflow_k;
    std::array<std::span<const double>, D> inflow_eps;
    momentum::per_pair<std::span<const double>, D> low;
    momentum::per_pair<std::span<const double>, D> high;
    std::span<const bool> eps_held;
    std::span<const double> eps_wall;
    std::span<const bool> production_given;
    std::span<const double> production;
};

// The iterate a step reads (`TurbulenceState`): k, eps and the kinematic eddy
// viscosity, m^2/s, cell fields, zero in SOLID cells.
struct state {
    std::span<const double> k;
    std::span<const double> eps;
    std::span<const double> nu_t;
};

// How a step ended: `nothing_moves` is the reference's refusal of a field whose
// pseudo-time step has no value, the others its PositivityError (REQ-S15).
enum class step_status { ok, nothing_moves, k_not_positive, eps_not_positive };

// The cells where a quantity is not positive and finite: how many, the first
// one's index and value, and the least value over the non-SOLID cells (NaN where
// one of them is).
struct bad_cells {
    std::size_t count = 0;
    std::size_t first = 0;
    double value = 0.0;
    double least = 0.0;
};

// What a step returns beside its fields: how it ended, the two solves' results
// (k's, then eps's) and the bad cells of the quantity the status names.
struct step_result {
    step_status status = step_status::ok;
    std::array<scalar::implicit_result, 2> solves{};
    bad_cells bad{};
};

// The quantity a block is asked about.
enum class quantity { k, eps };

// Why a step refuses its conditions (`_check_conditions`), in the reference's
// order: an array that is not finite (`array` its place in that order: the inflow
// k and eps per direction, x first; the tangential velocities per ordered pair,
// low edge then high; eps_wall; production), an inflow k or eps not positive on a
// domain face or on a vent's, eps_wall not positive where eps is held, or the
// production negative where it is given.
enum class condition_status {
    ok,
    not_finite,
    inflow_on_domain,
    inflow_on_vents,
    eps_wall,
    production
};
struct condition_check {
    condition_status status = condition_status::ok;
    int array = 0;
};

// Whether every value of an array is finite.
bool all_finite(std::span<const double> values);

// The eddy viscosity under-relaxed (`StaggeredSolver._turbulence_step`):
// `(1 - alpha) old + alpha stepped` per cell, with `1 - alpha` formed once, as the
// reference forms it from a Python float.
//
// old:     the relaxed nu_t the step read, m^2/s, a cell field
// stepped: the step's own nu_t, m^2/s, as long
// alpha:   `turbulence.alpha_turbulence`, dimensionless
// out:     output, as long, overlapping no input
//
// Throws std::invalid_argument for arrays of different lengths.
void relax(std::span<const double> old, std::span<const double> stepped, double alpha,
           std::span<double> out);

// The model on one grid, which must outlive it. Every array is allocated at
// construction; a call allocates nothing.
template <int D>
    requires(D >= 1)
class model {
public:
    // Why `input` cannot make a model on `g`, or an empty view when it can: an
    // array whose length is not the grid's, a width that is not finite and
    // positive, vents given for some directions and not for others, or a cfl, a
    // nu, a tol, a variant constant that is not finite and positive, or a
    // max_iter below one.
    static std::string_view check(const grid::grid<D>& g, const model_input<D>& input);

    // Builds the model from an input `check` accepts, copying every array. Throws
    // std::invalid_argument for one it does not. Every call below throws it for
    // an array whose length is not the grid's.
    model(const grid::grid<D>& g, const model_input<D>& input);

    // Whether a cell field is finite and positive (`positive`) or not negative
    // in every non-SOLID cell, and whether it is zero in every SOLID one.
    bool in_range(std::span<const double> field, bool positive) const;
    bool zero_in_solid(std::span<const double> field) const;

    // `C_mu k^2 / eps` per cell, m^2/s, zero in SOLID cells (`eddy_viscosity`).
    //
    // k, eps: cell fields; out: a cell field, overlapping no input
    void eddy_viscosity(std::span<const double> k, std::span<const double> eps,
                        std::span<double> out) const;

    // The gradient at a pair's corners of component c along the direction of its
    // slot q, 1/s (`_corner_difference`): du/dy for c = 0 and dv/dx for c = 1 of
    // D = 2. A domain edge's tangential velocity sits at the edge; a face that
    // carries nothing (beside a SOLID cell, not a vent) reads as zero at the
    // corner itself, and a gradient between two such is zero.
    //
    // velocity: the component's faces, m/s; low, high: the edges' tangential
    // velocities of the pair; out: the pair's corners, overlapping no input
    void corner_gradient(int c, int q, std::span<const double> velocity,
                         std::span<const double> low, std::span<const double> high,
                         std::span<double> out);

    // `S^2 = 2 S_ij S_ij` per cell, 1/s^2, zero in SOLID cells
    // (`strain_squared`), from the component's faces and the edges' tangential
    // velocities of `c`.
    //
    // velocity: the components' faces, m/s; out: a cell field, overlapping no
    // input
    void strain_squared(const std::array<std::span<const double>, D>& velocity,
                        const conditions<D>& c, std::span<double> out);

    // Step 2's growth and step 3's decay (`_terms`) into `production()`,
    // `rng_r()`, `growth_eps()`, `decay_k()` and `decay_eps()`, cell fields that
    // stand until the next call of it or of `step`; k's growth is the
    // production. rng_r is written for a variant with R and zero otherwise.
    void terms(const state& s, const std::array<std::span<const double>, D>& velocity,
               const conditions<D>& c);
    std::span<const double> production() const { return production_; }
    std::span<const double> rng_r() const { return rng_r_; }
    std::span<const double> growth_eps() const { return growth_eps_; }
    std::span<const double> decay_k() const { return ratio_; }
    std::span<const double> decay_eps() const { return decay_eps_; }

    // The local pseudo-time step per cell, s (`pseudo_time_step`), into
    // `time_step()`. False where no non-SOLID cell moves: then it has no value.
    bool pseudo_time_step(const std::array<std::span<const double>, D>& velocity);
    std::span<const double> time_step() const { return dt_; }

    // Step 1 of one quantity (`_advect`): q less dt times the divergence of its
    // advective flux over the volume, with the pseudo-time step of the last
    // `pseudo_time_step` and the flux of that call's velocities. SOLID cells are
    // not masked, as the reference's are not there.
    //
    // q, inflow: the field and its inflow values per direction; out: a cell
    // field, overlapping no input
    void advected(std::span<const double> q,
                  const std::array<std::span<const double>, D>& inflow, std::span<double> out);

    // Steps 1 and 2: the explicit fields of k and eps, zero in SOLID cells and
    // eps held where the conditions hold it, from the last `pseudo_time_step`
    // and `terms`.
    //
    // k_star, eps_star: cell fields, each overlapping no input and the other
    void explicit_fields(const state& s, const conditions<D>& c, std::span<double> k_star,
                         std::span<double> eps_star);

    // `(nu + nu_t / sigma) A / d` on the faces normal to each direction between
    // two non-SOLID cells, m^2/s, zero elsewhere (`_conductances`), of the
    // quantity's sigma.
    //
    // nu_t: a cell field, m^2/s; out: per direction, the faces normal to it,
    // each overlapping no input and the others
    void conductances(quantity which, std::span<const double> nu_t,
                      const std::array<std::span<double>, D>& out);

    // One step (`step` with dt None): k, eps and nu_t after it into the three
    // outputs, cell fields overlapping no input. Where the status is not ok they
    // hold nothing of use. The state, the velocities and the conditions are the
    // caller's to have checked (the reference's `_check_*`).
    step_result step(const state& s, const std::array<std::span<const double>, D>& velocity,
                     const conditions<D>& c, std::span<double> k, std::span<double> eps,
                     std::span<double> nu_t);

    // The checks of a step's conditions whose arrays are of the grid's lengths:
    // their ranges alone (`ranges`), or every value finite first and then the
    // ranges (`formed`), what conditions a boundary formed can meet.
    condition_check check_ranges(const conditions<D>& c) const;
    condition_check check_formed(const conditions<D>& c) const;

    // The positivity test of a step's field (`_check_positive`): false where the
    // quantity is positive and finite in every non-SOLID cell; true otherwise, with
    // the bad cells counted, the first one's index and value, and the least value
    // read over the non-SOLID cells alone (NaN where one of them is), as the
    // reference's `np.min(q[live])`.
    //
    // q: a cell field; bad: output
    bool find_bad(std::span<const double> q, bad_cells& bad) const;

    // The cells' volumes, m^2 per unit depth, one per cell (`_volume`).
    const std::vector<double>& volume() const { return volume_; }

    // Face masks: the faces normal to direction d that carry an advective flux
    // (the vents among them), and those diffusion crosses.
    std::span<const lanes::mask> live_faces(int d) const { return live_[d]; }
    std::span<const lanes::mask> inner_faces(int d) const { return inner_[d]; }

    const grid::grid<D>& mesh() const { return *grid_; }

private:
    // One pair's fixed arrays, in corners: the distance a gradient divides by
    // (1 where it is not defined) and where it is defined.
    struct pair_setup {
        std::vector<double> span;
        std::vector<lanes::mask> defined;
    };

    void require_velocity(const std::array<std::span<const double>, D>& velocity) const;
    void require_conditions(const conditions<D>& c) const;
    void require_state(const state& s) const;
    void carry(const std::array<std::span<const double>, D>& velocity);
    template <bool Add>
    void gradient(int c, int q, const double* faces, std::span<const double> low,
                  std::span<const double> high, double* out);
    template <int C, int T, bool Add>
    void gradient_rows(const double* faces, std::span<const double> low,
                       std::span<const double> high, double* out);
    void strain(const conditions<D>& c, double* out);
    void form_terms(const state& s, const conditions<D>& c);
    bool form_time_step();
    void advective_fluxes(std::span<const double> q,
                          const std::array<std::span<const double>, D>& inflow);
    void explicit_field(std::span<const double> q,
                        const std::array<std::span<const double>, D>& inflow,
                        const double* growth, const lanes::mask* held, const double* held_value,
                        double* out);
    scalar::implicit_result solve(std::span<const double> star, std::span<const double> decay,
                                  std::span<const bool> held, std::span<double> out);

    const grid::grid<D>* grid_;
    scalar::scheme<D> scheme_;
    scalar::implicit_scratch<D> solver_;
    constants variant_;
    double cfl_;
    bool upwind_;
    std::int64_t max_iter_;
    double tol_;
    double nu_;

    // Fixed on the grid.
    std::array<layout::layout<D>, D> faces_;
    std::vector<layout::layout<D>> corners_;   // by unordered pair, c < t
    std::vector<std::vector<pair_setup>> pairs_;  // [c][q]
    std::array<std::vector<double>, D> cell_widths_;
    std::array<std::vector<double>, D> face_widths_;
    std::array<std::vector<double>, D> far_share_;  // of the harmonic mean, per face
    std::array<std::vector<double>, D> area_;       // face fields, m
    std::vector<double> volume_;
    std::vector<lanes::mask> fluid_;  // set where not SOLID
    std::unique_ptr<bool[]> solid_;
    std::array<std::vector<lanes::mask>, D> live_;
    std::array<std::vector<lanes::mask>, D> inner_;
    std::array<std::unique_ptr<bool[]>, D> vents_;
    bool vented_ = false;
    // Per direction, the domain's faces and the vents' (interior faces with a flux
    // that diffusion does not cross), whose inflow values the checks read.
    std::array<std::vector<std::size_t>, D> domain_faces_;
    std::array<std::vector<std::size_t>, D> vent_faces_;

    // A step's fields.
    std::array<std::vector<double>, D> carried_;  // the velocities of the live faces
    std::array<std::vector<double>, D> flux_;     // volume flux
    std::array<std::vector<double>, D> advective_;
    std::array<std::vector<double>, D> conductance_;
    std::vector<std::vector<double>> shear_;  // by unordered pair
    std::vector<double> strain_;
    std::vector<double> production_;
    std::vector<double> ratio_;
    std::vector<double> eta_;
    std::vector<double> cube_;
    std::vector<double> rng_r_;
    std::vector<double> growth_eps_;
    std::vector<double> decay_eps_;
    std::vector<double> dt_;
    std::vector<double> gamma_;
    std::vector<double> sink_;
    std::vector<double> k_star_;
    std::vector<double> eps_star_;
    std::vector<lanes::mask> held_;
    std::vector<lanes::mask> given_;
};

extern template class model<1>;
extern template class model<2>;

}  // namespace cfdcore::turbulence
