// The steady solve's outer iteration (ADR-017 C, module `flow`; ADR-012 D,
// ADR-014 A and D).
//
// A port of `StaggeredSolver.solve_steady`'s loop with the fields resident: per
// outer iteration the outlets' extrapolation, the momentum prediction (laminar,
// under a fixed eddy viscosity, or coupled with the wall functions' viscosity),
// the pressure correction and its cap, the k-epsilon step with its positivity
// test and the eddy viscosity's relaxation, the cell-centred velocities, the
// residual, and the stopping rule, in the reference's order of stages. Each
// stage is the module that ports it (`momentum`, `pressure`, `turbulence`,
// `walls`, `stopping`); what this module adds is elementwise, the effective
// viscosity, the cell centres and the max-norms, each the reference's bit for bit
// (ADR-017 G, the first kind). A whole solve is a chain of blocks, judged at
// level 2 (ADR-017 H).
//
// What stays in Python (ADR-017 E): the solver's set-up and its refusals, the
// warm start's and the resume's checks, the normal velocities written before the
// loop, the checkpoints built from the fields and the window at the reference's
// iterations, and the log lines; `tests/core_driver.py` drives the chunks.
//
// Written per direction (ADR-017 decision 4): component c of the velocity lies on
// the faces normal to direction c, and its cell centre is the mean of its two
// faces along c. Every field is walked by its rows. Instantiated for D = 2 and, in
// the core's own tests, 3.

#pragma once

#include "cfdcore/grid.hpp"
#include "cfdcore/lanes.hpp"
#include "cfdcore/layout.hpp"
#include "cfdcore/momentum.hpp"
#include "cfdcore/pressure.hpp"
#include "cfdcore/stopping.hpp"
#include "cfdcore/turbulence.hpp"
#include "cfdcore/walls.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace cfdcore::flow {

// The cell centres of one component (`to_cell_centers`): `0.5 (low + high)` of
// each cell's two faces along direction c.
//
// fields:   the cells' layout
// c:        the component
// velocity: the component's faces, m/s
// out:      output, a cell field, overlapping no input
template <int D>
void cell_centres(const layout::layout<D>& fields, int c, std::span<const double> velocity,
                  std::span<double> out);

// The largest absolute change of a cell field over some cells (`np.max(np.abs(a[m]
// - b[m]))`): exact in any order, and NaN where one of the changes read is NaN.
//
// now, before: two cell fields; cells: the cells read, as masks, one per cell, at
// least one set
double largest_change(std::span<const double> now, std::span<const double> before,
                      std::span<const lanes::mask> cells);

// The coupled prediction's effective viscosity (`_effective_viscosity`): `mu + rho
// np.where(live, nu_t, 0.0)` per cell, after its two refusals in its order, a live
// value that is not finite and then a negative one. Returns false, and writes
// nothing, where it refuses, and `not_finite` says which.
//
// nu_t: the kinematic eddy viscosity, m^2/s, a cell field
// live: the non-SOLID cells, as masks, one per cell
// mu, rho: air's viscosity, Pa s, and density, kg/m^3
// out: output, Pa s, a cell field, overlapping no input
bool effective_viscosity(std::span<const double> nu_t, std::span<const lanes::mask> live,
                         double mu, double rho, std::span<double> out, bool& not_finite);

// How the prediction takes its viscosity: air's (laminar), a field held for the
// solve (`eddy_viscosity=`), or the k-epsilon model's iterate with the wall
// functions (the turbulence section).
enum class viscosity_mode { laminar, fixed, coupled };

// Which rule stops the solve (`solver.stopping_rule`).
enum class stop_rule { velocity_step, error_estimate };

// What a solver is built from besides its grid and its modules.
//
// fluid:            the type of a FLUID cell (`mesh.FLUID`), whose cells the residual
//                   and condition (e)'s median read
// mu, rho, nu:      air's viscosity, Pa s, density, kg/m^3, and kinematic viscosity,
//                   m^2/s, as the reference forms it (`mu / rho`)
// alpha_turbulence: the eddy viscosity's relaxation, read when coupled
// residual_scale:   what the residual divides the velocity step by, m/s:
//                   `max(reference_velocity, ZERO_SCALE)`
// rule:             the stop; `convergence_tol` the velocity step's bound, `keys`
//                   the error estimate's
// mode:             the viscosity; `mu_eff` the field held for a fixed one, Pa s,
//                   `mu + rho nu_t` as the reference forms it once per solve
template <int D>
struct flow_input {
    std::int32_t fluid = 0;
    double mu = 0.0;
    double rho = 0.0;
    double nu = 0.0;
    double alpha_turbulence = 0.0;
    double residual_scale = 0.0;
    stop_rule rule = stop_rule::velocity_step;
    double convergence_tol = 0.0;
    stopping::rule_keys keys{};
    viscosity_mode mode = viscosity_mode::laminar;
    std::span<const double> mu_eff{};
};

// The modules a solver calls, which must outlive it and be built on its grid; the
// turbulence model and its boundary only for the coupled solve.
template <int D>
struct modules {
    momentum::predictor<D>* predictor = nullptr;
    pressure::corrector<D>* corrector = nullptr;
    const walls::outlets<D>* outlets = nullptr;
    turbulence::model<D>* model = nullptr;
    walls::boundary<D>* walls = nullptr;
};

// Why an outer iteration ended the chunk short of its stop, as the reference raises
// it: the eddy viscosity not finite or negative in a non-SOLID cell
// (`_effective_viscosity`), the effective or the wall viscosity refused by the
// predictor (`_check_mu_eff`, `_check_wall_mu`), the pressure solve's own refusals,
// the faces or the conditions refused by the k-epsilon step (`_check_faces`,
// `_check_conditions`), a field at rest, k or eps not positive after the step
// (PositivityError), or condition (e)'s scale not positive and finite.
enum class failure {
    none,
    eddy_viscosity_not_finite,
    eddy_viscosity_negative,
    mu_eff_not_finite,
    mu_eff_not_positive,
    wall_mu_not_finite,
    wall_mu_not_positive,
    pressure,
    faces_not_finite,
    conditions,
    nothing_moves,
    k_not_positive,
    eps_not_positive,
    nu_scale
};

// A failure and what its message names: the outer iteration, the component of a
// wall viscosity, the pressure solve's status, the conditions' check, the bad
// cells of k or eps, condition (e)'s scale.
struct chunk_failure {
    failure kind = failure::none;
    std::int64_t iteration = 0;
    int component = 0;
    pressure::solve_status pressure = pressure::solve_status::ok;
    turbulence::condition_check conditions{};
    turbulence::bad_cells bad{};
    double nu_scale = 0.0;
};

// One outer iteration as the reference records it: the residual, the velocity step
// `max(du, dv)` (m/s) and with the model the viscosity step (m^2/s; NaN without),
// the correction's iterations and products and whether it reached its cap, and the
// rule's readings (NaN under velocity_step).
struct iteration_record {
    double residual = 0.0;
    double step = 0.0;
    double viscosity_step = 0.0;
    std::int64_t pressure_iterations = 0;
    std::int64_t pressure_products = 0;
    bool reached_cap = false;
    stopping::reading rule{};
};

// The wall time of a chunk's stages, s: "momentum" (the outlets and the
// prediction), "pressure", "turbulence", "correct" (the cell centres and the
// residual), as `stage_seconds` names them, and the rule's, which the reference
// counts in none.
struct stage_seconds {
    double momentum = 0.0;
    double pressure = 0.0;
    double turbulence = 0.0;
    double correct = 0.0;
    double stopping = 0.0;
};

// How a chunk ended: whether the rule stopped it, the outer iterations it completed
// (the failing one not among them), the failure, and the stages' seconds. A failing
// iteration's record holds its correction's numbers, which the reference counts
// before it raises.
struct chunk_result {
    bool stopped = false;
    std::int64_t outer = 0;
    chunk_failure failure{};
    stage_seconds seconds{};
};

// The steady solve on one grid with its fields resident. Every array is allocated
// at construction; a chunk allocates nothing.
template <int D>
    requires(D >= 1)
class solver {
public:
    // Why the modules and the input cannot make a solver on `g`, or an empty view
    // when they can: a module missing for the mode or on another grid, a fixed
    // viscosity of another length, a density, viscosity or scale that is not finite
    // and positive, a relaxation outside (0, 1], a velocity step's bound that is not
    // finite and positive, error-estimate keys `stopping::rule::check` refuses, or no
    // FLUID cell.
    static std::string_view check(const grid::grid<D>& g, const modules<D>& m,
                                  const flow_input<D>& input);

    // Builds the solver from what `check` accepts, copying the fixed viscosity.
    // Throws std::invalid_argument for what it does not.
    solver(const grid::grid<D>& g, const modules<D>& m, const flow_input<D>& input);

    // Begins a solve (`flow_state`): copies the faces (their normal velocities
    // already written), the pressure and, when coupled, k, eps and the relaxed nu_t;
    // forms the cell centres the first step is measured from; and starts the rule
    // fresh. Throws std::invalid_argument for a field of another length.
    void start(const std::array<std::span<const double>, D>& velocity,
               std::span<const double> p, std::span<const double> k,
               std::span<const double> eps, std::span<const double> nu_t);

    // Seeds the fresh rule with a checkpoint's window (`ErrorEstimateRule.restore`);
    // ok, and nothing seeded, under velocity_step with both windows empty.
    stopping::restore_status restore(std::span<const double> steps,
                                     std::span<const double> viscosity_steps);

    // At most `records.size()` outer iterations, the first numbered `first`: each
    // one's record into `records`, in order, up to the stop, the end of `records`,
    // or a failure.
    chunk_result run(std::int64_t first, std::span<iteration_record> records);

    // The resident fields: the faces of component c, p, and when coupled k, eps and
    // the relaxed nu_t (empty otherwise).
    std::span<const double> velocity(int c) const { return velocity_[c]; }
    std::span<const double> pressure() const { return p_; }
    std::span<const double> k() const { return k_; }
    std::span<const double> eps() const { return eps_; }
    std::span<const double> nu_t() const { return nu_t_; }

    // The rule, under error_estimate; null under velocity_step.
    const stopping::rule* rule() const { return rule_ ? &*rule_ : nullptr; }

    const grid::grid<D>& mesh() const { return *grid_; }

private:
    bool coupled() const { return mode_ == viscosity_mode::coupled; }
    std::array<std::span<const double>, D> const_velocity() const;
    bool predict(chunk_failure& failure);
    bool correct(iteration_record& record, chunk_failure& failure);
    bool turbulence_step(iteration_record& record, chunk_failure& failure);
    void residual(iteration_record& record);
    bool stop(iteration_record& record, chunk_failure& failure);

    const grid::grid<D>* grid_;
    modules<D> modules_;
    viscosity_mode mode_;
    stop_rule stop_rule_;
    double mu_;
    double rho_;
    double nu_;
    double alpha_turbulence_;
    double residual_scale_;
    double convergence_tol_;
    std::optional<stopping::rule> rule_;  // under error_estimate alone

    // Fixed on the grid.
    std::vector<lanes::mask> fluid_;  // set at FLUID cells
    std::vector<lanes::mask> live_;   // set where not SOLID
    std::vector<std::size_t> fluid_cells_;
    std::vector<double> fixed_mu_;

    // The resident fields.
    std::array<std::vector<double>, D> velocity_;
    std::vector<double> p_;
    std::vector<double> k_;
    std::vector<double> eps_;
    std::vector<double> nu_t_;
    std::array<std::vector<double>, D> centres_;
    std::array<std::vector<double>, D> previous_;

    // An outer iteration's scratch.
    std::array<std::vector<double>, D> star_;
    std::array<std::vector<double>, D> diagonal_;
    std::array<std::vector<double>, D> corrected_;
    std::vector<double> p_next_;
    std::vector<double> p_prime_;
    std::vector<double> mu_eff_;
    momentum::per_pair<std::vector<double>, D> stencil_mu_;
    momentum::per_pair<std::vector<double>, D> wall_mu_;
    std::vector<double> k_next_;
    std::vector<double> eps_next_;
    std::vector<double> nu_t_step_;
    std::vector<double> nu_t_next_;
    std::vector<double> imbalance_;
    std::vector<double> magnitude_;
    std::vector<double> median_;
};

extern template class solver<2>;

}  // namespace cfdcore::flow
