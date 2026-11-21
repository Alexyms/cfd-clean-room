// One particle class advanced one time step, and its budget (ADR-017 C, module
// `transport`; ADR-011 B to F, ADR-012 F, ADR-014 J, ADR-016).
//
// A port of `src/solver_transport.py`'s `TransportSolver.solve_timestep` and the
// helpers it calls, with `MassBudget`'s booking and ADR-016's breach test,
// written to equal them bit for bit (REQ-N03 level 1, ADR-017 G): the field,
// every budget term, the sweeps and the leftover. Every value is formed by the
// reference's operations in the reference's order, one rounding each, and every
// sum the budget, the bound or the breach test reads is formed in NumPy's order
// on the array the reference sums: `np.sum` through `reduce::pairwise_sum`, a
// sum over a strided slice through `reduce::buffered_sum`, and Python's own `+`
// left to right in the order written.
//
// The reference recomputes every class step what an interval fixes: the input
// checks, `stable_dt`, the advecting velocities and their volume fluxes, the
// face eddy viscosity, the conductances, the deposition per face and the masks
// the booking reads. Here `set_interval` and `set_step` compute them once, when
// an interval is set up (ADR-017 C), which changes no value; `step` does what
// changes from one step to the next.
//
// Direction D - 1 is the vertical, the one gravity acts along: settling is
// subtracted from the velocity normal to its faces (y for D = 2, the
// reference's v faces). A cell field and a face field are laid out as
// layout.hpp says; a face field normal to d is "faces(d)" below.
//
// The budget's slots are MassBudget's `deposited` keys in DEPOSIT_SURFACES'
// order: the floor, the ceiling, the walls, the obstacles.

#pragma once

#include "cfdcore/grid.hpp"
#include "cfdcore/lanes.hpp"
#include "cfdcore/layout.hpp"
#include "cfdcore/scalar.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace cfdcore::transport {

// The deposition slots, in DEPOSIT_SURFACES' order.
inline constexpr std::size_t floor_slot = 0;
inline constexpr std::size_t ceiling_slot = 1;
inline constexpr std::size_t wall_slot = 2;
inline constexpr std::size_t obstacle_slot = 3;
inline constexpr std::size_t slots = 4;
// The slots' names, MassBudget's `deposited` keys: the names a budget's terms
// are handed out and taken in by, so that the binding knows nothing of the
// reference's module. The parity module checks them against DEPOSIT_SURFACES.
inline constexpr std::array<std::string_view, slots> slot_names = {"floor", "ceiling", "wall",
                                                                    "obstacle"};

// The solver's keys (the configuration's `transport` section and `t_end`).
//
// cfl:           `cfl_number`, the Courant number `stable_dt` is set at
// upwind:        first-order upwind instead of the UMIST-limited QUICK value
// max_sweeps:    `max_diffusion_iter`, the Jacobi cap
// tol:           `diffusion_tol`, the max-norm stop, dimensionless
// limit:         `conservation_limit`, REQ-T05's bound on the accumulated
//                leftover as a fraction of the particles supplied
// share:         `leftover_share`, the fraction of the limit a step's solve gets
// horizon:       `t_end`, s, the time the share is spread over
// schmidt:       `turbulent_schmidt`, or none; an eddy viscosity needs it
// surface_codes: the surface codes (`boundary_concentration.SURFACE_*`) booked
//                to the floor, the ceiling and the wall slots, in that order,
//                so the core holds no copy of the boundary layer's codes
struct settings {
    double cfl = 0.0;
    bool upwind = false;
    std::int64_t max_sweeps = 0;
    double tol = 0.0;
    double limit = 0.0;
    double share = 0.0;
    double horizon = 0.0;
    std::optional<double> schmidt;
    std::array<std::int32_t, 3> surface_codes{};
};

// The mesh's spacings per direction d, direction 0 first (`Mesh`'s arrays).
//
// widths:         the n_d cell widths, m (`dx_cell`, `dy_cell`)
// face_distances: the n_d + 1 distances between the nodes either side of each
//                 face, m (`dx_face`, `dy_face`)
template <int D>
struct geometry {
    std::array<std::span<const double>, D> widths;
    std::array<std::span<const double>, D> face_distances;
};

// One class's physics: the settling velocity, m/s, positive downward, and the
// Brownian diffusion coefficient, m^2/s (`settling_velocity`, `diffusion_coeff`).
struct class_physics {
    double settling = 0.0;
    double diffusion = 0.0;
};

// A discharge that returns what its intake draws (`Recirculation`).
//
// intake, discharge: per direction d, one flag per face of faces(d)
// passed:            the fraction of the drawn concentration carried, in [0, 1]
template <int D>
struct recirculation {
    std::array<std::span<const bool>, D> intake;
    std::array<std::span<const bool>, D> discharge;
    double passed = 0.0;
};

// One class's conditions on every face (`ConcentrationFaces`).
//
// inflow:        per d, the concentration an inward flux carries, particles
//                per m^3, one per face of faces(d), finite and non-negative
// deposition:    per d, the deposition velocity, m/s, one per face of
//                faces(d), finite and non-negative
// surface:       per d, the surface code each depositing face books to
// settling:      one flag per face of faces(D - 1): where the class's settling
//                velocity is subtracted
// vents:         per d, one flag per face of faces(d) at an equipment vent, or
//                all empty for none; class 0's are every class's
// recirculation: the discharges that return their intake's air, in order
template <int D>
struct conditions {
    std::array<std::span<const double>, D> inflow;
    std::array<std::span<const double>, D> deposition;
    std::array<std::span<const std::int32_t>, D> surface;
    std::span<const bool> settling;
    std::array<std::span<const bool>, D> vents;
    std::vector<recirculation<D>> recirculations;
};

// What an interval holds fixed beside the conditions.
//
// faces:          per d, the air's velocity normal to faces(d), m/s, finite
// drift:          one per class, or empty for none: per d, the class's drift
//                 increment on faces(d), m/s (`v_ext`, ADR-007), all spans
//                 empty for none; acts on the interior faces between two
//                 non-SOLID cells
// eddy_viscosity: nu_t per cell, m^2/s, finite and non-negative in every
//                 non-SOLID cell (others not read), or empty for the laminar
//                 path
// sources:        one per class, or empty for none: the rate per cell,
//                 particles per m^3 per s, finite, non-negative and zero in
//                 SOLID cells, or an empty span for none
template <int D>
struct interval_input {
    std::array<std::span<const double>, D> faces;
    std::vector<std::array<std::span<const double>, D>> drift;
    std::span<const double> eddy_viscosity;
    std::vector<std::span<const double>> sources;
};

// One class's particle count as the solver books it (`MassBudget`), particles
// per metre of depth. `initial` and `current` are none before the first step.
struct budget {
    std::optional<double> initial;
    double inflow = 0.0;
    double outflow = 0.0;
    double source = 0.0;
    std::array<double, slots> deposited{};
    std::optional<double> current;
    double leftover = 0.0;
};

// Why an input was refused: the conditions a run can meet, each the
// reference's ValueError (or, for a breach, its ConservationBreachError). A
// caller's programming error that no run can meet (an array of the wrong
// length, a class out of range, a step before an interval) is a
// std::invalid_argument instead.
enum class status {
    ok,
    condition_not_finite_or_negative,  // a class's inflow or deposition array, by name
    recirculation_without_vents,       // a class names a recirculation and no vent faces
    passed_out_of_range,               // a recirculation's fraction is not in [0, 1]
    vents_differ,                      // a class names other vent faces than class 0
    faces_not_finite,                  // the air's velocity
    drift_not_finite,                  // a class's drift
    sources_not_finite,
    sources_negative,
    source_in_solid,
    eddy_viscosity_without_schmidt,
    eddy_viscosity_not_finite,
    eddy_viscosity_negative,
    dt_not_finite,
    dt_not_positive,
    field_not_finite,                  // the field a step is handed
    dt_above_stable,                   // the step exceeds the class's stable_dt
    bound_negative_or_nan,             // the step's leftover bound, from a negative supply
    breach,                            // the accumulated leftover reached the limit
};

// A refusal: its status, the class it concerns (-1 for none), the name of the
// array it concerns (`inflow_u` and the like, or null) and the number it
// read (a fraction, a step or a stable step), for the binding's message.
struct refusal {
    status code = status::ok;
    int size_class = -1;
    const char* name = nullptr;
    double value = 0.0;
    double other = 0.0;
    bool ok() const { return code == status::ok; }
};

// What a step records beside its field and budget (TransportSolver's
// `last_diffusion_sweeps`, `diffusion_converged`, `last_leftover` and
// `last_leftover_bound`), and on a breach the error's arguments: the class's
// accumulated leftover and its particles supplied.
struct step_record {
    std::int64_t sweeps = 0;
    bool converged = true;
    double leftover = 0.0;
    double bound = 0.0;
    double accumulated = 0.0;
    double supplied = 0.0;
};

template <int D>
    requires(D >= 1)
class solver {
public:
    // The solver on a scheme's grid (which must outlive it), with the mesh's
    // spacings, the keys and each class's physics; one budget per class, zero.
    // Allocates every array the solver holds. Throws std::invalid_argument
    // when a spacing array is not as long as the grid needs, or there is no
    // class.
    solver(const scalar::scheme<D>& scheme, const geometry<D>& spacing, const settings& keys,
           std::vector<class_physics> physics);
    ~solver();
    solver(const solver&) = delete;
    solver& operator=(const solver&) = delete;

    // The number of classes.
    std::size_t classes() const { return physics_.size(); }

    // An interval's set-up: the conditions of every class, in class order, and
    // what the interval holds fixed. Refuses, before any value is kept, what
    // the reference refuses: the conditions' values first (class by class),
    // then the air's faces, each class's drift, each class's sources, then the
    // eddy viscosity. Then computes, per class, the advecting velocities, their
    // volume fluxes, `stable_dt`, the deposition per face and per cell, the
    // conductances, the recirculating intakes' drawn flux, the source's total,
    // and the masks the booking reads. The budgets are kept. Throws
    // std::invalid_argument for an array of the wrong length or a count of
    // conditions, drifts or sources other than the class count (an empty
    // vector of drifts or sources is none for every class).
    refusal set_interval(const std::vector<conditions<D>>& each, const interval_input<D>& input);

    // The interval's step, s: finite and positive. Computes what the step
    // fixes: the deposition per face times dt and dt times each source rate.
    // Throws std::invalid_argument before an interval is set up.
    refusal set_step(double dt);

    // The largest step the class's explicit advection is stable at, s
    // (`stable_dt`), on the interval set up; infinite when nothing moves.
    double stable_dt(std::size_t k) const;

    // One step of class k (`solve_timestep`): the field `c` (a cell field,
    // particles per m^3, finite) advanced by the interval's step into `out` (a
    // cell field), the budget booked, the step's records written. Refuses a
    // field that is not finite and a step above the class's stable_dt before
    // any arithmetic, leaving the budget untouched. A breach is refused after
    // the step: the budget holds it and `out` the field. `out` may be `c`
    // itself, read only before `out` is written; it must overlap no other
    // array the solver reads. Throws std::invalid_argument for a class out of
    // range, a length that is not the cell count, or a step before set_step.
    refusal step(std::size_t k, std::span<const double> c, std::span<double> out,
                 step_record& record);

    // A class's budget, and every class's set whole (`restore_budgets`).
    const budget& budget_of(std::size_t k) const;
    // Throws std::invalid_argument for a count other than the class count.
    void set_budgets(const std::vector<budget>& budgets);

    // The interval's quantities, for the parity tests (prediction (b)): per
    // class and direction, the advecting velocity, its volume flux, the
    // conductance and the deposition per face times dt (faces(d)); the
    // deposition per cell and dt times the source rate (cell fields, the
    // latter empty without a source); the source's total; each recirculation's
    // drawn flux; per direction the face eddy viscosity (empty on the laminar
    // path) and the conductance per unit diffusivity.
    std::span<const double> advecting_velocity(std::size_t k, int d) const;
    std::span<const double> volume_flux(std::size_t k, int d) const;
    std::span<const double> conductance(std::size_t k, int d) const;
    std::span<const double> deposition_step(std::size_t k, int d) const;
    std::span<const double> deposition_cell(std::size_t k) const;
    std::span<const double> source_step(std::size_t k) const;
    double source_total(std::size_t k) const;
    std::span<const double> drawn(std::size_t k) const;
    std::span<const double> face_eddy_viscosity(int d) const;
    std::span<const double> unit_conductance(int d) const;
    // The masks the step reads, set where they hold: per class and direction,
    // along the lines of direction d (layout::lines), the low face's inward
    // and outward flux and the high face's (4 masks, in that order); and on
    // faces(d), the vent faces entering the room from the low side's SOLID
    // cell, from the high side's, and leaving it to the high side's and to the
    // low side's (4 masks, empty without vents).
    std::span<const lanes::mask> domain_mask(std::size_t k, int d, int which) const;
    std::span<const lanes::mask> vent_mask(std::size_t k, int d, int which) const;

private:
    struct per_class;

    void check_class(std::size_t k) const;
    void check_lengths(const std::vector<conditions<D>>& each,
                       const interval_input<D>& input) const;
    refusal check_values(const std::vector<conditions<D>>& each,
                         const interval_input<D>& input) const;
    const per_class& interval_class(std::size_t k) const;
    double content(std::span<const double> c);
    void recirculate(std::size_t k);
    void book_boundaries(std::size_t k, budget& b);
    void book_deposition(std::size_t k, std::span<const double> c, budget& b);

    const scalar::scheme<D>* scheme_;
    const grid::grid<D>* grid_;
    settings keys_;
    std::vector<class_physics> physics_;
    std::vector<budget> budgets_;

    // Fixed on the grid.
    std::array<std::vector<double>, D> widths_;
    std::vector<double> volume_;
    std::array<std::vector<double>, D> area_;
    std::array<std::vector<double>, D> unit_conductance_;
    std::array<std::vector<double>, D> reach_low_;
    std::array<std::vector<double>, D> reach_high_;
    std::vector<lanes::mask> fluid_;
    std::unique_ptr<bool[]> solid_;
    std::array<std::vector<lanes::mask>, D> live_;
    std::array<std::vector<lanes::mask>, D> inner_;
    std::array<std::vector<lanes::mask>, D> solid_low_;
    std::array<std::vector<lanes::mask>, D> solid_high_;
    struct run {
        std::size_t cell;
        std::size_t length;
    };
    std::vector<run> live_runs_;
    std::size_t live_count_ = 0;

    // Fixed by an interval, and by its step.
    std::vector<per_class> class_;
    bool vents_ = false;
    std::array<std::unique_ptr<bool[]>, D> vent_flags_;
    std::array<std::vector<lanes::mask>, D> vent_live_;
    std::vector<double> nu_cells_;
    std::array<std::vector<double>, D> nu_faces_;
    bool interval_ = false;
    bool stepped_ = false;
    double dt_ = 0.0;

    // A step's scratch.
    std::vector<double> c_;
    std::vector<double> c_star_;
    std::array<std::vector<double>, D> adv_;
    std::vector<double> terms_;
    scalar::implicit_scratch<D> implicit_;
};

extern template class solver<1>;
extern template class solver<2>;

}  // namespace cfdcore::transport
