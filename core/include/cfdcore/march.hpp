// The march: the step loop with the fields resident (ADR-017 B and C, modules
// `march` and `run`; ADR-014 C and D, ADR-015 C and D, ADR-016).
//
// A port of the inner loop of `src/time_integration.py`'s `_Run.march`: every
// class's step in class order (`transport.hpp`), the step count and time, the
// sensors' readings and channels, the arrival maps and the clean-out's recovery
// fields, and the hold. A run owns the class fields, the budgets, the maps and
// the recovery fields for its whole length; Python hands it each interval's plan
// and data, asks for at most n steps at a time (to its next frame, checkpoint or
// the interval's end), and reads per step what ADR-017 B's exchange table names.
// The alert monitor reads the readings returned and feeds nothing back, so a
// batch of steps is the same as one at a time; the hold does feed back, so it
// stops the march here.
//
// Each step is the reference's: the times are `IntervalPlan.time`'s, the means
// `MassBudget.current` over the free area, which is formed as the reference
// forms it (the non-SOLID cells' volumes in the cell field's order, NumPy's
// pairwise sum), and every block below is the reference's bit for bit.

#pragma once

#include "cfdcore/arrival.hpp"
#include "cfdcore/channels.hpp"
#include "cfdcore/recovery.hpp"
#include "cfdcore/transport.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace cfdcore::march {

// A budget's terms, in the order a step's row holds them: the reference's
// `MassBudget` fields, the deposition by slot.
inline constexpr std::size_t budget_terms = 10;
inline constexpr std::array<std::string_view, budget_terms> budget_term_names = {
    "initial", "inflow", "outflow", "source", "floor",
    "ceiling", "wall",   "obstacle", "current", "leftover"};

// What the step loop reads beside the transport.
//
// channels:   per channel, which is per class, the classes it sums
// thresholds: per class, particles per m^3
// sensors:    the cell field index each sensor reads, in the configuration's order
// onsets:     the contaminating events' onsets, s, at least one
// recovery:   the clean-out's keys, or none for a run without an initial event
struct monitor_setup {
    std::vector<std::vector<std::size_t>> channels;
    std::vector<double> thresholds;
    std::vector<std::size_t> sensors;
    std::vector<double> onsets;
    std::optional<recovery::keys> recovery;
};

// An interval as the march steps it (`IntervalPlan`): steps i = first + 1 to
// steps, step i at `t_start + i dt` and the last at `t_end` exactly.
struct plan {
    double t_start = 0.0;
    double t_end = 0.0;
    double dt = 0.0;
    std::int64_t steps = 0;
    std::int64_t first = 0;
};

// Why a march returned.
enum class stop {
    count,         // it took the steps asked for
    interval_end,  // it reached the interval's last step
    held,          // the clean-out's hold (at entry, or after its last step)
    breach,        // a class's step breached ADR-016's limit (the step not counted)
    refused,       // a class's step refused its input (the step not counted)
};

// Where a march writes its steps, row r for its r-th step: the run's step count,
// the time (s), every class's budget terms [classes, budget_terms] (NaN for a
// term not yet set), each class's reading and each channel's at each sensor
// [classes, sensors], each class's free-area mean [classes] (empty without a
// recovery), and each class's sweeps and whether its solve converged [classes].
// Each span holds at least as many rows as steps asked for.
struct rows {
    std::span<std::int64_t> step;
    std::span<double> t;
    std::span<double> budgets;
    std::span<double> readings;
    std::span<double> channels;
    std::span<double> means;
    std::span<std::int64_t> sweeps;
    std::span<std::uint8_t> converged;
};

// What a march did: the steps taken and written, why it returned, and for a
// breach or a refusal the class step's refusal and records (the breach's leftover
// and supply among them).
struct result {
    std::size_t steps = 0;
    stop reason = stop::count;
    transport::refusal refusal;
    transport::step_record record;
};

// The resident state a checkpoint carries (ADR-017 F, Restart), handed in by
// `restore`: the class fields [classes, cells], every class's budget, the run's
// step count and time, the arrival maps [events, channels, cells] and their last
// update's time, and with a recovery its state (recovery.hpp; empty otherwise).
struct resident {
    std::span<const double> fields;
    std::vector<transport::budget> budgets;
    std::int64_t step = 0;
    double t = 0.0;
    std::span<const double> first_above;
    double arrival_t = 0.0;
    std::span<const double> last_above;
    std::span<const double> sensor_time;
    std::span<const double> mean_time;
    std::span<const double> class_since;
    double room_since = 0.0;
    double recovery_t = 0.0;
};

template <int D>
    requires(D >= 1)
class run {
public:
    // A run on a scheme's grid (which must outlive it) with the transport's
    // spacings, keys and physics (transport.hpp) and the monitor's setup. Every
    // array a step needs is allocated here. Throws std::invalid_argument when the
    // setup's lists do not agree (one channel and threshold per class, a sensor's
    // cell outside the grid, a recovery's lists of another length, a recovery
    // without a sensor).
    run(const scalar::scheme<D>& scheme, const transport::geometry<D>& spacing,
        const transport::settings& keys, std::vector<transport::class_physics> physics,
        monitor_setup monitor);
    ~run();
    run(const run&) = delete;
    run& operator=(const run&) = delete;

    std::size_t classes() const { return classes_; }
    std::size_t sensors() const { return sensors_.size(); }
    bool has_recovery() const { return recovery_ != nullptr; }

    // The free area, m^2 per unit depth for D = 2: the non-SOLID cells' volumes
    // summed as `np.outer(dy, dx)[live].sum()` sums them.
    double free_area() const { return free_area_; }

    // Sets the whole resident state (ADR-017 B's `restore`), after which the next
    // step is the uninterrupted run's once its interval is set. Every interval is
    // cleared. Throws std::invalid_argument for an array of another length, or
    // recovery arrays given without a recovery.
    void restore(const resident& state);

    // An interval: its classes' conditions and fixed inputs (transport.hpp's
    // set_interval) and its plan, the step set to plan.dt. Returns the transport's
    // refusal, nothing kept on a refusal. Throws std::invalid_argument before a
    // restore, or for a plan whose first step is not in [0, steps).
    transport::refusal interval(const std::vector<transport::conditions<D>>& each,
                                const transport::interval_input<D>& input, const plan& p);

    // The steps left in the interval: 0 without one.
    std::size_t remaining() const;

    // At most n steps of the interval, each written to `out`'s next row: every
    // class's step, the step count and time, the readings and channels, the maps
    // and, with a recovery, the means, the recovery fields and the hold. Stops at
    // n steps, at the interval's end, after a step that holds, or before counting
    // a step a class breaches or refuses. Returns at once, with no step, when the
    // run holds already or no steps remain. Throws std::invalid_argument without
    // an interval, or rows shorter than the steps asked for.
    result march(std::size_t n, const rows& out);

    // The resident state, read-only: class k's field (a cell field), the budgets,
    // the step count and time, the maps and the recovery tracker (null without).
    std::span<const double> field(std::size_t k) const;
    std::span<const double> fields() const { return fields_; }
    const transport::budget& budget_of(std::size_t k) const { return transport_.budget_of(k); }
    std::int64_t step() const { return step_; }
    double time() const { return t_; }
    const arrival::maps<D>& maps() const { return maps_; }
    const recovery::tracker<D>* tracker() const { return recovery_.get(); }

    // Whether the clean-out's hold holds: false without a recovery.
    bool held() const { return recovery_ != nullptr && recovery_->held(); }

private:
    void observe(std::size_t row, const rows& out);

    const grid::grid<D>* grid_;
    std::size_t classes_;
    transport::solver<D> transport_;
    channels::set channels_;
    std::vector<std::size_t> sensors_;
    arrival::maps<D> maps_;
    std::unique_ptr<recovery::tracker<D>> recovery_;
    double free_area_ = 0.0;

    bool restored_ = false;
    bool interval_ = false;
    plan plan_{};
    std::int64_t i_ = 0;
    std::int64_t step_ = 0;
    double t_ = 0.0;

    // The resident class fields, and a step's scratch: the channel fields, the
    // readings and channel readings, the means.
    std::vector<double> fields_;
    std::vector<double> channel_fields_;
    std::vector<double> readings_;
    std::vector<double> channel_readings_;
    std::vector<double> means_;
};

extern template class run<1>;
extern template class run<2>;

}  // namespace cfdcore::march
