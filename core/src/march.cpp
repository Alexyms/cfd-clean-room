#include "cfdcore/march.hpp"

#include "cfdcore/reduce.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace cfdcore::march {

namespace {

constexpr double nan = std::numeric_limits<double>::quiet_NaN();

[[noreturn]] void refuse(const std::string& why) {
    throw std::invalid_argument("march: " + why);
}

void require_length(std::size_t got, std::size_t expected, const char* what) {
    if (got != expected) {
        refuse(std::string(what) + " does not have the length the run needs");
    }
}

void require_rows(std::size_t got, std::size_t needed, const char* what) {
    if (got < needed) {
        refuse(std::string(what) + " holds fewer rows than the steps asked for");
    }
}

// The setup's sensors and recovery checked against the grid and the classes.
monitor_setup checked(monitor_setup m, std::size_t classes, std::size_t cells) {
    if (m.channels.size() != classes || m.thresholds.size() != classes) {
        refuse("the channels and thresholds are not one per class");
    }
    for (const std::size_t cell : m.sensors) {
        if (cell >= cells) {
            refuse("a sensor's cell is outside the grid");
        }
    }
    if (m.recovery.has_value() && m.recovery->thresholds != m.thresholds) {
        refuse("the recovery's thresholds are not the run's");
    }
    return m;
}

}  // namespace

template <int D>
    requires(D >= 1)
run<D>::run(const scalar::scheme<D>& scheme, const transport::geometry<D>& spacing,
            const transport::settings& keys, std::vector<transport::class_physics> physics,
            monitor_setup monitor)
    : grid_(&scheme.mesh()),
      classes_(physics.size()),
      transport_(scheme, spacing, keys, std::move(physics)),
      channels_(checked(monitor, classes_, scheme.mesh().fields().cell_count()).channels),
      sensors_(monitor.sensors),
      maps_(scheme.mesh(), monitor.thresholds, monitor.onsets) {
    const std::size_t n = grid_->fields().cell_count();
    if (monitor.recovery.has_value()) {
        recovery_ = std::make_unique<recovery::tracker<D>>(scheme.mesh(), *monitor.recovery,
                                                           sensors_.size());
    }
    // `np.outer(dy_cell, dx_cell)[live].sum()`: the non-SOLID cells' volumes in
    // the cell field's order, a fresh contiguous array, summed pairwise.
    const auto volumes = transport_.volumes();
    std::vector<double> live;
    live.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (!grid_->solid(i)) {
            live.push_back(volumes[i]);
        }
    }
    free_area_ = reduce::pairwise_sum(live);
    fields_.assign(classes_ * n, 0.0);
    channel_fields_.assign(classes_ * n, 0.0);
    readings_.assign(classes_ * sensors_.size(), 0.0);
    channel_readings_.assign(classes_ * sensors_.size(), 0.0);
    means_.assign(classes_, 0.0);
}

template <int D>
    requires(D >= 1)
run<D>::~run() = default;

template <int D>
    requires(D >= 1)
void run<D>::restore(const resident& state) {
    require_length(state.fields.size(), fields_.size(), "the class fields");
    require_length(state.budgets.size(), classes_, "the budgets");
    require_length(state.first_above.size(), maps_.first_above().size(), "the arrival maps");
    if (recovery_ == nullptr) {
        if (!state.last_above.empty() || !state.sensor_time.empty() ||
            !state.mean_time.empty() || !state.class_since.empty()) {
            refuse("recovery fields given to a run without a recovery");
        }
    } else {
        recovery_->restore(state.last_above, state.sensor_time, state.mean_time,
                           state.class_since, state.room_since, state.recovery_t);
    }
    std::copy(state.fields.begin(), state.fields.end(), fields_.begin());
    transport_.set_budgets(state.budgets);
    maps_.restore(state.first_above, state.arrival_t);
    step_ = state.step;
    t_ = state.t;
    restored_ = true;
    interval_ = false;
}

template <int D>
    requires(D >= 1)
transport::refusal run<D>::interval(const std::vector<transport::conditions<D>>& each,
                                    const transport::interval_input<D>& input, const plan& p) {
    if (!restored_) {
        refuse("an interval before the state is restored");
    }
    if (p.steps < 1 || p.first < 0 || p.first >= p.steps) {
        refuse("the plan's first step is not in [0, steps)");
    }
    interval_ = false;
    transport::refusal r = transport_.set_interval(each, input);
    if (!r.ok()) {
        return r;
    }
    r = transport_.set_step(p.dt);
    if (!r.ok()) {
        return r;
    }
    plan_ = p;
    i_ = p.first;
    interval_ = true;
    return {};
}

template <int D>
    requires(D >= 1)
std::size_t run<D>::remaining() const {
    return interval_ ? static_cast<std::size_t>(plan_.steps - i_) : 0;
}

template <int D>
    requires(D >= 1)
void run<D>::observe(std::size_t row, const rows& out) {
    // After every class's step: the readings, the channels where something reads
    // them, the maps and the recovery, as `_Run.march` orders its calls.
    const std::size_t n = grid_->fields().cell_count();
    const std::size_t s = sensors_.size();
    channels::sensor_readings(fields_, n, sensors_, readings_);
    channels::sensor_channels(readings_, s, channels_, channel_readings_);
    if (maps_.started(t_) || recovery_ != nullptr) {
        channels::channel_fields(grid_->fields(), fields_, channels_, channel_fields_);
    }
    maps_.update(t_, channel_fields_);
    if (recovery_ != nullptr) {
        // `[b.current / self.free_area for b in solver.budget]`.
        for (std::size_t k = 0; k < classes_; ++k) {
            means_[k] = *transport_.budget_of(k).current / free_area_;
        }
        recovery_->update(t_, channel_fields_, readings_, channel_readings_, means_);
    }
    out.step[row] = step_;
    out.t[row] = t_;
    for (std::size_t k = 0; k < classes_; ++k) {
        const auto& b = transport_.budget_of(k);
        double* terms = out.budgets.data() + (row * classes_ + k) * budget_terms;
        terms[0] = b.initial.value_or(nan);
        terms[1] = b.inflow;
        terms[2] = b.outflow;
        terms[3] = b.source;
        for (std::size_t slot = 0; slot < transport::slots; ++slot) {
            terms[4 + slot] = b.deposited[slot];
        }
        terms[8] = b.current.value_or(nan);
        terms[9] = b.leftover;
    }
    std::copy(readings_.begin(), readings_.end(), out.readings.begin() + row * classes_ * s);
    std::copy(channel_readings_.begin(), channel_readings_.end(),
              out.channels.begin() + row * classes_ * s);
    if (recovery_ != nullptr) {
        std::copy(means_.begin(), means_.end(), out.means.begin() + row * classes_);
    }
}

template <int D>
    requires(D >= 1)
result run<D>::march(std::size_t n, const rows& out) {
    if (!interval_) {
        refuse("a march without an interval");
    }
    const std::size_t s = sensors_.size();
    const std::size_t cells = grid_->fields().cell_count();
    const std::size_t asked = std::min(n, remaining());
    require_rows(out.step.size(), asked, "the rows' step column");
    require_rows(out.t.size(), asked, "the rows' time column");
    require_rows(out.budgets.size(), asked * classes_ * budget_terms, "the rows' budgets");
    require_rows(out.readings.size(), asked * classes_ * s, "the rows' readings");
    require_rows(out.channels.size(), asked * classes_ * s, "the rows' channels");
    if (recovery_ != nullptr) {
        require_rows(out.means.size(), asked * classes_, "the rows' means");
    }
    require_rows(out.sweeps.size(), asked * classes_, "the rows' sweeps");
    require_rows(out.converged.size(), asked * classes_, "the rows' converged flags");
    result r;
    if (held()) {
        r.reason = stop::held;
        return r;
    }
    for (std::size_t row = 0; row < asked; ++row) {
        // Every class in class order, each field advanced in place.
        for (std::size_t k = 0; k < classes_; ++k) {
            const std::span<double> field(fields_.data() + k * cells, cells);
            transport::step_record record;
            const transport::refusal refusal = transport_.step(k, field, field, record);
            if (refusal.code == transport::status::breach) {
                r.reason = stop::breach;
                r.refusal = refusal;
                r.record = record;
                return r;
            }
            if (!refusal.ok()) {
                r.reason = stop::refused;
                r.refusal = refusal;
                r.record = record;
                return r;
            }
            out.sweeps[row * classes_ + k] = record.sweeps;
            out.converged[row * classes_ + k] = record.converged ? 1 : 0;
        }
        ++i_;
        ++step_;
        // `IntervalPlan.time`: the interval's end exactly at its last step.
        t_ = i_ == plan_.steps ? plan_.t_end
                               : plan_.t_start + static_cast<double>(i_) * plan_.dt;
        observe(row, out);
        r.steps = row + 1;
        if (held()) {
            r.reason = stop::held;
            return r;
        }
    }
    r.reason = i_ == plan_.steps ? stop::interval_end : stop::count;
    return r;
}

template <int D>
    requires(D >= 1)
std::span<const double> run<D>::field(std::size_t k) const {
    if (k >= classes_) {
        refuse("no class " + std::to_string(k));
    }
    const std::size_t n = grid_->fields().cell_count();
    return std::span<const double>(fields_.data() + k * n, n);
}

template class run<1>;
template class run<2>;

}  // namespace cfdcore::march
