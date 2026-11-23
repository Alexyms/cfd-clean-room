// The clean-out's recovery readings (ADR-017 C, module `recovery`; ADR-014 C,
// ADR-015 C as amended 2026-11-14).
//
// A port of `src/time_integration.py`'s `RecoveryTracker.update` and `held`,
// with its state. After each step it reads the channel fields
// (`channels.hpp`), the class and channel readings at the sensors, and each
// class's free-area mean:
//
// - the local recovery field: per channel and cell, the last update's time at
//   which the channel was at or above its threshold, NaN until then, SOLID cells
//   never set;
// - per channel, since when every sensor has read below its threshold, NaN while
//   one does not; and since when every channel has, the room's;
// - per class and sensor, the first time the class's reading was at or below its
//   100:1 target, where the target is defined; per class the same for its
//   free-area mean.
//
// Within spec is below, so a cell or a sensor exactly at its threshold has not
// recovered. Every test is the reference's comparison and every write a copy of
// the update's time, so the state is the reference's bit for bit.

#pragma once

#include "cfdcore/grid.hpp"
#include "cfdcore/lanes.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace cfdcore::recovery {

// What the tracker is set up with, per class (which is per channel).
//
// thresholds: particles per m^3 (`SimConfig.class_thresholds`)
// target:     the 100:1 level, the initial level over a hundred (the reference's
//             `initial / HUNDRED_TO_ONE`), particles per m^3
// defined:    whether the 100:1 time is defined: the initial level at least a
//             hundred times the threshold (the reference's `_defined`)
// hold:       `scenario.hold_time`, s, or none
struct keys {
    std::vector<double> thresholds;
    std::vector<double> target;
    std::vector<bool> defined;
    std::optional<double> hold;
};

template <int D>
    requires(D >= 1)
class tracker {
public:
    // The tracker on a grid (which must outlive it) for `sensors` sensors, every
    // field NaN. Throws std::invalid_argument when there is no sensor or no
    // class, or the per-class lists differ in length.
    tracker(const grid::grid<D>& grid, keys k, std::size_t sensors);

    std::size_t classes() const { return keys_.thresholds.size(); }
    std::size_t sensors() const { return sensors_; }

    // An update at time t, s, above the last update's (the caller's contract).
    //
    // channel_fields:   one cell field per channel
    // readings:         each class's reading at each sensor, [classes, sensors]
    // channel_readings: each channel's reading at each sensor, [channels, sensors]
    // means:            each class's free-area mean, particles per m^3
    //
    // Throws std::invalid_argument for lengths that do not agree.
    void update(double t, std::span<const double> channel_fields, std::span<const double> readings,
                std::span<const double> channel_readings, std::span<const double> means);

    // Whether the room has been within spec for the hold time: false without a
    // hold or while the room is not within spec, else the last update's time less
    // the start of the stretch at or above the hold.
    bool held() const;

    // The state (`RecoveryTracker.state`): the local field [channels, cells], the
    // sensors' 100:1 times [classes, sensors], the means' [classes], each
    // channel's within-spec start [channels], the room's, and the last update's
    // time; NaN where not set.
    std::span<const double> last_above() const { return last_above_; }
    std::span<const double> sensor_time() const { return sensor_time_; }
    std::span<const double> mean_time() const { return mean_time_; }
    std::span<const double> class_since() const { return class_since_; }
    double room_since() const { return room_since_; }
    double time() const { return t_; }

    // The state a checkpoint carried, copied. Throws std::invalid_argument for an
    // array of another length.
    void restore(std::span<const double> last_above, std::span<const double> sensor_time,
                 std::span<const double> mean_time, std::span<const double> class_since,
                 double room_since, double t);

private:
    const grid::grid<D>* grid_;
    keys keys_;
    std::size_t sensors_;
    std::vector<lanes::mask> live_;
    std::vector<double> last_above_;
    std::vector<double> sensor_time_;
    std::vector<double> mean_time_;
    std::vector<double> class_since_;
    double room_since_;
    double t_;
};

extern template class tracker<1>;
extern template class tracker<2>;

}  // namespace cfdcore::recovery
