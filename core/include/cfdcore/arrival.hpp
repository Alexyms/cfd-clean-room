// The arrival maps (ADR-017 C, module `arrival`; ADR-015 D, and C as amended
// 2026-11-14).
//
// A port of `src/monitor.py`'s `ArrivalTracker.update` and its state: per event,
// channel and cell, the first update's time at or after the event's onset at
// which the channel's value in the cell is at or above its class's threshold,
// NaN until then. An update reads the channel fields (`channels.hpp`); only the
// events whose onset is at or before the update's time are updated, a cell
// already set keeps its time, and SOLID cells stay NaN. Each cell's test is the
// reference's comparison and its write a copy of the update's time, so the maps
// are the reference's bit for bit.
//
// A map is [events, channels, cells]: event e's map of channel k is the cell
// field at (e * channels + k) times the cell count.

#pragma once

#include "cfdcore/grid.hpp"
#include "cfdcore/lanes.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace cfdcore::arrival {

template <int D>
    requires(D >= 1)
class maps {
public:
    // The maps on a grid (which must outlive them), all NaN, before any update.
    //
    // thresholds: per class, which is per channel, particles per m^3, finite and
    //             non-negative (`checked_thresholds`)
    // onsets:     the contaminating events' onsets, s, finite, non-negative and
    //             in order, at least one (`checked_onsets`)
    //
    // Throws std::invalid_argument for a refusal of the reference's checks, which
    // a validated configuration never meets.
    maps(const grid::grid<D>& grid, std::vector<double> thresholds, std::vector<double> onsets);

    std::size_t events() const { return onsets_.size(); }
    std::size_t channels() const { return thresholds_.size(); }

    // Whether some event's onset is at or before t: an update at t then reads the
    // channel fields, which the reference sums only then.
    bool started(double t) const { return !onsets_.empty() && onsets_.front() <= t; }

    // An update at time t, s, above the last update's (the caller's contract, as
    // the march's times rise): every started event's map of each channel set to t
    // where the cell is not SOLID, not yet set, and the channel at or above its
    // threshold. Before any event has started, only the time is kept.
    //
    // channel_fields: one cell field per channel (`channels::channel_fields`),
    //                 read only when an event has started
    //
    // Throws std::invalid_argument when channel_fields is read and is not one cell
    // field per channel.
    void update(double t, std::span<const double> channel_fields);

    // The maps, [events, channels, cells], s, NaN where not reached.
    std::span<const double> first_above() const { return first_; }

    // The last update's time, s, NaN before any (`arrival_t`).
    double time() const { return t_; }

    // The maps and time a checkpoint carried, copied (`ArrivalTracker.restore`;
    // the reference's checks of the times are the Python side's, made before).
    // Throws std::invalid_argument for maps of another length.
    void restore(std::span<const double> first_above, double t);

private:
    const grid::grid<D>* grid_;
    std::vector<double> thresholds_;
    std::vector<double> onsets_;
    std::vector<lanes::mask> live_;
    std::vector<double> first_;
    double t_;
};

extern template class maps<1>;
extern template class maps<2>;

}  // namespace cfdcore::arrival
