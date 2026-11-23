// The cumulative channels and the sensors' readings (ADR-017 C, module
// `channels`; ADR-015 C as amended 2026-11-14).
//
// A port of `src/monitor.py`'s `channel_sum` and of the reading the alert
// monitor takes at each sensor's cell. Channel k reads the classes its
// configuration names (`SimConfig.channel_classes`): zero, then each member's
// value added in turn, one rounding each, so a channel's value at a cell is the
// reference's bit for bit, and the channel read at a sensor's cell equals the
// sum of the class readings there (the two add the same terms in the same
// order).
//
// A set of class fields is one array, class k's cell field at k times the cell
// count (layout.hpp's cell field per class, class 0 first).

#pragma once

#include "cfdcore/layout.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace cfdcore::channels {

// Per channel, the classes it sums in the order they are added.
class set {
public:
    set() = default;

    // members: per channel k, its classes. Throws std::invalid_argument unless
    // there is one channel per class (the classes being 0 to members.size() - 1)
    // and each names its own class and others in that range, each once: the
    // refusals of `monitor.checked_channels`, which a validated configuration
    // never meets.
    explicit set(std::vector<std::vector<std::size_t>> members);

    // The number of channels, which is the number of classes.
    std::size_t size() const { return members_.size(); }

    // Channel k's classes in the order they are added.
    std::span<const std::size_t> members(std::size_t k) const { return members_.at(k); }

private:
    std::vector<std::vector<std::size_t>> members_;
};

// One channel's field (`channel_sum`): zero, then each member's field added in
// turn, walked by rows.
//
// fields:  the layout of a cell field
// classes: the class fields, one cell field per class, class k's at k times the
//          cell count, particles per m^3
// members: the channel's classes, each below the class count
// out:     output, one cell field; overlaps no input
//
// Throws std::invalid_argument when `out` is not one cell field long, `fields`
// is not a whole number of cell fields, or a member is not a class of it.
template <int D>
void channel_field(const layout::layout<D>& fields, std::span<const double> classes,
                   std::span<const std::size_t> members, std::span<double> out);

// Every channel's field, channel k's at k times the cell count (`channel_sums`).
//
// classes: the class fields, as channel_field takes them
// channels: one channel per class
// out:     output, one cell field per channel; overlaps no input
//
// Throws std::invalid_argument for lengths that are not one cell field per class.
template <int D>
void channel_fields(const layout::layout<D>& fields, std::span<const double> classes,
                    const set& channels, std::span<double> out);

// Each class's reading at each sensor's cell (`AlertMonitor.update`'s
// `fields[k][rows, cols]`), class k's at k times the sensor count.
//
// classes: the class fields, one cell field of `cells` values per class
// sensors: the cell field index each sensor reads
// out:     output, classes times sensors values; overlaps no input
//
// Throws std::invalid_argument for a sensor's cell outside the field or lengths
// that do not agree.
void sensor_readings(std::span<const double> classes, std::size_t cells,
                     std::span<const std::size_t> sensors, std::span<double> out);

// Each channel's reading at each sensor (`AlertMonitor._sums`): zero, then each
// member's reading added in turn, channel k's at k times the sensor count.
//
// readings: the class readings, as sensor_readings writes them
// sensors:  the sensor count
// out:      output, one value per channel and sensor; overlaps no input
//
// Throws std::invalid_argument for lengths that are not one value per class and
// sensor.
void sensor_channels(std::span<const double> readings, std::size_t sensors, const set& channels,
                     std::span<double> out);

extern template void channel_field<1>(const layout::layout<1>&, std::span<const double>,
                                      std::span<const std::size_t>, std::span<double>);
extern template void channel_field<2>(const layout::layout<2>&, std::span<const double>,
                                      std::span<const std::size_t>, std::span<double>);
extern template void channel_fields<1>(const layout::layout<1>&, std::span<const double>,
                                       const set&, std::span<double>);
extern template void channel_fields<2>(const layout::layout<2>&, std::span<const double>,
                                       const set&, std::span<double>);

}  // namespace cfdcore::channels
