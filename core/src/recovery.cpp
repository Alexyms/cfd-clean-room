#include "cfdcore/recovery.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace cfdcore::recovery {

namespace {

constexpr double nan = std::numeric_limits<double>::quiet_NaN();

[[noreturn]] void refuse(const std::string& why) {
    throw std::invalid_argument("recovery: " + why);
}

void require_length(std::size_t got, std::size_t expected, const char* what) {
    if (got != expected) {
        refuse(std::string(what) + " does not have the length the tracker needs");
    }
}

// The local recovery field's update over N cells of a row
// (`last_above[k][(sums[k] >= threshold) & live] = t`). The output is the field
// itself, read and written; it overlaps no input.
struct local_row {
    const double* sum;
    const lanes::mask* live;
};

template <std::size_t N>
inline void local_cells(lanes::cells<N>, const local_row in, double threshold, double t,
                        std::size_t first, double* CFDCORE_RESTRICT last) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const lanes::mask above = lanes::mask_of(in.sum[i] >= threshold) & in.live[i];
        last[i] = lanes::select(above, t, last[i]);
    }
}

}  // namespace

template <int D>
    requires(D >= 1)
tracker<D>::tracker(const grid::grid<D>& grid, keys k, std::size_t sensors)
    : grid_(&grid), keys_(std::move(k)), sensors_(sensors), room_since_(nan), t_(nan) {
    const std::size_t c = keys_.thresholds.size();
    if (c == 0) {
        refuse("there is no class");
    }
    if (keys_.target.size() != c || keys_.defined.size() != c) {
        refuse("the thresholds, targets and defined flags differ in length");
    }
    if (sensors_ == 0) {
        refuse("a clean-out needs at least one sensor");
    }
    const std::size_t n = grid.fields().cell_count();
    live_.resize(n);
    const auto solid = grid.solid_flags();
    for (std::size_t i = 0; i < n; ++i) {
        live_[i] = ~solid[i];
    }
    last_above_.assign(c * n, nan);
    sensor_time_.assign(c * sensors_, nan);
    mean_time_.assign(c, nan);
    class_since_.assign(c, nan);
}

template <int D>
    requires(D >= 1)
void tracker<D>::update(double t, std::span<const double> channel_fields,
                        std::span<const double> readings,
                        std::span<const double> channel_readings,
                        std::span<const double> means) {
    const auto& fields = grid_->fields();
    const std::size_t n = fields.cell_count();
    const std::size_t n0 = fields.cells(0);
    const std::size_t c = keys_.thresholds.size();
    const std::size_t s = sensors_;
    require_length(channel_fields.size(), c * n, "the channel fields");
    require_length(readings.size(), c * s, "the readings");
    require_length(channel_readings.size(), c * s, "the channel readings");
    require_length(means.size(), c, "the means");

    // The local field, channel by channel.
    for (std::size_t k = 0; k < c; ++k) {
        const double threshold = keys_.thresholds[k];
        const double* sum = channel_fields.data() + k * n;
        double* last = last_above_.data() + k * n;
        for (const auto& line : fields.lines(0)) {
            const local_row in{sum + line.cell, live_.data() + line.cell};
            double* row = last + line.cell;
            lanes::for_each_block(0, n0, [&](auto cells, std::size_t m) {
                local_cells(cells, in, threshold, t, m, row);
            });
        }
    }
    // Within spec per channel: every sensor below the threshold.
    bool room = true;
    for (std::size_t k = 0; k < c; ++k) {
        bool within = true;
        for (std::size_t m = 0; m < s; ++m) {
            within = within && channel_readings[k * s + m] < keys_.thresholds[k];
        }
        room = room && within;
        const double since = class_since_[k];
        class_since_[k] = within ? (std::isnan(since) ? t : since) : nan;
    }
    if (room) {
        if (std::isnan(room_since_)) {
            room_since_ = t;
        }
    } else {
        room_since_ = nan;
    }
    // The 100:1 times: each class's own reading and its free-area mean against
    // its target, where the target is defined.
    for (std::size_t k = 0; k < c; ++k) {
        const bool defined = keys_.defined[k];
        for (std::size_t m = 0; m < s; ++m) {
            double& when = sensor_time_[k * s + m];
            if (readings[k * s + m] <= keys_.target[k] && defined && std::isnan(when)) {
                when = t;
            }
        }
        if (means[k] <= keys_.target[k] && defined && std::isnan(mean_time_[k])) {
            mean_time_[k] = t;
        }
    }
    t_ = t;
}

template <int D>
    requires(D >= 1)
bool tracker<D>::held() const {
    if (!keys_.hold.has_value() || std::isnan(room_since_)) {
        return false;
    }
    return t_ - room_since_ >= *keys_.hold;
}

template <int D>
    requires(D >= 1)
void tracker<D>::restore(std::span<const double> last_above, std::span<const double> sensor_time,
                         std::span<const double> mean_time, std::span<const double> class_since,
                         double room_since, double t) {
    require_length(last_above.size(), last_above_.size(), "last_above");
    require_length(sensor_time.size(), sensor_time_.size(), "sensor_time");
    require_length(mean_time.size(), mean_time_.size(), "mean_time");
    require_length(class_since.size(), class_since_.size(), "class_since");
    last_above_.assign(last_above.begin(), last_above.end());
    sensor_time_.assign(sensor_time.begin(), sensor_time.end());
    mean_time_.assign(mean_time.begin(), mean_time.end());
    class_since_.assign(class_since.begin(), class_since.end());
    room_since_ = room_since;
    t_ = t;
}

template class tracker<1>;
template class tracker<2>;

}  // namespace cfdcore::recovery
