#include "cfdcore/arrival.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace cfdcore::arrival {

namespace {

[[noreturn]] void refuse(const std::string& why) {
    throw std::invalid_argument("arrival: " + why);
}

// One map's update over N cells of a row (`first[above & np.isnan(first)] = t`,
// `above` being `(sums[k] >= threshold) & live`). The output is the map itself,
// read and written; it overlaps no input.
struct map_row {
    const double* sum;
    const lanes::mask* live;
};

template <std::size_t N>
inline void map_cells(lanes::cells<N>, const map_row in, double threshold, double t,
                      std::size_t first, double* CFDCORE_RESTRICT map) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        const double before = map[i];
        const lanes::mask reached =
            lanes::mask_of(in.sum[i] >= threshold) & in.live[i] & lanes::mask_of(before != before);
        map[i] = lanes::select(reached, t, before);
    }
}

}  // namespace

template <int D>
    requires(D >= 1)
maps<D>::maps(const grid::grid<D>& grid, std::vector<double> thresholds,
              std::vector<double> onsets)
    : grid_(&grid),
      thresholds_(std::move(thresholds)),
      onsets_(std::move(onsets)),
      t_(std::numeric_limits<double>::quiet_NaN()) {
    if (thresholds_.empty()) {
        refuse("there is no class");
    }
    for (const double x : thresholds_) {
        if (!std::isfinite(x) || x < 0.0) {
            refuse("a threshold is not finite and non-negative");
        }
    }
    if (onsets_.empty()) {
        refuse("there is no event");
    }
    for (std::size_t e = 0; e < onsets_.size(); ++e) {
        if (!std::isfinite(onsets_[e]) || onsets_[e] < 0.0 ||
            (e > 0 && onsets_[e] < onsets_[e - 1])) {
            refuse("the onsets are not finite, non-negative and in order");
        }
    }
    const std::size_t n = grid.fields().cell_count();
    live_.resize(n);
    const auto solid = grid.solid_flags();
    for (std::size_t i = 0; i < n; ++i) {
        live_[i] = ~solid[i];
    }
    first_.assign(onsets_.size() * thresholds_.size() * n,
                  std::numeric_limits<double>::quiet_NaN());
}

template <int D>
    requires(D >= 1)
void maps<D>::update(double t, std::span<const double> channel_fields) {
    const auto& fields = grid_->fields();
    const std::size_t n = fields.cell_count();
    const std::size_t n0 = fields.cells(0);
    const std::size_t c = thresholds_.size();
    if (started(t)) {
        if (channel_fields.size() != c * n) {
            refuse("the channel fields are not one cell field per channel");
        }
        for (std::size_t k = 0; k < c; ++k) {
            const double threshold = thresholds_[k];
            const double* sum = channel_fields.data() + k * n;
            // `np.flatnonzero(onsets <= t)`: the started events, in order.
            for (std::size_t e = 0; e < onsets_.size() && onsets_[e] <= t; ++e) {
                double* map = first_.data() + (e * c + k) * n;
                for (const auto& line : fields.lines(0)) {
                    const map_row in{sum + line.cell, live_.data() + line.cell};
                    double* row = map + line.cell;
                    lanes::for_each_block(0, n0, [&](auto cells, std::size_t m) {
                        map_cells(cells, in, threshold, t, m, row);
                    });
                }
            }
        }
    }
    t_ = t;
}

template <int D>
    requires(D >= 1)
void maps<D>::restore(std::span<const double> first_above, double t) {
    if (first_above.size() != first_.size()) {
        refuse("the maps to restore are not [events, channels, cells]");
    }
    first_.assign(first_above.begin(), first_above.end());
    t_ = t;
}

template class maps<1>;
template class maps<2>;

}  // namespace cfdcore::arrival
