#include "cfdcore/channels.hpp"

#include "cfdcore/lanes.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace cfdcore::channels {

namespace {

[[noreturn]] void refuse(const std::string& why) {
    throw std::invalid_argument("channels: " + why);
}

// A channel's running sum over N cells of a row: `total += values[j]`, one
// rounding per cell. The output is the running sum itself, read and written;
// it overlaps no input.
struct add_row {
    const double* value;
};

template <std::size_t N>
inline void add_cells(lanes::cells<N>, const add_row in, std::size_t first,
                      double* CFDCORE_RESTRICT total) {
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t i = first + k;
        total[i] = total[i] + in.value[i];
    }
}

}  // namespace

set::set(std::vector<std::vector<std::size_t>> members) : members_(std::move(members)) {
    const std::size_t n = members_.size();
    if (n == 0) {
        refuse("there is no channel");
    }
    for (std::size_t k = 0; k < n; ++k) {
        const auto& m = members_[k];
        bool own = false;
        std::vector<bool> seen(n, false);
        for (const std::size_t j : m) {
            if (j >= n || seen[j]) {
                refuse("channel " + std::to_string(k) +
                       " names a class outside the classes or twice");
            }
            seen[j] = true;
            own = own || j == k;
        }
        if (!own) {
            refuse("channel " + std::to_string(k) + " leaves out its own class");
        }
    }
}

template <int D>
void channel_field(const layout::layout<D>& fields, std::span<const double> classes,
                   std::span<const std::size_t> members, std::span<double> out) {
    const std::size_t n = fields.cell_count();
    if (out.size() != n || n == 0 || classes.size() % n != 0) {
        refuse("the fields are not whole cell fields");
    }
    const std::size_t count = classes.size() / n;
    if (members.empty()) {
        refuse("a channel sums at least one class");
    }
    for (const std::size_t j : members) {
        if (j >= count) {
            refuse("a member is not a class of the fields");
        }
    }
    const std::size_t n0 = fields.cells(0);
    // `np.zeros(...)`, then `total += values[j]` for each member in turn.
    for (double& x : out) {
        x = 0.0;
    }
    for (const std::size_t j : members) {
        const double* field = classes.data() + j * n;
        for (const auto& line : fields.lines(0)) {
            const add_row in{field + line.cell};
            double* total = out.data() + line.cell;
            lanes::for_each_block(0, n0, [&](auto cells, std::size_t m) {
                add_cells(cells, in, m, total);
            });
        }
    }
}

template <int D>
void channel_fields(const layout::layout<D>& fields, std::span<const double> classes,
                    const set& channels, std::span<double> out) {
    const std::size_t n = fields.cell_count();
    if (classes.size() != channels.size() * n || out.size() != channels.size() * n) {
        refuse("the fields are not one cell field per class");
    }
    for (std::size_t k = 0; k < channels.size(); ++k) {
        channel_field(fields, classes, channels.members(k), out.subspan(k * n, n));
    }
}

void sensor_readings(std::span<const double> classes, std::size_t cells,
                     std::span<const std::size_t> sensors, std::span<double> out) {
    if (cells == 0 || classes.size() % cells != 0) {
        refuse("the fields are not whole cell fields");
    }
    const std::size_t count = classes.size() / cells;
    const std::size_t s = sensors.size();
    if (out.size() != count * s) {
        refuse("the readings are not one value per class and sensor");
    }
    for (const std::size_t cell : sensors) {
        if (cell >= cells) {
            refuse("a sensor's cell is outside the field");
        }
    }
    for (std::size_t k = 0; k < count; ++k) {
        for (std::size_t m = 0; m < s; ++m) {
            out[k * s + m] = classes[k * cells + sensors[m]];
        }
    }
}

void sensor_channels(std::span<const double> readings, std::size_t sensors, const set& channels,
                     std::span<double> out) {
    const std::size_t n = channels.size();
    if (readings.size() != n * sensors || out.size() != n * sensors) {
        refuse("the readings are not one value per class and sensor");
    }
    for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t m = 0; m < sensors; ++m) {
            double total = 0.0;
            for (const std::size_t j : channels.members(k)) {
                total = total + readings[j * sensors + m];
            }
            out[k * sensors + m] = total;
        }
    }
}

template void channel_field<1>(const layout::layout<1>&, std::span<const double>,
                               std::span<const std::size_t>, std::span<double>);
template void channel_field<2>(const layout::layout<2>&, std::span<const double>,
                               std::span<const std::size_t>, std::span<double>);
template void channel_fields<1>(const layout::layout<1>&, std::span<const double>, const set&,
                                std::span<double>);
template void channel_fields<2>(const layout::layout<2>&, std::span<const double>, const set&,
                                std::span<double>);

}  // namespace cfdcore::channels
