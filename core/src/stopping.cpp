#include "cfdcore/stopping.hpp"

#include "cfdcore/elementary.hpp"
#include "cfdcore/lanes.hpp"
#include "cfdcore/reduce.hpp"
#include "cfdcore/scalar.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace cfdcore::stopping {

namespace {

void require(bool condition, const char* what) {
    if (!condition) {
        throw std::invalid_argument(what);
    }
}

bool positive_and_finite(double x) { return std::isfinite(x) && x > 0.0; }

// The operands of the summary's block: the imbalances.
struct imbalance_operands {
    const double* imbalance;
};

// N cells' absolute values, and the running maximum of each lane, `np.max`'s
// propagation of a NaN kept by the select.
template <std::size_t N>
CFDCORE_INLINE void magnitude_cells(lanes::cells<N>, imbalance_operands in, std::size_t m,
                                    double* CFDCORE_RESTRICT out,
                                    double* CFDCORE_RESTRICT largest) {
    for (std::size_t i = 0; i < N; ++i) {
        const double a = std::fabs(in.imbalance[m + i]);
        out[m + i] = a;
        largest[i] = scalar::np_maximum(a, largest[i]);
    }
}

}  // namespace

imbalance_summary summarise(std::span<const double> imbalance, std::span<double> magnitude) {
    require(imbalance.size() == magnitude.size(),
            "stopping: the imbalance and its scratch differ in length");
    require(!imbalance.empty(), "stopping: an imbalance of no cells");
    // A maximum per lane of a block, combined after the walk: exact in any order,
    // and NaN where a value is NaN, as `np.max` returns it.
    double largest[lanes::width];
    std::fill(largest, largest + lanes::width, 0.0);
    const imbalance_operands in{imbalance.data()};
    lanes::for_each_block(0, imbalance.size(), [&](auto cells, std::size_t m) {
        magnitude_cells(cells, in, m, magnitude.data(), largest);
    });
    double worst = largest[0];
    for (std::size_t i = 1; i < lanes::width; ++i) {
        worst = scalar::np_maximum(largest[i], worst);
    }
    return {worst, reduce::pairwise_sum(magnitude), reduce::pairwise_sum(imbalance)};
}

double viscosity_scale(double nu, std::span<const double> nu_t,
                       std::span<const std::size_t> fluid, std::span<double> scratch) {
    require(!fluid.empty(), "stopping: the median of no FLUID cell");
    require(scratch.size() == fluid.size(), "stopping: the median's scratch is not one per cell");
    bool nan = false;
    for (std::size_t i = 0; i < fluid.size(); ++i) {
        scratch[i] = nu_t[fluid[i]];
        nan |= std::isnan(scratch[i]);
    }
    // `np.median` checks the largest value after its partition, which a NaN is.
    if (nan) {
        return nu + std::numeric_limits<double>::quiet_NaN();
    }
    const std::size_t n = scratch.size();
    const std::size_t half = n / 2;
    // A selection gives the value of a sorted position whatever its algorithm, and
    // for an even count the value below it is the largest of the part before it.
    std::nth_element(scratch.begin(), scratch.begin() + half, scratch.end());
    double middle[2] = {scratch[half], 0.0};
    std::size_t count = 1;
    if (n % 2 == 0) {
        middle[0] = *std::max_element(scratch.begin(), scratch.begin() + half);
        middle[1] = scratch[half];
        count = 2;
    }
    // `np.mean` of the one or two values: their sum as `np.sum` forms it, over the
    // count.
    const double median =
        reduce::pairwise_sum(std::span<const double>(middle, count)) / static_cast<double>(count);
    return nu + median;
}

std::string_view rule::check(const rule_keys& keys) {
    if (!positive_and_finite(keys.velocity_scale) || !positive_and_finite(keys.flux_scale) ||
        !positive_and_finite(keys.iteration_error_tol) ||
        !positive_and_finite(keys.mass_imbalance_tol)) {
        return "a scale or a tolerance is not finite and positive";
    }
    if (keys.window < 2) {
        return "the window holds fewer than two steps";
    }
    if (keys.version_without_e < 1 || keys.version_with_e < 1) {
        return "a version of the rule is below one";
    }
    return {};
}

rule::rule(const rule_keys& keys) : keys_(keys) {
    const std::string_view reason = check(keys);
    if (!reason.empty()) {
        throw std::invalid_argument(std::string("stopping: ") + std::string(reason));
    }
    const std::size_t n = keys.window;
    steps_.reserve(n);
    nu_steps_.reserve(n);
    logs_.assign(n, 0.0);
    abscissa_.assign(n, 0.0);
    // np.arange(n) - (n - 1) / 2.0: each entry exact.
    const double centre = static_cast<double>(n - 1) / 2.0;
    for (std::size_t k = 0; k < n; ++k) {
        abscissa_[k] = static_cast<double>(k) - centre;
    }
    // Each square is a multiple of a quarter below 2^51, and so is every partial
    // sum at any window a solve can hold, so the sum is exact in any order and
    // equals the reference's BLAS dot.
    abscissa_square_ = reduce::blocked_dot(abscissa_, abscissa_, n);
}

void rule::reset() {
    steps_.clear();
    nu_steps_.clear();
    last_ = {};
    recorded_ = false;
}

int rule::version() const {
    return keys_.viscosity_condition ? keys_.version_with_e : keys_.version_without_e;
}

void rule::push(std::vector<double>& window, std::size_t length, double value) {
    if (window.size() < length) {
        window.push_back(value);
        return;
    }
    std::copy(window.begin() + 1, window.end(), window.begin());
    window.back() = value;
}

restore_status rule::restore(std::span<const double> steps,
                             std::span<const double> viscosity_steps) {
    if (stepped()) {
        return restore_status::already_stepped;
    }
    for (const std::span<const double> values : {steps, viscosity_steps}) {
        if (values.size() > keys_.window) {
            return restore_status::too_long;
        }
        for (const double value : values) {
            if (!(std::isfinite(value) && value >= 0.0)) {
                return restore_status::not_finite;
            }
        }
    }
    const std::size_t expected = keys_.viscosity_condition ? steps.size() : 0;
    if (viscosity_steps.size() != expected) {
        return restore_status::viscosity_count;
    }
    steps_.assign(steps.begin(), steps.end());
    nu_steps_.assign(viscosity_steps.begin(), viscosity_steps.end());
    return restore_status::ok;
}

double rule::estimate_of(std::span<const double> steps, double scale) {
    const double infinity = std::numeric_limits<double>::infinity();
    const std::size_t n = keys_.window;
    if (steps.size() < n) {
        return infinity;
    }
    for (const double value : steps) {
        if (!positive_and_finite(value)) {
            return infinity;
        }
    }
    for (std::size_t k = 0; k < n; ++k) {
        logs_[k] = elementary::log(steps[k]);
    }
    // The second kind: the reference's dot is OpenBLAS's, an order the core does not
    // copy; this one is the fixed blocks' (one block of the window).
    const double slope = reduce::blocked_dot(abscissa_, logs_, n) / abscissa_square_;
    const double rho = elementary::exp(slope);
    if (!(0.0 < rho && rho < 1.0)) {
        return infinity;
    }
    return steps[n - 1] * rho / (1.0 - rho) / scale;
}

bool rule::record(double step, double viscosity_step, double nu_scale) {
    recorded_ = true;
    push(steps_, keys_.window, step);
    last_.estimate = estimate_of(steps_, keys_.velocity_scale);
    last_.viscosity_estimate = std::numeric_limits<double>::quiet_NaN();
    last_.nu_scale = std::numeric_limits<double>::quiet_NaN();
    bool viscosity_ok = true;
    if (keys_.viscosity_condition) {
        last_.nu_scale = nu_scale;
        push(nu_steps_, keys_.window, viscosity_step);
        last_.viscosity_estimate = estimate_of(nu_steps_, nu_scale);
        viscosity_ok = last_.viscosity_estimate < keys_.iteration_error_tol;
    }
    return last_.estimate < keys_.iteration_error_tol && viscosity_ok;
}

bool rule::accept(const imbalance_summary& found) const {
    return found.worst < keys_.mass_imbalance_tol &&
           found.absolute_sum / keys_.flux_scale < keys_.iteration_error_tol &&
           std::fabs(found.signed_sum) < keys_.mass_imbalance_tol;
}

}  // namespace cfdcore::stopping
