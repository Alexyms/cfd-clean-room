// The stopping rule of the steady solve (ADR-017 C, module `stopping`; ADR-011 G,
// ADR-012 E).
//
// A port of `src/stopping.py`: `ErrorEstimateRule` (conditions (a) to (e), the
// window of the rate fit, `restore`, the version), `viscosity_scale` and the
// solver's imbalance summary (`StaggeredSolver._imbalance_summary`). Every value is
// the reference's bit for bit (ADR-017 G, the first kind) but one: the rate fit's
// `x @ log(steps)`, which NumPy hands to OpenBLAS, whose order depends on the kernel
// it picks for the processor, is formed in `reduce`'s fixed blocks (the second
// kind), so an estimate is compared with the reference's on the same window to a
// relative difference. Its other dot product, `x @ x`, is a sum of multiples of a
// quarter, exact in any order at the window's length, and its logarithms and its
// exponential are the C library's (elementary.hpp), as NumPy's and Python's are.
//
// The rule knows nothing of the grid, so it has no direction to be written per.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace cfdcore::stopping {

// The rule's numbers (`ErrorEstimateRule`'s arguments and `RATE_WINDOW`).
//
// velocity_scale:      the velocity (a)'s estimate is relative to, m/s
// flux_scale:          the mass flux (c) is relative to, kg/s per unit depth
// iteration_error_tol: the bound of (a), (c) and (e), dimensionless
// mass_imbalance_tol:  the bound of (b) and (d), kg/s per unit depth
// viscosity_condition: whether (e) is applied
// window:              the steps the rate fit reads, at least two
// version_without_e,
// version_with_e:      the numbers `version` returns with (e) off and on, at least one:
//                      the reference's `RULE_VERSION_WITHOUT_E` and `RULE_VERSION_WITH_E`,
//                      handed in as the window is, so that the core holds no copy
struct rule_keys {
    double velocity_scale = 0.0;
    double flux_scale = 0.0;
    double iteration_error_tol = 0.0;
    double mass_imbalance_tol = 0.0;
    bool viscosity_condition = false;
    std::size_t window = 0;
    int version_without_e = 0;
    int version_with_e = 0;
};

// One field's per-cell mass imbalance read three ways (`ImbalanceSummary`), kg/s
// per unit depth: the largest absolute value, the sum of the absolute values and
// the signed sum.
struct imbalance_summary {
    double worst = 0.0;
    double absolute_sum = 0.0;
    double signed_sum = 0.0;
};

// The summary of a cell field of imbalances (`_imbalance_summary`): `np.max` of the
// absolute values (NaN where one is NaN), and the two sums in NumPy's pairwise order
// over the whole C-contiguous field (`reduce::pairwise_sum`).
//
// imbalance: the cells' imbalances, kg/s per unit depth, the whole field
// magnitude: scratch, as long; overlaps no input
//
// Throws std::invalid_argument for arrays of different lengths.
imbalance_summary summarise(std::span<const double> imbalance, std::span<double> magnitude);

// Condition (e)'s scale (`viscosity_scale`): nu plus the median of nu_t over the
// FLUID cells, the median as NumPy's (a selection; for an even count the mean of the
// two middle values, their sum over two; NaN where a value is NaN).
//
// nu:      air's kinematic viscosity, m^2/s
// nu_t:    the eddy viscosity of the iterate, m^2/s, a cell field
// fluid:   the FLUID cells' indices into it, ascending, at least one
// scratch: as long as `fluid`; overlaps no input
//
// Throws std::invalid_argument for an empty `fluid` or a scratch of another length.
double viscosity_scale(double nu, std::span<const double> nu_t,
                       std::span<const std::size_t> fluid, std::span<double> scratch);

// Why `restore` refuses a window, in the reference's order. `no_rule` is a window
// handed to a solve that has no rule (the velocity step's), which seeds nothing.
enum class restore_status { ok, already_stepped, too_long, not_finite, viscosity_count, no_rule };

// The rule's readings of one outer iteration: (a)'s estimate, (e)'s and its scale.
// The two of (e) are NaN with it off; an estimate is +inf where it has none.
struct reading {
    double estimate = 0.0;
    double viscosity_estimate = 0.0;
    double nu_scale = 0.0;
};

// The rule (`ErrorEstimateRule`), its window held oldest first. Every array is
// allocated at construction; an update allocates nothing.
class rule {
public:
    // Why `keys` cannot make a rule, or an empty view when they can: a scale or a
    // tolerance that is not finite and positive, a window below two, or a version
    // below one.
    static std::string_view check(const rule_keys& keys);

    // Builds the rule from keys `check` accepts. Throws std::invalid_argument for
    // keys it does not.
    explicit rule(const rule_keys& keys);

    // The version of the conditions applied (`ErrorEstimateRule.version`): the
    // keys' `version_with_e` with (e) on, their `version_without_e` with it off.
    int version() const;

    // The window, oldest first (`window`): the velocity steps, m/s, and with (e)
    // on the viscosity steps, m^2/s; the second empty with (e) off.
    std::span<const double> steps() const { return steps_; }
    std::span<const double> viscosity_steps() const { return nu_steps_; }

    // Seeds a rule that has taken no step with a window another kept (`restore`):
    // the steps, oldest first, at most `window` of them, each finite and not
    // negative; the viscosity steps as many with (e) on and none with it off.
    // Nothing is seeded where the status is not ok.
    restore_status restore(std::span<const double> steps,
                           std::span<const double> viscosity_steps);

    // Records one outer iteration (`update` up to its call of `imbalance`): the
    // velocity step, and with (e) on the viscosity step and its scale, which the
    // caller has checked is finite and positive. Returns whether (a) and, with it
    // on, (e) hold, so that the imbalance is asked for; `last` holds the readings.
    // The viscosity step and scale are not read with (e) off.
    bool record(double step, double viscosity_step, double nu_scale);

    // Conditions (b), (c) and (d) on the imbalance of the iterate just recorded.
    bool accept(const imbalance_summary& found) const;

    // The readings of the last `record`.
    const reading& last() const { return last_; }

    // Empties the window and the readings, as a new rule is: a new solve's.
    void reset();

    // Whether the rule holds a step, recorded or seeded: `restore` refuses it then.
    bool stepped() const { return recorded_ || !steps_.empty(); }

    // The estimate of one window over its scale (`_estimate_of`): +inf with fewer
    // than `window` steps, a step that is zero or not finite, or a rate fit whose
    // factor is not strictly between 0 and 1.
    double estimate_of(std::span<const double> steps, double scale);

    const rule_keys& keys() const { return keys_; }

private:
    static void push(std::vector<double>& window, std::size_t length, double value);

    rule_keys keys_;
    std::vector<double> steps_;
    std::vector<double> nu_steps_;
    std::vector<double> abscissa_;  // the window's centred positions, k - (n - 1) / 2
    std::vector<double> logs_;
    double abscissa_square_ = 0.0;  // x @ x
    reading last_{};
    bool recorded_ = false;
};

}  // namespace cfdcore::stopping
