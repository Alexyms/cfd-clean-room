// The stopping rule, the imbalance summary and the median on planted histories
// against their hand results: what needs no reference (ADR-017 D). Their parity is
// tests/test_core_flow.py's.

#include "cfdcore/stopping.hpp"

#include "cfdcore/reduce.hpp"

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cfdcore::stopping {
namespace {

constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr double nan_value = std::numeric_limits<double>::quiet_NaN();

std::uint64_t bits(double x) { return std::bit_cast<std::uint64_t>(x); }

rule_keys keys(bool e = false, std::size_t window = 5) {
    return {.velocity_scale = 0.5,
            .flux_scale = 2.0,
            .iteration_error_tol = 1e-6,
            .mass_imbalance_tol = 1e-9,
            .viscosity_condition = e,
            .window = window};
}

TEST(Rule, RefusesKeysThatAreNotFiniteAndPositiveOrAWindowBelowTwo) {
    EXPECT_TRUE(rule::check(keys()).empty());
    for (const double bad : {0.0, -1.0, nan_value, infinity}) {
        rule_keys k = keys();
        k.velocity_scale = bad;
        EXPECT_FALSE(rule::check(k).empty());
        k = keys();
        k.flux_scale = bad;
        EXPECT_FALSE(rule::check(k).empty());
        k = keys();
        k.iteration_error_tol = bad;
        EXPECT_FALSE(rule::check(k).empty());
        k = keys();
        k.mass_imbalance_tol = bad;
        EXPECT_FALSE(rule::check(k).empty());
        EXPECT_THROW(rule{k}, std::invalid_argument);
    }
    EXPECT_FALSE(rule::check(keys(false, 1)).empty());
    EXPECT_EQ(rule(keys()).version(), 3);
    EXPECT_EQ(rule(keys(true)).version(), 5);
}

TEST(Rule, TheWindowKeepsTheLastStepsOldestFirst) {
    rule r(keys());
    for (int k = 1; k <= 7; ++k) {
        r.record(static_cast<double>(k), 0.0, 0.0);
    }
    const std::vector<double> window(r.steps().begin(), r.steps().end());
    EXPECT_EQ(window, (std::vector<double>{3.0, 4.0, 5.0, 6.0, 7.0}));
    EXPECT_TRUE(r.viscosity_steps().empty());
    EXPECT_TRUE(r.stepped());
    r.reset();
    EXPECT_FALSE(r.stepped());
    EXPECT_TRUE(r.steps().empty());
}

TEST(Rule, AnEstimateNeedsAFullWindowOfPositiveFiniteStepsAndARateBelowOne) {
    rule r(keys());
    // Halving steps: the fit's rate is a half to rounding, and step rho / (1 - rho)
    // is the last step.
    const std::vector<double> halving{16.0, 8.0, 4.0, 2.0, 1.0};
    EXPECT_EQ(r.estimate_of(std::vector<double>(halving.begin(), halving.end() - 1), 0.5),
              infinity);
    EXPECT_NEAR(r.estimate_of(halving, 0.5), 1.0 / 0.5, 1e-14);
    for (const double bad : {0.0, -1.0, nan_value, infinity}) {
        std::vector<double> spoilt = halving;
        spoilt[2] = bad;
        EXPECT_EQ(r.estimate_of(spoilt, 0.5), infinity);
    }
    // A rate of exactly one, and one above it.
    EXPECT_EQ(r.estimate_of(std::vector<double>{1.0, 1.0, 1.0, 1.0, 1.0}, 0.5), infinity);
    EXPECT_EQ(r.estimate_of(std::vector<double>{1.0, 2.0, 4.0, 8.0, 16.0}, 0.5), infinity);
}

TEST(Rule, RecordAsksForTheImbalanceOnlyWhenAAndEHold) {
    rule r(keys(true));
    const std::vector<double> small{1e-6, 5e-7, 2.5e-7, 1.25e-7, 6.25e-8};
    bool asked = false;
    for (const double s : small) {
        asked = r.record(s * 1e-3, s * 1e-9, 1.5e-5);
    }
    EXPECT_TRUE(asked);
    EXPECT_EQ(bits(r.last().nu_scale), bits(1.5e-5));
    EXPECT_LT(r.last().estimate, 1e-6);
    EXPECT_LT(r.last().viscosity_estimate, 1e-6);
    // (e) failing alone: the viscosity's steps falling slowly.
    rule slow(keys(true));
    for (const double s : small) {
        asked = slow.record(s * 1e-3, 1e-9 * (1.0 - 1e-3 * s), 1.5e-5);
    }
    EXPECT_FALSE(asked);
    EXPECT_LT(slow.last().estimate, 1e-6);
    // (e) off: its readings NaN.
    rule off(keys());
    off.record(1e-3, 0.0, 0.0);
    EXPECT_TRUE(std::isnan(off.last().viscosity_estimate));
    EXPECT_TRUE(std::isnan(off.last().nu_scale));
}

TEST(Rule, AcceptIsConditionsBCAndD) {
    const rule r(keys());
    EXPECT_TRUE(r.accept({1e-10, 1e-7, -5e-10}));
    EXPECT_FALSE(r.accept({2e-9, 1e-7, -5e-10}));   // (b)
    EXPECT_FALSE(r.accept({1e-10, 3e-6, -5e-10}));  // (c): 3e-6 / 2 above 1e-6
    EXPECT_TRUE(r.accept({1e-10, 1.9e-6, -5e-10}));
    EXPECT_FALSE(r.accept({1e-10, 1e-7, -2e-9}));  // (d), in absolute value
    EXPECT_FALSE(r.accept({1e-10, 1e-7, 2e-9}));
    EXPECT_FALSE(r.accept({nan_value, 1e-7, 0.0}));
}

TEST(Rule, RestoreSeedsANewRuleAndRefusesInTheReferencesOrder) {
    rule r(keys(true));
    EXPECT_EQ(r.restore(std::vector<double>{1.0, 2.0}, std::vector<double>{3.0, 4.0}),
              restore_status::ok);
    EXPECT_EQ(r.steps()[1], 2.0);
    EXPECT_EQ(r.viscosity_steps()[0], 3.0);
    EXPECT_EQ(r.restore({}, {}), restore_status::already_stepped);
    rule fresh(keys(true));
    EXPECT_EQ(fresh.restore(std::vector<double>(6, 1.0), std::vector<double>(6, 1.0)),
              restore_status::too_long);
    EXPECT_EQ(fresh.restore(std::vector<double>{1.0, -1.0}, std::vector<double>{1.0, 1.0}),
              restore_status::not_finite);
    EXPECT_EQ(fresh.restore(std::vector<double>{1.0}, std::vector<double>{nan_value}),
              restore_status::not_finite);
    EXPECT_EQ(fresh.restore(std::vector<double>{1.0}, {}), restore_status::viscosity_count);
    EXPECT_FALSE(fresh.stepped());
    rule laminar(keys());
    EXPECT_EQ(laminar.restore(std::vector<double>{1.0}, std::vector<double>{1.0}),
              restore_status::viscosity_count);
    EXPECT_EQ(laminar.restore({}, {}), restore_status::ok);
    EXPECT_EQ(laminar.restore({}, {}), restore_status::ok);
}

TEST(Summary, IsTheLargestMagnitudeAndTheTwoPairwiseSums) {
    std::vector<double> imbalance(300);
    for (std::size_t i = 0; i < imbalance.size(); ++i) {
        imbalance[i] = (i % 3 == 0 ? -1.0 : 1.0) * (1.0 + 1e-3 * static_cast<double>(i));
    }
    std::vector<double> magnitude(imbalance.size());
    const imbalance_summary found = summarise(imbalance, magnitude);
    EXPECT_EQ(found.worst, 1.0 + 1e-3 * 299.0);
    EXPECT_EQ(bits(found.absolute_sum), bits(reduce::pairwise_sum(magnitude)));
    EXPECT_EQ(bits(found.signed_sum), bits(reduce::pairwise_sum(imbalance)));
    imbalance[17] = nan_value;
    EXPECT_TRUE(std::isnan(summarise(imbalance, magnitude).worst));
    std::vector<double> short_scratch(3);
    EXPECT_THROW(summarise(imbalance, short_scratch), std::invalid_argument);
}

TEST(Median, IsTheMiddleValueOrTheMeanOfTheTwo) {
    const std::vector<double> nu_t{3.0, 9.0, 1.0, 2.0, 7.0, 4.0};
    std::vector<double> scratch(3);
    EXPECT_EQ(viscosity_scale(0.5, nu_t, std::vector<std::size_t>{0, 2, 3}, scratch), 2.5);
    scratch.resize(4);
    EXPECT_EQ(viscosity_scale(0.5, nu_t, std::vector<std::size_t>{0, 1, 2, 3}, scratch), 3.0);
    // Ties.
    const std::vector<double> tied{2.0, 2.0, 2.0, 5.0};
    EXPECT_EQ(viscosity_scale(0.0, tied, std::vector<std::size_t>{0, 1, 2, 3}, scratch), 2.0);
    // A NaN gives a NaN, an infinity a value.
    std::vector<double> spoilt = nu_t;
    spoilt[3] = nan_value;
    EXPECT_TRUE(std::isnan(viscosity_scale(0.5, spoilt, std::vector<std::size_t>{0, 1, 2, 3},
                                           scratch)));
    spoilt[3] = infinity;
    EXPECT_EQ(viscosity_scale(0.0, spoilt, std::vector<std::size_t>{0, 1, 2, 3}, scratch), 6.0);
    std::vector<double> none;
    EXPECT_THROW(viscosity_scale(0.5, nu_t, std::vector<std::size_t>{}, none),
                 std::invalid_argument);
}

}  // namespace
}  // namespace cfdcore::stopping
