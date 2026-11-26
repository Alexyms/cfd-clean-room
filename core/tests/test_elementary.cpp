// The elementary functions the turbulence model and the wall functions call
// (elementary.hpp): the C library's, whatever the compiler would make of a known
// exponent. Their parity with the reference is tests/test_core_turbulence.py's.

#include "cfdcore/elementary.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace cfdcore::elementary {
namespace {

constexpr double infinity = std::numeric_limits<double>::infinity();

TEST(Elementary, TheFunctionsAreTheCLibrarys) {
    for (const double x : {0.3, 1.0, 7.5, 1e-9, 3e8}) {
        EXPECT_EQ(log(x), std::log(x));
        // A volatile exponent keeps the reference call from being folded to a
        // product, which is what the function under test must not be either.
        for (volatile double y : {0.25, 1.5, 2.0, 3.0}) {
            EXPECT_EQ(power(x, y), std::pow(x, y));
        }
    }
}

TEST(Elementary, TheLogarithmHasItsKnownValues) {
    EXPECT_EQ(log(1.0), 0.0);
    EXPECT_DOUBLE_EQ(log(std::exp(1.0)), 1.0);
    EXPECT_DOUBLE_EQ(log(1e10), 10.0 * std::log(10.0));
    EXPECT_EQ(log(0.0), -infinity);
    EXPECT_EQ(log(infinity), infinity);
    EXPECT_TRUE(std::isnan(log(-1.0)));
    EXPECT_TRUE(std::isnan(log(std::numeric_limits<double>::quiet_NaN())));
}

TEST(Elementary, ThePowerHasItsKnownValues) {
    EXPECT_EQ(power(2.0, 10.0), 1024.0);
    EXPECT_EQ(power(7.0, 0.0), 1.0);
    EXPECT_EQ(power(0.0, 1.5), 0.0);
    EXPECT_EQ(power(1.0, 1e300), 1.0);
    EXPECT_EQ(power(4.0, 0.5), 2.0);
    EXPECT_DOUBLE_EQ(power(9.0, 1.5), 27.0);
    EXPECT_EQ(power(10.0, 400.0), infinity);
    EXPECT_TRUE(std::isnan(power(-8.0, 0.25)));
}

// The exponents of the model: a power of a variant's constant, a speed's square,
// a cube and k^(3/2), each the library's on a spread of bases over twenty decades.
TEST(Elementary, TheModelsExponentsAreTheCLibrarysOverASpreadOfBases) {
    double x = 1e-12;
    for (int i = 0; i < 4000; ++i, x *= 1.0123) {
        for (volatile double y : {0.25, 1.5, 2.0, 3.0}) {
            ASSERT_EQ(power(x, y), std::pow(x, y)) << x << " ^ " << y;
        }
        ASSERT_EQ(log(x), std::log(x)) << x;
    }
    EXPECT_GT(x, 1e8);
}

}  // namespace
}  // namespace cfdcore::elementary
