#include "cfdcore/elementary.hpp"

#include <cmath>

namespace cfdcore::elementary {

double log(double x) { return std::log(x); }

double power(double base, double exponent) { return std::pow(base, exponent); }

}  // namespace cfdcore::elementary
