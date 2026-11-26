// The C library's logarithm and power (ADR-017 C; Alex, 2026-11-25, prompt 94).
//
// The reference's `np.log` and its cubes `x**3` are the only elementary
// functions its turbulence model and wall functions call on arrays, and its
// `c_mu**0.25`, `k**1.5` and `x**2` on Python's floats go to the C library's
// pow. NumPy runs the first two through the C library too, except on a processor
// with AVX-512, where it has loops of its own that can differ from it in the
// last bit (docs/reports/phase6_turbulence.md, section 4). The core calls the C
// library everywhere, and equals the reference bit for bit where NumPy does.
//
// Both are calls of a function in another source file, so the compiler cannot
// see an exponent and replace a power by the products it expands a known one to
// (x * x for 2.0, whose rounding is not pow's on every x).

#pragma once

namespace cfdcore::elementary {

// The natural logarithm, dimensionless in and out.
double log(double x);

// base to the exponent, as the C library's pow gives it.
double power(double base, double exponent);

// e to the x, as the C library's exp gives it: Python's `math.exp`, which the
// stopping rule's rate fit calls (prompt 95).
double exp(double x);

}  // namespace cfdcore::elementary
