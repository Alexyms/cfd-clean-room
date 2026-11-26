"""Planted defects of prompt 95: each a change of the port the tests must catch.

Each plant names the file it changes and the replacements it makes there, each
text found exactly once. ``probe95.py plants`` builds each in a scratch tree and
runs CTest and the parity module against it (``tests/test_core_flow.py``, and the
turbulence module's section 0 tests for the writes prompt 94 left). A plant is
caught when a test fails; one that cannot be caught by any net is an equivalent
mutant, listed in EQUIVALENT with the reason.
"""

Plant = tuple[str, str, str, list[tuple[str, str]]]

STOPPING = "core/src/stopping.cpp"
FLOW = "core/src/flow.cpp"
TURBULENCE = "core/src/turbulence.cpp"
WALLS = "core/src/walls.cpp"
MAPPING = "tests/core_mapping.py"
DRIVER = "tests/core_driver.py"


def one(name: str, description: str, path: str, old: str, new: str) -> Plant:
    """A plant of one replacement."""
    return (name, description, path, [(old, new)])


# The prompt's list, in its order.
PROMPT: list[Plant] = [
    (
        "window_one_short",
        "the rule's window one step short of RATE_WINDOW",
        MAPPING,
        [
            (
                "    return _cfdcore.StoppingRule2(\n        RATE_WINDOW,",
                "    return _cfdcore.StoppingRule2(\n        RATE_WINDOW - 1,",
            ),
            (
                '            "error_estimate",\n            RATE_WINDOW,',
                '            "error_estimate",\n            RATE_WINDOW - 1,',
            ),
        ],
    ),
    one(
        "abscissa_not_centred",
        "the rate fit's abscissa not centred",
        STOPPING,
        "        abscissa_[k] = static_cast<double>(k) - centre;",
        "        abscissa_[k] = static_cast<double>(k) + 0.0 * centre;",
    ),
    one(
        "c_against_inflow_velocity",
        "condition (c) against the inflow's velocity in place of the flux scale",
        MAPPING,
        "            boundary.get_max_boundary_velocity(),\n            flux_scale,",
        "            boundary.get_max_boundary_velocity(),\n            boundary.get_max_boundary_velocity(),",
    ),
    one(
        "d_without_abs",
        "condition (d) without the absolute value",
        STOPPING,
        "std::fabs(found.signed_sum) < keys_.mass_imbalance_tol",
        "found.signed_sum < keys_.mass_imbalance_tol",
    ),
    one(
        "median_over_live",
        "the median of every live cell instead of the FLUID cells",
        FLOW,
        "        if (fluid_[i] != lanes::none) {\n            fluid_cells_.push_back(i);",
        "        if (live_[i] != lanes::none) {\n            fluid_cells_.push_back(i);",
    ),
    one(
        "relaxation_swapped",
        "the relaxation's weights swapped",
        TURBULENCE,
        "        out[i] = in.keep * in.old[i] + in.alpha * in.stepped[i];",
        "        out[i] = in.alpha * in.old[i] + in.keep * in.stepped[i];",
    ),
    one(
        "outlet_from_own_face",
        "an outlet's extrapolation from its own face",
        WALLS,
        "                    copies_[d].push_back({f, side == 0 ? f + stride : f - stride});",
        "                    copies_[d].push_back({f, f});",
    ),
    one(
        "cap_on_converged",
        "the cap hit counted on a converged correction",
        FLOW,
        "    record.reached_cap = result.reached_cap;",
        "    record.reached_cap = !result.reached_cap;",
    ),
    one(
        "stop_on_capped",
        "the stop allowed on a capped correction under velocity_step",
        FLOW,
        "        return record.residual < convergence_tol_ && !record.reached_cap;",
        "        return record.residual < convergence_tol_;",
    ),
    one(
        "checkpoint_late",
        "a checkpoint one iteration late",
        DRIVER,
        "                n = min(n, cadence - iteration % cadence)",
        "                n = min(n, cadence - iteration % cadence + 1)",
    ),
    one(
        "resume_restarts_window",
        "the resume restarting the window",
        DRIVER,
        "        if resume is not None:\n            core.restore(resume.velocity_steps, resume.viscosity_steps)",
        "        if resume is not None:\n            pass",
    ),
]

# Prompt 94's survivor, carried by section 0.
CARRIED: list[Plant] = [
    one(
        "least_over_solid_too",
        "the least value of a failure read over the SOLID cells too",
        TURBULENCE,
        "        if (solid_[i]) {\n            continue;\n        }\n"
        "        if (!positive_and_finite(q[i])) {",
        "        if (!solid_[i] && !positive_and_finite(q[i])) {",
    ),
]

# The builder's, one on each other new piece of arithmetic or mapping.
OWN: list[Plant] = [
    one(
        "residual_scale_doubled",
        "the residual divided by twice its scale",
        MAPPING,
        "        max(reference_velocity, ZERO_SCALE),",
        "        2.0 * max(reference_velocity, ZERO_SCALE),",
    ),
    one(
        "python_max_keeps_a_later_nan",
        "the velocity step's max taking a NaN that comes second, as Python's max does not",
        FLOW,
        "        if (values[i] > m) {",
        "        if (!(values[i] <= m)) {",
    ),
    one(
        "effective_viscosity_unmasked",
        "the effective viscosity read in SOLID cells too",
        FLOW,
        "        out[i] = in.mu + in.rho * lanes::select(in.live[i], in.nu_t[i], 0.0);",
        "        out[i] = in.mu + in.rho * in.nu_t[i];",
    ),
    one(
        "signed_sum_of_magnitudes",
        "the signed imbalance summed from the magnitudes",
        STOPPING,
        "    return {worst, reduce::pairwise_sum(magnitude), reduce::pairwise_sum(imbalance)};",
        "    return {worst, reduce::pairwise_sum(magnitude), reduce::pairwise_sum(magnitude)};",
    ),
    one(
        "viscosity_step_over_fluid",
        "the viscosity step read over the FLUID cells, not every live one",
        FLOW,
        "    record.viscosity_step = largest_change(nu_t_next_, nu_t_, live_);",
        "    record.viscosity_step = largest_change(nu_t_next_, nu_t_, fluid_);",
    ),
    one(
        "estimate_from_oldest",
        "the estimate formed from the window's oldest step",
        STOPPING,
        "    return steps[n - 1] * rho / (1.0 - rho) / scale;",
        "    return steps[0] * rho / (1.0 - rho) / scale;",
    ),
    one(
        "median_upper_only",
        "the median of an even count the upper middle value alone",
        STOPPING,
        "        middle[0] = *std::max_element(scratch.begin(), scratch.begin() + half);",
        "        middle[0] = scratch[half];",
    ),
    one(
        "imbalance_without_estimates",
        "the imbalance read and the stop allowed without (a) and (e)",
        FLOW,
        "    if (!estimates) {\n        return false;\n    }\n    std::array<std::span<const double>, D> velocity",
        "    (void)estimates;\n    std::array<std::span<const double>, D> velocity",
    ),
    one(
        "warning_at_every_cap",
        "the cap's warning logged at every capped correction",
        DRIVER,
        "            if self.pressure_cap_hits == 1:",
        "            if self.pressure_cap_hits >= 1:",
    ),
    one(
        "chunks_with_on_iteration",
        "the chunks not cut to one outer iteration while on_iteration is given",
        DRIVER,
        "            if on_iteration is not None:\n                n = 1\n",
        "            if on_iteration is not None:\n                pass\n",
    ),
    one(
        "diverged_state_kept",
        "a non-finite solve's state kept",
        DRIVER,
        "        if not self._finite(u, v, p, state):\n            solver_logger.warning(",
        "        if False:\n            solver_logger.warning(",
    ),
    one(
        "nu_is_mu",
        "condition (e)'s scale from the dynamic viscosity in place of the kinematic",
        MAPPING,
        "        config.mu / config.rho,\n        1.0 if spec is None",
        "        config.mu,\n        1.0 if spec is None",
    ),
    one(
        "relaxation_off",
        "the eddy viscosity's relaxation weight one",
        MAPPING,
        "        1.0 if spec is None else spec.alpha_turbulence,",
        "        1.0,",
    ),
    one(
        "conditions_unchecked",
        "the formed conditions not checked before the k-epsilon step",
        FLOW,
        "    if (checked.status != turbulence::condition_status::ok) {\n        failure.kind = failure::conditions;",
        "    if (false) {\n        failure.kind = failure::conditions;",
    ),
    one(
        "failing_cap_dropped",
        "a failing iteration's capped correction not counted",
        DRIVER,
        "                if failing is not None:\n                    self._count_correction(",
        "                if False:\n                    self._count_correction(",
    ),
]

# Plants no net can tell from the code, with the reason.
EQUIVALENT: dict[str, str] = {}

PLANTS: list[Plant] = [("control", "nothing", FLOW, []), *PROMPT, *CARRIED, *OWN]

# The turbulence module's tests of section 0, which the carried and the section 0
# plants also face.
SECTION_0 = (
    "least_value_of_a_failure or relaxation_is_the_solvers or outlets_extrapolation"
)
