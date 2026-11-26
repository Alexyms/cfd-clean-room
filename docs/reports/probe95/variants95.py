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
BINDINGS = "core/python/bindings.cpp"


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
        "            config.mu / config.rho,\n            1.0 if spec is None",
        "            config.mu,\n            1.0 if spec is None",
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
        '        self._count_correction(at, iterations, bool(failing["reached_cap"]))',
        "        pass",
    ),
]

# Round 1 (prompt 95b): the tester's plants that no test caught at the head of
# round 0 (test 95: B1, B2, S1, S2; `t_stage_seconds_dropped` is left out, which
# times what nothing can compare), then one on each piece round 1 added.
ROUND1: list[Plant] = [
    # -- B1: the stop inside the solve ------------------------------------------
    one(
        "t_accept_always",
        "the solve stops once (a) and (e) hold, conditions (b), (c), (d) not read",
        FLOW,
        "    return r.accept(stopping::summarise(imbalance_, magnitude_));",
        "    (void)r.accept(stopping::summarise(imbalance_, magnitude_));\n    return true;",
    ),
    one(
        "t_imbalance_not_formed",
        "the stop's imbalance not recomputed from the corrected faces",
        FLOW,
        "    modules_.corrector->mass_imbalance(velocity, imbalance_);",
        "    (void)velocity;",
    ),
    one(
        "t_velocity_tol_doubled",
        "the velocity step's bound doubled",
        MAPPING,
        '        rule = ("velocity_step", config.convergence_tol)',
        '        rule = ("velocity_step", 2.0 * config.convergence_tol)',
    ),
    # -- B2: the outlets inside the solve -----------------------------------------
    one(
        "t_outlets_not_extrapolated",
        "the outlets' extrapolation left out of the outer iteration",
        FLOW,
        "        modules_.outlets->extrapolate([&] {",
        "        if (false) modules_.outlets->extrapolate([&] {",
    ),
    one(
        "t_outlets_no_top",
        "the top edge's outlets not handed to the outer iteration",
        MAPPING,
        "    outlets = boundary.pressure_outlets()\n    return _cfdcore.OutletFaces2(\n"
        "        core_grid(mesh) if grid is None else grid,\n"
        '        [outlets[edge].is_outlet for edge in ("left", "right", "bottom", "top")],',
        "    outlets = boundary.pressure_outlets()\n    return _cfdcore.OutletFaces2(\n"
        "        core_grid(mesh) if grid is None else grid,\n"
        '        [outlets[edge].is_outlet & (edge != "top") for edge in ("left", "right", "bottom", "top")],',
    ),
    one(
        "t_outlets_no_left",
        "the left edge's outlets not handed to the outer iteration",
        MAPPING,
        "    outlets = boundary.pressure_outlets()\n    return _cfdcore.OutletFaces2(\n"
        "        core_grid(mesh) if grid is None else grid,\n"
        '        [outlets[edge].is_outlet for edge in ("left", "right", "bottom", "top")],',
        "    outlets = boundary.pressure_outlets()\n    return _cfdcore.OutletFaces2(\n"
        "        core_grid(mesh) if grid is None else grid,\n"
        '        [outlets[edge].is_outlet & (edge != "left") for edge in ("left", "right", "bottom", "top")],',
    ),
    one(
        "t_outlets_no_right",
        "the right edge's outlets not handed to the outer iteration",
        MAPPING,
        "    outlets = boundary.pressure_outlets()\n    return _cfdcore.OutletFaces2(\n"
        "        core_grid(mesh) if grid is None else grid,\n"
        '        [outlets[edge].is_outlet for edge in ("left", "right", "bottom", "top")],',
        "    outlets = boundary.pressure_outlets()\n    return _cfdcore.OutletFaces2(\n"
        "        core_grid(mesh) if grid is None else grid,\n"
        '        [outlets[edge].is_outlet & (edge != "right") for edge in ("left", "right", "bottom", "top")],',
    ),
    one(
        "t_outlets_no_bottom",
        "the bottom edge's outlets not handed to the outer iteration",
        MAPPING,
        "    outlets = boundary.pressure_outlets()\n    return _cfdcore.OutletFaces2(\n"
        "        core_grid(mesh) if grid is None else grid,\n"
        '        [outlets[edge].is_outlet for edge in ("left", "right", "bottom", "top")],',
        "    outlets = boundary.pressure_outlets()\n    return _cfdcore.OutletFaces2(\n"
        "        core_grid(mesh) if grid is None else grid,\n"
        '        [outlets[edge].is_outlet & (edge != "bottom") for edge in ("left", "right", "bottom", "top")],',
    ),
    # -- S1: the failure paths -------------------------------------------------------
    one(
        "t_first_ignored",
        "a failure's outer iteration counted from the chunk's start, not the solve's",
        FLOW,
        "        failure.iteration = first + static_cast<std::int64_t>(n);",
        "        (void)first;\n        failure.iteration = static_cast<std::int64_t>(n);",
    ),
    one(
        "t_first_not_passed",
        "each chunk told it starts at outer iteration 0",
        DRIVER,
        "            chunk = core.solve_steady(n, iteration)",
        "            chunk = core.solve_steady(n, 0)",
    ),
    one(
        "t_faces_unchecked",
        "the corrected faces not checked finite before the k-epsilon step",
        FLOW,
        "        if (!turbulence::all_finite(velocity[c])) {",
        "        if (false && !turbulence::all_finite(velocity[c])) {",
    ),
    one(
        "t_eddy_failures_swapped",
        "a non-finite eddy viscosity reported as a negative one, and the reverse",
        FLOW,
        "            not_finite ? failure::eddy_viscosity_not_finite : failure::eddy_viscosity_negative;",
        "            not_finite ? failure::eddy_viscosity_negative : failure::eddy_viscosity_not_finite;",
    ),
    one(
        "t_nu_scale_unchecked",
        "condition (e)'s scale not refused when it is not positive and finite",
        FLOW,
        "    if (coupled() && !positive_and_finite(nu_scale)) {",
        "    if (false && coupled() && !positive_and_finite(nu_scale)) {",
    ),
    one(
        "t_eps_reported_as_k",
        "a positivity failure of eps reported as k's",
        FLOW,
        "        case turbulence::step_status::eps_not_positive:\n            failure.kind = failure::eps_not_positive;",
        "        case turbulence::step_status::eps_not_positive:\n            failure.kind = failure::k_not_positive;",
    ),
    one(
        "t_row_column_swapped",
        "a positivity failure's row and column swapped",
        BINDINGS,
        '            out["row"] = f.bad.first / self.grid->cells(0);\n'
        '            out["column"] = f.bad.first % self.grid->cells(0);',
        '            out["column"] = f.bad.first / self.grid->cells(0);\n'
        '            out["row"] = f.bad.first % self.grid->cells(0);',
    ),
    one(
        "t_conditions_failure_uncounted",
        "a refused condition's correction not handed back for the count",
        BINDINGS,
        "                                  kind == failure::conditions || kind == failure::nothing_moves ||",
        "                                  kind == failure::nothing_moves ||",
    ),
    one(
        "t_positivity_iteration_off",
        "a positivity failure's message naming the iteration after",
        MAPPING,
        "            f\"outer iteration {failure['iteration']}: {base}\", minimum",
        "            f\"outer iteration {int(failure['iteration']) + 1}: {base}\", minimum",
    ),
    # -- S2: callbacks, records, attributes and log lines -----------------------------
    one(
        "t_no_checkpoint_with_on_iteration",
        "no checkpoint while on_iteration is given",
        DRIVER,
        "        cadence = interval if on_checkpoint is not None else None",
        "        cadence = interval if on_checkpoint is not None and on_iteration is None else None",
    ),
    one(
        "t_on_iteration_v_is_u",
        "on_iteration's v the u component",
        DRIVER,
        '                v=v_c,\n                p=fields["p"],',
        '                v=u_c,\n                p=fields["p"],',
    ),
    one(
        "t_on_iteration_products",
        "on_iteration's pressure products the iterations",
        DRIVER,
        "                pressure_products=pressure_products,",
        "                pressure_products=pressure_iterations,",
    ),
    one(
        "t_on_iteration_iterations",
        "on_iteration's pressure iterations the products",
        DRIVER,
        "                pressure_iterations=pressure_iterations,",
        "                pressure_iterations=pressure_products,",
    ),
    one(
        "t_products_are_iterations",
        "a chunk's pressure products the iterations",
        BINDINGS,
        "        column<std::int64_t>(records, n, [](const record& r) { return r.pressure_products; });",
        "        column<std::int64_t>(records, n, [](const record& r) { return r.pressure_iterations; });",
    ),
    one(
        "t_cap_hits_not_carried",
        "a resumed solve's cap hits counted from zero",
        DRIVER,
        "        self.pressure_cap_hits = 0 if resume is None else resume.pressure_cap_hits",
        "        self.pressure_cap_hits = 0",
    ),
    one(
        "t_turbulence_state_dropped",
        "the solver's turbulence_state left None after a solve",
        DRIVER,
        "        self.turbulence_state = state\n        p = np.ascontiguousarray(p)",
        "        self.turbulence_state = None\n        p = np.ascontiguousarray(p)",
    ),
    one(
        "t_face_velocities_at_rest",
        "the solver's face_velocities left at rest after a solve",
        DRIVER,
        "        self.face_velocities = FaceVelocities.copy_of(u, v)",
        "        self.face_velocities = FaceVelocities.copy_of(np.zeros_like(u), np.zeros_like(v))",
    ),
    one(
        "t_no_log_at_stop",
        "no residual line logged at the stop",
        DRIVER,
        "                if at % 50 == 0 or stop:",
        "                if at % 50 == 0:",
    ),
    # -- round 1's own pieces ---------------------------------------------------------
    one(
        "fixed_viscosity_unchecked",
        "a fixed viscosity not read when the solver is built",
        FLOW,
        "    if (input.mode == viscosity_mode::fixed) {\n        switch (m.predictor->check_viscosity(input.mu_eff)) {",
        "    if (false) {\n        switch (m.predictor->check_viscosity(input.mu_eff)) {",
    ),
    one(
        "fixed_viscosity_positive_unchecked",
        "a fixed viscosity that is not positive accepted",
        FLOW,
        '            case momentum::viscosity_status::not_positive:\n                return "mu_eff must be positive in every non-SOLID cell";',
        "            case momentum::viscosity_status::not_positive:\n                break;",
    ),
    one(
        "fixed_viscosity_prefix_kept",
        "a fixed viscosity's refusal handed on with the binding's prefix",
        MAPPING,
        '        if text.startswith("Flow2: mu_eff must be "):',
        "        if False:",
    ),
    one(
        "versions_swapped_in_the_rule",
        "the rule's two versions swapped where its keys are handed in",
        MAPPING,
        "        RATE_WINDOW,\n        RULE_VERSION_WITHOUT_E,\n        RULE_VERSION_WITH_E,\n        velocity_scale,",
        "        RATE_WINDOW,\n        RULE_VERSION_WITH_E,\n        RULE_VERSION_WITHOUT_E,\n        velocity_scale,",
    ),
    one(
        "versions_swapped_in_the_flow",
        "the flow's two versions swapped where its rule is handed in",
        MAPPING,
        "            RATE_WINDOW,\n            RULE_VERSION_WITHOUT_E,\n            RULE_VERSION_WITH_E,\n            boundary.get_max_boundary_velocity(),",
        "            RATE_WINDOW,\n            RULE_VERSION_WITH_E,\n            RULE_VERSION_WITHOUT_E,\n            boundary.get_max_boundary_velocity(),",
    ),
    one(
        "version_hardcoded",
        "the rule's version the reference's two numbers, not the keys'",
        STOPPING,
        "    return keys_.viscosity_condition ? keys_.version_with_e : keys_.version_without_e;",
        "    return keys_.viscosity_condition ? 5 : 3;",
    ),
    one(
        "version_zero_allowed",
        "a version of zero accepted",
        STOPPING,
        "    if (keys.version_without_e < 1 || keys.version_with_e < 1) {",
        "    if (keys.version_without_e < 0 || keys.version_with_e < 0) {",
    ),
    one(
        "restore_no_rule_as_too_long",
        "a window handed to a solve with no rule reported as too long",
        FLOW,
        "                                                        : stopping::restore_status::no_rule;",
        "                                                        : stopping::restore_status::too_long;",
    ),
    one(
        "failing_residual_dropped",
        "a refused scale's residual not appended before the raise",
        DRIVER,
        "        self.residual_history.append(float(residual))\n        if on_iteration is not None:",
        "        if on_iteration is not None:",
    ),
    one(
        "failing_on_iteration_dropped",
        "a refused scale's on_iteration not called before the raise",
        DRIVER,
        "        self.residual_history.append(float(residual))\n        if on_iteration is not None:",
        "        self.residual_history.append(float(residual))\n        if False:",
    ),
    one(
        "failing_residual_not_returned",
        "the chunk's failing record holding no residual for a refused scale",
        BINDINGS,
        '        held["residual"] = kind == failure::nu_scale ? nb::object(nb::float_(r.residual))\n                                                     : nb::none();',
        '        held["residual"] = nb::none();',
    ),
    one(
        "fallback_warning_dropped",
        "the fall-back warning of the k-epsilon step not logged",
        DRIVER,
        "        if k_fallbacks or eps_fallbacks:",
        "        if False:",
    ),
    one(
        "converged_warning_dropped",
        "the cap warning of the k-epsilon step's solves not logged",
        DRIVER,
        "        if not converged:\n            spec = self._config.turbulence",
        "        if False:\n            spec = self._config.turbulence",
    ),
    one(
        "failing_warnings_dropped",
        "the k-epsilon step's warnings not logged for a step that then fails its signs",
        DRIVER,
        '        self._warn_of_the_step(\n            int(failing["k_fallbacks"]),\n            int(failing["eps_fallbacks"]),\n            bool(failing["solves_converged"]),\n        )\n',
        "",
    ),
    one(
        "fallbacks_not_recorded",
        "the step's fall-backs of k not carried out of the core",
        FLOW,
        "    record.k_fallbacks = result.solves[0].fallbacks;",
        "    record.k_fallbacks = 0;",
    ),
    one(
        "solves_converged_not_recorded",
        "the step's convergence not carried out of the core",
        FLOW,
        "    record.solves_converged = result.solves[0].converged && result.solves[1].converged;",
        "    record.solves_converged = true;",
    ),
]

# Plants no net can tell from the code, with the reason.
EQUIVALENT: dict[str, str] = {
    "t_rate_at_one_allowed": (
        "a fitted rate of exactly one divides by zero and gives the infinite "
        "estimate either way (test 95: 400 fuzzed histories, none differ)"
    ),
}

PLANTS: list[Plant] = [
    ("control", "nothing", FLOW, []),
    *PROMPT,
    *CARRIED,
    *OWN,
    *ROUND1,
    one(
        "t_rate_at_one_allowed",
        "a fitted rate of exactly one given an estimate",
        STOPPING,
        "    if (!(0.0 < rho && rho < 1.0)) {",
        "    if (!(0.0 < rho && rho <= 1.0)) {",
    ),
]

# The turbulence module's tests of section 0, which the carried and the section 0
# plants also face.
SECTION_0 = (
    "least_value_of_a_failure or relaxation_is_the_solvers or outlets_extrapolation"
)
