"""Round 1's planted defects (prompt 91b): each a list of edits to a copy of ``core/``.

Read by ``probe91.py plants-round1``. An edit is ``(file, old, new)``: the exact
text ``old`` replaced by ``new``, found exactly once, or the plant is refused
(``probe90.plant_sources``). Four groups:

- the breach and the refusal through the march (review 91 B1, test 91 B1), the
  within-spec rule (test B2), the restore of the recovery state (test B3);
- the tester's other plants of test 91, which the prompt asks to be run again
  (``results/tester91/plant.py``, with the anchors the binding's changes moved
  updated), and three more of the march's loop;
- one plant per refusal of the binding, each removing it (review 91 S4);
- one plant per guard of the plan, time and count inputs (review 91 S5).

The builder's sixteen of ``variants91.py`` are run by ``probe91.py plants``.
"""

CHANNELS = "src/channels.cpp"
ARRIVAL = "src/arrival.cpp"
ARRIVAL_H = "include/cfdcore/arrival.hpp"
RECOVERY = "src/recovery.cpp"
MARCH = "src/march.cpp"
BINDING = "python/bindings.cpp"

Edit = tuple[str, str, str]

# The march's breach and refusal branches, as quoted by review 91 B1.
BREACH_BRANCH = (
    "            if (refusal.code == transport::status::breach) {\n"
    "                r.reason = stop::breach;\n"
)
REFUSED_BRANCH = (
    "            if (!refusal.ok()) {\n                r.reason = stop::refused;"
)
WITHIN_LOOP = (
    "        for (std::size_t m = 0; m < s; ++m) {\n"
    "            within = within && channel_readings[k * s + m] < keys_.thresholds[k];"
)
SENSOR_TIME = "            if (readings[k * s + m] <= keys_.target[k] && defined && std::isnan(when)) {"
MEAN_TIME = (
    "        if (means[k] <= keys_.target[k] && defined && std::isnan(mean_time_[k])) {"
)
RESTORE_RECOVERY = (
    "        recovery_->restore(state.last_above, state.sensor_time, state.mean_time,\n"
    "                           state.class_since, state.room_since, state.recovery_t);"
)
SPACING_LOOP_CHECK = "        if (v->shape(0) != n) {"


def _off(file: str, condition_line: str, disabled_line: str) -> list[Edit]:
    return [(file, condition_line, disabled_line)]


PLANTS: list[tuple[str, str, list[Edit]]] = [
    # --- the march's breach and refusal (B1) -----------------------------------
    (
        "breach_ignored",
        "a breaching step is not a stop: the march carries on",
        [
            (
                MARCH,
                "            if (refusal.code == transport::status::breach) {",
                "            if (false && refusal.code == transport::status::breach) {",
            ),
            (
                MARCH,
                "            if (!refusal.ok()) {\n                r.reason = stop::refused;",
                "            if (!refusal.ok() && refusal.code != transport::status::breach) {\n"
                "                r.reason = stop::refused;",
            ),
        ],
    ),
    (
        "refusal_ignored",
        "a refused step is not a stop: the march carries on",
        [
            (
                MARCH,
                REFUSED_BRANCH,
                "            if (false && !refusal.ok()) {\n                r.reason = stop::refused;",
            )
        ],
    ),
    (
        "breach_step_counted",
        "the breaching step is counted in the steps returned",
        [
            (
                MARCH,
                BREACH_BRANCH,
                BREACH_BRANCH + "                r.steps = row + 1;\n",
            )
        ],
    ),
    (
        "breach_step_advanced",
        "the breaching step is counted in the run's step count",
        [(MARCH, BREACH_BRANCH, BREACH_BRANCH + "                ++step_;\n")],
    ),
    (
        "refused_step_advanced",
        "the refused step is counted in the run's step count",
        [(MARCH, REFUSED_BRANCH, REFUSED_BRANCH + "\n                ++step_;")],
    ),
    (
        "breach_leftover_is_the_steps",
        "the breach reports the step's leftover, not the accumulated",
        [
            (
                BINDING,
                '        breach["leftover"] = done.record.accumulated;',
                '        breach["leftover"] = done.record.leftover;',
            )
        ],
    ),
    (
        "breach_leftover_supplied_swapped",
        "the breach's leftover and supplied swapped",
        [
            (
                BINDING,
                '        breach["leftover"] = done.record.accumulated;\n'
                '        breach["supplied"] = done.record.supplied;',
                '        breach["leftover"] = done.record.supplied;\n'
                '        breach["supplied"] = done.record.accumulated;',
            )
        ],
    ),
    (
        "breach_supplied_and_limit_swapped",
        "the breach's supplied and limit swapped",
        [
            (
                BINDING,
                '        breach["supplied"] = done.record.supplied;\n        breach["limit"] = m.limit;',
                '        breach["supplied"] = m.limit;\n        breach["limit"] = done.record.supplied;',
            )
        ],
    ),
    (
        "breach_limit_wrong",
        "the breach reports half the run's limit",
        [
            (
                BINDING,
                '        breach["limit"] = m.limit;',
                '        breach["limit"] = 0.5 * m.limit;',
            )
        ],
    ),
    (
        "breach_class_zero",
        "the breach names class 0",
        [
            (
                BINDING,
                '        breach["size_class"] = done.refusal.size_class;',
                '        breach["size_class"] = 0;',
            )
        ],
    ),
    (
        "breach_raised_as_refusal",
        "a breach raised as the binding's own refusal",
        [
            (
                BINDING,
                "    if (done.reason == cfdcore::march::stop::refused) {\n        raise_refusal(done.refusal);",
                "    if (done.reason == cfdcore::march::stop::refused ||\n"
                "        done.reason == cfdcore::march::stop::breach) {\n        raise_refusal(done.refusal);",
            )
        ],
    ),
    (
        "refusal_returned_not_raised",
        "a refused step returned as rows, not raised",
        [
            (
                BINDING,
                "    if (done.reason == cfdcore::march::stop::refused) {\n        raise_refusal(done.refusal);",
                "    if (false && done.reason == cfdcore::march::stop::refused) {\n        raise_refusal(done.refusal);",
            )
        ],
    ),
    (
        "breach_without_record",
        "a breach stop returned with no breach record",
        [
            (
                BINDING,
                '        result["breach"] = breach;',
                '        result["breach"] = nb::none();',
            )
        ],
    ),
    # --- the within-spec rule (B2) ---------------------------------------------
    (
        "within_first_sensor_only",
        "within spec read on the first sensor alone",
        [(RECOVERY, WITHIN_LOOP, WITHIN_LOOP.replace("m < s", "m < 1"))],
    ),
    (
        "within_reads_the_class_not_the_channel",
        "within spec read on each class's own reading, not its channel's",
        [
            (
                RECOVERY,
                "            within = within && channel_readings[k * s + m] < keys_.thresholds[k];",
                "            within = within && readings[k * s + m] < keys_.thresholds[k];",
            )
        ],
    ),
    (
        "within_reads_the_first_class_channel",
        "within spec read on the first channel's reading for every channel",
        [
            (
                RECOVERY,
                "            within = within && channel_readings[k * s + m] < keys_.thresholds[k];",
                "            within = within && channel_readings[m] < keys_.thresholds[k];",
            )
        ],
    ),
    (
        "room_within_when_any_class_is",
        "the room within spec when any channel is, not every",
        [
            (RECOVERY, "    bool room = true;\n", "    bool room = false;\n"),
            (
                RECOVERY,
                "        room = room && within;",
                "        room = room || within;",
            ),
        ],
    ),
    (
        "mean_time_where_undefined",
        "a mean's 100:1 time set where the target is not defined",
        [(RECOVERY, MEAN_TIME, MEAN_TIME.replace(" && defined", ""))],
    ),
    (
        "sensor_time_where_undefined",
        "a sensor's 100:1 time set where the target is not defined",
        [(RECOVERY, SENSOR_TIME, SENSOR_TIME.replace(" && defined", ""))],
    ),
    (
        "mean_time_overwritten",
        "a mean's 100:1 time overwritten by every later update at or below the target",
        [(RECOVERY, MEAN_TIME, MEAN_TIME.replace(" && std::isnan(mean_time_[k])", ""))],
    ),
    (
        "mean_at_target_not_reached",
        "a mean exactly at its target not reached",
        [(RECOVERY, MEAN_TIME, MEAN_TIME.replace("<=", "<"))],
    ),
    (
        "reading_at_target_not_reached",
        "a reading exactly at its target not reached",
        [(RECOVERY, SENSOR_TIME, SENSOR_TIME.replace("<=", "<"))],
    ),
    (
        "target_reads_the_channel_not_the_class",
        "a sensor's 100:1 time read on the channel's reading",
        [
            (
                RECOVERY,
                SENSOR_TIME,
                SENSOR_TIME.replace(
                    "readings[k * s + m]", "channel_readings[k * s + m]"
                ),
            )
        ],
    ),
    (
        "class_since_kept",
        "a class's within-spec start kept after a sensor rises",
        [
            (
                RECOVERY,
                "        class_since_[k] = within ? (std::isnan(since) ? t : since) : nan;",
                "        class_since_[k] = within ? (std::isnan(since) ? t : since) : since;",
            )
        ],
    ),
    (
        "class_since_restarted_every_step",
        "a class's within-spec start restarted at every update",
        [
            (
                RECOVERY,
                "        class_since_[k] = within ? (std::isnan(since) ? t : since) : nan;",
                "        class_since_[k] = within ? t : nan;",
            )
        ],
    ),
    (
        "room_since_restarted_every_step",
        "the room's within-spec start restarted at every update",
        [
            (
                RECOVERY,
                "        if (std::isnan(room_since_)) {\n            room_since_ = t;\n        }",
                "        room_since_ = t;",
            )
        ],
    ),
    (
        "local_field_in_solid",
        "the local recovery field set in SOLID cells",
        [
            (
                RECOVERY,
                "        const lanes::mask above = lanes::mask_of(in.sum[i] >= threshold) & in.live[i];",
                "        const lanes::mask above = lanes::mask_of(in.sum[i] >= threshold);",
            )
        ],
    ),
    # --- the restore of the recovery state (B3) --------------------------------
    (
        "tracker_restore_swaps_mean_and_class",
        "the tracker's restore swaps the mean times and the class starts",
        [
            (
                RECOVERY,
                "    mean_time_.assign(mean_time.begin(), mean_time.end());\n"
                "    class_since_.assign(class_since.begin(), class_since.end());",
                "    mean_time_.assign(class_since.begin(), class_since.end());\n"
                "    class_since_.assign(mean_time.begin(), mean_time.end());",
            )
        ],
    ),
    (
        "march_restore_room_since_zero",
        "the run's restore hands the tracker a zero room start",
        [
            (
                MARCH,
                RESTORE_RECOVERY,
                RESTORE_RECOVERY.replace("state.room_since", "0.0"),
            )
        ],
    ),
    (
        "march_restore_swaps_mean_and_class",
        "the run's restore hands the tracker the mean times as the class starts and the reverse",
        [
            (
                MARCH,
                RESTORE_RECOVERY,
                "        recovery_->restore(state.last_above, state.sensor_time, state.class_since,\n"
                "                           state.mean_time, state.room_since, state.recovery_t);",
            )
        ],
    ),
    (
        "binding_restore_room_since_zero",
        "the binding restores the room's within-spec start as 0.0",
        [
            (
                BINDING,
                "        state.room_since = kept->room_since.data()[0];",
                "        state.room_since = 0.0;",
            )
        ],
    ),
    (
        "binding_restore_room_since_nan",
        "the binding restores the room's within-spec start as NaN",
        [
            (
                BINDING,
                "        state.room_since = kept->room_since.data()[0];",
                "        state.room_since = std::numeric_limits<double>::quiet_NaN();",
            )
        ],
    ),
    (
        "binding_restore_class_since_is_mean_time",
        "the binding restores the classes' starts from the mean times",
        [
            (
                BINDING,
                "        state.class_since = span_of_array(kept->class_since);",
                "        state.class_since = span_of_array(kept->mean_time);",
            )
        ],
    ),
    (
        "binding_restore_mean_time_is_class_since",
        "the binding restores the mean times from the classes' starts",
        [
            (
                BINDING,
                "        state.mean_time = span_of_array(kept->mean_time);",
                "        state.mean_time = span_of_array(kept->class_since);",
            )
        ],
    ),
    (
        "restore_drops_step_and_time",
        "a restore that drops the step count and time",
        [(MARCH, "    step_ = state.step;\n    t_ = state.t;\n", "")],
    ),
    (
        "restore_drops_maps",
        "a restore that drops the arrival maps",
        [(MARCH, "    maps_.restore(state.first_above, state.arrival_t);\n", "")],
    ),
    (
        "restore_drops_budgets",
        "a restore that drops the budgets",
        [(MARCH, "    transport_.set_budgets(state.budgets);\n", "")],
    ),
    (
        "restore_drops_fields",
        "a restore that drops the fields",
        [
            (
                MARCH,
                "    std::copy(state.fields.begin(), state.fields.end(), fields_.begin());\n",
                "",
            )
        ],
    ),
    # --- the march's loop, the driver's reading of it (S1, S3) -----------------
    (
        "hold_not_checked_after_a_step",
        "the hold not checked after a step",
        [
            (
                MARCH,
                "        r.steps = row + 1;\n        if (held()) {\n            r.reason = stop::held;\n            return r;\n        }\n",
                "        r.steps = row + 1;\n",
            )
        ],
    ),
    (
        "interval_end_reported_as_count",
        "the interval's end reported as a count",
        [
            (
                MARCH,
                "    r.reason = i_ == plan_.steps ? stop::interval_end : stop::count;",
                "    r.reason = stop::count;",
            )
        ],
    ),
    (
        "count_reported_as_interval",
        "a count reported as the interval's end",
        [
            (
                MARCH,
                "    r.reason = i_ == plan_.steps ? stop::interval_end : stop::count;",
                "    r.reason = stop::interval_end;",
            )
        ],
    ),
    (
        "channels_only_with_recovery",
        "the channel fields formed only with a recovery",
        [
            (
                MARCH,
                "    if (maps_.started(t_) || recovery_ != nullptr) {",
                "    if (recovery_ != nullptr) {",
            )
        ],
    ),
    (
        "means_from_class_zero",
        "every mean from class 0's content",
        [
            (
                MARCH,
                "            means_[k] = *transport_.budget_of(k).current / free_area_;",
                "            means_[k] = *transport_.budget_of(0).current / free_area_;",
            )
        ],
    ),
    (
        "time_accumulated",
        "a step's time as the last plus dt",
        [
            (
                MARCH,
                "                               : plan_.t_start + static_cast<double>(i_) * plan_.dt;",
                "                               : t_ + plan_.dt;",
            )
        ],
    ),
    (
        "interval_first_ignored",
        "the interval's first step ignored",
        [(MARCH, "    i_ = p.first;", "    i_ = 0;")],
    ),
    (
        "row_inflow_outflow_swapped",
        "a row's inflow and outflow swapped",
        [
            (
                MARCH,
                "        terms[1] = b.inflow;\n        terms[2] = b.outflow;",
                "        terms[1] = b.outflow;\n        terms[2] = b.inflow;",
            )
        ],
    ),
    (
        "row_deposition_slots_reversed",
        "a row's deposition slots reversed",
        [
            (
                MARCH,
                "            terms[4 + slot] = b.deposited[slot];",
                "            terms[4 + slot] = b.deposited[transport::slots - 1 - slot];",
            )
        ],
    ),
    (
        "row_current_is_initial",
        "a row's current from the initial content",
        [
            (
                MARCH,
                "        terms[8] = b.current.value_or(nan);",
                "        terms[8] = b.initial.value_or(nan);",
            )
        ],
    ),
    (
        "recovery_before_the_maps_mean_stale",
        "the maps not updated before the recovery reads the channel fields",
        [
            (
                MARCH,
                "    maps_.update(t_, channel_fields_);\n    if (recovery_ != nullptr) {",
                "    if (recovery_ != nullptr) {",
            )
        ],
    ),
    (
        "onset_strict",
        "an event's onset compared strictly",
        [
            (
                ARRIVAL,
                "            for (std::size_t e = 0; e < onsets_.size() && onsets_[e] <= t; ++e) {",
                "            for (std::size_t e = 0; e < onsets_.size() && onsets_[e] < t; ++e) {",
            )
        ],
    ),
    (
        "arrival_threshold_strict",
        "the arrival threshold compared strictly",
        [
            (
                ARRIVAL,
                "            lanes::mask_of(in.sum[i] >= threshold) & in.live[i] & lanes::mask_of(before != before);",
                "            lanes::mask_of(in.sum[i] > threshold) & in.live[i] & lanes::mask_of(before != before);",
            )
        ],
    ),
    (
        "every_event_writes_the_first_map",
        "every event writes the first event's map",
        [
            (
                ARRIVAL,
                "                double* map = first_.data() + (e * c + k) * n;",
                "                double* map = first_.data() + (0 * c + k) * n;",
            )
        ],
    ),
    (
        "started_strict",
        "the maps' start compared strictly",
        [
            (
                ARRIVAL_H,
                "    bool started(double t) const { return !onsets_.empty() && onsets_.front() <= t; }",
                "    bool started(double t) const { return !onsets_.empty() && onsets_.front() < t; }",
            )
        ],
    ),
    (
        "sensor_order_reversed",
        "the sensors read in reverse order",
        [
            (
                CHANNELS,
                "            out[k * s + m] = classes[k * cells + sensors[m]];",
                "            out[k * s + m] = classes[k * cells + sensors[s - 1 - m]];",
            )
        ],
    ),
    (
        "binding_interval_first_zero",
        "the binding ignores the interval's first step",
        [
            (
                BINDING,
                "    const cfdcore::march::plan p{start, end, read_dt(dt), n_steps, first_step};",
                "    const cfdcore::march::plan p{start, end, read_dt(dt), n_steps, 0};",
            )
        ],
    ),
    (
        "binding_channels_are_readings",
        "the returned channels are the readings",
        [
            (
                BINDING,
                '    result["channels"] = shaped_copy<double>(\n'
                "        std::span<const double>(channels).first(m_rows * c * s), {m_rows, c, s});",
                '    result["channels"] = shaped_copy<double>(\n'
                "        std::span<const double>(readings).first(m_rows * c * s), {m_rows, c, s});",
            )
        ],
    ),
    # --- the binding's refusals, each removed (S4) -----------------------------
    (
        "refusal_spacing_shape",
        "the four spacings' length check removed",
        _off(BINDING, SPACING_LOOP_CHECK, "        if (false && v->shape(0) != n) {"),
    ),
    (
        "refusal_spacing_dx_cell_length",
        "dx_cell's length not checked",
        [
            (
                BINDING,
                "{{&dx_cell, g.cells(0)}, {&dy_cell,",
                "{{&dx_cell, dx_cell.shape(0)}, {&dy_cell,",
            )
        ],
    ),
    (
        "refusal_spacing_dy_cell_length",
        "dy_cell's length not checked",
        [
            (
                BINDING,
                "{&dy_cell, g.cells(1)}, {&dx_face,",
                "{&dy_cell, dy_cell.shape(0)}, {&dx_face,",
            )
        ],
    ),
    (
        "refusal_spacing_dx_face_length",
        "dx_face's length not checked",
        [(BINDING, "{&dx_face, g.cells(0) + 1},\n", "{&dx_face, dx_face.shape(0)},\n")],
    ),
    (
        "refusal_spacing_dy_face_length",
        "dy_face's length not checked",
        [(BINDING, "{&dy_face, g.cells(1) + 1}}};", "{&dy_face, dy_face.shape(0)}}};")],
    ),
    (
        "refusal_spacing_aligned",
        "the spacings' alignment not checked",
        [
            (
                BINDING,
                "        require_aligned(v->data(), alignof(double), who);\n    }\n    if (settling.empty()",
                "    }\n    if (settling.empty()",
            )
        ],
    ),
    (
        "refusal_no_class",
        "a run with no class not refused",
        [
            (
                BINDING,
                "    if (settling.empty() || settling.size() != diffusion.size()) {",
                "    if (settling.size() != diffusion.size()) {",
            )
        ],
    ),
    (
        "refusal_physics_lengths_differ",
        "settling and diffusion lists of different lengths not refused",
        [
            (
                BINDING,
                "    if (settling.empty() || settling.size() != diffusion.size()) {",
                "    if (settling.empty()) {",
            )
        ],
    ),
    (
        "refusal_sensor_row_outside",
        "a sensor row outside the grid not refused",
        [
            (
                BINDING,
                "        if (j >= g.cells(1) || i >= g.cells(0)) {",
                "        if (i >= g.cells(0)) {",
            )
        ],
    ),
    (
        "refusal_sensor_column_outside",
        "a sensor column outside the grid not refused",
        [
            (
                BINDING,
                "        if (j >= g.cells(1) || i >= g.cells(0)) {",
                "        if (j >= g.cells(1)) {",
            )
        ],
    ),
    (
        "refusal_recovery_state_without_recovery",
        "a recovery state given to a run without a recovery not refused",
        [
            (
                BINDING,
                "    } else if (!recovery.is_none()) {",
                "    } else if (false && !recovery.is_none()) {",
            )
        ],
    ),
    (
        "refusal_recovery_state_missing",
        "a missing recovery state for a run with a recovery not refused",
        [
            (
                BINDING,
                '        if (recovery.is_none()) {\n            throw nb::value_error("March2.restore: a run with a recovery needs its state");',
                '        if (false && recovery.is_none()) {\n            throw nb::value_error("March2.restore: a run with a recovery needs its state");',
            )
        ],
    ),
    (
        "refusal_arrival_thresholds",
        "ArrivalMaps2's threshold count not checked against the channels",
        [
            (
                BINDING,
                "        if (maps.channels() != channels.size()) {",
                "        if (false && maps.channels() != channels.size()) {",
            )
        ],
    ),
    (
        "refusal_recovery_thresholds",
        "Recovery2's threshold count not checked against the channels",
        [
            (
                BINDING,
                "        if (tracker.classes() != channels.size()) {",
                "        if (false && tracker.classes() != channels.size()) {",
            )
        ],
    ),
    (
        "refusal_cube_classes",
        "a cube with another class count not refused",
        [
            (
                BINDING,
                "    if (c.shape(0) != classes || c.shape(1)",
                "    if (c.shape(1)",
            )
        ],
    ),
    (
        "refusal_cube_rows",
        "a cube with another row count not refused",
        [
            (
                BINDING,
                "c.shape(1) != g.cells(1) || c.shape(2) != g.cells(0)) {",
                "c.shape(2) != g.cells(0)) {",
            )
        ],
    ),
    (
        "refusal_cube_columns",
        "a cube with another column count not refused",
        [
            (
                BINDING,
                "c.shape(1) != g.cells(1) || c.shape(2) != g.cells(0)) {",
                "c.shape(1) != g.cells(1)) {",
            )
        ],
    ),
    (
        "refusal_cube_aligned",
        "an unaligned cube not refused, by any entry point",
        [
            (
                BINDING,
                "    require_aligned(c.data(), alignof(double), who);\n}\n\nstd::span<const double> span_of3",
                "}\n\nstd::span<const double> span_of3",
            )
        ],
    ),
    (
        "refusal_channel_fields_cube",
        "channel_fields reads its fields without the cube check",
        [
            (
                BINDING,
                '    require_cube(fields, set.size(), g, "channel_fields", "fields");\n',
                "",
            )
        ],
    ),
    (
        "refusal_arrival_update_cube",
        "ArrivalMaps2.update reads its fields without the cube check",
        [
            (
                BINDING,
                '    require_cube(fields, a.channels.size(), *a.grid, "ArrivalMaps2.update", "fields");\n',
                "",
            )
        ],
    ),
    (
        "refusal_recovery_update_cube",
        "Recovery2.update reads its fields without the cube check",
        [
            (
                BINDING,
                '    require_cube(fields, r.channels.size(), *r.grid, "Recovery2.update", "fields");\n',
                "",
            )
        ],
    ),
    (
        "refusal_march_restore_cube",
        "March2.restore reads its fields without the cube check",
        [(BINDING, '    require_cube(fields, r.classes(), g, who, "fields");\n', "")],
    ),
    (
        "refusal_recovery_update_means",
        "Recovery2.update's means length not checked",
        [
            (
                BINDING,
                "    if (means.size() != r.channels.size()) {",
                "    if (false && means.size() != r.channels.size()) {",
            )
        ],
    ),
    (
        "refusal_sensor_channels_shape",
        "sensor_channels' class count not checked",
        [
            (
                BINDING,
                "    if (readings.shape(0) != set.size()) {",
                "    if (false && readings.shape(0) != set.size()) {",
            )
        ],
    ),
    (
        "refusal_sensor_channels_aligned",
        "sensor_channels' alignment not checked",
        [
            (
                BINDING,
                '    require_aligned(readings.data(), alignof(double), "sensor_channels");\n',
                "",
            )
        ],
    ),
    (
        "refusal_maps_shape",
        "a maps array of another shape not refused, by any entry point",
        [
            (
                BINDING,
                "    if (first.shape(0) != m.events() || first.shape(1) != m.channels() ||\n"
                "        first.shape(2) != g.cells(1) || first.shape(3) != g.cells(0)) {",
                "    if (false) {",
            )
        ],
    ),
    (
        "refusal_maps_aligned",
        "an unaligned maps array not refused, by any entry point",
        [
            (
                BINDING,
                "    require_aligned(first.data(), alignof(double), who);\n}",
                "}",
            )
        ],
    ),
    (
        "refusal_arrival_restore_maps",
        "ArrivalMaps2.restore reads its maps without the check",
        [
            (
                BINDING,
                '                require_maps(first, a.maps, *a.grid, "ArrivalMaps2.restore");\n',
                "",
            )
        ],
    ),
    (
        "refusal_march_restore_maps",
        "March2.restore reads its maps without the check",
        [(BINDING, "    require_maps(first_above, r.maps(), g, who);\n", "")],
    ),
    (
        "refusal_recovery_array_length",
        "a recovery state array of another length not refused",
        [
            (
                BINDING,
                "    if (!nb::try_cast<Array>(state[name], a, false) || a.size() != length) {",
                "    if (!nb::try_cast<Array>(state[name], a, false)) {",
            )
        ],
    ),
    (
        "refusal_recovery_array_aligned",
        "an unaligned recovery state array not refused",
        [
            (
                BINDING,
                "    require_aligned(a.data(), alignof(double), who);\n    return a;",
                "    return a;",
            )
        ],
    ),
    (
        "refusal_maps_events",
        "a maps array with another event count not refused",
        [
            (
                BINDING,
                "    if (first.shape(0) != m.events() || first.shape(1) != m.channels() ||\n        first.shape(2) != g.cells(1) || first.shape(3) != g.cells(0)) {",
                "    if (first.shape(1) != m.channels() ||\n        first.shape(2) != g.cells(1) || first.shape(3) != g.cells(0)) {",
            )
        ],
    ),
    (
        "refusal_maps_channels",
        "a maps array with another channel count not refused",
        [
            (
                BINDING,
                "    if (first.shape(0) != m.events() || first.shape(1) != m.channels() ||\n        first.shape(2) != g.cells(1) || first.shape(3) != g.cells(0)) {",
                "    if (first.shape(0) != m.events() ||\n        first.shape(2) != g.cells(1) || first.shape(3) != g.cells(0)) {",
            )
        ],
    ),
    (
        "refusal_maps_rows",
        "a maps array with another row count not refused",
        [
            (
                BINDING,
                "    if (first.shape(0) != m.events() || first.shape(1) != m.channels() ||\n        first.shape(2) != g.cells(1) || first.shape(3) != g.cells(0)) {",
                "    if (first.shape(0) != m.events() || first.shape(1) != m.channels() ||\n        first.shape(3) != g.cells(0)) {",
            )
        ],
    ),
    (
        "refusal_maps_columns",
        "a maps array with another column count not refused",
        [
            (
                BINDING,
                "    if (first.shape(0) != m.events() || first.shape(1) != m.channels() ||\n        first.shape(2) != g.cells(1) || first.shape(3) != g.cells(0)) {",
                "    if (first.shape(0) != m.events() || first.shape(1) != m.channels() ||\n        first.shape(2) != g.cells(1)) {",
            )
        ],
    ),
    # --- the guards of the plan, time and count inputs (S5) --------------------
    (
        "guard_bool_number",
        "a bool read as a number (dt and the times)",
        [
            (
                BINDING,
                '    if (is_bool(value)) {\n        throw nb::type_error((std::string(what) + " must be a number, not a bool").c_str());',
                '    if (false && is_bool(value)) {\n        throw nb::type_error((std::string(what) + " must be a number, not a bool").c_str());',
            )
        ],
    ),
    (
        "guard_bool_count",
        "a bool read as a count",
        [
            (
                BINDING,
                '    if (is_bool(value)) {\n        throw nb::type_error((std::string(what) + " must be an int, not a bool").c_str());',
                '    if (false && is_bool(value)) {\n        throw nb::type_error((std::string(what) + " must be an int, not a bool").c_str());',
            )
        ],
    ),
    (
        "guard_float_count",
        "a float accepted as a count",
        [
            (
                BINDING,
                "nb::steal(PyNumber_Index(value.ptr()))",
                "nb::steal(PyNumber_Long(value.ptr()))",
            )
        ],
    ),
    (
        "guard_count_overflow",
        "a count past 64 bits not refused",
        [
            (
                BINDING,
                '    if (overflow != 0) {\n        throw nb::value_error((std::string(what) + " is out of range").c_str());',
                '    if (false && overflow != 0) {\n        throw nb::value_error((std::string(what) + " is out of range").c_str());',
            )
        ],
    ),
    (
        "guard_time_finite",
        "a time that is not finite accepted",
        [
            (
                BINDING,
                "    if (!std::isfinite(t) && !(nan_ok && std::isnan(t))) {",
                "    if (false) {",
            )
        ],
    ),
    (
        "guard_time_nan_always_ok",
        "a NaN accepted as any time",
        [
            (
                BINDING,
                "    if (!std::isfinite(t) && !(nan_ok && std::isnan(t))) {",
                "    if (!std::isfinite(t) && !std::isnan(t)) {",
            )
        ],
    ),
    (
        "guard_arrival_time_nan_refused",
        "a NaN arrival time (before the first update) refused",
        [
            (
                BINDING,
                'read_time(arrival_t, "arrival_t", true)',
                'read_time(arrival_t, "arrival_t")',
            )
        ],
    ),
    (
        "guard_restore_step_negative",
        "a negative step count accepted by restore",
        [(BINDING, "    if (state.step < 0) {", "    if (false && state.step < 0) {")],
    ),
    (
        "guard_interval_first_range",
        "an interval's first step beyond its steps accepted",
        [
            (
                BINDING,
                "    if (n_steps < 1 || first_step < 0 || first_step >= n_steps) {",
                "    if (n_steps < 1 || first_step < 0) {",
            )
        ],
    ),
    (
        "guard_interval_steps_positive",
        "an interval of no step accepted",
        [
            (
                BINDING,
                "    if (n_steps < 1 || first_step < 0 || first_step >= n_steps) {",
                "    if (first_step < 0 || first_step >= n_steps) {",
            )
        ],
    ),
    (
        "guard_march_n_positive",
        "march(0) accepted",
        [
            (
                BINDING,
                '    if (n < 1) {\n        throw nb::value_error("March2.march: n is at least 1");',
                '    if (false && n < 1) {\n        throw nb::value_error("March2.march: n is at least 1");',
            )
        ],
    ),
    # --- the march's own calls into the recovery (added after the first streams) ---
    (
        "march_recovery_channel_readings_are_readings",
        "the march hands the recovery the classes' readings as the channels' readings",
        [
            (
                MARCH,
                "        recovery_->update(t_, channel_fields_, readings_, channel_readings_, means_);",
                "        recovery_->update(t_, channel_fields_, readings_, readings_, means_);",
            )
        ],
    ),
    (
        "march_recovery_readings_are_channel_readings",
        "the march hands the recovery the channels' readings as the classes' readings",
        [
            (
                MARCH,
                "        recovery_->update(t_, channel_fields_, readings_, channel_readings_, means_);",
                "        recovery_->update(t_, channel_fields_, channel_readings_, channel_readings_, means_);",
            )
        ],
    ),
    (
        "breach_drops_the_rows_before_it",
        "a breach returns no step, the rows before it dropped",
        [(MARCH, BREACH_BRANCH, BREACH_BRANCH + "                r.steps = 0;\n")],
    ),
]


# Plants no test can tell from the unplanted core, each with why: equivalent mutants.
EQUIVALENT = {
    "guard_interval_steps_positive": (
        "the clause `steps < 1` is implied by `0 <= first < steps`, which the same check "
        "tests, so a plan it would refuse is refused by the others"
    ),
}
