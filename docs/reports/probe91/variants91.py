"""Prompt 91's planted defects: each a list of edits to a copy of the checkout's ``core/``.

Read by ``probe91.py plants``. An edit is ``(file, old, new)``: the exact text
``old`` replaced by ``new``, found exactly once, or the plant is refused
(``probe90.plant_sources``). The first nine after the control are the prompt's;
the rest are planted against the other rules the port copies.
"""

CHANNELS = "src/channels.cpp"
ARRIVAL = "src/arrival.cpp"
RECOVERY = "src/recovery.cpp"
MARCH = "src/march.cpp"
BINDING = "python/bindings.cpp"

Edit = tuple[str, str, str]

PLANTS: list[tuple[str, str, list[Edit]]] = [
    ("control", "no edit: every test passes", []),
    (
        "channel_order_reversed",
        "a channel's field summed over its members last to first",
        [
            (
                CHANNELS,
                "    for (const std::size_t j : members) {\n"
                "        const double* field = classes.data() + j * n;",
                "    for (auto it = members.rbegin(); it != members.rend(); ++it) {\n"
                "        const std::size_t j = *it;\n"
                "        const double* field = classes.data() + j * n;",
            )
        ],
    ),
    (
        "threshold_cell_recovered",
        "a cell exactly at its threshold read as recovered: the local field set only above it",
        [
            (
                RECOVERY,
                "lanes::mask_of(in.sum[i] >= threshold) & in.live[i];",
                "lanes::mask_of(in.sum[i] > threshold) & in.live[i];",
            )
        ],
    ),
    (
        "arrival_before_onset",
        "every event's map updated at every step, before its onset too",
        [
            (
                ARRIVAL,
                "for (std::size_t e = 0; e < onsets_.size() && onsets_[e] <= t; ++e) {",
                "for (std::size_t e = 0; e < onsets_.size(); ++e) {",
            )
        ],
    ),
    (
        "arrival_overwritten",
        "a cell's arrival time overwritten by every later step at or above the threshold",
        [
            (
                ARRIVAL,
                "lanes::mask_of(in.sum[i] >= threshold) & in.live[i] & lanes::mask_of(before != before);",
                "lanes::mask_of(in.sum[i] >= threshold) & in.live[i];",
            )
        ],
    ),
    (
        "hold_strict",
        "the hold compared strictly: the stretch must exceed the hold time",
        [
            (
                RECOVERY,
                "return t_ - room_since_ >= *keys_.hold;",
                "return t_ - room_since_ > *keys_.hold;",
            )
        ],
    ),
    (
        "free_area_whole",
        "the free-area mean over the whole domain's area, SOLID cells included",
        [(MARCH, "        if (!grid_->solid(i)) {", "        if (i < n) {")],
    ),
    (
        "restore_drops_recovery",
        "a restore that sets the fields, budgets and maps and drops the recovery fields",
        [
            (
                MARCH,
                "        recovery_->restore(state.last_above, state.sensor_time, state.mean_time,\n"
                "                           state.class_since, state.room_since, state.recovery_t);\n",
                "",
            )
        ],
    ),
    (
        "march_one_too_many",
        "march(n) steps n + 1 steps, its rows sized for them",
        [
            (
                MARCH,
                "    const std::size_t asked = std::min(n, remaining());",
                "    const std::size_t asked = std::min(n + 1, remaining());",
            ),
            (
                BINDING,
                "    const std::size_t rows_n = std::min(static_cast<std::size_t>(n), r.remaining());",
                "    const std::size_t rows_n = std::min(static_cast<std::size_t>(n) + 1, r.remaining());",
            ),
        ],
    ),
    (
        "sensor_channel_order_reversed",
        "a channel's reading at a sensor summed over its members last to first",
        [
            (
                CHANNELS,
                "            for (const std::size_t j : channels.members(k)) {\n"
                "                total = total + readings[j * sensors + m];",
                "            const auto members = channels.members(k);\n"
                "            for (auto it = members.rbegin(); it != members.rend(); ++it) {\n"
                "                total = total + readings[*it * sensors + m];",
            )
        ],
    ),
    (
        "channel_not_from_zero",
        "a channel started from its first member's value, not from zero (-0.0 kept)",
        [
            (
                CHANNELS,
                "    for (double& x : out) {\n        x = 0.0;\n    }\n",
                "    for (double& x : out) {\n        x = -0.0;\n    }\n",
            )
        ],
    ),
    (
        "threshold_sensor_within",
        "a sensor exactly at its threshold read as within spec",
        [
            (
                RECOVERY,
                "within = within && channel_readings[k * s + m] < keys_.thresholds[k];",
                "within = within && channel_readings[k * s + m] <= keys_.thresholds[k];",
            )
        ],
    ),
    (
        "arrival_in_solid",
        "the arrival maps set in SOLID cells too",
        [
            (
                ARRIVAL,
                "lanes::mask_of(in.sum[i] >= threshold) & in.live[i] & lanes::mask_of(before != before);",
                "lanes::mask_of(in.sum[i] >= threshold) & lanes::mask_of(before != before);",
            )
        ],
    ),
    (
        "room_since_kept",
        "the room's within-spec start kept when a sensor rises above its threshold again",
        [(RECOVERY, "    } else {\n        room_since_ = nan;\n    }\n", "    }\n")],
    ),
    (
        "sensor_time_overwritten",
        "a sensor's 100:1 time overwritten by every later reading at or below the target",
        [
            (
                RECOVERY,
                "if (readings[k * s + m] <= keys_.target[k] && defined && std::isnan(when)) {",
                "if (readings[k * s + m] <= keys_.target[k] && defined) {",
            )
        ],
    ),
    (
        "held_not_checked_at_entry",
        "a march that starts on a held run steps on",
        [
            (
                MARCH,
                "    result r;\n    if (held()) {\n        r.reason = stop::held;\n        return r;\n    }\n",
                "    result r;\n",
            )
        ],
    ),
    (
        "last_time_not_t_end",
        "the interval's last step at t_start + steps dt, not at t_end exactly",
        [
            (
                MARCH,
                "        t_ = i_ == plan_.steps ? plan_.t_end\n"
                "                               : plan_.t_start + static_cast<double>(i_) * plan_.dt;",
                "        t_ = plan_.t_start + static_cast<double>(i_) * plan_.dt;",
            )
        ],
    ),
]
