"""Plant each defect prompt 81's tests name, run the tests that should catch it, restore.

    python docs/reports/probe81/plant81.py --root WORKTREE [--set fixes] [--dry-run]

WORKTREE is a git worktree of this branch at the commit under test (the
plants edit its ``src/`` files, never this tree's, so the queue's runs,
which import this tree's ``src/``, are untouched). Each plant is one text
replacement in one file, made at the stated occurrence, and must change the
file; the named tests then run there with the venv's interpreter, one BLAS
thread, and the file is restored byte for byte whatever happens. A plant is
caught when the tests fail. A clean run of every named test first must pass.

The record goes to ``results/builder81/plants.json``: per plant, the file,
the replacement, the tests, pytest's exit code and the last summary line.
``--set fixes`` plants prompt 81b's list instead and records it in
``results/builder81/plants81b.json``. ``--dry-run`` only checks that every
plant of the set applies to WORKTREE's files and leaves them compiling.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

CHANNELS = "tests/test_cumulative_counting.py::TestChannels"
MONITOR = "tests/test_cumulative_counting.py::TestMonitorOnTheSum"
MAPS = "tests/test_cumulative_counting.py::TestMapsAndRecoveryOnTheSum"
FRAMES = "tests/test_cumulative_counting.py::TestFramesOnTheSum"
RECORDS = "tests/test_cumulative_counting.py::TestRecords"
HOLD = (
    "tests/test_cumulative_counting.py::test_the_hold_reads_the_start_on_the_channels"
)
RUN_MAPS = "tests/test_alert_latency.py::TestCumulativeMaps"
RESTART = (
    "tests/test_time_integration.py::TestArrivalMaps::"
    "test_a_per_class_checkpoint_refuses_the_restart"
)

# (name, what the defect is, file, old, new, occurrence (0-based), tests)
PLANTS: list[tuple[str, str, str, str, str, int, list[str]]] = [
    (
        "smaller",
        "each channel sums its class and the smaller classes",
        "src/config.py",
        "if diameters[j] >= diameters[k])",
        "if diameters[j] <= diameters[k])",
        0,
        [CHANNELS, MONITOR, MAPS, FRAMES, HOLD, RUN_MAPS],
    ),
    (
        "class_alone",
        "each channel is its class alone (the per-class comparison)",
        "src/config.py",
        "if diameters[j] >= diameters[k])",
        "if j == k)",
        0,
        [CHANNELS, MONITOR, MAPS, FRAMES, HOLD, RUN_MAPS],
    ),
    (
        "strict",
        "the monitor opens a channel's alert above its limit, not at it",
        "src/monitor.py",
        "above = sums >= self._thresholds[:, None]",
        "above = sums > self._thresholds[:, None]",
        0,
        [MONITOR],
    ),
    (
        "mix_allowed",
        "detection_units stacks per-class maps beside cumulative ones",
        "src/placement.py",
        "if len(set(countings.values())) > 1:",
        "if False:",
        0,
        [RECORDS],
    ),
    (
        "unmarked_cumulative",
        "an /arrival/ without the mark reads as cumulative",
        "src/io_manager.py",
        "counting=str(attrs.get(COUNTING_ATTR, PER_CLASS)),",
        "counting=str(attrs.get(COUNTING_ATTR, CUMULATIVE)),",
        1,
        [RECORDS],
    ),
    (
        "mark_not_written",
        "write_arrival writes no counting",
        "src/io_manager.py",
        "            group.attrs[COUNTING_ATTR] = record.counting\n",
        "",
        1,
        [RECORDS],
    ),
    (
        "flag_dropped",
        "arrival_from_run drops the file's counting",
        "src/placement.py",
        "            counting=record.counting,\n",
        "",
        0,
        [RECORDS],
    ),
    (
        "restart_unchecked",
        "a per-class checkpoint continues on the channels",
        "src/time_integration.py",
        "and COUNTING_STATE not in checkpoint.recovery",
        "and False",
        0,
        [RESTART],
    ),
    (
        "hold_per_class",
        "the hold's start check compares each class alone",
        "src/config.py",
        "for members in self.channel_classes():",
        "for members in ((k,) for k in range(len(thresholds))):",
        0,
        [HOLD],
    ),
]


# Prompt 81b's fix pass (round 1 of review 81 and test 81): the plants of its
# new and strengthened tests, and of the tests prompt 81's own record left
# without one (test 81 S2). Run with ``--set fixes``.
IDENTITY = "channels=tuple((k,) for k in range(len(config.particle_sizes))),"
SMALLER = (
    "channels=tuple(tuple(range(k + 1)) for k in range(len(config.particle_sizes))),"
)
COUNTING_TESTS = "tests/test_cumulative_counting.py"
CLEAR_SUM = (
    "tests/test_placement.py::TestClearUnitsFromRuns::"
    "test_a_channel_clears_with_its_sum_not_with_its_class_alone"
)
RANK_MIX = (
    "tests/test_placement.py::TestRank::"
    "test_layouts_scored_over_two_countings_are_not_ranked"
)
ROLES_MIX = (
    "tests/test_placement.py::TestAddAllClear::"
    "test_detection_and_all_clear_units_of_two_countings_are_refused"
)
MARKS = "tests/test_io_manager.py::TestCountingMarks"
NO_DEFAULT = f"{COUNTING_TESTS}::TestRecords::test_no_record_has_a_default_counting"
TABLE_MARK = (
    f"{COUNTING_TESTS}::TestMonitorOnTheSum::test_the_alert_table_records_its_counting"
)
RECOVERY_MARK = (
    f"{COUNTING_TESTS}::TestMapsAndRecoveryOnTheSum::"
    "test_the_recovery_record_says_cumulative_in_the_file"
)
KEEP = (
    f"{COUNTING_TESTS}::TestRecords::"
    "test_the_units_keep_their_counting_through_every_step"
)
SHAPE = (
    f"{COUNTING_TESTS}::TestRecords::"
    "test_channels_of_another_shape_than_the_readings_are_refused"
)
ORDER = (
    f"{COUNTING_TESTS}::TestChannels::"
    "test_the_channels_follow_the_sizes_not_the_class_order"
)
MALFORMED = f"{COUNTING_TESTS}::TestChannels::test_malformed_channels_are_refused"
RESTORED_MONITOR = (
    f"{COUNTING_TESTS}::TestMonitorOnTheSum::"
    "test_a_restored_monitor_sums_its_channels_again"
)
ONE_OF_TWO = f"{COUNTING_TESTS}::TestRecords::test_the_counting_is_one_of_two"
SERIES_CHANNELS = (
    f"{COUNTING_TESTS}::TestRecords::"
    "test_the_sensor_readings_hold_the_channels_beside_the_classes"
)
TRACKER_REFUSES = (
    f"{COUNTING_TESTS}::TestRecords::test_a_per_class_checkpoint_refuses_the_tracker"
)
ALERT_TABLE = (
    "tests/test_alert_latency.py::TestScenarioRun::"
    "test_the_alert_table_is_the_one_the_sensor_series_implies"
)
RESTART_TABLE = (
    "tests/test_alert_latency.py::TestRestart::"
    "test_a_restart_across_an_open_alert_writes_the_uninterrupted_table"
)
CLEAN_TIMES = (
    "tests/test_time_integration.py::TestCleanOut::"
    "test_the_recovery_times_are_the_run_files_readings"
)
CLEAN_LOCAL = (
    "tests/test_time_integration.py::TestCleanOut::"
    "test_the_local_field_and_the_mean_against_the_last_frame"
)
MAPS_RULE = (
    "tests/test_time_integration.py::TestArrivalMaps::"
    "test_the_maps_are_the_rule_on_every_steps_fields"
)
MAPS_SENSORS = (
    "tests/test_time_integration.py::TestArrivalMaps::"
    "test_the_sensors_latencies_from_the_map_equal_the_monitors"
)
MAPS_FRAMES = (
    "tests/test_time_integration.py::TestArrivalMaps::"
    "test_the_step_map_against_the_frames_map"
)
MAPS_PHASE_4 = (
    "tests/test_time_integration.py::TestArrivalMaps::"
    "test_a_phase_4_file_is_read_from_its_frames_and_flagged"
)
HOLD_START = (
    "tests/test_config.py::TestScenarioHoldTime::"
    "test_the_hold_needs_a_start_above_the_thresholds"
)
WRITE_MARK = "group.attrs[COUNTING_ATTR] = "
# Per record: the file, the field's two lines as written and with a default.
CLASS_FIELDS = {
    "RunArrival": (
        "src/placement.py",
        "    counting: str\n    fallback: str | None = None",
        "    counting: str = CUMULATIVE\n    fallback: str | None = None",
    ),
    "Units": (
        "src/placement.py",
        "    frame_resolved: tuple[bool, ...]\n    counting: str\n",
        "    frame_resolved: tuple[bool, ...]\n    counting: str = CUMULATIVE\n",
    ),
    "ClearUnits": (
        "src/placement.py",
        "    labels: tuple[tuple[str, int], ...]\n    counting: str\n",
        "    labels: tuple[tuple[str, int], ...]\n    counting: str = CUMULATIVE\n",
    ),
    "Layout": (
        "src/placement.py",
        "    counting: str\n    uncovered:",
        "    counting: str = CUMULATIVE\n    uncovered:",
    ),
    "ArrivalRecord": (
        "src/io_manager.py",
        "    t_final: float\n    counting: str\n",
        "    t_final: float\n    counting: str = CUMULATIVE\n",
    ),
    "AlertTable": (
        "src/io_manager.py",
        "    peak_t: np.ndarray\n    counting: str\n",
        "    peak_t: np.ndarray\n    counting: str = CUMULATIVE\n",
    ),
    "RecoveryRecord": (
        "src/io_manager.py",
        "    local_recovery_max: np.ndarray\n    counting: str\n",
        "    local_recovery_max: np.ndarray\n    counting: str = CUMULATIVE\n",
    ),
}
# The three groups that record a counting, with the write line and the read
# line of each and the occurrence of each in src/io_manager.py (file order:
# the recovery's, the alerts', the arrival's).
GROUP_MARKS = {
    "recovery": (
        WRITE_MARK + "record.counting",
        0,
        "counting=str(attrs.get(COUNTING_ATTR, PER_CLASS)),",
        0,
    ),
    "alerts": (
        WRITE_MARK + "table.counting",
        0,
        "counting=str(group.attrs.get(COUNTING_ATTR, PER_CLASS)),",
        0,
    ),
    "arrival": (
        WRITE_MARK + "record.counting",
        1,
        "counting=str(attrs.get(COUNTING_ATTR, PER_CLASS)),",
        1,
    ),
}
MARK_TESTS = {
    "recovery": [MARKS, RECOVERY_MARK],
    "alerts": [MARKS, TABLE_MARK],
    "arrival": [MARKS],
}

FIXES: list[tuple[str, str, str, str, str, int, list[str]]] = [
    # -- review B1: clear_units reads the channels
    (
        "clear_units_identity",
        "clear_units hands clear_times each class as its own channel",
        "src/placement.py",
        "channels=config.channel_classes(),",
        IDENTITY,
        1,
        [CLEAR_SUM],
    ),
    (
        "clear_units_smaller",
        "clear_units sums each class with the smaller classes",
        "src/placement.py",
        "channels=config.channel_classes(),",
        SMALLER,
        1,
        [CLEAR_SUM],
    ),
    (
        "clear_units_marked_per_class",
        "clear_units marks its units per class",
        "src/placement.py",
        "        tuple(labels),\n        CUMULATIVE,\n    )",
        '        tuple(labels),\n        "per_class",\n    )',
        0,
        [CLEAR_SUM],
    ),
    # -- the refusal reaches every scoring path (test S3)
    (
        "search_roles_mix_allowed",
        "search_roles pairs detection and all-clear units of two countings",
        "src/placement.py",
        "if units.counting != clear.counting:",
        "if False:",
        0,
        [ROLES_MIX],
    ),
    (
        "rank_mix_allowed",
        "rank orders layouts scored over two countings",
        "src/placement.py",
        "if layout.counting != out[0].counting:",
        "if False:",
        0,
        [RANK_MIX],
    ),
    (
        "search_layout_marked_cumulative",
        "search marks its layout cumulative whatever the units were",
        "src/placement.py",
        "        counting=units.counting,\n        uncovered=uncovered,",
        "        counting=CUMULATIVE,\n        uncovered=uncovered,",
        0,
        [RANK_MIX, KEEP],
    ),
    (
        "given_layout_marked_cumulative",
        "given_layout marks its layout cumulative whatever the units were",
        "src/placement.py",
        "        units=units.labels,\n        counting=units.counting,\n",
        "        units=units.labels,\n        counting=CUMULATIVE,\n",
        0,
        [KEEP],
    ),
    (
        "exclude_drops_counting",
        "exclude_units returns its units marked cumulative whatever they were",
        "src/placement.py",
        "        units.counting,\n    )",
        "        CUMULATIVE,\n    )",
        0,
        [KEEP],
    ),
    # -- no counting is assumed (review S5): a default restored on each record
    *[
        (
            f"default_{name}",
            f"{name}.counting given a default",
            file,
            old,
            new,
            0,
            [NO_DEFAULT],
        )
        for name, (file, old, new) in CLASS_FIELDS.items()
    ],
    # -- the marks, read back by value (review S1, test S1)
    *[
        plant
        for group, (write, w_at, read, r_at) in GROUP_MARKS.items()
        for plant in (
            (
                f"{group}_mark_not_written",
                f"the {group} writer leaves the counting off",
                "src/io_manager.py",
                write,
                "pass",
                w_at,
                MARK_TESTS[group],
            ),
            (
                f"{group}_mark_constant",
                f"the {group} writer writes cumulative whatever it was given",
                "src/io_manager.py",
                write,
                WRITE_MARK + "CUMULATIVE",
                w_at,
                [MARKS],
            ),
            (
                f"{group}_unmarked_cumulative",
                f"the {group} reader takes a missing mark as cumulative",
                "src/io_manager.py",
                read,
                read.replace("PER_CLASS", "CUMULATIVE"),
                r_at,
                [MARKS],
            ),
            (
                f"{group}_read_constant",
                f"the {group} reader returns cumulative whatever the file says",
                "src/io_manager.py",
                read,
                "counting=CUMULATIVE,",
                r_at,
                [MARKS],
            ),
        )
    ],
    (
        "alert_table_marked_per_class",
        "alert_table marks the table per class",
        "src/time_integration.py",
        "counting=CUMULATIVE,",
        'counting="per_class",',
        0,
        [TABLE_MARK],
    ),
    (
        "alert_table_counting_left_off",
        "alert_table leaves the counting out",
        "src/time_integration.py",
        "counting=CUMULATIVE,\n",
        "",
        0,
        [TABLE_MARK],
    ),
    (
        "recovery_record_marked_per_class",
        "RecoveryTracker.record marks the record per class",
        "src/time_integration.py",
        "counting=CUMULATIVE,",
        'counting="per_class",',
        1,
        [RECOVERY_MARK],
    ),
    (
        "recovery_record_counting_left_off",
        "RecoveryTracker.record leaves the counting out",
        "src/time_integration.py",
        "counting=CUMULATIVE,\n",
        "",
        1,
        [RECOVERY_MARK],
    ),
    (
        "shape_check_removed",
        "write_sensor_readings stores channels of another shape than the readings",
        "src/io_manager.py",
        "if channels.shape != readings.shape:",
        "if False:",
        0,
        [SHAPE],
    ),
    # -- the tests prompt 81's record left under no plant (test 81 S2)
    (
        "index_order",
        "the channels follow the class index, not the sizes",
        "src/config.py",
        "tuple(j for j in order if diameters[j] >= diameters[k])",
        "tuple(j for j in order if j >= k)",
        0,
        [ORDER],
    ),
    (
        "checked_channels_off",
        "checked_channels returns without checking",
        "src/monitor.py",
        "    out = tuple(tuple(members) for members in channels)",
        "    return tuple(tuple(members) for members in channels)",
        0,
        [MALFORMED],
    ),
    (
        "restore_without_sum",
        "AlertMonitor.restore does not sum the channels again",
        "src/monitor.py",
        "self._channel_readings = self._sums(readings)",
        "self._channel_readings = readings",
        0,
        [RESTORED_MONITOR],
    ),
    (
        "counting_unchecked",
        "checked_counting accepts any str",
        "src/io_manager.py",
        "    if counting not in COUNTINGS:",
        "    if False:",
        0,
        [ONE_OF_TWO],
    ),
    (
        "series_write_unchecked",
        "write_sensor_readings accepts channels given to a file without them",
        "src/io_manager.py",
        "if (CHANNELS in group) != (channels is not None):",
        "if False:",
        0,
        [SERIES_CHANNELS],
    ),
    (
        "series_channels_not_read",
        "read_sensor_readings returns no channels",
        "src/io_manager.py",
        "channels=group[CHANNELS][()] if CHANNELS in group else None,",
        "channels=None,",
        0,
        [SERIES_CHANNELS],
    ),
    (
        "tracker_restore_unchecked",
        "ArrivalTracker.restore accepts maps without the channels' mark",
        "src/monitor.py",
        "if COUNTING_STATE not in state:",
        "if False:",
        0,
        [TRACKER_REFUSES],
    ),
    # -- the strengthened run-level tests (test 81 S2)
    (
        "alerts_open_per_class",
        "the monitor opens an alert on the class alone",
        "src/monitor.py",
        "above = sums >= self._thresholds[:, None]",
        "above = readings >= self._thresholds[:, None]",
        0,
        [ALERT_TABLE, RESTART_TABLE],
    ),
    (
        "alerts_read_per_class",
        "the monitor's channels are the per-class readings",
        "src/monitor.py",
        "        sums = self._sums(readings)",
        "        sums = readings.copy()",
        0,
        [ALERT_TABLE, RESTART_TABLE],
    ),
    (
        "restart_open_alert_dropped",
        "the checkpoint carries no open alert",
        "src/monitor.py",
        "            for alert in self.alerts\n        ]",
        "            for alert in self.alerts\n            if alert.step_off is not None\n        ]",
        0,
        [RESTART_TABLE],
    ),
    (
        "restart_table_end_only",
        "the alert table is written only at the end of the march",
        "src/time_integration.py",
        "            self.run.write_alerts(alert_table(self.monitor))",
        "            pass",
        0,
        [RESTART_TABLE],
    ),
    (
        "recovery_within_per_class",
        "RecoveryTracker: within spec read on each class alone",
        "src/time_integration.py",
        "within = (channel_readings < self._thresholds[:, None]).all(axis=1)",
        "within = (readings < self._thresholds[:, None]).all(axis=1)",
        0,
        [CLEAN_TIMES],
    ),
    (
        "recovery_local_per_class",
        "RecoveryTracker: the local field read on each class alone",
        "src/time_integration.py",
        "above = (sums[k] >= threshold) & self._live",
        "above = (fields[k] >= threshold) & self._live",
        0,
        [CLEAN_LOCAL],
    ),
    (
        "march_arrival_identity",
        "the march hands ArrivalTracker each class as its own channel",
        "src/time_integration.py",
        "channels=config.channel_classes(),",
        IDENTITY,
        1,
        [MAPS_RULE, MAPS_SENSORS, MAPS_FRAMES],
    ),
    (
        "frames_run_identity",
        "arrival_from_run's fallback hands each class as its own channel",
        "src/placement.py",
        "channels=config.channel_classes(),",
        IDENTITY,
        0,
        [MAPS_PHASE_4],
    ),
    (
        "hold_at_limit_within",
        "the hold's start check takes a channel at its limit as within spec",
        "src/config.py",
        "if all(c < t for c, t in zip(sums, thresholds, strict=True)):",
        "if all(c <= t for c, t in zip(sums, thresholds, strict=True)):",
        0,
        [HOLD_START],
    ),
    (
        "hold_start_not_compared",
        "the hold's start is not compared with the thresholds",
        "src/config.py",
        "if all(c < t for c, t in zip(sums, thresholds, strict=True)):",
        "if False:",
        0,
        [HOLD_START],
    ),
    (
        "hold_any_channel_below",
        "the hold's start check takes any channel below its threshold for within spec",
        "src/config.py",
        "if all(c < t for c, t in zip(sums, thresholds, strict=True)):",
        "if any(c < t for c, t in zip(sums, thresholds, strict=True)):",
        0,
        [HOLD_START],
    ),
]

# The plant sets: prompt 81's own (T1) and the fix pass's, each with its record.
SETS = {
    "build": (PLANTS, "plants.json"),
    "fixes": (FIXES, "plants81b.json"),
}


def replace_at(text: str, old: str, new: str, occurrence: int) -> str:
    """TEXT with the OCCURRENCE-th OLD replaced by NEW; ValueError if it is not there.

    A worktree checked out on Windows holds CRLF line ends, so a newline in
    OLD or NEW is matched and written as the file's own.
    """
    if "\r\n" in text:
        old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
    start = -1
    for _ in range(occurrence + 1):
        start = text.find(old, start + 1)
        if start < 0:
            raise ValueError(f"occurrence {occurrence} of {old!r} not found")
    return text[:start] + new + text[start + len(old) :]


def pytest(root: Path, tests: list[str]) -> tuple[int, str]:
    """Run TESTS under ROOT; return the exit code and pytest's last line."""
    done = subprocess.run(
        [str(PYTHON), "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests],
        cwd=root,
        env={**os.environ, **ENV},
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [ln for ln in done.stdout.splitlines() if ln.strip()]
    return done.returncode, lines[-1] if lines else ""


def dry_run(root: Path, plants: list) -> int:
    """Check that each plant applies to ROOT's files and leaves them compiling."""
    bad = []
    for name, _, file, old, new, occurrence, _ in plants:
        text = (root / file).read_bytes().decode("utf-8")
        try:
            planted = replace_at(text, old, new, occurrence)
            if planted == text:
                raise ValueError("the plant changes nothing")
            compile(planted, file, "exec")
        except (ValueError, SyntaxError) as error:
            bad.append(f"{name}: {error}")
    print(f"{len(plants) - len(bad)} of {len(plants)} plants apply and compile")
    for line in bad:
        print("  " + line)
    return 1 if bad else 0


def main() -> int:
    """Plant, test and restore each defect; write the record."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--set", choices=sorted(SETS), default="build")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    plants, record_name = SETS[args.set]
    out = ROOT / "results" / "builder81" / record_name
    if args.dry_run:
        return dry_run(root, plants)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    every = sorted({t for plant in plants for t in plant[6]})
    clean_code, clean_line = pytest(root, every)
    record = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "commit": commit,
        "clean": {"tests": every, "exit": clean_code, "summary": clean_line},
        "plants": [],
    }
    print(f"clean: exit {clean_code}, {clean_line}", flush=True)
    if clean_code != 0:
        print("the tests fail without a plant; nothing planted")
        out.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return 1
    for name, what, file, old, new, occurrence, tests in plants:
        path = root / file
        original = path.read_bytes()
        try:
            text = original.decode("utf-8")
            planted = replace_at(text, old, new, occurrence)
            if planted == text:
                raise ValueError("the plant changes nothing")
            path.write_bytes(planted.encode("utf-8"))
            code, line = pytest(root, tests)
        finally:
            path.write_bytes(original)
        caught = code != 0
        record["plants"].append(
            {
                "name": name,
                "defect": what,
                "file": file,
                "old": old,
                "new": new,
                "occurrence": occurrence,
                "tests": tests,
                "exit": code,
                "summary": line,
                "caught": caught,
            }
        )
        print(
            f"{name}: exit {code}, {'caught' if caught else 'SURVIVED'}; {line}",
            flush=True,
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    survived = [p["name"] for p in record["plants"] if not p["caught"]]
    print(
        f"{len(plants) - len(survived)} of {len(plants)} caught; survived: {survived}"
    )
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
