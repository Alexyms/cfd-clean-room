"""Prompt 96's planted defects, and the runner that shows the tests catch each.

    python docs/reports/probe96/variants96.py check
    python docs/reports/probe96/variants96.py run --platform windows
    python docs/reports/probe96/variants96.py run --platform linux

A plant is one exact-text edit of one file (found exactly once, or the plant is
refused), made in a copy of the commit's tree (``git archive``), with the tests
that should catch it run there. Caught: pytest exits other than 0. The first
ten are the prompt's list; the next ten the builder's, one per rule the step
added that the prompt's list does not reach; the rest are round 1's (prompt 96b):
test 96's own plants of the comparator, the backend's routes and the harness,
rewritten against the code as it stands, and one for each defect a finding names.
``windows`` plants run on Windows' environment from ``results/builder96/plants/tree``;
``linux`` plants need the core and run under WSL2 from ``~/scratch96/tree`` with the
build tree on ``PYTHONPATH``. The comparator's plants read the run files of
``tests/test_compare_runs.py``'s fixtures, built once by ``build_runs`` from the
unplanted tree (``CFD_COMPARE_RUNS``): no plant changes how a run is made.
``check`` applies every plant to the sources in memory and reports one whose text
is not found once. The records go to ``results/builder96/plants_<platform>.json``.
"""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


@dataclasses.dataclass(frozen=True)
class Plant:
    """One planted defect: the file, the text replaced, the tests that catch it."""

    name: str
    what: str
    path: str
    old: str
    new: str
    platform: str
    tests: tuple[str, ...]


WINDOWS_BACKEND_RUN = ("tests/test_time_integration.py", "-k", "BackendKeyInTheRun")
CORE_BACKEND = ("tests/test_core_backend.py",)
CONFIG_BACKEND = ("tests/test_config.py", "-k", "BackendKey")
MONITOR = ("tests/test_monitor.py", "-k", "FromReadings")
COMPARATOR = ("tests/test_compare_runs.py",)
HARNESS = ("tests/test_benchmark.py", "-k", "backend or factory")
CMP = "scripts/compare_runs.py"


def c(name: str, what: str, old: str, new: str) -> Plant:
    """A plant in the comparator, caught by ``tests/test_compare_runs.py``."""
    return Plant(name, what, CMP, old, new, "windows", COMPARATOR)


# The class fields' four items, each reading the flag that explains a flow field.
EXPLAINED = '            False,\n            "",\n            explainable=False,\n'
EXPLAINED_TRUE = '            True,\n            "",\n            explainable=True,\n'

PLANTS = (
    # ---- the prompt's nine (prompt 96) --------------------------------------
    Plant(
        "restart_hash_without_key",
        "the key left out of the restart hash",
        "src/io_manager.py",
        '    ("solver", "checkpoint_interval"),\n)',
        '    ("solver", "checkpoint_interval"),\n    ("solver", "backend"),\n)',
        "windows",
        WINDOWS_BACKEND_RUN,
    ),
    Plant(
        "restart_hash_without_key_core",
        "the key left out of the restart hash (the core's whole runs)",
        "src/io_manager.py",
        '    ("solver", "checkpoint_interval"),\n)',
        '    ("solver", "checkpoint_interval"),\n    ("solver", "backend"),\n)',
        "linux",
        CORE_BACKEND,
    ),
    Plant(
        "default_core",
        "the default core",
        "src/config.py",
        "DEFAULT_BACKEND = REFERENCE_BACKEND",
        "DEFAULT_BACKEND = CORE_BACKEND",
        "windows",
        CONFIG_BACKEND,
    ),
    Plant(
        "readings_one_step_late",
        "the monitor reading the readings one step late",
        "src/core_backend.py",
        'step, t, result["readings"][r], result["channels"][r]',
        'step, t, result["readings"][max(r - 1, 0)], result["channels"][max(r - 1, 0)]',
        "linux",
        CORE_BACKEND,
    ),
    Plant(
        "channel_wrong_class",
        "a channel read from the wrong class",
        "src/monitor.py",
        'return self._advance(step, t, arrays["readings"], arrays["channels"])',
        'return self._advance(\n            step, t, arrays["readings"], np.roll(arrays["channels"], 1, axis=0)\n        )',
        "windows",
        MONITOR,
    ),
    Plant(
        "channel_wrong_class_core",
        "a channel read from the wrong class (the core's whole runs)",
        "src/monitor.py",
        'return self._advance(step, t, arrays["readings"], arrays["channels"])',
        'return self._advance(\n            step, t, arrays["readings"], np.roll(arrays["channels"], 1, axis=0)\n        )',
        "linux",
        CORE_BACKEND,
    ),
    c(
        "sign_reversed",
        "the comparator's sign reversed",
        "diff = float(np.abs(b[finite] - a[finite]).max())",
        "diff = float((b[finite] - a[finite]).max())",
    ),
    c(
        "tier_bounds_swapped",
        "its tier bounds swapped",
        "    spread, limit = BAND[name]\n    if value <= spread:",
        "    limit, spread = BAND[name]\n    if value <= spread:",
    ),
    c(
        "straddle_as_no_move",
        "a straddling move read as no move",
        "    out = np.full(ref.shape, math.inf)\n    both =",
        "    out = np.full(ref.shape, 0.0)\n    both =",
    ),
    c(
        "decision_wrong_step",
        "a decision compared on the reference's steps for both runs",
        "    return moved(grid_index(grid_ref, ref), grid_index(other_grid, other), 1.0)",
        "    return moved(grid_index(grid_ref, ref), grid_index(grid_ref, other), 1.0)",
    ),
    Plant(
        "factory_reference_under_core",
        "the factory building the reference under core",
        "src/core_backend.py",
        "        return CoreSolver(mesh, config, boundary)\n    return reference(",
        "        return reference(mesh, config, boundary)\n    return reference(",
        "linux",
        CORE_BACKEND,
    ),
    # ---- the builder's, round 0 ---------------------------------------------
    Plant(
        "core_with_transient",
        "the core accepted with a transient section",
        "src/config.py",
        "        if self.backend == CORE_BACKEND and self.transient is not None:\n",
        "        if False:\n",
        "windows",
        CONFIG_BACKEND,
    ),
    Plant(
        "core_without_its_module",
        "the core accepted where _cfdcore does not import",
        "src/config.py",
        "        importlib.import_module(CORE_MODULE)\n",
        "        pass\n",
        "windows",
        CONFIG_BACKEND,
    ),
    Plant(
        "run_state_reference_under_core",
        "time_integration building the reference's run under core",
        "src/time_integration.py",
        "    if config.backend == CORE_BACKEND:\n        return CoreRun(config, run, on_step)",
        "    if False:\n        return CoreRun(config, run, on_step)",
        "linux",
        CORE_BACKEND,
    ),
    Plant(
        "update_resums_channels",
        "the update from readings summing the channels itself",
        "src/monitor.py",
        'return self._advance(step, t, arrays["readings"], arrays["channels"])',
        'return self._advance(\n            step, t, arrays["readings"], self._sums(arrays["readings"])\n        )',
        "windows",
        MONITOR,
    ),
    c(
        "missing_alert_as_none",
        "an alert in one run only read as no move",
        "        moves.extend([math.inf] * (2 * abs(len(xs) - len(ys))))",
        "        moves.extend([])",
    ),
    c(
        "closure_unchecked",
        "the closure not checked",
        "            closure <= CLOSURE,",
        "            True,",
    ),
    c(
        "explained_by_equal_stops",
        "a middle-tier field explained by equal stops",
        "    return bool(aparts) and max(aparts) == 1",
        "    return bool(aparts) and max(aparts) <= 1",
    ),
    c(
        "kind_change_as_equal",
        "a value gone NaN compared on the finite entries only",
        "    if not np.array_equal(finite, np.isfinite(b)):\n        return math.inf",
        "    finite = finite & np.isfinite(b)",
    ),
    c(
        "kind_change_unchecked",
        "a value gone NaN compared on the finite entries only, both checks removed",
        "    if not np.array_equal(finite, np.isfinite(b)):\n        return math.inf\n"
        "    odd = ~finite\n"
        "    if not np.array_equal(a[odd], b[odd], equal_nan=True):\n"
        "        return math.inf\n",
        "    finite = finite & np.isfinite(b)\n",
    ),
    Plant(
        "harness_builds_staggered",
        "the harness building StaggeredSolver itself",
        "scripts/benchmark.py",
        "    solver = steady_solver(\n        mesh, config, StaggeredBoundary(mesh, config), reference=StaggeredSolver\n    )",
        "    solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))",
        "windows",
        HARNESS,
    ),
    Plant(
        "old_rows_as_core",
        "an old harness row read as the core's",
        "scripts/benchmark.py",
        'return str(row.get("params", {}).get("backend", DEFAULT_BACKEND))',
        'return str(row.get("params", {}).get("backend", "core"))',
        "windows",
        HARNESS,
    ),
    # ---- round 1: test 96's controls, on rules a test already named ----------
    c(
        "arrival_unread",
        "the arrival maps compared with themselves",
        "Moves.of(index_moves(ar.maps, ao.maps, times, times_other)),",
        "Moves.of(index_moves(ar.maps, ar.maps, times, times_other)),",
    ),
    c(
        "concentration_unread",
        "the class fields compared with themselves",
        "        conc.append(relative(a, b))",
        "        conc.append(relative(a, a))",
    ),
    c(
        "stop_reason_ignored",
        "a stop for another reason passes",
        "                apart <= 1\n                and sa.stop_reason == sb.stop_reason\n"
        "                and capped_a == capped_b,",
        "                apart <= 1 and capped_a == capped_b,",
    ),
    c(
        "stop_within_two",
        "a stop two apart passes",
        "                apart <= 1\n                and sa.stop_reason",
        "                apart <= 2\n                and sa.stop_reason",
    ),
    c(
        "closure_ref_only",
        "the closure read on the reference only",
        "    closure = max(_closure(ref), _closure(other))",
        "    closure = _closure(ref)",
    ),
    c(
        "closure_other_only",
        "the closure read on the other run only",
        "    closure = max(_closure(ref), _closure(other))",
        "    closure = _closure(other)",
    ),
    c(
        "budget_verdict_unread",
        "the REQ-T05 verdicts not compared",
        "            verdict_r == verdict_o,",
        "            True,",
    ),
    c(
        "explained_by_any_one_apart",
        "a middle-tier field explained though another stop is two apart",
        "    return bool(aparts) and max(aparts) == 1",
        "    return 1 in aparts",
    ),
    c(
        "req_t05_at_a_hundredth",
        "REQ-T05's limit a hundred times larger",
        "REQ_T05 = 1.0e-4",
        "REQ_T05 = 1.0e-2",
    ),
    # ---- round 1: the items no test named (test 96 B2) -----------------------
    c(
        "capped_ignored",
        "a state capped on one side only passes",
        "                apart <= 1\n                and sa.stop_reason == sb.stop_reason\n"
        "                and capped_a == capped_b,",
        "                apart <= 1 and sa.stop_reason == sb.stop_reason,",
    ),
    c(
        "missing_flow_state_passes",
        "a flow state in one run only passes",
        '                    False,\n                    {"kind": STOP, "ref": a is not None',
        '                    True,\n                    {"kind": STOP, "ref": a is not None',
    ),
    c(
        "pointwise_unread",
        "the pointwise measure compares the reference with itself",
        "        point.append(pointwise(a, b, POINTWISE_FLOOR * thresholds[k]))",
        "        point.append(pointwise(a, a, POINTWISE_FLOOR * thresholds[k]))",
    ),
    c(
        "pointwise_floor_at_threshold",
        "the pointwise floor the threshold, not a hundredth of it",
        "POINTWISE_FLOOR = 0.01",
        "POINTWISE_FLOOR = 1.0",
    ),
    c(
        "pointwise_floor_at_nothing",
        "the pointwise floor zero, every cell read",
        "POINTWISE_FLOOR = 0.01",
        "POINTWISE_FLOOR = 0.0",
    ),
    c(
        "sensor_channels_unread",
        "the sensor channels compared with themselves",
        "            relative(sr.channels, so.channels),",
        "            relative(sr.channels, sr.channels),",
    ),
    c(
        "leftover_change_unread",
        "the leftover's change not read",
        "    change = _leftover_change(end_r, end_o)",
        "    change = _leftover_change(end_r, end_r)",
    ),
    c(
        "flow_fields_u_only",
        "only u compared of the six flow fields",
        "        for name in FLOW_FIELDS:\n            x, y = getattr(a, name), getattr(b, name)",
        "        for name in FLOW_FIELDS[:1]:\n            x, y = getattr(a, name), getattr(b, name)",
    ),
    c(
        "clear_times_unread",
        "the cells' clear times compared with themselves",
        "                Moves.of(_frame_moves(cr, co, cr.clear, co.clear)),",
        "                Moves.of(_frame_moves(cr, co, cr.clear, cr.clear)),",
    ),
    c(
        "room_clear_unread",
        "the room's clear times compared with themselves",
        "                Moves.of(_frame_moves(cr, co, room_r, room_o)),",
        "                Moves.of(_frame_moves(cr, co, room_r, room_r)),",
    ),
    c(
        "latencies_unread",
        "the file's sensors' latencies compared with themselves",
        "Moves.of(index_moves(lr + onsets, lo + onsets, times, times_other)),",
        "Moves.of(index_moves(lr + onsets, lr + onsets, times, times_other)),",
    ),
    c(
        "step_times_unread",
        "the step times item always passes",
        '            "the step times",\n            flags["step times"],',
        '            "the step times",\n            True,',
    ),
    c(
        "last_step_unread",
        "the run's last step compared with itself",
        'items.append(_same("the run\'s last step", int(fr.steps[-1]), int(fo.steps[-1])))',
        'items.append(_same("the run\'s last step", int(fr.steps[-1]), int(fr.steps[-1])))',
    ),
    c(
        "last_time_unread",
        "the run's last time compared with itself",
        'items.append(_same("the run\'s last time", float(fr.times[-1]), float(fo.times[-1])))',
        'items.append(_same("the run\'s last time", float(fr.times[-1]), float(fr.times[-1])))',
    ),
    c(
        "hold_unread",
        "the hold compared with itself",
        "                None if ro is None else bool(ro.held),",
        "                None if rr is None else bool(rr.held),",
    ),
    c(
        "alert_closing_unread",
        "an alert's closing step not compared",
        "            out.setdefault(key, []).append((float(table.step_on[i]), closing))",
        "            out.setdefault(key, []).append((float(table.step_on[i]), 0.0))",
    ),
    c(
        "alert_opening_unread",
        "an alert's opening step not compared",
        "            out.setdefault(key, []).append((float(table.step_on[i]), closing))",
        "            out.setdefault(key, []).append((0.0, closing))",
    ),
    c(
        "more_than_one_ignored",
        "a decision moved by more than one step reads as equal",
        "        return self.one == 0 and self.more == 0 and self.only == 0",
        "        return self.one == 0 and self.only == 0",
    ),
    c(
        "recovery_unread",
        "the clean-out's recovery times compared with themselves",
        "                Moves.of(index_moves(recovery_r, recovery_o, times, times_other)),",
        "                Moves.of(index_moves(recovery_r, recovery_r, times, times_other)),",
    ),
    c(
        "closure_last_frame_only",
        "the closure read on the last frame only",
        "    for budgets in run.frames.budgets:\n        for b in budgets:\n            try:",
        "    for budgets in run.frames.budgets[-1:]:\n        for b in budgets:\n            try:",
    ),
    c(
        "first_tier_admits_between",
        "first_tier true with a field between the spread and the limit",
        '            item.detail["tier"] == INSIDE\n            for item in self.items',
        '            item.detail["tier"] != BEYOND\n            for item in self.items',
    ),
    c(
        "main_exits_zero",
        "the command line exits 0 on a FAIL",
        '    return 0 if comparison.verdict == "PASS" else 1',
        "    return 0",
    ),
    c(
        "req_t05_residual_only",
        "a class's verdict reads the residual alone",
        "    return abs(residual) < REQ_T05 and abs(leftover) < REQ_T05",
        "    return abs(residual) < REQ_T05",
    ),
    c(
        "req_t05_leftover_only",
        "a class's verdict reads the leftover alone",
        "    return abs(residual) < REQ_T05 and abs(leftover) < REQ_T05",
        "    return abs(leftover) < REQ_T05",
    ),
    c(
        "reached_ref_twice",
        "the arrival times' reach counted on the reference for both runs",
        "            _reached(ar.maps, ao.maps),",
        "            _reached(ar.maps, ar.maps),",
    ),
    # ---- round 1: the middle tier of a class field (review 96 B1) -----------
    c(
        "class_concentration_explained",
        "the concentration's middle tier explained by the outer count",
        "            CONCENTRATION,\n" + EXPLAINED,
        "            CONCENTRATION,\n" + EXPLAINED_TRUE,
    ),
    c(
        "class_pointwise_explained",
        "the pointwise measure's middle tier explained by the outer count",
        "            POINTWISE,\n" + EXPLAINED,
        "            POINTWISE,\n" + EXPLAINED_TRUE,
    ),
    c(
        "class_sensors_explained",
        "the sensor channels' middle tier explained by the outer count",
        "            SENSORS,\n" + EXPLAINED,
        "            SENSORS,\n" + EXPLAINED_TRUE,
    ),
    c(
        "class_leftover_explained",
        "the leftover change's middle tier explained by the outer count",
        "            LEFTOVER,\n" + EXPLAINED,
        "            LEFTOVER,\n" + EXPLAINED_TRUE,
    ),
    # ---- round 1: a decision falls at the same step or frame (test 96 B1) ---
    c(
        "clear_move_over_the_gap",
        "a clear time's move counted over the largest gap between frames, rounded",
        "            index_moves(\n                np.asarray(clear_ref[u]),\n                np.asarray(clear_other[u]),\n                ref.times[u],\n                other.times[u],\n            ).ravel()",
        "            np.rint(\n                moved(\n                    np.asarray(clear_ref[u]),\n                    np.asarray(clear_other[u]),\n                    ref.spacing[u],\n                )\n            ).ravel()",
    ),
    c(
        "moves_rounded_to_the_nearest",
        "a move of a fraction of a step rounded to none",
        "            one=int(((finite > 0.0) & (finite <= 1.0)).sum()),\n            more=int((finite > 1.0).sum()),",
        "            one=int((np.rint(finite) == 1).sum()),\n            more=int((np.rint(finite) > 1).sum()),",
    ),
    c(
        "time_on_the_next_step",
        "a time placed on the first step at or after it, not the nearest",
        "    out[finite] = np.where(nearer_below, below, above)",
        "    out[finite] = above",
    ),
    # ---- round 1: the placement (review 96 B2) --------------------------------
    c(
        "placement_one_side",
        "the placement reads the reference's runs on both sides",
        "    for runs in sides:\n        units = detection_units(",
        "    for runs in (sides[0], sides[0]):\n        units = detection_units(",
    ),
    c(
        "placement_cells_unread",
        "the layouts' cells compared with themselves",
        '                    r["cells"][name],\n                    o["cells"][name],\n',
        '                    r["cells"][name],\n                    r["cells"][name],\n',
    ),
    c(
        "placement_order_unread",
        "the layouts' order compared with itself",
        '        items.append(_same(f"{rule}: the layouts\' order", r["order"], o["order"]))',
        '        items.append(_same(f"{rule}: the layouts\' order", r["order"], r["order"]))',
    ),
    c(
        "placement_latencies_unread",
        "the layouts' latencies compared with themselves",
        "                    for side, per_rule_side in ((0, r), (1, o)):",
        "                    for side, per_rule_side in ((0, r), (1, r)):",
    ),
    c(
        "placement_first_onset",
        "every unit's latency added to the first event's onset",
        "                        e = list(run.arrival.event_names).index(event)",
        "                        e = 0",
    ),
    c(
        "placement_reference_steps",
        "the other run's latencies placed on the reference's steps",
        "                            sides[1][scenario].step_times,\n",
        "                            sides[0][scenario].step_times,\n",
    ),
    # ---- round 1: the signs (review 96 B3) and the closure's NaN (S1) --------
    c(
        "leftover_change_abs_removed",
        "the leftover's change read signed",
        "        largest = max(largest, abs(b - a))",
        "        largest = max(largest, b - a)",
    ),
    c(
        "leftover_change_negated",
        "the leftover's change read with the opposite sign",
        "        largest = max(largest, abs(b - a))",
        "        largest = max(largest, a - b)",
    ),
    c(
        "closure_abs_removed",
        "the closure's residual read signed",
        "                residual = abs(float(b.relative()))",
        "                residual = float(b.relative())",
    ),
    c(
        "closure_negated",
        "the closure's residual read with the opposite sign",
        "                residual = abs(float(b.relative()))",
        "                residual = -float(b.relative())",
    ),
    c(
        "verdict_abs_removed",
        "the REQ-T05 verdict reading both ratios signed",
        "    return abs(residual) < REQ_T05 and abs(leftover) < REQ_T05",
        "    return residual < REQ_T05 and leftover < REQ_T05",
    ),
    c(
        "verdict_leftover_signed",
        "the REQ-T05 verdict reading the leftover signed",
        "    return abs(residual) < REQ_T05 and abs(leftover) < REQ_T05",
        "    return abs(residual) < REQ_T05 and leftover < REQ_T05",
    ),
    c(
        "closure_nan_skipped",
        "a NaN residual skipped by the closure",
        "            largest = math.inf if math.isnan(residual) else max(largest, residual)",
        "            largest = max(largest, residual)",
    ),
    # ---- round 1: the backend's routes (test 96 S3, S4; review 96 S5) --------
    Plant(
        "with_backend_attribute_only",
        "with_backend setting the attribute and not the copy's mapping",
        "src/config.py",
        '        mapping = _thaw(self._mapping)\n        mapping.setdefault("solver", {})["backend"] = backend\n        out._mapping = _freeze(mapping)\n',
        "",
        "windows",
        CONFIG_BACKEND,
    ),
    Plant(
        "with_backend_attribute_only_core",
        "with_backend setting the attribute and not the copy's mapping (the core's run)",
        "src/config.py",
        '        mapping = _thaw(self._mapping)\n        mapping.setdefault("solver", {})["backend"] = backend\n        out._mapping = _freeze(mapping)\n',
        "",
        "linux",
        CORE_BACKEND,
    ),
    Plant(
        "with_backend_base_unnamed",
        "with_backend leaving a scenario's merged base without the backend",
        "src/config.py",
        "            out.scenario = replace(self.scenario, base=_freeze(base))\n",
        "            pass\n",
        "windows",
        CONFIG_BACKEND,
    ),
    Plant(
        "with_backend_keeps_the_files",
        "with_backend leaving the file's backend on the copy",
        "src/config.py",
        "        out.backend = backend\n        mapping = _thaw(self._mapping)",
        "        out.backend = self.backend\n        mapping = _thaw(self._mapping)",
        "windows",
        CONFIG_BACKEND,
    ),
    Plant(
        "with_backend_unchecked",
        "validation.cases.with_backend not looking for the module",
        "validation/cases.py",
        "    if backend == CORE_BACKEND:\n        check_core_module()\n",
        "",
        "windows",
        CONFIG_BACKEND,
    ),
    Plant(
        "module_looked_for_at_load",
        "the module looked for when a configuration loads",
        "src/config.py",
        "        if self.backend == CORE_BACKEND and self.transient is not None:\n",
        "        if self.backend == CORE_BACKEND:\n            check_core_module()\n"
        "        if self.backend == CORE_BACKEND and self.transient is not None:\n",
        "windows",
        (*CONFIG_BACKEND[:1], "-k", "BackendKey or core_run_file"),
    ),
    Plant(
        "module_looked_for_at_load_run_file",
        "the module looked for when a run file's configuration loads",
        "src/config.py",
        "        if self.backend == CORE_BACKEND and self.transient is not None:\n",
        "        if self.backend == CORE_BACKEND:\n            check_core_module()\n"
        "        if self.backend == CORE_BACKEND and self.transient is not None:\n",
        "windows",
        COMPARATOR,
    ),
    Plant(
        "core_solver_built_without_the_module",
        "a core solver built without looking for the module",
        "src/core_backend.py",
        "        require_core()\n        super().__init__(mesh, config, boundary)",
        "        super().__init__(mesh, config, boundary)",
        "windows",
        WINDOWS_BACKEND_RUN,
    ),
    Plant(
        "core_march_built_without_the_module",
        "a core march built without looking for the module",
        "src/core_backend.py",
        "        require_core()\n        super().__init__(config, run, on_step)",
        "        super().__init__(config, run, on_step)",
        "windows",
        WINDOWS_BACKEND_RUN,
    ),
    Plant(
        "flow_free_backend",
        "flows_from copies a flow state across backends",
        "src/time_integration.py",
        '    "checkpoint_interval",\n    "max_simple_iter",\n)',
        '    "checkpoint_interval",\n    "max_simple_iter",\n    "backend",\n)',
        "windows",
        WINDOWS_BACKEND_RUN,
    ),
    # ---- round 1: the harness (test 96 S1, review 96 S6) ----------------------
    Plant(
        "run_case_ignores_backend",
        "run_case builds and records the default backend whatever it is given",
        "scripts/benchmark.py",
        "        backend,\n    )\n    solver_block = solver_parameters(config)",
        "        DEFAULT_BACKEND,\n    )\n    solver_block = solver_parameters(config)",
        "windows",
        HARNESS,
    ),
    Plant(
        "run_case_ignores_backend_core",
        "run_case builds and records the default backend whatever it is given (a row)",
        "scripts/benchmark.py",
        "        backend,\n    )\n    solver_block = solver_parameters(config)",
        "        DEFAULT_BACKEND,\n    )\n    solver_block = solver_parameters(config)",
        "linux",
        CORE_BACKEND,
    ),
    Plant(
        "main_ignores_backend",
        "the harness's --backend not handed to run_case",
        "scripts/benchmark.py",
        "                case_id, args.method, args.sample_every, args.concurrent, args.backend\n",
        "                case_id, args.method, args.sample_every, args.concurrent\n",
        "windows",
        HARNESS,
    ),
    Plant(
        "summary_pools_backends",
        "the summary pooling a case's rows of both backends",
        "scripts/benchmark.py",
        "        rows.setdefault((*key, rule, version, backend, metric, reference), []).append(",
        '        rows.setdefault((*key, rule, version, "reference", metric, reference), []).append(',
        "windows",
        HARNESS,
    ),
    # ---- round 1: the monitor's two updates (test 96 S2) ------------------------
    Plant(
        "readings_not_copied",
        "update_from_readings keeps the caller's readings array",
        "src/monitor.py",
        '            "readings": np.array(readings, dtype=np.float64),',
        '            "readings": np.asarray(readings, dtype=np.float64),',
        "windows",
        MONITOR,
    ),
    Plant(
        "channels_not_copied",
        "update_from_readings keeps the caller's channels array",
        "src/monitor.py",
        '            "channels": np.array(channels, dtype=np.float64),',
        '            "channels": np.asarray(channels, dtype=np.float64),',
        "windows",
        MONITOR,
    ),
    Plant(
        "readings_time_unchecked",
        "update_from_readings does not check its step and time",
        "src/monitor.py",
        "        step, t = self._checked_time(step, t)\n        expected = self._readings.shape",
        "        expected = self._readings.shape",
        "windows",
        MONITOR,
    ),
    Plant(
        "fields_mode_reads_readings",
        "monitor_reads_fields set, the monitor still on the readings",
        "src/core_backend.py",
        "                    monitor_fields = fields_now if self.monitor_reads_fields else None",
        "                    monitor_fields = None",
        "linux",
        CORE_BACKEND,
    ),
    Plant(
        "readings_and_channels_swapped",
        "CoreMarch handing the monitor the channels as readings and the readings as channels",
        "src/core_backend.py",
        'step, t, result["readings"][r], result["channels"][r]',
        'step, t, result["channels"][r], result["readings"][r]',
        "linux",
        CORE_BACKEND,
    ),
    Plant(
        "batch_last_row",
        "every row of a batch reading the batch's last readings",
        "src/core_backend.py",
        'step, t, result["readings"][r], result["channels"][r]',
        'step, t, result["readings"][-1], result["channels"][-1]',
        "linux",
        CORE_BACKEND,
    ),
    Plant(
        "transient_not_refused",
        "CoreSolver.solve_transient running the reference's loop",
        "src/core_backend.py",
        "        raise NotImplementedError(\n"
        '            "the core carries the steady solve; tier 2\'s true-time step "',
        "        return super().solve_transient(*args, **kwargs)\n"
        "        raise NotImplementedError(\n"
        '            "the core carries the steady solve; tier 2\'s true-time step "',
        "linux",
        CORE_BACKEND,
    ),
    Plant(
        "march_one_step_a_call",
        "the march going to the core one step at a time whatever reads it",
        "src/core_backend.py",
        "                if every_step:\n                    n = 1",
        "                if True:\n                    n = 1",
        "linux",
        (*CORE_BACKEND, "tests/test_core_march.py"),
    ),
)


# Plants that change no result, with the reason. ``relative`` refuses a change of
# kind twice: the finite masks differ, or the non-finite entries do; a value
# finite in one run and not in the other is in the second set too, so removing
# the first check alone leaves the second to refuse it. ``kind_change_unchecked``
# removes both, and is caught. ``capped_ignored``: a state whose run was capped
# is read from the steady checkpoint, whose stop reason is the cap's or None;
# a finished run's flow state holds one of the rule's own reasons. The two
# reasons are never equal, so two states with equal reasons are both from a
# checkpoint or both from a finished run, and the capped flag never differs
# where the reasons agree.
EQUIVALENT = {
    "kind_change_as_equal": (
        "the second check (the non-finite entries compared) refuses every change of "
        "kind the first does"
    ),
    "capped_ignored": (
        "a capped run's state (from the steady checkpoint) has the stop reason of the "
        "cap or None, a finished run's one of the rule's: the reasons differ whenever "
        "the capped flags do, and the reason is compared"
    ),
}


def commit() -> str:
    """HEAD's hash read from the checkout's files (git runs on Windows only)."""
    head = (ROOT / ".git" / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    ref = ROOT / ".git" / head[5:]
    if ref.exists():
        return ref.read_text().strip()
    for line in (ROOT / ".git" / "packed-refs").read_text().splitlines():
        if line.endswith(head[5:]):
            return line.split()[0]
    return "unknown"


def _read(path: Path) -> str:
    """A file's text with LF line ends."""
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def check(tree: Path = ROOT) -> list[str]:
    """Every plant whose text is not found exactly once in TREE's file."""
    bad = []
    for plant in PLANTS:
        count = _read(tree / plant.path).count(plant.old)
        if count != 1:
            bad.append(f"{plant.name}: found {count} times")
    return bad


def apply(tree: Path, plant: Plant) -> str:
    """Write the plant into TREE; return the file's text before it."""
    path = tree / plant.path
    text = _read(path)
    if text.count(plant.old) != 1:
        raise SystemExit(f"{plant.name}: its text is not found once")
    path.write_bytes(text.replace(plant.old, plant.new).encode("utf-8"))
    return text


def prebuilt_runs(tree: Path, python: str, folder: Path) -> dict[str, str]:
    """The comparator tests' run files, built once from the unplanted TREE.

    Returns the environment entry that hands them to the tests.
    """
    if not (folder / "clean.h5").exists():
        code = (
            "import sys; from pathlib import Path; "
            "from tests import test_compare_runs; "
            "test_compare_runs.build_runs(Path(sys.argv[1]))"
        )
        subprocess.run([python, "-c", code, str(folder)], cwd=tree, check=True)
    return {"CFD_COMPARE_RUNS": str(folder)}


def run(
    platform: str,
    tree: Path,
    python: str,
    out: Path,
    only: list[str] | None = None,
    runs: Path | None = None,
) -> int:
    """Each plant of PLATFORM in TREE (those named in ONLY), its tests run there.

    The records go to OUT; with ONLY they are merged into OUT's earlier ones.
    """
    records = []
    chosen = [p for p in PLANTS if p.platform == platform]
    if only:
        chosen = [p for p in chosen if p.name in only]
    environment = os.environ.copy()
    if runs is not None and any(COMPARATOR[0] in p.tests for p in chosen):
        environment.update(prebuilt_runs(tree, python, runs))
    for plant in chosen:
        before = apply(tree, plant)
        started = time.perf_counter()
        try:
            done = subprocess.run(
                [
                    python,
                    "-m",
                    "pytest",
                    *plant.tests,
                    "-q",
                    "--tb=no",
                    "-x",
                    "-p",
                    "no:cacheprovider",
                ],
                cwd=tree,
                capture_output=True,
                text=True,
                env=environment,
            )
        finally:
            (tree / plant.path).write_bytes(before.encode("utf-8"))
        tail = [line for line in done.stdout.splitlines() if line.strip()][-1:]
        record = {
            "name": plant.name,
            "what": plant.what,
            "file": plant.path,
            "tests": " ".join(plant.tests),
            "returncode": done.returncode,
            "caught": done.returncode != 0,
            "equivalent": EQUIVALENT.get(plant.name),
            "summary": tail[0] if tail else done.stderr.strip().splitlines()[-1:],
            "seconds": round(time.perf_counter() - started, 1),
        }
        records.append(record)
        print(
            plant.name,
            "caught" if record["caught"] else "SURVIVES",
            record["summary"],
            flush=True,
        )
    # The tree unchanged: the same tests pass there.
    tests = sorted(
        {
            t
            for p in PLANTS
            if p.platform == platform
            for t in p.tests
            if t.startswith("tests/")
        }
    )
    clean = subprocess.run(
        [python, "-m", "pytest", *tests, "-q", "--tb=no", "-p", "no:cacheprovider"],
        cwd=tree,
        capture_output=True,
        text=True,
        env=environment,
    )
    lines = [line for line in clean.stdout.splitlines() if line.strip()]
    if only and out.exists():
        earlier = json.loads(out.read_text(encoding="utf-8"))["plants"]
        names = {r["name"] for r in records}
        records = [r for r in earlier if r["name"] not in names] + records
        order = [p.name for p in PLANTS]
        records.sort(key=lambda r: order.index(r["name"]))
        for record in records:
            record["equivalent"] = EQUIVALENT.get(record["name"])
    result = {
        "platform": platform,
        "commit": commit(),
        "plants": records,
        "unchanged_tree": {
            "returncode": clean.returncode,
            "summary": lines[-1] if lines else "",
        },
    }
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print("unchanged tree:", result["unchanged_tree"])
    return 0


def main(argv: list[str] | None = None) -> int:
    """The command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("check", "run"))
    parser.add_argument("--platform", choices=("windows", "linux"), default="windows")
    parser.add_argument("--tree", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--only", nargs="*")
    parser.add_argument(
        "--runs", type=Path, help="a folder for the comparator tests' built run files"
    )
    args = parser.parse_args(argv)
    if args.command == "check":
        bad = check(args.tree or ROOT)
        print("\n".join(bad) if bad else f"all {len(PLANTS)} plants found once")
        return 1 if bad else 0
    out = args.out or ROOT / "results" / "builder96" / f"plants_{args.platform}.json"
    return run(args.platform, args.tree, args.python, out, args.only, args.runs)


if __name__ == "__main__":
    sys.exit(main())
