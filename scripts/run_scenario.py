#!/usr/bin/env python
"""Run a scenario file under tier 1, or a transient file under tier 2, and summarise it.

Run from the repository root with the project's interpreter:

    python scripts/run_scenario.py configs/scenario_clean_out.yaml --out run.h5
    python scripts/run_scenario.py SCENARIO --out run.h5 --flows-from kept.h5
    python scripts/run_scenario.py SCENARIO --restart run.h5

A scenario file in; a run file (and its checkpoint file beside it) and a
short text summary out: the intervals, the flow states solved and reused,
the steps, the wall time, each class's budget closure, for a clean-out the
recovery outputs of ADR-014 C, and the alerts the run opened (ADR-015 C). A file with a transient section runs
under tier 2 (ADR-014 G, step 6b) from rest, and its summary is the steps,
the mean and largest outer count per step, the Strouhal number of the last
ten periods with the spectrum's cross-check, or why there is none, and the
drag coefficient. ``--flows-from`` names another run file
whose flow states this run may reuse (``time_integration.run``'s
``flows_from``); ``--restart`` continues the run a run file records, under
the same scenario file but for the keys a continuation may change
(``io_manager.RESTART_EXCLUDED_KEYS``).

The exit status is 0 when the run finished, 1 when a flow state did not
converge (a capped one leaves its steady checkpoint, so a restart under a
larger cap continues it) or a tier-2 step reached its cap, and 2 when the scenario, the run file or the
restart was refused. The loader and ``time_integration.run`` check what they
can before any flow is solved; a ValueError raised later in the run is
reported the same way.
"""

import argparse
import logging
import math
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SimConfig  # noqa: E402 -- follows sys.path.insert
from src.io_manager import RecoveryRecord  # noqa: E402 -- follows sys.path.insert
from src.monitor import Alert  # noqa: E402 -- follows sys.path.insert
from src.solver_staggered import (  # noqa: E402 -- follows sys.path.insert
    TransientStepError,
)
from src.time_integration import (  # noqa: E402 -- follows sys.path.insert
    FlowNotConvergedError,
    Provenance,
    RunRecord,
    TransientRunRecord,
    run,
)

EXIT_DONE = 0
EXIT_NOT_CONVERGED = 1
EXIT_REFUSED = 2


def commit() -> str:
    """The short commit of the checkout this script runs from, or "unknown"."""
    try:
        found = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return found.stdout.strip() or "unknown"


def _time(value: float, missing: str) -> str:
    """Seconds to four decimals, or MISSING where the value is NaN."""
    if math.isnan(value):
        return missing
    if math.isinf(value):
        return "not recovered"
    return f"{value:.4f} s"


def recovery_lines(record: RecoveryRecord, sizes: Sequence[float]) -> list[str]:
    """The clean-out's outputs (ADR-014 C), one class to a block.

    Parameters
    ----------
    record : RecoveryRecord
        The run's recovery outputs.
    sizes : Sequence[float]
        The class diameters, metres, to name the classes.

    Returns
    -------
    list[str]
        The lines, without newlines. A 100:1 time the run ended before is
        given with the ratio's lower bound, ``t_final`` over the well-mixed
        time, since the mean was still above its level then.
    """
    end = record.t_final
    lines = [
        "recovery (ADR-014 C):",
        f"  air change rate N = {record.air_change_rate:.6g} 1/s, well-mixed "
        f"100:1 time ln(100)/N = {record.well_mixed_time:.4f} s",
        f"  stopped by the hold of {record.hold_time} s"
        if record.held
        else f"  ran to t = {end:.4f} s",
        "  room recovery time: "
        + _time(record.room_recovery_time, f"not recovered by {end:.4f} s"),
    ]
    bound = end / record.well_mixed_time
    for k, size in enumerate(sizes):
        lines.append(f"  class {k} ({size * 1e6:g} um):")
        lines.append(
            "    recovery time: "
            + _time(float(record.recovery_time[k]), f"not recovered by {end:.4f} s")
        )
        if not record.defined[k]:
            lines.append("    100:1 times: not defined (initial below 100 thresholds)")
        else:
            mean = float(record.mean_time[k])
            if math.isnan(mean):
                lines.append(
                    f"    free-area mean 100:1: not reached by {end:.4f} s "
                    f"(ratio to well-mixed above {bound:.4f})"
                )
            else:
                lines.append(
                    f"    free-area mean 100:1: {mean:.4f} s, ratio to well-mixed "
                    f"{float(record.ratio[k]):.4f}"
                )
            sensors = ", ".join(
                f"{name} {_time(float(t), 'not reached')}"
                for name, t in zip(
                    record.sensor_names, record.sensor_time[k], strict=True
                )
            )
            lines.append(f"    sensor 100:1: {sensors}")
        largest = float(record.local_recovery_max[k])
        lines.append(
            "    local recovery field maximum: "
            + (
                "not recovered (a cell still above)"
                if math.isinf(largest)
                else f"{largest:.4f} s"
            )
        )
    return lines


def alert_lines(
    alerts: Sequence[Alert], sizes: Sequence[float], monitored: bool
) -> list[str]:
    """The alerts the run opened (ADR-015 C), one to a line.

    Parameters
    ----------
    alerts : Sequence[Alert]
        ``RunRecord.alerts``, closed ones first.
    sizes : Sequence[float]
        The class diameters, metres, to name the classes.
    monitored : bool
        Whether the run had sensors; without them there is no monitor.

    Returns
    -------
    list[str]
        The lines, without newlines: the count and how many are still open,
        then each alert's sensor, class, opening step and reading, closing
        step or "still open", and peak.
    """
    if not monitored:
        return ["alerts (ADR-015 C): no sensors"]
    if not alerts:
        return ["alerts (ADR-015 C): none (no sensor at or above its threshold)"]
    still = sum(alert.is_open for alert in alerts)
    lines = [f"alerts (ADR-015 C): {len(alerts)} opened, {still} still open at the end"]
    for alert in alerts:
        if alert.is_open:
            end = "still open"
        else:
            end = f"below at t = {alert.t_off:.4f} s (step {alert.step_off})"
        size = sizes[alert.size_class] * 1e6
        lines.append(
            f"  {alert.sensor}, class {alert.size_class} ({size:g} um): at or above "
            f"from t = {alert.t_on:.4f} s (step {alert.step_on}), reading "
            f"{alert.reading_on:.4g}; {end}; peak {alert.peak:.4g} at "
            f"t = {alert.peak_t:.4f} s"
        )
    return lines


def summary(record: RunRecord, config: SimConfig, scenario: Path) -> list[str]:
    """The run's summary lines.

    Parameters
    ----------
    record : RunRecord
        What ``time_integration.run`` returned.
    config : SimConfig
        The scenario's configuration.
    scenario : Path
        The scenario file.

    Returns
    -------
    list[str]
        The lines, without newlines.
    """
    lines = [f"scenario: {scenario}", f"run file: {record.path}"]
    if record.restarted_from is not None:
        lines.append(f"restarted from step {record.restarted_from}")
    lines.append(f"intervals: {len(record.intervals)}")
    for plan in record.intervals:
        events = ", ".join(plan.events) if plan.events else "none"
        lines.append(
            f"  [{plan.t_start:g}, {plan.t_end:g}) s: state {plan.fingerprint[:12]}, "
            f"{plan.steps} steps of {plan.dt:.6g} s, events {events}"
        )
    if record.resumed_plan is not None:
        plan = record.resumed_plan
        lines.append(
            f"  divided again from {plan.t_start:.6g} s: {plan.steps} steps of "
            f"{plan.dt:.6g} s"
        )
    solved = [f for f in record.flows if f.solved]
    read = [f for f in record.flows if not f.solved]
    repeated = len(record.intervals) - len({p.fingerprint for p in record.intervals})
    lines.append(
        f"flow states: {len(record.flows)} distinct, {len(solved)} solved, "
        f"{len(read)} read from a run file; intervals on a state already held: "
        f"{repeated}"
    )
    for flow in record.flows:
        warm = f" from {flow.warm_from[:12]}" if flow.warm_from else ""
        lines.append(
            f"  {flow.fingerprint[:12]}: {flow.origin}{warm}, {flow.outer} outer, "
            f"{flow.stop_reason}, {flow.wall_seconds:.1f} s"
        )
    stop = "the hold" if record.held else "t_end"
    lines.append(f"steps: {record.steps} to t = {record.t_final:.6g} s ({stop})")
    lines.append(f"wall time: {record.wall_seconds:.1f} s")
    lines.append("budget closure (residual, relative to the particles supplied):")
    for k, budget in enumerate(record.budgets):
        size = config.particle_sizes[k]
        if budget.current is None:
            lines.append(f"  class {k} ({size * 1e6:g} um): no step taken")
            continue
        try:
            relative = f"relative {budget.relative():.3e}"
        except ValueError:
            # A class no event and no inlet supplied has nothing to close
            # against (a single-size release leaves the others at zero).
            relative = "relative not defined (nothing supplied)"
        lines.append(
            f"  class {k} ({size * 1e6:g} um): residual {budget.residual():.6g}, "
            f"{relative}"
        )
    if record.recovery is not None:
        lines.extend(recovery_lines(record.recovery, config.particle_sizes))
    lines.extend(
        alert_lines(record.alerts, config.particle_sizes, bool(config.sensors))
    )
    return lines


def transient_summary(record: TransientRunRecord, scenario: Path) -> list[str]:
    """A tier-2 run's summary lines (ADR-014 G, step 6b).

    Parameters
    ----------
    record : TransientRunRecord
        What ``time_integration.run`` returned for a transient file.
    scenario : Path
        The file.

    Returns
    -------
    list[str]
        The lines, without newlines: the steps, the outer counts, St with
        its cross-check or the reason there is none, and C_D.
    """
    lines = [f"transient: {scenario}", f"run file: {record.path}"]
    if record.restarted_from is not None:
        lines.append(f"restarted from step {record.restarted_from}")
    lines.append(f"steps: {record.steps} to t = {record.t_final:.6g} s")
    lines.append(
        f"outer per step: mean {record.outer_mean:.2f}, largest {record.outer_max}"
    )
    reading = record.strouhal
    if reading is None:
        lines.append(f"St: not defined ({record.strouhal_refusal})")
    else:
        verdict = "agrees" if reading.agrees else "DISAGREES"
        lines.append(
            f"St: {reading.strouhal:.5f} over the last ten periods (period "
            f"{reading.period:.5f} s); spectrum {reading.spectral_strouhal:.5f} "
            f"+/- {reading.resolution:.5f}, {verdict}"
        )
    if record.drag_coefficient is not None:
        window = "over the St window" if reading is not None else "at the last step"
        lines.append(f"C_D: {record.drag_coefficient:.5f} ({window})")
    lines.append(f"wall time: {record.wall_seconds:.1f} s")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Run the scenario, print the summary, return the exit status.

    Parameters
    ----------
    argv : Sequence[str], optional
        The arguments; ``sys.argv[1:]`` when None.

    Returns
    -------
    int
        EXIT_DONE, EXIT_NOT_CONVERGED or EXIT_REFUSED.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenario", type=Path, help="the scenario file")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--out", type=Path, help="a new run's run file")
    target.add_argument("--restart", type=Path, help="the run file to continue")
    parser.add_argument(
        "--flows-from", type=Path, help="a run file whose flow states to reuse"
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=0,
        help="processes running beside this one, for the run file's record",
    )
    parser.add_argument("--quiet", action="store_true", help="no progress logging")
    args = parser.parse_args(argv)
    if not args.quiet:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        config = SimConfig(args.scenario)
        record = run(
            config,
            args.out,
            restart=args.restart,
            flows_from=args.flows_from,
            provenance=Provenance.here(commit(), args.concurrent),
        )
    except (FlowNotConvergedError, TransientStepError) as err:
        print(f"error: {err}", file=sys.stderr)
        return EXIT_NOT_CONVERGED
    except (ValueError, KeyError, FileNotFoundError, TypeError) as err:
        print(f"error: {err}", file=sys.stderr)
        return EXIT_REFUSED
    if isinstance(record, TransientRunRecord):
        print("\n".join(transient_summary(record, args.scenario)))
    else:
        print("\n".join(summary(record, config, args.scenario)))
    return EXIT_DONE


if __name__ == "__main__":
    sys.exit(main())
