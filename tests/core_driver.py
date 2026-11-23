"""A scenario run whose march steps through the compiled core (prompt 91, ADR-017 B).

``CoreRun`` is ``time_integration._Run`` with its ``march`` replaced: the plans,
the events, the flow states, the alert monitor, the frames, the checkpoints, the
restart and the end records stay the reference's own code, and each interval is
handed to a ``_cfdcore.March2``, which steps it with the fields resident and
returns per step what ADR-017 B's exchange table names. Python calls the core
between output steps: to the next frame, checkpoint or interval end, or one step
at a time when something reads the fields every step (``on_step``, or the
monitor while ``monitor_reads_fields`` is set, as it reads them today).

``run`` calls ``time_integration.run`` with ``CoreRun`` in place of ``_Run`` for
the call, so a scenario runs as the reference's ``run`` runs it and only the
march differs. Step 6 moves this into ``src/time_integration.py`` behind the
``solver.backend`` key; until then the parity tests and prompt 91's probe share
it here, and nothing under ``src/`` changes.
"""

import os
from collections.abc import Callable, Mapping, Sequence
from unittest import mock

import numpy as np

import src.time_integration as time_integration
from src.config import SimConfig
from src.io_manager import RunFile
from src.monitor import Alert
from src.solver_transport import MassBudget
from src.time_integration import (
    IntervalPlan,
    MarchStep,
    RunRecord,
    TransientRunRecord,
    _copies,
    _Run,
)
from tests import core_mapping


class CoreRun(_Run):
    """``_Run`` with the march stepped through the core.

    Parameters
    ----------
    config : SimConfig
        The run's configuration (a scenario section and a transport section).
    run : RunFile
        The run file.
    on_step : Callable[[MarchStep], None] or None
        Called after every step's writes, as the reference calls it.

    Notes
    -----
    ``monitor_reads_fields`` (class attribute, True) keeps the monitor on the
    fields: they are read from the core after every step and handed to
    ``AlertMonitor.update``, as the reference hands them, and the core's
    readings and channels are checked against the monitor's bit for bit. False
    hands the monitor the core's readings instead, on fields that hold them at
    the sensors' cells and zero elsewhere (the monitor reads nothing else), so
    the march goes to the core in batches. ``core_calls`` counts the core's
    ``march`` calls.
    """

    monitor_reads_fields: bool = True

    def __init__(
        self,
        config: SimConfig,
        run: RunFile,
        on_step: Callable[[MarchStep], None] | None,
    ) -> None:
        super().__init__(config, run, on_step)
        recovery = None
        if self.tracker is not None:
            spec = config.scenario
            assert spec is not None
            initial = self.manager.initial_fields()
            recovery = core_mapping.recovery_keys(
                config.class_thresholds("the clean-out's recovery"),
                [float(initial[k][self.live][0]) for k in range(self.n_classes)],
                spec.hold_time,
            )
        self.core = core_mapping.core_march(
            config, self.mesh, self.cells, [onset for _, onset in self.events], recovery
        )
        self.core_calls = 0
        self._stand_in: dict[int, np.ndarray] | None = None

    def _check_interval(self, plan: IntervalPlan) -> None:
        """Refuse an interval whose transport inputs the core was not built with."""
        theirs = self.intervals[plan.index].config
        if (
            theirs.transport != self.config.transport
            or theirs.particle_sizes != self.config.particle_sizes
            or theirs.t_end != self.config.t_end
        ):
            raise ValueError(
                f"interval {plan.index}'s transport keys, particles or t_end are not "
                "the run's, which the core was built with"
            )

    def _readings_fields(self, readings: np.ndarray) -> dict[int, np.ndarray]:
        """Fields holding the core's readings at the sensors' cells and zero elsewhere."""
        if self._stand_in is None:
            shape = self.mesh.cell_type.shape
            self._stand_in = {k: np.zeros(shape) for k in range(self.n_classes)}
        rows = [j for j, _ in self.cells]
        cols = [i for _, i in self.cells]
        for k, field in self._stand_in.items():
            field[rows, cols] = readings[k]
        return self._stand_in

    def _observe_row(
        self,
        step: int,
        t: float,
        result: Mapping[str, np.ndarray],
        r: int,
        fields: Mapping[int, np.ndarray] | None,
        pending: list[tuple[int, float, np.ndarray, np.ndarray]],
    ) -> tuple[Alert, ...]:
        """The monitor's update after row R of a march, on the fields or the readings."""
        if self.monitor is None:
            return ()
        if fields is None:
            return self.observe(
                step, t, self._readings_fields(result["readings"][r]), pending
            )
        alerts = self.observe(step, t, fields, pending)
        # The core's readings are the monitor's, read from the same fields.
        if (
            self.monitor.readings.tobytes() != result["readings"][r].tobytes()
            or self.monitor.channel_readings.tobytes()
            != result["channels"][r].tobytes()
        ):
            raise AssertionError(
                f"the core's readings at step {step} are not the monitor's"
            )
        return alerts

    def march(
        self,
        plans: list[IntervalPlan],
        start: tuple[int, int],
        fields: dict[int, np.ndarray],
        budgets: Sequence[MassBudget],
        step: int,
        t: float,
    ) -> tuple[dict[int, np.ndarray], tuple[MassBudget, ...], int, float, bool]:
        """Step every class from (plan, step within it) to the end or the hold, in the core.

        Returns the fields, budgets, step count and time at the end, and
        whether the hold stopped it, as ``_Run.march`` does.
        """
        core = self.core
        config = self.config
        output, cadence = config.output_interval, config.checkpoint_interval
        core_mapping.restore_march(
            core, fields, budgets, step, t, self.arrival, self.tracker
        )
        pending: list[tuple[int, float, np.ndarray, np.ndarray]] = []
        held = core.held()
        current = tuple(budgets)
        every_step = self.on_step is not None or (
            self.monitor is not None and self.monitor_reads_fields
        )
        first_plan, first_i = start
        for p in range(first_plan, len(plans)):
            if held:
                break
            plan = plans[p]
            self._check_interval(plan)
            interval = self.intervals[plan.index]
            rates = self.manager.sources_at(interval.t_start)
            core_mapping.set_interval(
                core,
                self.manager.conditions_for(interval),
                self.faces(plan.fingerprint),
                self.states[plan.fingerprint].nu_t,
                [rates[k] if rates[k].any() else None for k in range(self.n_classes)],
                plan,
                first_i if p == first_plan else 0,
            )
            while core.remaining > 0 and not held:
                n = core.remaining
                if every_step:
                    n = 1
                else:
                    n = min(n, output - step % output)
                    if cadence is not None:
                        n = min(n, cadence - step % cadence)
                result = core.march(n)
                self.core_calls += 1
                rows = len(result["step"])
                for r in range(rows):
                    step, t = int(result["step"][r]), float(result["t"][r])
                    current = tuple(core_mapping.budgets_of_row(result["budgets"][r]))
                    held = result["stopped"] == "held" and r == rows - 1
                    last = held or (
                        p == len(plans) - 1 and core.remaining == 0 and r == rows - 1
                    )
                    frame = step % output == 0 or last
                    checkpoint = cadence is not None and (step % cadence == 0 or last)
                    end_of_call = r == rows - 1
                    if (frame or checkpoint or every_step) and not end_of_call:
                        raise AssertionError(
                            f"step {step} needs the fields but is not a march's last"
                        )
                    fields_now = None
                    if end_of_call and (frame or checkpoint or every_step):
                        fields_now = core_mapping.unstacked(core.class_fields())
                        fields = fields_now
                    monitor_fields = fields_now if self.monitor_reads_fields else None
                    alerts = self._observe_row(
                        step, t, result, r, monitor_fields, pending
                    )
                    if frame or checkpoint:
                        core_mapping.sync_trackers(core, self.arrival, self.tracker)
                    if frame:
                        self.run.write_frame(step, t, fields, current)
                        self._flush(pending)
                    if checkpoint:
                        self._flush(pending)
                        self.run.write_arrival(self.arrival_record(t))
                        self.run.checkpoint(
                            step,
                            t,
                            fields,
                            current,
                            plan.fingerprint,
                            recovery=self.carried(),
                            plan_start=plan.t_start,
                        )
                    if self.on_step is not None:
                        self.on_step(
                            MarchStep(
                                step,
                                t,
                                plan.dt,
                                plan.index,
                                fields,
                                _copies(current),
                                alerts,
                            )
                        )
                error = core_mapping.breach_error(result)
                if error is not None:
                    raise error
        fields = core_mapping.unstacked(core.class_fields())
        current = tuple(core_mapping.budgets_of(core))
        core_mapping.sync_trackers(core, self.arrival, self.tracker)
        self._flush(pending)
        return fields, _copies(current), int(core.step), float(core.t), held


def run(
    config: SimConfig,
    run_file: str | os.PathLike[str] | None = None,
    *,
    restart: str | os.PathLike[str] | None = None,
    flows_from: str | os.PathLike[str] | None = None,
    on_step: Callable[[MarchStep], None] | None = None,
    provenance: time_integration.Provenance | None = None,
    monitor_reads_fields: bool = True,
) -> RunRecord | TransientRunRecord:
    """``time_integration.run`` with the march stepped through the core.

    Parameters
    ----------
    config, run_file, restart, flows_from, on_step, provenance
        As ``time_integration.run`` takes them (tier 1).
    monitor_reads_fields : bool
        ``CoreRun.monitor_reads_fields`` for the call.

    Returns
    -------
    RunRecord or TransientRunRecord
        ``time_integration.run``'s record.
    """
    driver = type(
        "CoreRunForCall", (CoreRun,), {"monitor_reads_fields": monitor_reads_fields}
    )
    with mock.patch.object(time_integration, "_Run", driver):
        return time_integration.run(
            config,
            run_file,
            restart=restart,
            flows_from=flows_from,
            on_step=on_step,
            provenance=provenance,
        )
