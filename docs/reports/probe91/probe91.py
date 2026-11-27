"""Prompt 91's probe: whole scenario runs through the core's march against the reference.

    python docs/reports/probe91/probe91.py configs
    python docs/reports/probe91/probe91.py run <scenario> <side>
    python docs/reports/probe91/probe91.py restart
    python docs/reports/probe91/probe91.py restart-clean
    python docs/reports/probe91/probe91.py compare
    python docs/reports/probe91/probe91.py timing --run N
    python docs/reports/probe91/probe91.py plants
    python docs/reports/probe91/probe91.py plants-round1 [--tree NAME] [--only NAME...]
    python docs/reports/probe91/probe91.py vecinfo

Run under WSL2 with the Linux environment (``~/.venvs/cfd_clean_room``) from the
checkout's root, one BLAS thread, through ``results/builder91/py.sh``. Nothing
in ``src/`` changes.

- ``configs``: each scenario's file under ``results/builder91/configs/``
  (SCENARIOS below), its base file named by its path here.
- ``run``: one scenario from its stored flow states (``flows_from``), on the
  reference's ``time_integration.run`` (side ``reference``) or on the same call
  with the march stepped through the core (side ``core``,
  ``tests.core_driver.run``, since prompt 96 ``_core_run`` here). The run file is written on the Linux file system
  and copied into ``results/builder91/runs/<side>/`` with a summary in prompt
  85's form. A hook on ``RunFile.write_frame`` records, at every frame, a digest
  of the arrival maps and of each recovery array as the run's trackers hold them
  when the frame is written (in the core-driven run, set from the core's state
  just before), so the maps and recovery fields are compared at every output
  step and not only at the end.
- ``restart``: the 80x30 dust release with a checkpoint every 1,000 steps,
  through the core, stopped by an exception from ``on_step`` after step 1,500
  and restarted through the core from its checkpoint.
- ``restart-clean``: the 80x30 held clean-out with a checkpoint every
  CLEAN_CADENCE steps, run through the core to its hold (``hold80r``), and twice
  more, each stopped after a step and restarted through the core: stopped after
  step 1,650 (checkpoint at 1,600, before the room's within-spec stretch, which
  begins at step 1,660) and after step 1,850 (checkpoint at 1,800, inside it). The
  state each restart carries is recorded with it (``room_since`` and the classes'
  starts).
- ``compare``: every pair of runs, bit for bit: the frames, the sensor table,
  the alert table, the arrival maps and the recovery record at the end, the
  digests at every frame, and the decisions; ``results/builder91/compare.json``.
- ``timing``: the 320x120 dust release's march over a window across the release,
  the reference's ``_Run.march`` against the core-driven march with and without
  the monitor's per-step field reads, in one process, with the machine's state.
- ``plants``: each planted defect of ``variants91.py`` built in turn, with CTest,
  the march's parity module and two whole 80x30 runs against the reference's.
- ``plants-round1``: round 1's planted defects (``variants91b.py``), each built
  in turn in the tree ``--tree`` names, with CTest and the march's parity module
  (no whole runs: a whole run of the dust release and the held clean-out cannot
  see a breach, a refusal or a restore). ``--check`` applies every plant's edits
  to the sources and reports one whose text is not found exactly once; the
  records go to ``results/builder91/plants_round1_<tree>.json``.
- ``vecinfo``: GCC's vectorisation report on the new modules.
"""

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np  # noqa: E402 -- the thread variables are set first
import yaml  # noqa: E402 -- the thread variables are set first

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "probe85"))
sys.path.insert(0, str(HERE.parent / "probe90"))

import probe85  # noqa: E402 -- follows sys.path.insert
import probe90  # noqa: E402 -- follows sys.path.insert; reads Windows base paths here

import src.time_integration as time_integration  # noqa: E402 -- follows sys.path.insert
from src.config import SimConfig  # noqa: E402 -- follows sys.path.insert
from src.io_manager import RunFile  # noqa: E402 -- follows sys.path.insert
from src.monitor import ArrivalTracker  # noqa: E402 -- follows sys.path.insert
from src.solver_transport import MassBudget  # noqa: E402 -- follows sys.path.insert
from src.time_integration import (  # noqa: E402 -- follows sys.path.insert
    Provenance,
    RecoveryTracker,
)

RESULTS = ROOT / "results" / "builder91"
CONFIGS = RESULTS / "configs"
RUNS = RESULTS / "runs"
LINUX_RUNS = Path.home() / "scratch91" / "runs"
B86 = ROOT / "results" / "builder86"
B85 = ROOT / "results" / "builder85"
B81 = ROOT / "results" / "builder81"
SIDES = ("reference", "core")
# One air change of the 80x30 room (configs/scenario_clean_out.yaml: 1/N is
# 5.562 s): the hold the scenario file says stopped its 80x30 run at 31 s.
HOLD = 5.562
# The restart: a checkpoint every RESTART_CADENCE steps, stopped after STOP_AT.
RESTART_CADENCE = 1000
STOP_AT = 1500
# The core-driven runs round 0 made, which round 1 makes again and compares with them.
ROUND0_RUNS = ("dust80", "door80", "breach80", "clean80", "hold80", "dust320")
# The clean-out's restarts: a checkpoint every CLEAN_CADENCE steps, stopped after
# each of CLEAN_STOPS (the within-spec stretch of the held run begins at step 1,660).
CLEAN_CADENCE = 100
CLEAN_STOPS = {"before_restart": 1650, "inside_restart": 1850}
# The timing's window: 200 steps from the frame at step 2,700 of the 320x120
# dust release (about 9.8 s), across the release at 10 s.
WINDOW_FROM, WINDOW_STEPS = 2700, 200


@dataclasses.dataclass(frozen=True)
class Scenario:
    """A run of the comparison: how its file is made, where its flows come from."""

    name: str
    flows_from: Path
    windows: Path | None = None


SCENARIOS = {
    "dust80": Scenario(
        "dust80",
        B86 / "runs/reference/dust_80x30.h5",
        B85 / "runs/reference/dust_80x30.h5",
    ),
    "door80": Scenario(
        "door80",
        B86 / "runs/reference/door_80x30.h5",
        B85 / "runs/reference/door_80x30.h5",
    ),
    "breach80": Scenario(
        "breach80",
        B86 / "runs/reference/breach_80x30.h5",
        B85 / "runs/reference/breach_80x30.h5",
    ),
    "clean80": Scenario("clean80", B86 / "runs/reference/dust_80x30.h5"),
    "hold80": Scenario("hold80", B86 / "runs/reference/dust_80x30.h5"),
    "dust320": Scenario("dust320", B81 / "product/equipment_dust.h5"),
    "dust80r": Scenario("dust80r", B86 / "runs/reference/dust_80x30.h5"),
    "hold80r": Scenario("hold80r", B86 / "runs/reference/dust_80x30.h5"),
}
# The pairs compare compares: (label, side and run of each).
PAIRS = [
    *[
        (name, ("reference", name), ("core", name))
        for name in SCENARIOS
        if name not in ("dust80r", "hold80r")
    ],
    ("restart", ("core", "dust80r"), ("core", "dust80r_restart")),
    # The restart's runs checkpoint every 1,000 steps; the dust release does not.
    ("checkpoints", ("core", "dust80"), ("core", "dust80r")),
    # The held clean-out, checkpointed every CLEAN_CADENCE steps, and restarted from
    # before and from inside its within-spec stretch.
    ("clean checkpoints", ("core", "hold80"), ("core", "hold80r")),
    ("clean restart before", ("core", "hold80r"), ("core", "hold80r_before_restart")),
    ("clean restart inside", ("core", "hold80r"), ("core", "hold80r_inside_restart")),
    # Round 1's remade core runs against round 0's, kept in runs_round0/ (made at e152bbd,
    # before the binding's guards and its shared setup).
    *[
        (f"round 0 {name}", ("../runs_round0/core", name), ("core", name))
        for name in ROUND0_RUNS
    ],
]


def _core_run(
    config: SimConfig, run_file: Path | None = None, **kwargs: object
) -> object:
    """``time_integration.run`` with the march through the core, as this probe ran it.

    Prompt 91's ``tests.core_driver.run``, which prompt 96 replaced by the key
    ``solver.backend``: ``_Run`` is replaced by ``time_integration.CoreRun`` for the
    call, the monitor on the fields as prompt 91's default had it.
    """
    from unittest import mock

    driver = type(
        "CoreRunForCall", (time_integration.CoreRun,), {"monitor_reads_fields": True}
    )
    with mock.patch.object(time_integration, "_Run", driver):
        return time_integration.run(config, run_file, **kwargs)


def config_path(name: str) -> Path:
    """A scenario's file."""
    return CONFIGS / f"{name}.yaml"


def local_base(base: str, source: Path) -> str:
    """A scenario's base file named by its path in this checkout, as this process sees it."""
    named = probe90.local_path(base)
    path = Path(named)
    if not path.is_absolute():
        path = (source.parent / path).resolve()
    return str(path)


def write_configs() -> None:
    """Every scenario's file, each loaded once to refuse an invalid one now."""
    CONFIGS.mkdir(parents=True, exist_ok=True)
    mappings: dict[str, dict] = {}
    for scenario in ("dust", "door", "breach"):
        source = B86 / "configs" / "reference" / f"{scenario}_80x30.yaml"
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        raw["scenario"]["base"] = local_base(raw["scenario"]["base"], source)
        mappings[f"{scenario}80"] = raw
    tolerance = mappings["dust80"]["solver"]["mass_imbalance_tol"]
    source = ROOT / "configs" / "scenario_clean_out.yaml"
    clean = yaml.safe_load(source.read_text(encoding="utf-8"))
    clean["scenario"]["base"] = local_base(clean["scenario"]["base"], source)
    clean["domain"] = {"nx": 80, "ny": 30}
    clean["solver"]["mass_imbalance_tol"] = tolerance
    mappings["clean80"] = clean
    held = json.loads(json.dumps(clean))
    held["scenario"]["hold_time"] = HOLD
    mappings["hold80"] = held
    source = B81 / "product" / "configs" / "scenario_equipment_dust.yaml"
    product = yaml.safe_load(source.read_text(encoding="utf-8"))
    product["scenario"]["base"] = local_base(product["scenario"]["base"], source)
    mappings["dust320"] = product
    restart = json.loads(json.dumps(mappings["dust80"]))
    restart["solver"]["checkpoint_interval"] = RESTART_CADENCE
    mappings["dust80r"] = restart
    clean_restart = json.loads(json.dumps(held))
    clean_restart["solver"]["checkpoint_interval"] = CLEAN_CADENCE
    mappings["hold80r"] = clean_restart
    for name, mapping in mappings.items():
        path = config_path(name)
        path.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
        SimConfig(path)
        print("wrote", path)


# ---------------------------------------------------------------------------
# The frame hook


def digest(array: np.ndarray) -> str:
    """A digest of a float64 array's shape and bytes."""
    a = np.ascontiguousarray(array, dtype=np.float64)
    h = hashlib.sha256(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()[:32]


class FrameHook:
    """At every frame written, digests of the run's trackers' state.

    Holds the last ArrivalTracker and RecoveryTracker made in this process (a
    run makes one of each, a restart one more), so the state recorded is the
    trackers' when the frame is written: the reference updates them before its
    frame, the core-driven run sets them from the core's state before its frame.
    """

    def __init__(self) -> None:
        self.arrival: ArrivalTracker | None = None
        self.tracker: RecoveryTracker | None = None
        self.frames: dict[int, dict[str, str]] = {}

    def install(self) -> None:
        """Wrap the two trackers' constructors and RunFile.write_frame."""
        hook = self
        arrival_init = ArrivalTracker.__init__
        tracker_init = RecoveryTracker.__init__
        write_frame = RunFile.write_frame

        def made_arrival(self: ArrivalTracker, *args: object, **kwargs: object) -> None:
            arrival_init(self, *args, **kwargs)
            hook.arrival = self

        def made_tracker(
            self: RecoveryTracker, *args: object, **kwargs: object
        ) -> None:
            tracker_init(self, *args, **kwargs)
            hook.tracker = self

        def frame(
            self: RunFile,
            step: int,
            t: float,
            fields: dict[int, np.ndarray],
            budgets: list[MassBudget],
        ) -> None:
            write_frame(self, step, t, fields, budgets)
            hook.record(int(step))

        ArrivalTracker.__init__ = made_arrival  # type: ignore[method-assign]
        RecoveryTracker.__init__ = made_tracker  # type: ignore[method-assign]
        RunFile.write_frame = frame  # type: ignore[method-assign]

    def record(self, step: int) -> None:
        """The trackers' state at this frame, as digests."""
        out: dict[str, str] = {}
        if self.arrival is not None:
            state = self.arrival.state()
            out["first_above"] = digest(state["first_above"])
            out["arrival_t"] = digest(state["arrival_t"])
        if self.tracker is not None:
            for name, array in self.tracker.state().items():
                out[f"recovery.{name}"] = digest(array)
        self.frames[step] = out


# ---------------------------------------------------------------------------
# The runs


def commit() -> str:
    """HEAD's hash read from the checkout's files (git runs on Windows only)."""
    head = (ROOT / ".git" / "HEAD").read_text().strip()
    if head.startswith("ref: "):
        ref = ROOT / ".git" / head[5:]
        if ref.exists():
            return ref.read_text().strip()
        packed = (ROOT / ".git" / "packed-refs").read_text().splitlines()
        for line in packed:
            if line.endswith(head[5:]):
                return line.split()[0]
        return "unknown"
    return head


def paths(side: str, name: str) -> tuple[Path, Path]:
    """The run file on the Linux file system and its copy's directory."""
    return LINUX_RUNS / side / f"{name}.h5", RUNS / side


def fresh(path: Path) -> None:
    """Remove a run file and its checkpoint file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for p in (path, path.with_name(path.stem + ".checkpoint.h5")):
        if p.exists():
            p.unlink()


def summary_of(record: object, wall: float, side: str, name: str, extra: dict) -> dict:
    """A run's summary in prompt 85's form, with the hold and the platform."""
    from threadpoolctl import threadpool_info

    return {
        "scenario": name,
        "side": side,
        "wall_seconds": wall,
        "commit": commit(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "blas": [
            {k: d.get(k) for k in ("internal_api", "num_threads", "version")}
            for d in threadpool_info()
        ],
        "steps": record.steps,
        "t_final": record.t_final,
        "held": record.held,
        "restarted_from": record.restarted_from,
        "flows": [dataclasses.asdict(f) for f in record.flows],
        "intervals": [dataclasses.asdict(p) for p in record.intervals],
        "budgets": [dataclasses.asdict(b) for b in record.budgets],
        "latencies": record.latencies,
        "map_latencies": record.map_latencies,
        "alerts": [dataclasses.asdict(a) for a in record.alerts],
        **extra,
    }


def finish(
    path: Path, copy_to: Path, summary: dict, hook: FrameHook, name: str
) -> None:
    """Copy the run file into the checkout's results, with its summary and frame digests."""
    copy_to.mkdir(parents=True, exist_ok=True)
    for p in (path, path.with_name(path.stem + ".checkpoint.h5")):
        if p.exists():
            shutil.copy2(p, copy_to / p.name)
    (copy_to / f"{name}.json").write_text(
        json.dumps(probe85._walk(summary), indent=1) + "\n", encoding="utf-8"
    )
    (copy_to / f"{name}.frames.json").write_text(
        json.dumps({str(k): v for k, v in sorted(hook.frames.items())}, indent=1)
        + "\n",
        encoding="utf-8",
    )


def run_one(name: str, side: str) -> None:
    """One scenario from its stored flow states, on one side, with its records."""
    hook = FrameHook()
    hook.install()
    config = SimConfig(config_path(name))
    path, copy_to = paths(side, name)
    fresh(path)
    kwargs = {
        "flows_from": SCENARIOS[name].flows_from,
        "provenance": Provenance.here(commit(), 0),
    }
    started = time.perf_counter()
    if side == "reference":
        record = time_integration.run(config, path, **kwargs)
    else:
        record = _core_run(config, path, **kwargs)
    wall = time.perf_counter() - started
    summary = summary_of(record, wall, side, name, {})
    finish(path, copy_to, summary, hook, name)
    print(name, side, record.steps, f"{wall:.1f} s", "held" if record.held else "")


class StopError(RuntimeError):
    """The crash the restart continues from."""


def restart(
    scenario: str = "dust80r", stop_at: int = STOP_AT, tag: str = "restart"
) -> None:
    """A scenario through the core, stopped after a step, restarted through the core.

    Parameters
    ----------
    scenario : str
        A key of SCENARIOS whose configuration checkpoints.
    stop_at : int
        The step an exception from ``on_step`` ends the first run after.
    tag : str
        The restarted run's name is ``<scenario>_<tag>``.
    """
    name = f"{scenario}_{tag}"
    hook = FrameHook()
    hook.install()
    config = SimConfig(config_path(scenario))
    path, copy_to = paths("core", name)
    fresh(path)

    def crash(step: object) -> None:
        if step.step == stop_at:
            raise StopError(f"stopped after step {stop_at}")

    started = time.perf_counter()
    try:
        _core_run(
            config,
            path,
            flows_from=SCENARIOS[scenario].flows_from,
            on_step=crash,
            provenance=Provenance.here(commit(), 0),
        )
    except StopError as stopped:
        print(stopped)
    else:
        raise SystemExit("the run was not stopped")
    stopped_at = time.perf_counter() - started
    # What the checkpoint carries: the recovery state a restart must hand back.
    carried = RunFile(path).read_checkpoint()
    recovery = {
        key: carried.recovery[key].tolist()
        for key in ("room_since", "class_since", "mean_time")
        if key in carried.recovery
    }
    hook.frames.clear()
    started = time.perf_counter()
    record = _core_run(config, restart=path)
    wall = time.perf_counter() - started
    summary = summary_of(
        record,
        wall,
        "core",
        name,
        {
            "stopped_after": stop_at,
            "stopped_wall_seconds": stopped_at,
            "checkpoint_step": carried.step,
            "checkpoint_t": carried.t,
            "checkpoint_recovery": recovery,
        },
    )
    finish(path, copy_to, summary, hook, name)
    print(name, record.steps, "restarted from", record.restarted_from, f"{wall:.1f} s")


def restart_clean() -> None:
    """The held clean-out checkpointed, and restarted from before and from inside its stretch."""
    run_one("hold80r", "core")
    for tag, stop_at in CLEAN_STOPS.items():
        restart("hold80r", stop_at, tag)


# ---------------------------------------------------------------------------
# The comparison


def bits_of(value: object) -> object:
    """A value's exact form: arrays by dtype, shape and bytes, floats by hex."""
    if isinstance(value, np.ndarray):
        return (str(value.dtype), value.shape, value.tobytes())
    if isinstance(value, (float, np.floating)):
        return float(value).hex()
    if isinstance(value, (list, tuple)):
        return tuple(bits_of(v) for v in value)
    if isinstance(value, dict):
        return tuple((k, bits_of(v)) for k, v in sorted(value.items()))
    if dataclasses.is_dataclass(value):
        return tuple(
            (f.name, bits_of(getattr(value, f.name))) for f in dataclasses.fields(value)
        )
    return repr(value)


def differing(a: object, b: object) -> list[str]:
    """The fields of two records whose exact forms differ."""
    if dataclasses.is_dataclass(a):
        return [
            f.name
            for f in dataclasses.fields(a)
            if bits_of(getattr(a, f.name)) != bits_of(getattr(b, f.name))
        ]
    return [] if bits_of(a) == bits_of(b) else ["value"]


def budget_terms(b: MassBudget) -> tuple[object, ...]:
    """A budget's ten terms, exact."""
    return bits_of(
        [
            b.initial,
            b.inflow,
            b.outflow,
            b.source,
            *b.deposited.values(),
            b.current,
            b.leftover,
        ]
    )


def first_index(flags: list[bool], keys: list[int]) -> int | None:
    """The key of the first True, or None."""
    for flag, key in zip(flags, keys, strict=True):
        if flag:
            return key
    return None


def verdicts(summary: dict, limit: float) -> list[dict]:
    """Each class's REQ-T05 numbers at the end: the leftover and the closure over the supply."""
    out = []
    for b in summary["budgets"]:
        budget = MassBudget(
            initial=b["initial"],
            inflow=b["inflow"],
            outflow=b["outflow"],
            source=b["source"],
            deposited=b["deposited"],
            current=b["current"],
            leftover=b["leftover"],
        )
        supplied = budget.initial + budget.inflow + budget.source
        leftover = budget.leftover / supplied if supplied else 0.0
        closure = budget.residual() / supplied if supplied else 0.0
        out.append(
            {
                "leftover_over_supplied": leftover,
                "closure_over_supplied": closure,
                "within": abs(leftover) < limit and abs(closure) < limit,
            }
        )
    return out


def compare_pair(a: tuple[str, str], b: tuple[str, str]) -> dict:
    """Two runs' files, digests and summaries, bit for bit."""
    fa, fb = RunFile(RUNS / a[0] / f"{a[1]}.h5"), RunFile(RUNS / b[0] / f"{b[1]}.h5")
    sa = json.loads((RUNS / a[0] / f"{a[1]}.json").read_text())
    sb = json.loads((RUNS / b[0] / f"{b[1]}.json").read_text())
    da = json.loads((RUNS / a[0] / f"{a[1]}.frames.json").read_text())
    db = json.loads((RUNS / b[0] / f"{b[1]}.frames.json").read_text())
    out: dict = {"a": list(a), "b": list(b)}
    ra, rb = fa.read_frames(), fb.read_frames()
    steps = [int(s) for s in ra.steps]
    out["frames"] = len(steps)
    out["frame_steps_equal"] = bits_of(ra.steps) == bits_of(rb.steps) and bits_of(
        ra.times
    ) == bits_of(rb.times)
    classes = sorted(ra.fields)
    out["classes"] = len(classes)
    fields_differ = []
    budgets_differ = []
    if out["frame_steps_equal"]:
        for f in range(len(steps)):
            fields_differ.append(
                any(
                    bits_of(ra.fields[k][f]) != bits_of(rb.fields[k][f])
                    for k in classes
                )
            )
            budgets_differ.append(
                any(
                    budget_terms(x) != budget_terms(y)
                    for x, y in zip(ra.budgets[f], rb.budgets[f], strict=True)
                )
            )
        out["fields_frames_differing"] = sum(fields_differ)
        out["budgets_frames_differing"] = sum(budgets_differ)
        out["first_field_difference"] = first_index(fields_differ, steps)
        out["first_budget_difference"] = first_index(budgets_differ, steps)
    xa, xb = fa.read_sensor_readings(), fb.read_sensor_readings()
    out["sensor_rows"] = int(xa.steps.size)
    out["sensor_steps_equal"] = bits_of(xa.steps) == bits_of(xb.steps) and bits_of(
        xa.times
    ) == bits_of(xb.times)
    if out["sensor_steps_equal"]:
        rows = [
            bits_of(xa.readings[r]) != bits_of(xb.readings[r])
            or bits_of(xa.channels[r]) != bits_of(xb.channels[r])
            for r in range(xa.steps.size)
        ]
        out["sensor_rows_differing"] = sum(rows)
        out["first_sensor_difference"] = first_index(rows, [int(s) for s in xa.steps])
    out["alerts_differing"] = differing(fa.read_alerts(), fb.read_alerts())
    out["arrival_differing"] = differing(fa.read_arrival(), fb.read_arrival())
    try:
        recovery_a, recovery_b = fa.read_recovery(), fb.read_recovery()
    except KeyError:
        recovery_a = recovery_b = None
    out["recovery"] = recovery_a is not None
    out["recovery_differing"] = (
        [] if recovery_a is None else differing(recovery_a, recovery_b)
    )
    common = sorted(set(da) & set(db), key=int)
    out["digest_frames"] = len(common)
    digest_differ = {}
    for key in sorted({k for v in da.values() for k in v}):
        flags = [da[s].get(key) != db[s].get(key) for s in common]
        digest_differ[key] = {
            "frames_differing": sum(flags),
            "first": first_index(flags, [int(s) for s in common]),
        }
    out["digests"] = digest_differ
    decisions = {}
    for key in (
        "steps",
        "t_final",
        "held",
        "latencies",
        "map_latencies",
        "alerts",
        "budgets",
    ):
        decisions[key] = sa[key] == sb[key]
    out["decisions_equal"] = decisions
    config = SimConfig.from_dict(fa.read_config().mapping)
    limit = config.transport.conservation_limit
    out["verdicts"] = {"a": verdicts(sa, limit), "b": verdicts(sb, limit)}
    out["held"] = sa["held"]
    out["steps"] = sa["steps"]
    out["wall_seconds"] = {"a": sa["wall_seconds"], "b": sb["wall_seconds"]}
    out["flows"] = [f["origin"] for f in sb["flows"]]
    out["bitwise"] = bool(
        out["frame_steps_equal"]
        and out.get("fields_frames_differing") == 0
        and out.get("budgets_frames_differing") == 0
        and out["sensor_steps_equal"]
        and out.get("sensor_rows_differing") == 0
        and not out["alerts_differing"]
        and not out["arrival_differing"]
        and not out["recovery_differing"]
        and all(v["frames_differing"] == 0 for v in digest_differ.values())
        and all(decisions.values())
    )
    return out


def step1_reference(name: str) -> dict:
    """This pass's Linux reference against prompt 86's (step 1's) on the frames and sensors."""
    scenario = {"dust80": "dust", "door80": "door", "breach80": "breach"}[name]
    old = RunFile(B86 / "runs" / "reference" / f"{scenario}_80x30.h5")
    new = RunFile(RUNS / "reference" / f"{name}.h5")
    fo, fn = old.read_frames(), new.read_frames()
    so, sn = old.read_sensor_readings(), new.read_sensor_readings()
    return {
        "frames": int(fo.steps.size),
        "fields_equal": all(
            bits_of(fo.fields[k]) == bits_of(fn.fields[k]) for k in fo.fields
        ),
        "budgets_equal": all(
            budget_terms(x) == budget_terms(y)
            for fx, fy in zip(fo.budgets, fn.budgets, strict=True)
            for x, y in zip(fx, fy, strict=True)
        ),
        "sensors_equal": bits_of(so.channels) == bits_of(sn.channels)
        and bits_of(so.readings) == bits_of(sn.readings),
        "arrival_equal": not differing(old.read_arrival(), new.read_arrival()),
    }


def compare() -> None:
    """Every pair, and this pass's references against step 1's."""
    out: dict = {"pairs": {}, "step1": {}}
    for label, a, b in PAIRS:
        if not all((RUNS / side / f"{run}.json").exists() for side, run in (a, b)):
            print(label, "missing")
            continue
        out["pairs"][label] = compare_pair(a, b)
        print(label, "bitwise" if out["pairs"][label]["bitwise"] else "DIFFERS")
    for name in ("dust80", "door80", "breach80"):
        out["step1"][name] = step1_reference(name)
        print("step 1", name, out["step1"][name])
    (RESULTS / "compare.json").write_text(
        json.dumps(probe85._walk(out), indent=1) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# The timing


def window_run(
    kind: type, config: SimConfig, path: Path, stored: object
) -> tuple[object, list, tuple[int, int], dict, list, int, float]:
    """A run object on the 320x120 dust release, its plans cut to the window."""
    from src.io_manager import RunFile as File

    fresh(path)
    run_file = File(path)
    given = Provenance.here(commit(), 0)
    run_file.write_config(
        config.merged_mapping(),
        commit=given.commit,
        started=given.started,
        machine=given.machine,
        concurrent=given.concurrent,
    )
    state = kind(config, run_file, None)
    state.flow_states(File(SCENARIOS["dust320"].flows_from), False)
    plans = state.plans()
    f = int(np.flatnonzero(stored.steps == WINDOW_FROM)[0])
    t = float(stored.times[f])
    fields = {k: np.array(stored.fields[k][f]) for k in sorted(stored.fields)}
    for c in fields.values():
        c.flags.writeable = False
    budgets = list(stored.budgets[f])
    # The plan holding t at the step whose time it is, and the window's end.
    out, start, left = [], None, WINDOW_STEPS
    for plan in plans:
        if start is None:
            if not plan.t_start <= t < plan.t_end:
                continue
            i = round((t - plan.t_start) / plan.dt)
            if plan.time(i) != t:
                raise SystemExit("the window's frame is not on its plan's step")
            start = (0, i)
            take = min(plan.steps - i, left)
            out.append(
                dataclasses.replace(plan, steps=i + take, t_end=plan.time(i + take))
            )
        else:
            take = min(plan.steps, left)
            out.append(dataclasses.replace(plan, steps=take, t_end=plan.time(take)))
        left -= take
        if left == 0:
            break
    assert start is not None
    return state, out, start, fields, budgets, WINDOW_FROM, t


def timing(run_number: int) -> None:
    """The window three ways in one process, each from the same frame, timed once each in turn."""
    from src.time_integration import _Run

    load_before = probe90.windows_load()
    calib_before = probe90.calibration()
    config = SimConfig(config_path("dust320"))
    stored = RunFile(RUNS / "reference" / "dust320.h5").read_frames()
    work = LINUX_RUNS / "timing"
    kinds: dict[str, type] = {
        "reference": _Run,
        "core, monitor on the fields": type(
            "Fields", (time_integration.CoreRun,), {"monitor_reads_fields": True}
        ),
        "core, monitor on the readings": type(
            "Readings", (time_integration.CoreRun,), {"monitor_reads_fields": False}
        ),
    }
    rows = []
    finals: dict[str, object] = {}
    for label, kind in kinds.items():
        state, plans, start, fields, budgets, step, t = window_run(
            kind, config, work / f"{label.split(',')[0]}_{len(rows)}.h5", stored
        )
        state.arrival.update(t, fields)
        started = time.perf_counter()
        out = state.march(plans, start, fields, budgets, step, t)
        wall = time.perf_counter() - started
        finals[label] = out
        rows.append(
            {
                "march": label,
                "steps": out[2] - step,
                "wall_s": wall,
                "core_calls": getattr(state, "core_calls", None),
            }
        )
        print(label, out[2] - step, f"{wall:.2f} s")
    reference = finals["reference"]
    same = {
        label: bool(
            all(bits_of(out[0][k]) == bits_of(reference[0][k]) for k in reference[0])
            and [budget_terms(b) for b in out[1]]
            == [budget_terms(b) for b in reference[1]]
            and out[2:] == reference[2:]
        )
        for label, out in finals.items()
    }
    record = {
        "run": run_number,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "window": {"from_step": WINDOW_FROM, "steps": WINDOW_STEPS},
        "windows_before": load_before,
        "calibration_before": calib_before,
        "rows": rows,
        "ratios": {
            r["march"]: rows[0]["wall_s"] / r["wall_s"]
            for r in rows
            if r["march"] != "reference"
        },
        "ends_equal_the_reference": same,
        "calibration_after": probe90.calibration(),
        "windows_after": probe90.windows_load(),
    }
    (RESULTS / f"timing_{run_number}.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# The planted defects

PLANT_WORK = Path.home() / "scratch91" / "plants"
PLANT_RUNS = ("dust80", "hold80")


def whole_run_differences(name: str) -> str:
    """A core-driven run of NAME on this process's module against the reference's files."""
    work = LINUX_RUNS / "plant"
    path = work / f"{name}.h5"
    fresh(path)
    config = SimConfig(config_path(name))
    try:
        _core_run(config, path, flows_from=SCENARIOS[name].flows_from)
    except Exception as error:  # a plant may break the run in any way
        return f"raised {type(error).__name__}"
    ref = RunFile(RUNS / "reference" / f"{name}.h5")
    mine = RunFile(path)
    fr, fm = ref.read_frames(), mine.read_frames()
    if bits_of(fr.steps) != bits_of(fm.steps):
        return f"frames at other steps ({fm.steps.size} against {fr.steps.size})"
    differ = sum(
        any(bits_of(fr.fields[k][f]) != bits_of(fm.fields[k][f]) for k in fr.fields)
        or any(
            budget_terms(x) != budget_terms(y)
            for x, y in zip(fr.budgets[f], fm.budgets[f], strict=True)
        )
        for f in range(fr.steps.size)
    )
    sr, sm = ref.read_sensor_readings(), mine.read_sensor_readings()
    sensors = bits_of(sr.channels) == bits_of(sm.channels) and bits_of(
        sr.steps
    ) == bits_of(sm.steps)
    arrival = not differing(ref.read_arrival(), mine.read_arrival())
    recovery = True
    with contextlib.suppress(KeyError):
        recovery = not differing(ref.read_recovery(), mine.read_recovery())
    if differ == 0 and sensors and arrival and recovery:
        return "equal"
    parts = [f"{differ}/{fr.steps.size} frames"]
    if not sensors:
        parts.append("sensors")
    if not arrival:
        parts.append("arrival")
    if not recovery:
        parts.append("recovery")
    return "differs: " + ", ".join(parts)


def test_plant(
    name: str, what: str, edits: list, tree: Path, run_dir: Path = ROOT
) -> dict[str, object]:
    """One planted defect built in ``tree``, with CTest and the march's parity module.

    Parameters
    ----------
    name, what : str
        The plant's name and what it changes.
    edits : list
        ``(file, old, new)`` edits to the checkout's ``core/``.
    tree : Path
        The scratch tree: its ``core/`` is rewritten where it differs (so ninja
        rebuilds only what changed) and built in its ``build/``.
    run_dir : Path
        The checkout whose tests run.

    Returns
    -------
    dict[str, object]
        The record: ``ctest``, ``ctest_failed``, ``pytest`` and
        ``parity_failed_by_test``, or ``build`` when the build failed.
    """
    import re
    from collections import Counter

    source = tree / "core"
    build = tree / "build"
    files = probe90.plant_sources(edits)
    for relative, text in files.items():
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # Written only when it differs, so ninja rebuilds what changed (a copy
        # keeps old file times and ninja would not: prompt 88's lesson).
        if not target.exists() or target.read_text(encoding="utf-8") != text:
            target.write_text(text, encoding="utf-8", newline="\n")
    record: dict[str, object] = {"name": name, "what": what}
    configure = subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-G",
            "Ninja",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST={probe90.GTEST_SRC}",
        ],
        capture_output=True,
        text=True,
    )
    made = subprocess.run(["ninja", "-C", str(build)], capture_output=True, text=True)
    if configure.returncode or made.returncode:
        record["build"] = "failed: " + (made.stdout + made.stderr)[-1500:]
        return record
    ctest = subprocess.run(
        ["ctest", "--test-dir", str(build)], capture_output=True, text=True
    )
    summary = [line for line in ctest.stdout.splitlines() if "tests passed" in line]
    record["ctest"] = summary[0].strip() if summary else ctest.stdout[-300:]
    record["ctest_failed"] = re.findall(r"\d+ - (\S+) \([^)]+\)", ctest.stdout)
    env = {**os.environ, "PYTHONPATH": str(build), "CFD_CORE_REQUIRED": "1"}
    pytest = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_core_march.py",
            "-q",
            "-rf",
            "--tb=no",
            "-p",
            "no:cacheprovider",
        ],
        capture_output=True,
        text=True,
        cwd=run_dir,
        env=env,
    )
    lines = pytest.stdout.splitlines()
    # A run that died (a plant that makes the binding crash) has no summary line.
    record["pytest_returncode"] = pytest.returncode
    record["pytest"] = next(
        (
            line.strip("= ")
            for line in reversed(lines)
            if re.search(r" in [0-9.]+s", line)
        ),
        lines[-1] if lines else "no output",
    )
    failed = [
        line.split("::", 1)[1].split(" ")[0]
        for line in lines
        if line.startswith("FAILED tests/test_core_march.py::")
    ]
    record["parity_failed_by_test"] = dict(
        sorted(Counter(case.split("[")[0] for case in failed).items())
    )
    return record


def plants() -> None:
    """Each planted defect built in turn: CTest, the march's parity module, two whole runs."""
    sys.path.insert(0, str(HERE))
    import variants91

    records = []
    only = sys.argv[sys.argv.index("--only") + 1 :] if "--only" in sys.argv else None
    for name, what, edits in variants91.PLANTS:
        if only and name not in only:
            continue
        record = test_plant(name, what, edits, PLANT_WORK)
        if "build" in record:
            records.append(record)
            print(name, "build failed")
            continue
        env = {
            **os.environ,
            "PYTHONPATH": str(PLANT_WORK / "build"),
            "CFD_CORE_REQUIRED": "1",
        }
        whole = subprocess.run(
            [sys.executable, "-u", str(Path(__file__).resolve()), "plant-runs"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            env=env,
        )
        record["whole_runs"] = dict(
            line.split(": ", 1) for line in whole.stdout.splitlines() if ": " in line
        )
        if whole.returncode:
            record["whole_runs"]["error"] = whole.stderr[-500:]
        records.append(record)
        print(
            name, "|", record["ctest"], "|", record["pytest"], "|", record["whole_runs"]
        )
        out = RESULTS / ("plants.json" if not only else f"plants_{only[0]}.json")
        out.write_text(json.dumps(records, indent=1) + "\n", encoding="utf-8")


def plants_round1(args: argparse.Namespace) -> None:
    """Round 1's planted defects: CTest and the march's parity module on each."""
    sys.path.insert(0, str(HERE))
    import variants91b

    chosen = [p for p in variants91b.PLANTS if not args.only or p[0] in args.only]
    if args.check:
        bad = 0
        for name, _what, edits in chosen:
            try:
                probe90.plant_sources(edits)
            except SystemExit as refused:
                bad += 1
                print(name, "REFUSED:", refused)
        print(f"{len(chosen)} plants, {bad} refused")
        if bad:
            raise SystemExit(1)
        return
    tree = Path.home() / "scratch91" / args.tree
    out = args.out or RESULTS / f"plants_round1_{args.tree}.json"
    records = []
    # The control first and last: the unplanted core passes everything.
    for name, what, edits in [("control", "no edit: every test passes", []), *chosen]:
        record = test_plant(name, what, edits, tree)
        records.append(record)
        print(
            name,
            "|",
            record.get("ctest", record.get("build")),
            "|",
            record.get("pytest"),
            "|",
            record.get("ctest_failed"),
            record.get("parity_failed_by_test"),
            flush=True,
        )
        Path(out).write_text(json.dumps(records, indent=1) + "\n", encoding="utf-8")
    record = test_plant("control", "no edit: every test passes", [], tree)
    record["name"] = "control (after)"
    records.append(record)
    Path(out).write_text(json.dumps(records, indent=1) + "\n", encoding="utf-8")
    print(
        record["name"], "|", record.get("ctest"), "|", record.get("pytest"), flush=True
    )


def plant_runs() -> None:
    """The two whole runs of a plant, on the module PYTHONPATH gives this process."""
    import _cfdcore

    if not str(_cfdcore.__file__).startswith(str(PLANT_WORK)):
        raise SystemExit(
            f"the plant's module is not the one imported: {_cfdcore.__file__}"
        )
    for name in PLANT_RUNS:
        print(f"{name}: {whole_run_differences(name)}")


# ---------------------------------------------------------------------------
# The vectorisation report


def vecinfo() -> None:
    """GCC's vectorised loops in the new modules, counted once by their place in the source."""
    import re

    compiler = subprocess.run(
        ["g++", "--version"], capture_output=True, text=True
    ).stdout
    flags = ["-std=c++20", "-O2", "-ffp-contract=off", "-march=x86-64-v3", "-DNDEBUG"]
    out: dict = {"compiler": compiler.splitlines()[0], "flags": flags, "files": {}}
    for name in ("channels", "arrival", "recovery", "march"):
        done = subprocess.run(
            [
                "g++",
                *flags,
                "-fopt-info-vec-optimized",
                "-I",
                str(ROOT / "core" / "include"),
                "-c",
                str(ROOT / "core" / "src" / f"{name}.cpp"),
                "-o",
                "/dev/null",
            ],
            capture_output=True,
            text=True,
        )
        lines = [line for line in done.stderr.splitlines() if "loop vectorized" in line]
        places = sorted(
            {m.group(1) for line in lines if (m := re.search(r"\.cpp:(\d+):", line))}
        )
        source = (ROOT / "core" / "src" / f"{name}.cpp").read_text().splitlines()
        out["files"][f"src/{name}.cpp"] = {
            "report_lines": len(lines),
            "loops": len(places),
            "places": [f"{p}: {source[int(p) - 1].strip()}" for p in places],
        }
        print(name, len(places), "loops", len(lines), "lines")
    (RESULTS / "vecinfo.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run one command of the probe."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("configs")
    one = commands.add_parser("run")
    one.add_argument("scenario", choices=list(SCENARIOS))
    one.add_argument("side", choices=SIDES)
    commands.add_parser("restart")
    commands.add_parser("restart-clean")
    commands.add_parser("compare")
    timed = commands.add_parser("timing")
    timed.add_argument("--run", type=int, required=True)
    planted = commands.add_parser("plants")
    planted.add_argument("--only", nargs="*")
    round_one = commands.add_parser("plants-round1")
    round_one.add_argument("--tree", default="round1")
    round_one.add_argument("--only", nargs="*")
    round_one.add_argument("--out")
    round_one.add_argument("--check", action="store_true")
    commands.add_parser("plant-runs")
    commands.add_parser("vecinfo")
    args = parser.parse_args(argv)
    if args.command == "configs":
        write_configs()
    elif args.command == "run":
        run_one(args.scenario, args.side)
    elif args.command == "restart":
        restart()
    elif args.command == "restart-clean":
        restart_clean()
    elif args.command == "compare":
        compare()
    elif args.command == "timing":
        timing(args.run)
    elif args.command == "plants":
        plants()
    elif args.command == "plants-round1":
        plants_round1(args)
    elif args.command == "plant-runs":
        plant_runs()
    elif args.command == "vecinfo":
        vecinfo()
    return 0


if __name__ == "__main__":
    sys.exit(main())
