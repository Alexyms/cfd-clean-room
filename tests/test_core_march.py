"""The compiled core's march against the NumPy reference (REQ-N03 level 1, ADR-017 D).

The step loop's blocks (``channels``, ``arrival``, ``recovery``) against the
reference's functions on planted fields, and the march itself (``_cfdcore.March2``
driven by ``tests.core_driver.CoreRun``) against ``time_integration._Run.march``
on the 80x30 product mesh with planted flows, every output compared bit for bit
after every step. Skipped with its reason when ``_cfdcore`` cannot be imported,
an error instead with ``CFD_CORE_REQUIRED=1``, as ``tests/test_core_parity.py``.
"""

import dataclasses
import functools
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.config import SimConfig
from src.io_manager import RunFile
from src.mesh import SOLID, Mesh
from src.monitor import AlertMonitor, ArrivalTracker, channel_sums
from src.solver_staggered import FlowState
from src.solver_transport import MassBudget
from src.time_integration import (
    IntervalPlan,
    MarchStep,
    Provenance,
    RecoveryTracker,
    _Run,
)
from tests import core_driver, core_mapping

if os.environ.get("CFD_CORE_REQUIRED") == "1":
    import _cfdcore
else:
    _cfdcore = pytest.importorskip(
        "_cfdcore",
        reason=(
            "the compiled core is not built: "
            "pip install --no-build-isolation ./core under WSL2 (ADR-017 A)"
        ),
    )

pytestmark = [pytest.mark.parity, pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
SEED = 91
# The rooms the blocks are planted on: the product room and the vents room.
ROOMS = {"default": "clean_room_default.yaml", "vents": "clean_room_vents.yaml"}
GRIDS = [(80, 30), (160, 60)]
# The scenarios the march is planted on, at 80x30.
SCENARIOS = {
    "dust": "scenario_equipment_dust.yaml",
    "door": "scenario_door_leak.yaml",
    "clean": "scenario_clean_out.yaml",
}


def _bits(a: np.ndarray | float | None) -> bytes | None:
    """The bytes of a float64 array or number, so that 0.0 and -0.0 differ."""
    if a is None:
        return None
    return np.ascontiguousarray(a, dtype=np.float64).tobytes()


@functools.cache
def _room(name: str, nx: int, ny: int) -> SimConfig:
    """A room's base file on an nx by ny grid."""
    raw = yaml.safe_load((CONFIGS / ROOMS[name]).read_text(encoding="utf-8"))
    raw["domain"] = {**raw.get("domain", {}), "nx": nx, "ny": ny}
    return SimConfig.from_dict(raw, CONFIGS)


@functools.cache
def _mesh(name: str, nx: int, ny: int) -> Mesh:
    return Mesh(_room(name, nx, ny))


def _grid(mesh: Mesh) -> "_cfdcore.Grid2":
    return _cfdcore.Grid2(mesh.x, mesh.xc, mesh.y, mesh.yc, mesh.cell_type, SOLID)


def _signed_fields(
    rng: np.random.Generator, n: int, shape: tuple[int, int]
) -> np.ndarray:
    """Values of both signs over many decades, so the order of a sum shows."""
    return rng.standard_normal((n, *shape)) * 10.0 ** rng.uniform(-6, 6, (n, *shape))


def _channel_sets(n: int, cumulative: list[list[int]]) -> dict[str, list[list[int]]]:
    """The configuration's channels and other orders of the same and other members."""
    return {
        "cumulative": cumulative,
        "cumulative reversed": [list(reversed(m)) for m in cumulative],
        "each alone": [[k] for k in range(n)],
        "mixed": [[k, *range(n - 1, k, -2)] for k in range(n)],
    }


# ---------------------------------------------------------------------------
# channels


@pytest.mark.parametrize("room", list(ROOMS))
@pytest.mark.parametrize(("nx", "ny"), GRIDS)
def test_channel_fields_equal_channel_sums(room: str, nx: int, ny: int) -> None:
    """Every channel's field is monitor.channel_sums' bit for bit, in every order of members."""
    mesh = _mesh(room, nx, ny)
    config = _room(room, nx, ny)
    n = len(config.particle_sizes)
    rng = np.random.default_rng([SEED, nx, len(room)])
    fields = _signed_fields(rng, n, mesh.cell_type.shape)
    fields[:, 0, 0] = -0.0  # the zero start makes a channel of -0.0 terms +0.0
    grid = _grid(mesh)
    results = {}
    for name, channels in _channel_sets(n, config.channel_classes()).items():
        reference = channel_sums({k: fields[k] for k in range(n)}, channels)
        core = _cfdcore.channel_fields(grid, fields, channels)
        for k in range(n):
            assert _bits(core[k]) == _bits(reference[k]), (name, k)
        results[name] = core
    # The comparison is sharp: another order of the same members gives other bits.
    assert _bits(results["cumulative"][0]) != _bits(results["cumulative reversed"][0])
    assert np.signbit(results["each alone"][:, 0, 0]).sum() == 0


def test_sensor_channels_equal_the_monitors_and_the_fields_at_the_sensors() -> None:
    """The channels at the sensors are the monitor's, and the channel fields read at the cells."""
    config = _room("default", 80, 30)
    mesh = _mesh("default", 80, 30)
    n = len(config.particle_sizes)
    rng = np.random.default_rng([SEED, 1])
    fields = np.abs(_signed_fields(rng, n, mesh.cell_type.shape))
    fields[:, mesh.cell_type == SOLID] = 0.0
    monitor = AlertMonitor(config, mesh)
    monitor.update(1, 0.5, {k: fields[k] for k in range(n)})
    channels = [list(m) for m in config.channel_classes()]
    core = _cfdcore.sensor_channels(monitor.readings, channels)
    assert _bits(core) == _bits(monitor.channel_readings)
    whole = _cfdcore.channel_fields(_grid(mesh), fields, channels)
    rows = [j for j, _ in core_mapping_cells(config, mesh)]
    cols = [i for _, i in core_mapping_cells(config, mesh)]
    assert _bits(whole[:, rows, cols]) == _bits(core)


def core_mapping_cells(config: SimConfig, mesh: Mesh) -> tuple[tuple[int, int], ...]:
    """The configuration's sensors' cells."""
    from src.monitor import sensor_cells

    return sensor_cells(config, mesh)


# ---------------------------------------------------------------------------
# arrival


def _near(
    rng: np.random.Generator, thresholds: np.ndarray, shape: tuple[int, ...]
) -> np.ndarray:
    """Per class, values around its threshold: below, above and far from it."""
    scale = 10.0 ** rng.uniform(-1.0, 1.0, shape)
    return thresholds[:, None, None] * scale


def _plant_at_thresholds(
    fields: np.ndarray, thresholds: np.ndarray, cells: list[tuple[int, int]]
) -> None:
    """Channel k exactly at its threshold at cell k, and one ulp below and above beside it.

    Only class k is non-zero at those cells, so every channel that sums it reads
    the class's value there; channel k reads it against its own threshold.
    """
    for k, (j, i) in enumerate(cells):
        for di, value in enumerate(
            (
                thresholds[k],
                np.nextafter(thresholds[k], 0.0),
                np.nextafter(thresholds[k], np.inf),
            )
        ):
            fields[:, j, i + di] = 0.0
            fields[k, j, i + di] = value


@pytest.mark.parametrize("room", list(ROOMS))
def test_arrival_maps_equal_the_trackers(room: str) -> None:
    """ArrivalMaps2 equals ArrivalTracker bit for bit after every update.

    Planted: values around every threshold with a cell exactly at it and one ulp
    either side, several events with their onsets (two sharing one, one after
    the first updates), large values in the SOLID cells, and a cell that is
    reached, falls below and rises again.
    """
    config = _room(room, 80, 30)
    mesh = _mesh(room, 80, 30)
    thresholds = np.array(config.class_thresholds("the test"), dtype=np.float64)
    n = thresholds.size
    channels = [list(m) for m in config.channel_classes()]
    onsets = [0.0, 1.0, 1.0, 2.5]
    live = mesh.cell_type != SOLID
    reference = ArrivalTracker(thresholds, live, onsets, channels=channels)
    core = _cfdcore.ArrivalMaps2(_grid(mesh), list(thresholds), onsets, channels)
    rng = np.random.default_rng([SEED, 2, len(room)])
    free = [
        (j, i) for j, i in zip(*np.nonzero(live), strict=True) if i + 2 < live.shape[1]
    ]
    planted = [
        free[m]
        for m in rng.choice(len(free), n, replace=False)
        if live[free[m][0], free[m][1] : free[m][1] + 3].all()
    ]
    times = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    for u, t in enumerate(times):
        fields = _near(rng, thresholds, (n, *live.shape))
        fields[:, ~live] = 1e30
        _plant_at_thresholds(fields, thresholds, planted)
        if u == 2:  # the watched cell falls below every threshold
            fields[:, planted[0][0], planted[0][1]] = 0.0
        reference.update(t, {k: fields[k] for k in range(n)})
        core.update(t, np.ascontiguousarray(fields))
        assert _bits(core.first_above) == _bits(reference.first_above), t
        assert core.t == t
    maps = reference.first_above
    j, i = planted[0]
    # The planted inputs reach the rules: a cell at its threshold is reached and
    # one ulp below is not; a time once set is kept after the cell falls below;
    # no map holds a time before its event's onset; SOLID cells stay NaN.
    assert maps[0, 0, j, i] == times[0]
    assert np.isnan(maps[0, 0, j, i + 1]) or maps[0, 0, j, i + 1] > times[0]
    assert np.isnan(maps[:, :, ~live]).all()
    assert np.nanmin(maps[3]) == 2.5
    assert np.nanmin(maps[1]) == 1.0
    # The core's restore takes the maps back.
    again = _cfdcore.ArrivalMaps2(_grid(mesh), list(thresholds), onsets, channels)
    again.restore(maps, times[-1])
    assert _bits(again.first_above) == _bits(maps)


# ---------------------------------------------------------------------------
# recovery


def _recovery_pair(
    room: str, hold: float | None
) -> tuple[RecoveryTracker, "_cfdcore.Recovery2", SimConfig, Mesh, np.ndarray]:
    config = _room(room, 80, 30)
    mesh = _mesh(room, 80, 30)
    thresholds = np.array(config.class_thresholds("the test"), dtype=np.float64)
    channels = [list(m) for m in config.channel_classes()]
    cells = list(core_mapping_cells(config, mesh))
    # Class 3's initial level a hundredth of a hundred times its threshold, so its
    # 100:1 target is not defined; the others' at a hundred times exactly.
    initial = 100.0 * thresholds
    initial[3] = thresholds[3]
    live = mesh.cell_type != SOLID
    reference = RecoveryTracker(
        thresholds, initial, cells, live, hold, channels=channels
    )
    keys = core_mapping.recovery_keys(thresholds, initial, hold)
    core = _cfdcore.Recovery2(_grid(mesh), list(thresholds), keys, cells, channels)
    return reference, core, config, mesh, thresholds


@pytest.mark.parametrize("room", list(ROOMS))
def test_recovery_equals_the_tracker(room: str) -> None:
    """Recovery2 equals RecoveryTracker bit for bit after every update, held() included.

    Planted: cells and sensors exactly at a threshold, readings and means
    exactly at the 100:1 target, a class whose target is not defined, a stretch
    within spec whose length is the hold exactly, then a sensor above again.
    """
    times = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    within_from, held_at = 2, 5
    hold = times[held_at] - times[within_from]
    reference, core, config, mesh, thresholds = _recovery_pair(room, hold)
    n = thresholds.size
    live = mesh.cell_type != SOLID
    cells = list(core_mapping_cells(config, mesh))
    rng = np.random.default_rng([SEED, 3, len(room)])
    target = core_mapping.recovery_keys(thresholds, 100.0 * thresholds, None)["target"]
    held = []
    for u, t in enumerate(times):
        fields = _near(rng, thresholds, (n, *live.shape))
        fields[:, ~live] = 1e30
        for s, (j, i) in enumerate(cells):
            if within_from <= u < held_at + 2:
                fields[:, j, i] = 0.0  # every sensor within spec
            if u == held_at + 2:
                fields[:, j, i] = thresholds[None, :][
                    0
                ]  # at the thresholds: not within
            if u == 0 and s == 0:
                fields[:, j, i] = 10.0 * np.asarray(target)  # above the targets first
            if u == 1 and s == 0:
                fields[:, j, i] = np.asarray(target)  # then each reading at its target
        if u == 0:
            _plant_at_thresholds(fields, thresholds, [(j, 1) for j in range(3, 3 + n)])
        means = [float(x) for x in rng.uniform(0.5, 2.0, n) * np.asarray(target)]
        if u == 3:
            means = [float(x) for x in target]
        reference.update(t, {k: fields[k] for k in range(n)}, means)
        core.update(t, np.ascontiguousarray(fields), means)
        state = core.state()
        for name, array in reference.state().items():
            assert _bits(state[name]) == _bits(array), (t, name)
        assert core.held() == reference.held(), t
        held.append(reference.held())
    # The planted inputs reach the rules: the hold holds at equality and ends
    # when a sensor reads its threshold; a reading at its target sets its time;
    # the class without a defined target never does.
    assert held[held_at] and not held[held_at - 1] and not held[held_at + 2]
    record = reference.state()
    assert record["sensor_time"][0, 0] == times[1]
    assert np.isnan(record["sensor_time"][3]).all() and np.isnan(record["mean_time"][3])
    assert record["mean_time"][0] <= times[3]
    again = _cfdcore.Recovery2(
        _grid(mesh),
        list(thresholds),
        core_mapping.recovery_keys(thresholds, 100.0 * thresholds, hold),
        cells,
        [list(m) for m in config.channel_classes()],
    )
    again.restore(core.state())
    assert all(_bits(again.state()[k]) == _bits(v) for k, v in core.state().items())


# ---------------------------------------------------------------------------
# the march


def _scenario(name: str, hold: float | None = None) -> SimConfig:
    """A reference scenario on 80x30, with a hold for the clean-out when given."""
    raw = yaml.safe_load((CONFIGS / SCENARIOS[name]).read_text(encoding="utf-8"))
    raw["domain"] = {"nx": 80, "ny": 30}
    if hold is not None:
        raw["scenario"]["hold_time"] = hold
    return SimConfig.from_dict(raw, CONFIGS)


def _planted_flow(mesh: Mesh, rng: np.random.Generator) -> FlowState:
    """Faces of both signs about 0.2 m/s with a smooth part, and a positive eddy viscosity."""
    ny, nx = mesh.cell_type.shape
    yu, xu = np.meshgrid(np.arange(ny), np.arange(nx + 1), indexing="ij")
    yv, xv = np.meshgrid(np.arange(ny + 1), np.arange(nx), indexing="ij")
    u = 0.15 * np.sin(0.2 * xu + 0.3 * yu) + 0.05 * rng.standard_normal((ny, nx + 1))
    v = 0.15 * np.cos(0.25 * xv - 0.2 * yv) + 0.05 * rng.standard_normal((ny + 1, nx))
    nu_t = 1e-4 * (1.0 + rng.uniform(0.0, 10.0, (ny, nx)))
    k = 1e-3 * (1.0 + rng.uniform(0.0, 1.0, (ny, nx)))
    eps = 1e-3 * (1.0 + rng.uniform(0.0, 1.0, (ny, nx)))
    return FlowState(u=u, v=v, p=np.zeros((ny, nx)), k=k, eps=eps, nu_t=nu_t, outer=1)


@dataclasses.dataclass
class Case:
    """A march planted on a scenario: its plans, start, fields and what it holds for."""

    name: str
    config: SimConfig
    plans: list[IntervalPlan]
    start: tuple[int, int]
    fields: dict[int, np.ndarray]


def _world(case: Case, tmp: Path, kind: type[_Run], tag: str) -> _Run:
    """A run of the case's configuration on planted flows, its trackers at the start."""
    run_file = RunFile(tmp / f"{case.name}_{tag}.h5")
    given = Provenance.here("test")
    run_file.write_config(
        case.config.merged_mapping(),
        commit=given.commit,
        started=given.started,
        machine=given.machine,
        concurrent=given.concurrent,
    )
    state = kind(case.config, run_file, None)
    rng = np.random.default_rng([SEED, 5])
    for interval in state.intervals:
        if interval.fingerprint not in state.states:
            state.states[interval.fingerprint] = _planted_flow(state.mesh, rng)
    return state


def _sliced(
    plans: list[IntervalPlan], keep: list[tuple[int, int, int]]
) -> tuple[list[IntervalPlan], tuple[int, int]]:
    """Plans cut to (plan, first, last) steps, each ending at its last step's time."""
    out = []
    for p, _first, last in keep:
        plan = plans[p]
        out.append(dataclasses.replace(plan, steps=last, t_end=plan.time(last)))
    return out, (0, keep[0][1])


def _case(name: str, tmp: Path) -> Case:
    """One of the planted marches.

    Across the dust release's onset; across the door leak's two flow switches;
    the clean-out from fields around its thresholds; the clean-out within spec
    from the start, held at its fifth step; the door leak's last interval to its
    end, whose last step's time is ``t_end`` and not ``t_start + steps dt``.
    """
    rng = np.random.default_rng([SEED, 4, len(name)])
    probe = _scenario({"held": "clean", "door_end": "door"}.get(name, name))
    world = _World(probe, tmp)
    plans = world.state.plans()
    thresholds = np.array(probe.class_thresholds("the test"), dtype=np.float64)
    shape = world.state.mesh.cell_type.shape
    live = world.state.mesh.cell_type != SOLID
    if name == "dust":
        keep = [(0, plans[0].steps - 3, plans[0].steps), (1, 0, 4), (2, 0, 2)]
    elif name == "door":
        keep = [(0, plans[0].steps - 2, plans[0].steps), (1, 0, 3), (2, 0, 2)]
    elif name == "door_end":
        keep = [(2, plans[2].steps - 3, plans[2].steps)]
    else:
        keep = [(0, 0, 8)]
    if name == "held":
        fields = world.state.manager.initial_fields()
        fields = {k: f * 1e-3 for k, f in fields.items()}
    elif name == "clean":
        initial = world.state.manager.initial_fields()
        fields = {k: f * rng.uniform(0.003, 0.03, shape) for k, f in initial.items()}
    else:
        fields = {
            k: np.where(live, thresholds[k] * rng.uniform(0.5, 1.5, shape), 0.0)
            for k in range(thresholds.size)
        }
    cut, start = _sliced(plans, keep)
    config = probe
    if name == "held":
        # Within spec from t = 0, so the hold ends the march at the step whose
        # time is the hold exactly.
        config = _scenario("clean", hold=cut[0].time(5))
    return Case(name, config, cut, start, fields)


class _World:
    def __init__(self, config: SimConfig, tmp: Path) -> None:
        self.state = _world(Case("probe", config, [], (0, 0), {}), tmp, _Run, "probe")


MARCH_CASES = ["dust", "door", "clean", "held", "door_end"]


@dataclasses.dataclass
class Taken:
    """What one march showed after each step."""

    steps: list[tuple[int, float]] = dataclasses.field(default_factory=list)
    fields: list[bytes] = dataclasses.field(default_factory=list)
    budgets: list[list[tuple[object, ...]]] = dataclasses.field(default_factory=list)
    maps: list[bytes] = dataclasses.field(default_factory=list)
    recovery: list[dict[str, bytes] | None] = dataclasses.field(default_factory=list)
    readings: list[bytes] = dataclasses.field(default_factory=list)
    channels: list[bytes] = dataclasses.field(default_factory=list)
    means: list[bytes | None] = dataclasses.field(default_factory=list)
    end: tuple[object, ...] = ()
    final_recovery: dict[str, np.ndarray] | None = None


def _budget_bits(b: MassBudget) -> tuple[object, ...]:
    terms = core_mapping.terms_of(b)
    return tuple(None if v is None else _bits(v) for v in terms.values())


def _start(
    state: _Run, case: Case
) -> tuple[dict[int, np.ndarray], list[MassBudget], int, float]:
    """The trackers updated at the start time with the case's fields, as run() does at t = 0."""
    plan = case.plans[0]
    t = plan.time(case.start[1]) if case.start[1] else 0.0
    fields = {k: np.array(f) for k, f in case.fields.items()}
    for f in fields.values():
        f.flags.writeable = False
    pending: list = []
    state.observe(case.start[1], t, fields, pending)
    state.arrival.update(t, fields)
    if state.tracker is not None:
        state.tracker.update(t, fields, state.means(fields))
    return fields, [MassBudget() for _ in fields], case.start[1], t


def _reference_march(case: Case, tmp: Path) -> Taken:
    state = _world(case, tmp, _Run, "reference")
    taken = Taken()

    def on_step(s: MarchStep) -> None:
        taken.steps.append((s.step, s.t))
        taken.fields.append(_bits(core_mapping.stacked(s.fields)))
        taken.budgets.append([_budget_bits(b) for b in s.budgets])
        taken.maps.append(_bits(state.arrival.first_above))
        taken.recovery.append(
            None
            if state.tracker is None
            else {k: _bits(v) for k, v in state.tracker.state().items()}
        )
        assert state.monitor is not None
        taken.readings.append(_bits(state.monitor.readings))
        taken.channels.append(_bits(state.monitor.channel_readings))
        taken.means.append(
            None
            if state.tracker is None
            else _bits(np.array([b.current / state.free_area for b in s.budgets]))
        )

    state.on_step = on_step
    fields, budgets, step, t = _start(state, case)
    out = state.march(case.plans, case.start, fields, budgets, step, t)
    taken.end = (
        _bits(core_mapping.stacked(out[0])),
        [_budget_bits(b) for b in out[1]],
        out[2],
        out[3],
        out[4],
    )
    if state.tracker is not None:
        taken.final_recovery = state.tracker.state()
    return taken


class _Recorder:
    """A March2 whose march results are kept, everything else passed through."""

    def __init__(self, core: "_cfdcore.March2") -> None:
        self._core = core
        self.results: list[dict] = []

    def march(self, n: int) -> dict:
        result = self._core.march(n)
        self.results.append(result)
        return result

    def __getattr__(self, name: str) -> object:
        return getattr(self._core, name)


def _core_march(
    case: Case, tmp: Path, every_step: bool = True
) -> tuple[Taken, core_driver.CoreRun]:
    state = _world(case, tmp, core_driver.CoreRun, "core" if every_step else "batched")
    assert isinstance(state, core_driver.CoreRun)
    recorder = _Recorder(state.core)
    state.core = recorder  # type: ignore[assignment]
    taken = Taken()

    def on_step(s: MarchStep) -> None:
        result = recorder.results[-1]
        taken.steps.append((s.step, s.t))
        taken.fields.append(_bits(core_mapping.stacked(s.fields)))
        taken.budgets.append([_budget_bits(b) for b in s.budgets])
        taken.maps.append(_bits(recorder.arrival()["first_above"]))
        recovery = recorder.recovery()
        taken.recovery.append(
            None if recovery is None else {k: _bits(v) for k, v in recovery.items()}
        )
        taken.readings.append(_bits(result["readings"][-1]))
        taken.channels.append(_bits(result["channels"][-1]))
        means = result["means"]
        taken.means.append(None if means is None else _bits(means[-1]))

    if every_step:
        state.on_step = on_step
    fields, budgets, step, t = _start(state, case)
    out = state.march(case.plans, case.start, fields, budgets, step, t)
    taken.end = (
        _bits(core_mapping.stacked(out[0])),
        [_budget_bits(b) for b in out[1]],
        out[2],
        out[3],
        out[4],
    )
    return taken, state


@pytest.mark.parametrize("name", MARCH_CASES)
def test_march_equals_run_march_after_every_step(name: str, tmp_path: Path) -> None:
    """The core's march equals _Run.march bit for bit after every step.

    The fields, every budget term, the arrival maps, the recovery fields, the
    readings, the channels and the means; the step count, the time and whether
    the hold stopped it at the end.
    """
    case = _case(name, tmp_path)
    reference = _reference_march(case, tmp_path)
    core, state = _core_march(case, tmp_path)
    assert core.steps == reference.steps
    for field in (
        "fields",
        "budgets",
        "maps",
        "recovery",
        "readings",
        "channels",
        "means",
    ):
        assert getattr(core, field) == getattr(reference, field), field
    assert core.end == reference.end
    assert state.free_area == state.core.free_area
    # The planted marches reach what they are planted for.
    steps = sum(p.steps for p in case.plans) - case.start[1]
    if name == "held":
        assert reference.end[4] is True and len(reference.steps) == 5
    else:
        assert reference.end[4] is False and len(reference.steps) == steps
    if name == "door_end":
        # The plan's last step is at t_end, which t_start + steps dt is not.
        plan = case.plans[0]
        assert plan.t_start + plan.steps * plan.dt != plan.t_end
        assert reference.steps[-1][1] == plan.t_end
    if name in ("dust", "door"):
        assert len(case.plans) == 3
        assert len({p.fingerprint for p in case.plans}) == (2 if name == "door" else 1)
    if name in ("clean", "held"):
        final = reference.final_recovery
        assert final is not None
        live = state.mesh.cell_type != SOLID
        local = final["last_above"][:, live]
        if name == "clean":
            # Cells at and below their thresholds both: the local field is set
            # in some and not in others.
            assert np.isfinite(local).any() and np.isnan(local).any()
        else:
            assert np.isnan(local).all() and final["room_since"] == 0.0


@pytest.mark.parametrize("name", MARCH_CASES)
def test_march_in_batches_equals_run_march(name: str, tmp_path: Path) -> None:
    """The core's march in batches, the monitor on its readings, ends as _Run.march ends.

    No on_step and no field read per step: the driver asks the core for every
    step to the next frame, checkpoint or interval end at once.
    """
    case = _case(name, tmp_path)
    reference = _reference_march(case, tmp_path)
    driver = type("Batched", (core_driver.CoreRun,), {"monitor_reads_fields": False})
    core, state = _core_march_with(driver, case, tmp_path)
    assert core.end == reference.end
    assert state.core_calls < len(reference.steps)


def _core_march_with(
    kind: type, case: Case, tmp: Path
) -> tuple[Taken, core_driver.CoreRun]:
    state = _world(case, tmp, kind, "batched")
    taken = Taken()
    fields, budgets, step, t = _start(state, case)
    out = state.march(case.plans, case.start, fields, budgets, step, t)
    taken.end = (
        _bits(core_mapping.stacked(out[0])),
        [_budget_bits(b) for b in out[1]],
        out[2],
        out[3],
        out[4],
    )
    return taken, state


def _snapshot(core: "_cfdcore.March2") -> dict[str, object]:
    return {
        "fields": core.class_fields(),
        "budgets": core.budgets(),
        "step": core.step,
        "t": core.t,
        "arrival": core.arrival(),
        "recovery": core.recovery(),
    }


def _same(a: dict[str, object], b: dict[str, object]) -> bool:
    if _bits(a["fields"]) != _bits(b["fields"]) or a["budgets"] != b["budgets"]:
        return False
    if (a["step"], _bits(a["t"])) != (b["step"], _bits(b["t"])):
        return False
    if _bits(a["arrival"]["first_above"]) != _bits(b["arrival"]["first_above"]):
        return False
    if (a["recovery"] is None) != (b["recovery"] is None):
        return False
    return a["recovery"] is None or all(
        _bits(a["recovery"][k]) == _bits(b["recovery"][k]) for k in a["recovery"]
    )


@pytest.mark.parametrize("name", ["clean", "dust"])
def test_march_restored_mid_interval_continues_bit_for_bit(
    name: str, tmp_path: Path
) -> None:
    """A second core run restored from the first's state mid-interval continues it bit for bit.

    The class fields, budgets, maps and (for the clean-out) the recovery fields
    leave through the binding as plain data and return through restore.
    """
    case = _case(name, tmp_path)
    first = _world(case, tmp_path, core_driver.CoreRun, "first")
    second = _world(case, tmp_path, core_driver.CoreRun, "second")
    assert isinstance(first, core_driver.CoreRun) and isinstance(
        second, core_driver.CoreRun
    )
    fields, budgets, step, t = _start(first, case)
    core_mapping.restore_march(
        first.core, fields, budgets, step, t, first.arrival, first.tracker
    )
    p = len(case.plans) - 1
    plan = case.plans[p]
    interval = first.intervals[plan.index]

    def hand(state: core_driver.CoreRun, at: int) -> None:
        rates = state.manager.sources_at(interval.t_start)
        core_mapping.set_interval(
            state.core,
            state.manager.conditions_for(interval),
            state.faces(plan.fingerprint),
            state.states[plan.fingerprint].nu_t,
            [rates[k] if rates[k].any() else None for k in range(state.n_classes)],
            plan,
            at,
        )

    hand(first, 0)
    middle = plan.steps // 2
    assert len(first.core.march(middle)["step"]) == middle
    kept = _snapshot(first.core)
    first.core.march(plan.steps)
    second.core.restore(
        kept["fields"],
        kept["budgets"],
        kept["step"],
        kept["t"],
        kept["arrival"]["first_above"],
        kept["arrival"]["arrival_t"],
        kept["recovery"],
    )
    hand(second, middle)
    second.core.march(plan.steps)
    assert _same(_snapshot(first.core), _snapshot(second.core))
    if name == "clean":
        assert kept["recovery"] is not None
        assert not np.isnan(kept["recovery"]["last_above"]).all()


def test_march_takes_the_steps_asked_for_and_stops_at_the_hold(tmp_path: Path) -> None:
    """march(n) takes n steps; the held clean-out stops at the reference's step and then takes none."""
    case = _case("held", tmp_path)
    reference = _reference_march(case, tmp_path)
    state = _world(case, tmp_path, core_driver.CoreRun, "direct")
    assert isinstance(state, core_driver.CoreRun)
    fields, budgets, step, t = _start(state, case)
    core_mapping.restore_march(
        state.core, fields, budgets, step, t, state.arrival, state.tracker
    )
    plan = case.plans[0]
    interval = state.intervals[plan.index]
    core_mapping.set_interval(
        state.core,
        state.manager.conditions_for(interval),
        state.faces(plan.fingerprint),
        state.states[plan.fingerprint].nu_t,
        [None] * state.n_classes,
        plan,
        0,
    )
    first = state.core.march(3)
    assert first["step"].tolist() == [1, 2, 3] and first["stopped"] == "count"
    assert state.core.step == 3 and state.core.remaining == plan.steps - 3
    rest = state.core.march(plan.steps)
    assert rest["stopped"] == "held"
    held_step = reference.steps[-1][0]
    assert int(rest["step"][-1]) == held_step == 5
    assert state.core.held()
    after = state.core.march(1)
    assert len(after["step"]) == 0 and after["stopped"] == "held"


def test_march_refuses_what_it_cannot_read(tmp_path: Path) -> None:
    """The binding's own refusals: shapes, a missing recovery state, a plan, a step count."""
    case = _case("clean", tmp_path)
    state = _world(case, tmp_path, core_driver.CoreRun, "refusals")
    assert isinstance(state, core_driver.CoreRun)
    core = state.core
    fields, budgets, _step, _t = _start(state, case)
    cube = core_mapping.stacked(fields)
    terms = [core_mapping.terms_of(b) for b in budgets]
    maps = state.arrival.state()["first_above"]
    recovery = {k: np.ascontiguousarray(v) for k, v in state.tracker.state().items()}
    with pytest.raises(TypeError):
        core.restore(cube[:, :, 1:].copy(), terms, 0, 0.0, maps, np.nan, recovery)
    with pytest.raises(TypeError):
        core.restore(cube, terms, 0, 0.0, maps[:, :, 1:].copy(), np.nan, recovery)
    with pytest.raises(ValueError, match="needs its state"):
        core.restore(cube, terms, 0, 0.0, maps, np.nan, None)
    with pytest.raises(TypeError, match="last_above"):
        core.restore(
            cube,
            terms,
            0,
            0.0,
            maps,
            np.nan,
            {**recovery, "last_above": recovery["last_above"][1:]},
        )
    with pytest.raises(ValueError):
        core.march(1)  # no interval
    core.restore(cube, terms, 0, 0.0, maps, np.nan, recovery)
    plan = case.plans[0]
    interval = state.intervals[plan.index]
    with pytest.raises(ValueError, match="first step"):
        core_mapping.set_interval(
            core,
            state.manager.conditions_for(interval),
            state.faces(plan.fingerprint),
            None,
            [None] * state.n_classes,
            plan,
            plan.steps,
        )
    with pytest.raises(TypeError, match="dt must be a number, not a bool"):
        core.interval(
            [
                state.manager.conditions_for(interval).faces_for(k)
                for k in range(state.n_classes)
            ],
            state.faces(plan.fingerprint).u,
            state.faces(plan.fingerprint).v,
            dt=True,
            steps=plan.steps,
            t_start=plan.t_start,
            t_end=plan.t_end,
        )
    with pytest.raises(ValueError, match="outside the grid"):
        _cfdcore.Recovery2(
            _grid(state.mesh),
            [1.0] * 5,
            {"target": [1.0] * 5, "defined": [True] * 5, "hold": None},
            [(30, 0)],
            [[k] for k in range(5)],
        )
    core_mapping.set_interval(
        core,
        state.manager.conditions_for(interval),
        state.faces(plan.fingerprint),
        state.states[plan.fingerprint].nu_t,
        [None] * state.n_classes,
        plan,
        0,
    )
    with pytest.raises(ValueError, match="at least 1"):
        core.march(0)


def test_march_binding_imports_nothing_from_src(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A restore, an interval's march and every read go through with every src module unimportable."""
    case = _case("clean", tmp_path)
    state = _world(case, tmp_path, core_driver.CoreRun, "plain")
    assert isinstance(state, core_driver.CoreRun)
    fields, budgets, _step, _t = _start(state, case)
    core = state.core
    cube = core_mapping.stacked(fields)
    terms = [core_mapping.terms_of(b) for b in budgets]
    maps = np.ascontiguousarray(state.arrival.state()["first_above"])
    recovery = {k: np.ascontiguousarray(v) for k, v in state.tracker.state().items()}
    plan = case.plans[0]
    interval = state.intervals[plan.index]
    conditions = [
        state.manager.conditions_for(interval).faces_for(k)
        for k in range(state.n_classes)
    ]
    faces = state.faces(plan.fingerprint)
    nu_t = state.states[plan.fingerprint].nu_t
    for name in [n for n in sys.modules if n == "src" or n.startswith("src.")]:
        monkeypatch.setitem(sys.modules, name, None)
    core.restore(cube, terms, 0, 0.0, maps, np.nan, recovery)
    core.interval(
        conditions,
        faces.u,
        faces.v,
        None,
        nu_t,
        None,
        dt=plan.dt,
        steps=plan.steps,
        t_start=plan.t_start,
        t_end=plan.t_end,
    )
    result = core.march(2)
    assert result["stopped"] == "count" and result["breach"] is None
    assert len(core.budgets()) == core.classes
    assert core.class_fields().shape[0] == core.classes
    assert core.arrival()["first_above"].shape == maps.shape
    assert set(core.recovery()) == set(recovery)
