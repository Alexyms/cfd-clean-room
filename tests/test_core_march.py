"""The compiled core's march against the NumPy reference (REQ-N03 level 1, ADR-017 D).

The step loop's blocks (``channels``, ``arrival``, ``recovery``) against the
reference's functions on planted fields, and the march itself (``_cfdcore.March2``
driven by ``tests.core_driver.CoreRun``) against ``time_integration._Run.march``
on the 80x30 product mesh with planted flows, every output compared bit for bit
after every step: the steps, a restore from the middle of a march (a clean-out
before and inside its within-spec stretch among them), and the stops a march
ends in, a hold, a breach of ADR-016's limit and a refused step, with the error,
the steps before it and the run file. Then the binding's own refusals, one by
one. Skipped with its reason when ``_cfdcore`` cannot be imported, an error
instead with ``CFD_CORE_REQUIRED=1``, as ``tests/test_core_parity.py``.
"""

import dataclasses
import functools
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from src.config import SimConfig
from src.io_manager import RunFile
from src.mesh import SOLID, Mesh
from src.monitor import AlertMonitor, ArrivalTracker, channel_sums
from src.solver_staggered import FlowState
from src.solver_transport import ConservationBreachError, MassBudget
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


def _initial_levels(thresholds: np.ndarray) -> np.ndarray:
    """Initial levels at which class 3's 100:1 target is not defined.

    A hundred times each threshold exactly, except class 3's, which is its
    threshold: a hundredth of a hundred times it, so its target is not defined.
    """
    initial = 100.0 * thresholds
    initial[3] = thresholds[3]
    return initial


def _recovery_pair(
    room: str, hold: float | None
) -> tuple[RecoveryTracker, "_cfdcore.Recovery2", SimConfig, Mesh, np.ndarray]:
    config = _room(room, 80, 30)
    mesh = _mesh(room, 80, 30)
    thresholds = np.array(config.class_thresholds("the test"), dtype=np.float64)
    channels = [list(m) for m in config.channel_classes()]
    cells = list(core_mapping_cells(config, mesh))
    initial = _initial_levels(thresholds)
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
    # The tracker's own targets: class 3's is a hundredth of its threshold, which
    # its readings and means sit above unless a plant puts them there.
    target = core_mapping.recovery_keys(thresholds, _initial_levels(thresholds), None)[
        "target"
    ]
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
        if u == 4:
            # Under class 3's own target, which is not defined: no time is set.
            means[3] = 0.5 * float(target[3])
        reference.update(t, {k: fields[k] for k in range(n)}, means)
        core.update(t, np.ascontiguousarray(fields), means)
        state = core.state()
        for name, array in reference.state().items():
            assert _bits(state[name]) == _bits(array), (t, name)
        assert core.held() == reference.held(), t
        held.append(reference.held())
    # The planted inputs reach the rules: the hold holds at equality and ends
    # when a sensor reads its threshold; a reading at its target sets its time;
    # the class without a defined target never does, whether a reading or its mean
    # is under it.
    assert held[held_at] and not held[held_at - 1] and not held[held_at + 2]
    record = reference.state()
    assert record["sensor_time"][0, 0] == times[1]
    assert np.isnan(record["sensor_time"][3]).all() and np.isnan(record["mean_time"][3])
    assert record["mean_time"][0] <= times[3]
    again = _cfdcore.Recovery2(
        _grid(mesh),
        list(thresholds),
        core_mapping.recovery_keys(thresholds, _initial_levels(thresholds), hold),
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


# The decay case's levels at the sensors' cells, as fractions of the classes'
# thresholds (zero elsewhere): the first three classes start above theirs, the other
# two below, so the classes are within spec at different steps.
DECAY_LEVELS = (1.4, 1.2, 1.1, 0.0, 0.0)
# The decay case's step at which the room is within spec, and at which it holds.
DECAY_WITHIN, DECAY_HELD = 11, 15


def _sensor_blobs(
    config: SimConfig, mesh: Mesh, thresholds: np.ndarray
) -> dict[int, np.ndarray]:
    """Fields that are zero but at the sensors' cells, which hold DECAY_LEVELS of the thresholds."""
    fields = {k: np.zeros(mesh.cell_type.shape) for k in range(thresholds.size)}
    for j, i in core_mapping_cells(config, mesh):
        for k, level in enumerate(DECAY_LEVELS):
            fields[k][j, i] = level * thresholds[k]
    return fields


def _case(name: str, tmp: Path) -> Case:
    """One of the planted marches.

    Across the dust release's onset; across the door leak's two flow switches;
    the clean-out from fields around its thresholds; the clean-out within spec
    from the start, held at its fifth step; the door leak's last interval to its
    end, whose last step's time is ``t_end`` and not ``t_start + steps dt``; and
    the clean-out from the third step of its first interval with the sensors
    above their thresholds, which they fall below at different steps (the
    channels one after another, the room's last), held four steps later.
    """
    rng = np.random.default_rng([SEED, 4, len(name)])
    probe = _scenario(
        {"held": "clean", "door_end": "door", "decay": "clean"}.get(name, name)
    )
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
    elif name == "decay":
        keep = [(0, 3, 20)]
    else:
        keep = [(0, 0, 8)]
    if name == "held":
        fields = world.state.manager.initial_fields()
        fields = {k: f * 1e-3 for k, f in fields.items()}
    elif name == "decay":
        fields = _sensor_blobs(probe, world.state.mesh, thresholds)
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
    if name == "decay":
        # The stretch from the step the room is within spec to the step it holds.
        hold = cut[0].time(DECAY_HELD) - cut[0].time(DECAY_WITHIN)
        config = _scenario("clean", hold=hold)
    return Case(name, config, cut, start, fields)


class _World:
    def __init__(self, config: SimConfig, tmp: Path) -> None:
        self.state = _world(Case("probe", config, [], (0, 0), {}), tmp, _Run, "probe")


MARCH_CASES = ["dust", "door", "clean", "held", "door_end", "decay"]


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
    elif name == "decay":
        assert reference.end[4] is True
        assert len(reference.steps) == DECAY_HELD - case.start[1]
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
    if name in ("clean", "held", "decay"):
        final = reference.final_recovery
        assert final is not None
        live = state.mesh.cell_type != SOLID
        local = final["last_above"][:, live]
        if name == "held":
            assert np.isnan(local).all() and final["room_since"] == 0.0
        else:
            # Cells at and below their thresholds both: the local field is set
            # in some and not in others.
            assert np.isfinite(local).any() and np.isnan(local).any()
    if name == "decay":
        # The classes are within spec at different steps, the room at the last of
        # them, the means from the start: every array of the state is its own.
        plan = case.plans[0]
        assert final["room_since"] == plan.time(DECAY_WITHIN)
        assert final["class_since"][0] == plan.time(DECAY_WITHIN)
        assert final["class_since"][3] == plan.time(case.start[1])
        assert 0.0 < final["class_since"][2] < final["room_since"]
        assert (final["mean_time"] == plan.time(case.start[1])).all()
        assert np.isfinite(final["sensor_time"]).all()
        assert not np.array_equal(final["class_since"], final["mean_time"])


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


def _hand(state: core_driver.CoreRun, plan: IntervalPlan, at: int) -> None:
    """Hand a core run the plan's interval, to march from step AT within it."""
    interval = state.intervals[plan.index]
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
    plan = case.plans[-1]
    _hand(first, plan, 0)
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
    _hand(second, plan, middle)
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
    with pytest.raises(ValueError, match=r"March2\.interval: the first step is not in"):
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


@pytest.mark.parametrize(
    "at", [8, 13], ids=["before the stretch", "inside the stretch"]
)
def test_march_restored_recovery_continues_bit_for_bit(at: int, tmp_path: Path) -> None:
    """A clean-out restored before and inside its within-spec stretch continues bit for bit.

    The decay case: the room is within spec from step 11 (the channels one after
    another from step 5) and holds at step 15. A second core run restored from the
    first's state at step 8 or 13 reaches the same hold with every array of the
    recovery state the first reaches: the room's start, the classes' starts, the
    means' times and the sensors' 100:1 times, each different from the others.
    """
    case = _case("decay", tmp_path)
    first = _world(case, tmp_path, core_driver.CoreRun, "first")
    second = _world(case, tmp_path, core_driver.CoreRun, "second")
    assert isinstance(first, core_driver.CoreRun)
    assert isinstance(second, core_driver.CoreRun)
    fields, budgets, step, t = _start(first, case)
    core_mapping.restore_march(
        first.core, fields, budgets, step, t, first.arrival, first.tracker
    )
    plan = case.plans[0]
    _hand(first, plan, case.start[1])
    assert len(first.core.march(at - case.start[1])["step"]) == at - case.start[1]
    kept = _snapshot(first.core)
    state = kept["recovery"]
    assert state is not None
    # The state handed over holds what the restore must not lose or mix up.
    assert (
        np.isfinite(state["mean_time"]).all()
        and np.isfinite(state["sensor_time"]).any()
    )
    if at < DECAY_WITHIN:
        assert np.isnan(state["room_since"])
        assert np.isnan(state["class_since"]).any()
        assert np.isfinite(state["class_since"]).any()
        assert not np.isfinite(state["sensor_time"]).all()
    else:
        assert state["room_since"] == plan.time(DECAY_WITHIN) > 0.0
        assert np.isfinite(state["class_since"]).all()
        assert np.isfinite(state["sensor_time"]).all()
    assert not np.array_equal(state["class_since"], state["mean_time"])
    rest = first.core.march(plan.steps)
    assert rest["stopped"] == "held" and int(rest["step"][-1]) == DECAY_HELD
    second.core.restore(
        kept["fields"],
        kept["budgets"],
        kept["step"],
        kept["t"],
        kept["arrival"]["first_above"],
        kept["arrival"]["arrival_t"],
        kept["recovery"],
    )
    _hand(second, plan, at)
    again = second.core.march(plan.steps)
    assert again["stopped"] == "held"
    assert again["step"].tolist() == rest["step"].tolist()
    assert _same(_snapshot(first.core), _snapshot(second.core))


@pytest.mark.parametrize("room", list(ROOMS))
def test_recovery_within_spec_is_every_sensor_below_its_channels_threshold(
    room: str,
) -> None:
    """Within spec is every sensor below every channel's threshold, on the channel's reading.

    Planted on the sensors' cells, after each of which the fields go back
    below: each channel in turn tipped at each sensor in turn by its class
    alone, at its threshold exactly; then each channel tipped by a class below
    its own threshold whose channel (it and the next class) is not. The
    core's state equals the tracker's after every update, and the planted
    updates are shown to separate the rule from the three wrong ones: the first
    sensor alone, each class on its own reading, and the room within when any
    channel is.
    """
    reference, core, config, mesh, thresholds = _recovery_pair(room, 0.5)
    n = thresholds.size
    channels = [list(m) for m in config.channel_classes()]
    cells = list(core_mapping_cells(config, mesh))
    sensors = len(cells)
    zero = np.zeros((n, *mesh.cell_type.shape))
    script: list[dict[tuple[int, int], float]] = [{}]
    for k in range(n):
        for m in range(sensors):
            script += [{(k, m): thresholds[k]}, {}, {}, {}]
    for k in range(n - 1):
        m = 1 + k % (sensors - 1)
        script += [{(k, m): 0.5 * thresholds[k], (k + 1, m): thresholds[k]}, {}, {}, {}]
    separated = {"first sensor": False, "class": False, "room": False}
    held = []
    for u, planted in enumerate(script):
        t = 0.25 * (u + 1)
        fields = zero.copy()
        for (k, m), value in planted.items():
            j, i = cells[m]
            fields[k, j, i] = value
        means = [1.0] * n
        reference.update(t, {k: fields[k] for k in range(n)}, means)
        core.update(t, np.ascontiguousarray(fields), means)
        state = core.state()
        for name, array in reference.state().items():
            assert _bits(state[name]) == _bits(array), (t, name)
        assert core.held() == reference.held(), t
        held.append(reference.held())
        # The rule, from the readings: every sensor on the channel's reading.
        readings = np.array([[fields[k, j, i] for j, i in cells] for k in range(n)])
        sums = _cfdcore.sensor_channels(readings, channels)
        within = (sums < thresholds[:, None]).all(axis=1)
        assert (np.isfinite(reference.state()["class_since"]) == within).all(), t
        assert np.isfinite(reference.state()["room_since"]) == within.all(), t
        separated["first sensor"] |= bool((within != (sums[:, 0] < thresholds)).any())
        separated["class"] |= bool(
            (within != (readings < thresholds[:, None]).all(axis=1)).any()
        )
        separated["room"] |= bool(0 < within.sum() < n)
    assert all(separated.values()), separated
    assert any(held) and not all(held)


# ---------------------------------------------------------------------------
# the breach and the refusals of a step, through the march

# Each case: scenario, ADR-016 limit, the class that breaches, the steps before it.
# The limits sit between consecutive leftover ratios with a margin of at least 13 %
# either side (explore the ratios with a limit of 0.5); the dust and door cases
# breach in the first or second step of their second interval.
BREACHES = {
    "class 0 before any step": ("clean", 1e-300, 0, 0),
    "class 0 after one step": ("clean", 9e-8, 0, 1),
    "class 1 before any step": ("dust", 1.2e-8, 1, 0),
    "class 1 at the interval switch": ("dust", 8.8e-8, 1, 3),
    "class 4 at the interval switch": ("dust", 3.2e-7, 4, 3),
    "class 0 inside the second interval": ("door", 1e-7, 0, 3),
}


def _capped(name: str, limit: float) -> SimConfig:
    """A scenario whose solve stops at two sweeps with no share of the limit to hold.

    Each step then leaves the leftover its cap leaves whatever the limit is, so a
    limit places a breach (ADR-016). Frames every second step and a checkpoint
    every third leave something in the run file to compare.
    """
    raw = yaml.safe_load((CONFIGS / SCENARIOS[name]).read_text(encoding="utf-8"))
    raw["domain"] = {"nx": 80, "ny": 30}
    raw["transport"] = {
        **raw.get("transport", {}),
        "conservation_limit": limit,
        "leftover_share": 1e-300,
        "max_diffusion_iter": 2,
    }
    raw["solver"] = {
        **raw.get("solver", {}),
        "output_interval": 2,
        "checkpoint_interval": 3,
    }
    return SimConfig.from_dict(raw, CONFIGS)


FileState = dict[str, tuple[str, tuple[int, ...], bytes]]


def _file_state(run: RunFile) -> FileState:
    """Every dataset of a run file and its checkpoint file but the configuration's.

    By name, with its dtype, shape and bytes, so that two runs' files compare by
    what they hold and not by what they were written with.
    """
    out: FileState = {}
    for label, path in (("run", run.path), ("checkpoint", run.checkpoint_path)):
        if not path.exists():
            continue
        with h5py.File(path, "r") as f:

            def visit(name: str, item: object, label: str = label) -> None:
                if isinstance(item, h5py.Dataset) and not name.startswith("config"):
                    out[f"{label}/{name}"] = (
                        str(item.dtype),
                        item.shape,
                        np.asarray(item[()]).tobytes(),
                    )

            f.visititems(visit)
    return out


@dataclasses.dataclass
class Outcome:
    """What a march did to its end or its error: each step, the error, the run file."""

    seen: list[tuple[int, str, bytes, tuple[object, ...]]]
    error: tuple[object, ...] | None
    file: FileState
    calls: int | None


def _outcome(
    kind: type[_Run], case: Case, tmp: Path, tag: str, every_step: bool
) -> Outcome:
    """One march of a case on a kind of run, the error it raises kept as plain values."""
    state = _world(case, tmp, kind, tag)
    seen: list[tuple[int, str, bytes, tuple[object, ...]]] = []

    def record(s: MarchStep) -> None:
        seen.append(
            (
                s.step,
                float(s.t).hex(),
                _bits(core_mapping.stacked(s.fields)),
                tuple(_budget_bits(b) for b in s.budgets),
            )
        )

    if every_step:
        state.on_step = record
    fields, budgets, step, t = _start(state, case)
    error: tuple[object, ...] | None = None
    try:
        state.march(case.plans, case.start, fields, budgets, step, t)
    except ConservationBreachError as breach:
        error = (
            "ConservationBreachError",
            breach.size_class,
            float(breach.leftover).hex(),
            float(breach.supplied).hex(),
            float(breach.limit).hex(),
            str(breach),
        )
    except ValueError as refused:
        error = ("ValueError", str(refused))
    return Outcome(
        seen, error, _file_state(state.run), getattr(state, "core_calls", None)
    )


def _outcomes(case: Case, tmp: Path) -> tuple[Outcome, Outcome, Outcome]:
    """The reference's march, the core's every step and the core's in batches, equal.

    Asserts what every route shares: the error, the steps before it, the run
    file. Returns the three.
    """
    reference = _outcome(_Run, case, tmp, "reference", True)
    core = _outcome(core_driver.CoreRun, case, tmp, "core", True)
    batched_kind = type(
        "Batched", (core_driver.CoreRun,), {"monitor_reads_fields": False}
    )
    batched = _outcome(batched_kind, case, tmp, "batched", False)
    assert core.error == reference.error
    assert core.seen == reference.seen
    assert core.file == reference.file
    assert batched.error == reference.error
    assert batched.file == reference.file
    return reference, core, batched


@pytest.mark.parametrize("label", list(BREACHES))
def test_march_breach_stops_as_run_march_stops(label: str, tmp_path: Path) -> None:
    """A step that breaches ADR-016's limit stops the core's march where it stops the reference's.

    The error carries the reference's class, leftover, supplied and limit in every
    bit and its message; the steps before it are the reference's, as is the run
    file they leave (the frames and checkpoint written before it, the readings not
    yet flushed dropped); in batches the same error and file. Each case places the
    breach: in class 0, 1 or 4, before any step, after one, and after the interval
    switch.
    """
    name, limit, size_class, before = BREACHES[label]
    case = dataclasses.replace(_case(name, tmp_path), config=_capped(name, limit))
    reference, _core, _batched = _outcomes(case, tmp_path)
    error = reference.error
    assert error is not None and error[:2] == ("ConservationBreachError", size_class)
    assert error[4] == float(limit).hex()
    assert len(reference.seen) == before
    if before >= 3:
        # A frame and a checkpoint were written before the breach.
        assert any(key.startswith("run/transport/sensors") for key in reference.file)
        assert any(key.startswith("checkpoint/") for key in reference.file)


def test_march_stops_as_run_march_does_on_a_field_that_is_not_finite(
    tmp_path: Path,
) -> None:
    """A NaN in class 2's field is refused by the core's march as the reference refuses it."""
    base = _case("clean", tmp_path)
    bad = {k: np.array(f) for k, f in base.fields.items()}
    live = np.argwhere(Mesh(base.config).cell_type != SOLID)
    bad[2][tuple(live[len(live) // 2])] = np.nan
    case = dataclasses.replace(base, fields=bad)
    reference, _core, _batched = _outcomes(case, tmp_path)
    assert reference.error == ("ValueError", "C_k must be finite")
    assert reference.seen == []


def test_march_stops_as_run_march_does_on_a_step_above_the_stable_step(
    tmp_path: Path,
) -> None:
    """A plan whose step is above the stable step is refused by the core as the reference refuses it."""
    base = _case("clean", tmp_path)
    plans = [dataclasses.replace(p, dt=2.0 * p.stable_dt) for p in base.plans]
    case = dataclasses.replace(base, plans=plans)
    reference, _core, _batched = _outcomes(case, tmp_path)
    assert reference.error is not None and reference.error[0] == "ValueError"
    assert "exceeds stable_dt" in str(reference.error[1])
    assert reference.seen == []


# ---------------------------------------------------------------------------
# the driver


class _Stub:
    """A core whose march is the real one's with its result altered, the rest passed through."""

    def __init__(self, core: "_cfdcore.March2", alter: object) -> None:
        self._core = core
        self._alter = alter
        self.calls = 0

    def march(self, n: int) -> dict:
        self.calls += 1
        if self.calls > 3:
            raise RuntimeError("the driver asked again")
        return self._alter(self._core, n)  # type: ignore[operator]

    def __getattr__(self, name: str) -> object:
        return getattr(self._core, name)


def _empty(core: "_cfdcore.March2", n: int) -> dict:
    """A march that took no step and says it counted."""
    return {
        "step": np.zeros(0, dtype=np.int64),
        "t": np.zeros(0),
        "stopped": "count",
        "breach": None,
    }


def test_driver_refuses_a_march_that_returns_no_step_without_holding(
    tmp_path: Path,
) -> None:
    """A march of no step that neither held nor breached is an error, not a loop."""
    case = _case("clean", tmp_path)
    state = _world(case, tmp_path, core_driver.CoreRun, "empty")
    assert isinstance(state, core_driver.CoreRun)
    state.core = _Stub(state.core, _empty)  # type: ignore[assignment]
    fields, budgets, step, t = _start(state, case)
    with pytest.raises(AssertionError, match="no step"):
        state.march(case.plans, case.start, fields, budgets, step, t)
    assert state.core.calls == 1


@pytest.mark.parametrize(
    ("claim", "match"),
    [
        ("interval", "stopped as 'interval' with"),
        ("held", "stopped as 'held' with"),
        ("breach", "stopped as 'breach' with"),
        ("something else", "stopped as 'something else' with"),
    ],
)
def test_driver_refuses_a_stop_reason_the_cores_state_does_not_bear_out(
    claim: str, match: str, tmp_path: Path
) -> None:
    """The driver reads the core's stop reason and refuses one the core's own state contradicts.

    After the first step of a march the interval has steps left, the run does
    not hold and there is no breach record; the core's reason is a count.
    """
    case = _case("clean", tmp_path)
    state = _world(case, tmp_path, core_driver.CoreRun, "claims")
    assert isinstance(state, core_driver.CoreRun)

    def lie(core: "_cfdcore.March2", n: int) -> dict:
        result = core.march(n)
        return {**result, "stopped": claim}

    state.core = _Stub(state.core, lie)  # type: ignore[assignment]
    fields, budgets, step, t = _start(state, case)
    with pytest.raises(AssertionError, match=match):
        state.march(case.plans, case.start, fields, budgets, step, t)


def test_driver_refuses_a_count_at_the_intervals_end(tmp_path: Path) -> None:
    """A march that reaches the interval's last step and reports a count is refused."""
    case = _case("clean", tmp_path)
    state = _world(case, tmp_path, core_driver.CoreRun, "count")
    assert isinstance(state, core_driver.CoreRun)

    def count(core: "_cfdcore.March2", n: int) -> dict:
        result = core.march(core.remaining)
        assert result["stopped"] == "interval"
        return {**result, "stopped": "count"}

    state.core = _Stub(state.core, count)  # type: ignore[assignment]
    fields, budgets, step, t = _start(state, case)
    with pytest.raises(AssertionError, match="stopped as 'count' with 0"):
        state.march(case.plans, case.start, fields, budgets, step, t)


def test_driver_refuses_an_interval_the_core_was_not_built_for(tmp_path: Path) -> None:
    """An interval whose transport keys, particles or t_end are not the core's is refused.

    The core is built from the run's configuration once, where the reference
    builds a solver per interval, so every whole-run claim rests on this check.
    """
    case = _case("clean", tmp_path)
    state = _world(case, tmp_path, core_driver.CoreRun, "built")
    assert isinstance(state, core_driver.CoreRun)
    other = _scenario("dust")
    assert other.t_end != case.config.t_end
    state.config = other
    with pytest.raises(
        ValueError, match="not the run's, which the core was built with"
    ):
        state._check_interval(case.plans[0])
    state.config = case.config
    state._check_interval(case.plans[0])


# ---------------------------------------------------------------------------
# the binding's refusals, one by one


def _unaligned(a: np.ndarray) -> np.ndarray:
    """A copy of an array whose data pointer is one byte past a multiple of eight."""
    raw = np.zeros(a.nbytes + 16, dtype=np.uint8)
    start = (1 - raw.ctypes.data) % 8
    out = raw[start : start + a.nbytes].view(a.dtype).reshape(a.shape)
    out[...] = a
    assert not out.flags.aligned
    return out


@dataclasses.dataclass
class Ready:
    """A clean-out core run with everything its restore and interval take, to change one thing at a time."""

    state: core_driver.CoreRun
    plan: IntervalPlan
    cube: np.ndarray
    terms: list[dict[str, float | None]]
    maps: np.ndarray
    recovery: dict[str, np.ndarray]

    def restore(self, **changes: object) -> None:
        """``March2.restore`` with the run's own arguments, those named changed."""
        args: dict[str, object] = {
            "fields": self.cube,
            "budgets": self.terms,
            "step": 0,
            "t": 0.0,
            "first_above": self.maps,
            "arrival_t": np.nan,
            "recovery": self.recovery,
        }
        args.update(changes)
        self.state.core.restore(**args)

    def interval(self, **changes: object) -> None:
        """``March2.interval`` with the clean-out's first interval, those named changed."""
        state, plan = self.state, self.plan
        interval = state.intervals[plan.index]
        args: dict[str, object] = {
            "conditions": [
                state.manager.conditions_for(interval).faces_for(k)
                for k in range(state.n_classes)
            ],
            "u": state.faces(plan.fingerprint).u,
            "v": state.faces(plan.fingerprint).v,
            "eddy_viscosity": state.states[plan.fingerprint].nu_t,
            "dt": plan.dt,
            "steps": plan.steps,
            "t_start": plan.t_start,
            "t_end": plan.t_end,
            "first": 0,
        }
        args.update(changes)
        state.core.interval(**args)


def _ready(tmp: Path) -> Ready:
    case = _case("clean", tmp)
    state = _world(case, tmp, core_driver.CoreRun, "ready")
    assert isinstance(state, core_driver.CoreRun)
    fields, budgets, _step, _t = _start(state, case)
    assert state.tracker is not None
    return Ready(
        state,
        case.plans[0],
        core_mapping.stacked(fields),
        [core_mapping.terms_of(b) for b in budgets],
        np.ascontiguousarray(state.arrival.state()["first_above"]),
        {k: np.ascontiguousarray(v) for k, v in state.tracker.state().items()},
    )


GUARDS = {
    "restore: step True": (
        "restore",
        {"step": True},
        TypeError,
        r"^step must be an int, not a bool",
    ),
    "restore: step a float": (
        "restore",
        {"step": 2.0},
        TypeError,
        r"^step must be an int$",
    ),
    "restore: step negative": ("restore", {"step": -1}, ValueError, "step is negative"),
    "restore: step beyond 64 bits": (
        "restore",
        {"step": 2**70},
        ValueError,
        r"^step is out of range",
    ),
    "restore: t True": (
        "restore",
        {"t": True},
        TypeError,
        r"^t must be a number, not a bool",
    ),
    "restore: t NaN": (
        "restore",
        {"t": float("nan")},
        ValueError,
        r"^t must be finite",
    ),
    "restore: t inf": (
        "restore",
        {"t": float("inf")},
        ValueError,
        r"^t must be finite",
    ),
    "restore: arrival_t True": (
        "restore",
        {"arrival_t": True},
        TypeError,
        r"^arrival_t must be a number, not a bool",
    ),
    "restore: arrival_t inf": (
        "restore",
        {"arrival_t": float("inf")},
        ValueError,
        r"^arrival_t must be finite",
    ),
    "interval: steps True": (
        "interval",
        {"steps": True},
        TypeError,
        r"^steps must be an int, not a bool",
    ),
    "interval: steps a float": (
        "interval",
        {"steps": 5.0},
        TypeError,
        r"^steps must be an int$",
    ),
    "interval: first True": (
        "interval",
        {"first": True},
        TypeError,
        r"^first must be an int, not a bool",
    ),
    "interval: first a float": (
        "interval",
        {"first": 1.0},
        TypeError,
        r"^first must be an int$",
    ),
    "interval: t_start True": (
        "interval",
        {"t_start": True},
        TypeError,
        r"^t_start must be a number, not a bool",
    ),
    "interval: t_start NaN": (
        "interval",
        {"t_start": float("nan")},
        ValueError,
        r"^t_start must be finite",
    ),
    "interval: t_start inf": (
        "interval",
        {"t_start": float("inf")},
        ValueError,
        r"^t_start must be finite",
    ),
    "interval: t_end True": (
        "interval",
        {"t_end": True},
        TypeError,
        r"^t_end must be a number, not a bool",
    ),
    "interval: t_end NaN": (
        "interval",
        {"t_end": float("nan")},
        ValueError,
        r"^t_end must be finite",
    ),
    "interval: t_end inf": (
        "interval",
        {"t_end": float("inf")},
        ValueError,
        r"^t_end must be finite",
    ),
    "interval: first beyond the steps": (
        "interval",
        {"first": 99999},
        ValueError,
        r"^March2\.interval: the first step is not in",
    ),
}


@pytest.mark.parametrize("label", list(GUARDS))
def test_march_refuses_a_bool_a_float_count_and_a_time_that_is_not_finite(
    label: str, tmp_path: Path
) -> None:
    """The plan, time and count a restore and an interval read are refused when they are not numbers.

    A bool is a TypeError (as ``dt=True`` is), a float where a count is read is
    one, and a time that is NaN or infinite, a negative step count and a first
    step outside the interval are ValueErrors.
    """
    ready = _ready(tmp_path)
    call, changes, exception, match = GUARDS[label]
    ready.restore()
    with pytest.raises(exception, match=match):
        getattr(ready, call)(**changes)


def test_march_refuses_a_bool_or_a_float_for_the_steps_asked(tmp_path: Path) -> None:
    """march(n) reads n as a count: a bool, a float and a number past 64 bits are refused."""
    ready = _ready(tmp_path)
    ready.restore()
    ready.interval()
    core = ready.state.core
    with pytest.raises(TypeError, match=r"^n must be an int, not a bool"):
        core.march(True)
    with pytest.raises(TypeError, match=r"^n must be an int$"):
        core.march(2.0)
    with pytest.raises(ValueError, match=r"^n is out of range"):
        core.march(2**70)
    assert core.step == 0 and core.remaining == ready.plan.steps


def test_march_reads_numpy_numbers_and_a_nan_arrival_time(tmp_path: Path) -> None:
    """The guards refuse no number the driver or a checkpoint hands over.

    NumPy's integers and floats are numbers, and the maps' last update's time is
    NaN before the first update.
    """
    ready = _ready(tmp_path)
    ready.restore(step=np.int64(2), t=np.float64(0.25), arrival_t=np.nan)
    core = ready.state.core
    assert core.step == 2 and core.t == 0.25
    ready.interval(steps=np.int64(ready.plan.steps), first=np.int64(1))
    assert core.remaining == ready.plan.steps - 1
    assert len(core.march(np.int64(2))["step"]) == 2
    ready.restore(t=3, arrival_t=0.5)
    assert core.t == 3.0 and core.arrival()["arrival_t"] == 0.5


@functools.cache
def _clean_room() -> tuple[SimConfig, Mesh]:
    """The 80x30 clean-out's configuration and mesh."""
    config = _scenario("clean")
    return config, Mesh(config)


def _march_args(**changes: object) -> dict[str, object]:
    """March2's arguments for the 80x30 clean-out, some changed."""
    config, mesh = _clean_room()
    thresholds = np.array(config.class_thresholds("the test"), dtype=np.float64)
    keys = core_mapping.recovery_keys(thresholds, _initial_levels(thresholds), None)
    args = core_mapping.march_arguments(
        config, mesh, list(core_mapping_cells(config, mesh)), [0.0], keys
    )
    args.update(changes)
    return args


_SPACINGS = "dx_cell, dy_cell, dx_face and dy_face are"
_ALIGNED = "March2: an array is not aligned"
_ONE_PER_CLASS = "settling and diffusion hold one value per class"
_NO_RECOVERY_SENSOR = {"target": [1.0] * 5, "defined": [True] * 5, "hold": None}
# Each: the arguments changed, as a function of the mesh; the exception; its message.
CONSTRUCTION = {
    "dx_cell one short": (
        lambda m: {"dx_cell": m.dx_cell[:-1].copy()},
        TypeError,
        _SPACINGS,
    ),
    "dy_cell one short": (
        lambda m: {"dy_cell": m.dy_cell[:-1].copy()},
        TypeError,
        _SPACINGS,
    ),
    "dx_face one long": (
        lambda m: {"dx_face": np.append(m.dx_face, 1.0)},
        TypeError,
        _SPACINGS,
    ),
    "dy_face one long": (
        lambda m: {"dy_face": np.append(m.dy_face, 1.0)},
        TypeError,
        _SPACINGS,
    ),
    "dx_cell unaligned": (
        lambda m: {"dx_cell": _unaligned(m.dx_cell)},
        TypeError,
        _ALIGNED,
    ),
    "dy_cell unaligned": (
        lambda m: {"dy_cell": _unaligned(m.dy_cell)},
        TypeError,
        _ALIGNED,
    ),
    "dx_face unaligned": (
        lambda m: {"dx_face": _unaligned(m.dx_face)},
        TypeError,
        _ALIGNED,
    ),
    "dy_face unaligned": (
        lambda m: {"dy_face": _unaligned(m.dy_face)},
        TypeError,
        _ALIGNED,
    ),
    "no class": (
        lambda m: {"settling": [], "diffusion": []},
        ValueError,
        _ONE_PER_CLASS,
    ),
    "one settling velocity short": (
        lambda m: {"settling": [1e-5] * 4},
        ValueError,
        _ONE_PER_CLASS,
    ),
    "one diffusion coefficient short": (
        lambda m: {"diffusion": [1e-5] * 4},
        ValueError,
        _ONE_PER_CLASS,
    ),
    "a sensor below the grid": (
        lambda m: {"sensors": [(30, 0)]},
        ValueError,
        r"March2: a sensor's cell \(30, 0\) is outside the grid",
    ),
    "a sensor beyond the grid": (
        lambda m: {"sensors": [(0, 80)]},
        ValueError,
        r"March2: a sensor's cell \(0, 80\) is outside the grid",
    ),
    "a recovery without a sensor": (
        lambda m: {"sensors": [], "recovery": _NO_RECOVERY_SENSOR},
        ValueError,
        "needs at least one sensor",
    ),
    "a threshold short": (
        lambda m: {"thresholds": [1.0] * 4},
        ValueError,
        "channels and thresholds are not one per class",
    ),
    "a channel short": (
        lambda m: {"channels": [[k] for k in range(4)]},
        ValueError,
        "channels and thresholds are not one per class",
    ),
}


@pytest.mark.parametrize("label", list(CONSTRUCTION))
def test_march2_refuses_what_it_is_built_from_when_it_cannot_read_it(
    label: str,
) -> None:
    """March2's construction refuses a spacing, a class count, a sensor or a monitor list that does not agree."""
    changes, exception, match = CONSTRUCTION[label]
    with pytest.raises(exception, match=match):
        _cfdcore.March2(**_march_args(**changes(_clean_room()[1])))
    _cfdcore.March2(**_march_args())


def test_march_restore_refuses_a_recovery_state_it_has_no_recovery_for(
    tmp_path: Path,
) -> None:
    """A run built without a recovery refuses a recovery state; with one it takes it."""
    ready = _ready(tmp_path)
    config = ready.state.config
    bare = core_mapping.core_march(
        config,
        ready.state.mesh,
        ready.state.cells,
        [onset for _, onset in ready.state.events],
        None,
    )
    with pytest.raises(
        ValueError, match="recovery state given to a run without a recovery"
    ):
        bare.restore(
            ready.cube, ready.terms, 0, 0.0, ready.maps, np.nan, ready.recovery
        )
    bare.restore(ready.cube, ready.terms, 0, 0.0, ready.maps, np.nan, None)
    ready.restore()


@pytest.mark.parametrize("what", ["fields", "first_above"])
def test_march_restore_refuses_an_unaligned_array(what: str, tmp_path: Path) -> None:
    """The fields and the maps a restore reads must be aligned, as the reference's sums need."""
    ready = _ready(tmp_path)
    array = {"fields": ready.cube, "first_above": ready.maps}[what]
    with pytest.raises(TypeError, match=r"March2.restore: an array is not aligned"):
        ready.restore(**{what: _unaligned(array)})


@pytest.mark.parametrize(
    "name", ["last_above", "sensor_time", "mean_time", "class_since"]
)
def test_march_restore_refuses_an_unaligned_or_short_recovery_array(
    name: str, tmp_path: Path
) -> None:
    """Each array of the recovery state is refused short and unaligned, by its name."""
    ready = _ready(tmp_path)
    state = ready.recovery
    with pytest.raises(TypeError, match=f"recovery state's {name} is not"):
        ready.restore(recovery={**state, name: state[name].reshape(-1)[1:].copy()})
    with pytest.raises(TypeError, match=r"March2.restore: an array is not aligned"):
        ready.restore(recovery={**state, name: _unaligned(state[name])})


def _blocks() -> dict[str, object]:
    """The blocks' construction arguments for the 80x30 clean-out room."""
    config, mesh = _clean_room()
    thresholds = list(config.class_thresholds("the test"))
    levels = _initial_levels(np.array(thresholds))
    keys = core_mapping.recovery_keys(np.array(thresholds), levels, None)
    return {
        "grid": _grid(mesh),
        "thresholds": thresholds,
        "channels": [list(m) for m in config.channel_classes()],
        "cube": np.zeros((len(thresholds), *mesh.cell_type.shape)),
        "cells": list(core_mapping_cells(config, mesh)),
        "keys": keys,
        # One class fewer in every list, for a tracker that agrees with itself and
        # not with the channels.
        "keys_short": {
            "target": keys["target"][:-1],
            "defined": keys["defined"][:-1],
            "hold": None,
        },
        "maps": np.full((1, len(thresholds), *mesh.cell_type.shape), np.nan),
    }


BLOCKS = {
    "ArrivalMaps2: a threshold short": lambda b: _cfdcore.ArrivalMaps2(
        b["grid"], b["thresholds"][:-1], [0.0], b["channels"]
    ),
    "Recovery2: a threshold short": lambda b: _cfdcore.Recovery2(
        b["grid"], b["thresholds"][:-1], b["keys_short"], b["cells"], b["channels"]
    ),
    "ArrivalMaps2.update: a class short": lambda b: _cfdcore.ArrivalMaps2(
        b["grid"], b["thresholds"], [0.0], b["channels"]
    ).update(0.5, b["cube"][:-1].copy()),
    "ArrivalMaps2.update: unaligned": lambda b: _cfdcore.ArrivalMaps2(
        b["grid"], b["thresholds"], [0.0], b["channels"]
    ).update(0.5, _unaligned(b["cube"])),
    "ArrivalMaps2.restore: a map short": lambda b: _cfdcore.ArrivalMaps2(
        b["grid"], b["thresholds"], [0.0], b["channels"]
    ).restore(b["maps"][:, :-1].copy(), 0.5),
    "ArrivalMaps2.restore: unaligned": lambda b: _cfdcore.ArrivalMaps2(
        b["grid"], b["thresholds"], [0.0], b["channels"]
    ).restore(_unaligned(b["maps"]), 0.5),
    "Recovery2.update: a class short": lambda b: _cfdcore.Recovery2(
        b["grid"], b["thresholds"], b["keys"], b["cells"], b["channels"]
    ).update(0.5, b["cube"][:-1].copy(), [0.0] * 5),
    "Recovery2.update: unaligned": lambda b: _cfdcore.Recovery2(
        b["grid"], b["thresholds"], b["keys"], b["cells"], b["channels"]
    ).update(0.5, _unaligned(b["cube"]), [0.0] * 5),
    "Recovery2.update: a mean short": lambda b: _cfdcore.Recovery2(
        b["grid"], b["thresholds"], b["keys"], b["cells"], b["channels"]
    ).update(0.5, b["cube"], [0.0] * 4),
    "channel_fields: a class short": lambda b: _cfdcore.channel_fields(
        b["grid"], b["cube"][:-1].copy(), b["channels"]
    ),
    "channel_fields: unaligned": lambda b: _cfdcore.channel_fields(
        b["grid"], _unaligned(b["cube"]), b["channels"]
    ),
    "channel_fields: a channel left out": lambda b: _cfdcore.channel_fields(
        b["grid"], b["cube"], [[0], [0], [2], [3], [4]]
    ),
    "sensor_channels: a class short": lambda b: _cfdcore.sensor_channels(
        np.zeros((4, 4)), b["channels"]
    ),
    "sensor_channels: unaligned": lambda b: _cfdcore.sensor_channels(
        _unaligned(np.zeros((5, 4))), b["channels"]
    ),
}
BLOCK_REFUSALS = {
    "ArrivalMaps2: a threshold short": (
        ValueError,
        "ArrivalMaps2: one threshold per channel",
    ),
    "Recovery2: a threshold short": (
        ValueError,
        "Recovery2: one threshold per channel",
    ),
    "ArrivalMaps2.update: a class short": (
        TypeError,
        r"ArrivalMaps2.update: fields is not \[5, 30, 80\]",
    ),
    "ArrivalMaps2.update: unaligned": (
        TypeError,
        "ArrivalMaps2.update: an array is not aligned",
    ),
    "ArrivalMaps2.restore: a map short": (
        TypeError,
        "ArrivalMaps2.restore: first_above is not",
    ),
    "ArrivalMaps2.restore: unaligned": (
        TypeError,
        "ArrivalMaps2.restore: an array is not aligned",
    ),
    "Recovery2.update: a class short": (
        TypeError,
        r"Recovery2.update: fields is not \[5, 30, 80\]",
    ),
    "Recovery2.update: unaligned": (
        TypeError,
        "Recovery2.update: an array is not aligned",
    ),
    "Recovery2.update: a mean short": (
        ValueError,
        "Recovery2.update: means hold one value per class",
    ),
    "channel_fields: a class short": (
        TypeError,
        r"channel_fields: fields is not \[5, 30, 80\]",
    ),
    "channel_fields: unaligned": (TypeError, "channel_fields: an array is not aligned"),
    "channel_fields: a channel left out": (
        ValueError,
        "channel 1 leaves out its own class",
    ),
    "sensor_channels: a class short": (
        TypeError,
        r"sensor_channels: readings is not \[classes, sensors\]",
    ),
    "sensor_channels: unaligned": (
        TypeError,
        "sensor_channels: an array is not aligned",
    ),
}


@pytest.mark.parametrize("label", list(BLOCKS))
def test_blocks_refuse_what_they_cannot_read(label: str) -> None:
    """Each block's binding refuses a wrong shape, an unaligned array and a list that does not agree."""
    exception, match = BLOCK_REFUSALS[label]
    with pytest.raises(exception, match=match):
        BLOCKS[label](_blocks())


# Another cube or maps array in each dimension in turn, copied so that it stays C-contiguous
# (a sliced view is refused for its layout before any shape is read).
CUBE_SHAPES = {
    "a class short": lambda a: a[:-1].copy(),
    "a row short": lambda a: a[:, 1:].copy(),
    "a column short": lambda a: a[:, :, 1:].copy(),
}
MAPS_SHAPES = {
    "an event too many": lambda a: np.concatenate([a, a]),
    "a channel short": lambda a: a[:, :-1].copy(),
    "a row short": lambda a: a[:, :, 1:].copy(),
    "a column short": lambda a: a[:, :, :, 1:].copy(),
}


@pytest.mark.parametrize("shape", list(CUBE_SHAPES))
def test_march_restore_refuses_a_cube_of_another_shape(
    shape: str, tmp_path: Path
) -> None:
    """March2.restore reads the class fields as [classes, ny, nx], each dimension checked."""
    ready = _ready(tmp_path)
    with pytest.raises(
        TypeError, match=r"March2\.restore: fields is not \[5, 30, 80\]"
    ):
        ready.restore(fields=CUBE_SHAPES[shape](ready.cube))


@pytest.mark.parametrize("shape", list(MAPS_SHAPES))
def test_march_restore_refuses_maps_of_another_shape(
    shape: str, tmp_path: Path
) -> None:
    """March2.restore reads the maps as [events, channels, ny, nx], each dimension checked."""
    ready = _ready(tmp_path)
    with pytest.raises(TypeError, match=r"March2\.restore: first_above is not"):
        ready.restore(first_above=MAPS_SHAPES[shape](ready.maps))


@pytest.mark.parametrize("shape", list(CUBE_SHAPES))
@pytest.mark.parametrize(
    "entry", ["ArrivalMaps2.update", "Recovery2.update", "channel_fields"]
)
def test_blocks_refuse_a_cube_of_another_shape(entry: str, shape: str) -> None:
    """Each block that reads a cube of fields checks every dimension of it."""
    b = _blocks()
    cube = CUBE_SHAPES[shape](b["cube"])
    calls = {
        "ArrivalMaps2.update": lambda: _cfdcore.ArrivalMaps2(
            b["grid"], b["thresholds"], [0.0], b["channels"]
        ).update(0.5, cube),
        "Recovery2.update": lambda: _cfdcore.Recovery2(
            b["grid"], b["thresholds"], b["keys"], b["cells"], b["channels"]
        ).update(0.5, cube, [0.0] * 5),
        "channel_fields": lambda: _cfdcore.channel_fields(
            b["grid"], cube, b["channels"]
        ),
    }
    with pytest.raises(TypeError, match=rf"{entry}: fields is not \[5, 30, 80\]"):
        calls[entry]()


@pytest.mark.parametrize("shape", list(MAPS_SHAPES))
def test_arrival_maps_refuse_maps_of_another_shape(shape: str) -> None:
    """ArrivalMaps2.restore checks every dimension of the maps it is handed."""
    b = _blocks()
    arrival = _cfdcore.ArrivalMaps2(b["grid"], b["thresholds"], [0.0], b["channels"])
    with pytest.raises(TypeError, match=r"ArrivalMaps2\.restore: first_above is not"):
        arrival.restore(MAPS_SHAPES[shape](b["maps"]), 0.5)


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
