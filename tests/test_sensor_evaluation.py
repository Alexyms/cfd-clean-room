"""REQ-A05: layouts scored, ranked and searched on scenario run files (ADR-015 D, step 4).

The three event scenarios (the door, the filter breach, the dust release)
run on 80x30 to their own t_end, the door from rest (both flow states) and
the other two on its kept closed state. Their files alone give the units
(``arrival_from_run``, ``detection_units``); the file's four sensors and the
searched four are scored over the same units and ranked, and the searched
layout is at least as good as the file's under each rule. Integration tier:
the runs take about three minutes.

Prediction (h) of ADR-015 D, asserted as it was scored (prompt 75, 2026-11-02;
prompt 75b made the test say it): fifteen units, and the file's four detect
none of them. Of the 215 mountable candidates (the product file's placement
section: walls, ceiling, obstacle tops and sides at 0.05 m, one cell on this
grid) none reaches a door unit, so the door's five are reported uncovered and
the searched four detect the breach's and the dust's ten, with a worst case of
1.97 s. The door's plume stays within 0.9 m of the left wall and 1.3 m of the
floor; the wall column's cells below 2.1 m are beside the open doorway's
faces and excluded, and the ones above it lie out of the plume. The free
set's four detect all fifteen, with a worst case of 0.19 s.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from src.config import MEAN, MOUNT_FREE, WORST_CASE, SimConfig
from src.mesh import Mesh
from src.monitor import sensor_cells
from src.placement import (
    SET_COVER,
    STEP_RESOLVED,
    arrival_from_run,
    candidates,
    detection_units,
    exclude_units,
    given_layout,
    rank,
    score,
    search,
)
from src.time_integration import Provenance, run

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
COARSE_GRID = {"nx": 80, "ny": 30}
# ADR-011 G's per-cell bound, 1e-4 rho V_min / t_end, on 80x30 at the base's
# t_end of 60 s, as tests/test_time_integration.py runs this grid.
COARSE_MASS_IMBALANCE_TOL = 1.0e-4 * 1.2 * (0.1 * 0.1) / 60.0
PROVENANCE = Provenance("test", "2026-11-02T00:00:00", "test-machine", 0)
SCENARIOS = {
    "door": "scenario_door_leak.yaml",
    "breach": "scenario_filter_breach.yaml",
    "dust": "scenario_equipment_dust.yaml",
}
# The coverage rule's time for these tests; the product file states none.
WITHIN = 10.0


def coarse(name: str) -> SimConfig:
    """A reference scenario file on 80x30, its base an absolute path."""
    raw = yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))
    raw["scenario"]["base"] = str(CONFIGS / "clean_room_default.yaml")
    raw["domain"] = dict(COARSE_GRID)
    raw["solver"]["mass_imbalance_tol"] = COARSE_MASS_IMBALANCE_TOL
    return SimConfig.from_dict(raw)


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """The three runs, their units, the candidates and the searched layouts."""
    folder = tmp_path_factory.mktemp("evaluation")
    paths = {}
    door = run(coarse(SCENARIOS["door"]), folder / "door.h5", provenance=PROVENANCE)
    paths["door"] = door.path
    for name in ("breach", "dust"):
        record = run(
            coarse(SCENARIOS[name]),
            folder / f"{name}.h5",
            flows_from=door.path,
            provenance=PROVENANCE,
        )
        paths[name] = record.path
    config = coarse(SCENARIOS["door"])
    mesh = Mesh(config)
    spec = config.placement
    arrivals = {name: arrival_from_run(path) for name, path in paths.items()}
    units = detection_units(arrivals)
    mounted = candidates(mesh, config, spec.mount, spec.standoff)
    free = candidates(mesh, config, (MOUNT_FREE,), spec.standoff)
    found = search(units, mounted, spec.count, spec.rule, within=WITHIN)
    return SimpleNamespace(
        paths=paths,
        config=config,
        mesh=mesh,
        arrivals=arrivals,
        units=units,
        mounted=mounted,
        free=free,
        found=found,
        kept=exclude_units(units, found.uncovered),
    )


@pytest.mark.integration
class TestSensorEvaluation:
    def test_the_units_come_from_the_run_files_alone(
        self, evaluated: SimpleNamespace
    ) -> None:
        """Fifteen units, the files' own step-resolved maps, read again bitwise."""
        assert len(evaluated.units.labels) == 15
        assert [label[0] for label in evaluated.units.labels] == [
            name for name in ("door", "breach", "dust") for _ in range(5)
        ]
        for name, path in evaluated.paths.items():
            again = arrival_from_run(path)
            assert again.resolution == STEP_RESOLVED and again.fallback is None
            assert again.maps.tobytes() == evaluated.arrivals[name].maps.tobytes()
        again = detection_units(
            {name: arrival_from_run(path) for name, path in evaluated.paths.items()}
        )
        assert again.latencies.tobytes() == evaluated.units.latencies.tobytes()

    def test_the_uncovered_units_are_the_ones_no_candidate_reaches(
        self, evaluated: SimpleNamespace
    ) -> None:
        """The search's report against a direct reading of the candidates' latencies."""
        cells = evaluated.mounted
        table = evaluated.units.latencies[:, cells[:, 0], cells[:, 1]]
        unreached = tuple(
            label
            for label, row in zip(evaluated.units.labels, table, strict=True)
            if not np.isfinite(row).any()
        )
        assert evaluated.found.uncovered == unreached
        assert evaluated.found.units == evaluated.kept.labels

    def test_the_files_four_detect_none_of_the_fifteen_units(
        self, evaluated: SimpleNamespace
    ) -> None:
        """(h), first clause: the file's four sensors detect nothing in any scenario.

        Defect caught: the units' latencies read at the wrong cells, or a
        sensor's cell taken from another mesh, so that a layout the room
        never reaches reads as detecting.
        """
        file_cells = sensor_cells(evaluated.config, evaluated.mesh)
        mine = score(evaluated.units, file_cells, within=WITHIN)
        assert len(evaluated.units.labels) == 15
        assert mine.undetected == 15 and np.isinf(mine.worst)
        assert mine.coverage == 0.0

    def test_on_the_mountable_set_the_door_is_uncovered_and_the_rest_detected(
        self, evaluated: SimpleNamespace
    ) -> None:
        """(h), second clause as scored: the breach's and the dust's units, not the door's.

        The door's five units are the uncovered ones; the searched four
        detect every other unit (worst case a few seconds, coverage 1 within
        10 s); the file's four over the same ten units detect none, and the
        search ranks above them. Defects caught: the interval exclusion in
        ``candidates`` removed (the door's units become coverable, so
        ``uncovered`` is empty); a search that leaves a coverable unit
        undetected.
        """
        found = evaluated.found
        doors = tuple(label for label in evaluated.units.labels if label[0] == "door")
        assert len(doors) == 5 and found.uncovered == doors
        assert {label[0] for label in found.units} == {"breach", "dust"}
        assert len(found.units) == 10 and found.units == evaluated.kept.labels
        assert found.score.undetected == 0 and found.score.coverage == 1.0
        assert 0.0 < found.score.worst < 5.0
        file_cells = sensor_cells(evaluated.config, evaluated.mesh)
        mine = given_layout(evaluated.kept, file_cells, within=WITHIN)
        assert mine.score.undetected == 10 and mine.score.coverage == 0.0
        assert (found.method, found.rule, found.exact) == (SET_COVER, WORST_CASE, True)
        assert rank([mine, found])[0] is found

    def test_the_mean_rule_detects_the_same_ten_units(
        self, evaluated: SimpleNamespace
    ) -> None:
        """The mean rule's four over the mountable set: the same units, all detected.

        Defect caught: the mean rule's greedy settling on a layout that
        leaves a coverable unit at its cap.
        """
        by_mean = search(evaluated.units, evaluated.mounted, 4, MEAN, within=WITHIN)
        assert by_mean.uncovered == evaluated.found.uncovered
        assert by_mean.units == evaluated.kept.labels
        assert by_mean.score.undetected == 0
        file_cells = sensor_cells(evaluated.config, evaluated.mesh)
        mine = given_layout(evaluated.kept, file_cells, within=WITHIN)
        assert by_mean.score.mean < mine.score.mean

    def test_on_the_free_set_every_unit_is_detected(
        self, evaluated: SimpleNamespace
    ) -> None:
        """(h), second clause on the free set: all fifteen units, the door's included.

        Nothing is uncovered, the set cover finds a layout of four detecting
        all fifteen, and its worst case is below the mountable layout's.
        Defect caught: the free set losing the cells beside the doorway (the
        door's units uncovered again).
        """
        free = search(evaluated.units, evaluated.free, 4, WORST_CASE, within=WITHIN)
        assert free.uncovered == () and free.units == evaluated.units.labels
        assert free.score.undetected == 0 and free.score.coverage == 1.0
        assert (free.method, free.rule) == (SET_COVER, WORST_CASE)
        assert free.score.worst < evaluated.found.score.worst

    def test_the_free_optimum_bounds_the_mountable_one(
        self, evaluated: SimpleNamespace
    ) -> None:
        """Every mountable candidate is a free one: over the same units, no worse."""
        mounted = {tuple(cell) for cell in evaluated.mounted.tolist()}
        assert mounted <= {tuple(cell) for cell in evaluated.free.tolist()}
        free = search(evaluated.kept, evaluated.free, 4, WORST_CASE)
        assert free.uncovered == ()
        assert free.score.worst <= evaluated.found.score.worst
        assert free.distinct_masks is not None and free.maximal_masks is not None
        assert free.maximal_masks <= free.distinct_masks
