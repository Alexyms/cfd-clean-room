"""REQ-A05: layouts scored, ranked and searched on scenario run files (ADR-015 D, step 4).

The three event scenarios (the door, the filter breach, the dust release)
run on 80x30 to their own t_end, the door from rest (both flow states) and
the other two on its kept closed state. Their files alone give the units
(``arrival_from_run``, ``detection_units``); the file's four sensors and the
searched four are scored over the same units and ranked, and the searched
layout is at least as good as the file's under each rule. Integration tier:
the runs take about three minutes.

Measured on these runs (prompt 75, 2026-11-02): fifteen units; the file's
four detect none of them. Of the 215 mountable candidates (the product
file's placement section: walls, ceiling, obstacle tops and sides at 0.05
m, one cell on this grid) none reaches a door unit, since the door's plume
stays within 0.9 m of the left wall and the wall's one column of candidates
lies beside the open door's faces; the searched four detect the other ten
with a worst case of 1.97 s. The free optimum's four detect all fifteen
with a worst case of 0.19 s.
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

    def test_the_searched_four_rank_above_the_files_four_under_each_rule(
        self, evaluated: SimpleNamespace
    ) -> None:
        """Over the same units: worst case, mean and coverage no worse, ranked first.

        The worst-case search is exact (the set cover); the mean rule's
        search at four is greedy and still no worse than the file's.
        """
        file_cells = sensor_cells(evaluated.config, evaluated.mesh)
        mine = given_layout(evaluated.kept, file_cells, within=WITHIN)
        found = evaluated.found
        assert (found.method, found.rule, found.exact) == (SET_COVER, WORST_CASE, True)
        assert found.score.worst <= mine.score.worst
        assert found.score.mean <= mine.score.mean
        assert found.score.coverage >= mine.score.coverage
        assert rank([mine, found])[0] is found
        by_mean = search(evaluated.units, evaluated.mounted, 4, MEAN, within=WITHIN)
        assert by_mean.units == evaluated.kept.labels
        assert by_mean.score.mean <= mine.score.mean
        assert by_mean.score.mean <= found.score.mean

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
