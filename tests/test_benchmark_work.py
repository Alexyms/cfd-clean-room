"""Unit test for the benchmark harness work accounting in scripts/benchmark.py.

Lives beside the solver tests rather than in a tests/test_benchmark.py module
because the validation-consolidation branch adds that file; a second copy
here would collide with it at the next rebase.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.config import SimConfig
from src.mesh import FLUID, Mesh
from src.solver_ns import IterationState

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import benchmark  # noqa: E402 -- scripts/ is not a package; path set above


def _make_config_with_obstacle(tmp_path: Path) -> SimConfig:
    """A 10x10 unit-square case with a block covering sixteen interior cells."""
    raw = {
        "domain": {"width": 1.0, "height": 1.0, "nx": 10, "ny": 10},
        "fluid": {"density": 1.2, "viscosity": 1.81e-5, "temperature": 293.0},
        "particles": {
            "density": 1000.0,
            "sizes": [0.1e-6],
            "mean_free_path": 67.0e-9,
            "boundary_layer_thickness": 1.0e-3,
            "hepa_reference": {"diameters": [0.1e-6], "efficiencies": [0.99999]},
        },
        "solver": {
            "dt": 0.01,
            "t_end": 1.0,
            "output_interval": 10,
            "convergence_tol": 1.0e-6,
            "max_simple_iter": 100,
            "alpha_velocity": 0.7,
            "alpha_pressure": 0.3,
            "max_pressure_iter": 200,
            "pressure_tol": 1.0e-6,
        },
        "boundaries": {
            "top": {
                "type": "velocity_inlet",
                "location": "top",
                "x_start": 0.0,
                "x_end": 1.0,
                "velocity": 0.1,
            },
        },
        "obstacles": [
            {
                "name": "block",
                "x_start": 0.2,
                "x_end": 0.6,
                "y_start": 0.2,
                "y_end": 0.6,
            }
        ],
        "sensors": [{"name": "center", "x": 0.5, "y": 0.5}],
        "thresholds": {"0.1e-6": 100.0},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(raw, default_flow_style=False), encoding="utf-8")
    return SimConfig(str(path))


@pytest.mark.unit
def test_cell_updates_equal_fluid_cells_times_sweeps(tmp_path: Path) -> None:
    """cells_per_sweep * (2 + sweeps) per iteration, counting FLUID cells only."""
    config = _make_config_with_obstacle(tmp_path)
    mesh = Mesh(config)
    cells = benchmark.fluid_cells_per_sweep(mesh)
    assert cells == int(np.count_nonzero(mesh.cell_type == FLUID))
    assert cells == (config.ny - 2) * (config.nx - 2) - 16

    sweeps, iterations = 7, 5
    counter = benchmark.WorkCounter(cells)
    fields = np.zeros((config.ny, config.nx))
    for i in range(iterations):
        counter.record(
            IterationState(
                iteration=i,
                residual=1.0,
                pressure_sweeps=sweeps,
                u=fields,
                v=fields,
                p=fields,
            )
        )
    assert counter.work["cell_updates"] == iterations * cells * (2 + sweeps)
    assert counter.work["inner_sweeps"] == iterations * sweeps
    assert counter.work["outer_iterations"] == iterations
