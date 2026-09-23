"""Tests that the harness and the field viewer run the solver their method names.

Before ECR-001 step 6 the harness ``--method`` flag was a free-text label that
selected nothing, so a row could claim a solver it did not run. Each test
here runs a real solve on a 6x6 cavity and checks which solver class was
built, for both methods: a test of the staggered branch alone would pass if
the flag did nothing and the default happened to be the one expected.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.boundary import BoundaryManager
from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.mesh import FLUID, Mesh
from src.solver_ns import IterationState, NavierStokesSolver
from src.solver_staggered import StaggeredSolver
from validation.cases import load_case

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import benchmark  # noqa: E402 -- scripts/ is not a package; path set above
import view_field  # noqa: E402 -- scripts/ is not a package; path set above

TINY = "tiny_cavity"


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Register a 6x6 cavity preset and record every solver either script builds."""
    monkeypatch.setitem(benchmark.CASES, TINY, ("cavity", 6, 6))
    monkeypatch.setitem(view_field.CASE_GRIDS, TINY, ("cavity", 6, 6))
    names: list[str] = []

    class SpyCollocated(NavierStokesSolver):
        def __init__(
            self, mesh: Mesh, config: SimConfig, boundary: BoundaryManager
        ) -> None:
            names.append("collocated")
            super().__init__(mesh, config, boundary)

    class SpyStaggered(StaggeredSolver):
        def __init__(
            self, mesh: Mesh, config: SimConfig, boundary: StaggeredBoundary
        ) -> None:
            names.append("staggered")
            super().__init__(mesh, config, boundary)

    for module in (benchmark, view_field):
        monkeypatch.setattr(module, "NavierStokesSolver", SpyCollocated)
        monkeypatch.setattr(module, "StaggeredSolver", SpyStaggered)
    return names


@pytest.mark.integration
@pytest.mark.parametrize(
    ("method", "expected", "stages"),
    [
        (
            "collocated-jacobi",
            "collocated",
            {"momentum", "flux", "pressure", "correct"},
        ),
        ("staggered-jacobi", "staggered", {"momentum", "pressure", "correct"}),
    ],
)
def test_harness_method_selects_the_solver(
    method: str, expected: str, stages: set[str], built: list[str], tmp_path: Path
) -> None:
    """The row's method, definition and stage keys all come from the solver that ran."""
    results = tmp_path / "results.jsonl"
    code = benchmark.main(
        [
            "--cases",
            TINY,
            "--repeats",
            "1",
            "--method",
            method,
            "--results",
            str(results),
        ]
    )
    assert code == 0
    assert built == [expected]
    (row,) = [
        json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()
    ]
    assert row["method"] == method
    assert (
        row["work"]["cell_update_definition"]
        == benchmark.CELL_UPDATE_DEFINITIONS[method]
    )
    assert set(row["time"]["stages"]) == stages


@pytest.mark.unit
def test_harness_rejects_an_unknown_method(
    capsys: pytest.CaptureFixture[str], built: list[str]
) -> None:
    """argparse refuses the label, and run_case refuses it for a direct caller."""
    with pytest.raises(SystemExit) as exc:
        benchmark.main(["--method", "collocated-gauss-seidel", "--cases", TINY])
    assert exc.value.code == 2
    assert "--method" in capsys.readouterr().err
    with pytest.raises(ValueError, match="unknown method"):
        benchmark.run_case(TINY, "staggered", sample_every=10, concurrent=1)
    assert built == []


@pytest.mark.unit
def test_collocated_definition_and_count_are_unchanged() -> None:
    """The collocated row still says and counts what the stored rows were measured with."""
    assert benchmark.CELL_UPDATE_DEFINITIONS["collocated-jacobi"] == (
        "stencil evaluations at FLUID cells: two momentum sweeps per outer iteration "
        "plus one per Jacobi pressure sweep; SOLID cells are not counted"
    )
    counter = benchmark.WorkCounter(324)
    assert counter.momentum_updates == 2 * 324


@pytest.mark.integration
def test_staggered_work_counts_faces_and_every_cell_with_an_equation() -> None:
    """On a 20x20 cavity: 380 + 380 interior faces, 400 cells, against 324 FLUID cells."""
    config = load_case("cavity", grid=(20, 20))
    mesh = Mesh(config)
    momentum, cells = benchmark.staggered_updates(mesh, config)
    assert momentum == 20 * 19 + 19 * 20
    assert cells == 400
    assert (
        benchmark.fluid_cells_per_sweep(mesh) == 324 == (mesh.cell_type == FLUID).sum()
    )

    counter = benchmark.WorkCounter(cells, momentum)
    fields = mesh.xc[None, :] * mesh.yc[:, None]
    for i, sweeps in enumerate((500, 312)):
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
    assert counter.work["cell_updates"] == 2 * momentum + cells * (500 + 312)
    assert counter.work["inner_sweeps"] == 812


@pytest.mark.integration
def test_staggered_work_excludes_the_outlet_faces() -> None:
    """The outlet column has no momentum diagonal, so it is not an unknown."""
    config = load_case("poiseuille", grid=(12, 6))
    mesh = Mesh(config)
    momentum, cells = benchmark.staggered_updates(mesh, config)
    assert momentum == 11 * 6 + 12 * 5
    assert cells == 72


@pytest.mark.integration
@pytest.mark.parametrize(
    ("method", "expected", "stem"),
    [
        ("collocated-jacobi", "collocated", TINY),
        ("staggered-jacobi", "staggered", f"{TINY}_staggered-jacobi"),
    ],
)
def test_viewer_method_selects_the_solver(
    method: str, expected: str, stem: str, built: list[str], tmp_path: Path
) -> None:
    """The viewer builds the named solver and keeps the two solvers' files apart."""
    path = view_field.solve_and_save(TINY, tmp_path, method)
    assert built == [expected]
    assert path == tmp_path / f"{stem}.npz"
    with view_field.np.load(path) as data:
        assert str(data["method"]) == method


@pytest.mark.unit
def test_viewer_rejects_an_unknown_method(
    capsys: pytest.CaptureFixture[str], built: list[str], tmp_path: Path
) -> None:
    """The command line refuses the label before any solve, and so does the function."""
    with pytest.raises(SystemExit) as exc:
        view_field.main([TINY, "--method", "staggered", "--out", str(tmp_path)])
    assert exc.value.code == 2
    assert "--method" in capsys.readouterr().err
    with pytest.raises(ValueError, match="unknown method"):
        view_field.solve_and_save(TINY, tmp_path, "staggered")
    assert built == []
