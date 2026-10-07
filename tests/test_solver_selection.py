"""Tests that the harness and the field viewer run the solver their method names.

Before ECR-001 step 6 the harness ``--method`` flag was a free-text label that
selected nothing, so a row could claim a solver it did not run. Each test
here runs a real solve on a 6x6 cavity and checks which solver class was
built. Since the collocated solver's retirement (2026-10-02, tag
collocated-final) one method runs; its old label is refused by name with its
own message, distinct from the unknown-method refusal, so a caller learns
where the solver went.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

from src.boundary_staggered import StaggeredBoundary
from src.config import SimConfig
from src.mesh import FLUID, Mesh
from src.pressure import PRESSURE_SOLVER_VERSION, STAGGERED_METHODS
from src.solver_staggered import StaggeredSolver
from src.stopping import IterationState
from validation.cases import load_case

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import benchmark  # noqa: E402 -- scripts/ is not a package; path set above
import self_convergence  # noqa: E402 -- scripts/ is not a package; path set above
import view_field  # noqa: E402 -- scripts/ is not a package; path set above

TINY = "tiny_cavity"


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Register a 6x6 cavity preset and record every solver either script builds."""
    monkeypatch.setitem(benchmark.CASES, TINY, ("cavity", 6, 6))
    monkeypatch.setitem(view_field.CASE_GRIDS, TINY, ("cavity", 6, 6))
    names: list[str] = []

    class SpyStaggered(StaggeredSolver):
        def __init__(
            self, mesh: Mesh, config: SimConfig, boundary: StaggeredBoundary
        ) -> None:
            names.append("staggered")
            super().__init__(mesh, config, boundary)

    for module in (benchmark, view_field):
        monkeypatch.setattr(module, "StaggeredSolver", SpyStaggered)
    return names


@pytest.mark.integration
def test_harness_method_selects_the_solver(built: list[str], tmp_path: Path) -> None:
    """The row's method, definition and stage keys all come from the solver that ran."""
    method = "staggered-cg"
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
    assert built == ["staggered"]
    (row,) = [
        json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()
    ]
    assert row["method"] == method
    assert (
        row["work"]["cell_update_definition"]
        == benchmark.CELL_UPDATE_DEFINITIONS[method]
    )
    assert set(row["time"]["stages"]) == {"momentum", "pressure", "correct"}


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
def test_harness_refuses_the_retired_collocated_method_by_name(
    capsys: pytest.CaptureFixture[str], built: list[str]
) -> None:
    """The refusal names the tag that holds the solver, not just an unknown label.

    Defect caught: the refusal removed, so the label falls through to the
    unknown-method error, which does not say where the solver went.
    """
    with pytest.raises(SystemExit) as exc:
        benchmark.main(["--method", "collocated-jacobi", "--cases", TINY])
    assert exc.value.code == 2
    assert "--method" in capsys.readouterr().err
    with pytest.raises(ValueError, match="collocated-final"):
        benchmark.run_case(TINY, "collocated-jacobi", sample_every=10, concurrent=1)
    assert "collocated-jacobi" not in benchmark.METHODS
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

    # Review 37 S5: the work counts every product with the operator, the
    # exit checks included, so the products, not the iterations, set it.
    counter = benchmark.WorkCounter(cells, momentum)
    fields = mesh.xc[None, :] * mesh.yc[:, None]
    for i, (iterations, products) in enumerate(((500, 501), (312, 314))):
        counter.record(
            IterationState(
                iteration=i,
                residual=1.0,
                pressure_iterations=iterations,
                u=fields,
                v=fields,
                p=fields,
                pressure_products=products,
            )
        )
    assert counter.work["cell_updates"] == 2 * momentum + cells * (501 + 314)
    assert counter.work["inner_iterations"] == 812
    assert counter.work["inner_products"] == 815


@pytest.mark.integration
def test_staggered_work_excludes_the_outlet_faces() -> None:
    """The outlet column has no momentum diagonal, so it is not an unknown."""
    config = load_case("poiseuille", grid=(12, 6))
    mesh = Mesh(config)
    momentum, cells = benchmark.staggered_updates(mesh, config)
    assert momentum == 11 * 6 + 12 * 5
    assert cells == 72


@pytest.mark.integration
def test_viewer_method_selects_the_solver(built: list[str], tmp_path: Path) -> None:
    """The viewer builds the named solver and names the file after the method."""
    method = "staggered-cg"
    path = view_field.solve_and_save(TINY, tmp_path, method)
    assert built == ["staggered"]
    assert path == tmp_path / f"{TINY}_{method}.npz"
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


@pytest.mark.unit
def test_every_script_files_its_results_under_the_current_solvers_one_label() -> None:
    """The harness, the viewer and the saved-field script import src/pressure.py's label.

    Review 37 S4: the label was spelled out in each script, so raising
    PRESSURE_SOLVER_VERSION could leave one of them filing a new solve's
    results under the old label. Defect caught: a script binding its own
    STAGGERED_METHOD, whatever its value, instead of importing the one
    pressure.py looks up by version.
    """
    label = STAGGERED_METHODS[PRESSURE_SOLVER_VERSION]
    for module in (benchmark, self_convergence, view_field):
        assert label == module.STAGGERED_METHOD, module.__name__
        assert (label,) == module.METHODS
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        bound = [
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign | ast.AnnAssign)
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name)
        ]
        imported = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "src.pressure"
            for alias in node.names
        ]
        assert "STAGGERED_METHOD" not in bound, module.__name__
        assert "STAGGERED_METHOD" in imported, module.__name__
    assert benchmark.DEFAULT_METHOD == view_field.DEFAULT_METHOD == label
    assert STAGGERED_METHODS[1] == benchmark.STAGGERED_JACOBI_METHOD
