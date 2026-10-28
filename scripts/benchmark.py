"""Benchmark harness: append one JSON record per solver run to benchmarks/results.jsonl.

Three independent changes are coming: the staggered rebuild changes the
accuracy obtained on a given grid, the linear solver choice changes how much
work is needed to reach that accuracy, and a GPU port changes how fast that
work executes. A record of wall time alone cannot attribute an improvement
to any of the three. Each record therefore carries accuracy, work in
hardware-independent cell updates, wall time, and a trajectory of error
against cumulative work sampled during the solve, so any fixed-accuracy
comparison can be read off later without rerunning anything.

The results file is append-only and never rewritten. Repeats are stored as
separate records; averaging is a presentation decision.

Each row runs the case file's stopping rule, and its params say which. A
velocity-step stop is labelled residual_below_tol, the label every stored
velocity-step row carries, including the collocated-jacobi rows taken before
that solver was retired (tag collocated-final). An error_estimate row also
records the rule's version in its params, read from the solver
(``rule_version``, ECR-002 step 6): 3, or with the turbulence model's
condition (e) 4 until 2026-10-11 and 5 since (its scale, ADR-012 E). A row
with the turbulence section records the section in its params as well.

Seven case families. Poiseuille and the cavity are scored from the
cell-centred field, every trajectory sample and the row alike. The Annex 20
room (VAL-017, ``val017_216x72``) is scored from the final face velocities
by the allowance of ADR-012 decision 6 (``validation.annex20.score_lines``):
each line's RMS against the measurements beside the published RMS, pass or
fail on the two scored lines, and as the row's value the larger ratio of
RMS to allowance over them, so 1 or below passes. The per-iteration callback
carries no faces, so its trajectory samples record no error and the last
entry carries the row's. The run moved here from the suite on 2026-10-11:
it costs about 20 minutes alone (the stored row's 4,934 s were taken beside
other runs). The backward-facing step (VAL-019, ``val019_656x60``) is
reported, not scored: no allowance is set (ADR-012 G (ii), OPEN), so the
row carries ``validation.backstep.evaluate``'s whole comparison (the
reattachment length, the inflow check, each profile's and the wall
coefficients' departures from Driver and Seegmiller), its value is the
reattachment length's relative departure from the measured 6.26 H (NaN when
the floor shear never turns positive), and ``passed`` is None. The impinging
jet (VAL-020, ``val020_272x90``) is reported the same way: the row carries
``validation.impinging_jet.evaluate``'s whole comparison with Khayrullina et
al. (2017), its value is the centreline's RMS departure in V_0,y / V0, and
``passed`` is None (ECR-002 criterion 13, OPEN). The product room (VAL-018,
``val018_320x120``) has no reference: its row carries the stop, the outer
count over the cap as its value, ``passed`` whether the solve stopped by
the rule within the cap (criterion 10's convergence clause), and the
readings the step 8 report compares its runs by (the worst cell imbalance,
the core eddy viscosity); the comparative clause is judged in that report.
The square cylinder (VAL-022, ``val022_*``, ADR-014 F) is a tier-2 run: the
probe's perturbed start-up (``validation.cylinder.perturbed_startup``), then
``time_integration.run`` marching it to the preset's end, the run file under
results/val022/. Each row carries its criterion's reading: Re 40 the decay of
the lift's amplitudes, Re 60 their sustain, Re 100 the Strouhal number of the
last ten periods against 0.139 to 0.153 with the drag coefficient beside it,
and the uniform rung's Strouhal number, reported with ``passed`` None. Its
value is the Strouhal number (Re 100 and the rung) or the extreme amplitude
ratio (the largest for Re 40, the smallest for Re 60), and every step's lift,
drag and outer count go to docs/reports/val022/<preset>.json beside the row.

The method label names the pressure solve as well as the grid: staggered-cg
since ECR-003 step 1 (2026-10-06), when conjugate gradients replaced the
weighted Jacobi sweep, and staggered-mgcg since prompt 54 (2026-10-14), when
multigrid became the preconditioner by default. The label selects the
preconditioner: a staggered-cg row runs the diagonal whatever the case file
says, so the stored rows keep their label and summarize apart from the
multigrid ones, as the collocated rows do. The 20 stored staggered-jacobi rows
keep their label too; run_case refuses both retired labels by name. Every row
records the solver's pressure_cap_hits beside its stop.

Run:

    python scripts/benchmark.py                       # seed cases, 3 repeats
    python scripts/benchmark.py --cases val002_20x20  # one case
    python scripts/benchmark.py --summary             # table of what is stored
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.boundary_staggered import (  # noqa: E402 -- follows sys.path.insert
    StaggeredBoundary,
)
from src.config import (  # noqa: E402 -- follows sys.path.insert
    ERROR_ESTIMATE,
    MULTIGRID,
    PRESSURE_JACOBI,
    SOLVER_KEYS,
    VELOCITY_STEP,
    SimConfig,
)
from src.io_manager import RunFile  # noqa: E402 -- follows sys.path.insert
from src.mesh import FLUID, SOLID, Mesh  # noqa: E402 -- follows sys.path.insert
from src.momentum import MomentumPredictor  # noqa: E402 -- follows sys.path.insert
from src.pressure import (  # noqa: E402 -- follows sys.path.insert
    STAGGERED_METHOD,
    STAGGERED_METHODS,
    PressureCorrector,
)
from src.solver_staggered import (  # noqa: E402 -- follows sys.path.insert
    StaggeredSolver,
    rule_version,
)
from src.staggered import (  # noqa: E402 -- follows sys.path.insert
    FaceVelocities,
    allocate_fields,
)
from src.stopping import (  # noqa: E402 -- follows sys.path.insert
    IterationState,
)
from src.time_integration import (  # noqa: E402 -- follows sys.path.insert
    Provenance,
    TransientMarchStep,
    TransientRunRecord,
    run,
)
from validation import (  # noqa: E402 -- follows sys.path.insert
    annex20,
    backstep,
    cylinder,
    impinging_jet,
)
from validation.cases import (  # noqa: E402 -- follows sys.path.insert
    CASE_GRIDS,
    CYLINDER_PRESETS,
    load_preset,
    with_pressure_preconditioner,
)
from validation.metrics import (  # noqa: E402 -- follows sys.path.insert
    cavity_marchi_centerline_errors,
    poiseuille_l2_error,
)

# 2 since ECR-003 step 1: work.inner_sweeps became inner_iterations, and
# work.inner_products and outcome.pressure_cap_hits were added. Rows of
# schema 1 are read as before; nothing reads the renamed key back.
SCHEMA_VERSION = 2
RESULTS_PATH = REPO_ROOT / "benchmarks" / "results.jsonl"
# The method label selects the solver, so a row cannot claim one it did not run.
# The solver of record's is src/pressure.py's, looked up there by
# PRESSURE_SOLVER_VERSION. The diagonal-preconditioned CG of version 2 stays
# runnable under its own label, for parity work, and run_case sets the
# preconditioner the label names.
CG_JACOBI_METHOD = STAGGERED_METHODS[2]
DEFAULT_METHOD = STAGGERED_METHOD
METHODS = (STAGGERED_METHOD, CG_JACOBI_METHOD)
METHOD_PRECONDITIONERS: dict[str, str] = {
    STAGGERED_METHODS[3]: MULTIGRID,
    CG_JACOBI_METHOD: PRESSURE_JACOBI,
}
# Retired labels stay known so their stored rows still summarize; run_case
# refuses each by name with where its solver went. The collocated solver was
# retired on 2026-10-02 (tag collocated-final); the staggered solver's weighted
# Jacobi sweep on 2026-10-06 (ECR-003 step 1), its loop in the history of
# src/pressure.py and its evidence in docs/reports/pressure_correction_step5.md.
COLLOCATED_METHOD = "collocated-jacobi"
STAGGERED_JACOBI_METHOD = STAGGERED_METHODS[1]
RETIRED_METHODS: dict[str, str] = {
    COLLOCATED_METHOD: (
        "was retired on 2026-10-02 and takes no new row; its stored rows still "
        "summarize, and the solver is at tag collocated-final"
    ),
    STAGGERED_JACOBI_METHOD: (
        "was retired on 2026-10-06 with the weighted Jacobi pressure sweep "
        "(ECR-003 step 1) and takes no new row; its stored rows still summarize, "
        f"and the solver of record runs as {STAGGERED_METHOD}"
    ),
}

# Grid presets come from validation.cases so the harness, the tests and the
# field viewer name the same solve the same way. Every other setting comes
# from the committed case file.
CASES = CASE_GRIDS
DEFAULT_CASES = ["val001_80x40", "val002_20x20", "val002_40x40"]

# The family validation.cases gives the Annex 20 presets, scored from the faces.
ANNEX20_FAMILY = "annex20"
ANNEX20_METRIC = "rms_u_over_published_rms_scored_lines"
ANNEX20_REFERENCE = "annex20_measured_nielsen_1990; allowance rong_nielsen_2008_rms"

# The family validation.cases gives the backward-facing step presets, reported
# from the final faces, pressure and k; no allowance is set.
BACKSTEP_FAMILY = "backstep"
BACKSTEP_METRIC = "x_r_relative_departure_report_only"
BACKSTEP_REFERENCE = (
    "driver_seegmiller_1985 x_r/H 6.26 +/- 0.10; report-only, no allowance "
    "(ADR-012 G (ii) OPEN)"
)

# The family validation.cases gives the impinging jet presets, reported from
# the final faces and k; no allowance is set.
IMPINGING_JET_FAMILY = "impinging_jet"
IMPINGING_JET_METRIC = "centreline_rms_v_over_v0_report_only"
IMPINGING_JET_REFERENCE = (
    "khayrullina_2017 case I digitised; report-only, no allowance "
    "(ECR-002 criterion 13 OPEN)"
)

# The product room (VAL-018), reported from the stop and the final state;
# no reference exists, and criterion 10's convergence clause is the pass.
PRODUCT_FAMILY = "product"
PRODUCT_METRIC = "outer_iterations_over_cap_report_only"
PRODUCT_REFERENCE = (
    "none; ECR-002 criterion 10: the stop by error_estimate_and_continuity "
    "within max_simple_iter is the pass, the comparative clause is "
    "docs/reports/ecr002_step8_product.md"
)

# The square cylinder (VAL-022): tier-2 runs, their records beside the row.
CYLINDER_FAMILY = "cylinder"
CYLINDER_REFERENCE = (
    "sohankar_norberg_davidson_1998 St 0.146 (Table IV), C_D 1.45 to 1.49; "
    "thresholds Alex 2026-10-17 (ADR-014 F)"
)
VAL022_RECORDS = REPO_ROOT / "docs" / "reports" / "val022"
VAL022_RUNS = REPO_ROOT / "results" / "val022"

# The families whose row needs the finished solve (the final faces, k, or
# the stop), which the per-iteration callback lacks: their trajectory
# samples carry no error but the last.
FACE_SCORED_FAMILIES = (
    ANNEX20_FAMILY,
    BACKSTEP_FAMILY,
    IMPINGING_JET_FAMILY,
    PRODUCT_FAMILY,
)

# One stop_reason per rule. The velocity-step label is the one every row stored
# before the staggered solver reported its own stop, and every collocated row has.
STOP_LABELS = {"velocity_step_below_tol": "residual_below_tol"}


def solver_parameters(config: SimConfig) -> dict:
    """Every solver parameter as loaded, and under error_estimate the rule's version.

    The names are SOLVER_KEYS, the one list of what the solver block accepts,
    so a key added to the configuration is in every row (GitHub issue 38).
    """
    params = {name: getattr(config, name) for name in SOLVER_KEYS}
    if config.stopping_rule == ERROR_ESTIMATE:
        params["rule_version"] = rule_version(config)
    # The k and eps solve tolerance moves a coupled row's stop as much as any
    # solver key does (docs/reports/ecr002_val017_annex20.md, section 5.2).
    if config.turbulence is not None:
        params["turbulence"] = dataclasses.asdict(config.turbulence)
    return params


def git_state() -> tuple[str, bool]:
    """Identify the commit a run is measured on and whether the tree matches it.

    Returns
    -------
    tuple[str, bool]
        The HEAD commit hash, and True when any tracked file other than
        benchmarks/results.jsonl differs from that commit.
    """
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    ).stdout.splitlines()
    dirty = any(not line.endswith("benchmarks/results.jsonl") for line in status)
    return commit, dirty


# Recorded with every row, so each row says what its cell_updates number counts.
# The two retired entries describe the stored rows; no new row can carry them.
CELL_UPDATE_DEFINITIONS: dict[str, str] = {
    COLLOCATED_METHOD: (
        "stencil evaluations at FLUID cells: two momentum sweeps per outer iteration "
        "plus one per Jacobi pressure sweep; SOLID cells are not counted"
    ),
    STAGGERED_JACOBI_METHOD: (
        "stencil evaluations at unknowns: the unknown u and v faces (a_p > 0) once "
        "per outer iteration for momentum, plus the cells with a pressure equation "
        "(a_P > 0) once per weighted Jacobi pressure sweep"
    ),
    CG_JACOBI_METHOD: (
        "stencil evaluations at unknowns: the unknown u and v faces (a_p > 0) once "
        "per outer iteration for momentum, plus the cells with a pressure equation "
        "(a_P > 0) once per five-point product of the conjugate gradient solve: "
        "one per iteration and one per true-residual check, at exit, at a restart "
        "and at the cap; the diagonal scaling, the vector updates and the "
        "reductions are not counted"
    ),
    STAGGERED_METHODS[3]: (
        "stencil evaluations at unknowns: the unknown u and v faces (a_p > 0) once "
        "per outer iteration for momentum, plus the cells with a pressure equation "
        "(a_P > 0) once per five-point product of the conjugate gradient solve: "
        "one per iteration and one per true-residual check, at exit, at a restart "
        "and at the cap; the vector updates and the reductions are not counted, "
        "and neither are the multigrid V-cycle's own products and transfers, "
        "which a cycle forms on every level and which a row of "
        f"{CG_JACOBI_METHOD} has no counterpart of"
    ),
}


def fluid_cells_per_sweep(mesh: Mesh) -> int:
    """Number of unknowns one solver sweep updates: the FLUID cells.

    The work axis counts what the algorithm updates, not what a vectorised
    implementation touches. Whether an implementation spends arithmetic on
    masked SOLID cells is throughput, which the time axis already carries as
    seconds per cell update; counting them here would put an implementation
    detail inside the axis built to exclude it. The seed cases have no
    obstacles, so for them this equals the interior count (ny - 2) * (nx - 2)
    and the stored rows are unchanged.

    Parameters
    ----------
    mesh : Mesh
        Classified mesh of the case.

    Returns
    -------
    int
        Count of cells typed FLUID.
    """
    return int(np.count_nonzero(mesh.cell_type == FLUID))


def axis_clustering(mesh: Mesh) -> dict[str, dict[str, float]]:
    """Each axis's geometric ratio and wall cell as the mesh derived them.

    A row's case id names a preset whose definition can change, and nx and
    ny alone do not tell a clustered mesh from a uniform one of the same
    size. Rows stored before this field record nx and ny only; every one of
    them is uniform except val001_80x40_stretched, whose git_commit pins it.

    Parameters
    ----------
    mesh : Mesh
        Mesh of the solve.

    Returns
    -------
    dict[str, dict[str, float]]
        {"x": {"stretch_ratio", "min_wall_spacing"}, "y": the same}, in the
        mesh's units; a uniform axis has ratio 1.0 and wall cell L / n.
    """
    return {
        "x": {
            "stretch_ratio": float(mesh.stretch_ratio_x),
            "min_wall_spacing": float(mesh.min_spacing_x),
        },
        "y": {
            "stretch_ratio": float(mesh.stretch_ratio_y),
            "min_wall_spacing": float(mesh.min_spacing_y),
        },
    }


def staggered_updates(mesh: Mesh, config: SimConfig) -> tuple[int, int]:
    """Unknowns the staggered solver updates, read from its own coefficient masks.

    On the staggered grid the BOUNDARY ring cells are ordinary solution
    cells, so the FLUID count would understate the work: a 20x20 cavity
    has 400 cells with a pressure equation against 324 FLUID cells. The
    masks are structural, so one prediction from the initial field gives
    them for the whole solve.

    Parameters
    ----------
    mesh : Mesh
        Classified mesh of the case.
    config : SimConfig
        Case configuration.

    Returns
    -------
    tuple[int, int]
        (momentum updates per outer iteration, the unknown u faces plus the
        unknown v faces with a_p > 0; cells per product with the pressure
        operator, those with a_P > 0).
    """
    boundary = StaggeredBoundary(mesh, config)
    u, v, p = allocate_fields(mesh)
    boundary.apply_normal_velocity(u, v)
    prediction = MomentumPredictor(mesh, config, boundary).predict(u, v, p)
    coefficients = PressureCorrector(mesh, config, boundary).coefficients(
        prediction.a_p_u, prediction.a_p_v
    )
    momentum = np.count_nonzero(prediction.a_p_u > 0.0) + np.count_nonzero(
        prediction.a_p_v > 0.0
    )
    return int(momentum), int(np.count_nonzero(coefficients.a_p > 0.0))


class WorkCounter:
    """Accumulate the work record from the solver's per-iteration callbacks.

    Parameters
    ----------
    cells_per_iteration : int
        Unknowns one product with the pressure operator touches, a stencil
        evaluation each: the second value of staggered_updates. A conjugate
        gradient iteration forms one product, and its true-residual checks
        one each; a Jacobi sweep was one. The stored collocated-jacobi rows
        counted fluid_cells_per_sweep here.
    momentum_updates : int, optional
        Unknowns the momentum step updates per outer iteration. Defaults to
        ``2 * cells_per_iteration``, the convention the stored
        collocated-jacobi rows were counted with: u and v sweeps over the
        same cells.
    """

    def __init__(
        self, cells_per_iteration: int, momentum_updates: int | None = None
    ) -> None:
        self.cells_per_iteration = cells_per_iteration
        self.momentum_updates = (
            2 * cells_per_iteration if momentum_updates is None else momentum_updates
        )
        # Stored rows written before ECR-003 step 1 (schema 1) carry the
        # iteration count as inner_sweeps and no product count; nothing reads
        # either back.
        self.work: dict[str, int] = {
            "outer_iterations": 0,
            "inner_iterations": 0,
            "inner_products": 0,
            "cell_updates": 0,
        }

    def record(self, state: IterationState) -> None:
        """Add one SIMPLE iteration: its momentum updates plus its pressure products.

        Parameters
        ----------
        state : IterationState
            Snapshot handed to the solve_steady callback.
        """
        self.work["outer_iterations"] = state.iteration + 1
        self.work["inner_iterations"] += state.pressure_iterations
        self.work["inner_products"] += state.pressure_products
        self.work["cell_updates"] += (
            self.momentum_updates + self.cells_per_iteration * state.pressure_products
        )


def positive_int(text: str) -> int:
    """Parse a command line integer that must be at least 1.

    Parameters
    ----------
    text : str
        The raw argument.

    Returns
    -------
    int
        The parsed value.

    Raises
    ------
    argparse.ArgumentTypeError
        When the value is not an integer or is below 1. A caller who asked
        for zero samples or a negative repeat count hears about it instead
        of getting a ZeroDivisionError or a silent no-op.
    """
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def cpu_name() -> str:
    """Best available CPU model string without a third-party dependency."""
    name = ""
    if sys.platform == "win32":
        import winreg

        try:
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            name = os.environ.get("PROCESSOR_IDENTIFIER", "")
    cpuinfo = Path("/proc/cpuinfo")
    if not name and cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                name = line.split(":", 1)[1].strip()
                break
    return name or platform.processor() or platform.machine()


def accuracy_of(
    kind: str, config: SimConfig, mesh: Mesh, u: np.ndarray, v: np.ndarray
) -> dict:
    """The accuracy a record carries for one field of one case family.

    Parameters
    ----------
    kind : str
        Case family, "poiseuille" or "cavity".
    config : SimConfig
        Case configuration.
    mesh : Mesh
        Mesh the field was computed on.
    u, v : np.ndarray
        Cell-centered velocity fields [ny, nx].

    Returns
    -------
    dict
        The metric's as_dict: its name, value, reference and components. The
        cavity is scored against Marchi et al. (2009), ECR-001 criteria 3
        and 3a since step 8.

    Raises
    ------
    ValueError
        If ``kind`` is neither "poiseuille" nor "cavity"; a third family
        would otherwise be scored as a cavity.
    """
    if kind == "poiseuille":
        return poiseuille_l2_error(config, mesh, u).as_dict()
    if kind == "cavity":
        return cavity_marchi_centerline_errors(config, mesh, u, v).as_dict()
    raise ValueError(f"unknown case family {kind!r}; expected 'poiseuille' or 'cavity'")


def annex20_accuracy(mesh: Mesh, faces: FaceVelocities) -> dict:
    """The Annex 20 room's lines scored by the allowance of ADR-012 decision 6.

    Parameters
    ----------
    mesh : Mesh
        Mesh the faces belong to.
    faces : FaceVelocities
        The solve's final face velocities; u is sampled at the stations.

    Returns
    -------
    dict
        ``metric``, ``value`` (the larger ratio of RMS to allowance over the
        scored lines; 1 or below passes), ``reference``, ``components``
        (each line's RMS and allowance), ``lines`` (each line's RMS, station
        count, allowance, whether it is scored and, if so, whether it
        passes) and ``passed`` (every scored line passes).
    """
    scores = annex20.score_lines(annex20.sample_lines(mesh, np.array(faces.u)))
    scored = [scores[key] for key in annex20.SCORED_LINES]
    components: dict[str, float] = {}
    for key, score in scores.items():
        components[f"rms_{key}"] = score.rms
        components[f"allowance_{key}"] = score.allowance
    return {
        "metric": ANNEX20_METRIC,
        "value": max(score.rms / score.allowance for score in scored),
        "reference": ANNEX20_REFERENCE,
        "components": components,
        "lines": {
            key: {
                "rms": score.rms,
                "stations": score.stations,
                "allowance": score.allowance,
                "scored": score.scored,
                "pass": score.passed,
            }
            for key, score in scores.items()
        },
        "passed": all(score.passed for score in scored),
    }


def backstep_accuracy(
    mesh: Mesh, config: SimConfig, faces: FaceVelocities, p: np.ndarray, k: np.ndarray
) -> dict:
    """The backward-facing step's comparison with the data, reported and not scored.

    Parameters
    ----------
    mesh : Mesh
        Mesh the fields belong to.
    config : SimConfig
        The case, with its step obstacle and turbulence section.
    faces : FaceVelocities
        The solve's final face velocities; u is sampled.
    p : np.ndarray
        The final pressure on the cells, [ny, nx].
    k : np.ndarray
        The final turbulent kinetic energy on the cells, [ny, nx].

    Returns
    -------
    dict
        ``metric``, ``value`` (``|x_r / H - 6.26| / 6.26``, or NaN when the
        floor shear never turns positive), ``reference``, ``components``
        (the reattachment length, the centreline speed's ratio to U_ref,
        delta_99 / H and Re_theta at x/H = -4, each profile's RMS, the Cf
        and Cp RMS), ``report`` (``validation.backstep.evaluate``'s whole
        dictionary) and ``passed`` None: nothing is scored until Alex sets
        the allowance (ADR-012 G (ii)).
    """
    report = backstep.evaluate(mesh, config, np.array(faces.u), p, k)
    x_r = report["x_r_over_h"]
    value = (
        float("nan")
        if x_r is None
        else abs(x_r - backstep.X_R_MEASURED) / backstep.X_R_MEASURED
    )
    components: dict[str, float | None] = {
        "x_r_over_h": x_r,
        "u_ref_over_measured": report["u_ref"]["over_measured"],
        "delta_99_over_h": report["inflow"]["delta_99_over_h"],
        "re_theta": report["inflow"]["re_theta"],
        "rms_cf": report["cf"]["rms"],
        "rms_cp": report["cp"]["rms"],
    }
    for station, errors in report["profiles"].items():
        components[f"rms_profile_x{station}"] = errors["rms"]
    return {
        "metric": BACKSTEP_METRIC,
        "value": value,
        "reference": BACKSTEP_REFERENCE,
        "components": components,
        "report": report,
        "passed": None,
    }


def impinging_jet_accuracy(
    mesh: Mesh, config: SimConfig, faces: FaceVelocities, k: np.ndarray
) -> dict:
    """The impinging jet's comparison with the data, reported and not scored.

    Parameters
    ----------
    mesh : Mesh
        Mesh the fields belong to.
    config : SimConfig
        The case, with its nozzle and turbulence section.
    faces : FaceVelocities
        The solve's final face velocities; v is sampled.
    k : np.ndarray
        The final turbulent kinetic energy on the cells, [ny, nx].

    Returns
    -------
    dict
        ``metric``, ``value`` (the RMS of V_0,y / V0 against the data over
        the whole centreline), ``reference``, ``components`` (the
        centreline RMS over the free-jet part and the impingement region,
        the potential core, the decay and spreading rates, the half-width
        RMS, the stagnation ratio at the data's nearest station and at the
        nearest Re 8,000 ring station, the plate's first-node y*), ``report``
        (``validation.impinging_jet.evaluate``'s whole dictionary) and
        ``passed`` None: nothing is scored until Alex sets the allowance
        (ECR-002 criterion 13).
    """
    report = impinging_jet.evaluate(mesh, config, np.array(faces.v), k)
    half_width = report["half_width"]
    components: dict[str, float | None] = {
        "rms_centreline_free_jet": report["centreline"]["free_jet"]["rms"],
        "rms_centreline_impingement": report["centreline"]["impingement"]["rms"],
        "potential_core_w": report["potential_core_w"]["computed"],
        "decay_rate": report["decay_rate"]["computed"],
        "spreading_rate": half_width["spreading_rate"]["computed"],
        "rms_half_width_w": None
        if half_width["errors"] is None
        else half_width["errors"]["rms"],
        "stagnation_k_ratio_isotropic": report["stagnation"]["ratio_isotropic"],
        "stagnation_k_ratio_isotropic_ring": report["stagnation"][
            "ring_ratio_isotropic"
        ],
        "plate_y_star_median": report["y_star"]["median"],
    }
    return {
        "metric": IMPINGING_JET_METRIC,
        "value": report["centreline"]["all"]["rms"],
        "reference": IMPINGING_JET_REFERENCE,
        "components": components,
        "report": report,
        "passed": None,
    }


def product_accuracy(
    mesh: Mesh,
    config: SimConfig,
    outer_iterations: int,
    converged: bool,
    imbalance: np.ndarray,
    nu_t: np.ndarray,
) -> dict:
    """The product room's row: the stop within the cap, with the room's readings.

    Parameters
    ----------
    mesh : Mesh
        Mesh the fields belong to.
    config : SimConfig
        The product configuration, whose cap and imbalance tolerance the row
        reports beside the readings.
    outer_iterations : int
        Outer iterations the solve took.
    converged : bool
        Whether the solve stopped by its rule (``StaggeredSolver.converged``).
    imbalance : np.ndarray
        The last correction's per-cell mass imbalance, [ny, nx], the reading
        condition (b) compares with ``mass_imbalance_tol``.
    nu_t : np.ndarray
        The final eddy viscosity on the cells, [ny, nx].

    Returns
    -------
    dict
        ``metric``, ``value`` (the outer count over ``max_simple_iter``, so
        below 1 is a stop before the cap), ``reference`` (none; the criterion
        named), ``components`` (the outer count, the cap, the worst cell
        imbalance and its tolerance, the median of nu_t / nu over the core,
        the non-SOLID cells above the equipment tops) and ``passed``, the
        stop by the rule within the cap: ECR-002 criterion 10's convergence
        clause. Its comparative clause is the step 8 report's.
    """
    cap = config.max_simple_iter
    tops = max(obstacle.y_end for obstacle in config.obstacles)
    core = (mesh.cell_type != SOLID) & (np.asarray(mesh.yc)[:, None] > tops)
    nu = config.mu / config.rho
    return {
        "metric": PRODUCT_METRIC,
        "value": outer_iterations / cap,
        "reference": PRODUCT_REFERENCE,
        "components": {
            "outer_iterations": int(outer_iterations),
            "max_simple_iter": int(cap),
            "worst_cell_imbalance": float(np.max(np.abs(imbalance))),
            "mass_imbalance_tol": float(config.mass_imbalance_tol),
            "core_nu_t_over_nu_median": float(np.median(nu_t[core] / nu)),
        },
        "passed": bool(converged),
    }


def run_case(case_id: str, method: str, sample_every: int, concurrent: int) -> dict:
    """Run one case once with the solver the method names and return its record.

    Raises
    ------
    ValueError
        If the method is not one of METHODS. The two retired labels,
        collocated-jacobi and staggered-jacobi, are refused by name with
        where each solver went.
    """
    if method in RETIRED_METHODS:
        raise ValueError(f"method {method!r} {RETIRED_METHODS[method]}")
    if method not in METHOD_PRECONDITIONERS:
        raise ValueError(f"unknown method {method!r}; known: {list(METHODS)}")
    kind, nx, ny = CASES[case_id]
    if kind == CYLINDER_FAMILY:
        return run_cylinder_case(case_id, method, concurrent)
    config = with_pressure_preconditioner(
        load_preset(case_id), METHOD_PRECONDITIONERS[method]
    )
    solver_block = solver_parameters(config)
    mesh = Mesh(config)

    # One cell update = one stencil evaluation at one unknown, counted per
    # method as CELL_UPDATE_DEFINITIONS states. Comparable across Jacobi,
    # Krylov, multigrid, CPU and GPU.
    solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))
    momentum_updates, pressure_cells = staggered_updates(mesh, config)
    counter = WorkCounter(pressure_cells, momentum_updates)

    def error_of(u: np.ndarray, v: np.ndarray) -> float | None:
        # The Annex 20 lines and the step's walls are read from the faces,
        # which the callback lacks.
        if kind in FACE_SCORED_FAMILIES:
            return None
        return accuracy_of(kind, config, mesh, u, v)["value"]

    work = counter.work
    trajectory: list[dict] = []
    t_start = time.perf_counter()

    def observe(state: IterationState) -> None:
        counter.record(state)
        if state.iteration % sample_every == 0:
            trajectory.append(
                {
                    "outer_iteration": state.iteration,
                    "cell_updates": work["cell_updates"],
                    "elapsed_seconds": time.perf_counter() - t_start,
                    "residual": state.residual,
                    "error": error_of(state.u, state.v),
                }
            )

    u, v, p = solver.solve_steady(on_iteration=observe)
    wall = time.perf_counter() - t_start
    if kind == ANNEX20_FAMILY:
        assert solver.face_velocities is not None
        accuracy = annex20_accuracy(mesh, solver.face_velocities)
    elif kind == BACKSTEP_FAMILY:
        assert solver.face_velocities is not None
        assert solver.turbulence_state is not None
        accuracy = backstep_accuracy(
            mesh, config, solver.face_velocities, p, np.array(solver.turbulence_state.k)
        )
    elif kind == IMPINGING_JET_FAMILY:
        assert solver.face_velocities is not None
        assert solver.turbulence_state is not None
        accuracy = impinging_jet_accuracy(
            mesh, config, solver.face_velocities, np.array(solver.turbulence_state.k)
        )
    elif kind == PRODUCT_FAMILY:
        assert solver.turbulence_state is not None
        accuracy = product_accuracy(
            mesh,
            config,
            work["outer_iterations"],
            solver.converged,
            solver.last_mass_imbalance,
            np.array(solver.turbulence_state.nu_t),
        )
    else:
        accuracy = accuracy_of(kind, config, mesh, u, v)
    final_iter = work["outer_iterations"] - 1
    if not trajectory or trajectory[-1]["outer_iteration"] != final_iter:
        trajectory.append(
            {
                "outer_iteration": final_iter,
                "cell_updates": work["cell_updates"],
                "elapsed_seconds": wall,
                "residual": solver.residual_history[-1],
                "error": accuracy["value"],
            }
        )
    elif kind in FACE_SCORED_FAMILIES:
        # These families' callbacks carry no faces, so their samples hold no
        # error; the last takes the row's. Every other family's last sample
        # keeps its own, which the harness test compares with the row's
        # accuracy.
        trajectory[-1]["error"] = accuracy["value"]

    # The solver knows which rule it ran and whether the cap stopped it; a
    # velocity-step stop is stored under the label every stored row carries.
    # The count of pressure corrections that reached max_pressure_iter is
    # stored beside the stop, so a row cannot hide a truncated solve.
    converged = solver.converged
    stop_reason = STOP_LABELS.get(solver.stop_reason, solver.stop_reason)
    commit, dirty = git_state()
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": uuid.uuid4().hex,
        "recorded_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "method": method,
        "case": case_id,
        "grid": {"nx": nx, "ny": ny, **axis_clustering(mesh)},
        "params": solver_block,
        "environment": {
            "cpu": cpu_name(),
            "cpu_count": os.cpu_count(),
            "os": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "concurrent_processes": concurrent,
        },
        "outcome": {
            "converged": bool(converged),
            "stop_reason": stop_reason,
            "pressure_cap_hits": int(solver.pressure_cap_hits),
        },
        "accuracy": accuracy,
        "work": {**work, "cell_update_definition": CELL_UPDATE_DEFINITIONS[method]},
        "time": {"wall_seconds": wall, "stages": dict(solver.stage_seconds)},
        "trajectory": trajectory,
    }


def cylinder_accuracy(
    case_id: str, t: np.ndarray, c_l: np.ndarray, record: TransientRunRecord
) -> dict:
    """A VAL-022 row's reading and verdict by the preset's criterion.

    Parameters
    ----------
    case_id : str
        One of CYLINDER_PRESETS.
    t, c_l : np.ndarray
        The run's per-step times and lift coefficients.
    record : TransientRunRecord
        What ``time_integration.run`` returned.

    Returns
    -------
    dict
        ``metric``, ``reference``, ``value``, ``passed`` and the readings:
        the Strouhal number with its cross-check or refusal, the drag
        coefficient, and the amplitude measure.
    """
    reynolds = CYLINDER_PRESETS[case_id][0]
    criterion = cylinder.ROW_CRITERIA[case_id]
    reading = record.strouhal
    common = {
        "reference": CYLINDER_REFERENCE,
        "reynolds": reynolds,
        "strouhal": None if reading is None else reading.strouhal,
        "spectral_strouhal": None if reading is None else reading.spectral_strouhal,
        "spectral_resolution": None if reading is None else reading.resolution,
        "spectra_agree": None if reading is None else reading.agrees,
        "strouhal_refusal": record.strouhal_refusal,
        "drag_coefficient": record.drag_coefficient,
        "strouhal_band": list(cylinder.STROUHAL_BAND),
        "sohankar_drag": list(cylinder.SOHANKAR_DRAG),
    }
    if criterion == cylinder.DECAY:
        verdict = cylinder.decays(t, c_l)
        ratios = verdict.get("ratios")
        value = max(ratios) if ratios else verdict.get("largest_lift", float("nan"))
        return {
            **common,
            "metric": "lift_amplitude_decay",
            "value": value,
            "passed": verdict["passed"],
            "amplitude_measure": verdict,
        }
    if criterion == cylinder.SUSTAIN:
        verdict = cylinder.sustains(t, c_l)
        ratios = verdict.get("ratios")
        value = min(ratios) if ratios else float("nan")
        return {
            **common,
            "metric": "lift_amplitude_sustain",
            "value": value,
            "passed": verdict["passed"],
            "amplitude_measure": verdict,
        }
    st = float("nan") if reading is None else reading.strouhal
    return {
        **common,
        "metric": "strouhal_last_ten_periods",
        "value": st,
        "passed": None
        if criterion == cylinder.REPORT
        else cylinder.strouhal_in_band(None if reading is None else reading.strouhal),
    }


def run_cylinder_case(case_id: str, method: str, concurrent: int) -> dict:
    """Run one VAL-022 preset through tier 2 and return its row (ADR-014 F).

    The probe's perturbed start-up, then ``time_integration.run`` to the
    preset's end; the run file goes to results/val022/ and the per-step
    series to docs/reports/val022/<preset>.json with the row's reading.
    """
    config = with_pressure_preconditioner(
        load_preset(case_id), METHOD_PRECONDITIONERS[method]
    )
    mesh = Mesh(config)
    momentum_updates, pressure_cells = staggered_updates(mesh, config)
    counter = WorkCounter(pressure_cells, momentum_updates)
    t_start = time.perf_counter()
    initial, startup_outer = cylinder.perturbed_startup(config, counter.record)
    startup_seconds = time.perf_counter() - t_start
    commit, dirty = git_state()
    run_id = uuid.uuid4().hex
    VAL022_RUNS.mkdir(parents=True, exist_ok=True)
    path = VAL022_RUNS / f"{case_id}_{run_id[:8]}.h5"
    march = {"outer": 0, "products": 0}

    def count(step: TransientMarchStep) -> None:
        march["outer"] += step.record.outer
        march["products"] += step.record.pressure_products
        if step.record.step % 500 == 0:
            print(
                f"{case_id} step {step.record.step}: {step.record.outer} outer, "
                f"C_L {step.c_l:+.4f}, C_D {step.c_d:.4f}",
                flush=True,
            )

    record = run(
        config,
        path,
        on_step=count,
        provenance=Provenance.here(commit, concurrent),
        initial=initial,
    )
    wall = time.perf_counter() - t_start
    series = RunFile(path).read_step_series()
    accuracy = cylinder_accuracy(case_id, series.t, series.c_l, record)
    work = dict(counter.work)
    work["outer_iterations"] = startup_outer + march["outer"]
    work["inner_products"] += march["products"]
    work["cell_updates"] += (
        momentum_updates * march["outer"] + pressure_cells * march["products"]
    )
    row = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "recorded_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "method": method,
        "case": case_id,
        "grid": {"nx": config.nx, "ny": config.ny, **axis_clustering(mesh)},
        "params": {
            **solver_parameters(config),
            "transient": dataclasses.asdict(config.transient),
            "startup": {
                "outer": startup_outer,
                "bottom_factor": cylinder.STARTUP_BOTTOM_FACTOR,
            },
        },
        "environment": {
            "cpu": cpu_name(),
            "cpu_count": os.cpu_count(),
            "os": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "concurrent_processes": concurrent,
        },
        "outcome": {
            "converged": True,
            "stop_reason": "transient_t_end",
            "pressure_cap_hits": int(series.pressure_cap_hits.sum()),
        },
        "accuracy": accuracy,
        "work": {**work, "cell_update_definition": CELL_UPDATE_DEFINITIONS[method]},
        "time": {
            "wall_seconds": wall,
            "stages": {"startup": startup_seconds, "march": record.wall_seconds},
        },
        "trajectory": [],
    }
    VAL022_RECORDS.mkdir(parents=True, exist_ok=True)
    detail = {
        "row": {
            k: row[k]
            for k in (
                "run_id",
                "git_commit",
                "git_dirty",
                "case",
                "grid",
                "accuracy",
                "time",
                "work",
            )
        },
        "run_file": str(path.relative_to(REPO_ROOT)),
        "steps": record.steps,
        "t_final": record.t_final,
        "outer_mean": record.outer_mean,
        "outer_max": record.outer_max,
        "concurrent_processes": concurrent,
        "series": {
            "step": series.step.tolist(),
            "t": series.t.tolist(),
            "outer": series.outer.tolist(),
            "worst_imbalance": series.worst_imbalance.tolist(),
            "c_d": series.c_d.tolist(),
            "c_l": series.c_l.tolist(),
        },
    }
    (VAL022_RECORDS / f"{case_id}.json").write_text(
        json.dumps(detail, indent=1), encoding="utf-8"
    )
    return row


def print_summary(path: Path) -> None:
    """Print one row per method, case, concurrency, rule, metric and reference; flag mixed.

    Wall time is only comparable between runs that shared the machine with
    the same number of processes, so ``concurrent_processes`` is part of the
    grouping key and shown as a column. Accuracy, outer iterations and cell
    updates are deterministic and do not depend on load. When the table
    holds more than one load a note says so, and any single case recorded
    under more than one load is named: the 80x80 cavity row was taken with
    two processes where the 40x40 rows used one, and nothing noticed until a
    reviewer read the raw file.

    An error is only comparable with errors against the same reference, so
    the accuracy reference is part of the row key too and shown last, and a
    case scored against more than one reference is named. The rows stored
    before revision r2 of the Ghia table carry ``ghia_1982_re100`` and would
    otherwise be pooled into one error range with the corrected rows.

    The metric is part of the key for the same reason. The cavity rows stored
    as ``max_normalized_centerline_error`` sample half a cell off the
    centerlines; ``max_normalized_centerline_error_r2`` samples on them, and
    ``max_normalized_centerline_error_cubic`` reads the same profiles by the
    cubic at Marchi's stations.

    So is the stopping rule, ``params.stopping_rule``, shown after the case:
    it sets how far each solve iterates, so outer counts, wall times and
    errors compare only within one rule. A row without the key predates it
    and ran velocity_step. Under error_estimate so is the rule's version: an
    added condition changes no parameter.
    """
    if not path.exists():
        print(f"{path} does not exist; nothing recorded yet.")
        return
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    groups: dict[tuple[str, str, int], list[dict]] = {}
    rows: dict[tuple[str, str, int, str, str, str, str], list[dict]] = {}
    for record in records:
        key = (
            record["method"],
            record["case"],
            record["environment"]["concurrent_processes"],
        )
        groups.setdefault(key, []).append(record)
        # Every harness row names its metric and reference; a hand-built record may not.
        metric = record["accuracy"].get("metric", "-")
        reference = record["accuracy"].get("reference", "-")
        params = record.get("params", {})
        rule = params.get("stopping_rule", VELOCITY_STEP)
        # error_estimate rows from before rule_version (0ca4ace, f3561ba) ran version 2.
        version = str(params.get("rule_version", 2)) if rule == ERROR_ESTIMATE else "-"
        rows.setdefault((*key, rule, version, metric, reference), []).append(record)

    header = (
        f"{'method':<20} {'case':<22} {'rule':<14} {'ver':>3} {'procs':>5} {'n':>2} "
        f"{'outer':>10} {'wall s (min/med/max)':>24} {'cell updates':>14} "
        f"{'error (min..max)':>20} {'conv':>5} {'metric':<34} reference"
    )
    print(header)
    print("-" * len(header))
    for (method, case, procs, rule, version, metric, reference), runs in sorted(
        rows.items()
    ):
        outer = [r["work"]["outer_iterations"] for r in runs]
        wall = [r["time"]["wall_seconds"] for r in runs]
        updates = [r["work"]["cell_updates"] for r in runs]
        error = [r["accuracy"]["value"] for r in runs]
        conv = sum(r["outcome"]["converged"] for r in runs)
        print(
            f"{method:<20} {case:<22} {rule:<14} {version:>3} {procs:>5} {len(runs):>2} "
            f"{min(outer):>4}..{max(outer):<4} "
            f"{min(wall):>7.1f}/{statistics.median(wall):>7.1f}/{max(wall):>7.1f} "
            f"{statistics.median(updates):>14.3e} "
            f"{min(error):>9.3e}..{max(error):<9.3e} {conv:>2}/{len(runs)} "
            f"{metric:<34} {reference}"
        )

    references: dict[tuple[str, str], set[str]] = {}
    metrics: dict[tuple[str, str], set[str]] = {}
    rules: dict[tuple[str, str], set[str]] = {}
    versions: dict[tuple[str, str], set[str]] = {}
    for method, case, _procs, rule, version, metric, reference in rows:
        references.setdefault((method, case), set()).add(reference)
        metrics.setdefault((method, case), set()).add(metric)
        rules.setdefault((method, case), set()).add(rule)
        if rule == ERROR_ESTIMATE:
            versions.setdefault((method, case), set()).add(version)
    for (method, case), seen in sorted(references.items()):
        if len(seen) > 1:
            print(
                f"note: {method} {case} has rows scored against {sorted(seen)}; "
                "errors are comparable only within one reference"
            )
    for (method, case), seen in sorted(metrics.items()):
        if len(seen) > 1:
            print(
                f"note: {method} {case} has rows measured by {sorted(seen)}; "
                "errors are comparable only within one metric"
            )
    for (method, case), seen in sorted(rules.items()):
        if len(seen) > 1:
            print(
                f"note: {method} {case} has rows under {sorted(seen)}; outer "
                "iterations, wall times and errors are comparable only within one rule"
            )
    for (method, case), seen in sorted(versions.items()):
        if len(seen) > 1:
            print(
                f"note: {method} {case} has error_estimate rows under rule versions "
                f"{sorted(seen)}; compare them only within one version"
            )

    loads_seen = sorted({procs for _, _, procs in groups})
    if len(loads_seen) > 1:
        print(
            f"note: rows above were recorded at concurrent_processes {loads_seen}; "
            "compare wall times only within one value. Accuracy, outer iterations "
            "and cell updates do not depend on load."
        )
    for method, case in sorted(mixed_load_cases(groups)):
        loads = sorted(p for m, c, p in groups if (m, c) == (method, case))
        print(
            f"note: {method} {case} has rows at concurrent_processes "
            f"{loads}; wall times are not comparable across them"
        )


def mixed_load_cases(
    groups: dict[tuple[str, str, int], list[dict]],
) -> set[tuple[str, str]]:
    """Return the (method, case) pairs recorded under more than one load."""
    loads: dict[tuple[str, str], set[int]] = {}
    for method, case, procs in groups:
        loads.setdefault((method, case), set()).add(procs)
    return {key for key, seen in loads.items() if len(seen) > 1}


def main(argv: list[str] | None = None) -> int:
    """Run the selected cases and append one record each, or print the summary.

    Parameters
    ----------
    argv : list[str] | None
        Command line arguments. None reads sys.argv.

    Returns
    -------
    int
        Process exit code, 0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--cases", nargs="+", choices=sorted(CASES), default=DEFAULT_CASES
    )
    parser.add_argument(
        "--repeats", type=positive_int, default=3, help="Runs per case, at least 1"
    )
    parser.add_argument(
        "--method",
        choices=METHODS,
        default=DEFAULT_METHOD,
        help="Solver to run; the label is recorded with each row",
    )
    parser.add_argument(
        "--sample-every",
        type=positive_int,
        default=10,
        help="Trajectory sample interval in outer iterations, at least 1",
    )
    parser.add_argument(
        "--concurrent",
        type=positive_int,
        default=1,
        help="Processes this run shared the machine with, itself included, at least 1",
    )
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--summary", action="store_true", help="Print stored results")
    args = parser.parse_args(argv)

    if args.summary:
        print_summary(args.results)
        return 0

    args.results.parent.mkdir(parents=True, exist_ok=True)
    for case_id in args.cases:
        for repeat in range(args.repeats):
            record = run_case(case_id, args.method, args.sample_every, args.concurrent)
            with args.results.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"{case_id} repeat {repeat + 1}/{args.repeats}: "
                f"{record['work']['outer_iterations']} outer, "
                f"{record['time']['wall_seconds']:.1f} s, "
                f"error {record['accuracy']['value']:.3e}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
