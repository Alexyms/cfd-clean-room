"""Probe for prompt 95: the stopping rule and the outer iteration in the core.

    python docs/reports/probe95/probe95.py capture product80
    python docs/reports/probe95/probe95.py capture product160
    python docs/reports/probe95/probe95.py capture cavity40
    python docs/reports/probe95/probe95.py capture door80
    python docs/reports/probe95/probe95.py solve product80
    python docs/reports/probe95/probe95.py solve product160
    python docs/reports/probe95/probe95.py solve cavity40
    python docs/reports/probe95/probe95.py solve door80
    python docs/reports/probe95/probe95.py solve poiseuille40
    python docs/reports/probe95/probe95.py timing --run 1
    python docs/reports/probe95/probe95.py vecinfo

Run under WSL2 with the Linux environment (``~/.venvs/cfd_clean_room``) from the
checkout's root, one BLAS thread, NumPy's AVX-512 loops off
(``NPY_DISABLE_CPU_FEATURES=X86_V4``, Alex for prompt 95: every parity run, the
reference's whole solves included). Nothing in ``src/`` changes: the probe wraps
the reference's methods on their classes and module, and each wrapper calls the
reference and compares what the core's block gives on the same inputs.

- ``capture``: the reference's own solve of a case to its stop. Every call of a
  block of the first kind is replayed through the core as it happens, on the
  reference's inputs, and compared bit for bit: the outlets' extrapolation, the
  effective viscosity, the eddy viscosity's relaxation, the cell centres, the two
  max-norms (the velocity step, the viscosity step) and the residual, the median
  (``viscosity_scale``), the imbalance summary, and the rule's verdict. Each outer
  iteration's steps, readings and both sides' estimates are kept, the core's
  formed by a rule fed the reference's steps (so on the reference's window), with
  the margin of each to ``iteration_error_tol``. The comparisons cover every
  outer iteration; the record names the first 50 and the 300 before the stop.
- ``solve``: the step's gate. The case from rest (the door leak's open state
  warm-started from its closed one, each side from its own), the reference's
  ``solve_steady`` and the core's ``CoreSolver`` in one process: each one's stop
  reason and outer count, each flow field's largest difference over its largest
  magnitude with its tier (ADR-017 H), and the corrections' iteration counts.
- ``timing``: the 160x60 product room's solve from rest, the reference then the
  core in one process, with the time in each stage and the machine's state.
- ``vecinfo``: GCC's vectorisation report on the new blocks.

The cases: the 80x30 and 160x60 product room (``configs/clean_room_default.yaml``
at those grids through the dust release's scenario file, its tolerance by the
base file's rule, as prompt 85's probe builds them), the 40x40 cavity
(``configs/validation_cavity.yaml``), and the door leak's two states on 80x30.
Round 1 (prompt 95b) adds a case that is solved and not captured: VAL-001's
channel at 40x20 (``configs/validation_poiseuille.yaml``), the one whole solve here
with a pressure outlet, which no other case has. The records are in
``results/builder95/`` (not committed).
"""

import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "probe85"))
sys.path.insert(0, str(HERE.parent / "probe90"))

# Prompt 85's probe: its scenario files at a grid; 90's: its timer and the
# machine's state. Their commands are not used.
import probe85  # noqa: E402 -- after sys.path
import probe90  # noqa: E402 -- after sys.path

import src.solver_staggered as solver_module  # noqa: E402 -- after sys.path
from src.boundary_staggered import StaggeredBoundary  # noqa: E402
from src.config import SimConfig  # noqa: E402
from src.mesh import FLUID, SOLID, Mesh  # noqa: E402
from src.pressure import ZERO_SCALE  # noqa: E402
from src.scenarios import ScenarioManager  # noqa: E402
from src.solver_staggered import FlowState, StaggeredSolver  # noqa: E402
from src.stopping import ErrorEstimateRule  # noqa: E402
from src.turbulence import KEpsilonModel  # noqa: E402
from tests import core_mapping  # noqa: E402
from tests.core_driver import CoreSolver  # noqa: E402

RESULTS = ROOT / "results" / "builder95"
CAPTURES = RESULTS / "captures"
SOLVES = RESULTS / "solves"
CASES = ("product80", "product160", "cavity40", "door80")
# Solved and compared, not captured: the channel's pressure outlet (prompt 95b).
SOLVE_ONLY = ("poiseuille40",)
# The iterations the prompt names: the first FIRST and the LAST before the stop.
FIRST = 50
LAST = 300
FLOW_FIELDS = ("u", "v", "p", "k", "eps", "nu_t")
# ADR-017 H's level-2 limit on a flow field, over its largest magnitude.
FLOW_LIMIT = 1e-8


# ---------------------------------------------------------------------------
# The cases


def head_commit() -> str:
    """HEAD's hash read from the checkout's files, or "unknown".

    Git runs from Windows only (ADR-017 A), so the Linux side reads the hash
    instead of asking git for it. Each record carries it, so that the tables
    can refuse records of two commits.
    """
    git = ROOT / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[len("ref: ") :]
        loose = git / ref
        if loose.exists():
            return loose.read_text().strip()
        for line in (git / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except OSError:
        pass
    return "unknown"


def scenario_config(scenario: str, grid: str) -> SimConfig:
    """Main's scenario file at a grid with the base file's tolerance, as prompt 85 builds it."""
    mapping, _changed = probe85.variant_mapping("reference", scenario, grid)
    return SimConfig.from_dict(mapping, ROOT / "configs")


def flow_configs(name: str) -> list[SimConfig]:
    """A case's flow states' configurations, in the order a run first uses them."""
    if name == "cavity40":
        path = ROOT / "configs" / "validation_cavity.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return [SimConfig.from_dict(raw, path.parent)]
    if name == "poiseuille40":
        path = ROOT / "configs" / "validation_poiseuille.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw["domain"].update(nx=40, ny=20)
        return [SimConfig.from_dict(raw, path.parent)]
    scenario, grid = {
        "product80": ("dust", "80x30"),
        "product160": ("dust", "160x60"),
        "door80": ("door", "80x30"),
    }[name]
    config = scenario_config(scenario, grid)
    manager = ScenarioManager(config, Mesh(config))
    seen: dict[str, SimConfig] = {}
    for interval in manager.intervals():
        seen.setdefault(interval.fingerprint, interval.config)
    return list(seen.values())


def solve_case(
    kind: type, name: str, keep: Callable[[str, StaggeredSolver], None]
) -> None:
    """A case's solves on one side: each state from rest or warm from the one before."""
    previous: FlowState | None = None
    for index, config in enumerate(flow_configs(name)):
        mesh = Mesh(config)
        solver = kind(mesh, config, StaggeredBoundary(mesh, config))
        with np.errstate(all="ignore"):
            solver.solve_steady(initial=previous)
        keep(f"state{index}", solver)
        previous = solver.flow_state


# ---------------------------------------------------------------------------
# Capture


def bits_equal(a: object, b: object) -> bool:
    """Whether two floats or arrays are equal bit for bit."""
    x = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    y = np.ascontiguousarray(np.asarray(b, dtype=np.float64))
    return x.shape == y.shape and bool(
        np.array_equal(x.view(np.uint64), y.view(np.uint64))
    )


def relative(a: float, b: float) -> float:
    """|a - b| / |b|, zero for equal values, infinities included."""
    if a == b:
        return 0.0
    return abs(a - b) / abs(b)


class Capture:
    """The wrappers on the reference's blocks, each comparing the core's on its inputs."""

    def __init__(self) -> None:
        self.blocks: dict[str, list[int]] = {}
        self.rows: list[dict[str, object]] = []
        self.shadows: dict[int, object] = {}
        self.centres: list[tuple[np.ndarray, np.ndarray]] = []
        self.solver: StaggeredSolver | None = None
        self.grid: object = None
        self.outlets: object = None
        self.stepped: object = None
        self.pending: dict[str, object] = {}

    def block(self, name: str, iteration: int, same: bool) -> None:
        """One comparison: the iterations where the block differs are kept."""
        counts = self.blocks.setdefault(name, [0, 0])
        counts[0] += 1
        if not same:
            counts[1] += 1
            self.pending.setdefault("differs", []).append(name)  # type: ignore[union-attr]

    def iteration(self) -> int:
        return len(self.solver.residual_history) if self.solver is not None else -1

    def bind(self, solver: StaggeredSolver) -> None:
        """The solver whose calls follow, and the core's blocks on its grid."""
        self.solver = solver
        self.grid = core_mapping.core_grid(solver._mesh)
        self.outlets = core_mapping.outlet_faces(
            solver._mesh, solver._boundary, self.grid
        )
        self.centres = []

    @contextlib.contextmanager
    def installed(self) -> Iterator[None]:
        import _cfdcore

        capture = self
        originals = {
            "outlets": StaggeredSolver._extrapolate_outlets,
            "effective": StaggeredSolver._effective_viscosity,
            "turbulence": StaggeredSolver._turbulence_step,
            "summary": StaggeredSolver._imbalance_summary,
            "update": ErrorEstimateRule.update,
            "step": KEpsilonModel.step,
            "centres": solver_module.to_cell_centers,
            "median": solver_module.viscosity_scale,
        }

        def outlets(self: StaggeredSolver, u: np.ndarray, v: np.ndarray) -> None:
            before = (np.array(u), np.array(v))
            originals["outlets"](self, u, v)
            got = capture.outlets.extrapolate(*before)
            capture.block(
                "outlets",
                capture.iteration(),
                bits_equal(got[0], u) and bits_equal(got[1], v),
            )

        def effective(self: StaggeredSolver, field: np.ndarray) -> np.ndarray:
            out = originals["effective"](self, field)
            live = self._mesh.cell_type != SOLID
            got = _cfdcore.effective_viscosity(field, live, self._mu, self._rho)
            capture.block(
                "effective_viscosity", capture.iteration(), bits_equal(got, out)
            )
            return out

        def step(model: KEpsilonModel, *args: object, **keys: object) -> object:
            out = originals["step"](model, *args, **keys)
            capture.stepped = out
            return out

        def turbulence(
            self: StaggeredSolver,
            state: object,
            u: np.ndarray,
            v: np.ndarray,
            iteration: int,
        ) -> object:
            out = originals["turbulence"](self, state, u, v, iteration)
            got = _cfdcore.relax_eddy_viscosity(
                state.nu_t, capture.stepped.nu_t, self._alpha_turbulence
            )
            capture.block("relaxation", iteration, bits_equal(got, out.nu_t))
            live = self._mesh.cell_type != SOLID
            capture.pending["viscosity_step"] = _cfdcore.largest_change(
                out.nu_t, state.nu_t, live
            )
            return out

        def centres(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            out = originals["centres"](u, v)
            if (
                capture.grid is not None
                and u.shape[0] == capture.solver._mesh.cell_type.shape[0]
            ):
                got = _cfdcore.cell_centres(capture.grid, u, v)
                capture.block(
                    "cell_centres",
                    capture.iteration(),
                    bits_equal(got[0], out[0]) and bits_equal(got[1], out[1]),
                )
                capture.centres.append(out)
                capture.centres = capture.centres[-2:]
                if len(capture.centres) == 2:
                    fluid = capture.solver._mesh.cell_type == FLUID
                    (u_prev, v_prev), (u_c, v_c) = capture.centres
                    du = _cfdcore.largest_change(u_c, u_prev, fluid)
                    dv = _cfdcore.largest_change(v_c, v_prev, fluid)
                    capture.pending["step"] = max(du, dv)
            return out

        def median(nu: float, nu_t: np.ndarray, fluid: np.ndarray) -> float:
            out = originals["median"](nu, nu_t, fluid)
            got = _cfdcore.viscosity_scale(nu, nu_t, fluid)
            capture.block("median", capture.iteration(), bits_equal(got, out))
            return out

        def summary(self: StaggeredSolver, u: np.ndarray, v: np.ndarray) -> object:
            out = originals["summary"](self, u, v)
            got = _cfdcore.imbalance_summary(self._corrector.mass_imbalance(u, v))
            same = all(
                bits_equal(a, b)
                for a, b in zip(
                    got, (out.worst, out.absolute_sum, out.signed_sum), strict=True
                )
            )
            capture.block("imbalance_summary", capture.iteration(), same)
            return out

        def update(
            rule: ErrorEstimateRule,
            step: float,
            imbalance: Callable,
            viscosity_step: float | None = None,
            nu_scale: float | None = None,
        ) -> bool:
            shadow = capture.shadows.get(id(rule))
            if shadow is None:
                shadow = core_mapping.error_estimate_rule(
                    rule._scale,
                    rule._flux_scale,
                    rule._error_tol,
                    rule._imbalance_tol,
                    viscosity_condition=rule._viscosity_condition,
                )
                capture.shadows[id(rule)] = shadow
            cached: list[object] = []
            ref_calls = [0]

            def reading() -> object:
                ref_calls[0] += 1
                found = imbalance()
                cached.append(found)
                return found

            keys = (
                {}
                if viscosity_step is None
                else {"viscosity_step": viscosity_step, "nu_scale": nu_scale}
            )
            verdict = originals["update"](rule, step, reading, **keys)
            core_calls = [0]

            def shadow_reading() -> object:
                core_calls[0] += 1
                return cached[0] if cached else imbalance()

            core_verdict = shadow.update(step, shadow_reading, **keys)
            # The residual of this outer iteration is in the history by now.
            iteration = capture.iteration() - 1
            pending = capture.pending
            # The step and the residual the reference formed against the core's max-norms.
            residual_scale = max(capture.solver.reference_velocity, ZERO_SCALE)
            capture.block(
                "velocity_step",
                iteration,
                bits_equal(pending.get("step", math.nan), step),
            )
            capture.block(
                "residual",
                iteration,
                bits_equal(
                    pending.get("step", math.nan) / residual_scale,
                    step / residual_scale,
                ),
            )
            if viscosity_step is not None:
                capture.block(
                    "viscosity_step",
                    iteration,
                    bits_equal(pending.get("viscosity_step", math.nan), viscosity_step),
                )
            capture.block(
                "rule_verdict",
                iteration,
                verdict == core_verdict and ref_calls[0] == core_calls[0],
            )
            tol = rule._error_tol
            row: dict[str, object] = {
                "iteration": iteration,
                "step": step,
                "estimate": [rule.estimate_history[-1], shadow.estimate_history[-1]],
                "verdict": [verdict, core_verdict],
                "imbalance_read": [ref_calls[0], core_calls[0]],
                "differs": list(pending.get("differs", [])),
            }
            if viscosity_step is not None:
                row["viscosity_step"] = viscosity_step
                row["nu_scale"] = nu_scale
                row["viscosity_estimate"] = [
                    rule.viscosity_estimate_history[-1],
                    shadow.viscosity_estimate_history[-1],
                ]
            row["tol"] = tol
            capture.rows.append(row)
            capture.pending = {}
            return verdict

        patches = [
            (StaggeredSolver, "_extrapolate_outlets", outlets),
            (StaggeredSolver, "_effective_viscosity", effective),
            (StaggeredSolver, "_turbulence_step", turbulence),
            (StaggeredSolver, "_imbalance_summary", summary),
            (ErrorEstimateRule, "update", update),
            (KEpsilonModel, "step", step),
            (solver_module, "to_cell_centers", centres),
            (solver_module, "viscosity_scale", median),
        ]
        saved = [(owner, name, getattr(owner, name)) for owner, name, _ in patches]
        try:
            for owner, name, value in patches:
                setattr(owner, name, value)
            yield
        finally:
            for owner, name, value in saved:
                setattr(owner, name, value)


def estimate_rows(rows: list[dict[str, object]], key: str) -> dict[str, object]:
    """Each estimate's relative difference, and the windows on opposite sides of the tolerance."""
    worst = 0.0
    worst_at = None
    opposite = []
    finite = 0
    for row in rows:
        if key not in row:
            continue
        ref, core = row[key]  # type: ignore[misc]
        tol = row["tol"]
        if math.isinf(ref) != math.isinf(core):
            opposite.append(row["iteration"])
            continue
        if math.isinf(ref):
            continue
        finite += 1
        d = relative(core, ref)
        if d > worst:
            worst, worst_at = d, row["iteration"]
        if (ref < tol) != (core < tol):  # type: ignore[operator]
            opposite.append(row["iteration"])
    return {
        "finite": finite,
        "largest_relative": worst,
        "at": worst_at,
        "opposite_sides": opposite,
    }


def capture(name: str) -> None:
    """The reference's own solves of a case, every first-kind block replayed through the core."""
    CAPTURES.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {
        "case": name,
        "commit": head_commit(),
        "states": [],
    }
    started = time.perf_counter()
    previous: FlowState | None = None
    for index, config in enumerate(flow_configs(name)):
        mesh = Mesh(config)
        solver = StaggeredSolver(mesh, config, StaggeredBoundary(mesh, config))
        kept = Capture()
        kept.bind(solver)
        with kept.installed(), np.errstate(all="ignore"):
            solver.solve_steady(initial=previous)
        previous = solver.flow_state
        rows = kept.rows
        n = len(rows)
        first = [r for r in rows if r["iteration"] < FIRST]
        last = [r for r in rows if r["iteration"] >= n - LAST]

        def differing(subset: list[dict[str, object]]) -> int:
            return sum(bool(r["differs"]) for r in subset)

        stop = rows[-1]
        margins = {"estimate": [e / stop["tol"] for e in stop["estimate"]]}  # type: ignore[union-attr]
        if "viscosity_estimate" in stop:
            margins["viscosity_estimate"] = [
                e / stop["tol"] for e in stop["viscosity_estimate"]
            ]  # type: ignore[union-attr]
        state = {
            "index": index,
            "origin": "rest" if index == 0 else "warm",
            "stop_reason": solver.stop_reason,
            "outer": len(solver.residual_history),
            "blocks": kept.blocks,
            "iterations_with_a_difference": {
                "first": differing(first),
                "last": differing(last),
                "all": differing(rows),
            },
            "verdicts_differ": sum(r["verdict"][0] != r["verdict"][1] for r in rows),  # type: ignore[index]
            "estimate": estimate_rows(rows, "estimate"),
            "viscosity_estimate": estimate_rows(rows, "viscosity_estimate"),
            "margins_at_stop": margins,
            "first_count": len(first),
            "last_count": len(last),
            "rows": rows,
        }
        record["states"].append(state)  # type: ignore[union-attr]
        print(
            name,
            index,
            solver.stop_reason,
            state["outer"],
            {k: v for k, v in kept.blocks.items()},
            "estimate",
            state["estimate"]["largest_relative"],
            state["estimate"]["opposite_sides"],
            "nu",
            state["viscosity_estimate"]["largest_relative"],
            state["viscosity_estimate"]["opposite_sides"],
            "margins",
            margins,
            flush=True,
        )
    record["seconds"] = time.perf_counter() - started
    record["loops"] = numpy_loops()
    (CAPTURES / f"{name}.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )


def numpy_loops() -> dict[str, str]:
    """The loops NumPy runs for a float64 log and power in this process."""
    from numpy.lib import introspect

    info = introspect.opt_func_info(func_name="^(log|power)$", signature="float64")
    return {
        key: "; ".join(loop["current"] for loop in loops.values())
        for key, loops in info.items()
    }


# ---------------------------------------------------------------------------
# Whole solves


def spreads() -> dict[str, float]:
    """Each flow field's measured spread: the largest P2 of prompt 85's probe (tables85.md)."""
    text = (HERE.parent / "probe85" / "tables85.md").read_text(encoding="utf-8")
    lines = text.split("### P2.")[1].split("\n\n")[1].splitlines()
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    out = dict.fromkeys(FLOW_FIELDS, 0.0)
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        for field in FLOW_FIELDS:
            value = cells[header.index(field)]
            out[field] = max(out[field], float(value))
    return out


def solve(name: str) -> None:
    """A case's solves on both sides in one process: the gate's comparison."""
    SOLVES.mkdir(parents=True, exist_ok=True)
    sides: dict[str, dict[str, StaggeredSolver]] = {"reference": {}, "core": {}}
    counts: dict[str, dict[str, list[int]]] = {"reference": {}, "core": {}}
    seconds: dict[str, dict[str, float]] = {"reference": {}, "core": {}}
    for side, kind in (("reference", StaggeredSolver), ("core", CoreSolver)):
        previous: FlowState | None = None
        for index, config in enumerate(flow_configs(name)):
            key = f"state{index}"
            mesh = Mesh(config)
            solver = kind(mesh, config, StaggeredBoundary(mesh, config))
            iterations: list[int] = []
            started = time.perf_counter()
            with np.errstate(all="ignore"):
                if side == "reference":
                    solver.solve_steady(
                        lambda s, kept=iterations: kept.append(s.pressure_iterations),
                        initial=previous,
                    )
                else:
                    solver.solve_steady(initial=previous)
                    iterations = [
                        int(n) for n in solver.rule_readings["pressure_iterations"]
                    ]
            seconds[side][key] = time.perf_counter() - started
            sides[side][key] = solver
            counts[side][key] = iterations
            previous = solver.flow_state
            print(
                side,
                name,
                key,
                solver.stop_reason,
                len(solver.residual_history),
                f"{seconds[side][key]:.1f}s",
                flush=True,
            )
    measured = spreads()
    states = []
    for key, ref in sides["reference"].items():
        core = sides["core"][key]
        fields = {}
        for field in FLOW_FIELDS:
            expected = (
                getattr(ref.flow_state, field) if ref.flow_state is not None else None
            )
            got = (
                getattr(core.flow_state, field) if core.flow_state is not None else None
            )
            if expected is None or got is None:
                continue
            scale = float(np.max(np.abs(expected)))
            difference = float(np.max(np.abs(got - expected))) / scale
            tier = (
                "inside the spread"
                if difference <= measured[field]
                else (
                    "between the spread and the limit"
                    if difference <= FLOW_LIMIT
                    else "beyond the limit"
                )
            )
            fields[field] = {
                "difference": difference,
                "spread": measured[field],
                "tier": tier,
            }
        a, b = counts["reference"][key], counts["core"][key]
        common = min(len(a), len(b))
        differ = [i for i in range(common) if a[i] != b[i]]
        states.append(
            {
                "state": key,
                "stop_reason": [ref.stop_reason, core.stop_reason],
                "outer": [len(ref.residual_history), len(core.residual_history)],
                "fields": fields,
                "pressure_iterations": [int(sum(a)), int(sum(b))],
                "counts_differ": len(differ),
                "first_count_differs": differ[0] if differ else None,
                "cap_hits": [ref.pressure_cap_hits, core.pressure_cap_hits],
                "seconds": [seconds["reference"][key], seconds["core"][key]],
                "residual_last": [ref.residual_history[-1], core.residual_history[-1]],
            }
        )
        print(
            name,
            key,
            json.dumps(states[-1]["fields"]),
            "counts differ",
            len(differ),
            flush=True,
        )
    record = {
        "case": name,
        "commit": head_commit(),
        "loops": numpy_loops(),
        "states": states,
    }
    (SOLVES / f"{name}.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Timing


def timing(run_number: int) -> None:
    """The 160x60 product room's solve from rest, the reference then the core, in one process."""
    before = probe90.windows_load()
    calibration = probe90.calibration()
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    (config,) = flow_configs("product160")
    mesh = Mesh(config)
    record: dict[str, object] = {
        "run": run_number,
        "commit": head_commit(),
        "started": started,
        "loops": numpy_loops(),
    }
    for side, kind in (("reference", StaggeredSolver), ("core", CoreSolver)):
        solver = kind(mesh, config, StaggeredBoundary(mesh, config))
        t0 = time.perf_counter()
        solver.solve_steady()
        wall = time.perf_counter() - t0
        record[side] = {
            "wall_seconds": wall,
            "outer": len(solver.residual_history),
            "stop_reason": solver.stop_reason,
            "stage_seconds": dict(solver.stage_seconds),
        }
        if side == "core":
            record[side]["core_seconds"] = dict(solver.core_seconds)  # type: ignore[index]
        print(
            side,
            f"{wall:.2f} s",
            len(solver.residual_history),
            solver.stage_seconds,
            flush=True,
        )
    ratio = record["reference"]["wall_seconds"] / record["core"]["wall_seconds"]  # type: ignore[index]
    record["ratio"] = ratio
    record["calibration_before"] = calibration
    record["calibration_after"] = probe90.calibration()
    record["windows_before"] = before
    record["windows_after"] = probe90.windows_load()
    (RESULTS / f"timing_{run_number}.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )
    print(f"run {run_number}: ratio {ratio:.2f}")


# ---------------------------------------------------------------------------
# The vectorisation report

VEC_FLAGS = ["-std=c++20", "-O2", "-ffp-contract=off", "-march=x86-64-v3", "-DNDEBUG"]
# Each new block: its file and the text that starts its loop's function.
VEC_BLOCKS = {
    "cell centres": ("flow", "CFDCORE_INLINE void centre_cells("),
    "largest change": ("flow", "CFDCORE_INLINE void change_cells("),
    "effective viscosity": ("flow", "void effective_cells("),
    "eddy viscosity's checks": ("flow", "CFDCORE_INLINE void check_cells("),
    "imbalance magnitude and maximum": (
        "stopping",
        "CFDCORE_INLINE void magnitude_cells(",
    ),
    "relaxation": ("turbulence", "void relaxed_cells("),
}


def vecinfo() -> None:
    """Which of the new blocks GCC vectorises, its loops counted by the block's lines."""
    out: dict[str, object] = {}
    for label, (name, start) in VEC_BLOCKS.items():
        source = ROOT / "core" / "src" / f"{name}.cpp"
        lines = source.read_text(encoding="utf-8").splitlines()
        first = next(i for i, line in enumerate(lines) if start in line) + 1
        last = next(i for i in range(first, len(lines)) if lines[i].startswith("}")) + 1
        report = subprocess.run(
            [
                "g++",
                *VEC_FLAGS,
                f"-I{ROOT / 'core' / 'include'}",
                "-fopt-info-vec-optimized",
                "-c",
                str(source),
                "-o",
                os.devnull,
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stderr
        loops = sum(
            1
            for line in report.splitlines()
            if "loop vectorized" in line
            and f"{name}.cpp:" in line
            and first <= int(line.split(f"{name}.cpp:")[1].split(":")[0]) <= last
        )
        out[label] = {
            "file": f"core/src/{name}.cpp",
            "lines": [first, last],
            "vectorised_loops": loops,
        }
        print(label, out[label])
    (RESULTS / "vecinfo.json").write_text(
        json.dumps({"commit": head_commit(), "blocks": out}, indent=1) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Planted defects

PYTHON = Path.home() / ".venvs" / "cfd_clean_room" / "bin" / "python"


def run_command(command: list[str], cwd: Path, env: dict[str, str]) -> tuple[int, str]:
    """A command's exit status and the tail of its output."""
    done = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    return done.returncode, (done.stdout + done.stderr)[-4000:]


def failed_tests(output: str) -> list[str]:
    """The tests a pytest or CTest output names as failing."""
    names = []
    for line in output.splitlines():
        if line.startswith("FAILED ") or line.startswith("ERROR "):
            names.append(line.split(" ")[1].split(" - ")[0])
        elif "***Failed" in line or "(Failed)" in line:
            names.append(line.strip())
    return names


def plants(
    stream: int, streams: int, only: list[str] | None = None, reverse: bool = False
) -> None:
    """Each planted defect of variants95.py built in this stream's tree, the nets run on it."""
    import variants95

    tree = Path.home() / "scratch95" / f"s{stream}"
    build = Path.home() / "build" / f"cfd-plant95-s{stream}"
    env = dict(os.environ)
    env.update(
        PYTHONPATH=f"{build}:{tree}",
        CFD_CORE_REQUIRED="1",
        NPY_DISABLE_CPU_FEATURES="X86_V4",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    configured, text = run_command(
        [
            "cmake",
            "-S",
            str(tree / "core"),
            "-B",
            str(build),
            "-G",
            "Ninja",
            f"-DPython_EXECUTABLE={PYTHON}",
            "-DCFD_CORE_WERROR=ON",
        ],
        tree,
        env,
    )
    if configured != 0:
        raise SystemExit(text)
    share = [p for i, p in enumerate(variants95.PLANTS) if i % streams == stream]
    if only:
        share = [p for p in variants95.PLANTS if p[0] in only]
    if reverse:
        # Round 1's plants come last in the list and are the ones to learn of first.
        share = share[::-1]
    records = []
    for name, description, path, pairs in share:
        source = tree / path
        original = source.read_text(encoding="utf-8")
        planted = original
        for old, new in pairs:
            assert planted.count(old) == 1, (name, old)
            planted = planted.replace(old, new)
        source.write_text(planted, encoding="utf-8")
        started = time.perf_counter()
        try:
            built, build_text = run_command(["cmake", "--build", str(build)], tree, env)
            record: dict[str, object] = {
                "plant": name,
                "commit": head_commit(),
                "description": description,
                "built": built == 0,
            }
            if built == 0:
                ctest, ctest_text = run_command(
                    ["ctest", "--test-dir", str(build), "-j2", "--output-on-failure"],
                    tree,
                    env,
                )
                flow, flow_text = run_command(
                    [
                        str(PYTHON),
                        "-m",
                        "pytest",
                        "tests/test_core_flow.py",
                        "-x",
                        "-q",
                        "--tb=line",
                        "-p",
                        "no:cacheprovider",
                        "-p",
                        "no:randomly",
                    ],
                    tree,
                    env,
                )
                section0, section0_text = run_command(
                    [
                        str(PYTHON),
                        "-m",
                        "pytest",
                        "tests/test_core_turbulence.py",
                        "-q",
                        "--tb=line",
                        "-p",
                        "no:cacheprovider",
                        "-k",
                        variants95.SECTION_0,
                    ],
                    tree,
                    env,
                )
                record.update(
                    ctest=ctest,
                    flow=flow,
                    section0=section0,
                    caught=ctest != 0 or flow != 0 or section0 != 0,
                    failing=failed_tests(ctest_text)[:3]
                    + failed_tests(flow_text)[:3]
                    + failed_tests(section0_text)[:3],
                )
            else:
                record.update(caught=None, build_tail=build_text[-1500:])
        finally:
            source.write_text(original, encoding="utf-8")
        record["seconds"] = time.perf_counter() - started
        records.append(record)
        print(
            name,
            record.get("caught"),
            record.get("failing"),
            f"{record['seconds']:.0f}s",
            flush=True,
        )
        suffix = f"s{stream}" if not only else f"s{stream}_only"
        (RESULTS / f"plants_{suffix}.json").write_text(
            json.dumps(records, indent=1) + "\n", encoding="utf-8"
        )


def main() -> None:
    """Run the command named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command", choices=["capture", "solve", "timing", "vecinfo", "plants"]
    )
    parser.add_argument("case", nargs="?", choices=(*CASES, *SOLVE_ONLY))
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--stream", type=int, default=0)
    parser.add_argument("--streams", type=int, default=3)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    if args.command == "capture":
        capture(args.case)
    elif args.command == "solve":
        solve(args.case)
    elif args.command == "timing":
        timing(args.run)
    elif args.command == "plants":
        sys.path.insert(0, str(HERE))
        plants(args.stream, args.streams, args.only, args.reverse)
    else:
        vecinfo()


if __name__ == "__main__":
    main()
