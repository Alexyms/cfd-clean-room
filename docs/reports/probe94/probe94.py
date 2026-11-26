"""Probe for prompt 94: the turbulence model and the wall functions in the core.

    python docs/reports/probe94/probe94.py capture product80
    python docs/reports/probe94/probe94.py capture vents80
    python docs/reports/probe94/probe94.py capture annex216
    python docs/reports/probe94/probe94.py capture backstep
    python docs/reports/probe94/probe94.py capture product80_umist
    python docs/reports/probe94/probe94.py replay
    python docs/reports/probe94/probe94.py replay --out replay_default
    python docs/reports/probe94/probe94.py elementary
    python docs/reports/probe94/probe94.py elementary --out elementary_default
    python docs/reports/probe94/probe94.py planted
    python docs/reports/probe94/probe94.py rest
    python docs/reports/probe94/probe94.py timing --run 1
    python docs/reports/probe94/probe94.py vecinfo
    python docs/reports/probe94/probe94.py plants

Run under WSL2 with the Linux environment (``~/.venvs/cfd_clean_room``) from the
checkout's root, one BLAS thread. NumPy's AVX-512 loops are off
(``NPY_DISABLE_CPU_FEATURES=X86_V4``) in every command but ``replay --out
replay_default``, ``elementary --out elementary_default`` and ``timing --run`` 1 to
3, which are the reference as NumPy runs it on this machine by default.
Nothing in ``src/`` changes: the probe wraps the reference's methods on their
classes, and each wrapper calls the reference and keeps a copy of what it was
handed and what it returned.

- ``capture``: a steady solve warm-started (``solve_steady(initial=...)``) and
  stopped after its first ``CALLS`` outer iterations, every call of
  ``TurbulenceBoundary.wall_viscosity`` and ``conditions`` and of
  ``KEpsilonModel.step`` kept. The cases start from prompt 92's and 93's states; the
  ``umist`` one is the 80x30 product room with ``turbulence.advection_scheme:
  umist``, its configuration written under ``results/builder94/``.
- ``replay``: every kept call through the core. The blocks of the first kind bit
  for bit; each CG solve of the kept step's systems by the rule (the reference's
  count, the field within 1e-12); the step's own differences. The reference called
  again on the kept arguments must give what was kept, where NumPy runs the same
  loops.
- ``elementary``: NumPy's ``log`` and cube against the C library's on planted values.
- ``planted``: the CG rule on the parity module's planted steps, which the module
  asserts on its gentle ones only.
- ``rest``: the same comparisons on the first outer iterations of each case's solve
  from rest, beside the captures and not scored.
- ``timing``: a step with its conditions on the first outer iteration's arguments of
  a solve warm-started from the stored 320x120 product state, the core against the
  reference in one process, the CG solves apart, with the machine's state.
- ``vecinfo``: GCC's vectorisation report on the two modules' blocks.
- ``plants``: each planted defect of ``variants94.py`` built in turn, with CTest, the
  parity module and the replay run against it.

The records are in ``results/builder94/`` (not committed).
"""

import argparse
import contextlib
import dataclasses
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "probe90"))
sys.path.insert(0, str(HERE.parent / "probe92"))
sys.path.insert(0, str(HERE.parent / "probe93"))
sys.path.insert(0, str(HERE))

# Prompt 90's probe: its timer and its record of the machine's state; 92's and
# 93's: their cases and the states they start from. Their commands are not used.
import probe90  # noqa: E402 -- after sys.path
import probe92  # noqa: E402 -- after sys.path
import probe93  # noqa: E402 -- after sys.path

from src import turbulence  # noqa: E402 -- after sys.path
from src.boundary_staggered import StaggeredBoundary  # noqa: E402 -- after sys.path
from src.config import SimConfig  # noqa: E402 -- after sys.path
from src.mesh import Mesh  # noqa: E402 -- after sys.path
from src.scalar_scheme import CONJUGATE_GRADIENT  # noqa: E402 -- after sys.path
from src.scalar_scheme import implicit_step as reference_solve  # noqa: E402 -- as above
from src.solver_staggered import FlowState  # noqa: E402 -- after sys.path
from src.staggered import FaceVelocities  # noqa: E402 -- after sys.path
from src.turbulence import (  # noqa: E402 -- after sys.path
    KEpsilonModel,
    TurbulenceBoundary,
    TurbulenceConditions,
    TurbulenceState,
)
from tests import core_mapping  # noqa: E402 -- after sys.path

RESULTS = ROOT / "results" / "builder94"
CAPTURES = RESULTS / "captures"
CASES = ("product80", "vents80", "annex216", "backstep", "product80_umist")
# The outer iterations captured after the warm start.
CALLS = 10
UMIST = "product80_umist"
STATE = ("k", "eps", "nu_t")
CONDITIONS = tuple(f.name for f in dataclasses.fields(TurbulenceConditions))
TERMS = tuple(f.name for f in dataclasses.fields(turbulence.StepTerms))
FIELD_TOL = 1e-12


def numpy_loops() -> dict[str, str]:
    """The loops NumPy runs for a float64 log and power in this process."""
    from numpy.lib import introspect

    info = introspect.opt_func_info(func_name="^(log|power)$", signature="float64")
    return {
        name: "; ".join(loop["current"] for loop in loops.values())
        for name, loops in info.items()
    }


# ---------------------------------------------------------------------------
# The cases


def case_config(name: str) -> SimConfig:
    """The configuration a case's model is built from."""
    if name != UMIST:
        return probe93.case_config(name)
    # The product room's file with its one key changed, and its base's path made
    # this system's, as probe 90 reads it.
    path = RESULTS / "product80_umist.yaml"
    raw = yaml.safe_load(
        Path(str(probe90.SCENARIOS["dust80"]["config"])).read_text(encoding="utf-8")
    )
    raw["scenario"]["base"] = probe90.local_path(raw["scenario"]["base"])
    raw["turbulence"] = {"advection_scheme": "umist"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return SimConfig.from_dict(
        yaml.safe_load(path.read_text(encoding="utf-8")), path.parent
    )


def start_of(name: str) -> tuple[FlowState, dict[str, object]]:
    """The state a case's solve is warm-started from, and where it came from."""
    return probe93.start_of("product80" if name == UMIST else name)


@dataclasses.dataclass
class Pair:
    """A case's models and boundaries, the reference's and the core's."""

    config: SimConfig
    mesh: Mesh
    model: KEpsilonModel
    walls: TurbulenceBoundary
    core: object
    core_walls: object


def pair_of(name: str) -> Pair:
    """A case's reference and core, built from the same arguments."""
    config = case_config(name)
    mesh = Mesh(config)
    boundary = StaggeredBoundary(mesh, config)
    grid = core_mapping.core_grid(mesh)
    return Pair(
        config,
        mesh,
        KEpsilonModel(mesh, config),
        TurbulenceBoundary(mesh, config, boundary),
        core_mapping.k_epsilon(mesh, config, boundary, grid),
        core_mapping.turbulence_boundary(mesh, config, boundary, grid),
    )


# ---------------------------------------------------------------------------
# Capture


class Recorder:
    """Wraps the model's step and the boundary's two calls and keeps each."""

    def __init__(self, calls: int) -> None:
        self.calls = calls
        self.kept: list[dict[str, np.ndarray]] = []
        self.walls: list[dict[str, np.ndarray]] = []

    def install(self) -> None:
        """Wrap the three methods; ``uninstall`` puts the reference's back."""
        self.methods = (
            TurbulenceBoundary.wall_viscosity,
            TurbulenceBoundary.conditions,
            KEpsilonModel.step,
        )
        recorder = self
        wall_viscosity, conditions, step = self.methods

        def kept_wall(
            self: TurbulenceBoundary, k: np.ndarray, elsewhere: dict[str, np.ndarray]
        ) -> dict[str, np.ndarray]:
            out = wall_viscosity(self, k, elsewhere)
            recorder.walls.append(
                {
                    "wall/k": np.array(k),
                    **{f"wall/elsewhere_{c}": np.array(elsewhere[c]) for c in "uv"},
                    **{f"wall/out_{c}": np.array(out[c]) for c in "uv"},
                }
            )
            return out

        def kept_conditions(
            self: TurbulenceBoundary, state: TurbulenceState, faces: FaceVelocities
        ) -> TurbulenceConditions:
            return conditions(self, state, faces)

        def kept_step(
            self: KEpsilonModel,
            state: TurbulenceState,
            faces: FaceVelocities,
            c: TurbulenceConditions,
            dt: float | None = None,
        ) -> TurbulenceState:
            call = {f"state/{n}": np.array(getattr(state, n)) for n in STATE}
            call["faces/u"], call["faces/v"] = np.array(faces.u), np.array(faces.v)
            call |= {f"conditions/{n}": np.array(getattr(c, n)) for n in CONDITIONS}
            out = step(self, state, faces, c, dt)
            call |= {f"out/{n}": np.array(getattr(out, n)) for n in STATE}
            call["out/iterations"] = np.array(self.last_iterations)
            call["out/fallbacks"] = np.array(self.last_fallbacks)
            recorder.kept.append(call | recorder.walls[-1])
            if len(recorder.kept) == recorder.calls:
                raise probe92.CapturedError
            return out

        TurbulenceBoundary.wall_viscosity = kept_wall
        TurbulenceBoundary.conditions = kept_conditions
        KEpsilonModel.step = kept_step

    def uninstall(self) -> None:
        """Put the reference's methods back."""
        (
            TurbulenceBoundary.wall_viscosity,
            TurbulenceBoundary.conditions,
            KEpsilonModel.step,
        ) = self.methods


def captured(name: str, calls: int) -> tuple[Recorder, dict[str, object]]:
    """The first ``calls`` outer iterations of a case's warm-started solve."""
    start, source = start_of(name)
    _mesh, _boundary, solver = probe92.solver_of(case_config(name))
    recorder = Recorder(calls)
    recorder.install()
    try:
        with contextlib.suppress(probe92.CapturedError):
            solver.solve_steady(initial=start)
    finally:
        recorder.uninstall()
    return recorder, source


def capture(name: str) -> None:
    """Keep a case's calls in ``results/builder94/captures/``."""
    recorder, source = captured(name, CALLS)
    CAPTURES.mkdir(parents=True, exist_ok=True)
    np.savez(
        CAPTURES / f"{name}.npz",
        **{
            f"call{i}/{key}": value
            for i, call in enumerate(recorder.kept)
            for key, value in call.items()
        },
    )
    first, last = recorder.kept[0], recorder.kept[-1]
    spec = case_config(name).turbulence
    assert spec is not None
    record = {
        "case": name,
        "start": source,
        "calls": len(recorder.kept),
        "shape": list(first["state/k"].shape),
        "variant": spec.variant,
        "advection": spec.advection_scheme,
        "tol": spec.tol,
        "max_iter": spec.max_iter,
        "iterations": [c["out/iterations"].tolist() for c in recorder.kept],
        "fallbacks": [c["out/fallbacks"].tolist() for c in recorder.kept],
        "largest_step_k": float(np.abs(last["state/k"] - first["state/k"]).max()),
        "distinct_calls": len({c["state/k"].tobytes() for c in recorder.kept}),
        "loops": numpy_loops(),
    }
    (CAPTURES / f"{name}.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps(record))


def loaded(name: str) -> list[dict[str, np.ndarray]]:
    """A case's kept calls, in order."""
    calls: dict[str, dict[str, np.ndarray]] = {}
    with np.load(CAPTURES / f"{name}.npz") as z:
        for key in z.files:
            call, part = key.split("/", 1)
            calls.setdefault(call, {})[part] = z[key]
    return list(calls.values())


@dataclasses.dataclass
class Arguments:
    """The arguments of a kept step."""

    state: TurbulenceState
    faces: FaceVelocities
    conditions: TurbulenceConditions

    def all(self) -> tuple[TurbulenceState, FaceVelocities, TurbulenceConditions]:
        """In the step's order."""
        return self.state, self.faces, self.conditions


def arguments_of(call: dict[str, np.ndarray]) -> Arguments:
    """A kept call's arguments as the reference's types."""
    return Arguments(
        TurbulenceState(**{n: call[f"state/{n}"] for n in STATE}),
        FaceVelocities.copy_of(call["faces/u"], call["faces/v"]),
        TurbulenceConditions(**{n: call[f"conditions/{n}"] for n in CONDITIONS}),
    )


# ---------------------------------------------------------------------------
# Replay


def differing(a: np.ndarray, b: np.ndarray) -> int:
    """How many entries of two arrays differ in their 64 bits; -1 for shapes or dtypes."""
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    if a.dtype != np.float64 and a.dtype == b.dtype and a.shape == b.shape:
        return int(np.count_nonzero(a != b))
    return probe92.differing(a, b)


def largest(a: np.ndarray, reference: np.ndarray) -> float:
    """The largest difference over the reference's largest magnitude."""
    return probe93.largest(a, reference)


def gradients(pair: Pair, args: Arguments) -> list[np.ndarray]:
    """The corner gradients the reference forms inside its strain."""
    found: list[np.ndarray] = []
    inner = turbulence._corner_difference

    def kept(*parts: np.ndarray) -> np.ndarray:
        found.append(inner(*parts))
        return found[-1]

    turbulence._corner_difference = kept
    try:
        pair.model.strain_squared(args.faces, args.conditions)
    finally:
        turbulence._corner_difference = inner
    return [found[0], found[1].T]


def first_kind(
    pair: Pair, call: dict[str, np.ndarray], args: Arguments
) -> dict[str, tuple[int, float]]:
    """Every block of the first kind on one kept call.

    Returns, by block, the entries that differ in a bit and the largest
    difference over the reference's largest magnitude.
    """
    model, core, walls, core_walls = pair.model, pair.core, pair.walls, pair.core_walls
    state, faces, c = args.all()
    out: dict[str, tuple[int, float]] = {}

    def compare(block: str, expected: list[np.ndarray], got: list[np.ndarray]) -> None:
        pairs = list(zip(expected, got, strict=True))
        out[block] = (
            sum(differing(np.asarray(a), np.asarray(b)) for a, b in pairs),
            max(largest(np.asarray(b, float), np.asarray(a, float)) for a, b in pairs),
        )

    elsewhere = {key: call[f"wall/elsewhere_{key}"] for key in "uv"}
    mu = walls.wall_viscosity(call["wall/k"], elsewhere)
    got_mu = core_walls.wall_viscosity(call["wall/k"], elsewhere)
    compare("wall_viscosity", [mu[k] for k in "uv"], [got_mu[k] for k in "uv"])
    conditions, got_c = (
        walls.conditions(state, faces),
        core_walls.conditions(state, faces),
    )
    for block, names in (
        ("inflow", ("inflow_k_u", "inflow_k_v", "inflow_eps_u", "inflow_eps_v")),
        ("tangential", tuple(n for n in CONDITIONS if n.startswith("tangential"))),
        ("eps_held", ("eps_held", "eps_wall")),
        ("production_given", ("production_given", "production")),
    ):
        compare(
            block, [getattr(conditions, n) for n in names], [got_c[n] for n in names]
        )
    dt = model.pseudo_time_step(faces)
    compare("pseudo_time_step", [dt], [core.pseudo_time_step(faces)])
    compare(
        "corner_gradients",
        gradients(pair, args),
        [
            core.corner_gradient(0, faces.u, c.tangential_bottom, c.tangential_top),
            core.corner_gradient(1, faces.v, c.tangential_left, c.tangential_right),
        ],
    )
    compare(
        "strain_squared",
        [model.strain_squared(faces, c)],
        [core.strain_squared(faces, c)],
    )
    terms, got_terms = model.terms(state, faces, c), core.terms(state, faces, c)
    compare("terms", [getattr(terms, n) for n in TERMS], [got_terms[n] for n in TERMS])
    flux_u = np.where(model._live_u, faces.u, 0.0) * model._area_u
    flux_v = np.where(model._live_v, faces.v, 0.0) * model._area_v
    for name, q, u, v in (
        ("k", state.k, c.inflow_k_u, c.inflow_k_v),
        ("eps", state.eps, c.inflow_eps_u, c.inflow_eps_v),
    ):
        compare(
            f"{name}_advected",
            [model._advect(q, flux_u, flux_v, u, v, dt)],
            [core.advected(faces, q, u, v)],
        )
    const = model.constants
    for name, sigma in (("k", const.sigma_k), ("eps", const.sigma_e)):
        compare(
            f"{name}_conductances",
            list(model._conductances(state.nu_t, sigma)),
            list(core.conductances(name, state.nu_t)),
        )
    compare(
        "eddy_viscosity",
        [model.eddy_viscosity(state.k, state.eps)],
        [core.eddy_viscosity(state.k, state.eps)],
    )
    return out


def systems_of(pair: Pair, args: Arguments) -> tuple[TurbulenceState, list, list]:
    """The reference's step, and what it handed and got from its two implicit solves."""
    handed: list[tuple[tuple, dict]] = []
    results: list[object] = []

    def kept(*parts: object, **keys: object) -> object:
        handed.append((parts, keys))
        results.append(reference_solve(*parts, **keys))
        return results[-1]

    turbulence.implicit_step = kept
    try:
        state = pair.model.step(*args.all())
    finally:
        turbulence.implicit_step = reference_solve
    return state, handed, results


def core_solve(pair: Pair, handed: tuple[tuple, dict]) -> tuple[np.ndarray, object]:
    """The core's solve of a system the reference's step handed its own."""
    import _cfdcore

    parts, keys = handed
    out = np.empty_like(parts[0])
    result = _cfdcore.implicit_step(
        _cfdcore.ImplicitScratch2(core_mapping.core_grid(pair.mesh)),
        *parts,
        out,
        held=keys.get("held"),
        method=CONJUGATE_GRADIENT,
    )
    return out, result


def solve_rows(pair: Pair, args: Arguments) -> tuple[TurbulenceState, list[dict], dict]:
    """The two solves of a kept step by the CG rule, and the explicit fields' bits."""
    state, handed, results = systems_of(pair, args)
    rows = []
    for name, system, reference in zip(("k", "eps"), handed, results, strict=True):
        field, result = core_solve(pair, system)
        rows.append(
            {
                "quantity": name,
                "iterations": [int(reference.iterations), int(result.iterations)],
                "fallbacks": [int(reference.fallbacks), int(result.fallbacks)],
                "restarts": int(result.restarts),
                "field": largest(field, reference.field),
                "field_differs": differing(reference.field, field),
            }
        )
    stars = pair.core.explicit_fields(*args.all())
    explicit = {
        name: differing(system[0][0], star)
        for name, system, star in zip(("k", "eps"), handed, stars, strict=True)
    }
    return state, rows, explicit


def replay(out_name: str) -> None:
    """Every kept call through the core, against the reference called again."""
    rows = []
    for name in CASES:
        pair = pair_of(name)
        for index, call in enumerate(loaded(name)):
            args = arguments_of(call)
            blocks = first_kind(pair, call, args)
            state, solves, explicit = solve_rows(pair, args)
            stars = {f"{q}_explicit": (n, 0.0) for q, n in explicit.items()}
            got = pair.core.step(*args.all())
            whole = pair.core.advance(pair.core_walls, args.state, args.faces)
            rows.append(
                {
                    "case": name,
                    "call": index,
                    "blocks": {k: list(v) for k, v in (blocks | stars).items()},
                    "solves": solves,
                    "step": {n: largest(got[n], getattr(state, n)) for n in STATE},
                    "advance": {n: largest(whole[n], getattr(state, n)) for n in STATE},
                    "counts": [
                        list(pair.model.last_iterations),
                        list(pair.core.last_iterations),
                    ],
                    "fallbacks": [
                        list(pair.model.last_fallbacks),
                        list(pair.core.last_fallbacks),
                    ],
                    # The kept outputs against the reference's again.
                    "reference_differs": sum(
                        differing(call[f"out/{n}"], getattr(state, n)) for n in STATE
                    ),
                }
            )
        mine = [r for r in rows if r["case"] == name]
        counts = [s for r in mine for s in r["solves"]]
        print(
            name,
            len(mine),
            "steps,",
            sum(any(v[0] for v in r["blocks"].values()) for r in mine),
            "with a block that differs,",
            sum(s["iterations"][0] != s["iterations"][1] for s in counts),
            "counts that differ,",
            sum(s["fallbacks"][0] != s["fallbacks"][1] for s in counts),
            "fallbacks that differ, worst solve",
            f"{max(s['field'] for s in counts):.1e},",
            "worst step",
            f"{max(max(r['step'].values()) for r in mine):.1e},",
            sum(bool(r["reference_differs"]) for r in mine),
            "not reproduced",
            flush=True,
        )
    record = {"loops": numpy_loops(), "rows": rows}
    target = RESULTS / f"{out_name}.json"
    if not out_name.startswith("replay"):
        target = RESULTS / "plants" / f"{out_name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# The elementary functions


def elementary(out_name: str) -> None:
    """NumPy's log and cube against the C library's, on the values the blocks meet."""
    rng = np.random.default_rng(94)
    n = 200000
    values = {
        "log(E y*)": (np.log, math.log, 10.0 ** rng.uniform(1.0, 4.0, n)),
        "u_k**3": (
            lambda x: x**3,
            lambda x: math.pow(x, 3.0),
            10.0 ** rng.uniform(-4.0, 0.0, n),
        ),
        "eta**3": (
            lambda x: x**3,
            lambda x: math.pow(x, 3.0),
            10.0 ** rng.uniform(-2.0, 2.0, n),
        ),
    }
    rows = {}
    for name, (by_numpy, by_c, x) in values.items():
        a = by_numpy(x)
        b = np.array([by_c(v) for v in x])
        rows[name] = {
            "values": n,
            "differ": int(np.count_nonzero(a != b)),
            "largest_relative": float(np.max(np.abs(a - b) / np.abs(b))),
            "largest_ulps": float(np.max(np.abs(a - b) / np.spacing(np.abs(b)))),
        }
    record = {"loops": numpy_loops(), "rows": rows}
    (RESULTS / f"{out_name}.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps(record))


# ---------------------------------------------------------------------------
# The planted steps


def planted() -> None:
    """The CG rule on the parity module's planted steps: counts and fields, recorded."""
    from tests import test_core_turbulence as module

    rows = []
    for name in ("80x30", "80x30-vents-shelf", "40x15-uniform-open"):
        for variant in module.VARIANTS:
            for scheme in module.SCHEMES:
                room = module._room(name, variant, scheme)
                pair = Pair(
                    room.config,
                    room.mesh,
                    room.model,
                    room.walls,
                    room.core,
                    room.core_walls,
                )
                for flow in module.STEP_FLOWS:
                    rng = module._rng(name, variant, scheme, flow)
                    inputs = module._planted(room, flow, rng)
                    kinds = {"planted": inputs, "gentle": module._gentle(room, rng)}
                    for kind, x in kinds.items():
                        args = Arguments(x.state, x.faces, x.conditions)
                        _state, solves, explicit = solve_rows(pair, args)
                        rows.append(
                            {
                                "kind": kind,
                                "name": f"{name} {variant} {scheme} {flow}",
                                "solves": solves,
                                "explicit": explicit,
                            }
                        )
    (RESULTS / "planted.json").write_text(
        json.dumps(rows, indent=1) + "\n", encoding="utf-8"
    )
    for kind in ("planted", "gentle"):
        mine = [s for r in rows if r["kind"] == kind for s in r["solves"]]
        print(
            kind,
            len(mine),
            "solves, counts that differ",
            sum(s["iterations"][0] != s["iterations"][1] for s in mine),
            "fallbacks that differ",
            sum(s["fallbacks"][0] != s["fallbacks"][1] for s in mine),
            "beyond 1e-12 at equal counts",
            sum(
                s["iterations"][0] == s["iterations"][1] and s["field"] > FIELD_TOL
                for s in mine
            ),
            "worst",
            f"{max(s['field'] for s in mine):.1e}",
        )


# ---------------------------------------------------------------------------
# From rest


def rest() -> None:
    """The first outer iterations of each case's solve from rest, beside the captures.

    Not among the prompt's captures and not scored: a solve from rest starts from
    uniform k and eps in a room mostly at rest, which no warm start meets.
    """
    rows = []
    for name in CASES:
        pair = pair_of(name)
        _mesh, _boundary, solver = probe92.solver_of(case_config(name))
        recorder = Recorder(CALLS)
        recorder.install()
        try:
            with contextlib.suppress(probe92.CapturedError):
                solver.solve_steady()
        finally:
            recorder.uninstall()
        for index, call in enumerate(recorder.kept):
            args = arguments_of(call)
            blocks = first_kind(pair, call, args)
            _state, solves, explicit = solve_rows(pair, args)
            stars = {f"{q}_explicit": (n, 0.0) for q, n in explicit.items()}
            rows.append(
                {
                    "case": name,
                    "call": index,
                    "blocks": {k: list(v) for k, v in (blocks | stars).items()},
                    "solves": solves,
                }
            )
        mine = [r for r in rows if r["case"] == name]
        solves = [s for r in mine for s in r["solves"]]
        print(
            name,
            len(mine),
            "steps from rest,",
            sum(any(v[0] for v in r["blocks"].values()) for r in mine),
            "with a block that differs,",
            sum(s["iterations"][0] != s["iterations"][1] for s in solves),
            "counts that differ, worst solve",
            f"{max(s['field'] for s in solves):.1e}",
            [s["iterations"][0] for s in solves[:4]],
            flush=True,
        )
    (RESULTS / "rest.json").write_text(
        json.dumps({"loops": numpy_loops(), "rows": rows}, indent=1) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Timing

REFERENCE_REPEATS = 40
CORE_REPEATS = 200


def timing(run_number: int) -> None:
    """A step with its conditions on the 320x120 product state, in one process."""
    before = probe90.windows_load()
    calibration = probe90.calibration()
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    recorder, _source = captured("product320", 1)
    call = recorder.kept[0]
    args = arguments_of(call)
    pair = pair_of("product320")
    state, faces, _conditions = args.all()
    model, walls, core, core_walls = pair.model, pair.walls, pair.core, pair.core_walls

    def reference_update() -> TurbulenceState:
        return model.step(state, faces, walls.conditions(state, faces))

    def core_update() -> dict[str, np.ndarray]:
        c = core_mapping.turbulence_conditions(core_walls.conditions(state, faces))
        return core.step(state, faces, c)

    expected, got = reference_update(), core_update()
    ref = probe90.timed(reference_update, REFERENCE_REPEATS)
    own = probe90.timed(core_update, CORE_REPEATS)
    whole = probe90.timed(lambda: core.advance(core_walls, state, faces), CORE_REPEATS)
    _state, handed, _results = systems_of(pair, args)
    elsewhere = {key: call[f"wall/elsewhere_{key}"] for key in "uv"}

    def solves(side: str) -> None:
        for parts, keys in handed:
            if side == "reference":
                reference_solve(*parts, **keys)
            else:
                core_solves(parts, keys)

    import _cfdcore

    scratch = _cfdcore.ImplicitScratch2(core_mapping.core_grid(pair.mesh))
    out = np.empty_like(state.k)

    def core_solves(parts: tuple, keys: dict) -> None:
        _cfdcore.implicit_step(
            scratch, *parts, out, held=keys.get("held"), method=CONJUGATE_GRADIENT
        )

    parts = {
        "two CG solves": (lambda: solves("reference"), lambda: solves("core")),
        "conditions": (
            lambda: walls.conditions(state, faces),
            lambda: core_walls.conditions(state, faces),
        ),
        "wall viscosity": (
            lambda: walls.wall_viscosity(state.k, elsewhere),
            lambda: core_walls.wall_viscosity(state.k, elsewhere),
        ),
        "terms": (
            lambda: model.terms(*args.all()),
            lambda: core.terms(*args.all()),
        ),
        "pseudo-time step": (
            lambda: model.pseudo_time_step(faces),
            lambda: core.pseudo_time_step(faces),
        ),
    }
    record = {
        "run": run_number,
        "started": started,
        "loops": numpy_loops(),
        "iterations": [list(model.last_iterations), list(core.last_iterations)],
        "fields": {n: largest(got[n], getattr(expected, n)) for n in STATE},
        "reference": ref,
        "core": own,
        "advance": whole,
        "ratio": ref["median_s"] / own["median_s"],
        "ratio_advance": ref["median_s"] / whole["median_s"],
        "parts": {
            name: {
                "reference": probe90.timed(a, REFERENCE_REPEATS),
                "core": probe90.timed(b, CORE_REPEATS),
            }
            for name, (a, b) in parts.items()
        },
        "calibration_before": calibration,
        "calibration_after": probe90.calibration(),
        "windows_before": before,
        "windows_after": probe90.windows_load(),
    }
    (RESULTS / f"timing_{run_number}.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )
    print(
        f"run {run_number}: {ref['median_s'] * 1e3:.2f} ms against "
        f"{own['median_s'] * 1e3:.3f} ms, ratio {record['ratio']:.2f}; advance "
        f"{whole['median_s'] * 1e3:.3f} ms, ratio {record['ratio_advance']:.2f}; "
        f"iterations {record['iterations']}; {record['loops']}"
    )
    for name, times in record["parts"].items():
        print(
            f"  {name}: {times['reference']['median_s'] * 1e3:.3f} against "
            f"{times['core']['median_s'] * 1e3:.3f} ms"
        )


# ---------------------------------------------------------------------------
# The vectorisation report

VEC_FLAGS = ["-std=c++20", "-O2", "-ffp-contract=off", "-march=x86-64-v3", "-DNDEBUG"]
VEC_FILES = ("turbulence", "walls")


def vecinfo() -> None:
    """Which blocks of the two files GCC vectorises, its loops counted by place."""
    version = subprocess.run(["g++", "--version"], capture_output=True, text=True)
    blocks = []
    for file in VEC_FILES:
        path = ROOT / "core" / "src" / f"{file}.cpp"
        done = subprocess.run(
            [
                "g++",
                *VEC_FLAGS,
                f"-I{ROOT / 'core' / 'include'}",
                "-c",
                str(path),
                "-o",
                "/dev/null",
                "-fopt-info-vec-optimized",
            ],
            capture_output=True,
            text=True,
        )
        found = re.findall(
            rf"{file}\.cpp:(\d+):\d+: optimized: +loop vectorized using (\d+) byte",
            done.stdout + done.stderr,
        )
        name = ""
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = re.match(r"inline [\w:]+ (\w+)\(lanes::cells<N>", line)
            if match:
                name = match.group(1)
            if name and re.search(r"for \(std::size_t \w = 0; \w < N; \+\+\w\)", line):
                widths = sorted({w for at, w in found if int(at) == number})
                blocks.append(
                    {
                        "file": file,
                        "block": name,
                        "vectorised": bool(widths),
                        "widths": widths,
                    }
                )
    record = {
        "flags": VEC_FLAGS,
        "compiler": version.stdout.splitlines()[0],
        "blocks": blocks,
    }
    (RESULTS / "vecinfo.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )
    for block in blocks:
        print(block["file"], block["block"], block["vectorised"], block["widths"])


# ---------------------------------------------------------------------------
# Planted defects

PLANT_WORK = Path.home() / "scratch94" / "plants"
PLANT_TIMEOUT = 900
MODULE = "tests/test_core_turbulence.py"


def plants(stream: int, streams: int, only: list[str]) -> None:
    """Each planted defect built in turn, with the tests and the replay that fail on it."""
    import variants94

    work = PLANT_WORK / f"s{stream}"
    mine = [
        plant
        for i, plant in enumerate(variants94.PLANTS)
        if plant[0] in only
        or (not only and (i % streams == stream or plant[0] == "control"))
    ]
    records = []
    for name, what, edits in mine:
        record: dict[str, object] = {"name": name, "what": what, "stream": stream}
        failed = probe92.build(work, edits, tests=True)
        if failed:
            record["build"] = failed
        else:
            run_nets(name, work / "build", record)
        records.append(record)
        print(
            name,
            "|",
            record.get("build", "")[-300:],
            record.get("ctest"),
            "|",
            record.get("pytest"),
            "|",
            "; ".join(record.get("replay", [])),
            flush=True,
        )
        (RESULTS / f"plants_s{stream}.json").write_text(
            json.dumps(records, indent=1) + "\n", encoding="utf-8"
        )


def run_nets(name: str, tree: Path, record: dict[str, object]) -> None:
    """CTest, the parity module and the replay on one build, their failures into the record."""

    def run(command: list[str], **keys: object) -> tuple[int, str]:
        try:
            done = subprocess.run(
                command, capture_output=True, text=True, timeout=PLANT_TIMEOUT, **keys
            )
        except subprocess.TimeoutExpired:
            # A plant that keeps a solve from ending is caught by that.
            return -1, "timed out"
        return done.returncode, done.stdout

    _code, out = run(["ctest", "--test-dir", str(tree), "--timeout", "60"])
    summary = [line for line in out.splitlines() if "tests passed" in line]
    record["ctest"] = summary[0].strip() if summary else out[-300:]
    # A test that crashes is listed as (SEGFAULT) or (Subprocess aborted), not (Failed).
    record["ctest_failed"] = re.findall(r"\d+ - (\S+) \([^)]+\)", out)
    env = {
        **os.environ,
        "PYTHONPATH": str(tree),
        "CFD_CORE_REQUIRED": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NPY_DISABLE_CPU_FEATURES": "X86_V4",
    }
    code, out = run(
        [
            sys.executable,
            "-m",
            "pytest",
            MODULE,
            "-q",
            "-rf",
            "--tb=no",
            "-p",
            "no:cacheprovider",
            "-W",
            "ignore",
        ],
        cwd=ROOT,
        env=env,
    )
    lines = out.splitlines()
    record["pytest_returncode"] = code
    record["pytest"] = next(
        (x.strip("= ") for x in reversed(lines) if re.search(r" in [0-9.]+s", x)),
        lines[-1] if lines else "no output",
    )
    bad = [
        line.split("::", 1)[1].split(" ")[0]
        for line in lines
        if line.startswith(f"FAILED {MODULE}::")
    ]
    record["parity_failed_by_test"] = dict(
        sorted(Counter(case.split("[")[0] for case in bad).items())
    )
    code, out = run(
        [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "replay",
            "--out",
            f"plant_{name}",
        ],
        cwd=ROOT,
        env=env,
    )
    record["replay_returncode"] = code
    record["replay"] = [x for x in out.splitlines() if " steps," in x]


# ---------------------------------------------------------------------------


def main() -> None:
    """Run one command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "capture",
            "replay",
            "elementary",
            "planted",
            "rest",
            "timing",
            "vecinfo",
            "plants",
        ),
    )
    parser.add_argument("case", nargs="?")
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--out", default="")
    parser.add_argument("--stream", type=int, default=0)
    parser.add_argument("--streams", type=int, default=1)
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args()
    commands = {
        "capture": lambda: capture(args.case),
        "replay": lambda: replay(args.out or "replay"),
        "elementary": lambda: elementary(args.out or "elementary"),
        "planted": planted,
        "rest": rest,
        "timing": lambda: timing(args.run),
        "vecinfo": vecinfo,
        "plants": lambda: plants(args.stream, args.streams, args.only),
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
