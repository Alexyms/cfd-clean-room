"""Prompt 92's probe: the reference's own predict calls, captured and replayed through the core.

    python docs/reports/probe92/probe92.py capture product80
    python docs/reports/probe92/probe92.py capture vents80
    python docs/reports/probe92/probe92.py capture cavity40
    python docs/reports/probe92/probe92.py capture annex216
    python docs/reports/probe92/probe92.py replay
    python docs/reports/probe92/probe92.py timing --run 1
    python docs/reports/probe92/probe92.py forms --run 1
    python docs/reports/probe92/probe92.py vecinfo
    python docs/reports/probe92/probe92.py plants
    python docs/reports/probe92/probe92.py kernels

Run under WSL2 with the Linux environment (``~/.venvs/cfd_clean_room``) from
the checkout's root, one BLAS thread, through ``results/builder92/probe.sh``.
Nothing in ``src/`` changes: the probe wraps ``MomentumPredictor.predict`` and
``stencil_viscosity`` on the class, and each wrapper calls the reference and
keeps a copy of what it was handed and what it returned.

- ``capture``: a steady solve warm-started (``solve_steady(initial=...)``) and
  stopped after its first ``CALLS`` outer iterations, every predict and
  stencil_viscosity call kept. The 80x30 product room starts from prompt 81's
  stored state, the vents room from prompt 90's, the cavity from its own
  converged solve, made here, and the Annex 20 room from the state its solve
  from rest reaches after ``ANNEX_OUTER`` outer iterations, made here (no
  stored file holds its pressure, k and eps).
- ``replay``: every kept call through the core, every output compared bit for
  bit, and through the reference again, which must give what was captured.
- ``timing``: predict on the first outer iteration's arguments of a solve
  warm-started from the stored 320x120 product state, the core against the
  reference in one process, with the machine's state.
- ``forms``: the kernels' forms built side by side and timed on the same
  arguments, in one session.
- ``kernels`` (prompt 92b): each kernel and each pass of ``momentum.cpp``, timed
  alone by ``kernels92b.cpp`` in the build of the commit before the restructure
  (6ca02d7, its ``core/`` in ``results/builder92b/head_core.tar``) and in this one.
- ``plants``: each planted defect of ``variants92.py`` built in turn, with
  CTest, the parity module and the replay run against it.

The captures stay in ``results/builder92/captures/`` (not committed). What ``replay``,
``vecinfo``, ``plants`` and ``kernels`` record is of the per-direction module and goes to
``results/builder92b/``.
"""

import argparse
import contextlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "probe90"))
sys.path.insert(0, str(HERE))

# Prompt 90's probe: its stored runs read under WSL2, its timer and its record
# of the machine's state. Its commands are not used.
import probe90  # noqa: E402 -- after sys.path

from src import core_backend as core_mapping  # noqa: E402 -- after sys.path
from src.boundary_staggered import StaggeredBoundary  # noqa: E402 -- after sys.path
from src.config import SimConfig  # noqa: E402 -- after sys.path
from src.io_manager import RunFile  # noqa: E402 -- after sys.path
from src.mesh import Mesh  # noqa: E402 -- after sys.path
from src.momentum import MomentumPredictor  # noqa: E402 -- after sys.path
from src.solver_staggered import (  # noqa: E402 -- after sys.path
    FlowState,
    StaggeredSolver,
)
from validation.cases import load_annex20, load_case  # noqa: E402 -- after sys.path

RESULTS = ROOT / "results" / "builder92"
CAPTURES = RESULTS / "captures"
# Prompt 92b's records, of the module written per direction.
CURRENT = ROOT / "results" / "builder92b"
BEFORE = CURRENT / "head_core.tar"
STORED = ROOT / "results" / "builder81"
VENTS_FLOW = ROOT / "results" / "builder90" / "vents80_flow.npz"
PRODUCT320 = STORED / "product" / "equipment_dust.h5"
CASES = ("product80", "vents80", "cavity40", "annex216")
# The outer iterations captured after the warm start, and those the Annex 20
# room takes from rest to reach the state it starts from.
CALLS = 10
ANNEX_OUTER = 400
OUTPUTS = ("u_star", "v_star", "a_p_u", "a_p_v")
ARGUMENTS = ("u", "v", "p", "mu_eff", "wall_u", "wall_v")


# ---------------------------------------------------------------------------
# The cases


class CapturedError(Exception):
    """Raised by the wrapper to leave a solve whose calls are all kept."""


def stored_state(run_file: Path) -> FlowState:
    """The one flow state a stored run file holds, as the warm start reads it."""
    stored = RunFile(run_file)
    (fingerprint,) = stored.flow_fingerprints()
    return stored.read_flow_state(fingerprint).state


def solver_of(config: SimConfig) -> tuple[Mesh, StaggeredBoundary, StaggeredSolver]:
    """A configuration's mesh, velocity boundary and steady solver."""
    mesh = Mesh(config)
    boundary = StaggeredBoundary(mesh, config)
    return mesh, boundary, StaggeredSolver(mesh, config, boundary)


def case_config(name: str) -> SimConfig:
    """The configuration a case's predictor is built from."""
    if name == "product80":
        return probe90.scenario_config("dust80")
    if name == "product320":
        return probe90.scenario_config("dust320")
    if name == "vents80":
        return probe90.scenario_config("vents80")
    if name == "cavity40":
        return load_case("cavity")
    if name == "annex216":
        return load_annex20()
    raise ValueError(name)


def solved(
    config: SimConfig, cap: int | None = None
) -> tuple[FlowState, dict[str, object]]:
    """The state a solve from rest reaches at its stop, or at a cap put on it."""
    if cap is not None:
        config.max_simple_iter = cap
    _mesh, _boundary, solver = solver_of(config)
    solver.solve_steady()
    if solver.flow_state is None:
        raise SystemExit("the solve left no state")
    return solver.flow_state, {
        "outer": len(solver.residual_history),
        "converged": bool(solver.converged),
        "stop_reason": solver.stop_reason,
    }


def start_of(name: str) -> tuple[FlowState, dict[str, object]]:
    """The state a case's solve is warm-started from, and where it came from."""
    if name == "product80":
        path = STORED / "placement" / "dust.h5"
        return stored_state(path), {"from": str(path.relative_to(ROOT))}
    if name == "product320":
        return stored_state(PRODUCT320), {"from": str(PRODUCT320.relative_to(ROOT))}
    if name == "vents80":
        with np.load(VENTS_FLOW) as z:
            state = FlowState(**{k: z[k] for k in z.files})
        return state, {"from": str(VENTS_FLOW.relative_to(ROOT))}
    if name == "cavity40":
        state, record = solved(case_config(name))
        return state, {"from": "its own solve from rest, to its stop", **record}
    if name == "annex216":
        state, record = solved(case_config(name), ANNEX_OUTER)
        return state, {"from": "its own solve from rest, to the cap", **record}
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Capture


class Recorder:
    """Wraps predict and stencil_viscosity on the class and keeps each call."""

    def __init__(self, calls: int) -> None:
        self.calls = calls
        self.kept: list[dict[str, np.ndarray]] = []
        self.stencils: list[dict[str, np.ndarray]] = []

    def install(self) -> None:
        """Wrap the two methods; ``uninstall`` puts the reference's back."""
        self.predict = MomentumPredictor.predict
        self.stencil = MomentumPredictor.stencil_viscosity
        recorder = self

        def predict(
            self: MomentumPredictor,
            u: np.ndarray,
            v: np.ndarray,
            p: np.ndarray,
            mu_eff: np.ndarray | None = None,
            wall_mu: dict[str, np.ndarray] | None = None,
            transient: object = None,
        ) -> object:
            call = {"u": np.array(u), "v": np.array(v), "p": np.array(p)}
            if mu_eff is not None:
                call["mu_eff"] = np.array(mu_eff)
            if wall_mu is not None:
                call["wall_u"] = np.array(wall_mu["u"])
                call["wall_v"] = np.array(wall_mu["v"])
            out = recorder.predict(self, u, v, p, mu_eff, wall_mu, transient)
            for key in OUTPUTS:
                call[key] = np.array(getattr(out, key))
            recorder.kept.append(call)
            if len(recorder.kept) == recorder.calls:
                raise CapturedError
            return out

        def stencil(self: MomentumPredictor, mu_eff: np.ndarray) -> object:
            out = recorder.stencil(self, mu_eff)
            recorder.stencils.append(
                {
                    "mu_eff": np.array(mu_eff),
                    "u": np.array(out["u"]),
                    "v": np.array(out["v"]),
                }
            )
            return out

        MomentumPredictor.predict = predict
        MomentumPredictor.stencil_viscosity = stencil

    def uninstall(self) -> None:
        """Put the reference's methods back."""
        MomentumPredictor.predict = self.predict
        MomentumPredictor.stencil_viscosity = self.stencil


def captured(name: str, calls: int) -> tuple[Recorder, dict[str, object]]:
    """The first ``calls`` outer iterations' calls of a case's warm-started solve."""
    start, source = start_of(name)
    _mesh, _boundary, solver = solver_of(case_config(name))
    recorder = Recorder(calls)
    recorder.install()
    try:
        with contextlib.suppress(CapturedError):
            solver.solve_steady(initial=start)
    finally:
        recorder.uninstall()
    return recorder, source


def capture(name: str) -> None:
    """Keep a case's calls in ``results/builder92/captures/``."""
    recorder, source = captured(name, CALLS)
    arrays = {}
    for i, call in enumerate(recorder.kept):
        for key, value in call.items():
            arrays[f"call{i}/{key}"] = value
    for i, call in enumerate(recorder.stencils):
        for key, value in call.items():
            arrays[f"stencil{i}/{key}"] = value
    CAPTURES.mkdir(parents=True, exist_ok=True)
    np.savez(CAPTURES / f"{name}.npz", **arrays)
    config = case_config(name)
    first, last = recorder.kept[0], recorder.kept[-1]
    record = {
        "case": name,
        "start": source,
        "calls": len(recorder.kept),
        "stencils": len(recorder.stencils),
        "shape": list(first["p"].shape),
        "sweeps": int(config.momentum_sweeps),
        "alpha": float(config.alpha_velocity),
        "field": "mu_eff" in first,
        "wall": "wall_u" in first,
        # The calls differ from one another, so replaying them is not one call ten times.
        "largest_step_u": float(np.abs(last["u"] - first["u"]).max()),
        "distinct_calls": len({c["u"].tobytes() for c in recorder.kept}),
    }
    (CAPTURES / f"{name}.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps(record))


# ---------------------------------------------------------------------------
# Replay


def differing(a: np.ndarray, b: np.ndarray) -> int:
    """How many entries of two arrays differ in their 64 bits; -1 for shapes or dtypes."""
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return -1
    return int(np.count_nonzero(a.view(np.uint64) != b.view(np.uint64)))


def loaded(name: str) -> list[dict[str, np.ndarray]]:
    """A case's kept calls, predict's and then stencil_viscosity's under "stencil"."""
    calls: dict[str, dict[str, np.ndarray]] = {}
    with np.load(CAPTURES / f"{name}.npz") as z:
        for key in z.files:
            call, part = key.split("/")
            calls.setdefault(call, {})[part] = z[key]
    return [dict(value, call=key) for key, value in calls.items()]  # type: ignore[misc]


def arguments(call: dict[str, np.ndarray]) -> tuple:
    """A kept call's arguments as predict takes them."""
    wall = None
    if "wall_u" in call:
        wall = {"u": call["wall_u"], "v": call["wall_v"]}
    return call["u"], call["v"], call["p"], call.get("mu_eff"), wall


def replay(out_name: str = "replay") -> None:
    """Every kept call through the core and the reference, every output compared."""
    rows = []
    for name in CASES:
        if not (CAPTURES / f"{name}.npz").exists():
            print(name, "not captured")
            continue
        config = case_config(name)
        mesh = Mesh(config)
        boundary = StaggeredBoundary(mesh, config)
        reference = MomentumPredictor(mesh, config, boundary)
        core = core_mapping.momentum_predictor(mesh, config, boundary)
        vents = boundary.vent_faces()
        for call in loaded(name):
            row: dict[str, object] = {
                "case": name,
                "call": call["call"],
                "vents": bool(vents.any),
            }
            if str(call["call"]).startswith("stencil"):
                got = core.stencil_viscosity(call["mu_eff"])
                again = reference.stencil_viscosity(call["mu_eff"])
                row["what"] = "stencil_viscosity"
                row["differ"] = {k: differing(call[k], got[k]) for k in ("u", "v")}
                row["reference_differs"] = {
                    k: differing(call[k], again[k]) for k in ("u", "v")
                }
            else:
                args = arguments(call)
                got = dict(zip(OUTPUTS, core.predict(*args), strict=True))
                again = reference.predict(*args)
                row["what"] = "predict"
                row["field"] = args[3] is not None
                row["wall"] = args[4] is not None
                row["differ"] = {k: differing(call[k], got[k]) for k in OUTPUTS}
                row["reference_differs"] = {
                    k: differing(call[k], getattr(again, k)) for k in OUTPUTS
                }
                row["unknowns"] = int(
                    (call["a_p_u"] > 0).sum() + (call["a_p_v"] > 0).sum()
                )
                # A vent's faces carry a velocity, so keeping them is tested.
                row["vent_faces_moving"] = int(
                    (call["u"][vents.normal_u] != 0).sum()
                    + (call["v"][vents.normal_v] != 0).sum()
                )
            rows.append(row)
        mine = [r for r in rows if r["case"] == name]
        bad = [r for r in mine if any(r["differ"].values())]
        wrong = [r for r in mine if any(r["reference_differs"].values())]
        print(
            name, len(mine), "calls,", len(bad), "differ,", len(wrong), "not reproduced"
        )
    target = CURRENT / f"{out_name}.json"
    if out_name != "replay":
        target = CURRENT / "plants" / f"{out_name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Timing

REFERENCE_REPEATS = 30
CORE_REPEATS = 300
FORM_REPEATS = 400


def product_call() -> tuple[SimConfig, dict[str, np.ndarray]]:
    """The first outer iteration's arguments on the stored 320x120 product state."""
    recorder, _source = captured("product320", 1)
    return case_config("product320"), recorder.kept[0]


def with_sweeps(config: SimConfig, sweeps: int) -> object:
    """The predictor's four keys with another sweep count."""
    from types import SimpleNamespace

    return SimpleNamespace(
        rho=config.rho,
        mu=config.mu,
        alpha_velocity=config.alpha_velocity,
        momentum_sweeps=sweeps,
    )


def timing(run_number: int) -> None:
    """predict on the 320x120 product state, core against reference, in one process."""
    import _cfdcore

    before = probe90.windows_load()
    calibration = probe90.calibration()
    config, call = product_call()
    mesh = Mesh(config)
    boundary = StaggeredBoundary(mesh, config)
    reference = MomentumPredictor(mesh, config, boundary)
    core = core_mapping.momentum_predictor(mesh, config, boundary)
    args = arguments(call)
    got = dict(zip(OUTPUTS, core.predict(*args), strict=True))
    if any(differing(call[k], got[k]) for k in OUTPUTS):
        raise SystemExit("the core's predict is not the reference's on this state")
    ref = probe90.timed(lambda: reference.predict(*args), REFERENCE_REPEATS)
    own = probe90.timed(lambda: core.predict(*args), CORE_REPEATS)
    # The parts: one sweep and the reference's other call of the stage.
    one_ref = MomentumPredictor(mesh, with_sweeps(config, 1), boundary)
    one_core = core_mapping.momentum_predictor(mesh, with_sweeps(config, 1), boundary)
    parts = {
        "one_sweep": {
            "reference": probe90.timed(
                lambda: one_ref.predict(*args), REFERENCE_REPEATS
            ),
            "core": probe90.timed(lambda: one_core.predict(*args), CORE_REPEATS),
        },
        "scalar_path": {
            "reference": probe90.timed(
                lambda: reference.predict(*args[:3]), REFERENCE_REPEATS
            ),
            "core": probe90.timed(lambda: core.predict(*args[:3]), CORE_REPEATS),
        },
        "stencil_viscosity": {
            "reference": probe90.timed(
                lambda: reference.stencil_viscosity(args[3]), REFERENCE_REPEATS
            ),
            "core": probe90.timed(
                lambda: core.stencil_viscosity(args[3]), CORE_REPEATS
            ),
        },
    }
    record = {
        "run": run_number,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "state": str(PRODUCT320.relative_to(ROOT)),
        "shape": list(call["p"].shape),
        "sweeps": int(config.momentum_sweeps),
        "module": str(_cfdcore.__file__),
        "windows_before": before,
        "calibration_before": calibration,
        "reference": ref,
        "core": own,
        "ratio": ref["median_s"] / own["median_s"],
        "parts": parts,
        "calibration_after": probe90.calibration(),
        "windows_after": probe90.windows_load(),
    }
    (RESULTS / f"timing_{run_number}.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )
    print(
        f"reference {ref['median_s'] * 1e3:.3f} ms, core {own['median_s'] * 1e3:.3f} ms, "
        f"ratio {record['ratio']:.2f}"
    )
    for key, part in parts.items():
        ratio = part["reference"]["median_s"] / part["core"]["median_s"]
        print(
            f"  {key}: {part['reference']['median_s'] * 1e3:.3f} ms, "
            f"{part['core']['median_s'] * 1e3:.3f} ms, {ratio:.2f}"
        )


# ---------------------------------------------------------------------------
# The forms of the kernels

FORM_WORK = Path.home() / "scratch92" / "forms"
GTEST_SRC = Path.home() / "build" / "cfd-core" / "_deps" / "googletest-src"


def sources_of(edits: list[tuple[str, str, str]]) -> dict[str, str]:
    """The checkout's core/ with edits made, path to text with LF line ends."""
    files = {
        path.relative_to(ROOT / "core").as_posix(): path.read_text(
            encoding="utf-8"
        ).replace("\r\n", "\n")
        for path in sorted((ROOT / "core").rglob("*"))
        if path.is_file()
    }
    for file, old, new in edits:
        if files[file].count(old) != 1:
            raise SystemExit(f"{file}: found {files[file].count(old)} times: {old!r}")
        files[file] = files[file].replace(old, new)
    return files


def build(work: Path, edits: list[tuple[str, str, str]], tests: bool) -> str | None:
    """Build a copy of core/ with edits made; None, or the compiler's last words."""
    source, tree = work / "core", work / "build"
    for relative, text in sources_of(edits).items():
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # Written only when it differs, so ninja rebuilds what changed.
        if not target.exists() or target.read_text(encoding="utf-8") != text:
            target.write_text(text, encoding="utf-8")
    configure = subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(tree),
            "-G",
            "Ninja",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST={GTEST_SRC}",
            f"-DCFD_CORE_TESTS={'ON' if tests else 'OFF'}",
        ],
        capture_output=True,
        text=True,
    )
    made = subprocess.run(["ninja", "-C", str(tree)], capture_output=True, text=True)
    if configure.returncode or made.returncode:
        return (configure.stderr + made.stdout + made.stderr)[-1500:]
    return None


def time_form() -> None:
    """One process of ``forms``: the imported build's predict, checked and timed."""
    import _cfdcore

    with np.load(RESULTS / "forms_call.npz") as z:
        call = {key: z[key] for key in z.files}
    config = case_config("product320")
    mesh = Mesh(config)
    boundary = StaggeredBoundary(mesh, config)
    args = arguments(call)
    out: dict[str, object] = {"module": str(_cfdcore.__file__)}
    core = core_mapping.momentum_predictor(mesh, config, boundary)
    got = dict(zip(OUTPUTS, core.predict(*args), strict=True))
    out["differ"] = {k: differing(call[k], got[k]) for k in OUTPUTS}
    one = core_mapping.momentum_predictor(mesh, with_sweeps(config, 1), boundary)
    out["predict"] = probe90.timed(lambda: core.predict(*args), FORM_REPEATS)
    out["one_sweep"] = probe90.timed(lambda: one.predict(*args), FORM_REPEATS)
    out["scalar_path"] = probe90.timed(lambda: core.predict(*args[:3]), FORM_REPEATS)
    out["scalar_one_sweep"] = probe90.timed(
        lambda: one.predict(*args[:3]), FORM_REPEATS
    )
    out["stencil_viscosity"] = probe90.timed(
        lambda: core.stencil_viscosity(args[3]), FORM_REPEATS
    )
    print(json.dumps(out))


def forms(run_number: int, only: list[str]) -> None:
    """Each form of variants92.FORMS built and timed, the head first and last."""
    import variants92

    _config, call = product_call()
    np.savez(RESULTS / "forms_call.npz", **call)
    names = [f for f in variants92.FORMS if not only or f[0] in only]
    failed = {}
    for name, _what, edits in names:
        failed[name] = build(FORM_WORK / name, edits, tests=False)
        if failed[name]:
            print(name, "build failed", failed[name])
    order = [n for n, _w, _e in names if not failed[n]]
    record: dict[str, object] = {
        "run": run_number,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "windows_before": probe90.windows_load(),
        "what": {n: w for n, w, _e in names},
        "rounds": [],
    }
    # Three rounds, the order reversed in the second, so a drift shows.
    for number in range(3):
        rows = {}
        for name in order if number != 1 else list(reversed(order)):
            env = {
                **os.environ,
                "PYTHONPATH": str(FORM_WORK / name / "build"),
                "OPENBLAS_NUM_THREADS": "1",
            }
            done = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "time-form"],
                capture_output=True,
                text=True,
                cwd=ROOT,
                env=env,
            )
            if done.returncode:
                raise SystemExit(done.stderr)
            rows[name] = json.loads(done.stdout.splitlines()[-1])
        record["rounds"].append(rows)
    record["windows_after"] = probe90.windows_load()
    (RESULTS / f"forms_{run_number}.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )
    for name in order:
        parts = {
            key: statistics.median(r[name][key]["median_s"] for r in record["rounds"])
            for key in (
                "predict",
                "one_sweep",
                "scalar_path",
                "scalar_one_sweep",
                "stencil_viscosity",
            )
        }
        differ = record["rounds"][0][name]["differ"]
        print(
            f"{name:32s}",
            " ".join(f"{k} {v * 1e6:8.1f}" for k, v in parts.items()),
            "differs" if any(differ.values()) else "",
        )


# ---------------------------------------------------------------------------
# The vectorisation report


VEC_FLAGS = ["-std=c++20", "-O2", "-ffp-contract=off", "-march=x86-64-v3", "-DNDEBUG"]


def vec_report(text: str) -> dict[str, object]:
    """GCC's vectorisation report for one text of momentum.cpp, its loops counted by place.

    The report names a loop once for each inlined copy it makes, so the count
    is of distinct source places (prompt 89's T8); the ``for (k < N)`` loops of
    the file are the blocks' and are listed with whether each is among them.
    """
    source = FORM_WORK / "vecinfo" / "momentum.cpp"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(text, encoding="utf-8")
    done = subprocess.run(
        [
            "g++",
            *VEC_FLAGS,
            f"-I{ROOT / 'core' / 'include'}",
            "-c",
            str(source),
            "-o",
            "/dev/null",
            "-fopt-info-vec-optimized",
        ],
        capture_output=True,
        text=True,
    )
    report = done.stdout + done.stderr
    found = re.findall(
        r"momentum\.cpp:(\d+):\d+: optimized: +loop vectorized using (\d+) byte",
        report,
    )
    places = sorted({int(line) for line, _width in found})
    blocks = []
    name = ""
    for number, line in enumerate(text.splitlines(), 1):
        match = re.match(r"inline \w+ (\w+)\(lanes::cells<N>", line)
        if match:
            name = match.group(1)
        if "for (std::size_t k = 0; k < N; ++k)" in line:
            blocks.append(
                {"block": name, "line": number, "vectorised": number in places}
            )
    return {
        "report_lines": len(found),
        "widths": dict(Counter(width for _line, width in found)),
        "places": places,
        "blocks": blocks,
    }


def vecinfo() -> None:
    """The vectorisation report of momentum.cpp, and of the np_maximum form of it."""
    import variants92

    version = subprocess.run(["g++", "--version"], capture_output=True, text=True)
    edits = next(e for n, _w, e in variants92.FORMS if n == "np_maximum")
    record = {
        "flags": VEC_FLAGS,
        "compiler": version.stdout.splitlines()[0],
        **vec_report(sources_of([])["src/momentum.cpp"]),
        "np_maximum": vec_report(sources_of(edits)["src/momentum.cpp"]),
    }
    (CURRENT / "vecinfo.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )
    for block in record["blocks"]:
        print(block["block"], block["line"], block["vectorised"])
    print(record["report_lines"], "report lines", record["widths"])
    left = [b["block"] for b in record["np_maximum"]["blocks"] if not b["vectorised"]]
    print("np_maximum leaves scalar:", left)


# ---------------------------------------------------------------------------
# The kernels of two commits (prompt 92b)

KERNEL_WORK = Path.home() / "scratch92" / "kernels"
# Where the passes' arrays fall on the heap, one placement a round.
PLACEMENTS = range(8)


def kernels() -> None:
    """Each kernel and pass timed in the commit before the restructure and in this one."""
    before = KERNEL_WORK / "before"
    before.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-x", "-C", str(before), "-f", str(BEFORE)], check=True)
    for name, core, flag in (
        ("before", before / "core", ["-DHEAD_FORM"]),
        ("after", ROOT / "core", []),
    ):
        subprocess.run(
            [
                "g++",
                *VEC_FLAGS,
                "-w",
                *flag,
                f"-I{core / 'include'}",
                f"-I{core / 'src'}",
                str(HERE / "kernels92b.cpp"),
                *(
                    str(ROOT / "core" / "src" / f"{part}.cpp")
                    for part in ("grid", "layout", "scalar", "reduce")
                ),
                "-o",
                str(KERNEL_WORK / f"kernels_{name}"),
            ],
            check=True,
        )
    record: dict[str, object] = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "windows_before": probe90.windows_load(),
        "flags": VEC_FLAGS,
        "rounds": [],
    }
    for place in PLACEMENTS:
        row = {}
        order = ("before", "after") if place % 2 == 0 else ("after", "before")
        for name in order:
            done = subprocess.run(
                [str(KERNEL_WORK / f"kernels_{name}"), str(place)],
                capture_output=True,
                text=True,
                check=True,
            )
            row[name] = json.loads(done.stdout)
        record["rounds"].append(row)
    record["windows_after"] = probe90.windows_load()
    (CURRENT / "kernels.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )
    for key in record["rounds"][0]["before"]:
        old = statistics.median(r["before"][key] for r in record["rounds"])
        new = statistics.median(r["after"][key] for r in record["rounds"])
        print(f"{key:34s} {old:9.2f} {new:9.2f} {new / old:.3f}")


# ---------------------------------------------------------------------------
# Planted defects

PLANT_WORK = Path.home() / "scratch92" / "plants"


def plants(stream: int, streams: int) -> None:
    """Each planted defect built in turn, with the tests and the replay that fail on it."""
    import variants92

    work = PLANT_WORK / f"s{stream}"
    mine = [
        plant
        for i, plant in enumerate(variants92.PLANTS)
        if i % streams == stream or plant[0] == "control"
    ]
    records = []
    for name, what, edits in mine:
        record: dict[str, object] = {"name": name, "what": what, "stream": stream}
        failed = build(work, edits, tests=True)
        if failed:
            record["build"] = failed
            records.append(record)
            print(name, "build failed", failed[-400:])
            continue
        tree = work / "build"
        ctest = subprocess.run(
            ["ctest", "--test-dir", str(tree)], capture_output=True, text=True
        )
        summary = [line for line in ctest.stdout.splitlines() if "tests passed" in line]
        record["ctest"] = summary[0].strip() if summary else ctest.stdout[-300:]
        # A test that crashes is listed as (SEGFAULT) or (Subprocess aborted), not (Failed).
        record["ctest_failed"] = re.findall(r"\d+ - (\S+) \([^)]+\)", ctest.stdout)
        env = {
            **os.environ,
            "PYTHONPATH": str(tree),
            "CFD_CORE_REQUIRED": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        tested = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_core_momentum.py",
                "-q",
                "-rf",
                "--tb=no",
                "-p",
                "no:cacheprovider",
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            env=env,
        )
        lines = tested.stdout.splitlines()
        record["pytest_returncode"] = tested.returncode
        record["pytest"] = next(
            (
                line.strip("= ")
                for line in reversed(lines)
                if re.search(r" in [0-9.]+s", line)
            ),
            lines[-1] if lines else "no output",
        )
        bad = [
            line.split("::", 1)[1].split(" ")[0]
            for line in lines
            if line.startswith("FAILED tests/test_core_momentum.py::")
        ]
        record["parity_failed_by_test"] = dict(
            sorted(Counter(case.split("[")[0] for case in bad).items())
        )
        replayed = subprocess.run(
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "replay",
                "--out",
                f"plant_{name}",
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            env=env,
        )
        record["replay"] = [
            line for line in replayed.stdout.splitlines() if " calls," in line
        ]
        records.append(record)
        print(
            name,
            "|",
            record["ctest"],
            "|",
            record["pytest"],
            "|",
            "; ".join(record["replay"]),
            flush=True,
        )
        (CURRENT / f"plants_s{stream}.json").write_text(
            json.dumps(records, indent=1) + "\n", encoding="utf-8"
        )


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run one command of the probe."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("capture").add_argument("name", choices=CASES)
    commands.add_parser("replay").add_argument("--out", default="replay")
    commands.add_parser("timing").add_argument("--run", type=int, required=True)
    former = commands.add_parser("forms")
    former.add_argument("--run", type=int, required=True)
    former.add_argument("--only", nargs="*", default=[])
    commands.add_parser("time-form")
    commands.add_parser("vecinfo")
    commands.add_parser("kernels")
    planter = commands.add_parser("plants")
    planter.add_argument("--stream", type=int, default=0)
    planter.add_argument("--streams", type=int, default=1)
    args = parser.parse_args(argv)
    commands_run: dict[str, Callable[[], None]] = {
        "capture": lambda: capture(args.name),
        "replay": lambda: replay(args.out),
        "timing": lambda: timing(args.run),
        "forms": lambda: forms(args.run, args.only),
        "time-form": time_form,
        "vecinfo": vecinfo,
        "kernels": kernels,
        "plants": lambda: plants(args.stream, args.streams),
    }
    commands_run[args.command]()
    return 0


if __name__ == "__main__":
    with contextlib.suppress(BrokenPipeError):
        raise SystemExit(main())
