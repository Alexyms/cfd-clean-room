"""Prompt 97b's probe: the Jacobi sweep counts behind the 320x120 spread.

    python docs/reports/probe97/sweeps97.py configs
    python docs/reports/probe97/sweeps97.py run <tag>
    python docs/reports/probe97/sweeps97.py analyse
    python docs/reports/probe97/sweeps97.py queue
    python docs/reports/probe97/sweeps97.py status

Run under WSL2 through ``docs/reports/probe97/run97.sh``, as ``probe97.py`` is.

- ``run``: one instrumented run of the reference's march (``RUNS``: the dust
  release as it is and with ``diffusion_tol`` times 0.99 on 320x120 and 160x60,
  the door leak as it is on 320x120), through ``run_scenario.main`` in this
  process with its flow states from its run of record (``--flows-from``), so
  that only the march is computed. The instrument records, for every step and
  class, the Jacobi sweeps, what stopped the solve, the stopping residual over
  the max-norm threshold (``diffusion_tol`` times the largest right-hand side),
  the first residual over it, the leftover over its bound and a digest of the
  field the step returns, into ``results/builder97/sweeps/<tag>.npz``. A base
  run whose variant has run takes the variant's counts and, at every class-step
  whose count differs (a flip), solves the same step from the same state again
  with one sweep fewer and one more, and records the difference each makes.
- ``analyse``: prediction (f) (each instrumented run against its run of record,
  bit for bit), the counts that differ by grid, class and time, (g) (the first
  differing count against the first differing field, per class), (h), the
  effect of one flip, and the distributions of the stopping residual, into
  ``results/builder97/sweeps/analysis.json``.
- ``queue``: the detached runner for prompt 97b: these runs and the analysis,
  then every job of ``probe97.py`` not yet done, under prompt 97's limits (three
  solver processes, two on 320x120, one comparison beside them, the timing last
  and alone) and its stop rules, with the stop on (f) or (g) added. It writes
  ``results/builder97/HALT`` on a stop and refuses to start while it exists.

The instrument changes nothing the solve computes. It patches, in this process
only, ``scalar_scheme._jacobi`` with a function that calls the original with
the solve's own arguments and an acceptance object that hands every call to the
solve's own (``_Watch``); ``TransportSolver.solve_timestep`` with one that
records the class and the field it returns; and ``_Run.observe`` with one that
records the step's time. The flip experiment works on copies. Prediction (f)
checks the result: each instrumented run is compared with its run of record.
"""

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np  # noqa: E402 -- the thread variables are set first
import yaml  # noqa: E402 -- the thread variables are set first

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import probe97  # noqa: E402 -- follows sys.path.insert

from src import scalar_scheme, solver_transport, time_integration  # noqa: E402
from src.config import SimConfig  # noqa: E402

# probe97 puts scripts/ on the path and imports the comparator.
compare_runs = probe97.compare_runs
ROOT = probe97.ROOT
SWEEPS = probe97.RESULTS / "sweeps"
LINUX_SWEEPS = Path.home() / "scratch97" / "sweeps"
ANALYSIS = SWEEPS / "analysis.json"
RELEASE_END = 40.0
# The flips the analysis tabulates one by one: at least 20 on 320x120 (prompt 97b),
# spread evenly over each run's flips; every flip is measured and summarised.
SAMPLE = 24
# Bins of time for the counts, seconds.
BINS = (0.0, 20.0, 40.0, 45.0, 50.0, 60.0, 80.0, 100.0, 120.0)
# The ratio above which a base stop fails the variant's tighter test from the same state.
FLIP_RATIO = probe97.DIFFUSION_FACTOR
CAUSES = ("max-norm", "leftover bound", "cap", "no solve")
BASE, VAR = "base", "var"
# The grids the dust release is paired on, the product's first.
GRIDS = (probe97.PRODUCT, probe97.MIDDLE)


@dataclass(frozen=True)
class Spec:
    """One instrumented run: its config, run of record, flow source and the run it pairs with."""

    tag: str
    name: str
    side: str
    config: Path
    of_record: Path | None
    flows_from: Path
    flips_from: str | None
    product: bool


def specs() -> dict[str, Spec]:
    """The five instrumented runs."""
    out = {}
    for grid in (probe97.PRODUCT, probe97.MIDDLE):
        name = probe97.run_name("dust", grid)
        source = probe97.linux_path(probe97.REF, name)
        var_config = (
            probe97.config_path(probe97.DIFFUSION, name)
            if grid == probe97.PRODUCT
            else SWEEPS / "configs" / f"{name}.var.yaml"
        )
        var_record = (
            probe97.run_file(probe97.DIFFUSION, name)
            if grid == probe97.PRODUCT
            else None
        )
        out[f"{name}.{VAR}"] = Spec(
            f"{name}.{VAR}",
            name,
            VAR,
            var_config,
            var_record,
            probe97.linux_path(probe97.DIFFUSION, name)
            if grid == probe97.PRODUCT
            else source,
            None,
            grid == probe97.PRODUCT,
        )
        out[f"{name}.{BASE}"] = Spec(
            f"{name}.{BASE}",
            name,
            BASE,
            probe97.config_path(probe97.REF, name),
            probe97.run_file(probe97.REF, name),
            source,
            f"{name}.{VAR}",
            grid == probe97.PRODUCT,
        )
    door = probe97.run_name("door", probe97.PRODUCT)
    out[f"{door}.{BASE}"] = Spec(
        f"{door}.{BASE}",
        door,
        BASE,
        probe97.config_path(probe97.REF, door),
        probe97.run_file(probe97.REF, door),
        probe97.linux_path(probe97.REF, door),
        None,
        True,
    )
    return out


def write_configs() -> None:
    """The 160x60 variant's scenario file, the one run with no file of prompt 97's."""
    name = probe97.run_name("dust", probe97.MIDDLE)
    raw, changed = probe97.mapping_of("dust", probe97.MIDDLE, probe97.DIFFUSION)
    path = specs()[f"{name}.{VAR}"].config
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    path.with_suffix(".changed.json").write_text(json.dumps(changed) + "\n")
    SimConfig(path)
    print("wrote", path)


def record_path(tag: str) -> Path:
    """An instrumented run's record, which marks it done."""
    return SWEEPS / f"{tag}.json"


def arrays_path(tag: str) -> Path:
    """An instrumented run's per class-step arrays."""
    return SWEEPS / f"{tag}.npz"


# ---------------------------------------------------------------------------
# The instrument


def digest(c: np.ndarray) -> int:
    """Eight bytes of BLAKE2b over the field's bytes, as an unsigned integer."""
    raw = hashlib.blake2b(np.ascontiguousarray(c).tobytes(), digest_size=8).digest()
    return int.from_bytes(raw, "little")


class _Watch:
    """The solve's acceptance object, every call handed on; records what the solve asked."""

    def __init__(self, inner: object) -> None:
        self.inner = inner
        self.tests: list[bool] = []
        self.final_max = 0.0

    def test(self, residual: np.ndarray) -> bool:
        """The solve's own test, its answer recorded."""
        ok = self.inner.test(residual)
        self.tests.append(bool(ok))
        return ok

    def final(self, residual: np.ndarray) -> None:
        """The solve's own final, the returned field's largest residual recorded."""
        self.inner.final(residual)
        self.final_max = float(np.abs(residual).max())


def _iterates(
    c_star: np.ndarray,
    a_p: np.ndarray,
    b: np.ndarray,
    g_u: np.ndarray,
    g_v: np.ndarray,
    solid: np.ndarray,
    held: np.ndarray | None,
    wanted: tuple[int, ...],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """The Jacobi iterates WANTED and their residuals, by ``scalar_scheme._jacobi``'s arithmetic."""
    c = c_star.copy()
    padded = np.pad(c, 1)
    out = {}
    for sweep in range(max(wanted) + 1):
        padded[1:-1, 1:-1] = c
        residual = b + scalar_scheme._neighbour_sum(padded, g_u, g_v) - a_p * c
        residual[solid] = 0.0
        if held is not None:
            residual[held] = 0.0
        if sweep in wanted:
            out[sweep] = (c, residual)
        if sweep == max(wanted):
            break
        c = c + residual / a_p
        c[solid] = 0.0
    return out


class Instrument:
    """The patches and what they record, per class in the order of the steps.

    Parameters
    ----------
    thresholds : list[float]
        Each class's alert threshold, for the pointwise measure's floor.
    flips_against : dict[int, np.ndarray] or None
        The variant's sweeps per class and step; a base run's flip experiment
        runs at every class-step whose count differs from it.
    """

    COLUMNS = (
        "sweeps",
        "cause",
        "tests",
        "ratio",
        "ratio0",
        "left_ratio",
        "dt",
        "digest",
    )

    def __init__(
        self, thresholds: list[float], flips_against: dict[int, np.ndarray] | None
    ) -> None:
        self.thresholds = thresholds
        self.flips_against = flips_against
        self.rows: dict[int, dict[str, list]] = {}
        self.times: list[tuple[int, float]] = []
        self.flips: list[dict] = []
        self._k: int | None = None
        self._dt = 0.0
        self._solver: object = None
        self._pending: dict | None = None
        self._saved: dict = {}

    def install(self) -> None:
        """Patch the three functions in this process."""
        self._saved = {
            "jacobi": scalar_scheme._jacobi,
            "solve": solver_transport.TransportSolver.solve_timestep,
            "observe": time_integration._Run.observe,
        }
        jacobi, solve, observe = (
            self._saved["jacobi"],
            self._saved["solve"],
            self._saved["observe"],
        )
        me = self

        def watched_jacobi(
            c_star, a_p, b, g_u, g_v, solid, held, stop, max_iter, accept=None
        ):
            if me._k is None or accept is None:
                return jacobi(
                    c_star, a_p, b, g_u, g_v, solid, held, stop, max_iter, accept
                )
            first = _iterates(c_star, a_p, b, g_u, g_v, solid, held, (0,))[0][1]
            watch = _Watch(accept)
            c, sweeps, converged = jacobi(
                c_star, a_p, b, g_u, g_v, solid, held, stop, max_iter, watch
            )
            me._after_solve(
                c_star,
                a_p,
                b,
                g_u,
                g_v,
                solid,
                held,
                stop,
                c,
                sweeps,
                converged,
                watch,
                float(np.abs(first).max()),
            )
            return c, sweeps, converged

        def watched_solve(solver, C_k, faces, size_class, dt, *args, **kwargs):  # noqa: N803 -- the solver's own name
            me._k, me._dt, me._solver, me._pending = size_class, float(dt), solver, None
            try:
                out = solve(solver, C_k, faces, size_class, dt, *args, **kwargs)
            finally:
                me._k = None
            me._after_step(size_class, out)
            return out

        def watched_observe(run, step, t, fields, pending):
            # The march observes the initial fields as step 0; the rows are of steps 1 on.
            if step > 0:
                me.times.append((int(step), float(t)))
            return observe(run, step, t, fields, pending)

        scalar_scheme._jacobi = watched_jacobi
        solver_transport.TransportSolver.solve_timestep = watched_solve
        time_integration._Run.observe = watched_observe

    def uninstall(self) -> None:
        """Put the three functions back."""
        scalar_scheme._jacobi = self._saved["jacobi"]
        solver_transport.TransportSolver.solve_timestep = self._saved["solve"]
        time_integration._Run.observe = self._saved["observe"]

    def _after_solve(
        self,
        c_star: np.ndarray,
        a_p: np.ndarray,
        b: np.ndarray,
        g_u: np.ndarray,
        g_v: np.ndarray,
        solid: np.ndarray,
        held: np.ndarray | None,
        stop: float,
        c: np.ndarray,
        sweeps: int,
        converged: bool,
        watch: _Watch,
        first_max: float,
    ) -> None:
        """One solve's record, and the flip experiment where the variant's count differs."""
        k = self._k
        if not converged:
            cause = 2
        elif len(watch.tests) == 1:
            cause = 0
        else:
            cause = 1
        bound = watch.inner._bound
        leftover = watch.inner.leftover
        self._pending = {
            "sweeps": sweeps,
            "cause": cause,
            "tests": len(watch.tests),
            # NaN where the field is zero (before the release), so the threshold is.
            "ratio": watch.final_max / stop if stop > 0.0 else np.nan,
            "ratio0": first_max / stop if stop > 0.0 else np.nan,
            "left_ratio": abs(leftover) / bound if bound > 0.0 else np.nan,
        }
        if self.flips_against is None:
            return
        index = len(self.rows.get(k, {}).get("sweeps", []))
        other = int(self.flips_against[k][index])
        if other == sweeps:
            return
        wanted = tuple(s for s in (sweeps - 1, sweeps, sweeps + 1) if s >= 0)
        its = _iterates(c_star, a_p, b, g_u, g_v, solid, held, wanted)
        c_n, r_n = its[sweeps]
        solver = self._solver
        supplied = bound * solver._horizon / (solver._share * solver._limit * self._dt)
        floor = compare_runs.POINTWISE_FLOOR * self.thresholds[k]
        flip = {
            "k": k,
            "index": index,
            "sweeps": sweeps,
            "variant_sweeps": other,
            "cause": cause,
            "ratio": self._pending["ratio"],
            "replica_equal": bool(np.array_equal(c_n, c)),
            "largest": float(np.abs(c_n).max()),
        }
        left_n = self._dt * float(np.sum(r_n))
        for label, s in (("fewer", sweeps - 1), ("more", sweeps + 1)):
            if s not in its:
                flip[label] = None
                continue
            c_s, r_s = its[s]
            flip[label] = {
                "relative": compare_runs.relative(c_n, c_s),
                "pointwise": compare_runs.pointwise(c_n, c_s, floor),
                "leftover_change_over_supply": (self._dt * float(np.sum(r_s)) - left_n)
                / supplied,
            }
        self.flips.append(flip)

    def _after_step(self, k: int, out: np.ndarray) -> None:
        """One class-step's row."""
        rows = self.rows.setdefault(k, {c: [] for c in self.COLUMNS})
        pending = self._pending or {
            "sweeps": 0,
            "cause": 3,
            "tests": 0,
            "ratio": 0.0,
            "ratio0": 0.0,
            "left_ratio": 0.0,
        }
        for key, value in pending.items():
            rows[key].append(value)
        rows["dt"].append(self._dt)
        rows["digest"].append(digest(out))
        self._pending = None

    def save(self, path: Path) -> None:
        """The arrays, one per class and column, and the steps' times."""
        arrays = {
            "step": np.array([s for s, _ in self.times], dtype=np.int64),
            "t": np.array([t for _, t in self.times], dtype=np.float64),
        }
        types = {
            "sweeps": np.int32,
            "cause": np.int8,
            "tests": np.int32,
            "digest": np.uint64,
        }
        for k, rows in self.rows.items():
            for column, values in rows.items():
                arrays[f"{column}_{k}"] = np.array(
                    values, dtype=types.get(column, np.float64)
                )
        lengths = {len(v) for key, v in arrays.items()}
        if len(lengths) != 1:
            raise RuntimeError(f"the records' lengths differ: {sorted(lengths)}")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)


def load_arrays(tag: str) -> dict[str, np.ndarray]:
    """An instrumented run's arrays."""
    with np.load(arrays_path(tag)) as data:
        return {key: data[key] for key in data.files}


def classes_of(arrays: dict[str, np.ndarray]) -> list[int]:
    """The classes an instrumented run recorded."""
    return sorted(int(key.split("_")[1]) for key in arrays if key.startswith("sweeps_"))


def run_one(tag: str) -> int:
    """One instrumented run, its arrays, flips and record."""
    import run_scenario

    spec = specs()[tag]
    config = SimConfig(spec.config)
    thresholds = [float(x) for x in config.class_thresholds("sweeps97")]
    against = None
    if spec.flips_from is not None:
        other = load_arrays(spec.flips_from)
        against = {k: other[f"sweeps_{k}"] for k in classes_of(other)}
    path = LINUX_SWEEPS / f"{tag}.h5"
    probe97.fresh(path)
    args = [
        str(spec.config),
        "--out",
        str(path),
        "--concurrent",
        str(probe97.concurrent()),
    ]
    args += ["--flows-from", str(spec.flows_from)]
    instrument = Instrument(thresholds, against)
    instrument.install()
    started = time.perf_counter()
    try:
        code = run_scenario.main(args)
    finally:
        instrument.uninstall()
    wall = time.perf_counter() - started
    instrument.save(arrays_path(tag))
    (SWEEPS / f"{tag}.flips.json").write_text(
        json.dumps(compare_runs._plain(instrument.flips)) + "\n", encoding="utf-8"
    )
    record = {
        "tag": tag,
        "exit": code,
        "wall_seconds": wall,
        "config": str(spec.config.relative_to(ROOT)),
        "of_record": None
        if spec.of_record is None
        else str(spec.of_record.relative_to(ROOT)),
        "flows_from": str(spec.flows_from),
        "flips_from": spec.flips_from,
        "flips": len(instrument.flips),
        "class_steps": {k: len(r["sweeps"]) for k, r in instrument.rows.items()},
        "steps": len(instrument.times),
        "diffusion_tol": config.transport.diffusion_tol,
        "max_diffusion_iter": config.transport.max_diffusion_iter,
        "thresholds": thresholds,
        "concurrent": probe97.concurrent(),
        "finished": datetime.now().isoformat(timespec="seconds"),
        **probe97.environment(),
    }
    probe97.finish(path, SWEEPS, tag, record)
    print(
        tag, "exit", code, f"{wall:.1f} s", "flips", len(instrument.flips), flush=True
    )
    return code


# ---------------------------------------------------------------------------
# The analysis


def _bins(t: np.ndarray, mask: np.ndarray) -> list[dict]:
    """Per bin of time, the class-steps and how many MASK marks."""
    out = []
    for lo, hi in itertools.pairwise(BINS):
        inside = (t > lo) & (t <= hi)
        out.append(
            {
                "from": lo,
                "to": hi,
                "class_steps": int(inside.sum()),
                "marked": int((mask & inside).sum()),
            }
        )
    return out


def _quantiles(values: np.ndarray) -> dict:
    """Count, the quantiles and the largest of VALUES."""
    if values.size == 0:
        return {"n": 0}
    q = np.quantile(values, (0.1, 0.5, 0.9, 0.99))
    return {
        "n": int(values.size),
        "q10": float(q[0]),
        "median": float(q[1]),
        "q90": float(q[2]),
        "q99": float(q[3]),
        "max": float(values.max()),
    }


def _distribution(arrays: dict[str, np.ndarray]) -> dict:
    """The stopping residual's distribution and the causes, before and after the release ends."""
    t = arrays["t"]
    out = {}
    for period, mask in (
        ("to 40 s", t <= RELEASE_END),
        ("after 40 s", t > RELEASE_END),
    ):
        ratio, sweeps, cause, ratio0, left = [], [], [], [], []
        for k in classes_of(arrays):
            left.append(arrays[f"left_ratio_{k}"][mask])
            ratio.append(arrays[f"ratio_{k}"][mask])
            sweeps.append(arrays[f"sweeps_{k}"][mask])
            cause.append(arrays[f"cause_{k}"][mask])
            ratio0.append(arrays[f"ratio0_{k}"][mask])
        ratio = np.concatenate(ratio)
        sweeps = np.concatenate(sweeps)
        cause = np.concatenate(cause)
        ratio0 = np.concatenate(ratio0)
        left = np.concatenate(left)
        read = np.isfinite(ratio) & (cause != 3)
        solved = read & (sweeps > 0) & (ratio > 0.0) & (ratio0 > 0.0)
        # The mean contraction per sweep, from the first residual to the stopping one.
        rho = np.full(ratio.shape, np.nan)
        rho[solved] = np.exp(np.log(ratio[solved] / ratio0[solved]) / sweeps[solved])
        out[period] = {
            "class_steps": int(ratio.size),
            "causes": {name: int((cause == i).sum()) for i, name in enumerate(CAUSES)},
            "read": int(read.sum()),
            "ratio": _quantiles(ratio[read]),
            "above_flip_ratio": int((ratio > FLIP_RATIO).sum()),
            "above_flip_ratio_by_max_norm": int(
                ((ratio > FLIP_RATIO) & (cause == 0)).sum()
            ),
            "sweeps": _quantiles(sweeps[read].astype(np.float64)),
            "contraction": _quantiles(rho[solved]),
            "first_ratio": _quantiles(ratio0[read]),
            "leftover_ratio": _quantiles(left[read & np.isfinite(left)]),
        }
    return out


def _first_frame_differing(a: Path, b: Path) -> dict[int, int | None]:
    """Per class, the step of the first stored frame at which the two files' fields differ."""
    from src.io_manager import RunFile

    fa, fb = RunFile(a).read_frames(), RunFile(b).read_frames()
    out = {}
    for k in sorted(fa.fields):
        x, y = fa.fields[k], fb.fields[k]
        differ = [n for n in range(x.shape[0]) if not np.array_equal(x[n], y[n])]
        out[k] = int(fa.steps[differ[0]]) if differ else None
    return out


def _flip_summary(flips: list[dict], t: np.ndarray) -> dict:
    """The effect of one flip, over every flip of a run, and a sample spread over the run."""
    out: dict = {
        "flips": len(flips),
        "replica_equal": all(f["replica_equal"] for f in flips),
    }
    for label in ("fewer", "more"):
        rows = [f[label] for f in flips if f[label] is not None]
        out[label] = {
            measure: _quantiles(
                np.array([abs(r[measure]) for r in rows], dtype=np.float64)
            )
            for measure in ("relative", "pointwise", "leftover_change_over_supply")
        }
    ordered = sorted(flips, key=lambda f: (f["index"], f["k"]))
    if ordered:
        picks = np.unique(
            np.linspace(0, len(ordered) - 1, min(SAMPLE, len(ordered))).astype(int)
        )
        out["sample"] = [
            {
                "k": ordered[i]["k"],
                "t": float(t[ordered[i]["index"]]),
                "sweeps": ordered[i]["sweeps"],
                "variant_sweeps": ordered[i]["variant_sweeps"],
                "cause": CAUSES[ordered[i]["cause"]],
                "ratio": ordered[i]["ratio"],
                "fewer": ordered[i]["fewer"],
                "more": ordered[i]["more"],
            }
            for i in picks
        ]
    else:
        out["sample"] = []
    return out


def analyse() -> int:
    """Predictions (f) to (h) and the measures prompt 97b asks for, into analysis.json."""
    the_specs = specs()
    out: dict = {"bitwise": {}, "grids": {}, "distributions": {}}
    # (f): each instrumented run against its run of record.
    for tag, spec in the_specs.items():
        if spec.of_record is None:
            continue
        comparison = compare_runs.compare(spec.of_record, SWEEPS / f"{tag}.h5")
        out["bitwise"][tag] = {
            "of_record": str(spec.of_record.relative_to(ROOT)),
            "flags": comparison.bitwise,
            "verdict": comparison.verdict,
        }
    scored = ("frames", "budgets", "sensors")
    out["f_holds"] = all(
        all(v["flags"].get(key, False) for key in scored)
        for v in out["bitwise"].values()
    )
    g_holds = True
    for grid in GRIDS:
        name = probe97.run_name("dust", grid)
        base, var = load_arrays(f"{name}.{BASE}"), load_arrays(f"{name}.{VAR}")
        t = base["t"]
        if not np.array_equal(t, var["t"]):
            raise SystemExit(f"{name}: the two runs' steps differ in time")
        frames = _first_frame_differing(
            SWEEPS / f"{name}.{BASE}.h5", SWEEPS / f"{name}.{VAR}.h5"
        )
        per_class, total, differing = {}, 0, 0
        all_flips = np.zeros(t.shape, dtype=bool)
        any_class = np.zeros(t.shape, dtype=np.int64)
        for k in classes_of(base):
            sb, sv = base[f"sweeps_{k}"], var[f"sweeps_{k}"]
            if sb.shape != sv.shape or sb.shape != t.shape:
                raise SystemExit(f"{name} class {k}: the records' lengths differ")
            flip = sb != sv
            field_differs = base[f"digest_{k}"] != var[f"digest_{k}"]
            first_flip = int(np.argmax(flip)) if flip.any() else None
            first_field = int(np.argmax(field_differs)) if field_differs.any() else None
            step_of = base["step"]
            frame_step = frames[k]
            holds = True
            if frame_step is not None:
                holds = (
                    first_flip is not None and int(step_of[first_flip]) <= frame_step
                )
            g_holds = g_holds and holds
            # A step's field differs first where its count differs (H1 by step).
            by_step = first_field is None or (
                first_flip is not None and first_flip <= first_field
            )
            per_class[k] = {
                "class_steps": int(flip.size),
                "differing": int(flip.sum()),
                "share": float(flip.mean()),
                "variant_more": int((sv > sb).sum()),
                "variant_fewer": int((sv < sb).sum()),
                "largest_difference": int(np.abs(sv.astype(np.int64) - sb).max()),
                "first_flip_step": None
                if first_flip is None
                else int(step_of[first_flip]),
                "first_flip_t": None if first_flip is None else float(t[first_flip]),
                "first_field_step": None
                if first_field is None
                else int(step_of[first_field]),
                "first_field_t": None if first_field is None else float(t[first_field]),
                "first_frame_step": frame_step,
                "g_holds": holds,
                "field_first_at_a_flip": by_step,
                "fields_differ_without_a_flip": int((field_differs & ~flip).sum()),
                "by_time": _bins(t, flip),
                "first_flip_before_any_field": bool(
                    first_flip is not None
                    and (first_field is None or first_flip <= first_field)
                ),
            }
            total += flip.size
            differing += int(flip.sum())
            all_flips |= flip
            any_class += flip
        flips = json.loads((SWEEPS / f"{name}.{BASE}.flips.json").read_text())
        out["grids"][grid] = {
            "run": name,
            "steps": int(t.size),
            "class_steps": total,
            "differing": differing,
            "share": differing / total,
            "steps_with_a_flip": int(all_flips.sum()),
            "classes": per_class,
            "by_time": _bins(t, all_flips),
            "flip_effect": _flip_summary(flips, t),
            "base_flips_recorded": len(flips),
        }
    out["g_holds"] = g_holds
    shares = [out["grids"][grid]["share"] for grid in GRIDS]
    out["h_holds"] = len(shares) == 2 and shares[0] > shares[1]
    for tag in the_specs:
        out["distributions"][tag] = _distribution(load_arrays(tag))
    out["finished"] = datetime.now().isoformat(timespec="seconds")
    out.update(probe97.environment())
    ANALYSIS.write_text(json.dumps(compare_runs._plain(out), indent=1) + "\n")
    print("f", out["f_holds"], "g", out["g_holds"], "h", out["h_holds"], flush=True)
    return 0


def analysis_halt() -> str | None:
    """Prompt 97b's stop: (f) or (g) failed."""
    record = json.loads(ANALYSIS.read_text())
    failed = [p for p in ("f", "g") if not record[f"{p}_holds"]]
    return f"sweeps97: prediction(s) {failed} failed" if failed else None


def run_halt(tag: str) -> str | None:
    """An instrumented run that did not finish."""
    record = json.loads(record_path(tag).read_text())
    return f"sweeps run {tag} exited {record['exit']}" if record["exit"] != 0 else None


# ---------------------------------------------------------------------------
# The runner


@dataclass
class Task:
    """A job of either script."""

    script: Path
    job: probe97.Job
    extra: list[Path] = field(default_factory=list)


def tasks() -> list[Task]:
    """Prompt 97b's jobs, then prompt 97's, in the order the runner prefers them."""
    me = Path(__file__).resolve()
    out = []
    the_specs = specs()
    order = [
        f"door_320x120.{BASE}",
        f"dust_320x120.{VAR}",
        f"dust_160x60.{VAR}",
        f"dust_320x120.{BASE}",
        f"dust_160x60.{BASE}",
    ]
    assert set(order) == set(the_specs)
    for tag in order:
        spec = the_specs[tag]
        needs = [] if spec.flips_from is None else [record_path(spec.flips_from)]
        out.append(
            Task(
                me,
                probe97.Job(
                    f"sweeps_{tag}",
                    ["run", tag],
                    record_path(tag),
                    needs,
                    product=spec.product,
                    halt=lambda t=tag: run_halt(t),
                ),
            )
        )
    out.append(
        Task(
            me,
            probe97.Job(
                "sweeps_analyse",
                ["analyse"],
                ANALYSIS,
                [record_path(tag) for tag in order],
                kind="compare",
                halt=analysis_halt,
            ),
        )
    )
    script97 = HERE / "probe97.py"
    # Prompt 97's remaining jobs fill the slots the probe leaves; the probe's come first.
    out.extend(Task(script97, job) for job in probe97.jobs())
    return out


def queue() -> int:
    """Start every job not done, within prompt 97's limits; stop on its rules and (f), (g)."""
    log_line = probe97.log_line
    if probe97.HALT.exists():
        log_line(f"refused: {probe97.HALT} exists: {probe97.HALT.read_text().strip()}")
        return 1
    every = tasks()
    waiting = [t for t in every if not t.job.done.exists()]
    running: dict[str, tuple[Task, subprocess.Popen]] = {}
    failed: list[str] = []
    halted: str | None = None
    probe97.LOGS.mkdir(parents=True, exist_ok=True)
    log_line(
        f"queue 97b at {probe97.commit()[:12]}: {len(waiting)} of {len(every)} jobs to do"
    )

    def count(kind: str, product: bool | None = None) -> int:
        return sum(
            1
            for task, _ in running.values()
            if task.job.kind == kind
            and (product is None or task.job.product == product)
        )

    while waiting or running:
        started = 0
        if halted is None:
            for task in list(waiting):
                job = task.job
                if not all(p.exists() for p in [*job.needs, *task.extra]):
                    continue
                if job.kind == "timing":
                    if running or any(t.job.kind != "timing" for t in waiting):
                        continue
                elif any(t.job.kind == "timing" for t, _ in running.values()):
                    continue
                if job.kind == "solver":
                    if count("solver") >= probe97.MAX_SOLVERS:
                        continue
                    if job.product and count("solver", True) >= probe97.MAX_PRODUCT:
                        continue
                if job.kind == "compare" and count("compare") >= 1:
                    continue
                environ = os.environ.copy()
                environ["PROBE97_CONCURRENT"] = str(count("solver"))
                with (probe97.LOGS / f"{job.name}.log").open("w") as out:
                    proc = subprocess.Popen(
                        [str(probe97.PYTHON), "-u", str(task.script), *job.args],
                        cwd=ROOT,
                        env=environ,
                        stdout=out,
                        stderr=subprocess.STDOUT,
                    )
                running[job.name] = (task, proc)
                waiting.remove(task)
                started += 1
                log_line(f"start {job.name} pid {proc.pid}")
                if job.kind == "timing":
                    break
            if not started and not running and waiting:
                log_line(f"blocked: {[t.job.name for t in waiting]}")
                break
        elif not running:
            break
        time.sleep(probe97.POLL_SECONDS)
        for name, (task, proc) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            del running[name]
            job = task.job
            reason = None
            if not job.done.exists():
                reason = f"{name} exited {code} without its record"
            elif job.halt is not None:
                try:
                    reason = job.halt()
                except Exception as err:
                    # A stop rule that cannot be read is a stop.
                    reason = f"{name}: the stop rules could not be read: {err!r}"
            if code != 0 or reason:
                failed.append(name)
            log_line(f"{'done' if code == 0 else 'failed'} {name} exit {code}")
            if reason and halted is None:
                halted = reason
                probe97.HALT.write_text(reason + "\n")
                log_line(f"HALT: {reason}; starting nothing more")
    log_line(
        f"queue 97b ended; failed: {failed if failed else 'none'}; "
        f"halt: {halted if halted else 'none'}; left: {[t.job.name for t in waiting]}"
    )
    return 1 if failed or halted else 0


def status() -> int:
    """Each job's state."""
    for task in tasks():
        job = task.job
        needs = [*job.needs, *task.extra]
        state = (
            "done"
            if job.done.exists()
            else ("ready" if all(p.exists() for p in needs) else "waiting")
        )
        print(f"{state:8s} {job.kind:8s} {job.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """The command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command")
    parser.add_argument("args", nargs="*")
    args = parser.parse_args(argv)
    if args.command == "configs":
        write_configs()
        return 0
    if args.command == "run":
        return run_one(args.args[0])
    if args.command == "analyse":
        return analyse()
    if args.command == "queue":
        return queue()
    if args.command == "status":
        return status()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
