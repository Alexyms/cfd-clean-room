"""Prompt 89's probe: the scalar kernels timed and checked build by build.

    python docs/reports/probe89/probe89.py build base
    python docs/reports/probe89/probe89.py check o2
    python docs/reports/probe89/probe89.py time o2 --run 1
    python docs/reports/probe89/probe89.py session --run 1
    python docs/reports/probe89/probe89.py plants
    python docs/reports/probe89/probe89.py vecinfo
    python docs/reports/probe89/probe89.py floor
    python docs/reports/probe89/probe89.py styles

Run under WSL2 with the Linux environment (``~/.venvs/cfd_clean_room``) from
the checkout's root, one BLAS thread, one timing process at a time. Nothing in
the checkout is built or changed: each build is a copy of ``core/`` on the
Linux file system (``~/scratch89/builds/<name>``), built out of source with the
main build's GoogleTest sources, and a timing or a check runs against that
copy's module, which the probe puts ahead of the installed one.

The builds (``BUILDS``):

- ``base``: ``core/`` as it was at ``origin/main`` before this pass, under the
  project's flags. It is "now" in the predictions. ``jacobi``, ``setup`` and
  ``cg``: ``core/`` after each kernel's commit, the flux's being the head. A
  commit's sources come from ``results/builder89/src_<commit>.tar`` (``git
  archive --format=tar -o results/builder89/src_<commit>.tar <commit> core``,
  run from Windows, where git is run).
- ``o2``: the checkout's ``core/`` under the project's flags.
- ``o3``, ``native``, ``o3native``: the checkout's ``core/`` with ``-O3`` in
  place of ``-O2``, with ``-march=native`` added, and with both; ``v3``: with
  ``-march=x86-64-v3`` added. The flags are changed in the copy's
  ``CMakeLists.txt``; the checkout's is not touched.
- the candidates of ``variants89.py``: the checkout's ``core/`` with one
  candidate's edits, under the project's flags, and the flux's block form
  under each other flag set as well (``flux_blocks_o3`` and so on).

The commands:

- ``build NAME``: copy, configure, build, and record the flags and the
  sources' hashes (``build_NAME.json``).
- ``check NAME``: CTest, the parity module and the captured replays of prompts
  87 and 88 against the build (``check_NAME.json`` and the JUnit records).
- ``time NAME --run N``: every kernel's timing on the developed frames of
  prompts 87 and 88 (``timing_NAME_N.json``). ``time reference --run N`` times
  the Python reference on the same inputs.
- ``session --run N``: ``time`` for the reference and every build, one process
  after another, with the busiest Windows processes sampled before each.
- ``plants``: each planted defect of ``variants89.py`` built in turn, with the
  GoogleTest and parity cases that fail on it (``plants.json``).
- ``vecinfo``: GCC's vectorisation report for ``scalar.cpp`` and ``reduce.cpp``
  of ``base`` and of the checkout, under each build's flags (``vecinfo.json``).
- ``floor``: build and run ``floor89.cpp``, the machine's rates the floor
  estimate reads (``floor.json``).
- ``styles``: build and run ``styles89.cpp``, one sweep written six ways, under
  each flag set (``styles.json``).
"""

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import tarfile
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "probe87"))
sys.path.insert(0, str(HERE.parent / "probe88"))

import variants89  # noqa: E402 -- after sys.path

RESULTS = ROOT / "results" / "builder89"
WORK = Path.home() / "scratch89" / "builds"
# The commits whose core/ is built: origin/main before this pass, and the
# commit that restructured each kernel but the last, which is the head.
COMMITS = {
    "base": "3c8d1ec",
    "jacobi": "101b1ad",
    "setup": "34e11fe",
    "cg": "104a02d",
}
GTEST_SRC = (
    Path(os.environ.get("CFD_CORE_BUILD", Path.home() / "build" / "cfd-core"))
    / "_deps"
    / "googletest-src"
)
# The project's flags as core/CMakeLists.txt states them, and the other sets.
PROJECT_FLAGS = "-Wall -Wextra -Wpedantic -O2 -ffp-contract=off"
FLAG_SETS = {
    "o2": PROJECT_FLAGS,
    "o3": PROJECT_FLAGS.replace("-O2", "-O3"),
    "native": PROJECT_FLAGS + " -march=native",
    "o3native": PROJECT_FLAGS.replace("-O2", "-O3") + " -march=native",
    # Not one of the prediction's sets: AVX2 by a name that means the same on
    # every machine, where -march=native means this one's processor.
    "v3": PROJECT_FLAGS + " -march=x86-64-v3",
}
# Each build: where its sources come from (a commit, or the checkout), the
# candidate whose edits it carries, and its flags.
BUILDS: dict[str, dict[str, str]] = {
    **{
        name: {"source": commit, "candidate": "", "flags": PROJECT_FLAGS}
        for name, commit in COMMITS.items()
    },
    **{
        name: {"source": "head", "candidate": "", "flags": flags}
        for name, flags in FLAG_SETS.items()
    },
    **{
        name: {"source": "head", "candidate": name, "flags": PROJECT_FLAGS}
        for name in variants89.CANDIDATES
    },
    **{
        f"flux_blocks_{name}": {
            "source": "head",
            "candidate": "flux_blocks",
            "flags": flags,
        }
        for name, flags in FLAG_SETS.items()
        if name != "o2"
    },
}
SOURCE_DIRECTORIES = ("src", "python", "include/cfdcore", "tests")
# The timing's repeats: the median of these many calls, after WARMUP unrecorded.
IMPLICIT_REPEATS = 200
FLUX_REPEATS = 400
REFERENCE_IMPLICIT_REPEATS = 20
REFERENCE_FLUX_REPEATS = 60
WARMUP = 5
# A stop no residual fails: the CG returns from its first residual.
CG_LOOSE_TOL = 1e300
# The calibration: np.sqrt over an array that fits the first-level cache.
CALIBRATION_LENGTH = 4096
CALIBRATION_REPEATS = 2000


def run_command(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
    """Run a command, its output captured as text."""
    return subprocess.run(command, capture_output=True, text=True, **kwargs)


def build_directory(name: str) -> Path:
    """Where a build's module and tests are."""
    return WORK / name / "build"


def apply_edits(files: dict[str, str], edits: list[variants89.Edit]) -> None:
    """Make each edit of a variant exactly once, or refuse the variant."""
    for edit in edits:
        file = edit[0]
        text = files[file]
        if len(edit) == 3:
            old, new = edit[1], edit[2]
            if text.count(old) != 1:
                raise SystemExit(f"{file}: found {text.count(old)} times: {old!r}")
            files[file] = text.replace(old, new)
            continue
        first, last, new = edit[1], edit[2], edit[3]
        if text.count(first) != 1 or text.count(last) != 1:
            raise SystemExit(
                f"{file}: a span's ends are not each found once: {first!r}"
            )
        begin, stop = text.index(first), text.index(last)
        if begin >= stop:
            raise SystemExit(f"{file}: a span ends before it starts: {first!r}")
        files[file] = text[:begin] + new + text[stop:]


def _source_files(name: str) -> dict[str, str]:
    """A build's sources, path under core/ to text with LF line ends, its edits made."""
    files: dict[str, str] = {}
    source = BUILDS[name]["source"]
    if source != "head":
        with tarfile.open(RESULTS / f"src_{source}.tar") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    handle = tar.extractfile(member)
                    if handle is None:
                        raise SystemExit(f"{member.name}: not readable in the tar")
                    text = handle.read().decode("utf-8").replace("\r\n", "\n")
                    files[member.name.removeprefix("core/")] = text
        return files
    for path in sorted((ROOT / "core").rglob("*")):
        if path.is_file():
            relative = path.relative_to(ROOT / "core").as_posix()
            files[relative] = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    candidate = BUILDS[name]["candidate"]
    if candidate:
        apply_edits(files, variants89.CANDIDATES[candidate]["edits"])
    return files


def build(name: str) -> None:
    """Copy a build's sources to the Linux file system, set its flags and build it."""
    spec = BUILDS[name]
    files = _source_files(name)
    cmake = files["CMakeLists.txt"]
    if cmake.count(PROJECT_FLAGS) != 1:
        raise SystemExit(
            "CMakeLists.txt: the project's flags are not found exactly once"
        )
    files["CMakeLists.txt"] = cmake.replace(PROJECT_FLAGS, spec["flags"])
    source = WORK / name / "core"
    for relative, text in files.items():
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # Written only when it differs, so ninja rebuilds what changed and no more.
        if not target.exists() or target.read_text(encoding="utf-8") != text:
            target.write_text(text, encoding="utf-8")
    for target in sorted(source.rglob("*")):
        if target.is_file() and target.relative_to(source).as_posix() not in files:
            target.unlink()
    out = build_directory(name)
    configure = run_command(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(out),
            "-G",
            "Ninja",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST={GTEST_SRC}",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        ]
    )
    if configure.returncode != 0:
        raise SystemExit(configure.stdout + configure.stderr)
    made = run_command(["ninja", "-C", str(out)])
    if made.returncode != 0:
        raise SystemExit(made.stdout[-6000:] + made.stderr[-2000:])
    warnings = sorted(set(re.findall(r"warning: .*", made.stdout + made.stderr)))
    commands = json.loads((out / "compile_commands.json").read_text(encoding="utf-8"))
    scalar = next(c["command"] for c in commands if c["file"].endswith("scalar.cpp"))
    record = {
        "name": name,
        "source": spec["source"],
        "candidate": spec["candidate"],
        "flags": spec["flags"],
        "scalar_command": scalar,
        "warnings": warnings,
        "sha256": {
            relative: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for relative, text in sorted(files.items())
            if relative.split("/")[0] in ("src", "python", "include")
        },
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"build_{name}.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )
    print(name, "built;", len(warnings), "warnings")


def use_build(name: str) -> str:
    """Put a build's module ahead of the installed one; its file's path."""
    sys.path.insert(0, str(build_directory(name)))
    import _cfdcore

    path = str(_cfdcore.__file__)
    if not path.startswith(str(build_directory(name))):
        raise SystemExit(f"_cfdcore came from {path}, not from the build {name}")
    return path


def _build_environment(name: str) -> dict[str, str]:
    """The environment of a subprocess that imports a build's module."""
    return {
        **os.environ,
        "PYTHONPATH": str(build_directory(name)),
        "CFD_CORE_REQUIRED": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }


def check(name: str) -> None:
    """CTest, the parity module and the captured replays against one build."""
    out = build_directory(name)
    junit = RESULTS / f"ctest_{name}.xml"
    ctest = run_command(["ctest", "--test-dir", str(out), "--output-junit", str(junit)])
    summary = [line for line in ctest.stdout.splitlines() if "tests passed" in line]
    record: dict[str, object] = {
        "name": name,
        "ctest": summary[0].strip() if summary else ctest.stdout[-500:],
        "ctest_failed": re.findall(r" - (\S+) \(Failed\)", ctest.stdout),
    }
    pytest = run_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_core_parity.py",
            "-q",
            "-rf",
            "-p",
            "no:cacheprovider",
            f"--junitxml={RESULTS / f'parity_{name}.xml'}",
        ],
        cwd=ROOT,
        env=_build_environment(name),
    )
    lines = pytest.stdout.splitlines()
    record["pytest_exit"] = pytest.returncode
    record["pytest"] = next(
        (
            line.strip("= ")
            for line in reversed(lines)
            if re.search(r" in [0-9.]+s", line)
        ),
        lines[-1] if lines else "",
    )
    failed = [
        line.split("::", 1)[1].split(" ")[0]
        for line in lines
        if line.startswith("FAILED tests/test_core_parity.py::")
    ]
    record["parity_failed_by_test"] = dict(
        sorted(Counter(case.split("[")[0] for case in failed).items())
    )
    replayed = run_command(
        [sys.executable, str(Path(__file__).resolve()), "replay", name],
        cwd=ROOT,
        env=_build_environment(name),
    )
    if replayed.returncode != 0:
        record["replay"] = "failed: " + (replayed.stdout + replayed.stderr)[-3000:]
    else:
        record["replay"] = json.loads(
            (RESULTS / f"replay_{name}.json").read_text(encoding="utf-8")
        )
    (RESULTS / f"check_{name}.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )
    print(name, record["ctest"], "|", record["pytest"], "|", record["replay"])


def replay(name: str) -> None:
    """The captured calls of prompts 87 and 88 through one build, summarised."""
    module = use_build(name)
    import probe87
    import probe88

    scratch = RESULTS / "replays" / name
    (scratch / "87").mkdir(parents=True, exist_ok=True)
    (scratch / "88").mkdir(parents=True, exist_ok=True)
    probe87.RESULTS = scratch / "87"
    probe88.RESULTS = scratch / "88"
    probe87.replay()
    probe88.replay()
    flux = json.loads((scratch / "87" / "replay.json").read_text(encoding="utf-8"))
    solves = json.loads((scratch / "88" / "replay.json").read_text(encoding="utf-8"))
    jacobi = [r for r in solves if r["method"] == "jacobi"]
    cg = [r for r in solves if r["method"] != "jacobi"]
    summary = {
        "module": module,
        "flux_calls": len(flux),
        "flux_calls_differing": sum(r["core_differs"] != 0 for r in flux),
        "jacobi_calls": len(jacobi),
        "jacobi_calls_differing": sum(
            r["field_cells_differing"] != 0
            or r["reference_iterations"] != r["core_iterations"]
            or r["reference_converged"] != r["core_converged"]
            or not r["leftover_same_bits"]
            or r["reference_forms"] != r["core_forms"]
            for r in jacobi
        ),
        "cg_calls": len(cg),
        "cg_calls_at_the_reference_count": sum(
            r["reference_iterations"] == r["core_iterations"] for r in cg
        ),
        "cg_largest_relative_difference": max(
            r["field_relative_difference"] for r in cg
        ),
        "cg_calls_differing_with_the_cores_dot": sum(
            r["with_cores_dot_bits_differ"] != 0 for r in cg
        ),
    }
    (RESULTS / f"replay_{name}.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8"
    )


def _timed(fn: Callable[[], object], repeats: int) -> dict[str, float]:
    """The median, minimum and quartiles of repeats calls of fn, after WARMUP calls."""
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        times.append(time.perf_counter() - started)
    quartiles = statistics.quantiles(times, n=4)
    return {
        "median_s": statistics.median(times),
        "min_s": min(times),
        "q1_s": quartiles[0],
        "q3_s": quartiles[2],
        "repeats": repeats,
    }


def windows_load() -> list[str]:
    """The Windows processes using most of a processor now, and the power state.

    Sampled over two seconds through ``powershell.exe`` (WSL's interop): the
    percent of one logical processor each of the eight busiest processes
    takes, whether the machine is on mains, and the processors' performance
    percentage. An empty list when Windows cannot be reached.
    """
    script = (
        "$s = Get-Counter '\\Process(*)\\% Processor Time' -SampleInterval 2 -MaxSamples 1"
        " -ErrorAction SilentlyContinue;"
        " $s.CounterSamples | Where-Object { $_.InstanceName -notin '_total','idle' }"
        " | Sort-Object CookedValue -Descending | Select-Object -First 8"
        " | ForEach-Object { '{0} {1:N0}' -f $_.InstanceName, $_.CookedValue };"
        " 'on mains ' + ((Get-CimInstance Win32_Battery).BatteryStatus -eq 2);"
        " 'processor performance ' + [int](Get-Counter"
        " '\\Processor Information(_Total)\\% Processor Performance')"
        ".CounterSamples[0].CookedValue"
    )
    try:
        done = run_command(
            ["powershell.exe", "-NoProfile", "-Command", script], timeout=60
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [line.strip() for line in done.stdout.splitlines() if line.strip()]


def calibration() -> dict[str, float]:
    """The time of a fixed NumPy loop that fits the first-level cache.

    The same compiled code in every timing process, whatever build is timed,
    so it reads the processor's speed in that process and nothing else.
    """
    values = np.linspace(1.0, 2.0, CALIBRATION_LENGTH)
    out = np.empty_like(values)
    return _timed(lambda: np.sqrt(values, out=out), CALIBRATION_REPEATS)


def _small_grid_calls() -> tuple[Callable[[], object], Callable[[], object]]:
    """An implicit solve and a flux call on a 2x2 grid: the binding's cost per call."""
    import _cfdcore

    x = np.array([0.0, 1.0, 2.0])
    xc = np.array([0.5, 1.5])
    grid = _cfdcore.Grid2(x, xc, x, xc, np.zeros((2, 2), dtype=np.int32), 2)
    scratch = _cfdcore.ImplicitScratch2(grid)
    scheme = _cfdcore.ScalarScheme2(grid)
    cells = np.ones((2, 2))
    g_u = np.zeros((2, 3))
    g_v = np.zeros((3, 2))
    g_u[:, 1] = 1.0
    solid = np.zeros((2, 2), dtype=bool)
    out = np.empty((2, 2))
    out_u = np.empty((2, 3))

    def solve() -> object:
        return _cfdcore.implicit_step(
            scratch,
            cells,
            cells,
            1.0,
            g_u,
            g_v,
            cells,
            solid,
            1e-10,
            0,
            out,
            None,
            "jacobi",
            1.0,
        )

    def flux() -> object:
        return _cfdcore.advective_flux(scheme, 0, cells, g_u, g_u, False, out_u, None)

    return solve, flux


def _implicit_cases() -> tuple[list[dict], dict[str, tuple[tuple, dict]]]:
    """The developed inputs of prompt 88 per class, and class 0's parts.

    The parts are 88's T6b: the cap at 0 (``no sweep``), the full solve and the
    early return (``nothing to solve``); and two CG solves of the same system
    without a bound, at the call's tolerance and at a stop the first residual
    meets (``cg`` and ``cg at the start``).
    """
    import probe88

    from src.scalar_scheme import implicit_step

    calls = probe88.developed_inputs()
    call = calls[0]
    args, kwargs = probe88.arguments_of(call)
    sweeps = implicit_step(*args, **kwargs).iterations
    empty = dict(call)
    empty.update(
        g_u=np.zeros_like(call["g_u"]),
        g_v=np.zeros_like(call["g_v"]),
        diagonal=np.zeros_like(call["diagonal"]),
    )
    parts = {
        "no sweep": probe88.arguments_of({**call, "max_iter": np.array(0)}),
        "the full solve": probe88.arguments_of({**call, "max_iter": np.array(sweeps)}),
        "nothing to solve": probe88.arguments_of(empty),
    }
    cg_args, cg_kwargs = probe88.arguments_of(call)
    cg_kwargs = {**cg_kwargs, "method": "cg", "leftover_bound": None}
    parts["cg"] = (cg_args, cg_kwargs)
    loose = (*cg_args[:7], CG_LOOSE_TOL, cg_args[8])
    parts["cg at the start"] = (loose, cg_kwargs)
    return calls, parts


def _flux_cases() -> list[dict]:
    """The developed rows of 87's T6b: each class's last frame in x and in y."""
    import probe87

    calls, _ = probe87.load("dust320")
    classes = 1 + max(int(c["class"]) for c in calls)
    frames = probe87.developed_fields(classes)
    cases = []
    for d in (0, 1):
        in_direction = [c for c in calls if int(c["d"]) == d]
        for k, frame in enumerate(frames):
            call = [c for c in in_direction if int(c["class"]) == k][-1]
            cases.append({**call, "c": frame, "class": np.array(k)})
    return cases


def time_reference(run_number: int) -> None:
    """The Python reference on every timed input."""
    import probe87
    import probe88

    from src.mesh import Mesh
    from src.scalar_scheme import advective_flux, implicit_step, mesh_axes

    record: dict[str, object] = {
        "build": "reference",
        "run": run_number,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "windows_load": windows_load(),
        "calibration": calibration(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
    }
    calls, parts = _implicit_cases()
    solves = []
    for call in calls:
        args, kwargs = probe88.arguments_of(call)
        result = implicit_step(*args, **kwargs)
        solves.append(
            {
                "class": int(call["class"]),
                "sweeps": int(result.iterations),
                **_timed(
                    lambda a=args, k=kwargs: implicit_step(*a, **k),
                    REFERENCE_IMPLICIT_REPEATS,
                ),
            }
        )
    record["solves"] = solves
    record["parts"] = {}
    for label, (args, kwargs) in parts.items():
        result = implicit_step(*args, **kwargs)
        record["parts"][label] = {
            "iterations": int(result.iterations),
            "fallbacks": int(result.fallbacks),
            **_timed(
                lambda a=args, k=kwargs: implicit_step(*a, **k),
                REFERENCE_IMPLICIT_REPEATS,
            ),
        }
    axes = mesh_axes(Mesh(probe87.scenario_config("dust320")))
    flux_rows = []
    for call in _flux_cases():
        d = int(call["d"])
        c, flux, inflow = call["c"], call["flux"], call["inflow"]
        upwind = bool(call["upwind"])

        def reference(
            d: int = d,
            c: np.ndarray = c,
            flux: np.ndarray = flux,
            inflow: np.ndarray = inflow,
            upwind: bool = upwind,
        ) -> np.ndarray:
            # As the transport solver calls it: y on the transposed arrays.
            if d == 0:
                return advective_flux(c, flux, inflow, axes[0], upwind)
            return advective_flux(c.T, flux.T, inflow.T, axes[1], upwind).T

        flux_rows.append(
            {
                "class": int(call["class"]),
                "d": d,
                **_timed(reference, REFERENCE_FLUX_REPEATS),
            }
        )
    record["flux"] = flux_rows
    (RESULTS / f"timing_reference_{run_number}.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )
    print("reference timed, run", run_number)


def time_build(name: str, run_number: int) -> None:
    """Every kernel's timing for one build, on the developed frames of 87 and 88."""
    module = use_build(name)
    import _cfdcore
    import probe87
    import probe88

    from src.scalar_scheme import implicit_step

    record: dict[str, object] = {
        "build": name,
        "run": run_number,
        "module": module,
        "flags": BUILDS[name]["flags"],
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "windows_load": windows_load(),
        "calibration": calibration(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
    }
    _, scratch = probe88.core_scratch("dust320")
    calls, parts = _implicit_cases()
    out = np.empty(calls[0]["c_star"].shape)
    solves = []
    for call in calls:
        args, kwargs = probe88.arguments_of(call)
        field, core = probe88.core_call(scratch, call)
        reference = implicit_step(*args, **kwargs)
        solves.append(
            {
                "class": int(call["class"]),
                "nonzero_cells": int(np.count_nonzero(call["c_star"])),
                "sweeps": int(core.iterations),
                "outputs_differing": probe88.bits_differ(field, reference.field)
                + int(core.iterations != reference.iterations)
                + int(core.converged != reference.converged)
                + int(not probe88.same_bits(core.leftover, reference.leftover)),
                **_timed(
                    lambda a=args, k=kwargs: _cfdcore.implicit_step(
                        scratch, *a, out, **k
                    ),
                    IMPLICIT_REPEATS,
                ),
            }
        )
    record["solves"] = solves
    record["parts"] = {}
    for label, (args, kwargs) in parts.items():
        core = _cfdcore.implicit_step(scratch, *args, out, **kwargs)
        record["parts"][label] = {
            "iterations": int(core.iterations),
            "fallbacks": int(core.fallbacks),
            "restarts": int(core.restarts),
            **_timed(
                lambda a=args, k=kwargs: _cfdcore.implicit_step(scratch, *a, out, **k),
                IMPLICIT_REPEATS,
            ),
        }
    _, scheme = probe87.core_scheme("dust320")
    flux_rows = []
    for call in _flux_cases():
        d = int(call["d"])
        face = np.empty(call["flux"].shape)
        arguments = (
            scheme,
            d,
            call["c"],
            call["flux"],
            call["inflow"],
            bool(call["upwind"]),
            face,
            None,
        )
        flux_rows.append(
            {
                "class": int(call["class"]),
                "d": d,
                "upwind": bool(call["upwind"]),
                "faces": int(face.size),
                **_timed(lambda a=arguments: _cfdcore.advective_flux(*a), FLUX_REPEATS),
                "overhead_median_s": _timed(
                    lambda a=arguments: _cfdcore.advective_flux_overhead(*a),
                    FLUX_REPEATS,
                )["median_s"],
            }
        )
    record["flux"] = flux_rows
    solve, flux = _small_grid_calls()
    record["small_grid"] = {
        "implicit_step": _timed(solve, FLUX_REPEATS),
        "advective_flux": _timed(flux, FLUX_REPEATS),
    }
    (RESULTS / f"timing_{name}_{run_number}.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )
    sweeps = record["parts"]["the full solve"]["iterations"]
    sweep = (
        record["parts"]["the full solve"]["median_s"]
        - record["parts"]["no sweep"]["median_s"]
    ) / max(sweeps, 1)
    print(
        f"{name} run {run_number}: sweep {sweep * 1e6:.1f} us,"
        f" no sweep {record['parts']['no sweep']['median_s'] * 1e6:.1f} us,"
        f" cg {record['parts']['cg']['median_s'] * 1e6:.1f} us"
        f" ({record['parts']['cg']['iterations']} iterations),"
        f" flux x {statistics.median(r['median_s'] for r in flux_rows if r['d'] == 0) * 1e6:.1f} us,"
        f" y {statistics.median(r['median_s'] for r in flux_rows if r['d'] == 1) * 1e6:.1f} us,"
        f" calibration {record['calibration']['median_s'] * 1e6:.2f} us"
    )


def session(run_number: int, names: list[str]) -> None:
    """``time`` for each name in turn, each in its own process."""
    for name in names:
        done = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "time",
                name,
                "--run",
                str(run_number),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
            check=False,
        )
        if done.returncode != 0:
            raise SystemExit(f"time {name} failed")


def plants() -> None:
    """Each planted defect built in turn, with the tests that fail on it."""
    name = "plants"
    BUILDS[name] = {"source": "head", "candidate": "", "flags": PROJECT_FLAGS}
    records = []
    for plant, what, edits in variants89.PLANTS:
        variants89.CANDIDATES["_plant"] = {"what": what, "edits": edits}
        BUILDS[name]["candidate"] = "_plant" if edits else ""
        record: dict[str, object] = {"name": plant, "what": what}
        try:
            build(name)
        except SystemExit as failure:
            record["build"] = "failed: " + str(failure)[-1500:]
            records.append(record)
            print(plant, "build failed")
            continue
        out = build_directory(name)
        ctest = run_command(["ctest", "--test-dir", str(out)])
        summary = [line for line in ctest.stdout.splitlines() if "tests passed" in line]
        record["ctest"] = summary[0].strip() if summary else ctest.stdout[-500:]
        record["ctest_failed"] = re.findall(
            r" - (\S+) \((?:Failed|[A-Za-z_ ]+)\)", ctest.stdout
        )
        pytest = run_command(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_core_parity.py",
                "-q",
                "-rf",
                "--tb=no",
                "-p",
                "no:cacheprovider",
            ],
            cwd=ROOT,
            env=_build_environment(name),
        )
        lines = pytest.stdout.splitlines()
        record["pytest_exit"] = pytest.returncode
        record["pytest"] = next(
            (
                line.strip("= ")
                for line in reversed(lines)
                if re.search(r" in [0-9.]+s", line)
            ),
            lines[-1] if lines else "no output: the run ended without a summary",
        )
        failed = [
            line.split("::", 1)[1].split(" ")[0]
            for line in lines
            if line.startswith("FAILED tests/test_core_parity.py::")
        ]
        record["parity_failed_by_test"] = dict(
            sorted(Counter(case.split("[")[0] for case in failed).items())
        )
        records.append(record)
        print(plant, record["ctest"], "|", record["pytest"])
        (RESULTS / "plants.json").write_text(
            json.dumps(records, indent=1) + "\n", encoding="utf-8"
        )


def vecinfo() -> None:
    """GCC's vectorisation report per source, build and flag set."""
    record: dict[str, dict[str, list[str]]] = {}
    for name in BUILDS:
        source = WORK / name / "core"
        if not source.exists():
            continue
        for file in ("src/scalar.cpp", "src/reduce.cpp"):
            flags = [f for f in BUILDS[name]["flags"].split() if not f.startswith("-W")]
            report = run_command(
                [
                    "g++",
                    "-std=c++20",
                    *flags,
                    "-DNDEBUG",
                    f"-I{source / 'include'}",
                    "-c",
                    str(source / file),
                    "-o",
                    "/dev/null",
                    "-fopt-info-vec-optimized",
                ]
            )
            lines = sorted(
                {
                    re.sub(r"^.*?/core/", "", line)
                    for line in (report.stdout + report.stderr).splitlines()
                    if "optimized:" in line
                }
            )
            record.setdefault(name, {})[file] = lines
    version = run_command(["g++", "--version"]).stdout.splitlines()[0]
    (RESULTS / "vecinfo.json").write_text(
        json.dumps({"compiler": version, "reports": record}, indent=1) + "\n",
        encoding="utf-8",
    )
    for name, files in record.items():
        print(name, {file: len(lines) for file, lines in files.items()})


def floor() -> None:
    """Build floor89.cpp under the project's flags and keep what it prints."""
    binary = WORK / "floor89"
    WORK.mkdir(parents=True, exist_ok=True)
    made = run_command(
        [
            "g++",
            "-std=c++20",
            "-O2",
            "-ffp-contract=off",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            str(HERE / "floor89.cpp"),
            "-o",
            str(binary),
        ]
    )
    if made.returncode != 0:
        raise SystemExit(made.stdout + made.stderr)
    ran = run_command([str(binary)])
    if ran.returncode != 0:
        raise SystemExit(ran.stdout + ran.stderr)
    record = json.loads(ran.stdout)
    record["warnings"] = made.stderr.strip()
    (RESULTS / "floor.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=1))


def styles() -> None:
    """Build styles89.cpp under each flag set and keep what each run prints."""
    binary = WORK / "styles89"
    WORK.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {"windows_load": windows_load()}
    for name, flags in FLAG_SETS.items():
        made = run_command(
            [
                "g++",
                "-std=c++20",
                *flags.split(),
                "-DNDEBUG",
                str(HERE / "styles89.cpp"),
                "-o",
                str(binary),
                "-fopt-info-vec-optimized",
            ]
        )
        if made.returncode != 0:
            raise SystemExit(made.stdout + made.stderr)
        ran = run_command([str(binary)])
        if ran.returncode != 0:
            raise SystemExit(ran.stdout + ran.stderr)
        record[name] = {
            "flags": flags,
            "styles": json.loads(ran.stdout),
            "vectorised": sorted(
                {
                    re.sub(r"^.*?styles89\.cpp:", "", line)
                    for line in (made.stdout + made.stderr).splitlines()
                    if "styles89.cpp" in line and "optimized:" in line
                }
            ),
        }
        print(
            name,
            {
                style: values["best_us"]
                for style, values in record[name]["styles"].items()
            },
        )
    (RESULTS / "styles.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    """Run one of the probe's commands."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command",
        choices=[
            "build",
            "check",
            "replay",
            "time",
            "session",
            "plants",
            "vecinfo",
            "floor",
            "styles",
        ],
    )
    parser.add_argument("name", nargs="?", choices=[*BUILDS, "reference"])
    parser.add_argument("--run", type=int, default=1, help="the timing's record")
    parser.add_argument(
        "--builds",
        nargs="+",
        default=["reference", *BUILDS],
        help="session's builds, in order",
    )
    args = parser.parse_args(argv)
    if args.command in ("build", "check", "replay", "time") and args.name is None:
        parser.error(f"{args.command} needs a build's name")
    if args.command == "build":
        build(args.name)
    elif args.command == "check":
        check(args.name)
    elif args.command == "replay":
        replay(args.name)
    elif args.command == "time":
        if args.name == "reference":
            time_reference(args.run)
        else:
            time_build(args.name, args.run)
    elif args.command == "session":
        session(args.run, args.builds)
    elif args.command == "plants":
        plants()
    elif args.command == "vecinfo":
        vecinfo()
    elif args.command == "floor":
        floor()
    else:
        styles()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
