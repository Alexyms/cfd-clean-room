"""Prompt 86's gate runs: the Python reference under WSL2 against the Windows records.

    python docs/reports/probe86/probe86.py configs
    python docs/reports/probe86/probe86.py run <location> <scenario>
    python docs/reports/probe86/probe86.py parity

Run under the WSL2 environment (``scripts/setup_linux_env.sh``).
The runs are prompt 85's three 80x30 references (``probe85.py``): the dust
release, the door leak and the filter breach, from main's scenario files with
the grid and ``mass_imbalance_tol`` written as ``probe85.variant_mapping``
writes them for ``reference``. Its stored configurations name ``C:\\`` paths
and stay as they are; ``configs`` writes the same mappings for this checkout's
Linux path into ``results/builder86/configs/`` and checks that nothing but the
base file's path differs from the Windows ones.

A location is where a run's file (and its checkpoint beside it) goes:

- ``reference``: ``results/builder86/runs/reference/``, on ``/mnt/c``; the three
  scenarios, run three at once as prompt 85's were (``run86.py``);
- ``mnt_c``: ``results/builder86/runs/mnt_c/``, on ``/mnt/c``, the dust release
  alone, for the run file's placement;
- ``linux_fs``: ``~/cfd-runs/``, on the Linux file system, the same;
- ``mnt_c_repeat`` and ``linux_fs_repeat``: each of the two again, an addition
  to the prompt's pair, to set the pair's difference against the noise;
- ``counted`` (under WSL2) and ``windows_counted`` (under Windows, from prompt
  85's stored configurations): the three references again with every inner
  solve's iteration count recorded (``StopCounts``), an addition, since the
  run files hold only the outer counts and the prompt asks which stop counts
  differ.

Each run is ``probe85.run``, unchanged: ``time_integration.run`` from rest on one
BLAS thread, with its summary beside the run file. This process points
``probe85``'s two path functions at the locations above before calling it, and
adds the platform to the summary afterwards. Nothing under ``src/`` changes.

``parity`` repeats ``tests/test_core_parity.py``'s comparison of
``_cfdcore.pairwise_sum`` with ``np.sum`` on prediction (b)'s arrays and writes
the counts to ``results/builder86/parity86.json`` for the report's tables.
"""

import json
import os
import platform
import sys
from pathlib import Path, PureWindowsPath

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np  # noqa: E402 -- the thread variables are set first
import yaml  # noqa: E402 -- the thread variables are set first

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "probe85"))

import probe85  # noqa: E402 -- follows sys.path.insert

from src.config import SimConfig  # noqa: E402 -- follows sys.path.insert

RESULTS = ROOT / "results" / "builder86"
CONFIGS = RESULTS / "configs" / "reference"
WINDOWS = probe85.RESULTS
GRID = "80x30"
SCENARIOS = probe85.GRID_SCENARIOS[GRID]
# Location name to the directory its run files go in.
LOCATIONS = {
    "reference": RESULTS / "runs" / "reference",
    "mnt_c": RESULTS / "runs" / "mnt_c",
    "linux_fs": Path.home() / "cfd-runs",
    # A repeat of each, added after the first pair differed by a fifth, to show
    # whether that difference is the placement or the run-to-run noise.
    "mnt_c_repeat": RESULTS / "runs" / "mnt_c_repeat",
    "linux_fs_repeat": Path.home() / "cfd-runs" / "repeat",
    # The three references again on each platform with the inner solves'
    # counts recorded, since the run files hold only the outer counts.
    "counted": RESULTS / "runs" / "counted",
    "windows_counted": RESULTS / "runs" / "windows_counted",
}
COUNTED = ("counted", "windows_counted")
PARITY = RESULTS / "parity86.json"


def config_path(scenario: str) -> Path:
    """The Linux scenario file of one run."""
    return CONFIGS / f"{scenario}_{GRID}.yaml"


def run_path(location: str, scenario: str) -> Path:
    """The run file of one run at LOCATION."""
    return LOCATIONS[location] / f"{scenario}_{GRID}.h5"


def windows_config(scenario: str) -> dict:
    """Prompt 85's stored configuration of the Windows reference run."""
    path = WINDOWS / "configs" / "reference" / f"{scenario}_{GRID}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_configs() -> None:
    """Each scenario's Linux file, checked against the Windows one and by loading it.

    Raises
    ------
    ValueError
        If the Linux mapping differs from the Windows one in anything but the
        base file's path, or names another base file.
    """
    CONFIGS.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        mapping, changed = probe85.variant_mapping("reference", scenario, GRID)
        theirs = windows_config(scenario)
        mine = json.loads(json.dumps(mapping))
        my_base = Path(mine["scenario"].pop("base"))
        their_base = PureWindowsPath(theirs["scenario"].pop("base"))
        if my_base != ROOT / "configs" / their_base.name:
            raise ValueError(f"{scenario}: base {my_base} is not {their_base.name}")
        if mine != theirs:
            raise ValueError(
                f"{scenario}: the Linux mapping differs beyond the base path"
            )
        path = config_path(scenario)
        path.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
        path.with_suffix(".changed.json").write_text(json.dumps(changed))
        SimConfig(path)


def head_commit() -> str:
    """HEAD's hash read from the checkout's files, or "unknown".

    Git runs from Windows only (ADR-017 A), so the Linux side reads the hash
    instead of asking git for it.
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


class StopCounts:
    """Every inner solve's iteration count in a run, in call order.

    Wraps, in this process only, the pressure correction's CG
    (``pressure.conjugate_gradient``) and the implicit solves of the transport
    and the turbulence model (``implicit_step`` as each module imported it),
    and records what each call returned. The arithmetic is untouched: a counted
    run is bitwise its platform's uncounted one, which the report checks.
    """

    def __init__(self) -> None:
        self.cg: list[tuple[int, int, int]] = []
        self.transport: list[tuple[int, int]] = []
        self.turbulence: list[tuple[int, int]] = []

    def install(self) -> None:
        """Replace the three names."""
        from src import pressure, solver_transport, turbulence

        cg = pressure.conjugate_gradient

        def counted_cg(*args: object, **kwargs: object) -> object:
            result = cg(*args, **kwargs)
            self.cg.append(
                (result.iterations, result.products, int(result.reached_cap))
            )
            return result

        pressure.conjugate_gradient = counted_cg
        for module, record in (
            (solver_transport, self.transport),
            (turbulence, self.turbulence),
        ):
            solve = module.implicit_step

            def counted(
                *args: object,
                _solve: object = solve,
                _record: list = record,
                **kwargs: object,
            ) -> object:
                result = _solve(*args, **kwargs)
                _record.append((result.iterations, result.fallback_sweeps))
                return result

            module.implicit_step = counted

    def save(self, path: Path) -> None:
        """The counts as int64 arrays beside the run file."""
        np.savez(
            path,
            cg=np.array(self.cg, dtype=np.int64).reshape(-1, 3),
            transport=np.array(self.transport, dtype=np.int64).reshape(-1, 2),
            turbulence=np.array(self.turbulence, dtype=np.int64).reshape(-1, 2),
        )


def counts_path(location: str, scenario: str) -> Path:
    """The stop counts of a counted run."""
    return run_path(location, scenario).with_suffix(".counts.npz")


def run(location: str, scenario: str) -> None:
    """One run from rest at LOCATION, by probe85.run, with the platform added.

    A counted location records the inner solves' counts (StopCounts); the
    Windows one runs from prompt 85's stored configuration, under Windows.
    """
    if location == "windows_counted":
        windows_file = WINDOWS / "configs" / "reference" / f"{scenario}_{GRID}.yaml"
        probe85.config_path = lambda _variant, _s, _grid: windows_file
    else:
        probe85.config_path = lambda _variant, s, _grid: config_path(s)
    probe85.run_path = lambda _variant, s, _grid: run_path(location, s)
    probe85.commit = head_commit
    counts = StopCounts() if location in COUNTED else None
    if counts is not None:
        counts.install()
    probe85.run("reference", scenario, GRID)
    if counts is not None:
        counts.save(counts_path(location, scenario))
    summary_path = run_path(location, scenario).with_suffix(".json")
    summary = json.loads(summary_path.read_text())
    summary["location"] = location
    summary["run_file"] = str(run_path(location, scenario))
    summary["platform"] = platform.platform()
    summary_path.write_text(json.dumps(summary, indent=1))
    # Windows reads the records from the checkout, so a summary written on the
    # Linux file system is copied there; its run file stays where it ran.
    copy = RESULTS / "runs" / location / summary_path.name
    if copy != summary_path:
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_text(json.dumps(summary, indent=1))


def parity() -> None:
    """Prediction (b)'s comparison, counted, written to PARITY."""
    from tests import test_core_parity as t

    pairwise_sum = t._cfdcore.pairwise_sum
    rng = np.random.default_rng(t.SEED)
    vectors = {"compared": 0, "differ": [], "order_shows": 0}
    for n in t.LENGTHS:
        values = t._values(rng, n)
        vectors["compared"] += 1
        if n and np.cumsum(values)[-1] != np.sum(values):
            vectors["order_shows"] += 1
        if t._bits(pairwise_sum(values)) != t._bits(np.sum(values)):
            vectors["differ"].append(n)
    rng = np.random.default_rng(t.SEED + 1)
    fields = {"compared": 0, "differ": [], "order_shows": 0}
    for shape in t.SHAPES:
        field = t._values(rng, shape)
        fields["compared"] += 1
        if np.cumsum(field)[-1] != np.sum(field):
            fields["order_shows"] += 1
        if t._bits(pairwise_sum(field.reshape(-1))) != t._bits(np.sum(field)):
            fields["differ"].append(list(shape))
    record = {
        "lengths": [min(t.LENGTHS), 1000, *t.LENGTHS[-3:]],
        "shapes": [list(s) for s in t.SHAPES],
        "vectors": vectors,
        "fields": fields,
        "module": t._cfdcore.__file__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    PARITY.write_text(json.dumps(record, indent=1))


def main(argv: list[str] | None = None) -> int:
    """The command line."""
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["configs"]:
        write_configs()
    elif args[:1] == ["run"] and len(args) == 3:
        run(args[1], args[2])
    elif args[:1] == ["parity"]:
        parity()
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
