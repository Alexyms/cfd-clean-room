"""Prompt 68: the planted defects of step 6b, each applied to a worktree, its tests run.

Usage:
    python docs/reports/val022/plant68.py WORKTREE [PLANT_ID ...]

WORKTREE is a scratch checkout of the commit under test (``git worktree add
--detach``), never the main checkout: the runner edits files there and
restores them with git checkout.

Each plant is one or more exact text replacements and the pytest selections
that should catch it. A plant is caught when a selection exits 1 (tests
failed); exit 2 or more (collection error) is BROKEN, never caught, and a
plant whose text does not match exactly once is NOT APPLIED. The worktree is
restored after every plant and checked clean. A control run of every
selection on the clean worktree comes first and must pass. Records:
results/builder68/plants[_IDS].json in the main checkout (gitignored).
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RECORDS = ROOT / "results" / "builder68"
CF = "src/config.py"
ME = "src/mesh.py"
FO = "src/forces.py"
TI = "src/time_integration.py"
CY = "validation/cylinder.py"
BM = "scripts/benchmark.py"
T_MESH = ("tests/test_mesh.py", "InteriorClustering")
T_CONF = ("tests/test_config.py", "Breakpoints")
T_HAND = ("tests/test_forces.py", "ByHand")
T_STEADY = ("tests/test_forces.py", "SteadyFlows")
T_ST = ("tests/test_forces.py", "Strouhal")
T_RUN = ("tests/test_transient_run.py", "")
T_CYL = ("tests/test_cylinder.py", "")
T_TRANSIENT = ("tests/test_config.py", "TransientSection")
T_GAUGE = ("tests/test_forces.py", "ByHand")

# (id, what, [(file, old, new), ...], [(test file, -k expression), ...])
PLANTS: list[tuple[str, str, list[tuple[str, str, str]], list[tuple[str, str]]]] = [
    (
        "C1",
        "a coarse-to-fine segment grown from its coarse end",
        [
            (
                CF,
                "    return widths if start <= end else widths[::-1]\n",
                "    return widths\n",
            )
        ],
        [T_MESH],
    ),
    (
        "C2",
        "the growth cap not checked",
        [(CF, "        if worst > MAX_GROWTH_RATIO:\n", "        if False:\n")],
        [T_CONF],
    ),
    (
        "C3",
        "the cell count not checked against the domain's",
        [(CF, "        if len(faces) - 1 != n:\n", "        if False:\n")],
        [T_CONF],
    ),
    (
        "C4",
        "an equal-spacing segment's count rounded, not up",
        [
            (
                CF,
                "        n = max(1, math.ceil(length / fine - _COUNT_ROUNDING))\n",
                "        n = max(1, round(length / fine))\n",
            )
        ],
        [T_MESH],
    ),
    (
        "M1",
        "a breakpoint axis put on the uniform path",
        [
            (
                ME,
                "uniform_x = self.stretch_ratio_x == 1.0 and not config.stretch_x.faces",
                "uniform_x = self.stretch_ratio_x == 1.0",
            )
        ],
        [T_MESH],
    ),
    (
        "F1",
        "the left face's pressure pushing the wrong way",
        [
            (
                FO,
                "((0, -1, 1.0), (0, 1, -1.0), (-1, 0, 1.0), (1, 0, -1.0))",
                "((0, -1, -1.0), (0, 1, -1.0), (-1, 0, 1.0), (1, 0, -1.0))",
            )
        ],
        [T_HAND, T_STEADY],
    ),
    (
        "F2",
        "the top face's pressure pushing the wrong way",
        [
            (
                FO,
                "((0, -1, 1.0), (0, 1, -1.0), (-1, 0, 1.0), (1, 0, -1.0))",
                "((0, -1, 1.0), (0, 1, -1.0), (-1, 0, 1.0), (1, 0, 1.0))",
            )
        ],
        [T_HAND, T_STEADY],
    ),
    (
        "F3",
        "the shear's distance a whole cell",
        [
            (
                FO,
                "half_y = 0.5 * mesh.dy_cell[:, None]",
                "half_y = mesh.dy_cell[:, None]",
            )
        ],
        [T_HAND, T_STEADY],
    ),
    (
        "F4",
        "the corner faces left out of the shear",
        [
            (
                FO,
                "u_face_body[:, 1:-1] = body[:, :-1] | body[:, 1:]",
                "u_face_body[:, 1:-1] = body[:, :-1] & body[:, 1:]",
            )
        ],
        [T_HAND],
    ),
    (
        "F5",
        "the zero crossings not interpolated",
        [
            (
                FO,
                "    crossings = t[k] - lift[k] * (t[k + 1] - t[k]) / (lift[k + 1] - lift[k])\n",
                "    crossings = t[k]\n",
            )
        ],
        [T_ST],
    ),
    (
        "F6",
        "the period counted over nine",
        [
            (
                FO,
                "period = float((crossings[-1] - crossings[0]) / STROUHAL_PERIODS)",
                "period = float((crossings[-1] - crossings[0]) / (STROUHAL_PERIODS - 1))",
            )
        ],
        [T_ST],
    ),
    (
        "F7",
        "ten crossings taken as ten periods",
        [
            (
                FO,
                "    if up.size < STROUHAL_PERIODS + 1:\n",
                "    if up.size < STROUHAL_PERIODS:\n",
            )
        ],
        [T_ST],
    ),
    (
        "R1",
        "the checkpoint without the previous level",
        [
            (
                TI,
                '                record.step, record.t, state, held["previous"]\n',
                "                record.step, record.t, state, None\n",
            )
        ],
        [T_RUN],
    ),
    (
        "R2",
        "the previous level taken after the step's own state",
        [
            (
                TI,
                "        if cadence is not None and (record.step % cadence == 0 or last):\n"
                "            flush()\n"
                "            run_path.write_transient_checkpoint(\n"
                '                record.step, record.t, state, held["previous"]\n'
                "            )\n"
                '        held["previous"] = state\n',
                '        held["previous"] = state\n'
                "        if cadence is not None and (record.step % cadence == 0 or last):\n"
                "            flush()\n"
                "            run_path.write_transient_checkpoint(\n"
                '                record.step, record.t, state, held["previous"]\n'
                "            )\n",
            )
        ],
        [T_RUN],
    ),
    (
        "R3",
        "a restart that keeps the rows after its checkpoint",
        [(TI, "            run_path.discard_after(resume.step)\n", "")],
        [T_RUN],
    ),
    (
        "R4",
        "the step-0 frame without the boundary's velocities",
        [
            (
                TI,
                "        boundary.apply_normal_velocity(u, v)\n"
                "        start = FlowState(u=u, v=v, p=p)\n",
                "        start = FlowState(u=u, v=v, p=p)\n",
            )
        ],
        [T_RUN],
    ),
    (
        "R5",
        "C_D averaged over the whole run",
        [
            (
                TI,
                "            window = series.t >= reading.crossings[0]\n",
                "            window = series.t >= 0.0\n",
            )
        ],
        [T_RUN],
    ),
    (
        "Y1",
        "the decay's comparison reversed",
        [
            (
                CY,
                '"passed": bool(np.all(ratios < 1.0)),',
                '"passed": bool(np.all(ratios > 1.0)),',
            )
        ],
        [T_CYL],
    ),
    (
        "Y2",
        "the sustain's margin dropped",
        [(CY, "np.all(ratios >= SUSTAIN_RATIO)", "np.all(ratios >= 1.0)")],
        [T_CYL],
    ),
    (
        "Y3",
        "an amplitude taken as the full range",
        [
            (
                CY,
                "            0.5 * (c_l[a + 1 : b + 1].max()",
                "            1.0 * (c_l[a + 1 : b + 1].max()",
            )
        ],
        [T_CYL],
    ),
    (
        "Y4",
        "the band's ends excluded",
        [
            (
                CY,
                "STROUHAL_BAND[0] <= st <= STROUHAL_BAND[1]",
                "STROUHAL_BAND[0] < st < STROUHAL_BAND[1]",
            )
        ],
        [T_CYL],
    ),
    (
        "Y5",
        "the start-up's perturbation dropped",
        [(CY, "*= STARTUP_BOTTOM_FACTOR", "*= 1.0")],
        [T_CYL],
    ),
    # Prompt 68c: the window clauses of the Re 40 and Re 60 measures (test 68's
    # B1, plants T5, T6, T18), the harness's reading (B2, plant T7) and the
    # suggestions' tests.
    (
        "Y6",
        "the decay's window the whole run, not its second half",
        [(CY, "    late = t >= 0.5 * t[-1]\n", "    late = t >= 0.0\n")],
        [T_CYL],
    ),
    (
        "Y7",
        "the quiet reading's window the second half, not the last tenth",
        [
            (
                CY,
                "    tail = t >= (1.0 - QUIET_FRACTION) * t[-1]\n",
                "    tail = t >= 0.5 * t[-1]\n",
            )
        ],
        [T_CYL],
    ),
    (
        "Y8",
        "the sustain reads the oldest five ratios, not the last",
        [
            (
                CY,
                "    last = every[-(AMPLITUDE_PERIODS + 1) :]\n",
                "    last = every[: AMPLITUDE_PERIODS + 1]\n",
            )
        ],
        [T_CYL],
    ),
    (
        "Y9",
        "the decay reads the first six amplitudes of its window, not the last",
        [
            (
                CY,
                "        last = recent[-(AMPLITUDE_PERIODS + 1) :]\n",
                "        last = recent[: AMPLITUDE_PERIODS + 1]\n",
            )
        ],
        [T_CYL],
    ),
    (
        "Z1",
        "the harness reports the spectrum's number as the row's value",
        [
            (
                BM,
                '    st = float("nan") if reading is None else reading.strouhal\n',
                '    st = float("nan") if reading is None else reading.spectral_strouhal\n',
            )
        ],
        [T_CYL],
    ),
    (
        "Z2",
        "the harness scores the band on the spectrum's number",
        [
            (
                BM,
                "cylinder.strouhal_in_band(None if reading is None else reading.strouhal)",
                "cylinder.strouhal_in_band(\n"
                "            None if reading is None else reading.spectral_strouhal\n"
                "        )",
            )
        ],
        [T_CYL],
    ),
    (
        "Z3",
        "the harness reports the spectrum's number as the crossing reading",
        [
            (
                BM,
                '        "strouhal": None if reading is None else reading.strouhal,\n',
                '        "strouhal": None if reading is None else reading.spectral_strouhal,\n',
            )
        ],
        [T_CYL],
    ),
    (
        "S1",
        "the transient section's refusal of the turbulence section removed",
        [
            (
                CF,
                "        if self.turbulence is not None:\n"
                "            raise ValueError(\n"
                '                f"{context}: a transient run is laminar',
                "        if False:\n"
                "            raise ValueError(\n"
                '                f"{context}: a transient run is laminar',
            )
        ],
        [T_TRANSIENT],
    ),
    (
        "S2a",
        "strouhal's finiteness check dropped",
        [
            (
                FO,
                "        if not math.isfinite(value) or value <= 0.0:\n",
                "        if value <= 0.0:\n",
            )
        ],
        [T_ST],
    ),
    (
        "S2b",
        "strouhal's bool check dropped",
        [
            (
                FO,
                "        if isinstance(value, bool) or not isinstance(value, (int, float)):\n",
                "        if not isinstance(value, (int, float)):\n",
            )
        ],
        [T_ST],
    ),
    (
        "F8",
        "D taken from the obstacle's x extent",
        [
            (
                FO,
                "self.height: float = spec.y_end - spec.y_start",
                "self.height: float = spec.x_end - spec.x_start",
            )
        ],
        [T_GAUGE],
    ),
    (
        "F9",
        "the spectrum's cross-check always agreeing",
        [
            (
                FO,
                "        agrees=abs(spectral - st) <= resolution,\n",
                "        agrees=True,\n",
            )
        ],
        [T_ST],
    ),
]


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout


def pytest(root: Path, test_file: str, expression: str) -> tuple[int, str]:
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    python = str(ROOT / ".venv" / "Scripts" / "python.exe")
    command = [python, "-m", "pytest", test_file, "-q", "-x", "-p", "no:cacheprovider"]
    if expression:
        command += ["-k", expression]
    done = subprocess.run(command, cwd=root, capture_output=True, text=True, env=env)
    tail = "\n".join(done.stdout.strip().splitlines()[-3:])
    return done.returncode, tail


def apply(root: Path, path: str, old: str, new: str) -> bool:
    target = root / path
    raw = target.read_bytes().decode("ascii")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    if text.count(old) != 1 or old == new:
        return False
    text = text.replace(old, new)
    target.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("ascii"))
    return True


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    wanted = set(sys.argv[2:])
    plants = [p for p in PLANTS if not wanted or p[0] in wanted]
    if git(root, "status", "--porcelain").strip():
        raise SystemExit("the worktree is not clean")
    controls = sorted({selection for p in plants for selection in p[3]})
    for test_file, expression in controls:
        code, tail = pytest(root, test_file, expression)
        last = tail.splitlines()[-1] if tail else ""
        print(f"control {test_file} -k {expression!r}: {code} {last}", flush=True)
        if code != 0:
            raise SystemExit("control failed")
    results = []
    for pid, what, edits, selections in plants:
        started = time.perf_counter()
        applied = all(apply(root, path, old, new) for path, old, new in edits)
        caught_by = []
        outcome = "NOT APPLIED"
        if applied:
            outcome = "SURVIVED"
            for test_file, expression in selections:
                code, tail = pytest(root, test_file, expression)
                verdict = {0: "passed", 1: "failed"}.get(code, f"broken ({code})")
                caught_by.append(
                    {
                        "tests": f"{test_file} -k {expression!r}",
                        "verdict": verdict,
                        "tail": tail,
                    }
                )
                if code >= 2:
                    outcome = f"BROKEN ({code})"
                elif code == 1 and outcome == "SURVIVED":
                    outcome = "CAUGHT"
        git(root, "checkout", "--", "src", "validation", "scripts")
        if git(root, "status", "--porcelain").strip():
            raise SystemExit(f"{pid}: the worktree is not clean after restoring")
        seconds = time.perf_counter() - started
        verdicts = ", ".join(f"{c['tests']}: {c['verdict']}" for c in caught_by)
        print(f"{pid} {outcome} {seconds:.0f} s  {what}  [{verdicts}]", flush=True)
        results.append(
            {
                "id": pid,
                "what": what,
                "outcome": outcome,
                "selections": caught_by,
                "seconds": seconds,
            }
        )
    RECORDS.mkdir(parents=True, exist_ok=True)
    out = RECORDS / f"plants{'_' + '_'.join(sorted(wanted)) if wanted else ''}.json"
    out.write_text(
        json.dumps(
            {
                "commit": git(root, "rev-parse", "--short", "HEAD").strip(),
                "plants": results,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
