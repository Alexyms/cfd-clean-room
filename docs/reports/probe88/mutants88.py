"""Plant defects in a copy of core/ and see which tests catch each (prompt 88).

    python docs/reports/probe88/mutants88.py

Under WSL2, with the Linux environment's Python. Each plant replaces exact
text in one or more files of a copy of ``core/`` on the Linux file system
(``~/scratch88/mutants``), and a replacement whose text is not found exactly
once refuses the plant instead of leaving it unplanted. The copy is built out
of source with the main build's GoogleTest sources (no download), its
GoogleTest tests run by CTest, and ``tests/test_core_parity.py`` runs against
its module, which PYTHONPATH puts ahead of the installed one. Before each
plant every file is restored from the checkout, so each plant starts from the
unchanged source; the build is incremental. ``control`` is the unchanged
source first, and ``control_end`` the unchanged source again after the last
plant. The records go to ``results/builder88/mutants88.json``, one entry per
plant with CTest's summary, the failing GoogleTest names, pytest's summary and
the failing parity cases by test function. The checkout is not changed.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WORK = Path.home() / "scratch88" / "mutants"
GTEST_SRC = (
    Path(os.environ.get("CFD_CORE_BUILD", Path.home() / "build" / "cfd-core"))
    / "_deps"
    / "googletest-src"
)
OUT = ROOT / "results" / "builder88" / "mutants88.json"
SRC = "src/scalar.cpp"
REDUCE = "src/reduce.cpp"
BIND = "python/bindings.cpp"

# Each plant: a name, what it changes, and its replacements (file, old, new).
PLANTS: list[tuple[str, str, list[tuple[str, str, str]]]] = [
    ("control", "the unchanged source", []),
    (
        "diagonal_order",
        "the diagonal's terms along y taken north before south",
        [
            (
                SRC,
                "        extra = extra + g[d][low];\n"
                "        extra = extra + g[d][low + s.stride[d]];\n",
                "        extra = extra + g[d][low + s.stride[d]];\n"
                "        extra = extra + g[d][low];\n",
            )
        ],
    ),
    (
        "neighbour_order",
        "the neighbour sum's terms along y taken south before north",
        [
            (
                SRC,
                "        sum = sum + g[d][low + s.stride[d]] * high_value;\n"
                "        sum = sum + g[d][low] * low_value;\n",
                "        sum = sum + g[d][low] * low_value;\n"
                "        sum = sum + g[d][low + s.stride[d]] * high_value;\n",
            )
        ],
    ),
    (
        "leftover_before_stop",
        "Jacobi's acceptance reads the residual of the sweep before the stop's",
        [
            (
                SRC,
                "                r[i] = ri;\n"
                "                largest.add(ri);\n"
                "                // c + r / a_P",
                "                scratch.z[i] = r[i];\n"
                "                r[i] = ri;\n"
                "                largest.add(ri);\n"
                "                // c + r / a_P",
            ),
            (
                SRC,
                "        if (largest.within(stop) && accept.test(residual, id)) {\n"
                "            converged = true;",
                "        if (largest.within(stop) &&\n"
                "            accept.test(std::span<const double>(scratch.z.data(), n), id)) {\n"
                "            converged = true;",
            ),
        ],
    ),
    (
        "final_twice",
        "final forms the leftover again from the residual the test formed it from",
        [
            (
                SRC,
                "    if (formed_ != residual_id) {\n"
                "        form(residual, residual_id);\n"
                "    }\n",
                "    form(residual, residual_id);\n",
            )
        ],
    ),
    (
        "fallback_from_cg",
        "the fallback's Jacobi solve starts from the CG iterate",
        [
            (
                SRC,
                "    const solve_outcome j = jacobi(in, setup.stop, fresh, scratch, out);\n",
                "    implicit_input<D> from_cg = in;\n"
                "    std::copy(scratch.c.begin(), scratch.c.end(), scratch.q.begin());\n"
                "    from_cg.c_star = scratch.q;\n"
                "    const solve_outcome j = jacobi(from_cg, setup.stop, fresh, scratch, out);\n",
            )
        ],
    ),
    (
        "restart_skipped",
        "a failed confirmation goes on with the old direction instead of restarting",
        [
            (
                SRC,
                "            rz = restart();\n            ++restarts;\n",
                "            ++restarts;\n",
            )
        ],
    ),
    (
        "p_as_z_plus_beta_p",
        "the direction updated as p = z + beta p, one expression",
        [
            (
                SRC,
                "            p[i] = p[i] * beta;\n            p[i] = p[i] + z[i];\n",
                "            p[i] = z[i] + beta * p[i];\n",
            )
        ],
    ),
    (
        "p_fused",
        "the direction updated as fma(beta, p, z), the contraction -ffp-contract=off forbids",
        [
            (
                SRC,
                "            p[i] = p[i] * beta;\n            p[i] = p[i] + z[i];\n",
                "            p[i] = std::fma(beta, p[i], z[i]);\n",
            )
        ],
    ),
    (
        "held_residual_jacobi",
        "a held cell's Jacobi residual not zeroed",
        [
            (
                SRC,
                "                if (in.solid[i] || (with_held && in.held[i])) {\n",
                "                if (in.solid[i] || (with_held && false)) {\n",
            )
        ],
    ),
    (
        "held_residual_cg",
        "a held cell's CG true residual not zeroed",
        [
            (
                SRC,
                "                if (is_fixed(in.solid, in.held, i)) {\n"
                "                    ri = 0.0;\n",
                "                if (in.solid[i]) {\n                    ri = 0.0;\n",
            )
        ],
    ),
    (
        "stop_skips_solid",
        "the stop's largest |b| taken over the non-SOLID cells only",
        [
            (
                SRC,
                "    const implicit_setup setup = implicit_system(in, scratch);\n",
                "    implicit_setup setup = implicit_system(in, scratch);\n"
                "    if (setup.diffuses) {\n"
                "        max_abs fluid;\n"
                "        for (std::size_t i = 0; i < in.c_star.size(); ++i) {\n"
                "            if (!in.solid[i]) {\n"
                "                fluid.add(scratch.b[i]);\n"
                "            }\n"
                "        }\n"
                "        setup.stop = in.tol * fluid.value();\n"
                "    }\n",
            )
        ],
    ),
    (
        "bound_strict",
        "the bound accepts a leftover strictly inside it only",
        [
            (
                SRC,
                "return std::fabs(form(residual, residual_id)) <= *bound_;",
                "return std::fabs(form(residual, residual_id)) < *bound_;",
            )
        ],
    ),
    (
        "scalar_leftover_as_sum_of_dt_r",
        "a scalar step's leftover summed as dt r per cell, not dt times the sum",
        [
            (
                SRC,
                "        leftover = dt_ * reduce::pairwise_sum(residual);\n",
                "        for (std::size_t i = 0; i < residual.size(); ++i) {\n"
                "            terms_[i] = dt_ * residual[i];\n"
                "        }\n"
                "        leftover = reduce::pairwise_sum(terms_);\n",
            )
        ],
    ),
    (
        "solid_step_unmasked",
        "a per-cell step read in SOLID cells too",
        [
            (
                SRC,
                "            dt_masked[i] = solid[i] ? 0.0 : dt_cells[i];\n",
                "            dt_masked[i] = dt_cells[i];\n",
            )
        ],
    ),
    (
        "positivity_admits_zero",
        "the fallback's test admits a zero",
        [(SRC, "scratch.c[i] > 0.0;", "scratch.c[i] >= 0.0;")],
    ),
    (
        "cg_start_keeps_solid",
        "the CG start keeps C*'s SOLID values",
        [
            (
                SRC,
                "        c[i] = in.solid[i] ? 0.0 : in.c_star[i];\n",
                "        c[i] = in.c_star[i];\n",
            )
        ],
    ),
    (
        "jacobi_solid_kept",
        "Jacobi's update does not zero the SOLID cells",
        [
            (
                SRC,
                "                next[i] = in.solid[i] ? 0.0 : c[i] + ri / a_p[i];\n",
                "                next[i] = c[i] + ri / a_p[i];\n",
            )
        ],
    ),
    (
        "negative_zero_diffuses",
        "a diagonal term of -0.0 counted as something to solve",
        [
            (
                SRC,
                "            any = any || extra != 0.0;\n",
                "            any = any || extra != 0.0 || std::signbit(extra);\n",
            )
        ],
    ),
    (
        "cap_one_more",
        "Jacobi's cap allows one sweep more",
        [
            (
                SRC,
                "        if (sweeps >= in.max_iter) {\n",
                "        if (sweeps > in.max_iter) {\n",
            )
        ],
    ),
    (
        "one_block",
        "the CG's dot products summed as one block, not a block per row",
        [
            (
                SRC,
                "    const std::size_t block = fields.cells(0);\n",
                "    const std::size_t block = n;\n",
            )
        ],
    ),
    (
        "bind_bool_bound_accepted",
        "the binding reads a bool bound as a number",
        [
            (
                BIND,
                "    if (nb::isinstance<bool>(bound) || PyObject_IsInstance(bound.ptr(), "
                "numpy_bool.ptr()) == 1) {\n",
                "    if (false) {\n",
            )
        ],
    ),
    (
        "bind_dt_shape_unchecked",
        "the binding does not check a per-cell step's shape",
        [
            (
                BIND,
                '    require_shape(cells, {g.cells(1), g.cells(0)}, who, "dt");\n',
                "",
            )
        ],
    ),
    (
        "bind_out_apart_unchecked",
        "the binding lets out share memory with an input array",
        [
            (
                BIND,
                "        require_apart(out.data(), out_bytes, a.data(), a.size_bytes(), who);\n",
                "",
            )
        ],
    ),
    # Round 1's plants (prompt 88b): one for each gap review 88 and test 88 found.
    (
        "cg_start_test_dropped",
        "the CG's acceptance dropped where the start residual meets the stop",
        [
            (
                SRC,
                "    if (true_residual() && accept.test(r_span, id)) {\n"
                "        accept.final(r_span, id);\n"
                "        return {0, true, 0};\n",
                "    if (true_residual()) {\n"
                "        accept.final(r_span, id);\n"
                "        return {0, true, 0};\n",
            )
        ],
    ),
    (
        "cg_confirmation_test_dropped",
        "the CG's acceptance dropped where a confirmed residual meets the stop",
        [
            (
                SRC,
                "            if (true_residual() && accept.test(r_span, id)) {\n"
                "                accept.final(r_span, id);\n"
                "                return {iterations, true, restarts};\n",
                "            if (true_residual()) {\n"
                "                accept.final(r_span, id);\n"
                "                return {iterations, true, restarts};\n",
            )
        ],
    ),
    (
        "padding_east_as_cell",
        "the zero read beyond a row's east end taken as the cell's own value",
        [
            (
                SRC,
                "    const double east = m + 1 < s.n0 ? x[i + 1] : 0.0;\n",
                "    const double east = m + 1 < s.n0 ? x[i + 1] : x[i];\n",
            )
        ],
    ),
    (
        "padding_west_as_cell",
        "the zero read beyond a row's west end taken as the cell's own value",
        [
            (
                SRC,
                "    const double west = m > 0 ? x[i - 1] : 0.0;\n",
                "    const double west = m > 0 ? x[i - 1] : x[i];\n",
            )
        ],
    ),
    (
        "padding_north_as_cell",
        "the zero read beyond a column's north end taken as the cell's own value",
        [
            (
                SRC,
                "        const double high_value = s.has_high[d] ? x[i + s.stride[d]] : 0.0;\n",
                "        const double high_value = s.has_high[d] ? x[i + s.stride[d]] : x[i];\n",
            )
        ],
    ),
    (
        "padding_south_as_cell",
        "the zero read beyond a column's south end taken as the cell's own value",
        [
            (
                SRC,
                "        const double low_value = s.has_low[d] ? x[i - s.stride[d]] : 0.0;\n",
                "        const double low_value = s.has_low[d] ? x[i - s.stride[d]] : x[i];\n",
            )
        ],
    ),
    (
        "positivity_reads_held",
        "the fallback's positivity test read in held cells too",
        [
            (
                SRC,
                "        positive = is_fixed(in.solid, in.held, i) || scratch.c[i] > 0.0;\n",
                "        positive = in.solid[i] || scratch.c[i] > 0.0;\n",
            )
        ],
    ),
    (
        "fallback_iterations_are_jacobis",
        "a fallback's result.iterations set to the Jacobi sweeps, not the CG's count",
        [
            (
                SRC,
                "    result.fallback_sweeps = j.iterations;\n",
                "    result.fallback_sweeps = j.iterations;\n"
                "    result.iterations = j.iterations;\n",
            )
        ],
    ),
    (
        "stop_strict",
        "the max-norm stop met strictly below the stop, not at it",
        [
            (
                SRC,
                "    bool within(double stop) const { return !nan_ && largest_ <= stop; }\n",
                "    bool within(double stop) const { return !nan_ && largest_ < stop; }\n",
            )
        ],
    ),
    (
        "jacobi_test_dropped",
        "Jacobi's acceptance dropped where its residual meets the stop",
        [
            (
                SRC,
                "        if (largest.within(stop) && accept.test(residual, id)) {\n",
                "        if (largest.within(stop)) {\n",
            )
        ],
    ),
    (
        "jacobi_update_halved",
        "Jacobi's update takes half the residual's step",
        [
            (
                SRC,
                "                next[i] = in.solid[i] ? 0.0 : c[i] + ri / a_p[i];\n",
                "                next[i] = in.solid[i] ? 0.0 : c[i] + 0.5 * ri / a_p[i];\n",
            )
        ],
    ),
    (
        "cg_alpha_from_pp",
        "the CG's step length divides by p.p, not p.A p",
        [
            (
                SRC,
                "        const double alpha = rz / reduce::blocked_dot(p_span, q_span, block);\n",
                "        const double alpha = rz / reduce::blocked_dot(p_span, p_span, block);\n",
            )
        ],
    ),
    (
        "blocked_dot_zero_block_returns_zero",
        "blocked_dot returns 0.0 for a block of zero instead of refusing it",
        [
            (
                REDUCE,
                "    if (block == 0) {\n"
                '        throw std::invalid_argument("blocked_dot: the block length is at least 1");\n'
                "    }\n",
                "    if (block == 0) {\n        return 0.0;\n    }\n",
            )
        ],
    ),
    (
        "blocked_dot_unequal_lengths_returns_zero",
        "blocked_dot returns 0.0 for arrays of different lengths instead of refusing them",
        [
            (
                REDUCE,
                "    if (a.size() != b.size()) {\n"
                '        throw std::invalid_argument("blocked_dot: the two arrays differ in length");\n'
                "    }\n",
                "    if (a.size() != b.size()) {\n        return 0.0;\n    }\n",
            )
        ],
    ),
    (
        "bind_row_dot_block_unclamped",
        "the binding hands row_dot's row length as the block, zero for an array with no columns",
        [
            (
                BIND,
                "    const std::size_t block = std::max<std::size_t>(a.shape(1), 1);\n",
                "    const std::size_t block = a.shape(1);\n",
            )
        ],
    ),
    ("control_end", "the unchanged source, after every plant", []),
]


def plant(replacements: list[tuple[str, str, str]]) -> None:
    """Restore every changed file from the checkout, then make each replacement exactly once.

    A file is restored by writing the checkout's text into it, so its time is
    new and ninja rebuilds it; a copy that kept the checkout's older time would
    leave the last plant's object in the build (the first run of this script
    did that, and its ``control_end`` failed).
    """
    for directory in ("src", "python", "include/cfdcore"):
        for source in sorted((ROOT / "core" / directory).iterdir()):
            target = WORK / "core" / directory / source.name
            text = source.read_text(encoding="utf-8").replace("\r\n", "\n")
            if target.read_text(encoding="utf-8") != text:
                target.write_text(text, encoding="utf-8")
    for file, old, new in replacements:
        path = WORK / "core" / file
        text = path.read_text(encoding="utf-8")
        if text.count(old) != 1:
            raise SystemExit(
                f"{file}: the plant's text is found {text.count(old)} times: {old!r}"
            )
        path.write_text(text.replace(old, new), encoding="utf-8")


def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
    """Run a command, its output captured as text."""
    return subprocess.run(command, capture_output=True, text=True, **kwargs)


def main() -> int:
    """Plant each defect in turn and record the tests that fail."""
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    shutil.copytree(ROOT / "core", WORK / "core")
    plant([])
    configure = run(
        [
            "cmake",
            "-S",
            str(WORK / "core"),
            "-B",
            str(WORK / "build"),
            "-G",
            "Ninja",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST={GTEST_SRC}",
        ]
    )
    if configure.returncode != 0:
        raise SystemExit(configure.stdout + configure.stderr)
    records = []
    for name, what, replacements in PLANTS:
        plant(replacements)
        build = run(["ninja", "-C", str(WORK / "build")])
        record: dict[str, object] = {"name": name, "what": what}
        if build.returncode != 0:
            record["build"] = "failed: " + build.stdout[-2000:]
            records.append(record)
            print(name, "build failed")
            continue
        ctest = run(["ctest", "--test-dir", str(WORK / "build")])
        summary = [line for line in ctest.stdout.splitlines() if "tests passed" in line]
        record["ctest"] = summary[0].strip() if summary else ctest.stdout[-500:]
        record["ctest_failed"] = re.findall(r" - (\S+) \(Failed\)", ctest.stdout)
        env = {
            **os.environ,
            "PYTHONPATH": str(WORK / "build"),
            "CFD_CORE_REQUIRED": "1",
        }
        pytest = run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_core_parity.py",
                "-q",
                "-rf",
                "-p",
                "no:cacheprovider",
            ],
            cwd=ROOT,
            env=env,
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
        module = run(
            [sys.executable, "-c", "import _cfdcore; print(_cfdcore.__file__)"],
            env=env,
        )
        record["module"] = module.stdout.strip()
        records.append(record)
        print(name, record["ctest"], "|", record["pytest"])
        OUT.write_text(json.dumps(records, indent=1) + "\n", encoding="utf-8")
    OUT.write_text(json.dumps(records, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
