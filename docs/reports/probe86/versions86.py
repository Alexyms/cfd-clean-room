"""Record the toolchain's versions, Windows side and WSL2 side, for prompt 86.

    python docs/reports/probe86/versions86.py

Run with Windows' ``.venv``. It asks Windows for its build and WSL's version,
then asks the WSL2 distribution for the kernel, GCC, CMake, Ninja, uv, the
Linux environment's Python, NumPy and its BLAS (``np.show_config``), nanobind
and scikit-build-core, and reads the GoogleTest release pinned in
``core/CMakeLists.txt``. Everything goes to ``results/builder86/versions.json``,
which ``tables86.py`` reads. It starts no process beyond the commands named.
"""

import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "results" / "builder86" / "versions.json"
DISTRO = "Ubuntu-24.04"
LINUX_PYTHON = "$HOME/.venvs/cfd_clean_room/bin/python"
# The commands asked of the distribution, name to shell text.
LINUX = {
    "kernel": "uname -r",
    "os": "grep ^PRETTY_NAME= /etc/os-release",
    "gcc": "gcc --version | head -1",
    "glibc": "ldd --version | head -1",
    "cmake": "cmake --version | head -1",
    "ninja": "ninja --version",
    "uv": "$HOME/.local/bin/uv --version",
    "pipx": "pipx --version",
    "python": f"{LINUX_PYTHON} --version",
    "numpy": f"{LINUX_PYTHON} -c 'import numpy; print(numpy.__version__)'",
    "nanobind": f"{LINUX_PYTHON} -m pip show nanobind | grep ^Version",
    "scikit-build-core": f"{LINUX_PYTHON} -m pip show scikit-build-core | grep ^Version",
}
NUMPY_CONFIG = (
    f"{LINUX_PYTHON} -c 'import json, numpy; "
    'print(json.dumps(numpy.show_config(mode="dicts"), default=str))\''
)


def _run(args: list[str]) -> str:
    """ARGS' standard output, stripped; WSL's own output is UTF-16."""
    out = subprocess.run(args, capture_output=True, check=True).stdout
    text = out.decode("utf-16-le") if out[1:2] == b"\x00" else out.decode()
    return text.replace("\x00", "").strip()


def linux(command: str) -> str:
    """COMMAND's output in the distribution."""
    return _run(["wsl", "-d", DISTRO, "--", "bash", "-lc", command])


def googletest_pin() -> str:
    """The GoogleTest release core/CMakeLists.txt fetches."""
    text = (ROOT / "core" / "CMakeLists.txt").read_text(encoding="utf-8")
    match = re.search(r"googletest/releases/download/(v[0-9.]+)/", text)
    return match.group(1) if match else "not found"


def main() -> int:
    """Collect and write the versions."""
    record = {
        "windows": {
            "build": _run(["cmd", "/c", "ver"]),
            "platform": platform.platform(),
            "wsl": _run(["wsl", "--version"]).splitlines(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "numpy_config": np.show_config(mode="dicts"),
        },
        "linux": {name: linux(command) for name, command in LINUX.items()},
        "googletest": googletest_pin(),
    }
    record["linux"]["numpy_config"] = json.loads(linux(NUMPY_CONFIG))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
