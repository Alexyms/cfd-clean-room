#!/usr/bin/env bash
# The WSL2 Python environment for the compiled core (ADR-017 A, prompt 86).
#
#   bash scripts/setup_linux_env.sh FREEZE
#
# FREEZE is the Windows environment's `pip freeze`, so the Linux environment
# holds the same interpreter patch release and the same package versions. The
# environment lives on the Linux file system, at $CFD_LINUX_VENV or
# ~/.venvs/cfd_clean_room: this machine mounts C: with uid 0 and no metadata,
# so pip cannot set file modes inside the checkout. User space only: uv through
# pipx, Astral's CPython build through uv, packages from PyPI.
set -euo pipefail

PYTHON_VERSION=3.13.3
FREEZE=${1:?usage: setup_linux_env.sh FREEZE}
VENV=${CFD_LINUX_VENV:-$HOME/.venvs/cfd_clean_room}
ROOT=$(cd "$(dirname "$0")/.." && pwd)

if ! command -v uv > /dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
    pipx install uv
fi
UV=$(command -v uv || echo "$HOME/.local/bin/uv")
"$UV" python install "$PYTHON_VERSION"
PYTHON=$("$UV" python find "$PYTHON_VERSION")

"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --only-binary=:all: -r "$FREEZE"
# The core's build dependencies, which the Windows environment does not hold.
"$VENV/bin/python" -m pip install --only-binary=:all: -r "$ROOT/requirements-dev.txt"
"$VENV/bin/python" --version
