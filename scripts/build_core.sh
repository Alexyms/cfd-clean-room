#!/usr/bin/env bash
# Build the compiled core out of source, run CTest, and install _cfdcore into
# PYTHON's environment (ADR-017 A and D). The same steps run locally under
# WSL2 and in CI (.github/workflows/ci.yml, the core job).
#
#   bash scripts/build_core.sh [PYTHON]
#
# PYTHON defaults to the WSL2 environment's interpreter
# (scripts/setup_linux_env.sh); the build tree to $CFD_CORE_BUILD or
# ~/build/cfd-core, on the Linux file system, never in the checkout.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${1:-${CFD_LINUX_VENV:-$HOME/.venvs/cfd_clean_room}/bin/python}
BUILD=${CFD_CORE_BUILD:-$HOME/build/cfd-core}

cmake -S "$ROOT/core" -B "$BUILD" -G Ninja \
    -DPython_EXECUTABLE="$PYTHON" -DCFD_CORE_WERROR=ON
cmake --build "$BUILD"
ctest --test-dir "$BUILD" --output-on-failure --output-junit "$BUILD/ctest.xml"
"$PYTHON" -m pip install --no-build-isolation "$ROOT/core"
"$PYTHON" -c "import _cfdcore; print('_cfdcore from', _cfdcore.__file__)"
