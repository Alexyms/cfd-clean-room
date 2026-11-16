#!/usr/bin/env bash
# Plant wrong orders in a copy of core/ and see which tests catch each (prompt 86).
#
#   bash docs/reports/probe86/mutants86.sh [PYTHON]
#
# Under WSL2. Each mutant is core/src/reduce.cpp with one change to the order
# of the additions, built out of source in ~/scratch86/<name>; its GoogleTest
# tests run by CTest, and tests/test_core_parity.py runs against its module,
# which PYTHONPATH puts ahead of the installed one. A mutant whose source equals
# the original is reported as not planted. The lines go to
# results/builder86/mutants86.txt. The checkout is not changed.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PYTHON=${1:-${CFD_LINUX_VENV:-$HOME/.venvs/cfd_clean_room}/bin/python}
WORK=$HOME/scratch86
OUT=$ROOT/results/builder86/mutants86.txt

declare -A MUTANT
# A plain left-to-right sum.
MUTANT[sequential]='s|return 0.0 + pairwise(values.data(), values.size());|double s = 0.0; for (double v : values) { s += v; } return 0.0 + s;|'
# The split at n / 2 without rounding it down to a multiple of eight.
MUTANT[split_unrounded]='s|half -= half % unroll;|(void)0;|'
# Blocks of 64 in place of NumPy's 128.
MUTANT[block64]='s|constexpr std::size_t block_size = 128;|constexpr std::size_t block_size = 64;|'
# The eight accumulators combined left to right in place of pairwise.
MUTANT[combine_linear]='s|double sum = ((r\[0\] + r\[1\]) + (r\[2\] + r\[3\])) + ((r\[4\] + r\[5\]) + (r\[6\] + r\[7\]));|double sum = r[0] + r[1] + r[2] + r[3] + r[4] + r[5] + r[6] + r[7];|'
# No leading 0.0, so a sum of negative zeros stays -0.0.
MUTANT[no_identity]='s|return 0.0 + pairwise(|return pairwise(|'

rm -rf "$WORK"
mkdir -p "$WORK" "$(dirname "$OUT")"
: > "$OUT"
for name in sequential split_unrounded block64 combine_linear no_identity; do
    dir=$WORK/$name
    mkdir -p "$dir"
    cp -r "$ROOT/core" "$dir/core"
    sed -i "${MUTANT[$name]}" "$dir/core/src/reduce.cpp"
    if cmp -s "$ROOT/core/src/reduce.cpp" "$dir/core/src/reduce.cpp"; then
        echo "$name | not planted" | tee -a "$OUT"
        continue
    fi
    if ! { cmake -S "$dir/core" -B "$dir/build" -G Ninja -DPython_EXECUTABLE="$PYTHON" \
        && cmake --build "$dir/build"; } > /dev/null; then
        echo "$name | build failed" | tee -a "$OUT"
        continue
    fi
    gtest=$(ctest --test-dir "$dir/build" 2>&1 | grep "tests passed")
    parity=$(cd "$ROOT" && PYTHONPATH="$dir/build" CFD_CORE_REQUIRED=1 "$PYTHON" -m pytest \
        tests/test_core_parity.py -q -p no:cacheprovider 2>&1 | tail -1)
    module=$(PYTHONPATH="$dir/build" "$PYTHON" -c "import _cfdcore; print(_cfdcore.__file__)")
    echo "$name | GoogleTest: $gtest | parity: $parity | module: $module" | tee -a "$OUT"
done
