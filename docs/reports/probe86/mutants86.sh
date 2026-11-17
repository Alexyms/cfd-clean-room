#!/usr/bin/env bash
# Plant wrong orders and a missing refusal in a copy of core/ and see which tests catch each
# (prompt 86).
#
#   bash docs/reports/probe86/mutants86.sh [PYTHON]
#
# Under WSL2. Each mutant is core/src/reduce.cpp with one change to the order
# of the additions (or, for no_alignment_check, core/python/bindings.cpp without its
# refusal of an unaligned array), built out of source in ~/scratch86/<name>; its GoogleTest
# tests run by CTest, and tests/test_core_parity.py runs against its module,
# which PYTHONPATH puts ahead of the installed one. A mutant whose source equals
# the original is reported as not planted. The first line, "control", is the
# unchanged source, so a reader sees what the tests print when nothing is
# planted. The lines go to results/builder86/mutants86.txt, one per mutant:
#
#   name | GoogleTest: <CTest's summary> [failed: <test names>] | parity: <pytest's last line> | module: <path>
#
# The checkout is not changed.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PYTHON=${1:-${CFD_LINUX_VENV:-$HOME/.venvs/cfd_clean_room}/bin/python}
WORK=$HOME/scratch86
OUT=$ROOT/results/builder86/mutants86.txt

declare -A MUTANT FILE
# The unchanged source (a sed that changes nothing).
MUTANT[control]='s|^$||'
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
# A plain sum at exactly eight terms, the first length the accumulators take.
MUTANT[unroll_le]='s|if (n < unroll) {|if (n <= unroll) {|'
# A block that ends below 128 terms, so exactly 128 terms are split in two.
MUTANT[block_lt]='s|if (n <= block_size) {|if (n < block_size) {|'

# The binding without its refusal of an unaligned array.
FILE[no_alignment_check]=python/bindings.cpp
MUTANT[no_alignment_check]='s|if (reinterpret_cast<std::uintptr_t>(values.data()) % alignof(double) != 0) {|if (false) {|'

rm -rf "$WORK"
mkdir -p "$WORK" "$(dirname "$OUT")"
: > "$OUT"
for name in control sequential split_unrounded block64 combine_linear no_identity unroll_le block_lt no_alignment_check; do
    dir=$WORK/$name
    mkdir -p "$dir"
    cp -r "$ROOT/core" "$dir/core"
    file=${FILE[$name]:-src/reduce.cpp}
    sed -i "${MUTANT[$name]}" "$dir/core/$file"
    if [ "$name" != control ] && cmp -s "$ROOT/core/$file" "$dir/core/$file"; then
        echo "$name | not planted" | tee -a "$OUT"
        continue
    fi
    if ! { cmake -S "$dir/core" -B "$dir/build" -G Ninja -DPython_EXECUTABLE="$PYTHON" \
        && cmake --build "$dir/build"; } > /dev/null; then
        echo "$name | build failed" | tee -a "$OUT"
        continue
    fi
    ctest_out=$(ctest --test-dir "$dir/build" 2>&1)
    gtest=$(echo "$ctest_out" | grep "tests passed")
    failed=$(echo "$ctest_out" | grep -E "\(Failed\)" | sed -E 's/.* - ([^ ]+) \(Failed\)/\1/' | paste -sd, -)
    [ -n "$failed" ] && gtest="$gtest [failed: $failed]"
    parity=$(cd "$ROOT" && PYTHONPATH="$dir/build" CFD_CORE_REQUIRED=1 "$PYTHON" -m pytest \
        tests/test_core_parity.py -q -p no:cacheprovider 2>&1 | tail -1)
    module=$(PYTHONPATH="$dir/build" "$PYTHON" -c "import _cfdcore; print(_cfdcore.__file__)")
    echo "$name | GoogleTest: $gtest | parity: $parity | module: $module" | tee -a "$OUT"
done
