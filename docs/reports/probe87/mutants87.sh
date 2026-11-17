#!/usr/bin/env bash
# Plant defects in a copy of core/ and see which tests catch each (prompt 87).
#
#   bash docs/reports/probe87/mutants87.sh [PYTHON]
#
# Under WSL2. Each mutant is one change to core/src/scalar.cpp or
# core/include/cfdcore/scalar.hpp, built out of source in ~/scratch87/mutants/<name>
# with the main build's GoogleTest sources (no download); its GoogleTest tests run
# by CTest, and tests/test_core_parity.py runs against its module, which
# PYTHONPATH puts ahead of the installed one. A mutant whose source equals the
# original is reported as not planted. The first line, "control", is the
# unchanged source. The lines go to results/builder87/mutants87.txt:
#
#   name | GoogleTest: <CTest's summary> [failed: <test names>] | parity: <pytest's last line> | module: <path>
#
# The checkout is not changed.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PYTHON=${1:-${CFD_LINUX_VENV:-$HOME/.venvs/cfd_clean_room}/bin/python}
WORK=$HOME/scratch87/mutants
GTEST_SRC=${CFD_CORE_BUILD:-$HOME/build/cfd-core}/_deps/googletest-src
OUT=$ROOT/results/builder87/mutants87.txt
SRC=src/scalar.cpp
HDR=include/cfdcore/scalar.hpp

declare -A MUTANT FILE
NAMES=(control swapped_weight far_solid_dropped clamp_two_raised clamp_two_lowered
       vent_reversed quick_reassociated tie_first boundary_zero_flux)
# The unchanged source (a sed that changes nothing).
MUTANT[control]='s|^$||'
# QUICK's far-upstream and downstream weights swapped, flow toward +x.
MUTANT[swapped_weight]='s|return w.positive\[0\]\[f\] \* line.node(f) + w.positive\[1\]\[f\] \* line.node(f + 1) +|return w.positive[2][f] * line.node(f) + w.positive[1][f] * line.node(f + 1) +|; s|^               w.positive\[2\]\[f\] \* line.node(f + 2);|               w.positive[0][f] * line.node(f + 2);|'
# The far-solid rule dropped: a far node inside an obstacle read as it is.
MUTANT[far_solid_dropped]='s|if (far_solid) {|if (false) {|'
# The clamp's cap of 2 changed, up and down.
FILE[clamp_two_raised]=$HDR
MUTANT[clamp_two_raised]='s|low = np_minimum(low, 2.0);|low = np_minimum(low, 2.5);|'
FILE[clamp_two_lowered]=$HDR
MUTANT[clamp_two_lowered]='s|low = np_minimum(low, 2.0);|low = np_minimum(low, 1.5);|'
# The vent rule reversed: the room cell's value entering, the inflow value leaving.
MUTANT[vent_reversed]='s|value = entering ? inflow\[face\] : room;|value = entering ? room : inflow[face];|'
# QUICK's sum reassociated, w0 a + (w1 b + w2 c), flow toward +x.
MUTANT[quick_reassociated]='s|return w.positive\[0\]\[f\] \* line.node(f) + w.positive\[1\]\[f\]|return w.positive[0][f] * line.node(f) + (w.positive[1][f]|; s|^               w.positive\[2\]\[f\] \* line.node(f + 2);|               w.positive[2][f] * line.node(f + 2));|'
# np.minimum's tie returning the first argument: the sign of a zero psi moves.
FILE[tie_first]=$HDR
MUTANT[tie_first]='s|return (a < b \|\| std::isnan(a)) ? a : b; }|return (a <= b \|\| std::isnan(a)) ? a : b; }|'
# A zero flux at the low boundary face read as inward (the inflow value).
MUTANT[boundary_zero_flux]='s|const double c_low = flux\[low_face\] > 0.0 ? inflow\[low_face\] : c\[ln.cell\];|const double c_low = flux[low_face] >= 0.0 ? inflow[low_face] : c[ln.cell];|'

rm -rf "$WORK"
mkdir -p "$WORK" "$(dirname "$OUT")"
: > "$OUT"
for name in "${NAMES[@]}"; do
    dir=$WORK/$name
    mkdir -p "$dir"
    cp -r "$ROOT/core" "$dir/core"
    file=${FILE[$name]:-$SRC}
    sed -i "${MUTANT[$name]}" "$dir/core/$file"
    if [ "$name" != control ] && cmp -s "$ROOT/core/$file" "$dir/core/$file"; then
        echo "$name | not planted" | tee -a "$OUT"
        continue
    fi
    if ! { cmake -S "$dir/core" -B "$dir/build" -G Ninja -DPython_EXECUTABLE="$PYTHON" \
        -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST="$GTEST_SRC" \
        && cmake --build "$dir/build"; } > /dev/null; then
        echo "$name | build failed" | tee -a "$OUT"
        continue
    fi
    ctest_out=$(ctest --test-dir "$dir/build" 2>&1)
    gtest=$(echo "$ctest_out" | grep "tests passed")
    failed=$(echo "$ctest_out" | grep -E "\(Failed\)" | sed -E 's/.* - ([^ ]+) \(Failed\)/\1/' | paste -sd, -)
    [ -n "$failed" ] && gtest="$gtest [failed: $failed]"
    pytest_out=$(cd "$ROOT" && PYTHONPATH="$dir/build" CFD_CORE_REQUIRED=1 "$PYTHON" -m pytest \
        tests/test_core_parity.py -q -rf -p no:cacheprovider 2>&1)
    parity=$(echo "$pytest_out" | tail -1)
    # The failing tests by function, with their counts of failing cases.
    by_test=$(echo "$pytest_out" | grep "^FAILED" \
        | sed -E 's/^FAILED tests\/test_core_parity.py::([A-Za-z0-9_]+).*/\1/' \
        | sort | uniq -c | awk '{print $2 " " $1}' | paste -sd, -)
    [ -n "$by_test" ] && parity="$parity [failed: $by_test]"
    module=$(PYTHONPATH="$dir/build" "$PYTHON" -c "import _cfdcore; print(_cfdcore.__file__)")
    echo "$name | GoogleTest: $gtest | parity: $parity | module: $module" | tee -a "$OUT"
done
