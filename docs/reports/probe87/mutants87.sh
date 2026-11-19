#!/usr/bin/env bash
# Plant defects in a copy of core/ and see which tests catch each (prompts 87 and 87b).
#
#   bash docs/reports/probe87/mutants87.sh [PYTHON]
#
# Under WSL2. Each plant is one change to one file of core/, built in a copy
# out of source in ~/scratch87/mutants with the main build's GoogleTest sources
# (no download); its GoogleTest tests run by CTest, and tests/test_core_parity.py
# runs against its module, which PYTHONPATH puts ahead of the installed one. The
# build is shared and incremental: before each plant the previous plant's file
# is restored from the checkout, so every plant starts from the unchanged
# source. A plant whose source equals the original is reported as not planted.
# The first line, "control", is the unchanged source, and the last, "control_end",
# is the unchanged source again after every plant has been restored. The lines go
# to results/builder87/mutants87.txt:
#
#   name | GoogleTest: <CTest's summary> [failed: <test names>] | parity: <pytest's last line> | module: <path>
#
# and the failing parity cases of each plant, by test id, to
# results/builder87/mutants87_failed/<name>.txt. The checkout is not changed.
#
# The plants fall in groups by name: the scalar module's own rules (prompt 87),
# the binding's refusals of a wrong shape, layout, alignment or aliased output
# (bind_*) and the grid's refusals (grid_*) (prompt 87b). Each bind_* and grid_*
# plant removes one check, and the tests that pin it must fail.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PYTHON=${1:-${CFD_LINUX_VENV:-$HOME/.venvs/cfd_clean_room}/bin/python}
WORK=$HOME/scratch87/mutants
GTEST_SRC=${CFD_CORE_BUILD:-$HOME/build/cfd-core}/_deps/googletest-src
OUT=$ROOT/results/builder87/mutants87.txt
FAILED=$ROOT/results/builder87/mutants87_failed
SRC=src/scalar.cpp
HDR=include/cfdcore/scalar.hpp
BIND=python/bindings.cpp
GRID=src/grid.cpp

declare -A MUTANT FILE
NAMES=(control
       # The scalar module's rules.
       swapped_weight far_solid_dropped far_solid_quick_deleted clamp_two_raised
       clamp_two_lowered vent_reversed quick_reassociated tie_first boundary_zero_flux
       # The binding's shape checks, each input alone (the alignment check stays).
       bind_flux_shape_c bind_flux_shape_flux bind_flux_shape_inflow bind_flux_shape_out
       bind_flux_shape_vents bind_quick_shape_c bind_quick_shape_positive
       bind_quick_shape_out bind_quick_lines_unchecked bind_quick_lines_skip_c_high
       bind_quick_lines_skip_c_low bind_limited_shapes_unchecked
       # The binding's layout check, nb::c_contig of each array alias.
       bind_contig_vector bind_contig_int_matrix bind_contig_matrix bind_contig_bool_matrix
       bind_contig_out_matrix bind_contig_array bind_contig_out_array
       # The binding's refusal of a conversion (a copy) instead of the array.
       bind_noconvert_all bind_noconvert_out
       # The binding's alignment checks.
       bind_aligned_matrix bind_aligned_quick_lines bind_aligned_grid_arrays
       bind_aligned_cell_type bind_aligned_limited_operands bind_aligned_limited_out
       # The binding's refusal of an out that shares memory with an input.
       bind_apart_flux_c bind_apart_flux_flux bind_apart_flux_inflow bind_apart_flux_vents
       bind_apart_quick_c bind_apart_quick_c_low bind_apart_quick_c_high
       bind_apart_quick_positive bind_apart_limited bind_apart_first_half
       bind_apart_second_half
       # The grid's shape and coordinate checks.
       grid_cell_type_shape grid_faces_count grid_no_cell grid_coords_finite
       grid_coords_faces_order grid_coords_centres_order grid_coords_low_end
       grid_coords_high_end grid_coords_direction grid_coords_unchecked
       grid_coords_type_error
       control_end)

# The unchanged source (a sed that changes nothing).
MUTANT[control]='s|^$||'
MUTANT[control_end]='s|^$||'

# ---- The scalar module's rules (prompt 87)
# QUICK's far-upstream and downstream weights swapped, flow toward +x.
MUTANT[swapped_weight]='s|return w.positive\[0\]\[f\] \* line.node(f) + w.positive\[1\]\[f\] \* line.node(f + 1) +|return w.positive[2][f] * line.node(f) + w.positive[1][f] * line.node(f + 1) +|; s|^               w.positive\[2\]\[f\] \* line.node(f + 2);|               w.positive[0][f] * line.node(f + 2);|'
# The far-solid rule dropped: a far node inside an obstacle read as it is.
MUTANT[far_solid_dropped]='s|if (far_solid) {|if (false) {|'
# The far-solid rule's `quick = c_c` line deleted: only c_up is read as the upstream value.
MUTANT[far_solid_quick_deleted]='/^                    quick = c_c;$/d'
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

# ---- The binding's shape checks (prompt 87b). Each replaces the input's
# require_shape by its alignment check alone, inside advective_flux's checks
# (check_flux) or quick_face_values', so the shape comparison is the only thing lost.
FLUX='/^flux_arguments check_flux/,/^}/'
QUICK='/^void quick_face_values(const Scheme2/,/^}/'
LIMITED='/^void limited_face_values(/,/^}/'
for name in flux_shape_c flux_shape_flux flux_shape_inflow flux_shape_out flux_shape_vents \
            quick_shape_c quick_shape_positive quick_shape_out quick_lines_unchecked \
            quick_lines_skip_c_high quick_lines_skip_c_low limited_shapes_unchecked; do
    FILE[bind_$name]=$BIND
done
MUTANT[bind_flux_shape_c]="$FLUX"'s|require_shape(c, {g.cells(1), g.cells(0)}, who, "c");|require_aligned(c.data(), alignof(double), who);|'
MUTANT[bind_flux_shape_flux]="$FLUX"'s|require_shape(flux, faces, who, "flux");|require_aligned(flux.data(), alignof(double), who);|'
MUTANT[bind_flux_shape_inflow]="$FLUX"'s|require_shape(inflow, faces, who, "inflow");|require_aligned(inflow.data(), alignof(double), who);|'
MUTANT[bind_flux_shape_out]="$FLUX"'s|require_shape(out, faces, who, "out");|require_aligned(out.data(), alignof(double), who);|'
MUTANT[bind_flux_shape_vents]="$FLUX"'s|require_shape(\*vents, faces, who, "vents");|require_aligned(vents->data(), alignof(bool), who);|'
MUTANT[bind_quick_shape_c]="$QUICK"'s|require_shape(c, {g.cells(1), g.cells(0)}, who, "c");|require_aligned(c.data(), alignof(double), who);|'
MUTANT[bind_quick_shape_positive]="$QUICK"'s|require_shape(positive, faces, who, "positive");|require_aligned(positive.data(), alignof(bool), who);|'
MUTANT[bind_quick_shape_out]="$QUICK"'s|require_shape(out, faces, who, "out");|require_aligned(out.data(), alignof(double), who);|'
# The line counts of c_low and c_high: both unchecked, then each skipped in turn
# (a skipped vector loses its alignment check too).
MUTANT[bind_quick_lines_unchecked]="$QUICK"'s|if (v->shape(0) != lines) {|if (false) {|'
MUTANT[bind_quick_lines_skip_c_high]="$QUICK"'s|for (const auto\* v : {&c_low, &c_high}) {|for (const auto* v : {\&c_low}) {|'
MUTANT[bind_quick_lines_skip_c_low]="$QUICK"'s|for (const auto\* v : {&c_low, &c_high}) {|for (const auto* v : {\&c_high}) {|'
MUTANT[bind_limited_shapes_unchecked]="$LIMITED"'s|if (!same) {|if (false) {|'

# ---- The binding's layout check: nb::c_contig dropped from one array alias.
for alias in Vector IntMatrix Matrix BoolMatrix OutMatrix Array OutArray; do
    lower=$(echo "$alias" | sed -E 's/([a-z])([A-Z])/\1_\2/g' | tr 'A-Z' 'a-z')
    FILE[bind_contig_$lower]=$BIND
    MUTANT[bind_contig_$lower]='s|^\(using '"$alias"' = .*\)nb::c_contig, |\1|'
done

# ---- The binding's refusal of a conversion: .noconvert() dropped from every
# argument, and from the outputs alone (a converted out would be a copy that
# the caller never sees written).
FILE[bind_noconvert_all]=$BIND
MUTANT[bind_noconvert_all]='s|\.noconvert()||g'
FILE[bind_noconvert_out]=$BIND
MUTANT[bind_noconvert_out]='s|nb::arg("out")\.noconvert()|nb::arg("out")|g'

# ---- The binding's alignment checks, each deleted.
FILE[bind_aligned_matrix]=$BIND
MUTANT[bind_aligned_matrix]='/^    require_aligned(a.data(), alignof(std::remove_cv_t<typename A::Scalar>), who);$/d'
FILE[bind_aligned_quick_lines]=$BIND
MUTANT[bind_aligned_quick_lines]='/^        require_aligned(v->data(), alignof(double), who);$/d'
FILE[bind_aligned_grid_arrays]=$BIND
MUTANT[bind_aligned_grid_arrays]='/^        require_aligned(v->data(), alignof(double), "Grid2");$/d'
FILE[bind_aligned_cell_type]=$BIND
MUTANT[bind_aligned_cell_type]='/^    require_aligned(cell_type.data(), alignof(std::int32_t), "Grid2");$/d'
FILE[bind_aligned_limited_operands]=$BIND
MUTANT[bind_aligned_limited_operands]='/^        require_aligned(a->data(), alignof(double), who);$/d'
FILE[bind_aligned_limited_out]=$BIND
MUTANT[bind_aligned_limited_out]='/^    require_aligned(out.data(), alignof(double), who);$/d'

# ---- The binding's aliasing checks: one input dropped from advective_flux's
# list, or one require_apart line deleted, and then each half of the overlap test.
for name in flux_c flux_flux flux_inflow flux_vents quick_c quick_c_low quick_c_high \
            quick_positive limited first_half second_half; do
    FILE[bind_apart_$name]=$BIND
done
MUTANT[bind_apart_flux_c]='s|for (const auto& in : {args.c, args.flux, args.inflow}) {|for (const auto\& in : {args.flux, args.inflow}) {|'
MUTANT[bind_apart_flux_flux]='s|for (const auto& in : {args.c, args.flux, args.inflow}) {|for (const auto\& in : {args.c, args.inflow}) {|'
MUTANT[bind_apart_flux_inflow]='s|for (const auto& in : {args.c, args.flux, args.inflow}) {|for (const auto\& in : {args.c, args.flux}) {|'
MUTANT[bind_apart_flux_vents]='/^        require_apart(out.data(), out_bytes, args.vents.data(), args.vents.size_bytes(), who);$/d'
MUTANT[bind_apart_quick_c]='/^    require_apart(out.data(), out_bytes, c.data(), c.size() \* sizeof(double), who);$/d'
MUTANT[bind_apart_quick_c_low]='/^    require_apart(out.data(), out_bytes, c_low.data(), lines \* sizeof(double), who);$/d'
MUTANT[bind_apart_quick_c_high]='/^    require_apart(out.data(), out_bytes, c_high.data(), lines \* sizeof(double), who);$/d'
MUTANT[bind_apart_quick_positive]='/^    require_apart(out.data(), out_bytes, positive.data(), positive.size(), who);$/d'
MUTANT[bind_apart_limited]='/^        require_apart(out.data(), out.size() \* sizeof(double), a->data(),$/,/who);$/d'
MUTANT[bind_apart_first_half]='s|if (o < i + in_bytes && i < o + out_bytes) {|if (o < i + in_bytes) {|'
MUTANT[bind_apart_second_half]='s|if (o < i + in_bytes && i < o + out_bytes) {|if (i < o + out_bytes) {|'

# ---- The grid's checks (prompt 87b): the binding's cell_type shape, the grid's
# own faces and no-cell checks, each clause of the coordinate check, the
# direction it names, and the binding's turning it into a ValueError.
FILE[grid_cell_type_shape]=$BIND
MUTANT[grid_cell_type_shape]='s#if (cell_type.shape(0) != yc.shape(0) || cell_type.shape(1) != xc.shape(0)) {#if (false) {#'
FILE[grid_faces_count]=$GRID
MUTANT[grid_faces_count]='s|if (input.faces\[d\].size() != input.centres\[d\].size() + 1) {|if (false) {|'
FILE[grid_no_cell]=$GRID
MUTANT[grid_no_cell]='s|if (input.centres\[d\].empty()) {|if (false) {|'
FILE[grid_coords_finite]=$GRID
MUTANT[grid_coords_finite]='s|if (!std::isfinite(face)) {|if (false) {|'
FILE[grid_coords_faces_order]=$GRID
MUTANT[grid_coords_faces_order]='s|if (!strictly_increasing(faces)) {|if (false) {|'
FILE[grid_coords_centres_order]=$GRID
MUTANT[grid_coords_centres_order]='s#|| !strictly_increasing(centres) ||#|| false ||#'
FILE[grid_coords_low_end]=$GRID
MUTANT[grid_coords_low_end]='s#if (!(faces.front() < centres.front()) ||#if (false ||#'
FILE[grid_coords_high_end]=$GRID
MUTANT[grid_coords_high_end]='s#^            !(centres.back() < faces.back())) {#            false) {#'
FILE[grid_coords_direction]=$GRID
MUTANT[grid_coords_direction]='/^int grid<D>::first_bad_direction/,/^}/s|return d;|return 0;|'
FILE[grid_coords_unchecked]=$BIND
MUTANT[grid_coords_unchecked]='s|if (bad >= 0) {|if (false) {|'
FILE[grid_coords_type_error]=$BIND
MUTANT[grid_coords_type_error]='s|throw nb::value_error((std::string("Grid2: the face or node|throw nb::type_error((std::string("Grid2: the face or node|'

rm -rf "$WORK"
mkdir -p "$WORK" "$(dirname "$OUT")" "$FAILED"
rm -f "$FAILED"/*.txt
: > "$OUT"
cp -r "$ROOT/core" "$WORK/core"
if ! { cmake -S "$WORK/core" -B "$WORK/build" -G Ninja -DPython_EXECUTABLE="$PYTHON" \
    -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST="$GTEST_SRC" && ninja -C "$WORK/build"; } > /dev/null; then
    echo "the unchanged source does not build" >&2
    exit 1
fi

planted=
for name in "${NAMES[@]}"; do
    # Restore the file the previous plant changed, so this one starts from the checkout's source.
    if [ -n "$planted" ]; then
        cp "$ROOT/core/$planted" "$WORK/core/$planted"
        planted=
    fi
    file=${FILE[$name]:-$SRC}
    if [ "$name" != control ] && [ "$name" != control_end ]; then
        sed -i "${MUTANT[$name]}" "$WORK/core/$file"
        planted=$file
        if cmp -s "$ROOT/core/$file" "$WORK/core/$file"; then
            echo "$name | not planted" | tee -a "$OUT"
            continue
        fi
    fi
    if ! ninja -C "$WORK/build" > "$WORK/build.log" 2>&1; then
        echo "$name | build failed" | tee -a "$OUT"
        continue
    fi
    ctest_out=$(ctest --test-dir "$WORK/build" 2>&1)
    gtest=$(echo "$ctest_out" | grep "tests passed")
    failed=$(echo "$ctest_out" | grep -E "\(Failed\)" | sed -E 's/.* - ([^ ]+) \(Failed\)/\1/' | paste -sd, -)
    [ -n "$failed" ] && gtest="$gtest [failed: $failed]"
    # Verbose, so that each case's result is printed as it finishes: a plant that
    # lets a write go past an array's end can kill the run (a crash is a result),
    # and the cases that failed before it are still listed.
    pytest_out=$(cd "$ROOT" && PYTHONPATH="$WORK/build" CFD_CORE_REQUIRED=1 "$PYTHON" -m pytest \
        tests/test_core_parity.py -v -p no:cacheprovider 2>&1; echo "pytest exit $?")
    status=$(echo "$pytest_out" | tail -1 | sed 's/^pytest exit //')
    echo "$pytest_out" | grep -E '^tests/test_core_parity.py::.* FAILED' \
        | sed -E 's/^tests\/test_core_parity.py:://; s/ FAILED.*//' > "$FAILED/$name.txt"
    if [ "$status" = 0 ] || [ "$status" = 1 ]; then
        parity=$(echo "$pytest_out" | grep -E '^=+ [0-9]+ .* in [0-9.]+s' | tail -1 | sed -E 's/^=+ //; s/ =+$//')
    else
        passed=$(echo "$pytest_out" | grep -cE '^tests/test_core_parity.py::.* PASSED')
        last=$(echo "$pytest_out" | grep -E '^tests/test_core_parity.py::' | tail -1 \
            | sed -E 's/^tests\/test_core_parity.py:://; s/ .*//')
        parity="the run crashed (exit $status) at $last after $passed passed and $(grep -c . "$FAILED/$name.txt") failed"
    fi
    # The failing tests by function, with their counts of failing cases.
    by_test=$(sed -E 's/\[.*//' "$FAILED/$name.txt" | sort | uniq -c | awk '{print $2 " " $1}' | paste -sd, -)
    [ -n "$by_test" ] && parity="$parity [failed: $by_test]"
    module=$(PYTHONPATH="$WORK/build" "$PYTHON" -c "import _cfdcore; print(_cfdcore.__file__)")
    echo "$name | GoogleTest: $gtest | parity: $parity | module: $module" | tee -a "$OUT"
done
