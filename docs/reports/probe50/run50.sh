#!/usr/bin/env bash
# Prompt 50 (ECR-002 step 7, condition (e)'s scale): the runs, in stages, in
# parallel, one BLAS thread each. Run from the repository root:
# bash docs/reports/probe50/run50.sh STAGE [ARGS]. Logs go to
# results/builder50/logs/NAME.log, the records to results/builder50/. Stages,
# in the order they were run:
#   laminar LABEL [TREE]  item 3: VAL-001, its stretched twin and VAL-002 on
#              TREE (default this checkout), labelled LABEL (check50.py)
#   flows      items 6 and 7: the Annex 20 harness row at the file's k and eps
#              tolerance and at 1e-13, each to 20,000 outer iterations, and
#              prompt 48's standard upwind product rows on 80x30 and 160x60
#   compare    item 7: each product row against prompt 48's (product50.py)
#   tables     every table, from the records (tables50.py)
set -u
PY=.venv/Scripts/python
PROBE=docs/reports/probe50
LOGS=results/builder50/logs
mkdir -p "$LOGS"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

launch() {
  local name=$1
  shift
  "$PY" "$@" > "$LOGS/$name.log" 2>&1 &
}

stage=${1:-}
case "$stage" in
  laminar)
    tree=${3:-.}
    for c in val001_80x40 val001_80x40_stretched val002_80x80; do
      launch "check_${c}_$2" "$PROBE/check50.py" --root "$tree" val "$c" --label "$2"
    done
    wait
    ;;
  flows)
    launch annex_file "$PROBE/annex50.py" run file --cap 20000 --concurrent 4
    launch annex_tol1e-13 "$PROBE/annex50.py" run tol1e-13 --tol 1e-13 --cap 20000 --concurrent 4
    for grid in 80x30 160x60; do
      launch "standard_${grid}_upwind" "$PROBE/product50.py" run "$grid"
    done
    wait
    ;;
  compare)
    for grid in 80x30 160x60; do
      "$PY" "$PROBE/product50.py" compare "$grid"
    done
    ;;
  tables)
    "$PY" "$PROBE/tables50.py"
    ;;
  *)
    echo "usage: run50.sh laminar LABEL [TREE] | flows | compare | tables" >&2
    exit 2
    ;;
esac
