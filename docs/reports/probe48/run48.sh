#!/usr/bin/env bash
# Prompt 48 (ECR-002 step 6, k and eps advected by upwind and the variant
# check): the runs, in stages, in parallel, one BLAS thread each. Run from the
# repository root: bash docs/reports/probe48/run48.sh STAGE [ARGS]. Logs go to
# results/builder48/logs/NAME.log; the records to results/builder48/NAME.json
# and NAME.npz. Stages, in the order they were run:
#   laminar LABEL [TREE]  item 3: VAL-001, its stretched twin and VAL-002 on
#              TREE (default this checkout), labelled LABEL (check48.py)
#   flows      items 4 and 5: the standard variant on 80x30, 160x60 and
#              320x120 and RNG on 160x60 and 320x120, the scheme key absent
#   march NAME SOURCE [ARGS]   item 6: one source's march on a converged record
#   marches NAME [NAME ...]    item 6: S1, S2 and S3 on each named record
#   bounded NAME [NAME ...]    bounded44's characterisation of bounded rows
#   figures    every figure, drawn under results/builder48/figures/
#   tables     every table, from the records
set -u
PY=.venv/Scripts/python
PROBE=docs/reports/probe48/coupled48.py
TRANSPORT=docs/reports/probe48/transport48.py
LOGS=results/builder48/logs
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
      launch "val_${c}_$2" docs/reports/probe48/check48.py --root "$tree" val "$c" --label "$2"
    done
    wait
    ;;
  flows)
    launch standard_80x30_upwind "$PROBE" run 80x30 standard
    for grid in 160x60 320x120; do
      for variant in standard rng; do
        launch "${variant}_${grid}_upwind" "$PROBE" run "$grid" "$variant"
      done
    done
    wait
    ;;
  march)
    launch "transport_${2}_${3}" "$TRANSPORT" run "$2" "$3" "${@:4}"
    wait
    ;;
  marches)
    shift
    for name in "$@"; do
      for source in S1 S2 S3; do
        launch "transport_${name}_${source}" "$TRANSPORT" run "$name" "$source"
      done
    done
    wait
    ;;
  bounded)
    shift
    for name in "$@"; do
      "$PY" docs/reports/probe48/bounded48.py history "$name" > "$LOGS/bounded_$name.log"
      "$PY" docs/reports/probe48/bounded48.py where "$name" > "$LOGS/bounded_where_$name.log"
    done
    ;;
  figures)
    "$PY" docs/reports/probe48/figures48.py all
    ;;
  tables)
    "$PY" docs/reports/probe48/tables48.py all
    ;;
  *)
    echo "usage: run48.sh laminar LABEL [TREE] | flows | march NAME SOURCE | marches NAME... | bounded NAME... | figures | tables" >&2
    exit 2
    ;;
esac
