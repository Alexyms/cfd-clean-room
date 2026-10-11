#!/usr/bin/env bash
# Prompt 49 (ECR-002 step 7, VAL-017): the Annex 20 runs, in parallel, one
# BLAS thread each. Run from the repository root:
# bash docs/reports/probe49/run49.sh STAGE. Logs go to results/builder49/logs/
# NAME.log; the records to results/builder49/NAME.json, NAME.npz and
# lines_NAME.json. Stages:
#   flows     item 3: standard under upwind (the default, key absent) and
#             umist on 216x72 and 375x125, and RNG under the default on 216x72
#   tables    item 4: every table, from the records (tables49.py)
#   figures   items 3 and 4: every figure (figures49.py)
set -u
PY=.venv/Scripts/python
PROBE=docs/reports/probe49/annex49.py
LOGS=results/builder49/logs
mkdir -p "$LOGS"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

launch() {
  local name=$1
  shift
  "$PY" "$@" > "$LOGS/$name.log" 2>&1 &
}

stage=${1:-}
case "$stage" in
  flows)
    for grid in 216x72 375x125; do
      launch "standard_${grid}_upwind" "$PROBE" run "$grid"
      launch "standard_${grid}_umist" "$PROBE" run "$grid" --scheme umist
    done
    launch rng_216x72_upwind "$PROBE" run 216x72 --variant rng
    wait
    ;;
  tables)
    "$PY" docs/reports/probe49/tables49.py all
    ;;
  figures)
    "$PY" docs/reports/probe49/figures49.py all
    ;;
  *)
    echo "usage: run49.sh flows | tables | figures" >&2
    exit 2
    ;;
esac
