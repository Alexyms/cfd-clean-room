#!/usr/bin/env bash
# Prompt 46 (ECR-002 step 6, the product room under the coupled model): the runs,
# in stages, in parallel, one BLAS thread each. Run from the repository root:
# bash docs/reports/probe46/run46.sh STAGE [ARGS]. Logs go to
# results/builder46/logs/NAME.log; the records to results/builder46/NAME.json and
# NAME.npz. Stages, in the order they were run:
#   time       the three twenty-iteration timing probes (section 5.1), before the set
#   matrix     measurement 1: both variants on every grid, and the 1e-8 check row
#   inlet      measurement 2: 80x30 standard at the two other inlet settings
#   locate GRID VARIANT [ARGS]   a rerun with the cell of the largest change recorded
#   transport NAME               measurement 4 on a converged record
#   figures    every figure beside the report, from the records
#   tables     every table, from the records
set -u
PY=.venv/Scripts/python
PROBE=docs/reports/probe46/coupled46.py
LOGS=results/builder46/logs
mkdir -p "$LOGS"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

launch() {
  local name=$1
  shift
  "$PY" "$@" > "$LOGS/$name.log" 2>&1 &
}

stage=${1:-}
case "$stage" in
  time)
    for grid in 40x15 80x30 200x75; do
      launch "standard_${grid}_time" "$PROBE" time "$grid"
    done
    wait
    ;;
  matrix)
    for grid in 40x15 80x30 200x75; do
      for variant in standard rng; do
        launch "${variant}_${grid}" "$PROBE" run "$grid" "$variant"
      done
    done
    launch standard_200x75_r1e-8 "$PROBE" run 200x75 standard --rtol 1e-8
    wait
    ;;
  inlet)
    launch standard_80x30_i0.02_l0.05 "$PROBE" run 80x30 standard --intensity 0.02 --length 0.05
    launch standard_80x30_i0.1_l0.3 "$PROBE" run 80x30 standard --intensity 0.1 --length 0.3
    wait
    ;;
  locate)
    shift
    launch "${2}_${1}_loc" "$PROBE" run "$1" "$2" --locate "${@:3}"
    wait
    ;;
  transport)
    launch "transport_${2}" docs/reports/probe46/transport46.py run "$2"
    wait
    ;;
  figures)
    "$PY" docs/reports/probe46/figures46.py all
    ;;
  tables)
    "$PY" docs/reports/probe46/tables46.py all
    ;;
  *)
    echo "usage: run46.sh time | matrix | inlet | locate GRID VARIANT [ARGS] | transport NAME | figures | tables" >&2
    exit 2
    ;;
esac
