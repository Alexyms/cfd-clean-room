#!/usr/bin/env bash
# Prompt 47 (ECR-002 step 6, the exact grid pair and the comparative check): the
# runs, in stages, in parallel, one BLAS thread each. Run from the repository
# root: bash docs/reports/probe47/run47.sh STAGE [ARGS]. Logs go to
# results/builder47/logs/NAME.log; the records to results/builder47/NAME.json
# and NAME.npz. Stages, in the order they were run:
#   rooms      which positions each grid represents exactly (section 5.1)
#   time       the two timing probes on the exact grids, before the set
#   cycle      measurement 1: the 80x30 control, the three counterfactuals and
#              the limiter diagnostic
#   pair       measurement 2: both variants on 160x60 and 320x120
#   pair-upwind  the supplementary pair: the upwind arm of measurement 1 on both
#              exact grids, launched when the standard 320x120 row's residual
#              had sat flat for 900 iterations (section 5.3)
#   check NAME SOURCE [ARGS]   measurement 3's discrimination check, one source,
#              on a converged record (transport47.py check)
#   march NAME SOURCE [ARGS]   measurement 3's march, one source, on a converged
#              record (transport47.py run)
#   bounded NAME [NAME ...]    bounded44's characterisation of bounded rows
#   figures    every figure beside the report, from the records
#   tables     every table, from the records
set -u
PY=.venv/Scripts/python
PROBE=docs/reports/probe47/coupled47.py
TRANSPORT=docs/reports/probe47/transport47.py
LOGS=results/builder47/logs
mkdir -p "$LOGS"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

launch() {
  local name=$1
  shift
  "$PY" "$@" > "$LOGS/$name.log" 2>&1 &
}

stage=${1:-}
case "$stage" in
  rooms)
    "$PY" docs/reports/probe47/rooms47.py
    ;;
  time)
    for grid in 160x60 320x120; do
      launch "standard_${grid}_time" "$PROBE" time "$grid"
    done
    wait
    ;;
  cycle)
    for arm in control upwind corner alpha limiter; do
      launch "standard_80x30_${arm}" "$PROBE" run 80x30 standard --arm "$arm"
    done
    wait
    ;;
  pair)
    for grid in 160x60 320x120; do
      for variant in standard rng; do
        launch "${variant}_${grid}" "$PROBE" run "$grid" "$variant"
      done
    done
    wait
    ;;
  pair-upwind)
    for grid in 160x60 320x120; do
      launch "standard_${grid}_upwind" "$PROBE" run "$grid" standard --arm upwind
    done
    wait
    ;;
  check)
    launch "check_${3}_${2}" "$TRANSPORT" check "$2" "$3" "${@:4}"
    wait
    ;;
  march)
    launch "transport_${3}_${2}" "$TRANSPORT" run "$2" "$3" "${@:4}"
    wait
    ;;
  bounded)
    shift
    for name in "$@"; do
      "$PY" docs/reports/probe47/bounded47.py history "$name" > "$LOGS/bounded_$name.log"
      "$PY" docs/reports/probe47/bounded47.py where "$name" > "$LOGS/bounded_where_$name.log"
    done
    ;;
  figures)
    "$PY" docs/reports/probe47/figures47.py all
    ;;
  tables)
    "$PY" docs/reports/probe47/tables47.py all
    ;;
  *)
    echo "usage: run47.sh rooms | time | cycle | pair | pair-upwind | check NAME SOURCE | march NAME SOURCE | bounded NAME... | figures | tables" >&2
    exit 2
    ;;
esac
