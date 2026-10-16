#!/usr/bin/env bash
# Prompt 56 (ECR-002 step 8, VAL-018 and the comparative check): the runs, in
# stages, in parallel, one BLAS thread each. Run from the repository root:
# bash docs/reports/probe56/run56.sh STAGE [ARGS]. Logs go to
# results/builder56/logs/NAME.log; the records to results/builder56/NAME.json
# and NAME.npz. Every launcher takes the number of processes the stage puts on
# the machine (N, recorded in each record as concurrent_processes; add any
# other builder's processes by hand). Stages, in the order they were run:
#   time N            the two timing probes of L0, before the set
#   flows N           measurement 1: the three layouts on both grids, six runs
#   harness N         VAL-018's harness row, val018_320x120, one repeat
#   marches N NAME SC [SC ...]   measurement 2: S1, S2 and S3 on the converged
#                     record NAME at each Schmidt number listed, in parallel
#   march N NAME SOURCE SC [ARGS]   one march
#   bounded NAME [NAME ...]   bounded44's characterisation of bounded rows
#   figures           every figure, beside the report and under results
#   tables            every table, from the records
set -u
PY=.venv/Scripts/python
PROBE=docs/reports/probe56/coupled56.py
TRANSPORT=docs/reports/probe56/transport56.py
LOGS=results/builder56/logs
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
    for grid in 160x60 320x120; do
      launch "L0_${grid}_time" "$PROBE" time "$grid" --concurrent "$2"
    done
    wait
    ;;
  flows)
    for layout in L0 L1 L2; do
      for grid in 160x60 320x120; do
        launch "${layout}_${grid}" "$PROBE" run "$layout" "$grid" --concurrent "$2"
      done
    done
    wait
    ;;
  harness)
    launch "harness_val018_320x120" scripts/benchmark.py --cases val018_320x120 --repeats 1 --concurrent "$2"
    wait
    ;;
  marches)
    n=$2
    name=$3
    shift 3
    for sc in "$@"; do
      for source in S1 S2 S3; do
        launch "transport_${name}_${source}_sc${sc}" "$TRANSPORT" run "$name" "$source" --schmidt "$sc" --concurrent "$n"
      done
    done
    wait
    ;;
  march)
    launch "transport_${3}_${4}_sc${5}" "$TRANSPORT" run "$3" "$4" --schmidt "$5" --concurrent "$2" "${@:6}"
    wait
    ;;
  bounded)
    shift
    for name in "$@"; do
      "$PY" docs/reports/probe56/bounded56.py history "$name" > "$LOGS/bounded_$name.log"
      "$PY" docs/reports/probe56/bounded56.py where "$name" > "$LOGS/bounded_where_$name.log"
    done
    ;;
  figures)
    "$PY" docs/reports/probe56/figures56.py all
    ;;
  tables)
    "$PY" docs/reports/probe56/tables56.py all
    ;;
  *)
    echo "usage: run56.sh time N | flows N | harness N | marches N NAME SC... | march N NAME SOURCE SC | bounded NAME... | figures | tables" >&2
    exit 2
    ;;
esac
