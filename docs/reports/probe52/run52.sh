#!/usr/bin/env bash
# Prompt 52 (ECR-002 step 7, VAL-019): the backward-facing step runs, in
# parallel, one BLAS thread each. Run from the repository root:
# bash docs/reports/probe52/run52.sh STAGE. Logs go to results/builder52/logs/
# NAME.log; the records to results/builder52/NAME.json, NAME.npz and
# NAME_pmean.json. Stages:
#   precursor item 2: the upstream channel alone (backstep52.py precursor),
#             then its layer along x (precursor52.py)
#   time      the 100-iteration timing probe on each grid
#   flows     item 4: standard under upwind (the default, key absent) on the
#             three grids, standard under umist and RNG under upwind on the
#             medium grid
#   row       item 3: the harness row on the medium grid (scripts/benchmark.py)
#   tables    item 5: every table, from the records (tables52.py)
#   figures   items 4 and 5: every figure (figures52.py)
set -u
PY=.venv/Scripts/python
PROBE=docs/reports/probe52/backstep52.py
LOGS=results/builder52/logs
mkdir -p "$LOGS"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

launch() {
  local name=$1
  shift
  "$PY" "$@" > "$LOGS/$name.log" 2>&1 &
}

stage=${1:-}
case "$stage" in
  precursor)
    "$PY" "$PROBE" precursor > "$LOGS/precursor.log" 2>&1
    "$PY" docs/reports/probe52/precursor52.py --every 10
    ;;
  time)
    for grid in 410x40 656x60 984x90; do
      launch "time_${grid}" "$PROBE" time "$grid"
    done
    wait
    ;;
  flows)
    for grid in 410x40 656x60 984x90; do
      launch "standard_${grid}_upwind" "$PROBE" run "$grid"
    done
    launch standard_656x60_umist "$PROBE" run 656x60 --scheme umist
    launch rng_656x60_upwind "$PROBE" run 656x60 --variant rng
    wait
    ;;
  row)
    "$PY" scripts/benchmark.py --cases val019_656x60 --repeats 1 --concurrent "${2:-1}" \
      > "$LOGS/row_val019_656x60.log" 2>&1
    ;;
  tables)
    "$PY" docs/reports/probe52/tables52.py all
    ;;
  figures)
    "$PY" docs/reports/probe52/figures52.py all
    ;;
  *)
    echo "usage: run52.sh precursor | time | flows | row [CONCURRENT] | tables | figures" >&2
    exit 2
    ;;
esac
