#!/usr/bin/env bash
# Prompt 55 (ECR-002 step 7, VAL-020): the impinging jet runs, in parallel,
# one BLAS thread each. Run from the repository root (or a worktree):
# bash docs/reports/probe55/run55.sh STAGE. Logs go to results/builder55/logs/
# NAME.log; the records to results/builder55/NAME.json, NAME.npz and
# NAME_pmean.json. Stages:
#   time      the 100-iteration timing probe on each grid
#   flows     item 3: standard under upwind (the default, key absent) on both
#             grids, standard under umist and RNG under upwind on the finer;
#             item 2's side-distance check, the coarse grid 100 w wide
#   row       the harness row on the coarse grid (scripts/benchmark.py)
#   tables    item 4: every table, from the records (tables55.py)
#   figures   items 3 and 4: every figure (figures55.py)
set -u
PY=${PY:-.venv/Scripts/python}
PROBE=docs/reports/probe55/impinging55.py
LOGS=results/builder55/logs
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
    for grid in 272x90 408x135; do
      launch "time_${grid}" "$PROBE" time "$grid"
    done
    wait
    ;;
  flows)
    for grid in 272x90 408x135; do
      launch "standard_${grid}_upwind" "$PROBE" run "$grid"
    done
    launch standard_408x135_umist "$PROBE" run 408x135 --scheme umist
    launch rng_408x135_upwind "$PROBE" run 408x135 --variant rng
    launch standard_272x90_upwind_w100 "$PROBE" run 272x90 --width-w 100 --tag w100
    wait
    ;;
  row)
    "$PY" scripts/benchmark.py --cases val020_272x90 --repeats 1 --concurrent "${2:-1}" \
      ${3:+--results "$3"} > "$LOGS/row_val020_272x90.log" 2>&1
    ;;
  tables)
    "$PY" docs/reports/probe55/tables55.py all
    ;;
  figures)
    "$PY" docs/reports/probe55/figures55.py all
    ;;
  *)
    echo "usage: run55.sh time | flows | row [CONCURRENT [RESULTS]] | tables | figures" >&2
    exit 2
    ;;
esac
