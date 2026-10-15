#!/usr/bin/env bash
# Prompt 54 (multigrid becomes the default pressure preconditioner): the runs,
# in stages, one BLAS thread each. Run from the repository root:
#   bash docs/reports/probe54/run54.sh STAGE
# Trees: results/builder54/B is a worktree of commit B (759b3ad, the commit
# that makes the default; commit C adds a report only) for the whole runs;
# results/builder54/plantsA and plantsB are worktrees of commits A (11fb6ad)
# and B that only plant54.py edits. Logs go to results/builder54/logs/,
# records to results/builder54/. The captured systems of item 6 are prompt 53's
# (results/builder53/systems/).
# Stages, in the order they were run:
#   weight   item 6: the old and the fixed weight on the twelve captured systems,
#            one process, alone (weight54.py)
#   plants   commit A's and commit B's planted defects (plant54.py), the two
#            groups together
#   whole    item 7: under the default (the harness label staggered-mgcg) the
#            VAL-001 presets, the three VAL-002 grids, the Annex 20 and step
#            rows, the product room on 80x30 and 160x60 and VAL-016; and under
#            the diagonal (staggered-cg) the five laminar presets, which show
#            the jacobi path is the one every Jacobi record came from. All
#            launched together, 15 processes
#   append   benchmarks/results.jsonl gains the staggered-mgcg rows (append54.py)
#   tables   every table of the report, from the records (tables54.py)
set -u
PY=.venv/Scripts/python
PROBE=docs/reports/probe54
B=results/builder54/B
LOGS=results/builder54/logs
mkdir -p "$LOGS"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

launch() {
  local name=$1
  shift
  "$PY" "$@" > "$LOGS/$name.log" 2>&1 &
}

stage=${1:-}
case "$stage" in
  weight)
    "$PY" "$PROBE/weight54.py" > "$LOGS/weight54.log" 2>&1
    ;;
  plants)
    launch plants_A "$PROBE/plant54.py" --root results/builder54/plantsA --group A
    launch plants_B "$PROBE/plant54.py" --root results/builder54/plantsB --group B
    wait
    ;;
  whole)
    n=15
    for c in val001_80x40 val001_80x40_stretched val002_20x20 val002_40x40 \
        val002_80x80 val017_216x72 val019_656x60; do
      launch "row_${c}_mg" "$PROBE/row54.py" "$c" mg --root "$B" --concurrent $n
    done
    for c in val001_80x40 val001_80x40_stretched val002_20x20 val002_40x40 \
        val002_80x80; do
      launch "row_${c}_jacobi" "$PROBE/row54.py" "$c" jacobi --root "$B" \
        --method staggered-cg --concurrent $n
    done
    for grid in 80x30 160x60; do
      launch "product_${grid}_mg" "$PROBE/inject54.py" --root "$B" product "$grid" mg
    done
    launch val016_mg "$PROBE/inject54.py" --root "$B" val016 mg
    wait
    ;;
  append)
    "$PY" "$PROBE/append54.py"
    ;;
  tables)
    "$PY" "$PROBE/tables54.py"
    ;;
  *)
    echo "usage: run54.sh weight|plants|whole|append|tables" >&2
    exit 2
    ;;
esac
