#!/usr/bin/env bash
# Prompt 53 (a multigrid preconditioner for the pressure correction): the runs,
# in stages, one BLAS thread each. Run from the repository root:
#   bash docs/reports/probe53/run53.sh STAGE
# Trees: results/builder53/base is a worktree of commit 0 (8ee82d7, main with
# ADR-013's note), results/builder53/A one of commit A (5c5eff4) for the runs,
# results/builder53/plants another of commit A that only plant53.py edits.
# Logs go to results/builder53/logs/, records to results/builder53/.
# Stages, in the order they were run:
#   capture   item 6's systems from the base: five cases early and late, and the
#             two development systems (capture53.py), launched together
#   dev       nu and the power-iteration count on the development systems
#             (dev53.py), alone; first run on a draft of the cycle, rerun on
#             commit A's code in the stage after (below)
#   plants    commit A's planted defects (plant53.py)
#   laminar   item 5: VAL-001, its stretched twin and VAL-002 under the default
#             on the base and on commit A, launched together (check53.py)
#   measure   item 6: one correction per captured system, both preconditioners,
#             alone (measure53.py)
#   whole     item 7: the whole runs, one after another so each runs alone:
#             the step's medium row and the Annex 20 row under multigrid, the
#             Annex 20 row again under jacobi, the product rows on 80x30 and
#             160x60 under each, VAL-016 under multigrid, and VAL-001, its
#             stretched twin and VAL-002 under multigrid
#   after     alone, at the end: dev53.py rerun on the committed code, item 6's
#             second pass (measure53_pass2) and a cycle's time by level
#             (anatomy53.py)
#   tables    every table of the report, from the records (tables53.py)
set -u
PY=.venv/Scripts/python
PROBE=docs/reports/probe53
BASE=results/builder53/base
A=results/builder53/A
LOGS=results/builder53/logs
mkdir -p "$LOGS"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

launch() {
  local name=$1
  shift
  "$PY" "$@" > "$LOGS/$name.log" 2>&1 &
}

alone() {
  local name=$1
  shift
  "$PY" "$@" > "$LOGS/$name.log" 2>&1
}

stage=${1:-}
case "$stage" in
  capture)
    for c in val001 cavity product160 product320 annex step; do
      launch "capture_$c" "$PROBE/capture53.py" "$c" --root "$BASE"
    done
    for c in product80 step410; do
      launch "capture_$c" "$PROBE/capture53.py" "$c" --root "$BASE" --late 300
    done
    wait
    ;;
  dev)
    alone dev "$PROBE/dev53.py"
    ;;
  plants)
    alone plants "$PROBE/plant53.py" --root results/builder53/plants
    ;;
  laminar)
    for c in val001_80x40 val001_80x40_stretched val002_80x80; do
      launch "val_${c}_base" "$PROBE/check53.py" --root "$BASE" val "$c" --label base
      launch "val_${c}_A" "$PROBE/check53.py" --root "$A" val "$c" --label A
    done
    wait
    "$PY" "$PROBE/check53.py" report --base base --built A
    ;;
  measure)
    alone measure "$PROBE/measure53.py"
    ;;
  whole)
    alone row_step_mg "$PROBE/row53.py" val019_656x60 mg --root "$A"
    alone row_annex_mg "$PROBE/row53.py" val017_216x72 mg --root "$A"
    alone row_annex_jacobi "$PROBE/row53.py" val017_216x72 jacobi --root "$A" \
      --preconditioner jacobi
    for grid in 80x30 160x60; do
      alone "product_${grid}_mg" "$PROBE/inject53.py" --root "$A" product "$grid" mg
      alone "product_${grid}_jacobi" "$PROBE/inject53.py" --root "$A" \
        --preconditioner jacobi product "$grid" jacobi
    done
    alone val016_mg "$PROBE/inject53.py" --root "$A" val016 mg
    for c in val001_80x40 val001_80x40_stretched val002_80x80; do
      alone "val_${c}_mg" "$PROBE/check53.py" --root "$A" val "$c" --label mg \
        --preconditioner multigrid
    done
    ;;
  after)
    alone dev "$PROBE/dev53.py"
    alone measure_pass2 "$PROBE/measure53.py" --name measure53_pass2
    alone anatomy "$PROBE/anatomy53.py"
    ;;
  tables)
    "$PY" "$PROBE/tables53.py"
    ;;
  *)
    echo "usage: run53.sh capture|dev|plants|laminar|measure|whole|after|tables" >&2
    exit 2
    ;;
esac
