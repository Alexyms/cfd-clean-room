#!/usr/bin/env bash
# Prompt 51 (k and eps by conjugate gradients): the runs, in stages, one BLAS
# thread each. Run from the repository root: bash docs/reports/probe51/run51.sh
# STAGE. The base is a worktree of origin/main at the branch point,
# results/builder51/base (git worktree add --detach results/builder51/base
# origin/main). Logs go to results/builder51/logs/, records to results/builder51/.
# Stages, in the order they were run:
#   plants    commit A's planted defects, alone (plant51.py)
#   laminar   item 4: VAL-001, its stretched twin and VAL-002, and the Jacobi
#             battery, on the base and on this checkout (check51.py)
#   annex     item 5: the Annex 20 harness row on the base and on this
#             checkout, launched together and alone on the machine; this
#             checkout's row is appended to benchmarks/results.jsonl
#   products  item 6: prompt 48's standard upwind product rows on 80x30 and
#             160x60, on the base and on this checkout, launched together
#   compare   the comparisons of items 4 to 6
set -u
PY=.venv/Scripts/python
PROBE=docs/reports/probe51
BASE=results/builder51/base
LOGS=results/builder51/logs
mkdir -p "$LOGS"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

launch() {
  local name=$1
  shift
  "$PY" "$@" > "$LOGS/$name.log" 2>&1 &
}

stage=${1:-}
case "$stage" in
  plants)
    "$PY" "$PROBE/plant51.py"
    ;;
  laminar)
    launch jacobi_base "$PROBE/check51.py" --root "$BASE" jacobi --label base
    launch jacobi_A "$PROBE/check51.py" jacobi --label A
    for c in val001_80x40 val001_80x40_stretched val002_80x80; do
      launch "val_${c}_base" "$PROBE/check51.py" --root "$BASE" val "$c" --label base
      launch "val_${c}_A" "$PROBE/check51.py" val "$c" --label A
    done
    wait
    ;;
  annex)
    launch annex_main "$PROBE/annex51.py" run main --root "$BASE" --concurrent 2
    launch annex_cg "$PROBE/annex51.py" run cg --concurrent 2 --append
    wait
    ;;
  products)
    for grid in 80x30 160x60; do
      launch "product_${grid}_main" "$PROBE/product51.py" run "$grid" main --root "$BASE"
      launch "product_${grid}_cg" "$PROBE/product51.py" run "$grid" cg
    done
    wait
    ;;
  compare)
    "$PY" "$PROBE/check51.py" report --base base --built A
    "$PY" "$PROBE/annex51.py" compare main cg
    for grid in 80x30 160x60; do
      "$PY" "$PROBE/product51.py" compare "$grid" main cg
    done
    ;;
  *)
    echo "usage: run51.sh plants | laminar | annex | products | compare" >&2
    exit 2
    ;;
esac
