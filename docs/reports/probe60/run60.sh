#!/usr/bin/env bash
# Prompt 60's runs, one BLAS thread each. Run from the repository root:
#   bash docs/reports/probe60/run60.sh STAGE
# Trees: results/builder60/B is a worktree of the branch's commit with the
# probes (the code of 7bc93df, nothing under src/ changed after it);
# results/builder60/base is a worktree of main at 1c58700, the commit the
# branch starts from, for the two product rows whose case file moved after
# the hashes of record were taken (prompt 56). Logs go to
# results/builder60/logs/, records to results/builder60/.
# Stages:
#   runs     prediction (a): every hash of record from rest under B, the two
#            product rows under base too; item 6: warm60.py self and open
#            under B. All launched together, 15 processes
#   tables   the report's tables from the records (tables60.py)
set -u
PY=.venv/Scripts/python
B=results/builder60/B
BASE=results/builder60/base
PROBE=$B/docs/reports/probe60
LOGS=results/builder60/logs
mkdir -p "$LOGS"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
N=15

launch() {
  local name=$1
  shift
  "$PY" "$@" > "$LOGS/$name.log" 2>&1 &
}

stage=${1:-}
case "$stage" in
  runs)
    for c in val001_80x40 val001_80x40_stretched val002_20x20 val002_40x40 \
        val002_80x80 val017_216x72 val019_656x60; do
      launch "hash_$c" "$PROBE/hash60.py" row "$c" --root "$B" --concurrent $N
    done
    for grid in 80x30 160x60; do
      launch "hash_product_$grid" "$PROBE/hash60.py" product "$grid" --root "$B" \
        --concurrent $N
      launch "hash_product_${grid}_base" "$PROBE/hash60.py" product "$grid" \
        --root "$BASE" --suffix _base --concurrent $N
    done
    launch hash_L0_320x120 "$PROBE/hash60.py" record L0_320x120 \
      results/builder56/L0_320x120.json --root "$B" --concurrent $N
    launch hash_t1_rest "$PROBE/hash60.py" record t1_rest \
      results/builder59/rest.json --root "$B" --concurrent $N
    launch warm60_self "$PROBE/warm60.py" self --root "$B" --concurrent $N
    launch warm60_open "$PROBE/warm60.py" open --root "$B" --concurrent $N
    wait
    ;;
  tables)
    "$PY" docs/reports/probe60/tables60.py
    ;;
  *)
    echo "usage: run60.sh runs|tables" >&2
    exit 2
    ;;
esac
