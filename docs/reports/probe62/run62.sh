#!/usr/bin/env bash
# Prompt 62's hash check: every hash of record (docs/reports/adr014_step0b.md
# section 4) from rest under the worktree B at bd5014c. One BLAS thread each.
# Run from the repository root with B a worktree of the commit to check:
#   git worktree add results/builder62/B bd5014c
#   bash docs/reports/probe62/run62.sh
set -u
PY=.venv/Scripts/python
B=results/builder62/B
H=docs/reports/probe62/hash62.py
LOGS=results/builder62/logs
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
N=11
for c in val001_80x40 val001_80x40_stretched val002_20x20 val002_40x40 \
    val002_80x80 val017_216x72 val019_656x60; do
  "$PY" "$H" row "$c" --root "$B" --concurrent $N > "$LOGS/hash_$c.log" 2>&1 &
done
for grid in 80x30 160x60; do
  "$PY" "$H" product "$grid" --root "$B" --concurrent $N > "$LOGS/hash_product_$grid.log" 2>&1 &
done
"$PY" "$H" record L0_320x120 results/builder56/L0_320x120.json --root "$B" --concurrent $N > "$LOGS/hash_L0_320x120.log" 2>&1 &
"$PY" "$H" record t1_rest results/builder59/rest.json --root "$B" --concurrent $N > "$LOGS/hash_t1_rest.log" 2>&1 &
wait
echo all done
