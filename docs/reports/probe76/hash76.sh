#!/usr/bin/env bash
# Prompt 76's hash check: the six rows the Phase 4 gate reproduced (its table H2),
# solved from rest at this commit by docs/reports/probe62/hash62.py, the records
# results/builder62/hash_CASE_76.json. One BLAS thread each. Beside the scenario
# runs and the suite, so the wall times are upper bounds.
set -u
PY=.venv/Scripts/python
H=docs/reports/probe62/hash62.py
LOGS=results/builder76/logs
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
N=10
for c in val001_80x40 val001_80x40_stretched val002_20x20 val002_40x40 val002_80x80; do
  "$PY" "$H" row "$c" --root . --concurrent $N --suffix _76 > "$LOGS/hash_$c.log" 2>&1 &
done
"$PY" "$H" product 80x30 --root . --concurrent $N --suffix _76 > "$LOGS/hash_product_80x30.log" 2>&1 &
wait
echo all done
