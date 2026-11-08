#!/usr/bin/env bash
# Prompt 76b's run: the seal leak on the product grid (320x120), alone, from
# rest, through scripts/run_scenario.py, the copy configs76.py writes (main's
# file plus a checkpoint every 1000 steps). One BLAS thread. The run file and
# its checkpoint go to results/builder76/, the summary beside them, the
# runner's log under results/builder76/logs/. The other four scenarios are not
# rerun: step 5's run files are re-scored.
set -u
PY=.venv/Scripts/python
R=results/builder76
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p $R/logs
"$PY" docs/reports/probe76/configs76.py
"$PY" scripts/run_scenario.py $R/configs/scenario_seal_leak.yaml --out $R/seal_leak.h5 \
  --concurrent 1 > $R/seal_leak.txt 2> $R/logs/seal_leak.log
echo done
