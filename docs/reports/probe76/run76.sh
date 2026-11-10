#!/usr/bin/env bash
# Prompt 76's runs: the four reference scenarios on the product grid (320x120)
# through scripts/run_scenario.py, concurrently, each from rest, the copies of
# configs76.py (main's files plus a checkpoint every 1000 steps). One BLAS
# thread each, as every product measurement since prompt 46. The run files and
# their checkpoint files go to results/builder76/, the summaries beside them,
# the runner's log under results/builder76/logs/.
set -u
PY=.venv/Scripts/python
R=results/builder76
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p $R/logs
"$PY" docs/reports/probe76/configs76.py
for name in clean_out door_leak equipment_dust filter_breach; do
  "$PY" scripts/run_scenario.py $R/configs/scenario_$name.yaml --out $R/$name.h5 \
    --concurrent 3 > $R/$name.txt 2> $R/logs/$name.log &
done
wait
echo all done
