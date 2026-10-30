#!/usr/bin/env bash
# Prompt 70's probe B: the filter breach on 160x60 at the arms of configs70.py,
# through scripts/run_scenario.py, each from rest, concurrently; the Schmidt
# arm afterwards, reusing the control's flow state (--flows-from). One BLAS
# thread each. Run files, summaries and logs go to results/builder70/.
set -u
PY=.venv/Scripts/python
R=results/builder70
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p $R/logs
"$PY" docs/reports/probe70/configs70.py
for arm in i05 i10 i10_l30 i02_l05; do
  "$PY" scripts/run_scenario.py $R/configs/scenario_filter_breach_$arm.yaml \
    --out $R/breach_$arm.h5 --concurrent 5 > $R/breach_$arm.txt 2> $R/logs/breach_$arm.log &
done
wait
"$PY" scripts/run_scenario.py $R/configs/scenario_filter_breach_sc02.yaml \
  --out $R/breach_sc02.h5 --flows-from $R/breach_i05.h5 --concurrent 2 \
  > $R/breach_sc02.txt 2> $R/logs/breach_sc02.log
echo all done
