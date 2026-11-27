#!/usr/bin/env bash
# Prompt 97's Python under WSL2: the Linux environment on the core built for this
# step (results/builder97/build.sh), one BLAS thread, NumPy's AVX-512 loops off
# for every run, the reference's included. "run97.sh queue" starts the detached
# runner, its output appended to results/builder97/queue_stdout.log.
cd /mnt/c/Users/tomat/Projects/cfd_clean_room
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NPY_DISABLE_CPU_FEATURES=X86_V4
export PYTHONPATH=${CORE_BUILD:-$HOME/build/cfd-core-97}:/mnt/c/Users/tomat/Projects/cfd_clean_room
if [ "$1" = queue ]; then
  ~/.venvs/cfd_clean_room/bin/python -u docs/reports/probe97/probe97.py queue >> results/builder97/queue_stdout.log 2>&1
  echo "queue exit $? $(date -Iseconds)" >> results/builder97/queue_stdout.log
else
  exec ~/.venvs/cfd_clean_room/bin/python -u "$@"
fi
