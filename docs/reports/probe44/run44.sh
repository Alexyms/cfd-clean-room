#!/usr/bin/env bash
# Prompt 44 (ECR-002 step 5): the runs, in stages, in parallel, one BLAS thread each.
# Run from the repository root: bash docs/reports/probe44/run44.sh STAGE [ARGS].
# Logs go to results/builder44/logs/NAME.log; the records to results/builder44/NAME.json
# and NAME.npz. Stages, in the order they were run:
#   a        the base fields (Re 90, ten sweeps), every U and L row at one sweep,
#            U1 and L at ten sweeps on every grid, and the cavity on 40x40 and 80x80
#   z        the Z rows at one sweep on every grid whose base field converged
#   ten F G  a field F on grid G at ten sweeps (the rows one sweep did not converge)
#   corner F G S   measurement 4: F on G at S sweeps with the corner QUICK zeroing removed
#   rtol F G S     measurement 5: F on G at S sweeps at pressure_rtol 1e-4 and 1e-2
set -u
PY=.venv/Scripts/python
PROBE=docs/reports/probe44/conv44.py
LOGS=results/builder44/logs
mkdir -p "$LOGS"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

launch() {
  local name=$1
  shift
  "$PY" "$PROBE" "$@" > "$LOGS/$name.log" 2>&1 &
}

stage=${1:-}
case "$stage" in
  a)
    for grid in 40x15 80x30 200x75; do
      launch "base_${grid}" base "$grid"
      for field in U1 U2 U3 L; do
        launch "${field}_${grid}_s1" run "$grid" "$field" 1
      done
      launch "U1_${grid}_s10" run "$grid" U1 10
      launch "L_${grid}_s10" run "$grid" L 10
    done
    launch cavity_40x40 cavity 40
    launch cavity_80x80 cavity 80
    wait
    ;;
  z)
    for grid in 40x15 80x30 200x75; do
      for field in Z2 Z3 Z4; do
        launch "${field}_${grid}_s1" run "$grid" "$field" 1
      done
    done
    wait
    ;;
  ten)
    launch "${2}_${3}_s10" run "$3" "$2" 10
    wait
    ;;
  corner)
    launch "${2}_${3}_s${4}_cf" run "$3" "$2" "$4" --corner-free
    wait
    ;;
  rtol)
    launch "${2}_${3}_s${4}_r1e-4" run "$3" "$2" "$4" --rtol 1e-4
    launch "${2}_${3}_s${4}_r1e-2" run "$3" "$2" "$4" --rtol 1e-2
    wait
    ;;
  *)
    echo "usage: run44.sh a | z | ten FIELD GRID | corner FIELD GRID SWEEPS | rtol FIELD GRID SWEEPS" >&2
    exit 2
    ;;
esac
