#!/usr/bin/env bash
# Prompt 44 (ECR-002 step 5): the runs, in stages, in parallel, one BLAS thread each.
# Run from the repository root: bash docs/reports/probe44/run44.sh STAGE [ARGS].
# Logs go to results/builder44/logs/NAME.log; the records to results/builder44/NAME.json
# and NAME.npz. Stages, in the order they were run:
#   time     the five twenty-iteration timing probes (section 5.1), run before stage a
#   a        the base fields (Re 90, ten sweeps), every U and L row at one sweep,
#            U1 and L at ten sweeps on every grid, and the cavity on 40x40 and 80x80
#   z        the Z rows at one sweep on every grid whose base field converged
#   sweeps F G S   a field F on grid G at S sweeps (ten for the rows one sweep did not
#            converge; fifty for the three 80x30 diagnostics of section 5.3.3)
#   corner F G S   measurement 4: F on G at S sweeps with the corner QUICK zeroing removed
#   rtol F G S     measurement 5: F on G at S sweeps at pressure_rtol 1e-4 and 1e-2
#   cap15k   the two 40x15 one-sweep rows rerun with the cap at 15,000 (section 5.3.1)
#   locate   round 2 (prompt 44b item 4): 80x30 U2 at ten sweeps with the cell of the
#            largest change recorded
# As run on 2026-10-09: time; a; z; sweeps U2/U3 on every grid at 10; corner U1 80x30 10,
# corner U1 200x75 10, rtol U1 80x30 10, rtol U1 200x75 10; cap15k; sweeps Z2/Z3/Z4 on
# 80x30 and 200x75 at 10; sweeps U2/U3/L 80x30 50; corner U2 and rtol U2 on both finer
# grids at 10; corner U3/L/Z4 80x30 10; locate (round 2).
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
  time)
    launch time_200x75_s1 run 200x75 U1 1 --cap 20 --tag time --log-every 5
    launch time_200x75_s10 run 200x75 U1 10 --cap 20 --tag time --log-every 5
    launch time_80x30_s1 run 80x30 U1 1 --cap 50 --tag time --log-every 10
    launch time_40x15_s1 run 40x15 U1 1 --cap 100 --tag time --log-every 25
    launch time_200x75_L run 200x75 L 1 --cap 20 --tag time --log-every 5
    wait
    ;;
  sweeps)
    launch "${2}_${3}_s${4}" run "$3" "$2" "$4"
    wait
    ;;
  cap15k)
    launch U2_40x15_s1_cap15k run 40x15 U2 1 --cap 15000 --tag cap15k
    launch U3_40x15_s1_cap15k run 40x15 U3 1 --cap 15000 --tag cap15k
    wait
    ;;
  locate)
    launch U2_80x30_s10_loc run 80x30 U2 10 --locate
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
    echo "usage: run44.sh time | a | z | sweeps FIELD GRID SWEEPS | corner FIELD GRID SWEEPS | rtol FIELD GRID SWEEPS | cap15k | locate" >&2
    exit 2
    ;;
esac
