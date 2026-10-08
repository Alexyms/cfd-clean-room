#!/usr/bin/env bash
# Prompt 41: every run of the outlet probe, in parallel, one BLAS thread each.
# Run from the repository root. Logs go to results/builder41/logs/NAME.log;
# the records to results/builder41/NAME.json and NAME.npz.
set -u
PY=.venv/Scripts/python
PROBE=docs/reports/probe41/outlet41.py
LOGS=results/builder41/logs
mkdir -p "$LOGS"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

launch() {
  local name=$1
  shift
  "$PY" "$PROBE" "$@" > "$LOGS/$name.log" 2>&1 &
}

# The control: the ten-sweep predictor against frozen34.py's, bitwise.
"$PY" "$PROBE" control > "$LOGS/control.log" 2>&1

# Measurement 1: the drift case, 80x30 at a thousand times air's viscosity.
for arm in A B C D F; do
  for rtol in 1e-8 1e-4 1e-2; do
    launch "drift_${arm}_${rtol}" drift "$arm" "$rtol"
  done
done
launch drift_E_1e-8 drift E 1e-8
launch drift_E0_1e-8 drift E0 1e-8
# Control A0: arm A with the hood's tangential velocity at zero gradient (33b's).
launch drift_A_1e-8_gradient drift A 1e-8 --hood-gradient
# The pressure's movement past the stop: 3,000 outer iterations, no stop.
for arm in A B C D E E0 F; do
  launch "drift_${arm}_1e-8_long" drift "$arm" 1e-8 --long --n-outer 3000
done

# Measurement 2: the ladder, 40x15 at Re 895 and Re 8,950.
for arm in A B C D E E0 F; do
  for rung in 895 8950; do
    launch "ladder_${arm}_${rung}" ladder "$arm" "$rung"
  done
done

# Measurement 3: VAL-001 80x40 under the committed path and arms B, C and F,
# and the committed path, B and F on 40x20 and 160x80.
for arm in committed B C F; do
  launch "val001_${arm}" val001 "$arm"
done
for arm in committed B F; do
  launch "val001_${arm}_40x20" val001 "$arm" --grid 40 20
  launch "val001_${arm}_160x80" val001 "$arm" --grid 160 80
done

wait

# Measurement 4 and the other pairs.
"$PY" "$PROBE" compare drift_D_1e-8 drift_B_1e-8 > "$LOGS/compare_D_B.log" 2>&1
"$PY" "$PROBE" compare drift_C_1e-8 drift_B_1e-8 > "$LOGS/compare_C_B.log" 2>&1
"$PY" "$PROBE" compare drift_E_1e-8 drift_B_1e-8 > "$LOGS/compare_E_B.log" 2>&1
"$PY" "$PROBE" compare drift_E_1e-8 drift_D_1e-8 > "$LOGS/compare_E_D.log" 2>&1
"$PY" "$PROBE" compare drift_A_1e-8 drift_B_1e-8 > "$LOGS/compare_A_B.log" 2>&1
"$PY" "$PROBE" compare drift_F_1e-8 drift_B_1e-8 > "$LOGS/compare_F_B.log" 2>&1
"$PY" "$PROBE" compare drift_F_1e-8 drift_A_1e-8 > "$LOGS/compare_F_A.log" 2>&1
"$PY" "$PROBE" compare ladder_B_8950 ladder_D_8950 --grid 40 15 > "$LOGS/compare_ladder_B_D_8950.log" 2>&1
"$PY" "$PROBE" compare ladder_B_895 ladder_D_895 --grid 40 15 > "$LOGS/compare_ladder_B_D_895.log" 2>&1

# Prompt 41b: the two arms of section 5.3, then their comparisons.
for rtol in 1e-8 1e-4 1e-2; do
  launch "drift_D0_${rtol}" drift D0 "$rtol"
done
launch drift_D0_1e-8_long drift D0 1e-8 --long --n-outer 3000
for arm in Aopen D0; do
  for rung in 895 8950; do
    launch "ladder_${arm}_${rung}" ladder "$arm" "$rung"
  done
done
wait
"$PY" "$PROBE" compare drift_D0_1e-8 drift_D_1e-8 > "$LOGS/compare_D0_D.log" 2>&1
"$PY" "$PROBE" compare ladder_D0_895 ladder_D_895 --grid 40 15 > "$LOGS/compare_ladder_D0_D_895.log" 2>&1
"$PY" "$PROBE" compare ladder_D0_8950 ladder_D_8950 --grid 40 15 > "$LOGS/compare_ladder_D0_D_8950.log" 2>&1
echo "all runs finished $(date -Iseconds)"
