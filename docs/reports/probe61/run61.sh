#!/usr/bin/env bash
# Prompt 61's runs, one BLAS thread each. Run from the repository root:
#   bash docs/reports/probe61/run61.sh STAGE
# Tree: results/builder61/B is a worktree of the branch at the commit the
# runs name in their records (the five layers built, nothing under src/
# changed after it). Logs go to results/builder61/logs/, records to
# results/builder61/.
# Stages:
#   flows      prediction (a): every hash of record from rest under B;
#              prediction (b): the vented and the default room at 320x120;
#              prediction (c): the closed box. All launched together.
#   transport  the marches of the two rooms, S1 to S3 at Sc_t 0.7 (transport61.py)
#   tables     the report's tables from the records (tables61.py)
set -u
PY=.venv/Scripts/python
B=results/builder61/B
PROBE=$B/docs/reports/probe61
LOGS=results/builder61/logs
mkdir -p "$LOGS"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
N=16

launch() {
  local name=$1
  shift
  "$PY" "$@" > "$LOGS/$name.log" 2>&1 &
}

stage=${1:-}
case "$stage" in
  flows)
    for c in val001_80x40 val001_80x40_stretched val002_20x20 val002_40x40 \
        val002_80x80 val017_216x72 val019_656x60; do
      launch "hash_$c" "$PROBE/hash61.py" row "$c" --root "$B" --concurrent $N
    done
    for grid in 80x30 160x60; do
      launch "hash_product_$grid" "$PROBE/hash61.py" product "$grid" --root "$B" \
        --concurrent $N
    done
    launch hash_L0_320x120 "$PROBE/hash61.py" record L0_320x120 \
      results/builder56/L0_320x120.json --root "$B" --concurrent $N
    launch hash_t1_rest "$PROBE/hash61.py" record t1_rest \
      results/builder59/rest.json --root "$B" --concurrent $N
    launch flow_vents "$PROBE/flow61.py" room vents_320x120 \
      configs/clean_room_vents.yaml --root "$B" --concurrent $N
    launch flow_default "$PROBE/flow61.py" room default_320x120 \
      configs/clean_room_default.yaml --root "$B" --concurrent $N
    launch flow_box "$PROBE/flow61.py" box closed_box --root "$B" --concurrent $N
    wait
    ;;
  transport)
    for room in vents_320x120 default_320x120; do
      for s in S1 S2 S3; do
        launch "transport_${room}_$s" docs/reports/probe61/transport61.py run "$room" "$s" \
          --root "$B" --concurrent 6
      done
    done
    wait
    ;;
  tables)
    "$PY" docs/reports/probe61/tables61.py
    ;;
  *)
    echo "usage: run61.sh flows|transport|tables" >&2
    exit 2
    ;;
esac
