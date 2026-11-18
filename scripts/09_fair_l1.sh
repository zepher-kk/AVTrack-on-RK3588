#!/usr/bin/env bash
# ⑨ 公平 L1：把每帧裁成 uint8 crop 包，PC 跑 ORT、板端跑 RKNN，输入逐比特一致
#    SSHPASS='****' bash scripts/09_fair_l1.sh [man street cars basketball]
set -e
source "$(dirname "$0")/ssh_lib.sh"

VIDS="${*:-man street}"
RK="build/$(basename "$RKNN_FP16")"
mkdir -p "$WORK/results/crops"

for v in $VIDS; do
  echo "===== $v ====="
  "$PY" "$WORK/scripts/dump_crops.py" --video "$VIDEOS/$v.mp4" \
    --track "$WORK/results/pc_ort/$v/track.csv" \
    --out "$WORK/results/crops/$v.npz"

  "$PY" "$WORK/scripts/run_crops.py" --backend ort --model "$ONNX_OP17" \
    --crops "$WORK/results/crops/$v.npz" --out "$WORK/results/crops/${v}_ort.csv"

  bscp "$WORK/scripts/run_crops.py" "$BOARD:$BOARD_DIR/scripts/" >/dev/null
  sshpass -e scp $SSHOPT "$WORK/results/crops/$v.npz" "$BOARD:$BOARD_DIR/results/crops_$v.npz"
  bssh "cd $BOARD_DIR && $BOARD_PY scripts/run_crops.py --backend rknn --model $RK \
    --crops results/crops_$v.npz --out results/crops_${v}_rknn.csv --core 0 2>&1 | tail -2"
  sshpass -e scp $SSHOPT "$BOARD:$BOARD_DIR/results/crops_${v}_rknn.csv" \
    "$WORK/results/crops/${v}_rknn.csv"
  bssh "rm -f $BOARD_DIR/results/crops_$v.npz"
done

echo "===== 汇总 ====="
"$PY" "$WORK/scripts/summarize_fair_l1.py" | tee "$WORK/logs/fair_l1.log"
