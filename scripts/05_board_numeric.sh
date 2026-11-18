#!/usr/bin/env bash
# ⑤ 板端数值验收：20 组同像素对照 + 100 帧稳定性
#    SSHPASS='****' bash scripts/05_board_numeric.sh
set -e
source "$(dirname "$0")/ssh_lib.sh"

bssh "cd $BOARD_DIR && $BOARD_PY scripts/run_board.py \
  --rknn build/$(basename "$RKNN_FP16") --ref ref --iters 100 --core 0 \
  --out results/board_fp16_core0.json" | tee "$WORK/logs/board_numeric.log" | tail -45

sshpass -e scp $SSHOPT "$BOARD:$BOARD_DIR/results/board_fp16_core0.json" "$WORK/results/" 2>/dev/null || true
echo
echo "期望：score<2e-2 size<2e-2 offset<1e-1；解码 dx/dy<2px w/h<3px；argmax 20/20"
