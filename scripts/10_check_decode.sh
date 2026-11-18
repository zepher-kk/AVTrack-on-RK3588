#!/usr/bin/env bash
# ⑩ 解码一致性检查：同一帧在板端与 PC 解码后逐像素比较
#    这一步解释"为什么跨设备比轨迹会失真"（实测 max ±2、61-63% 通道不同）
#    SSHPASS='****' bash scripts/10_check_decode.sh [street man]
set -e
source "$(dirname "$0")/ssh_lib.sh"

FRAME="${FRAME:-100}"
bscp "$WORK/scripts/dump_frame.py" "$BOARD:$BOARD_DIR/scripts/" >/dev/null

for v in ${*:-street man}; do
  bssh "cd $BOARD_DIR && $BOARD_PY scripts/dump_frame.py videos/$v.mp4 $FRAME results/${v}_board.png" | tail -1
  "$PY" "$WORK/scripts/dump_frame.py" "$VIDEOS/$v.mp4" "$FRAME" "$WORK/results/${v}_pc.png"
  sshpass -e scp $SSHOPT "$BOARD:$BOARD_DIR/results/${v}_board.png" "$WORK/results/${v}_board.png"
  "$PY" "$WORK/scripts/check_decode.py" "$WORK/results/${v}_pc.png" "$WORK/results/${v}_board.png"
  bssh "rm -f $BOARD_DIR/results/${v}_board.png"
done
