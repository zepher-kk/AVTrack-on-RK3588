#!/usr/bin/env bash
# ④ 把产物 + 脚本 + 视频传到板端（$BOARD:$BOARD_DIR）
#    SSHPASS='****' bash scripts/04_deploy.sh
set -e
source "$(dirname "$0")/ssh_lib.sh"

echo "=== mkdir on board ==="
bssh "mkdir -p $BOARD_DIR/{build,ref,scripts,videos,results}"

echo "=== push model / ref / scripts ==="
bscp "$RKNN_FP16" "$BOARD:$BOARD_DIR/build/"
bscp "$WORK/ref/pairs.npz" "$WORK/ref/out_ort.npz" "$WORK/ref/ref.json" "$BOARD:$BOARD_DIR/ref/"
bscp "$WORK/scripts/avtrack_common.py" "$WORK/scripts/run_board.py" "$WORK/scripts/bench.py" \
     "$WORK/scripts/track_headless.py" "$WORK/scripts/run_crops.py" \
     "$BOARD:$BOARD_DIR/scripts/"

if [ -d "$VIDEOS" ]; then
  echo "=== push videos ==="
  bscp "$VIDEOS"/*.mp4 "$VIDEOS"/*.txt "$BOARD:$BOARD_DIR/videos/" || \
    echo "（视频推送部分失败，可稍后手动 scp）"
fi

echo "=== verify ==="
bssh "du -sh $BOARD_DIR/*; md5sum $BOARD_DIR/build/*.rknn"
