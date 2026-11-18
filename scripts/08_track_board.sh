#!/usr/bin/env bash
# ⑧ 板端 RKNN 跟踪（4 段视频）+ 拉回结果
#    SSHPASS='****' bash scripts/08_track_board.sh
set -e
source "$(dirname "$0")/ssh_lib.sh"

RK="build/$(basename "$RKNN_FP16")"
run() {  # name "bbox" frames
  echo "=== RKNN $1 bbox=$2 frames=$3 ==="
  bssh "cd $BOARD_DIR && $BOARD_PY scripts/track_headless.py --backend rknn --model $RK \
    --video videos/$1.mp4 --bbox $2 --out results/rknn/$1 --frames $3 --core 0 --no-video \
    2>&1 | grep -vE 'RKNN|^W |^I ' | tail -12"
}

run man        "80 30 145 190"     300
run cars       "520 260 45 32"     300
run street     "255 143 37 95"     300
run basketball "1010 1600 436 740" 89

echo "=== pull ==="
mkdir -p "$WORK/results/rknn"
for v in man cars street basketball; do
  mkdir -p "$WORK/results/rknn/$v"
  for f in track.csv summary.json; do
    sshpass -e scp $SSHOPT "$BOARD:$BOARD_DIR/results/rknn/$v/$f" \
      "$WORK/results/rknn/$v/$f" 2>/dev/null || echo "miss $v/$f"
  done
done
"$PY" "$WORK/scripts/compare_tracks.py" | tee "$WORK/logs/compare_tracks.log"
