#!/usr/bin/env bash
# ⑦ PC 侧参考跟踪（ORT，同一张 gate9 图），4 段视频
#    输出 results/pc_ort/<name>/{track.csv,summary.json}
set -e
source "$(dirname "$0")/env.sh"

OUT="$WORK/results/pc_ort"
run() {  # name "bbox" frames
  echo "=== ORT $1 bbox=$2 frames=$3 ==="
  "$PY" "$WORK/scripts/track_headless.py" --backend ort --model "$ONNX_OP17" \
    --video "$VIDEOS/$1.mp4" --bbox $2 --out "$OUT/$1" --frames "$3" \
    2>&1 | grep -vE "^(WARNING|/|  File )" | tail -10
}

run man        "80 30 145 190"     300
run cars       "520 260 45 32"     300
run street     "255 143 37 95"     300
run basketball "1010 1600 436 740" 89
echo "=== done ==="
find "$OUT" -name track.csv | xargs -r wc -l
