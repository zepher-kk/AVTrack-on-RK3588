#!/usr/bin/env bash
# ② 建"金标准"：固定输入对 + 冻结语义的 torch / ORT 参考输出
#    视频目录由 $VIDEOS 指定，需要 <name>.mp4 与 <name>.txt（内容："<name> x y w h"）
set -e
source "$(dirname "$0")/env.sh"

if [ ! -d "$VIDEOS" ]; then
  echo "找不到视频目录：$VIDEOS" >&2
  echo "请设置 VIDEOS=/path/to/videos，并放入 man/cars/street/basketball.mp4 + .txt" >&2
  exit 2
fi

# 相对路径按上游仓库解析
case "$CFG" in /*) CFG_ABS="$CFG" ;; *) CFG_ABS="$SRC/$CFG" ;; esac

"$PY" "$WORK/scripts/make_ref.py" \
  --videos "$VIDEOS" \
  --out "$WORK/ref" \
  --onnx "$ONNX_OP17" \
  --source "$SRC" --config "$CFG_ABS" --checkpoint "$CKPT" \
  2>&1 | tee "$WORK/logs/make_ref.log" | grep -vE "^(WARNING|/|  File )" | tail -30

echo
echo "=== ref/ ==="
ls -lh "$WORK/ref"
