#!/usr/bin/env bash
# ③ ONNX → RKNN FP16，并在同一会话里做 simulator 精度检查
#    （load_rknn 加载的模型不能在 simulator 上推理，必须在 build 的同一进程里跑）
set -e
source "$(dirname "$0")/env.sh"

"$PY" "$WORK/scripts/build_and_sim.py" \
  --onnx "$ONNX_OP17" \
  --out  "$RKNN_FP16" \
  --ref  "$WORK/ref" \
  --report "$WORK/results/sim_op17_fp16.json" \
  2>&1 | tee "$WORK/logs/build_rknn.log" | grep -vE "^(D RKNN|I RKNN|W RKNN)" | tail -35

echo
echo "=== 算子分配检查（应当没有计算算子在 CPU）==="
grep -oE 'D RKNN: \[[0-9:.]+\] +[0-9]+ [A-Za-z]+' "$WORK/logs/build_rknn.log" \
  | sed -E 's/.*\] +[0-9]+ //' | sort | uniq -c | sort -rn | head -12
echo
echo "=== 模型 ==="
ls -lh "$RKNN_FP16"; md5sum "$RKNN_FP16"
