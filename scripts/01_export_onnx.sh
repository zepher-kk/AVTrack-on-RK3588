#!/usr/bin/env bash
# ① 导出 gate9 ONNX：op17（主用）+ op11（后备）
#    - 冻结跳块表 {跳 3,4,11}
#    - strict=True 全量加载
#    - 双 dummy 交叉校验 + ORT 断言
set -e
source "$(dirname "$0")/env.sh"

bash -c "cd '$SRC' && '$PY' '$WORK/scripts/export_avtrack.py' \
  --source '$SRC' --config '$CFG' --checkpoint '$CKPT' \
  --opset 17 --output '$ONNX_OP17'" 2>&1 | tee "$WORK/logs/export_op17.log" | tail -20

bash -c "cd '$SRC' && '$PY' '$WORK/scripts/export_avtrack.py' \
  --source '$SRC' --config '$CFG' --checkpoint '$CKPT' \
  --opset 11 --output '$ONNX_OP11'" 2>&1 | tee "$WORK/logs/export_op11.log" | tail -20

echo
echo "=== artifacts ==="
ls -lh "$WORK/onnx"
md5sum "$WORK/onnx"/*.onnx
echo "期望：blocks={0,1,2,5,6,7,8,9,10}，op17≈429 节点(LN 19)，op11=582 节点"
