#!/usr/bin/env bash
# 公共配置：全部参数都可用环境变量覆盖。
# 用法：  source "$(dirname "$0")/env.sh"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT
export WORK="$REPO_ROOT"

# ── 上游 AVTrack 仓库（自行 clone：https://github.com/wuyou3474/AVTrack）
export SRC="${SRC:-$REPO_ROOT/../AVTrack}"
export CFG="${CFG:-experiments/avtrack/deit_tiny_patch16_224.yaml}"
export CKPT="${CKPT:-$SRC/output/checkpoints/train/avtrack/deit_tiny_patch16_224/AVTrack_ep0300.pth.tar}"

# ── PC 侧 python：需要 torch / onnx / onnxruntime / rknn-toolkit2 2.5.x
export PY="${PY:-python3}"

# ── 板端（RK3588 + librknnrt 2.5.x + rknn-toolkit-lite2）
export BOARD="${BOARD:-user@board-ip}"
export BOARD_PY="${BOARD_PY:-/home/user/miniconda3/envs/rknn/bin/python}"
export BOARD_DIR="${BOARD_DIR:-/home/user/rknn_avtrack}"
# 密码只通过环境变量传入，仓库里任何文件都不含密码：
#   SSHPASS='****' bash scripts/05_board_numeric.sh
export SSHPASS="${SSHPASS:-}"

# ── 产物路径
export ONNX_OP17="$WORK/onnx/avtrack_gate9_op17.onnx"
export ONNX_OP11="$WORK/onnx/avtrack_gate9_op11.onnx"
export RKNN_FP16="$WORK/build/avtrack_gate9_op17_fp16.rknn"

# ── 演示视频目录（自行准备 man/cars/street/basketball 或改这里的名字）
export VIDEOS="${VIDEOS:-/path/to/videos}"

mkdir -p "$WORK"/{onnx,build,ref,results,logs}
