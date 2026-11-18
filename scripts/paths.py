"""集中管理所有路径：默认按仓库位置推导，可用环境变量覆盖。

    WORK     仓库根（默认 = 本文件的上一级）
    SRC      上游 AVTrack 仓库（默认 = 与本仓库同级的 AVTrack）
    VIDEOS   演示视频目录（默认 = $WORK/videos）
    CFG/CKPT 上游配置文件与 checkpoint（默认按 SRC 推导）
    PRIOR    可选：历史 TensorRT 移植目录，仅用于 L2 对照，未设置则跳过
    ONNX_OP17 / RKNN_FP16  产物路径（默认在 $WORK 下）
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

WORK = os.environ.get('WORK') or ROOT
SRC = os.environ.get('SRC') or os.path.join(os.path.dirname(ROOT), 'AVTrack')
VIDEOS = os.environ.get('VIDEOS') or os.path.join(WORK, 'videos')
PRIOR = os.environ.get('PRIOR') or ''

CFG = os.environ.get('CFG') or os.path.join(
    SRC, 'experiments', 'avtrack', 'deit_tiny_patch16_224.yaml')
CKPT = os.environ.get('CKPT') or os.path.join(
    SRC, 'output', 'checkpoints', 'train', 'avtrack',
    'deit_tiny_patch16_224', 'AVTrack_ep0300.pth.tar')

ONNX_OP17 = os.environ.get('ONNX_OP17') or os.path.join(
    WORK, 'onnx', 'avtrack_gate9_op17.onnx')
ONNX_OP11 = os.environ.get('ONNX_OP11') or os.path.join(
    WORK, 'onnx', 'avtrack_gate9_op11.onnx')
RKNN_FP16 = os.environ.get('RKNN_FP16') or os.path.join(
    WORK, 'build', 'avtrack_gate9_op17_fp16.rknn')

REF_DIR = os.path.join(WORK, 'ref')
RESULTS = os.path.join(WORK, 'results')
