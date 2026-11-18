# 从零复现

> 目标：在一台 x86 机器（WSL2/Ubuntu）上把上游 checkpoint 转成 RK3588 可跑的 `.rknn`，
> 并在板端完成数值 / 性能 / 跟踪三组验收。全程约 1 小时（不含下载与传输）。

## 0. 前置

| 需要 | 说明 |
|---|---|
| 上游代码 | `git clone https://github.com/wuyou3474/AVTrack ../AVTrack` |
| checkpoint | `../AVTrack/output/checkpoints/train/avtrack/deit_tiny_patch16_224/AVTrack_ep0300.pth.tar`（249 MiB，上游 Release 提供） |
| PC 环境 | Python 3.10：`torch==2.4.*`、`onnx`、`onnxruntime`、`timm`、`opencv-python`、`easydict`、`tensorboardX`、`rknn-toolkit2==2.5.*` |
| 板端环境 | Python 3.8：`numpy`、`opencv-python`、`rknn-toolkit-lite2==2.5.*`；`librknnrt.so` 2.5.x |
| 演示视频 | 4 段，分别 `<name>.mp4` + `<name>.txt`（内容：`<name> x y w h`，首帧框） |
| 密码 | 只通过 `SSHPASS` 环境变量传递 |

```bash
# PC 侧依赖（示例）
pip install torch onnx onnxruntime timm easydict tensorboardX opencv-python
pip install rknn-toolkit2==2.5.0          # 建议按官方 wheel 安装
# 板端依赖
pip install rknn-toolkit-lite2==2.5.0 numpy opencv-python
```

## 1. 导出 ONNX（gate9，冻结跳块表）

```bash
export PY=/path/to/python SRC=../AVTrack VIDEOS=../videos
bash scripts/01_export_onnx.sh
```

**预期**（用来判断这次导出是否正常）：

| 检查项 | 期望 |
|---|---|
| `strict=True` 加载 | `<All keys matched successfully>` |
| block 集合 | `{0,1,2,5,6,7,8,9,10}` |
| op17 节点 / LN | ≈ **429** / 19（写成 `d ** 0.5` 会是 474，数值等价） |
| op11 节点 | **582** |
| 两次 dummy 导出差 | **0.0** |
| ORT vs PyTorch | ≤ 1e-5（实测 ~1e-6） |

## 2. 建金标准

```bash
bash scripts/02_make_reference.sh
```

产出 `ref/pairs.npz`（20 组 uint8 RGB）、`ref/out_torch.npz`、`ref/out_ort.npz`、`ref/ref.json`。
**验收**：torch vs ort `1.2e-6 / 1.0e-6 / 4.7e-6`。

## 3. 构建 RKNN FP16

```bash
bash scripts/03_build_rknn.sh
```

**预期**：`15.89 MB`、构建 ≈6 s、日志中 187 个计算层全在 NPU（`exNorm` / `exSDPAttention` 等融合），
simulator 误差 `1.00e-3 / 5.10e-4 / 5.36e-3`（判据 `2e-3 / 5e-3 / 6e-3`）。

> `.rknn` 的 md5 **不可复现**（RKNN 会写入构建信息）：Release 里那份是 `ba715f60…`，你重跑会得到另一个 md5，
> 但体积、算子分配、板端数值都一致。ONNX 的 md5 是可复现的（`6b0c42c2…` / `dd66e27b…`）。

> `load_rknn` 加载的模型**不能**在 simulator 上推理；必须 `build()` 后同会话 `init_runtime()`。
> `scripts/build_and_sim.py` 已经这样做了。

## 4. 上板

```bash
export BOARD=user@<board-ip> SSHPASS='****'
bash scripts/04_deploy.sh
```

推送内容：`.rknn` → `build/`；`ref/*` → `ref/`；4 个 py 脚本 → `scripts/`；视频 → `videos/`。

## 5. 板端数值 + 稳定性

```bash
bash scripts/05_board_numeric.sh
```

**预期**：`score 9.98e-3 / size 6.29e-3 / offset 5.35e-2`；解码 `dx 0.63 / dy 0.25 / w 1.90 / h 0.95 px`；
argmax 20/20 相同；100 帧内存变化 ≈ 0。

## 6. 性能

```bash
bash scripts/06_board_bench.sh
```

**预期**：单核 39.9 ms（25.1 FPS）、三核 35.4 ms（28.3 FPS）、2 路 47.3 FPS、3 路 63.9 FPS；
8000 帧长稳温度 38.8→41.6 °C。

## 7. 跟踪 + 公平对照

```bash
bash scripts/07_track_pc.sh          # PC 侧 ORT 参考（4 段视频）
bash scripts/08_track_board.sh       # 板端 RKNN（4 段视频）+ 自动对照
bash scripts/09_fair_l1.sh man street   # 同像素公平对照
bash scripts/10_check_decode.sh      # 解码差异实测（解释跨设备差异来源）
```

**预期**：同像素 IoU 0.997+；解码同帧最大差 ±2、61–63% 通道不同。

## 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `ModuleNotFoundError: easydict` | `pip install easydict` |
| `torch.load` 报 `AverageMeter` / `tensorboardX` | `pip install tensorboardX` |
| 建模型卡在 `Downloading deit_tiny...` | `export_avtrack.py` 已 patch `pretrained=False`；若自写脚本请照做 |
| `NotImplementedError` in `build_avtrack` | 忘了 `update_config_from_file`（默认 cfg 是 vit_base） |
| `'list' object has no attribute 'size'` | `rknnlite.inference(inputs=[a, b])` 要传 ndarray，不要 `list(a)` |
| simulator 报 “not support inference” | 必须在 `build()` 同会话 `init_runtime()` |
| 板端与 PC 轨迹差几十像素 | 解码差异（±2 灰度级），用 `09_fair_l1.sh` 同像素比较 |
| 目录里冒出 `check*_*.onnx` | RKNN 优化 pass 的中间产物，已 gitignore，删掉即可 |
| 4 路并行失败 | RK3588 只有 3 个 NPU core |
