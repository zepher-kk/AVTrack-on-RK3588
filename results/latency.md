# 延迟与吞吐（RK3588 / RK3588，librknnrt 2.5.0）

- 模型：`build/avtrack_gate9_op17_fp16.rknn`（15.89 MB，md5 `ba715f60acb17083cb19e005c7189ad7`）
- 板子：rk3588，8 核 A76/A55，7.8 GB 内存，NPU 三核
- 测法：`scripts/bench.py`，预热 30 帧 + 计时 300 帧；每帧 = crop(factor 4.0, 256²) + 归一化 + `rk.inference` + 汉宁窗/argmax/状态更新
- 视频：`videos/man.mp4`，初始框 `[80,30,145,190]`
- 单位：ms（mean），FPS 由 mean 帧时间换算

## 1. 单帧三段耗时

| 配置 | pre | **npu** | post | 合计 | FPS |
|---|---|---|---|---|---|
| 1 实例 · core0 | 2.69 | **37.13** | 0.09 | 39.91 | **25.1** |
| 1 实例 · core0_1_2 | 3.02 | **32.27** | 0.10 | 35.38 | **28.3** |

- 预处理占 6.7%，后处理可忽略（0.09 ms），**瓶颈 100% 在 NPU 上**
- 三核并行推理单帧比单核快 15%（32.3 vs 37.1 ms），说明这个模型在 NPU 内部有一定并行度，但仍以单核为主

## 2. 多实例吞吐（每个进程绑一个核，独立 `rknn_context`）

| 实例数 | 各实例 FPS | 各实例单帧 | **总吞吐 FPS** | 相对单核加速 |
|---|---|---|---|---|
| 1（core0） | 25.06 | 39.91 ms | **25.06** | 1.00x |
| 1（core0_1_2） | 28.27 | 35.38 ms | **28.27** | 1.13x |
| 2（core0, core1） | 24.15 / 23.19 | 41.4 / 43.1 ms | **47.34** | 1.89x |
| 3（core0, core1, core2） | 22.59 / 21.44 / 19.84 | 44.3 / 46.6 / 50.4 ms | **63.87** | **2.55x** |

结论与 RT-DETR 的经验**相反**：本模型上「多进程各绑一核」的总吞吐明显优于「单进程用三核」
（63.9 vs 28.3 FPS），单实例延迟只从 39.9 涨到 44–50 ms。**如果目标是多路视频，用 3 进程 × 1 核。**

**注意**：RK3588 只有 3 个 NPU core。第 4 个进程无法绑核（`core=3` 直接异常退出），
所以本机并行上限就是 3 路；4 路只能靠分时复用，实测总吞吐不会再涨（62.4 FPS，与 3 路持平）。

## 3. 长稳（soak）

`results/bench_soak.json`：**连续 8000 帧（约 6 分钟，core0，流式读帧）**

| 指标 | 值 |
|---|---|
| NPU 单帧 | p50 **37.29 ms**、p90 47.89 ms、mean 41.40 ms |
| 整帧 | p50 **40.15 ms**、mean 46.94 ms |
| 吞吐 | 21.3 FPS（比 300 帧短测的 25.1 低 ~15%，属长跑抖动/热管理） |
| 温度 | 38.8 °C → **41.6 °C**（无明显升温） |
| 内存 | used 875 MB → 880 MB，**无泄漏** |

结论：连续 6 分钟不崩、不泄漏、不降频；长跑吞吐取 **21–25 FPS** 作为单路预期。

## 3.1 输入布局 A/B（NCHW vs NHWC）

rknnlite 日志提示"模型内部需要 NHWC，但传入 NCHW 会自动转换"，实测转换开销可忽略：

| 输入布局 | pre | **npu** | 整帧 | FPS |
|---|---|---|---|---|
| NCHW（默认，4 维） | 2.93 | **37.18** | 40.20 | 24.88 |
| NHWC（4 维 `(1,H,W,3)`） | 2.87 | **36.89** | 39.86 | 25.09 |

差 0.8%，在噪声范围内 → **保持 NCHW 即可**，不必为它改代码。
（注意：NHWC 要传 4 维 `(1,H,W,3)`，传 3 维会报 `need 4dims input`。）

## 4. 参考：同模型在桌面 GPU 上的 TRT FP16

`PyTracking_ONNX-TensorRT/results*/avtrack.json`：2.84–3.25 ms/帧（含前后处理，307–352 FPS）。
RK3588 单核约为它的 1/13，三路并行总吞吐约为它的 1/5。**这是 NPU 的正常量级，不是异常。**

## 5. 复现命令

```bash
# 单实例
python scripts/bench.py --rknn build/avtrack_gate9_op17_fp16.rknn \
  --video videos/man.mp4 --bbox 80 30 145 190 --frames 300 --core 0 \
  --out results/bench_1x_core0.json --tag 1x_core0

# 三实例（每个绑一个核）
for c in 0 1 2; do
  python scripts/bench.py --rknn build/avtrack_gate9_op17_fp16.rknn \
    --video videos/man.mp4 --bbox 80 30 145 190 --frames 300 --core $c \
    --out results/bench_3x_core$c.json --tag 3x_core$c > /dev/null 2>&1 &
done; wait
```
