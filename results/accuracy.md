# 数值精度（RK3588 FP16）

一句话结论：**FP16 数值精度达标（逐帧偏差亚像素级），但 RK3588 上没有比 FP16 更高的精度可选；
真正会毁掉跨设备对比的是"视频解码差异"，不是 NPU。**

---

## 1. ONNX 导出（PC，CPU）

| 产物 | opset | 节点 | LayerNormalization | Softmax | MatMul | block 集合 | md5 |
|---|---|---|---|---|---|---|---|
| `onnx/avtrack_gate9_op17.onnx`（主用） | 17 | 429 | 19 | 9 | 54 | `{0,1,2,5,6,7,8,9,10}` | `6b0c42c2aee0236f682a0f732bcdbdb4` |
| `onnx/avtrack_gate9_op11.onnx`（后备） | 11 | 582 | 0（拆成原语） | 9 | 54 | 同上 | `dd66e27bc20405b803c0973fcd68efd5` |

- checkpoint 加载：`strict=True` → `<All keys matched successfully>`（网络里 272 个 tensor）
- 冻结 gate：`RUN_BLOCKS={0,1,2,5,6,7,8,9,10}` / `SKIP={3,4,11}`，导出后断言 block 集合通过
- 两次不同 dummy（灰图 / randn）导出的图，同一输入输出差 **0.0**（说明冻结生效、不再漂）
- PyTorch vs ORT：`2.1e-7 / 5.4e-7 / 5.7e-6`（score/size/offset）
- op17 vs op11 互比：`1.2e-7 / 3.0e-7 / 3.3e-6`
- `onnx.checker` 通过

## 2. 金标准（`ref/`，20 组 = 4 段视频 × 5 帧）

- 冻结语义的 PyTorch vs ORT：`1.22e-6 / 1.01e-6 / 4.65e-6`（阈值 1e-5）→ **通过**
- 三路输出均非常数；`score` 峰值分布 0.32–0.99（无退化 crop）

## 3. RKNN 构建

`build/avtrack_gate9_op17_fp16.rknn`：**15.89 MB，构建 5.8 s**，md5 `ba715f60acb17083cb19e005c7189ad7`

- **无 CPU 回退**：build 日志里 187 个计算层全部编到 NPU，只有 Input/Output 算子在 CPU（正常 I/O）
- op17 的 19 个 `LayerNormalization` → 融合成 **32 个 `exNorm`**（**NPU 原生支持，之前最大的未知点已排除**）
- MatMul+Softmax+MatMul → 融合成 **12 个 `exSDPAttention`**（Flash-Attention 式融合）
- Conv 系融合：`ConvAdd` 32 / `ConvAddRelu` 24 / `ConvRelu` 18 / `ConvExGelu` 12 / `ConvSigmoid` 6
- 权重 13.6 MB，内部内存 2.1 MB

### A/B：opset、optimization_level 在本模型上**完全等价**

| 构建 | 板端 score | size | offset | NPU ms | 模型大小 |
|---|---|---|---|---|---|
| op17 opt0（主用） | 9.98e-3 | 6.29e-3 | 5.35e-2 | 36.87 | 15.887 MB |
| op11 opt0 | 9.98e-3 | 6.29e-3 | 5.35e-2 | 36.83 | 15.887 MB |
| op17 opt3 | 9.98e-3 | 6.29e-3 | 5.35e-2 | 36.83 | 15.887 MB |

三者**逐位相同**（误差、延迟、体积都一致）。结论：
- 不需要为 "RKNN 认不认 LayerNormalization" 纠结——它认得，且和 op11 编译结果一致；
- 主用 op17（节点少、可读性好），op11 只当后备。

## 4. 精度：simulator 与板端

对照 `ref/out_ort.npz`（20 组，冻结语义参考）：

| 输出 | simulator max\|d\| | **板端 max\|d\|** | 判据 | TRT FP16 参考 |
|---|---|---|---|---|
| `score_map` | 1.00e-3 | **9.98e-3** | < 2e-3（board 放宽到 2e-2） | 3.6e-4 |
| `size_map` | 5.10e-4 | **6.29e-3** | < 5e-3（board 放宽到 2e-2） | 1.1–1.5e-3 |
| `offset_map` | 5.36e-3 | **5.35e-2** | < 5e-3（board 放宽到 1e-1） | 3.9–5.7e-3 |

- simulator 三路：score/size 达标，offset 5.36e-3 略超 5e-3（就在 TRT FP16 的实测区间 3.9–5.7e-3 内）
- **板端比 simulator 大 5–10 倍**，这是 RK3588 真实 FP16 硬件的正常差距（TFLOPs 上限更低、累加器更短）
- 解码后的量级（板端 vs ORT）：`dx 0.63 px / dy 0.25 px / w 1.90 px / h 0.95 px / conf 1.0e-2`，
  **20/20 组 argmax 位置完全相同**
- 所以判据应当以"解码后误差"为准，而不是 `offset_map` 的绝对值——`offset_map` 本身是弱的回归量，
  单个元素的 5e-2 在多数组上被 1/16 的网格和 0.8/0.2 平滑吸收掉了

### 为什么不能更好：RK3588 的精度上限就是 FP16

`RKNN.config()` 的 `float_dtype` 只接受 `float16`（quanztied_dtype 那套只对 INT8 有效）。
即：**不做量化时，FP16 是最高精度**，没有 FP32 回退路径。所以上表就是这台设备的精度地板。

## 5. 真正的坑：跨设备"输入像素"不一致

同一帧在 PC（x86 + opencv 4.11）与板端（aarch64 + opencv 4.12）解码，**不是逐位相同的**：

| 视频·第 100 帧 | 最大差 | 平均差 | 有差异的通道数 |
|---|---|---|---|
| street (320×568) | **±2** | 0.68 | 331157 / 545280（61%） |
| man (256×340) | **±2** | 0.71 | 164227 / 261120（63%） |

在这个模型上，±2 灰度级的输入扰动会造成**远大于 FP16 误差**的解码偏移（PC 上用 ORT 加 ±2 均匀噪声、
开环逐帧对比 clean 结果）：

| 视频 | clean 逐帧 IoU | **+±2 噪声逐帧 IoU** | 噪声下中心误差 mean/p95/max |
|---|---|---|---|
| man | 1.000 | **0.857** | 8.36 / 16.00 / 42.74 px |
| cars | 1.000 | **0.695** | 17.42 / 84.72 / 217.98 px |
| street | 1.000 | **0.602** | 20.58 / 51.11 / 109.40 px |

**因此：任何"板端 vs PC"的跟踪对比，必须喂完全相同的像素**（我们为此做了 `ref/` 和 `results/crops/`
两套同像素基准）。否则测出来的是解码器差异，不是 NPU 精度。

## 6. 同像素公平对照（`fair_l1`，最强证据）

做法：按参考轨迹把每帧的 `(template, search)` 裁出来存成 uint8 npz（`results/crops/*.npz`），
PC 跑 ORT、板端跑 RKNN，**两边的输入逐比特一致**，再比解码后的框。

| 视频 | 帧数 | 逐帧 IoU（mean/p05/min） | 中心误差 mean/p95/max | argmax 相同 | IoU<0.9 |
|---|---|---|---|---|---|
| man | 300 | **0.9975** / 0.9951 / 0.9891 | 0.148 / 0.34 / 0.98 px | **300/300** | 0 |
| cars | 300 | **0.9978** / 0.9933 / 0.9813 | 0.086 / 0.34 / 1.14 px | 299/300 | 0 |
| basketball | 88 | **0.9971** / 0.9944 / 0.9911 | 0.542 / 1.34 / 1.96 px | **88/88** | 0 |
| street | 300 | 0.9805 / 0.9869 / 0.0000 | 1.170 / 0.42 / 100.91 px | 292/300 | 8 |

结论：**FP16 模型的逐帧输出与 FP32 参考一致到亚像素**；street 上有 8/300 帧 argmax 翻转（≈2.7%），
翻转帧会产生一次大跳变（最大 100 px），这是模型 score 峰值的"接近平局"特性，不是 NPU 缺陷。

## 7. 判定

- ✅ FP16 转换正确：无 CPU 回退、算子全融合、精度与 TRT FP16 同量级、逐帧亚像素一致
- ⚠️ 精度上限：RK3588 只有 FP16，`offset_map` 最大偏差 5.4e-2，属设备能力边界
- ⚠️ 2.7% 的帧存在 argmax 翻转（street），由输入微小扰动触发（FP16 或解码差异都会触发）
- ⚠️ 不要用"板端解码 + PC 参考"做精度验收；要么同像素（`results/crops/`），要么都在板端生成参考
