# AVTrack 模型转换 / 部署 / 调优全链路复盘

> 覆盖：PyTorch checkpoint → ONNX →（TensorRT | RKNN）→ 板端推理 → 跟踪闭环
> 包含：历史 TensorRT 路径（`PyTracking_ONNX-TensorRT`）的经验 + 本次 RK3588 路径的实测
> 时间：2026-11-16 → 11-17　产物：`/path/to/AVTrack-RK3588/`

---

## 0. 一句话总览

**这类单目标跟踪模型（OSTrack 系 joint-attention ViT）转换的真正难点，从来不是"算子支不支持"，
而是"你导出的那张图，是不是模型实际在跑的那张图"。** 本次 90% 的排查时间花在这一件事上：
一个数据相关的跳块开关被 `torch.onnx.export` 固化成常量，导致同一份权重能导出**两个不同函数**的两张图。

---

## 1. 全链路图

```
                     ┌──────────────────────────────────────────────────────────┐
 训练产物             │ AVTrack_ep0300.pth.tar (249 MiB, epoch 300, 272 tensors) │
                     └──────────────────────────────────────────────────────────┘
                                          │  strict=True 全量加载（0 missing）
                                          ▼
                     ┌──────────────────────────────────────────────────────────┐
  ① 模型复原          │ lib.models.avtrack.build_avtrack(cfg, training=False)     │
                     │  DeiT-Tiny(12 blk,192d,3head) + CENTER head              │
                     │  template 64 tok + search 256 tok = 320 tok joint attn   │
                     └──────────────────────────────────────────────────────────┘
                                          │  freeze_gate()：把动态跳块钉成常量表
                                          ▼
                     ┌──────────────────────────────────────────────────────────┐
  ② 导出 ONNX         │ wrapper(template,search) → (score,size,offset)            │
                     │  SDPA 手动分解 / CPU / opset 17（429 节点）/ opset 11（582）│
                     │  验收：block 集合、两次 dummy 差 0.0、ORT vs torch ≤6e-6    │
                     └──────────────────────────────────────────────────────────┘
                        │                                     │
            ③a 金标准    │                                     │  ③b 编译器路径
                        ▼                                     ▼
        ┌───────────────────────────────┐      ┌──────────────────────────────────┐
        │ ref/pairs.npz (20 组 uint8 RGB)│      │ TRT: FP16 engine (~2.8 ms, 桌面GPU)│
        │ out_torch.npz / out_ort.npz   │      │ RKNN: FP16 .rknn (15.89 MB)        │
        │ 用途：所有数值对照的唯一基准    │      │  RK3588: 37 ms/帧/核                │
        └───────────────────────────────┘      └──────────────────────────────────┘
                        │                                     │
                        └──────────────┬──────────────────────┘
                                       ▼
                     ┌──────────────────────────────────────────────────────────┐
  ④ 板端打通          │ run_board：20 组同像素对照（map 误差 + 解码误差 + argmax）│
                     │ 100 / 8000 帧稳定性、内存、温度                           │
                     └──────────────────────────────────────────────────────────┘
                                       ▼
                     ┌──────────────────────────────────────────────────────────┐
  ⑤ 性能              │ pre / npu / post 三段拆解；1/2/3 实例；NHWC vs NCHW         │
                     │ RK3588：25.1 FPS(单核) / 63.9 FPS(3 进程)                 │
                     └──────────────────────────────────────────────────────────┘
                                       ▼
                     ┌──────────────────────────────────────────────────────────┐
  ⑥ 跟踪闭环          │ track_headless（rknn/ort/torch 三后端，无 GUI，CSV+JSON）  │
                     │ 前处理 crop_hwc → 归一化 → 推理 → 汉宁窗 argmax → 状态更新  │
                     │ L0 复现 / L1 后端 / L2 语义；同像素基准才能谈"精度"        │
                     └──────────────────────────────────────────────────────────┘
```

---

## 2. 阶段 ①：先把"模型到底是什么"弄清楚（这一步不要省）

### 2.1 必须读的源码位置

| 文件 | 看什么 | 本次的关键发现 |
|---|---|---|
| `lib/models/avtrack/vision_transformer.py:678-709` | `forward_features` | **数据相关跳块**（后面 80% 的坑都在这里） |
| `lib/models/avtrack/avtrack.py:112-127` | `forward_head` | `cat_feature[:, -256:]` = search 特征 → 确认 joint attention |
| `lib/models/layers/head.py:145-163` | `cal_bbox` | 输出语义：`(idx+off)/16`，`size[0]=W`、`size[1]=H`，offset 无 sigmoid |
| `export_onnx.py` | SDPA 补丁、dummy 输入、opset | 补丁写法的差异会影响节点数（见 §4.6） |
| `inference_avtrack_trt.py` | `crop_hwc` / 后处理 / 状态更新 | 前后处理的**权威实现**，直接抄，别自己推导 |
| `experiments/.../deit_tiny_patch16_224.yaml` | 128/256、factor 2/4、mean/std | 预处理口径唯一来源 |

### 2.2 结构事实（实测确认，不是猜）

- `depth=12`、`embed_dim=192`、`num_heads=3`（`deit.py:250`）
- `pos_embed_z=(1,64,192)`、`pos_embed_x=(1,256,192)` 在 `__init__` 里**写死**（不是从 224 插值来的）
- 送入 backbone 的 token 数 = 64 + 256 = **320**，没有 cls token → attention 张量 `1×3×320×320`
- head 输出 `score(1,1,16,16)`、`size(1,2,16,16)`、`offset(1,2,16,16)`
- checkpoint 里 272 个 tensor，`strict=True` 全匹配 → **权重、config、代码三者一致**

> 教训：`strict=False` 是万恶之源。上游 `export_onnx.py` 用 `strict=False`，一旦真有 key 对不上（换 head、改 depth、加蒸馏头），
> 它会静默随机初始化那部分权重，最后表现为"转换成功但精度莫名其妙"。本项目一律 `strict=True`。

### 2.3 环境坑（会在第一步就拦住你）

| 现象 | 根因 | 解法 |
|---|---|---|
| `ModuleNotFoundError: easydict` | `lib/config/avtrack/config.py:1` | `pip install easydict` |
| `torch.load` 报 `Unsupported global: lib.train.admin.stats.AverageMeter` / `No module named 'tensorboardX'` | checkpoint 里 pickle 了训练期对象，反序列化会 import `lib.train.admin.*` → `tensorboard`/`tensorboardX` | `pip install tensorboardX`（或 tensorboard）；装完会打一行 `using tensorboardX instead...` 的 warning，正常 |
| 建模型时卡在 `Downloading: deit_tiny_patch16_224-a1311bcf.pth` | `build_avtrack()` 里硬编码 `deit_tiny_patch16_224(pretrained=True)`，**全量 ckpt 已经覆盖全部权重，这次下载纯属浪费** | monkeypatch `lib.models.avtrack.avtrack.deit_tiny_patch16_224` 强制 `pretrained=False`（历史 TRT 项目的 `export_wsl.py` 就是这么干的） |
| 忘了 `update_config_from_file` | 默认 cfg 是 `vit_base_patch16_224` → `build_avtrack` 抛 `NotImplementedError` | 必读 yaml |
| 想用 `/tmp` 存中间产物 | 本机 WSL 的 `/tmp` 不跨 `wsl -e bash -c` 调用保留 | 一律写到工作目录 |

---

## 3. 阶段 ②：pt → ONNX（含历史 TensorRT 路径的经验）

### 3.1 标准流程

1. **包一层 wrapper**：模型 `forward(template, search, template_anno, search_anno)` 需要 4 个入参，
   导出只想要 2 个；而且返回 dict。wrapper 里造零 anno、只返回 3 路输出。
2. **替换 SDPA**：torch ≥2.0 的 `F.scaled_dot_product_attention` 在 ONNX 里不好导出（且 flash 路径不可追踪），
   注入一个"纯 MatMul+Softmax"的手写实现：
   ```python
   torch.nn.functional.scaled_dot_product_attention = manual_sdpa
   ```
   注意：模型里是 `if self.fused_attn: F.sdpa(...) else: 手写分支`，所以**必须用 monkeypatch 而不是改 else 分支**——
   `fused_attn` 在不同环境可能为 True/False，改 else 分支等于换了一个函数（见 §4.5）。
3. **eval() + no_grad + CPU**：导出不需要 GPU（实测 0.4–0.9 s），CPU 导出能避开 CUDA 依赖。
4. **opset 选择**：
   - opset 11：LayerNorm 被拆成 ReduceMean/Sub/Pow/Sqrt/Div（19 个 LN → ~114 个碎节点）
   - opset 17：LN 合成单个 `LayerNormalization`（19 个），总节点 582 → 429
   - GELU 要 opset 20 才有 `Gelu` 算子；**不要为了它升到 20**（RKNN/TRT 都未必认，且 Erf 组合本来就能编）
5. **验证三连**（缺一不可）：
   - `onnx.checker.check_model`
   - 同一组固定输入：PyTorch vs ORT `max|d|`（本项目 1e-6 量级）
   - **两次不同 dummy 导出的图互比**（用来抓"跟踪态被固化"这一类问题，见 §4）

### 3.2 历史 TensorRT 路径（`PyTracking_ONNX-TensorRT`）的可复用经验

那个项目是**同一份 checkpoint 的 ONNX → TensorRT FP16**，跑通了并在桌面 GPU 上到 300+ FPS：

| 项 | 值 / 做法 |
|---|---|
| 导出 | CPU、opset 17、`strict=True`、`pretrained=False` patch、`torch.manual_seed(0)`、ORT `assert_allclose` |
| 产物 | `exports/avtrack.onnx`（26.9 MB）+ `exports/avtrack_fp16.engine`（15.7 MB）+ `avtrack.json` 报告 |
| 数值 | torch vs ONNX `1.9e-7 / 8.0e-7 / 4.4e-6`；ONNX FP32 vs TRT FP16 `3.6e-4 / 1.5e-3 / 5.7e-3` |
| 跟踪 | 无 GUI runner（显式 ROI、CSV、warmup 10 帧），`man.mp4` 299 帧全跟住；2.84–3.25 ms/帧 |
| 坑 | 早期版本的 runner 用 `cv2.selectROI`（板端/无显示器不可用）；TRT 引擎构建要几十分钟；引擎与 GPU 型号绑定 |

**它对本次最大的价值**：
1. 提供了"同模型 FP16 允许多大误差"的参照值（我们用它校准 RKNN 的判据）；
2. 提供了**唯一可用的历史 A/B 样本**（它导出的图冻结在 `{...4...}` 跳块表，仓库根目录那两份冻结在 `{...5...}`）；
3. 提供了 `export_wsl.py` 这个导出器骨架（本次的 `export_avtrack.py` 就是照它改的）。

---

## 4. 核心坑 A：数据相关跳块被 trace 固化成常量（本次最大的坑）

### 4.1 现象

同一份 `AVTrack_ep0300.pth.tar`，两个产物里的 block 数量/内容不一样：

| 产物 | producer | opset | 冻结的 block | md5 |
|---|---|---|---|---|
| 仓库根 `avtrack_deit_tiny_patch16.onnx` | pytorch 2.0.1 | 11 | `0,1,2,`**`5`**`,6,7,8,9,10` | `c431f59d…` |
| 历史项目 `exports/avtrack.onnx` | pytorch 2.3.1 | 17 | `0,1,2,`**`4`**`,6,7,8,9,10` | `a329d66d…` |

而 checkpoint 里明明有 **12 个 block**。

### 4.2 一开始的误判（以及为什么它危险）

原始部署手册写的是："现有 ONNX 只有 9 个 block，说明它不是用当前 checkpoint 导出的，可能被裁剪过，
或来自另一版权重；重导出后如果还是 9 个，就去查 `strict=False` 是不是静默漏了 3 个 block。"

这个判断**方向错了**：实测 `strict=True` → `<All keys matched successfully>`，一个 key 都没漏。
按原手册走，下一个 agent 会去翻 `build_avtrack()` 的分支、比对 checkpoint key，半天找不到问题，
最后"修"成 12 块——**那是换了个函数，不是修 bug**。

### 4.3 真正的根因

`lib/models/avtrack/vision_transformer.py`：

```python
for i, blk in enumerate(self.blocks):
    if i > 1 and not is_distill:
        prob_active = (blk.active_score_module(x[:, :, 0].clone())).sigmoid_()
        idx, _ = torch.where(prob_active > 0.5)
        if len(idx) > 0:        # ← python 层的 if，依赖 tensor 数值
            x = blk(x)          # 命中：整块跑
        probs_active.append(prob_active)
    else:
        x = blk(x)              # block 0/1 无条件跑
```

- `active_score_module` 是每块一个 `nn.Linear(320, 1)`，作用在**整个 batch** 上（batch=1 时就是一个标量开关）；
- `torch.onnx.export` 只能记录一条路径，这个 `if` 会被**固化成常量**，同时抛出：
  ```
  TracerWarning: ... vision_transformer.py:694  if len(idx) > 0:
  ```
- **导出日志里的 `TracerWarning` 是本次最高信噪比的信息**，第一眼就该看它。

### 4.4 实证：这两个"都正常"的图，跟踪结果差多少

历史项目里那两张图各建了 TRT engine，在 `man.mp4` 上跑 299 帧（同初始框）：

| 指标 | 实测 |
|---|---|
| 逐帧框 IoU | 均值 **0.913**，p05 0.748，最小 0.464，**82/299 帧 < 0.9** |
| 中心距离 | 均值 3.2 px，p95 8.5 px，**最大 37.1 px** |
| 单列最大差 | x 23.4 / y 20.5 / **w 32.0** / h 14.0 px |
| 同帧单图输出差 | score 6.6e-2 / size 1.5e-1 / **offset 3.0e-1**（argmax 位置一致） |

两张图都能"跟住目标"（视觉上都正常），但**框能差 32 px**。这就是"看起来对了"的典型陷阱。

### 4.5 这个开关本身其实"没训出来"

逐块打印 `active_score_module` 的权重（strict 加载后）：

| block | ‖w‖ | bias | sigmoid(bias) | 实际行为 |
|---|---|---|---|---|
| 2 | **2.0e-4** | +5.77e-4 | 0.500144 | 输入无关常数 → **永远跑**（余量 1.4e-4） |
| 3 | 7.5e-3 | −9.92e-4 | 0.499752 | 近似常数 → **永远跳**（余量 2.5e-4） |
| 4 | 1.1e-1 | −4.30e-3 | 0.498926 | 输入相关，**随帧翻转** |
| 5 | 1.0e-1 | +4.71e-3 | 0.501177 | 输入相关，**随帧翻转** |
| 6–10 | 1.1–8.3 | +0.13~+1.20 | 0.53~0.77 | 恒跑 |
| 11 | 1.1 | −0.033 | — | logit 深度饱和，13 组输入概率全 <1e-6 → **永远跳** |

结论：
- block 11 事实上被训练"关死"了；
- block 2/3 的 gate 是**准常数**，离 0.5 只差 1e-4~1e-3；
- block 4/5 会在帧间翻转（13 组合成输入里翻了 1 次）；
- **"跑不跑 2/3/4/5"由数值噪声决定** —— 训练时的 `activeness_loss` 显然没压住这几块。

这也解释了为什么两张图都能跟得住：训练时就是"逐样本随机子集"，模型对子集有鲁棒性。
但它同时意味着：**部署侧必须自己钉一张确定性的表**，没有"正确答案"，只有"选定的语义"。

### 4.6 解法：`freeze_gate()`（不改上游、不动参数）

```python
def freeze_gate(model, run_blocks=(0,1,2,5,6,7,8,9,10), logit=50.0):
    for i, blk in enumerate(model.backbone.blocks):
        lin = blk.active_score_module
        const = float(logit) if i in run_blocks else -float(logit)
        def fwd(x, _c=const, _lin=lin):
            return torch.full((x.shape[0], 1), _c, dtype=_lin.weight.dtype, device=x.device)
        lin.forward = fwd          # 只换实例方法，state_dict 不变
    return model
```

- **为什么选 `{跳 3,4,11}`**：等于训练的"期望路径"（平均 9 块：2 跑、3 跳、4/5 取一、11 跳），
  且**恰好与仓库里现有 ONNX / TRT engine 同语义** → 历史 TRT 验证可直接复用；
- **为什么用 monkeypatch 而不是改源码**：上游 `avtrack.py` 已有未提交改动、`__pycache__` 常年 dirty，
  `git apply` 并不干净；而且改源码 = 改了别人的训练代码，容易在别的实验里被误用；
- **为什么 logit 用 ±50**：`sigmoid(±50)` 就是 1/0，离 0.5 足够远，任何 FP16 噪声都翻不动。

验证方式（三条，缺一不可）：
1. 导出后解析 ONNX 里 `blocks.N.` 的名字，断言集合 == `RUN_BLOCKS`；
2. 用**两种不同的 dummy**（灰图 / randn）各导一次，同一输入输出差必须为 **0.0**（实测就是 0.0）；
3. 和金标准 `out_ort.npz` 比（见 §5）。

### 4.7 顺带收获：opset 会影响节点数，但不影响数值

SDPA 补丁里 `scale` 的写法会造成节点数差异（**数值等价**）：

| 写法 | op17 节点 | op11 节点 | 差异 |
|---|---|---|---|
| `1 / math.sqrt(query.size(-1))`（`export_onnx.py` 原版） | **429** | **582** | 基准 |
| `1 / (query.shape[-1] ** 0.5)` | 474 | 627 | 每块多 `Pow`+`Reciprocal`+2×`Constant`，共 +45 |

所以**不要拿"总节点数"当验收硬指标**，要看 `LayerNormalization=19 / Softmax=9 / MatMul=54` 和 block 集合。

---

## 5. 阶段 ③a：建"金标准"（后面所有对比的唯一基准）

### 5.1 为什么必须用 uint8 RGB 存输入

```
ref/pairs.npz:
  template (N,128,128,3) uint8 RGB
  search   (N,256,256,3) uint8 RGB
  meta     json（video / frame / bbox / crop_sz / 帧尺寸）
```

- 存 **uint8**：归一化留到下游同一个函数里做，PC 和板端走**同一份预处理代码**，不会因为"谁先归一化"产生 1e-3 级差异；
- 存 **RGB**：`crop_hwc` 出来的视频帧是 BGR，先转 RGB 再存；板端 `prep_rgb` 不再转色，避免二次转换；
- 存 **meta 里的 `crop_sz`**：后处理还原坐标要用它，漏了就没法复算。

20 组 = 4 段视频 × 5 帧（首帧做 template，其余帧做 search），覆盖小/中/大目标和边界情况
（`basketball` 的视频是 2160×4096、ROI 在画面下方，crop 会大幅越界补零，正好是个边界样本）。

### 5.2 参考输出必须"同语义"

- **PyTorch 参考**：`freeze_gate()` + `eval()` 后前向；
- **ORT 参考**：跑冻结后的 ONNX；
- 两者差 `1.22e-6 / 1.01e-6 / 4.65e-6` → 说明"导出没有改变函数"。

> 千万不要拿"活的 PyTorch eval"当参考：它每帧的跳块可能不同，跟固化图比出来的差异无法归因。

---

## 6. 阶段 ③b：ONNX → RKNN

### 6.1 构建脚本要点

```python
rknn.config(target_platform='rk3588', optimization_level=0)
rknn.load_onnx(model=onnx, inputs=['template','search'],
               input_size_list=[[1,3,128,128],[1,3,256,256]])
rknn.build(do_quantization=False)      # 第一版不加 mean/std、不做量化
rknn.export_rknn(out)
```

- `inputs` / `input_size_list` 的顺序必须和 ONNX 输入顺序一致（`template, search`），反了不报错、但数值全错；
- **第一版不要加 `mean_values/std_values`**：先喂 float32（已归一化）数据把数值打通，
  第二版再试 u8 + mean/std 做单变量对比。否则出错分不清是"模型问题"还是"预处理问题"。

### 6.2 三个"surprise"

1. **op17 的 LayerNormalization 完全支持**：19 个 LN 融合成 **32 个 `exNorm`**；
   MatMul+Softmax+MatMul 融合成 **12 个 `exSDPAttention`**（Flash Attention 式融合）；
   Conv 系融合：`ConvAdd 32 / ConvAddRelu 24 / ConvRelu 18 / ConvExGelu 12 / ConvSigmoid 6`。
   → 187 个计算层**全部上 NPU，零 CPU 回退**（只有 Input/Output 在 CPU，那是 I/O 胶水）。
2. **op17 / op11 / optimization_level=3 三种构建逐位等价**：误差、延迟、体积全都一样
   （`9.98e-3 / 6.29e-3 / 5.35e-2`、`36.83–36.87 ms`、`15.887 MB`）。
   → 不需要为"LN 会不会回退"做取舍，op11 只当保险。
3. **simulator 不能加载 `.rknn` 跑**：`load_rknn` 之后 `init_runtime()` 会报
   "not support inference on the simulator, please set 'target' first"。
   必须在 **`build()` 的同一个会话**里 `init_runtime()` + `inference()`（`build_and_sim.py` 已封装）。

### 6.3 精度上限：FP16 是天花板

`RKNN.config()` 的 `float_dtype` **只接受 `float16`**（`quantized_dtype` 那一套只对 INT8 有效）。
即：不做量化时，RK3588 最高精度就是 FP16，**没有 FP32 可退**。所以板端误差就是设备地板，不是配置问题。

### 6.4 simulator 与板端的差距（要预期到）

| 输出 | simulator | 板端 | 倍数 | TRT FP16 参照 |
|---|---|---|---|---|
| score | 1.00e-3 | 9.98e-3 | 10x | 3.6e-4 |
| size | 5.10e-4 | 6.29e-3 | 12x | 1.1–1.5e-3 |
| offset | 5.36e-3 | 5.35e-2 | 10x | 3.9–5.7e-3 |

**simulator 只用来筛明显错误，判据一律以板端为准**（RT-DETR 的老教训在这里再次成立）。
另外：`offset_map` 的绝对值不要当验收指标——它是弱回归量，单个元素 5e-2 会经过 1/16 网格和 0.8/0.2 平滑衰减，
**要看解码后的量**：`dx 0.63 px / dy 0.25 px / w 1.90 px / h 0.95 px / conf 1.0e-2`、argmax 20/20 相同。

---

## 7. 阶段 ④：板端打通

### 7.1 数值对照的正确姿势

`run_board.py`：读 `ref/pairs.npz` → 板端推理 → 和 `ref/out_ort.npz` 比三路 `max|d|` →
再**解码**成 `dx/dy/w/h/conf` 比一次 → 统计 argmax 是否同位置。

验收线（实测校准，不是拍脑袋）：
`score < 2e-2`、`size < 2e-2`、`offset < 1e-1`、`dx/dy < 2 px`、`w/h < 3 px`、`argmax 20/20`。

### 7.2 坑：`rknnlite` 输入格式

- `inference(inputs=[tpl, src], data_format='nchw')` 中，`inputs` 必须是**两个 ndarray 组成的 list**，
  不能写成 `[list(tpl), list(src)]`（会报 `'list' object has no attribute 'size'`）；
- 日志会提示 "input[0] need NHWC data format... will be changed to NHWC"：这说明模型内部要 NHWC，
  rknnlite 自动转。实测 NHWC 直传只快 0.8%（36.89 vs 37.18 ms），**保持 NCHW 即可**；
  但 NHWC 必须传 **4 维** `(1,H,W,3)`，传 3 维会报 `need 4dims input`。

### 7.3 稳定性

- 100 帧循环：`MemAvailable` 变化 −252 KB（噪声级）；
- 8000 帧（约 6 分钟）单核：NPU p50 37.3 ms、mean 41.4 ms；温度 38.8 → 41.6 °C；内存 875 → 880 MB。
  → 不崩、不泄漏、不降频；长跑吞吐按 21–25 FPS 预估。

> 坑：`bench.py` 早期版本把整段视频**预读进内存**，8000 帧长稳直接吃 2 GB（还是两个进程同时跑，
> 造成互相抢核、数字失真）。改成流式读帧后恢复正常。**长稳测试一定要看内存曲线。**

---

## 8. 阶段 ⑤：性能（RK3588）

### 8.1 三段拆解（300 帧，预热 30）

| 配置 | pre | npu | post | 合计 | FPS |
|---|---|---|---|---|---|
| 1 实例 · core0 | 2.69 | **37.13** | 0.09 | 39.91 | **25.1** |
| 1 实例 · 三核 | 3.02 | 32.27 | 0.10 | 35.38 | 28.3 |
| 3 进程 · 各一核 | 3.2–5.0 | 41–45 | 0.10 | 44.3–50.4 | **63.9 合计** |

- 后处理 0.09 ms，可忽略；预处理 2.7 ms（约 7%）；**瓶颈 100% 在 NPU**
- 与 RT-DETR 的经验相反：**本模型多实例（各绑一核）总吞吐 > 单实例用三核**（63.9 vs 28.3）
- 只有 3 个 NPU core → **第 4 路并行做不到**（`core=3` 直接异常退出）

### 8.2 还能优化什么（按性价比）

1. **C++ + 零拷贝**：现在 pre 只有 2.7 ms，Python 循环和 H2D 拷贝的收益有限；
   真正的大头是 37 ms 的 NPU 时间，模型级优化（剪枝/INT8）才是数量级手段；
2. **INT8**：attention 是瓶颈，量化收益未必大，且要重训校准集；本次没做（FP16 已达标）；
3. **多路**：直接 3 进程 × 1 核，别折腾三核单进程。

---

## 9. 阶段 ⑥：跟踪闭环 + "精度到底怎么评"

### 9.1 前后处理（唯一权威实现，别自己推）

| 步骤 | 公式 / 做法 |
|---|---|
| 裁图 | `crop_sz = ceil(sqrt(w*h)*factor)`，template factor 2.0→128，search factor 4.0→256，越界补 0 后 resize |
| 归一化 | BGR→RGB，`/255`，减 `[0.485,0.456,0.406]`、除 `[0.229,0.224,0.225]` |
| 定位 | `score * 汉宁窗(16×16)` → `argmax`；`conf` 取**惩罚前**的原始响应 |
| 还原 | `cx_rel=(x_idx+off_x)/16`；`w=size[0]`、`h=size[1]`（Ch0=W！）；再乘 `256` 与 `crop_sz/256` |
| 更新 | `cx += dx`；尺寸 `0.8*old + 0.2*new`；`conf > 0.15` 才更新；宽高不超画面 |

### 9.2 三级对比口径（本次最重要的方法论）

| 级别 | 对比什么 | 是否需要同像素 | 作用 |
|---|---|---|---|
| **L0 复现** | 我们导出的 ORT 轨迹 vs 历史 TRT 轨迹 | 否（都跟真实视频比） | 证明"重导出没有改变模型"：实测 IoU **0.9691** |
| **L1 后端** | 板端 RKNN vs PC ORT，**同一张 gate9 图** | **必须** | 反映 FP16/NPU 误差：**0.9971–0.9978**（street 0.9805） |
| **L2 语义** | vs 历史 5-block（同语义）/ 4-block（跨语义） | 否 | 反映"换后端/换冻结表"的代价：0.9717 / 0.9449 |

### 9.3 关键发现：±2 灰度级的输入差异 > FP16 误差

同一帧在 PC（x86 + OpenCV 4.11）和板端（aarch64 + OpenCV 4.12）解码结果**不是逐位相同**：

| 帧 | 最大差 | 平均差 | 有差异的通道 |
|---|---|---|---|
| street f100 | ±2 | 0.68 | 61% |
| man f100 | ±2 | 0.71 | 63% |

而在 PC 上用 ORT 加 ±2 均匀噪声，逐帧解码结果就崩了（对照 clean）：

| 视频 | clean | ±2 噪声 |
|---|---|---|
| man | IoU 1.000 | **0.857**（中心 8.4 px） |
| cars | 1.000 | **0.695**（17.4 px） |
| street | 1.000 | **0.602**（20.6 px） |

所以：
- **"板端解码 + PC 参考"直接比轨迹，测的是解码器差异，不是 NPU 精度**；
- 想衡量 NPU → 用 `dump_crops.py` 把每帧 `(template, search)` 裁成 uint8 存盘，两边跑同一份；
- 想要"板端与 PC 逐帧一致" → 必须统一解码路径（同一 ffmpeg/OpenCV 版本，或无损帧序列），否则不可能。

### 9.4 闭环 vs 开环（归因工具）

- **开环**（用参考轨迹的状态裁图、不更新自身状态）：能隔离"单帧解码误差"；
  实测 cars 中心 1.14 px、street 3.49 px，偶发大跳（67–83 px）来自 argmax 翻转帧；
- **闭环**（正常跟踪 300 帧）：误差会通过状态更新放大。本次闭环 L1：
  man 0.976 / basketball 0.938 / cars 0.882 / street 0.680（但 4 段视频肉眼都在目标上，见 `results/frames/cmp_*.png`）。

### 9.5 argmax 翻转帧

street 上 8/300 帧（≈2.7%）分数峰值接近平局，±1 LSB 的差异就会换峰，换峰帧会跳一次（最大 100 px）。
这是模型本身的特性（score map 的次峰很接近主峰），不是 NPU 缺陷。
若业务敏感，可以在后处理加"峰值/次峰裕度"判断，裕度不足时沿用上一帧的框（属于应用层策略，会改变指标）。

---

## 10. 踩坑总表（按类别）

### 10.1 环境 / 工具类

| # | 现象 | 根因 | 解法 |
|---|---|---|---|
| 1 | `No module named easydict` | config 依赖 | 装 |
| 2 | ckpt 反序列化失败（AverageMeter/StatValue） | pickle 了训练期对象 | 装 `tensorboardX` 或 `tensorboard` |
| 3 | 建模型卡在 Downloading deit 权重 | `pretrained=True` 硬编码 | monkeypatch 成 False |
| 4 | `NotImplementedError` @ build_avtrack | 忘了加载 yaml（默认 vit_base） | 先 `update_config_from_file` |
| 5 | `/tmp` 里的文件下次调用不见了 | WSL `/tmp` 不跨调用保留 | 写工作目录 |
| 6 | `scp` 一次传多个远端文件偶发失败 | 多源参数兼容性 | 逐文件传，或 tar 管道 |
| 7 | PowerShell 里 `wsl bash -c "...复杂引号..."` 被拆坏 | Windows 侧引号转义 | **一律写成 .sh/.py 文件再执行**（本次所有脚本都这么干） |
| 8 | `bench.py` 长稳吃 2 GB 内存 | 预读整段视频 | 流式读帧 |
| 9 | 两个长稳进程同时跑，数字失真 | 抢同一个 NPU core | 跑之前 `pkill`，确认空闲 |

### 10.2 模型 / 导出类

| # | 现象 | 根因 | 解法 |
|---|---|---|---|
| 10 | ONNX 只有 9 个 block（checkpoint 有 12） | **数据相关跳块被 trace 固化** | `freeze_gate()` + 断言 block 集合 + 双 dummy 校验 |
| 11 | 同一权重两次导出，跟踪框差 32 px | 同上（4 号 vs 5 号块） | 声明并钉死冻结表 |
| 12 | `strict=False` 静默漏权重 | 上游脚本的默认写法 | 一律 `strict=True` |
| 13 | 节点数从 429 变 474 | SDPA 的 `scale` 写法不同 | 用 `1/math.sqrt(d)`，节点数别当硬指标 |
| 14 | 想用 opset 20 拿到 `Gelu` | 没必要，Erf 组合能编 | 停在 17 |
| 15 | `.cuda()` 导出失败/多余 | 脚本写死 GPU | CPU 导出（0.4–0.9 s） |
| 16 | export 时报 `TracerWarning: len(idx)` | 就是坑 10 的信号 | **看到 TracerWarning 必须解释清楚再继续** |

### 10.3 板端 / 运行类

| # | 现象 | 根因 | 解法 |
|---|---|---|---|
| 17 | `init_runtime` 报 simulator 不支持 | `load_rknn` 加载的模型不能做 simulator | build 同会话里跑 |
| 18 | `'list' object has no attribute 'size'` | `inputs=[list(a), list(b)]` | 传 ndarray |
| 19 | NHWC 传 3 维报 `need 4dims` | 要 `(1,H,W,3)` | 加 batch 维 |
| 20 | 板端比 simulator 差 10 倍 | 真实 FP16 硬件累加精度更低 | 判据以板端为准，阈值放宽 |
| 21 | 想找"更高精度"配置 | `float_dtype` 只支持 float16 | 接受 FP16 是上限 |

### 10.4 评测 / 归因类（最容易自欺欺人）

| # | 现象 | 根因 | 解法 |
|---|---|---|---|
| 22 | 板端轨迹与 PC 差很多，以为 NPU 坏了 | 解码差 ±2 灰度级 | 同像素基准（crop 包） |
| 23 | 用"单帧 map 的 max|d|"当精度指标 | map 接近 ≠ 跟踪一致 | 看解码后量 + 跟踪级指标 |
| 24 | 对"活的 PyTorch"设 <5 px 判据 | PyTorch 每帧的图可能不同 | 参考必须冻结语义 |
| 25 | 跟踪闭环差异被当成实现错误 | 闭环会放大单帧差异 | 用开环拆解归因 |
| 26 | 只看均值不看尾部 | 少量翻转帧主导漂移 | 报 p05/min/#IoU<0.9 |

---

## 11. 方法论沉淀（下次换个模型也能用）

1. **先读源码找"数据相关控制流"**：`grep -nE "if .*(len|item|where|any|all)\(" *.py`；
   ONNX 只能表达静态图，任何依赖 tensor 数值的 python `if/while` 都会被"固化"。
2. **`TracerWarning` 是最高信噪比**，导出日志必须逐条解释。
3. **`strict=True`**，永远。
4. **导出必须可复现**：固定 dummy + 断言 block 集合 + 两次不同 dummy 互比（差值应为 0）。
5. **参考语义先冻结**，再谈精度；否则所有数字都无法归因。
6. **同像素对照**：跨设备（x86/aarch64、不同 OpenCV）比较前，先把输入落盘成 uint8。
7. **一级指标只回答一个变量**：L0 复现 / L1 后端 / L2 语义，不要混在一起。
8. **单变量实验**：op17 vs op11、opt0 vs opt3、NHWC vs NCHW、有/无噪声，一次只动一个。
9. **长稳必看内存与温度**，且先确认没有别的进程抢核。
10. **把"失败"也写进结果文档**（本次 6 个 A/B 里 3 个是"无差异"，这本身就是结论）。
11. **脚本可一键重跑** + 产物带 md5 + 每份报告写清"gate 表 / 是否同像素 / 对比对象"。
12. **密码不落盘**：只用 `SSHPASS` 环境变量传递。

---

## 12. 如果重来一次：最短路径清单

```text
[ ] 0. 装 easydict / tensorboardX；确认 timm、torch 版本
[ ] 1. 读 vision_transformer.forward_features 与 head.cal_bbox，确认 token 数与输出语义
[ ] 2. strict=True 加载 ckpt，打印 blocks 数 + pos_embed_z/x 形状
[ ] 3. grep 数据相关控制流 → 找到 active_score_module 跳块
[ ] 4. 打印每块 gate 权重/概率 → 定冻结表（本次 {跳 3,4,11}）
[ ] 5. freeze_gate() + 导出 op17（CPU、SDPA 手写、opset 17）
[ ] 6. 断言 block 集合；灰图/randn 双 dummy 互比 = 0.0；ORT vs torch ≤1e-5
[ ] 7. 建 ref/pairs.npz（uint8 RGB + meta.crop_sz）+ out_torch/out_ort
[ ] 8. RKNN：build+sim 同会话；看日志有没有 CPU 回退；记录 map 误差初值
[ ] 9. 板端：20 组同像素 → map 误差 + 解码误差 + argmax；100 帧/8000 帧稳定
[ ]10. bench：pre/npu/post 三段；1/2/3 实例；确认 4 路不可行
[ ]11. 跟踪：三后端 runner；L0/L1/L2；同像素 crop 包做 L1；开环拆解归因
[ ]12. 报告：latency / accuracy / tracking / ACCEPTANCE，每条都带证据路径
```

---

## 13. 数字速查

| 项 | 值 |
|---|---|
| checkpoint | 249 MiB / 261 MB，epoch 300，272 tensors，strict 全匹配 |
| ONNX op17 / op11 | 429 节点（LN 19/Softmax 9/MatMul 54）/ 582 节点；md5 `6b0c42c2…` / `dd66e27b…` |
| RKNN FP16 | 15.89 MB，build 5.8 s，md5 `ba715f60…`；187 层全 NPU |
| 融合 | exNorm 32、exSDPAttention 12、ConvAdd 32、ConvAddRelu 24、ConvRelu 18、ConvExGelu 12、ConvSigmoid 6 |
| simulator 误差 | 1.00e-3 / 5.10e-4 / 5.36e-3 |
| 板端误差 | 9.98e-3 / 6.29e-3 / 5.35e-2；解码 dx 0.63 / dy 0.25 / w 1.90 / h 0.95 px、conf 1.0e-2、argmax 20/20 |
| 延迟 | 单核 37.1 ms（整帧 39.9，25.1 FPS）；三核 32.3 ms（28.3 FPS）；3 进程 63.9 FPS |
| 长稳 | 8000 帧，npu p50 37.3/mean 41.4 ms，38.8→41.6 °C，内存 +5 MB |
| 公平 L1（同像素） | man 0.9975 / cars 0.9978 / basketball 0.9971 / street 0.9805 |
| L0 复现 | vs 旧 5-block TRT：0.9691，中心 2.40 px |
| L2 语义 | vs 旧 5-block：0.9717；vs 旧 4-block：0.9449 |
| 解码差异 | 同帧 max ±2、mean 0.68–0.71，61–63% 通道不同 |
| ±2 噪声敏感度 | 逐帧 IoU 0.857 / 0.695 / 0.602（man/cars/street） |
| TRT 参照（桌面 GPU） | 2.84–3.25 ms/帧，307–352 FPS；FP16 误差 3.6e-4 / 1.5e-3 / 5.7e-3 |

---

## 附：文件位置

| 内容 | 路径 |
|---|---|
| 部署手册（含首轮实测结论） | `/path/to/AVTrack/DEPLOY_RK3588.md` |
| 交付总览 / 复现 | `AVTrack-RK3588/README.md` |
| 验收清单 | `AVTrack-RK3588/results/ACCEPTANCE.md` |
| 三份报告 | `AVTrack-RK3588/results/{latency,accuracy,tracking}.md` |
| 全部脚本 | `AVTrack-RK3588/scripts/` |
| 历史 TRT 项目 | `~/projects/PyTracking_ONNX-TensorRT/`（`exports/*.json`、`results*/avtrack.csv`、`早期 TensorRT 项目的导出脚本`） |
