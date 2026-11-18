# AVTrack → RK3588 部署执行手册（v2）

> **v2 修订日**：2026-11-16。本文件不属于上游 v2/v3/v4 文档体系，面向接手执行的 agent：
> 目标是让另一个 agent 在不了解上下文的情况下也能按步骤做完。
>
> 上游模型：`C:\path\to\AVTrack`（ICML 2024 / TCSVT 2025 的 AVTrack-DeiT）
> 目标平台：RK3588（RK3588，`192.168.x.x`）
> 工具链：rknn-toolkit2 2.5.0 / librknnrt 2.5.0
> 已有先例：`~/projects/PyTracking_ONNX-TensorRT`
> —— 同一 checkpoint 的 **ONNX → TensorRT FP16** 全链路已跑通，本手册大量复用它的脚本与实测数字。

---

## 0. v2 相对 v1 的变更（先看这个）

| # | v1 的说法 | v2 的结论 | 为什么改 |
|---|---|---|---|
| 1 | 现有 ONNX 只有 9 个 block ⇒ “不是用当前 checkpoint 导出的、可能被裁剪过”，要查 `strict=False` 是否漏权重 | **错。** 根因是 `forward_features` 里**数据相关的动态跳块**，导出时被 trace 固化成常量 | `strict=True` 加载实测 **0 missing / 0 unexpected**；同一 checkpoint 两次导出得到**两个不同的 block 集合**（详见 §0.2） |
| 2 | 参考实现 = “新导出的 ONNX + ORT” | 参考实现 = **冻结 gate 表** 后的 PyTorch + ORT，两边必须同语义 | 未冻结的 PyTorch 每帧的跳块结果可能不同，和固化图比没有意义 |
| 3 | Step 6 验收：“RKNN 与 PyTorch 框中心误差 < 5 px” | 拆成 **L1 后端误差**（<2 px）与 **L2 语义敏感度**（已知上界 32 px） | 两个冻结图之间的框差实测最大 32 px；对“活的 PyTorch”无法设 <5 px |
| 4 | Step 1 要打 patch 删 `avtrack.py` 的 debug print | 导出/参考脚本里 `redirect_stdout` 屏蔽即可，**不改上游** | 少一处上游改动；板端跑 RKNN 不经过 PyTorch，根本不会打印 |
| 5 | Step 3 担心 head 里 `.view(-1,...)` 留动态 batch 维，要改上游 | **不存在**。两份 ONNX 的 Reshape 无 0 维，输入输出全静态 | 实测（`onnx` 扫 `Reshape` 第二输入） |
| 6 | 依赖只提 `tensorboard` | 还要 **`easydict`**；且 `build_avtrack` 会**联网下载 DeiT 权重**，必须 patch 掉 | 实测缺 `easydict` 直接 `ModuleNotFoundError`；下载依赖网络 |
| 7 | 数据集“本机没有”，需找用户 | **有素材**：`/path/to/videos/{man,cars,street,basketball}.mp4` + `.txt` ROI | 上个项目留下；但仍**没有 GT 框**，只能弱评测 |
| 8 | FP16 判据 `2e-3 / 5e-3`（凭经验） | 保留数量级，但注明来源：TRT FP16 vs ONNX FP32 实测 `score 3.6e-4 / size 1.5e-3 / offset 5.7e-3` | 让阈值有依据 |
| 9 | 工作目录放仓库内 `AVTrack-RK3588/` | 挪到**仓库外** `C:\path\to\AVTrack-RK3588\` | 上游仓库已有 100+ 个 tracked `.pyc` 常年 dirty，不再往里加东西 |

---

## 0.1 一句话结论

**转换本身不是难点，语义冻结才是。** 现有 ONNX/TRT 已证明这个模型能上推理引擎（FP16、300+ FPS、跟踪正常）；
RK3588 上真正的风险是：
① **必须先钉死一张“跳块表”**，否则 PyTorch / ONNX / RKNN / 历史 TRT 产物各自是不同函数；
② RKNN 对 opset-17 的 `LayerNormalization` 支持未验证；
③ 预处理/后处理链路要用“同语义参考”对齐，而不是和活的 PyTorch 硬比。

---

## 0.2 必须先理解的一件事：数据相关跳块被固化成常量

### 机制

`lib/models/avtrack/vision_transformer.py:678-709` 的 `forward_features`：

```python
for i, blk in enumerate(self.blocks):
    if i > 1 and not is_distill:
        prob_active = (blk.active_score_module(x[:, :, 0].clone())).sigmoid_()
        idx, _ = torch.where(prob_active > 0.5)
        if len(idx) > 0:
            x = blk(x)              # 命中 → 整块跑
        probs_active.append(prob_active)
    else:
        x = blk(x)                  # block 0/1 无条件跑
```

- `active_score_module` 是每块一个 `nn.Linear(320, 1)`，作用在整个 batch（batch=1 时就是一个标量开关）；
- `torch.onnx.export` 只留一条 trace 路径，把这个 python `if` **固化成常量**，并抛
  `TracerWarning: ...vision_transformer.py:694 if len(idx) > 0`；
- 因此 ONNX 里出现的 block 就是“导出那次 dummy 输入恰好命中的集合”，**不是** 12 个，也**不是**裁剪。

### 实测：这个开关本身是“没训出来”的（本 checkpoint `AVTrack_ep0300.pth.tar`）

逐块查 `active_score_module` 的权重（`strict=True` 加载后）：

| block | ‖w‖ | bias | sigmoid(bias) | 实际行为 |
|---|---|---|---|---|
| 0、1 | 0.34 | 0 | 0.5 | 无条件跑（代码保证） |
| **2** | **2.0e-4** | +5.77e-4 | 0.500144 | 输入无关常数 → **永远跑**，余量仅 1.4e-4 |
| **3** | 7.5e-3 | −9.92e-4 | 0.499752 | 近似常数 → **永远跳**，余量 2.5e-4 |
| **4** | 1.1e-1 | −4.30e-3 | 0.498926 | 输入相关，**随帧翻转** |
| **5** | 1.0e-1 | +4.71e-3 | 0.501177 | 输入相关，**随帧翻转** |
| 6–10 | 1.1–8.3 | +0.13~+1.20 | 0.53~0.77 | 恒跑 |
| **11** | 1.1 | −0.033 | — | logit 深度饱和，13 组输入概率全部 <1e-6 → **永远跳** |

结论：**“跑不跑 2/3/4/5” 由 1e-4 量级的数值噪声决定**（训练时的 `activeness_loss` 显然没压住这几块）。
训练是逐样本随机子集，所以模型对子集**有鲁棒性**——这解释了两个冻结图“都能跟踪，却又给出不同的框”。
**因此部署侧必须自己钉一张确定性的表**，这件事没有“正确答案”，只有“选定的语义”。

### 实证：同一 checkpoint 的两张图，跟踪结果不同

| 产物 | producer | opset | 冻结 block | md5 |
|---|---|---|---|---|
| `AVTrack/avtrack_deit_tiny_patch16.onnx`（仓库根目录那份） | pytorch 2.0.1 | 11 | `0,1,2,`**`5`**`,6,7,8,9,10` | `c431f59dca5a747e39122da9665782dc` |
| `AVTrack/avtrack_deit_tiny_fp16.engine`（对应上面这份） | — | — | 同上 | `de24a9a9d78d0e88f3ec4239ed1979fb` |
| `PyTracking_ONNX-TensorRT/exports/avtrack.onnx` | pytorch 2.3.1 | 17 | `0,1,2,`**`4`**`,6,7,8,9,10` | `a329d66d98461517c5d8311de8f003b4` |
| `PyTracking_ONNX-TensorRT/exports/avtrack_fp16.engine` | — | — | 同上 | `9ddb2926f2471d9ece07b9437b6a9cb7` |

两套 engine 在 `man.mp4` 上各跑 299 帧（同初始框 `[80,30,145,190]`）：

| 指标 | 实测 |
|---|---|
| 逐帧框 IoU | 均值 **0.913**，p05 0.748，最小 0.464，**82/299 帧 < 0.9** |
| 框中心距离 | 均值 3.2 px，p95 8.5 px，**最大 37.1 px** |
| 单列最大差 | x 23.4 / y 20.5 / **w 32.0** / h 14.0 px |
| 真实帧上两图输出差（`crop_sz=664`） | score 6.6e-2 / size 1.5e-1 / **offset 3.0e-1**（argmax 位置一致） |

> 这两套 CSV 是最好的“语义敏感度上界”样本：
> `PyTracking_ONNX-TensorRT/results/avtrack.csv`（5 号块）与 `results_exported/avtrack.csv`（4 号块），表头 `frame,x,y,w,h,conf,ms`。

### v2 的裁决：钉死 `{跳 3,4,11}`（跑 `0,1,2,5,6,7,8,9,10`）

理由：① 等于训练时的**期望**路径（平均 9 块：2 跑、3 跳、4/5 取一、11 跳）；② 恰好与仓库里现有 ONNX/TRT engine 同语义，历史 TRT 验证（含 §0.2 的数字）可直接复用；③ 计算量最小。
实现方式**不改上游**：把每块的 `active_score_module.forward` 换成常量（logit ±50），见 §4 Step 1 的 `freeze_gate()`。参数不改动，`state_dict` 仍严格匹配。
**如果用户要求“全 12 块确定性”，才改用 `is_distill=True`**（+33% 计算量，且不再是 eval 语义），属于用户决策。

---

## 0.3 风险表

| 项 | 结论 |
|---|---|
| 结构 | DeiT-Tiny(12 block, embed 192, 3 heads) + CENTER head；joint attention `1×3×320×320`（template 64 + search 256 token，无 cls） |
| 有效深度（冻结后） | **9 块**（`0,1,2,5,6,7,8,9,10`），attention 9 个，LN 19 个 |
| 输入 | `template (1,3,128,128)`、`search (1,3,256,256)`，**全静态、无动态轴** |
| 输出 | `score_map (1,1,16,16)`、`size_map (1,2,16,16)`、`offset_map (1,2,16,16)` |
| 算子（op11） | **582 节点**：MatMul 54 / Add 103 / Mul 46 / ReduceMean 38 / Softmax 9 / Erf 9 / Conv 17 / Split 9 …（与仓库里那份 ONNX 节点数完全一致，说明 SDPA 补丁写法一致） |
| 算子（op17） | **约 429 节点**，19 个 `LayerNormalization` + 9 Softmax + 54 MatMul（GELU 仍是 Erf 组合，ONNX `Gelu` 要 opset 20，不要升） |
| ⚠️ 节点数不是硬指标 | 若把 SDPA 补丁里的 `1/math.sqrt(d)` 写成 `d ** 0.5`，会多出 9 组 `Pow`/`Reciprocal`/`Constant` 常量节点（op17 429→474、op11 582→627），**数值等价**。验收看 `LN=19 / Softmax=9 / MatMul=54` 和 block 集合，不看总数 |
| 高风险算子 | **没有** 5D/6D、没有 GridSample、没有动态 shape |
| 主要缺点 | 19 个 LN 在 op11 下碎成 ReduceMean/Sub/Pow/Sqrt/Div；320×320 的 Softmax 有 9 个（NPU 上的碎算子风险点） |
| RKNN 未知 | op17 的 `LayerNormalization` 是否上 NPU（TRT 通过 ≠ RKNN 通过）；FP16 精度；INT8；三核吞吐 |
| 已有产物 | ONNX（op11，见上表）、TRT engine、`inference_avtrack_trt.py`（预处理/后处理可直接抄）、早期 TensorRT 项目的 `export_wsl.py`（导出器骨架，未随本仓库分发） |
| checkpoint | `output/checkpoints/train/avtrack/deit_tiny_patch16_224/AVTrack_ep0300.pth.tar`（249 MiB / 261 MB，epoch 300，`strict=True` 全匹配） |
| 数据集 | **无 GT 数据集**；有 4 段视频 + 初始 ROI（§1.4） |
| 依赖 | rk25 环境原本缺 `easydict`、`tensorboardX`（本机 2026-11-16 已装好；换机器要按 §1.2 重装） |

---

## 0.4 已经排除的“假问题”（不要再花时间）

1. **权重没漏**：`load_state_dict(..., strict=True)` 通过（0 missing / 0 unexpected）。
2. **不是别的权重版本**：现有 op11 ONNX 与当前 checkpoint 数值等价 —— ORT vs PyTorch `max|d|` = 1.5e-7 / 6.0e-7 / 6.2e-6（**前提：gate 表一致**）。
3. **没有动态 batch 维**：`Reshape` 里没有 0 值；输入输出 shape 全静态。不要在 head 里改 `.view(-1, ...)`。
4. **不需要 TRT 数值参考**：TRT 只能给你“同语义下 FP16 允许多大误差”的量级，数值参考用 ORT。
5. **`export_onnx.py` 的 `.cuda()` 不是必需**：CPU 导出实测 0.4–0.8 s 就能出图。

---

## 0.5 首轮执行实测结论（2026-11-17，已跑完全流程）

执行产物与全部原始数字在 `/path/to/AVTrack-RK3588/`（`README.md` + `results/*.md/json`）。以下几条**修正了本文档的假设**，后续按这里执行：

| # | 文档原假设 | 实测结论 |
|---|---|---|
| 1 | op17 的 `LayerNormalization` 可能不被 RKNN 支持，要准备 op11 后备 | **完全支持**：19 个 LN 融合成 32 个 `exNorm`；op17 / op11 / opt3 三种构建**逐位等价**（误差、延迟、体积全同）→ 主用 op17 即可，op11 只当保险 |
| 2 | 板端 FP16 允许误差“比 simulator 略大” | 实际**大 5–10 倍**：score 9.98e-3 / size 6.29e-3 / offset 5.35e-2。**板端判据应改为**：`score < 2e-2`、`size < 2e-2`、`offset < 1e-1`，且**以解码后误差为准**（实测 dx 0.63 px、w 1.90 px，argmax 20/20 相同） |
| 3 | 单实例绑单核吞吐 > 单实例用三核（RT-DETR 经验） | **本模型相反**：1×core0 = 25.1 FPS，1×三核 = 28.3 FPS，**3 进程各绑一核 = 63.9 FPS**（单实例延迟 39.9→44–50 ms）。要吞吐就上多进程 |
| 4 | 可以测 1/2/3/4 实例 | RK3588 **只有 3 个 NPU core**，第 4 个进程无法绑核（`core=3` 直接退出）；并行上限就是 3 路 |
| 5 | 板端 vs PyTorch 的框误差可以直接比 | **必须同像素**。板端/PC 解码同一帧就差 ±2 灰度级（61–63% 通道不同），而本模型对 ±2 噪声能把逐帧 IoU 从 1.000 打到 0.60–0.86。同像素基准下 RKNN vs ORT 逐帧 IoU = **0.9971–0.9978**、中心 0.09–0.54 px |
| 6 | 精度还有 FP32 可退 | **没有**：`RKNN.config(float_dtype=...)` 只接受 `float16`，非量化最高精度就是 FP16 |
| 7 | ——（新增） | 与历史 TRT 产物一致性已验证：我们 gate9 RKNN vs 旧 5-block TRT，IoU **0.9717**、中心 2.19 px（同语义换后端）；vs 旧 4-block TRT 为 0.9449（差别来自冻结表不同） |
| 8 | ——（新增） | street 视频有 8/300 帧 argmax 翻转（≈2.7%），翻转帧会跳一次；这是 score 峰值接近平局的模型特性，不是 NPU 缺陷 |

**可直接复用的脚本**（别重写）：`AVTrack-RK3588/scripts/` 下 `avtrack_common.py`（冻结 gate + 前后处理唯一实现）、
`export_avtrack.py`、`make_ref.py`、`build_and_sim.py`、`run_board.py`、`bench.py`、`track_headless.py`、
`dump_crops.py` + `run_crops.py`（同像素公平对照）、`compare_tracks.py`。

> 踩过的坑：`bench.py` 早期版本把整段视频预读进内存，8000 帧长稳会吃 2 GB 内存 —— 已改为流式读帧。
> 本机 WSL 的 `/tmp` 不跨 `wsl -e bash -c` 调用保留，中间产物别写 `/tmp`。

---

## 1. 环境

### 1.1 构建侧（WSL2 Ubuntu-22.04）

| 项 | 值 |
|---|---|
| Python | `python3` |
| 版本（实测） | Python 3.10、torch 2.4.0+cu121（CUDA 可用）、onnx 1.18.0、onnxruntime 1.23.2、rknn-toolkit2 2.5.0、timm 1.0.31、opencv 4.11 |
| 代码（Windows） | `C:\path\to\AVTrack` |
| 代码（WSL） | `/path/to/AVTrack` |
| 工作目录（WSL） | `/path/to/AVTrack-RK3588`（**仓库外**，见 §1.5） |

自检：

```bash
PY=python3
$PY -c "import torch,onnx,onnxruntime;print(torch.__version__,onnx.__version__,onnxruntime.__version__,torch.cuda.is_available())"
$PY -c "from rknn.api import RKNN; print('rknn ok')"
```

> WSL 里没有 `/usr/lib/librknnrt.so` 是**正常的**（运行时库在板端）。simulator 不需要它。

### 1.2 必须补的两个依赖（否则 Step 1 直接失败；本机已装好，换机器要重装）

```bash
PIP=pip
$PIP install easydict            # lib/config/avtrack/config.py:1 需要
$PIP install tensorboardX        # torch.load 反序列化 lib.train.admin.stats 需要
                                 # 或 install tensorboard，二选一即可
```

验证：

```bash
cd /path/to/AVTrack
$PY -c "import easydict; from lib.train.admin.stats import AverageMeter; print('deps ok')"
```

装完后第一次 import 会打印一行 `WARNING: You are using tensorboardX instead sis you have a too old pytorch version.`，
这是因为 `lib/train/admin/tensorboard.py` 优先 `torch.utils.tensorboard`（要另一个 `tensorboard` 包）失败后回退到 tensorboardX——**属于正常**，不影响加载 ckpt。

补充说明：
- `timm 1.0.31` 与 `requirements.txt` 里的 `0.9.10` 不同，但实测建模型/导出均正常，**不要降级**（降了反而可能踩 `timm.layers` 迁移问题）。
- `build_avtrack()` 内部 `deit_tiny_patch16_224(pretrained=True)` 会**联网下载** ~21 MB DeiT 权重（`dl.fbaipublicfiles.com`）。导出脚本里必须用 monkeypatch 关掉（§4 Step 1），否则离线环境报错位置很迷惑。

### 1.3 板端

| 项 | 值 |
|---|---|
| IP / 用户 | `192.168.x.x` / `cat` |
| Python | `/home/user/miniconda3/envs/rknn/bin/python`（3.8 + rknn-toolkit-lite2 2.5.0） |
| 运行时 | `/usr/lib/librknnrt.so` 2.5.0 |
| 注意 | **密码问用户要，不要写进任何文件、脚本、commit** |

```bash
ssh user@board-ip
ls -l /usr/lib/librknnrt.so
/home/user/miniconda3/envs/rknn/bin/python -c "from rknnlite.api import RKNNLite; print('lite ok')"
free -m; nproc; cat /sys/class/thermal/thermal_zone0/temp
```

板端**不要**用 `/home/user/opencl_shim`（那是 GridSample 用的，本模型不需要）。

### 1.4 素材（本机已有，不要再问用户要视频）

```
/path/to/videos/
  man.mp4        (340×256)  man.txt        -> helmet 143 28 64 40
  cars.mp4                  cars.txt       -> black_car 520 260 45 32
  street.mp4                street.txt     -> left_guy 255 143 37 95
  basketball.mp4            basketball.txt -> basket 1010 1600 436 740
```

`.txt` 格式：`name x y w h`（首帧左上角 + 宽高）。
**注意：没有 GT 逐帧框**，所以只能做“两两对比 + 看视频”的弱评测（§4 Step 7）；强评测（SR/PR）仍需用户提供 UAV123/UAVDT/DTB70 等数据集。

### 1.5 工作目录约定（仓库外，别污染上游）

```
C:\path\to\AVTrack-RK3588\        # WSL: /path/to/AVTrack-RK3588
  onnx\      # 冻结后重导出的 onnx
  build\     # rknn 产物
  ref\       # 固定输入 + 冻结语义参考输出（npz）
  scripts\   # 所有脚本（可一键重跑）
  logs\      # 构建/运行日志
  results\   # 延迟、精度、跟踪 CSV
  patches\   # 只在确实需要改上游时才放 patch
```

板端对应 `/home/user/rknn_avtrack/`，结构保持一致。

> 实操提醒：本机 WSL 的 `/tmp` **不会跨 `wsl -e bash -c ...` 调用保留**（每次调用可能是新实例）。
> 中间产物一律写进工作目录（`$WORK/...`），别写 `/tmp`，否则下次调用就找不到了。

> 上游仓库现状（动手前先知道）：`avtrack.py`、`lib/test/evaluation/local.py`、`lib/train/admin/local.py` 已是 M 状态；
> 100+ 个 tracked `__pycache__/*.pyc` 常年 dirty；`*.onnx`、`*.engine`、`export_onnx.py`、`inference_avtrack_trt.py`、`output/` 都未跟踪。
> 所以：**不要指望 `git apply` 干净回退**，改动一律走“仓库外脚本 + monkeypatch”。

---

## 2. 先读这 7 个文件（约 30 分钟）

1. `lib/models/avtrack/vision_transformer.py:678-709`
   - `forward_features` 里**动态跳块**（本手册的核心风险点）；`pos_embed_z=(1,64,C)`、`pos_embed_x=(1,256,C)` 写死在 `__init__`。
2. `lib/models/avtrack/avtrack.py`
   - `forward()` 把 template/search 一起丢进 backbone；
   - 第 96–105 行有 debug print（PC 侧会刷屏；导出脚本用 `redirect_stdout` 屏蔽，**不必改上游**）；
   - `forward_head()` 用 `cat_feature[:, -256:]` 取 search 特征 → 证实是 joint attention。
3. `lib/models/layers/head.py` — `CenterPredictor`（3 个分支 × 5 层 Conv，`_sigmoid` 只作用于 score/size，offset 无激活）。
4. `export_onnx.py` — SDPA monkeypatch、输入名 `template`/`search`、opset 11。
5. 早期 TensorRT 项目的 `export_wsl.py`（未随本仓库分发）
   - **v2 导出器的骨架**：CPU 导出、`pretrained=False` patch、`strict=True`、opset 17、ORT `assert_allclose`、写 json 报告。照抄它的结构即可。
6. `inference_avtrack_trt.py` — 预处理/后处理/状态更新**直接复用**（`crop_hwc` / 汉宁窗 / argmax / `0.8*old+0.2*new` / `conf>0.15`）。
7. `experiments/avtrack/deit_tiny_patch16_224.yaml` — 128/256、factor 2.0/4.0、CENTER head、ImageNet mean/std。

> 早期 TRT 项目的 `run_headless.py` 是**无 GUI 的跟踪 runner**（显式 ROI、CSV+JSON 证据、warmup 10 帧、可选 `--compare-onnx`）；本仓库对应 `scripts/track_headless.py`。
> Step 6 的板端 runner 直接照它的结构改，后端从 TRT 换成 RKNNLite（板子通常没显示器，`cv2.selectROI` 用不了）。

读完写一页“事实清单”：冻结表、token 数、输出含义、预处理顺序（BGR→RGB、/255、mean/std）、后处理公式。后面所有调试都以此为准。

---

## 3. 工作纪律（从 RT-DETR v4.x 继承，别省）

1. **单变量**：一次只改一个东西，并且同时留 A/B 两个模型。
2. **先闭环**：PC 构建 → 传板 → 推理 → 和参考对比，这一圈不通不优化。
3. **板端数字才算数**：simulator 只用来筛明显错误。
4. **每个输出都打 `min/max/nonzero`**：全 0 或常数立刻停（见 §5 教训 1）。
5. **失败留证据**：现象、最小复现、已否定假设、日志路径。
6. **没用户点头不 commit / 不 push**。
7. **不改上游仓库**：需要的改动一律用仓库外脚本 monkeypatch；万不得已才放 `patches/` 并用 `git apply`。
8. **【新增】任何比较都要先声明 gate 表**：写清“本次对比的两边，跳块表分别是 `{...}`”。没写清的对比结果作废。
9. **【新增】参考实现必须是冻结语义**：PyTorch 参考要跑 `freeze_gate()`，否则它每帧可能换图。

### 每轮迭代的固定循环

```text
写下一个可证伪的假设
  → 设计一次只改一个变量的最小实验（争取 < 30 分钟）
  → PC 构建 / 板端实测
  → 和参考比（数值、延迟、跟踪指标）
  → 记录到 results/（含失败）
  → 决策：继续 / 换路 / 回滚
```

同一个假设连续两次实验都不成立，就换假设，不要靠调参硬救。

---

## 4. 步骤

### Step 0：环境自检（15 分钟）

- 跑 §1.1、§1.2、§1.3 的命令，全部通过再继续；
- 确认板子空闲（`thermal_zone0/temp`、`free -m`）与磁盘（WSL `df -h /`、板端 `df -h /home`，各留 ≥2 GB）；
- 建目录：`mkdir -p /path/to/AVTrack-RK3588/{onnx,build,ref,scripts,logs,results}`。

**验收**：两边 rknn 都是 2.5.0；`import easydict` 与 `from lib.train.admin.stats import AverageMeter` 都通过。

---

### Step 1：冻结 gate + 重新导出（半天）

> **本节 recipe 已于 2026-11-16 在 rk25 环境端到端跑通**：`strict=True` 全匹配 → `freeze_gate()` → op17/op11 各导一份 →
> block 集合 `{0,1,2,5,6,7,8,9,10}` → 灰图/randn 两次导出的输出差 **0.0** → ORT vs PyTorch ≤ 6e-6 →
> op17 vs op11 ≤ 3.3e-6。照下面的写法即可复现，不需要再自己试。

#### 1.1 冻结函数（已并入 `scripts/avtrack_common.py` 的 `freeze_gate()`，不改上游、参数不动）

```python
"""把 AVTrack 的数据相关跳块 gate 钉成常量。

RUN_BLOCKS/SKIP_BLOCKS 是部署语义的一部分（见 DEPLOY_RK3588 §0.2）：
  - 实测 block 11 永远跳（概率 <1e-6）；
  - block 2/3 的 gate 是 0.500144 / 0.499752 的准常数（权重范数 2e-4 / 7.5e-3）；
  - block 4/5 随帧翻转（0.4989 / 0.5012），钉哪个都是“选一个语义”。
选 {跳 3,4,11}：等于训练的期望路径（平均 9 块），且与仓库现有 ONNX/TRT engine 同语义。
"""
import torch

RUN_BLOCKS = (0, 1, 2, 5, 6, 7, 8, 9, 10)
SKIP_BLOCKS = (3, 4, 11)
LOGIT = 50.0          # sigmoid(±50) ≈ 1 / 0，足够远离 0.5


def freeze_gate(model, run_blocks=RUN_BLOCKS, logit=LOGIT):
    blocks = model.backbone.blocks
    assert len(blocks) == 12, f'expected 12 blocks, got {len(blocks)}'
    assert set(run_blocks) | set(SKIP_BLOCKS) == set(range(len(blocks)))
    for i, blk in enumerate(blocks):
        lin = blk.active_score_module            # 保留参数，只换 forward
        const = logit if i in run_blocks else -logit
        def fwd(x, _c=const, _lin=lin):
            return torch.full((x.shape[0], 1), _c,
                              dtype=_lin.weight.dtype, device=x.device)
        lin.forward = fwd
    return model
```

#### 1.2 `scripts/export_avtrack.py`（结构照抄早期 TensorRT 项目的 `export_wsl.py`，加冻结核对）

关键点（完整脚本按这些要求写，别省）：
- `sys.path.insert(0, source)`；`torch.manual_seed(0)`；`torch.set_num_threads(4)`；
- `config.update_config_from_file(cfg_yaml)` ← **必须**，否则默认配置是 `vit_base_patch16_224` → `build_avtrack` 抛 `NotImplementedError`；
- **关掉联网下载**：
  ```python
  module = importlib.import_module('lib.models.avtrack.avtrack')
  orig = module.deit_tiny_patch16_224
  module.deit_tiny_patch16_224 = lambda *a, **k: orig(*a, **{**k, 'pretrained': False})
  ```
- **严格加载权重**：`model.load_state_dict(state.get('net', state), strict=True)`；
- **冻结 gate**：`freeze_gate(model)`，再 `model.eval()`；
- SDPA monkeypatch（torch 2.4 仍需要，**直接用 `export_onnx.py` 里那段**：`1 / math.sqrt(query.size(-1))`，别改写成 `** 0.5`，否则节点数会多 45 个常量节点）；
- wrapper 只返回 3 路输出：
  ```python
  def forward(self, template, search):
      anno = template.new_zeros((template.shape[0], 4))
      out = self.net(template, search, anno, anno)
      return out['score_map'], out['size_map'], out['offset_map']
  ```
- 导出用 `contextlib.redirect_stdout(io.StringIO())` 把 debug print 吞掉；
- `torch.onnx.export(..., opset_version=opset, do_constant_folding=True,
  input_names=['template','search'], output_names=['score_map','size_map','offset_map'])`；
- **导出后立刻断言 block 集合**（解析 ONNX 里 `blocks.N.` 的 tensor/node 名）：
  ```python
  assert block_set(out) == set(RUN_BLOCKS), f'gate 未冻结: {sorted(block_set(out))}'
  ```
- **两次不同 dummy 导出结果必须一致**（防“又漂回去了”）：
  用 `torch.randn`、灰图各导一次，比 block 集合与 ORT 输出 `max|d| < 1e-6`；
- ORT 自检 + 写 json 报告（`opset`、block 集合、节点数、3 路 `max|d|`、`md5`）。

命令：

```bash
cd /path/to/AVTrack && PY=python3
$PY /path/to/AVTrack-RK3588/scripts/export_avtrack.py \
  --source /path/to/AVTrack \
  --config experiments/avtrack/deit_tiny_patch16_224.yaml \
  --checkpoint output/checkpoints/train/avtrack/deit_tiny_patch16_224/AVTrack_ep0300.pth.tar \
  --opset 17 --output /path/to/AVTrack-RK3588/onnx/avtrack_gate9_op17.onnx

# 再导一份 op11 作后备（RKNN 若不认 LayerNormalization 就换它）
$PY /path/to/AVTrack-RK3588/scripts/export_avtrack.py \
  --source /path/to/AVTrack \
  --config experiments/avtrack/deit_tiny_patch16_224.yaml \
  --checkpoint output/checkpoints/train/avtrack/deit_tiny_patch16_224/AVTrack_ep0300.pth.tar \
  --opset 11 --output /path/to/AVTrack-RK3588/onnx/avtrack_gate9_op11.onnx
```

命名带语义：`avtrack_gate9_op17.onnx`（gate9 = 9 块冻结表），别再用含义模糊的 `avtrack_op17.onnx`。

**预期数字（已在 rk25 环境实测，用来判断“这次导出是否正常”）**：

| 检查项 | 期望 |
|---|---|
| checkpoint 加载 | `strict=True` 通过：`<All keys matched successfully>`（net 里 272 个 tensor） |
| block 集合 | `{0,1,2,5,6,7,8,9,10}`（**不是 12 个，也不是 `{...4...}`**） |
| 关键算子计数 | op17：`LayerNormalization=19`、`Softmax=9`、`MatMul=54`；op11：`Softmax=9`、`MatMul=54`、`ReduceMean=38` |
| 节点总数 | op17 ≈ **429**、op11 = **582**（SDPA 写成 `**0.5` 时是 474 / 627，数值等价，别当故障） |
| 输入/输出 shape | 全静态，`[1,3,128,128]` / `[1,3,256,256]` → `[1,1,16,16]` / `[1,2,16,16]` ×2 |
| ORT vs PyTorch（冻结后） | `max|d|` ≤ 1e-5（实测 2e-7 / 5e-7 / 6e-6） |
| 两次 dummy（灰图 vs randn）导出 | block 集合相同，同一输入上输出差 **= 0.0** |
| op11 vs op17 | 互相 `max|d|` ≤ 1e-5（实测 1.2e-7 / 3.0e-7 / 3.3e-6） |
| 导出耗时（CPU） | < 2 s |

**失败处理**：
- `NotImplementedError` → 忘了 `update_config_from_file`；
- `ModuleNotFoundError: easydict / tensorboardX` → §1.2；
- 卡在 `Downloading ... deit_tiny_patch16_224` → 没打 `pretrained=False` patch；
- block 集合不是 9 个或不是 `{0,1,2,5,...}` → `freeze_gate` 没生效（先确认 `len(model.backbone.blocks)==12`，再确认 `freeze_gate(model)` 在 `model.eval()` 之前或之后就调用过，且导出用的就是那个 model）；
- op17 导出报错 → 直接用 op11 继续 Step 2–4，把 LN 开销记进延迟表，别为它停进度。

---

### Step 2：建金标准（半天）

目的：后面每一次转换、量化，都跟这份参考比。**参考语义 = 同一个 gate9 冻结表。**

#### 2.1 固定输入对（用 §1.4 的视频，20 组；脚本名见 §8 的 `make_ref.py`）

每组 `(template, search)` 从同一段视频取：
- 首帧按 ROI 取 template（factor 2.0 → 128×128）；
- 第 0/25%/50%/75%/末帧取 search（factor 4.0 → 256×256）；
- 覆盖 4 段视频 × 小/中/大目标 + 4~5 帧位置 → 20 组，明显边界情况（目标贴边、跨出画面）各留 2 组。

```python
np.savez('/path/to/AVTrack-RK3588/ref/pairs.npz',
         template=tpl_u8,   # (N,128,128,3) uint8, RGB
         search=src_u8,     # (N,256,256,3) uint8, RGB
         meta=json.dumps(meta))   # video, frame_idx, bbox, crop_sz
```

> 存 **uint8 RGB**，归一化留给下游脚本做——保证 PC/板端走同一份预处理代码。
> `crop_hwc` 与 TRT 脚本完全一致（`crop_sz=ceil(sqrt(w*h)*factor)`，`BORDER_CONSTANT` 补 0，再 `cv2.resize`）。

#### 2.2 冻结语义的参考输出

```python
x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD     # MEAN/STD 见 yaml
x = x.transpose(2, 0, 1)[None]
```

- **PyTorch 参考**：`freeze_gate(model)` + `model.eval()`，存 `ref/out_torch.npz`；
- **ORT 参考**：跑 gate9 ONNX，存 `ref/out_ort.npz`；
- 每路都打印 `min/max/nonzero`。

**验收**：
- `torch` vs `ort` 三路 `max|d| < 1e-5`；
- `score_map` ∈ (0,1)、`size_map` ∈ (0,1)、`offset_map` 不是常数（实测典型范围 −1.5 ~ +2.5）；
- 若某组输入上 `score_map` 最大值 < 0.05，说明这组 crop 基本没目标，换一组（别用它当参考）。

> **不要拿现有 TRT engine 当唯一参考**：它对应 `{...5...}` 表；和你导出的 gate9 表一致时可以互相印证，
> 但数值参考一律用 ORT。

---

### Step 3：FP16 转换（半天）

`scripts/build_fp16.py`：

```python
import argparse, os
from rknn.api import RKNN

p = argparse.ArgumentParser()
p.add_argument('--onnx', required=True)
p.add_argument('--out', required=True)
p.add_argument('--opt-level', type=int, default=0)
p.add_argument('--quant', action='store_true')
p.add_argument('--dataset', default=None)
p.add_argument('--u8-input', action='store_true')
a = p.parse_args()

rknn = RKNN(verbose=True)
cfg = dict(target_platform='rk3588', optimization_level=a.opt_level)
if a.u8_input:      # 第二版再试，先跑 float32
    cfg['mean_values'] = [[123.675, 116.28, 103.53]] * 2
    cfg['std_values'] = [[58.395, 57.12, 57.375]] * 2
rknn.config(**cfg)
assert rknn.load_onnx(model=a.onnx,
                      inputs=['template', 'search'],
                      input_size_list=[[1, 3, 128, 128], [1, 3, 256, 256]]) == 0
assert rknn.build(do_quantization=a.quant, dataset=a.dataset) == 0
assert rknn.export_rknn(a.out) == 0
rknn.release()
print('OK', a.out, os.path.getsize(a.out))
```

```bash
$PY /path/to/AVTrack-RK3588/scripts/build_fp16.py \
  --onnx /path/to/AVTrack-RK3588/onnx/avtrack_gate9_op17.onnx \
  --out  /path/to/AVTrack-RK3588/build/avtrack_gate9_op17_fp16.rknn
```

**第一版先不要加 `mean_values`/`std_values`**：直接喂 float32（已归一化）数据，先把数值打通；第二版再试 u8 + mean/std 做单变量对比。这样出错能立刻分辨是“模型问题”还是“预处理问题”。

#### 检查点 A：op 分配

在 `rknn.build` 日志里 grep `Unsupported` / `CPU`，重点看 **Softmax（9 个 320×320）**、**LayerNormalization（19 个，op17）**、**MatMul（54 个）**。
- op17 若有 LN 回退 CPU → 换 op11（LN 被拆成原语）再比一次延迟/精度，**一次只换这一个变量**；
- 任何“回到 CPU”的算子都要记进 `results/latency.md`。

#### 检查点 B：simulator 精度

```python
rknn.load_rknn(path); rknn.init_runtime()
outs = rknn.inference(inputs=[tpl_f32_nchw, src_f32_nchw])   # 用 ref/pairs.npz 的 20 组
```

> 注意：`load_rknn` 加载的模型**不能在 simulator 上推理**（会报 “not support inference on the simulator”）。
> 必须在 `build()` 的同一个会话里 `init_runtime()` —— 用 `scripts/build_and_sim.py`（已封装）。

对照 `ref/out_ort.npz`：

| 输出 | 判据（FP16 simulator） | 参照（TRT FP16 实测） | 本机 simulator 实测 |
|---|---|---|---|
| `score_map` | `max\|d\| < 2e-3` | 3.6e-4 | **1.00e-3** |
| `size_map` | `max\|d\| < 5e-3` | 1.1e-3 ~ 1.5e-3 | **5.10e-4** |
| `offset_map` | `max\|d\| < 6e-3` | 3.9e-3 ~ 5.7e-3 | **5.36e-3** |

> 阈值来自 `PyTracking_ONNX-TensorRT/results*/avtrack.json` 的 `onnx_fp32_vs_trt_fp16`，不是拍脑袋。
> 比它明显大（比如 `offset` 差 > 5e-2）就先查输入顺序/RGB/mean-std，不要急着怀疑 NPU。

**常见失败原因（按概率）**：① 输入顺序反了；② RGB/BGR 反了；③ mean/std 做了两次；④ 输出顺序不是 `score,size,offset`。

**验收**：日志无 `<CPU>` 回退（或有回退但已记录并量化）；simulator 三路达标。

---

### Step 4：板端单帧打通（半天）

`scripts/run_board.py`（结构照早期 TRT 项目的 `run_headless.py`，后端换 RKNNLite）：

```python
import numpy as np
from rknnlite.api import RKNNLite

MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
STD  = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)

def prep(u8):                      # u8 RGB HWC -> f32 NCHW（与 PC 完全同一份公式）
    x = u8.astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None]
    return np.ascontiguousarray((x - MEAN) / STD)

d = np.load('/home/user/rknn_avtrack/ref/pairs.npz')
rk = RKNNLite()
assert rk.load_rknn('/home/user/rknn_avtrack/build/avtrack_gate9_op17_fp16.rknn') == 0
assert rk.init_runtime(core_mask=RKNNLite.NPU_CORE_0) == 0
for i in range(len(d['template'])):
    outs = rk.inference(inputs=[prep(d['template'][i]), prep(d['search'][i])],
                        data_format='nchw')
    for name, o in zip(('score', 'size', 'offset'), outs):
        a = np.asarray(o)
        print(f'[{i:02d}] {name:7s} shape={a.shape} min={a.min():.6f} '
              f'max={a.max():.6f} nonzero={np.count_nonzero(a)}/{a.size}')
rk.release()
```

**必须做的四件事**：
1. 三路输出都打 `min/max/nonzero`；
2. 和 `ref/out_ort.npz` 比 `max|d|`。**板端判据（实测校准）**：`score < 2e-2`、`size < 2e-2`、`offset < 1e-1`，
   并且**以解码后误差为准**：`dx/dy < 2 px`、`w/h < 3 px`、`conf < 2e-2`、argmax 20/20 相同。
   （本机实测：`9.98e-3 / 6.29e-3 / 5.35e-2`；解码后 `dx 0.63 / dy 0.25 / w 1.90 / h 0.95 px`、conf `1.0e-2`、argmax 20/20）
3. 连续跑 100 帧不崩、不泄漏（前后 `cat /proc/meminfo`）；
4. 记录 `init_runtime` 耗时与首次 `inference` 耗时（首帧含预热）。

**失败处理**：
- 全 0 / 常数 → 见 §5 教训 1，先怀疑编译路径，别先改预处理；
- shape 不对 → 检查 `data_format='nchw'`、inputs 顺序；
- 误差大但趋势对 → 大概率 mean/std 做了两次；
- `load_rknn` 失败 → 检查板端 `librknnrt` 版本是否 2.5.0（toolkit 与 runtime 版本必须匹配）。

**验收**：板端三路输出与 ORT 同量级，100 帧稳定。

---

### Step 5：延迟拆解 + 多实例（1 天）

`scripts/bench.py` 要求：

```python
# 预热 30 帧，正式测 300 帧，分别记录：
#   t_pre   = crop + resize + 归一化
#   t_npu   = rk.inference()
#   t_post  = hann + argmax + 状态更新
# 输出 p50 / p90 / FPS
```

两次测量：
- `rk.inference(inputs=[...])`：含 H2D/D2H（rknnlite 常规路径）；
- 若拷贝占比 > 20%，再上 C++（`rknn_inputs_set` + `rknn_run`）单独测 NPU 时间。rknnlite 阶段不为这件事浪费时间。

多实例：

```bash
python bench.py --core 0 --frames 300 &
python bench.py --core 1 --frames 300 &
python bench.py --core 2 --frames 300 &
```

再对比 `core_mask=NPU_CORE_0_1_2` 的单进程。

**已知经验（来自 RT-DETR v4.1）**：单实例绑单核、多实例并行总吞吐 > 单实例用三核。
**但本模型实测相反**（见 §0.5 #3）：1×core0 = 25.1 FPS，1×三核 = 28.3 FPS，
**3 进程各绑一核 = 63.9 FPS 合计**（单实例延迟 44–50 ms）。要吞吐就上多实例。

**参考量级**：同模型 TRT FP16 在桌面 GPU 上含预处理/后处理约 2.8–3.3 ms/帧（300+ FPS）。
RK3588 实测单核 39.9 ms/帧（pre 2.69 / npu 37.13 / post 0.09），三核 35.4 ms。
**注意**：`bench.py` 必须流式读帧（早前版本预读整段视频，8000 帧长稳会吃 2 GB 内存）。

**验收**：给出 `1/2/3` 实例的 FPS 表 + 单帧三段耗时表（RK3588 只有 3 个 NPU core，第 4 路无法绑核）。

---

### Step 6：跟踪链路（1–2 天）

直接移植 `inference_avtrack_trt.py`，只换推理后端；**runner 用无 GUI 版本**（板端没显示器，`cv2.selectROI` 不可用），结构照早期 TRT 项目的 `run_headless.py`：显式 `--bbox`、CSV（`frame,x,y,w,h,conf,ms`）+ JSON、warmup 10 帧。

| 模块 | 处理 |
|---|---|
| `crop_hwc` | 原样保留；bbox 是 `[x, y, w, h]`，`crop_sz = ceil(sqrt(w*h)*factor)` |
| 模板 | 首帧 crop 一次（factor 2.0 → 128×128）；不更新 |
| 搜索区 | 每帧 crop（factor 4.0 → 256×256） |
| 后处理 | `score.reshape(16,16)` → 汉宁窗 → `argmax` → 取 size/offset；`conf` 取窗口惩罚前的原值 |
| 坐标还原 | `cx_rel=(x_idx+off_x)/16`，`w_rel=size[0]`（Ch0=W、Ch1=H）；再乘 `search_size` 与 `crop_sz/search_size` |
| 状态更新 | `cx += dx`；尺寸 `0.8*old + 0.2*new` |
| 置信度 | `conf > 0.15` 才更新 |
| 越界 | 宽高不超过画面，中心限制在画面内 |

**关键点**：joint attention 意味着 template 每帧都参与计算，**没有“模板特征缓存”这种优化空间**。

**两级验收（v2 核心改动）**：

| 级别 | 对比对象 | 判据 | 说明 |
|---|---|---|---|
| **L1 后端误差** | RKNN 轨迹 vs **同 gate9 表**的 ORT 轨迹，**且输入像素必须完全一致**（用 `dump_crops.py` 导出 crop 包，两边跑同一份 uint8） | 中心误差均值 < 2 px、p95 < 5 px、IoU 均值 > 0.99 | 只反映 FP16/NPU 差异，是**你该负责的部分**。本机实测：man 0.9975 / cars 0.9978 / basketball 0.9971 / street 0.9805（street 有 8 帧 argmax 翻转） |
| **L2 语义敏感度** | RKNN 轨迹 vs 历史产物（旧 5-block TRT = 同语义；旧 4-block TRT = 不同语义） | **只记录不设门槛**：本机实测同语义 0.9717、跨语义 0.9449 | 反映“换后端/换冻结表”的代价，用来给用户做取舍 |

**⚠️ 不要用“板端解码 + PC 参考”直接比轨迹**：板端与 PC 解码同一帧就差 ±2 灰度级（实测 61–63% 通道不同），
而本模型对 ±2 噪声极敏感（逐帧 IoU 会从 1.000 掉到 0.60–0.86）。这种对比测的是解码器差异，不是 NPU 精度。

**不要**再写“RKNN 与 PyTorch 框中心误差 < 5 px”——活的 PyTorch 每帧可能换图，这个指标不成立。

**验收**：L1 达标；L2 数字记进 `results/tracking.md`。

---

### Step 7：精度评测（1–2 天，依赖数据集）

- **强评测**（需要用户提供数据）：UAV123 / UAVDT / DTB70，指标 Success / Precision（SR/PR）；FP16 相对“同 gate9 表的 PyTorch”掉点 ≤ 0.5。
- **弱评测**（本机可做，4 段视频 × 300 帧）：
  - 每段：RKNN vs ORT（同表）逐帧中心误差 + IoU + 漂移帧数；
  - 再各跑一次 `{...4...}` 表（或直接引用现成 CSV）做“gate 敏感度”对照；
  - 输出到 `results/accuracy.md`，附 3~4 张关键帧截图（`results/frames/`）。

**判据**：不要用“单帧 map 的 max|d|”代替跟踪指标——map 接近也可能漂移；反之亦然（本轮实测已证明）。

---

### Step 8：INT8（可选，2–4 天）

前置：FP16 达标，且用户明确要体积/速度。

1. 校准集：300–500 组 `(template, search)`，覆盖 4 段视频 × 不同尺度/背景/光照（用 §2.1 的抽取脚本批量生成）；
2. `do_quantization=True, dataset=calib.txt`；
3. 先全 INT8，测 L1/L2 + SR/PR；
4. 掉点 > 1 → hybrid：`patch_embed + MLP` INT8，`attention/softmax` FP16（RKNN 2.5 的 hybrid 流程不携带 `op_target`；本模型无 GridSample，不受影响）；
5. 仍不达标 → 放弃 INT8，交付 FP16。

**校准集格式**：先用 1 个样本试 `build(do_quantization=True, dataset=...)`；多输入报数量不匹配时，照 toolkit 自带 `examples/functions/quantization` 的写法（每个样本一个 npy，内部存两个数组）。

---

### Step 9：C++（可选，交付形态需要时才做）

- `rknn_init` → 查输入输出 attr → 分配 → `rknn_inputs_set(want_float=0)`（u8 输入）→ `rknn_run` → `rknn_outputs_get`；
- **输出只有 1 KB，直接用 `want_float=1`**，不要为零拷贝去处理 `want_float=0` 的连续 NCHW + 尾部 padding；
- 预处理用 RGA：crop → resize → RGB，注意 stride 对齐与 BGR/RGB；
- 线程模型：每路一个线程 + 一个 NPU core，避免共享 `rknn_context`；
- INT8 输入硬件要求 `u8 ^ 0x80`（v4.1 经验），FP16 模型不需要。

---

### Step 10：交付（1 天）

```
AVTrack-RK3588/
  build/avtrack_gate9_op17_fp16.rknn     # + md5
  build/avtrack_gate9_int8hybrid.rknn    # 可选
  onnx/avtrack_gate9_op17.onnx           # + md5，供复现
  scripts/                               # 一键重跑：export / check / build / run / bench / track
  ref/pairs.npz, ref/out_torch.npz, ref/out_ort.npz
  results/latency.md
  results/accuracy.md
  results/tracking.md                    # L1/L2 表
  README.md                              # 输入约定、mean/std、gate 表、后处理、已知坑
```

命名：`avtrack_gate9_<opset>_<precision>_<date>.rknn`。交付时必须给：md5、构建命令、工具链版本、**冻结的 gate 表**。

---

## 5. 从 RT-DETR 项目复用来的经验（逐条看）

| # | 现象 / 教训 | 原因 | 在本项目怎么用 |
|---|---|---|---|
| 1 | build 成功、推理全 0 | 禁用 `squeeze_nd_mul` 后 6D 广播 Mul 编译成零 kernel（issue #587） | 本模型无 >4D，**但仍不要照抄 RT-DETR 的 `disable_rules`**；任何输出先打 nonzero |
| 2 | `optimization_level=3` 构建崩溃 | `fuse_mul_into_gemm` 广播 bug | 默认 opt0；只有实测 opt3 更快且不崩才用 |
| 3 | `want_float=0` 输出布局与文档不符 | 返回连续 NCHW + 尾部 padding | 本模型输出小，直接用 `want_float=1` |
| 4 | INT8 输入全错 | 硬件输入需要 `u8 ^ 0x80` | 只有 INT8 + C++ 零拷贝时才会遇到 |
| 5 | `op_target` 设了没生效 | op_type 级 key 覆盖 tensor 级 key | 本项目不需要 op_target |
| 6 | 拆分模型配错 | 两个模型必须成对 | 本项目单模型；若以后拆 backbone/head，用 `custom_string` 校验 |
| 7 | 构建卡死 | 某些图让 `OpFusing` 进入病态循环 | 所有 build 加超时（30 min），超时 kill 并记录图特征 |
| 8 | 多实例不线性 | 多实例共享三核互相抢 | 每实例绑单核；测 1/2/3/4 路 |
| 9 | simulator 与板端有差 | 模拟器不等于硬件 | 判据以板端为准；simulator 只筛错 |
| 10 | 预处理成瓶颈 | CPU `cv2.resize` / 颜色转换 | 先测出来，再决定是否上 RGA |
| 11 | 日志刷屏 | 模型自带 debug print | 导出/参考脚本 `redirect_stdout`；不改上游 |
| 12 | “看起来对了” | 没有对照 | 每个结论留 A/B 模型和数字 |

### 本项目新增的 5 条

| # | 现象 | 原因 | 怎么办 |
|---|---|---|---|
| 13 | 同一 checkpoint 两次导出，block 数/内容不同 | `forward_features` 的数据相关跳块被 trace 固化 | **导出前 `freeze_gate()`**，导出后断言 block 集合，两次 dummy 交叉验证 |
| 14 | 两张“都正常”的图，跟踪框差 32 px | 上面那条的直接后果（4 号 vs 5 号块） | 所有对比先声明 gate 表（纪律 8）；L2 指标只记录不设门槛 |
| 15 | `strict=False` 让人怀疑漏权重 | 上游 `export_onnx.py` 用的是 `strict=False` | **一律 `strict=True`**；漏一个 key 就报错 |
| 16 | 导出成功但报错位置莫名 | `build_avtrack` 内 `pretrained=True` 会联网下载 | monkeypatch `pretrained=False`；离线环境尤其重要 |
| 17 | 参考输出和板端差很多、趋势却对 | PyTorch 参考没冻结 gate，或 mean/std 做了两次 | 参考必须冻结 + 预处理只有一处 |

---

## 6. 什么时候停下来问用户

只有这五种：

1. **指标不明确**：单路多少 ms 算达标？要不要多路？
2. **需要取舍**：INT8 掉 1.5 个点，保精度还是保速度？**以及 gate 表要不要改（当前钉 `{跳 3,4,11}`）**。
3. **缺资源**：GT 数据集不在本机、板子被占用、要账号/密码。
4. **本地无法闭环**：需要上游改 toolkit（像 #587 那样）。
5. **发现“冻结表改了会更好”的证据**：那是模型语义决策，不是部署细节，必须用户拍板。

其余时间自己迭代，有结论再汇报。

---

## 7. DoD 检查清单

- [ ] `avtrack_gate9_op17.onnx` 的 block 集合 = `{0,1,2,5,6,7,8,9,10}`，`LayerNormalization` 19 个
- [ ] 两次不同 dummy 导出的 ONNX 输出一致（`max|d| < 1e-6`）
- [ ] op17 与 op11 输出一致（`max|d| < 1e-5`）
- [ ] 冻结语义的 PyTorch vs ORT 三路一致（`max|d| < 1e-5`）
- [ ] FP16 `.rknn` 构建日志：无 CPU 回退，或有回退且已量化
- [ ] simulator 精度达标（score < 2e-3、size/offset < 5e-3）
- [ ] 板端三路输出与 ORT 同量级，100 帧稳定不泄漏
- [ ] 延迟拆成 pre/post/NPU 三段，给出 1/2/3/4 实例 FPS
- [ ] 跟踪 runner 无 GUI 可跑，产出 CSV/JSON/视频
- [ ] **L1 达标**（vs 同 gate9 表的 ORT：中心 < 2 px、IoU > 0.99）
- [ ] **L2 已记录**（vs `{...4...}` 表：中心/IoU 数字，含 32 px 上界对照）
- [ ] 精度表（强评测或弱评测）已写明 gate 表与参考实现
- [ ] （可选）INT8 掉点 ≤ 1，否则明确写“不采用”
- [ ] 脚本可一键重跑，模型/ONNX 带 md5
- [ ] 文档写清：输入约定、mean/std、gate 表、后处理、已知坑

---

## 8. 快速开始（复制粘贴）

```bash
# 0) 依赖（PC 侧）
pip install torch onnx onnxruntime timm easydict tensorboardX opencv-python
pip install rknn-toolkit2==2.5.0

# 1) 配置：全部可用环境变量覆盖，见 scripts/env.sh
export PY=/path/to/python
export SRC=/path/to/AVTrack            # 上游仓库
export VIDEOS=/path/to/videos          # <name>.mp4 + <name>.txt

# 2) 导出 + 金标准 + 构建（每步都有断言与预期数字）
bash scripts/01_export_onnx.sh
bash scripts/02_make_reference.sh
bash scripts/03_build_rknn.sh

# 3) 上板（密码只走 SSHPASS，不落盘）
export BOARD=user@<board-ip> SSHPASS='****'
bash scripts/04_deploy.sh
bash scripts/05_board_numeric.sh      # 20 组同像素数值 + 100 帧稳定
bash scripts/06_board_bench.sh        # 延迟 / 2-3 路并行 / 8000 帧长稳

# 4) 跟踪链路
bash scripts/07_track_pc.sh           # PC ORT 参考轨迹
bash scripts/08_track_board.sh        # 板端 RKNN + 自动对照
bash scripts/09_fair_l1.sh man street # 同像素公平对照（L1）
bash scripts/10_check_decode.sh       # 解码差异实测（解释跨设备差异）
```

---

## 附录 A：gate 实测原始数据（复核用）

`active_score_module`（每块 `Linear(320,1)`）在本 checkpoint 上的参数：

```
block  2: |w|max=3.34e-05 |w|mean=8.95e-06 bias=+5.768e-04  w_norm=2.04e-04  sigmoid(bias)=0.500144
block  3: |w|max=1.58e-03 |w|mean=3.40e-04 bias=-9.916e-04  w_norm=7.54e-03  sigmoid(bias)=0.499752
block  4: |w|max=1.93e-02 |w|mean=4.74e-03 bias=-4.297e-03  w_norm=1.07e-01  sigmoid(bias)=0.498926
block  5: |w|max=1.80e-02 |w|mean=4.43e-03 bias=+4.710e-03  w_norm=1.01e-01  sigmoid(bias)=0.501177
block  6: w_norm=8.31  bias=+1.203  sigmoid=0.769
block  7: w_norm=1.54  bias=+0.182  sigmoid=0.545
block  8: w_norm=1.35  bias=+0.153  sigmoid=0.538
block  9: w_norm=1.25  bias=+0.137  sigmoid=0.534
block 10: w_norm=1.13  bias=+0.149  sigmoid=0.537
block 11: w_norm=1.05  bias=-0.0328 （logit 饱和，实测概率 <1e-6）
```

13 组不同输入（灰/白/黑/噪声/渐变/斑块/条纹）上的 sigmoid 概率范围：
`block2: 0.5001–0.5002`、`block3: 0.4996–0.4999`、`block4: 0.4961–0.5008`、`block5: 0.4995–0.5039`、
`block6–10: 恒 1.0`、`block11: <1e-6`；其中 block4/5 在 1 组输入上确实翻转。

据此裁定冻结表 `RUN={0,1,2,5,6,7,8,9,10}`、`SKIP={3,4,11}`。

> 复核脚本思路：`load_state_dict(strict=True)` 后逐块打印
> `lin.weight.norm()`, `lin.bias.item()`, `torch.sigmoid(lin.bias).item()`；
> 再对一个 batch 前向，用 `blk.forward` 包装计数得到实际执行的 block 集合。
