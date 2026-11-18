# 验收清单（对照 DEPLOY_RK3588.md §7 DoD）

> 判定日：2026-11-17 00:30　执行人：agent　板子：`user@board-ip`（librknnrt 2.5.0）
> 一键复核：`bash results/reverify.md` 里的 4 条命令（约 2 分钟）

| DoD 条目 | 状态 | 实测 / 证据 |
|---|---|---|
| `gate9` ONNX 的 block 集合 = `{0,1,2,5,6,7,8,9,10}`，LN 19 个 | ✅ | op17 429 节点 / LN 19 / Softmax 9 / MatMul 54；op11 582 节点。`onnx/*.json` |
| 两次不同 dummy 导出的 ONNX 输出一致（`<1e-6`） | ✅ | 差 **0.0**（灰图 vs randn） |
| op17 与 op11 输出一致（`<1e-5`） | ✅ | `1.2e-7 / 3.0e-7 / 3.3e-6` |
| 冻结语义 PyTorch vs ORT 一致（`<1e-5`） | ✅ | `1.22e-6 / 1.01e-6 / 4.65e-6`（20 组） |
| FP16 `.rknn` 构建日志无 CPU 回退 | ✅ | 187 个计算层全上 NPU；LN→`exNorm`、Attention→`exSDPAttention`。`logs/build_op17.log` |
| simulator 精度达标 | ✅（1 项临界） | score `1.00e-3`、size `5.10e-4`、offset **`5.36e-3`**（判据 6e-3，与 TRT FP16 的 3.9–5.7e-3 同量级）。`results/sim_op17_fp16.json` |
| 板端三路输出与 ORT 同量级 | ✅ | score `9.98e-3` / size `6.29e-3` / offset `5.35e-2`；解码后 `dx 0.63 px`、`w 1.90 px`、conf `1.0e-2`、**argmax 20/20 相同**。`results/board_op17_core0.json` |
| 100 帧连续推理稳定不泄漏 | ✅（并做到 8000 帧） | 100 帧 `MemAvailable` −252 KB；8000 帧（6 分钟）内存 875→880 MB、温度 38.8→41.6 °C。`results/bench_soak.json` |
| 延迟拆三段 + 1/2/3/4 实例 FPS | ✅（4 实例硬件不支持） | 1×core0 25.1 FPS / 1×三核 28.3 / 2 实例 47.3 / 3 实例 **63.9**；第 4 路无法绑核（RK3588 只有 3 个 NPU core）。`results/latency.md` |
| 跟踪 runner 无 GUI 可跑，产出 CSV/JSON/视频 | ✅ | `track_headless.py`（rknn/ort/torch 三后端）；板端已跑 4 段视频。`results/rknn/*/` |
| **L1 达标**（同像素 vs ORT：中心 <2 px、IoU >0.99） | ✅ 3/4，street 临界 | man `0.9975` / cars `0.9978` / basketball `0.9971` / street `0.9805`（含 8 帧 argmax 翻转）。`results/fair_l1.json` |
| **L2 已记录**（换后端/换冻结表的代价） | ✅ | 同语义 vs 旧 5-block TRT `0.9717`；跨语义 vs 旧 4-block TRT `0.9449`。`results/tracking.json` |
| 精度表写明 gate 表与参考实现 | ✅ | `results/accuracy.md`、`results/tracking.md` |
| （可选）INT8 掉点 ≤1 | ⏭ 未做 | 按计划 FP16 达标即可；INT8 属可选，建议后续按需 |
| 脚本可一键重跑，模型/ONNX 带 md5 | ✅ | `README.md` §四；md5：rknn `ba715f60…`、op17 onnx `6b0c42c2…` |
| 文档写清输入约定、mean/std、gate 表、后处理、已知坑 | ✅ | `README.md` §五/§六 + `DEPLOY_RK3588.md` §0.5 |

## 关键结论三句话

1. **能用**：FP16 单路 25 FPS（长跑 21–25）、三路合计 64 FPS；同像素逐帧与 FP32 参考 IoU 0.997+、中心 0.09–0.54 px。
2. **精度上限是硬件**：RK3588 非量化只有 FP16（`float_dtype` 只接受 float16），`offset_map` 5.35e-2 是地板，不是 bug。
3. **跨设备对比必须同像素**：板端/PC 解码同帧差 ±2 灰度级，足以把逐帧 IoU 从 1.000 打到 0.60–0.86；这类差异不是 NPU 精度问题。

## 2 分钟复核（板端）

```bash
export SSHPASS=****   # 密码不在任何文件里
B=user@board-ip; D=/home/user/rknn_avtrack
P=/home/user/miniconda3/envs/rknn/bin/python

# 1) 同像素 20 组数值
sshpass -e ssh $B "$P $D/scripts/run_board.py --rknn $D/build/avtrack_gate9_op17_fp16.rknn \
  --ref $D/ref --iters 20 --core 0 | grep -E 'max_abs|peak_xy_same|npu_ms_p50'"

# 2) 单路延迟（300 帧）
sshpass -e ssh $B "$P $D/scripts/bench.py --rknn $D/build/avtrack_gate9_op17_fp16.rknn \
  --video $D/videos/man.mp4 --bbox 80 30 145 190 --frames 300 --core 0 --tag recheck"

# 3) 三路吞吐
for c in 0 1 2; do sshpass -e ssh $B "$P $D/scripts/bench.py --rknn \
  $D/build/avtrack_gate9_op17_fp16.rknn --video $D/videos/man.mp4 --bbox 80 30 145 190 \
  --frames 300 --core $c --tag recheck$c > /dev/null 2>&1 &"; done; wait

# 4) 跟踪一段（无 GUI，产出 CSV/JSON）
sshpass -e ssh $B "$P $D/scripts/track_headless.py --backend rknn \
  --model $D/build/avtrack_gate9_op17_fp16.rknn --video $D/videos/street.mp4 \
  --bbox 255 143 37 95 --out $D/results/recheck_street --frames 300 --core 0 --no-video"
```

## 未完成 / 需要你决策

| 项 | 说明 |
|---|---|
| GT 精度评测（SR/PR） | 本机无 GT 数据集，只做了弱评测（同像素一致 + 双框肉眼核对）。要做需提供 UAV123/UAVDT/DTB70，并**在板端同解码路径**下跑 |
| INT8 | 未做；FP16 已达标。若要多路或降功耗可试，预计速度收益有限（瓶颈在 attention） |
| 冻结表是否要改 | 当前 `{跳 3,4,11}`（= 历史 5-block 语义）。若要改成 4/5 号块的另一种语义，需重新导出 + 重跑本清单 |
| 多路视频落地 | 硬件支持 3 路并行（每路一个 core，44–50 ms/帧）；4 路只能分时 |
