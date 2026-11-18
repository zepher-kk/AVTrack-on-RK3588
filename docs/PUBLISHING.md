# 发布到 GitHub（维护者用）

> 这个目录已经是"可直接 commit"的状态：3.2 MB、无大文件、无凭据、无本机路径。
> 模型二进制不在仓库里，走 **Release**（单文件上限 2 GB，不占 LFS 配额）。

## 1. 上传前自检（三条，应全部通过）

```bash
# 1) 不该有任何 >10 MB 的文件
find . -path ./.git -prune -o -type f -size +10M -print

# 2) 不该有凭据 / 内网信息（把你自己环境的关键词也加进 pattern）
grep -rniE 'password|api[_-]?key|token|secret' \
     --include='*.md' --include='*.py' --include='*.sh' --include='*.json' . | grep -v '.git/'
grep -rnE '/mnt/d/|/home/[a-z]+/miniconda' --include='*.md' --include='*.py' . | grep -v '.git/'

# 3) 脚本语法
python -m py_compile scripts/*.py && for f in scripts/*.sh; do bash -n "$f"; done
```

## 2. 初始化并推送

```bash
git init -b main
git add .
git commit -m "AVTrack on RK3588: frozen gate9, FP16 RKNN, 25 FPS/core"
git remote add origin git@github.com:<owner>/AVTrack-RK3588.git
git push -u origin main
```

## 3. 上传模型（Release，不要 commit 进仓库）

```bash
# 构建产物（本地已 gitignore）
#   build/avtrack_gate9_op17_fp16.rknn   15.89 MB
#   onnx/avtrack_gate9_op17.onnx         26.9 MB
gh release create v1.0 \
  build/avtrack_gate9_op17_fp16.rknn \
  onnx/avtrack_gate9_op17.onnx \
  --title "AVTrack gate9 · RK3588 FP16" \
  --notes "rknn md5 ba715f60acb17083cb19e005c7189ad7 / onnx md5 6b0c42c2aee0236f682a0f732bcdbdb4
实测：单核 39.9 ms (25.1 FPS)，三路 63.9 FPS；同像素逐帧 IoU 0.9971–0.9978。
注意：RKNN 会写入构建信息，自己重跑 md5 会不同（体积/数值一致）。"
```

> 想用 LFS 的话，取消 `.gitattributes` 里 `*.onnx` / `*.rknn` 的注释再 `git lfs track`；
> 但免费配额是 1 GB 存储 + 1 GB/月流量，249 MB 的 checkpoint 很容易打满，Release 更合适。

## 4. 有意不放进仓库的东西

| 内容 | 体积 | 原因 | 怎么得到 |
|---|---|---|---|
| `AVTrack_ep0300.pth.tar` | 249 MiB | 超过 GitHub 单文件 100 MB；且训练数据集授权需单独说明 | 上游 Release |
| `avtrack_gate9_op1{1,7}.onnx` | 27 MB ×2 | 可复现（同环境 md5 稳定） | `scripts/01_export_onnx.sh` 或本站 Release |
| `*.rknn` | 15.9 MB | Release | `scripts/03_build_rknn.sh` 或本站 Release |
| `ref/pairs.npz`、`results/crops/*.npz` | 5 MB / 225 MB | 由视频逐帧生成，可重算 | `scripts/02_make_reference.sh`、`09_fair_l1.sh` |
| 演示视频 `*.mp4` | 15 MB | 版权状态不明 | 自备 |
| `results/*/track.mp4` | 45 MB | 可视化产物 | 去掉 `--no-video` 重跑 |
| `check*_*.onnx` | 104 MB | RKNN 构建的中间 dump | 不应存在；`.gitignore` 已忽略 |

## 5. 两个容易踩的坑

1. **RKNN 的 md5 不可复现**：同样的 ONNX、同样的工具链，两次构建字节不同（体积一样）。
   文档里写 md5 时要说清"用于校验你下载的那一份"，不要承诺可复现；ONNX 才是可复现的。
2. **构建请在 `AVTrack-RK3588/` 目录里跑**：`rknn-toolkit2` 会把 `check0_base_optimize.onnx` 等
   4 个中间图 dump 到当前目录（约 104 MB），在源码仓库里跑会把仓库弄脏。

## 6. 建议的仓库设置

- About：`RK3588 · NPU · FP16 · single object tracking · AVTrack`，主页填 Release 链接
- Topics：`rk3588` `rknn` `npu` `visual-tracking` `uav` `onnx` `edge-ai`
- 关闭 Wiki / Projects（用 docs/ 就够）
- 若想向上游提 PR：`freeze_gate()` + "动态跳块被 trace 固化"的说明对他们同样有价值
