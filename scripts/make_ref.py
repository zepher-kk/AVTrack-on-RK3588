#!/usr/bin/env python
"""Step 2: 用 4 段视频建固定输入对 + 冻结语义参考输出（torch / ort）。

输出（ref/ 目录）:
  pairs.npz       template(N,128,128,3) uint8 RGB / search(N,256,256,3) uint8 RGB / meta
  out_torch.npz   冻结 gate 的 PyTorch 输出
  out_ort.npz     gate9 ONNX 的 ORT 输出
  ref.json        每组 crop_sz、输出 min/max/nonzero、torch-vs-ort 误差
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import avtrack_common as C  # noqa: E402
import paths as P  # noqa: E402


def sample_frames(video, roi, tpl_size, src_size, n_search):
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError('cannot read video: %s' % video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError('cannot read frame0: %s' % video)
    tpl_bgr, tpl_crop_sz = C.crop_hwc(frame0, roi, tpl_size, C.TPL_FACTOR)

    idxs = sorted(set(int(round(x)) for x in np.linspace(0, total - 1, n_search)))
    rows = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            continue
        crop_bgr, crop_sz = C.crop_hwc(fr, roi, src_size, C.SRC_FACTOR)
        rows.append((fi, crop_bgr, crop_sz, frame0.shape[:2]))
    cap.release()
    return tpl_bgr, tpl_crop_sz, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos', default=P.VIDEOS)
    ap.add_argument('--out', default=P.REF_DIR)
    ap.add_argument('--onnx', default=P.ONNX_OP17)
    ap.add_argument('--source', default=P.SRC)
    ap.add_argument('--config', default=P.CFG)
    ap.add_argument('--checkpoint', default=P.CKPT)
    ap.add_argument('--n-search', type=int, default=5)
    a = ap.parse_args()

    # 相对路径按上游仓库解析，避免受当前工作目录影响
    cfg_path = a.config if os.path.isabs(a.config) else os.path.join(a.source, a.config)
    ckpt = a.checkpoint if os.path.isabs(a.checkpoint) else os.path.join(a.source, a.checkpoint)

    os.makedirs(a.out, exist_ok=True)
    videos = ['man', 'cars', 'street', 'basketball']
    tpls, srcs, metas = [], [], []

    for name in videos:
        vpath = os.path.join(a.videos, name + '.mp4')
        tpath = os.path.join(a.videos, name + '.txt')
        if not (os.path.exists(vpath) and os.path.exists(tpath)):
            print('SKIP', name)
            continue
        roi_name, roi = C.read_roi(tpath)
        tpl_bgr, tpl_cs, rows = sample_frames(vpath, roi, C.TPL_SIZE, C.SRC_SIZE, a.n_search)
        tpl_rgb = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2RGB)
        for fi, crop_bgr, crop_sz, fshape in rows:
            tpls.append(tpl_rgb)
            srcs.append(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            metas.append(dict(video=name, roi_name=roi_name, roi=roi, frame=fi,
                              tpl_crop_sz=tpl_cs, search_crop_sz=crop_sz,
                              frame_hw=list(fshape)))
        print('%-11s roi=%s frames=%d tpl_crop_sz=%.1f' % (name, roi, len(rows), tpl_cs))

    tpl = np.stack(tpls, 0).astype(np.uint8)
    src = np.stack(srcs, 0).astype(np.uint8)
    np.savez(os.path.join(a.out, 'pairs.npz'), template=tpl, search=src,
             meta=json.dumps(metas, ensure_ascii=False))
    print('pairs:', tpl.shape, src.shape)

    import torch
    torch.nn.functional.scaled_dot_product_attention = lambda q, k, v, **kw: (
        torch.softmax(q @ k.transpose(-2, -1) * (q.size(-1) ** -0.5), dim=-1) @ v)

    model, info = C.build_model(a.source, cfg_path, ckpt, frozen=True)
    print('load:', info)

    import onnxruntime as ort
    sess = ort.InferenceSession(a.onnx, providers=['CPUExecutionProvider'])

    names = ('score_map', 'size_map', 'offset_map')
    t_out = [[], [], []]
    o_out = [[], [], []]
    for i in range(len(tpl)):
        t_in = C.prep_rgb(tpl[i])
        s_in = C.prep_rgb(src[i])
        pt = C.model_forward(model, t_in, s_in)
        ot = sess.run(None, {'template': t_in, 'search': s_in})
        for k in range(3):
            t_out[k].append(pt[k][0])
            o_out[k].append(np.asarray(ot[k])[0])

    t_out = [np.stack(x, 0) for x in t_out]
    o_out = [np.stack(x, 0) for x in o_out]
    np.savez(os.path.join(a.out, 'out_torch.npz'), score=t_out[0], size=t_out[1], offset=t_out[2])
    np.savez(os.path.join(a.out, 'out_ort.npz'), score=o_out[0], size=o_out[1], offset=o_out[2])

    report = dict(pairs=int(len(tpl)), metas=metas, torch_vs_ort={}, outputs={})
    for nm, x, y in zip(names, t_out, o_out):
        d = np.abs(x - y)
        report['torch_vs_ort'][nm] = dict(max_abs=float(d.max()), mean_abs=float(d.mean()))
        report['outputs'][nm] = dict(min=float(x.min()), max=float(x.max()),
                                     nonzero=int(np.count_nonzero(x)), size=int(x.size))
    # 每组 score 峰值，便于后续挑阈值
    report['score_peak_per_pair'] = [float(v) for v in o_out[0].reshape(len(tpl), -1).max(1)]
    C.dump_json(os.path.join(a.out, 'ref.json'), report)
    print(json.dumps(report['torch_vs_ort'], indent=2))
    print('score peak per pair:', [round(v, 3) for v in report['score_peak_per_pair']])
    print('outputs:', json.dumps(report['outputs'], indent=2))


if __name__ == '__main__':
    main()
