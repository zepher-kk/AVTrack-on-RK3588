#!/usr/bin/env python
"""Step 5: 板端延迟拆解（pre / npu / post）+ 多实例。

单进程:
  python bench.py --rknn build/x.rknn --video videos/man.mp4 --bbox 80 30 145 190 \
      --frames 300 --core 0 --out results/bench_core0.json
三实例并行由 shell 起 3 个进程各自 --core 0/1/2。
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import avtrack_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rknn', required=True)
    ap.add_argument('--video', required=True)
    ap.add_argument('--bbox', type=float, nargs=4, required=True)
    ap.add_argument('--frames', type=int, default=300)
    ap.add_argument('--warmup', type=int, default=30)
    ap.add_argument('--core', default='0')
    ap.add_argument('--out', default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--nhwc', action='store_true',
                    help='按 NHWC 传输入（rknnlite 日志提示模型内部要 NHWC，可省一次内部转置）')
    a = ap.parse_args()

    from rknnlite.api import RKNNLite

    mask_map = {'0': RKNNLite.NPU_CORE_0, '1': RKNNLite.NPU_CORE_1,
                '2': RKNNLite.NPU_CORE_2, '012': RKNNLite.NPU_CORE_0_1_2}
    rk = RKNNLite()
    assert rk.load_rknn(a.rknn) == 0
    assert rk.init_runtime(core_mask=mask_map[str(a.core)]) == 0

    cap = cv2.VideoCapture(a.video)
    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError('cannot read %s' % a.video)
    h_img, w_img = frame0.shape[:2]
    state = list(a.bbox)
    tpl_bgr, _ = C.crop_hwc(frame0, state, C.TPL_SIZE, C.TPL_FACTOR)
    tpl_in = C.prep_bgr(tpl_bgr)

    frames = []
    # 流式读帧，避免长稳测试时把整段视频堆在内存里
    def next_frame():
        nonlocal cap
        ok, fr = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, fr = cap.read()
        return fr

    def one(fr):
        t0 = time.perf_counter()
        crop_bgr, crop_sz = C.crop_hwc(fr, state, C.SRC_SIZE, C.SRC_FACTOR)
        src_in = C.prep_bgr(crop_bgr)
        t1 = time.perf_counter()
        if a.nhwc:
            outs = rk.inference(
                inputs=[np.ascontiguousarray(tpl_in[0].transpose(1, 2, 0)[None]),
                        np.ascontiguousarray(src_in[0].transpose(1, 2, 0)[None])],
                data_format='nhwc')
        else:
            outs = rk.inference(inputs=[tpl_in, src_in], data_format='nchw')
        t2 = time.perf_counter()
        pp = C.postprocess(outs[0], outs[1], outs[2], crop_sz)
        C.update_state(state, pp, w_img, h_img)
        t3 = time.perf_counter()
        return (t1 - t0) * 1000.0, (t2 - t1) * 1000.0, (t3 - t2) * 1000.0

    for i in range(a.warmup):
        one(next_frame())

    pre, npu, post = [], [], []
    for i in range(a.frames):
        p, n, q = one(next_frame())
        pre.append(p)
        npu.append(n)
        post.append(q)

    pre, npu, post = map(np.array, (pre, npu, post))
    total = pre + npu + post
    report = dict(
        tag=a.tag, core=str(a.core), frames=int(a.frames), video=os.path.basename(a.video),
        pre_ms=dict(p50=float(np.percentile(pre, 50)), p90=float(np.percentile(pre, 90)),
                    mean=float(pre.mean())),
        npu_ms=dict(p50=float(np.percentile(npu, 50)), p90=float(np.percentile(npu, 90)),
                    mean=float(npu.mean())),
        post_ms=dict(p50=float(np.percentile(post, 50)), p90=float(np.percentile(post, 90)),
                     mean=float(post.mean())),
        total_ms=dict(p50=float(np.percentile(total, 50)), p90=float(np.percentile(total, 90)),
                      mean=float(total.mean())),
        fps=float(1000.0 / total.mean()),
    )
    rk.release()
    print(json.dumps(report, indent=2))
    if a.out:
        C.dump_json(a.out, report)


if __name__ == '__main__':
    main()
