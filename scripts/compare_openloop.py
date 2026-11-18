#!/usr/bin/env python
"""开环诊断对照：同一 crop（状态来自 ORT 闭环）下，RKNN 与 ORT 的逐帧解码差异。

用途：区分"单帧数值误差"与"闭环反馈放大"。若逐帧差异很小而闭环漂移很大，
说明漂移是跟踪回路的固有放大，不是算子问题。
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import avtrack_common as C  # noqa: E402
import paths as P  # noqa: E402

WORK = P.WORK


def load(p):
    a = np.loadtxt(p, delimiter=',', skiprows=1)
    return np.atleast_2d(a)


def main():
    vids = sys.argv[1:] or ['cars', 'street']
    rep = {}
    for v in vids:
        a = load(os.path.join(WORK, 'results/openloop_ort', v, 'track.csv'))
        b = load(os.path.join(WORK, 'results/openloop_rknn', v, 'track.csv'))
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        d = np.abs(a[:, 1:5] - b[:, 1:5])
        dist = np.array([C.center_dist(a[i, 1:5], b[i, 1:5]) for i in range(n)])
        iou = np.array([C.bbox_iou(a[i, 1:5], b[i, 1:5]) for i in range(n)])
        conf_d = np.abs(a[:, 5] - b[:, 5])
        rep[v] = dict(frames=int(n),
                      center_mean=float(dist.mean()), center_p95=float(np.percentile(dist, 95)),
                      center_max=float(dist.max()),
                      iou_mean=float(iou.mean()), iou_min=float(iou.min()),
                      wh_mean=float(d[:, 2:].mean()), wh_max=float(d[:, 2:].max()),
                      conf_mean=float(conf_d.mean()), conf_max=float(conf_d.max()))
        print('[openloop] %-7s frames=%d center=%.3f/%.2f/%.2f px  IoU=%.4f(min %.3f)  '
              'wh=%.2f/%.2f px  conf=%.4f/%.4f'
              % (v, n, rep[v]['center_mean'], rep[v]['center_p95'], rep[v]['center_max'],
                 rep[v]['iou_mean'], rep[v]['iou_min'], rep[v]['wh_mean'], rep[v]['wh_max'],
                 rep[v]['conf_mean'], rep[v]['conf_max']))
    C.dump_json(os.path.join(WORK, 'results/openloop.json'), rep)


if __name__ == '__main__':
    main()
