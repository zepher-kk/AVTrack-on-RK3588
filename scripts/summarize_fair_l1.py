#!/usr/bin/env python
"""汇总公平开环 L1（同像素）与相关证据，写 results/fair_l1.json。"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import avtrack_common as C  # noqa: E402
import paths as P  # noqa: E402

W = P.RESULTS
CROPS = os.path.join(W, 'crops')


def load(p):
    return np.atleast_2d(np.loadtxt(p, delimiter=',', skiprows=1))


def metrics(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    iou = np.array([C.bbox_iou(a[i, 1:5], b[i, 1:5]) for i in range(n)])
    dist = np.array([C.center_dist(a[i, 1:5], b[i, 1:5]) for i in range(n)])
    wh = np.abs(a[:, 3:5] - b[:, 3:5])
    conf = np.abs(a[:, 5] - b[:, 5])
    if a.shape[1] >= 9 and b.shape[1] >= 9:
        peak_same = int(((a[:, 6] == b[:, 6]) & (a[:, 7] == b[:, 7])).sum())
    else:
        peak_same = -1
    return dict(frames=int(n),
                iou_mean=float(iou.mean()), iou_p05=float(np.percentile(iou, 5)),
                iou_min=float(iou.min()), iou_lt_090=int((iou < 0.9).sum()),
                center_mean_px=float(dist.mean()),
                center_p95_px=float(np.percentile(dist, 95)),
                center_max_px=float(dist.max()),
                wh_mean_px=float(wh.mean()), wh_max_px=float(wh.max()),
                conf_mean_abs=float(conf.mean()), conf_max_abs=float(conf.max()),
                peak_same=peak_same)


def main():
    rep = {}
    for f in sorted(glob.glob(os.path.join(CROPS, '*_ort.csv'))):
        v = os.path.basename(f)[:-8]
        rk = os.path.join(CROPS, v + '_rknn.csv')
        if not os.path.exists(rk):
            continue
        m = metrics(load(f), load(rk))
        rep[v] = m
        print('%-11s frames=%3d IoU=%.4f p05=%.4f center=%.3f/%.2f/%.2f px  '
              'peak_same=%d/%d  #IoU<0.9=%d'
              % (v, m['frames'], m['iou_mean'], m['iou_p05'], m['center_mean_px'],
                 m['center_p95_px'], m['center_max_px'], m['peak_same'], m['frames'],
                 m['iou_lt_090']))

    # 闭环（含解码差异）与噪声敏感度，一起存进同一份报告
    tr = os.path.join(W, 'tracking.json')
    if os.path.exists(tr):
        rep['_closed_loop'] = json.load(open(tr))
    nl = {}
    for v in ('man', 'street', 'cars'):
        a = os.path.join(W, 'pc_ort', v, 'track.csv')
        b = os.path.join(W, 'noise2_ort', v, 'track.csv')
        if os.path.exists(a) and os.path.exists(b):
            nl[v] = metrics(load(a), load(b))
    rep['_noise2_openloop'] = nl
    for v, m in nl.items():
        print('noise=2 %-7s IoU=%.4f center=%.2f/%.2f/%.2f px'
              % (v, m['iou_mean'], m['center_mean_px'], m['center_p95_px'],
                 m['center_max_px']))
    C.dump_json(os.path.join(W, 'fair_l1.json'), rep)


if __name__ == '__main__':
    main()
