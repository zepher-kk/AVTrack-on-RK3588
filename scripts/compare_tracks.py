#!/usr/bin/env python
"""Step 6c: 跟踪结果对照（L0 复现校验 / L1 后端误差 / L2 语义敏感度）。

L1: gate9 RKNN(板端) vs gate9 ORT(PC)  —— 同一个冻结图，只差后端精度
L2: gate9 vs 历史两张图（仓库 5-block TRT / 上个项目 4-block TRT），man.mp4
L0: gate9 ORT vs 历史 5-block TRT —— 验证“我们复现的是同一套语义”
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
PRIOR = P.PRIOR


def load(path):
    a = np.loadtxt(path, delimiter=',', skiprows=1)
    if a.ndim == 1:
        a = a[None, :]
    return a


def metrics(a, b, n=None):
    """a,b: [frame,x,y,w,h,...]。按 frame 对齐后算中心距离/IoU。"""
    fa = {int(r[0]): r[1:5] for r in a}
    fb = {int(r[0]): r[1:5] for r in b}
    fc = {int(r[0]): r[5] for r in a}          # conf 取 a（参考侧）
    keys = sorted(set(fa) & set(fb))
    if n:
        keys = keys[:n]
    ious, dists, dsz = [], [], []
    for k in keys:
        ious.append(C.bbox_iou(fa[k], fb[k]))
        dists.append(C.center_dist(fa[k], fb[k]))
        dsz.append(max(abs(fa[k][2] - fb[k][2]), abs(fa[k][3] - fb[k][3])))
    ious = np.array(ious)
    dists = np.array(dists)
    # 只在参考侧"确信"的帧上统计（目标小/丢失的退化帧会主导 IoU）
    solid = [i for i, k in enumerate(keys) if fc[k] >= 0.20]
    iou_s = ious[solid] if solid else np.array([np.nan])
    dist_s = dists[solid] if solid else np.array([np.nan])
    return dict(
        frames=len(keys),
        iou_mean=float(ious.mean()), iou_p05=float(np.percentile(ious, 5)),
        iou_min=float(ious.min()),
        iou_lt_090=int((ious < 0.9).sum()), iou_lt_050=int((ious < 0.5).sum()),
        center_mean_px=float(dists.mean()), center_p95_px=float(np.percentile(dists, 95)),
        center_max_px=float(dists.max()),
        size_max_px=float(max(dsz)), size_mean_px=float(np.mean(dsz)),
        solid_frames=len(solid),
        solid_iou_mean=float(np.nanmean(iou_s)),
        solid_center_mean_px=float(np.nanmean(dist_s)),
        solid_center_p95_px=float(np.nanpercentile(dist_s, 95)),
        solid_iou_lt_090=int((iou_s < 0.9).sum()),
    )


def main():
    videos = ['man', 'cars', 'street', 'basketball']
    report = {'L1_rknn_vs_ort': {}, 'L0_ort_vs_prior5': None,
              'L2_gate9_vs_prior': {}, 'meta': {}}

    for v in videos:
        po = os.path.join(WORK, 'results/pc_ort', v, 'track.csv')
        pr = os.path.join(WORK, 'results/rknn', v, 'track.csv')
        if not (os.path.exists(po) and os.path.exists(pr)):
            print('skip', v)
            continue
        m = metrics(load(po), load(pr))
        report['L1_rknn_vs_ort'][v] = m
        so = json.load(open(os.path.join(WORK, 'results/pc_ort', v, 'summary.json')))
        sr = json.load(open(os.path.join(WORK, 'results/rknn', v, 'summary.json')))
        report['meta'][v] = dict(
            ort_fps=so['fps'], rknn_fps=sr['fps'],
            ort_conf_mean=so['conf_mean'], rknn_conf_mean=sr['conf_mean'],
            ort_final=so['final_state'], rknn_final=sr['final_state'],
            frames=so['frames'])
        print('[L1] %-11s frames=%3d IoU=%.4f p05=%.4f min=%.4f center=%.3f/%.2f/%.2f px  #IoU<0.9=%d'
              % (v, m['frames'], m['iou_mean'], m['iou_p05'], m['iou_min'],
                 m['center_mean_px'], m['center_p95_px'], m['center_max_px'],
                 m['iou_lt_090']))
        print('     solid(conf>=0.2, %d frames): IoU=%.4f center=%.3f/%.2f px  #IoU<0.9=%d'
              % (m['solid_frames'], m['solid_iou_mean'], m['solid_center_mean_px'],
                 m['solid_center_p95_px'], m['solid_iou_lt_090']))

    p5 = os.path.join(PRIOR, 'results/avtrack.csv') if PRIOR else ''
    p4 = os.path.join(PRIOR, 'results_exported/avtrack.csv') if PRIOR else ''
    man_ort = os.path.join(WORK, 'results/pc_ort/man/track.csv')
    man_rknn = os.path.join(WORK, 'results/rknn/man/track.csv')
    if p5 and os.path.exists(p5) and os.path.exists(man_ort):
        report['L0_ort_vs_prior5'] = metrics(load(man_ort), load(p5), n=299)
        print('[L0] ours(gate9 ORT) vs prior 5-block TRT: IoU=%.4f center=%.2f/%.2f/%.2f px'
              % (report['L0_ort_vs_prior5']['iou_mean'],
                 report['L0_ort_vs_prior5']['center_mean_px'],
                 report['L0_ort_vs_prior5']['center_p95_px'],
                 report['L0_ort_vs_prior5']['center_max_px']))
    if p4 and os.path.exists(p4) and os.path.exists(man_rknn):
        report['L2_gate9_vs_prior'] = {
            'rknn_gate9_vs_prior4': metrics(load(man_rknn), load(p4), n=299),
            'rknn_gate9_vs_prior5': (metrics(load(man_rknn), load(p5), n=299)
                                     if os.path.exists(p5) else None),
        }
        m = report['L2_gate9_vs_prior']['rknn_gate9_vs_prior4']
        print('[L2] ours(gate9 RKNN) vs prior 4-block TRT: IoU=%.4f p05=%.4f center=%.2f/%.2f/%.2f'
              % (m['iou_mean'], m['iou_p05'], m['center_mean_px'],
                 m['center_p95_px'], m['center_max_px']))

    C.dump_json(os.path.join(WORK, 'results/tracking.json'), report)
    print('->', os.path.join(WORK, 'results/tracking.json'))


if __name__ == '__main__':
    main()
