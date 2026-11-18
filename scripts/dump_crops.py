#!/usr/bin/env python
"""导出"开环 crop 包"：按参考轨迹的 state 逐帧裁剪，供板端/PC 用**完全相同的像素**做对照。

输出 npz: template(uint8 128² RGB, 1 张), search(uint8 256² RGB, N 张), crop_sz(N), state(N,4)
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import avtrack_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--track', required=True, help='参考轨迹 CSV(frame,x,y,w,h,...)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    tr = np.atleast_2d(np.loadtxt(a.track, delimiter=',', skiprows=1))
    states = {int(r[0]): [float(v) for v in r[1:5]] for r in tr}

    cap = cv2.VideoCapture(a.video)
    ok, f0 = cap.read()
    if not ok:
        raise RuntimeError('cannot read video')
    s0 = states[0]
    tpl_bgr, tpl_cs = C.crop_hwc(f0, s0, C.TPL_SIZE, C.TPL_FACTOR)
    tpl_rgb = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2RGB)

    crops, css, used = [], [], []
    fi = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fi in states:
            st = states[fi]
            cb, cs = C.crop_hwc(fr, st, C.SRC_SIZE, C.SRC_FACTOR)
            crops.append(cv2.cvtColor(cb, cv2.COLOR_BGR2RGB))
            css.append(cs)
            used.append(st)
        fi += 1
    cap.release()

    np.savez(a.out, template=tpl_rgb, search=np.stack(crops, 0),
             crop_sz=np.array(css, np.float32), state=np.array(used, np.float32),
             tpl_crop_sz=np.float32(tpl_cs))
    print('wrote %s  frames=%d  size=%.1fMB' % (a.out, len(crops),
                                               os.path.getsize(a.out) / 1e6))


if __name__ == '__main__':
    main()
