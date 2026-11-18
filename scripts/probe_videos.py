#!/usr/bin/env python
"""列出视频基本信息（尺寸/帧数/fps）与 ROI 文本。"""
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P  # noqa: E402

VID_DIR = sys.argv[1] if len(sys.argv) > 1 else P.VIDEOS
for name in ('man', 'cars', 'street', 'basketball'):
    p = os.path.join(VID_DIR, name + '.mp4')
    if not os.path.exists(p):
        print(name, 'missing')
        continue
    c = cv2.VideoCapture(p)
    w, h = int(c.get(3)), int(c.get(4))
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = c.get(cv2.CAP_PROP_FPS)
    c.release()
    t = os.path.join(VID_DIR, name + '.txt')
    roi = open(t).readline().strip() if os.path.exists(t) else '-'
    print('%-11s %4dx%-4d frames=%-4d fps=%.1f  roi=%s' % (name, w, h, n, fps, roi))
