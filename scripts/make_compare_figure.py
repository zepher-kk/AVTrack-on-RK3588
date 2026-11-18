#!/usr/bin/env python
"""把 ORT 与 RKNN 的跟踪框画在同一帧上，输出对照图（人工/视觉验收用）。

绿=ORT(PC, 同 gate9)、红=RKNN(板端)；每段视频取若干帧竖向拼接。
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import avtrack_common as C  # noqa: E402
import paths as P  # noqa: E402

WORK = P.WORK
VID = P.VIDEOS


def load(path):
    a = np.loadtxt(path, delimiter=',', skiprows=1)
    return {int(r[0]): [float(v) for v in r[1:5]] for r in np.atleast_2d(a)}


def draw(frame, box, color, tag):
    x, y, w, h = [int(round(v)) for v in box]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, tag, (max(0, x), max(12, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main():
    names = sys.argv[1:] or ['street', 'cars', 'man']
    outdir = os.path.join(WORK, 'results/frames')
    os.makedirs(outdir, exist_ok=True)
    for v in names:
        po = load(os.path.join(WORK, 'results/pc_ort', v, 'track.csv'))
        pr = load(os.path.join(WORK, 'results/rknn', v, 'track.csv'))
        cap = cv2.VideoCapture(os.path.join(VID, v + '.mp4'))
        picks = [0, 40, 80, 120, 160, 200, 240, 299]
        tiles = []
        fi = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if fi in picks and fi in po and fi in pr:
                f = frame.copy()
                draw(f, po[fi], (0, 255, 0), 'ORT %.2f IoU' % C.bbox_iou(po[fi], pr[fi]))
                draw(f, pr[fi], (0, 0, 255), 'RKNN')
                scale = 320.0 / f.shape[1]
                f = cv2.resize(f, (320, int(f.shape[0] * scale)))
                cv2.putText(f, 'frame %d' % fi, (4, f.shape[0] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                tiles.append(f)
            fi += 1
        cap.release()
        if not tiles:
            continue
        nh = min(t.shape[0] for t in tiles)
        tiles = [t[:nh] for t in tiles]
        rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
        grid = np.vstack([cv2.copyMakeBorder(r, 0, 4, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
                          for r in rows])
        p = os.path.join(outdir, 'cmp_%s.png' % v)
        cv2.imwrite(p, grid)
        print('wrote', p, grid.shape)


if __name__ == '__main__':
    main()
