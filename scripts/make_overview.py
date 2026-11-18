#!/usr/bin/env python
"""生成 README 用的总览图：4 段视频 × 板端 RKNN 预测框，统一尺寸 2×2 拼图。

用法:
  python make_overview.py --videos /path/to/videos \
      --tracks results/tracks/board_rknn --out assets/overview.png
"""
import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P  # noqa: E402

# 每段视频取一帧 + 图上标注
PICKS = [('man', 160, 'man / person'), ('cars', 40, 'cars / vehicle'),
         ('street', 120, 'street / pedestrian'), ('basketball', 40, 'basketball / hoop')]
TILE = (600, 400)
GREEN = (86, 214, 124)
FONTS = ['/mnt/c/Windows/Fonts/msyh.ttc', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']


def font(size, bold=False):
    for p in FONTS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size, index=1 if (bold and p.endswith('.ttc')) else 0)
            except OSError:
                continue
    return ImageFont.load_default()


def load_track(path):
    a = np.loadtxt(path, delimiter=',', skiprows=1)
    return {int(r[0]): (np.array(r[1:5], float), float(r[5])) for r in np.atleast_2d(a)}


def crop_around(frame, box, tile=TILE, pad=2.6, min_side=300):
    """以目标为中心裁一块固定长宽比的区域；窗口放不下时先整体缩放，再越界补白。"""
    H, W = frame.shape[:2]
    cx = box[0] + box[2] / 2.0
    cy = box[1] + box[3] / 2.0
    aspect = tile[0] / float(tile[1])
    side = max(max(box[2], box[3]) * pad, min_side)
    cw, ch = side * aspect, side
    # 关键：窗口不能大于画面（否则会补出大片白边），按需等比缩小
    k = min(1.0, W / cw, H / ch)
    cw, ch = cw * k, ch * k
    x0, y0 = int(round(cx - cw / 2)), int(round(cy - ch / 2))
    x1, y1 = x0 + int(round(cw)), y0 + int(round(ch))

    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x1), min(H, y1)
    crop = frame[sy0:sy1, sx0:sx1]
    need_w, need_h = x1 - x0, y1 - y0
    pw, ph = need_w - crop.shape[1], need_h - crop.shape[0]
    left, top = sx0 - x0, sy0 - y0
    crop = np.pad(crop, ((top, ph - top), (left, pw - left), (0, 0)),
                  mode='constant', constant_values=255)
    return cv2.resize(crop, tile, interpolation=cv2.INTER_AREA), (
        (box[0] - x0) * tile[0] / need_w, (box[1] - y0) * tile[1] / need_h,
        box[2] * tile[0] / need_w, box[3] * tile[1] / need_h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos', default=P.VIDEOS)
    ap.add_argument('--tracks', default=os.path.join(P.WORK, 'results/tracks/board_rknn'))
    ap.add_argument('--out', default=os.path.join(P.WORK, 'assets/overview.png'))
    a = ap.parse_args()

    tiles = []
    for name, idx, label in PICKS:
        vp = os.path.join(a.videos, name + '.mp4')
        tp = os.path.join(a.tracks, name, 'track.csv')
        if not (os.path.exists(vp) and os.path.exists(tp)):
            print('skip', name)
            continue
        tr = load_track(tp)
        cap = cv2.VideoCapture(vp)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print('cannot read', vp)
            continue
        box, conf = tr.get(idx, (np.array([0, 0, 32, 32.0]), 0.0))
        tile_rgb, b = crop_around(frame, box)
        img = Image.fromarray(cv2.cvtColor(tile_rgb, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        d.rectangle([b[0], b[1], b[0] + b[2], b[1] + b[3]], outline=GREEN, width=3)
        tag = '%s  frame %d  conf %.2f' % (label, idx, conf)
        tw = d.textlength(tag, font=font(13))
        d.rectangle([6, 6, 6 + tw + 12, 30], fill=(20, 22, 26))
        d.text((12, 9), tag, font=font(13), fill=(235, 240, 245))
        tiles.append(img)
        print('tile %-22s %s' % (label, img.size))

    if len(tiles) != 4:
        raise SystemExit('需要 4 段视频，当前 %d' % len(tiles))

    gap, pad = 10, 14
    W = TILE[0] * 2 + gap + pad * 2
    H = TILE[1] * 2 + gap + pad * 2 + 34
    canvas = Image.new('RGB', (W, H), 'white')
    pos = [(pad, pad), (pad + TILE[0] + gap, pad),
           (pad, pad + TILE[1] + gap), (pad + TILE[0] + gap, pad + TILE[1] + gap)]
    for img, (x, y) in zip(tiles, pos):
        canvas.paste(img, (x, y))
        ImageDraw.Draw(canvas).rectangle([x - 1, y - 1, x + TILE[0], y + TILE[1]],
                                         outline=(214, 220, 228), width=1)
    d = ImageDraw.Draw(canvas)
    d.text((pad, H - 26), 'RK3588 · RKNN FP16 · 板端预测框（绿框 + 置信度），单核 25 FPS',
           font=font(14), fill=(60, 66, 76))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    canvas.save(a.out, optimize=True)
    print('wrote %s  %s  %.2f MB' % (a.out, canvas.size, os.path.getsize(a.out) / 1e6))


if __name__ == '__main__':
    main()
