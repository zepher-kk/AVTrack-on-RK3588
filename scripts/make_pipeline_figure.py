#!/usr/bin/env python
"""生成 README 用的流水线示意图（assets/pipeline.png）。

用法: python make_pipeline_figure.py --out assets/pipeline.png
"""
import argparse
import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P  # noqa: E402

FONT_CANDIDATES = [
    '/mnt/c/Windows/Fonts/msyh.ttc',
    '/mnt/c/Windows/Fonts/simhei.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
]

STEPS = [
    ('AVTrack\ncheckpoint', '249 MiB · 12 blocks'),
    ('冻结 gate', 'skip {3,4,11} · 9 blocks'),
    ('ONNX', '429 节点 · 26.9 MB'),
    ('RKNN FP16', '15.9 MB · 全 NPU'),
    ('RK3588', '37 ms / 帧 / 核'),
    ('Tracker', '25 FPS · 3 路 64 FPS'),
]


def font(size, bold=False):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                idx = 1 if (bold and p.endswith('.ttc')) else 0
                return ImageFont.truetype(p, size, index=idx)
            except OSError:
                continue
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(P.WORK, 'assets/pipeline.png'))
    ap.add_argument('--scale', type=float, default=2.0, help='超采样倍数（保证清晰）')
    a = ap.parse_args()
    s = a.scale

    W, H = int(1500 * s), int(238 * s)
    img = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(img)

    f_title = font(int(20 * s), bold=True)
    f_step = font(int(19 * s), bold=True)
    f_meta = font(int(14 * s))
    f_arrow = font(int(13 * s))

    n = len(STEPS)
    margin = int(28 * s)
    gap = int(40 * s)
    box_w = (W - 2 * margin - gap * (n - 1)) // n
    box_h = int(104 * s)
    y0 = int(58 * s)

    fill = (244, 246, 250)
    edge = (196, 204, 216)
    accent = (46, 111, 235)
    text = (32, 36, 44)
    muted = (108, 116, 130)

    for i, (title, meta) in enumerate(STEPS):
        x0 = margin + i * (box_w + gap)
        d.rounded_rectangle([x0, y0, x0 + box_w, y0 + box_h], radius=int(12 * s),
                            fill=fill, outline=edge, width=int(1.6 * s))
        # 序号圆点
        r = int(11 * s)
        cx, cy = x0 + int(18 * s), y0 + int(18 * s)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=accent)
        d.text((cx, cy), str(i + 1), font=font(int(13 * s), bold=True), fill='white', anchor='mm')

        lines = title.split('\n')
        ty = y0 + int(44 * s)
        for ln in lines:
            d.text((x0 + box_w // 2, ty), ln, font=f_step, fill=text, anchor='mm')
            ty += int(24 * s)
        d.text((x0 + box_w // 2, y0 + box_h - int(18 * s)), meta,
               font=f_meta, fill=muted, anchor='mm')

        if i < n - 1:
            ax0 = x0 + box_w + int(2 * s)
            ax1 = x0 + box_w + gap - int(2 * s)
            ay = y0 + box_h // 2
            d.line([ax0, ay, ax1, ay], fill=accent, width=int(2 * s))
            d.polygon([(ax1, ay), (ax1 - int(7 * s), ay - int(4 * s)),
                       (ax1 - int(7 * s), ay + int(4 * s))], fill=accent)

    d.text((margin, int(18 * s)), 'AVTrack-DeiT  →  RK3588 部署链路', font=f_title, fill=text)
    d.text((W - margin, int(22 * s)), '全部数字为实测', font=f_meta, fill=muted, anchor='ra')

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    img = img.resize((W // int(s), H // int(s)), Image.LANCZOS)
    img.save(a.out, optimize=True)
    print('wrote %s  %s  %.1f KB' % (a.out, img.size, os.path.getsize(a.out) / 1024))


if __name__ == '__main__':
    main()
