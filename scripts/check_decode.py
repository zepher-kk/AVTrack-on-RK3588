#!/usr/bin/env python
"""比较两张 PNG 的逐像素差异（板端 vs PC 解码一致性）。

用法: python check_decode.py a.png b.png
输出: shape / max|d| / mean|d| / 有差异的通道占比
"""
import sys

import cv2
import numpy as np


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a = cv2.imread(sys.argv[1])
    b = cv2.imread(sys.argv[2])
    if a is None or b is None:
        raise SystemExit('读图失败：%s / %s' % (sys.argv[1], sys.argv[2]))
    if a.shape != b.shape:
        raise SystemExit('尺寸不同：%s vs %s' % (a.shape, b.shape))
    d = a.astype(np.int16) - b.astype(np.int16)
    ad = np.abs(d)
    print('shape=%s  max|d|=%d  mean|d|=%.4f  channels_diff=%d/%d (%.1f%%)'
          % (a.shape, ad.max(), ad.mean(), int((d != 0).sum()), d.size,
             100.0 * (d != 0).sum() / d.size))
    return 0


if __name__ == '__main__':
    sys.exit(main())
