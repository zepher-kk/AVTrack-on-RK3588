#!/usr/bin/env python
"""从视频里导出一帧为 PNG（用于板端/PC 解码一致性检查）。

用法: python dump_frame.py <video> <frame_index> <out.png>
"""
import sys

import cv2


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    video, idx, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit('cannot read frame %d from %s' % (idx, video))
    cv2.imwrite(out, frame)
    print('%s shape=%s sum=%d' % (out, frame.shape, int(frame.sum())))
    return 0


if __name__ == '__main__':
    sys.exit(main())
