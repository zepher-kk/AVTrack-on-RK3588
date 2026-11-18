#!/usr/bin/env python
"""在 crop 包上跑推理（backend: ort / rknn），输出每帧解码结果 CSV。

用于"完全相同像素"的公平对照：PC 跑 ORT、板端跑 RKNN，两边的输入逐比特一致。
CSV 列: frame,dec_x,dec_y,dec_w,dec_h,conf,peak_y,peak_x,ms
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import avtrack_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['ort', 'rknn'], required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--crops', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--core', default='0')
    a = ap.parse_args()

    d = np.load(a.crops)
    tpl, src, cs = d['template'], d['search'], d['crop_sz']
    st = d['state']
    tpl_in = C.prep_rgb(tpl)

    if a.backend == 'ort':
        import onnxruntime as ort
        sess = ort.InferenceSession(a.model, providers=['CPUExecutionProvider'])
        infer = lambda t, s: sess.run(None, {'template': t, 'search': s})
    else:
        from rknnlite.api import RKNNLite
        mask = {'0': RKNNLite.NPU_CORE_0, '1': RKNNLite.NPU_CORE_1,
                '2': RKNNLite.NPU_CORE_2, '012': RKNNLite.NPU_CORE_0_1_2}[str(a.core)]
        rk = RKNNLite()
        assert rk.load_rknn(a.model) == 0
        assert rk.init_runtime(core_mask=mask) == 0
        infer = lambda t, s: [np.asarray(x) for x in
                              rk.inference(inputs=[t, s], data_format='nchw')]

    rows = []
    times = []
    for i in range(len(src)):
        t0 = time.perf_counter()
        outs = infer(tpl_in, C.prep_rgb(src[i]))
        pp = C.postprocess(outs[0], outs[1], outs[2], float(cs[i]))
        dt = (time.perf_counter() - t0) * 1000.0
        times.append(dt)
        x = st[i][0] + st[i][2] / 2.0 + pp['dx'] - pp['w'] / 2.0
        y = st[i][1] + st[i][3] / 2.0 + pp['dy'] - pp['h'] / 2.0
        rows.append((i, x, y, pp['w'], pp['h'], pp['conf'], pp['peak'][0], pp['peak'][1], dt))

    with open(a.out, 'w') as f:
        f.write('frame,x,y,w,h,conf,peak_y,peak_x,ms\n')
        for r in rows:
            f.write('%d,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d,%.6f\n' % r)
    t = np.array(times)
    print('%s frames=%d ms_p50=%.2f peak_xy=%s'
          % (a.backend, len(rows), np.percentile(t, 50),
             [(r[6], r[7]) for r in rows[:5]]))
    if a.backend == 'rknn':
        rk.release()


if __name__ == '__main__':
    main()
