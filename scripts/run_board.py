#!/usr/bin/env python
"""Step 4: 板端单帧打通（20 组参考 + 100 帧稳定性 + 内存泄漏检查）。

板端运行:
  python run_board.py --rknn build/xxx.rknn --ref ref --iters 100
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import avtrack_common as C  # noqa: E402
import paths as P  # noqa: E402


def meminfo():
    out = {}
    with open('/proc/meminfo') as f:
        for line in f:
            k, v = line.split(':', 1)
            if k in ('MemTotal', 'MemFree', 'MemAvailable', 'CmaTotal', 'CmaFree'):
                out[k] = int(v.strip().split()[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rknn', required=True)
    ap.add_argument('--ref', default='ref')
    ap.add_argument('--iters', type=int, default=100)
    ap.add_argument('--core', default='0')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    from rknnlite.api import RKNNLite

    mask_map = {'0': RKNNLite.NPU_CORE_0, '1': RKNNLite.NPU_CORE_1,
                '2': RKNNLite.NPU_CORE_2, '012': RKNNLite.NPU_CORE_0_1_2}
    mask = mask_map[str(a.core)]

    d = np.load(os.path.join(a.ref, 'pairs.npz'))
    tpl, src = d['template'], d['search']
    meta = json.loads(str(d['meta'])) if d['meta'].ndim == 0 else json.loads(str(d['meta'][0]))
    ref = np.load(os.path.join(a.ref, 'out_ort.npz'))
    refs = [ref['score'], ref['size'], ref['offset']]

    t0 = time.time()
    rk = RKNNLite()
    assert rk.load_rknn(a.rknn) == 0, 'load_rknn failed'
    assert rk.init_runtime(core_mask=mask) == 0, 'init_runtime failed'
    init_s = time.time() - t0
    print('init_runtime %.2fs core=%s' % (init_s, a.core))

    inputs = [(C.prep_rgb(tpl[i]), C.prep_rgb(src[i])) for i in range(len(tpl))]

    # warmup
    t = time.time()
    rk.inference(inputs=list(inputs[0]), data_format='nchw')
    print('first inference %.1f ms' % ((time.time() - t) * 1000))

    names = ('score_map', 'size_map', 'offset_map')
    diffs = [[], [], []]
    decoded = []
    for i in range(len(inputs)):
        outs = rk.inference(inputs=list(inputs[i]), data_format='nchw')
        for k in range(3):
            got = np.asarray(outs[k]).reshape(refs[k][i].shape)
            diffs[k].append(np.abs(got - refs[k][i]))
        if i == 0:
            for nm, o in zip(names, outs):
                arr = np.asarray(o)
                print('%-10s shape=%s min=%.6f max=%.6f nonzero=%d/%d'
                      % (nm, arr.shape, arr.min(), arr.max(),
                         np.count_nonzero(arr), arr.size))
        cs = float(meta[i]['search_crop_sz'])
        pa = C.postprocess(outs[0], outs[1], outs[2], cs)
        pb = C.postprocess(refs[0][i], refs[1][i], refs[2][i], cs)
        decoded.append([abs(pa['dx'] - pb['dx']), abs(pa['dy'] - pb['dy']),
                        abs(pa['w'] - pb['w']), abs(pa['h'] - pb['h']),
                        abs(pa['conf'] - pb['conf']), pa['peak'], pb['peak']])

    report = dict(rknn=a.rknn, core=str(a.core), init_seconds=init_s,
                  pairs=len(inputs), per_output={})
    for k, nm in enumerate(names):
        arr = np.stack(diffs[k], 0)
        report['per_output'][nm] = dict(max_abs=float(arr.max()), mean_abs=float(arr.mean()))
    dec = np.array([[x[0], x[1], x[2], x[3], x[4]] for x in decoded])
    report['decoded'] = dict(max_abs=[float(v) for v in dec.max(0)],
                             mean_abs=[float(v) for v in dec.mean(0)])
    report['peak_xy_same'] = int(sum(1 for x in decoded if tuple(x[5]) == tuple(x[6])))

    # 100 帧稳定性 + 内存
    m0 = meminfo()
    lat = []
    for _ in range(a.iters):
        t = time.time()
        rk.inference(inputs=list(inputs[0]), data_format='nchw')
        lat.append((time.time() - t) * 1000)
    m1 = meminfo()
    lat = np.array(lat)
    report['stability'] = dict(iters=int(a.iters),
                               mem_before=m0, mem_after=m1,
                               mem_available_delta_kb=m1['MemAvailable'] - m0['MemAvailable'],
                               npu_ms_p50=float(np.percentile(lat, 50)),
                               npu_ms_p90=float(np.percentile(lat, 90)))
    rk.release()

    print(json.dumps(report, indent=2))
    if a.out:
        C.dump_json(a.out, report)
    print('BOARD NUM OK' if report['per_output']['offset_map']['max_abs'] < 2e-2 else 'BOARD NUM CHECK')


if __name__ == '__main__':
    main()
