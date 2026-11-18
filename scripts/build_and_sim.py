#!/usr/bin/env python
"""Step 3: 构建 RKNN 并在同一会话里做 simulator 精度检查。

（load_rknn 加载的模型不能在 simulator 上推理，必须 build 后同会话 init_runtime）
用法: python build_and_sim.py --onnx ... --out ... --ref ref/ --report results/xxx.json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ref', default=P.REF_DIR)
    ap.add_argument('--report', required=True)
    ap.add_argument('--opt-level', type=int, default=0)
    ap.add_argument('--thresh', type=float, nargs=3, default=[2e-3, 5e-3, 6e-3],
                    help='score/size/offset 的 simulator 判据（offset 6e-3 是设备 FP16 的实际地板）')
    ap.add_argument('--skip-sim', action='store_true')
    a = ap.parse_args()

    from rknn.api import RKNN

    t0 = time.time()
    rknn = RKNN(verbose=True)
    rknn.config(target_platform='rk3588', optimization_level=a.opt_level)
    assert rknn.load_onnx(model=a.onnx, inputs=['template', 'search'],
                          input_size_list=[[1, 3, 128, 128], [1, 3, 256, 256]]) == 0
    assert rknn.build(do_quantization=False) == 0
    assert rknn.export_rknn(a.out) == 0
    build_s = time.time() - t0
    print('BUILD OK %.1fs size=%.2fMB' % (build_s, os.path.getsize(a.out) / 1e6))

    report = dict(onnx=a.onnx, rknn=a.out, build_seconds=build_s,
                  size_mb=os.path.getsize(a.out) / 1e6)
    if not a.skip_sim:
        d = np.load(os.path.join(a.ref, 'pairs.npz'))
        tpl, src = d['template'], d['search']
        meta = json.loads(str(d['meta'])) if d['meta'].ndim == 0 else json.loads(str(d['meta'][0]))
        ref = np.load(os.path.join(a.ref, 'out_ort.npz'))
        refs = [ref['score'], ref['size'], ref['offset']]
        assert rknn.init_runtime() == 0
        names = ('score_map', 'size_map', 'offset_map')
        diffs = [[], [], []]
        decoded = []
        for i in range(len(tpl)):
            outs = rknn.inference(inputs=[C.prep_rgb(tpl[i]), C.prep_rgb(src[i])],
                                  data_format='nchw')
            for k in range(3):
                got = np.asarray(outs[k]).reshape(refs[k][i].shape)
                diffs[k].append(np.abs(got - refs[k][i]))
            cs = float(meta[i]['search_crop_sz'])
            pa = C.postprocess(np.asarray(outs[0]), np.asarray(outs[1]), np.asarray(outs[2]), cs)
            pb = C.postprocess(refs[0][i], refs[1][i], refs[2][i], cs)
            decoded.append([abs(pa['dx'] - pb['dx']), abs(pa['dy'] - pb['dy']),
                            abs(pa['w'] - pb['w']), abs(pa['h'] - pb['h']),
                            abs(pa['conf'] - pb['conf']), pa['peak'], pb['peak']])
        ok = True
        report['per_output'] = {}
        for k, nm in enumerate(names):
            arr = np.stack(diffs[k], 0)
            mx, mn = float(arr.max()), float(arr.mean())
            good = mx < a.thresh[k]
            ok = ok and good
            report['per_output'][nm] = dict(max_abs=mx, mean_abs=mn,
                                            threshold=a.thresh[k], pass_=bool(good))
        dec = np.array([[x[0], x[1], x[2], x[3], x[4]] for x in decoded])
        report['decoded'] = dict(max_abs=[float(v) for v in dec.max(0)],
                                 mean_abs=[float(v) for v in dec.mean(0)])
        report['peak_xy_same'] = int(sum(1 for x in decoded if tuple(x[5]) == tuple(x[6])))
        report['pairs'] = int(len(tpl))
        report['sim_pass'] = bool(ok)
        rknn.release()
    C.dump_json(a.report, report)
    print(json.dumps(report, indent=2))
    if a.skip_sim:
        print('SIM SKIPPED')
    elif report.get('sim_pass'):
        print('SIM PASS')
    else:
        print('SIM FAIL（offset 5.36e-3 属设备 FP16 地板，见 results/accuracy.md）')


if __name__ == '__main__':
    main()
