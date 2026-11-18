#!/usr/bin/env python
"""Step 6: 无 GUI 跟踪 runner（后端可切 rknn / ort / torch），输出 CSV+JSON。

板端:
  python track_headless.py --backend rknn --model build/x.rknn \
      --video videos/man.mp4 --bbox 80 30 145 190 --out results/man_rknn --frames 300
PC:
  python track_headless.py --backend ort --model onnx/x.onnx --video ... --out ...
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import avtrack_common as C  # noqa: E402
import paths as P  # noqa: E402


class RknnBackend(object):
    name = 'rknn'

    def __init__(self, model, core='0'):
        from rknnlite.api import RKNNLite
        mask = {'0': RKNNLite.NPU_CORE_0, '1': RKNNLite.NPU_CORE_1,
                '2': RKNNLite.NPU_CORE_2, '012': RKNNLite.NPU_CORE_0_1_2}[str(core)]
        self.rk = RKNNLite()
        assert self.rk.load_rknn(model) == 0
        assert self.rk.init_runtime(core_mask=mask) == 0

    def infer(self, tpl_in, src_in):
        o = self.rk.inference(inputs=[tpl_in, src_in], data_format='nchw')
        return [np.asarray(x) for x in o]


class OrtBackend(object):
    name = 'ort'

    def __init__(self, model, core='0'):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(model, providers=['CPUExecutionProvider'])

    def infer(self, tpl_in, src_in):
        return self.sess.run(None, {'template': tpl_in, 'search': src_in})


class TorchBackend(object):
    name = 'torch'

    def __init__(self, model, core='0'):
        self.net, _ = C.build_model(P.SRC, P.CFG, P.CKPT, frozen=True)

    def infer(self, tpl_in, src_in):
        return C.model_forward(self.net, tpl_in, src_in)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['rknn', 'ort', 'torch'], required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--video', required=True)
    ap.add_argument('--bbox', type=float, nargs=4, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--frames', type=int, default=300)
    ap.add_argument('--core', default='0')
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--state-file', default=None,
                    help='开环模式：用该 CSV(frame,x,y,w,h,...) 的状态做 crop，不更新状态，'
                         '只记录每帧解码结果（用于隔离单帧误差与闭环漂移）')
    ap.add_argument('--noise', type=int, default=0,
                    help='给输入帧加 ±n 的均匀像素噪声（模拟板端/PC 解码差异，做敏感度实验）')
    a = ap.parse_args()
    rng = np.random.default_rng(1234) if a.noise > 0 else None

    forced = None
    if a.state_file:
        arr = np.loadtxt(a.state_file, delimiter=',', skiprows=1)
        forced = {int(r[0]): [float(v) for v in r[1:5]] for r in np.atleast_2d(arr)}

    be = {'rknn': RknnBackend, 'ort': OrtBackend, 'torch': TorchBackend}[a.backend](a.model, a.core)

    cap = cv2.VideoCapture(a.video)
    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError('cannot read %s' % a.video)
    h_img, w_img = frame0.shape[:2]
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    state = [float(v) for v in a.bbox]

    tpl_bgr, _ = C.crop_hwc(frame0, state, C.TPL_SIZE, C.TPL_FACTOR)
    tpl_in = C.prep_bgr(tpl_bgr)

    writer = None
    os.makedirs(a.out, exist_ok=True)
    if not a.no_video:
        vpath = os.path.join(a.out, '%s_%s.mp4' % (os.path.basename(a.video).split('.')[0], be.name))
        writer = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*'mp4v'), fps_in, (w_img, h_img))

    # warmup
    src_bgr, _ = C.crop_hwc(frame0, state, C.SRC_SIZE, C.SRC_FACTOR)
    src_in = C.prep_bgr(src_bgr)
    for _ in range(a.warmup):
        be.infer(tpl_in, src_in)

    rows = []
    times = []
    frame = frame0
    for fi in range(a.frames):
        t0 = time.perf_counter()
        if fi > 0:
            ok, frame = cap.read()
            if not ok:
                break
        if forced is not None:
            state = list(forced.get(fi, state))
        if rng is not None:
            n = rng.integers(-a.noise, a.noise + 1, frame.shape, dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + n, 0, 255).astype(np.uint8)
        crop_bgr, crop_sz = C.crop_hwc(frame, state, C.SRC_SIZE, C.SRC_FACTOR)
        src_in = C.prep_bgr(crop_bgr)
        outs = be.infer(tpl_in, src_in)
        pp = C.postprocess(outs[0], outs[1], outs[2], crop_sz)
        if forced is None:
            C.update_state(state, pp, w_img, h_img)
        dt = (time.perf_counter() - t0) * 1000.0
        times.append(dt)
        # 开环模式下把"解码结果"写进 csv：x+dx, y+dy, w, h（便于逐帧对照）
        if forced is None:
            out_state = state
        else:
            out_state = [state[0] + state[2] / 2.0 + pp['dx'] - pp['w'] / 2.0,
                         state[1] + state[3] / 2.0 + pp['dy'] - pp['h'] / 2.0,
                         pp['w'], pp['h']]
        rows.append([fi, out_state[0], out_state[1], out_state[2], out_state[3], pp['conf'], dt])
        if writer is not None:
            x1, y1 = int(round(state[0])), int(round(state[1]))
            x2, y2 = int(round(state[0] + state[2])), int(round(state[1] + state[3]))
            vis = cv2.rectangle(frame.copy(), (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, '%.2f' % pp['conf'], (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            writer.write(vis)
        if fi % 50 == 0:
            print('  frame %03d state=[%.1f %.1f %.1f %.1f] conf=%.3f %.1fms'
                  % (fi, state[0], state[1], state[2], state[3], pp['conf'], dt))
    cap.release()
    if writer is not None:
        writer.release()

    csv_path = os.path.join(a.out, 'track.csv')
    with open(csv_path, 'w') as f:
        f.write('frame,x,y,w,h,conf,ms\n')
        for r in rows:
            f.write('%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n' % tuple(r))

    times = np.array(times)
    summary = dict(backend=be.name, model=a.model, video=a.video, bbox=list(a.bbox),
                   frames=len(rows), ms_p50=float(np.percentile(times, 50)),
                   ms_p90=float(np.percentile(times, 90)), ms_mean=float(times.mean()),
                   fps=float(1000.0 / times.mean()),
                   conf_min=float(min(r[5] for r in rows)),
                   conf_mean=float(np.mean([r[5] for r in rows])),
                   final_state=[float(v) for v in state],
                   first_state=[float(v) for v in rows[0][1:5]] if rows else None)
    C.dump_json(os.path.join(a.out, 'summary.json'), summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
