#!/usr/bin/env python
"""Step 3: ONNX -> RKNN（FP16 / INT8）。

第一版不加 mean/std：直接喂 float32 已归一化数据（与 PC 参考同一份预处理）。
"""
import argparse
import os
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--opt-level', type=int, default=0)
    ap.add_argument('--quant', action='store_true')
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--log', default=None)
    ap.add_argument('--u8-input', action='store_true')
    a = ap.parse_args()

    from rknn.api import RKNN

    t0 = time.time()
    rknn = RKNN(verbose=True)
    cfg = dict(target_platform='rk3588', optimization_level=a.opt_level)
    if a.u8_input:
        cfg['mean_values'] = [[123.675, 116.28, 103.53]] * 2
        cfg['std_values'] = [[58.395, 57.12, 57.375]] * 2
    rknn.config(**cfg)

    assert rknn.load_onnx(model=a.onnx, inputs=['template', 'search'],
                          input_size_list=[[1, 3, 128, 128], [1, 3, 256, 256]]) == 0
    assert rknn.build(do_quantization=a.quant, dataset=a.dataset) == 0
    assert rknn.export_rknn(a.out) == 0
    rknn.release()

    size = os.path.getsize(a.out)
    print('OK %s  %.2f MB  %.1fs' % (a.out, size / 1e6, time.time() - t0))


if __name__ == '__main__':
    main()
