#!/usr/bin/env python
"""导出 AVTrack gate9 ONNX（冻结跳块表 + 严格加载 + CPU + ORT 自检）。

示例:
  python export_avtrack.py --source /path/to/AVTrack \
      --config experiments/avtrack/deit_tiny_patch16_224.yaml \
      --checkpoint output/checkpoints/train/avtrack/deit_tiny_patch16_224/AVTrack_ep0300.pth.tar \
      --opset 17 --output ../onnx/avtrack_gate9_op17.onnx
"""
import argparse
import collections
import contextlib
import hashlib
import importlib
import io
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import avtrack_common as C  # noqa: E402
import paths as P  # noqa: E402


def manual_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    """与 export_onnx.py 等价（用 math.sqrt 写法，避免多余的 Pow/Reciprocal 常量节点）。"""
    import math
    import torch
    sf = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    w = query @ key.transpose(-2, -1) * sf
    if is_causal:
        raise NotImplementedError('is_causal not used by AVTrack')
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            w = w.masked_fill(~attn_mask, float('-inf'))
        else:
            w = w + attn_mask
    return torch.softmax(w, dim=-1) @ value


def onnx_block_set(path):
    import onnx
    m = onnx.load(str(path))
    idx = set()
    for n in m.graph.node:
        for t in list(n.input) + list(n.output):
            for part in t.split('/'):
                mt = re.match(r'^blocks[._](\d+)', part)
                if mt:
                    idx.add(int(mt.group(1)))
    return idx


def onnx_ops(path):
    import onnx
    return collections.Counter(n.op_type for n in onnx.load(str(path)).graph.node)


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def dummy_inputs(kind):
    """两种 dummy：灰图（有真实统计）与 randn（最坏情况）。"""
    if kind == 'gray':
        g = np.full((256, 256, 3), 128, np.uint8)
        return C.prep_rgb(g[64:192, 64:192].copy()), C.prep_rgb(g)
    rng = np.random.default_rng(0)
    return (C.prep_rgb(rng.integers(0, 256, (128, 128, 3), np.uint8)),
            C.prep_rgb(rng.integers(0, 256, (256, 256, 3), np.uint8)))


def export_once(model, tpl, src, out_path, opset):
    import torch
    w = C.ExportWrapper.make(model).eval()
    with contextlib.redirect_stdout(io.StringIO()):
        torch.onnx.export(
            w, (torch.from_numpy(tpl.copy()), torch.from_numpy(src.copy())), str(out_path),
            opset_version=opset, do_constant_folding=True,
            input_names=['template', 'search'],
            output_names=['score_map', 'size_map', 'offset_map'])
    return str(out_path)


def ort_run(path, tpl, src):
    import onnxruntime as ort
    s = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    return s.run(None, {'template': tpl, 'search': src})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default=P.SRC)
    ap.add_argument('--config', default='experiments/avtrack/deit_tiny_patch16_224.yaml')
    ap.add_argument('--checkpoint',
                    default='output/checkpoints/train/avtrack/deit_tiny_patch16_224/AVTrack_ep0300.pth.tar')
    ap.add_argument('--opset', type=int, default=17)
    ap.add_argument('--output', required=True)
    ap.add_argument('--report', default=None)
    a = ap.parse_args()

    import onnx
    import torch
    torch.nn.functional.scaled_dot_product_attention = manual_sdpa

    src_root = Path(a.source).resolve()
    cfg_path = Path(a.config)
    if not cfg_path.is_absolute():
        cfg_path = src_root / a.config
    ckpt = Path(a.checkpoint)
    if not ckpt.is_absolute():
        ckpt = src_root / a.checkpoint

    model, info = C.build_model(str(src_root), cfg_path, ckpt, frozen=True)
    print('load_state_dict:', info)
    print('blocks:', len(model.backbone.blocks), 'frozen run set:', sorted(C.RUN_BLOCKS))

    out_path = Path(a.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tpl_g, src_g = dummy_inputs('gray')
    tpl_r, src_r = dummy_inputs('rand')

    path_g = export_once(model, tpl_g, src_g, out_path, a.opset)
    tmp_r = str(out_path.with_name(out_path.stem + '_randn' + out_path.suffix))
    path_r = export_once(model, tpl_r, src_r, tmp_r, a.opset)

    blocks = onnx_block_set(path_g)
    ops = onnx_ops(path_g)
    expect = set(C.RUN_BLOCKS)
    assert blocks == expect, 'GATE NOT FROZEN: %s != %s' % (sorted(blocks), sorted(expect))

    # 两次不同 dummy 导出的图必须等价
    ref_g = ort_run(path_g, tpl_g, src_g)
    ref_r = ort_run(path_r, tpl_g, src_g)
    det = [float(np.abs(x - y).max()) for x, y in zip(ref_g, ref_r)]

    pt = C.model_forward(model, tpl_g, src_g)
    errs = [float(np.abs(x - y).max()) for x, y in zip(pt, ref_g)]

    names = ('score_map', 'size_map', 'offset_map')
    report = {
        'onnx': str(out_path),
        'md5': md5(path_g),
        'opset': a.opset,
        'checkpoint': str(ckpt),
        'load_state_dict': str(info),
        'blocks': sorted(blocks),
        'nodes': int(sum(ops.values())),
        'op_hist': dict(sorted(ops.items())),
        'layernorm': int(ops.get('LayerNormalization', 0)),
        'softmax': int(ops.get('Softmax', 0)),
        'matmul': int(ops.get('MatMul', 0)),
        'torch_vs_ort_max_abs': dict(zip(names, errs)),
        'two_dummy_delta_max_abs': dict(zip(names, det)),
        'outputs': {},
    }
    for nm, v in zip(names, ref_g):
        v = np.asarray(v)
        report['outputs'][nm] = dict(shape=list(v.shape), min=float(v.min()),
                                     max=float(v.max()),
                                     nonzero=int(np.count_nonzero(v)), size=int(v.size))
    try:
        onnx.checker.check_model(str(out_path))
        report['onnx_checker'] = True
    except Exception as e:                                  # noqa: BLE001
        report['onnx_checker'] = 'FAIL: %s' % e

    os.remove(tmp_r)
    rep = Path(a.report) if a.report else out_path.with_suffix('.json')
    C.dump_json(str(rep), report)
    print(json.dumps(report, indent=2, ensure_ascii=False)[:2000])
    print('REPORT ->', rep)


if __name__ == '__main__':
    main()
