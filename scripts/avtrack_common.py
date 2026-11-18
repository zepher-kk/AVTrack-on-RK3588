"""AVTrack RK3588 部署公共模块（PC 与板端共用，唯一实现）。

- 冻结 gate 表（RUN_BLOCKS / SKIP_BLOCKS）是部署语义，见 DEPLOY_RK3588.md §0.2
- 预处理/后处理与 inference_avtrack_trt.py 保持一致
- torch 相关 import 全部放在函数内，板端（无 torch）也能 import 本模块

用法（PC）:
    from avtrack_common import build_model, freeze_gate, prep_rgb, crop_hwc, postprocess
用法（板端）:
    from avtrack_common import prep_rgb, crop_hwc, postprocess   # 只需要 numpy+cv2
"""
import json
import math

import cv2
import numpy as np

# ---------------------------------------------------------------- 冻结 gate 表
RUN_BLOCKS = (0, 1, 2, 5, 6, 7, 8, 9, 10)
SKIP_BLOCKS = (3, 4, 11)
GATE_LOGIT = 50.0

# ---------------------------------------------------------------- 输入约定
TPL_SIZE = 128
SRC_SIZE = 256
TPL_FACTOR = 2.0
SRC_FACTOR = 4.0
FEAT_SZ = 16.0
CONF_THRESH = 0.15

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

_HANN = None


# ================================================================= 模型构建
def freeze_gate(model, run_blocks=RUN_BLOCKS, logit=GATE_LOGIT):
    """把每块 active_score_module.forward 换成常量输出。

    只替换实例方法，不新增/不删参数，state_dict 仍与 checkpoint 严格匹配。
    """
    import torch

    blocks = model.backbone.blocks
    assert len(blocks) == 12, 'expect 12 blocks, got %d' % len(blocks)
    assert set(run_blocks) | set(SKIP_BLOCKS) == set(range(len(blocks)))
    for i, blk in enumerate(blocks):
        lin = blk.active_score_module
        const = float(logit) if i in run_blocks else -float(logit)

        def fwd(x, _c=const, _lin=lin):
            return torch.full((x.shape[0], 1), _c, dtype=_lin.weight.dtype,
                              device=x.device)

        lin.forward = fwd
    return model


def build_model(source, cfg_path, ckpt, frozen=True):
    """构建 AVTrack：关掉联网下载、strict 加载、可选冻结 gate。返回 eval 模型。"""
    import importlib
    import sys

    import torch

    sys.path.insert(0, source)
    torch.set_num_threads(4)

    cfgmod = importlib.import_module('lib.config.avtrack.config')
    cfgmod.update_config_from_file(str(cfg_path))
    module = importlib.import_module('lib.models.avtrack.avtrack')

    # 全量 checkpoint 已含所有参数，禁止联网下载 DeiT 预训练权重
    orig = module.deit_tiny_patch16_224
    module.deit_tiny_patch16_224 = lambda *a, **k: orig(*a, **dict(k, pretrained=False))

    model = module.build_avtrack(cfgmod.cfg, training=False).cpu()
    state = torch.load(str(ckpt), map_location='cpu')
    info = model.load_state_dict(state.get('net', state), strict=True)
    model.eval()
    if frozen:
        freeze_gate(model)
        model.eval()
    return model, info


class ExportWrapper(object):
    """ONNX/参考输出用的 wrapper（延迟 import torch 不方便，这里用工厂）。"""

    @staticmethod
    def make(net):
        import torch

        class _W(torch.nn.Module):
            def __init__(self, net):
                super(_W, self).__init__()
                self.net = net

            def forward(self, template, search):
                anno = template.new_zeros((template.shape[0], 4))
                out = self.net(template, search, anno, anno)
                return out['score_map'], out['size_map'], out['offset_map']

        return _W(net)


def model_forward(net, tpl_f32, src_f32):
    """PyTorch 前向，返回 numpy 三路输出（tpl/src 为 NCHW float32）。"""
    import torch

    with torch.no_grad():
        w = ExportWrapper.make(net)
        outs = w(torch.from_numpy(np.ascontiguousarray(tpl_f32)),
                 torch.from_numpy(np.ascontiguousarray(src_f32)))
        return [o.detach().cpu().numpy() for o in outs]


# ================================================================= 预处理
def crop_hwc(image, bbox, out_sz, factor):
    """与 inference_avtrack_trt.py 完全一致。bbox=[x,y,w,h]。返回 (crop, crop_sz)。

    image 的通道顺序原样保留（视频是 BGR，用 prep_bgr；npz 里是 RGB，用 prep_rgb）。
    """
    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    crop_sz = np.ceil(np.sqrt(w * h) * factor)
    crop_sz = max(crop_sz, 10.0)
    x1 = int(round(cx - crop_sz / 2.0))
    y1 = int(round(cy - crop_sz / 2.0))
    x2 = int(x1 + round(crop_sz))
    y2 = int(y1 + round(crop_sz))

    h_img, w_img = image.shape[:2]
    x1p, y1p = max(0, -x1), max(0, -y1)
    x2p, y2p = max(0, x2 - w_img), max(0, y2 - h_img)
    im = image[y1 + y1p:y2 - y2p, x1 + x1p:x2 - x2p]
    if x1p or x2p or y1p or y2p:
        im = cv2.copyMakeBorder(im, y1p, y2p, x1p, x2p, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    if im.shape[0] == 0 or im.shape[1] == 0:
        return np.zeros((out_sz, out_sz, 3), dtype=np.uint8), float(crop_sz)
    return cv2.resize(im, (out_sz, out_sz)), float(crop_sz)


def prep_rgb(rgb_u8):
    """(H,W,3) uint8 RGB -> (1,3,H,W) float32 归一化 NCHW。"""
    x = rgb_u8.astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None]
    return np.ascontiguousarray((x - MEAN) / STD)


def prep_bgr(bgr_u8):
    return prep_rgb(cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2RGB))


# ================================================================= 后处理
def hann16():
    global _HANN
    if _HANN is None:
        h = np.hanning(16)
        _HANN = np.outer(h, h).astype(np.float32)
    return _HANN


def postprocess(score_map, size_map, offset_map, crop_sz, search_size=SRC_SIZE):
    """与 inference_avtrack_trt.py 的 postprocess 等价。

    返回 dict: dx, dy, w, h, conf, peak(y,x), score, crop_sz
        dx/dy = 中心相对上一帧的像素位移（未加）；w/h = 新的目标尺寸（像素）
        conf  = 汉宁窗惩罚之前的原始响应值
    """
    score = np.asarray(score_map, dtype=np.float32).reshape(16, 16)
    size = np.asarray(size_map, dtype=np.float32).reshape(2, 16, 16)
    offset = np.asarray(offset_map, dtype=np.float32).reshape(2, 16, 16)

    score_final = score * hann16()
    idx = int(np.argmax(score_final))
    y_idx, x_idx = np.unravel_index(idx, (16, 16))
    conf = float(score[y_idx, x_idx])

    w_raw = float(size[0, y_idx, x_idx])   # Ch0 = W
    h_raw = float(size[1, y_idx, x_idx])   # Ch1 = H
    off_x = float(offset[0, y_idx, x_idx])
    off_y = float(offset[1, y_idx, x_idx])

    cx_rel = (x_idx + off_x) / FEAT_SZ
    cy_rel = (y_idx + off_y) / FEAT_SZ
    cx_pix, cy_pix = cx_rel * search_size, cy_rel * search_size
    w_pix, h_pix = w_raw * search_size, h_raw * search_size

    scale = crop_sz / float(search_size)
    return dict(
        dx=(cx_pix - search_size / 2.0) * scale,
        dy=(cy_pix - search_size / 2.0) * scale,
        w=w_pix * scale,
        h=h_pix * scale,
        conf=conf,
        peak=(int(y_idx), int(x_idx)),
        score=score,
        crop_sz=float(crop_sz),
    )


def update_state(state, pp, w_img, h_img, thresh=CONF_THRESH):
    """state=[x,y,w,h]；返回更新后的 state（原地修改并返回）。"""
    x, y, w, h = state
    if pp['conf'] > thresh:
        cx = x + w / 2.0 + pp['dx']
        cy = y + h / 2.0 + pp['dy']
        w = w * 0.8 + pp['w'] * 0.2
        h = h * 0.8 + pp['h'] * 0.2
        x = cx - w / 2.0
        y = cy - h / 2.0
    w = min(w, float(w_img))
    h = min(h, float(h_img))
    x = min(max(x, 0.0), float(w_img) - w)
    y = min(max(y, 0.0), float(h_img) - h)
    state[:] = [x, y, w, h]
    return state


# ================================================================= 杂项
def read_roi(txt_path):
    """读取 '<name> x y w h' 格式的 ROI 文本。"""
    with open(txt_path) as f:
        line = f.readline().strip()
    parts = line.split()
    name = parts[0]
    bbox = [float(v) for v in parts[1:5]]
    return name, bbox


def bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix = max(0.0, min(ax2, bx2) - max(ax, bx))
    iy = max(0.0, min(ay2, by2) - max(ay, by))
    inter = ix * iy
    return inter / max(aw * ah + bw * bh - inter, 1e-9)


def center_dist(a, b):
    return math.hypot(a[0] + a[2] / 2.0 - b[0] - b[2] / 2.0,
                      a[1] + a[3] / 2.0 - b[1] - b[3] / 2.0)


def dump_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
