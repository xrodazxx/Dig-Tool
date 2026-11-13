"""Image network for pits and foundations: running a trained network with plain numpy.

The network is trained elsewhere (a free GPU notebook: tools/train_cnn.ipynb) and saved as
sitecheck/model_cnn.npz. Running it needs nothing but numpy and OpenCV - no PyTorch, no internet,
no AI service. Without that file the screenshot mode works exactly as before.

Architecture (fixed; the notebook builds the same one in PyTorch):
  encoder  = ResNet-18 stem + layer1 + layer2 + layer3 (stride 16), batch-norm folded into the convs
  decoder  = up x2 + layer2 skip -> conv3x3 128, up x2 + layer1 skip -> conv3x3 64, conv1x1 -> 2 maps
  output   = 2 heat maps at 1/4 of the picture size: [0] pits / hollows, [1] foundations
Input: RGB 0..1, ImageNet mean/std, size padded to a multiple of 16 (reflect).
"""
from __future__ import annotations

import os

import cv2
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cnn.npz")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
CLASSES = ("pit", "foundation")
_CACHE = {}


# --------------------------------------------------------------------------- numpy layers
def conv2d(x, w, b, stride=1, pad=0):
    """x: (C,H,W), w: (O,C,k,k), b: (O,) -> (O,H',W'). im2col + one matrix product."""
    C, H, W = x.shape
    O, _, k, _ = w.shape
    if pad:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    Hp, Wp = x.shape[1:]
    Ho, Wo = (Hp - k) // stride + 1, (Wp - k) // stride + 1
    if k == 1:
        cols = x[:, ::stride, ::stride][:, :Ho, :Wo].reshape(C, -1)
        out = w.reshape(O, C) @ cols
    else:
        win = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(1, 2))   # C,Hp-k+1,Wp-k+1,k,k
        win = win[:, ::stride, ::stride][:, :Ho, :Wo]                           # C,Ho,Wo,k,k
        cols = np.ascontiguousarray(win.transpose(0, 3, 4, 1, 2)).reshape(C * k * k, Ho * Wo)
        out = w.reshape(O, C * k * k) @ cols
    out += b[:, None]
    return out.reshape(O, Ho, Wo)


def relu(x):
    return np.maximum(x, 0, out=x)


def maxpool3s2(x):
    C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)), constant_values=-np.inf)
    win = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(1, 2))[:, ::2, ::2]
    Ho, Wo = (H + 2 - 3) // 2 + 1, (W + 2 - 3) // 2 + 1
    return win[:, :Ho, :Wo].max(axis=(3, 4))


def _up2_axis(x, axis):
    """x2 bilinear along one axis, half-pixel centres, edges clamped - exactly what
    torch.nn.functional.interpolate(scale_factor=2, mode="bilinear", align_corners=False) does."""
    prev = np.concatenate([np.take(x, [0], axis), np.take(x, range(x.shape[axis] - 1), axis)], axis)
    nxt = np.concatenate([np.take(x, range(1, x.shape[axis]), axis), np.take(x, [-1], axis)], axis)
    even = 0.75 * x + 0.25 * prev
    odd = 0.75 * x + 0.25 * nxt
    out = np.stack([even, odd], axis + 1)
    shp = list(x.shape)
    shp[axis] *= 2
    return out.reshape(shp)


def up2(x, size):
    """(C,h,w) -> (C,2h,2w); the padded input makes every level exactly half of the one above."""
    y = _up2_axis(_up2_axis(x, 1), 2)
    assert y.shape[1:] == tuple(size), (y.shape, size)
    return y.astype(np.float32)


def _block(x, P, name, stride):
    """ResNet basic block (batch norm already folded into conv weights)."""
    y = relu(conv2d(x, P[name + ".c1.w"], P[name + ".c1.b"], stride, 1))
    y = conv2d(y, P[name + ".c2.w"], P[name + ".c2.b"], 1, 1)
    if name + ".down.w" in P:
        x = conv2d(x, P[name + ".down.w"], P[name + ".down.b"], stride, 0)
    return relu(y + x)


def forward(rgb, P):
    """rgb: (H,W,3) 0..1 -> logits (2, H/4, W/4) for the padded picture, and the padding used."""
    H, W = rgb.shape[:2]
    ph, pw = (-H) % 16, (-W) % 16
    x = (rgb.astype(np.float32) - MEAN) / STD
    if ph or pw:
        x = np.pad(x, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    x = np.ascontiguousarray(x.transpose(2, 0, 1))
    x = relu(conv2d(x, P["stem.w"], P["stem.b"], 2, 3))
    x = maxpool3s2(x)
    f1 = _block(_block(x, P, "l1.0", 1), P, "l1.1", 1)                 # 64,  1/4
    f2 = _block(_block(f1, P, "l2.0", 2), P, "l2.1", 1)                # 128, 1/8
    f3 = _block(_block(f2, P, "l3.0", 2), P, "l3.1", 1)                # 256, 1/16
    d2 = np.concatenate([up2(f3, f2.shape[1:]), f2], 0)
    d2 = relu(conv2d(d2, P["dec2.w"], P["dec2.b"], 1, 1))              # 128, 1/8
    d1 = np.concatenate([up2(d2, f1.shape[1:]), f1], 0)
    d1 = relu(conv2d(d1, P["dec1.w"], P["dec1.b"], 1, 1))              # 64,  1/4
    return conv2d(d1, P["head.w"], P["head.b"], 1, 0), (ph, pw)


# --------------------------------------------------------------------------- model file
def load_model(path=None):
    """The trained network, or None (no file, or switched off with SITECHECK_NO_CNN=1)."""
    if os.environ.get("SITECHECK_NO_CNN") and path is None:
        return None
    p = path or MODEL_PATH
    if p in _CACHE:
        return _CACHE[p]
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=False)
    P = {k: z[k].astype(np.float32) for k in z.files if not k.startswith("meta.")}
    meta = {k[5:]: z[k].item() if z[k].ndim == 0 else z[k].tolist() for k in z.files if k.startswith("meta.")}
    _CACHE[p] = (P, meta)
    return _CACHE[p]


def heatmaps(rgb, model, mpp=None):
    """Probability maps (2, H, W) at the picture's own size. The picture is first brought to the
    scale the network was trained at, when the scale is known."""
    P, meta = model
    H, W = rgb.shape[:2]
    f = 1.0
    work_mpp = meta.get("work_mpp")
    if mpp and work_mpp:
        f = float(np.clip(mpp / work_mpp, 0.4, 2.5))
    img = rgb if abs(f - 1) < 0.03 else cv2.resize(rgb.astype(np.float32), (max(16, round(W * f)), max(16, round(H * f))),
                                                   interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
    logits, (ph, pw) = forward(np.clip(img, 0, 1), P)
    prob = 1 / (1 + np.exp(-logits))
    h4, w4 = (img.shape[0] + ph) // 4, (img.shape[1] + pw) // 4
    prob = prob[:, :h4, :w4]
    full = np.stack([cv2.resize(prob[c], (img.shape[1] + pw, img.shape[0] + ph), interpolation=cv2.INTER_LINEAR)
                     [:img.shape[0], :img.shape[1]] for c in range(prob.shape[0])])
    if img is not rgb:
        full = np.stack([cv2.resize(full[c], (W, H), interpolation=cv2.INTER_LINEAR) for c in range(full.shape[0])])
    return full


def detect(rgb, valid=None, mpp=None, model=None, min_p=0.08, max_n=300):
    """Peaks of the heat maps as candidates (same form as the other detectors)."""
    model = model or load_model()
    if model is None:
        return []
    hm = heatmaps(rgb, model, mpp)
    out = []
    r_px = max(3, int(round(2.0 / (mpp or 0.3))))                     # peaks at least ~2 m apart
    for ci, name in enumerate(CLASSES[:hm.shape[0]]):
        m = hm[ci]
        peak = (m == cv2.dilate(m, np.ones((2 * r_px + 1, 2 * r_px + 1), np.uint8))) & (m >= min_p)
        if valid is not None:
            peak &= valid
        ys, xs = np.nonzero(peak)
        size_m = 4.0 if name == "pit" else 10.0          # nominal: the network gives a place, not a size
        half = size_m / 2 / (mpp or 0.3)
        for y, x in zip(ys, xs):
            p = float(m[y, x])
            c = {"type": "pit" if name == "pit" else "rectangle", "kind": "cnn_" + name,
                 "row": float(y), "col": float(x), "confidence": round(p, 3), "cnn_p": round(p, 3),
                 "diameter_m": size_m, "size_nominal": True, "detected_by": "image network",
                 "note": f"image network: {name} score {p:.2f} (size drawn is nominal)"}
            if name != "pit":
                c["corners_px"] = [[x - half, y - half], [x + half, y - half], [x + half, y + half], [x - half, y + half]]
            out.append(c)
    out.sort(key=lambda c: -c["confidence"])
    return out[:max_n]


def merge(cands, cnn_c, mpp, w_old=0.2, w_new=0.9, demote=0.55, add=0.5):
    """The network re-ranks what the other detectors found (same rule as the tree model): a
    candidate the network also marks takes mostly the network's score; one it does not mark is
    demoted; network-only spots are added at half their score. Weights are chosen in the
    training notebook on held-out places and stored in the model file."""
    import math
    used = set()
    for c in cands:
        if c.get("type") == "linear" or c.get("kind") in ("crop_mark", "soil_mark", "clearing", "canopy_anomaly"):
            continue
        best, bj = None, None
        for j, l in enumerate(cnn_c):
            d = math.hypot(c["row"] - l["row"], c["col"] - l["col"])
            if d <= max(8.0, 2.5 / (mpp or 0.3)) and (best is None or d < best):
                best, bj = d, j
        if bj is None:
            c["cnn_p"] = 0.0
            c["confidence"] = round(c["confidence"] * demote, 3)
        else:
            used.add(bj)
            p = cnn_c[bj]["cnn_p"]
            c["cnn_p"] = p
            c["confidence"] = round(min(1.0, w_old * c["confidence"] + w_new * p), 3)
    for j, l in enumerate(cnn_c):
        if j not in used:
            l = dict(l)
            l["confidence"] = round(add * l["cnn_p"], 3)
            cands.append(l)
    return cands


def random_model(seed=0):
    """Random weights with the right shapes (tests only)."""
    rng = np.random.default_rng(seed)
    P = {}

    def cw(name, o, i, k):
        P[name + ".w"] = (rng.standard_normal((o, i, k, k)) * np.sqrt(2 / (i * k * k))).astype(np.float32)
        P[name + ".b"] = np.zeros(o, np.float32)
    cw("stem", 64, 3, 7)
    for li, (cin, cout) in (("l1", (64, 64)), ("l2", (64, 128)), ("l3", (128, 256))):
        for bi in (0, 1):
            i = cin if bi == 0 else cout
            cw(f"{li}.{bi}.c1", cout, i, 3)
            cw(f"{li}.{bi}.c2", cout, cout, 3)
            if bi == 0 and li != "l1":
                cw(f"{li}.{bi}.down", cout, i, 1)
    cw("dec2", 128, 256 + 128, 3)
    cw("dec1", 64, 128 + 64, 3)
    cw("head", 2, 64, 1)
    return P, {"work_mpp": 0.3}
