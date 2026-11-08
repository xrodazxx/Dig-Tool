"""Detector learned from the user's marked examples (no AI service, a small statistical model).

1. candidates: dark and light round spots at several scales (Laplacian of Gaussian)
2. features: simple, interpretable measurements around each spot, all relative to the
   spot's own size and to the local surroundings (so they work at any zoom):
     centre / rim / surroundings brightness and colour, roughness, light-shadow pair and
     its direction relative to the sun, land cover, how many similar spots are nearby
3. model: logistic regression on those features (weights stored in a small JSON file),
   trained and validated area-by-area on the marked examples (see tools/train_model.py)

The model's output is a probability-like score in 0..1 that the spot is the kind of thing the
user marked (house pit, house remains, soil patch of a house) - it is only as good as the
examples it was trained on.
"""
from __future__ import annotations

import json
import math
import os

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import blob_log

from . import imagery as im

SCALES_PX = (3.0, 4.5, 6.5, 9.5, 14.0, 20.0)       # spot radius in pixels
FEATURES = None                                     # filled by feature_names()


def _disk(r):
    r = max(1.0, r)
    n = int(math.ceil(r))
    y, x = np.mgrid[-n:n + 1, -n:n + 1]
    k = (x * x + y * y <= r * r).astype(np.float32)
    return k / k.sum()


def _ring(r0, r1):
    n = int(math.ceil(r1))
    y, x = np.mgrid[-n:n + 1, -n:n + 1]
    d2 = x * x + y * y
    k = ((d2 > r0 * r0) & (d2 <= r1 * r1)).astype(np.float32)
    return k / max(k.sum(), 1)


def _conv(img, k):
    return cv2.filter2D(img, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT)


def channels(rgb):
    L = im.luminance(rgb).astype(np.float32)
    lab = cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
    exg = (2 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]).astype(np.float32)
    mx, mn = rgb.max(-1), rgb.min(-1)
    sat = ((mx - mn) / (mx + 1e-6)).astype(np.float32)
    return {"L": L, "a": lab[..., 1] / 100, "b": lab[..., 2] / 100, "g": exg, "s": sat}


def feature_names():
    names = []
    for ch in ("L", "a", "b", "g", "s"):
        names += [f"{ch}_centre_vs_rim", f"{ch}_rim_vs_out", f"{ch}_centre_vs_out"]
    names += ["L_centre_z", "L_rim_z", "rough_in", "rough_out", "rough_ratio",
              "dipole", "dipole_sun", "log_resp", "log_rank", "radius_px", "dark_spot",
              "near_same", "nn_dist_r", "snow", "forest", "field", "open", "bright_core", "image_contrast",
              "L_abs", "b_abs", "g_abs"]
    return names


def candidates_and_features(rgb, valid=None, sun_az=None, lat=None, max_per_polarity=220):
    """Returns (cands, X): cands = [{row, col, radius_px, dark}], X = (n, n_features)."""
    if valid is None:
        valid = rgb.max(-1) > 0.02
    H, W = valid.shape
    ch = channels(rgb)
    L = ch["L"]
    Lv = np.where(valid, L, np.median(L[valid]) if valid.any() else 0.5).astype(np.float32)
    ch["L"] = Lv
    mpp_guess = 0.3
    cls = im.classify(rgb, valid, mpp_guess)
    s = im.shading(rgb, cls, mpp_guess)
    est = im.estimate_sun(s, cls["open"], mpp_guess)
    az = sun_az if sun_az is not None else im.plausible_sun(est[0], lat)[0]
    sun_vec = None if az is None else np.array([math.sin(math.radians(az)), -math.cos(math.radians(az))])  # x, y(image)

    # ---- candidate spots (both polarities) on a contrast-normalised image
    Lg = cv2.GaussianBlur(Lv, (0, 0), 1.0)
    loc_mean = cv2.GaussianBlur(Lg, (0, 0), 25)
    loc_sd = np.sqrt(np.maximum(cv2.GaussianBlur((Lg - loc_mean) ** 2, (0, 0), 25), 1e-6))
    Z = (Lg - loc_mean) / loc_sd
    cands = []
    for dark in (True, False):
        img = np.clip(-Z if dark else Z, 0, None)
        img[~valid] = 0
        bl = blob_log(img, min_sigma=SCALES_PX[0] / 1.414, max_sigma=SCALES_PX[-1] / 1.414,
                      num_sigma=len(SCALES_PX), threshold=0.05, overlap=0.5)
        if len(bl) == 0:
            continue
        resp = img[bl[:, 0].astype(int), bl[:, 1].astype(int)]
        order = np.argsort(-resp)[:max_per_polarity]
        for k in order:
            y, x, sg = bl[k]
            cands.append({"row": float(y), "col": float(x), "radius_px": float(sg * 1.414), "dark": dark,
                          "log_resp": float(resp[k])})
    if not cands:
        return [], np.zeros((0, len(feature_names())), np.float32)

    # ---- per-scale disk / ring maps
    maps = {}
    for si, r in enumerate(SCALES_PX):
        kd, kr, ko = _disk(r), _ring(r, 1.8 * r), _ring(1.8 * r, 3.0 * r)
        for name, c in ch.items():
            maps[(name, si, "d")] = _conv(c, kd)
            maps[(name, si, "r")] = _conv(c, kr)
            maps[(name, si, "o")] = _conv(c, ko)
        sq_in = _conv(Lv * Lv, kd) - maps[("L", si, "d")] ** 2
        k3 = _disk(3 * r)
        m3 = _conv(Lv, k3)
        sq_out = _conv(Lv * Lv, k3) - m3 ** 2
        maps[("rough", si, "in")] = np.sqrt(np.maximum(sq_in, 0))
        maps[("rough", si, "out")] = np.sqrt(np.maximum(sq_out, 0))
        # first moment of brightness in a 2r disk: light-shadow pair
        n = int(math.ceil(2 * r))
        yy, xx = np.mgrid[-n:n + 1, -n:n + 1].astype(np.float32)
        m = (xx * xx + yy * yy <= 4 * r * r).astype(np.float32)
        kx, ky = xx * m / max(m.sum() * r, 1), yy * m / max(m.sum() * r, 1)
        maps[("dip", si, "x")] = _conv(Lv, kx)
        maps[("dip", si, "y")] = _conv(Lv, ky)

    img_contrast = float(np.std(Lv[valid])) if valid.any() else 0.1
    sd_map = loc_sd
    pts = np.array([[c["row"], c["col"]] for c in cands])
    darks = np.array([c["dark"] for c in cands])
    rads = np.array([c["radius_px"] for c in cands])
    resp_all = np.array([c["log_resp"] for c in cands])
    X = []
    for i, c in enumerate(cands):
        r_, c_ = int(np.clip(c["row"], 0, H - 1)), int(np.clip(c["col"], 0, W - 1))
        si = int(np.argmin([abs(c["radius_px"] - r) for r in SCALES_PX]))
        sd = float(sd_map[r_, c_]) + 1e-4
        f = []
        for name in ("L", "a", "b", "g", "s"):
            d, rr, o = (maps[(name, si, k)][r_, c_] for k in ("d", "r", "o"))
            norm = sd if name == "L" else 1.0
            f += [(d - rr) / norm, (rr - o) / norm, (d - o) / norm]
        rin, rout = maps[("rough", si, "in")][r_, c_], maps[("rough", si, "out")][r_, c_]
        dx, dy = maps[("dip", si, "x")][r_, c_] / sd, maps[("dip", si, "y")][r_, c_] / sd
        dip = math.hypot(dx, dy)
        # bright side towards the sun = raised (bush, mound); dark side towards the sun = hollow
        dip_sun = float(dx * sun_vec[0] + dy * sun_vec[1]) if sun_vec is not None else 0.0
        same = (darks == c["dark"])
        dist = np.hypot(pts[:, 0] - c["row"], pts[:, 1] - c["col"])
        dist[i] = 1e9
        near = int(np.sum(same & (dist < 8 * c["radius_px"])))
        nn = float(np.min(dist[same])) / c["radius_px"] if same.sum() > 1 else 50.0
        f += [maps[("L", si, "d")][r_, c_] / sd - maps[("L", si, "o")][r_, c_] / sd,
              (maps[("L", si, "r")][r_, c_] - maps[("L", si, "o")][r_, c_]) / sd,
              rin / sd, rout / sd, rin / (rout + 1e-4),
              dip, dip_sun, c["log_resp"], float((resp_all[same] < c["log_resp"]).mean()),
              c["radius_px"], float(c["dark"]),
              near, min(nn, 50.0),
              float(cls["snow"][r_, c_]), float(cls["trees"][r_, c_]), float(cls["field"][r_, c_]),
              float(cls["open"][r_, c_]), float(Lv[r_, c_] > np.percentile(Lv[valid], 97)),
              img_contrast, float(maps[("L", si, "d")][r_, c_]), float(maps[("b", si, "d")][r_, c_]),
              float(maps[("g", si, "d")][r_, c_])]
        X.append(f)
    return cands, np.nan_to_num(np.array(X, np.float32), nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- picture context

CONTEXT_KEYS = ["L_centre_z", "L_rim_z", "L_centre_vs_rim", "L_centre_vs_out", "dipole", "dipole_sun",
                "log_resp", "rough_ratio", "g_centre_vs_out", "b_centre_vs_out"]


def augment(cands, X):
    """Add (a) each key measurement's rank within this picture (a pit in a dull summer picture is
    weak in absolute terms but still stands out among the other spots), and (b) the mean of the
    key measurements of the 5 nearest similar spots (pits of one village look alike)."""
    names = feature_names()
    ki = [names.index(k) for k in CONTEXT_KEYS]
    n = len(X)
    if n == 0:
        return np.zeros((0, X.shape[1] + 2 * len(ki)), np.float32)
    dark = X[:, names.index("dark_spot")] > 0.5
    rk = np.zeros((n, len(ki)), np.float32)
    nb = np.zeros((n, len(ki)), np.float32)
    P = np.array([[c["row"], c["col"]] for c in cands])
    for m in (dark, ~dark):
        idx = np.where(m)[0]
        if len(idx) < 2:
            continue
        for j, i in enumerate(ki):
            v = X[idx, i]
            rk[idx, j] = np.argsort(np.argsort(v)) / max(1, len(v) - 1)
        if len(idx) >= 3:
            D = np.hypot(P[idx, None, 0] - P[None, idx, 0], P[idx, None, 1] - P[None, idx, 1])
            np.fill_diagonal(D, 1e9)
            nn = np.argsort(D, 1)[:, :5]
            nb[idx] = X[idx][:, ki][nn].mean(1)
    return np.hstack([X, rk, nb]).astype(np.float32)


# --------------------------------------------------------------------------- model

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_pits.json")


_CACHE = {}


def load_model(path=None):
    """The shipped model, or None if missing / switched off (SITECHECK_NO_MODEL=1)."""
    if os.environ.get("SITECHECK_NO_MODEL") and path is None:
        return None
    p = path or MODEL_PATH
    if p in _CACHE:
        return _CACHE[p]
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        _CACHE[p] = json.load(fh)
    return _CACHE[p]


def expand(X, model):
    """Optional pairwise products listed in the model (small, interpretable non-linearity)."""
    names = feature_names()
    cols = [X]
    for a, b in model.get("products", []):
        cols.append((X[:, names.index(a)] * X[:, names.index(b)])[:, None])
    return np.hstack(cols)


def _tree_raw(XA, tree):
    feat, thr, left, right, leaf, val = (np.asarray(tree[k]) for k in ("f", "t", "l", "r", "leaf", "v"))
    node = np.zeros(len(XA), np.int64)
    for _ in range(64):
        is_leaf = leaf[node].astype(bool)
        if is_leaf.all():
            break
        go_left = XA[np.arange(len(XA)), feat[node]] <= thr[node]
        nxt = np.where(go_left, left[node], right[node])
        node = np.where(is_leaf, node, nxt)
    return val[node]


def predict(X, model, cands=None):
    if model.get("type") == "gbt":
        XA = augment(cands, X) if model.get("augment") else X
        raw = np.full(len(XA), model["baseline"], np.float64)
        for tree in model["trees"]:
            raw += _tree_raw(XA, tree)
        return 1 / (1 + np.exp(-raw))
    Xe = expand(X, model)
    mu, sd = np.array(model["mean"]), np.array(model["std"])
    z = (Xe - mu) / sd
    z = np.clip(z, -6, 6)
    logit = z @ np.array(model["coef"]) + model["intercept"]
    return 1 / (1 + np.exp(-logit))


def detect(rgb, valid=None, sun_az=None, lat=None, model=None, min_p=0.35, max_n=60):
    """Spots the learned model believes are what the user marked. Same dict format as the
    other image candidates."""
    model = model or load_model()
    if model is None:
        return []
    cands, X = candidates_and_features(rgb, valid, sun_az, lat)
    if not cands:
        return []
    p = predict(X, model, cands)
    out = []
    for c, pi in sorted(zip(cands, p), key=lambda t: -t[1]):
        if pi < min_p or len(out) >= max_n:
            break
        if any(math.hypot(c["row"] - o["row"], c["col"] - o["col"]) < 1.2 * max(c["radius_px"], o["radius_px"]) for o in out):
            continue
        out.append({"type": "pit" if c["dark"] else "mark", "kind": None if c["dark"] else "light_spot",
                    "row": c["row"], "col": c["col"], "diameter_m": None, "radius_px": c["radius_px"],
                    "confidence": round(float(pi), 2), "learned_p": round(float(pi), 3),
                    "source_signal": "learned from marked examples",
                    "note": ("dark spot that looks like the marked house pits / remains" if c["dark"]
                             else "light spot that looks like the marked examples (snow in a pit, soil patch)")
                             + f" - model score {pi:.2f}"})
    return out
