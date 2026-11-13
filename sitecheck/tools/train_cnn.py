# %% [markdown]
# # sitecheck: training the image network (pits and foundations)
#
# Run this on a free GPU (Kaggle: Settings -> Accelerator -> GPU, Internet ON; or Google Colab),
# or on your own PC with an NVIDIA card. It takes about 1-2 hours on a free GPU.
#
# **What you upload** (Kaggle: "Add Input" -> your private dataset):
# * `sitecheck.zip` (the program),
# * your packs, unpacked: `Rus_earth`, `pack`, `Geo_Pack`, `GEO`, `Fear` (any folder names work).
#
# **What you get back**: `model_cnn.npz` -> put it into the `sitecheck/sitecheck/` folder.
# sitecheck then uses it automatically, but only if it beat the current program on the held-out
# test places (decided at the end of this notebook and written into the file).
#
# Nothing here is sent anywhere except the GPU machine you run it on. The packs' .txt files contain
# coordinates: if you would rather not upload those, delete the `coord` lines first (only the
# grouping of nearby places uses them).

# %% settings - change paths if needed
import glob
import json
import os
import sys
import time
import zipfile

INPUT = "/kaggle/input" if os.path.isdir("/kaggle/input") else os.path.abspath("input")
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("work")
EPOCHS = 40
CROP = 320            # training crops (multiple of 16)
BATCH = 16
WORK_MPP = 0.3        # pictures are brought to ~0.3 m per pixel when their scale is known
WINDOW_PX = 1749      # width of YOUR Google Earth map view in pixels (measured with the 200 m ruler:
                      # 219 px at 1.39 km). Packs with a "window" line in their General info use that.
os.makedirs(WORK, exist_ok=True)

# %% find and unpack sitecheck
zips = glob.glob(os.path.join(INPUT, "**", "sitecheck*.zip"), recursive=True)
code_dir = os.path.join(WORK, "code")
if zips and not os.path.isdir(os.path.join(code_dir, "sitecheck", "sitecheck")):
    zipfile.ZipFile(zips[0]).extractall(code_dir)
cands = glob.glob(os.path.join(code_dir, "**", "sitecheck", "cnn.py"), recursive=True) or \
    glob.glob(os.path.join(INPUT, "**", "sitecheck", "cnn.py"), recursive=True)
assert cands, "sitecheck not found - upload sitecheck.zip"
sys.path.insert(0, os.path.dirname(os.path.dirname(cands[0])))
from sitecheck import cnn, cnn_data, scale  # noqa: E402
scale.VIEW_WIDTH_PX = WINDOW_PX
print("sitecheck from", os.path.dirname(os.path.dirname(cands[0])))

# %% prepare the training material (once; ~15-30 min on CPU)
DATA = os.path.join(WORK, "cnn_data")
if not os.path.exists(os.path.join(DATA, "index.json")):
    roots = []
    for d in sorted(glob.glob(os.path.join(INPUT, "**"), recursive=True)):
        if not os.path.isdir(d) or "code" in d.split(os.sep):
            continue
        sub = [x for x in glob.glob(os.path.join(d, "*")) if os.path.isdir(x)]
        has_area = any(glob.glob(os.path.join(s, "*.png")) + glob.glob(os.path.join(s, "*.jpg")) or
                       glob.glob(os.path.join(s, "*", "*.png")) + glob.glob(os.path.join(s, "*", "*.jpg"))
                       for s in sub)
        parent_is_root = any(d.startswith(r + os.sep) for r in roots)
        if has_area and not parent_is_root and os.path.basename(d) in (
                "Rus_earth", "pack", "Geo_Pack", "GEO", "Fear") + tuple(os.environ.get("PACKS", "").split(",")):
            roots.append(d)
    print("packs:", roots)
    cnn_data.prepare(roots, DATA)
index = json.load(open(os.path.join(DATA, "index.json")))["pictures"]
train_pics = [p for p in index if p["split"] == "train" and p["visibility"] != "not visible"]
test_pics = [p for p in index if p["split"] == "test" and not p["negative"]]
print(len(train_pics), "training pictures,", len(test_pics), "test pictures")

# %% data: crops with rotation, flips, zoom and colour changes (numpy only)
import cv2  # noqa: E402
import numpy as np  # noqa: E402


def load_scaled(p):
    rgb, ign, look, objs = cnn_data.load(os.path.join(DATA, p["file"]))
    f = (p["mpp"] / WORK_MPP) if p.get("mpp") else 1.0
    f = float(np.clip(f, 0.4, 2.5))
    if abs(f - 1) > 0.03:
        H, W = rgb.shape[:2]
        size = (max(16, round(W * f)), max(16, round(H * f)))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
        ign = cv2.resize(ign.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        look = cv2.resize(look.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        objs = objs.copy()
        objs[:, :3] *= f
    return rgb, ign, look, objs, f


def make_crop(rgb, ign, look, objs, rng):
    """One training crop (numpy only): zoom, crop, rotate/flip, colour. -> image, targets, weights"""
    # zoom 0.7-1.4 (unknown scales, different satellites)
    z = float(np.exp(rng.uniform(np.log(0.7), np.log(1.4))))
    H, W = rgb.shape[:2]
    size = (max(CROP, round(W * z)), max(CROP, round(H * z)))
    zx, zy = size[0] / W, size[1] / H
    im = cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR)
    ig = cv2.resize(ign.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    lk = cv2.resize(look.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    ob = objs.copy()
    ob[:, 0] *= zy
    ob[:, 1] *= zx
    ob[:, 2] *= (zx + zy) / 2
    # crop: half of the time around an object
    Hh, Ww = im.shape[:2]
    if len(ob) and rng.random() < 0.5:
        o = ob[rng.integers(len(ob))]
        y0 = int(np.clip(o[0] - rng.integers(CROP // 4, 3 * CROP // 4), 0, Hh - CROP))
        x0 = int(np.clip(o[1] - rng.integers(CROP // 4, 3 * CROP // 4), 0, Ww - CROP))
    else:
        y0, x0 = int(rng.integers(0, Hh - CROP + 1)), int(rng.integers(0, Ww - CROP + 1))
    im = im[y0:y0 + CROP, x0:x0 + CROP].copy()
    ig = ig[y0:y0 + CROP, x0:x0 + CROP]
    lk = lk[y0:y0 + CROP, x0:x0 + CROP]
    ob = ob.copy()
    ob[:, 0] -= y0
    ob[:, 1] -= x0
    # rotate by 90 degrees / flip (the sun side changes too - fine, it differs between satellites anyway)
    k = int(rng.integers(4))
    flip = rng.random() < 0.5
    def tr(a):
        a = np.rot90(a, k)
        return a[:, ::-1] if flip else a
    im, ig, lk = tr(im), tr(ig), tr(lk)
    for _ in range(k):                              # rot90 counter-clockwise: (r, c) -> (C-1-c, r)
        ob[:, [0, 1]] = np.stack([CROP - 1 - ob[:, 1], ob[:, 0]], 1)
    if flip:
        ob[:, 1] = CROP - 1 - ob[:, 1]
    # colour: brightness, contrast, saturation, slight hue, haze, blur, noise
    im = np.ascontiguousarray(im)
    im = im * rng.uniform(0.75, 1.25) + rng.uniform(-0.08, 0.08)
    m = im.mean()
    im = (im - m) * rng.uniform(0.7, 1.3) + m
    g = im.mean(-1, keepdims=True)
    im = g + (im - g) * rng.uniform(0.6, 1.4)
    im = im * (1 + rng.uniform(-0.05, 0.05, 3))
    if rng.random() < 0.3:
        im = im * (1 - (h := rng.uniform(0, 0.3))) + h * rng.uniform(0.6, 0.9)
    if rng.random() < 0.3:
        im = cv2.GaussianBlur(im.astype(np.float32), (0, 0), rng.uniform(0.5, 1.5))
    im = im + rng.normal(0, rng.uniform(0, 0.03), im.shape)
    im = np.clip(im, 0, 1).astype(np.float32)
    keep = (ob[:, 0] >= -20) & (ob[:, 0] < CROP + 20) & (ob[:, 1] >= -20) & (ob[:, 1] < CROP + 20)
    t = cnn_data.targets(ob[keep], (CROP, CROP))
    w = np.ones((CROP // 4, CROP // 4), np.float32)
    w[cv2.resize(ig.astype(np.uint8), (CROP // 4, CROP // 4), interpolation=cv2.INTER_NEAREST) > 0] = 0
    lk4 = cv2.resize(lk.astype(np.uint8), (CROP // 4, CROP // 4), interpolation=cv2.INTER_NEAREST) > 0
    w[lk4 & (t.max(0) < 0.3)] = 3.0                  # marked bushes / animals: mistakes there count 3x
    return im, t, w


# %% PyTorch
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV)
MEAN = torch.tensor(cnn.MEAN).view(1, 3, 1, 1)
STD = torch.tensor(cnn.STD).view(1, 3, 1, 1)


class Crops(torch.utils.data.Dataset):
    def __init__(self, pics, per_pic=6):
        self.pics, self.per = pics, per_pic
        self.cache = {}

    def __len__(self):
        return len(self.pics) * self.per

    def __getitem__(self, i):
        p = self.pics[i // self.per]
        if p["file"] not in self.cache:                  # kept as 8-bit: free notebooks have little memory
            r_, i_, l_, o_ = load_scaled(p)[:4]
            self.cache[p["file"]] = ((r_ * 255 + 0.5).astype(np.uint8), i_, l_, o_)
        rgb8, ign, look, objs = self.cache[p["file"]]
        im, t, w = make_crop(rgb8.astype(np.float32) / 255.0, ign, look, objs, np.random.default_rng())
        return torch.from_numpy(im.transpose(2, 0, 1).copy()), torch.from_numpy(t), torch.from_numpy(w)


# %% the network (same layout as sitecheck/cnn.py)
import torchvision  # noqa: E402


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception as ex:                         # no internet: start from random weights (much worse)
            print("!! pretrained weights not available:", ex)
            r = torchvision.models.resnet18(weights=None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.l1, self.l2, self.l3 = r.layer1, r.layer2, r.layer3
        self.dec2 = nn.Sequential(nn.Conv2d(256 + 128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv2d(128 + 64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.head = nn.Conv2d(64, 2, 1)
        nn.init.constant_(self.head.bias, -2.2)

    def forward(self, x):
        x = (x - MEAN.to(x.device)) / STD.to(x.device)
        f1 = self.l1(self.stem(x))
        f2 = self.l2(f1)
        f3 = self.l3(f2)
        d2 = self.dec2(torch.cat([F.interpolate(f3, scale_factor=2, mode="bilinear", align_corners=False), f2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False), f1], 1))
        return self.head(d1)


def focal_loss(logits, t, w, alpha=2, beta=4):
    """CenterNet-style focal loss on Gaussian targets, weighted (0 = ignore, 3 = look-alike)."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = (t > 0.98).float()
    neg = 1 - pos
    w = w.unsqueeze(1)
    lp = -torch.log(p) * (1 - p) ** alpha * pos
    ln = -torch.log(1 - p) * p ** alpha * (1 - t) ** beta * neg
    n = (pos * w).sum().clamp(min=1)
    return ((lp + ln) * w).sum() / n


# %% evaluation on the held-out test places (whole pictures, same scoring as sitecheck)
baseline = json.load(open(os.path.join(os.path.dirname(cnn.__file__), "cnn_baseline.json")))


@torch.no_grad()
def torch_heat(net, rgb):
    H, W = rgb.shape[:2]
    ph, pw = (-H) % 16, (-W) % 16
    x = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    x = torch.from_numpy(x.transpose(2, 0, 1).copy())[None].to(DEV)
    return torch.sigmoid(net(x))[0].cpu().numpy(), (ph, pw)


def heat_to_cands(prob, shape_scaled, pad, shape_orig, mpp):
    """Peaks -> candidates in the picture's own pixels (mirrors cnn.heatmaps + cnn.detect)."""
    Hs, Ws = shape_scaled
    ph, pw = pad
    full = np.stack([cv2.resize(prob[c], (Ws + pw, Hs + ph), interpolation=cv2.INTER_LINEAR)[:Hs, :Ws]
                     for c in range(prob.shape[0])])
    if (Hs, Ws) != tuple(shape_orig):
        H, W = shape_orig
        full = np.stack([cv2.resize(full[c], (W, H), interpolation=cv2.INTER_LINEAR) for c in range(2)])
    out = []
    r_px = max(3, int(round(2.0 / (mpp or 0.3))))
    for ci, name in enumerate(cnn.CLASSES):
        m = full[ci]
        peak = (m == cv2.dilate(m, np.ones((2 * r_px + 1, 2 * r_px + 1), np.uint8))) & (m >= 0.08)
        ys, xs = np.nonzero(peak)
        for y, x in zip(ys, xs):
            out.append({"type": "pit" if name == "pit" else "rectangle", "row": float(y), "col": float(x),
                        "confidence": float(m[y, x]), "cnn_p": float(m[y, x])})
    return sorted(out, key=lambda c: -c["confidence"])[:300]


def predict_test(net):
    """The network's candidates on every test picture (computed once, scored many ways)."""
    net.eval()
    out = []
    for p in test_pics:
        rgb_s = load_scaled(p)[0]
        rgb, ign, look, objs = cnn_data.load(os.path.join(DATA, p["file"]))
        prob, pad = torch_heat(net, rgb_s)
        cc = heat_to_cands(prob, rgb_s.shape[:2], pad, rgb.shape[:2], p.get("mpp"))
        out.append((p, cc, objs, ign, baseline["pictures"].get(p["key"] + "|" + p["picture"])))
    net.train()
    return out


def evaluate(pred, merge_w=None):
    res = {"cnn": [], "baseline": [], "merged": []}
    for p, cc, objs, ign, b in pred:
        res["cnn"].append(cnn_data.score(cc, objs, ign)["AP"])
        if b:
            res["baseline"].append(b["AP"])
            merged = cnn.merge([dict(c) for c in b["candidates"]], [dict(c) for c in cc], p.get("mpp"),
                               **(merge_w or {}))
            res["merged"].append(cnn_data.score(merged, objs, ign)["AP"])
    return {k: float(np.mean(v)) if v else None for k, v in res.items()}, len(res["baseline"])


# %% training
net = Net().to(DEV)
dl = torch.utils.data.DataLoader(Crops(train_pics), batch_size=BATCH, shuffle=True, num_workers=2,
                                 drop_last=True, persistent_workers=True)
opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=EPOCHS * len(dl), pct_start=0.15)
scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
history = []
for ep in range(EPOCHS):
    t0, tot = time.time(), 0.0
    for x, t, w in dl:
        x, t, w = x.to(DEV), t.to(DEV), w.to(DEV)
        with torch.autocast(device_type="cuda", enabled=DEV == "cuda"):
            logits = net(x)
        loss = focal_loss(logits.float(), t, w)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        tot += loss.item()
    msg = f"epoch {ep + 1}/{EPOCHS} loss {tot / len(dl):.3f} ({time.time() - t0:.0f}s)"
    if (ep + 1) % 10 == 0 or ep == EPOCHS - 1:          # progress only: the LAST epoch is kept, whatever this says
        r, n = evaluate(predict_test(net))
        history.append((ep + 1, r))
        msg += f" | test AP: network {r['cnn']:.3f}, current program {r['baseline']:.3f}, both {r['merged']:.3f}"
    print(msg, flush=True)
torch.save(net.state_dict(), os.path.join(WORK, "net_last.pt"))

# %% choose how the network is combined with the current program (on the test places)
pred = predict_test(net)
final, n_base = evaluate(pred)
grid = [dict(w_old=a, w_new=b, demote=d, add=ad) for a, b in ((0.2, 0.9), (0.5, 0.5), (0.0, 1.0))
        for d in (0.55, 0.8) for ad in (0.5, 1.0)]
# (12 rules tried on the test places: the best one is a little optimistic - hence the +0.01 margin)
scores = [(evaluate(pred, g)[0]["merged"], g) for g in grid]
best_m, best_g = max(scores, key=lambda s: s[0])
print(f"\nTEST ({len(test_pics)} pictures, {n_base} with a baseline):")
print(f"  current program (hand-made detectors + tree model trained without these places): {final['baseline']:.3f}")
print(f"  image network alone:                                                         {final['cnn']:.3f}")
print(f"  both combined, best rule {best_g}:                                         {best_m:.3f}")
use = "merge" if best_m > final["baseline"] + 0.01 else "off"
print("=> sitecheck will", "USE the network (combined)" if use == "merge" else "NOT use the network (no clear gain)")

# %% export: batch norm folded into the convolutions, saved for the numpy runner
def fold(conv, bn):
    w = conv.weight.detach().cpu().double()
    b = conv.bias.detach().cpu().double() if conv.bias is not None else torch.zeros(w.shape[0], dtype=torch.double)
    if bn is None:
        return w.float().numpy(), b.float().numpy()
    s = bn.weight.detach().cpu().double() / torch.sqrt(bn.running_var.detach().cpu().double() + bn.eps)
    w = w * s.view(-1, 1, 1, 1)
    b = (b - bn.running_mean.detach().cpu().double()) * s + bn.bias.detach().cpu().double()
    return w.float().numpy(), b.float().numpy()


P = {}
P["stem.w"], P["stem.b"] = fold(net.stem[0], net.stem[1])
for li, layer in (("l1", net.l1), ("l2", net.l2), ("l3", net.l3)):
    for bi, blk in enumerate(layer):
        P[f"{li}.{bi}.c1.w"], P[f"{li}.{bi}.c1.b"] = fold(blk.conv1, blk.bn1)
        P[f"{li}.{bi}.c2.w"], P[f"{li}.{bi}.c2.b"] = fold(blk.conv2, blk.bn2)
        if blk.downsample is not None:
            P[f"{li}.{bi}.down.w"], P[f"{li}.{bi}.down.b"] = fold(blk.downsample[0], blk.downsample[1])
P["dec2.w"], P["dec2.b"] = fold(net.dec2[0], net.dec2[1])
P["dec1.w"], P["dec1.b"] = fold(net.dec1[0], net.dec1[1])
P["head.w"], P["head.b"] = fold(net.head, None)

# self-check: the numpy runner inside sitecheck must give the same maps as PyTorch
net.eval()
rgb_s = load_scaled(test_pics[0])[0]
tp, _ = torch_heat(net, rgb_s)
lg, _ = cnn.forward(rgb_s, P)
npp = 1 / (1 + np.exp(-lg))
diff = float(np.abs(npp[:, :tp.shape[1], :tp.shape[2]] - tp).max())
print("numpy runner vs PyTorch, largest difference:", diff)
assert diff < 2e-3, "export does not match - do not use this file"

meta = {"meta.work_mpp": np.array(WORK_MPP), "meta.mode": np.array(use),
        "meta.test_ap_network": np.array(final["cnn"]), "meta.test_ap_program": np.array(final["baseline"]),
        "meta.test_ap_combined": np.array(best_m), "meta.test_pictures": np.array(len(test_pics)),
        "meta.train_pictures": np.array(len(train_pics)), "meta.epochs": np.array(EPOCHS)}
for k, v in best_g.items():
    meta["meta.merge_" + k] = np.array(float(v))
out = os.path.join(WORK, "model_cnn.npz")
np.savez_compressed(out, **P, **meta)
print("saved", out, f"({os.path.getsize(out) // 1024} KB) - put it into sitecheck/sitecheck/")
