"""Training material for the image network, from the marked packs.

    sitecheck cnn-prep Rus_earth pack Geo_Pack GEO Fear -o cnn_data

For every picture that can be used: the picture (reference grid, as marked), the marked objects
(centre, size, class 0 = pit / hollow, 1 = foundation), a mask of what counts neither right nor
wrong (roads, yards, "not sure", ramparts, legend, outside the purple outline) and a mask of the
marked look-alikes (bushes, animals). Empty places: pictures with no objects at all.

Also:
  * duplicates are dropped - the same place in two packs (same picture size, near-identical clean
    picture) is kept once;
  * places closer than 3 km form one group; ~20 % of the groups are held out as the TEST set
    (fixed list in sitecheck/cnn_split.json, so every run and every baseline uses the same places).
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import cv2
import numpy as np

from . import groundtruth as G

SPLIT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnn_split.json")


def _thumb(rgb):
    t = cv2.resize(rgb.astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA)
    return (t - t.mean()) / (t.std() + 1e-6)


def _class_of(o):
    k = o.get("kind")
    if k == "foundation" or (k in (None, "object") and o.get("shape") == "rectangle"):
        return 1
    return 0


def _ignore_mask(gt, shape):
    ign = np.zeros(shape, bool)
    for k in ("roads", "enclosures", "unsure", "ramparts", "legend", "zones"):
        m = gt.get(k)
        if m is not None and getattr(m, "shape", None) == shape:
            ign |= m
    ch = gt.get("checked")
    if ch is not None and ch.any():
        ign |= ~ch
    return ign


def _look_mask(gt, shape):
    m = np.zeros(shape, bool)
    for la in gt.get("lookalikes") or []:
        rc = np.array(la["mask_rc"], int)
        if len(rc):
            m[np.clip(rc[:, 0], 0, shape[0] - 1), np.clip(rc[:, 1], 0, shape[1] - 1)] = True
    return m


def place_groups(areas, km=3.0):
    """areas: {key: (lat, lon) or None} -> {key: group id}"""
    keys = sorted(areas)
    parent = {k: k for k in keys}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    pos = [(k, v) for k, v in areas.items() if v and v[0] is not None]
    for i, (a, (la, lo)) in enumerate(pos):
        for b, (lb, lob) in pos[i + 1:]:
            if math.hypot((la - lb) * 111.0, (lo - lob) * 111.0 * math.cos(math.radians(la))) < km:
                parent[find(a)] = find(b)
    return {k: find(k) for k in keys}


def make_split(groups, n_pos, frac=0.2, seed="sitecheck"):
    """Hold out ~frac of the pictures with objects, whole groups at a time, from every pack."""
    by_pack = {}
    for k, g in groups.items():
        by_pack.setdefault(k.split("/")[0], set()).add(g)
    test = set()
    for pack, gs in by_pack.items():
        gs = sorted(gs, key=lambda g: hashlib.md5((seed + g).encode()).hexdigest())
        total = sum(n_pos.get(k, 0) for k in groups if k.split("/")[0] == pack)
        got = 0
        for g in gs:
            if got >= frac * total:
                break
            members = [k for k in groups if groups[k] == g]
            n = sum(n_pos.get(k, 0) for k in members)
            if n == 0:
                continue
            test.add(g)
            got += n
    return sorted(k for k, g in groups.items() if g in test)


def prepare(roots, out_dir, split_path=None, log=print):
    roots = [roots] if isinstance(roots, str) else list(roots)
    os.makedirs(out_dir, exist_ok=True)
    pics, areas, seen = [], {}, []
    for root in roots:
        pack = os.path.basename(os.path.normpath(root))
        for e in G.discover(root):
            key = f"{pack}/{e['category']}_{e['area']}"
            try:
                ref, gt, seasons = G.prepare(e, log=lambda *a: None)
            except Exception as ex:                          # a broken area must not stop the rest
                log(f"  {key}: skipped ({ex})")
                continue
            if not e.get("negative") and not gt["objects"]:
                log(f"  {key}: no objects marked - skipped")
                continue
            th = _thumb(ref)
            dup = next((k for k, s, t in seen if s == ref.shape and float(np.abs(t - th).mean()) < 0.15), None)
            if dup:
                log(f"  {key}: same place as {dup} - skipped")
                continue
            seen.append((key, ref.shape, th))
            areas[key] = (e.get("lat"), e.get("lon"))
            shape = ref.shape[:2]
            ign = _ignore_mask(gt, shape)
            look = _look_mask(gt, shape)
            objs = np.array([[o["row"], o["col"], o["diameter_px"], _class_of(o)] for o in gt["objects"]],
                            np.float32).reshape(-1, 4)
            for i, s in enumerate(seasons):
                if not s["ok"]:
                    continue
                vis = s.get("visibility") or "not stated"
                fid = hashlib.md5(f"{key}|{s['name']}".encode()).hexdigest()[:12]
                rgb8 = (np.clip(s["rgb"], 0, 1) * 255 + 0.5).astype(np.uint8)
                valid = s["rgb"].max(-1) > 0.02
                np.savez_compressed(os.path.join(out_dir, fid + ".npz"), rgb=rgb8, ignore=ign | ~valid,
                                    look=look, objects=objs)
                pics.append({"file": fid + ".npz", "key": key, "pack": pack, "picture": s["name"],
                             "visibility": vis, "mpp": s.get("mpp"), "negative": bool(e.get("negative")),
                             "n_obj": int(len(objs)), "shape": list(shape)})
            log(f"  {key}: {sum(p['key'] == key for p in pics)} picture(s), {len(objs)} objects"
                + (" (empty place)" if e.get("negative") else ""))
    groups = place_groups(areas)
    split_path = split_path or SPLIT_PATH
    if os.path.exists(split_path):
        with open(split_path, encoding="utf-8") as fh:
            test_keys = set(json.load(fh)["test_areas"])
        log(f"held-out test areas from {split_path}: {len(test_keys)}")
    else:
        n_pos = {}
        for p in pics:
            if not p["negative"]:
                n_pos[p["key"]] = n_pos.get(p["key"], 0) + 1
        test_keys = set(make_split(groups, n_pos))
    # a test place pulls its whole 3 km group into the test set
    test_groups = {groups[k] for k in test_keys if k in groups}
    for p in pics:
        p["group"] = groups[p["key"]]
        p["split"] = "test" if p["group"] in test_groups else "train"
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as fh:
        json.dump({"pictures": pics, "roots": [os.path.basename(os.path.normpath(r)) for r in roots]}, fh, indent=1)
    if not os.path.exists(split_path):
        with open(split_path, "w", encoding="utf-8") as fh:
            json.dump({"test_areas": sorted({p["key"] for p in pics if p["split"] == "test"})}, fh, indent=1)
    n_t = sum(p["split"] == "test" for p in pics)
    log(f"{len(pics)} pictures ({n_t} held out for testing) in {out_dir}")
    return pics


def targets(objects, shape, down=4, sigma_frac=0.25, min_sigma_cells=1.0):
    """Heat-map targets at 1/down of the picture: a Gaussian bump per object in its class channel.
    The top of every bump is exactly 1 on the output cell nearest the object's centre (cell i covers
    picture pixels around i*down + (down-1)/2), so every object has a positive cell however small."""
    H, W = shape
    h, w = H // down, W // down
    t = np.zeros((2, h, w), np.float32)
    yy, xx = np.mgrid[:h, :w]
    off = (down - 1) / 2.0
    for r, c, d, k in objects:
        cy, cx = int(round((r - off) / down)), int(round((c - off) / down))
        if not (-3 <= cy < h + 3 and -3 <= cx < w + 3):
            continue
        s = max(min_sigma_cells, sigma_frac * d / down)
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
        t[int(k)] = np.maximum(t[int(k)], g)
    return t


def load(path):
    """One prepared picture: rgb (H,W,3) 0..1, ignore, look, objects."""
    z = np.load(path)
    return z["rgb"].astype(np.float32) / 255.0, z["ignore"], z["look"], z["objects"]


def gt_of(objects, ignore):
    """The ground truth in the form sitecheck.evaluate.match expects (ignore -> 'not sure')."""
    objs = [{"row": float(r), "col": float(c), "diameter_px": float(d),
             "kind": "foundation" if int(k) == 1 else "pit", "shape": "rectangle" if int(k) == 1 else "round"}
            for r, c, d, k in objects]
    z = np.zeros(ignore.shape, bool)
    return {"objects": objs, "zones": z, "unsure": ignore, "roads": z, "enclosures": z}


def score(cands, objects, ignore):
    """AP, share found among the top 30, share of correct among the top 30 - for one picture."""
    from . import evaluate as EV
    gt = gt_of(objects, ignore)
    pts = [c for c in cands if c.get("type") != "linear"]
    sc, found = EV.match(pts, gt, ignore.shape)
    n = len(gt["objects"])
    if n == 0:
        return {"AP": None, "strong_false": int(sum(c["confidence"] >= 0.5 for c in pts))}
    top = sorted(sc, key=lambda x: -x[0])[:30]
    return {"AP": float(EV.average_precision(sc, n)),
            "found_top30": len({i for s in top for i in s[2] if i != -1}) / n,
            "correct_top30": float(np.mean([s[1] for s in top])) if top else 0.0}
