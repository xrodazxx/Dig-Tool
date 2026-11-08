"""Measure detectors against the user's marked examples (see groundtruth.py).

For every picture (the reference and every season aligned to it) the candidates of a
detector are matched with the marked objects:
    recall     share of marked objects with a candidate on / next to them
    precision  share of candidates that hit a marked object (or a marked zone) -
               pessimistic, the user did not necessarily mark every pit
    AP         average precision of the confidence ranking (how well real objects come first)
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi


def match(cands, gt, shape, extra_px=6.0, keep_ignored=False):
    """cands: [{row, col, confidence}] -> list of (confidence, is_hit, gt_ids), set of found GT ids.
    gt id -1 = inside a marked zone. Candidates on a marked road or inside a marked enclosure
    (but not on an object) are left out."""
    objs = gt["objects"]
    zones = gt["zones"]
    found = set()
    scored = []
    for c in cands:
        r, cc = c["row"], c["col"]
        ids = []
        for i, o in enumerate(objs):
            rad = o["diameter_px"] / 2 + max(extra_px, 0.5 * o["diameter_px"])
            if math.hypot(r - o["row"], cc - o["col"]) <= rad:
                ids.append(i)
                found.add(i)
        ri, ci = int(np.clip(r, 0, shape[0] - 1)), int(np.clip(cc, 0, shape[1] - 1))
        if not ids and zones.any() and zones[ri, ci]:
            ids = [-1]
        if not ids and any(gt.get(k) is not None and gt[k][ri, ci] for k in ("roads", "enclosures")):
            if keep_ignored:                  # (training: keep the slot, label unknown)
                scored.append((float(c.get("confidence", 0.5)), False, None))
            continue     # on a marked road / inside a marked yard: neither right nor wrong
        scored.append((float(c.get("confidence", 0.5)), bool(ids), ids))
    return scored, found


def average_precision(scored, n_pos):
    """Each marked object counts once (its best-ranked candidate); zone hits count each time."""
    if not scored or n_pos == 0:
        return 0.0
    scored = sorted(scored, key=lambda x: -x[0])
    claimed, tp, ap = set(), 0, 0.0
    for k, (_, h, ids) in enumerate(scored, 1):
        new = [i for i in ids if i == -1 or i not in claimed]
        if new:
            claimed.update(i for i in new if i != -1)
            tp += 1
            ap += tp / k
    return min(1.0, ap / max(n_pos, 1))


def _in_mask_rc(items, shape):
    m = np.zeros(shape, np.int32)
    for i, it in enumerate(items):
        rc = np.array(it["mask_rc"], int)
        if len(rc):
            m[np.clip(rc[:, 0], 0, shape[0] - 1), np.clip(rc[:, 1], 0, shape[1] - 1)] = i + 1
    return m


def fooled_by(cands, gt, shape, top=30):
    """How many of the top candidates sit on something the user marked as NOT a site
    (bushes/trees, animals). Returns {kind: count}."""
    la = gt.get("lookalikes") or []
    if not la:
        return {}
    lm = _in_mask_rc(la, shape)
    lm = ndi.grey_dilation(lm, size=5)
    out = {}
    pts = sorted([c for c in cands if c.get("type") != "linear"], key=lambda c: -c.get("confidence", 0))[:top]
    for c in pts:
        r, cc = int(np.clip(c["row"], 0, shape[0] - 1)), int(np.clip(c["col"], 0, shape[1] - 1))
        k = lm[r, cc]
        if k:
            kind = la[k - 1]["kind"]
            out[kind] = out.get(kind, 0) + 1
    return out


def kind_recall(found, gt):
    """Share of marked objects of each kind (pit / foundation / object) that were found."""
    tot, hit = {}, {}
    for i, o in enumerate(gt["objects"]):
        k = o.get("kind", "object")
        tot[k] = tot.get(k, 0) + 1
        if i in found:
            hit[k] = hit.get(k, 0) + 1
    return {k: round(hit.get(k, 0) / n, 3) for k, n in tot.items()}


def line_cover(cands, lines, shape, tol_px):
    """Share of a marked line (rampart / road centre line) that lies within tol of a found linear feature."""
    P = [p for ln in lines for p in (ln.get("centre_px") or ln["ends_px"])]
    if not P:
        return None
    segs = [c["ends_px"] for c in cands if c.get("type") == "linear" and c.get("ends_px")]
    if not segs:
        return 0.0
    P = np.array(P, float)
    best = np.full(len(P), 1e9)
    for (r1, c1), (r2, c2) in segs:
        a, b = np.array([r1, c1]), np.array([r2, c2])
        ab = b - a
        t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
        d = np.hypot(*(P - (a + t[:, None] * ab)).T)
        best = np.minimum(best, d)
    return round(float((best <= tol_px).mean()), 3)


def summarize(scored, found, n_obj, top=None):
    sc = sorted(scored, key=lambda x: -x[0])
    if top:
        sc = sc[:top]
    hits = sum(x[1] for x in sc)
    return {"candidates": len(sc), "hits": hits,
            "precision": round(hits / len(sc), 3) if sc else 0.0,
            "recall": round(len(found) / n_obj, 3) if n_obj else None}


def run_folder(root, out_dir=None, mpp=0.3, use_model=True, log=print):
    """Test the screenshot detector on marked areas: every picture of every area (seasons are
    aligned to the marked one), scored against the red marks. Prints recall / precision / AP
    per category. Use it on places you already know to see how far to trust the program."""
    import json
    import os
    from . import groundtruth as G, imagery as I
    rows = []
    for e in G.discover(root):
        ref, gt, seasons = G.prepare(e, log=lambda *a: None)
        key = f"{e['category']}/{e['area']}"
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            G.draw_gt(ref, gt, os.path.join(out_dir, f"{e['category']}_{e['area']}_marks.jpg"))
        for s in seasons:
            if not s["ok"]:
                log(f"  {key} {s['name']}: could not be aligned to the marked picture - skipped")
                continue
            rgb = s["rgb"]
            valid = rgb.max(-1) > 0.02
            mpp_s = s.get("mpp", mpp)                     # from the area's notes (eye altitude), if given
            r = I.analyse_image(rgb, valid, mpp_s, lat=e.get("lat"), model="default" if use_model else None)
            cands = [c for c in r["candidates"] if c["type"] != "linear"]
            n = len(gt["objects"])
            if e.get("negative"):
                strong = int(sum(c["confidence"] >= 0.5 for c in cands))
                log(f"  {key:22s} {s['name']:18s} EMPTY PLACE: {len(cands)} candidates, {strong} scored >= 0.5 (all false)")
                rows.append({"category": e["category"], "area": e["area"], "picture": s["name"], "objects": 0,
                             "candidates": len(cands), "false_strong": strong, "negative": True})
                continue
            scored, found = match(cands, gt, rgb.shape[:2])
            S = summarize(scored, found, n)
            ap = average_precision(scored, n) if n else 0.0
            rows.append({"category": e["category"], "area": e["area"], "picture": s["name"], "objects": n,
                         "candidates": S["candidates"], "recall": S["recall"], "precision": S["precision"],
                         "AP": round(ap, 3)})
            log(f"  {key:22s} {s['name']:18s} marked {n:3d}  candidates {S['candidates']:4d}  "
                f"recall {S['recall'] if S['recall'] is not None else '-'}  precision {S['precision']:.2f}  AP {ap:.2f}")
    cats = {}
    for r in rows:
        if not r.get("negative"):
            cats.setdefault(r["category"], []).append(r)
    summary = {}
    for c, v in list(cats.items()) + [("ALL", [r for r in rows if not r.get("negative")])]:
        if not v:
            continue
        summary[c] = {"pictures": len(v), "recall": round(float(np.mean([x["recall"] or 0 for x in v])), 2),
                      "precision": round(float(np.mean([x["precision"] for x in v])), 2),
                      "AP": round(float(np.mean([x["AP"] for x in v])), 2)}
        log(f"{c:16s} pictures {len(v):2d}  recall {summary[c]['recall']:.2f}  precision "
            f"{summary[c]['precision']:.2f}  AP {summary[c]['AP']:.2f}")
    if out_dir:
        with open(os.path.join(out_dir, "evaluation.json"), "w") as fh:
            json.dump({"pictures": rows, "summary": summary}, fh, indent=1, default=float)
    return rows, summary
