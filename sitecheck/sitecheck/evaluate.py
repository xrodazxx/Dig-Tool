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
    gt id -1 = inside a marked zone. Candidates on a marked road, inside a marked enclosure / fortress,
    in a white "not sure" loop or outside the purple "checked" outline (and not on an object) are left out."""
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
        ch = gt.get("checked")
        outside = ch is not None and ch.any() and not ch[ri, ci]
        if not ids and (outside or any(gt.get(k) is not None and gt[k][ri, ci]
                                       for k in ("roads", "enclosures", "unsure", "ramparts"))):
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
    rows, together = [], []
    for e in G.discover(root):
        key = f"{e['category']}/{e['area']}"
        ref, gt, seasons = G.prepare(e, log=lambda *a: None)
        if not e.get("negative") and not gt["objects"]:
            log(f"  {key}: no single objects marked (ramparts / roads only) - not scored for spots")
            continue
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            G.draw_gt(ref, gt, os.path.join(out_dir, f"{e['category']}_{e['area']}_marks.jpg"))
        per_year, ap_single = [], []
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
            vis = s.get("visibility") or "not stated"
            rows.append({"category": e["category"], "area": e["area"], "picture": s["name"], "objects": n,
                         "candidates": S["candidates"], "recall": S["recall"], "precision": S["precision"],
                         "AP": round(ap, 3), "visibility": vis, "date": s.get("date"),
                         "fooled_by": fooled_by(cands, gt, rgb.shape[:2])})
            if vis != "not visible":
                per_year.append({"candidates": cands})
                ap_single.append(ap)
            log(f"  {key:22s} {s['name']:18s} {vis:11s} marked {n:3d}  candidates {S['candidates']:4d}  "
                f"recall {S['recall'] if S['recall'] is not None else '-'}  precision {S['precision']:.2f}  AP {ap:.2f}")
        if len(per_year) >= 2:          # all years of the place put together (as `sitecheck image` does)
            fused = I.fuse(per_year, seasons[0].get("mpp", mpp))
            sc_f, _ = match(fused, gt, ref.shape[:2])
            ap_f = average_precision(sc_f, len(gt["objects"]))
            together.append({"category": e["category"], "area": e["area"], "pictures": len(per_year),
                             "AP_together": round(ap_f, 3),
                             "AP_single_mean": round(float(np.mean(ap_single)), 3)})
            log(f"  {key:22s} {len(per_year)} pictures together: AP {ap_f:.2f}")
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
    pos = [r for r in rows if not r.get("negative")]
    by_vis = {}
    for v_ in ("visible", "doubtful", "not visible", "not stated"):
        sel = [r for r in pos if r.get("visibility") == v_]
        if sel:
            by_vis[v_] = {"pictures": len(sel), "AP": round(float(np.mean([x["AP"] for x in sel])), 2),
                          "recall": round(float(np.mean([x["recall"] or 0 for x in sel])), 2)}
            log(f"marked things {v_:12s} pictures {len(sel):2d}  recall {by_vis[v_]['recall']:.2f}  AP {by_vis[v_]['AP']:.2f}")
    if together:
        a_t = float(np.mean([t["AP_together"] for t in together]))
        a_s = float(np.mean([t["AP_single_mean"] for t in together]))
        log(f"years together   places {len(together):2d}  AP {a_t:.2f}  (the same places, one picture at a time: {a_s:.2f})")
        summary["years_together"] = {"places": len(together), "AP": round(a_t, 2), "AP_single_pictures": round(a_s, 2)}
    neg = [r for r in rows if r.get("negative")]
    if neg:
        log(f"empty places     pictures {len(neg):2d}  spots scored >= 0.5 per picture: "
            f"{np.mean([r['false_strong'] for r in neg]):.1f}")
    summary["by_visibility"] = by_vis
    if out_dir:
        with open(os.path.join(out_dir, "evaluation.json"), "w") as fh:
            json.dump({"pictures": rows, "summary": summary, "years_together": together}, fh, indent=1, default=float)
    return rows, summary
