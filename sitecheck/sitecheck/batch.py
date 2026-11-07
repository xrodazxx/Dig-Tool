"""`sitecheck batch FOLDER`: many areas at once. FOLDER contains one sub-folder per area,
each with screenshots of that area and a .txt file with its coordinates, e.g.

    russland/1/1.txt          53°00′57.00″ N, 49°30′57.16″ E
    russland/1/file.webp
    russland/1/file (1).webp

Writes one result folder per area plus summary.md and overview.jpg.
"""
from __future__ import annotations

import glob
import os
import re

import cv2
import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp")

_NUM = r"(\d+(?:[.,]\d+)?)"


def parse_coords(text: str):
    """DMS ('53°00′57.00″ N, 49°30′57.16″ E', also Russian С/Ю/В/З) or decimal ('53.0158, 49.5159')."""
    t = text.replace(",", " , ").replace("’", "′").replace("'", "′").replace("''", "″").replace('"', "″")
    dms = re.findall(_NUM + r"\s*°\s*" + _NUM + r"?\s*′?\s*" + _NUM + r"?\s*″?\s*([NSEWСЮВЗnsew])", t)
    if len(dms) >= 2:
        vals = []
        for d, m, s, h in dms[:2]:
            v = float(d.replace(",", ".")) + float((m or "0").replace(",", ".")) / 60 + float((s or "0").replace(",", ".")) / 3600
            if h.upper() in ("S", "W", "Ю", "З"):
                v = -v
            vals.append((h.upper(), v))
        lat = next(v for h, v in vals if h in ("N", "S", "С", "Ю"))
        lon = next(v for h, v in vals if h in ("E", "W", "В", "З"))
        return lat, lon
    nums = re.findall(r"-?\d+\.\d+", text)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    raise ValueError(f"no coordinates found in: {text[:80]!r}")


def _natural(s):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]


def _sharpness(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return 0
    return float(cv2.Laplacian(g, cv2.CV_64F).var()) * g.size


def attribution_mask(path):
    """Google Earth / Bing / Yandex print the imagery credit at the bottom centre."""
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = g.shape
    return (w * 0.3, h - max(22, h * 0.045), w * 0.7, h)


def run_batch(folder, out_root, mpp=None, log=print):
    from .image_mode import run_images
    areas = sorted([d for d in glob.glob(os.path.join(folder, "*")) if os.path.isdir(d)],
                   key=lambda d: _natural(os.path.basename(d)))
    if not areas:
        areas = [folder]
    os.makedirs(out_root, exist_ok=True)
    rows = []
    for d in areas:
        name = os.path.basename(d.rstrip("/"))
        imgs = sorted([f for f in glob.glob(os.path.join(d, "*")) if f.lower().endswith(IMG_EXT)], key=_sharpness,
                      reverse=True)                    # sharpest, largest first = reference
        txts = glob.glob(os.path.join(d, "*.txt"))
        if not imgs:
            log(f"[{name}] no images, skipped")
            continue
        lat = lon = None
        if txts:
            try:
                with open(txts[0], encoding="utf-8", errors="ignore") as fh:
                    lat, lon = parse_coords(fh.read())
            except ValueError as ex:
                log(f"[{name}] {ex}")
        out = os.path.join(out_root, name)
        log(f"[{name}] {len(imgs)} image(s), {lat}, {lon}")
        try:
            rep = run_images(imgs, out, lat, lon, mpp=mpp, masks=[attribution_mask(p) for p in imgs[:1]])
        except Exception as ex:  # noqa: BLE001 - one bad area must not stop the batch
            log(f"[{name}] FAILED: {ex}")
            rows.append({"name": name, "error": str(ex)})
            continue
        F = rep["features"]
        kinds = {}
        for f in F:
            k = f.get("kind") or f["type"]
            kinds[k] = kinds.get(k, 0) + 1
        strong = [f for f in F if len(f.get("seen_in_images", [])) > 1 or f["confidence"] >= 0.8]
        rows.append({"name": name, "lat": lat, "lon": lon, "images": len(imgs), "score": rep["score"]["points"],
                     "band": rep["score"]["band"], "n": len(F), "strong": len(strong), "kinds": kinds,
                     "aligned": [i.get("ok", True) for i in rep["data"]["images"]],
                     "sun": [i["sun_azimuth_used"] for i in rep["data"]["per_image"]], "out": out})
    _summary(rows, out_root)
    return rows


def _summary(rows, out_root):
    lines = ["# sitecheck batch summary", "",
             "Screenshot mode: rough evidence, check every candidate. Scale assumed unless given.", "",
             "| area | coordinates | imgs | aligned | score | candidates | strong | what was found |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['name']} | | | | | | | FAILED: {r['error'][:60]} |")
            continue
        found = ", ".join(f"{v} {k}" for k, v in sorted(r["kinds"].items(), key=lambda kv: -kv[1]))
        al = "yes" if all(r["aligned"]) else "partly" if any(r["aligned"][1:]) else ("-" if r["images"] == 1 else "no")
        lines.append(f"| {r['name']} | {r['lat']:.5f}, {r['lon']:.5f} | {r['images']} | {al} | {r['score']} ({r['band']}) "
                     f"| {r['n']} | {r['strong']} | {found} |")
    with open(os.path.join(out_root, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    thumbs = []
    for r in rows:
        p = os.path.join(r.get("out", ""), "candidates.png")
        if os.path.exists(p):
            im = cv2.imread(p)
            h, w = im.shape[:2]
            s = 520 / w
            im = cv2.resize(im, (520, int(h * s)), interpolation=cv2.INTER_AREA)
            cv2.putText(im, f"area {r['name']}  score {r['score']}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 3)
            cv2.putText(im, f"area {r['name']}  score {r['score']}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1)
            thumbs.append(im)
    if thumbs:
        cols = 3
        th = max(t.shape[0] for t in thumbs)
        sheet = np.full((th * ((len(thumbs) + cols - 1) // cols), 520 * cols, 3), 255, np.uint8)
        for i, t in enumerate(thumbs):
            y, x = (i // cols) * th, (i % cols) * 520
            sheet[y:y + t.shape[0], x:x + 520] = t
        cv2.imwrite(os.path.join(out_root, "overview.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


def rebuild_summary(out_root):
    """Re-create summary.md / overview.jpg from the result folders (e.g. after `review`)."""
    import json
    rows = []
    for d in sorted([d for d in glob.glob(os.path.join(out_root, "*")) if os.path.isdir(d)],
                    key=lambda d: _natural(os.path.basename(d))):
        p = os.path.join(d, "report.json")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as fh:
            rep = json.load(fh)
        F = rep["features"]
        kinds = {}
        for f in F:
            k = f.get("kind") or f["type"]
            kinds[k] = kinds.get(k, 0) + 1
        rows.append({"name": os.path.basename(d), "lat": rep["spot"]["lat"], "lon": rep["spot"]["lon"],
                     "images": len(rep["data"]["images"]), "score": rep["score"]["points"],
                     "band": rep["score"]["band"], "n": len(F),
                     "strong": len([f for f in F if len(f.get("seen_in_images", [])) > 1 or f["confidence"] >= 0.8]),
                     "kinds": kinds, "aligned": [i.get("ok", True) for i in rep["data"]["images"]], "out": d})
    _summary(rows, out_root)
    return rows
