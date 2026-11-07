"""Classical (non-AI) detection of man-made micro-relief on a DEM.

Detections are *candidates*: every one gets an id, a confidence and
user_status="unreviewed". The user confirms or rejects them; the score only
counts what is not rejected, and confirmed features weigh more.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from skimage import measure, morphology


def noise_level(lrm: np.ndarray) -> float:
    """Robust estimate of micro-relief 'noise' (MAD of the local relief model)."""
    return float(1.4826 * np.median(np.abs(lrm - np.median(lrm))))


def _threshold(lrm: np.ndarray, min_abs: float) -> float:
    return max(min_abs, 2.5 * noise_level(lrm))


# --------------------------------------------------------------------------- pits

def detect_pits(z, lrm, cell_m, min_depth=0.15, min_diam=1.5, max_diam=20.0):
    """Closed depressions (house pits, cellar pits, storage pits, craters, root throws).

    Depth = how much water the hollow would hold: filled surface minus DEM."""
    seed = z.copy()
    seed[1:-1, 1:-1] = z.max()
    filled = morphology.reconstruction(seed, z, method="erosion")
    depth = filled - z
    thr = max(min_depth, 2.0 * noise_level(lrm))
    labels = measure.label(depth > thr * 0.5)  # region = whole hollow, filtered by max depth below
    out = []
    for p in measure.regionprops(labels, intensity_image=depth):
        d = float(p.intensity_max)
        diam = p.equivalent_diameter_area * cell_m
        if d < thr or not (min_diam <= diam <= max_diam):
            continue
        if p.solidity < 0.7:
            continue
        r, c = p.centroid_weighted if hasattr(p, "centroid_weighted") else p.centroid
        roundness = p.axis_minor_length / max(p.axis_major_length, 1e-6)
        if roundness < 0.5:
            continue
        conf = min(1.0, 0.2 + 0.3 * min(d / (3 * thr), 1) + 0.35 * roundness + 0.15 * p.solidity)
        out.append({
            "type": "pit",
            "row": float(r), "col": float(c),
            "diameter_m": round(diam, 1),
            "depth_m": round(d, 2),
            "volume_m3": round(float(p.image_intensity[p.image].sum()) * cell_m ** 2, 1),
            "roundness": round(roundness, 2),
            "confidence": round(conf, 2),
            "note": "house/cellar pit, storage pit, crater, quarry pit or root throw",
        })
    return out


# --------------------------------------------------------------------------- rectangles

def detect_rectangles(lrm, cell_m, min_side=4.0, max_side=40.0, min_rel=0.08):
    """Rectangular outlines and platforms: foundations, wall lines, house platforms,
    enclosures. Looks at raised (+) and sunken (-) micro-relief separately."""
    thr = _threshold(lrm, min_rel)
    out = []
    k = max(1, int(round(1.0 / cell_m)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    for sign, kind in ((1, "raised"), (-1, "sunken")):
        mask = ((sign * lrm) > thr).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        band = max(1, int(round(1.5 / cell_m)))
        for cnt in contours:
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull) * cell_m ** 2
            if hull_area < min_side ** 2:
                continue
            (cx, cy), (w, h), ang = cv2.minAreaRect(hull)
            w_m, h_m = w * cell_m, h * cell_m
            long_side, short_side = max(w_m, h_m), min(w_m, h_m)
            if short_side < min_side or long_side > max_side or long_side / short_side > 4:
                continue
            rectangularity = hull_area / (w_m * h_m)
            if rectangularity < 0.8:
                continue
            box = cv2.boxPoints(((cx, cy), (w, h), ang))
            # Two ways to be a rectangle: a solid platform/floor, or an outline (wall, bank,
            # ditch) that runs along most of the rectangle's perimeter, gaps allowed.
            inside = np.zeros_like(mask)
            cv2.fillPoly(inside, [np.round(box).astype(np.int32)], 1)
            solid = float((inside & mask).sum() / max(inside.sum(), 1))   # share of raised/sunken pixels
            outline = np.zeros_like(mask)
            cv2.polylines(outline, [np.round(box).astype(np.int32)], True, 1, thickness=band)
            fat = cv2.dilate(mask, kernel)
            support = float((outline & fat).sum() / max(outline.sum(), 1))
            if solid < 0.75:
                # an outline must have walls on every side and real corners (rings have none)
                sides = []
                for k2 in range(4):
                    seg = np.zeros_like(mask)
                    cv2.line(seg, tuple(np.round(box[k2]).astype(int)),
                             tuple(np.round(box[(k2 + 1) % 4]).astype(int)), 1, thickness=band)
                    sides.append((seg & fat).sum() / max(seg.sum(), 1))
                corners = sum(fat[max(0, int(y) - band):int(y) + band + 1,
                                  max(0, int(x) - band):int(x) + band + 1].any() for x, y in box)
                if support < 0.72 or min(sides) < 0.4 or corners < 3:
                    continue
            shape = "platform / solid" if solid >= 0.75 else "outline"
            q = max((solid - 0.75) / 0.25, (support - 0.6) / 0.4)
            conf = min(1.0, 0.45 + 0.35 * max(q, 0) + 0.2 * (rectangularity - 0.8) / 0.2)
            out.append({
                "type": "rectangle",
                "relief": kind,
                "shape": shape,
                "row": float(cy), "col": float(cx),
                "length_m": round(long_side, 1), "width_m": round(short_side, 1),
                "orientation_deg": round(float(ang if w >= h else ang + 90) % 180, 0),
                "rectangularity": round(rectangularity, 2),
                "perimeter_support": round(support, 2),
                "corners_px": [[float(x), float(y)] for x, y in box],
                "confidence": round(conf, 2),
                "note": ("foundation / wall line / bank enclosure" if kind == "raised" and shape == "outline"
                         else "house platform / mound" if kind == "raised"
                         else "rectangular ditch or inner edge of walls" if shape == "outline"
                         else "sunken floor / cellar / rectangular pit"),
            })
    return _dedupe(out, cell_m, 3.0)


# --------------------------------------------------------------------------- linear

def detect_linear(lrm, cell_m, min_length=40.0, min_rel=0.1):
    """Long narrow features: ditches, sunken roads, hollow ways, banks, field boundaries.
    Natural gullies also look like this; straightness helps tell them apart."""
    thr = _threshold(lrm, min_rel)
    out = []
    for sign, kind in ((-1, "ditch / sunken road"), (1, "bank / embankment / raised road")):
        mask = morphology.remove_small_objects((sign * lrm) > thr, max_size=max(4, int(20 / cell_m ** 2)))
        mask = morphology.closing(mask, morphology.disk(max(1, int(round(1.5 / cell_m))))).astype(bool)
        labels = measure.label(mask)
        for p in measure.regionprops(labels):
            if p.axis_major_length * cell_m < min_length:
                continue
            if p.axis_minor_length / max(p.axis_major_length, 1e-6) > 0.25:
                continue
            skel = morphology.skeletonize(p.image)
            pts = np.argwhere(skel)
            if len(pts) < 2:
                continue
            length = len(pts) * cell_m
            # straightness: 1 - (max sideways deviation from the end-to-end chord) / (chord / 4)
            sub = pts[:: max(1, len(pts) // 400)]
            d = np.linalg.norm(sub[:, None] - sub[None], axis=2)
            i, j = np.unravel_index(d.argmax(), d.shape)
            e1, e2 = sub[i], sub[j]
            chord = max(float(d[i, j]), 1e-6)
            u = (e2 - e1) / chord
            dev = np.abs((pts[:, 0] - e1[0]) * u[1] - (pts[:, 1] - e1[1]) * u[0]).max()
            straightness = float(np.clip(1 - 4 * dev / chord, 0, 1))
            length = max(length, chord * cell_m)
            r0, c0 = p.bbox[:2]
            width = p.area * cell_m ** 2 / length
            conf = min(1.0, 0.3 + 0.4 * straightness + 0.3 * min(length / 150, 1))
            out.append({
                "type": "linear",
                "relief": "sunken" if sign < 0 else "raised",
                "row": float(p.centroid[0]), "col": float(p.centroid[1]),
                "length_m": round(length, 0),
                "mean_width_m": round(width, 1),
                "straightness": round(straightness, 2),
                "orientation_deg": round((90 - math.degrees(p.orientation)) % 180, 0),
                "ends_px": [[float(e1[0] + r0), float(e1[1] + c0)],
                            [float(e2[0] + r0), float(e2[1] + c0)]],
                "confidence": round(conf, 2),
                "note": kind + ("; very straight - likely man-made" if straightness > 0.9
                                else "; curved - could be a natural gully or stream bed"),
            })
    return out


def _dedupe(items, cell_m, dist_m):
    items = sorted(items, key=lambda f: -f["confidence"])
    kept = []
    for f in items:
        if all(math.hypot(f["row"] - k["row"], f["col"] - k["col"]) * cell_m > dist_m for k in kept):
            kept.append(f)
    return kept


def _near_border(f, shape, margin):
    h, w = shape
    if f["type"] == "linear":   # roads may cross the border; drop only edge artefacts
        (r1, c1), (r2, c2) = f["ends_px"]
        on_edge = lambda r, c: r < margin or c < margin or r > h - 1 - margin or c > w - 1 - margin
        return on_edge(r1, c1) and on_edge(r2, c2) and (
            min(abs(r1 - r2), abs(c1 - c2)) < margin)  # runs along the edge
    return (f["row"] < margin or f["col"] < margin
            or f["row"] > h - 1 - margin or f["col"] > w - 1 - margin)


def _inside(pt, corners):
    return cv2.pointPolygonTest(np.array(corners, np.float32), (pt[1], pt[0]), False) >= 0


def _line_angle(f):
    (r1, c1), (r2, c2) = f["ends_px"]
    return math.degrees(math.atan2(-(r2 - r1), c2 - c1)) % 180


def _seg_dist(p, a, b):
    p, a, b = map(np.asarray, (p, a, b))
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def _link_parallel(lines, cell_m):
    """A ditch/road throws up banks on both sides (spoil, or just the relief model's
    response). Fold weaker parallel neighbours into the stronger feature."""
    lines.sort(key=lambda f: -f["confidence"])
    for i, strong in enumerate(lines):
        if strong.get("part_of"):
            continue
        for weak in lines[i + 1:]:
            if weak.get("part_of") or weak["relief"] == strong["relief"]:
                continue
            dang = abs(_line_angle(strong) - _line_angle(weak)) % 180
            if min(dang, 180 - dang) > 15:
                continue
            gap = _seg_dist((weak["row"], weak["col"]), *strong["ends_px"]) * cell_m
            if gap < 10:
                weak["part_of"] = strong
                weak["confidence"] = round(weak["confidence"] * 0.3, 2)
    for f in lines:
        if f.get("part_of"):
            f["note"] = "flank of a neighbouring linear feature (bank beside a ditch/road)"
            f.pop("part_of")


def _flag_contour_parallel(lines, aspect, slope):
    """Lines running along the contour on a real slope may be natural terrace edges
    or slope breaks - or a terraced road. Say so instead of guessing."""
    for f in lines:
        r, c = int(f["row"]), int(f["col"])
        if slope[r, c] < 3:
            continue
        contour_dir = (90 - aspect[r, c] + 90) % 180   # perpendicular to downslope, math angle
        d = abs(_line_angle(f) - contour_dir) % 180
        if min(d, 180 - d) < 20:
            f["note"] += "; runs along the contour - could also be a natural terrace edge / river bank"
            f["confidence"] = round(f["confidence"] * 0.8, 2)


def detect_all(z, lrm, cell_m, aspect=None, slope=None) -> list[dict]:
    pits = detect_pits(z, lrm, cell_m)
    rects = detect_rectangles(lrm, cell_m)
    lines = detect_linear(lrm, cell_m)
    margin = max(3, int(round(10 / cell_m)))

    # a pit (or its spoil ring) also passes as a small rectangle - keep it as a pit only
    # (outlines passed the corner test, so rings are already gone; a floor hollow inside a
    #  real foundation must not delete the foundation)
    rects = [r for r in rects if not (r["shape"] != "outline" and any(
        math.hypot(p["row"] - r["row"], p["col"] - r["col"]) * cell_m < 3 for p in pits))]
    # banks alongside a road/ditch are part of that linear feature
    def along_line(r):
        for ln in lines:
            if _seg_dist((r["row"], r["col"]), *ln["ends_px"]) * cell_m > 8:
                continue
            d = abs(r["orientation_deg"] - _line_angle(ln)) % 180
            if min(d, 180 - d) < 20 or r["width_m"] < 6:
                return True
        return False
    rects = [r for r in rects if not along_line(r)]
    for ln in lines:
        if ln["mean_width_m"] > 8:
            ln["confidence"] = round(ln["confidence"] * 0.5, 2)
            ln["note"] = "wide channel - most likely a natural watercourse or valley floor"
    _link_parallel(lines, cell_m)
    if aspect is not None and slope is not None:
        _flag_contour_parallel(lines, aspect, slope)
    feats = [f for f in pits + rects + lines if not _near_border(f, z.shape, margin)]
    feats = [f for f in feats if f["confidence"] >= 0.4]

    for i, f in enumerate(sorted(feats, key=lambda f: -f["confidence"]), 1):
        f["id"] = i
        f["detected_by"] = "algorithm"
        f["user_status"] = "unreviewed"  # user sets: confirmed / rejected
    for p in (f for f in feats if f["type"] == "pit"):
        for r in (f for f in feats if f["type"] == "rectangle"):
            if _inside((p["row"], p["col"]), r["corners_px"]):
                p["inside_feature"] = r["id"]
                p["note"] = f"hollow inside rectangle #{r['id']} - sunken floor or cellar?"
    for f in feats:  # plain floats for JSON
        for k, v in f.items():
            if isinstance(v, np.floating):
                f[k] = float(v)
    return sorted(feats, key=lambda f: f["id"])
