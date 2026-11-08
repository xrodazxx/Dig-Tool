"""Labelled examples: the user's marked screenshots -> ground truth for testing and training.

Folder layout (as in Rus_earth):
    <category>/<area>/
        clear* | clean* | original* | plane*      the clean reference picture
        *mark* | *mix* | quad*                    the same view with red marks drawn on it
                                                  (several marked copies are fine: each is paired
                                                  with the picture it was drawn on, marks are merged)
        anything else                             the same place in other seasons / years
        *.txt                                     coordinates (optional)
    <category>/<picture>                          pictures straight in a category folder with no
                                                  marks = places where nothing ever stood
                                                  ("negatives": every candidate there is false)
Pictures of one area with the same size are taken as already aligned.

Marks are read from the difference between the marked and the clean picture:
    closed loops   -> objects (round = pit / patch, 4 corners = rectangle / house remains)
    zigzags        -> zones ("there is something in here")
    long lines     -> streets / roads (two parallel lines = a road strip, straight or curved)
    a loop around several other marks -> enclosure (yard / homestead): the marks inside count,
                      other candidates inside it are neither right nor wrong
A box touching the picture edge with writing inside (a legend) is ignored.
"""
from __future__ import annotations

import glob
import math
import os

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import measure
from skimage.morphology import skeletonize

from . import imagery as im
from . import scale as SC
from .batch import parse_coords

REF_KEYS = ("clear", "clean", "original", "plane")
MARK_KEYS = ("mark", "mix", "quad", "note")
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")


def _images(d):
    return sorted(f for f in glob.glob(os.path.join(d, "*")) if f.lower().endswith(IMG_EXT))


def _small(path, n=96):
    from PIL import Image
    im_ = Image.open(path).convert("RGB")
    return im_.size, np.asarray(im_.resize((n, n)), np.float32)


def discover(root):
    """[{category, area, path, ref, pairs[(ref, marked)], others, lat, lon, negative}]"""
    out = []
    # pictures lying directly in a category folder, without marks: places with nothing there.
    # (a category folder that itself holds a marked picture is one area: handled below)
    area_dirs = sorted(glob.glob(os.path.join(root, "*", "*")))
    for cdir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(cdir):
            continue
        if any(any(k in os.path.basename(f).lower() for k in MARK_KEYS) for f in _images(cdir)):
            area_dirs.append(cdir)
            continue
        imgs = _images(cdir)
        sc = SC.read_folder(cdir, [os.path.basename(f) for f in imgs]) if imgs else {}
        for f in imgs:
            if any(k in os.path.basename(f).lower() for k in MARK_KEYS):
                continue
            out.append({"category": os.path.basename(cdir), "area": os.path.splitext(os.path.basename(f))[0],
                        "path": cdir, "ref": f, "pairs": [], "marked": None, "others": [], "lat": None,
                        "lon": None, "negative": True, "scale": sc})
    for d in area_dirs:
        if not os.path.isdir(d):
            continue
        files = _images(d)
        names = [os.path.basename(f).lower() for f in files]
        marked = [f for f, n in zip(files, names) if any(k in n for k in MARK_KEYS)]
        plain = [f for f in files if f not in marked]
        if not marked or not plain:
            continue
        # the main marked copy first ("marks.png" before "FieldMarks.png")
        marked.sort(key=lambda f: (len(os.path.basename(f)), f))
        small = {f: _small(f) for f in files}
        pairs = []
        for m in marked:
            same = [f for f in plain if small[f][0] == small[m][0]]
            if same:     # the picture the marks were drawn on = the most similar one
                ref = min(same, key=lambda f: float(np.abs(small[f][1] - small[m][1]).mean()))
            else:
                refs = [f for f, n in zip(files, names) if f in plain and n.startswith(REF_KEYS)]
                ref = refs[0] if refs else plain[0]
            pairs.append((ref, m))
        ref = pairs[0][0]
        others = [f for f in plain if f != ref]
        lat = lon = None
        for t in glob.glob(os.path.join(d, "*.txt")):
            try:
                with open(t, encoding="utf-8", errors="ignore") as fh:
                    lat, lon = parse_coords(fh.read())
            except ValueError:
                pass
        top = os.path.dirname(d) == os.path.normpath(root)
        out.append({"category": os.path.basename(d) if top else os.path.basename(os.path.dirname(d)),
                    "area": "1" if top else os.path.basename(d), "path": d,
                    "ref": ref, "pairs": pairs, "marked": pairs[0][1], "others": others, "lat": lat, "lon": lon,
                    "negative": False, "scale": SC.read_folder(d, [os.path.basename(f) for f in files])})
    return out


def load_rgb(path):
    return im.load(path)[0]


def align_pair(ref_rgb, rgb):
    """Warp `rgb` into the reference pixel grid. Returns (warped, warp, correlation, scale)."""
    valid_r = np.ones(ref_rgb.shape[:2], bool)
    valid = np.ones(rgb.shape[:2], bool)
    w, cc = im.align(ref_rgb, valid_r, rgb, valid)
    sc = float(np.sqrt(abs(np.linalg.det(w[:, :2]))))
    return im.warp_to(ref_rgb.shape[:2], rgb, w), w, cc, sc


def red_mask(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (r > 0.55) & (r - g > 0.3) & (r - b > 0.3)


def _along_hull(p, hull):
    """An unclosed hand-drawn loop runs along its own convex hull for most of the way round;
    a zigzag (hatching) crosses the inside."""
    edge = hull & ~ndi.binary_erosion(hull, iterations=1)
    d_edge = ndi.distance_transform_edt(~edge)
    tol = max(4.0, 0.12 * math.sqrt(hull.sum()))
    on = d_edge[p.coords[:, 0], p.coords[:, 1]] <= tol
    stroke = np.zeros(hull.shape, bool)
    stroke[p.coords[:, 0], p.coords[:, 1]] = True
    covered = (ndi.distance_transform_edt(~stroke)[edge] <= tol).mean()
    return on.mean() > 0.85 and covered > 0.6


def _holes_as_objects(p, add_object, H, W):
    """Small closed loops attached to a road line (a pit circled right next to the road)."""
    filled = ndi.binary_fill_holes(p.image)
    holes = measure.label(filled & ~p.image)
    r0, c0 = p.bbox[:2]
    for h in measure.regionprops(holes):
        hr0, hc0, hr1, hc1 = h.bbox
        if 6 < h.area < 0.01 * H * W and (hr1 - hr0) < 3 * (hc1 - hc0) and (hc1 - hc0) < 3 * (hr1 - hr0):
            m = np.zeros(filled.shape, bool)
            m[h.coords[:, 0], h.coords[:, 1]] = True
            add_object(ndi.binary_dilation(m, iterations=2), r0, c0)


# colours of a marking scheme (hue ranges in degrees, OpenCV HSV with H 0-360)
COLOUR_RULES = {
    "red":    lambda H, S, V: (S > 0.45) & (V > 0.35) & ((H < 15) | (H > 340)),
    "orange": lambda H, S, V: (S > 0.45) & (V > 0.45) & (H >= 15) & (H < 42),
    "yellow": lambda H, S, V: (S > 0.35) & (V > 0.5) & (H >= 42) & (H < 75),
    "green":  lambda H, S, V: (S > 0.35) & (V > 0.35) & (H >= 75) & (H < 170),
    "blue":   lambda H, S, V: (S > 0.35) & (V > 0.3) & (H >= 185) & (H < 265),
    "black":  lambda H, S, V: V < 0.25,
}
# what each colour means (the user's General_Info.txt); red = old one-colour packs
COLOUR_MEANING = {"red": "pit", "blue": "pit", "yellow": "foundation", "orange": "road",
                  "green": "rampart", "black": "vegetation"}


def colour_strokes(ref_rgb, marked):
    """Stroke pixels per colour: where the marked copy differs from the clean picture."""
    D = np.abs(marked - ref_rgb).max(-1) > 0.2
    b_ = max(4, int(0.006 * max(D.shape)))          # picture frame / warp border is not a mark
    D[:b_], D[-b_:], D[:, :b_], D[:, -b_:] = False, False, False, False
    hsv = cv2.cvtColor(np.clip(marked, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    out = {}
    for name, rule in COLOUR_RULES.items():
        m = D & rule(H, S, V)
        if name == "black":                       # must be darker than the picture underneath
            m &= V < cv2.cvtColor(np.clip(ref_rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)[..., 2] - 0.1
        m = ndi.binary_closing(m, iterations=2)
        if name != "red":                         # keep pen strokes only: thin and long, not blotches
            keep = np.zeros(m.shape, bool)
            Hh, Ww = m.shape
            for p in measure.regionprops(measure.label(m)):
                sk = float(skeletonize(p.image).sum())
                rr_, cc_ = p.coords[:, 0], p.coords[:, 1]
                edge_d = np.minimum(np.minimum(rr_, Hh - 1 - rr_), np.minimum(cc_, Ww - 1 - cc_))
                if np.mean(edge_d < 0.025 * max(Hh, Ww)) > 0.6:
                    continue                      # warp padding / frame along the picture edge
                if sk >= 25 and p.area <= 7.0 * sk:
                    keep[p.coords[:, 0], p.coords[:, 1]] = True
            m = keep
        out[name] = m
    # marks drawn on another capture of the place (re-aligned): the pictures differ everywhere,
    # so dark/colour differences are not pen strokes - read red only, as before
    out["_drawn_on_same_picture"] = bool((D & ~np.any([out[k] for k in COLOUR_RULES], axis=0)).mean() < 0.05)
    return out


def _strip(mask, H, W):
    """A band drawn as two lines (road, rampart; zigzag filling allowed) -> filled strip + centre line."""
    m = ndi.binary_dilation(mask, iterations=2)
    keep = np.zeros(m.shape, bool)
    for p in measure.regionprops(measure.label(m)):
        r0, c0, r1, c1 = p.bbox
        if max(r1 - r0, c1 - c0) > 0.12 * max(H, W):       # drop stray dots
            keep[p.coords[:, 0], p.coords[:, 1]] = True
    if not keep.any():
        return keep, []
    # gap between the two lines -> closing size
    d_in = ndi.distance_transform_edt(~keep)
    filled = ndi.binary_fill_holes(keep)
    gap = d_in[filled & ~keep]
    it = max(4, np.percentile(gap, 90) + 2 if gap.size else 6)
    comps_ = [p for p in measure.regionprops(measure.label(keep))]
    if len(comps_) >= 2:                      # two separate side lines: half the distance between them
        comps_.sort(key=lambda p: -p.area)
        a_ = np.zeros(keep.shape, bool)
        a_[comps_[0].coords[:, 0], comps_[0].coords[:, 1]] = True
        da = ndi.distance_transform_edt(~a_)
        gap2 = float(np.median(da[comps_[1].coords[:, 0], comps_[1].coords[:, 1]]))
        if gap2 < 0.25 * min(H, W):
            it = max(it, gap2 / 2 + 2)
    it = int(min(it, 0.15 * min(H, W)))
    band = ndi.binary_fill_holes(ndi.binary_closing(keep, iterations=it)) | keep
    lines = []
    for p in measure.regionprops(measure.label(band)):
        sk = skeletonize(p.image)
        ys, xs = np.nonzero(sk)
        P = np.column_stack([ys + p.bbox[0], xs + p.bbox[1]]).astype(float)
        if len(P) < 2:
            continue
        cv_ = P - P.mean(0)
        _, _, vt = np.linalg.svd(cv_, full_matrices=False)
        order = np.argsort(cv_ @ vt[0])
        lines.append({"ends_px": [P[order[0]].tolist(), P[order[-1]].tolist()],
                      "centre_px": P[order][:: max(1, len(order) // 40)].tolist(),
                      "width_px": round(float(2 * p.area / max(len(P), 1)), 1), "double_line": True})
    return band, lines


def _lookalikes(mask, kind, H, W):
    """Black outlines (bushes/trees) and red crosses (animals): marked things that are NOT sites."""
    out = []
    for p in measure.regionprops(measure.label(mask)):
        r0, c0, r1, c1 = p.bbox
        if p.area < 8:
            continue
        filled = ndi.binary_fill_holes(p.image)
        hull = np.zeros(mask.shape, np.uint8)
        h_ = cv2.convexHull(np.column_stack([p.coords[:, 1], p.coords[:, 0]]).astype(np.int32))
        cv2.fillPoly(hull, [h_], 1)
        hull = hull.astype(bool)
        if filled.sum() - p.area > 0.15 * filled.sum():
            region = np.zeros(mask.shape, bool)
            region[r0:r1, c0:c1] = filled
        elif hull.sum() < 0.06 * H * W:
            region = hull
        else:
            continue                                   # a long edge line (forest edge): skip
        ys, xs = np.nonzero(region)
        out.append({"kind": kind, "row": float(ys.mean()), "col": float(xs.mean()),
                    "mask_rc": np.column_stack([ys, xs]).tolist()})
    return out


def _is_cross(p):
    """A small drawn X: not a loop, strokes cross in the middle of their own hull."""
    r0, c0, r1, c1 = p.bbox
    if max(r1 - r0, c1 - c0) > 60 or ndi.binary_fill_holes(p.image).sum() - p.area > 0.15 * p.area:
        return False
    sk = skeletonize(np.pad(p.image, 1))
    nb = ndi.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - 1
    ends = int(((nb == 1) & sk).sum())
    return bool(((nb >= 3) & sk).any()) and ends >= 3 and max(r1 - r0, c1 - c0) >= 8


def extract(ref_rgb, marked_warped):
    """Ground truth from the drawn marks. One-colour (red) packs: loops = objects, lines = roads.
    Several colours: the meaning of each colour (COLOUR_MEANING)."""
    cs = colour_strokes(ref_rgb, marked_warped)
    H, W = ref_rgb.shape[:2]
    same = cs.pop("_drawn_on_same_picture")
    n_px = {k: int(v.sum()) for k, v in cs.items()}
    multi = same and sum(1 for k, v in n_px.items() if k != "red" and v > 200) >= 1
    if not multi:
        g = _extract_strokes(red_mask(marked_warped) & ~red_mask(ref_rgb), "object")
        g["lookalikes"], g["ramparts"], g["rampart_lines"] = [], np.zeros((H, W), bool), []
        return g
    # several colours: pits (red/blue), foundations (yellow), road (orange), rampart (green),
    # vegetation (black outlines) and animals (red crosses) as named look-alikes
    red = cs["red"] | cs["blue"]
    lookalikes = []
    lab = measure.label(red)
    inside_loops = ndi.binary_fill_holes(red) & ~red        # marks drawn inside a red loop are pits
    for p in measure.regionprops(lab):
        if _is_cross(p) and not inside_loops[int(p.centroid[0]), int(p.centroid[1])] and \
                not ndi.binary_dilation(inside_loops, iterations=2)[p.coords[:, 0], p.coords[:, 1]].mean() > 0.5:
            red[p.coords[:, 0], p.coords[:, 1]] = False
            lookalikes += [dict(x, kind="animal") for x in _lookalikes(
                np.isin(lab, [p.label]), "animal", H, W)] or [
                {"kind": "animal", "row": float(p.centroid[0]), "col": float(p.centroid[1]),
                 "mask_rc": p.coords.tolist()}]
    g = _extract_strokes(red, "pit")
    gy = _extract_strokes(cs["yellow"], "foundation")
    g["objects"] += gy["objects"]
    g["enclosures"] |= gy["enclosures"]
    roads, road_lines = _strip(cs["orange"], H, W)
    g["roads"] = g["roads"] | gy["roads"] | roads
    g["streets"] += gy["streets"] + road_lines
    g["ramparts"], g["rampart_lines"] = _strip(cs["green"], H, W)
    g["lookalikes"] = lookalikes + _lookalikes(cs["black"], "vegetation", H, W)
    g["colours"] = {k: v for k, v in n_px.items() if v > 0}
    return g


def _extract_strokes(strokes, kind="object"):
    """Objects, zones and streets from strokes of one colour."""
    strokes = ndi.binary_closing(strokes, iterations=2)
    H, W = strokes.shape
    lab = measure.label(strokes)
    objects, zones, streets, legend = [], np.zeros(strokes.shape, bool), [], np.zeros(strokes.shape, bool)
    filled_all = ndi.binary_fill_holes(strokes)
    # legend: a big closed box touching the picture edge
    for p in measure.regionprops(measure.label(filled_all)):
        r0, c0, r1, c1 = p.bbox
        touches = r0 <= 3 or c0 <= 3 or r1 >= H - 3 or c1 >= W - 3
        if touches and p.area > 0.004 * H * W and (p.area - (lab[p.slice] > 0).sum()) > 0.3 * p.area:
            legend[p.slice] |= p.image
    legend = ndi.binary_dilation(legend, iterations=4)
    # open strokes whose hull is big and touches the picture edge = legend boxes (+ their contents)
    comps = list(measure.regionprops(lab))
    hulls = {}
    for p in comps:
        hull = np.zeros(strokes.shape, np.uint8)
        h_ = cv2.convexHull(np.column_stack([p.coords[:, 1], p.coords[:, 0]]).astype(np.int32))
        cv2.fillPoly(hull, [h_], 1)
        hulls[p.label] = hull.astype(bool)
        r0, c0, r1, c1 = p.bbox
        m_ = 0.03 * max(H, W)
        touches = r0 <= m_ or c0 <= m_ or r1 >= H - m_ or c1 >= W - m_
        if touches and hulls[p.label].sum() > 0.02 * H * W and p.area < 0.25 * hulls[p.label].sum():
            # a legend box has writing inside it: several other strokes within its outline
            # (letters are open strokes; closed loops inside = marked objects in a yard, not writing)
            inside_n = sum(1 for q in comps if q.label != p.label
                           and hulls[p.label][int(q.centroid[0]), int(q.centroid[1])]
                           and ndi.binary_fill_holes(q.image).sum() - q.area < 0.15 * q.area)
            sk_ = float(skeletonize(p.image).sum())
            is_line = max(r1 - r0, c1 - c0) > 0.5 * sk_          # a road line, not a box
            if inside_n >= 3 and not is_line:
                legend[r0:r1, c0:c1] = True      # whole box (it may be drawn unclosed)
    legend = ndi.binary_dilation(legend, iterations=6)

    def add_object(mask, r0=0, c0=0):
        cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnt, key=cv2.contourArea)
        approx = cv2.approxPolyDP(c, 0.06 * cv2.arcLength(c, True), True)
        (_, _), (w, h), _ = cv2.minAreaRect(c)
        rect_fill = cv2.contourArea(c) / max(w * h, 1)
        # a hand-drawn square fills its bounding rectangle (~0.9+), a hand-drawn circle ~0.79
        shape = "rectangle" if 4 <= len(approx) <= 5 and rect_fill > 0.87 else "round"
        ys, xs = np.nonzero(mask)
        objects.append({"row": float(ys.mean() + r0), "col": float(xs.mean() + c0), "kind": kind,
                        "diameter_px": float(2 * math.sqrt(mask.sum() / math.pi)), "shape": shape,
                        "mask_rc": (np.column_stack([ys + r0, xs + c0])).tolist()})

    line_pts = []
    for p in comps:
        if legend[int(p.centroid[0]), int(p.centroid[1])]:
            continue
        filled = ndi.binary_fill_holes(p.image)
        inside = filled.sum() - p.area
        r0, c0, r1, c1 = p.bbox
        hull = hulls[p.label]
        pts = p.coords.astype(float)
        cvec = pts - pts.mean(0)
        _, sv, vt = np.linalg.svd(cvec, full_matrices=False)
        thin = (sv[1] if len(sv) > 1 else 0.0) / max(sv[0], 1e-6) < 0.08
        L = max(r1 - r0, c1 - c0)
        # a long line, straight or curved: skeleton long, stroke thin
        skel_len = float(skeletonize(p.image).sum())
        # (an open hand-drawn oval also has a long skeleton, but it doubles back: extent << length)
        # (both sides of a road touching = one piece with twice the length: then it must be narrow)
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.column_stack([p.coords[:, 1], p.coords[:, 0]]).astype(np.float32))
        narrow = min(rw, rh) < 0.3 * max(rw, rh, 1)
        long_line = (skel_len > 0.18 * max(H, W) and p.area < 6.0 * skel_len and L > 0.15 * max(H, W)
                     and (L > 0.5 * skel_len or (narrow and L > 0.3 * skel_len)))
        # a closed loop that is a long narrow strip = the two sides of a road joined at the ends
        fill_area = float(filled.sum())
        strip = (inside > 0.15 * fill_area and skel_len > 0.4 * max(H, W) and L > 0.35 * max(H, W)
                 and 2.0 * fill_area / max(skel_len, 1) < 0.08 * min(H, W))
        if strip or (long_line and not (inside > 0.15 * fill_area and inside > 6)):
            p.both_sides = (strip or L <= 0.5 * skel_len) and min(rw, rh) > 6
            p.side_gap = float(min(rw, rh)) if p.both_sides else 0.0
            line_pts.append(p)
            _holes_as_objects(p, add_object, H, W)
        elif inside > 0.15 * fill_area and inside > 6:                  # closed loop -> object
            # several boxes drawn sharing walls (a row of foundations): one object per box
            holes = [h for h in measure.regionprops(measure.label(filled & ~p.image)) if h.area > 6]
            big = max((h.area for h in holes), default=0)
            parts = [h for h in holes if h.area > 0.15 * big]
            if len(parts) >= 2:
                for h in parts:
                    m = np.zeros(filled.shape, bool)
                    m[h.coords[:, 0], h.coords[:, 1]] = True
                    add_object(ndi.binary_dilation(m, iterations=2), r0, c0)
            else:
                add_object(filled, r0, c0)
        elif thin and L > 0.15 * max(H, W):                             # long straight line -> street
            p.both_sides, p.side_gap = False, 0.0
            line_pts.append(p)
        elif hull.sum() < 0.06 * H * W and _along_hull(p, hull) and \
                (r1 - r0) < 3.5 * (c1 - c0) and (c1 - c0) < 3.5 * (r1 - r0):
            add_object(hull)                                             # almost closed small loop
        elif L > 20:                                                     # zigzag -> zone
            zones |= hull
    # roads: lines (single, or two side by side) -> centre line + the strip they cover
    roads = np.zeros(strokes.shape, bool)
    used = set()
    for i, p in enumerate(line_pts):
        if i in used:
            continue
        group = [p]
        mi = np.zeros(strokes.shape, bool)
        mi[p.coords[:, 0], p.coords[:, 1]] = True
        dist_i = ndi.distance_transform_edt(~mi)
        for j in range(i + 1, len(line_pts)):
            q = line_pts[j]
            if j in used:
                continue
            dq = dist_i[q.coords[:, 0], q.coords[:, 1]]
            if np.median(dq) < 0.12 * min(H, W):        # the other side of the same road
                group.append(q)
                used.add(j)
        gm = np.zeros(strokes.shape, bool)
        for q in group:
            gm[q.coords[:, 0], q.coords[:, 1]] = True
        width = max(q.side_gap for q in group)
        if len(group) > 1:
            a = np.zeros(strokes.shape, bool)
            a[group[0].coords[:, 0], group[0].coords[:, 1]] = True
            db = ndi.distance_transform_edt(~a)
            width = float(np.median(np.concatenate([db[q.coords[:, 0], q.coords[:, 1]] for q in group[1:]])))
        strip_m = ndi.binary_closing(gm, iterations=max(2, int(width / 2) + 2))
        strip_m = ndi.binary_fill_holes(strip_m) | ndi.binary_dilation(gm, iterations=max(3, int(width / 2)))
        roads |= strip_m
        sk = skeletonize(strip_m)
        ys, xs = np.nonzero(sk)
        P = np.column_stack([ys, xs]).astype(float)
        if len(P) < 2:
            continue
        cvec2 = P - P.mean(0)
        _, _, vt2 = np.linalg.svd(cvec2, full_matrices=False)
        t = cvec2 @ vt2[0]
        order = np.argsort(t)
        pts_line = P[order][:: max(1, len(order) // 40)]
        streets.append({"ends_px": [P[order[0]].tolist(), P[order[-1]].tolist()],
                        "centre_px": pts_line.tolist(), "width_px": round(width, 1),
                        "double_line": len(group) > 1 or any(q.both_sides for q in group)})
    # an outline around 2+ other marks = enclosure (yard): not an object itself
    enclosures = np.zeros(strokes.shape, bool)
    keep = []
    for o in objects:
        m = np.zeros(strokes.shape, bool)
        rc = np.array(o["mask_rc"], int)
        m[rc[:, 0], rc[:, 1]] = True
        inner = sum(1 for q in objects if q is not o and q["diameter_px"] < o["diameter_px"]
                    and m[int(q["row"]), int(q["col"])])
        if inner >= 2:
            enclosures |= m
        else:
            keep.append(o)
    objects[:] = keep
    return {"objects": objects, "zones": zones, "streets": streets, "legend": legend,
            "roads": roads, "enclosures": enclosures}


def _empty_gt(shape):
    z = np.zeros(shape, bool)
    return {"objects": [], "zones": z, "streets": [], "legend": z.copy(), "roads": z.copy(),
            "enclosures": z.copy(), "lookalikes": [], "ramparts": z.copy(), "rampart_lines": []}


def _scale_of(entry, path):
    """Screen m/px and imagery resolution from the area's notes (only for pictures in the reference
    grid; a picture warped to the reference takes the reference's screen scale)."""
    sc = (entry.get("scale") or {})
    own = sc.get(os.path.basename(path), {})
    refd = sc.get(os.path.basename(entry["ref"]), {})
    d = {}
    if "mpp" in refd or "mpp" in own:
        d["mpp"] = refd.get("mpp", own.get("mpp"))
    if "imagery_m" in own:
        d["imagery_m"] = own["imagery_m"]
    return d


def _to_ref(ref, rgb):
    """Same size = already aligned by the user; otherwise align automatically."""
    if rgb.shape == ref.shape:
        return rgb, 1.0, 1.0
    w_rgb, _, cc, sc = align_pair(ref, rgb)
    return w_rgb, cc, sc


def prepare(entry, log=print):
    """Load the reference, extract ground truth (merged over all marked copies), and bring every
    other picture into the reference grid."""
    ref = load_rgb(entry["ref"])
    if entry.get("negative"):
        gt = _empty_gt(ref.shape[:2])
        gt["negative"] = True
        return ref, gt, [dict({"name": os.path.basename(entry["ref"]), "rgb": ref, "ok": True, "corr": 1.0},
                              **_scale_of(entry, entry["ref"]))]
    gt = None
    for r_path, m_path in entry.get("pairs") or [(entry["ref"], entry["marked"])]:
        base = ref if r_path == entry["ref"] else _to_ref(ref, load_rgb(r_path))[0]
        mk_w, cc, sc = _to_ref(ref, load_rgb(m_path))
        g = extract(base, mk_w)
        if gt is None:
            gt = g
            gt["mark_alignment"] = {"correlation": round(cc, 3), "scale": round(sc, 3)}
        else:                                    # merge: objects far from existing ones are added
            for o in g["objects"]:
                if all(math.hypot(o["row"] - q["row"], o["col"] - q["col"]) > 0.5 * max(o["diameter_px"], 8)
                       for q in gt["objects"]):
                    gt["objects"].append(o)
            gt["streets"] += g["streets"]
            gt["lookalikes"] += g.get("lookalikes", [])
            gt["rampart_lines"] += g.get("rampart_lines", [])
            for k in ("zones", "legend", "roads", "enclosures", "ramparts"):
                gt[k] = gt[k] | g[k]
    seasons = [dict({"name": os.path.basename(entry["ref"]), "rgb": ref, "ok": True, "corr": 1.0},
                    **_scale_of(entry, entry["ref"]))]
    for o in entry["others"]:
        rgb = load_rgb(o)
        w_rgb, cc2, sc2 = _to_ref(ref, rgb)
        ok = cc2 > 0.3 and 0.6 < sc2 < 1.6
        seasons.append(dict({"name": os.path.basename(o), "rgb": w_rgb, "ok": ok, "corr": round(cc2, 3)},
                            **_scale_of(entry, o)))
    return ref, gt, seasons


def draw_gt(ref_rgb, gt, path):
    vis = (ref_rgb * 255).astype(np.uint8).copy()
    if gt["zones"].any():
        vis[gt["zones"]] = (0.6 * vis[gt["zones"]] + 0.4 * np.array([0, 255, 255])).astype(np.uint8)
    for o in gt["objects"]:
        col = {"pit": (0, 120, 255), "foundation": (255, 220, 0)}.get(
            o.get("kind"), (0, 255, 0) if o["shape"] == "round" else (255, 160, 0))
        cv2.circle(vis, (int(o["col"]), int(o["row"])), int(max(5, o["diameter_px"] / 2 + 3)), col, 2)
    for la in gt.get("lookalikes", []):
        col = (40, 40, 40) if la["kind"] == "vegetation" else (255, 0, 0)
        rc = np.array(la["mask_rc"], int)
        m = np.zeros(vis.shape[:2], bool)
        m[rc[:, 0], rc[:, 1]] = True
        vis[m] = (0.6 * vis[m] + 0.4 * np.array(col)).astype(np.uint8)
        cv2.drawMarker(vis, (int(la["col"]), int(la["row"])), col, cv2.MARKER_TILTED_CROSS, 10, 2)
    for k, col in (("roads", (255, 0, 255)), ("enclosures", (255, 255, 0)), ("ramparts", (0, 200, 0))):
        m = gt.get(k)
        if m is not None and m.any():
            vis[m] = (0.65 * vis[m] + 0.35 * np.array(col)).astype(np.uint8)
    for s in gt["streets"]:
        pl = s.get("centre_px") or s["ends_px"]
        pts = np.array([[int(c), int(r)] for r, c in pl], np.int32)
        cv2.polylines(vis, [pts], False, (255, 0, 255), 2)
    cv2.imwrite(path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
