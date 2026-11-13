"""Closed outlines: ramparts, banks and ditches around a fortified place or an old yard.

A rampart seen from above is rarely one clear line. It shows as a band of shade, a change of
vegetation, a ring of lighter soil - but it goes ROUND something. So instead of looking for lines,
this looks for a closed outline: from many trial centres it looks outwards in 72 directions, and
finds the smooth closed path (radius may change slowly from one direction to the next) that runs
along as much "edge" as possible. Edges = changes of brightness, of colour (vegetation) and of
texture, each compared with what is normal in this picture.

What it reports is an OUTLINE, not a verdict: ponds, groves, single plots and clearings are closed
outlines too. Wooded or water-filled interiors are flagged. Seen in several years = stronger.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import ndimage as ndi

WORK_MPP = 1.0          # work at ~1 m per pixel: ramparts are 5-20 m wide
N_ANG = 72
R_MIN_M, R_MAX_M, R_RATIO = 12.0, 200.0, 1.06


def _evidence(rgb, valid, mpp):
    """Edge strength (0..1) and unit normal of the edge, on the working grid."""
    f = min(1.0, mpp / WORK_MPP)
    small = cv2.resize(np.clip(rgb, 0, 1).astype(np.float32), None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    vs = cv2.resize(valid.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    wm = mpp / f                                            # metres per working pixel
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2Lab)
    L = lab[..., 0]
    tex = np.sqrt(np.maximum(ndi.uniform_filter(L ** 2, 5) - ndi.uniform_filter(L, 5) ** 2, 0))
    chans = [L, 2.0 * lab[..., 1], 2.0 * lab[..., 2], 3.0 * tex]
    sig = 3.0 / wm
    best = np.zeros(L.shape, np.float32)
    ny = np.zeros(L.shape, np.float32)
    nx = np.zeros(L.shape, np.float32)
    yy, xx = np.mgrid[:L.shape[0], :L.shape[1]].astype(np.float32)
    for ch in chans:
        gy = ndi.gaussian_filter(ch, sig, order=(1, 0))
        gx = ndi.gaussian_filter(ch, sig, order=(0, 1))
        g = np.hypot(gy, gx)
        hi = float(np.percentile(g[vs], 90)) if vs.any() else 1.0
        e = np.clip((g / max(hi, 1e-6) - 0.55) / 0.9, 0, 1)      # only the clearest ~15 % of edges count
        uy, ux = gy / np.maximum(g, 1e-6), gx / np.maximum(g, 1e-6)
        # keep the crest of each edge only (thin lines), so that textured ground is not "edge everywhere"
        a = cv2.remap(g, xx + ux, yy + uy, cv2.INTER_LINEAR)
        b = cv2.remap(g, xx - ux, yy - uy, cv2.INTER_LINEAR)
        e = np.where((g >= a) & (g >= b), e, 0)
        up = e > best
        best = np.where(up, e, best)
        ny = np.where(up, uy, ny)
        nx = np.where(up, ux, nx)
    # the path may pass one pixel beside the crest
    grown = ndi.maximum_filter(best, 3)
    idx = ndi.maximum_filter(best, 3) > best
    if idx.any():
        iy, ix = ndi.distance_transform_edt(best <= 0, return_distances=False, return_indices=True)
        ny = np.where(idx, ny[iy, ix], ny)
        nx = np.where(idx, nx[iy, ix], nx)
    best = grown
    best[~vs] = 0
    return best, ny, nx, vs, wm, f


def _contours(E, ny, nx, vs, wm, step_m=8.0, r_max_m=R_MAX_M):
    """Best closed star-shaped contour around every trial centre (all centres at once)."""
    H, W = E.shape
    r_max = min(r_max_m, 0.6 * max(H, W) * wm)
    radii = [R_MIN_M]
    while radii[-1] * R_RATIO < r_max:
        radii.append(radii[-1] * R_RATIO)
    radii = np.array(radii) / wm                           # working px
    J = len(radii)
    step = max(2, int(round(step_m / wm)))
    cy, cx = np.mgrid[step // 2:H:step, step // 2:W:step]
    cy, cx = cy.ravel().astype(np.float32), cx.ravel().astype(np.float32)
    ok = vs[cy.astype(int), cx.astype(int)]
    cy, cx = cy[ok], cx[ok]
    N = len(cy)
    if N == 0 or J < 4:
        return None
    ang = np.linspace(0, 2 * math.pi, N_ANG, endpoint=False)
    S = np.zeros((N_ANG, N, J), np.float32)
    for k, a in enumerate(ang):
        dy, dx = math.sin(a), math.cos(a)
        rr = cy[:, None] + radii[None, :] * dy
        cc = cx[:, None] + radii[None, :] * dx
        inside = (rr >= 0) & (rr <= H - 1) & (cc >= 0) & (cc <= W - 1)
        ri = np.clip(np.round(rr).astype(int), 0, H - 1)
        ci = np.clip(np.round(cc).astype(int), 0, W - 1)
        e = E[ri, ci]
        tang = np.abs(ny[ri, ci] * dy + nx[ri, ci] * dx)    # edge runs across the ray
        S[k] = np.where(inside, e * tang ** 2, 0.0)        # only edges that run round the centre
    # circular Viterbi (two laps; the second lap is kept)
    lam = 0.25
    shifts = (-1, 0, 1)
    score = np.zeros((N, J), np.float32)
    back = np.zeros((2 * N_ANG, N, J), np.int8)
    for t in range(2 * N_ANG):
        cands = []
        for sh in shifts:
            sc = np.full((N, J), -1e9, np.float32)
            if sh < 0:
                sc[:, :sh] = score[:, -sh:]
            elif sh > 0:
                sc[:, sh:] = score[:, :-sh]
            else:
                sc = score.copy()
            cands.append(sc - lam * abs(sh))
        st = np.stack(cands, 0)
        arg = st.argmax(0)
        score = np.take_along_axis(st, arg[None], 0)[0] + S[t % N_ANG]
        back[t] = arg.astype(np.int8)
    j = score.argmax(1)
    path = np.zeros((2 * N_ANG, N), np.int32)
    idx = np.arange(N)
    for t in range(2 * N_ANG - 1, -1, -1):
        path[t] = j
        j = np.clip(j - (np.array(shifts)[back[t, idx, j]]), 0, J - 1)
    path = path[N_ANG:]                                     # (N_ANG, N)
    val = S[np.arange(N_ANG)[:, None], idx[None, :], path]  # evidence along the contour
    return {"cy": cy, "cx": cx, "radii": radii, "path": path, "val": val, "ang": ang, "S": S}


def find_enclosures(rgb, valid, mpp, max_n=3, min_cover=0.55, classes=None):
    """Closed outlines in a picture, best first. Each: contour_px (in picture pixels), centre,
    area_ha, coverage (share of the way round with a visible edge), strength, score, flags."""
    E, ny, nx, vs, wm, f = _evidence(rgb, valid, mpp)
    res = _contours(E, ny, nx, vs, wm)
    if res is None:
        return []
    H, W = E.shape
    val, path, radii, ang = res["val"], res["path"], res["radii"], res["ang"]
    cover = (val > 0.22).mean(0)
    # longest gap without evidence (in directions): a rampart open on one side is fine, half missing is not
    strength = val.mean(0)
    rad_m = radii[path] * wm                                # (N_ANG, N)
    perim = (rad_m * (2 * math.pi / N_ANG)).sum(0)          # rough length of the outline, m
    edge_m = ((val > 0.22) * rad_m * (2 * math.pi / N_ANG)).sum(0)   # metres of visible edge followed
    total = cover ** 2 * np.sqrt(edge_m / 100.0)
    # how far above "what any spot of this picture gives" is each outline?
    base = float(np.percentile(total, 30))
    spread = float(1.4826 * np.median(np.abs(total - np.median(total)))) + 1e-6
    standout = (total - base) / spread
    order = np.argsort(-total)
    out, masks = [], []
    for i in order[:400]:
        if cover[i] < min_cover or len(out) >= max_n:
            break
        r = radii[path[:, i]]
        py = res["cy"][i] + r * np.sin(ang)
        px = res["cx"][i] + r * np.cos(ang)
        if (py < 1).mean() + (py > H - 2).mean() + (px < 1).mean() + (px > W - 2).mean() > 0.2:
            continue                                        # mostly the picture frame
        poly = np.column_stack([px, py]).astype(np.int32)
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [poly], 1)
        m = m.astype(bool)
        area = float(m.sum()) * wm * wm
        if area < 400 or area > 9e4:
            continue
        if any((m & q).sum() > 0.4 * min(m.sum(), q.sum()) for q in masks):
            continue
        # radius must not be glued to the smallest / largest radius that is searched
        if (path[:, i] == 0).mean() > 0.3 or (path[:, i] == len(radii) - 1).mean() > 0.3:
            continue
        # a second outline at a steady distance inside or outside (bank + ditch, double rampart)
        dbl = 0.0
        Sfull = res["S"][:, i, :]
        for dj in list(range(-10, -2)) + list(range(3, 11)):
            jj = np.clip(path[:, i] + dj, 0, len(radii) - 1)
            dbl = max(dbl, float((Sfull[np.arange(N_ANG), jj] > 0.22).mean()))
        # smoothness: a rampart outline is not ragged
        rough = float(np.mean(np.abs(np.diff(np.r_[path[:, i], path[0, i]]))))
        info = {"type": "enclosure", "kind": "closed_outline",
                "row": float(res["cy"][i] / f), "col": float(res["cx"][i] / f),
                "contour_px": [[float(y / f), float(x / f)] for y, x in zip(py, px)],
                "area_ha": round(area / 1e4, 2),
                "size_m": [round(float((px.max() - px.min()) * wm)), round(float((py.max() - py.min()) * wm))],
                "coverage": round(float(cover[i]), 2), "strength": round(float(strength[i]), 2),
                "second_outline": round(dbl, 2), "raggedness": round(rough, 2),
                "edge_m": round(float(edge_m[i])), "outline_m": round(float(perim[i])),
                "standout": round(float(standout[i]), 2), "score": round(float(total[i]), 3)}
        if classes is not None:
            big = cv2.resize(m.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            if big.any():
                info["interior_wooded"] = round(float(classes["forest"][big].mean()), 2)
                ring = ndi.binary_dilation(big, iterations=max(2, int(15 / mpp))) & ~big
                info["outside_wooded"] = round(float(classes["forest"][ring].mean()), 2) if ring.any() else 0.0
        out.append(info)
        masks.append(m)
    return out


def outlines_over_pictures(per_picture, shape, mpp, min_overlap=0.5):
    """per_picture: list (one per picture, already in the reference picture's pixels) of outline
    lists. Outlines that enclose nearly the same ground in different pictures are one outline
    "seen in N pictures". Returns outlines sorted by (pictures, score)."""
    H, W = shape
    groups = []
    for k, lst in enumerate(per_picture):
        for o in lst:
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [np.array([[c, r] for r, c in o["contour_px"]], np.int32)], 1)
            m = m.astype(bool)
            if not m.any():
                continue
            for g in groups:
                inter = (m & g["mask"]).sum()
                if inter >= min_overlap * max(m.sum(), g["mask"].sum()):
                    if k + 1 not in g["seen"]:
                        g["seen"].append(k + 1)
                    if o["score"] > g["best"]["score"]:
                        g["best"] = o
                    break
            else:
                groups.append({"mask": m, "seen": [k + 1], "best": o})
    out = []
    for g in groups:
        o = dict(g["best"])
        o["seen_in_images"] = sorted(g["seen"])
        out.append(o)
    out.sort(key=lambda o: (-len(o["seen_in_images"]), -o["score"]))
    return out
