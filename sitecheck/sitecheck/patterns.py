"""Regularity: patches (or pits) standing in a row at an even spacing.

A ploughed-out village rarely shows clear single features, but its houses stood along a
street at fairly regular intervals (homestead width, typically 15-40 m). So a row of
weak, similar patches at an even spacing is a presumption that something stood there,
even when every patch alone could be natural.

Steps
  1. find_blobs: many weak, roughly round colour/brightness patches (Laplacian-of-Gaussian
     blob detection on a background-normalised anomaly map), 4-25 m across.
  2. find_rows: lines through 4+ patches with near-constant spacing (one gap allowed).
  3. chance check: the same number of patches is scattered at random over the same area
     many times; the share of random layouts producing an equally good row is reported
     as `chance`. Small = unlikely to be coincidence. (Natural soil spots can cluster more
     than random, so treat `chance` as optimistic.)
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import blob_log

from .imagery import _lab, _masked_mean


# --------------------------------------------------------------------------- patches

def line_mask(rgb, region, mpp, min_len_m=50.0):
    """Long straight thin lines - tramlines, tractor tracks, field roads.
    Patches sitting on them are farming artefacts, not houses."""
    import cv2
    from skimage.filters import sato
    from .imagery import luminance
    f = max(1, int(round(0.6 / mpp)))
    g = luminance(rgb)[::f, ::f].astype("float64")
    reg = region[::f, ::f]
    sig = [max(0.8, 0.6 / (mpp * f)), max(1.2, 1.2 / (mpp * f))]
    ridge = np.maximum(sato(g, sigmas=sig, black_ridges=False), sato(g, sigmas=sig, black_ridges=True))
    ridge[~reg] = 0
    thr = np.percentile(ridge[reg], 95) if reg.any() else 1
    bw = (ridge > thr).astype(np.uint8) * 255
    L = int(min_len_m / (mpp * f))
    lines = cv2.HoughLinesP(bw, 1, np.pi / 360, threshold=int(L * 0.35), minLineLength=L, maxLineGap=int(6 / (mpp * f)))
    m = np.zeros(reg.shape, np.uint8)
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(m, (int(x1), int(y1)), (int(x2), int(y2)), 1, thickness=max(2, int(3 / (mpp * f))))
    m = cv2.resize(m, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_NEAREST)
    return m.astype(bool)


def find_blobs(rgb, region, mpp, min_m=4.0, max_m=25.0, z_min=1.2, max_n=200, return_maps=False):
    """Weak round patches in the region (field / open ground). Returns list of dicts with
    row, col, diameter_m, strength (robust z), kind."""
    if region.sum() < 2000:
        return ([], {}) if return_maps else []
    lab = _lab(rgb)
    w = region.astype("float32")
    sig_small = max(1.0, 1.5 / mpp)        # furrows, rows, grain
    sig_bg = max(5.0, 40 / mpp)
    edge = ndi.binary_dilation(~region, iterations=max(1, int(8 / mpp)))
    m_ = max(1, int(12 / mpp))                       # frame border: normalisation is blind there
    edge[:m_], edge[-m_:], edge[:, :m_], edge[:, -m_:] = True, True, True, True
    chans = {}
    for ci, name in ((0, "L"), (1, "a"), (2, "b")):
        ch = ndi.gaussian_filter(lab[..., ci] * w, sig_small) / np.maximum(ndi.gaussian_filter(w, sig_small), 1e-3)
        d = ch - _masked_mean(ch, w, sig_bg)
        v = d[region & ~edge]
        mad = 1.4826 * np.median(np.abs(v - np.median(v))) + 1e-3
        chans[name] = np.where(region & ~edge, d / mad, 0).astype("float32")
    maps = {"darker": -chans["L"], "lighter": chans["L"], "greener": -chans["a"], "redder": chans["a"]}
    # downsample for speed: blobs are >= 4 m
    f = max(1, int(round(1.0 / mpp)))
    blobs = []
    for kind, m in maps.items():
        img = np.clip(m[::f, ::f], 0, None)
        if img.max() <= z_min:
            continue
        smin = min_m / (mpp * f) / (2 * math.sqrt(2))
        smax = max_m / (mpp * f) / (2 * math.sqrt(2))
        for y, x, s in blob_log(img, min_sigma=max(1.0, smin), max_sigma=max(1.5, smax), num_sigma=6,
                                threshold=z_min * 0.08, overlap=0.3):
            r, c = int(y * f), int(x * f)
            if not region[r, c] or edge[r, c]:
                continue
            z = float(m[r, c])
            if z < z_min:
                continue
            blobs.append({"row": float(r), "col": float(c), "diameter_m": round(2 * math.sqrt(2) * s * f * mpp, 1),
                          "strength": round(z, 2), "kind": kind})
    on_line = line_mask(rgb, region, mpp)
    blobs = [b for b in blobs if not on_line[int(b["row"]), int(b["col"])]]
    # one patch can show in two channels (darker + redder): keep the strongest
    blobs.sort(key=lambda b: -b["strength"])
    kept = []
    for b in blobs:
        if all(math.hypot(b["row"] - k["row"], b["col"] - k["col"]) * mpp > 0.6 * max(b["diameter_m"], k["diameter_m"])
               for k in kept):
            kept.append(b)
    if return_maps:
        return kept[:max_n], maps
    return kept[:max_n]


# --------------------------------------------------------------------------- rows

def _rows_from_points(P, S, D, min_n=4, min_gap=10.0, max_gap=75.0):
    """P: (n,2) metres, S: strengths, D: diameters. Returns candidate rows sorted by score.

    A row = at least `min_n` similar patches whose centres lie on one line, each next one
    10-60 m on (not overlapping the previous one), without a hole of more than ~2.6x the usual
    gap. Real village houses are NOT evenly spaced (yards differ), so even spacing is a bonus
    in the score, not a requirement; the chance test decides what is unusual."""
    n = len(P)
    if n < min_n:
        return []
    rows = []
    seen = set()
    for i in range(n):
        d = np.hypot(*(P - P[i]).T)
        for j in np.where((d >= min_gap) & (d <= max_gap))[0]:
            if j <= i:
                continue
            u = (P[j] - P[i]) / d[j]
            rel = P - P[i]
            t = rel @ u
            perp = rel[:, 0] * u[1] - rel[:, 1] * u[0]
            tol = max(4.0, 0.45 * float(np.mean(D[[i, j]])))
            on = np.where(np.abs(perp) <= tol)[0]
            if len(on) < min_n:
                continue
            on = on[np.argsort(t[on])]
            # keep patches that do not overlap the previous one (one per house)
            chain = [on[0]]
            for k in on[1:]:
                if t[k] - t[chain[-1]] >= max(0.6 * 0.5 * (D[k] + D[chain[-1]]), 0.35 * min_gap):
                    chain.append(k)
                elif S[k] > S[chain[-1]]:
                    chain[-1] = k
            chain = np.array(chain)
            if len(chain) < min_n:
                continue
            gaps = np.diff(t[chain])
            # split at holes: > max_gap, or > 2.6x the median gap
            med = float(np.median(gaps))
            cut = np.where((gaps > max_gap * 1.5) | (gaps > 2.6 * max(med, min_gap)))[0]
            pieces = np.split(chain, cut + 1)
            for hits in pieces:
                if len(hits) < min_n or i not in hits and j not in hits:
                    continue
                key = tuple(sorted(int(h) for h in hits))
                if key in seen:
                    continue
                seen.add(key)
                g = np.diff(t[hits])
                unit = np.median(g[g < 1.5 * np.median(g)]) if len(g) else d[j]
                steps = np.maximum(1, np.round(g / unit))
                g1 = g / steps
                cv = float(np.std(g1) / max(np.mean(g1), 1e-6))
                cv_raw = float(np.std(g) / max(np.mean(g), 1e-6))
                if cv_raw > 0.75:
                    continue
                size_cv = float(np.std(D[hits]) / max(np.mean(D[hits]), 1e-6))
                straight = float(np.std(perp[hits]))
                straight_f = 1.0 - min(0.5, straight / max(tol, 1e-6) * 0.5)
                regular_f = 1.0 - 0.6 * min(cv, 1.0)
                score = (len(hits) * regular_f * straight_f * (1 - min(size_cv, 0.8) * 0.5)
                         * float(np.mean(np.minimum(S[hits], 4)) / 2))
                rows.append({"members": [int(h) for h in hits], "spacing_m": float(np.mean(g)),
                             "spacing_cv": round(cv_raw, 2), "missing": int(steps.sum() - len(g)),
                             "size_cv": size_cv, "offset_sd_m": straight, "score": score,
                             "dir": u, "t0": float(t[hits[0]]), "t1": float(t[hits[-1]]), "origin": P[i]})
    rows.sort(key=lambda r: -r["score"])
    # drop rows that mostly repeat a better one
    out = []
    for r in rows:
        if all(len(set(r["members"]) & set(o["members"])) < 0.5 * len(r["members"]) for o in out):
            out.append(r)
    return out


FAMILIES = {"dark/reddish soil": ("darker", "redder"), "light soil": ("lighter",), "green crop": ("greener",),
            "hollow": ("hollow",)}


def _is_track(lum, a_rc, b_rc, mpp):
    """True if the row runs along a continuous thin line (tramline, track): brightness on
    the path differs from strips 3 m to the side with the same sign almost all the way."""
    (r1, c1), (r2, c2) = a_rc, b_rc
    n = int(max(abs(r2 - r1), abs(c2 - c1)))
    if n < 10:
        return False
    rr = np.linspace(r1, r2, n)
    cc = np.linspace(c1, c2, n)
    L = math.hypot(r2 - r1, c2 - c1)
    nr, nc = -(c2 - c1) / L, (r2 - r1) / L               # unit normal (pixels)
    off = 3.0 / mpp
    samp = lambda r, c: ndi.map_coordinates(lum, [np.clip(r, 0, lum.shape[0] - 1), np.clip(c, 0, lum.shape[1] - 1)], order=1)
    on = samp(rr, cc)
    side = 0.5 * (samp(rr + off * nr, cc + off * nc) + samp(rr - off * nr, cc - off * nc))
    d = on - side
    med = np.median(d)
    same = float(np.mean(np.sign(d) == np.sign(med)))
    return same > 0.78 and abs(med) > 0.5 * (1.4826 * np.median(np.abs(d - med)) + 1e-6)


def _is_edge(lum, a_rc, b_rc, mpp):
    """True if the row runs along a boundary: one side of it is lighter than the other almost
    all the way (field edge, gully side, boundary of a soil band, edge of a plough strip).
    Spots strung along such a boundary are a property of the boundary, not houses."""
    (r1, c1), (r2, c2) = a_rc, b_rc
    n = int(max(abs(r2 - r1), abs(c2 - c1)))
    if n < 10:
        return False
    rr = np.linspace(r1, r2, n)
    cc = np.linspace(c1, c2, n)
    L = math.hypot(r2 - r1, c2 - c1)
    nr, nc = -(c2 - c1) / L, (r2 - r1) / L
    g = ndi.gaussian_filter(lum, max(1.0, 1.5 / mpp))
    samp = lambda r, c: ndi.map_coordinates(g, [np.clip(r, 0, g.shape[0] - 1), np.clip(c, 0, g.shape[1] - 1)], order=1)
    for off_m in (4.0, 8.0):
        off = off_m / mpp
        d = samp(rr + off * nr, cc + off * nc) - samp(rr - off * nr, cc - off * nc)
        med = np.median(d)
        same = float(np.mean(np.sign(d) == np.sign(med)))
        if same > 0.85 and abs(med) > 0.8 * (1.4826 * np.median(np.abs(d - med)) + 1e-6):
            return True
    return False


def tillage_bearing(lum, region, mpp):
    """Dominant direction of fine stripes (furrows, tramlines) in degrees 0-180 (bearing),
    and how strong it is (0..1). From the structure tensor of the image."""
    g = ndi.gaussian_filter(lum, max(0.7, 0.5 / mpp))
    g = g - ndi.gaussian_filter(lum, max(2.0, 3.0 / mpp))   # keep only fine stripes, drop patches/edges
    gx, gy = ndi.sobel(g, 1), ndi.sobel(g, 0)
    inner = ndi.binary_erosion(region, iterations=max(1, int(15 / mpp)))
    w = inner & (np.hypot(gx, gy) > 0)
    if w.sum() < 1000:
        return 0.0, 0.0
    jxx, jyy, jxy = (gx * gx)[w].sum(), (gy * gy)[w].sum(), (gx * gy)[w].sum()
    theta = 0.5 * math.atan2(2 * jxy, jxx - jyy)          # gradient direction (image x/y)
    coher = math.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / max(jxx + jyy, 1e-9)
    # stripes run perpendicular to the gradient; convert to bearing (0 = up/north, clockwise)
    sx, sy = -math.sin(theta), math.cos(theta)              # stripe direction in image coords
    bearing = math.degrees(math.atan2(sx, -sy)) % 180
    return bearing, coher


def _is_band(maps, kinds, a_rc, b_rc, mpp, members_rc=None):
    """True if the anomaly is continuous along the row (a band: slope break, field margin,
    wet strip, the road itself) rather than separate patches with ordinary ground between them.
    With the members known: a band has no dip between neighbouring patches."""
    ks = [k for k in kinds if k in maps]
    if not ks:
        return False
    zmap = np.max([ndi.gaussian_filter(maps[k], 1.0 / mpp) for k in ks], axis=0)
    H, W = zmap.shape
    at = lambda r, c: float(zmap[int(np.clip(r, 0, H - 1)), int(np.clip(c, 0, W - 1))])
    if members_rc is not None and len(members_rc) >= 3:
        M = np.array(members_rc, float)
        dips = 0
        for a, b in zip(M[:-1], M[1:]):
            lo = min(at(*a), at(*b))
            # lowest point on the way from one patch to the next (middle 60 %)
            ts = np.linspace(0.2, 0.8, 9)
            mid = min(at(*(a + t_ * (b - a))) for t_ in ts)
            if mid < 0.55 * lo or mid < 0.5:
                dips += 1
        return dips < 0.5 * (len(M) - 1)
    (r1, c1), (r2, c2) = a_rc, b_rc
    n = int(max(abs(r2 - r1), abs(c2 - c1)))
    if n < 10:
        return False
    rr, cc = np.linspace(r1, r2, n), np.linspace(c1, c2, n)
    z = np.array([at(r, c) for r, c in zip(rr, cc)])
    return float(np.mean(z > 0.8)) > 0.65


def find_rows(blobs, region, mpp, n_random=80, seed=0, max_rows=6, lum=None, maps=None):
    """Rows of similar patches + how often random patches give an equally good row.
    Houses of one village leave similar marks, so rows are built within one family of
    patches (dark/reddish, light, green)."""
    out = []
    rng = np.random.default_rng(seed)
    er = ndi.binary_erosion(region, iterations=max(1, int(8 / mpp)))
    m_ = max(1, int(12 / mpp))
    er[:m_], er[-m_:], er[:, :m_], er[:, -m_:] = False, False, False, False
    rr, cc = np.nonzero(er)
    if len(rr) == 0:
        return []
    for fam, kinds in FAMILIES.items():
        fb = sorted([b for b in blobs if b["kind"] in kinds], key=lambda b: -b["strength"])[:45]
        if len(fb) < 4:
            continue
        P = np.array([[b["col"] * mpp, -b["row"] * mpp] for b in fb])     # metres, x east, y north
        S = np.array([b["strength"] for b in fb])
        D = np.array([b["diameter_m"] for b in fb])
        rows = _rows_from_points(P, S, D)[:max_rows]
        if not rows:
            continue
        # chance check, two ways; the larger chance counts:
        #  (a) the same patches scattered at random over the area
        #  (b) each patch moved up to ~25 m from where it is: keeps the patchiness of the soil
        #      (clusters, bands of lighter ground) but breaks lines that are only "almost" there.
        #      Patchy soil makes lines more often than (a) assumes; (b) is the stricter test.
        best_a, best_b = [], []
        inside = er
        Hh, Ww = inside.shape

        def trials(n):
            for _ in range(n):
                k = rng.integers(0, len(rr), len(P))
                Pr = np.column_stack([cc[k] * mpp, -rr[k] * mpp])
                rr_rows = _rows_from_points(Pr, rng.permutation(S), rng.permutation(D))
                best_a.append(rr_rows[0]["score"] if rr_rows else 0.0)
                ang = rng.uniform(0, 2 * math.pi, len(P))
                rad = 25.0 * np.sqrt(rng.uniform(0.15, 1.0, len(P)))
                Pj = P + np.column_stack([np.cos(ang), np.sin(ang)]) * rad[:, None]
                rj, cj = (-Pj[:, 1] / mpp).astype(int), (Pj[:, 0] / mpp).astype(int)
                bad = ~((rj >= 0) & (rj < Hh) & (cj >= 0) & (cj < Ww))
                bad[~bad] = ~inside[rj[~bad], cj[~bad]]
                Pj[bad] = P[bad] - (Pj[bad] - P[bad])                         # mirror back
                rj_rows = _rows_from_points(Pj, S, D)
                best_b.append(rj_rows[0]["score"] if rj_rows else 0.0)

        trials(n_random)
        # the best row is beaten (almost) never: more trials so a small chance can be told apart
        # from "just under the floor" (80 trials cannot show anything below ~1 %)
        top = rows[0]["score"]
        if max(sum(x >= top for x in best_a), sum(x >= top for x in best_b)) <= 2:
            trials(3 * n_random)
        n_random_used = len(best_a)
        best_a, best_b = np.array(best_a), np.array(best_b)
        for r in rows:
            chance = float(max(np.sum(best_a >= r["score"]), np.sum(best_b >= r["score"])) + 1) / (n_random_used + 1)
            u = r["dir"]
            a = r["origin"] + u * r["t0"]
            b = r["origin"] + u * r["t1"]
            to_rc = lambda p: [float(-p[1] / mpp), float(p[0] / mpp)]
            if lum is not None and _is_edge(lum, to_rc(a), to_rc(b), mpp):
                continue          # strung along a boundary (field edge, gully, soil band edge)
            if fam != "hollow" and lum is not None and _is_track(lum, to_rc(a), to_rc(b), mpp):
                continue          # patches strung along a tramline / track
            members = [fb[k] for k in r["members"]]
            if maps and _is_band(maps, kinds, to_rc(a), to_rc(b), mpp,
                                 [[m["row"], m["col"]] for m in members]):
                continue          # one continuous band, not separate patches
            ang = math.degrees(math.atan2(u[0], u[1])) % 180       # bearing of the row
            out.append({
                "type": "row", "kind": "regular_row", "patch_family": fam,
                "row": float(np.mean([m["row"] for m in members])), "col": float(np.mean([m["col"] for m in members])),
                "n_patches": len(members), "spacing_m": round(r["spacing_m"], 1),
                "spacing_variation": round(r["spacing_cv"], 2), "missing_in_row": r["missing"],
                "length_m": round(r["t1"] - r["t0"], 1), "bearing_deg": round(ang),
                "mean_patch_m": round(float(np.mean([m["diameter_m"] for m in members])), 1),
                "chance": round(chance, 3),
                "ends_px": [to_rc(a), to_rc(b)],
                "members_px": [[m["row"], m["col"]] for m in members],
                "confidence": round(float(np.clip(1 - chance * 4, 0.05, 0.95)), 2),
                "source_signal": "regular spacing of patches",
                "note": (f"{len(members)} similar {fam + ('s' if fam == 'hollow' else ' patches')} in a line, every {r['spacing_m']:.0f} m "
                         f"(±{r['spacing_cv'] * 100:.0f}%)"
                         + (f", {r['missing']} gap(s) where one would be expected" if r["missing"] else "")
                         + f"; in this picture random patches give a row this good in {chance * 100:.0f}% of trials "
                           "(see 'chance' for the value corrected for all searches). Like homesteads along a street."),
            })
    if lum is not None and out:
        tb, coher = tillage_bearing(lum, region, mpp)
        if coher > 0.25:
            for r in out:
                dd = abs(r["bearing_deg"] - tb) % 180
                if min(dd, 180 - dd) < 6:
                    r["along_cultivation"] = True
                    r["confidence"] = round(r["confidence"] * 0.5, 2)
                    r["note"] += (" CAUTION: runs along the field's cultivation lines - could be patches strung "
                                  "along a tramline or drill row.")
    out.sort(key=lambda r: (r.get("along_cultivation", False), r["chance"]))
    # parallel rows (two sides of a street / two streets) strengthen each other
    for a_ in out:
        par = [b_ for b_ in out if b_ is not a_ and min(abs(a_["bearing_deg"] - b_["bearing_deg"]),
                                                         180 - abs(a_["bearing_deg"] - b_["bearing_deg"])) < 12
               and b_["chance"] < 0.3 and 10 < math.hypot(a_["row"] - b_["row"], a_["col"] - b_["col"]) * mpp < 200]
        if par and a_["chance"] < 0.3:
            a_["parallel_rows"] = len(par) + 1
            a_["confidence"] = round(min(0.95, a_["confidence"] + 0.1), 2)
            a_["note"] += f" Parallel to {len(par)} other row(s)."
    return out


# --------------------------------------------------------------------------- the other side of the street

def _any_zmap(maps, region, mpp):
    z = np.max([ndi.gaussian_filter(m, max(1.0, 1.5 / mpp)) for m in maps.values()], axis=0)
    z[~region] = 0
    return z


def _best_parallel(zmax, region, c, u, s, t0, t1, mpp, offsets, z_min, z=None):
    """Best line parallel to (c, u) at the given perpendicular offsets, any phase along it,
    sampled every s over the row's extent (+1 step each end). zmax = window-max of z."""
    nrm = np.array([-u[1], u[0]])
    ks = np.arange(-1, int(round((t1 - t0) / s)) + 2)
    phases = np.linspace(0, s, 8, endpoint=False)
    best, best_key = (0, None), (0, -1e9)
    H, W = zmax.shape
    for d in offsets:
        for ph in phases:
            tt = t0 + ph + ks * s
            P = c[None, :] + np.outer(tt, u) + d * nrm[None, :]
            r = np.round(P[:, 0]).astype(int)
            q = np.round(P[:, 1]).astype(int)
            ok = (r >= 0) & (r < H) & (q >= 0) & (q < W)
            if ok.sum() < 3:
                continue
            ok[ok] = region[r[ok], q[ok]]
            if ok.sum() < 3:
                continue
            hit = zmax[r[ok], q[ok]] >= z_min
            hits = int(hit.sum())
            # ties: the line through the patch centres, not along their edge
            key = (hits, float(np.minimum(z[r[ok], q[ok]], 8).sum()) if z is not None else 0.0)
            if key > best_key:
                best_key = key
                best = (hits, (d, ph, P[ok], hit))
    return best


def parallel_row(row, maps, region, mpp, n_random=150, seed=3, z_min=1.0):
    """Houses usually stood on both sides of a street. Given a found row, look for a second
    row of patches parallel to it at the same spacing, 20-150 m away on either side (any
    shift along it). Chance = how often the same search around a random line of the same
    length, spacing and direction finds as many patches."""
    if not maps:
        return None
    M = np.array(row["members_px"], float)
    if len(M) < 3:
        return None
    c = M.mean(0)
    _, _, vt = np.linalg.svd(M - c)
    u = vt[0] / np.linalg.norm(vt[0])
    t = (M - c) @ u
    s = row["spacing_m"] / mpp
    t0, t1 = float(t.min()), float(t.max())
    # same kind of patch as the row (houses of one village leave similar marks)
    fam = row.get("patch_family")
    sub = {k: maps[k] for k in FAMILIES.get(fam, ()) if k in maps} or maps
    z = _any_zmap(sub, region, mpp)
    win = max(3, int(0.22 * s))
    zmax = ndi.maximum_filter(z, size=win)
    # a hit must stand out among the window maxima of this field, not just be above average
    z_min = max(z_min, float(np.percentile(zmax[region], 85))) if region.any() else z_min
    step = max(2.0, 0.08 * s)
    d_min = max(20.0, 1.5 * row.get("mean_patch_m", 8.0))     # clear of the row's own patches
    offs = np.arange(d_min / mpp, 150 / mpp, step)
    offsets = np.concatenate([offs, -offs])
    n_obs, info = _best_parallel(zmax, region, c, u, s, t0, t1, mpp, offsets, z_min, z)
    if info is None or n_obs < 3:
        return None
    rng = np.random.default_rng(seed)
    er = ndi.binary_erosion(region, iterations=max(1, int(10 / mpp)))
    rr, cc = np.nonzero(er)
    if len(rr) == 0:
        return None
    rand = []
    for _ in range(n_random):
        k = rng.integers(len(rr))
        a = rng.uniform(0, math.pi)
        ur = np.array([math.sin(a), math.cos(a)])
        cr = np.array([rr[k], cc[k]], float)
        rand.append(_best_parallel(zmax, region, cr, ur, s, t0, t1, mpp, offsets, z_min)[0])
    rand = np.array(rand)
    chance = float((np.sum(rand >= n_obs) + 1) / (n_random + 1))
    d, ph, P, hit = info
    members = [list(map(float, p)) for p, h in zip(P, hit) if h]
    gaps = [list(map(float, p)) for p, h in zip(P, hit) if not h]
    nrm = np.array([-u[1], u[0]])
    street = [list(map(float, c + t0 * u + 0.5 * d * nrm)), list(map(float, c + t1 * u + 0.5 * d * nrm))]
    return {"type": "row", "kind": "parallel_row", "patch_family": fam if sub is not maps else "any",
            "row": float(np.mean([m[0] for m in members])), "col": float(np.mean([m[1] for m in members])),
            "members_px": members, "expected_but_missing_px": gaps, "n_patches": len(members),
            "spacing_m": row["spacing_m"], "bearing_deg": row["bearing_deg"],
            "offset_m": round(abs(d) * mpp, 1), "street_px": street,
            "ends_px": [list(map(float, P[0])), list(map(float, P[-1]))],
            "length_m": round(float(np.hypot(*(P[-1] - P[0]))) * mpp, 1),
            "chance": round(chance, 3), "random_mean_hits": round(float(rand.mean()), 2),
            "confidence": round(float(np.clip(1 - chance * 4, 0.05, 0.95)), 2),
            "source_signal": "second row parallel to a found row",
            "note": (f"{len(members)} patches on a line parallel to the row, {abs(d) * mpp:.0f} m away, at the same "
                     f"~{row['spacing_m']:.0f} m spacing - like houses on the other side of a street"
                     + (f"; {len(gaps)} expected place(s) without a patch" if gaps else "")
                     + f". The same search around random lines finds this many in {chance * 100:.0f}% of "
                       f"{n_random} trials (on average {rand.mean():.1f}).")}


# --------------------------------------------------------------------------- guided

def _family_at(maps, pts, mpp):
    """Which kind of patch the user clicked: the family whose anomaly is strongest at the seeds."""
    best, best_z = None, -1e9
    for fam, kinds in FAMILIES.items():
        ks = [k for k in kinds if k in maps]
        if not ks:
            continue
        zs = []
        for r, c in pts:
            r0, r1 = max(0, int(r - 3 / mpp)), int(r + 3 / mpp) + 1
            c0, c1 = max(0, int(c - 3 / mpp)), int(c + 3 / mpp) + 1
            zs.append(max(float(maps[k][r0:r1, c0:c1].max()) for k in ks))
        z = float(np.mean(zs))
        if z > best_z:
            best, best_z = fam, z
    return best, best_z


def _walk(zmap, start, u, s, n_max, tol_along, tol_perp, z_min, mpp, shape):
    """From `start`, step along unit vector u (pixel units, (dr, dc)) by s pixels; at each
    expected place look for the strongest local response within the tolerance box.
    Returns list of (expected_rc, found_rc or None, z)."""
    out = []
    pos = np.array(start, float)
    misses = 0
    for _ in range(n_max):
        exp = pos + u * s
        if not (0 <= exp[0] < shape[0] and 0 <= exp[1] < shape[1]):
            break
        rr = np.arange(-tol_along, tol_along + 1, max(1, tol_along / 6))
        pp = np.arange(-tol_perp, tol_perp + 1, max(1, tol_perp / 4))
        A, P = np.meshgrid(rr, pp)
        nvec = np.array([-u[1], u[0]])
        cand = exp[None, :] + A.ravel()[:, None] * u[None, :] + P.ravel()[:, None] * nvec[None, :]
        ok = (cand[:, 0] >= 0) & (cand[:, 0] < shape[0]) & (cand[:, 1] >= 0) & (cand[:, 1] < shape[1])
        cand = cand[ok]
        if len(cand) == 0:
            break
        z = zmap[cand[:, 0].astype(int), cand[:, 1].astype(int)]
        k = int(np.argmax(z))
        if z[k] >= z_min:
            out.append((exp.tolist(), cand[k].tolist(), float(z[k])))
            pos = cand[k]
            misses = 0
        else:
            out.append((exp.tolist(), None, float(z[k])))
            pos = exp
            misses += 1
            if misses >= 2:
                out = out[:-2] if len(out) >= 2 and out[-2][1] is None else out[:-1]
                break
    return out


def guided_row(rgb, region, mpp, seeds_rc, n_random=300, seed=0, z_min=1.0, extra_maps=None, family=None):
    """The user clicked 2-3 patches they think are houses in a row. Continue the row both
    ways at the same spacing, report which further patches are there, where the expected
    ones are missing (= places to check on the ground), and how often a random line with
    the same spacing would find as many patches."""
    if len(seeds_rc) < 2:
        raise ValueError("click at least 2 patches")
    _, maps = find_blobs(rgb, region, mpp, return_maps=True)
    if not maps:
        raise ValueError("no usable open ground / field in this picture")
    maps = dict(maps)
    if extra_maps:
        maps.update(extra_maps)
    S = np.array(seeds_rc, float)
    fam, z_seed = _family_at(maps, S, mpp) if not family else (family, None)
    kinds = [k for k in FAMILIES[fam] if k in maps]
    zmap = np.max([ndi.gaussian_filter(maps[k], max(1.0, 1.5 / mpp)) for k in kinds], axis=0)
    zmap[~region] = 0
    # line through the seeds (principal direction) and spacing (median gap of sorted seeds)
    c = S.mean(0)
    _, _, vt = np.linalg.svd(S - c)
    u = vt[0] / np.linalg.norm(vt[0])
    t = (S - c) @ u
    order = np.argsort(t)
    S, t = S[order], t[order]
    s = float(np.median(np.diff(t)))
    if s * mpp < 5:
        raise ValueError("clicked patches are closer than 5 m - click separate houses")
    tol_along, tol_perp = 0.22 * s, max(4.0 / mpp, 0.15 * s)
    n_max = int(max(region.shape) / s) + 2
    fwd = _walk(zmap, S[-1], u, s, n_max, tol_along, tol_perp, z_min, mpp, region.shape)
    bwd = _walk(zmap, S[0], -u, s, n_max, tol_along, tol_perp, z_min, mpp, region.shape)
    found = [f for f in fwd + bwd if f[1] is not None]
    gaps = [f for f in fwd + bwd if f[1] is None]
    n_found = len(found)
    # chance: random lines with the same spacing and the same number of steps, from random
    # starting points in the same ground, in random directions
    rng = np.random.default_rng(seed)
    rr, cc = np.nonzero(region)
    n_steps = len(fwd) + len(bwd)
    random_hits = []
    for _ in range(n_random):
        k = rng.integers(len(rr))
        a = rng.uniform(0, math.pi)
        ur = np.array([math.sin(a), math.cos(a)])
        start = np.array([rr[k], cc[k]], float)
        w = _walk(zmap, start, ur, s, max(n_steps, 2), tol_along, tol_perp, z_min, mpp, region.shape)
        random_hits.append(sum(1 for x in w if x[1] is not None))
    random_hits = np.array(random_hits)
    chance = float((np.sum(random_hits >= n_found) + 1) / (n_random + 1)) if n_found else 1.0
    members = [list(map(float, p)) for p in S] + [f[1] for f in found]
    allpts = np.array(members)
    tt = (allpts - c) @ u
    a_rc, b_rc = c + u * tt.min(), c + u * tt.max()
    pres = "presumption" if chance < 0.10 and n_found >= 2 else "weak presumption" if chance <= 0.30 and n_found >= 1 \
        else "none"
    return {
        "type": "row", "kind": "guided_row", "patch_family": fam, "detected_by": "user + algorithm",
        "row": float(allpts[:, 0].mean()), "col": float(allpts[:, 1].mean()),
        "seeds_px": [list(map(float, p)) for p in S],
        "members_px": members, "n_patches": len(members), "n_found_beyond_seeds": n_found,
        "expected_but_missing_px": [g[0] for g in gaps],
        "spacing_m": round(s * mpp, 1), "bearing_deg": round(math.degrees(math.atan2(u[1], -u[0])) % 180),
        "length_m": round(float(tt.max() - tt.min()) * mpp, 1),
        "mean_patch_m": 8.0, "ends_px": [list(map(float, a_rc)), list(map(float, b_rc))],
        "chance": round(chance, 3), "random_mean_hits": round(float(random_hits.mean()), 2),
        "presumption": pres, "confidence": round(float(np.clip(1 - chance * 3, 0.05, 0.95)), 2),
        "source_signal": "user-seeded row of similar patches",
        "note": (f"You marked {len(S)} {fam} patches ~{s * mpp:.0f} m apart. Continuing the line found "
                 f"{n_found} more similar patch(es) at the expected places"
                 + (f" and {len(gaps)} expected place(s) with nothing - check those on the ground" if gaps else "")
                 + f". A random line with the same spacing finds this many in {chance * 100:.0f}% of "
                   f"{n_random} trials (on average {random_hits.mean():.1f}). Result: {pres}."),
    }


def family_zmap(rgb, region, mpp, fam, extra_maps=None):
    _, maps = find_blobs(rgb, region, mpp, return_maps=True)
    maps = dict(maps or {})
    if extra_maps:
        maps.update(extra_maps)
    ks = [k for k in FAMILIES[fam] if k in maps]
    if not ks:
        return None
    z = np.max([ndi.gaussian_filter(maps[k], max(1.0, 1.5 / mpp)) for k in ks], axis=0)
    z[~region] = 0
    return z


def check_row_in_other_picture(rgb2, region2, mpp, row, n_random=300, seed=1, z_min=1.0, extra_maps=None):
    """Second, independent look: are the places predicted by a guided row (found patches
    and expected-but-missing ones) also patches in another picture of the same place?
    Chance = how often the same pattern of places, put at a random spot and angle, hits
    as many patches in that picture."""
    z = family_zmap(rgb2, region2, mpp, row["patch_family"], extra_maps)
    if z is None:
        return None
    pts = np.array(row["members_px"] + row["expected_but_missing_px"], float)
    (r1, c1), (r2, c2) = row["ends_px"]
    u = np.array([r2 - r1, c2 - c1], float)
    u /= max(np.linalg.norm(u), 1e-6)
    s = row["spacing_m"] / mpp
    tol_along, tol_perp = 0.22 * s, max(4.0 / mpp, 0.15 * s)

    def hits_at(P, uu):
        nvec = np.array([-uu[1], uu[0]])
        n = 0
        for p in P:
            best = -1e9
            for a in np.linspace(-tol_along, tol_along, 7):
                for b in np.linspace(-tol_perp, tol_perp, 5):
                    q = p + a * uu + b * nvec
                    if 0 <= q[0] < z.shape[0] and 0 <= q[1] < z.shape[1]:
                        best = max(best, z[int(q[0]), int(q[1])])
            n += best >= z_min
        return n

    observed = hits_at(pts, u)
    rng = np.random.default_rng(seed)
    rr, cc = np.nonzero(region2)
    c0 = pts.mean(0)
    rand = []
    for _ in range(n_random):
        ang = rng.uniform(0, 2 * math.pi)
        R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
        k = rng.integers(len(rr))
        P = (pts - c0) @ R.T + np.array([rr[k], cc[k]])
        rand.append(hits_at(P, R @ u))
    rand = np.array(rand)
    chance = float((np.sum(rand >= observed) + 1) / (n_random + 1))
    return {"hits": int(observed), "places": len(pts), "chance": round(chance, 3),
            "random_mean_hits": round(float(rand.mean()), 2)}
