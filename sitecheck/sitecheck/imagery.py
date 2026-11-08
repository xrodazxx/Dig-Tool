"""Screenshot mode: when there is no lidar (most of Russia), work from ordinary satellite
screenshots (Google Earth, Yandex, Bing, ...). Everything is classical image maths.

What it does, per screenshot:
  1. Masks: program UI (user rectangles), forest/water (large dark areas), snow.
  2. Shading -> relief ("shape from shading", linear photoclinometry, Pentland 1988):
     on gently sloping ground lit from the side, brightness changes are proportional to
     the slope towards the sun. Integrating that slope in the Fourier domain gives a
     relative relief model. Its heights are rough (albedo changes leak in), but pits and
     banks show up as they would on a DEM, and the normal DEM detectors run on it.
  3. Sun direction: estimated from the shadows of trees/bushes (bright crown on the sun
     side, shadow on the far side), or given by the user.
  4. Marks: local colour anomalies - lingering snow in hollows (spring), lush/dark growth
     in pits (summer), isolated bushes in open fields (in abandoned villages they often
     grow in cellar pits), soil marks.
  5. Several screenshots of the same place (different seasons / years) are aligned
     automatically, and candidates seen in more than one image get more weight.

Scale: give --mpp (metres per pixel) or --known X1,Y1,X2,Y2,METRES (two pixels and the
distance between them, e.g. measured with the Google Earth ruler). Without it, sizes are
in pixels and the report says so.

Honesty: this is much weaker evidence than lidar. Every candidate says which signal
produced it, so the user (and the AI) can judge.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import measure

from . import features as feat
from . import terrain


# --------------------------------------------------------------------------- loading

def load(path: str, masks=()) -> tuple[np.ndarray, np.ndarray]:
    """RGB float32 0..1 and a validity mask (False = UI or user-masked)."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot read image {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255
    valid = np.ones(rgb.shape[:2], bool)
    for x0, y0, x1, y1 in masks:
        valid[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)] = False
    return rgb, valid


def luminance(rgb):
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype("float32")


# --------------------------------------------------------------------------- alignment

def _edges(rgb, valid):
    g8 = (np.clip(luminance(rgb), 0, 1) * 255).astype(np.uint8)
    g8 = cv2.createCLAHE(2.0, (8, 8)).apply(g8)          # hazy / low-contrast screenshots
    g = cv2.GaussianBlur(g8.astype(np.float32) / 255, (0, 0), 3)
    m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    m[~valid] = 0
    m = cv2.GaussianBlur(m, (0, 0), 4)
    return m / (m.max() + 1e-6)


def _coarse_search(ea, eb, scales=None):
    """Best scale + translation mapping image b onto a (screenshots of the same place at
    different zoom / framing, north-up). Normalised cross-correlation of edge maps,
    done at low resolution over a range of scales."""
    if scales is None:
        scales = np.geomspace(0.4, 2.5, 33)
    f = 360.0 / max(ea.shape)
    A = cv2.resize(ea, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    best = (-1.0, 1.0, 0.0, 0.0)
    for sc in scales:
        B = cv2.resize(eb, None, fx=f * sc, fy=f * sc, interpolation=cv2.INTER_AREA)
        if min(B.shape) < 24:
            continue
        # the overlap must be substantial: pad the larger image so either can slide
        ph, pw = B.shape[0] // 2, B.shape[1] // 2
        Ap = cv2.copyMakeBorder(A, ph, ph, pw, pw, cv2.BORDER_CONSTANT, value=0)
        if Ap.shape[0] < B.shape[0] or Ap.shape[1] < B.shape[1]:
            continue
        res = cv2.matchTemplate(Ap, B, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        # penalise tiny overlaps
        ox = max(0, min(A.shape[1], loc[0] - pw + B.shape[1]) - max(0, loc[0] - pw))
        oy = max(0, min(A.shape[0], loc[1] - ph + B.shape[0]) - max(0, loc[1] - ph))
        overlap = ox * oy / (A.shape[0] * A.shape[1])
        # mild preference for similar zoom: in feature-poor fields one straight road matches at any scale
        score = mx * min(1.0, overlap / 0.35) * math.exp(-math.log(sc) ** 2 / (2 * 0.4 ** 2))
        if score > best[0]:
            best = (score, sc, (loc[0] - pw) / f, (loc[1] - ph) / f)
    return best


def align(ref_rgb, ref_valid, rgb, valid):
    """Affine warp mapping `rgb` onto the reference. Seasons differ too much for keypoints,
    so we align large edge structure (forest edges, field borders, roads): first a coarse
    search over zoom and offset, then ECC refinement. Returns (warp 2x3, correlation 0..1).
    The warp maps reference pixel -> image pixel (use with WARP_INVERSE_MAP)."""
    ea, eb = _edges(ref_rgb, ref_valid), _edges(rgb, valid)
    score, sc, tx, ty = _coarse_search(ea, eb)
    # b scaled by sc and placed at (tx, ty) in a:  x_a = sc * x_b + t  ->  x_b = (x_a - t) / sc
    warp = np.array([[1 / sc, 0, -tx / sc], [0, 1 / sc, -ty / sc]], np.float32)
    H, W = ea.shape
    eb_full = np.zeros((max(H, eb.shape[0]), max(W, eb.shape[1])), np.float32)
    eb_full[:eb.shape[0], :eb.shape[1]] = eb
    cc = 0.0
    for s_ in (0.25, 0.5, 1.0):
        A = cv2.resize(ea, None, fx=s_, fy=s_)
        B = cv2.resize(eb_full, None, fx=s_, fy=s_)
        Wm = warp.copy()
        Wm[:, 2] *= s_
        try:
            cc, Wm = cv2.findTransformECC(A, B, Wm, cv2.MOTION_AFFINE,
                                          (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-6), None, 5)
            warp = Wm.copy()
            warp[:, 2] /= s_
        except cv2.error:
            break
    return warp, float(cc)


def warp_to(ref_shape, img, warp, interp=cv2.INTER_LINEAR):
    return cv2.warpAffine(img, warp, (ref_shape[1], ref_shape[0]), flags=interp + cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# --------------------------------------------------------------------------- masks

def classify(rgb, valid, mpp):
    """Rough land-cover masks:
      snow    - bright, colourless
      forest  - large rough-textured areas (crowns + shadows) or large very dark areas
                (water, conifers); also cultivated fields with strong row texture
      open    - smooth open ground (meadow, pasture, bare soil): where pits are readable
      zones   - open ground split into brightness zones (e.g. meadow vs ploughed field),
                so each is normalised on its own and field edges don't fake slopes."""
    L = luminance(rgb)
    mx, mn = rgb.max(-1), rgb.min(-1)
    sat = (mx - mn) / (mx + 1e-6)
    snow = (L > 0.72) & (sat < 0.18) & valid
    snow = ndi.binary_opening(snow, iterations=1)
    nosnow = valid & ~snow
    # texture = local standard deviation at ~2 m, compared with the smoothest ground
    sig = max(1.0, 2.0 / mpp)
    Lf = np.where(nosnow, L, np.median(L[nosnow]) if nosnow.any() else 0.5).astype("float32")
    w = cv2.dilate(nosnow.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=0).astype("float32")
    w = (ndi.binary_erosion(nosnow, iterations=2) | ~ndi.binary_dilation(snow, iterations=3)).astype("float32") * nosnow
    W1 = np.maximum(cv2.GaussianBlur(w, (0, 0), sig), 1e-3)
    m1 = cv2.GaussianBlur(Lf * w, (0, 0), sig) / W1
    m2 = cv2.GaussianBlur(Lf * Lf * w, (0, 0), sig) / W1
    tex = np.sqrt(np.clip(m2 - m1 * m1, 0, None))
    # forest texture is persistent over tens of metres; a row of pits is not
    tex = cv2.GaussianBlur(tex, (0, 0), max(2.0, 6 / mpp))
    smooth_ref = np.percentile(tex[nosnow], 10) if nosnow.any() else 0.02
    Ls = cv2.GaussianBlur(Lf, (0, 0), max(1.0, 3 / mpp))
    smooth = nosnow & (tex < 1.5 * smooth_ref)
    # the brighter smooth surfaces (meadow, pasture) are the reference, not a dark ploughed field
    ground_L = np.percentile(Ls[smooth], 75) if smooth.any() else np.median(Ls[nosnow])
    # forest: rough AND clearly darker than open ground, or very rough (bare winter woods);
    # a row of pits or a weedy strip is rough but as bright as the meadow
    rough = ((tex > 2.0 * smooth_ref) & (Ls < 0.7 * ground_L)) | (tex > 3.5 * smooth_ref)
    very_dark = Ls < 0.45 * ground_L
    min_area = 600 / mpp ** 2          # > 600 m2: forest, water, crop field; not a single bush

    def big_parts(mask, close_m):
        mask = ndi.binary_closing(mask & valid, iterations=max(1, int(close_m / mpp)))
        lab = measure.label(mask)
        keep = np.zeros_like(mask)
        for p in measure.regionprops(lab):
            if p.area >= min_area:
                keep[lab == p.label] = True
        return keep

    # fine-scale 'tree-like' mask (crown texture + darker than the image median). Used ONLY to
    # look for rows of trees - never to hide candidates, because a field of pits looks similar.
    s1 = max(1.0, 1.5 / mpp)
    f1 = cv2.GaussianBlur(Lf, (0, 0), s1)
    f2 = cv2.GaussianBlur(Lf * Lf, (0, 0), s1)
    ftex = cv2.GaussianBlur(np.sqrt(np.clip(f2 - f1 * f1, 0, None)), (0, 0), max(1.0, 3 / mpp))
    Ld = cv2.GaussianBlur(Lf, (0, 0), max(1.0, 2 / mpp))
    treeish = (ftex > 2.0 * np.percentile(ftex[nosnow], 10)) & (Ld < np.percentile(Ld[nosnow], 55)) & nosnow
    treeish = ndi.binary_opening(treeish, iterations=max(1, int(1.5 / mpp)))
    trees = big_parts(rough, 1.0)
    dark_field = big_parts(very_dark & ~trees, 1.0)
    # cultivated field: rough but bright, and WIDE (survives an opening of ~20 m); a strip
    # of pits along a field edge is rough too, but only ~10 m wide
    rb = (tex > 2.0 * smooth_ref) & ~trees & ~dark_field & valid
    r_open = max(2, int(10 / mpp))
    small = cv2.resize(rb.astype(np.uint8), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(1, r_open // 4) + 1,) * 2)
    wide = cv2.dilate(cv2.erode(small, k), k)
    wide = cv2.resize(wide, (rb.shape[1], rb.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool) & rb
    field = big_parts(wide, 1.0) | dark_field
    trees = ndi.binary_dilation(trees, iterations=max(1, int(1.5 / mpp)))
    field = ndi.binary_dilation(field, iterations=max(1, int(1.0 / mpp)))
    big = trees | field
    openm = valid & ~big & ~snow
    # brightness zones on open ground (2-means on smoothed luminance)
    zones = np.zeros(L.shape, np.int32)
    if openm.sum() > 1000:
        Lz = cv2.GaussianBlur(Lf, (0, 0), max(2.0, 8 / mpp))
        v = Lz[openm]
        c1, c2 = np.percentile(v, 20), np.percentile(v, 80)
        for _ in range(10):
            lab2 = np.abs(v - c1) > np.abs(v - c2)
            if lab2.all() or (~lab2).all():
                break
            c1, c2 = v[~lab2].mean(), v[lab2].mean()
        if abs(c2 - c1) > 0.08:   # two clearly different surfaces
            zones[openm] = 1 + (np.abs(Lz[openm] - c1) > np.abs(Lz[openm] - c2))
        else:
            zones[openm] = 1
    return {"snow": snow, "forest": big & valid, "trees": trees & valid, "field": field & valid,
            "treeish": treeish | (trees & valid),
            "open": openm, "zones": zones}


# --------------------------------------------------------------------------- shading

def shading(rgb, cls, mpp, window_m=25.0):
    """Relative brightness s = L / local mean - 1, computed inside each brightness zone
    of open ground separately; masked areas are 0 (no slope information)."""
    L = luminance(rgb).astype("float64")
    sig = max(2.0, window_m / mpp / 2.5)
    s = np.zeros_like(L)
    for zid in np.unique(cls["zones"]):
        if zid == 0:
            continue
        zm = cls["zones"] == zid
        w = zm.astype("float64")
        mean = ndi.gaussian_filter(L * w, sig) / np.maximum(ndi.gaussian_filter(w, sig), 1e-3)
        s[zm] = L[zm] / np.maximum(mean[zm], 1e-3) - 1
    s = ndi.gaussian_filter(s, max(0.5, 0.3 / mpp))
    edge = ndi.binary_dilation(cls["zones"] == 0, iterations=max(1, int(1.5 / mpp)))
    s[edge] *= 0.3     # zone borders: brightness steps, not slopes
    return np.clip(s, -0.6, 0.6)


def estimate_sun(s, open_mask, mpp, forest=None):
    """Sun azimuth (degrees clockwise from image 'up') from object/shadow pairs.

    Isolated bushes and trees in open ground are the strongest compact objects: their
    sun side is bright and the shadow lies on the far side. For each dark compact blob we
    take the vector from the nearby bright mass to the dark mass (= away from the sun);
    the weighted circular mean of these vectors gives the sun direction.
    Returns (azimuth, reliability 0..1, number of objects used)."""
    thr = np.percentile(s[open_mask], 3) if open_mask.any() else -0.2
    dark = (s < min(thr, -0.12)) & open_mask
    lab = measure.label(dark)
    vecs = []
    for p in measure.regionprops(lab, intensity_image=-s):
        area_m2 = p.area * mpp ** 2
        if not (1.5 <= area_m2 <= 150):
            continue
        r = max(3, int(2.5 * p.equivalent_diameter_area))
        cy, cx = p.centroid
        y0, y1 = max(0, int(cy) - r), min(s.shape[0], int(cy) + r + 1)
        x0, x1 = max(0, int(cx) - r), min(s.shape[1], int(cx) + r + 1)
        win = s[y0:y1, x0:x1]
        pos = np.clip(win, 0, None) * open_mask[y0:y1, x0:x1]
        if pos.sum() < 1e-3:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        by, bx = (pos * yy).sum() / pos.sum(), (pos * xx).sum() / pos.sum()
        v = np.array([cx - bx, cy - by])      # bright -> dark, image coords (x right, y down)
        n = np.linalg.norm(v)
        if n < 0.5:
            continue
        vecs.append(v / n * p.intensity_mean * math.sqrt(p.area))
    if len(vecs) < 3:
        return None, 0.0, len(vecs)
    vecs = np.array(vecs)
    # strongest objects are mostly bushes/trees (positive relief); use the top half
    mag = np.linalg.norm(vecs, axis=1)
    top = vecs[mag >= np.median(mag)]
    m = top.sum(axis=0)
    reliab = float(np.linalg.norm(m) / np.linalg.norm(top, axis=1).sum())
    away = math.degrees(math.atan2(m[0], -m[1])) % 360   # azimuth of 'away from sun' (up = 0)
    return (away + 180) % 360, reliab, len(top)


def photoclinometry(s, sun_az_deg, sun_el_deg, mpp, highpass_m=20.0):
    """Linear shape-from-shading. s = -cot(e) * dh/du, u = horizontal unit vector towards
    the sun. Integrate along u in the Fourier domain with a high-pass so only relief
    smaller than ~highpass_m survives. Returns heights in metres (rough)."""
    h, w = s.shape
    az = math.radians(sun_az_deg)
    ux, uy = math.sin(az), -math.cos(az)            # image x right, y down; up = north
    slope_u = -s * math.tan(math.radians(sun_el_deg))   # dh/du, dimensionless
    pad = 32
    S = np.fft.fft2(np.pad(slope_u, pad, mode="reflect"))
    fy = np.fft.fftfreq(h + 2 * pad)[:, None]
    fx = np.fft.fftfreq(w + 2 * pad)[None, :]
    k = 2j * math.pi * (fx * ux + fy * uy)
    f0 = mpp / highpass_m                           # cycles per pixel
    r2 = fx ** 2 + fy ** 2
    reg = (2 * math.pi) ** 2 * f0 ** 2
    # dh/du = slope -> H = S * conj(k) / (|k|^2 + reg); also damp very low freq isotropically
    H = S * np.conj(k) / (np.abs(k) ** 2 + reg) * (1 - np.exp(-r2 / f0 ** 2))
    z = np.real(np.fft.ifft2(H))[pad:pad + h, pad:pad + w] * mpp
    return z.astype("float32")


# --------------------------------------------------------------------------- marks

def marks(rgb, cls, mpp, min_m=1.5, max_m=25.0):
    """Compact colour anomalies on open ground."""
    out = []
    L = luminance(rgb)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    exg = 2 * g - r - b                                   # greenness
    openm = cls["open"] | cls["snow"]
    sig = max(3.0, 15 / mpp)
    w = openm.astype("float32")

    def z(x):
        m = ndi.gaussian_filter(x * w, sig) / np.maximum(ndi.gaussian_filter(w, sig), 1e-3)
        v = ndi.gaussian_filter((x - m) ** 2 * w, sig) / np.maximum(ndi.gaussian_filter(w, sig), 1e-3)
        return (x - m) / np.sqrt(v + 1e-6)

    zL, zG = z(L), z(exg)
    border = ndi.binary_dilation(ndi.morphological_gradient(cls["zones"], size=3) > 0,
                                 iterations=max(1, int(2 / mpp)))
    near_trees = ndi.binary_dilation(cls["trees"], iterations=max(1, int(12 / mpp)))
    kinds = [
        ("snow_patch", cls["snow"] & ~near_trees, "lingering snow: hollows and shaded pits keep snow longest"),
        ("dark_spot", (zL < -2.0) & openm & ~cls["snow"] & ~border,
         "dark spot: bush/tree, its shadow, damp soil or lush growth in a hollow"),
        ("green_spot", (zG > 2.5) & (zL > -2.0) & openm & ~cls["snow"] & ~border,
         "greener growth: deeper, wetter soil (filled pit, ditch, old yard)"),
        ("light_spot", (zL > 2.5) & openm & ~cls["snow"] & ~border, "light soil or dry mound"),
    ]
    for kind, mask, note in kinds:
        mask = ndi.binary_opening(mask, iterations=1)
        for p in measure.regionprops(measure.label(mask)):
            d = p.equivalent_diameter_area * mpp
            if not (min_m <= d <= max_m) or p.solidity < 0.6 or p.eccentricity > 0.95:
                continue
            strength = float(np.abs(zL[p.coords[:, 0], p.coords[:, 1]]).mean())
            conf = 0.75 if kind == "snow_patch" else float(np.clip((strength - 2.0) / 4.0, 0.05, 0.9))
            out.append({"type": "mark", "kind": kind, "row": float(p.centroid[0]), "col": float(p.centroid[1]),
                        "diameter_m": round(d, 1), "contrast_z": round(strength, 1),
                        "confidence": round(conf, 2), "note": note})
    return out


# --------------------------------------------------------------------------- one image

def plausible_sun(az, lat):
    """Satellite images are taken around late morning; outside the tropics the sun is
    then in the southern half of the sky (northern hemisphere) or the northern half
    (southern hemisphere). Bushes and hollows give mirror-image shading, so the shadow
    estimate is ambiguous by 180 deg - the latitude settles it."""
    if az is None or lat is None or abs(lat) < 25:
        return az, False
    south = 90 <= az <= 270
    if (lat > 0) != south:
        return (az + 180) % 360, True
    return az, False


def _fine_texture(rgb, valid):
    """Small-scale brightness texture (std of the Laplacian of L after a 3 px blur). Ploughed or
    harrowed fields are very smooth (< 0.13 on 0.3 m/px screenshots); grass, scrub and village
    sites are rougher."""
    L = cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)[..., 0]
    lap = cv2.Laplacian(cv2.GaussianBlur(L, (0, 0), 3), cv2.CV_32F)
    return float(np.std(lap[valid])) if valid.any() else 1.0


def _merge_learned(cands, learned_c, mpp):
    """The learned detector re-ranks what the hand-made detectors found: a candidate the model
    also picked keeps the higher score and says so; one the model did not pick is demoted (the
    model was right more often on the marked examples); model-only spots are added."""
    used = set()
    for c in cands:
        if c["type"] == "linear" or c.get("kind") in ("crop_mark", "soil_mark", "clearing", "canopy_anomaly"):
            continue
        best, bj = None, None
        for j, l in enumerate(learned_c):
            d = math.hypot(c["row"] - l["row"], c["col"] - l["col"])
            if d <= max(2.0 * l["radius_px"], 6.0 / mpp ** 0.5) and (best is None or d < best):
                best, bj = d, j
        if bj is None:
            c["learned_p"] = 0.0
            if c.get("depth_to_noise", 0) < 9:        # a very clear hollow in the relief keeps its score
                c["confidence"] = round(c["confidence"] * 0.55, 2)
        else:
            used.add(bj)
            lp = learned_c[bj]["learned_p"]
            c["learned_p"] = lp
            c["confidence"] = round(min(1.0, 0.2 * c["confidence"] + 0.9 * lp), 2)
            c["note"] = c.get("note", "") + f"; also picked by the learned model ({lp:.2f})"
    for j, l in enumerate(learned_c):
        l["diameter_m"] = round(2 * l["radius_px"] * mpp, 1)
        if j not in used:
            # a spot only the model saw goes in at half its score: tested on a second marked pack
            # (model trained on the other one), full weight pushed real field patches down the list
            l["confidence"] = round(0.5 * l["learned_p"], 2)
            cands.append(l)
    return cands


def analyse_image(rgb, valid, mpp, sun_az=None, sun_el=30.0, lat=None, north_up_deg=0.0, model="default"):
    cls = classify(rgb, valid, mpp)
    s = shading(rgb, cls, mpp)
    est = estimate_sun(s, cls["open"], mpp)
    est_az = None if est[0] is None else (est[0] + north_up_deg) % 360  # image -> true azimuth
    est_az, flipped = plausible_sun(est_az, lat)
    est_img = None if est_az is None else (est_az - north_up_deg) % 360
    az_used, az_src = ((sun_az - north_up_deg) % 360, "user") if sun_az is not None else \
        (est_img, "estimated from shadows" + (" (180 deg ambiguity resolved by latitude)" if flipped else ""))
    info = {"sun_azimuth_estimate": None if est_az is None else round(est_az),
            "sun_estimate_reliability": round(est[1], 2), "sun_objects": est[2],
            "sun_azimuth_used": None if az_used is None else round(az_used), "sun_source": az_src,
            "sun_elevation_used": sun_el,
            "snow_share": round(float(cls["snow"].mean()), 3),
            "forest_share": round(float(cls["forest"].mean()), 3)}
    result = {"classes": cls, "shading": s, "info": info, "relief": None, "candidates": []}
    cands = marks(rgb, cls, mpp)
    if az_used is not None:
        z = photoclinometry(s, az_used, sun_el, mpp)
        result["relief"] = z
        lrm = terrain.local_relief(z, mpp, window_m=30)
        lrm[~cls["open"]] = 0
        zz = np.where(cls["open"], z, np.median(z[cls["open"]]) if cls["open"].any() else 0)
        noise = feat.noise_level(lrm[cls["open"]]) if cls["open"].any() else 0.05
        for p in feat.detect_pits(zz, lrm, mpp, min_depth=max(0.03, 2.5 * noise), min_diam=2.0, max_diam=25.0):
            p["source_signal"] = "shading (photoclinometric relief)"
            snr = p["depth_m"] / max(noise, 1e-6)
            p["depth_to_noise"] = round(snr, 1)
            # on a screenshot the shape is unreliable; how far the hollow stands out is what counts
            p["confidence"] = round(float(np.clip((snr - 3.0) / 6.0, 0.05, 1.0)), 2)
            p["depth_m"] = round(p["depth_m"], 2)
            p["note"] = "hollow from shading - " + p["note"]
            cands.append(p)
        for ln in feat.detect_linear(lrm, mpp, min_length=30, min_rel=max(0.04, 2.0 * noise)):
            # banks, lanes and terraces of deserted villages are often curved; keep them,
            # but long ones only, and say how straight they are
            if ln["straightness"] >= 0.8 or ln["length_m"] >= 45:
                ln["source_signal"] = "shading (photoclinometric relief)"
                ln["confidence"] = round(float(np.clip(0.35 + ln["length_m"] / 200 + 0.2 * ln["straightness"], 0, 0.95)), 2)
                ln["note"] += "; from an image - could also be a modern track or field edge"
                cands.append(ln)
        for r in feat.detect_rectangles(lrm, mpp, min_side=5.0, max_side=40.0, min_rel=max(0.04, 2.0 * noise)):
            r["source_signal"] = "shading (photoclinometric relief)"
            r["confidence"] = round(r["confidence"] * 0.8, 2)
            cands.append(r)
        result["lrm"] = lrm
    for c in cands:
        c.setdefault("source_signal", "colour anomaly")
    # nothing from the open-ground detectors inside forest/fields, nothing in masked UI
    cands = [c for c in cands if not cls["forest"][int(c["row"]), int(c["col"])]
             and valid[int(c["row"]), int(c["col"])]]
    # a group of hollows (3+ others within 30 m) is how house pits of a village look;
    # random noise does not cluster like that
    pits_ = [c for c in cands if c["type"] == "pit"]
    if pits_:
        P = np.array([[c["row"], c["col"]] for c in pits_])
        D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1]) * mpp
        for c, n in zip(pits_, (D < 30).sum(1) - 1):
            if n >= 3:
                c["group_size"] = int(n + 1)
                c["confidence"] = round(min(1.0, c["confidence"] + 0.04 * min(n, 5)), 2)
    if model is not None:
        from . import learned
        mdl = learned.load_model() if model == "default" else model
        if mdl is not None:
            lc = learned.detect(rgb, valid, None if az_used is None else az_used, lat, model=mdl, min_p=0.2)
            lc = [c for c in lc if not cls["forest"][int(c["row"]), int(c["col"])]]
            top = max([c["learned_p"] for c in lc] or [0.0])
            smooth = _fine_texture(rgb, valid)
            # if nothing in the picture looks like the marked examples, the model probably does not
            # know this kind of picture (other region, zoom, sensor): then it is not allowed to re-rank.
            # Very smooth ground = a ploughed field: the model was trained on grass/scrub village sites
            # and made field results worse in testing - there the rows of patches are what counts.
            used_model = top >= 0.4 and smooth >= 0.13
            if used_model:
                cands = _merge_learned(cands, lc, mpp)
            info["learned_model"] = {"spots": len(lc), "best_score": round(top, 2), "fine_texture": round(smooth, 3),
                                     "used": used_model, "validation": mdl.get("validation"),
                                     "note": None if used_model else
                                     ("smooth bare field - model not used (rows of patches matter here)"
                                      if smooth < 0.13 else
                                      "nothing here resembles the marked training examples - model not used")}
    extra = field_marks(rgb, cls, mpp) + forest_marks(rgb, cls, mpp)
    for c in extra:
        r_, c_ = int(np.clip(c["row"], 0, valid.shape[0] - 1)), int(np.clip(c["col"], 0, valid.shape[1] - 1))
        if valid[r_, c_]:
            cands.append(c)
    result["candidates"] = cands
    return result


# --------------------------------------------------------------------------- fusion

def fuse(per_image, mpp, radius_m=4.0):
    """Merge candidates from several aligned images. A place flagged by several signals
    or several images is more convincing than one blob in one picture."""
    pts = []
    for idx, res in enumerate(per_image):
        for c in res["candidates"]:
            pts.append((idx, c))
    used = [False] * len(pts)
    groups = []
    order = sorted(range(len(pts)), key=lambda i: -pts[i][1]["confidence"])
    for i in order:
        if used[i]:
            continue
        ii, ci = pts[i]
        if ci["type"] == "linear":
            used[i] = True
            groups.append([pts[i]])
            continue
        grp = [pts[i]]
        used[i] = True
        for j in order:
            if used[j] or pts[j][1]["type"] == "linear":
                continue
            cj = pts[j][1]
            if math.hypot(ci["row"] - cj["row"], ci["col"] - cj["col"]) * mpp <= radius_m:
                grp.append(pts[j])
                used[j] = True
        groups.append(grp)
    out = []
    for grp in groups:
        main = dict(max((c for _, c in grp), key=lambda c: (c["type"] == "pit", c["confidence"])))
        images = sorted({i for i, _ in grp})
        signals = sorted({(c.get("kind") or c["type"]) for _, c in grp})
        main["seen_in_images"] = [i + 1 for i in images]
        main["signals"] = signals
        # independent evidence only: another picture, or snow lying in the hollow. A shading
        # hollow and a dark/green spot in the SAME picture are usually the same patch of grass.
        best = max(c["confidence"] for _, c in grp)
        snow = any(c.get("kind") == "snow_patch" for _, c in grp)
        conf = best + 0.2 * (len(images) - 1) + (0.15 if snow and main.get("kind") != "snow_patch" else 0)
        main["confidence"] = round(min(1.0, conf), 2)
        if main["type"] == "mark" and any(c["type"] == "pit" for _, c in grp):
            main["type"] = "pit"
        out.append(main)
    return out


# --------------------------------------------------------------------------- fields

def _lab(rgb):
    return cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)


def _masked_mean(x, w, sig):
    return ndi.gaussian_filter(x * w, sig) / np.maximum(ndi.gaussian_filter(w, sig), 1e-3)


FIELD_NOTES = {
    "soil_dark": "darker soil patch: organic-rich / burnt cultural layer of a ploughed-out house or yard, "
                 "or simply damp ground",
    "soil_light": "lighter soil patch: chalk, clay, brick or stone rubble brought up by the plough, or an eroded knoll",
    "crop_green": "greener / denser crop: deeper soil over a filled pit or ditch keeps moisture",
    "crop_parched": "parched / yellower crop: shallow soil over a wall, road or rubble",
}


def field_marks(rgb, cls, mpp, min_m=7.0, max_m=90.0):
    """Soil and crop marks in fields and on open ground: patches of different colour
    10-90 m across, after smoothing away furrows, rows and tramlines. Ploughed-out
    settlements often show up only like this."""
    region = (cls["field"] | cls["open"]) & ~cls["snow"]
    if region.sum() < 2000:
        return []
    lab = _lab(rgb)
    sig_small = max(1.0, 3.0 / mpp)              # wipes out furrows / crop rows
    sig_bg = max(5.0, 30.0 / mpp)                # background of the same field
    w = region.astype("float32")
    # work per zone, so the meadow is not compared with the ploughed field next to it
    zones = np.where(cls["field"], 99, cls["zones"])
    zones[~region] = 0
    zmap = {k: np.zeros(region.shape, np.float32) for k in ("L", "a", "b")}
    for zid in np.unique(zones):
        if zid == 0:
            continue
        zm = (zones == zid).astype("float32")
        if zm.sum() < 1500:
            continue
        for ci, k in enumerate(("L", "a", "b")):
            ch = ndi.gaussian_filter(lab[..., ci] * zm, sig_small) / np.maximum(ndi.gaussian_filter(zm, sig_small), 1e-3)
            d = ch - _masked_mean(ch, zm, sig_bg)
            v = d[zm > 0]
            mad = 1.4826 * np.median(np.abs(v - np.median(v))) + 1e-3
            zmap[k] = np.where(zm > 0, d / mad, zmap[k])
    edge = ndi.binary_dilation(~region, iterations=max(1, int(6 / mpp)))
    out = []
    kinds = [("soil_dark", zmap["L"] < -2.5), ("soil_light", zmap["L"] > 2.5),
             ("crop_green", (zmap["a"] < -2.5) & (np.abs(zmap["L"]) < 4)),
             ("crop_parched", (zmap["b"] > 2.5) & (zmap["a"] > 0))]
    # mottling: how much of the ground is 'anomalous' at all. Many patches everywhere is
    # what natural soil (alkaline spots, subsidence hollows, wet patches) looks like.
    loose = ((np.abs(zmap["L"]) > 2.0) | (np.abs(zmap["a"]) > 2.0)) & region & ~edge
    mottling = float(loose.sum() / max((region & ~edge).sum(), 1))
    for kind, mask in kinds:
        mask = mask & region & ~edge
        mask = ndi.binary_opening(mask, iterations=max(1, int(1.5 / mpp)))
        for p in measure.regionprops(measure.label(mask)):
            d = p.equivalent_diameter_area * mpp
            if not (min_m <= d <= max_m):
                continue
            if p.eccentricity > 0.96 or p.solidity < 0.5:       # tramlines, tracks, field strips
                continue
            zz = np.abs(zmap["L" if kind.startswith("soil") else ("a" if kind == "crop_green" else "b")]
                        [p.coords[:, 0], p.coords[:, 1]]).mean()
            in_field = bool(cls["field"][int(p.centroid[0]), int(p.centroid[1])])
            if not in_field and d > 25:       # big patches in a meadow are just grass variation
                continue
            conf = float(np.clip(0.25 + 0.12 * (zz - 2.5) + (0.1 if in_field else 0), 0.1, 0.85))
            if mottling > 0.12:
                conf *= 0.7
            out.append({"type": "mark", "kind": kind, "row": float(p.centroid[0]), "col": float(p.centroid[1]),
                        "diameter_m": round(d, 1), "area_m2": round(p.area * mpp ** 2),
                        "contrast_z": round(float(zz), 1), "in_field": in_field,
                        "confidence": round(conf, 2), "source_signal": "soil/crop colour",
                        "note": FIELD_NOTES[kind] + ("; the whole area is mottled - natural soil patches are likely"
                                                     if mottling > 0.12 else "")})
    # a cluster of marks is what a ploughed-out village looks like
    for m in out:
        n = sum(1 for o in out if o is not m and math.hypot(o["row"] - m["row"], o["col"] - m["col"]) * mpp < 120)
        if n >= 3:
            m["cluster_size"] = n + 1
            m["confidence"] = round(min(0.95, m["confidence"] + 0.1), 2)
    for m in out:
        m["area_mottling"] = round(mottling, 2)
    return out


# --------------------------------------------------------------------------- forest

FOREST_NOTES = {
    "forest_clearing": "opening inside woodland: former yard, house site or glade",
    "canopy_anomaly": "trees of a different colour/texture: different species or age - old orchards, lilac, "
                      "poplars and willows often mark former farmsteads",
    "tree_row": "straight row of trees: shelterbelt, field boundary or a planted village street / estate line",
}


def forest_marks(rgb, cls, mpp):
    out = []
    trees = cls["trees"]
    # 1) clearings: non-tree holes enclosed by woodland
    if trees.any():
        holes = ndi.binary_fill_holes(trees) & ~trees
        for p in measure.regionprops(measure.label(holes)):
            a = p.area * mpp ** 2
            if 60 <= a <= 6000 and p.solidity > 0.6:
                out.append({"type": "mark", "kind": "forest_clearing", "row": float(p.centroid[0]),
                            "col": float(p.centroid[1]), "diameter_m": round(p.equivalent_diameter_area * mpp, 1),
                            "area_m2": round(a), "confidence": 0.45, "source_signal": "forest structure",
                            "note": FOREST_NOTES["forest_clearing"]})
    # 2) canopy colour anomalies inside woodland
    if trees.sum() > 3000:
        lab = _lab(rgb)
        w = ndi.binary_erosion(trees, iterations=max(1, int(3 / mpp))).astype("float32")
        if w.sum() > 2000:
            z = np.zeros(trees.shape, np.float32)
            for ci in range(3):
                ch = ndi.gaussian_filter(lab[..., ci], max(1.0, 4 / mpp))
                d = ch - _masked_mean(ch, w, max(5.0, 40 / mpp))
                v = d[w > 0]
                mad = 1.4826 * np.median(np.abs(v - np.median(v))) + 1e-3
                z = np.maximum(z, np.abs(d / mad))
            mask = ndi.binary_opening((z > 3.0) & (w > 0), iterations=max(1, int(2 / mpp)))
            for p in measure.regionprops(measure.label(mask)):
                a = p.area * mpp ** 2
                if 80 <= a <= 15000:
                    zz = float(z[p.coords[:, 0], p.coords[:, 1]].mean())
                    out.append({"type": "mark", "kind": "canopy_anomaly", "row": float(p.centroid[0]),
                                "col": float(p.centroid[1]), "diameter_m": round(p.equivalent_diameter_area * mpp, 1),
                                "area_m2": round(a), "contrast_z": round(zz, 1),
                                "confidence": round(float(np.clip(0.2 + 0.08 * (zz - 3), 0.1, 0.6)), 2),
                                "source_signal": "forest canopy colour", "note": FOREST_NOTES["canopy_anomaly"]})
    # 3) straight rows of trees in open land, and grids of them
    rows = []
    t = cls["treeish"] & ~ndi.binary_erosion(trees, iterations=max(1, int(6 / mpp)))
    for p in measure.regionprops(measure.label(t)):
        L = p.axis_major_length * mpp
        W = p.axis_minor_length * mpp
        if L < 40 or W > 18 or W / max(L, 1e-6) > 0.25:
            continue
        pts = p.coords.astype(float)
        c = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        dev = np.abs((pts - c) @ vt[1]).max() * mpp
        if dev > max(10.0, 0.12 * L):
            continue
        ang = (90 - math.degrees(p.orientation)) % 180
        e = vt[0] * (p.axis_major_length / 2)
        rows.append({"type": "linear", "kind": "tree_row", "relief": "trees", "row": float(c[0]), "col": float(c[1]),
                     "length_m": round(L), "mean_width_m": round(W, 1), "straightness": round(1 - dev / max(L, 1), 2),
                     "orientation_deg": round(ang), "ends_px": [[float(c[0] - e[0]), float(c[1] - e[1])],
                                                              [float(c[0] + e[0]), float(c[1] + e[1])]],
                     "confidence": 0.4, "source_signal": "tree pattern", "note": FOREST_NOTES["tree_row"]})
    for r in rows:   # parallel rows 15-80 m apart = planted grid (streets, estates, orchards)
        par = [o for o in rows if o is not r and min(abs(o["orientation_deg"] - r["orientation_deg"]),
                                                       180 - abs(o["orientation_deg"] - r["orientation_deg"])) < 12
               and 15 <= math.hypot(o["row"] - r["row"], o["col"] - r["col"]) * mpp <= 120]
        if len(par) >= 2:
            r["parallel_rows"] = len(par) + 1
            r["confidence"] = 0.6
            r["note"] += f"; part of a set of {len(par) + 1} parallel rows - planted grid (estates? orchard? shelterbelts?)"
    return out + rows
