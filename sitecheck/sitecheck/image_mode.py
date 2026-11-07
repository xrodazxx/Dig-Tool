"""`sitecheck image`: the whole screenshot workflow -> the same report / prompt / review
files as the DEM workflow, so the user and the AI see one format."""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import shutil

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import __version__, context as ctxmod, imagery as im, score  # noqa: E402
from .pipeline import _public, _summary  # noqa: E402
from .report import build_prompt  # noqa: E402

KIND_COLORS = {"row": "#ff00ff", "pit": "#e8452c", "snow_patch": "#18c6e6", "dark_spot": "#f2b300", "green_spot": "#3ccf4e",
               "light_spot": "#ff8c1a", "linear": "#d63cd6", "rectangle": "#1f6fff",
               "soil_dark": "#7a3b12", "soil_light": "#f5e6b0", "crop_green": "#00e08a", "crop_parched": "#ffd400",
               "forest_clearing": "#ffffff", "canopy_anomaly": "#ff4fd8", "tree_row": "#9b5cff"}
FIELD_FOREST = {"soil_dark", "soil_light", "crop_green", "crop_parched", "forest_clearing", "canopy_anomaly", "tree_row"}


def scale_from_known(spec):
    x1, y1, x2, y2, metres = map(float, spec.split(","))
    return metres / math.hypot(x2 - x1, y2 - y1)


def run_images(paths, out_dir, lat=None, lon=None, mpp=None, known=None, spot_px=None,
               masks=(), sun_az=None, sun_el=30.0, north_deg=0.0, context_path=None,
               weights_path=None, water_m=None):
    os.makedirs(out_dir, exist_ok=True)
    warnings = []
    if known:
        mpp = scale_from_known(known)
    scale_known = mpp is not None
    if not scale_known:
        mpp = 0.3
        warnings.append("Scale unknown: assumed 0.3 m per pixel (typical close Google Earth view). Sizes and "
                        "distances may be off by a factor of 2-3. Measure a distance with the ruler and pass "
                        "--known X1,Y1,X2,Y2,METRES or --mpp.")

    imgs = [im.load(p, masks) for p in paths]
    ref_rgb, ref_valid = imgs[0]
    H, W = ref_valid.shape
    if spot_px is None:
        spot_px = (W / 2, H / 2)
    sx, sy = spot_px

    # --- align every other image to the first ------------------------------------------
    warps, align_info = [np.eye(2, 3, dtype=np.float32)], [{"image": 1, "reference": True}]
    for k, (rgb, valid) in enumerate(imgs[1:], 2):
        wmat, cc = im.align(ref_rgb, ref_valid, rgb, valid)
        sc = math.sqrt(abs(np.linalg.det(wmat[:, :2])))
        good = cc > 0.3 and 0.8 < sc < 1.25
        align_info.append({"image": k, "correlation": round(cc, 2), "scale": round(sc, 3), "ok": bool(good)})
        if not good:
            warnings.append(f"Image {k} could not be aligned reliably (correlation {cc:.2f}); its candidates are "
                            "kept but not merged with the others. Use screenshots of the same view.")
        warps.append(wmat if good else None)

    # --- analyse each image ----------------------------------------------------------------
    results, reliefs = [], []
    for k, ((rgb, valid), wmat) in enumerate(zip(imgs, warps)):
        r = im.analyse_image(rgb, valid, mpp, sun_az, sun_el, lat, north_deg)
        r["info"]["image"] = os.path.basename(paths[k])
        # candidates -> reference pixel coordinates
        if wmat is not None and k > 0:
            A = np.vstack([wmat, [0, 0, 1]])
            Ainv = np.linalg.inv(A)
            for c in r["candidates"]:
                c["img_row"], c["img_col"] = c["row"], c["col"]
                x, y, _ = Ainv @ np.array([c["col"], c["row"], 1.0])
                c["col"], c["row"] = float(x), float(y)
                for key in ("ends_px",):
                    if key in c:
                        c[key] = [[float((Ainv @ [cc_, rr, 1])[1]), float((Ainv @ [cc_, rr, 1])[0])]
                                  for rr, cc_ in c[key]]
        if r.get("lrm") is not None and wmat is not None:
            lrm = r["lrm"] if k == 0 else im.warp_to((H, W), r["lrm"], wmat)
            okm = r["classes"]["open"] if k == 0 else im.warp_to((H, W), r["classes"]["open"].astype(np.uint8), wmat,
                                                                  cv2.INTER_NEAREST).astype(bool)
            reliefs.append((lrm, okm))
        results.append(r)

    # --- regular rows of patches / hollows (houses along a street) ----------------------------
    # Rows are searched in each picture separately (positions of weak patches differ by metres
    # between seasons, so merging points only adds noise), then compared: the same row in two
    # seasons is strong evidence; a row in one of several pictures is corrected for the extra tries.
    from scipy import ndimage as _ndi
    from . import patterns as pt
    rows_per_img, n_patch, n_tried = [], 0, 0
    for k, ((rgb, valid), wmat, r) in enumerate(zip(imgs, warps, results)):
        if wmat is None and k > 0:
            continue
        n_tried += 1
        reg = (r["classes"]["field"] | r["classes"]["open"]) & ~r["classes"]["snow"]
        bl, maps_k = pt.find_blobs(rgb, reg, mpp, return_maps=True)
        for c in r["candidates"]:
            if c["type"] == "pit":
                bl.append({"row": c.get("img_row", c["row"]), "col": c.get("img_col", c["col"]),
                           "diameter_m": c.get("diameter_m", 4.0), "kind": "hollow",
                           "strength": round(min(6.0, c.get("depth_to_noise", 3.0)) / 2, 2)})
        n_patch += len(bl)
        rk = pt.find_rows(bl, reg, mpp, lum=_ndi.gaussian_filter(im.luminance(rgb), 1.0), maps=maps_k)
        if k > 0:            # to reference-image pixels
            Ainv = np.linalg.inv(np.vstack([wmat, [0, 0, 1]]))
            tr = lambda rc: [float((Ainv @ [rc[1], rc[0], 1])[1]), float((Ainv @ [rc[1], rc[0], 1])[0])]
            for x in rk:
                x["members_px"] = [tr(m) for m in x["members_px"]]
                x["ends_px"] = [tr(e) for e in x["ends_px"]]
                x["row"], x["col"] = tr([x["row"], x["col"]])
        for x in rk:
            x["seen_in_images"] = [k + 1]
        rows_per_img.append(rk)
    rows = []
    for i, rk in enumerate(rows_per_img):
        for x in rk:
            # same row in another picture? (same family, direction within 10 deg, centres < 30 m)
            twins = [y for j, ry in enumerate(rows_per_img) if j != i for y in ry
                     if y["patch_family"] == x["patch_family"]
                     and min(abs(y["bearing_deg"] - x["bearing_deg"]), 180 - abs(y["bearing_deg"] - x["bearing_deg"])) < 10
                     and math.hypot(y["row"] - x["row"], y["col"] - x["col"]) * mpp < 30]
            if twins:
                tw = min(twins, key=lambda y: y["chance"])
                if any(x is r_ or tw in r_.get("_twins", []) for r_ in rows):
                    continue
                x["_twins"] = [tw]
                x["chance_single"] = x["chance"]
                x["chance"] = round(x["chance"] * tw["chance"], 4)
                x["seen_in_images"] = sorted(set(x["seen_in_images"] + tw["seen_in_images"]))
                x["note"] += f" SAME ROW SEEN IN PICTURE {tw['seen_in_images'][0]} - combined chance {x['chance'] * 100:.1f}%."
            else:
                x["chance_single"] = x["chance"]
                x["chance"] = round(min(1.0, x["chance"] * max(1, n_tried)), 3)   # more pictures = more tries
            # four kinds of patch were searched: correct for that too (honest multiple-testing)
            n_fam = len(pt.FAMILIES)
            x["chance"] = round(min(1.0, x["chance"] * n_fam), 3)
            x["presumption"] = ("presumption" if x["chance"] < 0.10 else
                                "weak presumption" if x["chance"] <= 0.30 else "none")
            x["confidence"] = round(float(np.clip(1 - x["chance"] * 4, 0.05, 0.95)) *
                                    (0.5 if x.get("along_cultivation") else 1.0), 2)
            rows.append(x)
    rows = [x for x in rows if x["chance"] <= 0.30]
    for x in rows:
        x.pop("_twins", None)
        x["signals"] = ["regular_row"]
    rows.sort(key=lambda x: x["chance"])
    pattern_info = {"patches_and_hollows_examined": n_patch, "pictures_searched": n_tried, "rows_found": len(rows)}

    fused = im.fuse([r if (k == 0 or warps[k] is not None) else {"candidates": []} for k, r in enumerate(results)], mpp)
    for k, r in enumerate(results):        # unaligned images: keep their own candidates, unmerged
        if k and warps[k] is None:
            for c in r["candidates"]:
                c["seen_in_images"] = [k + 1]
                c["signals"] = [c.get("kind") or c["type"]]
                c["unaligned"] = True
                fused.append(c)
    edge = max(10.0 / mpp, 0.02 * max(H, W))     # brightness normalisation is blind at the frame edge
    fused = [c for c in fused if edge <= c["row"] < H - edge and edge <= c["col"] < W - edge]
    fused.sort(key=lambda c: -c["confidence"])
    n_all = len(fused)
    group = lambda c: (c.get("kind") in FIELD_FOREST) and c["type"] != "pit"
    fused = (rows[:6] + [c for c in fused if not group(c) and c["confidence"] >= 0.5][:40]
             + [c for c in fused if group(c) and c["confidence"] >= 0.35][:20])
    fused.sort(key=lambda c: -c["confidence"])
    if n_all > len(fused):
        warnings.append(f"{n_all - len(fused)} weaker single-image candidates were dropped (kept the best {len(fused)}).")
    for i, c in enumerate(fused, 1):
        c["id"] = i
        c["detected_by"] = "algorithm (image)"
        c["n_images"] = len(paths)
        c["user_status"] = "unreviewed"
        c["x_m"] = round((c["col"] - sx) * mpp, 1)
        c["y_m"] = round((sy - c["row"]) * mpp, 1)
        c["distance_from_spot_m"] = round(math.hypot(c["x_m"], c["y_m"]))
        rot = math.radians(north_deg)        # image 'up' is north_deg clockwise from true north
        e = c["x_m"] * math.cos(rot) + c["y_m"] * math.sin(rot)
        n = -c["x_m"] * math.sin(rot) + c["y_m"] * math.cos(rot)
        c["bearing_deg"] = round(math.degrees(math.atan2(e, n)) % 360)
        if lat is not None:
            c["lat"] = round(lat + n / 110_574, 6)
            c["lon"] = round(lon + e / (111_320 * math.cos(math.radians(lat))), 6)

    # combined relief (average of aligned images: albedo noise of one season averages out)
    if reliefs:
        num = sum(np.where(m, l, 0) for l, m in reliefs)
        den = sum(m.astype(float) for _, m in reliefs)
        relief = np.where(den > 0, num / np.maximum(den, 1), 0).astype("float32")
    else:
        relief = np.zeros((H, W), "float32")

    ctx = ctxmod.load(context_path)
    water = {"distance_m": water_m, "source": "user" if water_m is not None else None}
    if water_m is None and ctx["water"].get("user_distance_m") is not None:
        water.update(distance_m=ctx["water"]["user_distance_m"], source="user")
    from .water import distance_band
    water["band"] = distance_band(water["distance_m"])
    spot = {"landform": None, "slope_deg": None}     # unknown without elevation data
    sc = score.compute(spot, water, fused, ctx, score.load_weights(weights_path))

    report = {
        "schema": "sitecheck-report/1",
        "generator": f"sitecheck {__version__} (screenshot mode)",
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "spot": {"lat": lat, "lon": lon, "spot_px": [round(sx), round(sy)]},
        "data": {
            "mode": "screenshots - relief estimated from shading, NOT measured",
            "images": [{"file": os.path.basename(p), **ai,
                        "warp": None if w_ is None else np.asarray(w_).round(5).tolist()}
                       for p, ai, w_ in zip(paths, align_info, warps)],
            "image_size": [int(H), int(W)],
            "metres_per_pixel": round(mpp, 3), "scale_known": scale_known,
            "image_north_deg": north_deg,
            "per_image": [r["info"] for r in results],
            "patterns": pattern_info,
        },
        "context": ctx,
        "terrain": {"_source": "not available in screenshot mode (no elevation data)"},
        "water": water,
        "features": [_public_img(c) for c in fused],
        "feature_summary": _summary(fused),
        "score": sc,
        "warnings": warnings + [
            "Screenshot mode: heights come from shading and colour, so they are rough. Treat candidates as "
            "places to look at, not as findings. A hollow seen in two seasons is much stronger evidence."],
        "how_to_read": {
            "signals": "what produced the candidate: pit = hollow in shading-based relief; snow_patch = snow kept "
                       "in a hollow; dark_spot / green_spot = vegetation or soil mark; linear = straight line",
            "seen_in_images": "which screenshots show it (aligned automatically)",
            "x_m / y_m": "metres east / north of the spot (image assumed north-up unless --north given)",
        },
    }
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, default=_np)
    prompt = build_prompt(report).replace(
        "Below is a JSON report produced by an offline tool from a terrain model (DEM / lidar)",
        "Below is a JSON report produced by an offline tool from ordinary satellite screenshots "
        "(no elevation data: hollows were estimated from shading, snow and vegetation marks)")
    prompt = prompt.replace("- `terrain_panels.png` - hillshade, local relief, slope, sky-view factor.",
                            "- `relief_from_shading.png` - relief reconstructed from the shading of the screenshots.\n"
                            + "".join(f"- `{os.path.basename(p)}` - original screenshot {k}.\n" for k, p in enumerate(paths, 1)))
    with open(os.path.join(out_dir, "prompt.md"), "w", encoding="utf-8") as fh:
        fh.write(prompt)
    for p in paths:
        shutil.copy(p, os.path.join(out_dir, os.path.basename(p)))
    with open(os.path.join(out_dir, "features_full.json"), "w", encoding="utf-8") as fh:
        json.dump(fused, fh, indent=1, default=_np)
    gray = im.luminance(ref_rgb)
    np.savez_compressed(os.path.join(out_dir, "layers.npz"), z=relief, lrm=relief,
                        slope=np.zeros_like(relief), hillshade=gray, photo=(ref_rgb * 255).astype(np.uint8),
                        spot_rc=np.array([round(sy), round(sx)]), cell=np.array(mpp))
    render_images(ref_rgb, relief, fused, (sx, sy), mpp, out_dir, [imgs[k][0] for k in range(len(imgs))], warps)
    save_viewer_layers(out_dir, ref_rgb, relief, results[0], imgs, warps, mpp)
    return report


def save_viewer_layers(out_dir, ref_rgb, relief, res0, imgs, warps, mpp):
    """Plain images for the interface, all in the reference picture's pixel grid."""
    H, W = ref_rgb.shape[:2]
    lay = os.path.join(out_dir, "layers")
    os.makedirs(lay, exist_ok=True)
    cv2.imwrite(os.path.join(lay, "picture1.jpg"), cv2.cvtColor((ref_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    for k in range(1, len(imgs)):
        if warps[k] is None:
            continue
        al = im.warp_to((H, W), imgs[k][0], warps[k])
        cv2.imwrite(os.path.join(lay, f"picture{k + 1}.jpg"),
                    cv2.cvtColor((np.clip(al, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    # relief from shading: blue = lower, red = higher, over grey photo
    g = im.luminance(ref_rgb)
    nz = relief[relief != 0]
    lim = float(np.percentile(np.abs(nz), 98)) if nz.size else 0.1
    t = np.clip(relief / max(lim, 1e-6), -1, 1)
    rgbr = np.stack([0.5 + 0.5 * np.clip(t, 0, 1) - 0.3 * np.clip(-t, 0, 1),
                     0.5 - 0.35 * np.abs(t),
                     0.5 + 0.5 * np.clip(-t, 0, 1) - 0.3 * np.clip(t, 0, 1)], -1)
    mix = np.where((relief != 0)[..., None], 0.75 * rgbr + 0.25 * g[..., None], g[..., None] * 0.6)
    cv2.imwrite(os.path.join(lay, "relief.jpg"), cv2.cvtColor((np.clip(mix, 0, 1) * 255).astype(np.uint8),
                                                               cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    # soil / crop colour anomaly: dark-red patches red, light patches yellow, green patches cyan
    from . import patterns as pt
    reg = (res0["classes"]["field"] | res0["classes"]["open"]) & ~res0["classes"]["snow"]
    _, maps = pt.find_blobs(ref_rgb, reg, mpp, return_maps=True)
    if maps:
        sc = lambda m: np.clip((m - 1.5) / 2.5, 0, 1)          # only clear deviations
        dark = sc(np.maximum(maps["darker"], maps["redder"]))
        light = sc(maps["lighter"])
        green = sc(maps["greener"])
        base = g[..., None] * 0.7
        a = np.clip(dark + light + green, 0, 1)[..., None]
        tint = (dark[..., None] * np.array([0.95, 0.2, 0.15]) + light[..., None] * np.array([1.0, 0.9, 0.25])
                + green[..., None] * np.array([0.1, 0.9, 0.85])) / np.maximum(a, 1e-6)
        col = base * (1 - 0.85 * a) + tint * 0.85 * a
        cv2.imwrite(os.path.join(lay, "anomaly.jpg"), cv2.cvtColor((np.clip(col, 0, 1) * 255).astype(np.uint8),
                                                                    cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    np.save(os.path.join(lay, "region.npy"), reg)


def _np(o):
    """numpy scalars/arrays -> plain JSON values"""
    return o.item() if hasattr(o, "item") and np.ndim(o) == 0 else (o.tolist() if hasattr(o, "tolist") else str(o))


def _public_img(c):
    d = _public(c)
    for k in ("kind", "signals", "seen_in_images", "source_signal", "contrast_z", "depth_to_noise", "unaligned",
              "area_m2", "cluster_size", "in_field", "parallel_rows", "group_size", "patch_family", "n_patches",
              "spacing_m", "spacing_variation", "missing_in_row", "bearing_deg", "mean_patch_m", "chance",
              "members_px", "along_cultivation", "chance_single", "presumption"):
        if k in c:
            d[k] = c[k]
    return d


def _draw(ax, feats, to_xy=lambda r, c: (c, r), mpp=0.3, scale=1.0):
    for f in feats:
        if f.get("user_status") == "rejected":
            continue
        k = f.get("kind") if f.get("kind") in KIND_COLORS else (f["type"] if f["type"] != "mark" else "dark_spot")
        col = KIND_COLORS.get(k, "#ffffff")
        ls = "-" if f.get("user_status") == "confirmed" or len(f.get("seen_in_images", [])) > 1 else "--"
        if f["type"] == "row":
            pts = [to_xy(r_, c_) for r_, c_ in f["members_px"]]
            (x1, y1), (x2, y2) = to_xy(*f["ends_px"][0]), to_xy(*f["ends_px"][1])
            ax.plot([x1, x2], [y1, y2], color="#ff00ff", lw=2.2, ls="-" if not f.get("along_cultivation") else ":")
            for x_, y_ in pts:
                ax.add_patch(plt.Circle((x_, y_), max(10, f["mean_patch_m"] / mpp / 2 * scale), fill=False,
                                        ec="#ff00ff", lw=1.8))
            x, y = to_xy(f["row"], f["col"])
            ax.annotate(f"{f['id']}: {f.get('presumption', 'row')} - {f['n_patches']} {'hollows' if f.get('patch_family') == 'hollow' else 'patches'}"
                        f" every {f['spacing_m']:.0f} m, chance {f['chance'] * 100:.0f}%",
                        (x, y), xytext=(10, -16), textcoords="offset points", color="white", fontsize=8, weight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="#b000b0", ec="none", alpha=.9))
            continue
        if f["type"] == "linear" and "ends_px" in f:
            (r1, c1), (r2, c2) = f["ends_px"]
            (x1, y1), (x2, y2) = to_xy(r1, c1), to_xy(r2, c2)
            ax.plot([x1, x2], [y1, y2], color=col, lw=1.5, ls=ls)
        x, y = to_xy(f["row"], f["col"])
        rad = max(14, 0.55 * f.get("diameter_m", 0) / mpp * scale)
        ax.add_patch(plt.Circle((x, y), rad, fill=False, ec=col, lw=1.6, ls=ls))
        ax.annotate(str(f["id"]), (x, y), xytext=(8, -12), textcoords="offset points", color="white", fontsize=7,
                    weight="bold", bbox=dict(boxstyle="round,pad=0.12", fc=col, ec="none", alpha=.85))


def render_images(ref_rgb, relief, feats, spot, mpp, out_dir, rgbs, warps):
    H, W = ref_rgb.shape[:2]
    figw = 16
    figh = figw * H / W + 0.8
    fig, ax = plt.subplots(figsize=(figw, figh), dpi=100)
    ax.imshow(ref_rgb)
    _draw(ax, feats, mpp=mpp)
    ax.plot(*spot, marker="+", ms=24, mew=3, color="black")
    ax.plot(*spot, marker="+", ms=20, mew=1.4, color="white")
    _legend(ax)
    ax.set_title("Candidates on screenshot 1 (solid = seen in 2+ images or confirmed, dashed = one image)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "candidates.png"))
    plt.close(fig)

    for k in range(1, len(rgbs)):
        if warps[k] is None:
            continue
        A = np.vstack([warps[k], [0, 0, 1]])
        to_xy = lambda r, c, A=A: tuple((A @ [c, r, 1])[:2])
        h, w = rgbs[k].shape[:2]
        fig, ax = plt.subplots(figsize=(figw, figw * h / w + 0.8), dpi=100)
        ax.imshow(rgbs[k])
        _draw(ax, feats, to_xy, mpp=mpp, scale=float(np.sqrt(abs(np.linalg.det(A[:2, :2])))))
        ax.set_title(f"The same candidates on screenshot {k + 1}")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"candidates_image{k + 1}.png"))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(figw, figh), dpi=100)
    nz = relief[relief != 0]
    lim = np.percentile(np.abs(nz), 98) if nz.size else 0.1
    ax.imshow(im.luminance(ref_rgb), cmap="gray", alpha=1)
    ax.imshow(np.ma.masked_equal(relief, 0), cmap="RdBu_r", vmin=-lim, vmax=lim, alpha=.8)
    _draw(ax, [f for f in feats if f["type"] in ("pit", "linear", "rectangle")], mpp=mpp)
    ax.set_title("Relief reconstructed from shading (red = higher, blue = lower; rough, relative)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "relief_from_shading.png"))
    plt.close(fig)


def _legend(ax):
    from matplotlib.lines import Line2D
    names = {"pit": "hollow (shading)", "snow_patch": "snow kept in hollow", "dark_spot": "dark spot / bush",
             "green_spot": "lusher growth", "light_spot": "light soil", "linear": "line (bank/lane)",
             "rectangle": "platform / outline", "soil_dark": "dark soil mark", "soil_light": "light soil mark",
             "crop_green": "green crop mark", "crop_parched": "parched crop mark", "forest_clearing": "clearing in wood",
             "canopy_anomaly": "different trees", "tree_row": "tree row", "row": "regular row of patches"}
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="none", mec=KIND_COLORS[k], mew=2, ms=10, label=v)
                       for k, v in names.items()], loc="lower left", fontsize=8, framealpha=.8)
