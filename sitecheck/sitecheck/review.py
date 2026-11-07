"""Manual verification: confirm / reject candidates, add features the algorithm missed,
then re-score and redraw - all offline, from the files of a previous run."""
from __future__ import annotations

import json
import math
import os

import numpy as np

from . import score
from .pipeline import _public, _summary
from .report import build_prompt, render

ADDABLE = {"pit", "rectangle", "linear"}


def parse_add(spec: str, spot_lat: float, spot_lon: float) -> dict:
    """'pit:35,-20' (meters east,north of the spot) or 'pit@60.4981,24.9036' (lat,lon).
    Optional size after a second colon: 'pit:35,-20:6' (diameter or length, m)."""
    by_latlon = "@" in spec
    kind, _, rest = spec.partition("@" if by_latlon else ":")
    if kind not in ADDABLE:
        raise ValueError(f"type must be one of {sorted(ADDABLE)}")
    if by_latlon:
        if spot_lat is None:
            raise ValueError("this report has no coordinates; use pit:X,Y (metres from the spot)")
        latlon, _, size = rest.partition(":")
        lat, lon = map(float, latlon.split(","))
        x = (lon - spot_lon) * 111_320 * math.cos(math.radians(spot_lat))
        y = (lat - spot_lat) * 110_574
    else:
        xy, _, size = rest.partition(":")
        x, y = map(float, xy.split(","))
        if spot_lat is not None:
            lat = spot_lat + y / 110_574
            lon = spot_lon + x / (111_320 * math.cos(math.radians(spot_lat)))
        else:
            lat = lon = None
    f = {"type": kind, "x_m": round(x, 1), "y_m": round(y, 1), "lat": round(lat, 6), "lon": round(lon, 6),
         "distance_from_spot_m": round(math.hypot(x, y)), "bearing_deg": round(math.degrees(math.atan2(x, y)) % 360),
         "confidence": 1.0, "detected_by": "user", "user_status": "confirmed",
         "note": "marked by the user", "straightness": 1.0 if kind == "linear" else None}
    size = float(size) if size else None
    if kind == "pit":
        f["diameter_m"] = size or 5.0
    elif size:
        f["length_m"] = size
    return {k: v for k, v in f.items() if v is not None}  # lat/lon dropped when unknown


def review(out_dir: str, confirm=(), reject=(), add=(), weights_path=None, extra_features=(), unreview=()) -> dict:
    rp = os.path.join(out_dir, "report.json")
    with open(rp, encoding="utf-8") as fh:
        report = json.load(fh)
    with open(os.path.join(out_dir, "features_full.json"), encoding="utf-8") as fh:
        feats = json.load(fh)
    lay = np.load(os.path.join(out_dir, "layers.npz"))
    rc, cell = tuple(int(v) for v in lay["spot_rc"]), float(lay["cell"])

    by_id = {f["id"]: f for f in feats}
    for ids, status in ((confirm, "confirmed"), (reject, "rejected")):
        for i in ids:
            if i not in by_id:
                raise ValueError(f"no feature #{i}")
            by_id[i]["user_status"] = status

    for i in unreview:
        if i in by_id:
            by_id[i]["user_status"] = "unreviewed"
    next_id = max(by_id, default=0) + 1
    for f in extra_features:          # e.g. a guided row from the interface
        f = dict(f)
        f["id"] = next_id
        next_id += 1
        f.setdefault("user_status", "confirmed")
        feats.append(f)
    for spec in add:
        f = parse_add(spec, report["spot"]["lat"], report["spot"]["lon"])
        f["id"] = next_id
        next_id += 1
        f["row"], f["col"] = rc[0] - f["y_m"] / cell, rc[1] + f["x_m"] / cell
        if f["type"] == "linear":   # a short marker line, user gives only the centre
            half = f.get("length_m", 20) / 2 / cell
            f["ends_px"] = [[f["row"], f["col"] - half], [f["row"], f["col"] + half]]
        if f["type"] == "rectangle":
            s = f.get("length_m", 10) / 2 / cell
            f["corners_px"] = [[f["col"] + dx * s, f["row"] + dy * s]
                               for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        feats.append(f)

    ctx, spot = report["context"], report["terrain"]
    if "photo" in lay:
        from .image_mode import _public_img
        report["features"] = [_public_img(f) for f in feats]
    else:
        report["features"] = [_public(f) for f in feats]
    report["feature_summary"] = _summary(feats)
    report["score"] = score.compute(spot, report["water"], feats, ctx, score.load_weights(weights_path))
    report["review"] = {"confirmed": sorted(f["id"] for f in feats if f["user_status"] == "confirmed"),
                        "rejected": sorted(f["id"] for f in feats if f["user_status"] == "rejected"),
                        "user_added": sorted(f["id"] for f in feats if f.get("detected_by") == "user")}

    _d = lambda o: o.item() if hasattr(o, "item") else str(o)
    with open(rp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, default=_d)
    with open(os.path.join(out_dir, "features_full.json"), "w", encoding="utf-8") as fh:
        json.dump(feats, fh, indent=1, default=_d)
    with open(os.path.join(out_dir, "prompt.md"), "w", encoding="utf-8") as fh:
        fh.write(build_prompt(report))
    if "photo" in lay:      # screenshot mode
        from .image_mode import render_images
        photo = lay["photo"].astype("float32") / 255
        render_images(photo, lay["z"], feats, (rc[1], rc[0]), cell, out_dir, [photo], [None])
        return report
    layers = {k: lay[k] for k in ("z", "lrm", "slope", "hillshade")}
    layers["svf"] = lay["svf"] if "svf" in lay else None
    if "overlay" in lay:
        layers["overlay"] = lay["overlay"]
    render(layers, rc, cell, feats, out_dir)
    return report
