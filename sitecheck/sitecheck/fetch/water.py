"""Rivers, streams, lakes, springs and wetlands from OpenStreetMap (Overpass API)."""
from __future__ import annotations

import json

from .common import bbox, post

OVERPASS = ["https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.private.coffee/api/interpreter"]


def osm_water(lat, lon, radius_m, out_path, timeout=90):
    w, s, e, n = bbox(lat, lon, radius_m)
    b = f"{s},{w},{n},{e}"
    q = (f'[out:json][timeout:{timeout}];('
         f'way["waterway"~"river|stream|brook|canal|ditch|drain|spring"]({b});'
         f'way["natural"~"water|wetland"]({b});'
         f'relation["natural"="water"]({b});'
         f'node["natural"="spring"]({b}););out geom;')
    last = None
    for url in OVERPASS:
        try:
            r = post(url, data={"data": q}, timeout=timeout + 30)
            if r.status_code == 200:
                data = r.json()
                break
            last = f"HTTP {r.status_code}"
        except Exception as ex:  # noqa: BLE001 - try the next mirror
            last = ex
    else:
        raise RuntimeError(f"OpenStreetMap water download failed: {last}")
    feats = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        if el["type"] == "node":
            geom = {"type": "Point", "coordinates": [el["lon"], el["lat"]]}
        elif el["type"] == "way" and "geometry" in el:
            geom = {"type": "LineString", "coordinates": [[p["lon"], p["lat"]] for p in el["geometry"]]}
        elif el["type"] == "relation":
            lines = [[[p["lon"], p["lat"]] for p in m["geometry"]]
                     for m in el.get("members", []) if "geometry" in m]
            if not lines:
                continue
            geom = {"type": "MultiLineString", "coordinates": lines}
        else:
            continue
        feats.append({"type": "Feature", "geometry": geom,
                      "properties": {k: tags[k] for k in ("waterway", "natural", "water", "name") if k in tags}})
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": feats,
                   "attribution": "(c) OpenStreetMap contributors (ODbL)"}, fh)
    return {"file": out_path, "count": len(feats)}
