"""Distance from the spot to water, from a local GeoJSON file (e.g. downloaded once
from OpenStreetMap by `sitecheck fetch`, or exported from any GIS). Pure Python."""
from __future__ import annotations

import json
import math


def _local_xy(lon, lat, lon0, lat0):
    return ((lon - lon0) * 111_320 * math.cos(math.radians(lat0)), (lat - lat0) * 110_574)


def _seg_dist(p, a, b):
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = 0 if L == 0 else max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / L))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _lines(geom):
    t, c = geom["type"], geom["coordinates"]
    if t == "LineString":
        yield c
    elif t in ("MultiLineString", "Polygon"):
        yield from c
    elif t == "MultiPolygon":
        for poly in c:
            yield from poly
    elif t == "Point":
        yield [c, c]


def nearest_water(geojson_path: str, lon: float, lat: float) -> dict | None:
    with open(geojson_path, encoding="utf-8") as fh:
        data = json.load(fh)
    best = None
    for feat in data.get("features", []):
        props = feat.get("properties") or {}
        for line in _lines(feat["geometry"]):
            pts = [_local_xy(x, y, lon, lat) for x, y, *_ in line]
            for a, b in zip(pts, pts[1:]):
                d = _seg_dist((0, 0), a, b)
                if best is None or d < best[0]:
                    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                    best = (d, props, math.degrees(math.atan2(mid[0], mid[1])) % 360)
    if best is None:
        return None
    d, props, bearing = best
    kind = props.get("waterway") or props.get("natural") or props.get("water") or "water"
    return {
        "distance_m": round(d),
        "kind": kind,
        "name": props.get("name"),
        "bearing_deg": round(bearing),
    }


def distance_band(d_m: float | None) -> str:
    if d_m is None:
        return "unknown"
    for lim, label in ((300, "<300 m"), (1000, "300 m-1 km"), (2000, "1-2 km"),
                       (5000, "2-5 km"), (10000, "5-10 km")):
        if d_m < lim:
            return label
    return ">10 km"
