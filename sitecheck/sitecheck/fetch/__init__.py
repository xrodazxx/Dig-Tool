"""Optional online step: download data for one spot ONCE, then everything runs offline.

Sources (all free, no account needed):
  - USGS 3DEP 1 m lidar DEM (USA)                        -> source "3dep"
  - Copernicus GLO-30 (worldwide, 30 m, landscape only)  -> source "glo30"
  - OpenStreetMap rivers, streams, lakes (Overpass API)  -> water.geojson
UK / Finland / other national lidar: download the tile from the national portal and
pass it with --dem (see README).
"""
from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np
import requests

from ..raster import crop_around, load_dem, save_geotiff_wgs84

UA = {"User-Agent": "sitecheck/0.1 (offline terrain analysis tool)"}
OVERPASS = "https://overpass-api.de/api/interpreter"
TDEP = ("https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/"
        "ImageServer/exportImage")
GLO30 = ("https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif")


def _bbox(lat, lon, r):
    dlat = r / 110_574
    dlon = r / (111_320 * math.cos(math.radians(lat)))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def in_usa(lat, lon):
    return (24 <= lat <= 49.5 and -125 <= lon <= -66.5) or (51 <= lat <= 71.5 and -170 <= lon <= -129) \
        or (18.5 <= lat <= 22.5 and -161 <= lon <= -154)


# --------------------------------------------------------------------------- USGS 3DEP

def usgs_3dep(lat, lon, radius_m, out_path, res_m=1.0, timeout=120):
    """Clip from the 3DEP dynamic elevation service, delivered in EPSG:4326."""
    w, s, e, n = _bbox(lat, lon, radius_m)
    size = int(min(4000, 2 * radius_m / res_m))
    params = {"bbox": f"{w},{s},{e},{n}", "bboxSR": 4326, "imageSR": 4326,
              "size": f"{size},{size}", "format": "tiff", "pixelType": "F32",
              "noData": -9999, "interpolation": "RSP_BilinearInterpolation", "f": "image"}
    r = requests.get(TDEP, params=params, headers=UA, timeout=timeout)
    r.raise_for_status()
    if not r.content.startswith((b"II*\x00", b"MM\x00*")):
        raise RuntimeError("3DEP did not return a GeoTIFF: " + r.text[:300])
    with open(out_path, "wb") as fh:
        fh.write(r.content)
    dem = load_dem(out_path)
    if np.isnan(dem.z).all() or np.nanmax(dem.z) < -1000:
        raise RuntimeError("3DEP has no data here")
    return {"file": out_path, "label": f"USGS 3DEP (~{res_m:g} m, bare earth)", "cell_m": res_m}


# --------------------------------------------------------------------------- Copernicus GLO-30

def _glo30_name(lat_i, lon_i):
    ns = f"{'N' if lat_i >= 0 else 'S'}{abs(lat_i):02d}_00"
    ew = f"{'E' if lon_i >= 0 else 'W'}{abs(lon_i):03d}_00"
    return f"Copernicus_DSM_COG_10_{ns}_{ew}_DEM"


def glo30(lat, lon, radius_m, out_path, timeout=300):
    """One 1x1 degree tile (spots near a tile edge get a clipped window)."""
    name = _glo30_name(math.floor(lat), math.floor(lon))
    url = GLO30.format(n=name)
    try:   # rasterio reads only the needed part of the cloud-optimized GeoTIFF
        import rasterio
        from rasterio.windows import from_bounds
        w, s, e, n = _bbox(lat, lon, radius_m + 200)
        with rasterio.open(url) as ds:
            win = from_bounds(w, s, e, n, ds.transform).round_offsets().round_lengths()
            z = ds.read(1, window=win).astype("float32")
            t = ds.window_transform(win)
        save_geotiff_wgs84(z, t.c, t.f, t.a, -t.e, out_path)
    except ImportError:  # no rasterio: download the whole tile (~20-50 MB) and cut it
        tmp = os.path.join(tempfile.gettempdir(), name + ".tif")
        if not os.path.exists(tmp):
            with requests.get(url, headers=UA, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
        sub, _ = crop_around(load_dem(tmp), lon, lat, radius_m + 200)
        save_geotiff_wgs84(sub.z, sub.x0, sub.y0, sub.dx, sub.dy, out_path)
    return {"file": out_path, "label": "Copernicus GLO-30 (30 m surface model)", "cell_m": 30}


# --------------------------------------------------------------------------- OSM water

def osm_water(lat, lon, radius_m, out_path, timeout=90):
    w, s, e, n = _bbox(lat, lon, radius_m)
    b = f"{s},{w},{n},{e}"
    q = (f'[out:json][timeout:{timeout}];('
         f'way["waterway"~"river|stream|brook|canal|ditch|drain"]({b});'
         f'way["natural"~"water|spring|wetland"]({b});'
         f'relation["natural"="water"]({b});'
         f'node["natural"="spring"]({b}););out geom;')
    r = requests.post(OVERPASS, data={"data": q}, headers=UA, timeout=timeout + 30)
    r.raise_for_status()
    feats = []
    for el in r.json().get("elements", []):
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
                   "attribution": "© OpenStreetMap contributors (ODbL)"}, fh)
    return {"file": out_path, "count": len(feats)}


# --------------------------------------------------------------------------- one call

def fetch_all(lat, lon, radius_m, out_dir, source="auto", water_radius_m=5000, log=print):
    os.makedirs(out_dir, exist_ok=True)
    dem_path = os.path.join(out_dir, "dem.tif")
    info = {"dem": None, "water": None}
    order = {"auto": (["3dep", "glo30"] if in_usa(lat, lon) else ["glo30"]),
             "3dep": ["3dep"], "glo30": ["glo30"]}[source]
    for src in order:
        try:
            log(f"downloading elevation: {src} ...")
            info["dem"] = (usgs_3dep if src == "3dep" else glo30)(lat, lon, radius_m + 50, dem_path)
            break
        except Exception as ex:  # noqa: BLE001 - report and try the next source
            log(f"  {src} failed: {ex}")
    try:
        log("downloading water from OpenStreetMap ...")
        info["water"] = osm_water(lat, lon, water_radius_m, os.path.join(out_dir, "water.geojson"))
    except Exception as ex:  # noqa: BLE001
        log(f"  water failed: {ex}")
    with open(os.path.join(out_dir, "sources.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    return info
