"""ArcticDEM (Polar Geospatial Center): 2 m surface model from satellite stereo, covering
land north of ~60 N - the north of Russia (Karelia, Arkhangelsk, Komi, Siberia...), Finland,
Scandinavia, Alaska, Canada.

It is a SURFACE model (trees and roofs included), so the pipeline runs the ground filter
on it. Heights are ellipsoidal - fine for relief analysis.

Access: PGC STAC API -> Cloud-Optimized GeoTIFFs on AWS; only the window is read.
Needs rasterio. Terms: PGC states 2 m products are for education/research users and
10 m / 32 m are open to all; the AWS copy is listed as CC BY 4.0. Check what applies to you;
--source arcticdem10 uses the 10 m version.
"""
from __future__ import annotations

from ..raster import interpolation_check, save_geotiff
from .common import NoData, bbox, get, post

STAC = "https://stac.pgc.umn.edu/api/v1"


def _collection(res):
    cols = get(STAC + "/collections", timeout=60).json().get("collections", [])
    ids = [c["id"] for c in cols]
    want = [i for i in ids if "arcticdem" in i and "mosaic" in i and i.endswith(res)]
    if not want:
        want = [i for i in ids if "arcticdem" in i and "mosaic" in i and res in i]
    if not want:
        raise RuntimeError("ArcticDEM mosaic collection not found; available: " + ", ".join(ids[:15]))
    return sorted(want)[-1]   # newest version sorts last (v3.0 < v4.1)


def arcticdem(lat, lon, radius_m, out_path, res="2m"):
    try:
        import numpy as np
        import rasterio
        from rasterio.warp import transform_bounds
        from rasterio.windows import from_bounds
    except ImportError:
        raise RuntimeError("ArcticDEM needs rasterio: pip install rasterio")
    if lat < 50:
        raise NoData("ArcticDEM covers only the far north (~60 N and above)")
    w, s, e, n = bbox(lat, lon, radius_m + 30)
    col = _collection(res)
    r = post(STAC + "/search", json={"collections": [col], "bbox": [w, s, e, n], "limit": 20})
    items = r.json().get("features", [])
    if not items:
        raise NoData("no ArcticDEM tile here")
    parts = []
    for it in items:
        href = (it.get("assets", {}).get("dem") or {}).get("href")
        if not href:
            continue
        with rasterio.open(href) as ds:
            bx = transform_bounds("EPSG:4326", ds.crs, w, s, e, n)
            win = from_bounds(*bx, ds.transform).round_offsets().round_lengths()
            z = ds.read(1, window=win, boundless=True, fill_value=ds.nodata or -9999).astype("float32")
            if ds.nodata is not None:
                z[z == ds.nodata] = np.nan
            t = ds.window_transform(win)
            parts.append((z, t, ds.crs.to_epsg()))
    if not parts:
        raise NoData("ArcticDEM tiles have no DEM asset")
    z, t, epsg = parts[0]
    for z2, _, _ in parts[1:]:          # same grid (one projection, tiles share the grid)
        if z2.shape == z.shape:
            z = np.where(np.isnan(z), z2, z)
    if np.isnan(z).mean() > 0.5:
        raise NoData("ArcticDEM has a gap at this spot")
    save_geotiff(z, t.c, t.f, t.a, -t.e, out_path, epsg)
    return {"file": out_path, "label": f"ArcticDEM mosaic {res} (surface model)", "cell_m": float(res[:-1]),
            "is_dsm": True, "interpolation": interpolation_check(z),
            "licence": "PGC / NSF - see PGC acknowledgement policy"}
