"""Copernicus GLO-30: worldwide 30 m surface model. Landscape context only."""
from __future__ import annotations

import math
import os
import tempfile

from ..raster import crop_around, load_dem, mosaic, save_geotiff
from .common import UA, bbox

URL = "https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif"


def tile_name(lat_i, lon_i):
    ns = f"{'N' if lat_i >= 0 else 'S'}{abs(lat_i):02d}_00"
    ew = f"{'E' if lon_i >= 0 else 'W'}{abs(lon_i):03d}_00"
    return f"Copernicus_DSM_COG_10_{ns}_{ew}_DEM"


def tiles_for(w, s, e, n):
    return [tile_name(la, lo) for la in range(math.floor(s), math.floor(n) + 1)
            for lo in range(math.floor(w), math.floor(e) + 1)]


def glo30(lat, lon, radius_m, out_path):
    """Handles spots near tile edges by joining up to 4 tiles."""
    w, s, e, n = bbox(lat, lon, radius_m + 200)
    names = tiles_for(w, s, e, n)
    dems = []
    try:
        import rasterio
        from rasterio.windows import from_bounds
        from ..raster import Dem
        for nm in names:
            try:
                with rasterio.open(URL.format(n=nm)) as ds:
                    win = from_bounds(w, s, e, n, ds.transform).intersection(
                        rasterio.windows.Window(0, 0, ds.width, ds.height))
                    win = win.round_offsets().round_lengths()
                    z = ds.read(1, window=win).astype("float32")
                    t = ds.window_transform(win)
                    dems.append(Dem(z, t.c, t.f, t.a, -t.e, 4326))
            except Exception:  # noqa: BLE001 - ocean tiles don't exist
                continue
    except ImportError:
        import requests
        for nm in names:
            tmp = os.path.join(tempfile.gettempdir(), nm + ".tif")
            if not os.path.exists(tmp):
                r = requests.get(URL.format(n=nm), headers=UA, stream=True, timeout=300)
                if r.status_code != 200:
                    continue
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            dems.append(load_dem(tmp))
    if not dems:
        raise RuntimeError("no Copernicus GLO-30 tile for this spot")
    sub, _ = crop_around(mosaic(dems), lon, lat, radius_m + 200)
    save_geotiff(sub.z, sub.x0, sub.y0, sub.dx, sub.dy, out_path, 4326)
    return {"file": out_path, "label": "Copernicus GLO-30 (30 m surface model)", "cell_m": 30.0,
            "is_dsm": True, "licence": "(c) DLR e.V. 2010-2014 and (c) Airbus 2014-2018, provided under COPERNICUS by the EU and ESA"}
