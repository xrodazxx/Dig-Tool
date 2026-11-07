"""Optional online step: download elevation + water for ONE spot, once. Everything after
that is offline. The best source for the spot is picked automatically and the next one
is tried if it fails or has no data there.

  USA        3dep        USGS 3DEP lidar DTM, 1 m where flown (most of the country)
  England    ea          Environment Agency lidar composite DTM, 1 m (~99 % of England)
  Finland    nls         National Land Survey 2 m DTM from lidar (free API key)
  far north  arcticdem   ArcticDEM 2 m surface model (Russia north of ~60 N, etc.)
  anywhere   glo30       Copernicus 30 m - landscape context only
"""
from __future__ import annotations

import json
import os

from .common import NoData

SOURCES = ("3dep", "ea", "nls", "arcticdem", "arcticdem10", "glo30")


def in_usa(lat, lon):
    return ((24 <= lat <= 49.5 and -125 <= lon <= -66.5) or (51 <= lat <= 71.5 and -170 <= lon <= -129)
            or (18.5 <= lat <= 22.5 and -161 <= lon <= -154))


def in_england(lat, lon):
    # generous box; Wales / Scotland parts simply return 'no data' and fall through
    return 49.85 <= lat <= 55.85 and -6.5 <= lon <= 1.9


def in_finland(lat, lon):
    return 59.6 <= lat <= 70.2 and 19.0 <= lon <= 31.7


def plan(lat, lon, have_nls_key=False) -> list[str]:
    order = []
    if in_usa(lat, lon):
        order.append("3dep")
    if in_england(lat, lon):
        order.append("ea")
    if in_finland(lat, lon) and have_nls_key:
        order.append("nls")
    if lat >= 59.5:
        order += ["arcticdem", "arcticdem10"]
    order.append("glo30")
    return order


def fetch_dem(source, lat, lon, radius_m, out_path, nls_key=None, log=print):
    if source == "3dep":
        from .arcgis import usgs_3dep
        return usgs_3dep(lat, lon, radius_m, out_path)
    if source == "ea":
        from .arcgis import ea_england
        return ea_england(lat, lon, radius_m, out_path)
    if source == "nls":
        from .nls import nls_finland
        return nls_finland(lat, lon, radius_m, out_path, key=nls_key, log=log)
    if source in ("arcticdem", "arcticdem10"):
        from .arctic import arcticdem
        return arcticdem(lat, lon, radius_m, out_path, res="10m" if source == "arcticdem10" else "2m")
    if source == "glo30":
        from .glo30 import glo30
        return glo30(lat, lon, radius_m, out_path)
    raise ValueError(f"unknown source {source}; choose from {', '.join(SOURCES)}")


def fetch_all(lat, lon, radius_m, out_dir, source="auto", water_radius_m=5000,
              nls_key=None, skip_dem=False, log=print):
    from .nls import api_key
    os.makedirs(out_dir, exist_ok=True)
    dem_path = os.path.join(out_dir, "dem.tif")
    key = api_key(nls_key)
    order = plan(lat, lon, bool(key)) if source == "auto" else [source]
    info = {"dem": None, "water": None, "attempts": []}
    if not skip_dem:
        if source == "auto" and in_finland(lat, lon) and not key:
            log("Finland: no NLS API key, so no 2 m lidar. Get a free key and pass --nls-key "
                "(see README). Falling back to coarser data.")
        for src in order:
            try:
                log(f"elevation: trying {src} ...")
                info["dem"] = fetch_dem(src, lat, lon, radius_m + 50, dem_path, key, log)
                info["dem"]["source"] = src
                info["attempts"].append({"source": src, "ok": True})
                if info["dem"].get("warning"):
                    log("  note: " + info["dem"]["warning"])
                break
            except NoData as ex:
                log(f"  {src}: {ex}")
                info["attempts"].append({"source": src, "ok": False, "why": str(ex)})
            except Exception as ex:  # noqa: BLE001 - report and try the next source
                log(f"  {src} failed: {ex}")
                info["attempts"].append({"source": src, "ok": False, "why": str(ex)})
    try:
        from .water import osm_water
        log("water: OpenStreetMap ...")
        info["water"] = osm_water(lat, lon, water_radius_m, os.path.join(out_dir, "water.geojson"))
    except Exception as ex:  # noqa: BLE001
        log(f"  water failed: {ex}")
    with open(os.path.join(out_dir, "sources.json"), "w") as fh:
        json.dump(info, fh, indent=1, default=str)
    return info
