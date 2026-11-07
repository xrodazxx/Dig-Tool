"""ArcGIS ImageServer 'exportImage' sources: USGS 3DEP (USA) and Environment Agency
lidar (England). Both are clipped on the server, so only the window is downloaded."""
from __future__ import annotations

import json

import numpy as np

from ..raster import interpolation_check, load_dem
from .common import NoData, bbox, get, is_tiff

TDEP = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer"

# Environment Agency composite DTM (England, ~99% coverage, 1 m, Open Government Licence).
# Newer composites are published under new names; we try the newest first.
EA_SERVICES = [
    "https://environment.data.gov.uk/image/rest/services/SURVEY/LIDAR_Composite_1m_DTM_{y}_Elevation/ImageServer"
    .format(y=y) for y in (2025, 2024, 2023, 2022)
]


def export_image(service, lat, lon, radius_m, out_path, res_m=1.0, timeout=180):
    """Ask the server for a square window in EPSG:4326 with ~res_m pixels.

    adjustAspectRatio=false matters: by default the server reshapes the box to square
    *degrees*, which stretches the window north-south away from the equator (the bug
    in v0.1)."""
    w, s, e, n = bbox(lat, lon, radius_m)
    size = int(min(4000, round(2 * radius_m / res_m)))
    params = {"bbox": f"{w},{s},{e},{n}", "bboxSR": 4326, "imageSR": 4326,
              "size": f"{size},{size}", "adjustAspectRatio": "false",
              "format": "tiff", "pixelType": "F32", "noData": -9999,
              "noDataInterpretation": "esriNoDataMatchAny",
              "interpolation": "RSP_BilinearInterpolation", "f": "image"}
    r = get(service.rstrip("/") + "/exportImage", params=params, timeout=timeout)
    if r.status_code != 200 or not is_tiff(r.content):
        raise RuntimeError(f"{service.split('/services/')[-1]}: no GeoTIFF returned "
                           f"(HTTP {r.status_code}): {r.text[:200]!r}")
    with open(out_path, "wb") as fh:
        fh.write(r.content)
    dem = load_dem(out_path)
    z = dem.z
    bad = np.isnan(z) | (z < -1000) | (z > 9000)
    if bad.mean() > 0.5:
        raise NoData(f"{service.split('/services/')[-1]}: no elevation data at this spot")
    return dem


def _catalog_resolution(service, lat, lon):
    """Best effort: ask which source rasters cover the point and their pixel size."""
    try:
        r = get(service + "/identify", retries=1, timeout=30, params={
            "geometry": json.dumps({"x": lon, "y": lat, "spatialReference": {"wkid": 4326}}),
            "geometryType": "esriGeometryPoint", "returnCatalogItems": "true",
            "returnGeometry": "false", "f": "json"})
        items = (r.json().get("catalogItems") or {}).get("features") or []
    except Exception:  # noqa: BLE001
        return None
    found = []
    for it in items:
        a = it.get("attributes", {})
        ps = next((a[k] for k in ("LowPS", "MinPS", "PixelSize") if isinstance(a.get(k), (int, float))), None)
        name = next((a[k] for k in ("ProductName", "Name", "name") if a.get(k)), None)
        if ps or name:
            found.append({"pixel_size": ps, "name": name})
    return found or None


def usgs_3dep(lat, lon, radius_m, out_path):
    dem = export_image(TDEP, lat, lon, radius_m, out_path, res_m=1.0)
    chk = interpolation_check(dem.z)
    cat = _catalog_resolution(TDEP, lat, lon)
    best = min((c["pixel_size"] for c in cat or [] if c.get("pixel_size")), default=None)
    label = "USGS 3DEP (bare earth)"
    note = None
    if chk["linear_share"] and chk["linear_share"] > 0.4:
        note = ("3DEP has no 1 m lidar here yet: the service filled the window from a coarser "
                f"DEM (looks ~{max(3, chk['factor']):.0f}x coarser). Pits and foundations will not show.")
        label += ", coarse source resampled"
    elif best:
        label += f", source ~{best:g} (catalog units)"
    else:
        label += ", 1 m"
    return {"file": out_path, "label": label, "cell_m": 1.0, "catalog": cat,
            "interpolation": chk, "warning": note}


def ea_england(lat, lon, radius_m, out_path):
    errors = []
    for svc in EA_SERVICES:
        try:
            dem = export_image(svc, lat, lon, radius_m, out_path, res_m=1.0)
        except NoData:
            raise
        except Exception as ex:  # noqa: BLE001 - try older composite names
            errors.append(str(ex))
            continue
        year = svc.split("_DTM_")[1].split("_")[0]
        return {"file": out_path, "label": f"Environment Agency lidar composite DTM {year}, 1 m",
                "cell_m": 1.0, "interpolation": interpolation_check(dem.z),
                "licence": "Open Government Licence; (c) Environment Agency"}
    raise RuntimeError("Environment Agency service unavailable: " + " | ".join(errors[-2:]))
