"""The whole offline analysis: DEM + context -> report.json, prompt.md, images."""
from __future__ import annotations

import datetime as dt
import math
import os
import shutil

import numpy as np
from scipy import ndimage as ndi

from . import __version__, context as ctxmod, features as feat, raster, score, terrain, water as watermod
from .report import write_outputs

MICRO_RELIEF_MAX_CELL_M = 3.0   # coarser than this, pits and walls are invisible


def analyze(dem_path: str, lat: float, lon: float, radius_m: float = 300.0,
            water_path: str | None = None, context_path: str | None = None,
            image_path: str | None = None, weights_path: str | None = None,
            out_dir: str = "sitecheck_out", dem_label: str | None = None,
            ground_filter: str = "auto", is_dsm: bool | None = None,
            overlay_path: str | None = None, fetch_info: dict | None = None) -> dict:
    """ground_filter: 'auto' (on for surface models: ArcticDEM, GLO-30, drone DSM),
    'on' or 'off'."""
    os.makedirs(out_dir, exist_ok=True)
    full = raster.load_dem(dem_path)
    dem, rc = raster.crop_around(full, lon, lat, radius_m)
    csx, csy = dem.cell_size_m()
    cell = (csx + csy) / 2
    z = terrain.fill_nodata(dem.z.astype("float64"))
    nodata_share = float(np.isnan(dem.z).mean())
    warnings = []

    # --- is the resolution real? ----------------------------------------------------------
    interp = raster.interpolation_check(dem.z)
    if interp["linear_share"] is not None and interp["linear_share"] > 0.4:
        warnings.append(
            f"The DEM looks resampled from a much coarser source ({interp['linear_share']:.0%} of cells "
            "are perfectly linear between neighbours). Its real resolution is several times worse "
            f"than {cell:.1f} m, so small features cannot be seen. Look for real lidar for this area.")

    # --- surface model -> terrain model -------------------------------------------------------
    if is_dsm is None:
        lbl = (dem_label or "").lower()
        is_dsm = any(k in lbl for k in ("glo-30", "glo30", "copernicus", "arcticdem", "surface", "dsm"))
    gf_info = None
    if ground_filter == "on" or (ground_filter == "auto" and is_dsm and cell <= 10):
        from .ground import dsm_to_dtm
        z, gf_info = dsm_to_dtm(z, cell)
        z = z.astype("float64")
        if gf_info.get("warning"):
            warnings.append(gf_info["warning"])
        warnings.append(f"Surface model: trees/buildings were removed by a filter "
                        f"({gf_info['object_share']:.0%} of the area). Under dense canopy the ground is guessed.")
    elif is_dsm:
        warnings.append("Surface model: tree and building heights are included.")

    # --- landscape-scale terrain ------------------------------------------------------
    z_smooth = ndi.gaussian_filter(z, max(1.0, 5.0 / cell))   # slope of the ground, not of the noise
    slope, aspect = terrain.slope_aspect(z_smooth, csx, csy)
    tpi_s = terrain.tpi(z, cell, 50)
    tpi_l = terrain.tpi(z, cell, min(250, radius_m * 0.8))
    spot = terrain.spot_metrics(z, slope, aspect, tpi_s, tpi_l, rc, cell)

    # --- micro-relief ---------------------------------------------------------------------
    lrm = terrain.local_relief(z, cell)
    if cell <= MICRO_RELIEF_MAX_CELL_M:
        features = feat.detect_all(z, lrm, cell, aspect, slope)
        svf = terrain.sky_view_factor(z, cell, radius_m=10)
    else:
        features, svf = [], None
        warnings.append(f"DEM cell size is {cell:.1f} m: too coarse to see pits, foundations or "
                        "ditches (need <= 3 m, ideally 1 m lidar). Only landscape metrics were computed.")
    if nodata_share > 0.05:
        warnings.append(f"{nodata_share:.0%} of the window has no elevation data.")

    for f in features:
        _georef_feature(f, dem, rc, csx, csy)

    # --- water ------------------------------------------------------------------------------
    ctx = ctxmod.load(context_path)
    water = {"distance_m": None, "source": None}
    if water_path and os.path.exists(water_path):
        nw = watermod.nearest_water(water_path, lon, lat)
        if nw:
            water.update(nw, source=f"computed:{os.path.basename(water_path)}")
    if ctx["water"].get("user_distance_m") is not None:
        water["computed_distance_m"] = water.get("distance_m")
        water.update(distance_m=ctx["water"]["user_distance_m"], source="user",
                     kind=ctx["water"].get("water_kind") or water.get("kind"))
    water["band"] = watermod.distance_band(water["distance_m"])

    sc = score.compute(spot, water, features, ctx, score.load_weights(weights_path))

    report = {
        "schema": "sitecheck-report/1",
        "generator": f"sitecheck {__version__}",
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "spot": {"lat": round(lat, 6), "lon": round(lon, 6), "analysis_radius_m": radius_m},
        "data": {
            "dem": {"file": os.path.basename(dem_path), "label": dem_label or "user-supplied DEM",
                    "epsg": dem.epsg, "cell_size_m": round(cell, 2), "window_px": list(dem.z.shape),
                    "nodata_share": round(nodata_share, 3), "surface_model": bool(is_dsm),
                    "ground_filter": gf_info, "interpolation_check": interp},
            "download": fetch_info,
            "overlay": os.path.basename(overlay_path) if overlay_path else None,
            "water": os.path.basename(water_path) if water_path else None,
            "user_image": os.path.basename(image_path) if image_path else None,
        },
        "context": ctx,
        "terrain": {"_source": "computed from DEM", **spot,
                    "micro_relief_noise_m": round(feat.noise_level(lrm), 3)},
        "water": water,
        "features": [_public(f) for f in features],
        "feature_summary": _summary(features),
        "score": sc,
        "warnings": warnings,
        "how_to_read": {
            "user_status": "unreviewed = found by the algorithm only; confirmed / rejected = checked by the user",
            "confidence": "0-1, how clean the shape is - not how likely it is to be archaeology",
            "x_m / y_m": "meters east / north of the spot",
        },
    }
    if image_path:
        shutil.copy(image_path, os.path.join(out_dir, os.path.basename(image_path)))
    layers = {"z": z, "lrm": lrm, "slope": slope, "svf": svf,
              "hillshade": terrain.multi_hillshade(z, csx, csy)}
    if overlay_path:
        from .overlay import load_rgb, sample_onto
        img, ref = load_rgb(overlay_path)
        layers["overlay"] = sample_onto(dem, img, ref)
        if layers["overlay"][..., 3].max() == 0:
            warnings.append("The overlay map does not cover the analysis window (check its georeference).")
            layers.pop("overlay")
    write_outputs(report, layers, rc, cell, features, out_dir)
    return report


def _georef_feature(f, dem, rc, csx, csy):
    lon, lat = dem.pixel_to_lonlat(f["row"], f["col"])
    f["lat"], f["lon"] = round(lat, 6), round(lon, 6)
    f["x_m"] = round((f["col"] - rc[1]) * csx, 1)
    f["y_m"] = round((rc[0] - f["row"]) * csy, 1)
    f["distance_from_spot_m"] = round(math.hypot(f["x_m"], f["y_m"]), 0)
    f["bearing_deg"] = round(math.degrees(math.atan2(f["x_m"], f["y_m"])) % 360, 0)


_PUBLIC = ["id", "type", "relief", "lat", "lon", "x_m", "y_m", "distance_from_spot_m", "bearing_deg",
           "diameter_m", "depth_m", "volume_m3", "roundness", "length_m", "width_m", "mean_width_m",
           "orientation_deg", "shape", "rectangularity", "perimeter_support", "straightness", "inside_feature",
           "confidence", "detected_by", "user_status", "note"]


def _public(f):
    return {k: f[k] for k in _PUBLIC if k in f}


def _summary(features):
    live = [f for f in features if f.get("user_status") != "rejected"]
    s = {"pits": sum(f["type"] == "pit" and not f.get("inside_feature") for f in live),
         "rectangles": sum(f["type"] == "rectangle" for f in live),
         "straight_linear": sum(f["type"] == "linear" and f.get("straightness", 0) >= 0.9 for f in live),
         "other_linear": sum(f["type"] == "linear" and f.get("straightness", 0) < 0.9 for f in live)}
    within = [f for f in live if f.get("distance_from_spot_m", 1e9) <= 100]
    s["within_100m_of_spot"] = len(within)
    return s
