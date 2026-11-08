"""Command line.

  sitecheck run     --point LAT,LON                 download (once) + analyze, one command
  sitecheck analyze --point LAT,LON --dem FILE      fully offline, your own DEM / lidar
  sitecheck ask     -o context.json                 the questionnaire
  sitecheck review  OUT_DIR --confirm 1,3 --reject 7 --add pit:35,-20
  sitecheck fetch   --point LAT,LON -o data/        download only
  sitecheck grid    scan.las --epsg 32637 -o dem.tif   point cloud (drone/phone) -> DEM
  sitecheck image   shot1.jpg [shot2.jpg ...] --point LAT,LON --known X1,Y1,X2,Y2,M
                                                    no elevation data: satellite screenshots

Sources (--source): auto | 3dep (USA) | ea (England) | nls (Finland, free key) |
                    arcticdem / arcticdem10 (far north incl. Russia) | glo30 (world, 30 m)
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _point(s):
    try:
        lat, lon = (float(v) for v in s.replace(" ", "").split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("use LAT,LON, e.g. 60.4981,24.9036")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise argparse.ArgumentTypeError("latitude/longitude out of range")
    return lat, lon


def _ids(s):
    return [int(v) for v in s.split(",") if v.strip()] if s else []


def _print_summary(rep, out_dir):
    sc, t, fs = rep["score"], rep["terrain"], rep["feature_summary"]
    if "dem" in rep["data"]:
        print(f"\nSpot {rep['spot']['lat']}, {rep['spot']['lon']}  |  DEM cell {rep['data']['dem']['cell_size_m']} m")
        print(f"Terrain: {t['landform']}, slope {t['slope_deg']} deg, faces {t['aspect']}, "
              f"elevation {t['elevation_m']} m")
    else:
        d = rep["data"]
        print(f"\nSpot {rep['spot']['lat']}, {rep['spot']['lon']}  |  screenshots: {len(d['images'])}, "
              f"{d['metres_per_pixel']} m/px{'' if d['scale_known'] else ' (ASSUMED)'}")
        for i in d["per_image"]:
            print(f"  {i['image']}: sun from {i['sun_azimuth_used']} deg ({i['sun_source']}), "
                  f"snow {i['snow_share']:.0%}, forest/field masked {i['forest_share']:.0%}")
    w = rep["water"]
    print(f"Water: {w['band']}" + (f" ({w.get('kind')}, {w.get('name') or 'unnamed'})" if w.get("distance_m") is not None else ""))
    print(f"Candidates: {fs['pits']} pits, {fs['rectangles']} rectangles, {fs['straight_linear']} straight "
          f"+ {fs['other_linear']} other linear; {fs['within_100m_of_spot']} within 100 m")
    for f in rep["features"]:
        size = f.get("diameter_m") or f.get("length_m")
        extra = ("+".join(f["signals"]) + f" img{f['seen_in_images']}") if "signals" in f else f.get("relief", "")
        print(f"  #{f['id']:<3} {f['type']:<9} {extra:<7} {size} m  "
              f"{f['distance_from_spot_m']:.0f} m @ {f['bearing_deg']:.0f} deg  conf {f['confidence']}  "
              f"[{f['user_status']}]  {f['note'][:60]}")
    print(f"\nEvidence score: {sc['points']}/100 ({sc['band']}) - heuristic, not a probability")
    for c in sc["contributions"]:
        print(f"  {c['points']:+6.1f}  {c['factor']:<18} {c['why']}")
    if sc["unknown_inputs"]:
        print("  unknown (0 points): " + ", ".join(sc["unknown_inputs"]))
    for wmsg in rep["warnings"]:
        print("WARNING: " + wmsg)
    print(f"\nOpen {os.path.join(out_dir, 'candidates.png')}, check the numbered features, then e.g.\n"
          f"  sitecheck review {out_dir} --confirm 1,2 --reject 5\n"
          f"For a second opinion: paste {os.path.join(out_dir, 'prompt.md')} into any AI chat "
          f"and attach the images listed at the top of it.")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sitecheck", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--point", type=_point, required=True, help="LAT,LON of your spot")
        p.add_argument("--radius", type=float, default=300, help="analysis radius, m (default 300)")
        p.add_argument("--context", help="context.json from `sitecheck ask`")
        p.add_argument("--image", help="your own screenshot/photo to pass along")
        p.add_argument("--weights", help="custom weights.json")
        p.add_argument("-o", "--out", default=None, help="output folder")
        p.add_argument("--ground-filter", choices=["auto", "on", "off"], default="auto",
                       help="remove trees/buildings from surface models (auto: on for DSMs)")
        p.add_argument("--overlay", help="georeferenced old map (GeoTIFF) to draw under the candidates")

    from .fetch import SOURCES
    p = sub.add_parser("run", help="download data once, then analyze")
    common(p)
    p.add_argument("--source", choices=("auto",) + SOURCES, default="auto")
    p.add_argument("--nls-key", help="Finland: free NLS API key (or env NLS_API_KEY)")
    p.add_argument("--dem", help="use this DEM instead of downloading one (water is still downloaded)")

    p = sub.add_parser("analyze", help="offline analysis of a DEM you already have")
    common(p)
    p.add_argument("--dem", required=True)
    p.add_argument("--water", help="GeoJSON with rivers/lakes")
    p.add_argument("--dem-label", help="describe the DEM, e.g. 'NLS lidar 2 m'")
    p.add_argument("--dsm", action="store_true", help="the DEM is a surface model (trees/roofs included)")

    p = sub.add_parser("fetch", help="download only")
    p.add_argument("--point", type=_point, required=True)
    p.add_argument("--radius", type=float, default=300)
    p.add_argument("--source", choices=("auto",) + SOURCES, default="auto")
    p.add_argument("--nls-key")
    p.add_argument("-o", "--out", default="sitecheck_data")

    p = sub.add_parser("grid", help="point cloud (LAS/LAZ/PLY/XYZ) -> DEM GeoTIFF")
    p.add_argument("points")
    p.add_argument("-o", "--out", default="dem_from_points.tif")
    p.add_argument("--cell", type=float, default=0.5, help="grid cell, m (default 0.5)")
    p.add_argument("--epsg", type=int, help="CRS of projected coordinates, e.g. 32637")
    p.add_argument("--anchor", type=_point, help="LAT,LON of the scan origin (local phone scans)")
    p.add_argument("--center", action="store_true", help="put the scan centre (not origin) at --anchor")
    p.add_argument("--no-ground-filter", action="store_true")

    p = sub.add_parser("image", help="screenshots only (no elevation data): shading, snow, vegetation marks")
    p.add_argument("images", nargs="+", help="screenshots of the same place; different seasons/years help")
    p.add_argument("--point", type=_point, help="LAT,LON of the spot (also settles the sun direction)")
    p.add_argument("--spot-px", help="X,Y pixel of the spot in the first image (default: centre)")
    p.add_argument("--mpp", type=float, help="metres per pixel")
    p.add_argument("--altitude", type=float,
                   help="Google Earth eye altitude in metres (straight-down view): sets metres per pixel")
    p.add_argument("--view-width", type=int, default=1920,
                   help="width of the map window in screen pixels when the screenshot was taken (default 1920)")
    p.add_argument("--known", help="X1,Y1,X2,Y2,METRES: two pixels and the real distance between them")
    p.add_argument("--mask", action="append", default=[], help="X0,Y0,X1,Y1 rectangle to ignore (program UI)")
    p.add_argument("--sun-azimuth", type=float, help="sun azimuth if known (deg from north)")
    p.add_argument("--sun-elevation", type=float, default=30.0)
    p.add_argument("--north", type=float, default=0.0, help="if the image is rotated: azimuth of image 'up'")
    p.add_argument("--water-m", type=float, help="distance to water, m, if you know it")
    p.add_argument("--context")
    p.add_argument("--weights")
    p.add_argument("-o", "--out", default=None)

    p = sub.add_parser("batch", help="screenshot mode for many areas: one sub-folder per area (images + coords .txt)")
    p.add_argument("folder")
    p.add_argument("-o", "--out", default="batch_results")
    p.add_argument("--mpp", type=float, help="metres per pixel, if the same for all screenshots")

    p = sub.add_parser("ui", help="open the local interface in the browser (offline)")
    p.add_argument("--workspace", default="sitecheck_workspace")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")

    p = sub.add_parser("evaluate", help="test screenshot detection on areas you marked in red (known places)")
    p.add_argument("folder", help="<category>/<area>/ folders: clean picture, a copy with red marks, other seasons")
    p.add_argument("-o", "--out", help="write marks drawings + evaluation.json here")
    p.add_argument("--mpp", type=float, default=0.3)
    p.add_argument("--no-model", action="store_true", help="hand-made detectors only")

    p = sub.add_parser("train", help="train the learned spot detector on areas you marked (needs scikit-learn)")
    p.add_argument("folder", nargs="+", help="one or more marked packs")
    p.add_argument("-o", "--out", help="model file (default: replace the shipped sitecheck/model_pits.json)")

    p = sub.add_parser("ask", help="fill in the questionnaire")
    p.add_argument("-o", "--out", default="context.json")

    p = sub.add_parser("review", help="confirm/reject/add features and re-score")
    p.add_argument("out_dir")
    p.add_argument("--confirm", type=_ids, default=[])
    p.add_argument("--reject", type=_ids, default=[])
    p.add_argument("--add", action="append", default=[],
                   help="pit:X,Y | rectangle:X,Y:SIZE | linear@LAT,LON  (X,Y = m east,north of spot)")
    p.add_argument("--weights")

    a = ap.parse_args(argv)

    if a.cmd == "ask":
        from .context import interactive
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(interactive(), fh, ensure_ascii=False, indent=2)
        print(f"saved {a.out}")
        return 0

    if a.cmd == "grid":
        from .pointcloud import points_to_dem
        info = points_to_dem(a.points, a.out, a.cell, a.epsg, a.anchor, a.center,
                             ground_filter=not a.no_ground_filter)
        print(json.dumps(info, indent=1, default=str))
        print(f"\nNow: sitecheck analyze --dem {a.out} --point LAT,LON --dem-label \"{info['label']}\"")
        return 0

    if a.cmd == "evaluate":
        from .evaluate import run_folder
        run_folder(a.folder, a.out, a.mpp, use_model=not a.no_model)
        return 0

    if a.cmd == "train":
        from .train import main as train_main
        train_main(a.folder, a.out)
        return 0

    if a.cmd == "ui":
        from .ui.server import serve
        serve(a.workspace, a.port, not a.no_browser)
        return 0

    if a.cmd == "batch":
        from .batch import run_batch
        run_batch(a.folder, a.out, a.mpp)
        print(f"\nsummary: {os.path.join(a.out, 'summary.md')}, overview: {os.path.join(a.out, 'overview.jpg')}")
        return 0

    if a.cmd == "image":
        from .image_mode import run_images
        lat, lon = a.point if a.point else (None, None)
        out = a.out or (f"site_{lat:.5f}_{lon:.5f}_img" if lat is not None else "site_images")
        masks = [tuple(map(float, m.split(","))) for m in a.mask]
        spot = tuple(map(float, a.spot_px.split(","))) if a.spot_px else None
        if a.altitude and not a.mpp:
            from .scale import mpp_from_altitude
            a.mpp = round(mpp_from_altitude(a.altitude, a.view_width), 4)
            print(f"eye altitude {a.altitude:.0f} m, window {a.view_width} px wide -> {a.mpp} m per pixel")
        rep = run_images(a.images, out, lat, lon, a.mpp, a.known, spot, masks, a.sun_azimuth,
                         a.sun_elevation, a.north, a.context, a.weights, a.water_m)
        _print_summary(rep, out)
        return 0

    if a.cmd == "review":
        from .review import review
        rep = review(a.out_dir, a.confirm, a.reject, a.add, a.weights)
        _print_summary(rep, a.out_dir)
        return 0

    lat, lon = a.point
    out = a.out or f"site_{lat:.5f}_{lon:.5f}"

    if a.cmd == "fetch":
        from .fetch import fetch_all
        print(json.dumps(fetch_all(lat, lon, a.radius, out, a.source, nls_key=a.nls_key), indent=1, default=str))
        return 0

    from .pipeline import analyze
    water, label, dem, is_dsm, info = None, None, getattr(a, "dem", None), None, None
    if a.cmd == "run":
        from .fetch import fetch_all
        data_dir = os.path.join(out, "data")
        info = fetch_all(lat, lon, a.radius, data_dir, a.source, nls_key=a.nls_key, skip_dem=bool(dem))
        if not dem:
            if not info["dem"]:
                print("Could not download elevation data. Pass a DEM file with --dem "
                      "(own lidar, drone DTM, or `sitecheck grid` output).")
                return 2
            dem, label = info["dem"]["file"], info["dem"]["label"]
            is_dsm = info["dem"].get("is_dsm", False)
        wp = os.path.join(data_dir, "water.geojson")
        water = wp if os.path.exists(wp) else None
    else:
        water, label, is_dsm = a.water, a.dem_label, (True if a.dsm else None)

    rep = analyze(dem, lat, lon, a.radius, water, a.context, a.image, a.weights, out, label,
                  ground_filter=a.ground_filter, is_dsm=is_dsm, overlay_path=a.overlay, fetch_info=info)
    _print_summary(rep, out)
    return 0


def run():
    try:
        return main()
    except (ValueError, RuntimeError, FileNotFoundError) as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run())
