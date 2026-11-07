"""Command line.

  sitecheck run     --point LAT,LON                 download (once) + analyze, one command
  sitecheck analyze --point LAT,LON --dem FILE      fully offline, your own DEM / lidar
  sitecheck ask     -o context.json                 the questionnaire
  sitecheck review  OUT_DIR --confirm 1,3 --reject 7 --add pit:35,-20
  sitecheck fetch   --point LAT,LON -o data/        download only
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
    print(f"\nSpot {rep['spot']['lat']}, {rep['spot']['lon']}  |  DEM cell {rep['data']['dem']['cell_size_m']} m")
    print(f"Terrain: {t['landform']}, slope {t['slope_deg']} deg, faces {t['aspect']}, "
          f"elevation {t['elevation_m']} m")
    w = rep["water"]
    print(f"Water: {w['band']}" + (f" ({w.get('kind')}, {w.get('name') or 'unnamed'})" if w.get("distance_m") is not None else ""))
    print(f"Candidates: {fs['pits']} pits, {fs['rectangles']} rectangles, {fs['straight_linear']} straight "
          f"+ {fs['other_linear']} other linear; {fs['within_100m_of_spot']} within 100 m")
    for f in rep["features"]:
        size = f.get("diameter_m") or f.get("length_m")
        print(f"  #{f['id']:<3} {f['type']:<9} {f.get('relief', ''):<7} {size} m  "
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
          f"and attach candidates.png and terrain_panels.png.")


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

    p = sub.add_parser("run", help="download data once, then analyze")
    common(p)
    p.add_argument("--source", choices=["auto", "3dep", "glo30"], default="auto")
    p.add_argument("--dem", help="use this DEM instead of downloading one (water is still downloaded)")

    p = sub.add_parser("analyze", help="offline analysis of a DEM you already have")
    common(p)
    p.add_argument("--dem", required=True)
    p.add_argument("--water", help="GeoJSON with rivers/lakes")
    p.add_argument("--dem-label", help="describe the DEM, e.g. 'NLS lidar 2 m'")

    p = sub.add_parser("fetch", help="download only")
    p.add_argument("--point", type=_point, required=True)
    p.add_argument("--radius", type=float, default=300)
    p.add_argument("--source", choices=["auto", "3dep", "glo30"], default="auto")
    p.add_argument("-o", "--out", default="sitecheck_data")

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

    if a.cmd == "review":
        from .review import review
        rep = review(a.out_dir, a.confirm, a.reject, a.add, a.weights)
        _print_summary(rep, a.out_dir)
        return 0

    lat, lon = a.point
    out = a.out or f"site_{lat:.5f}_{lon:.5f}"

    if a.cmd == "fetch":
        from .fetch import fetch_all
        print(json.dumps(fetch_all(lat, lon, a.radius, out, a.source), indent=1))
        return 0

    from .pipeline import analyze
    water, label, dem = None, None, getattr(a, "dem", None)
    if a.cmd == "run":
        from .fetch import fetch_all, osm_water
        data_dir = os.path.join(out, "data")
        if dem:
            os.makedirs(data_dir, exist_ok=True)
            try:
                osm_water(lat, lon, 5000, os.path.join(data_dir, "water.geojson"))
            except Exception as ex:  # noqa: BLE001
                print(f"water download failed ({ex}); continuing without it")
        else:
            info = fetch_all(lat, lon, a.radius, data_dir, a.source)
            if not info["dem"]:
                print("Could not download elevation data. Pass a DEM file with --dem.")
                return 2
            dem, label = info["dem"]["file"], info["dem"]["label"]
        wp = os.path.join(data_dir, "water.geojson")
        water = wp if os.path.exists(wp) else None
    else:
        water, label = a.water, a.dem_label

    rep = analyze(dem, lat, lon, a.radius, water, a.context, a.image, a.weights, out, label)
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
