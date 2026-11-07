# sitecheck

You already have a spot in mind. `sitecheck` checks whether the ground there shows traces
of people: house pits, foundations, old roads, ditches. It combines that with what you know
(finds, legends, old maps) into one JSON report and a ready prompt for any AI.

- **No built-in AI, no API costs.** All analysis is local Python.
- **Offline analysis.** Internet is used only once, optionally, to download the elevation and
  water data for your spot. After that everything runs without a connection.
- **Every number has a source:** `user`, `computed:DEM` or `computed:water.geojson`.
- **You stay in control.** Automatic detections are *candidates* until you confirm or reject them.

## Install

```bash
pip install -e .            # core
pip install -e ".[geo]"     # + rasterio/pyproj: needed for projected lidar (UK, Finland, most national data)
```

## Use

```bash
# 1. once: what you know about the place (saved, reusable)
sitecheck ask -o context.json

# 2a. one command: downloads data for the spot, then analyzes
sitecheck run --point 44.9271,-93.2650 --context context.json

# 2b. or fully offline with a lidar tile you downloaded yourself
sitecheck analyze --point 60.4981,24.9036 --dem my_lidar_tile.tif --dem-label "NLS lidar 2 m" \
                  --water water.geojson --context context.json --image my_screenshot.png

# 3. look at site_.../candidates.png, then confirm / reject / add what you see
sitecheck review site_44.92710_-93.26500 --confirm 1,2,4 --reject 6 --add pit:35,-20:6
```

`--add` formats: `pit:X,Y[:diameter]`, `rectangle:X,Y[:size]`, `linear:X,Y[:length]` where X,Y
are meters east/north of your spot (read them off the picture axes), or `pit@LAT,LON`.

Output folder:

| file | what |
|---|---|
| `candidates.png` | local relief over hillshade, numbered candidates, your spot as `+` |
| `terrain_panels.png` | hillshade, local relief, slope, sky-view factor |
| `report.json` | everything: context, terrain metrics, features, score with breakdown |
| `prompt.md` | paste into any AI chat, attach the two PNGs |
| `features_full.json`, `layers.npz` | internal, used by `review` |

## Elevation data: what works where

Pits and foundations are 3-15 m across and 0.2-1 m deep. **You need lidar at 1-2 m.**
30 m data (Copernicus GLO-30, SRTM) only gives landscape context, and the tool says so.

| region | best free source | how |
|---|---|---|
| USA | USGS 3DEP 1 m (bare earth) | automatic in `sitecheck run` |
| UK (England) | Environment Agency National LIDAR Programme DTM 1 m | download the tile from the DEFRA Data Services Platform, use `--dem` |
| Finland | Maanmittauslaitos (NLS) 2 m DEM / laser data | NLS open data file service, use `--dem` |
| Russia, elsewhere | no open lidar; ArcticDEM 2 m north of 60 N, else GLO-30 | automatic GLO-30, or `--dem` |

Any single-band GeoTIFF in any projection works (with the `geo` extra).

## How detection works (no AI)

1. **Local Relief Model**: the DEM minus a local quadratic trend surface (35 m window).
   Hills and valleys disappear; what is left is micro-relief.
2. **Pits**: closed depressions (morphological fill), filtered by depth, diameter and roundness.
3. **Rectangles**: raised or sunken shapes whose hull is rectangular, either solid (platforms,
   floors) or outlines with wall support on all four sides and real corners.
4. **Linear**: long narrow ridges or hollows. Straightness separates roads and ditches
   from natural gullies. Banks along a ditch are merged into it, and lines that follow
   the contour are flagged as possibly natural terrace edges.
5. **Score**: points from `weights.json`, each explained in the report. Features far from
   the spot count less; unreviewed features count half. **It is a heuristic, not a probability.**
   Edit the weights, and they get a version string in every report.

## Tests

```bash
python tests/make_synthetic.py /tmp/syn.tif     # 1 m terrain with known pits, foundation, road, gully
sitecheck analyze --point 60.498006,24.903657 --dem /tmp/syn.tif -o /tmp/out
```

On 20 noise variations: 59/60 pits, 18/20 foundations and 20/20 roads found, with about 1.4 false
rectangles per run. Real lidar will be harder (trees, ploughing, root throws), so treat
detections as candidates.

## Roadmap

- local map GUI (pick the spot, click to confirm/reject), still offline after download
- automatic UK / Finland lidar download
- historical map overlays (georeferenced old maps as layers)
- point-cloud (LAZ) input with own ground classification
- calibration of weights on known sites
